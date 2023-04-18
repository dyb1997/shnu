#!/usr/bin/python3
"""Recipe for training speaker embeddings (e.g, xvectors) using the VoxCeleb Dataset.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train_speaker_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/train_x_vectors.yaml (for standard xvectors)
    hyperparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
"""
import os
import sys
import random
import torch
import torchaudio
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main


class SpeakerBrain(sb.coreBYOLTS.Brain):
    """Class for speaker embedding training"
    """

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        if stage == sb.Stage.TRAIN:
            wavs_aug = []
            # wavs_aug2 = []
            # Applying the augmentation pipeline
            wavs_aug_tot = []
            # wavs_aug_tot.append(wavs)
            for count, augment in enumerate(self.hparams.augment_pipeline):
                # # Apply augment
                # if count ==1 :
                wavs_aug = augment(wavs, lens)
                # else :
                #     wavs_aug2 = augment(wavs,lens)
                # # Managing speed change
                # if wavs_aug.shape[1] > wavs.shape[1]:
                #     wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                # else:
                #     zero_sig = torch.zeros_like(wavs)
                #     zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                #     wavs_aug = zero_sig
                #
                if self.hparams.concat_augment:
                    wavs_aug_tot.append(wavs_aug)
                else:
                    wavs = wavs_aug
                    wavs_aug_tot[0] = wavs
            # wavs1 = torch.cat((wavs_aug_tot[0], wavs), dim=0)
            # lensC = torch.cat([lens] * 2)
            print(stage == sb.Stage.TRAIN)
            wavs_T = wavs
            wavs_S = wavs_aug
            #wavs = torch.cat(wavs_aug_tot, dim=0)
            len_S = lens
            len_T = lens
            # lens = torch.cat([lens] * self.n_augment)
            # self.n_augment_T = len(wavs_aug_tot)
            # lens = torch.cat([lens] * self.n_augment)
        else:
            wavs_T = wavs
            wavs_S = wavs
            len_T = lens
            len_S = lens
        #print(wavs_T,"wav-T")
        #print(wavs_S,"wav-S")
        # Feature extraction and normalization
        print(stage == sb.Stage.TRAIN)
        if stage == sb.Stage.TRAIN:
            feats_T_sup = self.modules.compute_features(wavs)
            feats_T_sup = self.modules.mean_var_norm(feats_T_sup,len_T)
            feats_S_sup = self.modules2.compute_features(wavs_S)
            feats_S_sup = self.modules2.mean_var_norm(feats_S_sup, len_S)
            feats_T = self.modules.compute_features(wavs_T)
            feats_T = self.modules.mean_var_norm(feats_T, len_T)
            feats_S = self.modules2.compute_features(wavs_T)
            feats_S = self.modules2.mean_var_norm(feats_S, len_S)
            embeddings_T_sup = self.modules.embedding_model(feats_T_sup)
            # outputs_T_sup = self.modules.classifier(embeddings_T_sup)
            embeddings_S_sup,pre= self.modules2.embedding_model(feats_S_sup)
            # outputs_S_sup = self.modules2.classifier(pre)
            embeddings_T_sup2 = self.modules.embedding_model(feats_S_sup)
            # outputs_T_sup2 = self.modules.classifier(embeddings_T_sup2)
            embeddings_S_sup2, pre2 = self.modules2.embedding_model(feats_T_sup)
            # outputs_S_sup2 = self.modules2.classifier(pre)
            # embeddings_C, y_3 = self.modules2.embedding_model(feats_C)
            # outputs_C = self.modules2.classifier(embeddings_C)
            embeddings_T= self.modules.embedding_model(feats_T)
            outputs_T = self.modules.classifier(embeddings_T)
            embeddings_S,pre3 = self.modules2.embedding_model(feats_S)
            outputs_S = self.modules2.classifier(embeddings_S)
            return outputs_T,outputs_S,len_T,len_S,embeddings_T_sup,embeddings_S_sup,embeddings_T,pre,embeddings_T_sup2,pre2
            # outputs_S = self.modules2.classifier(embeddings_S)
        else:
            feats_T = self.modules.compute_features(wavs_T)
            feats_T = self.modules.mean_var_norm(feats_T, len_T)
            feats_S = self.modules2.compute_features(wavs_S)
            feats_S = self.modules2.mean_var_norm(feats_S, len_S)
        # Embeddings + speaker classifier
            embeddings_T = self.modules.embedding_model(feats_T)
            outputs_T = self.modules.classifier(embeddings_T)
            embeddings_S ,pre= self.modules2.embedding_model(feats_S)
            outputs_S = self.modules2.classifier(embeddings_S)
            return outputs_T,outputs_S, len_T,len_S,embeddings_T,embeddings_S

    def compute_objectives(self, predictions,lens, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        # predictions, lens = predictions
        uttid = batch.id
        spkid, _ = batch.spk_id_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN:
            spkid = spkid
            #spkid = torch.cat([spkid] * self.n_augment, dim=0)

        loss = self.hparams.compute_cost(predictions, spkid, lens)

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss
    def compute_objectives2(self, predictions,lens, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        # predictions, lens = predictions
        uttid = batch.id
        spkid, _ = batch.spk_id_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN:
            spkid = spkid
            #spkid = torch.cat([spkid] * self.n_augment, dim=0)

        loss = self.hparams.compute_cost(predictions, spkid, lens)

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing2, "on_batch_end"
        ):
            self.hparams.lr_annealing2.on_batch_end(self.optimizer2)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss

    def MSE(self,output_t,output_s):
        loss_mse = torch.nn.MSELoss(reduction='mean')
        loss = loss_mse(output_t, output_s)
        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            # old_lr2, new_lr2 = self.hparams.lr_annealing2(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            sb.nnet.schedulers.update_learning_rate(self.optimizer2, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.hparams.train_logger2.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )
            self.checkpointer2.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["data_folder"]

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        if hparams["random_chunk"]:
            duration_sample = int(duration * hparams["sample_rate"])
            start = random.randint(0, duration_sample - snt_len_sample - 1)
            stop = start + snt_len_sample
        else:
            start = int(start)
            stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file, from_didatasets=[train_data], output_key="spk_id",
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])

    return train_data, valid_data, label_encoder


if __name__ == "__main__":

    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Download verification list (to exlude verification sentences from train)
    veri_file_path = os.path.join(
        hparams["save_folder"], os.path.basename(hparams["verification_file"])
    )
    download_file(hparams["verification_file"], veri_file_path)

    # Dataset prep (parsing VoxCeleb and annotation into csv files)
    from voxceleb_prepare import prepare_voxceleb  # noqa

    run_on_main(
        prepare_voxceleb,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "verification_pairs_file": veri_file_path,
            "splits": ["train", "dev"],
            "split_ratio": [90, 10],
            "seg_dur": hparams["sentence_len"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, label_encoder = dataio_prep(hparams)


    # Create experiment directory
    sb.coreBYOLTS.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        modules2=hparams["modules2"],
        opt_class=hparams["opt_class"],
        opt_class2=hparams["opt_class2"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        checkpointer2=hparams["checkpointer2"],
        # max_epoch=hparams["max_epoch"],
        # max_consistency_cost=hparams["max_consistency_cost"],
    )
    # speaker_brain2 = SpeakerBrain(
    #
    #     opt_class2=hparams["opt_class2"],
    #     hparams2=hparams,
    #     run_opts2=run_opts,
    #
    # )
    #
    # # Training
    #teacher
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
        )
    #student
    # speaker_brain2.fit(
    #     speaker_brain.hparams.epoch_counter,
    #     train_data,
    #     valid_data,
    #     train_loader_kwargs=hparams["dataloader_options"],
    #     valid_loader_kwargs=hparams["dataloader_options"],
    # )