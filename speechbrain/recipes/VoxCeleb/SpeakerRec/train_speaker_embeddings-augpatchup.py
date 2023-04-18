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
import numpy as np
import torch
import torchaudio
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main


class SpeakerBrain(sb.core1.Brain):
    """Class for speaker embedding training"
    """

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        spkid, _ = batch.spk_id_encoded
        if stage == sb.Stage.TRAIN:

            # Applying the augmentation pipeline
            wavs_aug_tot = []
            wavs_aug_tot.append(wavs)
            # print(wavs.shape)
            # for count, augment in enumerate(self.hparams.augment_pipeline):
            #     # Apply augment
            #     wavs_aug = augment(wavs, lens)#torch.Size([16, 32000]
            #     # print(wavs_aug.shape)
            #     # Managing speed change
            #     if wavs_aug.shape[1] > wavs.shape[1]:
            #         wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
            #         # print(wavs_aug.shape,"1")
            #     else:
            #         zero_sig = torch.zeros_like(wavs)
            #         zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
            #         wavs_aug = zero_sig
            #         # print(wavs_aug.shape,"2")
            #
            #     if self.hparams.concat_augment:
            #         wavs_aug_tot.append(wavs_aug)
            #         # print(wavs_aug.shape,"3")
            #     else:
            #         wavs = wavs_aug
            #         wavs_aug_tot[0] = wavs

            wavs = torch.cat(wavs_aug_tot, dim=0)
            self.n_augment = len(wavs_aug_tot)
            lens = torch.cat([lens] * self.n_augment)
            spkid = torch.cat([spkid] * self.n_augment, dim=0)
        # wavs, y_a, y_b, lam = self.mixup_data(wavs, spkid, 0.5)
        # print(wavs.shape)#(16*32000)
        # r = np.random.rand(1)

        #     wavs, y_a, y_b, lam = self.cutmix(wavs, spkid, 0.5)
        # else:
        #     wavs = wavs
        #     y_a = y_b= spkid
        #     lam=1
        # Feature extraction and normalization
        if stage== sb.Stage.TRAIN:
            r = np.random.rand(1)
            if r < 1.0 :#部分使用patchup
                layer_mix = random.randint(0, 2)#选择在那一层使用
                if layer_mix!=0:#除第一层外其他的层都是patchup
                    feats = self.modules.compute_features(wavs)#得到FBANK
                    feats = self.modules.mean_var_norm(feats, lens)
                    embeddings, target_a, target_b, target_reweighted, portion = self.modules.embedding_model(
                        feats, spkid, layer_mix=layer_mix)
                else:#cutmix
                    wavs,target_a, target_b, portion = self.cutmix(wavs, spkid, 1.)
                    target_reweighted = portion * target_a + (1.0 - portion) * target_b
                    target_b = None
                    feats = self.modules.compute_features(wavs)
                    feats = self.modules.mean_var_norm(feats, lens)
                    embeddings = self.modules.embedding_model(feats,spkid,layer_mix=None)
            else:
                # lam_value = torch.from_numpy(np.array([lam_value]).astype('float32')).to(device)
                out, target_reweighted = self.mixup_process(wavs, spkid, lam=1.)
                feats = self.modules.compute_features(out)
                feats = self.modules.mean_var_norm(feats, lens)
                embeddings = self.modules.embedding_model(feats,spkid,layer_mix=None)
                target_a, target_b,portion = None,None,None
        else:
            feats = self.modules.compute_features(wavs)
            feats = self.modules.mean_var_norm(feats, lens)
            embeddings= self.modules.embedding_model(feats,spkid,layer_mix=None)
            target_reweighted = spkid
            target_a, target_b, portion = None, None, None
        # Embeddings + speaker classifier
        outputs = self.modules.classifier(embeddings)

        return outputs, lens,target_a,target_b,target_reweighted,portion

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        # predictions, lens,y_a,y_b,lam = predictions
        outputs, lens, target_a, target_b, target_reweighted, portion =predictions
        uttid = batch.id
        # spkid, _ = batch.spk_id_encoded

        # # Concatenate labels (due to data augmentation)
        # if stage == sb.Stage.TRAIN:
        #     spkid = torch.cat([spkid] * self.n_augment, dim=0)
        if target_b != None:#patchup
            loss = self.hparams.compute_cost(outputs, target_a, lens) * \
                   portion + self.hparams.compute_cost(outputs, target_b, lens) * (1. - portion) + \
                   self.hparams.compute_cost(outputs, target_reweighted, lens)
        else:#cutmix&mixup
            loss = self.hparams.compute_cost(outputs, target_reweighted, lens)
        # loss = lam *self.hparams.compute_cost(predictions, y_a, lens)+ (1 - lam) * self.hparams.compute_cost(predictions, y_b, lens)

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, outputs, target_reweighted, lens)
            self.error_metrics.append(uttid, outputs, target_reweighted, lens)

        return loss
    def cutmix(self,input,target,beta,use_cuda=True):
        # r = np.random.rand(1)
        # if beta > 0 and r < cutmix_prob:
        # generate mixed sample
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(input.size()[0]).cuda()
        target_a = target
        target_b = target[rand_index]
        # bbx1, bby1, bbx2, bby2 = self.rand_bbox(input.size(), lam)
        bbx1,bbx2= self.rand_bbox(input.size(), lam)
        input[:, bbx1:bbx2] = input[rand_index,bbx1:bbx2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1)  / input.size()[-1] )
        return input,target_a,target_b,lam
    def rand_bbox(self,size, lam):
        W = size[1]
        # H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        # cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        # cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        # bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        # bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bbx2
    def mixup_data(self,x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_process(self,out, target_reweighted, lam):
        """
        It can be use in Mixup and ManifoldMixup.
        https://arxiv.org/abs/1710.09412
        it applies mixup process for a mini-batch.
        Args:
            out: it is a samples in mini-batch or hidden representations
            for a mini-batch for Mixup or ManifoldMixup, respectively.
            target_reweighted: re-weighted target or interpolated targets in the mini-batch.
            iIn mixup it is the one hot embedding vector. And, in ManifoldMixup it is the re-weighted target
            calculated from previous layers.
            lam: the mixing interpolation policy coefficient.
        Returns:
            out: the interpolated of randomly selected pairs.
            target_reweighted: re-weighted target or interpolated targets in the mini-batch.
        """
        indices = np.random.permutation(out.size(0))
        out = out * lam + out[indices] * (1 - lam)
        target_shuffled_onehot = target_reweighted[indices]
        target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
        return out, target_reweighted

    def get_lambda(self,alpha=1.0):
        """
        computes the interpolation policy coefficient in the mixup.
        Args:
            alpha: controls the shape of the Beta distribution.
        Returns:
            lam: a float number in [0, 1] that is the interpolation policy coefficient.
        """
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.
        return lam


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
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
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
    sb.core1.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    #Brain class initialization
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
