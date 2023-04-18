#!/usr/bin/python3
"""Recipe for training a speaker verification system based on cosine distance.
The cosine distance is computed on the top of pre-trained embeddings.
The pre-trained model is automatically downloaded from the web if not specified.
This recipe is designed to work on a single GPU.

To run this recipe, run the following command:
    >  python speaker_verification_cosine.py hyperparams/verification_ecapa_tdnn.yaml

Authors
    * Hwidong Na 2020
    * Mirco Ravanelli 2020
"""
import os
import sys
import torch
import numpy as np
import logging
import torchaudio
import speechbrain as sb
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import EER, minDCF
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main


# Compute embeddings from the waveforms
def compute_embedding(wavs, wav_lens):
    """Compute speaker embeddings.

    Arguments
    ---------
    wavs : Torch.Tensor
        Tensor containing the speech waveform (batch, time).
        Make sure the sample rate is fs=16000 Hz.
    wav_lens: Torch.Tensor
        Tensor containing the relative length for each sentence
        in the length (e.g., [0.8 0.6 1.0])
    """
    with torch.no_grad():
        feats = params["compute_features"](wavs)
        feats = params["mean_var_norm"](feats, wav_lens)
        embeddings = params["embedding_model"](feats, wav_lens)
        embeddings = params["mean_var_norm_emb"](
            embeddings, torch.ones(embeddings.shape[0]).to(embeddings.device)
        )
    return embeddings.squeeze(1)
def compute_embedding2(wavs, wav_lens):
    """Compute speaker embeddings.

    Arguments
    ---------
    wavs : Torch.Tensor
        Tensor containing the speech waveform (batch, time).
        Make sure the sample rate is fs=16000 Hz.
    wav_lens: Torch.Tensor
        Tensor containing the relative length for each sentence
        in the length (e.g., [0.8 0.6 1.0])
    """
    with torch.no_grad():
        feats2 = params["compute_features2"](wavs)
        feats2 = params["mean_var_norm2"](feats2, wav_lens)
        embeddings2 = params["embedding_model2"](feats2, wav_lens)
        embeddings2 = params["mean_var_norm_emb2"](
            embeddings2, torch.ones(embeddings2.shape[0]).to(embeddings2.device)
        )
    return embeddings2.squeeze(1)

def compute_embedding_loop(data_loader):
    """Computes the embeddings of all the waveforms specified in the
    dataloader.
    """
    embedding_dict = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, dynamic_ncols=True):
            batch = batch.to(params["device"])
            seg_ids = batch.id
            wavs, lens = batch.sig

            found = False
            for seg_id in seg_ids:
                if seg_id not in embedding_dict:
                    found = True
            if not found:
                continue
            # wavs = wavs.cpu()
            # wavs = wavs + np.abs(np.random.normal(0,0.5 ** 2,wavs.shape))
            # wavs = wavs.to(torch.float32)
            wavs, lens = wavs.to(params["device"]), lens.to(params["device"])
            emb = compute_embedding(wavs, lens).unsqueeze(1)
            for i, seg_id in enumerate(seg_ids):
                embedding_dict[seg_id] = emb[i].detach().clone()
    return embedding_dict

def compute_embedding_loop2(data_loader):
    """Computes the embeddings of all the waveforms specified in the
    dataloader.
    """
    embedding_dict2 = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, dynamic_ncols=True):
            batch = batch.to(params["device"])
            seg_ids2 = batch.id
            wavs2, lens2 = batch.sig

            found = False
            for seg_id2 in seg_ids2:
                if seg_id2 not in embedding_dict2:
                    found = True
            if not found:
                continue
            # wavs = wavs.cpu()
            # wavs = wavs + np.abs(np.random.normal(0,0.5 ** 2,wavs.shape))
            # wavs = wavs.to(torch.float32)
            wavs2, lens2 = wavs2.to(params["device"]), lens2.to(params["device"])
            emb2 = compute_embedding2(wavs2, lens2).unsqueeze(1)
            for i, seg_id in enumerate(seg_ids2):
                embedding_dict2[seg_id] = emb2[i].detach().clone()
    return embedding_dict2
def get_verification_scores(veri_test):
    """ Computes positive and negative scores given the verification split.
    """
    scores = []
    scores2 = []
    scores3 = []
    scores4 = []
    scores5 = []
    scores6 = []
    scores7 = []
    scores8 = []
    scores9 = []
    positive_scores = []
    negative_scores = []
    positive_scores2 = []
    negative_scores2 = []
    positive_scores3 = []
    negative_scores3 = []
    positive_scores4 = []
    negative_scores4 = []
    positive_scores5 = []
    negative_scores5 = []
    positive_scores6 = []
    negative_scores6 = []
    positive_scores7 = []
    negative_scores7 = []
    positive_scores8 = []
    negative_scores8 = []
    positive_scores9 = []
    negative_scores9 = []

    save_file = os.path.join(params["output_folder"], "scores.txt")
    s_file = open(save_file, "w")
    save_file2 = os.path.join(params["output_folder"], "scores2.txt")
    s_file2 = open(save_file2, "w")
    save_file3 = os.path.join(params["output_folder"], "scores3.txt")
    s_file3 = open(save_file3, "w")
    save_file4 = os.path.join(params["output_folder"], "scores4.txt")
    s_file4 = open(save_file4, "w")
    save_file5 = os.path.join(params["output_folder"], "scores5.txt")
    s_file5 = open(save_file5, "w")

    # Cosine similarity initialization
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    # creating cohort for score normalization
    # if "score_norm" in params:
    #     train_cohort = torch.stack(list(train_dict.values()))

    for i, line in enumerate(veri_test):

        # Reading verification file (enrol_file test_file label)
        lab_pair = int(line.split(" ")[0].rstrip().split(".wav")[0].strip())
        enrol_id = line.split(" ")[1].rstrip().split(".wav")[0].strip()
        test_id = line.split(" ")[2].rstrip().split(".wav")[0].strip()
        enrol = enrol_dict[enrol_id]
        test = test_dict[test_id]
        enrol2 = enrol_dict2[enrol_id]
        test2 = test_dict2[test_id]
        # if "score_norm" in params:
        #     # Getting norm stats for enrol impostors
        #     enrol_rep = enrol.repeat(train_cohort.shape[0], 1, 1)
        #     score_e_c = similarity(enrol_rep, train_cohort)
        #
        #     if "cohort_size" in params:
        #         score_e_c = torch.topk(
        #             score_e_c, k=params["cohort_size"], dim=0
        #         )[0]
        #
        #     mean_e_c = torch.mean(score_e_c, dim=0)
        #     std_e_c = torch.std(score_e_c, dim=0)
        #
        #     # Getting norm stats for test impostors
        #     test_rep = test.repeat(train_cohort.shape[0], 1, 1)
        #     score_t_c = similarity(test_rep, train_cohort)
        #
        #     if "cohort_size" in params:
        #         score_t_c = torch.topk(
        #             score_t_c, k=params["cohort_size"], dim=0
        #         )[0]
        #
        #     mean_t_c = torch.mean(score_t_c, dim=0)
        #     std_t_c = torch.std(score_t_c, dim=0)
        #
        # # Compute the score for the given sentence
        score = similarity(enrol, test)[0]
        score2 = similarity(enrol2,test2)[0]
        score3 = similarity(enrol,test2)[0]
        score4 = similarity(enrol2 ,test)[0]
        score5 = (score2+score4)/2
        score6 = (score+score4)/2
        score7 = (score2 + score3) / 2
        score8 = (score+score2) / 2
        score9 = (score + score2  +score3+score4) / 4
        # Perform score normalization
        # if "score_norm" in params:
        #     if params["score_norm"] == "z-norm":
        #         score = (score - mean_e_c) / std_e_c
        #     elif params["score_norm"] == "t-norm":
        #         score = (score - mean_t_c) / std_t_c
        #     elif params["score_norm"] == "s-norm":
        #         score_e = (score - mean_e_c) / std_e_c
        #         score_t = (score - mean_t_c) / std_t_c
        #         score = 0.5 * (score_e + score_t)

        # write score file
        s_file.write("%s %s %i %f\n" % (enrol_id, test_id, lab_pair, score))
        s_file2.write("%s %s %i %f\n" % (enrol_id, test_id, lab_pair, score2))
        s_file3.write("%s %s %i %f\n" % (enrol_id, test_id, lab_pair, score3))
        s_file4.write("%s %s %i %f\n" % (enrol_id, test_id, lab_pair, score4))
        s_file5.write("%s %s %i %f\n" % (enrol_id, test_id, lab_pair, score5))
        scores.append(score)
        scores2.append(score2)
        scores3.append(score3)
        scores4.append(score4)
        scores5.append(score5)
        scores6.append(score6)
        scores7.append(score7)
        scores8.append(score8)
        scores9.append(score9)

        if lab_pair == 1:
            positive_scores.append(score)
            positive_scores2.append(score2)
            positive_scores3.append(score3)
            positive_scores4.append(score4)
            positive_scores5.append(score5)
            positive_scores6.append(score6)
            positive_scores7.append(score7)
            positive_scores8.append(score8)
            positive_scores9.append(score9)
        else:
            negative_scores.append(score)
            negative_scores2.append(score2)
            negative_scores3.append(score3)
            negative_scores4.append(score4)
            negative_scores5.append(score5)
            negative_scores6.append(score6)
            negative_scores7.append(score7)
            negative_scores8.append(score8)
            negative_scores9.append(score9)

    s_file.close()
    s_file2.close()
    s_file3.close()
    s_file4.close()
    s_file5.close()
    return positive_scores, negative_scores,positive_scores2, negative_scores2,positive_scores3, negative_scores3,positive_scores4, negative_scores4,positive_scores5, negative_scores5,positive_scores6, negative_scores6,positive_scores7, negative_scores7,positive_scores8, negative_scores8,positive_scores9, negative_scores9
def get_verification_scores2(veri_test,enrol_dict, test_dict, enrol_dict2, test_dict2):
    """ Computes positive and negative scores given the verification split.
    """
    scores = []
    scores2 = []
    scores3 = []
    scores4 = []
    positive_scores = []
    negative_scores = []
    positive_scores2 = []
    negative_scores2 = []
    positive_scores3 = []
    negative_scores3 = []
    positive_scores4 = []
    negative_scores4 = []


    save_file = os.path.join(params["output_folder"], "scores.txt")
    s_file = open(save_file, "w")
    save_file2 = os.path.join(params["output_folder"], "scores2.txt")
    s_file2 = open(save_file2, "w")
    save_file3 = os.path.join(params["output_folder"], "scores3.txt")
    s_file3 = open(save_file3, "w")
    save_file4 = os.path.join(params["output_folder"], "scores4.txt")
    s_file4 = open(save_file4, "w")
    save_file5 = os.path.join(params["output_folder"], "scores5.txt")
    s_file5 = open(save_file5, "w")

    # Cosine similarity initialization
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    # creating cohort for score normalization
    # if "score_norm" in params:
    #     train_cohort = torch.stack(list(train_dict.values()))

    for i, line in enumerate(veri_test):

        # Reading verification file (enrol_file test_file label)
        lab_pair = int(line.split(" ")[0].rstrip().split(".wav")[0].strip())
        enrol_id = line.split(" ")[1].rstrip().split(".wav")[0].strip()
        test_id = line.split(" ")[2].rstrip().split(".wav")[0].strip()
        enrol = enrol_dict[enrol_id]
        test = test_dict[test_id]
        enrol2 = enrol_dict2[enrol_id]
        test2 = test_dict2[test_id]
        # if "score_norm" in params:
        #     # Getting norm stats for enrol impostors
        #     enrol_rep = enrol.repeat(train_cohort.shape[0], 1, 1)
        #     score_e_c = similarity(enrol_rep, train_cohort)
        #
        #     if "cohort_size" in params:
        #         score_e_c = torch.topk(
        #             score_e_c, k=params["cohort_size"], dim=0
        #         )[0]
        #
        #     mean_e_c = torch.mean(score_e_c, dim=0)
        #     std_e_c = torch.std(score_e_c, dim=0)
        #
        #     # Getting norm stats for test impostors
        #     test_rep = test.repeat(train_cohort.shape[0], 1, 1)
        #     score_t_c = similarity(test_rep, train_cohort)
        #
        #     if "cohort_size" in params:
        #         score_t_c = torch.topk(
        #             score_t_c, k=params["cohort_size"], dim=0
        #         )[0]
        #
        #     mean_t_c = torch.mean(score_t_c, dim=0)
        #     std_t_c = torch.std(score_t_c, dim=0)
        #
        # # Compute the score for the given sentence
        score = similarity(enrol, test)[0]
        score2 = similarity(enrol2,test2)[0]
        score3 = similarity(enrol,test2)[0]
        score4 = similarity(enrol2 ,test)[0]
        score5 = (score2+score4)/2
        score6 = (score+score4)/2
        score7 = (score2 + score3) / 2
        score8 = (score+score2) / 2
        score9 = (score + score2  +score3+score4) / 4
        # Perform score normalization
        # if "score_norm" in params:
        #     if params["score_norm"] == "z-norm":
        #         score = (score - mean_e_c) / std_e_c
        #     elif params["score_norm"] == "t-norm":
        #         score = (score - mean_t_c) / std_t_c
        #     elif params["score_norm"] == "s-norm":
        #         score_e = (score - mean_e_c) / std_e_c
        #         score_t = (score - mean_t_c) / std_t_c
        #         score = 0.5 * (score_e + score_t)

        # write score file
        s_file.write("%s %s %i %f\n" % (enrol_id, test_id, lab_pair, score))
        s_file2.write("%s %s %i %f\n" % (enrol_id, test_id, lab_pair, score2))
        s_file3.write("%s %s %i %f\n" % (enrol_id, test_id, lab_pair, score3))
        s_file4.write("%s %s %i %f\n" % (enrol_id, test_id, lab_pair, score4))
        s_file5.write("%s %s %i %f\n" % (enrol_id, test_id, lab_pair, score5))
        scores.append(score)
        scores2.append(score2)
        scores3.append(score3)
        scores4.append(score4)


        if lab_pair == 1:
            positive_scores.append(score)
            positive_scores2.append(score2)
            positive_scores3.append(score3)
            positive_scores4.append(score4)

        else:
            negative_scores.append(score)
            negative_scores2.append(score2)
            negative_scores3.append(score3)
            negative_scores4.append(score4)


    s_file.close()
    s_file2.close()
    s_file3.close()
    s_file4.close()
    s_file5.close()
    return positive_scores, negative_scores,positive_scores2, negative_scores2,positive_scores3, negative_scores3,positive_scores4, negative_scores4

def dataio_prep(params):
    "Creates the dataloaders and their data processing pipelines."

    data_folder = params["data_folder"]

    # 1. Declarations:

    # # Train data (used for normalization)
    # train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
    #     csv_path=params["train_data"], replacements={"data_root": data_folder},
    # )
    # train_data = train_data.filtered_sorted(
    #     sort_key="duration", select_n=params["n_train_snts"]
    # )

    # Enrol data
    enrol_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["enrol_data"], replacements={"data_root": data_folder},
    )
    enrol_data = enrol_data.filtered_sorted(sort_key="duration")

    # Test data
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["test_data"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [enrol_data, test_data]
    # Enrol data2
    enrol_data2 = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["enrol_data2"], replacements={"data_root": data_folder},
    )
    enrol_data2 = enrol_data2.filtered_sorted(sort_key="duration")

    # Test data2
    test_data2 = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["test_data2"], replacements={"data_root": data_folder},
    )
    test_data2 = test_data2.filtered_sorted(sort_key="duration")

    datasets2 = [enrol_data2, test_data2]
    # Enrol data3
    enrol_data3 = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["enrol_data3"], replacements={"data_root": data_folder},
    )
    enrol_data3 = enrol_data3.filtered_sorted(sort_key="duration")

    # Test data3
    test_data3 = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["test_data3"], replacements={"data_root": data_folder},
    )
    test_data3 = test_data3.filtered_sorted(sort_key="duration")

    datasets3 = [enrol_data3, test_data3]
    # Enrol data4
    enrol_data4 = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["enrol_data4"], replacements={"data_root": data_folder},
    )
    enrol_data4 = enrol_data4.filtered_sorted(sort_key="duration")

    # Test data4
    test_data4 = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["test_data4"], replacements={"data_root": data_folder},
    )
    test_data4 = test_data4.filtered_sorted(sort_key="duration")

    datasets4 = [enrol_data4, test_data4]
    # Enrol data5
    enrol_data5 = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["enrol_data5"], replacements={"data_root": data_folder},
    )
    enrol_data5 = enrol_data5.filtered_sorted(sort_key="duration")

    # Test data5
    test_data5 = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["test_data5"], replacements={"data_root": data_folder},
    )
    test_data5 = test_data5.filtered_sorted(sort_key="duration")

    datasets5 = [enrol_data5, test_data5]
    # Enrol data6
    enrol_data6 = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["enrol_data6"], replacements={"data_root": data_folder},
    )
    enrol_data6 = enrol_data6.filtered_sorted(sort_key="duration")

    # Test data6
    test_data6 = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["test_data6"], replacements={"data_root": data_folder},
    )
    test_data6 = test_data6.filtered_sorted(sort_key="duration")

    datasets6 = [enrol_data6, test_data6]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop):
        start = int(start)
        stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.add_dynamic_item(datasets2, audio_pipeline)
    sb.dataio.dataset.add_dynamic_item(datasets3, audio_pipeline)
    sb.dataio.dataset.add_dynamic_item(datasets4, audio_pipeline)
    sb.dataio.dataset.add_dynamic_item(datasets5, audio_pipeline)
    sb.dataio.dataset.add_dynamic_item(datasets6, audio_pipeline)
    # 3. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])
    sb.dataio.dataset.set_output_keys(datasets2, ["id", "sig"])
    sb.dataio.dataset.set_output_keys(datasets3, ["id", "sig"])
    sb.dataio.dataset.set_output_keys(datasets4, ["id", "sig"])
    sb.dataio.dataset.set_output_keys(datasets5, ["id", "sig"])
    sb.dataio.dataset.set_output_keys(datasets6, ["id", "sig"])

    # 4 Create dataloaders
    # train_dataloader = sb.dataio.dataloader.make_dataloader(
    #     train_data, **params["train_dataloader_opts"]
    # )
    enrol_dataloader = sb.dataio.dataloader.make_dataloader(
        enrol_data, **params["enrol_dataloader_opts"]
    )
    test_dataloader = sb.dataio.dataloader.make_dataloader(
        test_data, **params["test_dataloader_opts"]
    )
    enrol_dataloader2 = sb.dataio.dataloader.make_dataloader(
        enrol_data2, **params["enrol_dataloader_opts"]
    )
    test_dataloader2 = sb.dataio.dataloader.make_dataloader(
        test_data2, **params["test_dataloader_opts"]
    )
    enrol_dataloader3 = sb.dataio.dataloader.make_dataloader(
        enrol_data3, **params["enrol_dataloader_opts"]
    )
    test_dataloader3 = sb.dataio.dataloader.make_dataloader(
        test_data3, **params["test_dataloader_opts"]
    )
    enrol_dataloader4 = sb.dataio.dataloader.make_dataloader(
        enrol_data4, **params["enrol_dataloader_opts"]
    )
    test_dataloader4 = sb.dataio.dataloader.make_dataloader(
        test_data4, **params["test_dataloader_opts"]
    )
    enrol_dataloader5 = sb.dataio.dataloader.make_dataloader(
        enrol_data5, **params["enrol_dataloader_opts"]
    )
    test_dataloader5 = sb.dataio.dataloader.make_dataloader(
        test_data5, **params["test_dataloader_opts"]
    )
    enrol_dataloader6 = sb.dataio.dataloader.make_dataloader(
        enrol_data6, **params["enrol_dataloader_opts"]
    )
    test_dataloader6 = sb.dataio.dataloader.make_dataloader(
        test_data6, **params["test_dataloader_opts"]
    )

    return  enrol_dataloader, test_dataloader,enrol_dataloader2, test_dataloader2,enrol_dataloader3, test_dataloader3,enrol_dataloader4, test_dataloader4,enrol_dataloader5, test_dataloader5,enrol_dataloader6, test_dataloader6


if __name__ == "__main__":
    # Logger setup
    logger = logging.getLogger(__name__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core4.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)

    # Download verification list (to exlude verification sentences from train)
    veri_file_path = os.path.join(
        params["save_folder"], os.path.basename(params["verification_file"])
    )
    veri_file_path2 = os.path.join(
        params["save_folder"], os.path.basename(params["verification_file2"])
    )
    veri_file_path3 = os.path.join(
        params["save_folder"], os.path.basename(params["verification_file3"])
    )
    veri_file_path4 = os.path.join(
        params["save_folder"], os.path.basename(params["verification_file4"])
    )
    veri_file_path5 = os.path.join(
        params["save_folder"], os.path.basename(params["verification_file5"])
    )
    veri_file_path6 = os.path.join(
        params["save_folder"], os.path.basename(params["verification_file6"])
    )
    download_file(params["verification_file"], veri_file_path)
    download_file(params["verification_file2"], veri_file_path2)
    download_file(params["verification_file3"], veri_file_path3)
    download_file(params["verification_file4"], veri_file_path4)
    download_file(params["verification_file5"], veri_file_path5)
    download_file(params["verification_file6"], veri_file_path6)
    from voxceleb_prepare import prepare_voxceleb2  # noqa E402

    # Create experiment directory
    sb.core4.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Prepare data from dev of Voxceleb1
    prepare_voxceleb2(
        data_folder=params["data_folder"],
        save_folder=params["save_folder"],
        verification_pairs_file=veri_file_path,
        verification_pairs_file2=veri_file_path2,
        verification_pairs_file3=veri_file_path3,
        verification_pairs_file4=veri_file_path4,
        verification_pairs_file5=veri_file_path5,
        verification_pairs_file6=veri_file_path6,
        splits=["dev", "test"],
        split_ratio=[90, 10],
        seg_dur=3.0,
        source=params["voxceleb_source"]
        if "voxceleb_source" in params
        else None,
    )

    # here we create the datasets objects as well as tokenization and encoding
    enrol_dataloader, test_dataloader,enrol_dataloader2, test_dataloader2,enrol_dataloader3, test_dataloader3,enrol_dataloader4, test_dataloader4,enrol_dataloader5, test_dataloader5,enrol_dataloader6, test_dataloader6 = dataio_prep(params)

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(params["pretrainer"].collect_files)
    params[ "pretrainer"].load_collected(params["device"])
    params["embedding_model"].eval()
    params["embedding_model"].to(params["device"])
    run_on_main(params["pretrainer2"].collect_files)
    params["pretrainer2"].load_collected(params["device"])
    params["embedding_model2"].eval()
    params["embedding_model2"].to(params["device"])
    # Computing  enrollment and test embeddings
    logger.info("Computing enroll/test embeddings...")

    # First run
    enrol_dict = compute_embedding_loop(enrol_dataloader)
    test_dict = compute_embedding_loop(test_dataloader)

    enrol_dict2 = compute_embedding_loop2(enrol_dataloader)
    test_dict2 = compute_embedding_loop2(test_dataloader)
    # Second run (normalization stats are more stable)
    enrol_dict = compute_embedding_loop(enrol_dataloader)
    test_dict = compute_embedding_loop(test_dataloader)
    ###########################
    # Second run (normalization stats are more stable)
    enrol_dict2 = compute_embedding_loop2(enrol_dataloader)
    test_dict2 = compute_embedding_loop2(test_dataloader)
    logger.info("Computing EER..")
    # Reading standard verification split
    with open(veri_file_path) as f:
        veri_test = [line.rstrip() for line in f]
    positive_scores, negative_scores,positive_scores2, negative_scores2,positive_scores3, negative_scores3,positive_scores4, negative_scores4 = get_verification_scores2(veri_test,enrol_dict, test_dict, enrol_dict2, test_dict2)
    del enrol_dict, test_dict, enrol_dict2, test_dict2
    eer, th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    eer2,th2 = EER(torch.tensor(positive_scores2), torch.tensor(negative_scores2))

    # First run
    enrol_dict21 = compute_embedding_loop(enrol_dataloader2)
    test_dict21 = compute_embedding_loop(test_dataloader2)

    enrol_dict22 = compute_embedding_loop2(enrol_dataloader2)
    test_dict22 = compute_embedding_loop2(test_dataloader2)
    # Second run (normalization stats are more stable)
    enrol_dict21 = compute_embedding_loop(enrol_dataloader2)
    test_dict21 = compute_embedding_loop(test_dataloader2)
    ###########################
    # Second run (normalization stats are more stable)
    enrol_dict22 = compute_embedding_loop2(enrol_dataloader2)
    test_dict22 = compute_embedding_loop2(test_dataloader2)
    with open(veri_file_path2) as f2:
        veri_test2 = [line.rstrip() for line in f2]
    positive_scorese2, negative_scorese2, positive_scorese22, negative_scorese22, positive_scorese23, negative_scorese23, positive_scorese24, negative_scorese24 = get_verification_scores2(veri_test2,enrol_dict21, test_dict21, enrol_dict22, test_dict22)
    del enrol_dict21, test_dict21, enrol_dict22, test_dict22
    eere2, the2 = EER(torch.tensor(positive_scorese2), torch.tensor(negative_scorese2))
    eere22,the22 = EER(torch.tensor(positive_scorese22), torch.tensor(negative_scorese22))


    # First run
    enrol_dict31 = compute_embedding_loop(enrol_dataloader3)
    test_dict31 = compute_embedding_loop(test_dataloader3)

    enrol_dict32 = compute_embedding_loop2(enrol_dataloader3)
    test_dict32 = compute_embedding_loop2(test_dataloader3)
    # Second run (normalization stats are more stable)
    enrol_dict31 = compute_embedding_loop(enrol_dataloader3)
    test_dict31 = compute_embedding_loop(test_dataloader3)
    ###########################
    # Second run (normalization stats are more stable)
    enrol_dict32 = compute_embedding_loop2(enrol_dataloader3)
    test_dict32 = compute_embedding_loop2(test_dataloader3)
    with open(veri_file_path3) as f3:
        veri_test3 = [line.rstrip() for line in f3]
    positive_scorese3, negative_scorese3, positive_scorese32, negative_scorese32, positive_scorese33, negative_scorese33, positive_scorese34, negative_scorese34 = get_verification_scores2(veri_test3,enrol_dict31, test_dict31, enrol_dict32, test_dict32)
    del enrol_dict31, test_dict31, enrol_dict32, test_dict32
    eere3, the3 = EER(torch.tensor(positive_scorese3), torch.tensor(negative_scorese3))
    eere32,the32 = EER(torch.tensor(positive_scorese32), torch.tensor(negative_scorese32))


    # First run
    enrol_dict41 = compute_embedding_loop(enrol_dataloader4)
    test_dict41 = compute_embedding_loop(test_dataloader4)

    enrol_dict42 = compute_embedding_loop2(enrol_dataloader4)
    test_dict42 = compute_embedding_loop2(test_dataloader4)
    # Second run (normalization stats are more stable)
    enrol_dict41 = compute_embedding_loop(enrol_dataloader4)
    test_dict41 = compute_embedding_loop(test_dataloader4)
    ###########################
    # Second run (normalization stats are more stable)
    enrol_dict42 = compute_embedding_loop2(enrol_dataloader4)
    test_dict42 = compute_embedding_loop2(test_dataloader4)
    with open(veri_file_path4) as f4:
        veri_test4 = [line.rstrip() for line in f4]
    positive_scorese4, negative_scorese4, positive_scorese42, negative_scorese42, positive_scorese43, negative_scorese43, positive_scorese44, negative_scorese44 = get_verification_scores2(veri_test4,enrol_dict41, test_dict41, enrol_dict42, test_dict42)
    del enrol_dict41, test_dict41, enrol_dict42, test_dict42
    eere4, the4 = EER(torch.tensor(positive_scorese4), torch.tensor(negative_scorese4))
    eere42,the42 = EER(torch.tensor(positive_scorese42), torch.tensor(negative_scorese42))




    # First run
    enrol_dict51 = compute_embedding_loop(enrol_dataloader5)
    test_dict51 = compute_embedding_loop(test_dataloader5)

    enrol_dict52 = compute_embedding_loop2(enrol_dataloader5)
    test_dict52 = compute_embedding_loop2(test_dataloader5)
    # Second run (normalization stats are more stable)
    enrol_dict51 = compute_embedding_loop(enrol_dataloader5)
    test_dict51 = compute_embedding_loop(test_dataloader5)
    ###########################
    # Second run (normalization stats are more stable)
    enrol_dict52 = compute_embedding_loop2(enrol_dataloader5)
    test_dict52 = compute_embedding_loop2(test_dataloader5)
    with open(veri_file_path5) as f5:
        veri_test5 = [line.rstrip() for line in f5]
    positive_scorese5, negative_scorese5, positive_scorese52, negative_scorese52, positive_scorese53, negative_scorese53, positive_scorese54, negative_scorese54 = get_verification_scores2(veri_test5, enrol_dict51, test_dict51, enrol_dict52, test_dict52)
    del enrol_dict51, test_dict51, enrol_dict52, test_dict52
    eere5, the5 = EER(torch.tensor(positive_scorese5), torch.tensor(negative_scorese5))
    eere52,the52 = EER(torch.tensor(positive_scorese52), torch.tensor(negative_scorese52))


    # First run
    enrol_dict61 = compute_embedding_loop(enrol_dataloader6)
    test_dict61 = compute_embedding_loop(test_dataloader6)

    enrol_dict62 = compute_embedding_loop2(enrol_dataloader6)
    test_dict62 = compute_embedding_loop2(test_dataloader6)
    # Second run (normalization stats are more stable)
    enrol_dict61 = compute_embedding_loop(enrol_dataloader6)
    test_dict61 = compute_embedding_loop(test_dataloader6)
    ###########################
    # Second run (normalization stats are more stable)
    enrol_dict62 = compute_embedding_loop2(enrol_dataloader6)
    test_dict62 = compute_embedding_loop2(test_dataloader6)
    with open(veri_file_path6) as f6:
        veri_test6 = [line.rstrip() for line in f6]
    positive_scorese6, negative_scorese6, positive_scorese62, negative_scorese62, positive_scorese63, negative_scorese63, positive_scorese64, negative_scorese64 = get_verification_scores2(veri_test6,enrol_dict61, test_dict61, enrol_dict62, test_dict62)
    del enrol_dict61, test_dict61, enrol_dict62, test_dict62
    eere6, the6 = EER(torch.tensor(positive_scorese6), torch.tensor(negative_scorese6))
    eere62, the62 = EER(torch.tensor(positive_scorese62), torch.tensor(negative_scorese62))

    # if "score_norm" in params:
    #     train_dict = compute_embedding_loop(train_dataloader)
    # Compute the EER
    logger.info("Computing EER..")
    # Reading standard verification split

    # positive_scores, negative_scores,positive_scores2, negative_scores2,positive_scores3, negative_scores3,positive_scores4, negative_scores4 ,positive_scores5, negative_scores5,positive_scores6, negative_scores6,positive_scores7, negative_scores7,positive_scores8, negative_scores8,positive_scores9, negative_scores9= get_verification_scores(veri_test)
    # del enrol_dict, test_dict, enrol_dict2, test_dict2
    # positive_scorese2, negative_scorese2, positive_scorese22, negative_scorese22, positive_scorese23, negative_scorese23, positive_scorese24, negative_scorese24, positive_scorese25, negative_scorese25, positive_scorese26, negative_scorese26, positive_scorese27, negative_scorese27, positive_scorese28, negative_scorese28, positive_scorese29, negative_scorese29 = get_verification_scores(
    #     veri_test2)
    # del enrol_dict21, test_dict21, enrol_dict22, test_dict22
    # positive_scorese3, negative_scorese3, positive_scorese32, negative_scorese32, positive_scorese33, negative_scorese33, positive_scorese34, negative_scorese34, positive_scorese35, negative_scorese35, positive_scorese36, negative_scorese36, positive_scorese37, negative_scorese37, positive_scorese38, negative_scorese38, positive_scorese39, negative_scorese39 = get_verification_scores(
    #     veri_test3)
    # del enrol_dict31, test_dict31, enrol_dict32, test_dict32
    # positive_scorese4, negative_scorese4, positive_scorese42, negative_scorese42, positive_scorese43, negative_scorese43, positive_scorese44, negative_scorese44, positive_scorese45, negative_scorese45, positive_scorese46, negative_scorese46, positive_scorese47, negative_scorese47, positive_scorese48, negative_scorese48, positive_scorese49, negative_scorese49 = get_verification_scores(
    #     veri_test4)
    # del enrol_dict41, test_dict41, enrol_dict42, test_dict42
    # positive_scorese5, negative_scorese5, positive_scorese52, negative_scorese52, positive_scorese53, negative_scorese53, positive_scorese54, negative_scorese54, positive_scorese55, negative_scorese55, positive_scorese56, negative_scorese56, positive_scorese57, negative_scorese57, positive_scorese58, negative_scorese58, positive_scorese59, negative_scorese59 = get_verification_scores(
    #     veri_test5)
    # del enrol_dict51, test_dict51, enrol_dict52, test_dict52
    #####################
    logger.info("EER(%%)=%f", eer * 100)
    min_dcf, th = minDCF(
        torch.tensor(positive_scores), torch.tensor(negative_scores)
    )
    logger.info("minDCFe1=%f", min_dcf * 100)
    logger.info("EER2(%%)=%f", eer2 * 100)
    min_dcf2, th2 = minDCF(
        torch.tensor(positive_scores2), torch.tensor(negative_scores2)
    )
    logger.info("minDCFe12=%f", min_dcf2 * 100)
    logger.info("EER3(%%)=%f", eere2 * 100)
    min_dcf3, th3 = minDCF(
        torch.tensor(positive_scorese2), torch.tensor(negative_scorese2)
    )
    logger.info("minDCFe2=%f", min_dcf3 * 100)
    logger.info("EER4(%%)=%f", eere22 * 100)
    min_dcf4, th4 = minDCF(
        torch.tensor(positive_scorese22), torch.tensor(negative_scorese22)
    )
    logger.info("minDCFe22=%f", min_dcf4 * 100)
    logger.info("EER5(%%)=%f", eere3 * 100)
    min_dcf5, th5 = minDCF(
        torch.tensor(positive_scorese3), torch.tensor(negative_scorese3)
    )
    logger.info("minDCFe3=%f", min_dcf5 * 100)
    logger.info("EER6(%%)=%f", eere32 * 100)
    min_dcf6, th6 = minDCF(
        torch.tensor(positive_scorese32), torch.tensor(negative_scorese32)
    )
    logger.info("minDCFe32=%f", min_dcf6 * 100)
    logger.info("EER7(%%)=%f", eere4 * 100)
    min_dcf7, th7 = minDCF(
        torch.tensor(positive_scorese4), torch.tensor(negative_scorese4)
    )
    logger.info("minDCFe4=%f", min_dcf7 * 100)
    logger.info("EER8(%%)=%f", eere42 * 100)
    min_dcf8, th8 = minDCF(
        torch.tensor(positive_scorese42), torch.tensor(negative_scorese42)
    )
    logger.info("minDCFe42=%f", min_dcf8 * 100)
    logger.info("EER9(%%)=%f", eere5 * 100)
    min_dcf9, th9 = minDCF(
        torch.tensor(positive_scorese5), torch.tensor(negative_scorese5)
    )
    logger.info("minDCFe5=%f", min_dcf9 * 100)
    logger.info("EER9(%%)=%f", eere52 * 100)
    min_dcf10, th10 = minDCF(
        torch.tensor(positive_scorese52), torch.tensor(negative_scorese52)
    )
    logger.info("minDCFe52=%f", min_dcf10 * 100)
    logger.info("EER-O(%%)=%f", eere6 * 100)
    min_dcf11, th11 = minDCF(
        torch.tensor(positive_scorese6), torch.tensor(negative_scorese6)
    )
    logger.info("minDCFe-O=%f", min_dcf11 * 100)
    logger.info("EER-O1(%%)=%f", eere62 * 100)
    min_dcf12, th12 = minDCF(
        torch.tensor(positive_scorese62), torch.tensor(negative_scorese62)
    )
    logger.info("minDCFe-O1=%f", min_dcf12 * 100)
    #################E1-E5
    # eer, th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    # eer2,th2 = EER(torch.tensor(positive_scores2), torch.tensor(negative_scores2))
    # eer3, th3 = EER(torch.tensor(positive_scores3), torch.tensor(negative_scores3))
    # eer4, th4 = EER(torch.tensor(positive_scores4), torch.tensor(negative_scores4))
    # eer5, th5 = EER(torch.tensor(positive_scores5), torch.tensor(negative_scores5))
    # eer6, th6 = EER(torch.tensor(positive_scores6), torch.tensor(negative_scores6))
    # eer7, th7 = EER(torch.tensor(positive_scores7), torch.tensor(negative_scores7))
    # eer8, th8 = EER(torch.tensor(positive_scores8), torch.tensor(negative_scores8))
    # eer9, th9 = EER(torch.tensor(positive_scores9), torch.tensor(negative_scores9))
    # logger.info("EER(%%)=%f", eer * 100)
    # logger.info("EER2(%%)=%f", eer2 * 100)
    # logger.info("EER3(%%)=%f", eer3 * 100)
    # logger.info("EER4(%%)=%f", eer4 * 100)
    # logger.info("EER5(%%)=%f", eer5 * 100)
    # logger.info("EER6(%%)=%f", eer6 * 100)
    # logger.info("EER7(%%)=%f", eer7 * 100)
    # logger.info("EER8(%%)=%f", eer8 * 100)
    # logger.info("EER9(%%)=%f", eer9 * 100)
    #
    # min_dcf, th = minDCF(
    #     torch.tensor(positive_scores), torch.tensor(negative_scores)
    # )
    # logger.info("minDCF=%f", min_dcf * 100)
    # min_dcf2, th2 = minDCF(
    #     torch.tensor(positive_scores2), torch.tensor(negative_scores2)
    # )
    # logger.info("minDCF2=%f", min_dcf2 * 100)
    # min_dcf3, th3 = minDCF(
    #     torch.tensor(positive_scores3), torch.tensor(negative_scores3)
    # )
    # logger.info("minDCF3=%f", min_dcf3 * 100)
    # min_dcf4, th4 = minDCF(
    #     torch.tensor(positive_scores4), torch.tensor(negative_scores4)
    # )
    # logger.info("minDCF4=%f", min_dcf4 * 100)
    # min_dcf5, th5 = minDCF(
    #     torch.tensor(positive_scores5), torch.tensor(negative_scores5)
    # )
    # logger.info("minDCF5=%f", min_dcf5 * 100)
    # min_dcf6, th6 = minDCF(
    #     torch.tensor(positive_scores6), torch.tensor(negative_scores6)
    # )
    # logger.info("minDCF6=%f", min_dcf6 * 100)
    # min_dcf7, th7 = minDCF(
    #     torch.tensor(positive_scores7), torch.tensor(negative_scores7)
    # )
    # logger.info("minDCF7=%f", min_dcf7 * 100)
    # min_dcf8, th8 = minDCF(
    #     torch.tensor(positive_scores8), torch.tensor(negative_scores8)
    # )
    # logger.info("minDCF8=%f", min_dcf8 * 100)
    # min_dcf9, th9 = minDCF(
    #     torch.tensor(positive_scores9), torch.tensor(negative_scores9)
    # )
    # logger.info("minDCF9=%f", min_dcf9 * 100)
