import mir_eval
import torch
import torch.nn as nn

from librosa.beat import beat_track
from mir_eval.beat import trim_beats
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from tqdm import tqdm

import beat_crnn.config as config
from beat_crnn import beatTracker
from beat_crnn.data import dataset_splits
from beat_crnn.model import Beat_CRNN
from beat_crnn.postprocess import findBeats, madmomBeats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(
    checkpoint_path, dataloader, device=DEVICE, postprocessor="simple", verbose=False
):
    print(f"Evaluating {checkpoint_path}")
    model = Beat_CRNN()
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    f1_scores = []
    db_f1_scores = []
    cmlts = []
    db_cmlts = []
    amlts = []
    db_amlts = []
    # librosa_f1_scores = []
    with torch.no_grad():
        model.eval()
        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, item in bar:
            X = item["audio"].to(device)
            _, Y_p = model(X)
            Y_p = Y_p.squeeze()
            Y_db_p = Y_p[:, 1]
            Y_p = Y_p[:, 0]
            cut_off = 3000 #max(item["beats_sec"][0]).item() + 0.5
            beats = findBeats(Y_p.cpu().numpy())
            downbeats = findBeats(Y_db_p.cpu().numpy(), beat_type="downbeat")

            ref = trim_beats(item["beats_sec"][0].numpy())
            est = trim_beats(beats[beats < cut_off])
            ref_db = trim_beats(item["db_sec"][0].numpy())
            est_db = trim_beats(downbeats)
            f1_score = mir_eval.beat.f_measure(
                ref, est
            )
            db_f1_score = mir_eval.beat.f_measure(
                ref_db, est_db
            )
            _, cmlt, _, amlt = mir_eval.beat.continuity(
                ref, est
            )
            _, db_cmlt, _, db_amlt = mir_eval.beat.continuity(
                ref_db, est_db
            )

            f1_scores.append(f1_score)
            db_f1_scores.append(db_f1_score)
            cmlts.append(cmlt)
            db_cmlts.append(db_cmlt)
            amlts.append(amlt)
            db_amlts.append(db_amlt)

            avg_f1 = sum(f1_scores) / len(f1_scores)
            db_avg_f1 = sum(db_f1_scores) / len(db_f1_scores)
            bar.set_description(f"Beat F1: {avg_f1} || DB F1: {db_avg_f1}")

            if verbose:
                print("Predicted beats:", beats)
                print("Actual beats:", item["beats_sec"])
                print("F1 score:", f1_score)
                print("Predicted downbeats:", downbeats)
                print("Actual downbeats:", item["db_sec"])
                print("DB F1 score:", db_f1_score)

            # Comparison with librosa beat tracker

            # tempo, beats = beat_track(
            #    y=item["audio"].squeeze().numpy(), sr=config.SAMPLE_RATE, units="time"
            # )
            # f1_score = mir_eval.beat.f_measure(
            #    trim_beats(beats), trim_beats(item["beats_sec"][0])
            # )
            # librosa_f1_scores.append(f1_score)

    print("avg F1 score (beats):", sum(f1_scores) / len(f1_scores))
    print("min F1 score (beats):", min(f1_scores))
    print("avg F1 score (downbeats):", sum(db_f1_scores) / len(db_f1_scores))
    print("min F1 score (downbeats):", min(db_f1_scores))
    print("avg CML_t score (beats):", sum(cmlts) / len(cmlts))
    print("avg CML_t score (downbeats):", sum(db_cmlts) / len(db_cmlts))
    print("avg AML_t score (beats):", sum(amlts) / len(amlts))
    print("avg AML_t score (downbeats):", sum(db_amlts) / len(db_amlts))


if __name__ == "__main__":
    _, _, D_test = dataset_splits()
    DL_test = DataLoader(D_test, batch_size=1, shuffle=False)
    checkpoint_path = Path("checkpoints", config.CHECKPOINT)
    evaluate_model(checkpoint_path, DL_test)
