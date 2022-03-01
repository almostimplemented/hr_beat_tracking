import mir_eval
import torch
import torch.nn as nn

from librosa.beat import beat_track
from madmom.features import DBNBeatTrackingProcessor
from mir_eval.beat import trim_beats
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

import beat_crnn.config as config
from beat_crnn.model import Beat_CRNN
from beat_crnn.data import BallroomDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(data):
    max_audio_len = max(item["audio"].shape[0] for item in data)
    max_target_len = max(item["beats"].shape[0] for item in data)
    audio_batch = torch.zeros((len(data), max_audio_len))
    beats_batch = torch.zeros((len(data), max_target_len))
    downbeats_batch = torch.zeros((len(data), max_target_len))
    beats_sec_batch = []
    db_sec_batch = []
    audio_fn_batch = []
    for i in range(len(data)):
        audio_len = data[i]["audio"].shape[0]
        target_len = data[i]["beats"].shape[0]
        audio_batch[i] = torch.cat(
            [data[i]["audio"], torch.zeros((max_audio_len - audio_len))]
        )
        beats_batch[i] = torch.cat(
            [data[i]["beats"], torch.zeros((max_target_len - target_len))]
        )
        downbeats_batch[i] = torch.cat(
            [data[i]["downbeats"], torch.zeros((max_target_len - target_len))]
        )
        beats_sec_batch.append(data[i]["beats_sec"])
        db_sec_batch.append(data[i]["db_sec"])
        audio_fn_batch.append(data[i]["audio_fn"])

    return {
        "audio": audio_batch,
        "beats": beats_batch,
        "downbeats": downbeats_batch,
        "beats_sec": beats_sec_batch,
        "db_sec": db_sec_batch,
        "audio_fn": audio_fn_batch,
    }


def train_epoch(model, dataloader, optimizer, criterion, device=DEVICE):
    epoch_loss = 0
    model.train()
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        X = batch["audio"].to(device)
        Y = batch["beats"].to(device)
        Y_db = batch["downbeats"].to(device)
        Y_p, Y_db_p = model(X)
        Y_p, Y_db_p = Y_p.squeeze(), Y_db_p.squeeze()
        loss = criterion(Y_p, Y) + criterion(Y_db_p, Y_db)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        epoch_loss += loss.item()
        del X, Y, Y_p
    return epoch_loss / len(dataloader)


def validate_model(model, dataloader, criterion, device=DEVICE):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            X = batch["audio"].to(device)
            Y = batch["beats"].to(device)
            Y_db = batch["downbeats"].to(device)
            Y_p, Y_db_p = model(X)
            Y_p, Y_db_p = Y_p.squeeze(), Y_db_p.squeeze()
            loss = criterion(Y_p, Y.squeeze()) + criterion(Y_db_p, Y_db.squeeze())
            val_loss += loss.item()
            del X, Y, Y_p
    return val_loss / len(dataloader)


def evaluate_model(checkpoint_path, dataloader, device=DEVICE):
    print(f"Evaluating {checkpoint_path}")
    model = Beat_CRNN()
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    dbn = DBNBeatTrackingProcessor(
        min_bpm=55, max_bpm=215, transition_lambda=100, fps=config.FRAMES_PER_SEC, online=False
    )
    db_dbn = DBNBeatTrackingProcessor(
        min_bpm=10, max_bpm=75, transition_lambda=100, fps=config.FRAMES_PER_SEC, online=False
    )

    f1_scores = []
    db_f1_scores = []
    # librosa_f1_scores = []
    with torch.no_grad():
        model.eval()
        for i, item in enumerate(dataloader):
            X = item["audio"].to(device)
            Y_p, Y_db_p = model(X)
            Y_p, Y_db_p = Y_p.squeeze(), Y_db_p.squeeze()
            Y_p = Y_p.to("cpu")
            Y_db_p = Y_db_p.to("cpu")
            dbn.reset()
            db_dbn.reset()
            beats = dbn.process_offline(Y_p)
            downbeats = db_dbn.process_offline(Y_db_p)
            f1_score = mir_eval.beat.f_measure(
                trim_beats(beats), trim_beats(item["beats_sec"][0])
            )
            f1_scores.append(f1_score)
            db_f1_score = mir_eval.beat.f_measure(
                trim_beats(downbeats), trim_beats(item["db_sec"][0])
            )
            db_f1_scores.append(db_f1_score)
            #tempo, beats = beat_track(
            #    y=item["audio"].squeeze().numpy(), sr=config.SAMPLE_RATE, units="time"
            #)
            #f1_score = mir_eval.beat.f_measure(
            #    trim_beats(beats), trim_beats(item["beats_sec"][0])
            #)
            # librosa_f1_scores.append(f1_score)

    print("avg score (beats):", sum(f1_scores) / len(f1_scores))
    print("min score (beats):", min(f1_scores))
    print("avg score (downbeats):", sum(db_f1_scores) / len(db_f1_scores))
    print("min score (downbeats):", min(db_f1_scores))


if __name__ == "__main__":
    SEED = 42
    dataset = BallroomDataset()

    N_total = len(dataset)
    N_train = int(0.8 * N_total)
    N_val = (N_total - N_train) // 2
    N_test = N_total - N_train - N_val

    D_train, D_val, D_test = random_split(
        dataset, [N_train, N_val, N_test], generator=torch.Generator().manual_seed(SEED)
    )
    DL_train = DataLoader(
        D_train, batch_size=4, shuffle=True, collate_fn=collate_fn, drop_last=True
    )
    DL_val = DataLoader(D_val, batch_size=1, shuffle=False)
    DL_test = DataLoader(D_test, batch_size=1, shuffle=False)

    NUM_EPOCHS = 100
    DECAY_PERIOD = 10
    LR = 0.0002

    beat_crnn = Beat_CRNN()
    beat_crnn = beat_crnn.to(DEVICE)
    opt = torch.optim.Adam(beat_crnn.parameters(), lr=LR)
    criterion = nn.BCELoss()

    min_val_loss = 0.5
    min_val_loss_path = None

    for i in []: #range(NUM_EPOCHS):
        epoch_loss = train_epoch(beat_crnn, DL_train, opt, criterion)
        print("Epoch loss:", epoch_loss)

        val_loss = validate_model(beat_crnn, DL_val, criterion)
        print("Validation loss:", val_loss)

        if val_loss < min_val_loss:
            fp = f"beat_crnn_epoch_{i}.pth"
            print("Writing model to disk:", fp)
            torch.save(beat_crnn.state_dict(), fp)
            min_val_loss = val_loss
            min_val_loss_path = fp

        if i > 0 and i % DECAY_PERIOD == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.9

    evaluate_model('beat_crnn_epoch_89.pth', DL_test)
