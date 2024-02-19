import mir_eval
import torch
import torch.nn as nn

from librosa.beat import beat_track
from madmom.features import DBNBeatTrackingProcessor
from mir_eval.beat import trim_beats
from torch.utils.data import DataLoader

import beat_crnn.config as config
from beat_crnn.model import Beat_CRNN
from beat_crnn.data import dataset_splits, collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, dataloader, optimizer, criterion, device=DEVICE):
    epoch_loss = 0
    model.train()
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        X = batch["audio"].to(device)
        Y = batch["beats"].to(device)
        Y_db = batch["downbeats"].to(device)
        Y = torch.cat([Y.unsqueeze(-1), Y_db.unsqueeze(-1)], dim=-1)
        Y_p, _ = model(X)
        loss = criterion(Y_p, Y)
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
            Y = torch.cat([Y.unsqueeze(-1), Y_db.unsqueeze(-1)], dim=-1)
            Y_p, _ = model(X)
            loss = criterion(Y_p, Y)
            val_loss += loss.item()
            del X, Y, Y_p
    return val_loss / len(dataloader)


if __name__ == "__main__":
    D_train, D_val, D_test = dataset_splits()
    DL_train = DataLoader(
        D_train, batch_size=8, shuffle=True, collate_fn=collate_fn, drop_last=True
    )
    DL_val = DataLoader(D_val, batch_size=1, shuffle=False)

    NUM_EPOCHS = 100
    DECAY_PERIOD = 10
    LR = 0.0006

    beat_crnn = Beat_CRNN()
    beat_crnn = beat_crnn.to(DEVICE)
    opt = torch.optim.Adam(beat_crnn.parameters(), lr=LR)
    p = torch.Tensor([20]).to(DEVICE)
    db_p = torch.Tensor([50]).to(DEVICE)
    b_criterion = nn.BCEWithLogitsLoss(pos_weight=p)
    db_criterion = nn.BCEWithLogitsLoss(pos_weight=db_p)

    def weighted_loss(output, target):
        beats_p, db_p = output[..., 0], output[..., 1]
        beats_gt, db_gt = target[..., 0], target[..., 1]
        return b_criterion(beats_p, beats_gt) + db_criterion(db_p, db_gt)

    min_val_loss = 1
    min_val_loss_path = None

    for i in range(NUM_EPOCHS):
        print("Epoch:", i)
        epoch_loss = train_epoch(beat_crnn, DL_train, opt, weighted_loss)
        print("Training loss:", epoch_loss)
        val_loss = validate_model(beat_crnn, DL_val, weighted_loss)
        print("Validation loss:", val_loss)

        if val_loss < min_val_loss:
            fp = f"asym_pos_class_beat_crnn_epoch_{i}_LR_{LR}.pth"
            print("Writing model to disk:", fp)
            torch.save(beat_crnn.state_dict(), fp)
            min_val_loss = val_loss
            min_val_loss_path = fp

        if i > 0 and i % DECAY_PERIOD == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.9

    print("Best model:", min_val_loss_path)
