import beat_crnn.config as config
import librosa
import torch

from beat_crnn.model import Beat_CRNN
from beat_crnn.postprocess import findBeats, madmomBeats
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def beatTracker(inputFile, postprocessor="peaks"):
    w, _ = librosa.load(inputFile, sr=config.SAMPLE_RATE)
    audio = torch.from_numpy(w).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        model = Beat_CRNN()
        model.load_state_dict(torch.load(Path("checkpoints", config.CHECKPOINT)))
        model.to(DEVICE)
        _, Y_p = model(audio)
        Y_p = Y_p.squeeze()
        Y_db_p = Y_p[:, 1]
        Y_p = Y_p[:, 0]

        if postprocessor == "peaks":
            beats = findBeats(Y_p.cpu().numpy())
            downbeats = findBeats(Y_db_p.cpu().numpy(), beat_type="downbeat")
        elif postprocessor == "madmom":
            beats = madmomBeats(Y_p.cpu().numpy())
            downbeats = madmomBeats(Y_db_p.cpu().numpy(), beat_type="downbeat")
        else:
            raise ValueError(f"Unrecognized beat_type: {beat_type}")

        return beats, downbeats
