import mir_eval
import numpy as np
import scipy.signal
import beat_crnn.config as config
from madmom.features import DBNBeatTrackingProcessor


def findBeats(
    Y_p,
    threshold=0.7,
    fps=config.FRAMES_PER_SEC,
    beat_type="beat",
):
    if beat_type == "beat":
        distance = fps / 4
    elif beat_type == "downbeat":
        distance = fps / 2
    else:
        raise RuntimeError(f"Invalid beat_type: `{beat_type}`.")

    # apply simple peak picking
    beats, _ = scipy.signal.find_peaks(Y_p, height=threshold, distance=distance)

    # compute beat points (samples) to seconds
    beats = beats / float(fps)

    return beats


def madmomBeats(Y_p, beat_type="beat"):
    dbn = DBNBeatTrackingProcessor(
        min_bpm=55,
        max_bpm=215,
        transition_lambda=100,
        fps=config.FRAMES_PER_SEC,
        online=False,
    ) if beat_type == "beat" else DBNBeatTrackingProcessor(
        min_bpm=10,
        max_bpm=75,
        transition_lambda=100,
        fps=config.FRAMES_PER_SEC,
        online=False,
    )

    beats = dbn.process_offline(Y_p)

    return beats
