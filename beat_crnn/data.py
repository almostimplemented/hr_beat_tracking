import librosa
import numpy as np
import os
import torch

from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import beat_crnn.config as config


def annotation_fp_from_audio_fp(audio_fp, anno_dir=config.ANNOTATIONS_ROOT):
    return Path(anno_dir, Path(audio_fp).stem + ".beats")


def read_annotation_file(anno_fp):
    raw_anno = [l.split() for l in open(anno_fp).read().splitlines()]
    return np.array([int(beat) for _, beat in raw_anno]), np.array(
        [float(t) for t, _ in raw_anno]
    )


def ballroom_filepaths(audio_root=config.AUDIO_ROOT, dataset_list=config.DATASET_LIST):
    f = open(os.path.join(audio_root, dataset_list))
    ballroom_filepaths = [
        os.path.join(audio_root, filename) for filename in f.read().splitlines()
    ]
    deduped_ballroom_fps = [
        fp
        for fp in ballroom_filepaths
        if not any(dup[1] == Path(fp).name for dup in duplicates)
    ]
    return deduped_ballroom_fps


def resample_and_write_file(input_fp, output_fp, sr=config.SAMPLE_RATE):
    w, _ = librosa.load(input_fp, sr=sr)
    np.save(output_fp, w)


def resample_data(output_dir=config.RESAMPLED_OUTPUT_DIR):
    fps = ballroom_filepaths()
    for fp in tqdm(fps):
        output_fp = os.path.join(output_dir, Path(fp).stem)
        resample_and_write_file(fp, output_fp)


class BallroomDataset(Dataset):
    def __init__(
        self,
        sr=config.SAMPLE_RATE,
        frames_per_sec=config.FRAMES_PER_SEC,
        anno_window=3,
        audio_dir=config.RESAMPLED_OUTPUT_DIR,
        anno_dir=config.ANNOTATIONS_ROOT,
    ):
        hop_size = sr // frames_per_sec
        audio_filenames = os.listdir(audio_dir)
        items = []
        for fn in audio_filenames:
            fp = Path(audio_dir, fn)
            beats, times = read_annotation_file(
                annotation_fp_from_audio_fp(fp, anno_dir)
            )
            # create beat and downbeat vectors
            beattimes = np.column_stack((beats, times))
            audio = np.load(fp)
            beats_signal = torch.zeros(audio.shape[0] // hop_size + 1)
            downbeats_signal = torch.zeros(audio.shape[0] // hop_size + 1)
            for (beat, time) in beattimes:
                beat_center = int(frames_per_sec * time + 0.5)
                for sigma in range(-anno_window, anno_window + 1):
                    idx = beat_center + sigma
                    if 0 <= idx < len(beats_signal):
                        v = np.exp(-(((np.abs(beat_center - idx)) / 1.5) ** 2 / 2))
                        beats_signal[idx] = v
                        if beat == 1:
                            downbeats_signal[idx] = v

            item = {
                "audio": torch.from_numpy(audio),
                "beats": beats_signal,
                "downbeats": downbeats_signal,
                "beats_sec": times,
                "db_sec": np.array([t for (b, t) in beattimes if b == 1]),
                "audio_fn": fn,
            }
            items.append(item)
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]
