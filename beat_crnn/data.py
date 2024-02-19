import librosa
import numpy as np
import os
import torch

from pathlib import Path
from torch.utils.data import Dataset, random_split

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


def dataset_splits(seed=42, train_ratio=0.8):
    """Returns deterministic, pseudo-random splits.

    Args:
        seed: Random seed for shuffling dataset
        train_ratio: Value between 0 and 1; proportion of data to use for training.
            The remaining amount (1 - train_ratio) is split evenly between val/test.

    Returns:
        Tuple of Datasets: D_train, D_val, D_test
    """
    dataset = BallroomDataset()

    N_total = len(dataset)
    N_train = int(train_ratio * N_total)
    N_val = (N_total - N_train) // 2
    N_test = N_total - N_train - N_val

    D_train, D_val, D_test = random_split(
        dataset, [N_train, N_val, N_test], generator=torch.Generator().manual_seed(seed)
    )

    return D_train, D_val, D_test


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
