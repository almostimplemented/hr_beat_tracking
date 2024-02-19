# File defaults
AUDIO_ROOT = "data"
DATASET_LIST = "allBallroomFiles"
ANNOTATIONS_ROOT = "annotations"
RESAMPLED_OUTPUT_DIR = "resampled"
DUPLICATES = [
    ("Albums-AnaBelen_Veneo-11.wav", "Albums-Chrisanne2-12.wav"),
    ("Albums-Fire-08.wav", "Albums-Fire-09.wav"),
    ("Albums-Latin_Jam2-05.wav", "Albums-Latin_Jam2-13.wav"),
    ("Albums-Secret_Garden-01.wav", "Media-104705.wav"),
    ("Albums-AnaBelen_Veneo-03.wav", "Albums-AnaBelen_Veneo-15.wav"),
    ("Albums-Ballroom_Magic-03.wav", "Albums-Ballroom_Magic-18.wav"),
    ("Albums-Latin_Jam-04.wav", "Albums-Latin_Jam-13.wav"),
    ("Albums-Latin_Jam-08.wav", "Albums-Latin_Jam-14.wav"),
    ("Albums-Latin_Jam-06.wav", "Albums-Latin_Jam-15.wav"),
    ("Albums-Latin_Jam2-02.wav", "Albums-Latin_Jam2-14.wav"),
    ("Albums-Latin_Jam2-07.wav", "Albums-Latin_Jam2-15.wav"),
    ("Albums-Latin_Jam3-02.wav", "Media-103414.wav"),
    ("Media-103402.wav", "Media-103415.wav"),
]

# Spectrogram defaults
SAMPLE_RATE = 16000
WINDOW_SIZE = 2048
FRAMES_PER_SEC = 100
HOP_SIZE = SAMPLE_RATE // FRAMES_PER_SEC
MEL_BINS = 128

# Default model to use
CHECKPOINT = "asym_pos_class_beat_crnn_epoch_87_LR_0.0006.pth"
