from pathlib import Path
import numpy as np
import random
import torch

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


PROJECT_ROOT = Path(r"")

RAW_DATA_ROOT = PROJECT_ROOT / "data" / "ds002722-download" / "ds002722"
PROCESSED_ROOT = PROJECT_ROOT / "processed"
MODELS_ROOT = PROJECT_ROOT / "models"
CODE_ROOT = PROJECT_ROOT / "code"

PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
MODELS_ROOT.mkdir(parents=True, exist_ok=True)


SUBJECTS = [
    "sub-17"
]

EEG_EXT = ".edf"

EEG_FILENAME_TEMPLATE = "{subject}_task-run1_eeg.edf"

ECG_CHANNEL_NAMES = ["ECG", "EKG"]
EOG_CHANNEL_NAMES = ["EOG", "HEOG", "VEOG"]

RAW_EEG_SAMPLING_RATE = None
TARGET_EEG_SAMPLING_RATE = 100


EEG_BANDPASS_LOW = 0.1
EEG_BANDPASS_HIGH = 40.0


ICA_N_COMPONENTS = 20
ICA_RANDOM_STATE = SEED
ICA_MAX_ITER = 1000


TRIAL_DURATION_SECONDS = 40.0
BASELINE = None


AUDIO_SAMPLE_RATE = 44100
ENVELOPE_LOW_PASS = 10.0
ENVELOPE_RESAMPLE_RATE = TARGET_EEG_SAMPLING_RATE


BILSTM_INPUT_DIM = None
BILSTM_HIDDEN_SIZE = 250
BILSTM_NUM_LAYERS = 4
BILSTM_DROPOUT = 0.2
BILSTM_BIDIRECTIONAL = True

LEARNING_RATE = 1e-3
BATCH_SIZE = 8
NUM_EPOCHS = 50


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def raw_eeg_paths(subject):
    eeg_dir = RAW_DATA_ROOT / subject / "eeg"
    assert eeg_dir.exists(), f"Missing EEG directory: {eeg_dir}"

    edf_files = []
    for f in eeg_dir.iterdir():
        name = f.name.lower()
        if (
            name.startswith(subject.lower())
            and "_task-run" in name
            and name.endswith("_eeg.edf")
        ):
            edf_files.append(f)

    edf_files = sorted(set(edf_files))

    assert len(edf_files) == 5, (
        f"Expected 5 runs, found {len(edf_files)}:\n"
        + "\n".join([f.name for f in edf_files])
    )

    for f in edf_files:
        assert f.stat().st_size > 0, f"EDF file is empty: {f}"

    return edf_files


def processed_subject_dir(subject):
    d = PROCESSED_ROOT / subject
    d.mkdir(parents=True, exist_ok=True)
    return d


def pairs_dir(subject):
    d = processed_subject_dir(subject) / "pairs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def listen_dir(subject):
    d = pairs_dir(subject) / "listen"
    d.mkdir(parents=True, exist_ok=True)
    return d
