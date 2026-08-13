"""
Central configuration for the Adaptive Neuro-AI BCI system.

Every tunable value lives here. No module is allowed to hard-code a path,
a sampling rate or a class order. Values can be overridden with environment
variables so the same code runs on a laptop, a lab machine or a server.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Root folder of the PhysioNet EEG Motor Movement/Imagery dataset.
# Expected layout:  <DATASET_ROOT>/S001/S001R03.edf ...
DATASET_ROOT = Path(
    os.environ.get("BCI_DATASET_ROOT", PROJECT_ROOT / "data" / "physionet")
)

MODEL_DIR = Path(os.environ.get("BCI_MODEL_DIR", PROJECT_ROOT / "models"))
MODEL_PATH = MODEL_DIR / "pytorch_bci_model.pth"
HISTORY_PATH = MODEL_DIR / "training_history.pkl"
ADAPTED_MODEL_PATH = MODEL_DIR / "pytorch_bci_model_adapted.pth"
REPORT_DIR = Path(os.environ.get("BCI_REPORT_DIR", PROJECT_ROOT / "reports"))

# --------------------------------------------------------------------------
# Signal parameters
# --------------------------------------------------------------------------
TARGET_SAMPLING_RATE = 160          # Hz - native rate of the PhysioNet set
EPOCH_DURATION = 4.0                # seconds per trial
TARGET_SAMPLES = int(TARGET_SAMPLING_RATE * EPOCH_DURATION)   # 640
N_CHANNELS = 64                     # PhysioNet uses the 64-channel montage
BANDPASS_LOW = 7.0                  # Hz - mu rhythm lower edge
BANDPASS_HIGH = 30.0                # Hz - beta rhythm upper edge

# --------------------------------------------------------------------------
# Classes  (index order is a contract - never reorder)
# --------------------------------------------------------------------------
CLASS_NAMES = ["LEFT", "RIGHT", "REST"]
CLASS_LEFT, CLASS_RIGHT, CLASS_REST = 0, 1, 2
N_CLASSES = len(CLASS_NAMES)

# PhysioNet annotation label -> class index.
# T0 = rest, T1 = left fist (runs 3/7/11), T2 = right fist (runs 3/7/11).
ANNOTATION_TO_CLASS = {"T0": CLASS_REST, "T1": CLASS_LEFT, "T2": CLASS_RIGHT}

# Motor-imagery runs of the PhysioNet protocol
MOTOR_IMAGERY_RUNS = [4, 8, 12]     # imagined movement
MOTOR_EXECUTION_RUNS = [3, 7, 11]   # executed movement
TRAIN_SUBJECTS = list(range(1, 101))
TEST_SUBJECTS = list(range(101, 110))

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Self-healing layer
# --------------------------------------------------------------------------
CONFIDENCE_WINDOW = 20              # predictions kept in the drift window
CONFIDENCE_DRIFT_THRESHOLD = 0.45   # mean confidence below this = drift
ENTROPY_DRIFT_THRESHOLD = 0.95      # normalised entropy above this = drift
FLATLINE_STD_THRESHOLD = 1e-6       # channel treated as dead below this std
SATURATION_Z_THRESHOLD = 8.0        # |z| above this = artefact / saturation
ADAPTATION_LEARNING_RATE = 1e-4     # online fine-tuning rate
ADAPTATION_STEPS = 5                # gradient steps per healing event
MIN_SECONDS_BETWEEN_HEALS = 15      # cool-down so healing cannot thrash


def ensure_directories() -> None:
    """Create the writable folders the app expects. Safe to call repeatedly."""
    for folder in (MODEL_DIR, REPORT_DIR):
        folder.mkdir(parents=True, exist_ok=True)
