"""
EEG preprocessing utilities.

One pipeline, used identically by training and by inference. If these two
ever diverge, accuracy collapses silently - so every caller goes through
the functions in this module.

Pipeline: load EDF -> band-pass 7-30 Hz -> epoch 0-4 s around each event ->
resample/trim to 640 samples -> per-channel z-score -> (batch, 1, C, T).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import signal as sp_signal   # aliased: never shadow with a variable

from .config import (
    ANNOTATION_TO_CLASS,
    BANDPASS_HIGH,
    BANDPASS_LOW,
    CLASS_REST,
    EPOCH_DURATION,
    N_CHANNELS,
    TARGET_SAMPLES,
    TARGET_SAMPLING_RATE,
)

logger = logging.getLogger(__name__)

_EPS = 1e-8


# --------------------------------------------------------------------------
# Array level helpers
# --------------------------------------------------------------------------
def normalize_per_channel(data: np.ndarray) -> np.ndarray:
    """
    Z-score every channel of every epoch independently.

    Vectorised, and guarded against zero-variance (dead electrode) channels,
    which would otherwise produce NaN and poison the whole batch.
    """
    data = np.asarray(data, dtype=np.float64)
    mean = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True)
    std = np.where(std < _EPS, 1.0, std)
    out = (data - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def resample_to_target(
    data: np.ndarray,
    original_sfreq: float,
    target_sfreq: int = TARGET_SAMPLING_RATE,
    target_samples: int = TARGET_SAMPLES,
) -> np.ndarray:
    """Resample along the last axis, then pad or trim to a fixed length."""
    data = np.asarray(data, dtype=np.float64)

    if abs(original_sfreq - target_sfreq) > 1e-6:
        n = int(round(data.shape[-1] * target_sfreq / float(original_sfreq)))
        data = sp_signal.resample(data, n, axis=-1)

    current = data.shape[-1]
    if current < target_samples:
        pad = [(0, 0)] * data.ndim
        pad[-1] = (0, target_samples - current)
        data = np.pad(data, pad, mode="constant")
    elif current > target_samples:
        data = data[..., :target_samples]
    return data


def conform_channels(data: np.ndarray, n_channels: int = N_CHANNELS) -> np.ndarray:
    """
    Force the channel axis to the size the model expects.

    Devices with fewer electrodes (Muse: 4, OpenBCI Cyton: 8) are tiled and
    zero-padded; devices with more are truncated. Without this, a 4-channel
    headset raises a shape error inside the depthwise convolution.
    """
    data = np.asarray(data, dtype=np.float64)
    have = data.shape[-2]
    if have == n_channels:
        return data
    if have > n_channels:
        return data[..., :n_channels, :]

    reps = int(np.ceil(n_channels / have))
    tiled = np.concatenate([data] * reps, axis=-2)
    return tiled[..., :n_channels, :]


def to_model_input(data: np.ndarray) -> np.ndarray:
    """
    Take (C, T) or (N, C, T) and return normalised (N, 1, C, T) float32.
    This is the only accepted way to build a tensor for the network.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    if data.ndim != 3:
        raise ValueError(f"Expected (C, T) or (N, C, T), got shape {data.shape}")

    data = conform_channels(data)
    if data.shape[-1] != TARGET_SAMPLES:
        data = resample_to_target(data, TARGET_SAMPLING_RATE)
    data = normalize_per_channel(data)
    return data[:, np.newaxis, :, :].astype(np.float32)


# --------------------------------------------------------------------------
# EDF level helpers
# --------------------------------------------------------------------------
def load_edf_epochs(
    edf_path: str | Path,
    tmin: float = 0.0,
    tmax: float = EPOCH_DURATION,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Read one EDF recording and return (epochs, labels, annotation_map).

    Labels are derived from the annotation *description* (T0/T1/T2), never
    from the raw integer event code. MNE assigns those codes alphabetically
    (T0->1, T1->2, T2->3), so mapping the integers directly shifts every
    label by one class and quietly destroys accuracy.
    """
    import mne
    from mne.io import read_raw_edf

    raw = read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw.filter(
    BANDPASS_LOW,
    BANDPASS_HIGH,
    method="iir",
    iir_params=dict(order=4, ftype="butter"),
    verbose=False,
)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    if events.size == 0:
        return np.empty((0, raw.info["nchan"], TARGET_SAMPLES)), np.empty((0,), int), {}

    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=tmin, tmax=tmax, baseline=None,
        preload=True, verbose=False, on_missing="ignore",
    )

    data = epochs.get_data()
    code_to_desc = {code: desc for desc, code in event_id.items()}
    labels = np.array(
        [ANNOTATION_TO_CLASS.get(code_to_desc.get(code, ""), CLASS_REST)
         for code in epochs.events[:, -1]],
        dtype=np.int64,
    )

    data = resample_to_target(data, raw.info["sfreq"])
    return data, labels, event_id


def find_subject_edf_files(
    dataset_root: str | Path, subject: int, runs: Iterable[int]
) -> list[Path]:
    """Locate <root>/S001/S001R04.edf style files for one subject."""
    root = Path(dataset_root) / f"S{subject:03d}"
    if not root.exists():
        return []
    found = []
    for run in runs:
        candidate = root / f"S{subject:03d}R{run:02d}.edf"
        if candidate.exists():
            found.append(candidate)
        else:
            logger.debug("Missing run file: %s", candidate)
    return found
