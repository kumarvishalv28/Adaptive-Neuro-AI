"""
Smoke tests. They run without the dataset and without a trained checkpoint,
so `pytest` is a valid pre-commit gate on any machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import N_CHANNELS, N_CLASSES, TARGET_SAMPLES
from src.core.model import EEGNetTCN
from src.core.preprocessing import conform_channels, normalize_per_channel, to_model_input
from src.core.self_healing import (
    SelfHealingController,
    detect_bad_channels,
    normalized_entropy,
    repair_channels,
)


def test_normalisation_handles_dead_channel():
    data = np.random.randn(2, 8, 100)
    data[0, 3, :] = 0.0                       # flat channel
    out = normalize_per_channel(data)
    assert np.isfinite(out).all()
    assert out.shape == data.shape


def test_channel_conforming_pads_and_truncates():
    assert conform_channels(np.random.randn(4, 640)).shape[0] == N_CHANNELS
    assert conform_channels(np.random.randn(128, 640)).shape[0] == N_CHANNELS


def test_model_input_shape_and_forward():
    tensor = torch.from_numpy(to_model_input(np.random.randn(8, 500)))
    assert tuple(tensor.shape) == (1, 1, N_CHANNELS, TARGET_SAMPLES)
    logits = EEGNetTCN()(tensor)
    assert tuple(logits.shape) == (1, N_CLASSES)


def test_bad_channel_detection_and_repair():
    window = np.random.randn(16, 640)
    window[2] = 0.0
    window[9] = window[9] * 1000.0
    bad = detect_bad_channels(window)
    assert 2 in bad
    repaired = repair_channels(window, bad)
    assert repaired[2].std() > 0


def test_entropy_bounds():
    assert normalized_entropy(np.array([1.0, 0.0, 0.0])) < 1e-6
    assert abs(normalized_entropy(np.ones(3) / 3) - 1.0) < 1e-6


def test_controller_detects_low_confidence_drift():
    controller = SelfHealingController(EEGNetTCN(), torch.device("cpu"))
    for _ in range(20):
        controller.observe(np.array([0.34, 0.33, 0.33]))
    assert controller.is_drifting()
