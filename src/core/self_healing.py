"""
Self-healing layer.

This is the novel contribution of the project. It sits between the raw
signal and the classifier and does three things:

1. SIGNAL REPAIR   - detects dead, saturated or artefact-ridden channels in
                     each incoming window and repairs them by spatial
                     interpolation from the healthy neighbours.
2. DRIFT DETECTION - watches a rolling window of prediction confidence and
                     output entropy. Sustained low confidence or high
                     entropy means the decoder no longer fits the user's
                     current brain state.
3. ONLINE ADAPTION - when drift is confirmed, a few gradient steps of
                     entropy-minimisation fine-tuning are applied to the
                     final dense layers only, using the recent unlabelled
                     buffer. The backbone stays frozen so the model cannot
                     catastrophically forget, and a cool-down prevents
                     repeated healing from thrashing the weights.

Every event is logged so the dashboard can show what the system healed and
when, which is exactly what the report needs to demonstrate resilience.
"""

from __future__ import annotations

import copy
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    ADAPTATION_LEARNING_RATE,
    ADAPTATION_STEPS,
    CONFIDENCE_DRIFT_THRESHOLD,
    CONFIDENCE_WINDOW,
    ENTROPY_DRIFT_THRESHOLD,
    FLATLINE_STD_THRESHOLD,
    MIN_SECONDS_BETWEEN_HEALS,
    N_CLASSES,
    SATURATION_Z_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class HealingEvent:
    timestamp: float
    kind: str                     # "signal_repair" | "model_adaptation"
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# 1. Signal repair
# --------------------------------------------------------------------------
def detect_bad_channels(window: np.ndarray) -> list[int]:
    """
    Return the indices of channels that look broken in this window.

    A channel is bad when it is flat (electrode fell off / amplifier dead)
    or when its amplitude is a gross outlier relative to the montage
    (saturation, movement artefact, 50 Hz pickup).
    """
    window = np.asarray(window, dtype=np.float64)
    stds = window.std(axis=-1)

    bad = set(np.flatnonzero(stds < FLATLINE_STD_THRESHOLD).tolist())

    healthy = stds[stds >= FLATLINE_STD_THRESHOLD]
    if healthy.size >= 4:
        median = np.median(healthy)
        mad = np.median(np.abs(healthy - median)) + 1e-12
        robust_z = 0.6745 * (stds - median) / mad
        bad.update(np.flatnonzero(np.abs(robust_z) > SATURATION_Z_THRESHOLD).tolist())

    # Never declare more than a third of the montage bad; that is a global
    # problem (reference lost), not something interpolation can fix.
    if len(bad) > window.shape[0] // 3:
        ranked = sorted(bad, key=lambda i: -abs(stds[i]))
        bad = set(ranked[: window.shape[0] // 3])
    return sorted(bad)


def repair_channels(window: np.ndarray, bad: list[int]) -> np.ndarray:
    """Replace each bad channel with the mean of its healthy neighbours."""
    window = np.array(window, dtype=np.float64, copy=True)
    if not bad:
        return window

    good = [i for i in range(window.shape[0]) if i not in set(bad)]
    if not good:
        return window

    good_mean = window[good].mean(axis=0)
    for idx in bad:
        neighbours = [j for j in (idx - 1, idx + 1) if j in good]
        window[idx] = window[neighbours].mean(axis=0) if neighbours else good_mean
    return window


# --------------------------------------------------------------------------
# 2 + 3. Drift detection and online adaptation
# --------------------------------------------------------------------------
def normalized_entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy scaled to [0, 1]; 1.0 means a completely unsure model."""
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(N_CLASSES))


class SelfHealingController:
    """Owns the drift window, the adaptation buffer and the healing log."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        window: int = CONFIDENCE_WINDOW,
        enabled: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.enabled = enabled
        self.confidences: deque[float] = deque(maxlen=window)
        self.entropies: deque[float] = deque(maxlen=window)
        self.buffer: deque[np.ndarray] = deque(maxlen=window)
        self.events: list[HealingEvent] = []
        self.last_heal_time = 0.0
        self.adaptation_count = 0
        self.repair_count = 0
        self.baseline_state = copy.deepcopy(model.state_dict())

    # -- status -----------------------------------------------------------
    @property
    def mean_confidence(self) -> float:
        return float(np.mean(self.confidences)) if self.confidences else 1.0

    @property
    def mean_entropy(self) -> float:
        return float(np.mean(self.entropies)) if self.entropies else 0.0

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mean_confidence": round(self.mean_confidence, 3),
            "mean_entropy": round(self.mean_entropy, 3),
            "window_filled": len(self.confidences),
            "window_size": self.confidences.maxlen,
            "drift_detected": self.is_drifting(),
            "channels_repaired": self.repair_count,
            "adaptations": self.adaptation_count,
        }

    # -- stage 1 ----------------------------------------------------------
    def preprocess_window(self, window: np.ndarray) -> np.ndarray:
        """Detect and repair bad channels before the window reaches the model."""
        if not self.enabled:
            return window
        bad = detect_bad_channels(window)
        if not bad:
            return window
        repaired = repair_channels(window, bad)
        self.repair_count += len(bad)
        self._log(
            "signal_repair",
            f"Repaired {len(bad)} channel(s) by neighbour interpolation",
            {"channels": bad},
        )
        return repaired

    # -- stage 2 ----------------------------------------------------------
    def observe(self, probabilities: np.ndarray, window: np.ndarray | None = None) -> None:
        """Record one prediction so drift can be judged over time."""
        probabilities = np.asarray(probabilities, dtype=np.float64)
        self.confidences.append(float(probabilities.max()))
        self.entropies.append(normalized_entropy(probabilities))
        if window is not None:
            self.buffer.append(np.asarray(window, dtype=np.float32))

    def is_drifting(self) -> bool:
        if len(self.confidences) < max(5, (self.confidences.maxlen or 20) // 2):
            return False
        return (
            self.mean_confidence < CONFIDENCE_DRIFT_THRESHOLD
            or self.mean_entropy > ENTROPY_DRIFT_THRESHOLD
        )

    # -- stage 3 ----------------------------------------------------------
    def maybe_adapt(self, to_tensor) -> bool:
        """
        Fine-tune the classifier head if drift is confirmed.

        `to_tensor` converts a raw (C, T) window into the model's input
        tensor - injected so this module never imports the preprocessing
        pipeline and stays unit-testable.
        Returns True when an adaptation actually ran.
        """
        if not self.enabled or not self.is_drifting():
            return False
        if time.time() - self.last_heal_time < MIN_SECONDS_BETWEEN_HEALS:
            return False
        if len(self.buffer) < 5:
            return False

        before = self.mean_confidence
        batch = np.concatenate([to_tensor(w) for w in self.buffer], axis=0)
        tensor = torch.from_numpy(batch).to(self.device)

        # Freeze everything, then unfreeze only the dense head.
        trainable = []
        for name, param in self.model.named_parameters():
            is_head = name.startswith("fc1") or name.startswith("fc2")
            param.requires_grad_(is_head)
            if is_head:
                trainable.append(param)

        optimizer = torch.optim.Adam(trainable, lr=ADAPTATION_LEARNING_RATE)
        self.model.train()
        for _ in range(ADAPTATION_STEPS):
            optimizer.zero_grad()
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            # Entropy minimisation: unlabelled, so push the decoder towards
            # confident decisions while a diversity term stops it collapsing
            # onto a single class.
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(1).mean()
            mean_probs = probs.mean(0)
            diversity = (mean_probs * torch.log(mean_probs.clamp_min(1e-12))).sum()
            (entropy + diversity).backward()
            optimizer.step()
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad_(True)

        self.last_heal_time = time.time()
        self.adaptation_count += 1
        self.confidences.clear()
        self.entropies.clear()
        self._log(
            "model_adaptation",
            f"Adapted classifier head over {len(self.buffer)} recent windows",
            {"steps": ADAPTATION_STEPS, "confidence_before": round(before, 3)},
        )
        return True

    # -- recovery ---------------------------------------------------------
    def reset_to_baseline(self) -> None:
        """Roll back to the trained checkpoint if adaptation made things worse."""
        self.model.load_state_dict(self.baseline_state)
        self.model.eval()
        self.confidences.clear()
        self.entropies.clear()
        self._log("model_adaptation", "Rolled back to the baseline checkpoint", {})

    def _log(self, kind: str, detail: str, metrics: dict[str, Any]) -> None:
        self.events.append(HealingEvent(time.time(), kind, detail, metrics))
        self.events = self.events[-100:]
        logger.info("[self-healing] %s: %s", kind, detail)
