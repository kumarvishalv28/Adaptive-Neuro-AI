"""
BCI inference engine.

Owns the trained model, the self-healing controller and the simulated
signal source. Deliberately free of Streamlit imports so it can be driven
from tests, a CLI or any other front end.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    CLASS_NAMES,
    EPOCH_DURATION,
    MODEL_PATH,
    N_CHANNELS,
    TARGET_SAMPLES,
    TARGET_SAMPLING_RATE,
)
from .model import EEGNetTCN, resolve_device
from .preprocessing import load_edf_epochs, to_model_input
from .self_healing import SelfHealingController

logger = logging.getLogger(__name__)

SIMULATED_DEVICES = {
    "OpenBCI Cyton": {"channels": 8, "sampling_rate": 250, "battery": 85},
    "Muse 2": {"channels": 4, "sampling_rate": 256, "battery": 72},
    "NeuroSky MindWave": {"channels": 1, "sampling_rate": 512, "battery": 90},
    "PhysioNet 64-channel montage": {"channels": 64, "sampling_rate": 160, "battery": 100},
}


@dataclass
class Prediction:
    timestamp: float
    label: str
    class_index: int
    probabilities: np.ndarray
    confidence: float
    source: str
    healed: bool = False


class BCIEngine:
    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        device: str | None = None,
        self_healing: bool = True,
    ) -> None:
        self.device = resolve_device(device)
        self.model_path = Path(model_path)
        self.model_loaded = False
        self.checkpoint_accuracy: float | None = None
        self.load_message = ""
        self.model = self._load_model()
        self.healer = SelfHealingController(self.model, self.device, enabled=self_healing)

        self.current_prediction = "REST"
        self.probabilities = np.full(len(CLASS_NAMES), 1 / len(CLASS_NAMES))
        self.history: list[Prediction] = []
        self.connected_device: dict[str, Any] | None = None
        self.noise_level = 0.2       # raised by the UI to test resilience

    # ---------------------------------------------------------------- model
    def _load_model(self) -> EEGNetTCN:
        model = EEGNetTCN()
        if self.model_path.exists():
            try:
                checkpoint = torch.load(
                    self.model_path, map_location=self.device, weights_only=False
                )
                state = checkpoint.get("model_state_dict", checkpoint)
                model.load_state_dict(state)
                self.checkpoint_accuracy = checkpoint.get("val_accuracy")
                self.model_loaded = True
                self.load_message = (
                    f"Model loaded from {self.model_path.name}"
                    + (f" (validation accuracy {self.checkpoint_accuracy:.2f}%)"
                       if isinstance(self.checkpoint_accuracy, (int, float)) else "")
                )
            except Exception as exc:                     # noqa: BLE001
                self.load_message = (
                    f"Checkpoint at {self.model_path} could not be loaded ({exc}). "
                    "Running with untrained weights - predictions are meaningless."
                )
                logger.error(self.load_message)
        else:
            self.load_message = (
                f"No checkpoint at {self.model_path}. Run "
                "`python -m src.train_pytorch_real` first. The interface still "
                "runs, but with untrained weights."
            )
            logger.warning(self.load_message)

        model.to(self.device)
        model.eval()
        return model

    # ------------------------------------------------------------ inference
    def predict_window(self, window: np.ndarray, source: str = "SIMULATED") -> Prediction:
        """Heal, classify and record a single (channels, timesteps) window."""
        healed_window = self.healer.preprocess_window(window)
        was_repaired = not np.array_equal(healed_window, window)

        tensor = torch.from_numpy(to_model_input(healed_window)).to(self.device)
        with torch.no_grad():
            probabilities = F.softmax(self.model(tensor), dim=1).cpu().numpy()[0]

        index = int(np.argmax(probabilities))
        prediction = Prediction(
            timestamp=time.time(),
            label=CLASS_NAMES[index],
            class_index=index,
            probabilities=probabilities,
            confidence=float(probabilities[index]),
            source=source,
            healed=was_repaired,
        )

        self.current_prediction = prediction.label
        self.probabilities = probabilities
        self.healer.observe(probabilities, healed_window)
        if self.healer.maybe_adapt(to_model_input):
            prediction.healed = True

        self.history.append(prediction)
        self.history = self.history[-100:]
        return prediction

    def analyse_edf(self, edf_path: str | Path) -> dict[str, Any]:
        """Classify every epoch of an EDF recording."""
        data, labels, event_id = load_edf_epochs(edf_path)
        if data.shape[0] == 0:
            raise ValueError("No annotated events were found in this EDF file.")

        predictions, confidences, probabilities, healed_flags = [], [], [], []
        for epoch in data:
            result = self.predict_window(epoch, source="EDF")
            predictions.append(result.label)
            confidences.append(result.confidence)
            probabilities.append(result.probabilities)
            healed_flags.append(result.healed)

        return {
            "predictions": predictions,
            "confidences": np.array(confidences),
            "probabilities": np.array(probabilities),
            "true_labels": labels,
            "healed_flags": healed_flags,
            "event_id": event_id,
            "n_epochs": int(data.shape[0]),
            "raw_data": data,
        }

    # ------------------------------------------------------------- devices
    def scan_devices(self) -> list[str]:
        """
        Real hardware discovery goes here (BrainFlow / LSL / serial).
        No hardware is present, so the list is honestly empty.
        """
        return []

    def connect_simulated_device(self, name: str) -> tuple[bool, str]:
        spec = SIMULATED_DEVICES.get(name)
        if not spec:
            return False, f"Unknown device: {name}"
        self.connected_device = {"name": f"{name} (simulated)", **spec}
        return True, f"Connected to {name} in simulation mode."

    def disconnect_device(self) -> None:
        self.connected_device = None

    def device_info(self) -> dict[str, Any]:
        if self.connected_device:
            return {"connected": True, **self.connected_device}
        return {"connected": False, "name": "No device connected",
                "channels": 0, "sampling_rate": 0, "battery": 0}

    # ------------------------------------------------------- signal source
    def simulate_window(self, intent: str | None = None, inject_fault: bool = False) -> np.ndarray:
        """
        Generate a physiologically plausible motor-imagery window.

        Contralateral mu/beta desynchronisation is modelled: imagining the
        LEFT hand suppresses the right-hemisphere channels and vice versa.
        `inject_fault` drops two electrodes and saturates one so the
        self-healing layer can be demonstrated on demand.
        """
        info = self.device_info()
        n_channels = info["channels"] or N_CHANNELS
        sampling_rate = info["sampling_rate"] or TARGET_SAMPLING_RATE
        n_samples = max(64, int(EPOCH_DURATION * sampling_rate))
        t = np.linspace(0, EPOCH_DURATION, n_samples, endpoint=False)

        intent = intent or self.current_prediction
        half = max(1, n_channels // 2)
        data = np.zeros((n_channels, n_samples))

        for channel in range(n_channels):
            right_hemisphere = channel < half
            active = (
                (intent == "LEFT" and right_hemisphere)
                or (intent == "RIGHT" and not right_hemisphere)
            )
            frequency = 12.0 + np.random.random() * 4 if active else 10.0 + np.random.random() * 2
            amplitude = (0.8 + np.random.random() * 0.4) if active else (0.3 + np.random.random() * 0.2)

            wave = amplitude * np.sin(2 * np.pi * frequency * t)
            wave += 0.3 * amplitude * np.sin(2 * np.pi * 2 * frequency * t)
            data[channel] = wave + self.noise_level * np.random.randn(n_samples)

        if inject_fault and n_channels >= 4:
            data[0] = 0.0                      # electrode fell off
            data[1] = 0.0                      # amplifier dead
            data[2] = data[2] * 500.0          # saturated channel
        return data

    def reset(self) -> None:
        self.history.clear()
        self.healer.reset_to_baseline()
