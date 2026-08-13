"""
EEGNet-TCN model definition.

A compact EEGNet-style network: temporal convolution, depthwise spatial
convolution across electrodes, a second temporal convolution, then two
fully-connected layers. Input shape is (batch, 1, channels, timesteps).

The definition lives in exactly one module. The training script and the
Streamlit app both import it, so the checkpoint can never disagree with the
architecture that loads it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import N_CHANNELS, N_CLASSES, TARGET_SAMPLES


class EEGNetTCN(nn.Module):
    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_timesteps: int = TARGET_SAMPLES,
        n_classes: int = N_CLASSES,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.n_classes = n_classes

        # Block 1 - temporal filtering
        self.conv1 = nn.Conv2d(1, 16, (1, 32), padding=(0, 16))
        self.bn1 = nn.BatchNorm2d(16)

        # Block 1 - depthwise spatial filtering across all electrodes
        self.depthwise = nn.Conv2d(16, 32, (n_channels, 1), groups=16)
        self.bn2 = nn.BatchNorm2d(32)

        # Block 2 - separable temporal convolution
        self.conv2 = nn.Conv2d(32, 32, (1, 16), padding=(0, 8))
        self.bn3 = nn.BatchNorm2d(32)

        self.pool1 = nn.AvgPool2d((1, 4))
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.conv_output_size = self._infer_flatten_size()
        self.fc1 = nn.Linear(self.conv_output_size, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def _infer_flatten_size(self) -> int:
        """Run a dry forward pass to size the first dense layer."""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_timesteps)
            x = self.pool1(F.elu(self.bn2(self.depthwise(F.elu(self.bn1(self.conv1(x)))))))
            x = self.pool2(F.elu(self.bn3(self.conv2(x))))
            return int(x.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.depthwise(x)))
        x = self.dropout1(self.pool1(x))

        x = F.elu(self.bn3(self.conv2(x)))
        x = self.dropout2(self.pool2(x))

        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1(x))
        return self.fc2(x)


# Backwards-compatible alias for checkpoints and notebooks written against
# the original class name.
EEGNet_TCN = EEGNetTCN


def resolve_device(preferred: str | None = None) -> torch.device:
    """Pick the best available device: explicit choice, then CUDA, MPS, CPU."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
