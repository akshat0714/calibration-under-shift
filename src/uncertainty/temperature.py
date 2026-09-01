"""Post-hoc scalar and vector logit scaling fit on calibration data only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def _assert_calibration_split(split) -> None:
    values = {str(value) for value in np.atleast_1d(split)}
    if values != {"calibration"}:
        raise ValueError(
            f"post-hoc scaling requires calibration split only, received {sorted(values)}"
        )


class TemperatureScaler(nn.Module):
    """A positive scalar temperature; argmax predictions are unchanged."""

    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=1e-4, max=1e4)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


class VectorScaler(nn.Module):
    """Per-class positive scales and biases; more flexible but less data-efficient."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_scale.exp().clamp(min=1e-4, max=1e4) + self.bias


@dataclass(frozen=True)
class ScalingFit:
    model: nn.Module
    before_nll: float
    after_nll: float
    iterations: int


def fit_scaler(
    logits,
    labels,
    split,
    method: str = "temperature",
    max_iter: int = 100,
) -> ScalingFit:
    """Minimize calibration-set NLL with LBFGS and return frozen parameters."""

    _assert_calibration_split(split)
    tensor_logits = torch.as_tensor(logits, dtype=torch.float32).detach()
    tensor_labels = torch.as_tensor(labels, dtype=torch.long).detach()
    if tensor_logits.ndim != 2 or tensor_labels.shape != tensor_logits.shape[:1]:
        raise ValueError("logits must be NxC and labels length N")
    if method == "temperature":
        scaler: nn.Module = TemperatureScaler()
    elif method == "vector":
        scaler = VectorScaler(tensor_logits.shape[1])
    else:
        raise ValueError("method must be 'temperature' or 'vector'")
    criterion = nn.CrossEntropyLoss()
    before = float(criterion(tensor_logits, tensor_labels).item())
    optimizer = torch.optim.LBFGS(
        scaler.parameters(), lr=0.05, max_iter=max_iter, line_search_fn="strong_wolfe"
    )
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad()
        loss = criterion(scaler(tensor_logits), tensor_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    scaler.eval()
    for parameter in scaler.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        after = float(criterion(scaler(tensor_logits), tensor_labels).item())
    return ScalingFit(model=scaler, before_nll=before, after_nll=after, iterations=calls)


def scaled_probabilities(scaler: nn.Module, logits) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.as_tensor(logits, dtype=torch.float32)
        return torch.softmax(scaler(tensor), dim=1).cpu().numpy()
