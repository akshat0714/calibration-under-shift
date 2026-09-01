"""Monte Carlo dropout inference without updating batch-normalization statistics."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.uncertainty.ensembles import ensemble_statistics


def enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers while leaving all other modules in evaluation mode."""

    model.eval()
    for module in model.modules():
        if isinstance(module, nn.modules.dropout._DropoutNd):
            module.train()


@torch.inference_mode()
def mc_dropout_predict(
    model: nn.Module,
    inputs: torch.Tensor,
    passes: int = 30,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    if passes < 2:
        raise ValueError("at least two MC-dropout passes are required")
    enable_dropout(model)
    outputs = []
    devices = [inputs.device] if inputs.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if inputs.is_cuda:
            torch.cuda.manual_seed_all(seed)
        for _ in range(passes):
            outputs.append(model(inputs).detach().cpu().numpy())
    model.eval()
    return ensemble_statistics(np.stack(outputs, axis=0))
