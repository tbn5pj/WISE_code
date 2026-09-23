"""Diagnostic token support is intentionally distinct from deployed B32 support."""

from __future__ import annotations

import numpy as np
import torch


def causal_top16(probabilities: torch.Tensor) -> torch.Tensor:
    """Per-query Top-min(16, i+1) causal attended-key set."""
    n = probabilities.shape[-1]
    if probabilities.shape[-2] != n:
        raise ValueError("expected square attention probabilities")
    output = torch.zeros_like(probabilities, dtype=torch.bool)
    # Match the historical per-query topk call exactly, including tie behavior.
    for i in range(n):
        indices = torch.topk(probabilities[..., i, :i + 1], k=min(16, i + 1), sorted=True).indices
        output[..., i, :i + 1].scatter_(-1, indices, True)
    return output


def jaccard(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    return (previous & current).sum(-1).float() / (previous | current).sum(-1).clamp_min(1)


def first_three_transition_run(values, *, threshold: float, higher: bool, first_step: int = 2, final_step: int = 32) -> int:
    """First transition in a three-transition run; 33 means censored at T32."""
    series = np.asarray(values, dtype=np.float64)
    condition = series >= threshold if higher else series < threshold
    for index in range(first_step - 1, min(len(series), final_step) - 2):
        if bool(condition[index:index + 3].all()):
            return index + 1
    return final_step + 1


def representation_threshold(distances) -> float:
    """0.05 times median first four finite distances; historical fallback."""
    values = np.asarray(distances, dtype=np.float64)
    finite = values[np.isfinite(values)]
    early = finite[:4]
    scale = float(np.median(early)) if len(early) else float("nan")
    if not np.isfinite(scale) or scale <= 0:
        scale = float(finite.max()) if len(finite) else 1.0
    return 0.05 * max(scale, 2 ** -52)
