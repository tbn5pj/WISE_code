"""Direct block-space WISE support; no token-support rounding."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F


def block_key_mass(probabilities: torch.Tensor, block_size: int) -> torch.Tensor:
    """Sum post-softmax probability in each key block, retaining query rows."""
    n = probabilities.shape[-1]
    pad = (-n) % block_size
    values = F.pad(probabilities, (0, pad)) if pad else probabilities
    return values.reshape(*values.shape[:-1], math.ceil(n / block_size), block_size).sum(-1)


def mean_block_mass(probabilities: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mean [head, query-block, key-block] mass, including a partial final block."""
    if probabilities.ndim != 3 or probabilities.shape[-2] != probabilities.shape[-1]:
        raise ValueError("expected [heads, query tokens, key tokens]")
    heads, n, _ = probabilities.shape
    blocks = math.ceil(n / block_size)
    key_mass = block_key_mass(probabilities, block_size)
    qpad = blocks * block_size - n
    if qpad:
        key_mass = F.pad(key_mass, (0, 0, 0, qpad))
    aggregate = key_mass.reshape(heads, blocks, block_size, blocks).sum(2)
    counts = torch.full((blocks,), block_size, device=aggregate.device, dtype=aggregate.dtype)
    counts[-1] = n - (blocks - 1) * block_size
    return aggregate / counts.view(1, -1, 1)


def stable_prefix_support(mass: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Minimal stable-ranked causal mass prefix for each local routing unit.

    Ties follow ascending key-block index. `target` has shape `mass.shape[:-1]`.
    The selected prefix is nonempty, including for a zero-mass row.
    """
    blocks = mass.shape[-1]
    if mass.shape[-2] != blocks or target.shape != mass.shape[:-1]:
        raise ValueError("expected [..., query blocks, key blocks] and matching target")
    causal = torch.ones(blocks, blocks, device=mass.device, dtype=torch.bool).tril()
    ranked, indices = torch.sort(mass.masked_fill(~causal, -torch.inf), dim=-1, descending=True, stable=True)
    counts = ranked.clamp_min(0).cumsum(-1).lt(target.unsqueeze(-1)).sum(-1).add(1)
    valid = torch.arange(1, blocks + 1, device=mass.device)
    counts = torch.minimum(counts, valid).clamp_min(1)
    selected = torch.arange(blocks, device=mass.device) < counts.unsqueeze(-1)
    layout = torch.zeros_like(mass, dtype=torch.bool)
    layout.scatter_(-1, indices, selected)
    return layout & causal


def direct_block_support(probabilities: torch.Tensor, block_size: int = 32, eta: float = 0.95) -> torch.Tensor:
    """Paper support for one recurrent step, [heads, query blocks, key blocks]."""
    if not (0 < eta <= 1):
        raise ValueError("eta must lie in (0,1]")
    mass = mean_block_mass(probabilities.float(), block_size)
    return stable_prefix_support(mass, eta * mass.sum(-1))


def temporal_union(step_layouts: Mapping[int, torch.Tensor], steps: tuple[int, ...] = (9, 10, 11, 12)) -> torch.Tensor:
    """Freeze precisely the direct supports from paper steps 9--12."""
    if set(step_layouts) != set(steps):
        raise ValueError("step layouts must contain exactly the discovery window")
    layouts = [step_layouts[t] for t in steps]
    if any(layout.shape != layouts[0].shape for layout in layouts):
        raise ValueError("inconsistent layout shapes")
    result = layouts[0].clone()
    for layout in layouts[1:]:
        result |= layout
    return result


def token_support(layout: torch.Tensor, sequence_length: int, block_size: int = 32) -> torch.Tensor:
    """Expand an already-selected block mask only for reference attention."""
    positions = torch.arange(sequence_length, device=layout.device)
    result = layout[..., positions[:, None] // block_size, positions[None, :] // block_size]
    return result & (positions[:, None] >= positions[None, :])


def block_density(layout: torch.Tensor) -> float:
    blocks = layout.shape[-1]
    if layout.shape[-2] != blocks:
        raise ValueError("expected square block layout")
    prefix = math.prod(layout.shape[:-2])
    return float(layout.sum()) / (prefix * blocks * (blocks + 1) / 2)


def retained_future_mass(full_probabilities: torch.Tensor, layout: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Per-head/query-token retained mass from later unrestricted Full attention."""
    n = full_probabilities.shape[-1]
    masses = block_key_mass(full_probabilities, block_size)
    query_blocks = torch.arange(n, device=layout.device) // block_size
    return (masses * layout[..., query_blocks, :]).sum(-1)
