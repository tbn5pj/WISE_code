"""Causal controls, all branched after unrestricted paper step 12."""

from __future__ import annotations

import torch

from .config import PAPER_CONFIG
from .support import mean_block_mass, stable_prefix_support, token_support


def recurrence_one_controls(step_one_probabilities: torch.Tensor, wise_layout: torch.Tensor, block_size: int = 32, eta: float = 0.95) -> dict[str, torch.Tensor]:
    """Static-95, MassMatched and SizeMatched from the step-1 stable ranking.

    `wise_layout` supplies only a per-unit mass target or cardinality; it never
    supplies the composition of any recurrence-1 control's support.
    """
    aggregate = mean_block_mass(step_one_probabilities.float(), block_size)
    if aggregate.shape != wise_layout.shape:
        raise ValueError("WISE and recurrence-1 routing units must match")
    total = aggregate.sum(-1)
    static = stable_prefix_support(aggregate, eta * total)
    mass_target = (aggregate * wise_layout).sum(-1)
    massmatched = stable_prefix_support(aggregate, mass_target)
    blocks = aggregate.shape[-1]
    causal = torch.ones(blocks, blocks, device=aggregate.device, dtype=torch.bool).tril()
    _, ranked = torch.sort(aggregate.masked_fill(~causal, -torch.inf), dim=-1, descending=True, stable=True)
    count = wise_layout.sum(-1)
    chosen = torch.arange(blocks, device=aggregate.device) < count.unsqueeze(-1)
    sizematched = torch.zeros_like(wise_layout)
    sizematched.scatter_(-1, ranked, chosen)
    sizematched &= causal
    if not torch.equal(sizematched.sum(-1), count):
        raise RuntimeError("SizeMatched did not match local WISE cardinalities")
    return {"Static-95": static, "MassMatched": massmatched, "SizeMatched": sizematched}


def freeze_a_weights(step12_postsoftmax: torch.Tensor, wise_layout: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """A-bar(12): restrict native post-softmax A(12) to final WISE support."""
    mask = token_support(wise_layout, step12_postsoftmax.shape[-1], block_size)
    restricted = step12_postsoftmax.float() * mask
    denominator = restricted.sum(-1, keepdim=True)
    if not bool(torch.all(denominator > 0)):
        raise ValueError("step-12 restriction produced an empty probability row")
    return restricted / denominator


def freeze_a_output(frozen_weights: torch.Tensor, current_values: torch.Tensor) -> torch.Tensor:
    """Use current V(t), never cached V(12) or O(12)."""
    return torch.matmul(frozen_weights.float(), current_values.float())


def masked_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Small explicit reference; q/k/v are [heads, tokens, dimension]."""
    import math
    allowed = token_support(layout, q.shape[-2], block_size)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    return torch.matmul(torch.softmax(scores.masked_fill(~allowed, -torch.inf), -1), v.float())


def truncate_recurrence(step_fn, initial_state, exit_fn, depth: int = PAPER_CONFIG.discovery_depth):
    """Execute paper steps 1..12, then the unchanged native exit path."""
    state = initial_state
    for paper_step in range(1, depth + 1):
        state = step_fn(state, paper_step - 1)
    return exit_fn(state)
