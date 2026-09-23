"""Frozen validated exact-32x32 Ampere kernel interface.

The arithmetic kernel is a byte-for-byte copy of the final validated winner.
Only this wrapper provides a small public API; it never changes support.
"""

from __future__ import annotations

import torch


WINNER_CONFIG = {"warps": 2, "stages": 3, "split_d": True, "order": "head_descending"}


def schedule_from_layout(layout: torch.Tensor, *, validate: bool = False):
    """Prepare reusable GPU CSR and within-head row order from exact B32 layout.

    `layout` is [layer, head, query-block, key-block], causal, bool CUDA.
    Schedule construction is one-time setup, not part of prepared call latency.
    """
    if not layout.is_cuda or layout.dtype != torch.bool or layout.ndim != 4:
        raise ValueError("expected CUDA boolean layout [layer,head,query-block,key-block]")
    layers, heads, blocks, keys = layout.shape
    if blocks != keys:
        raise ValueError("layout must be square")
    if validate:  # Run once outside the timed production path; checks synchronize.
        causal = torch.ones(blocks, blocks, device=layout.device, dtype=torch.bool).tril()
        if bool((layout & ~causal).any()) or not bool(layout.any(-1).all()):
            raise ValueError("layout must be causal and nonempty per query block")
    from . import _ampere_winner as kernel
    from .gpu_schedule import from_gpu_block_indices
    counts = layout.sum(-1, dtype=torch.int32).contiguous()
    # Stable partition puts selected key IDs first, preserving ascending key order.
    indices = torch.argsort((~layout).to(torch.int32), dim=-1, stable=True).to(torch.int32).contiguous()
    csr = from_gpu_block_indices(counts, indices)
    extra = kernel.prepare(counts, indices, order=WINNER_CONFIG["order"], bits=False)
    return csr, indices, extra


def sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
                     schedule, indices: torch.Tensor, extra, layer: int = 0) -> torch.Tensor:
    """One prepared exact-B32 attention call; Q/K/V/out are [N,H,96] BF16."""
    if any(x.device != q.device or x.shape != q.shape for x in (k, v, out)):
        raise ValueError("Q/K/V/output must have identical shapes and device")
    if q.dtype != torch.bfloat16 or q.ndim != 3 or q.shape[-1] != 96:
        raise ValueError("validated kernel requires [N,H,96] BF16")
    if q.device.type != "cuda":
        raise ValueError("the sparse kernel requires CUDA")
    from ._ampere_winner import attention
    return attention(q, k, v, schedule[layer], indices[layer], extra, out, WINNER_CONFIG, layer)
