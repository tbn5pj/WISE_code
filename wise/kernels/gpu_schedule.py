"""Direct GPU-only conversion of discovered B32 block IDs to reusable CSR.

Input counts/indices are *already GPU-resident* post-discovery tensors.
No scalar transfer, nonzero/dynamic-shape operation, CPU-side loop over rows,
or synchronization occurs in this construction path. The padded CSR column
buffer is sized from public tensor shapes; only its active prefix is written.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_b32_ids(COUNTS, IDS, ROWPTR, COLIDX,
                  MAX_KEYS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    count = tl.load(COUNTS + row)
    begin = tl.load(ROWPTR + row, mask=row > 0, other=0)
    if row == 0:
        tl.store(ROWPTR, 0)
    slots = tl.arange(0, BLOCK)
    ids = tl.load(IDS + row * MAX_KEYS + slots, mask=slots < count, other=0)
    tl.store(COLIDX + begin + slots, ids, mask=slots < count)


def from_gpu_block_indices(counts: torch.Tensor, indices: torch.Tensor):
    """Return four `(rowptr, shared_colidx)` GPU CSR schedules for exact B32.

    counts: [L,H,M] int32 CUDA; indices: [L,H,M,C] sorted int32 CUDA.
    The active CSR prefix length is rowptr[-1], but never read on the host.
    All four layers share one padded column tensor; row pointers use global
    offsets into it. This is accepted unchanged by frozen `sparse_attention`.
    """
    if not counts.is_cuda or not indices.is_cuda or counts.device != indices.device:
        raise ValueError("counts and indices must be on the same CUDA device")
    if counts.dtype != torch.int32 or indices.dtype != torch.int32:
        raise ValueError("counts and indices must be int32")
    if counts.ndim != 3 or indices.ndim != 4 or indices.shape[:3] != counts.shape:
        raise ValueError("expected counts [L,H,M], indices [L,H,M,C]")
    if not counts.is_contiguous() or not indices.is_contiguous():
        raise ValueError("GPU discovery tensors must be contiguous")
    layers, heads, query_blocks = counts.shape
    capacity = indices.shape[-1]
    if capacity < query_blocks:
        raise ValueError("index capacity smaller than the number of key blocks")
    rows_per_layer = heads * query_blocks
    total_rows = layers * rows_per_layer
    ptr = torch.empty(total_rows + 1, device=counts.device, dtype=torch.int32)
    col = torch.empty(total_rows * capacity, device=counts.device, dtype=torch.int32)
    torch.cumsum(counts.view(-1), dim=0, dtype=torch.int32, out=ptr[1:])
    _pack_b32_ids[(total_rows,)](
        counts, indices, ptr, col, capacity, triton.next_power_of_2(capacity),
        num_warps=4,
    )
    return [(ptr[layer * rows_per_layer:(layer + 1) * rows_per_layer + 1], col)
            for layer in range(layers)]
