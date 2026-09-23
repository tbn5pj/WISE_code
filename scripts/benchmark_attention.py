"""Attention-only 20-reuse benchmark, not model-forward timing.

Requires a user-provided verified saved fixture (not bundled): torch file
containing q/k/v [N,55,96] BF16 and layout [4,55,M,M] bool. Do not report
this benchmark as a reproduction unless the fixture matches a locked manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from wise.kernels.block_sparse_attention import schedule_from_layout, sparse_attention


def timed(fn) -> float:
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--fixture-sha256", default=None, help="expected SHA-256 for a locked saved fixture")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA GPU required")
    torch.cuda.set_device(args.gpu)
    if args.fixture_sha256 is not None:
        digest = hashlib.sha256(args.fixture.read_bytes()).hexdigest()
        if digest != args.fixture_sha256:
            raise ValueError("saved fixture SHA-256 mismatch")
    data = torch.load(args.fixture, map_location="cpu", weights_only=True)
    q, k, v = (data[name].cuda() for name in ("q", "k", "v"))
    layout = data["layout"].cuda()
    if (q.shape != k.shape or q.shape != v.shape or q.shape[1:] != (55, 96)
            or layout.shape != (4, 55, math.ceil(q.shape[0] / 32), math.ceil(q.shape[0] / 32))):
        raise ValueError("expected real Huginn 4-layer, 55-head, d96 fixture")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("expected BF16 Q/K/V")
    out = torch.empty_like(q)
    prepared = schedule_from_layout(layout, validate=True)

    def dense():
        for _ in range(20):
            for _layer in range(4):
                F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                               v.transpose(0, 1)[None], is_causal=True)

    def sparse(setup: bool):
        csr, indices, extra = schedule_from_layout(layout) if setup else prepared
        for _ in range(20):
            for layer in range(4):
                sparse_attention(q, k, v, out, csr, indices, extra, layer)

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.inference_mode():
        for _ in range(4):
            dense(); sparse(False); sparse(True)
        rows = []
        for p in range(args.passes):
            order = ("dense", "prepared", "setup_inclusive") if p % 2 == 0 else ("setup_inclusive", "prepared", "dense")
            for name in order:
                rows.append({"pass": p, "backend": name,
                             "milliseconds": timed(dense if name == "dense" else lambda n=name: sparse(n == "setup_inclusive")),
                             "sequence_length": q.shape[0], "density": float(layout.sum()) /
                             (4 * 55 * layout.shape[-1] * (layout.shape[-1] + 1) / 2)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote raw attention-only passes to {args.output}; verify uncontended GPU separately")


if __name__ == "__main__":
    main()
