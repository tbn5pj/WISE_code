import hashlib
from pathlib import Path

import pytest
import torch


EXPECTED_WINNER_SHA256 = "d066af2c2d28a34290cd2179c354df8144e578b1099fd521dfeb0872acfa1b46"


def test_validated_kernel_source_unchanged():
    path = Path(__file__).resolve().parents[1] / "wise/kernels/_ampere_winner.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_WINNER_SHA256


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/Triton unavailable")
def test_exact_sparse_kernel_against_masked_dense(monkeypatch):
    triton = pytest.importorskip("triton")
    if tuple(int(part) for part in triton.__version__.split(".")[:2]) < (3, 8):
        pytest.skip("frozen kernel requires the validated Triton 3.8 stack")
    from wise.interventions import masked_attention
    from wise.kernels.block_sparse_attention import schedule_from_layout, sparse_attention

    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("validated kernel targets NVIDIA Ampere or newer")
    torch.manual_seed(7)
    n, h, d = 64, 55, 96
    q, k, v = (torch.randn(n, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    layout = torch.zeros(4, h, 2, 2, device="cuda", dtype=torch.bool)
    layout[..., 0, 0] = True
    layout[..., 1, 0] = True
    layout[0, :h // 2, 1, 1] = True
    csr, indices, extra = schedule_from_layout(layout, validate=True)
    out = torch.empty_like(q)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dense SDPA fallback was called")

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", forbidden)
    result = sparse_attention(q, k, v, out, csr, indices, extra, layer=0)
    assert result.data_ptr() == out.data_ptr()
    assert bool(torch.isfinite(result).all())
    for head in (0, h // 2, h - 1):
        reference = masked_attention(q[:, head].unsqueeze(0), k[:, head].unsqueeze(0),
                                     v[:, head].unsqueeze(0), layout[0, head].unsqueeze(0))
        error = (out[:, head].float() - reference[0]).abs()
        assert float(error.max()) <= 0.03
        assert float(error.mean()) <= 0.002
