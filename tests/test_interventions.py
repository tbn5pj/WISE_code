import torch

from wise.interventions import freeze_a_output, freeze_a_weights, masked_attention, recurrence_one_controls, truncate_recurrence
from wise.support import direct_block_support


def test_static_controls_and_exact_sizematched_cardinality():
    n = 8
    p = torch.ones(2, n, n).tril().float()
    p /= p.sum(-1, keepdim=True)
    wise = direct_block_support(p, block_size=2)
    controls = recurrence_one_controls(p, wise, block_size=2)
    assert torch.equal(controls["SizeMatched"].sum(-1), wise.sum(-1))
    for layout in controls.values():
        assert not bool((layout & ~torch.ones(4, 4, dtype=torch.bool).tril()).any())


def test_freeze_a_support_and_dynamic_values():
    p = torch.ones(1, 6, 6).tril().float()
    p /= p.sum(-1, keepdim=True)
    layout = direct_block_support(p, block_size=2)
    frozen = freeze_a_weights(p, layout, block_size=2)
    assert torch.allclose(frozen.sum(-1), torch.ones_like(frozen.sum(-1)))
    v12 = torch.randn(1, 6, 4)
    v13 = v12 + 1
    assert not torch.allclose(freeze_a_output(frozen, v12), freeze_a_output(frozen, v13))


def test_dynamic_qkv_inside_fixed_wise_support():
    n, d = 4, 8
    p = torch.ones(1, n, n).tril().float()
    p /= p.sum(-1, keepdim=True)
    layout = direct_block_support(p, block_size=2)
    q, k, v = torch.randn(1, n, d), torch.randn(1, n, d), torch.randn(1, n, d)
    first = masked_attention(q, k, v, layout, block_size=2)
    later = masked_attention(q + 0.4, k - 0.2, v + 0.7, layout, block_size=2)
    assert not torch.allclose(first, later)


def test_truncate_executes_step12_and_native_exit_only():
    called = []

    def step(state, index):
        called.append(index)
        return state + 1

    def exit_path(state):
        called.append("native_exit")
        return state * 3

    assert truncate_recurrence(step, 0, exit_path) == 36
    assert called == list(range(12)) + ["native_exit"]
