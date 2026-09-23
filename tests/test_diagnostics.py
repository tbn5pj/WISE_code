import torch

from wise.diagnostics import causal_top16, first_three_transition_run


def test_diagnostic_top16_is_causal_and_not_block_support():
    p = torch.rand(2, 20, 20)
    support = causal_top16(p)
    expected = torch.arange(1, 21).clamp_max(16).expand(2, -1)
    assert torch.equal(support.sum(-1), expected)
    assert not bool((support & ~torch.ones(20, 20, dtype=torch.bool).tril()).any())


def test_three_transition_persistence():
    values = [float("nan"), 0.85, 0.91, 0.92, 0.93]
    assert first_three_transition_run(values, threshold=0.90, higher=True, first_step=1) == 3
