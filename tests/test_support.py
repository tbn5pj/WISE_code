import torch

from wise.routing import WISEDiscovery
from wise.support import block_density, direct_block_support, temporal_union, token_support


def probabilities(n=8):
    scores = torch.arange(n, dtype=torch.float32)[None, None, :].expand(2, n, n).clone()
    scores.masked_fill_(~torch.ones(n, n, dtype=torch.bool).tril(), -torch.inf)
    return scores.softmax(-1)


def test_direct_cumulative_block_mass_and_causality():
    p = probabilities()
    layout = direct_block_support(p, block_size=2, eta=0.95)
    assert layout.shape == (2, 4, 4)
    assert not bool((layout & ~torch.ones(4, 4, dtype=torch.bool).tril()).any())
    assert bool(layout.any(-1).all())
    assert torch.equal(layout[0, 0], torch.tensor([True, False, False, False]))
    assert block_density(layout) > 0
    mask = token_support(layout, 8, 2)
    assert not bool((mask & ~torch.ones(8, 8, dtype=torch.bool).tril()).any())


def test_temporal_union_and_frozen_timing():
    discovery = WISEDiscovery()
    p = probabilities(64).unsqueeze(0)
    layouts = {}
    for t in (9, 10, 11, 12):
        discovery.observe(t, p)
        layouts[t] = direct_block_support(p[0])
    assert torch.equal(discovery.support, temporal_union({t: x.unsqueeze(0) for t, x in layouts.items()}))
    assert discovery.mask_for_step(12) is None
    assert torch.equal(discovery.mask_for_step(13), discovery.mask_for_step(32))
    assert discovery.support.data_ptr() == discovery.mask_for_step(32).data_ptr()


def test_partial_final_block():
    p = probabilities(5)
    layout = direct_block_support(p, block_size=2)
    assert layout.shape == (2, 3, 3)
    assert token_support(layout, 5, 2).shape == (2, 5, 5)
