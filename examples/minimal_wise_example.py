"""CPU-only illustration: native block discovery and temporal union."""

import torch

from wise.routing import WISEDiscovery


def main():
    length = 64
    scores = torch.randn(1, length, length)
    scores = scores.masked_fill(~torch.ones(length, length, dtype=torch.bool).tril(), -torch.inf)
    probabilities = scores.softmax(-1)
    discovery = WISEDiscovery()
    for step in (9, 10, 11, 12):
        discovery.observe(step, probabilities.unsqueeze(0))  # [layer, head, query, key]
    assert discovery.mask_for_step(12) is None
    assert torch.equal(discovery.mask_for_step(13), discovery.mask_for_step(32))
    print("Active B32 blocks:", int(discovery.support.sum()))


if __name__ == "__main__":
    main()
