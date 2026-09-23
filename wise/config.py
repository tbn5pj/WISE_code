"""Paper configuration. Override only for an explicitly labeled ablation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WISEConfig:
    recurrent_depth: int = 32
    discovery_depth: int = 12
    block_size: int = 32
    mass_threshold: float = 0.95
    discovery_steps: tuple[int, ...] = (9, 10, 11, 12)

    def __post_init__(self) -> None:
        if not (1 <= self.discovery_depth < self.recurrent_depth):
            raise ValueError("discovery depth must precede the final recurrent step")
        if self.discovery_steps != tuple(range(self.discovery_depth - 3, self.discovery_depth + 1)):
            raise ValueError("the paper uses the four steps ending at discovery depth")
        if self.block_size <= 0 or not (0 < self.mass_threshold <= 1):
            raise ValueError("invalid block size or mass threshold")


PAPER_CONFIG = WISEConfig()
