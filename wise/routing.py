"""Discovery state: only block support is frozen, never recurrent content."""

from __future__ import annotations

import torch

from .config import WISEConfig
from .support import direct_block_support, temporal_union


class WISEDiscovery:
    def __init__(self, config: WISEConfig | None = None):
        self.config = config or WISEConfig()
        self._steps: dict[int, torch.Tensor] = {}
        self._support: torch.Tensor | None = None

    def observe(self, paper_step: int, postsoftmax_by_layer: torch.Tensor) -> None:
        """Input [layer, head, query, key] from unrestricted Full attention."""
        if paper_step not in self.config.discovery_steps:
            raise ValueError("only paper discovery steps are observed")
        if self._support is not None or paper_step in self._steps:
            raise RuntimeError("support is frozen or step was observed twice")
        layouts = [direct_block_support(p, self.config.block_size, self.config.mass_threshold)
                   for p in postsoftmax_by_layer]
        self._steps[paper_step] = torch.stack(layouts)
        if paper_step == self.config.discovery_depth:
            self._support = temporal_union(self._steps, self.config.discovery_steps)

    @property
    def support(self) -> torch.Tensor:
        if self._support is None:
            raise RuntimeError("discovery has not completed")
        return self._support

    def mask_for_step(self, paper_step: int) -> torch.Tensor | None:
        """Full through 12; exact fixed support for 13..32."""
        if paper_step <= self.config.discovery_depth:
            return None
        if paper_step > self.config.recurrent_depth:
            raise ValueError("step exceeds recurrent depth")
        return self.support
