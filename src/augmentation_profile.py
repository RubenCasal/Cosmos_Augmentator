from __future__ import annotations

import random
from dataclasses import dataclass

from .types import AugmentationConfig, ImageSample


@dataclass(frozen=True)
class AugmentationProfile:
    name: str
    output_dir_name: str
    fraction: float
    prompt: str
    negative_prompt: str
    seed_base: int

    @classmethod
    def from_config(cls, aug_cfg: AugmentationConfig) -> "AugmentationProfile":
        return cls(
            name=aug_cfg.name,
            output_dir_name=aug_cfg.output_dir,
            fraction=aug_cfg.fraction,
            prompt=aug_cfg.prompt,
            negative_prompt=aug_cfg.negative_prompt,
            seed_base=aug_cfg.seed_base,
        )

    def select_samples(self, all_samples: list[ImageSample]) -> list[ImageSample]:
        if not all_samples or self.fraction <= 0.0:
            return []

        k = int(self.fraction * len(all_samples))
        if k <= 0:
            return []

        indices = list(range(len(all_samples)))
        rng = random.Random(self.seed_base)
        rng.shuffle(indices)

        selected = [all_samples[i] for i in sorted(indices[:k], key=lambda i: all_samples[i].name)]
        return selected
