from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ControlName = Literal["seg", "depth", "edge"]
ControlMode = Literal["disabled", "external", "on_the_fly"]
CONTROL_NAMES: tuple[ControlName, ...] = ("seg", "depth", "edge")


@dataclass(frozen=True)
class ControlConfig:
    mode: ControlMode
    weight: float
    subdir: str

    @property
    def is_disabled(self) -> bool:
        return self.mode == "disabled"

    @property
    def is_external(self) -> bool:
        return self.mode == "external"

    @property
    def is_on_the_fly(self) -> bool:
        return self.mode == "on_the_fly"


@dataclass(frozen=True)
class CosmosControls:
    seg: ControlConfig
    depth: ControlConfig
    edge: ControlConfig

    def as_dict(self) -> dict[ControlName, ControlConfig]:
        return {
            "seg": self.seg,
            "depth": self.depth,
            "edge": self.edge,
        }


@dataclass(frozen=True)
class CosmosConfig:
    repo_root: Path
    disable_guardrails: bool
    resolution: str
    guidance: float
    num_steps: int
    max_frames: int
    num_video_frames_per_chunk: int
    model: str | None
    model_variant: str
    model_distilled: bool
    controls: CosmosControls


@dataclass(frozen=True)
class DatasetConfig:
    input_root: Path
    output_root: Path
    original_dir: str
    image_subdir: str
    image_ext: str


@dataclass(frozen=True)
class AugmentationConfig:
    name: str
    output_dir: str
    fraction: float
    seed_base: int
    prompt: str
    negative_prompt: str


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    file_path: Path | None = None


@dataclass(frozen=True)
class ImageSample:
    name: str
    image_path: Path
    control_paths: dict[ControlName, Path | None]


@dataclass(frozen=True)
class AugmentationJob:
    augmentation_name: str
    output_dir_name: str
    request_name: str
    seed: int
    image_name: str
    image_path: Path
    control_paths: dict[ControlName, Path | None]
    prompt: str
    negative_prompt: str


@dataclass(frozen=True)
class GlobalConfig:
    cosmos: CosmosConfig
    dataset: DatasetConfig
    augmentations: list[AugmentationConfig]
    logging: LoggingConfig = field(default_factory=LoggingConfig)
