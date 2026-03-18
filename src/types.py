from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ImageSample:
    name: str
    image_path: Path
    label_path: Path


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
    use_edge_control: bool
    edge_control_weight: float
    seg_control_weight: float


@dataclass(frozen=True)
class DatasetConfig:
    @dataclass(frozen=True)
    class SegmentationConfig:
        encoding: str = "rgb"
        convert_ids_to_rgb_for_cosmos: bool = True
        converted_cache_dir: str = ".cosmos_seg_rgb_cache"

    root: Path
    original_dir: str
    image_subdir: str
    label_subdir: str
    image_ext: str
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)


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
class GlobalConfig:
    cosmos: CosmosConfig
    dataset: DatasetConfig
    augmentations: list[AugmentationConfig]
    logging: LoggingConfig = field(default_factory=LoggingConfig)
