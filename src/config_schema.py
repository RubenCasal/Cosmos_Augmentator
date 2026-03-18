from __future__ import annotations

from pathlib import Path

import yaml

from .types import (
    AugmentationConfig,
    CosmosConfig,
    DatasetConfig,
    GlobalConfig,
    LoggingConfig,
)


class ConfigError(ValueError):
    pass


def _require_dict(value: object, field_name: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"'{field_name}' must be a mapping.")
    return value


def _require_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{field_name}' must be a non-empty string.")
    return value.strip()


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"'{field_name}' must be a boolean.")
    return value


def _optional_str(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, field_name)


def _optional_bool(value: object, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    return _require_bool(value, field_name)


def _optional_float(value: object, field_name: str, default: float) -> float:
    if value is None:
        return default
    return _require_float(value, field_name)


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{field_name}' must be an integer.")
    return value


def _require_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{field_name}' must be numeric.")
    return float(value)


def _resolve_optional_path(path_value: object, field_name: str, config_path: Path) -> Path | None:
    raw = _optional_str(path_value, field_name)
    if raw is None:
        return None

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return candidate.resolve()


def load_config(path: Path) -> GlobalConfig:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ConfigError("Config root must be a mapping.")

    cosmos_data = _require_dict(data.get("cosmos"), "cosmos")
    dataset_data = _require_dict(data.get("dataset"), "dataset")
    augmentations_data = data.get("augmentations")
    if not isinstance(augmentations_data, list) or not augmentations_data:
        raise ConfigError("'augmentations' must be a non-empty list.")

    cosmos = CosmosConfig(
        repo_root=Path(_require_str(cosmos_data.get("repo_root"), "cosmos.repo_root")).expanduser().resolve(),
        disable_guardrails=_require_bool(cosmos_data.get("disable_guardrails"), "cosmos.disable_guardrails"),
        resolution=_require_str(cosmos_data.get("resolution"), "cosmos.resolution"),
        guidance=_require_float(cosmos_data.get("guidance"), "cosmos.guidance"),
        num_steps=_require_int(cosmos_data.get("num_steps"), "cosmos.num_steps"),
        max_frames=_require_int(cosmos_data.get("max_frames"), "cosmos.max_frames"),
        num_video_frames_per_chunk=_require_int(
            cosmos_data.get("num_video_frames_per_chunk"), "cosmos.num_video_frames_per_chunk"
        ),
        model=_optional_str(cosmos_data.get("model"), "cosmos.model"),
        model_variant=_optional_str(cosmos_data.get("model_variant"), "cosmos.model_variant") or "edge",
        model_distilled=_optional_bool(cosmos_data.get("model_distilled"), "cosmos.model_distilled", False),
        use_edge_control=_optional_bool(cosmos_data.get("use_edge_control"), "cosmos.use_edge_control", True),
        edge_control_weight=_optional_float(cosmos_data.get("edge_control_weight"), "cosmos.edge_control_weight", 1.0),
        seg_control_weight=_optional_float(cosmos_data.get("seg_control_weight"), "cosmos.seg_control_weight", 0.6),
    )
    if not 0.0 <= cosmos.edge_control_weight <= 1.0:
        raise ConfigError(f"cosmos.edge_control_weight must be in [0.0, 1.0], got {cosmos.edge_control_weight}")
    if not 0.0 <= cosmos.seg_control_weight <= 1.0:
        raise ConfigError(f"cosmos.seg_control_weight must be in [0.0, 1.0], got {cosmos.seg_control_weight}")

    segmentation_data = dataset_data.get("segmentation")
    segmentation_cfg = DatasetConfig.SegmentationConfig()
    if segmentation_data is not None:
        seg_map = _require_dict(segmentation_data, "dataset.segmentation")
        encoding = (_optional_str(seg_map.get("encoding"), "dataset.segmentation.encoding") or "rgb").lower()
        if encoding not in {"rgb", "id"}:
            raise ConfigError("dataset.segmentation.encoding must be 'rgb' or 'id'.")

        segmentation_cfg = DatasetConfig.SegmentationConfig(
            encoding=encoding,
            convert_ids_to_rgb_for_cosmos=_optional_bool(
                seg_map.get("convert_ids_to_rgb_for_cosmos"),
                "dataset.segmentation.convert_ids_to_rgb_for_cosmos",
                True,
            ),
            converted_cache_dir=_optional_str(
                seg_map.get("converted_cache_dir"),
                "dataset.segmentation.converted_cache_dir",
            )
            or ".cosmos_seg_rgb_cache",
        )

    dataset = DatasetConfig(
        root=Path(_require_str(dataset_data.get("root"), "dataset.root")).expanduser().resolve(),
        original_dir=_require_str(dataset_data.get("original_dir"), "dataset.original_dir"),
        image_subdir=_require_str(dataset_data.get("image_subdir"), "dataset.image_subdir"),
        label_subdir=_require_str(dataset_data.get("label_subdir"), "dataset.label_subdir"),
        image_ext=_require_str(dataset_data.get("image_ext"), "dataset.image_ext"),
        segmentation=segmentation_cfg,
    )

    if not cosmos.repo_root.is_dir():
        raise ConfigError(f"cosmos.repo_root is not a directory: {cosmos.repo_root}")

    if not dataset.root.is_dir():
        raise ConfigError(f"dataset.root is not a directory: {dataset.root}")

    original_root = dataset.root / dataset.original_dir
    images_dir = original_root / dataset.image_subdir
    labels_dir = original_root / dataset.label_subdir

    if not images_dir.is_dir():
        raise ConfigError(f"Missing images dir: {images_dir}")
    if not labels_dir.is_dir():
        raise ConfigError(f"Missing labels dir: {labels_dir}")

    logging_data = data.get("logging")
    logging_cfg = LoggingConfig()
    if logging_data is not None:
        log_map = _require_dict(logging_data, "logging")
        level = _optional_str(log_map.get("level"), "logging.level") or "INFO"
        file_path = _resolve_optional_path(log_map.get("file"), "logging.file", path)
        logging_cfg = LoggingConfig(level=level.upper(), file_path=file_path)

    names_seen: set[str] = set()
    outputs_seen: set[str] = set()
    augmentations: list[AugmentationConfig] = []

    for idx, raw_aug in enumerate(augmentations_data):
        aug_key = f"augmentations[{idx}]"
        aug_data = _require_dict(raw_aug, aug_key)

        aug = AugmentationConfig(
            name=_require_str(aug_data.get("name"), f"{aug_key}.name"),
            output_dir=_require_str(aug_data.get("output_dir"), f"{aug_key}.output_dir"),
            fraction=_require_float(aug_data.get("fraction"), f"{aug_key}.fraction"),
            seed_base=_require_int(aug_data.get("seed_base"), f"{aug_key}.seed_base"),
            prompt=_require_str(aug_data.get("prompt"), f"{aug_key}.prompt"),
            negative_prompt=_require_str(aug_data.get("negative_prompt"), f"{aug_key}.negative_prompt"),
        )

        if not 0.0 <= aug.fraction <= 1.0:
            raise ConfigError(f"{aug_key}.fraction must be in [0.0, 1.0], got {aug.fraction}")
        if aug.name in names_seen:
            raise ConfigError(f"Duplicate augmentation name: {aug.name}")
        if aug.output_dir in outputs_seen:
            raise ConfigError(f"Duplicate augmentation output_dir: {aug.output_dir}")

        names_seen.add(aug.name)
        outputs_seen.add(aug.output_dir)
        augmentations.append(aug)

    return GlobalConfig(
        cosmos=cosmos,
        dataset=dataset,
        augmentations=augmentations,
        logging=logging_cfg,
    )
