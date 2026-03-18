from __future__ import annotations

from pathlib import Path

import yaml

from .types import (
    CONTROL_NAMES,
    AugmentationConfig,
    ControlConfig,
    ControlMode,
    CosmosConfig,
    CosmosControls,
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
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _parse_control_mode(value: object, field_name: str) -> ControlMode:
    mode = _require_str(value, field_name).lower()
    if mode not in {"disabled", "external", "on_the_fly"}:
        raise ConfigError(f"{field_name} must be one of: disabled, external, on_the_fly")
    return mode  # type: ignore[return-value]


def _parse_single_control(raw: object, control_name: str) -> ControlConfig:
    field = f"cosmos.controls.{control_name}"
    control_map = _require_dict(raw, field)

    mode = _parse_control_mode(control_map.get("mode"), f"{field}.mode")
    weight = _optional_float(control_map.get("weight"), f"{field}.weight", 1.0)
    if not 0.0 <= weight <= 1.0:
        raise ConfigError(f"{field}.weight must be in [0.0, 1.0], got {weight}")

    default_subdir = {
        "seg": "labels",
        "depth": "depth",
        "edge": "edges",
    }[control_name]
    subdir = _optional_str(control_map.get("subdir"), f"{field}.subdir") or default_subdir

    return ControlConfig(mode=mode, weight=weight, subdir=subdir)


def _parse_controls(cosmos_data: dict, dataset_data: dict) -> CosmosControls:
    controls_data = cosmos_data.get("controls")
    if controls_data is not None:
        controls_map = _require_dict(controls_data, "cosmos.controls")
        missing = [name for name in CONTROL_NAMES if name not in controls_map]
        if missing:
            raise ConfigError(f"Missing control config for: {', '.join(missing)}")

        return CosmosControls(
            seg=_parse_single_control(controls_map.get("seg"), "seg"),
            depth=_parse_single_control(controls_map.get("depth"), "depth"),
            edge=_parse_single_control(controls_map.get("edge"), "edge"),
        )

    # Backward compatibility for old schema.
    seg_subdir = _optional_str(dataset_data.get("label_subdir"), "dataset.label_subdir") or "labels"
    edge_enabled = _optional_bool(cosmos_data.get("use_edge_control"), "cosmos.use_edge_control", True)

    return CosmosControls(
        seg=ControlConfig(
            mode="external",
            weight=_optional_float(cosmos_data.get("seg_control_weight"), "cosmos.seg_control_weight", 0.6),
            subdir=seg_subdir,
        ),
        depth=ControlConfig(mode="disabled", weight=1.0, subdir="depth"),
        edge=ControlConfig(
            mode="on_the_fly" if edge_enabled else "disabled",
            weight=_optional_float(cosmos_data.get("edge_control_weight"), "cosmos.edge_control_weight", 1.0),
            subdir="edges",
        ),
    )


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

    controls = _parse_controls(cosmos_data=cosmos_data, dataset_data=dataset_data)

    cosmos = CosmosConfig(
        repo_root=Path(_require_str(cosmos_data.get("repo_root"), "cosmos.repo_root")).expanduser().resolve(),
        disable_guardrails=_require_bool(cosmos_data.get("disable_guardrails"), "cosmos.disable_guardrails"),
        resolution=_require_str(cosmos_data.get("resolution"), "cosmos.resolution"),
        guidance=_require_float(cosmos_data.get("guidance"), "cosmos.guidance"),
        num_steps=_require_int(cosmos_data.get("num_steps"), "cosmos.num_steps"),
        max_frames=_require_int(cosmos_data.get("max_frames"), "cosmos.max_frames"),
        num_video_frames_per_chunk=_require_int(
            cosmos_data.get("num_video_frames_per_chunk"),
            "cosmos.num_video_frames_per_chunk",
        ),
        model=_optional_str(cosmos_data.get("model"), "cosmos.model"),
        model_variant=_optional_str(cosmos_data.get("model_variant"), "cosmos.model_variant") or "edge",
        model_distilled=_optional_bool(cosmos_data.get("model_distilled"), "cosmos.model_distilled", False),
        controls=controls,
    )

    input_root_raw = _optional_str(dataset_data.get("input_root"), "dataset.input_root")
    fallback_root = _optional_str(dataset_data.get("root"), "dataset.root")
    input_root_str = input_root_raw or fallback_root
    if input_root_str is None:
        raise ConfigError("dataset.input_root is required (or legacy dataset.root).")

    output_root_raw = _optional_str(dataset_data.get("output_root"), "dataset.output_root")
    output_root_str = output_root_raw or input_root_str

    dataset = DatasetConfig(
        input_root=Path(input_root_str).expanduser().resolve(),
        output_root=Path(output_root_str).expanduser().resolve(),
        original_dir=_optional_str(dataset_data.get("original_dir"), "dataset.original_dir") or ".",
        image_subdir=_optional_str(dataset_data.get("image_subdir"), "dataset.image_subdir") or "images",
        image_ext=_optional_str(dataset_data.get("image_ext"), "dataset.image_ext") or ".png",
    )

    if not cosmos.repo_root.is_dir():
        raise ConfigError(f"cosmos.repo_root is not a directory: {cosmos.repo_root}")

    if not dataset.input_root.is_dir():
        raise ConfigError(f"dataset.input_root is not a directory: {dataset.input_root}")

    original_root = dataset.input_root / dataset.original_dir
    images_dir = original_root / dataset.image_subdir
    if not images_dir.is_dir():
        raise ConfigError(f"Missing images dir: {images_dir}")

    for control_name, control in cosmos.controls.as_dict().items():
        if not control.is_external:
            continue
        control_dir = original_root / control.subdir
        if not control_dir.is_dir():
            raise ConfigError(
                f"Missing external control directory for '{control_name}': {control_dir}"
            )

    logging_data = data.get("logging")
    logging_cfg = LoggingConfig()
    if logging_data is not None:
        logging_map = _require_dict(logging_data, "logging")
        level = _optional_str(logging_map.get("level"), "logging.level") or "INFO"
        file_path = _resolve_optional_path(logging_map.get("file"), "logging.file", path)
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

    return GlobalConfig(cosmos=cosmos, dataset=dataset, augmentations=augmentations, logging=logging_cfg)
