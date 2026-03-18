from __future__ import annotations

import shutil
from pathlib import Path

from .types import CONTROL_NAMES, GlobalConfig


class MergeError(RuntimeError):
    pass


def _copy_prefixed(src_files: list[Path], dst_dir: Path, prefix: str) -> None:
    for src in src_files:
        dst_name = f"{prefix}_{src.name}"
        shutil.copy2(src, dst_dir / dst_name)


def merge_datasets(config: GlobalConfig) -> None:
    input_root = Path(config.dataset.input_root)
    output_root = Path(config.dataset.output_root)
    complete = output_root / "complete_dataset"

    if complete.exists():
        shutil.rmtree(complete)

    img_out = complete / config.dataset.image_subdir
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out = complete / config.dataset.label_subdir
    lbl_out.mkdir(parents=True, exist_ok=True)

    controls = config.cosmos.controls.as_dict()
    control_outputs: dict[str, Path] = {}
    for control_name in CONTROL_NAMES:
        control_cfg = controls[control_name]
        if not control_cfg.is_external:
            continue
        dst = complete / control_cfg.subdir
        dst.mkdir(parents=True, exist_ok=True)
        control_outputs[control_name] = dst

    orig_root = input_root / config.dataset.original_dir
    orig_images = orig_root / config.dataset.image_subdir
    if not orig_images.is_dir():
        raise MergeError(f"Original images dir missing: {orig_images}")
    orig_labels = orig_root / config.dataset.label_subdir
    if not orig_labels.is_dir():
        raise MergeError(f"Original labels dir missing: {orig_labels}")

    orig_img_files = sorted(orig_images.glob(f"*{config.dataset.image_ext}"))
    _copy_prefixed(orig_img_files, img_out, "orig")
    orig_lbl_files = sorted(orig_labels.glob(f"*{config.dataset.image_ext}"))
    _copy_prefixed(orig_lbl_files, lbl_out, "orig")

    for control_name, dst_dir in control_outputs.items():
        src_dir = orig_root / controls[control_name].subdir
        if not src_dir.is_dir():
            raise MergeError(f"Original control dir missing for '{control_name}': {src_dir}")
        src_files = sorted(src_dir.glob(f"*{config.dataset.image_ext}"))
        _copy_prefixed(src_files, dst_dir, "orig")

    for aug_cfg in config.augmentations:
        aug_root = output_root / aug_cfg.output_dir
        aug_img_dir = aug_root / config.dataset.image_subdir
        if not aug_img_dir.is_dir():
            raise MergeError(f"Augmentation image output missing for '{aug_cfg.name}': {aug_img_dir}")
        aug_lbl_dir = aug_root / config.dataset.label_subdir
        if not aug_lbl_dir.is_dir():
            raise MergeError(f"Augmentation label output missing for '{aug_cfg.name}': {aug_lbl_dir}")

        aug_images = sorted(aug_img_dir.glob(f"*{config.dataset.image_ext}"))
        _copy_prefixed(aug_images, img_out, aug_cfg.name)
        aug_labels = sorted(aug_lbl_dir.glob(f"*{config.dataset.image_ext}"))
        _copy_prefixed(aug_labels, lbl_out, aug_cfg.name)

        for control_name, dst_dir in control_outputs.items():
            aug_control_dir = aug_root / controls[control_name].subdir
            if not aug_control_dir.is_dir():
                raise MergeError(
                    f"Augmentation control output missing for '{aug_cfg.name}' control '{control_name}': {aug_control_dir}"
                )
            aug_controls = sorted(aug_control_dir.glob(f"*{config.dataset.image_ext}"))
            _copy_prefixed(aug_controls, dst_dir, aug_cfg.name)
