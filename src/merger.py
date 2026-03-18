from __future__ import annotations

import shutil
from pathlib import Path

from .types import GlobalConfig


class MergeError(RuntimeError):
    pass


def _copy_prefixed(src_files: list[Path], dst_dir: Path, prefix: str) -> None:
    for src in src_files:
        dst_name = f"{prefix}_{src.name}"
        shutil.copy2(src, dst_dir / dst_name)


def merge_datasets(config: GlobalConfig) -> None:
    root = Path(config.dataset.root)
    complete = root / "complete_dataset"

    if complete.exists():
        shutil.rmtree(complete)

    img_out = complete / config.dataset.image_subdir
    lbl_out = complete / config.dataset.label_subdir
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    orig_root = root / config.dataset.original_dir
    orig_images = orig_root / config.dataset.image_subdir
    orig_labels = orig_root / config.dataset.label_subdir

    orig_img_files = sorted(orig_images.glob(f"*{config.dataset.image_ext}"))
    orig_lbl_files = sorted(orig_labels.glob(f"*{config.dataset.image_ext}"))
    _copy_prefixed(orig_img_files, img_out, "orig")
    _copy_prefixed(orig_lbl_files, lbl_out, "orig")

    for aug_cfg in config.augmentations:
        aug_root = root / aug_cfg.output_dir
        aug_img_dir = aug_root / config.dataset.image_subdir
        aug_lbl_dir = aug_root / config.dataset.label_subdir

        if not aug_img_dir.exists() or not aug_lbl_dir.exists():
            raise MergeError(
                f"Augmentation output missing for '{aug_cfg.name}'. Expected {aug_img_dir} and {aug_lbl_dir}."
            )

        aug_images = sorted(aug_img_dir.glob(f"*{config.dataset.image_ext}"))
        aug_labels = sorted(aug_lbl_dir.glob(f"*{config.dataset.image_ext}"))

        _copy_prefixed(aug_images, img_out, aug_cfg.name)
        _copy_prefixed(aug_labels, lbl_out, aug_cfg.name)
