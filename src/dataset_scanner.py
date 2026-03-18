from __future__ import annotations

from pathlib import Path

from .types import ImageSample


class DatasetScanError(ValueError):
    pass


def scan_dataset(
    root: Path,
    original_dir: str,
    image_subdir: str,
    label_subdir: str,
    image_ext: str,
) -> list[ImageSample]:
    original_root = root / original_dir
    images_dir = original_root / image_subdir
    labels_dir = original_root / label_subdir

    if not images_dir.is_dir():
        raise DatasetScanError(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise DatasetScanError(f"Labels directory not found: {labels_dir}")

    image_files = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() == image_ext.lower()
    )
    if not image_files:
        raise DatasetScanError(f"No image files with extension '{image_ext}' found in {images_dir}")

    label_names = {
        p.name for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == image_ext.lower()
    }

    samples: list[ImageSample] = []
    missing_labels: list[str] = []
    extra_labels = set(label_names)

    for image_path in image_files:
        label_path = labels_dir / image_path.name
        if not label_path.exists():
            missing_labels.append(image_path.name)
            continue

        extra_labels.discard(image_path.name)
        samples.append(
            ImageSample(
                name=image_path.name,
                image_path=image_path.resolve(),
                label_path=label_path.resolve(),
            )
        )

    if missing_labels:
        suffix = "..." if len(missing_labels) > 10 else ""
        raise DatasetScanError(
            "Missing label files for images: " + ", ".join(missing_labels[:10]) + suffix
        )

    if extra_labels:
        extras = sorted(extra_labels)
        suffix = "..." if len(extras) > 10 else ""
        raise DatasetScanError(
            "Found labels without matching images: " + ", ".join(extras[:10]) + suffix
        )

    return samples
