from __future__ import annotations

from pathlib import Path

from .types import CONTROL_NAMES, ControlName, CosmosControls, DatasetConfig, ImageSample


class DatasetScanError(ValueError):
    pass


def _list_images(images_dir: Path, image_ext: str) -> list[Path]:
    image_files = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() == image_ext.lower()
    )
    if not image_files:
        raise DatasetScanError(f"No image files with extension '{image_ext}' found in {images_dir}")
    return image_files


def scan_dataset(dataset: DatasetConfig, controls: CosmosControls) -> list[ImageSample]:
    original_root = dataset.input_root / dataset.original_dir
    images_dir = original_root / dataset.image_subdir

    if not images_dir.is_dir():
        raise DatasetScanError(f"Images directory not found: {images_dir}")

    controls_map = controls.as_dict()
    external_dirs: dict[ControlName, Path] = {}
    for control_name in CONTROL_NAMES:
        control = controls_map[control_name]
        if not control.is_external:
            continue

        control_dir = original_root / control.subdir
        if not control_dir.is_dir():
            raise DatasetScanError(
                f"External control directory for '{control_name}' not found: {control_dir}"
            )
        external_dirs[control_name] = control_dir

    image_files = _list_images(images_dir=images_dir, image_ext=dataset.image_ext)

    missing_by_control: dict[ControlName, list[str]] = {name: [] for name in external_dirs}
    samples: list[ImageSample] = []

    for image_path in image_files:
        control_paths: dict[ControlName, Path | None] = {}

        for control_name in CONTROL_NAMES:
            control = controls_map[control_name]
            if not control.is_external:
                control_paths[control_name] = None
                continue

            control_path = external_dirs[control_name] / image_path.name
            if not control_path.exists():
                missing_by_control[control_name].append(image_path.name)
                control_paths[control_name] = None
            else:
                control_paths[control_name] = control_path.resolve()

        samples.append(
            ImageSample(
                name=image_path.name,
                image_path=image_path.resolve(),
                control_paths=control_paths,
            )
        )

    missing_messages: list[str] = []
    for control_name, missing in missing_by_control.items():
        if not missing:
            continue
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        missing_messages.append(f"{control_name}: {preview}{suffix}")

    if missing_messages:
        raise DatasetScanError(
            "Missing external control files for images. " + " | ".join(missing_messages)
        )

    return samples
