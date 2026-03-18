from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from .types import DatasetConfig

logger = logging.getLogger(__name__)


def _id_to_rgb(class_id: int) -> tuple[int, int, int]:
    # Pascal VOC-like deterministic palette from integer ids.
    r = 0
    g = 0
    b = 0
    value = int(class_id)
    for bit in range(8):
        r |= ((value >> 0) & 1) << (7 - bit)
        g |= ((value >> 1) & 1) << (7 - bit)
        b |= ((value >> 2) & 1) << (7 - bit)
        value >>= 3
    return r, g, b


class SegmentationAdapter:
    def __init__(self, dataset_cfg: DatasetConfig) -> None:
        self.dataset_cfg = dataset_cfg
        self.seg_cfg = dataset_cfg.segmentation
        self.cache_root = dataset_cfg.root / self.seg_cfg.converted_cache_dir

    def prepare_for_cosmos(self, label_path: Path) -> Path:
        if self.seg_cfg.encoding == "rgb":
            return label_path
        if not self.seg_cfg.convert_ids_to_rgb_for_cosmos:
            return label_path

        source = label_path.resolve()
        try:
            relative = source.relative_to(self.dataset_cfg.root.resolve())
        except ValueError:
            relative = Path(source.name)

        target = (self.cache_root / relative).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
            return target

        self._convert_id_mask_to_rgb(source, target)
        return target

    def _convert_id_mask_to_rgb(self, source: Path, target: Path) -> None:
        with Image.open(source) as image:
            if image.mode in {"RGB", "RGBA"}:
                logger.warning(
                    "Segmentation encoding is set to 'id' but '%s' is already RGB-like (%s). "
                    "Converting anyway.",
                    source,
                    image.mode,
                )

            id_image = image.convert("I")
            ids = list(id_image.getdata())

        palette_cache: dict[int, tuple[int, int, int]] = {}
        rgb_pixels: list[tuple[int, int, int]] = []
        for class_id in ids:
            key = int(class_id)
            color = palette_cache.get(key)
            if color is None:
                color = _id_to_rgb(key)
                palette_cache[key] = color
            rgb_pixels.append(color)

        rgb_image = Image.new("RGB", id_image.size)
        rgb_image.putdata(rgb_pixels)
        rgb_image.save(target)

