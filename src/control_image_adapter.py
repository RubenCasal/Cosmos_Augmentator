from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from .types import ControlName, GlobalConfig

logger = logging.getLogger(__name__)


def _id_to_rgb(class_id: int) -> tuple[int, int, int]:
    # Deterministic palette mapping from integer ids to RGB.
    red = 0
    green = 0
    blue = 0
    value = int(class_id)
    for bit in range(8):
        red |= ((value >> 0) & 1) << (7 - bit)
        green |= ((value >> 1) & 1) << (7 - bit)
        blue |= ((value >> 2) & 1) << (7 - bit)
        value >>= 3
    return red, green, blue


def _normalize_to_uint8(values: list[float]) -> list[int]:
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0 for _ in values]
    scale = 255.0 / (max_val - min_val)
    return [max(0, min(255, int(round((v - min_val) * scale)))) for v in values]


def _to_gray_from_numeric(values: list[float], size: tuple[int, int], invert: bool = False) -> Image.Image:
    normalized = _normalize_to_uint8(values)
    if invert:
        normalized = [255 - value for value in normalized]
    gray = Image.new("L", size)
    gray.putdata(normalized)
    return gray


class ControlImageAdapter:
    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        self.cache_root = config.dataset.output_root / config.dataset.cache_dir

    def adapt_external_control_path(self, control_name: ControlName, source_path: Path) -> Path:
        """
        Returns a Cosmos-friendly control image path.

        - seg: supports encoding=rgb|id. If id, converts to RGB.
        - depth/edge: if external control is not RGB, converts to RGB by repeating channel(s).
        """
        control_cfg = self.config.cosmos.controls.as_dict()[control_name]
        if not control_cfg.is_external:
            return source_path

        if control_name == "seg":
            encoding = (control_cfg.encoding or "rgb").lower()
            if encoding == "id":
                return self._convert_seg_id_to_rgb(source_path)
            return source_path

        if control_name in {"depth", "edge"}:
            return self._convert_mono_to_rgb_if_needed(control_name, source_path)

        return source_path

    def _convert_seg_id_to_rgb(self, source_path: Path) -> Path:
        source = source_path.resolve()
        target_dir = (self.cache_root / "seg_rgb").resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        target = (target_dir / source.name).resolve()

        try:
            if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
                return target
        except OSError:
            pass

        with Image.open(source) as image:
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
        logger.debug("Converted seg id mask to RGB: %s -> %s", source, target)
        return target

    def _convert_mono_to_rgb_if_needed(self, control_name: ControlName, source_path: Path) -> Path:
        source = source_path.resolve()
        cache_subdir = f"{control_name}_rgb"
        if control_name == "depth":
            # New cache namespace to avoid reusing stale depth conversions from previous logic.
            cache_subdir = "depth_rgb_v2"
        target_dir = (self.cache_root / cache_subdir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        target = (target_dir / source.name).resolve()

        try:
            if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
                return target
        except OSError:
            pass

        with Image.open(source) as image:
            # For depth, always convert to normalized/inverted gray so it matches Cosmos convention:
            # white=near, black=far.
            if control_name == "depth":
                if image.mode in {"I;16", "I", "F"}:
                    gray = _to_gray_from_numeric(
                        values=[float(value) for value in image.getdata()],
                        size=image.size,
                        invert=True,
                    )
                else:
                    source_gray = image.convert("L")
                    gray = _to_gray_from_numeric(
                        values=[float(value) for value in source_gray.getdata()],
                        size=image.size,
                        invert=True,
                    )
            else:
                if image.mode == "RGB":
                    return source

                # Edge maps can also come in high precision. Normalize them to 8-bit for Cosmos.
                if image.mode in {"I;16", "I", "F"}:
                    gray = _to_gray_from_numeric(
                        values=[float(value) for value in image.getdata()],
                        size=image.size,
                        invert=False,
                    )
                else:
                    gray = image.convert("L")

            rgb = Image.merge("RGB", (gray, gray, gray))
            rgb.save(target)

        logger.debug("Converted %s control to RGB: %s -> %s", control_name, source, target)
        return target
