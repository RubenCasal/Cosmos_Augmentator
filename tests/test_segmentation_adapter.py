from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.segmentation_adapter import SegmentationAdapter
from src.types import DatasetConfig


class SegmentationAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "labels").mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_id_masks_are_converted_to_rgb_for_cosmos(self) -> None:
        label_path = self.root / "labels" / "sample.png"
        mask = Image.new("L", (2, 2))
        mask.putdata([0, 1, 2, 3])
        mask.save(label_path)

        dataset_cfg = DatasetConfig(
            root=self.root,
            original_dir=".",
            image_subdir="images",
            label_subdir="labels",
            image_ext=".png",
            segmentation=DatasetConfig.SegmentationConfig(
                encoding="id",
                convert_ids_to_rgb_for_cosmos=True,
                converted_cache_dir=".cache_seg",
            ),
        )
        adapter = SegmentationAdapter(dataset_cfg)
        converted = adapter.prepare_for_cosmos(label_path)

        self.assertNotEqual(converted, label_path)
        self.assertTrue(converted.exists())
        with Image.open(converted) as converted_image:
            self.assertEqual(converted_image.mode, "RGB")
            self.assertEqual(converted_image.size, (2, 2))

    def test_rgb_masks_are_used_directly(self) -> None:
        label_path = self.root / "labels" / "sample_rgb.png"
        mask = Image.new("RGB", (2, 2), color=(255, 0, 0))
        mask.save(label_path)

        dataset_cfg = DatasetConfig(
            root=self.root,
            original_dir=".",
            image_subdir="images",
            label_subdir="labels",
            image_ext=".png",
            segmentation=DatasetConfig.SegmentationConfig(encoding="rgb"),
        )
        adapter = SegmentationAdapter(dataset_cfg)
        prepared = adapter.prepare_for_cosmos(label_path)
        self.assertEqual(prepared, label_path)


if __name__ == "__main__":
    unittest.main()

