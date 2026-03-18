from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from src.config_schema import load_config


class ConfigSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

        self.cosmos_repo = self.root / "cosmos"
        self.cosmos_repo.mkdir()

        self.dataset_root = self.root / "dataset"
        images = self.dataset_root / "images"
        labels = self.dataset_root / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        (images / "a.png").write_bytes(b"img")
        (labels / "a.png").write_bytes(b"lbl")

        self.config_path = self.root / "augmentations.yaml"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_config(self, data: dict) -> None:
        self.config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    def _base_config(self) -> dict:
        return {
            "cosmos": {
                "repo_root": str(self.cosmos_repo),
                "model_variant": "edge",
                "model_distilled": False,
                "use_edge_control": True,
                "edge_control_weight": 1.0,
                "seg_control_weight": 1.0,
                "disable_guardrails": True,
                "resolution": "720",
                "guidance": 4,
                "num_steps": 36,
                "max_frames": 1,
                "num_video_frames_per_chunk": 1,
            },
            "dataset": {
                "root": str(self.dataset_root),
                "original_dir": ".",
                "image_subdir": "images",
                "label_subdir": "labels",
                "image_ext": ".png",
            },
            "augmentations": [
                {
                    "name": "sunset",
                    "output_dir": "dataset_sunset",
                    "fraction": 1,
                    "seed_base": 400,
                    "prompt": "p",
                    "negative_prompt": "n",
                }
            ],
        }

    def test_backwards_compatible_logging_defaults(self) -> None:
        data = self._base_config()
        self._write_config(data)

        config = load_config(self.config_path)
        self.assertEqual(config.logging.level, "INFO")
        self.assertIsNone(config.logging.file_path)
        self.assertEqual(config.dataset.segmentation.encoding, "rgb")
        self.assertTrue(config.dataset.segmentation.convert_ids_to_rgb_for_cosmos)
        self.assertEqual(config.dataset.segmentation.converted_cache_dir, ".cosmos_seg_rgb_cache")

    def test_logging_section_is_parsed(self) -> None:
        data = self._base_config()
        data["logging"] = {
            "level": "debug",
            "file": "logs/run.log",
        }
        self._write_config(data)

        config = load_config(self.config_path)
        self.assertEqual(config.logging.level, "DEBUG")
        self.assertEqual(config.logging.file_path, (self.root / "logs" / "run.log").resolve())

    def test_segmentation_id_mode_is_parsed(self) -> None:
        data = self._base_config()
        data["dataset"]["segmentation"] = {
            "encoding": "id",
            "convert_ids_to_rgb_for_cosmos": True,
            "converted_cache_dir": ".cache/seg_rgb",
        }
        self._write_config(data)

        config = load_config(self.config_path)
        self.assertEqual(config.dataset.segmentation.encoding, "id")
        self.assertTrue(config.dataset.segmentation.convert_ids_to_rgb_for_cosmos)
        self.assertEqual(config.dataset.segmentation.converted_cache_dir, ".cache/seg_rgb")


if __name__ == "__main__":
    unittest.main()
