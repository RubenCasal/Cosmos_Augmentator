from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from tqdm import tqdm

from .augmentation_profile import AugmentationProfile
from .cosmos_runner import CosmosRunner
from .dataset_scanner import scan_dataset
from .metrics import AugmentationTiming, RunTiming, format_seconds
from .segmentation_adapter import SegmentationAdapter
from .types import GlobalConfig

logger = logging.getLogger(__name__)


class DatasetAugmentor:
    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        self.samples = scan_dataset(
            root=config.dataset.root,
            original_dir=config.dataset.original_dir,
            image_subdir=config.dataset.image_subdir,
            label_subdir=config.dataset.label_subdir,
            image_ext=config.dataset.image_ext,
        )
        self.cosmos_runner = CosmosRunner(config.cosmos)
        self.segmentation_adapter = SegmentationAdapter(config.dataset)
        logger.info(
            "Segmentation mode: encoding=%s convert_ids_to_rgb_for_cosmos=%s cache_dir=%s",
            config.dataset.segmentation.encoding,
            config.dataset.segmentation.convert_ids_to_rgb_for_cosmos,
            config.dataset.segmentation.converted_cache_dir,
        )

    def _prepare_output_dirs_for_profile(self, dataset_root: Path, output_dir_name: str) -> tuple[Path, Path, Path]:
        aug_root = dataset_root / output_dir_name
        aug_images = aug_root / self.config.dataset.image_subdir
        aug_labels = aug_root / self.config.dataset.label_subdir
        temp_output = aug_root / "_cosmos_output"

        if temp_output.exists():
            shutil.rmtree(temp_output)

        aug_images.mkdir(parents=True, exist_ok=True)
        aug_labels.mkdir(parents=True, exist_ok=True)
        temp_output.mkdir(parents=True, exist_ok=True)
        return aug_images, aug_labels, temp_output

    @staticmethod
    def _save_generated_as_dataset_image(generated_path: Path, expected_img_path: Path) -> None:
        if not generated_path.exists():
            raise RuntimeError(f"Generated image not found: {generated_path}")
        if expected_img_path.exists():
            expected_img_path.unlink()
        shutil.copy2(generated_path, expected_img_path)

    def run_augmentations(self) -> None:
        dataset_root = Path(self.config.dataset.root)
        run_timing = RunTiming()

        logger.info("Found %d source samples in original dataset.", len(self.samples))

        run_start = time.perf_counter()
        for aug_cfg in self.config.augmentations:
            profile = AugmentationProfile.from_config(aug_cfg)
            selected_samples = profile.select_samples(self.samples)

            logger.info(
                "Augmentation '%s': selected %d/%d samples (fraction=%.3f).",
                profile.name,
                len(selected_samples),
                len(self.samples),
                profile.fraction,
            )

            if not selected_samples:
                run_timing.add(AugmentationTiming(name=profile.name))
                continue

            aug_images, aug_labels, temp_output = self._prepare_output_dirs_for_profile(
                dataset_root=dataset_root,
                output_dir_name=profile.output_dir_name,
            )

            aug_timing = AugmentationTiming(name=profile.name)
            try:
                for idx, sample in enumerate(
                    tqdm(selected_samples, desc=f"Generating {profile.name}", unit="image"),
                    start=1,
                ):
                    seed = profile.seed_base + idx
                    request_name = f"{profile.name}_{sample.image_path.stem}"

                    image_start = time.perf_counter()
                    try:
                        generated_path = self.cosmos_runner.run_single(
                            image_path=sample.image_path,
                            seg_path=self.segmentation_adapter.prepare_for_cosmos(sample.label_path),
                            prompt=profile.prompt,
                            negative_prompt=profile.negative_prompt,
                            output_dir=temp_output,
                            seed=seed,
                            name=request_name,
                        )

                        expected_img_path = aug_images / sample.image_path.name
                        self._save_generated_as_dataset_image(generated_path, expected_img_path)

                        label_dst = aug_labels / sample.label_path.name
                        if label_dst.exists():
                            label_dst.unlink()
                        shutil.copy2(sample.label_path, label_dst)

                        elapsed = time.perf_counter() - image_start
                        aug_timing.record_success(sample.name, elapsed)
                        logger.info(
                            "[img %d/%d][%s] %.2fs | %s",
                            idx,
                            len(selected_samples),
                            profile.name,
                            elapsed,
                            sample.name,
                        )
                    except Exception as exc:
                        elapsed = time.perf_counter() - image_start
                        aug_timing.record_failure(sample.name, elapsed, exc)
                        logger.exception(
                            "[img %d/%d][%s] FAILED in %.2fs | %s",
                            idx,
                            len(selected_samples),
                            profile.name,
                            elapsed,
                            sample.name,
                        )
            finally:
                shutil.rmtree(temp_output, ignore_errors=True)

            run_timing.add(aug_timing)
            logger.info(
                "[augmentation %s] success=%d failed=%d avg=%.2fs total=%s",
                profile.name,
                aug_timing.successful_images,
                aug_timing.failed_images,
                aug_timing.avg_success_seconds,
                format_seconds(aug_timing.total_seconds),
            )
            if aug_timing.failures:
                logger.warning("[augmentation %s] first errors: %s", profile.name, "; ".join(aug_timing.failures[:5]))

        run_timing.total_seconds = time.perf_counter() - run_start
        logger.info(
            "[run summary] augmentations=%d images=%d success=%d failed=%d avg=%.2fs total=%s",
            run_timing.augmentations,
            run_timing.total_images,
            run_timing.successful_images,
            run_timing.failed_images,
            run_timing.avg_success_seconds,
            format_seconds(run_timing.total_seconds),
        )
