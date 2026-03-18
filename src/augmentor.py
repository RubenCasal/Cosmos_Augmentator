from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from tqdm import tqdm

from .augmentation_profile import AugmentationProfile
from .control_image_adapter import ControlImageAdapter
from .cosmos_runner import CosmosGenerationRequest, CosmosRunner
from .dataset_scanner import scan_dataset
from .metrics import AugmentationTiming, RunTiming, format_seconds
from .types import CONTROL_NAMES, ControlName, AugmentationJob, GlobalConfig, ImageSample

logger = logging.getLogger(__name__)


class DatasetAugmentor:
    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        self.samples = scan_dataset(dataset=config.dataset, controls=config.cosmos.controls)
        self.cosmos_runner = CosmosRunner(config.cosmos)
        self.control_adapter = ControlImageAdapter(config)
        self._adapted_control_cache: dict[tuple[ControlName, str], Path] = {}

        controls = config.cosmos.controls.as_dict()
        logger.info(
            "Controls: seg=%s depth=%s edge=%s",
            controls["seg"].mode,
            controls["depth"].mode,
            controls["edge"].mode,
        )

    def _build_jobs_for_profile(self, profile: AugmentationProfile) -> list[AugmentationJob]:
        selected_samples: list[ImageSample] = profile.select_samples(self.samples)
        jobs: list[AugmentationJob] = []

        for idx, sample in enumerate(selected_samples, start=1):
            request_name = f"{profile.name}_{Path(sample.name).stem}"
            jobs.append(
                AugmentationJob(
                    augmentation_name=profile.name,
                    output_dir_name=profile.output_dir_name,
                    request_name=request_name,
                    seed=profile.seed_base + idx,
                    image_name=sample.name,
                    image_path=sample.image_path,
                    gt_seg_path=sample.gt_seg_path,
                    control_paths=sample.control_paths,
                    prompt=profile.prompt,
                    negative_prompt=profile.negative_prompt,
                )
            )

        return jobs

    def _prepare_output_dirs_for_profile(
        self,
        output_root: Path,
        output_dir_name: str,
    ) -> tuple[Path, Path, dict[str, Path], Path]:
        aug_root = output_root / output_dir_name
        aug_images = aug_root / self.config.dataset.image_subdir
        aug_labels = aug_root / self.config.dataset.label_subdir
        temp_output = aug_root / "_cosmos_output"

        if temp_output.exists():
            shutil.rmtree(temp_output)

        aug_images.mkdir(parents=True, exist_ok=True)
        aug_labels.mkdir(parents=True, exist_ok=True)
        temp_output.mkdir(parents=True, exist_ok=True)

        control_dirs: dict[str, Path] = {}
        for control_name in CONTROL_NAMES:
            control_cfg = self.config.cosmos.controls.as_dict()[control_name]
            if not control_cfg.is_external:
                continue
            dst_dir = aug_root / control_cfg.subdir
            dst_dir.mkdir(parents=True, exist_ok=True)
            control_dirs[control_name] = dst_dir

        return aug_images, aug_labels, control_dirs, temp_output

    @staticmethod
    def _copy_file(src: Path, dst: Path) -> None:
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)

    def _adapt_control_paths(self, control_paths: dict[ControlName, Path | None]) -> dict[ControlName, Path | None]:
        adapted: dict[ControlName, Path | None] = {}
        for control_name, source_path in control_paths.items():
            if source_path is None:
                adapted[control_name] = None
                continue

            source_resolved = str(source_path.resolve())
            cache_key = (control_name, source_resolved)
            cached = self._adapted_control_cache.get(cache_key)
            if cached is not None:
                adapted[control_name] = cached
                continue

            converted = self.control_adapter.adapt_external_control_path(control_name, source_path)
            self._adapted_control_cache[cache_key] = converted
            adapted[control_name] = converted
        return adapted

    def _materialize_job(
        self,
        job: AugmentationJob,
        generated_path: Path,
        aug_images: Path,
        aug_labels: Path,
        control_dirs: dict[str, Path],
    ) -> None:
        if not generated_path.exists():
            raise RuntimeError(f"Generated image not found: {generated_path}")

        image_dst = aug_images / job.image_name
        self._copy_file(generated_path, image_dst)

        label_dst = aug_labels / job.image_name
        self._copy_file(job.gt_seg_path, label_dst)

        for control_name, dst_dir in control_dirs.items():
            src_control = job.control_paths.get(control_name)
            if src_control is None:
                continue
            self._copy_file(src_control, dst_dir / job.image_name)

    def _run_with_python_api(
        self,
        jobs: list[AugmentationJob],
        profile: AugmentationProfile,
        temp_output: Path,
        aug_images: Path,
        aug_labels: Path,
        control_dirs: dict[str, Path],
        aug_timing: AugmentationTiming,
    ) -> None:
        for idx, job in enumerate(tqdm(jobs, desc=f"Generating {profile.name}", unit="image"), start=1):
            image_start = time.perf_counter()
            try:
                generated_path = self.cosmos_runner.run_single(
                    image_path=job.image_path,
                    prompt=job.prompt,
                    negative_prompt=job.negative_prompt,
                    output_dir=temp_output,
                    seed=job.seed,
                    name=job.request_name,
                    control_paths=self._adapt_control_paths(job.control_paths),
                )
                self._materialize_job(
                    job=job,
                    generated_path=generated_path,
                    aug_images=aug_images,
                    aug_labels=aug_labels,
                    control_dirs=control_dirs,
                )

                elapsed = time.perf_counter() - image_start
                aug_timing.record_success(job.image_name, elapsed)
                logger.info(
                    "[img %d/%d][%s] %.2fs | %s",
                    idx,
                    len(jobs),
                    profile.name,
                    elapsed,
                    job.image_name,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - image_start
                aug_timing.record_failure(job.image_name, elapsed, exc)
                logger.exception(
                    "[img %d/%d][%s] FAILED in %.2fs | %s",
                    idx,
                    len(jobs),
                    profile.name,
                    elapsed,
                    job.image_name,
                )

    def _run_with_subprocess_batches(
        self,
        jobs: list[AugmentationJob],
        profile: AugmentationProfile,
        temp_output: Path,
        aug_images: Path,
        aug_labels: Path,
        control_dirs: dict[str, Path],
        aug_timing: AugmentationTiming,
    ) -> None:
        requests: list[CosmosGenerationRequest] = []
        for job in jobs:
            request = CosmosGenerationRequest(
                image_path=job.image_path,
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                seed=job.seed,
                name=job.request_name,
                control_paths=self._adapt_control_paths(job.control_paths),
            )
            requests.append(request)

        logger.info(
            "Generating %d samples for '%s' in single-process subprocess mode.",
            len(requests),
            profile.name,
        )
        batch_result = self.cosmos_runner.run_many(requests=requests, output_dir=temp_output)

        for idx, job in enumerate(jobs, start=1):
            elapsed = batch_result.elapsed_seconds.get(job.request_name, 0.0)
            error = batch_result.errors.get(job.request_name)
            if error is not None:
                aug_timing.record_failure(job.image_name, elapsed, error)
                logger.error(
                    "[img %d/%d][%s] FAILED in %.2fs | %s | %s",
                    idx,
                    len(jobs),
                    profile.name,
                    elapsed,
                    job.image_name,
                    error,
                )
                continue

            generated_path = batch_result.outputs.get(job.request_name)
            if generated_path is None:
                error = RuntimeError(f"No generated path found for request '{job.request_name}'.")
                aug_timing.record_failure(job.image_name, elapsed, error)
                logger.error(
                    "[img %d/%d][%s] FAILED in %.2fs | %s | %s",
                    idx,
                    len(jobs),
                    profile.name,
                    elapsed,
                    job.image_name,
                    error,
                )
                continue

            try:
                self._materialize_job(
                    job=job,
                    generated_path=generated_path,
                    aug_images=aug_images,
                    aug_labels=aug_labels,
                    control_dirs=control_dirs,
                )
                aug_timing.record_success(job.image_name, elapsed)
                logger.info(
                    "[img %d/%d][%s] %.2fs | %s",
                    idx,
                    len(jobs),
                    profile.name,
                    elapsed,
                    job.image_name,
                )
            except Exception as exc:
                aug_timing.record_failure(job.image_name, elapsed, exc)
                logger.exception(
                    "[img %d/%d][%s] FAILED in %.2fs | %s",
                    idx,
                    len(jobs),
                    profile.name,
                    elapsed,
                    job.image_name,
                )

    def run_augmentations(self) -> None:
        output_root = Path(self.config.dataset.output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        run_timing = RunTiming()
        logger.info("Found %d source samples in input dataset.", len(self.samples))

        run_start = time.perf_counter()
        for aug_cfg in self.config.augmentations:
            profile = AugmentationProfile.from_config(aug_cfg)
            jobs = self._build_jobs_for_profile(profile)

            logger.info(
                "Augmentation '%s': selected %d/%d samples (fraction=%.3f).",
                profile.name,
                len(jobs),
                len(self.samples),
                profile.fraction,
            )

            if not jobs:
                run_timing.add(AugmentationTiming(name=profile.name))
                continue

            aug_images, aug_labels, control_dirs, temp_output = self._prepare_output_dirs_for_profile(
                output_root=output_root,
                output_dir_name=profile.output_dir_name,
            )

            aug_timing = AugmentationTiming(name=profile.name)
            try:
                if self.cosmos_runner.uses_python_api:
                    self._run_with_python_api(
                        jobs=jobs,
                        profile=profile,
                        temp_output=temp_output,
                        aug_images=aug_images,
                        aug_labels=aug_labels,
                        control_dirs=control_dirs,
                        aug_timing=aug_timing,
                    )
                else:
                    self._run_with_subprocess_batches(
                        jobs=jobs,
                        profile=profile,
                        temp_output=temp_output,
                        aug_images=aug_images,
                        aug_labels=aug_labels,
                        control_dirs=control_dirs,
                        aug_timing=aug_timing,
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
