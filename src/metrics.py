from __future__ import annotations

from dataclasses import dataclass, field


def format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:02d}m{secs:02d}s"


@dataclass
class AugmentationTiming:
    name: str
    total_images: int = 0
    successful_images: int = 0
    failed_images: int = 0
    total_seconds: float = 0.0
    successful_seconds: float = 0.0
    failures: list[str] = field(default_factory=list)

    def record_success(self, image_name: str, seconds: float) -> None:
        self.total_images += 1
        self.successful_images += 1
        self.total_seconds += seconds
        self.successful_seconds += seconds

    def record_failure(self, image_name: str, seconds: float, error: Exception) -> None:
        self.total_images += 1
        self.failed_images += 1
        self.total_seconds += seconds
        self.failures.append(f"{image_name}: {error}")

    @property
    def avg_success_seconds(self) -> float:
        if self.successful_images == 0:
            return 0.0
        return self.successful_seconds / self.successful_images


@dataclass
class RunTiming:
    augmentations: int = 0
    total_images: int = 0
    successful_images: int = 0
    failed_images: int = 0
    successful_seconds: float = 0.0
    total_seconds: float = 0.0

    def add(self, timing: AugmentationTiming) -> None:
        self.augmentations += 1
        self.total_images += timing.total_images
        self.successful_images += timing.successful_images
        self.failed_images += timing.failed_images
        self.successful_seconds += timing.successful_seconds
        self.total_seconds += timing.total_seconds

    @property
    def avg_success_seconds(self) -> float:
        if self.successful_images == 0:
            return 0.0
        return self.successful_seconds / self.successful_images
