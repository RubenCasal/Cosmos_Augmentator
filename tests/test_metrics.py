from __future__ import annotations

import unittest

from src.metrics import AugmentationTiming, RunTiming, format_seconds


class MetricsTests(unittest.TestCase):
    def test_augmentation_timing_averages_only_successful_images(self) -> None:
        timing = AugmentationTiming(name="sunset")
        timing.record_success("a.png", 1.5)
        timing.record_success("b.png", 2.5)
        timing.record_failure("c.png", 10.0, RuntimeError("boom"))

        self.assertEqual(timing.total_images, 3)
        self.assertEqual(timing.successful_images, 2)
        self.assertEqual(timing.failed_images, 1)
        self.assertAlmostEqual(timing.avg_success_seconds, 2.0)

    def test_run_timing_aggregation(self) -> None:
        first = AugmentationTiming(name="sunset")
        first.record_success("a.png", 2.0)

        second = AugmentationTiming(name="night")
        second.record_success("b.png", 1.0)
        second.record_failure("c.png", 3.0, RuntimeError("error"))

        run = RunTiming()
        run.add(first)
        run.add(second)

        self.assertEqual(run.augmentations, 2)
        self.assertEqual(run.total_images, 3)
        self.assertEqual(run.successful_images, 2)
        self.assertEqual(run.failed_images, 1)
        self.assertAlmostEqual(run.avg_success_seconds, 1.5)

    def test_format_seconds(self) -> None:
        self.assertEqual(format_seconds(9.2), "00m09s")
        self.assertEqual(format_seconds(598.0), "09m58s")


if __name__ == "__main__":
    unittest.main()
