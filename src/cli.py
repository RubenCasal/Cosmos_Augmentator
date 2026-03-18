from __future__ import annotations

import argparse
from pathlib import Path

from .augmentor import DatasetAugmentor
from .config_schema import load_config
from .logging_utils import configure_logging
from .merger import merge_datasets


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automate dataset augmentation with Cosmos-Transfer2.5")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    parser.add_argument("--log-level", type=str, default=None, help="Override log level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional path to a log file")
    parser.add_argument("command", choices=["run-all", "run-augmentations", "merge"])
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = load_config(args.config)

    log_level = args.log_level or config.logging.level
    log_file = args.log_file or config.logging.file_path
    configure_logging(level_name=log_level, file_path=log_file)

    if args.command in ("run-all", "run-augmentations"):
        augmentor = DatasetAugmentor(config)
        augmentor.run_augmentations()

    if args.command in ("run-all", "merge"):
        merge_datasets(config)


if __name__ == "__main__":
    main()
