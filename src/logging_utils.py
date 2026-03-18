from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(level_name: str = "INFO", file_path: Path | None = None) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if file_path is not None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(file_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
