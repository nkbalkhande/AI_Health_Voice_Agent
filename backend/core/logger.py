import logging
import sys
from pathlib import Path

from backend.core.config import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _build_handlers() -> list[logging.Handler]:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if settings.log_file_path:
        log_path = Path(settings.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    return handlers


def setup_logging() -> logging.Logger:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        handlers=_build_handlers(),
    )
    return logging.getLogger("genai_platform")


logger = setup_logging()
