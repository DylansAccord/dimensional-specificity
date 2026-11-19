"""
Structured logging for the experiment pipeline.

Each pipeline phase emits timestamped log entries to both stderr and a
phase-specific log file. Phase timing is recorded so cost overruns can be
diagnosed after the fact.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path


def get_logger(name: str, log_dir: str | Path | None = None) -> logging.Logger:
    """Get a configured logger. If log_dir provided, also writes to a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler if log_dir provided
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


@contextmanager
def timed_phase(logger: logging.Logger, phase_name: str):
    """Context manager that logs phase start, end, and elapsed time."""
    logger.info(f"PHASE START: {phase_name}")
    start = time.time()
    try:
        yield
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"PHASE FAILED: {phase_name} after {elapsed:.1f}s: {type(e).__name__}: {e}")
        raise
    else:
        elapsed = time.time() - start
        logger.info(f"PHASE END: {phase_name} ({elapsed:.1f}s)")
