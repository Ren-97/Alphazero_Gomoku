import logging
import os
import sys
from typing import Optional


def setup_logging(
    *,
    log_file: Optional[str] = "logs/train.log",
    level: int = logging.INFO,
) -> None:
    """
    Configure root logging once.

    - Console: always enabled (stdout)
    - File: enabled when log_file is not None
    """
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

