import logging
import os
import sys
from typing import Optional


def setup_logging(log_file: Optional[str] = None, level: Optional[str] = None) -> None:
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_file = log_file if log_file is not None else os.getenv("LOG_FILE", "tkg.log")
    if log_file is not None and not str(log_file).strip():
        log_file = None

    root = logging.getLogger()
    if getattr(root, "_tkg_logging_configured", False):
        return

    root.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root._tkg_logging_configured = True
