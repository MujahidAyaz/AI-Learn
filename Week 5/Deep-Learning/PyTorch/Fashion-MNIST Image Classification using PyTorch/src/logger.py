"""
============================================================
Project Logger
============================================================

Creates a timestamped log file for every training run.

Author : Mujahid Ayaz
============================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

import logging
from datetime import datetime

from configs.config import LOGS_DIR

# ==========================================================
# Create Logs Directory
# ==========================================================

LOGS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ==========================================================
# Timestamp
# ==========================================================

timestamp = datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S"
)

log_file = LOGS_DIR / f"training_{timestamp}.log"

# ==========================================================
# Logger Configuration
# ==========================================================

logger = logging.getLogger("FashionMNIST")

logger.setLevel(logging.INFO)

logger.handlers.clear()

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

# ==========================================================
# File Handler
# ==========================================================

file_handler = logging.FileHandler(
    log_file,
    encoding="utf-8",
)

file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# ==========================================================
# Console Handler
# ==========================================================

console_handler = logging.StreamHandler()

console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

logger.propagate = False