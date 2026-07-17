"""
============================================================
Logging Configuration
============================================================

This module configures logging for the project.

Responsibilities
----------------
1. Create log directory
2. Configure file logger
3. Configure console logger
============================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

import logging

from configs.config import PROJECT_ROOT


# ==========================================================
# Log Directory
# ==========================================================

LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE = LOG_DIR / "training.log"


# ==========================================================
# Logger Configuration
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)