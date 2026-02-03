# Copyright (c) 2026 Zhe Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import logging.handlers
import sys
from pathlib import Path


def configure_logging():
    """Configure logging for the serving app"""
    log_dir = Path("/app/logs") if Path("/app").exists() else Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # Console handler
            logging.StreamHandler(sys.stdout),
            # File handler with rotation
            logging.handlers.RotatingFileHandler(
                log_dir / "app.log",
                maxBytes=10485760,  # 10MB
                backupCount=5,
            ),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info("Logging initialized for nanoRecSys serving app")


# Initialize logging on import
configure_logging()
