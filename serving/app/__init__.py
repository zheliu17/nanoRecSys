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
