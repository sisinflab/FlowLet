import logging
import os
from datetime import datetime

def setup_logging(log_dir: str, filename_prefix: str = "flowlet_run"):
    """Configures logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{filename_prefix}_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    # Optional: Configure specific loggers differently if needed later
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete.")

def get_logger(name: str):
    """Gets a logger instance."""
    return logging.getLogger(name)