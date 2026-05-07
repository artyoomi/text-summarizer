import logging
import sys

def setup_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Add handler if not already added
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        ))
        logger.addHandler(handler)
        logger.propagate = False

    # Silence root
    logging.getLogger().setLevel(logging.ERROR)

    return logger
