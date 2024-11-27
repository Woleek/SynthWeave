import logging

# Create a logger for the library
logger = logging.getLogger("SynthWeave")
logger.setLevel(logging.DEBUG) # Default logging level for the library

# Create a default handler
default_handler = logging.StreamHandler()
default_handler.setLevel(logging.WARNING)  # Default to WARNING level
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
default_handler.setFormatter(formatter)

# Add the handler only if no handlers exist
if not logger.handlers:
    logger.addHandler(default_handler)