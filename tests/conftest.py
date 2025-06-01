"""
Pytest configuration file.

This file configures pytest for the NIRS Tomato project.
"""

import logging
import sys

import pytest

# Import fixtures so they're available to all tests


# Configure logging for tests
@pytest.fixture(autouse=True)
def configure_logging():
    """Set up logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    # Suppress excessive logging from libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Reset handlers to avoid duplicate logs
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    yield
