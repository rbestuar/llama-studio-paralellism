"""Detect llama-server version via --version flag."""

import subprocess
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def get_version(binary_path: str) -> Optional[str]:
    """
    Run llama-server --version and parse the version string.

    Expected format:
        version: 9030 (17df5830e)

    Returns:
        "9030_17df5830e" or None if detection fails
    """
    try:
        binary_path = Path(binary_path)
        if not binary_path.exists():
            logger.warning(f"⚠ Llama-server binary not found: {binary_path}")
            return None

        result = subprocess.run(
            [str(binary_path), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        # Parse "version: 9030 (17df5830e)" - check both stdout and stderr
        combined_output = result.stdout + result.stderr
        match = re.search(r'version:\s+(\d+)\s+\(([a-f0-9]+)\)', combined_output)
        if match:
            version_num, commit = match.groups()
            version_str = f"{version_num}_{commit}"
            logger.info(f"✓ Detected llama-server version: {version_str}")
            return version_str
        else:
            logger.warning(f"⚠ Could not parse version from: {result.stdout}")
            return None

    except subprocess.TimeoutExpired:
        logger.warning("⚠ llama-server --version timed out")
        return None
    except Exception as e:
        logger.warning(f"⚠ Error detecting version: {e}")
        return None
