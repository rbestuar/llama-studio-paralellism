"""Port availability checking utility."""

import socket
import logging

logger = logging.getLogger(__name__)


def is_port_available(port: int) -> bool:
    """
    Check if a port is available for binding.
    Uses SO_REUSEADDR to detect ports in TIME_WAIT (reclaimable).

    Args:
        port: Port number to check

    Returns:
        True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
            return True
    except OSError:
        logger.warning(f"Port {port} is already in use")
        return False


