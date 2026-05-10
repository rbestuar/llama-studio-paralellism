"""Llama.cpp server options schema and validation."""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Runtime schema loaded from config/llama-server/
_RUNTIME_SCHEMA: Optional[Dict] = None


def set_runtime_schema(schema: Dict) -> None:
    """Set the runtime schema (called during startup)."""
    global _RUNTIME_SCHEMA
    _RUNTIME_SCHEMA = schema
    logger.debug(f"✓ Runtime schema loaded ({len(schema)} options)")


def get_option_schema() -> Dict:
    """Return the current option schema."""
    return _RUNTIME_SCHEMA or {}


def validate_option(key: str, value: str) -> Tuple[bool, str]:
    """
    Validate a single option against the runtime schema.

    Returns (is_valid, error_msg). Unknown options are treated as valid
    (permissive for forward compatibility with newer llama-server versions).
    """
    schema = get_option_schema()

    # If no schema loaded, be permissive
    if not schema:
        return True, ""

    # Unknown options are allowed (schema may be incomplete or outdated)
    if key not in schema:
        return True, ""

    spec = schema[key]

    try:
        if spec.get("type") == "int":
            int(value)
        elif spec.get("type") == "str":
            str(value)
        else:
            return False, f"Unknown type: {spec.get('type')}"
        return True, ""
    except (ValueError, TypeError) as e:
        return False, f"Invalid {spec.get('type', 'value')}: {e}"


def get_options_by_category() -> Dict:
    """Return options grouped by category."""
    schema = get_option_schema()
    by_cat = {}
    for key, spec in schema.items():
        cat = spec.get("category", "other")
        if cat not in by_cat:
            by_cat[cat] = {}
        by_cat[cat][key] = spec
    return by_cat
