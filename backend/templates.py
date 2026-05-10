"""Shared Jinja2 template environment and helpers."""

from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup

PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATE_DIR = PROJECT_ROOT / "frontend" / "templates"

jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=True,
)


def status_badge(state: str) -> Markup:
    """Return an HTML status-badge span for a given state string."""
    MAP = {
        "idle": '<span class="status-badge status-idle">Idle</span>',
        "loading": '<span class="status-badge status-loading">Loading</span>',
        "running": '<span class="status-badge status-running">Running</span>',
        "failed": '<span class="status-badge status-failed">Failed</span>',
    }
    return Markup(MAP.get(state, '<span class="status-badge status-unknown">Unknown</span>'))


jinja_env.filters["status_badge"] = status_badge
