"""Event bus for WebSocket broadcasts."""

import logging
from typing import Set
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self):
        self._connections: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._connections.add(ws)
        logger.debug(f"WS client connected. Total: {len(self._connections)}")

    def disconnect(self, ws: WebSocket):
        self._connections.discard(ws)
        logger.debug(f"WS client disconnected. Total: {len(self._connections)}")

    async def broadcast(self, event: dict):
        if not self._connections:
            return
        dead = set()
        for ws in list(self._connections):
            try:
                await ws.send_json(event)
            except Exception:
                dead.add(ws)
        self._connections -= dead


event_bus = EventBus()
