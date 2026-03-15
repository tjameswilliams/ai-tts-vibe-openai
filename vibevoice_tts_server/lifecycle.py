"""Idle timeout / model lifecycle manager."""

from __future__ import annotations

import asyncio
import logging
import threading
import time

from .config import Settings
from . import model

logger = logging.getLogger(__name__)

_lock = asyncio.Lock()
_last_use: float = 0.0
_timer_task: asyncio.Task | None = None
_settings: Settings | None = None

# Thread-based lock for MCP (sync) usage
_sync_lock = threading.Lock()


async def ensure_model(settings: Settings) -> None:
    """Load the model if not already loaded. Resets idle timer."""
    global _last_use, _settings
    _settings = settings

    async with _lock:
        if not model.is_loaded():
            await asyncio.to_thread(model.load_model, settings)
        _last_use = time.monotonic()
        _start_timer(settings)


def ensure_model_sync(settings: Settings) -> None:
    """Synchronous version for MCP server."""
    global _last_use, _settings
    _settings = settings

    with _sync_lock:
        if not model.is_loaded():
            model.load_model(settings)
        _last_use = time.monotonic()


async def touch() -> None:
    """Reset the idle timer (call on each request)."""
    global _last_use
    _last_use = time.monotonic()


def _start_timer(settings: Settings) -> None:
    """Start or restart the idle unload timer."""
    global _timer_task
    if settings.idle_timeout <= 0:
        return
    if _timer_task is not None and not _timer_task.done():
        _timer_task.cancel()
    _timer_task = asyncio.create_task(_idle_watcher(settings))


async def _idle_watcher(settings: Settings) -> None:
    """Background task that unloads the model after idle_timeout seconds."""
    while True:
        await asyncio.sleep(settings.idle_timeout)
        elapsed = time.monotonic() - _last_use
        if elapsed >= settings.idle_timeout and model.is_loaded():
            async with _lock:
                # Double-check after acquiring lock
                elapsed = time.monotonic() - _last_use
                if elapsed >= settings.idle_timeout and model.is_loaded():
                    logger.info(
                        "Model idle for %.0fs, unloading to free memory",
                        elapsed,
                    )
                    await asyncio.to_thread(model.unload_model)
                    return
