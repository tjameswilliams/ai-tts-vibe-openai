"""Tests for the lifecycle manager — idle timeout and model load/unload."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vibevoice_tts_server.config import Settings


def _settings(**overrides) -> Settings:
    defaults = dict(device="cpu", dtype="float32", idle_timeout=300)
    defaults.update(overrides)
    return Settings(**defaults)


# ── ensure_model (async) ────────────────────────────────────────────────


class TestEnsureModel:
    async def test_loads_model_when_not_loaded(self):
        with (
            patch("vibevoice_tts_server.lifecycle.model") as mock_model,
            patch("vibevoice_tts_server.lifecycle._timer_task", None),
        ):
            mock_model.is_loaded.return_value = False
            mock_model.load_model = MagicMock()

            import vibevoice_tts_server.lifecycle as lc

            # Reset module state
            lc._last_use = 0.0
            lc._timer_task = None

            settings = _settings(idle_timeout=0)  # no timer
            await lc.ensure_model(settings)

            mock_model.load_model.assert_called_once_with(settings)

    async def test_skips_load_when_already_loaded(self):
        with (
            patch("vibevoice_tts_server.lifecycle.model") as mock_model,
            patch("vibevoice_tts_server.lifecycle._timer_task", None),
        ):
            mock_model.is_loaded.return_value = True

            import vibevoice_tts_server.lifecycle as lc

            lc._last_use = 0.0
            lc._timer_task = None

            settings = _settings(idle_timeout=0)
            await lc.ensure_model(settings)

            mock_model.load_model.assert_not_called()

    async def test_updates_last_use(self):
        with (
            patch("vibevoice_tts_server.lifecycle.model") as mock_model,
            patch("vibevoice_tts_server.lifecycle._timer_task", None),
        ):
            mock_model.is_loaded.return_value = True

            import vibevoice_tts_server.lifecycle as lc

            lc._last_use = 0.0
            lc._timer_task = None

            settings = _settings(idle_timeout=0)
            before = time.monotonic()
            await lc.ensure_model(settings)

            assert lc._last_use >= before


# ── ensure_model_sync ───────────────────────────────────────────────────


class TestEnsureModelSync:
    def test_loads_model_sync(self):
        with patch("vibevoice_tts_server.lifecycle.model") as mock_model:
            mock_model.is_loaded.return_value = False
            mock_model.load_model = MagicMock()

            import vibevoice_tts_server.lifecycle as lc

            settings = _settings()
            lc.ensure_model_sync(settings)

            mock_model.load_model.assert_called_once_with(settings)

    def test_skips_load_sync_when_loaded(self):
        with patch("vibevoice_tts_server.lifecycle.model") as mock_model:
            mock_model.is_loaded.return_value = True

            import vibevoice_tts_server.lifecycle as lc

            settings = _settings()
            lc.ensure_model_sync(settings)

            mock_model.load_model.assert_not_called()


# ── touch ───────────────────────────────────────────────────────────────


class TestTouch:
    async def test_touch_updates_last_use(self):
        import vibevoice_tts_server.lifecycle as lc

        lc._last_use = 0.0
        before = time.monotonic()
        await lc.touch()
        assert lc._last_use >= before


# ── idle watcher ────────────────────────────────────────────────────────


class TestIdleWatcher:
    async def test_unloads_after_timeout(self):
        """Model should be unloaded after idle_timeout seconds."""
        with patch("vibevoice_tts_server.lifecycle.model") as mock_model:
            mock_model.is_loaded.return_value = True
            mock_model.unload_model = MagicMock()

            import vibevoice_tts_server.lifecycle as lc

            # Set last_use far in the past
            lc._last_use = time.monotonic() - 1000

            settings = _settings(idle_timeout=1)

            # Run watcher — it should sleep 1s then unload
            await asyncio.wait_for(lc._idle_watcher(settings), timeout=3)

            mock_model.unload_model.assert_called_once()

    async def test_does_not_unload_if_recently_used(self):
        """Model should not be unloaded if recently used."""
        with patch("vibevoice_tts_server.lifecycle.model") as mock_model:
            mock_model.is_loaded.return_value = True

            import vibevoice_tts_server.lifecycle as lc

            settings = _settings(idle_timeout=2)

            # Keep resetting last_use so the watcher never fires
            async def keep_touching():
                for _ in range(5):
                    lc._last_use = time.monotonic()
                    await asyncio.sleep(0.3)

            touch_task = asyncio.create_task(keep_touching())

            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(lc._idle_watcher(settings), timeout=1.5)

            touch_task.cancel()
            try:
                await touch_task
            except asyncio.CancelledError:
                pass

            mock_model.unload_model.assert_not_called()

    async def test_no_timer_when_idle_timeout_zero(self):
        """_start_timer should be a no-op when idle_timeout is 0."""
        import vibevoice_tts_server.lifecycle as lc

        lc._timer_task = None
        settings = _settings(idle_timeout=0)
        lc._start_timer(settings)
        assert lc._timer_task is None

    async def test_timer_started_when_positive_timeout(self):
        """_start_timer should create an asyncio task."""
        with patch("vibevoice_tts_server.lifecycle.model") as mock_model:
            mock_model.is_loaded.return_value = False

            import vibevoice_tts_server.lifecycle as lc

            lc._timer_task = None
            settings = _settings(idle_timeout=60)
            lc._start_timer(settings)

            assert lc._timer_task is not None
            assert not lc._timer_task.done()

            # Clean up
            lc._timer_task.cancel()
            try:
                await lc._timer_task
            except asyncio.CancelledError:
                pass

    async def test_timer_cancelled_and_restarted(self):
        """Calling _start_timer again should cancel the previous task."""
        with patch("vibevoice_tts_server.lifecycle.model") as mock_model:
            mock_model.is_loaded.return_value = False

            import vibevoice_tts_server.lifecycle as lc

            lc._timer_task = None
            settings = _settings(idle_timeout=60)

            lc._start_timer(settings)
            first_task = lc._timer_task

            lc._start_timer(settings)
            second_task = lc._timer_task

            assert first_task is not second_task
            # Let the event loop process the cancellation
            await asyncio.sleep(0)
            assert first_task.cancelled()

            # Clean up
            second_task.cancel()
            try:
                await second_task
            except asyncio.CancelledError:
                pass


# ── MCP server tools ────────────────────────────────────────────────────


class TestMCPTools:
    def test_list_voices_tool(self):
        """list_voices MCP tool returns JSON with all voices."""
        import json
        from vibevoice_tts_server.mcp_server import list_voices

        result = list_voices()
        voices = json.loads(result)
        assert len(voices) == 7
        names = {v["name"] for v in voices}
        assert "alloy" in names

    def test_get_tts_status_not_loaded(self):
        """get_tts_status should report not loaded."""
        import json

        with patch("vibevoice_tts_server.mcp_server.model") as mock_model:
            mock_model.is_loaded.return_value = False

            from vibevoice_tts_server.mcp_server import get_tts_status

            result = json.loads(get_tts_status())
            assert result["model_loaded"] is False
            assert result["device"] is None

    def test_get_tts_status_loaded(self):
        """get_tts_status should report device/dtype when loaded."""
        import json

        with patch("vibevoice_tts_server.mcp_server.model") as mock_model:
            mock_model.is_loaded.return_value = True
            mock_model.get_device.return_value = "cuda"
            mock_model.get_dtype.return_value = "torch.bfloat16"

            from vibevoice_tts_server.mcp_server import get_tts_status

            result = json.loads(get_tts_status())
            assert result["model_loaded"] is True
            assert result["device"] == "cuda"
            assert result["dtype"] == "torch.bfloat16"

    def test_synthesize_speech_empty_text(self):
        """synthesize_speech should reject empty text."""
        from vibevoice_tts_server.mcp_server import synthesize_speech

        result = synthesize_speech(text="   ")
        assert "Error" in result
        assert "empty" in result

    def test_synthesize_speech_bad_format(self):
        """synthesize_speech should reject unsupported formats."""
        from vibevoice_tts_server.mcp_server import synthesize_speech

        result = synthesize_speech(text="Hello", response_format="ogg")
        assert "Error" in result
        assert "unsupported" in result

    def test_synthesize_speech_success(self, fake_audio, tmp_path):
        """synthesize_speech should write audio and return JSON."""
        import json

        audio, sr = fake_audio
        out_file = tmp_path / "test_output.wav"

        with (
            patch("vibevoice_tts_server.mcp_server.ensure_model_sync"),
            patch(
                "vibevoice_tts_server.mcp_server.model.generate_speech",
                return_value=(audio, sr),
            ),
        ):
            from vibevoice_tts_server.mcp_server import synthesize_speech

            result = synthesize_speech(
                text="Hello world",
                voice="alloy",
                output_path=str(out_file),
                response_format="wav",
            )

        data = json.loads(result)
        assert data["status"] == "ok"
        assert data["format"] == "wav"
        assert data["speaker"] == "Emma"
        assert out_file.exists()
        assert out_file.stat().st_size > 0

    def test_synthesize_speech_model_error(self):
        """synthesize_speech should handle model load errors gracefully."""
        with patch(
            "vibevoice_tts_server.mcp_server.ensure_model_sync",
            side_effect=RuntimeError("GPU OOM"),
        ):
            from vibevoice_tts_server.mcp_server import synthesize_speech

            result = synthesize_speech(text="Hello")
            assert "Error loading model" in result
            assert "GPU OOM" in result

    def test_synthesize_speech_generation_error(self, fake_audio):
        """synthesize_speech should handle generation errors gracefully."""
        with (
            patch("vibevoice_tts_server.mcp_server.ensure_model_sync"),
            patch(
                "vibevoice_tts_server.mcp_server.model.generate_speech",
                side_effect=RuntimeError("CUDA error"),
            ),
        ):
            from vibevoice_tts_server.mcp_server import synthesize_speech

            result = synthesize_speech(text="Hello")
            assert "Error during synthesis" in result

    def test_synthesize_speech_default_output_path(self, fake_audio):
        """When no output_path given, should use ~/vibevoice_output.<ext>."""
        import json

        audio, sr = fake_audio

        with (
            patch("vibevoice_tts_server.mcp_server.ensure_model_sync"),
            patch(
                "vibevoice_tts_server.mcp_server.model.generate_speech",
                return_value=(audio, sr),
            ),
        ):
            from vibevoice_tts_server.mcp_server import synthesize_speech

            result = synthesize_speech(text="Hello", response_format="pcm")

        data = json.loads(result)
        assert data["output_path"].endswith(".pcm")
