"""Shared fixtures for all tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vibevoice_tts_server.config import Settings


@pytest.fixture
def settings():
    """Default settings for testing."""
    return Settings(
        host="127.0.0.1",
        port=8100,
        model_id="microsoft/VibeVoice-7B-hf",
        device="cpu",
        dtype="float32",
        idle_timeout=300,
    )


@pytest.fixture
def fake_audio():
    """Generate a short sine wave for testing audio encoding."""
    sr = 24000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return audio, sr


@pytest.fixture
def mock_model(fake_audio):
    """Patch model module so no real model is loaded."""
    audio, sr = fake_audio
    with (
        patch("vibevoice_tts_server.model._loaded", True),
        patch("vibevoice_tts_server.model._device", "cpu"),
        patch("vibevoice_tts_server.model._dtype", "torch.float32"),
        patch("vibevoice_tts_server.model.is_loaded", return_value=True),
        patch("vibevoice_tts_server.model.get_device", return_value="cpu"),
        patch("vibevoice_tts_server.model.get_dtype", return_value="torch.float32"),
        patch("vibevoice_tts_server.model.load_model") as mock_load,
        patch("vibevoice_tts_server.model.unload_model") as mock_unload,
        patch("vibevoice_tts_server.model.generate_speech", return_value=(audio, sr)) as mock_gen,
    ):
        yield {
            "load": mock_load,
            "unload": mock_unload,
            "generate": mock_gen,
            "audio": audio,
            "sample_rate": sr,
        }
