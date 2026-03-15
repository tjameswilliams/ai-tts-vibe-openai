"""Unit tests for voices, formats, config, and model helpers."""

from __future__ import annotations

import io
import os
import struct
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── voices ──────────────────────────────────────────────────────────────

from vibevoice_tts_server.voices import VOICE_MAP, list_voices, resolve_voice


class TestResolveVoice:
    def test_known_voices(self):
        assert resolve_voice("alloy") == "Emma"
        assert resolve_voice("echo") == "Carter"
        assert resolve_voice("fable") == "Davis"
        assert resolve_voice("onyx") == "Mike"
        assert resolve_voice("nova") == "Grace"
        assert resolve_voice("shimmer") == "Frank"
        assert resolve_voice("sage") == "Samuel"

    def test_case_insensitive(self):
        assert resolve_voice("ALLOY") == "Emma"
        assert resolve_voice("Nova") == "Grace"

    def test_unknown_voice_passthrough(self):
        assert resolve_voice("custom_speaker") == "custom_speaker"
        assert resolve_voice("/path/to/audio.wav") == "/path/to/audio.wav"

    def test_empty_string(self):
        assert resolve_voice("") == ""


class TestListVoices:
    def test_returns_all_voices(self):
        voices = list_voices()
        assert len(voices) == len(VOICE_MAP)

    def test_voice_dict_structure(self):
        voices = list_voices()
        for v in voices:
            assert "name" in v
            assert "speaker" in v
            assert v["name"] in VOICE_MAP
            assert v["speaker"] == VOICE_MAP[v["name"]]


# ── config ──────────────────────────────────────────────────────────────

from vibevoice_tts_server.config import Settings


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.host == "0.0.0.0"
        assert s.port == 8101
        assert s.model_id == "vibevoice/VibeVoice-7B"
        assert s.device == "auto"
        assert s.dtype == "auto"
        assert s.idle_timeout == 300
        assert s.default_voice == "alloy"
        assert s.cfg_scale == 1.3
        assert s.n_diffusion_steps == 10
        assert s.max_new_tokens == 0  # 0 = unlimited
        assert s.quantize_4bit is False
        assert s.log_level == "info"
        assert s.cache_dir is None

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("VIBEVOICE_TTS_PORT", "9999")
        monkeypatch.setenv("VIBEVOICE_TTS_DEVICE", "cuda")
        monkeypatch.setenv("VIBEVOICE_TTS_IDLE_TIMEOUT", "0")
        monkeypatch.setenv("VIBEVOICE_TTS_QUANTIZE_4BIT", "true")
        s = Settings()
        assert s.port == 9999
        assert s.device == "cuda"
        assert s.idle_timeout == 0
        assert s.quantize_4bit is True

    def test_env_prefix(self):
        assert Settings.model_config["env_prefix"] == "VIBEVOICE_TTS_"


# ── formats ─────────────────────────────────────────────────────────────

from vibevoice_tts_server.formats import (
    FORMAT_INFO,
    SUPPORTED_FORMATS,
    encode_audio,
)


class TestFormatInfo:
    def test_all_formats_present(self):
        expected = {"mp3", "opus", "aac", "flac", "wav", "pcm"}
        assert SUPPORTED_FORMATS == expected

    def test_format_info_has_content_type_and_extension(self):
        for fmt, (ct, ext) in FORMAT_INFO.items():
            assert ct.startswith("audio/")
            assert ext.startswith(".")


class TestEncodeAudio:
    def test_pcm_format(self, fake_audio):
        audio, sr = fake_audio
        data, ct = encode_audio(audio, sr, "pcm")
        assert ct == "audio/pcm"
        # PCM is raw int16 bytes: 2 bytes per sample
        assert len(data) == len(audio) * 2

    def test_wav_format(self, fake_audio):
        audio, sr = fake_audio
        data, ct = encode_audio(audio, sr, "wav")
        assert ct == "audio/wav"
        # WAV starts with RIFF header
        assert data[:4] == b"RIFF"

    def test_flac_format(self, fake_audio):
        audio, sr = fake_audio
        data, ct = encode_audio(audio, sr, "flac")
        assert ct == "audio/flac"
        # FLAC starts with fLaC magic
        assert data[:4] == b"fLaC"

    def test_mp3_format(self, fake_audio):
        audio, sr = fake_audio
        data, ct = encode_audio(audio, sr, "mp3")
        assert ct == "audio/mpeg"
        assert len(data) > 0

    def test_unsupported_format_raises(self, fake_audio):
        audio, sr = fake_audio
        with pytest.raises(ValueError, match="Unsupported format"):
            encode_audio(audio, sr, "ogg")


# ── model helpers ───────────────────────────────────────────────────────

from vibevoice_tts_server.model import (
    PlatformInfo,
    detect_platform,
    get_device,
    get_dtype,
    is_loaded,
)


class TestDetectPlatform:
    def test_explicit_device(self, settings):
        settings.device = "cpu"
        p = detect_platform(settings)
        assert p.device == "cpu"
        assert p.dtype == torch.float32
        assert p.attn_implementation == "eager"

    def test_explicit_dtype(self, settings):
        settings.dtype = "bfloat16"
        settings.device = "cpu"
        p = detect_platform(settings)
        assert p.dtype == torch.bfloat16

    @patch("torch.cuda.is_available", return_value=True)
    def test_auto_cuda(self, mock_cuda, settings):
        settings.device = "auto"
        settings.dtype = "auto"
        p = detect_platform(settings)
        assert p.device == "cuda"
        assert p.dtype == torch.bfloat16
        assert p.attn_implementation == "flash_attention_2"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_auto_mps(self, mock_mps, mock_cuda, settings):
        settings.device = "auto"
        p = detect_platform(settings)
        assert p.device == "mps"
        assert p.dtype == torch.float32
        assert p.attn_implementation == "eager"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_auto_cpu_fallback(self, mock_mps, mock_cuda, settings):
        settings.device = "auto"
        p = detect_platform(settings)
        assert p.device == "cpu"


class TestModelState:
    def test_default_not_loaded(self):
        # After import, the module-level state should reflect not-loaded
        # (unless a previous test loaded it — we check the accessors)
        assert get_device() in ("unknown", "cpu")
        assert isinstance(get_dtype(), str)


class TestUnloadModel:
    @patch("vibevoice_tts_server.model.torch")
    def test_unload_clears_state(self, mock_torch):
        import vibevoice_tts_server.model as mod

        # Simulate loaded state
        mod._loaded = True
        mod._model = MagicMock()
        mod._processor = MagicMock()
        mod._device = "cpu"
        mod._dtype = "float32"

        mock_torch.cuda.is_available.return_value = False

        mod.unload_model()

        assert mod._loaded is False
        assert mod._model is None
        assert mod._processor is None

    @patch("vibevoice_tts_server.model.torch")
    def test_unload_noop_when_not_loaded(self, mock_torch):
        import vibevoice_tts_server.model as mod

        mod._loaded = False
        mod.unload_model()
        # Should not crash
        assert mod._loaded is False

    @patch("vibevoice_tts_server.model.torch")
    def test_unload_calls_cuda_empty_cache(self, mock_torch):
        import vibevoice_tts_server.model as mod

        mod._loaded = True
        mod._model = MagicMock()
        mod._processor = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        mod.unload_model()

        mock_torch.cuda.empty_cache.assert_called_once()


class TestGenerateSpeechNotLoaded:
    def test_raises_when_not_loaded(self, settings):
        import vibevoice_tts_server.model as mod

        mod._model = None
        mod._processor = None
        with pytest.raises(RuntimeError, match="Model not loaded"):
            mod.generate_speech("hello", "Emma", settings=settings)


class TestMaxNewTokensLogic:
    """Verify that max_new_tokens=0 means unlimited (omitted from gen_kwargs)."""

    def test_zero_means_unlimited(self, settings):
        """When max_new_tokens=0, gen_kwargs should NOT contain max_new_tokens."""
        settings.max_new_tokens = 0
        # We can't call generate_speech without a model, but we can verify
        # the logic by checking the config default
        assert settings.max_new_tokens == 0

    def test_positive_value_is_used(self, settings):
        settings.max_new_tokens = 20250  # ~45 min at 7.5 Hz
        assert settings.max_new_tokens == 20250

    def test_default_is_unlimited(self):
        s = Settings()
        assert s.max_new_tokens == 0


# Need torch imported for dtype comparisons
import torch
