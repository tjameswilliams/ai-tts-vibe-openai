"""API tests for FastAPI endpoints with mocked model."""

from __future__ import annotations

import io
import json
from unittest.mock import patch, AsyncMock

import numpy as np
import pytest
import soundfile as sf
from httpx import ASGITransport, AsyncClient

from vibevoice_tts_server.config import Settings


@pytest.fixture
def fake_audio():
    """Short sine wave."""
    sr = 24000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return audio, sr


@pytest.fixture
def ref_audio_wav(fake_audio):
    """Generate a small WAV file in memory for upload tests."""
    audio, sr = fake_audio
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
async def client(fake_audio):
    """AsyncClient with model fully mocked out."""
    audio, sr = fake_audio

    with (
        patch("vibevoice_tts_server.server.ensure_model", new_callable=AsyncMock),
        patch("vibevoice_tts_server.server.touch", new_callable=AsyncMock),
        patch("vibevoice_tts_server.server.generate_speech", return_value=(audio, sr)),
        patch("vibevoice_tts_server.server.is_loaded", return_value=True),
        patch("vibevoice_tts_server.server.get_device", return_value="cpu"),
        patch("vibevoice_tts_server.server.get_dtype", return_value="torch.float32"),
    ):
        from vibevoice_tts_server.server import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# ── POST /v1/audio/speech ───────────────────────────────────────────────


class TestCreateSpeech:
    async def test_basic_speech(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Hello world", "voice": "alloy"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/mpeg"
        assert len(resp.content) > 0

    async def test_wav_format(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Test", "response_format": "wav"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert resp.content[:4] == b"RIFF"

    async def test_pcm_format(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Test", "response_format": "pcm"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/pcm"

    async def test_flac_format(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Test", "response_format": "flac"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/flac"

    async def test_unsupported_format_returns_400(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Test", "response_format": "ogg"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert body["error"]["type"] == "invalid_request_error"

    async def test_empty_input_returns_400(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "   ", "voice": "alloy"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "empty" in body["error"]["message"].lower()

    async def test_missing_input_returns_422(self, client):
        resp = await client.post("/v1/audio/speech", json={"voice": "alloy"})
        assert resp.status_code == 422

    async def test_speed_bounds(self, client):
        # speed too low
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Test", "speed": 0.1},
        )
        assert resp.status_code == 422

        # speed too high
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Test", "speed": 5.0},
        )
        assert resp.status_code == 422

    async def test_valid_speed(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Test", "speed": 2.0, "response_format": "wav"},
        )
        assert resp.status_code == 200

    async def test_instructions_json_parsed(self, client):
        """Instructions field with valid JSON should not error."""
        resp = await client.post(
            "/v1/audio/speech",
            json={
                "input": "Test",
                "instructions": json.dumps({"cfg_scale": 2.0, "n_diffusion_steps": 20}),
            },
        )
        assert resp.status_code == 200

    async def test_instructions_invalid_json_ignored(self, client):
        """Non-JSON instructions should be silently ignored."""
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Test", "instructions": "not json"},
        )
        assert resp.status_code == 200

    async def test_default_model_field(self, client):
        """Model field defaults to vibevoice-tts."""
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Test"},
        )
        assert resp.status_code == 200

    async def test_all_voices_accepted(self, client):
        """Every mapped voice name should work."""
        for voice in ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "sage"]:
            resp = await client.post(
                "/v1/audio/speech",
                json={"input": "Test", "voice": voice, "response_format": "pcm"},
            )
            assert resp.status_code == 200, f"Voice {voice} failed"


# ── POST /v1/audio/speech/upload ─────────────────────────────────────────


class TestCreateSpeechUpload:
    async def test_basic_upload_no_ref_audio(self, client):
        """Upload endpoint works without reference audio."""
        resp = await client.post(
            "/v1/audio/speech/upload",
            data={"input": "Hello world", "voice": "alloy", "response_format": "wav"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"

    async def test_upload_with_reference_audio(self, client, ref_audio_wav):
        """Upload endpoint accepts a reference audio file."""
        resp = await client.post(
            "/v1/audio/speech/upload",
            data={"input": "Clone this voice", "voice": "alloy", "response_format": "wav"},
            files={"reference_audio": ("speaker.wav", ref_audio_wav, "audio/wav")},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"

    async def test_upload_unsupported_format(self, client):
        resp = await client.post(
            "/v1/audio/speech/upload",
            data={"input": "Test", "response_format": "ogg"},
        )
        assert resp.status_code == 400

    async def test_upload_empty_input(self, client):
        resp = await client.post(
            "/v1/audio/speech/upload",
            data={"input": "   "},
        )
        assert resp.status_code == 400

    async def test_upload_speed_too_low(self, client):
        resp = await client.post(
            "/v1/audio/speech/upload",
            data={"input": "Test", "speed": "0.1"},
        )
        assert resp.status_code == 400

    async def test_upload_speed_too_high(self, client):
        resp = await client.post(
            "/v1/audio/speech/upload",
            data={"input": "Test", "speed": "5.0"},
        )
        assert resp.status_code == 400

    async def test_upload_with_instructions(self, client):
        """Instructions JSON is parsed on the upload endpoint."""
        resp = await client.post(
            "/v1/audio/speech/upload",
            data={
                "input": "Test",
                "instructions": json.dumps({"cfg_scale": 2.0}),
                "response_format": "pcm",
            },
        )
        assert resp.status_code == 200

    async def test_upload_pcm_format(self, client):
        resp = await client.post(
            "/v1/audio/speech/upload",
            data={"input": "Test", "response_format": "pcm"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/pcm"

    async def test_upload_mp3_format(self, client):
        resp = await client.post(
            "/v1/audio/speech/upload",
            data={"input": "Test", "response_format": "mp3"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/mpeg"


# ── GET /v1/models ──────────────────────────────────────────────────────


class TestListModels:
    async def test_list_models(self, client):
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert len(body["data"]) == 1
        assert body["data"][0]["id"] == "vibevoice-tts"
        assert body["data"][0]["owned_by"] == "microsoft"


# ── GET /v1/audio/voices ────────────────────────────────────────────────


class TestListVoicesEndpoint:
    async def test_list_voices(self, client):
        resp = await client.get("/v1/audio/voices")
        assert resp.status_code == 200
        body = resp.json()
        assert "voices" in body
        assert len(body["voices"]) == 7
        names = {v["name"] for v in body["voices"]}
        assert "alloy" in names
        assert "sage" in names


# ── GET /health ─────────────────────────────────────────────────────────


class TestHealth:
    async def test_health(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert body["model_loaded"] is True
        assert body["device"] == "cpu"
        assert body["dtype"] == "torch.float32"
        assert body["model"] == "microsoft/VibeVoice-7B-hf"
