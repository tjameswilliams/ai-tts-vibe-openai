"""MCP server exposing VibeVoice-TTS synthesis as tools."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .config import Settings
from .formats import FORMAT_INFO, encode_audio
from .lifecycle import ensure_model_sync
from .voices import list_voices as _list_voices, resolve_voice
from . import model

logger = logging.getLogger(__name__)

mcp = FastMCP("vibevoice-tts")

_inference_lock = threading.Lock()


def _ensure() -> Settings:
    """Load model if needed, return settings."""
    settings = Settings()
    ensure_model_sync(settings)
    return settings


@mcp.tool()
def synthesize_speech(
    text: str,
    voice: str = "alloy",
    output_path: str | None = None,
    response_format: str = "mp3",
    speed: float = 1.0,
    cfg_scale: float | None = None,
    n_diffusion_steps: int | None = None,
) -> str:
    """Generate speech from text using VibeVoice-TTS.

    Args:
        text: The text to synthesize into speech.
        voice: Voice name (alloy, echo, fable, onyx, nova, shimmer, sage) or path to reference audio.
        output_path: Where to save the audio file. If not provided, saves to ~/vibevoice_output.<format>.
        response_format: Audio format: mp3, wav, opus, flac, aac, pcm (default: mp3).
        speed: Speech speed multiplier, 0.25 to 4.0 (default: 1.0).
        cfg_scale: Classifier-free guidance scale (default: from server config).
        n_diffusion_steps: Number of diffusion steps (default: from server config).
    """
    if not text.strip():
        return "Error: text must not be empty"

    if response_format not in FORMAT_INFO:
        return f"Error: unsupported format '{response_format}'. Use: {', '.join(FORMAT_INFO)}"

    try:
        settings = _ensure()
    except Exception as e:
        return f"Error loading model: {e}"

    speaker = resolve_voice(voice)

    # Check if voice is a file path for reference audio
    reference_audio = None
    voice_path = Path(voice)
    if voice_path.exists() and voice_path.is_file():
        reference_audio = voice
        speaker = "Emma"  # Default speaker when using reference audio

    try:
        with _inference_lock:
            audio, sample_rate = model.generate_speech(
                text,
                speaker,
                settings=settings,
                reference_audio=reference_audio,
                cfg_scale=cfg_scale,
                n_diffusion_steps=n_diffusion_steps,
                speed=speed,
            )
    except Exception as e:
        return f"Error during synthesis: {e}"

    audio_bytes, content_type = encode_audio(audio, sample_rate, response_format)

    # Determine output path
    _, ext = FORMAT_INFO[response_format]
    if output_path is None:
        out = Path.home() / f"vibevoice_output{ext}"
    else:
        out = Path(output_path)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(audio_bytes)

    return json.dumps({
        "status": "ok",
        "output_path": str(out),
        "format": response_format,
        "content_type": content_type,
        "size_bytes": len(audio_bytes),
        "voice": voice,
        "speaker": speaker,
    }, indent=2)


@mcp.tool()
def list_voices() -> str:
    """List available voice presets for VibeVoice-TTS.

    Returns a JSON list of voice names and their VibeVoice speaker mappings.
    """
    return json.dumps(_list_voices(), indent=2)


@mcp.tool()
def get_tts_status() -> str:
    """Check the current status of the VibeVoice-TTS server.

    Returns model loaded state, device, dtype, and version info.
    """
    status = {
        "model_loaded": model.is_loaded(),
        "device": model.get_device() if model.is_loaded() else None,
        "dtype": model.get_dtype() if model.is_loaded() else None,
        "version": "0.1.0",
    }
    return json.dumps(status, indent=2)
