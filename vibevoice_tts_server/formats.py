"""Audio format conversion for TTS output."""

from __future__ import annotations

import io
import logging

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Format → (Content-Type, file extension)
FORMAT_INFO: dict[str, tuple[str, str]] = {
    "mp3": ("audio/mpeg", ".mp3"),
    "opus": ("audio/opus", ".opus"),
    "aac": ("audio/aac", ".aac"),
    "flac": ("audio/flac", ".flac"),
    "wav": ("audio/wav", ".wav"),
    "pcm": ("audio/pcm", ".pcm"),
}

SUPPORTED_FORMATS = set(FORMAT_INFO.keys())


def encode_audio(
    audio: np.ndarray,
    sample_rate: int,
    fmt: str,
) -> tuple[bytes, str]:
    """Encode audio array to the requested format.

    Returns (audio_bytes, content_type).
    """
    if fmt not in FORMAT_INFO:
        raise ValueError(f"Unsupported format: {fmt}. Use: {', '.join(SUPPORTED_FORMATS)}")

    content_type, _ = FORMAT_INFO[fmt]

    if fmt == "pcm":
        # Raw 16-bit signed little-endian PCM
        pcm_data = (audio * 32767).astype(np.int16)
        return pcm_data.tobytes(), content_type

    if fmt == "wav":
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue(), content_type

    if fmt == "flac":
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="FLAC")
        return buf.getvalue(), content_type

    # MP3, OPUS, AAC — use pydub + ffmpeg
    from pydub import AudioSegment

    # First write to WAV in memory
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    wav_buf.seek(0)

    segment = AudioSegment.from_wav(wav_buf)

    out_buf = io.BytesIO()
    export_format = fmt if fmt != "aac" else "adts"
    segment.export(out_buf, format=export_format)
    return out_buf.getvalue(), content_type
