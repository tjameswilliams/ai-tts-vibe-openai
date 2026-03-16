"""FastAPI app for OpenAI-compatible TTS API."""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings
from .formats import SUPPORTED_FORMATS, encode_audio
from .lifecycle import ensure_model, touch
from .model import generate_speech, get_device, get_dtype, is_loaded
from .voices import list_voices, resolve_voice

logger = logging.getLogger(__name__)
settings = Settings()

_inference_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VibeVoice TTS server starting (model loads on first request)")
    yield


app = FastAPI(title="VibeVoice-TTS Server", version=__version__, lifespan=lifespan)


class SpeechRequest(BaseModel):
    model: str = "vibevoice-tts"
    input: str
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instructions: str | None = None


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    if request.response_format not in SUPPORTED_FORMATS:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Unsupported response_format: {request.response_format}. "
                    f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
                    "type": "invalid_request_error",
                }
            },
        )

    if not request.input.strip():
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Input text must not be empty.",
                    "type": "invalid_request_error",
                }
            },
        )

    # Resolve voice name
    speaker = resolve_voice(request.voice)

    # Parse optional VibeVoice-specific params from instructions
    cfg_scale = None
    n_diffusion_steps = None
    max_new_tokens = None
    reference_audio = None

    if request.instructions:
        try:
            extra = json.loads(request.instructions)
            cfg_scale = extra.get("cfg_scale")
            n_diffusion_steps = extra.get("n_diffusion_steps")
            max_new_tokens = extra.get("max_new_tokens")
            # reference_audio can be a single path or a list of paths
            # (one per speaker for multi-speaker scripts)
            reference_audio = extra.get("reference_audio")
        except (json.JSONDecodeError, TypeError):
            pass  # Ignore non-JSON instructions

    # Ensure model is loaded (resets idle timer)
    await ensure_model(settings)

    async with _inference_lock:
        await touch()
        audio, sample_rate = await asyncio.to_thread(
            generate_speech,
            request.input,
            speaker,
            settings=settings,
            reference_audio=reference_audio,
            cfg_scale=cfg_scale,
            n_diffusion_steps=n_diffusion_steps,
            max_new_tokens=max_new_tokens,
            speed=request.speed,
        )

    audio_bytes, content_type = encode_audio(audio, sample_rate, request.response_format)

    return Response(content=audio_bytes, media_type=content_type)


@app.post("/v1/audio/speech/upload")
async def create_speech_with_upload(
    input: str = Form(...),
    voice: str = Form("alloy"),
    response_format: str = Form("mp3"),
    speed: float = Form(1.0),
    model: str = Form("vibevoice-tts"),
    instructions: str | None = Form(None),
    reference_audio: list[UploadFile] | None = File(None),
):
    """Generate speech with optional reference audio file uploads for voice cloning.

    For multi-speaker scripts, upload multiple reference_audio files — one per
    speaker, in order (Speaker 1, Speaker 2, etc.). The input text should use
    "Speaker 1: ...\nSpeaker 2: ..." format.

    For single-speaker, upload one file or omit entirely.
    """
    if response_format not in SUPPORTED_FORMATS:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Unsupported response_format: {response_format}. "
                    f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
                    "type": "invalid_request_error",
                }
            },
        )

    if not input.strip():
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Input text must not be empty.",
                    "type": "invalid_request_error",
                }
            },
        )

    if speed < 0.25 or speed > 4.0:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Speed must be between 0.25 and 4.0.",
                    "type": "invalid_request_error",
                }
            },
        )

    speaker = resolve_voice(voice)

    # Parse optional params from instructions
    cfg_scale = None
    n_diffusion_steps = None
    max_new_tokens = None
    ref_audio_paths_from_instructions = None

    if instructions:
        try:
            extra = json.loads(instructions)
            cfg_scale = extra.get("cfg_scale")
            n_diffusion_steps = extra.get("n_diffusion_steps")
            max_new_tokens = extra.get("max_new_tokens")
            ref_audio_paths_from_instructions = extra.get("reference_audio")
        except (json.JSONDecodeError, TypeError):
            pass

    # Handle reference audio uploads: save to temp files
    # Multiple files supported for multi-speaker (one per speaker, ordered)
    tmp_files: list[str] = []
    ref_audio_paths: str | list[str] | None = None

    has_uploads = (
        reference_audio is not None
        and len(reference_audio) > 0
        and reference_audio[0].filename
    )

    if has_uploads:
        for upload in reference_audio:
            if upload.filename:
                suffix = Path(upload.filename).suffix or ".wav"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(await upload.read())
                tmp.close()
                tmp_files.append(tmp.name)
        if len(tmp_files) == 1:
            ref_audio_paths = tmp_files[0]
        elif len(tmp_files) > 1:
            ref_audio_paths = tmp_files
    elif ref_audio_paths_from_instructions:
        ref_audio_paths = ref_audio_paths_from_instructions

    try:
        await ensure_model(settings)

        async with _inference_lock:
            await touch()
            audio, sample_rate = await asyncio.to_thread(
                generate_speech,
                input,
                speaker,
                settings=settings,
                reference_audio=ref_audio_paths,
                cfg_scale=cfg_scale,
                n_diffusion_steps=n_diffusion_steps,
                max_new_tokens=max_new_tokens,
                speed=speed,
            )

        audio_bytes, content_type = encode_audio(audio, sample_rate, response_format)
        return Response(content=audio_bytes, media_type=content_type)
    finally:
        for tmp_path in tmp_files:
            Path(tmp_path).unlink(missing_ok=True)


@app.get("/v1/audio/voices")
async def get_voices():
    return {"voices": list_voices()}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "vibevoice-tts",
                "object": "model",
                "created": 0,
                "owned_by": "microsoft",
            }
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": __version__,
        "model": settings.model_id,
        "model_loaded": is_loaded(),
        "device": get_device(),
        "dtype": get_dtype(),
    }
