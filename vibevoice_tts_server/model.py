"""Torch backend: load/generate/unload VibeVoice-7B for TTS."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import Settings

logger = logging.getLogger(__name__)

_model = None
_processor = None
_device: str | None = None
_dtype: torch.dtype | None = None
_loaded: bool = False

SAMPLE_RATE = 24000


@dataclass
class PlatformInfo:
    device: str
    dtype: torch.dtype
    attn_implementation: str


def detect_platform(settings: Settings) -> PlatformInfo:
    """Detect the best device/dtype/attention for the current hardware."""
    if settings.device != "auto":
        device = settings.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if settings.dtype != "auto":
        dtype = getattr(torch, settings.dtype)
    elif device == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    if device == "cuda":
        attn = "flash_attention_2"
    else:
        attn = "eager"

    return PlatformInfo(device=device, dtype=dtype, attn_implementation=attn)


def load_model(settings: Settings) -> None:
    """Download (if needed) and load the model + processor."""
    global _model, _processor, _device, _dtype, _loaded
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    platform = detect_platform(settings)
    _device = platform.device
    _dtype = platform.dtype

    logger.info(
        "Loading model %s on %s (%s, attn=%s)",
        settings.model_id,
        platform.device,
        platform.dtype,
        platform.attn_implementation,
    )

    _processor = VibeVoiceProcessor.from_pretrained(
        settings.model_id,
        cache_dir=settings.cache_dir,
    )

    model_kwargs = {
        "torch_dtype": platform.dtype,
        "attn_implementation": platform.attn_implementation,
        "cache_dir": settings.cache_dir,
    }

    if settings.quantize_4bit and platform.device == "cuda":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=platform.dtype,
        )
    else:
        model_kwargs["device_map"] = platform.device

    _model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        settings.model_id, **model_kwargs
    )
    _model.eval()

    # Set default diffusion steps
    _model.set_ddpm_inference_steps(num_steps=settings.n_diffusion_steps)

    _loaded = True
    logger.info("Model loaded successfully")


def unload_model() -> None:
    """Unload model and free GPU memory."""
    global _model, _processor, _device, _dtype, _loaded

    if not _loaded:
        return

    logger.info("Unloading model to free memory")
    _model = None
    _processor = None
    _loaded = False

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Model unloaded")


def is_loaded() -> bool:
    return _loaded


def generate_speech(
    text: str,
    speaker: str,
    *,
    settings: Settings,
    reference_audio: str | Path | None = None,
    cfg_scale: float | None = None,
    n_diffusion_steps: int | None = None,
    max_new_tokens: int | None = None,
    speed: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Generate speech audio from text.

    Returns (audio_array, sample_rate).
    """
    if _model is None or _processor is None:
        raise RuntimeError("Model not loaded — call load_model() first")

    _cfg = cfg_scale if cfg_scale is not None else settings.cfg_scale
    _steps = n_diffusion_steps if n_diffusion_steps is not None else settings.n_diffusion_steps
    _max_tokens = max_new_tokens if max_new_tokens is not None else settings.max_new_tokens

    # Update diffusion steps if different from current
    _model.set_ddpm_inference_steps(num_steps=_steps)

    # Format text with speaker prefix for VibeVoice
    # VibeVoice expects: "Speaker 1: text\n"
    script = f"Speaker 1: {text}"

    # Prepare voice samples for cloning
    voice_samples: list[str] = []
    is_prefill = False
    if reference_audio is not None:
        ref_path = str(reference_audio)
        if Path(ref_path).exists():
            voice_samples = [ref_path]
            is_prefill = True

    # Process inputs
    inputs = _processor(
        text=[script],
        voice_samples=[voice_samples] if voice_samples else None,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # Move tensors to device
    inputs = {k: v.to(_device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    # Build generation kwargs
    # max_new_tokens are audio frames at 7.5 Hz (e.g. 20250 tokens ≈ 45 min).
    # 0 or None = unlimited — model generates until it finishes the script.
    gen_kwargs: dict = {
        "cfg_scale": _cfg,
        "tokenizer": _processor.tokenizer,
        "is_prefill": is_prefill,
    }
    if _max_tokens and _max_tokens > 0:
        gen_kwargs["max_new_tokens"] = _max_tokens

    with torch.no_grad():
        outputs = _model.generate(**inputs, **gen_kwargs)

    # Extract audio from VibeVoiceGenerationOutput
    audio_tensor = outputs.speech_outputs[0]

    # Convert to numpy
    if torch.is_tensor(audio_tensor):
        audio = audio_tensor.cpu().float().numpy()
    else:
        audio = np.asarray(audio_tensor, dtype=np.float32)

    # Squeeze extra dimensions
    audio = audio.squeeze()

    # Apply speed adjustment if needed
    if speed != 1.0 and speed > 0:
        import scipy.signal
        target_len = int(len(audio) / speed)
        if target_len > 0:
            audio = scipy.signal.resample(audio, target_len)

    return audio, SAMPLE_RATE


def get_device() -> str:
    return _device or "unknown"


def get_dtype() -> str:
    return str(_dtype) if _dtype else "unknown"
