# VibeVoice TTS Server

Local OpenAI-compatible text-to-speech API server powered by [VibeVoice-7B](https://huggingface.co/vibevoice/VibeVoice-7B). Generates up to 45 minutes of speech in a single request, with voice cloning from reference audio.

## Features

- **OpenAI-compatible API** — drop-in replacement for `POST /v1/audio/speech`
- **Voice cloning** — upload reference audio to clone any speaker's voice
- **Long-form generation** — natively supports up to ~45 minutes per request (full podcast episodes)
- **On-demand model loading** — loads the 7B model on first request, auto-unloads after idle timeout to free VRAM
- **Multiple formats** — MP3, WAV, OPUS, FLAC, AAC, PCM
- **MCP server** — use as a Claude Code tool for speech synthesis
- **Platform detection** — CUDA > MPS > CPU with optional 4-bit quantization

## Quickstart

### 1. Install

```bash
pip install -e .

# With CUDA flash attention:
pip install -e ".[cuda]"

# With 4-bit quantization:
pip install -e ".[quant]"
```

Requires `ffmpeg` for MP3/OPUS/AAC encoding:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### 2. Start the server

```bash
vibevoice-tts-server
```

The model downloads on first request (~14 GB) and loads into VRAM/RAM. Subsequent requests reuse the loaded model.

Options:

```bash
vibevoice-tts-server --device cuda --port 8100 --idle-timeout 600
```

### 3. Generate speech

```bash
curl -X POST http://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world, this is VibeVoice!", "voice": "alloy"}' \
  --output hello.mp3
```

### 4. Voice cloning with reference audio

Upload a reference audio file to clone a speaker's voice:

```bash
curl -X POST http://localhost:8100/v1/audio/speech/upload \
  -F "input=Welcome to the show, I'm your host." \
  -F "voice=alloy" \
  -F "response_format=wav" \
  -F "reference_audio=@speaker_sample.wav" \
  --output cloned.wav
```

Or pass a file path via the JSON endpoint:

```bash
curl -X POST http://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Welcome to the show.",
    "voice": "alloy",
    "instructions": "{\"reference_audio\": \"/path/to/speaker_sample.wav\"}"
  }' \
  --output cloned.mp3
```

## API Reference

### `POST /v1/audio/speech`

JSON body (OpenAI-compatible):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | string | *required* | Text to synthesize |
| `voice` | string | `"alloy"` | Voice preset or speaker name |
| `model` | string | `"vibevoice-tts"` | Model identifier |
| `response_format` | string | `"mp3"` | `mp3`, `wav`, `opus`, `flac`, `aac`, `pcm` |
| `speed` | float | `1.0` | Speed multiplier (0.25 - 4.0) |
| `instructions` | string | `null` | JSON string with advanced params (see below) |

**Instructions JSON fields:**

| Field | Type | Description |
|-------|------|-------------|
| `reference_audio` | string | Path to reference audio for voice cloning |
| `cfg_scale` | float | Classifier-free guidance scale (default: 1.3) |
| `n_diffusion_steps` | int | Diffusion denoising steps (default: 10) |
| `max_new_tokens` | int | Audio token limit at 7.5 Hz (0 = unlimited) |

Returns raw audio bytes with the appropriate `Content-Type` header.

### `POST /v1/audio/speech/upload`

Multipart form — same fields as above, plus:

| Field | Type | Description |
|-------|------|-------------|
| `reference_audio` | file | Audio file upload for voice cloning |

### `GET /v1/audio/voices`

List available voice presets.

### `GET /v1/models`

List available models.

### `GET /health`

Server status, model load state, device info.

## Voice Presets

| OpenAI Name | VibeVoice Speaker |
|-------------|-------------------|
| alloy | Emma |
| echo | Carter |
| fable | Davis |
| onyx | Mike |
| nova | Grace |
| shimmer | Frank |
| sage | Samuel |

## Configuration

All settings can be set via environment variables with the `VIBEVOICE_TTS_` prefix:

```bash
VIBEVOICE_TTS_HOST=0.0.0.0
VIBEVOICE_TTS_PORT=8100
VIBEVOICE_TTS_MODEL_ID=vibevoice/VibeVoice-7B
VIBEVOICE_TTS_DEVICE=auto          # auto, cuda, mps, cpu
VIBEVOICE_TTS_DTYPE=auto           # auto, bfloat16, float32
VIBEVOICE_TTS_IDLE_TIMEOUT=300     # seconds before unloading model (0 = never)
VIBEVOICE_TTS_MAX_NEW_TOKENS=0     # 0 = unlimited; tokens are audio frames at 7.5 Hz
VIBEVOICE_TTS_CFG_SCALE=1.3
VIBEVOICE_TTS_N_DIFFUSION_STEPS=10
VIBEVOICE_TTS_QUANTIZE_4BIT=false
```

## MCP Server

Use as a [Claude Code](https://claude.ai/claude-code) tool:

```bash
vibevoice-tts-mcp
```

Add to your Claude Code MCP config:

```json
{
  "mcpServers": {
    "vibevoice-tts": {
      "command": "vibevoice-tts-mcp",
      "args": ["--device", "auto"]
    }
  }
}
```

**Tools:** `synthesize_speech`, `list_voices`, `get_tts_status`

## Running Tests

```bash
pip install -e ".[test]"
pytest tests/ -v
```

## Audio Token Math

VibeVoice generates audio tokens at **7.5 Hz** (7.5 tokens per second of audio). The 7B model has a 32K context window shared between text input tokens and audio output tokens.

| Tokens | Duration |
|--------|----------|
| 450 | ~1 minute |
| 4,500 | ~10 minutes |
| 13,500 | ~30 minutes |
| 20,250 | ~45 minutes |

By default, `max_new_tokens=0` (unlimited), allowing the model to generate until it finishes the input text naturally.

## Licensing

This server code is released under the [MIT License](LICENSE).

**Model license note:** Microsoft released VibeVoice-7B under the MIT License. However, Microsoft's model card states the model is "limited to research purpose use" and later removed the TTS code from their official repository citing misuse concerns. Community forks and model weights remain available under MIT. Users should review the [model card](https://huggingface.co/vibevoice/VibeVoice-7B) and applicable terms before deploying in production.

This project is an independent wrapper and is not affiliated with or endorsed by Microsoft.
