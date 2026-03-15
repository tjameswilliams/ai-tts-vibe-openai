"""Voice name mapping from OpenAI voice names to VibeVoice speakers."""

from __future__ import annotations

# OpenAI voice name → VibeVoice speaker name
VOICE_MAP: dict[str, str] = {
    "alloy": "Emma",
    "echo": "Carter",
    "fable": "Davis",
    "onyx": "Mike",
    "nova": "Grace",
    "shimmer": "Frank",
    "sage": "Samuel",
}


def resolve_voice(voice: str) -> str:
    """Resolve an OpenAI voice name to a VibeVoice speaker.

    If voice is a known OpenAI name, return the mapped speaker.
    Otherwise return it as-is (could be a direct speaker name or file path).
    """
    return VOICE_MAP.get(voice.lower(), voice)


def list_voices() -> list[dict[str, str]]:
    """Return list of available voice presets."""
    return [
        {"name": openai_name, "speaker": vibe_name}
        for openai_name, vibe_name in VOICE_MAP.items()
    ]
