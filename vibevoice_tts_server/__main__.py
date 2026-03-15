"""CLI entry point for vibevoice-tts-server."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="VibeVoice-TTS OpenAI-compatible API server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8100, help="Bind port (default: 8100)")
    parser.add_argument("--device", default=None, help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--dtype", default=None, help="Dtype: auto, bfloat16, float32")
    parser.add_argument("--idle-timeout", type=int, default=None, help="Idle timeout in seconds (0=never unload)")
    parser.add_argument("--log-level", default="info", help="Log level (default: info)")
    args = parser.parse_args()

    # Set env vars so Settings picks them up
    os.environ.setdefault("VIBEVOICE_TTS_HOST", args.host)
    os.environ.setdefault("VIBEVOICE_TTS_PORT", str(args.port))
    os.environ.setdefault("VIBEVOICE_TTS_LOG_LEVEL", args.log_level)
    if args.device:
        os.environ["VIBEVOICE_TTS_DEVICE"] = args.device
    if args.dtype:
        os.environ["VIBEVOICE_TTS_DTYPE"] = args.dtype
    if args.idle_timeout is not None:
        os.environ["VIBEVOICE_TTS_IDLE_TIMEOUT"] = str(args.idle_timeout)

    import logging

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    import uvicorn

    uvicorn.run(
        "vibevoice_tts_server.server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
