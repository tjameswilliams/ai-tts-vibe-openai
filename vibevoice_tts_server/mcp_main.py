"""CLI entry point for vibevoice-tts-mcp server."""

import argparse
import logging
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="VibeVoice-TTS MCP server")
    parser.add_argument("--device", default=None, help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--dtype", default=None, help="Dtype: auto, bfloat16, float32")
    parser.add_argument("--log-level", default="warning", help="Log level (default: warning)")
    args = parser.parse_args()

    # Set env vars so Settings picks them up
    if args.device:
        os.environ["VIBEVOICE_TTS_DEVICE"] = args.device
    if args.dtype:
        os.environ["VIBEVOICE_TTS_DTYPE"] = args.dtype

    # All logging must go to stderr — stdout is reserved for MCP JSON-RPC
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from .mcp_server import mcp

    mcp.run()


if __name__ == "__main__":
    main()
