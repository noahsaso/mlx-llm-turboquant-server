#!/usr/bin/env python3

"""
mlx-llm-turboquant-server

MLX inference server with:
  • TurboQuant 3-bit KV cache compression  (arozanov/turboquant-mlx)
  • Speculative decoding with a small draft model

Exposes an OpenAI-compatible API (same as mlx_lm.server).

────────────────────────────────────────────────────────────────────
Quick start:
  uv run mlx-llm-turboquant-server

With explicit model + draft model + tuning:
  uv run mlx-llm-turboquant-server \
    --model mlx-community/Qwen2.5-32B-Instruct-4bit \
    --draft-model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
    --num-draft-tokens 4 \
    --turboquant-bits 3 \
    --fp16-layers 0 \
    --port 8080

Test:
  curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
      "messages": [{"role": "user", "content": "Hello!"}],
      "stream": true
    }'
────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
import sys

# ═══════════════════════════════════════════════════════════════════
#  1.  Apply TurboQuant patches BEFORE importing server internals
# ═══════════════════════════════════════════════════════════════════

try:
    from turboquant_mlx import make_adaptive_cache, apply_patch
except ImportError:
    print(
        "ERROR: turboquant_mlx not installed.\n"
        "  pip install git+https://github.com/arozanov/turboquant-mlx.git",
        file=sys.stderr,
    )
    sys.exit(1)

# Monkey-patch mlx-lm's scaled-dot-product attention to route through
# the fused Metal dequant kernels when it encounters TurboQuant caches.
apply_patch()

import mlx_lm.models.cache as cache_module

# Keep originals for fallback
_original_make_prompt_cache = cache_module.make_prompt_cache
_original_can_trim = cache_module.can_trim_prompt_cache

# ── Globals set from CLI before the server boots ──
_tq_bits: float = 3
_tq_fp16_layers: int = 0


def _turboquant_make_prompt_cache(model, max_kv_size=None, **kwargs):
    """
    Drop-in replacement for ``mlx_lm.models.cache.make_prompt_cache``.

    Returns a list whose *full-attention* layers use TurboQuantKVCache
    while any non-standard layers (SSM / ArraysCache) are left alone.
    """
    from mlx_lm.models.cache import KVCache
    from turboquant_mlx.cache import TurboQuantKVCache

    default_cache = _original_make_prompt_cache(model, max_kv_size=max_kv_size, **kwargs)

    if not _tq_bits:
        logging.info("[TQ] TurboQuant disabled — using standard KV cache")
        return default_cache

    num_layers = len(default_cache)
    tq_count = 0

    for i, c in enumerate(default_cache):
        if isinstance(c, KVCache):
            is_edge = i < _tq_fp16_layers or i >= num_layers - _tq_fp16_layers
            if not is_edge:
                default_cache[i] = TurboQuantKVCache(bits=_tq_bits)
                tq_count += 1

    logging.info(
        f"[TQ] TurboQuant cache — "
        f"{tq_count}/{num_layers} layers compressed at {_tq_bits} bits, "
        f"{num_layers - tq_count} layers unchanged (SSM/FP16)"
    )
    return default_cache


def _safe_can_trim(cache):
    """
    TurboQuant caches don't support trimming (yet).
    Returning False makes the server reset the full cache on mismatch,
    which is fine for single-user / low-concurrency serving.
    """
    try:
        return _original_can_trim(cache)
    except (AttributeError, TypeError):
        return False


# Patch the module-level symbols
cache_module.make_prompt_cache = _turboquant_make_prompt_cache
cache_module.can_trim_prompt_cache = _safe_can_trim

# Also patch the names that server.py already bound via
# ``from .models.cache import make_prompt_cache, can_trim_prompt_cache``
import mlx_lm.server as server_module

server_module.make_prompt_cache = _turboquant_make_prompt_cache
server_module.can_trim_prompt_cache = _safe_can_trim


# ═══════════════════════════════════════════════════════════════════
#  2.  Import the rest of the server machinery
# ═══════════════════════════════════════════════════════════════════

from mlx_lm.server import ModelProvider, run  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
#  3.  CLI — mirrors mlx_lm.server args + TurboQuant / draft extras
# ═══════════════════════════════════════════════════════════════════

DEFAULT_MODEL = "mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"
# DEFAULT_DRAFT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DEFAULT_DRAFT_MODEL = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MLX Server with TurboQuant KV Cache + Speculative Decoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Model ────────────────────────────────────────────────────
    p.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"HF repo or local path to the main MLX model (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--adapter-path", type=str, default=None,
        help="Optional LoRA / adapter weights",
    )

    # ── Speculative decoding ─────────────────────────────────────
    p.add_argument(
        "--draft-model",
        type=str,
        default=DEFAULT_DRAFT_MODEL,
        help=(
            "HF repo or local path to the draft model for speculative "
            "decoding (default: disabled). "
            "Pass 'none' to disable speculative decoding."
        ),
    )
    p.add_argument(
        "--num-draft-tokens",
        type=int,
        default=3,
        help="Tokens the draft model proposes per step (default: 3)",
    )

    # ── TurboQuant ───────────────────────────────────────────────
    p.add_argument(
        "--turboquant-bits",
        type=float,
        default=3,
        help="KV cache quantization: 2 | 3 | 3.5 | 4 (default: 3)",
    )
    p.add_argument(
        "--fp16-layers",
        type=int,
        default=0,
        help=(
            "First/last N layers kept in FP16 for quality. "
            "Use 0 for 32B+ models, 1-4 for 7B (default: 0)"
        ),
    )

    # ── Server ───────────────────────────────────────────────────
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    # ── Generation defaults ──────────────────────────────────────
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument(
        "--prompt-cache-size", type=int, default=10,
        help="Maximum number of distinct KV caches in the prompt cache (default: 10)",
    )
    p.add_argument("--prompt-cache-bytes", type=int, default=None)
    p.add_argument("--prefill-step-size", type=int, default=2048)
    p.add_argument("--decode-concurrency", type=int, default=32)
    p.add_argument("--prompt-concurrency", type=int, default=8)
    p.add_argument("--pipeline", action="store_true")

    # ── Tokenizer / chat ─────────────────────────────────────────
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--chat-template", type=str, default="")
    p.add_argument("--use-default-chat-template", action="store_true")
    p.add_argument(
        "--chat-template-args",
        type=json.loads,
        default="{}",
        help='JSON string, e.g. \'{"enable_thinking":false}\'',
    )

    return p


# ═══════════════════════════════════════════════════════════════════
#  4.  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Handle --draft-model none  →  disable speculative decoding
    if args.draft_model and args.draft_model.lower() == "none":
        args.draft_model = None

    # Push TurboQuant settings into the globals the patched
    # make_prompt_cache reads at cache-creation time.
    global _tq_bits, _tq_fp16_layers
    _tq_bits = args.turboquant_bits
    _tq_fp16_layers = args.fp16_layers

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    # ── Pretty startup banner ────────────────────────────────────
    logging.info("╔══════════════════════════════════════════════════════╗")
    logging.info("║    MLX Server — TurboQuant + Speculative Decoding    ║")
    logging.info("╚══════════════════════════════════════════════════════╝")
    logging.info(f"  Main model   : {args.model}")
    logging.info(f"  Draft model  : {args.draft_model or 'disabled'}")
    logging.info(f"  Draft tokens : {args.num_draft_tokens}")
    logging.info(f"  TQ bits      : {args.turboquant_bits}")
    logging.info(f"  FP16 layers  : {args.fp16_layers}")
    logging.info(f"  Max tokens   : {args.max_tokens}")
    logging.info(f"  Endpoint     : http://{args.host}:{args.port}/v1/chat/completions")

    # ModelProvider reads these attrs from the namespace:
    #   .model, .adapter_path, .draft_model, .num_draft_tokens,
    #   .trust_remote_code, .chat_template, .use_default_chat_template,
    #   .chat_template_args, .temp, .top_p, .top_k, .min_p, .max_tokens
    run(args.host, args.port, ModelProvider(args))


if __name__ == "__main__":
    main()