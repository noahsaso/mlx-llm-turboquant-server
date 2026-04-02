# my-llm

MLX inference server with [TurboQuant](https://github.com/noahsaso/turboquant-mlx) KV cache compression. Exposes an OpenAI-compatible API.

Default model: `mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit`

## Quick start

```bash
uv run my-llm
```

## Options

```bash
uv run my-llm \
  --model mlx-community/Qwen2.5-32B-Instruct-4bit \
  --turboquant-bits 3 \
  --fp16-layers 0 \
  --port 8080
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit | HF repo or local path |
| `--draft-model` | disabled | Draft model for speculative decoding |
| `--num-draft-tokens` | 3 | Tokens proposed per speculative step |
| `--turboquant-bits` | 3 | KV cache quantization (2, 3, 3.5, 4; 0 to disable) |
| `--fp16-layers` | 0 | First/last N layers kept in FP16 |
| `--host` | 127.0.0.1 | |
| `--port` | 8080 | |
| `--max-tokens` | 4096 | Max generation tokens |
| `--temp` | 0.0 | |

## Test

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```
