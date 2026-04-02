# my-llm

MLX inference server with [TurboQuant](https://github.com/noahsaso/turboquant-mlx) KV cache compression. Exposes an OpenAI-compatible API.

Default model: [`mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit`](https://huggingface.co/mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit)

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

Clone with submodules:

```bash
git clone --recurse-submodules git@github.com:noahsaso/my-llm.git
cd my-llm
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

Download the default model:

```bash
hf download mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit
```

Install dependencies:

```bash
uv sync
```

## Usage

```bash
uv run my-llm
```

With custom options:

```bash
uv run my-llm \
  --model mlx-community/Qwen2.5-32B-Instruct-4bit \
  --turboquant-bits 3 \
  --fp16-layers 0 \
  --port 8080
```

## Options

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

## [Pi](https://shittycodingagent.ai/) coding agent

Add the following to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "my-llm": {
      "baseUrl": "http://localhost:8080/v1",
      "apiKey": "unused",
      "models": [
        {
          "id": "mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
          "name": "Local Qwen 3.5 (Opus 4.6 Distilled)",
          "api": "openai-completions",
          "contextWindow": 128000,
          "maxTokens": 4096,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
```

Then start the server and select the model in Pi:

```bash
uv run my-llm
pi
```

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
