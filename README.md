# LLM Benchmark Script

This script benchmarks OpenAI-compatible LLM endpoints, generating statistics similar to `llama-bench`.

## Features

- Measures Prompt Processing (pp) and Token Generation (tg) speeds.
- Reports Time To First Token (TTFT), Time To First Response (TTFR), and End-to-End TTFT.
- Supports configurable prompt length (`--pp`), generation length (`--tg`), and context depth (`--depth`).
- Can run multiple iterations (`--runs`) and report mean ± std.
- Uses HuggingFace tokenizers for accurate token counts.
- Can download a book from Project Gutenberg to use as source text for prompts, or use synthetic data.
- Supports executing a command after each run (e.g., to clear cache).
- Configurable latency measurement mode.

## Installation

1.  Create a virtual environment:
    ```bash
    uv venv .venv
    source .venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    uv pip install -r requirements.txt
    ```

## Usage

```bash
python llm_bench.py --base-url <ENDPOINT_URL> --model <MODEL_NAME> --pp <PROMPT_TOKENS> --tg <GEN_TOKENS> [OPTIONS]
```

### Arguments

-   `--base-url`: OpenAI compatible endpoint URL (Required).
-   `--api-key`: API Key (Default: "EMPTY").
-   `--model`: Model name (Required).
-   `--tokenizer`: HuggingFace tokenizer name (Defaults to model name).
-   `--pp`: List of prompt processing token counts (Required).
-   `--tg`: List of token generation counts (Required).
-   `--depth`: List of context depths (Default: [0]).
-   `--runs`: Number of runs per test (Default: 3).
-   `--no-cache`: Ensure unique requests to avoid prefix caching.
-   `--post-run-cmd`: Command to execute after each test run.
-   `--book-url`: URL of a book to use for text generation.
-   `--latency-mode`: Method to measure latency: 'models' (list models) - default, 'generation' (single token generation), or 'none' (skip latency measurement).
-   `--no-warmup`: Skip warmup phase.

### Metrics

The script outputs a table with the following metrics:

-   `t/s`: Tokens per second (processing speed).
-   `ttft (ms)`: Time To First Token (End-to-End TTFT minus estimated network latency).
-   `ttfr (ms)`: Time To First Response (Time to receive the first chunk of data minus estimated network latency).
-   `e2e_ttft (ms)`: End-to-End Time To First Token (Time from sending request to receiving first token).

### Example

```bash
python llm_bench.py \
  --base-url http://localhost:8000/v1 \
  --model meta-llama/Llama-2-7b-chat-hf \
  --pp 128 256 \
  --tg 32 64 \
  --depth 0 1024
```

This will run benchmarks for all combinations of pp (128, 256), tg (32, 64), and depth (0, 1024).
