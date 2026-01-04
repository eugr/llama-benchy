# llama-benchy - llama-bench style benchmarking tool for all backends

This script benchmarks OpenAI-compatible LLM endpoints, generating statistics similar to `llama-bench`.

## Motivation

`llama-bench` is a CLI tool that is a part of a very popular [llama.cpp](https://github.com/ggml-org/llama.cpp) inference engine. It is widely used in LLM community to benchmark models and allows to perform measurement at different context sizes.
However, it is available only for llama.cpp and cannot be used with other inference engines, like vllm or SGLang.

Also, it performs measurements using the C++ engine directly which is not representative of the end user experience which can be quite different in practice.

vLLM has its own powerful benchmarking tool, but while it can be used with other inference engines, there are a few issues:

- It's very tricky and even impossible to calculate prompt processing speeds at different context lengths. You can use `vllm bench sweep serve`, but it only works well with vLLM with prefix caching disabled on the server. Even with random prompts it will reuse the same prompt between multiple runs which will hit the cache in `llama-server` for instance. So you will get very low median TTFT times and very high prompt processing speeds. 
- The TTFT measurement it uses is not actually until the first usable token, it's until the very first response from the server which may not contain any tokens in /v1/chat/completions mode.
- Random dataset is the only ones that allows to specify an arbitrary number of tokens, but randomly generated token sequence doesn't let you adequately measure speculative decoding/MTP.

As of January 2nd, 2026, I wasn't able to find any existing benchmarking tool that brings llama-bench style measurements at different context lengths to any OpenAI-compatible endpoint.

## Features

- Measures Prompt Processing (pp) and Token Generation (tg) speeds at different context depths.
- Can measure separate context prefill and prompt processing over existing cached context at different context depths.
- Reports Time To First Response (TTFR), Estimated Prompt Processing Time (est_ppt), and End-to-End TTFT.
- Supports configurable prompt length (`--pp`), generation length (`--tg`), and context depth (`--depth`).
- Can run multiple iterations (`--runs`) and report mean ± std.
- Uses HuggingFace tokenizers for accurate token counts.
- Downloads a book from Project Gutenberg to use as source text for prompts to ensure better benchmarking of spec.decoding/MTP models.
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
python llama-benchy.py --base-url <ENDPOINT_URL> --model <MODEL_NAME> --pp <PROMPT_TOKENS> --tg <GEN_TOKENS> [OPTIONS]
```

Example:

```bash
python llama-benchy.py \
  --base-url http://localhost:8000/v1 \
  --model openai/gpt-oss-120b \
  --depth 0 4096 8192 16384 32768 \
  --adapt-prompt \
  --latency-mode generation
```

Output:


| model               |            test |             t/s |          ttfr (ms) |       est_ppt (ms) |      e2e_ttft (ms) |
|:--------------------|----------------:|----------------:|-------------------:|-------------------:|-------------------:|
| openai/gpt-oss-120b |          pp2048 | 2019.02 ± 34.98 |    1054.64 ± 17.57 |    1014.66 ± 17.57 |    1115.41 ± 18.70 |
| openai/gpt-oss-120b |            tg32 |    52.94 ± 1.01 |                    |                    |                    |
| openai/gpt-oss-120b |  pp2048 @ d4096 | 1994.49 ± 77.97 |   3129.18 ± 120.27 |   3089.19 ± 120.27 |   3198.97 ± 122.24 |
| openai/gpt-oss-120b |    tg32 @ d4096 |    46.69 ± 1.11 |                    |                    |                    |
| openai/gpt-oss-120b |  pp2048 @ d8192 | 1751.68 ± 34.44 |   5892.61 ± 114.68 |   5852.63 ± 114.68 |   5971.27 ± 115.77 |
| openai/gpt-oss-120b |    tg32 @ d8192 |    40.40 ± 1.19 |                    |                    |                    |
| openai/gpt-oss-120b | pp2048 @ d16384 | 1475.63 ± 31.41 |  12542.02 ± 265.86 |  12502.04 ± 265.86 |  12634.67 ± 269.43 |
| openai/gpt-oss-120b |   tg32 @ d16384 |    33.86 ± 1.45 |                    |                    |                    |
| openai/gpt-oss-120b | pp2048 @ d32768 | 1131.86 ± 50.53 | 30869.90 ± 1410.15 | 30829.92 ± 1410.15 | 30992.96 ± 1417.33 |
| openai/gpt-oss-120b |   tg32 @ d32768 |    25.34 ± 1.31 |                    |                    |                    |

llama-benchy (build: 75bc129)
date: 2026-01-02 17:11:19 | latency mode: generation

-------

It's recommended to use "generation" latency mode to get prompt processing speeds closer to real numbers, especially on shorter prompts.
`--adapt-prompt` will ensure the prompt tokens match the specified value, regardless of the chat template applied.

Generally you don't need to disable prompt caching on the server, as a probability of cache hits is fairly small. You can add `--no-cache` that will add some random noise if you get cache hits.

### Arguments

-   `--base-url`: OpenAI compatible endpoint URL (Required).
-   `--api-key`: API Key (Default: "EMPTY").
-   `--model`: Model name (Required).
-   `--served-model-name`: Model name used in API calls (Defaults to --model if not specified).
-   `--tokenizer`: HuggingFace tokenizer name (Defaults to model name).
-   `--pp`: List of prompt processing token counts (Default: [2048]).
-   `--tg`: List of token generation counts (Default: [32]).
-   `--depth`: List of context depths (Default: [0]).
-   `--runs`: Number of runs per test (Default: 3).
-   `--no-cache`: Add noise to requests to improve prefix caching avoidance. Also sends `cache-prompt=false` to the server.
-   `--post-run-cmd`: Command to execute after each test run.
-   `--book-url`: URL of a book to use for text generation (Defaults to Sherlock Holmes).
-   `--latency-mode`: Method to measure latency: 'api' (call list models function) - default, 'generation' (single token generation), or 'none' (skip latency measurement).
-   `--no-warmup`: Skip warmup phase.
-   `--adapt-prompt`: Adapt prompt size based on warmup token usage delta.
-   `--enable-prefix-caching`: Enable prefix caching performance measurement. When enabled (and depth > 0), it performs a two-step benchmark: first loading the context (reported as `ctx_pp`), then running the prompt with the cached context.

### Metrics

The script outputs a table with the following metrics. All time measurements are in milliseconds (ms).

#### Latency Adjustment
The script attempts to estimate network or processing latency to provide "server-side" processing times.
- **Latency**: Measured based on `--latency-mode`.
  - `api`: Time to fetch `/models` (from sending request to getting first byte of the response). Eliminates network latency only.
  - `generation`: Time to generate 1 token (from sending request to getting first byte of the response). Tries to eliminate network and server overhead latency.
  - `none`: Assumed to be 0.
- This measured latency is subtracted from `ttfr` to calculate `est_ppt`.

#### Table Columns

-   **`t/s` (Tokens per Second)**:
    -   **For Prompt Processing (pp)**: Calculated as `Total Prompt Tokens / est_ppt`. This represents the prefill speed.
    -   **For Token Generation (tg)**: Calculated as `(Total Generated Tokens - 1) / (Time of Last Token - Time of First Token)`. This represents the decode speed, excluding the first token latency.

-   **`ttfr (ms)` (Time To First Response)**:
    -   Calculation: `Time of First Response Chunk - Start Time`.
    -   Represents the raw time until the client receives *any* stream data from the server (including empty chunks or role definitions, but excluding initial http response header). This includes network latency. The same measurement method is used by `vllm bench serve` to report TTFT.

-   **`est_ppt (ms)` (Estimated Prompt Processing Time)**:
    -   Calculation: `TTFR - Estimated Latency`.
    -   Estimated time the server spent processing the prompt. Used for calculating Prompt Processing speed.

-   **`e2e_ttft (ms)` (End-to-End Time To First Token)**:
    -   Calculation: `Time of First Content Token - Start Time`.
    -   The total time perceived by the client from sending the request to seeing the first generated content.

### Prefix Caching Benchmarking

When `--enable-prefix-caching` is used (with `--depth` > 0), the script performs a two-step process for each run to measure the impact of prefix caching:

1.  **Context Load**: Sends the context tokens (as a system message) with an empty user message. This forces the server to process and cache the context.
    -   Reported as `ctx_pp @ d{depth}` (Context Prompt Processing) and `ctx_tg @ d{depth}`.
2.  **Inference**: Sends the same context (system message) followed by the actual prompt (user message). The server should reuse the cached context.
    -   Reported as standard `pp{tokens} @ d{depth}` and `tg{tokens} @ d{depth}`.

By comparing the `pp` speed of the Inference step with a non-cached run (or the Context Load step), you can observe the speedup provided by prefix caching.

### Example

```bash
python llama-benchy.py \
  --base-url http://localhost:8000/v1 \
  --model openai/gpt-oss-120b \
  --pp 128 256 \
  --tg 32 64 \
  --depth 0 1024
```

This will run benchmarks for all combinations of pp (128, 256), tg (32, 64), and depth (0, 1024).
