# llama-benchy - llama-bench style benchmarking tool for all backends

This script benchmarks OpenAI-compatible LLM endpoints, generating statistics similar to `llama-bench`.

## Motivation

`llama-bench` is a CLI tool that is a part of a very popular [llama.cpp](https://github.com/ggml-org/llama.cpp) inference engine. It is widely used in LLM community to benchmark models and allows to perform measurement at different context sizes.
However, it is available only for llama.cpp and cannot be used with other inference engines, like vllm or SGLang.

Also, it performs measurements using the C++ engine directly which is not representative of the end user experience which can be quite different in practice.

vLLM has its own powerful benchmarking tool, but while it can be used with other inference engines, there are a few issues:

- It's very tricky and even impossible to calculate prompt processing speeds at different context lengths. You can use `vllm bench sweep serve`, but it only works well with vLLM with prefix caching disabled on the server. Even with random prompts it will reuse the same prompt between multiple runs which will hit the cache in `llama-server` for instance. So you will get very low median TTFT times and very high prompt processing speeds. 
- The TTFT measurement it uses is not actually until the first usable token, it's until the very first data chunk from the server which may not contain any generated tokens in /v1/chat/completions mode.
- Random dataset is the only ones that allows to specify an arbitrary number of tokens, but randomly generated token sequence doesn't let you adequately measure speculative decoding/MTP.

As of January 2nd, 2026, I wasn't able to find any existing benchmarking tool that brings llama-bench style measurements at different context lengths to any OpenAI-compatible endpoint.

## Features

- Measures Prompt Processing (pp) and Token Generation (tg) speeds at different context depths.
- Can measure separate context prefill and prompt processing over existing cached context at different context depths.
- Reports Time To First Response (data chunk) (TTFR), Estimated Prompt Processing Time (est_ppt), and End-to-End TTFT.
- Supports configurable prompt length (`--pp`), generation length (`--tg`), and context depth (`--depth`).
- Can run multiple iterations (`--runs`) and report mean ± std.
- Uses HuggingFace tokenizers for accurate token counts.
- Downloads a book from Project Gutenberg to use as source text for prompts to ensure better benchmarking of spec.decoding/MTP models.
- Supports executing a command after each run (e.g., to clear cache).
- Configurable latency measurement mode.
- Supports concurrent requests (`--concurrency`) to measure throughput under load.
- Can save results to file in Markdown, JSON, or CSV format.
- Can save granular time-series data for token generation when JSON output is used (`--save-total-throughput-timeseries` and `--save-total-throughput-timeseries`).

# Current Limitations

- Evaluates against `/v1/chat/completions` endpoint only.

## Installation

Using `uv` is recommended. You can install `uv` here: https://docs.astral.sh/uv/getting-started/installation/

### Option 1: Run without installation using `uvx`

Run the release version from PyPI:

```bash
uvx llama-benchy --base-url <ENDPOINT_URL> --model <MODEL_NAME>
```

Run the latest version from the main branch:

```bash
uvx --from git+https://github.com/eugr/llama-benchy llama-benchy --base-url <ENDPOINT_URL> --model <MODEL_NAME>
```

### Option 2: Install into virtual environment

```bash
# Clone the repository
git clone https://github.com/eugr/llama-benchy.git
cd llama-benchy

# Create virtual environment
uv venv

# Install with uv (installs into a virtual environment automatically)
uv pip install -e .
```

To run, activate the environment first

```bash
source .venv/bin/activate
```

Then execute the command:

```bash
llama-benchy --base-url <ENDPOINT_URL> --model <MODEL_NAME>
```


### Option 3: Run without installing (`uv run`)

```bash
# Clone the repository
git clone https://github.com/eugr/llama-benchy.git
cd llama-benchy

# Using uv run (creates a virtual environment if it doesn't exist and runs the command)
uv run llama-benchy --base-url <ENDPOINT_URL> --model <MODEL_NAME>
```

### Option 3: Install into system path

Release version from PyPI:

```bash
uv pip install -U llama-benchy
```

Current version from the main branch:

```bash
uv pip install git+https://github.com/eugr/llama-benchy --system
```

## Usage

After installation, you can run the tool directly:

```bash
llama-benchy --base-url <ENDPOINT_URL> --model <MODEL_NAME> --pp <PROMPT_TOKENS> --tg <GEN_TOKENS> [OPTIONS]
```

Example:

```bash
llama-benchy \
  --base-url http://spark:8888/v1 \
  --model openai/gpt-oss-120b \
  --depth 0 4096 8192 16384 32768 \
  --latency-mode generation
```

Output:


| model               |            test |             t/s |     peak t/s |      ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:--------------------|----------------:|----------------:|-------------:|---------------:|---------------:|----------------:|
| openai/gpt-oss-120b |          pp2048 | 8521.08 ± 69.61 |              |  297.14 ± 1.97 |  240.36 ± 1.97 |   340.65 ± 3.49 |
| openai/gpt-oss-120b |            tg32 |    73.18 ± 0.45 | 75.84 ± 0.48 |                |                |                 |
| openai/gpt-oss-120b |  pp2048 @ d4096 | 9450.36 ± 24.73 |              |  706.92 ± 1.70 |  650.14 ± 1.70 |   750.96 ± 3.08 |
| openai/gpt-oss-120b |    tg32 @ d4096 |    72.22 ± 0.83 | 74.81 ± 0.86 |                |                |                 |
| openai/gpt-oss-120b |  pp2048 @ d8192 | 8481.42 ± 38.50 |              | 1264.15 ± 5.50 | 1207.37 ± 5.50 |  1307.31 ± 6.20 |
| openai/gpt-oss-120b |    tg32 @ d8192 |    71.78 ± 0.74 | 74.36 ± 0.77 |                |                |                 |
| openai/gpt-oss-120b | pp2048 @ d16384 | 7954.96 ± 14.20 |              | 2373.83 ± 4.14 | 2317.05 ± 4.14 |  2418.63 ± 4.87 |
| openai/gpt-oss-120b |   tg32 @ d16384 |    70.48 ± 0.84 | 73.02 ± 0.86 |                |                |                 |
| openai/gpt-oss-120b | pp2048 @ d32768 |  6896.57 ± 4.62 |              | 5105.09 ± 3.38 | 5048.31 ± 3.38 |  5153.34 ± 2.87 |
| openai/gpt-oss-120b |   tg32 @ d32768 |    65.80 ± 0.79 | 68.17 ± 0.82 |                |                |                 |

llama-benchy (0.2.2.dev1+g52d2b0d55.d20260206)
date: 2026-02-06 15:52:14 | latency mode: generation

-------

It's recommended to use "generation" latency mode to get prompt processing speeds closer to real numbers, especially on shorter prompts.
By default, the script adapts the prompt size to match the specified value, regardless of the chat template applied. Use `--no-adapt-prompt` to disable this behavior.

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
-   `--adapt-prompt`: Adapt prompt size based on warmup token usage delta (Default: True).
-   `--no-adapt-prompt`: Disable prompt size adaptation.
-   `--enable-prefix-caching`: Enable prefix caching performance measurement. When enabled (and depth > 0), it performs a two-step benchmark: first loading the context (reported as `ctx_pp`), then running the prompt with the cached context.
-   `--concurrency`: List of concurrency levels (number of concurrent requests per test) (Default: [1]).
-   `--save-result`: File to save results to.
-   `--format`: Output format: 'md', 'json', 'csv' (Default: 'md').
-   `--save-total-throughput-timeseries`: Save calculated TOTAL throughput for each 1 second window inside peak throughput calculation during the run (default: off).
-   `--save-all-throughput-timeseries`: Save calculated throughput timeseries for EACH individual request (default: off).

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
        -   When `concurrency` > 1:
        -   **`t/s (total)`**: Total throughput across all concurrent requests.
        -   **`t/s (req)`**: Average throughput per individual request.

- **`peak t/s` (Maximum observed Tokens per Second)**: 
    - **Only for Token Generation (tg)**: The highest token‑generation throughput observed in any 1‑second window during the run across all concurrent requests.

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

In this case, `pp` and `tg` speeds will show an actual prompt processing / token generation speeds for a follow up prompt with a context pre-filled.

**Example**:

```bash
llama-benchy \
  --base-url http://spark:8888/v1 \
  --model openai/gpt-oss-120b \
  --depth 0 4096 8192 16384 32768 \
  --latency-mode generation \
  --enable-prefix-caching
```

Output:


| model               |            test |              t/s |      peak t/s |        ttfr (ms) |     est_ppt (ms) |    e2e_ttft (ms) |
|:--------------------|----------------:|-----------------:|--------------:|-----------------:|-----------------:|-----------------:|
| openai/gpt-oss-120b |          pp2048 | 8236.95 ± 134.25 |               |    298.95 ± 4.08 |    248.70 ± 4.08 |    342.07 ± 3.40 |
| openai/gpt-oss-120b |            tg32 |     73.96 ± 1.19 |  76.63 ± 1.24 |                  |                  |                  |
| openai/gpt-oss-120b |  ctx_pp @ d4096 |  9259.71 ± 76.35 |               |    492.62 ± 3.63 |    442.38 ± 3.63 |    535.67 ± 3.75 |
| openai/gpt-oss-120b |  ctx_tg @ d4096 |     73.28 ± 0.80 |  75.93 ± 0.82 |                  |                  |                  |
| openai/gpt-oss-120b |  pp2048 @ d4096 | 7467.44 ± 131.59 |               |    324.59 ± 4.84 |    274.34 ± 4.84 |    367.60 ± 4.95 |
| openai/gpt-oss-120b |    tg32 @ d4096 |     72.25 ± 0.12 |  74.86 ± 0.12 |                  |                  |                  |
| openai/gpt-oss-120b |  ctx_pp @ d8192 | 9177.24 ± 167.37 |               |   943.19 ± 16.45 |   892.94 ± 16.45 |    973.01 ± 5.31 |
| openai/gpt-oss-120b |  ctx_tg @ d8192 |     73.43 ± 0.50 |  76.09 ± 0.54 |                  |                  |                  |
| openai/gpt-oss-120b |  pp2048 @ d8192 | 6846.48 ± 135.57 |               |    349.50 ± 5.96 |    299.25 ± 5.96 |    394.26 ± 5.16 |
| openai/gpt-oss-120b |    tg32 @ d8192 |     72.62 ± 0.66 |  75.23 ± 0.68 |                  |                  |                  |
| openai/gpt-oss-120b | ctx_pp @ d16384 | 8235.05 ± 179.06 |               |  2040.75 ± 43.92 |  1990.50 ± 43.92 |  2073.53 ± 23.55 |
| openai/gpt-oss-120b | ctx_tg @ d16384 |     73.87 ± 5.04 |  76.53 ± 5.22 |                  |                  |                  |
| openai/gpt-oss-120b | pp2048 @ d16384 | 5441.56 ± 484.88 |               |   429.81 ± 35.96 |   379.57 ± 35.96 |   483.42 ± 48.92 |
| openai/gpt-oss-120b |   tg32 @ d16384 |    62.80 ± 10.73 | 65.06 ± 11.12 |                  |                  |                  |
| openai/gpt-oss-120b | ctx_pp @ d32768 | 6904.92 ± 217.24 |               | 4800.62 ± 151.68 | 4750.38 ± 151.68 | 4832.95 ± 157.53 |
| openai/gpt-oss-120b | ctx_tg @ d32768 |     69.77 ± 5.32 |  72.29 ± 5.52 |                  |                  |                  |
| openai/gpt-oss-120b | pp2048 @ d32768 | 4549.10 ± 105.92 |               |   500.69 ± 10.32 |   450.44 ± 10.32 |    548.23 ± 8.98 |
| openai/gpt-oss-120b |   tg32 @ d32768 |     62.18 ± 6.87 |  64.59 ± 6.87 |                  |                  |                  |

llama-benchy (0.2.2.dev1+g52d2b0d55.d20260206)
date: 2026-02-06 16:15:38 | latency mode: generation

### Combining multiple parameters

You can specify multiple parameters for `--depth`, `--pp`, `--tg` and `--concurrency`. The benchmarks will run using the following hierarchy: depth -> pp -> tg -> concurrency.

```bash
llama-benchy \
  --base-url http://localhost:8000/v1 \
  --model openai/gpt-oss-120b \
  --pp 128 256 \
  --tg 32 64 \
  --depth 0 1024
```

This will run benchmarks for all combinations of pp (128, 256), tg (32, 64), and depth (0, 1024).

### Concurrency measurement

To test how the server performs in concurrent requests scenario, you can specify one or more concurrency levels using `--concurrency <N>` (e.g. `--concurrency 1 2 4`).

When running with `concurrency > 1`, `llama-benchy` launches N parallel clients. The results table will include:
*   **t/s (total)**: The aggregate throughput (tokens/sec) of all clients combined.
*   **t/s (req)**: The average throughput per client.

This allows you to measure how the server scales and find the saturation point where adding more clients doesn't increase the total throughput.

Please note, that currently all batches run at the same time. If `--enable-prefix-caching` is used, then all prefill requests are executed simultaneously, followed by concurrent follow up requests.
Other concurrency scenarios may be added in the future.

**Example**

```bash
llama-benchy \
  --base-url http://spark:8888/v1 \
  --model openai/gpt-oss-120b \
  --depth 0 4096 \
  --latency-mode generation \
  --enable-prefix-caching \
  --concurrency 1 2
```

Output:

| model               |                test |       t/s (total) |         t/s (req) |       peak t/s |       ttfr (ms) |    est_ppt (ms) |    e2e_ttft (ms) |
|:--------------------|--------------------:|------------------:|------------------:|---------------:|----------------:|----------------:|-----------------:|
| openai/gpt-oss-120b |         pp2048 (c1) |   7803.68 ± 63.74 |   7803.68 ± 63.74 |                |   292.86 ± 2.15 |   262.46 ± 2.15 |    337.08 ± 2.48 |
| openai/gpt-oss-120b |           tg32 (c1) |      74.83 ± 0.59 |      74.83 ± 0.59 |   77.54 ± 0.62 |                 |                 |                  |
| openai/gpt-oss-120b |         pp2048 (c2) |  7198.20 ± 541.69 | 4872.80 ± 1298.05 |                |  473.18 ± 85.71 |  442.77 ± 85.71 |   568.97 ± 40.44 |
| openai/gpt-oss-120b |           tg32 (c2) |     111.27 ± 3.61 |      56.20 ± 1.27 |  115.25 ± 3.74 |                 |                 |                  |
| openai/gpt-oss-120b | ctx_pp @ d4096 (c1) |   8816.20 ± 55.40 |   8816.20 ± 55.40 |                |   495.02 ± 2.91 |   464.62 ± 2.91 |    540.31 ± 1.64 |
| openai/gpt-oss-120b | ctx_tg @ d4096 (c1) |      72.92 ± 0.63 |      72.92 ± 0.63 |   75.56 ± 0.67 |                 |                 |                  |
| openai/gpt-oss-120b | pp2048 @ d4096 (c1) | 5918.71 ± 1447.23 | 5918.71 ± 1447.23 |                | 403.38 ± 110.24 | 372.98 ± 110.24 |   432.07 ± 90.02 |
| openai/gpt-oss-120b |   tg32 @ d4096 (c1) |      65.27 ± 9.85 |      65.27 ± 9.85 |  67.62 ± 10.21 |                 |                 |                  |
| openai/gpt-oss-120b | ctx_pp @ d4096 (c2) | 7934.38 ± 1145.70 |  4665.38 ± 660.55 |                | 928.79 ± 146.59 | 898.38 ± 146.59 | 1053.95 ± 165.93 |
| openai/gpt-oss-120b | ctx_tg @ d4096 (c2) |     111.64 ± 3.25 |      56.38 ± 1.06 |  115.63 ± 3.36 |                 |                 |                  |
| openai/gpt-oss-120b | pp2048 @ d4096 (c2) |  6659.05 ± 231.76 |  3623.78 ± 278.02 |                |  598.81 ± 42.34 |  568.40 ± 42.34 |   615.33 ± 22.02 |
| openai/gpt-oss-120b |   tg32 @ d4096 (c2) |    116.93 ± 10.24 |      58.47 ± 5.12 | 121.11 ± 10.61 |                 |                 |                  |

llama-benchy (0.2.2.dev1+g52d2b0d55.d20260206)
date: 2026-02-06 16:36:05 | latency mode: generation

### Further analysis

To perform additional analysis or generate any visualizations, you can output results in JSON or CSV. 
JSON (`--format json`) will give you the most detailed data. If you specify `--save-total-throughput-data`, then JSON will include total throughput in 1 second intervals.

- [Sample JSON file](schemas/sample.json)
- [Sample JSON file with embedded documentation](schemas/sample.jsonc)
- [JSON schema](schemas/benchmark_report_schema.json)

You can also instantiate llama-benchy classes and run analysis directly from Python. See [Jupyter Notebook example](examples/benchmark_visualization.ipynb).

## Development

### Running Integration Tests

This repository includes a mock server and an integration test suite to verify `llama-benchy` logic without needing a real GPU server.

The mock server emulates:
-   **Prompt Processing (PP):** ~1000 t/s drift-corrected.
-   **Token Generation (TG):** ~50 t/s.
-   **Prefix Caching:** Emulates cache hits by skipping processing time for cached prefixes (system messages).
-   **OpenAI API Compatibility**: Serves `/v1/chat/completions` and `/v1/models`.

To run the integration tests:

```bash
# Install development dependencies
uv sync --all-extras --dev

# Run tests
uv run pytest tests/test_mock_integration.py
```

This test will:
1.  Spin up the mock server on port 8001.
2.  Run `llama-benchy` against it.
3.  Parse the JSON output.
4.  Verify that throughputs match the emulated speeds (PP ~1000, TG ~50) and that caching effectively increases effective throughput.
