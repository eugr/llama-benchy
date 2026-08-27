"""
Main entry point for the llama-benchy CLI.
"""

import asyncio
import datetime
import requests
import sys
from . import __version__
from .config import BenchmarkConfig
from .corpus import TokenizedCorpus
from .prompts import (
    HuggingFaceConversationPromptGenerator,
    PromptGenerator,
)
from .client import LLMClient
from .runner import BenchmarkRunner
from .progress import ProgressEmitter

async def main_async():
    # 1. Parse Configuration
    config = BenchmarkConfig.from_args()

    # 1b. If JSONL is going to stdout, route llama-benchy's status prints to
    # stderr so they don't corrupt the JSONL stream a consumer is parsing.
    if config.emit_progress == "-":
        sys.stdout = sys.stderr

    # 2. Print Header
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"llama-benchy ({__version__})")
    print(f"Date: {current_time}")
    print(f"Benchmarking model: {config.model} at {config.base_url}")
    print(f"Concurrency levels: {config.concurrency_levels}")

    # 3. Prepare Data
    corpus = TokenizedCorpus(
        config.book_url,
        config.tokenizer,
        config.model,
        load_data=not bool(config.messages_dataset),
    )
    if config.messages_dataset:
        print(f"Loading dataset prompt source: {config.messages_dataset}")
    else:
        print(f"Total tokens available in text corpus: {len(corpus)}")

    # 4. Initialize Components
    prompt_gen: (
        PromptGenerator | HuggingFaceConversationPromptGenerator
    )
    if config.messages_dataset:
        try:
            prompt_gen = HuggingFaceConversationPromptGenerator(
                corpus,
                config.messages_dataset,
            )
        except (OSError, ValueError, requests.RequestException) as e:
            print(f"Error loading messages dataset: {e}")
            raise SystemExit(2)
    else:
        prompt_gen = PromptGenerator(corpus)
    client = LLMClient(
        config.base_url,
        config.api_key,
        config.served_model_name,
        config.extra_body,
        config.exact_tg,
    )

    progress = None
    if config.emit_progress:
        progress = ProgressEmitter(config.emit_progress, llama_benchy_version=__version__)
    runner = BenchmarkRunner(config, client, prompt_gen, progress=progress)

    # 5. Run Benchmark Suite
    status = "ok"
    try:
        await runner.run_suite()
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except BaseException:
        status = "error"
        raise
    finally:
        if progress is not None:
            try:
                progress.bench_complete(status=status)
            finally:
                progress.close()

    print(f"\nllama-benchy ({__version__})")
    print(f"date: {current_time} | latency mode: {config.latency_mode}")

def main():
    """Entry point for the CLI command."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        sys.exit(1)

if __name__ == "__main__":
    main()
