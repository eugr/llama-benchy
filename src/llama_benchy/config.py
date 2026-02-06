from dataclasses import dataclass, field
from typing import List, Optional
import argparse
import os
from ._version import __version__

@dataclass
class BenchmarkConfig:
    base_url: str
    api_key: str
    model: str
    served_model_name: str
    tokenizer: Optional[str]
    pp_counts: List[int]
    tg_counts: List[int]
    depths: List[int]
    num_runs: int
    no_cache: bool
    latency_mode: str
    no_warmup: bool
    adapt_prompt: bool
    enable_prefix_caching: bool
    book_url: str
    post_run_cmd: Optional[str]
    concurrency_levels: List[int]
    save_result: Optional[str] = None
    result_format: str = "md"
    save_all_throughput_data: bool = False

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="LLM Benchmark Script")
        parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
        parser.add_argument("--base-url", type=str, required=True, help="OpenAI compatible endpoint URL")
        parser.add_argument("--api-key", type=str, default="EMPTY", help="API Key for the endpoint")
        parser.add_argument("--model", type=str, required=True, help="Model name to use for benchmarking")
        parser.add_argument("--served-model-name", type=str, default=None, help="Model name used in API calls (defaults to --model if not specified)")
        parser.add_argument("--tokenizer", type=str, default=None, help="HuggingFace tokenizer name (defaults to model name)")
        parser.add_argument("--pp", type=int, nargs='+', required=False, default=[2048], help="List of prompt processing token counts - default: 2048")
        parser.add_argument("--tg", type=int, nargs='+', required=False, default=[32], help="List of token generation counts - default: 32")
        parser.add_argument("--depth", type=int, nargs='+', default=[0], help="List of context depths (previous conversation tokens) - default: 0")
        parser.add_argument("--runs", type=int, default=3, help="Number of runs per test - default: 3")
        parser.add_argument("--no-cache", action="store_true", help="Ensure unique requests to avoid prefix caching and send cache_prompt=false to the server")
        parser.add_argument("--post-run-cmd", type=str, default=None, help="Command to execute after each test run")
        parser.add_argument("--book-url", type=str, default="https://www.gutenberg.org/files/1661/1661-0.txt", help="URL of a book to use for text generation, defaults to Sherlock Holmes")
        parser.add_argument("--latency-mode", type=str, default="api", choices=["api", "generation", "none"], help="Method to measure latency: 'api' (list models) - default, 'generation' (single token generation), or 'none' (skip latency measurement)")
        parser.add_argument("--no-warmup", action="store_true", help="Skip warmup phase")
        parser.add_argument("--adapt-prompt", action="store_true", default=True, help="Adapt prompt size based on warmup token usage delta (default: True)")
        parser.add_argument("--no-adapt-prompt", action="store_false", dest="adapt_prompt", help="Disable prompt size adaptation")
        parser.add_argument("--enable-prefix-caching", action="store_true", help="Enable prefix caching performance measurement")
        parser.add_argument("--concurrency", type=int, nargs='+', default=[1], help="List of concurrency levels (number of concurrent requests per test) - default: [1]")
        parser.add_argument("--save-result", type=str, help="File to save results to")
        parser.add_argument("--format", type=str, default="md", choices=["md", "json", "csv"], help="Output format")
        parser.add_argument("--save-all-throughput-data", action="store_true", help="Save calculated throughput for each 1 second window inside peak throughput calculation during the run.")
          
        args = parser.parse_args()
        
        return cls(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            served_model_name=args.served_model_name if args.served_model_name else args.model,
            tokenizer=args.tokenizer,
            pp_counts=args.pp,
            tg_counts=args.tg,
            depths=args.depth,
            num_runs=args.runs,
            no_cache=args.no_cache,
            latency_mode=args.latency_mode,
            no_warmup=args.no_warmup,
            adapt_prompt=args.adapt_prompt,
            enable_prefix_caching=args.enable_prefix_caching,
            book_url=args.book_url,
            post_run_cmd=args.post_run_cmd,
            concurrency_levels=args.concurrency,
            save_result=args.save_result,
            result_format=args.format,
            save_all_throughput_data=args.save_all_throughput_data
        )
