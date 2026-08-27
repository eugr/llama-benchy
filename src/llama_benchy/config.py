from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple
import argparse
import json
import os
import re
import requests
import sys
from ._version import __version__


class BenchmarkConfig(BaseModel):
    base_url: str = Field(..., description="OpenAI compatible endpoint URL")
    api_key: str = Field(..., description="API Key for the endpoint")
    model: str = Field(..., description="Model name to use for benchmarking")
    served_model_name: str = Field(
        ...,
        description="Model name used in API calls (defaults to --model if not specified)",
    )
    tokenizer: Optional[str] = Field(
        None,
        description="Tokenizer to use (HF model name or local path; defaults to model name)",
    )
    pp_counts: List[int] = Field(
        ..., description="List of prompt processing token counts"
    )
    tg_counts: List[int] = Field(..., description="List of token generation counts")
    exact_tg: bool = Field(
        False,
        description="Force generated output length to match --tg using server-specific min_tokens and ignore_eos fields",
    )
    depths: List[int] = Field(
        ..., description="List of context depths (previous conversation tokens)"
    )
    num_runs: int = Field(..., description="Number of runs per test")
    warmup_runs: int = Field(
        1,
        description="Number of discarded warmup runs per test shape; also used for generation latency probes",
    )
    no_cache: bool = Field(
        ..., description="Ensure unique requests to avoid prefix caching"
    )
    latency_mode: str = Field(
        ..., description="Method to measure latency: 'api', 'generation', or 'none'"
    )
    no_warmup: bool = Field(..., description="Skip warmup phase")
    skip_coherence: bool = Field(..., description="Skip coherence test after warmup")
    adapt_prompt: bool = Field(
        ..., description="Adapt prompt size based on warmup token usage delta"
    )
    enable_prefix_caching: bool = Field(
        ..., description="Enable prefix caching performance measurement"
    )
    book_url: str = Field(..., description="URL of a book to use for text generation")
    messages_dataset: Optional[str] = Field(
        None, description="Hugging Face agent trajectory dataset used as the prompt source"
    )
    post_run_cmd: Optional[str] = Field(
        None, description="Command to execute after each test run"
    )
    concurrency_levels: List[int] = Field(..., description="List of concurrency levels")
    save_result: Optional[str] = Field(None, description="File to save results to")
    result_format: str = Field("md", description="Output format (md, json, csv)")
    save_total_throughput_timeseries: bool = Field(
        False,
        description="Save calculated TOTAL throughput for each 1 second window inside peak throughput calculation during the run.",
    )
    save_all_throughput_timeseries: bool = Field(
        False,
        description="Save calculated throughput timeseries for EACH individual request.",
    )
    exit_on_first_fail: bool = Field(
        False,
        description="Stop execution on first failed test and exit with non-zero status",
    )
    no_results_on_fail: bool = Field(
        False,
        description="Prevent saving/printing results when error is experienced, turns on --exit-on-first-fail as well",
    )
    extra_body: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra JSON fields to merge into benchmark chat completion requests",
    )
    emit_progress: Optional[str] = Field(
        None,
        description="Emit progress events as JSONL to PATH (or '-' for stdout). See docs/progress-schema.md.",
    )

    @staticmethod
    def _parse_extra_body(values: Optional[List[str]]) -> Dict[str, Any]:
        extra: Dict[str, Any] = {}
        if not values:
            return extra

        for item in values:
            entries = [entry.strip() for entry in item.split(",") if entry.strip()]
            for entry in entries:
                if "=" in entry:
                    key, raw_value = entry.split("=", 1)
                elif ":" in entry:
                    key, raw_value = entry.split(":", 1)
                else:
                    raise ValueError(
                        f"Invalid --extra-body entry '{entry}'. Use key=value or key:value."
                    )

                key = key.strip()
                raw_value = raw_value.strip()
                if not key:
                    raise ValueError(f"Invalid --extra-body entry '{entry}': empty key.")

                try:
                    extra[key] = json.loads(raw_value)
                except json.JSONDecodeError:
                    extra[key] = raw_value

        return extra

    @staticmethod
    def _detect_hf_model_from_endpoint(base_url: str, api_key: str) -> Tuple[str, str]:
        """
        Fetch models from {base_url}/models endpoint and identify HF model name.

        Returns:
            tuple of (hf_model_name, served_model_name)
        """
        HF_MODEL_PATTERN = re.compile(r"^[^/]+/[^/]+$")

        try:
            headers = (
                {"Authorization": f"Bearer {api_key}"}
                if api_key and api_key != "EMPTY"
                else {}
            )
            response = requests.get(f"{base_url}/models", headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise ValueError(
                f"Unable to connect to {base_url}/models endpoint: {e}\n"
                "Please specify --model explicitly."
            )

        # Collect all available models and separate by HF format
        hf_formatted = []  # (hf_name, served_name)
        non_hf_formatted = []  # model names without HF format

        # Parse response based on server type
        # Parse data array first
        if "data" in data:
            for model in data["data"]:
                model_id = model.get("id", "")
                root = model.get("root", "")

                if root and HF_MODEL_PATTERN.match(root):
                    hf_formatted.append((root, model_id))
                elif HF_MODEL_PATTERN.match(model_id):
                    hf_formatted.append((model_id, model_id))
                else:
                    non_hf_formatted.append(model_id)

        # parse models array as a fallback
        # Only process if "data" is not present to avoid duplicates
        elif "models" in data:
            for model in data["models"]:
                model_name = model.get("model", "") or model.get("id", "")
                if HF_MODEL_PATTERN.match(model_name):
                    hf_formatted.append((model_name, model_name))
                else:
                    non_hf_formatted.append(model_name)

        # Guard: Multiple models available - cannot determine which one to use
        if len(hf_formatted) + len(non_hf_formatted) > 1:
            error_msg = "Multiple models available at the endpoint. Please specify --model explicitly.\n\n"

            if hf_formatted:
                error_msg += "Models with HF format:\n"
                error_msg += "\n".join(f"  - {m[0]}" for m in hf_formatted) + "\n"

            if non_hf_formatted:
                error_msg += "\nModels without HF format:\n"
                error_msg += "\n".join(f"  - {m}" for m in non_hf_formatted) + "\n"

            error_msg += "\nPlease specify --model explicitly with the model name you want to test."
            raise ValueError(error_msg)

        # No models found
        if not hf_formatted and not non_hf_formatted:
            raise ValueError(
                "No models found at the endpoint.\nPlease specify --model explicitly."
            )

        # Single non-HF model found
        if not hf_formatted and non_hf_formatted:
            raise ValueError(
                f"Model '{non_hf_formatted[0]}' is not in HF format (namespace/model).\n"
                "Please specify --model explicitly with a valid HF model name."
            )

        # Single HF-formatted model found - validate against HF Hub
        hf_name, served_name = hf_formatted[0]
        try:
            hf_response = requests.get(
                f"https://huggingface.co/api/models/{hf_name}", timeout=3
            )
            if hf_response.status_code in (200, 401):
                return (hf_name, served_name)
        except requests.RequestException:
            pass

        raise ValueError(
            f"Model '{hf_name}' is not a valid HuggingFace model.\n"
            "Please specify --model explicitly with a valid HF model name."
        )

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="LLM Benchmark Script")
        parser.add_argument(
            "--version", action="version", version=f"%(prog)s {__version__}"
        )
        parser.add_argument(
            "--base-url", type=str, required=True, help="OpenAI compatible endpoint URL"
        )
        parser.add_argument(
            "--api-key", type=str, default="EMPTY", help="API Key for the endpoint"
        )
        parser.add_argument(
            "--model",
            type=str,
            required=False,
            default=None,
            help="Model name to use for benchmarking (auto-detected from endpoint if not specified)",
        )
        parser.add_argument(
            "--served-model-name",
            type=str,
            default=None,
            help="Model name used in API calls (defaults to --model if not specified)",
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            default=None,
            help="Tokenizer to use (HF model name or local path; defaults to model name)",
        )
        parser.add_argument(
            "--pp",
            type=int,
            nargs="+",
            required=False,
            default=[2048],
            help="List of prompt processing token counts - default: 2048",
        )
        parser.add_argument(
            "--tg",
            type=int,
            nargs="+",
            required=False,
            default=[32],
            help="List of token generation counts - default: 32",
        )
        parser.add_argument(
            "--exact-tg",
            action="store_true",
            help="Force output length to match --tg by sending min_tokens=<tg> and ignore_eos=true in benchmark requests.",
        )
        parser.add_argument(
            "--depth",
            type=int,
            nargs="+",
            default=[0],
            help="List of context depths (previous conversation tokens) - default: 0",
        )
        parser.add_argument(
            "--runs", type=int, default=3, help="Number of runs per test - default: 3"
        )
        parser.add_argument(
            "--warmup-runs",
            type=int,
            default=1,
            help=(
                "Number of discarded warmup runs per test shape - default: 1. "
                "Also controls discarded probes for --latency-mode generation. "
                "Does not affect the initial prompt-adaptation warmup."
            ),
        )
        parser.add_argument(
            "--no-cache",
            action="store_true",
            help="Ensure unique requests to avoid prefix caching and send cache_prompt=false to the server",
        )
        parser.add_argument(
            "--post-run-cmd",
            type=str,
            default=None,
            help="Command to execute after each test run",
        )
        parser.add_argument(
            "--book-url",
            type=str,
            default="https://www.gutenberg.org/files/1661/1661-0.txt",
            help="URL of a book to use for text generation, defaults to Sherlock Holmes",
        )
        parser.add_argument(
            "--messages-dataset",
            type=str,
            default=None,
            metavar="SELECTOR",
            help=(
                "Hugging Face trajectory selector, e.g. "
                "owner/dataset?subset=default&split=train&instance_id=task-1"
            ),
        )
        parser.add_argument(
            "--latency-mode",
            type=str,
            default="api",
            choices=["api", "generation", "none"],
            help="Method to measure latency: 'api' (list models) - default, 'generation' (single token generation), or 'none' (skip latency measurement)",
        )
        parser.add_argument(
            "--no-warmup", action="store_true", help="Skip warmup phase"
        )
        parser.add_argument(
            "--skip-coherence",
            action="store_true",
            help="Skip coherence test after warmup",
        )
        parser.add_argument(
            "--adapt-prompt",
            action="store_true",
            default=True,
            help="Adapt prompt size based on warmup token usage delta (default: True)",
        )
        parser.add_argument(
            "--no-adapt-prompt",
            action="store_false",
            dest="adapt_prompt",
            help="Disable prompt size adaptation",
        )
        parser.add_argument(
            "--enable-prefix-caching",
            action="store_true",
            help="Enable prefix caching performance measurement",
        )
        parser.add_argument(
            "--concurrency",
            type=int,
            nargs="+",
            default=[1],
            help="List of concurrency levels (number of concurrent requests per test) - default: [1]",
        )
        parser.add_argument("--save-result", type=str, help="File to save results to")
        parser.add_argument(
            "--format",
            type=str,
            default="md",
            choices=["md", "json", "csv"],
            help="Output format",
        )
        parser.add_argument(
            "--save-total-throughput-timeseries",
            action="store_true",
            help="Save calculated TOTAL throughput for each 1 second window inside peak throughput calculation during the run.",
        )
        parser.add_argument(
            "--save-all-throughput-timeseries",
            action="store_true",
            help="Save calculated throughput timeseries for EACH individual request.",
        )
        parser.add_argument(
            "--exit-on-first-fail",
            action="store_true",
            help="Stop execution on first failed test and exit with non-zero status",
        )
        parser.add_argument(
            "--no-results-on-fail",
            action="store_true",
            help="Prevent saving/printing results when error is experienced, turns on --exit-on-first-fail as well",
        )
        parser.add_argument(
            "--extra-body",
            action="append",
            default=[],
            help="Extra JSON fields to merge into benchmark chat completion requests. Accepts key=value or key:value, comma-separated or repeated.",
        )
        parser.add_argument(
            "--emit-progress",
            type=str,
            default=None,
            metavar="PATH",
            help=(
                "Emit benchmark progress events as JSONL to PATH (or '-' for stdout). "
                "External visualizers (live TUIs, web dashboards, post-hoc charts) "
                "consume this stream. Schema: docs/progress-schema.md."
            ),
        )

        args = parser.parse_args()

        if args.no_results_on_fail:
            args.exit_on_first_fail = True
        if args.warmup_runs < 0:
            parser.error("--warmup-runs must be >= 0")
        if args.messages_dataset and (args.depth != [0] or args.enable_prefix_caching):
            parser.error("conversation prompts require --depth 0 and cannot use --enable-prefix-caching")
        try:
            extra_body = BenchmarkConfig._parse_extra_body(args.extra_body)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        # Auto-detect model if not specified
        if args.model is None:
            print("No model specified, attempting to auto-detect from endpoint...")
            try:
                hf_model, served_model = BenchmarkConfig._detect_hf_model_from_endpoint(
                    args.base_url, args.api_key
                )
                model_to_use = hf_model
                served_model_name_to_use = (
                    args.served_model_name if args.served_model_name else served_model
                )
                print(
                    f"Auto-detected HF model: {model_to_use} (served as: {served_model_name_to_use})"
                )
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(1)
        else:
            model_to_use = args.model
            served_model_name_to_use = (
                args.served_model_name if args.served_model_name else args.model
            )

        return cls(
            base_url=args.base_url,
            api_key=args.api_key,
            model=model_to_use,
            served_model_name=served_model_name_to_use,
            tokenizer=args.tokenizer,
            pp_counts=[0] if args.messages_dataset else args.pp,
            tg_counts=args.tg,
            exact_tg=args.exact_tg,
            depths=args.depth,
            num_runs=args.runs,
            warmup_runs=args.warmup_runs,
            no_cache=args.no_cache,
            latency_mode=args.latency_mode,
            no_warmup=args.no_warmup,
            skip_coherence=args.skip_coherence,
            adapt_prompt=args.adapt_prompt,
            enable_prefix_caching=args.enable_prefix_caching,
            book_url=args.book_url,
            messages_dataset=args.messages_dataset,
            post_run_cmd=args.post_run_cmd,
            concurrency_levels=args.concurrency,
            save_result=args.save_result,
            result_format=args.format,
            save_total_throughput_timeseries=args.save_total_throughput_timeseries,
            save_all_throughput_timeseries=args.save_all_throughput_timeseries,
            exit_on_first_fail=args.exit_on_first_fail,
            no_results_on_fail=args.no_results_on_fail,
            extra_body=extra_body,
            emit_progress=args.emit_progress,
        )
