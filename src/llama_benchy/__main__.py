"""
Main entry point for the llama-benchy CLI.
"""

import argparse
import os
import time
import uuid
import subprocess
import datetime
import numpy as np
from tabulate import tabulate
import aiohttp
import asyncio
import json
import codecs
import hashlib
from transformers import AutoTokenizer
import requests

# Build number is now imported from __init__.py
from . import __version__



def format_result(values, multiplier=1.0):
    if not values:
        return ""
    mean = np.mean(values) * multiplier
    std = np.std(values) * multiplier
    return f"{mean:.2f} ± {std:.2f}"


def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM Benchmark Script")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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
    parser.add_argument("--book-url", type=str, default="https://www.gutenberg.org/files/1661/1661-0.txt", help="URL of a book to use for text generation, defaults to Sherlock Holmes (https://www.gutenberg.org/files/1661/1661-0.txt)")
    parser.add_argument("--latency-mode", type=str, default="api", choices=["api", "generation", "none"], help="Method to measure latency: 'api' (list models) - default, 'generation' (single token generation), or 'none' (skip latency measurement)")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup phase")
    parser.add_argument("--adapt-prompt", action="store_true", default=True, help="Adapt prompt size based on warmup token usage delta (default: True)")
    parser.add_argument("--no-adapt-prompt", action="store_false", dest="adapt_prompt", help="Disable prompt size adaptation")
    parser.add_argument("--enable-prefix-caching", action="store_true", help="Enable prefix caching performance measurement")
    parser.add_argument("--concurrency", type=int, nargs='+', default=[1], help="List of concurrency levels (simultaneous requests) - default: 1 (serial)")
    return parser.parse_args()


def get_tokenizer(model_name, tokenizer_name=None):
    try:
        name = tokenizer_name if tokenizer_name else model_name
        return AutoTokenizer.from_pretrained(name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Falling back to 'gpt2' tokenizer as approximation.")
        return AutoTokenizer.from_pretrained("gpt2")


def prepare_text_data(book_url, tokenizer):
    try:
        # Create cache directory if it doesn't exist
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "llama-benchy")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate hash of the URL for the filename
        url_hash = hashlib.md5(book_url.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"{url_hash}.txt")
        
        if os.path.exists(cache_file):
            print(f"Loading text from cache: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            print(f"Downloading book from {book_url}...")
            response = requests.get(book_url)
            response.raise_for_status()
            text = response.text
            # Basic cleanup
            start_idx = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
            if start_idx != -1:
                text = text[start_idx:]
            
            # Save to cache
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Saved text to cache: {cache_file}")
            
        return tokenizer.encode(text, add_special_tokens=False)
    except Exception as e:
        print(f"Error downloading book: {e}")
        exit(1)


def generate_prompt(all_tokens, tokenizer, prompt_tokens, context_tokens=0, no_cache=False):
    suffix = ""
    suffix_len = 0
    if no_cache:
        suffix = f" {uuid.uuid4()}"
        suffix_len = len(tokenizer.encode(suffix, add_special_tokens=False))
    
    # Adjust prompt tokens to fetch from text
    text_prompt_tokens = max(0, prompt_tokens - suffix_len)
    
    # Create a pool of tokens large enough
    total_needed = text_prompt_tokens + context_tokens
    
    if len(all_tokens) < total_needed:
        # Repeat tokens if not enough
        all_tokens = all_tokens * (total_needed // len(all_tokens) + 2)
    
    # Pick a random start position
    max_start = len(all_tokens) - total_needed
    start_idx = np.random.randint(0, max_start)
    
    selected_tokens = all_tokens[start_idx : start_idx + total_needed]
    
    context_text = tokenizer.decode(selected_tokens[:context_tokens]) if context_tokens > 0 else ""
    prompt_text = tokenizer.decode(selected_tokens[context_tokens:])
    
    if no_cache:
        prompt_text += suffix
        
    return context_text, prompt_text


async def measure_latency(session, base_url, api_key, mode="api", model_name=None):
    if mode == "none":
        print("Skipping latency measurement (assuming 0 ms).")
        return 0

    print(f"Measuring latency using mode: {mode}...")
    latencies = []
    headers = {"Authorization": f"Bearer {api_key}"}
    
    for _ in range(3):
        start = time.perf_counter()
        try:
            if mode == "api":
                async with session.get(f"{base_url}/models", headers=headers) as response:
                    await response.read()
                latencies.append(time.perf_counter() - start)
            elif mode == "generation":
                if not model_name:
                    raise ValueError("Model name required for generation latency mode")
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 1,
                    "stream": True
                }
                async with session.post(f"{base_url}/chat/completions", json=payload, headers=headers) as response:
                    async for _ in response.content:
                        # record latency as soon as the first byte is received
                        latencies.append(time.perf_counter() - start)
                        break
                    # Drain the rest of the response to keep the connection alive
                    async for _ in response.content: pass
        except Exception as e:
            print(f"Error measuring latency: {e}")
    
    if latencies:
        avg_latency = np.mean(latencies)
        print(f"Average latency ({mode}): {avg_latency*1000:.2f} ms")
        return avg_latency
    return 0


async def warmup(session, base_url, api_key, model, tokenizer=None):
    print("Warming up...")
    headers = {"Authorization": f"Bearer {api_key}"}
    warmup_text = "Warmup " * 10
    
    delta_user = 0
    delta_context = 0
    
    # 1. User only (No Context)
    payload_user = {
        "model": model,
        "messages": [{"role": "user", "content": warmup_text}],
        "max_tokens": 1
    }
    
    try:
        async with session.post(f"{base_url}/chat/completions", json=payload_user, headers=headers) as response:
            response_json = await response.json()
            if tokenizer:
                if 'usage' in response_json:
                    prompt_tokens = response_json['usage']['prompt_tokens']
                    local_tokens = len(tokenizer.encode(warmup_text, add_special_tokens=False))
                    delta_user = prompt_tokens - local_tokens
                    print(f"Warmup (User only) complete. Delta: {delta_user} tokens (Server: {prompt_tokens}, Local: {local_tokens})")
                else:
                    print("Warmup (User only) complete (no usage stats found).")
            else:
                print("Warmup complete.")

        if tokenizer:
            # 2. System + Empty User (Context Only)
            payload_sys_empty = {
                "model": model,
                "messages": [
                    {"role": "system", "content": warmup_text},
                    {"role": "user", "content": ""}
                ],
                "max_tokens": 1
            }
            async with session.post(f"{base_url}/chat/completions", json=payload_sys_empty, headers=headers) as response:
                response_json = await response.json()
                if 'usage' in response_json:
                    prompt_tokens = response_json['usage']['prompt_tokens']
                    local_tokens = len(tokenizer.encode(warmup_text, add_special_tokens=False))
                    delta_context = prompt_tokens - local_tokens
                    print(f"Warmup (System+Empty) complete. Delta: {delta_context} tokens (Server: {prompt_tokens}, Local: {local_tokens})")
                else:
                    print("Warmup (System+Empty) complete (no usage stats found).")
                    delta_context = delta_user

    except Exception as e:
        print(f"Warmup failed: {e}")
    return delta_user, delta_context


async def run_benchmark(session, base_url, api_key, model_name, context_text, prompt_text, expected_pp_tokens, tg, no_cache, latency, post_run_cmd):
    messages = []
    if context_text:
        messages.append({"role": "system", "content": context_text})
    messages.append({"role": "user", "content": prompt_text})
    
    ttft = 0
    e2e_ttft = 0
    token_count = 0
    first_token_time = 0
    first_response_time = 0
    prompt_usage_tokens = 0
    
    result = {
        "pp_speed": None,
        "tg_speed": None,
        "ttft": None,
        "ttfr": None,
        "est_ppt": None,
        "e2e_ttft": None,
        "token_count": 0
    }
    
    # DEBUG: Buffer to store first few lines of raw response
    debug_lines = []

    try:
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": tg,
            "stream": True,
            "stream_options": {"include_usage": True},
            # "temperature": 0,
            # "seed": 42
        }
        
        if no_cache:
            payload["cache_prompt"] = False
        
        headers = {"Authorization": f"Bearer {api_key}"}
        
        start_time = time.perf_counter()

        async with session.post(f"{base_url}/chat/completions", json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"Error: {response.status} - {error_text}")
                return None

            buffer = ""
            decoder = codecs.getincrementaldecoder("utf-8")(errors='replace')
            async for chunk_bytes in response.content:
                chunk_time = time.perf_counter()
                decoded_chunk = decoder.decode(chunk_bytes, final=False)
                buffer += decoded_chunk
                
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Capture first 5 lines for debugging if needed
                    if len(debug_lines) < 5:
                        debug_lines.append(line)

                    if line == 'data: [DONE]' or line == 'data:[DONE]':
                        continue
                    
                    if line.startswith('data:'):
                        try:
                            # Strip 'data:' and potential whitespace
                            json_str = line[5:].strip()
                            chunk = json.loads(json_str)

                            if 'usage' in chunk:
                                prompt_usage_tokens = chunk['usage'].get('prompt_tokens', 0)
                            
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                if first_response_time == 0:
                                    first_response_time = chunk_time

                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content')
                                reasoning_content = delta.get('reasoning_content')
                                reasoning = delta.get('reasoning')
                                
                                if content or reasoning_content or reasoning:
                                    if token_count == 0:
                                        first_token_time = chunk_time
                                        e2e_ttft = first_token_time - start_time
                                        ttft = e2e_ttft-latency
                                        if ttft < 0:
                                            ttft = 0
                                    
                                    token_count += 1
                        except json.JSONDecodeError:
                            continue
        
        end_time = time.perf_counter()
        
        # DEBUG: Print warning if no tokens were collected
        if token_count == 0:
            print(f"\n[Warning] Run generated 0 tokens. Raw response sample: {debug_lines}")
        
        if token_count > 0:
            # Calculate decode time (time for subsequent tokens)
            # If only 1 token, decode_time is effectively 0, so we can't calculate inter-token speed
            if token_count > 1:
                decode_time = end_time - first_token_time
                if decode_time > 0:
                    # Speed for the generated tokens (excluding the first one which is TTFT)
                    result["tg_speed"] = (token_count - 1) / decode_time
                else:
                    # Fallback if time is too small
                    result["tg_speed"] = (token_count - 1) / 0.0001
            
            # Use expected_pp_tokens for speed calculation
            total_prompt_tokens = expected_pp_tokens
            
            # Only use reported usage if it's close to expected (to handle tokenizer differences)
            # but not if it's vastly different (which happens in prefix caching where usage includes cached tokens)
            if prompt_usage_tokens > 0:
                diff = abs(prompt_usage_tokens - expected_pp_tokens)
                if diff < expected_pp_tokens * 0.2: # 20% tolerance
                     total_prompt_tokens = prompt_usage_tokens

            # Calculate TTFR and Estimated Prompt Processing Time
            ttfr = 0
            est_ppt = 0
            if first_response_time > 0:
                    ttfr = first_response_time - start_time
                    est_ppt = ttfr - latency
                    if est_ppt < 0: est_ppt = 0

            if est_ppt > 0:
                    result["pp_speed"] = total_prompt_tokens / est_ppt
                    result["est_ppt"] = est_ppt
            
            if ttfr > 0:
                    result["ttfr"] = ttfr
            
            if ttft > 0:
                result["ttft"] = ttft

            if e2e_ttft > 0:
                result["e2e_ttft"] = e2e_ttft

            result["token_count"] = token_count

    except Exception as e:
        print(f"Error during run: {e}")
        return None
    
    if post_run_cmd:
        try:
            subprocess.run(post_run_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Post-run command failed: {e}")

    return result


def adapt_token_counts(pp, depth, adapt_prompt, delta_user, delta_context):
    """Adjust pp and depth based on warmup token usage delta."""
    if adapt_prompt:
        if depth == 0:
            return max(1, pp - delta_user), depth
        else:
            return pp, max(1, depth - delta_context)
    return pp, depth


async def run_serial_test(session, base_url, api_key, model_name,
        all_tokens, tokenizer, pp, tg, depth, runs,
        no_cache, latency, post_run_cmd, adapt_prompt, delta_user, delta_context,
        enable_prefix_caching, display_model_name=None):
    """Run serial (single-request) benchmark tests and return formatted result rows."""
    display_name = display_model_name or model_name
    print(f"Running test: pp={pp}, tg={tg}, depth={depth}")
    pp_speeds = []
    tg_speeds = []
    ttft_values = []
    ttfr_values = []
    est_ppt_values = []
    e2e_ttft_values = []

    ctx_pp_speeds = []
    ctx_tg_speeds = []
    ctx_ttfr_values = []
    ctx_est_ppt_values = []
    ctx_e2e_ttft_values = []

    for run in range(runs):
        current_pp, current_depth = adapt_token_counts(pp, depth, adapt_prompt, delta_user, delta_context)
        context, prompt = generate_prompt(all_tokens, tokenizer, current_pp, current_depth, no_cache)

        if enable_prefix_caching and depth > 0:
            # Request 1: Context only — loads context into server cache
            print(f"  Run {run+1}/{runs} (Context Load)...")
            ctx_result = await run_benchmark(session, base_url, api_key, model_name, context, "", current_depth, tg, no_cache, latency, None)

            if ctx_result:
                if ctx_result["pp_speed"] is not None:
                    ctx_pp_speeds.append(ctx_result["pp_speed"])
                if ctx_result["tg_speed"] is not None:
                    ctx_tg_speeds.append(ctx_result["tg_speed"])
                if ctx_result["ttfr"] is not None:
                    ctx_ttfr_values.append(ctx_result["ttfr"])
                if ctx_result["est_ppt"] is not None:
                    ctx_est_ppt_values.append(ctx_result["est_ppt"])
                if ctx_result["e2e_ttft"] is not None:
                    ctx_e2e_ttft_values.append(ctx_result["e2e_ttft"])

            # Request 2: Inference with cached context
            print(f"  Run {run+1}/{runs} (Inference)...")
            run_result = await run_benchmark(session, base_url, api_key, model_name, context, prompt, current_pp, tg, no_cache, latency, post_run_cmd)
        else:
            # Standard run — expected PP tokens = current_pp + current_depth
            expected_tokens = current_pp + current_depth
            run_result = await run_benchmark(session, base_url, api_key, model_name, context, prompt, expected_tokens, tg, no_cache, latency, post_run_cmd)

        if run_result:
            if run_result["tg_speed"] is not None:
                tg_speeds.append(run_result["tg_speed"])
            if run_result["pp_speed"] is not None:
                pp_speeds.append(run_result["pp_speed"])
            if run_result["est_ppt"] is not None:
                est_ppt_values.append(run_result["est_ppt"])
            if run_result["ttfr"] is not None:
                ttfr_values.append(run_result["ttfr"])
            if run_result["ttft"] is not None:
                ttft_values.append(run_result["ttft"])
            if run_result["e2e_ttft"] is not None:
                e2e_ttft_values.append(run_result["e2e_ttft"])

    rows = []
    if ctx_pp_speeds:
        test_name = f"ctx_pp @ d{depth}"
        rows.append([
            display_name,
            test_name,
            format_result(ctx_pp_speeds),
            format_result(ctx_ttfr_values, 1000),
            format_result(ctx_est_ppt_values, 1000),
            format_result(ctx_e2e_ttft_values, 1000)
        ])

    if ctx_tg_speeds:
        test_name = f"ctx_tg @ d{depth}"
        rows.append([display_name, test_name, format_result(ctx_tg_speeds), "", "", ""])

    if pp_speeds:
        test_name = f"pp{pp}"
        if depth > 0:
            test_name += f" @ d{depth}"
        rows.append([
            display_name,
            test_name,
            format_result(pp_speeds),
            format_result(ttfr_values, 1000),
            format_result(est_ppt_values, 1000),
            format_result(e2e_ttft_values, 1000)
        ])

    if tg_speeds:
        test_name = f"tg{tg}"
        if depth > 0:
            test_name += f" @ d{depth}"
        rows.append([display_name, test_name, format_result(tg_speeds), "", "", ""])

    return rows


async def run_concurrent_batch(session, base_url, api_key, model_name,
        all_tokens, tokenizer, pp, tg, depth, concurrency,
        no_cache, latency, post_run_cmd, adapt_prompt, delta_user, delta_context):
    prompts = []
    expected_tokens_list = []
    for _ in range(concurrency):
        current_pp, current_depth = adapt_token_counts(pp, depth, adapt_prompt, delta_user, delta_context)
        context, prompt = generate_prompt(all_tokens, tokenizer, current_pp, current_depth, no_cache)
        prompts.append((context, prompt))
        expected_tokens_list.append(current_pp + current_depth)

    tasks = [
        run_benchmark(session, base_url, api_key, model_name,
                      ctx, pmt, exp_tok, tg, no_cache, latency, None)
        for (ctx, pmt), exp_tok in zip(prompts, expected_tokens_list)
    ]

    wall_start = time.perf_counter()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    wall_end = time.perf_counter()

    if post_run_cmd:
        try:
            subprocess.run(post_run_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Post-run command failed: {e}")

    return results, wall_start, wall_end, expected_tokens_list


def aggregate_concurrent_results(batch_results, wall_start, wall_end, expected_tokens_list):
    wall_time = wall_end - wall_start
    if wall_time <= 0:
        wall_time = 0.0001

    successful = []
    successful_expected_pp = []
    failed = 0
    for r, exp_pp in zip(batch_results, expected_tokens_list):
        if isinstance(r, Exception) or r is None:
            failed += 1
        else:
            successful.append(r)
            successful_expected_pp.append(exp_pp)

    total_pp_tokens = sum(successful_expected_pp)
    total_tg_tokens = sum(
        max(0, r.get("token_count", 0) - 1)
        for r in successful
        if r.get("tg_speed") is not None
    )

    agg_pp_tps = total_pp_tokens / wall_time
    agg_tg_tps = total_tg_tokens / wall_time
    req_per_sec = len(successful) / wall_time

    e2e_ttft_values = [r["e2e_ttft"] for r in successful if r.get("e2e_ttft") is not None]

    return {
        "agg_pp_tps": agg_pp_tps,
        "agg_tg_tps": agg_tg_tps,
        "req_per_sec": req_per_sec,
        "e2e_ttft_values": e2e_ttft_values,
        "failed": failed,
        "successful": len(successful),
    }


async def main_async():
    args = parse_arguments()
    
    if args.enable_prefix_caching and args.no_cache:
        print("Error: --enable-prefix-caching and --no-cache are incompatible.")
        return

    args.concurrency = sorted(set(max(1, c) for c in args.concurrency))

    version_number = __version__

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"llama-benchy ({version_number})")
    print(f"Date: {current_time}")
    print(f"Benchmarking model: {args.model} at {args.base_url}")
    
    served_model_name = args.served_model_name if args.served_model_name else args.model

    tokenizer = get_tokenizer(args.model, args.tokenizer)
    all_tokens = prepare_text_data(args.book_url, tokenizer)
    print(f"Total tokens available in text corpus: {len(all_tokens)}")
    
    # Use a large timeout for long-running benchmarks
    timeout = aiohttp.ClientTimeout(total=3600)
    connector = aiohttp.TCPConnector(limit=max(args.concurrency), force_close=False, keepalive_timeout=600)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=True) as session:
        delta_user = 0
        delta_context = 0
        should_warmup = not args.no_warmup
        if args.adapt_prompt:
            should_warmup = True
            
        if should_warmup:
            delta_user, delta_context = await warmup(session, args.base_url, args.api_key, served_model_name, tokenizer if args.adapt_prompt else None)

        latency = await measure_latency(session, args.base_url, args.api_key, args.latency_mode, served_model_name)
        
        has_concurrent = any(c > 1 for c in args.concurrency)
        if has_concurrent and args.enable_prefix_caching:
            print("Warning: --enable-prefix-caching is not supported with concurrency > 1. Prefix caching tests will only run at concurrency 1.")
        if has_concurrent and not args.no_cache and not args.enable_prefix_caching:
            print("Warning: Running concurrent tests without --no-cache. Server-side KV cache may inflate throughput numbers. Use --no-cache for reliable results.")

        results = []
        concurrent_results = []

        for depth in args.depth:
            for pp in args.pp:
                for tg in args.tg:
                    for c in args.concurrency:
                        if c == 1:
                            rows = await run_serial_test(
                                session, args.base_url, args.api_key, served_model_name,
                                all_tokens, tokenizer, pp, tg, depth, args.runs,
                                args.no_cache, latency, args.post_run_cmd,
                                args.adapt_prompt, delta_user, delta_context,
                                args.enable_prefix_caching,
                                display_model_name=args.model)
                            results.extend(rows)

                        else:
                            # Concurrent path
                            print(f"Running concurrent test: pp={pp}, tg={tg}, depth={depth}, concurrency={c}")
                            batch_agg_pp = []
                            batch_agg_tg = []
                            batch_req_per_sec = []
                            batch_e2e_ttft_means = []
                            batch_e2e_ttft_all = []
                            batch_failed = 0

                            for run in range(args.runs):
                                print(f"  Batch {run+1}/{args.runs} (concurrency={c})...")
                                batch_results, wall_start, wall_end, expected_tokens_list = await run_concurrent_batch(
                                    session, args.base_url, args.api_key, served_model_name,
                                    all_tokens, tokenizer, pp, tg, depth, c,
                                    args.no_cache, latency, args.post_run_cmd,
                                    args.adapt_prompt, delta_user, delta_context)

                                agg = aggregate_concurrent_results(batch_results, wall_start, wall_end, expected_tokens_list)
                                batch_agg_pp.append(agg["agg_pp_tps"])
                                batch_agg_tg.append(agg["agg_tg_tps"])
                                batch_req_per_sec.append(agg["req_per_sec"])
                                if agg["e2e_ttft_values"]:
                                    batch_e2e_ttft_means.append(np.mean(agg["e2e_ttft_values"]))
                                batch_e2e_ttft_all.extend(agg["e2e_ttft_values"])
                                batch_failed += agg["failed"]

                            test_name = f"pp{pp}/tg{tg} x{c}"
                            if depth > 0: test_name += f" @ d{depth}"

                            avg_e2e = format_result(batch_e2e_ttft_means, 1000) if batch_e2e_ttft_means else ""
                            p99_e2e = ""
                            if batch_e2e_ttft_all:
                                n = len(batch_e2e_ttft_all)
                                if n >= 20:
                                    p99_val = np.percentile(batch_e2e_ttft_all, 99) * 1000
                                    p99_e2e = f"{p99_val:.2f}"
                                else:
                                    max_val = max(batch_e2e_ttft_all) * 1000
                                    p99_e2e = f"{max_val:.2f} (max)"

                            concurrent_results.append([
                                args.model,
                                test_name,
                                format_result(batch_agg_pp),
                                format_result(batch_agg_tg),
                                format_result(batch_req_per_sec),
                                avg_e2e,
                                p99_e2e,
                                str(batch_failed)
                            ])

        print()
        if not results and not concurrent_results:
            print("No results collected. Check if the model is generating tokens.")

        if results:
            print(tabulate(results, headers=["model", "test", "t/s", "ttfr (ms)", "est_ppt (ms)", "e2e_ttft (ms)"], tablefmt="pipe", colalign=("left", "right", "right", "right", "right", "right")))

        if concurrent_results:
            if results:
                print()
            print("Concurrent throughput results:")
            print(tabulate(concurrent_results, headers=["model", "test", "pp agg t/s", "tg agg t/s", "req/s", "avg e2e_ttft (ms)", "p99 e2e_ttft (ms)", "errors"], tablefmt="pipe", colalign=("left", "right", "right", "right", "right", "right", "right", "right")))

        if results or concurrent_results:
            print(f"\nllama-benchy ({version_number})")
            concurrency_str = ""
            if has_concurrent:
                levels = sorted(c for c in args.concurrency if c > 1)
                concurrency_str = f" | concurrency: {','.join(str(c) for c in levels)}"
            print(f"date: {current_time} | latency mode: {args.latency_mode}{concurrency_str}")


def main():
    """Entry point for the CLI command."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()