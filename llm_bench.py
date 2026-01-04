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

def get_git_revision_short_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM Benchmark Script")
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
    parser.add_argument("--adapt-prompt", action="store_true", help="Adapt prompt size based on warmup token usage delta")
    parser.add_argument("--enable-prefix-caching", action="store_true", help="Enable prefix caching performance measurement")
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
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "llama-bench-4all")
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
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": warmup_text}],
        "max_tokens": 1
    }
    token_usage_delta = 0
    try:
        async with session.post(f"{base_url}/chat/completions", json=payload, headers=headers) as response:
            response_json = await response.json()
            if tokenizer:
                if 'usage' in response_json:
                    prompt_tokens = response_json['usage']['prompt_tokens']
                    local_tokens = len(tokenizer.encode(warmup_text, add_special_tokens=False))
                    token_usage_delta = prompt_tokens - local_tokens
                    print(f"Warmup complete. Delta: {token_usage_delta} tokens (Server: {prompt_tokens}, Local: {local_tokens})")
                else:
                    print("Warmup complete (no usage stats found).")
            else:
                print("Warmup complete.")
    except Exception as e:
        print(f"Warmup failed: {e}")
    return token_usage_delta

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
        "e2e_ttft": None
    }

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
                    if not line or line == 'data: [DONE]':
                        continue
                    
                    if line.startswith('data: '):
                        try:
                            chunk = json.loads(line[6:])
                            if 'usage' in chunk:
                                prompt_usage_tokens = chunk['usage'].get('prompt_tokens', 0)
                            
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                if first_response_time == 0:
                                    first_response_time = chunk_time

                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content')
                                reasoning_content = delta.get('reasoning_content')
                                
                                if content or reasoning_content:
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

    except Exception as e:
        print(f"Error during run: {e}")
        return None
    
    if post_run_cmd:
        try:
            subprocess.run(post_run_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Post-run command failed: {e}")

    return result

async def main():
    args = parse_arguments()
    
    if args.enable_prefix_caching and args.no_cache:
        print("Error: --enable-prefix-caching and --no-cache are incompatible.")
        return

    build_number = get_git_revision_short_hash()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"llama-bench-4all (build: {build_number})")
    print(f"Date: {current_time}")
    print(f"Benchmarking model: {args.model} at {args.base_url}")
    
    served_model_name = args.served_model_name if args.served_model_name else args.model

    tokenizer = get_tokenizer(args.model, args.tokenizer)
    all_tokens = prepare_text_data(args.book_url, tokenizer)
    print(f"Total tokens available in text corpus: {len(all_tokens)}")
    
    # Use a large timeout for long-running benchmarks
    timeout = aiohttp.ClientTimeout(total=3600)
    connector = aiohttp.TCPConnector(limit=1, force_close=False, keepalive_timeout=600)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=True) as session:
        token_usage_delta = 0
        should_warmup = not args.no_warmup
        if args.adapt_prompt:
            should_warmup = True
            
        if should_warmup:
            token_usage_delta = await warmup(session, args.base_url, args.api_key, served_model_name, tokenizer if args.adapt_prompt else None)

        latency = await measure_latency(session, args.base_url, args.api_key, args.latency_mode, served_model_name)
        
        results = []
        
        for depth in args.depth:
            for pp in args.pp:
                for tg in args.tg:
                    print(f"Running test: pp={pp}, tg={tg}, depth={depth}")
                    pp_speeds = []
                    tg_speeds = []
                    ttft_values = []
                    ttfr_values = []
                    est_ppt_values = []
                    e2e_ttft_values = []
                    
                    ctx_pp_speeds = []
                    ctx_tg_speeds = []
                    
                    for run in range(args.runs):
                        current_pp = pp
                        current_depth = depth
                        if args.adapt_prompt:
                            if depth == 0:
                                current_pp = max(1, pp - token_usage_delta)
                            else:
                                current_depth = max(1, depth - token_usage_delta)
                        print(f"current_pp={current_pp}, current_depth={current_depth}")
                        context, prompt = generate_prompt(all_tokens, tokenizer, current_pp, current_depth, args.no_cache)
                        
                        if args.enable_prefix_caching and depth > 0:
                            # Request 1: Context only
                            # We send context as system message, and empty prompt as user message.
                            # This establishes the prefix: [System: Context] [User: ""]
                            # Expected PP tokens = current_depth (context size)
                            print(f"  Run {run+1}/{args.runs} (Context Load)...")
                            ctx_result = await run_benchmark(session, args.base_url, args.api_key, served_model_name, context, "", current_depth, tg, args.no_cache, latency, None)
                            
                            if ctx_result:
                                if ctx_result["pp_speed"] is not None:
                                    ctx_pp_speeds.append(ctx_result["pp_speed"])
                                if ctx_result["tg_speed"] is not None:
                                    ctx_tg_speeds.append(ctx_result["tg_speed"])
                            
                            # Request 2: Context + Prompt
                            # We send context as system message, and prompt as user message.
                            # The prefix [System: Context] should be cached.
                            # Expected PP tokens = current_pp (prompt size only)
                            print(f"  Run {run+1}/{args.runs} (Inference)...")
                            run_result = await run_benchmark(session, args.base_url, args.api_key, served_model_name, context, prompt, current_pp, tg, args.no_cache, latency, args.post_run_cmd)
                        else:
                            # Standard run
                            # Expected PP tokens = current_pp + current_depth
                            expected_tokens = current_pp + current_depth
                            run_result = await run_benchmark(session, args.base_url, args.api_key, served_model_name, context, prompt, expected_tokens, tg, args.no_cache, latency, args.post_run_cmd)
                        
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

                    # Aggregate results
                    def format_result(values, multiplier=1.0):
                        if not values: return ""
                        mean = np.mean(values) * multiplier
                        std = np.std(values) * multiplier
                        return f"{mean:.2f} ± {std:.2f}"

                    # Context PP (if enabled)
                    if ctx_pp_speeds:
                        test_name = f"ctx_pp @ d{depth}"
                        results.append([args.model, test_name, format_result(ctx_pp_speeds), "", "", ""])

                    # Context TG (if enabled)
                    if ctx_tg_speeds:
                        test_name = f"ctx_tg @ d{depth}"
                        results.append([args.model, test_name, format_result(ctx_tg_speeds), "", "", ""])

                    # Standard PP
                    if pp_speeds:
                        test_name = f"pp{pp}"
                        if depth > 0: test_name += f" @ d{depth}"
                        results.append([
                            args.model, 
                            test_name, 
                            format_result(pp_speeds), 
                            format_result(ttfr_values, 1000), 
                            format_result(est_ppt_values, 1000), 
                            format_result(e2e_ttft_values, 1000)
                        ])
                    
                    # Standard TG
                    if tg_speeds:
                        test_name = f"tg{tg}"
                        if depth > 0: test_name += f" @ d{depth}"
                        results.append([args.model, test_name, format_result(tg_speeds), "", "", ""])

        print()
        if not results:
            print("No results collected. Check if the model is generating tokens.")
        else:
            print(tabulate(results, headers=["model", "test", "t/s", "ttfr (ms)", "est_ppt (ms)", "e2e_ttft (ms)"], tablefmt="pipe", colalign=("left", "right", "right", "right", "right", "right")))
            print(f"\nllama-bench-4all (build: {build_number})")
            print(f"date: {current_time} | latency mode: {args.latency_mode}")

if __name__ == "__main__":
    asyncio.run(main())
