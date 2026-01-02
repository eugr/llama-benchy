import argparse
import os
# Disable tokenizers parallelism to avoid warnings/errors when forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
import time
import uuid
import subprocess
import numpy as np
from tabulate import tabulate
import aiohttp
import asyncio
import json
import codecs
from transformers import AutoTokenizer
import requests

def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM Benchmark Script")
    parser.add_argument("--base-url", type=str, required=True, help="OpenAI compatible endpoint URL")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API Key for the endpoint")
    parser.add_argument("--model", type=str, required=True, help="Model name to use for benchmarking")
    parser.add_argument("--tokenizer", type=str, default=None, help="HuggingFace tokenizer name (defaults to model name)")
    parser.add_argument("--pp", type=int, nargs='+', required=True, help="List of prompt processing token counts")
    parser.add_argument("--tg", type=int, nargs='+', required=True, help="List of token generation counts")
    parser.add_argument("--depth", type=int, nargs='+', default=[0], help="List of context depths (previous conversation tokens)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per test")
    parser.add_argument("--no-cache", action="store_true", help="Ensure unique requests to avoid prefix caching")
    parser.add_argument("--post-run-cmd", type=str, default=None, help="Command to execute after each test run")
    parser.add_argument("--book-url", type=str, default="https://www.gutenberg.org/files/11/11-0.txt", help="URL of a book to use for text generation")
    parser.add_argument("--latency-mode", type=str, default="models", choices=["models", "generation"], help="Method to measure latency: 'models' (list models) or 'generation' (single token generation)")
    return parser.parse_args()

def get_tokenizer(model_name, tokenizer_name=None):
    try:
        name = tokenizer_name if tokenizer_name else model_name
        return AutoTokenizer.from_pretrained(name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Falling back to 'gpt2' tokenizer as approximation.")
        return AutoTokenizer.from_pretrained("gpt2")

def prepare_text_data(book_url):
    try:
        response = requests.get(book_url)
        response.raise_for_status()
        text = response.text
        # Basic cleanup
        start_idx = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
        if start_idx != -1:
            text = text[start_idx:]
        return text
    except Exception as e:
        print(f"Error downloading book: {e}")
        print("Using synthetic data.")
        return " ".join(["word"] * 100000)

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

async def measure_latency(session, base_url, api_key, mode="models", model_name=None):
    print(f"Measuring latency using mode: {mode}...")
    latencies = []
    headers = {"Authorization": f"Bearer {api_key}"}
    
    for _ in range(3):
        start = time.perf_counter()
        try:
            if mode == "models":
                async with session.get(f"{base_url}/models", headers=headers) as response:
                    await response.read()
            elif mode == "generation":
                if not model_name:
                    raise ValueError("Model name required for generation latency mode")
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 1
                }
                async with session.post(f"{base_url}/chat/completions", json=payload, headers=headers) as response:
                    await response.read()
            latencies.append(time.perf_counter() - start)
        except Exception as e:
            print(f"Error measuring latency: {e}")
    
    if latencies:
        avg_latency = np.mean(latencies)
        print(f"Average latency ({mode}): {avg_latency*1000:.2f} ms")
        return avg_latency
    return 0

async def warmup(session, base_url, api_key, model):
    print("Warming up...")
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Warmup " * 10}],
        "max_tokens": 1
    }
    try:
        async with session.post(f"{base_url}/chat/completions", json=payload, headers=headers) as response:
            await response.read()
        print("Warmup complete.")
    except Exception as e:
        print(f"Warmup failed: {e}")

async def main():
    args = parse_arguments()
    print(f"Benchmarking model: {args.model} at {args.base_url}")
    
    tokenizer = get_tokenizer(args.model, args.tokenizer)
    text_data = prepare_text_data(args.book_url)
    all_tokens = tokenizer.encode(text_data, add_special_tokens=False)
    
    # Use a large timeout for long-running benchmarks
    timeout = aiohttp.ClientTimeout(total=3600)
    connector = aiohttp.TCPConnector(limit=1, force_close=False, keepalive_timeout=600)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=True) as session:
        await warmup(session, args.base_url, args.api_key, args.model)
        latency = await measure_latency(session, args.base_url, args.api_key, args.latency_mode, args.model)
        
        results = []
        
        for depth in args.depth:
            for pp in args.pp:
                for tg in args.tg:
                    print(f"Running test: pp={pp}, tg={tg}, depth={depth}")
                    pp_speeds = []
                    tg_speeds = []
                    ttft_values = []
                    e2e_ttft_values = []
                    
                    for run in range(args.runs):
                        context, prompt = generate_prompt(all_tokens, tokenizer, pp, depth, args.no_cache)
                        messages = []
                        if context:
                            messages.append({"role": "system", "content": context})
                        messages.append({"role": "user", "content": prompt})
                        
                        ttft = 0
                        e2e_ttft = 0
                        token_count = 0
                        first_token_time = 0
                        
                        try:
                            payload = {
                                "model": args.model,
                                "messages": messages,
                                "max_tokens": tg,
                                "stream": True,
                                # "temperature": 0,
                                # "seed": 42
                            }
                            
                            if args.no_cache:
                                payload["cache_prompt"] = False
                            
                            headers = {"Authorization": f"Bearer {args.api_key}"}
                            
                            start_time = time.perf_counter()

                            async with session.post(f"{args.base_url}/chat/completions", json=payload, headers=headers) as response:
                                if response.status != 200:
                                    error_text = await response.text()
                                    print(f"Error: {response.status} - {error_text}")
                                    continue

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
                                                if 'choices' in chunk and len(chunk['choices']) > 0:
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
                                        tg_speeds.append((token_count - 1) / decode_time)
                                    else:
                                        # Fallback if time is too small
                                        tg_speeds.append((token_count - 1) / 0.0001)
                                
                                if ttft > 0:
                                    # Use total prompt tokens (pp + depth) for speed calculation
                                    # as the server processes the full context (especially with no-cache).
                                    total_prompt_tokens = pp + depth
                                    pp_speeds.append(total_prompt_tokens / ttft)
                                    ttft_values.append(ttft)

                                if e2e_ttft > 0:
                                    e2e_ttft_values.append(e2e_ttft)

                        except Exception as e:
                            print(f"Error during run: {e}")
                            continue
                        
                        if args.post_run_cmd:
                            try:
                                subprocess.run(args.post_run_cmd, shell=True, check=True)
                            except subprocess.CalledProcessError as e:
                                print(f"Post-run command failed: {e}")

                    # Aggregate results
                    if pp_speeds:
                        pp_mean = np.mean(pp_speeds)
                        pp_std = np.std(pp_speeds)
                        
                        ttft_str = ""
                        if ttft_values:
                            ttft_mean = np.mean(ttft_values) * 1000
                            ttft_std = np.std(ttft_values) * 1000
                            ttft_str = f"{ttft_mean:.2f} ± {ttft_std:.2f}"

                        e2e_ttft_str = ""
                        if e2e_ttft_values:
                            e2e_ttft_mean = np.mean(e2e_ttft_values) * 1000
                            e2e_ttft_std = np.std(e2e_ttft_values) * 1000
                            e2e_ttft_str = f"{e2e_ttft_mean:.2f} ± {e2e_ttft_std:.2f}"

                        test_name = f"pp{pp}"
                        if depth > 0:
                            test_name += f" @ d{depth}"
                        results.append([args.model, test_name, f"{pp_mean:.2f} ± {pp_std:.2f}", ttft_str, e2e_ttft_str])
                    
                    if tg_speeds:
                        tg_mean = np.mean(tg_speeds)
                        tg_std = np.std(tg_speeds)
                        test_name = f"tg{tg}"
                        if depth > 0:
                            test_name += f" @ d{depth}"
                        results.append([args.model, test_name, f"{tg_mean:.2f} ± {tg_std:.2f}", "", ""])

        if not results:
            print("No results collected. Check if the model is generating tokens.")
        else:
            print(tabulate(results, headers=["model", "test", "t/s", "ttft (ms)", "e2e_ttft (ms)"], tablefmt="pipe", colalign=("left", "right", "right", "right", "right")))

if __name__ == "__main__":
    asyncio.run(main())
