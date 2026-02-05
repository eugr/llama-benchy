import time
import json
import codecs
import aiohttp
import asyncio
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class RequestResult:
    tg_speed: Optional[float] = None
    pp_speed: Optional[float] = None
    ttft: Optional[float] = None
    ttfr: Optional[float] = None
    est_ppt: Optional[float] = None
    e2e_ttft: Optional[float] = None
    start_ts: float = 0.0
    end_ts: float = 0.0
    total_tokens: int = 0
    error: Optional[str] = None

class LLMClient:
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.headers = {"Authorization": f"Bearer {api_key}"}

    async def measure_latency(self, session: aiohttp.ClientSession, mode: str = "api") -> float:
        if mode == "none":
            print("Skipping latency measurement (assuming 0 ms).")
            return 0

        print(f"Measuring latency using mode: {mode}...")
        latencies = []
        
        for _ in range(3):
            start = time.perf_counter()
            try:
                if mode == "api":
                    async with session.get(f"{self.base_url}/models", headers=self.headers) as response:
                        await response.read()
                    latencies.append(time.perf_counter() - start)
                elif mode == "generation":
                    payload = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 1,
                        "stream": True
                    }
                    async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=self.headers) as response:
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

    async def warmup(self, session: aiohttp.ClientSession, tokenizer=None):
        print("Warming up...")
        warmup_text = "Warmup " * 10
        
        delta_user = 0
        delta_context = 0
        
        # 1. User only (No Context)
        payload_user = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": warmup_text}],
            "max_tokens": 1
        }
        
        try:
            async with session.post(f"{self.base_url}/chat/completions", json=payload_user, headers=self.headers) as response:
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
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": warmup_text},
                        {"role": "user", "content": ""}
                    ],
                    "max_tokens": 1
                }
                async with session.post(f"{self.base_url}/chat/completions", json=payload_sys_empty, headers=self.headers) as response:
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

    async def run_generation(
            self, 
            session: aiohttp.ClientSession, 
            context_text: str, 
            prompt_text: str, 
            expected_pp_tokens: int, # Used for calculating PP speed if usage info is unreliable or unavailable
            max_tokens: int, 
            no_cache: bool, 
            latency_offset: float
        ) -> Optional[RequestResult]:

        messages = []
        if context_text:
            messages.append({"role": "system", "content": context_text})
        messages.append({"role": "user", "content": prompt_text})
        
        ttft = 0.0
        e2e_ttft = 0.0
        token_count = 0
        first_token_time = 0.0
        first_response_time = 0.0
        prompt_usage_tokens = 0
        
        result = RequestResult()
        
        # DEBUG: Buffer to store first few lines of raw response
        debug_lines = []

        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            
            if no_cache:
                payload["cache_prompt"] = False
            
            start_time = time.perf_counter()
            result.start_ts = start_time

            async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=self.headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error: {response.status} - {error_text}")
                    result.error = f"HTTP {response.status}: {error_text}"
                    return result

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
                                            ttft = e2e_ttft - latency_offset
                                            if ttft < 0:
                                                ttft = 0
                                        
                                        token_count += 1
                            except json.JSONDecodeError:
                                continue
            
            end_time = time.perf_counter()
            result.end_ts = end_time
            result.total_tokens = token_count
            
            # DEBUG: Print warning if no tokens were collected
            if token_count == 0:
                print(f"\n[Warning] Run generated 0 tokens. Raw response sample: {debug_lines}")
            
            if token_count > 0:
                # Calculate decode time (time for subsequent tokens)
                if token_count > 1:
                    decode_time = end_time - first_token_time
                    if decode_time > 0:
                        # Speed for the generated tokens (excluding the first one which is TTFT)
                        result.tg_speed = (token_count - 1) / decode_time
                    else:
                        result.tg_speed = (token_count - 1) / 0.0001
                
                # Use expected_pp_tokens for speed calculation
                total_prompt_tokens = expected_pp_tokens
                
                # Only use reported usage if it's close to expected (to handle tokenizer differences)
                if prompt_usage_tokens > 0:
                    diff = abs(prompt_usage_tokens - expected_pp_tokens)
                    if diff < expected_pp_tokens * 0.2: # 20% tolerance
                         total_prompt_tokens = prompt_usage_tokens

                # Calculate TTFR and Estimated Prompt Processing Time
                ttfr = 0.0
                est_ppt = 0.0
                if first_response_time > 0:
                    ttfr = first_response_time - start_time
                    est_ppt = ttfr - latency_offset
                    if est_ppt < 0: est_ppt = 0

                if est_ppt > 0:
                    result.pp_speed = total_prompt_tokens / est_ppt
                    result.est_ppt = est_ppt
                
                if ttfr > 0:
                    result.ttfr = ttfr
                
                if ttft > 0:
                    result.ttft = ttft

                if e2e_ttft > 0:
                    result.e2e_ttft = e2e_ttft

        except Exception as e:
            print(f"Error during run: {e}")
            import traceback
            traceback.print_exc()
            result.error = str(e)
            return result
        
        return result
