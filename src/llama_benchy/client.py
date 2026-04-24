import time
import json
import codecs
import aiohttp
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class RequestResult:
    start_ts: float = 0.0
    end_ts: float = 0.0
    first_token_ts: Optional[float] = None
    first_response_ts: Optional[float] = None
    prompt_tokens: int = 0
    total_tokens: int = 0
    error: Optional[str] = None
    token_timestamps: List[float] = field(default_factory=list)

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
                            latencies.append(time.perf_counter() - start)
                            break
                        async for _ in response.content: pass
            except Exception as e:
                print(f"Error measuring latency: {e}")
        
        if latencies:
            avg_latency = float(np.mean(latencies))
            print(f"Average latency ({mode}): {avg_latency*1000:.2f} ms")
            return avg_latency
        return 0

    async def run_coherence_test(self, session: aiohttp.ClientSession) -> bool:
        """Run coherence test after warmup to verify model responds correctly."""
        print("\nRunning coherence test...")
        prompt = "What is the capital of France? Please reply with one word only"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100
        }

        try:
            async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=self.headers) as response:
                response_json = await response.json()

                if 'choices' not in response_json or len(response_json['choices']) == 0:
                    print("Coherence test FAILED: No choices in response")
                    return False

                choice = response_json['choices'][0].get('message', {})
                content = choice.get('content') or ''
                # Also check reasoning fields for thinking models
                reasoning = choice.get('reasoning') or choice.get('reasoning_content') or ''
                full_content = (content + reasoning).lower()

                if 'paris' in full_content:
                    print("Coherence test PASSED.")
                    return True
                else:
                    print(f"Coherence test FAILED: Expected 'Paris'. Got: {content[:200]}...")
                    return False
        except Exception as e:
            print(f"Coherence test FAILED with error: {e}")
            return False

    async def warmup(self, session: aiohttp.ClientSession, tokenizer=None):
        print("Warming up...")
        warmup_text = "Warmup " * 10

        delta_user = 0
        delta_context = 0

        # 1. User only
        payload_user = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": warmup_text}],
            "max_tokens": 1
        }

        try:
            async with session.post(f"{self.base_url}/chat/completions", json=payload_user, headers=self.headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Warmup failed: HTTP {response.status}: {error_text}")
                    raise SystemExit(1)
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
                # 2. Context Only
                payload_sys_empty = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": warmup_text},
                        {"role": "user", "content": ""}
                    ],
                    "max_tokens": 1
                }
                async with session.post(f"{self.base_url}/chat/completions", json=payload_sys_empty, headers=self.headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Warmup failed: HTTP {response.status}: {error_text}")
                        raise SystemExit(1)
                    response_json = await response.json()
                    if 'usage' in response_json:
                        prompt_tokens = response_json['usage']['prompt_tokens']
                        local_tokens = len(tokenizer.encode(warmup_text, add_special_tokens=False))
                        delta_context = prompt_tokens - local_tokens
                        print(f"Warmup (System+Empty) complete. Delta: {delta_context} tokens (Server: {prompt_tokens}, Local: {local_tokens})")
                    else:
                        delta_context = delta_user
        except Exception as e:
            print(f"Warmup failed: {e}")
            raise SystemExit(1)
        return delta_user, delta_context

    async def run_generation(
            self, 
            session: aiohttp.ClientSession, 
            context_text: str, 
            prompt_text: str, 
            max_tokens: int, 
            no_cache: bool
        ) -> RequestResult:

        messages = []
        if context_text:
            messages.append({"role": "system", "content": context_text})
        messages.append({"role": "user", "content": prompt_text})
        
        result = RequestResult()
        
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
            
            result.start_ts = time.perf_counter()

            async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=self.headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    result.error = f"HTTP {response.status}: {error_text}"
                    print(result.error)
                    return result

                decoder = codecs.getincrementaldecoder("utf-8")(errors='replace')
                buffer = ""
                
                async for chunk_bytes in response.content:
                    chunk_time = time.perf_counter()
                    decoded_chunk = decoder.decode(chunk_bytes, final=False)
                    buffer += decoded_chunk
                    
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        
                        if line == 'data: [DONE]' or line == 'data:[DONE]':
                            continue
                        
                        if line.startswith('data:'):
                            try:
                                json_str = line[5:].strip()
                                chunk = json.loads(json_str)

                                if 'usage' in chunk and chunk['usage'] is not None:
                                    result.prompt_tokens = chunk['usage'].get('prompt_tokens', 0)
                                    # Use server-side completion_tokens when available; spec decoding
                                    # can pack multiple tokens per SSE chunk, so chunk-counting undercounts.
                                    completion_tokens = chunk['usage'].get('completion_tokens')
                                    if completion_tokens is not None and completion_tokens > result.total_tokens:
                                        result.total_tokens = completion_tokens

                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    if result.first_response_ts is None:
                                        result.first_response_ts = chunk_time

                                    delta = chunk['choices'][0].get('delta', {})
                                    content = delta.get('content')
                                    reasoning_content = delta.get('reasoning_content')
                                    reasoning = delta.get('reasoning')
                                    
                                    if content or reasoning_content or reasoning:
                                        if result.first_token_ts is None:
                                            result.first_token_ts = chunk_time
                                        
                                        result.total_tokens += 1
                                        result.token_timestamps.append(chunk_time)
                            except json.JSONDecodeError:
                                continue
            
            result.end_ts = time.perf_counter()

        except Exception as e:
            print(f"Error during run: {e}")
            result.error = str(e)
        
        return result
