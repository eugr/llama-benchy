import time
import json
import codecs
import aiohttp
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

_warned_about_fallback = False

# vLLM's Rust frontend rejects empty user content, so context preloads use a
# tiny probe turn instead.
CONTEXT_LOAD_USER_MESSAGE = "."


def _warn_once(message: str):
    global _warned_about_fallback
    if not _warned_about_fallback:
        print(message)
        _warned_about_fallback = True

@dataclass
class RequestResult:
    start_ts: float = 0.0
    end_ts: float = 0.0
    first_token_ts: Optional[float] = None
    first_response_ts: Optional[float] = None
    prompt_tokens: int = 0
    total_tokens: int = 0
    server_decode_seconds: Optional[float] = None
    error: Optional[str] = None
    token_timestamps: List[float] = field(default_factory=list)

class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        extra_body: Optional[Dict[str, Any]] = None,
        exact_tg: bool = False,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.extra_body = extra_body or {}
        self.exact_tg = exact_tg
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def _build_generation_payload(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        no_cache: bool,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "return_token_ids": True,
            "stream_options": {"include_usage": True},
        }

        if tools:
            payload["tools"] = tools

        if no_cache:
            payload["cache_prompt"] = False

        payload.update(self.extra_body)

        if self.exact_tg:
            payload["max_tokens"] = max_tokens
            payload["min_tokens"] = max_tokens
            payload["ignore_eos"] = True

        return payload

    @staticmethod
    def _non_empty_user_content(content: str) -> str:
        if content.strip():
            return content
        return CONTEXT_LOAD_USER_MESSAGE

    @staticmethod
    def _append_observed_token_timestamps(result: RequestResult, chunk_time: float, token_count: int):
        if token_count <= 0:
            return

        if token_count == 1:
            result.token_timestamps.append(chunk_time)
            return

        last_ts = result.token_timestamps[-1] if result.token_timestamps else result.first_token_ts
        if last_ts is None:
            last_ts = result.start_ts
        time_window = chunk_time - last_ts
        for i in range(token_count):
            ts = last_ts + (time_window * (i + 1) / token_count)
            result.token_timestamps.append(ts)

    @staticmethod
    def _interpolate_token_timestamps(chunk_times: List[float], token_count: int) -> List[float]:
        if token_count <= 0 or not chunk_times:
            return []

        if token_count == 1 or len(chunk_times) == 1:
            return [chunk_times[0]] * token_count

        first_ts = chunk_times[0]
        last_ts = chunk_times[-1]
        if last_ts <= first_ts:
            return [first_ts] * token_count

        step = (last_ts - first_ts) / (token_count - 1)
        return [first_ts + (step * i) for i in range(token_count)]

    @staticmethod
    def _tool_call_text(tool_calls: Any) -> str:
        if not isinstance(tool_calls, list):
            return ""
        parts = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            arguments = function.get("arguments")
            if isinstance(name, str):
                parts.append(name)
            if isinstance(arguments, str):
                parts.append(arguments)
        return "".join(parts)

    @staticmethod
    def _apply_server_decode_timestamps(result: RequestResult) -> None:
        seconds = result.server_decode_seconds
        if seconds is None or seconds <= 0 or result.total_tokens <= 0 or result.end_ts <= 0:
            return
        step = seconds / result.total_tokens
        first_ts = result.end_ts - seconds + step
        result.token_timestamps = [
            first_ts + step * index for index in range(result.total_tokens)
        ]
        result.first_token_ts = first_ts

    def _finalize_stream_tokens(
            self,
            result: RequestResult,
            content_chunks: List[Dict[str, Any]],
            usage_completion_tokens: Optional[int],
            tokenizer=None
        ):
        if not content_chunks:
            if usage_completion_tokens is not None:
                result.total_tokens = usage_completion_tokens
            return

        token_id_chunks = [
            chunk for chunk in content_chunks
            if isinstance(chunk.get("token_ids"), list)
        ]

        if len(token_id_chunks) == len(content_chunks):
            for chunk in content_chunks:
                token_count = len(chunk["token_ids"])
                result.total_tokens += token_count
                self._append_observed_token_timestamps(result, chunk["timestamp"], token_count)
            return

        chunk_times = [chunk["timestamp"] for chunk in content_chunks]

        if usage_completion_tokens is not None:
            _warn_once("  No complete token_ids in response, using stream usage token count")
            result.total_tokens = usage_completion_tokens
            result.token_timestamps = self._interpolate_token_timestamps(chunk_times, usage_completion_tokens)
            return

        if tokenizer is not None:
            _warn_once("  No token_ids or usage in response, using local tokenization")
            full_content = "".join(chunk["text"] for chunk in content_chunks)
            token_count = len(tokenizer.encode(full_content, add_special_tokens=False))
            result.total_tokens = token_count
            result.token_timestamps = self._interpolate_token_timestamps(chunk_times, token_count)
            return

        _warn_once("  No token_ids, usage, or tokenizer, assuming 1 token per chunk")
        result.total_tokens = len(content_chunks)
        result.token_timestamps = chunk_times

    async def measure_latency(
        self,
        session: aiohttp.ClientSession,
        mode: str = "api",
        warmup_runs: int = 1,
        measured_runs: int = 3,
    ) -> float:
        if mode == "none":
            print("Skipping latency measurement (assuming 0 ms).")
            return 0

        warmup_runs = max(0, warmup_runs)
        measured_runs = max(0, measured_runs)
        if mode == "generation" and warmup_runs > 0:
            print(
                f"Measuring latency using mode: {mode} "
                f"({warmup_runs} warmup + {measured_runs} measured probes)..."
            )
        else:
            print(f"Measuring latency using mode: {mode}...")
        latencies = []
        total_runs = measured_runs + (warmup_runs if mode == "generation" else 0)
        
        for probe_idx in range(total_runs):
            is_warmup = mode == "generation" and probe_idx < warmup_runs
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
                            elapsed = time.perf_counter() - start
                            if not is_warmup:
                                latencies.append(elapsed)
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
                payload_sys_probe = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": warmup_text},
                        {"role": "user", "content": CONTEXT_LOAD_USER_MESSAGE}
                    ],
                    "max_tokens": 1
                }
                async with session.post(f"{self.base_url}/chat/completions", json=payload_sys_probe, headers=self.headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Warmup failed: HTTP {response.status}: {error_text}")
                        raise SystemExit(1)
                    response_json = await response.json()
                    if 'usage' in response_json:
                        prompt_tokens = response_json['usage']['prompt_tokens']
                        local_tokens = len(tokenizer.encode(warmup_text, add_special_tokens=False))
                        probe_tokens = len(tokenizer.encode(CONTEXT_LOAD_USER_MESSAGE, add_special_tokens=False))
                        delta_context = prompt_tokens - local_tokens - probe_tokens
                        print(f"Warmup (System+Probe) complete. Delta: {delta_context} tokens (Server: {prompt_tokens}, Local context: {local_tokens}, Probe: {probe_tokens})")
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
            no_cache: bool,
            tokenizer=None,
            progress=None,
            request_id: Optional[int] = None,
            messages: Optional[List[Dict[str, Any]]] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
        ) -> RequestResult:

        if messages is None:
            messages = []
            if context_text:
                messages.append({"role": "system", "content": context_text})
            messages.append({"role": "user", "content": self._non_empty_user_content(prompt_text)})
        
        result = RequestResult()
        
        try:
            payload = self._build_generation_payload(messages, max_tokens, no_cache, tools)
            
            result.start_ts = time.perf_counter()

            async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=self.headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    result.error = f"HTTP {response.status}: {error_text}"
                    print(result.error)
                    self._emit_request_end(progress, request_id, result)
                    return result

                decoder = codecs.getincrementaldecoder("utf-8")(errors='replace')
                buffer = ""
                content_chunks: List[Dict[str, Any]] = []
                usage_completion_tokens: Optional[int] = None
                
                async for chunk_bytes in response.content.iter_any():
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

                                usage = chunk.get('usage')
                                if isinstance(usage, dict):
                                    prompt_tokens = usage.get('prompt_tokens')
                                    if isinstance(prompt_tokens, int):
                                        result.prompt_tokens = prompt_tokens
                                    completion_tokens = usage.get('completion_tokens')
                                    if isinstance(completion_tokens, int) and completion_tokens >= 0:
                                        usage_completion_tokens = completion_tokens

                                timings = chunk.get('timings')
                                if not isinstance(timings, dict) and isinstance(usage, dict):
                                    timings = usage.get('timings')
                                if isinstance(timings, dict):
                                    decode_ms = timings.get('decode_ms')
                                    if not isinstance(decode_ms, (int, float)):
                                        decode_ms = timings.get('predicted_ms')
                                    if isinstance(decode_ms, (int, float)) and decode_ms > 0:
                                        result.server_decode_seconds = decode_ms / 1000.0
                                
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    if result.first_response_ts is None:
                                        result.first_response_ts = chunk_time
                                        if progress is not None and request_id is not None:
                                            try:
                                                progress.request_first_response(
                                                    request_id=request_id,
                                                    ttfr_s=chunk_time - result.start_ts,
                                                )
                                            except Exception:
                                                pass

                                    delta = chunk['choices'][0].get('delta', {})
                                    content = delta.get('content')
                                    reasoning_content = delta.get('reasoning_content')
                                    reasoning = delta.get('reasoning')
                                    tool_call_text = self._tool_call_text(delta.get('tool_calls'))
                                    token_ids = chunk['choices'][0].get('token_ids')

                                    if (
                                        content
                                        or reasoning_content
                                        or reasoning
                                        or tool_call_text
                                        or (isinstance(token_ids, list) and token_ids)
                                    ):
                                        if result.first_token_ts is None:
                                            result.first_token_ts = chunk_time
                                            if progress is not None and request_id is not None:
                                                try:
                                                    progress.request_first_token(
                                                        request_id=request_id,
                                                        ttft_s=chunk_time - result.start_ts,
                                                    )
                                                except Exception:
                                                    pass

                                        text = content or reasoning_content or reasoning or tool_call_text
                                        content_chunks.append({
                                            "text": text,
                                            "timestamp": chunk_time,
                                            "token_ids": token_ids,
                                        })

                                        if progress is not None and request_id is not None:
                                            # Best-effort per-chunk count for the live stream.
                                            # The authoritative total is reconciled in
                                            # _finalize_stream_tokens and reported by request_end.
                                            has_token_ids = isinstance(token_ids, list)
                                            chunk_count = len(token_ids) if has_token_ids else 1
                                            try:
                                                progress.tokens(
                                                    request_id=request_id,
                                                    count=chunk_count,
                                                    snippet=text or "",
                                                    estimated=not has_token_ids,
                                                )
                                            except Exception:
                                                pass
                            except json.JSONDecodeError:
                                continue

                result.end_ts = time.perf_counter()
                self._finalize_stream_tokens(result, content_chunks, usage_completion_tokens, tokenizer)
                self._apply_server_decode_timestamps(result)

        except Exception as e:
            print(f"Error during run: {e}")
            result.error = str(e)

        self._emit_request_end(progress, request_id, result)
        return result

    @staticmethod
    def _emit_request_end(progress, request_id: Optional[int], result: "RequestResult") -> None:
        """Emit the request_end progress event for a finished request."""
        if progress is None or request_id is None:
            return
        decode_seconds = 0.0
        if result.first_token_ts is not None and result.end_ts:
            decode_seconds = max(0.0, result.end_ts - result.first_token_ts)
        try:
            progress.request_end(
                request_id=request_id,
                total_tokens=result.total_tokens,
                prompt_tokens=result.prompt_tokens,
                decode_seconds=decode_seconds,
                error=result.error or "",
            )
        except Exception:
            pass
