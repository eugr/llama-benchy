"""Regression test for speculative-decoding token accounting in LLMClient.run_generation.

Without the fix, llama-benchy counts one token per SSE streaming chunk that carries
`delta.content`. That is correct for plain autoregressive decoding (one token per
step → one chunk per token) but WRONG for speculative decoding (DFlash/MTP/Eagle/
Medusa/ngram), where a single step can accept multiple tokens and they are emitted
together in one chunk. The fix prefers the authoritative server-side
`usage.completion_tokens` over the chunk tally.

These tests verify both cases without needing a live vLLM server:

  1. Plain AR (1 token per chunk) -> total_tokens equals chunk count == completion_tokens
  2. Spec decode (K tokens per chunk) -> total_tokens equals completion_tokens (NOT chunk count)
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Iterable, List, Optional

import pytest

from llama_benchy.client import LLMClient


# -----------------------------------------------------------------------------
# Fake aiohttp plumbing: yield bytes over `response.content` like the real thing.
# -----------------------------------------------------------------------------

class _FakeContent:
    """Minimal async iterator over pre-recorded SSE bytes."""

    def __init__(self, chunks: Iterable[bytes]):
        self._chunks = list(chunks)

    def __aiter__(self):
        self._i = 0
        return self

    async def __anext__(self):
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._i]
        self._i += 1
        return chunk


class _FakeResponse:
    def __init__(self, sse_lines: Iterable[str]):
        # Each SSE line is one 'data: {...}\n\n' block
        self.status = 200
        body = "".join(line if line.endswith("\n\n") else line + "\n\n" for line in sse_lines)
        self.content = _FakeContent([body.encode("utf-8")])

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return ""


class _FakeSession:
    """Stands in for aiohttp.ClientSession — records the last request, returns canned SSE."""

    def __init__(self, sse_lines: Iterable[str]):
        self._sse_lines = list(sse_lines)
        self.last_json: Optional[dict] = None

    def post(self, url, *, json=None, headers=None):  # noqa: A002 -- match aiohttp sig
        self.last_json = json
        return _FakeResponse(self._sse_lines)


# -----------------------------------------------------------------------------
# SSE stream builders mimicking vLLM's Chat Completions streaming format
# -----------------------------------------------------------------------------

def _ar_stream(n_tokens: int) -> List[str]:
    """One SSE chunk per token (plain autoregressive decoding)."""
    lines: List[str] = []
    for i in range(n_tokens):
        chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": f"tok{i} "}, "finish_reason": None}],
        }
        lines.append(f"data: {json.dumps(chunk)}")
    # Final stop chunk (no content)
    lines.append(
        "data: " + json.dumps({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })
    )
    # Usage chunk (include_usage=True path)
    lines.append(
        "data: " + json.dumps({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": n_tokens,
                "total_tokens": 7 + n_tokens,
            },
        })
    )
    lines.append("data: [DONE]")
    return lines


def _spec_decode_stream(n_chunks: int, tokens_per_chunk: int) -> List[str]:
    """K accepted tokens emitted in each SSE chunk (speculative decoding)."""
    lines: List[str] = []
    total = n_chunks * tokens_per_chunk
    for ci in range(n_chunks):
        chunk_tokens = " ".join(f"tok{ci*tokens_per_chunk + j}" for j in range(tokens_per_chunk))
        chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": chunk_tokens + " "}, "finish_reason": None}],
        }
        lines.append(f"data: {json.dumps(chunk)}")
    # Final stop chunk
    lines.append(
        "data: " + json.dumps({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })
    )
    # Authoritative usage chunk — this is what the fix relies on
    lines.append(
        "data: " + json.dumps({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": total,
                "total_tokens": 7 + total,
            },
        })
    )
    lines.append("data: [DONE]")
    return lines


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def _run(sse_lines):
    client = LLMClient(base_url="http://fake", api_key="x", model_name="test-model")
    session = _FakeSession(sse_lines)
    return asyncio.run(
        client.run_generation(session, context_text="", prompt_text="hello", max_tokens=128, no_cache=False)
    )


def test_autoregressive_accounting_unchanged():
    """Plain AR: 1 chunk = 1 token, chunk tally == completion_tokens == 20."""
    res = _run(_ar_stream(n_tokens=20))
    assert res.error is None, f"unexpected error: {res.error}"
    assert res.prompt_tokens == 7
    assert res.total_tokens == 20, f"expected 20, got {res.total_tokens}"
    # Timestamps recorded once per content-bearing chunk -> 20
    assert len(res.token_timestamps) == 20


@pytest.mark.parametrize("tokens_per_chunk", [2, 4, 8, 16])
def test_speculative_decoding_uses_completion_tokens(tokens_per_chunk):
    """Spec decode: K tokens per chunk, total_tokens must come from usage.completion_tokens.

    With the old chunk-count logic, total_tokens would be `n_chunks` and throughput
    would be under-reported by a factor of `tokens_per_chunk`.
    """
    n_chunks = 10
    expected_total = n_chunks * tokens_per_chunk
    res = _run(_spec_decode_stream(n_chunks=n_chunks, tokens_per_chunk=tokens_per_chunk))

    assert res.error is None, f"unexpected error: {res.error}"
    assert res.prompt_tokens == 7
    # The fix: total_tokens reflects authoritative server-side completion_tokens,
    # not the chunk count (which would be n_chunks).
    assert res.total_tokens == expected_total, (
        f"spec-decode undercount: expected {expected_total} tokens "
        f"(n_chunks={n_chunks} * tokens_per_chunk={tokens_per_chunk}), "
        f"got {res.total_tokens}. This indicates the chunk-count regression has returned."
    )
    # Chunk-level timestamps are still recorded per chunk (used for peak throughput,
    # which is a separate metric representing streaming burst behavior).
    assert len(res.token_timestamps) == n_chunks


def test_usage_without_completion_tokens_falls_back_to_chunk_count():
    """If the server omits completion_tokens, we fall back to chunk-count accounting."""
    lines = _ar_stream(5)
    # Strip completion_tokens from the usage chunk to simulate a legacy server
    usage_idx = [i for i, l in enumerate(lines) if '"usage"' in l][0]
    payload = json.loads(lines[usage_idx][len("data: "):])
    payload["usage"].pop("completion_tokens", None)
    lines[usage_idx] = "data: " + json.dumps(payload)

    res = _run(lines)
    assert res.error is None
    # No completion_tokens -> chunk tally wins
    assert res.total_tokens == 5
