import json

import pytest

import llama_benchy.client as client_module
from llama_benchy.client import CONTEXT_LOAD_USER_MESSAGE, LLMClient


_MISSING = object()


class _FakeContent:
    def __init__(self, events):
        self._events = events

    def __aiter__(self):
        return self.iter_any()

    async def iter_any(self):
        for event in self._events:
            yield event


class _FakeResponse:
    def __init__(self, events, json_body=None, status=200):
        self.status = status
        self.content = _FakeContent(events)
        self._json_body = json_body or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def text(self):
        return ""

    async def json(self):
        return self._json_body


class _FakeSession:
    def __init__(self, events, json_bodies=None):
        self._events = events
        self._json_bodies = list(json_bodies or [])
        self.requests = []

    def post(self, *args, **kwargs):
        self.requests.append({"args": args, "kwargs": kwargs})
        json_body = self._json_bodies.pop(0) if self._json_bodies else None
        return _FakeResponse(self._events, json_body=json_body)


class _ChunkBoundaryTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        self.calls.append((text, add_special_tokens))
        tokens_by_text = {
            "Hel": [1],
            "lo": [2],
            " world": [3],
            "Hello world": [15496, 995],
        }
        return tokens_by_text[text]


class _WarmupTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text == "Warmup " * 10:
            return list(range(10))
        if text == CONTEXT_LOAD_USER_MESSAGE:
            return [99]
        raise AssertionError(f"Unexpected text to tokenize: {text!r}")


class _RecordingProgress:
    def __init__(self):
        self.events = []

    def request_first_response(self, **fields):
        self.events.append(("request_first_response", fields))

    def request_first_token(self, **fields):
        self.events.append(("request_first_token", fields))

    def tokens(self, **fields):
        self.events.append(("tokens", fields))

    def request_end(self, **fields):
        self.events.append(("request_end", fields))


def _sse_event(payload):
    if payload == "[DONE]":
        return b"data: [DONE]\n\n"
    return f"data: {json.dumps(payload)}\n\n".encode()


def _content_event(content, token_ids=_MISSING):
    choice = {
        "index": 0,
        "delta": {"content": content},
        "finish_reason": None,
    }
    if token_ids is not _MISSING:
        choice["token_ids"] = token_ids
    return {"choices": [choice]}


def _tool_call_event(arguments, token_ids=_MISSING):
    choice = {
        "index": 0,
        "delta": {
            "tool_calls": [{
                "index": 0,
                "function": {"arguments": arguments},
            }],
        },
        "finish_reason": None,
    }
    if token_ids is not _MISSING:
        choice["token_ids"] = token_ids
    return {"choices": [choice]}


def _usage_event(completion_tokens):
    return {
        "choices": [],
        "usage": {
            "prompt_tokens": 16,
            "completion_tokens": completion_tokens,
            "total_tokens": 16 + completion_tokens,
        },
    }


async def _run_stream(events, tokenizer=None, progress=None, request_id=None):
    client = LLMClient("http://example.test/v1", "EMPTY", "model")
    session = _FakeSession([_sse_event(event) for event in events])
    return await client.run_generation(
        session,
        context_text="",
        prompt_text="hello",
        max_tokens=8,
        no_cache=False,
        tokenizer=tokenizer,
        progress=progress,
        request_id=request_id,
    )


def test_generation_payload_defaults():
    client = LLMClient("http://example.test/v1", "EMPTY", "model")
    messages = [{"role": "user", "content": "hello"}]

    payload = client._build_generation_payload(messages, max_tokens=128, no_cache=False)

    assert payload["model"] == "model"
    assert payload["messages"] == messages
    assert payload["max_tokens"] == 128
    assert payload["stream"] is True
    assert payload["return_token_ids"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert "min_tokens" not in payload
    assert "ignore_eos" not in payload


def test_generation_payload_merges_extra_body():
    client = LLMClient(
        "http://example.test/v1",
        "EMPTY",
        "model",
        extra_body={"temperature": 0, "ignore_eos": True},
    )

    payload = client._build_generation_payload([], max_tokens=128, no_cache=True)

    assert payload["cache_prompt"] is False
    assert payload["temperature"] == 0
    assert payload["ignore_eos"] is True


def test_generation_payload_preserves_tools():
    client = LLMClient("http://example.test/v1", "EMPTY", "model")
    tools = [{"type": "function", "function": {"name": "bash"}}]

    payload = client._build_generation_payload(
        [], max_tokens=128, no_cache=False, tools=tools
    )

    assert payload["tools"] == tools


def test_exact_tg_forces_min_tokens_and_ignore_eos():
    client = LLMClient(
        "http://example.test/v1",
        "EMPTY",
        "model",
        extra_body={"max_tokens": 64, "min_tokens": 16, "ignore_eos": False},
        exact_tg=True,
    )

    payload = client._build_generation_payload([], max_tokens=128, no_cache=False)

    assert payload["max_tokens"] == 128
    assert payload["min_tokens"] == 128
    assert payload["ignore_eos"] is True


@pytest.mark.asyncio
async def test_run_generation_replaces_empty_user_message_with_context_probe():
    client = LLMClient("http://example.test/v1", "EMPTY", "model")
    session = _FakeSession([_sse_event("[DONE]")])

    await client.run_generation(
        session,
        context_text="cached context",
        prompt_text="",
        max_tokens=1,
        no_cache=False,
    )

    messages = session.requests[0]["kwargs"]["json"]["messages"]
    assert messages == [
        {"role": "system", "content": "cached context"},
        {"role": "user", "content": CONTEXT_LOAD_USER_MESSAGE},
    ]
    assert messages[-1]["content"].strip()


@pytest.mark.asyncio
async def test_run_generation_preserves_explicit_conversation_messages():
    client = LLMClient("http://example.test/v1", "EMPTY", "model")
    session = _FakeSession([_sse_event("[DONE]")])
    messages = [
        {"role": "system", "content": "bootstrap"},
        {"role": "user", "content": "inspect the repository"},
        {"role": "assistant", "content": "I inspected it."},
        {"role": "user", "content": "now fix the tests"},
    ]

    await client.run_generation(
        session,
        context_text="ignored",
        prompt_text="ignored",
        max_tokens=8,
        no_cache=False,
        messages=messages,
    )

    assert session.requests[0]["kwargs"]["json"]["messages"] == messages


@pytest.mark.asyncio
async def test_warmup_uses_context_probe_and_excludes_it_from_delta():
    client = LLMClient("http://example.test/v1", "EMPTY", "model")
    session = _FakeSession(
        [],
        json_bodies=[
            {"usage": {"prompt_tokens": 22}},
            {"usage": {"prompt_tokens": 24}},
        ],
    )

    delta_user, delta_context = await client.warmup(session, _WarmupTokenizer())

    assert delta_user == 12
    assert delta_context == 13
    messages = session.requests[1]["kwargs"]["json"]["messages"]
    assert messages == [
        {"role": "system", "content": "Warmup " * 10},
        {"role": "user", "content": CONTEXT_LOAD_USER_MESSAGE},
    ]
    assert messages[-1]["content"].strip()


@pytest.mark.asyncio
async def test_generation_latency_discards_warmup_probes(monkeypatch):
    client = LLMClient("http://example.test/v1", "EMPTY", "model")
    session = _FakeSession([b"data: {}\n\n", b"data: [DONE]\n\n"])
    ticks = iter([
        0.0, 100.0,  # discarded warmup probe
        10.0, 11.0,
        20.0, 22.0,
        30.0, 33.0,
    ])
    monkeypatch.setattr(client_module.time, "perf_counter", lambda: next(ticks))

    latency = await client.measure_latency(
        session,
        mode="generation",
        warmup_runs=1,
        measured_runs=3,
    )

    assert latency == pytest.approx(2.0)
    assert len(session.requests) == 4


@pytest.mark.asyncio
async def test_streaming_uses_final_usage_when_token_ids_are_missing():
    tokenizer = _ChunkBoundaryTokenizer()

    result = await _run_stream([
        _content_event("Hel"),
        _content_event("lo"),
        _content_event(" world"),
        _usage_event(2),
        "[DONE]",
    ], tokenizer=tokenizer)

    assert result.prompt_tokens == 16
    assert result.total_tokens == 2
    assert len(result.token_timestamps) == 2
    assert tokenizer.calls == []


@pytest.mark.asyncio
async def test_streaming_times_tool_call_argument_deltas():
    result = await _run_stream([
        _tool_call_event('{"command":'),
        _tool_call_event('"pytest"}'),
        _usage_event(12),
        "[DONE]",
    ])

    assert result.prompt_tokens == 16
    assert result.total_tokens == 12
    assert result.first_token_ts is not None
    assert len(result.token_timestamps) == 12
    assert result.token_timestamps[0] <= result.token_timestamps[-1]


@pytest.mark.asyncio
async def test_streaming_prefers_server_reported_decode_duration():
    usage = _usage_event(10)
    usage["timings"] = {"decode_ms": 2000.0}

    result = await _run_stream([
        _content_event("buffered response"),
        usage,
        "[DONE]",
    ])

    assert result.server_decode_seconds == pytest.approx(2.0)
    assert result.total_tokens == 10
    assert len(result.token_timestamps) == 10
    assert result.token_timestamps[-1] - result.token_timestamps[0] == pytest.approx(1.8)


@pytest.mark.asyncio
async def test_streaming_local_fallback_tokenizes_concatenated_content_once():
    tokenizer = _ChunkBoundaryTokenizer()

    result = await _run_stream([
        _content_event("Hel"),
        _content_event("lo"),
        _content_event(" world"),
        "[DONE]",
    ], tokenizer=tokenizer)

    assert result.total_tokens == 2
    assert len(result.token_timestamps) == 2
    assert tokenizer.calls == [("Hello world", False)]


@pytest.mark.asyncio
async def test_streaming_token_ids_take_precedence_over_final_usage():
    result = await _run_stream([
        _content_event("Hel", token_ids=[1]),
        _content_event("lo", token_ids=[2, 3]),
        _usage_event(99),
        "[DONE]",
    ])

    assert result.prompt_tokens == 16
    assert result.total_tokens == 3
    assert len(result.token_timestamps) == 3


@pytest.mark.asyncio
async def test_progress_marks_missing_token_ids_as_estimated_and_ends_with_authoritative_total():
    progress = _RecordingProgress()
    tokenizer = _ChunkBoundaryTokenizer()

    result = await _run_stream([
        _content_event("Hel"),
        _content_event("lo"),
        _content_event(" world"),
        _usage_event(2),
        "[DONE]",
    ], tokenizer=tokenizer, progress=progress, request_id=7)

    token_events = [fields for event, fields in progress.events if event == "tokens"]
    assert token_events == [
        {"request_id": 7, "count": 1, "snippet": "Hel", "estimated": True},
        {"request_id": 7, "count": 1, "snippet": "lo", "estimated": True},
        {"request_id": 7, "count": 1, "snippet": " world", "estimated": True},
    ]

    end_event = progress.events[-1]
    assert end_event[0] == "request_end"
    assert end_event[1]["request_id"] == 7
    assert end_event[1]["total_tokens"] == 2
    assert result.total_tokens == 2


@pytest.mark.asyncio
async def test_progress_token_ids_are_exact_not_estimated():
    progress = _RecordingProgress()

    await _run_stream([
        _content_event("Hel", token_ids=[1]),
        _content_event("lo", token_ids=[2, 3]),
        _usage_event(99),
        "[DONE]",
    ], progress=progress, request_id=11)

    token_events = [fields for event, fields in progress.events if event == "tokens"]
    assert token_events == [
        {"request_id": 11, "count": 1, "snippet": "Hel", "estimated": False},
        {"request_id": 11, "count": 2, "snippet": "lo", "estimated": False},
    ]
