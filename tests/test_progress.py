import io
import json

import pytest

from llama_benchy.client import RequestResult
from llama_benchy.config import BenchmarkConfig
from llama_benchy.progress import ConsoleProgressBar, ProgressEmitter, SCHEMA_VERSION
from llama_benchy.runner import BenchmarkRunner


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_console_progress_bar_formats_completion_and_eta():
    stream = io.StringIO()
    bar = ConsoleProgressBar(total_steps=3, enabled=True, stream=stream)

    bar.render(1, description="Context Load", elapsed=2.0)

    output = stream.getvalue()
    assert "1/3" in output
    assert "Context Load" in output
    assert "ETA" in output


def test_progress_emitter_writes_estimated_tokens_and_terminal_status(tmp_path):
    progress_path = tmp_path / "progress.jsonl"
    emitter = ProgressEmitter(str(progress_path), llama_benchy_version="test-version")

    emitter.tokens(request_id=3, count=1, snippet="hello", estimated=True)
    emitter.tokens(request_id=3, count=2, snippet=" world")
    emitter.bench_complete(status="interrupted")
    emitter.close()

    events = _read_jsonl(progress_path)

    assert events[0]["schema"] == SCHEMA_VERSION
    assert events[0]["type"] == "header"
    assert events[0]["llama_benchy_version"] == "test-version"

    assert events[1]["type"] == "tokens"
    assert events[1]["estimated"] is True

    assert events[2]["type"] == "tokens"
    assert "estimated" not in events[2]

    assert events[3]["type"] == "bench_complete"
    assert events[3]["status"] == "interrupted"


class _FakeCorpus:
    def get_tokenizer(self):
        return None


class _FakePromptGenerator:
    corpus = _FakeCorpus()

    def generate_batch(self, concurrency, pp, depth, no_cache):
        return [("", "hello") for _ in range(concurrency)]


class _FakeBenchmarkClient:
    def __init__(self):
        self.latency_calls = []
        self.generation_progress_flags = []

    async def warmup(self, session, tokenizer=None):
        return 0, 0

    async def run_coherence_test(self, session):
        return True

    async def measure_latency(self, session, mode="api", warmup_runs=1, measured_runs=3):
        self.latency_calls.append(
            {
                "mode": mode,
                "warmup_runs": warmup_runs,
                "measured_runs": measured_runs,
            }
        )
        return 0.001

    async def run_generation(
        self,
        session,
        context_text,
        prompt_text,
        max_tokens,
        no_cache,
        tokenizer=None,
        progress=None,
        request_id=None,
    ):
        self.generation_progress_flags.append(progress is not None)
        result = RequestResult(
            start_ts=1.0,
            first_response_ts=1.1,
            first_token_ts=1.1,
            end_ts=1.3,
            prompt_tokens=4,
            total_tokens=2,
            token_timestamps=[1.1, 1.2],
        )

        if progress is not None and request_id is not None:
            progress.request_first_response(request_id=request_id, ttfr_s=0.1)
            progress.request_first_token(request_id=request_id, ttft_s=0.1)
            progress.tokens(request_id=request_id, count=1, snippet="a")
            progress.tokens(request_id=request_id, count=1, snippet="b")
            progress.request_end(
                request_id=request_id,
                total_tokens=result.total_tokens,
                prompt_tokens=result.prompt_tokens,
                decode_seconds=0.2,
            )

        return result


@pytest.mark.asyncio
async def test_warmup_runs_preserve_progress_streaming_contract(tmp_path):
    progress_path = tmp_path / "progress.jsonl"
    result_path = tmp_path / "results.json"
    progress = ProgressEmitter(str(progress_path), llama_benchy_version="test-version")
    client = _FakeBenchmarkClient()
    config = BenchmarkConfig(
        base_url="http://example.test/v1",
        api_key="EMPTY",
        model="model",
        served_model_name="model",
        tokenizer=None,
        pp_counts=[4],
        tg_counts=[2],
        exact_tg=False,
        depths=[0],
        num_runs=2,
        warmup_runs=2,
        no_cache=False,
        latency_mode="generation",
        no_warmup=False,
        skip_coherence=True,
        adapt_prompt=False,
        enable_prefix_caching=False,
        book_url="",
        post_run_cmd=None,
        concurrency_levels=[1],
        save_result=str(result_path),
        result_format="json",
        save_total_throughput_timeseries=False,
        save_all_throughput_timeseries=False,
        exit_on_first_fail=False,
        no_results_on_fail=False,
        extra_body={},
        emit_progress=str(progress_path),
    )

    try:
        runner = BenchmarkRunner(config, client, _FakePromptGenerator(), progress=progress)
        await runner.run_suite()
        progress.bench_complete(status="ok")
    finally:
        progress.close()

    events = _read_jsonl(progress_path)
    event_types = [event["type"] for event in events]

    assert client.latency_calls == [
        {"mode": "generation", "warmup_runs": 2, "measured_runs": 3}
    ]
    assert client.generation_progress_flags == [False, False, True, True]
    assert event_types[0] == "header"
    assert event_types[1] == "latency_measured"
    assert event_types[-1] == "bench_complete"

    request_starts = [event for event in events if event["type"] == "request_start"]
    assert [event["request_id"] for event in request_starts] == [0, 1]
    assert [event["run_index"] for event in request_starts] == [0, 1]

    request_ids = {event["request_id"] for event in request_starts}
    for request_id in request_ids:
        per_request_events = [
            event["type"] for event in events if event.get("request_id") == request_id
        ]
        assert per_request_events == [
            "request_start",
            "request_first_response",
            "request_first_token",
            "tokens",
            "tokens",
            "request_end",
        ]
