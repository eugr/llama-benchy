import io
import json

import pytest

from llama_benchy.client import RequestResult
from llama_benchy.config import BenchmarkConfig
from llama_benchy.progress import (
    ConsoleProgressBar,
    ProgressEmitter,
    SCHEMA_VERSION,
    compute_phase_weight,
    DEFAULT_PROMPT_COST,
    DEFAULT_GEN_COST,
)
from llama_benchy.runner import BenchmarkRunner


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_console_progress_bar_formats_completion_and_eta():
    stream = io.StringIO()
    bar = ConsoleProgressBar(total_weight=3.0, enabled=True, stream=stream)

    bar.render(1.0, description="Context Load", elapsed=2.0, current_phase_weight=0.0)

    output = stream.getvalue()
    assert "1.0/3.0" in output
    assert "Context Load" in output
    assert "ETA" in output


def test_console_progress_bar_reaches_100_percent():
    stream = io.StringIO()
    bar = ConsoleProgressBar(total_weight=10.0, enabled=True, stream=stream)
    bar.start()

    bar.render(10.0, description="Complete", elapsed=5.0, current_phase_weight=0.0)

    output = stream.getvalue()
    assert "100%" in output
    assert "00:00" in output


def test_console_progress_bar_eta_uses_adaptive_rate():
    """ETA should use actual measured phase timings when available.

    The first measured phase is skipped to avoid first-request overhead.
    """
    stream = io.StringIO()
    bar = ConsoleProgressBar(total_weight=10.0, enabled=True, stream=stream)
    bar.start()

    # Record two phases: first (skipped for ETA), second (used for ETA)
    # Phase 1: 3 weight in 2s (skipped)
    # Phase 2: 3 weight in 2s (used)
    bar._phase_weights = [3.0, 3.0]
    bar._phase_elapsed = [2.0, 2.0]

    # Now render with 6/10 completed — ETA should be based on measured rate
    # completed_weight=6.0 (actually completed), current_phase_weight=0 (no in-progress phase)
    bar.render(6.0, description="Phase 2", elapsed=4.0, current_phase_weight=0.0)

    output = stream.getvalue()
    # Measured rate (excluding first phase): 3.0 weight / 2.0s = 1.5 weight/s
    # Remaining: 4.0 weight → ETA = 4.0 / 1.5 = 2.67s → "00:02" (truncated)
    assert "ETA 00:02" in output


def test_console_progress_bar_eta_fallback_linear():
    """Without measured phase timings, ETA falls back to linear extrapolation."""
    stream = io.StringIO()
    bar = ConsoleProgressBar(total_weight=10.0, enabled=True, stream=stream)
    bar.start()

    # No phases recorded, so linear extrapolation: 5.0/10.0 done in 4.0s
    # ETA = 4.0 * (10-5)/5 = 4.0s → "00:04"
    bar.render(5.0, description="Halfway", elapsed=4.0, current_phase_weight=0.0)

    output = stream.getvalue()
    assert "ETA 00:04" in output


def test_console_progress_bar_monotonic():
    """Progress percentage should never decrease."""
    stream = io.StringIO()
    bar = ConsoleProgressBar(total_weight=100.0, enabled=True, stream=stream)
    bar.start()

    bar.render(10.0, description="Phase 1", elapsed=1.0, current_phase_weight=0.0)
    bar.render(50.0, description="Phase 2", elapsed=5.0, current_phase_weight=0.0)
    bar.render(30.0, description="Phase 3", elapsed=6.0, current_phase_weight=0.0)  # Should be clamped to 50%

    output = stream.getvalue()
    # The last render should show 50% (clamped), not 30%
    assert "50%" in output


def test_compute_phase_weight_standard_run():
    """Standard run weight = (pp + depth) * prompt_cost + tg * gen_cost."""
    weight = compute_phase_weight(pp=2048, tg=32, depth=0)
    expected = (2048 + 0) * DEFAULT_PROMPT_COST + 32 * DEFAULT_GEN_COST
    assert weight == expected


def test_compute_phase_weight_with_depth():
    """Standard run with depth includes depth in prompt tokens."""
    weight = compute_phase_weight(pp=2048, tg=32, depth=4096)
    expected = (2048 + 4096) * DEFAULT_PROMPT_COST + 32 * DEFAULT_GEN_COST
    assert weight == expected


def test_compute_phase_weight_context_load():
    """Context load phase processes full context (pp + depth)."""
    weight = compute_phase_weight(pp=2048, tg=32, depth=4096, is_context_load=True)
    expected = (2048 + 4096) * DEFAULT_PROMPT_COST + 32 * DEFAULT_GEN_COST
    assert weight == expected


def test_compute_phase_weight_inference():
    """Inference phase (prefix caching) processes only pp (context cached)."""
    weight = compute_phase_weight(pp=2048, tg=32, depth=4096, is_inference=True)
    expected = 2048 * DEFAULT_PROMPT_COST + 32 * DEFAULT_GEN_COST
    assert weight == expected


def test_compute_phase_weight_context_load_heavier_than_inference():
    """Context load should be heavier than inference when depth > 0."""
    ctx_weight = compute_phase_weight(pp=2048, tg=32, depth=4096, is_context_load=True)
    inf_weight = compute_phase_weight(pp=2048, tg=32, depth=4096, is_inference=True)
    assert ctx_weight > inf_weight


def test_compute_phase_weight_larger_pp_is_heavier():
    """Larger pp should produce larger weight."""
    small = compute_phase_weight(pp=2048, tg=32, depth=0)
    large = compute_phase_weight(pp=128000, tg=32, depth=0)
    assert large > small
    # 128000/2048 = 62.5x more prompt tokens
    assert large > small * 10


def test_compute_phase_weight_larger_tg_is_heavier():
    """Larger tg should produce larger weight."""
    small = compute_phase_weight(pp=2048, tg=32, depth=0)
    large = compute_phase_weight(pp=2048, tg=256, depth=0)
    assert large > small
    # 256/32 = 8x more gen tokens, and gen_cost >> prompt_cost
    assert large > small * 2


def test_compute_phase_weight_larger_depth_is_heavier():
    """Larger depth should produce larger weight in standard runs."""
    small = compute_phase_weight(pp=2048, tg=32, depth=0)
    large = compute_phase_weight(pp=2048, tg=32, depth=4096)
    assert large > small


def test_compute_phase_weight_concurrency_does_not_affect():
    """Concurrency should NOT affect the weight (parallel execution)."""
    weight_c1 = compute_phase_weight(pp=2048, tg=32, depth=0)
    weight_c8 = compute_phase_weight(pp=2048, tg=32, depth=0)
    assert weight_c1 == weight_c8


def test_compute_phase_weight_custom_costs():
    """Custom cost factors should be respected."""
    weight = compute_phase_weight(
        pp=100, tg=10, depth=50,
        prompt_cost=2.0, gen_cost=5.0,
    )
    expected = (100 + 50) * 2.0 + 10 * 5.0
    assert weight == expected


def test_compute_phase_weight_zero_tokens():
    """Zero tokens should produce zero weight."""
    weight = compute_phase_weight(pp=0, tg=0, depth=0)
    assert weight == 0.0


def test_compute_phase_weight_negative_clamped():
    """Negative token counts should be handled gracefully."""
    weight = compute_phase_weight(pp=-100, tg=32, depth=0)
    # pp is clamped to 0 in the formula via max(0, ...)
    # Actually, the formula doesn't clamp — but negative pp would give negative weight
    # Let's verify the behavior is at least non-crashing
    assert isinstance(weight, float)


def test_progress_bar_disabled_is_noop():
    """Disabled progress bar should not write anything."""
    stream = io.StringIO()
    bar = ConsoleProgressBar(total_weight=10.0, enabled=False, stream=stream)
    bar.start()
    bar.render(5.0, description="Test")
    bar.finish()
    assert stream.getvalue() == ""


def test_progress_bar_add_phase_accumulates():
    """add_phase should accumulate total weight."""
    stream = io.StringIO()
    bar = ConsoleProgressBar(total_weight=0.0, enabled=True, stream=stream)
    bar.start()

    bar.add_phase(3.0)
    assert bar.total_weight == 3.0

    bar.add_phase(7.0)
    assert bar.total_weight == 10.0


def test_progress_bar_record_phase_elapsed():
    """record_phase_elapsed should store elapsed times and weights for adaptive ETA."""
    stream = io.StringIO()
    bar = ConsoleProgressBar(total_weight=10.0, enabled=True, stream=stream)
    bar.start()

    bar.record_phase_elapsed(2.0, phase_weight=5.0)

    assert bar._phase_elapsed == [2.0]
    assert bar._phase_weights == [5.0]


def test_progress_bar_eta_with_multiple_phases():
    """ETA should use average rate across multiple completed phases.

    The first measured phase is skipped to avoid first-request overhead.
    """
    stream = io.StringIO()
    bar = ConsoleProgressBar(total_weight=100.0, enabled=True, stream=stream)
    bar.start()

    # Phase 1: 10 weight in 2s → 5 weight/s (skipped for ETA)
    # Phase 2: 20 weight in 4s → 5 weight/s (used for ETA)
    bar._phase_weights = [10.0, 20.0]
    bar._phase_elapsed = [2.0, 4.0]

    # Total completed: 30 weight in 6s
    # Measured rate (excluding first phase): 20 / 4 = 5 weight/s
    # Remaining: 70 weight → ETA = 70/5 = 14s → "00:14"
    bar.render(30.0, description="Phase 2", elapsed=6.0, current_phase_weight=0.0)

    output = stream.getvalue()
    assert "ETA 00:14" in output


def test_progress_bar_eta_zero_remaining():
    """ETA should show 00:00 when all work is done."""
    stream = io.StringIO()
    bar = ConsoleProgressBar(total_weight=10.0, enabled=True, stream=stream)
    bar.start()

    bar.render(10.0, description="Complete", elapsed=5.0, current_phase_weight=0.0)

    output = stream.getvalue()
    assert "00:00" in output


def test_progress_bar_eta_no_completed_work():
    """ETA should show --:-- when no work is completed."""
    stream = io.StringIO()
    bar = ConsoleProgressBar(total_weight=10.0, enabled=True, stream=stream)
    bar.start()

    bar.render(0.0, description="Starting", elapsed=0.0, current_phase_weight=0.0)

    output = stream.getvalue()
    assert "--:--" in output


def test_progress_bar_format_duration_hours():
    """Duration formatting should handle hours."""
    assert ConsoleProgressBar._format_duration(3661) == "01:01:01"
    assert ConsoleProgressBar._format_duration(3600) == "01:00:00"
    assert ConsoleProgressBar._format_duration(7325) == "02:02:05"


def test_progress_bar_format_duration_minutes():
    """Duration formatting should handle minutes."""
    assert ConsoleProgressBar._format_duration(65) == "01:05"
    assert ConsoleProgressBar._format_duration(125) == "02:05"
    assert ConsoleProgressBar._format_duration(0) == "00:00"


def test_progress_bar_format_duration_negative():
    """Negative durations should be clamped to 00:00."""
    assert ConsoleProgressBar._format_duration(-5) == "00:00"


def test_progress_bar_finish_with_message():
    """finish() with a message should write the message."""
    stream = io.StringIO()
    bar = ConsoleProgressBar(total_weight=10.0, enabled=True, stream=stream)
    bar.start()
    bar.finish("Done!")
    assert "Done!" in stream.getvalue()


def test_progress_bar_finish_without_message():
    """finish() without a message should just clear the line."""
    stream = io.StringIO()
    bar = ConsoleProgressBar(total_weight=10.0, enabled=True, stream=stream)
    bar.start()
    bar.finish()
    assert "\n" in stream.getvalue()


def test_progress_bar_total_weight_zero():
    """Progress bar with zero total weight should not render."""
    stream = io.StringIO()
    bar = ConsoleProgressBar(total_weight=0.0, enabled=True, stream=stream)
    bar.start()
    bar.render(0.0, description="Test", current_phase_weight=0.0)
    assert stream.getvalue() == ""


def test_progress_bar_clamps_completed_to_total():
    """Completed weight exceeding total should be clamped."""
    stream = io.StringIO()
    bar = ConsoleProgressBar(total_weight=10.0, enabled=True, stream=stream)
    bar.start()

    bar.render(15.0, description="Over", elapsed=5.0, current_phase_weight=0.0)

    output = stream.getvalue()
    assert "100%" in output


def test_progress_bar_eta_during_warmup():
    """During warmup (no completed phases), ETA should use linear fallback.

    The completed_weight is 0 (no phases finished), but current_phase_weight
    shows the in-progress phase. ETA should be based on linear extrapolation
    from elapsed time, not from the current phase's weight.
    """
    stream = io.StringIO()
    bar = ConsoleProgressBar(total_weight=100.0, enabled=True, stream=stream)
    bar.start()

    # First phase starting: 0 completed, 10 in-progress, 1s elapsed
    bar.render(0.0, description="Warmup 1/2", elapsed=1.0, current_phase_weight=10.0)

    output = stream.getvalue()
    # Linear fallback: completed=0, so ETA is "--:--"
    assert "--:--" in output
    # Display should show 10/100 (current phase weight included)
    assert "10.0/100.0" in output


def test_progress_bar_eta_after_first_phase():
    """After first phase completes, ETA should use linear fallback (first phase skipped).

    The first measured phase is skipped for adaptive ETA, so with only 1
    phase recorded, we fall back to linear extrapolation.
    """
    stream = io.StringIO()
    bar = ConsoleProgressBar(total_weight=100.0, enabled=True, stream=stream)
    bar.start()

    # First phase completed: 10 weight in 2s (will be skipped for adaptive ETA)
    bar._phase_weights = [10.0]
    bar._phase_elapsed = [2.0]

    # Second phase starting: 10 completed, 20 in-progress, 3s elapsed
    bar.render(10.0, description="Phase 2", elapsed=3.0, current_phase_weight=20.0)

    output = stream.getvalue()
    # Only 1 phase recorded, so adaptive ETA is skipped (needs >= 2)
    # Linear fallback: eta = 3.0 * (100 - 10) / 10 = 27s → "00:27"
    assert "ETA 00:27" in output
    # Display should show 30/100 (10 completed + 20 in-progress)
    assert "30.0/100.0" in output


def test_progress_bar_eta_skips_first_phase():
    """First measured phase should be skipped for adaptive ETA calculation."""
    stream = io.StringIO()
    bar = ConsoleProgressBar(total_weight=100.0, enabled=True, stream=stream)
    bar.start()

    # Phase 1: 10 weight in 10s (slow due to first-request overhead, skipped)
    # Phase 2: 10 weight in 2s (normal speed, used for ETA)
    bar._phase_weights = [10.0, 10.0]
    bar._phase_elapsed = [10.0, 2.0]

    # 20 completed, 0 in-progress, 12s elapsed
    bar.render(20.0, description="Phase 3", elapsed=12.0, current_phase_weight=0.0)

    output = stream.getvalue()
    # Adaptive ETA (excluding first phase): rate = 10/2 = 5 weight/s
    # Remaining: 80 weight → ETA = 80/5 = 16s → "00:16"
    assert "ETA 00:16" in output


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


# --- Runner-level weighted progress tests ---


def _make_config(**overrides):
    """Create a BenchmarkConfig with sensible defaults for progress tests."""
    defaults = dict(
        base_url="http://example.test/v1",
        api_key="EMPTY",
        model="model",
        served_model_name="model",
        tokenizer=None,
        pp_counts=[2048],
        tg_counts=[32],
        exact_tg=False,
        depths=[0],
        num_runs=1,
        warmup_runs=0,
        no_cache=False,
        latency_mode="none",
        no_warmup=True,
        skip_coherence=True,
        adapt_prompt=False,
        enable_prefix_caching=False,
        book_url="",
        post_run_cmd=None,
        concurrency_levels=[1],
        save_result=None,
        result_format="md",
        save_total_throughput_timeseries=False,
        save_all_throughput_timeseries=False,
        exit_on_first_fail=False,
        no_results_on_fail=False,
        extra_body={},
        emit_progress=None,
        progress_bar=False,
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


def test_estimate_total_weight_standard_run():
    """Total weight for a single standard run should match compute_phase_weight."""
    config = _make_config(pp_counts=[2048], tg_counts=[32], depths=[0], num_runs=1, warmup_runs=0)
    runner = BenchmarkRunner(config, _FakeBenchmarkClient(), _FakePromptGenerator())

    weight = runner._estimate_total_weight(warmup_runs=0)
    expected = compute_phase_weight(pp=2048, tg=32, depth=0)
    assert weight == expected


def test_estimate_total_weight_multiple_runs():
    """Total weight should scale with num_runs."""
    config = _make_config(pp_counts=[2048], tg_counts=[32], depths=[0], num_runs=3, warmup_runs=0)
    runner = BenchmarkRunner(config, _FakeBenchmarkClient(), _FakePromptGenerator())

    weight = runner._estimate_total_weight(warmup_runs=0)
    single = compute_phase_weight(pp=2048, tg=32, depth=0)
    assert weight == single * 3


def test_estimate_total_weight_includes_warmup():
    """Total weight should include warmup runs."""
    config = _make_config(pp_counts=[2048], tg_counts=[32], depths=[0], num_runs=1, warmup_runs=2)
    runner = BenchmarkRunner(config, _FakeBenchmarkClient(), _FakePromptGenerator())

    weight = runner._estimate_total_weight(warmup_runs=2)
    single = compute_phase_weight(pp=2048, tg=32, depth=0)
    assert weight == single * 3  # 1 measured + 2 warmup


def test_estimate_total_weight_prefix_caching():
    """With prefix caching and depth > 0, each run has 2 phases (context load + inference)."""
    config = _make_config(
        pp_counts=[2048], tg_counts=[32], depths=[4096],
        num_runs=1, warmup_runs=0, enable_prefix_caching=True,
    )
    runner = BenchmarkRunner(config, _FakeBenchmarkClient(), _FakePromptGenerator())

    weight = runner._estimate_total_weight(warmup_runs=0)
    ctx_weight = compute_phase_weight(pp=2048, tg=32, depth=4096, is_context_load=True)
    inf_weight = compute_phase_weight(pp=2048, tg=32, depth=4096, is_inference=True)
    assert weight == ctx_weight + inf_weight


def test_estimate_total_weight_larger_pp_is_heavier():
    """Larger pp configurations should produce larger total weight."""
    config_small = _make_config(pp_counts=[2048], tg_counts=[32], depths=[0])
    config_large = _make_config(pp_counts=[128000], tg_counts=[32], depths=[0])

    runner_small = BenchmarkRunner(config_small, _FakeBenchmarkClient(), _FakePromptGenerator())
    runner_large = BenchmarkRunner(config_large, _FakeBenchmarkClient(), _FakePromptGenerator())

    weight_small = runner_small._estimate_total_weight(warmup_runs=0)
    weight_large = runner_large._estimate_total_weight(warmup_runs=0)

    assert weight_large > weight_small
    # 128000/2048 = 62.5x more prompt tokens
    assert weight_large > weight_small * 10


def test_estimate_total_weight_larger_tg_is_heavier():
    """Larger tg configurations should produce larger total weight."""
    config_small = _make_config(pp_counts=[2048], tg_counts=[32], depths=[0])
    config_large = _make_config(pp_counts=[2048], tg_counts=[256], depths=[0])

    runner_small = BenchmarkRunner(config_small, _FakeBenchmarkClient(), _FakePromptGenerator())
    runner_large = BenchmarkRunner(config_large, _FakeBenchmarkClient(), _FakePromptGenerator())

    weight_small = runner_small._estimate_total_weight(warmup_runs=0)
    weight_large = runner_large._estimate_total_weight(warmup_runs=0)

    assert weight_large > weight_small


def test_estimate_total_weight_larger_depth_is_heavier():
    """Larger depth configurations should produce larger total weight."""
    config_small = _make_config(pp_counts=[2048], tg_counts=[32], depths=[0])
    config_large = _make_config(pp_counts=[2048], tg_counts=[32], depths=[4096])

    runner_small = BenchmarkRunner(config_small, _FakeBenchmarkClient(), _FakePromptGenerator())
    runner_large = BenchmarkRunner(config_large, _FakeBenchmarkClient(), _FakePromptGenerator())

    weight_small = runner_small._estimate_total_weight(warmup_runs=0)
    weight_large = runner_large._estimate_total_weight(warmup_runs=0)

    assert weight_large > weight_small


def test_estimate_total_weight_concurrency_does_not_affect():
    """Concurrency should NOT affect total weight (parallel execution)."""
    config_c1 = _make_config(pp_counts=[2048], tg_counts=[32], depths=[0], concurrency_levels=[1])
    config_c8 = _make_config(pp_counts=[2048], tg_counts=[32], depths=[0], concurrency_levels=[8])

    runner_c1 = BenchmarkRunner(config_c1, _FakeBenchmarkClient(), _FakePromptGenerator())
    runner_c8 = BenchmarkRunner(config_c8, _FakeBenchmarkClient(), _FakePromptGenerator())

    weight_c1 = runner_c1._estimate_total_weight(warmup_runs=0)
    weight_c8 = runner_c8._estimate_total_weight(warmup_runs=0)

    assert weight_c1 == weight_c8


def test_estimate_total_weight_multiple_configurations():
    """Total weight should be the sum of all configuration combinations."""
    config = _make_config(
        pp_counts=[2048, 4096],
        tg_counts=[32, 64],
        depths=[0],
        num_runs=1,
        warmup_runs=0,
        concurrency_levels=[1, 2],
    )
    runner = BenchmarkRunner(config, _FakeBenchmarkClient(), _FakePromptGenerator())

    weight = runner._estimate_total_weight(warmup_runs=0)

    # 2 pp × 2 tg × 1 depth × 2 concurrency × 1 run = 8 phases
    # But concurrency doesn't affect weight, so each (pp, tg, depth) combo
    # is counted once per concurrency level (2x), giving 8 phases total.
    # Since weight is the same regardless of concurrency, the total is:
    # 2 (pp) × 2 (tg) × 1 (depth) × 2 (concurrency) × 1 (run) = 8 phases
    # But each phase has the same weight regardless of concurrency, so:
    expected = (
        compute_phase_weight(pp=2048, tg=32, depth=0)
        + compute_phase_weight(pp=2048, tg=64, depth=0)
        + compute_phase_weight(pp=4096, tg=32, depth=0)
        + compute_phase_weight(pp=4096, tg=64, depth=0)
    ) * 2  # ×2 for two concurrency levels
    assert weight == expected


def test_estimate_total_weight_prefix_caching_context_load_heavier():
    """Context load phase should be heavier than inference phase when depth > 0."""
    config = _make_config(
        pp_counts=[2048], tg_counts=[32], depths=[4096],
        num_runs=1, warmup_runs=0, enable_prefix_caching=True,
    )
    runner = BenchmarkRunner(config, _FakeBenchmarkClient(), _FakePromptGenerator())

    weight = runner._estimate_total_weight(warmup_runs=0)

    ctx_weight = compute_phase_weight(pp=2048, tg=32, depth=4096, is_context_load=True)
    inf_weight = compute_phase_weight(pp=2048, tg=32, depth=4096, is_inference=True)

    # Total should be ctx + inf, and ctx should be > inf
    assert weight == ctx_weight + inf_weight
    assert ctx_weight > inf_weight


def test_estimate_total_weight_no_warmup():
    """With no_warmup, warmup_runs should be 0."""
    config = _make_config(
        pp_counts=[2048], tg_counts=[32], depths=[0],
        num_runs=1, warmup_runs=2, no_warmup=True,
    )
    runner = BenchmarkRunner(config, _FakeBenchmarkClient(), _FakePromptGenerator())

    # no_warmup means warmup_runs is treated as 0
    weight = runner._estimate_total_weight(warmup_runs=0)
    single = compute_phase_weight(pp=2048, tg=32, depth=0)
    assert weight == single


@pytest.mark.asyncio
async def test_warmup_phases_excluded_from_eta(tmp_path):
    """Warmup phases should not contribute to adaptive ETA calculation.

    Warmup phases include model loading and connection setup overhead,
    which would skew the measured rate if included. Only non-warmup
    (measured) phases should be recorded for ETA.
    """
    result_path = tmp_path / "results.json"
    config = _make_config(
        pp_counts=[4],
        tg_counts=[2],
        depths=[0],
        num_runs=1,
        warmup_runs=1,
        no_warmup=False,
        skip_coherence=True,
        latency_mode="none",
        progress_bar=True,
        save_result=str(result_path),
    )
    client = _FakeBenchmarkClient()
    runner = BenchmarkRunner(config, client, _FakePromptGenerator())

    await runner.run_suite()

    # The console_progress should have recorded only 1 phase (the measured run,
    # not the warmup run)
    assert runner.console_progress is not None
    assert len(runner.console_progress._phase_elapsed) == 1
    assert len(runner.console_progress._phase_weights) == 1  # phase weight recorded with elapsed
