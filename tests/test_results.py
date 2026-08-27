import pytest

from llama_benchy.client import RequestResult
from llama_benchy.results import BenchmarkResults


def test_decode_throughput_uses_observed_token_interval():
    result = RequestResult(
        start_ts=0.0,
        first_response_ts=1.0,
        first_token_ts=1.0,
        end_ts=1.35,
        prompt_tokens=100,
        total_tokens=4,
        token_timestamps=[1.0, 1.1, 1.2, 1.3],
    )

    results = BenchmarkResults()
    results.add("model", 100, 4, 0, 1, [[result]], latency=0.0, expected_pp_tokens=100)

    assert results.runs[0].tg_throughput is not None
    assert results.runs[0].tg_throughput.mean == pytest.approx(10.0)


def test_burst_output_does_not_report_decode_throughput():
    result = RequestResult(
        start_ts=0.0,
        first_response_ts=3.0,
        first_token_ts=3.0,
        end_ts=3.001,
        prompt_tokens=2048,
        total_tokens=130,
        token_timestamps=[3.0] * 130,
    )

    results = BenchmarkResults()
    results.add("model", 2048, 1024, 0, 1, [[result]], latency=0.0, expected_pp_tokens=2048)

    run = results.runs[0]
    assert run.tg_throughput is None
    assert run.peak_throughput is not None
    assert run.peak_throughput.mean == pytest.approx(130.0)

    rows = results._generate_rows()
    tg_row = next(row for row in rows if row["test_name"] == "tg1024")
    assert tg_row["t_s"] is None
    assert tg_row["peak_ts"] is not None


def test_block_streaming_excludes_first_observed_block_from_decode_throughput():
    first_block = [1.0] * 256
    second_block = [1.25 + (0.75 * (i + 1) / 256) for i in range(256)]

    result = RequestResult(
        start_ts=0.0,
        first_response_ts=0.9,
        first_token_ts=1.0,
        end_ts=2.01,
        prompt_tokens=2048,
        total_tokens=512,
        token_timestamps=first_block + second_block,
    )

    results = BenchmarkResults()
    results.add("model", 2048, 512, 0, 1, [[result]], latency=0.0, expected_pp_tokens=2048)

    run = results.runs[0]
    assert run.tg_throughput is not None
    assert run.tg_throughput.mean == pytest.approx(256.0)


def test_prefill_throughput_uses_first_content_token_not_empty_response():
    result = RequestResult(
        start_ts=0.0,
        first_response_ts=0.01,
        first_token_ts=2.0,
        end_ts=2.2,
        prompt_tokens=2_000,
        total_tokens=3,
        token_timestamps=[2.0, 2.1, 2.2],
    )

    results = BenchmarkResults()
    results.add(
        "model",
        2_000,
        3,
        0,
        1,
        [[result]],
        latency=0.1,
        expected_pp_tokens=2_000,
    )

    run = results.runs[0]
    assert run.ttfr is not None
    assert run.ttfr.mean == pytest.approx(10.0)
    assert run.est_ppt is not None
    assert run.est_ppt.mean == pytest.approx(1_900.0)
    assert run.pp_throughput is not None
    assert run.pp_throughput.mean == pytest.approx(2_000 / 1.9)


def test_dataset_run_reports_observed_sizes_and_prompt_id():
    result = RequestResult(
        start_ts=0.0,
        first_response_ts=1.0,
        first_token_ts=1.1,
        end_ts=1.4,
        prompt_tokens=25_711,
        total_tokens=309,
        token_timestamps=[1.1, 1.2, 1.3],
    )

    results = BenchmarkResults()
    results.add(
        "model",
        25_711,
        309,
        0,
        1,
        [[result]],
        latency=0.0,
        expected_pp_tokens=25_711,
        prompt_id="task@trajectory",
    )

    run = results.runs[0]
    assert (run.prompt_id, run.prompt_size, run.response_size) == (
        "task@trajectory", 25_711, 309
    )
    assert {row["test_name"] for row in results._generate_rows()} == {
        "task@trajectory pp25711",
        "task@trajectory tg309",
    }


def test_dataset_concurrency_uses_each_requests_observed_prompt_size():
    first = RequestResult(
        start_ts=0.0,
        first_response_ts=0.01,
        first_token_ts=1.0,
        end_ts=1.2,
        prompt_tokens=100,
        total_tokens=3,
        token_timestamps=[1.0, 1.1, 1.2],
    )
    second = RequestResult(
        start_ts=0.0,
        first_response_ts=0.01,
        first_token_ts=2.0,
        end_ts=2.2,
        prompt_tokens=500,
        total_tokens=3,
        token_timestamps=[2.0, 2.1, 2.2],
    )

    results = BenchmarkResults()
    results.add(
        "model", 300, 3, 0, 2, [[first, second]], 0.0, 300,
        use_observed_prompt_tokens=True,
    )

    run = results.runs[0]
    assert run.pp_throughput is not None
    assert run.pp_throughput.mean == pytest.approx(300.0)
    assert run.pp_req_throughput is not None
    assert run.pp_req_throughput.mean == pytest.approx(175.0)
