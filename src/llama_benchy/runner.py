import asyncio
import subprocess
import time
import sys
from datetime import datetime, timezone
from typing import List, Optional
import aiohttp

from ._version import __version__
from .config import BenchmarkConfig
from .client import CONTEXT_LOAD_USER_MESSAGE, LLMClient
from .prompts import PromptGenerator
from .results import BenchmarkResults, BenchmarkMetadata
from .progress import ConsoleProgressBar, compute_phase_weight

class BenchmarkFailure(Exception):
    pass

class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, client: LLMClient, prompt_generator: PromptGenerator, progress=None):
        self.config = config
        self.client = client
        self.prompt_gen = prompt_generator
        self.results = BenchmarkResults()
        self.progress = progress
        self.console_progress: ConsoleProgressBar | None = None
        self._next_request_id = 0

        # We need to track deltas from warmup to adapt prompts
        self.delta_user = 0
        self.delta_context = 0

    def _new_request_id(self) -> int:
        rid = self._next_request_id
        self._next_request_id += 1
        return rid

    def _emit_request_start(self, request_id: int, pp: int, tg: int, depth: int, concurrency: int, run_index: int) -> None:
        if self.progress is None:
            return
        try:
            self.progress.request_start(
                request_id=request_id,
                model=self.config.model,
                base_url=self.config.base_url,
                prompt_size=pp,
                response_size=tg,
                context_size=depth,
                concurrency=concurrency,
                run_index=run_index,
                target_label="",
            )
        except Exception:
            pass

    def _estimate_total_weight(self, warmup_runs: int) -> float:
        """Estimate the total work weight for the entire benchmark suite.

        Each phase's weight is proportional to its estimated token-processing
        cost (prompt tokens * prompt_cost + generation tokens * gen_cost).
        This accounts for the fact that larger pp/depth/tg configurations
        take proportionally longer, and generation is typically much slower
        per token than prompt processing.

        Concurrency does NOT multiply the weight: concurrent requests
        execute in parallel via asyncio.gather, so wall-clock duration is
        approximately the same as a single request.
        """
        total_weight = 0.0
        for depth in self.config.depths:
            for pp in self.config.pp_counts:
                for tg in self.config.tg_counts:
                    for _concurrency in self.config.concurrency_levels:
                        phase_count = 2 if self.config.enable_prefix_caching and depth > 0 else 1
                        for _ in range(self.config.num_runs + warmup_runs):
                            if phase_count == 2:
                                # Context Load phase: full context + probe
                                total_weight += compute_phase_weight(
                                    pp, tg, depth, is_context_load=True
                                )
                                # Inference phase: only pp tokens (context cached)
                                total_weight += compute_phase_weight(
                                    pp, tg, depth, is_inference=True
                                )
                            else:
                                # Standard run: pp + depth tokens
                                total_weight += compute_phase_weight(
                                    pp, tg, depth
                                )
        return total_weight

    def _render_progress(self, completed_weight: float, description: str, current_phase_weight: float = 0.0) -> None:
        if self.console_progress is not None:
            self.console_progress.render(
                completed_weight,
                description=description,
                current_phase_weight=current_phase_weight,
            )

    async def run_suite(self):
        # Initialize session
        timeout = aiohttp.ClientTimeout(total=3600)
        max_concurrency = max(self.config.concurrency_levels)
        connector = aiohttp.TCPConnector(limit=max_concurrency + 5, force_close=False, keepalive_timeout=600)
        latency = 0.0  # default in case of early interrupt
        suite_start_time = time.perf_counter()

        try:
            async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=True) as session:
                # Warmup
                should_warmup = not self.config.no_warmup
                if self.config.adapt_prompt:
                    should_warmup = True

                tokenizer = self.prompt_gen.corpus.get_tokenizer()

                if should_warmup:
                    self.delta_user, self.delta_context = await self.client.warmup(session, tokenizer)

                # Coherence test after warmup (by default, unless skipped)
                if not self.config.skip_coherence:
                    if not await self.client.run_coherence_test(session):
                        print("\nBenchmark failed due to coherence test failure.")
                        raise SystemExit(1)
                else:
                    print("\nSkipping coherence test (--skip-coherence specified)")

                # Measure latency
                warmup_runs = 0 if self.config.no_warmup else self.config.warmup_runs
                latency = await self.client.measure_latency(
                    session,
                    self.config.latency_mode,
                    warmup_runs=warmup_runs,
                )
                if self.progress is not None:
                    try:
                        self.progress.latency_measured(
                            latency_s=latency, mode=self.config.latency_mode
                        )
                    except Exception:
                        pass

                warmup_runs = 0 if self.config.no_warmup else self.config.warmup_runs
                if self.config.progress_bar:
                    self.console_progress = ConsoleProgressBar(
                        self._estimate_total_weight(warmup_runs),
                        enabled=True,
                    )
                    self.console_progress.start()
                else:
                    self.console_progress = None

                completed_weight = 0.0
                current_phase_weight = 0.0
                phase_start_time: Optional[float] = None

                # Main Loop
                for depth in self.config.depths:
                    for pp in self.config.pp_counts:
                        for tg in self.config.tg_counts:
                            for concurrency in self.config.concurrency_levels:
                                # Ensure progress bar line is cleared before printing
                                if self.console_progress is not None:
                                    self.console_progress.stream.write("\n")
                                    self.console_progress.stream.flush()
                                ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                                print(f"[{ts}] Running test: pp={pp}, tg={tg}, depth={depth}, concurrency={concurrency}")

                                run_std_results = []
                                run_ctx_results = []
                                expected_pp = pp
                                expected_ctx = depth

                                total_runs = self.config.num_runs + warmup_runs
                                for run in range(total_runs):
                                    is_warmup = run < warmup_runs
                                    measured_run_index = run - warmup_runs
                                    run_label = (
                                        f"Warmup {run + 1}/{warmup_runs}"
                                        if is_warmup
                                        else f"Run {measured_run_index + 1}/{self.config.num_runs}"
                                    )

                                    # Adapt prompt tokens
                                    current_pp = pp
                                    current_depth = depth
                                    if self.config.adapt_prompt:
                                        if depth == 0:
                                            current_pp = max(1, pp - self.delta_user)
                                        else:
                                            current_depth = max(1, depth - self.delta_context)

                                    expected_pp = current_pp
                                    expected_ctx = current_depth

                                    prompt_batch = self.prompt_gen.generate_batch(
                                        concurrency,
                                        current_pp,
                                        current_depth,
                                        self.config.no_cache
                                    )

                                    if self.config.enable_prefix_caching and depth > 0:
                                        # Phase 1: Context Load
                                        phase_start_time = time.perf_counter()
                                        current_phase_weight = compute_phase_weight(
                                            current_pp, tg, current_depth, is_context_load=True
                                        )
                                        self._render_progress(
                                            completed_weight,
                                            f"{run_label} (Context Load, batch size {concurrency})",
                                            current_phase_weight=current_phase_weight,
                                        )
                                        load_tasks = []
                                        for i in range(concurrency):
                                            context, _ = prompt_batch[i]
                                            if not is_warmup:
                                                rid = self._new_request_id()
                                                self._emit_request_start(rid, pp, tg, depth, concurrency, measured_run_index)
                                            load_tasks.append(self.client.run_generation(
                                                session,
                                                context_text=context,
                                                prompt_text=CONTEXT_LOAD_USER_MESSAGE,
                                                max_tokens=tg,
                                                no_cache=self.config.no_cache,
                                                tokenizer=tokenizer,
                                                progress=None if is_warmup else self.progress,
                                                request_id=None if is_warmup else rid,
                                            ))

                                        load_results = await asyncio.gather(*load_tasks)
                                        if self.console_progress is not None and phase_start_time is not None:
                                            if not is_warmup:
                                                self.console_progress.record_phase_elapsed(
                                                    time.perf_counter() - phase_start_time,
                                                    phase_weight=current_phase_weight,
                                                )
                                            completed_weight += current_phase_weight
                                        if not is_warmup:
                                            run_ctx_results.append(load_results)

                                        if self.config.exit_on_first_fail and any(r.error for r in load_results):
                                            first_error = next(r.error for r in load_results if r.error)
                                            print(f"\n[Error] Stopping due to error in context load: {first_error}")
                                            raise BenchmarkFailure()

                                        # Phase 2: Inference
                                        phase_start_time = time.perf_counter()
                                        current_phase_weight = compute_phase_weight(
                                            current_pp, tg, current_depth, is_inference=True
                                        )
                                        self._render_progress(
                                            completed_weight,
                                            f"{run_label} (Inference, batch size {concurrency})",
                                            current_phase_weight=current_phase_weight,
                                        )
                                        inf_tasks = []
                                        for i in range(concurrency):
                                            context, prompt = prompt_batch[i]
                                            if not is_warmup:
                                                rid = self._new_request_id()
                                                self._emit_request_start(rid, pp, tg, depth, concurrency, measured_run_index)
                                            inf_tasks.append(self.client.run_generation(
                                                session,
                                                context_text=context,
                                                prompt_text=prompt,
                                                max_tokens=tg,
                                                no_cache=self.config.no_cache,
                                                tokenizer=tokenizer,
                                                progress=None if is_warmup else self.progress,
                                                request_id=None if is_warmup else rid,
                                            ))

                                        batch_results = await asyncio.gather(*inf_tasks)
                                        if self.console_progress is not None and phase_start_time is not None:
                                            if not is_warmup:
                                                self.console_progress.record_phase_elapsed(
                                                    time.perf_counter() - phase_start_time,
                                                    phase_weight=current_phase_weight,
                                                )
                                            completed_weight += current_phase_weight
                                        if not is_warmup:
                                            run_std_results.append(batch_results)

                                        if self.config.exit_on_first_fail and any(r.error for r in batch_results):
                                            first_error = next(r.error for r in batch_results if r.error)
                                            print(f"\n[Error] Stopping due to error in inference: {first_error}")
                                            raise BenchmarkFailure()

                                    else:
                                        # Standard Run
                                        phase_start_time = time.perf_counter()
                                        current_phase_weight = compute_phase_weight(
                                            current_pp, tg, current_depth
                                        )
                                        self._render_progress(
                                            completed_weight,
                                            f"{run_label} (batch size {concurrency})",
                                            current_phase_weight=current_phase_weight,
                                        )
                                        expected_tokens = current_pp + current_depth
                                        batch_tasks = []
                                        for i in range(concurrency):
                                            context, prompt = prompt_batch[i]
                                            if not is_warmup:
                                                rid = self._new_request_id()
                                                self._emit_request_start(rid, pp, tg, depth, concurrency, measured_run_index)
                                            batch_tasks.append(self.client.run_generation(
                                                session,
                                                context_text=context,
                                                prompt_text=prompt,
                                                max_tokens=tg,
                                                no_cache=self.config.no_cache,
                                                tokenizer=tokenizer,
                                                progress=None if is_warmup else self.progress,
                                                request_id=None if is_warmup else rid,
                                            ))

                                        batch_results = await asyncio.gather(*batch_tasks)
                                        if self.console_progress is not None and phase_start_time is not None:
                                            if not is_warmup:
                                                self.console_progress.record_phase_elapsed(
                                                    time.perf_counter() - phase_start_time,
                                                    phase_weight=current_phase_weight,
                                                )
                                            completed_weight += current_phase_weight
                                        if not is_warmup:
                                            run_std_results.append(batch_results)

                                        if self.config.exit_on_first_fail and any(r.error for r in batch_results):
                                            first_error = next(r.error for r in batch_results if r.error)
                                            print(f"\n[Error] Stopping due to error in standard run: {first_error}")
                                            raise BenchmarkFailure()


                                    # Post Run Command
                                    if self.config.post_run_cmd:
                                        try:
                                            subprocess.run(self.config.post_run_cmd, shell=True, check=True)
                                        except subprocess.CalledProcessError as e:
                                            print(f"Post-run command failed: {e}")

                                # Aggregate and Record
                                if self.config.enable_prefix_caching and depth > 0:
                                    self.results.add(self.config.model, pp, tg, depth, concurrency, run_ctx_results, latency, expected_ctx, is_context_phase=True, save_total_throughput_timeseries=self.config.save_total_throughput_timeseries, save_all_throughput_timeseries=self.config.save_all_throughput_timeseries)
                                    self.results.add(self.config.model, pp, tg, depth, concurrency, run_std_results, latency, expected_pp, is_context_phase=False, save_total_throughput_timeseries=self.config.save_total_throughput_timeseries, save_all_throughput_timeseries=self.config.save_all_throughput_timeseries)
                                else:
                                    # Standard run expected tokens = pp + depth (usually depth=0 or concatenated)
                                    # In the loop above: expected_tokens = current_pp + current_depth
                                    self.results.add(self.config.model, pp, tg, depth, concurrency, run_std_results, latency, expected_pp + expected_ctx, is_context_phase=False, save_total_throughput_timeseries=self.config.save_total_throughput_timeseries, save_all_throughput_timeseries=self.config.save_all_throughput_timeseries)

                self.results.metadata = BenchmarkMetadata(
                    version=__version__,
                    timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
                    latency_mode=self.config.latency_mode,
                    latency_ms=latency * 1000,
                    model=self.config.model,
                    prefix_caching_enabled=self.config.enable_prefix_caching,
                    max_concurrency=max(self.config.concurrency_levels) if self.config.concurrency_levels else 1
                )

            self.results.save_report(self.config.save_result, self.config.result_format, max(self.config.concurrency_levels) if self.config.concurrency_levels else 1)

            # Print total benchmark time
            total_elapsed = time.perf_counter() - suite_start_time
            print(f"\nTotal benchmark time: {ConsoleProgressBar._format_duration(total_elapsed)}")

        except (asyncio.CancelledError, KeyboardInterrupt, BenchmarkFailure) as e:
            if self.results.runs:
                should_save = True
                if isinstance(e, BenchmarkFailure) and self.config.no_results_on_fail:
                    should_save = False
                    print("\n[Failed] Results discarded per --no-results-on-fail.")

                if should_save:
                    print("\n[Interrupted/Failed] Saving partial results...")
                    if self.results.metadata is None:
                        self.results.metadata = BenchmarkMetadata(
                            version=__version__,
                            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
                            latency_mode=self.config.latency_mode,
                            latency_ms=latency * 1000,
                            model=self.config.model,
                            prefix_caching_enabled=self.config.enable_prefix_caching,
                            max_concurrency=max_concurrency
                        )
                    self.results.save_report(self.config.save_result, self.config.result_format, max_concurrency)

                    # Print total benchmark time even on interruption
                    total_elapsed = time.perf_counter() - suite_start_time
                    print(f"\nTotal benchmark time: {ConsoleProgressBar._format_duration(total_elapsed)}")
            
            if isinstance(e, BenchmarkFailure):
                sys.exit(1)
            raise
        finally:
            if self.console_progress is not None:
                # Render 100% completion before finishing
                self.console_progress.render(
                    self.console_progress.total_weight,
                    description="Complete",
                )
                self.console_progress.finish()
