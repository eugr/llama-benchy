"""Optional progress-event emitter for llama-benchy.

When the user passes ``--emit-progress PATH``, llama-benchy writes a stream
of newline-delimited JSON events to PATH (or stdout when PATH is ``-``).
Consumers — separate tools, e.g. live TUIs, web dashboards, post-hoc
visualizers — parse that stream and render whatever they like.

Schema spec:        docs/progress-schema.md
Schema version tag: ``llama-benchy-progress.v1``

This module is intentionally small. It carries no UI, no rendering, no
optional deps. Anything fancier lives in a separate consumer repo.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from typing import IO, List, Optional, Tuple

SCHEMA_VERSION = "llama-benchy-progress.v1"

# Relative cost weights for prompt-processing tokens vs. generation tokens.
# Prompt processing (prefill) is typically much faster per token than
# token generation (decode) on modern LLM servers. These defaults reflect
# a typical ratio (~20x), but the WeightedProgressTracker allows callers to
# override them. The key invariant: GEN_COST > PROMPT_COST, so generation
# workloads contribute proportionally more progress than prompt workloads.
DEFAULT_PROMPT_COST = 1.0
DEFAULT_GEN_COST = 20.0


class ConsoleProgressBar:
    """Simple console progress indicator with ETA for benchmark phases.

    Uses a weighted-progress model: each phase contributes progress
    proportional to its estimated work (token-processing cost) rather
    than counting every phase as an equal unit. ETA is refined
    adaptively using actual elapsed time from completed phases.
    """

    def __init__(
        self,
        total_weight: float = 0.0,
        *,
        enabled: bool = False,
        stream: Optional[IO[str]] = None,
        prompt_cost: float = DEFAULT_PROMPT_COST,
        gen_cost: float = DEFAULT_GEN_COST,
    ) -> None:
        self.total_weight = max(0.0, total_weight)
        self.enabled = enabled
        self.stream = stream or sys.stdout
        self.prompt_cost = prompt_cost
        self.gen_cost = gen_cost
        self._started_at: Optional[float] = None
        self._completed_weight: float = 0.0
        self._phase_weights: List[float] = []
        self._phase_elapsed: List[float] = []
        self._current_phase_start: Optional[float] = None

    def start(self) -> None:
        if not self.enabled:
            return
        self._started_at = time.perf_counter()

    def add_phase(self, weight: float) -> None:
        """Register a phase with its estimated work weight.

        Called before the benchmark starts to pre-register all phases,
        or incrementally as phases are discovered. The total weight is
        updated accordingly.
        """
        if not self.enabled:
            return
        self._phase_weights.append(max(0.0, weight))
        self.total_weight = sum(self._phase_weights)

    def render(
        self,
        completed_weight: float,
        *,
        description: str = "",
        total_weight: Optional[float] = None,
        elapsed: Optional[float] = None,
        current_phase_weight: float = 0.0,
    ) -> None:
        """Render the progress bar.

        Args:
            completed_weight: Cumulative weight of **fully completed** phases.
                This is used for ETA calculation (only completed work counts).
            description: Human-readable description of the current phase.
            total_weight: Override total weight (defaults to self.total_weight).
            elapsed: Override elapsed time (for testing).
            current_phase_weight: Weight of the current in-progress phase,
                added to the display for visual progress but NOT included
                in ETA calculation.
        """
        if not self.enabled:
            return

        total = self.total_weight if total_weight is None else max(0.0, total_weight)
        if total <= 0:
            return

        # For display: include current phase weight
        display_completed = max(0.0, min(completed_weight + current_phase_weight, total))
        # For ETA: only use actually completed weight
        eta_completed = max(0.0, min(completed_weight, total))

        if self._started_at is None:
            self._started_at = time.perf_counter()

        elapsed = time.perf_counter() - self._started_at if elapsed is None else elapsed
        percent = (display_completed / total) * 100 if total else 0
        filled_width = 24
        filled = int((display_completed / total) * filled_width) if total else 0
        bar = "[" + "=" * filled + (">" if display_completed < total and filled < filled_width else "") + "-" * max(0, filled_width - filled - (1 if display_completed < total and filled < filled_width else 0)) + "]"

        eta_text = self._compute_eta(eta_completed, total, elapsed)

        message = f"{bar} {display_completed:.1f}/{total:.1f} {description} | {percent:3.0f}% | ETA {eta_text}".rstrip()
        self.stream.write("\r\033[K")
        self.stream.write(message)
        self.stream.flush()

    def _compute_eta(self, completed: float, total: float, elapsed: float) -> str:
        """Compute ETA using a hybrid approach.

        - If we have actual per-phase timings, use the average weight-per-second
          rate from completed phases to estimate remaining time.
        - Otherwise, fall back to a simple linear extrapolation based on
          elapsed time and completed weight fraction.

        The first measured phase is excluded from the rate calculation to
        avoid first-request overhead (connection pooling, server-side warmup)
        skewing the estimate.
        """
        if completed <= 0 or elapsed <= 0:
            return "--:--"

        remaining = total - completed
        if remaining <= 0:
            return "00:00"

        # Adaptive ETA: use actual measured rate from completed phases
        # Skip the first measured phase to avoid first-request overhead skewing
        if len(self._phase_elapsed) >= 2 and sum(self._phase_elapsed[1:]) > 0:
            total_phase_weight = sum(self._phase_weights[1:len(self._phase_elapsed)])
            if total_phase_weight > 0:
                measured_rate = total_phase_weight / sum(self._phase_elapsed[1:])
                eta_seconds = remaining / measured_rate
                return self._format_duration(eta_seconds)

        # Fallback: linear extrapolation
        eta_seconds = elapsed * (total - completed) / completed
        return self._format_duration(eta_seconds)

    def record_phase_elapsed(self, elapsed: float, phase_weight: float = 0.0) -> None:
        """Record the actual elapsed time and weight for the most recent phase.

        Used for adaptive ETA computation. Both elapsed time and phase weight
        must be recorded together so the measured rate (weight/time) is accurate.
        """
        if not self.enabled:
            return
        self._phase_elapsed.append(max(0.0, elapsed))
        self._phase_weights.append(max(0.0, phase_weight))

    def finish(self, message: Optional[str] = None) -> None:
        if not self.enabled:
            return
        if message:
            self.stream.write("\r\033[K")
            self.stream.write(f"{message}\n")
        else:
            self.stream.write("\r\033[K\n")
        self.stream.flush()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        minutes, sec = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{sec:02d}"
        return f"{minutes:02d}:{sec:02d}"


def compute_phase_weight(
    pp: int,
    tg: int,
    depth: int,
    *,
    is_context_load: bool = False,
    is_inference: bool = False,
    prompt_cost: float = DEFAULT_PROMPT_COST,
    gen_cost: float = DEFAULT_GEN_COST,
) -> float:
    """Compute the estimated work weight for a single benchmark phase.

    The weight is proportional to the estimated wall-clock duration of the
    phase, based on the number of tokens the server must process and
    generate.

    Args:
        pp: Prompt processing token count (user-facing --pp).
        tg: Token generation count (user-facing --tg).
        depth: Context depth (previous conversation tokens).
        is_context_load: True if this is a prefix-cache context-load phase
            (full context + probe message sent, KV cache warmed).
        is_inference: True if this is a prefix-cache inference phase
            (only user prompt sent, context is cached).
        prompt_cost: Relative cost per prompt-processing token.
        gen_cost: Relative cost per generation token.

    Returns:
        A non-negative float representing the estimated work weight.

    Weighting rationale:
        - Context Load phase: processes (pp + depth) prompt tokens + generates tg tokens.
          The full context is sent to warm the KV cache.
        - Inference phase (prefix caching): processes only pp prompt tokens
          (context is cached) + generates tg tokens.
        - Standard run (no prefix caching): processes (pp + depth) prompt tokens
          + generates tg tokens.
        - Generation cost dominates: gen_cost >> prompt_cost, reflecting that
          token generation (decode) is typically 10-50x slower per token
          than prompt processing (prefill) on modern LLM servers.
        - Concurrency does NOT multiply the weight: concurrent requests
          execute in parallel via asyncio.gather, so wall-clock duration
          is approximately the same as a single request (assuming the
          server can handle the load). The total server work increases,
          but the user-visible duration does not.
    """
    if is_context_load:
        # Full context + probe message processed, then tg tokens generated
        prompt_tokens = pp + depth
    elif is_inference:
        # Only user prompt processed (context cached), then tg tokens generated
        prompt_tokens = pp
    else:
        # Standard run: pp + depth tokens processed, tg tokens generated
        prompt_tokens = pp + depth

    return prompt_tokens * prompt_cost + tg * gen_cost


class ProgressEmitter:
    """Append-only JSONL writer for benchmark progress events.

    Thread-safe (a lock guards the underlying file write). Methods are
    no-throwing — emit failures are silently dropped so a broken consumer
    can't take down a benchmark run.
    """

    def __init__(self, target: str, *, llama_benchy_version: str = "unknown") -> None:
        self._target = target
        self._lock = threading.Lock()
        self._stream: Optional[IO[str]] = None
        self._owns_stream = False
        self._open(llama_benchy_version)

    def _open(self, llama_benchy_version: str) -> None:
        if self._target == "-":
            # Caller is expected to have already redirected sys.stdout to
            # sys.stderr (see __main__) so llama-benchy's regular status
            # prints don't corrupt the JSONL stream we emit here.
            self._stream = sys.__stdout__
            self._owns_stream = False
        else:
            self._stream = open(self._target, "w", buffering=1)  # line-buffered
            self._owns_stream = True
        self._write(
            {
                "schema": SCHEMA_VERSION,
                "type": "header",
                "ts": time.time(),
                "llama_benchy_version": llama_benchy_version,
            }
        )

    # event API
    def request_start(
        self,
        *,
        request_id: int,
        model: str,
        base_url: str,
        prompt_size: int,
        response_size: int,
        context_size: int,
        concurrency: int,
        run_index: int,
        target_label: str = "",
    ) -> None:
        self._emit(
            "request_start",
            request_id=request_id,
            model=model,
            base_url=base_url,
            prompt_size=prompt_size,
            response_size=response_size,
            context_size=context_size,
            concurrency=concurrency,
            run_index=run_index,
            target_label=target_label,
        )

    def request_first_response(self, *, request_id: int, ttfr_s: float) -> None:
        """First chunk of any kind arrived (may be empty / role-only)."""
        self._emit("request_first_response", request_id=request_id, ttfr_s=ttfr_s)

    def request_first_token(self, *, request_id: int, ttft_s: float) -> None:
        """First content-bearing token arrived (== e2e_ttft)."""
        self._emit("request_first_token", request_id=request_id, ttft_s=ttft_s)

    def latency_measured(self, *, latency_s: float, mode: str) -> None:
        """Network latency probe complete (used to derive est_ppt = ttfr − latency)."""
        self._emit("latency_measured", latency_s=latency_s, mode=mode)

    def tokens(self, *, request_id: int, count: int, snippet: str = "", estimated: bool = False) -> None:
        if count <= 0 and not snippet:
            return
        fields = {"request_id": request_id, "count": count, "snippet": snippet}
        if estimated:
            fields["estimated"] = True
        self._emit("tokens", **fields)

    def request_end(
        self,
        *,
        request_id: int,
        total_tokens: int,
        prompt_tokens: int,
        decode_seconds: float,
        error: str = "",
    ) -> None:
        self._emit(
            "request_end",
            request_id=request_id,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            decode_seconds=decode_seconds,
            error=error,
        )

    def bench_complete(self, status: str = "ok") -> None:
        self._emit("bench_complete", status=status)

    def close(self) -> None:
        with self._lock:
            if self._stream is not None and self._owns_stream:
                try:
                    self._stream.close()
                except Exception:
                    pass
            self._stream = None

    def __enter__(self) -> "ProgressEmitter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # internals
    def _emit(self, event_type: str, **fields) -> None:
        self._write(
            {
                "schema": SCHEMA_VERSION,
                "type": event_type,
                "ts": time.time(),
                **fields,
            }
        )

    def _write(self, obj: dict) -> None:
        if self._stream is None:
            return
        try:
            line = json.dumps(obj, separators=(",", ":"))
        except (TypeError, ValueError):
            return
        with self._lock:
            try:
                self._stream.write(line)
                self._stream.write("\n")
                self._stream.flush()
            except Exception:
                # Consumer hung up / disk full — don't crash the benchmark.
                pass
