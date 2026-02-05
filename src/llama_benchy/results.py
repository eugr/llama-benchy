import numpy as np
from tabulate import tabulate
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
import csv
import sys

from .client import RequestResult

@dataclass
class BenchmarkMetric:
    mean: float
    std: float
    values: List[float]

@dataclass
class BenchmarkMetadata:
    version: str
    timestamp: str
    latency_mode: str
    latency_ms: float
    model: str
    prefix_caching_enabled: bool
    max_concurrency: int

@dataclass
class BenchmarkRun:
    concurrency: int
    context_size: int
    prompt_size: int
    response_size: int
    is_context_prefill_phase: bool
    
    # Metrics (using BenchmarkMetric)
    pp_throughput: Optional[BenchmarkMetric]
    pp_req_throughput: Optional[BenchmarkMetric]
    tg_throughput: Optional[BenchmarkMetric]
    tg_req_throughput: Optional[BenchmarkMetric]
    ttfr: Optional[BenchmarkMetric]
    est_ppt: Optional[BenchmarkMetric]
    e2e_ttft: Optional[BenchmarkMetric]

class BenchmarkResults:
    def __init__(self):
        self.runs: List[BenchmarkRun] = []
        self.metadata: Optional[BenchmarkMetadata] = None
        self.model_name: Optional[str] = None

    def _calculate_metric(self, values: List[float], multiplier: float = 1.0) -> Optional[BenchmarkMetric]:
        if not values:
            return None
        scaled_values = [v * multiplier for v in values]
        return BenchmarkMetric(
            mean=np.mean(values) * multiplier,
            std=np.std(values) * multiplier,
            values=scaled_values
        )

    def add(self, 
            model: str, 
            pp: int, 
            tg: int, 
            depth: int, 
            concurrency: int, 
            run_results: List[List[RequestResult]], # List of batches (one batch per run)
            latency: float, 
            expected_pp_tokens: int,
            is_context_phase: bool = False):
        
        if self.model_name is None:
            self.model_name = model

        # Aggregators
        agg_pp_speeds = []
        agg_tg_speeds = []
        agg_ttft_values = []
        agg_ttfr_values = []
        agg_est_ppt_values = []
        agg_e2e_ttft_values = []
        
        agg_batch_pp_throughputs = []
        agg_batch_tg_throughputs = []

        for batch in run_results:
            self._process_batch(
                batch, 
                expected_pp_tokens, 
                latency, 
                agg_pp_speeds, 
                agg_tg_speeds, 
                agg_ttft_values, 
                agg_ttfr_values, 
                agg_est_ppt_values, 
                agg_e2e_ttft_values, 
                agg_batch_pp_throughputs, 
                agg_batch_tg_throughputs
            )

        # Calculate metrics for BenchmarkRun
        run_metric_pp_throughput = self._calculate_metric(agg_batch_pp_throughputs if concurrency > 1 else agg_pp_speeds)
        run_metric_pp_req_throughput = self._calculate_metric(agg_pp_speeds) if concurrency > 1 else None
        
        run_metric_tg_throughput = self._calculate_metric(agg_batch_tg_throughputs if concurrency > 1 else agg_tg_speeds)
        run_metric_tg_req_throughput = self._calculate_metric(agg_tg_speeds) if concurrency > 1 else None

        run_metric_ttfr = self._calculate_metric(agg_ttfr_values, 1000)
        run_metric_est_ppt = self._calculate_metric(agg_est_ppt_values, 1000)
        run_metric_e2e_ttft = self._calculate_metric(agg_e2e_ttft_values, 1000)

        self.runs.append(BenchmarkRun(
            concurrency=concurrency,
            context_size=depth,
            prompt_size=pp, # Configured prompt size
            response_size=tg,
            is_context_prefill_phase=is_context_phase,
            pp_throughput=run_metric_pp_throughput,
            pp_req_throughput=run_metric_pp_req_throughput,
            tg_throughput=run_metric_tg_throughput,
            tg_req_throughput=run_metric_tg_req_throughput,
            ttfr=run_metric_ttfr,
            est_ppt=run_metric_est_ppt,
            e2e_ttft=run_metric_e2e_ttft
        ))

    def _process_batch(self, 
                       results: List[RequestResult], 
                       expected_pp_tokens: int, 
                       latency: float,
                       agg_pp_speeds: List[float],
                       agg_tg_speeds: List[float],
                       agg_ttft_values: List[float],
                       agg_ttfr_values: List[float],
                       agg_est_ppt_values: List[float],
                       agg_e2e_ttft_values: List[float],
                       agg_batch_pp_throughputs: List[float],
                       agg_batch_tg_throughputs: List[float]):
        
        valid_results = [r for r in results if r and not r.error]
        if not valid_results:
            return

        batch_prompt_tokens = 0
        batch_gen_tokens = 0
        
        start_times = []
        end_times = []
        first_token_times = []

        for res in valid_results:
            start_times.append(res.start_ts)
            end_times.append(res.end_ts)
            
            # Use reported usage if available and reasonable, else expected
            prompt_tokens = expected_pp_tokens
            if res.prompt_tokens > 0:
                diff = abs(res.prompt_tokens - expected_pp_tokens)
                if diff < expected_pp_tokens * 0.2:
                    prompt_tokens = res.prompt_tokens
            
            batch_prompt_tokens += prompt_tokens
            batch_gen_tokens += res.total_tokens

            # Metrics Calculation
            ttft = 0.0
            e2e_ttft = 0.0
            ttfr = 0.0
            est_ppt = 0.0
            
            if res.first_response_ts:
                ttfr = res.first_response_ts - res.start_ts
                agg_ttfr_values.append(ttfr)
            
            if res.first_token_ts:
                first_token_times.append(res.first_token_ts)
                e2e_ttft = res.first_token_ts - res.start_ts
                ttft = max(0, e2e_ttft - latency)
                est_ppt = max(0, ttfr - latency)

                agg_e2e_ttft_values.append(e2e_ttft)
                agg_ttft_values.append(ttft)
                agg_est_ppt_values.append(est_ppt)

            # Individual Speeds
            if est_ppt > 0:
                pp_speed = prompt_tokens / est_ppt
                agg_pp_speeds.append(pp_speed)
            
            if res.total_tokens > 1 and res.first_token_ts:
                decode_time = res.end_ts - res.first_token_ts
                if decode_time > 0:
                    tg_speed = (res.total_tokens - 1) / decode_time
                    agg_tg_speeds.append(tg_speed)

        # Batch-Level Throughput
        if start_times and end_times and first_token_times:
            min_start = min(start_times)
            max_end = max(end_times)
            
            max_first_token = max(first_token_times)
            pp_duration = max_first_token - min_start
            
            if pp_duration > 0:
                batch_pp_throughput = batch_prompt_tokens / pp_duration
                agg_batch_pp_throughputs.append(batch_pp_throughput)
            
            min_first_token = min(first_token_times)
            tg_duration = max_end - min_first_token
            
            if tg_duration > 0:
                if batch_gen_tokens > len(valid_results):
                     batch_tg_throughput = (batch_gen_tokens - len(valid_results)) / tg_duration
                     agg_batch_tg_throughputs.append(batch_tg_throughput)


    def _generate_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for run in self.runs:
            c_suffix = ""
            if self.metadata and self.metadata.max_concurrency > 1:
                c_suffix = f" (c{run.concurrency})"

            if run.is_context_prefill_phase:
                # Context Phase Prompt Processing
                if run.pp_throughput:
                    rows.append({
                        "model": self.model_name or "Unknown",
                        "test_name": f"ctx_pp @ d{run.context_size}{c_suffix}",
                        "t_s": run.pp_throughput,
                        "t_s_req": run.pp_req_throughput,
                        "ttfr": run.ttfr,
                        "est_ppt": run.est_ppt,
                        "e2e_ttft": run.e2e_ttft
                    })
                
                # Context Phase Token Generation
                if run.tg_throughput:
                    rows.append({
                        "model": self.model_name or "Unknown",
                        "test_name": f"ctx_tg @ d{run.context_size}{c_suffix}",
                        "t_s": run.tg_throughput,
                        "t_s_req": run.tg_req_throughput,
                        "ttfr": None,
                        "est_ppt": None,
                        "e2e_ttft": None
                    })
            else:
                # Standard Phase
                d_suffix = f" @ d{run.context_size}" if run.context_size > 0 else ""
                
                # Prompt Processing
                if run.pp_throughput:
                    rows.append({
                        "model": self.model_name or "Unknown",
                        "test_name": f"pp{run.prompt_size}{d_suffix}{c_suffix}",
                        "t_s": run.pp_throughput,
                        "t_s_req": run.pp_req_throughput,
                        "ttfr": run.ttfr,
                        "est_ppt": run.est_ppt,
                        "e2e_ttft": run.e2e_ttft
                    })
                
                # Token Generation
                if run.tg_throughput:
                    rows.append({
                        "model": self.model_name or "Unknown",
                        "test_name": f"tg{run.response_size}{d_suffix}{c_suffix}",
                        "t_s": run.tg_throughput,
                        "t_s_req": run.tg_req_throughput,
                        "ttfr": None,
                        "est_ppt": None,
                        "e2e_ttft": None
                    })
        return rows

    def _generate_md_report(self, concurrency: int) -> str:
        rows = self._generate_rows()
        if not rows:
            return "No results collected. Check if the model is generating tokens."

        def fmt(metric: Optional[BenchmarkMetric]) -> str:
            if metric is None:
                return ""
            return f"{metric.mean:.2f} ± {metric.std:.2f}"
            
        data = [[
            row["model"], 
            row["test_name"], 
            fmt(row["t_s"]), 
            fmt(row["t_s_req"]), 
            fmt(row["ttfr"]), 
            fmt(row["est_ppt"]), 
            fmt(row["e2e_ttft"])
        ] for row in rows]

        ts_header = "t/s (total)" if concurrency > 1 else "t/s"
        headers = ["model", "test", ts_header, "t/s (req)", "ttfr (ms)", "est_ppt (ms)", "e2e_ttft (ms)"]
        
        if concurrency == 1:
            data = [[
                row["model"], 
                row["test_name"], 
                fmt(row["t_s"]), 
                fmt(row["ttfr"]), 
                fmt(row["est_ppt"]), 
                fmt(row["e2e_ttft"])
            ] for row in rows]
            headers = ["model", "test", ts_header, "ttfr (ms)", "est_ppt (ms)", "e2e_ttft (ms)"]

        return tabulate(data, headers=headers, tablefmt="pipe", colalign=("left", "right", "right", "right", "right", "right", "right") if concurrency > 1 else ("left", "right", "right", "right", "right", "right"))

    def save_report(self, filename: Optional[str], format: str, concurrency: int = 1):
        msg = ""
        if filename:
            msg += f"Saving results to {filename} in {format.upper()} format...\n"
        else:            
            msg += f"Printing results in {format.upper()} format:\n"

        print(f"{msg}\n")

        if format == "md":
            output = self._generate_md_report(concurrency)
            if filename:
                with open(filename, "w") as f:
                    f.write(output)
            else:
                 print("\n" + output)
        
        elif format == "json":
            data = asdict(self.metadata) if self.metadata else {}
            data["benchmarks"] = [asdict(run) for run in self.runs]
            
            if filename:
                 with open(filename, "w") as f:
                     json.dump(data, f, indent=2)
            else:
                 print(json.dumps(data, indent=2))
        
        elif format == "csv":
             rows = self._generate_rows()
             csv_rows = []
             headers = ["model", "test_name", "t_s_mean", "t_s_std", "t_s_req_mean", "t_s_req_std", "ttfr_mean", "ttfr_std", "est_ppt_mean", "est_ppt_std", "e2e_ttft_mean", "e2e_ttft_std"]
             
             for r in rows:
                 row = {
                     "model": r["model"],
                     "test_name": r["test_name"],
                     "t_s_mean": r["t_s"].mean if r["t_s"] else None,
                     "t_s_std": r["t_s"].std if r["t_s"] else None,
                     "t_s_req_mean": r["t_s_req"].mean if r["t_s_req"] else None,
                     "t_s_req_std": r["t_s_req"].std if r["t_s_req"] else None,
                     "ttfr_mean": r["ttfr"].mean if r["ttfr"] else None,
                     "ttfr_std": r["ttfr"].std if r["ttfr"] else None,
                     "est_ppt_mean": r["est_ppt"].mean if r["est_ppt"] else None,
                     "est_ppt_std": r["est_ppt"].std if r["est_ppt"] else None,
                     "e2e_ttft_mean": r["e2e_ttft"].mean if r["e2e_ttft"] else None,
                     "e2e_ttft_std": r["e2e_ttft"].std if r["e2e_ttft"] else None,
                 }
                 csv_rows.append(row)
             
             output_file = filename if filename else sys.stdout
             is_file = isinstance(output_file, str)
             
             if is_file:
                 with open(output_file, "w", newline="") as f:
                      writer = csv.DictWriter(f, fieldnames=headers)
                      writer.writeheader()
                      writer.writerows(csv_rows)
             else:
                 writer = csv.DictWriter(sys.stdout, fieldnames=headers)
                 writer.writeheader()
                 writer.writerows(csv_rows)


