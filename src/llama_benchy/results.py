import numpy as np
from tabulate import tabulate
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .client import RequestResult

@dataclass
class BenchmarkResultEntry:
    model: str
    test_name: str
    t_s: str
    ttfr: str
    est_ppt: str
    e2e_ttft: str

class BenchmarkResults:
    def __init__(self):
        self.entries: List[BenchmarkResultEntry] = []

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

        # Determine Test Name
        prefix = "ctx_" if is_context_phase else ""
        if is_context_phase:
             test_name = f"ctx_pp @ d{depth}"
             # Check if we have data
             has_data = len(agg_pp_speeds) > 0 or len(agg_batch_pp_throughputs) > 0
             if has_data:
                 self._create_entry(model, test_name, concurrency, agg_batch_pp_throughputs if concurrency > 1 else agg_pp_speeds, agg_ttfr_values, agg_est_ppt_values, agg_e2e_ttft_values)
             
             test_name_tg = f"ctx_tg @ d{depth}"
             has_data_tg = len(agg_tg_speeds) > 0 or len(agg_batch_tg_throughputs) > 0
             if has_data_tg:
                 self._create_entry(model, test_name_tg, concurrency, agg_batch_tg_throughputs if concurrency > 1 else agg_tg_speeds, [], [], [])

        else:
            # Standard
            test_name = f"pp{pp}"
            if depth > 0: test_name += f" @ d{depth}"
            
            has_data = len(agg_pp_speeds) > 0 or len(agg_batch_pp_throughputs) > 0
            if has_data:
                self._create_entry(model, test_name, concurrency, agg_batch_pp_throughputs if concurrency > 1 else agg_pp_speeds, agg_ttfr_values, agg_est_ppt_values, agg_e2e_ttft_values)
            
            test_name_tg = f"tg{tg}"
            if depth > 0: test_name_tg += f" @ d{depth}"
            
            has_data_tg = len(agg_tg_speeds) > 0 or len(agg_batch_tg_throughputs) > 0
            if has_data_tg:
                self._create_entry(model, test_name_tg, concurrency, agg_batch_tg_throughputs if concurrency > 1 else agg_tg_speeds, [], [], [])


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


    def _create_entry(self, model, test_name, concurrency, speed_values, ttfr_values, est_ppt_values, e2e_ttft_values):
        def format_result(values, multiplier=1.0):
            if not values: return ""
            mean = np.mean(values) * multiplier
            std = np.std(values) * multiplier
            return f"{mean:.2f} ± {std:.2f}"

        self.entries.append(BenchmarkResultEntry(
            model=model,
            test_name=test_name,
            t_s=format_result(speed_values),
            ttfr=format_result(ttfr_values, 1000),
            est_ppt=format_result(est_ppt_values, 1000),
            e2e_ttft=format_result(e2e_ttft_values, 1000)
        ))

    def print_report(self, concurrency: int = 1):
        data = [[e.model, e.test_name, e.t_s, e.ttfr, e.est_ppt, e.e2e_ttft] for e in self.entries]
        print()
        if not data:
            print("No results collected. Check if the model is generating tokens.")
        else:
            ts_header = "t/s (total)" if concurrency > 1 else "t/s"
            print(tabulate(data, headers=["model", "test", ts_header, "ttfr (ms)", "est_ppt (ms)", "e2e_ttft (ms)"], tablefmt="pipe", colalign=("left", "right", "right", "right", "right", "right")))
