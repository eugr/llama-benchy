import asyncio
import subprocess
import numpy as np
from tabulate import tabulate
from typing import List, Dict, Any, Tuple
import aiohttp

from .config import BenchmarkConfig
from .client import LLMClient, RequestResult
from .prompts import PromptGenerator

class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, client: LLMClient, prompt_generator: PromptGenerator):
        self.config = config
        self.client = client
        self.prompt_gen = prompt_generator
        self.results_table = []
        
        # We need to track deltas from warmup to adapt prompts
        self.delta_user = 0
        self.delta_context = 0

    async def run_suite(self):
        # Initialize session
        timeout = aiohttp.ClientTimeout(total=3600)
        connector = aiohttp.TCPConnector(limit=self.config.concurrency + 5, force_close=False, keepalive_timeout=600)
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=True) as session:
            # Warmup
            should_warmup = not self.config.no_warmup
            if self.config.adapt_prompt:
                should_warmup = True
            
            if should_warmup:
                tokenizer = self.prompt_gen.corpus.get_tokenizer() if self.config.adapt_prompt else None
                self.delta_user, self.delta_context = await self.client.warmup(session, tokenizer)

            # Measure latency
            latency = await self.client.measure_latency(session, self.config.latency_mode)

            # Main Loop
            for depth in self.config.depths:
                for pp in self.config.pp_counts:
                    for tg in self.config.tg_counts:
                        print(f"Running test: pp={pp}, tg={tg}, depth={depth}, concurrency={self.config.concurrency}")
                        
                        # Storage for aggregated results across all runs
                        agg_results = {
                            "pp_speeds": [], "tg_speeds": [], "ttft_values": [], 
                            "ttfr_values": [], "est_ppt_values": [], "e2e_ttft_values": [],
                            "ctx_pp_speeds": [], "ctx_tg_speeds": [], "ctx_ttfr_values": [], 
                            "ctx_est_ppt_values": [], "ctx_e2e_ttft_values": []
                        }

                        for run in range(self.config.num_runs):
                            
                            # Adapt prompt tokens
                            current_pp = pp
                            current_depth = depth
                            if self.config.adapt_prompt:
                                if depth == 0:
                                    current_pp = max(1, pp - self.delta_user)
                                else:
                                    current_depth = max(1, depth - self.delta_context)

                            # Generate prompts for this batch
                            # For simplicity, we generate one pair and reuse it if concurrency defined 
                            # OR we could generate a batch. Let's start with single generation per run logic 
                            # but duplicated for concurrency.
                            # Better: generate unique prompts for each concurrent request
                            
                            prompt_batch = self.prompt_gen.generate_batch(
                                self.config.concurrency, 
                                current_pp, 
                                current_depth, 
                                self.config.no_cache
                            )
                            
                            batch_tasks = []
                            
                            if self.config.enable_prefix_caching and depth > 0:
                                # Complex flow: Sequence of Load -> Inference for each user
                                # But we want to measure them distinctly?
                                # Original code:
                                # await request1 (Context Load)
                                # await request2 (Inference)
                                
                                # In concurrent mode, we probably want N users doing Load, then N users doing Inference?
                                # OR N users doing (Load then Inference).
                                # To measure "Context Load Speed" under load, we should run N Loads parallel.
                                
                                # Phase 1: Context Load
                                print(f"  Run {run+1}/{self.config.num_runs} (Context Load, batch size {self.config.concurrency})...")
                                load_tasks = []
                                for i in range(self.config.concurrency):
                                    context, _ = prompt_batch[i]
                                    load_tasks.append(self.client.run_generation(
                                        session, 
                                        context_text=context, 
                                        prompt_text="", 
                                        expected_pp_tokens=current_depth,
                                        max_tokens=tg,
                                        no_cache=self.config.no_cache,
                                        latency_offset=0 # Usually latency offset not applied to context load or? Original passed 'latency'
                                        # actually original passed `latency` to ctx run too.
                                    ))
                                
                                load_results = await asyncio.gather(*load_tasks)
                                
                                for res in load_results:
                                    if res and not res.error:
                                        if res.pp_speed: agg_results["ctx_pp_speeds"].append(res.pp_speed)
                                        if res.tg_speed: agg_results["ctx_tg_speeds"].append(res.tg_speed)
                                        if res.ttfr: agg_results["ctx_ttfr_values"].append(res.ttfr)
                                        if res.est_ppt: agg_results["ctx_est_ppt_values"].append(res.est_ppt)
                                        if res.e2e_ttft: agg_results["ctx_e2e_ttft_values"].append(res.e2e_ttft)

                                # Phase 2: Inference
                                print(f"  Run {run+1}/{self.config.num_runs} (Inference, batch size {self.config.concurrency})...")
                                inf_tasks = []
                                for i in range(self.config.concurrency):
                                    context, prompt = prompt_batch[i]
                                    inf_tasks.append(self.client.run_generation(
                                        session,
                                        context_text=context,
                                        prompt_text=prompt,
                                        expected_pp_tokens=current_pp,
                                        max_tokens=tg,
                                        no_cache=self.config.no_cache,
                                        latency_offset=latency
                                    ))
                                
                                batch_results = await asyncio.gather(*inf_tasks)

                            else:
                                # Standard Run
                                print(f"  Run {run+1}/{self.config.num_runs} (batch size {self.config.concurrency})...")
                                expected_tokens = current_pp + current_depth
                                for i in range(self.config.concurrency):
                                    context, prompt = prompt_batch[i]
                                    batch_tasks.append(self.client.run_generation(
                                        session,
                                        context_text=context,
                                        prompt_text=prompt,
                                        expected_pp_tokens=expected_tokens,
                                        max_tokens=tg,
                                        no_cache=self.config.no_cache,
                                        latency_offset=latency
                                    ))
                                
                                batch_results = await asyncio.gather(*batch_tasks)

                            # Collect Standard/Inference Results
                            for res in batch_results:
                                if res and not res.error:
                                    if res.tg_speed: agg_results["tg_speeds"].append(res.tg_speed)
                                    if res.pp_speed: agg_results["pp_speeds"].append(res.pp_speed)
                                    if res.est_ppt: agg_results["est_ppt_values"].append(res.est_ppt)
                                    if res.ttfr: agg_results["ttfr_values"].append(res.ttfr)
                                    if res.ttft: agg_results["ttft_values"].append(res.ttft)
                                    if res.e2e_ttft: agg_results["e2e_ttft_values"].append(res.e2e_ttft)
                            
                            # Post Run Command
                            if self.config.post_run_cmd:
                                try:
                                    subprocess.run(self.config.post_run_cmd, shell=True, check=True)
                                except subprocess.CalledProcessError as e:
                                    print(f"Post-run command failed: {e}")

                        self._record_results(agg_results, self.config.model, pp, tg, depth)

        self._print_final_report()

    def _record_results(self, agg_results: Dict[str, List[float]], model: str, pp: int, tg: int, depth: int):
        def format_result(values, multiplier=1.0):
            if not values: return ""
            mean = np.mean(values) * multiplier
            std = np.std(values) * multiplier
            return f"{mean:.2f} ± {std:.2f}"

        # Context PP (if enabled)
        if agg_results["ctx_pp_speeds"]:
            test_name = f"ctx_pp @ d{depth}"
            # TODO: Concurrency math for throughput?
            # For now, reporting per-request latencies/speeds as averaged.
            self.results_table.append([
                model, 
                test_name, 
                format_result(agg_results["ctx_pp_speeds"]), 
                format_result(agg_results["ctx_ttfr_values"], 1000), 
                format_result(agg_results["ctx_est_ppt_values"], 1000), 
                format_result(agg_results["ctx_e2e_ttft_values"], 1000)
            ])

        # Context TG (if enabled)
        if agg_results["ctx_tg_speeds"]:
            test_name = f"ctx_tg @ d{depth}"
            self.results_table.append([model, test_name, format_result(agg_results["ctx_tg_speeds"]), "", "", ""])

        # Standard PP
        if agg_results["pp_speeds"]:
            test_name = f"pp{pp}"
            if depth > 0: test_name += f" @ d{depth}"
            self.results_table.append([
                model, 
                test_name, 
                format_result(agg_results["pp_speeds"]), 
                format_result(agg_results["ttfr_values"], 1000), 
                format_result(agg_results["est_ppt_values"], 1000), 
                format_result(agg_results["e2e_ttft_values"], 1000)
            ])
        
        # Standard TG
        if agg_results["tg_speeds"]:
            test_name = f"tg{tg}"
            if depth > 0: test_name += f" @ d{depth}"
            self.results_table.append([model, test_name, format_result(agg_results["tg_speeds"]), "", "", ""])

    def _print_final_report(self):
        print()
        if not self.results_table:
            print("No results collected. Check if the model is generating tokens.")
        else:
            print(tabulate(self.results_table, headers=["model", "test", "t/s", "ttfr (ms)", "est_ppt (ms)", "e2e_ttft (ms)"], tablefmt="pipe", colalign=("left", "right", "right", "right", "right", "right")))
