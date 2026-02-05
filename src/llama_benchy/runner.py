import asyncio
import subprocess
from typing import List
import aiohttp

from .config import BenchmarkConfig
from .client import LLMClient
from .prompts import PromptGenerator
from .results import BenchmarkResults

class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, client: LLMClient, prompt_generator: PromptGenerator):
        self.config = config
        self.client = client
        self.prompt_gen = prompt_generator
        self.results = BenchmarkResults()
        
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
                        
                        run_std_results = []
                        run_ctx_results = []
                        expected_pp = pp
                        expected_ctx = depth

                        for run in range(self.config.num_runs):
                            
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
                                self.config.concurrency, 
                                current_pp, 
                                current_depth, 
                                self.config.no_cache
                            )
                            
                            if self.config.enable_prefix_caching and depth > 0:
                                # Phase 1: Context Load
                                print(f"  Run {run+1}/{self.config.num_runs} (Context Load, batch size {self.config.concurrency})...")
                                load_tasks = []
                                for i in range(self.config.concurrency):
                                    context, _ = prompt_batch[i]
                                    load_tasks.append(self.client.run_generation(
                                        session, 
                                        context_text=context, 
                                        prompt_text="", 
                                        max_tokens=tg,
                                        no_cache=self.config.no_cache
                                    ))
                                
                                load_results = await asyncio.gather(*load_tasks)
                                run_ctx_results.append(load_results)

                                # Phase 2: Inference
                                print(f"  Run {run+1}/{self.config.num_runs} (Inference, batch size {self.config.concurrency})...")
                                inf_tasks = []
                                for i in range(self.config.concurrency):
                                    context, prompt = prompt_batch[i]
                                    inf_tasks.append(self.client.run_generation(
                                        session,
                                        context_text=context,
                                        prompt_text=prompt,
                                        max_tokens=tg,
                                        no_cache=self.config.no_cache
                                    ))
                                
                                batch_results = await asyncio.gather(*inf_tasks)
                                run_std_results.append(batch_results)

                            else:
                                # Standard Run
                                print(f"  Run {run+1}/{self.config.num_runs} (batch size {self.config.concurrency})...")
                                expected_tokens = current_pp + current_depth
                                batch_tasks = []
                                for i in range(self.config.concurrency):
                                    context, prompt = prompt_batch[i]
                                    batch_tasks.append(self.client.run_generation(
                                        session,
                                        context_text=context,
                                        prompt_text=prompt,
                                        max_tokens=tg,
                                        no_cache=self.config.no_cache
                                    ))
                                
                                batch_results = await asyncio.gather(*batch_tasks)
                                run_std_results.append(batch_results)
                                
                            
                            # Post Run Command
                            if self.config.post_run_cmd:
                                try:
                                    subprocess.run(self.config.post_run_cmd, shell=True, check=True)
                                except subprocess.CalledProcessError as e:
                                    print(f"Post-run command failed: {e}")

                        # Aggregate and Record
                        if self.config.enable_prefix_caching and depth > 0:
                             self.results.add(self.config.model, pp, tg, depth, self.config.concurrency, run_ctx_results, latency, expected_ctx, is_context_phase=True)
                             self.results.add(self.config.model, pp, tg, depth, self.config.concurrency, run_std_results, latency, expected_pp, is_context_phase=False)
                        else:
                             # Standard run expected tokens = pp + depth (usually depth=0 or concatenated)
                             # For standard run, expected_tokens was passed to client call? Wait.
                             # In standard run, run_benchmark received `expected_tokens`
                             # In the loop above: expected_tokens = current_pp + current_depth
                             self.results.add(self.config.model, pp, tg, depth, self.config.concurrency, run_std_results, latency, expected_pp + expected_ctx, is_context_phase=False)

        self.results.print_report(self.config.concurrency)
