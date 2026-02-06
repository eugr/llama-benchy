import json
import asyncio
import sys
import time
import subprocess
import uvicorn
import pytest
import multiprocessing
import aiohttp
from typing import Dict, Any, List

# --- Helpers to run server in background ---

def run_server_process():
    """Entry point for the server process."""
    # Import inside process to avoid polluting parent scope
    from tests.mock_server import app 
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="error")

@pytest.fixture(scope="module")
def mock_server_url():
    """Fixture to start/stop the mock server."""
    # Start server in a separate process
    proc = multiprocessing.Process(target=run_server_process, daemon=True)
    proc.start()
    
    # Wait for server to be ready
    base_url = "http://127.0.0.1:8001/v1"
    
    max_retries = 20
    for _ in range(max_retries):
        try:
            # We can't use requests because we are in async env, but we might simply sleep
            import requests
            requests.get(f"{base_url}/models", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        proc.terminate()
        pytest.fail("Could not start mock server")

    yield base_url
    
    # Cleanup
    proc.terminate()
    proc.join()


# --- The Test ---

@pytest.mark.asyncio
async def test_benchy_integration(mock_server_url):
    """
    Runs llama-benchy against the mock server and verifies:
    1. Throughput (t/s) is near 50 (generation speed emulation)
    2. Prompt processing speed is near 1000 t/s
    3. Prefix caching logic reduces latency for the cached/depth run.
    """
    
    # Command to run llama-benchy
    # We use subprocess to run the CLI tool exactly as a user would
    # Using JSON output format for reliable parsing
    cmd = [
        sys.executable, "-m", "llama_benchy",
        "--base-url", mock_server_url,
        "--model", "test-model",
        "--depth", "0", "4096",
        "--format", "json"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        pytest.fail("llama-benchy command failed")

    # Parse JSON output from stdout
    try:
        stdout = result.stdout
        json_start = stdout.find('{')
        if json_start == -1:
             json_start = stdout.find('[')
        
        if json_start == -1:
             print("Output:\n", stdout)
             pytest.fail("Could not find JSON output")

        # Extract until the last matching brace
        if stdout[json_start] == '{':
            json_end = stdout.rfind('}') + 1
        else:
            json_end = stdout.rfind(']') + 1
            
        json_str = stdout[json_start:json_end]
        data = json.loads(json_str)
        # print("Parsed JSON Data:", json.dumps(data, indent=2))
        
    except json.JSONDecodeError as e:
        print("Output:\n", result.stdout)
        pytest.fail(f"Failed to parse JSON output: {e}")

    # Handle object wrapper if present
    if isinstance(data, dict) and "benchmarks" in data:
        benchmarks = data["benchmarks"]
    elif isinstance(data, list):
        benchmarks = data
    else:
        benchmarks = [data]

    # Helper to find benchmark result
    def find_benchmark(depth, prompt_len):
        return next((
            b for b in benchmarks 
            if b.get("context_size") == depth and b.get("prompt_size") == prompt_len
        ), None)

    # 1. Baseline (Depth 0, PP 2048)
    baseline = find_benchmark(0, 2048)
    assert baseline, "Missing baseline result (depth 0, pp 2048)"

    # Verify Generation Speed (Target ~50 t/s)
    # The JSON structure has tg_throughput -> mean
    gen_speed = baseline["tg_throughput"]["mean"]
    print(f"Generation Speed: {gen_speed} t/s")
    assert 45 < gen_speed < 55, f"Generation speed {gen_speed} outside 45-55 t/s range"

    # Verify Prompt Processing Speed (Target ~1000 t/s)
    pp_speed = baseline["pp_throughput"]["mean"]
    print(f"PP Speed: {pp_speed} t/s")
    assert 900 < pp_speed < 1100, f"Baseline PP speed {pp_speed} outside 900-1100 t/s range"

    # 2. Cached (Depth 4096, PP 2048)
    cached = find_benchmark(4096, 2048)
    assert cached, "Missing cached result (depth 4096, pp 2048)"

    # Verify Cached Speed (Target ~3000 t/s)
    # The caching logic: delay is based on (total_tokens - cached_tokens) + overhead
    cached_speed = cached["pp_throughput"]["mean"]
    print(f"Cached Speed: {cached_speed} t/s")
    
    assert cached_speed > 2500, f"Cached speed {cached_speed} is not significantly boosted (>2500)"
    assert cached_speed < 3500, f"Cached speed {cached_speed} seems too high (>3500)"

