#!/usr/bin/env python3
"""Measure cache reuse for repeated long-prefix chat requests."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sequential chat completions with a stable long prefix."
    )
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--model", default="gemma4-26b-it")
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--prefix-lines", type=int, default=219)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--cache-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send cache_prompt in the request payload.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON result path. Defaults to results/mcp/custom-prefix-cache-<timestamp>.json.",
    )
    return parser.parse_args()


def make_shared_prefix(lines: int) -> str:
    body = "\n".join(
        f"Shared context line {i:04d}: service latency, cache state, and "
        "queueing behavior were sampled under steady load."
        for i in range(1, lines + 1)
    )
    return (
        "You are analyzing a long operational incident report. "
        "Use the shared context exactly and answer briefly.\n\n"
        f"{body}"
    )


def chat_completion(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    output = args.output or (
        Path("results/mcp") / f"custom-prefix-cache-{timestamp}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    shared_prefix = make_shared_prefix(args.prefix_lines)
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    rows: list[dict[str, Any]] = []

    for index in range(1, args.requests + 1):
        user_prompt = (
            f"{shared_prefix}\n\n"
            f"Question {index}: In one sentence, identify whether cache reuse "
            f"should improve request {index} if the shared prefix is retained."
        )
        payload: dict[str, Any] = {
            "model": args.model,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "cache_prompt": args.cache_prompt,
        }

        started = time.perf_counter()
        body = chat_completion(endpoint, payload)
        wall_ms = (time.perf_counter() - started) * 1000
        usage = body.get("usage", {})
        content = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        row = {
            "request": index,
            "wall_ms": wall_ms,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "response_preview": content[:160],
        }
        rows.append(row)
        print(
            f"request {index}: wall_ms={wall_ms:.1f} "
            f"prompt_tokens={row['prompt_tokens']} "
            f"completion_tokens={row['completion_tokens']}"
        )

    result = {
        "model": args.model,
        "base_url": args.base_url,
        "cache_prompt": args.cache_prompt,
        "shared_prefix_lines": args.prefix_lines,
        "requests": rows,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
