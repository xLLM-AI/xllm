# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.
"""Measure fixed-length requests; reject runs with incorrect token counts."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from urllib.request import Request, urlopen


def _request(url: str, payload: dict, barrier: Barrier | None = None) -> dict:
    encoded = json.dumps(payload).encode()
    request = Request(url + "/v1/chat/completions", encoded, {"Content-Type": "application/json"})
    if barrier is not None:
        barrier.wait(timeout=60)
    start = time.perf_counter()
    first = None
    usage = None
    finished = None
    output = []
    with urlopen(request, timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                break
            event = json.loads(raw)
            if event.get("error"):
                raise RuntimeError(event["error"])
            if event.get("usage") is not None:
                usage = event["usage"]
            for choice in event.get("choices", []):
                value = choice.get("delta", {}).get("content")
                if value:
                    if first is None:
                        first = time.perf_counter()
                    output.append(value)
                finished = choice.get("finish_reason") or finished
    end = time.perf_counter()
    if usage is None or finished is None or first is None:
        raise RuntimeError("Incomplete streaming response")
    ttft = first - start
    elapsed = end - start
    return {
        "usage": usage,
        "finish_reason": finished,
        "ttft_seconds": ttft,
        "elapsed_seconds": elapsed,
        "tpot_seconds": (elapsed - ttft) / max(1, usage["completion_tokens"] - 1),
        "request_sha256": hashlib.sha256(encoded).hexdigest(),
        "output_sha256": hashlib.sha256("".join(output).encode()).hexdigest(),
    }


def _control_phase(root: Path, phase: str, state: str) -> None:
    """Fence a wave with the offline memory sampler before sending requests."""
    temporary = root / "client_control.tmp"
    temporary.write_text(json.dumps({"phase": phase, "state": state}))
    temporary.replace(root / "control.json")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        path = root / "memory.json"
        if path.exists():
            row = json.loads(path.read_text()).get("phases", {}).get(phase, {})
            if row.get("armed" if state == "arm" else "completed"):
                return
        time.sleep(0.05)
    raise TimeoutError(f"Memory monitor did not acknowledge {phase} {state}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18001")
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--instrumentation-dir", "--control-dir", type=Path)
    args = parser.parse_args()
    content = []
    if args.image:
        uri = "data:image/png;base64," + base64.b64encode(args.image.read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": uri}})
    content.append({"type": "text", "text": ""})
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 1,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.model == "Qwen3-8B":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    # Calibrate with the server's actual tokenizer and vision preprocessing.
    # These requests are excluded from all timings below.
    filler = 1024
    for _ in range(8):
        content[-1]["text"] = "Describe the scene and continue in detail." + " test" * filler
        probe = _request(args.url, payload)
        difference = 2048 - probe["usage"]["prompt_tokens"]
        if difference == 0:
            break
        filler += difference
        if filler < 0:
            raise ValueError("Image/template exceeds the 2048-token budget")
    else:
        raise ValueError("Cannot construct an exact 2048-token input")
    payload["max_tokens"] = 2048
    with args.output.open("x", encoding="utf-8", buffering=1) as target:
        target.write(json.dumps({"kind": "configuration", "payload": payload, "rounds": args.rounds}) + "\n")
        for concurrency in args.concurrency:
            # One full-length warmup wave per batch size.
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for iteration in range(args.rounds + 1):
                    phase = f"c{concurrency}_i{iteration}"
                    if args.instrumentation_dir is not None:
                        _control_phase(args.instrumentation_dir, phase, "arm")
                    barrier = Barrier(concurrency)
                    start = time.perf_counter()
                    calls = [pool.submit(_request, args.url, payload, barrier) for _ in range(concurrency)]
                    results = [call.result() for call in calls]
                    elapsed = time.perf_counter() - start
                    if args.instrumentation_dir is not None:
                        _control_phase(args.instrumentation_dir, phase, "finish")
                    for result in results:
                        if (result["usage"]["prompt_tokens"], result["usage"]["completion_tokens"]) != (2048, 2048):
                            raise ValueError(f"Token-count mismatch: {result['usage']}")
                    row = {
                        "kind": "warmup" if iteration == 0 else "measurement",
                        "concurrency": concurrency,
                        "iteration": iteration,
                        "wave_seconds": elapsed,
                        "output_tokens_per_second": concurrency * 2048 / elapsed,
                        "ttft_median_seconds": statistics.median(r["ttft_seconds"] for r in results),
                        "tpot_median_seconds": statistics.median(r["tpot_seconds"] for r in results),
                        "requests": results,
                    }
                    target.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)
        target.write(json.dumps({"kind": "completed"}) + "\n")


if __name__ == "__main__":
    main()
