# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.
"""Fixed single-image, multi-image, video and mixed-length xLLM requests."""

import argparse
import base64
import hashlib
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


def _uri(name: str, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode((ROOT / "fixtures" / name).read_bytes()).decode()


def _cases() -> dict:
    def image(name: str) -> dict:
        return {"type": "image_url", "image_url": {"url": _uri(name, "image/png")}}

    def text(value: str) -> dict:
        return {"type": "text", "text": value}

    simple = "What color is the circle? Answer with one color word."
    return {
        "single": [image("red_448.png"), text(simple)],
        "large": [image("red_896.png"), text(simple)],
        "multi": [
            image("red_448.png"),
            image("blue_448.png"),
            text("Name the circle colors in image order, using two color words."),
        ],
        "video": [
            {"type": "video_url", "video_url": {"url": _uri("moving_red_circle.mp4", "video/mp4")}},
            text("Does the red circle move left or right? Answer one word."),
        ],
        "long": [image("blue_448.png"), text("Context: " + "A circle is a round shape. " * 300 + "\n" + simple)],
    }


def _request(content: list, out_tokens: int, barrier: Barrier, url: str) -> dict:
    payload = {
        "model": "Qwen3-VL-8B-Instruct",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": out_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    encoded = json.dumps(payload).encode()
    request = Request(url + "/v1/chat/completions", encoded, {"Content-Type": "application/json"})
    barrier.wait(timeout=30)
    start = time.perf_counter()
    first = None
    text, usage, finished = [], None, None
    try:
        with urlopen(request, timeout=240) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(event["error"])
                if event.get("usage") is not None:
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    value = choice.get("delta", {}).get("content")
                    if value:
                        if first is None:
                            first = time.perf_counter()
                        text.append(value)
                    if choice.get("finish_reason") is not None:
                        finished = choice["finish_reason"]
    except HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from error
    end = time.perf_counter()
    if not usage or not finished or first is None:
        raise RuntimeError(f"Incomplete SSE response: usage={usage}, finish={finished}, text={text}")
    return {
        "tt_first_text_ms": 1000 * (first - start),
        "elapsed_ms": 1000 * (end - start),
        "text": "".join(text),
        "usage": usage,
        "finish_reason": finished,
        "payload_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18001")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--case", default="all")
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    cases = _cases()
    groups = {name: [content] for name, content in cases.items()}
    groups["mixed"] = [cases[name] for name in ("single", "multi", "long", "large")]
    if args.case != "all":
        groups = {args.case: groups[args.case]}

    def profile(action: str) -> None:
        with urlopen(Request(args.url + "/" + action, data=b"", method="POST"), timeout=360) as response:
            assert response.status == 200

    with args.output.open("x", encoding="utf-8", buffering=1) as target, ThreadPoolExecutor(max_workers=4) as pool:

        def log(row: dict) -> None:
            target.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

        for name, contents in groups.items():
            for iteration in range((1 if args.capture else args.rounds) + 1):
                if args.capture and iteration == 1:
                    profile("start_profile")
                barrier = Barrier(len(contents))
                start = time.perf_counter()
                calls = [pool.submit(_request, c, 24 if iteration == 0 else 1, barrier, args.url) for c in contents]
                results = [call.result() for call in calls]
                elapsed = time.perf_counter() - start
                if args.capture and iteration == 1:
                    profile("stop_profile")
                row = {
                    "kind": "warmup" if iteration == 0 else "capture" if args.capture else "benchmark",
                    "case": name,
                    "round": iteration,
                    "concurrency": len(contents),
                    "ttft_median_ms": statistics.median(r["tt_first_text_ms"] for r in results),
                    "wave_ms": elapsed * 1000,
                    "requests": results,
                }
                log(row)
        log({"kind": "completed"})


if __name__ == "__main__":
    main()
