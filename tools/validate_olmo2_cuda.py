# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate an OLMo2 FP16/BF16 eager HTTP service and optional local baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from tokenizers import Tokenizer
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger  # noqa: E402


def _save(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _validate_stream(result: dict[str, Any], events: list[dict[str, Any]], *, include_usage: bool) -> None:
    # Matching text alone can hide an early EOF or an in-band server error.
    for event in events:
        line = event["line"]
        if line.startswith("data: ") and line != "data: [DONE]" and "error" in json.loads(line[6:]):
            raise ValueError("stream returned a server error")
    if not events or events[-1]["line"] != "data: [DONE]":
        raise ValueError("stream ended without [DONE]")
    if result["finish_reason"] is None:
        raise ValueError("stream ended without a finish reason")
    if include_usage and result["usage"] is None:
        raise ValueError("stream ended without the requested usage")


def _validate_loaded_configuration(loaded: list[str], tp: int, dtype: str) -> None:
    """Require one real checkpoint load per expected rank and compute dtype."""
    observed = []
    for line in loaded:
        rank = re.search(r"\btp_rank=(\d+)\b", line)
        size = re.search(r"\btp_size=(\d+)\b", line)
        compute_dtype = re.search(r"\bdtype=torch\.(\w+)\b", line)
        if rank is None or size is None or compute_dtype is None:
            raise ValueError(f"incomplete OLMo2 loaded configuration: {line}")
        observed.append((int(rank[1]), int(size[1]), compute_dtype[1]))
    expected = {(rank, tp, dtype) for rank in range(tp)}
    if len(observed) != tp or set(observed) != expected:
        raise ValueError(f"OLMo2 loaded topology/dtype differs: expected={expected}, observed={observed}")


def main() -> None:
    if not __debug__:
        raise RuntimeError("OLMo2 validation requires assertions enabled; do not use python -O")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, choices=[1, 2], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="Expected compute dtype in the model config.",
    )
    parser.add_argument("--devices", help="Comma-separated physical GPU IDs, one per rank.")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--max-memory-utilization", type=float, default=0.4)
    parser.add_argument("--max-cache-size", type=int, default=1073741824, help="Per-rank KV cache cap in bytes.")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--build-label", default="unspecified; record compiler flags separately")
    args = parser.parse_args()
    if not 0 < args.max_memory_utilization <= 1 or args.max_cache_size <= 0:
        parser.error("memory utilization must be in (0, 1] and cache size must be positive")
    model = str(args.model.resolve())
    devices = args.devices.split(",") if args.devices else [str(i) for i in range(args.tp)]
    if len(devices) != args.tp or len(set(devices)) != args.tp:
        parser.error("--devices must contain one distinct device per TP rank")
    out = args.output
    out.mkdir(exist_ok=False, parents=True)
    port = args.port
    model_id = args.model.name
    binary = args.binary.resolve()
    command = [
        str(binary),
        f"--model={model}",
        f"--model_id={model_id}",
        "--model_impl=python",
        "--host=127.0.0.1",
        f"--port={port}",
        f"--master_node_addr=127.0.0.1:{port + 1000}",
        f"--max_memory_utilization={args.max_memory_utilization}",
        f"--max_cache_size={args.max_cache_size}",
        "--max_tokens_per_batch=256",
        "--max_seqs_per_batch=4",
        "--enable_graph=false",
        "--python_graph_backend=off",
        "--disable_graph_warmup=true",
        "--enable_prefix_cache=false",
        "--enable_in_batch_prefix_cache=false",
    ]
    overrides = dict(
        CUDA_VISIBLE_DEVICES=",".join(devices),
        PYTHONPATH=str(ROOT),
        XLLM_PYTHON_MODEL_PATH=str(ROOT),
        XLLM_PYTHON_EXECUTABLE=sys.executable,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        MAX_JOBS="4",
    )
    env = os.environ.copy()
    env.update(overrides)
    _save(
        out / "launch.json",
        dict(
            command=command,
            env_overrides=overrides,
            binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            build=args.build_label,
            expected_dtype=args.dtype,
            # Leave architecture detection to FlashInfer unless the caller set it.
            flashinfer_cuda_arch_list=env.get("FLASHINFER_CUDA_ARCH_LIST"),
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    raw_tokenizer = Tokenizer.from_file(model + "/tokenizer.json")
    checks = []

    def _request(name: str, payload: dict | None, chat: bool = False, expected: tuple = (200,)) -> dict:
        url = f"http://127.0.0.1:{port}/v1/" + (
            "models" if payload is None else "chat/completions" if chat else "completions"
        )
        start = time.perf_counter()
        with requests.Session() as session:
            session.trust_env = False
            response = (
                session.get(url, timeout=3)
                if payload is None
                else session.post(url, json=payload, timeout=300, stream=bool(payload.get("stream")))
            )
            events, usage, text, finish = [], None, "", None
            if payload and payload.get("stream") and response.status_code == 200:
                for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                    if not line:
                        continue
                    elapsed = time.perf_counter() - start
                    events.append(dict(time=elapsed, line=line))
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    item = json.loads(line[6:])
                    if item.get("usage"):
                        usage = item["usage"]
                    for choice in item.get("choices", []):
                        text += choice.get("delta", {}).get("content") or choice.get("text") or ""
                        finish = choice.get("finish_reason") or finish
                result = dict(text=text, finish_reason=finish, usage=usage)
            else:
                result = response.json()
            record = dict(
                request=payload,
                status=response.status_code,
                result=result,
                elapsed=time.perf_counter() - start,
                events=events,
            )
            _save(out / f"{name}.json", record)
            assert response.status_code in expected, (name, record)
            if payload and payload.get("stream") and response.status_code == 200:
                _validate_stream(
                    result, events, include_usage=bool(payload.get("stream_options", {}).get("include_usage"))
                )
            return record

    def _base(prompt: object, n: int = 12) -> dict:
        return dict(model=model_id, prompt=prompt, temperature=0, max_tokens=n, ignore_eos=True)

    def _ok(name: str) -> None:
        checks.append(name)
        _save(out / "checks-in-progress.json", checks)

    processes = []
    extra_logs = []
    with (out / "server.log").open("w") as log:
        try:
            for rank in range(args.tp):
                rank_env = env.copy()
                rank_env["CUDA_VISIBLE_DEVICES"] = devices[rank]
                rank_command = [arg if not arg.startswith("--port=") else f"--port={port + rank}" for arg in command]
                rank_command += [f"--nnodes={args.tp}", f"--node_rank={rank}"]
                rank_log = log if rank == 0 else (out / f"server-rank{rank}.log").open("w")
                if rank:
                    extra_logs.append(rank_log)
                p = subprocess.Popen(
                    rank_command,
                    cwd=ROOT,
                    env=rank_env,
                    stdout=rank_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                processes.append(p)
                _save(
                    out / f"launch-rank{rank}.json",
                    dict(
                        command=rank_command,
                        env_overrides={**overrides, "CUDA_VISIBLE_DEVICES": rank_env["CUDA_VISIBLE_DEVICES"]},
                        pid=p.pid,
                    ),
                )
            _save(out / "pid.json", dict(pids=[p.pid for p in processes]))
            deadline = time.monotonic() + 600
            while True:
                assert all(p.poll() is None for p in processes), f"server exited: {[p.returncode for p in processes]}"
                try:
                    models = _request("models", None)
                    break
                except requests.ConnectionError:
                    assert time.monotonic() < deadline, "readiness deadline"
                    time.sleep(2)
            assert model_id in [m["id"] for m in models["result"]["data"]]
            loaded = [
                line
                for file in out.glob("server*.log")
                for line in file.read_text().splitlines()
                if "Loaded 179 OLMo2 checkpoint tensors" in line
            ]
            _validate_loaded_configuration(loaded, args.tp, args.dtype)
            _save(out / "bridge.json", loaded)
            _ok("models-and-real-configuration")
            ids = tokenizer.encode("Explain in one sentence what an inference engine does.", add_special_tokens=False)
            regular = _request("completion", _base(ids))["result"]
            assert regular["choices"][0]["finish_reason"] == "length" and regular["usage"]["completion_tokens"] == 12
            streamed = _request(
                "completion-stream", {**_base(ids), "stream": True, "stream_options": {"include_usage": True}}
            )
            assert streamed["result"]["text"] == regular["choices"][0]["text"], streamed
            assert streamed["result"]["finish_reason"] == "length" and streamed["events"][-1]["line"] == "data: [DONE]"
            assert streamed["result"]["usage"]["completion_tokens"] == 12
            _ok("completion-stream-equivalence-length-usage")
            messages = [
                {"role": "system", "content": "Give a brief answer."},
                {"role": "user", "content": "什么是推理引擎？"},
            ]
            chat = dict(model=model_id, messages=messages, temperature=0, max_tokens=12, ignore_eos=True)
            result = _request("chat", chat, True)["result"]
            sresult = _request(
                "chat-stream", {**chat, "stream": True, "stream_options": {"include_usage": True}}, True
            )["result"]
            assert result["choices"][0]["message"]["content"] == sresult["text"]
            assert sresult["finish_reason"] == result["choices"][0]["finish_reason"] == "length"
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                assert sresult["usage"][field] == result["usage"][field]
            assert sresult["usage"]["completion_tokens"] == 12
            rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            # The official serialized tokenizer is also used by the C++ server.
            # Transformers 5's GPT2 reconstruction changes newline pretokenization.
            chatids = raw_tokenizer.encode(rendered, add_special_tokens=False).ids
            explicit = _request("chat-template-explicit", _base(chatids))["result"]
            assert explicit["usage"]["prompt_tokens"] == result["usage"]["prompt_tokens"] == len(chatids)
            assert explicit["choices"][0]["text"] == result["choices"][0]["message"]["content"]
            _save(out / "chat-template-ids.json", chatids)
            _ok("chat-template-and-chat-stream-equivalence")

            # Exercise the CUDA random sampler; greedy requests never load it.
            sampled_payload = {**_base(ids), "temperature": 0.7}
            sampled = _request("completion-sampled", sampled_payload)["result"]
            assert sampled["choices"][0]["text"]
            assert sampled["choices"][0]["finish_reason"] == "length"
            assert sampled["usage"]["completion_tokens"] == 12
            sampled_stream = _request(
                "completion-sampled-stream",
                {**sampled_payload, "stream": True, "stream_options": {"include_usage": True}},
            )
            assert sampled_stream["result"]["text"]
            assert sampled_stream["result"]["finish_reason"] == "length"
            assert sampled_stream["result"]["usage"]["completion_tokens"] == 12
            _ok("random-sampling-completion-stream-length-usage")

            eos = tokenizer.eos_token_id
            for ignore, expected_length, reason in [(False, 1, "stop"), (True, 4, "length")]:
                r = _request(
                    f"eos-ignore-{ignore}", {**_base(ids, 4), "allowed_token_ids": [eos], "ignore_eos": ignore}
                )["result"]
                assert (
                    r["choices"][0]["finish_reason"] == reason and r["usage"]["completion_tokens"] == expected_length
                ), r
            token = tokenizer.encode("Hello", add_special_tokens=False)[0]
            for streaming in (False, True):
                p = {**_base(ids, 8), "allowed_token_ids": [token], "stop_token_ids": [token], "stream": streaming}
                r = _request(f"stop-token-{streaming}", p)["result"]
                assert (r["finish_reason"] if streaming else r["choices"][0]["finish_reason"]) == "stop", r
                p = {
                    **_base(ids, 8),
                    "allowed_token_ids": [token],
                    "stop": tokenizer.decode([token]),
                    "stream": streaming,
                }
                r = _request(f"stop-string-{streaming}", p)["result"]
                assert (r["finish_reason"] if streaming else r["choices"][0]["finish_reason"]) == "stop", r
                assert (r["text"] if streaming else r["choices"][0]["text"]) == "", r
            _ok("eos-ignore-eos-stop-token-stop-string")
            for name, payload, chat_mode in [
                ("unknown-model", {**_base(ids), "model": "missing"}, False),
                ("negative-length", {**_base(ids), "max_tokens": -1}, False),
                ("invalid-token", _base([100352]), False),
                ("over-context", _base([42] * 4097), False),
                ("empty-chat", {**chat, "messages": []}, True),
                ("invalid-stream-option", {**_base(ids), "stream_options": {"include_usage": "bad"}}, False),
            ]:
                _request("error-" + name, payload, chat_mode, (400, 404))
            _ok("invalid-input-rejection")
            histories = [[42 + (i % 100) for i in range(n)] for n in (63, 127, 129, 1025)]
            serial = [_request(f"mixed-serial-{i}", _base(h, 8))["result"] for i, h in enumerate(histories)]
            with ThreadPoolExecutor(4) as pool:
                parallel = list(
                    pool.map(
                        lambda item: _request(f"mixed-concurrent-{item[0]}", _base(item[1], 8))["result"],
                        enumerate(histories),
                    )
                )
            for s, p in zip(serial, parallel, strict=True):
                assert s["choices"][0]["text"] == p["choices"][0]["text"], (s, p)
                assert s["usage"] == p["usage"]
            _ok("mixed-length-concurrent-isolation")
            # Measurements are HTTP end-to-end, with graph off and prefix cache policy logged.
            measurements = []
            for length in (64, 512, 2048) if args.benchmark else ():
                for concurrency in (1, 4):
                    histories = [[42 + ((i + j * 13) % 100) for i in range(length)] for j in range(concurrency)]
                    for repeat in range(-1, 3):
                        start = time.perf_counter()
                        with ThreadPoolExecutor(concurrency) as pool:
                            records = list(
                                pool.map(
                                    lambda item: _request(
                                        item[0],
                                        {
                                            **_base(item[1], 32),
                                            "stream": True,
                                            "stream_options": {"include_usage": True},
                                        },
                                    ),
                                    [
                                        (f"bench-{length}-c{concurrency}-r{repeat}-{index}", history)
                                        for index, history in enumerate(histories)
                                    ],
                                )
                            )
                        wall = time.perf_counter() - start
                        rows = []
                        for record in records:
                            assert record["result"]["usage"]["completion_tokens"] == 32, record
                            assert record["result"]["finish_reason"] == "length"
                            content_times = []
                            for event in record["events"]:
                                if event["line"] == "data: [DONE]":
                                    continue
                                data = json.loads(event["line"][6:])
                                if any(c.get("text") for c in data.get("choices", [])):
                                    content_times.append(event["time"])
                            assert content_times
                            rows.append(
                                dict(
                                    ttft_s=content_times[0],
                                    tpot_s=(record["elapsed"] - content_times[0]) / 31,
                                    latency_s=record["elapsed"],
                                    content_events=len(content_times),
                                    output_tokens=32,
                                )
                            )
                        if repeat >= 0:
                            measurements.append(
                                dict(
                                    prompt_tokens=length,
                                    concurrency=concurrency,
                                    repeat=repeat,
                                    wall_s=wall,
                                    output_tokens_per_s=32 * concurrency / wall,
                                    requests=rows,
                                )
                            )
                        _save(out / "performance-in-progress.json", measurements)
                        subprocess.run(
                            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
                            stdout=(out / f"memory-{length}-c{concurrency}-r{repeat}.csv").open("w"),
                            check=True,
                        )
            _save(
                out / "performance.json",
                dict(
                    measurements=measurements,
                    profiling=False,
                    build=args.build_label,
                    prefix_cache=False,
                    max_memory_utilization=args.max_memory_utilization,
                    max_cache_size_bytes=args.max_cache_size,
                    tpot_definition="(request_end - first_nonempty_text) / (completion_tokens - 1)",
                    limitations=[
                        "client-observed effective TPOT, including invisible tokens and transport",
                        "three repetitions only",
                    ],
                ),
            )
            if args.benchmark:
                _ok("repeated-unprofiled-performance")
            _request("health-after-all", None)
        finally:
            for p in processes:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
            for p in processes:
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait(timeout=10)
            for handle in extra_logs:
                handle.close()
            _save(out / "shutdown.json", dict(stopped_own_server=True, exit_codes=[p.returncode for p in processes]))

    # Publish success only after every owned rank has exited cleanly. A late
    # crash or forced kill must not leave a successful acceptance artifact.
    exit_codes = [p.returncode for p in processes]
    if len(processes) != args.tp or any(code != 0 for code in exit_codes):
        raise RuntimeError(f"OLMo2 server ranks did not exit cleanly: {exit_codes}")
    _save(
        out / "result.json",
        dict(
            status="PASS",
            tp=args.tp,
            checks=checks,
            bench_groups=6 if args.benchmark else 0,
            repetitions=3 if args.benchmark else 0,
        ),
    )
    logger.info("OLMo2 service acceptance passed: TP=%s, benchmark=%s", args.tp, args.benchmark)


if __name__ == "__main__":
    main()
