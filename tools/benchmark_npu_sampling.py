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

import argparse
import asyncio
import hashlib
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

import aiohttp
from npu_sampling_runtime_switch import RuntimeSwitch

_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACTS = _ROOT / "local/npu_sampling"
_BINARY = _ROOT / "build" / f"lib.{sysconfig.get_platform()}-{sys.implementation.cache_tag}" / "xllm/xllm"
_URL = "http://127.0.0.1:18674"


def _emit(output: dict) -> None:
    print(json.dumps({"timestamp_ns": time.time_ns(), **output}), flush=True)


def _require(condition: bool, message: object) -> None:
    if not condition:
        raise RuntimeError(str(message))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


async def _request(
    session: aiohttp.ClientSession,
    index: int,
    mode: str,
    tokens: int,
    speculative: bool,
) -> dict:
    body = {
        "model": "npu-sampling-ablation",
        "prompt": "Explain in detail how a computer executes a program and manages its memory.",
        "max_tokens": tokens,
        "temperature": 0 if mode == "greedy" or (mode == "mixed" and index % 4) else 0.7,
        "top_k": 20,
        "top_p": 0.95,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not speculative and mode != "unseeded":
        body["seed"] = 100 + index
    if mode in ("logprobs", "top5"):
        body["logprobs"] = 5 if mode == "top5" else 0
    started = time.perf_counter()
    arrivals = []
    pieces = []
    reported_logprobs = []
    usage = None
    async with session.post(_URL + "/v1/completions", json=body) as response:
        if response.status != 200:
            raise RuntimeError(await response.text())
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            for choice in event.get("choices", []):
                if choice.get("logprobs"):
                    reported_logprobs.append(choice["logprobs"])
                if choice.get("text"):
                    arrivals.append(time.perf_counter())
                    pieces.append(choice["text"])
            if event.get("usage"):
                usage = event["usage"]
    ended = time.perf_counter()
    _require(bool(usage) and usage["completion_tokens"] == tokens, usage)
    _require(bool(arrivals), pieces)
    text = "".join(pieces)
    return {
        "latency_s": ended - started,
        "ttft_s": arrivals[0] - started,
        "tpot_s": (arrivals[-1] - arrivals[0]) / (tokens - 1),
        "stream_chunks": len(arrivals),
        "usage": usage,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
        "reported_logprobs": reported_logprobs if mode in ("logprobs", "top5") else None,
    }


async def _wait_ready(session: aiohttp.ClientSession, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        _require(process.poll() is None, process.returncode)
        try:
            async with session.get(_URL + "/health") as response:
                if response.status == 200:
                    return
        except aiohttp.ClientConnectionError:
            pass
        await asyncio.sleep(2)
    raise TimeoutError("Server startup timed out")


async def _wait_idle(session: aiohttp.ClientSession) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        async with session.get(_URL + "/brpc_metrics") as response:
            if response.status != 200:
                raise RuntimeError(await response.text())
            metrics = await response.text()
        values = {}
        for line in metrics.splitlines():
            if line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) == 2 and fields[0].split("{")[0] in (
                "num_running_requests",
                "num_pending_requests",
            ):
                name = fields[0].split("{")[0]
                values[name] = values.get(name, 0) + float(fields[1])
        _require(len(values) == 2, metrics[:2000])
        if not any(values.values()):
            await asyncio.sleep(0.1)
            return
        await asyncio.sleep(0.05)
    raise TimeoutError("Requests did not drain before config switch")


async def _speculative_metrics(session: aiohttp.ClientSession) -> dict[str, float]:
    async with session.get(_URL + "/brpc_metrics") as response:
        if response.status != 200:
            raise RuntimeError(await response.text())
        metrics = await response.text()
    return {
        fields[0]: float(fields[1])
        for line in metrics.splitlines()
        if line.startswith("speculative_") and len(fields := line.split()) == 2
    }


async def _run(args: argparse.Namespace) -> None:
    global _BINARY, _ARTIFACTS, _URL
    _BINARY = args.binary.resolve()
    _ARTIFACTS = args.artifacts.resolve()
    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    _URL = f"http://127.0.0.1:{args.port}"
    ranks = len(args.devices.split(","))
    args.target_model = args.target_model.resolve()
    if args.draft_model is not None:
        args.draft_model = args.draft_model.resolve()
    speculative = args.kind == "dflash"
    command = [
        sys.executable,
        str(_ROOT / "xllm/launch_server.py"),
        "serve",
        "--binary-path",
        str(_BINARY),
        "--no-log-files",
        "--nnodes",
        str(ranks),
        "--port",
        str(args.port),
        f"--master_node_addr=127.0.0.1:{args.master_port}",
        "--host=127.0.0.1",
        f"--model={args.target_model}",
        "--model_id=npu-sampling-ablation",
        "--model_impl=python",
        "--npu_kernel_backend=TORCH",
        "--backend=llm",
        "--enable_prefix_cache=false",
        "--enable_chunked_prefill=true",
        "--enable_schedule_overlap=false",
        f"--enable_task_pipeline={str(args.task_pipeline).lower()}",
        "--enable_prefill_piecewise_graph=false",
        "--enable_flashcomm1=false",
        "--enable_graph=true",
        "--python_graph_backend=aclgraph",
        f"--acl_graph_decode_batch_size_limit={32 if speculative else 128}",
        f"--max_seqs_per_batch={32 if speculative else 128}",
        "--max_tokens_per_batch=1024",
        "--max_tokens_for_graph_mode=256",
        "--max_memory_utilization=.8",
        "--max_cache_size=17179869184",
        f"--max_linear_state_cache_slots={32 if speculative else 128}",
        "--random_seed=42",
        "--enable_online_profile=true",
        "--profile_backend=ascend",
        f"--profile_dir={_ARTIFACTS / ('profile_' + args.kind)}",
    ]
    if speculative:
        command += [
            f"--draft_model={args.draft_model}",
            "--speculative_algorithm=DFlash2",
            "--enable_probabilistic_draft=true",
            "--num_speculative_tokens=7",
        ]
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = args.devices
    environment["HCCL_IF_BASE_PORT"] = str(args.hccl_base_port)
    environment["OMP_NUM_THREADS"] = "8"
    # The online Ascend backend uses aclprof APIs. CANN 9 rejects these APIs
    # when dynamic attachment mode is enabled in the service environment.
    environment.pop("PROFILING_MODE", None)
    environment["PYTHONPATH"] = str(_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    log_path = _ARTIFACTS / f"same_process_{args.kind}_server.log"
    (_ARTIFACTS / "benchmark_source.py").write_bytes(Path(__file__).read_bytes())
    (_ARTIFACTS / "switch_source.py").write_bytes((_ROOT / "tools/npu_sampling_runtime_switch.py").read_bytes())
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, cwd=_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            _emit(
                {
                    "event": "launch",
                    "kind": args.kind,
                    "pid": process.pid,
                    "command": command,
                    "binary_sha256": _file_sha256(_BINARY),
                    "benchmark_script_sha256": _file_sha256(Path(__file__)),
                    "switch_script_sha256": _file_sha256(_ROOT / "tools/npu_sampling_runtime_switch.py"),
                    "log": str(log_path),
                }
            )
            controller = RuntimeSwitch(process.pid, _BINARY, _BINARY.parent / "npu_sampling_benchmark", ranks)
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0),
                timeout=aiohttp.ClientTimeout(total=900),
            ) as session:
                await _wait_ready(session, process)
                masks = args.masks or ([0, 8, 16, 24, 32, 40, 48, 56, 7, 63] if speculative else [0, 1, 2, 3, 5, 7])
                cases = [
                    (mode, batch)
                    for mode in ("random", "mixed")
                    for batch in (args.batch_sizes or ([1, 8, 32] if speculative else [1, 32, 128]))
                ]
                if args.smoke:
                    cases = [("random", 1)]
                elif not speculative:
                    cases += [
                        (mode, batch)
                        for mode, default_batch in [("unseeded", 32), ("logprobs", 128), ("top5", 128)]
                        for batch in (args.batch_sizes or [default_batch])
                    ]
                if not args.smoke:
                    cases += [("greedy", 8 if speculative else 32)]
                if args.modes:
                    cases = [case for case in cases if case[0] in args.modes]
                if args.profile_only:
                    cases = []
                for mode, batch in cases:
                    current_masks = masks
                    if mode in ("logprobs", "top5") and not args.masks:
                        current_masks = [0, 1, 8, 9, 5, 13]
                    if mode == "greedy" and not args.masks:
                        current_masks = [0, 56, 63] if speculative else [0, 7]
                    for repeat in range(args.repeats):
                        order = list(current_masks)
                        random.Random(20261028 + repeat).shuffle(order)
                        for mask in order:
                            await _wait_idle(session)
                            readback = controller.switch(mask)
                            _emit(
                                {
                                    "event": "switch",
                                    "kind": args.kind,
                                    "mask": mask,
                                    "mode": mode,
                                    "batch": batch,
                                    "repeat": repeat,
                                    "workers": readback,
                                }
                            )
                            await asyncio.gather(*(_request(session, i, mode, 8, speculative) for i in range(batch)))
                            await _wait_idle(session)
                            metrics_before = await _speculative_metrics(session) if speculative else None
                            started = time.perf_counter()
                            results = await asyncio.gather(
                                *(_request(session, i, mode, args.tokens, speculative) for i in range(batch))
                            )
                            elapsed = time.perf_counter() - started
                            if speculative:
                                await _wait_idle(session)
                            metrics_after = await _speculative_metrics(session) if speculative else None
                            _emit(
                                {
                                    "event": "measurement",
                                    "kind": args.kind,
                                    "mask": mask,
                                    "mode": mode,
                                    "batch": batch,
                                    "repeat": repeat,
                                    "elapsed_s": elapsed,
                                    "tokens_per_s": batch * args.tokens / elapsed,
                                    "median_tpot_s": statistics.median(r["tpot_s"] for r in results),
                                    "speculative_metrics_before": metrics_before,
                                    "speculative_metrics_after": metrics_after,
                                    "requests": results,
                                }
                            )
                if args.profile:
                    profile_cases = (
                        [(0, "random", 32), (56, "random", 32), (63, "random", 32)]
                        if speculative
                        else [
                            (0, "random", 32),
                            (7, "random", 32),
                            (0, "mixed", 32),
                            (7, "mixed", 32),
                            (5, "logprobs", 128),
                            (13, "logprobs", 128),
                        ]
                    )
                    for mask, profile_mode, profile_batch in profile_cases:
                        await _wait_idle(session)
                        readback = controller.switch(mask)
                        await asyncio.gather(
                            *(_request(session, i, profile_mode, 8, speculative) for i in range(profile_batch))
                        )
                        await _wait_idle(session)
                        async with session.post(_URL + "/start_profile") as response:
                            if response.status != 200:
                                raise RuntimeError(await response.text())
                            start_response = await response.text()
                        _emit(
                            {
                                "event": "profile_start",
                                "mask": mask,
                                "mode": profile_mode,
                                "batch": profile_batch,
                                "workers": readback,
                                "response": start_response,
                            }
                        )
                        results = await asyncio.gather(
                            *(_request(session, i, profile_mode, 32, speculative) for i in range(profile_batch))
                        )
                        await _wait_idle(session)
                        async with session.post(_URL + "/stop_profile") as response:
                            if response.status != 200:
                                raise RuntimeError(await response.text())
                            stop_response = await response.text()
                        _emit({"event": "profile_stop", "mask": mask, "response": stop_response, "requests": results})
                await _wait_idle(session)
                readback = controller.switch(0)
                _emit({"event": "restore", "workers": readback})
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    await asyncio.to_thread(process.wait, timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    await asyncio.to_thread(process.wait, timeout=30)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["ordinary", "dflash"])
    parser.add_argument("--target-model", required=True, type=Path)
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--binary", type=Path, default=_BINARY)
    parser.add_argument("--artifacts", type=Path, default=_ARTIFACTS)
    parser.add_argument("--port", type=int, default=18674)
    parser.add_argument("--master-port", type=int, default=23474)
    parser.add_argument("--hccl-base-port", type=int, default=53480)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--task-pipeline", action="store_true")
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--modes", nargs="+", choices=["random", "mixed", "unseeded", "logprobs", "top5", "greedy"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--masks", type=int, nargs="+")
    args = parser.parse_args()
    if args.kind == "dflash" and args.draft_model is None:
        parser.error("--draft-model is required for dflash")
    if args.tokens < 2 or args.repeats < 1 or (args.batch_sizes and min(args.batch_sizes) < 1):
        parser.error("tokens must be >= 2; repeats and batch sizes must be positive")
    if args.profile_only:
        args.profile = True
    asyncio.run(_run(args))


if __name__ == "__main__":
    _main()
