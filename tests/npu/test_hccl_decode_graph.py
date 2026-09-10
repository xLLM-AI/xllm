# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Two-rank transport probe, isolated from tests/python runtime stubs.

The Gloo control validates payload order and the subprocess harness, NOT graph
support. HCCL tests require XLLM_TEST_HCCL_DEVICES=<logical-id>,<logical-id> after
the operator has verified both devices are free. No native xLLM binary is used.
Set XLLM_TEST_HCCL_ARTIFACT_DIR to retain unique per-attempt rank logs/results.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import pytest

from scripts.logger import logger


def _run_probe(tmp_path: Path, mode: str, dtype: str, devices: tuple[int, int] | None = None) -> None:
    artifact_parent = Path(os.environ.get("XLLM_TEST_HCCL_ARTIFACT_DIR", str(tmp_path))).resolve()
    artifact_parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(tempfile.mkdtemp(prefix=f"{mode}-{dtype}-", dir=artifact_parent))
    worker = Path(__file__).resolve().with_name("hccl_decode_graph_worker.py")
    repo = Path(__file__).resolve().parents[2]
    source_sha256 = {}
    for source in (Path(__file__).resolve(), worker):
        shutil.copyfile(source, artifact_dir / source.name)
        source_sha256[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    commands: list[list[str]] = []
    processes: list[subprocess.Popen[bytes]] = []
    timed_out = False
    logger.info("Transport probe artifacts: %s", artifact_dir)
    with ExitStack() as stack:
        try:
            for rank in range(2):
                command = [
                    sys.executable,
                    str(worker),
                    "--mode",
                    mode,
                    "--dtype",
                    dtype,
                    "--rank",
                    str(rank),
                    "--artifact-dir",
                    str(artifact_dir),
                ]
                if devices is not None:
                    command.extend(("--device", str(devices[rank])))
                commands.append(command)
                log = stack.enter_context((artifact_dir / f"rank-{rank}.log").open("xb"))
                processes.append(
                    subprocess.Popen(
                        command,
                        cwd=repo,
                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
            deadline = time.monotonic() + 120
            while True:
                codes = [process.poll() for process in processes]
                if all(code is not None for code in codes) or any(code not in (None, 0) for code in codes):
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                time.sleep(0.1)
        finally:
            # Only terminate these two probe children, never unrelated jobs.
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            summary = {
                "mode": mode,
                "dtype": dtype,
                "devices": devices,
                "commands": commands,
                "returncodes": [process.returncode for process in processes],
                "timed_out": timed_out,
                "source_sha256": source_sha256,
            }
            (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    assert not timed_out, f"transport probe exceeded 120s; inspect {artifact_dir}"
    assert summary["returncodes"] == [0, 0], f"rank failures: {summary['returncodes']}; inspect {artifact_dir}"
    for rank in range(2):
        result = json.loads((artifact_dir / f"rank-{rank}.json").read_text(encoding="utf-8"))
        assert result["rank"] == rank
        assert result["mode"] == mode
        assert result["dtype"] == dtype
        assert result["worker_sha256"] == source_sha256[worker.name]
        assert result["checked_values"] == [2, 5, 9] * 3
        assert result["graph_replays"] == (9 if mode == "hccl-graph" else 0)


def test_two_rank_gloo_payload_control(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo is required for the CPU payload control")
    _run_probe(tmp_path, "gloo-eager", "float32")


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_two_rank_hccl_graph_replays_changing_payloads(tmp_path: Path, dtype: str) -> None:
    device_list = os.environ.get("XLLM_TEST_HCCL_DEVICES")
    if device_list is None:
        pytest.skip("set XLLM_TEST_HCCL_DEVICES to two verified-free logical NPU ids")
    parts = device_list.split(",")
    assert len(parts) == 2 and all(part.strip().isdecimal() for part in parts), "expected two nonnegative NPU ids"
    devices = (int(parts[0]), int(parts[1]))
    assert devices[0] != devices[1], "the two HCCL ranks must use distinct devices"
    # An explicit opt-in must fail, not silently skip, if torch_npu is missing.
    _run_probe(tmp_path, "hccl-graph", dtype, devices)
