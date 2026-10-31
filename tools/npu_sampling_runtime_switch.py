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

"""Debugger-style ablation control for an owned, idle server process tree.

No request may be in flight during switch(). This changes the test-only bitmask
with every worker stopped, and reads it back before resuming. It requires a
BUILD_NPU_SAMPLING_ABLATION build; production builds contain no control symbol.
The caller must be an ancestor of the server for Linux ptrace access checks.
"""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

_BITS = {"A": 1, "B": 2, "C": 4, "D": 8, "E": 16, "G": 32}


def _require(condition: bool, message: object) -> None:
    if not condition:
        raise RuntimeError(str(message))


def _descendants(parent: int) -> set[int]:
    found = {parent}
    pending = [parent]
    while pending:
        pid = pending.pop()
        children = Path(f"/proc/{pid}/task/{pid}/children")
        if children.exists():
            for child in map(int, children.read_text().split()):
                if child not in found:
                    found.add(child)
                    pending.append(child)
    return found


class RuntimeSwitch:
    def __init__(self, launcher_pid: int, binary: Path, benchmark_binary: Path, ranks: int) -> None:
        self._launcher_pid = launcher_pid
        self._ranks = ranks
        self._binary = binary.resolve()
        with self._binary.open("rb") as executable:
            self._is_pie = int.from_bytes(executable.read(18)[16:18], "little") == 3
        symbol = "xllm_npu_sampling_ablation_mask"
        symbols = subprocess.check_output(["nm", "-a", str(self._binary)], text=True)
        matches = [line.split()[0] for line in symbols.splitlines() if line.split()[-1] == symbol]
        _require(
            len(matches) == 1,
            "Service lacks the test-only control; rebuild with BUILD_NPU_SAMPLING_ABLATION=ON",
        )
        layout = json.loads(subprocess.check_output([str(benchmark_binary.resolve()), "--sampling-ablation-layout"]))
        _require(layout == {"symbol": symbol, "bytes": 1}, "Unsupported sampling ablation layout")
        self._symbol = int(matches[0], 16)
        self._pids: list[int] | None = None

    def switch(self, mask: int) -> list[dict]:
        _require(mask >= 0 and mask & ~63 == 0, f"Invalid sampling mask: {mask}")
        pids = []
        for pid in _descendants(self._launcher_pid):
            executable = Path(f"/proc/{pid}/exe")
            if executable.exists() and executable.resolve() == self._binary:
                pids.append(pid)
        pids.sort()
        _require(len(pids) == self._ranks, pids)
        if self._pids is None:
            self._pids = pids
        _require(pids == self._pids, (pids, self._pids))
        stopped = []
        result = []
        try:
            for pid in pids:
                os.kill(pid, signal.SIGSTOP)
                stopped.append(pid)
            deadline = time.monotonic() + 5
            while any("\nState:\tT" not in Path(f"/proc/{pid}/status").read_text() for pid in pids):
                _require(time.monotonic() < deadline, "Workers did not stop")
                time.sleep(0.005)
            for pid in pids:
                mappings = Path(f"/proc/{pid}/maps").read_text().splitlines()
                starts = [
                    int(line.split()[0].split("-")[0], 16)
                    for line in mappings
                    if line.endswith(str(self._binary)) and int(line.split()[2], 16) == 0
                ]
                _require(len(starts) == 1, starts)
                base = (starts[0] if self._is_pie else 0) + self._symbol
                descriptor = os.open(f"/proc/{pid}/mem", os.O_RDWR)
                try:
                    current = os.pread(descriptor, 1, base)
                    _require(len(current) == 1 and current[0] <= 63, (pid, current))
                    before = {name: int(bool(current[0] & bit)) for name, bit in _BITS.items()}
                    value = bytes([mask])
                    written = os.pwrite(descriptor, value, base)
                    _require(written == 1, (pid, "Incomplete ablation write", written))
                    actual = os.pread(descriptor, 1, base)
                    _require(actual == value, (pid, actual, value))
                    after = {name: int(bool(actual[0] & bit)) for name, bit in _BITS.items()}
                    result.append({"pid": pid, "before": before, "after": after})
                finally:
                    os.close(descriptor)
        finally:
            for pid in stopped:
                os.kill(pid, signal.SIGCONT)
        return result
