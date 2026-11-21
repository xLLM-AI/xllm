# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preserve an explicit failure record even if the native compiler aborts Python."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def main() -> int:
    output = ROOT / "results/npu"
    args = sys.argv[1:]
    if "--output" in args:
        output = Path(args[args.index("--output") + 1])
    output.mkdir(parents=True, exist_ok=True)
    before = set(output.iterdir())
    code = subprocess.call([sys.executable, str(ROOT / "verify.py"), *args])
    if code:
        for directory in set(output.iterdir()) - before:
            if not directory.is_dir():
                continue
            path = directory / "result.json"
            record = json.loads(path.read_text()) if path.exists() else {}
            if record.get("status") in (None, "RUNNING"):
                record.update(
                    status="FAIL",
                    error=f"Verifier process exited {code}; inspect raw.log and console log",
                    finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                )
                path.write_text(json.dumps(record, indent=2) + "\n")
            (directory / "process_exit.json").write_text(json.dumps({"returncode": code}) + "\n")
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
