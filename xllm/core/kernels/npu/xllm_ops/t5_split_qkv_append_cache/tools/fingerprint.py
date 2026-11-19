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

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _create() -> Any:
    records = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(
            p
            in {
                ".git",
                "node_modules",
                "__pycache__",
                "reports",
                "research",
                "task_submission",
                "artifacts",
                "build",
                "python_deps",
            }
            for p in rel.parts
        ):
            continue
        records.append(
            {
                "path": rel.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    deployment = ROOT / "DEPLOYMENT_MANIFEST.json"
    try:
        if deployment.exists():
            manifest = json.loads(deployment.read_text(encoding="utf-8"))
            commit = manifest["commit"]
            mismatches = []
            for record in manifest["files"]:
                path = ROOT / Path(record["path"]).relative_to(manifest["prefix"])
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                    mismatches.append(record["path"])
            dirty = "archive verified" if not mismatches else "modified: " + repr(mismatches)
            return {
                "source_commit": commit,
                "worktree_status": dirty,
                "python": platform.python_version(),
                "files": records,
            }
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "not_recorded", "unknown"
    return {"source_commit": commit, "worktree_status": dirty, "python": platform.python_version(), "files": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_create(), indent=2) + "\n", encoding="utf-8")
