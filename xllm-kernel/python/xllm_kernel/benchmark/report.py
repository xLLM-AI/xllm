# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""JSON and Markdown performance reports."""

import json
from pathlib import Path
from typing import Any


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        "# RMSNorm benchmark",
        "",
        f"Status: {report['status']}",
        "",
        "| Case | Implementation | Status | Device median (µs/op) |",
        "| --- | --- | --- | ---: |",
    ]
    for row in report["cases"]:
        median = row.get("timing", {}).get("device_interval_us", {}).get("median")
        lines.append(
            f"| {row['case_id']} | {row['implementation']} | {row['status']} | {median if median is not None else '—'} |"
        )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
