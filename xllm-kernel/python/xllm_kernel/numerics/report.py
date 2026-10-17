# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Operation-independent JSON and Markdown numerical reports."""

import json
from pathlib import Path
from typing import Any


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Kernel validation",
        "",
        f"Status: {report['status']}",
        "",
        "| Case | Operation | Implementation | Status |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["cases"]:
        lines.append(f"| {row['case_id']} | {row['operation']} | {row['implementation']} | {row['status']} |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
