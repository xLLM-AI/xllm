# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Run and reconcile the complete published acceptance suite."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(command: list[str], log: Path, root: Path) -> None:
    with log.open("w") as stream:
        subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)


def _coverage(
    functional: dict[str, Any], performance: dict[str, Any], extended: dict[str, Any], contract: dict[str, Any]
) -> None:
    assert all(record["status"] == "PASS" for record in (functional, performance, extended))
    assert {r["id"] for r in functional["cases"]} == {r["id"] for r in contract["input_cases"]}
    expected_negative = {r["id"] for r in contract["expected_checks"]["result.json"]["negative"]}
    assert {r["id"] for r in functional["negative"]} == expected_negative
    assert {r["id"] for r in extended["negative"]} == expected_negative
    for case in ("packed_perf_small", "packed_perf_large"):
        dynamic = {r["id"] for r in functional["dynamic_updates"][case]}
        assert {"logical_values_changed", "mapping_values_changed", "cu_counts_changed_same_total"} <= dynamic
        assert {r["id"] for r in functional["storage_rebindings"][case]} == {
            "mapping_set_storage",
            "logical_set_storage",
            "cu_set_storage_same_total",
        }
        memory = functional["engineering_by_case"][case]
        assert memory["active_before_bytes"] == memory["active_after_bytes"]
        assert memory["baseline_active_before_bytes"] == memory["baseline_active_after_bytes"]
        assert all(
            memory[key] == "PASS" for key in ("two_streams", "retained_outputs_independent", "repeat20_released")
        )
    assert "large_path_int32_output_boundary" in {r["id"] for r in functional["dynamic_updates"]["packed_perf_large"]}
    assert functional["large_single_layer"]["status"] == "PASS"
    records = performance["benchmarks"]
    assert [r["id"] for r in records] == [
        "packed_perf_small",
        "packed_perf_large",
        "packed_route_32736",
        "packed_route_32768",
    ]
    for record in records:
        assert len(record["raw_rounds"]) == 6 and record["warmup"] == 50 and record["iterations"] == 200
        if record["required"]:
            assert all(record["gates"].values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output_dir.resolve()
    if output.is_relative_to(root) or output.exists():
        parser.error("results must use a new directory outside the submission project")
    output.mkdir(parents=True)
    report: dict[str, Any] = {"status": "RUNNING"}
    try:
        _run(
            [
                sys.executable,
                str(root / "fixtures/golden.py"),
                "--verify-functional",
                "--implementation",
                str(root),
                "--output",
                str(output / "functional.json"),
            ],
            output / "functional.log",
            root,
        )
        _run(
            [sys.executable, str(root / "test_contract.py"), "--output", str(output / "contract.json")],
            output / "contract.log",
            root,
        )
        _run(
            [sys.executable, str(root / "benchmark.py"), "--output", str(output / "performance.json")],
            output / "performance.log",
            root,
        )
        functional = json.loads((output / "functional.json").read_text())
        performance = json.loads((output / "performance.json").read_text())
        extended = json.loads((output / "contract.json").read_text())
        contract = json.loads((root / "fixtures/case.json").read_text())
        _coverage(functional, performance, extended, contract)
        report = {
            **functional,
            "status": "PASS",
            "benchmarks": performance["benchmarks"],
            "contract_checks": extended,
            "performance_environment": performance["environment"],
            "coverage": "all published input cases, negative/state/storage/stream/memory checks and timings",
        }
        ids = set(contract["generalization"]["case_ids"])
        generalization = {
            "status": "PASS",
            "cases": [r for r in functional["cases"] if r["id"] in ids],
            "benchmarks": [r for r in performance["benchmarks"] if r["id"] in ids],
        }
        (output / "generalization.json").write_text(json.dumps(generalization, ensure_ascii=False, indent=2) + "\n")
        manifest = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in (".py", ".cpp", ".txt")
            and not any(
                part in ("build", "__pycache__", ".git", "task_submission") for part in path.relative_to(root).parts
            )
        }
        (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    except Exception as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
