"""Generate the submitted reports from the official verifier outputs.

Inputs (committed under ``evidence/``):
    functional_result.json  produced by golden.py --verify-functional
    result.json             produced by the paired benchmark driver
    functional.log          raw verifier stdout
    perf.log                raw benchmark stdout

Outputs:
    evidence/generalization.json
    task_submission/2.1 .. 4.2
    task_submission.zip
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
OUT = ROOT / "task_submission"
HEADER_FILL = PatternFill("solid", fgColor="DDEBF7")


def load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text())


def fill_sheet(sheet, headers: list[str], rows: list[list], widths: list[int]) -> None:
    sheet.append(headers)
    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.fill = HEADER_FILL
    for row in rows:
        sheet.append(row)
    sheet.freeze_panes = "A2"
    for index, width in enumerate(widths, 1):
        sheet.column_dimensions[get_column_letter(index)].width = width


def write_precision(functional: dict) -> None:
    workbook = Workbook()
    summary = workbook.active
    summary.title = "汇总"
    environment = functional["environment"]
    versions = environment["versions"]
    passed = sum(1 for case in functional["cases"] if case["status"] == "PASS")
    negative_passed = sum(1 for item in functional["negative"] if item["status"] == "PASS")
    dynamic_records = [record for records in functional["dynamic_updates"].values() for record in records]
    storage_records = [record for records in functional["storage_rebindings"].values() for record in records]
    summary_rows = [
        ["总判定", functional["status"]],
        ["任务", functional["task"]],
        ["范围", functional["scope"]],
        ["环境时间（UTC）", environment["utc"]],
        ["主机", environment["hostname"]],
        ["设备", environment["device"]],
        ["Python", environment["python"]],
        ["torch", versions["torch"]],
        ["torch-npu", versions["torch-npu"]],
        ["triton-ascend", versions["triton-ascend"]],
        ["triton target", environment["triton_target"]],
        ["正向用例", f"{passed}/{len(functional['cases'])} PASS"],
        ["非法输入", f"{negative_passed}/{len(functional['negative'])} PASS"],
        ["动态更新检查", f"{sum(1 for r in dynamic_records if r['status'] == 'PASS')}/{len(dynamic_records)} PASS"],
        ["存储重绑定检查", f"{sum(1 for r in storage_records if r['status'] == 'PASS')}/{len(storage_records)} PASS"],
        ["单层大输入", functional["large_single_layer"]["status"]],
        ["数据来源", "evidence/functional_result.json"],
        ["原始日志", "task_submission/2.2 精度自验证日志.log"],
    ]
    fill_sheet(summary, ["项", "值"], summary_rows, [26, 72])

    cases = workbook.create_sheet("正向用例")
    case_rows = []
    for case in functional["cases"]:
        checks = case["checks"]
        case_rows.append(
            [
                case["id"],
                case["status"],
                json.dumps(case["shape"]),
                json.dumps(case["input_shapes"]),
                checks["candidate_mismatches"],
                checks["baseline_mismatches"],
                checks["shape"],
                checks["dtype"],
                checks["npu"],
                checks["contiguous"],
                checks["inputs_unchanged"],
                checks["independent_storage"],
            ]
        )
    fill_sheet(
        cases,
        ["用例", "判定", "输出形状", "输入形状", "候选差异数", "基线差异数", "形状", "类型", "NPU", "连续", "输入不变", "独立存储"],
        case_rows,
        [26, 8, 12, 24, 12, 12, 8, 8, 8, 8, 10, 10],
    )

    negative = workbook.create_sheet("非法输入")
    fill_sheet(
        negative,
        ["场景", "判定", "期望异常"],
        [[item["id"], item["status"], "TypeError 或 ValueError"] for item in functional["negative"]],
        [34, 8, 22],
    )

    dynamic = workbook.create_sheet("动态更新")
    fill_sheet(
        dynamic,
        ["用例组", "检查", "判定", "候选差异数", "对照"],
        [
            [group, record["id"], record["status"], record["candidate_mismatches"], record["compared_with"]]
            for group, records in functional["dynamic_updates"].items()
            for record in records
        ],
        [22, 38, 8, 12, 52],
    )

    storage = workbook.create_sheet("存储重绑定")
    fill_sheet(
        storage,
        ["用例组", "检查", "判定", "候选差异数", "storage_offset", "指针变化", "旧对齐 mod16", "新对齐 mod16"],
        [
            [
                group,
                record["id"],
                record["status"],
                record["candidate_mismatches"],
                record["storage_offset"],
                record["pointer_changed"],
                record["old_alignment_mod16"],
                record["new_alignment_mod16"],
            ]
            for group, records in functional["storage_rebindings"].items()
            for record in records
        ],
        [22, 34, 8, 12, 14, 10, 14, 14],
    )

    single = workbook.create_sheet("单层大输入")
    large = functional["large_single_layer"]
    fill_sheet(
        single,
        ["项", "值"],
        [
            ["用例", large["id"]],
            ["判定", large["status"]],
            ["形状", json.dumps(large["shape"])],
            ["输入形状", json.dumps(large["input_shapes"])],
            *[[key, value] for key, value in large["checks"].items()],
        ],
        [24, 40],
    )

    workbook.save(OUT / "2.1 精度自验证报告.xlsx")


def write_performance(result: dict, functional: dict) -> None:
    workbook = Workbook()
    summary = workbook.active
    summary.title = "汇总"
    rows = []
    for entry in result["benchmarks"]:
        acceptance = result["acceptance"].get(entry["id"], {})
        target = acceptance.get("target_us", "—")
        low, high = entry["saved_mean_ci95_us"]
        verdict = "PASS"
        if acceptance and not all(
            acceptance.get(key, False)
            for key in ("mean_ok", "mean_saved_positive", "median_saved_positive", "ci_lower_positive")
        ):
            verdict = "FAIL"
        rows.append(
            [
                entry["id"],
                target,
                round(entry["baseline_mean_us"], 3),
                round(entry["candidate_mean_us"], 3),
                round(entry["baseline_median_us"], 3),
                round(entry["candidate_median_us"], 3),
                round(entry["saved_mean_us"], 3),
                round(entry["saved_median_us"], 3),
                round(low, 3),
                round(high, 3),
                acceptance.get("mean_ok", "—"),
                acceptance.get("mean_saved_positive", "—"),
                acceptance.get("median_saved_positive", "—"),
                acceptance.get("ci_lower_positive", "—"),
                verdict if acceptance else "参考",
            ]
        )
    fill_sheet(
        summary,
        ["用例", "目标（μs）", "基线均值", "提交均值", "基线中位数", "提交中位数", "均值差", "中位数差", "CI95 下界", "CI95 上界", "均值达标", "均值差为正", "中位数差为正", "CI 下界为正", "判定"],
        rows,
        [22, 12, 12, 12, 13, 13, 12, 13, 12, 12, 10, 13, 14, 13, 8],
    )

    rounds = workbook.create_sheet("逐轮原始数据")
    round_rows = []
    for entry in result["benchmarks"]:
        for index, (baseline, candidate) in enumerate(zip(entry["baseline_rounds_us"], entry["candidate_rounds_us"]), 1):
            round_rows.append([entry["id"], index, round(baseline, 3), round(candidate, 3), round(baseline - candidate, 3)])
    fill_sheet(rounds, ["用例", "轮次", "基线（μs）", "提交（μs）", "每轮差（μs）"], round_rows, [22, 8, 14, 14, 14])

    environment = workbook.create_sheet("环境与协议")
    functional_versions = functional["environment"]["versions"]
    fill_sheet(
        environment,
        ["项", "值"],
        [
            ["设备", functional["environment"]["device"]],
            ["Python", functional["environment"]["python"]],
            ["torch", functional_versions["torch"]],
            ["torch-npu", functional_versions["torch-npu"]],
            ["triton-ascend", functional_versions["triton-ascend"]],
            ["triton（运行库版本）", result["environment"]["triton"]],
            ["预热", "50 次"],
            ["轮数", "6 轮交替"],
            ["每轮迭代", "200 次"],
            ["同步", "每轮末 torch.npu.synchronize()"],
            ["bootstrap", "10000 次重采样，seed 20261031，线性插值"],
            ["数据来源", "evidence/result.json（逐轮原始数组）"],
            ["原始日志", "task_submission/3.2 性能自验证日志.log"],
        ],
        [26, 72],
    )
    workbook.save(OUT / "3.1 性能自验证报告.xlsx")


def write_memory(functional: dict) -> None:
    workbook = Workbook()
    summary = workbook.active
    summary.title = "汇总"
    rows = []
    for case, metrics in functional["engineering_by_case"].items():
        peak = metrics["peak_increment_bytes"]
        output = metrics["output_bytes"]
        rows.append(
            [
                case,
                peak,
                output,
                peak - output,
                metrics["baseline_peak_increment_bytes"],
                metrics["baseline_output_bytes"],
                metrics["baseline_peak_increment_bytes"] - metrics["baseline_output_bytes"],
                metrics["candidate_minus_baseline_peak_bytes"],
                metrics["active_before_bytes"],
                metrics["active_after_bytes"],
                metrics["two_streams"],
                metrics["retained_outputs_independent"],
                metrics["repeat20_released"],
            ]
        )
    fill_sheet(
        summary,
        ["用例", "峰值增量（B）", "输出（B）", "临时（B）", "基线峰值增量（B）", "基线输出（B）", "基线临时（B）", "峰值差（B）", "测量前活跃（B）", "测量后活跃（B）", "双流", "保留 3 份输出", "20 次回收"],
        rows,
        [22, 16, 12, 12, 18, 14, 12, 14, 16, 16, 8, 14, 12],
    )

    notes = workbook.create_sheet("口径与判定")
    fill_sheet(
        notes,
        ["项", "内容"],
        [
            ["峰值定义", next(iter(functional["engineering_by_case"].values()))["peak_definition"]],
            ["双流", "两条 stream 各保留 3 份输出，均独立可读"],
            ["回收", "连续 20 次调用后释放并同步，活跃分配回到测量前水平"],
            ["重绑定", "X/M/cu 以同形状 set_ 换到 offset=1 的存储后结果仍与 CPU 标杆一致"],
            ["单层大输入", functional["large_single_layer"]["status"]],
            ["结论", "两组性能输入的活跃分配均恢复；峰值增量为输出加当次临时张量"],
            ["数据来源", "evidence/functional_result.json 的 engineering_by_case"],
            ["原始日志", "task_submission/4.2 内存自验证日志.log"],
        ],
        [20, 100],
    )
    workbook.save(OUT / "4.1 内存自验证报告.xlsx")

    log_lines = [
        "PackedPageIndices 内存自验证日志",
        "来源：golden.py --verify-functional 的 engineering_by_case 与 large_single_layer 原样输出",
        f"环境时间（UTC）：{functional['environment']['utc']}",
        f"设备：{functional['environment']['device']}",
        "",
        json.dumps(
            {
                "engineering_by_case": functional["engineering_by_case"],
                "large_single_layer": functional["large_single_layer"],
                "storage_rebindings": functional["storage_rebindings"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
    ]
    (OUT / "4.2 内存自验证日志.log").write_text("\n".join(log_lines))


def write_generalization(functional: dict, result: dict) -> None:
    payload = {
        "task": functional["task"],
        "source": "golden.py --verify-functional",
        "environment": functional["environment"],
        "input_cases": [
            {
                "id": case["id"],
                "status": case["status"],
                "shape": case["shape"],
                "checks": case["checks"],
            }
            for case in functional["cases"]
        ],
        "negative": functional["negative"],
        "dynamic_updates": functional["dynamic_updates"],
        "storage_rebindings": functional["storage_rebindings"],
        "large_single_layer": functional["large_single_layer"],
        "paired_timing": {
            "source": "evidence/result.json",
            "benchmarks": [
                {
                    "id": entry["id"],
                    "baseline_rounds_us": entry["baseline_rounds_us"],
                    "candidate_rounds_us": entry["candidate_rounds_us"],
                    "baseline_mean_us": entry["baseline_mean_us"],
                    "candidate_mean_us": entry["candidate_mean_us"],
                    "baseline_median_us": entry["baseline_median_us"],
                    "candidate_median_us": entry["candidate_median_us"],
                    "saved_mean_us": entry["saved_mean_us"],
                    "saved_median_us": entry["saved_median_us"],
                    "saved_mean_ci95_us": entry["saved_mean_ci95_us"],
                }
                for entry in result["benchmarks"]
            ],
            "acceptance": result["acceptance"],
        },
    }
    (EVIDENCE / "generalization.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def write_logs() -> None:
    (OUT / "2.2 精度自验证日志.log").write_bytes((EVIDENCE / "functional.log").read_bytes())
    (OUT / "3.2 性能自验证日志.log").write_bytes((EVIDENCE / "perf.log").read_bytes())


def write_zip() -> None:
    archive = ROOT / "task_submission.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(OUT.iterdir()):
            if path.is_file():
                bundle.write(path, f"task_submission/{path.name}")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    functional = load("functional_result.json")
    result = load("result.json")
    write_precision(functional)
    write_performance(result, functional)
    write_memory(functional)
    write_logs()
    write_generalization(functional, result)
    write_zip()
    print("reports written to", OUT)
    print("generalization written to", EVIDENCE / "generalization.json")


if __name__ == "__main__":
    main()
