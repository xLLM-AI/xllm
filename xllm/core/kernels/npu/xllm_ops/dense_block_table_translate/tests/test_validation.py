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

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paired_statistics import summarize
from verify import check_coverage

CONTRACT = json.loads((ROOT / "official/case.json").read_text(encoding="utf-8"))


class ValidationTests(unittest.TestCase):
    def test_paired_statistics(self) -> None:
        result = summarize([80.0] * 6, [50.0] * 6, CONTRACT["acceptance"], "dense_perf_large")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["saved_mean_ci95_us"], [30.0, 30.0])

    def test_latency_failures_are_rejected(self) -> None:
        for candidate in ([80.0] * 6, [61.0] * 6, [90.0] * 6):
            result = summarize([80.0] * 6, candidate, CONTRACT["acceptance"], "dense_perf_large")
            self.assertEqual(result["status"], "FAIL")
        for invalid in ([1.0] * 5, [1.0] * 5 + [float("nan")], [1.0] * 5 + [0.0]):
            with self.assertRaises(ValueError):
                summarize([80.0] * 6, invalid, CONTRACT["acceptance"], "dense_perf_large")

    def test_missing_and_duplicate_coverage_are_rejected(self) -> None:
        record = copy.deepcopy(CONTRACT["expected_checks"]["result.json"])
        record["status"] = "PASS"
        record["cases"] = [{"id": case["id"], "status": "PASS"} for case in CONTRACT["input_cases"]]
        check_coverage(record, CONTRACT)
        record["cases"].pop()
        with self.assertRaises(AssertionError):
            check_coverage(record, CONTRACT)
        record["cases"].append(record["cases"][0])
        with self.assertRaises(AssertionError):
            check_coverage(record, CONTRACT)

    def test_failed_negative_checks_are_rejected(self) -> None:
        record = copy.deepcopy(CONTRACT["expected_checks"]["result.json"])
        record["status"] = "PASS"
        record["cases"] = [{"id": case["id"], "status": "PASS"} for case in CONTRACT["input_cases"]]
        record["negative"][0]["status"] = "FAIL"
        with self.assertRaises(AssertionError):
            check_coverage(record, CONTRACT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
