#!/usr/bin/env bash
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

set -euo pipefail
TASK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT:$TASK_ROOT/reference${PYTHONPATH:+:$PYTHONPATH}"
python3 tools/preflight.py --static
python3 tools/preflight.py --output reports/environment_before.json
: "${ASCEND_RT_VISIBLE_DEVICES:?Set the assigned single logical device visibility}"
: "${T5_DEVICE_CONFIRMED:?Review ports, worker processes and device occupancy; set T5_DEVICE_CONFIRMED=1 only for your assigned free device}"
[[ "$T5_DEVICE_CONFIRMED" == 1 ]] || exit 2
python3 tools/check_environment.py --input reports/environment_before.json --accepted-driver "${T5_ACCEPTED_DRIVER:-25.5.t6.b001}"
for file in scripts/*.sh; do bash -n "$file"; done
if [[ -e reports/public_accuracy.json ]]; then
  echo 'Existing device reports detected; preserve this baseline and use a new checkout for a discriminating rerun.' >&2
  exit 2
fi
trap 'python3 tools/preflight.py --output reports/environment_after.json > reports/environment_after.log 2>&1' EXIT
# No server or listening port is used. Begin with the independent C++ smoke call.
build/examples/t5_example > reports/cpp_smoke.log 2>&1
build/examples/t5_example --bf16 >> reports/cpp_smoke.log 2>&1
build/examples/t5_example --negative > reports/aclnn_invalid_metadata.log 2>&1
python3 reference/verify.py --solution-file "$TASK_ROOT/solution.py" --report reports/public_accuracy.json \
  > reports/public_accuracy.log 2>&1
python3 tests/npu_suite.py --group accuracy --report reports/extended_accuracy.json \
  > reports/extended_accuracy.log 2>&1
python3 tests/npu_suite.py --group engineering --report reports/engineering.json \
  > reports/engineering.log 2>&1
python3 tests/npu_suite.py --group memory --report reports/memory.json \
  > reports/memory.log 2>&1
python3 reference/benchmark.py --solution-file "$TASK_ROOT/solution.py" --report reports/performance.json \
  > reports/performance.log 2>&1
python3 reference/check_performance.py --report reports/performance.json --output reports/performance_acceptance.json
python3 tests/compare_baseline.py --report reports/baseline_comparison.json > reports/baseline_comparison.log 2>&1
python3 tools/check_coverage.py --reports reports
python3 tools/fingerprint.py --output artifacts/tested_build_manifest.json
