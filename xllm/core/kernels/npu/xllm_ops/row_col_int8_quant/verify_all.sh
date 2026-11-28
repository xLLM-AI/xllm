#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "$(uname -m)" != aarch64 ]]; then
  echo "Run this verification on the documented aarch64 910C development host." >&2
  exit 1
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo "Load the CANN 9.1.0 set_env.sh before running this script." >&2
  exit 1
fi
mkdir -p task_submission
rm -f task_submission/acceptance.json
python3 tools/build_aclnn.py 2>&1 | tee task_submission/aclnn_build.log
bash build.sh 2>&1 | tee task_submission/build.log
export ASCEND_CUSTOM_OPP_PATH="$(pwd)/aclnn_install/vendors/community_task2${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
./build/run_aclnn_example 0 2>&1 | tee task_submission/aclnn_example.log
./build/run_example 0 2>&1 | tee task_submission/example.log
python3 tests/run_tests.py --task-dir tests/task 2>&1 | tee 'task_submission/2.2 精度自验证日志.log'
python3 tests/task/verify.py --solution-file python/solution.py --report task_submission/public_accuracy.json 2>&1 | tee task_submission/public_accuracy.log
python3 tests/run_audit.py 2>&1 | tee task_submission/audit.log
python3 tests/run_engineering.py --task-dir tests/task 2>&1 | tee 'task_submission/4.2 内存自验证日志.log'
python3 tests/run_benchmark.py --task-dir tests/task 2>&1 | tee 'task_submission/3.2 性能自验证日志.log'
python3 tests/check_acceptance.py 2>&1 | tee task_submission/acceptance.log
