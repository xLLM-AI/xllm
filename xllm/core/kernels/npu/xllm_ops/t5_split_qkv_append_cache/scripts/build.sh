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
: "${ASCEND_HOME_PATH:?Load the assigned CANN 9.1.0 environment first}"
: "${T5_OPP_ROOT:=$HOME/.local/t5_qkv_opp}"
export T5_OPP_ROOT
case "$(realpath -m -- "$T5_OPP_ROOT")" in
  "$HOME"/*) ;;
  *) echo 'T5_OPP_ROOT must be a private directory below HOME' >&2; exit 2 ;;
esac
for command in python3 msopgen cmake c++; do command -v "$command" >/dev/null; done
python3 "$TASK_ROOT/tools/preflight.py" --static
python3 "$TASK_ROOT/tools/check_cache_policy.py" --cann-root "$ASCEND_HOME_PATH"
# Never overwrite an earlier generated/built baseline.
if [[ -e "$TASK_ROOT/build/aclnn" ]]; then
  echo 'build/aclnn already exists. Preserve its fingerprint; choose a fresh checkout for a rebuild.' >&2
  exit 2
fi
mkdir -p "$TASK_ROOT/build" "$TASK_ROOT/artifacts" "$T5_OPP_ROOT"
msopgen gen -i "$TASK_ROOT/op_project.json" -c ai_core-Ascend910_9382 -lan cpp \
  -out "$TASK_ROOT/build/aclnn" -f aclnn
cp -a "$TASK_ROOT/op_host/." "$TASK_ROOT/build/aclnn/op_host/"
cp -a "$TASK_ROOT/op_kernel/." "$TASK_ROOT/build/aclnn/op_kernel/"
cp -a "$TASK_ROOT/common" "$TASK_ROOT/build/aclnn/common"
python3 "$TASK_ROOT/tools/configure_generated.py" "$TASK_ROOT/build/aclnn"
( cd "$TASK_ROOT/build/aclnn"; bash -n build.sh; bash build.sh )
mapfile -t packages < <(find "$TASK_ROOT/build/aclnn/build_out" -maxdepth 1 -name '*.run' -type f)
if [[ ${#packages[@]} -ne 1 ]]; then echo 'Expected one .run package' >&2; exit 2; fi
cp -- "${packages[0]}" "$TASK_ROOT/artifacts/"
bash "${packages[0]}" --quiet --install-path="$T5_OPP_ROOT"
export ASCEND_CUSTOM_OPP_PATH="$T5_OPP_ROOT/vendors/t5qkv${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
cmake -S "$TASK_ROOT/direct" -B "$TASK_ROOT/build/direct" \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" -DASCEND_COMPUTE_UNIT=ascend910_93
cmake --build "$TASK_ROOT/build/direct" -j "${T5_BUILD_JOBS:-8}"
( cd "$TASK_ROOT"; python3 setup.py build_ext --inplace )
( cd "$TASK_ROOT"; python3 -c 'import torch, torch_npu, _t5_qkv; print("extension import passed")' )
cmake -S "$TASK_ROOT/examples" -B "$TASK_ROOT/build/examples" \
  -DCANN_ROOT="$ASCEND_HOME_PATH" -DT5_OPP_ROOT="$T5_OPP_ROOT"
cmake --build "$TASK_ROOT/build/examples" -j "${T5_BUILD_JOBS:-8}"
python3 "$TASK_ROOT/tools/fingerprint.py" --output "$TASK_ROOT/artifacts/build_manifest.json"
echo 'Build completed. Run scripts/validate.sh only after checking the assigned device is free.'
