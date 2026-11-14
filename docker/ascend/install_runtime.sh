#!/usr/bin/env bash
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

set -euo pipefail

# Keep the plugin download working directory local to this step.
(
    mkdir -p /tmp/xllm
    cd /tmp/xllm
    wget -q "https://gitcode.com/xLLM-AI/xllm/releases/download/${ATB_PLUGIN_RELEASE_TAG}/ops_customize_aarch64.tar.gz"
    tar -zxvf ops_customize_aarch64.tar.gz
    cd output/ops_customize/cxx_abi_1/
    cp -r configs/* "${ATB_HOME_PATH}/configs/"
    cp -r include/* "${ATB_HOME_PATH}/include/"
    cp lib/* "${ATB_HOME_PATH}/lib/"
    rm -rf /tmp/xllm
)

cat >> /root/.bashrc <<'EOF'
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/opp/vendors/custom_xllm_math}
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/opp/vendors/custom_xllm_math/op_api/lib/:${LD_LIBRARY_PATH}
EOF
