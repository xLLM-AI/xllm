#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export XLLM_ROOT="$ROOT"
export XLLM_SRC="$ROOT" XLLM_SRC_OVERRIDE="$ROOT" XLLM_BUILD_DIR="$ROOT/build"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/config.env}"
[[ -f "$CONFIG_FILE" ]] || { echo "Copy config.env.example to config.env first" >&2; return 2; }
source "$CONFIG_FILE"
DS_PYTHON="$(command -v "$DS_PYTHON")"
export DS_PYTHON PYTHON_EXECUTABLE="$DS_PYTHON"
export PATH="$(dirname "$DS_PYTHON"):$PATH"
[[ -f "$CANN_ENV_SH" && -f "$ATB_ENV_SH" ]] || { echo "Missing CANN/ATB environment" >&2; return 2; }
# Clear allocator injection before executing Python or importing the SDK.
unset LD_PRELOAD
source "$CANN_ENV_SH"
source "$ATB_ENV_SH"
unset LD_PRELOAD
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_OPP_PATH="${ASCEND_OPP_PATH:-$ASCEND_HOME_PATH/opp}"
export ATB_HOME_PATH="${ATB_HOME_PATH:-/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1}"
export CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" CLEAN_BUILD="${CLEAN_BUILD:-1}"
DATASYSTEM_SDK="$("$DS_PYTHON" "$SCRIPT_DIR/sdk_info.py" --root)" || return 2
export DATASYSTEM_SDK
[[ -f "$DATASYSTEM_SDK/lib/libdatasystem.so" ]] && SDK_LIB_DIR="$DATASYSTEM_SDK/lib" || SDK_LIB_DIR="$DATASYSTEM_SDK/lib64"
export LD_LIBRARY_PATH="$SDK_LIB_DIR:${LD_LIBRARY_PATH:-}"
export XLLM_BIN="${XLLM_BIN:-$ROOT/build/xllm/core/server/xllm}"
export BRIDGE_SO="${BRIDGE_SO:-$ROOT/build/xllm/core/server/libxllm_yuanrong_datasystem_bridge.so}"
export XLLM_YUANRONG_DATASYSTEM_BRIDGE="$BRIDGE_SO"
export XLLM_YUANRONG_BRIDGE_BASE_NAMESPACE=1
export XLLM_YUANRONG_ENABLE_REMOTE_H2D=0 XLLM_YUANRONG_TIMEOUT_MS=60000
export XLLM_YUANRONG_WORKER_HOST="$DATASYSTEM_WORKER_HOST"
export XLLM_YUANRONG_WORKER_PORT="$DATASYSTEM_WORKER_PORT"
export ETCD_ADDR="$ETCD_HOST:$ETCD_CLIENT_PORT"
export DATASYSTEM_ETCD_HOST="$ETCD_HOST" DATASYSTEM_ETCD_PORT="$ETCD_CLIENT_PORT"
export DATASYSTEM_ETCD_ADDR="$ETCD_ADDR" START_DATASYSTEM_WORKER=0
export BUILD_WITH_MOONCAKE=ON BUILD_WITH_DATASYSTEM=ON
export no_proxy="localhost,127.0.0.1,${no_proxy:-}" NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"
