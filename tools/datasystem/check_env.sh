#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

echo "XLLM_SRC=${XLLM_SRC}"
echo "XLLM_BUILD_DIR=${XLLM_BUILD_DIR}"
echo "DATASYSTEM_SDK=${DATASYSTEM_SDK}"
echo "PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}"
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH}"
echo "ATB_HOME_PATH=${ATB_HOME_PATH}"
echo "BUILD_WITH_MOONCAKE=${BUILD_WITH_MOONCAKE}"
echo "BUILD_WITH_DATASYSTEM=${BUILD_WITH_DATASYSTEM}"

case "${BUILD_WITH_MOONCAKE}" in
  ON|OFF) ;;
  *) fail "BUILD_WITH_MOONCAKE must be ON or OFF" ;;
esac
case "${BUILD_WITH_DATASYSTEM}" in
  ON|OFF) ;;
  *) fail "BUILD_WITH_DATASYSTEM must be ON or OFF" ;;
esac
if [[ "${BUILD_WITH_MOONCAKE}" == "OFF" &&
      "${BUILD_WITH_DATASYSTEM}" == "OFF" ]]; then
  fail "At least one KV cache backend must be enabled."
fi

[[ -d "${XLLM_SRC}" ]] || fail "xLLM source directory not found: ${XLLM_SRC}"
[[ -f "${XLLM_SRC}/CMakeLists.txt" ]] || fail "CMakeLists.txt not found under ${XLLM_SRC}"
if [[ "${BUILD_WITH_DATASYSTEM}" == "ON" ]]; then
  [[ -f "${XLLM_SRC}/xllm/core/framework/kv_cache_transfer/yuanrong_kv_cache_transfer.cpp" ]] ||
    fail "DataSystem transfer source is missing. Use the packaged source tree."
fi

[[ -n "${PYTHON_EXECUTABLE}" ]] || fail "python3/python3.11 not found. Set PYTHON_EXECUTABLE=/path/to/python."
[[ -x "${PYTHON_EXECUTABLE}" ]] || fail "PYTHON_EXECUTABLE is not executable: ${PYTHON_EXECUTABLE}"
"${PYTHON_EXECUTABLE}" --version

if [[ "${BUILD_WITH_DATASYSTEM}" == "ON" ]]; then
  [[ -f "${DATASYSTEM_SDK}/include/datasystem/hetero_client.h" ]] ||
    fail "DataSystem header not found: ${DATASYSTEM_SDK}/include/datasystem/hetero_client.h"
  if [[ ! -f "${DATASYSTEM_SDK}/lib/libdatasystem.so" &&
        ! -f "${DATASYSTEM_SDK}/lib64/libdatasystem.so" ]]; then
    fail "libdatasystem.so not found under ${DATASYSTEM_SDK}/lib or ${DATASYSTEM_SDK}/lib64"
  fi
fi

[[ -d "${ASCEND_HOME_PATH}" ]] || fail "ASCEND_HOME_PATH not found: ${ASCEND_HOME_PATH}"
[[ -d "${ASCEND_OPP_PATH}" ]] || fail "ASCEND_OPP_PATH not found: ${ASCEND_OPP_PATH}"
[[ -d "${ATB_HOME_PATH}" ]] || fail "ATB_HOME_PATH not found: ${ATB_HOME_PATH}"

if ! command -v cmake >/dev/null 2>&1; then
  fail "cmake not found in PATH"
fi
cmake --version | head -n 1

if command -v ninja >/dev/null 2>&1; then
  ninja --version | sed 's/^/ninja /'
else
  echo "WARN: ninja not found; CMake will use its default generator."
fi

echo "Environment check OK."
