#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

bash "${SCRIPT_DIR}/check_env.sh"

cd "${XLLM_SRC}"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [[ "${OFFLINE_BUILD:-0}" == "1" ]]; then
    echo "OFFLINE_BUILD=1: skipping network submodule sync; verifying submodules are already present..."
    # Verify submodules are already populated without touching the network.
    # `git submodule status` exits 0 even when submodules are uninitialized;
    # an uninitialized submodule shows a leading '-' in its status line.
    if git submodule status --recursive | grep -qE '^-'; then
      echo "ERROR: OFFLINE_BUILD=1 but one or more submodules are not initialized." >&2
      echo "       Run git submodule update --init --recursive before building." >&2
      git submodule status --recursive >&2 || true
      exit 1
    fi
    echo "  all submodules already initialized (offline)."
  else
    echo "Syncing/updating xLLM git submodules through the official git path..."
    git submodule sync --recursive
    git submodule update --init --recursive
  fi
else
  echo "ERROR: ${XLLM_SRC} is not a git work tree. This package must include .git metadata." >&2
  exit 1
fi

if [[ "${CLEAN_BUILD}" == "1" ]]; then
  official_build_dir="${XLLM_SRC}/build/cmake.linux-aarch64-cpython-311"
  echo "Removing official setup.py CMake build directory: ${official_build_dir}"
  rm -rf "${official_build_dir}"
fi

export MAX_JOBS="${MAX_JOBS}"
if [[ "${BUILD_TESTS:-0}" == 1 ]]; then
  unset SKIP_TEST
else
  export SKIP_TEST=1
fi
export SKIP_EXPORT="${SKIP_EXPORT:-1}"

cmake_args=(
  "-DUSE_MOONCAKE=${BUILD_WITH_MOONCAKE}"
  "-DUSE_YUANRONG_DATASYSTEM=${BUILD_WITH_DATASYSTEM}"
  "-DPython_EXECUTABLE=${PYTHON_EXECUTABLE}"
  "-DPython3_EXECUTABLE=${PYTHON_EXECUTABLE}"
)

if [[ "${BUILD_WITH_DATASYSTEM}" == "ON" ]]; then
  cmake_args+=("-DYUANRONG_DATASYSTEM_ROOT=${DATASYSTEM_SDK}")
fi

if [[ -n "${VCPKG_ROOT:-}" && -f "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" ]]; then
  cmake_args+=("-DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
fi

if [[ -n "${EXTRA_CMAKE_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${EXTRA_CMAKE_ARGS} )
  cmake_args+=("${extra_args[@]}")
fi

export CMAKE_ARGS="${cmake_args[*]}"

# A pre-20260710 build produced this C++ plugin. It is intentionally gone: the
# adapter now lives in the xllm binary and only the C-ABI bridge remains. Remove
# exact stale outputs so a successful rebuild cannot accidentally run old code.
rm -f \
  "${XLLM_SRC}/build/xllm/core/server/libxllm_yuanrong_kv_cache_transfer.so" \
  "${XLLM_SRC}/build/cmake.linux-aarch64-cpython-311/xllm/core/server/libxllm_yuanrong_kv_cache_transfer.so"

echo "Building xLLM through official setup.py build_ext path..."
echo "CMAKE_ARGS=${CMAKE_ARGS}"
echo "MAX_JOBS=${MAX_JOBS}"
echo "BUILD_TESTS=${BUILD_TESTS:-0}"
echo "SKIP_EXPORT=${SKIP_EXPORT}"
printf ' %q' "${PYTHON_EXECUTABLE}" setup.py --device npu build_ext
echo
"${PYTHON_EXECUTABLE}" setup.py --device npu build_ext

official_build_dir="${XLLM_SRC}/build/cmake.linux-aarch64-cpython-311"
server_output_dir="${XLLM_SRC}/build/xllm/core/server"

echo
echo "Verifying setup.py outputs..."
# The YuanRong KV cache transfer adapter is compiled directly into the xllm
# binary (kv_cache_transfer target), so libxllm_yuanrong_kv_cache_transfer.so
# is intentionally NOT produced any more. Only the pure-C-ABI DataSystem
# bridge remains a separate shared library.
artifacts=("${XLLM_SRC}/build/xllm/core/server/xllm")
if [[ "${BUILD_WITH_DATASYSTEM}" == "ON" ]]; then
  artifacts+=(
    "${official_build_dir}/xllm/core/server/libxllm_yuanrong_datasystem_bridge.so"
    "${server_output_dir}/libxllm_yuanrong_datasystem_bridge.so"
  )
fi
for artifact in "${artifacts[@]}"; do
  if [[ ! -f "${artifact}" ]]; then
    echo "ERROR: setup.py build did not produce expected artifact: ${artifact}" >&2
    exit 1
  fi
  echo "  OK ${artifact}"
done

echo
echo "Build finished."
echo "Expected outputs:"
echo "  ${XLLM_SRC}/build/xllm/core/server/xllm"
if [[ "${BUILD_WITH_DATASYSTEM}" == "ON" ]]; then
  echo "  ${official_build_dir}/xllm/core/server/libxllm_yuanrong_datasystem_bridge.so"
  echo "  ${server_output_dir}/libxllm_yuanrong_datasystem_bridge.so"
fi
echo
echo "Runtime symbol check (ldd -r must show no unresolved symbols):"
XLLM_LDD_OUTPUT="$(ldd -r "${XLLM_SRC}/build/xllm/core/server/xllm" 2>&1 || true)"
if grep -Eiq 'undefined symbol|not found' <<< "${XLLM_LDD_OUTPUT}"; then
  echo "${XLLM_LDD_OUTPUT}" >&2
  echo "ERROR: xllm has unresolved runtime symbols." >&2
  exit 1
fi
echo "  OK xllm has no unresolved runtime symbols."
if [[ "${BUILD_WITH_DATASYSTEM}" == "ON" ]]; then
  BRIDGE_LDD_OUTPUT="$(ldd -r "${server_output_dir}/libxllm_yuanrong_datasystem_bridge.so" 2>&1 || true)"
  if grep -Eiq 'undefined symbol|not found' <<< "${BRIDGE_LDD_OUTPUT}"; then
    echo "${BRIDGE_LDD_OUTPUT}" >&2
    echo "ERROR: libxllm_yuanrong_datasystem_bridge.so has unresolved runtime symbols." >&2
    exit 1
  fi
  echo "  OK libxllm_yuanrong_datasystem_bridge.so has no unresolved runtime symbols."
fi

bash "${SCRIPT_DIR}/verify_build.sh"
