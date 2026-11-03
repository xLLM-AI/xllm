#!/usr/bin/env bash
set -eo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
XLLM_BIN="${XLLM_BIN:-${ROOT}/build/xllm/core/server/xllm}"
BRIDGE_SO="${BRIDGE_SO:-${ROOT}/build/xllm/core/server/libxllm_yuanrong_datasystem_bridge.so}"
BUILD_WITH_MOONCAKE="${BUILD_WITH_MOONCAKE:-ON}"
BUILD_WITH_DATASYSTEM="${BUILD_WITH_DATASYSTEM:-ON}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "${XLLM_BIN}" ]] || fail "xllm binary not found: ${XLLM_BIN}"

if grep -aFq 'libxllm_yuanrong_kv_cache_transfer.so' "${XLLM_BIN}"; then
  fail "xllm still contains the removed YuanRong C++ plugin path; this is an old binary."
fi
if grep -aFq 'Failed to load YuanRong KV cache transfer plugin' "${XLLM_BIN}"; then
  fail "xllm still contains the old dlopen plugin implementation."
fi

if [[ "${BUILD_WITH_DATASYSTEM}" == "ON" ]]; then
  [[ -f "${BRIDGE_SO}" ]] || fail "DataSystem bridge not found: ${BRIDGE_SO}"
  grep -aFq 'YuanRong DataSystem requires --kv_cache_transfer_mode=PULL' "${XLLM_BIN}" ||
    fail "xllm does not contain the new DataSystem PULL guard; clean rebuild was not applied."
  BRIDGE_LDD_OUTPUT="$(ldd -r "${BRIDGE_SO}" 2>&1 || true)"
  if grep -Eiq 'undefined symbol|not found' <<< "${BRIDGE_LDD_OUTPUT}"; then
    echo "${BRIDGE_LDD_OUTPUT}" >&2
    fail "DataSystem bridge has unresolved runtime symbols or missing libraries."
  fi
fi

if [[ "${BUILD_WITH_MOONCAKE}" == "ON" ]]; then
  grep -aFq 'Mooncake' "${XLLM_BIN}" ||
    fail "Mooncake was requested but no Mooncake marker exists in xllm."
fi

echo "Backend build verification passed."
echo "  xllm: ${XLLM_BIN}"
echo "  Mooncake compiled: ${BUILD_WITH_MOONCAKE}"
echo "  DataSystem compiled: ${BUILD_WITH_DATASYSTEM}"
if [[ "${BUILD_WITH_DATASYSTEM}" == "ON" ]]; then
  echo "  DataSystem bridge: ${BRIDGE_SO}"
fi
