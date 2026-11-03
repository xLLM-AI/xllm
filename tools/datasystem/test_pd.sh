#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
backend="${1:-DataSystem}"
case "$backend" in
  DataSystem) export KV_BACKEND=DataSystem KV_TRANSFER_MODE=PULL ;;
  Mooncake) export KV_BACKEND=Mooncake KV_TRANSFER_MODE=PUSH ;;
  *) echo 'Usage: test_pd.sh DataSystem|Mooncake' >&2; exit 2 ;;
esac
mkdir -p "$ROOT/diagnostics"
LOG_DIR="$(mktemp -d "$ROOT/diagnostics/pd_${backend}_XXXXXX")"
export LOG_DIR
export STOP_AFTER_TEST=0 XLLM_KV_TRANSFER_DIAGNOSTICS=1
cleanup() {
  "$DS_PYTHON" "$ROOT/tools/datasystem/stop_installation.py" "$XLLM_BIN" "$XLLM_SERVICE_BIN" || true
}
trap cleanup EXIT
bash "$ROOT/tools/datasystem/smoke_pd.sh"
"$DS_PYTHON" "$ROOT/tools/datasystem/validate_smoke.py" "$LOG_DIR" "$backend"
echo "PD_ACCEPTANCE_OK backend=$backend logs=$LOG_DIR"
