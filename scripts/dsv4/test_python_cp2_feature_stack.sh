#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:20050}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/dsv4_py_cp2_feature_stack_logs}"
TEST_DIR="${TEST_DIR:-$LOG_DIR/manual_test_$(date +%Y%m%d_%H%M%S)}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
mkdir -p "$TEST_DIR"

pass=0
fail=0
report() {
  if [[ "$1" == pass ]]; then
    pass=$((pass + 1))
    printf '[PASS] %s\n' "$2"
  else
    fail=$((fail + 1))
    printf '[FAIL] %s\n' "$2"
  fi
}

if curl -fsS --max-time 10 "$BASE_URL/health" > "$TEST_DIR/health.out"; then
  report pass health
else
  report fail "health ($BASE_URL/health)"
  exit 1
fi

make_payload() {
  local file="$1" max_tokens="$2" stream="$3" tag="$4"
  printf '{"model":"deepseek_v4","messages":[{"role":"user","content":"Reply with a short acknowledgment. test=%s"}],"max_tokens":%s,"temperature":0,"stream":%s}\n' \
    "$tag" "$max_tokens" "$stream" > "$file"
}

validate_json() {
  python3 - "$1" <<'PY'
import json
import sys
path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as f:
        value = json.load(f)
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("missing choices")
    if "error" in value:
        raise ValueError(value["error"])
except Exception as exc:
    print(f"invalid response: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

run_concurrent() {
  local max_tokens="$1" label="$2"
  local dir="$TEST_DIR/$label"
  mkdir -p "$dir"
  declare -a pids=()
  for i in $(seq 1 4); do
    make_payload "$dir/request_$i.json" "$max_tokens" false "${label}_$i"
    curl -fsS --connect-timeout 10 --max-time "$REQUEST_TIMEOUT" \
      -H 'Content-Type: application/json' \
      --data-binary "@$dir/request_$i.json" "$BASE_URL/v1/chat/completions" \
      > "$dir/response_$i.json" 2> "$dir/response_$i.err" &
    pids+=("$!")
  done
  local rc=0 pid i
  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    if ! wait "$pid"; then
      rc=1
    elif ! validate_json "$dir/response_$((i + 1)).json"; then
      rc=1
    fi
  done
  if ((rc == 0)); then
    report pass "4-way non-streaming max_tokens=$max_tokens"
  else
    report fail "4-way non-streaming max_tokens=$max_tokens; see $dir"
  fi
}

run_streaming() {
  local dir="$TEST_DIR/streaming"
  mkdir -p "$dir"
  declare -a pids=()
  for i in $(seq 1 4); do
    make_payload "$dir/request_$i.json" 1 true "stream_$i"
    curl -fsS -N --connect-timeout 10 --max-time "$REQUEST_TIMEOUT" \
      -H 'Content-Type: application/json' \
      --data-binary "@$dir/request_$i.json" "$BASE_URL/v1/chat/completions" \
      > "$dir/response_$i.sse" 2> "$dir/response_$i.err" &
    pids+=("$!")
  done
  local rc=0 pid i
  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    wait "$pid" || rc=1
    grep -q '\[DONE\]' "$dir/response_$((i + 1)).sse" || rc=1
  done
  if ((rc == 0)); then
    report pass '4-way streaming max_tokens=1'
  else
    report fail "4-way streaming max_tokens=1; see $dir"
  fi
}

run_concurrent 1 nonstream_1
run_concurrent 2 nonstream_2
run_concurrent 8 nonstream_8
run_streaming

vars_file="$TEST_DIR/vars.txt"
if curl -fsS --max-time 20 "$BASE_URL/vars" > "$vars_file"; then
  report pass '/vars'
  echo '--- selected counters ---'
  rg -i 'host_kv_(offload|restore)|request.*(success|failure)|success.*request|failure.*request' "$vars_file" || true
else
  report fail '/vars'
fi

echo '--- log error scan ---'
error_file="$TEST_DIR/log_errors.txt"
if compgen -G "$LOG_DIR/node_*.log" > /dev/null; then
  rg -a -i '507015|507018|HCCL.*(error|failed)|FATAL|Check failed|Traceback|out of memory|oom' \
    "$LOG_DIR"/node_*.log > "$error_file" || true
else
  : > "$error_file"
fi
if [[ ! -s "$error_file" ]]; then
  report pass 'node log error scan'
else
  report fail "node log error scan; see $error_file"
  sed -n '1,80p' "$error_file"
fi

echo "Results: pass=$pass fail=$fail"
echo "Artifacts: $TEST_DIR"
((fail == 0))
