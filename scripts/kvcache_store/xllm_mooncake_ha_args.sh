#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  case "${1:-}" in
    -h|--help|help)
      cat <<'EOF'
Set the required environment variables, then source this file:

  export MOONCAKE_ETCD_ENDPOINTS='10.0.0.1:2379;10.0.0.2:2379;10.0.0.3:2379'
  export MOONCAKE_CLUSTER_ID=xllm-mooncake
  export XLLM_STORE_LOCAL_HOSTNAME=10.0.3.31:12345
  source scripts/kvcache_store/xllm_mooncake_ha_args.sh
  /path/to/xllm --model=/path/to/model "${XLLM_MOONCAKE_HA_ARGS[@]}"

When executed directly, the script prints the generated Bash array.
EOF
      exit 0
      ;;
  esac
fi

configure_xllm_mooncake_ha_args() {
  local endpoints="${MOONCAKE_ETCD_ENDPOINTS:-}"
  endpoints="${endpoints#etcd://}"
  if [[ -z "$endpoints" ]]; then
    echo "ERROR: MOONCAKE_ETCD_ENDPOINTS is required" >&2
    return 1
  fi
  if [[ "$endpoints" == *"http://"* || "$endpoints" == *"https://"* ]]; then
    echo "ERROR: MOONCAKE_ETCD_ENDPOINTS must not include http:// or https://" >&2
    return 1
  fi
  if [[ -z "${XLLM_STORE_LOCAL_HOSTNAME:-}" ]]; then
    echo "ERROR: XLLM_STORE_LOCAL_HOSTNAME is required" >&2
    return 1
  fi

  MOONCAKE_CLUSTER_ID="${MOONCAKE_CLUSTER_ID:-xllm-mooncake}"
  if [[ ! "$MOONCAKE_CLUSTER_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: invalid MOONCAKE_CLUSTER_ID: $MOONCAKE_CLUSTER_ID" >&2
    return 1
  fi

  MOONCAKE_HA_ENTRY="etcd://$endpoints"
  export MC_STORE_CLUSTER_ID="$MOONCAKE_CLUSTER_ID"
  export MOONCAKE_HA_ENTRY

  XLLM_MOONCAKE_HA_ARGS=(
    "--enable_prefix_cache=true"
    "--host_blocks_factor=${XLLM_HOST_BLOCKS_FACTOR:-4}"
    "--enable_kvcache_store=true"
    "--store_protocol=${XLLM_STORE_PROTOCOL:-tcp}"
    "--store_master_server_address=$MOONCAKE_HA_ENTRY"
    "--store_metadata_server=${XLLM_STORE_METADATA_SERVER:-P2PHANDSHAKE}"
    "--store_local_hostname=$XLLM_STORE_LOCAL_HOSTNAME"
    "--prefetch_batch_size=${XLLM_PREFETCH_BATCH_SIZE:-8}"
    "--prefetch_timeout=${XLLM_PREFETCH_TIMEOUT:-30000}"
  )
}

if ! configure_xllm_mooncake_ha_args; then
  return 1 2>/dev/null || exit 1
fi
unset -f configure_xllm_mooncake_ha_args

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'export MC_STORE_CLUSTER_ID=%q\n' "$MC_STORE_CLUSTER_ID"
  printf 'export MOONCAKE_HA_ENTRY=%q\n' "$MOONCAKE_HA_ENTRY"
  printf 'XLLM_MOONCAKE_HA_ARGS=(\n'
  for argument in "${XLLM_MOONCAKE_HA_ARGS[@]}"; do
    printf '  %q\n' "$argument"
  done
  printf ')\n'
fi
