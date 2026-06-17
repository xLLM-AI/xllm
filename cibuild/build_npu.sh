#!/bin/bash
set -e

function error() {
  echo "Require build command, e.g. python setup.py build --device npu"
  exit 1
}

IMAGE="quay.io/jd_xllm/xllm-ai:xllm-dev-a2-x86-cann9-20260605"

XLLM_OPS_CACHE="/export/home/npu_xllm_ops_build_cache"
TRITON_BINARY_CACHE="/export/home/npu_triton_binary_cache"
mkdir -p "${XLLM_OPS_CACHE}" "${TRITON_BINARY_CACHE}"

RUN_OPTS=(
  --rm
  -t
  --privileged
  --ipc=host
  --network=host
  --device=/dev/davinci0
  --device=/dev/davinci_manager
  --device=/dev/devmm_svm
  --device=/dev/hisi_hdc
  -v /var/queue_schedule:/var/queue_schedule
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi
  -v /usr/local/sbin/:/usr/local/sbin/
  -v /export/home:/export/home
  -v /export/home/npu_vcpkg_cache_abi_1:/root/.cache/vcpkg # cached vcpkg installed dir
  -v "${XLLM_OPS_CACHE}":"${XLLM_OPS_CACHE}" # cached xllm_ops build dir
  -v "${TRITON_BINARY_CACHE}":"${TRITON_BINARY_CACHE}" # cached triton_npu binary dir
  -v /etc/hccn.conf:/etc/hccn.conf
  -w /export/home
  -e XLLM_OPS_BUILD_DIR="${XLLM_OPS_CACHE}"
  -e TRITON_BINARY_PATH="${TRITON_BINARY_CACHE}"
)

CMD="$*"
[[ -z "${CMD}" ]] && error

[[ ! -x $(command -v docker) ]] && echo "ERROR: 'docker' command is missing." && exit 1

docker run "${RUN_OPTS[@]}" "${IMAGE}" bash -c "set -euo pipefail; cd $(pwd); ${CMD}"
