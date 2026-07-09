#!/bin/bash
set -ex

function error() {
  echo "Require build command, e.g. python setup.py build --device ilu"
  exit 1
}

REGISTRY="registry.iluvatar.com.cn:10443/infra"
COREX_VERSION="4.4.0.20251229"
IMAGE="${REGISTRY}/xllm-builder:${COREX_VERSION}-ubuntu22.04-py310-xllm-x86_64"

RUN_OPTS=(
  --rm
  -t
  --privileged
  --ipc=host
  --network=host
  -v /export/home:/export/home
  -v /export/home/ilu_vcpkg_cache:/root/.cache/vcpkg # cached vcpkg installed dir
  -w /export/home
)

if [[ -d /data/export-home ]]; then
  RUN_OPTS+=(-v /data/export-home:/data/export-home)
fi

CMD="$*"
[[ -z "${CMD}" ]] && error

[[ ! -x $(command -v docker) ]] && echo "ERROR: 'docker' command is missing." && exit 1

WORKDIR="$(pwd)"
docker run "${RUN_OPTS[@]}" "${IMAGE}" bash -c "set -euo pipefail; cd \"${WORKDIR}\"; ${CMD}"
