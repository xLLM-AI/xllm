#!/bin/bash
set -e

function error() {
  echo "Require build command, e.g. python setup.py build --device mlu"
  exit 1
}

IMAGE="cambricon-xllm:v26.07.14-torch2.10.0-torchmlu1.32.2-ubuntu24.04-py312-mlu-dev-x86"

RUN_OPTS=(
  --rm
  -t
  --privileged
  --ipc=host
  --network=host
  --pid=host
  --shm-size '128gb'
  -v /export/home:/export/home
  -v /export/home/mlu_vcpkg_cache:/root/.cache/vcpkg # cached vcpkg installed dir
  -w /export/home
)

if [[ -d /data/export-home ]]; then
  RUN_OPTS+=(-v /data/export-home:/data/export-home)
fi

if [[ -e /usr/bin/cnmon ]]; then
  RUN_OPTS+=(-v /usr/bin/cnmon:/usr/bin/cnmon)
fi

CMD="$*"
[[ -z "${CMD}" ]] && error

[[ ! -x $(command -v docker) ]] && echo "ERROR: 'docker' command is missing." && exit 1

WORKDIR="$(pwd)"
docker run "${RUN_OPTS[@]}" "${IMAGE}" bash -c "set -euo pipefail; cd \"${WORKDIR}\"; ${CMD}"
