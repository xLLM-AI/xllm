#!/bin/bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

IMAGE="registry.mthreads.com/presale/devtech/xllm:musa-cicd-20260820"
COMMAND="unset VCPKG_ROOT VCPKG_BINARY_SOURCES DEPENDENCES_ROOT FETCHCONTENT_SOURCE_DIR_VCPKG CMAKE_TOOLCHAIN_FILE
export VCPKG_DEFAULT_BINARY_CACHE=/root/.cache/vcpkg/archives
export VCPKG_DOWNLOADS=/root/.cache/vcpkg/downloads
$*"

docker run \
  --rm \
  --runtime=mthreads \
  --network=host \
  --env "GITHUB_WORKSPACE=${PWD}" \
  --env MAX_JOBS=16 \
  --env "CMAKE_ARGS=-DCMAKE_CUDA_COMPILER=/usr/local/musa/tools/musamapping/mcc_wrapper -DCMAKE_MODULE_PATH=/usr/local/musa/tools/musamapping/cmake/Modules" \
  --env "XLLM_HOST_GID=$(id -g)" \
  --env "XLLM_HOST_UID=$(id -u)" \
  --volume "${PWD}:${PWD}" \
  --volume /export/home/musa_vcpkg_cache:/root/.cache/vcpkg \
  --workdir "${PWD}" \
  --entrypoint /usr/local/bin/run-xllm-musa-ci \
  "${IMAGE}" run "${COMMAND}"
