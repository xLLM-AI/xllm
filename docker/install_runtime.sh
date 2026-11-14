#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# TileLang/Triton can compile kernels and their host adapters during inference.
if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
        ca-certificates curl tar g++ ninja-build libgoogle-perftools4 \
        libibverbs1 libnuma1 libunwind8 libxxhash0 libzstd1 libnl-3-200
    apt-get clean
    rm -rf /var/lib/apt/lists/*
elif command -v yum >/dev/null 2>&1; then
    yum install -y ca-certificates curl tar gcc-c++ ninja-build \
        gperftools-libs libibverbs numactl-libs libunwind xxhash-libs zstd \
        libnl3 libboundscheck
    yum clean all
    rm -rf /var/cache/yum
else
    echo "Unsupported operating system: apt-get or yum is required" >&2
    exit 1
fi

case "$(uname -m)" in
    aarch64|arm64) BINARY_ARCH=arm64 ;;
    x86_64|amd64) BINARY_ARCH=amd64 ;;
    *) echo "Unsupported CPU architecture: $(uname -m)" >&2; exit 1 ;;
esac

# etcd is a runtime service; the Go compiler is only needed in development.
ETCD_DIR="etcd-v${ETCD_VERSION}-linux-${BINARY_ARCH}"
curl -fsSL --max-time 600 -o /tmp/etcd.tar.gz \
    "https://github.com/etcd-io/etcd/releases/download/v${ETCD_VERSION}/${ETCD_DIR}.tar.gz"
tar -xzf /tmp/etcd.tar.gz -C /tmp
install -m 0755 "/tmp/${ETCD_DIR}/etcd" "/tmp/${ETCD_DIR}/etcdctl" \
    "/tmp/${ETCD_DIR}/etcdutl" /usr/local/bin/
rm -rf /tmp/etcd.tar.gz "/tmp/${ETCD_DIR}"
