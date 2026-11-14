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

case "$(uname -m)" in
    aarch64|arm64)
        GNU_PREFIX=aarch64-linux-gnu
        DEFAULT_VCPKG_TRIPLET=arm64-linux
        ;;
    x86_64|amd64)
        GNU_PREFIX=x86_64-linux-gnu
        DEFAULT_VCPKG_TRIPLET=x64-linux
        ;;
    *)
        echo "Unsupported CPU architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

export CARGO_HOME="${CARGO_HOME:-/root/.cargo}"
export VCPKG_ROOT="${VCPKG_ROOT:-/vcpkg}"
export VCPKG_TARGET_TRIPLET="${VCPKG_TARGET_TRIPLET:-$DEFAULT_VCPKG_TRIPLET}"

# install system dependencies
if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl wget git zip unzip tar xz-utils \
        ninja-build libgmp-dev libmpfr-dev libncurses-dev libibverbs-dev \
        libdbus-1-dev rpm libgoogle-perftools-dev libunwind-dev \
        nasm bison libx11-dev libxft-dev libxext-dev autoconf automake \
        libtool libxi-dev libxtst-dev libxrandr-dev locales \
        gdb ccache numactl libnuma-dev zstd libzstd-dev libxxhash-dev \
        libboost-all-dev cmake libssl-dev patchelf pkg-config libmsgpack-cxx-dev \
        python3-dev python3-pip zlib1g-dev rustc cargo \
        libgoogle-glog-dev libgtest-dev libjsoncpp-dev libjemalloc-dev \
        liburing-dev libnl-3-dev libnl-route-3-dev
    locale-gen zh_CN.UTF-8
    apt-get clean
    rm -rf /var/lib/apt/lists/*

    # Ubuntu does not package the secure-C runtime required by custom ops.
    mkdir -p /tmp/libboundscheck
    curl -fsSL --max-time 600 -o /tmp/libboundscheck.tar.gz \
        "https://github.com/openeuler-mirror/libboundscheck/archive/refs/tags/${LIBBOUNDSCHECK_VERSION}.tar.gz"
    tar -xzf /tmp/libboundscheck.tar.gz -C /tmp/libboundscheck --strip-components=1
    make -C /tmp/libboundscheck -j "${VCPKG_BUILD_JOBS}"
    install -m 0755 /tmp/libboundscheck/lib/libboundscheck.so /usr/local/lib/
    ldconfig
    rm -rf /tmp/libboundscheck /tmp/libboundscheck.tar.gz
elif command -v yum >/dev/null 2>&1; then
    yum install -y gcc-c++ ca-certificates curl wget git zip unzip tar xz \
        ninja-build gmp-devel mpfr-devel libtermcap-devel \
        libibverbs libibverbs-devel dbus-libs dbus-devel rpm-build \
        gperftools-libs gperftools-devel libunwind \
        nasm bison libX11-devel libXft-devel libXext-devel autoconf automake \
        libtool libXi-devel libXtst-devel libXrandr-devel glibc-langpack-zh \
        gdb ccache numactl numactl-libs numactl-devel zstd zstd-devel \
        xxhash-libs xxhash-devel boost-devel cmake make openssl-devel \
        patchelf pkgconf-pkg-config python3-devel zlib-devel rust cargo \
        glog-devel gtest-devel jsoncpp-devel jemalloc-devel liburing-devel \
        libnl3 libboundscheck
    yum clean all
    rm -rf /var/cache/yum
else
    echo "Unsupported operating system: apt-get or yum is required" >&2
    exit 1
fi

# install Rust and Cargo
mkdir -p "$CARGO_HOME"
cat > "$CARGO_HOME/config.toml" <<'EOF'
[source.crates-io]
replace-with = "rsproxy-sparse"

[source.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"

[net]
git-fetch-with-cli = true
EOF
chmod -R a+w "$CARGO_HOME"

# install vcpkg
git clone --branch "${VCPKG_VERSION}" https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"

# Autoconf ports expect GNU-prefixed native tool names on both distributions.
for tool in gcc g++ ar as ld nm objcopy objdump ranlib readelf size strings strip pkg-config; do
    tool_path=$(command -v "$tool")
    ln -s "$tool_path" "/usr/local/bin/${GNU_PREFIX}-$tool"
done

# Shared Mooncake dependencies use the pinned submodule sources and Go version.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$SCRIPT_DIR/install_mooncake_dependencies.sh"
GO_INSTALL_DIR=/usr/local/go bash "$SCRIPT_DIR/install_mooncake_dependencies.sh" --ensure-go

# set up vim and bashrc
cat >> /root/.vimrc <<'EOF'
set encoding=utf-8
set fileencodings=utf-8,gbk,gb2312,gb18030,latin1
set termencoding=utf-8
set fileformat=unix
set fileformats=unix,dos,mac
set nu
EOF

cat >> /root/.bashrc <<'EOF'

# vcpkg binary cache auto setup
for VCPKG_CACHE in ${VCPKG_DEFAULT_BINARY_CACHE:-} ${VCPKG_CACHE_DIR:-} /vcpkg-cache /export/home/vcpkg-cache /home/vcpkg-cache /workspace/vcpkg-cache /data/vcpkg-cache /mnt/vcpkg-cache; do
    if [ -n "${VCPKG_CACHE}" ] && [ -d "${VCPKG_CACHE}" ] && [ -w "${VCPKG_CACHE}" ] && [ -r "${VCPKG_CACHE}" ]; then
        export VCPKG_DEFAULT_BINARY_CACHE="${VCPKG_CACHE}"
        export VCPKG_BINARY_SOURCES="clear;files,/opt/xllm/vcpkg-cache,read;files,${VCPKG_CACHE},readwrite"
        break
    fi
done

# ccache dir auto setup
for CCACHE_SEARCH in ${CCACHE_DIR:-} /xllm-ccache /export/home/xllm-ccache /home/xllm-ccache /workspace/xllm-ccache /data/xllm-ccache /mnt/xllm-ccache; do
    if [ -n "${CCACHE_SEARCH}" ] && [ -d "${CCACHE_SEARCH}" ] && [ -w "${CCACHE_SEARCH}" ] && [ -r "${CCACHE_SEARCH}" ]; then
        export CCACHE_DIR="${CCACHE_SEARCH}"
        export CCACHE_NOHASHDIR="1"
        break
    fi
done

EOF
cp /root/.bashrc /etc/xllm-bashrc

export VCPKG_MAX_CONCURRENCY="${VCPKG_BUILD_JOBS}"
mkdir -p /opt/xllm/vcpkg-cache
"$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
VCPKG_BINARY_SOURCES="clear;files,/opt/xllm/vcpkg-cache,readwrite" \
    "$VCPKG_ROOT/vcpkg" install --triplet="${VCPKG_TARGET_TRIPLET}" \
    --x-manifest-root=/opt/xllm/vcpkg-manifest \
    --x-install-root=/tmp/xllm-vcpkg-installed
"$VCPKG_ROOT/vcpkg" list --x-install-root=/tmp/xllm-vcpkg-installed \
    > /opt/xllm/vcpkg-manifest/packages.txt
git -C "$VCPKG_ROOT" rev-parse HEAD > /opt/xllm/vcpkg-manifest/vcpkg.commit
rm -rf /tmp/xllm-vcpkg-installed "$VCPKG_ROOT/buildtrees" "$VCPKG_ROOT/packages"
