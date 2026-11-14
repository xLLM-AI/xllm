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

case "${1:-}" in
    ""|--ensure-go) ;;
    --help)
        echo "Usage: bash docker/install_mooncake_dependencies.sh [--ensure-go]"
        exit 0
        ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MOONCAKE_SOURCE_DIR="${MOONCAKE_SOURCE_DIR:-${SCRIPT_DIR}/../third_party/Mooncake}"
INSTALL_PREFIX=$(python3 -c 'import os; print(os.path.abspath(os.path.expanduser(os.getenv("YALANTINGLIBS_PREFIX", "/usr/local"))))')
MULTIARCH=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("MULTIARCH") or "")')
LIBRARY_DIRS=(/usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64)
if [ -n "$MULTIARCH" ]; then
    LIBRARY_DIRS+=("/usr/lib/$MULTIARCH" "/usr/local/lib/$MULTIARCH")
fi
WORK_DIR=$(mktemp -d -t xllm-build-deps-XXXXXX)
trap 'rm -rf "$WORK_DIR"' EXIT

header_exists() {
    [ -f "/usr/include/$1" ] || [ -f "/usr/local/include/$1" ]
}

library_exists() {
    local directory
    for directory in "${LIBRARY_DIRS[@]}"; do
        if [ -f "$directory/$1" ]; then
            return 0
        fi
    done
    return 1
}

system_dependencies_ready() {
    header_exists zstd.h && library_exists libzstd.so &&
        header_exists liburing.h && library_exists liburing.so &&
        header_exists xxhash.h && library_exists libxxhash.so
}

install_system_dependencies() {
    if system_dependencies_ready; then
        echo "Mooncake system development libraries are already installed"
        return
    fi
    if command -v apt-get >/dev/null 2>&1; then
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends \
            libzstd-dev liburing-dev libxxhash-dev libmsgpack-cxx-dev
        apt-get clean
        rm -rf /var/lib/apt/lists/*
    elif command -v dnf >/dev/null 2>&1; then
        dnf install -y zstd-devel liburing-devel xxhash-devel
        dnf clean all
    elif command -v yum >/dev/null 2>&1; then
        yum install -y zstd-devel liburing-devel xxhash-devel
        yum clean all
    else
        echo "Unsupported package manager: apt-get, dnf or yum is required" >&2
        exit 1
    fi
}

install_msgpack() {
    if header_exists msgpack.hpp; then
        echo "msgpack-cxx headers are already installed"
        return
    fi
    if command -v apt-get >/dev/null 2>&1; then
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends libmsgpack-cxx-dev
        apt-get clean
        rm -rf /var/lib/apt/lists/*
        return
    fi
    echo "Installing msgpack-cxx cpp-6.1.0 headers"
    curl -fsSL --max-time 600 -o "$WORK_DIR/msgpack.tar.gz" \
        https://github.com/msgpack/msgpack-c/archive/refs/tags/cpp-6.1.0.tar.gz
    mkdir "$WORK_DIR/msgpack"
    tar -xzf "$WORK_DIR/msgpack.tar.gz" -C "$WORK_DIR/msgpack" --strip-components=1
    mkdir -p /usr/local/include
    cp -r "$WORK_DIR/msgpack/include/msgpack" /usr/local/include/
    install -m 0644 "$WORK_DIR/msgpack/include/msgpack.hpp" /usr/local/include/msgpack.hpp
}

install_yalantinglibs() {
    if [ -f "$INSTALL_PREFIX/lib/cmake/yalantinglibs/config.cmake" ]; then
        echo "yalantinglibs is already installed at $INSTALL_PREFIX"
        return
    fi
    local source="$MOONCAKE_SOURCE_DIR/extern/yalantinglibs"
    if [ ! -f "$source/CMakeLists.txt" ]; then
        echo "Initialize the pinned Mooncake/extern/yalantinglibs submodule first" >&2
        exit 1
    fi
    echo "Installing the pinned yalantinglibs at $INSTALL_PREFIX"
    cmake -S "$source" -B "$WORK_DIR/yalantinglibs" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF \
        -DYLT_ENABLE_IBV=ON
    cmake --build "$WORK_DIR/yalantinglibs" \
        --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-${VCPKG_BUILD_JOBS:-$(nproc)}}"
    cmake --install "$WORK_DIR/yalantinglibs"
}

go_version() {
    local output
    output=$("$1" version 2>/dev/null) || return 1
    [[ "$output" =~ go([0-9]+\.[0-9]+\.[0-9]+) ]] || return 1
    printf '%s\n' "${BASH_REMATCH[1]}"
}

go_satisfies() {
    local version
    version=$(go_version "$1") || return 1
    printf '%s\n%s\n' "$REQUIRED_GO_VERSION" "$version" | sort -V -C
}

ensure_go() {
    REQUIRED_GO_VERSION=$(sed -n 's/^GOVER=["'"'"']*\([0-9.]*\).*/\1/p' "$MOONCAKE_SOURCE_DIR/dependencies.sh")
    if [[ ! "$REQUIRED_GO_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Cannot read a Go version from $MOONCAKE_SOURCE_DIR/dependencies.sh" >&2
        exit 1
    fi
    local install_dir="${GO_INSTALL_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/xllm/go${REQUIRED_GO_VERSION}}"
    if command -v go >/dev/null 2>&1 && go_satisfies "$(command -v go)"; then
        echo "Go on PATH satisfies Mooncake's minimum version $REQUIRED_GO_VERSION"
        return
    fi
    if [ -x "$install_dir/bin/go" ] && go_satisfies "$install_dir/bin/go"; then
        echo "Go is already installed at $install_dir"
        return
    fi
    if [ -e "$install_dir" ]; then
        echo "Remove the incompatible or incomplete Go installation at $install_dir before retrying" >&2
        exit 1
    fi
    local arch
    case "$(uname -m)" in
        aarch64|arm64) arch=arm64 ;;
        x86_64|amd64) arch=amd64 ;;
        *) echo "Unsupported CPU architecture: $(uname -m)" >&2; exit 1 ;;
    esac
    local archive="v0.0.1-go${REQUIRED_GO_VERSION}.linux-${arch}.zip"
    echo "Installing Go $REQUIRED_GO_VERSION at $install_dir"
    curl -fsSL --max-time 900 -o "$WORK_DIR/go.zip" \
        "https://goproxy.cn/golang.org/toolchain/@v/$archive"
    unzip -q "$WORK_DIR/go.zip" -d "$WORK_DIR/go"
    local toolchain="$WORK_DIR/go/golang.org/toolchain@${archive%.zip}"
    [ "$(go_version "$toolchain/bin/go")" = "$REQUIRED_GO_VERSION" ]
    mkdir -p "$(dirname "$install_dir")"
    mv "$toolchain" "$install_dir"
}

if [ "${1:-}" = --ensure-go ]; then
    ensure_go
else
    install_system_dependencies
    install_msgpack
    install_yalantinglibs
    system_dependencies_ready && header_exists msgpack.hpp
fi
