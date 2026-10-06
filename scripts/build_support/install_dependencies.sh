#!/bin/bash
# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Color definitions
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INSTALL_PREFIX="$(python3 -c 'import os; print(os.path.abspath(os.path.expanduser(os.getenv("YALANTINGLIBS_PREFIX", "/usr/local"))))')"
GOVER="$(sed -n 's/^GOVER=["\x27]*\([0-9.]*\).*/\1/p' "${REPO_ROOT}/third_party/Mooncake/dependencies.sh")"
GO_INSTALL_DIR="${XDG_CACHE_HOME:-${HOME}/.cache}/xllm/go${GOVER}"

print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
    exit 1
}

run_or_die() {
    local error_message="$1"
    shift
    "$@"
    if [ $? -ne 0 ]; then
        print_error "$error_message"
    fi
}

ensure_dir() {
    local dir_path="$1"
    local error_message="$2"
    if [ ! -d "${dir_path}" ]; then
        run_or_die "${error_message}" mkdir -p "${dir_path}"
    fi
}

installed_go_version() {
    local go_binary="$1"
    "${go_binary}" version 2>/dev/null | sed -n 's/.* go\([0-9][0-9.]*\).*/\1/p'
}

install_go() {
    local arch
    local go_tarball
    local work_dir
    local download_success=false
    local url
    local download_urls

    arch="$(uname -m)"
    case "${arch}" in
        aarch64)
            arch="arm64"
            ;;
        x86_64)
            arch="amd64"
            ;;
        *)
            print_error "Unsupported architecture: ${arch}"
            ;;
    esac

    go_tarball="go${GOVER}.linux-${arch}.tar.gz"
    ensure_dir "$(dirname "${GO_INSTALL_DIR}")" "Failed to create toolchain cache"
    work_dir="$(mktemp -d "$(dirname "${GO_INSTALL_DIR}")/.go-install-XXXXXX")" || print_error "Failed to create download directory"
    download_urls=(
        "https://go.dev/dl/${go_tarball}"
        "https://golang.google.cn/dl/${go_tarball}"
        "https://mirrors.aliyun.com/golang/${go_tarball}"
    )

    for url in "${download_urls[@]}"; do
        echo "Downloading Go ${GOVER} from ${url}..."
        if wget -q --show-progress --timeout=30 --tries=2 \
            -O "${work_dir}/${go_tarball}" "${url}"; then
            download_success=true
            break
        fi
        print_warning "Failed to download from ${url}, trying next mirror..."
    done

    if [ "${download_success}" = false ]; then
        rm -rf "${work_dir}"
        print_error "Failed to download Go ${GOVER} from all mirrors"
    fi

    run_or_die "Failed to extract Go ${GOVER}" tar -C "${work_dir}" -xzf "${work_dir}/${go_tarball}"
    if [ "$(installed_go_version "${work_dir}/go/bin/go")" != "${GOVER}" ]; then
        print_error "Downloaded Go toolchain version does not match ${GOVER}"
    fi
    if [ -e "${GO_INSTALL_DIR}" ]; then
        print_error "Incomplete toolchain at ${GO_INSTALL_DIR}; remove it before retrying"
    fi
    run_or_die "Failed to install Go ${GOVER}" mv -T "${work_dir}/go" "${GO_INSTALL_DIR}"
    rm -rf "${work_dir}"
    print_success "Go ${GOVER} installed successfully to ${GO_INSTALL_DIR}"
}

ensure_go() {
    local go_binary
    local go_version

    go_binary="$(command -v go || true)"
    if [ -n "${go_binary}" ]; then
        go_version="$(installed_go_version "${go_binary}")"
        if [ "${go_version}" = "${GOVER}" ]; then
            print_success "Go ${GOVER} is already installed"
            return
        fi
    fi

    if [ -x "${GO_INSTALL_DIR}/bin/go" ] && \
        [ "$(installed_go_version "${GO_INSTALL_DIR}/bin/go")" = "${GOVER}" ]; then
        print_success "Go ${GOVER} is already installed"
        return
    fi

    print_section "Installing Go ${GOVER} for Mooncake HA"
    install_go
}

patch_yalantinglibs_config() {
    local config_file="$1"
    if [ -f "${config_file}" ]; then
        run_or_die \
            "Failed to patch yalantinglibs config.cmake" \
            sed -i \
            's|target_link_libraries(${ylt_target_name} -libverbs)|target_link_libraries(${ylt_target_name} INTERFACE -libverbs)|' \
            "${config_file}"
    fi
}

install_yalantinglibs() {
    print_section "Installing yalantinglibs"

    local yalanting_source_dir="${REPO_ROOT}/third_party/Mooncake/extern/yalantinglibs"
    local yalanting_build_dir="${REPO_ROOT}/build/dependencies/yalantinglibs"
    local yalanting_config_file="${INSTALL_PREFIX}/lib/cmake/yalantinglibs/config.cmake"

    if [ ! -f "${yalanting_source_dir}/CMakeLists.txt" ]; then
        print_error "Initialize the pinned Mooncake/extern/yalantinglibs submodule first"
    fi

    echo "Configuring yalantinglibs..."
    run_or_die \
        "Failed to configure yalantinglibs" \
        cmake \
        -S "${yalanting_source_dir}" \
        -B "${yalanting_build_dir}" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARK=OFF \
        -DBUILD_UNIT_TESTS=OFF \
        -DYLT_ENABLE_IBV=ON

    echo "Building yalantinglibs (using $(nproc) cores)..."
    run_or_die "Failed to build yalantinglibs" cmake --build "${yalanting_build_dir}" -j"$(nproc)"

    echo "Installing yalantinglibs..."
    run_or_die "Failed to install yalantinglibs" cmake --install "${yalanting_build_dir}"

    patch_yalantinglibs_config "${yalanting_config_file}"
    print_success "yalantinglibs installed successfully to ${INSTALL_PREFIX}"
}

header_exists() {
    local header_name="$1"
    [ -f "/usr/include/${header_name}" ] || \
        [ -f "/usr/local/include/${header_name}" ]
}

library_exists() {
    local library_name="$1"
    find \
        /usr/lib \
        /usr/lib64 \
        /usr/local/lib \
        /usr/local/lib64 \
        -name "${library_name}" \
        -print \
        -quit 2>/dev/null | grep -q .
}

system_dependencies_ready() {
    header_exists "zstd.h" && \
        library_exists "libzstd.so" && \
        header_exists "xxhash.h" && \
        library_exists "libxxhash.so" && \
        header_exists "msgpack.hpp"
}

install_system_dependencies() {
    print_section "Installing system development packages required by Mooncake"

    if command -v apt-get >/dev/null 2>&1; then
        run_or_die "Failed to update apt package metadata" \
            env DEBIAN_FRONTEND=noninteractive apt-get update
        run_or_die "Failed to install apt development packages" \
            env DEBIAN_FRONTEND=noninteractive apt-get install -y \
                libzstd-dev \
                libxxhash-dev \
                libmsgpack-cxx-dev
        print_success \
            "Installed apt packages: libzstd-dev libxxhash-dev libmsgpack-cxx-dev"
        return
    fi

    if command -v dnf >/dev/null 2>&1; then
        run_or_die "Failed to install dnf development packages" \
            dnf --disablerepo=update install -y zstd-devel xxhash-devel
        print_success "Installed dnf packages: zstd-devel xxhash-devel"
        return
    fi

    if command -v yum >/dev/null 2>&1; then
        run_or_die "Failed to install yum development packages" \
            yum install -y zstd-devel xxhash-devel
        print_success "Installed yum packages: zstd-devel xxhash-devel"
        return
    fi

    print_error \
        "Unsupported package manager. Install zstd, xxHash, and msgpack-cxx development packages manually."
}

install_msgpack_cxx_headers() {
    if header_exists "msgpack.hpp"; then
        print_success "msgpack-cxx headers are already installed"
        return
    fi

    print_section "Installing msgpack-cxx headers"

    local version="cpp-6.1.0"
    local repo_url="https://gitcode.com/gh_mirrors/ms/msgpack-c.git"
    local work_dir
    work_dir="$(mktemp -d -t msgpack-cxx-XXXXXX)"

    echo "Cloning msgpack-c ${version} from ${repo_url}"
    run_or_die "Failed to clone msgpack-c" \
        git clone --depth 1 --branch "${version}" "${repo_url}" "${work_dir}/msgpack-c"

    echo "Installing msgpack.hpp and msgpack/ into /usr/local/include"
    run_or_die "Failed to install msgpack/ headers" \
        cp -r "${work_dir}/msgpack-c/include/msgpack" /usr/local/include/
    local marker
    marker="$(mktemp /usr/local/include/.msgpack.hpp-XXXXXX)" || print_error "Failed to create header marker"
    run_or_die "Failed to stage msgpack.hpp" cp "${work_dir}/msgpack-c/include/msgpack.hpp" "${marker}"
    run_or_die "Failed to publish msgpack.hpp" mv -f "${marker}" /usr/local/include/msgpack.hpp

    rm -rf "${work_dir}"
    print_success "msgpack-cxx headers installed to /usr/local/include"
}

install_build_dependencies() {
    ensure_dir "${INSTALL_PREFIX}" "Failed to create install directory: ${INSTALL_PREFIX}"
    echo -e "${YELLOW}Installing to: ${INSTALL_PREFIX}${NC}"

    if system_dependencies_ready; then
        print_success "Mooncake system development dependencies are already installed"
    else
        install_system_dependencies
    fi

    install_msgpack_cxx_headers

    if [ -f "${INSTALL_PREFIX}/lib/cmake/yalantinglibs/config.cmake" ]; then
        print_success "yalantinglibs is already installed to ${INSTALL_PREFIX}"
    else
        install_yalantinglibs
    fi

    if ! system_dependencies_ready; then
        print_error \
            "Mooncake dependencies are incomplete after installation. Check zstd, xxHash, and msgpack-cxx development files."
    fi
}

main() {
    case "${1:-}" in
        --ensure-go)
            if [ -z "${GOVER}" ]; then
                print_error "Cannot read Go version from Mooncake/dependencies.sh"
            fi
            ensure_go
            ;;
        "")
            install_build_dependencies
            ;;
        *)
            print_error "Unknown argument: $1"
            ;;
    esac
}

main "$@"
