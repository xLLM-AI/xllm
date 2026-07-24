#!/bin/bash

# Color definitions
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTALL_PREFIX="/usr/local/yalantinglibs"

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

remove_dir_if_exists() {
    local dir_path="$1"
    local dir_name="$2"
    if [ -d "${dir_path}" ]; then
        print_warning "${dir_name} directory already exists. Removing for fresh install..."
        run_or_die "Failed to remove existing ${dir_name} directory" rm -rf "${dir_path}"
    fi
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

patch_yalantinglibs_easylog() {
    local easylog_header="$1"
    if [ -f "${easylog_header}" ]; then
        run_or_die \
            "Failed to patch yalantinglibs easylog severity default" \
            sed -i \
            '/std::atomic<Severity> min_severity_ =/,/#endif/c\  std::atomic<Severity> min_severity_ = Severity::WARN;' \
            "${easylog_header}"
    fi
}

install_yalantinglibs() {
    print_section "Installing yalantinglibs"

    local thirdparties_dir="${REPO_ROOT}/third_party/Mooncake/thirdparties"
    local yalanting_repo_url="https://gitcode.com/gh_mirrors/ya/yalantinglibs.git"
    local yalanting_source_dir="${thirdparties_dir}/yalantinglibs"
    local yalanting_build_dir="${yalanting_source_dir}/build"
    local yalanting_config_file="${INSTALL_PREFIX}/lib/cmake/yalantinglibs/config.cmake"
    local yalanting_easylog_header="${yalanting_source_dir}/include/ylt/easylog.hpp"

    ensure_dir "${thirdparties_dir}" "Failed to create Mooncake/thirdparties directory"
    remove_dir_if_exists "${yalanting_source_dir}" "yalantinglibs"

    echo "Cloning yalantinglibs from ${yalanting_repo_url}"
    run_or_die "Failed to clone yalantinglibs" git clone "${yalanting_repo_url}" "${yalanting_source_dir}"

    echo "Checking out yalantinglibs version 0.5.5..."
    run_or_die "Failed to checkout yalantinglibs version 0.5.5" git -C "${yalanting_source_dir}" checkout 0.5.5

    patch_yalantinglibs_easylog "${yalanting_easylog_header}"

    ensure_dir "${yalanting_build_dir}" "Failed to create build directory"

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

# Install system development headers required by Mooncake v0.3.12
# (mooncake-store links -lzstd, includes xxhash.h). The runtime libs
# already ship with the base image; only the -devel symlinks/headers
# are missing.
install_mooncake_system_devel() {
    print_section "Installing system devel packages required by Mooncake"
    run_or_die "Failed to install system devel packages" \
        yum install -y zstd-devel xxhash-devel
    print_success "System devel packages installed: zstd-devel xxhash-devel"
}

# Install the header-only msgpack-cxx headers that Mooncake v0.3.12's
# serializer.h expects. No matching yum package exists on the base
# image, so drop the headers into /usr/local/include/. Header-only,
# no build step needed.
install_msgpack_cxx_headers() {
    print_section "Installing msgpack-cxx headers"

    local version="cpp-6.1.0"
    local repo_url="https://gitcode.com/gh_mirrors/ms/msgpack-c.git"
    local work_dir
    work_dir="$(mktemp -d -t msgpack-cxx-XXXXXX)"

    echo "Cloning msgpack-c ${version} from ${repo_url}"
    run_or_die "Failed to clone msgpack-c" \
        git clone --depth 1 --branch "${version}" "${repo_url}" "${work_dir}/msgpack-c"

    echo "Installing msgpack.hpp and msgpack/ into /usr/local/include"
    run_or_die "Failed to install msgpack.hpp" \
        cp "${work_dir}/msgpack-c/include/msgpack.hpp" /usr/local/include/
    run_or_die "Failed to install msgpack/ headers" \
        cp -r "${work_dir}/msgpack-c/include/msgpack" /usr/local/include/

    rm -rf "${work_dir}"
    print_success "msgpack-cxx headers installed to /usr/local/include"
}

main() {
    ensure_dir "${INSTALL_PREFIX}" "Failed to create install directory: ${INSTALL_PREFIX}"
    echo -e "${YELLOW}Installing to: ${INSTALL_PREFIX}${NC}"

    install_mooncake_system_devel
    install_msgpack_cxx_headers
    install_yalantinglibs
}

main "$@"
