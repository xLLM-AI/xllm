#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BIN_DIR="${SCRIPT_DIR}/../../../bin"

TEMP_DIR="xllm"
INCLUDE_DIR="${TEMP_DIR}/include"
LIB_DIR="${TEMP_DIR}/lib"

VERSION_FILE="${SCRIPT_DIR}/../../../version.txt"
TAR_BASE_NAME="xllm"
LOCAL_INSTALL_DIR="/usr/local"
LOCAL_TARGET_DIR="${LOCAL_INSTALL_DIR}/xllm"

HEADERS=("${SCRIPT_DIR}/../llm.h" "${SCRIPT_DIR}/../rec.h" "${SCRIPT_DIR}/../default.h" "${SCRIPT_DIR}/../types.h")
SO_FILES=(
    "${SCRIPT_DIR}/../../../build/xllm/core/server/libxllm.so"
)

error_exit() {
    echo -e "\033[31merror: $1\033[0m" >&2
    exit 1
}

cd_bin_dir() {
    if [ ! -d "${BIN_DIR}" ]; then
        mkdir -p "${BIN_DIR}" || error_exit "failed to create bin directory: ${BIN_DIR}"
    fi
    
    cd "${BIN_DIR}" || error_exit "failed to enter bin directory: ${BIN_DIR}"
}

read_version() {
    if [ ! -f "${VERSION_FILE}" ]; then
        error_exit "${VERSION_FILE} is not existed"
    fi
    
    VERSION=$(cat "${VERSION_FILE}" | tr -d '[:space:]')
    if [ -z "${VERSION}" ]; then
        error_exit "version content is empty"
    fi


    # get the full hash of the latest commit
    FULL_HASH=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    # get the first 7 characters of the commit hash
    if [ "${FULL_HASH}" = "unknown" ]; then
        SHORT_HASH="unknown"
    else
        SHORT_HASH="${FULL_HASH:0:7}"
    fi

    PLATFORM=$(case $(uname -m) in x86_64|i?86) echo "x86";; aarch64|arm64) echo "aarch64";; *) echo "unknown";; esac)
    
    TAR_FILE="lib${TAR_BASE_NAME}_${PLATFORM}_v${VERSION}_${SHORT_HASH}.tar.gz"
}

check_files() {
    for header in "${HEADERS[@]}"; do
        if [ ! -f "${header}" ]; then
            error_exit "${header} is not existed"
        fi
    done
    
    for so_file in "${SO_FILES[@]}"; do
        if [ ! -f "${so_file}" ]; then
            error_exit "${so_file} is not existed"
        fi
    done
}

create_dirs() {
    mkdir -p "${INCLUDE_DIR}" || error_exit "create include directory failed"
    mkdir -p "${LIB_DIR}" || error_exit "create lib directory failed"
}

copy_headers() {
    for header in "${HEADERS[@]}"; do
        cp -f "${header}" "${INCLUDE_DIR}/" || error_exit "copy ${header} failed"
    done
}

copy_so() {
    for so_file in "${SO_FILES[@]}"; do
        cp -f "${so_file}" "${LIB_DIR}/" || error_exit "copy ${so_file} failed"
    done
}

package_tar() {
    tar -czf "${TAR_FILE}" "${TEMP_DIR}" || error_exit "tar failed"
}

cleanup_temp() {
    rm -rf "${TEMP_DIR}" || error_exit "rm temp directory failed"
}

extract_to_local() {
    if [ ! -f "${TAR_FILE}" ]; then
        error_exit "${TAR_FILE} is not existed"
    fi
    
    if [ ! -d "${LOCAL_INSTALL_DIR}" ]; then
        error_exit "local install directory is not existed"
    fi
    
    if [ -d "${LOCAL_TARGET_DIR}" ]; then
        rm -rf "${LOCAL_TARGET_DIR}" || error_exit "rm old xllm directory failed"
    fi

    tar -xzf "${TAR_FILE}" -C "${LOCAL_INSTALL_DIR}" || error_exit "extract failed"
}

main() {
    cd_bin_dir
    read_version
    check_files
    create_dirs
    copy_headers
    copy_so
    package_tar
    cleanup_temp
    extract_to_local
    
    echo -e "install file: \033[33m${TAR_FILE}\033[0m"
    echo -e "install path: \033[33m/usr/local/${TEMP_DIR}\033[0m"
}

main