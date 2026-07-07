#!/usr/bin/env bash
# Build bundled shared libtcmalloc.so.4 from third_party/gperftools submodule.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${REPO_ROOT}/build/cmake.linux-aarch64-cpython-311}"
STANDALONE_BUILD="${BUILD_DIR}/_gperftools_build"

bash "${REPO_ROOT}/scripts/ensure_gperftools_submodule.sh"

mkdir -p "${STANDALONE_BUILD}"
cat > "${STANDALONE_BUILD}/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.26)
project(xllm_tcmalloc_build NONE)
set(XLLM_SOURCE_DIR "${REPO_ROOT}" CACHE PATH "" FORCE)
list(APPEND CMAKE_MODULE_PATH "${REPO_ROOT}/cmake")
include(gperftools)
EOF

cmake -S "${STANDALONE_BUILD}" -B "${STANDALONE_BUILD}"
cmake --build "${STANDALONE_BUILD}" -j"$(nproc)"

INSTALL_LIB="${BUILD_DIR}/third_party/gperftools/install/lib"
mkdir -p "${INSTALL_LIB}"
cp -P "${STANDALONE_BUILD}/third_party/gperftools/install/lib/libtcmalloc.so"* "${INSTALL_LIB}/"

echo "Bundled tcmalloc installed to ${INSTALL_LIB}"
ls -la "${INSTALL_LIB}"/libtcmalloc.so*
