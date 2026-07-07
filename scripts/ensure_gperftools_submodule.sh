#!/usr/bin/env bash
# Initialize third_party/gperftools and run autogen.sh when configure is missing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPERFTOOLS_DIR="${REPO_ROOT}/third_party/gperftools"

git_repo() {
  git -c "safe.directory=${REPO_ROOT}" -c "safe.directory=${GPERFTOOLS_DIR}" \
    -C "${REPO_ROOT}" "$@"
}

if ! command -v git >/dev/null 2>&1; then
  echo "[gperftools] git is required" >&2
  exit 1
fi

echo "[gperftools] initializing submodule third_party/gperftools"
git_repo submodule update --init third_party/gperftools

if [[ ! -f "${GPERFTOOLS_DIR}/configure.ac" ]]; then
  echo "[gperftools] submodule source missing at ${GPERFTOOLS_DIR}" >&2
  exit 1
fi

if [[ ! -f "${GPERFTOOLS_DIR}/configure" ]]; then
  echo "[gperftools] running autogen.sh in ${GPERFTOOLS_DIR}"
  (cd "${GPERFTOOLS_DIR}" && ./autogen.sh)
fi

if [[ ! -f "${GPERFTOOLS_DIR}/configure" ]]; then
  echo "[gperftools] configure script missing after autogen.sh" >&2
  exit 1
fi

echo "[gperftools] ready at ${GPERFTOOLS_DIR}"
