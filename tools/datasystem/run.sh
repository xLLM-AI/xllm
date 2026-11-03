#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
exec "$@"
