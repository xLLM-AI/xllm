# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Initialize the matching host once for native-adapter integration tests."""

import pytest

pytest.importorskip("torch_npu", reason="NPU integration tests require torch_npu")

from xllm import xllm_export  # noqa: F401
from xllm.python import initialize_runtime

initialize_runtime()
