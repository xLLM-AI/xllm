# Copyright 2026 The xLLM Authors.
#
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

"""MLA model-facing aliases use the package-owned projection bindings."""

import pytest
from xllm_kernel.ops.gemm import matmul_ein_sum, matmul_ein_sum_out

pytest.importorskip("torch_npu")

from xllm.python import kernels
from xllm.python.kernels_npu import linear


def test_mla_projection_aliases() -> None:
    assert kernels.atb_matmul_ein_sum is linear.atb_matmul_ein_sum is matmul_ein_sum
    assert kernels.atb_matmul_ein_sum_out is linear.atb_matmul_ein_sum_out is matmul_ein_sum_out
