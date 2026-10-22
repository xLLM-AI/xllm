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

"""Tests for the FlashInfer attention backend."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("FlashInfer tests require CUDA", allow_module_level=True)
pytest.importorskip("flashinfer", reason="FlashInfer is not installed")

from xllm.python.attention.flashinfer import FlashInferBackend, _should_use_tensor_core_decode


def test_tensor_core_decode_for_large_gqa_groups():
    assert _should_use_tensor_core_decode(torch.bfloat16, 24, 4)
    assert _should_use_tensor_core_decode(torch.float16, 32, 8)


def test_cuda_core_decode_for_small_gqa_groups_or_float32():
    assert not _should_use_tensor_core_decode(torch.bfloat16, 8, 4)
    assert not _should_use_tensor_core_decode(torch.float32, 24, 4)


@pytest.mark.parametrize("override,expected", [(None, False), (False, False), (True, True)])
def test_mha_decode_precision_choice_is_explicit(override: bool | None, expected: bool) -> None:
    backend = FlashInferBackend(
        2, 2, 128, 128**-0.5, 0, torch.device("cuda:0"), torch.bfloat16, use_tensor_core_decode=override
    )
    assert backend._decode_use_tensor_cores is expected
    assert backend._decode_wrapper.use_tensor_cores is expected


def test_float32_tensor_core_decode_is_rejected_before_allocation() -> None:
    with pytest.raises(ValueError, match="requires float16 or bfloat16"):
        FlashInferBackend(2, 2, 128, 128**-0.5, 0, torch.device("cuda:0"), torch.float32, use_tensor_core_decode=True)
