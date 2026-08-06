# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

pytest.importorskip("torch_npu")

from xllm.python.models import deepseek_v32_mtp


def test_mtp_constructor_defers_shared_target_modules() -> None:
    config = {
        "device": "cpu",
        "dtype": "float32",
        "tp_size": 1,
        "tp_rank": 0,
    }
    model_config = SimpleNamespace(vocab_size=16)
    mtp_body = nn.Module()
    mtp_body.embed_tokens = None

    with (
        patch.object(
            deepseek_v32_mtp.DeepseekV3Config,
            "from_dict",
            return_value=model_config,
        ),
        patch.object(
            deepseek_v32_mtp.DeepseekV3ForCausalLM,
            "resolve_dtype",
            return_value=torch.float32,
        ),
        patch.object(
            deepseek_v32_mtp,
            "DeepseekV32MtpModel",
            return_value=mtp_body,
        ),
    ):
        draft = deepseek_v32_mtp.DeepseekV32MtpForCausalLM(config)

    assert draft.lm_head is None
    assert draft.model is mtp_body
    assert draft.model.embed_tokens is None

    target_lm_head = nn.Linear(4, 16, bias=False)
    target_embedding = nn.Embedding(16, 4)
    draft.lm_head = target_lm_head
    draft.model.embed_tokens = target_embedding

    assert draft.lm_head is target_lm_head
    assert draft.model.embed_tokens is target_embedding


def test_mtp_load_rejects_missing_required_weights() -> None:
    config = {
        "device": "cpu",
        "dtype": "float32",
        "tp_size": 1,
        "tp_rank": 0,
    }
    model_config = SimpleNamespace(vocab_size=16)
    mtp_body = nn.Module()
    mtp_body.embed_tokens = None

    with (
        patch.object(
            deepseek_v32_mtp.DeepseekV3Config,
            "from_dict",
            return_value=model_config,
        ),
        patch.object(
            deepseek_v32_mtp.DeepseekV3ForCausalLM,
            "resolve_dtype",
            return_value=torch.float32,
        ),
        patch.object(
            deepseek_v32_mtp,
            "DeepseekV32MtpModel",
            return_value=mtp_body,
        ),
        patch.object(deepseek_v32_mtp.DeepseekV3ForCausalLM, "load_weights"),
    ):
        draft = deepseek_v32_mtp.DeepseekV32MtpForCausalLM(config)
        with pytest.raises(KeyError, match="missing required MTP weight"):
            draft.load_weights([], tp_rank=0, tp_size=1)
