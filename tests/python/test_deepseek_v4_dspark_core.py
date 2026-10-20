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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.models import deepseek_v4
from xllm.python.models.deepseek_v4_dspark import (
    DeepseekV4DSparkConfig,
    DeepseekV4DSparkForCausalLM,
    DeepseekV4DSparkModel,
)
from xllm.python.models.dspark import DSparkForCausalLMBase


def test_dspark_context_kv_validates_cache_and_hidden_shapes() -> None:
    model = DeepseekV4DSparkModel.__new__(DeepseekV4DSparkModel)
    torch.nn.Module.__init__(model)
    model.cfg = SimpleNamespace(n_layers=2, hidden_size=4, dspark_num_layers=2)
    target_hidden = torch.zeros((3, 8))

    with pytest.raises(ValueError, match="cache/layer count"):
        model.write_context_kv(target_hidden, torch.arange(3), torch.arange(3), [], None)

    model.cfg = SimpleNamespace(n_layers=1, hidden_size=4, dspark_num_layers=2)
    with pytest.raises(ValueError, match="target hidden width"):
        model.write_context_kv(torch.zeros((3, 4)), torch.arange(3), torch.arange(3), [object()], None)


@pytest.mark.parametrize("record_result", [True, False])
def test_dspark_context_kv_writes_real_decoder_layer_cache(
    monkeypatch: pytest.MonkeyPatch,
    record_result: bool,
) -> None:
    config = DeepseekV4DSparkConfig.from_dict(
        {
            "hidden_size": 4,
            "num_attention_heads": 1,
            "head_dim": 4,
            "vocab_size": 8,
            "q_lora_rank": 4,
            "qk_rope_head_dim": 2,
            "o_lora_rank": 2,
            "o_groups": 1,
            "max_position_embeddings": 8,
            "moe_intermediate_size": 4,
            "first_k_dense_replace": 2,
            "dspark_num_layers": 1,
            "dspark_target_layer_ids": [7],
            "dspark_markov_rank": 4,
            "dspark_block_size": 5,
        }
    )
    config.validate()
    model = DeepseekV4DSparkModel(config, torch.float32, torch.device("cpu"))
    model.main_proj = torch.nn.Linear(4, 4, bias=False)
    model.main_proj.weight.data.copy_(torch.eye(4))
    model.main_norm = torch.nn.Identity()
    attention = model.layers[0].self_attn
    attention.kv_proj = torch.nn.Linear(4, 4, bias=False)
    attention.kv_proj.weight.data.copy_(torch.eye(4))
    attention.kv_a_layernorm = torch.nn.Identity()
    monkeypatch.setattr(
        deepseek_v4.kernels,
        "npu_inplace_partial_rotary_mul",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    target_hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    swa_cache = torch.zeros((1, 8, 1, 4), dtype=torch.float32)
    synchronizer = SimpleNamespace(record_event=MagicMock(return_value=record_result))

    projected = model.write_context_kv(
        target_hidden,
        torch.arange(3),
        torch.tensor([1, 3, 5]),
        [LayerCache(key=None, value=None, swa=swa_cache)],
        synchronizer,
    )

    if record_result:
        torch.testing.assert_close(projected, target_hidden)
    else:
        assert projected is None
    torch.testing.assert_close(swa_cache.view(-1, 4)[[1, 3, 5]], target_hidden)
    synchronizer.record_event.assert_called_once_with(0)


def test_dspark_runtime_reuses_shared_head_base() -> None:
    assert issubclass(DeepseekV4DSparkForCausalLM, DSparkForCausalLMBase)
