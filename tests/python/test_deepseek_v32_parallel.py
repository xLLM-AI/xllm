# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""Parallel-layout tests for the DeepSeek-V3.2 Python model (DP/EP)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

_python_root = Path(__file__).parents[2] / "xllm" / "python"
_kernels_cuda = types.ModuleType("xllm.python.kernels_cuda")
_kernels_cuda.__path__ = [str(_python_root / "kernels_cuda")]
sys.modules.setdefault("xllm.python.kernels_cuda", _kernels_cuda)

from xllm.python import distributed, kernels  # noqa: E402

kernels.grouped_moe = MagicMock()
kernels.prepare_grouped_moe_weights = MagicMock(side_effect=lambda w13, w2: (w13, w2))
kernels.supports_cutlass_moe = MagicMock(return_value=False)
kernels.moe_fused_topk = MagicMock()
kernels.cutlass_fused_moe = MagicMock()
kernels.fused_moe = MagicMock()
kernels.dynamic_quant = MagicMock()
kernels.quant_matmul = MagicMock()
kernels.silu_and_mul = MagicMock()
distributed.all_gather_variable = MagicMock()
distributed.all_reduce_ = MagicMock()
distributed.all_gather = MagicMock(side_effect=lambda x, **kw: x)
distributed.tp_rank = MagicMock(return_value=0)

from xllm.python.model_executor.forward_context import (  # noqa: E402
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)
from xllm.python.models.deepseek_v32 import (  # noqa: E402
    DeepseekV3Config,
    DeepseekV3MoE,
)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _config(**overrides) -> DeepseekV3Config:
    values = {
        "hidden_size": 64,
        "n_layers": 2,
        "n_heads": 4,
        "head_dim": 16,
        "intermediate_size": 128,
        "vocab_size": 1024,
        "q_lora_rank": 32,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "index_n_heads": 4,
        "index_head_dim": 16,
        "index_topk": 64,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "n_routed_experts": 16,
        "n_shared_experts": 1,
        "num_experts_per_tok": 4,
        "n_group": 4,
        "topk_group": 2,
        "routed_scaling_factor": 2.5,
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "moe_intermediate_size": 32,
        "tp_size": 1,
        "tp_rank": 0,
        "ep_size": 1,
        "ep_rank": 0,
        "dp_size": 1,
        "dp_rank": 0,
        "moe_tp_size": 1,
        "moe_tp_rank": 0,
        "world_size": 1,
    }
    values.update(overrides)
    cfg = DeepseekV3Config.from_dict(values)
    cfg.enable_eplb = values.get("enable_eplb", False)
    cfg.redundant_experts_num = values.get("redundant_experts_num", 0)
    return cfg


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestDeepseekV3ConfigValidation:
    def test_pure_tp_valid(self):
        cfg = _config(tp_size=2, ep_size=1, world_size=2)
        cfg.validate()

    def test_ep2_valid(self):
        cfg = _config(ep_size=2, moe_tp_size=1, world_size=2)
        cfg.validate()

    def test_ep_equals_world_size_valid(self):
        cfg = _config(ep_size=4, moe_tp_size=1, world_size=4)
        cfg.validate()

    def test_ep_invalid_not_1_or_world(self):
        cfg = _config(ep_size=3, world_size=4)
        with pytest.raises(ValueError, match="ep_size must be 1 or world_size"):
            cfg.validate()

    def test_ep_experts_not_divisible(self):
        cfg = _config(n_routed_experts=15, ep_size=2, moe_tp_size=1, world_size=2)
        with pytest.raises(ValueError, match="divisible"):
            cfg.validate()

    def test_ep_moe_tp_world_mismatch(self):
        cfg = _config(ep_size=2, moe_tp_size=2, world_size=3)
        with pytest.raises(ValueError, match="world_size"):
            cfg.validate()


# ---------------------------------------------------------------------------
# MoE layer construction tests
# ---------------------------------------------------------------------------


def _make_moe(
    ep_size: int = 1,
    ep_rank: int = 0,
    dp_size: int = 1,
    dp_rank: int = 0,
    moe_tp_size: int = 1,
    n_experts: int = 16,
) -> DeepseekV3MoE:
    cfg = _config(
        n_routed_experts=n_experts,
        ep_size=ep_size,
        ep_rank=ep_rank,
        dp_size=dp_size,
        dp_rank=dp_rank,
        moe_tp_size=moe_tp_size,
        world_size=max(ep_size, 1) * dp_size,
    )
    return DeepseekV3MoE(cfg, layer_id=0, dtype=torch.float32, device=torch.device("cpu"))


class TestDeepseekV3MoEConstruction:
    def test_ep1_all_experts_local(self):
        moe = _make_moe(ep_size=1)
        assert moe.num_local_experts == 16
        assert moe.local_expert_start == 0
        assert moe.local_expert_end == 16

    def test_ep2_rank0_first_half(self):
        moe = _make_moe(ep_size=2, ep_rank=0)
        assert moe.num_local_experts == 8
        assert moe.local_expert_start == 0
        assert moe.local_expert_end == 8

    def test_ep2_rank1_second_half(self):
        moe = _make_moe(ep_size=2, ep_rank=1)
        assert moe.num_local_experts == 8
        assert moe.local_expert_start == 8
        assert moe.local_expert_end == 16

    def test_weight_shape_ep1(self):
        moe = _make_moe(ep_size=1, n_experts=16)
        assert moe.num_local_experts == 16

    def test_weight_shape_ep2(self):
        moe = _make_moe(ep_size=2, ep_rank=0, n_experts=16)
        assert moe.num_local_experts == 8

    def test_intermediate_tp_sharding(self):
        moe = _make_moe(ep_size=2, ep_rank=0, moe_tp_size=2)
        assert moe.inter_local == 32 // 2  # moe_intermediate_size // moe_tp_size


# ---------------------------------------------------------------------------
# MoE forward call tests
# ---------------------------------------------------------------------------


def _mock_forward_context(dp_token_counts=(4,), is_graph=False, dp_is_decode=None):
    metadata = SimpleNamespace(
        dp_token_counts=dp_token_counts,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    if dp_is_decode is not None:
        metadata.dp_is_decode = dp_is_decode
    execution_state = AclGraphExecutionState(persistent_buffers={}) if is_graph else None
    ctx = ForwardContext(
        attention_backend=MagicMock(),
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[],
        execution_state=execution_state,
    )
    return ctx


class TestDeepseekV3MoEForward:
    def setup_method(self):
        distributed.all_gather.reset_mock()
        distributed.all_gather.side_effect = lambda x, **kw: x.repeat(kw.get("world_size", 1), *([1] * (x.dim() - 1)))
        distributed.all_reduce_.reset_mock()
        kernels.grouped_moe.reset_mock()

    @staticmethod
    def _patch_shared_experts(moe: DeepseekV3MoE, num_tokens: int):
        """Replace shared_experts.forward to avoid W8A8 kernel calls."""
        moe.shared_experts.forward = MagicMock(return_value=torch.zeros(num_tokens, moe.hidden))

    def test_dp1_no_gather(self):
        moe = _make_moe(dp_size=1)
        hidden = torch.randn(4, 64)
        kernels.grouped_moe.return_value = torch.zeros(4, 64)
        self._patch_shared_experts(moe, 4)

        ctx = _mock_forward_context(dp_token_counts=(4,))
        with forward_context(ctx):
            moe.forward(hidden)

        distributed.all_gather.assert_not_called()

    def test_dp2_calls_gather(self):
        moe = _make_moe(dp_size=2, dp_rank=0)
        hidden = torch.randn(3, 64)
        # dp_token_counts=(3,4), padded_tokens=4, pad to [4,64], all_gather → [8,64]
        kernels.grouped_moe.return_value = torch.zeros(8, 64)
        self._patch_shared_experts(moe, 8)

        ctx = _mock_forward_context(dp_token_counts=(3, 4), is_graph=True)
        with forward_context(ctx):
            moe.forward(hidden)

        distributed.all_gather.assert_called_once()
        call_kwargs = distributed.all_gather.call_args[1]
        assert call_kwargs["dim"] == 0
        assert call_kwargs["world_size"] == 2
        assert call_kwargs["group_name"] == "dp"

    def test_ep2_calls_allreduce(self):
        moe = _make_moe(ep_size=2, ep_rank=0)
        hidden = torch.randn(4, 64)
        kernels.grouped_moe.return_value = torch.zeros(4, 64)
        self._patch_shared_experts(moe, 4)

        ctx = _mock_forward_context()
        with forward_context(ctx):
            moe.forward(hidden)

        reduce_calls = [c for c in distributed.all_reduce_.call_args_list if c[0][1] == "moe_ep"]
        assert len(reduce_calls) == 1

    def test_ep1_no_ep_allreduce(self):
        moe = _make_moe(ep_size=1)
        hidden = torch.randn(4, 64)
        kernels.grouped_moe.return_value = torch.zeros(4, 64)
        self._patch_shared_experts(moe, 4)

        ctx = _mock_forward_context()
        with forward_context(ctx):
            moe.forward(hidden)

        reduce_calls = [c for c in distributed.all_reduce_.call_args_list if len(c[0]) > 1 and c[0][1] == "moe_ep"]
        assert len(reduce_calls) == 0

    def test_moe_tp_calls_allreduce(self):
        moe = _make_moe(moe_tp_size=2, ep_size=2, ep_rank=0)
        hidden = torch.randn(4, 64)
        kernels.grouped_moe.return_value = torch.zeros(4, 64)
        self._patch_shared_experts(moe, 4)

        ctx = _mock_forward_context()
        with forward_context(ctx):
            moe.forward(hidden)

        reduce_calls = [c for c in distributed.all_reduce_.call_args_list if len(c[0]) > 1 and c[0][1] == "moe_tp"]
        assert len(reduce_calls) == 1

    def test_grouped_moe_active_range_ep2_rank1(self):
        moe = _make_moe(ep_size=2, ep_rank=1, n_experts=16)
        hidden = torch.randn(4, 64)
        kernels.grouped_moe.return_value = torch.zeros(4, 64)
        self._patch_shared_experts(moe, 4)

        ctx = _mock_forward_context()
        with forward_context(ctx):
            moe.forward(hidden)

        call_args = kernels.grouped_moe.call_args
        # grouped_moe positional signature: hidden, gating, w13, w2, w13_scale,
        # w2_scale, correction_bias, topk, topk_group, num_expert_groups,
        # renormalize, routed_scaling, active_expert_range — index 12 is active_expert_range.
        active_range = call_args[0][12]
        assert active_range == [8, 16]

    def test_dp2_output_sliced_to_local(self):
        moe = _make_moe(dp_size=2, dp_rank=1)
        hidden = torch.randn(4, 64)
        # dp_token_counts=(3,4), padded_tokens=4, pad_size=0, all_gather → [8,64]
        # dp_rank=1: narrow(0, 4, 4) → [4, 64]
        moe_output = torch.randn(8, 64)
        kernels.grouped_moe.return_value = moe_output
        self._patch_shared_experts(moe, 8)

        ctx = _mock_forward_context(dp_token_counts=(3, 4), is_graph=True)
        with forward_context(ctx):
            result = moe.forward(hidden)

        assert result.shape[0] == 4

    def test_dp2_eager_uses_compact_gather(self):
        moe = _make_moe(dp_size=2, dp_rank=0)
        hidden = torch.randn(3, 64)
        # eager mode: all_gather_variable returns compact [7, 64] (3+4 tokens)
        compact_output = torch.randn(7, 64)
        distributed.all_gather_variable.reset_mock()
        distributed.all_gather_variable.return_value = compact_output
        kernels.grouped_moe.return_value = torch.zeros(7, 64)
        self._patch_shared_experts(moe, 7)

        ctx = _mock_forward_context(dp_token_counts=(3, 4), is_graph=False, dp_is_decode=(1, 1))
        with forward_context(ctx):
            result = moe.forward(hidden)

        distributed.all_gather_variable.assert_called_once()
        distributed.all_gather.assert_not_called()
        # dp_rank=0: offset=0, narrow(0, 0, 3) → [3, 64]
        assert result.shape[0] == 3

    def test_dp2_eager_output_sliced_rank1(self):
        moe = _make_moe(dp_size=2, dp_rank=1)
        hidden = torch.randn(4, 64)
        # eager mode: all_gather_variable returns compact [7, 64] (3+4 tokens)
        compact_output = torch.randn(7, 64)
        distributed.all_gather_variable.reset_mock()
        distributed.all_gather_variable.return_value = compact_output
        moe_output = torch.randn(7, 64)
        kernels.grouped_moe.return_value = moe_output
        self._patch_shared_experts(moe, 7)

        ctx = _mock_forward_context(dp_token_counts=(3, 4), is_graph=False, dp_is_decode=(1, 1))
        with forward_context(ctx):
            result = moe.forward(hidden)

        # dp_rank=1: offset=sum([3])=3, narrow(0, 3, 4) → [4, 64]
        assert result.shape[0] == 4


# ---------------------------------------------------------------------------
# EPLB (Expert Parallel Load Balancing) tests
# ---------------------------------------------------------------------------


class TestEplbHelpers:
    """Unit tests for xllm.python.layers.eplb helper functions."""

    def test_build_initial_expert_ids_basic(self):
        from xllm.python.layers.eplb import build_initial_expert_ids

        ids = build_initial_expert_ids(num_total_experts=16, ep_size=2, device_experts_num=9, redundant_experts_num=1)
        assert len(ids) == 18
        assert ids[:8] == list(range(8))
        assert ids[8] == 7
        assert ids[9:17] == list(range(8, 16))
        assert ids[17] == 15

    def test_slice_rank_expert_ids(self):
        from xllm.python.layers.eplb import build_initial_expert_ids, slice_rank_expert_ids

        ids = build_initial_expert_ids(16, 2, 9, 1)
        rank0 = slice_rank_expert_ids(ids, 0, 9)
        rank1 = slice_rank_expert_ids(ids, 1, 9)
        assert len(rank0) == 9
        assert rank0 == ids[:9]
        assert rank1 == ids[9:]

    def test_build_log2phy_map_all_mapped(self):
        from xllm.python.layers.eplb import build_initial_expert_ids, build_log2phy_map

        ids = build_initial_expert_ids(16, 2, 9, 1)
        log2phy = build_log2phy_map(ids, 16, ep_rank=0)
        assert len(log2phy) == 16
        assert all(p >= 0 for p in log2phy)

    def test_build_log2phy_map_rotation(self):
        from xllm.python.layers.eplb import build_initial_expert_ids, build_log2phy_map

        ids = build_initial_expert_ids(16, 2, 9, 1)
        map_r0 = build_log2phy_map(ids, 16, ep_rank=0, moe_tp_rank_in_group=0)
        map_r0_tp1 = build_log2phy_map(ids, 16, ep_rank=0, moe_tp_rank_in_group=1)
        assert map_r0[7] != map_r0_tp1[7]

    def test_remap_expert_ids_tensor(self):
        from xllm.python.layers.eplb import remap_expert_ids

        log2phy = torch.tensor([5, 3, 1, 0, 4, 2], dtype=torch.int32)
        topk_ids = torch.tensor([[0, 2], [4, 5]], dtype=torch.int32)
        remapped = remap_expert_ids(topk_ids, log2phy)
        expected = torch.tensor([[5, 1], [4, 2]], dtype=torch.int32)
        assert torch.equal(remapped, expected)

    def test_expand_redundant_weight_storage(self):
        from xllm.python.layers.eplb import expand_redundant_weight_storage

        tensor = torch.randn(10, 4, 4)
        tensor[8] = torch.ones(4, 4) * 99.0
        expand_redundant_weight_storage(tensor, num_local_experts=9, device_experts_num=10)
        assert torch.equal(tensor[9], tensor[8])


class TestDeepseekV3MoEEplb:
    """Test DeepseekV3MoE with EPLB enabled."""

    def setup_method(self):
        distributed.all_gather.reset_mock()
        distributed.all_gather.side_effect = lambda x, **kw: x
        distributed.all_reduce_.reset_mock()
        kernels.grouped_moe.reset_mock()

    def test_eplb_moe_has_log2phy_map(self):
        cfg = _config(
            ep_size=2,
            ep_rank=0,
            moe_tp_size=1,
            world_size=2,
            enable_eplb=True,
            redundant_experts_num=1,
        )
        moe = DeepseekV3MoE(cfg, layer_id=0, dtype=torch.float32, device=torch.device("cpu"))
        assert hasattr(moe, "log2phy_map")
        assert moe.log2phy_map.shape[0] == 16
        # Slot-reuse: device_experts_num == num_local_experts (no extra slots)
        assert moe.device_experts_num == 8


# ---------------------------------------------------------------------------
# EPLB dynamic prepare/activate lifecycle tests
# ---------------------------------------------------------------------------


class TestEplbLifecycle:
    """Test the prepare/activate dynamic EPLB bridge."""

    def _make_causal_lm(self) -> DeepseekV3ForCausalLM:
        from xllm.python.models.deepseek_v32 import DeepseekV3ForCausalLM

        config = {
            "hidden_size": 64,
            "n_layers": 2,
            "n_heads": 4,
            "head_dim": 16,
            "intermediate_size": 128,
            "vocab_size": 1024,
            "q_lora_rank": 32,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "index_n_heads": 4,
            "index_head_dim": 16,
            "index_topk": 64,
            "first_k_dense_replace": 0,
            "moe_layer_freq": 1,
            "n_routed_experts": 16,
            "n_shared_experts": 1,
            "num_experts_per_tok": 4,
            "n_group": 4,
            "topk_group": 2,
            "routed_scaling_factor": 2.5,
            "topk_method": "noaux_tc",
            "norm_topk_prob": True,
            "moe_intermediate_size": 32,
            "tp_size": 1,
            "tp_rank": 0,
            "ep_size": 2,
            "ep_rank": 0,
            "dp_size": 1,
            "dp_rank": 0,
            "moe_tp_size": 1,
            "moe_tp_rank": 0,
            "world_size": 2,
            "enable_eplb": True,
            "redundant_experts_num": 1,
            "device": "cpu",
        }
        return DeepseekV3ForCausalLM(config, build_model=True)

    def test_prepare_sets_pending_state(self):
        from xllm.python.layers.eplb import build_initial_expert_ids

        model = self._make_causal_lm()
        moe_layers = model._moe_layers()
        assert len(moe_layers) >= 1

        # Simulate C++ EplbManager expert_ids (length = ep_size * cpp_device_experts_num)
        # cpp_device_experts_num = num_local + redundant = 8 + 1 = 9
        new_expert_ids = build_initial_expert_ids(16, 2, 9, 1)
        new_expert_ids[8] = 5  # swap redundant slot to expert 5

        model.prepare_expert_weight(0, new_expert_ids)
        moe = moe_layers[0]
        assert hasattr(moe, "_pending_log2phy")
        assert moe._pending_log2phy.shape[0] == 16

    def test_update_activates_new_map(self):
        from xllm.python.layers.eplb import build_initial_expert_ids

        model = self._make_causal_lm()
        moe = model._moe_layers()[0]
        old_map = moe.log2phy_map.clone()

        # Simulate C++ moving expert 10 into rank 0's slot 7 (replacing expert 7)
        # C++ expert_ids: rank 0 has 9 slots, rank 1 has 9 slots
        new_expert_ids = build_initial_expert_ids(16, 2, 9, 1)
        # Replace slot 7 on rank 0 with expert 10 (from rank 1)
        new_expert_ids[7] = 10

        model.prepare_expert_weight(0, new_expert_ids)
        model.start_expert_weight_transfer(0)
        model.update_expert_weight(0)

        # Expert 10 should now map to local slot 7
        assert moe.log2phy_map[10].item() == 7
        assert not hasattr(moe, "_pending_log2phy")

    def test_last_prepare_ok_returns_true(self):
        model = self._make_causal_lm()
        assert model.last_prepare_expert_weight_ok(0) is True
