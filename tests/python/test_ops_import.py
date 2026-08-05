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

"""Import isolation and graph-contract tests for public Python operators."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from xllm.python import platform

_REPO_ROOT = Path(__file__).parents[2]

_COMMON_OP_SCHEMAS = (
    "rms_norm(Tensor input, Tensor weight, float eps) -> Tensor",
    "fused_add_rms_norm(Tensor(a!) input, Tensor(b!) residual, Tensor weight, "
    "float eps) -> (Tensor, Tensor)",
    "silu_and_mul(Tensor input) -> Tensor",
    "fused_qk_norm_rope(Tensor(a!) qkv, int num_heads_q, int num_heads_k, "
    "int num_heads_v, int head_dim, float eps, Tensor q_weight, Tensor k_weight, "
    "Tensor cos_sin_cache, bool interleaved, Tensor position_ids) -> Tensor",
    "reshape_paged_cache(Tensor slot_mapping, Tensor keys, Tensor values, "
    "Tensor(a!) key_cache, Tensor(b!) value_cache) -> Tensor",
    "update_decode_graph_metadata(Tensor tokens, Tensor positions, Tensor "
    "slot_mapping, Tensor kv_seq_lens, Tensor paged_kv_indptr, Tensor "
    "paged_kv_indices, Tensor paged_kv_last_page_len, Tensor(a!) dst_tokens, "
    "Tensor(b!) dst_positions, Tensor(c!) dst_slot_mapping, Tensor(d!) "
    "dst_kv_seq_lens, Tensor(e!) dst_kv_seq_lens_delta, Tensor(f!) "
    "dst_paged_kv_indptr, Tensor(g!) dst_paged_kv_indices, Tensor(h!) "
    "dst_paged_kv_last_page_len, int padded_num_tokens) -> Tensor",
)

_NPU_OP_SCHEMAS = (
    "quant_matmul(Tensor x1, Tensor x2, bool transpose2, Tensor scale, "
    "Tensor? offset, Tensor? pertoken_scale, Tensor? bias, ScalarType? "
    "output_dtype) -> Tensor",
    "quantize_per_tensor(Tensor self, Tensor scales, Tensor zero_points, "
    "ScalarType dtype, int axis) -> Tensor",
    "dynamic_quant(Tensor input, Tensor? smooth_scales, Tensor? group_index, "
    "ScalarType? dst_type) -> (Tensor, Tensor?)",
    "lightning_indexer(Tensor query, Tensor key, Tensor weights, Tensor? "
    "query_seq_lengths, Tensor? key_seq_lengths, Tensor? block_table, str "
    "layout_query, str layout_key, int selected_count, int sparse_mode, int "
    "pre_tokens, int next_tokens, bool return_value) -> Tensor",
    "scatter_nd_update(Tensor(a!) var, Tensor indices, Tensor updates) -> ()",
    "sparse_flash_attention(Tensor query, Tensor key, Tensor value, Tensor "
    "sparse_indices, Tensor? block_table, Tensor? actual_seq_lengths_query, "
    "Tensor? actual_seq_lengths_kv, Tensor? query_rope, Tensor? key_rope, "
    "float scale_value, int sparse_block_size, str layout_query, str "
    "layout_kv, int sparse_mode) -> Tensor",
)


def _run_isolated_python(script: str, schemas: tuple[str, ...] = ()) -> None:
    definitions = [
        "import torch",
        'library = torch.library.Library("xllm_ops", "DEF")',
        *(f"library.define({schema!r})" for schema in schemas),
    ]
    program = "\n".join((*definitions, textwrap.dedent(script)))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(_REPO_ROOT), env.get("PYTHONPATH", "")) if value
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_platform_queries_are_no_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "current_platform", lambda: "npu")
    assert platform.is_npu() and not platform.is_gpu()
    monkeypatch.setattr(platform, "current_platform", lambda: "cuda")
    assert platform.is_gpu() and not platform.is_npu()


def test_public_imports_are_lazy_and_platform_isolated() -> None:
    _run_isolated_python(
        """
        import importlib
        import sys

        from xllm.python import platform

        selected_platform = platform.current_platform()
        if selected_platform == "npu":
            forbidden = (
                "flashinfer",
                "xllm.python.kernels.cuda",
                "xllm.python.ops.cuda",
            )
        elif selected_platform == "cuda":
            forbidden = (
                "torch_npu",
                "xllm.python.kernels.npu",
                "xllm.python.ops.npu",
            )
        else:
            forbidden = (
                "flashinfer",
                "torch_npu",
                "xllm.python.kernels.cuda",
                "xllm.python.kernels.npu",
                "xllm.python.ops.cuda",
                "xllm.python.ops.npu",
            )

        def matching_modules():
            return {
                name
                for name in sys.modules
                if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
            }

        baseline = matching_modules()
        registry = importlib.import_module("xllm.python.registry")
        assert {"qwen3", "deepseek_v32", "glm_moe_dsa"} <= registry._REGISTRY.keys()
        assert not any(name.startswith("xllm.python.models.") for name in sys.modules)

        qwen3_cls = registry.get_model_class("qwen3")
        assert qwen3_cls.__name__ == "Qwen3ForCausalLM"
        assert "xllm.python.models.deepseek_v32" not in sys.modules
        assert "xllm.python.models.glm5_2" not in sys.modules

        for module_name in (
            "xllm.python.ops",
            "xllm.python.models.deepseek_v32",
            "xllm.python.models.glm5_2",
            "xllm.python.layers.linear",
        ):
            importlib.import_module(module_name)

        assert matching_modules() == baseline
        assert "xllm.python.ops.npu_compute" not in sys.modules
        """,
        _COMMON_OP_SCHEMAS + _NPU_OP_SCHEMAS,
    )


def test_native_ops_fake_tensor_and_mutation_contracts() -> None:
    _run_isolated_python(
        """
        import xllm.python.ops as ops

        mode = torch._subclasses.fake_tensor.FakeTensorMode()
        with mode:
            output = ops.quant_matmul(
                torch.empty(2, 16),
                torch.empty(16, 4),
                False,
                torch.empty(4),
                None,
                None,
                None,
                torch.bfloat16,
            )
            assert output.shape == (2, 4)
            assert output.dtype == torch.bfloat16

            quantized, scale = ops.dynamic_quant(
                torch.empty(2, 16), dst_type=torch.quint4x2
            )
            assert quantized.shape == (2, 2)
            assert quantized.dtype == torch.int32
            assert scale.shape == (2,)
            assert scale.dtype == torch.float32

            query = torch.empty(8, 4, 16)
            key = torch.empty(8, 2, 16)
            indices = ops.lightning_indexer(
                query, key, torch.empty(4, 2), None, None, None,
                "TND", "TND", 3, 0, 0, 0, False,
            )
            assert indices.shape == (8, 2, 3)
            sparse = ops.sparse_flash_attention(
                query, key, torch.empty_like(key), indices, None, None, None,
                None, None, 1.0, 128, "TND", "TND", 0,
            )
            assert sparse.shape == query.shape

            destination = torch.empty(8, 2, 16)
            assert ops.scatter_nd_update(
                destination,
                torch.empty(2, 1, dtype=torch.int64),
                torch.empty(2, 2, 16),
            ) is None

            try:
                ops.dynamic_quant(torch.empty(2, 15), dst_type=torch.quint4x2)
            except ValueError as error:
                assert "divisible by 8" in str(error)
            else:
                raise AssertionError("invalid int4 input shape was accepted")
        """,
        _COMMON_OP_SCHEMAS + _NPU_OP_SCHEMAS,
    )


def test_npu_custom_ops_are_fake_safe_and_lazy() -> None:
    _run_isolated_python(
        """
        import sys
        import types

        from xllm.python import platform
        from xllm.python.ops.moe import grouped_moe, prepare_grouped_moe_weights
        from xllm.python.ops.rotary_embedding import interleaved_rotary_embedding

        mode = torch._subclasses.fake_tensor.FakeTensorMode()
        with mode:
            hidden = torch.empty(4, 16)
            grouped = grouped_moe(
                hidden, torch.empty(4, 8), torch.empty(8, 16, 32),
                torch.empty(8, 32, 16), torch.empty(8, 32),
                torch.empty(8, 16), None, 2, 1, 1, True,
            )
            assert grouped.shape == hidden.shape
            value = torch.empty(4, 2, 8)
            rope = interleaved_rotary_embedding(
                value, torch.empty(4, 1, 1, 8), torch.empty(4, 1, 1, 8)
            )
            assert rope.shape == value.shape

        calls = {}
        backend = types.ModuleType("xllm.python.ops.npu.moe")
        backend.prepare_grouped_moe_weights = lambda w13, w2: (
            w13.clone(), w2.clone()
        )

        def execute(*args):
            calls["bias_dtype"] = args[6].dtype
            return args[0].clone()

        backend.grouped_moe = execute
        sys.modules[backend.__name__] = backend
        platform.current_platform = lambda: "npu"

        w13 = torch.empty(2, 4, 8)
        w2 = torch.empty(2, 8, 4)
        assert prepare_grouped_moe_weights(w13, w2)[0].shape == w13.shape
        hidden = torch.empty(3, 8, dtype=torch.bfloat16)
        output = grouped_moe(
            hidden, torch.empty(3, 2, dtype=torch.bfloat16), w13, w2,
            torch.empty(2, 8), torch.empty(2, 8),
            torch.empty(2, dtype=torch.float32), 1, 1, 1, True,
        )
        assert output.shape == hidden.shape
        assert calls["bias_dtype"] == torch.bfloat16
        assert "xllm.python.ops.cuda.moe" not in sys.modules
        """,
        _COMMON_OP_SCHEMAS,
    )


def test_cuda_backend_loader_does_not_import_npu() -> None:
    _run_isolated_python(
        """
        import sys
        import types

        from xllm.python import platform
        from xllm.python.ops.moe import supports_cutlass_moe

        backend = types.ModuleType("xllm.python.ops.cuda.moe")
        backend.supports_cutlass = lambda device: device.index == 1
        sys.modules[backend.__name__] = backend
        platform.current_platform = lambda: "cuda"

        assert supports_cutlass_moe(torch.device("cuda:1"))
        assert "xllm.python.ops.npu.moe" not in sys.modules
        """,
        _COMMON_OP_SCHEMAS,
    )
