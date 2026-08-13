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

"""Import isolation and graph contracts for the platform kernel packages."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from xllm.python import platform

_REPO_ROOT = Path(__file__).parents[2]
_PYTHON_ROOT = _REPO_ROOT / "xllm" / "python"

_COMMON_SCHEMAS = (
    "rms_norm(Tensor input, Tensor weight, float eps) -> Tensor",
    "fused_add_rms_norm(Tensor(a!) input, Tensor(b!) residual, Tensor weight, float eps) -> (Tensor, Tensor)",
    "silu_and_mul(Tensor input) -> Tensor",
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
_CUDA_SCHEMAS = (
    *_COMMON_SCHEMAS,
    "fused_qk_norm_rope(Tensor(a!) qkv, int num_heads_q, int num_heads_k, "
    "int num_heads_v, int head_dim, float eps, Tensor q_weight, Tensor k_weight, "
    "Tensor cos_sin_cache, bool interleaved, Tensor position_ids) -> Tensor",
)
_NPU_SCHEMAS = (
    *_COMMON_SCHEMAS,
    "quant_matmul(Tensor x1, Tensor x2, bool transpose2, Tensor scale, "
    "Tensor? offset, Tensor? pertoken_scale, Tensor? bias, ScalarType? "
    "output_dtype) -> Tensor",
    "quantize_per_tensor(Tensor self, Tensor scales, Tensor zero_points, ScalarType dtype, int axis) -> Tensor",
    "dynamic_quant(Tensor input, Tensor? smooth_scales, Tensor? group_index, "
    "ScalarType? dst_type) -> (Tensor, Tensor?)",
    "lightning_indexer(Tensor query, Tensor key, Tensor weights, Tensor? "
    "query_seq_lengths, Tensor? key_seq_lengths, Tensor? block_table, str "
    "layout_query, str layout_key, int selected_count, int sparse_mode, int "
    "pre_tokens, int next_tokens, bool return_value) -> Tensor",
    "lightning_indexer_out(Tensor query, Tensor key, Tensor weights, Tensor? "
    "query_seq_lengths, Tensor? key_seq_lengths, Tensor? block_table, str "
    "layout_query, str layout_key, int selected_count, int sparse_mode, int "
    "pre_tokens, int next_tokens, bool return_value, Tensor(a!) "
    "sparse_indices_out, Tensor(b!) sparse_values_out) -> Tensor",
    "scatter_nd_update(Tensor(a!) var, Tensor indices, Tensor updates) -> ()",
    "sparse_flash_attention(Tensor query, Tensor key, Tensor value, Tensor "
    "sparse_indices, Tensor? block_table, Tensor? actual_seq_lengths_query, "
    "Tensor? actual_seq_lengths_kv, Tensor? query_rope, Tensor? key_rope, "
    "float scale_value, int sparse_block_size, str layout_query, str "
    "layout_kv, int sparse_mode) -> Tensor",
    "sparse_flash_attention_out(Tensor query, Tensor key, Tensor value, Tensor "
    "sparse_indices, Tensor? block_table, Tensor? actual_seq_lengths_query, "
    "Tensor? actual_seq_lengths_kv, Tensor? query_rope, Tensor? key_rope, "
    "float scale_value, int sparse_block_size, str layout_query, str layout_kv, "
    "int sparse_mode, Tensor(a!) output) -> Tensor",
    "rms_norm_dynamic_quant(Tensor input, Tensor weight, float eps) -> (Tensor, Tensor)",
    "npu_inplace_partial_rotary_mul(Tensor(a!) x, Tensor r1, Tensor r2, str rotary_mode, int[] partial_slice) -> ()",
    "moe_gating_top_k_hash(Tensor x, int k, Tensor? bias, Tensor? input_ids, Tensor? tid2eid, int k_group, int group_count, float routed_scaling_factor, float eps, int group_select_mode, int renorm, int norm_type, bool out_flag) -> (Tensor, Tensor, Tensor)",
    "dequant_swiglu_quant(Tensor x, Tensor? weight_scale, Tensor? activation_scale, Tensor? bias, Tensor? quant_scale, Tensor? quant_offset, Tensor? group_index, bool activate_left, int quant_mode, int swiglu_mode, float clamp_limit, float glu_alpha, float glu_bias) -> (Tensor, Tensor)",
    "hc_pre(Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, int hc_mult, int hc_sinkhorn_iters, float norm_eps, float hc_eps) -> (Tensor, Tensor, Tensor)",
    "hc_post(Tensor x, Tensor residual, Tensor post, Tensor comb) -> Tensor",
    "compressor(Tensor x, Tensor wkv, Tensor wgate, Tensor(a!) kv_state, Tensor(b!) score_state, Tensor ape, Tensor norm_weight, Tensor rope_sin, Tensor rope_cos, Tensor? kv_block_table, Tensor? score_block_table, Tensor? cu_seqlens, Tensor? seqused, Tensor? start_pos, int rope_head_dim, int cmp_ratio, int coff, float norm_eps, int rotary_mode, bool enable_grad) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
    "sparse_attn_sharedkv(Tensor q, Tensor? ori_kv, Tensor? cmp_kv, Tensor? ori_sparse_indices, Tensor? cmp_sparse_indices, Tensor? ori_block_table, Tensor? cmp_block_table, Tensor? cu_seqlens_q, Tensor? cu_seqlens_ori_kv, Tensor? cu_seqlens_cmp_kv, Tensor? seqused_q, Tensor? seqused_kv, Tensor? sinks, Tensor? metadata, float softmax_scale, int cmp_ratio, int ori_mask_mode, int cmp_mask_mode, int ori_win_left, int ori_win_right, str layout_q, str layout_kv, bool return_softmax_lse) -> (Tensor, Tensor)",
    "sparse_attn_sharedkv_metadata(int num_heads_q, int num_heads_kv, int head_dim, Tensor? cu_seqlens_q, Tensor? cu_seqlens_ori_kv, Tensor? cu_seqlens_cmp_kv, Tensor? seqused_q, Tensor? seqused_kv, int batch_size, int max_seqlen_q, int max_seqlen_kv, int ori_topk, int cmp_topk, int cmp_ratio, int ori_mask_mode, int cmp_mask_mode, int ori_win_left, int ori_win_right, str layout_q, str layout_kv, bool has_ori_kv, bool has_cmp_kv) -> Tensor",
    "quant_lightning_indexer(Tensor query, Tensor key, Tensor weights, Tensor query_dequant_scale, Tensor key_dequant_scale, int query_quant_mode, int key_quant_mode, Tensor? actual_seq_lengths_query, Tensor? actual_seq_lengths_key, Tensor? block_table, Tensor? metadata, str layout_query, str layout_key, int sparse_count, int sparse_mode, int pre_tokens, int next_tokens, int cmp_ratio, bool return_value) -> (Tensor, Tensor)",
    "quant_lightning_indexer_metadata(int num_heads_q, int num_heads_k, int head_dim, int query_quant_mode, int key_quant_mode, Tensor? actual_seq_lengths_query, Tensor? actual_seq_lengths_key, int batch_size, int max_seqlen_q, int max_seqlen_k, str layout_query, str layout_key, int sparse_count, int sparse_mode, int pre_tokens, int next_tokens, int cmp_ratio, str device) -> Tensor",
)

_PLATFORM_REQUIRED = pytest.mark.skipif(
    not (platform.current_platform.is_cuda() or platform.current_platform.is_npu()),
    reason="no kernel package for this platform",
)


def _package_stub(kernel_package: str | None = None) -> str:
    lines = [
        "import sys",
        "import types",
        'package = types.ModuleType("xllm.python")',
        f"package.__path__ = [{str(_PYTHON_ROOT)!r}]",
        'sys.modules["xllm.python"] = package',
    ]
    if kernel_package:
        qualified_name = f"xllm.python.{kernel_package}"
        lines.extend(
            (
                f"kernels = types.ModuleType({qualified_name!r})",
                f"kernels.__path__ = [{str(_PYTHON_ROOT / kernel_package)!r}]",
                f"package.{kernel_package} = kernels",
                f"sys.modules[{qualified_name!r}] = kernels",
            )
        )
    return "\n".join(lines)


def _run_isolated_python(
    script: str,
    *,
    schemas: tuple[str, ...] = (),
    standalone: bool = False,
    kernel_package: str | None = None,
) -> None:
    definitions: list[str] = []
    if schemas:
        definitions.extend(
            (
                "import torch",
                'library = torch.library.Library("xllm_ops", "DEF")',
                *(f"library.define({schema!r})" for schema in schemas),
            )
        )
    if standalone or kernel_package is not None:
        definitions.append(_package_stub(kernel_package))

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(value for value in (str(_REPO_ROOT), env.get("PYTHONPATH", "")) if value)
    result = subprocess.run(
        [sys.executable, "-c", "\n".join((*definitions, textwrap.dedent(script)))],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_platform_queries_are_no_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    current_platform = platform.current_platform
    monkeypatch.setattr(type(current_platform), "enum", classmethod(lambda cls: platform.PlatformEnum.NPU))
    assert current_platform.is_npu() and not current_platform.is_cuda()
    monkeypatch.setattr(type(current_platform), "enum", classmethod(lambda cls: platform.PlatformEnum.CUDA))
    assert current_platform.is_cuda() and not current_platform.is_npu()


@_PLATFORM_REQUIRED
def test_binding_import_contract() -> None:
    schemas = _CUDA_SCHEMAS if platform.current_platform.is_cuda() else _NPU_SCHEMAS
    _run_isolated_python(
        """
        import sys

        import xllm.python
        from xllm.python import kernels
        import xllm.python.kernels
        from xllm.python.kernels import rms_norm
        from xllm.python.platform import current_platform

        selected = f"xllm.python.kernels_{'cuda' if current_platform.is_cuda() else 'npu'}"
        peer = (
            "xllm.python.kernels_npu"
            if current_platform.is_cuda()
            else "xllm.python.kernels_cuda"
        )
        assert kernels is sys.modules[selected]
        assert kernels is sys.modules["xllm.python.kernels"]
        assert kernels is xllm.python.kernels
        assert rms_norm is kernels.rms_norm
        assert kernels.__name__ == selected
        assert not any(
            name == peer or name.startswith(peer + ".") for name in sys.modules
        )
        """,
        schemas=schemas,
    )


def test_registry_does_not_preload_model_modules() -> None:
    _run_isolated_python(
        """
        import sys

        from xllm.python.registry import get_model_class, register_model

        assert callable(get_model_class)
        assert callable(register_model)
        assert not any(name.startswith("xllm.python.models.") for name in sys.modules)
        """,
        standalone=True,
    )


def test_npu_fake_tensor_and_mutation_contracts() -> None:
    """NPU wrapper shape and mutation contracts traced without an NPU."""
    _run_isolated_python(
        """
        import xllm.python.kernels_npu._custom_op  # noqa: F401
        from xllm.python.kernels_npu import dsa, normalization, quantization
        from xllm.python.kernels_npu import rotary_embedding, sparse_attention

        mode = torch._subclasses.fake_tensor.FakeTensorMode()
        with mode:
            output = quantization.quant_matmul(
                torch.empty(2, 16), torch.empty(16, 4), False, torch.empty(4),
                None, None, None, torch.bfloat16,
            )
            assert output.shape == (2, 4) and output.dtype == torch.bfloat16

            quantized, scale = quantization.dynamic_quant(
                torch.empty(2, 16), dst_type=torch.quint4x2
            )
            assert quantized.shape == (2, 2) and quantized.dtype == torch.int32
            assert scale.shape == (2,) and scale.dtype == torch.float32

            query = torch.empty(8, 4, 16)
            key = torch.empty(8, 2, 16)
            indices = sparse_attention.lightning_indexer(
                query, key, torch.empty(4, 2), None, None, None,
                "TND", "TND", 3, 0, 0, 0, False,
            )
            assert indices.shape == (8, 2, 3)
            assert sparse_attention.sparse_flash_attention(
                query, key, torch.empty_like(key), indices, None, None, None,
                None, None, 1.0, 128, "TND", "TND", 0,
            ).shape == query.shape
            assert sparse_attention.scatter_nd_update(
                torch.empty(8, 2, 16), torch.empty(2, 1, dtype=torch.int64),
                torch.empty(2, 2, 16),
            ) is None

            normed, norm_scale = normalization.rms_norm_dynamic_quant(
                torch.empty(8, 16), torch.empty(16), 1e-6
            )
            assert normed.shape == (8, 16) and normed.dtype == torch.int8
            assert norm_scale.shape == (8,) and norm_scale.dtype == torch.float32

            rotary_input = torch.empty(8, 2, 128)
            rotary_ptr = rotary_input.data_ptr()
            assert rotary_embedding.npu_inplace_partial_rotary_mul(
                rotary_input, torch.empty(8, 64), torch.empty(8, 64), 64, 64
            ).data_ptr() == rotary_ptr

            compressor_out = dsa.compressor(
                torch.empty(8, 16),
                torch.empty(8, 16),
                torch.empty(8, 16),
                torch.empty(1, 128, 8),
                torch.empty(1, 128, 8),
                torch.empty(4, 8),
                torch.empty(8),
                torch.empty(2, 4),
                torch.empty(2, 4),
                None, None, None, None, None,
                4, 4, 1, 1e-6, 1, False,
            )
            assert compressor_out[0].shape == (2, 8)
            assert all(tensor.numel() == 0 for tensor in compressor_out[1:])

            seq_lens = torch.empty(1, dtype=torch.int32)
            sparse_metadata = dsa.sparse_attn_sharedkv_metadata(
                64, 1, 512, None, None, None, seq_lens, seq_lens,
                1, 4, 16, 0, 0, 1, 4, 3, 127, 0,
                "BSND", "PA_ND", True, False,
            )
            assert sparse_metadata.shape == (1024,)
            assert sparse_metadata.dtype == torch.int32

            dsa_query = torch.empty(1, 4, 64, 512)
            sparse_out, sparse_lse = dsa.sparse_attn_sharedkv(
                dsa_query, None, None, None, None, None, None,
                None, None, None, None, None, None, sparse_metadata,
                1.0, 1, 4, 3, 127, 0, "BSND", "PA_ND", False,
            )
            assert sparse_out.shape == dsa_query.shape
            assert sparse_lse.numel() == 0

            qli_metadata = dsa.quant_lightning_indexer_metadata(
                64, 1, 128, 0, 0, seq_lens, seq_lens,
                1, 8, 8, "TND", "PA_BSND", 512, 3,
                2**63 - 1, 2**63 - 1, 4, "cpu",
            )
            assert qli_metadata.shape == (1024,)
            assert qli_metadata.dtype == torch.int32

            qli_indices, qli_values = dsa.quant_lightning_indexer(
                torch.empty(8, 64, 128, dtype=torch.int8),
                torch.empty(1, 128, 1, 128, dtype=torch.int8),
                torch.empty(8, 64),
                torch.empty(8, 64),
                torch.empty(1, 128, 1),
                0, 0, seq_lens, seq_lens, torch.empty(1, 1), qli_metadata,
                "TND", "PA_BSND", 512, 3,
                2**63 - 1, 2**63 - 1, 4, False,
            )
            assert qli_indices.shape == (8, 1, 512)
            assert qli_indices.dtype == torch.int32
            assert qli_values.numel() == 0

            hc_input = torch.empty(8, 4, 16)
            hc_attn, hc_post, hc_comb = dsa.hc_pre(
                hc_input,
                torch.empty(24, 64),
                torch.empty(3),
                torch.empty(24),
                4, 20, 1e-6, 1e-6,
            )
            assert hc_attn.shape == (8, 16)
            assert hc_post.shape == (8, 4)
            assert hc_comb.shape == (8, 4, 4)
            assert dsa.hc_post(
                hc_attn, hc_input, hc_post, hc_comb
            ).shape == hc_input.shape

            gate_weights, expert_idx, gate_out = dsa.moe_gating_top_k_hash(
                torch.empty(8, 256), 6, None, None, None,
                1, 1, 1.0, 1e-20, 1, 0, 2, False,
            )
            assert gate_weights.shape == (8, 6)
            assert expert_idx.shape == (8, 6)
            assert expert_idx.dtype == torch.int32
            assert gate_out.shape == (8, 256)
            assert gate_out.dtype == torch.float32

            swiglu_out, swiglu_scale = dsa.dequant_swiglu_quant(
                torch.empty(8, 32, dtype=torch.int32), None, None
            )
            assert swiglu_out.shape == (8, 16)
            assert swiglu_out.dtype == torch.int8
            assert swiglu_scale.shape == (8,)
            assert swiglu_scale.dtype == torch.float32

            try:
                quantization.dynamic_quant(
                    torch.empty(2, 15), dst_type=torch.quint4x2
                )
            except ValueError as error:
                assert "divisible by 8" in str(error)
            else:
                raise AssertionError("invalid int4 input shape was accepted")
        """,
        schemas=_NPU_SCHEMAS,
        kernel_package="kernels_npu",
    )


def test_cuda_fake_tensor_and_mutation_contracts() -> None:
    """Normalization and activation shapes traced without a GPU."""
    _run_isolated_python(
        """
        import xllm.python.kernels_cuda._custom_op  # noqa: F401
        from xllm.python.kernels_cuda import activation, normalization

        mode = torch._subclasses.fake_tensor.FakeTensorMode()
        with mode:
            hidden = torch.empty(4, 16)
            assert normalization.rms_norm(
                hidden, torch.empty(16), 1e-6
            ).shape == hidden.shape

            residual = torch.empty(4, 16)
            normed, updated = normalization.fused_add_rms_norm(
                hidden, residual, torch.empty(16), 1e-6
            )
            assert normed.shape == hidden.shape
            assert updated.shape == residual.shape

            gated = torch.empty(4, 32)
            assert activation.silu_and_mul(gated).shape == (4, 16)
        """,
        schemas=_CUDA_SCHEMAS,
        kernel_package="kernels_cuda",
    )
