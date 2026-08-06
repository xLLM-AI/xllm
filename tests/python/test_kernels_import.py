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
    "fused_add_rms_norm(Tensor(a!) input, Tensor(b!) residual, Tensor weight, "
    "float eps) -> (Tensor, Tensor)",
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

_PLATFORM_REQUIRED = pytest.mark.skipif(
    not (platform.is_gpu() or platform.is_npu()),
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
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(_REPO_ROOT), env.get("PYTHONPATH", "")) if value
    )
    result = subprocess.run(
        [sys.executable, "-c", "\n".join((*definitions, textwrap.dedent(script)))],
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


@_PLATFORM_REQUIRED
def test_binding_import_contract() -> None:
    schemas = _CUDA_SCHEMAS if platform.is_gpu() else _NPU_SCHEMAS
    _run_isolated_python(
        """
        import sys

        import xllm.python
        from xllm.python import kernels
        import xllm.python.kernels
        from xllm.python.kernels import rms_norm
        from xllm.python import platform

        selected = f"xllm.python.kernels_{'cuda' if platform.is_gpu() else 'npu'}"
        peer = (
            "xllm.python.kernels_npu"
            if platform.is_gpu()
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
    """Quantization and sparse attention shapes traced without an NPU."""
    _run_isolated_python(
        """
        import xllm.python.kernels_npu._custom_op  # noqa: F401
        from xllm.python.kernels_npu import quantization, sparse_attention

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
