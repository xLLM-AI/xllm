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

"""Architecture and import tests for Python graph ops and hardware kernels."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]
_PYTHON_ROOT = _REPO_ROOT / "xllm" / "python"


def _python_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _decorator_name(decorator: ast.expr) -> str:
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    parts: list[str] = []
    while isinstance(decorator, ast.Attribute):
        parts.append(decorator.attr)
        decorator = decorator.value
    if isinstance(decorator, ast.Name):
        parts.append(decorator.id)
    return ".".join(reversed(parts))


def test_kernel_modules_do_not_register_torch_ops() -> None:
    violations: list[str] = []
    for path in _python_files(_PYTHON_ROOT / "kernels"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                name = _decorator_name(decorator)
                if name.endswith("custom_op") or name.endswith("register_fake"):
                    violations.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}")
    assert violations == []


def test_op_modules_do_not_define_triton_kernels() -> None:
    violations: list[str] = []
    for path in _python_files(_PYTHON_ROOT / "ops"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                name = _decorator_name(decorator)
                if name.endswith("triton.jit") or name.endswith("triton.autotune"):
                    violations.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}")
    assert violations == []


def test_triton_kernels_are_partitioned_by_hardware() -> None:
    triton_root = _PYTHON_ROOT / "kernels" / "triton"
    assert (triton_root / "cuda" / "silu_and_mul.py").is_file()
    assert (triton_root / "cuda" / "fused_moe.py").is_file()
    assert (triton_root / "npu" / "split_qkv_rmsnorm_rope.py").is_file()
    legacy_ops = _PYTHON_ROOT / "ops" / "triton"
    assert not any(legacy_ops.glob("*.py"))
    assert not (_PYTHON_ROOT / "kernels" / "triton_ops.py").exists()

    compute_source = (_PYTHON_ROOT / "ops" / "compute.py").read_text()
    assert "xllm.python.kernels.triton.npu.split_qkv_rmsnorm_rope" in compute_source
    assert "xllm.python.ops.triton" not in compute_source


def test_public_ops_import_does_not_load_hardware_kernels() -> None:
    script = r'''
import sys
import types
from pathlib import Path
import torch

python_package = types.ModuleType("xllm.python")
python_package.__path__ = [str(Path.cwd() / "xllm/python")]
sys.modules["xllm.python"] = python_package

library = torch.library.Library("xllm_ops", "DEF")
library.define("rms_norm(Tensor input, Tensor weight, float eps) -> Tensor")
library.define(
    "fused_add_rms_norm(Tensor(a!) input, Tensor(b!) residual, Tensor weight, "
    "float eps) -> (Tensor, Tensor)"
)
library.define("silu_and_mul(Tensor input) -> Tensor")
library.define(
    "fused_qk_norm_rope(Tensor(a!) qkv, int num_heads_q, int num_heads_k, "
    "int num_heads_v, int head_dim, float eps, Tensor q_weight, Tensor k_weight, "
    "Tensor cos_sin_cache, bool interleaved, Tensor position_ids) -> Tensor"
)
library.define(
    "quant_matmul(Tensor x1, Tensor x2, bool transpose2, Tensor scale, "
    "Tensor? offset, Tensor? pertoken_scale, Tensor? bias, ScalarType? "
    "output_dtype) -> Tensor"
)
library.define(
    "quantize_per_tensor(Tensor self, Tensor scales, Tensor zero_points, "
    "ScalarType dtype, int axis) -> Tensor"
)
library.define(
    "dynamic_quant(Tensor input, Tensor? smooth_scales, Tensor? group_index, "
    "ScalarType? dst_type) -> (Tensor, Tensor?)"
)
library.define(
    "lightning_indexer(Tensor query, Tensor key, Tensor weights, "
    "Tensor? query_seq_lengths, Tensor? key_seq_lengths, Tensor? block_table, "
    "str layout_query, str layout_key, int selected_count, int sparse_mode, "
    "int pre_tokens, int next_tokens, bool return_value) -> Tensor"
)
library.define(
    "scatter_nd_update(Tensor(a!) var, Tensor indices, Tensor updates) -> ()"
)
library.define(
    "sparse_flash_attention(Tensor query, Tensor key, Tensor value, "
    "Tensor sparse_indices, Tensor? block_table, Tensor? actual_seq_lengths_query, "
    "Tensor? actual_seq_lengths_kv, Tensor? query_rope, Tensor? key_rope, "
    "float scale_value, int sparse_block_size, str layout_query, str layout_kv, "
    "int sparse_mode) -> Tensor"
)
library.define(
    "reshape_paged_cache(Tensor slot_mapping, Tensor keys, Tensor values, "
    "Tensor(a!) key_cache, Tensor(b!) value_cache) -> Tensor"
)
library.define(
    "update_decode_graph_metadata(Tensor tokens, Tensor positions, "
    "Tensor slot_mapping, Tensor kv_seq_lens, Tensor paged_kv_indptr, "
    "Tensor paged_kv_indices, Tensor paged_kv_last_page_len, "
    "Tensor(a!) dst_tokens, Tensor(b!) dst_positions, "
    "Tensor(c!) dst_slot_mapping, Tensor(d!) dst_kv_seq_lens, "
    "Tensor(e!) dst_kv_seq_lens_delta, Tensor(f!) dst_paged_kv_indptr, "
    "Tensor(g!) dst_paged_kv_indices, Tensor(h!) dst_paged_kv_last_page_len, "
    "int padded_num_tokens) -> Tensor"
)

import xllm.python.ops  # noqa: F401

assert "triton" not in sys.modules
assert "flashinfer" not in sys.modules
assert not any(
    name.startswith("xllm.python.kernels.triton") for name in sys.modules
)
assert not any(
    name.startswith("xllm.python.kernels.flashinfer") for name in sys.modules
)
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
