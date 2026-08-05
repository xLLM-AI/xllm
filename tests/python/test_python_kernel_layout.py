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

"""Architecture tests for Python models, ops, and hardware kernels."""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]
_PYTHON_ROOT = _REPO_ROOT / "xllm" / "python"

_FORBIDDEN_IMPORT_PREFIXES = (
    "flashinfer",
    "torch_npu",
    "triton",
    "xllm.python.kernels",
    "xllm.python.ops.cuda",
    "xllm.python.ops.npu",
)
_FORBIDDEN_ATTRIBUTE_PREFIXES = (
    "torch.cuda",
    "torch.ops.npu",
    "torch_npu",
)


def _python_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _decorator_name(decorator: ast.expr) -> str:
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    return _attribute_name(decorator)


def _attribute_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def test_models_and_layers_are_hardware_independent() -> None:
    violations: list[str] = []
    for root in (_PYTHON_ROOT / "models", _PYTHON_ROOT / "layers"):
        for path in _python_files(root):
            tree = ast.parse(path.read_text(), filename=str(path))
            relative_path = path.relative_to(_REPO_ROOT)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if _matches_prefix(alias.name, _FORBIDDEN_IMPORT_PREFIXES):
                            violations.append(
                                f"{relative_path}:{node.lineno}: import {alias.name}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if _matches_prefix(module, _FORBIDDEN_IMPORT_PREFIXES):
                        violations.append(
                            f"{relative_path}:{node.lineno}: from {module}"
                        )
                elif isinstance(node, ast.Attribute):
                    name = _attribute_name(node)
                    if _matches_prefix(name, _FORBIDDEN_ATTRIBUTE_PREFIXES):
                        violations.append(f"{relative_path}:{node.lineno}: {name}")
    assert violations == []


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


def test_kernels_do_not_reverse_depend_on_graph_layers() -> None:
    violations: list[str] = []
    forbidden = (
        "xllm.python.layers",
        "xllm.python.models",
        "xllm.python.ops",
    )
    for path in _python_files(_PYTHON_ROOT / "kernels"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules = (node.module or "",)
            else:
                continue
            for module in modules:
                if _matches_prefix(module, forbidden):
                    violations.append(
                        f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {module}"
                    )
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


def test_public_ops_init_does_not_import_platform_backends() -> None:
    path = _PYTHON_ROOT / "ops" / "__init__.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules = (node.module or "",)
        else:
            continue
        for module in modules:
            if _matches_prefix(
                module, ("xllm.python.ops.cuda", "xllm.python.ops.npu")
            ):
                violations.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}")
    assert violations == []


def test_kernels_are_partitioned_by_hardware_then_framework() -> None:
    kernels_root = _PYTHON_ROOT / "kernels"
    assert (kernels_root / "cuda" / "triton" / "silu_and_mul.py").is_file()
    assert (kernels_root / "cuda" / "triton" / "fused_moe.py").is_file()
    assert (
        kernels_root / "cuda" / "flashinfer" / "gated_delta_net.py"
    ).is_file()
    assert (
        kernels_root / "npu" / "triton" / "split_qkv_rmsnorm_rope.py"
    ).is_file()
    assert not (kernels_root / "triton").exists()
    assert not (kernels_root / "flashinfer").exists()
    assert not (_PYTHON_ROOT / "ops" / "triton").exists()
    assert not (kernels_root / "triton_ops.py").exists()

    compute_source = (_PYTHON_ROOT / "ops" / "compute.py").read_text()
    assert "xllm.python.kernels.npu.triton.split_qkv_rmsnorm_rope" in compute_source
    assert "xllm.python.kernels.triton" not in compute_source


def test_public_and_platform_ops_are_partitioned_by_ownership() -> None:
    ops_root = _PYTHON_ROOT / "ops"
    assert (ops_root / "quantization.py").is_file()
    assert (ops_root / "sparse_attention.py").is_file()
    assert (ops_root / "rotary_embedding.py").is_file()
    assert (ops_root / "linear.py").is_file()
    assert (ops_root / "npu" / "linear.py").is_file()
    assert (ops_root / "npu" / "moe.py").is_file()
    assert (ops_root / "npu" / "rotary_embedding.py").is_file()
    assert (ops_root / "cuda" / "moe.py").is_file()
    assert not (ops_root / "npu" / "quantization.py").exists()
    assert not (ops_root / "npu" / "sparse_attention.py").exists()
    assert not (ops_root / "npu_compute.py").exists()
