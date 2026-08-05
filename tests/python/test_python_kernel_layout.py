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

"""Architecture boundaries for Python models, layers, and kernel packages."""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]
_PYTHON_ROOT = _REPO_ROOT / "xllm" / "python"

# Models and layers reach kernels through the bound name ``xllm.python.kernels``
# only. Naming a platform package, a vendor library, or the platform query would
# put a hardware branch above the kernel layer.
_MODEL_IMPORTS = (
    "flashinfer",
    "torch_npu",
    "triton",
    "xllm.python.kernels_cuda",
    "xllm.python.kernels_npu",
    "xllm.python.platform",
)
_MODEL_ATTRIBUTES = ("torch.cuda", "torch.ops.npu", "torch_npu")

_KERNEL_PACKAGES = ("kernels_cuda", "kernels_npu")


def _python_files(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.py"))


def _qualified_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _imported_modules(tree: ast.AST) -> list[tuple[int, str]]:
    modules: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append((node.lineno, node.module or ""))
    return modules


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def test_models_and_layers_are_hardware_independent() -> None:
    violations: list[str] = []
    for root in (_PYTHON_ROOT / "models", _PYTHON_ROOT / "layers"):
        for path in _python_files(root):
            tree = ast.parse(path.read_text(), filename=str(path))
            relative = path.relative_to(_REPO_ROOT)
            for line, module in _imported_modules(tree):
                if _matches(module, _MODEL_IMPORTS):
                    violations.append(f"{relative}:{line}: {module}")
            for node in ast.walk(tree):
                name = _qualified_name(node)
                if isinstance(node, ast.Attribute) and _matches(
                    name, _MODEL_ATTRIBUTES
                ):
                    violations.append(f"{relative}:{node.lineno}: {name}")
    assert violations == []


def test_kernel_packages_do_not_depend_on_layers_or_peers() -> None:
    violations: list[str] = []
    for package in _KERNEL_PACKAGES:
        peers = tuple(
            f"xllm.python.{name}" for name in _KERNEL_PACKAGES if name != package
        )
        forbidden = ("xllm.python.layers", "xllm.python.models", *peers)
        for path in _python_files(_PYTHON_ROOT / package):
            tree = ast.parse(path.read_text(), filename=str(path))
            relative = path.relative_to(_REPO_ROOT)
            for line, module in _imported_modules(tree):
                if _matches(module, forbidden):
                    violations.append(f"{relative}:{line}: {module}")
    assert violations == []


def test_kernel_packages_name_themselves_only_through_relative_imports() -> None:
    """A package that never spells its own name can be copied to seed a peer."""
    violations: list[str] = []
    for package in _KERNEL_PACKAGES:
        own_name = f"xllm.python.{package}"
        for path in _python_files(_PYTHON_ROOT / package):
            tree = ast.parse(path.read_text(), filename=str(path))
            relative = path.relative_to(_REPO_ROOT)
            for line, module in _imported_modules(tree):
                if _matches(module, (own_name,)):
                    violations.append(f"{relative}:{line}: {module}")
    assert violations == []


def test_launchers_stay_under_their_framework_directory() -> None:
    """``triton.jit`` belongs in ``triton/``, not in the modules that export."""
    violations: list[str] = []
    for package in _KERNEL_PACKAGES:
        root = _PYTHON_ROOT / package
        for path in _python_files(root):
            if path.parent != root:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            relative = path.relative_to(_REPO_ROOT)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for decorator in node.decorator_list:
                    name = _qualified_name(
                        decorator.func
                        if isinstance(decorator, ast.Call)
                        else decorator
                    )
                    if name.endswith(("triton.jit", "triton.autotune")):
                        violations.append(f"{relative}:{node.lineno}: {name}")
    assert violations == []


def test_platform_packages_are_peers() -> None:
    for package in _KERNEL_PACKAGES:
        root = _PYTHON_ROOT / package
        assert (root / "__init__.py").is_file()
        assert (root / "_custom_op.py").is_file()
        assert (root / "triton").is_dir()
    assert (_PYTHON_ROOT / "kernels_cuda" / "flashinfer").is_dir()
    assert not (_PYTHON_ROOT / "kernels").exists()
    assert not (_PYTHON_ROOT / "ops" / "__init__.py").exists()


def test_the_platform_branch_lives_only_in_the_package_init() -> None:
    """``platform.is_*`` selects the kernel package once, in one place."""
    branching: list[str] = []
    for path in _python_files(_PYTHON_ROOT):
        tree = ast.parse(path.read_text(), filename=str(path))
        relative = path.relative_to(_REPO_ROOT)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _qualified_name(node.func)
            if name.endswith(("platform.is_gpu", "platform.is_npu")):
                branching.append(str(relative))
    # The kernel binding, plus the attention backend and graph runner selection
    # in the executor. Attention backends hold per-step state and are wired into
    # the executor, so they are selected there rather than exported by a kernel
    # package.
    assert set(branching) == {
        "xllm/python/__init__.py",
        "xllm/python/model_executor/executor.py",
    }
