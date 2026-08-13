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
from collections.abc import Iterator
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]
_PYTHON_ROOT = _REPO_ROOT / "xllm" / "python"
_KERNEL_PACKAGES = ("kernels_cuda", "kernels_npu")

_HARDWARE_IMPORTS = (
    "flashinfer",
    "torch_npu",
    "triton",
    "xllm.python.kernels_cuda",
    "xllm.python.kernels_npu",
    "xllm.python.platform",
)
_HARDWARE_ATTRIBUTES = ("torch.cuda", "torch.ops.npu", "torch_npu")

_NODES = {
    path: tuple(ast.walk(ast.parse(path.read_text(), filename=str(path))))
    for path in sorted(_PYTHON_ROOT.rglob("*.py"))
}


def _files_under(directory: Path) -> Iterator[tuple[Path, tuple[ast.AST, ...]]]:
    for path, nodes in _NODES.items():
        if path.is_relative_to(directory):
            yield path, nodes


def _qualified_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _imported_modules(nodes: tuple[ast.AST, ...]) -> Iterator[tuple[int, str]]:
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            yield node.lineno, module
            if module and node.level == 0:
                for alias in node.names:
                    if alias.name != "*":
                        yield node.lineno, f"{module}.{alias.name}"


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def _relative(path: Path) -> Path:
    return path.relative_to(_REPO_ROOT)


def _exports(package: str) -> tuple[str, ...]:
    path = _PYTHON_ROOT / package / "__init__.py"
    for node in _NODES[path]:
        if not isinstance(node, ast.Assign):
            continue
        defines_all = any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
        if defines_all:
            exports = ast.literal_eval(node.value)
            return tuple(exports)
    raise AssertionError(f"{_relative(path)} does not define __all__")


def test_models_and_layers_are_hardware_independent() -> None:
    violations: list[str] = []
    for directory in ("models", "layers"):
        for path, nodes in _files_under(_PYTHON_ROOT / directory):
            for line, module in _imported_modules(nodes):
                if _matches(module, _HARDWARE_IMPORTS):
                    violations.append(f"{_relative(path)}:{line}: {module}")
            for node in nodes:
                if not isinstance(node, ast.Attribute):
                    continue
                name = _qualified_name(node)
                if _matches(name, _HARDWARE_ATTRIBUTES):
                    violations.append(f"{_relative(path)}:{node.lineno}: {name}")
    assert violations == []


def test_kernel_package_import_boundaries() -> None:
    forbidden = (
        "xllm.python.layers",
        "xllm.python.models",
        *(f"xllm.python.{package}" for package in _KERNEL_PACKAGES),
    )
    violations: list[str] = []
    for package in _KERNEL_PACKAGES:
        for path, nodes in _files_under(_PYTHON_ROOT / package):
            for line, module in _imported_modules(nodes):
                if _matches(module, forbidden):
                    violations.append(f"{_relative(path)}:{line}: {module}")
    assert violations == []


def test_platform_packages_export_the_same_kernel_api() -> None:
    cuda_exports = _exports("kernels_cuda")
    npu_exports = _exports("kernels_npu")
    assert cuda_exports
    assert set(cuda_exports) == set(npu_exports)


def test_platform_queries_stay_in_the_owning_layers() -> None:
    violations: list[str] = []
    binding = Path("xllm/python/__init__.py")
    executor_root = Path("xllm/python/model_executor")
    for path, nodes in _NODES.items():
        relative = _relative(path)
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            name = _qualified_name(node.func)
            if not name.endswith(("platform.is_cuda", "platform.is_npu")):
                continue
            if relative != binding and not relative.is_relative_to(executor_root):
                violations.append(f"{relative}:{node.lineno}: {name}")
    assert violations == []
