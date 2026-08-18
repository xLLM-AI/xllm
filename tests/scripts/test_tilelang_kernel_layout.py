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

import ast
import os
import unittest
from pathlib import Path

_REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_RUNTIME_ROOT = _REPO_ROOT / "xllm" / "python" / "kernels_npu" / "tilelang"
_TRITON_ROOT = _REPO_ROOT / "xllm" / "python" / "kernels_npu" / "triton"
_AOT_ROOT = _REPO_ROOT / "xllm" / "compiler" / "tilelang" / "targets" / "ascend" / "aot"
_OLD_ROOT = _REPO_ROOT / "xllm" / "compiler" / "tilelang" / "targets" / "ascend" / "kernels"
_KERNEL_MODULES = {
    "apply_token_bitmask",
    "causal_conv1d",
    "causal_conv1d_decode",
    "chunk_gated_delta_rule_fwd_h",
    "fused_gdn_gating",
    "fused_sigmoid_gating_delta_rule",
    "rope",
    "spec_verify_attention_tiling_update",
    "spec_verify_token_update",
    "split_qkv_rmsnorm_mrope",
}
_FORBIDDEN_RUNTIME_IMPORT_PREFIXES = (
    "compiler",
    "xllm.compiler",
    "xllm.core",
    "xllm.pybind",
    "xllm.python.kernels_npu._custom_op",
)


def _module_imports(tree: ast.Module) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    return imports


class TilelangKernelLayoutTest(unittest.TestCase):
    def test_all_kernel_modules_have_runtime_and_aot_halves(self) -> None:
        runtime_modules = {path.stem for path in _RUNTIME_ROOT.glob("*.py")} - {"__init__", "utils"}
        aot_modules = {path.stem for path in _AOT_ROOT.glob("*.py")} - {"__init__", "utils"}
        self.assertEqual(runtime_modules, _KERNEL_MODULES)
        self.assertEqual(aot_modules, _KERNEL_MODULES)
        self.assertFalse(list(_OLD_ROOT.glob("*.py")))

    def test_dsl_modules_do_not_depend_on_compiler_or_native_runtime(self) -> None:
        for runtime_root in (_RUNTIME_ROOT, _TRITON_ROOT):
            for module_path in sorted(runtime_root.glob("*.py")):
                if module_path.name == "__init__.py":
                    continue
                module_name = f"{runtime_root.name}/{module_path.stem}"
                tree = ast.parse(module_path.read_text(encoding="utf-8"))
                imports = _module_imports(tree)
                self.assertFalse(
                    any(
                        name == prefix or name.startswith(f"{prefix}.")
                        for name in imports
                        for prefix in _FORBIDDEN_RUNTIME_IMPORT_PREFIXES
                    ),
                    module_name,
                )
                self.assertNotIn("scripts.logger", imports)
                self.assertFalse(
                    any(
                        name.startswith("xllm.python.kernels_npu.")
                        and not name.startswith(f"xllm.python.kernels_npu.{runtime_root.name}")
                        for name in imports
                    ),
                    module_name,
                )

    def test_tilelang_runtime_modules_do_not_contain_aot_descriptors(self) -> None:
        for module_name in _KERNEL_MODULES:
            tree = ast.parse((_RUNTIME_ROOT / f"{module_name}.py").read_text(encoding="utf-8"))
            self.assertFalse(
                any(
                    isinstance(node, ast.ClassDef)
                    and any(isinstance(base, ast.Name) and base.id == "TilelangKernel" for base in node.bases)
                    for node in tree.body
                ),
                module_name,
            )

    def test_aot_modules_are_thin_descriptors(self) -> None:
        for module_name in _KERNEL_MODULES:
            tree = ast.parse((_AOT_ROOT / f"{module_name}.py").read_text(encoding="utf-8"))
            kernel_classes = [
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef)
                and any(isinstance(base, ast.Name) and base.id == "TilelangKernel" for base in node.bases)
            ]
            self.assertEqual(len(kernel_classes), 1, module_name)
            dependency_assignments = [
                node
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "DEPENDENCY_MODULES" for target in node.targets)
            ]
            self.assertEqual(len(dependency_assignments), 1, module_name)
            dependency_value = dependency_assignments[0].value
            self.assertIsInstance(dependency_value, ast.Tuple, module_name)
            self.assertEqual(
                {element.id for element in dependency_value.elts if isinstance(element, ast.Name)},
                {"kernel_impl", "tilelang_utils"},
                module_name,
            )
            self.assertTrue(
                any(
                    isinstance(node, ast.ImportFrom)
                    and node.module == f"xllm.python.kernels_npu.tilelang.{module_name}"
                    for node in tree.body
                ),
                module_name,
            )
            self.assertFalse(
                any(
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and any(
                        isinstance(decorator, ast.Attribute)
                        and isinstance(decorator.value, ast.Name)
                        and decorator.value.id == "T"
                        and decorator.attr == "prim_func"
                        for decorator in node.decorator_list
                    )
                    for node in ast.walk(tree)
                ),
                module_name,
            )


if __name__ == "__main__":
    unittest.main()
