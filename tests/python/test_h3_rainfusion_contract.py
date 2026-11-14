# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

from xllm.compiler.tilelang.common.spec import TilelangKernel

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("h3_ref2va_rf_v2_selector", "h3_rainfusion_attention")


def _load_descriptor(name: str) -> tuple[type[TilelangKernel], Mock]:
    """Exercise descriptor logic without importing the optional TileLang DSL."""
    path = ROOT / "xllm/compiler/tilelang/targets/ascend/aot" / f"{name}.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
        or isinstance(node, ast.ImportFrom)
        and node.module == "common.spec"
        and node.level == 4
    ]
    kernel = Mock(return_value=SimpleNamespace(get_kernel_source=lambda: "frozen-source"))
    namespace: dict[str, Any] = {
        "__name__": "xllm.compiler.tilelang.targets.ascend.aot.h3_descriptor_contract",
        "__package__": "xllm.compiler.tilelang.targets.ascend.aot",
        "tilelang": SimpleNamespace(disable_cache=lambda: None),
        "kernel_impl": ModuleType(name),
        "h3_rainfusion_attention": kernel,
        "h3_ref2va_rf_v2_selector_rfa": kernel,
    }
    exec(compile(tree, str(path), "exec"), namespace)
    descriptor = next(
        value
        for value in namespace.values()
        if isinstance(value, type) and issubclass(value, TilelangKernel) and value is not TilelangKernel
    )
    return descriptor, kernel


def _dsl_tree(name: str) -> ast.Module:
    path = ROOT / "xllm/python/kernels_npu/tilelang" / f"{name}.py"
    return ast.parse(path.read_text())


class H3RainfusionContractTest(unittest.TestCase):
    def test_descriptors_keep_dispatch_axes_and_instance_source(self) -> None:
        expected_axes = (
            ["total_rows", "valid_rows", "video_start", "grid_t", "grid_h", "grid_w", "record_count", "topk_r"],
            [
                "total_rows",
                "valid_rows",
                "num_blocks",
                "record_count",
                "records_padded",
                "dense_prefix_blocks",
                "sparse_end_block",
                "dtype",
            ],
        )
        for name, axes in zip(FAMILIES, expected_axes, strict=True):
            descriptor, kernel = _load_descriptor(name)
            self.assertEqual([field.name for field in descriptor.DISPATCH_SCHEMA], axes)
            self.assertEqual(len(descriptor.SPECIALIZATIONS), 1)
            kwargs = {key: value for key, value in descriptor.SPECIALIZATIONS[0].items() if key != "variant_key"}
            self.assertEqual(descriptor.generate_source(**kwargs), "frozen-source")
            kernel.assert_called_once_with(**kwargs)

    def test_debug_attention_modes_cannot_enter_aot(self) -> None:
        descriptor, kernel = _load_descriptor(FAMILIES[1])
        kwargs = {key: value for key, value in descriptor.SPECIALIZATIONS[0].items() if key != "variant_key"}
        for changed in ({"output_order": "reordered"}, {"task_limit": 1}):
            with self.assertRaises(ValueError):
                descriptor.generate_source(**(kwargs | changed))
        kernel.assert_not_called()

    def test_manual_sync_pass_configs_are_frozen(self) -> None:
        expected = {
            "TL_ASCEND_AUTO_CV_COMBINE": False,
            "TL_ASCEND_AUTO_CV_SYNC": False,
            "TL_ASCEND_AUTO_SYNC": False,
            "TL_ASCEND_MEMORY_PLANNING": False,
            "TL_ASCEND_AUTO_SYNC_VS": True,
        }
        for name in FAMILIES:
            tree = _dsl_tree(name)
            assignment = next(
                node
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "PASS_CONFIGS" for target in node.targets)
            )
            actual = {
                key.attr: ast.literal_eval(value)
                for key, value in zip(assignment.value.keys, assignment.value.values, strict=True)
            }
            self.assertEqual(actual, expected)

    def test_device_pointer_order_matches_cpp_launch_contract(self) -> None:
        expected = (
            [
                "q_gm",
                "k_gm",
                "v_gm",
                "records_gm",
                "plan_ids_gm",
                "plan_count_gm",
                "rq_gm",
                "rk_gm",
                "rv_gm",
                "q_pool_gm",
                "k_pool_gm",
                "ring_gm",
                "rt_total_blocks",
                "rt_topk_r",
                "rt_record_count",
                "rt_records_per_worker",
                "rt_tail_rows",
                "rt_ff_blocks",
                "rt_vfb",
                "rt_num_sq",
            ],
            [
                "reordered_q",
                "reordered_k",
                "reordered_v",
                "plan_ids",
                "plan_count",
                "reorder_records_flat",
                "output",
                "workspace_s",
                "workspace_p",
                "workspace_o",
                "packed_k",
                "packed_v",
                "reordered_output",
                "rt_valid_rows",
                "rt_num_blocks",
                "rt_kv_tail_rows",
                "rt_dense_prefix_blocks",
                "rt_sparse_end_block",
                "rt_record_count",
                "rt_records_per_worker",
                "rt_pack_row_tiles",
            ],
        )
        for name, parameters in zip(FAMILIES, expected, strict=True):
            main = next(
                node for node in ast.walk(_dsl_tree(name)) if isinstance(node, ast.FunctionDef) and node.name == "main"
            )
            self.assertEqual([parameter.arg for parameter in main.args.args], parameters)


if __name__ == "__main__":
    unittest.main()
