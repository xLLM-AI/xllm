# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Host dispatch contract only; no Torch import, kernel, or model execution.

Load the real method bodies from the source AST to keep these branch checks
independent of CUDA initialization. Opaque metadata and dtype values are only
forwarded to the external planner. Numerical semantics require GPU qualification.
"""

from __future__ import annotations
import __future__

import ast
import builtins
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import Mock

_SOURCE = Path(__file__).resolve().parents[2] / "xllm/python/attention/flashinfer_native_kv.py"


def _load_method(
    name: str,
    *,
    imports: dict[str, Any] | None = None,
    globals_: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    tree = ast.parse(_SOURCE.read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FlashInferNativeKVBackend"
    )
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)

    def _resolve_import(module: str, *args: Any, **kwargs: Any) -> Any:
        if imports is not None and module in imports:
            return imports[module]
        raise AssertionError(f"Unexpected device dispatch import: {module}")

    scope = {**(globals_ or {}), "__builtins__": {**vars(builtins), "__import__": _resolve_import}}
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(_SOURCE), "exec", flags=__future__.annotations.compiler_flag), scope)
    return scope[name]


class NativeKVPrefillDispatchTest(unittest.TestCase):
    def test_prefill_preserves_query_dtype_and_full_packed_metadata(self) -> None:
        prepare = _load_method("prepare")
        for dtype in ("BF16", "FP16"):
            for prefill in ("fa2", "fa3"):
                for phase in ("prefill", "chunked"):
                    with self.subTest(dtype=dtype, prefill=prefill, phase=phase):
                        pointers = [object() for _ in range(4)]
                        metadata = SimpleNamespace(
                            is_prefill=phase == "prefill",
                            is_chunked_prefill=phase == "chunked",
                            qo_indptr=pointers[0],
                            paged_kv_indptr=pointers[1],
                            paged_kv_indices=pointers[2],
                            paged_kv_last_page_len=pointers[3],
                        )
                        query_dtype = "FP8" if prefill == "fa3" else dtype
                        backend = SimpleNamespace(
                            _cache_format="fp8_e4m3",
                            _prefill_backend=prefill,
                            _decode_backend="xqa",
                            sliding_window=0,
                            scale=256**-0.5,
                            dtype=dtype,
                            _cache_dtype="FP8",
                            _prefill_query_dtype=query_dtype,
                            _prefill_paged_wrapper=Mock(),
                            num_heads=24,
                            num_kv_heads=4,
                            head_dim=256,
                            page_size=128,
                        )
                        prepare(backend, metadata)
                        backend._prefill_paged_wrapper.plan.assert_called_once_with(
                            *pointers,
                            24,
                            4,
                            256,
                            128,
                            causal=True,
                            sm_scale=256**-0.5,
                            window_left=-1,
                            q_data_type=query_dtype,
                            kv_data_type="FP8",
                            o_data_type=dtype,
                        )

    def test_pure_decode_preserves_eager_and_graph_xqa_plan(self) -> None:
        constructor = Mock()
        prepare = _load_method(
            "prepare",
            imports={"xllm.python.attention.xqa_decode": SimpleNamespace(XQADecodePlan=constructor)},
        )
        for graph_mode in (False, True):
            with self.subTest(graph_mode=graph_mode):
                plan = Mock(batch_size=2)
                metadata = SimpleNamespace(
                    is_prefill=False,
                    is_chunked_prefill=False,
                    paged_kv_last_page_len=SimpleNamespace(numel=lambda: 2),
                )
                backend = SimpleNamespace(
                    sliding_window=0,
                    scale=256**-0.5,
                    dtype="BF16",
                    _cache_dtype="FP8",
                    _decode_backend="xqa",
                    _graph_xqa_plans={2: plan} if graph_mode else {},
                    _eager_xqa_plan=None if graph_mode else plan,
                    page_size=128,
                )
                prepare(backend, metadata, graph_mode=graph_mode)
                plan.prepare.assert_called_once_with(metadata, graph_mode=graph_mode)
                self.assertIs(backend._active_xqa_plan, plan)
        constructor.assert_not_called()

    def test_execute_keeps_full_fa3_and_pure_xqa_scale_contracts(self) -> None:
        for prefill in (False, True):
            for value_scale in (0.0625, 0.5):
                with self.subTest(prefill=prefill, value_scale=value_scale):
                    writer, quantize = Mock(), Mock()
                    execute = _load_method(
                        "execute",
                        globals_={"torch": SimpleNamespace(float8_e4m3fn="FP8"), "write_fp8_kv": writer},
                        imports={
                            "xllm.python.attention.query_quantization": SimpleNamespace(quantize_query_fp8=quantize)
                        },
                    )
                    q, k, v = Mock(), Mock(), Mock()
                    pages, workspace = object(), object()
                    metadata = SimpleNamespace(is_prefill=False, is_chunked_prefill=prefill, slot_mapping=object())
                    backend = SimpleNamespace(
                        _metadata=metadata,
                        _cache_format="fp8_e4m3",
                        _pages={1: pages},
                        num_heads=24,
                        num_kv_heads=4,
                        head_dim=256,
                        _fixed_scales=(0.125, value_scale),
                        _prefill_query_dtype="FP8",
                        _decode_backend="xqa",
                        _prefill_paged_wrapper=Mock(),
                        _active_xqa_plan=Mock(),
                        _decode_workspace=workspace,
                        scale=256**-0.5,
                    )
                    execute(backend, q, k, v, SimpleNamespace(layer_id=1))
                    writer.assert_called_once_with(
                        k.view.return_value,
                        v.view.return_value,
                        metadata.slot_mapping,
                        pages,
                        (0.125, value_scale),
                    )
                    if prefill:
                        quantize.assert_called_once_with(q.view.return_value, 1.0)
                        backend._prefill_paged_wrapper.run.assert_called_once_with(
                            quantize.return_value,
                            pages,
                            k_scale=0.125,
                            v_scale=value_scale,
                            q_scale=1.0,
                        )
                        backend._active_xqa_plan.run.assert_not_called()
                    else:
                        quantize.assert_not_called()
                        backend._active_xqa_plan.run.assert_called_once_with(
                            q.view.return_value,
                            pages,
                            workspace,
                            256**-0.5,
                            0.125,
                            value_scale,
                        )
                        backend._prefill_paged_wrapper.run.assert_not_called()

    def test_missing_decode_plan_does_not_modify_kv(self) -> None:
        writer = Mock()
        execute = _load_method("execute", globals_={"write_fp8_kv": writer})
        backend = SimpleNamespace(
            _metadata=SimpleNamespace(is_prefill=False, is_chunked_prefill=False),
            _decode_backend="xqa",
            _active_xqa_plan=None,
        )
        with self.assertRaisesRegex(RuntimeError, "before writing KV"):
            execute(backend, object(), object(), object(), object())
        writer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
