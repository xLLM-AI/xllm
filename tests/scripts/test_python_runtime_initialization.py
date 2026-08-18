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

import os
import subprocess
import sys
import textwrap
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PythonRuntimeInitializationTest(unittest.TestCase):
    def _run_probe(self, source: str) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = _REPO_ROOT
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)],
            cwd=_REPO_ROOT,
            env=env,
            check=True,
        )

    def test_package_import_does_not_initialize_runtime(self) -> None:
        self._run_probe(
            """
            import sys

            import xllm.python

            assert "xllm.python.platform" not in sys.modules
            assert "xllm.python.kernels_npu" not in sys.modules

            import xllm.python.kernels_npu
            import xllm.python.kernels_npu.tilelang
            import xllm.python.kernels_npu.triton

            assert "xllm.python.kernels_npu._custom_op" not in sys.modules
            assert "xllm.python.kernels_npu.normalization" not in sys.modules

            try:
                xllm.python.kernels
            except RuntimeError as error:
                assert "initialize_runtime" in str(error)
            else:
                raise AssertionError("kernel access initialized the runtime implicitly")
            """
        )

    def test_explicit_runtime_initialization_is_idempotent(self) -> None:
        self._run_probe(
            """
            import sys
            import types

            import xllm.python

            calls = []
            platform_module = types.ModuleType("xllm.python.platform")
            platform_module.current_platform = types.SimpleNamespace(
                is_cuda=lambda: False,
                is_npu=lambda: True,
                device_type=lambda: "npu",
            )
            backend = types.ModuleType("xllm.python.kernels_npu")
            backend._initialize_runtime = lambda: calls.append("initialized")
            sys.modules[platform_module.__name__] = platform_module
            sys.modules[backend.__name__] = backend

            xllm.python.initialize_runtime()
            xllm.python.initialize_runtime()

            assert calls == ["initialized"]
            assert xllm.python.kernels is backend
            assert sys.modules["xllm.python.kernels"] is backend
            """
        )

    def test_npu_runtime_initialization_is_idempotent(self) -> None:
        self._run_probe(
            """
            import types

            import xllm.python.kernels_npu as kernels_npu

            calls = []
            custom_op_module = types.ModuleType(
                "xllm.python.kernels_npu._custom_op"
            )
            semantic_module = types.ModuleType(
                "xllm.python.kernels_npu.semantic"
            )
            semantic_module.exported_kernel = object()

            def import_module(name):
                calls.append(name)
                if name == custom_op_module.__name__:
                    return custom_op_module
                if name == semantic_module.__name__:
                    return semantic_module
                raise AssertionError(f"unexpected import: {name}")

            kernels_npu._EXPORTS = {"semantic": ("exported_kernel",)}
            kernels_npu.importlib = types.SimpleNamespace(
                import_module=import_module
            )

            kernels_npu._initialize_runtime()
            kernels_npu._initialize_runtime()

            assert calls == [custom_op_module.__name__, semantic_module.__name__]
            assert kernels_npu.exported_kernel is semantic_module.exported_kernel
            """
        )


if __name__ == "__main__":
    unittest.main()
