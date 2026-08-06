# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Staging of the Python model executor's per-platform kernel packages."""

import importlib.util
import os
import tempfile
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_setup_module():
    """Import ``setup.py`` as a module; its side effects sit under __main__."""
    spec = importlib.util.spec_from_file_location(
        "xllm_setup", os.path.join(_REPO_ROOT, "setup.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PythonKernelStagingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.setup = _load_setup_module()

    def _stage(self, device: str, peers: tuple[str, ...]) -> list[str]:
        with tempfile.TemporaryDirectory() as source_root:
            for peer in peers:
                os.makedirs(os.path.join(source_root, f"kernels_{peer}"))
                open(
                    os.path.join(source_root, f"kernels_{peer}", "__init__.py"),
                    "w",
                    encoding="utf-8",
                ).close()
            with tempfile.TemporaryDirectory() as dest_root:
                self.setup._stage_python_kernel_package(
                    source_root, dest_root, device
                )
                return sorted(os.listdir(dest_root))

    def test_only_the_devices_own_package_is_staged(self) -> None:
        self.assertEqual(
            self._stage("cuda", peers=("cuda", "npu")), ["kernels_cuda"]
        )

    def test_device_without_a_package_stages_none(self) -> None:
        # xLLM builds for more devices than the Python model executor covers,
        # so a device with no peer package must not fail the build: the wheel
        # ships without kernels and xllm/python/__init__.py rejects the
        # platform at import.
        self.assertEqual(self._stage("mlu", peers=("cuda", "npu")), [])


if __name__ == "__main__":
    unittest.main()
