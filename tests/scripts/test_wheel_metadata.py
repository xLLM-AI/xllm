# Copyright 2026 The xLLM Authors.
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

"""Wheel distribution metadata tests."""

import os
import tempfile
import unittest
from unittest import mock

from setuptools import Distribution

from scripts.build_support.utils import load_setup_module as _load_setup_module

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class WheelMetadataTest(unittest.TestCase):
    def test_wheel_name_refreshes_cached_metadata(self) -> None:
        setup_module = _load_setup_module()
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch.object(setup_module, "get_torch_version", return_value="2.10.0"),
        ):
            distribution = Distribution(
                {
                    "name": "xllm",
                    "version": "0.11.0",
                    "packages": [],
                    "cmdclass": {"bdist_wheel": setup_module.BuildDistWheel},
                }
            )
            distribution.script_name = os.path.join(_REPO_ROOT, "setup.py")
            distribution.command_options = {
                "egg_info": {"egg_base": ("test", temp_dir)},
                "bdist_wheel": {"device": ("test", "mlu")},
            }

            distribution.get_command_obj("egg_info").ensure_finalized()
            distribution.run_command("egg_info")
            distribution.get_command_obj("bdist_wheel").ensure_finalized()
            distribution.run_command("egg_info")

            egg_info = distribution.get_command_obj("egg_info")
            with open(os.path.join(egg_info.egg_info, "PKG-INFO"), encoding="utf-8") as metadata:
                self.assertIn("Name: xllm_mlu_torch2.10.0\n", metadata.read())


if __name__ == "__main__":
    unittest.main()
