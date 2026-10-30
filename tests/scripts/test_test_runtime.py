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


import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from setuptools import Distribution


class TestRunnerEnvironmentTest(unittest.TestCase):
    def setUp(self) -> None:
        setup_path = Path(__file__).resolve().parents[2] / "setup.py"
        spec = importlib.util.spec_from_file_location("xllm_setup_test_runtime", setup_path)
        self.setup_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.setup_module)

    def test_ctest_phases_share_one_isolated_cache(self) -> None:
        command = self.setup_module.TestUT(Distribution())
        process = mock.Mock()
        process.stdout.readline.return_value = ""
        process.wait.return_value = 0
        with (
            tempfile.TemporaryDirectory() as build_dir,
            mock.patch.dict(os.environ, {"CTEST_PARALLEL": "4"}, clear=True),
            mock.patch.object(self.setup_module.subprocess, "Popen", return_value=process) as popen,
        ):
            self.assertEqual(command.run_ctest(build_dir), 0)
            self.assertEqual(popen.call_count, 1 + len(command.SEQUENTIAL_TESTS))
            environments = [call.kwargs["env"] for call in popen.call_args_list]
            self.assertEqual(len({env["TORCH_EXTENSIONS_DIR"] for env in environments}), 1)
            self.assertFalse(Path(environments[0]["TORCH_EXTENSIONS_DIR"]).exists())
            self.assertIn("4", popen.call_args_list[0].args[0])

            first_cache = environments[0]["TORCH_EXTENSIONS_DIR"]
            popen.reset_mock()
            self.assertEqual(command.run_ctest(build_dir), 0)
            self.assertNotEqual(popen.call_args.kwargs["env"]["TORCH_EXTENSIONS_DIR"], first_cache)
            explicit_cache = Path(build_dir) / "explicit-cache"
            explicit_cache.mkdir()
            with mock.patch.dict(os.environ, {"TORCH_EXTENSIONS_DIR": str(explicit_cache)}):
                self.assertEqual(command.run_ctest(build_dir), 0)
                self.assertEqual(popen.call_args.kwargs["env"]["TORCH_EXTENSIONS_DIR"], str(explicit_cache))
            self.assertTrue(explicit_cache.is_dir())
            process.wait.return_value = 1
            with self.assertRaises(SystemExit):
                command.run_ctest(build_dir)
            self.assertFalse(Path(popen.call_args.kwargs["env"]["TORCH_EXTENSIONS_DIR"]).exists())

    def test_single_target_uses_isolated_cache(self) -> None:
        command = self.setup_module.ExtBuildSingleTest(Distribution())
        command.test_name = "example_test"
        with (
            tempfile.TemporaryDirectory() as build_dir,
            mock.patch.object(self.setup_module, "get_cmake_dir", return_value=build_dir),
            mock.patch.object(command, "_configure_cmake"),
            mock.patch.object(self.setup_module.subprocess, "check_call") as check_call,
        ):
            executable = Path(build_dir) / command.test_name
            executable.touch(mode=0o755)
            environment = {"PATH": "/bin"}
            command.build_cmake_targets(None, [], [], environment, build_dir, "unused")
            invocation = check_call.call_args
            self.assertEqual(invocation.args[0], [str(executable)])
            self.assertFalse(Path(invocation.kwargs["env"]["TORCH_EXTENSIONS_DIR"]).exists())

            first_cache = invocation.kwargs["env"]["TORCH_EXTENSIONS_DIR"]
            command.build_cmake_targets(None, [], [], environment, build_dir, "unused")
            self.assertNotEqual(check_call.call_args.kwargs["env"]["TORCH_EXTENSIONS_DIR"], first_cache)
            self.assertNotIn("TORCH_EXTENSIONS_DIR", environment)
            check_call.side_effect = [None, subprocess.CalledProcessError(1, ["example_test"])]
            with self.assertRaises(subprocess.CalledProcessError):
                command.build_cmake_targets(None, [], [], environment, build_dir, "unused")
            self.assertFalse(Path(check_call.call_args.kwargs["env"]["TORCH_EXTENSIONS_DIR"]).exists())


if __name__ == "__main__":
    unittest.main()
