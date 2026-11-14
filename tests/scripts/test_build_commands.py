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

"""Build command boundaries without compiling native dependencies."""

import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from setuptools import Command, Distribution

from scripts.build_support.utils import load_setup_module as _load_setup_module


class BuildCommandsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.setup = _load_setup_module()
        self._stack = ExitStack()
        self.addCleanup(self._stack.close)
        self.root = Path(self._stack.enter_context(tempfile.TemporaryDirectory()))
        self.cmake_dir = self.root / "build" / "cmake"
        self.extdir = self.root / "build" / "lib" / "xllm"
        self.cmake_dir.mkdir(parents=True)
        self.extdir.mkdir(parents=True)
        (self.cmake_dir / "mooncake_ha_runtime_libraries.txt").touch()
        (self.extdir / "xllm").write_text("native executable", encoding="utf-8")
        compiler = self.root / "xllm" / "compiler"
        (compiler / "triton").mkdir(parents=True)
        (self.root / "xllm" / "__init__.py").write_text("STAGED_ROOT = True\n", encoding="utf-8")
        for name in ("__init__.py", "triton/__init__.py", "triton/compile.py"):
            (compiler / name).write_text("STAGED = True\n", encoding="utf-8")
        kernels = self.root / "xllm" / "core" / "kernels" / "mlu" / "triton_kernel"
        kernels.mkdir(parents=True)
        (kernels / "fixture_kernel.py").write_text("STAGED = True\n", encoding="utf-8")
        self._stack.enter_context(mock.patch.object(self.setup, "get_cmake_dir", return_value=str(self.cmake_dir)))
        self._stack.enter_context(mock.patch.object(self.setup.TestUT, "run_ctest"))
        self.ext = self.setup.CMakeExtension("xllm", "xllm/")
        self.env = {"PATH": "/toolchain/bin", "CCACHE_BASEDIR": str(self.root)}

    def _command(self, command_type: type[Command], device: str = "npu") -> Command:
        command = command_type(Distribution({"ext_modules": [self.ext]}))
        command.base_dir = str(self.root)
        command.device = device
        command.arch = "x86"
        return command

    def _build_targets(self, command: Command) -> None:
        command.build_cmake_targets(self.ext, [], ["--config", "Release", "-j8"], self.env, str(self.extdir), "xllm")

    def test_product_build_excludes_tests_and_shares_one_build_invocation(self) -> None:
        command = self._command(self.setup.ExtBuild)
        for export in (True, False):
            with self.subTest(export=export), ExitStack() as stack:
                stack.enter_context(mock.patch.object(self.setup, "BUILD_EXPORT", export))
                for stage in (
                    "_stage_triton_npu_runtime_binaries",
                    "_stage_npu_ops_vendor",
                    "_stage_mooncake_runtime_binaries",
                    "_stage_mlu_triton_kernels",
                    "_stage_triton_compiler_package",
                    "_stage_auto_tuning_config",
                ):
                    stack.enter_context(mock.patch.object(self.setup, stage))
                call = stack.enter_context(mock.patch.object(self.setup.subprocess, "check_call"))

                self._build_targets(command)

                self.setup._stage_npu_ops_vendor.assert_called_once_with(
                    str(self.root), str(self.extdir), "npu", self.env
                )

                configure, build = call.call_args_list
                self.assertEqual(configure.args[0][-1], "-DBUILD_TESTING=ON")
                targets = build.args[0][build.args[0].index("--target") + 1 :]
                expected = ["xllm", "mooncake_master", "mooncake_client", "stage_mooncake_ha_runtime"]
                if export:
                    expected.append("export_module")
                self.assertEqual(targets, expected)
                self.assertEqual(build.kwargs["env"], self.env)

        self.assertEqual(
            (self.root / "build" / "xllm" / "core" / "server" / "xllm").read_text(encoding="utf-8"),
            "native executable",
        )

    def test_nested_ops_build_inherits_job_budget_and_preserves_override(self) -> None:
        for override, expected in (({}, "8"), ({"OPS_CPU_NUMBER": "4"}, "4")):
            with self.subTest(override=override):
                command = self._command(self.setup.ExtBuild)
                command.build_temp = str(self.cmake_dir)
                with (
                    mock.patch.dict(os.environ, {"MAX_JOBS": "8", **override}, clear=True),
                    mock.patch.object(command, "get_ext_fullpath", return_value=str(self.extdir / "xllm")),
                    mock.patch.object(self.setup, "set_npu_envs"),
                    mock.patch.object(self.setup, "_maybe_compile_tilelang_kernels"),
                    mock.patch.object(self.setup, "get_cxx_abi", return_value=True),
                    mock.patch.object(self.setup, "configure_ccache"),
                    mock.patch.object(command, "build_cmake_targets") as build,
                ):
                    command.build_extension(self.ext)
                self.assertEqual(build.call_args.args[3]["OPS_CPU_NUMBER"], expected)
                self.assertIn("-j8", build.call_args.args[2])

    def test_tilelang_aot_uses_the_h3_cmake_profile(self) -> None:
        profile = "h3_a3_rf_cann91_cp312"
        overrides = [
            f"-DXLLM_BUILD_PROFILE:STRING={profile}",
            "-DXLLM_ENABLE_H3_RAINFUSION:BOOL=ON",
        ]
        with (
            mock.patch.object(self.setup, "get_ascend_platform", return_value="a3"),
            mock.patch.object(self.setup.subprocess, "check_call") as call,
        ):
            self.setup._maybe_compile_tilelang_kernels([])
            self.assertNotIn("--build-profile", call.call_args.args[0])
            self.setup._maybe_compile_tilelang_kernels(overrides)
            self.assertEqual(
                call.call_args.args[0][-3:],
                ["--build-profile", profile, "--enable-h3-rainfusion"],
            )
            with self.assertRaises(ValueError):
                self.setup._maybe_compile_tilelang_kernels(overrides[:1])
            self.assertEqual(call.call_count, 2)

    def test_test_command_builds_the_test_tree(self) -> None:
        for device, cmake in (("npu", "cmake"), ("maca", "cmake_maca")):
            with self.subTest(device=device), mock.patch.object(self.setup.subprocess, "check_call") as call:
                self._build_targets(self._command(self.setup.TestUT, device))

                configure, build = call.call_args_list
                self.assertEqual(configure.args[0][-1], "-DBUILD_TESTING=ON")
                self.assertEqual(build.args[0][0], cmake)
                self.assertEqual(build.args[0][-2:], ["--target", "all_tests"])
                self.assertEqual(build.kwargs["env"], self.env)
                self.assertFalse((self.root / "build" / "xllm").exists())

    def test_command_controls_test_registration_after_opposite_configuration(self) -> None:
        commands = (
            (self.setup.ExtBuild, "OFF", "ON"),
            (self.setup.TestUT, "OFF", "ON"),
            (self.setup.ExtBuildSingleTest, "OFF", "ON"),
        )
        for command_type, previous, expected in commands:
            with self.subTest(command=command_type.__name__):
                command = self._command(command_type)
                with mock.patch.object(self.setup.subprocess, "check_call") as call:
                    command._configure_cmake("cmake", [f"-DBUILD_TESTING={previous}"], self.env)
                self.assertEqual(call.call_args.args[0][-1], f"-DBUILD_TESTING={expected}")

    def test_standalone_mlu_tests_can_import_staged_runtime(self) -> None:
        command = self._command(self.setup.TestUT, "mlu")
        with mock.patch.object(self.setup.subprocess, "check_call"):
            self._build_targets(command)

        installed = self.root / "installed"
        (installed / "xllm").mkdir(parents=True)
        (installed / "xllm" / "__init__.py").write_text(
            "raise RuntimeError('stale installed wheel')\n", encoding="utf-8"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "from xllm.compiler.triton import compile; "
                "from xllm.core.kernels.mlu.triton_kernel import fixture_kernel; "
                "assert compile.STAGED and fixture_kernel.STAGED",
            ],
            cwd=self.cmake_dir,
            env={**os.environ, "PYTHONPATH": os.pathsep.join((str(self.extdir.parent), str(installed)))},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_single_test_builds_and_runs_only_selected_target(self) -> None:
        command = self._command(self.setup.ExtBuildSingleTest)
        command.test_name = "slot_buffer_test"
        executable = self.extdir / command.test_name
        executable.touch(mode=0o755)

        with mock.patch.object(self.setup.subprocess, "check_call") as call:
            self._build_targets(command)

        configure, build, run = call.call_args_list
        self.assertEqual(configure.args[0][-1], "-DBUILD_TESTING=ON")
        self.assertEqual(build.args[0][-2:], ["--target", "slot_buffer_test"])
        self.assertEqual(run.args[0], [str(executable)])

    def test_unsupported_test_backend_fails_before_configuration(self) -> None:
        command = self._command(self.setup.TestUT, "musa")
        with (
            mock.patch.object(self.setup.subprocess, "check_call") as call,
            self.assertRaisesRegex(ValueError, "not supported.*musa"),
        ):
            self._build_targets(command)
        call.assert_not_called()

    def test_test_command_builds_before_running_ctest(self) -> None:
        command = self._command(self.setup.TestUT)
        calls = mock.Mock()
        calls.attach_mock(mock.Mock(), "native")
        calls.attach_mock(mock.Mock(), "ctest")
        with (
            mock.patch.object(self.setup.subprocess, "check_call", calls.native),
            mock.patch.object(command, "run_ctest", calls.ctest),
        ):
            self._build_targets(command)
        self.assertEqual([call[0] for call in calls.mock_calls], ["native", "native", "ctest"])
        self.assertEqual(calls.mock_calls[1].args[0][-2:], ["--target", "all_tests"])
        calls.ctest.assert_called_once_with(str(self.cmake_dir))

    def test_test_build_failure_prevents_running_stale_tests(self) -> None:
        command = self._command(self.setup.TestUT)
        with (
            mock.patch.object(
                self.setup.subprocess,
                "check_call",
                side_effect=[None, subprocess.CalledProcessError(1, "cmake")],
            ),
            mock.patch.object(command, "run_ctest") as run_ctest,
            self.assertRaises(subprocess.CalledProcessError),
        ):
            self._build_targets(command)
        run_ctest.assert_not_called()

    def test_setuptools_build_test_dispatches_both_commands_with_build_options(self) -> None:
        distribution = Distribution(
            {
                "ext_modules": [self.ext],
                "cmdclass": {"build_ext": self.setup.ExtBuild, "test": self.setup.TestUT},
                "script_name": "setup.py",
                "script_args": ["build", "test"],
            }
        )
        options = {"device": "npu", "arch": "x86", "dev": True, "tilelang_jobs": 4}
        distribution.command_options = {
            name: {key: ("test", value) for key, value in options.items()} for name in ("build_ext", "test")
        }
        self.assertTrue(distribution.parse_command_line())
        commands = []

        def record_build(command: Command, ext: object) -> None:
            commands.append(type(command))
            self.assertIs(ext, self.ext)
            for key, value in options.items():
                self.assertEqual(getattr(command, key), value)

        with (
            mock.patch.object(self.setup.subprocess, "check_output", return_value=b"cmake version 3.26.0"),
            mock.patch.object(self.setup.ExtBuild, "build_extension", autospec=True, side_effect=record_build),
        ):
            distribution.run_commands()
        self.assertEqual(commands, [self.setup.ExtBuild, self.setup.TestUT])

    def test_wheel_builds_product_without_invoking_tests(self) -> None:
        for device, skip in (
            ("npu", False),
            ("npu", True),
            ("musa", False),
        ):
            with self.subTest(device=device, skip=skip):
                command = self._command(self.setup.BuildDistWheel, device)
                with (
                    mock.patch.dict(os.environ, {"SKIP_TEST": "1"} if skip else {}, clear=True),
                    mock.patch.object(command, "get_finalized_command"),
                    mock.patch.object(command, "run_command") as run,
                    mock.patch.object(self.setup, "get_base_dir", return_value=str(self.root)),
                    mock.patch.object(self.setup.bdist_wheel, "run"),
                ):
                    command.run()
                self.assertEqual(run.call_args_list, [mock.call("build")])
                self.assertTrue(command.skip_build)
                self.assertFalse(self.setup.InstallWheel._building_wheel)


if __name__ == "__main__":
    unittest.main()
