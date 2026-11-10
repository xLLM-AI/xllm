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

"""Collect complete, matching NPU operators without an Ascend installation."""

import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from setuptools import Distribution
from setuptools.command.install_lib import install_lib

from scripts.build_support import utils
from scripts.build_support.npu_ops import stage_npu_ops_vendor
from scripts.build_support.utils import get_python_version

_REPO_ROOT = Path(__file__).resolve().parents[2]


class NpuOpsStagingTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.ops_source = self.source / "third_party" / "xllm_ops"
        self.ops_source.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(self.ops_source)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(self.ops_source),
                "-c",
                "user.name=xLLM test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "--allow-empty",
                "-qm",
                "operator fixture",
            ],
            check=True,
        )
        self.git_head = subprocess.check_output(
            ["git", "-C", str(self.ops_source), "rev-parse", "HEAD"], text=True
        ).strip()
        self.toolkit = self.root / "cann"
        self.vendor = self.toolkit / "opp" / "vendors" / "custom_xllm_math"
        self.extdir = self.root / "build" / "lib" / "xllm"
        self.destination = self.extdir / "opp" / "vendors" / "custom_xllm_math"
        self.env = {"ASCEND_OPP_PATH": str(self.toolkit / "opp"), "NPU_HOME_PATH": str(self.toolkit)}
        self.files = {
            ".xllm_ops_git_head": self.git_head + "\n",
            "version.info": "Version=9.0.0\n",
            "bin/set_env.bash": "export ASCEND_CUSTOM_OPP_PATH=build-machine-only\n",
            "op_api/lib/libcust_opapi.so": "operator API",
            "op_api/include/aclnnop/aclnn_example.h": "operator API header",
            "op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so": "operator prototypes",
            "op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json": "{}",
            "op_impl/ai_core/tbe/kernel/ascend910_93/example/kernel.o": "compiled kernel",
            "op_impl/ai_core/tbe/kernel/ascend910_93/example/kernel.json": "{}",
            "op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so": "operator tiling",
            "op_impl/cpu/aicpu_kernel/impl/libmath_aicpu_kernels.so": "CPU kernels",
            "op_impl/cpu/config/cust_aicpu_kernel.json": "{}",
            "scripts/uninstall.sh": "vendor uninstall script",
        }
        for name, contents in self.files.items():
            resource = self.vendor / name
            resource.parent.mkdir(parents=True, exist_ok=True)
            resource.write_text(contents, encoding="utf-8")

    def _stage(self, device: str = "npu", env: dict[str, str] | None = None) -> None:
        stage_npu_ops_vendor(str(self.source), str(self.extdir), device, self.env if env is None else env)

    def _use_legacy_vendor(self) -> None:
        legacy = self.vendor.parent / "xllm"
        self.vendor.rename(legacy)
        self.vendor = legacy
        old_header = "op_api/include/aclnnop/aclnn_example.h"
        legacy_header = "op_api/include/aclnn_example.h"
        (self.vendor / old_header).rename(self.vendor / legacy_header)
        (self.vendor / "op_api/include/aclnnop").rmdir()
        self.files[legacy_header] = self.files.pop(old_header)

    def test_normalizes_complete_legacy_vendor_to_bundled_path(self) -> None:
        self._use_legacy_vendor()
        self._stage()
        shutil.rmtree(self.vendor)
        for name, contents in self.files.items():
            with self.subTest(resource=name):
                self.assertEqual((self.destination / name).read_text(encoding="utf-8"), contents)
        self.assertFalse((self.destination.parent / "xllm").exists())

    def test_prefers_current_layout_when_both_vendors_are_installed(self) -> None:
        legacy = self.vendor.parent / "xllm"
        shutil.copytree(self.vendor, legacy)
        (legacy / ".xllm_ops_git_head").write_text("older revision\n", encoding="utf-8")
        self._stage()
        self.assertEqual((self.destination / ".xllm_ops_git_head").read_text().strip(), self.git_head)

    def test_uses_legacy_layout_when_current_header_probe_is_absent(self) -> None:
        self._use_legacy_vendor()
        leftover = self.vendor.parent / "custom_xllm_math"
        leftover.mkdir()
        (leftover / ".xllm_ops_git_head").write_text("older revision\n", encoding="utf-8")
        self._stage()
        self.assertTrue((self.destination / "op_api/include/aclnn_example.h").is_file())

    def test_does_not_fall_back_to_legacy_when_selected_current_vendor_is_invalid(self) -> None:
        shutil.copytree(self.vendor, self.vendor.parent / "xllm")
        (self.vendor / ".xllm_ops_git_head").write_text("older revision\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "NPU operator revision mismatch"):
            self._stage()
        self.assertFalse(self.destination.exists())

    def test_rejects_legacy_vendor_with_different_operator_revision(self) -> None:
        self._use_legacy_vendor()
        (self.vendor / ".xllm_ops_git_head").write_text("older revision\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "NPU operator revision mismatch"):
            self._stage()
        self.assertFalse(self.destination.exists())

    def test_preserves_complete_vendor_after_build_installation_is_removed(self) -> None:
        self._stage()
        shutil.rmtree(self.vendor)
        for name, contents in self.files.items():
            with self.subTest(resource=name):
                self.assertEqual((self.destination / name).read_text(encoding="utf-8"), contents)

        # Wheel installation copies this build tree using install_lib; native
        # kernel objects and metadata must survive that boundary as well.
        command = install_lib(Distribution())
        command.build_dir = str(self.extdir.parent)
        command.install_dir = str(self.root / "installed")
        command.install()
        installed = Path(command.install_dir) / "xllm" / "opp" / "vendors" / "custom_xllm_math"
        for name, contents in self.files.items():
            with self.subTest(installed_resource=name):
                self.assertEqual((installed / name).read_text(encoding="utf-8"), contents)

    def test_materializes_external_file_and_directory_symlinks(self) -> None:
        external = self.root / "external"
        external.mkdir()
        library = external / "operator.so"
        library.write_text("linked operator API", encoding="utf-8")
        api = self.vendor / "op_api" / "lib" / "libcust_opapi.so"
        api.unlink()
        api.symlink_to(library)
        kernels = self.vendor / "op_impl" / "ai_core" / "tbe" / "kernel"
        external_kernels = external / "kernels"
        kernels.rename(external_kernels)
        kernels.symlink_to(external_kernels, target_is_directory=True)

        self._stage()
        shutil.rmtree(external)
        shutil.rmtree(self.vendor)

        self.assertEqual(
            (self.destination / "op_api" / "lib" / "libcust_opapi.so").read_text(encoding="utf-8"),
            "linked operator API",
        )
        self.assertEqual(
            (self.destination / "op_impl/ai_core/tbe/kernel/ascend910_93/example/kernel.o").read_text(encoding="utf-8"),
            "compiled kernel",
        )
        self.assertFalse(any(resource.is_symlink() for resource in self.destination.rglob("*")))

    def test_actual_wheel_preserves_complete_vendor_and_test_named_kernel(self) -> None:
        self._assert_wheel_preserves_complete_vendor()

    def test_actual_wheel_preserves_complete_legacy_vendor(self) -> None:
        self._use_legacy_vendor()
        self._assert_wheel_preserves_complete_vendor()

    def test_install_builds_complete_wheel_without_invoking_tests(self) -> None:
        self._assert_wheel_preserves_complete_vendor(command="install")

    def _assert_wheel_preserves_complete_vendor(self, command: str = "bdist_wheel") -> None:
        kernel = "op_impl/ai_core/tbe/kernel/ascend910_93/example/kernel_test.o"
        self.files[kernel] = "a production kernel whose name contains _test"
        (self.vendor / kernel).write_text(self.files[kernel], encoding="utf-8")
        self._stage()
        shutil.rmtree(self.vendor)

        build_lib = self.root / "build" / f"lib.linux-x86_64-cpython-{get_python_version()}"
        shutil.copytree(self.extdir.parent, build_lib)
        shutil.copy2(_REPO_ROOT / "xllm/_npu_runtime.py", build_lib / "xllm/_npu_runtime.py")
        (build_lib / "xllm/xllm").write_text("native host fixture", encoding="utf-8")
        (build_lib / "xllm" / "slot_buffer_test").write_text("native unit test", encoding="utf-8")
        script = textwrap.dedent(
            """
            import os
            import sys
            from pathlib import Path
            from unittest import mock
            from setuptools import Distribution
            from scripts.build_support.utils import load_setup_module

            setup = load_setup_module()
            setup.get_base_dir = lambda: sys.argv[1]
            setup.get_torch_version = lambda device: None
            distribution = Distribution({
                "name": "xllm_fixture",
                "version": "0.0",
                "packages": [],
                "ext_modules": [setup.CMakeExtension("xllm", "xllm/")],
                "cmdclass": {
                    "bdist_wheel": setup.BuildDistWheel,
                    "install": setup.InstallWheel,
                    "test": setup.TestUT,
                },
            })
            distribution.script_name = "setup.py"
            distribution.command_options = {"build": {"build_lib": ("fixture", sys.argv[2])}}
            build = distribution.get_command_obj("build")
            wheel = distribution.get_command_obj("bdist_wheel")
            wheel.device = "npu"
            wheel.arch = "x86"
            wheel.plat_name = "linux_x86_64"
            wheel.dist_dir = sys.argv[3]
            wheel.ensure_finalized()
            os.environ.pop("SKIP_TEST", None)
            with (
                mock.patch.object(build, "run"),
                mock.patch.object(setup.TestUT, "run", side_effect=AssertionError("unexpected unit tests")),
                mock.patch.object(setup.subprocess, "check_call") as install,
            ):
                distribution.run_command(sys.argv[4])
            if sys.argv[4] == "install":
                wheel_path = next(Path(wheel.dist_dir).glob("*.whl"))
                install.assert_called_once_with([
                    sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel_path)
                ])
            else:
                install.assert_not_called()
            """
        )
        env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
        dist_dir = self.root / "dist"
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.root), str(build_lib), str(dist_dir), command],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        wheels = list(dist_dir.glob("*.whl"))
        self.assertEqual(len(wheels), 1)
        with zipfile.ZipFile(wheels[0]) as wheel:
            for name, contents in self.files.items():
                with self.subTest(resource=name):
                    self.assertEqual(wheel.read(f"xllm/opp/vendors/custom_xllm_math/{name}").decode("utf-8"), contents)
            self.assertNotIn("xllm/slot_buffer_test", wheel.namelist())
            installed = self.root / "installed-wheel"
            wheel.extractall(installed)
        runtime = runpy.run_path(str(installed / "xllm/_npu_runtime.py"))
        bundle = (installed / "xllm/opp/vendors/custom_xllm_math").resolve()
        with mock.patch.dict(os.environ, {"ASCEND_OPP_PATH": "/base/opp"}, clear=True):
            runtime["configure_npu_runtime"](str(installed / "xllm/xllm"))
            self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], str(bundle))
            self.assertEqual(os.environ["LD_LIBRARY_PATH"], str(bundle / "op_api/lib"))
            self.assertEqual(os.environ["ASCEND_OPP_PATH"], "/base/opp")

    def test_rejects_missing_or_empty_required_resources(self) -> None:
        required = (
            ".xllm_ops_git_head",
            "version.info",
            "bin/set_env.bash",
            "op_api/lib/libcust_opapi.so",
            "op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so",
            "op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json",
            "op_impl/ai_core/tbe/kernel/ascend910_93/example/kernel.o",
            "op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so",
        )
        for name in required:
            resource = self.vendor / name
            for contents in (None, ""):
                with self.subTest(resource=name, contents=contents):
                    if contents is None:
                        resource.unlink()
                    else:
                        resource.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "Incomplete NPU operator vendor"):
                        self._stage()
                    self.assertFalse(self.destination.exists())
                    self.assertEqual(list(self.extdir.glob(".xllm-ops-*")), [])
                    resource.write_text(self.files[name], encoding="utf-8")

    def test_prebuild_reuses_matching_revision_and_invalidates_stale_cache(self) -> None:
        cache = self.root / "CMakeCache.txt"
        cache.write_text("XLLM_OPS_GIT_HEAD_CACHED:INTERNAL=old\nKEEP:BOOL=ON\n", encoding="utf-8")
        with (
            mock.patch.dict(os.environ, self.env, clear=True),
            mock.patch.object(utils, "set_npu_envs"),
            mock.patch.object(utils, "get_base_dir", return_value=str(self.source)),
            mock.patch.object(utils, "_get_cmake_cache_path", return_value=str(cache)),
            mock.patch.object(utils, "_get_cann_version_info", return_value=("fixture", "9.0.0")),
        ):
            utils._ensure_xllm_ops_rebuild_state("npu")
            self.assertEqual(os.environ["XLLM_OPS_GIT_HEAD_CACHED"], self.git_head)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.ops_source),
                    "-c",
                    "user.name=xLLM test",
                    "-c",
                    "user.email=test@example.com",
                    "commit",
                    "--allow-empty",
                    "-qm",
                    "updated operator fixture",
                ],
                check=True,
            )
            utils._ensure_xllm_ops_rebuild_state("npu")
            self.assertNotIn("XLLM_OPS_GIT_HEAD_CACHED", os.environ)
            self.assertEqual(cache.read_text(), "KEEP:BOOL=ON\n")

    def test_rejects_different_operator_revision(self) -> None:
        (self.vendor / ".xllm_ops_git_head").write_text("different revision\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "NPU operator revision mismatch"):
            self._stage()
        self.assertFalse(self.destination.exists())

    def test_rejects_uninitialized_operator_submodule(self) -> None:
        shutil.rmtree(self.ops_source / ".git")
        with self.assertRaisesRegex(RuntimeError, "Initialize the xllm_ops submodule"):
            self._stage()

    def test_uses_current_opp_instead_of_unrelated_ascend_home(self) -> None:
        self._stage(env={**self.env, "ASCEND_HOME_PATH": str(self.root / "other-toolkit")})
        self.assertTrue(self.destination.is_dir())

    def test_supports_ascend_home_when_opp_path_is_not_set(self) -> None:
        self._stage(env={"ASCEND_HOME_PATH": str(self.toolkit)})
        self.assertTrue(self.destination.is_dir())

    def test_accepts_toolkit_aliases_resolving_to_same_installation(self) -> None:
        alias = self.root / "toolkit-latest"
        alias.symlink_to(self.toolkit, target_is_directory=True)
        self._stage(env={**self.env, "NPU_HOME_PATH": str(alias)})
        self.assertTrue(self.destination.is_dir())

    def test_rejects_different_marker_and_linker_roots(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "different operator installations"):
            self._stage(env={**self.env, "NPU_HOME_PATH": str(self.root / "other-toolkit")})

    def test_does_not_search_other_toolkits_when_current_vendor_is_missing(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not installed by this build"):
            self._stage(env={"ASCEND_OPP_PATH": str(self.root / "missing"), "ASCEND_HOME_PATH": str(self.toolkit)})

    def test_requires_explicit_build_toolkit(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ASCEND_OPP_PATH or ASCEND_HOME_PATH is required"):
            self._stage(env={})

    def test_rejects_broken_symlinks_instead_of_dropping_resources(self) -> None:
        (self.vendor / "broken-resource").symlink_to(self.root / "missing-resource")
        with self.assertRaisesRegex(RuntimeError, "Failed to collect the complete NPU operator vendor"):
            self._stage()
        self.assertFalse(self.destination.exists())

    def test_removes_previous_vendor_resources_on_rebuild(self) -> None:
        self._stage()
        (self.destination / "removed-operator.o").write_text("stale", encoding="utf-8")
        self._stage()
        self.assertFalse((self.destination / "removed-operator.o").exists())

    def test_non_npu_build_removes_stale_vendor_without_toolkit_or_source(self) -> None:
        self._stage()
        shutil.rmtree(self.source)
        for device in ("mlu", "cuda"):
            with self.subTest(device=device):
                self.destination.mkdir(parents=True, exist_ok=True)
                (self.destination / "stale.so").touch()
                self._stage(device=device, env={})
                self.assertFalse((self.extdir / "opp").exists())


if __name__ == "__main__":
    unittest.main()
