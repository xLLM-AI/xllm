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

"""Staging of Python kernel packages and their compiler entry points."""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import ModuleType

from scripts.build_support.utils import load_setup_module as _load_setup_module

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
                self.setup._stage_python_kernel_package(source_root, dest_root, device)
                return sorted(os.listdir(dest_root))

    def test_only_the_devices_own_package_is_staged(self) -> None:
        self.assertEqual(self._stage("cuda", peers=("cuda", "npu")), ["kernels_cuda"])

    def test_device_without_a_package_stages_none(self) -> None:
        # xLLM builds for more devices than the Python model executor covers,
        # so a device with no peer package must not fail the build: the wheel
        # ships without kernels and xllm/python/__init__.py rejects the
        # platform at import.
        self.assertEqual(self._stage("mlu", peers=("cuda", "npu")), [])

    def test_triton_compiler_imports_from_staged_package(self) -> None:
        with tempfile.TemporaryDirectory() as install_root:
            package_dir = os.path.join(install_root, "xllm")
            os.makedirs(package_dir)
            shutil.copy2(os.path.join(_REPO_ROOT, "xllm", "__init__.py"), package_dir)
            legacy_dir = os.path.join(package_dir, "core", "triton_jit", "scripts")
            os.makedirs(legacy_dir)
            with open(os.path.join(legacy_dir, "triton_compile.py"), "w", encoding="utf-8") as legacy_file:
                legacy_file.write("raise AssertionError('obsolete compiler entry point')\n")

            self.setup._stage_triton_compiler_package(_REPO_ROOT, package_dir)

            self.assertFalse(os.path.exists(os.path.join(package_dir, "core", "triton_jit")))
            compiler_dir = os.path.join(package_dir, "compiler")
            compiler_files = sorted(
                os.path.relpath(os.path.join(root, filename), compiler_dir)
                for root, _, filenames in os.walk(compiler_dir)
                for filename in filenames
            )
            self.assertEqual(compiler_files, ["__init__.py", "triton/__init__.py", "triton/compile.py"])

            # Run outside the checkout against the installed package layout.
            # Triton itself is optional on the build host, so stub only its
            # import-time version; this checks packaging rather than hardware.
            script = textwrap.dedent(
                """
                import importlib
                import os
                import sys
                from types import ModuleType

                import xllm.compiler.triton
                assert "triton" not in sys.modules
                assert "xllm_export" not in sys.modules

                triton = ModuleType("triton")
                triton.__version__ = "3.3.0"
                sys.modules["triton"] = triton
                compiler = importlib.import_module("xllm.compiler.triton.compile")
                assert compiler.__file__ == os.path.join(
                    os.getcwd(), "xllm", "compiler", "triton", "compile.py"
                )
                assert callable(compiler.compile)
                assert callable(compiler.signature)
                """
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = install_root
            subprocess.run([sys.executable, "-c", script], cwd=install_root, env=env, check=True)

    def test_missing_triton_compiler_fails_before_staging(self) -> None:
        with tempfile.TemporaryDirectory() as source_root, tempfile.TemporaryDirectory() as dest_root:
            compiler_dir = os.path.join(source_root, "xllm", "compiler")
            triton_dir = os.path.join(compiler_dir, "triton")
            os.makedirs(triton_dir)
            for package_dir in (compiler_dir, triton_dir):
                open(os.path.join(package_dir, "__init__.py"), "w", encoding="utf-8").close()

            with self.assertRaisesRegex(RuntimeError, "Triton compiler package file does not exist:.*compile.py"):
                self.setup._stage_triton_compiler_package(source_root, dest_root)

            self.assertEqual(os.listdir(dest_root), [])


class CompilerLauncherTest(unittest.TestCase):
    def test_tilelang_launcher_imports_external_triton(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_compiler = Path(_REPO_ROOT) / "xllm" / "compiler"
            compiler_dir = root / "xllm" / "compiler"
            cli_dir = compiler_dir / "tilelang" / "cli"
            cli_dir.mkdir(parents=True)
            shutil.copy2(source_compiler / "tilelang_launcher.py", compiler_dir)
            shutil.copy2(source_compiler / "__init__.py", compiler_dir)
            shutil.copytree(
                source_compiler / "triton",
                compiler_dir / "triton",
                ignore=shutil.ignore_patterns("include", "src", "CMakeLists.txt", "__pycache__", "*.pyc"),
            )
            (cli_dir / "compile_kernels.py").write_text(
                textwrap.dedent(
                    """
                    import sys
                    import triton.language as language

                    def main(argv: list[str] | None = None) -> None:
                        assert language.SOURCE == "external Triton"
                        assert "xllm_export" not in sys.modules
                        assert argv == ["--target", "ascend"]
                    """
                ),
                encoding="utf-8",
            )
            external_dir = root / "site-packages"
            external_triton = external_dir / "triton"
            external_triton.mkdir(parents=True)
            (external_triton / "__init__.py").touch()
            (external_triton / "language.py").write_text('SOURCE = "external Triton"\n', encoding="utf-8")
            compiler_alias = root / "compiler-alias"
            compiler_alias.symlink_to(compiler_dir, target_is_directory=True)

            for python_paths in ((external_dir,), (compiler_dir, compiler_alias, external_dir)):
                with self.subTest(python_paths=python_paths):
                    env = os.environ.copy()
                    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_paths)
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(compiler_dir / "tilelang_launcher.py"),
                            "compile-kernels",
                            "--target",
                            "ascend",
                        ],
                        cwd=root,
                        env=env,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
