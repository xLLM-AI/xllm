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
# ==============================================================================

"""Verify serving arguments and bundled operators for packaged entry points."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import xllm
from xllm import launch_server
from xllm._npu_runtime import configure_npu_runtime

_CC = shutil.which("cc")
_COMPILER_PATH = os.environ.get("PATH", os.defpath)


class LaunchServerArgumentsTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="xllm serve arguments ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.binary = self.root / "server"
        self.binary.write_text(
            f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n",
            encoding="utf-8",
        )
        self.binary.chmod(0o755)

    def _run_serve(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "xllm.launch_server",
                "serve",
                *args,
                "--binary-path",
                str(self.binary),
                "--no-log-files",
            ],
            cwd=Path(xllm.__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

    def test_model_forms_preserve_server_flag_values(self) -> None:
        model_path = str(self.root / "model with spaces")
        for model_args, expected in (
            ([model_path], [f"--model={model_path}"]),
            ([f"--model={model_path}"], [f"--model={model_path}"]),
            (["--model", model_path], ["--model", model_path]),
        ):
            with self.subTest(model_args=model_args):
                result = self._run_serve(*model_args, "--port", "8015", "--devices", "npu:0", "--enable_graph=false")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    json.loads(result.stdout),
                    [
                        "--port=8015",
                        "--nnodes=1",
                        "--node_rank=0",
                        *expected,
                        "--devices",
                        "npu:0",
                        "--enable_graph=false",
                    ],
                )

    def test_config_only_launch_preserves_server_flag_values(self) -> None:
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps({"model": "/models/from-config", "port": 8020}), encoding="utf-8")
        result = self._run_serve("--config_json_file", str(config_path), "--devices", "npu:0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            [f"--config_json_file={config_path}", "--port=8020", "--nnodes=1", "--node_rank=0", "--devices", "npu:0"],
        )

    def test_positional_model_rejects_explicit_model_before_launch(self) -> None:
        for model_args in (["--model=/models/other"], ["--model", "/models/other"]):
            with self.subTest(model_args=model_args):
                result = self._run_serve("/models/positional", *model_args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("specify the model either as MODEL_PATH or with --model, not both", result.stderr)
                self.assertEqual(result.stdout, "")


class NpuRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="xllm packaged runtime ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        environment = mock.patch.dict(
            os.environ,
            {
                "LD_LIBRARY_PATH": "/cann/lib:/external/lib",
                "ASCEND_CUSTOM_OPP_PATH": "/external/vendor",
                "ASCEND_OPP_PATH": "/cann/opp",
            },
            clear=True,
        )
        environment.start()
        self.addCleanup(environment.stop)

    def _make_binary(self, package_dir: Path, *, bundle: bool = True) -> Path:
        package_dir.mkdir(parents=True, exist_ok=True)
        binary = package_dir / "xllm"
        binary.write_text(
            f"#!{sys.executable}\n"
            "import json, os\n"
            "print(json.dumps({name: os.environ.get(name) for name in "
            "('LD_LIBRARY_PATH', 'ASCEND_CUSTOM_OPP_PATH', 'ASCEND_OPP_PATH')}))\n",
            encoding="utf-8",
        )
        binary.chmod(0o755)
        if bundle:
            vendor = package_dir / "opp" / "vendors" / "custom_xllm_math"
            for directory in ("op_api/lib", "op_impl", "op_proto"):
                (vendor / directory).mkdir(parents=True)
            for directory in ("op_impl", "op_proto"):
                (vendor / directory / "operator-resource").write_bytes(b"operator resource")
            (vendor / "op_api" / "lib" / "libcust_opapi.so").write_bytes(b"operator library")
        return binary

    def _assert_bundle_environment(self, package_dir: Path) -> None:
        vendor = package_dir / "opp" / "vendors" / "custom_xllm_math"
        self.assertEqual(os.environ["LD_LIBRARY_PATH"], f"{vendor / 'op_api' / 'lib'}:/cann/lib:/external/lib")
        self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], f"{vendor}:/external/vendor")
        self.assertEqual(os.environ["ASCEND_OPP_PATH"], "/cann/opp")

    def _run_offline_export(self, package_dir: Path, *, bundle: bool, npu: bool) -> None:
        self._make_binary(package_dir, bundle=bundle)
        source_package = Path(xllm.__file__).parent
        for name in ("__init__.py", "_npu_runtime.py"):
            shutil.copy2(source_package / name, package_dir / name)

        expected_environment = {
            name: os.environ[name] for name in ("LD_LIBRARY_PATH", "ASCEND_CUSTOM_OPP_PATH", "ASCEND_OPP_PATH")
        }
        if bundle:
            vendor = package_dir / "opp" / "vendors" / "custom_xllm_math"
            expected_environment["LD_LIBRARY_PATH"] = (
                f"{vendor / 'op_api' / 'lib'}:{expected_environment['LD_LIBRARY_PATH']}"
            )
            expected_environment["ASCEND_CUSTOM_OPP_PATH"] = (
                f"{vendor}:{expected_environment['ASCEND_CUSTOM_OPP_PATH']}"
            )
        check_environment = textwrap.dedent(
            f"""
            import os
            expected = {expected_environment!r}
            actual = {{name: os.environ.get(name) for name in expected}}
            assert actual == expected, (actual, expected)
            """
        )
        if npu:
            (package_dir.parent / "torch_npu.py").write_text(
                check_environment + "INITIALIZED = True\n", encoding="utf-8"
            )

        # Use a Python module in place of the native extension, preserving the
        # actual lazy loader and torch_npu import without requiring either SDK.
        export = package_dir / "offline_export.py"
        export.write_text(
            check_environment
            + f"import sys\nassert ('torch_npu' in sys.modules) is {npu!r}\n"
            + ("assert sys.modules['torch_npu'].INITIALIZED\n" if npu else "")
            + "LOADED = True\n",
            encoding="utf-8",
        )
        script = textwrap.dedent(
            f"""
            import sys
            from unittest import mock
            import xllm

            def load_operator_library(path: str, mode: int) -> None:
                assert {bundle!r}
                assert ('torch_npu' in sys.modules) is {npu!r}
                assert path == {str(package_dir / "opp/vendors/custom_xllm_math/op_api/lib/libcust_opapi.so")!r}

            assert 'torch_npu' not in sys.modules
            assert 'xllm_export' not in sys.modules
            with (
                mock.patch.object(xllm, '_find_export_so_path', return_value={str(export)!r}),
                mock.patch('xllm._npu_runtime.ctypes.CDLL', side_effect=load_operator_library) as preload,
            ):
                from xllm import xllm_export
            assert preload.call_count == {int(bundle)}
            assert xllm_export.LOADED
            assert sys.modules['xllm_export'] is xllm_export
            assert 'torch' not in sys.modules
            """
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", script],
            cwd=package_dir.parent,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_offline_export_registers_bundle_before_torch_npu_import(self) -> None:
        self._run_offline_export(self.root / "installed" / "xllm", bundle=True, npu=True)

    def test_offline_export_without_bundle_preserves_external_npu_environment(self) -> None:
        self._run_offline_export(self.root / "installed" / "xllm", bundle=False, npu=True)

    def test_offline_export_without_bundle_or_torch_npu_remains_supported(self) -> None:
        self._run_offline_export(self.root / "installed" / "xllm", bundle=False, npu=False)

    def _build_native_offline_fixture(self) -> tuple[Path, Path]:
        package = self.root / "installed" / "xllm"
        self._make_binary(package)
        for name in ("__init__.py", "_npu_runtime.py"):
            shutil.copy2(Path(xllm.__file__).parent / name, package / name)
        (package.parent / "torch_npu.py").write_text("INITIALIZED = True\n", encoding="utf-8")
        bundled = package / "opp/vendors/custom_xllm_math/op_api/lib"
        external = self.root / "external-vendor"
        external.mkdir()

        def compile_library(source: str, output: Path, *options: str) -> None:
            result = subprocess.run(
                [_CC, "-shared", "-fPIC", "-x", "c", "-", "-o", str(output), *options],
                input=source,
                env=dict(os.environ, PATH=_COMPILER_PATH),
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        for directory, revision in ((bundled, 7), (external, 99)):
            compile_library(
                f"int operator_revision(void) {{ return {revision}; }}\n",
                directory / "libcust_opapi.so",
                "-Wl,-soname,libcust_opapi.so",
            )
        compile_library(
            "#include <Python.h>\n"
            "extern int operator_revision(void);\n"
            "PyMODINIT_FUNC PyInit_xllm_export(void) {\n"
            '  static struct PyModuleDef definition = {PyModuleDef_HEAD_INIT, "xllm_export", NULL, -1, NULL};\n'
            "  PyObject *module = PyModule_Create(&definition);\n"
            '  if (module) PyModule_AddIntConstant(module, "OPERATOR_REVISION", operator_revision());\n'
            "  return module;\n"
            "}\n",
            package / f"xllm_export{sysconfig.get_config_var('EXT_SUFFIX')}",
            f"-I{sysconfig.get_paths()['include']}",
            f"-L{bundled}",
            "-lcust_opapi",
            "-Wl,--enable-new-dtags,-rpath,$ORIGIN/opp/vendors/custom_xllm_math/op_api/lib",
        )
        return package, external

    def _run_native_offline_fixture(self, package: Path, external: Path, script: str) -> None:
        environment = dict(os.environ, LD_LIBRARY_PATH=str(external))
        result = subprocess.run(
            [sys.executable, "-S", "-c", script],
            cwd=package.parent,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(sys.platform.startswith("linux") and _CC, "ELF loader and a C compiler are required")
    def test_offline_extension_prefers_bundle_over_startup_library_search_path(self) -> None:
        package, external = self._build_native_offline_fixture()
        # Prove that startup LD_LIBRARY_PATH would win over the extension's
        # RUNPATH without the package's in-process library initialization.
        self._run_native_offline_fixture(
            package,
            external,
            "import importlib.util, xllm\n"
            "spec = importlib.util.spec_from_file_location('xllm_export', xllm._find_export_so_path())\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "assert module.OPERATOR_REVISION == 99\n",
        )
        self._run_native_offline_fixture(
            package,
            external,
            "from pathlib import Path\n"
            "from xllm import xllm_export\n"
            "assert xllm_export.OPERATOR_REVISION == 7\n"
            "maps = Path('/proc/self/maps').read_text()\n"
            f"assert {str(external / 'libcust_opapi.so')!r} not in maps\n"
            f"assert {str(package / 'opp/vendors/custom_xllm_math/op_api/lib/libcust_opapi.so')!r} in maps\n",
        )

    @unittest.skipUnless(sys.platform.startswith("linux") and _CC, "ELF loader and a C compiler are required")
    def test_offline_extension_without_bundle_keeps_external_operator_library(self) -> None:
        package, external = self._build_native_offline_fixture()
        shutil.rmtree(package / "opp")
        self._run_native_offline_fixture(
            package,
            external,
            "from xllm import xllm_export\nassert xllm_export.OPERATOR_REVISION == 99\n",
        )

    @unittest.skipUnless(sys.platform.startswith("linux") and _CC, "ELF loader and a C compiler are required")
    def test_unloadable_bundle_fails_without_using_external_operator_library(self) -> None:
        package, external = self._build_native_offline_fixture()
        library = package / "opp/vendors/custom_xllm_math/op_api/lib/libcust_opapi.so"
        library.write_bytes(b"invalid ELF library")
        self._run_native_offline_fixture(
            package,
            external,
            "import sys\n"
            "try:\n"
            "    from xllm import xllm_export\n"
            "except OSError as error:\n"
            f"    assert {str(library)!r} in str(error)\n"
            "else:\n"
            "    raise AssertionError('invalid bundle was ignored')\n"
            "assert 'xllm_export' not in sys.modules\n",
        )

    def test_binary_without_bundle_keeps_existing_environment(self) -> None:
        binary = self._make_binary(self.root / "source", bundle=False)
        original = os.environ.copy()

        configure_npu_runtime(str(binary))

        self.assertEqual(dict(os.environ), original)

    def test_bundled_vendor_precedes_external_paths_and_repeated_calls_are_idempotent(self) -> None:
        binary = self._make_binary(self.root / "installed")

        configure_npu_runtime(str(binary))
        configure_npu_runtime(str(binary))

        self._assert_bundle_environment(binary.parent)

    def test_existing_duplicates_are_removed_without_dropping_empty_search_entries(self) -> None:
        binary = self._make_binary(self.root / "installed")
        vendor = binary.parent / "opp" / "vendors" / "custom_xllm_math"
        library_dir = vendor / "op_api" / "lib"
        os.environ["LD_LIBRARY_PATH"] = f"/cann/lib::{library_dir}:/cann/lib"
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = f"/external/vendor:{vendor}:/external/vendor"

        configure_npu_runtime(str(binary))

        self.assertEqual(os.environ["LD_LIBRARY_PATH"], f"{library_dir}:/cann/lib:")
        self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], f"{vendor}:/external/vendor")

    def test_missing_environment_variables_are_initialized_without_empty_path_entries(self) -> None:
        binary = self._make_binary(self.root / "installed")
        os.environ.pop("LD_LIBRARY_PATH")
        os.environ.pop("ASCEND_CUSTOM_OPP_PATH")

        configure_npu_runtime(str(binary))

        vendor = binary.parent / "opp" / "vendors" / "custom_xllm_math"
        self.assertEqual(os.environ["LD_LIBRARY_PATH"], str(vendor / "op_api" / "lib"))
        self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], str(vendor))

    def test_corrupt_bundle_fails_before_changing_environment(self) -> None:
        for missing in ("op_api/lib/libcust_opapi.so", "op_impl", "op_proto"):
            with self.subTest(missing=missing):
                binary = self._make_binary(self.root / missing.replace("/", "_"))
                path = binary.parent / "opp" / "vendors" / "custom_xllm_math" / missing
                if path.is_file():
                    path.unlink()
                else:
                    (path / "operator-resource").unlink()
                    path.rmdir()
                original = os.environ.copy()

                with self.assertRaisesRegex(RuntimeError, "Incomplete bundled xllm_ops") as raised:
                    configure_npu_runtime(str(binary))

                self.assertIn(str(path), str(raised.exception))
                self.assertEqual(dict(os.environ), original)

    def test_empty_operator_resource_directory_is_reported_as_incomplete(self) -> None:
        binary = self._make_binary(self.root / "installed")
        resource = binary.parent / "opp" / "vendors" / "custom_xllm_math" / "op_impl" / "operator-resource"
        resource.unlink()

        with self.assertRaisesRegex(RuntimeError, "missing or empty"):
            configure_npu_runtime(str(binary))

    def test_broken_bundle_symlink_is_reported_as_incomplete(self) -> None:
        binary = self._make_binary(self.root / "installed", bundle=False)
        (binary.parent / "opp").symlink_to(self.root / "missing-operators", target_is_directory=True)

        with self.assertRaisesRegex(RuntimeError, "Incomplete bundled xllm_ops"):
            configure_npu_runtime(str(binary))

    def test_binary_symlink_uses_operators_belonging_to_its_actual_binary(self) -> None:
        binary = self._make_binary(self.root / "installed")
        alias = self.root / "xllm-alias"
        alias.symlink_to(binary)

        configure_npu_runtime(str(alias))

        self._assert_bundle_environment(binary.parent)

    def test_start_process_passes_operator_paths_to_the_child(self) -> None:
        binary = self._make_binary(self.root / "installed")
        log_dir = self.root / "logs"

        server = launch_server._start_process([str(binary)], rank=1, log_dir=str(log_dir))
        try:
            self.assertEqual(server.process.wait(timeout=10), 0)
        finally:
            launch_server._close_logs([server])

        child_environment = json.loads((log_dir / "node_1.log").read_text(encoding="utf-8"))
        self._assert_bundle_environment(binary.parent)
        self.assertEqual(child_environment, dict(os.environ))

    def test_source_launcher_uses_installed_binary_bundle(self) -> None:
        install_root = self.root / "site-packages"
        binary = self._make_binary(install_root / "xllm")
        with (
            mock.patch.object(launch_server, "_package_binary_path", return_value=str(self.root / "missing")),
            mock.patch.object(sys, "path", [str(install_root)]),
        ):
            selected = launch_server._resolve_binary_path(None)

        configure_npu_runtime(selected)

        self.assertEqual(selected, str(binary))
        self._assert_bundle_environment(binary.parent)

    def test_external_binary_override_does_not_use_launchers_bundle(self) -> None:
        installed = self._make_binary(self.root / "installed")
        external = self._make_binary(self.root / "external", bundle=False)
        original = os.environ.copy()
        with mock.patch.object(launch_server, "__file__", str(installed.parent / "launch_server.py")):
            selected = launch_server._resolve_binary_path(str(external))
            configure_npu_runtime(selected)

        self.assertEqual(dict(os.environ), original)

    def test_server_help_prepares_the_bundle_before_running_the_binary(self) -> None:
        binary = self._make_binary(self.root / "installed")

        def check_environment(*args: object, **kwargs: object) -> None:
            self._assert_bundle_environment(binary.parent)

        with mock.patch.object(launch_server.subprocess, "run", side_effect=check_environment) as run:
            launch_server._print_binary_help(argparse.ArgumentParser(), str(binary))

        run.assert_called_once_with([str(binary), "--help"], check=False)

    def test_direct_binary_invocation_prepares_the_bundle_before_exec(self) -> None:
        binary = self._make_binary(self.root / "installed")

        def check_environment(*args: object) -> None:
            self._assert_bundle_environment(binary.parent)
            raise SystemExit(0)

        with (
            mock.patch.object(launch_server, "_resolve_binary_path", return_value=str(binary)),
            mock.patch.object(launch_server.os, "execv", side_effect=check_environment) as execv,
            self.assertRaises(SystemExit) as raised,
        ):
            launch_server._exec_binary(["--help"])

        self.assertEqual(raised.exception.code, 0)
        execv.assert_called_once_with(str(binary), [str(binary), "--help"])


if __name__ == "__main__":
    unittest.main()
