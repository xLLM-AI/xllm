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

import os
import tempfile
import unittest
from unittest import mock

from scripts.build_support import utils


class BuildDependenciesTest(unittest.TestCase):
    def test_ubuntu_paths_are_supported(self) -> None:
        with mock.patch.object(utils.sysconfig, "get_config_var", return_value="x86_64-linux-gnu"):
            dependencies = utils._get_required_dependency_files()

        self.assertIn("/usr/include/msgpack.hpp", dependencies["msgpack-cxx"])
        self.assertIn("/usr/include/xxhash.h", dependencies["xxhash-header"])
        self.assertIn(
            "/usr/lib/x86_64-linux-gnu/libxxhash.so",
            dependencies["xxhash-library"],
        )
        self.assertIn("/usr/include/zstd.h", dependencies["zstd-header"])
        self.assertIn(
            "/usr/lib/x86_64-linux-gnu/libzstd.so",
            dependencies["zstd-library"],
        )

    def test_dependency_requires_header_and_library(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            header = os.path.join(temp_dir, "include", "xxhash.h")
            library = os.path.join(temp_dir, "lib", "libxxhash.so")
            os.makedirs(os.path.dirname(header))
            os.makedirs(os.path.dirname(library))
            open(header, "w", encoding="utf-8").close()

            dependencies = {
                "xxhash-header": [header],
                "xxhash-library": [library],
            }
            missing = utils._collect_missing_dependencies(dependencies)

        self.assertNotIn("xxhash-header", missing)
        self.assertIn("xxhash-library", missing)

    def test_yalantinglibs_uses_mooncake_install_prefix(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("YALANTINGLIBS_PREFIX", None)
            dependencies = utils._get_required_dependency_files()

        self.assertEqual(
            dependencies["yalantinglibs"],
            ["/usr/local/lib/cmake/yalantinglibs/config.cmake"],
        )

    def test_missing_dependencies_run_mooncake_dependencies(self) -> None:
        with mock.patch.object(utils, "_run_shell_command", return_value=True) as run:
            utils._run_dependencies_script_or_exit("/repo")

        run.assert_called_once_with(
            "bash scripts/build_support/install_dependencies.sh",
            cwd="/repo",
            passthrough_output=True,
        )

    def test_prebuild_skips_installed_mooncake_dependencies(self) -> None:
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(utils, "_run_command", return_value=(True, utils._GO_DEFAULT_PROXY, "")),
            mock.patch.object(utils, "_run_shell_command") as run,
            mock.patch.object(utils, "_get_required_dependency_files", return_value={}),
            mock.patch.object(utils, "_export_mooncake_go_path", return_value=True),
            mock.patch.object(utils, "_export_cmake_prefix_paths"),
        ):
            utils._ensure_prebuild_dependencies_installed(
                "/repo",
            )
            self.assertNotIn("GOPROXY", os.environ)

        run.assert_not_called()

    def test_prebuild_reports_missing_go_toolchain_without_installing(self) -> None:
        with (
            mock.patch.object(utils, "_run_shell_command") as run,
            mock.patch.object(utils, "_get_mooncake_go_version", return_value="1.25.10"),
            mock.patch.object(utils, "_export_mooncake_go_path", return_value=False),
            self.assertRaises(SystemExit),
        ):
            utils.ensure_mooncake_go_toolchain("/repo")
        run.assert_not_called()

    def test_prebuild_exports_newly_installed_go_toolchain(self) -> None:
        with (
            mock.patch.object(utils, "_run_shell_command", return_value=True) as run,
            mock.patch.object(utils, "_export_mooncake_go_path", side_effect=[False, True]) as export_go_path,
            mock.patch.object(utils, "_export_mooncake_go_proxy") as export_go_proxy,
        ):
            utils.ensure_mooncake_go_toolchain("/repo", install=True)

        self.assertEqual(export_go_path.call_count, 2)
        export_go_proxy.assert_called_once_with()
        run.assert_called_once_with(
            "bash scripts/build_support/install_dependencies.sh --ensure-go", cwd="/repo", passthrough_output=True
        )

    def test_export_mooncake_go_path_prepends_install_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            go_binary = os.path.join(temp_dir, "go")
            open(go_binary, "w", encoding="utf-8").close()
            os.chmod(go_binary, 0o755)

            with (
                mock.patch.object(utils, "_MOONCAKE_GO_BIN_DIR", temp_dir),
                mock.patch.object(utils, "_get_mooncake_go_version", return_value="1.25.9"),
                mock.patch.object(utils, "_get_go_version", side_effect=[None, "1.25.9"]),
                mock.patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}),
            ):
                utils._export_mooncake_go_path("/repo/third_party/Mooncake")

                self.assertEqual(os.environ["PATH"].split(os.pathsep)[0], temp_dir)

    def test_export_mooncake_go_path_preserves_matching_path_toolchain(self) -> None:
        original_path = "/opt/go1.25.9/bin:/usr/bin:/bin"
        with (
            mock.patch.object(utils, "_get_mooncake_go_version", return_value="1.25.9"),
            mock.patch.object(utils, "_get_go_version", return_value="1.25.9") as get_go_version,
            mock.patch.dict(os.environ, {"PATH": original_path}),
        ):
            utils._export_mooncake_go_path("/repo/third_party/Mooncake")

            self.assertEqual(os.environ["PATH"], original_path)
            get_go_version.assert_called_once_with("go")

    def test_export_mooncake_go_path_rejects_stale_install(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            go_binary = os.path.join(temp_dir, "go")
            open(go_binary, "w", encoding="utf-8").close()
            os.chmod(go_binary, 0o755)
            original_path = "/opt/go1.24.0/bin:/usr/bin:/bin"

            with (
                mock.patch.object(utils, "_MOONCAKE_GO_BIN_DIR", temp_dir),
                mock.patch.object(utils, "_get_mooncake_go_version", return_value="1.25.9"),
                mock.patch.object(utils, "_get_go_version", side_effect=["1.24.0", "1.23.0"]),
                mock.patch.dict(os.environ, {"PATH": original_path}),
            ):
                utils._export_mooncake_go_path("/repo/third_party/Mooncake")

                self.assertEqual(os.environ["PATH"], original_path)

    def test_export_mooncake_go_proxy_replaces_default_proxy(self) -> None:
        for environment in ({}, {"GOPROXY": ""}):
            with (
                self.subTest(environment=environment),
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(
                    utils, "_run_command", return_value=(True, "https://proxy.golang.org,direct", "")
                ) as run,
            ):
                utils._export_mooncake_go_proxy()

                self.assertEqual(os.environ["GOPROXY"], "https://goproxy.cn|https://goproxy.io|direct")
                run.assert_called_once_with(["go", "env", "GOPROXY"], check=False)

    def test_export_mooncake_go_proxy_preserves_environment(self) -> None:
        for proxy in ("https://proxy.example.com", "https://proxy.golang.org,direct", "direct", "off"):
            with (
                self.subTest(proxy=proxy),
                mock.patch.dict(os.environ, {"GOPROXY": proxy}, clear=True),
                mock.patch.object(utils, "_run_command") as run,
            ):
                utils._export_mooncake_go_proxy()

                self.assertEqual(os.environ["GOPROXY"], proxy)
                run.assert_not_called()

    def test_export_mooncake_go_proxy_preserves_saved_configuration(self) -> None:
        for proxy in ("https://proxy.example.com", "direct", "off"):
            with (
                self.subTest(proxy=proxy),
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(utils, "_run_command", return_value=(True, proxy, "")),
            ):
                utils._export_mooncake_go_proxy()

                self.assertNotIn("GOPROXY", os.environ)

    def test_export_mooncake_go_proxy_reports_configuration_errors(self) -> None:
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(utils, "_run_command", return_value=(False, "", "go env failed")),
            self.assertRaises(SystemExit) as error,
        ):
            utils._export_mooncake_go_proxy()

        self.assertEqual(error.exception.code, 1)

    def test_prebuild_force_installs_mooncake_dependencies(self) -> None:
        with (
            mock.patch.object(utils, "_run_shell_command", return_value=True) as run,
            mock.patch.object(utils, "_get_required_dependency_files", return_value={}),
            mock.patch.object(utils, "_export_mooncake_go_proxy"),
            mock.patch.object(utils, "_export_mooncake_go_path", return_value=True),
            mock.patch.object(utils, "_export_cmake_prefix_paths"),
        ):
            utils._ensure_prebuild_dependencies_installed(
                "/repo",
                force_install=True,
            )

        run.assert_called_once_with(
            "bash scripts/build_support/install_dependencies.sh",
            cwd="/repo",
            passthrough_output=True,
        )

    def test_mooncake_safe_directory_is_added_once(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            utils._ensure_git_safe_directory_or_exit("/repo/third_party/Mooncake")
            utils._ensure_git_safe_directory_or_exit("/repo/third_party/Mooncake")
            self.assertEqual(os.environ["GIT_CONFIG_COUNT"], "1")
            self.assertEqual(os.environ["GIT_CONFIG_KEY_0"], "safe.directory")
            self.assertEqual(os.environ["GIT_CONFIG_VALUE_0"], "/repo/third_party/Mooncake")

    def test_existing_mooncake_safe_directory_is_not_added_again(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.quotePath", "GIT_CONFIG_VALUE_0": "false"},
            clear=True,
        ):
            utils._ensure_git_safe_directory_or_exit("/repo/third_party/Mooncake")
            utils._ensure_git_safe_directory_or_exit("/repo/third_party/Mooncake")
            self.assertEqual(os.environ["GIT_CONFIG_COUNT"], "2")
            self.assertEqual(os.environ["GIT_CONFIG_KEY_0"], "core.quotePath")
            self.assertEqual(os.environ["GIT_CONFIG_VALUE_0"], "false")
            self.assertEqual(os.environ["GIT_CONFIG_KEY_1"], "safe.directory")
            self.assertEqual(os.environ["GIT_CONFIG_VALUE_1"], "/repo/third_party/Mooncake")


if __name__ == "__main__":
    unittest.main()
