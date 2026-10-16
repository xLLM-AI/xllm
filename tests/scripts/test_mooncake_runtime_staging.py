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

"""Staging of Mooncake runtime libraries into the xLLM wheel."""

import importlib.util
import os
import tempfile
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_setup_module():
    """Import ``setup.py`` as a module; its side effects sit under __main__."""
    spec = importlib.util.spec_from_file_location("xllm_setup", os.path.join(_REPO_ROOT, "setup.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MooncakeRuntimeStagingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.setup = _load_setup_module()

    def test_stages_asio_and_liburing_soname(self) -> None:
        with (
            tempfile.TemporaryDirectory() as cmake_dir,
            tempfile.TemporaryDirectory() as library_dir,
            tempfile.TemporaryDirectory() as extdir,
        ):
            asio_dir = os.path.join(cmake_dir, "mooncake-common")
            os.makedirs(asio_dir)
            with open(os.path.join(asio_dir, "libasio.so"), "w", encoding="utf-8") as library:
                library.write("asio")

            versioned_uring = os.path.join(library_dir, "liburing.so.2.5")
            with open(versioned_uring, "w", encoding="utf-8") as library:
                library.write("uring")
            uring_link = os.path.join(library_dir, "liburing.so")
            os.symlink(os.path.basename(versioned_uring), uring_link)
            with open(os.path.join(cmake_dir, "CMakeCache.txt"), "w", encoding="utf-8") as cache:
                cache.write(f"URING_LIB:FILEPATH={uring_link}\n")

            self.setup._stage_mooncake_runtime_binaries(cmake_dir, extdir)

            with open(os.path.join(extdir, "libasio.so"), encoding="utf-8") as library:
                self.assertEqual(library.read(), "asio")
            with open(os.path.join(extdir, "liburing.so.2"), encoding="utf-8") as library:
                self.assertEqual(library.read(), "uring")

    def test_fails_when_cmake_did_not_resolve_liburing(self) -> None:
        with tempfile.TemporaryDirectory() as cmake_dir, tempfile.TemporaryDirectory() as extdir:
            with open(os.path.join(cmake_dir, "CMakeCache.txt"), "w", encoding="utf-8") as cache:
                cache.write("URING_LIB:FILEPATH=URING_LIB-NOTFOUND\n")

            with self.assertRaisesRegex(RuntimeError, "CMake did not resolve URING_LIB"):
                self.setup._stage_mooncake_runtime_binaries(cmake_dir, extdir)

    def test_fails_when_resolved_liburing_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as cmake_dir, tempfile.TemporaryDirectory() as extdir:
            asio_dir = os.path.join(cmake_dir, "mooncake-common")
            os.makedirs(asio_dir)
            open(os.path.join(asio_dir, "libasio.so"), "w", encoding="utf-8").close()
            missing_library = os.path.join(cmake_dir, "liburing.so.2")
            with open(os.path.join(cmake_dir, "CMakeCache.txt"), "w", encoding="utf-8") as cache:
                cache.write(f"URING_LIB:FILEPATH={missing_library}\n")

            with self.assertRaisesRegex(RuntimeError, missing_library):
                self.setup._stage_mooncake_runtime_binaries(cmake_dir, extdir)

    def test_rejects_incompatible_liburing_soname(self) -> None:
        with (
            tempfile.TemporaryDirectory() as cmake_dir,
            tempfile.TemporaryDirectory() as library_dir,
            tempfile.TemporaryDirectory() as extdir,
        ):
            incompatible_library = os.path.join(library_dir, "liburing.so.1")
            open(incompatible_library, "w", encoding="utf-8").close()
            with open(os.path.join(cmake_dir, "CMakeCache.txt"), "w", encoding="utf-8") as cache:
                cache.write(f"URING_LIB:FILEPATH={incompatible_library}\n")

            with self.assertRaisesRegex(RuntimeError, "incompatible library"):
                self.setup._stage_mooncake_runtime_binaries(cmake_dir, extdir)


if __name__ == "__main__":
    unittest.main()
