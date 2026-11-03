# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CMAKE = os.environ.get("CMAKE_EXECUTABLE") or shutil.which("cmake")


@unittest.skipUnless(CMAKE, "CMake executable is required for allocator regression tests")
class AllocatorCmakeTest(unittest.TestCase):
    def _configure(self, enabled: bool, avoid: bool, dependencies: str, expected: str) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "CMakeLists.txt").write_text(f"""cmake_minimum_required(VERSION 3.26)
project(AllocatorPolicy NONE)
set(USE_NPU ON)
set(USE_YUANRONG_DATASYSTEM {'ON' if enabled else 'OFF'})
set(DATASYSTEM_AVOID_TCMALLOC {'ON' if avoid else 'OFF'})
add_library(leveldb::leveldb INTERFACE IMPORTED)
set_property(TARGET leveldb::leveldb PROPERTY INTERFACE_LINK_LIBRARIES "{dependencies}")
include("{(ROOT / 'cmake/datasystem_allocator.cmake').as_posix()}")
xllm_configure_datasystem_allocator()
xllm_configure_datasystem_allocator()
get_target_property(actual leveldb::leveldb INTERFACE_LINK_LIBRARIES)
if(NOT actual STREQUAL "{expected}")
  message(FATAL_ERROR "Unexpected dependencies: ${{actual}}")
endif()
""", encoding="utf-8")
            result = subprocess.run(
                [CMAKE, "-S", str(directory), "-B", str(directory / "build"), "-G", "Ninja"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_data_system_removes_only_optional_allocator(self) -> None:
        self._configure(True, True, "tcmalloc;Threads::Threads;OtherDependency", "Threads::Threads;OtherDependency")

    def test_mooncake_only_build_keeps_upstream_links(self) -> None:
        self._configure(False, True, "tcmalloc;Threads::Threads", "tcmalloc;Threads::Threads")

    def test_explicit_opt_out_keeps_allocator(self) -> None:
        self._configure(True, False, "tcmalloc;Threads::Threads", "tcmalloc;Threads::Threads")

    def test_allocator_free_package_is_unchanged(self) -> None:
        self._configure(True, True, "Threads::Threads", "Threads::Threads")


if __name__ == "__main__":
    unittest.main()
