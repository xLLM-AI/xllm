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

"""Verify shared runtime registration with a lightweight native initializer."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CMAKE = shutil.which("cmake")
_NINJA = shutil.which("ninja")
_CXX = shutil.which("c++")

_PROJECT = """
cmake_minimum_required(VERSION 3.26)
project(NpuRuntimeLinkTest LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_TESTING "Build tests" ON)
set(USE_NPU ON)
set(COMMON_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}")
add_custom_target(all_tests)
foreach(dep GTest::gtest ascendcl Python::Python pybind11::embed torch_npu torch_python)
  add_library(${dep} INTERFACE IMPORTED)
endforeach()
include(GoogleTest)
include("${XLLM_CC_TEST_MODULE}")
add_subdirectory(first)
add_subdirectory(second)
"""
_RUNTIME = """
#include <cstdlib>
namespace {
struct Registration {
  Registration() { setenv("XLLM_RUNTIME_REGISTERED", "1", 1); }
};
Registration registration;
}
"""


@unittest.skipUnless(_CMAKE and _NINJA and _CXX, "CMake, Ninja, and a C++ compiler are required")
class NpuTestRuntimeTest(unittest.TestCase):
    def test_runtime_compiles_once_and_registers_in_each_npu_executable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xllm runtime ") as temporary:
            root = Path(temporary)
            build = root / "build"
            files = {
                "CMakeLists.txt": _PROJECT,
                "tests/npu_test_environment.cpp": _RUNTIME,
                "first/CMakeLists.txt": "cc_test(NAME first_test SRCS main.cpp)\n",
                "second/CMakeLists.txt": (
                    "cc_test(NAME second_test SRCS main.cpp)\ncc_test(NAME host_test NO_NPU_RUNTIME SRCS host.cpp)\n"
                ),
                "first/main.cpp": (
                    '#include <cstdlib>\nint main() { return std::getenv("XLLM_RUNTIME_REGISTERED") == nullptr; }\n'
                ),
                "second/main.cpp": (
                    '#include <cstdlib>\nint main() { return std::getenv("XLLM_RUNTIME_REGISTERED") == nullptr; }\n'
                ),
                "second/host.cpp": (
                    '#include <cstdlib>\nint main() { return std::getenv("XLLM_RUNTIME_REGISTERED") != nullptr; }\n'
                ),
            }
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            env = os.environ.copy()
            env.pop("XLLM_RUNTIME_REGISTERED", None)

            def run(*args: str) -> None:
                result = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            for testing in ("OFF", "ON", "OFF"):
                run(
                    _CMAKE,
                    "-S",
                    str(root),
                    "-B",
                    str(build),
                    "-G",
                    "Ninja",
                    f"-DCMAKE_CXX_COMPILER={_CXX}",
                    f"-DXLLM_CC_TEST_MODULE={_REPO_ROOT / 'cmake' / 'cc_test.cmake'}",
                    f"-DBUILD_TESTING={testing}",
                )
                run(_CMAKE, "--build", str(build), "--target", "all_tests")
                graph = (build / "build.ninja").read_text(encoding="utf-8")
                if testing == "OFF":
                    self.assertNotIn("npu_test_environment.cpp.o", graph)
                    self.assertNotIn("build first_test:", graph)
                    continue
                commands = json.loads((build / "compile_commands.json").read_text(encoding="utf-8"))
                runtime_commands = [c for c in commands if c["file"].endswith("/npu_test_environment.cpp")]
                self.assertEqual(len(runtime_commands), 1)
                for target in ("first/first_test", "second/second_test", "second/host_test"):
                    run(str(build / target))


if __name__ == "__main__":
    unittest.main()
