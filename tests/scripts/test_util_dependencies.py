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

"""Keep util independent of other xLLM source layers."""

import re
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_XLLM_ROOT = _REPO_ROOT / "xllm"
_UTIL_ROOT = _XLLM_ROOT / "core/util"
_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)
_SOURCE_SUFFIXES = {".h", ".hpp", ".c", ".cc", ".cpp", ".cu", ".cuh", ".inl"}


def _project_include(source: Path, include: str) -> Path | None:
    for directory in (source.parent, _REPO_ROOT, _XLLM_ROOT, _XLLM_ROOT / "core"):
        candidate = (directory / include).resolve()
        if candidate.is_relative_to(_XLLM_ROOT) and candidate.is_file():
            return candidate
    return None


class UtilDependenciesTest(unittest.TestCase):
    def test_includes_stay_within_util(self) -> None:
        violations = []
        for source in sorted(_UTIL_ROOT.rglob("*")):
            if source.suffix not in _SOURCE_SUFFIXES:
                continue
            contents = source.read_text()
            for match in _INCLUDE.finditer(contents):
                include = match.group(1)
                target = _project_include(source, include)
                forbidden = (
                    target is not None and not target.is_relative_to(_UTIL_ROOT) and not include.endswith(".pb.h")
                )
                if forbidden:
                    line = contents.count("\n", 0, match.start()) + 1
                    violations.append(f"{source.relative_to(_REPO_ROOT)}:{line}: {include}")
        self.assertEqual(violations, [], "util must not include another xLLM source layer")

    def test_build_dependencies_do_not_point_to_project_layers(self) -> None:
        project_targets = set()
        for cmake in _XLLM_ROOT.rglob("CMakeLists.txt"):
            contents = re.sub(r"#[^\n]*", "", cmake.read_text())
            project_targets.update(re.findall(r"cc_library\s*\(\s*NAME\s+(\w+)", contents))
            project_targets.update(re.findall(r"add_library\s*\(\s*(\w+)", contents))
        project_targets.difference_update({"util", "xllm_proto"})

        cmake = re.sub(r"#[^\n]*", "", (_UTIL_ROOT / "CMakeLists.txt").read_text())
        # Target names cannot occur in filenames; tokenizing also handles the
        # project's :target aliases without interpreting CMake expressions.
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_:.-]*", cmake))
        local_dependencies = project_targets.intersection(token.lstrip(":") for token in tokens)
        self.assertEqual(local_dependencies, set(), "util must not depend on another project layer")


if __name__ == "__main__":
    unittest.main()
