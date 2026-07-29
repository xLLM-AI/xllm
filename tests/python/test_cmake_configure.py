# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from scripts.build_support.cmake_configure import (
    CMAKE_CONFIGURE_FINGERPRINT,
    cmake_configure_required,
    get_cmake_configure_fingerprint,
    record_cmake_configure_fingerprint,
)


def test_fingerprint_tracks_command_and_cmake_environment() -> None:
    command = ["cmake", "source", "-DUSE_NPU=ON"]
    original = get_cmake_configure_fingerprint(
        command, {"NPU_HOME_PATH": "cann-v1", "UNRELATED": "one"}
    )

    assert original != get_cmake_configure_fingerprint(
        [*command, "-DUSE_CXX11_ABI=ON"],
        {"NPU_HOME_PATH": "cann-v1", "UNRELATED": "one"},
    )
    assert original != get_cmake_configure_fingerprint(
        command, {"NPU_HOME_PATH": "cann-v2", "UNRELATED": "one"}
    )
    assert original == get_cmake_configure_fingerprint(
        command, {"NPU_HOME_PATH": "cann-v1", "UNRELATED": "two"}
    )


def test_configure_required_checks_build_tree_and_fingerprint(
    tmp_path: Path,
) -> None:
    fingerprint = "expected"
    assert cmake_configure_required(str(tmp_path), fingerprint)

    (tmp_path / "CMakeCache.txt").touch()
    (tmp_path / "build.ninja").touch()
    assert cmake_configure_required(str(tmp_path), fingerprint)

    record_cmake_configure_fingerprint(str(tmp_path), fingerprint)
    assert not cmake_configure_required(str(tmp_path), fingerprint)
    assert cmake_configure_required(str(tmp_path), "changed")
    assert (
        tmp_path / CMAKE_CONFIGURE_FINGERPRINT
    ).read_text(encoding="utf-8") == f"{fingerprint}\n"

    (tmp_path / CMAKE_CONFIGURE_FINGERPRINT).write_bytes(b"\xff")
    assert cmake_configure_required(str(tmp_path), fingerprint)
