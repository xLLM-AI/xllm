# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for CMake integration with persistent Cargo target directories."""

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("cmake_module", "cmake_function", "library_filename"),
    [
        ("cargo_library", "cargo_library", "libdemo.a"),
        ("cargo_shared_library", "cargo_shared_library", "libdemo.so"),
    ],
)
def test_cargo_target_always_checks_cargo_fingerprint(
    tmp_path: Path,
    cmake_module: str,
    cmake_function: str,
    library_filename: str,
) -> None:
    if shutil.which("cmake") is None or shutil.which("ninja") is None:
        pytest.skip("CMake and Ninja are required")

    repo_root = Path(__file__).resolve().parents[2]
    source_dir = tmp_path / "source"
    build_dir = tmp_path / "build"
    cache_dir = tmp_path / "cargo-cache"
    source_dir.mkdir()
    (source_dir / "src").mkdir()
    (source_dir / "src" / "lib.rs").write_text("pub fn value() {}\n", encoding="utf-8")
    (source_dir / "Cargo.toml").write_text("[package]\nname='demo'\n", encoding="utf-8")
    (source_dir / "Cargo.lock").write_text("version = 3\n", encoding="utf-8")

    invocation_count_path = tmp_path / "cargo-invocations"
    cached_artifact_path = tmp_path / "cached-artifact-path"
    fake_cargo_path = tmp_path / "fake-cargo"
    fake_cargo_path.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/sh
            set -eu
            if [ "$1" != "build" ]; then
              echo "expected cargo build, got: $*" >&2
              exit 2
            fi
            target=""
            profile="debug"
            while [ "$#" -gt 0 ]; do
              if [ "$1" = "--target" ]; then
                shift
                target="$1"
              elif [ "$1" = "--release" ]; then
                profile="release"
              fi
              shift
            done
            output="$CARGO_TARGET_DIR/$target/$profile/{library_filename}"
            mkdir -p "$(dirname "$output")"
            : > "$output"
            echo "$output" > "{cached_artifact_path}"
            count=0
            if [ -f "{invocation_count_path}" ]; then
              count=$(cat "{invocation_count_path}")
            fi
            echo $((count + 1)) > "{invocation_count_path}"
            """
        ),
        encoding="utf-8",
    )
    fake_cargo_path.chmod(0o755)

    (source_dir / "CMakeLists.txt").write_text(
        textwrap.dedent(
            f"""\
            cmake_minimum_required(VERSION 3.18)
            project(cargo_cache_test LANGUAGES C)
            list(APPEND CMAKE_MODULE_PATH "{repo_root / 'cmake'}")
            set(CARGO_EXECUTABLE "{fake_cargo_path}")
            include({cmake_module})
            {cmake_function}(NAME demo)
            """
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["XLLM_CARGO_TARGET_ROOT"] = str(cache_dir)
    subprocess.run(
        ["cmake", "-S", str(source_dir), "-B", str(build_dir), "-G", "Ninja"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    for _ in range(2):
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--target", "demo_target"],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )

    assert invocation_count_path.read_text(encoding="utf-8").strip() == "2"
    ninja_file = (build_dir / "build.ninja").read_text(encoding="utf-8")
    assert str(source_dir / "Cargo.toml") in ninja_file
    assert str(source_dir / "Cargo.lock") in ninja_file

    cached_artifact = Path(
        cached_artifact_path.read_text(encoding="utf-8").strip()
    )
    local_artifact = build_dir / "cargo-artifacts" / library_filename
    assert cached_artifact.is_file()
    assert local_artifact.is_file()

    subprocess.run(
        ["ninja", "-C", str(build_dir), "-t", "clean"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    assert cached_artifact.is_file()
    assert not local_artifact.exists()

    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "demo_target"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    assert invocation_count_path.read_text(encoding="utf-8").strip() == "3"
    assert local_artifact.is_file()
