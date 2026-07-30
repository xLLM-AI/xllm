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

"""Tests for build-time parallelism and persistent cache configuration."""

from pathlib import Path
from typing import Optional

import pytest

from scripts.build_support import utils


def test_default_build_jobs_uses_cores_in_one_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cores_per_socket = 80
    socket_count = 4
    cpu_ids = set(range(cores_per_socket * socket_count))

    def _read_topology(cpu_id: int, topology_field: str) -> Optional[int]:
        if topology_field == "physical_package_id":
            return cpu_id // cores_per_socket
        return cpu_id % cores_per_socket

    monkeypatch.setattr(utils, "_get_available_cpu_ids", lambda: cpu_ids)
    monkeypatch.setattr(utils, "_read_cpu_topology_id", _read_topology)

    assert utils.get_default_build_jobs() == cores_per_socket


def test_default_build_jobs_ignores_smt_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logical_cpus_per_socket = 8
    socket_count = 2
    threads_per_core = 2
    cpu_ids = set(range(logical_cpus_per_socket * socket_count))

    def _read_topology(cpu_id: int, topology_field: str) -> Optional[int]:
        if topology_field == "physical_package_id":
            return cpu_id // logical_cpus_per_socket
        return (cpu_id % logical_cpus_per_socket) // threads_per_core

    monkeypatch.setattr(utils, "_get_available_cpu_ids", lambda: cpu_ids)
    monkeypatch.setattr(utils, "_read_cpu_topology_id", _read_topology)

    assert utils.get_default_build_jobs() == (
        logical_cpus_per_socket // threads_per_core
    )


def test_default_build_jobs_respects_available_cpu_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cores_per_socket = 80
    available_cores_in_first_socket = 8
    available_cores_in_second_socket = 4
    cpu_ids = set(range(available_cores_in_first_socket)) | set(
        range(
            cores_per_socket,
            cores_per_socket + available_cores_in_second_socket,
        )
    )

    def _read_topology(cpu_id: int, topology_field: str) -> Optional[int]:
        if topology_field == "physical_package_id":
            return cpu_id // cores_per_socket
        return cpu_id % cores_per_socket

    monkeypatch.setattr(utils, "_get_available_cpu_ids", lambda: cpu_ids)
    monkeypatch.setattr(utils, "_read_cpu_topology_id", _read_topology)

    assert utils.get_default_build_jobs() == available_cores_in_first_socket


def test_default_build_jobs_falls_back_when_topology_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available_cpu_count = 6
    cpu_ids = set(range(available_cpu_count))

    monkeypatch.setattr(utils, "_get_available_cpu_ids", lambda: cpu_ids)
    monkeypatch.setattr(
        utils,
        "_read_cpu_topology_id",
        lambda _cpu_id, _topology_field: None,
    )

    assert utils.get_default_build_jobs() == available_cpu_count


@pytest.mark.parametrize(
    ("max_jobs", "expected_archive_jobs"),
    [
        (1, 1),
        (4, 1),
        (16, 4),
        (32, 8),
        (128, 8),
    ],
)
def test_archive_build_jobs_are_bounded(
    max_jobs: int,
    expected_archive_jobs: int,
) -> None:
    assert utils.get_archive_build_jobs(max_jobs) == expected_archive_jobs


def test_archive_build_jobs_rejects_nonpositive_value() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        utils.get_archive_build_jobs(0)


def test_tilelang_cache_root_uses_configured_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XLLM_TILELANG_CACHE_ROOT", "tilelang-cache")

    assert utils.get_tilelang_cache_root(str(tmp_path / "unused-default")) == str(
        tmp_path / "tilelang-cache"
    )


def test_tilelang_cache_root_defaults_to_build_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("XLLM_TILELANG_CACHE_ROOT", raising=False)

    assert utils.get_tilelang_cache_root(str(tmp_path / "tilelang")) == str(
        tmp_path / "tilelang"
    )


def test_tilelang_cache_namespace_is_independent_of_checkout_path(
    tmp_path: Path,
) -> None:
    first_source_root = tmp_path / "first" / "tilelang"
    second_source_root = tmp_path / "second" / "tilelang"
    for source_root in (first_source_root, second_source_root):
        source_root.mkdir(parents=True)
        (source_root / "kernel.py").write_text("KERNEL = 1\n", encoding="utf-8")

    first_namespace = utils.get_tilelang_cache_namespace(
        str(first_source_root),
        "a3",
        {},
    )
    second_namespace = utils.get_tilelang_cache_namespace(
        str(second_source_root),
        "a3",
        {},
    )

    assert first_namespace == second_namespace


def test_tilelang_cache_namespace_tracks_source_and_platform(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "tilelang"
    source_root.mkdir()
    source_path = source_root / "kernel.py"
    source_path.write_text("KERNEL = 1\n", encoding="utf-8")

    original_namespace = utils.get_tilelang_cache_namespace(
        str(source_root),
        "a3",
        {},
    )
    a2_namespace = utils.get_tilelang_cache_namespace(
        str(source_root),
        "a2",
        {},
    )
    source_path.write_text("KERNEL = 2\n", encoding="utf-8")
    modified_namespace = utils.get_tilelang_cache_namespace(
        str(source_root),
        "a3",
        {},
    )

    assert original_namespace != a2_namespace
    assert original_namespace != modified_namespace


def test_tilelang_cache_namespace_tracks_toolchain_content(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    toolchain_root = tmp_path / "tilelang"
    source_root.mkdir()
    toolchain_root.mkdir()
    (source_root / "kernel.py").write_text("KERNEL = 1\n", encoding="utf-8")
    toolchain_source = toolchain_root / "compiler.py"
    toolchain_source.write_text("VERSION = 1\n", encoding="utf-8")

    original_namespace = utils.get_tilelang_cache_namespace(
        str(source_root),
        "a3",
        {"tilelang": str(toolchain_root)},
    )
    toolchain_source.write_text("VERSION = 2\n", encoding="utf-8")
    modified_namespace = utils.get_tilelang_cache_namespace(
        str(source_root),
        "a3",
        {"tilelang": str(toolchain_root)},
    )

    assert original_namespace != modified_namespace


def test_tilelang_cache_namespace_ignores_toolchain_location(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    first_toolchain_root = tmp_path / "first" / "tilelang"
    second_toolchain_root = tmp_path / "second" / "tilelang"
    source_root.mkdir()
    first_toolchain_root.mkdir(parents=True)
    second_toolchain_root.mkdir(parents=True)
    (source_root / "kernel.py").write_text("KERNEL = 1\n", encoding="utf-8")
    for toolchain_root in (first_toolchain_root, second_toolchain_root):
        (toolchain_root / "compiler.py").write_text(
            "VERSION = 1\n",
            encoding="utf-8",
        )

    first_namespace = utils.get_tilelang_cache_namespace(
        str(source_root),
        "a3",
        {"tilelang": str(first_toolchain_root)},
    )
    second_namespace = utils.get_tilelang_cache_namespace(
        str(source_root),
        "a3",
        {"tilelang": str(second_toolchain_root)},
    )

    assert first_namespace == second_namespace
