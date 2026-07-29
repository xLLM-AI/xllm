# Copyright 2026 The xLLM Authors.
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

"""Tests for build-time CPU parallelism detection."""

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

    def read_topology(cpu_id: int, topology_field: str) -> Optional[int]:
        if topology_field == "physical_package_id":
            return cpu_id // cores_per_socket
        return cpu_id % cores_per_socket

    monkeypatch.setattr(utils, "_get_available_cpu_ids", lambda: cpu_ids)
    monkeypatch.setattr(utils, "_read_cpu_topology_id", read_topology)

    assert utils.get_default_build_jobs() == cores_per_socket


def test_default_build_jobs_ignores_smt_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logical_cpus_per_socket = 8
    socket_count = 2
    threads_per_core = 2
    cpu_ids = set(range(logical_cpus_per_socket * socket_count))

    def read_topology(cpu_id: int, topology_field: str) -> Optional[int]:
        if topology_field == "physical_package_id":
            return cpu_id // logical_cpus_per_socket
        return (cpu_id % logical_cpus_per_socket) // threads_per_core

    monkeypatch.setattr(utils, "_get_available_cpu_ids", lambda: cpu_ids)
    monkeypatch.setattr(utils, "_read_cpu_topology_id", read_topology)

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

    def read_topology(cpu_id: int, topology_field: str) -> Optional[int]:
        if topology_field == "physical_package_id":
            return cpu_id // cores_per_socket
        return cpu_id % cores_per_socket

    monkeypatch.setattr(utils, "_get_available_cpu_ids", lambda: cpu_ids)
    monkeypatch.setattr(utils, "_read_cpu_topology_id", read_topology)

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
