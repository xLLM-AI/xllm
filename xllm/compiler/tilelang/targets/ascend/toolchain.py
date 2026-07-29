from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from scripts.logger import logger

from ...common.toolchain import require_env
TILELANG_BISHENG_COMMON_FLAGS = [
    "-O2",
    "-std=gnu++17",
    "-xcce",
    "-mllvm",
    "-cce-aicore-stack-size=0x8000",
    "-mllvm",
    "-cce-aicore-function-stack-size=0x8000",
    "-mllvm",
    "-cce-aicore-record-overflow=true",
    "-mllvm",
    "-cce-aicore-addr-transform",
    "-mllvm",
    "-cce-aicore-dcci-insert-for-scalar=false",
    "-DL2_CACHE_HINT",
    "-fPIC",
    "-Wno-macro-redefined",
    "-Wno-ignored-attributes",
]

DEFAULT_ASCEND_BISHENG_ARCH = "dav-c220"
ASCEND_DEVICE_TO_BISHENG_ARCH = {
    "a2": DEFAULT_ASCEND_BISHENG_ARCH,
    "a3": DEFAULT_ASCEND_BISHENG_ARCH,
    "a5": "dav-c310",
}


@dataclass(frozen=True)
class AscendBuildContext:
    device: str | None
    bisheng_arch: str
    bisheng_executable: str
    toolchain_options: dict[str, str]
    fingerprint: dict[str, str]
    include_dirs: list[str]


def normalize_ascend_device(device: str | None) -> str | None:
    if device is None:
        return None
    normalized = device.strip().lower()
    if not normalized:
        return None
    if normalized not in ASCEND_DEVICE_TO_BISHENG_ARCH:
        supported = ", ".join(sorted(ASCEND_DEVICE_TO_BISHENG_ARCH))
        raise ValueError(
            f"Unsupported Ascend TileLang device {device!r}. Expected one of: "
            f"{supported}"
        )
    return normalized


def resolve_bisheng_arch(device: str | None) -> tuple[str | None, str]:
    normalized_device = normalize_ascend_device(device)
    if normalized_device is None:
        logger.warning(
            "TileLang Ascend build did not receive --device. Falling back "
            f"to default bisheng_arch={DEFAULT_ASCEND_BISHENG_ARCH}. Prefer "
            "running via xLLM main build path or pass `--device npu` explicitly."
        )
        return None, DEFAULT_ASCEND_BISHENG_ARCH
    return normalized_device, ASCEND_DEVICE_TO_BISHENG_ARCH[normalized_device]


def build_toolchain_options(device: str | None, bisheng_arch: str) -> dict[str, str]:
    toolchain_options = {"bisheng_arch": bisheng_arch}
    if device is not None:
        toolchain_options["device"] = device
    return toolchain_options


def resolve_npu_home_path() -> str:
    for env_name in ("NPU_HOME_PATH", "NPU_TOOLKIT_HOME"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value

    for candidate in (
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/ascend-toolkit",
    ):
        if Path(candidate).exists():
            return candidate

    raise RuntimeError(
        "Required NPU toolkit root is not set. Expected NPU_HOME_PATH or "
        "NPU_TOOLKIT_HOME, or a standard install path under "
        "/usr/local/Ascend/ascend-toolkit."
    )


def bisheng_include_dirs() -> list[str]:
    tl_root = require_env("TL_ROOT")
    npu_home_path = resolve_npu_home_path()
    return [
        f"{npu_home_path}/include",
        f"{npu_home_path}/include/experiment/runtime",
        f"{npu_home_path}/include/experiment/msprof",
        f"{npu_home_path}/compiler/tikcpp",
        f"{npu_home_path}/compiler/tikcpp/tikcfw",
        f"{npu_home_path}/compiler/tikcpp/tikcfw/impl",
        f"{npu_home_path}/compiler/tikcpp/tikcfw/interface",
        f"{tl_root}/3rdparty/catlass/include",
        f"{tl_root}/3rdparty/pto-isa/include",
        f"{tl_root}/3rdparty/shmem/include",
        f"{tl_root}/3rdparty/shmem/src/device",
        f"{tl_root}/src",
    ]


def build_fingerprint(bisheng_executable: str, bisheng_arch: str) -> dict[str, str]:
    tl_root = require_env("TL_ROOT")
    npu_home_path = resolve_npu_home_path()
    return {
        "target": "ascend",
        "tl_root": tl_root,
        "npu_home_path": npu_home_path,
        "bisheng_executable": bisheng_executable,
        "bisheng_arch": bisheng_arch,
    }


def resolve_build_context(device: str | None, bisheng_executable: str) -> AscendBuildContext:
    normalized_device, bisheng_arch = resolve_bisheng_arch(device)
    fingerprint = build_fingerprint(bisheng_executable, bisheng_arch)
    if normalized_device is not None:
        fingerprint["device"] = normalized_device
    return AscendBuildContext(
        device=normalized_device,
        bisheng_arch=bisheng_arch,
        bisheng_executable=bisheng_executable,
        toolchain_options=build_toolchain_options(normalized_device, bisheng_arch),
        fingerprint=fingerprint,
        include_dirs=bisheng_include_dirs(),
    )
