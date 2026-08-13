from __future__ import annotations

from pathlib import Path

from ...common.manifest import KernelFamilyManifest


def build_kernels(
    output_root: str | Path,
    kernel_names: list[str] | None = None,
    force: bool = False,
) -> list[KernelFamilyManifest]:
    if kernel_names:
        raise NotImplementedError("CUDA TileLang AOT build pipeline is scaffolded but no kernels are registered yet.")
    return []
