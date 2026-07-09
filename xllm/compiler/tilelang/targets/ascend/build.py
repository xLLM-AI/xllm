import importlib
import os
import pkgutil
import re
from dataclasses import dataclass
from pathlib import Path

from ...common.manifest import KernelFamilyManifest
from ...common.toolchain import find_required_executable
from .kernel_family_builder import build_kernel_family as _build_kernel_family
from .kernel_registry import RegisteredKernelFamily, get_default_families
from .toolchain import resolve_build_context


def build_kernel_family(
    family: RegisteredKernelFamily,
    output_root: str | Path,
    force: bool = False,
    device: str | None = None,
    jobs: int | str | None = None,
) -> KernelFamilyManifest:
    context = resolve_build_context(
        device=device,
        bisheng_executable=find_required_executable("bisheng"),
    )
    return _build_kernel_family(
        family,
        output_root=output_root,
        context=context,
        force=force,
        jobs=jobs,
    )


def build_kernels(
    output_root: str | Path,
    kernel_names: list[str] | None = None,
    force: bool = False,
    device: str | None = None,
    jobs: int | str | None = None,
) -> list[KernelFamilyManifest]:
    context = resolve_build_context(
        device=device,
        bisheng_executable=find_required_executable("bisheng"),
    )
    manifests = []
    for family in get_default_families(kernel_names):
        manifests.append(
            _build_kernel_family(
                family,
                output_root=output_root,
                context=context,
                force=force,
                jobs=jobs,
            )
        )
    return manifests
