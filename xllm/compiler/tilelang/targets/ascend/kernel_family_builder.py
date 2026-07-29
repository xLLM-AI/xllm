from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import importlib
import multiprocessing
import os
from pathlib import Path

from ...common.cache import compute_cache_key, is_cache_hit
from ...common.manifest import KernelAbi, KernelFamilyManifest, KernelVariantManifest
from ...common.spec import DispatchField, KernelCompileSpec, KernelSpec, TilelangKernel
from ...common.toolchain import repo_root, run_checked
from . import abi_entry, kernel_registry, toolchain
from .kernel_registry import RegisteredKernelFamily
from .kernels import utils as kernel_utils
from .kernels.utils import render_family_registry_inc, render_family_variants_inc
from .toolchain import AscendBuildContext, TILELANG_BISHENG_COMMON_FLAGS


# TileLang variant compilation is much heavier than ordinary C++ compilation:
# each worker may lower Python/TVM IR and launch bisheng.  Keep the automatic
# default conservative on large hosts; for example, a 640-CPU build machine uses
# 16 workers by default instead of trying to spawn hundreds of workers.  The
# default mapping is <=4 -> 1, <=8 -> 2, <=16 -> 4, <=64 -> 8, and >64 -> 16.
_DEFAULT_TILELANG_JOB_LIMITS = (
    (4, 1),
    (8, 2),
    (16, 4),
    (64, 8),
)
_DEFAULT_TILELANG_JOBS_FOR_LARGE_HOST = 16


def default_tilelang_jobs(cpu_count: int | None = None) -> int:
    cpus = cpu_count or os.cpu_count() or 1
    for max_cpus, jobs in _DEFAULT_TILELANG_JOB_LIMITS:
        if cpus <= max_cpus:
            return jobs
    return _DEFAULT_TILELANG_JOBS_FOR_LARGE_HOST


def _resolve_tilelang_jobs(jobs: int | str | None) -> int:
    if jobs is None:
        return default_tilelang_jobs()
    resolved_jobs = int(jobs)
    if resolved_jobs < 1:
        raise ValueError(f"TileLang jobs must be positive, got: {jobs}")
    return resolved_jobs


@dataclass(frozen=True)
class _VariantBuildPlan:
    compile_spec: KernelCompileSpec
    kernel_spec: KernelSpec
    generated_source: Path
    compiled_binary: Path
    cache_key: str


@dataclass(frozen=True)
class _VariantBuildResult:
    manifest: KernelVariantManifest
    kernel_abi: KernelAbi


@dataclass(frozen=True)
class _VariantWorkerArgs:
    kernel_cls_module: str
    kernel_cls_name: str
    plan: _VariantBuildPlan
    entry_symbol: str
    bisheng_executable: str
    bisheng_arch: str
    include_dirs: tuple[str, ...]
    toolchain_options: dict
    fingerprint: dict
    compile_cwd: str


def _variant_entry_symbol(spec: KernelCompileSpec) -> str:
    kernel_entry_name = spec.entry_name or spec.kernel_name
    return f"{kernel_entry_name}__{spec.variant_key}_call"


def _write_text_if_changed(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def _run_variant_worker(args: _VariantWorkerArgs) -> _VariantBuildResult:
    mod = importlib.import_module(args.kernel_cls_module)
    kernel_cls = getattr(mod, args.kernel_cls_name)

    plan = args.plan
    compile_spec = plan.compile_spec
    kernel_spec = plan.kernel_spec

    source = kernel_cls.generate_source(**compile_spec.specialization)
    rendered_source = abi_entry.rename_variant_internal_symbols(
        abi_entry.rename_entry_symbol(
            source, compile_spec.source_entry_symbol, args.entry_symbol
        ),
        compile_spec.variant_key,
    )
    kernel_abi = abi_entry.parse_kernel_abi(rendered_source, args.entry_symbol)
    plan.generated_source.write_text(rendered_source, encoding="utf-8")

    compile_cmd = [
        args.bisheng_executable,
        f"--cce-aicore-arch={args.bisheng_arch}",
        "-DREGISTER_BASE" if args.bisheng_arch == "dav-c310" else "-DMEMORY_BASE",
        *TILELANG_BISHENG_COMMON_FLAGS,
        f"-Dg_tilingKey=g_tilingKey__{compile_spec.variant_key}",
        *[f"-I{d}" for d in args.include_dirs],
        str(plan.generated_source),
        "-c",
        "-o",
        str(plan.compiled_binary),
    ]
    run_checked(compile_cmd, cwd=args.compile_cwd)

    manifest = KernelVariantManifest(
        variant_key=compile_spec.variant_key,
        specialization=dict(compile_spec.specialization),
        dispatch_values=dict(compile_spec.dispatch_values),
        generated_source=str(plan.generated_source),
        compiled_binary=str(plan.compiled_binary),
        entry_symbol=args.entry_symbol,
        cache_key=plan.cache_key,
        toolchain_options=dict(args.toolchain_options),
        fingerprint=dict(args.fingerprint),
        compile_definitions=kernel_spec.render_compile_definitions(
            entry_symbol=args.entry_symbol
        ),
    )
    return _VariantBuildResult(manifest=manifest, kernel_abi=kernel_abi)


def _read_family_manifest(path: Path) -> KernelFamilyManifest | None:
    if not path.is_file():
        return None
    try:
        return KernelFamilyManifest.read(path)
    except Exception:
        return None


def _render_variants_inc(
    kernel_name: str,
    kernel_cls: type[TilelangKernel],
    dispatch_schema: list[DispatchField],
    variants: list[KernelVariantManifest],
) -> str:
    renderer = getattr(kernel_cls, "render_variants_inc", None)
    if renderer is None:
        return render_family_variants_inc(
            kernel_name=kernel_name,
            dispatch_schema=dispatch_schema,
            variants=variants,
        )
    if not callable(renderer):
        raise TypeError(
            f"registered kernel class '{kernel_cls.__name__}' defines "
            "non-callable render_variants_inc"
        )
    rendered = renderer(variants, dispatch_schema)
    if not isinstance(rendered, str):
        raise TypeError(
            f"registered kernel class '{kernel_cls.__name__}' "
            "render_variants_inc(...) must return str"
        )
    return rendered


def _render_registry_inc(
    kernel_name: str,
    kernel_cls: type[TilelangKernel],
    dispatch_schema: list[DispatchField],
    kernel_abi: KernelAbi,
    variants: list[KernelVariantManifest],
) -> str:
    renderer = getattr(kernel_cls, "render_registry_inc", None)
    if renderer is None:
        return render_family_registry_inc(
            kernel_name=kernel_name,
            dispatch_schema=dispatch_schema,
            kernel_abi=kernel_abi,
            variants=variants,
        )
    if not callable(renderer):
        raise TypeError(
            f"registered kernel class '{kernel_cls.__name__}' defines "
            "non-callable render_registry_inc"
        )
    rendered = renderer(variants, dispatch_schema, kernel_abi)
    if not isinstance(rendered, str):
        raise TypeError(
            f"registered kernel class '{kernel_cls.__name__}' "
            "render_registry_inc(...) must return str"
        )
    return rendered


def _build_dependency_files(family: RegisteredKernelFamily) -> list[Path]:
    # Keep cache invalidation aligned with the split builder implementation.
    files = [
        Path(family.module.__file__).resolve(),
        Path(__file__).resolve(),
        Path(toolchain.__file__).resolve(),
        Path(kernel_registry.__file__).resolve(),
        Path(abi_entry.__file__).resolve(),
        Path(kernel_utils.__file__).resolve(),
        Path(__file__).resolve().with_name("build.py"),
    ]
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def build_kernel_family(
    family: RegisteredKernelFamily,
    output_root: str | Path,
    context: AscendBuildContext,
    force: bool = False,
    jobs: int | str | None = None,
) -> KernelFamilyManifest:
    family_output_dir = Path(output_root) / "targets" / "ascend" / family.kernel_name
    family_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = family_output_dir / "manifest.json"
    existing_manifest = _read_family_manifest(manifest_path)
    dependency_files = _build_dependency_files(family)

    variant_manifest_by_key: dict[str, KernelVariantManifest] = {}
    uncached_plans: list[_VariantBuildPlan] = []
    family_kernel_abi: KernelAbi | None = None

    for compile_spec, kernel_spec in family.spec_pairs:
        if compile_spec.target != "ascend":
            raise ValueError(
                f"Unsupported target for Ascend build.py: {compile_spec.target}"
            )

        variant_output_dir = family_output_dir / compile_spec.variant_key
        variant_output_dir.mkdir(parents=True, exist_ok=True)
        generated_source = (
            variant_output_dir
            / f"{compile_spec.kernel_name}_{compile_spec.variant_key}_kernel.cpp"
        )
        compiled_binary = (
            variant_output_dir
            / f"{compile_spec.kernel_name}_{compile_spec.variant_key}_kernel.o"
        )

        cache_key = compute_cache_key(
            compile_spec,
            context.fingerprint,
            dependency_files,
        )

        cached_variant = (
            existing_manifest.get_variant(compile_spec.variant_key)
            if existing_manifest is not None
            else None
        )
        if (
            not force
            and cached_variant is not None
            and Path(cached_variant.generated_source).is_file()
            and Path(cached_variant.compiled_binary).is_file()
            and is_cache_hit(manifest_path, compile_spec.variant_key, cache_key)
        ):
            cached_source = Path(cached_variant.generated_source).read_text(
                encoding="utf-8"
            )
            kernel_abi = abi_entry.parse_kernel_abi(
                cached_source, cached_variant.entry_symbol
            )
            if family_kernel_abi is None:
                family_kernel_abi = kernel_abi
            elif kernel_abi != family_kernel_abi:
                raise ValueError(
                    "All variants in a TileLang kernel must share the same exported "
                    f"C ABI. Mismatch found in variant {compile_spec.variant_key!r}."
                )
            variant_manifest_by_key[compile_spec.variant_key] = KernelVariantManifest(
                variant_key=compile_spec.variant_key,
                specialization=dict(compile_spec.specialization),
                dispatch_values=dict(compile_spec.dispatch_values),
                generated_source=cached_variant.generated_source,
                compiled_binary=cached_variant.compiled_binary,
                entry_symbol=cached_variant.entry_symbol,
                cache_key=cached_variant.cache_key,
                toolchain_options=dict(context.toolchain_options),
                fingerprint=dict(context.fingerprint),
                compile_definitions=kernel_spec.render_compile_definitions(
                    entry_symbol=cached_variant.entry_symbol
                ),
            )
            continue

        uncached_plans.append(
            _VariantBuildPlan(
                compile_spec=compile_spec,
                kernel_spec=kernel_spec,
                generated_source=generated_source,
                compiled_binary=compiled_binary,
                cache_key=cache_key,
            )
        )

    compile_cwd = str(repo_root())
    kernel_cls_module = family.kernel_cls.__module__
    kernel_cls_name = family.kernel_cls.__name__

    worker_args_list: list[_VariantWorkerArgs] = []
    for plan in uncached_plans:
        worker_args_list.append(
            _VariantWorkerArgs(
                kernel_cls_module=kernel_cls_module,
                kernel_cls_name=kernel_cls_name,
                plan=plan,
                entry_symbol=_variant_entry_symbol(plan.compile_spec),
                bisheng_executable=context.bisheng_executable,
                bisheng_arch=context.bisheng_arch,
                include_dirs=tuple(str(d) for d in context.include_dirs),
                toolchain_options=dict(context.toolchain_options),
                fingerprint=dict(context.fingerprint),
                compile_cwd=compile_cwd,
            )
        )

    if worker_args_list:
        max_workers = _resolve_tilelang_jobs(jobs)
        # Use spawn rather than Linux's default fork.  Fork can inherit
        # TileLang/TVM lowering state from the parent process; in repeated full
        # builds this intermittently stalled after 242 variants
        # (fused_gdn_gating 240 + rope 2), before split_qkv_rmsnorm_mrope made
        # bisheng progress.
        multiprocessing_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing_context,
        ) as executor:
            future_to_args = {
                executor.submit(_run_variant_worker, args): args
                for args in worker_args_list
            }
            for future in as_completed(future_to_args):
                args = future_to_args[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        "Ascend variant build failed for variant "
                        f"{args.plan.compile_spec.variant_key!r}"
                    ) from exc
                if family_kernel_abi is None:
                    family_kernel_abi = result.kernel_abi
                elif result.kernel_abi != family_kernel_abi:
                    raise ValueError(
                        "All variants in a TileLang kernel must share the same exported "
                        "C ABI. Mismatch found in variant "
                        f"{result.manifest.variant_key!r}."
                    )
                variant_manifest_by_key[result.manifest.variant_key] = result.manifest

    variant_manifests: list[KernelVariantManifest] = [
        variant_manifest_by_key[compile_spec.variant_key]
        for compile_spec, _ in family.spec_pairs
    ]

    if family_kernel_abi is None:
        raise ValueError(
            f"TileLang kernel {family.kernel_name!r} produced no exported kernel ABI"
        )

    variants_inc_path = family_output_dir / "variants.inc"
    _write_text_if_changed(
        variants_inc_path,
        _render_variants_inc(
            family.kernel_name,
            family.kernel_cls,
            family.dispatch_schema,
            variant_manifests,
        ),
    )
    registry_inc_path = family_output_dir / "registry.inc"
    _write_text_if_changed(
        registry_inc_path,
        _render_registry_inc(
            family.kernel_name,
            family.kernel_cls,
            family.dispatch_schema,
            family_kernel_abi,
            variant_manifests,
        ),
    )

    manifest = KernelFamilyManifest(
        target="ascend",
        kernel_name=family.kernel_name,
        output_dir=str(family_output_dir),
        variants_inc=str(variants_inc_path),
        registry_inc=str(registry_inc_path),
        dispatch_schema=list(family.dispatch_schema),
        kernel_abi=family_kernel_abi,
        variants=variant_manifests,
    )
    manifest.write_if_changed(manifest_path)
    return manifest
