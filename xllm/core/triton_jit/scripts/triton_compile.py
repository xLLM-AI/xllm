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
# ==============================================================================
"""Runtime Triton compilation entry point (backend-aware, multi-backend).

Receives a *structured* per-argument descriptor list from C++ and builds the
Triton ASTSource inputs directly (no string-signature try-parsing). The
backend tag selects the triton.compile branch; backend-specific options arrive
as a JSON object merged on top of {num_warps, num_stages}.

    args_spec = [
        {"kind": "ptr",  "type": "fp32", "hint": "16", "specialize": True,  "const_val": ""},
        {"kind": "const","type": "i32",  "hint": "0",  "specialize": True,  "const_val": "1024"},
        ...
    ]
"""

import importlib
import importlib.util
import os
import re
from pathlib import Path
from typing import Any, List, Tuple, Union

import triton
from packaging.version import Version

triton_version = Version(triton.__version__)


_DOTTED_MODULE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+")


def _is_dotted_module(py_path: str) -> bool:
    """True for a bare dotted module path like 'torch_mlu_ops.triton.conv.kernels'.

    Distinguished from a file path (which contains '/' or ends in '.py') and
    from the explicit 'pkg://' scheme. Requires at least one dot so a lone
    identifier is never misread as a module.
    """
    if py_path.startswith("pkg://") or "/" in py_path or py_path.endswith(".py"):
        return False
    return bool(_DOTTED_MODULE_RE.fullmatch(py_path))


def _load_jit_function(py_path: str, fn_name: str) -> "triton.runtime.JITFunction":
    """Import ``py_path`` as a module and return the JIT function ``fn_name``,
    unwrapping Autotuner/Heuristics wrappers.

    ``py_path`` may be:

    * a bare dotted module path (e.g. 'torch_mlu_ops.triton.conv.kernels') --
      imported via :func:`importlib.import_module`, so kernels shipped inside an
      installed package are located the same way user code imports them, without
      baking a file path. If a normal import fails because an intermediate path
      segment is not a package, it falls back to loading the .py by file
      location resolved against the top-level package's install dir (mirroring
      the ``pkg://`` scheme), so the dotted spelling works regardless of whether
      the vendor shipped ``__init__.py`` files along the way.
    * a ``pkg://<package>/<relpath>`` reference -- resolved against the
      package's install directory (see :func:`_resolve_py_path`).
    * an absolute/relative file path -- loaded from disk (local kernels).
    """
    if _is_dotted_module(py_path):
        mod = _import_dotted_module(py_path)
    else:
        p = Path(_resolve_py_path(py_path))
        spec = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    fn = getattr(mod, fn_name)
    while type(fn) is not triton.runtime.JITFunction:
        fn = fn.fn
    return fn


def _import_dotted_module(py_path: str):
    """Import a dotted module path, falling back to file-location loading.

    Prefers a normal import (honors package ``__init__`` side effects). On
    ``ImportError`` -- typically because an intermediate segment is not a
    package -- resolves the leaf .py under the top-level package's install dir
    and loads it directly via :func:`importlib.util.spec_from_file_location`.
    """
    try:
        return importlib.import_module(py_path)
    except ImportError:
        top, _, rest = py_path.partition(".")
        base = importlib.import_module(top).__path__[0]
        file_path = os.path.join(base, *rest.split(".")) + ".py"
        spec = importlib.util.spec_from_file_location(py_path, file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def signature(py_path: str, fn_name: str) -> List[dict]:
    """Return the upstream .py's per-parameter constexpr/specialize info.

    Each entry is ``{"is_constexpr": bool, "do_not_specialize": bool}``, in
    parameter order. Consumed by the C++ JIT wrapper to verify that its
    con()/runtime classification matches the kernel's actual ``tl.constexpr``
    / ``do_not_specialize`` declarations (a mismatch shifts argument offsets
    and crashes the kernel at launch).
    """
    fn = _load_jit_function(py_path, fn_name)
    return [
        {"is_constexpr": p.is_constexpr, "do_not_specialize": p.do_not_specialize}
        for p in fn.params
    ]


def _dump_signature(py_path: str, fn_name: str) -> None:
    """Print the upstream .py's per-parameter constexpr/specialize info.

    Used to verify that a JIT wrapper's con()/runtime classification matches the
    kernel's actual `tl.constexpr` / `do_not_specialize` declarations (a mismatch
    shifts argument offsets and crashes the kernel at launch).
    """
    fn = _load_jit_function(py_path, fn_name)
    for i, p in enumerate(fn.params):
        print(f"{i}\t{p.name}\tconstexpr={p.is_constexpr}\t"
              f"do_not_specialize={p.do_not_specialize}")


def _resolve_py_path(py_path: str) -> str:
    """Resolve a kernel .py path.

    * An absolute/relative path is used as-is (local kernels).
    * A ``pkg://<package>/<relpath>`` reference is resolved against an installed
      Python package's directory, so kernels shipped inside a package (e.g. the
      ``torch_mlu_ops`` triton kernels) can be JIT-compiled without baking their
      install path at build time. Resolution happens here, in the venv
      subprocess, where the package is importable.
    """
    if py_path.startswith("pkg://"):
        rest = py_path[len("pkg://"):]
        pkg, _, relpath = rest.partition("/")
        if not pkg or not relpath:
            raise ValueError(f"invalid pkg:// path: {py_path!r}")
        base = importlib.import_module(pkg).__path__[0]
        return os.path.join(base, relpath)
    return py_path


def _parse_const(spec: dict) -> Union[bool, int, float]:
    """Type-aware constexpr parsing -- no int/float guessing."""
    t = spec["type"]
    v = spec["const_val"]
    if t == "i1":
        return v == "1"
    if t in ("fp16", "fp32", "fp64", "f16", "f32", "f64", "bf16"):
        return float(v)
    return int(v)


def _build_ast_inputs(
    args_spec: List[dict],
) -> Tuple[dict, dict, Tuple[int, ...], Tuple[int, ...]]:
    """Turn structured descriptors into (constants, signature, divisible_by_16,
    equal_to_1) for triton.compile."""
    constants: dict = {}
    signature: dict = {}
    divisible_by_16: List[int] = []
    equal_to_1: List[int] = []

    for i, s in enumerate(args_spec):
        kind = s["kind"]
        specialize = s["specialize"]
        hint = s["hint"]
        ty = s["type"]

        if kind == "const":
            constants[i] = _parse_const(s)
            continue

        baked_eq1 = kind == "val" and specialize and hint == "1"
        if baked_eq1:
            constants[i] = 1
            equal_to_1.append(i)
            continue

        sig_ty = ("*" + ty) if kind == "ptr" else ty
        signature[i] = sig_ty

        if specialize:
            if hint == "16":
                divisible_by_16.append(i)
            elif hint == "1":
                equal_to_1.append(i)

    return constants, signature, tuple(divisible_by_16), tuple(equal_to_1)


def _make_attrs(
    divisible_by_16: Tuple[int, ...], equal_to_1: Tuple[int, ...]
) -> Any:
    if triton_version.major == 3 and triton_version.minor == 1:
        return triton.compiler.AttrsDescriptor(
            divisible_by_16=divisible_by_16, equal_to_1=equal_to_1
        )
    if triton_version.major == 3 and triton_version.minor == 2:
        return triton.backends.compiler.AttrsDescriptor.from_dict(
            {
                "arg_properties": {
                    "tt.divisibility": divisible_by_16,
                    "tt.equal_to": equal_to_1,
                },
                "cls": "AttrsDescriptor",
            }
        )
    if triton_version >= Version("3.3.0"):
        return {(i,): [["tt.divisibility", 16]] for i in divisible_by_16}
    raise RuntimeError(f"unsupported triton version {triton_version}")


def _make_ast_source(
    fn: "triton.runtime.JITFunction",
    constants: dict,
    signature: dict,
    attrs: Any,
    num_args: int,
) -> Any:
    if Version("3.1.0") <= triton_version < Version("3.2.0"):
        return triton.compiler.ASTSource(
            fn=fn, constants=constants, signature=signature, attrs=attrs
        )
    if Version("3.2.0") <= triton_version < Version("3.3.0"):
        arg_names = fn.arg_names
        c = {arg_names[i]: v for i, v in constants.items()}
        sig = {arg_names[i]: v for i, v in signature.items()}
        return triton.compiler.ASTSource(fn=fn, constants=c, signature=sig, attrs=attrs)
    if triton_version >= Version("3.3.0"):
        arg_names = fn.arg_names
        c = {(i,): v for i, v in constants.items()}
        sig: dict = {}
        for i in range(num_args):
            if i in signature:
                sig[arg_names[i]] = signature[i]
            elif i in constants:
                sig[arg_names[i]] = "constexpr"
            else:
                raise ValueError(f"argument {i} neither in signature nor constants")
        return triton.compiler.ASTSource(
            fn=fn, signature=sig, constexprs=c, attrs=attrs
        )
    raise RuntimeError(f"unsupported triton version {triton_version}")


def _do_compile(
    src: Any,
    backend: str,
    options: dict,
    device_id: int,
) -> Any:
    """Backend-specific triton.compile invocation."""
    if backend == "mlu":
        import torch  # noqa: F401  -- ensures torch_mlu autoload path is consistent
        target = triton.runtime.driver.active.get_current_target()
        mlu_backend = triton.compiler.make_backend(target)
        opts = mlu_backend.parse_options(options).__dict__
        return triton.compile(src, target=target, options=opts)
    if backend == "cuda":
        import torch
        with torch.cuda.device(device_id):
            target = triton.runtime.driver.active.get_current_target()
            return triton.compile(src, target=target, options=options)
    raise ValueError(f"unknown backend {backend!r}")


def compile(
    py_path: str,
    fn_name: str,
    args_spec: list,
    backend: str,
    extra_opts: dict,
    num_warps: int,
    num_stages: int,
    device_id: int,
) -> str:
    """Compile one specialization and return the triton cache directory
    containing the backend's compiled artifacts (e.g. {fn_name}.cnbin / .cubin)
    and {fn_name}.json metadata."""
    fn = _load_jit_function(py_path, fn_name)
    num_args = len(args_spec)
    assert num_args == len(fn.params), (
        f"argument count mismatch: got {num_args}, function has {len(fn.params)}"
    )

    constants, signature, div16, eq1 = _build_ast_inputs(args_spec)
    attrs = _make_attrs(div16, eq1)
    src = _make_ast_source(fn, constants, signature, attrs, num_args)

    options = {"num_warps": num_warps, "num_stages": num_stages}
    options.update(extra_opts)
    ccinfo = _do_compile(src, backend, options, device_id)

    from triton.runtime.cache import get_cache_manager
    cache_dir = get_cache_manager(ccinfo.hash).cache_dir
    return str(cache_dir)


if __name__ == "__main__":
    import sys

    # Diagnostic: dump the upstream .py's parameter constexpr/specialize info.
    #   python -m xllm.core.triton_jit.scripts.triton_compile --dump-sig \
    #       <py_path> <fn_name>
    if len(sys.argv) == 4 and sys.argv[1] == "--dump-sig":
        _dump_signature(sys.argv[2], sys.argv[3])
        sys.exit(0)

    sys.stderr.write(
        "usage: python -m xllm.core.triton_jit.scripts.triton_compile "
        "--dump-sig <py_path> <fn_name>\n"
    )
    sys.exit(2)
