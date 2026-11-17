"""Build the host-side planning extension with torch's Ninja launcher."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent


def _ensure_ninja() -> None:
    if shutil.which("ninja"):
        return
    for candidate in (Path(sys.prefix) / "bin" / "ninja", Path.home() / ".local" / "bin" / "ninja"):
        if candidate.is_file():
            os.environ["PATH"] = f"{candidate.parent}{os.pathsep}{os.environ['PATH']}"
            return
    raise RuntimeError("ninja is required to build the host extension: pip install ninja")


def load_host():
    """Compile once and return the cached extension module.

    The build directory defaults to ``host/build`` inside the submission. Set
    ``PACKED_HOST_BUILD_DIR`` to reuse a build cached elsewhere, for example on
    a read-only checkout.
    """
    _ensure_ninja()
    build_directory = Path(
        os.environ.get("PACKED_HOST_BUILD_DIR", str(_HERE / "build"))
    )
    build_directory.mkdir(parents=True, exist_ok=True)
    return load(
        name="packed_page_indices_host",
        sources=[str(_HERE / "packed_host.cpp")],
        extra_cflags=["-O3"],
        build_directory=str(build_directory),
        verbose=False,
    )
