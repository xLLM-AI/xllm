from __future__ import annotations

import hashlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def find_installed_tilelang_root() -> Path | None:
    try:
        files = importlib.metadata.files("tilelang")
    except importlib.metadata.PackageNotFoundError:
        return None
    if files is None:
        return None

    for file in files:
        if str(file) == "tilelang/__init__.py":
            return Path(file.locate()).resolve().parent
    return None


def resolve_tilelang_root() -> Path:
    value = os.environ.get("TL_ROOT", "").strip()
    if value:
        return Path(value).resolve()

    installed_root = find_installed_tilelang_root()
    if installed_root is not None:
        return installed_root

    raise RuntimeError(
        "TileLang is not installed. Run `python xllm/compiler/tilelang_launcher.py "
        "prepare-ascend` first."
    )


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is not set: {name}")
    return value


def prepend_pythonpath(env: dict[str, str], path: str) -> None:
    current = env.get("PYTHONPATH", "")
    items = [item for item in current.split(os.pathsep) if item]
    items = [item for item in items if item != path]
    items.insert(0, path)
    env["PYTHONPATH"] = os.pathsep.join(items)


def prepare_tilelang_import(tilelang_root: str | Path | None = None) -> Path:
    tl_root = (
        Path(tilelang_root).resolve() if tilelang_root is not None else resolve_tilelang_root()
    )
    import_path = tl_root.parent
    os.environ["TL_ROOT"] = str(tl_root)
    prepend_pythonpath(os.environ, str(import_path))

    import_path_str = str(import_path)
    sys.path = [path for path in sys.path if path != import_path_str]
    sys.path.insert(0, import_path_str)
    os.environ.setdefault("ACL_OP_INIT_MODE", "1")
    return tl_root


def run_checked(
    cmd: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.check_call(list(cmd), cwd=cwd, env=env)


def sha256_file(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def git_head(path: str | Path) -> str:
    repo_path = str(Path(path).resolve())
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repo_path}", "-C", repo_path, "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def find_required_executable(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise RuntimeError(f"Required executable was not found in PATH: {name}")
    return executable
