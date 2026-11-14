from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_import_paths() -> None:
    compiler_dir = Path(__file__).resolve().parent
    package_root = compiler_dir.parent
    repo_root = package_root.parent
    # Direct script execution exposes compiler/triton as the top-level Triton
    # package. Keep this directory out of the search path for external imports.
    sys.path[:] = [path for path in sys.path if Path(path).resolve() != compiler_dir]
    for path in (repo_root, package_root, repo_root / "xllm-kernel" / "python"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Source-tree launcher for xLLM TileLang compile flows.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    subparsers.add_parser(
        "compile-kernels",
        add_help=False,
        help="Compile TileLang kernels and emit manifests.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    sys.setrecursionlimit(10000)

    parser = _build_parser()
    args, remainder = parser.parse_known_args(argv)

    _bootstrap_import_paths()

    if args.command == "compile-kernels":
        from compiler.tilelang.cli.compile_kernels import main as entrypoint
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(f"Unsupported TileLang launcher command: {args.command}")

    entrypoint(remainder)


if __name__ == "__main__":
    main()
