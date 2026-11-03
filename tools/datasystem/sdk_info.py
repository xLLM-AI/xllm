#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Print the selected SDK prefix or its deployment metadata as JSON."""
import argparse
from importlib import metadata, util
import json
import os
from pathlib import Path


def _inspect() -> dict[str, str]:
    actual = metadata.version("openyuanrong-datasystem")
    expected = os.environ.get("DATASYSTEM_VERSION", "")
    if expected and actual != expected:
        raise RuntimeError(f"SDK version mismatch: expected {expected}, installed {actual}")
    spec = util.find_spec("yr.datasystem")
    if spec is None or spec.origin is None:
        raise RuntimeError("yr.datasystem is unavailable in the selected Python")
    root = Path(spec.origin).resolve().parent
    library = next((root / p for p in ("lib/libdatasystem.so", "lib64/libdatasystem.so")
                    if (root / p).is_file()), None)
    if library is None or not (root / "include/datasystem/hetero_client.h").is_file():
        raise RuntimeError("Install the complete SDK wheel with C++ headers and libraries")
    if not (root / "datasystem_worker").is_file():
        raise RuntimeError("The selected wheel has no matching worker")
    return {"version": actual, "root": str(root), "library": str(library)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="store_true")
    args = parser.parse_args()
    result = _inspect()
    print(result["root"] if args.root else json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
