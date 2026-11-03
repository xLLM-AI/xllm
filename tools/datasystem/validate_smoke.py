#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Validate a PD response and evidence of successful backend KV transfers."""
import argparse
import json
from pathlib import Path
import re


def validate(directory: Path, backend: str) -> dict[str, object]:
    response = json.loads((directory / "response.json").read_text())
    choices = response.get("choices", [])
    content = choices[0].get("message", {}).get("content") if choices else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("PD response has no non-empty assistant content")
    prefill = (directory / "prefill.log").read_text(errors="replace")
    decode = (directory / "decode.log").read_text(errors="replace")
    if backend == "DataSystem":
        if not re.search(r"op=MSetD2H[^\n]*success=1", prefill):
            raise ValueError("No successful Prefill MSetD2H diagnostic")
        if not re.search(r"op=MGetH2D[^\n]*success=1", decode):
            raise ValueError("No successful Decode MGetH2D diagnostic")
        published = set(re.findall(
            r"op=MSetD2H[^\n]*key0=([A-Za-z0-9_-]+)[^\n]*\bsuccess=1\b", prefill
        ))
        retrieved = set(re.findall(
            r"op=MGetH2D[^\n]*key0=([A-Za-z0-9_-]+)[^\n]*\bsuccess=1\b", decode
        ))
        common = published & retrieved
        if not common:
            raise ValueError("Prefill and Decode key0 evidence does not match")
        return {"backend": backend, "matching_key0": sorted(common)}
    if not re.search(r"backend=Mooncake[^\n]*op=push[^\n]*success=1", prefill):
        raise ValueError("No successful Mooncake PUSH diagnostic")
    return {"backend": backend, "push_success": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("backend", choices=["DataSystem", "Mooncake"])
    args = parser.parse_args()
    print(json.dumps(validate(args.directory, args.backend), indent=2))


if __name__ == "__main__":
    main()
