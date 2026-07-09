from __future__ import annotations

import argparse
from pathlib import Path

from scripts.logger import logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile xLLM TileLang kernels and emit manifests."
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["ascend", "cuda"],
        help="Compilation target backend.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output root for compiled TileLang artifacts.",
    )
    parser.add_argument(
        "--device",
        choices=["a2", "a3", "a5"],
        default=None,
        help="Ascend device type used to resolve build-time toolchain settings.",
    )
    parser.add_argument(
        "--kernels",
        nargs="*",
        default=None,
        help="Optional kernel names. Compile all registered kernels when omitted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompilation even when cache is hit.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Maximum parallel workers for compiling TileLang variants, e.g. --jobs 16.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.target == "ascend":
        from ..bootstrap import ensure_ascend_ready

        if args.device is None:
            raise ValueError("--device is required for Ascend TileLang kernel compilation.")
        ensure_ascend_ready()
        from ..targets.ascend.build import build_kernels

        manifests = build_kernels(
            output_root=output_root,
            kernel_names=args.kernels,
            force=args.force,
            device=args.device,
            jobs=args.jobs,
        )
    elif args.target == "cuda":
        from ..targets.cuda.build import build_kernels

        manifests = build_kernels(
            output_root=output_root,
            kernel_names=args.kernels,
            force=args.force,
        )
    else:
        raise ValueError(f"Unsupported target: {args.target}")
    for manifest in manifests:
        logger.info(f"built {manifest.target}:{manifest.kernel_name}")
        logger.info(f"manifest: {Path(manifest.output_dir) / 'manifest.json'}")


if __name__ == "__main__":
    main()
