from __future__ import annotations

import argparse

from scripts.logger import logger

from ..bootstrap import PREPARE_ASCEND_COMMAND, prepare_ascend


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install and verify the TileLang Ascend wheel for xLLM builds."
    )
    parser.add_argument(
        "--target-platform",
        required=True,
        choices=["a2", "a3", "a5"],
        help="Ascend platform used to select the TileLang wheel.",
    )
    parser.add_argument(
        "--arch",
        required=True,
        choices=["arm", "x86"],
        help="CPU architecture used to select the TileLang wheel.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reinstalling the TileLang Ascend wheel even when it looks ready.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tilelang_root = prepare_ascend(
        args.target_platform,
        args.arch,
        force=args.force,
    )
    logger.info(f"tilelang is ready under: {tilelang_root}")
    logger.info("Next step: run your usual `python setup.py build ...` or `python setup.py test ...` command.")
    logger.info(f"Re-run this step explicitly with: `{PREPARE_ASCEND_COMMAND}`")


if __name__ == "__main__":
    main()
