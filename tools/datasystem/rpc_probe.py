#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Initialize a real HeteroClient against the configured worker."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
from yr.datasystem import HeteroClient


def main() -> None:
    client = HeteroClient(os.environ["DATASYSTEM_WORKER_HOST"],
                          int(os.environ["DATASYSTEM_WORKER_PORT"]),
                          connect_timeout_ms=60000)
    client.init()
    logger.info("SDK_RPC_INIT_OK")


if __name__ == "__main__":
    main()
