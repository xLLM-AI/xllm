# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Register the package vendor before preparing an AscendC binding."""

import os
from pathlib import Path

from xllm_kernel import _npu_binding  # noqa: F401

_vendor = str(Path(__file__).resolve().parent / "ascendc_opp" / "xllm_kernel")
_previous_vendors = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(dict.fromkeys([_vendor, *filter(None, _previous_vendors)]))
