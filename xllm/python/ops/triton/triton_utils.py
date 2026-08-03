from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as _cann_ext

insert_slice = _cann_ext.insert_slice
extract_slice = _cann_ext.extract_slice
get_element = _cann_ext.get_element

_NUM_VECTORCORE = -1


def get_vectorcore_num() -> int:
    global _NUM_VECTORCORE
    if _NUM_VECTORCORE == -1:
        props = triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
        _NUM_VECTORCORE = props.get("num_vectorcore", -1)
    return _NUM_VECTORCORE
