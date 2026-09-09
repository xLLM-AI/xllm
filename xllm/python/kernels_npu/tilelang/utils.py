"""Helpers shared only by the independent Ascend TileLang kernels."""

import tilelang.language as T

DEFAULT_ASCEND_PASS_CONFIGS = {
    # Use raw pass-config strings to avoid hard dependency on
    # tilelang.PassConfigKey export timing/version.
    "tl.ascend_auto_sync": True,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": True,
    "tl.ascend_auto_cv_combine": True,
}

ASCEND_VEC_CORE_NUM_PROPERTY_KEYS = (
    "vector_core_num",
    "aiv_core_num",
    "vec_core_num",
)

# This is the single source of truth for the generated speculative-verification
# registries. Verification width includes the base target token, so these
# variants cover MTP depths 3, 4, and 5.
SUPPORTED_SPEC_VERIFY_WIDTHS = (4, 5, 6)


def detect_vec_core_num(default_vec_core_num: int = 48) -> int:
    try:
        import torch

        if hasattr(torch, "npu") and torch.npu.is_available():
            props = torch.npu.get_device_properties(torch.npu.current_device())
            for key in ASCEND_VEC_CORE_NUM_PROPERTY_KEYS:
                value = getattr(props, key, None)
                if isinstance(value, int) and value > 0:
                    return value
    except Exception:
        pass

    return default_vec_core_num


# ---------------------------------------------------------------------------
# Pipeline sync macros for TileLang Ascend kernels with
# tl.ascend_auto_sync=False. These helpers keep kernel implementations small
# and centralize flag-direction naming.
# ---------------------------------------------------------------------------


@T.macro
def mte2_notify_v(event_id: T.int32):
    T.set_flag("mte2", "v", event_id)


@T.macro
def v_wait_mte2(event_id: T.int32):
    T.wait_flag("mte2", "v", event_id)


@T.macro
def v_notify_mte2(event_id: T.int32):
    T.set_flag("v", "mte2", event_id)


@T.macro
def mte2_wait_v(event_id: T.int32):
    T.wait_flag("v", "mte2", event_id)


@T.macro
def v_notify_mte3(event_id: T.int32):
    T.set_flag("v", "mte3", event_id)


@T.macro
def mte3_wait_v(event_id: T.int32):
    T.wait_flag("v", "mte3", event_id)


@T.macro
def mte3_notify_v(event_id: T.int32):
    T.set_flag("mte3", "v", event_id)


@T.macro
def v_wait_mte3(event_id: T.int32):
    T.wait_flag("mte3", "v", event_id)


@T.macro
def mte3_notify_mte2(event_id: T.int32):
    T.set_flag("mte3", "mte2", event_id)


@T.macro
def mte2_wait_mte3(event_id: T.int32):
    T.wait_flag("mte3", "mte2", event_id)


@T.macro
def mte2_notify_mte3(event_id: T.int32):
    T.set_flag("mte2", "mte3", event_id)


@T.macro
def mte3_wait_mte2(event_id: T.int32):
    T.wait_flag("mte2", "mte3", event_id)
