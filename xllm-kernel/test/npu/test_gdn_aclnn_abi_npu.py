# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Exercise the packaged ACLNN ABI used by the native GDN callers."""

import ctypes as c
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch
import xllm_kernel
from xllm_kernel.ops.attention.gdn import causal_conv1d_qkv_prefill

_ACL_DTYPES = {torch.float32: 0, torch.int32: 3, torch.bfloat16: 27}
_ACL_FORMAT_ND = 2


@pytest.fixture
def aclnn_libraries() -> tuple[c.CDLL, c.CDLL]:
    """Load the operator package explicitly, independently of vendor search order."""
    vendor = Path(xllm_kernel.__file__).resolve().parent / "ascendc_opp/xllm_kernel"
    api = c.CDLL(str(vendor / "op_api/lib/libcust_opapi.so"))
    meta = c.CDLL("libnnopbase.so")
    pointer = c.POINTER(c.c_int64)
    meta.aclCreateTensor.argtypes = [
        pointer,
        c.c_uint64,
        c.c_int32,
        pointer,
        c.c_int64,
        c.c_int32,
        pointer,
        c.c_uint64,
        c.c_void_p,
    ]
    meta.aclCreateTensor.restype = c.c_void_p
    meta.aclCreateIntArray.argtypes = [pointer, c.c_uint64]
    meta.aclCreateIntArray.restype = c.c_void_p
    meta.aclDestroyTensor.argtypes = [c.c_void_p]
    meta.aclDestroyIntArray.argtypes = [c.c_void_p]
    return api, meta


def _tensor(meta: c.CDLL, stack: ExitStack, value: torch.Tensor) -> c.c_void_p:
    assert value.is_contiguous() and value.storage_offset() == 0
    dims = (c.c_int64 * value.ndim)(*value.shape)
    strides = (c.c_int64 * value.ndim)(*value.stride())
    handle = meta.aclCreateTensor(
        dims, value.ndim, _ACL_DTYPES[value.dtype], strides, 0, _ACL_FORMAT_ND, dims, value.ndim, value.data_ptr()
    )
    assert handle
    stack.callback(meta.aclDestroyTensor, handle)
    return c.c_void_p(handle)


def _int_array(meta: c.CDLL, stack: ExitStack, values: list[int]) -> c.c_void_p:
    array = (c.c_int64 * len(values))(*values)
    handle = meta.aclCreateIntArray(array, len(values))
    assert handle
    stack.callback(meta.aclDestroyIntArray, handle)
    return c.c_void_p(handle)


def _launch(api: c.CDLL, name: str, *args: object) -> None:
    size, executor = c.c_uint64(), c.c_void_p()
    workspace_query = getattr(api, f"{name}GetWorkspaceSize")
    status = workspace_query(*args, c.byref(size), c.byref(executor))
    assert status == 0 and executor.value, f"{name} workspace query returned {status}"
    workspace = torch.empty(size.value, dtype=torch.uint8, device="npu")
    launch = getattr(api, name)
    launch.argtypes = [c.c_void_p, c.c_uint64, c.c_void_p, c.c_void_p]
    status = launch(workspace.data_ptr(), size, executor, torch.npu.current_stream().npu_stream)
    assert status == 0, f"{name} launch returned {status}"
    torch.npu.synchronize()


def test_legacy_packed_qkv_abi_matches_package_binding(aclnn_libraries: tuple[c.CDLL, c.CDLL]) -> None:
    api, meta = aclnn_libraries
    generator = torch.Generator().manual_seed(2799)
    x = torch.randn((3, 384), generator=generator).bfloat16().npu()
    weight = (torch.rand((4, 384), generator=generator) * 0.2).bfloat16().npu()
    state = torch.zeros((1, 3, 384), dtype=x.dtype, device=x.device)
    expected_state = state.clone()
    expected = causal_conv1d_qkv_prefill(
        x,
        weight,
        expected_state,
        query_start_loc=[0, 3],
        state_indices=[0],
        has_initial_state=[1],
        num_qk_heads=1,
        num_v_heads=1,
        head_k_dim=128,
        head_v_dim=128,
    )
    packed = torch.empty_like(x)
    torch.npu.synchronize()
    with ExitStack() as stack:
        _launch(
            api,
            "aclnnCausalConv1dQkv",
            _tensor(meta, stack, x),
            _tensor(meta, stack, weight),
            None,
            _tensor(meta, stack, state),
            _int_array(meta, stack, [0, 3]),
            _int_array(meta, stack, [0]),
            _int_array(meta, stack, [1]),
            None,
            c.c_int64(2),
            c.c_int64(-1),
            c.c_int64(0),
            c.c_int64(128),
            c.c_int64(128),
            c.c_int64(128),
            c.c_int64(128),
            _tensor(meta, stack, packed),
        )
    torch.testing.assert_close(packed.flatten(), torch.cat([t.flatten() for t in expected]), atol=0, rtol=0)
    torch.testing.assert_close(state, expected_state, atol=0, rtol=0)


def test_legacy_recurrent_abi_updates_selected_states(aclnn_libraries: tuple[c.CDLL, c.CDLL]) -> None:
    api, meta = aclnn_libraries
    if not hasattr(api, "aclnnRecurrentGatedDeltaRule"):
        pytest.skip("The recurrent ACLNN implementation is packaged on A2/A3")

    q = torch.zeros((2, 1, 128), dtype=torch.bfloat16, device="npu")
    q[..., 0] = 1
    value = torch.linspace(-0.5, 0.5, 128, dtype=torch.bfloat16, device=q.device).repeat(2, 1, 1)
    value[1].neg_()
    beta = torch.full((2, 1), 0.5, dtype=q.dtype, device=q.device)
    # The native ABI takes a starting offset followed by per-sequence lengths.
    actual_seq_lengths = torch.tensor([0, 1, 1], dtype=torch.int32, device=q.device)
    slots = torch.tensor([2, 0], dtype=torch.int32, device=q.device)
    state = torch.zeros((3, 1, 128, 128), dtype=torch.float32, device=q.device)
    state[1].fill_(0.25)
    output = torch.empty_like(value)

    torch.npu.synchronize()
    with ExitStack() as stack:
        _launch(
            api,
            "aclnnRecurrentGatedDeltaRule",
            _tensor(meta, stack, q),
            _tensor(meta, stack, q),
            _tensor(meta, stack, value),
            _tensor(meta, stack, beta),
            _tensor(meta, stack, state),
            _tensor(meta, stack, actual_seq_lengths),
            _tensor(meta, stack, slots),
            None,
            None,
            None,
            c.c_float(1.0),
            _tensor(meta, stack, output),
        )

    torch.testing.assert_close(output, value * 0.5, atol=0, rtol=0)
    expected_state = torch.zeros_like(state)
    expected_state[1].fill_(0.25)
    expected_state[2, :, :, 0] = value[0].float() * 0.5
    expected_state[0, :, :, 0] = value[1].float() * 0.5
    torch.testing.assert_close(state, expected_state, atol=0, rtol=0)
