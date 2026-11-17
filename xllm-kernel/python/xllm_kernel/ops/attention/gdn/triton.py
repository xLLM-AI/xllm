# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Triton Ascend GDN solution, prepared before model import and capture."""

import torch

from xllm_kernel.platform import PlatformEnum, current_platform

if current_platform.is_npu():
    import torch_npu  # noqa: F401
    import triton
    from triton.runtime.jit import MockTensor

    from xllm_kernel.ops.attention.gdn import (
        FUSED_QKVZBA_SPLIT_RESHAPE_CAT_CONTRACT,
        FUSED_RECURRENT_GATED_DELTA_RULE_CONTRACT,
        FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE_CONTRACT,
    )
    from xllm_kernel.ops.attention.gdn.npu._recurrent import (
        _fused_recurrent_gated_delta_rule_fwd_kernel,
        _fused_recurrent_gated_delta_rule_spec_fwd_kernel,
    )
    from xllm_kernel.ops.attention.gdn.npu._sigmoid import _fused_sigmoid_gating_delta_rule_update_kernel
    from xllm_kernel.ops.attention.gdn.npu._split import (
        _fused_qkvzba_split_reshape_cat_gqa_r1_kernel,
        _fused_qkvzba_split_reshape_cat_gqa_r2_kernel,
        _fused_qkvzba_split_reshape_cat_gqa_r3_kernel,
        _fused_qkvzba_split_reshape_cat_gqa_r4_kernel,
    )
    from xllm_kernel.registry import register_kernel

    _SPLIT_KERNELS = (
        _fused_qkvzba_split_reshape_cat_gqa_r1_kernel,
        _fused_qkvzba_split_reshape_cat_gqa_r2_kernel,
        _fused_qkvzba_split_reshape_cat_gqa_r3_kernel,
        _fused_qkvzba_split_reshape_cat_gqa_r4_kernel,
    )
    _NUM_VECTOR_CORES = triton.runtime.driver.active.utils.get_device_properties(device=torch.npu.current_device())[
        "num_vectorcore"
    ]
    if _NUM_VECTOR_CORES <= 0:
        raise RuntimeError("Failed to detect NPU vector cores")

    # warmup only wraps positional dtypes; named pointer arguments need the
    # same Triton dtype placeholders explicitly.
    _BF16 = MockTensor(dtype=torch.bfloat16)
    _FP32 = MockTensor(dtype=torch.float32)
    _INT32 = MockTensor(dtype=torch.int32)
    _INT64 = MockTensor(dtype=torch.int64)

    # Only these structural specializations are supported. Scalar dimensions,
    # scale and pointer alignment do not specialize, so changing a batch or
    # checkpoint index cannot compile a new kernel during capture/replay.
    for _normalize in (False, True):
        for _inplace in (False, True):
            _fused_recurrent_gated_delta_rule_fwd_kernel.warmup(
                q=_BF16,
                k=_BF16,
                v=_BF16,
                g=_FP32,
                beta=_BF16,
                o=_BF16,
                h0=_FP32,
                ht=_FP32,
                cu_seqlens=_INT64,
                ssm_state_indices=_INT32,
                num_accepted_tokens=None,
                scale=0.125,
                N=2,
                T=3,
                B=1,
                H=2,
                HV=4,
                K=128,
                V=128,
                BK=128,
                BV=64,
                stride_init_state_token=65536,
                stride_final_state_token=65536,
                stride_indices_seq=1,
                stride_indices_tok=1,
                USE_INITIAL_STATE=True,
                INPLACE_FINAL_STATE=_inplace,
                IS_BETA_HEADWISE=False,
                USE_QK_L2NORM_IN_KERNEL=_normalize,
                IS_VARLEN=True,
                IS_CONTINUOUS_BATCHING=True,
                IS_SPEC_DECODING=False,
                IS_KDA=False,
                grid=(1, 2, 8),
                num_warps=1,
                num_stages=3,
            )._init_handles()
        _fused_recurrent_gated_delta_rule_spec_fwd_kernel.warmup(
            q=_BF16,
            k=_BF16,
            v=_BF16,
            g=_FP32,
            beta=_BF16,
            o=_BF16,
            h0=_FP32,
            ht=_FP32,
            cu_seqlens=_INT64,
            ssm_state_indices=_INT32,
            num_accepted_tokens=_INT32,
            scale=0.125,
            N=2,
            T=3,
            stride_indices_seq=3,
            stride_indices_tok=1,
            B=1,
            H=2,
            HV=4,
            K=128,
            V=128,
            BK=128,
            BV=64,
            stride_init_state_token=65536,
            stride_final_state_token=65536,
            IS_BETA_HEADWISE=False,
            USE_QK_L2NORM_IN_KERNEL=_normalize,
            IS_KDA=False,
            grid=(1, 2, 8),
            num_warps=1,
            num_stages=3,
            multibuffer=False,
        )._init_handles()
        _fused_sigmoid_gating_delta_rule_update_kernel.warmup(
            A_log=_FP32,
            a=_BF16,
            dt_bias=_FP32,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=_BF16,
            k=_BF16,
            v=_BF16,
            b=_BF16,
            o=_BF16,
            h0_source=_FP32,
            h0_indices=_INT32,
            cu_seqlens=_INT32,
            scale=0.125,
            B=2,
            T=3,
            H=2,
            HV=4,
            K=128,
            V=128,
            BK=128,
            BV=64,
            USE_QK_L2NORM_IN_KERNEL=_normalize,
            grid=(1, 2, 8),
            num_warps=1,
            num_stages=3,
        )._init_handles()
    for _kernel in _SPLIT_KERNELS:
        _kernel.warmup(
            mixed_qkv=_BF16,
            z=_BF16,
            b=_BF16,
            a=_BF16,
            mixed_qkvz=_BF16,
            mixed_ba=_BF16,
            num_heads_qk=2,
            num_heads_v=4,
            total_rows=3,
            rows_per_vec=3,
            qkvz_row_stride=1024,
            ba_row_stride=8,
            qkv_row_stride=768,
            z_row_stride=512,
            ba_out_row_stride=4,
            HEAD_QK=128,
            HEAD_V=128,
            ROWS_PER_ITER=1,
            grid=(1, 1),
        )._init_handles()

    def _check_tensor(tensor: torch.Tensor, name: str, dtype: torch.dtype, *, contiguous: bool = True) -> None:
        if tensor.device.type != "npu" or tensor.device.index != torch.npu.current_device():
            raise ValueError(f"{name} must be on the current NPU device")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}")
        if contiguous and not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    def _check_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            _check_tensor(tensor=tensor, name=name, dtype=torch.bfloat16)
            if tensor.ndim != 4 or tensor.numel() == 0 or tensor.shape[-1] != 128:
                raise ValueError(f"{name} must be nonempty [B, T, heads, 128]")
        if q.shape != k.shape or v.shape[:2] != q.shape[:2] or v.shape[2] % q.shape[2]:
            raise ValueError("Q/K shapes must match and value heads must be divisible by Q/K heads")

    def _check_state(state: torch.Tensor, heads: int) -> None:
        _check_tensor(tensor=state, name="state", dtype=torch.float32)
        if state.ndim != 4 or state.shape[0] == 0 or state.shape[1:] != (heads, 128, 128):
            raise ValueError("state must be [slots, value_heads, 128, 128] in KV layout")

    @register_kernel(
        name="npu.triton.fused_recurrent_gated_delta_rule",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Packed BF16 128D Q/K/V and beta, FP32 g/KV state, int64 lengths, int32 slots/checkpoints.",
        solution="triton",
        variant="jit",
        binding="python",
        contract=FUSED_RECURRENT_GATED_DELTA_RULE_CONTRACT,
    )
    def fused_recurrent_gated_delta_rule(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        inplace_final_state: bool = True,
        scale: float | None = None,
        use_qk_l2norm_in_kernel: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _check_qkv(q=q, k=k, v=v)
        if q.shape[0] != 1:
            raise ValueError("recurrent GDN requires packed [1, tokens, heads, 128] input")
        _check_tensor(tensor=g, name="g", dtype=torch.float32)
        _check_tensor(tensor=beta, name="beta", dtype=torch.bfloat16)
        if g.shape != v.shape[:-1] or beta.shape != g.shape:
            raise ValueError("g/beta must be [1, tokens, value_heads]")
        _check_state(state=initial_state, heads=v.shape[2])
        _check_tensor(tensor=cu_seqlens, name="cu_seqlens", dtype=torch.int64)
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be [sequences + 1]")
        sequences = cu_seqlens.numel() - 1
        _check_tensor(tensor=ssm_state_indices, name="ssm_state_indices", dtype=torch.int32, contiguous=False)
        spec_verify = num_accepted_tokens is not None
        if spec_verify:
            _check_tensor(tensor=num_accepted_tokens, name="num_accepted_tokens", dtype=torch.int32)
            if not inplace_final_state:
                raise ValueError("spec verify requires in-place checkpoint state")
            if ssm_state_indices.ndim != 2 or ssm_state_indices.shape[0] != sequences:
                raise ValueError("spec verify requires [sequences, checkpoint_capacity] state indices")
            if ssm_state_indices.shape[1] == 0 or num_accepted_tokens.shape != (sequences,):
                raise ValueError("checkpoint capacity must be positive and accepted counts must be [sequences]")
        else:
            if not ssm_state_indices.is_contiguous():
                raise ValueError("ordinary recurrence requires contiguous state indices")
            if ssm_state_indices.shape != (sequences,):
                raise ValueError("ordinary recurrence requires state indices [sequences]")
            if inplace_final_state and q.shape[1] != sequences:
                raise ValueError("in-place decode requires one token per sequence")

        output = torch.empty_like(input=v)
        final_state = (
            initial_state
            if inplace_final_state
            else torch.empty(size=(sequences, v.shape[2], 128, 128), dtype=torch.float32, device=q.device)
        )
        common = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            o=output,
            h0=initial_state,
            ht=final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            scale=128**-0.5 if scale is None else float(scale),
            N=sequences,
            T=q.shape[1],
            B=1,
            H=q.shape[2],
            HV=v.shape[2],
            K=128,
            V=128,
            BK=128,
            BV=64,
            stride_init_state_token=initial_state.stride(dim=0),
            stride_final_state_token=final_state.stride(dim=0),
            stride_indices_seq=ssm_state_indices.stride(dim=0),
            stride_indices_tok=ssm_state_indices.stride(dim=1) if spec_verify else 1,
            IS_BETA_HEADWISE=False,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            IS_KDA=False,
            num_warps=1,
            num_stages=3,
        )
        grid = (1, 2, sequences * v.shape[2])
        if spec_verify:
            _fused_recurrent_gated_delta_rule_spec_fwd_kernel[grid](**common, multibuffer=False)
        else:
            _fused_recurrent_gated_delta_rule_fwd_kernel[grid](
                **common,
                USE_INITIAL_STATE=True,
                INPLACE_FINAL_STATE=inplace_final_state,
                IS_VARLEN=True,
                IS_CONTINUOUS_BATCHING=True,
                IS_SPEC_DECODING=False,
            )
        return output, final_state

    @register_kernel(
        name="npu.triton.fused_sigmoid_gating_delta_rule_update",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="BF16 128D Q/K/V and gates, FP32 parameters/KV state, int32 lengths and state slots.",
        solution="triton",
        variant="jit",
        binding="python",
        contract=FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE_CONTRACT,
    )
    def fused_sigmoid_gating_delta_rule_update(
        a_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        initial_state_source: torch.Tensor,
        initial_state_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        scale: float | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        softplus_beta: float = 1.0,
        softplus_threshold: float = 20.0,
    ) -> torch.Tensor:
        _check_qkv(q=q, k=k, v=v)
        for name, tensor in (("a", a), ("b", b)):
            _check_tensor(tensor=tensor, name=name, dtype=torch.bfloat16)
            if tensor.shape != v.shape[:-1]:
                raise ValueError(f"{name} must be [B, T, value_heads]")
        for name, tensor in (("a_log", a_log), ("dt_bias", dt_bias)):
            _check_tensor(tensor=tensor, name=name, dtype=torch.float32)
            if tensor.shape != (v.shape[2],):
                raise ValueError(f"{name} must be [value_heads]")
        _check_state(state=initial_state_source, heads=v.shape[2])
        _check_tensor(tensor=cu_seqlens, name="cu_seqlens", dtype=torch.int32)
        _check_tensor(tensor=initial_state_indices, name="initial_state_indices", dtype=torch.int32)
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be [sequences + 1]")
        sequences = cu_seqlens.numel() - 1
        if initial_state_indices.shape != (sequences,):
            raise ValueError("initial_state_indices must be [sequences]")
        if softplus_beta <= 0:
            raise ValueError("softplus_beta must be positive")
        output = torch.empty_like(input=v)
        _fused_sigmoid_gating_delta_rule_update_kernel[(1, 2, sequences * v.shape[2])](
            A_log=a_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=float(softplus_beta),
            softplus_threshold=float(softplus_threshold),
            q=q,
            k=k,
            v=v,
            b=b,
            o=output,
            h0_source=initial_state_source,
            h0_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            scale=128**-0.5 if scale is None else float(scale),
            B=q.shape[0],
            T=q.shape[1],
            H=q.shape[2],
            HV=v.shape[2],
            K=128,
            V=128,
            BK=128,
            BV=64,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            num_warps=1,
            num_stages=3,
        )
        return output

    @register_kernel(
        name="npu.triton.fused_qkvzba_split_reshape_cat",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Contiguous BF16 per-QK-head QKVZ/BA matrices, 128D heads, GQA ratios 1–4.",
        solution="triton",
        variant="jit_gqa_r1_r4",
        binding="python",
        contract=FUSED_QKVZBA_SPLIT_RESHAPE_CAT_CONTRACT,
    )
    def fused_qkvzba_split_reshape_cat(
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
        num_heads_qk: int,
        num_heads_v: int,
        head_qk: int = 128,
        head_v: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _check_tensor(tensor=mixed_qkvz, name="mixed_qkvz", dtype=torch.bfloat16)
        _check_tensor(tensor=mixed_ba, name="mixed_ba", dtype=torch.bfloat16)
        if head_qk != 128 or head_v != 128:
            raise ValueError("projection split requires 128D key and value heads")
        if num_heads_qk <= 0 or num_heads_v <= 0 or num_heads_v % num_heads_qk:
            raise ValueError("value heads must be a positive multiple of Q/K heads")
        ratio = num_heads_v // num_heads_qk
        if ratio not in (1, 2, 3, 4):
            raise ValueError("projection split requires a GQA ratio in [1, 4]")
        if mixed_qkvz.ndim != 2 or mixed_ba.ndim != 2 or mixed_qkvz.shape[0] == 0:
            raise ValueError("packed projections must be nonempty matrices")
        tokens = mixed_qkvz.shape[0]
        qkv_width = num_heads_qk * 256 + num_heads_v * 128
        z_width = num_heads_v * 128
        if mixed_qkvz.shape[1] != qkv_width + z_width or mixed_ba.shape != (tokens, num_heads_v * 2):
            raise ValueError("packed projection dimensions do not match the declared heads")
        mixed_qkv = mixed_qkvz.new_empty(size=(tokens, qkv_width))
        z = mixed_qkvz.new_empty(size=(tokens, num_heads_v, 128))
        b = mixed_ba.new_empty(size=(tokens, num_heads_v))
        a = torch.empty_like(input=b)
        grid_size = min(_NUM_VECTOR_CORES, tokens)
        _SPLIT_KERNELS[ratio - 1][(grid_size, 1)](
            mixed_qkv=mixed_qkv,
            z=z,
            b=b,
            a=a,
            mixed_qkvz=mixed_qkvz,
            mixed_ba=mixed_ba,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            total_rows=tokens,
            rows_per_vec=triton.cdiv(x=tokens, y=grid_size),
            qkvz_row_stride=qkv_width + z_width,
            ba_row_stride=num_heads_v * 2,
            qkv_row_stride=qkv_width,
            z_row_stride=z_width,
            ba_out_row_stride=num_heads_v,
            HEAD_QK=128,
            HEAD_V=128,
            ROWS_PER_ITER=1,
        )
        return mixed_qkv, z, b, a
