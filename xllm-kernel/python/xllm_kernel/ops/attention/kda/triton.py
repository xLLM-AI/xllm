# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Startup-prepared Triton Ascend KDA V2 convolution solution."""

import torch

from xllm_kernel.platform import PlatformEnum, current_platform

if current_platform.is_npu():
    import torch_npu  # noqa: F401
    import triton
    from triton.runtime.jit import MockTensor

    from xllm_kernel.ops.attention.kda import CAUSAL_CONV1D_UPDATE_V2_CONTRACT
    from xllm_kernel.ops.attention.kda.npu._causal_conv1d import _causal_conv1d_update_kernel_npu_tiled_v2
    from xllm_kernel.registry import register_kernel

    # All pointers and runtime scalars opt out of value/alignment specialization.
    # Dense, varlen and verify therefore share this preloaded BF16 kernel.
    _causal_conv1d_update_kernel_npu_tiled_v2.warmup(
        x_ptr=MockTensor(dtype=torch.bfloat16),
        w_ptr=MockTensor(dtype=torch.bfloat16),
        bias_ptr=MockTensor(dtype=torch.bfloat16),
        conv_state_ptr=MockTensor(dtype=torch.bfloat16),
        conv_state_indices_ptr=MockTensor(dtype=torch.int32),
        num_accepted_tokens_ptr=MockTensor(dtype=torch.int32),
        query_start_loc_ptr=MockTensor(dtype=torch.int32),
        block_idx_last_scheduled_token_ptr=MockTensor(dtype=torch.int32),
        initial_state_idx_ptr=MockTensor(dtype=torch.int32),
        o_ptr=MockTensor(dtype=torch.bfloat16),
        batch=2,
        dim=257,
        seqlen=3,
        state_len=5,
        num_cache_lines=4,
        stride_x_dim=1,
        stride_x_token=257,
        stride_w_dim=1,
        stride_w_width=257,
        stride_conv_state_seq=1285,
        stride_conv_state_dim=1,
        stride_conv_state_tok=257,
        stride_state_indices=2,
        stride_o_dim=1,
        stride_o_token=257,
        kernel_width=4,
        is_spec_decoding=1,
        pad_slot_id=-1,
        BLOCK_N=256,
        B_TILE=1,
        T_CHUNK=16,
        grid=(2, 2),
    )._init_handles()

    def _check_tensor(tensor: torch.Tensor, name: str, *, metadata: bool = False) -> None:
        if tensor.device.type != "npu" or tensor.device.index != torch.npu.current_device():
            raise ValueError(f"{name} must be on the current NPU device")
        dtypes = (torch.int32, torch.int64) if metadata else (torch.bfloat16,)
        if tensor.dtype not in dtypes:
            raise ValueError(f"{name} must have dtype in {dtypes}")

    @register_kernel(
        name="npu.triton.causal_conv1d_update_v2",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="BF16 dense/packed tokens, width 1–6, mutable [slot,state,dim] views; int32/int64 metadata.",
        solution="triton",
        variant="jit",
        binding="python",
        contract=CAUSAL_CONV1D_UPDATE_V2_CONTRACT,
    )
    def causal_conv1d_update_v2(
        value: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        state_indices: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        max_query_len: int = 1,
        num_accepted_tokens: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        activation: bool = True,
        pad_slot_id: int = -1,
    ) -> torch.Tensor:
        for name, tensor in (("value", value), ("weight", weight), ("conv_state", conv_state)):
            _check_tensor(tensor=tensor, name=name)
        if weight.ndim != 2 or not 1 <= weight.shape[1] <= 6:
            raise ValueError("weight must be [dim, width] with width in [1, 6]")
        dim, width = weight.shape
        if dim == 0 or conv_state.ndim != 3 or conv_state.shape[0] == 0 or conv_state.shape[2] != dim:
            raise ValueError("conv_state must be [slots, state, dim] with positive slots and dim")
        if query_start_loc is None:
            if value.ndim not in (2, 3) or value.shape[1] != dim:
                raise ValueError("dense value must be [B, dim] or [B, dim, T]")
            dense = value.unsqueeze(dim=-1) if value.ndim == 2 else value
            batch, _, seqlen = dense.shape
            if batch == 0 or seqlen == 0:
                raise ValueError("dense batch and sequence length must be positive")
            x = dense.transpose(dim0=1, dim1=2).contiguous().view(size=(batch * seqlen, dim))
            lengths = torch.arange(end=batch + 1, dtype=torch.int32, device=value.device) * seqlen
        else:
            _check_tensor(tensor=query_start_loc, name="query_start_loc", metadata=True)
            if query_start_loc.ndim != 1 or query_start_loc.numel() < 2:
                raise ValueError("query_start_loc must be [sequences + 1]")
            if value.ndim != 2 or value.shape[1] != dim or max_query_len < 1:
                raise ValueError("packed value must be [tokens, dim] with positive max_query_len")
            batch = query_start_loc.numel() - 1
            seqlen = max_query_len
            x = value.contiguous()
            lengths = query_start_loc.to(dtype=torch.int32).contiguous()
        spec = num_accepted_tokens is not None
        state_len = width - 1 + (seqlen - 1 if spec else 0)
        if conv_state.shape[1] < state_len:
            raise ValueError("conv_state capacity is smaller than the convolution/verify history")
        if state_indices is None:
            if conv_state.shape[0] < batch:
                raise ValueError("default state indices require at least one slot per sequence")
            indices = torch.arange(end=batch, dtype=torch.int32, device=value.device)
        else:
            _check_tensor(tensor=state_indices, name="state_indices", metadata=True)
            if state_indices.ndim not in (1, 2) or state_indices.shape[0] != batch:
                raise ValueError("state_indices must be [sequences] or [sequences, >=2]")
            if state_indices.ndim == 2 and state_indices.shape[1] < 2:
                raise ValueError("state_indices slot table requires at least two columns")
            indices = state_indices.to(dtype=torch.int32).contiguous()
        if num_accepted_tokens is None:
            accepted = torch.ones(size=(batch,), dtype=torch.int32, device=value.device)
        else:
            _check_tensor(tensor=num_accepted_tokens, name="num_accepted_tokens", metadata=True)
            if num_accepted_tokens.shape != (batch,):
                raise ValueError("num_accepted_tokens must be [sequences]")
            accepted = num_accepted_tokens.to(dtype=torch.int32).contiguous()
        if bias is None:
            bias_kernel = torch.zeros(size=(dim,), dtype=weight.dtype, device=value.device)
        else:
            _check_tensor(tensor=bias, name="bias")
            if bias.shape != (dim,):
                raise ValueError("bias must be [dim]")
            bias_kernel = bias.contiguous()
        state = conv_state.contiguous()
        weights = weight.transpose(dim0=0, dim1=1).contiguous()
        output = torch.empty_like(input=x)
        zero_indices = torch.zeros(size=(batch,), dtype=torch.int32, device=value.device)
        _causal_conv1d_update_kernel_npu_tiled_v2[(batch, triton.cdiv(x=dim, y=256))](
            x_ptr=x,
            w_ptr=weights,
            bias_ptr=bias_kernel,
            conv_state_ptr=state,
            conv_state_indices_ptr=indices,
            num_accepted_tokens_ptr=accepted,
            query_start_loc_ptr=lengths,
            block_idx_last_scheduled_token_ptr=zero_indices,
            initial_state_idx_ptr=zero_indices,
            o_ptr=output,
            batch=batch,
            dim=dim,
            seqlen=seqlen,
            state_len=state_len,
            num_cache_lines=state.shape[0],
            stride_x_dim=1,
            stride_x_token=x.stride(dim=0),
            stride_w_dim=1,
            stride_w_width=weights.stride(dim=0),
            stride_conv_state_seq=state.stride(dim=0),
            stride_conv_state_dim=1,
            stride_conv_state_tok=state.stride(dim=1),
            stride_state_indices=indices.stride(dim=0),
            stride_o_dim=1,
            stride_o_token=output.stride(dim=0),
            kernel_width=width,
            is_spec_decoding=int(spec),
            pad_slot_id=pad_slot_id,
            BLOCK_N=256,
            B_TILE=1,
            T_CHUNK=16,
        )
        if not conv_state.is_contiguous():
            conv_state.copy_(other=state)
        return torch.nn.functional.silu(input=output) if activation else output
