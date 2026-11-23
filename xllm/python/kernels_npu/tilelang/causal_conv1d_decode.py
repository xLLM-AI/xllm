# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Source-tree diagnostic wrapper for the packaged TileLang decode DSL."""

import torch
from xllm_kernel.ops.attention.gdn._tilelang._causal_conv1d_decode import DIM_PER_CORE, _build_decode_kernel_jit


def causal_conv1d_decode(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | bool | None = "silu",
    conv_state_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = -1,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    original_dtype = x.dtype
    if isinstance(activation, bool):
        activation = "silu" if activation else None

    has_silu = activation in ("silu", "swish")
    width = weight.shape[1]
    dim = weight.shape[0]

    if original_dtype == torch.float16:
        x = x.to(torch.bfloat16)
        weight = weight.to(torch.bfloat16)
        conv_state_work = conv_state.to(torch.bfloat16)
        if bias is not None:
            bias_work = bias.to(torch.bfloat16).contiguous()
        else:
            bias_work = torch.zeros(dim, dtype=torch.bfloat16, device=conv_state.device)
    else:
        conv_state_work = conv_state
        if bias is not None:
            bias_work = bias.contiguous()
        else:
            bias_work = torch.zeros(dim, dtype=torch.bfloat16, device=conv_state.device)

    weight_t = weight.transpose(0, 1).contiguous()
    conv_state_t = conv_state_work.transpose(1, 2).contiguous()

    if query_start_loc is not None:
        qsl_kernel = query_start_loc.to(torch.int32).contiguous()
        batch = qsl_kernel.numel() - 1
        x_kernel = x.contiguous()
    else:
        if x.dim() == 2:
            x_work = x.unsqueeze(-1)
        else:
            x_work = x
        batch, dim_check, seqlen = x_work.shape
        assert dim_check == dim
        assert seqlen == 1
        x_kernel = x_work.reshape(batch, dim).contiguous()

    if conv_state_indices is None:
        init_indices = torch.arange(batch, dtype=torch.int32, device=conv_state.device)
        current_indices = torch.arange(batch, dtype=torch.int32, device=conv_state.device)
    elif conv_state_indices.dim() == 1:
        ci = conv_state_indices.to(torch.int32).contiguous()
        init_indices = ci
        current_indices = ci.clone()
    else:
        ci = conv_state_indices.to(torch.int32).contiguous()
        if initial_state_idx is None:
            init_indices = ci[:, 0].contiguous()
        else:
            isi = initial_state_idx.to(torch.int32).contiguous()
            init_indices = torch.where(isi == 0, ci[:, 0], ci[:, 1]).contiguous()
        if block_idx_last_scheduled_token is None:
            current_indices = ci[:, 0].contiguous()
        else:
            bilt = block_idx_last_scheduled_token.to(torch.int32).contiguous()
            current_indices = torch.where(bilt == 0, ci[:, 0], ci[:, 1]).contiguous()

    initial_state_mode = torch.ones(batch, dtype=torch.int32, device=conv_state.device)

    dim_chunks = (dim + DIM_PER_CORE - 1) // DIM_PER_CORE
    kernel = _build_decode_kernel_jit(width, dim_chunks, DIM_PER_CORE, "bfloat16", has_silu)
    output = kernel(
        x_kernel,
        weight_t,
        conv_state_t,
        init_indices,
        current_indices,
        initial_state_mode,
        bias_work,
    )

    conv_state.copy_(conv_state_t.transpose(1, 2).contiguous().to(original_dtype))

    if query_start_loc is None and x.dim() == 2:
        output = output.squeeze(-1) if output.dim() == 3 and output.shape[-1] == 1 else output

    if original_dtype == torch.float16:
        output = output.to(torch.float16)

    return output


if __name__ == "__main__":
    import torch
    import torch_npu

    batch = 1
    dim = 8192
    kernel_width = 4
    state_len = kernel_width - 1  # 3
    slots = 202

    device = "npu:0"
    tilelang.disable_cache()
    tilelang.cache.clear_cache()

    x = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    weight = torch.randn(dim, kernel_width, dtype=torch.bfloat16, device=device)
    # conv_state in PyTorch convention: [slots, dim, state_len]
    conv_state = torch.zeros(slots, dim, state_len, dtype=torch.bfloat16, device=device)
    state_indices = torch.tensor([1], dtype=torch.int32, device=device)

    print(f"x={x.shape}, weight={weight.shape}, conv_state={conv_state.shape}, state_indices={state_indices}")
    print("Calling causal_conv1d_decode...")
    result = causal_conv1d_decode(x, conv_state, weight, conv_state_indices=state_indices)
    print(f"result={result.shape}, dtype={result.dtype}")
    print(f"result first 5: {result[0, :5].tolist()}")
    print("SUCCESS")
