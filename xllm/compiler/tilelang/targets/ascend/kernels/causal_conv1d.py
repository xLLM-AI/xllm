#!/usr/bin/env python3

# Copyright 2025-2026 The xLLM Authors.
import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from ....common.spec import DispatchField, TilelangKernel, register_kernel
from .utils import detect_vec_core_num

VEC_NUM = 2

symbol_cache_lines = T.symbolic("num_cache_lines")
symbol_state_len = T.symbolic("state_len")

pass_configs_config = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_prefill_kernel_cache = {}


def build_causal_conv1d_kernel(
    width: int,
    block_dim: int,
    vec_core_num: int,
    dtype_str: str = "float16",
    has_silu: bool = False,
) -> torch.nn.Module:
    hist_len = width - 1
    padded_dim = block_dim * vec_core_num
    m_num = vec_core_num // VEC_NUM
    symbol_total_len = T.symbolic("total_len")

    @T.prim_func
    def kernel_func(
        x: T.Tensor((symbol_total_len, padded_dim), dtype_str),
        weight: T.Tensor((width, padded_dim), dtype_str),
        conv_state: T.Tensor((symbol_cache_lines, symbol_state_len, padded_dim), dtype_str),
        conv_state_indices_init: T.Tensor((1,), "int32"),
        conv_state_indices_current: T.Tensor((1,), "int32"),
        cu_seqlens: T.Tensor((2,), "int32"),
        initial_state_mode: T.Tensor((1,), "int32"),
        bias: T.Tensor((padded_dim,), dtype_str),
        y: T.Tensor((symbol_total_len, padded_dim), dtype_str),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            d_offset = task_id * block_dim

            read_cache_line = conv_state_indices_init[0]
            write_cache_line = conv_state_indices_current[0]
            seq_start = cu_seqlens[0]
            seq_end = cu_seqlens[1]
            seqlen = seq_end - seq_start
            global_start = seq_start

            has_initial = initial_state_mode[0]
            hist_base = global_start - hist_len

            with T.Scope("V"):
                hist0 = T.alloc_ub((block_dim,), dtype_str)
                hist1 = T.alloc_ub((block_dim,), dtype_str)
                hist2 = T.alloc_ub((block_dim,), dtype_str)
                w0 = T.alloc_ub((block_dim,), dtype_str)
                w1 = T.alloc_ub((block_dim,), dtype_str)
                w2 = T.alloc_ub((block_dim,), dtype_str)
                w3 = T.alloc_ub((block_dim,), dtype_str)
                state0 = T.alloc_ub((block_dim,), dtype_str)
                state1 = T.alloc_ub((block_dim,), dtype_str)
                state2 = T.alloc_ub((block_dim,), dtype_str)
                tmp = T.alloc_ub((block_dim,), dtype_str)
                bias_ub = T.alloc_ub((block_dim,), dtype_str)
                x_ub = T.alloc_ub((block_dim,), dtype_str)
                y_ub = T.alloc_ub((block_dim,), dtype_str)
                save0 = T.alloc_ub((block_dim,), dtype_str)
                save1 = T.alloc_ub((block_dim,), dtype_str)
                save2 = T.alloc_ub((block_dim,), dtype_str)

                T.copy(weight[0, d_offset], w0)
                T.copy(weight[1, d_offset], w1)
                T.copy(weight[2, d_offset], w2)
                T.copy(weight[3, d_offset], w3)
                T.copy(bias[d_offset], bias_ub)

                T.tile.fill(hist0, 0.0)
                T.tile.fill(hist1, 0.0)
                T.tile.fill(hist2, 0.0)

                if has_initial != 0:
                    if hist_len >= 1 and symbol_state_len > 0:
                        T.copy(conv_state[read_cache_line, 0, d_offset], hist0)
                    if hist_len >= 2 and symbol_state_len > 1:
                        T.copy(conv_state[read_cache_line, 1, d_offset], hist1)
                    if hist_len >= 3 and symbol_state_len > 2:
                        T.copy(conv_state[read_cache_line, 2, d_offset], hist2)
                else:
                    if hist_len >= 1:
                        T.copy(x[hist_base, d_offset], hist0)
                    if hist_len >= 2:
                        T.copy(x[hist_base + 1, d_offset], hist1)
                    if hist_len >= 3:
                        T.copy(x[hist_base + 2, d_offset], hist2)

                T.tile.mul(state2, w0, hist2)
                T.tile.mul(state1, w0, hist1)
                T.tile.mul(tmp, w1, hist2)
                T.tile.add(state1, state1, tmp)
                T.tile.mul(state0, w0, hist0)
                T.tile.mul(tmp, w1, hist1)
                T.tile.add(state0, state0, tmp)
                T.tile.mul(tmp, w2, hist2)
                T.tile.add(state0, state0, tmp)

                for t in T.serial(seqlen):
                    token_idx = global_start + t
                    T.copy(x[token_idx, d_offset], x_ub)

                    T.tile.mul_add_dst(state0, x_ub, w3)
                    T.tile.add(tmp, state0, bias_ub)

                    if has_silu:
                        T.tile.silu(y_ub, tmp)
                    else:
                        T.tile.add(y_ub, state0, bias_ub)

                    T.copy(y_ub, y[token_idx, d_offset])

                    T.tile.mul(tmp, w2, x_ub)
                    T.tile.add(state0, tmp, state1)
                    T.tile.mul(tmp, w1, x_ub)
                    T.tile.add(state1, tmp, state2)
                    T.tile.mul(state2, w0, x_ub)

                if seqlen > 0:
                    T.copy(x[seq_end - 1, d_offset], save2)

                    if seqlen >= 2:
                        T.copy(x[seq_end - 2, d_offset], save1)
                    else:
                        T.copy(hist2, save1)

                    if seqlen >= 3:
                        T.copy(x[seq_end - 3, d_offset], save0)
                    else:
                        if seqlen >= 2:
                            T.copy(hist2, save0)
                        else:
                            T.copy(hist1, save0)

                    if hist_len >= 1 and symbol_state_len > 0:
                        T.copy(save0, conv_state[write_cache_line, 0, d_offset])
                    if hist_len >= 2 and symbol_state_len > 1:
                        T.copy(save1, conv_state[write_cache_line, 1, d_offset])
                    if hist_len >= 3 and symbol_state_len > 2:
                        T.copy(save2, conv_state[write_cache_line, 2, d_offset])

    return kernel_func


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs_config)
def _build_prefill_kernel_jit(
    width: int,
    block_dim: int,
    vec_core_num: int,
    dtype_str: str = "float16",
    has_silu: bool = False,
) -> torch.nn.Module:
    return build_causal_conv1d_kernel(
        width=width,
        block_dim=block_dim,
        vec_core_num=vec_core_num,
        dtype_str=dtype_str,
        has_silu=has_silu,
    )


@register_kernel
class CausalConv1dKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("batch_size", "int32"),
        DispatchField("dim", "int32"),
        DispatchField("width", "int32"),
        DispatchField("has_silu", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": f"bs1_d{d}_w4_silu0_f16",
            "batch_size": 1,
            "dim": d,
            "width": 4,
            "has_silu": 0,
            "dtype": "float16",
        }
        for d in sorted(
            {
                dim // tp
                for dim in [2048, 4096, 5120, 6144, 8192]
                for tp in [1, 2, 4, 8]
                if dim % tp == 0
            }
        )
    ]

    @staticmethod
    def generate_source(
        batch_size: int,
        dim: int,
        width: int,
        has_silu: int,
        dtype: str,
    ) -> str:
        if dtype not in ("float16", "bfloat16"):
            raise ValueError(
                f"CausalConv1D TileLang kernel only supports dtype=float16/bfloat16, "
                f"got {dtype}"
            )
        vec_core_num = detect_vec_core_num()
        block_dim = (dim + vec_core_num - 1) // vec_core_num
        tilelang.disable_cache()
        tilelang_kernel = build_causal_conv1d_kernel(
            width=width,
            block_dim=block_dim,
            vec_core_num=vec_core_num,
            dtype_str=dtype,
            has_silu=bool(has_silu),
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3,
            config={
                "tl.ascend_auto_cv_combine": True,
                "tl.ascend_auto_sync": True,
                "tl.ascend_memory_planning": True,
            },
        ):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source


# ======================== Kernel Cache & Getter ========================


def get_prefill_kernel(
    width: int, num_batches: int, dim: int,
    dtype_str: str = "float16", has_silu: bool = False,
) -> torch.nn.Module:
    vec_core_num = detect_vec_core_num()
    block_dim = (dim + vec_core_num - 1) // vec_core_num
    cache_key = (width, block_dim, vec_core_num, dtype_str, has_silu)
    if cache_key not in _prefill_kernel_cache:
        _prefill_kernel_cache[cache_key] = _build_prefill_kernel_jit(
            width, block_dim, vec_core_num, dtype_str, has_silu
        )
    return _prefill_kernel_cache[cache_key]


def _compute_padded_dim(dim: int, vec_core_num: int) -> int:
    block_dim = (dim + vec_core_num - 1) // vec_core_num
    return block_dim * vec_core_num


def _pad_last_dim(tensor: torch.Tensor, padded_dim: int) -> torch.Tensor:
    dim = tensor.shape[-1]
    if padded_dim <= dim:
        return tensor
    pad_size = padded_dim - dim
    return F.pad(tensor, (0, pad_size))


# ======================== Wrapper ========================


def causal_conv1d_update_v2(
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
    """Unified causal_conv1d wrapper, feature-aligned with Triton V2.

    Args:
        x: [total_tokens, dim] (varlen) or [batch, dim] (decode) or [batch, dim, seqlen]
        conv_state: [cache_lines, dim, state_len] — PyTorch convention
        weight: [dim, width] — PyTorch convention
        bias: [dim] or None
        activation: "silu"/"swish" or None
        conv_state_indices: [batch] (1D) or [batch, >=2] (2D APC), int32
        query_start_loc: [batch+1], int32 (varlen) or None
        max_query_len: max query length (used when query_start_loc is provided)
        block_idx_last_scheduled_token: [batch], int32 (APC) or None
        initial_state_idx: [batch], int32 (APC) or None
    """
    original_dtype = x.dtype
    if isinstance(activation, bool):
        activation = "silu" if activation else None

    has_silu = activation in ("silu", "swish")
    width = weight.shape[1]
    dim = weight.shape[0]

    vec_core_num = detect_vec_core_num()
    padded_dim = _compute_padded_dim(dim, vec_core_num)

    if original_dtype == torch.bfloat16:
        x = x.to(torch.float16)
        weight = weight.to(torch.float16)
        conv_state_work = conv_state.to(torch.float16).clone()
        if bias is not None:
            bias_work = bias.to(torch.float16).contiguous()
        else:
            bias_work = torch.zeros(dim, dtype=torch.float16, device=conv_state.device)
    else:
        conv_state_work = conv_state.clone()
        if bias is not None:
            bias_work = bias.contiguous()
        else:
            bias_work = torch.zeros(dim, dtype=conv_state.dtype, device=conv_state.device)

    weight_t = weight.transpose(0, 1).contiguous()
    conv_state_t = conv_state_work.transpose(1, 2).contiguous()

    if query_start_loc is not None:
        x_kernel = x.contiguous()
        qsl_kernel = query_start_loc.to(torch.int32).contiguous()
        batch = qsl_kernel.numel() - 1
        seqlen = max_query_len if max_query_len > 0 else 1
    else:
        if x.dim() == 2:
            x_work = x.unsqueeze(-1)
        else:
            x_work = x
        batch, dim_check, seqlen = x_work.shape
        assert dim_check == dim
        x_kernel = x_work.transpose(1, 2).contiguous().reshape(batch * seqlen, dim)
        qsl_kernel = torch.arange(
            0, (batch + 1) * seqlen, seqlen,
            device=conv_state.device, dtype=torch.int32,
        )

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

    if initial_state_idx is not None:
        initial_state_mode = torch.ones(batch, dtype=torch.int32, device=conv_state.device)
    elif conv_state_indices is not None and conv_state_indices.dim() == 1:
        initial_state_mode = torch.ones(batch, dtype=torch.int32, device=conv_state.device)
    else:
        initial_state_mode = torch.ones(batch, dtype=torch.int32, device=conv_state.device)

    kernel_dtype_str = "float16" if (original_dtype == torch.float16 or original_dtype == torch.bfloat16) else "float32"

    output_parts = []
    for b in range(batch):
        seq_start_b = qsl_kernel[b].item()
        seq_end_b = qsl_kernel[b + 1].item()
        sb_len = seq_end_b - seq_start_b
        if sb_len <= 0:
            continue

        cu_b = torch.tensor([0, sb_len], dtype=torch.int32, device=conv_state.device)
        x_b = x_kernel[seq_start_b:seq_end_b]
        init_b = init_indices[b:b+1]
        curr_b = current_indices[b:b+1]
        ism_b = initial_state_mode[b:b+1]

        kernel = get_prefill_kernel(
            width, 1, dim, kernel_dtype_str, has_silu=False
        )

        x_b_padded = _pad_last_dim(x_b, padded_dim)
        weight_t_padded = _pad_last_dim(weight_t, padded_dim)
        conv_state_t_padded = _pad_last_dim(conv_state_t, padded_dim)
        bias_work_padded = _pad_last_dim(bias_work, padded_dim)

        output_padded = kernel(
            x_b_padded, weight_t_padded, conv_state_t_padded,
            init_b, curr_b, cu_b, ism_b, bias_work_padded,
        )

        output_b = output_padded[:, :dim]
        output_parts.append((seq_start_b, seq_end_b, output_b))

        conv_state_t = conv_state_t_padded[:, :, :dim].contiguous()

    total_tokens = x_kernel.shape[0]
    output = torch.zeros(total_tokens, dim, dtype=x_kernel.dtype, device=conv_state.device)
    for seq_start_b, seq_end_b, output_b in output_parts:
        output[seq_start_b:seq_end_b] = output_b

    if has_silu:
        output = F.silu(output)

    conv_state.copy_(conv_state_t.transpose(1, 2).contiguous().to(original_dtype))

    if query_start_loc is None:
        output = output.view(batch, seqlen, dim).transpose(1, 2).contiguous()
        if x.dim() == 2:
            output = output.squeeze(-1)

    if original_dtype == torch.bfloat16:
        output = output.to(torch.bfloat16)

    return output


# ======================== Reference ========================


def causal_conv1d_update_v2_ref(
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
    """CPU golden reference, matching Triton V2 behavior."""
    dtype_in = x.dtype
    x_f = x.float()
    weight_f = weight.float()
    conv_state_f = conv_state.float().clone()
    bias_f = bias.float() if bias is not None else None
    width = weight.shape[1]
    hist_len = width - 1

    if query_start_loc is None:
        if x.dim() == 2:
            x_work = x_f.unsqueeze(-1)
        else:
            x_work = x_f
        batch = x_work.shape[0]
        dim = x_work.shape[1]
        seqlen = x_work.shape[2]
        qsl = torch.arange(0, (batch + 1) * seqlen, dtype=torch.long)
    else:
        batch = conv_state_indices.shape[0] if conv_state_indices is not None else query_start_loc.numel() - 1
        dim = x_f.shape[1]
        seqlen = max_query_len
        qsl = query_start_loc.long()

    out = torch.zeros_like(x_f if query_start_loc is not None else x_work)

    if conv_state_indices is None:
        ci = torch.arange(batch, dtype=torch.long)
    elif conv_state_indices.dim() == 1:
        ci = conv_state_indices.long()
    else:
        ci = conv_state_indices.long()

    for b in range(batch):
        if ci.dim() == 1:
            read_line = ci[b].item()
            write_line = ci[b].item()
        else:
            init_slot = initial_state_idx[b].item() if initial_state_idx is not None else 0
            last_slot = block_idx_last_scheduled_token[b].item() if block_idx_last_scheduled_token is not None else 0
            read_line = ci[b, init_slot].item()
            write_line = ci[b, last_slot].item()

        if read_line == pad_slot_id:
            continue

        qs = qsl[b].item()
        qe = qsl[b + 1].item()
        seq_len_run = qe - qs
        if seq_len_run == 0:
            continue

        history = torch.zeros(hist_len, dim, dtype=torch.float32)
        has_initial = True
        if has_initial:
            for h in range(hist_len):
                if h < conv_state_f.shape[2]:
                    history[h] = conv_state_f[read_line, :, h].clone()

        for t in range(seq_len_run):
            acc = torch.zeros(dim, dtype=torch.float32)
            for w in range(hist_len):
                acc += weight_f[:, w] * history[w]
            acc += weight_f[:, width - 1] * x_f[qs + t]
            if bias_f is not None:
                acc += bias_f
            if activation in ("silu", "swish"):
                acc = acc / (1 + torch.exp(-acc))
            if query_start_loc is not None:
                out[qs + t] = acc
            else:
                out[b, :, t] = acc
            if hist_len > 1:
                history[:-1] = history[1:].clone()
            history[-1] = x_f[qs + t].clone()

        for h in range(hist_len):
            if h < conv_state_f.shape[2]:
                idx = seq_len_run - hist_len + h
                if idx >= 0:
                    conv_state_f[write_line, :, h] = x_f[qs + idx]

    conv_state.copy_(conv_state_f.to(dtype_in))

    if query_start_loc is None:
        out = out.to(dtype_in)
        if x.dim() == 2:
            out = out.squeeze(-1)
    else:
        out = out.to(dtype_in)

    return out


# ======================== Tests ========================


def _run_decode_test(dtype_str: str, dtype: torch.dtype, batch_size: int = 2) -> None:
    tilelang.cache.clear_cache()
    _prefill_kernel_cache.clear()
    torch.manual_seed(42)

    dim = 2048
    width = 4
    state_len = width - 1
    num_cache_lines = max(batch_size + 2, 8)
    seqlen = 1

    print(f"  Decode: batch={batch_size}, dim={dim}, width={width}, dtype={dtype_str}")

    x = torch.randn(batch_size, dim, dtype=dtype, device="npu")
    conv_state = torch.randn(num_cache_lines, dim, state_len, dtype=dtype, device="npu")
    weight = torch.randn(dim, width, dtype=dtype, device="npu")
    bias = torch.randn(dim, dtype=dtype, device="npu")

    ci = torch.arange(batch_size, dtype=torch.int32, device="npu")
    qsl = torch.arange(0, batch_size + 1, dtype=torch.int32, device="npu")

    x_ref = x.cpu()
    cs_ref = conv_state.cpu().clone()
    w_ref = weight.cpu()
    b_ref = bias.cpu()
    ci_ref = ci.cpu()
    qsl_ref = qsl.cpu()

    golden = causal_conv1d_update_v2_ref(
        x_ref, cs_ref, w_ref, bias=b_ref, activation="silu",
        conv_state_indices=ci_ref, query_start_loc=qsl_ref, max_query_len=1,
    )

    out = causal_conv1d_update_v2(
        x, conv_state, weight, bias=bias, activation="silu",
        conv_state_indices=ci, query_start_loc=qsl, max_query_len=1,
    )

    torch.testing.assert_close(out.cpu(), golden, rtol=1e-2, atol=1e-2)
    print(f"    PASS — decode batch={batch_size}")


def _run_decode_apc_test(dtype_str: str, dtype: torch.dtype) -> None:
    tilelang.cache.clear_cache()
    _prefill_kernel_cache.clear()
    torch.manual_seed(42)

    batch_size = 2
    dim = 2048
    width = 4
    state_len = width - 1
    num_cache_lines = 8
    seqlen = 1

    print(f"  Decode APC: batch={batch_size}, dim={dim}, dtype={dtype_str}")

    x = torch.randn(batch_size, dim, dtype=dtype, device="npu")
    conv_state = torch.randn(num_cache_lines, dim, state_len, dtype=dtype, device="npu")
    weight = torch.randn(dim, width, dtype=dtype, device="npu")
    bias = torch.randn(dim, dtype=dtype, device="npu")

    ci = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="npu")
    isi = torch.zeros(batch_size, dtype=torch.int32, device="npu")
    bilt = torch.ones(batch_size, dtype=torch.int32, device="npu")
    qsl = torch.arange(0, batch_size + 1, dtype=torch.int32, device="npu")

    x_ref = x.cpu()
    cs_ref = conv_state.cpu().clone()
    w_ref = weight.cpu()
    b_ref = bias.cpu()
    ci_ref = ci.cpu()
    isi_ref = isi.cpu()
    bilt_ref = bilt.cpu()
    qsl_ref = qsl.cpu()

    golden = causal_conv1d_update_v2_ref(
        x_ref, cs_ref, w_ref, bias=b_ref, activation="silu",
        conv_state_indices=ci_ref, query_start_loc=qsl_ref, max_query_len=1,
        initial_state_idx=isi_ref, block_idx_last_scheduled_token=bilt_ref,
    )

    out = causal_conv1d_update_v2(
        x, conv_state, weight, bias=bias, activation="silu",
        conv_state_indices=ci, query_start_loc=qsl, max_query_len=1,
        initial_state_idx=isi, block_idx_last_scheduled_token=bilt,
    )

    torch.testing.assert_close(out.cpu(), golden, rtol=1e-2, atol=1e-2)
    print(f"    PASS — decode APC")


def _run_prefill_varlen_test(dtype_str: str, dtype: torch.dtype) -> None:
    tilelang.cache.clear_cache()
    _prefill_kernel_cache.clear()
    torch.manual_seed(42)

    num_batches = 2
    dim = 2048
    width = 4
    state_len = width - 1
    num_cache_lines = 804
    total_tokens = 2048
    batch0_len = 662
    batch1_len = total_tokens - batch0_len

    print(f"  Prefill varlen: batch0={batch0_len}, batch1={batch1_len}, dim={dim}, dtype={dtype_str}")

    x = torch.randn(total_tokens, dim, dtype=dtype, device="npu")
    conv_state = torch.randn(num_cache_lines, dim, state_len, dtype=dtype, device="npu")
    weight = torch.randn(dim, width, dtype=dtype, device="npu")
    bias = torch.randn(dim, dtype=dtype, device="npu")

    ci = torch.arange(num_batches, dtype=torch.int32, device="npu")
    qsl = torch.tensor([0, batch0_len, total_tokens], dtype=torch.int32, device="npu")

    x_ref = x.cpu()
    cs_ref = conv_state.cpu().clone()
    w_ref = weight.cpu()
    b_ref = bias.cpu()
    ci_ref = ci.cpu()
    qsl_ref = qsl.cpu()

    golden = causal_conv1d_update_v2_ref(
        x_ref, cs_ref, w_ref, bias=b_ref, activation="silu",
        conv_state_indices=ci_ref, query_start_loc=qsl_ref, max_query_len=total_tokens,
    )

    out = causal_conv1d_update_v2(
        x, conv_state, weight, bias=bias, activation="silu",
        conv_state_indices=ci, query_start_loc=qsl, max_query_len=total_tokens,
    )

    torch.testing.assert_close(out.cpu(), golden, rtol=1e-2, atol=1e-2)
    print(f"    PASS — prefill varlen")


def _run_decode_no_bias_test(dtype_str: str, dtype: torch.dtype) -> None:
    tilelang.cache.clear_cache()
    _prefill_kernel_cache.clear()
    torch.manual_seed(42)

    batch_size = 4
    dim = 2048
    width = 4
    state_len = width - 1
    num_cache_lines = 8

    print(f"  Decode no-bias: batch={batch_size}, dim={dim}, dtype={dtype_str}")

    x = torch.randn(batch_size, dim, dtype=dtype, device="npu")
    conv_state = torch.randn(num_cache_lines, dim, state_len, dtype=dtype, device="npu")
    weight = torch.randn(dim, width, dtype=dtype, device="npu")

    ci = torch.arange(batch_size, dtype=torch.int32, device="npu")
    qsl = torch.arange(0, batch_size + 1, dtype=torch.int32, device="npu")

    x_ref = x.cpu()
    cs_ref = conv_state.cpu().clone()
    w_ref = weight.cpu()
    ci_ref = ci.cpu()
    qsl_ref = qsl.cpu()

    golden = causal_conv1d_update_v2_ref(
        x_ref, cs_ref, w_ref, bias=None, activation="silu",
        conv_state_indices=ci_ref, query_start_loc=qsl_ref, max_query_len=1,
    )

    out = causal_conv1d_update_v2(
        x, conv_state, weight, bias=None, activation="silu",
        conv_state_indices=ci, query_start_loc=qsl, max_query_len=1,
    )

    torch.testing.assert_close(out.cpu(), golden, rtol=1e-2, atol=1e-2)
    print(f"    PASS — decode no-bias")


if __name__ == "__main__":
    print("=" * 60)
    print("Causal Conv1D V2 — Decode + Varlen Prefill, APC, bias")
    print("=" * 60)

    print("\n--- Decode tests ---")
    _run_decode_test("float16", torch.float16, batch_size=1)
    _run_decode_test("float16", torch.float16, batch_size=2)
    _run_decode_test("float16", torch.float16, batch_size=4)
    _run_decode_no_bias_test("float16", torch.float16)
    _run_decode_apc_test("float16", torch.float16)

    print("\n--- Prefill varlen test ---")
    _run_prefill_varlen_test("float16", torch.float16)

    print("\n--- Bfloat16 tests ---")
    _run_decode_test("bfloat16", torch.bfloat16, batch_size=2)
    _run_prefill_varlen_test("bfloat16", torch.bfloat16)

    print("\nAll tests passed!")