# mhcpresinkhorn_rmsnorm

A PyPTO-Pro MIX kernel for Ascend 950PR (A5), fusing hyper-connection inverse RMS, gating matmul, pre/post gates, Sinkhorn normalization, and RmsNorm. The K dimension is split across 28 Cube cores; Vector processing overlaps the matmul and computes the final normalized output.

[中文说明](README_zh.md)

## Requirements

- Ascend 950PR (A5) NPU with a compatible CANN environment (the original development environment used CANN 9.2.0).
- Python with compatible PyTorch, `torch_npu`, and PyPTO-Pro installed.
- `PTO_TILE_LIB_CODE_PATH` pointing to a local `pto-isa` checkout for PyPTO-Pro compilation; it is not included here.
- Optionally set `TILE_FWK_DEVICE_ID` to the target NPU index (default: `0`).

Load the CANN environment for your installation before running the example. The first call triggers JIT compilation. This directory does not provide CPU execution or a precompiled kernel.

## Public API

The entry point is `mhcpresinkhorn_rmsnorm(x, hc_fn, hc_scale, hc_base, gamma, hc_mult=4, sinkhorn_iters=20, eps=1e-6, rms_eps=1e-6)`.

| Input | Shape | dtype |
| --- | --- | --- |
| `x` | `[t,4,4096]` | bf16 |
| `hc_fn` | `[24,16384]` | bf16 |
| `hc_scale` | `[3]` | fp32 |
| `hc_base` | `[24]` | fp32 |
| `gamma` | `[4096]` | bf16 |

Provide a nonempty token dimension (`t >= 1`). The implementation requires contiguous `x`, `hc_fn`, and `gamma` and only accepts `hc_mult=4`, `sinkhorn_iters=20`, and `eps=1e-6`. `rms_eps` defaults to `1e-6`. The wrapper selects the NPU via `TILE_FWK_DEVICE_ID` and moves inputs to it.

Returns `(y, rstd, post, comb)` with shapes and dtypes `[t,4096]` bf16, `[t,1]` fp32, `[t,4]` fp32, and `[t,4,4]` fp32, respectively. **`y` is already RmsNorm-normalized**; the intermediate unnormalized bf16 value is not returned.

## Run the example

From the xllm repository root, in a configured Ascend environment:

```bash
TILE_FWK_DEVICE_ID=0 PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa \
  python3 xllm/python/kernels_npu/pypto-pro/mhcpresinkhorn_rmsnorm/examples/example.py 3
```

The optional trailing argument is `t` (default `3`). The example constructs sample inputs and compares all four outputs against a PyTorch reference; it requires a working NPU environment.

## Documentation

- [English design](docs/design_en.md) / [中文设计](docs/design_zh.md)
- [English API reference](docs/api_reference_en.md) / [中文接口说明](docs/api_reference_zh.md)
- [Runnable example](examples/example.py)