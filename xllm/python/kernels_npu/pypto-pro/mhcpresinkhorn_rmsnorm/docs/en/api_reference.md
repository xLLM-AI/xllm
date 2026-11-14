# mhcpresinkhorn_rmsnorm (v4-K + RmsNorm fused) parameter reference

## 1. Public API

Run [example.py](../examples/example.py) from the xllm repository root to see input creation,
local-module import and output verification. Since `pypto-pro` contains a hyphen,
import the module from its own directory, as the example does:

```python
from mhcpresinkhorn_rmsnorm import mhcpresinkhorn_rmsnorm
y, rstd, post, comb = mhcpresinkhorn_rmsnorm(x, hc_fn, hc_scale, hc_base, gamma,
                                  hc_mult=4, sinkhorn_iters=20,
                                  eps=1e-6, rms_eps=1e-6)
```

### 1.1 Inputs

| Parameter | Shape / dtype | Meaning | Constraints |
|------|-----------|------|------|
| `x` | `[t, 4, 4096]` bf16 | Hyper-connection input (`t` is dynamic); inv-RMS is taken over the (hc, d) dims | Must be contiguous; provide `t >= 1` (not explicitly checked) |
| `hc_fn` | `[24, 16384]` bf16 | Gating weights (rows = 24 output channels: 4 pre + 4 post + 16 comb). **bf16** -- the fp32->bf16 weight conversion has been hoisted into the model side, so the kernel no longer casts and runs BF16xBF16->FP32 matmul directly (the products are bit-equivalent to the fp32 input and it halves GM traffic) | Must be contiguous |
| `hc_scale` | `[3]` fp32 | Scales for the three gating paths `[s_pre, s_post, s_comb]`, multiplied into the pre / post / comb_flag paths respectively | -- |
| `hc_base` | `[24]` fp32 | Biases for the three gating paths, split as `[0:4 | 4:8 | 8:24]` and added to each path | -- |
| `gamma` | `[4096]` bf16 | Scale weights of the fused RmsNorm (per-d element) | Must be contiguous |
| `hc_mult` | int = 4 | Hyper-connection multiplier hc (fixed to 4 by contract) | Raises if != 4 |
| `sinkhorn_iters` | int = 20 | Number of Sinkhorn row/column alternating normalization rounds (fixed to 20 by contract) | Raises if != 20 |
| `eps` | float = 1e-6 | Numerical-stability constant of the gating paths (added to `pre` and to the Sinkhorn denominators/results, matching golden) | Raises if != 1e-6 |
| `rms_eps` | float = 1e-6 | eps of the fused RmsNorm: `rstd = 1/sqrt(mean(y^2) + rms_eps)` | -- |

### 1.2 Outputs

| Return value | Shape / dtype | Meaning |
|--------|-----------|------|
| `y` | `[t, 4096]` bf16 | **y_norm** (note: this is the post-RmsNorm value `y * rstd * gamma`; the raw bf16 `y` is no longer returned -- in the production chain it is only consumed by RmsNorm) |
| `rstd` | `[t, 1]` fp32 | Reciprocal standard deviation of RmsNorm (kept for the backward pass) |
| `post` | `[t, 4]` fp32 | post gate = `sigmoid(z) * 2` |
| `comb` | `[t, 4, 4]` fp32 | Combination matrix after Sinkhorn normalization |

## 2. Internal contract constants (mhcpresinkhorn_rmsnorm.py)

| Constant | Value | Meaning |
|------|----|------|
| `HC` | 4 | Hyper-connection multiplier hc (dim 1 of `x`) |
| `D` | 4096 | Hidden dim d (dim 2 of `x` / width of `y` / length of `gamma`) |
| `MIX_HC` | 24 | Number of gating output channels `(2+hc)*hc` = 4 pre + 4 post + 16 comb |
| `HC_D` | 16384 | K dim of the matmul, `hc*d` |
| `SINKHORN_ITERS` | 20 | Sinkhorn rounds (1 column-normalization round + 19 row/column rounds) |
| `EPS` | 1e-6 | eps of the gating paths |
| `RMS_EPS` | 1e-6 | eps of the fused RmsNorm |
| `NB` | 28 | Number of MIX logical blocks (28 AIC + 56 AIV, the platform limit; launched as `kernel[None, 28]`) |
| `TG` | 16 | Tokens per token group (granularity of the M dim of the cube matmul) |
| `MIXC` | 32 | Tile width of the gating output (24 -> padded to 32 for 16-wide inner-frame alignment) |
| `DC` | 512 | Chunk width of the y d-loop (tile width of `x_sub`/`y_ub`; 4096/512 = 8 iterations) |
| `NDC` | 8 | Iteration count of the y d-loop, `D/DC` |
| `SSQ_SLICES` / `SSQ_DK` | 32 / 512 | Slicing scheme of the RMSNorm ssq: 32 AIV slices x 512 wide, reduced in Stage 2 |
| `KA` | 592 | K-slice width of the full-block K-split cores (`kstart = core*592`; aligned to 37x16 fractals) |
| `KLAST` | 400 | K-slice width of the last core (core 27), `16384 - 27*592` (aligned to 25x16) |

### K-split coverage

`kstart = core * 592`: cores 0-26 take 592 each and core 27 takes 400, which
covers exactly 16384 with no overlap; the load imbalance is only
592/585 ~= 1.2%.

## 3. Direct kernel signature (advanced usage)

```python
mhcpresinkhorn_rmsnorm_kernel_k[None, 28](
    x_2d,      # [t, 16384] bf16 -- 2D view of x (x.reshape(t, HC_D))
    x_3d,      # [t, 4, 4096] bf16 -- x itself (the x slices of y are loaded as 3D)
    hc_fn,     # [24, 16384] bf16
    hc_scale,  # [1, 3] fp32 view
    hc_base,   # [1, 24] fp32 view
    gamma,     # [1, 4096] bf16 view
    y,         # [t, 4096] bf16 (out: y_norm)
    rstd,      # [t, 1] fp32 (out)
    post,      # [t, 4] fp32 (out)
    comb16,    # [t, 16] fp32 (out: flattened view of comb, comb.view(t, 16))
    pmix3,     # workspace [28, ceil(t/16)*16, 32] fp32 (K-split partial)
    pmix3f,    # 2D flat view of the pmix3 workspace, [28, .*32]
    ssq_ws)    # workspace [32, ceil(t/16)*16] fp32 (32-slice ssq partial)
```

Note that `x_2d`/`x_3d` are two views of the same `x` (same GM data); the
workspaces must be allocated with `ceil(t/TG)*TG` alignment (the wrapper's
`_get_workspaces` already implements cached allocation).

## 4. Runtime environment variables

| Variable | Description |
|------|------|
| `TILE_FWK_DEVICE_ID` | Target NPU device id (use an idle device; measurements on a shared device get polluted by preemption) |
| `PTO_TILE_LIB_CODE_PATH` | Path to the local `pto-isa` checkout (PyPTO-Pro compilation dependency; not bundled in this directory) |

The original development environment used CANN 9.2.0 and loaded its environment
with `source /usr/local/Ascend/ascend-toolkit/set_env.sh`. Use the setup script
for the CANN installation on your device; install compatible PyTorch,
`torch_npu`, and PyPTO-Pro versions before running the JIT example.