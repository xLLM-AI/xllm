# mhcpresinkhorn_rmsnorm (v4-K + RmsNorm fused) design document

> A single PyPTO-Pro MIX kernel fusing 6 operators of the production chain:
> HcPreInvRms -> Cast[24,16384] -> MatMulV3 -> Cast[t,24] -> HcPreSinkhorn -> RmsNorm
> Platform: Ascend950PR (A5), CANN 9.2.0, PyPTO-Pro JIT (`import pypto_pro.language as pl`)

## 1. Operator semantics (fully fp32 compute chain)

```
r[t,1]           = rsqrt(mean(x^2 over (hc,d)) + eps)           # x [t,4,4096] bf16
mixes[t,24]      = x_2d @ hc_fn[24,16384]^T                    # BF16xBF16->FP32
res[t,24]        = mixes * r
pre[t,4]         = sigmoid(res[:,0:4]*s0 + base) + eps
post[t,4]        = sigmoid(res[:,4:8]*s1 + base) * 2
comb_flag[t,4,4] = res[:,8:24]*s2 + base
y[t,4096]        = sum_m pre[m]*x[t,m,:]                       # bf16
comb             = Sinkhorn(comb_flag, 20 alternating row/col rounds)
rstd[t,1]        = rsqrt(mean(y_bf16^2) + rms_eps)             # the fused RmsNorm
y_norm[t,4096]   = y_bf16 * rstd * gamma                       # bf16 (the y output)
```

Note: the sum of squares is taken over the y that has **already been rounded to
bf16** (matching the semantics of the production RmsNorm, which reads the bf16 y).

## 2. Overall structure (NB=28 MIX blocks = 28 AIC + 56 AIV, 1 barrier per token group)

```
AIC(28): [preload rhs (once)] - g loop: load lhs(g) -> MM(g) -> store pmix(g) -> B(g) -+
                                                                                      | B(g) = sync_all(MIX)
AIV(56): [preload aux/gamma] - g loop: ssq(g) (32 slices) --------------------------- +
                                   | after B(g)                                       |
           Stage2(g) (token-distributed, one token per AIV; runs in parallel with
           the AIC's g+1 pass):
             28-way reduction of mixes -> r -> gates (pre/post/comb_flag)
             -> y pass1 (8x512: FMA computes y + staged in UB + accumulates sum of squares)
             -> rstd -> write GM [t,1]
             -> y pass2 (8x512: y*rstd*gamma -> y_norm written to GM, double buffered)
             -> Sinkhorn (4x4, 20 rounds) -> comb/post outputs
```

- **matmul in parallel with RMSNorm**: the Cube reads bf16 x (ND->NZ) and
  hc_fn (DN->ZN) straight from GM with zero cross dependency on the Vector side
  -- this is the fundamental reason the barrier count drops from 2 to 1.
- **Cube mm(g+1) in parallel with Vector Stage2(g)**: right after the barrier
  the Cube immediately starts the next pass.

## 3. Cube side: K-axis split (scheme K)

- `kstart = core * 592`; 27 full-block cores x K=592 plus a last core with
  K=400 (`27*592+400=16384` covers it exactly with no overlap, and both are
  16-aligned fractals).
- **Two tile shapes cover everything**: full-block cores use `[16,592]/[592,32]`
  and the last core uses `[16,400]/[400,32]` (L0B totals 63.5KB <= 64KB) -- no
  tail block and no zero padding needed.
  (fillpad cannot be used for padding: Mat tiles do not support it; a
  zero-baseline load cannot be used either: the cube section has no usable
  barrier for GM->L1, so there would be a load race.)
- rhs (the per-core slice of hc_fn) does not vary with g and is preloaded once;
  each (core, g) issues a single `[16,K]x[K,32]` BF16xBF16->FP32 matmul; the 28
  partials are stored in GM as `pmix3[28,t,32]` and reduced in Stage2 with
  `pl.sum(dim=1)`.

## 4. Vector-side Stage2 (token-distributed)

### 4.1 Reduction and gating (`_gates1_vf`)

- mixes: a `[28,32]` slab load on the `pmix3f` flat view + `pl.sum` -> 24 channels.
- r: `ssq_ws [32,t]` slab + `pl.sum` -> BRC picks out the ssq of this token.
- The gating decomposition uses a **select prefix mask plus pattern vectors**
  (`scale24=[s0]x4+[s1]x4+[s2]x16`, `pmul24=[1]x4+[2]x4+[1]x16`,
  `eps24=[eps]x4+[0]x20`) to encode the three heterogeneous path semantics
  (sigmoid+eps / sigmoid*2 / linear) as purely lane-parallel vector
  multiply-adds.

### 4.2 Two-pass y stream (d-loop, DC=512, double buffered, fused RmsNorm)

| pass | VF | Contents |
|------|----|------|
| pass1 x8 | `_y_pass1_vf` | 4 rows (m) x 4 segments (128) `load_align` + `astype ZERO/ONE` even/odd unpacking + an 8-accumulator `mul_add_dst` FMA chain; pk (bf16) is **not written to GM** but kept in the yrow UB `[1,4096]`; the even/odd halves of pk are re-fetched for a squared FMA that accumulates yssq (64 lanes carried across chunks) |
| rstd | `_rstd_vf` | `reduce_sum -> muls(1/4096) -> adds(eps) -> sqrt -> div`; broadcast over 64 lanes into the rstd tile; stored to GM as `[t,1]` fp32 with 4B stores (`[1,64] tile valid [1,1]`, verified legal by a platform probe) |
| pass2 x8 | `_y_norm_vf` | Takes segments from the yrow/gamma UB (gamma is preloaded once per AIV, 8KB), computes `y*rstd*gamma`, packs ZERO/ONE and writes y_norm to GM double buffered |

- **Double buffering of the x input**: the two `[4,512]` xsub slots rotate --
  the prefetch of chunk0 overlaps with the reduction/gates;
  `load(dc)` runs in parallel with `VF(dc-1)`; the cross-iteration dependency
  is established automatically by the slot rotation plus auto_mutex.
- **What the fusion saves**: the GM write + read of y (8KB x2 per token) and
  the entire RmsNorm kernel launch.

### 4.3 Sinkhorn (`_sinkhorn_vf`)

comb_flag's 4x4 is unrolled into 16 registers and normalized over 20
alternating row/column rounds (numerically stable softmax: subtract the row
max, no eps in the denominator, eps added to the result); comb/post are written
to GM through a transposed tile.

## 5. Memory layout

- **UB (~66KB of 248KB)**: 16K ssq input, 8K yrow, 8K gamma, 2x4K xsub,
  2x1K y_norm, slab/ssq reduction buffers, aux
  (scale24/pmul24/eps24/base), sinkhorn scratch, 256B rstd; tile slots are
  spaced >=128B apart (to prevent `load_align` from over-reading across 64 lanes).
- **L1 (~94KB of 512KB)**: the two-shape source tiles for lhs/rhs.
- **L0**: L0A 31KB, L0B 63.5KB, L0C 2KB.
- mutex id in [0,31]; the five aux tiles share id 16 (sharing across tiles is
  allowed by the documentation).

## 6. Platform constraints and design rationale (measured)

| Constraint | Design impact |
|------|----------|
| The N of a ZN/Acc tile must be an integer multiple of the 16-wide inner frame (N=8 is rejected), while valid N=8 is legal | Scheme K uses rhs N=32 (24 effective); rstd uses [1,64] with valid [1,1] |
| Mat tiles do not support fillpad; the cube section has no barrier for GM->L1 | Two tile shapes cover everything (no zero baseline / no padding) |
| section_cube forbids Vec tiles | All Vec tiles live in the vector section |
| `.next()` cannot be called inside a VF | The slot rotation happens in the kernel body; the VF receives the current slot |
| Merging a pair of m rows into a single `load_align` is not feasible (there is no primitive for cross-lane summation) | y is loaded with 4 per-row loads |
| The INTRA_BLOCK event counters are globally shared (a bug) | Cross-core synchronization only uses `sync_all(MIX)` |

## 7. Version evolution and rationale

v1 pure Vector (8.04us) -> v2 Cube+Vector (11.19) -> v3 Mhc pipeline (13.52) ->
v4-K with two schemes (K 11.28 / N 13.76) -> v4-K with DC=512 + double
buffering (7.00) -> **v4-K + RmsNorm fused (7.98/10.44us, x2.78/x2.53 versus
the 6-kernel chain)**.
The figures above are historical measurements from the original design and have not
been reproduced in the current environment; they are not performance guarantees.
For inputs and outputs see [API reference](api_reference_en.md); for usage see
[example.py](../examples/example.py).