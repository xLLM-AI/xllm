## Mandatory benchmark capture

**Never report only summaries. Generate a raw, per-operation benchmark table for every run.**

Capture:
- Operation/kernel/shader name
- Invocation count
- Total and average device time
- Host time
- Data volume, bandwidth and throughput where measurable
- Dispatch dimensions, workload shape and execution configuration
- GPU utilisation, memory usage and available hardware counters
- End-to-end latency, throughput and unattributed time

Example:

KRK_TIME=1 ./build-hip/kraken.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --bench --ctx 2048 2>&1 | sed -n '/KRK_TIME/,$p' | head -45

stdout:
== KRK_TIME per-op device time (138576 ops)
op                            calls     dev ms   host ms        MB     GB/s  TFLOP/s
gemm(gemv)                    43096    1256.67     88.00 123597.19     98.4     0.28
rmsnorm                       17152     367.45     34.13      0.00      0.0     0.00
attention                      1536     292.92      3.47      0.00      0.0     0.00
silu_mul                      10752     241.88     19.95      0.00      0.0     0.00
add_inplace                   12288     154.57     22.42      0.00      0.0     0.00
l2norm                         9216     151.79     16.76      0.00      0.0     0.00
download_f32                    256     149.88    170.38      0.00      0.0     0.00
conv1d_silu                    4608     129.19     10.39      0.00      0.0     0.00
sigmoid_act                    6144     111.72     11.55      0.00      0.0     0.00
gemm_group                     2040     111.39      4.19      0.00      0.0     0.00
softplus_act                   4608     111.31      8.52      0.00      0.0     0.00
scale_act                      4608     109.55      8.62      0.00      0.0     0.00
delta_rule                     4608      91.45      9.82      0.00      0.0     0.00
upload                          320      81.65    174.79      0.00      0.0     0.00
add_bias_cols                  4608      80.27      9.29      0.00      0.0     0.00
scale_cols                     4608      71.45      8.93      0.00      0.0     0.00
rope                           1536      58.38    898.06      0.00      0.0     0.00
embed                           256      50.54     49.81      0.00      0.0     0.00
qk_norm                        1536      47.90      4.24      0.00      0.0     0.00
kv_append                      1536      30.12      3.33      0.00      0.0     0.00
q3_split                       1536      29.24      3.16      0.00      0.0     0.00
gemm(wmma)                      186       9.21      0.62    305.96     33.2     5.94
mul_act                        1536       1.81      3.00      0.00      0.0     0.00
fill0                             6       0.31      0.07      0.00      0.0     0.00
TOTAL                        138576    3740.64                                     
device spans are 54.5% of 6858.5 ms wall (3117.9 ms unattributed)
KRAKEN 0.1.0 | AMD Radeon RX 9070 XT (gfx1201, RDNA4, 64 CU, wave32, WMMA gfx12)

model          Qwen3.5 0.8B
device         AMD Radeon RX 9070 XT
prefill        55 tokens in 35.3 ms  (1557.9 tok/s)
decode         256 tokens in 6504.7 ms  (39.4 tok/s)
engine split   pid 3832, per-process GPU engine utilization (%), sampled every 50 ms:
  Compute 0            30.61 % avg    33.16 % max  (n=108)
  Copy                  0.36 % avg     1.04 % max  (n=218)
  3D                    0.00 % avg     0.00 % max  (n=109)
  Security 1            0.00 % avg     0.00 % max  (n=109)
  Timer 0               0.00 % avg     0.00 % max  (n=109)
  Video Codec Engine    0.00 % avg     0.00 % max  (n=109)
  Video JPEG 0          0.00 % avg     0.00 % max  (n=109)

**Mandatory evidence outputs:**
1. Raw per-operation timing table.
2. Per-kernel or per-shader sample details, including individual invocations or representative latency distributions.
3. Hardware and execution trace.
4. Ranked bottleneck analysis.
5. Baseline versus optimised comparison.

**Sampling rules:**
- Collect individual timings when practical.
- Otherwise report min, median, p95, p99, max and sample count.
- Group samples by kernel variant, workload shape and execution path.
- Do not invent unavailable metrics. Mark them `N/A` and explain the instrumentation needed to obtain them.
- Preserve raw logs and machine-readable results, preferably CSV or JSON.

**Enforcement:** If detailed measurements are unavailable, create instrumentation traps or use suitable profilers before declaring the investigation complete.