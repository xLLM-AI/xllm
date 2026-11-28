# OPT-01: ordinary QKV, QK RMSNorm and mRoPE fusion

## Submission validation scope

The submission is rebased onto upstream
`2339aba64eaf98b07b77b143875cb72385eee76f`. Upstream migrated the
unfused RMSNorm API to `xllm_kernel.ops.normalization.rms_norm`; the reference tests and
offline validation hooks use that same current API.

The current baseline also returns a `MultimodalEmbeddings` object from the
multimodal embedding interface. This change preserves that upstream interface;
the fusion does not reinstate the older tuple return. Revalidation requires
native artifacts built from the current source, as well as the updated Python
files. The legacy RMSNorm registration still exists; its continued availability
does not establish compatibility of an older native build with this interface.

The measured NPU results below were collected on the earlier
`8da1fc5a3937c4314206354bdc98ecd930921735` baseline. They are historical
experiment evidence, **not a completed NPU validation of the rebased code**.
The updated reference API needs fresh NPU numerical, graph and full-model
validation before this draft is ready for review. Tolerances remain zero;
they have not been relaxed to accommodate the migration.

Machine-readable measurement summaries and the original source fingerprints
are included in `evidence/`. They contain no SSH credentials or model weights.

Qwen3-VL's three-axis attention branch applies Q RMSNorm, K RMSNorm and mRoPE
separately in every decoder layer. This change fuses those operations with
ordinary packed-QKV splitting. A model-level operation gathers the shared
position table once per forward; each layer retains its own QKV and norm weights.

The reference model source is xLLM commit
`8da1fc5a3937c4314206354bdc98ecd930921735`. The implementation is opt-in through
`XLLM_OPT01_FUSED_MROPE=1` on the server process. Use `0` for the baseline.

## Supported path

The model selects the fused path during initialization for NPU BF16,
`head_dim=128`, sections `[24, 20, 20]`, and local Q/KV head counts
`(32,8)`, `(16,4)`, `(8,2)` or `(4,1)`. It preserves the ordinary branch's
interleaved cache contract, NeoX half rotation and BF16 norm-output rounding.
An explicitly non-interleaved rotary dictionary selects the existing branch.
Flat ModelArgs uses the ordinary attention branch's existing interleaved contract.
Other configurations select the existing branch before execution.

The one-dimensional rotary branch and Qwen3.5's gated QKVG implementation retain
their existing behavior. Execution failures propagate; there is no exception retry.

The internal tensor interfaces are:

| Interface | Inputs | Outputs |
| --- | --- | --- |
| `prepare_mrope` | BF16 cache `[max_pos,128]`, contiguous positions `[3,N]`, persistent axis table `[64]` | Shared BF16 table `[N,128]` |
| `split_qkv_norm_mrope` | Packed QKV `[N,(Hq+2Hkv)*128]`, Q/K weights `[128]`, shared table, local head counts, epsilon | Contiguous Q `[N,Hq*128]` and K/V `[N,Hkv*128]` |
| `build_axis_pattern` | Three sections, interleaving flag and NPU device | Initialization-time axis table |

Both operations register fake implementations. The position count is a runtime
scalar in the table kernel, so different prompt lengths do not require a separate
constexpr specialization for every length.

## Environment

The measured model directories were `/home/models/Qwen3-VL-8B-Instruct`
(optimization target, MMBench and GSM8K) and `/home/models/Qwen3-8B`
(unchanged ordinary-RoPE regression path, GSM8K). The supplied
`/home/models/Qwen3-30B-A3B` is outside this Qwen3-VL optimization's evaluated
scope; no full-model MoE accuracy or performance result is claimed.

Use a working NPU xLLM build containing the patched Python files and Triton.
Source the CANN/ATB environment required by that build. The server and validation
scripts must import the same xLLM installation.

The evaluated image uses Python 3.12, PyTorch/torch_npu 2.10, CANN 9.2.1 and
Triton 3.6.0. The experiment's private binary-loader compatibility changes are
recorded separately from this optimization.

Install evaluation tools into an isolated environment. The tested combination is
EvalScope 1.8.1, NumPy 1.26.4 and PyArrow 19.0.1. Fixture generation also requires
Pillow, NumPy and PyAV.

```bash
python -m pip install evalscope==1.8.1 numpy==1.26.4 pyarrow==19.0.1 pillow av
```

## Start each server variant

Keep the visible device and all serving options the same for the performance
comparison. Restart the server between A, B and returning A.

For this launcher's actual two-device TP test, use
`ASCEND_RT_VISIBLE_DEVICES=0,1` **and** `--nnodes=2` in both variants. A two-device
visibility mask alone leaves the default one-worker configuration unchanged.
Check both worker logs for `tp_world_size=2`, and the candidate logs for the
OPT-01 enabled message. Release all workers before restarting the next variant;
the primary HTTP endpoint closing does not prove every rank has released HBM.

```bash
ASCEND_RT_VISIBLE_DEVICES=0 XLLM_OPT01_FUSED_MROPE=0 xllm serve \
  --model=/home/models/Qwen3-VL-8B-Instruct \
  --model_id=Qwen3-VL-8B-Instruct --model_impl=python --npu_kernel_backend=TORCH \
  --enable_graph=true --python_graph_backend=aclgraph --host=127.0.0.1 --port=18001 \
  --master_node_addr=127.0.0.1:9750 --nnodes=1 --block_size=128 \
  --max_body_size=536870912 --max_memory_utilization=0.85 \
  --enable_prefix_cache=false --max_encoder_cache_size=0 --max_processor_cache_items=0 \
  --enable_chunked_prefill=false --enable_schedule_overlap=false --enable_shm=false \
  --max_seqs_per_batch=32 --max_tokens_per_batch=8192 \
  --max_tokens_for_graph_mode=32 --acl_graph_decode_batch_size_limit=32
```

Set `XLLM_OPT01_FUSED_MROPE=1` for B. The 512 MiB RPC limit permits full image
batches that exceed brpc's 64 MiB default; it is identical for both variants.
The prefill token budget can schedule a wave of 32 requests in several batches.
Concurrency 32 therefore describes client concurrency, not a guarantee that all
32 prompts execute in one prefill batch.

## Numerical and graph checks

Run these on NPU from a fresh output directory to preserve earlier results:

```bash
python /path/to/xllm/tests/python/test_opt01_mrope_kernel.py
python /path/to/xllm/tests/python/test_opt01_mrope_dispatch.py
# These also expose test functions to the repository's pytest/CMake discovery:
python -m pytest -q /path/to/xllm/tests/python/test_opt01_mrope_kernel.py \
  /path/to/xllm/tests/python/test_opt01_mrope_dispatch.py
```

The kernel test compares actual unfused xLLM RMSNorm/mRoPE operators with fused
outputs for token counts 17, 65, 257 and 2048 and all listed local head counts.
It captures an NPU graph, changes QKV and three-axis positions, and checks both
replays against freshly computed references. It also checks fake output shapes
and dtypes. The dispatch test covers supported shapes and unsupported dtype,
device, head size, head counts, sections and explicit layout.

The validated BF16 specialization is bitwise equal to the native reference on
the tested stack, so the current kernel and boundary tests enforce `atol=rtol=0`.
The boundary test explicitly covers position 0 and 262143 (the actual model's
table limit), different T/H/W combinations, TP-local head counts, the legal
`[16,24,24]` native section fallback and the FP16 fallback. It compares the Q/K/V
produced by the actual attention preprocessing branch. Selecting the legacy
branch retains that backend's support limits; it does not make arbitrary
section/head configurations valid.

```bash
python -m pytest -q tests/python/test_opt01_mrope_kernel.py \
  tests/python/test_opt01_mrope_dispatch.py tests/python/test_opt01_mrope_boundaries.py
```

## Additional offline Attention and memory instrumentation

`opt01_validation_hooks.install(model)` attaches validation hooks to an already
constructed `Qwen3VLModel`. It is deliberately outside the production forward
path. The offline experiment launcher temporarily invokes it at the end of model
initialization and restores the source afterwards. Set `XLLM_OPT01_TEST_MODE` to
`attention` or `measure`, and `XLLM_OPT01_TEST_DIR` to a fresh directory. Put this
benchmark directory on `PYTHONPATH` for the offline launcher.

Attention mode recomputes the real unfused Q/K/V and calls the same Attention
backend with the same context and KV slots. It clones the fused Attention output
before the reference call because a backend may reuse its output workspace. It
enforces exact Q/K/V and Attention equality and passes the saved fused result to
the next layer, never substituting the reference result.

Measurement mode records NPU events around **decoder model forward**, excluding
vision encoding, the LM head and request queuing. It also records synchronized
Host duration. These are independent prefill measurements, not estimates from
TTFT or sums of individual operators. Measurement adds explicit synchronization;
keep it separate from the uninstrumented end-to-end performance series.
The synchronized Host wall time is auxiliary and can include waiting for work
already queued before the call; use the in-forward NPU event span for the
independent decoder-prefill comparison.

The memory sampler acknowledges a reset while the service is idle before the
client sends each wave. Exact Torch allocator peaks therefore cover the complete
wave, including vision, prefill and 2,048-token decode. Device HBM usage is also
sampled every 50 ms; its peak is a sampled lower bound, not an exact allocator
high-water mark. Report allocated and reserved peaks separately, including any
increase caused by the shared table.

```bash
python benchmarks/opt01/bench_2k2k.py --rounds 3 \
  --image benchmarks/opt01/fixtures/red_448.png \
  --instrumentation-dir /path/to/fresh/worker-measurements \
  --output results/instrumented_a_image.jsonl
```

## Single image, multiple images, video and mixed lengths

```bash
python benchmarks/opt01/make_fixtures.py
mkdir -p results
python benchmarks/opt01/vlm_workload.py --rounds 5 --output results/functional_a.jsonl
```

Repeat against B into `functional_b.jsonl`. Compare request hashes and outputs,
including the 24-token warmup responses. These deterministic fixtures validate
integration and complement the complete accuracy datasets below.

## Complete accuracy evaluation

Run without `--smoke` for formal evaluation:

```bash
python benchmarks/opt01/run_accuracy.py --variant a --dataset gsm8k
python benchmarks/opt01/run_accuracy.py --variant a --dataset mm_bench
```

Repeat with `--variant b` while the fused server is running. GSM8K uses the
default 4-shot test split, 1,319 examples. MMBench uses the complete Chinese and
English dev splits, 4,329 examples each. Temperature is 0, top-p is 1, the output
limit is 2,048, and evaluation concurrency is 32. Both variants use the same
parameters and cached official datasets. Scoring uses EvalScope's rule graders.
Outputs go to `accuracy/<variant>/<dataset>`; existing directories are rejected
to prevent mixing old and new runs.

The two-example `--smoke` option is a connection check and does not satisfy the
complete-dataset acceptance requirement. `--api-url` can select a second local
server for parallel precision evaluation on an equivalent device. Record that
device assignment explicitly; conduct performance runs in isolation.

For the dense Qwen3-8B regression, serve `/home/models/Qwen3-8B` with the matching
model ID, then add `--model Qwen3-8B` to the evaluator. This runner supplies
`chat_template_kwargs.enable_thinking=false` for that model, consistently in A/B.

## Fixed 2k input + 2k output performance

```bash
python benchmarks/opt01/bench_2k2k.py --rounds 3 --output results/perf_a_text.jsonl
python benchmarks/opt01/bench_2k2k.py --rounds 3 \
  --image benchmarks/opt01/fixtures/red_448.png --output results/perf_a_image.jsonl
```

Each run covers concurrency 1 and 32. Calibration queries use the server's
reported token counts to construct exactly 2,048 input tokens, including image
and template tokens. `ignore_eos=true` requests exactly 2,048 output tokens.
The script rejects any measured response with different counts.

Each concurrency has one full-length warmup wave and three measured waves.
JSONL records include request hashes, usage, first-text latency, total latency,
TPOT and output-token throughput. TPOT is `(total_latency - TTFT)/(output_tokens - 1)`.
Report all A/B/returning-A measurements; the first-token improvement and the
long-decode throughput improvement are separate measurements.

## Profile evidence

Enable online Ascend profiling on the server with `--enable_online_profile=true`,
`--profile_backend=ascend` and a unique `--profile_dir` for each variant. Then run:

```bash
python benchmarks/opt01/vlm_workload.py --case long --capture --output results/profile_a.jsonl
msprof --export=on --output=/path/to/profile/xllm_npu_DIRECTORY
```

Use the same request for B and compare exported operator/API CSVs. Verify that
the ordinary decoder's separate Q/K norm and mRoPE work is replaced by one fused
call per layer, with one shared table preparation per forward. Keep encoder
operations separate when interpreting the counts. Record TTFT alongside these
operator changes rather than inferring end-to-end speed from kernel counts alone.

## Source references

- [xLLM ordinary mRoPE branch](https://github.com/xLLM-AI/xllm/blob/58fce9835d047586b92ed026ae677e634fa7f1a2/xllm/python/models/qwen3.py#L190)
- [vLLM-Ascend fusion entry](https://github.com/vllm-project/vllm-ascend/blob/a1fcee98601b8da223d20afc46b7ba3c6c7da393/vllm_ascend/patch/worker/patch_qwen3vl.py#L30)
- [SGLang first-layer position preparation](https://github.com/sgl-project/sglang/blob/77c3bcbe2c28e1a6d1ad4121ee57231582d39bdc/python/sglang/srt/models/qwen3.py#L354)
- [xLLM Qwen3.5 specialized gated path](https://github.com/xLLM-AI/xllm/blob/58fce9835d047586b92ed026ae677e634fa7f1a2/xllm/python/layers/npu/qwen3_5/attention.py#L49)
- [EvalScope GSM8K](https://evalscope.readthedocs.io/en/v1.8.1/benchmarks/gsm8k.html)
- [EvalScope MMBench](https://evalscope.readthedocs.io/en/v1.8.1/benchmarks/mm_bench.html)
