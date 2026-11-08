---
title: "MiniMax-H3 Deployment and Regression"
sidebar:
  order: 1
---

The Ascend BF16 baseline serves text-to-video-and-audio (T2VA) through the
existing video generation API. It uses the `fl2va` pipeline without image
conditions, precomputed H3 text conditioning, dense attention, and no DiT cache.
Four NPUs use TP=2/SP=2; eight NPUs use TP=2/SP=4. Other tasks, checkpoint
formats, dtypes, and parallel layouts require separate validation.

## Checkpoint and conditioning

The pinned public model repository already contains **both** layouts: the
root Diffusers components used by xLLM and the native `FL2VA/` components used
by SGLang. Download these published artifacts rather than writing a converter.
The formats are not interchangeable: Q/K/V layout and SwiGLU `[up, gate]`
ordering differ.

| Asset | Pinned source |
| --- | --- |
| Both checkpoints | `MiniMaxAI/MiniMax-H3`, revision `42ed227ee7df40d41602854ae760620d6eb651fe` |
| SGLang reference implementation | `sgl-project/sglang`, commit `297e54fd105e974cdf401e5b4d20253a22c16f90` (package `0.5.23`) |
| Reference environment | Python 3.11, PyTorch `2.9.0+cpu`, torch_npu `2.9.0.post2+gitbe0dd17`, Transformers `5.19.0` |

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/models/.cache/huggingface
export H3_REVISION=42ed227ee7df40d41602854ae760620d6eb651fe
uvx --from huggingface_hub==1.33.0 hf download MiniMaxAI/MiniMax-H3 --revision "$H3_REVISION" \
  --local-dir /data/models/MiniMax-H3 \
  --include model_index.json --include 'transformer/*' \
  --include 'transformer_ref/*' --include 'vae/*' --include 'audio_vae/*' \
  --include 'scheduler/*' --include 'audio_scheduler/*' --include 'processor/*' \
  --include 'text_encoder/*' --include 'tokenizer/*'
uvx --from huggingface_hub==1.33.0 hf download MiniMaxAI/MiniMax-H3 --revision "$H3_REVISION" \
  --local-dir /data/models/MiniMax-H3-sglang --include 'FL2VA/*'
```

The xLLM root `model_index.json` SHA-256 is
`5a587fe13b2371427415ac892463142683aefcd8d322e274a3a095eac37ac7d2`;
`transformer/config.json` is
`74c11bff524336576096993cbfcdcdc2ef4fa2fa4409df693bdcbc6c666282ae`.
Hugging Face download metadata identifies the pinned weight objects; keep it
with the checkpoint. All model downloads and caches above stay in `/data/models`.

Install FFmpeg and ensure both `ffmpeg` and `ffprobe` are on `PATH` before
starting either runtime. In a separate SGLang checkout, use the pinned commit
and its NPU dependency manifest. The headless OpenCV substitution and
constraints below match the verified environment; reuse the prepared host
CANN/PyTorch/TorchNPU stack rather than replacing it during installation:

```bash
git clone https://github.com/sgl-project/sglang.git /home/lhp/sglang
git -C /home/lhp/sglang checkout 297e54fd105e974cdf401e5b4d20253a22c16f90
cp /home/lhp/sglang/python/pyproject_npu.toml /home/lhp/sglang/python/pyproject.toml
sed -i 's/opencv-python==/opencv-python-headless==/' /home/lhp/sglang/python/pyproject.toml
uv venv --system-site-packages --python python3.11 /home/lhp/sglang/.venv
cat > /home/lhp/sglang/.venv/h3-constraints.txt <<'EOF'
compressed-tensors==0.15.0
numpy==1.26.4
scipy==1.13.1
attrs==24.2.0
decorator==5.1.1
psutil==6.0.0
pandas==2.2.3
EOF
cat > /home/lhp/sglang/.venv/h3-excludes.txt <<'EOF'
torch
torchvision
torch-npu
triton
triton-ascend
EOF
uv pip install --python /home/lhp/sglang/.venv/bin/python \
  --constraints /home/lhp/sglang/.venv/h3-constraints.txt \
  --excludes /home/lhp/sglang/.venv/h3-excludes.txt \
  -e '/home/lhp/sglang/python[all_npu]'
export SGLANG_PYTHON=/home/lhp/sglang/.venv/bin/python
export OMP_NUM_THREADS=8
export HCCL_BUFFSIZE=256
export H3_PROMPT='At night, while their owner sleeps in a bedroom, three cats march in loudly playing tiny brass instruments, then abruptly file out.'

# Run from the xLLM checkout. This creates a new output directory.
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 "$SGLANG_PYTHON" \
  tools/generate_minimax_h3_reference.py \
  --model /data/models/MiniMax-H3-sglang/FL2VA --devices 4 \
  --prompt "$H3_PROMPT" --output reference-4
```

The tool runs the real 50-step pipeline and writes `reference.mp4`,
`conditioning.pt`, and `reference.json` with prompt, topology, package versions,
the resolved model path, model-index/Transformer-config SHA-256 digests, the
actual runtime and sampling arguments, source fingerprints for this tool and
the observed SGLang modules, and output digests. The exporter supports SGLang
`0.5.23` at the pinned commit above and rejects other package versions; source
fingerprints distinguish local changes within that version. Config fingerprints
do not replace the pinned weight revision and download metadata.
The conditioning contains `[1, L, 5120]` BF16
`prompt_embed` and `[L]` `prompt_token_tags`, captured from that run's **TP2**
text stage. A single-device encoding is not a bitwise substitute. Reuse this
file in the xLLM request below. For TP2/SP4, use eight visible devices,
`--devices 8`, and a separate `reference-8` output directory. Shut down each
runtime before starting the other on the same devices.

## Build and regression tests

Build in the supported Ascend development environment described in the
[Wan2.2 recipe](/en/cookbook/diffusion_models/wan/wan2_2/):
With etcd enabled, also put the Go toolchain required by
`third_party/Mooncake/mooncake-common/etcd/go.mod` on `PATH`.

```bash
git submodule update --init --recursive
python setup.py build --device npu

export CMAKE_BUILD_DIR=$(python -c 'from scripts.build_support.utils import get_cmake_dir; print(get_cmake_dir())')
cmake --build "$CMAKE_BUILD_DIR" --target \
  minimax_h3_transformer_test minimax_h3_attention_npu_test \
  minimax_h3_video_vae_test mm_codec_test -j8
python -m unittest discover -s tests/tools -p test_compare_minimax_h3.py
"$SGLANG_PYTHON" -m unittest discover -s tests/tools -p test_generate_minimax_h3_reference.py
# Assign two idle devices; the attention executable starts both ranks.
ASCEND_RT_VISIBLE_DEVICES=0,1 ctest --test-dir "$CMAKE_BUILD_DIR" \
  --output-on-failure --no-tests=error -j1 \
  -R '^(MiniMaxH3(Transformer|AttentionNpu|VideoVae)Test\.|FFmpegVideoEncoderTest\.)'
```

The CPU regressions cover streamed Q/K/V and SwiGLU TP loading, required biases,
RNG/audio packing, and sequence alignment. The NPU attention regression covers
63/64/65 live rows, batch segmentation, padding removal, both communication
overlap settings, and real two-rank SP collectives. VAE and codec tests cover
the causal tail and MP4/H.264 and AVI/MJPEG round trips, including single-frame,
fractional-FPS, and audio-muxed outputs. A failed test or an empty CTest
selection causes a nonzero exit. These small regressions need no model weights.
The media-comparison tests run with real FFmpeg when
`ffmpeg` and `ffprobe` are available. The reference-export tests use CPU tensors
and isolate the SGLang model-execution boundary. Run them in the reference
environment above; they do not load weights or launch NPU inference.
The full 50-step reference comparison is
a separate hardware acceptance run.

Do not update submodule revisions to obtain a passing build without including
and reviewing those dependency changes. Keep SDK/library versions and the
production build configuration with acceptance results.

## Start the service

After loading the Ascend environment and dependency paths, start on four idle
devices. Set `XLLM_BINARY` to the executable from the build being validated.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MODEL_PATH=/data/models/MiniMax-H3
export XLLM_BINARY=/path/to/xllm
python -m xllm.launch_server serve --binary-path="$XLLM_BINARY" \
  --nnodes=4 --backend=dit --model="$MODEL_PATH" --model_id=minimax-h3 \
  --tp_size=2 --sp_size=2 --cfg_size=1 --vae_size=1 \
  --text_encoder_tp_size=1 --dit_h3_task_type=fl2va --dit_cache_policy=None \
  --dit_laser_attention_enabled=false --dit_sparse_attention_enabled=false \
  --master_node_addr=127.0.0.1:19885 --host=127.0.0.1 --port=8000 \
  --max_body_size=2147483648 --logtostderr=true
```

For eight devices, expose all eight, set `--nnodes=8`, and use `--sp_size=4`.
The model requires `heads % (TP * SP) == 0`, `video_patch_dim % TP == 0`,
and `audio_in_channels % TP == 0`; the standard checkpoint has 56 heads,
96 video patch channels, and 32 audio channels. Invalid output-head layouts
are rejected before allocating transformer weights.

## Run and assess the fixed baseline

```bash
python examples/generate_minimax_h3.py \
  --conditioning reference-4/conditioning.pt \
  --prompt "$H3_PROMPT" --output result-4.mp4
python tools/compare_minimax_h3.py \
  --actual result-4.mp4 --reference reference-4/reference.mp4 \
  --output comparison-4.json
```

The example fixes seed 42, 1344×768, 124 frames, 24 FPS, 50 steps, batch one,
video flow shift 12, and 32 kHz stereo audio. The pipeline uses audio flow
shift 3. Compare with the reference at the **same TP/SP layout**. The comparison
tool requires FFmpeg/FFprobe and NumPy; it fully decodes both containers, checks
124 frames, geometry, FPS, stereo sample rate and A/V duration, then requires
FFmpeg video PSNR >= 30 dB, SSIM >= 0.95 and audio SI-SDR >= 20 dB per channel.
Audio/video track start times must agree within one audio sample. Audio uses
the common origin without lag search, allowing at most one AAC frame of
trailing padding. The JSON output path must not identify either input, including
through symbolic or hard links. A failed gate exits nonzero. These decoded-media thresholds
are engineering regressions, not proof of latent equality or semantic quality;
listen to the audio and inspect the video too. Internal numerical regressions
are covered by the transformer/attention tests above. Keep the reference
manifest, comparison JSON, xLLM revision and binary hash, build configuration,
SDK versions, and `pip freeze` with each acceptance result. For performance,
warm up once and measure at least three requests separately from this check.

The full-row, TP-local FP32 output projections are a numerical compatibility
constraint: selecting media rows before GEMM can change rounding and the
50-step trajectory. Do not remove this work based only on a short smoke test.
Cross-layout outputs are not required to be bitwise equal; keep separate
four-device and eight-device references.
