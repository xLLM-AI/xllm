---
title: "MiniMax-H3 部署与回归"
sidebar:
  order: 1
---

Ascend BF16 基线通过现有视频生成 API 提供文生音视频（T2VA）：使用
`fl2va` 管线，不传图像条件，输入预计算的 H3 文本条件，关闭稀疏注意力和
DiT 缓存。四卡采用 TP=2/SP=2，八卡采用 TP=2/SP=4。其他任务、权重格式、
精度和并行布局需要单独验证。

## 权重与文本条件

固定版本的公开模型仓库已经提供**两套格式**：根目录的 Diffusers 组件供
xLLM 使用，`FL2VA/` 中的原生组件供 SGLang 使用。直接下载发布的权重即可，
无需另写转换器。两种格式的 Q/K/V 布局和 SwiGLU `[up, gate]` 顺序不同，不能混用。

| 资产 | 固定来源 |
| --- | --- |
| 两套权重 | `MiniMaxAI/MiniMax-H3`，版本 `42ed227ee7df40d41602854ae760620d6eb651fe` |
| SGLang 参考实现 | `sgl-project/sglang`，提交 `297e54fd105e974cdf401e5b4d20253a22c16f90`（包版本 `0.5.23`） |
| 参考环境 | Python 3.11、PyTorch `2.9.0+cpu`、torch_npu `2.9.0.post2+gitbe0dd17`、Transformers `5.19.0` |

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

xLLM 根目录 `model_index.json` 的 SHA-256 为
`5a587fe13b2371427415ac892463142683aefcd8d322e274a3a095eac37ac7d2`，
`transformer/config.json` 为
`74c11bff524336576096993cbfcdcdc2ef4fa2fa4409df693bdcbc6c666282ae`。
Hugging Face 下载元数据标识固定版本的权重对象，应随权重保留。
上述模型下载与缓存均位于 `/data/models`。

先安装 FFmpeg，确保 `ffmpeg` 和 `ffprobe` 均在 `PATH` 中，
再启动运行时。在独立 SGLang 仓库中使用固定提交及其 NPU 依赖清单。
下面的无界面 OpenCV 替换与依赖约束对应已验证环境；安装时复用事先准备好的
宿主机 CANN/PyTorch/TorchNPU，不替换这套运行时：

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

# 在 xLLM 仓库根目录运行；输出目录必须是新目录。
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 "$SGLANG_PYTHON" \
  tools/generate_minimax_h3_reference.py \
  --model /data/models/MiniMax-H3-sglang/FL2VA --devices 4 \
  --prompt "$H3_PROMPT" --output reference-4
```

该工具执行真实的 50 步参考管线，生成 `reference.mp4`、`conditioning.pt`，
以及记录提示词、拓扑、包版本、模型绝对路径、模型索引与 Transformer 配置哈希、
实际运行与采样参数、该工具及被观察 SGLang 模块的源码指纹和输出哈希的 `reference.json`。
导出器支持上述固定提交的 SGLang `0.5.23`，会拒绝其他包版本；源码指纹用于区分
同版本下的本地修改。配置指纹不能替代固定的权重版本和下载元数据。
条件文件中的 BF16 `prompt_embed` 形状为 `[1, L, 5120]`，
`prompt_token_tags` 为 `[L]`，直接取自该次运行的 **TP2** 文本编码阶段。
单卡编码不能作为逐位等价的替代输入。后面的 xLLM 请求复用此文件。
TP2/SP4 时暴露八张设备，使用 `--devices 8` 和独立的 `reference-8` 输出目录。
同一批设备上切换运行时前，先结束前一个运行时。

## 构建与回归

按照 [Wan2.2 部署指南](/zh/cookbook/diffusion_models/wan/wan2_2/)
准备受支持的 Ascend 开发环境，然后执行：
如果启用 etcd，还需将 `third_party/Mooncake/mooncake-common/etcd/go.mod`
要求的 Go 工具链加入 `PATH`。

```bash
git submodule update --init --recursive
python setup.py build --device npu

export CMAKE_BUILD_DIR=$(python -c 'from scripts.build_support.utils import get_cmake_dir; print(get_cmake_dir())')
cmake --build "$CMAKE_BUILD_DIR" --target \
  minimax_h3_transformer_test minimax_h3_attention_npu_test \
  minimax_h3_video_vae_test mm_codec_test -j8
python -m unittest discover -s tests/tools -p test_compare_minimax_h3.py
"$SGLANG_PYTHON" -m unittest discover -s tests/tools -p test_generate_minimax_h3_reference.py
# 分配两张空闲设备；attention 测试程序自行启动两个 rank。
ASCEND_RT_VISIBLE_DEVICES=0,1 ctest --test-dir "$CMAKE_BUILD_DIR" \
  --output-on-failure --no-tests=error -j1 \
  -R '^(MiniMaxH3(Transformer|AttentionNpu|VideoVae)Test\.|FFmpegVideoEncoderTest\.)'
```

CPU 回归覆盖跨文件 Q/K/V 与 SwiGLU TP 加载、必需 bias、随机噪声与音频布局、
序列对齐。NPU attention 回归覆盖 63/64/65 个有效行、批次分隔、填充移除、
通信重叠开关和真实双卡 SP 通信。VAE 与 codec 测试覆盖因果尾帧，以及
MP4/H.264、AVI/MJPEG 的单帧、多帧、分数帧率和混音复编解码。
测试失败或 CTest 未选中任何测试都会报错。这些小型回归不依赖模型权重。
在 `ffmpeg` 和 `ffprobe` 可用时，媒体比较测试会运行真实媒体回归。
参考导出测试在上述参考环境中使用 CPU 张量并隔离 SGLang 模型执行边界，
不加载权重或启动 NPU 推理。
完整 50 步参考对比是独立的硬件验收任务。
不能通过私自更换子模块版本获得构建成功；依赖变更应一并提交审查。
验收时记录 SDK、库版本及正式构建配置。

## 启动服务

加载 Ascend 环境与依赖库路径后，在四张空闲设备上启动。
`XLLM_BINARY` 指向本次待验收构建产物。

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

八卡时暴露八张设备，设置 `--nnodes=8 --sp_size=4`。
模型要求 `heads % (TP * SP) == 0`、`video_patch_dim % TP == 0`、
`audio_in_channels % TP == 0`。标准权重对应 56 个头、96 个视频 patch 通道、
32 个音频通道；不合法的输出头布局会在分配 Transformer 权重前被拒绝。

## 固定基线与验收

```bash
python examples/generate_minimax_h3.py \
  --conditioning reference-4/conditioning.pt \
  --prompt "$H3_PROMPT" --output result-4.mp4
python tools/compare_minimax_h3.py \
  --actual result-4.mp4 --reference reference-4/reference.mp4 \
  --output comparison-4.json
```

示例固定 seed 42、1344×768、124 帧、24 FPS、50 步、batch 1、视频 flow shift 12、
32 kHz 双声道音频；管线使用音频 flow shift 3。
使用**相同 TP/SP 布局**的参考。对比工具需要 FFmpeg/FFprobe 和 NumPy，
会完整解码两个容器，检查 124 帧、分辨率、帧率、双声道采样率和音视频时长，
再要求 FFmpeg 视频 PSNR >= 30 dB、SSIM >= 0.95、各声道音频 SI-SDR >= 20 dB。
音视频轨道起始时间差不得超过一个音频采样周期。音频从共同起点对齐，
不搜索时移，只允许不超过一个 AAC 帧的尾部填充差异。
JSON 输出路径不能指向任一输入，符号链接和硬链接也会被拒绝。
未通过阈值时返回非零状态。这些是解码媒体的工程回归阈值，不能证明 latent
逐位一致或语义质量；还需要观看视频并试听音频。内部数值约束由前述
Transformer/attention 回归覆盖。每次验收保留参考清单、对比 JSON、xLLM 提交与
二进制哈希、构建配置、SDK 版本和 `pip freeze`。性能测试单独进行：预热一次，
至少测量三次，不要与上述检查混用。

在 TP 局部的全部序列行上执行 FP32 输出投影，是数值兼容性约束。
先选媒体行再做 GEMM 可能改变舍入与 50 步轨迹，不能仅凭短步数 smoke 测试
就删掉这部分计算。四卡与八卡使用各自参考，不要求跨布局逐位相等。
