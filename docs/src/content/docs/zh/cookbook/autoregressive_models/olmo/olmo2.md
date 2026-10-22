---
title: "OLMo2 CUDA 验证预览"
sidebar:
  order: 1
---

<!-- Copyright 2026 The xLLM Authors. -->

本预览针对 `allenai/OLMo-2-0425-1B-Instruct`，固定 revision
`48d788eca847d4d7548f375ad03d3c9312f6139e`，使用 Python executor、
FlashInfer 分页 attention、FP16/BF16、eager、TP=1/2。两种计算精度均已在
本机 CUDA 环境通过冻结数值矩阵和真实 HTTP 验收。本次已完整包含原先在
#2518（Z25）提交的 OLMo2 C++ 配置加载、字段校验、fixture 与测试，
可作为一份完整实现审阅和合并。当前不是完整平台支持声明；NPU、量化、图执行
以及大于1的 DP/CP/PP 尚未完成本模型验收。
本次贡献目标为 NVIDIA CUDA；Ascend 适配是后续独立任务。

模型为16层、hidden2048、16个128维 MHA head、intermediate8192、vocab100352、
context4096。Q/K RMSNorm 在 reshape head 和 RoPE 前覆盖完整投影维度；
TP=2 跨 rank 分别归约 FP32 Q/K 平方和，再应用分片 norm 权重。
attention/MLP 的 post-norm 输出加入 FP32 残差流，投影权重和 KV cache
保持所选 FP16/BF16 计算 dtype；embedding 与 LM head 使用独立权重。

CUDA BF16 将 FP32 投影输入拆成 BF16 舍入值与 BF16 残差补偿，两次 GEMM
输出 FP32 后相加，再做归一化；行并行归约和 LM head logits 也保留 FP32。
BF16 decode 显式选择 FlashInfer tensor cores。权重和 KV cache 仍为 BF16。
此混合精度策略增加 GEMM 计算以及 FP32 激活/通信内存，并非纯 BF16 运算，
也没有据此声明提速。FP16 保留原有投影及 decode 策略。

## 创建独立 FP16 配置

保持原始 checkpoint 不变。源路径指向固定下载目录，目标路径必须不存在；
以下只创建独立 config，并符号链接原权重和 tokenizer。

```bash
export OLMO2_SOURCE=/path/to/OLMo-2-0425-1B-Instruct
export OLMO2_FP16=/path/to/OLMo-2-0425-1B-Instruct-fp16
python - <<'PY'
import json
import os
from pathlib import Path

source = Path(os.environ["OLMO2_SOURCE"]).resolve()
target = Path(os.environ["OLMO2_FP16"])
target.mkdir()
for item in source.iterdir():
    if item.name != "config.json":
        (target / item.name).symlink_to(item)
config = json.loads((source / "config.json").read_text())
config.update(dtype="float16", torch_dtype="float16")
(target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
PY
```

原权重为 BF16，转为 FP16 时部分极小值会改变。这是显式计算配置，不是
BF16 逐位不变的执行。token ID 参考使用 checkpoint 自带的 `tokenizer.json`；
Transformers5.12.1 的 `AutoTokenizer` 会重建 GPT2 tokenizer，换行预切分规则
与原文件不同。先用 HF 渲染模板字符串，再用
`tokenizers.Tokenizer.from_file(...).encode(..., add_special_tokens=False)`
编码，可保持与服务相同的原始 tokenization。

## 复现正式服务验收

先构建本分支 CUDA executable 和匹配 Python extension，安装 Python 依赖，
在仓库根目录运行。设置 `XLLM_BINARY` 为实际产物，并保留该 CUDA 构建所需
的运行库路径。本机验证环境为 Python3.12.3、Torch2.10.0+cu130、
Transformers5.12.1、FlashInfer0.6.14、TVM-FFI0.1.13。

```bash
export XLLM_BINARY=/path/to/built/xllm
python tools/validate_olmo2_cuda.py \
  --model "$OLMO2_FP16" --binary "$XLLM_BINARY" --dtype float16 \
  --tp 1 --devices 0 --port 18000 --output /tmp/olmo2-tp1

python tools/validate_olmo2_cuda.py \
  --model "$OLMO2_FP16" --binary "$XLLM_BINARY" --dtype float16 \
  --tp 2 --devices 0,1 --port 18000 --output /tmp/olmo2-tp2
```

BF16 使用原始目录 `--model "$OLMO2_SOURCE" --dtype bfloat16`。
`--dtype` 仅核对实际加载配置，不转换 checkpoint，也不覆盖服务端计算配置。

每次使用新的输出目录。工具只启动和停止自己的服务，保存命令、请求与 SSE，
遇到不一致即失败。TP=2 使用两个本机进程，分别设置
`--nnodes=2 --node_rank=0/1` 并共享 rendezvous 地址；当前引擎只让一个进程
可见两卡不会自动形成 TP=2。工具核对实际加载的 TP/dtype、模板 token 数、
流式一致性、EOS/ignore_eos、stop token/string、长度、错误输入及混合并发隔离。
这些属于服务验收，不能代替 logits 数值对照。
共享 GPU 可通过 `--max-memory-utilization` 和 `--max-cache-size` 显式设置
引擎已有内存参数，默认仍为0.4及每rank 1073741824字节。设置前须检查实际
空闲显存；启动记录和性能文件保存有效值。小cache可能增加抢占/重算，
不同容量的吞吐不能作受控性能对比。这些参数不改变请求或验收断言。
所有流式响应必须带有结束原因并以 `[DONE]` 结束，请求了 usage 时还必须返回
usage。即使完整文本已到达，截断的 chat 流或流内错误也不能通过验收。
只有全部服务 rank 以退出码0结束后，工具才写入最终 PASS 结果。
崩溃或强制终止不能通过验收，各 rank 退出码保存在 `shutdown.json`。

工具保留调用者设置的 `FLASHINFER_CUDA_ARCH_LIST`；未设置时由 FlashInfer
检测 GPU 架构，不会将 RTX4090 的架构强制应用到其他 NVIDIA GPU。
`launch.json` 记录实际架构覆盖值；其他型号仍需实机验证。

本机成功命令额外设置
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4`，并在
`LD_LIBRARY_PATH` 首位放置配套 Python TVM-FFI 的 `lib` 目录，以匹配本机分配器和 FFI 环境。

还需将 `FLASHINFER_OPS_PATH` 指向 native FlashInfer 内核目录（CUDA 镜像中为 `/flashinfer_ops`）。
服务工具增加未指定 seed、`temperature=0.7` 的普通与流式请求，检查完成长度和 token 用量，
覆盖贪心请求不会加载的 CUDA 随机采样内核。native 内核通过 TVM-FFI 打包接口查询函数，
避免嵌入式 Python 加载不同 TVM-FFI 版本时出现 C++ `Optional` 返回布局不匹配。

## 测试与性能测量

```bash
python -m pytest tests/python/test_olmo2.py tests/python/test_registry.py \
  tests/python/test_flashinfer_attention.py tests/python/test_validate_olmo2_cuda.py -q

python tools/validate_olmo2_cuda.py \
  --model "$OLMO2_FP16" --binary "$XLLM_BINARY" --dtype float16 \
  --tp 1 --devices 0 --port 18000 --output /tmp/olmo2-benchmark \
  --benchmark --build-label "填写实际编译选项"
```

可选性能矩阵为 prompt64/512/2048、output32、并发1/4，每组预热1次后正式
重复3次。关闭两种 prefix cache 和图，每rank KV cache上限1GiB，保留原始
SSE、显存快照及逐请求时间。TTFT 为首个非空文本到达时间；有效 TPOT 为
`(请求结束时间−TTFT)/(N−1)`，N取实际completion tokens，包含不可见special
tokens。这是客户端指标，不是逐token到达间隔或kernel时间。分析吞吐时需核对
其他GPU使用者及cache压力。

历史本机 native 使用 `XLLM_DEV_MODE=ON`（O1）开发构建；其测量只属于开发基线，
不代表发布性能，也没有据此声明加速。
10月20–21日变基到 main `b6123035` 后，使用匹配的 Release/O3 native 与 Python
产物重新验收。该轮 HTTP 计时与仓库测试程序的 CPU 编译重叠，属于诊断测量，
不作为受控的发布性能基准。


本机开发基线使用两张各报告48537MiB的RTX4090。下表为历史 FP16 三次重复的总输出吞吐
中位数及范围，原始请求保留在贡献证据中。两种TP配置分别记录，不视为优化对照。

| TP | Prompt tokens | 并发 | 输出tokens/s，中位数（范围） |
| --- | --- | --- | --- |
| 1 | 64 | 1 | 24.18 (22.93–24.38) |
| 1 | 64 | 4 | 90.64 (84.99–93.50) |
| 1 | 512 | 1 | 21.22 (21.12–22.27) |
| 1 | 512 | 4 | 59.63 (56.75–67.72) |
| 1 | 2048 | 1 | 17.06 (14.88–17.46) |
| 1 | 2048 | 4 | 34.03 (32.11–34.64) |
| 2 | 64 | 1 | 19.30 (19.01–19.56) |
| 2 | 64 | 4 | 67.22 (63.69–68.49) |
| 2 | 512 | 1 | 16.96 (16.80–17.43) |
| 2 | 512 | 4 | 51.36 (49.65–52.22) |
| 2 | 2048 | 1 | 14.76 (13.36–15.26) |
| 2 | 2048 | 4 | 33.96 (32.81–34.73) |

此配置TP1只有8192个cache slots，2048×4加输出需8320个，因此包含cache压力；
TP2分片后的cache容量也不同。不能据此声明发布加速或量化collective开销。
独立profiling覆盖prefill512及16步同历史decode，产生272次Q/K norm和544次
独立Q/K求和。多个小Torch算子及转换提示后续可调查CPU发射开销；本轮未新增
kernel优化，也没有声明优化后加速。

## 数值验收状态

BF16 和 FP16 均在原冻结阈值、HF BF16 参考下通过 TP1 的全部591项，
以及真实 NCCL TP2 每 rank 的1064项。两种精度、两种TP均通过七组 HTTP
功能及六组性能条件各三次重复。变基到 main `b6123035` 后，匹配的 Release/O3
产物再次通过完整矩阵：数值测试使用80页缓存，最终 HTTP 使用
TP1 `--max-memory-utilization 0.4`、TP2 `0.99`，全部 rank 均使用
`--max-cache-size 1073741824`。首次TP2按0.4配额启动，在共享GPU负载下
于ready前失败，原始失败记录保留。
本轮数值运行开始时两卡均可用；此前共享卡运行需要64页和更小的 HTTP cache
上限，资源失败与成功重试均保留在贡献证据中。结果适用于本机 CUDA 环境，
不据此声明性能提升。
混合精度策略仍需维护者评审和其他贡献者独立复现；
支持模型矩阵暂不改为完整支持。

## 复现完整 checkpoint 数值验收

`tools/validate_olmo2_numerics.py` 使用入库的原 B06/B07 固定输入与原 v2
预算（`tests/models/fixtures/olmo2/numerics.json`），没有按 xLLM 输出调整
阈值。模型参数应指向 revision `48d788eca847d4d7548f375ad03d3c9312f6139e`
的原始 BF16 checkpoint/config；SHA256 校验会拒绝不同权重或配置。
`--dtype float16` 在内存中转换权重（含次正规数的正常舍入），不改写原模型。
参考生成要求 Transformers 5.12.1，以未修改的 HF eager BF16 为验收目标，
另保存 FP32 诊断参考。

```bash
# Run from a built checkout with its matching xllm_export/runtime environment.
MODEL=/path/to/OLMo-2-0425-1B-Instruct
OUT=/path/to/new-validation
CUDA_VISIBLE_DEVICES=0 python tools/validate_olmo2_numerics.py reference \
  --model "$MODEL" --output "$OUT/reference"
CUDA_VISIBLE_DEVICES=0 python tools/validate_olmo2_numerics.py validate \
  --model "$MODEL" --reference "$OUT/reference" --output "$OUT/fp16-tp1" \
  --dtype float16
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  tools/validate_olmo2_numerics.py validate --tp 2 --dtype float16 \
  --model "$MODEL" --reference "$OUT/reference" --baseline "$OUT/fp16-tp1" \
  --output "$OUT/fp16-tp2"
# This is a diagnostic control; a threshold failure deliberately exits 1.
CUDA_VISIBLE_DEVICES=0 python tools/validate_olmo2_numerics.py hf-control \
  --model "$MODEL" --reference "$OUT/reference" --output "$OUT/hf-control"
```

每次使用新输出目录，生成的张量和 manifest 不提交 Git。TP2 要求相同 dtype
的 TP1 证据，并核对 checkpoint、Python runtime 源码及张量文件 SHA。
每个权重/分片逐值检查是否精确等于原checkpoint显式dtype转换后的结果，另记录
转换误差。双 rank 使用真实 xLLM NCCL collective 和 FlashInfer 分页执行，比较 HF、TP1
及两 rank 的全词表 logits。cache 初值为 NaN，覆盖非单调物理页、127/128/129
边界、chunked prefill、混合 batch 和总长4096。TP1 有591项比较，TP2 每rank
有1064项（比原 B08 扩展长序列覆盖）。缺数据、非有限值、形状及数值失败均
返回非零退出码，报告保留全部失败；TP2 共享失败状态，运行异常直接传播，
torchrun 负责终止整个 worker group。
数值工具默认预分配80页；`--cache-blocks 64` 仅减少未使用的缓存预分配，
最长增量历史与独立full参考各需32页且互不重叠。工具根据fixture计算完整矩阵
容量要求，拒绝更小容量并记录实际页数；序列长度、decode步数、混合batch和
数值阈值均不变。

将两个验证命令的 dtype 改为 `bfloat16`，并使用不同输出目录，可重跑修复后的
BF16 实现。`hf-control` 重跑未修改官方 eager BF16 的 cache/batch 与自身
full-prefill 参考之间的差异，不据此豁免 xLLM 失败或改变冻结标准。
自动隔离复现不能代替另一位贡献者的独立审核/复现。
