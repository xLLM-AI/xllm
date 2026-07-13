---
title: "MTP投机推理"
sidebar:
  order: 80
---
## 背景
MTP是一种创新的推理阶段加速技术，专注于解决大语言模型生成过程中的效率瓶颈。MTP的本质是通过预训练阶段的特殊设计，为推理阶段提供高效的草稿token预测能力，从而显著提升模型的生成速度。其核心价值在于平衡推理效率与输出质量，为大语言模型的长序列生成问题提供了一种高效的解决方案，最终实现推理性能的优化。

## 功能介绍
MTP在推理加速方面具有以下核心功能：

- **高效草稿生成**：使用低成本的MTP结构快速生成草稿token，这些草稿token作为主模型验证的基础，大幅减少了传统自回归生成的计算开销。

- **批量验证机制**：主模型能够同时批量验证多个MTP生成的草稿token，而不必逐个生成和验证，显著提升了推理速度。

- **高采样准确率**：MTP解决了Eagle、Medusa等现有推理加速方法中的关键痛点——训练后生成的draft模块token采样率低的问题。由于MTP在预训练阶段就优化了草稿生成能力，其生成的token具有更高的准确率，减少了主模型的验证负担。

- **推理延迟降低**：通过预先生成多个可能的后续token，MTP有效降低了模型生成长文本时的累积延迟，使用户体验更加流畅。

- **资源消耗优化**：相比其他推理加速技术，MTP在保持加速效果的同时，对计算资源的额外需求更少，适合在资源受限环境下部署。

MTP技术为大语言模型的推理阶段提供了一种全新的效率优化方案，特别适合需要快速响应的实时应用场景，代表了语言模型推理优化的重要发展方向。

:::note[模型支持]
目前支持以下模型的MTP结构导出：
- DeepSeek-V3 (输入 model_type: deepseek_v3, 导出 MTP model_type: deepseek_v3_mtp)
- DeepSeek-V3.2 (输入 model_type: deepseek_v3, 导出 MTP model_type: deepseek_v32_mtp)
- DeepSeek-R1 (输入 model_type: deepseek_v3, 导出 MTP model_type: deepseek_v3_mtp)
- GLM4 MoE (如 GLM-4.5-Air, 导出 MTP model_type: glm4_moe_mtp)

注意：
- DeepSeek V3 和 R1 的输入 model_type 都是 "deepseek_v3"，导出的 MTP 模型 model_type 为 "deepseek_v3_mtp"
- DeepSeek V3.2 的输入 model_type 是 "deepseek_v3"（但可通过 index_head_dim 等字段自动识别），导出的 MTP 模型 model_type 为 "deepseek_v32_mtp"

:::
## 使用示例

### 导出模型

脚本会自动检测模型类型，也可以手动指定。

#### DeepSeek-V3
```bash
python3 tools/export_mtp.py \
    --input-dir /path/to/DeepSeek-V3 \
    --output-dir /path/to/DeepSeek-V3-mtp
```

#### DeepSeek-V3.2
```bash
python3 tools/export_mtp.py \
    --input-dir /path/to/DeepSeek-V3.2 \
    --output-dir /path/to/DeepSeek-V3.2-mtp
```

#### DeepSeek-R1
```bash
python3 tools/export_mtp.py \
    --input-dir /path/to/DeepSeek-R1 \
    --output-dir /path/to/DeepSeek-R1-mtp
```

#### GLM4 MoE
```bash
python3 tools/export_mtp.py \
    --input-dir /path/to/GLM-4.5-Air \
    --output-dir /path/to/GLM-4.5-Air-mtp
```

#### 手动指定模型类型
如果自动检测失败，可以手动指定模型类型：
```bash
python3 tools/export_mtp.py \
    --input-dir /path/to/model \
    --output-dir /path/to/model-mtp \
    --model-type deepseek_v3  # 可选: deepseek_v3 (用于V3/R1), deepseek_v32 (用于V3.2), glm4_moe
```

输入模型参考:
- [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)
- [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)
- [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- [GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5-Air)

### 启动脚本

使用MTP进行推理时，需要同时指定主模型和草稿模型（MTP模型）。

#### DeepSeek-V3/V3.2/R1 启动示例
```bash
MODEL_PATH="/models/DeepSeek-V3"
DRAFT_MODEL_PATH="/models/DeepSeek-V3-mtp"
MASTER_NODE_ADDR="127.0.0.1:42123"
START_PORT=13222
START_DEVICE=0
LOG_DIR="log"
NNODES=16

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup ./xllm \
    --model $MODEL_PATH \
    --devices="npu:$DEVICE" \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --draft_model $DRAFT_MODEL_PATH \
    --draft_devices="npu:$DEVICE" \
    --num_speculative_tokens 1 \
    --max_memory_utilization=0.90 \
    --max_tokens_per_batch=10000 \
    --max_seqs_per_batch=256 \
    --block_size=128 \
    --ep_size=1 \
    --dp_size=1 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --node_rank=$i > $LOG_FILE 2>&1 &
  sleep 0.5
done
```

#### GLM4 MoE 启动示例
```bash
MODEL_PATH="/models/GLM-4.5-Air"
DRAFT_MODEL_PATH="/models/GLM-4.5-Air-mtp"
# ... 其他配置相同
```

## 性能数据
基于sharegpt数据集，输入长度2500，输出长度1500，请求总数80。

| method    | Concurrency | Mean TPOT(ms) | Mean TTFT(ms) | Output Tokens/s | Total Tokens/s |
|:---------:|:-----------:|:-------------:|:-------------:|:---------------:|:--------------:|
| baseline  |      1      |     40.61     |    141.80     |      24.20      |     65.77      |
| mtp       |      1      |     28.33     |    142.35     |      35.19      |     95.52      |
| baseline  |      2      |     42.69     |    178.59     |      45.16      |    122.74      |
| mtp       |      2      |     29.81     |    187.97     |      64.75      |    175.78      |
| baseline  |      4      |     46.18     |    172.34     |      79.83      |    216.96      |
| mtp       |      4      |     33.54     |    194.22     |     111.18      |    301.81      |
| baseline  |      8      |     53.16     |    181.49     |     110.68      |    300.81      |
| mtp       |      8      |     40.99     |    203.37     |     154.46      |    419.34      |
| baseline  |     16      |     68.50     |    213.89     |     143.81      |    390.84      |
| mtp       |     16      |     57.04     |    254.99     |     201.89      |    548.04      |
| baseline  |     20      |     74.72     |    228.80     |     154.77      |    420.65      |
| mtp       |     20      |     61.73     |    264.34     |     206.24      |    559.84      |
| baseline  |     40      |    119.68     |    559.32     |     180.22      |    489.80      |
| mtp       |     40      |    105.70     |    544.54     |     252.91      |    686.74      |
| baseline  |     80      |    180.89     |   2996.21     |     192.09      |    522.06      |
| mtp       |     80      |    152.19     |   2163.72     |     278.07      |    755.12      |

