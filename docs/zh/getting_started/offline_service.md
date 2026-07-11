# 离线推理

为了方便用户快速使用xLLM进行离线推理，我们提供了启动离线推理的python脚本例子

## LLM

LLM推理示例：[:simple-github: https://github.com/jd-opensource/xllm/blob/main/examples/generate.py](https://github.com/jd-opensource/xllm/blob/main/examples/generate.py)

LLM Beam Search 示例：[:simple-github: https://github.com/jd-opensource/xllm/blob/main/examples/generate_beam_search.py](https://github.com/jd-opensource/xllm/blob/main/examples/generate_beam_search.py)

使用 `BeamSearchParams` 设置大于 `1` 的 `beam_width`，然后调用 `llm.beam_search(...)`：

```python
from xllm import BeamSearchParams, LLM

llm = LLM(model="/path/models/Qwen2-7B-Instruct", devices="npu:0")
params = BeamSearchParams(
    beam_width=2,
    top_logprobs=4,
    max_tokens=20,
)

outputs = llm.beam_search(
    [{"prompt": "Hello, my name is "}],
    params=params,
)
print(outputs[0].sequences[0].text)

llm.finish()
```

LLM Beam Search 使用 `beam_width` 作为开启参数。`top_logprobs` 控制每个解码步用于 beam 扩展的 top-k 候选数量。如果 `top_logprobs` 保持默认值，xLLM 会使用 `beam_width` 作为 top logprob 数量。如果希望每个 beam 考虑更多候选 token，可以将 `top_logprobs` 设置为大于 `beam_width` 的值。这里的 beam-search top-k 不同于采样截断参数 `top_k`。`best_of` 不是 Beam Search 开关，本文档也不使用 `num_return_sequences` 来控制 LLM 返回的 beam 数。

## 多机离线推理

当单台机器的设备数量不足以加载模型时，可以使用多机离线推理将模型分布在多个节点上。所有节点运行相同的脚本，通过 `nnodes`、`node_rank` 和 `master_node_addr` 参数区分角色：

- `node_rank=0` 的节点为驱动节点（driver），负责发起推理请求和收集结果
- `node_rank>0` 的节点为辅助节点（assistant），启动后自动进入等待状态，由驱动节点调度执行

### 参数说明

| 参数 | 说明 |
|------|------|
| `nnodes` | 总节点数，默认为 1（单机） |
| `node_rank` | 当前节点编号，范围 `[0, nnodes)`，驱动节点为 0 |
| `master_node_addr` | 驱动节点的通信地址，格式为 `host:port`，所有节点必须能访问该地址 |
| `rank_tablefile` | HCCL 集合通信拓扑文件路径（多机场景必须提供） |

### 使用示例

以 2 机 16 卡为例，假设驱动节点 IP 为 `11.87.191.99`：

**节点 0（驱动节点）：**

```bash
python examples/generate.py \
    --model='/path/models/GLM-5-final-w8a8' \
    --devices='npu:0,npu:1,npu:2,npu:3,npu:4,npu:5,npu:6,npu:7' \
    --max_seqs_per_batch=300 \
    --max_memory_utilization=0.9 \
    --nnodes=2 \
    --node_rank=0 \
    --rank_tablefile=hccl_tools/hccl_2s_16p.json \
    --master_node_addr="11.87.191.99:19888"
```

**节点 1（辅助节点）：**

```bash
python examples/generate.py \
    --model='/path/models/GLM-5-final-w8a8' \
    --devices='npu:0,npu:1,npu:2,npu:3,npu:4,npu:5,npu:6,npu:7' \
    --max_seqs_per_batch=300 \
    --max_memory_utilization=0.9 \
    --nnodes=2 \
    --node_rank=1 \
    --rank_tablefile=hccl_tools/hccl_2s_16p.json \
    --master_node_addr="11.87.191.99:19888"
```

### 注意事项

1. 所有节点必须使用相同的模型路径、`devices` 配置和 `nnodes` 参数
2. `master_node_addr` 必须为驱动节点的真实可达 IP，不能使用 `127.0.0.1`、`localhost` 等回环地址
3. `rank_tablefile` 需预先通过 `hccl_tools` 生成，描述多机集合通信拓扑
4. 辅助机器只需构造 `LLM` 对象即可，无需调用 `generate()` 或定义 prompts；`LLM` 初始化时检测到 `node_rank>0` 会自动进入等待循环，直到驱动节点调用 `llm.finish()` 时被远程关闭
5. 各节点启动顺序无要求，框架内部会自动等待所有节点就绪

## Embedding

生成Embedding示例：[:simple-github: https://github.com/jd-opensource/xllm/blob/main/examples/generate_embedding.py](https://github.com/jd-opensource/xllm/blob/main/examples/generate_embedding.py)

## VLM

VLM推理示例：[:simple-github: https://github.com/jd-opensource/xllm/blob/main/examples/generate_vlm.py](https://github.com/jd-opensource/xllm/blob/main/examples/generate_vlm.py)
