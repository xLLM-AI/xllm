# Qwen3Model 32B CompileRunner 性能优化报告

## 问题背景

在之前的测试中，torchair 和 inductor 后端存在明显的性能差距，特别是在长序列（image_size=2048）下差距达到 10.5%。

## 根因分析

通过 `TORCH_LOGS="recompiles"` 分析发现：

1. **Recompilation 问题**：`NpuPagedAttentionBackend.execute()` 中使用 `layer.layer_id` 作为索引
2. **Guard Failure**：Dynamo 将 `layer_id` 视为静态整数，每层的 `layer_id` 不同（0-63），导致每层都需要重编译
3. **Recompile Limit**：默认 `recompile_limit=8`，只有前 8 层被编译，其余 56 层回退到 eager 执行

## 优化方案

### 方案1：提高 recompile_limit（已验证）

```python
import torch._dynamo.config
torch._dynamo.config.recompile_limit = 128
```

**效果**：允许所有 64 层都被编译，避免 eager 回退

### 方案2：将 layer_id 转为 Tensor（待验证）

修改 `NpuPagedAttentionBackend.execute()`：
```python
def execute(self, q, k, v, layer):
    layer_id_tensor = torch.tensor(layer.layer_id, device=q.device)
    k_cache, v_cache, _ = self._kv_caches[layer_id_tensor.item()]
    # ...
```

**预期效果**：Dynamo 不再为每个 `layer_id` 重编译，生成通用代码

### 方案3：重构 Attention 接口（长期方案）

将 `layer_id` 作为参数传递：
```python
class Attention(nn.Module):
    def forward(self, q, k, v, layer_id):
        backend = get_forward_context().attention_backend
        return backend.execute(q, k, v, layer_id)
```

## 性能对比

### 优化前（recompile_limit=8）

| 图像尺寸 | Torchair (ms) | Inductor (ms) | 差距 |
|---------|--------------|--------------|------|
| 256 | 136.02 | 129.96 | 4.5% |
| 512 | 212.01 | 206.64 | 2.5% |
| 800 | 331.62 | 328.79 | 0.9% |
| 1024 | 365.60 | 359.42 | 1.7% |
| **2048** | **799.43** | **715.15** | **10.5%** |

### 优化后（recompile_limit=128）

| 图像尺寸 | Torchair (ms) | Inductor (ms) | 差距 |
|---------|--------------|--------------|------|
| 256 | 136.14 | 130.51 | 4.1% |
| 512 | 212.78 | 205.49 | 3.4% |
| 800 | 329.08 | 326.35 | 0.8% |
| 1024 | 366.56 | 357.89 | 2.4% |
| **2048** | **730.16** | **716.25** | **1.9%** |

## 关键发现

1. **2048 图像尺寸差距显著缩小**：从 10.5% 缩小到 1.9%，改善幅度达 82%
2. **Torchair 性能提升**：2048 下从 799.43ms 提升到 730.16ms，提升 8.7%
3. **Recompilation 是主要原因**：长序列下注意力计算占比大，编译更多层带来的收益更明显
4. **短序列差距保持稳定**：256-1024 下差距在 0.8%-4.1% 之间，属于正常范围

## 建议

### 短期方案（推荐）

在 benchmark 脚本中添加：
```python
import torch._dynamo.config
torch._dynamo.config.recompile_limit = 128
```

**优点**：
- 无需修改模型代码
- 立即生效
- 性能提升显著（特别是长序列）

**缺点**：
- 首次编译时间增加（需要编译 64 个不同的图）
- 内存占用略增

### 长期方案

实施方案2或方案3，从根本上解决 recompilation 问题：
- 方案2：修改 `NpuPagedAttentionBackend`，将 `layer_id` 转为 Tensor
- 方案3：重构 `Attention` 接口，显式传递 `layer_id`

## 测试产物

- **优化后结果**：
  - `/tmp/bench_torchair_recompile128.json`
  - `/tmp/bench_inductor_recompile128.json`
- **Benchmark 脚本**：`tests/python/bench_qwen3_compile_runner.py`（已添加 recompile_limit=128）

## 结论

通过提高 `recompile_limit` 到 128，成功将 torchair 和 inductor 在长序列下的性能差距从 10.5% 缩小到 1.9%。这验证了 recompilation 问题是导致性能差距的主要原因。建议在生产环境中也应用此配置，或进一步实施方案2/3 从根本上解决问题。
