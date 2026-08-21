# ViT Executor 设计文档

## 概述

本文档描述 xLLM 框架中 ViT Executor 的设计，用于支持 Qwen3-VL 视觉编码器（Vision Transformer）的多后端图编译执行。

## 背景与动机

### 问题

当前 `ModelExecutor` 专为 LLM 模型设计，具有以下特性：
- 依赖 attention backend 基础设施（FlashInfer/NPU Paged Attention）
- 管理 KV cache 绑定和生命周期
- 支持 decode graph runner（CUDA Graph/ACL Graph）
- 复杂的 forward 签名：`execute(input_ids, positions, metadata, ...)`

然而，ViT 模型（如 `Qwen3VLVisionTransformer`）具有不同的需求：
- **无 attention backend 依赖**：ViT 使用自定义的 `Qwen3VLVisionAttention`，不需要 xLLM 的 attention 基础设施
- **无 KV cache**：ViT 是无状态的编码器，不需要管理 KV cache
- **简单的 forward 签名**：`forward(pixel_values, grid_thw)`
- **需要图编译支持**：为了性能优化，需要支持 torchair/inductor 图编译

### 目标

设计一个轻量级的 ViT Executor，专门用于视觉编码器，支持：
1. **Eager 模式**：直接调用模型（无编译）
2. **TorchAir 模式**：使用 torchair 后端进行 NPU 图编译
3. **Inductor 模式**：使用 inductor 后端进行通用图编译

## 设计

### 架构

```
Qwen3VLForConditionalGeneration
├── vision_model: Qwen3VLVisionTransformer
│   └── ViTExecutor (可选)
│       ├── eager: 直接调用 vision_model
│       ├── torchair: torch.compile(vision_model, backend=torchair)
│       └── inductor: torch.compile(vision_model, backend=inductor)
├── model: Qwen3VLModel (LLM)
│   └── ModelExecutor (由 C++ 侧管理)
└── lm_head
```

### ViTExecutor 类

**文件位置**: `xllm/python/model_executor/vit_executor.py`

```python
class ViTExecutor:
    def __init__(
        self,
        model: nn.Module,
        backend: str = "eager",
        compile_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Args:
            model: 视觉编码器模型（如 Qwen3VLVisionTransformer）
            backend: 执行后端，可选 "eager", "torchair", "inductor"
            compile_kwargs: 传递给 torch.compile 的额外参数
        """
        
    @torch.inference_mode()
    def execute(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """执行视觉编码
        
        Args:
            pixel_values: 展平的 patches，shape (total_patches, C*t*p*p)
            grid_thw: 时间/高度/宽度网格，shape (num_images, 3)
            
        Returns:
            图像嵌入，shape (total_image_tokens, out_hidden_size * (1 + num_deepstacks))
        """
```

### 后端实现

#### 1. Eager 模式
```python
# 直接调用模型，无编译
self._compiled_model = None
return self.model(pixel_values, grid_thw)
```

#### 2. TorchAir 模式
```python
# 设置 torchair 环境变量
os.environ.setdefault(
    "AUTOFUSE_FLAGS",
    "--enable_autofuse=true;--autofuse_enable_pass=reduce,concat,transpose,gather,split,slice",
)

# 使用 torchair 后端编译
import torchair
from torchair import get_npu_backend
compiler_config = torchair.CompilerConfig()
backend = get_npu_backend(compiler_config=compiler_config)
self._compiled_model = torch.compile(model, backend=backend)
```

#### 3. Inductor 模式
```python
# 设置 inductor 环境变量
os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "ascendc")

# 使用 inductor 后端编译
self._compiled_model = torch.compile(model, backend="inductor")
```

### 集成到 Qwen3VLForConditionalGeneration

#### 配置参数

在 `config` dict 中添加：
- `vit_backend`: ViT 执行后端（"eager", "torchair", "inductor"）
- `vit_compile_kwargs`: 传递给 torch.compile 的额外参数（可选）

#### __init__ 修改

```python
def __init__(self, config: dict) -> None:
    # ... 现有代码 ...
    
    # Vision tower
    self.vision_model = Qwen3VLVisionTransformer(
        vision_cfg, dtype=dtype, device=device
    )
    
    # ViT executor (支持 eager/torchair/inductor 后端)
    from xllm.python.model_executor.vit_executor import ViTExecutor
    vit_backend = config.get("vit_backend", "eager")
    self.vit_executor = ViTExecutor(
        self.vision_model,
        backend=vit_backend,
        compile_kwargs=config.get("vit_compile_kwargs"),
    )
```

#### encode() 修改

```python
def encode(
    self,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
) -> torch.Tensor:
    pixel_values = pixel_values.to(
        dtype=self.vision_model.dtype, device=self.vision_model.device
    )
    grid_thw = grid_thw.to(dtype=torch.int32, device=self.vision_model.device)
    
    # 统一使用 ViTExecutor 执行（内部处理 eager/torchair/inductor）
    image_embeds = self.vit_executor.execute(pixel_values, grid_thw)
    
    return image_embeds
```

## 使用示例

### 1. Eager 模式（默认）

```python
config = {
    "hidden_size": 5120,
    "n_layers": 64,
    # ... 其他 LLM 配置 ...
    "mm_hidden_size": 1152,
    "mm_num_hidden_layers": 27,
    # ... 其他 ViT 配置 ...
    # vit_backend 默认为 "eager"
}

model = Qwen3VLForConditionalGeneration(config)
image_embeds = model.encode(pixel_values, grid_thw)
```

### 2. TorchAir 模式

```python
config = {
    # ... LLM 和 ViT 配置 ...
    "vit_backend": "torchair",  # 启用 torchair 图编译
}

model = Qwen3VLForConditionalGeneration(config)
# 首次调用会触发编译，后续调用使用编译后的图
image_embeds = model.encode(pixel_values, grid_thw)
```

### 3. Inductor 模式

```python
config = {
    # ... LLM 和 ViT 配置 ...
    "vit_backend": "inductor",  # 启用 inductor 图编译
    "vit_compile_kwargs": {
        "fullgraph": False,
        "dynamic": False,
    },
}

model = Qwen3VLForConditionalGeneration(config)
image_embeds = model.encode(pixel_values, grid_thw)
```

## 与 ModelExecutor 的对比

| 特性 | ModelExecutor | ViTExecutor |
|------|---------------|-------------|
| **目标模型** | LLM（Qwen3VLModel） | ViT（Qwen3VLVisionTransformer） |
| **Attention Backend** | FlashInfer / NPU Paged Attention | 无（ViT 使用自定义 attention） |
| **KV Cache** | 需要绑定和管理 | 无状态，不需要 |
| **Decode Graph** | 支持 CUDA Graph / ACL Graph | 不支持（ViT 只在 prefill 阶段使用） |
| **Forward 签名** | `execute(input_ids, positions, metadata, ...)` | `execute(pixel_values, grid_thw)` |
| **图编译后端** | inductor（通过 InductorRunner） | eager / torchair / inductor |
| **复杂度** | 高（213 行） | 低（~100 行） |

## 性能优化

### 图编译收益

根据基准测试（Qwen3-VL ViT 32B，NPU Ascend 910）：

| Image Size | Eager (ms) | TorchAir (ms) | 加速比 |
|------------|-----------|---------------|--------|
| 256        | 19.39     | 11.57         | **1.68x** |
| 512        | 19.00     | 19.31         | 0.98x |
| 800        | 28.80     | 29.36         | 0.98x |
| 1024       | 41.73     | 43.65         | 0.96x |
| 2048       | 152.16    | 162.78        | 0.93x |

**关键发现**：
- 小图像（256 patches）：TorchAir 比 Eager 快 68%
- 大图像（≥512 patches）：Eager 略优于 TorchAir（图调度开销）

### 建议

- **小图像场景**（≤256 patches）：使用 `vit_backend="torchair"`
- **大图像场景**（≥512 patches）：使用 `vit_backend="eager"`

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `xllm/python/model_executor/vit_executor.py` | 新增 | ViTExecutor 类实现 |
| `xllm/python/models/qwen3_vl.py` | 修改 | 集成 ViTExecutor 到 encode() |

## 后续扩展

1. **动态 shape 支持**：添加 `dynamic=True` 编译选项，支持不同 image_size 的批量推理
2. **多后端自动选择**：根据 image_size 自动选择最优后端（小图用 torchair，大图用 eager）
3. **编译缓存**：持久化编译后的图，避免重复编译开销
4. **性能监控**：添加编译时间和执行时间的性能指标

## 总结

ViTExecutor 是一个轻量级的视觉编码器执行器，专门用于支持 Qwen3-VL ViT 模型的图编译优化。与 ModelExecutor 相比，它：
- 更简单（无 attention backend、无 KV cache）
- 更灵活（支持 eager/torchair/inductor 三种后端）
- 更高效（小图像场景下最高 1.68x 加速）

通过配置 `vit_backend` 参数，用户可以灵活选择执行模式，无需修改模型代码。
