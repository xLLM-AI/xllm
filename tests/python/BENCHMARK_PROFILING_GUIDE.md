# Qwen3Model 32B Benchmark Profiling 使用指南

## 概述

`bench_qwen3_compile_runner.py` 现在支持 NPU profiling 功能，可以帮助分析不同后端（eager/torchair/inductor）的性能瓶颈。

## 基本用法

### 1. 运行性能基准测试（无 profiling）

```bash
# 测试所有后端和所有 image_size
python3 tests/python/bench_qwen3_compile_runner.py

# 测试特定后端
python3 tests/python/bench_qwen3_compile_runner.py --backend torchair

# 测试特定 image_size
python3 tests/python/bench_qwen3_compile_runner.py --image-size 2048

# 保存结果到 JSON 文件
python3 tests/python/bench_qwen3_compile_runner.py --output results.json
```

### 2. 运行 Profiling

```bash
# 对单个后端和 image_size 进行 profiling
python3 tests/python/bench_qwen3_compile_runner.py --profile --backend eager --image-size 256

# 指定 profiling 步数（默认 5 步）
python3 tests/python/bench_qwen3_compile_runner.py --profile --backend torchair --image-size 512 --profile-steps 10

# 指定输出目录
python3 tests/python/bench_qwen3_compile_runner.py --profile --backend inductor --image-size 1024 --profile-dir ./my_profiler

# 对所有后端和 image_size 进行 profiling
python3 tests/python/bench_qwen3_compile_runner.py --profile
```

## Profiling 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--profile` | 启用 profiling 模式 | 禁用 |
| `--profile-dir` | Profiling 输出目录 | `./profiler` |
| `--profile-steps` | Profiling 步数 | 5 |

## 查看 Profiling 结果

Profiling 结果会保存到指定的目录，每个后端和 image_size 组合会有独立的子目录：

```
./profiler/
├── eager_img256/
├── eager_img512/
├── torchair_img256/
├── torchair_img512/
├── inductor_img256/
└── inductor_img512/
```

### 使用 TensorBoard 查看

```bash
# 查看所有结果
tensorboard --logdir ./profiler

# 查看特定后端的结果
tensorboard --logdir ./profiler/torchair_img256

# 在浏览器中访问
# http://localhost:6006
```

### 使用 msprof 查看原始数据

```bash
# 查看原始 profiling 数据
msprof ./profiler/torchair_img256/*_ascend_pt
```

## Profiling 配置说明

当前 profiling 配置：

```python
experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=[torch_npu.profiler.ExportType.Text],
    profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
    msprof_tx=False,
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    l2_cache=False,
    op_attr=False,
    data_simplification=False,
    record_op_args=False,
    gc_detect_threshold=None,
)
```

**关键配置：**
- `profiler_level=Level2`: 详细的 profiling 信息
- `aic_metrics=PipeUtilization`: 记录 AI Core 管道利用率
- `record_shapes=True`: 记录张量形状
- `profile_memory=False`: 不记录内存分配（减少开销）

## 性能优化建议

### 1. 对比不同后端的 profiling 结果

```bash
# 对三个后端进行 profiling
python3 tests/python/bench_qwen3_compile_runner.py --profile --image-size 2048

# 使用 TensorBoard 对比
tensorboard --logdir ./profiler
```

### 2. 分析长序列的性能瓶颈

```bash
# 对长序列进行详细 profiling
python3 tests/python/bench_qwen3_compile_runner.py --profile --backend torchair --image-size 2048 --profile-steps 10
```

### 3. 识别算子融合效果

```bash
# 对比 eager 和 torchair 的算子执行差异
python3 tests/python/bench_qwen3_compile_runner.py --profile --backend eager --image-size 1024
python3 tests/python/bench_qwen3_compile_runner.py --profile --backend torchair --image-size 1024

# 在 TensorBoard 中对比算子列表和执行时间
```

## 注意事项

1. **Profiling 会增加执行开销**
   - Profiling 模式下的性能数据不代表真实性能
   - 仅用于分析性能瓶颈，不用于性能对比

2. **Profiling 数据文件较大**
   - 每个 profiling 会话可能生成数百 MB 的数据
   - 建议定期清理旧的 profiling 数据

3. **Parser warnings 是正常的**
   - 输出中的 "Incorrect schedule" 警告是预期的
   - 不影响 profiling 结果的分析

4. **内存占用**
   - Profiling 会增加内存占用
   - 如果 OOM，可以减少 `--profile-steps` 或使用更小的 `--image-size`

## 示例：完整的性能分析流程

```bash
# 1. 运行基准测试，获取性能数据
python3 tests/python/bench_qwen3_compile_runner.py --output baseline.json

# 2. 对性能较差的后端进行 profiling
python3 tests/python/bench_qwen3_compile_runner.py --profile --backend torchair --image-size 2048 --profile-dir ./profiler_torchair

# 3. 对性能较好的后端进行 profiling
python3 tests/python/bench_qwen3_compile_runner.py --profile --backend inductor --image-size 2048 --profile-dir ./profiler_inductor

# 4. 使用 TensorBoard 对比分析
tensorboard --logdir ./profiler_torchair --port 6006
tensorboard --logdir ./profiler_inductor --port 6007

# 5. 根据分析结果调整优化策略
# - 检查算子融合效果
# - 分析 AI Core 利用率
# - 识别内存瓶颈
# - 优化关键路径
```

## 故障排查

### Profiling 失败：OOM

```bash
# 减少 profiling 步数
python3 tests/python/bench_qwen3_compile_runner.py --profile --backend torchair --image-size 2048 --profile-steps 2

# 使用更小的 image_size
python3 tests/python/bench_qwen3_compile_runner.py --profile --backend torchair --image-size 1024
```

### Profiling 数据解析失败

```bash
# 检查磁盘空间
df -h

# 清理旧的 profiling 数据
rm -rf ./profiler/*

# 重新运行 profiling
python3 tests/python/bench_qwen3_compile_runner.py --profile --backend eager --image-size 256
```

### TensorBoard 无法启动

```bash
# 安装 TensorBoard
pip install tensorboard

# 指定端口
tensorboard --logdir ./profiler --port 6006

# 允许远程访问
tensorboard --logdir ./profiler --host 0.0.0.0 --port 6006
```

## 相关文档

- [Qwen3Model 32B 性能分析报告](./QWEN3_FINAL_PERFORMANCE_REPORT.md)
- [Recompilation 优化报告](./QWEN3_RECOMPILATION_OPTIMIZATION.md)
- [Torchair 优化分析](./TORCHAIR_OPTIMIZATION_ANALYSIS.md)
