# xLLM Python 测试文档索引

本目录包含 xLLM Python 模型执行器的测试和性能基准测试工具。

## 测试文件

### 单元测试

- **test_model_executor.py** - ModelExecutor 单元测试
  - 测试 eager/torchair/inductor 后端
  - 测试 Qwen3Model 32B 编译和执行
  - 测试 KV Cache 管理
  - 运行测试：`pytest tests/python/test_model_executor.py -v`

### 性能基准测试

- **bench_qwen3_compile_runner.py** - Qwen3Model 32B 性能基准测试
  - 支持 eager/torchair/inductor 三种后端
  - 支持多种 image_size (256, 512, 800, 1024, 2048)
  - 支持 NPU profiling
  - 详细用法见 [BENCHMARK_PROFILING_GUIDE.md](./BENCHMARK_PROFILING_GUIDE.md)

### 优化实验

- **bench_torchair_optimizations.py** - Torchair 优化配置实验
  - 测试不同的 AUTOFUSE 配置
  - 测试 GE 编译器优化选项
  - 对比不同配置的性能差异

## 性能报告

### 核心报告

- **[QWEN3_FINAL_PERFORMANCE_REPORT.md](./QWEN3_FINAL_PERFORMANCE_REPORT.md)** - Qwen3Model 32B 最终性能报告
  - 三种后端的完整性能对比
  - 详细的性能数据分析
  - 优化建议和结论

### 优化分析报告

- **[QWEN3_RECOMPILATION_OPTIMIZATION.md](./QWEN3_RECOMPILATION_OPTIMIZATION.md)** - Recompilation 优化报告
  - 分析 layer_id 导致的 recompilation 问题
  - 对比三种解决方案
  - 推荐方案：设置 recompile_limit=128

- **[QWEN3_RECOMPILATION_SOLUTION2_EVALUATION.md](./QWEN3_RECOMPILATION_SOLUTION2_EVALUATION.md)** - 方案2评估报告
  - 详细评估将 layer_id 转为 Tensor 的方案
  - 分析失败原因
  - 总结经验教训

- **[TORCHAIR_OPTIMIZATION_ANALYSIS.md](./TORCHAIR_OPTIMIZATION_ANALYSIS.md)** - Torchair 优化分析报告
  - 分析 torchair 相比 inductor 慢的原因
  - 测试多种优化配置
  - 提供优化建议

### 使用指南

- **[BENCHMARK_PROFILING_GUIDE.md](./BENCHMARK_PROFILING_GUIDE.md)** - Profiling 使用指南
  - 如何运行性能基准测试
  - 如何使用 NPU profiling
  - 如何分析 profiling 结果
  - 故障排查指南

## 快速开始

### 运行性能基准测试

```bash
# 测试所有后端和所有 image_size
python3 tests/python/bench_qwen3_compile_runner.py

# 测试特定后端
python3 tests/python/bench_qwen3_compile_runner.py --backend torchair

# 保存结果到 JSON
python3 tests/python/bench_qwen3_compile_runner.py --output results.json
```

### 运行 Profiling

```bash
# 对特定后端和 image_size 进行 profiling
python3 tests/python/bench_qwen3_compile_runner.py --profile --backend torchair --image-size 2048

# 查看 profiling 结果
tensorboard --logdir ./profiler
```

### 运行单元测试

```bash
# 运行所有测试
pytest tests/python/test_model_executor.py -v

# 运行特定测试类
pytest tests/python/test_model_executor.py::TestQwen3CompileRunner -v

# 运行特定测试
pytest tests/python/test_model_executor.py::TestQwen3CompileRunner::test_qwen3_inductor_fullgraph -v
```

## 性能数据

### 最新性能数据（2026-08-07）

**测试环境：**
- 模型：Qwen3Model 32B (64 layers, hidden=5120)
- 设备：NPU (Ascend 910, 61GB)
- 配置：recompile_limit=128

**性能对比（image_size=2048）：**

| 后端 | 平均延迟 (ms) | 相对 Eager | 相对 Torchair |
|------|--------------|-----------|---------------|
| Eager | 813.94 | - | - |
| Torchair | 730.16 | 10.3% ↓ | - |
| Inductor | 716.25 | 12.0% ↓ | 1.9% ↓ |

**结论：**
- Inductor 性能最优，比 Eager 快 12%
- Torchair 次之，比 Eager 快 10.3%
- Inductor 比 Torchair 快 1.9%

详细数据见 [QWEN3_FINAL_PERFORMANCE_REPORT.md](./QWEN3_FINAL_PERFORMANCE_REPORT.md)

## 已知问题

### 1. Recompilation 问题（已解决）

**问题：** layer_id 导致 Dynamo 为每一层重新编译图

**解决方案：** 设置 `torch._dynamo.config.recompile_limit = 128`

**状态：** ✅ 已解决

### 2. Torchair 性能略低于 Inductor

**问题：** Torchair 比 Inductor 慢约 1.9%

**原因：** 
- 编译器架构差异
- 优化策略差异
- 代码生成差异

**状态：** ⚠️ 已知问题，优化空间有限

详细分析见 [TORCHAIR_OPTIMIZATION_ANALYSIS.md](./TORCHAIR_OPTIMIZATION_ANALYSIS.md)

## 贡献指南

### 添加新的测试

1. 在 `test_model_executor.py` 中添加测试类
2. 使用 `@pytest.mark.skipif` 标记需要特定环境的测试
3. 添加详细的 docstring 说明测试目的
4. 运行测试确保通过：`pytest tests/python/test_model_executor.py -v`

### 添加新的 Benchmark

1. 创建新的 benchmark 脚本（如 `bench_xxx.py`）
2. 参考 `bench_qwen3_compile_runner.py` 的结构
3. 支持 `--backend`、`--image-size`、`--output` 等标准参数
4. 添加 profiling 支持（可选）
5. 更新本文档

### 更新性能报告

1. 运行完整的性能基准测试
2. 保存结果到 JSON 文件
3. 更新对应的性能报告文档
4. 添加日期和测试环境信息

## 相关文件

### 源代码

- `xllm/python/model_executor/executor.py` - ModelExecutor 实现
- `xllm/python/model_executor/runners/compile_runner.py` - CompileRunner 实现
- `xllm/python/models/qwen3.py` - Qwen3Model 实现
- `xllm/python/attention/npu_paged_attention.py` - NPU 注意力后端

### 配置文件

- `xllm/pybind/args.py` - 命令行参数定义
- `xllm/core/framework/config/execution_config.h` - 执行配置定义

## 联系与支持

如有问题或建议，请联系 xLLM 开发团队。
