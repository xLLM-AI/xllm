# Packed Page Indices

紧凑页索引（ragged page indices）算子：在长短请求混合批/连续批处理中，把各请求
页表的有效前缀按请求顺序压紧，并逐层翻译为物理页号，供变长 Attention / KV cache
组件读取。

## 接口

```python
from solution import prepare, run

prepared = prepare([X, M, cu], {})
out = prepared.run()
```

| 参数 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `X` | `[R, C]` | int32 | 逻辑页表，有效前缀由 `cu` 确定 |
| `M` | `[B]` 或 `[L, B]` | int32 | 逻辑页到物理页映射，一维单层，二维逐层 |
| `cu` | `[R+1]` | int32 | 累计有效页数，`cu[0]=0`，单调，`0 <= cu[r+1]-cu[r] <= C` |
| `out` | `[P]` 或 `[L, P]` | int32 | `P = cu[-1]`，独立连续输出 |

语义：

```text
out[l, cu[r] + j] = -1                 if X[r, j] < 0
                  = M[l, X[r, j]]      otherwise
```

无效尾部不参与值域检查，也不在设备上读取。`prepare` 后 `P`、shape、dtype、device
固定；允许在总 `P` 不变时更新各请求 counts，以及 `X/M/cu` 的同形状存储重绑定。

## 实现

- **融合路径**（`L*P < 32768`）：单个 Triton Ascend kernel 完成有效前缀打包、逐层
  翻译和负页号处理。
- **原生 Gather 路径**（`L*P >= 32768`）：Triton 核生成打包索引，原生 NPU
  `index_select` 从 `[-1, M]` 增广映射中 gather。索引张量每次调用分配，保证双
  stream 并发安全。
- **主机侧 C++ 扩展**：元数据校验、输出分配、增广映射构建/刷新、原生 Gather 调度。
- 每次 `run` 使用当前 NPU stream；不缓存输出；每次返回独立存储。

详见 [DESIGN.md](DESIGN.md)。

## 构建

```bash
source ~/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
python build_ext.py
```

依赖：CANN 9.1.0、Python、PyTorch、torch_npu、triton-ascend、gcc/g++、Ninja。
`build_ext.py` 通过 `torch.utils.cpp_extension.load` 构建 `csrc/packed_ext.cpp`。

## 测试

```bash
python tests/test_smoke.py
# 或
pytest -q tests/test_smoke.py
```

冒烟测试覆盖字面例、前缀 padding、单层、空请求/空列、动态 counts、storage
重绑定等场景。

完整自验证材料见 `task_submission/`（`result.json`、`generalization.json`、
精度/性能/内存报告与日志）。

## 实测性能

环境：Atlas A3（Ascend 910C）、CANN 9.1.0、torch 2.10.0+cpu、
torch_npu 2.10.0.post4、triton-ascend 3.2.2。

| 用例 | 基线均值 | 提交均值 | 均值差 | 95% CI 下限 | 目标 |
|---|---:|---:|---:|---:|---:|
| `packed_perf_small` | 143.61 us | 31.15 us | 112.46 us | 110.21 us | <= 40 us |
| `packed_perf_large` | 143.29 us | 53.50 us | 89.80 us | 89.08 us | <= 75 us |

测量协议：预热 50 次，6 轮 × 200 次，交替基线/提交，计时完整 `prepared.run`，
bootstrap 10000 次（seed=20261031，线性分位数）。
