# PackedPageIndices

昇腾社区任务「题目6：ragged page indices 打包算子」的提交工程。算子把 ragged 页表的有效前缀压紧，并逐层翻译为物理页号，供变长 Attention 与缓存组件读取。

上游贡献分支：`Fun10165/xllm` 的 `feat/packed-page-indices`，目录 `xllm/core/kernels/npu/xllm_ops/packed_page_indices/`（自包含子目录，不修改既有 CMake 目标）。本仓库是完整交付：主机侧 C++ 规划、Triton 设备核、测试与自验证材料。

## 目录

| 路径 | 职责 |
| --- | --- |
| `solution.py` | 提交入口：`prepare` / `run`，调用 C++ 规划并构造 Triton 计划 |
| `packed_page_indices.py` | 设备核与可复用计划（与上游 PR 同一份实现） |
| `host/packed_host.cpp` | 主机侧 C++ 规划：cu 校验、页号范围、P、层/页形状、元素上限 |
| `host/build_host.py` | 用 torch 的 Ninja 引导器编译主机扩展，带缓存 |
| `build.py` | 显式构建入口 |
| `tests/test_packed_page_indices_npu.py` | 29 项真机测试 |
| `tools/make_reports.py` | 从官方验收 JSON 生成 xlsx 报告与 `generalization.json` |
| `evidence/` | 官方功能门禁与配对计时的原始 JSON/日志 |
| `task_submission/` | 任务书要求的七份自验证材料与打包结果 |

## 环境

| 项 | 值 |
| --- | --- |
| 设备 | Ascend910_9362（CANNLab A3，单逻辑 NPU） |
| CANN | 9.1.0 |
| Python | 3.11.4 |
| torch | 2.7.1+cpu |
| torch-npu | 2.7.1.post4 |
| triton-ascend | 3.2.1 |

```bash
source /home/developer/Ascend/cann-9.1.0/set_env.sh
python3 -m pip install -r requirements.txt
python3 build.py
```

`ninja` 由 pip 安装；若不在 PATH，`host/build_host.py` 会在 `~/.local/bin` 与当前解释器的 `bin` 下查找。主机扩展只在第一次 `prepare` 时编译，之后命中 `host/build` 缓存。

## 接口

```python
from solution import prepare, run

prepared = prepare([x, m, cu], {})
out = prepared.run()
```

`X` 为 `[R, C]` int32，`M` 为 `[B]` 或 `[L, B]`，`cu` 为 `[R+1]`，三者在同一 NPU 上连续、为 ND 存储，允许非零 `storage_offset`。输出为 `[P]` 或 `[L, P]`，`P = cu[-1]`。

```text
out[l, cu[r] + j] = -1            if X[r, j] < 0
out[l, cu[r] + j] = M[l, X[r, j]] otherwise
```

`prepare` 之后形状、类型、设备与总 `P` 固定；同形状的内存重绑定与总 `P` 不变的有效页数变化会被下一次 `run` 观察到。

## 验证

```bash
# 29 项真机测试
PYTHONPATH=. python3 -m pytest tests/test_packed_page_indices_npu.py -q

# 官方功能与工程门禁（输出文件必须不存在）
python3 /mnt/workspace/pkg/golden.py --verify-functional \
  --implementation /mnt/workspace/submission \
  --output /mnt/workspace/functional2.json

# 配对计时（驱动见证据目录对应的执行记录）
python3 bench.py --package /mnt/workspace/pkg --implementation . --output evidence/result.json
```

## 实测结果

| 用例 | 目标 | 基线均值 | 提交均值 | 均值差 | 95% CI 下界 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `packed_perf_small` | ≤ 40 μs | 124.15 μs | 37.93 μs | 86.22 μs | 84.13 μs | PASS |
| `packed_perf_large` | ≤ 75 μs | 122.78 μs | 45.75 μs | 77.03 μs | 75.79 μs | PASS |

功能门禁：14 个正向用例、8 个非法输入、7 项动态更新、6 项存储重绑定、单层大输入全部 PASS；两组性能输入的活跃分配在释放后回到测量前水平。详见 `task_submission/` 与 `evidence/`。

## 与上游 PR 的关系

任务书 §5 指向 `xllm/core/kernels/npu/xllm_ops`，该目录现有文件都是既有 aclnn 符号的 C++ 包装（`EXEC_NPU_CMD`），构建时统一链接 `xllm_ops`。本题明确无需提供 ACLNN 自定义安装包，因此上游 PR 在 `xllm_ops` 下新增自包含子目录 `packed_page_indices/`：自带主机侧 C++ 构建、Triton 核、测试与自验证材料，不修改既有 CMake 目标，也不依赖 aclnn 符号。详见 `DESIGN.md` 第 8 节。

## 说明

本仓库文字与代码由 AI 生成，用户提供任务书与验收环境。`evidence/` 下的 JSON 由官方 `golden.py` 与配对计时驱动直接产生。
