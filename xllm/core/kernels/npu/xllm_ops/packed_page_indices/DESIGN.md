# Ragged Page Indices 打包算子设计文档

## 1. 任务与接口

面向 Qwen3-8B、Llama-3.1-8B-Instruct 等自回归模型的长短请求混合批处理/连续批处理场景。
各请求的逻辑页表为 `X[R, C]`，累计有效页数为 `cu[R+1]`，逻辑页到物理页的映射为
`M[B]`（单层）或 `M[L, B]`（逐层）。算子把各请求的有效前缀按请求顺序压紧，并翻译为
物理页号；负页号表示 padding，输出 `-1`。

```python
def prepare(inputs: Sequence[torch.Tensor], attributes: dict) -> Prepared: ...
def run(inputs: Sequence[torch.Tensor], attributes: dict) -> torch.Tensor: ...
# prepared.run() -> torch.Tensor
```

输入固定为 `X`、`M`、`cu`，`attributes` 为空字典。`prepare` 做一次性校验并准备运行
元数据；`prepared.run()` 使用调用时的当前 NPU stream，读取当前 `X/M/cu`，每次分配
独立输出。总输出页数 `P = cu[-1]` 在 prepare 后固定；允许在总 `P` 不变时更新各请求
有效页数，也支持 `X/M/cu` 同形状 `set_` 重绑定。

## 2. 语义与地址推导

对请求 `r`、层 `l`、有效列 `j`（`0 <= j < count_r = cu[r+1] - cu[r]`）：

```text
out[l, cu[r] + j] = -1                 if X[r, j] < 0
                  = M[l, X[r, j]]      otherwise
```

输出为 `int32[P]`（`M` 为一维）或 `int32[L, P]`（`M` 为二维）。行主序下输出地址为：

```text
flat_index = l * P + cu[r] + j
```

无效尾部（`j >= count_r`）不参与值域检查，也不在设备上读取。`cu[0]=0`、单调不减、
每段长度不超过 `C`；有效前缀内非负页号必须小于 `B`。`R/C/P` 可以为 0，`B/L >= 1`。
输入/输出元素总数不超过 `2^31-1`。

## 3. 总体架构

```text
solution.py
  ├─ Python 封装 prepare / prepared.run
  ├─ Triton Ascend 设备核（两条路径）
  └─ csrc/packed_ext.cpp（主机侧 C++ 扩展，Ninja 构建）
        ├─ packed_meta          一次性元数据/属性校验
        ├─ packed_empty         当前 NPU 设备上的独立 int32 输出分配
        ├─ packed_make_aug      构建 [-1, M] 增广映射
        ├─ packed_refresh_aug   M 原地更新/重绑定后的映射刷新
        └─ packed_gather        原生 NPU Gather（index_select）
```

设备计算全部由 Triton Ascend 核或原生 NPU Gather 完成；CPU 仅做 prepare 阶段的一次性
校验（`detach().cpu().tolist()`），`prepared.run()` 不做任何主机侧张量拷贝。

## 4. 两条设备路径

### 4.1 融合路径（小输出，`L*P < 32768`）

一个 Triton 核完成“有效前缀打包 + 逐层翻译 + 负页号处理”：

```text
grid = (ceil(C / BLOCK), R, L)
program (tile, r, l):
  start = cu[r]; end = cu[r+1]; count = end - start
  j = tile*BLOCK + arange(0, BLOCK); mask = j < count
  x = load(X + r*C + j, mask=mask, other=0)
  idx = where(x < 0, 0, x)
  m = load(M + l*B + idx, mask=mask, other=0)
  store(OUT + l*P + start + j, where(x < 0, -1, m), mask=mask)
```

- `count` 在设备上从当前 `cu` 读取，因此 `prepared.run()` 能看到动态 counts；
- `mask` 同时阻止无效尾部读取和越界写；
- 负页号只读 `M[0]` 后由 `where` 覆盖为 `-1`，不产生额外的 `where` kernel；
- 小输出下该路径一次 kernel launch 完成，实测约 31.1 us。

### 4.2 原生 Gather 路径（大输出，`L*P >= 32768`）

大输出时 Triton 内的随机 gather 随 `L*R*C` 增长较快；改用原生 NPU Gather：

1. `prepare` 构建增广映射 `M_aug`：`M_aug[..., 0] = -1`，`M_aug[..., 1:] = M`。
2. 每次 `run` 分配本调用专属的 `idx[P]`（int32），由 Triton 打包核写入：

   ```text
   idx[cu[r] + j] = 0            if X[r, j] < 0
                  = X[r, j] + 1  otherwise
   ```

3. 原生 NPU Gather：一维映射 `index_select(M_aug, 0, idx)`；二维映射
   `index_select(M_aug, 1, idx)`，直接得到 `[P]` 或 `[L, P]` 输出。

该路径把负页号编码进索引 0，避免额外的 `where` kernel；`idx` 每次调用独立分配，
两条 stream 并发执行时不会互相覆盖。实测大用例约 53.5 us。

### 4.3 路径选择

`prepare` 时按 `layers * P` 一次性选择路径，阈值 32768 与题目的 route 用例分界一致：

| 用例 | `L*P` | 路径 |
|---|---:|---|
| `packed_perf_small` | 2,296 | fused |
| `packed_route_32736` | 32,736 | fused |
| `packed_route_32768` | 32,768 | hybrid |
| `packed_perf_large` | 252,736 | hybrid |

## 5. 动态更新、stream 与生命周期

- **动态 counts**：核函数每次从设备读取 `cu[r]`、`cu[r+1]`，不缓存 `ids`；总 `P` 不变时
  各请求 counts 可任意合法变化，输出位置随之平移。
- **动态值**：`X`、`cu` 每次读取当前张量；`M` 在融合路径直接读取，在混合路径由
  `(data_ptr, _version)` 检测是否变化，变化时刷新 `M_aug`。
- **存储重绑定**：`prepared.inputs` 保存张量对象而非裸指针；`set_` 后 launch 时读取新的
  `data_ptr`，支持非零 `storage_offset`。
- **当前 stream**：每次 `run` 调用
  `triton.runtime.driver.active.get_current_stream()`，不固定 stream。
- **输出独立**：融合路径每次 `packed_empty` 分配输出；混合路径每次分配 `idx`，原生
  Gather 另分配输出。两条 stream 各保留 3 份输出时 storage 指针互不相同。
- **不缓存输出**：不读取/复用上一次结果，不修改输入。

## 6. 内存设计

- `M_aug` 为 prepare 阶段的一次性静态布局，大小 `L*(B+1)*4` 字节；混合路径保留，
  融合路径不创建。
- 每次调用的临时量只有 `idx`（混合路径，`P*4` 字节）和输出本身（`L*P*4` 字节）。
- 融合路径每次只有输出分配；混合路径为 `idx + 输出`。
- 官方内存检查口径（预热后活跃分配增量）：
  - small：峰值增量 9,216 B，输出 9,184 B，基线峰值 23,552 B；
  - large：峰值增量 1,043,968 B，输出 1,010,944 B，基线峰值 2,126,336 B；
  - 20 次释放回收后活跃分配回到测量前水平（residual 0）。

## 7. 输入校验

`prepare` 按契约逐项校验，非法输入抛出 `TypeError` 或 `ValueError`：

- 输入个数、`attributes` 必须为空；
- 所有输入必须是同一 NPU 上的连续 `int32` 张量，`numel <= 2^31-1`；
- `X` 二维；`M` 一维或二维且 `B >= 1`、`L >= 1`；`M` 非负；
- `cu` 一维、长度 `R+1`、`cu[0]=0`、单调不减、每段长度不超过 `C`；
- 有效前缀内的非负页号 `< B`；无效尾部不检查、不读取；
- `L*P <= 2^31-1`。

8 项负向检查（input_count、unknown_attribute、cpu_input、dtype、noncontiguous、
negative_mapping、logical_address_out_of_range、nonmonotonic_cu）全部通过。

## 8. 依赖与构建

- 硬件：昇腾 Atlas A3（Ascend 910C），单逻辑 NPU；
- CANN 9.1.0（含 driver/firmware/toolkit/ops）；
- Python 3.11、PyTorch（CPU wheel）、torch_npu、triton-ascend；
- host C++ 扩展：gcc/g++、Ninja，`torch.utils.cpp_extension.load` 构建。

本机自测实际版本：

| 组件 | 版本 |
|---|---|
| Python | 3.11.14 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0.post4 |
| triton-ascend | 3.2.2 |
| triton | 3.2.0（triton-ascend 自带） |
| 设备 | Ascend910_9362 |
| CANN | 9.1.0 |

构建：

```bash
source ~/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
python build_ext.py
```

运行：

```bash
python -m selftest.run_all
```

## 9. 性能测量与结果

协议：预热 50 次；6 轮 × 200 次；每轮交替基线/提交；计时范围为完整
`prepared.run`（含 Python、输出分配、当次临时量、设备执行及最终同步）；bootstrap
10000 次，seed=20261031，线性分位数。结果（本机 A3，CANN 9.1.0）：

| 用例 | 基线均值 (us) | 提交均值 (us) | 均值差 (us) | 95% CI 下限 | 目标上限 (us) | 判定 |
|---|---:|---:|---:|---:|---:|---|
| `packed_perf_small` | 143.61 | 31.15 | 112.46 | 110.21 | 40 | PASS |
| `packed_perf_large` | 143.29 | 53.50 | 89.80 | 89.08 | 75 | PASS |
| `packed_route_32736` | 142.73 | 31.10 | 111.63 | 110.58 | — | 记录 |
| `packed_route_32768` | 141.28 | 53.27 | 88.01 | 87.27 | — | 记录 |

均值、中位数差均大于 0，bootstrap 95% CI 下限均大于 0。基线为题目定义的
PyTorch NPU 实现（`golden.reference("baseline")`），提交实现与基线在同一进程、
同一设备、同一输入上交替测量。

## 10. 工程结构

```text
solution.py                 接口、校验、两条 Triton 路径
csrc/packed_ext.cpp         主机侧 C++ 扩展
build_ext.py                扩展预构建
task_pkg/                   发布包（case.json、golden.py、generate_data.py、任务书）
selftest/
  common.py                 环境/用例/哈希/统计
  functional.py             官方功能与工程检查入口
  performance.py            配对性能测量
  memory.py                 内存/stream/storage 检查
  generalization.py         补充用例与 SHA-256
  report.py                 result.json / generalization.json / xlsx
  run_all.py                一键自测与报告
task_submission/            自验证材料（报告、日志、result.json、zip）
```

## 11. 限制与说明

- 总 `P` 或输入 shape 改变必须重新 `prepare`；本设计遵循该契约。
- 混合路径要求 `M` 的值在合法范围内非负；`M_aug` 通过 `(data_ptr, _version)` 检测
  直接对 `M` 的原地更新；通过其他别名修改同一 storage 且不更新版本号的情况不在契约内。
- 自测在单逻辑 NPU 上完成；`prepared.run` 不依赖固定 stream，双 stream 检查通过。
