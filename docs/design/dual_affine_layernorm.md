<!--
Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
-->

# 需求背景（required）

## 需求来源

双分支 LayerNorm 算子（DualAffineLayerNorm）任务书：在同一输入上共享一次行均值/方差计算，
再分别应用两套独立的仿射参数，产生两路归一化输出。xLLM 的 NPU 算子库需要在 Atlas A3 上
以 Ascend C 实现该融合算子，并提供 ACLNN 两阶段 API、kernel 直调入口和 PyTorch NPU 封装。

## 背景介绍

### 双分支 LayerNorm 的现状

典型实现是两次独立 LayerNorm，或先以 torch 组合算出共享统计量再做两路仿射：

```python
a = x.float()
centered = a - a.mean(dim=1, keepdim=True)
z = centered * torch.rsqrt((centered * centered).mean(dim=1, keepdim=True) + eps)
y1 = (z * g1 + b1).to(x.dtype)
y2 = (z * g2 + b2).to(x.dtype)
```

两次 LayerNorm 会把每行的均值/方差算两遍，且小 shape 下主机侧调用开销占主导；
torch 组合实现则产生多次 kernel 发射和中间张量。二者都明显慢于一次融合发射。

### 算子功能分析

对 `x[N,H]` 的每一行：

```text
mu    = mean(x)
var   = mean((x - mu)^2)          # 中心化方差，不用 E[x^2]-E[x]^2
z     = (x - mu) / sqrt(var + eps)
y1    = cast(z * gamma1 + beta1)
y2    = cast(z * gamma2 + beta2)
```

统计、归一化和仿射计算全部在 FP32 完成，最后转换为输入类型。两路输出独立分配、连续、
storage 互不相同；输入不被修改。

# 需求分析（required）

## 需求描述

使用 Ascend C 实现 DualAffineLayerNorm，支持 FP16/BF16、ND 连续输入，
`1<=N<=4096`、`1<=H<=8192`、`epsilon∈[1e-7,1e-2]`，支持非对齐 H；常量行与 H=1
必须精确输出对应 beta；显式全局 workspace 为 0。

交付物包括：ACLNN 算子工程（OpDef、shape/dtype 推导、tiling、可安装 `.run`）、两阶段
ACLNN API、kernel 直调工程、独立 C++ 样例、PyTorch NPU 接口（`run`/`run_direct`/prepared
变体）、以及覆盖公开 46 用例、额外种子/偏移、非法输入、stream、生命周期、内存、
验证器反例的自测工具。

## 需求拆解

| 工作项 | 内容 |
|---|---|
| 算子工程 | `op_host/` 注册五输入两输出、proto 推导、tiling；`op_kernel/` 共享计算类；`build.sh` 生成 `.run` |
| 直调工程 | `kernel_direct/` 显式 `__global__` 入口，复用同一计算类与 tiling 参数 |
| PyTorch 封装 | `csrc/` 扩展 + `dual_affine_layernorm/` 公共接口，prepared plan 只保存元数据 |
| 验证工具 | `tests/run_device_acceptance.py` 逐项 JSONL、指纹绑定、fail-closed 报告与打包 |

# 详细设计（required）

## 算子分析

### 数学公式

```text
mu      = (Σ x) / H
var     = (Σ (x - mu)^2) / H
z       = (x - mu) / sqrt(var + eps)
y1      = cast(z * gamma1 + beta1)
y2      = cast(z * gamma2 + beta2)
```

每行只共享一次 FP32 均值与中心化方差；不使用 `E[x^2]-E[x]^2`，避免大偏置输入
（如 `x≈1000、方差≈1e-6`）的灾难性消减。常量行 `var=0`，此时 `z=0`，输出精确等于 beta。

### 支持数据类型

FP16、BF16（tiling key 0/1）。中间统计与仿射计算为 FP32。

### 支持形状

`x:[N,H]`，`gamma1/beta1/gamma2/beta2:[H]`，`y1/y2:[N,H]`；ND 连续，支持任意 H（含非 32 字节对齐）。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

**1. 分核策略：**

48 个 Vector Core。`N >= 48` 时每核负责若干整行：

```text
blockDim    = min(N, coreNum)
rowsPerCore = ceil(N / blockDim)
blockDim    = ceil(N / rowsPerCore)        # 回算，避免 N=49 时 23 个空核
```

`N` 小于本平台 Vector Core 数（A3 为 48、A2 为 40）时把一行拆到多个核，
以便小 N、大 H 的用例也能吃满核数；`N >= coreNum` 一律走整行路径，避免 `coreNum / N == 0`：

```text
coresPerRow = max(1, min(coreNum / N, max(1, H / 512)))  # 至少 1，每片 >=512 元素
# 仅当 N < coreNum 时进入；A2 的 coreNum=40，N=41..47 走整行路径
sliceH      = ceil(H / coresPerRow) 向上取到 64 的倍数
coresPerRow = ceil(H / sliceH)
blockDim    = N * coresPerRow
row   = GetBlockIdx() / coresPerRow
start = (GetBlockIdx() % coresPerRow) * sliceH
count = min(sliceH, H - start)
```

**每个分片核都从 GM 读取整行 x 并独立重算同一份行统计**，再只处理自己的
`[start, start+count)` 分片。各核执行的指令与数据完全相同，FP32 统计量逐位一致，
因此不需要跨核归约、原子累加或 GM workspace。`H < 1024` 时 `coresPerRow=1`，
退化为整行划分。分片起点 `start` 恒为 64 的倍数；但行的基址 `row*H` 在 H 不是 16 的倍数、
或输入带 storage offset 时并不保证 32 字节对齐（例如 FP16、H=1025、row=1 的字节偏移是 2050），
因此 x/y 的搬运依赖 `DataCopyPad` 对非对齐字节基址与长度的处理，而不是依赖行首对齐；
H=1537、8191 等非对齐用例已在真实设备上通过。

N=2/H=8192 由 2 核变为 32 核后，该用例的设备时间从约 20 μs 降到约 3 μs。

**2. 数据分块和内存优化策略：**

令 `alignedH=ceil(H/64)*64`，`sliceAligned=ceil(sliceH/64)*64`，片上 UB 布局：

| 区域 | 大小 | 字节数 |
|---|---:|---:|
| x16（整行输入） | `alignedH` 个 16 位元素 | `2*alignedH` |
| g1/b1/g2/b2（分片） | `4*sliceAligned` 个 16 位元素 | `8*sliceAligned` |
| z/tmp0/tmp1（整行 FP32） | `3*alignedH` 个 FP32 元素 | `12*alignedH` |
| 合计 | — | **`14*alignedH + 8*sliceAligned`** |

`coresPerRow=1` 时退化为 `22*alignedH`，H=8192 为 180224 字节，仍在 192 KiB UB 内；
host 只校验实际 `TBuf` 分配，不额外保留未使用的 16 KiB。显式 workspace 恒为 0。

输出复用 `tmp1.ReinterpretCast<T>()`，不与下一行的 `x16` 输入别名，消除逐行循环中
上一行 CopyOut 与下一行 CopyIn 的 WAR 竞争。`DataCopyPad`/`DataCopyExtParams`
处理非 32 字节对齐的 H。

**3. tilingkey规划策略：**

| tiling key | dtype |
|---|---|
| 0 | FP16 |
| 1 | BF16 |

`DualAffineLayerNormTilingData` 携带 `n/h/alignedH/rowsPerCore/ubBytes/blockDim/coresPerRow/sliceH/epsilon`，
Python tiling、host tiling、直调 launcher 三处使用同一公式。

**4. 主机侧调用路径（ACLNN）：**

ACLNN 两阶段 API 的 executor 按严格地址键复用：键包含七个数据指针、N、H、dtype、device 和
epsilon，任一变化立即重建；命中时只调用执行 API，省掉 `GetWorkspaceSize`、描述符创建和
executor 销毁。缓存在进程内为 4 槽表并受互斥锁保护，只保存 ACL 句柄，不保存 torch tensor；
`aclSetAclOpExecutorRepeatable` 通过 `dlopen` 显式解析，缺失时自动退回逐调用路径。
输出分配直接走 `at::detail::empty_generic` + NPU caching allocator，省掉 dispatcher。

#### 3.2.2 kernel侧设计：

每个分片核的执行流程：

1. 读取整行 `x`（`DataCopyPad`），加载本分片的四组仿射参数；一次 `MTE2_V` 同步。
2. `Cast` 到 FP32 → `ReduceSum` 求行和 → 标量读出均值。
3. `Adds(z, z, -mean)` → `Mul(z,z)` → `ReduceSum` 求中心化平方和 → 标量读出方差。
4. 计算 `rstd = 1/sqrt(var+eps)` → `Muls(z, z, rstd)`。
5. 两路分支各自 `Cast(gamma)/Cast(beta)` → `Mul` → `Add` → `Cast` 回 T，写入 `tmp1` 的
   T 视图并 `CopyOut` 到 `row*H + start`。

同步只发生在核内，且与代码逐项对应（`ReadScalar` 内部就是 `SetFlag<V_S>` + `WaitFlag<V_S>`）：

| 时机 | 事件 | 作用 |
|---|---|---|
| 每行 `x` 与参数 `CopyIn` 之后 | `MTE2_V` ×1 | 等 GM→UB 搬运完成再做 `Cast` |
| 首次 `ReduceSum`（行和）之后 | `V_S` ×1 | 等归约完成再读标量求 `mean` |
| 第二次 `ReduceSum`（居中平方和）之后 | `V_S` ×1 | 等归约完成再读标量求 `variance` |
| 写入 `variance + eps` 之后 | `S_V` ×1 | 等标量写入可见再做向量 `Sqrt`/`Div` |
| `Div` 得到 `rstd` 之后 | `V_S` ×1 | 等向量计算完成再读标量 |
| 每个分支的 `CopyOut` 之前/之后 | `V_MTE3` + `MTE3_V` 各 ×1 | 等向量写完成再发 DMA；等 DMA 读完才允许复用 `tmp1` |

向量内部依赖另有 `PipeBarrier<PIPE_V>`。两路输出共用 `tmp1` 视图，所以 `V_MTE3`/`MTE3_V`
是**每个 CopyOut 各一组**，而不是整行一组。

多行复用（`rowsPerCore > 1`，例如 A3 上 N=96）不需要额外的 `V_MTE2`：`x16` 只在行首被 `Cast`
读取一次，而行内最后一个 `CopyOut` 已经在发 DMA 前执行了 `WaitFlag<V_MTE3>`——该 flag 覆盖此前
**所有**向量指令（包括读 `x16` 的 `Cast`），标量核在它返回后才进入下一轮循环，因此下一次
`CopyIn` 覆盖 `x16` 时上一行的读取早已完成；四组仿射参数在每核 `Init` 后只加载一次，行循环
不会重写它们。若将来改成逐行重载参数或双缓冲，需要相应补 `V_MTE2` / 双缓冲同步。

无跨核同步、无 workspace。

## 支持硬件

- Atlas A3：`Ascend910_9382`（48 Vector Core / 24 Cube Core），CANN 9.1.0
- Atlas A2 兼容：`ascend910b`（CANN 9.0 兼容构建）

## 算子约束限制

- 输入必须为 ND 连续、同设备、同 dtype、有限值；拒绝空输入、错误 rank/dtype/length、
  非连续、非法 epsilon 与未知属性。
- 值域契约（任务书 2.3）：`abs(x) <= 1e4`，`abs(gamma), abs(beta) <= 8`。FP32 两遍中心化
  统计在该范围内不会溢出：行和 `<= H*1e4 = 8.192e7`，居中平方和
  `<= H*(2e4)^2 = 3.277e12`，远小于 `FLT_MAX = 3.4e38`；因此 FP16/BF16 常量行都能精确
  输出 beta。超出该值域的输入（例如贴近 `max_bfloat16` 的数据）不在本算子契约内，需要
  调用方先做缩放。
- 两路输出分别 `at::empty` 独立分配，storage 互不相同且不与任何输入重叠。
- 不支持反向与训练态。

# 可维可测分析

## 精度标准/性能标准

- 精度：CPU FP64 golden；FP16 `atol=rtol=2e-3`，BF16 `atol=rtol=2e-2`；常量行与 H=1
  精确输出 beta。公开 46 用例 + 12 个额外种子/偏移用例 + 20 个非法场景。
- 性能：官方 50 次预热、5 轮 × 每轮 200 次，wall 轮均中位数；3.3 节目标 22/21/20/55/20/
  20/20/20/55/20 μs。

在 Ascend910_9382 / CANN 9.1.0 上（torch 2.9.0 兼容栈）连续三次全组验收：

| 用例 | 目标 | 第1次 | 第2次 | 第3次 |
|---|---:|---:|---:|---:|
| 005 | 22 | 16.60 | 15.31 | 15.80 |
| 006 | 21 | 16.37 | 15.94 | 15.65 |
| 007 | 20 | 18.11 | 16.82 | 16.17 |
| 008 | 55 | 39.11 | 39.01 | 39.26 |
| 009 | 20 | 17.61 | 15.99 | 16.47 |
| 020 | 20 | 17.15 | 15.70 | 15.52 |
| 021 | 20 | 16.54 | 15.75 | 15.92 |
| 022 | 20 | 17.39 | 16.31 | 15.99 |
| 023 | 55 | 39.92 | 40.15 | 40.56 |
| 024 | 20 | 17.11 | 17.58 | 15.88 |

精度 46/46，额外 24/24，非法 60/60，stream 2/2，生命周期 2/2，跨 stream 压力 2/2，
内存 20/20（workspace=0），验证器反例 10/10。

## 兼容性分析

- CANN 9.0 / 9.1：同一份 Ascend C 与 host 代码，standalone `.run` 分别以 `ascend910b`、
  `ascend910_93` 构建通过。
- PyTorch：torch 2.9.0+cpu / 2.10.0+cpu，torch_npu 2.9.0 / 2.10.0.post4（git 5dd8ef3）。
- 驱动：25.5.t6.b001 及以上（高版本驱动向下兼容）。
- 非对齐 H、storage offset 输入、多 stream、提前释放输入均已覆盖。
