---
title: "T5 QKV 拆分与缓存追加算子设计"
---
<!--
Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## 概述

本文描述 Ascend NPU 上 T5 单 token 解码场景的 QKV 拆分与 KV Cache 追加算子。算子从融合的当前 token QKV 中拆出 Q，并把当前 K/V 分别追加到历史缓存，一次内核执行产生三个相互独立的连续输出。

设计目标如下：

- FP16/BF16 数据按 16 位原始位模式搬运，不执行浮点计算或类型转换。
- 支持空历史缓存、连续存储偏移视图及非 32 字节对齐的行尾。
- ACLNN 两阶段接口、Kernel Launch 直调和 PyTorch 入口共用同一套地址与分块逻辑。
- 每次调用独立分配输出，显式不使用全局 workspace，并保证跨 stream 输入生命周期安全。

本文是设计评审材料，不代替实现、精度报告或性能验收报告。固定门槛是否通过应以后续代码 PR 中的可复现实测结果为准。

## 1. 接口与语义

令 `S=H×D`，接口逻辑形状为：

| 名称 | 形状 | 说明 |
| --- | --- | --- |
| `qkv` | `[B, 1, 3S]` | 当前 token，按全部 Q、全部 K、全部 V 排列 |
| `past_key` | `[B, H, L, D]` | 历史 K 缓存 |
| `past_value` | `[B, H, L, D]` | 历史 V 缓存 |
| `query` | `[B, H, 1, D]` | 独立 Q 输出 |
| `key` | `[B, H, L+1, D]` | 历史 K 后追加当前 K |
| `value` | `[B, H, L+1, D]` | 历史 V 后追加当前 V |

`headNum=H`、`headDim=D` 作为原生算子属性；框架入口可以从形状推导，但显式提供的属性必须与形状一致。输入必须为同一 NPU 设备、同一 dtype 的连续 ND 张量。`B/H/L/D` 的范围以及尺寸乘积在 Host 侧统一检查，非法元数据在启动内核前报错。

三个输出均通过独立分配获得，不与输入或彼此共享存储。算子不原地修改输入，不缓存上一调用的输出，也不使用 CPU 代算。

## 2. 地址推导

以下地址均以元素为单位，`b∈[0,B)`、`h∈[0,H)`、`d∈[0,D)`。

Q 输出的线性地址为：

```text
q_out = (b×H + h)×D + d
q_src = b×3S + h×D + d
```

K/V 输出的线性地址为：

```text
kv_out = ((b×H + h)×(L+1) + t)×D + d
```

当 `t<L` 时，来源为历史缓存：

```text
kv_src = ((b×H + h)×L + t)×D + d
```

当 `t=L` 时，来源为当前融合 QKV。K、V 分别使用 `s=1`、`s=2`：

```text
append_src = (b×3 + s)×S + h×D + d
```

该推导把每个输出位置映射到唯一输入区间。分块实现必须保持这一映射，而不能依赖浮点数值比较验证地址正确性。

## 3. 位模式与边界处理

FP16/BF16 均解释为 `uint16_t` 搬运，因此 NaN 的 payload、Inf、正负零和非正规数不会发生数值转换或静默位变化。

`L=0` 时，K/V 只从当前 QKV 生成，不读取空历史张量的存储。连续存储偏移视图通过已经包含 `storage_offset` 的数据指针描述，ACL tensor 内部偏移保持为 0，避免重复叠加。

普通 DMA 路径要求片段长度及真实源/目的地址满足 32 字节对齐。对齐判断必须包含 GM 基地址，不能仅根据逻辑索引推断。未对齐片段使用带边界参数的搬运路径，既不补读输入末尾，也不把 padding 写出到相邻输出区间。

## 4. 分核与片上缓冲

### 4.1 工作划分

Q、K、V 分别划分为 32768 元素（64 KiB）的连续 tile，工作编号按 Q、K、V 排列。最多 48 个 AIV 以 `blockIdx + n×blockNum` 的方式领取工作；小输入只启动实际有工作的核数。

tile 内再根据 batch、历史行和追加 token 边界拆成 DMA 片段。对于大且对齐的缓存，可以采用按 `[B,H]` 行独立分块的策略，避免追加 token 导致的行偏移在每个全局 tile 上产生额外跨界片段。行内最后一个工作项应同时处理行尾历史数据和追加 token，避免为每个小 token 单独制造负载不均的工作项。

每个输出位置只能由一个工作项负责。即使一个逻辑行跨越多个工作项，也只能按完整的 32 字节区块划分所有权，防止不同核竞争同一对齐区块。

### 4.2 UB 与 workspace

每个 AIV 使用 `TQueBind<VECIN, VECOUT, 2>` 管理两份 64 KiB UB 缓冲，形成 128 KiB 的双缓冲队列。GM→UB 与 UB→GM 搬运通过同一队列同步，标量状态只保存少量形状、tile 和偏移信息。

算子显式设置 `workspace=0`。ACLNN 调用方在 `GetWorkspaceSize` 后再次检查结果，若不是 0 则报错，而不是隐式分配额外全局内存。

## 5. Host、框架与 stream 设计

ACLNN 入口使用标准两阶段调用：创建六个 tensor 描述，调用 `GetWorkspaceSize`，再在调用方当前 stream 上提交执行。Kernel Launch 直调入口复用相同的 plan 与设备内核，用于交叉检查 ACLNN 路径。

PyTorch 入口遵循以下顺序：

1. 校验输入、属性和形状，并使用输入设备 guard。
2. 通过三次独立 `at::empty` 分配输出。
3. 获取输入设备上的当前 NPU stream，并按 torch_npu 的提交语义建立与先前操作的顺序。
4. 对可能被其他 stream 使用的输入执行 `recordStream`，随后提交 ACLNN 或直调内核。

输出在当前 stream 上分配，其复用由分配 stream 的操作顺序保证，不再为同 stream 输出添加多余回收事件。输入的跨 stream 登记仍保留，以覆盖调用返回后输入提前释放的场景。

`prepare` 只保存 shape、dtype、device 和属性等元数据，不保存输入张量或固定 stream。后续每次调用都重新校验元数据、获取当前 stream 并分配新输出。

## 6. 资源预算与整数安全

Host 使用 64 位整数计算形状乘积，并在进入内核前检查范围。设备侧只有在所有线性索引及乘积均被证明有界后才使用 32 位索引，避免小索引路径引入溢出。

除三个输出外，算子不分配全局结果缓存或临时工作区。输出总字节数为：

```text
2×B×H×D×(2L+3)
```

该公式分别覆盖 Q 的 `BHD` 元素以及 K/V 的 `2×BHD×(L+1)` 元素，并包含 16 位元素大小。内存验证还应单独记录框架分配器开销，不能把它混入算子 workspace。

## 7. 验证设计

实现 PR 应提供以下可复现验证，而不是仅覆盖常规随机输入：

- 全部公开 shape 的 ACLNN 与直调逐位对照，以及 FP16/BF16 特殊位模式检查。
- 空缓存、奇数维度、非对齐行尾、连续存储偏移视图和连续两次缓存追加。
- 非法属性、dtype、设备、rank、shape、连续性和内部格式拒绝检查。
- 双 stream 调用、输入提前释放、输出持续引用及重复释放后的活跃内存恢复。
- CPU 地址模型对全部 tile/片段进行覆盖、边界和源地址检查；该模型只验证地址算法，不替代 NPU 执行。
- 固定性能协议为 50 次预热、5 轮测试、每轮 200 次调用，报告每轮 API 到完成的 wall time 及其中位数，并保留原始日志。

只有实际执行并通过的项目才能标记为通过。环境版本差异、未执行项目及未达门槛的指标必须明确披露。
