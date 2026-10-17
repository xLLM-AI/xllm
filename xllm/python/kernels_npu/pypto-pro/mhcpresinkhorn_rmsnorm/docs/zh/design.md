# mhcpresinkhorn_rmsnorm（v4-K + RmsNorm 融合）设计说明

本实现以一个 PyPTO-Pro MIX Kernel 融合生产链路的 HcPreInvRms → Cast → MatMulV3 → Cast → HcPreSinkhorn → RmsNorm。目标平台为 Ascend 950PR（A5），原设计环境为 CANN 9.2.0、PyPTO-Pro JIT。

## 1. 算子语义

令 `x` 的形状为 `[t,4,4096]`（bf16），`hc_fn` 为 `[24,16384]`（bf16）：

```text
r[t,1]           = rsqrt(mean(float(x)^2, dims=(hc,d)) + eps)
mixes[t,24]      = reshape(x,[t,16384]) @ hc_fn.T     # BF16×BF16→FP32
res[t,24]        = mixes * r
pre[t,4]         = sigmoid(res[:,0:4]*s0 + base[0:4]) + eps
post[t,4]        = sigmoid(res[:,4:8]*s1 + base[4:8]) * 2
comb_flag[t,4,4] = reshape(res[:,8:24]*s2 + base[8:24],[t,4,4])
y_bf16[t,4096]   = bf16(sum_m pre[m]*float(x[t,m,:]))
comb             = Sinkhorn(comb_flag, 20 轮行列交替归一化)
rstd[t,1]        = rsqrt(mean(float(y_bf16)^2) + rms_eps)
y_norm[t,4096]   = bf16(float(y_bf16)*rstd*float(gamma))
```

RmsNorm 的平方和针对**已经舍入为 bf16 的中间结果**计算；公开输出 `y` 是归一化后的 `y_norm`，不单独返回原始 `y_bf16`。`comb` 在首轮进行稳定 softmax，后续按行、列归一化，详见[接口说明](api_reference_zh.md)。

## 2. MIX 核与流水

使用 28 个 MIX 逻辑块（28 AIC + 56 AIV），每组 16 个 token 仅做一次 `sync_all(MIX)`：

```text
AIC：预加载权重分片 → 每组加载 x → 分 K 矩阵乘 → 写入部分结果 → MIX 同步
AIV：预加载辅助参数和 gamma → 并行计算 x 平方和 → 等待 MIX 同步
     → 跨 28 个部分结果归约 → 求 r 和门控
     → y 第一遍（8×512，保留 bf16 中间值并累加平方和）
     → rstd → y 第二遍（融合 gamma 的归一化）→ Sinkhorn → 写回
```

Cube 从 GM 直接读取 bf16 的 `x` 和 `hc_fn`，与 Vector 计算输入平方和并行。同步后，Cube 的下一 token 组矩阵乘可与 Vector 的当前组后处理重叠。中间 `y` 留在 UB，省去向 GM 写回再读取以及独立的 RmsNorm Kernel 启动。

## 3. Cube：K 维切分

`K=16384` 被分给 28 个 AIC：核 0–26 的起点为 `core*592`，各处理 592 个元素；核 27 处理末尾 400 个元素，`27*592+400=16384`。两个分片均为 16 的整数倍，分别使用 `[16,592]×[592,32]` 与 `[16,400]×[400,32]` 的 tile，无需补零或尾块。权重分片对同一核的各 token 组保持不变，预加载一次。28 份 FP32 部分乘积写入工作区 `pmix3[28, ceil(t/16)*16, 32]`，由 Vector 侧归约为 24 个有效门控通道。

Mat tile 不支持 `fillpad`；原设计亦指出 Cube 段无法借助所需 GM→L1 barrier 安全地实现补零方案，因此采用上述两种 tile 形状。

## 4. Vector：归约、门控与融合 RmsNorm

### 4.1 归约及门控

`_gates1_vf` 对 28 路矩阵乘部分结果和 32 路输入平方和分片分别求和；用 `scale24`、`pmul24`、`eps24` 模式向量及前缀掩码并行处理三类门控：`pre` 为 sigmoid 后加 `eps`，`post` 为 sigmoid 后乘 2，`comb_flag` 走线性分支。

### 4.2 两遍 y 数据流

| 阶段 | 实现 | 作用 |
| --- | --- | --- |
| 第一遍，8 个 512 维片段 | `_y_pass1_vf` | 展开 4 条 hc 行的 FMA，bf16 中间 `y` 保存在 UB，同时累计 `y` 的平方和 |
| rstd | `_rstd_vf` | 对平方和归约，计算 `rsqrt(mean(y_bf16²)+rms_eps)` 并写回 `[t,1]` fp32 |
| 第二遍，8 个 512 维片段 | `_y_norm_vf` | 从 UB 读取中间 `y` 和预加载的 `gamma`，生成并写回 bf16 `y_norm` |

`x` 子块使用双缓冲；加载下一片段可与上一片段的 VF 计算重叠。`gamma` 每个 AIV 预加载一次。

### 4.3 Sinkhorn

`_sinkhorn_vf` 将 4×4 门控矩阵展开为寄存器值，先做减去行最大值的稳定 softmax，再完成规定的行列交替归一化；输出 `comb` 和 `post` 写回 GM。

## 5. 存储及平台约束

- UB 存放输入平方和片段、中间 `y`、`gamma`、双缓冲 `x` 子块、归一化输出、门控归约和 Sinkhorn 临时值；相邻 tile 槽位保持足够间隔，避免 `load_align` 跨槽读取。
- L1 使用分别适配 K=592 和 K=400 的源 tile；L0B 最大占用约 63.5 KB（64 KB 上限）。
- ZN/Acc tile 的 N 必须按 16 宽内框对齐，24 个有效门控通道使用 N=32。
- Vec tile 仅在 Vector 段使用；`.next()` 在 Kernel 主体轮换槽位，不在 VF 中调用。
- 跨核同步使用 `sync_all(MIX)`，而非原设计中发现存在计数器共享问题的 INTRA_BLOCK 事件。

## 6. 演进与历史测量

早期实现从纯 Vector、Cube+Vector、流水版本演进为 K 分片与双缓冲，再融合 RmsNorm。原设计记录的融合版本在 `t=1` / `t=13` 时约为 7.98 / 10.44 µs，属于**历史测量数据，未在当前环境复测**；不能将其当作本仓库的性能保证或当前测试结果。

参数及返回值参见[中文版接口说明](api_reference_zh.md)，执行方式参见[示例](../examples/example.py)。