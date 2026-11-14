# mhcpresinkhorn_rmsnorm（v4-K + RmsNorm 融合）接口说明

## 1. 公开接口

从 xllm 仓库根目录执行[示例](../examples/example.py)，了解输入构造、模块导入与四个输出的 PyTorch 参考值对比。由于 `pypto-pro` 含连字符，示例通过将算子所在目录加入 `sys.path` 来导入本地模块。

```python
from mhcpresinkhorn_rmsnorm import mhcpresinkhorn_rmsnorm

y, rstd, post, comb = mhcpresinkhorn_rmsnorm(
    x, hc_fn, hc_scale, hc_base, gamma,
    hc_mult=4, sinkhorn_iters=20, eps=1e-6, rms_eps=1e-6,
)
```

### 1.1 输入

| 参数 | 形状及类型 | 含义及约束 |
| --- | --- | --- |
| `x` | `[t,4,4096]` bf16 | 超连接输入，对 hc 和 d 维计算逆 RMS；要求连续，`t >= 1` |
| `hc_fn` | `[24,16384]` bf16 | 门控矩阵权重，24 路为 4 pre + 4 post + 16 comb；要求连续。权重应在模型侧转换为 bf16，算子直接使用 BF16×BF16→FP32 矩阵乘 |
| `hc_scale` | `[3]` fp32 | 三条路径的 `[s_pre,s_post,s_comb]` 缩放值 |
| `hc_base` | `[24]` fp32 | 三条路径各自的偏置，分别为 `[0:4]`、`[4:8]`、`[8:24]` |
| `gamma` | `[4096]` bf16 | RmsNorm 每维的缩放权重；要求连续 |
| `hc_mult` | 整数，默认 4 | 超连接倍数，当前仅支持 4，否则报错 |
| `sinkhorn_iters` | 整数，默认 20 | Sinkhorn 迭代次数，当前仅支持 20，否则报错 |
| `eps` | 浮点数，默认 `1e-6` | 门控及 Sinkhorn 的稳定常数，当前仅支持 `1e-6`，否则报错 |
| `rms_eps` | 浮点数，默认 `1e-6` | RmsNorm 中 `rstd = 1/sqrt(mean(y²)+rms_eps)` 的常数 |

包装函数检查输入形状与类型及 `x`、`hc_fn`、`gamma` 的连续性；随后依据 `TILE_FWK_DEVICE_ID` 将输入移至相应 NPU。实现未显式检查 `t >= 1` 或所有输入是否在同一设备，调用者应提供非空 token 维度及合适的输入。

### 1.2 输出

| 返回值 | 形状及类型 | 含义 |
| --- | --- | --- |
| `y` | `[t,4096]` bf16 | 已经完成 RmsNorm 的 `y_norm = y_bf16*rstd*gamma`；不返回未经归一化的中间 `y_bf16` |
| `rstd` | `[t,1]` fp32 | 根据 bf16 中间 `y` 计算的 RmsNorm 逆标准差，供反向使用 |
| `post` | `[t,4]` fp32 | 后置门控 `sigmoid(z)*2` |
| `comb` | `[t,4,4]` fp32 | Sinkhorn 归一化后的组合矩阵 |

## 2. 内部常量与切分

| 常量 | 数值 | 含义 |
| --- | --- | --- |
| `HC` / `D` / `HC_D` | 4 / 4096 / 16384 | 超连接倍数、隐藏维度及矩阵乘 K 维 |
| `MIX_HC` | 24 | 门控通道数：4 pre + 4 post + 16 comb |
| `SINKHORN_ITERS` / `EPS` / `RMS_EPS` | 20 / `1e-6` / `1e-6` | 默认迭代次数和数值常数 |
| `NB` / `TG` | 28 / 16 | MIX 逻辑块数 / 每组 token 数 |
| `MIXC` | 32 | 门控矩阵乘 tile 宽度（有效通道为 24） |
| `DC` / `NDC` | 512 / 8 | y 数据流分片宽度 / 分片次数 |
| `SSQ_SLICES` / `SSQ_DK` | 32 / 512 | 输入平方和的切片数 / 每片宽度 |
| `KA` / `KLAST` | 592 / 400 | 前 27 个 AIC 与最后一个 AIC 的 K 分片宽度 |

K 维覆盖为 `27*592+400=16384`，没有重叠或未覆盖片段。

## 3. 直接调用 Kernel（高级用法）

一般应使用公开包装函数；直接调用需要自行分配输出和工作区，并保证工作区 token 维按 `ceil(t/16)*16` 对齐。包装函数中的 `_get_workspaces` 会按形状和设备缓存这些工作区。

```python
mhcpresinkhorn_rmsnorm_kernel_k[None, 28](
    x_2d,      # [t,16384] bf16，x 的二维视图
    x_3d,      # [t,4,4096] bf16，原始 x
    hc_fn,     # [24,16384] bf16
    hc_scale,  # [1,3] fp32 视图
    hc_base,   # [1,24] fp32 视图
    gamma,     # [1,4096] bf16 视图
    y,         # [t,4096] bf16 输出
    rstd,      # [t,1] fp32 输出
    post,      # [t,4] fp32 输出
    comb16,    # [t,16] fp32，comb 的平铺视图
    pmix3,     # [28,ceil(t/16)*16,32] fp32 工作区
    pmix3f,    # pmix3 的二维视图
    ssq_ws,    # [32,ceil(t/16)*16] fp32 工作区
)
```

## 4. 运行环境

| 环境变量 | 用途 |
| --- | --- |
| `TILE_FWK_DEVICE_ID` | 目标 NPU 设备编号；默认 0 |
| `PTO_TILE_LIB_CODE_PATH` | 本地 `pto-isa` 检出目录，供 PyPTO-Pro 编译使用；本目录不包含该依赖 |

原开发环境使用 CANN 9.2.0，并通过 `/usr/local/Ascend/ascend-toolkit/set_env.sh` 加载环境；实际安装路径请以当前设备为准。示例调用会触发 JIT 编译，须安装兼容版本的 PyTorch、`torch_npu`、PyPTO-Pro 和 CANN。更多实现信息见[中文版设计文档](design_zh.md)。