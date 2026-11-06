# mhcpresinkhorn_rmsnorm

面向 Ascend 950PR（A5）的 PyPTO-Pro MIX 算子，将超连接逆 RMS、门控矩阵乘、前后置门控、Sinkhorn 归一化和 RmsNorm 融合在一个 Kernel 中。矩阵乘按 K 维分给 28 个 Cube 核，并与 Vector 侧计算形成流水。

[English README](README_en.md)

## 运行要求

- Ascend 950PR（A5）NPU 和兼容的 CANN 环境；原开发环境使用 CANN 9.2.0。
- 安装兼容版本的 PyTorch、`torch_npu` 和 PyPTO-Pro。
- 将 `PTO_TILE_LIB_CODE_PATH` 指向本地 `pto-isa` 检出目录，供 PyPTO-Pro 编译；本目录不包含该依赖。
- 可通过 `TILE_FWK_DEVICE_ID` 指定目标 NPU 编号，默认 `0`。

运行前需加载当前 CANN 安装的环境。首次调用会触发 JIT 编译；本目录不提供 CPU 执行路径或预编译 Kernel。

## 公开接口

入口为 `mhcpresinkhorn_rmsnorm(x, hc_fn, hc_scale, hc_base, gamma, hc_mult=4, sinkhorn_iters=20, eps=1e-6, rms_eps=1e-6)`。

| 输入 | 形状 | 类型 |
| --- | --- | --- |
| `x` | `[t,4,4096]` | bf16 |
| `hc_fn` | `[24,16384]` | bf16 |
| `hc_scale` | `[3]` | fp32 |
| `hc_base` | `[24]` | fp32 |
| `gamma` | `[4096]` | bf16 |

需提供非空的 token 维度（`t >= 1`）。`x`、`hc_fn`、`gamma` 必须连续；`hc_mult`、`sinkhorn_iters` 和 `eps` 目前分别固定为 `4`、`20` 和 `1e-6`。`rms_eps` 默认为 `1e-6`。包装函数根据 `TILE_FWK_DEVICE_ID` 选择 NPU 并将输入移至该设备。

返回 `(y, rstd, post, comb)`，形状及类型依次为 `[t,4096]` bf16、`[t,1]` fp32、`[t,4]` fp32、`[t,4,4]` fp32。**`y` 已完成 RmsNorm**，不另外返回未经归一化的 bf16 中间结果。

## 运行示例

配置好昇腾环境后，从 xllm 仓库根目录执行：

```bash
TILE_FWK_DEVICE_ID=0 PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa \
  python3 xllm/python/kernels_npu/pypto-pro/mhcpresinkhorn_rmsnorm/examples/example.py 3
```

末尾参数为可选的 `t`（默认 `3`）。示例构造输入，并用 PyTorch 参考实现比较四个输出；执行需要可用的 NPU 环境。

## 更多文档

- [中文设计说明](docs/design_zh.md) / [English design](docs/design_en.md)
- [中文接口说明](docs/api_reference_zh.md) / [English API reference](docs/api_reference_en.md)
- [可运行示例](examples/example.py)