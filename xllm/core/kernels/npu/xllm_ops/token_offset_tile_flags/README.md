<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# 题目10：token-offset tile标志算子

本工程以 `official/任务书.md` 为最高依据。发布包原件未修改，封存在 `official.zip`；验证入口自动解压至忽略跟踪的 `official/`。

## 当前状态

- 已实现 `solution.py`、Triton 融合核和一次性输入校验。
- 已实现官方精度、非法输入、性能和内存验收入口 `verify.py`。
- CPU 内核逻辑模拟：14 个公开输入、350 个随机输入通过；值域校验和配对统计通过。
- 提交版的 A3 / CANN 9.1.0 实测结果见随附 `result.json` 与 `verification/`，两项主性能均要求不超过 85µs。
- 三份 Excel、三份日志、步骤说明和机器可读证据随工程交付；源码与验收报告通过本 GitHub PR 提交。

## 环境

任务书要求 Atlas A3 单个逻辑 NPU、CANN 9.1.0，以及匹配的 Python、PyTorch、torch_npu、
triton-ascend。另需 NumPy（统计的 percentile(method='linear') 需要 NumPy≥1.22）。
应使用发布方提供或匹配 CANN 的软件镜像；不要从通用 PyPI 安装 CUDA Triton 替代 triton-ascend。
本工程不自动升级共享环境、不安装驱动、不替换 CANN。

## 启动

在 A3 环境解压本工程，按实际安装位置加载 CANN，并设置环境分配的运行时可见设备编号：

```bash
source /实际安装路径/cann-9.1.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=获分配的设备编号
cd xllm/core/kernels/npu/xllm_ops/token_offset_tile_flags
python3 -c 'import torch,torch_npu,triton,numpy; print(torch.__version__,torch_npu.__version__,triton.__version__); print(torch.npu.get_device_name(0))'
python3 run_guard.py
```

程序使用可见逻辑设备 0；prepare 阶段完成 Triton 编译，不需要 ACLNN 构建步骤。
在共享设备上先核对 `npu-smi info` 并遵守该机器的锁约定，避免并发测试污染性能。
每次运行新建 `results/npu/<UTC时间>/`，返回非零代表失败、阻塞或环境证据不完整。

无 NPU 的开发检查：

```bash
python3 offline_checks.py
```

该命令仅验证 NumPy 模拟的实际内核函数体、值域校验与统计，不能代替真机验收。

## 结果复核

`result.json` 含基本精度、非法输入、工程记录、两项主性能 benchmark、环境、源码哈希及错误。
`generalization.json` 含 4 个补充用例和 T=255/257 的配对计时。
`engineering_supplement.json` 含两组输入双流和分配器检查。
`raw.log` 按实际执行顺序逐项刷新，保留全部逐轮耗时。
环境自动识别不充分会返回 ENVIRONMENT_UNVERIFIED；需核实真实设备和生效 CANN 路径，
不能仅修改报告中的版本字符串来绕过环境检查。

## 验收交付

遵照任务书第 4、5 节，最终提交设计文档、用例与测试代码、真实自测报告、完整工程。
任务书未要求单独先提设计文档 PR。全部规定检查和性能判据通过后，将材料放入：

```text
task_submission/
  1 自验证步骤说明.md
  2.1 精度自验证报告.xlsx
  2.2 精度自验证日志.log
  3.1 性能自验证报告.xlsx
  3.2 性能自验证日志.log
  4.1 内存自验证报告.xlsx
  4.2 内存自验证日志.log
```

打包 task_submission.zip，并保留上述机器可读结果、环境、源码版本、原始日志。
向 https://github.com/xLLM-AI/xllm/tree/main/xllm/core/kernels/npu/xllm_ops 提交 PR，
随附设计文档、代码、测试和自验证材料，并注明个人仓库地址、分支、目录和固定 commit。
本次任务书未提供审核人、平台验收按钮或审批时限；本工程不推定这些流程。
