<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. See LICENSE. -->

# 设计说明

对任意层 l、行 r、列 c，令 q = X[r,c]。若 q < 0，输出 -1；否则输出 M[l,q]。
一维 M 使用单层语义并返回 [R,C]；二维 M 返回 [L,R,C]。
所有映射值保留 INT32，物理地址不经过浮点转换，支持 INT32_MAX。

## 准备和执行

prepare 校验输入数、空 attributes、INT32、同一 NPU、连续布局、维数、
B/L 下界、输入及输出元素数上限。一次性回读检查 M 非负和 X 的非负地址 < B。
prepare 不缓存翻译结果、有效位置掩码或映射值。后续调用直接读取所持有张量，
允许 X/M 在合法范围内更新；shape、dtype、device 和合法值域由调用方保持。

B <= 4096 时，使用 Triton 融合核。输出少于 32768 个元素时每个 program
处理一层的 256 个连续位置，其余使用 512 个位置，连续读取该层映射表到局部向量，
以 tl.gather 完成查表，统一写入负值哨兵。该环境的 Gather 不接受 INT32 lane，
因此查表前后以 bitcast 将 INT32 位模式重解释为 FP32 再还原，不进行浮点数值转换
或浮点运算。INT32_MAX 等地址的 32 位内容不变。尾部使用显式掩码。
表长补齐到 2 的幂，补齐元素为 0；合法正索引不会访问补齐位置。
输入的非零 storage_offset 由 Tensor 传入的数据指针自然处理。

B > 4096 时使用 Triton 将 X 转换为非负 INT64 索引及布尔无效掩码，随后进入
C++ 扩展，用 ATen index_select 分派到原生 NPU Gather，并原地填充新输出的
负位置。C++ 不解引用设备指针，不在 CPU 上计算翻译结果。
分派只依赖真实规模，不依赖用例 ID、数据值、测试模式或计时状态。
prepare 完成首次编译并丢弃预热输出，只保留已编译的启动器。run 直接调用启动器，
避免重复 JIT 分派和参数特化；启动器每次读取当前设备的 stream，未缓存 stream。

## stream、存储与生命周期

run 在输入设备的调用时当前 NPU stream 上执行，不缓存 stream 或设备地址。
不同 stream 调用由调用方建立输入写入依赖；实现不插入设备同步。
每次输出和临时张量重新分配，无跨调用共享 scratch。
prepared 持有原始 Tensor，直到对象释放。调用方应按 PyTorch 跨 stream 规则
保留输入、输出并建立 wait/event 依赖，避免与未同步的输入写入并发。

小分支仅分配输出。大分支分配输出、N 个 INT64 索引、N 个布尔掩码；
原生 NPU 算子可能额外分配内部临时区，须由预热后的分配器峰值实测记录。
R 或 C 为 0 时返回正确形状的新空 Tensor，不发射零大小 kernel。

## 验收方法

使用原始 generate_data 与 golden.calculate 生成独立 CPU 标杆，原始 golden.py
验证基本功能与工程行为；补充检查直接 run、逐份双 stream CPU 比对和基线回收。
性能使用完整 prepared.run 入口，基线通过 golden.reference('baseline') 获取。
统计使用 6 个配对轮次和共同重采样索引，10000 次 bootstrap，seed=20261031，
2.5%/97.5% 线性插值分位数。均值、中位数收益和置信区间下界均须大于 0，
小/大主用例均值分别不超过 75/60 微秒。

参考：[Triton Ascend 官方文档](https://triton-ascend.readthedocs.io/zh-cn/latest/)、
[xLLM 目标算子目录](https://github.com/xLLM-AI/xllm/tree/main/xllm/core/kernels/npu/xllm_ops)。
