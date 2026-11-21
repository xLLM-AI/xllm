<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# TokenOffsetTileFlags 设计文档

## 依据与范围

最高依据为本次发布包的《题目10：token-offset tile标志算子开发任务书》。
`official.zip` 保存该任务书及其引用的 case.json、op.json、generate_data.py、golden.py 的原始副本。
既往其他任务的环境、报告结论和验收流程不适用于本工程。
只实现 tile 布局标志；不接入 Attention 快路径，不加载模型，不提供 ACLNN 安装包。

## 接口与校验

`solution.prepare(inputs, attributes)` 返回具有 `run()` 的 Prepared 对象。
`solution.run(inputs, attributes)` 完成 prepare 和一次执行。
四个输入均须是同一 NPU 上连续的一维 INT32 Tensor，允许非零 storage_offset。
属性仅接受 tile_size=T 和 max_tiles_per_seq=K；拒绝 bool、未知键和缺失键。
prepare 校验 T∈[1,4096]、K≥1、输入/输出元素上限、分段边界、有效长度、非负 offsets、
tile 容量以及稠密 tileptr。一次性 CPU 回读仅用于校验，不计算输出，不保留 CPU 输入缓存。
输入非法时抛出 TypeError 或 ValueError。

shape、dtype、device、T、K 改变须重新 prepare；原 Tensor 的合法值更新不需要重新 prepare。
调用方须保证更新后的值仍符合契约，并处理跨 stream 的生产者/消费者同步。

## 融合算法

输出索引 q∈[0,RK)，请求 r=q//K，tile 编号 t=q%K，段内起点 b=tT。
完整性条件为 b+T≤lengths[r]，起地址 a=indptr[r]+b。
仅完整 tile 读取 offsets[a]，记为 first；所有有效 lane i∈[0,T) 检查
`offsets[a+i]-first == i`，并归约不匹配个数。输出为：

```
full && first % T == 0 && mismatches == 0
```

一个 Triton program 处理 GROUP 个 tile，lane 维为 next_power_of_2(T)。
通用 GROUP=min(32,max(1,1024/LANES))，保持每个 program 的工作量有界。
GROUP=1 使用真正的一维归约，避开当前编译器退化二维大 tile 路径的问题。
T=128 时按请求划分 program，每组 32 个 tile，通过一维连续加载再 reshape 进行归约，
避免跨请求 gathers；最后一组的多余输出严格掩码。
非二幂的补齐 lane 不读数据、不参与归约。无效输出 lane 不读元数据、不写输出。
不完整 tile 直接产生 0，段内 padding 不进入有效 tile。

## 整数与地址安全

首先使用 `tile < length//T` 判断完整性，再将不完整 tile 的位置置为零后相乘，
避免大 K 时无效 tile 的地址乘法溢出。通用核与一维核的有效地址均不超过 N−1，使用 INT32。
补齐 lane 的地址夹到 T−1，且不参与归约。按请求加载的核在 N 接近 INT32_MAX 时使用 INT64
构造补齐地址，其他情况以编译期 N 上界保证 INT32 加法不溢出。
offsets 值为 [0,INT32_MAX]，两值之差位于 [-INT32_MAX,INT32_MAX]，可用 INT32 安全比较。
不形成可能溢出的 `first+i`。所有实际加载均受完整性和 N 上界掩码约束。
Triton 接收 Tensor 的逻辑起始地址，因此合法 storage_offset 不需要额外复制或加偏移。

## Stream、生命周期和内存

prepare 使用 Triton 的 warmup 接口编译内核并取得 CompiledKernel 启动函数，
编译阶段的临时输出分配随后释放，不用于后续输出。输入指针对齐特化关闭，允许合法存储重绑定。
每次 run 在输入设备上下文中分配新的连续 UINT8 输出，再调用已编译的内核。
不缓存 stream，使用调用时该设备的当前 stream；不修改输入，不复用旧输出。
空 offsets 使用同设备的新 zeros 输出；R=0 返回空输出。
一般路径只有一次输出分配和一次融合内核调用，无全局临时张量。
GROUP×LANES 的中间值在内核局部计算，不将其伪称为零设备内存开销。

## 验证与计时

verify.py 使用原始 generate_data 和独立 golden，覆盖全部 14 个正向输入、9 个非法输入。
对每个正向输入保留多次 candidate/baseline 输出并检查精确值、shape、dtype、device、
连续性、输入不变和存储独立；额外检查合法值更新和调用时 stream。
两组主性能输入各运行双流每流 3 个保留输出，candidate/baseline 各预热 3 次，
测量单次活跃分配峰值及 20 次逐次调用释放后的活跃分配。
峰值只表示 PyTorch NPU 分配器可见字节，不表示全部 HBM 或 CANN/UB scratch。

主性能及 T=255/257 补充计时均为两侧各预热 50 次、交替 6 轮、每轮 200 次完整 prepared.run，
轮末同步；计入 Python 提交、输出/临时分配、设备计算和完成同步。
全部配对数据使用 seed=20261031、10000 次共同 bootstrap 重采样、linear 分位数计算 CI。
主性能必须均值差>0、中位数差>0、CI 下界>0、candidate 均值≤85µs。
补充计时如实报告，不增加任务书没有规定的性能门槛。

## 本次验收证据

已执行的 CPU 测试通过 NumPy 解释实际 kernel 函数体，与原始 golden 比较 14 个公开输入和
350 个随机输入，并测试值域校验与配对统计；它仅作为辅助检查。
真机完整验收记录位于 `verification/`，实测状态以 `result.json` 为准。
环境为 Ascend910_9362（A3）、CANN 9.1.0、torch 2.7.1+cpu、torch_npu 2.7.1.post4、triton-ascend 3.2.1。
两项主性能须通过完整四项性能判据，逐轮数值与判断见 `result.json`。
14 个正向、9 个负向、动态更新、非零存储偏移/重绑定及两组双流/内存检查均通过。
本实现没有将 case.json 中历史 PASS 或历史性能当作本次结果。
