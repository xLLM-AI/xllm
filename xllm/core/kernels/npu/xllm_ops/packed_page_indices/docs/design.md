# PackedPageIndices 算子设计

## 1. 需求与环境

题目6要求把每个请求页表的有效前缀按请求顺序压紧，并逐层翻译为物理页号。输入连续NPU INT32 X[R,C]、M[B]或M[L,B]、cu[R+1]，属性为空。目标为Atlas A3单逻辑设备、CANN9.1.0。

验证环境为Ascend910_9382、Ubuntu22.04 aarch64、Python3.12.13、PyTorch2.9.0、torch_npu2.9.0.post6、triton-ascend3.2.2、GCC11.4、Ninja1.13.2。

## 2. 接口与边界

`prepare(inputs, attributes) -> Prepared`完成一次性校验、元数据计算与编译。`Prepared.run() -> Tensor`每次读取当前输入，返回独立连续INT32输出。直接`run`包含prepare。

令n_r=cu[r+1]-cu[r]，P=cu[R]，对0≤j<n_r：

```text
output[l, cu[r]+j] = -1                     当 X[r,j] < 0
output[l, cu[r]+j] = M[l, X[r,j]]            当 X[r,j] >= 0
```

一维M时输出[P]，二维M时输出[L,P]。R/C/P可为0，B/L≥1。cu从0开始，非递减，每段≤C。M非负，有效前缀内非负页号<B；每个输入及输出的元素数≤2147483647。

校验只向CPU传输X的有效前缀，尾部越界值保持未消费。cu和M的一次性回读用于prepare参数校验。设备核读取也使用有效前缀mask。

## 3. 路径选择

按输出元素总数选择路径，阈值32768只依赖固定形状和P。

- 输出元素<32768：Triton融合读取X、查询M并写输出，省去中间张量与额外提交。
- 输出元素≥32768：Triton打包当前页号，并复制当前映射到含负页哨兵的对齐映射表，由C++调用一次原生NPU index_select完成逐层翻译。
- P=0：C++分配规定shape的独立空输出。

初版融合大输入实测473.91μs，组合路径119.47μs。缓存启动器后降至约70至79μs；最终改为哨兵映射，并将映射复制分块扩大为4096，固定指标集大输入55.64μs。最终固定源码全量结果见验收报告。

## 4. Triton核

融合路径grid为(R,L,ceil(C/column_tile))。column_tile=min(256,next_power_of_2(max(1,C)))，layer_tile=1。逐层独立program提高小输入并行度，小输入分块诊断为31.21μs。各program加载当前cu[r:r+2]，以j<本请求计数且j<C为读写mask；负页号关闭M读取mask并输出-1。

组合路径令S=ceil((B+1)/32)*32，分配pages[P]和padded[L,S]。页号非负时保持原值，负页号改写为B；映射表前B列复制当前M，其余列填-1。打包和映射复制在同一核中使用各自掩码，grid=(max(R*ceil(C/column_tile),L*ceil(S/4096)),1,1)。映射复制按层及4096列块组织，避免逐元素除法。请求写入不相交区间，每个位置恰有一个写入者；原生Gather在同一当前stream按序消费两个临时张量。

prepare调用Triton warmup编译，保存CompiledKernel公开的grid启动器。每次run传入当前Tensor对象，由启动器取得当前地址和当前stream。所有指针参数关闭地址对齐特化，支持set_到offset=1后的地址变化。

## 5. C++调度与内存

HostPlan保存固定输出shape、层数和映射容量，并保留输入Tensor对象用于输出options。allocate使用torch::empty分配独立输出。gather对padded[L,S]执行index_select(dim=1,pages)，B索引读取-1哨兵。二维输出直接返回，一维映射时恢复[P]形状。

组合路径的pages和padded通过一次HostPlan.allocate_workspace调用在C++内分别新分配，每次复制当前映射，不缓存输入数值或输出。

融合路径的张量分配为4LP字节输出。组合路径的有效张量字节为4LP+4P+4LS，实际分配器对齐与NPU内部临时开销由实测峰值统计。prepared阶段临时编译样本退出后释放，prepared只保留输入、元数据与编译启动器。

每次调用的pages、padded和输出均独立分配，生命周期由本次调用管理。张量框架和Triton启动器跟随调用时当前stream。双stream分别保留3份输出，验证6个地址互异、数值正确；20次释放后同步，检查活跃分配回到测前值。

## 6. 动态更新与精度

prepare后固定shape/dtype/device/P；调用方可更新X/M，以及总P保持不变的合法cu。核每次读取当前cu和当前X/M。三输入同形状set_不会改变保存的Tensor身份，调用时地址由当前Tensor解析。

全流程使用INT32页号和物理映射值，负值语义通过布尔mask实现，支持INT32最大输出值。每个规定输入与独立CPU列表标杆逐元素比较；输入不变、形状、设备、连续性与独立存储同时检查。

## 7. 验证与性能统计

覆盖14项规定正向输入、额外长单层、8项非法输入、两性能输入动态更新、offset=1重绑定、双stream和释放回收。附件golden.py以原样调用；test_contract.py补充首个性能输入上的8项负向构造与直接run检查。

每个计时用例两侧分别预热50次，6轮配对，每轮200次，按轮交替执行顺序。每轮墙钟包括完整prepared.run、分配、Python提交和轮末设备同步。保存elapsed_ns及每次均值μs。主性能分别≤40/75μs，同时基准减候选的均值差、中位数差、配对差均值95%bootstrap CI下界均>0。共同重采样10000次，seed=20261031，numpy.default_rng，linear分位数。

两项route补充用例提交相同协议逐轮数据及统计。内存按预热后PyTorch活跃分配峰值统计，保留两侧输入和prepared静态布局。

## 8. 构建与集成

独立根目录包含solution.py、kernels.py、host.cpp、setup.py、requirements.txt、verify.py、benchmark.py、test_contract.py、fixtures/和task_submission/。使用PyTorch CppExtension与Ninja构建，无ACLNN安装包。

目标仓为https://github.com/xLLM-AI/xllm，基线e55a2abc1109866ec9b0bad8363109b92a27bfe0，新增目录为xllm/core/kernels/npu/xllm_ops/packed_page_indices。完整复现命令见README.md。
