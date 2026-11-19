# 题目3-T5QKV拆分与缓存追加算子开发任务书

## 1. 任务概述

使用 Ascend C 实现 T5 QKV 拆分与缓存追加算子，在单 token 解码时拆分融合 QKV 输入，输出 Q，并将当前 K/V 追加到历史缓存。

## 2. 核心开发要求

### 2.1 功能实现要求

单 token 解码时，将按全部 Q、全部 K、全部 V 顺序拼接的 QKV 输入拆分，输出独立连续 Q，并将当前 K/V 追加到两份历史缓存。

### 2.2 算子工程模式

采用 ACLNN 算子工程化开发，提供 Ascend C 内核、OpDef 注册、shape/dtype 推导、tiling、可构建安装包、两阶段 ACLNN API 和独立 C++ 调用样例。安装包安装到用户目录，不覆盖系统 CANN。

同时需要支持kernel直调工程开发，可复用同一tiling计算函数或者等价的参数构造逻辑

提供 PyTorch NPU 接口 `run(inputs, attributes) -> tuple[Tensor,...]`，调用本题 ACLNN API，作为精度、性能和内存验收入口；同时提供 Kernel 直调入口进行交叉验证。计算和数据搬运均在 NPU 上完成，使用当前 stream，不修改输入。输出须为同一设备上的独立连续张量。

支持多阶段实现。框架入口须校验输入个数、维度、dtype、连续性、尺寸和属性，对非法或未知属性报错。C++ ACLNN 调用方须保证设备指针有效、ND 张量描述正确，且输出存储不重叠。

### 2.3 接口定义

#### 算子语义

`Q[b,h,0,d]=qkv[b,0,h*D+d]`。

K/V 的前 L 个 token 分别逐位复制 pastKey/pastValue；末尾 token 分别来自 `qkv[b,0,H*D+h*D+d]` 和 `qkv[b,0,2*H*D+h*D+d]`。全部输出重新分配，均为连续存储。L=0 时不能读取空历史缓存。

#### 参数说明

| 参数 | 形状 | 类型 | 说明 |
|---|---|---|---|
| qkv | [B,1,3HD] | FP16 / BF16 | 按 Q、K、V 三段排列 |
| pastKey、pastValue | [B,H,L,D] | 与 qkv 一致 | 连续历史缓存 |
| query | [B,H,1,D] | 与 qkv 一致 | 独立连续输出 |
| key、value | [B,H,L+1,D] | 与 qkv 一致 | 历史后追加当前 token |
| headNum、headDim | 标量 | int | 与 H、D 一致 |

#### 算子约束

范围为 1≤B≤64、1≤H≤128、1≤D≤256、0≤L≤8192，总输出元素数≤2^28。支持非对齐 D。所有输入连续且类型一致。拒绝尺寸、类型、布局、属性不匹配。输出互不重叠，不修改输入。不包含多 token、GQA、RoPE 或分页缓存协议。

## 3. 验收标准

### 3.1 软硬件环境要求

- **适配硬件**：昇腾 910C（Atlas A3），单个逻辑设备，48 个 Vector Core、24 个 Cube Core。
- **CANN 版本**：9.1.0。
- **Python 版本**：3.12.13。
- **框架版本**：PyTorch 2.10.0、torch_npu 2.10.0.post4+git5dd8ef3。

开发镜像及依赖见 3.5。使用其他软硬件版本时须重新验证。

### 3.2 精度要求

要求全部输出逐位一致，包括正负零、Inf 和各类 NaN 位模式；不得用浮点近似比较替代。检查输入保持不变、输出形状正确、连续且独立。

精度测试覆盖 `case.json` 中全部 **35** 个用例。使用 `generate_data.py` 生成固定种子输入，以 `golden.py` 的 CPU 结果为标杆，按 3.5 验证 `run()` 输出及存储要求。

### 3.3 性能要求

基线通过视图拆分 Q/K/V，将 Q 复制为独立连续输出，K/V 分别与历史缓存做 Concat，不额外复制或转置 K/V。

各用例须满足 `提交实现耗时 / 最快合格基线耗时 ≤ 下表对应的耗时比上限`。提交实现与基线须在同一 NPU 环境、相同输入下重新测量，每次调用均分配新输出。下表耗时为参考值，验收以同机实测比值为准。

| 用例 ID | 基线 μs | 目标耗时上限 |
|---|---:|---:|
| T5SplitQkvAppendCache_001_float16_random | 64.48 | 28 |
| T5SplitQkvAppendCache_005_float16_random | 73.58 | 28 |
| T5SplitQkvAppendCache_006_float16_random | 67.16 | 29 |
| T5SplitQkvAppendCache_007_float16_random | 201.72 | 75 |
| T5SplitQkvAppendCache_008_float16_random | 73.81 | 29 |
| T5SplitQkvAppendCache_011_bfloat16_random | 64.57 | 28 |
| T5SplitQkvAppendCache_015_bfloat16_random | 74.09 | 28 |
| T5SplitQkvAppendCache_016_bfloat16_random | 68.52 | 29 |
| T5SplitQkvAppendCache_017_bfloat16_random | 204.52 | 77 |
| T5SplitQkvAppendCache_018_bfloat16_random | 76.26 | 29 |

#### 基线计算定义

基线代码如下，`candidates(inputs, attrs)` 返回各候选实现的调用函数。输入张量须位于 NPU。

```python
import torch
import torch_npu
import torch.nn.functional as F

def candidates(inputs, attrs):
    (qkv, pk, pv) = inputs
    (b, h, l, d) = pk.shape

    def views_and_concatenation():
        parts = qkv.reshape(b, 3, h, d)
        q = parts[:, 0].reshape(b, h, 1, d).clone(memory_format=torch.contiguous_format)
        k = parts[:, 1].reshape(b, h, 1, d)
        v = parts[:, 2].reshape(b, h, 1, d)
        return (q, torch.cat((pk, k), dim=2), torch.cat((pv, v), dim=2))
    return {'views_concat_q_copy': views_and_concatenation}
```

计时前，提交实现和候选基线均须通过 3.2 的精度检查，每个用例至少保留一个合格基线。分别预热 50 次并同步设备，再测量 5 轮，每轮连续调用 200 次。从每轮开始提交前计时，到该轮设备操作全部完成为止，除以 200 得到单次耗时，取 5 轮中位数。输入传输和标杆比较不计入耗时。记录每轮数据、入选基线及排除原因。

同时记录设备事件时间，其中包含提交间隙，不作为纯内核耗时。

### 3.4 内存要求

显式全局工作区为 0。在输入已分配、运行时已预热的前提下，活跃张量分配峰值增量不超过输出字节数、上述工作区上限及 16 KiB 分配对齐余量之和。释放并同步后，重复 20 次调用的活跃张量分配须回到调用前水平。

内存按 PyTorch NPU 分配器的活跃张量字节数统计，包含输出和工作区，不计整卡 HBM、驱动缓存和片上 UB。报告须说明片上缓冲设计，并通过 3.5 的非法元数据、stream、输入提前释放和重复调用测试。

### 3.5 自验要求

#### 测试shape

必须执行 case.json 中全部 **35 个精度用例**，其中标记为 performance 的 **10 个**同时用于性能和内存验收。下表与 case.json 一一对应；尺寸元组分别表示 (B,H,L,D)。

| 用例 ID | 尺寸 | dtype | 输入特征/属性 | 性能及内存 |
|---|---|---|---|---|
| T5SplitQkvAppendCache_000_float16_random | (1, 1, 0, 1) | float16 | 随机 | 否 |
| T5SplitQkvAppendCache_001_float16_random | (4, 8, 0, 64) | float16 | 随机 | 是 |
| T5SplitQkvAppendCache_002_float16_random | (2, 3, 2, 5) | float16 | 随机 | 否 |
| T5SplitQkvAppendCache_003_float16_random | (3, 7, 31, 80) | float16 | 随机 | 否 |
| T5SplitQkvAppendCache_004_float16_random | (2, 3, 127, 65) | float16 | 随机 | 否 |
| T5SplitQkvAppendCache_005_float16_random | (1, 8, 128, 64) | float16 | 随机 | 是 |
| T5SplitQkvAppendCache_006_float16_random | (8, 12, 512, 64) | float16 | 随机 | 是 |
| T5SplitQkvAppendCache_007_float16_random | (16, 16, 2048, 64) | float16 | 随机 | 是 |
| T5SplitQkvAppendCache_008_float16_random | (1, 1, 8192, 128) | float16 | 随机 | 是 |
| T5SplitQkvAppendCache_009_float16_bits | (1, 1, 512, 128) | float16 | 原始 16 位模式 | 否 |
| T5SplitQkvAppendCache_010_bfloat16_random | (1, 1, 0, 1) | bfloat16 | 随机 | 否 |
| T5SplitQkvAppendCache_011_bfloat16_random | (4, 8, 0, 64) | bfloat16 | 随机 | 是 |
| T5SplitQkvAppendCache_012_bfloat16_random | (2, 3, 2, 5) | bfloat16 | 随机 | 否 |
| T5SplitQkvAppendCache_013_bfloat16_random | (3, 7, 31, 80) | bfloat16 | 随机 | 否 |
| T5SplitQkvAppendCache_014_bfloat16_random | (2, 3, 127, 65) | bfloat16 | 随机 | 否 |
| T5SplitQkvAppendCache_015_bfloat16_random | (1, 8, 128, 64) | bfloat16 | 随机 | 是 |
| T5SplitQkvAppendCache_016_bfloat16_random | (8, 12, 512, 64) | bfloat16 | 随机 | 是 |
| T5SplitQkvAppendCache_017_bfloat16_random | (16, 16, 2048, 64) | bfloat16 | 随机 | 是 |
| T5SplitQkvAppendCache_018_bfloat16_random | (1, 1, 8192, 128) | bfloat16 | 随机 | 是 |
| T5SplitQkvAppendCache_019_bfloat16_bits | (1, 1, 512, 128) | bfloat16 | 原始 16 位模式 | 否 |
| T5SplitQkvAppendCache_020_float16_boundary | (64, 128, 0, 256) | float16 | 随机 | 否 |
| T5SplitQkvAppendCache_021_float16_boundary | (1, 128, 3, 255) | float16 | 随机 | 否 |
| T5SplitQkvAppendCache_022_float16_boundary | (2, 1, 4097, 3) | float16 | 随机 | 否 |
| T5SplitQkvAppendCache_023_bfloat16_boundary | (64, 128, 0, 256) | bfloat16 | 随机 | 否 |
| T5SplitQkvAppendCache_024_bfloat16_boundary | (1, 128, 3, 255) | bfloat16 | 随机 | 否 |
| T5SplitQkvAppendCache_025_bfloat16_boundary | (2, 1, 4097, 3) | bfloat16 | 随机 | 否 |
| T5SplitQkvAppendCache_audit_026_float16_bits | (64, 128, 0, 256) | float16 | 原始 16 位模式 | 否 |
| T5SplitQkvAppendCache_audit_027_float16_bits | (1, 1, 8192, 256) | float16 | 原始 16 位模式 | 否 |
| T5SplitQkvAppendCache_audit_028_float16_bits | (2, 3, 63, 255) | float16 | 原始 16 位模式 | 否 |
| T5SplitQkvAppendCache_audit_029_float16_bits | (1, 128, 1, 1) | float16 | 原始 16 位模式 | 否 |
| T5SplitQkvAppendCache_audit_030_bfloat16_bits | (64, 128, 0, 256) | bfloat16 | 原始 16 位模式 | 否 |
| T5SplitQkvAppendCache_audit_031_bfloat16_bits | (1, 1, 8192, 256) | bfloat16 | 原始 16 位模式 | 否 |
| T5SplitQkvAppendCache_audit_032_bfloat16_bits | (2, 3, 63, 255) | bfloat16 | 原始 16 位模式 | 否 |
| T5SplitQkvAppendCache_audit_033_bfloat16_bits | (1, 128, 1, 1) | bfloat16 | 原始 16 位模式 | 否 |
| T5SplitQkvAppendCache_audit_034_float16_bits | (64, 128, 62, 256) | float16 | 原始 16 位模式 | 否 |

工程测试必须覆盖两个独立 stream、提前释放输入后另一 stream 的分配压力，以及直调和 ACLNN 各重复 20 次调用后的活跃张量回收。另用现有公开输入保留 8 次调用的全部输出，同时执行 32 次提前释放输出和同流分配写入压力；8 组输出必须保持独立且结果正确。重复调用不另计独立 case。缓存额外测试连续两次追加。

本题共 **47 个有效输入用例**（35 个公开 + 12 个额外种子/存储偏移），另有 **28 个非法输入/属性场景**。非法场景在三个入口执行得到 84 条检查记录；额外输入在两个入口执行得到 24 条记录。性能、内存、预热及重复调用均复用上述输入，不另计用例。

#### 环境依赖与数据生成

开发镜像：`quay.io/jd_xllm/xllm-ai:xllm-dev-cann9.1.0-a3-openeuler24.03-py3.12`。

镜像依赖：Python 3.12.13、torch 2.10.0+cpu、torch_npu 2.10.0.post4+git5dd8ef3、NumPy 2.2.6、ml_dtypes 0.6.0。torch 使用 +cpu 基础包，配合 torch_npu 和自定义内核执行 NPU 计算。

按分配的设备设置 `ASCEND_RT_VISIBLE_DEVICES`，容器内使用逻辑设备 0。运行前挂载驱动目录并加载 CANN 环境。

从本题目录执行以下命令，按 `case.json` 导出全部测试输入：

```bash
python3 generate_data.py --output-dir generated_data
```

可用 `--case 用例ID` 选择单个用例，重复传入该参数可选择多个用例。脚本生成原始 `.bin` 输入和 `manifest.json`，记录文件字节数及 SHA-256。输入生成使用 `SHA-256("20261014/" + case_name)` 的前 8 字节按小端序解释为种子，交给 NumPy 默认随机数生成器；特殊数据模式由用例的 `kind` 决定。

输入和输出顺序与 `op.json` 一致，属性为名称到数值的字典。可直接调用 `generate_data.generate(case)` 生成内存输入，无需先导出二进制文件；精度期望值调用 `golden.calculate(inputs, attributes)` 获取，`golden.calc_expect_func` 也提供标杆调用入口。使用其他测试框架时，按上述接口接入。

#### 精度自验证步骤

1. 逐条读取 `case.json`，生成确定性输入，并从 `attr_desc` 构造属性字典；先以 CPU 输入计算标杆。
2. 输入按原始 16 位表示传到选定 NPU，再解释为 FP16/BF16，避免改写特殊位模式。调用提交实现 `run(inputs, attributes)` 并同步设备。
3. 检查输出个数、NPU 设备及编号、shape、dtype、连续性；逐项按 3.2 的精度规则比较。BF16 回传也按原始 16 位解释，避免浮点转换改变位模式。
4. 所有非空输出的底层存储地址必须互不相同，且不得与任一非空输入的底层存储相同；不能只比较各张量的起始指针。逐位核对全部输入未被修改。
5. 每个用例保存输出检查结果与通过／失败状态。记录每个输出的位不一致元素数。任何失败返回非零退出码并保存错误日志，不得仅以总体状态代替逐项检查。

#### 必测覆盖清单

公开用例的 ID、输入输出 shape、dtype、属性和数据模式以本包 `case.json` 为准。性能和内存用例为其中 `performance=true` 的子集，与上表一致。规定用例不得删除、修改输入定义或跳过，可另行增加用例。

本包公开用例文件的 SHA-256 为 `9ce34ba31fc08abe1969ad4997f7b6173fa950533473d6977398a8ed4db741e8`。新增用例时保留原有定义。

**非法输入和属性场景**

以下场景须在 `prepare`、`run`、`run_direct` 三个入口分别测试，均应报错。以第一个性能用例为基础，仅修改表中指定项，使用本题输入类型和 NPU 设备构造张量。

| 场景 ID | 构造／检查内容 |
|---|---|
| `attributes_type` | 属性容器为列表而非字典 |
| `cache_dtype` | 历史 Key 类型改为 FP32 |
| `cache_mismatch` | 历史 Value 的 L 比 Key 多 1 |
| `cache_rank` | 历史 Key 形状为 (2,3) |
| `cpu_input` | 第一个输入位于 CPU |
| `float32` | 第一个输入为 FP32 |
| `headDim_'8'` | headDim 设置为 '8' |
| `headDim_-1` | headDim 设置为 -1 |
| `headDim_0` | headDim 设置为 0 |
| `headDim_1.5` | headDim 设置为 1.5 |
| `headDim_65` | headDim 设置为 65 |
| `headDim_True` | headDim 设置为 True |
| `headNum_'8'` | headNum 设置为 '8' |
| `headNum_-1` | headNum 设置为 -1 |
| `headNum_0` | headNum 设置为 0 |
| `headNum_1.5` | headNum 设置为 1.5 |
| `headNum_9` | headNum 设置为 9 |
| `headNum_True` | headNum 设置为 True |
| `input_count` | 缺少最后一个输入 |
| `noncontiguous` | 输入为隔元素切片等非连续布局 |
| `packed_mismatch` | 融合 QKV 最后一维为 3HD+1 |
| `packed_tokens` | 融合 QKV 的 token 维为 2 |
| `shape_limit_(0, 1, 0, 1)` | 按 (B,H,L,D)=(0, 1, 0, 1) 构造 QKV 和两份缓存，属性为空字典 |
| `shape_limit_(1, 1, 0, 257)` | 按 (B,H,L,D)=(1, 1, 0, 257) 构造 QKV 和两份缓存，属性为空字典 |
| `shape_limit_(1, 1, 8193, 1)` | 按 (B,H,L,D)=(1, 1, 8193, 1) 构造 QKV 和两份缓存，属性为空字典 |
| `shape_limit_(1, 129, 0, 1)` | 按 (B,H,L,D)=(1, 129, 0, 1) 构造 QKV 和两份缓存，属性为空字典 |
| `shape_limit_(65, 1, 0, 1)` | 按 (B,H,L,D)=(65, 1, 0, 1) 构造 QKV 和两份缓存，属性为空字典 |
| `unknown_attribute` | 包含未定义的属性 unknown |

**额外固定种子和存储偏移用例**

下列 12 个用例分别经过 `run` 和 `run_direct`，共 24 条记录。用例名的 `__extra_seed_` 前缀部分对应公开用例，复制其输入输出描述及属性，将名称替换为完整 ID，设置 `kind="random"`、`performance=false`，再调用 `generate_data.generate(case)`。数据模式统一使用 `kind="random"`，不按用例名称中的 zeros、bits 等字样生成。

| 额外用例 ID |
|---|
| `T5SplitQkvAppendCache_000_float16_random__extra_seed_0` |
| `T5SplitQkvAppendCache_001_float16_random__extra_seed_1` |
| `T5SplitQkvAppendCache_002_float16_random__extra_seed_2` |
| `T5SplitQkvAppendCache_002_float16_random__extra_seed_6` |
| `T5SplitQkvAppendCache_002_float16_random__extra_seed_7` |
| `T5SplitQkvAppendCache_002_float16_random__extra_seed_8` |
| `T5SplitQkvAppendCache_003_float16_random__extra_seed_3` |
| `T5SplitQkvAppendCache_004_float16_random__extra_seed_4` |
| `T5SplitQkvAppendCache_005_float16_random__extra_seed_5` |
| `T5SplitQkvAppendCache_009_float16_bits__extra_seed_9` |
| `T5SplitQkvAppendCache_010_bfloat16_random__extra_seed_10` |
| `T5SplitQkvAppendCache_011_bfloat16_random__extra_seed_11` |

每个输入张量分配“元素数 + 16”的同类型设备存储，取从第 1 个元素偏移开始、长度等于原元素数的视图，恢复原 shape 后复制输入。输出必须通过相同精度检查，输入位模式保持不变，输出不得与输入或其他输出共享底层存储。

**验证器与工程检查**

- 验证器必须拒绝三类反例：CPU 输出、复用输入存储、起始指针不同但底层存储重叠的输出。
- 两个独立 stream 分别使用 `T5SplitQkvAppendCache_001_float16_random`、`T5SplitQkvAppendCache_018_bfloat16_random`。每个 stream 至少执行 10 次准备后的调用，并检查正式 ACLNN 入口结果。
- 工程测试另外至少执行 5 次非法元数据拒绝检查；提前释放输入后的跨 stream 分配压力、重复调用回收和输出生命周期要求见前述工程要求。
- 按“用例 ID、入口”逐项检查 NPU 测试记录。精度、性能、内存和扩展测试全部通过方可验收，重复记录不重复计数。
- 连续缓存追加测试：选择公开用例中首个 L=31 的输入，执行一次追加后，将输出 Key/Value 作为下一次历史缓存，使用同一 QKV 再追加；第二次 Key/Value 的 L 增加 2，并逐位对照标杆。

## 4. 验收交付件

| 序号 | 交付件名称 | 交付件要求 |
|---|---|---|
| 1 | 算子设计文档 | 算法、分核、数值/地址推导、片上缓冲、工作区预算及接口设计。正式社区设计评审按任务仓流程提交。 |
| 2 | 自测用例及测试代码 | 覆盖全部公开精度用例、标记的性能/内存用例及 3.5 的工程检查；提供独立标杆、公开验证器和完整复现步骤。 |
| 3 | 自测报告 | 按[官方自验证材料模板](https://cann-op.obs.cn-north-4.myhuaweicloud.com/task_submission.zip)提交下列七件，填写真实精度、性能、内存数据并保存原始日志；另附环境和源码指纹。 |
| 4 | 待验收代码及工程 | 完整 ACLNN 源码、构建/私有安装脚本、.run 安装包、两阶段 API、独立 C++ 样例、PyTorch NPU 封装及直调交叉验证入口。提交个人代码仓地址、分支、算子目录和固定 commit。 |

交付件放在提交工程的 task_submission/，并将同名七件打包为 task_submission.zip：

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

## 5. PR 申请合入

https://github.com/xLLM-AI/xllm/tree/main/xllm/core/kernels/npu/xllm_ops

## 6. 参考资料

- [天数公开功能对标](https://developer.iluvatar.com/docs/ixformer#t5_split_qkv_update_kv_cache)。
- [Ascend C Kernel 直调说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/programming_guide/appendix/completing_kernel_launch_based_on_sample_project.md)。


## 7. 环境获取

使用发布方分配或自行准备的 910C 开发环境，按 3.1 和 3.5 配置软件及镜像。活动算力申请按官方流程办理。

## 8. 特别注意事项

1. 输入值域和布局须符合接口要求，非法元数据须报错。
2. 算子须在 NPU 上完成计算，不得返回预存结果或使用 CPU 代算。
3. 性能以本题 NPU 基线为准。测试受其他任务干扰时，保留日志并重新测量。
4. 自测报告须与实际结果一致，失败或中断的测试不得标记为通过。
