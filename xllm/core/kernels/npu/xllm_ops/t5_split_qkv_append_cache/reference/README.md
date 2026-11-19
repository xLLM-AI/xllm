# 题目3-T5QKV拆分与缓存追加算子：固定耗时目标版

本包定义开发任务、测试输入和验收规则。任务书列出每个性能用例的固定耗时上限，单位为微秒（μs）。版本：2026-10-23。

## 文件

- `题目3-T5QKV拆分与缓存追加算子任务书.md`：完整功能、工程、精度、性能、内存和交付要求。
- `case.json`：35 个公开精度用例，其中 10 个同时用于性能和内存验收。
- `op.json`：输入、输出和属性描述。
- `golden.py`、`generate_data.py`：独立 CPU 标杆和固定种子数据生成。
- `verify.py`：公开精度、输出存储和输入不变检查；其通过状态不代表性能或全部工程验收通过。
- `performance_targets.json`：逐用例绝对耗时上限及固定测量协议。
- `benchmark.py`：测量提交实现的正式 `run()`，输出所有性能用例的逐轮记录并进行性能判定。
- `check_performance.py`：离线核对性能报告的协议、覆盖、原始轮次及目标；不依赖 NPU 库，也不能代替实际设备测量。
- `coverage_requirements.json`：公开及扩展检查的必测清单。
- `SHA256SUMS.txt`：本包文件校验值。

## 指定环境

910C / Ascend910_9382，单个逻辑设备，48 个 Vector Core、24 个 Cube Core；CANN 9.1.0、驱动 25.5.t6.b001、aarch64 openEuler 24.03 LTS-SP3、Python 3.12.13、torch 2.10.0+cpu、torch_npu 2.10.0.post4+git5dd8ef3、NumPy 2.2.6、ml_dtypes 0.6.0。

开发镜像：`quay.io/jd_xllm/xllm-ai:xllm-dev-cann9.1.0-a3-openeuler24.03-py3.12`。torch 的 `+cpu` 是基础包标记，NPU 计算由 torch_npu 和自定义内核提供。管理员按分配配置设备可见性和驱动挂载，容器内使用逻辑设备 0，并加载 CANN 环境。

## 运行与验收

在上述 NPU 环境中，从本题目录运行：

```bash
python3 verify.py --solution-file /absolute/path/to/solution.py --report public_accuracy.json
python3 benchmark.py --solution-file /absolute/path/to/solution.py --report performance.json > performance.log 2>&1
python3 check_performance.py --report performance.json --output performance_acceptance.json
```

命令失败会返回非零退出码并保留错误信息；不要仅依据文件是否生成判断通过。调用约定是 `run(inputs, attributes) -> tuple[Tensor,...]`；inputs 按 `op.json` 排列，attributes 为属性字典。正式 `run()` 必须调用本题 ACLNN API，并在每次调用时返回新分配的独立输出；另外提供 Kernel 直调交叉验证入口及任务书规定的准备接口。

性能协议固定为预热 50 次、5 轮每轮 200 次，取轮均耗时的中位数，逐项对照 `performance_targets.json`。全部 5 轮必须保留；单轮偶发偏慢不直接决定结果。外部干扰、复测及环境要求详见任务书第 3.3 节。公开检查器只核对性能部分，不能替代第 3.5 节规定的 stream、生命周期、非法输入、额外输入、内存等检查。

测量脚本记录能够查询到的环境信息；标为 `unknown` 的项目须另附实际版本证据。验收方须核对指定硬件、驱动和软件环境，不能把脚本通过视为环境自动核验通过。入口文件哈希不代表整个实现版本，须另附完整源码及构建产物指纹。

输入按原始 16 位编码传输，避免浮点类型转换改写位模式。数据可由验证器直接生成，也可执行 `python3 generate_data.py --output-dir generated_data` 导出。提交者须提供完整 ACLNN 工程、可构建私有安装包、两阶段 API、独立 C++ 样例和实际测试报告。

测试数量按独立输入统计：35 个公开、12 个额外输入、28 类非法场景。性能、内存和预热复用已有输入，不重复计数。输入精度、内存限制及工程交付范围见任务书。
