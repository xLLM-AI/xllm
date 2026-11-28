# 题目2-行列双向INT8量化算子参考解答

完整 ACLNN 工程与实机验收材料。正式入口是 python/solution.py 的 run()，调用生成的 ACLNN 两阶段 API；run_direct() 和 prepare() 保留作独立交叉验证。

## 环境和运行

910C / Ascend910_9382，CANN 9.1.0，驱动 25.5.t6.b001；aarch64 openEuler 24.03 LTS-SP3，Python 3.12.13，torch 2.10.0+cpu，torch_npu 2.10.0.post4+git5dd8ef3。镜像：`quay.io/jd_xllm/xllm-ai:xllm-dev-cann9.1.0-a3-openeuler24.03-py3.12`。全部构建、编译、测试和运行都在 SSH 开发环境完成。本机仅编辑和同步文件。

管理员按实际分配配置 NPU 和驱动挂载，容器内使用逻辑设备 0。进入本解答目录后执行：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
python3 tools/build_aclnn.py
bash build.sh
export ASCEND_CUSTOM_OPP_PATH="$(pwd)/aclnn_install/vendors/community_task2${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
./build/run_aclnn_example 0
./build/run_example 0
python3 tests/run_tests.py --task-dir tests/task
python3 tests/task/verify.py --solution-file python/solution.py --report task_submission/public_accuracy.json
python3 tests/run_audit.py
python3 tests/run_engineering.py --task-dir tests/task
python3 tests/run_benchmark.py --task-dir tests/task
python3 tests/check_acceptance.py
```

也可执行 `bash verify_all.sh`，自动构建、运行完整验证并保存原始日志。工具链构建和安装使用自己的英文临时目录，规避 CANN 对中文路径的限制；最终产物回到本目录。每次重建只清理本工程自己的临时目录。

## 交付内容

| 路径 | 内容 |
|---|---|
| aclnn/ | OpDef、推导、tiling、Ascend C 内核及官方生成工程配置 |
| tools/build_aclnn.py | 构建 .run 安装包并安装到本目录 aclnn_install/ |
| op_host/aclnn_binding.cpp | 正式 ACLNN 框架封装；保留输入跨流保护；新输出和工作区在创建流上按 FIFO 安全复用 |
| examples/run_aclnn_example.cpp | 独立 C++ 两阶段 ACLNN 调用与可解析结果检查 |
| op_kernel/、include/api.h、examples/run_example.cpp | 直调交叉验证实现与样例 |
| tests/task/ | 与题目一致的公开用例、独立标杆、验证器和冻结的性能门槛 |
| tests/ | 精度、边界/非法输入、并发、内存、性能及综合验收脚本 |
| [docs/任务书.md](docs/任务书.md) | 本解答满足的完整任务要求 |
| [docs/design.md](docs/design.md) | 实现设计与工作区预算 |
| [task_submission/](task_submission/) | 七件模板材料及完整 JSON、日志、版本和源码指纹 |
| task_submission.zip | 与官方模板同名的七件自验证材料 |

公开精度 27 个，性能/内存 5 个；具体通过记录见 acceptance.json。报告按同一台 910C 上较快且精度合法的附件基线比较，全部规定性能用例均不得慢于同机最快合法基线，任一项退化即失败。复测会更新 JSON，应同时保存日志并更新 Excel，不能沿用旧报告冒充新结果。

本包是待发布的题目和参考解答。社区立项、邀请维护者、设计评审及外部 PR 属于正式发布流程；本次没有替用户执行这些外部操作。

测试数量按独立输入统计：27 个公开、12 个扩展、13 个负向场景；性能/内存复用公开子集。必测清单见 tests/task/coverage_requirements.json，验收器检查缺失与重复记录。
