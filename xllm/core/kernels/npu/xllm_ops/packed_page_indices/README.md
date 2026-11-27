# PackedPageIndices

题目6：ragged page indices 打包算子。面向 Atlas A3 单逻辑 NPU、CANN 9.1.0，提供独立 Python/Triton/C++ 工程。

## 接口

```python
from solution import prepare, run
prepared = prepare([X, M, cu], {})
output = prepared.run()
output_direct = run([X, M, cu], {})
```

输入为同设备连续 INT32：X[R,C]、M[B] 或 [L,B]、cu[R+1]。令 P=cu[R]，输出为 [P] 或 [L,P]。每行处理长度为 cu[r+1]-cu[r] 的有效前缀，负页号输出 -1，其余查询 M。每次产生独立输出，使用调用时当前 stream。

prepare 后 shape、dtype、device、总 P 固定。调用方可合法修改 X/M、调整总 P 保持不变的 counts，或对原 Tensor 执行同形状 set_。改变形状或总 P 时重新 prepare。

## 环境与构建

验证组合：Ubuntu 22.04 aarch64、Python 3.12.13、Ascend910_9382、CANN 9.1.0、驱动26.1.1、PyTorch2.9.0、torch_npu2.9.0.post6、Triton Ascend3.2.2、GCC11.4、Ninja1.13.2。

```bash
source /usr/local/Ascend/cann/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
MAX_JOBS=4 python setup.py build_ext --inplace
```

Python3.12须为带ensurepip的发行版。构建使用PyTorch C++扩展与Ninja，生成本机Python ABI对应的so。

## 完整自测

从工程根目录运行，结果写入工程外的新目录：

```bash
python verify.py --output-dir ../verification-new
```

入口依次执行附件原始功能/工程检查、首个性能输入的8项负向测试与直接run检查、4项配对性能测试，再核对覆盖矩阵。失败返回非零并保留错误记录。

主要输出：result.json、generalization.json、functional.json、performance.json、contract.json、source_manifest.json，以及3份原始运行日志。主性能完整prepared.run计时分别预热50次、6轮、每轮200次，轮末同步；共同bootstrap重采样10000次、seed=20261031、linear分位数。

验收文档位于 task_submission/，设计位于 docs/design.md。fixtures/保留发布方用例和独立CPU标杆，其参考结果与本工程实测分开。

## 来源

上游目标：xLLM-AI/xllm，基线e55a2abc1109866ec9b0bad8363109b92a27bfe0，集成位置xllm/core/kernels/npu/xllm_ops/packed_page_indices。

Triton文档：https://triton-ascend.readthedocs.io/zh-cn/latest/ 。工程许可证见LICENSE。发布方fixtures作为本任务提供的验收材料保留。
