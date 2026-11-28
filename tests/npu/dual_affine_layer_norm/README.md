# 双分支 LayerNorm 验证工程

本目录提供独立 ACLNN 两阶段样例和使用同一 Ascend C 内核的直调绑定。默认产品入口在 `xllm/python/kernels_npu/dual_affine_layer_norm.py`：`dual_affine_layer_norm(...)` 是模型调用接口，`run(inputs, attributes)` 是验收接口。`prepare` 校验并保留输入，每次准备后的调用仍分配两个独立输出。`run_direct` 需要加载本目录构建的验证库。

C++ 封装校验形状、dtype、连续性和 epsilon，按输入设备设置 guard，使用当前 NPU stream。输入记录到当前 stream。任务队列复制指针和连续张量的元数据，复用 torch_npu 分配器的 stream 顺序管理，不在队列闭包中持有输出 Tensor。两阶段 ACLNN 的准备和执行都在队列中完成，不缓存输出或单次 executor。

## 构建和运行

先构建并私有安装子模块的 DualAffineLayerNorm 算子，以及子模块 README 中的 direct 工程。以下命令从主仓根目录执行。`OUT` 为独立工作目录，`VENDOR` 为私有安装后的 vendor 目录。

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source "$VENDOR/bin/set_env.bash"
export ASCEND_RT_VISIBLE_DEVICES=0
cmake -S tests/npu/dual_affine_layer_norm -B "$OUT/build-validation" \
  -DCMAKE_BUILD_TYPE=Release -DDUAL_AFFINE_VENDOR="$VENDOR" \
  -DBUILD_TORCH_BINDING=ON \
  -DDUAL_AFFINE_DIRECT_BUILD="$OUT/build-direct" \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
cmake --build "$OUT/build-validation" -j2
"$OUT/build-validation/dual_affine_ln_example"
```

独立测试不要求构建整个 xLLM。先加载验证库，再运行本目录的 pytest：

```bash
export DUAL_AFFINE_TEST_LIBRARY="$OUT/build-validation/dual_affine_ln_test.so"
python -c 'import os,torch,torch_npu,pytest; torch.ops.load_library(os.environ["DUAL_AFFINE_TEST_LIBRARY"]); raise SystemExit(pytest.main(["tests/npu/dual_affine_layer_norm/test_dual_affine_layer_norm.py", "-q"]))'
```

每个设备测试进程仅暴露一张卡，并在进程内使用逻辑设备 0。所有长任务在远端 tmux 中执行。

## 验证范围

正式回归测试覆盖两种 dtype、非对齐 H、非零存储偏移、两路输出的独立性、输入不变、常量行严格返回 beta、epsilon 边界，以及输入提前释放后的 stream 生命周期。任务附件的 46 个公开输入、12 个额外输入、60 条非法输入入口检查、性能和分配器内存检查由附件验证器和独立任务材料执行，结果保存在任务工作目录，不进入产品仓。

性能使用附件 `benchmark.py` 的正式 `run`，预热 50 次、5 轮各 200 次，取 API 到设备完成的轮均耗时中位数。设备事件计时包含提交间隙，不能当作纯内核耗时。环境要求包括精确驱动版本；仅脚本通过不代表环境自动满足验收要求。
