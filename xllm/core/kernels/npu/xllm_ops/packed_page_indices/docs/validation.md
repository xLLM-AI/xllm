# PackedPageIndices 自测报告

日期：2026-11-17。源码固定 commit：eeb559ec2489c7460cee9447cb21b9480f23f6e2。

环境：Atlas A3 / Ascend910_9382，单逻辑 NPU 0，CANN 9.1.0，驱动26.1.1，Ubuntu22.04.5 aarch64，Python3.12.13，PyTorch2.9.0，torch_npu2.9.0.post6，triton-ascend3.2.2，GCC11.4，Ninja1.13.2。

## 覆盖与精度

14/14个规定正向输入、额外长单层输入、8/8项非法输入、7项动态更新、6项同形状offset=1存储重绑定、直接run及独立输出检查全部通过。所有逐元素比较差异数为0。两项主性能输入均执行双stream各保留3份输出、20次释放回收；大输入执行INT32输出上界检查。附件golden.py、generate_data.py和case.json保持原样。

最终工程从固定commit全新checkout，重新构建C++扩展，再运行verify.py，结果PASS。源码清单逐文件SHA-256与交付源码一致。结果目录位于工程外。

## 性能（μs）

| 用例 | 基准均值 | 候选均值 | 均值上限 | 均值节省 | 配对节省95%CI | 结果 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| packed_perf_small | 132.49249 | 35.92186 | 40 | 96.57063 | [94.15335, 99.26894] | PASS |
| packed_perf_large | 132.50016 | 57.27728 | 75 | 75.22288 | [74.20900, 76.23675] | PASS |
| packed_route_32736 | 129.46162 | 35.35052 | 配对记录 | 94.11111 | [93.35667, 94.89033] | PASS |
| packed_route_32768 | 129.95316 | 57.04560 | 配对记录 | 72.90756 | [71.91742, 73.85143] | PASS |

两侧分别预热50次，6轮配对，每轮200次，按轮交替先后顺序。完整prepared.run墙钟包含Python提交、新输出和当次临时分配、设备计算及轮末同步。全部6轮参与统计。10000次共同重采样，seed=20261031，numpy.default_rng，linear分位数。主性能的均值上限、均值节省、中位数节省、CI下界四项均通过。

固定调优指标集为packed_perf_large：主性能共2项，ceil(2/10)=1。早期融合方案473.91μs，组合方案119.47μs；缓存启动器后全量曾出现79.17μs失败。哨兵映射初版135.06μs未通过；4096列映射复制分块后55.64μs通过。小输入层分块4时一次全量44.02μs未通过，改为layer_tile=1后完成本次全量通过。历史测量保留于本任务归档，最终判定仅引用本次固定commit的完整结果。

## 内存（字节）

| 用例 | 候选峰值增量 | 基准峰值增量 | 候选测前活跃 | 候选释放后 | 生命周期 |
| --- | ---: | ---: | ---: | ---: | --- |
| packed_perf_small | 9216 | 23552 | 38912 | 38912 | PASS |
| packed_perf_large | 1572864 | 2126336 | 654848 | 654848 | PASS |

采用预热后PyTorch NPU分配器可见的活跃张量峰值增量。输入、prepared和准备阶段静态布局保留，每次输出、pages和padded临时映射均计入。两侧同步释放后均恢复各自测前活跃分配。两stream共6份输出数值正确且地址互异。

## 证据与复现

源码入口solution.py，构建setup.py，一键验证verify.py。逐项数据见本目录validation-data/下的result.json、generalization.json、functional.json、performance.json、contract.json与source_manifest.json。该目录同时保存功能、配对性能和接口检查的原始日志。

完整七件套与Excel报告位于私人仓https://github.com/Wzy-wm/cann-task06-packed-page-indices-20261117的main分支，工程根目录为算子目录。实测源码固定commit为eeb559ec2489c7460cee9447cb21b9480f23f6e2。私人仓后续提交只补充报告与证据；本PR完整包含可独立重建的相同源码和测试。

最终源码的Ruff、Codespell和clang-format20.1.6检查通过。独立组件C++构建和NPU全量测试通过，xLLM全仓构建未执行。
