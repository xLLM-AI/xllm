# xLLM v0.10：DataSystem 与 Mooncake 双后端

GitCode fork：[https://gitcode.com/qq_45595516/xllm-datasystem-clean](https://gitcode.com/qq_45595516/xllm-datasystem-clean)。功能分支：`feat/datasystem-kv-backend`。

本仓保留 xLLM 完整源码，在 Ascend NPU 的 PD 分离路径中加入 DataSystem PULL，保留 Mooncake PUSH。两个后端编入同一个 xLLM 二进制，通过启动参数选择。

此分支以现有 v0.10 上游提交 `5e814b188b89c5e937bf681178814d3667a5c96a` 为基线。DataSystem 固定为 2026-10-09 核对的最新公开发行版 **0.8.1**，并集成 10 月修复中的 SDK 动态定位、进程共享 CANN runtime 和 LevelDB tcmalloc 处理。这个版本是本仓的新部署目标，并非从历史服务器日志取得的版本号。

本仓已经过本地源码、脚本和 Python 检查；本次整理没有在 NPU 上执行编译和 PD 测试。下面给出可复现的部署流程和明确验收条件。当前上游 main 已改变接口，本分支先用于复现 v0.10、审查独立功能及准备 PR；向当前 main 合入前仍需迁移和 NPU 回归。

## 功能改动

- 新增 DataSystem adapter，直接编入主程序；只有依赖 SDK 的 pure-C ABI bridge 为独立 `.so`。
- DataSystem Prefill 使用 `MSetD2H` 发布 KV，Decode 使用 `MGetH2D` 拉取 KV；强制 `PULL`。
- KV key 包含模型和并行拓扑身份，以及链式 token block 内容哈希，避免物理 block 复用造成旧 KV 命中。
- 将 request ID、block hashes 和源信息贯穿调度、worker RPC 与传输路径，处理未完成的 partial block。
- 保留 Mooncake 原有 PUSH 路径；CLI/config 的后端选择优先于旧环境变量。
- 在 DataSystem 构建中移除 LevelDB 导出的精确 `tcmalloc` 依赖；不改写 vcpkg 安装文件。
- 提供可配置的 NPU 构建、真实 RPC、双后端 PD 验收，以及一份受控 HTTP 性能比较工具。

## 版本与目录

| 组件 | 固定配置 |
|---|---|
| 主程序基线 | 上述 v0.10 commit，保留原有上游历史 |
| 平台 | Linux aarch64，Ascend A2 / 910，示例使用卡 0 和 1 |
| DataSystem SDK + worker | `openyuanrong-datasystem==0.8.1` 完整发行包 |
| Python / glibc | CPython 3.11；wheel 要求 glibc ≥ 2.34 |
| 开发镜像 | `quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-cann9-20260605` |
| Mooncake | 原工程 gitlink `1adcb2fb399661e69df0bc4c342c3f4a7028b8e8` |
| etcd | 3.5.21，Linux ARM64 |
| xllm-service | 示例使用候选 commit `a54981bd4ba08fb7665b55794905f5145e70b9f9`；目标 NPU 上尚待验证 |
| 模型 | `Qwen/Qwen2.5-7B-Instruct`，固定 revision 见第 4 步 |

完整版本记录见 [datasystem_versions.json](docs/deployment/datasystem_versions.json)。DataSystem 发行信息见 [PyPI](https://pypi.org/project/openyuanrong-datasystem/)。换 SDK 版本时，必须同时更换配套 worker，并重新编译 bridge、完成 RPC 和 PD 验收。

```text
README.md                         xLLM 上游介绍及本指南入口
README_DATASYSTEM.md               从零安装、运行、验证、准备 PR
cmake/datasystem_allocator.cmake   10 月 allocator 修复的源码实现
tools/datasystem/                  环境模板、构建、RPC、PD smoke、验收
tools/kv_cache_benchmark/          唯一一份双后端性能套件
tests/datasystem/                  源码合同、部署工具及 CMake 回归检查
tests/core/framework/batch/        partial block 的 C++ 回归测试
xllm/                             完整 xLLM 源码，含 DataSystem adapter/bridge
```

镜像、模型、SDK wheel 和 Git 子模块通过网络取得。它们不作为二进制文件提交到功能仓。宿主机驱动、固件和 Docker 应事先可用。

## 第 1 步：取得仓库并检查 NPU 宿主机

提供的 `xllm-datasystem-clean.tar.gz` 含源码和 `.git`，可以直接解压。另一种方式是从新建远端 clone 功能分支；远端仓为 [https://gitcode.com/qq_45595516/xllm-datasystem-clean](https://gitcode.com/qq_45595516/xllm-datasystem-clean)。

```bash
# 方式 A：使用交付的完整 Git 仓库
mkdir -p /workspace/xllm-handoff
tar -xzf xllm-datasystem-clean.tar.gz -C /workspace/xllm-handoff
cd /workspace/xllm-handoff/xllm-datasystem-clean
git status --short
git branch --show-current
git log -3 --oneline

# 方式 B：远端仓已上传后使用
# git clone --branch feat/datasystem-kv-backend https://gitcode.com/qq_45595516/xllm-datasystem-clean.git xllm-datasystem-clean
```

功能分支应为 `feat/datasystem-kv-backend`；`baseline/v0.10` 是没有开发材料的上游基线。首次解压不要执行 `git init`，否则会失去提供的历史和分支关系。

```bash
uname -m                       # 本指南要求 aarch64
npu-smi info                   # 确认驱动正常且 NPU 0/1 可用
docker --version
ls -l /dev/davinci0 /dev/davinci1 /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
ss -ltn | grep -E ':(2379|2380|31501|28888|28889|18010|18020|18888|18898|7777|7787|26000|26100)\b' || true
```

上述端口应空闲，或明确属于本次使用的 etcd/worker；新旧容器采用 host 网络时仍会冲突。不要在已有正式服务的 NPU 上同时运行本 smoke。

## 第 2 步：建立独立 NPU 开发容器

以下命令在宿主机仓库目录中执行。挂载主机驱动，工具链来自开发镜像。

```bash
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-cann9-20260605
docker run -d --name xllm-datasystem-v010-clean \
  --privileged --network host --shm-size 16g \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v "$PWD":/workspace/xllm-datasystem-clean \
  -w /workspace/xllm-datasystem-clean \
  quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-cann9-20260605 sleep infinity
docker exec -it xllm-datasystem-v010-clean bash
```

后续第 3–10 步在容器内执行。

```bash
cd /workspace/xllm-datasystem-clean
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
unset LD_PRELOAD
uname -m
ldd --version | head -1
python3.11 --version
python3.11 -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__)'
cmake --version | head -1
npu-smi info
git submodule sync --recursive
git submodule update --init --recursive
git submodule status --recursive
python3.11 -m pip install pre-commit
pre-commit install
```

子模块按仓库 gitlink 固定提交，禁止使用 `--remote` 自动升到最新 Mooncake。`setup.py` 会继续调用工程原有依赖构建流程，需要网络和足够磁盘空间。容器内 CANN、ATB、torch_npu 与驱动须配套；导入 torch_npu 失败时先修复基础环境。

## 第 3 步：安装 DataSystem 0.8.1 完整包

为 ARM64 / CPython 3.11 固定官方 wheel 和 SHA-256。SDK 和 worker 都由这个 wheel 提供，避免旧 worker 与新 SDK 混用。

```bash
mkdir -p .deployment-cache
curl -fL --retry 3 \
  https://files.pythonhosted.org/packages/5e/d0/25b5e42116302976fe924ee70bbbc85f8a19acac693dba3f451665e6c229/openyuanrong_datasystem-0.8.1-cp311-cp311-manylinux_2_34_aarch64.whl \
  -o .deployment-cache/openyuanrong_datasystem-0.8.1-cp311-cp311-manylinux_2_34_aarch64.whl
printf '%s  %s\n' \
  c26a210b6411d19b28a0d28c05cc4aebf8f3fe7015bd0869183e2f24f0399e07 \
  .deployment-cache/openyuanrong_datasystem-0.8.1-cp311-cp311-manylinux_2_34_aarch64.whl \
  | sha256sum -c -
python3.11 -m pip install .deployment-cache/openyuanrong_datasystem-0.8.1-cp311-cp311-manylinux_2_34_aarch64.whl
python3.11 -c "from importlib.metadata import version; print(version('openyuanrong-datasystem'))"
python3.11 tools/datasystem/sdk_info.py
command -v dscli
```

最后输出应为 0.8.1，SDK 检查应找到 `include/datasystem/hetero_client.h`、`lib/libdatasystem.so`（或 `lib64`）和配套 `datasystem_worker`。本仓环境入口总是从所选 Python 的 `yr.datasystem` 定位 SDK，不沿用开发机上的旧绝对路径。

## 第 4 步：准备模型与本机配置

用独立下载环境准备模型，避免改变推理环境中的依赖。也可以直接使用已有的完整模型目录。

```bash
python3.11 -m venv .model-download
.model-download/bin/python -m pip install 'huggingface_hub>=0.34'
.model-download/bin/hf download Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --local-dir "$PWD/models/qwen2.5_7B_Instruct"
ls models/qwen2.5_7B_Instruct/config.json
ls models/qwen2.5_7B_Instruct/model*.safetensors
cp tools/datasystem/config.env.example tools/datasystem/config.env
```

编辑 `tools/datasystem/config.env`，至少填好以下三项。容器内所用 Python 必须同时能导入 torch_npu 和 DataSystem。

```bash
export DS_PYTHON=python3.11
export MODEL_PATH=/workspace/xllm-datasystem-clean/models/qwen2.5_7B_Instruct
export XLLM_SERVICE_BIN=/workspace/xllm-service-v010/build/xllm_service/xllm_master_serving
```

上面是配置内容示例，应写入 `config.env`。脚本会读取它；仅在终端临时 export 后又运行脚本，会被配置文件覆盖。`config.env` 已被 `.gitignore` 排除，不应提交服务器路径或凭据。

默认 worker `31501`、registry `2379`、NPU `0/1`。需要换端口或卡号时在同一配置中修改。其他引擎测试端口可在 smoke 启动环境中指定。

## 第 5 步：安装并启动 etcd 和配套 worker

```bash
curl -fL --retry 3 \
  https://github.com/etcd-io/etcd/releases/download/v3.5.21/etcd-v3.5.21-linux-arm64.tar.gz \
  -o .deployment-cache/etcd-v3.5.21-linux-arm64.tar.gz
printf '%s  %s\n' \
  95bf6918623a097c0385b96f139d90248614485e781ec9bee4768dbb6c79c53f \
  .deployment-cache/etcd-v3.5.21-linux-arm64.tar.gz | sha256sum -c -
tar -xzf .deployment-cache/etcd-v3.5.21-linux-arm64.tar.gz -C .deployment-cache
install -m 755 .deployment-cache/etcd-v3.5.21-linux-arm64/etcd /usr/local/bin/etcd
install -m 755 .deployment-cache/etcd-v3.5.21-linux-arm64/etcdctl /usr/local/bin/etcdctl
mkdir -p diagnostics registry-data
source tools/datasystem/env.sh
nohup etcd --name xllm-datasystem-clean --data-dir "$ROOT/registry-data" \
  --listen-client-urls "http://$ETCD_HOST:$ETCD_CLIENT_PORT" \
  --advertise-client-urls "http://$ETCD_HOST:$ETCD_CLIENT_PORT" \
  --listen-peer-urls "http://$ETCD_HOST:$ETCD_PEER_PORT" \
  --initial-advertise-peer-urls "http://$ETCD_HOST:$ETCD_PEER_PORT" \
  --initial-cluster "xllm-datasystem-clean=http://$ETCD_HOST:$ETCD_PEER_PORT" \
  > diagnostics/etcd.log 2>&1 &
echo $! > diagnostics/etcd.pid
for attempt in $(seq 1 30); do
  etcdctl --endpoints="http://$ETCD_HOST:$ETCD_CLIENT_PORT" endpoint health && break
  sleep 2
done
etcdctl --endpoints="http://$ETCD_HOST:$ETCD_CLIENT_PORT" endpoint health
dscli start -w --worker_address "$DATASYSTEM_WORKER_HOST:$DATASYSTEM_WORKER_PORT" \
  --etcd_address "$ETCD_ADDR"
timeout --foreground --kill-after=15 90 "$DS_PYTHON" tools/datasystem/rpc_probe.py
```

etcd health 必须成功，RPC probe 必须输出 `SDK_RPC_INIT_OK`。端口能连接只能说明 TCP 可达；只有真实 `HeteroClient.init()` 能通过这一层验收。worker 和 registry 是不同端口。

如使用已有 etcd，省略 `nohup etcd`；如已有旧 worker，先按对应环境的方式停止，再从新 wheel 的 dscli 启动。不要同时启动两套相同端口服务。

## 第 6 步：编译 xllm-service 网关

PD smoke 使用 OpenAI HTTP 网关，需要与 v0.10 引擎协议配套的 `xllm_master_serving`。有原环境验收过的二进制时，优先在 `config.env` 中指定它；否则可以从下列同日源码候选版本构建。该候选版本本次尚未在 NPU 上验证，是否配套以第 8 步为准。

```bash
git clone https://github.com/xLLM-AI/xllm-service.git /workspace/xllm-service-v010
cd /workspace/xllm-service-v010
git checkout a54981bd4ba08fb7665b55794905f5145e70b9f9
git submodule update --init --recursive
bash prepare.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
unset LD_PRELOAD
command -v python
python -c 'import torch; print(torch.__version__)'
cmake -S . -B build -DUSE_ASCEND=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target xllm_master_serving -j8
find "$PWD/build" -type f -name xllm_master_serving -perm /111
cd /workspace/xllm-datasystem-clean
```

构建时 `python` 必须能导入推理环境中的 torch，防止网关 CMake 误判 ABI；CMake 至少 3.26。如果该镜像的 vcpkg 未自动配置，请在上面的 configure 中加 `-DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"`。将 `find` 输出的实际网关路径填回 `config.env`，不要依据目录名猜测路径。

`prepare.sh` 给 cpprestsdk 打补丁，仅首次执行；重复执行时若提示补丁已应用，检查工作树后继续，不要反向恢复它。网关的子模块、依赖与主程序各自构建。

## 第 7 步：编译同一份双后端 xLLM

```bash
cd /workspace/xllm-datasystem-clean
bash tools/datasystem/check_env.sh
MAX_JOBS=8 CLEAN_BUILD=1 bash tools/datasystem/build.sh
bash tools/datasystem/run.sh bash tools/datasystem/verify_build.sh
bash tools/datasystem/run.sh ldd -r build/xllm/core/server/xllm
bash tools/datasystem/run.sh ldd -r build/xllm/core/server/libxllm_yuanrong_datasystem_bridge.so
bash tools/datasystem/run.sh bash -c 'ldd "$XLLM_BIN" | grep -i tcmalloc && exit 1 || true'
```

构建入口执行官方 `python setup.py --device npu build_ext`，开启 `USE_MOONCAKE=ON`、`USE_YUANRONG_DATASYSTEM=ON`，SDK prefix 从当前 Python 自动生成。默认 `CLEAN_BUILD=1` 只清理本仓官方 CMake build 目录，升级 SDK 后应重新构建。

必须产生：

```text
build/xllm/core/server/xllm
build/xllm/core/server/libxllm_yuanrong_datasystem_bridge.so
```

主程序里包含 adapter；不应存在旧架构的 `libxllm_yuanrong_kv_cache_transfer.so`。`ldd -r` 不能出现 `not found` 或 `undefined symbol`，主程序不应再依赖 tcmalloc。

10 月修复已落实在 `cmake/datasystem_allocator.cmake`：仅在 NPU DataSystem 构建中移除 LevelDB 的精确可选 allocator 依赖，保留 Threads 和其他依赖；Mooncake-only 构建保留上游行为。若其他库仍间接带入 tcmalloc，上述动态依赖检查会暴露它，应先排查链接来源再继续。

不要用手工编译 bridge、旧主程序和新 SDK 混搭。运行入口会清除 `LD_PRELOAD`，避免 allocator 注入；bridge 默认在进程基础 namespace 中加载，共享 torch_npu 使用的 CANN runtime。

## 第 8 步：依次验收 DataSystem 和 Mooncake

```bash
source tools/datasystem/env.sh
timeout --foreground --kill-after=15 90 "$DS_PYTHON" tools/datasystem/rpc_probe.py
bash tools/datasystem/test_pd.sh DataSystem
bash tools/datasystem/test_pd.sh Mooncake
```

两个测试读取同一配置、同一主程序。每次启动 Prefill、Decode 和网关，保存独立日志，结束时清理这份安装的引擎/网关及其子进程。etcd 和 worker 继续保留，供后续测试使用。

通过条件：

| DataSystem PULL | Mooncake PUSH |
|---|---|
| Prefill 和 Decode 都选中 DataSystem | Prefill 和 Decode 都选中 Mooncake |
| Prefill `MSetD2H success=1` | Prefill `backend=Mooncake op=push success=1` |
| Decode `MGetH2D success=1` | HTTP 非空 assistant 输出 |
| 两侧出现匹配的 `key0` | 输出 `PD_ACCEPTANCE_OK` |
| HTTP 非空 assistant 输出 | |
| 输出 `PD_ACCEPTANCE_OK` | |

HTTP 200、进程启动或二进制里能搜到后端名字都不能替代 KV 传输验收。详细日志保存在 `diagnostics/pd_<backend>_*`。

测试示例针对 Qwen2.5、TP=1、一卡 Prefill + 一卡 Decode；默认关闭 prefix cache、chunked prefill 和 schedule overlap，remote H2D 为 0，使用同一个 DataSystem worker。多机、TP/DP/CP、多种模型、linear-state/speculative/hierarchy 组合的兼容性应另外验证，不由本 smoke 的通过结果推定。

## 第 9 步：运行受控性能比较

先运行小规模试测，确认完成率、输出 token 数和后端日志，再扩大并发。详细传输诊断在功能测试中打开，在正式计时中关闭。

```bash
bash tools/kv_cache_benchmark/verify_package.sh
CONCURRENCIES=1,2,4 ROUNDS=1 REQUESTS=20 WARMUP=3 \
  bash tools/kv_cache_benchmark/run_xllm_controlled_dual_backend_suite.sh

# 试测通过后再运行两类完整场景：重复前缀 / 唯一 prompt
ROUNDS=4 REQUESTS=100 WARMUP=10 \
  bash tools/kv_cache_benchmark/run_full_performance_campaign.sh
```

数据保存于 `tools/kv_cache_benchmark/results/`，包含原始逐请求 CSV、summary JSON、服务日志、异常分析和成对比较。比较时保持模型、prompt、token 上限、设备、主程序与配置相同，交替执行两个后端。输出 token 差异过大、SSE 不完整或传输证据缺失时，不把吞吐/TTFT 当作有效后端对比。

## 第 10 步：保存证据及停止本次服务

```bash
mkdir -p diagnostics/deployment-record
git rev-parse HEAD > diagnostics/deployment-record/engine.commit
git submodule status --recursive > diagnostics/deployment-record/submodules.txt
python3.11 -m pip freeze > diagnostics/deployment-record/pip-freeze.txt
bash tools/datasystem/run.sh python3.11 tools/datasystem/sdk_info.py \
  > diagnostics/deployment-record/sdk.json
sha256sum build/xllm/core/server/xllm \
  build/xllm/core/server/libxllm_yuanrong_datasystem_bridge.so \
  > diagnostics/deployment-record/binaries.sha256
npu-smi info > diagnostics/deployment-record/npu.txt

# 停止当前克隆目录下的引擎和已配置网关，含子进程及 PID 复用检查。
bash tools/datasystem/run.sh bash -c \
  '"$DS_PYTHON" "$ROOT/tools/datasystem/stop_installation.py" "$XLLM_BIN" "$XLLM_SERVICE_BIN"'
# 仅在 worker/etcd 是本次单独创建、无其他业务依赖时执行以下两项：
dscli stop --worker_address 127.0.0.1:31501
# 运行停止命令前核对 diagnostics/etcd.pid 与该进程的命令行。
ps -fp "$(cat diagnostics/etcd.pid)"
kill "$(cat diagnostics/etcd.pid)"
```

如果复用了已有 etcd/worker，只停止测试引擎和网关。若未新建 etcd，不会有上述 pid 文件。主机退出容器后可执行 `docker stop xllm-datasystem-v010-clean`。

## 开发检查和 PR 准备

```bash
PYTHONDONTWRITEBYTECODE=1 python3.11 -m unittest discover \
  -s tests/datasystem -p 'test_*.py'
bash tools/kv_cache_benchmark/verify_package.sh
git diff --check baseline/v0.10..HEAD
```

CMake allocator 回归需本机有 CMake 和 Ninja；没有 CMake 时四项检查会 skip，不能记录为通过。C++ partial block 测试需在完整 NPU 工具链中构建：

```bash
# 配置必须开启 BUILD_TESTING；setup.py 默认可通过 SKIP_TEST 控制。
BUILD_TESTS=1 CLEAN_BUILD=0 bash tools/datasystem/build.sh
cmake --build build/cmake.linux-aarch64-cpython-311 --target batch_test -j8
ctest --test-dir build/cmake.linux-aarch64-cpython-311 -R batch_test --output-on-failure
```

本仓已在 GitCode 上从 [xLLM-AI/xllm](https://gitcode.com/xLLM-AI/xllm) 创建 fork，并推送整理后的功能分支。该 fork 与原开发仓 `xllm-datasystem` 分开维护。

```bash
git remote -v
# origin:   https://gitcode.com/qq_45595516/xllm-datasystem-clean.git
# upstream: https://gitcode.com/xLLM-AI/xllm.git
git fetch upstream release/v0.10.0
git diff --stat baseline/v0.10..HEAD
git log --oneline baseline/v0.10..HEAD
# 后续修改完成后推送同一功能分支
git push origin feat/datasystem-kv-backend
```

创建 PR 时选择：

- 源仓：`qq_45595516/xllm-datasystem-clean`，源分支：`feat/datasystem-kv-backend`。
- 目标仓：`xLLM-AI/xllm`，先按已确认的 v0.10 范围选择 `release/v0.10.0`。
- 当前上游 main 的接口已有变化；向 main 提 PR 前，应先迁移并重新验收，避免把整个旧版本差异混进后端 PR。

本分支基于官方 `v0.10.0` 标签，`baseline/v0.10` 也已推送，便于精确审查功能 diff。2026-10-09 核对时，上游 `release/v0.10.0` 比该标签多 13 个维护提交；这次保留原基线，正式申请合入前可根据维护者要求 rebase 到该分支并回归。

PR 应聚焦 DataSystem 后端、内容 key、构建/运行修复和必要测试，不提交模型、wheel、编译目录、机器配置、历史报告或重复源码。先开 draft 记录 NPU 验收状态；完成第 8 步和目标分支回归后再申请正式合入。提交前按仓库 `AGENTS.md` 和 pre-commit 规范检查修改文件，并在 PR 中附上两后端日志及版本证据。

常见失败：RPC init 失败先核对 worker 与 SDK 同版本及地址；bridge 崩溃先核对旧二进制、SDK prefix、tcmalloc 和 LD_PRELOAD；Mooncake 被切到 DataSystem 时检查 CLI/config 与残留环境变量；网关请求失败先核对 xllm-service 协议版本、注册端口及引擎存活，再看响应内容。
