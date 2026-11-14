<!-- Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

# Docker images

## Image naming

Release and development images use the repository `xllm-ai/xllm` and the tag format
`<version>-<device>-<cpu>`:

| Field | Values |
| --- | --- |
| `version` | Release version including the `v` prefix, such as `v0.10.0`; development images always use `latest`. |
| `device` | `a2`, `a3`, or `cuda`. |
| `cpu` | `x86_64` or `arm64`. |

For example, `xllm-ai/xllm:v0.10.0-a2-arm64` contains xLLM 0.10.0 for A2
on ARM64. Its development-image counterpart is
`xllm-ai/xllm:latest-a2-arm64`. Other development tags include
`xllm-ai/xllm:latest-a3-arm64` and `xllm-ai/xllm:latest-cuda-x86_64`.
The CPU tag must match the image architecture: Docker's `linux/amd64`
platform uses the `x86_64` tag, and `linux/arm64` uses `arm64`.

## Ascend build targets

Run these commands from the repository root. Use `Dockerfile.a3` for A3.
The wheel must match the chip, Python ABI and PyTorch version in the image.

| Target | Contents |
| --- | --- |
| `xllm-runtime` | CANN/ATB, runtime libraries and Python inference dependencies; ready for `pip install` of an xLLM wheel. |
| `xllm-release` | Runtime plus the local xLLM wheel. Starts `xllm serve`. |
| `xllm-development` (default) | Runtime plus vcpkg, Go, Rust, build caches, development, benchmark and profiling tools. |

Release images do not inherit the development stage. CANN and the small host
compiler/JIT dependencies remain because Python kernels can compile during
inference. EvalScope, Jupyter, MindStudio tooling and format/test tools are
added only in the development stage. Components already supplied by the CANN
base image are retained. Kernel wheels are installed with
`--no-deps` after their inference dependencies: their upstream metadata also
lists pytest/coverage tools that are needed only for testing.

## Build a release image

Place an already-built wheel in `dist/`, then pass its filename. Match the
image version to the wheel version:

```bash
docker build --network=host -f docker/Dockerfile.a2 \
  --target xllm-release \
  --build-arg XLLM_WHEEL=xllm_npu_torch2_10_0-0.12.0-cp312-cp312-linux_aarch64.whl \
  -t xllm-ai/xllm:v0.12.0-a2-arm64 .
```

BuildKit mounts `dist/` for installation; the wheel archive is not retained in
an image layer. The default command prints the serving help:

```bash
docker run --rm xllm-ai/xllm:v0.12.0-a2-arm64
```

Mount the host driver and models when serving on an Ascend host:

```bash
docker run --rm --privileged --network=host --shm-size=32g \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v "$(command -v npu-smi)":/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /home/models:/models:ro \
  xllm-ai/xllm:v0.12.0-a2-arm64 /models/Qwen3.8-27B \
  --nnodes=2 --model_impl=python --backend=vlm \
  --enable_graph=true --python_graph_backend=aclgraph
```

The entrypoint initializes CANN/ATB and the PyTorch library paths. Override it
with `--entrypoint bash` for an interactive shell. The image preloads libc before tcmalloc: this lets the Python extension load
on AArch64 while keeping libc first in symbol lookup.

## Build an environment for later wheel installation

```bash
docker build --network=host -f docker/Dockerfile.a2 \
  --target xllm-runtime --iidfile /tmp/xllm-runtime-a2-arm64.id .
```

This intermediate environment uses an image ID, saved in
`/tmp/xllm-runtime-a2-arm64.id`, rather than a release or development tag.
Mount or copy a matching wheel into the container, then run `pip3 install`.
This target uses the same inference dependencies as the release target.

## Build a development image

Initialize the pinned build-dependency submodules before building A2, A3 or
CUDA development images:

```bash
git submodule update --init third_party/Mooncake
git -C third_party/Mooncake submodule update --init extern/yalantinglibs
```

Development builds install yalantinglibs, msgpack-cxx, zstd, liburing, xxHash
and the Go toolchain through `docker/install_mooncake_dependencies.sh`. Go follows
`GOVER` in the pinned `third_party/Mooncake/dependencies.sh`; yalantinglibs uses
the pinned submodule. BuildKit mounts these sources only during installation.
A fresh development container already satisfies the dependency checks used by
`setup.py`.

```bash
docker build --network=host -f docker/Dockerfile.a2 \
  --target xllm-development -t xllm-ai/xllm:latest-a2-arm64 .
```

For a CUDA development image on x86_64:

```bash
docker build --network=host -f docker/Dockerfile.cuda \
  -t xllm-ai/xllm:latest-cuda-x86_64 .
```

The Ascend Dockerfiles use `docker/requirements.txt` and
`docker/ascend/requirements.txt` for runtime dependencies. Their
`requirements-dev.txt` counterparts add development and profiling dependencies.
Other component versions remain build arguments in the Dockerfiles.

The Docker-specific installer is `docker/install_mooncake_dependencies.sh`.
Use `--ensure-go` to install only the required Go toolchain. Source builds continue
to use the unchanged `scripts/build_support/install_dependencies.sh` when needed.
