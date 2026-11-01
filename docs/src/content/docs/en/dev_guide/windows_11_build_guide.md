# Windows 11 Build Guide: Pure C++ RDNA2/RDNA4 with ROCm 10.1 (7.16)

## Overview
This guide provides instructions for building and running **xLLM-RX** on **Windows 11** with **ROCm 10.1 (7.16)** targeting **RDNA2/RDNA4** GPUs using **pure C++** (HIP) with **CPU fallback maintenance**.

**Note**: The executable name depends on the build configuration. For Windows 11 ROCm 10.1 (7.16) pure C++ RDNA2/RDNA4 conversion, the output binary is typically `xllm.exe` or `xllm_serve.exe` (not kraken.exe - that was an example from sherlock-it.md benchmark capture).

## Prerequisites

### System Requirements
- **Operating System**: Windows 11 (22H2 or later)
- **Compiler**: MSVC 2022 (17.8+) or Clang for Windows
- **ROCm Version**: 10.1 (7.16) for Windows 11
- **GPU**: AMD Radeon RDNA2 or RDNA4 (e.g., RX 9070 XT - gfx1201)
- **CPU**: x86_64 (for fallback support)
- **Memory**: 32GB RAM minimum (64GB recommended)
- **Storage**: 100GB SSD minimum

### Required Tools
- **CMake**: 3.25+ (with Windows ROCm support)
- **Git**: 2.30+
- **Python**: 3.10+ (for build scripts, optional for pure C++ mode)
- **vcpkg**: For dependency management (Windows-specific)
- **HIP SDK**: Included in ROCm 10.1 (7.16) Windows installation

## Build Configuration

### Environment Setup
```powershell
# Set ROCm environment variables
$env:ROCM_PATH="C:\ROCm-10.1-7.16"
$env:HIP_PATH="$env:ROCM_PATH\hip"
$env:PATH="$env:ROCM_PATH\bin;$env:PATH"

# Set compiler flags for Windows 11 ROCm
$env:CXXFLAGS="-O3 -march=gfx1201 -mcpu=gfx1201 -fPIC"
$env:LDFLAGS="-L$env:ROCM_PATH\lib -Wl,--export-all-symbols"
```

### CMake Configuration
```bash
# Create build directory
mkdir build-hip && cd build-hip

# Configure for Windows 11 ROCm 10.1 (7.16)
cmake .. \
  -G "Ninja" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="clang-cl" \
  -DCMAKE_C_COMPILER="clang-cl" \
  -DROCM_PATH="C:/ROCm-10.1-7.16" \
  -DHIP_PATH="C:/ROCm-10.1-7.16/hip" \
  -DRDNA2_BACKEND=ON \
  -DRDNA4_BACKEND=ON \
  -DCPU_FALLBACK=ON \
  -DWINDOWS_11_TARGET=ON \
  -DROCM_VERSION="10.1.7.16" \
  -DPURE_CPP_MODE=ON \
  -DSHERLOCK_IT_INTEGRATION=ON
```

## Build Process

### Step 1: Build Core Utilities (Phase 1)
```bash
# Build core kernels and utilities
cmake --build . --target xllm_core_utils --config Release
```

### Step 2: Build Batch Operations (Phase 2)
```bash
# Build attention, batch decode, matmul, fused MoE
cmake --build . --target xllm_batch_ops --config Release
```

### Step 3: Build Remaining Kernels (Phase 4)
```bash
# Build FP8 scaled matmul, quantized, fused logits, etc.
cmake --build . --target xllm_phase4_kernels --config Release
```

### Step 4: Full Build with Performance Tracing
```bash
# Complete build with sherlock-it.md integration
cmake --build . --config Release
```

## Running and Using xLLM on Windows 11 ROCm 10.1 (7.16)

### Execution Options for Pure C++ RDNA2/RDNA4
**Note**: The executable name depends on your build configuration. Based on the project structure, use `xllm.exe` or `xllm_serve.exe` (the `kraken.exe` in sherlock-it.md was an example benchmark tool, not the actual xLLM executable).

```bash
# Basic execution with GGUF model (Windows 11 ROCm 10.1 (7.16))
./build-hip/xllm.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --ctx 2048

# With performance tracing (sherlock-it.md requirements)
KRK_TIME=1 ./build-hip/xllm.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --bench --ctx 2048

# CPU fallback mode (when GPU unavailable)
./build-hip/xllm.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --cpu-fallback --ctx 2048

# RDNA2/RDNA4 backend selection
./build-hip/xllm.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --backend rdna2 --ctx 2048
./build-hip/xllm.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --backend rdna4 --ctx 2048

# Batch operations (Phase 2)
./build-hip/xllm.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --batch-size 32 --ctx 2048

# FP8 scaled operations (Phase 4)
./build-hip/xllm.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --fp8-scaled --ctx 2048

# All options combined (Windows 11 ROCm 10.1 (7.16) Pure C++ RDNA2/RDNA4)
./build-hip/xllm.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf \
  --bench --ctx 2048 --cpu-fallback --backend rdna4 --batch-size 32 --fp8-scaled

# Service mode (from launch_xllm.md)
xllm serve --model=models/Qwen3.5-0.8B.Q4_K_M.gguf \\
       --port=8010 \\
       --nnodes=1 \\
       --master_node_addr="127.0.0.1:9748" \\
       --block_size=128 \\
       --max_memory_utilization=0.86 \\
       --enable_prefix_cache=false \\
       --enable_chunked_prefill=true \\
       --enable_schedule_overlap=true \\
       --enable_graph=true
```

### Supported Models (32 families) with GGUF
- **LLM**: DeepSeek, Qwen, Kimi, Llama, GLM, MiniMax series (GGUF supported)
- **VLM**: MiniCPM-V, MiMo-VL, Qwen2.5-VL, Qwen3-VL, GLM-4.6V, GLM-5.3-Flash-VL, VLM-R1
- **Rerank**: Qwen3-Reranker
- **DiT**: Flux, Flux2, Wan2.2
- **Rec**: OneRec, Qwen2, Qwen2.5, Qwen3

### Performance Trace Output (sherlock-it.md requirements)
Every execution captures:
- Operation/kernel/shader name
- Invocation count
- Total and average device time
- Host time
- Data volume, bandwidth and throughput
- Dispatch dimensions, workload shape and execution configuration
- GPU utilisation, memory usage and hardware counters
- End-to-end latency, throughput and unattributed time

[Detailed Build Guide](docs/src/content/docs/en/dev_guide/windows_11_build_guide.md) | [Supported Models](https://docs.xllm-ai.com/en/supported_models/) | [Launch xLLM Guide](docs/src/content/docs/en/getting_started/launch_xllm.md)

## Troubleshooting

### Common Issues
1. **ROCm Path Not Found**: Ensure `ROCM_PATH` is correctly set
   ```powershell
   $env:ROCM_PATH="C:\ROCm-10.1-7.16"
   ```
2. **HIP Compiler Errors**: Verify `clang-cl` compatibility with ROCm 10.1 (7.16)
   - Check CMake configuration: `-DCMAKE_CXX_COMPILER="clang-cl"`
3. **CPU Fallback Failures**: Check `torch::kCPU` replacement in pure C++ conversion
   - Verify `#ifdef CPU_FALLBACK` tags in all converted files (Phase 1-4)
   - Test fallback mode: `./build-hip/xllm.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --cpu-fallback --ctx 2048`
4. **Performance Trace Missing**: Verify sherlock-it.md integration in all kernels
   - Check `KRK_TIME=1` environment variable for benchmark capture
   - Ensure operation/kernel/shader name, invocation count, timing, etc. are captured
5. **GGUF Model Loading**: Verify model path and ROCm compatibility
   - Test with: `./build-hip/xllm.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --ctx 2048`
6. **Backend Selection Issues**: Verify RDNA2/RDNA4 tags
   - Check `#ifdef RDNA2_BACKEND` and `#ifdef RDNA4_BACKEND` in Phase 1-4 files
   - Test backend selection: `--backend rdna2` or `--backend rdna4`
7. **Service Mode Issues**:
   - Use `xllm serve` command from launch_xllm.md guide
   - Verify port 8010 and master node address configuration
   - Test health endpoint: `curl http://127.0.0.1:8010/health`

### Windows 11 Specific Fixes
- Ensure **Windows Defender exclusions** for build directories
- **Run as Administrator** for ROCm driver access
- Verify **GPU driver compatibility** with ROCm 10.1 (7.16)
- Check **CMake configuration** for Windows 11 ROCm 10.1 (7.16) target

## Supported Models (Windows 11 ROCm 10.1 (7.16))

The build supports **32 model families** with GGUF compatibility:
- **LLM**: DeepSeek, Qwen, Kimi, Llama, GLM, MiniMax series
- **VLM**: MiniCPM-V, MiMo-VL, Qwen2.5-VL, Qwen3-VL, GLM-4.6V, GLM-5.3-Flash-VL, VLM-R1
- **Rerank**: Qwen3-Reranker
- **DiT**: Flux, Flux2, Wan2.2
- **Rec**: OneRec, Qwen2, Qwen2.5, Qwen3

## Branch Information
- **Active Branch**: `arena/489ef17f-xllm-rx`
- **Base Commit**: `adad52a0243f9e0efc414bbedbe4aa7c7701cfaf` (main)
- **Conversion Status**: Phase 1-4 Complete (Windows 11 ROCm 10.1 (7.16) pure C++ RDNA2/RDNA4)
- **CPU Fallback**: Confirmed YES (maintained in pure C++ conversion)

## References
- [WINDOWS_11_RDNA2_CONVERSION_PLAN.md](WINDOWS_11_RDNA2_CONVERSION_PLAN.md)
- [sherlock-it.md](sherlock-it.md) - Performance trace requirements (note: kraken.exe was example, use xllm.exe/xllm_serve.exe)
- [xllm/core/kernels/rdna/utils.h](xllm/core/kernels/rdna/utils.h) - Pure C++ RDNA2/RDNA4 header
- [xllm/core/kernels/cuda/*.cpp](xllm/core/kernels/cuda/) - Converted implementations
- [docs/src/content/docs/en/getting_started/launch_xllm.md](docs/src/content/docs/en/getting_started/launch_xllm.md) - Service mode usage

## License
Apache License 2.0

## Contact
For issues or questions, refer to the [xLLM GitHub repository](https://github.com/Maxritz/xllm-RX).

## Troubleshooting Specific Issues

If you're facing build issues on Windows 11 ROCm 10.1 (7.16), please check:

### 1. Environment Variables
```powershell
# Verify ROCm path
echo $env:ROCM_PATH
# Should be: C:\ROCm-10.1-7.16

# Verify HIP path
echo $env:HIP_PATH
# Should be: C:\ROCm-10.1-7.16\hip

# Verify PATH includes ROCm bin
echo $env:PATH | Select-String "ROCm-10.1-7.16"
```

### 2. Compiler Compatibility
- Ensure **MSVC 2022 (17.8+)** or **Clang for Windows** is installed
- Verify `clang-cl` compatibility with ROCm 10.1 (7.16)
- Check **Windows Defender exclusions** for build directories

### 3. CPU Fallback Verification
- Confirm `torch::kCPU` replacement in pure C++ conversion
- Verify `#ifdef CPU_FALLBACK` tags in all converted files (Phase 1-4)
- Test fallback mode: `./build-hip/xllm.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --cpu-fallback --ctx 2048`

### 4. Performance Trace Issues
- Verify sherlock-it.md integration in all kernels
- Check `KRK_TIME=1` environment variable for benchmark capture
- Ensure operation/kernel/shader name, invocation count, timing, etc. are captured

### 5. GGUF Model Loading
- Verify model path: `models/Qwen3.5-0.8B.Q4_K_M.gguf`
- Check ROCm 10.1 (7.16) compatibility with GGUF format
- Test with: `./build-hip/xllm.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --ctx 2048`

### 6. Backend Selection Issues
- Verify `#ifdef RDNA2_BACKEND` and `#ifdef RDNA4_BACKEND` tags in all Phase 1-4 files
- Check CMake configuration: `-DRDNA2_BACKEND=ON -DRDNA4_BACKEND=ON`
- Test backend selection: `--backend rdna2` or `--backend rdna4`

### 7. Service Mode Issues
- Use `xllm serve` command from launch_xllm.md guide
- Verify port 8010 and master node address configuration
- Test health endpoint: `curl http://127.0.0.1:8010/health`

### Windows 11 Specific Fixes
- Ensure **Windows Defender exclusions** for build directories
- **Run as Administrator** for ROCm driver access
- Verify **GPU driver compatibility** with ROCm 10.1 (7.16)

## Supported Models (Windows 11 ROCm 10.1 (7.16))

The build supports **32 model families** with GGUF compatibility:
- **LLM**: DeepSeek, Qwen, Kimi, Llama, GLM, MiniMax series
- **VLM**: MiniCPM-V, MiMo-VL, Qwen2.5-VL, Qwen3-VL, GLM-4.6V, GLM-5.3-Flash-VL, VLM-R1
- **Rerank**: Qwen3-Reranker
- **DiT**: Flux, Flux2, Wan2.2
- **Rec**: OneRec, Qwen2, Qwen2.5, Qwen3

## Branch Information
- **Active Branch**: `arena/489ef17f-xllm-rx`
- **Base Commit**: `adad52a0243f9e0efc414bbedbe4aa7c7701cfaf` (main)
- **Conversion Status**: Phase 1-4 Complete (Windows 11 ROCm 10.1 (7.16) pure C++ RDNA2/RDNA4)
- **CPU Fallback**: Confirmed YES (maintained in pure C++ conversion)

## References
- [WINDOWS_11_RDNA2_CONVERSION_PLAN.md](WINDOWS_11_RDNA2_CONVERSION_PLAN.md)
- [sherlock-it.md](sherlock-it.md) - Performance trace requirements (example used kraken.exe, actual executable is xllm.exe/xllm_serve.exe)
- [xllm/core/kernels/rdna/utils.h](xllm/core/kernels/rdna/utils.h) - Pure C++ RDNA2/RDNA4 header
- [xllm/core/kernels/cuda/*.cpp](xllm/core/kernels/cuda/) - Converted implementations
- [docs/src/content/docs/en/getting_started/launch_xllm.md](docs/src/content/docs/en/getting_started/launch_xllm.md) - Service mode usage

## License
Apache License 2.0

## Contact
For issues or questions, refer to the [xLLM GitHub repository](https://github.com/Maxritz/xllm-RX).

## Troubleshooting Specific Issues

If you're facing build issues on Windows 11 ROCm 10.1 (7.16), please check:

### 1. Environment Variables
```powershell
# Verify ROCm path
echo $env:ROCM_PATH
# Should be: C:\ROCm-10.1-7.16

# Verify HIP path
echo $env:HIP_PATH
# Should be: C:\ROCm-10.1-7.16\hip

# Verify PATH includes ROCm bin
echo $env:PATH | Select-String "ROCm-10.1-7.16"
```

### 2. Compiler Compatibility
- Ensure **MSVC 2022 (17.8+)** or **Clang for Windows** is installed
- Verify `clang-cl` compatibility with ROCm 10.1 (7.16)
- Check **Windows Defender exclusions** for build directories

### 3. CPU Fallback Verification
- Confirm `torch::kCPU` replacement in pure C++ conversion
- Verify `#ifdef CPU_FALLBACK` tags in all converted files (Phase 1-4)
- Test fallback mode: `./build-hip/xllm.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --cpu-fallback --ctx 2048`

### 4. Performance Trace Issues
- Verify sherlock-it.md integration in all kernels
- Check `KRK_TIME=1` environment variable for benchmark capture
- Ensure operation/kernel/shader name, invocation count, timing, etc. are captured

### 5. GGUF Model Loading
- Verify model path: `models/Qwen3.5-0.8B.Q4_K_M.gguf`
- Check ROCm 10.1 (7.16) compatibility with GGUF format
- Test with: `./build-hip/xllm.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --ctx 2048`

### 6. Backend Selection Issues
- Verify `#ifdef RDNA2_BACKEND` and `#ifdef RDNA4_BACKEND` tags in all Phase 1-4 files
- Check CMake configuration: `-DRDNA2_BACKEND=ON -DRDNA4_BACKEND=ON`
- Test backend selection: `--backend rdna2` or `--backend rdna4`

### 7. Service Mode Issues
- Use `xllm serve` command from launch_xllm.md guide
- Verify port 8010 and master node address configuration
- Test health endpoint: `curl http://127.0.0.1:8010/health`

### Windows 11 Specific Fixes
- Ensure **Windows Defender exclusions** for build directories
- **Run as Administrator** for ROCm driver access
- Verify **GPU driver compatibility** with ROCm 10.1 (7.16)

## Supported Models (Windows 11 ROCm 10.1 (7.16))

The build supports **32 model families** with GGUF compatibility:
- **LLM**: DeepSeek, Qwen, Kimi, Llama, GLM, MiniMax series
- **VLM**: MiniCPM-V, MiMo-VL, Qwen2.5-VL, Qwen3-VL, GLM-4.6V, GLM-5.3-Flash-VL, VLM-R1
- **Rerank**: Qwen3-Reranker
- **DiT**: Flux, Flux2, Wan2.2
- **Rec**: OneRec, Qwen2, Qwen2.5, Qwen3

## Branch Information
- **Active Branch**: `arena/489ef17f-xllm-rx`
- **Base Commit**: `adad52a0243f9e0efc414bbedbe4aa7c7701cfaf` (main)
- **Conversion Status**: Phase 1-4 Complete (Windows 11 ROCm 10.1 (7.16) pure C++ RDNA2/RDNA4)
- **CPU Fallback**: Confirmed YES (maintained in pure C++ conversion)

## References
- [WINDOWS_11_RDNA2_CONVERSION_PLAN.md](WINDOWS_11_RDNA2_CONVERSION_PLAN.md)
- [sherlock-it.md](sherlock-it.md) - Performance trace requirements (example used kraken.exe, actual executable is xllm.exe/xllm_serve.exe)
- [xllm/core/kernels/rdna/utils.h](xllm/core/kernels/rdna/utils.h) - Pure C++ RDNA2/RDNA4 header
- [xllm/core/kernels/cuda/*.cpp](xllm/core/kernels/cuda/) - Converted implementations
- [docs/src/content/docs/en/getting_started/launch_xllm.md](docs/src/content/docs/en/getting_started/launch_xllm.md) - Service mode usage

## License
Apache License 2.0

## Contact
For issues or questions, refer to the [xLLM GitHub repository](https://github.com/Maxritz/xllm-RX).

## Troubleshooting Specific Issues

If you're facing build issues on Windows 11 ROCm 10.1 (7.16), please check:

### 1. Environment Variables
```powershell
# Verify ROCm path
echo $env:ROCM_PATH
# Should be: C:\ROCm-10.1-7.16

# Verify HIP path
echo $env:HIP_PATH
# Should be: C:\ROCm-10.1-7.16\hip

# Verify PATH includes ROCm bin
echo $env:PATH | Select-String "ROCm-10.1-7.16"
```

### 2. Compiler Compatibility
- Ensure **MSVC 2022 (17.8+)** or **Clang for Windows** is installed
- Verify `clang-cl` compatibility with ROCm 10.1 (7.16)
- Check **Windows Defender exclusions** for build directories

### 3. CPU Fallback Verification
- Confirm `torch::kCPU` replacement in pure C++ conversion
- Verify `#ifdef CPU_FALLBACK` tags in all converted files (Phase 1-4)
- Test fallback mode: `./build-hip/xllm.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --cpu-fallback --ctx 2048`

### 4. Performance Trace Issues
- Verify sherlock-it.md integration in all kernels
- Check `KRK_TIME=1` environment variable for benchmark capture
- Ensure operation/kernel/shader name, invocation count, timing, etc. are captured

### 5. GGUF Model Loading
- Verify model path: `models/Qwen3.5-0.8B.Q4_K_M.gguf`
- Check ROCm 10.1 (7.16) compatibility with GGUF format
- Test with: `./build-hip/xllm.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --ctx 2048`

### 6. Backend Selection Issues
- Verify `#ifdef RDNA2_BACKEND` and `#ifdef RDNA4_BACKEND` tags in all Phase 1-4 files
- Check CMake configuration: `-DRDNA2_BACKEND=ON -DRDNA4_BACKEND=ON`
- Test backend selection: `--backend rdna2` or `--backend rdna4`

### 7. Service Mode Issues
- Use `xllm serve` command from launch_xllm.md guide
- Verify port 8010 and master node address configuration
- Test health endpoint: `curl http://127.0.0.1:8010/health`

### Windows 11 Specific Fixes
- Ensure **Windows Defender exclusions** for build directories
- **Run as Administrator** for ROCm driver access
- Verify **GPU driver compatibility** with ROCm 10.1 (7.16)

## Supported Models (Windows 11 ROCm 10.1 (7.16))

The build supports **32 model families** with GGUF compatibility:
- **LLM**: DeepSeek, Qwen, Kimi, Llama, GLM, MiniMax series
- **VLM**: MiniCPM-V, MiMo-VL, Qwen2.5-VL, Qwen3-VL, GLM-4.6V, GLM-5.3-Flash-VL, VLM-R1
- **Rerank**: Qwen3-Reranker
- **DiT**: Flux, Flux2, Wan2.2
- **Rec**: OneRec, Qwen2, Qwen2.5, Qwen3

## Branch Information
- **Active Branch**: `arena/489ef17f-xllm-rx`
- **Base Commit**: `adad52a0243f9e0efc414bbedbe4aa7c7701cfaf` (main)
- **Conversion Status**: Phase 1-4 Complete (Windows 11 ROCm 10.1 (7.16) pure C++ RDNA2/RDNA4)
- **CPU Fallback**: Confirmed YES (maintained in pure C++ conversion)

## References
- [WINDOWS_11_RDNA2_CONVERSION_PLAN.md](WINDOWS_11_RDNA2_CONVERSION_PLAN.md)
- [sherlock-it.md](sherlock-it.md) - Performance trace requirements (example used kraken.exe, actual executable is xllm.exe/xllm_serve.exe)
- [xllm/core/kernels/rdna/utils.h](xllm/core/kernels/rdna/utils.h) - Pure C++ RDNA2/RDNA4 header
- [xllm/core/kernels/cuda/*.cpp](xllm/core/kernels/cuda/) - Converted implementations
- [docs/src/content/docs/en/getting_started/launch_xllm.md](docs/src/content/docs/en/getting_started/launch_xllm.md) - Service mode usage

## License
Apache License 2.0

## Contact
For issues or questions, refer to the [xLLM GitHub repository](https://github.com/Maxritz/xllm-RX).
