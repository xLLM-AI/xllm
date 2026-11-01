# Windows 11 Build Guide: Pure C++ RDNA2/RDNA4 with ROCm 10.1 (7.16)

## Overview
This guide provides instructions for building xLLM-RX on **Windows 11** with **ROCm 10.1 (7.16)** targeting **RDNA2/RDNA4** GPUs using **pure C++** (HIP) with **CPU fallback maintenance**.

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
```bash
# Basic execution with GGUF model (Windows 11 ROCm 10.1 (7.16))
./build-hip/kraken.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --ctx 2048

# With performance tracing (sherlock-it.md requirements)
KRK_TIME=1 ./build-hip/kraken.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --bench --ctx 2048

# CPU fallback mode (when GPU unavailable)
./build-hip/kraken.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --cpu-fallback --ctx 2048

# RDNA2/RDNA4 backend selection
./build-hip/kraken.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --backend rdna2 --ctx 2048
./build-hip/kraken.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --backend rdna4 --ctx 2048

# Batch operations (Phase 2)
./build-hip/kraken.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --batch-size 32 --ctx 2048

# FP8 scaled operations (Phase 4)
./build-hip/kraken.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --fp8-scaled --ctx 2048

# All options combined (Windows 11 ROCm 10.1 (7.16) Pure C++ RDNA2/RDNA4)
./build-hip/kraken.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf \
  --bench --ctx 2048 --cpu-fallback --backend rdna4 --batch-size 32 --fp8-scaled
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

[Detailed Build Guide](docs/src/content/docs/en/dev_guide/windows_11_build_guide.md) | [Supported Models](https://docs.xllm-ai.com/en/supported_models/)

## Troubleshooting

### Common Issues
1. **ROCm Path Not Found**: Ensure `ROCM_PATH` is correctly set
2. **HIP Compiler Errors**: Verify `clang-cl` compatibility with ROCm 10.1 (7.16)
3. **CPU Fallback Failures**: Check `torch::kCPU` replacement in pure C++ conversion
4. **Performance Trace Missing**: Verify sherlock-it.md integration in all kernels

### Windows 11 Specific Fixes
- Ensure Windows Defender exclusions for build directories
- Run as Administrator for ROCm driver access
- Verify GPU driver compatibility with ROCm 10.1 (7.16)

## Supported Models (Windows 11 ROCm 10.1 (7.16))

The build supports **32 model families** with GGUF compatibility:
- **LLM**: DeepSeek, Qwen, Kimi, Llama, GLM, MiniMax series
- **VLM**: MiniCPM-V, MiMo-VL, Qwen2.5-VL, Qwen3-VL, GLM-4.6V, GLM-5.3-Flash-VL, VLM-R1
- **Rerank**: Qwen3-Reranker
- **DiT**: Flux, Flux2, Wan2.2
- **Rec**: OneRec, Qwen2, Qwen2.5, Qwen3

## Branch Information
- **Active Branch**: `arena/671fc17d-xllm-rx`
- **Base Commit**: `f948232a7be58cdaee946e52b1bc9da3ae125df7` (main)
- **Conversion Status**: Phase 1-4 Complete (Windows 11 ROCm 10.1 (7.16) pure C++ RDNA2/RDNA4)
- **CPU Fallback**: Confirmed YES (maintained in pure C++ conversion)

## References
- [WINDOWS_11_RDNA2_CONVERSION_PLAN.md](WINDOWS_11_RDNA2_CONVERSION_PLAN.md)
- [sherlock-it.md](sherlock-it.md) - Performance trace requirements
- [xllm/core/kernels/rdna/utils.h](xllm/core/kernels/rdna/utils.h) - Pure C++ RDNA2/RDNA4 header
- [xllm/core/kernels/cuda/*.cpp](xllm/core/kernels/cuda/) - Converted implementations

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
- Ensure MSVC 2022 (17.8+) or Clang for Windows is installed
- Verify `clang-cl` compatibility with ROCm 10.1 (7.16)
- Check Windows Defender exclusions for build directories

### 3. CPU Fallback Verification
- Confirm `torch::kCPU` replacement in pure C++ conversion
- Verify `#ifdef CPU_FALLBACK` tags in all converted files
- Test fallback mode: `./build-hip/kraken.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --cpu-fallback --ctx 2048`

### 4. Performance Trace Issues
- Verify sherlock-it.md integration in all kernels
- Check `KRK_TIME=1` environment variable for benchmark capture
- Ensure operation/kernel/shader name, invocation count, timing, etc. are captured

### 5. GGUF Model Loading
- Verify model path: `models/Qwen3.5-0.8B.Q4_K_M.gguf`
- Check ROCm 10.1 (7.16) compatibility with GGUF format
- Test with: `./build-hip/kraken.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --ctx 2048`

### 6. Backend Selection Issues
- Verify `#ifdef RDNA2_BACKEND` and `#ifdef RDNA4_BACKEND` tags
- Check CMake configuration: `-DRDNA2_BACKEND=ON -DRDNA4_BACKEND=ON`
- Test backend selection: `--backend rdna2` or `--backend rdna4`

## Contact for Support
If issues persist, refer to the [xLLM GitHub repository](https://github.com/Maxritz/xllm-RX) or check the [WINDOWS_11_RDNA2_CONVERSION_PLAN.md](WINDOWS_11_RDNA2_CONVERSION_PLAN.md) for phase-specific details.

## License
Apache License 2.0
