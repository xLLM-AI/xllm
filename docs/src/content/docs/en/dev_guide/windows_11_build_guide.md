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

## Performance Trace Integration (sherlock-it.md)

All build outputs include mandatory benchmark capture:
- **Operation/kernel/shader name**
- **Invocation count**
- **Total and average device time**
- **Host time**
- **Data volume, bandwidth and throughput**
- **Dispatch dimensions, workload shape and execution configuration**
- **GPU utilisation, memory usage and hardware counters**
- **End-to-end latency, throughput and unattributed time**

## CPU Fallback Maintenance

The build automatically maintains CPU fallback:
- **torch::kCPU** replaced with pure C++ CPU fallback
- **RDNA2/RDNA4 backend tags** integrated throughout
- **Windows 11 ROCm 10.1 (7.16)** compatibility ensured

## Testing and Verification

### Build Verification
```bash
# Run build tests
ctest --output-on-failure --config Release
```

### Performance Benchmark
```bash
# Run sherlock-it.md benchmark capture
./build-hip/kraken.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --bench --ctx 2048
```

### GGUF Model Support
```bash
# Test GGUF model loading (Windows 11 ROCm 10.1 (7.16))
./build-hip/kraken.exe -m models/Qwen3.5-0.8B.Q4_K_M.gguf --ctx 2048
```

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
- **Active Branch**: `arena/22748a60-xllm-rx`
- **Base Commit**: `f8fc4577f7c16bd36265d5f7b30229bb377f76bf` (main)
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
