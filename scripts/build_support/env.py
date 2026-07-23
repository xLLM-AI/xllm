import os
import platform

from typing import Optional


def get_cxx_abi() -> bool:
    try:
        import torch
        return torch.compiled_with_cxx11_abi()
    except ImportError:
        return False


def get_python_include_path() -> Optional[str]:
    try:
        from sysconfig import get_paths
        return get_paths()["include"]
    except ImportError:
        return None


def get_torch_root_path() -> Optional[str]:
    try:
        import torch
        import os
        return os.path.dirname(os.path.abspath(torch.__file__))
    except ImportError:
        return None


def get_torch_mlu_root_path() -> Optional[str]:
    try:
        import torch_mlu
        import os
        return os.path.dirname(os.path.abspath(torch_mlu.__file__))
    except ImportError:
        return None


def get_ixformer_root_path() -> Optional[str]:
    try:
        import ixformer
        import os
        return os.path.dirname(os.path.abspath(ixformer.__file__))
    except ImportError:
        return None


def get_cuda_root_path() -> Optional[str]:
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
        if CUDA_HOME is None:
            raise RuntimeError(
                "PyTorch was not built with CUDA, or nvcc is not in PATH. "
                "Please set CUDA_TOOLKIT_ROOT_DIR manually."
            )
        return CUDA_HOME
    except ImportError:
        return None

def get_dcu_root_path() -> Optional[str]:
    try:
        import torch
        from torch.utils.cpp_extension import ROCM_HOME
        if ROCM_HOME is None:
            raise RuntimeError(
                "PyTorch was not built with dcu, or hipcc is not in PATH. "
                "Please set ROCM_PATH manually."
            )
        return ROCM_HOME
    except ImportError:
        return None


def _find_dcu_so(package: str, pattern: str) -> Optional[str]:
    try:
        import glob
        import importlib.util
        import os
        spec = importlib.util.find_spec(package)
    except Exception:
        return None
    if not spec or not spec.submodule_search_locations:
        return None
    files = glob.glob(os.path.join(spec.submodule_search_locations[0], pattern))
    return files[0] if files else None


def prepend_path_env(var_name: str, path: str, sep: str = os.pathsep) -> None:
    """Prepend a path into a path env var without duplicates."""
    if not path:
        return
    current = os.getenv(var_name, "")
    entries = [item for item in current.split(sep) if item]
    if path in entries:
        entries = [item for item in entries if item != path]
    entries.insert(0, path)
    os.environ[var_name] = sep.join(entries)


def set_npu_torch_ld_library_path() -> None:
    """Only for NPU flow: ensure torch runtime libraries are discoverable."""
    torch_root = os.getenv("PYTORCH_INSTALL_PATH") or get_torch_root_path() or ""
    if not torch_root:
        return

    # Order keeps current behavior: torch.libs > torch > torch/lib
    for path in (f"{torch_root}.libs", torch_root, os.path.join(torch_root, "lib")):
        if os.path.isdir(path):
            prepend_path_env("LD_LIBRARY_PATH", path)


def set_common_envs() -> None:
    os.environ["PYTHON_INCLUDE_PATH"] = get_python_include_path() or ""
    torch_root = get_torch_root_path() or ""
    os.environ["PYTHON_LIB_PATH"] = torch_root
    os.environ["LIBTORCH_ROOT"] = torch_root
    os.environ["PYTORCH_INSTALL_PATH"] = torch_root


def set_npu_envs() -> None:
    PYTORCH_NPU_INSTALL_PATH = os.getenv("PYTORCH_NPU_INSTALL_PATH")
    if not PYTORCH_NPU_INSTALL_PATH:
        os.environ["PYTORCH_NPU_INSTALL_PATH"] = "/usr/local/libtorch_npu"

    set_common_envs()
    set_npu_torch_ld_library_path()
    NPU_TOOLKIT_HOME = os.getenv("NPU_TOOLKIT_HOME")
    if not NPU_TOOLKIT_HOME:
        os.environ["NPU_TOOLKIT_HOME"] = "/usr/local/Ascend/ascend-toolkit/latest"
        NPU_TOOLKIT_HOME = "/usr/local/Ascend/ascend-toolkit/latest"
    LD_LIBRARY_PATH = os.getenv("LD_LIBRARY_PATH", "")
    arch = platform.machine()
    LD_LIBRARY_PATH = NPU_TOOLKIT_HOME+"/lib64" + ":" + \
        NPU_TOOLKIT_HOME+"/lib64/plugin/opskernel" + ":" + \
        NPU_TOOLKIT_HOME+"/lib64/plugin/nnengine" + ":" + \
        NPU_TOOLKIT_HOME+"/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/"+arch + ":" + \
        NPU_TOOLKIT_HOME+"/opp/vendors/xllm/op_api/lib" + ":" + \
        NPU_TOOLKIT_HOME+"/tools/aml/lib64" + ":" + \
        NPU_TOOLKIT_HOME+"/tools/aml/lib64/plugin" + ":" + \
        LD_LIBRARY_PATH
    os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH
    PYTHONPATH = os.getenv("PYTHONPATH", "")
    PYTHONPATH = NPU_TOOLKIT_HOME+"/python/site-packages" + ":" + \
        NPU_TOOLKIT_HOME+"/opp/built-in/op_impl/ai_core/tbe" + ":" + \
        PYTHONPATH
    os.environ["PYTHONPATH"] = PYTHONPATH
    PATH = os.getenv("PATH", "")
    PATH = NPU_TOOLKIT_HOME+"/bin" + ":" + \
        NPU_TOOLKIT_HOME+"/compiler/ccec_compiler/bin" + ":" + \
        NPU_TOOLKIT_HOME+"/tools/ccec_compiler/bin" + ":" + \
        PATH
    os.environ["PATH"] = PATH
    os.environ["ASCEND_AICPU_PATH"] = NPU_TOOLKIT_HOME
    os.environ["ASCEND_OPP_PATH"] = NPU_TOOLKIT_HOME+"/opp"
    os.environ["TOOLCHAIN_HOME"] = NPU_TOOLKIT_HOME+"/toolkit"
    os.environ["NPU_HOME_PATH"] = NPU_TOOLKIT_HOME

    ATB_PATH = os.getenv("ATB_PATH")
    if not ATB_PATH:
        os.environ["ATB_PATH"] = "/usr/local/Ascend/nnal/atb"
        ATB_PATH = "/usr/local/Ascend/nnal/atb"


    cxx_abi = "1" if get_cxx_abi() else "0"
    ATB_HOME_PATH = os.path.join(ATB_PATH, "latest", "atb", "cxx_abi_" + cxx_abi)
    os.environ["ATB_HOME_PATH"] = ATB_HOME_PATH
    LD_LIBRARY_PATH = os.getenv("LD_LIBRARY_PATH", "")
    LD_LIBRARY_PATH = ATB_HOME_PATH+"/lib" + ":" + \
        ATB_HOME_PATH+"/examples" + ":" + \
        ATB_HOME_PATH+"/tests/atbopstest" + ":" + \
        LD_LIBRARY_PATH
    os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH
    PATH = os.getenv("PATH", "")
    PATH = ATB_HOME_PATH+"/bin" + ":" + PATH
    os.environ["PATH"] = PATH

    os.environ["ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE"] = "0"
    os.environ["ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE"] = "0"
    os.environ["ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE"] = "0"
    os.environ["ATB_OPSRUNNER_SETUP_CACHE_ENABLE"] = "1"
    os.environ["ATB_OPSRUNNER_KERNEL_CACHE_TYPE"] = "3"
    os.environ["ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT"] = "1"
    os.environ["ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT"] = "5"
    os.environ["ATB_OPSRUNNER_KERNEL_CACHE_TILING_SIZE"] = "10240"
    os.environ["ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE"] = "1"
    os.environ["ATB_WORKSPACE_MEM_ALLOC_GLOBAL"] = "0"
    os.environ["ATB_COMPARE_TILING_EVERY_KERNEL"] = "0"
    os.environ["ATB_HOST_TILING_BUFFER_BLOCK_NUM"] = "128"
    os.environ["ATB_DEVICE_TILING_BUFFER_BLOCK_NUM"] = "32"
    os.environ["ATB_SHARE_MEMORY_NAME_SUFFIX"] = ""
    os.environ["ATB_LAUNCH_KERNEL_WITH_TILING"] = "1"
    os.environ["ATB_MATMUL_SHUFFLE_K_ENABLE"] = "1"
    os.environ["ATB_RUNNER_POOL_SIZE"] = "64"
    os.environ["ASDOPS_HOME_PATH"] = ATB_HOME_PATH
    os.environ["ASDOPS_MATMUL_PP_FLAG"] = "1"
    os.environ["ASDOPS_LOG_LEVEL"] = "ERROR"
    os.environ["ASDOPS_LOG_TO_STDOUT"] = "0"
    os.environ["ASDOPS_LOG_TO_FILE"] = "1"
    os.environ["ASDOPS_LOG_TO_FILE_FLUSH"] = "0"
    os.environ["ASDOPS_LOG_TO_BOOST_TYPE"] = "atb"
    os.environ["ASDOPS_LOG_PATH"] = "~"
    os.environ["ASDOPS_TILING_PARSE_CACHE_DISABLE"] = "0"
    os.environ["LCCL_DETERMINISTIC"] = "0"
    os.environ["LCCL_PARALLEL"] = "0"


def set_mlu_envs() -> None:
    set_common_envs()
    os.environ["PYTORCH_MLU_INSTALL_PATH"] = get_torch_mlu_root_path() or ""


def set_cuda_envs() -> None:
    set_common_envs()
    os.environ["CUDA_TOOLKIT_ROOT_DIR"] = get_cuda_root_path() or ""

def set_dcu_envs() -> None:
    set_common_envs()
    os.environ["DCU_PATH"] = get_dcu_root_path() or ""
    if not os.getenv("FLASH_ATTENTION_LIB"):
        flash_attn_lib = _find_dcu_so("flash_attn", "lib/libflash_attention.so")
        if flash_attn_lib:
            os.environ["FLASH_ATTENTION_LIB"] = flash_attn_lib
    if not os.getenv("FLASH_MLA_LIB"):
        flash_mla_lib = _find_dcu_so("flash_mla", "cuda*.so")
        if flash_mla_lib:
            os.environ["FLASH_MLA_LIB"] = flash_mla_lib
    if not os.getenv("AITER_CPP_API_LIB"):
        aiter_cpp_api_lib = _find_dcu_so("aiter", "jit/module_cpp_api.so")
        if aiter_cpp_api_lib:
            os.environ["AITER_CPP_API_LIB"] = aiter_cpp_api_lib
    if not os.getenv("AITER_MOE_C_KERNEL_LIB"):
        aiter_moe_c_kernel_lib = _find_dcu_so(
            "aiter", "jit/module_moe_c_kernel.so"
        )
        if aiter_moe_c_kernel_lib:
            os.environ["AITER_MOE_C_KERNEL_LIB"] = aiter_moe_c_kernel_lib

def set_maca_envs():
    os.environ["PYTHON_INCLUDE_PATH"] = get_python_include_path()
    os.environ["PYTHON_LIB_PATH"] = get_torch_root_path()
    os.environ["LIBTORCH_ROOT"] = get_torch_root_path()
    os.environ["PYTORCH_INSTALL_PATH"] = get_torch_root_path()

    MACA_PATH = os.getenv("MACA_PATH", "/opt/maca")
    os.environ["CUCC_CMAKE_ENTRY"] = "2"
    os.environ["CUCC_PATH"] = MACA_PATH + "/tools/cu-bridge"
    os.environ["CUDA_PATH"] = MACA_PATH + "/tools/cu-bridge"
    PATH = os.getenv("PATH", "")
    PATH = MACA_PATH + "/mxgpu_llvm/bin" + ":" + \
        MACA_PATH + "/bin" + ":" + \
        MACA_PATH + "/tools/cu-bridge/bin" + ":" + \
        MACA_PATH + "/tools/cu-bridge/tools" + ":" + \
        ":" + PATH
    os.environ["PATH"] = PATH
    LD_LIBRARY_PATH = os.getenv("LD_LIBRARY_PATH", "")
    LD_LIBRARY_PATH = MACA_PATH + "/lib" + ":" + \
        MACA_PATH + "/ompi/lib" + ":" + \
        MACA_PATH + "/mxgpu_llvm/lib" + ":" + \
        MACA_PATH + "/tools/cu-bridge/lib" + ":" + \
        LD_LIBRARY_PATH
    os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH
    os.environ["PYTHON_EXECUTABLE"] = "/opt/conda/bin/python"

def set_ilu_envs() -> None:
    set_common_envs()
    os.environ["IXFORMER_INSTALL_PATH"] = get_ixformer_root_path() or ""


def set_musa_envs() -> None:
    """Configure MUSA through mcc_wrapper and the CUDA compatibility path."""
    from sysconfig import get_paths
    set_common_envs()
    import torch_musa
    from torch_musa.utils.musa_extension import MUSA_HOME as _MUSA_HOME
    musa_home = os.getenv("MUSA_HOME") or _MUSA_HOME or "/usr/local/musa"
    os.environ["MUSA_HOME"] = musa_home
    os.environ["CUDA_HOME"] = musa_home
    os.environ["CUDAToolkit_ROOT"] = musa_home
    os.environ["CUDA_TOOLKIT_ROOT_DIR"] = musa_home
    os.environ["MUSAMAPPING_PATH"] = os.path.join(
        musa_home, "tools", "musamapping"
    )

    cmake_prefix = torch_musa.core.cmake_prefix_path
    os.environ["TORCH_MUSA_PYTHONPATH"] = cmake_prefix
    os.environ["TorchMusa_DIR"] = os.path.join(cmake_prefix, "TorchMusa")

    torch_musa_root = os.path.abspath(os.path.join(cmake_prefix, "../.."))
    library_paths: list[str] = [
        os.path.join(musa_home, "lib"),
        os.path.join(torch_musa_root, "lib"),
        os.path.join(get_torch_root_path() or "", "lib"),
    ]
    python_platlib = get_paths()["platlib"]
    library_paths.append(os.path.join(python_platlib, "tvm_ffi", "lib"))

    mkl_root = os.getenv("MKLROOT")
    if mkl_root:
        os.environ.setdefault(
            "MKL_DIR", os.path.join(mkl_root, "lib", "cmake", "mkl")
        )
        library_paths.extend(
            [
                os.path.join(mkl_root, "lib", "intel64"),
                os.path.join(mkl_root, "lib"),
            ]
        )

    for path in library_paths:
        if path and os.path.isdir(path):
            prepend_path_env("LD_LIBRARY_PATH", path)
