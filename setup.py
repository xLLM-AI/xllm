import glob
import io
import os
import re
import shutil
import subprocess
import sys
import argparse
from typing import Any, Optional

from setuptools import Command, Extension, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ModuleNotFoundError:
    from wheel.bdist_wheel import bdist_wheel

from scripts.build_support.env import (
    get_cxx_abi,
    get_torch_root_path,
    set_cuda_envs,
    set_ilu_envs,
    set_maca_envs,
    set_mlu_envs,
    set_musa_envs,
    set_npu_envs,
    set_dcu_envs,
)
from scripts.build_support.utils import (
    check_and_install_pre_commit,
    get_ascend_platform,
    get_base_dir,
    get_cmake_dir,
    get_cpu_arch,
    get_device_type,
    get_python_version,
    get_torch_cmake_prefix_path,
    get_torch_version,
    get_version,
    pre_build,
    read_readme,
)
from scripts.logger import logger

BUILD_TEST_FILE: bool = True
BUILD_EXPORT: bool = True


def _ensure_tilelang_ascend_ready(target_platform: str, arch: str) -> None:
    compiler_parent = os.path.join(get_base_dir(), "xllm")
    if compiler_parent not in sys.path:
        sys.path.insert(0, compiler_parent)
    from compiler.tilelang.bootstrap import prepare_ascend

    prepare_ascend(target_platform, arch)


def _maybe_compile_tilelang_kernels(device: str, jobs: int | str | None = None) -> None:
    if device != "npu":
        return
    target_platform = get_ascend_platform()

    output_root = os.path.join(get_cmake_dir(), "xllm", "compiler", "tilelang")
    os.makedirs(output_root, exist_ok=True)

    env = os.environ.copy()
    base_dir = get_base_dir()

    cmd = [
        sys.executable,
        os.path.join(base_dir, "xllm", "compiler", "tilelang_launcher.py"),
        "compile-kernels",
        "--target",
        "ascend",
        "--output-root",
        output_root,
        "--device",
        target_platform,
    ]
    if jobs is not None:
        cmd.extend(["--jobs", str(jobs)])
    logger.info("compiling TileLang kernels via source-tree launcher")
    subprocess.check_call(cmd, cwd=base_dir, env=env)


def _stage_triton_npu_runtime_binaries(base_dir: str, extdir: str, device: str) -> None:
    if device != "npu":
        return

    env_path = os.environ.get("TRITON_BINARY_PATH")
    if env_path and os.path.isdir(env_path):
        source_dir = env_path
    else:
        source_dir = os.path.join(
            get_cmake_dir(),
            "third_party",
            "torch_npu_ops",
            "triton_npu",
            "binary",
        )

    if not os.path.isdir(source_dir):
        raise RuntimeError(
            f"Triton NPU binary directory does not exist: {source_dir}\n"
            "Hint: Ensure the CMake build completed successfully, or set "
            "TRITON_BINARY_PATH to a directory containing pre-built binaries."
        )

    dest_dir = os.path.join(extdir, "triton_npu", "binary")
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    copied_count = 0
    for item in sorted(os.listdir(source_dir)):
        if not item.endswith((".npubin", ".json")):
            continue
        source_path = os.path.join(source_dir, item)
        if not os.path.isfile(source_path):
            continue
        shutil.copy2(source_path, os.path.join(dest_dir, item))
        copied_count += 1

    if copied_count == 0:
        raise RuntimeError(
            f"No Triton NPU runtime binaries were found under: {source_dir}"
        )
    logger.info(f"Staged {copied_count} Triton NPU runtime asset(s) into {dest_dir}")


def _stage_mlu_triton_kernels(base_dir: str, extdir: str, device: str) -> None:
    if device != "mlu":
        return

    source_dir = os.path.join(
        base_dir, "xllm", "core", "kernels", "mlu", "triton_kernel"
    )
    if not os.path.isdir(source_dir):
        raise RuntimeError(
            f"MLU triton_kernel directory does not exist: {source_dir}\n"
            "Hint: Ensure the source tree is intact (xllm/core/kernels/mlu/"
            "triton_kernel must contain the JIT kernel .py files)."
        )

    # extdir already points at the installed ``xllm`` package dir (it is the
    # dirname of ``get_ext_fullpath("xllm/")`` and ends in ".../xllm"), so the
    # kernel package lands directly under it as ``xllm.core.kernels.mlu.
    # triton_kernel.<name>``. Do NOT add another ``xllm`` segment here -- the
    # CMake ``xllm`` product is written into extdir itself as a *file*, and
    # ``extdir/xllm`` would collide with it (NotADirectoryError).
    dest_dir = os.path.join(
        extdir, "core", "kernels", "mlu", "triton_kernel"
    )
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    copied_count = 0
    for item in sorted(os.listdir(source_dir)):
        if not item.endswith(".py"):
            continue
        source_path = os.path.join(source_dir, item)
        if not os.path.isfile(source_path):
            continue
        shutil.copy2(source_path, os.path.join(dest_dir, item))
        copied_count += 1

    if copied_count == 0:
        raise RuntimeError(
            f"No MLU triton kernel .py files were found under: {source_dir}"
        )
    logger.info(f"Staged {copied_count} MLU triton kernel(s) into {dest_dir}")


def _stage_triton_jit_scripts(base_dir: str, extdir: str) -> None:
    """Stage the triton_jit compile script as an importable package module.

    triton_compile.py drives JIT compilation and signature dumps from C++ via a
    direct ``import`` in the embedded interpreter
    (``xllm.core.triton_jit.scripts.triton_compile``). Shipping it inside the
    installed xllm package -- instead of baking a build-tree ``__FILE__`` path
    -- keeps the script resolvable after ``pip install``.
    """
    source_dir = os.path.join(
        base_dir, "xllm", "core", "triton_jit", "scripts"
    )
    source_script = os.path.join(source_dir, "triton_compile.py")
    if not os.path.isfile(source_script):
        raise RuntimeError(
            f"triton_jit compile script does not exist: {source_script}\n"
            "Hint: Ensure the source tree is intact (xllm/core/triton_jit/"
            "scripts/triton_compile.py must exist)."
        )

    # Only the script ships; the C++ sources (src/, include/) stay out of the
    # wheel. __init__.py files make xllm.core.triton_jit.scripts.triton_compile
    # importable. Create-if-missing so we never overwrite a packaged xllm
    # __init__.py or a source-tree file.
    # extdir already points at the installed ``xllm`` package dir, so the
    # scripts package lands directly under it. Do NOT add another ``xllm``
    # segment (see _stage_mlu_triton_kernels for the collision rationale).
    dest_dir = os.path.join(
        extdir, "core", "triton_jit", "scripts"
    )
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(source_script, os.path.join(dest_dir, "triton_compile.py"))

    pkg_init = (
        "# Copyright 2026 The xLLM Authors. All Rights Reserved.\n"
        "# Licensed under the Apache License, Version 2.0 (the \"License\").\n"
        "# See LICENSE for details.\n"
        "\"\"\"xllm internal package marker.\"\"\"\n"
    )
    for pkg_dir in (
        os.path.join(extdir, "core"),
        os.path.join(extdir, "core", "triton_jit"),
        dest_dir,
    ):
        os.makedirs(pkg_dir, exist_ok=True)
        init_path = os.path.join(pkg_dir, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w", encoding="utf-8") as f:
                f.write(pkg_init)

    logger.info(f"Staged triton_jit compile script into {dest_dir}")

class CMakeExtension(Extension):
    def __init__(self, name: str, path: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.path.realpath(os.path.abspath(sourcedir))
        self.path = path

class ExtBuild(build_ext):
    user_options = build_ext.user_options + [
        ("base-dir=", None, "base directory of xLLM project"),
        ("device=", None, "target device type (npu or mlu or cuda or ilu or musa or maca)"),
        ("arch=", None, "target arch type (x86 or arm)"),
        ("generate-so=", None, "generate so or binary"),
        ("tilelang-jobs=", None, "maximum parallel TileLang compile workers"),
    ]

    def initialize_options(self) -> None:
        build_ext.initialize_options(self)
        self.base_dir = get_base_dir()
        self.device: Optional[str] = None
        self.arch: Optional[str] = None
        self.generate_so: bool = False
        self.tilelang_jobs: int | str | None = None

    def finalize_options(self) -> None:
        build_ext.finalize_options(self)

    def run(self) -> None:
        # check if cmake is installed
        try:
            out: bytes = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )
            exit(1)

        match = re.search(
            r"version\s*(?P<major>\d+)\.(?P<minor>\d+)([\d.]+)?", out.decode()
        )
        if match is None:
            raise RuntimeError(f"Failed to parse CMake version from: {out!r}")
        cmake_major, cmake_minor = int(match.group("major")), int(match.group("minor"))
        if (cmake_major, cmake_minor) < (3, 18):
            raise RuntimeError("CMake >= 3.18.0 is required")

        try:
            # build extensions
            for ext in self.extensions:
                self.build_extension(ext)
        except Exception:
            logger.exception("Build failed.")
            exit(1)

    def build_extension(self, ext: CMakeExtension) -> None:
        ninja_dir = shutil.which("ninja")
        # the output dir for the extension
        extdir: str = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.path)))

        # create build directory
        os.makedirs(self.build_temp, exist_ok=True)

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug: int = int(os.environ.get("DEBUG", 0)) if self.debug is None else int(self.debug)
        build_type: str = "Debug" if debug else "Release"

        default_jobs = os.cpu_count() or 1
        max_jobs: str = os.getenv("MAX_JOBS", str(default_jobs))
        max_jobs_int: int = int(max_jobs)

        # Limit archive (ar/ranlib) concurrency to avoid file locking conflicts.
        # The ar tool requires exclusive access to archive files (.a files) when
        # creating or updating static libraries. When multiple ar processes attempt
        # to modify the same archive file simultaneously, they compete for file locks,
        # which can cause deadlocks and hang the build process.
        archive_jobs: int = min(8, max(1, max_jobs_int // 4))

        if self.device is None:
            raise ValueError("Please set --device to npu, mlu, cuda, dcu, ilu or musa.")
        if self.arch is None:
            raise ValueError("Please set --arch to x86 or arm.")

        cmake_args: list[str] = [
            "-G",
            "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={ninja_dir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}",
            "-DUSE_CCACHE=ON",
            f"-DPython_EXECUTABLE:FILEPATH={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DBUILD_SHARED_LIBS=OFF",
            f"-DDEVICE_TYPE=USE_{self.device.upper()}",
            f"-DDEVICE_ARCH={self.arch.upper()}",
            f"-DXLLM_ATB_LAYERS_SOURCE_DIR={os.path.join(self.base_dir, 'third_party', 'xllm_atb_layers')}",
            f"-DCMAKE_JOB_POOLS=archive={archive_jobs}",
        ]
        if self.device != 'maca':
            cmake_args += ["-DUSE_CCACHE=ON"]

        if self.device == "npu":
            cmake_args += ["-DUSE_NPU=ON"]
            set_npu_envs()
            _maybe_compile_tilelang_kernels(self.device, self.tilelang_jobs)
        elif self.device == "mlu":
            cmake_args += ["-DUSE_MLU=ON"]
            set_mlu_envs()
        elif self.device == "musa":
            torch_cuda_architectures = os.getenv("TORCH_CUDA_ARCH_LIST")
            if not torch_cuda_architectures:
                torch_cuda_architectures = "9.0"
            cmake_args += [
                "-DUSE_MUSA=ON",
                "-DUSE_CUDA=ON",
                f"-DTORCH_CUDA_ARCH_LIST={torch_cuda_architectures}",
                "-DCMAKE_CUDA_ARCHITECTURES=90",
                "-DBUILD_TESTING=OFF",
            ]
            set_musa_envs()
            global BUILD_TEST_FILE
            BUILD_TEST_FILE = False
        elif self.device == "cuda":
            torch_cuda_architectures = os.getenv("TORCH_CUDA_ARCH_LIST")
            if not torch_cuda_architectures:
                raise ValueError("Please set TORCH_CUDA_ARCH_LIST environment variable, e.g. export TORCH_CUDA_ARCH_LIST=\"8.0 8.9 9.0 10.0 12.0\"")
            cmake_args += ["-DUSE_CUDA=ON",
                           f"-DTORCH_CUDA_ARCH_LIST={torch_cuda_architectures}"]
            set_cuda_envs()

        elif self.device == "dcu":
            import torch

            if getattr(torch.version, "hip", None):
                dcu_arch = os.getenv("PYTORCH_ROCM_ARCH") or os.getenv("CMAKE_HIP_ARCHITECTURES")
                torch_root = get_torch_root_path()
                if not torch_root:
                    raise RuntimeError("Unable to locate PyTorch package directory.")
                cmake_args += [
                    "-DUSE_DCU=ON",
                    f"-DROCM_PATH={os.getenv('DCU_PATH', '/opt/dtk')}",
                    f"-DTORCH_CMAKE_PREFIX={get_torch_cmake_prefix_path()}",
                    f"-DTORCH_PKG_DIR={torch_root}",
                ]
                if dcu_arch:
                    cmake_args += [f"-DCMAKE_HIP_ARCHITECTURES={dcu_arch}"]

                set_dcu_envs()

                # Pass FLASH_ATTENTION_LIB to CMake so the DCU layers can
                # link against libflash_attention.so (prefix prefill/decode).
                flash_attn_lib = os.getenv("FLASH_ATTENTION_LIB")
                if flash_attn_lib:
                    cmake_args += [f"-DFLASH_ATTENTION_LIB={flash_attn_lib}"]

                flash_mla_lib = os.getenv("FLASH_MLA_LIB")
                if flash_mla_lib:
                    cmake_args += [f"-DFLASH_MLA_LIB={flash_mla_lib}"]

                aiter_cpp_api_lib = os.getenv("AITER_CPP_API_LIB")
                if aiter_cpp_api_lib:
                    cmake_args += [f"-DAITER_CPP_API_LIB={aiter_cpp_api_lib}"]

                aiter_moe_c_kernel_lib = os.getenv("AITER_MOE_C_KERNEL_LIB")
                if aiter_moe_c_kernel_lib:
                    cmake_args += [
                        f"-DAITER_MOE_C_KERNEL_LIB={aiter_moe_c_kernel_lib}"
                    ]
            else:
                raise RuntimeError(
                    "DCU build requires a HIP/ROCm PyTorch environment. "
                    "Please install a PyTorch build with torch.version.hip set, "
                    "or choose a different --device."
                )
        elif self.device == "maca":
            torch_cuda_architectures = os.getenv("TORCH_CUDA_ARCH_LIST")
            if not torch_cuda_architectures:
                torch_cuda_architectures = "8.0 8.6+PTX"
            cmake_args += ["-DUSE_CUDA=ON",
                           "-DUSE_MACA=ON",
                           "-DCMAKE_CUDA_STANDARD=17",
                           "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
                           f"-DTORCH_CUDA_ARCH_LIST={torch_cuda_architectures}"]
            set_maca_envs()
        elif self.device == "ilu":
            cmake_args += ["-DUSE_ILU=ON"]
            set_ilu_envs()
        else:
            raise ValueError("Please set --device to npu, mlu, cuda, dcu, ilu, musa or maca.")

        product: str = "xllm"
        if self.generate_so:
            product = "libxllm.so"
            cmake_args += ["-DGENERATE_SO=ON"]
        else:
            cmake_args += ["-DGENERATE_SO=OFF"]

        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # check if torch binary is built with cxx11 abi
        if get_cxx_abi():
            cmake_args += ["-DUSE_CXX11_ABI=ON", "-D_GLIBCXX_USE_CXX11_ABI=1"]
        else:
            cmake_args += ["-DUSE_CXX11_ABI=OFF", "-D_GLIBCXX_USE_CXX11_ABI=0"]

        build_args = ["--config", build_type]
        build_args += ["-j" + max_jobs]

        env: dict[str, str] = os.environ.copy()
        env["VCPKG_MAX_CONCURRENCY"] = str(max_jobs)
        logger.info(f"CMake Args: {cmake_args}")
        logger.info(f"Env: {env}")

        self.build_cmake_targets(ext, cmake_args, build_args, env, extdir, product)

    def build_cmake_targets(
        self,
        ext: CMakeExtension,
        cmake_args: list[str],
        build_args: list[str],
        env: dict[str, str],
        extdir: str,
        product: str,
    ) -> None:
        """Build CMake targets"""
        cmake_dir = get_cmake_dir()
        cmake_cmd = "cmake_maca" if self.device == "maca" else "cmake"
        subprocess.check_call([cmake_cmd, self.base_dir] + cmake_args, cwd=cmake_dir, env=env)

        base_build_args = build_args
        # add build target to speed up the build process
        build_args += ["--target", ext.name, "xllm"]
        subprocess.check_call([cmake_cmd, "--build", ".", "--verbose"] + build_args, cwd=cmake_dir)

        os.makedirs(os.path.join(os.path.dirname(cmake_dir), "xllm/core/server/"), exist_ok=True)
        shutil.copy(
            os.path.join(extdir, product),
            os.path.join(os.path.dirname(cmake_dir), "xllm/core/server/"),
        )

        # Stage the Python model-executor package into the wheel as the
        # ``xllm.python`` subpackage (xllm/python/...). The installed ``xllm``
        # package is a regular package, so ``import xllm.python`` resolves
        # straight from site-packages with no sys.path manipulation;
        # --python_model_path / XLLM_PYTHON_MODEL_PATH only overrides the
        # directory containing the ``xllm`` package (e.g. source-tree runs).
        py_pkg_src = os.path.join(self.base_dir, "xllm", "python")
        if os.path.isdir(py_pkg_src):
            py_pkg_dst = os.path.join(extdir, "python")
            if os.path.isdir(py_pkg_dst):
                shutil.rmtree(py_pkg_dst)
            shutil.copytree(
                py_pkg_src,
                py_pkg_dst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

        _stage_triton_npu_runtime_binaries(self.base_dir, extdir, self.device)

        _stage_mlu_triton_kernels(self.base_dir, extdir, self.device)

        _stage_triton_jit_scripts(self.base_dir, extdir)

        if BUILD_EXPORT:
            # build export module
            build_args = base_build_args + ["--target export_module"]
            subprocess.check_call([cmake_cmd, "--build", ".", "--verbose"] + build_args, cwd=cmake_dir)

        if BUILD_TEST_FILE:
            # build tests target
            build_args = base_build_args + ["--target all_tests"]
            subprocess.check_call([cmake_cmd, "--build", ".", "--verbose"] + build_args, cwd=cmake_dir)

class ExtBuildSingleTest(ExtBuild):
    """Inherit ExtBuild, used to build and run a single test"""
    user_options = ExtBuild.user_options + [
        ("test-name=", None, "name of the test target to build and run"),
    ]

    def initialize_options(self) -> None:
        ExtBuild.initialize_options(self)
        self.test_name: Optional[str] = None

    def finalize_options(self) -> None:
        ExtBuild.finalize_options(self)
        if not self.test_name:
            raise ValueError("--test-name is required for ExtBuildSingleTest")

    def build_cmake_targets(
        self,
        ext: CMakeExtension,
        cmake_args: list[str],
        build_args: list[str],
        env: dict[str, str],
        extdir: str,
        product: str,
    ) -> None:
        """Override method: only build the specified test target and run"""
        cmake_dir = get_cmake_dir()
        subprocess.check_call(["cmake", self.base_dir] + cmake_args, cwd=cmake_dir, env=env)

        base_build_args = build_args
        # Only build the specified test target
        build_args += ["--target", self.test_name]
        subprocess.check_call(["cmake", "--build", ".", "--verbose"] + build_args, cwd=cmake_dir)

        # Find test executable
        # CMake usually places executables in CMAKE_RUNTIME_OUTPUT_DIRECTORY or build directory
        test_executable: Optional[str] = None
        possible_paths: list[str] = [
            os.path.join(cmake_dir, self.test_name),
            os.path.join(extdir, self.test_name),
            os.path.join(cmake_dir, "xllm", "core", self.test_name),
        ]

        # Check possible paths first
        for path in possible_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                test_executable = path
                break

        # If not found, try recursive search in build directory
        if not test_executable:
            for root, dirs, files in os.walk(cmake_dir):
                if self.test_name in files:
                    candidate = os.path.join(root, self.test_name)
                    if os.access(candidate, os.X_OK):
                        test_executable = candidate
                        break

        if not test_executable:
            # If not found, try using ctest to run
            logger.warning(f"⚠️  Could not find test executable {self.test_name}, trying ctest...")
            try:
                subprocess.check_call(
                    ["ctest", "-R", self.test_name, "--verbose"],
                    cwd=cmake_dir,
                    env=env
                )
                logger.info(f"✅ Test {self.test_name} passed!")
            except subprocess.CalledProcessError:
                logger.exception(f"❌ Failed to run test {self.test_name}")
                raise
        else:
            logger.info(f"🚀 Running test: {test_executable}")
            try:
                subprocess.check_call([test_executable], cwd=os.path.dirname(test_executable), env=env)
                logger.info(f"✅ Test {self.test_name} passed!")
            except subprocess.CalledProcessError as e:
                logger.exception(f"❌ Test {self.test_name} failed with exit code {e.returncode}")
                raise

class BuildDistWheel(bdist_wheel):
    user_options = bdist_wheel.user_options + [
        ("device=", None, "target device type (npu or mlu or cuda or ilu or musa)"),
        ("arch=", None, "target arch type (x86 or arm)"),
        ("tilelang-jobs=", None, "maximum parallel TileLang compile workers"),
    ]

    def initialize_options(self) -> None:
        super().initialize_options()
        self.device: Optional[str] = None
        self.arch: Optional[str] = None
        self.tilelang_jobs: int | str | None = None
        # Cache the original dist name early so finalize_options is idempotent
        # and so name changes are visible to egg_info/metadata generation.
        self._base_dist_name = self.distribution.metadata.name

    def finalize_options(self) -> None:
        # IMPORTANT: mutate distribution name BEFORE super().finalize_options().
        # bdist_wheel finalization may finalize/cache egg_info metadata; if we
        # change the name afterwards, the wheel filename and METADATA can diverge
        # (pip will reject the wheel as "inconsistent name").
        name = self._base_dist_name

        # generate distribution name suffix
        if self.device:
            name += f"_{self.device}"

        torch_version = get_torch_version(self.device)
        if torch_version:
            name += f"_torch{torch_version}"

        self.distribution.metadata.name = name
        super().finalize_options()

    def run(self) -> None:
        build_ext_cmd = self.get_finalized_command('build_ext')
        build_ext_cmd.device = self.device
        build_ext_cmd.arch = self.arch
        build_ext_cmd.tilelang_jobs = self.tilelang_jobs

        logger.info("🔨 build project...")
        self.run_command('build')

        logger.info("🧪 testing UT...")
        self.run_command('test')

        if self.arch == 'arm':
            ext_path = get_base_dir() + f"/build/lib.linux-aarch64-cpython-{get_python_version()}/"
        else:
            ext_path = get_base_dir() + f"/build/lib.linux-x86_64-cpython-{get_python_version()}/"
        if len(ext_path) == 0:
            logger.error("❌ Build wheel failed, not found path.")
            exit(1)
        tmp_path = os.path.join(ext_path, 'xllm')
        for root, dirs, files in os.walk(tmp_path):
            for item in files:
                path = os.path.join(root, item)
                if '_test' in item and os.path.isfile(path):
                    os.remove(path)
        global BUILD_TEST_FILE
        BUILD_TEST_FILE = False

        self.skip_build = True
        InstallWheel._building_wheel = True
        try:
            super().run()
        finally:
            InstallWheel._building_wheel = False

class InstallWheel(install):
    """`python setup.py install` builds the wheel, then pip-installs it.

    bdist_wheel re-invokes the install command internally to stage files into
    its own build directory; the _building_wheel guard makes that re-entrant
    call fall back to the standard install behavior to avoid infinite recursion.
    """
    _building_wheel = False

    def run(self) -> None:
        if InstallWheel._building_wheel:
            super().run()
            return

        InstallWheel._building_wheel = True
        try:
            logger.info("📦 building wheel before install...")
            self.run_command("bdist_wheel")
        finally:
            InstallWheel._building_wheel = False

        wheel_path = self._locate_built_wheel()
        if not wheel_path:
            logger.error("❌ Install failed: no built wheel found.")
            exit(1)

        logger.info(f"⬇️  installing wheel: {wheel_path}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "--force-reinstall", "--no-deps", wheel_path,
            ])
        except subprocess.CalledProcessError:
            logger.exception(f"❌ Failed to install wheel: {wheel_path}")
            exit(1)
        logger.info(f"✅ Installed {os.path.basename(wheel_path)}")

    def _locate_built_wheel(self) -> Optional[str]:
        # bdist_wheel records produced artifacts in distribution.dist_files.
        wheels = [
            path
            for command, _, path in self.distribution.dist_files
            if command == "bdist_wheel" and path.endswith(".whl") and os.path.isfile(path)
        ]
        if wheels:
            return max(wheels, key=os.path.getmtime)

        # Fallback: scan the wheel output directory directly.
        bdist_wheel_cmd = self.get_finalized_command("bdist_wheel")
        dist_dir = getattr(bdist_wheel_cmd, "dist_dir", None) or "dist"
        candidates = glob.glob(os.path.join(dist_dir, "*.whl"))
        if candidates:
            return max(candidates, key=os.path.getmtime)
        return None

class TestUT(Command):
    description = "Run all testing binary."
    user_options = []

    # Whitelist: tests that must run sequentially (not in parallel with others)
    # Add test names here if they use fork() or have device initialization conflicts
    # Note: Use test case name patterns (from gtest), not executable names
    SEQUENTIAL_TESTS = [
        'ReduceScatterMultiDeviceTest',
        'BroadcastMultiDeviceTest',
        'DeepEPMultiDeviceTest',
        'AttentionMultiDeviceTest',
        'FusedMoEAll2AllMultiDeviceTest',
    ]

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run_ctest(self, cmake_dir: str) -> int:
        default_parallel: int = max(os.cpu_count() or 1, 8)
        test_parallel: str = os.getenv("CTEST_PARALLEL", str(default_parallel))
        logger.info(f"Test parallelism: {test_parallel} (set CTEST_PARALLEL to override)")

        def run_subprocess_with_streaming(
            cmd: list[str],
            error_message: str,
            warn_if_no_tests: bool = False,
        ) -> None:
            """Helper function to run subprocess and stream output"""
            process = subprocess.Popen(
                cmd,
                cwd=cmake_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            if process.stdout is None:
                raise RuntimeError("Failed to capture subprocess stdout for streaming.")

            output_lines: list[str] = []
            for line in iter(process.stdout.readline, ''):
                # Stream raw subprocess output as-is to preserve ctest formatting.
                sys.stdout.write(line)
                sys.stdout.flush()
                output_lines.append(line)

            return_code: int = process.wait()

            # Warn if no tests were found, but don't fail (some backends may not compile certain tests)
            if warn_if_no_tests and return_code == 0:
                output_text: str = ''.join(output_lines)
                if 'No tests were found' in output_text:
                    logger.warning("No tests matched the pattern (this is OK for some backends).")
                    return

            if return_code != 0:
                logger.error(error_message)
                raise subprocess.CalledProcessError(return_code, cmd)

        try:
            # Step 1: Run all tests EXCEPT sequential ones in parallel
            if self.SEQUENTIAL_TESTS:
                exclude_pattern = '|'.join(self.SEQUENTIAL_TESTS)
                logger.info("=" * 80)
                logger.info(f"Running tests in parallel (excluding: {', '.join(self.SEQUENTIAL_TESTS)})...")
                logger.info("=" * 80)
                run_subprocess_with_streaming(
                    ['ctest', '--parallel', test_parallel, '--repeat', 'until-pass:5', '-E', exclude_pattern],
                    "Parallel tests failed."
                )
            else:
                logger.info("=" * 80)
                logger.info("Running all tests in parallel...")
                logger.info("=" * 80)
                run_subprocess_with_streaming(
                    ['ctest', '--parallel', test_parallel, '--repeat', 'until-pass:5'],
                    "Parallel tests failed."
                )

            # Step 2: Run sequential tests one by one
            for idx, test_name in enumerate(self.SEQUENTIAL_TESTS, start=2):
                logger.info("=" * 80)
                logger.info(f"Step {idx}: Running {test_name} sequentially...")
                logger.info("=" * 80)
                # Use pattern matching to include all test cases under the test class
                # e.g., ReduceScatterMultiDeviceTest matches ReduceScatterMultiDeviceTest.BasicTest, etc.
                run_subprocess_with_streaming(
                    ['ctest', '--repeat', 'until-pass:5', '-R', test_name],
                    f"Sequential test {test_name} failed.",
                    warn_if_no_tests=True
                )

            logger.info("=" * 80)
            logger.info("All tests passed!")
            logger.info("=" * 80)
            return 0
        except subprocess.CalledProcessError as e:
            logger.exception(f"ctest failed: {e.stderr}")
            exit(1)

    def run(self) -> None:
        self.run_ctest(get_cmake_dir())

class SingleTest(Command):
    """Command to build and run a single test"""
    description = "Build and run a single test target."
    # test_name should match a CMake/CTest target name, for example:
    #   python setup.py test --test-name common_test
    user_options = [
        ("test-name=", None, "name of the test target to build and run (e.g. platform_vmm_test)"),
        ("device=", None, "target device type (npu or mlu or cuda or ilu)"),
        ("arch=", None, "target arch type (x86 or arm)"),
        ("generate-so=", None, "generate so or binary"),
        ("tilelang-jobs=", None, "maximum parallel TileLang compile workers"),
    ]

    def initialize_options(self) -> None:
        self.test_name: Optional[str] = None
        self.device: Optional[str] = None
        self.arch: Optional[str] = None
        self.generate_so: bool = False
        self.tilelang_jobs: int | str | None = None

    def finalize_options(self) -> None:
        if not self.test_name:
            raise ValueError("--test-name is required for single_test command")

    def run(self) -> None:
        # Create ExtBuildSingleTest instance and set parameters
        build_ext = ExtBuildSingleTest(self.distribution)
        build_ext.initialize_options()
        build_ext.test_name = self.test_name
        build_ext.device = self.device
        build_ext.arch = self.arch
        build_ext.generate_so = self.generate_so
        build_ext.tilelang_jobs = self.tilelang_jobs
        build_ext.finalize_options()

        # Ensure extension modules are set
        if not hasattr(build_ext, 'extensions') or not build_ext.extensions:
            build_ext.extensions = self.distribution.ext_modules

        # Run build
        build_ext.run()

def parse_arguments() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description='Setup helper for building xllm',
        epilog='Example: python setup.py build',
        usage='%(prog)s [COMMAND] [OPTIONS]'
    )

    parser.add_argument(
        'setup_args',
        nargs='*',
        metavar='argparse.REMAINDER',
        help='setup command (build, test, bdist_wheel, etc.)'
    )

    parser.add_argument(
        '--device',
        type=str.lower,
        choices=['auto', 'npu', 'mlu', 'cuda', 'ilu', 'musa', 'dcu', 'maca'],
        default='auto',
        help='Device type: npu, mlu, ilu, cuda or musa or maca (case-insensitive)'
    )
    parser.add_argument(
        '--generate-so',
        type=str.lower,
        choices=['true', 'false', '1', '0', 'yes', 'no', 'y', 'n', 'on', 'off'],
        default='false',
        help='Whether to generate so or binary'
    )

    parser.add_argument(
        '--test-name',
        type=str,
        default=None,
        help='Name of the test target to build and run; when omitted, all tests run'
    )
    parser.add_argument(
        '--tilelang-jobs',
        type=int,
        default=None,
        help='Maximum parallel TileLang compile workers, e.g. --tilelang-jobs 16; auto-selects a safe default when omitted'
    )

    args = parser.parse_args()

    sys.argv = [sys.argv[0]] + args.setup_args

    generate_so = args.generate_so.lower() in ('true', '1', 'yes', 'y', 'on')

    return {
        'device': args.device,
        'generate_so': generate_so,
        'test_name': args.test_name,
        'tilelang_jobs': args.tilelang_jobs,
    }

if __name__ == "__main__":
    config = parse_arguments()

    arch = get_cpu_arch()
    device = config['device']
    if device == 'auto':
        device = get_device_type()
    target_platform = get_ascend_platform() if device == "npu" else None
    logger.info(f"🚀 Build xllm with CPU arch: {arch} and target device: {device}")

    if device == "npu":
        _ensure_tilelang_ascend_ready(target_platform, arch)
    pre_build(device)

    generate_so = config['generate_so']
    test_name = config.get('test_name')
    tilelang_jobs = config.get('tilelang_jobs')

    if "SKIP_TEST" in os.environ:
        BUILD_TEST_FILE = False
    if "SKIP_EXPORT" in os.environ:
        BUILD_EXPORT = False

    version = get_version()

    # check and install git pre-commit
    check_and_install_pre_commit()

    test_cmd = SingleTest if test_name else TestUT
    options = {
        'build_ext': {
            'device': device,
            'arch': arch,
            'generate_so': generate_so,
            'tilelang_jobs': tilelang_jobs,
        },
        'bdist_wheel': {
            'device': device,
            'arch': arch,
            'tilelang_jobs': tilelang_jobs,
        }
    }
    if test_name:
        options['test'] = {
            'device': device,
            'arch': arch,
            'generate_so': generate_so,
            'test_name': test_name,
            'tilelang_jobs': tilelang_jobs,
        }

    setup(
        name="xllm",
        version=version,
        license="Apache 2.0",
        author="xLLM Team",
        author_email="infer@xllm.ai",
        description="A high-performance inference system for large language models.",
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        url="https://github.com/jd-opensource/xllm",
        project_urls={
            "Homepage": "https://xllm-ai.com/",
            "Documentation": "https://docs.xllm-ai.com/",
        },
        classifiers=[
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Programming Language :: C++",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Operating System :: POSIX",
            "License :: OSI Approved :: Apache Software License",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        ext_modules=[CMakeExtension("xllm", "xllm/")],
        cmdclass={"build_ext": ExtBuild,
                  "test": test_cmd,
                  "install": InstallWheel,
                  'bdist_wheel': BuildDistWheel},
        options=options,
        packages=find_namespace_packages(include=["scripts", "scripts.*"]),
        zip_safe=False,
        py_modules=["xllm/launch_server", "xllm/__init__",
                    "xllm/pybind/llm", "xllm/pybind/vlm",
                    "xllm/pybind/embedding", "xllm/pybind/utils",
                    "xllm/pybind/args", "xllm/pybind/params",
                    "xllm/pybind/errors", "xllm/pybind/mm_utils"],
        python_requires=">=3.10",
    )
