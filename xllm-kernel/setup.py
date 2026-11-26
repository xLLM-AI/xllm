# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Build Python kernels and the optional NPU device binding."""

import os
import shutil
import subprocess
import tempfile
from importlib.util import find_spec
from pathlib import Path

from setuptools import setup

extensions = []
commands = {}
if (npu_spec := find_spec("torch_npu")) is not None:
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    npu_root = Path(npu_spec.origin).parent
    ascend_root = Path(os.environ.get("NPU_HOME_PATH") or os.environ["ASCEND_HOME_PATH"])
    atb_root = Path(os.environ["ATB_HOME_PATH"])
    ascendc_vendor = Path("xllm_kernel/ascendc_opp/xllm_kernel")

    class NpuBuildExtension(BuildExtension):
        def run(self) -> None:
            super().run()

            compute_unit = os.environ.get("ASCEND_COMPUTE_UNIT")
            if not compute_unit:
                from torch_npu import _C

                soc_version = _C._npu_get_soc_version()
                compute_unit = (
                    "ascend910b" if soc_version < 250 else "ascend950" if soc_version == 260 else "ascend910_93"
                )

            source = Path(__file__).resolve().parent / "csrc" / "ascendc"
            build = Path(self.build_temp).resolve() / "ascendc"
            subprocess.run(
                [
                    "cmake",
                    "-S",
                    str(source),
                    "-B",
                    str(build),
                    "-DCMAKE_BUILD_TYPE=Release",
                    f"-DASCEND_CANN_PACKAGE_PATH={ascend_root}",
                    f"-DASCEND_COMPUTE_UNIT={compute_unit}",
                ],
                check=True,
            )
            jobs = os.environ.get("MAX_JOBS") or os.environ.get("OPS_CPU_NUMBER") or "8"
            subprocess.run(
                ["cmake", "--build", str(build), "--target", "binary", "all", "--parallel", jobs], check=True
            )

            with tempfile.TemporaryDirectory(dir=build) as output:
                subprocess.run(["cmake", "--install", str(build), "--prefix", output], check=True)
                vendor = Path(output) / "packages/vendors/xllm_kernel"
                headers = vendor / "op_api/include"
                (headers / "aclnn_xllm_kernel_gamma_add_rms_norm.h").unlink(missing_ok=True)
                gamma = source / "gamma_add_rms_norm"
                for header in ("aclnn_gamma_add_rms_norm.h", "aclnn_util.h"):
                    shutil.copy2(gamma / "op_host/op_api" / header, headers)
                for component in ("moe_init_routing_v3", "moe_grouped_matmul_swiglu_quant", "scatter_nd_update_v2"):
                    packaged_header = headers / f"aclnn_xllm_kernel_{component}.h"
                    if packaged_header.exists():
                        header = source / component / "op_host/op_api" / f"aclnn_{component}.h"
                        shutil.copy2(header, packaged_header)
                for component in ("quant_lightning_indexer_metadata", "sparse_attn_sharedkv_metadata"):
                    metadata_header = source / component / "op_api" / f"aclnn_{component}.h"
                    shutil.copy2(metadata_header, headers)
                chunk_gdn = source / "mega_chunk_gdn/op_host/op_api"
                shutil.copy2(chunk_gdn / "aclnn_mega_chunk_gdn.h", headers / "aclnn_xllm_kernel_mega_chunk_gdn.h")
                recurrent_header = headers / "aclnn_xllm_kernel_recurrent_gated_delta_rule.h"
                if recurrent_header.exists():
                    recurrent_header.unlink()
                    recurrent_gdn = source / "recurrent_gated_delta_rule/op_host/op_api"
                    shutil.copy2(recurrent_gdn / "aclnn_recurrent_gated_delta_rule.h", headers)
                for component in ("dispatch_ffn_combine", "dispatch_gmm_combine_decode"):
                    generated_header = headers / f"aclnn_xllm_kernel_{component}_op.h"
                    if generated_header.exists():
                        header = source / component / "op_host/op_api" / f"aclnn_xllm_kernel_{component}.h"
                        shutil.copy2(header, headers)
                shutil.copy2(gamma / "CANN_LICENSE.txt", vendor)
                for provenance in sorted(source.glob("*/SOURCE.txt")):
                    shutil.copy2(provenance, vendor / f"{provenance.parent.name}_SOURCE.txt")
                paths = [Path(self.build_lib)]
                if self.inplace:
                    paths.append(Path("python"))
                for package in paths:
                    destination = package / ascendc_vendor
                    if destination.exists():
                        shutil.rmtree(destination)
                    shutil.copytree(vendor, destination, symlinks=True)

        def get_outputs(self) -> list[str]:
            vendor = Path(self.build_lib) / ascendc_vendor
            return [*super().get_outputs(), *(str(path) for path in vendor.rglob("*") if path.is_file())]

    extensions.append(
        CppExtension(
            "xllm_kernel._atb",
            sources=[
                "csrc/atb/matmul_ein_sum.cpp",
                "csrc/atb/reshape_paged_cache.cpp",
                "csrc/npu/collectives.cpp",
                "csrc/npu/cp_context.cpp",
                "csrc/npu/decode_graph_metadata.cpp",
                "csrc/npu/gamma_add_rms_norm.cpp",
                "csrc/npu/gdn.cpp",
                "csrc/npu/grouped_matmul.cpp",
                "csrc/npu/hyper_connection.cpp",
                "csrc/npu/inplace_partial_rotary_mul.cpp",
                "csrc/npu/laser_attention_binding.cpp",
                "csrc/npu/lightning_indexer.cpp",
                "csrc/npu/mla_preprocess_v2.cpp",
                "csrc/npu/mega_gdn.cpp",
                "csrc/npu/mega_moe.cpp",
                "csrc/npu/moe.cpp",
                "csrc/npu/moe_collectives_bindings.cpp",
                "csrc/npu/normalization.cpp",
                "csrc/npu/quant_matmul.cpp",
                "csrc/npu/quantization.cpp",
                "csrc/npu/reshape_paged_cache.cpp",
                "csrc/npu/sparse_attention.cpp",
                "csrc/npu/sparse_indexing.cpp",
            ],
            include_dirs=[
                str(Path(__file__).resolve().parent),
                str(npu_root / "include"),
                str(npu_root / "include" / "third_party" / "op-plugin"),
                str(ascend_root / "include"),
                str(atb_root / "include"),
            ],
            library_dirs=[str(npu_root / "lib"), str(ascend_root / "lib64"), str(atb_root / "lib")],
            libraries=["torch_npu", "atb", "glog", "hccl", "hcomm", "dl"],
            extra_compile_args=["-O2", "-fvisibility=hidden", "-fvisibility-inlines-hidden"],
        )
    )
    commands["build_ext"] = NpuBuildExtension.with_options(no_python_abi_suffix=True)

setup(ext_modules=extensions, cmdclass=commands)
