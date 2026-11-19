#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
"""

import re
import os, sys
import ctypes
import json
import shutil
from asc_op_compile_base.common.platform import get_soc_spec
from asc_op_compile_base.common.utils import para_check
from asc_op_compile_base.asc_op_compiler import compile_op, replay_op, check_op_cap, generalize_op_params, get_code_channel, OpInfo
from asc_op_compile_base.asc_op_compiler.compile_op import CommonUtility, AscendCLogLevel
from asc_op_compile_base.common.buildcfg import get_default_build_config
from asc_op_compile_base.common import register as tbe_register
from asc_op_compile_base.common.buildcfg import get_current_build_config
PYF_PATH = os.path.dirname(os.path.realpath(__file__))

__version__ = '2.0.0'

DTYPE_MAP = {"float32": ["DT_FLOAT", "float"],
    "float16": ["DT_FLOAT16", "half"],
    "int8": ["DT_INT8", "int8_t"],
    "int16": ["DT_INT16", "int16_t"],
    "int32": ["DT_INT32", "int32_t"],
    "int64": ["DT_INT64", "int64_t"],
    "uint1": ["DT_UINT1", "uint1b_t"],
    "uint8": ["DT_UINT8", "uint8_t"],
    "uint16": ["DT_UINT16", "uint16_t"],
    "uint32": ["DT_UINT32", "uint32_t"],
    "uint64": ["DT_UINT64", "uint64_t"],
    "bool": ["DT_BOOL", "bool"],
    "double": ["DT_DOUBLE", "double"],
    "dual": ["DT_DUAL", "unknown"],
    "dual_sub_int8": ["DT_DUAL_SUB_INT8", "unknown"],
    "dual_sub_uint8": ["DT_DUAL_SUB_UINT8", "unknown"],
    "string": ["DT_STRING", "unknown"],
    "complex32": ["DT_COMPLEX32", "complex32"],
    "complex64": ["DT_COMPLEX64", "complex64"],
    "complex128": ["DT_COMPLEX128", "unknown"],
    "qint8": ["DT_QINT8", "unknown"],
    "qint16": ["DT_QINT16", "unknown"],
    "qint32": ["DT_QINT32", "unknown"],
    "quint8": ["DT_QUINT8", "unknown"],
    "quint16": ["DT_QUINT16", "unknown"],
    "resource": ["DT_RESOURCE", "unknown"],
    "string_ref": ["DT_STRING_REF", "unknown"],
    "int4": ["DT_INT4", "int4b_t"],
    "bfloat16": ["DT_BF16", "bfloat16_t"],
    "float8_e5m2": ["DT_FLOAT8_E5M2", "fp8_e5m2_t"],
    "float8_e4m3fn": ["DT_FLOAT8_E4M3FN", "fp8_e4m3fn_t"],
    "hifloat8":["DT_HIFLOAT8", "hifloat8_t"],
    "float8_e8m0":["DT_FLOAT8_E8M0", "fp8_e8m0_t"],
    "float4_e2m1":["DT_FLOAT4_E2M1", "fp4x2_e2m1_t"],
    "float4_e1m2":["DT_FLOAT4_E1M2", "fp4x2_e1m2_t"],
    "int2":["DT_INT2","int2b_t"]}

def add_dtype_fmt_option_single(x, x_n, is_ref: bool = False):
    options = []
    x_fmt = x.get("format")
    x_dtype = x.get("dtype")
    x_n_in_kernel = x_n + '_REF' if is_ref else x_n
    options.append("-DDTYPE_{n}={t}".format(n=x_n_in_kernel, t=DTYPE_MAP.get(x_dtype)[1]))
    options.append("-DORIG_DTYPE_{n}={ot}".format(n=x_n_in_kernel, ot=DTYPE_MAP.get(x_dtype)[0]))
    options.append("-DFORMAT_{n}=FORMAT_{f}".format(n=x_n_in_kernel, f=x_fmt))
    return options

def get_dtype_fmt_options(__inputs__, __outputs__):
    options = []
    input_names = ['x']
    output_names = ['qRow', 'qCol', 'rowAbsmax', 'colAbsmax']
    unique_param_name_set = set()
    for idx, x in enumerate(__inputs__):
        if x is None:
            continue
        x_n = input_names[idx].upper()
        unique_param_name_set.add(x_n)
        options += add_dtype_fmt_option_single(x, x_n)

    for idx, x in enumerate(__outputs__):
        if x is None:
            continue
        x_n = output_names[idx].upper()
        if x_n in unique_param_name_set:
            options += add_dtype_fmt_option_single(x, x_n, True)
        else:
            options += add_dtype_fmt_option_single(x, x_n)
    return options

def load_dso(so_path):
    try:
        ctypes.CDLL(so_path)
    except OSError as error :
        CommonUtility.print_compile_log("", error, AscendCLogLevel.LOG_ERROR)
        raise RuntimeError("cannot open %s" %(so_path))
    else:
        msg = "load so succ " + so_path
        CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_INFO)

def get_shortsoc_compile_option(compile_option_list: list, shortsoc:str):
    compile_options = []
    if shortsoc in compile_option_list:
        compile_options.extend(compile_option_list[shortsoc])
    if '__ALLSOC__' in compile_option_list:
        compile_options.extend(compile_option_list['__ALLSOC__'])
    return compile_options

def get_src_file_from_dict(src_file_dict, shortsoc):
    src_file = ""
    if shortsoc in src_file_dict:
        src_file = src_file_dict[shortsoc]
    elif "__ALLSOC__" in src_file_dict:
        src_file = src_file_dict["__ALLSOC__"]
    return src_file

def get_kernel_source(src_file, dir_snake, dir_ex):
    src_ex = os.path.join(PYF_PATH, "..", "ascendc", dir_ex, src_file)
    if os.path.exists(src_ex):
        return src_ex
    src = os.environ.get('BUILD_KERNEL_SRC')
    if src and os.path.exists(src):
        return src
    src = os.path.join(PYF_PATH, "..", "ascendc", dir_snake, src_file)
    if os.path.exists(src):
        return src
    src = os.path.join(PYF_PATH, src_file)
    if os.path.exists(src):
        return src
    src = os.path.join(PYF_PATH, "..", "ascendc", dir_snake, dir_snake + ".cpp")
    if os.path.exists(src):
        return src
    src = os.path.join(PYF_PATH, "..", "ascendc", dir_ex, dir_ex + ".cpp")
    if os.path.exists(src):
        return src
    src = os.path.join(PYF_PATH, "..", "ascendc", os.path.splitext(src_file)[0], src_file)
    if os.path.exists(src):
        return src
    return src_ex

def _build_args(x_in__, qRow_out_, qCol_out_, rowAbsmax_out_, colAbsmax_out_):
    __inputs__ = []
    for arg in [x_in__]:
        if arg != None:
            if isinstance(arg, (list, tuple)):
                if len(arg) == 0:
                    continue
                __inputs__.append(arg[0])
            else:
                __inputs__.append(arg)
        else:
            __inputs__.append(arg)
    __outputs__ = []
    for arg in [qRow_out_, qCol_out_, rowAbsmax_out_, colAbsmax_out_]:
        if arg != None:
            if isinstance(arg, (list, tuple)):
                if len(arg) == 0:
                    continue
                __outputs__.append(arg[0])
            else:
                __outputs__.append(arg)
        else:
            __outputs__.append(arg)
    __attrs__ = []
    return __inputs__, __outputs__, __attrs__

@tbe_register.register_operator("RowColInt8Quant", trans_bool_to_s8=False)
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def row_col_int8_quant(x_in__, qRow_out_, qCol_out_, rowAbsmax_out_, colAbsmax_out_, kernel_name="row_col_int8_quant", impl_mode = ""):
    # do ascendc build step
    if get_current_build_config("enable_op_prebuild"):
        return
    __inputs__, __outputs__, __attrs__ = _build_args(x_in__, qRow_out_, qCol_out_, rowAbsmax_out_, colAbsmax_out_)
    options = get_dtype_fmt_options(__inputs__, __outputs__)
    options += ["-x", "cce"]

    ascend_home_path = os.environ.get('ASCEND_HOME_PATH')
    import platform
    archlinux = platform.machine()
    if ascend_home_path is None or ascend_home_path == '':
        asc_opc_path = shutil.which("asc_opc")
        if asc_opc_path is not None:
            asc_opc_path_link = os.path.dirname(asc_opc_path)
            asc_opc_real_path = os.path.realpath(asc_opc_path_link)
            ascend_home_path = os.path.realpath(
                    os.path.join(asc_opc_real_path, "..", ".."))
        else:
            ascend_home_path = "/usr/local/Ascend/cann"

    if 'x86' in archlinux:
        asc_path = os.path.realpath(os.path.join(ascend_home_path, "x86_64-linux", "asc"))
    else:
        asc_path = os.path.realpath(os.path.join(ascend_home_path, "aarch64-linux", "asc"))
    if asc_path is None:
        asc_path = os.path.realpath(os.path.join(ascend_home_path, "compiler", "asc"))

    options.append("-I" + os.path.join(asc_path, "..", "ascendc", "act"))
    options.append("-I" + os.path.join(PYF_PATH, "..", "ascendc", "common"))
    if "impl_mode" in locals():
        if impl_mode == "high_performance":
            options.append("-DHIGH_PERFORMANCE=1")
        elif impl_mode == "high_precision":
            options.append("-DHIGH_PRECISION=1")
        elif "high_precision" in impl_mode and "high_performance" in impl_mode:
            options.append("-DHIGH_PRECISION=1 -DHIGH_PERFORMANCE=1")
    if get_current_build_config("enable_deterministic_mode") == 1:
        options.append("-DDETERMINISTIC_MODE=1")
    else:
        options.append("-DDETERMINISTIC_MODE=0")
    custom_compile_options = {},
    custom_all_compile_options = {},
    soc_version = get_soc_spec("SOC_VERSION")
    soc_short = get_soc_spec("SHORT_SOC_VERSION").lower()
    custom_compile_options_soc = get_shortsoc_compile_option(custom_compile_options[0], soc_short)
    custom_all_compile_options_soc = get_shortsoc_compile_option(custom_all_compile_options[0], soc_short)
    options += custom_all_compile_options_soc
    options += custom_compile_options_soc

    def replace_env_vars(input_str):
        pattern = r'\$\{([^}]+)\}|\$([A-Za-z0-9_]+)'
        def replace_match(match):
            var_name = match.group(1) or match.group(2)
            return os.environ.get(var_name, match.group(0))
        return re.sub(pattern, replace_match, input_str)
    options = [replace_env_vars(opt) for opt in options]

    origin_func_name = "row_col_int8_quant"
    ascendc_src_dir_ex = "row_col_int8_quant"
    ascendc_src_dir = "row_col_int8_quant"

    src_file_dict = {}
    src_file = get_src_file_from_dict(src_file_dict, soc_short)
    if src_file != "":
        ascendc_src_file = src_file
    else:
        ascendc_src_file = "row_col_int8_quant.cpp"
    src = get_kernel_source(ascendc_src_file, ascendc_src_dir, ascendc_src_dir_ex)

    msg = "start compile Ascend C Operator RowColInt8Quant, kernel name is " + kernel_name
    CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_INFO)
    op_type = "RowColInt8Quant"
    code_channel = get_code_channel(src, kernel_name, op_type, options)
    op_info = OpInfo(kernel_name = kernel_name, op_type = op_type, inputs = __inputs__, outputs = __outputs__,\
        attrs = __attrs__ , impl_mode = impl_mode, origin_inputs=[x_in__], origin_outputs = [qRow_out_, qCol_out_, rowAbsmax_out_, colAbsmax_out_],\
                param_type_dynamic = False, mc2_ctx = [], param_type_list = ['required', 'required', 'required', 'required', 'required'], init_value_list = [None, None, None, None],\
                output_shape_depend_on_compute = [])
    compile_op(src, origin_func_name, op_info, options, code_channel, '{}', {'valueDepend': {}})

def op_select_format(x_in__, qRow_out_, qCol_out_, rowAbsmax_out_, colAbsmax_out_, impl_mode = ""):
    __inputs__, __outputs__, __attrs__ = _build_args(x_in__, qRow_out_, qCol_out_, rowAbsmax_out_, colAbsmax_out_)
    result = check_op_cap("op_select_format", "RowColInt8Quant", __inputs__, __outputs__, __attrs__)
    return result.decode("utf-8")

def get_op_specific_info(x_in__, qRow_out_, qCol_out_, rowAbsmax_out_, colAbsmax_out_, impl_mode = ""):
    __inputs__, __outputs__, __attrs__ = _build_args(x_in__, qRow_out_, qCol_out_, rowAbsmax_out_, colAbsmax_out_)
    result = check_op_cap("get_op_specific_info", "RowColInt8Quant", __inputs__, __outputs__, __attrs__)
    return result.decode("utf-8")
