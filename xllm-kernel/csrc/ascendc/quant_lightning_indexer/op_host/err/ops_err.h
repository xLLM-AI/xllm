/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
the terms and conditions of  * CANN Open Software License Agreement Version 2.0
(the "License").  * Please refer to the License for details. You may not use
this file except in compliance with the License.  * THIS SOFTWARE IS PROVIDED ON
AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,  *
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE.  * See LICENSE in the root of the software repository for
the full text of the License.  */

/*!
 * \file ops_err.h
 * \brief
 */

#ifndef Xllm_COMMON_OPS_ERR_H
#define Xllm_COMMON_OPS_ERR_H

#include "log/log.h"

#ifndef XLLM_OP_MODULE_ID
#if defined(OP)
#define XLLM_OP_MODULE_ID OP
#else
// CANN 9.1 中 OP_MODULE_ID 为 log.h 内 constexpr，9.0
// 中不存在；统一回退固定模块 ID 63（仅影响日志过滤，不影响功能）
#define XLLM_OP_MODULE_ID 63
#endif
#endif

#define OPS_INNER_ERR_STUB(ERR_CODE_STR, OPS_DESC, FMT, ...)               \
  do {                                                                     \
    OpLogSub(XLLM_OP_MODULE_ID, DLOG_ERROR, OPS_DESC, FMT, ##__VA_ARGS__); \
    REPORT_INNER_ERR_MSG(ERR_CODE_STR, FMT, ##__VA_ARGS__);                \
  } while (0)

/* 基础报错 */
#define OPS_REPORT_VECTOR_INNER_ERR(OPS_DESC, ...) \
  OPS_INNER_ERR_STUB("E89999", OPS_DESC, __VA_ARGS__)
#define OPS_REPORT_CUBE_INNER_ERR(OPS_DESC, ...) \
  OPS_INNER_ERR_STUB("E69999", OPS_DESC, __VA_ARGS__)

#endif  // Xllm_COMMON_OPS_ERR_H