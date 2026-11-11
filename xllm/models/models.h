/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

#pragma once

#if defined(USE_NPU)
#include "dit/pipelines/pipeline_flux.h"                 // IWYU pragma: keep
#include "dit/pipelines/pipeline_flux2.h"                // IWYU pragma: keep
#include "dit/pipelines/pipeline_flux_control.h"         // IWYU pragma: keep
#include "dit/pipelines/pipeline_flux_fill.h"            // IWYU pragma: keep
#include "dit/pipelines/pipeline_joyimage_edit_plus.h"   // IWYU pragma: keep
#include "dit/pipelines/pipeline_minimax_h3.h"           // IWYU pragma: keep
#include "dit/pipelines/pipeline_qwenimage_edit_plus.h"  // IWYU pragma: keep
#include "dit/pipelines/pipeline_wan_i2v.h"              // IWYU pragma: keep
#include "llm/deepseek_v4.h"                             // IWYU pragma: keep
#include "llm/deepseek_v4_dspark.h"                      // IWYU pragma: keep
#include "llm/deepseek_v4_mtp.h"                         // IWYU pragma: keep
#include "llm/glm5_dspark.h"                             // IWYU pragma: keep
#include "llm/glm5_next.h"                               // IWYU pragma: keep
#include "llm/npu/minimax_m2.h"                          // IWYU pragma: keep
#include "llm/npu/qwen3_dflash2.h"                       // IWYU pragma: keep
#include "llm/qwen3.h"                                   // IWYU pragma: keep
#include "llm/qwen3_5.h"                                 // IWYU pragma: keep
#include "llm/qwen3_5_mtp.h"                             // IWYU pragma: keep
#include "llm/qwen3_moe.h"                               // IWYU pragma: keep
#include "llm/qwen3_next.h"                              // IWYU pragma: keep
#include "models/npu_model_metadata.h"                   // IWYU pragma: keep
#include "rec/npu/onerec.h"                              // IWYU pragma: keep
#include "vlm/deepseek_ocr_vlm.h"                        // IWYU pragma: keep
#include "vlm/glm5_next_vlm.h"                           // IWYU pragma: keep
#include "vlm/joyai_asr.h"                               // IWYU pragma: keep
#include "vlm/qwen3_5.h"                                 // IWYU pragma: keep
#include "vlm/qwen3_vl.h"                                // IWYU pragma: keep

#elif defined(USE_MLU)
#include "dit/pipelines/pipeline_flux.h"          // IWYU pragma: keep
#include "dit/pipelines/pipeline_flux_control.h"  // IWYU pragma: keep
#include "dit/pipelines/pipeline_flux_fill.h"     // IWYU pragma: keep
#include "llm/deepseek_v2.h"                      // IWYU pragma: keep
#include "llm/deepseek_v3.h"                      // IWYU pragma: keep
#include "llm/deepseek_v32.h"                     // IWYU pragma: keep
#include "llm/glm5.h"                             // IWYU pragma: keep
#include "llm/glm52.h"                            // IWYU pragma: keep
#include "llm/glm5_next.h"                        // IWYU pragma: keep
#include "llm/joyai_llm_flash.h"                  // IWYU pragma: keep
#include "llm/mlu/deepseek_mtp.h"                 // IWYU pragma: keep
#include "llm/mlu/deepseek_v4.h"                  // IWYU pragma: keep
#include "llm/mlu/deepseek_v4_mtp.h"              // IWYU pragma: keep
#include "llm/mlu/glm5_mtp.h"                     // IWYU pragma: keep
#include "llm/mlu/glm5_next.h"                    // IWYU pragma: keep
#include "llm/mlu/glm5_next_mtp.h"                // IWYU pragma: keep
#include "llm/mlu/joyai_llm_flash_mtp.h"          // IWYU pragma: keep
#include "llm/mlu/qwen3_5_mtp.h"                  // IWYU pragma: keep
#include "llm/mtp_model_base.h"                   // IWYU pragma: keep
#include "llm/oxygen.h"                           // IWYU pragma: keep
#include "llm/qwen2.h"                            // IWYU pragma: keep
#include "llm/qwen3.h"                            // IWYU pragma: keep
#include "llm/qwen3_moe.h"                        // IWYU pragma: keep
#include "vlm/oxygen_vlm.h"                       // IWYU pragma: keep
#include "vlm/qwen2_5_vl.h"                       // IWYU pragma: keep
#include "vlm/qwen2_vl.h"                         // IWYU pragma: keep
#include "vlm/qwen3_5.h"                          // IWYU pragma: keep
#include "vlm/qwen3_vl.h"                         // IWYU pragma: keep
#include "vlm/qwen3_vl_moe.h"                     // IWYU pragma: keep
#elif defined(USE_ILU)
#include "llm/qwen2.h"      // IWYU pragma: keep
#include "llm/qwen3.h"      // IWYU pragma: keep
#include "llm/qwen3_moe.h"  // IWYU pragma: keep
#elif defined(USE_MUSA)
#include "llm/qwen3_5.h"      // IWYU pragma: keep
#include "llm/qwen3_5_mtp.h"  // IWYU pragma: keep
#include "llm/qwen3_next.h"   // IWYU pragma: keep
#elif defined(USE_CUDA)
#include "dit/pipelines/pipeline_cola_dlm.h"            // IWYU pragma: keep
#include "dit/pipelines/pipeline_longcat_audiodit.h"    // IWYU pragma: keep
#include "dit/pipelines/pipeline_longcat_image.h"       // IWYU pragma: keep
#include "dit/pipelines/pipeline_longcat_image_edit.h"  // IWYU pragma: keep
#include "llm/mimo.h"                                   // IWYU pragma: keep
#include "llm/mimo_mtp.h"                               // IWYU pragma: keep
#include "llm/phi3.h"                                   // IWYU pragma: keep
#include "llm/qwen2.h"                                  // IWYU pragma: keep
#include "llm/qwen3.h"                                  // IWYU pragma: keep
#include "llm/qwen3_5.h"                                // IWYU pragma: keep
#include "llm/qwen3_moe.h"                              // IWYU pragma: keep
#include "llm/rwkv7.h"                                  // IWYU pragma: keep
#include "vlm/qwen2_5_vl.h"                             // IWYU pragma: keep
#include "vlm/qwen2_vl.h"                               // IWYU pragma: keep
#include "vlm/qwen3_vl.h"                               // IWYU pragma: keep
#include "vlm/qwen3_vl_moe.h"                           // IWYU pragma: keep
#elif defined(USE_DCU)
#include "dit/pipelines/pipeline_flux.h"                 // IWYU pragma: keep
#include "dit/pipelines/pipeline_longcat_image.h"        // IWYU pragma: keep
#include "dit/pipelines/pipeline_qwenimage_edit_plus.h"  // IWYU pragma: keep
#include "dit/pipelines/pipeline_wan_i2v.h"              // IWYU pragma: keep
#include "llm/deepseek_v2.h"                             // IWYU pragma: keep
#include "llm/deepseek_v3.h"                             // IWYU pragma: keep
#include "llm/mimo.h"                                    // IWYU pragma: keep
#include "llm/mimo_mtp.h"                                // IWYU pragma: keep
#include "llm/minimax_m2.h"                              // IWYU pragma: keep
#include "llm/qwen2.h"                                   // IWYU pragma: keep
#include "llm/qwen3.h"                                   // IWYU pragma: keep
#include "llm/qwen3_moe.h"                               // IWYU pragma: keep
#include "vlm/qwen2_5_vl.h"                              // IWYU pragma: keep
#include "vlm/qwen2_vl.h"                                // IWYU pragma: keep
#include "vlm/qwen3_5.h"                                 // IWYU pragma: keep
#include "vlm/qwen3_vl.h"                                // IWYU pragma: keep
#include "vlm/qwen3_vl_moe.h"                            // IWYU pragma: keep
#else
#error "Unsupported device type, only support NPU, CUDA, MLU, ILU and MUSA now."
#endif
