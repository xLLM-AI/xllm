/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLLM_C_API_TEST_UTILS_H_
#define XLLM_C_API_TEST_UTILS_H_

#include <gflags/gflags.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "types.h"
#include "xllm_test.pb.h"

DECLARE_string(model_path);
DECLARE_int32(port);
DECLARE_string(listen_addr);
DECLARE_int32(idle_timeout_s);
DECLARE_string(backend);

namespace xllm_capi_test {

// Applies gflags (after ParseCommandLineFlags) into XLLM_InitOptions.
void ApplyGflagsToXllmInitOptions(XLLM_InitOptions* opt);

// Owns buffers referenced by XLLM_MM_Data after PbToXllmMmData.
struct MmDataOwned {
  std::vector<std::vector<uint8_t>> tensor_byte_buffers;
  std::vector<std::vector<XLLM_Tensor>> tensor_lists;
  std::vector<XLLM_MM_Item> mm_items;
  std::vector<XLLM_MM_DictEntry> mm_dict_entries;
};

// Owns heap data for XLLM_Response filled by PbToXllmResponse (text, message,
// token_ids, logprobs arrays).
struct ResponseOwned {
  std::vector<std::unique_ptr<char[]>> choice_text_bufs;
  std::vector<XLLM_ChatMessage> chat_messages;
  std::vector<std::unique_ptr<char[]>> chat_message_contents;
  std::vector<std::vector<int32_t>> token_ids_vecs;
  std::vector<std::vector<XLLM_LogProb>> logprob_vecs;
  std::vector<XLLM_Choice> choices;
};

void PbToXllmRequestParams(const c_api_test::XLLM_RequestParams& pb,
                           XLLM_RequestParams* out);

void XllmRequestParamsToPb(const XLLM_RequestParams& in,
                           c_api_test::XLLM_RequestParams* pb);

void PbToXllmChatMessage(const c_api_test::XLLM_ChatMessage& pb,
                         XLLM_ChatMessage* out);

void FreeXllmChatMessageContent(XLLM_ChatMessage* out);

void XllmChatMessageToPb(const XLLM_ChatMessage* in,
                         c_api_test::XLLM_ChatMessage* pb);

bool PbToXllmMmData(const c_api_test::XLLM_MM_Data& pb,
                    XLLM_MM_Data* out,
                    MmDataOwned* owned);

void XllmMmDataToPb(const XLLM_MM_Data& in, c_api_test::XLLM_MM_Data* pb);

void XllmResponseToPb(const XLLM_Response* in, c_api_test::XLLM_Response* pb);

void PbToXllmResponse(const c_api_test::XLLM_Response& pb,
                      XLLM_Response* out,
                      ResponseOwned* owned);

// --- Lower-level (types.h <-> pb) ---

void PbToXllmDims(const c_api_test::XLLM_Dims& pb, XLLM_Dims* out);
void XllmDimsToPb(const XLLM_Dims& in, c_api_test::XLLM_Dims* pb);

void PbToXllmTensor(const c_api_test::XLLM_Tensor& pb,
                    XLLM_Tensor* out,
                    MmDataOwned* owned);
void XllmTensorToPb(const XLLM_Tensor& in, c_api_test::XLLM_Tensor* pb);

void PbToXllmTensors(const c_api_test::XLLM_Tensors& pb,
                     XLLM_Tensors* out,
                     MmDataOwned* owned);
void XllmTensorsToPb(const XLLM_Tensors& in, c_api_test::XLLM_Tensors* pb);

void PbToXllmMmValue(const c_api_test::XLLM_MM_Value& pb,
                     XLLM_MM_Value* out,
                     MmDataOwned* owned);
void XllmMmValueToPb(const XLLM_MM_Value& in, c_api_test::XLLM_MM_Value* pb);

void PbToXllmMmDict(const c_api_test::XLLM_MM_Dict& pb,
                    XLLM_MM_Dict* out,
                    MmDataOwned* owned);
void XllmMmDictToPb(const XLLM_MM_Dict& in, c_api_test::XLLM_MM_Dict* pb);

void PbToXllmMmItems(const c_api_test::XLLM_MM_Items& pb,
                     XLLM_MM_Items* out,
                     MmDataOwned* owned);
void XllmMmItemsToPb(const XLLM_MM_Items& in, c_api_test::XLLM_MM_Items* pb);

void PbToXllmMmState(const c_api_test::XLLM_MM_State& pb, XLLM_MM_State* out);
void XllmMmStateToPb(const XLLM_MM_State& in, c_api_test::XLLM_MM_State* pb);

void PbToXllmMmItem(const c_api_test::XLLM_MM_Item& pb,
                    XLLM_MM_Item* out,
                    MmDataOwned* owned);
void XllmMmItemToPb(const XLLM_MM_Item& in, c_api_test::XLLM_MM_Item* pb);

void PbToXllmUsage(const c_api_test::XLLM_Usage& pb, XLLM_Usage* out);
void XllmUsageToPb(const XLLM_Usage& in, c_api_test::XLLM_Usage* pb);

void PbToXllmLogProbs(const c_api_test::XLLM_LogProbs& pb,
                      XLLM_LogProbs* out,
                      std::vector<XLLM_LogProb>* storage);

void XllmLogProbsToPb(const XLLM_LogProbs& in, c_api_test::XLLM_LogProbs* pb);

void PbToXllmChoice(const c_api_test::XLLM_Choice& pb,
                    XLLM_Choice* out,
                    ResponseOwned* owned);

void XllmChoiceToPb(const XLLM_Choice& in, c_api_test::XLLM_Choice* pb);

void PbToXllmChoices(const c_api_test::XLLM_Choices& pb,
                     XLLM_Choices* out,
                     ResponseOwned* owned);

void XllmChoicesToPb(const XLLM_Choices& in, c_api_test::XLLM_Choices* pb);

}  // namespace xllm_capi_test

#endif  // XLLM_C_API_TEST_UTILS_H_
