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
==============================================================================*/

#pragma once

#include <cstdint>
#include <string>

#include "api_service/api_service_impl.h"
#include "api_service/non_stream_call.h"
#include "core/common/types.h"
#include "core/framework/request/request_params.h"
#include "sample.pb.h"

namespace xllm {

using SampleCall = NonStreamCall<proto::SampleRequest, proto::SampleResponse>;

namespace sample_service_internal {

inline constexpr uint32_t kDefaultSampleLogprobs = 5;
inline constexpr uint32_t kMinSampleLogprobs = 1;
inline constexpr uint32_t kMaxSampleLogprobs = 5;

Status validate_request(const proto::SampleRequest& request);

Status validate_runtime_config(bool enable_schedule_overlap);

bool build_request_params(const proto::SampleRequest& request,
                          const Tokenizer& tokenizer,
                          RequestParams* request_params);

bool build_empty_response(const proto::SampleRequest& request,
                          const Tokenizer& tokenizer,
                          const std::string& request_id,
                          proto::SampleResponse* response);

bool build_response(const std::string& request_id,
                    const std::string& model,
                    uint32_t created_time,
                    const RequestOutput& req_output,
                    proto::SampleResponse* response);

}  // namespace sample_service_internal

class SampleServiceImpl final : public APIServiceImpl<SampleCall> {
 public:
  SampleServiceImpl(LLMMaster* master, const std::vector<std::string>& models);

  bool process_request(const proto::SampleRequest& request,
                       proto::SampleResponse* response,
                       Status* status) const;

  void process_async_impl(std::shared_ptr<SampleCall> call) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(SampleServiceImpl);

  LLMMaster* master_ = nullptr;
};

}  // namespace xllm
