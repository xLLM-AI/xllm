/* Copyright 2026 The xLLM Authors.

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

#include <google/protobuf/service.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "api_service/api_service_impl.h"
#include "api_service/stream_call.h"
#include "audio_transcription.pb.h"
#include "models/speech_model.h"

namespace xllm {

class VLMMaster;

using SpeechToTextCall =
    StreamCall<proto::SpeechToTextRequest, proto::SpeechToTextResponse>;

class SpeechToTextServiceImpl final : public APIServiceImpl<SpeechToTextCall> {
 public:
  SpeechToTextServiceImpl(VLMMaster* master,
                          const std::vector<std::string>& models,
                          SpeechToTextTask task);

  void process_async_impl(std::shared_ptr<SpeechToTextCall> call) override;

  static void process_http_request(SpeechToTextServiceImpl* service,
                                   google::protobuf::RpcController* controller,
                                   google::protobuf::Closure* done,
                                   std::string_view service_name);

 private:
  DISALLOW_COPY_AND_ASSIGN(SpeechToTextServiceImpl);

  VLMMaster* master_ = nullptr;
  SpeechToTextTask task_ = SpeechToTextTask::TRANSCRIBE;
  std::shared_ptr<const SpeechModelInterface> adapter_;
};

}  // namespace xllm
