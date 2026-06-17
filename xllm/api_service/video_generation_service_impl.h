/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#pragma once
#include <absl/container/flat_hash_set.h>

#include "api_service/api_service_impl.h"
#include "api_service/non_stream_call.h"
#include "video_generation.pb.h"

namespace xllm {

using VideoGenerationCall = NonStreamCall<proto::VideoGenerationRequest,
                                          proto::VideoGenerationResponse>;
class DiTMaster;
// Handles /v1/video/generation requests
class VideoGenerationServiceImpl final
    : public APIServiceImpl<VideoGenerationCall> {
 public:
  VideoGenerationServiceImpl(DiTMaster* master,
                             const std::vector<std::string>& models);

  void process_async_impl(std::shared_ptr<VideoGenerationCall> call);

 private:
  DISALLOW_COPY_AND_ASSIGN(VideoGenerationServiceImpl);
  DiTMaster* master_ = nullptr;
};

}  // namespace xllm
