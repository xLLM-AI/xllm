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

#include "api_service/service_impl_factory.h"

#include <glog/logging.h>

#include <functional>
#include <unordered_map>

#include "api_service.h"
#include "api_service/serving_mode.h"
#include "core/common/global_flags.h"
#include "core/distributed_runtime/dit_master.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/distributed_runtime/rec_master.h"
#include "core/distributed_runtime/vlm_master.h"
#include "core/framework/config/model_config.h"

namespace xllm {

namespace {

template <typename T, typename MasterT>
std::unique_ptr<T> create_service_impl(
    MasterT* master,
    const std::vector<std::string>& model_names) {
  return std::make_unique<T>(master, model_names);
}

}  // namespace

void ServiceImplFactory::create(
    APIService* service,
    Master* master,
    const std::vector<std::string>& model_names,
    const std::vector<std::string>& model_versions) {
  using InitFn = std::function<void(
      APIService*, Master*, const std::vector<std::string>&)>;

  static const std::unordered_map<int8_t, InitFn> kRegistry = {
      {static_cast<int8_t>(ServingMode::LLM),
       [](APIService* self,
          Master* master,
          const std::vector<std::string>& models) {
         auto* llm_master = dynamic_cast<LLMMaster*>(master);
         self->anthropic_service_impl_ =
             std::make_unique<AnthropicServiceImpl>(llm_master, models);
         self->completion_service_impl_ =
             create_service_impl<CompletionServiceImpl>(llm_master, models);
         self->sample_service_impl_ =
             create_service_impl<SampleServiceImpl>(llm_master, models);
         self->chat_service_impl_ =
             create_service_impl<ChatServiceImpl>(llm_master, models);
         self->embedding_service_impl_ =
             create_service_impl<EmbeddingServiceImpl>(llm_master, models);
         if (::xllm::ModelConfig::get_instance().enable_qwen3_reranker()) {
           self->rerank_service_impl_ =
               create_service_impl<Qwen3RerankServiceImpl>(llm_master, models);
         } else {
           self->rerank_service_impl_ =
               create_service_impl<RerankServiceImpl>(llm_master, models);
         }
       }},
      {static_cast<int8_t>(ServingMode::VLM),
       [](APIService* self,
          Master* master,
          const std::vector<std::string>& models) {
         auto* vlm_master = dynamic_cast<VLMMaster*>(master);
         self->mm_chat_service_impl_ =
             std::make_unique<MMChatServiceImpl>(vlm_master, models);
         self->mm_embedding_service_impl_ =
             std::make_unique<MMEmbeddingServiceImpl>(vlm_master, models);
       }},
      {static_cast<int8_t>(ServingMode::DIT),
       [](APIService* self,
          Master* master,
          const std::vector<std::string>& models) {
         auto* dit_master = dynamic_cast<DiTMaster*>(master);
         self->image_generation_service_impl_ =
             std::make_unique<ImageGenerationServiceImpl>(dit_master, models);
         self->audio_generation_service_impl_ =
             std::make_unique<AudioGenerationServiceImpl>(dit_master, models);
         self->text_generation_service_impl_ =
             std::make_unique<TextGenerationServiceImpl>(dit_master, models);
         self->video_generation_service_impl_ =
             std::make_unique<VideoGenerationServiceImpl>(dit_master, models);
       }},
      {static_cast<int8_t>(ServingMode::REC),
       [](APIService* self,
          Master* master,
          const std::vector<std::string>& models) {
         auto* rec_master = dynamic_cast<RecMaster*>(master);
         self->rec_completion_service_impl_ =
             std::make_unique<RecCompletionServiceImpl>(rec_master, models);
         self->chat_service_impl_ =
             std::make_unique<ChatServiceImpl>(rec_master, models);
       }},
  };

  ServingMode mode = to_serving_mode(master->engine_type());
  auto it = kRegistry.find(static_cast<int8_t>(mode));
  if (it != kRegistry.end()) {
    it->second(service, master, model_names);
  } else {
    LOG(FATAL) << "Unsupported serving mode for engine type: "
               << master->engine_type().to_string();
  }

  CHECK_EQ(model_names.size(), model_versions.size())
      << "Models and model_versions size mismatch: model_names.size()="
      << model_names.size()
      << ", model_versions.size()=" << model_versions.size();

  service->models_service_impl_ =
      std::make_unique<ModelsServiceImpl>(model_names, model_versions);
}

}  // namespace xllm
