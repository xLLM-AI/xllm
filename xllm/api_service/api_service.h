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

#pragma once

#include <functional>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "anthropic_service_impl.h"
#include "audio_generation_service_impl.h"
#include "chat_service_impl.h"
#include "completion_service_impl.h"
#include "embedding_service_impl.h"
#include "image_generation_service_impl.h"
#include "models_service_impl.h"
#include "qwen3_rerank_service_impl.h"
#include "rec_completion_service_impl.h"
#include "rerank_service_impl.h"
#include "sample_service_impl.h"
#include "text_generation_service_impl.h"
#include "video_generation_service_impl.h"
#include "xllm_service.pb.h"

namespace xllm {

class ClosureGuard;
class ServiceImplFactory;

class APIService : public proto::XllmAPIService {
  friend class ServiceImplFactory;

 public:
  APIService(Master* master,
             const std::vector<std::string>& model_names,
             const std::vector<std::string>& model_versions);
  ~APIService() = default;

  void Completions(::google::protobuf::RpcController* controller,
                   const proto::CompletionRequest* request,
                   proto::CompletionResponse* response,
                   ::google::protobuf::Closure* done) override;

  void CompletionsHttp(::google::protobuf::RpcController* controller,
                       const proto::HttpRequest* request,
                       proto::HttpResponse* response,
                       ::google::protobuf::Closure* done) override;

  void Sample(::google::protobuf::RpcController* controller,
              const proto::SampleRequest* request,
              proto::SampleResponse* response,
              ::google::protobuf::Closure* done) override;

  void SampleHttp(::google::protobuf::RpcController* controller,
                  const proto::HttpRequest* request,
                  proto::HttpResponse* response,
                  ::google::protobuf::Closure* done) override;

  void ChatCompletions(::google::protobuf::RpcController* controller,
                       const proto::ChatRequest* request,
                       proto::ChatResponse* response,
                       ::google::protobuf::Closure* done) override;

  void ChatCompletionsHttp(::google::protobuf::RpcController* controller,
                           const proto::HttpRequest* request,
                           proto::HttpResponse* response,
                           ::google::protobuf::Closure* done) override;

  void Embeddings(::google::protobuf::RpcController* controller,
                  const proto::EmbeddingRequest* request,
                  proto::EmbeddingResponse* response,
                  ::google::protobuf::Closure* done) override;

  void EmbeddingsHttp(::google::protobuf::RpcController* controller,
                      const proto::HttpRequest* request,
                      proto::HttpResponse* response,
                      ::google::protobuf::Closure* done) override;

  void ImageGeneration(::google::protobuf::RpcController* controller,
                       const proto::ImageGenerationRequest* request,
                       proto::ImageGenerationResponse* response,
                       ::google::protobuf::Closure* done) override;

  void ImageGenerationHttp(::google::protobuf::RpcController* controller,
                           const proto::HttpRequest* request,
                           proto::HttpResponse* response,
                           ::google::protobuf::Closure* done) override;

  void AudioGeneration(::google::protobuf::RpcController* controller,
                       const proto::AudioGenerationRequest* request,
                       proto::AudioGenerationResponse* response,
                       ::google::protobuf::Closure* done) override;

  void AudioGenerationHttp(::google::protobuf::RpcController* controller,
                           const proto::HttpRequest* request,
                           proto::HttpResponse* response,
                           ::google::protobuf::Closure* done) override;

  void TextGeneration(::google::protobuf::RpcController* controller,
                      const proto::TextGenerationRequest* request,
                      proto::TextGenerationResponse* response,
                      ::google::protobuf::Closure* done) override;

  void TextGenerationHttp(::google::protobuf::RpcController* controller,
                          const proto::HttpRequest* request,
                          proto::HttpResponse* response,
                          ::google::protobuf::Closure* done) override;

  void VideoGeneration(::google::protobuf::RpcController* controller,
                       const proto::VideoGenerationRequest* request,
                       proto::VideoGenerationResponse* response,
                       ::google::protobuf::Closure* done) override;

  void VideoGenerationHttp(::google::protobuf::RpcController* controller,
                           const proto::HttpRequest* request,
                           proto::HttpResponse* response,
                           ::google::protobuf::Closure* done) override;

  void Rerank(::google::protobuf::RpcController* controller,
              const proto::RerankRequest* request,
              proto::RerankResponse* response,
              ::google::protobuf::Closure* done) override;

  void RerankHttp(::google::protobuf::RpcController* controller,
                  const proto::HttpRequest* request,
                  proto::HttpResponse* response,
                  ::google::protobuf::Closure* done) override;

  void Models(::google::protobuf::RpcController* controller,
              const proto::ModelListRequest* request,
              proto::ModelListResponse* response,
              ::google::protobuf::Closure* done) override;

  void ModelsHttp(::google::protobuf::RpcController* controller,
                  const proto::HttpRequest* request,
                  proto::HttpResponse* response,
                  ::google::protobuf::Closure* done) override;

  void ModelVersionsHttp(::google::protobuf::RpcController* controller,
                         const proto::HttpRequest* request,
                         proto::HttpResponse* response,
                         ::google::protobuf::Closure* done) override;

  void AnthropicMessagesHttp(::google::protobuf::RpcController* controller,
                             const proto::HttpRequest* request,
                             proto::HttpResponse* response,
                             ::google::protobuf::Closure* done) override;

  void ForkMaster(::google::protobuf::RpcController* controller,
                  const proto::MasterInfos* request,
                  proto::Status* response,
                  ::google::protobuf::Closure* done) override;

  void ForkMasterHttp(::google::protobuf::RpcController* controller,
                      const proto::HttpRequest* request,
                      proto::HttpResponse* response,
                      ::google::protobuf::Closure* done) override;

  void Sleep(::google::protobuf::RpcController* controller,
             const proto::MasterInfos* request,
             proto::Status* response,
             ::google::protobuf::Closure* done) override;

  void SleepHttp(::google::protobuf::RpcController* controller,
                 const proto::HttpRequest* request,
                 proto::HttpResponse* response,
                 ::google::protobuf::Closure* done) override;

  void Wakeup(::google::protobuf::RpcController* controller,
              const proto::MasterInfos* request,
              proto::Status* response,
              ::google::protobuf::Closure* done) override;

  void WakeupHttp(::google::protobuf::RpcController* controller,
                  const proto::HttpRequest* request,
                  proto::HttpResponse* response,
                  ::google::protobuf::Closure* done) override;

  void StartProfileHttp(::google::protobuf::RpcController* controller,
                        const proto::HttpRequest* request,
                        proto::HttpResponse* response,
                        ::google::protobuf::Closure* done) override;

  void StopProfileHttp(::google::protobuf::RpcController* controller,
                       const proto::HttpRequest* request,
                       proto::HttpResponse* response,
                       ::google::protobuf::Closure* done) override;

  void LinkP2P(::google::protobuf::RpcController* controller,
               const proto::P2PLinkRequest* request,
               proto::Status* response,
               ::google::protobuf::Closure* done) override;

  void LinkP2PHttp(::google::protobuf::RpcController* controller,
                   const proto::HttpRequest* request,
                   proto::HttpResponse* response,
                   ::google::protobuf::Closure* done) override;

  void UnlinkP2P(::google::protobuf::RpcController* controller,
                 const proto::P2PLinkRequest* request,
                 proto::Status* response,
                 ::google::protobuf::Closure* done) override;

  void UnlinkP2PHttp(::google::protobuf::RpcController* controller,
                     const proto::HttpRequest* request,
                     proto::HttpResponse* response,
                     ::google::protobuf::Closure* done) override;

  // Async RL training support: pause/resume
  void Pause(::google::protobuf::RpcController* controller,
             const proto::PauseRequest* request,
             proto::PauseResponse* response,
             ::google::protobuf::Closure* done) override;

  void PauseHttp(::google::protobuf::RpcController* controller,
                 const proto::HttpRequest* request,
                 proto::HttpResponse* response,
                 ::google::protobuf::Closure* done) override;

  void Resume(::google::protobuf::RpcController* controller,
              const proto::ResumeRequest* request,
              proto::ResumeResponse* response,
              ::google::protobuf::Closure* done) override;

  void ResumeHttp(::google::protobuf::RpcController* controller,
                  const proto::HttpRequest* request,
                  proto::HttpResponse* response,
                  ::google::protobuf::Closure* done) override;

 private:
  using ChatHttpHandler = std::function<void(ClosureGuard&,
                                             brpc::Controller*,
                                             const proto::HttpRequest*,
                                             proto::HttpResponse*)>;

  void register_chat_completions_handler();

  bool ParseForkMasterRequest(const proto::MasterInfos* request,
                              Options& options);
  void set_model_master(const std::string& model_id, Master* master);
  bool has_model_master(const std::string& model_id) const;
  bool add_model_master_if_absent(const std::string& model_id, Master* master);
  Master* get_model_master(const std::string& model_id) const;

  // Core action helpers shared between brpc-typed and Http variants.
  // Each returns true on success. On failure, the human readable reason is
  // written to `error_message` so the caller can either set the HTTP response
  // body or call `brpc::Controller::SetFailed`.
  bool do_fork_master(const proto::MasterInfos& request,
                      std::string* error_message);
  bool do_sleep(const proto::MasterInfos& request, std::string* error_message);
  bool do_wakeup(const proto::MasterInfos& request, std::string* error_message);

  Master* master_;
  ChatHttpHandler chat_completions_handler_;
  mutable std::shared_mutex masters_mutex_;
  std::unordered_map<std::string, Master*> masters_;
  std::unique_ptr<AnthropicServiceImpl> anthropic_service_impl_;
  std::unique_ptr<CompletionServiceImpl> completion_service_impl_;
  std::unique_ptr<SampleServiceImpl> sample_service_impl_;
  std::unique_ptr<ChatServiceImpl> chat_service_impl_;
  std::unique_ptr<MMChatServiceImpl> mm_chat_service_impl_;
  std::unique_ptr<EmbeddingServiceImpl> embedding_service_impl_;
  std::unique_ptr<MMEmbeddingServiceImpl> mm_embedding_service_impl_;
  std::unique_ptr<ModelsServiceImpl> models_service_impl_;
  std::unique_ptr<ImageGenerationServiceImpl> image_generation_service_impl_;
  std::unique_ptr<AudioGenerationServiceImpl> audio_generation_service_impl_;
  std::unique_ptr<TextGenerationServiceImpl> text_generation_service_impl_;
  std::unique_ptr<VideoGenerationServiceImpl> video_generation_service_impl_;
  std::unique_ptr<RerankServiceImpl> rerank_service_impl_;
  std::unique_ptr<RecCompletionServiceImpl> rec_completion_service_impl_;
};

}  // namespace xllm
