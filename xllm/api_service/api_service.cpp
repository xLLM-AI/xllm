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

#include "api_service.h"

#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <json2pb/json_to_pb.h>
#include <json2pb/pb_to_json.h>

#include <filesystem>

#include "api_service/chat_json_parser.h"
#include "api_service/completion_json_parser.h"
#include "api_service/service_impl_factory.h"
#include "api_service/serving_mode.h"
#include "call.h"
#include "chat.pb.h"
#include "common.pb.h"
#include "completion.pb.h"
#include "core/common/constants.h"
#include "core/common/metrics.h"
#include "core/common/types.h"
#include "core/distributed_runtime/dit_master.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/distributed_runtime/rec_master.h"
#include "core/distributed_runtime/vlm_master.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/config/profile_config.h"
#include "core/util/closure_guard.h"
#include "embedding.pb.h"
#include "image_generation.pb.h"
#include "models.pb.h"
#include "service_impl_factory.h"
#include "text_generation.pb.h"
#include "video_generation.pb.h"
#include "xllm_metrics.h"
namespace xllm {

namespace {
template <typename Call>
google::protobuf::Arena* GetArenaWithCheck(
    const google::protobuf::Message* message) {
  if (xllm::is_stream_call_v<Call>) {
    return nullptr;
  } else {
    return message->GetArena();
  }
}

const char* kSampleNotSupportedError = "/v1/sample is only supported for LLM";

// Shared dispatch for brpc-typed (non-Http) APIs that follow the standard
// service_impl -> process_async pattern.  Performs argument validation,
// instantiates the appropriate Call wrapper from the typed proto request and
// hands it off to the corresponding service implementation.
//
// `service_name` is the human readable endpoint name used in error messages
// when the corresponding service implementation is not available (for example,
// invoking /v1/images/generations against an LLM master).
template <typename CallT, typename Service>
void process_typed_brpc_request(std::unique_ptr<Service>& service_impl,
                                ::google::protobuf::RpcController* controller,
                                const typename CallT::ReqType* request,
                                typename CallT::ResType* response,
                                ::google::protobuf::Closure* done,
                                const char* service_name) {
  xllm::ClosureGuard done_guard(
      done,
      [](void* /*unused*/) { request_in_metric(nullptr); },
      [controller](void* /*unused*/) {
        request_out_metric(static_cast<void*>(controller));
      });
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  if (!service_impl) {
    std::string msg =
        std::string(service_name) + " service is not available on this server";
    ctrl->SetFailed(msg);
    LOG(ERROR) << msg;
    return;
  }

  auto arena = GetArenaWithCheck<CallT>(response);
  // brpc passes the request as `const`, but downstream Call wrappers only read
  // from it.  We cast away constness so the Call can hold a non-const pointer.
  auto req_pb = const_cast<typename CallT::ReqType*>(request);
  std::shared_ptr<Call> call = std::make_shared<CallT>(
      ctrl, done_guard.release(), req_pb, response, arena != nullptr);
  service_impl->process_async(call);
}

}  // namespace

APIService::APIService(Master* master,
                       const std::vector<std::string>& model_names,
                       const std::vector<std::string>& model_versions)
    : master_(master) {
  set_model_master(model_names[0], master);
  if (::xllm::DistributedConfig::get_instance().node_rank() != 0) {
    return;
  }
  ServiceImplFactory::create(this, master, model_names, model_versions);
  register_chat_completions_handler();
}

void APIService::set_model_master(const std::string& model_id, Master* master) {
  std::unique_lock<std::shared_mutex> lock(masters_mutex_);
  masters_.insert_or_assign(model_id, master);
}

bool APIService::has_model_master(const std::string& model_id) const {
  std::shared_lock<std::shared_mutex> lock(masters_mutex_);
  return masters_.find(model_id) != masters_.end();
}

bool APIService::add_model_master_if_absent(const std::string& model_id,
                                            Master* master) {
  std::unique_lock<std::shared_mutex> lock(masters_mutex_);
  return masters_.emplace(model_id, master).second;
}

Master* APIService::get_model_master(const std::string& model_id) const {
  std::shared_lock<std::shared_mutex> lock(masters_mutex_);
  auto it = masters_.find(model_id);
  if (it == masters_.end()) {
    return nullptr;
  }
  return it->second;
}

void APIService::Completions(::google::protobuf::RpcController* controller,
                             const proto::CompletionRequest* request,
                             proto::CompletionResponse* response,
                             ::google::protobuf::Closure* done) {
  xllm::ClosureGuard done_guard(
      done,
      [](void* /*unused*/) { request_in_metric(nullptr); },
      [controller](void* /*unused*/) {
        request_out_metric(static_cast<void*>(controller));
      });
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null.";
    return;
  }
  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  if (completion_service_impl_) {
    completion_service_impl_->process_async_rpc_impl(request);
  } else if (rec_completion_service_impl_) {
    auto arena = GetArenaWithCheck<CompletionCall>(response);
    std::shared_ptr<Call> call = std::make_shared<CompletionCall>(
        ctrl,
        done_guard.release(),
        const_cast<proto::CompletionRequest*>(request),
        response,
        arena != nullptr);
    rec_completion_service_impl_->process_async(call);
  }
}

void APIService::CompletionsHttp(::google::protobuf::RpcController* controller,
                                 const proto::HttpRequest* request,
                                 proto::HttpResponse* response,
                                 ::google::protobuf::Closure* done) {
  xllm::ClosureGuard done_guard(
      done,
      [](void* /*unused*/) { request_in_metric(nullptr); },
      [controller](void* /*unused*/) {
        request_out_metric(static_cast<void*>(controller));
      });
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto arena = GetArenaWithCheck<CompletionCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::CompletionRequest>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::CompletionResponse>(arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  auto [preprocess_status, processed_json] =
      preprocess_completion_prompt(ctrl->request_attachment().to_string());
  if (!preprocess_status.ok()) {
    ctrl->SetFailed(preprocess_status.message());
    LOG(ERROR) << "completion prompt preprocessing failed: "
               << preprocess_status.message();
    return;
  }

  std::string error;
  json2pb::Json2PbOptions options;
  auto st =
      json2pb::JsonToProtoMessage(processed_json, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  std::shared_ptr<Call> call = std::make_shared<CompletionCall>(
      ctrl, done_guard.release(), req_pb, resp_pb, arena != nullptr);
  if (completion_service_impl_) {
    completion_service_impl_->process_async(call);
  } else if (rec_completion_service_impl_) {
    rec_completion_service_impl_->process_async(call);
  }
}

void APIService::Sample(::google::protobuf::RpcController* controller,
                        const proto::SampleRequest* request,
                        proto::SampleResponse* response,
                        ::google::protobuf::Closure* done) {
  xllm::ClosureGuard done_guard(
      done,
      [](void* /*unused*/) { request_in_metric(nullptr); },
      [controller](void* /*unused*/) {
        request_out_metric(static_cast<void*>(controller));
      });
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null.";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  if (!sample_service_impl_) {
    ctrl->SetFailed(kSampleNotSupportedError);
    return;
  }

  Status status;
  if (!sample_service_impl_->process_request(*request, response, &status)) {
    ctrl->SetFailed(status.message());
    LOG(ERROR) << "sample request failed: " << status.message();
  }
}

void APIService::SampleHttp(::google::protobuf::RpcController* controller,
                            const proto::HttpRequest* request,
                            proto::HttpResponse* response,
                            ::google::protobuf::Closure* done) {
  xllm::ClosureGuard done_guard(
      done,
      [](void* /*unused*/) { request_in_metric(nullptr); },
      [controller](void* /*unused*/) {
        request_out_metric(static_cast<void*>(controller));
      });
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  if (!sample_service_impl_) {
    ctrl->SetFailed(kSampleNotSupportedError);
    return;
  }

  auto arena = GetArenaWithCheck<SampleCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::SampleRequest>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::SampleResponse>(arena);

  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  auto st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  std::shared_ptr<Call> call = std::make_shared<SampleCall>(
      ctrl, done_guard.release(), req_pb, resp_pb, arena != nullptr);
  sample_service_impl_->process_async(call);
}

namespace {

size_t get_json_content_length(const brpc::Controller* ctrl) {
  const auto infer_content_len =
      ctrl->http_request().GetHeader(kInferContentLength);
  if (infer_content_len != nullptr) {
    return std::stoul(*infer_content_len);
  }

  const auto content_len = ctrl->http_request().GetHeader(kContentLength);
  if (content_len != nullptr) {
    return std::stoul(*content_len);
  }

  LOG(ERROR) << "Content-Length header is missing.";
  return (size_t)-1L;
}

}  // namespace

namespace {

template <typename ChatCall, typename Service>
void chat_completions_http_impl(std::unique_ptr<Service>& service,
                                xllm::ClosureGuard& guard,
                                brpc::Controller* ctrl,
                                const proto::HttpRequest* request,
                                proto::HttpResponse* response,
                                const ChatJsonParser& chat_json_parser) {
  auto arena = GetArenaWithCheck<ChatCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<typename ChatCall::ReqType>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<typename ChatCall::ResType>(arena);

  auto content_len = get_json_content_length(ctrl);
  if (content_len == (size_t)-1L) {
    ctrl->SetFailed("Content-Length header is missing.");
    return;
  }

  std::string attachment;
  ctrl->request_attachment().copy_to(&attachment, content_len, 0);

  auto [preprocess_status, processed_json] =
      chat_json_parser.preprocess(std::move(attachment));
  if (!preprocess_status.ok()) {
    ctrl->SetFailed(preprocess_status.message());
    LOG(ERROR) << "Complex message preprocessing failed: "
               << preprocess_status.message();
    return;
  }

  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  auto status = google::protobuf::util::JsonStringToMessage(
      processed_json, req_pb, options);
  if (!status.ok()) {
    ctrl->SetFailed(status.ToString());
    LOG(ERROR) << "parse json to proto failed: " << status.ToString();
    return;
  }

  auto call = std::make_shared<ChatCall>(
      ctrl, guard.release(), req_pb, resp_pb, arena != nullptr /*use_arena*/);
  service->process_async(call);
}

}  // namespace

void APIService::register_chat_completions_handler() {
  if (mm_chat_service_impl_) {
    chat_completions_handler_ = [this](ClosureGuard& guard,
                                       brpc::Controller* ctrl,
                                       const proto::HttpRequest* request,
                                       proto::HttpResponse* response) {
      chat_completions_http_impl<MMChatCall, MMChatServiceImpl>(
          mm_chat_service_impl_,
          guard,
          ctrl,
          request,
          response,
          ChatJsonParser::get(ServingMode::VLM));
    };
  } else if (chat_service_impl_) {
    chat_completions_handler_ = [this](ClosureGuard& guard,
                                       brpc::Controller* ctrl,
                                       const proto::HttpRequest* request,
                                       proto::HttpResponse* response) {
      chat_completions_http_impl<ChatCall, ChatServiceImpl>(
          chat_service_impl_,
          guard,
          ctrl,
          request,
          response,
          ChatJsonParser::get(ServingMode::LLM));
    };
  }
}

void APIService::ChatCompletions(::google::protobuf::RpcController* controller,
                                 const proto::ChatRequest* request,
                                 proto::ChatResponse* response,
                                 ::google::protobuf::Closure* done) {
  // TODO with xllm-service
  xllm::ClosureGuard done_guard(
      done,
      [](void* /*unused*/) { request_in_metric(nullptr); },
      [controller](void* /*unused*/) {
        request_out_metric(static_cast<void*>(controller));
      });
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  // Maybe need double check later

  chat_service_impl_->process_async_rpc_impl(request);
}

void APIService::ChatCompletionsHttp(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  xllm::ClosureGuard done_guard(
      done,
      [](void* /*unused*/) { request_in_metric(nullptr); },
      [controller](void* /*unused*/) {
        request_out_metric(static_cast<void*>(controller));
      });
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  if (!chat_completions_handler_) {
    LOG(ERROR) << "No chat completions handler registered";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  chat_completions_handler_(done_guard, ctrl, request, response);
}

void APIService::Embeddings(::google::protobuf::RpcController* controller,
                            const proto::EmbeddingRequest* request,
                            proto::EmbeddingResponse* response,
                            ::google::protobuf::Closure* done) {
  xllm::ClosureGuard done_guard(
      done,
      [](void* /*unused*/) { request_in_metric(nullptr); },
      [controller](void* /*unused*/) {
        request_out_metric(static_cast<void*>(controller));
      });
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  if (!embedding_service_impl_) {
    const char* msg =
        "Embeddings brpc API only supports the text embedding service. "
        "Use /v1/embeddings http endpoint for multimodal embeddings.";
    ctrl->SetFailed(msg);
    LOG(ERROR) << msg;
    return;
  }

  auto arena = GetArenaWithCheck<EmbeddingCall>(response);
  // The brpc-typed request is passed as const, but downstream code only reads
  // from it (after defaulting the encoding format below).
  auto req_pb = const_cast<proto::EmbeddingRequest*>(request);
  if (req_pb->encoding_format().empty()) {
    req_pb->set_encoding_format("float");
  }

  std::shared_ptr<Call> call = std::make_shared<EmbeddingCall>(
      ctrl, done_guard.release(), req_pb, response, arena != nullptr);
  embedding_service_impl_->process_async(call);
}

namespace {
template <typename EmbeddingCall, typename Service>
void handle_embedding_request(std::unique_ptr<Service>& embedding_service_impl_,
                              ::google::protobuf::RpcController* controller,
                              const proto::HttpRequest* request,
                              proto::HttpResponse* response,
                              ::google::protobuf::Closure* done) {
  xllm::ClosureGuard done_guard(
      done,
      [](void* /*unused*/) { request_in_metric(nullptr); },
      [controller](void* /*unused*/) {
        request_out_metric(static_cast<void*>(controller));
      });
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }
  auto arena = GetArenaWithCheck<EmbeddingCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<typename EmbeddingCall::ReqType>(
          arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<typename EmbeddingCall::ResType>(
          arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  auto st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  // default set to "float"
  if (req_pb->encoding_format().empty()) {
    req_pb->set_encoding_format("float");
  }

  std::shared_ptr<Call> call = std::make_shared<EmbeddingCall>(
      ctrl, done_guard.release(), req_pb, resp_pb, arena != nullptr);
  embedding_service_impl_->process_async(call);
}
}  // namespace

void APIService::EmbeddingsHttp(::google::protobuf::RpcController* controller,
                                const proto::HttpRequest* request,
                                proto::HttpResponse* response,
                                ::google::protobuf::Closure* done) {
  if (embedding_service_impl_) {
    handle_embedding_request<EmbeddingCall, EmbeddingServiceImpl>(
        embedding_service_impl_, controller, request, response, done);
  } else if (mm_embedding_service_impl_) {
    handle_embedding_request<MMEmbeddingCall, MMEmbeddingServiceImpl>(
        mm_embedding_service_impl_, controller, request, response, done);
  }
}

void APIService::ImageGeneration(::google::protobuf::RpcController* controller,
                                 const proto::ImageGenerationRequest* request,
                                 proto::ImageGenerationResponse* response,
                                 ::google::protobuf::Closure* done) {
  process_typed_brpc_request<ImageGenerationCall, ImageGenerationServiceImpl>(
      image_generation_service_impl_,
      controller,
      request,
      response,
      done,
      "ImageGeneration");
}

void APIService::ImageGenerationHttp(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  xllm::ClosureGuard done_guard(
      done,
      [](void* /*unused*/) { request_in_metric(nullptr); },
      [controller](void* /*unused*/) {
        request_out_metric(static_cast<void*>(controller));
      });
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto arena = GetArenaWithCheck<ImageGenerationCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::ImageGenerationRequest>(
          arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::ImageGenerationResponse>(
          arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  auto st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }
  std::shared_ptr<ImageGenerationCall> call =
      std::make_shared<ImageGenerationCall>(
          ctrl, done_guard.release(), req_pb, resp_pb, arena != nullptr);
  image_generation_service_impl_->process_async(call);
}

void APIService::AudioGeneration(::google::protobuf::RpcController* controller,
                                 const proto::AudioGenerationRequest* request,
                                 proto::AudioGenerationResponse* response,
                                 ::google::protobuf::Closure* done) {
  process_typed_brpc_request<AudioGenerationCall, AudioGenerationServiceImpl>(
      audio_generation_service_impl_,
      controller,
      request,
      response,
      done,
      "AudioGeneration");
}

void APIService::AudioGenerationHttp(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  xllm::ClosureGuard done_guard(
      done,
      [](void* /*unused*/) { request_in_metric(nullptr); },
      [controller](void* /*unused*/) {
        request_out_metric(static_cast<void*>(controller));
      });
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  auto arena = GetArenaWithCheck<AudioGenerationCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::AudioGenerationRequest>(
          arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::AudioGenerationResponse>(
          arena);

  brpc::Controller* ctrl = static_cast<brpc::Controller*>(controller);
  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  bool st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }
  std::shared_ptr<AudioGenerationCall> call =
      std::make_shared<AudioGenerationCall>(
          ctrl, done_guard.release(), req_pb, resp_pb, arena != nullptr);
  audio_generation_service_impl_->process_async(call);
}

void APIService::TextGeneration(::google::protobuf::RpcController* controller,
                                const proto::TextGenerationRequest* request,
                                proto::TextGenerationResponse* response,
                                ::google::protobuf::Closure* done) {
  process_typed_brpc_request<TextGenerationCall, TextGenerationServiceImpl>(
      text_generation_service_impl_,
      controller,
      request,
      response,
      done,
      "TextGeneration");
}

void APIService::TextGenerationHttp(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  xllm::ClosureGuard done_guard(
      done,
      [](void* /*unused*/) { request_in_metric(nullptr); },
      [controller](void* /*unused*/) {
        request_out_metric(static_cast<void*>(controller));
      });
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  auto arena = GetArenaWithCheck<TextGenerationCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::TextGenerationRequest>(
          arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::TextGenerationResponse>(
          arena);

  brpc::Controller* ctrl = static_cast<brpc::Controller*>(controller);
  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  bool st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }
  std::shared_ptr<TextGenerationCall> call =
      std::make_shared<TextGenerationCall>(
          ctrl, done_guard.release(), req_pb, resp_pb, arena != nullptr);
  text_generation_service_impl_->process_async(call);
}

void APIService::VideoGeneration(::google::protobuf::RpcController* controller,
                                 const proto::VideoGenerationRequest* request,
                                 proto::VideoGenerationResponse* response,
                                 ::google::protobuf::Closure* done) {
  process_typed_brpc_request<VideoGenerationCall, VideoGenerationServiceImpl>(
      video_generation_service_impl_,
      controller,
      request,
      response,
      done,
      "VideoGeneration");
}

void APIService::VideoGenerationHttp(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  xllm::ClosureGuard done_guard(
      done,
      [](void* /*unused*/) { request_in_metric(nullptr); },
      [controller](void* /*unused*/) {
        request_out_metric(static_cast<void*>(controller));
      });
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto arena = GetArenaWithCheck<VideoGenerationCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::VideoGenerationRequest>(
          arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::VideoGenerationResponse>(
          arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  auto st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }
  std::shared_ptr<VideoGenerationCall> call =
      std::make_shared<VideoGenerationCall>(
          ctrl, done_guard.release(), req_pb, resp_pb, arena != nullptr);
  video_generation_service_impl_->process_async(call);
}

void APIService::Rerank(::google::protobuf::RpcController* controller,
                        const proto::RerankRequest* request,
                        proto::RerankResponse* response,
                        ::google::protobuf::Closure* done) {
  process_typed_brpc_request<RerankCall, RerankServiceImpl>(
      rerank_service_impl_, controller, request, response, done, "Rerank");
}

void APIService::RerankHttp(::google::protobuf::RpcController* controller,
                            const proto::HttpRequest* request,
                            proto::HttpResponse* response,
                            ::google::protobuf::Closure* done) {
  xllm::ClosureGuard done_guard(
      done,
      [](void* /*unused*/) { request_in_metric(nullptr); },
      [controller](void* /*unused*/) {
        request_out_metric(static_cast<void*>(controller));
      });
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto arena = GetArenaWithCheck<RerankCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::RerankRequest>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::RerankResponse>(arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  auto st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  std::shared_ptr<Call> call = std::make_shared<RerankCall>(
      ctrl, done_guard.release(), req_pb, resp_pb, arena != nullptr);
  rerank_service_impl_->process_async(call);
}

void APIService::Models(::google::protobuf::RpcController* controller,
                        const proto::ModelListRequest* request,
                        proto::ModelListResponse* response,
                        ::google::protobuf::Closure* done) {
  // TODO with xllm-service
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  bool st_models = models_service_impl_->list_models(nullptr, response);
  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  if (!st_models) {
    ctrl->SetFailed("list models failed.");
    LOG(ERROR) << "list models failed.";
    return;
  }
}

void APIService::ModelsHttp(::google::protobuf::RpcController* controller,
                            const proto::HttpRequest* request,
                            proto::HttpResponse* response,
                            ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto arena = response->GetArena();
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::ModelListResponse>(arena);

  bool st_models = models_service_impl_->list_models(nullptr, resp_pb);
  if (!st_models) {
    LOG(ERROR) << "list models failed.";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  json2pb::Pb2JsonOptions json_options;
  json_options.bytes_to_base64 = false;
  std::string err_msg;
  butil::IOBufAsZeroCopyOutputStream json_output(&ctrl->response_attachment());
  if (!json2pb::ProtoMessageToJson(
          *resp_pb, &json_output, json_options, &err_msg)) {
    LOG(ERROR) << "proto to json failed";
    return;
  }
}

void APIService::ModelVersionsHttp(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  ctrl->response_attachment().append(
      models_service_impl_->list_model_versions());

  return;
}

namespace {

void handle_anthropic_messages(std::unique_ptr<AnthropicServiceImpl>& service,
                               xllm::ClosureGuard& guard,
                               brpc::Controller* ctrl,
                               const proto::HttpRequest* request,
                               proto::HttpResponse* response) {
  auto arena = GetArenaWithCheck<AnthropicCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<typename AnthropicCall::ReqType>(
          arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<typename AnthropicCall::ResType>(
          arena);

  auto content_len = get_json_content_length(ctrl);
  if (content_len == (size_t)-1L) {
    ctrl->SetFailed("Content-Length header is missing.");
    return;
  }
  std::string attachment;
  ctrl->request_attachment().copy_to(&attachment, content_len, 0);

  auto [preprocess_status, processed_json] =
      ChatJsonParser::anthropic().preprocess(std::move(attachment));
  if (!preprocess_status.ok()) {
    ctrl->SetFailed(preprocess_status.message());
    LOG(ERROR) << "Anthropic JSON preprocessing failed: "
               << preprocess_status.message();
    return;
  }

  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  auto status = google::protobuf::util::JsonStringToMessage(
      processed_json, req_pb, options);
  if (!status.ok()) {
    ctrl->SetFailed(status.ToString());
    LOG(ERROR) << "parse json to proto failed: " << status.ToString();
    return;
  }

  auto call = std::make_shared<AnthropicCall>(
      ctrl, guard.release(), req_pb, resp_pb, arena != nullptr /*use_arena*/);

  service->process_async(call);
}

}  // namespace

void APIService::AnthropicMessagesHttp(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  xllm::ClosureGuard done_guard(
      done,
      [](void* /*unused*/) { request_in_metric(nullptr); },
      [controller](void* /*unused*/) {
        request_out_metric(static_cast<void*>(controller));
      });

  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  if (anthropic_service_impl_) {
    handle_anthropic_messages(
        anthropic_service_impl_, done_guard, ctrl, request, response);
  } else {
    ctrl->SetFailed("Anthropic messages API is only supported for LLM engine");
    LOG(ERROR) << "Anthropic messages API is only supported for LLM engine";
  }
}

bool APIService::ParseForkMasterRequest(const proto::MasterInfos* request,
                                        Options& options) {
  if (!std::filesystem::exists(request->model_path())) {
    LOG(ERROR) << "Model path " << request->model_path() << " does not exist.";
    return false;
  }

  std::filesystem::path model_path =
      std::filesystem::path(request->model_path()).lexically_normal();
  std::string model_id;
  if (model_path.has_filename()) {
    model_id = std::filesystem::path(request->model_path()).filename();
  } else {
    model_id =
        std::filesystem::path(request->model_path()).parent_path().filename();
  }
  options.model_id() = model_id;
  options.master_node_addr() = request->master_node_addr();
  options.model_path() = request->model_path();
  options.master_status() = MasterStatus(request->master_status());

  // Parse nnodes and dp_size (tp_size = nnodes / dp_size, computed by engine)
  if (request->nnodes() > 0) {
    options.nnodes() = request->nnodes();
  }
  if (request->dp_size() > 0) {
    options.dp_size() = request->dp_size();
  }

  return true;
}

bool APIService::do_fork_master(const proto::MasterInfos& request,
                                std::string* error_message) {
  if (to_serving_mode(master_->engine_type()) != ServingMode::LLM) {
    *error_message = "fork master only supports LLM engine";
    return false;
  }

  Options master_options;
  if (!ParseForkMasterRequest(&request, master_options)) {
    *error_message = "Failed to parse fork master request";
    return false;
  }

  if (has_model_master(master_options.model_id())) {
    *error_message =
        "Master for model " + master_options.model_id() + " already exists";
    LOG(INFO) << *error_message;
    return true;
  }

  auto master = fork_master(master_, master_options);
  if (!master) {
    *error_message = "Failed to fork master: " + master_options.model_id();
    return false;
  }

  // CAS: only succeed if num_concurrent_requests == 0.
  if (master->is_sleeping() &&
      !master->get_rate_limiter()->try_set_sleeping()) {
    // Notice: this branch is only entered in exceptional cases.
    int32_t num_requests =
        master->get_rate_limiter()->get_num_concurrent_requests();
    LOG(FATAL) << "Cannot sleep model " << request.model_id() << " with "
               << num_requests << " in-flight requests";
    *error_message = "Cannot sleep model with in-flight requests";
    return false;
  }

  if (!add_model_master_if_absent(master_options.model_id(), master.get())) {
    *error_message =
        "Master for model " + master_options.model_id() + " already exists";
    LOG(INFO) << *error_message;
    return true;
  }
  if (::xllm::DistributedConfig::get_instance().node_rank() == 0) {
    auto llm_master = dynamic_cast<LLMMaster*>(master.get());
    completion_service_impl_->add_model_master(master_options.model_id(),
                                               llm_master);
    chat_service_impl_->add_model_master(master_options.model_id(), llm_master);
  }
  master.release();
  return true;
}

void APIService::ForkMaster(::google::protobuf::RpcController* controller,
                            const proto::MasterInfos* request,
                            proto::Status* response,
                            ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  std::string error_message;
  bool ok = do_fork_master(*request, &error_message);
  response->set_ok(ok);
  if (!ok) {
    LOG(ERROR) << "fork_master failed: " << error_message;
    ctrl->SetFailed(error_message);
  }
}

void APIService::ForkMasterHttp(::google::protobuf::RpcController* controller,
                                const proto::HttpRequest* request,
                                proto::HttpResponse* response,
                                ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::MasterInfos>(arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  auto st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  std::string error_message;
  if (!do_fork_master(*req_pb, &error_message)) {
    LOG(ERROR) << "fork_master failed: " << error_message;
    ctrl->SetFailed(error_message);
  }
}

bool APIService::do_sleep(const proto::MasterInfos& request,
                          std::string* error_message) {
  const auto req_master_status = MasterStatus(request.master_status());
  if (req_master_status != MasterStatus::LIGHT_SLEEP &&
      req_master_status != MasterStatus::DEEP_SLEEP) {
    LOG(ERROR) << "Invalid sleep status: " << request.master_status();
    *error_message = "Invalid sleep status";
    return false;
  }

  Master* master = get_model_master(request.model_id());
  if (master == nullptr) {
    LOG(ERROR) << "Master for model " << request.model_id() << " not found";
    *error_message = "Master for model not found";
    return false;
  }
  if (master->is_sleeping()) {
    LOG(INFO) << "Master for model " << request.model_id()
              << " is already sleeping";
    *error_message = "Master for model is already sleeping";
    return false;
  }

  // CAS: only succeed if num_concurrent_requests == 0.
  if (!master->get_rate_limiter()->try_set_sleeping()) {
    int32_t num_requests =
        master->get_rate_limiter()->get_num_concurrent_requests();
    LOG(ERROR) << "Cannot sleep model " << request.model_id() << " with "
               << num_requests << " in-flight requests";
    *error_message = "Cannot sleep model with in-flight requests";
    return false;
  }

  auto master_status = master->get_master_status();
  master->set_master_status(req_master_status);
  if (!master->sleep()) {
    master->set_master_status(master_status);
    LOG(ERROR) << "Failed to sleep model " << request.model_id();
    *error_message = "Failed to sleep model";
    return false;
  }
  return true;
}

void APIService::Sleep(::google::protobuf::RpcController* controller,
                       const proto::MasterInfos* request,
                       proto::Status* response,
                       ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  std::string error_message;
  bool ok = do_sleep(*request, &error_message);
  response->set_ok(ok);
  if (!ok) {
    ctrl->SetFailed(error_message);
  }
}

void APIService::SleepHttp(::google::protobuf::RpcController* controller,
                           const proto::HttpRequest* request,
                           proto::HttpResponse* response,
                           ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::MasterInfos>(arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  auto st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  std::string error_message;
  if (!do_sleep(*req_pb, &error_message)) {
    ctrl->SetFailed(error_message);
  }
  // Success: return HTTP 200 with empty body
}

bool APIService::do_wakeup(const proto::MasterInfos& request,
                           std::string* error_message) {
  Master* master = get_model_master(request.model_id());
  if (master == nullptr) {
    LOG(ERROR) << "Master for model " << request.model_id() << " not found";
    *error_message = "Master for model not found";
    return false;
  }
  if (!master->is_sleeping()) {
    LOG(INFO) << "Master for model " << request.model_id()
              << " is already awake";
    *error_message = "Master for model is already awake";
    return false;
  }

  // Check if remote weight transfer is requested
  if (request.remote_addrs_size() > 0) {
    WakeupOptions wakeup_options;
    wakeup_options.remote_addrs.assign(request.remote_addrs().begin(),
                                       request.remote_addrs().end());
    if (request.src_weight_segments_size() > 0) {
      for (const auto& seg_list : request.src_weight_segments()) {
        std::vector<WeightSegment> segments;
        segments.reserve(seg_list.segments_size());
        for (const auto& proto_seg : seg_list.segments()) {
          segments.emplace_back(proto_seg.offset(), proto_seg.size());
        }
        wakeup_options.src_weight_segments.push_back(std::move(segments));
      }
    }
    if (!master->wakeup(wakeup_options)) {
      LOG(ERROR) << "Failed to wakeup model " << request.model_id()
                 << " with remote weight transfer";
      *error_message = "Failed to wakeup model with remote weight transfer";
      return false;
    }
  } else {
    if (!master->wakeup()) {
      LOG(ERROR) << "Failed to wakeup model " << request.model_id();
      *error_message = "Failed to wakeup model";
      return false;
    }
  }

  // Restore rate limiter from sleeping state
  if (!master->get_rate_limiter()->try_wakeup()) {
    LOG(ERROR) << "Failed to restore rate limiter for model "
               << request.model_id();
    *error_message = "Failed to restore rate limiter";
    return false;
  }

  master->set_master_status(MasterStatus::WAKEUP);
  return true;
}

void APIService::Wakeup(::google::protobuf::RpcController* controller,
                        const proto::MasterInfos* request,
                        proto::Status* response,
                        ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  std::string error_message;
  bool ok = do_wakeup(*request, &error_message);
  response->set_ok(ok);
  if (!ok) {
    ctrl->SetFailed(error_message);
  }
}

void APIService::WakeupHttp(::google::protobuf::RpcController* controller,
                            const proto::HttpRequest* request,
                            proto::HttpResponse* response,
                            ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::MasterInfos>(arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  auto st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  std::string error_message;
  if (!do_wakeup(*req_pb, &error_message)) {
    ctrl->SetFailed(error_message);
  }
  // Success: return HTTP 200 with empty body
}

void APIService::StartProfileHttp(::google::protobuf::RpcController* controller,
                                  const proto::HttpRequest* request,
                                  proto::HttpResponse* response,
                                  ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  if (!ProfileConfig::get_instance().enable_online_profile()) {
    LOG(ERROR) << "Profiling is disabled. Start the server with "
                  "--enable_online_profile=true to use /start_profile.";
    ctrl->SetFailed(
        "Profiling is disabled. Start the server with "
        "--enable_online_profile=true.");
    return;
  }
  if (master_ == nullptr) {
    LOG(ERROR) << "No master available to start profiling.";
    ctrl->SetFailed("No master available to start profiling.");
    return;
  }

  LOG(INFO) << "Starting profiler.";
  if (!master_->start_profile()) {
    LOG(ERROR) << "Failed to start profiler.";
    ctrl->SetFailed("Failed to start profiler.");
    return;
  }
  LOG(INFO) << "Profiler started.";
  // Success: return HTTP 200 with empty body
}

void APIService::StopProfileHttp(::google::protobuf::RpcController* controller,
                                 const proto::HttpRequest* request,
                                 proto::HttpResponse* response,
                                 ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  if (!ProfileConfig::get_instance().enable_online_profile()) {
    LOG(ERROR) << "Profiling is disabled. Start the server with "
                  "--enable_online_profile=true to use /stop_profile.";
    ctrl->SetFailed(
        "Profiling is disabled. Start the server with "
        "--enable_online_profile=true.");
    return;
  }
  if (master_ == nullptr) {
    LOG(ERROR) << "No master available to stop profiling.";
    ctrl->SetFailed("No master available to stop profiling.");
    return;
  }

  LOG(INFO) << "Stopping profiler.";
  if (!master_->stop_profile()) {
    LOG(ERROR) << "Failed to stop profiler.";
    ctrl->SetFailed("Failed to stop profiler.");
    return;
  }
  LOG(INFO) << "Profiler stopped.";
  // Success: return HTTP 200 with empty body
}

void APIService::LinkP2P(::google::protobuf::RpcController* controller,
                         const proto::P2PLinkRequest* request,
                         proto::Status* response,
                         ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  Master* master = get_model_master(request->model_id());
  if (master == nullptr) {
    LOG(ERROR) << "Master for model " << request->model_id() << " not found";
    response->set_ok(false);
    return;
  }
  bool status = master->link_p2p(
      {request->remote_addrs().begin(), request->remote_addrs().end()});
  response->set_ok(status);
}

void APIService::LinkP2PHttp(::google::protobuf::RpcController* controller,
                             const proto::HttpRequest* request,
                             proto::HttpResponse* response,
                             ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::P2PLinkRequest>(arena);
  auto resp_pb = google::protobuf::Arena::CreateMessage<proto::Status>(arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  auto st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  Master* master = get_model_master(req_pb->model_id());
  if (master == nullptr) {
    LOG(ERROR) << "Master for model " << req_pb->model_id() << " not found";
    ctrl->SetFailed("Master for model not found");
    return;
  }
  bool status = master->link_p2p(
      {req_pb->remote_addrs().begin(), req_pb->remote_addrs().end()});
  resp_pb->set_ok(status);

  json2pb::Pb2JsonOptions json_options;
  json_options.bytes_to_base64 = false;
  std::string err_msg;
  butil::IOBufAsZeroCopyOutputStream json_output(&ctrl->response_attachment());
  if (!json2pb::ProtoMessageToJson(
          *resp_pb, &json_output, json_options, &err_msg)) {
    LOG(ERROR) << "proto to json failed: " << err_msg;
    return;
  }
}

void APIService::UnlinkP2P(::google::protobuf::RpcController* controller,
                           const proto::P2PLinkRequest* request,
                           proto::Status* response,
                           ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  Master* master = get_model_master(request->model_id());
  if (master == nullptr) {
    LOG(ERROR) << "Master for model " << request->model_id() << " not found";
    response->set_ok(false);
    return;
  }
  bool status = master->unlink_p2p(
      {request->remote_addrs().begin(), request->remote_addrs().end()});
  response->set_ok(status);
}

void APIService::UnlinkP2PHttp(::google::protobuf::RpcController* controller,
                               const proto::HttpRequest* request,
                               proto::HttpResponse* response,
                               ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::P2PLinkRequest>(arena);
  auto resp_pb = google::protobuf::Arena::CreateMessage<proto::Status>(arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  auto st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  Master* master = get_model_master(req_pb->model_id());
  if (master == nullptr) {
    LOG(ERROR) << "Master for model " << req_pb->model_id() << " not found";
    ctrl->SetFailed("Master for model not found");
    return;
  }
  bool status = master->unlink_p2p(
      {req_pb->remote_addrs().begin(), req_pb->remote_addrs().end()});
  resp_pb->set_ok(status);

  json2pb::Pb2JsonOptions json_options;
  json_options.bytes_to_base64 = false;
  std::string err_msg;
  butil::IOBufAsZeroCopyOutputStream json_output(&ctrl->response_attachment());
  if (!json2pb::ProtoMessageToJson(
          *resp_pb, &json_output, json_options, &err_msg)) {
    LOG(ERROR) << "proto to json failed: " << err_msg;
    return;
  }
}

// ============== Async RL training support: Pause/Resume ==============
void APIService::Pause(::google::protobuf::RpcController* controller,
                       const proto::PauseRequest* request,
                       proto::PauseResponse* response,
                       ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  if (!request || !response || !controller) {
    LOG(ERROR) << "Pause: request/response/controller is null";
    return;
  }

  auto* llm_master = dynamic_cast<LLMMaster*>(master_);
  if (!llm_master) {
    LOG(ERROR) << "Master is not an LLMMaster";
    response->set_status("error: not an LLMMaster");
    return;
  }

  const std::string& mode = request->mode();
  llm_master->pause_scheduler(mode);

  response->set_status("paused");
  LOG(INFO) << "Pause completed successfully (mode="
            << (mode.empty() ? "keep" : mode) << ")";
}

void APIService::PauseHttp(::google::protobuf::RpcController* controller,
                           const proto::HttpRequest* request,
                           proto::HttpResponse* response,
                           ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  if (!request || !response || !controller) {
    LOG(ERROR) << "PauseHttp: request/response/controller is null";
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::PauseRequest>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::PauseResponse>(arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  // Parse JSON request (if body provided, otherwise use defaults)
  if (ctrl->request_attachment().size() > 0) {
    std::string error;
    json2pb::Json2PbOptions options;
    butil::IOBuf& buf = ctrl->request_attachment();
    butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
    auto st =
        json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
    if (!st) {
      ctrl->SetFailed(error);
      LOG(ERROR) << "PauseHttp: parse json to proto failed: " << error;
      return;
    }
  }

  // Call Pause
  Pause(controller, req_pb, resp_pb, nullptr);

  // Convert response to JSON
  std::string json_output;
  json2pb::ProtoMessageToJson(*resp_pb, &json_output, nullptr);
  ctrl->response_attachment().append(json_output);
}

void APIService::Resume(::google::protobuf::RpcController* controller,
                        const proto::ResumeRequest* request,
                        proto::ResumeResponse* response,
                        ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  if (!request || !response || !controller) {
    LOG(ERROR) << "Resume: request/response/controller is null";
    return;
  }

  auto* llm_master = dynamic_cast<LLMMaster*>(master_);
  if (!llm_master) {
    LOG(ERROR) << "Master is not an LLMMaster";
    response->set_status("error: not an LLMMaster");
    return;
  }

  llm_master->resume_scheduler();

  response->set_status("running");
  LOG(INFO) << "Resume completed successfully";
}

void APIService::ResumeHttp(::google::protobuf::RpcController* controller,
                            const proto::HttpRequest* request,
                            proto::HttpResponse* response,
                            ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  if (!request || !response || !controller) {
    LOG(ERROR) << "ResumeHttp: request/response/controller is null";
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::ResumeRequest>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::ResumeResponse>(arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  // Call Resume (no request body needed)
  Resume(controller, req_pb, resp_pb, nullptr);

  // Convert response to JSON
  std::string json_output;
  json2pb::ProtoMessageToJson(*resp_pb, &json_output, nullptr);
  ctrl->response_attachment().append(json_output);
}

}  // namespace xllm
