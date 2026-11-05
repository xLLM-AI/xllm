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

#include "api_service/speech_to_text_service_impl.h"

#include <absl/strings/ascii.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/controller.h>
#include <butil/base64.h>
#include <glog/logging.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "api_service/multipart_form_data.h"
#include "api_service/request_id.h"
#include "api_service/rpc_request_metrics.h"
#include "core/common/instance_name.h"
#include "core/common/message.h"
#include "core/common/types.h"
#include "core/distributed_runtime/vlm_master.h"
#include "core/framework/config/model_config.h"
#include "core/framework/multimodal/mm_input.h"
#include "core/framework/request/request_output.h"
#include "core/framework/request/request_params.h"
#include "core/util/closure_guard.h"
#include "core/util/scope_guard.h"
#include "core/util/uuid.h"
#include "models/model_registry.h"
#include "models/speech_model.h"

namespace xllm {

namespace {

constexpr std::string_view kDefaultMimeType = "audio/wav";
constexpr std::string_view kJsonFormat = "json";
constexpr std::string_view kTextFormat = "text";
constexpr std::string_view kVerboseJsonFormat = "verbose_json";

std::string guess_mime_type(const proto::SpeechToTextRequest& request) {
  // Only trust explicit audio/* types; curl often sends octet-stream.
  if (request.file_content_type().rfind("audio/", 0) == 0) {
    return request.file_content_type();
  }
  const std::string& filename = request.filename();
  const size_t dot = filename.rfind('.');
  if (dot == std::string::npos) {
    return std::string(kDefaultMimeType);
  }
  std::string extension = filename.substr(dot + 1);
  absl::AsciiStrToLower(&extension);
  if (extension == "mp3" || extension == "mpga" || extension == "mpeg") {
    return "audio/mpeg";
  }
  if (extension == "m4a" || extension == "mp4") {
    return "audio/mp4";
  }
  if (extension == "ogg" || extension == "opus") {
    return "audio/ogg";
  }
  if (extension == "webm") {
    return "audio/webm";
  }
  if (extension == "flac") {
    return "audio/flac";
  }
  if (extension == "aac") {
    return "audio/aac";
  }
  return std::string(kDefaultMimeType);
}

bool parse_form_value(std::string_view name,
                      std::string_view value,
                      bool* out,
                      std::string* error) {
  if (value == "true" || value == "1") {
    *out = true;
    return true;
  }
  if (value == "false" || value == "0") {
    *out = false;
    return true;
  }
  *error = "Field `" + std::string(name) + "` must be a boolean.";
  return false;
}

template <typename T>
bool parse_form_value(std::string_view name,
                      std::string_view value,
                      T* out,
                      std::string* error) {
  if constexpr (std::is_same_v<T, uint32_t>) {
    int64_t parsed = 0;
    if (!parse_form_value(name, value, &parsed, error)) {
      return false;
    }
    if (parsed < 0 || parsed > static_cast<int64_t>(UINT32_MAX)) {
      *error = "Field `" + std::string(name) + "` is out of range.";
      return false;
    }
    *out = static_cast<uint32_t>(parsed);
    return true;
  } else {
    T parsed{};
    const auto result =
        std::from_chars(value.data(), value.data() + value.size(), parsed);
    const bool parsed_ok =
        result.ec == std::errc() && result.ptr == value.data() + value.size();
    if constexpr (std::is_floating_point_v<T>) {
      if (!parsed_ok || !std::isfinite(parsed)) {
        *error = "Field `" + std::string(name) + "` must be a finite number.";
        return false;
      }
    } else {
      if (!parsed_ok) {
        *error = "Field `" + std::string(name) + "` must be an integer.";
        return false;
      }
    }
    *out = parsed;
    return true;
  }
}

template <typename T>
bool set_form_field(const api_service::MultipartFormData& form,
                    std::string_view field,
                    void (proto::SpeechToTextRequest::*setter)(T),
                    proto::SpeechToTextRequest& request,
                    std::string* error) {
  const auto value = api_service::find_multipart_field_value(form, field);
  if (!value.has_value()) {
    return true;
  }
  T parsed{};
  if (!parse_form_value(field, value.value(), &parsed, error)) {
    return false;
  }
  (request.*setter)(parsed);
  return true;
}

Status load_speech_audio(const proto::SpeechToTextRequest& request,
                         butil::IOBuf payload,
                         uint64_t max_bytes,
                         std::string& audio_bytes) {
  if (request.file().type() == "binary") {
    const auto& binary = request.file().binary();
    const uint64_t offset = binary.offset();
    const uint64_t length = binary.length();
    if (offset > payload.size() || length > payload.size() - offset) {
      return {StatusCode::INVALID_ARGUMENT,
              "Audio file binary reference is out of range."};
    }
    if (length > max_bytes) {
      return {StatusCode::INVALID_ARGUMENT,
              "Maximum audio file size exceeded."};
    }
    payload.pop_front(static_cast<size_t>(offset));
    payload.copy_to(&audio_bytes, static_cast<size_t>(length));
  } else {
    // Bound the allocation before decoding the inline payload.
    const uint64_t max_base64_bytes = ((max_bytes + 2) / 3) * 4;
    if (request.file().base64().size() > max_base64_bytes) {
      return {StatusCode::INVALID_ARGUMENT,
              "Maximum audio file size exceeded."};
    }
    if (!butil::Base64Decode(butil::StringPiece(request.file().base64()),
                             &audio_bytes)) {
      return {StatusCode::INVALID_ARGUMENT,
              "Failed to decode base64 audio file."};
    }
    if (audio_bytes.size() > max_bytes) {
      return {StatusCode::INVALID_ARGUMENT,
              "Maximum audio file size exceeded."};
    }
  }
  if (audio_bytes.empty()) {
    return {StatusCode::INVALID_ARGUMENT, "Audio file must not be empty."};
  }
  return {};
}

}  // namespace

SpeechToTextServiceImpl::SpeechToTextServiceImpl(
    VLMMaster* master,
    const std::vector<std::string>& models,
    SpeechToTextTask task)
    : APIServiceImpl(models), master_{master}, task_{task} {
  CHECK(master_ != nullptr);
  std::string resolved_name;
  if (resolve_model_registration_name(master_->model_type(), &resolved_name)) {
    const SpeechModelFactory factory =
        ModelRegistry::get_speech_model_factory(resolved_name);
    adapter_ = factory != nullptr ? factory() : nullptr;
  }
}

void SpeechToTextServiceImpl::process_http_request(
    SpeechToTextServiceImpl* service,
    google::protobuf::RpcController* controller,
    google::protobuf::Closure* done,
    std::string_view service_name) {
  ClosureGuard guard(
      done,
      [](void* /*unused*/) { request_in_metric(nullptr); },
      [controller](void* /*unused*/) {
        request_out_metric(static_cast<void*>(controller));
      });

  auto* ctrl = static_cast<brpc::Controller*>(controller);
  api_service::ensure_http_x_request_id(ctrl);
  if (service == nullptr) {
    ctrl->SetFailed(std::string(service_name) +
                    " service is not available on this server");
    return;
  }

  const std::string& content_type = ctrl->http_request().content_type();
  if (content_type.rfind("multipart/form-data", 0) != 0) {
    ctrl->SetFailed(
        "Content-Type must be multipart/form-data for speech-to-text "
        "requests.");
    return;
  }

  auto req_pb = std::make_unique<proto::SpeechToTextRequest>();
  auto resp_pb = std::make_unique<proto::SpeechToTextResponse>();

  api_service::MultipartFormData form;
  Status parse_status = api_service::parse_multipart_form_data(
      content_type,
      ctrl->request_attachment(),
      static_cast<size_t>(
          ModelConfig::get_instance().audio_max_upload_file_mb()) *
          1024 * 1024,
      form);
  if (!parse_status.ok()) {
    ctrl->SetFailed(parse_status.message());
    LOG(ERROR) << "parse multipart form data failed: "
               << parse_status.message();
    return;
  }

  const auto file_iter =
      std::find_if(form.parts.begin(), form.parts.end(), [](const auto& part) {
        return !part.filename.empty() && part.name == "file";
      });
  if (file_iter == form.parts.end()) {
    ctrl->SetFailed("Expected `file` to be a file-like object.");
    return;
  }
  api_service::MultipartPart& file_part = *file_iter;

  std::string error;
  auto& request = *req_pb;
  auto set_string_field =
      [&form, &request](
          std::string_view field,
          void (proto::SpeechToTextRequest::*setter)(const std::string&)) {
        const auto value = api_service::find_multipart_field_value(form, field);
        if (value.has_value()) {
          (request.*setter)(value.value());
        }
      };

  set_string_field("model", &proto::SpeechToTextRequest::set_model);
  set_string_field("language", &proto::SpeechToTextRequest::set_language);
  set_string_field("prompt", &proto::SpeechToTextRequest::set_prompt);
  set_string_field("response_format",
                   &proto::SpeechToTextRequest::set_response_format);
  set_string_field("to_language", &proto::SpeechToTextRequest::set_to_language);
  set_string_field("hotwords", &proto::SpeechToTextRequest::set_hotwords);
  set_string_field("user", &proto::SpeechToTextRequest::set_user);

  for (const auto& part : form.parts) {
    if (!part.filename.empty() || part.name != "timestamp_granularities[]") {
      continue;
    }
    request.add_timestamp_granularities(part.value.to_string());
  }

  auto set_typed_field = [&form, &request, &error](std::string_view field,
                                                   auto setter) {
    return set_form_field(form, field, setter, request, &error);
  };

  if (!set_typed_field("stream", &proto::SpeechToTextRequest::set_stream) ||
      !set_typed_field("stream_include_usage",
                       &proto::SpeechToTextRequest::set_stream_include_usage) ||
      !set_typed_field(
          "stream_continuous_usage_stats",
          &proto::SpeechToTextRequest::set_stream_continuous_usage_stats) ||
      !set_typed_field("temperature",
                       &proto::SpeechToTextRequest::set_temperature) ||
      !set_typed_field("top_p", &proto::SpeechToTextRequest::set_top_p) ||
      !set_typed_field("min_p", &proto::SpeechToTextRequest::set_min_p) ||
      !set_typed_field("seed", &proto::SpeechToTextRequest::set_seed) ||
      !set_typed_field("frequency_penalty",
                       &proto::SpeechToTextRequest::set_frequency_penalty) ||
      !set_typed_field("presence_penalty",
                       &proto::SpeechToTextRequest::set_presence_penalty) ||
      !set_typed_field("repetition_penalty",
                       &proto::SpeechToTextRequest::set_repetition_penalty) ||
      !set_typed_field("top_k", &proto::SpeechToTextRequest::set_top_k) ||
      !set_typed_field(
          "max_completion_tokens",
          &proto::SpeechToTextRequest::set_max_completion_tokens)) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "invalid speech-to-text form fields: " << error;
    return;
  }

  if (!file_part.filename.empty()) {
    request.set_filename(file_part.filename);
  }
  if (!file_part.content_type.empty()) {
    request.set_file_content_type(file_part.content_type);
  }

  auto* file = req_pb->mutable_file();
  file->set_type("binary");
  file->mutable_binary()->set_offset(0);
  file->mutable_binary()->set_length(file_part.value.size());

  auto call = std::make_shared<SpeechToTextCall>(ctrl,
                                                 guard.release(),
                                                 req_pb.release(),
                                                 resp_pb.release(),
                                                 /*use_arena=*/false,
                                                 /*is_http_request=*/true,
                                                 /*defer_stream_start=*/true);
  call->set_request_payload(std::move(file_part.value));
  service->process_async_impl(std::move(call));
}

void SpeechToTextServiceImpl::process_async_impl(
    std::shared_ptr<SpeechToTextCall> call) {
  const auto& rpc_request = call->request();

  std::string model = rpc_request.model();
  if (model.empty()) {
    model = *models_.begin();
  }
  if (!models_.contains(model)) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  const std::string response_format = rpc_request.response_format().empty()
                                          ? std::string(kJsonFormat)
                                          : rpc_request.response_format();
  if (response_format != kJsonFormat && response_format != kTextFormat &&
      response_format != kVerboseJsonFormat) {
    call->finish_with_error(
        StatusCode::INVALID_ARGUMENT,
        "Currently only support response_format: `text`, `json` or "
        "`verbose_json`.");
    return;
  }

  if (adapter_ == nullptr) {
    call->finish_with_error(
        StatusCode::INVALID_ARGUMENT,
        "The current model does not support speech-to-text.");
    return;
  }

  const bool verbose = response_format == kVerboseJsonFormat;
  if (verbose && !adapter_->supports_verbose_json()) {
    call->finish_with_error(
        StatusCode::INVALID_ARGUMENT,
        "Currently do not support verbose_json for " + model);
    return;
  }

  if (rpc_request.timestamp_granularities_size() > 0 &&
      !adapter_->supports_segment_timestamps()) {
    call->finish_with_error(
        StatusCode::INVALID_ARGUMENT,
        "Timestamp granularities are not supported by the current model.");
    return;
  }
  const bool stream = rpc_request.stream();
  if (verbose && stream) {
    call->finish_with_error(
        StatusCode::INVALID_ARGUMENT,
        "`verbose_json` format doesn't support streaming case.");
    return;
  }
  if (!stream && (rpc_request.stream_include_usage() ||
                  rpc_request.stream_continuous_usage_stats())) {
    call->finish_with_error(
        StatusCode::INVALID_ARGUMENT,
        "Stream options can only be defined when `stream=true`.");
    return;
  }
  if (task_ == SpeechToTextTask::TRANSLATE &&
      !adapter_->supports_translation()) {
    call->finish_with_error(StatusCode::INVALID_ARGUMENT,
                            "Task 'translate' is not supported by the "
                            "current model.");
    return;
  }
  const Status language_status =
      adapter_->validate_language(rpc_request.language());
  if (!language_status.ok()) {
    call->finish_with_error(language_status.code(), language_status.message());
    return;
  }
  if (!rpc_request.has_file() || (rpc_request.file().type() != "binary" &&
                                  rpc_request.file().type() != "base64")) {
    call->finish_with_error(StatusCode::INVALID_ARGUMENT,
                            "Expected `file` to be a file-like object.");
    return;
  }

  const Status admission = master_->get_rate_limiter()->acquire();
  if (!admission.ok()) {
    call->finish_with_error(admission.code(), admission.message());
    return;
  }
  ScopeGuard admission_guard(
      [this] { master_->get_rate_limiter()->decrease_one_request(); });

  std::string audio_bytes;
  const auto& config = ModelConfig::get_instance();
  const uint64_t max_bytes =
      static_cast<uint64_t>(config.audio_max_upload_file_mb()) * 1024 * 1024;
  const Status audio_status = load_speech_audio(
      rpc_request, call->take_request_iobuf(), max_bytes, audio_bytes);
  if (!audio_status.ok()) {
    call->finish_with_error(audio_status.code(), audio_status.message());
    return;
  }

  SpeechToTextParams speech_params;
  speech_params.task = task_;
  speech_params.request_prompt = rpc_request.prompt();
  speech_params.hotwords = rpc_request.hotwords();
  speech_params.mime_type = guess_mime_type(rpc_request);
  speech_params.audio_bytes = audio_bytes.size();
  speech_params.max_duration_s = config.audio_max_decode_duration_s();
  std::vector<Message> messages = adapter_->build_messages(speech_params);
  RequestParams request_params;
  if (rpc_request.request_id().empty()) {
    thread_local ShortUUID short_uuid;
    const char* prefix =
        task_ == SpeechToTextTask::TRANSLATE ? "trsl-" : "trsc-";
    request_params.request_id = std::string(prefix) +
                                InstanceName::name()->get_name_hash() + "-" +
                                short_uuid.random();
  } else {
    request_params.request_id = rpc_request.request_id();
  }
  request_params.x_request_id = call->get_x_request_id();
  request_params.x_request_time = call->get_x_request_time();
  request_params.streaming = stream;
  request_params.max_tokens = rpc_request.max_completion_tokens();
  if (rpc_request.has_temperature()) {
    request_params.temperature = rpc_request.temperature();
  }
  if (rpc_request.has_top_p()) {
    request_params.top_p = rpc_request.top_p();
  }
  if (rpc_request.has_top_k()) {
    request_params.top_k = rpc_request.top_k() == 0 ? -1 : rpc_request.top_k();
  }
  if (rpc_request.has_min_p()) {
    request_params.min_p = rpc_request.min_p();
  }
  if (rpc_request.has_frequency_penalty()) {
    request_params.frequency_penalty = rpc_request.frequency_penalty();
  }
  if (rpc_request.has_presence_penalty()) {
    request_params.presence_penalty = rpc_request.presence_penalty();
  }
  if (rpc_request.has_repetition_penalty()) {
    request_params.repetition_penalty = rpc_request.repetition_penalty();
  }

  std::string saved_request_id = request_params.request_id;

  admission_guard.dismiss();
  master_->handle_request(
      std::move(messages),
      std::move(request_params),
      MMInput(std::move(audio_bytes)),
      [call,
       model = std::move(model),
       task = task_,
       stream,
       include_usage = rpc_request.stream_include_usage(),
       continuous_usage = rpc_request.stream_continuous_usage_stats(),
       request_id = std::move(saved_request_id),
       created_time = static_cast<uint32_t>(absl::ToUnixSeconds(absl::Now())),
       response_format,
       language = rpc_request.language(),
       stream_usage = std::optional<proto::Usage>()](
          const RequestOutput& req_output) mutable -> bool {
        req_output.log_request_status();
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            return call->finish_with_error(status.code(), status.message());
          }
        }
        if (req_output.cancelled) {
          return call->finish_with_error(StatusCode::CANCELLED,
                                         "The request was cancelled.");
        }
        if (!stream) {
          auto& response = call->response();
          response.set_text(req_output.outputs.empty()
                                ? ""
                                : req_output.outputs.front().text);
          if (response_format == kVerboseJsonFormat) {
            response.set_duration(req_output.input_audio_duration_s);
            response.set_language(language);
          } else if (response_format == kJsonFormat &&
                     task == SpeechToTextTask::TRANSCRIBE) {
            auto* usage = response.mutable_usage();
            usage->set_type("duration");
            usage->set_seconds(static_cast<int32_t>(
                std::ceil(req_output.input_audio_duration_s)));
          }
          if (response_format == kTextFormat) {
            return call->write_text_and_finish(response.text());
          }
          return call->write_and_finish(response);
        }

        const char* chunk_object = task == SpeechToTextTask::TRANSLATE
                                       ? "translation.chunk"
                                       : "transcription.chunk";
        if (req_output.usage.has_value()) {
          const auto& usage = req_output.usage.value();
          auto& proto_usage = stream_usage.emplace();
          proto_usage.set_prompt_tokens(usage.num_prompt_tokens);
          proto_usage.set_completion_tokens(usage.num_generated_tokens);
          proto_usage.set_total_tokens(usage.num_total_tokens);
        }
        auto init_chunk = [&](proto::SpeechToTextStreamChunk& chunk) {
          chunk.set_id(request_id);
          chunk.set_object(chunk_object);
          chunk.set_created(created_time);
          chunk.set_model(model);
        };
        auto write_chunk = [&](std::string_view delta,
                               const std::string* finish_reason) {
          proto::SpeechToTextStreamChunk chunk;
          init_chunk(chunk);
          auto* choice = chunk.add_choices();
          choice->set_index(0);
          auto* content = choice->mutable_delta();
          if (!delta.empty()) {
            content->set_content(delta.data(), delta.size());
          }
          if (finish_reason != nullptr) {
            choice->set_finish_reason(*finish_reason);
          }
          if (continuous_usage && stream_usage.has_value()) {
            *chunk.mutable_usage() = stream_usage.value();
          }
          return call->write(chunk);
        };
        for (const auto& seq_output : req_output.outputs) {
          if (!seq_output.text.empty() &&
              !write_chunk(seq_output.text, /*finish_reason=*/nullptr)) {
            return false;
          }
          if (seq_output.finish_reason.has_value() &&
              !write_chunk(/*delta=*/"", &seq_output.finish_reason.value())) {
            return false;
          }
        }
        if (!req_output.finished) {
          return true;
        }
        if (include_usage) {
          proto::SpeechToTextStreamChunk chunk;
          init_chunk(chunk);
          auto* usage = chunk.mutable_usage();
          if (stream_usage.has_value()) {
            *usage = stream_usage.value();
          }
          if (!call->write(chunk)) {
            return false;
          }
        }
        return call->finish();
      });
}

}  // namespace xllm
