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

#include "dit_generation.h"

#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>

#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "core/distributed_runtime/dit_master.h"

namespace xllm {
namespace {

template <typename Request>
std::optional<DiTRequestParams> parse_request(const std::string& json) {
  Request proto;
  const auto status = google::protobuf::util::JsonStringToMessage(json, &proto);
  if (!status.ok()) {
    LOG(ERROR) << "Invalid generation request: " << status.ToString();
    return std::nullopt;
  }
  // Request decoding may reject malformed numeric strings (such as size).
  // Keep this untrusted-input boundary outside model execution.
  try {
    return DiTRequestParams(proto, /*x_rid=*/"", /*x_rtime=*/"");
  } catch (const std::invalid_argument& error) {
    LOG(ERROR) << "Invalid generation request: " << error.what();
  } catch (const std::out_of_range& error) {
    LOG(ERROR) << "Invalid generation request: " << error.what();
  }
  return std::nullopt;
}

bool write_output(const std::filesystem::path& path, std::string_view data) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(data.data(), static_cast<std::streamsize>(data.size()));
  stream.close();
  if (!stream) {
    LOG(ERROR) << "Failed to save generation output to " << path;
    return false;
  }
  LOG(INFO) << "Saved " << path << " (" << data.size() << " bytes)";
  return true;
}

}  // namespace

std::optional<DiTRequestParams> load_dit_generation_request(
    const DiTGenerationCommand& command) {
  std::ifstream stream(command.request_path);
  if (!stream) {
    LOG(ERROR) << "Cannot open generation request: " << command.request_path;
    return std::nullopt;
  }
  const std::string json((std::istreambuf_iterator<char>(stream)),
                         std::istreambuf_iterator<char>());
  if (command.modality == "image") {
    return parse_request<proto::ImageGenerationRequest>(json);
  }
  if (command.modality == "video") {
    return parse_request<proto::VideoGenerationRequest>(json);
  }
  if (command.modality == "audio") {
    return parse_request<proto::AudioGenerationRequest>(json);
  }
  if (command.modality == "text") {
    return parse_request<proto::TextGenerationRequest>(json);
  }
  LOG(ERROR) << "Unsupported generation modality: " << command.modality;
  return std::nullopt;
}

int32_t generate_dit_output(DiTMaster& master,
                            DiTRequestParams request,
                            const std::string& output_path) {
  const Status admission = master.get_rate_limiter()->acquire();
  if (!admission.ok()) {
    LOG(ERROR) << admission.message();
    return 1;
  }
  const DiTRequestKind kind = request.request_kind;
  std::promise<DiTRequestOutput> completion;
  auto result = completion.get_future();
  master.handle_request(
      std::move(request), std::nullopt, [&completion](DiTRequestOutput output) {
        completion.set_value(std::move(output));
        return true;
      });
  const DiTRequestOutput output = result.get();
  if (!output.status.has_value() || !output.status->ok() || output.cancelled ||
      !output.finished || output.outputs.empty()) {
    LOG(ERROR) << "Generation failed: "
               << (output.status.has_value() ? output.status->message()
                                             : "missing completion status");
    return 1;
  }

  const std::filesystem::path base(output_path);
  for (const auto& item : output.outputs) {
    std::filesystem::path path = base;
    if (output.outputs.size() > 1) {
      path = base.parent_path() /
             (base.stem().string() + "_" + std::to_string(item.index) +
              base.extension().string());
    }
    std::string_view data;
    switch (kind) {
      case DiTRequestKind::kImage:
        data = item.image;
        break;
      case DiTRequestKind::kVideo:
        data = item.video;
        break;
      case DiTRequestKind::kAudio:
        data = item.audio;
        break;
      case DiTRequestKind::kText:
        data = item.text;
        break;
    }
    if (!write_output(path, data)) {
      return 1;
    }
  }
  return 0;
}

}  // namespace xllm
