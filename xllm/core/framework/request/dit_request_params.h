#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "audio_generation.pb.h"
#include "dit_request_output.h"
#include "dit_request_state.h"
#include "image_generation.pb.h"
#include "request.h"
#include "tensor.pb.h"
#include "text_generation.pb.h"
#include "video_generation.pb.h"
namespace xllm {

struct DiTRequestParams {
  DiTRequestParams() = default;
  DiTRequestParams(const proto::ImageGenerationRequest& request,
                   const std::string& x_rid,
                   const std::string& x_rtime);
  DiTRequestParams(const proto::AudioGenerationRequest& request,
                   const std::string& x_rid,
                   const std::string& x_rtime);
  DiTRequestParams(const proto::VideoGenerationRequest& request,
                   const std::string& x_rid,
                   const std::string& x_rtime);
  DiTRequestParams(const proto::TextGenerationRequest& request,
                   const std::string& x_rid,
                   const std::string& x_rtime);

  // When neither the x-request-id header nor the request body supplied a client
  // request id, fall back to `fallback` (the Call's generated trace id) so the
  // logs, the verbose trace and the echoed response header all share one id.
  // No-op when an id is already present.
  void set_x_request_id_if_absent(const std::string& fallback) {
    if (x_request_id.empty()) {
      x_request_id = fallback;
    }
  }

  bool verify_params(DiTOutputCallback callback) const;

  // request id
  std::string request_id;
  std::string x_request_id;
  std::string x_request_time;

  std::string model;

  DiTRequestKind request_kind = DiTRequestKind::kImage;

  DiTInputParams input_params;
  // Mandatory: Generation control parameters (encapsulates all fields related
  // to "image generation process")
  DiTGenerationParams generation_params;
};

}  // namespace xllm
