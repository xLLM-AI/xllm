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

#include "core/framework/speculative/speculative_profile_registry.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <mutex>

namespace xllm {
namespace {

double sanitize_non_negative(double value) {
  if (!std::isfinite(value)) {
    return 0.0;
  }
  return std::max(value, 0.0);
}

}  // namespace

SpeculativeProfileRegistry& SpeculativeProfileRegistry::get_instance() {
  static SpeculativeProfileRegistry registry;
  return registry;
}

void SpeculativeProfileRegistry::set_validate_time_predictor(
    const ValidateTimePredictor& predictor) {
  ValidateTimePredictor sanitized_predictor = predictor;
  sanitized_predictor.intercept_ms =
      sanitize_non_negative(predictor.intercept_ms);
  sanitized_predictor.batch_ms = sanitize_non_negative(predictor.batch_ms);
  sanitized_predictor.query_token_ms =
      sanitize_non_negative(predictor.query_token_ms);
  sanitized_predictor.query_prefix_ms =
      sanitize_non_negative(predictor.query_prefix_ms);
  std::lock_guard<std::mutex> lock(mutex_);
  validate_time_predictor_ = sanitized_predictor;
}

void SpeculativeProfileRegistry::reset_validate_time_predictor() {
  std::lock_guard<std::mutex> lock(mutex_);
  validate_time_predictor_.reset();
}

bool SpeculativeProfileRegistry::has_validate_time_predictor() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return validate_time_predictor_.has_value();
}

std::optional<SpeculativeProfileRegistry::ValidateTimePredictor>
SpeculativeProfileRegistry::validate_time_predictor() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return validate_time_predictor_;
}

double SpeculativeProfileRegistry::predict_validate_time_ms(
    int32_t batch_size,
    int32_t query_len,
    double avg_prefix_len) const {
  std::optional<ValidateTimePredictor> predictor = validate_time_predictor();
  if (!predictor.has_value() || batch_size <= 0 || query_len <= 0 ||
      !std::isfinite(avg_prefix_len)) {
    return 0.0;
  }

  const double batch = static_cast<double>(batch_size);
  const double query = static_cast<double>(query_len);
  const double prefix = std::max(avg_prefix_len, 0.0);
  const double predicted_time =
      predictor->intercept_ms + predictor->batch_ms * batch +
      predictor->query_token_ms * batch * query +
      predictor->query_prefix_ms * batch * query * prefix;
  return sanitize_non_negative(predicted_time);
}

}  // namespace xllm
