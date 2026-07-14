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

#include <cstdint>
#include <mutex>
#include <optional>

namespace xllm {

// Thread-safe singleton storing the validate time predictor coefficients
// fitted by ProfileManager. Used by AdaptiveSpeculativeController to estimate
// per-seq validate cost. Also serves as the gate for enabling the adaptive
// path: adaptive decode only activates after the predictor is set.
class SpeculativeProfileRegistry final {
 public:
  struct ValidateTimePredictor {
    double intercept_ms = 0.0;
    double batch_ms = 0.0;
    double query_token_ms = 0.0;
    double query_prefix_ms = 0.0;
  };

  static SpeculativeProfileRegistry& get_instance();

  void set_validate_time_predictor(const ValidateTimePredictor& predictor);
  void reset_validate_time_predictor();

  bool has_validate_time_predictor() const;
  std::optional<ValidateTimePredictor> validate_time_predictor() const;

  double predict_validate_time_ms(int32_t batch_size,
                                  int32_t query_len,
                                  double avg_prefix_len) const;

 private:
  SpeculativeProfileRegistry() = default;

  mutable std::mutex mutex_;
  std::optional<ValidateTimePredictor> validate_time_predictor_;
};

}  // namespace xllm
