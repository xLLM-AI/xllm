/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/options.h"
#include "common/rate_limiter.h"
#include "core/framework/multimodal/mm_data.h"
#include "framework/model/model_args.h"
#include "framework/request/rec_type.h"
#include "framework/request/request.h"
#include "framework/request/request_output.h"
#include "framework/request/request_params.h"
#include "rec.pb.h"
#include "util/rec_model_utils.h"

namespace xllm {

class Tokenizer;

// Factory that assembles a domain Request aggregate for the Rec path from raw
// user input (a text prompt, pre-tokenized prompt, OneRec input tensors, or
// LlmRec prompt tokens plus multimodal embedding data) and RequestParams.
//
// Request construction is delegated to a request-builder strategy selected
// from the model kind at construction time: OneRec (prefill-only or
// xattention) and LlmRec (with or without multimodal embedding) each build
// requests differently. The factory owns the builder instances and the shared
// build_request_common step that assembles sampling/stopping parameters and
// the final Request aggregate.
//
// The factory holds non-owning references to model configuration owned by
// RecMaster (model args, tokenizer, options, rate limiter); the owner must
// outlive the factory. The tokenizer may be null for OneRec models.
class RecRequestFactory final {
 public:
  RecRequestFactory(const ModelArgs* model_args,
                    Tokenizer* tokenizer,
                    const Options* options,
                    RateLimiter* rate_limiter,
                    RecType rec_type,
                    RecPipelineType pipeline_type);

  // prompt / prompt_tokens / input_tensors (OneRec and LlmRec without mm_data).
  std::shared_ptr<Request> create(
      std::string prompt,
      std::optional<std::vector<int>> prompt_tokens,
      std::optional<std::vector<proto::InferInputTensor>> input_tensors,
      const RequestParams& sp,
      OutputCallback callback);

  // raw prompt_tokens plus multimodal embedding data (LlmRec).
  std::shared_ptr<Request> create(const std::vector<int>& prompt_tokens,
                                  std::optional<MMData> mm_data,
                                  const RequestParams& sp,
                                  OutputCallback callback);

 private:
  // ============================================================
  // RequestBuilder: strategy base for request construction.
  // ============================================================
  class RequestBuilder {
   public:
    explicit RequestBuilder(RecRequestFactory& factory) : factory_(factory) {}
    virtual ~RequestBuilder() = default;

    // For prompt-based input (OneRec and LlmRec without mm_data).
    virtual std::shared_ptr<Request> generate_request(
        std::string prompt,
        std::optional<std::vector<int>> prompt_tokens,
        std::optional<std::vector<proto::InferInputTensor>> input_tensors,
        const RequestParams& sp,
        OutputCallback callback);

    // For raw input (LlmRec with mm_data).
    virtual std::shared_ptr<Request> generate_request(
        const std::vector<int>& prompt_tokens,
        std::optional<MMData> mm_data,
        const RequestParams& sp,
        OutputCallback callback);

   protected:
    std::shared_ptr<Request> generate_onerec_request_common(
        std::string prompt,
        std::optional<std::vector<int>> prompt_tokens,
        std::optional<std::vector<proto::InferInputTensor>> input_tensors,
        const RequestParams& sp,
        OutputCallback callback,
        bool build_stop_checker);

    RecRequestFactory& factory_;
  };

  // LlmRecRequestBuilder - pure qwen3 (prompt-based, no mm_data).
  class LlmRecRequestBuilder final : public RequestBuilder {
   public:
    explicit LlmRecRequestBuilder(RecRequestFactory& factory)
        : RequestBuilder(factory) {}
    std::shared_ptr<Request> generate_request(
        std::string prompt,
        std::optional<std::vector<int>> prompt_tokens,
        std::optional<std::vector<proto::InferInputTensor>> input_tensors,
        const RequestParams& sp,
        OutputCallback callback) override;
  };

  // LlmRecWithMmDataRequestBuilder - qwen3 with embedding (raw input).
  class LlmRecWithMmDataRequestBuilder final : public RequestBuilder {
   public:
    explicit LlmRecWithMmDataRequestBuilder(RecRequestFactory& factory)
        : RequestBuilder(factory) {}
    std::shared_ptr<Request> generate_request(
        const std::vector<int>& prompt_tokens,
        std::optional<MMData> mm_data,
        const RequestParams& sp,
        OutputCallback callback) override;
  };

  // OneRecPrefillOnlyRequestBuilder - legacy OneRec without stop checker.
  class OneRecPrefillOnlyRequestBuilder final : public RequestBuilder {
   public:
    explicit OneRecPrefillOnlyRequestBuilder(RecRequestFactory& factory)
        : RequestBuilder(factory) {}
    std::shared_ptr<Request> generate_request(
        std::string prompt,
        std::optional<std::vector<int>> prompt_tokens,
        std::optional<std::vector<proto::InferInputTensor>> input_tensors,
        const RequestParams& sp,
        OutputCallback callback) override;
  };

  // OneRecXAttentionRequestBuilder - OneRec xattention with stop checker for
  // device-side multi-round decode.
  class OneRecXAttentionRequestBuilder final : public RequestBuilder {
   public:
    explicit OneRecXAttentionRequestBuilder(RecRequestFactory& factory)
        : RequestBuilder(factory) {}
    std::shared_ptr<Request> generate_request(
        std::string prompt,
        std::optional<std::vector<int>> prompt_tokens,
        std::optional<std::vector<proto::InferInputTensor>> input_tensors,
        const RequestParams& sp,
        OutputCallback callback) override;
  };

  // Factory method to create a request builder (can access the private
  // builder classes).
  static std::unique_ptr<RequestBuilder> create_request_builder(
      RecPipelineType type,
      RecRequestFactory& factory);

  std::shared_ptr<Request> build_request_common(
      std::string prompt,
      std::vector<int32_t> prompt_tokens,
      MMData mm_data,
      const RequestParams& sp,
      OutputCallback callback,
      bool build_stop_checker);

  const ModelArgs* model_args_ = nullptr;
  Tokenizer* tokenizer_ = nullptr;
  const Options* options_ = nullptr;
  RateLimiter* rate_limiter_ = nullptr;
  RecType rec_type_ = RecType::kNone;

  // Request builder strategies (created from the model kind at construction
  // time).
  std::unique_ptr<RequestBuilder> request_builder_;
  std::unique_ptr<RequestBuilder> mm_data_request_builder_;
};

}  // namespace xllm
