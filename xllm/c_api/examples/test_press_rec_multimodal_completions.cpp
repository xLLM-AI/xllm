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

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rec.h"

namespace {

using Clock = std::chrono::steady_clock;

constexpr int32_t kFixedRequestMaxTokens = 3;
constexpr int32_t kFixedRequestBeamWidth = 128;
constexpr int32_t kFixedRequestTopK = 128;
constexpr int32_t kFixedRequestTopLogprobs = 128;
constexpr bool kFixedRequestLogprobs = true;
constexpr int32_t kFixedMmItemsPerRequest = 2;
constexpr int32_t kMaxClientParallelism = 128;

struct CliOptions {
  std::string model_path;
  std::string master_node_addr = "127.0.0.1:18899";
  int32_t prompt_size = 128;
  int32_t token_min_size = 1024;
  int32_t token_max_size = 1024;
  double qps = 1.0;
  int32_t duration_s = 60;
  int32_t client_threads = 0;
  int32_t timeout_ms = 30000;
  int32_t mm_min_span = 8;
  int32_t mm_max_span = 64;
  uint32_t seed = 20260410U;
};

struct ModelConfig {
  int32_t hidden_size = 0;
  int32_t vocab_size = 0;
  int32_t max_position_embeddings = 0;
  std::string model_type;
};

struct Metrics {
  std::atomic<uint64_t> sent{0};
  std::atomic<uint64_t> succeeded{0};
  std::atomic<uint64_t> failed{0};
  std::atomic<uint64_t> timeout{0};
  std::atomic<uint64_t> invalid_request{0};
  std::atomic<uint64_t> internal_error{0};
  std::atomic<uint64_t> total_prompt_tokens{0};
  std::atomic<uint64_t> total_completion_tokens{0};
  std::atomic<uint64_t> total_latency_us{0};
  std::atomic<uint64_t> max_latency_us{0};
  std::atomic<uint64_t> current_in_flight{0};
  std::atomic<uint64_t> max_in_flight{0};
};

struct RequestPayload {
  std::vector<int32_t> token_ids;
};

struct ScheduledRequest {
  uint64_t request_index = 0;
  size_t pool_index = 0;
};

class EmbeddingMmDataBuilder {
 public:
  EmbeddingMmDataBuilder() = default;

  const XLLM_MM_Data* Build(
      const std::vector<std::pair<uint32_t, uint32_t>>& spans,
      int32_t hidden_size,
      uint64_t request_index,
      std::mt19937& rng) {
    Reset();

    if (spans.empty()) {
      return nullptr;
    }

    mm_data_.type_mask = static_cast<uint32_t>(XLLM_MM_TYPE_EMBEDDING);
    mm_data_.is_dict = false;

    items_.reserve(spans.size());
    buffers_.reserve(spans.size());

    for (size_t item_idx = 0; item_idx < spans.size(); ++item_idx) {
      const auto [offset, length] = spans[item_idx];
      if (length == 0) {
        continue;
      }

      XLLM_MM_Item item{};
      item.type = XLLM_MM_TYPE_EMBEDDING;
      item.state.token_pos.offset = offset;
      item.state.token_pos.length = length;
      item.data.is_single_tensor = true;
      item.data.data.tensor.dtype = XLLM_DTYPE_BFLOAT16;
      item.data.data.tensor.dims.rank = 2;
      item.data.data.tensor.dims.dim[0] = static_cast<int>(length);
      item.data.data.tensor.dims.dim[1] = hidden_size;

      auto buffer = std::make_unique<uint16_t[]>(
          static_cast<size_t>(length) * static_cast<size_t>(hidden_size));
      FillEmbeddingBuffer(
          buffer.get(), length, hidden_size, request_index, item_idx, rng);
      item.data.data.tensor.data = buffer.get();

      buffers_.push_back(std::move(buffer));
      items_.push_back(item);
    }

    if (items_.empty()) {
      return nullptr;
    }

    mm_data_.data.items.entries = items_.data();
    mm_data_.data.items.entries_size = items_.size();
    return &mm_data_;
  }

 private:
  static uint16_t FloatToBFloat16(float value) {
    union {
      float f32;
      uint32_t u32;
    } bits;
    bits.f32 = value;
    return static_cast<uint16_t>(bits.u32 >> 16);
  }

  void FillEmbeddingBuffer(uint16_t* dst,
                           uint32_t length,
                           int32_t hidden_size,
                           uint64_t request_index,
                           size_t item_idx,
                           std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const size_t element_count =
        static_cast<size_t>(length) * static_cast<size_t>(hidden_size);
    for (size_t element_idx = 0; element_idx < element_count; ++element_idx) {
      const float noise = dist(rng) * 0.03125f;
      const float base =
          std::sin(static_cast<float>((request_index + 1) * 0.013) +
                   static_cast<float>(item_idx) * 0.17f +
                   static_cast<float>(element_idx % hidden_size) * 0.001f);
      dst[element_idx] = FloatToBFloat16(base + noise);
    }
  }

  void Reset() {
    std::memset(&mm_data_, 0, sizeof(mm_data_));
    items_.clear();
    buffers_.clear();
  }

  XLLM_MM_Data mm_data_{};
  std::vector<XLLM_MM_Item> items_;
  std::vector<std::unique_ptr<uint16_t[]>> buffers_;
};

std::string Trim(const std::string& value) {
  const auto begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  const auto end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

std::string RemoveJsonComments(std::string content) {
  std::string output;
  output.reserve(content.size());

  bool in_string = false;
  bool escape = false;
  for (size_t i = 0; i < content.size(); ++i) {
    const char ch = content[i];
    if (in_string) {
      output.push_back(ch);
      if (escape) {
        escape = false;
      } else if (ch == '\\') {
        escape = true;
      } else if (ch == '"') {
        in_string = false;
      }
      continue;
    }

    if (ch == '"') {
      in_string = true;
      output.push_back(ch);
      continue;
    }

    if (ch == '/' && i + 1 < content.size()) {
      if (content[i + 1] == '/') {
        i += 2;
        while (i < content.size() && content[i] != '\n') {
          ++i;
        }
        if (i < content.size()) {
          output.push_back('\n');
        }
        continue;
      }
      if (content[i + 1] == '*') {
        i += 2;
        while (i + 1 < content.size() &&
               !(content[i] == '*' && content[i + 1] == '/')) {
          ++i;
        }
        ++i;
        continue;
      }
    }

    output.push_back(ch);
  }

  return output;
}

const std::string* FindObjectRange(const std::string& content,
                                   const std::string& key,
                                   size_t* object_begin,
                                   size_t* object_end) {
  const std::string pattern = "\"" + key + "\"";
  const size_t key_pos = content.find(pattern);
  if (key_pos == std::string::npos) {
    return nullptr;
  }

  size_t pos = content.find(':', key_pos + pattern.size());
  if (pos == std::string::npos) {
    return nullptr;
  }
  ++pos;
  while (pos < content.size() &&
         std::isspace(static_cast<unsigned char>(content[pos]))) {
    ++pos;
  }
  if (pos >= content.size() || content[pos] != '{') {
    return nullptr;
  }

  const size_t begin = pos;
  int depth = 0;
  bool in_string = false;
  bool escape = false;
  for (; pos < content.size(); ++pos) {
    const char ch = content[pos];
    if (in_string) {
      if (escape) {
        escape = false;
      } else if (ch == '\\') {
        escape = true;
      } else if (ch == '"') {
        in_string = false;
      }
      continue;
    }

    if (ch == '"') {
      in_string = true;
      continue;
    }
    if (ch == '{') {
      ++depth;
    } else if (ch == '}') {
      --depth;
      if (depth == 0) {
        *object_begin = begin;
        *object_end = pos + 1;
        return &content;
      }
    }
  }
  return nullptr;
}

std::optional<std::string> FindJsonStringValueInRange(
    const std::string& content,
    size_t begin,
    size_t end,
    const std::string& key) {
  const std::string pattern = "\"" + key + "\"";
  size_t pos = content.find(pattern, begin);
  if (pos == std::string::npos || pos >= end) {
    return std::nullopt;
  }

  pos = content.find(':', pos + pattern.size());
  if (pos == std::string::npos || pos >= end) {
    return std::nullopt;
  }
  ++pos;
  while (pos < end && std::isspace(static_cast<unsigned char>(content[pos]))) {
    ++pos;
  }
  if (pos >= end || content[pos] != '"') {
    return std::nullopt;
  }

  ++pos;
  std::string value;
  bool escape = false;
  while (pos < end) {
    const char ch = content[pos++];
    if (escape) {
      value.push_back(ch);
      escape = false;
      continue;
    }
    if (ch == '\\') {
      escape = true;
      continue;
    }
    if (ch == '"') {
      return value;
    }
    value.push_back(ch);
  }
  return std::nullopt;
}

std::optional<int32_t> FindJsonIntValueInRange(const std::string& content,
                                               size_t begin,
                                               size_t end,
                                               const std::string& key) {
  const std::string pattern = "\"" + key + "\"";
  size_t pos = content.find(pattern, begin);
  if (pos == std::string::npos || pos >= end) {
    return std::nullopt;
  }

  pos = content.find(':', pos + pattern.size());
  if (pos == std::string::npos || pos >= end) {
    return std::nullopt;
  }
  ++pos;
  while (pos < end && std::isspace(static_cast<unsigned char>(content[pos]))) {
    ++pos;
  }
  if (pos >= end) {
    return std::nullopt;
  }

  size_t value_end = pos;
  if (content[value_end] == '-') {
    ++value_end;
  }
  while (value_end < end &&
         std::isdigit(static_cast<unsigned char>(content[value_end]))) {
    ++value_end;
  }
  if (value_end == pos || (value_end == pos + 1 && content[pos] == '-')) {
    return std::nullopt;
  }

  return std::stoi(content.substr(pos, value_end - pos));
}

ModelConfig LoadModelConfig(const std::string& model_path) {
  const std::filesystem::path config_path =
      std::filesystem::path(model_path) / "config.json";
  std::ifstream ifs(config_path);
  if (!ifs.is_open()) {
    throw std::runtime_error("failed to open model config: " +
                             config_path.string());
  }

  std::stringstream buffer;
  buffer << ifs.rdbuf();
  const std::string content = RemoveJsonComments(buffer.str());

  ModelConfig cfg;
  cfg.hidden_size =
      FindJsonIntValueInRange(content, 0, content.size(), "hidden_size")
          .value_or(0);
  cfg.vocab_size =
      FindJsonIntValueInRange(content, 0, content.size(), "vocab_size")
          .value_or(0);
  cfg.max_position_embeddings =
      FindJsonIntValueInRange(
          content, 0, content.size(), "max_position_embeddings")
          .value_or(0);
  cfg.model_type =
      FindJsonStringValueInRange(content, 0, content.size(), "model_type")
          .value_or(FindJsonStringValueInRange(
                        content, 0, content.size(), "model_name")
                        .value_or(""));

  size_t text_begin = 0;
  size_t text_end = 0;
  if (FindObjectRange(content, "text_config", &text_begin, &text_end) !=
      nullptr) {
    if (cfg.hidden_size <= 0) {
      cfg.hidden_size =
          FindJsonIntValueInRange(content, text_begin, text_end, "hidden_size")
              .value_or(0);
    }
    if (cfg.vocab_size <= 0) {
      cfg.vocab_size =
          FindJsonIntValueInRange(content, text_begin, text_end, "vocab_size")
              .value_or(0);
    }
    if (cfg.max_position_embeddings <= 0) {
      cfg.max_position_embeddings =
          FindJsonIntValueInRange(
              content, text_begin, text_end, "max_position_embeddings")
              .value_or(0);
    }
  }

  if (cfg.hidden_size <= 0) {
    throw std::runtime_error(
        "config.json missing hidden_size/text_config.hidden_size");
  }
  if (cfg.vocab_size <= 0) {
    throw std::runtime_error(
        "config.json missing vocab_size/text_config.vocab_size");
  }
  if (cfg.max_position_embeddings <= 0) {
    throw std::runtime_error(
        "config.json missing "
        "max_position_embeddings/text_config.max_position_embeddings");
  }

  return cfg;
}

std::string ResolveModelId(const std::string& model_path) {
  std::filesystem::path path =
      std::filesystem::path(model_path).lexically_normal();
  if (path.has_filename()) {
    return path.filename().string();
  }
  return path.string();
}

void PrintUsage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " --model_path PATH [options]\n"
      << "Options:\n"
      << "  --master_node_addr STR   default: 127.0.0.1:18899\n"
      << "  --prompt_size N          number of requests kept in pool, default: "
         "128\n"
      << "  --token_min_size N       minimum prompt token length, default: "
         "1024\n"
      << "  --token_max_size N       maximum prompt token length, default: "
         "1024\n"
      << "  --qps FLOAT              target QPS, default: 1.0\n"
      << "  --client_threads N       concurrent client request threads; 0 "
         "means auto (ceil(qps))\n"
      << "  --duration_s N           duration in seconds, default: 60; 0 "
         "means run forever\n"
      << "  --mm_min_span N          default: 8\n"
      << "  --mm_max_span N          default: 64\n"
      << "  --seed N                 default: 20260410\n";
}

CliOptions ParseArgs(int argc, char** argv) {
  CliOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };

    if (arg == "--model_path") {
      options.model_path = next("--model_path");
    } else if (arg == "--master_node_addr") {
      options.master_node_addr = next("--master_node_addr");
    } else if (arg == "--prompt_size") {
      options.prompt_size = std::stoi(next("--prompt_size"));
    } else if (arg == "--token_min_size") {
      options.token_min_size = std::stoi(next("--token_min_size"));
    } else if (arg == "--token_max_size") {
      options.token_max_size = std::stoi(next("--token_max_size"));
    } else if (arg == "--qps") {
      options.qps = std::stod(next("--qps"));
    } else if (arg == "--client_threads") {
      options.client_threads = std::stoi(next("--client_threads"));
    } else if (arg == "--duration_s") {
      options.duration_s = std::stoi(next("--duration_s"));
    } else if (arg == "--mm_min_span") {
      options.mm_min_span = std::stoi(next("--mm_min_span"));
    } else if (arg == "--mm_max_span") {
      options.mm_max_span = std::stoi(next("--mm_max_span"));
    } else if (arg == "--seed") {
      options.seed = static_cast<uint32_t>(std::stoul(next("--seed")));
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (options.model_path.empty()) {
    throw std::runtime_error("--model_path is required");
  }
  if (options.prompt_size <= 0) {
    throw std::runtime_error("--prompt_size must be > 0");
  }
  if (options.token_min_size <= 0 || options.token_max_size <= 0) {
    throw std::runtime_error(
        "--token_min_size and --token_max_size must be > 0");
  }
  if (options.token_min_size > options.token_max_size) {
    throw std::runtime_error("--token_min_size must be <= --token_max_size");
  }
  if (options.qps <= 0.0) {
    throw std::runtime_error("--qps must be > 0");
  }
  if (options.client_threads < 0) {
    throw std::runtime_error("--client_threads must be >= 0");
  }
  if (options.duration_s < 0) {
    throw std::runtime_error("--duration_s must be >= 0");
  }
  if (options.mm_min_span <= 0 || options.mm_max_span <= 0 ||
      options.mm_min_span > options.mm_max_span) {
    throw std::runtime_error("invalid mm span range");
  }

  return options;
}

std::vector<int32_t> MakeRandomTokens(int32_t token_size,
                                      int32_t vocab_size,
                                      std::mt19937& rng) {
  std::uniform_int_distribution<int32_t> dist(1, std::max(2, vocab_size - 1));
  std::vector<int32_t> token_ids(token_size);
  for (int32_t& token_id : token_ids) {
    token_id = dist(rng);
  }
  return token_ids;
}

std::vector<std::pair<uint32_t, uint32_t>> MakeRandomMmSpans(
    int32_t token_size,
    const CliOptions& options,
    std::mt19937& rng) {
  std::vector<std::pair<uint32_t, uint32_t>> spans;
  spans.reserve(kFixedMmItemsPerRequest);

  const int32_t max_valid_end = token_size - 1;
  int32_t cursor = 1;
  for (int32_t item_idx = 0; item_idx < kFixedMmItemsPerRequest; ++item_idx) {
    const int32_t remaining_items = kFixedMmItemsPerRequest - item_idx;
    const int32_t remaining_tokens = max_valid_end - cursor;
    if (remaining_tokens <= options.mm_min_span) {
      break;
    }

    const int32_t max_span =
        std::min(options.mm_max_span, remaining_tokens - remaining_items + 1);
    if (max_span < options.mm_min_span) {
      break;
    }

    std::uniform_int_distribution<int32_t> span_dist(options.mm_min_span,
                                                     max_span);
    const int32_t length = span_dist(rng);

    const int32_t max_offset =
        max_valid_end - length - (remaining_items - 1) * options.mm_min_span;
    if (max_offset < cursor) {
      break;
    }
    std::uniform_int_distribution<int32_t> offset_dist(cursor, max_offset);
    const int32_t offset = offset_dist(rng);
    spans.emplace_back(static_cast<uint32_t>(offset),
                       static_cast<uint32_t>(length));
    cursor = offset + length + 1;
  }

  if (spans.empty()) {
    const uint32_t fallback_len =
        static_cast<uint32_t>(std::min(options.mm_min_span, token_size - 2));
    spans.emplace_back(1U, std::max<uint32_t>(1U, fallback_len));
  }

  return spans;
}

std::vector<RequestPayload> BuildRequestPool(const CliOptions& options,
                                             const ModelConfig& config) {
  std::mt19937 rng(options.seed);
  std::vector<RequestPayload> pool;
  pool.reserve(options.prompt_size);

  const int32_t max_model_token_size = config.max_position_embeddings - 1;
  const int32_t token_min_size =
      std::min(options.token_min_size, max_model_token_size);
  const int32_t token_max_size =
      std::min(options.token_max_size, max_model_token_size);
  if (token_min_size <= 1 || token_max_size <= 1) {
    throw std::runtime_error(
        "token_size range is too large for model max_position_embeddings");
  }
  if (token_min_size > token_max_size) {
    throw std::runtime_error(
        "token_size range becomes invalid after clamping to model limits");
  }
  std::uniform_int_distribution<int32_t> token_size_dist(token_min_size,
                                                         token_max_size);

  for (int32_t i = 0; i < options.prompt_size; ++i) {
    const int32_t token_size = token_size_dist(rng);
    RequestPayload payload;
    payload.token_ids = MakeRandomTokens(token_size, config.vocab_size, rng);
    pool.push_back(std::move(payload));
  }
  return pool;
}

void UpdateMax(std::atomic<uint64_t>& target, uint64_t value) {
  uint64_t prev = target.load(std::memory_order_relaxed);
  while (
      prev < value &&
      !target.compare_exchange_weak(
          prev, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
  }
}

void RecordResponseMetrics(const XLLM_Response* resp,
                           uint64_t latency_us,
                           Metrics* metrics) {
  metrics->sent.fetch_add(1, std::memory_order_relaxed);
  metrics->total_latency_us.fetch_add(latency_us, std::memory_order_relaxed);
  UpdateMax(metrics->max_latency_us, latency_us);

  if (resp == nullptr) {
    metrics->failed.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  metrics->total_prompt_tokens.fetch_add(
      static_cast<uint64_t>(std::max(resp->usage.prompt_tokens, 0)),
      std::memory_order_relaxed);
  metrics->total_completion_tokens.fetch_add(
      static_cast<uint64_t>(std::max(resp->usage.completion_tokens, 0)),
      std::memory_order_relaxed);

  if (resp->status_code == kSuccess) {
    metrics->succeeded.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  metrics->failed.fetch_add(1, std::memory_order_relaxed);
  if (resp->status_code == kTimeout) {
    metrics->timeout.fetch_add(1, std::memory_order_relaxed);
  } else if (resp->status_code == kInvalidRequest) {
    metrics->invalid_request.fetch_add(1, std::memory_order_relaxed);
  } else if (resp->status_code == kInternalError) {
    metrics->internal_error.fetch_add(1, std::memory_order_relaxed);
  }
}

void IncrementInFlight(Metrics* metrics) {
  const uint64_t current =
      metrics->current_in_flight.fetch_add(1, std::memory_order_relaxed) + 1;
  UpdateMax(metrics->max_in_flight, current);
}

void DecrementInFlight(Metrics* metrics) {
  metrics->current_in_flight.fetch_sub(1, std::memory_order_relaxed);
}

void PrintSummary(const CliOptions& options,
                  const ModelConfig& config,
                  const Metrics& metrics,
                  double actual_duration_s) {
  const uint64_t sent = metrics.sent.load(std::memory_order_relaxed);
  const uint64_t succeeded = metrics.succeeded.load(std::memory_order_relaxed);
  const uint64_t failed = metrics.failed.load(std::memory_order_relaxed);
  const uint64_t total_latency_us =
      metrics.total_latency_us.load(std::memory_order_relaxed);
  const uint64_t max_latency_us =
      metrics.max_latency_us.load(std::memory_order_relaxed);
  const uint64_t max_in_flight =
      metrics.max_in_flight.load(std::memory_order_relaxed);

  const double avg_latency_ms = sent == 0
                                    ? 0.0
                                    : static_cast<double>(total_latency_us) /
                                          static_cast<double>(sent) / 1000.0;
  const double actual_qps = actual_duration_s <= 0.0
                                ? 0.0
                                : static_cast<double>(sent) / actual_duration_s;

  std::cout << "=== stress_rec_multimodal_completions summary ===\n";
  std::cout << "model_id=" << ResolveModelId(options.model_path) << "\n";
  std::cout << "model_type=" << config.model_type << "\n";
  std::cout << "master_node_addr=" << options.master_node_addr << "\n";
  std::cout << "hidden_size=" << config.hidden_size
            << ", vocab_size=" << config.vocab_size
            << ", max_position_embeddings=" << config.max_position_embeddings
            << "\n";
  std::cout << "prompt_size=" << options.prompt_size << ", token_size_range=["
            << options.token_min_size << ", " << options.token_max_size << "]"
            << ", qps_target=" << options.qps
            << ", duration_s=" << options.duration_s << "\n";
  std::cout << "client_threads="
            << (options.client_threads == 0
                    ? static_cast<int32_t>(std::ceil(options.qps))
                    : options.client_threads)
            << "\n";
  std::cout << "mm_span_range=[" << options.mm_min_span << ", "
            << options.mm_max_span
            << "], mm_items_per_request=" << kFixedMmItemsPerRequest << "\n";
  std::cout << "sent=" << sent << ", succeeded=" << succeeded
            << ", failed=" << failed << "\n";
  std::cout << "max_client_in_flight=" << max_in_flight << "\n";
  std::cout << "timeout=" << metrics.timeout.load(std::memory_order_relaxed)
            << ", invalid_request="
            << metrics.invalid_request.load(std::memory_order_relaxed)
            << ", internal_error="
            << metrics.internal_error.load(std::memory_order_relaxed) << "\n";
  std::cout << std::fixed << std::setprecision(3)
            << "actual_duration_s=" << actual_duration_s
            << ", actual_qps=" << actual_qps
            << ", avg_latency_ms=" << avg_latency_ms << ", max_latency_ms="
            << static_cast<double>(max_latency_us) / 1000.0 << "\n";
  std::cout << "prompt_tokens_total="
            << metrics.total_prompt_tokens.load(std::memory_order_relaxed)
            << ", completion_tokens_total="
            << metrics.total_completion_tokens.load(std::memory_order_relaxed)
            << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions options = ParseArgs(argc, argv);
    const ModelConfig config = LoadModelConfig(options.model_path);
    auto request_pool = BuildRequestPool(options, config);
    const std::string model_id = ResolveModelId(options.model_path);

    std::cout << "Loaded model config from " << options.model_path << "\n";
    std::cout << "hidden_size=" << config.hidden_size
              << ", vocab_size=" << config.vocab_size
              << ", max_position_embeddings=" << config.max_position_embeddings
              << ", model_type=" << config.model_type << "\n";
    std::cout << "model_id=" << model_id << "\n";

    XLLM_REC_Handler* rec_handler = xllm_rec_create();
    if (rec_handler == nullptr) {
      throw std::runtime_error("xllm_rec_create returned nullptr");
    }

    XLLM_InitOptions init_options;
    xllm_rec_init_options_default(&init_options);
    std::snprintf(init_options.master_node_addr,
                  sizeof(init_options.master_node_addr),
                  "%s",
                  options.master_node_addr.c_str());
    const bool init_ok = xllm_rec_initialize(
        rec_handler, options.model_path.c_str(), &init_options);
    if (!init_ok) {
      xllm_rec_destroy(rec_handler);
      throw std::runtime_error("xllm_rec_initialize failed");
    }

    XLLM_RequestParams request_params;
    xllm_rec_request_params_default(&request_params);
    request_params.max_tokens = kFixedRequestMaxTokens;
    request_params.beam_width = kFixedRequestBeamWidth;
    request_params.logprobs = kFixedRequestLogprobs;
    request_params.top_k = kFixedRequestTopK;
    request_params.top_logprobs = kFixedRequestTopLogprobs;

    std::cout << "Initialized REC with fixed request params: max_tokens="
              << request_params.max_tokens
              << ", beam_width=" << request_params.beam_width
              << ", logprobs=" << (request_params.logprobs ? 1 : 0)
              << ", top_k=" << request_params.top_k
              << ", top_logprobs=" << request_params.top_logprobs << "\n";

    Metrics metrics;
    const auto start_time = Clock::now();
    const bool run_forever = options.duration_s == 0;
    const auto stop_time =
        run_forever ? Clock::time_point::max()
                    : start_time + std::chrono::seconds(options.duration_s);
    const double interval_us = 1e6 / options.qps;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::deque<ScheduledRequest> queue;
    bool producer_done = false;

    const int32_t auto_worker_count =
        std::max<int32_t>(1, static_cast<int32_t>(std::ceil(options.qps)));
    const int32_t worker_count = std::max<int32_t>(
        1,
        std::min<int32_t>(kMaxClientParallelism,
                          options.client_threads > 0 ? options.client_threads
                                                     : auto_worker_count));

    std::cout << "Using client_threads=" << worker_count
              << (options.client_threads > 0 ? " (explicit)" : " (auto)")
              << "\n";

    auto worker_fn = [&](int32_t worker_id) {
      std::mt19937 rng(options.seed ^ 0x9e3779b9U ^
                       static_cast<uint32_t>(worker_id * 0x85ebca6bU));
      EmbeddingMmDataBuilder mm_builder;
      while (true) {
        ScheduledRequest scheduled;
        {
          std::unique_lock<std::mutex> lock(queue_mutex);
          queue_cv.wait(lock,
                        [&]() { return producer_done || !queue.empty(); });
          if (queue.empty()) {
            if (producer_done) {
              return;
            }
            continue;
          }
          scheduled = queue.front();
          queue.pop_front();
        }

        const RequestPayload& payload = request_pool[scheduled.pool_index];
        const auto mm_spans = MakeRandomMmSpans(
            static_cast<int32_t>(payload.token_ids.size()), options, rng);
        const XLLM_MM_Data* mm_data = mm_builder.Build(
            mm_spans, config.hidden_size, scheduled.request_index, rng);

        IncrementInFlight(&metrics);
        const auto req_begin = Clock::now();
        XLLM_Response* resp =
            xllm_rec_multimodal_completions(rec_handler,
                                            model_id.c_str(),
                                            payload.token_ids.data(),
                                            payload.token_ids.size(),
                                            mm_data,
                                            options.timeout_ms,
                                            &request_params);
        const auto req_end = Clock::now();
        DecrementInFlight(&metrics);

        const uint64_t latency_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(req_end -
                                                                  req_begin)
                .count());
        RecordResponseMetrics(resp, latency_us, &metrics);

        if (resp != nullptr && resp->status_code != kSuccess) {
          std::cerr << "request " << scheduled.request_index
                    << " failed: status=" << resp->status_code
                    << ", error=" << resp->error_info << "\n";
          std::cerr << "request " << scheduled.request_index
                    << " shape: token_size=" << payload.token_ids.size()
                    << ", mm_items=" << mm_spans.size();
          for (const auto& [offset, length] : mm_spans) {
            std::cerr << " [" << offset << "," << length << "]";
          }
          std::cerr << "\n";
        }
        xllm_rec_free_response(resp);
      }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(worker_count));
    for (int32_t worker_id = 0; worker_id < worker_count; ++worker_id) {
      workers.emplace_back(worker_fn, worker_id);
    }

    uint64_t request_index = 0;
    size_t next_pool_index = 0;
    while (Clock::now() < stop_time) {
      const auto scheduled_time =
          start_time + std::chrono::microseconds(
                           static_cast<int64_t>(request_index * interval_us));
      std::this_thread::sleep_until(scheduled_time);
      {
        std::lock_guard<std::mutex> lock(queue_mutex);
        queue.push_back(ScheduledRequest{request_index, next_pool_index});
      }
      queue_cv.notify_one();
      next_pool_index = (next_pool_index + 1) % request_pool.size();
      ++request_index;
    }

    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      producer_done = true;
    }
    queue_cv.notify_all();
    for (auto& worker : workers) {
      worker.join();
    }

    const double actual_duration_s =
        std::chrono::duration_cast<std::chrono::duration<double>>(Clock::now() -
                                                                  start_time)
            .count();
    PrintSummary(options, config, metrics, actual_duration_s);
    xllm_rec_destroy(rec_handler);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << "\n";
    return 1;
  }
}
