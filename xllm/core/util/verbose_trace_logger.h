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

#include <atomic>
#include <charconv>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace spdlog {
class logger;
namespace details {
class thread_pool;
}  // namespace details
}  // namespace spdlog

namespace xllm {

[[nodiscard]] std::string resolve_verbose_trace_log_path(
    const std::string& configured_path,
    int32_t nnodes,
    int32_t node_rank);

// Asynchronous request-trace logger.
//
// When enabled, serving threads format an entry into a thread-local buffer and
// hand it to spdlog's async queue with a single non-blocking memcpy (no file
// I/O, no heap allocation after the first call per thread). A dedicated
// background thread drains the queue and appends entries to a rotating file.
// When disabled, the only cost is one relaxed atomic load.
class VerboseTraceLogger final {
 public:
  static VerboseTraceLogger& get_instance();

  VerboseTraceLogger(const VerboseTraceLogger&) = delete;
  VerboseTraceLogger& operator=(const VerboseTraceLogger&) = delete;

  void initialize(bool enabled,
                  const std::string& file_path,
                  int32_t max_size_mb,
                  int32_t max_files);

  bool enabled() const { return enabled_.load(std::memory_order_relaxed); }

  void log(const std::string& entry);

  void shutdown();

 private:
  VerboseTraceLogger() = default;
  ~VerboseTraceLogger();

  std::atomic<bool> enabled_{false};
  std::shared_ptr<spdlog::details::thread_pool> thread_pool_;
  std::shared_ptr<spdlog::logger> logger_;
};

namespace detail {

// Lightweight message builder that accumulates into a thread-local pre-
// allocated string buffer. Eliminates std::ostringstream overhead (no locale,
// no virtual dispatch, no per-call heap allocation after the first invocation
// on each thread). On destruction the buffer content is handed to spdlog's
// async queue (one memcpy into the ring buffer).
class VerboseTraceMessage final {
 public:
  VerboseTraceMessage() { buf().clear(); }

  VerboseTraceMessage& operator<<(std::string_view val) {
    buf().append(val);
    return *this;
  }

  VerboseTraceMessage& operator<<(int32_t val) {
    char tmp[12];
    auto [ptr, ec] = std::to_chars(tmp, tmp + sizeof(tmp), val);
    buf().append(tmp, static_cast<size_t>(ptr - tmp));
    return *this;
  }

  VerboseTraceMessage& operator<<(int64_t val) {
    char tmp[24];
    auto [ptr, ec] = std::to_chars(tmp, tmp + sizeof(tmp), val);
    buf().append(tmp, static_cast<size_t>(ptr - tmp));
    return *this;
  }

  VerboseTraceMessage& operator<<(uint64_t val) {
    char tmp[24];
    auto [ptr, ec] = std::to_chars(tmp, tmp + sizeof(tmp), val);
    buf().append(tmp, static_cast<size_t>(ptr - tmp));
    return *this;
  }

  ~VerboseTraceMessage() { VerboseTraceLogger::get_instance().log(buf()); }

 private:
  static constexpr size_t kDefaultCapacity = 512;

  static std::string& buf() {
    thread_local std::string tl_buf = [] {
      std::string s;
      s.reserve(kDefaultCapacity);
      return s;
    }();
    return tl_buf;
  }
};

class VerboseTraceVoidify final {
 public:
  void operator&(const VerboseTraceMessage&) {}
};

}  // namespace detail

#define XLLM_VERBOSE_TRACE()                            \
  !::xllm::VerboseTraceLogger::get_instance().enabled() \
      ? (void)0                                         \
      : ::xllm::detail::VerboseTraceVoidify() &         \
            ::xllm::detail::VerboseTraceMessage()

}  // namespace xllm
