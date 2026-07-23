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

#include "core/util/verbose_trace_logger.h"

#include <glog/logging.h>
#include <spdlog/async_logger.h>
#include <spdlog/details/thread_pool.h>
#include <spdlog/sinks/rotating_file_sink.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>

namespace xllm {

namespace {

// Background queue capacity (entries) and writer-thread count for the async
// logger. One writer keeps file output strictly ordered.
constexpr size_t kQueueSize = 8192;
constexpr size_t kWriterThreads = 1;

constexpr int64_t kBytesPerMiB = 1024 * 1024;

}  // namespace

std::string resolve_verbose_trace_log_path(const std::string& configured_path,
                                           int32_t nnodes,
                                           int32_t node_rank) {
  if (configured_path.empty() || nnodes <= 1) {
    return configured_path;
  }

  const std::filesystem::path path(configured_path);
  const std::string extension = path.extension().string();
  const size_t extension_position = configured_path.size() - extension.size();
  std::string resolved_path = configured_path;
  resolved_path.insert(extension_position,
                       "_rank_" + std::to_string(node_rank));
  return resolved_path;
}

VerboseTraceLogger& VerboseTraceLogger::get_instance() {
  static VerboseTraceLogger instance;
  return instance;
}

VerboseTraceLogger::~VerboseTraceLogger() { shutdown(); }

void VerboseTraceLogger::initialize(bool enabled,
                                    const std::string& file_path,
                                    int32_t max_size_mb,
                                    int32_t max_files) {
  if (!enabled) {
    LOG(INFO) << "Verbose trace logging is disabled.";
    return;
  }
  if (enabled_.load(std::memory_order_relaxed)) {
    LOG(WARNING) << "Verbose trace logger already initialized; ignoring.";
    return;
  }
  if (file_path.empty()) {
    LOG(ERROR) << "Verbose trace logging enabled but no file path provided; "
                  "verbose trace logging stays disabled.";
    return;
  }

  std::error_code ec;
  const std::filesystem::path path(file_path);
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
      LOG(ERROR) << "Failed to create directory for verbose trace log "
                 << file_path << ": " << ec.message()
                 << "; verbose trace logging stays disabled.";
      return;
    }
  }

  // Guard against non-positive configuration that spdlog would reject.
  const size_t max_size_bytes =
      static_cast<size_t>(std::max(1, max_size_mb)) * kBytesPerMiB;
  const size_t total_files = static_cast<size_t>(std::max(1, max_files));
  // spdlog's max_files parameter counts rotated backups and excludes the
  // active file. The service option is the total on-disk file limit.
  const size_t backup_files = total_files - 1;

  try {
    thread_pool_ = std::make_shared<spdlog::details::thread_pool>(
        kQueueSize, kWriterThreads);
    auto sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        file_path, max_size_bytes, backup_files);
    logger_ = std::make_shared<spdlog::async_logger>(
        "verbose_trace",
        std::move(sink),
        thread_pool_,
        // Never block the serving hot path: drop the oldest queued entry if the
        // background writer cannot keep up.
        spdlog::async_overflow_policy::overrun_oldest);
    // Prefix each entry with a millisecond wall-clock timestamp; the entry body
    // is emitted verbatim and spdlog appends the trailing newline.
    logger_->set_pattern("%Y-%m-%d %H:%M:%S.%e %v");
    logger_->set_level(spdlog::level::info);
    // Flush each entry so a live tail and post-crash inspection see it
    // promptly; this runs on the background thread, off the serving hot path.
    logger_->flush_on(spdlog::level::info);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to open verbose trace log file " << file_path << ": "
               << e.what() << "; verbose trace logging stays disabled.";
    logger_.reset();
    thread_pool_.reset();
    return;
  }

  enabled_.store(true, std::memory_order_release);
  LOG(INFO) << "Verbose trace logging enabled, writing to " << file_path
            << " (rotating at " << max_size_mb << " MiB, keeping up to "
            << total_files << " files).";
}

void VerboseTraceLogger::log(const std::string& entry) {
  if (!enabled_.load(std::memory_order_relaxed)) {
    return;
  }
  // spdlog copies the message into its async ring buffer internally, so passing
  // by const-ref is safe (the thread-local source buffer can be reused
  // immediately after this returns).
  logger_->info(entry);
}

void VerboseTraceLogger::shutdown() {
  if (!enabled_.exchange(false, std::memory_order_acq_rel)) {
    return;
  }
  if (logger_) {
    logger_->flush();
    logger_.reset();
  }
  // Destroying the thread pool drains any queued entries, then joins the
  // background writer.
  thread_pool_.reset();
}

}  // namespace xllm
