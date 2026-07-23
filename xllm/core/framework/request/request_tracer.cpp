/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "request_tracer.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <cctype>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "api_service/call.h"
#include "common/global_flags.h"
#include "finish_reason.h"
#include "request.h"

namespace xllm {

namespace {

std::string sanitize_filename(const std::string& name) {
  std::string out;
  out.reserve(name.size());
  for (char c : name) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' ||
        c == '.') {
      out.push_back(c);
    } else {
      out.push_back('_');
    }
  }
  if (out.empty()) {
    out = "unknown";
  }
  return out;
}

}  // namespace

RequestTracer& RequestTracer::get_instance() {
  static RequestTracer instance;
  return instance;
}

bool RequestTracer::enabled() const { return FLAGS_enable_request_trace; }

RequestTracer::RequestTracer() = default;

RequestTracer::~RequestTracer() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }
  queue_cv_.notify_one();
  if (writer_thread_.joinable()) {
    writer_thread_.join();
  }
  if (ofs_.is_open()) {
    ofs_.flush();
    ofs_.close();
  }
}

void RequestTracer::ensure_initialized() {
  std::call_once(init_flag_, [this] {
    per_file_ = FLAGS_request_trace_per_file;
    trace_path_ = FLAGS_request_trace_path;

    try {
      if (per_file_) {
        std::filesystem::create_directories(trace_path_);
      } else {
        std::filesystem::path file_path(trace_path_);
        if (file_path.has_parent_path()) {
          std::filesystem::create_directories(file_path.parent_path());
        }
      }
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to create directory for request trace: "
                 << trace_path_ << ", error: " << e.what();
      init_failed_ = true;
      return;
    }

    if (!per_file_) {
      ofs_.open(trace_path_, std::ios::out | std::ios::app);
      if (!ofs_.is_open()) {
        LOG(ERROR) << "Failed to open request trace file: " << trace_path_;
        init_failed_ = true;
        return;
      }
    }

    writer_thread_ = std::thread(&RequestTracer::writer_loop, this);

    LOG(INFO) << "Request input/output trace enabled, dumping to: "
              << trace_path_
              << (per_file_ ? " (one file per request)" : " (jsonl)");
  });
}

void RequestTracer::enqueue(WriteTask task) {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    write_queue_.push_back(std::move(task));
  }
  queue_cv_.notify_one();
}

void RequestTracer::writer_loop() {
  while (true) {
    std::deque<WriteTask> batch;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [this] { return stop_ || !write_queue_.empty(); });
      batch.swap(write_queue_);
    }

    for (auto& task : batch) {
      if (task.file_path.empty()) {
        if (ofs_.is_open()) {
          ofs_ << task.serialized << "\n";
        }
      } else {
        std::ofstream file_ofs(task.file_path, std::ios::out | std::ios::trunc);
        if (file_ofs.is_open()) {
          file_ofs << task.serialized;
        } else {
          LOG(ERROR) << "Failed to open request trace file: " << task.file_path;
        }
      }
    }

    if (!batch.empty() && ofs_.is_open()) {
      ofs_.flush();
    }

    if (stop_) {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      for (auto& task : write_queue_) {
        if (task.file_path.empty()) {
          if (ofs_.is_open()) {
            ofs_ << task.serialized << "\n";
          }
        } else {
          std::ofstream file_ofs(task.file_path,
                                 std::ios::out | std::ios::trunc);
          if (file_ofs.is_open()) {
            file_ofs << task.serialized;
          }
        }
      }
      write_queue_.clear();
      break;
    }
  }
}

std::map<size_t, RequestTracer::SeqRecord> RequestTracer::build_seq_records(
    const std::vector<SequenceOutput>& outputs) {
  std::map<size_t, SeqRecord> records;
  for (const auto& seq_output : outputs) {
    auto& record = records[seq_output.index];
    record.text = seq_output.text;
    if (seq_output.finish_reason.has_value()) {
      record.finish_reason = seq_output.finish_reason;
    }
    record.token_ids = seq_output.token_ids;
  }
  return records;
}

void RequestTracer::write_record(Request& request,
                                 const std::map<size_t, SeqRecord>& seq_records,
                                 const std::optional<Usage>& usage,
                                 bool finished,
                                 bool cancelled) {
  ensure_initialized();
  if (init_failed_) {
    return;
  }

  const auto& state = request.state();

  nlohmann::json record;
  record["request_id"] = request.request_id();
  record["x_request_id"] = request.x_request_id();
  record["x_request_time"] = request.x_request_time();
  record["service_request_id"] = request.service_request_id();
  record["timestamp"] = absl::FormatTime(
      "%Y-%m-%dT%H:%M:%E3S%z", absl::Now(), absl::LocalTimeZone());
  record["stream"] = state.stream;

  // input
  nlohmann::json input;
  input["prompt"] = state.prompt;
  input["prompt_token_ids"] = state.prompt_tokens;
  input["prompt_tokens_num"] = state.prompt_tokens.size();
  input["max_tokens"] = state.stopping_checker.get_max_generated_tokens();
  input["n"] = state.n;
  input["best_of"] = state.best_of;
  input["temperature"] = state.sampling_param.temperature;
  input["top_p"] = state.sampling_param.top_p;
  input["top_k"] = state.sampling_param.top_k;
  input["frequency_penalty"] = state.sampling_param.frequency_penalty;
  input["presence_penalty"] = state.sampling_param.presence_penalty;
  input["repetition_penalty"] = state.sampling_param.repetition_penalty;
  record["input"] = std::move(input);

  // output
  nlohmann::json output;
  nlohmann::json sequences = nlohmann::json::array();
  for (const auto& [index, seq] : seq_records) {
    nlohmann::json seq_json;
    seq_json["index"] = index;
    seq_json["text"] = seq.text;
    seq_json["token_ids"] = seq.token_ids;
    if (seq.finish_reason.has_value()) {
      seq_json["finish_reason"] = seq.finish_reason.value();
    }
    sequences.push_back(std::move(seq_json));
  }
  output["sequences"] = std::move(sequences);
  if (usage.has_value()) {
    nlohmann::json usage_json;
    usage_json["prompt_tokens"] = usage->num_prompt_tokens;
    usage_json["generated_tokens"] = usage->num_generated_tokens;
    usage_json["total_tokens"] = usage->num_total_tokens;
    output["usage"] = std::move(usage_json);
  }
  output["finished"] = finished;
  output["cancelled"] = cancelled;
  record["output"] = std::move(output);

  // raw HTTP request for replay
  if (state.call_.has_value() && state.call_.value() != nullptr) {
    const auto& raw_body = state.call_.value()->raw_request_body();
    if (!raw_body.empty()) {
      nlohmann::json raw_req;
      raw_req["body"] =
          nlohmann::json::parse(raw_body, nullptr, /*allow_exceptions=*/false);
      raw_req["path"] = state.call_.value()->request_endpoint();
      record["raw_request"] = std::move(raw_req);
    }
  }

  WriteTask task;
  if (per_file_) {
    task.file_path = (std::filesystem::path(trace_path_) /
                      (sanitize_filename(request.request_id()) + ".json"))
                         .string();
    task.serialized = record.dump(
        /*indent=*/2,
        /*indent_char=*/' ',
        /*ensure_ascii=*/false,
        nlohmann::json::error_handler_t::replace);
  } else {
    task.serialized = record.dump(
        /*indent=*/-1,
        /*indent_char=*/' ',
        /*ensure_ascii=*/false,
        nlohmann::json::error_handler_t::replace);
  }

  enqueue(std::move(task));
}

void RequestTracer::trace_completed_request(Request& request,
                                            const RequestOutput& output) {
  if (request.state().stream) {
    if (enabled()) {
      flush_stream_state(request);
    } else {
      // Tracing may have been disabled at runtime while this streaming request
      // was in flight. Drop any accumulated state so it does not leak.
      std::lock_guard<std::mutex> lock(stream_mutex_);
      stream_states_.erase(request.request_id());
    }
    return;
  }
  if (!enabled()) {
    return;
  }
  write_record(request,
               build_seq_records(output.outputs),
               output.usage,
               output.finished,
               output.cancelled);
}

void RequestTracer::trace_stream_output(Request& request,
                                        const RequestOutput& output) {
  if (!enabled()) {
    return;
  }

  // Only accumulate the delta here. The accumulated state is flushed exactly
  // once by trace_completed_request (which routes streaming requests through
  // flush_stream_state) when the request is finished or cancelled. Flushing
  // here based on request.finished()/cancelled() is unsafe under
  // enable_schedule_overlap: an earlier stream callback may observe the
  // request as already finished because a later scheduling step has advanced
  // the sequences, and flush partial state, producing one record per chunk plus
  // trailing empty records.
  std::lock_guard<std::mutex> lock(stream_mutex_);
  auto& stream_state = stream_states_[request.request_id()];
  for (const auto& seq_output : output.outputs) {
    auto& record = stream_state[seq_output.index];
    record.text += seq_output.text;
    record.token_ids.insert(record.token_ids.end(),
                            seq_output.token_ids.begin(),
                            seq_output.token_ids.end());
    if (seq_output.finish_reason.has_value()) {
      record.finish_reason = seq_output.finish_reason;
    }
  }
}

void RequestTracer::flush_stream_state(Request& request) {
  std::map<size_t, SeqRecord> seq_records;
  {
    std::lock_guard<std::mutex> lock(stream_mutex_);
    auto it = stream_states_.find(request.request_id());
    if (it == stream_states_.end()) {
      return;
    }
    seq_records = std::move(it->second);
    stream_states_.erase(it);
  }

  // Streaming deltas never carry finish_reason (generate_streaming_output does
  // not populate it), so backfill it from each sequence's final finish reason.
  const auto& sequences = request.sequences();
  for (auto& [index, record] : seq_records) {
    if (!record.finish_reason.has_value() && index < sequences.size()) {
      const auto& seq = sequences[index];
      if (seq->finish_reason() != FinishReason::NONE) {
        record.finish_reason = seq->finish_reason().to_string();
      }
    }
  }

  write_record(request,
               seq_records,
               /*usage=*/std::nullopt,
               request.finished(),
               request.cancelled());
}

}  // namespace xllm
