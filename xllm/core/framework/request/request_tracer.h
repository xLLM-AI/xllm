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

#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "request_output.h"

namespace xllm {

class Request;

// RequestTracer dumps the input and output of each request to a file when
// FLAGS_enable_request_trace is set. The flag is checked on every request so
// tracing can be enabled/disabled at runtime without restarting the service.
//
// Two output layouts are supported (FLAGS_request_trace_per_file):
//   - false: append one JSON record per line to FLAGS_request_trace_path.
//   - true : write one pretty-printed JSON file per request, named
//            "<FLAGS_request_trace_path>/<request_id>.json".
//
// For non-streaming requests the complete output is written directly once the
// request is finished. For streaming requests the delta outputs (text and
// token ids) are accumulated per request and the complete output is written
// only after all the streaming outputs are finished (or the request is
// cancelled).
//
// All file I/O is performed asynchronously on a dedicated background thread
// to avoid blocking the response threads.
class RequestTracer {
 public:
  static RequestTracer& get_instance();

  // Check the live flag value; callers use this for the fast-path skip.
  bool enabled() const;

  // Dump a finished non-streaming request. The given output already contains
  // the complete generated text.
  void trace_completed_request(Request& request, const RequestOutput& output);

  // Accumulate the delta output of a streaming request. The complete record is
  // written out once the request is finished or cancelled.
  void trace_stream_output(Request& request, const RequestOutput& output);

 private:
  RequestTracer();
  ~RequestTracer();
  RequestTracer(const RequestTracer&) = delete;
  RequestTracer& operator=(const RequestTracer&) = delete;

  // Fully assembled output for a single sequence of a request.
  struct SeqRecord {
    std::string text;
    std::optional<std::string> finish_reason;
    // raw generated token id sequence (the original/unparsed sequence).
    std::vector<int32_t> token_ids;
  };

  // A unit of work for the background writer thread.
  struct WriteTask {
    std::string serialized;
    std::string file_path;  // non-empty only in per-file mode
  };

  // Lazily initialize the writer thread and output file/directory on first
  // use (called when the flag transitions from off to on at runtime).
  void ensure_initialized();

  // Write out and erase any accumulated streaming state for the request. Used
  // both for the normal streaming finish and as a cleanup path when a streaming
  // request is finalized without a final streaming callback (e.g. cancelled).
  void flush_stream_state(Request& request);

  // Build the seq records (text/finish_reason/token_ids) from a RequestOutput.
  static std::map<size_t, SeqRecord> build_seq_records(
      const std::vector<SequenceOutput>& outputs);

  // Serialize and enqueue a record for async writing.
  void write_record(Request& request,
                    const std::map<size_t, SeqRecord>& seq_records,
                    const std::optional<Usage>& usage,
                    bool finished,
                    bool cancelled);

  // Enqueue a serialized record for async writing.
  void enqueue(WriteTask task);

  // Background writer thread entry point.
  void writer_loop();

  std::once_flag init_flag_;
  bool init_failed_ = false;

  // when true, write one file per request; otherwise append JSONL.
  bool per_file_ = false;

  // the configured output path (file when per_file_ is false, directory when
  // per_file_ is true).
  std::string trace_path_;

  // Guards stream_states_ only (no longer guards file I/O).
  std::mutex stream_mutex_;

  // request_id -> accumulated streaming state (per sequence index).
  std::unordered_map<std::string, std::map<size_t, SeqRecord>> stream_states_;

  // --- Async writer ---
  std::thread writer_thread_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<WriteTask> write_queue_;
  bool stop_ = false;

  // only used when per_file_ is false; owned by writer thread.
  std::ofstream ofs_;
};

}  // namespace xllm
