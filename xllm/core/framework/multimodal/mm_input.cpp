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

#include "mm_input.h"

#include <butil/strings/string_number_conversions.h>
#include <glog/logging.h>

#include <algorithm>
#include <atomic>

#include "core/util/blocking_counter.h"
#include "core/util/threadpool.h"
#include "mm_handler.h"

namespace xllm {

bool is_url_type(const std::string& type) {
  return type == "image_url" || type == "video_url" || type == "audio_url";
}

namespace {

struct ProcessTask {
  size_t content_index = 0;
  size_t input_index = 0;
  MMPayload payload;
};

bool is_binary_data_url(std::string_view url) {
  constexpr std::string_view kPrefix = "data:";
  constexpr std::string_view kMarker = ";binary,";

  if (url.substr(0, kPrefix.size()) != kPrefix) {
    return false;
  }
  const size_t pos = url.find(kMarker);
  if (pos == std::string_view::npos) {
    return false;
  }
  return true;
}

size_t parse_binary_url_size(std::string_view url) {
  constexpr std::string_view kMarker = ";binary,";
  const auto tail = url.substr(url.find(kMarker) + kMarker.size());
  size_t size = 0;
  if (!butil::StringToSizeT(std::string(tail), &size)) {
    LOG(ERROR) << "failed to parse size from url: " << url;
    return 0;
  }
  return size;
}

MMPayload slice_payload(const MMContent& item, MMPayload& payload) {
  std::string_view url;
  if (item.type == "image_url") {
    url = item.image_url.url;
  } else if (item.type == "video_url") {
    url = item.video_url.url;
  } else if (item.type == "audio_url") {
    url = item.audio_url.url;
  } else {
    return {};
  }

  // Only binary data urls (e.g. "data:image/jpeg;binary,1024") consume bytes
  // from the shared payload. http(s), base64 data urls, and local/file://
  // paths are loaded by the per-modality handler later, so leave them alone.
  if (!is_binary_data_url(url)) {
    return {};
  }

  const size_t size = parse_binary_url_size(url);
  if (size == 0) {
    return {};
  }
  std::string buf;
  if (!payload.get(buf, size)) {
    LOG(ERROR) << "failed to slice " << size << " bytes from payload";
    return {};
  }
  return MMPayload(std::move(buf));
}

}  // namespace

bool MMInput::foreach (MMInputItem::IVisitor& v) const {
  for (const MMInputItem& item : items_) {
    if (!v.visit(item)) {
      return false;
    }
  }
  return true;
}

std::optional<XXH3Key> hash_mm_item(const MMInputItem& item) {
  if (item.is_embedding()) {
    return std::nullopt;
  }
  if (!item.uuid.empty()) {
    return hash_string(item.uuid);
  }
  if (!item.raw_data.empty()) {
    return hash_string(item.raw_data);
  }
  return std::nullopt;
}

MMInputTransfer::MMInputTransfer() {
  mm_handlers_ = std::make_unique<MMHandlerSet>();
  threadpool_ = std::make_unique<ThreadPool>(/*num_threads=*/16,
                                             /*cpu_binding=*/false,
                                             /*name=*/"MMInputTransfer");
}

MMInputTransfer::~MMInputTransfer() {}

MMErrCode MMInputTransfer::trans(const std::vector<Message>& messages,
                                 MMInput& inputs) {
  size_t item_count = 0;
  for (const Message& message : messages) {
    const MMContentVec& contents = std::get<MMContentVec>(message.content);
    item_count += std::count_if(
        contents.begin(), contents.end(), [](const MMContent& item) {
          return item.type != "text";
        });
  }
  return trans(messages, std::vector<bool>(item_count, true), inputs);
}

MMErrCode MMInputTransfer::trans(const std::vector<Message>& messages,
                                 const std::vector<bool>& selected_items,
                                 MMInput& inputs) {
  inputs.clear();
  std::vector<MMInputItem> ins;
  size_t selected_index = 0;

  for (const Message& message : messages) {
    const MMContentVec& mmc = std::get<MMContentVec>(message.content);

    MMErrCode code = this->trans_parallel(
        mmc, selected_items, selected_index, ins, inputs.payload());
    if (code != MMErrCode::SUCCESS) {
      return code;
    }

    inputs.insert(ins);
  }
  CHECK_EQ(selected_index, selected_items.size());
  return MMErrCode::SUCCESS;
}

MMErrCode MMInputTransfer::trans_parallel(
    const MMContentVec& mmc,
    const std::vector<bool>& selected_items,
    size_t& selected_index,
    std::vector<MMInputItem>& inputs,
    MMPayload& payload) {
  inputs.clear();
  std::vector<ProcessTask> tasks;
  tasks.reserve(mmc.size());

  for (size_t idx = 0; idx < mmc.size(); ++idx) {
    const std::string& type = mmc[idx].type;
    if (type == "text") {
      continue;
    }

    CHECK_LT(selected_index, selected_items.size());
    const bool selected = selected_items[selected_index++];
    MMPayload item_payload;
    if (is_url_type(type)) {
      item_payload = slice_payload(mmc[idx], payload);
    }
    if (!selected) {
      CHECK(is_url_type(type))
          << "Only URL items can be skipped during multimodal transfer.";
      continue;
    }

    inputs.emplace_back();
    const size_t input_index = inputs.size() - 1;
    if (is_url_type(type)) {
      tasks.emplace_back(ProcessTask{
          .content_index = idx,
          .input_index = input_index,
          .payload = std::move(item_payload),
      });
    } else {
      MMErrCode code =
          mm_handlers_->process(type, mmc[idx], inputs[input_index], payload);
      if (code != MMErrCode::SUCCESS) {
        return code;
      }
      inputs[input_index].uuid = mmc[idx].uuid;
    }
  }

  if (tasks.empty()) {
    return MMErrCode::SUCCESS;
  }

  std::atomic<MMErrCode> error{MMErrCode::SUCCESS};
  BlockingCounter counter(static_cast<int32_t>(tasks.size()));
  for (size_t i = 0; i < tasks.size(); ++i) {
    threadpool_->schedule([&, i]() {
      if (error.load() == MMErrCode::SUCCESS) {
        ProcessTask& t = tasks[i];
        const std::string& type = mmc[t.content_index].type;
        MMErrCode code = mm_handlers_->process(
            type, mmc[t.content_index], inputs[t.input_index], t.payload);
        if (code != MMErrCode::SUCCESS) {
          LOG(ERROR) << "process failed at input index " << t.input_index
                     << ", type=" << type;
          MMErrCode expected = MMErrCode::SUCCESS;
          error.compare_exchange_strong(expected, code);
        } else {
          inputs[t.input_index].uuid = mmc[t.content_index].uuid;
        }
      }
      counter.decrement_count();
    });
  }
  counter.wait();
  return error.load();
}

MMErrCode MMInputTransfer::trans(const MMContentVec& mmc,
                                 std::vector<MMInputItem>& inputs,
                                 MMPayload& payload) {
  inputs.clear();
  for (int idx = 0; idx < mmc.size(); ++idx) {
    const auto& item = mmc[idx];
    const auto& type = item.type;

    if (type != "text") {
      MMInputItem input;
      MMErrCode code = mm_handlers_->process(type, item, input, payload);
      if (code != MMErrCode::SUCCESS) {
        return code;
      }
      input.uuid = item.uuid;

      inputs.emplace_back(std::move(input));
    }
  }

  return MMErrCode::SUCCESS;
}

}  // namespace xllm
