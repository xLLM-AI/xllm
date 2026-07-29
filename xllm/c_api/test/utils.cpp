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

#include "utils.h"

#include <gflags/gflags.h>

#include <algorithm>
#include <cstring>
#include <string>

// --- Server + XLLM_InitOptions gflags (defaults aligned with REC defaults) ---
DEFINE_string(model_path, "", "Path to REC model weights");
DEFINE_int32(port, 8000, "brpc TCP port");
DEFINE_string(listen_addr,
              "",
              "If non-empty, brpc listen endpoint (host:port), overrides port");
DEFINE_int32(idle_timeout_s,
             -1,
             "brpc connection idle timeout in seconds; -1 = no limit");
DEFINE_string(backend,
              "rec",
              "C API mode for xllm_test: \"llm\" (c_api/llm.h) or \"rec\" "
              "(c_api/rec.h); only one is loaded");

DEFINE_bool(enable_chunked_prefill, false, "");
DEFINE_bool(enable_prefix_cache, false, "");
DEFINE_bool(enable_disagg_pd, false, "");
DEFINE_bool(enable_pd_ooc, false, "");
DEFINE_bool(enable_schedule_overlap, false, "");
DEFINE_bool(enable_shm, false, "");

DEFINE_uint32(transfer_listen_port, 26000, "");
DEFINE_uint32(nnodes, 1, "");
DEFINE_uint32(node_rank, 0, "");
DEFINE_uint32(dp_size, 1, "");
DEFINE_uint32(cp_size, 1, "");
DEFINE_uint32(ep_size, 1, "");
DEFINE_uint32(block_size, 1, "");
DEFINE_uint32(max_cache_size, 1000000, "");
DEFINE_uint32(max_tokens_per_batch, 4096, "");
DEFINE_uint32(max_seqs_per_batch, 4, "");
DEFINE_uint32(max_tokens_per_chunk_for_prefill, 0, "");
DEFINE_uint32(num_speculative_tokens, 0, "");
DEFINE_uint32(num_request_handling_threads, 4, "");
DEFINE_uint32(expert_parallel_degree, 0, "");
DEFINE_uint32(server_idx, 0, "");
DEFINE_uint32(beam_width, 128, "");
DEFINE_uint32(max_decode_rounds, 3, "");
DEFINE_uint32(max_token_per_req, 1000, "");

DEFINE_double(max_memory_utilization, 0.55, "");

DEFINE_string(init_task, "generate", "XLLM_InitOptions.task");
DEFINE_string(communication_backend, "lccl", "");
DEFINE_string(instance_role, "DEFAULT", "");
DEFINE_string(master_node_addr, "127.0.0.1:18899", "");
DEFINE_string(xservice_addr, "", "");
DEFINE_string(instance_name, "", "");
DEFINE_string(kv_cache_transfer_mode, "PUSH", "");
// Not named "log_dir": glog already registers FLAGS_log_dir.
DEFINE_string(xllm_init_log_dir, "", "");
DEFINE_string(draft_model, "", "");

namespace xllm_capi_test {

namespace {

void CopyToFixed(char* dst, const std::string& s, size_t cap) {
  if (cap == 0) {
    return;
  }
  std::strncpy(dst, s.c_str(), cap - 1);
  dst[cap - 1] = '\0';
}

std::unique_ptr<char[]> CopyCStr(const std::string& s) {
  auto p = std::make_unique<char[]>(s.size() + 1);
  if (!s.empty()) {
    std::memcpy(p.get(), s.data(), s.size());
  }
  p[s.size()] = '\0';
  return p;
}

size_t TensorNumElements(const XLLM_Dims& d) {
  if (d.rank <= 0) {
    return 0;
  }
  size_t n = 1;
  for (int i = 0; i < d.rank && i < 8; ++i) {
    if (d.dim[i] <= 0) {
      return 0;
    }
    n *= static_cast<size_t>(d.dim[i]);
  }
  return n;
}

size_t DTypeSize(XLLM_DataType dt) {
  switch (dt) {
    case XLLM_DTYPE_FLOAT16:
    case XLLM_DTYPE_BFLOAT16:
      return 2;
    case XLLM_DTYPE_FLOAT32:
      return 4;
    case XLLM_DTYPE_FLOAT64:
      return 8;
    case XLLM_DTYPE_INT8:
    case XLLM_DTYPE_UINT8:
      return 1;
    case XLLM_DTYPE_INT16:
    case XLLM_DTYPE_UINT16:
      return 2;
    case XLLM_DTYPE_INT32:
    case XLLM_DTYPE_UINT32:
      return 4;
    case XLLM_DTYPE_INT64:
    case XLLM_DTYPE_UINT64:
      return 8;
    case XLLM_DTYPE_BOOL:
      return 1;
    case XLLM_DTYPE_STRING:
    case XLLM_DTYPE_UNDEFINED:
    default:
      return 0;
  }
}

}  // namespace

void ApplyGflagsToXllmInitOptions(XLLM_InitOptions* o) {
  o->enable_chunked_prefill = FLAGS_enable_chunked_prefill;
  o->enable_prefix_cache = FLAGS_enable_prefix_cache;
  o->enable_disagg_pd = FLAGS_enable_disagg_pd;
  o->enable_pd_ooc = FLAGS_enable_pd_ooc;
  o->enable_schedule_overlap = FLAGS_enable_schedule_overlap;
  o->enable_shm = FLAGS_enable_shm;

  o->transfer_listen_port = FLAGS_transfer_listen_port;
  o->nnodes = FLAGS_nnodes;
  o->node_rank = FLAGS_node_rank;
  o->dp_size = FLAGS_dp_size;
  o->cp_size = FLAGS_cp_size;
  o->ep_size = FLAGS_ep_size;
  o->block_size = FLAGS_block_size;
  o->max_cache_size = FLAGS_max_cache_size;
  o->max_tokens_per_batch = FLAGS_max_tokens_per_batch;
  o->max_seqs_per_batch = FLAGS_max_seqs_per_batch;
  o->max_tokens_per_chunk_for_prefill = FLAGS_max_tokens_per_chunk_for_prefill;
  o->num_speculative_tokens = FLAGS_num_speculative_tokens;
  o->num_request_handling_threads = FLAGS_num_request_handling_threads;
  o->expert_parallel_degree = FLAGS_expert_parallel_degree;
  o->server_idx = FLAGS_server_idx;
  o->beam_width = FLAGS_beam_width;
  o->max_decode_rounds = FLAGS_max_decode_rounds;
  o->max_token_per_req = FLAGS_max_token_per_req;

  o->max_memory_utilization = static_cast<float>(FLAGS_max_memory_utilization);

  CopyToFixed(o->task, FLAGS_init_task, XLLM_META_STRING_FIELD_MAX_LEN);
  CopyToFixed(o->communication_backend,
              FLAGS_communication_backend,
              XLLM_META_STRING_FIELD_MAX_LEN);
  CopyToFixed(
      o->instance_role, FLAGS_instance_role, XLLM_META_STRING_FIELD_MAX_LEN);
  CopyToFixed(o->master_node_addr,
              FLAGS_master_node_addr,
              XLLM_META_STRING_FIELD_MAX_LEN);
  CopyToFixed(
      o->xservice_addr, FLAGS_xservice_addr, XLLM_META_STRING_FIELD_MAX_LEN);
  CopyToFixed(
      o->instance_name, FLAGS_instance_name, XLLM_META_STRING_FIELD_MAX_LEN);
  CopyToFixed(o->kv_cache_transfer_mode,
              FLAGS_kv_cache_transfer_mode,
              XLLM_META_STRING_FIELD_MAX_LEN);
  CopyToFixed(
      o->log_dir, FLAGS_xllm_init_log_dir, XLLM_META_STRING_FIELD_MAX_LEN);
  CopyToFixed(
      o->draft_model, FLAGS_draft_model, XLLM_META_STRING_FIELD_MAX_LEN);
}

void PbToXllmDims(const c_api_test::XLLM_Dims& pb, XLLM_Dims* out) {
  std::memset(out->dim, 0, sizeof(out->dim));
  out->rank = pb.rank();
  const int n = std::min(8, pb.dim_size());
  for (int i = 0; i < n; ++i) {
    out->dim[i] = pb.dim(i);
  }
}

void XllmDimsToPb(const XLLM_Dims& in, c_api_test::XLLM_Dims* pb) {
  pb->set_rank(in.rank);
  pb->clear_dim();
  const int n = std::min(8, in.rank);
  for (int i = 0; i < n; ++i) {
    pb->add_dim(in.dim[i]);
  }
}

void PbToXllmTensor(const c_api_test::XLLM_Tensor& pb,
                    XLLM_Tensor* out,
                    MmDataOwned* owned) {
  out->dtype = static_cast<XLLM_DataType>(pb.dtype());
  PbToXllmDims(pb.dims(), &out->dims);
  owned->tensor_byte_buffers.emplace_back(pb.data().begin(), pb.data().end());
  std::vector<uint8_t>& buf = owned->tensor_byte_buffers.back();
  out->data = buf.empty() ? nullptr : static_cast<const void*>(buf.data());
}

void XllmTensorToPb(const XLLM_Tensor& in, c_api_test::XLLM_Tensor* pb) {
  pb->set_dtype(static_cast<c_api_test::XLLM_DataType>(in.dtype));
  XllmDimsToPb(in.dims, pb->mutable_dims());
  const size_t n = TensorNumElements(in.dims);
  const size_t es = DTypeSize(in.dtype);
  if (in.data != nullptr && n > 0 && es > 0) {
    pb->set_data(static_cast<const char*>(in.data), n * es);
  } else {
    pb->clear_data();
  }
}

void PbToXllmTensors(const c_api_test::XLLM_Tensors& pb,
                     XLLM_Tensors* out,
                     MmDataOwned* owned) {
  owned->tensor_lists.emplace_back();
  std::vector<XLLM_Tensor>& row = owned->tensor_lists.back();
  row.reserve(static_cast<size_t>(pb.entries_size()));
  for (int i = 0; i < pb.entries_size(); ++i) {
    XLLM_Tensor t{};
    PbToXllmTensor(pb.entries(i), &t, owned);
    row.push_back(t);
  }
  out->entries = row.data();
  out->entries_size = row.size();
}

void XllmTensorsToPb(const XLLM_Tensors& in, c_api_test::XLLM_Tensors* pb) {
  pb->clear_entries();
  for (size_t i = 0; i < in.entries_size; ++i) {
    XllmTensorToPb(in.entries[i], pb->add_entries());
  }
}

void PbToXllmMmValue(const c_api_test::XLLM_MM_Value& pb,
                     XLLM_MM_Value* out,
                     MmDataOwned* owned) {
  std::memset(out, 0, sizeof(*out));
  out->is_single_tensor = pb.is_single_tensor();
  switch (pb.data_case()) {
    case c_api_test::XLLM_MM_Value::kTensor:
      out->is_single_tensor = true;
      PbToXllmTensor(pb.tensor(), &out->data.tensor, owned);
      break;
    case c_api_test::XLLM_MM_Value::kTensors:
      out->is_single_tensor = false;
      PbToXllmTensors(pb.tensors(), &out->data.tensors, owned);
      break;
    default:
      break;
  }
}

void XllmMmValueToPb(const XLLM_MM_Value& in, c_api_test::XLLM_MM_Value* pb) {
  pb->set_is_single_tensor(in.is_single_tensor);
  if (in.is_single_tensor) {
    XllmTensorToPb(in.data.tensor, pb->mutable_tensor());
  } else {
    XllmTensorsToPb(in.data.tensors, pb->mutable_tensors());
  }
}

void PbToXllmMmDict(const c_api_test::XLLM_MM_Dict& pb,
                    XLLM_MM_Dict* out,
                    MmDataOwned* owned) {
  owned->mm_dict_entries.clear();
  owned->mm_dict_entries.reserve(static_cast<size_t>(pb.entries_size()));
  for (int i = 0; i < pb.entries_size(); ++i) {
    owned->mm_dict_entries.emplace_back();
    XLLM_MM_DictEntry& e = owned->mm_dict_entries.back();
    std::memset(e.key, 0, sizeof(e.key));
    const std::string& k = pb.entries(i).key();
    std::strncpy(e.key, k.c_str(), XLLM_META_STRING_FIELD_MAX_LEN - 1);
    PbToXllmMmValue(pb.entries(i).value(), &e.value, owned);
  }
  out->entries = owned->mm_dict_entries.data();
  out->entries_size = owned->mm_dict_entries.size();
}

void XllmMmDictToPb(const XLLM_MM_Dict& in, c_api_test::XLLM_MM_Dict* pb) {
  pb->clear_entries();
  for (size_t i = 0; i < in.entries_size; ++i) {
    c_api_test::XLLM_MM_DictEntry* e = pb->add_entries();
    e->set_key(in.entries[i].key);
    XllmMmValueToPb(in.entries[i].value, e->mutable_value());
  }
}

void PbToXllmMmItems(const c_api_test::XLLM_MM_Items& pb,
                     XLLM_MM_Items* out,
                     MmDataOwned* owned) {
  owned->mm_items.clear();
  owned->mm_items.reserve(static_cast<size_t>(pb.entries_size()));
  for (int i = 0; i < pb.entries_size(); ++i) {
    owned->mm_items.emplace_back();
    PbToXllmMmItem(pb.entries(i), &owned->mm_items.back(), owned);
  }
  out->entries = owned->mm_items.data();
  out->entries_size = owned->mm_items.size();
}

void XllmMmItemsToPb(const XLLM_MM_Items& in, c_api_test::XLLM_MM_Items* pb) {
  pb->clear_entries();
  for (size_t i = 0; i < in.entries_size; ++i) {
    XllmMmItemToPb(in.entries[i], pb->add_entries());
  }
}

void PbToXllmMmState(const c_api_test::XLLM_MM_State& pb, XLLM_MM_State* out) {
  out->token_pos.offset = pb.token_pos().offset();
  out->token_pos.length = pb.token_pos().length();
}

void XllmMmStateToPb(const XLLM_MM_State& in, c_api_test::XLLM_MM_State* pb) {
  pb->mutable_token_pos()->set_offset(in.token_pos.offset);
  pb->mutable_token_pos()->set_length(in.token_pos.length);
}

void PbToXllmMmItem(const c_api_test::XLLM_MM_Item& pb,
                    XLLM_MM_Item* out,
                    MmDataOwned* owned) {
  std::memset(out, 0, sizeof(*out));
  out->type = static_cast<XLLM_MM_Type>(pb.type());
  PbToXllmMmValue(pb.data(), &out->data, owned);
  PbToXllmMmState(pb.state(), &out->state);
}

void XllmMmItemToPb(const XLLM_MM_Item& in, c_api_test::XLLM_MM_Item* pb) {
  pb->set_type(static_cast<uint32_t>(in.type));
  XllmMmValueToPb(in.data, pb->mutable_data());
  XllmMmStateToPb(in.state, pb->mutable_state());
}

bool PbToXllmMmData(const c_api_test::XLLM_MM_Data& pb,
                    XLLM_MM_Data* out,
                    MmDataOwned* owned) {
  std::memset(out, 0, sizeof(*out));
  out->type_mask = pb.type_mask();
  out->is_dict = pb.is_dict();
  switch (pb.storage_case()) {
    case c_api_test::XLLM_MM_Data::kDict:
      out->is_dict = true;
      PbToXllmMmDict(pb.dict(), &out->data.dict, owned);
      return true;
    case c_api_test::XLLM_MM_Data::kItems:
      out->is_dict = false;
      PbToXllmMmItems(pb.items(), &out->data.items, owned);
      return true;
    default:
      return false;
  }
}

void XllmMmDataToPb(const XLLM_MM_Data& in, c_api_test::XLLM_MM_Data* pb) {
  pb->set_type_mask(in.type_mask);
  pb->set_is_dict(in.is_dict);
  if (in.is_dict) {
    XllmMmDictToPb(in.data.dict, pb->mutable_dict());
  } else {
    XllmMmItemsToPb(in.data.items, pb->mutable_items());
  }
}

void PbToXllmRequestParams(const c_api_test::XLLM_RequestParams& pb,
                           XLLM_RequestParams* out) {
  out->echo = pb.echo();
  out->offline = pb.offline();
  out->logprobs = pb.logprobs();
  out->ignore_eos = pb.ignore_eos();
  out->n = pb.n();
  out->max_tokens = pb.max_tokens();
  out->best_of = pb.best_of();
  out->ttlt_slo_ms = pb.ttlt_slo_ms();
  out->ttft_slo_ms = pb.ttft_slo_ms();
  out->tpot_slo_ms = pb.tpot_slo_ms();
  out->beam_width = pb.beam_width();
  out->top_logprobs = pb.top_logprobs();
  out->top_k = pb.top_k();
  out->top_p = pb.top_p();
  out->frequency_penalty = pb.frequency_penalty();
  out->presence_penalty = pb.presence_penalty();
  out->repetition_penalty = pb.repetition_penalty();
  out->temperature = pb.temperature();
  std::strncpy(out->request_id,
               pb.request_id().c_str(),
               XLLM_META_STRING_FIELD_MAX_LEN - 1);
  out->request_id[XLLM_META_STRING_FIELD_MAX_LEN - 1] = '\0';
}

void XllmRequestParamsToPb(const XLLM_RequestParams& in,
                           c_api_test::XLLM_RequestParams* pb) {
  pb->set_echo(in.echo);
  pb->set_offline(in.offline);
  pb->set_logprobs(in.logprobs);
  pb->set_ignore_eos(in.ignore_eos);
  pb->set_n(in.n);
  pb->set_max_tokens(in.max_tokens);
  pb->set_best_of(in.best_of);
  pb->set_ttlt_slo_ms(in.ttlt_slo_ms);
  pb->set_ttft_slo_ms(in.ttft_slo_ms);
  pb->set_tpot_slo_ms(in.tpot_slo_ms);
  pb->set_beam_width(in.beam_width);
  pb->set_top_logprobs(in.top_logprobs);
  pb->set_top_k(in.top_k);
  pb->set_top_p(in.top_p);
  pb->set_frequency_penalty(in.frequency_penalty);
  pb->set_presence_penalty(in.presence_penalty);
  pb->set_repetition_penalty(in.repetition_penalty);
  pb->set_temperature(in.temperature);
  pb->set_request_id(in.request_id);
}

void PbToXllmChatMessage(const c_api_test::XLLM_ChatMessage& pb,
                         XLLM_ChatMessage* out) {
  std::memset(out->role, 0, sizeof(out->role));
  std::strncpy(
      out->role, pb.role().c_str(), XLLM_META_STRING_FIELD_MAX_LEN - 1);
  out->role[XLLM_META_STRING_FIELD_MAX_LEN - 1] = '\0';
  if (!pb.content().empty()) {
    out->content = new char[pb.content().size() + 1];
    std::memcpy(out->content, pb.content().data(), pb.content().size());
    out->content[pb.content().size()] = '\0';
  } else {
    out->content = nullptr;
  }
}

void FreeXllmChatMessageContent(XLLM_ChatMessage* out) {
  delete[] out->content;
  out->content = nullptr;
}

void XllmChatMessageToPb(const XLLM_ChatMessage* in,
                         c_api_test::XLLM_ChatMessage* pb) {
  if (!in) {
    return;
  }
  pb->set_role(in->role);
  if (in->content != nullptr) {
    pb->set_content(in->content);
  } else {
    pb->clear_content();
  }
}

void PbToXllmUsage(const c_api_test::XLLM_Usage& pb, XLLM_Usage* out) {
  out->prompt_tokens = pb.prompt_tokens();
  out->completion_tokens = pb.completion_tokens();
  out->total_tokens = pb.total_tokens();
  out->cached_tokens = pb.cached_tokens();
}

void XllmUsageToPb(const XLLM_Usage& in, c_api_test::XLLM_Usage* pb) {
  pb->set_prompt_tokens(in.prompt_tokens);
  pb->set_completion_tokens(in.completion_tokens);
  pb->set_total_tokens(in.total_tokens);
  pb->set_cached_tokens(in.cached_tokens);
}

void PbToXllmLogProbs(const c_api_test::XLLM_LogProbs& pb,
                      XLLM_LogProbs* out,
                      std::vector<XLLM_LogProb>* storage) {
  storage->clear();
  storage->reserve(static_cast<size_t>(pb.entries_size()));
  for (int i = 0; i < pb.entries_size(); ++i) {
    XLLM_LogProb e{};
    e.token_id = pb.entries(i).token_id();
    e.logprob = pb.entries(i).logprob();
    storage->push_back(e);
  }
  out->entries = storage->empty() ? nullptr : storage->data();
  out->entries_size = storage->size();
}

void XllmLogProbsToPb(const XLLM_LogProbs& in, c_api_test::XLLM_LogProbs* pb) {
  pb->clear_entries();
  if (in.entries == nullptr || in.entries_size == 0) {
    return;
  }
  for (size_t i = 0; i < in.entries_size; ++i) {
    c_api_test::XLLM_LogProb* e = pb->add_entries();
    e->set_token_id(in.entries[i].token_id);
    e->set_logprob(in.entries[i].logprob);
  }
}

void PbToXllmChoice(const c_api_test::XLLM_Choice& pb,
                    XLLM_Choice* out,
                    ResponseOwned* ro) {
  std::memset(out, 0, sizeof(*out));
  out->index = pb.index();
  if (!pb.text().empty()) {
    ro->choice_text_bufs.push_back(CopyCStr(pb.text()));
    out->text = ro->choice_text_bufs.back().get();
  }
  if (pb.has_chat_message()) {
    ro->chat_messages.emplace_back();
    XLLM_ChatMessage& cm = ro->chat_messages.back();
    std::memset(cm.role, 0, sizeof(cm.role));
    std::strncpy(cm.role,
                 pb.chat_message().role().c_str(),
                 XLLM_META_STRING_FIELD_MAX_LEN - 1);
    cm.role[XLLM_META_STRING_FIELD_MAX_LEN - 1] = '\0';
    if (!pb.chat_message().content().empty()) {
      ro->chat_message_contents.push_back(
          CopyCStr(pb.chat_message().content()));
      cm.content = ro->chat_message_contents.back().get();
    } else {
      cm.content = nullptr;
    }
    out->message = &ro->chat_messages.back();
  }
  ro->token_ids_vecs.emplace_back();
  ro->token_ids_vecs.back().reserve(static_cast<size_t>(pb.token_ids_size()));
  for (int i = 0; i < pb.token_ids_size(); ++i) {
    ro->token_ids_vecs.back().push_back(pb.token_ids(i));
  }
  out->token_ids = ro->token_ids_vecs.back().data();
  out->token_size = ro->token_ids_vecs.back().size();

  ro->logprob_vecs.emplace_back();
  std::vector<XLLM_LogProb>& le = ro->logprob_vecs.back();
  le.reserve(static_cast<size_t>(pb.logprobs().entries_size()));
  for (int i = 0; i < pb.logprobs().entries_size(); ++i) {
    XLLM_LogProb e{};
    e.token_id = pb.logprobs().entries(i).token_id();
    e.logprob = pb.logprobs().entries(i).logprob();
    le.push_back(e);
  }
  out->logprobs.entries = le.data();
  out->logprobs.entries_size = le.size();

  std::strncpy(out->finish_reason,
               pb.finish_reason().c_str(),
               XLLM_META_STRING_FIELD_MAX_LEN - 1);
  out->finish_reason[XLLM_META_STRING_FIELD_MAX_LEN - 1] = '\0';
}

void XllmChoiceToPb(const XLLM_Choice& in, c_api_test::XLLM_Choice* pb) {
  pb->set_index(in.index);
  if (in.text != nullptr) {
    pb->set_text(in.text);
  } else {
    pb->clear_text();
  }
  if (in.message != nullptr) {
    XllmChatMessageToPb(in.message, pb->mutable_chat_message());
  } else {
    pb->clear_chat_message();
  }
  pb->clear_token_ids();
  if (in.token_ids != nullptr) {
    for (size_t i = 0; i < in.token_size; ++i) {
      pb->add_token_ids(in.token_ids[i]);
    }
  }
  XllmLogProbsToPb(in.logprobs, pb->mutable_logprobs());
  pb->set_finish_reason(in.finish_reason);
}

void PbToXllmChoices(const c_api_test::XLLM_Choices& pb,
                     XLLM_Choices* out,
                     ResponseOwned* ro) {
  ro->choices.clear();
  ro->choices.reserve(static_cast<size_t>(pb.entries_size()));
  for (int i = 0; i < pb.entries_size(); ++i) {
    ro->choices.emplace_back();
    PbToXllmChoice(pb.entries(i), &ro->choices.back(), ro);
  }
  out->entries = ro->choices.data();
  out->entries_size = ro->choices.size();
}

void XllmChoicesToPb(const XLLM_Choices& in, c_api_test::XLLM_Choices* pb) {
  pb->clear_entries();
  if (in.entries == nullptr) {
    return;
  }
  for (size_t i = 0; i < in.entries_size; ++i) {
    XllmChoiceToPb(in.entries[i], pb->add_entries());
  }
}

void PbToXllmResponse(const c_api_test::XLLM_Response& pb,
                      XLLM_Response* out,
                      ResponseOwned* owned) {
  std::memset(out, 0, sizeof(*out));
  out->status_code = static_cast<XLLM_StatusCode>(pb.status_code());
  std::strncpy(
      out->error_info, pb.error_info().c_str(), XLLM_ERROR_INFO_MAX_LEN - 1);
  out->error_info[XLLM_ERROR_INFO_MAX_LEN - 1] = '\0';
  std::strncpy(out->id, pb.id().c_str(), XLLM_META_STRING_FIELD_MAX_LEN - 1);
  out->id[XLLM_META_STRING_FIELD_MAX_LEN - 1] = '\0';
  std::strncpy(
      out->object, pb.object().c_str(), XLLM_META_STRING_FIELD_MAX_LEN - 1);
  out->object[XLLM_META_STRING_FIELD_MAX_LEN - 1] = '\0';
  out->created = pb.created();
  std::strncpy(
      out->model, pb.model().c_str(), XLLM_META_STRING_FIELD_MAX_LEN - 1);
  out->model[XLLM_META_STRING_FIELD_MAX_LEN - 1] = '\0';
  PbToXllmUsage(pb.usage(), &out->usage);
  PbToXllmChoices(pb.choices(), &out->choices, owned);
}

void XllmResponseToPb(const XLLM_Response* in, c_api_test::XLLM_Response* pb) {
  if (!in) {
    pb->Clear();
    return;
  }
  pb->set_status_code(
      static_cast<c_api_test::XLLM_StatusCode>(in->status_code));
  pb->set_error_info(in->error_info);
  pb->set_id(in->id);
  pb->set_object(in->object);
  pb->set_created(in->created);
  pb->set_model(in->model);
  XllmUsageToPb(in->usage, pb->mutable_usage());
  XllmChoicesToPb(in->choices, pb->mutable_choices());
}

}  // namespace xllm_capi_test
