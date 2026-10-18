/* Copyright 2026 The xLLM Authors.

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

#include "core/runtime/mtp_py_executor_pair.h"

#include <torch/python.h>

#include <utility>

#include "core/runtime/py_executor_impl.h"
#include "core/util/pybind_helper.h"

namespace py = pybind11;

namespace xllm::detail {
namespace {

MtpPyGraphOutput parse_graph_output(const py::object& output) {
  MtpPyGraphOutput result;
  result.token_state = tensor_from_python(output.attr("token_state"));
  result.committed_tokens =
      output.attr("committed_tokens").cast<torch::Tensor>();
  result.target_embeddings =
      tensor_from_python(output.attr("target_embeddings"));
  result.committed_log_probs = tensor_from_python(output.attr("logprobs"));
  result.target_top_log_probs = tensor_from_python(output.attr("top_logprobs"));
  result.target_top_tokens = tensor_from_python(output.attr("top_tokens"));
  return result;
}

}  // namespace

std::unique_ptr<MtpPyGraphVariantRegistry> MtpPyGraphVariantRegistry::create(
    PyExecutorImpl& target_executor,
    PyExecutorImpl& draft_executor,
    int32_t max_variants) {
  py::gil_scoped_acquire gil;
  py::object registry = target_executor.create_mtp_graph_variant_registry(
      draft_executor, max_variants);
  return std::unique_ptr<MtpPyGraphVariantRegistry>(
      new MtpPyGraphVariantRegistry(std::move(registry)));
}

MtpPyGraphVariantRegistry::MtpPyGraphVariantRegistry(py::object registry)
    : registry_(std::move(registry)),
      execute_sparse_(registry_.attr("execute_sparse")) {}

MtpPyGraphVariantRegistry::~MtpPyGraphVariantRegistry() {
  clear_python_object(execute_sparse_);
  clear_python_object(registry_);
}

MtpPyGraphOutput MtpPyGraphVariantRegistry::execute_sparse(
    const torch::Tensor& block_table,
    const torch::Tensor& first_kv_seq_lens,
    const torch::Tensor& first_slots,
    const torch::Tensor& repair_token_ids,
    const torch::Tensor& seed_token_ids,
    const torch::Tensor& base_positions,
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& draft_input_embedding,
    int32_t batch_size,
    int32_t speculative_tokens,
    int64_t vocab_size,
    int32_t block_size,
    bool target_step_major_layout,
    bool logprobs,
    int32_t max_top_logprobs) {
  py::gil_scoped_acquire gil;
  py::object output = execute_sparse_(block_table,
                                      first_kv_seq_lens,
                                      first_slots,
                                      repair_token_ids,
                                      seed_token_ids,
                                      base_positions,
                                      kv_seq_lens,
                                      draft_input_embedding,
                                      batch_size,
                                      speculative_tokens,
                                      vocab_size,
                                      block_size,
                                      target_step_major_layout,
                                      logprobs,
                                      max_top_logprobs);
  MtpPyGraphOutput parsed = parse_graph_output(output);
  parsed.draft_embedding_destination =
      tensor_from_python(registry_.attr("draft_embedding_destination"));
  return parsed;
}

}  // namespace xllm::detail
