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

#include <absl/strings/str_split.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
// #include <pybind11/embed.h>

#include <folly/init/Init.h>
#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "distributed_runtime/vlm_master.h"
#include "framework/request/sequence.h"
#include "framework/sampling/sampling_params.h"
#include "runtime/utils.h"

std::vector<char> get_the_bytes(std::string filename) {
  std::ifstream input(filename, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                          (std::istreambuf_iterator<char>()));

  input.close();
  return bytes;
}

torch::Tensor load_tensor(std::string filename) {
  std::vector<char> f = get_the_bytes(filename);
  torch::IValue x = torch::pickle_load(f);
  torch::Tensor my_tensor = x.toTensor();
  return my_tensor;
}

bool run_inference(const std::string& input_embedding_path,
                   xllm::VLMMaster& master,
                   const xllm::RequestParams& sp) {
  torch::Tensor input_embedding = xllm::load_tensor(input_embedding_path);
  std::string prompt =
      "<|im_start|>system\nYou are a helpful "
      "assistant.<|im_end|>\n<|im_start|>user\n(<image>./</"
      "image>)\nStructured output text information in the "
      "graph?<|im_end|>\n<|im_"
      "start|>assistant\n";
  bool stream = false;
  xllm::OutputCallback callback = [](xllm::RequestOutput output) -> bool {
    if (output.finished) {
      std::cout << "output.outputs.size(): " << output.outputs.size()
                << std::endl;
      if (output.outputs.size() > 0) {
        const auto& text = output.outputs[0].text;
        std::cout << "Callback called. text: " << text << std::endl;
      } else {
        std::cout << "Callback called. output.outputs.size() = 0" << std::endl;
      }
      return true;
    } else {
      std::cout << "not finished" << std::endl;
      return false;
    }
  };
  /*
  std::future<bool> future = master.handle_request(
      input_embedding, prompt, sp, callback);
  bool result = future.get();
  std::cout << "Final result: " << (result ? "Success" : "Failure")
            << std::endl;*/
  master.run_until_complete();
  return true;
}

bool run_batch_inference(const std::string& input_embedding_path,
                         xllm::VLMMaster& master,
                         const xllm::RequestParams& sp) {
  std::vector<torch::Tensor> input_embeddings;
  torch::Tensor input_embedding = xllm::load_tensor(input_embedding_path);
  xllm::print_tensor(input_embedding, "input_embedding", -1, false, false);
  input_embeddings.emplace_back(input_embedding);
  input_embeddings.emplace_back(input_embedding);
  std::vector<std::string> prompts;
  std::string prompt =
      "<|im_start|>system\nYou are a helpful "
      "assistant.<|im_end|>\n<|im_start|>user\n(<image>./</"
      "image>)\nStructured output text information in the "
      "graph?<|im_end|>\n<|im_"
      "start|>assistant\n";
  prompts.emplace_back(prompt);
  prompts.emplace_back(prompt);
  std::vector<xllm::RequestParams> sps;
  sps.emplace_back(sp);
  bool stream = false;
  xllm::BatchOutputCallback batch_callback =
      [](size_t index, xllm::RequestOutput output) -> bool {
    if (output.finished) {
      std::cout << "output.outputs.size(): " << output.outputs.size()
                << std::endl;
      if (output.outputs.size() > 0) {
        const auto& text = output.outputs[0].text;
        std::cout << "Callback called. "
                  << "index: " << index << ", text: " << text << std::endl;
      } else {
        std::cout << "Callback called. output.outputs.size() = 0" << std::endl;
      }
      return true;
    } else {
      std::cout << "not finished" << std::endl;
      return false;
    }
  };
  master.handle_batch_request(input_embeddings, prompts, sps, batch_callback);
  /*
  std::vector<bool> results = futures.get();
  for (const auto& result : results) {
    std::cout << "Final result: " << (result ? "Success" : "Failure")
              << std::endl;
  }*/
  master.run_until_complete();
  return true;
  // return results[0];
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
    return 1;
  }
  FLAGS_minloglevel = 0;
  folly::Init init(&argc, &argv);
  std::string model_path = argv[1];
  std::string input_embedding_path = argv[2];
  xllm::VLMMaster::Options option;
  // option.model_path() = "/ktd/llava-1.5-7b-hf";
  // option.model_path() = "/ktd/MiniCPM-V-2_6";
  option.model_path() = model_path;
  option.max_tokens_per_batch() = 2048;
  // option.enable_prefix_cache() = false;

  std::cout << "begin init xllm::VLMMaster==========" << std::endl;
  xllm::VLMMaster master(option);

  // torch::Tensor image_tensor;
  // torch::Tensor image_tensor = torch::empty({});
  // torch::load(image_tensor, "/ktd/xllm/image_tensor.pt");
  // torch::Tensor image_tensor = load_tensor("/ktd/xllm/image_tensor.pt");
  // std::string prompt = "USER: <image>\nWhat are these?\nASSISTANT:";

  xllm::RequestParams sp;
  sp.max_tokens = 1024;
  sp.temperature = 0;
  sp.stop_token_ids = {151645, 151643};
  // run_inference(input_embedding_path, master);
  run_batch_inference(input_embedding_path, master);

  return 0;
}
