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

#include "tokenizer_factory.h"

#include <filesystem>

namespace xllm {

std::unique_ptr<Tokenizer> TokenizerFactory::create_tokenizer(
    const std::string& model_weights_path,
    TokenizerArgs tokenizer_args,
    bool proxy) {
  std::unique_ptr<Tokenizer> tokenizer;
  if (tokenizer_args.tokenizer_type() == "rwkv") {
    LOG(INFO) << "Create RWKV trie tokenizer.";
    tokenizer =
        std::make_unique<RwkvTokenizer>(model_weights_path, tokenizer_args);
  } else if (tokenizer_args.tokenizer_type() == "fast") {
    // 1. fast tokenizer
    LOG(INFO) << "Create fast tokenizer.";
    tokenizer = std::make_unique<FastTokenizer>(tokenizer_args);
  } else if (tokenizer_args.tokenizer_type() == "tiktoken" ||
             tokenizer_args.tokenizer_class() == "TikTokenTokenizer") {
    // 2. create tiktoken tokenizer
    LOG(INFO) << "Create Tiktoken tokenizer.";
    tokenizer =
        std::make_unique<TiktokenTokenizer>(model_weights_path, tokenizer_args);
  } else if (tokenizer_args.tokenizer_type() == "rec") {
    // 3. create rec tokenizer
    LOG(INFO) << "Create rec tokenizer.";
    tokenizer =
        std::make_unique<RecTokenizer>(model_weights_path, tokenizer_args);
  } else {
    // 4. create sentencepiece tokenizer
    LOG(INFO) << "Create SentencePiece tokenizer.";
    tokenizer = std::make_unique<SentencePieceTokenizer>(model_weights_path,
                                                         tokenizer_args);
  }

  if (proxy) {
    return std::make_unique<TokenizerProxy>(std::move(tokenizer));
  }
  return tokenizer;
}

}  // namespace xllm
