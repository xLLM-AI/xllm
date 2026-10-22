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

#include "core/framework/model_loader/hf_model_loader_profile.h"

#include "models/model_registry.h"

namespace xllm {

bool select_hf_model_loader_profile(const ResolvedModelDescriptor& descriptor,
                                    const std::string& requested_family,
                                    HFModelLoaderProfile* profile,
                                    std::string* error_message) {
  auto fail = [&](const std::string& message) {
    if (error_message != nullptr) {
      *error_message = message;
    }
    return false;
  };
  if (profile == nullptr) {
    return fail("HF loader profile must not be null.");
  }
  if (descriptor.registration_name.empty()) {
    return fail("HF loader requires a resolved model registration.");
  }
  if (descriptor.family == ModelFamily::DIT ||
      descriptor.meta.dit_model_factory != nullptr) {
    return fail("DiT pipelines require the DiTModelLoader entry point.");
  }

  ModelFamily family = descriptor.family;
  const std::string& family_name =
      requested_family.empty() ? descriptor.family_name : requested_family;
  if (family_name == "llm") {
    family = ModelFamily::LLM;
  } else if (family_name == "vlm") {
    family = ModelFamily::VLM;
  } else if (family_name == "rec") {
    family = ModelFamily::REC;
  } else if (!family_name.empty()) {
    return fail("Unsupported HF loader model family: " + family_name);
  }

  // Parser-only registrations have no execution factory or default family.
  // Preserve their shared text-config loading behavior until a caller selects
  // an execution contract explicitly.
  if (family == ModelFamily::UNKNOWN) {
    family = ModelFamily::LLM;
  }

  HFModelLoaderProfile selected;
  selected.family = family;
  selected.load_media_preprocessor_args = family == ModelFamily::VLM;
  selected.load_rec_vocab =
      family == ModelFamily::REC && descriptor.canonical_name == "onerec";
  *profile = selected;
  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

}  // namespace xllm
