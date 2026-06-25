/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arg.h"
#include "arg_pack.h"
#include "backend.h"
#include "dtype.h"
#include "spec.h"
#include "types.h"

namespace xllm::triton_jit {

inline constexpr bool kAppendGlobalScratch = true;

struct ParamInfo {
  bool is_constexpr = false;
  bool do_not_specialize = false;
};

class JITKernel final {
 public:
  static JITKernel& get(std::string py_path, std::string fn_name);

  template <typename... Args>
  void launch(void* stream, Grid g, LaunchCfg cfg, Args&&... args) {
    ensure_signature();
    ArgPack pack;
    SpecList specs;
    size_t idx = 0;
    (process_one(std::forward<Args>(args), pack, specs, sig_, idx), ...);
    if (kAppendGlobalScratch) {
      pack.push<void*>(nullptr);
      pack.push<void*>(nullptr);
    }
    CompiledKernel& k = compile_or_get(specs, cfg);
    k.launch(stream, g, pack.data());
  }

  const std::string& path() const { return path_; }
  const std::string& name() const { return name_; }

  void set_autotune(bool on, std::vector<LaunchCfg> space = {});

  template <typename... Args>
  void autotune(void* stream, Grid g, Args&&... args) {
    ensure_signature();
    ArgPack pack;
    SpecList specs;
    size_t idx = 0;
    (process_one(std::forward<Args>(args), pack, specs, sig_, idx), ...);
    if (kAppendGlobalScratch) {
      pack.push<void*>(nullptr);
      pack.push<void*>(nullptr);
    }
    autotune_impl(stream, g, pack, specs);
  }

 private:
  explicit JITKernel(std::string py_path, std::string fn_name);

  void ensure_signature();
  CompiledKernel& compile_or_get(const SpecList& specs,
                                 const LaunchCfg& cfg,
                                 bool use_winner = true);

  void autotune_impl(void* stream,
                     Grid g,
                     const ArgPack& pack,
                     const SpecList& specs);

  static void emit_ptr(void* data,
                       c10::ScalarType dtype,
                       SpecHint hint,
                       bool specialize,
                       ArgPack& pack,
                       SpecList& specs) {
    pack.push(data);
    ArgSpec s;
    s.kind = Kind::PTR;
    s.type = scalar_to_triton(dtype);
    s.hint = hint;
    s.specialize = specialize;
    specs.push_back(std::move(s));
  }

  template <typename T>
  static void emit_val(T v,
                       SpecHint hint,
                       bool specialize,
                       ArgPack& pack,
                       SpecList& specs) {
    ArgSpec s;
    s.kind = Kind::VAL;
    s.type = cpp_to_triton<T>();
    s.hint = hint;
    s.specialize = specialize;
    bool baked = specialize && hint == SpecHint::EQ1;
    if (!baked) {
      pack.push(v);
    }
    specs.push_back(std::move(s));
  }

  template <typename T>
  static void emit_const(T v, ArgPack& /*pack*/, SpecList& specs) {
    ArgSpec s;
    s.kind = Kind::CONST;
    s.type = cpp_to_triton<T>();
    s.const_val = format_const(v);
    specs.push_back(std::move(s));
  }

  template <typename A>
  static void process_one(A&& a,
                          ArgPack& pack,
                          SpecList& specs,
                          const std::vector<ParamInfo>& sig,
                          size_t& idx) {
    using T = std::remove_cvref_t<A>;
    const ParamInfo& info = sig.at(idx);
    ++idx;

    // Bare argument: classify by the upstream .py signature (position).
    if (info.is_constexpr) {
      if constexpr (std::is_arithmetic_v<T>) {
        emit_const(a, pack, specs);
      } else {
        LOG(FATAL) << "triton_jit: constexpr parameter at index " << (idx - 1)
                   << " expects a scalar, got a non-scalar";
      }
      return;
    }

    // Bare runtime argument: auto-deduce by type; honor do_not_specialize.
    const bool specialize = !info.do_not_specialize;
    if constexpr (std::is_same_v<T, torch::Tensor>) {
      void* data = a.data_ptr();
      emit_ptr(
          data, a.scalar_type(), hint_from_ptr(data), specialize, pack, specs);
    } else if constexpr (is_optional_tensor_v<T>) {
      if (a.has_value()) {
        void* data = a->data_ptr();
        emit_ptr(data,
                 a->scalar_type(),
                 hint_from_ptr(data),
                 specialize,
                 pack,
                 specs);
      } else {
        emit_ptr(nullptr,
                 kNullPtrDtype,
                 hint_from_ptr(nullptr),
                 specialize,
                 pack,
                 specs);
      }
    } else if constexpr (std::is_same_v<T, std::nullptr_t>) {
      emit_ptr(nullptr,
               kNullPtrDtype,
               hint_from_ptr(nullptr),
               specialize,
               pack,
               specs);
    } else if constexpr (std::is_arithmetic_v<T>) {
      SpecHint h = SpecHint::NONE;
      if constexpr (std::is_integral_v<T>) {
        h = hint_from_int(a);
      }
      emit_val(a, specialize ? h : SpecHint::NONE, specialize, pack, specs);
    } else {
      static_assert(sizeof(A) == 0,
                    "triton_jit: unsupported argument type; pass a Tensor, "
                    "optional<Tensor>, "
                    "nullptr, or an arithmetic scalar");
    }
  }

  std::string path_;
  std::string name_;
  std::unordered_map<std::string, std::unique_ptr<CompiledKernel>> kernels_;
  std::mutex kernels_mutex_;
  std::vector<ParamInfo> sig_;
  std::once_flag sig_once_;
  bool autotune_on_ = false;
  std::vector<LaunchCfg> autotune_space_;
  std::unordered_map<std::string, LaunchCfg> winners_;
  std::mutex winners_mutex_;
};

}  // namespace xllm::triton_jit
