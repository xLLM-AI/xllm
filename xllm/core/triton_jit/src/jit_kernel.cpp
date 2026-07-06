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

#include "jit_kernel.h"

#include <glog/logging.h>
#include <pybind11/embed.h>

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace xllm::triton_jit {

namespace {

std::vector<ParamInfo> fetch_signature(const std::string& path,
                                       const std::string& name) {
  ensure_embedded_interpreter();
  py::gil_scoped_acquire gil;
  prepare_triton_compile_import_path();
  py::module_ mod = py::module_::import(kTritonCompileModule);
  std::vector<ParamInfo> sig;
  try {
    py::list sig_py = mod.attr("signature")(path, name);
    sig.reserve(static_cast<size_t>(sig_py.size()));
    for (py::handle entry : sig_py) {
      py::dict d = py::cast<py::dict>(entry);
      ParamInfo p;
      p.is_constexpr = d["is_constexpr"].cast<bool>();
      p.do_not_specialize = d["do_not_specialize"].cast<bool>();
      sig.push_back(p);
    }
  } catch (py::error_already_set& e) {
    e.restore();
    LOG(FATAL) << "triton_jit: dump-sig failed for " << name << " from " << path
               << ": " << e.what();
  }
  if (sig.empty()) {
    LOG(FATAL) << "triton_jit: dump-sig returned no parameters for " << name
               << " from " << path;
  }
  return sig;
}

#if defined(USE_MLU)
void patch_torch_mlu_accelerator() {
  // Skip torch's own backend-autoload machinery; we drive the import explicitly
  // below and only need the accelerator probe to pass.
  setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 1);
  ensure_embedded_interpreter();
  py::gil_scoped_acquire gil;
  py::module_ torch = py::module_::import("torch");
  // Stub that satisfies the accelerator probe without touching real MLU state.
  auto cpu_accelerator = [torch]() -> py::object {
    return torch.attr("device")("cpu");
  };
  py::object torch_accelerator = torch.attr("accelerator");
  py::object origin_func = torch_accelerator.attr("current_accelerator");
  // Swap in the CPU stub for the duration of the device-backend import.
  torch_accelerator.attr("current_accelerator") =
      py::cpp_function(cpu_accelerator);
  torch.attr("_import_device_backends")();
  py::module_::import("torch_mlu");
  // Restore the real accelerator so later queries see the genuine backend.
  torch_accelerator.attr("current_accelerator") = origin_func;
}
#endif
}  // namespace

JITKernel& JITKernel::get(std::string py_path, std::string fn_name) {
  static std::unordered_map<std::string, std::unique_ptr<JITKernel>> registry;
  static std::mutex registry_mutex;
  std::string key = py_path + ":" + fn_name;
  std::lock_guard<std::mutex> lock(registry_mutex);
  auto it = registry.find(key);
  if (it == registry.end()) {
    std::unique_ptr<JITKernel> p(
        new JITKernel(std::move(py_path), std::move(fn_name)));
    auto [ins, ok] = registry.emplace(key, std::move(p));
    it = ins;
  }
  return *it->second;
}

JITKernel::JITKernel(std::string py_path, std::string fn_name)
    : path_(std::move(py_path)), name_(std::move(fn_name)) {
#if defined(USE_MLU)
  static std::once_flag once;
  std::call_once(once, []() { patch_torch_mlu_accelerator(); });
#endif
}

void JITKernel::ensure_signature() {
  std::call_once(sig_once_, [this]() { sig_ = fetch_signature(path_, name_); });
}

CompiledKernel& JITKernel::compile_or_get(const SpecList& specs,
                                          const LaunchCfg& cfg,
                                          bool use_winner) {
  TritonBackend& backend = get_backend();
  int32_t dev = backend.current_device();
  LaunchCfg use_cfg = cfg;
  if (use_winner && autotune_on_) {
    std::string sig_base = serialize_key_base(specs, dev);
    std::lock_guard<std::mutex> lock(winners_mutex_);
    auto w = winners_.find(sig_base);
    if (w != winners_.end()) {
      use_cfg = w->second;
    }
  }
  std::string key = serialize_key(specs, use_cfg, dev);

  {
    std::lock_guard<std::mutex> lock(kernels_mutex_);
    auto it = kernels_.find(key);
    if (it != kernels_.end()) {
      return *it->second;
    }
  }
  LOG(INFO) << "triton_jit: compiling " << name_
            << " (backend=" << backend.name() << ", key=" << key << ")";
  std::string cache_dir = backend.compile(path_, name_, specs, use_cfg, dev);
  std::unique_ptr<CompiledKernel> compiled =
      backend.load_kernel(cache_dir, name_);
  {
    std::lock_guard<std::mutex> lock(kernels_mutex_);
    auto [ins, ok] = kernels_.emplace(key, std::move(compiled));
    return *ins->second;
  }
}

void JITKernel::set_autotune(bool on, std::vector<LaunchCfg> space) {
  if (on) {
    if (autotune_on_) {
      return;
    }
    autotune_on_ = true;
    // Each backend reports its own viable search space (e.g. MLU only supports
    // num_warps in {1,4,8,16,32}; 16/32 usually overflow SRAM). Override via
    // `space`.
    autotune_space_ = space.empty() ? get_backend().default_autotune_space()
                                    : std::move(space);
  }
}

void JITKernel::autotune_impl(void* stream,
                              Grid g,
                              const ArgPack& pack,
                              const SpecList& specs) {
  TritonBackend& backend = get_backend();
  int32_t dev = backend.current_device();
  std::string sig_base = serialize_key_base(specs, dev);
  {
    std::lock_guard<std::mutex> lock(winners_mutex_);
    if (winners_.find(sig_base) != winners_.end()) {
      return;  // already tuned
    }
  }
  if (backend.is_capturing(stream)) {
    LOG(WARNING) << "triton_jit: autotune skipped under graph capture for "
                 << name_ << "; call warmup() before begin_capture";
    return;
  }

  constexpr int kWarmup = 3;
  constexpr int kIters = 10;
  LaunchCfg best{};
  double best_ms = 1e18;
  for (LaunchCfg cfg : autotune_space_) {
    // use_winner=false so we benchmark this exact cfg, not a cached winner.
    CompiledKernel& k = compile_or_get(specs, cfg, /*use_winner=*/false);
    try {
      for (int i = 0; i < kWarmup; ++i) {
        k.launch(stream, g, pack.data());
      }
      backend.sync(stream);
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < kIters; ++i) {
        k.launch(stream, g, pack.data());
      }
      backend.sync(stream);
      auto t1 = std::chrono::steady_clock::now();
      double ms =
          static_cast<double>(
              std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                  .count()) /
          kIters / 1000.0;
      if (ms < best_ms) {
        best_ms = ms;
        best = cfg;
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "triton_jit: autotune cfg(w=" << cfg.num_warps
                   << ",s=" << cfg.num_stages
                   << ") failed, skipping: " << e.what();
    }
  }
  if (best_ms >= 1e18) {
    LOG(WARNING) << "triton_jit: autotune found no working cfg for " << name_
                 << "; keeping caller cfg";
    return;
  }
  {
    std::lock_guard<std::mutex> lock(winners_mutex_);
    winners_[sig_base] = best;
  }
  LOG(INFO) << "triton_jit: autotune " << name_
            << " winner: w=" << best.num_warps << ",s=" << best.num_stages
            << " (" << best_ms << "ms)";
}

}  // namespace xllm::triton_jit
