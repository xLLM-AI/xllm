include(cc_library)

cc_library(
  NAME 
    sampler
  HDRS
    npu_sampling_policy.h
    sampling_params.h
    logits_processor.h
    sampler.h
    rec_sampler.h
    beam_searcher.h
  SRCS
    sampling_params.cpp
    logits_processor.cpp
    sampler.cpp
    rec_sampler.cpp
    beam_searcher.cpp
  DEPS
    util
    :common
    :config
    glog::glog
    torch
    :kernels
    $<$<BOOL:${USE_NPU}>:xllm_ops>
    $<$<BOOL:${USE_NPU}>:tilelang_kernels>
)


if(BUILD_NPU_SAMPLING_ABLATION)
  include(cc_binary)
  cc_binary(
    NAME npu_sampling_benchmark
    SRCS npu_sampling_benchmark.cpp
    DEPS :sampler :speculative :config :platform :npu_startup torch torch_npu
         benchmark::benchmark glog::glog Python::Python
         brpc OpenSSL::SSL OpenSSL::Crypto leveldb::leveldb protobuf::libprotobuf
         ascendcl unified_dlog hccl c_sec nnopbase
  )
  add_dependencies(npu_sampling_benchmark brpc-static)
endif()
