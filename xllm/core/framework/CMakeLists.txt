include(cc_library)

include_directories(.)
if(USE_NPU)
  include_directories(
    ${CMAKE_SOURCE_DIR}/third_party/spdlog/include
  )
endif()
add_subdirectory(batch)
add_subdirectory(xtensor)
add_subdirectory(config)
add_subdirectory(chat_template)
add_subdirectory(kv_cache)
add_subdirectory(model)
add_subdirectory(model_context)
add_subdirectory(model_loader)
add_subdirectory(parallel_state)
add_subdirectory(multimodal)
add_subdirectory(encoder_cache)
add_subdirectory(state_dict)
add_subdirectory(quantization)
add_subdirectory(tokenizer)
add_subdirectory(sampling)
add_subdirectory(request)
add_subdirectory(eplb)
add_subdirectory(dit_cache)
add_subdirectory(speculative)

# Factory orchestration may query WorkerClient and XTensor, while :block and
# :kv_cache_estimation remain independent of the runtime implementation.
cc_library(
  NAME
    kv_cache_manager_factory
  HDRS
    block/kv_cache_manager_factory.h
  SRCS
    block/kv_cache_manager_factory.cpp
  DEPS
    :block
    :kv_cache
    :xtensor
    :common
    :config
    :runtime
    :util
    Folly::folly
    :models
)
