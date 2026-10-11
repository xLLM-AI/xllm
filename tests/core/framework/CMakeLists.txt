include(cc_test)

add_subdirectory(allocator)
add_subdirectory(batch)
add_subdirectory(block)
add_subdirectory(chat_template)
add_subdirectory(config)
add_subdirectory(encoder_cache)
add_subdirectory(eplb)
add_subdirectory(kv_cache)
add_subdirectory(model_loader)
if(USE_NPU OR USE_MLU)
  add_subdirectory(transfer)
endif()
add_subdirectory(parallel_state)
add_subdirectory(request)
add_subdirectory(sampling)
add_subdirectory(speculative)
add_subdirectory(state_dict)
add_subdirectory(tokenizer)

cc_test(
  NAME
    hf_model_loader_test
  SRCS
    hf_model_loader_test.cpp
  DEPS
    :model_loader
    :xllm_server
    GTest::gtest_main
)

cc_test(
  NAME
    causal_lm_test
  SRCS
    causal_lm_test.cpp
  DEPS
    :model_loader
    :xllm_server
    GTest::gtest_main
)
