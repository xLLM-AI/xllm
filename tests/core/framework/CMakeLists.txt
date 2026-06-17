include(cc_test)

add_subdirectory(batch)
add_subdirectory(block)
add_subdirectory(chat_template)
add_subdirectory(config)
add_subdirectory(encoder_cache)
add_subdirectory(eplb)
add_subdirectory(kv_cache)
add_subdirectory(kv_cache_transfer)
add_subdirectory(parallel_state)
add_subdirectory(prefix_cache)
add_subdirectory(request)
add_subdirectory(sampling)
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
target_link_libraries(hf_model_loader_test PRIVATE
  "$<LINK_GROUP:RESCAN,xtensor,xllm_server>")
