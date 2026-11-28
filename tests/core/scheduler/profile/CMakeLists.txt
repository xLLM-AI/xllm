include(cc_test)

cc_test(
  NAME
    profile_graph_warmup_test
  SRCS
    profile_graph_warmup_test.cpp
  DEPS
    :xllm_server
    :block
    :kv_cache_manager
    :profile
    :platform
    :request
    GTest::gtest_main
)
