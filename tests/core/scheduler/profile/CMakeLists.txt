include(cc_test)

cc_test(
  NAME
    profile_graph_warmup_test
  SRCS
    profile_graph_warmup_test.cpp
  DEPS
    :xllm_server
    :block
    :profile
    :request
    GTest::gtest_main
)
target_link_libraries(profile_graph_warmup_test PRIVATE
  "$<LINK_GROUP:RESCAN,xtensor,xllm_server>")
