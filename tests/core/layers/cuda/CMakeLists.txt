include(cc_test)


cc_test(
  NAME
    rwkv7_decoder_layer_test
  SRCS
    rwkv7_decoder_layer_test.cpp
  DEPS
    :config
    :layers
    :common_layers
    :kv_cache
    :model_context
    :state_dict
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(rwkv7_decoder_layer_test
                      PRIVATE
                      brpc
                      "$<LINK_GROUP:RESCAN,layers,common_layers>")
