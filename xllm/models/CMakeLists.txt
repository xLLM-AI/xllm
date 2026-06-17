include(cc_library)

add_subdirectory(vlm/mposition)

if(USE_NPU)
  include_directories(
    ${CMAKE_SOURCE_DIR}/third_party/spdlog/include
  )
endif()

cc_library(
  NAME
    models
  HDRS
    model_registry.h
    models.h
    dit/utils/lanczos_resample.h
  SRCS
    model_registry.cpp
    dit/utils/lanczos_resample.cpp
    vlm/utils/multimodal_utils.cpp
  DEPS
    :model
    :mposition
    :parallel_state
    :model_context
    :dit_cache
    :model_loader
    :layers
    $<$<BOOL:${USE_NPU}>:npu_layers>
)
