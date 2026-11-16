# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

function(xllm_configure_npu_runtime target)
  foreach(property BUILD_RPATH INSTALL_RPATH)
    get_property(existing TARGET ${target} PROPERTY ${property})
    set_property(TARGET ${target} PROPERTY ${property}
      "$ORIGIN/../xllm_kernel;$ORIGIN/opp/vendors/custom_xllm_math/op_api/lib;${existing}")
  endforeach()
  get_target_property(type ${target} TYPE)
  if(type STREQUAL "EXECUTABLE")
    # Direct launches resolve DT_NEEDED before main. DT_RPATH makes the
    # packaged API win over a stale vendor in startup LD_LIBRARY_PATH.
    # Python extensions keep their default dtags and use the Python preload.
    target_link_options(${target} PRIVATE "LINKER:--disable-new-dtags")
  endif()
endfunction()
