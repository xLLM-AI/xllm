/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "core/framework/request/request_state.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/sampling_params.h"
#include "disagg_pd.pb.h"

namespace xllm {

RequestSamplingParam make_disagg_sampling_param(
    const proto::DisaggRequest& req);

SchedulerParam make_disagg_scheduler_param(const proto::DisaggRequest& req);

StoppingChecker make_disagg_stopping_checker(const proto::DisaggRequest& req);

}  // namespace xllm
