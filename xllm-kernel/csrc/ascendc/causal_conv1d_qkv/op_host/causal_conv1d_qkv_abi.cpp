// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include "aclnn_xllm_kernel_causal_conv1d_qkv.h"

extern "C" {

__attribute__((visibility("default"))) aclnnStatus
aclnnCausalConv1dQkvGetWorkspaceSize(const aclTensor* x,
                                     const aclTensor* weight,
                                     const aclTensor* bias,
                                     const aclTensor* conv_states,
                                     const aclIntArray* query_start_loc,
                                     const aclIntArray* cache_indices,
                                     const aclIntArray* initial_state_mode,
                                     const aclIntArray* num_accepted_tokens,
                                     int64_t activation_mode,
                                     int64_t pad_slot_id,
                                     int64_t run_mode,
                                     int64_t q_dim,
                                     int64_t k_dim,
                                     int64_t v_dim,
                                     int64_t head_dim,
                                     const aclTensor* y,
                                     uint64_t* workspace_size,
                                     aclOpExecutor** executor) {
  return aclnnXllmKernelCausalConv1dQkvGetWorkspaceSize(x,
                                                        weight,
                                                        bias,
                                                        conv_states,
                                                        query_start_loc,
                                                        cache_indices,
                                                        initial_state_mode,
                                                        num_accepted_tokens,
                                                        activation_mode,
                                                        pad_slot_id,
                                                        run_mode,
                                                        q_dim,
                                                        k_dim,
                                                        v_dim,
                                                        head_dim,
                                                        y,
                                                        workspace_size,
                                                        executor);
}

__attribute__((visibility("default"))) aclnnStatus
aclnnCausalConv1dQkv(void* workspace,
                     uint64_t workspace_size,
                     aclOpExecutor* executor,
                     aclrtStream stream) {
  return aclnnXllmKernelCausalConv1dQkv(
      workspace, workspace_size, executor, stream);
}
}
