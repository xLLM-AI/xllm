#include "causal_conv1d/op_kernel/causal_conv1d_fn.h"
#include "causal_conv1d/op_kernel/causal_conv1d_update.h"

namespace {

template <typename T,
          uint32_t runModeKey,
          uint32_t widthKey,
          uint32_t fnPlanKey>
__aicore__ inline void RunCausalConv1dQkv(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR convStates,
    GM_ADDR queryStartLoc,
    GM_ADDR cacheIndices,
    GM_ADDR initialStateMode,
    GM_ADDR numAcceptedTokens,
    GM_ADDR y,
    GM_ADDR workspace,
    const CausalConv1dTilingData* tilingData) {
  if constexpr (runModeKey == CAUSAL_CONV1D_TPL_RUN_MODE_FN) {
    NsCausalConv1d::RunCausalConv1dFn<T, widthKey, fnPlanKey>(x,
                                                              weight,
                                                              bias,
                                                              convStates,
                                                              queryStartLoc,
                                                              cacheIndices,
                                                              initialStateMode,
                                                              numAcceptedTokens,
                                                              y,
                                                              workspace,
                                                              tilingData);
  } else {
    NsCausalConv1d::RunCausalConv1dUpdate<T>(x,
                                             weight,
                                             bias,
                                             convStates,
                                             queryStartLoc,
                                             cacheIndices,
                                             initialStateMode,
                                             numAcceptedTokens,
                                             y,
                                             workspace,
                                             tilingData);
  }
}

}  // namespace

template <uint32_t runModeKey, uint32_t widthKey, uint32_t fnPlanKey>
__global__ __aicore__ void xllm_kernel_causal_conv1d_qkv(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR convStates,
    GM_ADDR queryStartLoc,
    GM_ADDR cacheIndices,
    GM_ADDR initialStateMode,
    GM_ADDR numAcceptedTokens,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  REGISTER_TILING_DEFAULT(CausalConv1dTilingData);
  GET_TILING_DATA(tilingData, tiling);
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  GM_ADDR userWorkspace = workspace;
  if (workspace != nullptr) {
    userWorkspace = AscendC::GetUserWorkspace(workspace);
  }

  RunCausalConv1dQkv<DTYPE_X, runModeKey, widthKey, fnPlanKey>(
      x,
      weight,
      bias,
      convStates,
      queryStartLoc,
      cacheIndices,
      initialStateMode,
      numAcceptedTokens,
      y,
      userWorkspace,
      &tilingData);
}
