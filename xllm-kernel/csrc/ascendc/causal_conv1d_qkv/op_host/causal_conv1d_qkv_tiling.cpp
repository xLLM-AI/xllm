#include "causal_conv1d/op_host/causal_conv1d_tiling.h"
#include "causal_conv1d/op_host/causal_conv1d_tiling_utils.h"

namespace optiling {

IMPL_OP_OPTILING(XllmKernelCausalConv1dQkv)
    .Tiling(CausalConv1dTilingFunc)
    .TilingParse<causal_conv1d_host::CausalConv1dCompileInfo>(
        TilingParseForCausalConv1d);

}  // namespace optiling
