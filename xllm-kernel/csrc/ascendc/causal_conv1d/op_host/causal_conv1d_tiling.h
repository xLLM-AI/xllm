#ifndef CAUSAL_CONV1D_HOST_TILING_H
#define CAUSAL_CONV1D_HOST_TILING_H

#include "register/op_impl_registry.h"

namespace optiling {

ge::graphStatus CausalConv1dTilingFunc(gert::TilingContext* context);
ge::graphStatus TilingParseForCausalConv1d(gert::TilingParseContext* context);

}  // namespace optiling

#endif  // CAUSAL_CONV1D_HOST_TILING_H
