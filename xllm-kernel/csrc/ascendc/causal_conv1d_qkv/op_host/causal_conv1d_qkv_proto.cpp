#include "common/error/tiling_error_log.h"
#include "register/op_impl_registry.h"

namespace ops {

static ge::graphStatus InferShapeCausalConv1dQkv(
    gert::InferShapeContext* context) {
  const gert::Shape* xShape = context->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
  gert::Shape* yShape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
  *yShape = *xShape;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(XllmKernelCausalConv1dQkv)
    .InferShape(InferShapeCausalConv1dQkv);

}  // namespace ops
