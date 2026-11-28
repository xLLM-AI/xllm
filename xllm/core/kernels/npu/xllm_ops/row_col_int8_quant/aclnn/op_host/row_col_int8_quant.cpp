#include "../op_kernel/row_col_int8_quant_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cmath>
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context){
 auto* tiling=context->GetTilingData<RowColInt8QuantTilingData>();
 
  const auto& shape=context->GetInputShape(0)->GetStorageShape();
  const auto dtype=context->GetInputDesc(0)->GetDataType();
  if (dtype!=ge::DT_FLOAT16 && dtype!=ge::DT_BF16) return ge::GRAPH_FAILED;

  if(dtype!=ge::DT_FLOAT16 || shape.GetDimNum()!=2) return ge::GRAPH_FAILED;
  const int64_t m=shape.GetDim(0),n=shape.GetDim(1);
  if(m<1 || m>4096 || n<2 || n>4096 || n%2) return ge::GRAPH_FAILED;
  tiling->a=m;tiling->b=n;
  const uint64_t tiles=((m+63)/64)*((n+255)/256);
  context->SetBlockDim(std::min<uint64_t>(48,tiles));
  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  context->GetWorkspaceSizes(1)[0]=tiles*(64+256)*sizeof(float)+platform.GetLibApiWorkSpaceSize();

 return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context){const auto& x=*context->GetInputShape(0);
  if(x.GetDimNum()!=2)return ge::GRAPH_FAILED;
  *context->GetOutputShape(0)=x;*context->GetOutputShape(1)=x;
  context->GetOutputShape(2)->SetDimNum(1);context->GetOutputShape(2)->SetDim(0,x.GetDim(0));
  context->GetOutputShape(3)->SetDimNum(1);context->GetOutputShape(3)->SetDim(0,x.GetDim(1)); return ge::GRAPH_SUCCESS;}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context){context->SetOutputDataType(0,ge::DT_INT8);context->SetOutputDataType(1,ge::DT_INT8);context->SetOutputDataType(2,ge::DT_FLOAT);context->SetOutputDataType(3,ge::DT_FLOAT);return ge::GRAPH_SUCCESS;}
}
namespace ops {
class RowColInt8Quant : public OpDef {
public:
    explicit RowColInt8Quant(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("qRow")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("qCol")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("rowAbsmax")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("colAbsmax")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910_93");

    }
};

OP_ADD(RowColInt8Quant);
}

