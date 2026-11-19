#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(RowColInt8Quant)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(qRow, ge::TensorType::ALL())
    .OUTPUT(qCol, ge::TensorType::ALL())
    .OUTPUT(rowAbsmax, ge::TensorType::ALL())
    .OUTPUT(colAbsmax, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(RowColInt8Quant);

}

#endif
