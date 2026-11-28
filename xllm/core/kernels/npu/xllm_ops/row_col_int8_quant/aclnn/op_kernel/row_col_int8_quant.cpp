#include "kernel_impl.h"
#include "row_col_int8_quant_tiling.h"
extern "C" __global__ __aicore__ void row_col_int8_quant(GM_ADDR x,GM_ADDR qRow,GM_ADDR qCol,GM_ADDR rowAbsmax,GM_ADDR colAbsmax,GM_ADDR workspace,GM_ADDR tiling){
 REGISTER_TILING_DEFAULT(RowColInt8QuantTilingData);
 GET_TILING_DATA(t,tiling);
 SetSysWorkspace(workspace);
 GM_ADDR pr=GetUserWorkspace(workspace);
 uint64_t tiles=((t.a+63)/64)*((t.b+255)/256);
 GM_ADDR pc=pr+tiles*64*sizeof(float);
 {RowColStatsKernel k;k.Init(x,pr,pc,t.a,t.b);k.Process();}
 SyncAll();
 RowColMerge(pr,pc,rowAbsmax,colAbsmax,t.a,t.b);
 SyncAll();
 {RowColQuantKernel k;k.Init(x,rowAbsmax,colAbsmax,qRow,qCol,t.a,t.b);k.Process();}
}
