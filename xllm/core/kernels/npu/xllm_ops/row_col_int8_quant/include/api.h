#pragma once
#include <cstdint>
extern "C" void row_col_stats_launch(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint32_t,uint32_t);
extern "C" void row_col_merge_launch(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint32_t,uint32_t);
extern "C" void row_col_quant_launch(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint32_t,uint32_t);
