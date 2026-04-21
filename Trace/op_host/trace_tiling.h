#ifndef TRACE_TILING_H
#define TRACE_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
TILING_DATA_FIELD_DEF(uint32_t, N1);
TILING_DATA_FIELD_DEF(uint32_t, N2);
TILING_DATA_FIELD_DEF(uint32_t, tilingKey);  
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Trace, TilingData)
} // namespace optiling
#endif // TRACE_TILING_H
