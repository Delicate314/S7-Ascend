#include "register/tilingdata_base.h"

namespace optiling
{
  BEGIN_TILING_DATA_DEF(PdistGradTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, n);           // 4 bytes
  TILING_DATA_FIELD_DEF(uint32_t, m);           // 4 bytes
  TILING_DATA_FIELD_DEF(float, p);              // 4 bytes
  TILING_DATA_FIELD_DEF(uint32_t, tile_m_ub);   // 4 bytes
  TILING_DATA_FIELD_DEF(uint32_t, pad0);        // 4 bytes
  TILING_DATA_FIELD_DEF(uint32_t, pad1);        // 4 bytes
  TILING_DATA_FIELD_DEF(uint32_t, pad2);        // 4 bytes
  TILING_DATA_FIELD_DEF(uint32_t, pad3);        // 4 bytes -> Total 32 bytes
  END_TILING_DATA_DEF;

  REGISTER_TILING_DATA_CLASS(PdistGrad, PdistGradTilingData)
}