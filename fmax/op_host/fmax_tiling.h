#ifndef FMAX_TILING_H
#define FMAX_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FmaxTilingData)
  // --- Block 维度切分参数 ---
  TILING_DATA_FIELD_DEF(uint32_t, bigDataCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallBlockLength);
  TILING_DATA_FIELD_DEF(uint32_t, bigBlockLength);

  // --- Tile 维度切分参数 ---
  TILING_DATA_FIELD_DEF(uint32_t, smallTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallTileLength);
  TILING_DATA_FIELD_DEF(uint32_t, smallLasttileLength);
  TILING_DATA_FIELD_DEF(uint32_t, bigTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigTileLength);
  TILING_DATA_FIELD_DEF(uint32_t, bigLasttileLength);

  // --- Broadcast 广播参数 ---
  // 使用 uint64_t 避免 totalLength 在高维场景溢出
  TILING_DATA_FIELD_DEF(uint64_t, totalLength);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, shape);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, reduce1);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, reduce2);
  TILING_DATA_FIELD_DEF(uint32_t, dim);
  TILING_DATA_FIELD_DEF_ARR(int,2, dim1);
  TILING_DATA_FIELD_DEF_ARR(int,2, dim2);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Fmax, FmaxTilingData)

} // namespace optiling

#endif // FMAX_TILING_H