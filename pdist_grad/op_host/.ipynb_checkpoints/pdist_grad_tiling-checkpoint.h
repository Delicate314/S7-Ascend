#include "register/tilingdata_base.h"

namespace optiling
{
  BEGIN_TILING_DATA_DEF(PdistGradTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, n);           // 样本数量 N
  TILING_DATA_FIELD_DEF(uint32_t, m);           // 特征维度 M
  TILING_DATA_FIELD_DEF(float, p);              // p范数
  TILING_DATA_FIELD_DEF(uint32_t, m_per_core);  // 每个核分配的基础列数
  TILING_DATA_FIELD_DEF(uint32_t, tail_m_core); // 最后一个核的列数
  TILING_DATA_FIELD_DEF(uint32_t, copym);       // 对齐后的处理长度
  END_TILING_DATA_DEF;

  REGISTER_TILING_DATA_CLASS(PdistGrad, PdistGradTilingData)
}