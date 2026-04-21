#ifndef SOFTPLUS_TILING_H
#define SOFTPLUS_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
TILING_DATA_FIELD_DEF(uint32_t, N1);
TILING_DATA_FIELD_DEF(uint32_t, N2);
TILING_DATA_FIELD_DEF(uint32_t, N3);
TILING_DATA_FIELD_DEF(uint32_t, N4);
TILING_DATA_FIELD_DEF(uint32_t, tilingKey);  // 用于区分不同的计算类型：0=mean/sum, 2=mean/sum+广播, 200/201/202=mean/sum+1D广播
TILING_DATA_FIELD_DEF(uint32_t, dataType);  // 数据类型：0=fp16, 1=fp32, 2=bf16
TILING_DATA_FIELD_DEF(uint32_t, size_average);  // size_average 属性：0=false, 1=true，默认值 1
TILING_DATA_FIELD_DEF(uint32_t, reduce);  // reduce 属性：0=false, 1=true，默认值 1
TILING_DATA_FIELD_DEF(uint32_t, reduction);  // reduction 属性：1='mean', 2='sum'，默认值 1
TILING_DATA_FIELD_DEF(uint32_t, collapsedDimNum);  // 合并后的维度数量
TILING_DATA_FIELD_DEF_ARR(uint32_t, 4, dim1);  // 合并后的 x 维度数组
TILING_DATA_FIELD_DEF_ARR(uint32_t, 4, dim2);  // 合并后的 target 维度数组
// 通用广播参数（借鉴 fmax）
TILING_DATA_FIELD_DEF(uint32_t, totalLength);  // 输出总元素数
TILING_DATA_FIELD_DEF(uint32_t, dim);  // 输出维度数
TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, shape);  // 输出形状
TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, reduce1);  // x 的广播标记（1=需要广播）
TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, reduce2);  // target 的广播标记（1=需要广播）
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(L1Loss, TilingData)
} // namespace optiling
#endif // SOFTPLUS_TILING_H

