#ifndef SEGMENTREDUCE_TILING_H
#define SEGMENTREDUCE_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

  // SegmentReduce 算子的 tiling 数据结构
  BEGIN_TILING_DATA_DEF(SegmentReduceGradTilingData)
    // 数据维度信息
    TILING_DATA_FIELD_DEF(uint32_t, N);              // 输入数据第一维大小
  TILING_DATA_FIELD_DEF(uint32_t, N2);             // 输入数据第二维大小
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, K);      // 段数量数组（大小为20）
  TILING_DATA_FIELD_DEF(uint32_t, dim_K);          // K 数组的实际维度（实际使用的元素数量）
  TILING_DATA_FIELD_DEF(int32_t, axis);            // 归约轴

  // 数据布局信息
  TILING_DATA_FIELD_DEF(uint32_t, outerOffset);    // 外部维度偏移
  TILING_DATA_FIELD_DEF(uint32_t, innerOffset);    // 内部维度偏移

  // Reduce 类型和其他参数
  TILING_DATA_FIELD_DEF(uint32_t, reduceType);     // 0: SUM, 1: MEAN, 2: MAX, 3: MIN, 4: PROD
  TILING_DATA_FIELD_DEF(uint32_t, useOffsets);     // 0: lengths, 1: offsets
  TILING_DATA_FIELD_DEF(uint32_t, hasInitial);     // 是否提供初始值

  // Host 预计算的标量，减少 kernel 端标量开销
  TILING_DATA_FIELD_DEF(uint32_t, shardsPerSeg);   // 每 segment 的 shard 数
  TILING_DATA_FIELD_DEF(uint32_t, maxTileLength);  // dtype+reduceType 决定的 tile 容量
  TILING_DATA_FIELD_DEF(uint32_t, rowSizeBytes);   // N2 * sizeof(dtype)
  TILING_DATA_FIELD_DEF(uint32_t, rowComputeLen);  // RoundUp(N2, 64)，用于 Compare/Select
  END_TILING_DATA_DEF;

  REGISTER_TILING_DATA_CLASS(SegmentReduceGrad, SegmentReduceGradTilingData)

} // namespace optiling

#endif 