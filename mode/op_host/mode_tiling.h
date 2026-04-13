#ifndef MODE_TILING_H
#define MODE_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ModeTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalTasks);     // 总共需要计算多少个输出结果
    TILING_DATA_FIELD_DEF(uint32_t, axisLen);        // 目标轴的长度 (N2)
    TILING_DATA_FIELD_DEF(uint32_t, innerStride);    // 目标轴上相邻元素的物理间隔
    TILING_DATA_FIELD_DEF(uint32_t, tasksPerBlock);  // 基础任务数（小核任务数）
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);     // 大核数量
    TILING_DATA_FIELD_DEF(uint32_t, dtype);          // 数据类型
    TILING_DATA_FIELD_DEF(uint32_t, maxChunkSize);   // Host端动态计算的最大分块大小
    TILING_DATA_FIELD_DEF(uint32_t, tmpSize);        // Sort所需的临时空间大小
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Mode, ModeTilingData)
}
#endif // MODE_TILING_H