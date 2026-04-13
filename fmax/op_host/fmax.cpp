#include "fmax_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <vector>
#include <algorithm>

namespace optiling {

    enum BroadcastType {
        BCAST_NONE = 0, // N vs N
        BCAST_X = 1,    // 1 vs N (X广播)
        BCAST_Y = 2     // N vs 1 (Y广播)
    };

    const uint32_t BLOCK_SIZE = 32;
    constexpr int32_t CALC_ALIGN_NUM = 256;
    constexpr int32_t ALIGN_NUM = 32;

    // 向上取整
    inline uint64_t RoundUp(uint64_t a, uint64_t b) {
        return (b == 0) ? 0 : ((a + b - 1) / b) * b;
    }

    // 【核心修复1】形状右对齐处理，防止 6D 访问 1D 时越界
    // 输入: shape=[3,4], outDim=4 -> 输出 vector=[1, 1, 3, 4]
    static void GetTensorShape(const gert::StorageShape* shape, std::vector<uint64_t>& inshapeVector, uint32_t outDimNum) {
        int n = outDimNum - 1;
        int32_t dimNum = shape->GetStorageShape().GetDimNum();
        for (int j = dimNum - 1; j >= 0; --j) {
            if (n >= 0) {
                inshapeVector[n--] = shape->GetStorageShape().GetDim(j);
            }
        }
    }

    static void SetBroadCastParams(const std::vector<uint64_t>& inshapeVector1,
        const std::vector<uint64_t>& inshapeVector2,
        const gert::Shape& outshape,
        FmaxTilingData& tiling) {
        uint32_t outDimNum = outshape.GetDimNum();

        // 局部暂存数组
        uint32_t tmp_shape[20] = { 0 };
        uint32_t tmp_reduce1[20] = { 0 };
        uint32_t tmp_reduce2[20] = { 0 };

        for (uint32_t i = 0; i < outDimNum; i++) {
            tmp_shape[i] = static_cast<uint32_t>(outshape.GetDim(i));
            // 因为 vector 已经对齐，可以直接比较，安全
            if (inshapeVector1[i] != outshape.GetDim(i)) tmp_reduce1[i] = 1;
            if (inshapeVector2[i] != outshape.GetDim(i)) tmp_reduce2[i] = 1;
        }

        // 写入 Tiling Data
        tiling.set_shape(tmp_shape);
        tiling.set_reduce1(tmp_reduce1);
        tiling.set_reduce2(tmp_reduce2);
        tiling.set_dim(outDimNum);
    }

    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        FmaxTilingData tiling;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto aivNum = ascendcPlatform.GetCoreNumAiv();

        uint64_t ub_size;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 获取输出形状
        auto outshape = context->GetOutputShape(0)->GetOriginShape();
        uint32_t outDimNum = outshape.GetDimNum();
        uint32_t originDimNum = outshape.GetDimNum();
        // 计算总长度，使用 uint64 防止溢出
        uint64_t totalLength = 1;
        for (uint32_t i = 0; i < outDimNum; i++) {
            totalLength *= outshape.GetDim(i);
        }

        // 获取输入形状并对齐
        const gert::StorageShape* shape1 = context->GetInputShape(0);

        const gert::StorageShape* shape2 = context->GetInputShape(1);

        std::vector<uint64_t> inshapeVector1(outDimNum, 1);
        std::vector<uint64_t> inshapeVector2(outDimNum, 1);
        GetTensorShape(shape1, inshapeVector1, outDimNum);
        GetTensorShape(shape2, inshapeVector2, outDimNum);

        // printf("outDimNum= %d\n", outDimNum);
        // 判定广播：使用对齐后的 vector 比较
        bool needBroadcast = false;
        for (uint32_t i = 0; i < outDimNum; i++) {
            // printf("inshapeVector1[%d]=%d, inshapeVector2[%d]=%d\n",i,inshapeVector1[i],i,inshapeVector2[i]);
            if (inshapeVector1[i] != inshapeVector2[i]) {
                needBroadcast = true;
                break;
            }
        }

        // printf("needBroadcast: %d\n",needBroadcast);
        // 获取数据类型相关参数
        auto dt = context->GetInputTensor(0)->GetDataType();
        uint32_t dataWidth = 0;
        uint32_t dtypeOffset = 0;

        switch (dt) {
        case ge::DT_FLOAT16: dataWidth = 2; dtypeOffset = 0; break;
        case ge::DT_FLOAT:   dataWidth = 4; dtypeOffset = 2; break;
        case ge::DT_INT32:   dataWidth = 4; dtypeOffset = 4; break;
        case ge::DT_INT8:    dataWidth = 1; dtypeOffset = 6; break;
        case ge::DT_INT16:   dataWidth = 2; dtypeOffset = 8; break;
        case ge::DT_INT64:   dataWidth = 8; dtypeOffset = 10; break;
        case ge::DT_UINT8:   dataWidth = 1; dtypeOffset = 12; break;
        case ge::DT_BF16:    dataWidth = 2; dtypeOffset = 14; break;
        case ge::DT_BOOL:    dataWidth = 1; dtypeOffset = 16; break;
        default: return ge::GRAPH_FAILED;
        }

        uint64_t elementsPerBlock = ALIGN_NUM / dataWidth;
        uint64_t elementsPerRepeat = CALC_ALIGN_NUM / dataWidth;
        uint64_t totalLengthBlockAlign = RoundUp(totalLength, elementsPerBlock);

        // 核数分配
        if (totalLengthBlockAlign / elementsPerBlock < aivNum) {
            aivNum = totalLengthBlockAlign / elementsPerBlock;
        }
        if (aivNum < 1) aivNum = 1;

        // Block 计算
        uint32_t smallBlockLength = (totalLengthBlockAlign / aivNum / elementsPerBlock) * elementsPerBlock;
        uint32_t bigDataCoreNum = (totalLengthBlockAlign / elementsPerBlock) % aivNum;
        uint32_t bigBlockLength = (bigDataCoreNum > 0) ? smallBlockLength + elementsPerBlock : 0;

        // Tile 计算 (UB 切分)
        uint32_t ubfactor = 6;
        // 留足余量，BufferNum=2
        uint32_t maxTileCalcBlock = ub_size / (CALC_ALIGN_NUM * ubfactor) / 2;
        uint32_t maxTileLength = maxTileCalcBlock * (CALC_ALIGN_NUM / dataWidth);
        if (maxTileLength == 0) maxTileLength = elementsPerRepeat;

        // Big Core Tile
        uint32_t bigTileLength = 0, bigBlockLengthCalcAlign = 0, bigTileNum = 0, bigLasttileLength = 0;
        if (bigDataCoreNum > 0) {
            bigBlockLengthCalcAlign = RoundUp(bigBlockLength, elementsPerRepeat);
            bigTileLength = (bigBlockLengthCalcAlign > maxTileLength) ? maxTileLength : bigBlockLengthCalcAlign;
            // 简单防守策略
            if (bigTileLength >= (CALC_ALIGN_NUM * 2 / dataWidth)) bigTileLength /= 2;

            bigTileNum = (bigBlockLengthCalcAlign + bigTileLength - 1) / bigTileLength;
            bigLasttileLength = bigBlockLength - (bigTileNum - 1) * bigTileLength;
        }

        // Small Core Tile
        uint32_t smallTileLength = 0, smallBlockLengthCalcAlign = 0, smallTileNum = 0, smallLasttileLength = 0;
        if (bigDataCoreNum < aivNum) {
            smallBlockLengthCalcAlign = RoundUp(smallBlockLength, elementsPerRepeat);
            smallTileLength = (smallBlockLengthCalcAlign > maxTileLength) ? maxTileLength : smallBlockLengthCalcAlign;
            if (smallTileLength >= (CALC_ALIGN_NUM * 2 / dataWidth)) smallTileLength /= 2;

            smallTileNum = (smallBlockLengthCalcAlign + smallTileLength - 1) / smallTileLength;
            smallLasttileLength = smallBlockLength - (smallTileNum - 1) * smallTileLength;
        }

        // 设置 Tiling Key
        uint32_t baseTilingKey = needBroadcast ? 2 : 1;
        uint32_t tilingKey = baseTilingKey + dtypeOffset;


        std::vector<int64_t> s1, s2;
        for (uint32_t i = 0; i < originDimNum; ++i) {
            s1.push_back(shape1->GetOriginShape().GetDim(i));
            s2.push_back(shape2->GetOriginShape().GetDim(i));
        }

        // 2. 轴折叠逻辑 (Axis Collapsing)
        // ---------------------------------------------------------
        std::vector<int64_t> foldS1, foldS2;

        // 临时变量，用于记录正在合并的维度
        int64_t curDim1 = 1;
        int64_t curDim2 = 1;
        int lastBcastType = -1; // -1: 初始状态

        // 遍历所有维度
        for (uint32_t i = 0; i < originDimNum; ++i) {
            int64_t d1 = s1[i];
            int64_t d2 = s2[i];

            // 策略：去除双方都为 1 的轴 (对计算无影响，只影响 offset 且 stride 为 0)
            // 注意：如果最后所有轴都被去掉了（比如输入全是 1），要在循环外补一个 [1]
            if (d1 == 1 && d2 == 1) {
                continue;
            }

            // 判定当前轴的广播类型
            int currentBcastType = BCAST_NONE;
            if (d1 == d2) {
                currentBcastType = BCAST_NONE;
            }
            else if (d1 == 1 && d2 > 1) {
                currentBcastType = BCAST_X; // Input1 需要广播
            }
            else if (d1 > 1 && d2 == 1) {
                currentBcastType = BCAST_Y; // Input2 需要广播
            }
            else {
                // N vs M (N!=M)，非法广播，直接报错
                return ge::GRAPH_FAILED;
            }

            // 尝试合并
            if (lastBcastType == -1) {
                // 第一个有效轴
                lastBcastType = currentBcastType;
                curDim1 = d1;
                curDim2 = d2;
            }
            else if (currentBcastType == lastBcastType) {
                // 类型相同，合并到当前轴
                curDim1 *= d1;
                curDim2 *= d2;
            }
            else {
                // 类型不同，保存上一段，开启新的一段
                foldS1.push_back(curDim1);
                foldS2.push_back(curDim2);

                curDim1 = d1;
                curDim2 = d2;
                lastBcastType = currentBcastType;
            }
        }

        // 保存最后一段
        if (lastBcastType != -1) {
            foldS1.push_back(curDim1);
            foldS2.push_back(curDim2);
        }
        else {
            // 特殊情况：所有轴都是 1 vs 1，被全部跳过
            // 此时应视为 1个元素
            foldS1.push_back(1);
            foldS2.push_back(1);
        }

        // 3. 填充 Tiling Data
        // ---------------------------------------------------------
        uint32_t collapsedDimNum = foldS1.size();

        // 准备传给 Kernel 的定长数组 (假设最大支持 4 维折叠结果)
        int32_t dim1_array[4] = { 1, 1, 1, 1 };
        int32_t dim2_array[4] = { 1, 1, 1, 1 };


        for (size_t i = 0; i < collapsedDimNum && i < 4; ++i) {
            dim1_array[i] = (int32_t)foldS1[i];
            dim2_array[i] = (int32_t)foldS2[i];
        }

        tiling.set_dim1(dim1_array);
        tiling.set_dim2(dim2_array);


        if (collapsedDimNum == 2) {
            // 维度折叠后只有 2 维
            // 模式 1: [N, M] vs [1, M] (FirstD) -> Key 3
            // 模式 2: [N, M] vs [N, 1] (LastD)  -> Key 4 (假设你有这个 Kernel)
            // printf("foldS1[0]=%ld, foldS1[1]=%ld, foldS2[0]=%ld, foldS2[1]=%ld\n", foldS1[0], foldS1[1], foldS2[0], foldS2[1]);
            // 检查轴 0 是否广播
            bool axis0_diff = (foldS1[0] != foldS2[0]);
            // 检查轴 1 是否广播
            bool axis1_diff = (foldS1[1] != foldS2[1]);

            if (axis0_diff && !axis1_diff) {
                // 首维广播: [N, M] vs [1, M]
                tilingKey = 666;
            }
            else if (!axis0_diff && axis1_diff) {
                // 末维广播: [N, M] vs [N, 1]

            }
            else {
                // [N, M] vs [1, 1] -> 双重广播，需要更复杂的 Kernel

            }
        }

        // printf("tilingkey=%d\n", tilingKey);
        context->SetTilingKey(tilingKey);
        // 填充数据
        if (needBroadcast) {
            SetBroadCastParams(inshapeVector1, inshapeVector2, outshape, tiling);
            context->SetBlockDim(aivNum); // 广播简化为单核
        }
        else {
            context->SetBlockDim(aivNum);
        }

        tiling.set_totalLength(totalLength);
        tiling.set_bigDataCoreNum(bigDataCoreNum);
        tiling.set_smallBlockLength(smallBlockLength);
        tiling.set_bigBlockLength(bigBlockLength);
        tiling.set_smallTileNum(smallTileNum);
        tiling.set_smallTileLength(smallTileLength);
        tiling.set_smallLasttileLength(smallLasttileLength);
        tiling.set_bigTileNum(bigTileNum);
        tiling.set_bigTileLength(bigTileLength);
        tiling.set_bigLasttileLength(bigLasttileLength);

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        return ge::GRAPH_SUCCESS;
    }
} // namespace optiling

namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context) {
        const gert::Shape* in_shape = context->GetInputShape(0);
        gert::Shape* out_shape = context->GetOutputShape(0);
        *out_shape = *in_shape;
        return GRAPH_SUCCESS;
    }
}

namespace ops {
    class Fmax : public OpDef {
    public:
        explicit Fmax(const char* name) : OpDef(name) {
            this->Input("input")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8,
                           ge::DT_INT16, ge::DT_INT64, ge::DT_UINT8, ge::DT_BF16,
                           ge::DT_BOOL })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                         ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                         ge::FORMAT_ND });

            this->Input("other")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8,
                           ge::DT_INT16, ge::DT_INT64, ge::DT_UINT8, ge::DT_BF16,
                           ge::DT_BOOL })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                         ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                         ge::FORMAT_ND });

            this->Output("out")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8,
                           ge::DT_INT16, ge::DT_INT64, ge::DT_UINT8, ge::DT_BF16,
                           ge::DT_BOOL })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                         ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                         ge::FORMAT_ND });

            this->SetInferShape(ge::InferShape);
            this->AICore().SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
        }
    };
    OP_ADD(Fmax);
}