#include "segment_reduce_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <vector>
#include <string>

enum ReduceType {
    REDUCE_SUM = 0,
    REDUCE_MEAN = 1,
    REDUCE_MAX = 2,
    REDUCE_MIN = 3,
    REDUCE_PROD = 4
};

namespace optiling {
    // Reduce 类型字符串转枚举
    uint32_t GetReduceType(const std::string& reduce) {
        if (reduce == "sum" || reduce == "SUM") return 0;  // REDUCE_SUM
        if (reduce == "mean" || reduce == "MEAN") return 1;  // REDUCE_MEAN
        if (reduce == "max" || reduce == "MAX") return 2;  // REDUCE_MAX
        if (reduce == "min" || reduce == "MIN") return 3;  // REDUCE_MIN
        if (reduce == "prod" || reduce == "PROD") return 4;  // REDUCE_PROD
        return 0;  // 默认 SUM
    }

    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        SegmentReduceGradTilingData tiling;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto aivNum = ascendcPlatform.GetCoreNumAiv();

        // 1. 获取 data 形状 (Index 2)
        const gert::StorageShape* dataStorageShape = context->GetInputShape(2);
        const gert::Shape& dataShape = dataStorageShape->GetStorageShape();

        uint32_t N = dataShape.GetDim(0);
        uint32_t N2 = (dataShape.GetDimNum() > 1) ? dataShape.GetDim(1) : 1;
        uint32_t dim_K = 0;
        // 2. 获取属性
        std::string reduceStr = "sum";
        const char* reducePtr = context->GetAttrs()->GetStr(0);
        if (reducePtr != nullptr) reduceStr = std::string(reducePtr);
        uint32_t reduceType = GetReduceType(reduceStr);

        int32_t axis = 0;
        const int64_t* axisPtr = context->GetAttrs()->GetInt(1);
        if (axisPtr != nullptr) axis = static_cast<int32_t>(*axisPtr);

        // 3. 处理 Lengths (Index 3) / Offsets (Index 4)
        uint32_t useOffsets = 0;

        // 检查 lengths 是否存在
        const gert::Tensor* lengthsTensor = context->GetInputTensor(3);
        bool hasLengths = (lengthsTensor != nullptr);

        // 调试打印 (可选)
        // printf("hasLengths = %d\n", hasLengths);

        if (hasLengths) {
            useOffsets = 0;
            const gert::StorageShape* lengthsStorageShape = context->GetInputShape(3);

            const gert::Shape& lengthsShape = lengthsStorageShape->GetStorageShape();
            if (lengthsShape.GetDimNum() == 2) {
                dim_K = lengthsShape.GetDim(1);
            }
            else {
                dim_K = lengthsShape.GetDim(0);
            }
        }
        else {
            useOffsets = 1;
            // 检查 offsets 是否存在 (理论上 lengths 和 offsets 必须有一个)
            const gert::StorageShape* offsetsStorageShape = context->GetInputShape(4);
            const gert::Shape& offsetsShape = offsetsStorageShape->GetStorageShape();
            if (offsetsShape.GetDimNum() == 2) {
                dim_K = offsetsShape.GetDim(1) - 1;
            }
            else {
                dim_K = offsetsShape.GetDim(0) - 1;
            }
        }
        // printf("useOffsets = %d\n", useOffsets);
        // 4. 处理 Initial (Index 5)
        // [修复点] 使用 Index 5，且只用于判断是否存在
        bool hasInitial = (context->GetInputTensor(5) != nullptr);
        // 5. 计算 Offsets
        uint32_t outerOffset = 1;
        uint32_t innerOffset = 1;
        for (int32_t d = 0; d < axis; ++d) outerOffset *= dataShape.GetDim(d);
        if (axis == 0) {
            innerOffset = N2;
            outerOffset = 1;
        }
        else {
            innerOffset = 1;
            outerOffset = N;
        }
        // 6. 设置 Tiling Key   
        // 使用 Input 0 (grad) 或 Input 2 (data) 的类型即可，它们必须一致
        auto dt = context->GetInputTensor(2)->GetDataType();
        uint32_t tilingKey;
        // sum/mean
        if (axis == 0) {
            tilingKey = 0;
        }
        else {
            if(dt == ge::DT_FLOAT) {
                tilingKey = 1;
            }
            else if(dt == ge::DT_FLOAT16) {
                tilingKey = 2;
            }
        }
        context->SetTilingKey(tilingKey);
        context->SetBlockDim(40);

        // 7. 设置 Tiling Data
        tiling.set_N(N);
        tiling.set_N2(N2);
        tiling.set_dim_K(dim_K);
        tiling.set_axis(axis);
        tiling.set_outerOffset(outerOffset);
        tiling.set_innerOffset(innerOffset);
        tiling.set_reduceType(reduceType);
        tiling.set_useOffsets(useOffsets);
        tiling.set_hasInitial(hasInitial ? 1 : 0);

        // Host 预计算：减少 kernel 标量开销
        const uint32_t MAX_CORES = 40;
        uint32_t shardsPerSeg = (dim_K == 0) ? 1 : std::max(1u, MAX_CORES / dim_K);
        tiling.set_shardsPerSeg(shardsPerSeg);

        uint32_t maxTileLength = 0;
        uint32_t dtypeBytes = (dt == ge::DT_FLOAT) ? 4 : 2;
        if (reduceType == 0 || reduceType == 1) {
            if (dt == ge::DT_FLOAT) maxTileLength = 8192;
            else if (dt == ge::DT_FLOAT16) maxTileLength = 32512;
            else maxTileLength = 10240;
        } else {
            if (dt == ge::DT_BF16 || dt == ge::DT_FLOAT16) maxTileLength = 4608;
            else maxTileLength = 5632;
        }
        tiling.set_maxTileLength(maxTileLength);
        tiling.set_rowSizeBytes(N2 * dtypeBytes);
        tiling.set_rowComputeLen((N2 + 63) / 64 * 64);

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    // SegmentReduceGrad 的形状推断函数
    static ge::graphStatus InferShape(gert::InferShapeContext* context) {
        // 输入顺序: 0=grad, 1=output, 2=data, 3=lengths(可选), 4=offsets(可选), 5=initial(可选)
        const gert::Shape* dataShape = context->GetInputShape(2);  // data 是第2个输入
        gert::Shape* outShape = context->GetOutputShape(0);


        // SegmentReduceGrad 的输出形状与 data 相同，都是 [N, N2]
        // 直接复制 data 的形状
        outShape->SetDimNum(dataShape->GetDimNum());
        for (uint32_t i = 0; i < dataShape->GetDimNum(); ++i) {
            outShape->SetDim(i, dataShape->GetDim(i));
        }

        return GRAPH_SUCCESS;
    }
}

namespace ops {
    class SegmentReduceGrad : public OpDef {
    public:
        explicit SegmentReduceGrad(const char* name) : OpDef(name) {
            // 输入: grad - 上游梯度，形状为 (K, N2) 如果 axis == 0，否则 (N, K)
            this->Input("grad")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16 })
                .Format({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND });

            // 输入: output - SegmentReduce 的前向输出，形状为 [K, N2] (axis==0) 或 [*, K, *]
            this->Input("output")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16 })
                .Format({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND });

            // 输入: data - SegmentReduce 的前向输入，形状为 [N, N2]
            this->Input("data")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16 })
                .Format({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND });

            // 可选输入: lengths - 形状为 [K]，使用 int32
            this->Input("lengths")
                .ParamType(OPTIONAL)
                .DataType({ ge::DT_INT32,ge::DT_INT32,ge::DT_INT32 })
                .Format({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND });

            // 可选输入: offsets - 形状为 [K]，使用 int32
            this->Input("offsets")
                .ParamType(OPTIONAL)
                .DataType({ ge::DT_INT32,ge::DT_INT32,ge::DT_INT32 })
                .Format({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND });

            // 可选输入: initial - 形状为 [1]
            this->Input("initial")
                .ParamType(OPTIONAL)
                .DataType({ ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16 })
                .Format({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND });

            // 输出: grad_output - 对 data 的梯度，形状为 [N, N2]
            this->Output("grad_output")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16 })
                .Format({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND });

            // 属性定义主要在 JSON 配置文件中完成
            // 这里只需要声明属性名称，具体类型和默认值在 JSON 中定义
            this->Attr("reduce")
                .AttrType(REQUIRED)
                .String();

            // 2. axis: 整数类型，可选，默认值为 0
            this->Attr("axis")
                .AttrType(OPTIONAL)
                .Int(0);

            this->SetInferShape(ge::InferShape);
            this->AICore().SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(SegmentReduceGrad);
}
