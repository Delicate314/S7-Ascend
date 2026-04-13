#include "segment_reduce_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <vector>
#include <string>

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
        SegmentReduceTilingData tiling;

        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto aivNum = ascendcPlatform.GetCoreNumAiv();

        // 获取输入输出形状
        const gert::StorageShape* dataStorageShape = context->GetInputShape(0);
        gert::Shape outShape = context->GetOutputShape(0)->GetOriginShape();

        const gert::Shape& dataShape = dataStorageShape->GetStorageShape();

        uint32_t N = dataShape.GetDim(0);   // 输入第一维
        uint32_t N2 = (dataShape.GetDimNum() > 1) ? dataShape.GetDim(1) : 1;  // 输入第二维
        uint32_t dim_K = 0;  // K 数组的实际维度

        // 获取属性值（按 JSON 中定义的顺序：0=reduce, 1=axis, 2=unsafe）
        // reduce 属性 (索引 0, 类型: string, 必填)
        std::string reduceStr = "sum";
        const char* reducePtr = context->GetAttrs()->GetStr(0);
        if (reducePtr != nullptr) {
            reduceStr = std::string(reducePtr);
        }
        uint32_t reduceType = GetReduceType(reduceStr);

        // axis 属性 (索引 1, 类型: int, 可选, 默认值: 0)
        int32_t axis = 0;
        const int64_t* axisPtr = context->GetAttrs()->GetInt(1);
        if (axisPtr != nullptr) {
            axis = static_cast<int32_t>(*axisPtr);
        }

        // 判断是 lengths 还是 offsets
        uint32_t useOffsets = 0;

        // 检查是否有 lengths 或 offsets 输入
        bool hasLengths = (context->GetInputTensor(1) != nullptr);
        bool hasOffsets = (context->GetInputTensor(3) != nullptr);

        if (hasOffsets) {
            useOffsets = 1;
            const gert::StorageShape* offsetsStorageShape = context->GetInputShape(3);
            if (offsetsStorageShape == nullptr) {
                printf("[ERROR] offsetsStorageShape is nullptr\n");
                return ge::GRAPH_FAILED;
            }
            const gert::Shape& offsetsShape = offsetsStorageShape->GetStorageShape();
            // offsets 的 shape 是 (N+1,)，所以 K = N = offsetsShape.GetDim(0) - 1
            dim_K = offsetsShape.GetDim(0) - 1;
            // printf("[DEBUG] Using offsets, dim_K = %d (from offsets shape dim0=%d)\n", dim_K, offsetsShape.GetDim(0));
        }
        else if (hasLengths) {
            useOffsets = 0;
            const gert::StorageShape* lengthsStorageShape = context->GetInputShape(1);
            if (lengthsStorageShape == nullptr) {
                printf("[ERROR] lengthsStorageShape is nullptr\n");
                return ge::GRAPH_FAILED;
            }
            const gert::Shape& lengthsShape = lengthsStorageShape->GetStorageShape();

            // 在 tiling 阶段无法读取数据，只能从 shape 获取 dim_K
            if (lengthsShape.GetDimNum() == 2) {
                // lengths 是 2 维 (N, K)，取第二维
                dim_K = lengthsShape.GetDim(1);
                uint32_t batchSize = lengthsShape.GetDim(0);
                // printf("[DEBUG] Using lengths (2D), batchSize=%d, dim_K=%d\n", batchSize, dim_K);
            }
            else {
                // lengths 是 1 维 (K,)
                dim_K = lengthsShape.GetDim(0);
                // printf("[DEBUG] Using lengths (1D), dim_K=%d\n", dim_K);
            }

        }

        // 检查是否有 initial 值
        uint32_t hasInitial = (context->GetInputTensor(4) != nullptr) ? 1 : 0;

        // 计算 outer_offset 和 inner_offset（根据 PyTorch 逻辑）
        uint32_t outerOffset = 1;
        uint32_t innerOffset = 1;

        // outer_offset 是 axis 之前的维度乘积
        for (int32_t d = 0; d < axis; ++d) {
            outerOffset *= dataShape.GetDim(d);
        }
        // inner_offset 是 axis 之后的维度乘积
        for (int32_t d = axis + 1; d < static_cast<int32_t>(dataShape.GetDimNum()); ++d) {
            innerOffset *= dataShape.GetDim(d);
        }

        // 如果 axis == 0，innerOffset = N2，否则需要重新计算
        if (axis == 0) {
            innerOffset = N2;
            outerOffset = 1;
        }
        else {
            innerOffset = 1;
            for (int32_t d = axis + 1; d < static_cast<int32_t>(dataShape.GetDimNum()); ++d) {
                innerOffset *= dataShape.GetDim(d);
            }
        }
        uint32_t tilingKey;
        // 获取数据类型和计算 tiling key
        auto dt = context->GetInputTensor(0)->GetDataType();
        uint32_t dtypeOffset = 0;

        if (dt == ge::DT_FLOAT) {
            if (dataShape.GetDimNum() == 1) {
                tilingKey = 10;
            }
            else {
                tilingKey = 0;
            }
        }
        else if (dt == ge::DT_FLOAT16) {
            if (dataShape.GetDimNum() == 1) {
                tilingKey = 11;
            }
            else {
                tilingKey = 1;
            }
        }
        else if (dt == ge::DT_BF16) {
            printf("dataShape.GetDimNum(): %d\n", dataShape.GetDimNum());
            tilingKey = 2;
        }

        printf("tilingKey: %d\n", tilingKey);
        // printf("axis: %d\n", axis);
        // printf("hasLengths: %d\n", hasLengths);
        // printf("hasOffsets: %d\n", hasOffsets);
        // printf("hasInitial: %d\n", hasInitial);
        // printf("reduceType: %d\n", reduceType);



        context->SetTilingKey(tilingKey);
        context->SetBlockDim(40);

        // 验证 dim_K 是否有效
        if (dim_K == 0 || dim_K > 20) {
            // printf("[ERROR] Invalid dim_K: %d, must be between 1 and 20\n", dim_K);
            return ge::GRAPH_FAILED;
        }

        // printf("[DEBUG] Before setting tiling data: dim_K=%d\n", dim_K);

        // 设置 tiling 数据
        tiling.set_N(N);
        tiling.set_N2(N2);
        // 设置 K 数组
        tiling.set_dim_K(dim_K);  // 设置 K 数组的实际维度
        tiling.set_axis(axis);
        tiling.set_outerOffset(outerOffset);
        tiling.set_innerOffset(innerOffset);
        tiling.set_reduceType(reduceType);
        tiling.set_useOffsets(useOffsets);
        tiling.set_hasInitial(hasInitial);

        // printf("[DEBUG] Tiling data set successfully, saving to buffer...\n");
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        // printf("[DEBUG] Tiling function completed successfully\n");
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    // SegmentReduce 的形状推断函数
    static ge::graphStatus InferShape(gert::InferShapeContext* context) {
        const gert::Shape* dataShape = context->GetInputShape(0);
        gert::Shape* outShape = context->GetOutputShape(0);

        if (dataShape == nullptr || outShape == nullptr) {
            return GRAPH_FAILED;
        }

        // 获取 axis 属性 (索引 1, 类型: int, 可选, 默认值: 0)
        int32_t axis = 0;
        const int64_t* axisPtr = context->GetAttrs()->GetInt(1);
        if (axisPtr != nullptr) {
            axis = static_cast<int32_t>(*axisPtr);
        }
        if (axis < 0) {
            axis += static_cast<int32_t>(dataShape->GetDimNum());
        }

        // 获取段数量 K
        uint32_t K = 0;
        bool hasLengths = (context->GetInputTensor(1) != nullptr);
        bool hasOffsets = (context->GetInputTensor(3) != nullptr);

        if (hasOffsets) {
            const gert::Shape* offsetsShape = context->GetInputShape(3);
            if (offsetsShape != nullptr && offsetsShape->GetDimNum() > 0) {
                K = offsetsShape->GetDim(0) - 1;
            }
        }
        else if (hasLengths) {
            const gert::Shape* lengthsShape = context->GetInputShape(1);
            if (lengthsShape != nullptr && lengthsShape->GetDimNum() > 0) {
                K = lengthsShape->GetDim(0);
            }
        }

        if (K == 0) {
            return GRAPH_FAILED;
        }

        // 根据 PyTorch 逻辑：输出形状为 (K, N2) 如果 axis == 0，否则为 (N, K)
        std::vector<int64_t> outputDims;
        if (axis == 0) {
            // 输出形状: (K, N2)
            outputDims.push_back(static_cast<int64_t>(K));
            if (dataShape->GetDimNum() > 1) {
                outputDims.push_back(static_cast<int64_t>(dataShape->GetDim(1)));
            }
            // 复制其他维度
            for (uint32_t d = 2; d < dataShape->GetDimNum(); ++d) {
                outputDims.push_back(static_cast<int64_t>(dataShape->GetDim(d)));
            }
        }
        else {
            // 输出形状: 在前 N-1 个维度之后插入 K
            for (int32_t d = 0; d < axis; ++d) {
                outputDims.push_back(static_cast<int64_t>(dataShape->GetDim(d)));
            }
            outputDims.push_back(static_cast<int64_t>(K));
            for (uint32_t d = axis + 1; d < dataShape->GetDimNum(); ++d) {
                outputDims.push_back(static_cast<int64_t>(dataShape->GetDim(d)));
            }
        }

        // 设置输出形状
        outShape->SetDimNum(static_cast<uint32_t>(outputDims.size()));
        for (size_t i = 0; i < outputDims.size(); ++i) {
            outShape->SetDim(static_cast<uint32_t>(i), static_cast<uint64_t>(outputDims[i]));
        }
        for (size_t i = 0; i < outputDims.size(); ++i) {
            printf("outputDims[%zu]: %d\n", i, static_cast<int32_t>(outputDims[i]));
        }

        return GRAPH_SUCCESS;
    }
}

namespace ops {
    class SegmentReduce : public OpDef {
    public:
        explicit SegmentReduce(const char* name) : OpDef(name) {
            // 输入: data
            this->Input("data")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16 })
                .Format({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND });

            // 可选输入: lengths
            this->Input("lengths")
                .ParamType(OPTIONAL)
                .DataType({ ge::DT_INT64,ge::DT_INT64,ge::DT_INT64 })
                .Format({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND });

            // 可选输入: indices (暂不支持)
            this->Input("indices")
                .ParamType(OPTIONAL)
                .DataType({ ge::DT_INT32,ge::DT_INT32,ge::DT_INT32 })
                .Format({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND });

            // 可选输入: offsets
            this->Input("offsets")
                .ParamType(OPTIONAL)
                .DataType({ ge::DT_INT64,ge::DT_INT64,ge::DT_INT64 })
                .Format({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND });

            // 可选输入: initial
            this->Input("initial")
                .ParamType(OPTIONAL)
                .DataType({ ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16 })
                .Format({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND });

            // 输出: out
            this->Output("out")
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

            // 3. unsafe: 布尔类型，可选，默认值为 false
            this->Attr("unsafe")
                .AttrType(OPTIONAL)
                .Bool(false);

            this->SetInferShape(ge::InferShape);
            this->AICore().SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(SegmentReduce);
}
