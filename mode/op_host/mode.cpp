#include "mode_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <vector>
#include <algorithm> // for std::max

namespace optiling
{
    enum ModeDataType {
        DTYPE_FLOAT32 = 0,
        DTYPE_FLOAT16 = 1,
        DTYPE_INT32   = 2,
        DTYPE_INT8    = 3,
        DTYPE_INT64   = 4,
        DTYPE_INT16   = 5,
        DTYPE_UINT8   = 6,
        DTYPE_UNKNOWN = 99
    };

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        ModeTilingData tiling;
        const gert::StorageShape *x_shape = context->GetInputShape(0);
        const gert::Shape &shape = x_shape->GetStorageShape();
        
        int32_t dim = -1;
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        if (attrs) {
            const int32_t *dim_ptr = attrs->GetAttrPointer<int32_t>(0);
            if (dim_ptr) dim = *dim_ptr;
        }

        size_t dim_num = shape.GetDimNum();
        int32_t real_dim = dim < 0 ? dim + dim_num : dim;

        // 1. 计算 Axis Length
        uint32_t axisLen = shape.GetDim(real_dim);

        // 2. 计算 Inner Stride
        uint32_t innerStride = 1;
        for (size_t i = real_dim + 1; i < dim_num; ++i) {
            innerStride *= shape.GetDim(i);
        }

        // Total Tasks
        uint64_t totalElements = shape.GetShapeSize();
        uint32_t totalTasks = totalElements / axisLen;
        
        // 3. 多核切分
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t core_num = platform.GetCoreNumAic();
        if (core_num == 0) core_num = 1;

        uint32_t actualCoreNum = totalTasks < core_num ? totalTasks : core_num;
        if (actualCoreNum == 0) actualCoreNum = 1;

        uint32_t tasksPerBlock = totalTasks / actualCoreNum;
        uint32_t bigCoreNum = totalTasks % actualCoreNum;

        auto input_dtype = context->GetInputDesc(0)->GetDataType();
        uint32_t dtype_enum = DTYPE_UNKNOWN;
        uint32_t typeSize = 4; // 默认4字节
        
        if (input_dtype == ge::DT_FLOAT) { dtype_enum = DTYPE_FLOAT32; typeSize = 4; }
        else if (input_dtype == ge::DT_FLOAT16) { dtype_enum = DTYPE_FLOAT16; typeSize = 2; }
        else if (input_dtype == ge::DT_INT32) { dtype_enum = DTYPE_INT32; typeSize = 4; }
        else if (input_dtype == ge::DT_INT8) { dtype_enum = DTYPE_INT8; typeSize = 1; }
        else if (input_dtype == ge::DT_INT64) { dtype_enum = DTYPE_INT64; typeSize = 8; }
        else if (input_dtype == ge::DT_INT16) { dtype_enum = DTYPE_INT16; typeSize = 2; }
        else if (input_dtype == ge::DT_UINT8) { dtype_enum = DTYPE_UINT8; typeSize = 1; }
        
        // 4. 动态计算 MaxChunkSize
        uint64_t ubSize;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        ubSize = 196608;

        // Kernel 中 paddedTotalLength 的计算逻辑 (32对齐)
        uint32_t paddedTotalLength = (axisLen + 31) / 32 * 32;
        if (paddedTotalLength < 64) paddedTotalLength = 64;

        // 计算固定占用的 Buffer 大小 (sortStorageBuf, 存储 float 类型的转换结果)
        uint32_t fixedOverhead = paddedTotalLength * sizeof(float);
        
        // 预留一些系统开销
        uint64_t reserveSize = 32; // 稍微多留一点防止溢出
        
        uint32_t maxChunkSize = 256; // 默认最小值
        
        if (ubSize > fixedOverhead + reserveSize) {
            uint64_t availableUB = ubSize - fixedOverhead - reserveSize;
            
            uint32_t bytesPerElement = 0;
            
            // 【优化点 1】：针对 Float32，不再计算输入 Buffer 的开销
            // concatBuf(4) + indexBuf(4) + dstIndexBuf(4) + sortDstBuf(8) + sortTmpBuf(9)
            // = 29 Bytes
            if (dtype_enum == DTYPE_FLOAT32) {
                bytesPerElement = 29; 
            } else {
                // 其他类型需要单独的 Input Buffer (typeSize)
                bytesPerElement = typeSize + 29;
            }
            
            maxChunkSize = availableUB / bytesPerElement;
            
            // 向下对齐到 256 (Sort建议对齐)
            maxChunkSize = (maxChunkSize / 256) * 256;
            if (maxChunkSize == 0) maxChunkSize = 256;
        } else {
            maxChunkSize = 64;
        }

        uint32_t tmpSize = maxChunkSize * 9;

        tiling.set_totalTasks(totalTasks);
        tiling.set_axisLen(axisLen);
        tiling.set_innerStride(innerStride);
        tiling.set_tasksPerBlock(tasksPerBlock);
        tiling.set_bigCoreNum(bigCoreNum);
        tiling.set_dtype(dtype_enum);
        tiling.set_maxChunkSize(maxChunkSize); // 传递计算出的最优 Block
        tiling.set_tmpSize(tmpSize);

        context->SetBlockDim(actualCoreNum);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        return ge::GRAPH_SUCCESS;
    }
}
// InferShape 和 Ops 定义保持不变，此处省略...

namespace ge
{
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *values_shape = context->GetOutputShape(0);
        gert::Shape *indices_shape = context->GetOutputShape(1);
        
        int32_t dim = -1;
        bool keepdim = false;
        
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        if (attrs) {
            const int32_t *dim_ptr = attrs->GetAttrPointer<int32_t>(0);
            if (dim_ptr) dim = *dim_ptr;
            const bool *keepdim_ptr = attrs->GetAttrPointer<bool>(1);
            if (keepdim_ptr) keepdim = *keepdim_ptr;
        }

        size_t dim_num = x_shape->GetDimNum();
        int32_t real_dim = dim < 0 ? dim + dim_num : dim;

        for (size_t i = 0; i < dim_num; ++i) {
            if (i == (size_t)real_dim) {
                if (keepdim) {
                    values_shape->AppendDim(1);
                    indices_shape->AppendDim(1);
                }
            } else {
                values_shape->AppendDim(x_shape->GetDim(i));
                indices_shape->AppendDim(x_shape->GetDim(i));
            }
        }
        return GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        context->SetOutputDataType(1, ge::DT_INT64);
        return GRAPH_SUCCESS;
    }
}

namespace ops
{
    class Mode : public OpDef
    {
    public:
        explicit Mode(const char *name) : OpDef(name)
        {
            this->Input("input")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8,
                           ge::DT_INT64, ge::DT_INT16, ge::DT_UINT8})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                         ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

            this->Output("values")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8,
                           ge::DT_INT64, ge::DT_INT16, ge::DT_UINT8})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                         ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

            this->Output("indices")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                           ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                         ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

            this->Attr("dim").AttrType(OPTIONAL).Int(-1);
            this->Attr("keepdim").AttrType(OPTIONAL).Bool(false);

            this->SetInferShape(ge::InferShape);
            this->SetInferDataType(ge::InferDataType);
            this->AICore().SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
        }
    };
    OP_ADD(Mode);
}