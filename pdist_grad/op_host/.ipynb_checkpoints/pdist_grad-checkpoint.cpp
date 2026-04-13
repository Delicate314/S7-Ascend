#include "pdist_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling
{
    const uint32_t BLOCK_DIM = 8;
    // 最小处理元素个数，保证至少 32 字节 (8 * 4 bytes)
    const uint32_t MIN_M_PER_CORE = 8;

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        PdistGradTilingData tiling;

        const gert::StorageShape *input_shape = context->GetInputShape(1);
        if (input_shape == nullptr)
            return ge::GRAPH_FAILED;

        int32_t N = input_shape->GetStorageShape().GetDim(0);
        int32_t M = input_shape->GetStorageShape().GetDim(1);

        float p = 2.0;
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        if (attrs)
        {
            const float *p_ptr = attrs->GetAttrPointer<float>(0);
            if (p_ptr)
                p = *p_ptr;
        }

        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t core_num = platform.GetCoreNumAiv();
        if (core_num == 0)
            core_num = BLOCK_DIM;

        // 策略修正：确保每个核处理的 M 至少为 8 (32Bytes)
        // 如果 M 很小，减少核数
        if (M < core_num * MIN_M_PER_CORE) {
            if (M < MIN_M_PER_CORE) {
                core_num = 1;
            } else {
                core_num = M / MIN_M_PER_CORE;
            }
        }
        
        // 兜底
        if (core_num == 0) core_num = 1;

        uint32_t m_per_core = M / core_num;
        uint32_t tail_m_core = M - m_per_core * (core_num - 1);

        uint32_t copym = (m_per_core * sizeof(float) + 32 - 1) / 32 * 32 / sizeof(float);

        tiling.set_n(N);
        tiling.set_m(M);
        tiling.set_p(p);
        tiling.set_m_per_core(m_per_core);
        tiling.set_tail_m_core(tail_m_core);
        tiling.set_copym(copym);

        context->SetBlockDim(core_num);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        return ge::GRAPH_SUCCESS;
    }
}
// ... 剩余注册代码不变 ...

namespace ge
{
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        // Output 'out' has same shape as 'input' (index 1)
        const gert::Shape *input_shape = context->GetInputShape(1);
        gert::Shape *y_shape = context->GetOutputShape(0);
        *y_shape = *input_shape;
        return GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        const auto inputDataType = context->GetInputDataType(1);
        context->SetOutputDataType(0, inputDataType);
        return ge::GRAPH_SUCCESS;
    }
}

namespace ops
{
    class PdistGrad : public OpDef
    {
    public:
        explicit PdistGrad(const char *name) : OpDef(name)
        {
            this->Input("grad")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("input")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("pdist")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Attr("p").AttrType(OPTIONAL).Float(2.0);

            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910");
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(PdistGrad);
}