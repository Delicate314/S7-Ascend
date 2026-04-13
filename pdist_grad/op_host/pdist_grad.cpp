#include "pdist_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling
{
    const uint32_t BLOCK_DIM = 40; // 默认最大核数

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        PdistGradTilingData tiling;

        const gert::StorageShape *input_shape = context->GetInputShape(1);
        if (input_shape == nullptr) return ge::GRAPH_FAILED;

        int32_t N = input_shape->GetStorageShape().GetDim(0);
        int32_t M = input_shape->GetStorageShape().GetDim(1);

        float p = 2.0;
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        if (attrs) {
            const float *p_ptr = attrs->GetAttrPointer<float>(0);
            if (p_ptr) p = *p_ptr;
        }

        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t core_num = platform.GetCoreNumAiv();
        if (core_num == 0) core_num = BLOCK_DIM;

        // 1. 确定启动的核数
        // 任务数 = N。如果 N 小于核数，只启动 N 个核。
        uint32_t used_core_num = (uint32_t)N;
        if (used_core_num > core_num) used_core_num = core_num;
        if (used_core_num == 0) used_core_num = 1;

        // 2. 计算核内 M 维度切片 (UB Tiling)
        uint64_t ub_size_bytes = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size_bytes);
        if (ub_size_bytes == 0) ub_size_bytes = 192 * 1024;
        
        // 预留 4KB 给 Scale Buffer 等开销
        uint32_t scale_size = (N + 7) / 8 * 8 * sizeof(float);
        if (scale_size < 32) scale_size = 32;
        uint64_t avail_ub = 196608 - scale_size - 32;

        // 需要 4 个 buffer (vec_acc, vec_i, vec_j, vec_tmp)
        uint32_t tile_m_ub = avail_ub / (5 * sizeof(float));
        
        // 对齐处理
        tile_m_ub = (tile_m_ub / 8) * 8;
        if (tile_m_ub < 8) tile_m_ub = 8;
        if (tile_m_ub > M) tile_m_ub = (M + 7) / 8 * 8; // 向上取整对齐

        tiling.set_n(N);
        tiling.set_m(M);
        tiling.set_p(p);
        tiling.set_tile_m_ub(tile_m_ub);
        // 设置 Padding 为 0
        tiling.set_pad0(0); tiling.set_pad1(0); tiling.set_pad2(0); tiling.set_pad3(0);

        // 设置实际启动的核数
        context->SetBlockDim(used_core_num);
        
        // 由于所有核共享同一份 Tiling 参数（除了自己的 BlockIdx），只需保存一份
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *input_shape = context->GetInputShape(1);
        gert::Shape *y_shape = context->GetOutputShape(0);
        *y_shape = *input_shape;
        return ge::GRAPH_SUCCESS;
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
            this->Input("grad").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("input").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("pdist").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
            this->Attr("p").AttrType(OPTIONAL).Float(2.0);

            this->SetInferShape(optiling::InferShape).SetInferDataType(optiling::InferDataType);
            this->AICore().SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910");
            this->AICore().AddConfig("ascend910b");
        }
    };
    OP_ADD(PdistGrad);
}