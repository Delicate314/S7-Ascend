/**
 * @file trace.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
 #include "trace_tiling.h"
 #include "register/op_def_registry.h"
 #include "tiling/platform/platform_ascendc.h"
 namespace optiling {
 const uint32_t BLOCK_DIM = 1;  // 单核处理
 static ge::graphStatus TilingFunc(gert::TilingContext *context)
 {
    TilingData tiling;
     

    const gert::StorageShape* dataStorageShape = context->GetInputShape(0);
    const gert::Shape& dataShape = dataStorageShape->GetStorageShape();
     // auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
     // auto aivNum = ascendcPlatform.GetCoreNumAiv();
     // uint64_t ub_size;
     // ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
     // printf("aivNum: %d, ub_size: %ld\n", aivNum, ub_size);
     // if(aivNum > 1) {
     //     return ge::GRAPH_FAILED;
     // }
     // Trace 计算：输入必须是矩阵 (N, N)
     uint32_t N = dataShape.GetDim(0);   // 矩阵第一维
     uint32_t N2 = (dataShape.GetDimNum() > 1) ? dataShape.GetDim(1) : N;  // 矩阵第二维
     
     
    // 获取输入数据类型，设置 tilingKey
    auto dt = context->GetInputTensor(0)->GetDataType();
    uint32_t tilingKey;
    if (dt == ge::DT_FLOAT) {
        tilingKey = 0;  // 0: int32 -> int64
    } else if (dt == ge::DT_INT32  ) {
        tilingKey = 1;
    } else if (dt == ge::DT_INT16 ) {
        tilingKey = 2;
    } else if (dt == ge::DT_INT8 || dt == ge::DT_UINT8) {
        tilingKey = 3;
    } 

    printf("[DEBUG] tilingKey: %d\n", tilingKey);
    context->SetBlockDim(1);  // 单核处理
    tiling.set_N1(N); 
    tiling.set_N2(N2);
    tiling.set_tilingKey(tilingKey);  // 设置 tilingKey
     
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
     
     // 单核处理，不需要 workspace
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;  // 不需要 workspace
     
    return ge::GRAPH_SUCCESS;
 }
 } // namespace optiling
 
 namespace ge {
 static graphStatus InferShape(gert::InferShapeContext *context)
 {
     const gert::Shape *dataShape = context->GetInputShape(0);
     gert::Shape *outShape = context->GetOutputShape(0);
 

     uint32_t dim0 = dataShape->GetDim(0);
     uint32_t dim1 = dataShape->GetDim(1);
     

     // 输出形状：标量 []
     outShape->SetDimNum(0);
 
     return GRAPH_SUCCESS;
 }
 
 static graphStatus InferDataType(gert::InferDataTypeContext *context)
 {
     // Trace 输出类型：根据输入类型推断
     // int32 -> int64, float32 -> float32
     ge::DataType inputDataType = context->GetInputDataType(0);
     ge::DataType outputDataType;
     
     if (inputDataType == ge::DT_FLOAT) {
         outputDataType = ge::DT_FLOAT;
     } else {
         outputDataType = ge::DT_INT64;
     }
     
     context->SetOutputDataType(0, outputDataType);
     return ge::GRAPH_SUCCESS;
 }
 } // namespace ge
 
 namespace ops {
 class Trace : public OpDef {
 public:
     explicit Trace(const char *name) : OpDef(name)
     {
         // 输入: data (矩阵) - 当前仅支持 int32 和 float32
         this->Input("data")
             .ParamType(REQUIRED)
             .DataType({ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT16,ge::DT_INT8,ge::DT_UINT8})
             .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
         
         // 输出: out (标量) - 类型由 InferDataType 推断
         this->Output("out")
             .ParamType(REQUIRED)
             .DataType({ge::DT_INT64, ge::DT_FLOAT, ge::DT_INT64,ge::DT_INT64,ge::DT_INT64})
             .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
 
         this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
         this->AICore()
             .SetTiling(optiling::TilingFunc)
             .AddConfig("ascend310b")
             .AddConfig("ascend910b");
     }
 };
 OP_ADD(Trace);
 } // namespace ops
 