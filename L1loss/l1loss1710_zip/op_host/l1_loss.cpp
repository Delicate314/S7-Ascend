#include "l1_loss_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstring>
#include <vector>

namespace optiling {

    enum BroadcastType {
        BCAST_NONE = 0, // N vs N
        BCAST_X = 1,    // 1 vs N (X广播)
        BCAST_Y = 2     // N vs 1 (Y广播)
    };

const uint32_t BLOCK_DIM = 1;  // 单核处理
constexpr int32_t CALC_ALIGN_NUM = 256;

// 形状右对齐处理（借鉴 fmax）
static void GetTensorShape(const gert::StorageShape* shape, std::vector<uint64_t>& inshapeVector, uint32_t outDimNum) {
    int n = outDimNum - 1;
    int32_t dimNum = shape->GetStorageShape().GetDimNum();
    for (int j = dimNum - 1; j >= 0; --j) {
        if (n >= 0) {
            inshapeVector[n--] = shape->GetStorageShape().GetDim(j);
        }
    }
}

// 设置广播参数（借鉴 fmax）
static void SetBroadCastParams(const std::vector<uint64_t>& inshapeVector1,
    const std::vector<uint64_t>& inshapeVector2,
    const gert::Shape& outshape,
    TilingData& tiling) {
    uint32_t outDimNum = outshape.GetDimNum();

    // 局部暂存数组
    uint32_t tmp_shape[20] = { 0 };
    uint32_t tmp_reduce1[20] = { 0 };
    uint32_t tmp_reduce2[20] = { 0 };

    for (uint32_t i = 0; i < outDimNum; i++) {
        tmp_shape[i] = static_cast<uint32_t>(outshape.GetDim(i));
        // 如果输入形状与输出形状不同，标记为需要广播
        if (inshapeVector1[i] != outshape.GetDim(i)) tmp_reduce1[i] = 1;
        if (inshapeVector2[i] != outshape.GetDim(i)) tmp_reduce2[i] = 1;
    }

    // 写入 Tiling Data
    tiling.set_shape(tmp_shape);
    tiling.set_reduce1(tmp_reduce1);
    tiling.set_reduce2(tmp_reduce2);
    tiling.set_dim(outDimNum);
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData tiling;
    
    // 获取输入形状
    const gert::StorageShape* xStorageShape = context->GetInputShape(0);
    const gert::StorageShape* targetStorageShape = context->GetInputShape(1);
    const gert::Shape& xShape = xStorageShape->GetStorageShape();
    const gert::Shape& targetShape = targetStorageShape->GetStorageShape();
    
    uint32_t xDimNum = xShape.GetDimNum();
    uint32_t targetDimNum = targetShape.GetDimNum();
    uint32_t maxDimNum = (xDimNum > targetDimNum) ? xDimNum : targetDimNum;
    
    // 计算广播后的输出形状（不依赖 InferShape 的输出）
    gert::Shape outshape;
    outshape.SetDimNum(maxDimNum);
    for (uint32_t i = 0; i < maxDimNum; ++i) {
        int64_t d1 = 1, d2 = 1;
        if (i < maxDimNum - xDimNum) {
            d1 = 1;
        } else {
            d1 = xShape.GetDim(i - (maxDimNum - xDimNum));
        }
        if (i < maxDimNum - targetDimNum) {
            d2 = 1;
        } else {
            d2 = targetShape.GetDim(i - (maxDimNum - targetDimNum));
        }
        // 广播规则：取较大值
        int64_t outDim = (d1 > d2) ? d1 : d2;
        outshape.SetDim(i, outDim);
    }
    
    uint32_t outDimNum = maxDimNum;
    
    // 获取输入形状并对齐（借鉴 fmax）
    std::vector<uint64_t> inshapeVector1(outDimNum, 1);
    std::vector<uint64_t> inshapeVector2(outDimNum, 1);
    GetTensorShape(xStorageShape, inshapeVector1, outDimNum);
    GetTensorShape(targetStorageShape, inshapeVector2, outDimNum);
    
    // 计算输出总长度
    uint64_t totalLength = 1;
    for (uint32_t i = 0; i < outDimNum; i++) {
        totalLength *= outshape.GetDim(i);
    }
    tiling.set_totalLength(static_cast<uint32_t>(totalLength));
    
    // 判定是否需要广播
    bool needBroadcast = false;
    for (uint32_t i = 0; i < outDimNum; i++) {
        if (inshapeVector1[i] != inshapeVector2[i]) {
            needBroadcast = true;
            break;
        }
    }
    
    // PyTorch 广播是从右对齐的，需要对齐维度（用于轴折叠）
    std::vector<int64_t> s1, s2;
    for (uint32_t i = 0; i < outDimNum; ++i) {
        s1.push_back(static_cast<int64_t>(inshapeVector1[i]));
        s2.push_back(static_cast<int64_t>(inshapeVector2[i]));
    }
    
    // 轴折叠逻辑 (Axis Collapsing)
    std::vector<int64_t> foldS1, foldS2;
    
    // 临时变量，用于记录正在合并的维度
    int64_t curDim1 = 1;
    int64_t curDim2 = 1;
    int lastBcastType = -1; // -1: 初始状态
    
    // 遍历所有维度
    for (uint32_t i = 0; i < outDimNum; ++i) {
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
            currentBcastType = BCAST_X; // Input1 (x) 需要广播
        }
        else if (d1 > 1 && d2 == 1) {
            currentBcastType = BCAST_Y; // Input2 (target) 需要广播
        }
        else {
            // N vs M (N!=M)，非法广播，直接报错
            printf("[ERROR] l1_loss broadcast failed: dimension %u: %ld vs %ld\n", i, d1, d2);
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
    
    // 将合并后的维度映射到 N1, N2, N3, N4 (使用 x 的合并后维度)
    uint32_t collapsedDimNum = foldS1.size();
    uint32_t N1 = 1, N2 = 1, N3 = 1, N4 = 1;
    
    // 从后往前填充 N1, N2, N3, N4
    if (collapsedDimNum > 0) {
        N1 = (uint32_t)foldS1[collapsedDimNum - 1];
    }
    if (collapsedDimNum > 1) {
        N2 = (uint32_t)foldS1[collapsedDimNum - 2];
    }
    if (collapsedDimNum > 2) {
        N3 = (uint32_t)foldS1[collapsedDimNum - 3];
    }
    if (collapsedDimNum > 3) {
        N4 = (uint32_t)foldS1[collapsedDimNum - 4];
        // 如果维度数超过4，需要将前面的维度合并到N4
        for (uint32_t i = 0; i < collapsedDimNum - 4; ++i) {
            N4 *= (uint32_t)foldS1[i];
        }
    }
    
    
    // 准备传给 Kernel 的定长数组 (假设最大支持 4 维折叠结果)
    uint32_t dim1_array[4] = { 1, 1, 1, 1 };
    uint32_t dim2_array[4] = { 1, 1, 1, 1 };
    
    for (size_t i = 0; i < collapsedDimNum && i < 4; ++i) {
        dim1_array[i] = (int32_t)foldS1[i];
        dim2_array[i] = (int32_t)foldS2[i];
    }
    
    // 设置合并后的逻辑维度数组
    tiling.set_collapsedDimNum(collapsedDimNum);
    tiling.set_dim1(dim1_array);
    tiling.set_dim2(dim2_array);
    
    // 获取属性值：reduction (string, 可选, 默认值: "mean")
    // size_average 和 reduce 使用默认值（都是 1，即 true），不在属性中定义
    uint32_t size_average = 1;  // 默认 true（已废弃，不使用）
    uint32_t reduce = 1;  // 默认 true（已废弃，不使用）
    
    // reduction 属性 (索引 0, 类型: string, 可选, 默认值: "mean")
    // 0='none', 1='mean', 2='sum'
    uint32_t reduction = 0;  // 默认 'mean'
    const char* reductionStr = context->GetAttrs()->GetStr(0);
    // printf("[DEBUG] TilingFunc: reductionStr = %s\n", reductionStr ? reductionStr : "(null)");
    if (reductionStr != nullptr) {
        if (strcmp(reductionStr, "mean") == 0) {
            reduction = 1;
        } else if (strcmp(reductionStr, "sum") == 0) {
            reduction = 2;
        } else {
            reduction = 0;
        }
    }
    // printf("[DEBUG] TilingFunc: reduction = %u\n", reduction);
    
    // 获取输入数据类型，设置 tilingKey（x 和 target 类型相同）
    auto dt = context->GetInputTensor(0)->GetDataType();
    uint32_t tilingKey = 0;  // 默认值
    
    // 检查是否有广播（通过合并后的维度数组判断）
    bool hasBroadcast = false;
    for (uint32_t i = 0; i < collapsedDimNum; ++i) {
        if (dim1_array[i] != dim2_array[i]) {
            hasBroadcast = true;
            break;
        }
    }
    
    // 设置广播参数（借鉴 fmax）
    if (hasBroadcast) {
        SetBroadCastParams(inshapeVector1, inshapeVector2, outshape, tiling);
    }

    printf("[DEBUG] TilingFunc: collapsedDimNum = %d (after expansion: %d)\n", 
           (int)foldS1.size(), collapsedDimNum);
    
    if (hasBroadcast) {
        if (reduction == 0) {
            tilingKey = 101;  
        } else {
            tilingKey = 100;  
        }
    } else {
        tilingKey = 0;
    }
    
    
    printf("[DEBUG] TilingFunc: tilingKey = %d\n", tilingKey);
    
    // blockDim 设置：
    // - tilingKey=101 (none + broadcast): 输出按行独立，可多核并行
    // - 其它 tilingKey: 目前 kernel 侧未实现跨核归约/同步，保持单核
    if (tilingKey == 101) {
        context->SetBlockDim(8);
    } else {
        context->SetBlockDim(1);
    }
    tiling.set_N1(N1); 
    tiling.set_N2(N2);
    tiling.set_N3(N3);
    tiling.set_N4(N4);
    tiling.set_tilingKey(tilingKey);  // 设置 tilingKey
    
    tiling.set_size_average(size_average);
    tiling.set_reduce(reduce);
    tiling.set_reduction(reduction);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->SetTilingKey(tilingKey);
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
    const gert::Shape *xShape = context->GetInputShape(0);
    const gert::Shape *targetShape = context->GetInputShape(1);
    gert::Shape *outShape = context->GetOutputShape(0);


    // 获取 reduction 属性（索引 0，因为 size_average 和 reduce 不在属性中定义）
    uint32_t reduction = 1;  // 默认 'mean'
    const char* reductionStr = context->GetAttrs()->GetStr(0);
    if (reductionStr != nullptr) {
        if (strcmp(reductionStr, "mean") == 0) {
            reduction = 1;
        } else if (strcmp(reductionStr, "sum") == 0) {
            reduction = 2;
        } else if (strcmp(reductionStr, "none") == 0) {
            reduction = 0;
        }
    }

    // 根据 reduction 决定输出形状
    if (reduction == 0) {
        // 'none': 输出形状是广播后的形状
        uint32_t xDimNum = xShape->GetDimNum();
        uint32_t targetDimNum = targetShape->GetDimNum();
        uint32_t maxDimNum = (xDimNum > targetDimNum) ? xDimNum : targetDimNum;
        
        // 计算广播后的形状
        outShape->SetDimNum(maxDimNum);
        for (uint32_t i = 0; i < maxDimNum; ++i) {
            int64_t d1 = 1, d2 = 1;
            if (i < maxDimNum - xDimNum) {
                d1 = 1;
            } else {
                d1 = xShape->GetDim(i - (maxDimNum - xDimNum));
            }
            if (i < maxDimNum - targetDimNum) {
                d2 = 1;
            } else {
                d2 = targetShape->GetDim(i - (maxDimNum - targetDimNum));
            }
            // 广播规则：取较大值
            outShape->SetDim(i, (d1 > d2) ? d1 : d2);
        }
    } else {
        // 'mean' 或 'sum': 输出是标量（0维张量）
        outShape->SetDimNum(0);
    }

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // l1_loss 输出类型：与输入类型相同
    ge::DataType inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class L1Loss : public OpDef {
public:
    explicit L1Loss(const char *name) : OpDef(name)
    {
        // 输入: x - 支持 float16, float, bfloat16
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        
        // 输入: target - 支持 float16, float, bfloat16，形状与 x 相同
        this->Input("target")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        
        // 属性: reduction (string 类型，可选，默认值 "mean")
        this->Attr("reduction")
            .AttrType(REQUIRED)
            .String();
        
        // 输出: y - 形状和类型与输入相同
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend310b");
            // .AddConfig("ascend910b");
    }
};
OP_ADD(L1Loss);
} // namespace ops

