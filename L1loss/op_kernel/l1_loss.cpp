#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t ALIGN_BYTES = 32;
// KernelL1LossMean: 处理 reduction='mean' 或 'sum' 的情况，输出标量
template<typename T>
class KernelL1LossMean {
private:
    TPipe* pipe;
    // 统一使用双缓冲输入队列，实现 copy/compute overlap（对所有类型生效）
    static constexpr int32_t IN_BUFFER_NUM = 2;
    TQue<QuePosition::VECIN, IN_BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, IN_BUFFER_NUM> inQueueTarget;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    
    // BF16 专用的 cast 队列
    TQue<QuePosition::VECCALC, BUFFER_NUM> sumQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueFloat;
    TQue<QuePosition::VECCALC, BUFFER_NUM> castQueueX;
    TQue<QuePosition::VECCALC, BUFFER_NUM> castQueueTarget;
    TQue<QuePosition::VECCALC, BUFFER_NUM> castQueueDiff;
    TQue<QuePosition::VECCALC, BUFFER_NUM> calcQueue;  // 用于 ReduceSum 的工作缓冲区

    GlobalTensor<T> xGm;
    GlobalTensor<T> targetGm;
    GlobalTensor<T> outputGm;

    uint32_t totalLength;  // 总元素数
    int tilelength;
    int max_tile_length;
    uint32_t reduction;  // 1='mean', 2='sum'
public:
    __aicore__ inline KernelL1LossMean() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR target, GM_ADDR output, GM_ADDR workspace, 
                                 uint32_t N1, uint32_t N2, uint32_t N3, uint32_t N4,
                                 uint32_t reduction,
                                 TPipe* pipeIn)
    {
        // 计算总元素数
        this->pipe = pipeIn;
        this->totalLength = N4 * N3 * N2 * N1;
        this->reduction = reduction;
        if constexpr (std::is_same_v<T, float>) {
            this->max_tile_length = 13056; 
        } else {
            this->max_tile_length = 10912; 
        }
        xGm.SetGlobalBuffer((__gm__ T*)x);
        targetGm.SetGlobalBuffer((__gm__ T*)target);
        outputGm.SetGlobalBuffer((__gm__ T*)output);
        int align_num = ALIGN_BYTES/sizeof(T);
        if (totalLength < this->max_tile_length) {
            this->tilelength = (totalLength+align_num-1)/align_num*align_num;
        } else {
            this->tilelength = this->max_tile_length;
        }
        pipe->InitBuffer(inQueueX, IN_BUFFER_NUM, this->tilelength * sizeof(T));
        pipe->InitBuffer(inQueueTarget, IN_BUFFER_NUM, this->tilelength * sizeof(T));
        pipe->InitBuffer(outQueue, 1, 32);
        pipe->InitBuffer(sumQueue, 1, 32);
        pipe->InitBuffer(outQueueFloat, 1, 32);
        if constexpr (std::is_same_v<T, __bf16> || std::is_same_v<T, half>) {
            pipe->InitBuffer(castQueueX, 1, this->tilelength * sizeof(float));
            pipe->InitBuffer(castQueueTarget, 1, this->tilelength * sizeof(float));
            pipe->InitBuffer(castQueueDiff, 1, this->tilelength * sizeof(float));
        }
        pipe->InitBuffer(calcQueue, 1, this->tilelength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        DataCopyPadExtParams<T> padParams{ false, 0, 0, 0 };
        
        LocalTensor<float> totalSumFloatTensor = outQueueFloat.AllocTensor<float>();
        LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
        LocalTensor<float> diffCast;
        LocalTensor<float> xLocalFloat;
        LocalTensor<float> targetLocalFloat;
        totalSumFloatTensor.SetValue(0, 0.0f);
        // 先预取第一个 tile，为后续双缓冲流水做准备
        uint32_t firstTile = (totalLength < (uint32_t)this->tilelength) ? totalLength : (uint32_t)this->tilelength;
        DataCopyExtParams firstCopyParams = { 1, (uint32_t)(firstTile * sizeof(T)), 0, 0, 0 };
        {
            LocalTensor<T> xPrefetch = inQueueX.AllocTensor<T>();
            LocalTensor<T> targetPrefetch = inQueueTarget.AllocTensor<T>();
            DataCopyPad(xPrefetch, xGm[0], firstCopyParams, padParams);
            DataCopyPad(targetPrefetch, targetGm[0], firstCopyParams, padParams);
            inQueueX.EnQue(xPrefetch);
            inQueueTarget.EnQue(targetPrefetch);
        }

        // 双缓冲流水：循环内预取下一块，同时计算当前块
        for (uint32_t tile_offset = 0; tile_offset < totalLength; tile_offset += this->tilelength) {
            uint32_t currentTile = (totalLength - tile_offset < (uint32_t)this->tilelength) 
                                    ? (totalLength - tile_offset) 
                                    : (uint32_t)this->tilelength;

            // 预取下一 tile（若存在）
            uint32_t next_offset = tile_offset + (uint32_t)this->tilelength;
            if (next_offset < totalLength) {
                uint32_t nextTile = (totalLength - next_offset < (uint32_t)this->tilelength)
                                        ? (totalLength - next_offset)
                                        : (uint32_t)this->tilelength;
                DataCopyExtParams nextCopyParams = { 1, (uint32_t)(nextTile * sizeof(T)), 0, 0, 0 };
                LocalTensor<T> xPrefetch = inQueueX.AllocTensor<T>();
                LocalTensor<T> targetPrefetch = inQueueTarget.AllocTensor<T>();
                DataCopyPad(xPrefetch, xGm[next_offset], nextCopyParams, padParams);
                DataCopyPad(targetPrefetch, targetGm[next_offset], nextCopyParams, padParams);
                inQueueX.EnQue(xPrefetch);
                inQueueTarget.EnQue(targetPrefetch);
            }

            // 取出当前 tile（上一次预取的结果）进行计算
            LocalTensor<T> xLocal = inQueueX.DeQue<T>();
            LocalTensor<T> targetLocal = inQueueTarget.DeQue<T>();
            if constexpr (std::is_same_v<T, __bf16>) {
                xLocalFloat = castQueueX.AllocTensor<float>();
                targetLocalFloat = castQueueTarget.AllocTensor<float>();
                Cast(xLocalFloat, xLocal, RoundMode::CAST_NONE, currentTile);
                Cast(targetLocalFloat, targetLocal, RoundMode::CAST_NONE, currentTile);
                castQueueX.EnQue(xLocalFloat);
                castQueueTarget.EnQue(targetLocalFloat);
                xLocalFloat = castQueueX.DeQue<float>();
                targetLocalFloat = castQueueTarget.DeQue<float>();
            }

            //COMPUTE
            if constexpr (std::is_same_v<T, __bf16>) {
                Sub(xLocalFloat, xLocalFloat, targetLocalFloat, currentTile);
                Abs(xLocalFloat, xLocalFloat, currentTile);
                diffCast = xLocalFloat;
            }
            else {
                Sub(xLocal, xLocal, targetLocal, currentTile);
                Abs(xLocal, xLocal, currentTile);
                if constexpr (std::is_same_v<T, half>) {
                    diffCast = castQueueDiff.AllocTensor<float>();
                    Cast(diffCast, xLocal, RoundMode::CAST_NONE, currentTile);
                }
                else {
                    diffCast = xLocal;
                }
                }
            LocalTensor<float> workLocal = calcQueue.AllocTensor<float>();
            LocalTensor<float> sumFloat = sumQueue.AllocTensor<float>();
            sumFloat.SetValue(0, 0.0f); // 初始化为 0
            ReduceSum<float>(sumFloat, diffCast, workLocal, currentTile);
            sumQueue.EnQue(sumFloat);
            sumFloat = sumQueue.DeQue<float>();
            Add(totalSumFloatTensor, totalSumFloatTensor, sumFloat, 1);

            // 释放当前 tile 相关的 LocalTensor
            calcQueue.FreeTensor(workLocal);
            sumQueue.FreeTensor(sumFloat);
            inQueueX.FreeTensor(xLocal);
            inQueueTarget.FreeTensor(targetLocal);
            if constexpr (std::is_same_v<T, __bf16>) {
                castQueueX.FreeTensor(xLocalFloat);
                castQueueTarget.FreeTensor(targetLocalFloat);
            }
            if constexpr (std::is_same_v<T, half>) {
                // half 路径下 diffCast 来源于 castQueueDiff
                castQueueDiff.FreeTensor(diffCast);
            }
        }
        
        if (reduction == 1) {
            float total_lenth_inv = 1.0f / totalLength;
            Muls(totalSumFloatTensor, totalSumFloatTensor, total_lenth_inv, 1);
        }
        outQueueFloat.EnQue(totalSumFloatTensor);
        totalSumFloatTensor = outQueueFloat.DeQue<float>();
        //COPY OUT
        DataCopyExtParams outputCopyParams = { 1, (uint32_t)sizeof(T), 0, 0, 0 };
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, __bf16>) {
            Cast(outputLocal, totalSumFloatTensor, RoundMode::CAST_ROUND, 1);
            outQueue.EnQue(outputLocal);
            outputLocal = outQueue.DeQue<T>();
            DataCopyPad(outputGm[0], outputLocal, outputCopyParams);
        }
        else {
            DataCopyPad(outputGm[0], totalSumFloatTensor, outputCopyParams);
        }
        outQueue.FreeTensor(outputLocal);
       
    }
};



// KernelL1LossNoneBroadcast: 处理 reduction='none' 的广播情况，输出与输入同形状
template<typename T>
class KernelL1LossNoneBroadcast {
private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueTarget;
    
    // 运算辅助队列
    TQue<QuePosition::VECCALC, BUFFER_NUM> castQueueX;
    TQue<QuePosition::VECCALC, BUFFER_NUM> castQueueTarget;

    GlobalTensor<T> xGm;
    GlobalTensor<T> targetGm;
    GlobalTensor<T> outputGm;

    uint32_t totalLength, lastDimLength, tileLength;
    uint32_t startRow, endRow;
    uint32_t *reduce1;
    uint32_t *reduce2;
    uint32_t *shape;
    uint32_t dim;
    bool doBcast1 = false, doBcast2 = false;
    uint32_t curIdx[16];  // 坐标数组，假设最大维度不超过16

    // 广播标量缓存：避免同一 start 上重复从 GM 拷贝 1 个元素
    bool hasCachedX = false;
    bool hasCachedTarget = false;
    uint32_t cachedStart1 = 0;
    uint32_t cachedStart2 = 0;
    T cachedXScalar;
    T cachedTargetScalar;

public:
    __aicore__ inline KernelL1LossNoneBroadcast() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR target, GM_ADDR output,
                                uint32_t totalLength, uint32_t dim, uint32_t *shape, 
                                uint32_t *reduce1, uint32_t *reduce2,
                                TPipe* pipeIn)
    {
        this->pipe = pipeIn;
        this->reduce1 = reduce1;
        this->reduce2 = reduce2;
        this->shape = shape;
        this->dim = dim;
        this->totalLength = totalLength;

        // 1. 获取最后一维长度
        this->lastDimLength = this->shape[this->dim - 1];

        // 2. 多核任务切分（按行切分）
        uint32_t totalRows = this->totalLength / this->lastDimLength;
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockNum == 0) {
            blockNum = 1;
        }
        uint32_t rowsPerBlock = (totalRows + blockNum - 1) / blockNum;
        this->startRow = blockIdx * rowsPerBlock;
        uint32_t end = this->startRow + rowsPerBlock;
        this->endRow = (end > totalRows) ? totalRows : end;

        // 3. UB 空间分配（对齐）
        const int32_t CALC_ALIGN_NUM = 256;
        uint32_t alignUnit = CALC_ALIGN_NUM / sizeof(T);
        if (alignUnit < 32)
            alignUnit = 32;
        this->tileLength = ((this->lastDimLength + alignUnit - 1) / alignUnit) * alignUnit;
        
        // 限制最大 tile 长度
        uint32_t maxTileLength = 7168;
        if (this->tileLength > maxTileLength) {
            this->tileLength = maxTileLength;
        }

        // 4. GM 指针初始化
        xGm.SetGlobalBuffer((__gm__ T *)x);
        targetGm.SetGlobalBuffer((__gm__ T *)target);
        outputGm.SetGlobalBuffer((__gm__ T *)output);

        // 5. Pipe Buffer 初始化
        pipe->InitBuffer(inQueueX, 2, this->tileLength * sizeof(T));
        pipe->InitBuffer(inQueueTarget, 2, this->tileLength * sizeof(T));
        
        if constexpr (std::is_same_v<T, __bf16> || std::is_same_v<T, half>) {
            pipe->InitBuffer(castQueueX, 2, this->tileLength * sizeof(float));
            pipe->InitBuffer(castQueueTarget, 2, this->tileLength * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        // 初始化步长数组
        uint32_t d[21] = {0};
        uint32_t dn1[21] = {0};
        uint32_t dn2[21] = {0};
        InitializeDnArrays(d, dn1, this->reduce1, this->dim, this->shape);
        InitializeDnArrays(d, dn2, this->reduce2, this->dim, this->shape);

        this->doBcast1 = (this->reduce1[this->dim - 1] == 1);
        this->doBcast2 = (this->reduce2[this->dim - 1] == 1);

        // 初始化当前块起始行的坐标
        uint32_t tempLinear = this->startRow * this->lastDimLength;
        for (int k = this->dim - 1; k >= 0; k--)
        {
            uint32_t idx = (tempLinear / d[k + 1]) % this->shape[k];
            if (k < 16)
                this->curIdx[k] = idx;
        }

        // 逐行计算并写回 GM
        for (uint32_t j = this->startRow; j < this->endRow; j++)
        {
            uint32_t start1 = 0;
            uint32_t start2 = 0;

            // 使用预计算的坐标累加计算偏移
            for (int k = 0; k < this->dim - 1; k++)
            {
                if (this->reduce1[k] == 0)
                {
                    start1 += this->curIdx[k] * dn1[k + 1];
                }
                if (this->reduce2[k] == 0)
                {
                    start2 += this->curIdx[k] * dn2[k + 1];
                }
            }

            CopyIn(start1, start2);
            ComputeAndWriteBack(j);

            // 更新坐标：类似于计数器进位
            for (int k = this->dim - 2; k >= 0; k--)
            {
                this->curIdx[k]++;
                if (this->curIdx[k] < this->shape[k])
                {
                    break;
                }
                this->curIdx[k] = 0;
            }
        }
    }

private:
    __aicore__ inline void InitializeDnArrays(uint32_t *d, uint32_t *dn, uint32_t *reduce, uint32_t dim, uint32_t *shape)
    {
        d[dim] = dn[dim] = 1;
        for (int k = dim - 1; k >= 0; k--)
        {
            d[k] = d[k + 1] * shape[k];
            if (reduce[k] == 0)
                dn[k] = dn[k + 1] * shape[k];
            else
                dn[k] = dn[k + 1];
        }
    }

    // 安全的广播赋值（借鉴 fmax）
    __aicore__ inline void BroadcastT(LocalTensor<T> &tensor, T val, uint32_t len)
    {
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float>)
        {
            // Duplicate 可能需要对齐长度，先处理对齐部分，再用循环填充剩余
            constexpr uint32_t alignUnit = 32 / sizeof(T);  // 32字节对齐单位
            uint32_t alignedLen = (len / alignUnit) * alignUnit;
            
            if (alignedLen > 0) {
                Duplicate(tensor, val, alignedLen);
            }
            
            // 填充剩余的非对齐部分
            for (uint32_t i = alignedLen; i < len; ++i) {
                tensor.SetValue(i, val);
            }
        }
        else
        {
            for (uint32_t i = 0; i < len; ++i)
            {
                tensor.SetValue(i, val);
            }
        }
    }

    __aicore__ inline void CopyIn(uint32_t start1, uint32_t start2)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> targetLocal = inQueueTarget.AllocTensor<T>();

        DataCopyPadExtParams<T> padParams{ false, 0, 0, 0 };
        
        // 定义最小拷贝长度（32字节），避免非对齐标量拷贝失败
        uint32_t minCopyLen = 32 / sizeof(T);
        if (minCopyLen < 1)
            minCopyLen = 1;

        if (this->doBcast1) {
            // 广播：start1 可能在很多 row 上重复，命中缓存则跳过 GM->UB 拷贝
            if (this->hasCachedX && start1 == this->cachedStart1) {
                xLocal.SetValue(0, this->cachedXScalar);
            } else {
                DataCopyExtParams copyParams = { 1, (uint32_t)(minCopyLen * sizeof(T)), 0, 0, 0 };
                DataCopyPad(xLocal, xGm[start1], copyParams, padParams);
                this->cachedXScalar = xLocal.GetValue(0);
                this->cachedStart1 = start1;
                this->hasCachedX = true;
            }
        } else {
            // 非广播：拷贝完整长度
            DataCopyExtParams copyParams = { 1, (uint32_t)(this->lastDimLength * sizeof(T)), 0, 0, 0 };
            DataCopyPad(xLocal, xGm[start1], copyParams, padParams);
        }

        if (this->doBcast2) {
            // 广播：start2 可能在很多 row 上重复，命中缓存则跳过 GM->UB 拷贝
            if (this->hasCachedTarget && start2 == this->cachedStart2) {
                targetLocal.SetValue(0, this->cachedTargetScalar);
            } else {
                DataCopyExtParams copyParams = { 1, (uint32_t)(minCopyLen * sizeof(T)), 0, 0, 0 };
                DataCopyPad(targetLocal, targetGm[start2], copyParams, padParams);
                this->cachedTargetScalar = targetLocal.GetValue(0);
                this->cachedStart2 = start2;
                this->hasCachedTarget = true;
            }
        } else {
            // 非广播：拷贝完整长度
            DataCopyExtParams copyParams = { 1, (uint32_t)(this->lastDimLength * sizeof(T)), 0, 0, 0 };
            DataCopyPad(targetLocal, targetGm[start2], copyParams, padParams);
        }

        inQueueX.EnQue(xLocal);
        inQueueTarget.EnQue(targetLocal);
    }

    // 计算并写回 GM
    __aicore__ inline void ComputeAndWriteBack(uint32_t rowIndex)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> targetLocal = inQueueTarget.DeQue<T>();

        // 使用实际数据长度，而不是对齐后的 tileLength
        uint32_t calCount = this->lastDimLength;

        // 广播处理：使用 BroadcastT 安全函数，广播到实际计算长度
        if (this->doBcast1)
        {
            T val = xLocal.GetValue(0);
            BroadcastT(xLocal, val, calCount);
        }
        if (this->doBcast2)
        {
            T val = targetLocal.GetValue(0);
            BroadcastT(targetLocal, val, calCount);
        }

        // 计算 L1Loss: |x - target|
        LocalTensor<float> diffTensor;
        LocalTensor<float> xLocalFloat;
        LocalTensor<float> targetLocalFloat;

        if constexpr (std::is_same_v<T, float>) {
            // 直接在 float 上计算
            constexpr uint32_t alignUnit = 32 / sizeof(float);  // 8 for float32
            uint32_t alignedCount = (calCount / alignUnit) * alignUnit;
            uint32_t remainder = calCount - alignedCount;

            if (alignedCount > 0) {
                Sub(xLocal, xLocal, targetLocal, alignedCount);
                Abs(xLocal, xLocal, alignedCount);
            }

            if (remainder > 0) {
                for (uint32_t i = alignedCount; i < calCount; ++i) {
                    float diff = xLocal.GetValue(i) - targetLocal.GetValue(i);
                    float absDiff = (diff >= 0) ? diff : -diff;
                    xLocal.SetValue(i, absDiff);
                }
            }
        } else {
            if constexpr (std::is_same_v<T, __bf16>) {
                xLocalFloat = castQueueX.AllocTensor<float>();
                targetLocalFloat = castQueueTarget.AllocTensor<float>();
                Cast(xLocalFloat, xLocal, RoundMode::CAST_NONE, calCount);
                Cast(targetLocalFloat, targetLocal, RoundMode::CAST_NONE, calCount);
                // Cast 后立即释放 xLocal 和 targetLocal
                inQueueX.FreeTensor(xLocal);
                inQueueTarget.FreeTensor(targetLocal);
                
                castQueueX.EnQue(xLocalFloat);
                castQueueTarget.EnQue(targetLocalFloat);
                xLocalFloat = castQueueX.DeQue<float>();
                targetLocalFloat = castQueueTarget.DeQue<float>();

                Sub(targetLocalFloat, xLocalFloat, targetLocalFloat, calCount);
                Abs(targetLocalFloat, targetLocalFloat, calCount);
                diffTensor = targetLocalFloat;
            } else if constexpr (std::is_same_v<T, half>) {
                Sub(xLocal, xLocal, targetLocal, calCount);
                Abs(xLocal, xLocal, calCount);
                // 计算完成后，targetLocal 不再需要，立即释放
                inQueueTarget.FreeTensor(targetLocal);
                
                targetLocalFloat = castQueueTarget.AllocTensor<float>();
                Cast(targetLocalFloat, xLocal, RoundMode::CAST_NONE, calCount);
                // Cast 后立即释放 xLocal
                inQueueX.FreeTensor(xLocal);
                diffTensor = targetLocalFloat;
            }
        }

        // 写回 GM：输出与输入同形状
        DataCopyPadExtParams<T> padParams{ false, 0, 0, 0 };
        DataCopyExtParams copyParams = { 1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0 };
        uint32_t outOffset = rowIndex * this->lastDimLength;

        if constexpr (std::is_same_v<T, float>) {
            DataCopyPad(outputGm[outOffset], xLocal, copyParams);
            // 同步：确保数据写回完成
        } else {
            LocalTensor<T> outLocal = inQueueTarget.AllocTensor<T>();
            Cast(outLocal, diffTensor, RoundMode::CAST_ROUND, calCount);
            inQueueTarget.EnQue(outLocal);
            outLocal = inQueueTarget.DeQue<T>();
            DataCopyPad(outputGm[outOffset], outLocal, copyParams);
            // 同步：确保数据写回完成
            inQueueTarget.FreeTensor(outLocal);
        }

        // 释放 Tensor
        if constexpr (std::is_same_v<T, __bf16>) {
            castQueueX.FreeTensor(xLocalFloat);
            castQueueTarget.FreeTensor(targetLocalFloat);
        } else if constexpr (std::is_same_v<T, half>) {
            castQueueTarget.FreeTensor(targetLocalFloat);
        } else if constexpr (std::is_same_v<T, float>) {
            // float 类型需要释放 xLocal 和 targetLocal
            inQueueX.FreeTensor(xLocal);
            inQueueTarget.FreeTensor(targetLocal);
        }
    }
};

template<typename T>
class KernelL1LossBroadcast {
private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueTarget;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    
    // 运算辅助队列
    TQue<QuePosition::VECCALC, BUFFER_NUM> sumQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueFloat;
    TQue<QuePosition::VECCALC, BUFFER_NUM> castQueueX;
    TQue<QuePosition::VECCALC, BUFFER_NUM> castQueueTarget;
    TQue<QuePosition::VECCALC, BUFFER_NUM> castQueueDiff;
    TQue<QuePosition::VECCALC, BUFFER_NUM> calcQueue;

    GlobalTensor<T> xGm;
    GlobalTensor<T> targetGm;
    GlobalTensor<T> outputGm;

    uint32_t totalLength, lastDimLength, tileLength;
    uint32_t startRow, endRow;
    uint32_t *reduce1;
    uint32_t *reduce2;
    uint32_t *shape;
    uint32_t dim;
    bool doBcast1 = false, doBcast2 = false;
    uint32_t curIdx[16];  // 坐标数组，假设最大维度不超过16
    uint32_t reduction;  // 1='mean', 2='sum'

public:
    __aicore__ inline KernelL1LossBroadcast() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR target, GM_ADDR output,
                                uint32_t totalLength, uint32_t dim, uint32_t *shape, 
                                uint32_t *reduce1, uint32_t *reduce2, uint32_t reduction,
                                TPipe* pipeIn)
    {
        this->pipe = pipeIn;
        this->reduce1 = reduce1;
        this->reduce2 = reduce2;
        this->shape = shape;
        this->dim = dim;
        this->totalLength = totalLength;
        this->reduction = reduction;

        // 1. 获取最后一维长度
        this->lastDimLength = this->shape[this->dim - 1];

        // 2. 多核任务切分（单核处理，简化）
        this->startRow = 0;
        this->endRow = this->totalLength / this->lastDimLength;

        // 3. UB 空间分配（对齐）
        const int32_t CALC_ALIGN_NUM = 256;
        uint32_t alignUnit = CALC_ALIGN_NUM / sizeof(T);
        if (alignUnit < 32)
            alignUnit = 32;
        this->tileLength = ((this->lastDimLength + alignUnit - 1) / alignUnit) * alignUnit;
        
        // 限制最大 tile 长度
        uint32_t maxTileLength = 10240;
        if (this->tileLength > maxTileLength) {
            this->tileLength = maxTileLength;
        }

        // 4. GM 指针初始化
        xGm.SetGlobalBuffer((__gm__ T *)x);
        targetGm.SetGlobalBuffer((__gm__ T *)target);
        outputGm.SetGlobalBuffer((__gm__ T *)output);

        // 5. Pipe Buffer 初始化
        pipe->InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe->InitBuffer(inQueueTarget, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe->InitBuffer(outQueue, BUFFER_NUM, 32);
        pipe->InitBuffer(sumQueue, BUFFER_NUM, 32);
        pipe->InitBuffer(outQueueFloat, BUFFER_NUM, 32);
        pipe->InitBuffer(calcQueue, BUFFER_NUM, this->tileLength * sizeof(float));
        
        if constexpr (std::is_same_v<T, __bf16> || std::is_same_v<T, half>) {
            pipe->InitBuffer(castQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
            pipe->InitBuffer(castQueueTarget, BUFFER_NUM, this->tileLength * sizeof(float));
            pipe->InitBuffer(castQueueDiff, BUFFER_NUM, this->tileLength * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        // 初始化步长数组
        uint32_t d[21] = {0};
        uint32_t dn1[21] = {0};
        uint32_t dn2[21] = {0};
        InitializeDnArrays(d, dn1, this->reduce1, this->dim, this->shape);
        InitializeDnArrays(d, dn2, this->reduce2, this->dim, this->shape);

        this->doBcast1 = (this->reduce1[this->dim - 1] == 1);
        this->doBcast2 = (this->reduce2[this->dim - 1] == 1);

        // 初始化当前块起始行的坐标
        uint32_t tempLinear = this->startRow * this->lastDimLength;
        for (int k = this->dim - 1; k >= 0; k--)
        {
            uint32_t idx = (tempLinear / d[k + 1]) % this->shape[k];
            if (k < 16)
                this->curIdx[k] = idx;
        }

        LocalTensor<float> totalSumTensor = outQueueFloat.AllocTensor<float>();
        totalSumTensor.SetValue(0, 0.0f);

        // 循环处理每一行
        for (uint32_t j = this->startRow; j < this->endRow; j++)
        {
            uint32_t start1 = 0;
            uint32_t start2 = 0;

            // 使用预计算的坐标累加计算偏移
            for (int k = 0; k < this->dim - 1; k++)
            {
                if (this->reduce1[k] == 0)
                {
                    start1 += this->curIdx[k] * dn1[k + 1];
                }
                if (this->reduce2[k] == 0)
                {
                    start2 += this->curIdx[k] * dn2[k + 1];
                }
            }

            CopyIn(start1, start2);
            Compute(totalSumTensor);

            // 更新坐标：类似于计数器进位
            for (int k = this->dim - 2; k >= 0; k--)
            {
                this->curIdx[k]++;
                if (this->curIdx[k] < this->shape[k])
                {
                    break;
                }
                this->curIdx[k] = 0;
            }
        }

        // 后处理（Mean）
        if (reduction == 1) {
            float scale = 1.0f / totalLength;
            Muls(totalSumTensor, totalSumTensor, scale, 1);
        }

        // 输出结果
        outQueueFloat.EnQue(totalSumTensor);
        totalSumTensor = outQueueFloat.DeQue<float>();
        DataCopyExtParams outputCopyParams = { 1, (uint32_t)sizeof(T), 0, 0, 0 };
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, __bf16>) {
            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            Cast(outputLocal, totalSumTensor, RoundMode::CAST_ROUND, 1);
            outQueue.EnQue(outputLocal);
            outputLocal = outQueue.DeQue<T>();
            DataCopyPad(outputGm[0], outputLocal, outputCopyParams);
            outQueue.FreeTensor(outputLocal);
        }
        else {
            DataCopyPad(outputGm[0], totalSumTensor, outputCopyParams);
        }
        outQueueFloat.FreeTensor(totalSumTensor);
    }

    __aicore__ inline void InitializeDnArrays(uint32_t *d, uint32_t *dn, uint32_t *reduce, uint32_t dim, uint32_t *shape)
    {
        d[dim] = dn[dim] = 1;
        for (int k = dim - 1; k >= 0; k--)
        {
            d[k] = d[k + 1] * shape[k];
            if (reduce[k] == 0)
                dn[k] = dn[k + 1] * shape[k];
            else
                dn[k] = dn[k + 1];
        }
    }

    // 安全的广播赋值（借鉴 fmax）
    __aicore__ inline void BroadcastT(LocalTensor<T> &tensor, T val, uint32_t len)
    {
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float>)
        {
            // Duplicate 可能需要对齐长度，先处理对齐部分，再用循环填充剩余
            constexpr uint32_t alignUnit = 32 / sizeof(T);  // 32字节对齐单位
            uint32_t alignedLen = (len / alignUnit) * alignUnit;
            
            if (alignedLen > 0) {
                Duplicate(tensor, val, alignedLen);
            }
            
            // 填充剩余的非对齐部分
            for (uint32_t i = alignedLen; i < len; ++i) {
                tensor.SetValue(i, val);
            }
        }
        else
        {
            for (uint32_t i = 0; i < len; ++i)
            {
                tensor.SetValue(i, val);
            }
        }
    }

    __aicore__ inline void CopyIn(uint32_t start1, uint32_t start2)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> targetLocal = inQueueTarget.AllocTensor<T>();

        DataCopyPadExtParams<T> padParams{ false, 0, 0, 0 };
        
        // 定义最小拷贝长度（32字节），避免非对齐标量拷贝失败
        uint32_t minCopyLen = 32 / sizeof(T);
        if (minCopyLen < 1)
            minCopyLen = 1;

        if (this->doBcast1)
        {
            // 广播情况：只需要拷贝1个元素，后续会广播
            DataCopyExtParams copyParams = { 1, (uint32_t)(minCopyLen * sizeof(T)), 0, 0, 0 };
            DataCopyPad(xLocal, xGm[start1], copyParams, padParams);
        }
        else
        {
            // 非广播情况：拷贝完整长度
            DataCopyExtParams copyParams = { 1, (uint32_t)(this->lastDimLength * sizeof(T)), 0, 0, 0 };
            DataCopyPad(xLocal, xGm[start1], copyParams, padParams);
        }

        if (this->doBcast2)
        {
            // 广播情况：只需要拷贝1个元素，后续会广播
            DataCopyExtParams copyParams = { 1, (uint32_t)(minCopyLen * sizeof(T)), 0, 0, 0 };
            DataCopyPad(targetLocal, targetGm[start2], copyParams, padParams);
        }
        else
        {
            // 非广播情况：拷贝完整长度
            DataCopyExtParams copyParams = { 1, (uint32_t)(this->lastDimLength * sizeof(T)), 0, 0, 0 };
            DataCopyPad(targetLocal, targetGm[start2], copyParams, padParams);
        }

        inQueueX.EnQue(xLocal);
        inQueueTarget.EnQue(targetLocal);
    }

    __aicore__ inline void Compute(LocalTensor<float>& totalSumTensor)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> targetLocal = inQueueTarget.DeQue<T>();

        // 使用实际数据长度，而不是对齐后的 tileLength
        uint32_t calCount = this->lastDimLength;

        // 广播处理：使用 BroadcastT 安全函数，广播到实际计算长度
        if (this->doBcast1)
        {
            T val = xLocal.GetValue(0);
            BroadcastT(xLocal, val, calCount);
        }
        if (this->doBcast2)
        {
            T val = targetLocal.GetValue(0);
            BroadcastT(targetLocal, val, calCount);
        }

        // 计算 L1Loss: |x - target|
        LocalTensor<float> diffTensor;
        LocalTensor<float> xLocalFloat;
        LocalTensor<float> targetLocalFloat;

        if constexpr (std::is_same_v<T, float>) {
            // Sub 和 Abs 可能需要对齐长度，先处理对齐部分，然后手动处理剩余部分
            constexpr uint32_t alignUnit = 32 / sizeof(float);  // 8 for float32
            uint32_t alignedCount = (calCount / alignUnit) * alignUnit;
            uint32_t remainder = calCount - alignedCount;
            
            if (alignedCount > 0) {
                Sub(xLocal, xLocal, targetLocal, alignedCount);
                Abs(xLocal, xLocal, alignedCount);
            }
            
            // 手动处理剩余的非对齐部分
            if (remainder > 0) {
                for (uint32_t i = alignedCount; i < calCount; ++i) {
                    float diff = xLocal.GetValue(i) - targetLocal.GetValue(i);
                    float absDiff = (diff >= 0) ? diff : -diff;
                    xLocal.SetValue(i, absDiff);
                }
            }
            
            diffTensor = xLocal;
        } else {
            if constexpr (std::is_same_v<T, __bf16>) {
                xLocalFloat = castQueueX.AllocTensor<float>();
                targetLocalFloat = castQueueTarget.AllocTensor<float>();
                Cast(xLocalFloat, xLocal, RoundMode::CAST_NONE, calCount);
                Cast(targetLocalFloat, targetLocal, RoundMode::CAST_NONE, calCount);
                castQueueX.EnQue(xLocalFloat);
                castQueueTarget.EnQue(targetLocalFloat);
                xLocalFloat = castQueueX.DeQue<float>();
                targetLocalFloat = castQueueTarget.DeQue<float>();
                
                Sub(xLocalFloat, xLocalFloat, targetLocalFloat, calCount);
                Abs(xLocalFloat, xLocalFloat, calCount);
                diffTensor = xLocalFloat;
            } else if constexpr (std::is_same_v<T, half>) {
                Sub(xLocal, xLocal, targetLocal, calCount);
                Abs(xLocal, xLocal, calCount);
                diffTensor = castQueueDiff.AllocTensor<float>();
                Cast(diffTensor, xLocal, RoundMode::CAST_NONE, calCount);
            }
        }

        // ReduceSum
        LocalTensor<float> workLocal = calcQueue.AllocTensor<float>();
        LocalTensor<float> chunkSum = sumQueue.AllocTensor<float>();
        chunkSum.SetValue(0, 0.0f);
        
        // ReduceSum 可能需要对齐长度，先处理对齐部分，然后手动处理剩余部分
        constexpr uint32_t alignUnit = 32 / sizeof(float);  // 8 for float32
        uint32_t alignedCount = (calCount / alignUnit) * alignUnit;
        uint32_t remainder = calCount - alignedCount;
        
        if (alignedCount > 0) {
            ReduceSum<float>(chunkSum, diffTensor, workLocal, alignedCount);
        }
        
        // 手动处理剩余的非对齐部分
        if (remainder > 0) {
            float remainderSum = 0.0f;
            for (uint32_t i = alignedCount; i < calCount; ++i) {
                remainderSum += diffTensor.GetValue(i);
            }
            float currentSum = chunkSum.GetValue(0);
            chunkSum.SetValue(0, currentSum + remainderSum);
        }
        
        sumQueue.EnQue(chunkSum);
        chunkSum = sumQueue.DeQue<float>();
        Add(totalSumTensor, totalSumTensor, chunkSum, 1);

        // Free Tensors
        sumQueue.FreeTensor(chunkSum);
        calcQueue.FreeTensor(workLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueTarget.FreeTensor(targetLocal);
        if constexpr (std::is_same_v<T, __bf16>) {
            castQueueX.FreeTensor(xLocalFloat);
            castQueueTarget.FreeTensor(targetLocalFloat);
        } else if constexpr (std::is_same_v<T, half>) {
            castQueueDiff.FreeTensor(diffTensor);
        }
    }
};



extern "C" __global__ __aicore__ void l1_loss(GM_ADDR x, GM_ADDR target, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    
    if (TILING_KEY_IS(0)) {
            KernelL1LossMean<DTYPE_X> op;
            op.Init(x, target, output, workspace, 
                    tiling_data.N1, tiling_data.N2, tiling_data.N3, tiling_data.N4,
                    tiling_data.reduction,
                    &pipe);
            op.Process();
    }
    else if (TILING_KEY_IS(100)) {
        // 使用通用广播 kernel（mean/sum reduction）
        KernelL1LossBroadcast<DTYPE_X> op;
        op.Init(x, target, output, 
                tiling_data.totalLength, tiling_data.dim, 
                tiling_data.shape, tiling_data.reduce1, tiling_data.reduce2,
                tiling_data.reduction,
                &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(101)) {
        // 使用 none reduction 广播 kernel
        KernelL1LossNoneBroadcast<DTYPE_X> op;
        op.Init(x, target, output, 
                tiling_data.totalLength, tiling_data.dim, 
                tiling_data.shape, tiling_data.reduce1, tiling_data.reduce2,
                &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(200)) {
        
    }
    else if (TILING_KEY_IS(300)) {

    }
}
