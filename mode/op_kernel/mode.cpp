#include "kernel_operator.h"

using namespace AscendC;

enum ModeDataType {
    DTYPE_FLOAT32 = 0,
    DTYPE_FLOAT16 = 1,
    DTYPE_INT32   = 2,
    DTYPE_INT8    = 3,
    DTYPE_INT64   = 4,
    DTYPE_INT16   = 5,
    DTYPE_UINT8   = 6
};

template<typename T>
class KernelMode {
public:
    __aicore__ inline KernelMode() {}
    
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR values, GM_ADDR indices, GM_ADDR tiling) {
        GET_TILING_DATA(tiling_data, tiling);

        totalTasks = tiling_data.totalTasks;
        axisLen = tiling_data.axisLen;
        innerStride = tiling_data.innerStride;
        tasksPerBlock = tiling_data.tasksPerBlock;
        bigCoreNum = tiling_data.bigCoreNum;
        
        // 使用 Host 计算的最优 maxChunkSize
        maxChunkSize = tiling_data.maxChunkSize;
        
        coreId = GetBlockIdx();
        
        if (coreId < bigCoreNum) {
            taskNum = tasksPerBlock + 1;
            startTaskIdx = coreId * taskNum;
        } else {
            taskNum = tasksPerBlock;
            startTaskIdx = bigCoreNum * (tasksPerBlock + 1) + (coreId - bigCoreNum) * taskNum;
        }

        inputGm.SetGlobalBuffer((__gm__ T*)input);
        valuesGm.SetGlobalBuffer((__gm__ T*)values + startTaskIdx);
        indicesGm.SetGlobalBuffer((__gm__ int64_t*)indices + startTaskIdx);

        paddedTotalLength = (axisLen + 31) / 32 * 32;
        if (paddedTotalLength < 64) paddedTotalLength = 64;

        numChunks = (paddedTotalLength + maxChunkSize - 1) / maxChunkSize;
        uint32_t storageSize = paddedTotalLength * sizeof(float);
        
        uint32_t chunkAlignSize = 256; 
        uint32_t workSize4B = (maxChunkSize * 4 + chunkAlignSize) / chunkAlignSize * chunkAlignSize;
        uint32_t workSize8B = (maxChunkSize * 8 + chunkAlignSize) / chunkAlignSize * chunkAlignSize;
        uint32_t tmpSize = tiling_data.tmpSize; 

        // 仅在非 Float32 时分配 Input Queue
        if constexpr (!IsSameType<T, float>::value) {
            uint32_t inputChunkSize = (maxChunkSize * sizeof(T) + chunkAlignSize) / chunkAlignSize * chunkAlignSize;
            pipe.InitBuffer(inQueueX, 1, inputChunkSize); 
        }
        
        pipe.InitBuffer(sortStorageBuf, storageSize); 
        pipe.InitBuffer(concatBuf, workSize4B);
        pipe.InitBuffer(indexBuf, workSize4B);
        pipe.InitBuffer(dstIndexBuf, workSize4B); 
        pipe.InitBuffer(sortDstBuf, workSize8B); 
        pipe.InitBuffer(sortTmpBuf, tmpSize);
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < taskNum; i++) {
            Compute(i);
        }
    }

    __aicore__ inline float ToF(T val) {
        if constexpr (IsSameType<T, float>::value) return (float)val;
        else if constexpr (IsSameType<T, half>::value) return (float)val;
        else if constexpr (IsSameType<T, int32_t>::value) return (float)val;
        else if constexpr (IsSameType<T, int64_t>::value) return (float)val;
        else return (float)(int32_t)val;
    }

private:
    __aicore__ inline uint64_t GetInputOffset(uint32_t localTaskIdx) {
        uint32_t globalTaskIdx = startTaskIdx + localTaskIdx;
        uint32_t rightDimSize = innerStride; 
        uint32_t outerIdx = globalTaskIdx / rightDimSize;
        uint32_t innerIdx = globalTaskIdx % rightDimSize;
        return (uint64_t)outerIdx * axisLen * rightDimSize + innerIdx;
    }

    __aicore__ inline void Compute(uint32_t index) {
        LocalTensor<float> storageLocal = sortStorageBuf.Get<float>();
        LocalTensor<float> concatLocal = concatBuf.Get<float>();
        
        // 【修改点 1】：Sort需要 uint32_t，但 ArithProgression 生成时用 int32_t 避免编译器报错
        LocalTensor<uint32_t> indexLocal = indexBuf.Get<uint32_t>();
        LocalTensor<int32_t> indexLocalInt32 = indexBuf.Get<int32_t>(); // 共享同一块内存
        
        LocalTensor<uint32_t> dstIndexLocal = dstIndexBuf.Get<uint32_t>();
        LocalTensor<float> dstLocal = sortDstBuf.Get<float>();
        LocalTensor<float> tmpLocal = sortTmpBuf.Get<float>();
        
        uint64_t inputStartOffset = GetInputOffset(index);
        float padVal = -3.40282e+38F;

        // Phase 1: Load, Convert, Sort by Chunks
        for (uint32_t k = 0; k < numChunks; k++) {
            uint32_t chunkStart = k * maxChunkSize;
            uint32_t chunkLen = maxChunkSize;
            if (chunkStart + chunkLen > paddedTotalLength) chunkLen = paddedTotalLength - chunkStart;
            
            // Fast Path: 连续内存读取 (innerStride == 1)
            if (innerStride == 1) {
                uint32_t copyLen = 0;
                if (chunkStart < axisLen) {
                    copyLen = axisLen - chunkStart;
                    if (copyLen > chunkLen) copyLen = chunkLen;
                }

                // Float32 特化路径 - Zero Copy & Vectorization
                if constexpr (IsSameType<T, float>::value) {
                    // 【修复编译错误】：使用 int32_t 类型生成索引 0, 1, 2...
                    // 只要 chunkLen < 2^31，int32_t 和 uint32_t 的二进制表示是一样的，且 float<->int32 转换合法
                    ArithProgression(indexLocalInt32, (int32_t)0, (int32_t)1, chunkLen);

                    // 2. 直接搬运 GM -> concatBuf (UB)
                    if (copyLen > 0) {
                        DataCopyExtParams copyParams{1, (uint32_t)(copyLen * sizeof(float)), 0, 0, 0};
                        DataCopyPadExtParams<float> padParams{true, 0, 0, padVal};
                        DataCopyPad(concatLocal, inputGm[inputStartOffset + chunkStart], copyParams, padParams);
                    } else {
                        Duplicate(concatLocal, padVal, chunkLen);
                    }
                    
                    // PipeBarrier 确保搬运完成
                    PipeBarrier<PIPE_ALL>();

                } else {
                    // 非 Float32 保持原有逻辑
                    LocalTensor<T> inputChunk = inQueueX.AllocTensor<T>();
                    
                    DataCopyExtParams copyParams{1, (uint32_t)(copyLen * sizeof(T)), 0, 0, 0};
                    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
                    if (copyLen > 0) {
                        DataCopyPad(inputChunk, inputGm[inputStartOffset + chunkStart], copyParams, padParams);
                    }
                    inQueueX.EnQue(inputChunk);
                    LocalTensor<T> curInput = inQueueX.DeQue<T>();
                    
                    for (uint32_t i = 0; i < chunkLen; i++) {
                        indexLocal.SetValue(i, i);
                        if (i < copyLen) concatLocal.SetValue(i, ToF(curInput.GetValue(i)));
                        else concatLocal.SetValue(i, padVal);
                    }
                    inQueueX.FreeTensor(curInput);
                }
            } else {
                // Strided Load
                 if constexpr (IsSameType<T, float>::value) {
                     // 【修复编译错误】：Strided 路径也同样修复
                     ArithProgression(indexLocalInt32, (int32_t)0, (int32_t)1, chunkLen);

                     for (uint32_t i = 0; i < chunkLen; i++) {
                        uint32_t logicIdx = chunkStart + i;
                        if (logicIdx < axisLen) {
                            concatLocal.SetValue(i, inputGm.GetValue(inputStartOffset + logicIdx * innerStride));
                        } else {
                            concatLocal.SetValue(i, padVal);
                        }
                    }
                 } else {
                     for (uint32_t i = 0; i < chunkLen; i++) {
                        indexLocal.SetValue(i, i);
                        uint32_t logicIdx = chunkStart + i;
                        if (logicIdx < axisLen) {
                            T val = inputGm.GetValue(inputStartOffset + logicIdx * innerStride);
                            concatLocal.SetValue(i, ToF(val));
                        } else {
                            concatLocal.SetValue(i, padVal);
                        }
                    }
                 }
            }
            
            // Sort
            int32_t repeat = chunkLen / 32;
            if (repeat > 0) {
                // indexLocal (uint32_t) 此时已经包含了正确的数据（由 ArithProgression 通过 int32 视图写入）
                Sort<float, true>(dstLocal, concatLocal, indexLocal, tmpLocal, repeat);
                
                LocalTensor<float> targetSlice = storageLocal[chunkStart];
                Extract(targetSlice, dstIndexLocal, dstLocal, repeat);
            }
            PipeBarrier<PIPE_ALL>();
        }

        // Phase 2: Merge Scan (Scalar)
        uint32_t ptrs[32]; 
        for(int k=0; k<numChunks; k++) ptrs[k] = 0;
        
        float curVal = -3.40282e+38F;
        int32_t curCount = 0;
        float modeValF = 0.0f;
        int32_t maxCount = -1;
        bool first = true;
        
        for(uint32_t step = 0; step < axisLen; step++) {
            float maxHeadVal = -3.40282e+38F;
            int maxHeadIdx = -1;
            
            for(int k=0; k<numChunks; k++) {
                uint32_t chunkStart = k * maxChunkSize;
                uint32_t chunkLen = maxChunkSize;
                if (chunkStart + chunkLen > paddedTotalLength) chunkLen = paddedTotalLength - chunkStart;
                
                if (ptrs[k] < chunkLen) {
                    float val = storageLocal.GetValue(chunkStart + ptrs[k]);
                    if (val > maxHeadVal || maxHeadIdx == -1) {
                        maxHeadVal = val;
                        maxHeadIdx = k;
                    }
                }
            }
            
            if (maxHeadIdx != -1) {
                ptrs[maxHeadIdx]++;
                if (first) {
                    curVal = maxHeadVal; curCount = 1;
                    modeValF = maxHeadVal; maxCount = 1;
                    first = false;
                } else {
                    if (maxHeadVal == curVal) {
                        curCount++;
                    } else {
                        if (curCount >= maxCount) { maxCount = curCount; modeValF = curVal; }
                        curVal = maxHeadVal; curCount = 1;
                    }
                }
            }
        }
        if (curCount >= maxCount) { modeValF = curVal; }

        // Phase 3: Find Index (Reload & Search)
        T finalModeVal = (T)0;
        int64_t finalModeIdx = 0;
        bool found = false;
        
        if (innerStride == 1) {
            LocalTensor<T> reloadBuf = sortDstBuf.Get<T>(); // 复用 Buffer

            for (uint32_t k = 0; k < numChunks; k++) {
                if (found) break;
                uint32_t chunkStart = k * maxChunkSize;
                uint32_t loadLen = 0;
                if (chunkStart < axisLen) {
                    loadLen = axisLen - chunkStart;
                    if (loadLen > maxChunkSize) loadLen = maxChunkSize;
                } else break;
                
                DataCopyExtParams p{1, (uint32_t)(loadLen * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> pp{true, 0, 0, (T)0};
                DataCopyPad(reloadBuf, inputGm[inputStartOffset + chunkStart], p, pp);
                PipeBarrier<PIPE_ALL>();
                
                if constexpr (IsSameType<T, float>::value) {
                    for (uint32_t i = 0; i < loadLen; i++) {
                        T raw = reloadBuf.GetValue(i);
                        if (raw == modeValF) { 
                            finalModeVal = raw;
                            finalModeIdx = (int64_t)(chunkStart + i);
                            found = true; break;
                        }
                    }
                } else {
                    for (uint32_t i = 0; i < loadLen; i++) {
                        T raw = reloadBuf.GetValue(i);
                        if (ToF(raw) == modeValF) {
                            finalModeVal = raw;
                            finalModeIdx = (int64_t)(chunkStart + i);
                            found = true; break;
                        }
                    }
                }
                PipeBarrier<PIPE_ALL>();
            }
        } else {
            // Generic Path (Stride)
            for (uint32_t i = 0; i < axisLen; i++) {
                T raw = inputGm.GetValue(inputStartOffset + i * innerStride);
                bool isEqual = false;
                if constexpr (IsSameType<T, float>::value) isEqual = (raw == modeValF);
                else isEqual = (ToF(raw) == modeValF);

                if (isEqual) {
                    finalModeVal = raw;
                    finalModeIdx = (int64_t)i;
                    found = true;
                    break;
                }
            }
        }
        
        if (!found) {
            finalModeVal = inputGm.GetValue(inputStartOffset);
            finalModeIdx = 0;
        }

        // 写回结果
        LocalTensor<T> outValLocal = concatBuf.Get<T>();
        LocalTensor<int64_t> outIdxLocal = indexBuf.Get<int64_t>();

        outValLocal.SetValue(0, finalModeVal);
        outIdxLocal.SetValue(0, finalModeIdx);

        DataCopyExtParams valParams{1, (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyPad(valuesGm[index], outValLocal, valParams);

        DataCopyExtParams idxParams{1, (uint32_t)sizeof(int64_t), 0, 0, 0};
        DataCopyPad(indicesGm[index], outIdxLocal, idxParams);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    
    TBuf<TPosition::VECCALC> sortStorageBuf; 
    TBuf<TPosition::VECCALC> concatBuf;
    TBuf<TPosition::VECCALC> indexBuf;
    TBuf<TPosition::VECCALC> dstIndexBuf; 
    TBuf<TPosition::VECCALC> sortDstBuf; 
    TBuf<TPosition::VECCALC> sortTmpBuf;
    
    GlobalTensor<T> inputGm;
    GlobalTensor<T> valuesGm;
    GlobalTensor<int64_t> indicesGm;

    uint32_t totalTasks;
    uint32_t axisLen;
    uint32_t innerStride;
    uint32_t tasksPerBlock;
    uint32_t bigCoreNum;
    uint32_t paddedTotalLength;
    uint32_t startTaskIdx;
    uint32_t taskNum;
    uint32_t coreId;
    
    uint32_t maxChunkSize; 
    uint32_t numChunks;
};

extern "C" __global__ __aicore__ void mode(GM_ADDR input, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    uint32_t dtype = tiling_data.dtype;

    if (dtype == DTYPE_FLOAT32) {
        KernelMode<float> op;
        op.Init(input, values, indices, tiling);
        op.Process();
    } 
    else if (dtype == DTYPE_FLOAT16) {
        KernelMode<half> op;
        op.Init(input, values, indices, tiling);
        op.Process();
    }
    else if (dtype == DTYPE_INT32) {
        KernelMode<int32_t> op;
        op.Init(input, values, indices, tiling);
        op.Process();
    }
    else if (dtype == DTYPE_INT8) {
        KernelMode<int8_t> op;
        op.Init(input, values, indices, tiling);
        op.Process();
    }
    else if (dtype == DTYPE_INT64) {
        KernelMode<int64_t> op;
        op.Init(input, values, indices, tiling);
        op.Process();
    }
    else if (dtype == DTYPE_INT16) {
        KernelMode<int16_t> op;
        op.Init(input, values, indices, tiling);
        op.Process();
    }
    else if (dtype == DTYPE_UINT8) {
        KernelMode<uint8_t> op;
        op.Init(input, values, indices, tiling);
        op.Process();
    }
}