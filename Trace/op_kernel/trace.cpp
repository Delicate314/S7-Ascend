#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
const uint32_t ALIGN_NUM = 8; // UB 中相邻有效元素之间的跨度（与 DataCopyPad 的 padding 配合使用）

__aicore__ inline int RoundUp(int a, int b)
{ 
    return (a + b - 1) / b;
}

// 通用整数 Trace 内核：输入类型为 T，累加、输出统一使用 int64_t，避免溢出
template<typename T>
class KernelTrace {
    private:
        TPipe* pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueData;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOutput;
    
        GlobalTensor<T> dataGm;
        GlobalTensor<int64_t> outputGm;
        int tile_length;
        uint32_t N1; 
        uint32_t N2;
    
    public:
        __aicore__ inline KernelTrace() {}
    
        __aicore__ inline void Init(GM_ADDR data, GM_ADDR output, GM_ADDR workspace, uint32_t N1, uint32_t N2, TPipe* pipeIn)
        {
            this->tile_length = 3584;
            this->pipe = pipeIn;
            this->N1 = N1;
            this->N2 = N2;
            dataGm.SetGlobalBuffer((__gm__ T*)data);
            outputGm.SetGlobalBuffer((__gm__ int64_t*)output);
            int alignNum = 32 / sizeof(T);
            // UB buffer 大小：按照最大 tile 长度和对齐跨度预申请
            uint32_t bufferSize = this->tile_length * alignNum * sizeof(T);
            pipe->InitBuffer(inQueueData,BUFFER_NUM, bufferSize);
            
            // 输出队列初始化：只需要存一个 int64，给 32 字节足够了
            pipe->InitBuffer(outQueueOutput,BUFFER_NUM, 32);
        }
    
        __aicore__ inline void Process()
        {
            int64_t total_sum = 0;
            int matrix_row = this->N1<=this->N2?this->N1:this->N2;
            uint32_t dataSize = sizeof(T); 
            uint32_t blockLen = dataSize;
            uint32_t srcStride = this->N2 * dataSize;
            uint32_t dstStride = 0;
    
            DataCopyExtParams copyParams;
            copyParams.blockLen = blockLen;   
            copyParams.srcStride = srcStride; 
            copyParams.dstStride = dstStride; 
            copyParams.rsv = 0;
    
            DataCopyPadExtParams<T> padParams;
            padParams.isPad = true;         
            padParams.leftPadding = 0;      
            padParams.rightPadding = (32 / sizeof(T)) - 1;    
            padParams.paddingValue = static_cast<T>(0);
    
            for (uint32_t i = 0; i < matrix_row; i += this->tile_length) {
                
                uint32_t currentTile = (matrix_row - i < this->tile_length) ? (matrix_row - i) : this->tile_length;
                
                LocalTensor<T> dataLocal = inQueueData.AllocTensor<T>();
    
                copyParams.blockCount = (uint16_t)currentTile;
                uint64_t gmOffset = (uint64_t)i * this->N1 + i;
    
                DataCopyPad(dataLocal, dataGm[gmOffset], copyParams, padParams);
    
                inQueueData.EnQue(dataLocal);
                dataLocal = inQueueData.DeQue<T>();
                
                // 2. 累加计算
                // dataLocal.GetValue 返回 T，在加法运算中统一提升为 int64_t
                for (uint32_t k = 0; k < currentTile; ++k) {
                    uint32_t jumpStride = 32 / sizeof(T); // int32是8，int8是32
                    total_sum += static_cast<int64_t>(dataLocal.GetValue(k * jumpStride)); 
                }
    
                inQueueData.FreeTensor(dataLocal);
            }
            LocalTensor<int64_t> sumInt64 = outQueueOutput.AllocTensor<int64_t>();
            sumInt64.SetValue(0, total_sum);
            outQueueOutput.EnQue(sumInt64);
            sumInt64 = outQueueOutput.DeQue<int64_t>();
            DataCopyExtParams outParams = {1, (uint32_t)(1 * sizeof(int64_t)), 0, 0, 0};
            DataCopyPad(outputGm[0], sumInt64, outParams);
    
            outQueueOutput.FreeTensor(sumInt64);
        }
    };


class KernelTraceFP32 {
private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueData;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOutput;
    TQue<QuePosition::VECCALC, BUFFER_NUM> calcQueue;  
    TQue<QuePosition::VECCALC, BUFFER_NUM> sumQueue; 
    GlobalTensor<float> dataGm;
    GlobalTensor<float> outputGm;
    int tile_length;
    uint32_t N1;  
    uint32_t N2;
public:
    __aicore__ inline KernelTraceFP32() {}

    __aicore__ inline void Init(GM_ADDR data, GM_ADDR output, GM_ADDR workspace, uint32_t N1, uint32_t N2, TPipe* pipeIn)
    {
        this->pipe = pipeIn;
        this->N1 = N1;
        this->N2 = N2;
        this->tile_length = 4064;
        dataGm.SetGlobalBuffer((__gm__ float*)data);
        outputGm.SetGlobalBuffer((__gm__ float*)output);
        pipe->InitBuffer(inQueueData, 1, this->tile_length*8*sizeof(float));
        pipe->InitBuffer(outQueueOutput, 1, 32);
        pipe->InitBuffer(sumQueue, 1, 32);
        pipe->InitBuffer(calcQueue, 1, 512*sizeof(float));
    }

    __aicore__ inline void Process()
    {
        float total_sum = 0.0f;
        int matrix_row = this->N1<=this->N2?this->N1:this->N2;
        uint32_t dataSize = sizeof(float); 
        uint32_t blockLen = dataSize;
        uint32_t srcStride = this->N2 * dataSize;
        uint32_t dstStride = 0;

        DataCopyExtParams copyParams;
        copyParams.blockLen = blockLen;   
        copyParams.srcStride = srcStride; 
        copyParams.dstStride = dstStride; 

        DataCopyPadExtParams<float> padParams;
        padParams.isPad = true;         
        padParams.leftPadding = 0;      
        padParams.rightPadding = 7;    
        padParams.paddingValue = 0.0f;

        DataCopyExtParams outParams = {1, (uint32_t)(1 * sizeof(float)), 0, 0, 0};
        uint32_t currentTile=this->tile_length;
        LocalTensor<float> dataLocal = inQueueData.AllocTensor<float>();
        LocalTensor<float> workLocal = calcQueue.AllocTensor<float>();
        LocalTensor<float> sumFloat = sumQueue.AllocTensor<float>();
        for (uint32_t i = 0; i < matrix_row; i += currentTile) {
            if(i+this->tile_length>matrix_row){
                currentTile = matrix_row - i;
            }
               
            copyParams.blockCount = (uint16_t)currentTile;
            uint64_t gmOffset = (uint64_t)i * this->N2 + i;
            DataCopyPad(dataLocal, dataGm[gmOffset], copyParams, padParams);
            inQueueData.EnQue(dataLocal);
            dataLocal = inQueueData.DeQue<float>();
            ReduceSum<float>(sumFloat, dataLocal, workLocal, currentTile*8);
            total_sum += sumFloat.GetValue(0);
            // printf("[DEBUG] total_sum = %f\n", total_sum);
        }
        LocalTensor<float> outFloat = outQueueOutput.AllocTensor<float>();
        outFloat.SetValue(0, total_sum);
        outQueueOutput.EnQue(outFloat);
        outFloat = outQueueOutput.DeQue<float>();
        DataCopyPad(outputGm[0], outFloat, outParams);
        outQueueOutput.FreeTensor(outFloat);
        inQueueData.FreeTensor(dataLocal);
        sumQueue.FreeTensor(sumFloat);
        calcQueue.FreeTensor(workLocal);
    }
};

template<typename T>
class KernelTraceInt {
    private:
        TPipe* pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueData;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueDataFloat;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOutput;
        TQue<QuePosition::VECCALC, BUFFER_NUM> calcQueue;       // 用于 ReduceSum 的 workLocal
        TQue<QuePosition::VECCALC, BUFFER_NUM> calcQueuefloat;  // 用于 ReduceSum 的 workLocal
        GlobalTensor<T> dataGm;
        GlobalTensor<int64_t> outputGm;
    
        uint32_t N1;  // 矩阵维度 (N x N)
        uint32_t N2;
        int align_num;
        int tile_length;
        int max_tile_length;
    public:
        __aicore__ inline KernelTraceInt() {}
    
        __aicore__ inline void Init(GM_ADDR data, GM_ADDR output, GM_ADDR workspace, uint32_t N1, uint32_t N2, TPipe* pipeIn)
        {
            this->pipe = pipeIn;
            this->N1 = N1;
            this->N2 = N2;
            this->tile_length = 1024;
            this->align_num = 32 / sizeof(T);

            dataGm.SetGlobalBuffer((__gm__ T*)data);
            outputGm.SetGlobalBuffer((__gm__ int64_t*)output);
            
            uint32_t bufferSize = this->tile_length * this->align_num * sizeof(T);
            pipe->InitBuffer(inQueueData, BUFFER_NUM, bufferSize);
            // dataLocal 转 float 需要单独的 float buffer
            uint32_t bufferSizeFloat = this->tile_length * this->align_num * sizeof(float);
            pipe->InitBuffer(inQueueDataFloat, BUFFER_NUM, bufferSizeFloat);
            pipe->InitBuffer(outQueueOutput, BUFFER_NUM, 32);
            pipe->InitBuffer(calcQueuefloat, BUFFER_NUM, 32);
            pipe->InitBuffer(calcQueue, BUFFER_NUM, 1024*sizeof(float));
        }
    
        __aicore__ inline void Process()
        {
            int64_t total_sum = 0;
            int matrix_row = this->N1<=this->N2?this->N1:this->N2;
            uint32_t dataSize = sizeof(T); 
            uint32_t blockLen = dataSize;
            uint32_t srcStride = this->N2 * dataSize;
            uint32_t dstStride = 0;
            DataCopyExtParams copyParams;
            copyParams.blockLen = blockLen;   
            copyParams.srcStride = srcStride; 
            copyParams.dstStride = dstStride; 
            int padnum = (32 / sizeof(T)) - 1;
            DataCopyPadExtParams<T> padParams;
            padParams.isPad = true;         
            padParams.leftPadding = padnum;      
            padParams.rightPadding = 0;    
            padParams.paddingValue = static_cast<T>(0);
            LocalTensor<int64_t> sumInt64;
            LocalTensor<float> workLocal = calcQueue.AllocTensor<float>();
            LocalTensor<float> sumFloat = calcQueuefloat.AllocTensor<float>();
            for (uint32_t i = 0; i < matrix_row; i += this->tile_length) {
                uint32_t currentTile = (matrix_row - i < this->tile_length) ? (matrix_row - i) : this->tile_length;
                LocalTensor<T> dataLocal = inQueueData.AllocTensor<T>();
                
                sumFloat.SetValue(0, 0.0f);
                copyParams.blockCount = (uint16_t)currentTile;
                uint64_t gmOffset = (uint64_t)i * this->N2 + i;
                DataCopyPad(dataLocal, dataGm[gmOffset], copyParams, padParams);
                inQueueData.EnQue(dataLocal);
                dataLocal = inQueueData.DeQue<T>();
                // for (uint32_t k = 0; k < 16; ++k) {
                //     printf("[DEBUG] dataLocal = %d\n", dataLocal.GetValue(k));
                // }
                // 使用 Cast 将整数数据转换为 float，再做后续计算
                LocalTensor<float> dataLocalFloat = inQueueDataFloat.AllocTensor<float>();
                Cast(dataLocalFloat, dataLocal, RoundMode::CAST_NONE, currentTile * this->align_num);
                // for (uint32_t k = 0; k < 16; ++k) {
                //     printf("[DEBUG] dataLocalFloat = %f\n", dataLocalFloat.GetValue(k));
                // }
                ReduceSum<float>(sumFloat, dataLocalFloat, workLocal, currentTile * this->align_num);
                // printf("[DEBUG] sumFloat (float) = %f\n", sumFloat.GetValue(0));
                total_sum += sumFloat.ReinterpretCast<float>().GetValue(0);
                // printf("[DEBUG] total_sum = %d\n", total_sum);
                inQueueData.FreeTensor(dataLocal);
                inQueueDataFloat.FreeTensor(dataLocalFloat);
            }
            sumInt64 = outQueueOutput.AllocTensor<int64_t>();
            sumInt64.SetValue(0, total_sum);
            outQueueOutput.EnQue(sumInt64);
            sumInt64 = outQueueOutput.DeQue<int64_t>();
            DataCopyExtParams outParams = {1, (uint32_t)(1 * sizeof(int64_t)), 0, 0, 0};
            DataCopyPad(outputGm[0], sumInt64, outParams);
            calcQueuefloat.FreeTensor(sumFloat);
            calcQueue.FreeTensor(workLocal);
            outQueueOutput.FreeTensor(sumInt64);
        }
};

template<typename T>
class KernelTraceInt8 {
    private:
        TPipe* pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueData;
        // int8/uint8 转 half 计算
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueDataHalf;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueDataFloat;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOutput;
        // 用于 ReduceSum 的 workLocal
        TQue<QuePosition::VECCALC, BUFFER_NUM> calcQueue;
        // 存放累积结果
        TQue<QuePosition::VECCALC, BUFFER_NUM> calcQueueFloat;
        GlobalTensor<T> dataGm;
        GlobalTensor<int64_t> outputGm;
    
        uint32_t N1;  // 矩阵维度 (N x N)
        uint32_t N2;
        int align_num;
        int tile_length;
    public:
        __aicore__ inline KernelTraceInt8() {}
    
        __aicore__ inline void Init(GM_ADDR data, GM_ADDR output, GM_ADDR workspace, uint32_t N1, uint32_t N2, TPipe* pipeIn)
        {
            this->pipe = pipeIn;
            this->N1 = N1;
            this->N2 = N2;
            this->tile_length = 512;
            this->align_num = 32 / sizeof(T);

            dataGm.SetGlobalBuffer((__gm__ T*)data);
            outputGm.SetGlobalBuffer((__gm__ int64_t*)output);
            
            uint32_t bufferSize = this->tile_length * this->align_num * sizeof(T);
            pipe->InitBuffer(inQueueData, BUFFER_NUM, bufferSize);
            // dataLocal 转 half 需要单独的 half buffer
            uint32_t bufferSizeHalf = this->tile_length * this->align_num * sizeof(half);
            uint32_t bufferSizeFloat = this->tile_length * this->align_num * sizeof(float);
            pipe->InitBuffer(inQueueDataHalf, BUFFER_NUM, bufferSizeHalf);
            pipe->InitBuffer(inQueueDataFloat, BUFFER_NUM, bufferSizeFloat);
            pipe->InitBuffer(outQueueOutput, BUFFER_NUM, 32);
            pipe->InitBuffer(calcQueueFloat, BUFFER_NUM, 32);
            pipe->InitBuffer(calcQueue, BUFFER_NUM, 512*sizeof(float));
        }
    
        __aicore__ inline void Process()
        {
            int64_t total_sum = 0;
            int matrix_row = this->N1<=this->N2?this->N1:this->N2;
            uint32_t dataSize = sizeof(T); 
            uint32_t blockLen = dataSize;
            uint32_t srcStride = this->N2 * dataSize;
            uint32_t dstStride = 0;
            
            DataCopyExtParams copyParams;
            copyParams.blockLen = blockLen;   
            copyParams.srcStride = srcStride; 
            copyParams.dstStride = dstStride; 
            int padnum = (32 / sizeof(T)) - 1;
            DataCopyPadExtParams<T> padParams;
            padParams.isPad = true;         
            padParams.leftPadding = 0;      
            padParams.rightPadding = padnum;    
            padParams.paddingValue = static_cast<T>(0);
            LocalTensor<int64_t> sumInt64;
    
            for (uint32_t i = 0; i < matrix_row; i += this->tile_length) {
                uint32_t currentTile = (matrix_row - i < this->tile_length) ? (matrix_row - i) : this->tile_length;
                LocalTensor<T> dataLocal = inQueueData.AllocTensor<T>();
                // 使用 half 作为中间计算精度
                LocalTensor<float> workLocal = calcQueue.AllocTensor<float>();
                LocalTensor<float> sumFloat = calcQueueFloat.AllocTensor<float>();
                sumFloat.SetValue(0, 0.0f);
                copyParams.blockCount = (uint16_t)currentTile;
                uint64_t gmOffset = (uint64_t)i * this->N2 + i;
                DataCopyPad(dataLocal, dataGm[gmOffset], copyParams, padParams);
                inQueueData.EnQue(dataLocal);
                dataLocal = inQueueData.DeQue<T>();
                // for (uint32_t k = 0; k < 8; ++k) {
                //     printf("[DEBUG] dataLocal = %d\n", dataLocal.GetValue(k));
                // }
                // 使用 Cast 将 int8/uint8 转换为 half，再做后续计算
                LocalTensor<half> dataLocalHalf = inQueueDataHalf.AllocTensor<half>();
                Cast(dataLocalHalf, dataLocal, RoundMode::CAST_NONE, currentTile * this->align_num);
                // for (uint32_t k = 0; k < 8; ++k) {
                //     printf("[DEBUG] dataLocalHalf = %f\n", dataLocalHalf.GetValue(k));
                // }
                LocalTensor<float> dataLocalFloat = inQueueDataFloat.AllocTensor<float>();
                Cast(dataLocalFloat, dataLocalHalf, RoundMode::CAST_NONE, currentTile * this->align_num);
                // for (uint32_t k = 0; k < 8; ++k) {
                //     printf("[DEBUG] dataLocalFloat = %f\n", dataLocalFloat.GetValue(k));
                // }
                // 在 half 精度下做 ReduceSum
                ReduceSum<float>(sumFloat, dataLocalFloat, workLocal, currentTile * this->align_num);
                // 将 half 累积结果转换成 float，再累加到 int64 total_sum
                // printf("[DEBUG] sumFloat = %f\n", sumFloat.GetValue(0));
                total_sum += sumFloat.GetValue(0);
                inQueueData.FreeTensor(dataLocal);
                inQueueDataHalf.FreeTensor(dataLocalHalf);
                inQueueDataFloat.FreeTensor(dataLocalFloat);
                calcQueueFloat.FreeTensor(sumFloat);
                calcQueue.FreeTensor(workLocal);
            }
            sumInt64 = outQueueOutput.AllocTensor<int64_t>();
            sumInt64.SetValue(0, total_sum);
            outQueueOutput.EnQue(sumInt64);
            sumInt64 = outQueueOutput.DeQue<int64_t>();
            DataCopyExtParams outParams = {1, (uint32_t)(1 * sizeof(int64_t)), 0, 0, 0};
            DataCopyPad(outputGm[0], sumInt64, outParams);
    
            outQueueOutput.FreeTensor(sumInt64);
        }
};

extern "C" __global__ __aicore__ void trace(GM_ADDR data, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    if (tiling_data.tilingKey == 0) {
        // fp32
        KernelTraceFP32 op;
        op.Init(data, output, workspace, tiling_data.N1, tiling_data.N2, &pipe);
        op.Process();
    } else if (tiling_data.tilingKey == 1) {
        // int32
        KernelTraceInt<int32_t> op;
        op.Init(data, output, workspace, tiling_data.N1, tiling_data.N2, &pipe);
        op.Process();
    } else if (tiling_data.tilingKey == 2) {
        // int8
        KernelTraceInt<int16_t> op;
        op.Init(data, output, workspace, tiling_data.N1, tiling_data.N2, &pipe);
        op.Process();
    } else if (tiling_data.tilingKey == 3) {
        KernelTraceInt8<int8_t> op;
        op.Init(data, output, workspace, tiling_data.N1, tiling_data.N2, &pipe);
        op.Process();
    } 
}

