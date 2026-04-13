#include "kernel_operator.h"
#include <type_traits>
#include <limits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;  // 双缓冲支持流水线
constexpr uint32_t UB_SIZE_BYTES = 196352;  
constexpr uint32_t ALIGN_BYTES = 32; 
// Reduce 类型枚举
enum ReduceType {
    REDUCE_SUM = 0,
    REDUCE_MEAN = 1,
    REDUCE_MAX = 2,
    REDUCE_MIN = 3,
    REDUCE_PROD = 4
};

template<typename T>
class KernelSegmentReduceVector {
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueData;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOutput;
    TQue<QuePosition::VECCALC, BUFFER_NUM> calcQueue;
    TQue<QuePosition::VECCALC, BUFFER_NUM> castFloatInQueue;
    TQue<QuePosition::VECCALC, BUFFER_NUM> castFloatOutQueue;
    GlobalTensor<T> dataGm;
    GlobalTensor<T> outputGm;
    GlobalTensor<int64_t> lengthsGm; // 仅用于设置地址
    GlobalTensor<T> initialGm;

    T initial_val;
    uint32_t n1_dim;
    uint32_t n2_dim;

    // 核心数组，存储每个段的长度，最多20个
    uint32_t K[20];
    uint32_t dim_K;

    // 多核并行控制变量（两级并行）
    uint32_t assigned_seg_idx;  // 当前核分配的 segment 索引
    uint32_t cores_per_seg;     // 每个 segment 分配的核数
    uint32_t col_start_offset;  // 当前核负责的列起始偏移（在segment内）
    uint32_t col_end_offset;    // 当前核负责的列结束偏移（在segment内，不包含）
    bool has_task;              // 当前核是否有任务

    uint32_t tile_length;
    uint32_t reduce_type;
    uint32_t has_initial;
    uint32_t use_offsets;
    int BlockIdx, BlockNum;
    int max_tile_length;
    int aligned_buffer_size_rest;
    uint64_t start_row_offset;
public:
    __aicore__ inline KernelSegmentReduceVector() {}

    __aicore__ inline void Init(GM_ADDR data, GM_ADDR lengths, GM_ADDR output, GM_ADDR initial,
        SegmentReduceTilingData& tiling_data) {
        this->n1_dim = tiling_data.N;
        this->n2_dim = tiling_data.N2;
        this->dim_K = tiling_data.dim_K;
        this->use_offsets = tiling_data.useOffsets;
        this->reduce_type = tiling_data.reduceType;
        this->has_initial = tiling_data.hasInitial;
        
        // 1. 获取当前核的信息
        this->BlockIdx = GetBlockIdx();
        this->BlockNum = GetBlockNum();


        // 设置 GM 指针
        lengthsGm.SetGlobalBuffer((__gm__ int64_t *)lengths);
        dataGm.SetGlobalBuffer((__gm__ T*)data);
        outputGm.SetGlobalBuffer((__gm__ T*)output);

        if (this->use_offsets == 1) {
            // 输入是 Offsets (int64)，转换为 Lengths 存入 K
            __gm__ long* offsets_ptr = (__gm__ long*)lengths;
            // this->start_row_offset = offsets_ptr[0];
            for (uint32_t i = 0; i < this->dim_K; i++) {
                int64_t offset_start = offsets_ptr[i];
                int64_t offset_end = offsets_ptr[i + 1];
                this->K[i] = static_cast<uint32_t>(offset_end - offset_start);
            }
        }
        else {

            __gm__ int64_t* lengths_ptr = (__gm__ int64_t*)lengths;
            // this->start_row_offset = 0;
            for (uint32_t i = 0; i < this->dim_K; i++) {
                this->K[i] = static_cast<uint64_t>(lengths_ptr[i]);
            }
        }


        // 两级并行策略：
        // 1. 第一级：按segment数量分配核组
        // 2. 第二级：每个segment内部按列切分给核组内的核
        
        // 计算每个segment分配的核数
        this->cores_per_seg = this->BlockNum / this->dim_K;
        uint32_t rem_cores = this->BlockNum % this->dim_K;
        
        // 分配segment到当前核
        // 核按顺序分配给segment：segment 0 -> 核0-cores_per_seg-1, segment 1 -> 核cores_per_seg-cores_per_seg*2-1, ...
        uint32_t seg_base_core = 0;  // segment的起始核索引
        this->assigned_seg_idx = 0;
        this->has_task = false;
        
        for (uint32_t seg = 0; seg < this->dim_K; seg++) {
            uint32_t seg_core_count = this->cores_per_seg;
            if (seg < rem_cores) {
                seg_core_count += 1;  // 前rem_cores个segment多分配一个核
            }
            
            if (this->BlockIdx >= seg_base_core && this->BlockIdx < seg_base_core + seg_core_count) {
                // 当前核属于这个segment
                this->assigned_seg_idx = seg;
                this->has_task = true;
                
                // 计算当前核在segment内的列切分
                uint32_t core_in_seg = this->BlockIdx - seg_base_core;
                uint32_t cols_per_core = this->n2_dim / seg_core_count;
                uint32_t rem_cols = this->n2_dim % seg_core_count;
                
                // 前rem_cols个核多分配一列
                if (core_in_seg < rem_cols) {
                    this->col_start_offset = core_in_seg * (cols_per_core + 1);
                    this->col_end_offset = this->col_start_offset + cols_per_core + 1;
                } else {
                    this->col_start_offset = rem_cols * (cols_per_core + 1) + (core_in_seg - rem_cols) * cols_per_core;
                    this->col_end_offset = this->col_start_offset + cols_per_core;
                }
                break;
            }
            
            seg_base_core += seg_core_count;
        }
        
        // 如果当前核没有分配到segment，标记为无任务
        if (!this->has_task) {
            this->assigned_seg_idx = 0;
            this->col_start_offset = 0;
            this->col_end_offset = 0;
        }
        
        // 计算当前segment的行偏移（用于读取GM）
        this->start_row_offset = 0;
        for (uint32_t k = 0; k < this->assigned_seg_idx; k++) {
            this->start_row_offset += (uint64_t)this->K[k];
        }


        uint32_t Max_Tile_Length;
        if constexpr (std::is_same_v<T, float>) {
            Max_Tile_Length = 8192;
            this->max_tile_length = 8192;
        }
        else if constexpr (std::is_same_v<T, half>) {
            Max_Tile_Length = 6144;
            this->max_tile_length = 6144;
        }
        this->tile_length = (this->n2_dim < Max_Tile_Length) ? this->n2_dim : Max_Tile_Length;
        if (this->tile_length == 0) this->tile_length = Max_Tile_Length;
        uint32_t aligned_num = 32/sizeof(T);
        uint32_t aligned_tile_length = (this->tile_length+aligned_num-1) / aligned_num * aligned_num;
        if constexpr (std::is_same_v<T, half>) {
            this->aligned_buffer_size_rest = (65525 - aligned_tile_length*sizeof(float) * 4 + 31) / 32 * 32;
            pipe.InitBuffer(inQueueData, BUFFER_NUM, this->aligned_buffer_size_rest);
            pipe.InitBuffer(outQueueOutput, BUFFER_NUM, aligned_tile_length*sizeof(T));
            pipe.InitBuffer(calcQueue, BUFFER_NUM, aligned_tile_length*sizeof(float));
            pipe.InitBuffer(castFloatInQueue, BUFFER_NUM, this->aligned_buffer_size_rest*2);
            pipe.InitBuffer(castFloatOutQueue, BUFFER_NUM, aligned_tile_length*sizeof(float));
        }
        else {
        this->aligned_buffer_size_rest = (98288 - aligned_tile_length*sizeof(T) * 2 + 31) / 32 * 32;
        // printf("aligned_buffer_size_rest: %d\n", this->aligned_buffer_size_rest);
        // printf("aligned_tile_length: %d\n", aligned_tile_length);
        pipe.InitBuffer(inQueueData, BUFFER_NUM, this->aligned_buffer_size_rest);
        pipe.InitBuffer(outQueueOutput, BUFFER_NUM, aligned_tile_length*sizeof(T));
        pipe.InitBuffer(calcQueue, BUFFER_NUM, aligned_tile_length*sizeof(T));
        }
        if (this->has_initial == 1 && initial != nullptr) {
            initialGm.SetGlobalBuffer((__gm__ T*)initial);
            LocalTensor<T> initialLocal = inQueueData.AllocTensor<T>();
            DataCopyExtParams copyParamsInitial = { 1, (uint32_t)(1 * sizeof(T)), 0, 0, 0 };
            DataCopyPadExtParams<T> padParams{ false, 0, 0, 0 };
            DataCopyPad(initialLocal, initialGm[0], copyParamsInitial, padParams);
            inQueueData.EnQue(initialLocal);
            initialLocal = inQueueData.DeQue<T>();
            this->initial_val = initialLocal.GetValue(0);
            inQueueData.FreeTensor(initialLocal);
        }
        else {
            this->initial_val = (T)0;
        }
    }

    __aicore__ inline void Process() {
        // 如果当前核没有分配到任务，直接退出
        if (!this->has_task) return;

        // 计算当前核的数据起始全局偏移
        uint64_t global_data_offset = this->start_row_offset * this->n2_dim + this->col_start_offset;

        // 从本地 K 数组获取当前segment的长度
        int32_t current_seg_len = static_cast<int32_t>(this->K[this->assigned_seg_idx]);
        
        // 处理当前segment的列切分部分
        uint32_t col_range = this->col_end_offset - this->col_start_offset;
        ProcessSegmentColumns(this->assigned_seg_idx, current_seg_len, global_data_offset, 
                              this->col_start_offset, col_range);
    }

    __aicore__ inline void ProcessSegmentColumns(uint32_t seg_idx, int32_t seg_len, uint64_t start_offset,
                                                  uint32_t col_start, uint32_t col_range) {
        if (seg_len < 0 || col_range == 0) return;
        // seg_idx: segment索引，用于计算输出位置
        // seg_len: segment的行数
        // start_offset: 全局数据偏移（已包含列偏移）
        // col_start: 在当前segment内的列起始偏移
        // col_range: 当前核处理的列数

        // 按列切分处理（只处理分配给当前核的列范围）
        for (uint32_t offset = 0; offset < col_range; offset += this->tile_length) {
            uint32_t current_tile_length = this->tile_length;
            if (offset + this->tile_length > col_range) {
                current_tile_length = col_range - offset;
            }
            
            if constexpr (std::is_same_v<T, half>) {
                LocalTensor<float> accLocal = castFloatOutQueue.AllocTensor<float>();
                float init_val;
                if (this->has_initial == 1) {
                    init_val = (float)this->initial_val;
                }
                else {
                    if (this->reduce_type == REDUCE_MAX) {
                        init_val = -65504.0f;
                    }
                    else if (this->reduce_type == REDUCE_MIN) {
                        init_val = 65504.0f;
                    }
                    else if (this->reduce_type == REDUCE_PROD) {
                        init_val = 1.0f;
                    }
                    else {
                        init_val = 0.0f;
                    }
                }
                Duplicate(accLocal, init_val, current_tile_length);
                const uint32_t ALIGN_ELEMENTS = 32 / sizeof(T);  // float: 8, half: 16
                uint32_t aligned_tile_length = (current_tile_length+ALIGN_ELEMENTS-1) / ALIGN_ELEMENTS * ALIGN_ELEMENTS;
                uint32_t srcStride = (this->n2_dim - current_tile_length) * sizeof(T);
                uint8_t pad_length = aligned_tile_length - current_tile_length;
                DataCopyExtParams copyParamsInput = { 1, (uint32_t)(current_tile_length * sizeof(T)), srcStride, 0, 0 };
                DataCopyExtParams copyParamOutput = { 1, (uint32_t)(current_tile_length * sizeof(T)), 0, 0, 0 };
                DataCopyPadExtParams<T> padParams{ true, 0, pad_length, 0 };
                int row_offset = 0;
                int32_t next_row = 0;
                int32_t next_row_offset = 0;
                constexpr bool isReuse = true;
                
                // 流水线：预加载第一个 batch
                if (seg_len > 0) {
                    next_row_offset = this->aligned_buffer_size_rest/sizeof(T) / aligned_tile_length;
                    if(next_row_offset > seg_len) next_row_offset = seg_len;
                    copyParamsInput.blockCount = next_row_offset;
                    LocalTensor<T> nextDataLocal = inQueueData.AllocTensor<T>();
                    uint64_t next_gm_offset = start_offset + (uint64_t)next_row * this->n2_dim + offset;
                    DataCopyPad(nextDataLocal, dataGm[next_gm_offset], copyParamsInput, padParams);
                    inQueueData.EnQue(nextDataLocal);
                }
                
                for (int32_t row = 0; row < seg_len; row += row_offset) {
                    row_offset = next_row_offset;
                    // 准备下一次迭代的参数（如果还有数据）
                    next_row = row + row_offset;
                    if (next_row < seg_len) {
                        next_row_offset = this->aligned_buffer_size_rest/sizeof(T) / aligned_tile_length;
                        if(next_row + next_row_offset > seg_len) next_row_offset = seg_len - next_row;
                        copyParamsInput.blockCount = next_row_offset;
                        LocalTensor<T> nextDataLocal = inQueueData.AllocTensor<T>();
                        uint64_t next_gm_offset = start_offset + (uint64_t)next_row * this->n2_dim + offset;
                        // 流水线：在计算当前 batch 的同时启动下一次的 copyin
                        DataCopyPad(nextDataLocal, dataGm[next_gm_offset], copyParamsInput, padParams);
                        inQueueData.EnQue(nextDataLocal);
                    }
                    
                    // 获取当前 batch 的数据（已经在队列中）
                    LocalTensor<T> dataLocal = inQueueData.DeQue<T>();
                    
                    uint32_t srcShape[] = { static_cast<uint32_t>(row_offset), current_tile_length };
                    LocalTensor<float> batchResult = calcQueue.AllocTensor<float>();
                    LocalTensor<float> dataLocal_float = castFloatInQueue.AllocTensor<float>();
                    uint32_t cast_len = (uint32_t)row_offset * aligned_tile_length;
                    Cast(dataLocal_float, dataLocal, RoundMode::CAST_NONE, cast_len);
                    castFloatInQueue.EnQue(dataLocal_float);
                    dataLocal_float = castFloatInQueue.DeQue<float>();
                    if(this->reduce_type == REDUCE_SUM||this->reduce_type == REDUCE_MEAN) {
                        ReduceSum<float, Pattern::Reduce::RA, isReuse>(batchResult, dataLocal_float, srcShape, true);
                        Add(accLocal, accLocal, batchResult, current_tile_length);
                    }
                    else if(this->reduce_type == REDUCE_MAX) {
                        ReduceMax<float, Pattern::Reduce::RA, isReuse>(batchResult, dataLocal_float, srcShape, true);
                        Max(accLocal, accLocal, batchResult, current_tile_length);
                    }
                    else if(this->reduce_type == REDUCE_MIN) {
                        ReduceMin<float, Pattern::Reduce::RA, isReuse>(batchResult, dataLocal_float, srcShape, true);
                        Min(accLocal, accLocal, batchResult, current_tile_length);
                    }
                    castFloatInQueue.FreeTensor(dataLocal_float);
                    calcQueue.FreeTensor(batchResult);
                    inQueueData.FreeTensor(dataLocal);
                }

                PostProcess(accLocal, current_tile_length, seg_len);
                LocalTensor<T> outLocal = outQueueOutput.AllocTensor<T>();
                Cast(outLocal, accLocal, RoundMode::CAST_ROUND, current_tile_length);
                uint64_t out_gm_offset = (uint64_t)seg_idx * this->n2_dim + col_start + offset;
                DataCopyPad(outputGm[out_gm_offset], outLocal, copyParamOutput);
                outQueueOutput.FreeTensor(outLocal);
            } else {
                LocalTensor<float> accLocal = outQueueOutput.AllocTensor<float>();
                float init_val;
                if (this->has_initial == 1) {
                    init_val = (float)this->initial_val;
                }
                else {
                    if (this->reduce_type == REDUCE_MAX) {
                        init_val = -3.402823466e+38F;
                    }
                    else if (this->reduce_type == REDUCE_MIN) {
                        init_val = 3.402823466e+38F;
                    }
                    else if (this->reduce_type == REDUCE_PROD) {
                        init_val = 1.0f;
                    }
                    else {
                        init_val = 0.0f;
                    }
                }
                Duplicate(accLocal, init_val, current_tile_length);
                const uint32_t ALIGN_ELEMENTS = 32 / sizeof(T);  // float: 8, half: 16
                uint32_t aligned_tile_length = (current_tile_length+ALIGN_ELEMENTS-1) / ALIGN_ELEMENTS * ALIGN_ELEMENTS;
                uint32_t srcStride = (this->n2_dim - current_tile_length) * sizeof(T);
                uint8_t pad_length = aligned_tile_length - current_tile_length;
                DataCopyExtParams copyParamsInput = { 1, (uint32_t)(current_tile_length * sizeof(T)), srcStride, 0, 0 };
                DataCopyExtParams copyParamOutput = { 1, (uint32_t)(current_tile_length * sizeof(T)), 0, 0, 0 };
                DataCopyPadExtParams<T> padParams{ true, 0, pad_length, 0 };
                int row_offset = 0;
                int32_t next_row = 0;
                int32_t next_row_offset = 0;
                constexpr bool isReuse = true;
                
                // 流水线：预加载第一个 batch
                if (seg_len > 0) {
                    next_row_offset = this->aligned_buffer_size_rest/sizeof(T) / aligned_tile_length;
                    if(next_row_offset > seg_len) next_row_offset = seg_len;
                    copyParamsInput.blockCount = next_row_offset;
                    LocalTensor<T> nextDataLocal = inQueueData.AllocTensor<T>();
                    uint64_t next_gm_offset = start_offset + (uint64_t)next_row * this->n2_dim + offset;
                    DataCopyPad(nextDataLocal, dataGm[next_gm_offset], copyParamsInput, padParams);
                    inQueueData.EnQue(nextDataLocal);
                }
                
                for (int32_t row = 0; row < seg_len; row += row_offset) {
                    row_offset = next_row_offset;
                    // 准备下一次迭代的参数（如果还有数据）
                    next_row = row + row_offset;
                    if (next_row < seg_len) {
                        next_row_offset = this->aligned_buffer_size_rest/sizeof(T) / aligned_tile_length;
                        if(next_row + next_row_offset > seg_len) next_row_offset = seg_len - next_row;
                        copyParamsInput.blockCount = next_row_offset;
                        LocalTensor<T> nextDataLocal = inQueueData.AllocTensor<T>();
                        uint64_t next_gm_offset = start_offset + (uint64_t)next_row * this->n2_dim + offset;
                        // 流水线：在计算当前 batch 的同时启动下一次的 copyin
                        DataCopyPad(nextDataLocal, dataGm[next_gm_offset], copyParamsInput, padParams);
                        inQueueData.EnQue(nextDataLocal);
                    }
                    
                    // 获取当前 batch 的数据（已经在队列中）
                    LocalTensor<T> dataLocal = inQueueData.DeQue<T>();
                    
                    LocalTensor<float> batchResult = calcQueue.AllocTensor<float>();
                    uint32_t srcShape[] = { static_cast<uint32_t>(row_offset), current_tile_length };
                    if(this->reduce_type == REDUCE_SUM||this->reduce_type == REDUCE_MEAN) {
                        ReduceSum<float, Pattern::Reduce::RA, isReuse>(batchResult, dataLocal, srcShape, true);
                        Add(accLocal, accLocal, batchResult, current_tile_length);
                    }
                    else if(this->reduce_type == REDUCE_MAX) {
                        ReduceMax<float, Pattern::Reduce::RA, isReuse>(batchResult, dataLocal, srcShape, true);
                        Max(accLocal, accLocal, batchResult, current_tile_length);
                    }
                    else if(this->reduce_type == REDUCE_MIN) {
                        ReduceMin<float, Pattern::Reduce::RA, isReuse>(batchResult, dataLocal, srcShape, true);
                        Min(accLocal, accLocal, batchResult, current_tile_length);
                    }
                    calcQueue.FreeTensor(batchResult);
                    inQueueData.FreeTensor(dataLocal);
                }

                PostProcess(accLocal, current_tile_length, seg_len);
                outQueueOutput.EnQue(accLocal);
                LocalTensor<T> outLocal = outQueueOutput.DeQue<T>();
                uint64_t out_gm_offset = (uint64_t)seg_idx * this->n2_dim + col_start + offset;
                DataCopyPad(outputGm[out_gm_offset], outLocal, copyParamOutput);
                outQueueOutput.FreeTensor(outLocal);
            }
        }
    }

    __aicore__ inline void ComputeStep(LocalTensor<T>& acc, LocalTensor<T>& input, uint32_t len) {
        switch (this->reduce_type) {
        case REDUCE_SUM:
        case REDUCE_MEAN: Add(acc, acc, input, len); break;
        case REDUCE_MAX: Max(acc, acc, input, len); break;
        case REDUCE_MIN: Min(acc, acc, input, len); break;
        case REDUCE_PROD: Mul(acc, acc, input, len); break;
        default: break;
        }
    }

    __aicore__ inline void PostProcess(LocalTensor<float>& acc, uint32_t len, int32_t seg_len) {
        if (this->reduce_type == REDUCE_MEAN && seg_len > 0) {
            float scale = (float)(1.0f / seg_len);
            Muls(acc, acc, scale, len);
        }
    }
};

template<typename T>
class KernelSegmentReduceOneDim {
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueData;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOutput;

    GlobalTensor<T> dataGm;
    GlobalTensor<T> outputGm;
    GlobalTensor<int64_t> lengthsGm; // 仅用于设置地址
    GlobalTensor<T> initialGm;

    T initial_val;
    uint32_t n1_dim;
    uint32_t n2_dim;

    // 核心数组，存储每个段的长度，最多20个
    uint32_t K[20];
    uint32_t dim_K;

    // 多核并行控制变量
    uint32_t start_seg_idx;    // 当前核负责的起始 segment 索引
    uint32_t task_num;         // 当前核负责处理的 segment 数量
    uint64_t start_row_offset; // 当前核负责的数据在 dataGm 中的起始行偏移

    uint32_t tile_length;
    uint32_t reduce_type;
    uint32_t has_initial;
    uint32_t use_offsets;
    int BlockIdx, BlockNum;
    int max_tile_length;
    int aligned_buffer_size_rest;
public:
    __aicore__ inline KernelSegmentReduceOneDim() {}

    __aicore__ inline void Init(GM_ADDR data, GM_ADDR lengths, GM_ADDR output, GM_ADDR initial,
        SegmentReduceTilingData& tiling_data) {
        this->n1_dim = tiling_data.N;
        this->n2_dim = tiling_data.N2;
        this->dim_K = tiling_data.dim_K;
        this->use_offsets = tiling_data.useOffsets;
        this->reduce_type = tiling_data.reduceType;
        this->has_initial = tiling_data.hasInitial;

        // 1. 获取当前核的信息
        this->BlockIdx = GetBlockIdx();
        this->BlockNum = GetBlockNum();

        if (this->dim_K == 0 || this->dim_K > 20) return;

        // 设置 GM 指针
        lengthsGm.SetGlobalBuffer((__gm__ int64_t *)lengths);
        dataGm.SetGlobalBuffer((__gm__ T*)data);
        outputGm.SetGlobalBuffer((__gm__ T*)output);

        if (this->use_offsets == 1) {
            // 输入是 Offsets (int64)，转换为 Lengths 存入 K
            __gm__ long* offsets_ptr = (__gm__ long*)lengths;
            for (uint32_t i = 0; i < this->dim_K; i++) {
                int64_t offset_start = offsets_ptr[i];
                int64_t offset_end = offsets_ptr[i + 1];
                this->K[i] = static_cast<uint32_t>(offset_end - offset_start);
            }
        }
        else {

            __gm__ int64_t* lengths_ptr = (__gm__ int64_t*)lengths;
            for (uint32_t i = 0; i < this->dim_K; i++) {
                this->K[i] = static_cast<uint64_t>(lengths_ptr[i]);
            }
        }


        // 计算每个核处理多少个 Segment
        uint32_t avg_tasks = this->dim_K / this->BlockNum;
        uint32_t rem_tasks = this->dim_K % this->BlockNum;

        if (this->BlockIdx < rem_tasks) {
            this->task_num = avg_tasks + 1;
            this->start_seg_idx = this->BlockIdx * this->task_num;
        }
        else {
            this->task_num = avg_tasks;
            this->start_seg_idx = rem_tasks * (avg_tasks + 1) + (this->BlockIdx - rem_tasks) * this->task_num;
        }

        this->start_row_offset = 0;
        for (uint32_t k = 0; k < this->start_seg_idx; k++) {
            this->start_row_offset += (uint64_t)this->K[k];
        }


        uint32_t Max_Tile_Length;
        if constexpr (std::is_same_v<T, float>) {
            Max_Tile_Length = 12288;
            this->max_tile_length = 12288;
        }
        else if constexpr (std::is_same_v<T, half>) {
            Max_Tile_Length = 24576;
            this->max_tile_length = 24576;
        }
        this->tile_length = (this->n2_dim < Max_Tile_Length) ? this->n2_dim : Max_Tile_Length;
        if (this->tile_length == 0) this->tile_length = Max_Tile_Length;

        uint32_t buffer_size_bytes = this->tile_length * sizeof(T);
        uint32_t aligned_buffer_size = (buffer_size_bytes + 31) / 32 * 32;
        this->aligned_buffer_size_rest = (61440 - aligned_buffer_size + 31) / 32 * 32;
        pipe.InitBuffer(inQueueData, BUFFER_NUM, this->aligned_buffer_size_rest);
        pipe.InitBuffer(outQueueOutput, BUFFER_NUM, aligned_buffer_size);

        if (this->has_initial == 1 && initial != nullptr) {
            initialGm.SetGlobalBuffer((__gm__ T*)initial);
            LocalTensor<T> initialLocal = inQueueData.AllocTensor<T>();
            DataCopyExtParams copyParamsInitial = { 1, (uint32_t)(1 * sizeof(T)), 0, 0, 0 };
            DataCopyPadExtParams<T> padParams{ false, 0, 0, 0 };
            DataCopyPad(initialLocal, initialGm[0], copyParamsInitial, padParams);
            inQueueData.EnQue(initialLocal);
            initialLocal = inQueueData.DeQue<T>();
            this->initial_val = initialLocal.GetValue(0);
            inQueueData.FreeTensor(initialLocal);
        }
        else {
            this->initial_val = (T)0;
        }
    }

    __aicore__ inline void Process() {
        // 如果当前核没有分配到任务（核数 > dim_K），直接退出
        if (this->task_num == 0) return;

        // 计算当前核的数据起始全局偏移
        uint64_t global_data_offset = this->start_row_offset * this->n2_dim;

        // 每个核只处理一个 segment
        // 从本地 K 数组获取长度
        int32_t current_seg_len = static_cast<int32_t>(this->K[this->start_seg_idx]);
        // 处理单个 Segment
        ProcessOneSegment(this->start_seg_idx, current_seg_len, global_data_offset);
    }

    __aicore__ inline void ProcessOneSegment(uint32_t seg_idx, int32_t seg_len, uint64_t start_offset) {
        if (seg_len < 0) return;
        // seg_idx: 全局索引，用于计算输出位置 outputGm[seg_idx * n2]
        // start_offset: 全局数据偏移，用于读取 dataGm[start_offset + row*n2]
        // 只处理一维情况：n2_dim == 1

        // 分配输入和输出 tensor
        LocalTensor<T> dataLocal = inQueueData.AllocTensor<T>();
        LocalTensor<T> accLocal = outQueueOutput.AllocTensor<T>();
        
        // 初始化累加器
        T init_val;
        if (this->has_initial == 1) {
            init_val = this->initial_val;
        }
        else {
            if (this->reduce_type == REDUCE_MAX) {
                if constexpr (std::is_same_v<T, float>) init_val = -3.402823466e+38F;
                else init_val = -65504.0f;
            }
            else if (this->reduce_type == REDUCE_MIN) {
                if constexpr (std::is_same_v<T, float>) init_val = 3.402823466e+38F;
                else init_val = 65504.0f;
            }
            else {
                init_val = (T)0;  // SUM 或 MEAN
            }
        }
        accLocal.SetValue(0, init_val);
        
        // 准备 Copy 参数
        DataCopyExtParams copyParamsInput = { 1, (uint32_t)(seg_len * sizeof(T)), 0, 0, 0 };
        DataCopyPadExtParams<T> padParams{ false, 0, 0, 0 };
        
        // 从 GM 加载整个 segment 的数据
        DataCopyPad(dataLocal, dataGm[start_offset], copyParamsInput, padParams);
        inQueueData.EnQue(dataLocal);
        dataLocal = inQueueData.DeQue<T>();
        
        // 调用 ComputeStep 处理对齐部分
        ComputeStep(accLocal, dataLocal, seg_len);
        
        // 后处理（MEAN）
        if (this->reduce_type == REDUCE_MEAN && seg_len > 0) {
            float result = accLocal.GetValue(0);
            result = result / (float)seg_len;
            accLocal.SetValue(0, (T)result);
        }
        
        // 输出结果
        outQueueOutput.EnQue(accLocal);
        LocalTensor<T> outLocal = outQueueOutput.DeQue<T>();
        uint64_t out_gm_offset = (uint64_t)seg_idx * this->n2_dim;
        DataCopyExtParams copyParamsOutput = { 1, (uint32_t)(1 * sizeof(T)), 0, 0, 0 };
        DataCopyPad(outputGm[out_gm_offset], outLocal, copyParamsOutput);
        
        outQueueOutput.FreeTensor(outLocal);
        inQueueData.FreeTensor(dataLocal);
    }

    __aicore__ inline void ComputeStep(LocalTensor<T>& acc, LocalTensor<T>& input, uint32_t len) {
        switch (this->reduce_type) {
        case REDUCE_SUM:
        case REDUCE_MEAN:
            ReduceSum<T>(acc, input, acc, len);
            break;
        case REDUCE_MAX:
            ReduceMax<T>(acc, input, acc, len);
            break;
        case REDUCE_MIN:
            ReduceMin<T>(acc, input, acc, len);
            break;
        case REDUCE_PROD:
            // uint32_t shape[] = { 1, len };
            // ReduceProd<T>(acc, input, acc, shape, true);
            break;
        default:
            break;
        }
    }

    __aicore__ inline void PostProcess(LocalTensor<T>& acc, uint32_t len, int32_t seg_len) {
        if (this->reduce_type == REDUCE_MEAN && seg_len > 0) {
            T scale = (T)(1.0f / seg_len);
            Muls(acc, acc, scale, len);
        }
    }
};

template<typename T>
class KernelSegmentReduceBF16 {
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueData;
    TQue<QuePosition::VECCALC, BUFFER_NUM> FloatInQueueData;  // 用于 BF16->FP32 的中间结果和累加器
    TQue<QuePosition::VECCALC, 1> accBf16Queue;              // 单独 buffer：累加结果 fp32->bf16 暂存，不复用 inQueueData
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOutput;

    GlobalTensor<T> dataGm;
    GlobalTensor<T> outputGm;
    GlobalTensor<int64_t> lengthsGm;  // lengths 或 offsets 的 GM tensor
    GlobalTensor<T> initialGm;
    float initial_val;  // 存储 initial 值（转换为 float，因为 bfloat16 使用 float 计算）
    uint32_t has_initial;
    uint32_t n1_dim;
    uint32_t n2_dim;
    uint32_t K[20];      // K 数组，存储每个 segment 的长度
    uint32_t dim_K;      // K 数组的实际维度（从 tiling_data 获取）
    uint32_t max_length;
    // 多核并行控制变量（与 KernelSegmentReduceVector 保持一致语义）
    uint32_t start_seg_idx;    // 当前核负责的起始 segment 索引
    uint32_t task_num;         // 当前核负责处理的 segment 数量
    uint64_t start_row_offset; // 当前核负责的数据在 dataGm 中的起始行偏移（按行计）
    uint32_t tile_length;
    uint32_t reduce_type;
    uint32_t use_offsets;  // 0: lengths, 1: offsets
    int BlockIdx, BlockNum;
    T initial_val_t; // 新增：用于存储 BF16 类型的初始值
    uint32_t aligned_buffer_size_rest;  // 用于批量处理的剩余 buffer 大小
public:
    __aicore__ inline KernelSegmentReduceBF16() {}

    __aicore__ inline void Init(GM_ADDR data, GM_ADDR lengths, GM_ADDR output, GM_ADDR initial,
        SegmentReduceTilingData& tiling_data) {
        this->n1_dim = tiling_data.N;
        this->n2_dim = tiling_data.N2;
        this->dim_K = tiling_data.dim_K;
        this->use_offsets = tiling_data.useOffsets;
        this->reduce_type = tiling_data.reduceType;
        this->has_initial = tiling_data.hasInitial;
        this->max_length = 0;  // 初始化 max_length

        // 1. 获取当前核的信息
        this->BlockIdx = GetBlockIdx();
        this->BlockNum = GetBlockNum();
        // 验证 dim_K 是否有效
        if (this->dim_K == 0 || this->dim_K > 20) {
            // printf("[ERROR] Invalid dim_K in kernel: %d\n", this->dim_K);
            return;  // 无法继续执行
        }

        // 设置 lengths/offsets 的 GM tensor 和数据/输出 GM
        lengthsGm.SetGlobalBuffer((__gm__ int64_t*)lengths);
        dataGm.SetGlobalBuffer((__gm__ T*)data);
        outputGm.SetGlobalBuffer((__gm__ T*)output);

        if (this->use_offsets == 1) {
            // offsets 是 int64 (N+1,)，转换为 lengths 存入 K
            __gm__ int64_t* offsets_ptr = (__gm__ int64_t*)lengths;
            for (uint32_t i = 0; i < this->dim_K; i++) {
                int64_t offset_start = offsets_ptr[i];
                int64_t offset_end = offsets_ptr[i + 1];
                this->K[i] = static_cast<uint32_t>(offset_end - offset_start);
                this->max_length = (this->K[i] > this->max_length) ? this->K[i] : this->max_length;
            }
        }
        else {
            // lengths 统一按 int64 读取并转为 uint32 存入 K
            __gm__ int64_t* lengths_ptr = (__gm__ int64_t*)lengths;
            for (uint32_t i = 0; i < this->dim_K; i++) {
                this->K[i] = static_cast<uint32_t>(lengths_ptr[i]);
                this->max_length = (this->K[i] > this->max_length) ? this->K[i] : this->max_length;
            }
        }

        uint32_t avg_tasks = this->dim_K / this->BlockNum;
        uint32_t rem_tasks = this->dim_K % this->BlockNum;

        if (this->BlockIdx < rem_tasks) {
            this->task_num = avg_tasks + 1;
            this->start_seg_idx = this->BlockIdx * this->task_num;
        }
        else {
            this->task_num = avg_tasks;
            this->start_seg_idx = rem_tasks * (avg_tasks + 1) + (this->BlockIdx - rem_tasks) * this->task_num;
        }

        if (this->use_offsets == 1) {
            // 对于 offsets，直接使用 offsets[start_seg_idx] 作为起始偏移
            __gm__ long* offsets_ptr = (__gm__ long*)lengths;
            this->start_row_offset = offsets_ptr[this->start_seg_idx];
        } else {
            // 对于 lengths，累加前面的 lengths
            this->start_row_offset = 0;
            for (uint32_t k = 0; k < this->start_seg_idx; k++) {
                this->start_row_offset += (uint64_t)this->K[k];
            }
        }


        uint32_t Max_Tile_Length = 12288;

        this->tile_length = (this->n2_dim < Max_Tile_Length) ? this->n2_dim : Max_Tile_Length;
        if (this->tile_length == 0) {
            this->tile_length = Max_Tile_Length;  // 至少保证对齐大小
        }


        uint32_t buffer_size_bytes_t = this->tile_length * sizeof(T);
        uint32_t aligned_buffer_size_t = (buffer_size_bytes_t + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        uint32_t buffer_size_bytes_f = this->tile_length * sizeof(float);
        uint32_t aligned_buffer_size_f = (buffer_size_bytes_f + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;

        // 计算用于批量处理的剩余 buffer 大小（类似 KernelSegmentReduceVector）
        this->aligned_buffer_size_rest = (196500 - aligned_buffer_size_t*2-aligned_buffer_size_f + 64) / 64 * 64;

        pipe.InitBuffer(inQueueData, 2, this->aligned_buffer_size_rest / 2);
        pipe.InitBuffer(accBf16Queue, 1, aligned_buffer_size_t);
        pipe.InitBuffer(outQueueOutput, 2, aligned_buffer_size_t);
        pipe.InitBuffer(FloatInQueueData, 2, aligned_buffer_size_f);

        // 如果提供了 initial，从 GM 读取并转换为 float 存储
        if (this->has_initial == 1 && initial != nullptr) {
            initialGm.SetGlobalBuffer((__gm__ T*)initial);
            // 从队列分配 LocalTensor 来读取 initial 值
            LocalTensor<T> initialLocal = inQueueData.AllocTensor<T>();
            DataCopyExtParams copyParamsInitial = { 1, (uint32_t)(1 * sizeof(T)), 0, 0, 0 };
            DataCopyPadExtParams<T> padParams{ false, 0, 0, 0 };
            DataCopyPad(initialLocal, initialGm[0], copyParamsInitial, padParams);
            // 转换为 float
            inQueueData.EnQue(initialLocal);
            initialLocal = inQueueData.DeQue<T>();
            LocalTensor<float> initialLocalFloat = FloatInQueueData.AllocTensor<float>();
            Cast(initialLocalFloat, initialLocal, RoundMode::CAST_NONE, 1);
            this->initial_val = initialLocalFloat.GetValue(0);
            FloatInQueueData.FreeTensor(initialLocalFloat);
            inQueueData.FreeTensor(initialLocal);
            // printf("initial_val=%f\n", this->initial_val);
        }
        else {
            this->initial_val = 0.0f;  // 默认值，但会被 reduce_type 覆盖
        }
    }

    __aicore__ inline void Process() {
        // 如果当前核没有分配到任务（核数 > dim_K），直接退出
        if (this->task_num == 0) return;

        // 计算当前核的数据起始全局偏移
        uint64_t global_data_offset = this->start_row_offset * this->n2_dim;

        // 每个核只处理一个 segment
        int32_t current_seg_len = static_cast<int32_t>(this->K[this->start_seg_idx]);
        // 处理单个 Segment
        ProcessOneSegment(this->start_seg_idx, current_seg_len, global_data_offset);
    }

private:
    __aicore__ inline void ProcessOneSegment(uint32_t seg_idx, int32_t seg_len, uint64_t start_offset) {
        if (seg_len < 0) return; // 简单保护

        // Inner Dimension Tiling
        for (uint32_t offset = 0; offset < this->n2_dim; offset += this->tile_length) {
            uint32_t current_tile_length = this->tile_length;
            if (offset + this->tile_length > this->n2_dim) {
                current_tile_length = this->n2_dim - offset;
            }

            // 1. 初始化累加器（使用 FloatInQueueData 作为 FP32 累加 Buffer）
            LocalTensor<float> accLocal = FloatInQueueData.AllocTensor<float>();

            // 初始化累加器值
            float init_val;
            if (this->has_initial == 1) {
                init_val = this->initial_val;
            }
            else {
                if (this->reduce_type == REDUCE_MAX) {
                    init_val = -65504.0f; // half min
                }
                else if (this->reduce_type == REDUCE_MIN) {
                    init_val = 65504.0f; // half max
                }
                else if (this->reduce_type == REDUCE_PROD) {
                    init_val = 1.0f;
                }
                else {
                    init_val = 0.0f;  // SUM, MEAN 默认值为 0
                }
            }
            Duplicate(accLocal, init_val, current_tile_length);

            // 计算对齐长度和批量处理参数
            const uint32_t ALIGN_ELEMENTS = 32 / sizeof(T);  // bf16: 16
            uint32_t aligned_tile_length = (current_tile_length + ALIGN_ELEMENTS - 1) / ALIGN_ELEMENTS * ALIGN_ELEMENTS;
            uint32_t srcStride = (this->n2_dim - current_tile_length) * sizeof(T);
            uint8_t pad_length = aligned_tile_length - current_tile_length;
            DataCopyExtParams copyParamsInput = { 1, (uint32_t)(current_tile_length * sizeof(T)), srcStride, 0, 0 };
            DataCopyExtParams copyParamOutput = { 1, (uint32_t)(current_tile_length * sizeof(T)), 0, 0, 0 };
            DataCopyPadExtParams<T> padParams{ false, 0, pad_length, 0 };

            // 2. 批量处理行进行 Reduce
            int row_offset = 0;
            for (int32_t row = 0; row < seg_len; row += row_offset) {
                row_offset = this->aligned_buffer_size_rest / sizeof(T) / aligned_tile_length;
                if (row + row_offset > seg_len) row_offset = seg_len - row;
                
                copyParamsInput.blockCount = row_offset;
                
                // 批量读取 bf16 数据
                LocalTensor<T> dataLocal = inQueueData.AllocTensor<T>();
                uint64_t cur_gm_offset = start_offset + (uint64_t)row * this->n2_dim + offset;
                DataCopyPad(dataLocal, dataGm[cur_gm_offset], copyParamsInput, padParams);
                inQueueData.EnQue(dataLocal);
                dataLocal = inQueueData.DeQue<T>();
                LocalTensor<T> dataLocal_row;
                LocalTensor<T> accBf16 = accBf16Queue.AllocTensor<T>();
                // 批量转换为 fp32 并逐行累加
                for (int row_in = 0; row_in < row_offset; row_in++) {
                    dataLocal_row = dataLocal[row_in * aligned_tile_length];
                    LocalTensor<float> dataLocalFloat = FloatInQueueData.AllocTensor<float>();
                    Cast(dataLocalFloat, dataLocal_row, RoundMode::CAST_NONE, current_tile_length);
                    // 累加：accLocal(fp32) + dataLocalFloat(fp32)
                    ComputeStep(accLocal, dataLocalFloat, current_tile_length);
                    FloatInQueueData.FreeTensor(dataLocalFloat);

                    // 累加后立刻量化回 BF16，再反量化回 FP32（accBf16 使用独立 accBf16Queue，不复用 inQueueData）
                    
                    Cast(accBf16, accLocal, RoundMode::CAST_ROUND, current_tile_length);
                    // LocalTensor<float> accFloatNew = FloatInQueueData.AllocTensor<float>();
                    Cast(accLocal, accBf16, RoundMode::CAST_NONE, current_tile_length);
                    // FloatInQueueData.FreeTensor(accFloatNew);
                    // accLocal = accFloatNew;
                    
                }
                accBf16Queue.FreeTensor(accBf16);
                // 释放 dataLocal（必须在所有使用完成后）
                inQueueData.FreeTensor(dataLocal);
            }
            
            // 3. 后处理 (Mean)
            PostProcess(accLocal, current_tile_length, seg_len);

            // 4. 输出
            LocalTensor<T> outLocal = outQueueOutput.AllocTensor<T>();
            Cast(outLocal, accLocal, RoundMode::CAST_ROUND, current_tile_length);
            outQueueOutput.EnQue(outLocal);
            outLocal = outQueueOutput.DeQue<T>();

            // 计算输出在 GM 中的偏移：seg_idx * n2_dim + offset
            uint64_t out_gm_offset = (uint64_t)seg_idx * this->n2_dim + offset;
            DataCopyPad(outputGm[out_gm_offset], outLocal, copyParamOutput);

            outQueueOutput.FreeTensor(outLocal);
            FloatInQueueData.FreeTensor(accLocal);
        }
    }


    __aicore__ inline void ComputeStep(LocalTensor<float>& acc, LocalTensor<float>& input, uint32_t len) {
        switch (this->reduce_type) {
        case REDUCE_SUM:
        case REDUCE_MEAN:
            Add(acc, acc, input, len);
            break;
        case REDUCE_MAX:
            Max(acc, acc, input, len);
            break;
        case REDUCE_MIN:
            Min(acc, acc, input, len);
            break;
        case REDUCE_PROD:
            Mul(acc, acc, input, len);
            break;
        default:
            break;
        }
    }

    __aicore__ inline void PostProcess(LocalTensor<float>& acc, uint32_t len, int32_t seg_len) {
        if (this->reduce_type == REDUCE_MEAN && seg_len > 0) {
            // 避免除以0，虽然前面应该已经过滤了
            LocalTensor<float> seg_len_local = FloatInQueueData.AllocTensor<float>();
            // 将 seg_len 明确转换为 float，确保与 LocalTensor<float> 类型一致
            Duplicate(seg_len_local, (float)seg_len, len);
            Div(acc, acc, seg_len_local, len);
            FloatInQueueData.FreeTensor(seg_len_local);
        }
    }


};


extern "C" __global__ __aicore__ void segment_reduce(GM_ADDR data, GM_ADDR lengths, GM_ADDR indices, GM_ADDR offsets, GM_ADDR initial, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    // 根据 useOffsets 决定使用 lengths 还是 offsets
    GM_ADDR lengths_or_offsets = (tiling_data.useOffsets == 1) ? offsets : lengths;

    // 根据 Tiling Key 选择数据类型
    // Host 端设置：0=float32, 1=float16, 2=bfloat16
    if (TILING_KEY_IS(0)) {  // float32
        KernelSegmentReduceVector<float> op;
        op.Init(data, lengths_or_offsets, output, initial, tiling_data);
        op.Process();
    }
    else if (TILING_KEY_IS(1)) {  // float16
        KernelSegmentReduceVector<half> op;
        op.Init(data, lengths_or_offsets, output, initial, tiling_data);
        op.Process();
    }
    else if (TILING_KEY_IS(2)) {  // bfloat16
        KernelSegmentReduceBF16<bfloat16_t> op;
        op.Init(data, lengths_or_offsets, output, initial, tiling_data);
        op.Process();
    }
    else if (TILING_KEY_IS(10)) {  // float32 one dim
        KernelSegmentReduceOneDim<float> op;
        op.Init(data, lengths_or_offsets, output, initial, tiling_data);
        op.Process();
    }
    else if (TILING_KEY_IS(11)) {  // float16 one dim
        KernelSegmentReduceOneDim<half> op;
        op.Init(data, lengths_or_offsets, output, initial, tiling_data);
        op.Process();
    }
}