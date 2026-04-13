#include "kernel_operator.h"
#include <type_traits>
#include <limits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t UB_SIZE_BYTES = 196352;  // 192KB UB 大小（910B）
constexpr uint32_t ALIGN_BYTES = 32;  // 32字节对齐

__aicore__ inline int RoundUp(int a, int b)
{ 
    return (a + b - 1) / b *b;
}

// Reduce 类型枚举
enum ReduceType {
    REDUCE_SUM = 0,
    REDUCE_MEAN = 1,
    REDUCE_MAX = 2,
    REDUCE_MIN = 3,
    REDUCE_PROD = 4
};

template<typename T>
class KernelSegmentReduceGrad {
private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGrad;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueData;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueGradOutput;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueFwd;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueMask;

    // BF16 计算用队列（仅在 T=bfloat16_t 时使用）
    TQue<QuePosition::VECCALC, BUFFER_NUM> calcQueue;         // SUM/MEAN BF16
    TQue<QuePosition::VECCALC, BUFFER_NUM> gradFloatQueue;    // MAX/MIN BF16
    TQue<QuePosition::VECCALC, BUFFER_NUM> fwdOutFloatQueue;  // MAX/MIN BF16
    TQue<QuePosition::VECCALC, BUFFER_NUM> dataFloatQueue;    // MAX/MIN BF16

    GlobalTensor<T> gradGm;        // 上游梯度
    GlobalTensor<T> dataGm;        // 前向输入
    GlobalTensor<T> outputGm;      // 输出梯度（对 data 的梯度）
    GlobalTensor<int32_t> lengthsGm; // lengths 或 offsets（int32）
    GlobalTensor<T> initialGm;
    GlobalTensor<T> fwdOutGm;  // 前向输出

    uint32_t n1_dim;  // N
    uint32_t n2_dim;  // N2
    uint32_t dim_K;   // K (segment 数量)
    int32_t axis;      // 归约轴

    // 核心数组，存储每个段的长度，最多20个
    uint32_t K[20];
    // 每个 segment 的前缀和：prefixK[i] = sum_{j < i} K[j]，长度 dim_K+1，最大 21
    uint64_t prefixK[21];

    // 任务调度：根据 dim_K 动态计算每个 segment 拆分的份数，使总任务数尽可能接近但不超过 40
    uint32_t shards_per_seg; // 每个 segment 拆分的份数
    uint32_t task_id;        // 当前 Block 负责的任务 id

    uint32_t tile_length;
    uint32_t max_tile_length;
    uint32_t row_size_bytes;   // host 预计算：N2 * sizeof(T)
    uint32_t row_compute_len;  // host 预计算：RoundUp(N2, 64)
    uint32_t reduce_type;
    uint32_t has_initial;
    uint32_t use_offsets;
    int BlockIdx, BlockNum;

public:
    __aicore__ inline KernelSegmentReduceGrad() {}

    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR output, GM_ADDR data, GM_ADDR lengths, GM_ADDR offsets, GM_ADDR grad_output, GM_ADDR initial,
        SegmentReduceGradTilingData& tiling_data, TPipe* pipeIn) {
        pipe = pipeIn;
        this->n1_dim = tiling_data.N;
        this->n2_dim = tiling_data.N2;
        this->dim_K = tiling_data.dim_K;
        this->axis = tiling_data.axis;
        this->use_offsets = tiling_data.useOffsets;
        this->reduce_type = tiling_data.reduceType;
        this->has_initial = tiling_data.hasInitial;
        this->row_size_bytes = tiling_data.rowSizeBytes;
        this->row_compute_len = tiling_data.rowComputeLen;
        // 1. 获取当前核的信息
        this->BlockIdx = GetBlockIdx();
        this->BlockNum = GetBlockNum();

        if (this->dim_K == 0 || this->dim_K > 20) return;

        // 设置 GM 指针
        gradGm.SetGlobalBuffer((__gm__ T*)grad);
        dataGm.SetGlobalBuffer((__gm__ T*)data);
        outputGm.SetGlobalBuffer((__gm__ T*)grad_output);
        fwdOutGm.SetGlobalBuffer((__gm__ T*)output);

        if (this->use_offsets == 1) {
            // 输入是 Offsets (int32)，转换为 Lengths 存入 K
            __gm__ int32_t* offsets_ptr = (__gm__ int32_t*)offsets;
            for (uint32_t i = 0; i < this->dim_K; i++) {
                int32_t offset_start = offsets_ptr[i];
                int32_t offset_end = offsets_ptr[i + 1];
                this->K[i] = static_cast<uint32_t>(offset_end - offset_start);
            }
        }
        else {
            // 输入是 Lengths (int32)，统一转为 uint32 存入 K
            __gm__ int32_t* lengths_ptr = (__gm__ int32_t*)lengths;
            for (uint32_t i = 0; i < this->dim_K; i++) {
                this->K[i] = static_cast<uint32_t>(lengths_ptr[i]);
            }
            // printf("dimK = %d\n", this->dim_K);
            // for (int i = 0;i <= 8;i++) {
            //     printf("K[%d] = %d\n", i, this->K[i]);
            // }
        }

        // 任务调度：使用 host 预计算的 shards_per_seg
        this->shards_per_seg = tiling_data.shardsPerSeg;
        
        uint32_t total_tasks = this->dim_K * this->shards_per_seg;
        this->task_id = this->BlockIdx;
        
        // 如果 BlockIdx 超出总任务数，直接返回，跳过后续初始化
        if (this->BlockIdx >= total_tasks) {
            
        }
        
        // 计算前缀和 prefixK：prefixK[0]=0, prefixK[i+1]=sum_{j<=i}K[j]
        prefixK[0] = 0;
        for (uint32_t i = 0; i < this->dim_K; ++i) {
            prefixK[i + 1] = prefixK[i] + static_cast<uint64_t>(this->K[i]);
        }

        // 根据 reduce_type 选择不同的 UB 规划与 tile_length 计算，
        // 实现 SUM/MEAN 与 MAX/MIN 的统一内核，同时对 BF16 做专门优化
        if (this->reduce_type == REDUCE_SUM || this->reduce_type == REDUCE_MEAN) {
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                // BF16 SUM/MEAN：使用 host 预计算的 maxTileLength
                uint32_t Max_Tile_Length = tiling_data.maxTileLength;
                this->tile_length = (this->n2_dim < Max_Tile_Length) ? this->n2_dim : Max_Tile_Length;
                if (this->tile_length == 0) this->tile_length = Max_Tile_Length;

                // 1. BF16 队列 (Input/Output) 内存计算
                uint32_t bf16_bytes = this->tile_length * sizeof(bfloat16_t);
                uint32_t aligned_bf16 = (bf16_bytes + 31) / 32 * 32;

                pipe->InitBuffer(inQueueGrad, BUFFER_NUM, aligned_bf16);
                pipe->InitBuffer(outQueueGradOutput, BUFFER_NUM, aligned_bf16);

                // 2. Float 队列 (Calc) 内存计算 (BF16 -> Float 需要 2 倍空间)
                uint32_t float_bytes = this->tile_length * sizeof(float);
                uint32_t aligned_float = (float_bytes + 31) / 32 * 32;
                pipe->InitBuffer(calcQueue, BUFFER_NUM, aligned_float);
            } else {
                // float / half SUM/MEAN 模式：使用 host 预计算的 maxTileLength
                uint32_t Max_Tile_Length = tiling_data.maxTileLength;
                this->tile_length = (this->n2_dim < Max_Tile_Length) ? this->n2_dim : Max_Tile_Length;
                if (this->tile_length == 0) this->tile_length = Max_Tile_Length;

                uint32_t buffer_size_bytes = this->tile_length * sizeof(T);
                uint32_t aligned_buffer_size = (buffer_size_bytes + 31) / 32 * 32;

                pipe->InitBuffer(inQueueGrad, BUFFER_NUM, aligned_buffer_size);
                pipe->InitBuffer(inQueueData, BUFFER_NUM, aligned_buffer_size);
                pipe->InitBuffer(outQueueGradOutput, BUFFER_NUM, aligned_buffer_size);
            }
        } else {
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                // BF16 MAX/MIN：使用 host 预计算的 maxTileLength
                uint32_t Max_Tile_Length = tiling_data.maxTileLength;
                this->tile_length = (this->n2_dim < Max_Tile_Length) ? this->n2_dim : Max_Tile_Length;
                if (this->tile_length == 0) this->tile_length = Max_Tile_Length;

                // 1. BF16 队列初始化 (2字节/元素)，保证 256 字节对齐
                uint32_t min_bytes = 256;
                uint32_t bf16_bytes = this->tile_length * sizeof(bfloat16_t);
                uint32_t final_bf16 = (bf16_bytes > min_bytes) ? bf16_bytes : min_bytes;
                uint32_t aligned_bf16 = (final_bf16 + 255) / 256 * 256;

                pipe->InitBuffer(inQueueGrad, BUFFER_NUM, aligned_bf16);
                pipe->InitBuffer(inQueueData, BUFFER_NUM, aligned_bf16);
                pipe->InitBuffer(outQueueGradOutput, BUFFER_NUM, aligned_bf16);
                pipe->InitBuffer(inQueueFwd, BUFFER_NUM, aligned_bf16);

                // 2. Float32 计算队列初始化 (4字节/元素)
                uint32_t float_bytes = this->tile_length * sizeof(float);
                uint32_t final_float = (float_bytes > min_bytes) ? float_bytes : min_bytes;
                uint32_t aligned_float = (final_float + 255) / 256 * 256;

                pipe->InitBuffer(gradFloatQueue, BUFFER_NUM, aligned_float);
                pipe->InitBuffer(fwdOutFloatQueue, BUFFER_NUM, aligned_float);
                pipe->InitBuffer(dataFloatQueue, BUFFER_NUM, aligned_float);

                // 3. Mask 队列
                uint32_t mask_size = this->tile_length * sizeof(uint8_t);
                uint32_t aligned_mask_size = (mask_size + 256) / 256 * 256;
                pipe->InitBuffer(inQueueMask, BUFFER_NUM, aligned_mask_size);
            } else if constexpr (std::is_same_v<T, half>) {
                // float16 MAX/MIN：使用 host 预计算的 maxTileLength
                uint32_t Max_Tile_Length = tiling_data.maxTileLength;
                this->tile_length = (this->n2_dim < Max_Tile_Length) ? this->n2_dim : Max_Tile_Length;
                if (this->tile_length == 0) this->tile_length = Max_Tile_Length;

                // 1. Half 队列初始化 (2字节/元素)，保证 256 字节对齐
                uint32_t min_bytes = 256;
                uint32_t half_bytes = this->tile_length * sizeof(half);
                uint32_t final_half = (half_bytes > min_bytes) ? half_bytes : min_bytes;
                uint32_t aligned_half = (final_half + 255) / 256 * 256;

                pipe->InitBuffer(inQueueGrad, BUFFER_NUM, aligned_half);
                pipe->InitBuffer(inQueueData, BUFFER_NUM, aligned_half);
                pipe->InitBuffer(outQueueGradOutput, BUFFER_NUM, aligned_half);
                pipe->InitBuffer(inQueueFwd, BUFFER_NUM, aligned_half);

                // 2. Float32 计算队列初始化 (4字节/元素)
                uint32_t float_bytes = this->tile_length * sizeof(float);
                uint32_t final_float = (float_bytes > min_bytes) ? float_bytes : min_bytes;
                uint32_t aligned_float = (final_float + 255) / 256 * 256;

                pipe->InitBuffer(gradFloatQueue, BUFFER_NUM, aligned_float);
                pipe->InitBuffer(fwdOutFloatQueue, BUFFER_NUM, aligned_float);
                pipe->InitBuffer(dataFloatQueue, BUFFER_NUM, aligned_float);

                // 3. Mask 队列
                uint32_t mask_size = this->tile_length * sizeof(uint8_t);
                uint32_t aligned_mask_size = (mask_size + 256) / 256 * 256;
                pipe->InitBuffer(inQueueMask, BUFFER_NUM, aligned_mask_size);
            } else {
                // float32 MAX/MIN 等模式：使用 host 预计算的 maxTileLength
                this->max_tile_length = tiling_data.maxTileLength;
                uint32_t Max_Tile_Length = this->max_tile_length;
                this->tile_length = (this->n2_dim < Max_Tile_Length) ? this->n2_dim : Max_Tile_Length;
                if (this->tile_length == 0) this->tile_length = Max_Tile_Length;

                uint32_t min_buffer_bytes = 256;
                uint32_t calc_buffer_bytes = this->tile_length * sizeof(T);
                uint32_t final_buffer_bytes = (calc_buffer_bytes > min_buffer_bytes) ? calc_buffer_bytes : min_buffer_bytes;
                uint32_t aligned_buffer_size = (final_buffer_bytes + 255) / 256 * 256;
                 
                pipe->InitBuffer(inQueueGrad, BUFFER_NUM, aligned_buffer_size);
                pipe->InitBuffer(inQueueFwd, BUFFER_NUM, aligned_buffer_size);
                uint32_t mask_size = this->tile_length * sizeof(uint8_t);
                uint32_t aligned_mask_size = (mask_size + 256) / 256 * 256;
                pipe->InitBuffer(inQueueMask, BUFFER_NUM, aligned_mask_size);
                uint32_t buffer_left = 98304 - BUFFER_NUM * aligned_buffer_size - BUFFER_NUM * aligned_mask_size;
                uint32_t aligned_data_buffer_size = buffer_left/512*256;
                // printf("aligned_data_buffer_size = %d\n", aligned_data_buffer_size);
                pipe->InitBuffer(inQueueData, BUFFER_NUM, aligned_data_buffer_size);
                pipe->InitBuffer(outQueueGradOutput, BUFFER_NUM, aligned_data_buffer_size);
            }
        }

    }

    __aicore__ inline void Process() {
        // 检查当前 Block 是否有有效任务
        uint32_t total_tasks = this->dim_K * this->shards_per_seg;
        if (this->task_id >= total_tasks) {
            return;  // 当前 Block 没有分配到任务
        }

        // 每个任务对应一个 (seg_idx, shard_idx)：
        // seg_idx  ∈ [0, dim_K)
        // shard_idx ∈ [0, shards_per_seg)，表示该 segment 的第几个 shard
        uint32_t seg_idx  = this->task_id % this->dim_K;
        uint32_t shard_idx = this->task_id / this->dim_K;  // [0, shards_per_seg)

        uint32_t seg_len = this->K[seg_idx];
        if (seg_len == 0) {
            return;  // 空 segment，直接返回
        }

        // 根据 shards_per_seg 计算每个 shard 的行范围
        // 将 seg_len 尽可能均匀地分成 shards_per_seg 份
        uint32_t base_len = seg_len / this->shards_per_seg;  // 每个 shard 的基础长度
        uint32_t rem_len = seg_len % this->shards_per_seg;    // 余数，需要分配给前 rem_len 个 shard

        uint32_t local_row_offset_in_seg;
        uint32_t local_seg_len;

        if (shard_idx < rem_len) {
            // 前 rem_len 个 shard，每个多分配 1 行
            local_row_offset_in_seg = shard_idx * (base_len + 1);
            local_seg_len = base_len + 1;
        } else {
            // 后面的 shard，每个分配 base_len 行
            local_row_offset_in_seg = rem_len * (base_len + 1) + (shard_idx - rem_len) * base_len;
            local_seg_len = base_len;
        }

        if (local_seg_len == 0) {
            return;  // 空 shard，直接返回
        }

        // 计算该 segment 在整体中的起始行号
        uint64_t base_row_offset   = prefixK[seg_idx];                 // 以行计
        uint64_t shard_row_offset  = base_row_offset + local_row_offset_in_seg; // 以行计

        // start_offset 以元素计，因此乘以 n2_dim
        uint64_t start_offset_elems = shard_row_offset * this->n2_dim;

        ProcessOneSegment(seg_idx,
                          static_cast<int32_t>(local_seg_len),
                          start_offset_elems);
    }

    __aicore__ inline void ProcessOneSegment(uint32_t seg_idx, int32_t seg_len, uint64_t start_offset) {
        if (seg_len <= 0) return;

        // seg_idx: segment 索引，用于读取 grad[seg_idx, :] 或 grad[:, seg_idx]
        // start_offset: 全局数据偏移，用于读取和写入 dataGm[start_offset + row*n2]
        // 计算 grad 的读取偏移（根据 axis 判断形状）
        uint64_t grad_read_offset;
        if (this->axis == 0) {
            // grad 形状为 (K, N2)，直接使用 seg_idx
            grad_read_offset = (uint64_t)seg_idx * this->n2_dim;
        } else {
            // grad 形状为 (N, K)，需要根据实际位置计算
            grad_read_offset = start_offset;  // 需要根据实际情况调整
        }

        if constexpr (std::is_same_v<T, bfloat16_t>) {
            // ===================== BF16 路径 =====================
            for (uint32_t offset = 0; offset < this->n2_dim; offset += this->tile_length) {
                uint32_t current_tile_length = this->tile_length;
                if (offset + this->tile_length > this->n2_dim) {
                    current_tile_length = this->n2_dim - offset;
                }

                if (this->reduce_type == REDUCE_SUM || this->reduce_type == REDUCE_MEAN) {
                    // -------- BF16 SUM / MEAN：复用原 KernelSegmentReduceGradSumBF16::ProcessOneSegment 逻辑 --------
                    DataCopyExtParams copyParams = { 1, (uint32_t)(current_tile_length * sizeof(bfloat16_t)), 0, 0, 0 };
                    DataCopyPadExtParams<bfloat16_t> padParams{ false, 0, 0, 0 };

                    // 1. 读取梯度 (BF16)
                    LocalTensor<bfloat16_t> gradLocal = inQueueGrad.AllocTensor<bfloat16_t>();
                    uint64_t grad_gm_offset = grad_read_offset;
                    DataCopyPad(gradLocal, gradGm[grad_gm_offset], copyParams, padParams);
                    inQueueGrad.EnQue(gradLocal);
                    gradLocal = inQueueGrad.DeQue<bfloat16_t>();

                    // 2. 转换为 Float 进行计算
                    LocalTensor<float> workLocal = calcQueue.AllocTensor<float>();
                    Cast(workLocal, gradLocal, RoundMode::CAST_NONE, current_tile_length);
                    calcQueue.EnQue(workLocal);
                    workLocal = calcQueue.DeQue<float>();
                    // 用完 BF16 输入即可释放
                    inQueueGrad.FreeTensor(gradLocal);

                    // 3. 执行 Muls 计算 (Float)
                    if (this->reduce_type == REDUCE_MEAN) {
                        // 注意：seg_len 是切分后的 shard 长度，需要使用整个 segment 的长度
                        uint32_t full_seg_len = this->K[seg_idx];
                        float scale = 1.0f / full_seg_len;
                        Muls(workLocal, workLocal, scale, current_tile_length);
                    } else { // REDUCE_SUM
                        Muls(workLocal, workLocal, 1.0f, current_tile_length);
                    }

                    // 4. 转换回 BF16 并写回
                    LocalTensor<bfloat16_t> gradOutputLocal = outQueueGradOutput.AllocTensor<bfloat16_t>();
                    Cast(gradOutputLocal, workLocal, RoundMode::CAST_ROUND, current_tile_length);

                    // 释放 Float 中间变量
                    calcQueue.FreeTensor(workLocal);

                    // 5. 广播/写回到 GM
                    for (int32_t row = 0; row < seg_len; row++) {
                        uint64_t out_gm_offset = start_offset + (uint64_t)row * this->n2_dim;
                        DataCopyPad(outputGm[out_gm_offset], gradOutputLocal, copyParams);
                    }
                    outQueueGradOutput.FreeTensor(gradOutputLocal);
                } else if (this->reduce_type == REDUCE_MAX || this->reduce_type == REDUCE_MIN) {
                    // -------- BF16 MAX / MIN：复用原 KernelSegmentReduceGradMinBF16::ProcessOneSegment 逻辑 --------
                    DataCopyExtParams copyParams = { 1, (uint32_t)(current_tile_length * sizeof(bfloat16_t)), 0, 0, 0 };
                    DataCopyPadExtParams<bfloat16_t> padParams{ false, 0, 0, 0 };

                    // 1. 读取梯度 (BF16)
                    LocalTensor<bfloat16_t> gradLocal = inQueueGrad.AllocTensor<bfloat16_t>();
                    uint64_t grad_gm_offset = grad_read_offset;
                    DataCopyPad(gradLocal, gradGm[grad_gm_offset], copyParams, padParams);
                    inQueueGrad.EnQue(gradLocal);
                    gradLocal = inQueueGrad.DeQue<bfloat16_t>();

                    uint32_t aligned_float_len = 256 / sizeof(float); // 64
                    if (current_tile_length > aligned_float_len) {
                        aligned_float_len = (current_tile_length * sizeof(float) + 255) / 256 * (256 / sizeof(float));
                    }

                    // 2. 将 grad 转换为 Float
                    LocalTensor<float> gradFloat = gradFloatQueue.AllocTensor<float>();
                    Cast(gradFloat, gradLocal, RoundMode::CAST_NONE, current_tile_length);
                    gradFloatQueue.EnQue(gradFloat);
                    gradFloat = gradFloatQueue.DeQue<float>();
                    // 此时可以释放 gradLocal (BF16)
                    inQueueGrad.FreeTensor(gradLocal);

                    // 3. 读取前向输出 (BF16) 并转为 Float
                    LocalTensor<bfloat16_t> fwdOutLocal = inQueueFwd.AllocTensor<bfloat16_t>();
                    DataCopyPad(fwdOutLocal, fwdOutGm[grad_read_offset + offset], copyParams, padParams);
                    inQueueFwd.EnQue(fwdOutLocal);
                    fwdOutLocal = inQueueFwd.DeQue<bfloat16_t>();

                    LocalTensor<float> fwdOutFloat = fwdOutFloatQueue.AllocTensor<float>();
                    Cast(fwdOutFloat, fwdOutLocal, RoundMode::CAST_NONE, current_tile_length);
                    fwdOutFloatQueue.EnQue(fwdOutFloat);
                    fwdOutFloat = fwdOutFloatQueue.DeQue<float>();
                    inQueueFwd.FreeTensor(fwdOutLocal);

                    // 4. 遍历 Segment
                    for (int32_t row = 0; row < seg_len; row++) {
                        // 4.1 读取 Data (BF16)
                        LocalTensor<bfloat16_t> dataLocal = inQueueData.AllocTensor<bfloat16_t>();
                        uint64_t data_gm_offset = start_offset + (uint64_t)row * this->n2_dim;
                        DataCopyPad(dataLocal, dataGm[data_gm_offset], copyParams, padParams);
                        inQueueData.EnQue(dataLocal);
                        dataLocal = inQueueData.DeQue<bfloat16_t>();

                        // 4.2 Data 转 Float
                        LocalTensor<float> dataFloat = dataFloatQueue.AllocTensor<float>();
                        Cast(dataFloat, dataLocal, RoundMode::CAST_NONE, current_tile_length);
                        dataFloatQueue.EnQue(dataFloat);
                        dataFloat = dataFloatQueue.DeQue<float>();
                        inQueueData.FreeTensor(dataLocal);

                        // 4.3 比较 (Float vs Float) -> 生成 Mask
                        LocalTensor<uint8_t> maskLocal = inQueueMask.AllocTensor<uint8_t>();
                        AscendC::Compare(maskLocal, dataFloat, fwdOutFloat, AscendC::CMPMODE::EQ, aligned_float_len);
                        inQueueMask.EnQue(maskLocal);
                        maskLocal = inQueueMask.DeQue<uint8_t>();

                        dataFloatQueue.FreeTensor(dataFloat);

                        // 4.4 准备结果容器 (Float)
                        LocalTensor<float> resultFloat = dataFloatQueue.AllocTensor<float>();
                        Duplicate(resultFloat, 0.0f, aligned_float_len);

                        // 4.5 Select (Float)
                        AscendC::Select(resultFloat, maskLocal, gradFloat, resultFloat,
                                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, aligned_float_len);
                        dataFloatQueue.EnQue(resultFloat);
                        resultFloat = dataFloatQueue.DeQue<float>();

                        // 4.6 结果转回 BF16
                        LocalTensor<bfloat16_t> gradOutputLocal = outQueueGradOutput.AllocTensor<bfloat16_t>();
                        Cast(gradOutputLocal, resultFloat, RoundMode::CAST_ROUND, current_tile_length);
                        outQueueGradOutput.EnQue(gradOutputLocal);
                        gradOutputLocal = outQueueGradOutput.DeQue<bfloat16_t>();

                        // 4.7 写回
                        uint64_t out_gm_offset = start_offset + (uint64_t)row * this->n2_dim;
                        DataCopyPad(outputGm[out_gm_offset], gradOutputLocal, copyParams);

                        // 释放资源
                        dataFloatQueue.FreeTensor(resultFloat);
                        outQueueGradOutput.FreeTensor(gradOutputLocal);
                        inQueueMask.FreeTensor(maskLocal);
                    }

                    // Segment 结束后释放常驻的 Float Tensor
                    gradFloatQueue.FreeTensor(gradFloat);
                    fwdOutFloatQueue.FreeTensor(fwdOutFloat);
                }
            }
        } else {
            // ===================== float / half 路径 =====================
            // UB 足够大，tile_length 较小，一次性处理整行 n2_dim
            uint32_t current_tile_length = this->n2_dim;

            DataCopyExtParams copyParams = { 1, (uint32_t)(current_tile_length * sizeof(T)), 0, 0, 0 };
            DataCopyPadExtParams<T> padParams{ false, 0, 0, 0 };

            // 读取上游梯度 grad[seg_idx, :]
            LocalTensor<T> gradLocal = inQueueGrad.AllocTensor<T>();
            uint64_t grad_gm_offset = grad_read_offset;
            DataCopyPad(gradLocal, gradGm[grad_gm_offset], copyParams, padParams);
            inQueueGrad.EnQue(gradLocal);
            gradLocal = inQueueGrad.DeQue<T>();

            if (this->reduce_type == REDUCE_SUM || this->reduce_type == REDUCE_MEAN) {
                // SUM/MEAN 路径：只对整段做缩放并在段内广播
                LocalTensor<T> gradOutputLocal = outQueueGradOutput.AllocTensor<T>();

                if (this->reduce_type == REDUCE_SUM) {
                    T scale = (T)(1.0);
                    Muls(gradOutputLocal, gradLocal, scale, current_tile_length);
                } else {
                    // MEAN: grad_output = grad / full_seg_len 广播到每个元素
                    uint32_t full_seg_len = this->K[seg_idx];
                    T scale = (T)(1.0f / full_seg_len);
                    Muls(gradOutputLocal, gradLocal, scale, current_tile_length);
                }
                outQueueGradOutput.EnQue(gradOutputLocal);
                gradOutputLocal = outQueueGradOutput.DeQue<T>();
                // 广播到所有行
                for (int32_t row = 0; row < seg_len; row++) {
                    uint64_t out_gm_offset = start_offset + (uint64_t)row * this->n2_dim;
                    DataCopyPad(outputGm[out_gm_offset], gradOutputLocal, copyParams);
                }
                outQueueGradOutput.FreeTensor(gradOutputLocal);
            } else if (this->reduce_type == REDUCE_MAX || this->reduce_type == REDUCE_MIN) {
                // MAX/MIN 路径：根据前向输出的位置生成掩码，然后选择性回传梯度
                if constexpr (std::is_same_v<T, half>) {
                    // float16 MAX/MIN：使用 float 进行计算以提高精度
                    uint32_t aligned_float_len = 256 / sizeof(float); // 64
                    if (current_tile_length > aligned_float_len) {
                        aligned_float_len = (current_tile_length * sizeof(float) + 255) / 256 * (256 / sizeof(float));
                    }

                    // 1. 读取梯度 (Half)
                    LocalTensor<half> gradLocalHalf = inQueueGrad.AllocTensor<half>();
                    uint64_t grad_gm_offset = grad_read_offset;
                    DataCopyPad(gradLocalHalf, gradGm[grad_gm_offset], copyParams, padParams);
                    inQueueGrad.EnQue(gradLocalHalf);
                    gradLocalHalf = inQueueGrad.DeQue<half>();

                    // 2. 将 grad 转换为 Float
                    LocalTensor<float> gradFloat = gradFloatQueue.AllocTensor<float>();
                    Cast(gradFloat, gradLocalHalf, RoundMode::CAST_NONE, current_tile_length);
                    gradFloatQueue.EnQue(gradFloat);
                    gradFloat = gradFloatQueue.DeQue<float>();
                    inQueueGrad.FreeTensor(gradLocalHalf);

                    // 3. 读取前向输出 (Half) 并转为 Float
                    LocalTensor<half> fwdOutLocalHalf = inQueueFwd.AllocTensor<half>();
                    DataCopyPad(fwdOutLocalHalf, fwdOutGm[grad_read_offset], copyParams, padParams);
                    inQueueFwd.EnQue(fwdOutLocalHalf);
                    fwdOutLocalHalf = inQueueFwd.DeQue<half>();

                    LocalTensor<float> fwdOutFloat = fwdOutFloatQueue.AllocTensor<float>();
                    Cast(fwdOutFloat, fwdOutLocalHalf, RoundMode::CAST_NONE, current_tile_length);
                    fwdOutFloatQueue.EnQue(fwdOutFloat);
                    fwdOutFloat = fwdOutFloatQueue.DeQue<float>();
                    inQueueFwd.FreeTensor(fwdOutLocalHalf);

                    // 4. 遍历 Segment
                    for (int32_t row = 0; row < seg_len; row++) {
                        // 4.1 读取 Data (Half)
                        LocalTensor<half> dataLocalHalf = inQueueData.AllocTensor<half>();
                        uint64_t data_gm_offset = start_offset + (uint64_t)row * this->n2_dim;
                        DataCopyPad(dataLocalHalf, dataGm[data_gm_offset], copyParams, padParams);
                        inQueueData.EnQue(dataLocalHalf);
                        dataLocalHalf = inQueueData.DeQue<half>();

                        // 4.2 Data 转 Float
                        LocalTensor<float> dataFloat = dataFloatQueue.AllocTensor<float>();
                        Cast(dataFloat, dataLocalHalf, RoundMode::CAST_NONE, current_tile_length);
                        dataFloatQueue.EnQue(dataFloat);
                        dataFloat = dataFloatQueue.DeQue<float>();
                        inQueueData.FreeTensor(dataLocalHalf);

                        // 4.3 比较 (Float vs Float) -> 生成 Mask
                        LocalTensor<uint8_t> maskLocal = inQueueMask.AllocTensor<uint8_t>();
                        AscendC::Compare(maskLocal, dataFloat, fwdOutFloat, AscendC::CMPMODE::EQ, aligned_float_len);
                        inQueueMask.EnQue(maskLocal);
                        maskLocal = inQueueMask.DeQue<uint8_t>();

                        dataFloatQueue.FreeTensor(dataFloat);

                        // 4.4 准备结果容器 (Float)
                        LocalTensor<float> resultFloat = dataFloatQueue.AllocTensor<float>();
                        Duplicate(resultFloat, 0.0f, aligned_float_len);

                        // 4.5 Select (Float)
                        AscendC::Select(resultFloat, maskLocal, gradFloat, resultFloat,
                                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, aligned_float_len);
                        dataFloatQueue.EnQue(resultFloat);
                        resultFloat = dataFloatQueue.DeQue<float>();

                        // 4.6 结果转回 Half
                        LocalTensor<half> gradOutputLocal = outQueueGradOutput.AllocTensor<half>();
                        Cast(gradOutputLocal, resultFloat, RoundMode::CAST_ROUND, current_tile_length);
                        outQueueGradOutput.EnQue(gradOutputLocal);
                        gradOutputLocal = outQueueGradOutput.DeQue<half>();

                        // 4.7 写回
                        uint64_t out_gm_offset = start_offset + (uint64_t)row * this->n2_dim;
                        DataCopyPad(outputGm[out_gm_offset], gradOutputLocal, copyParams);

                        // 释放资源
                        dataFloatQueue.FreeTensor(resultFloat);
                        outQueueGradOutput.FreeTensor(gradOutputLocal);
                        inQueueMask.FreeTensor(maskLocal);
                    }

                    // Segment 结束后释放常驻的 Float Tensor
                    gradFloatQueue.FreeTensor(gradFloat);
                    fwdOutFloatQueue.FreeTensor(fwdOutFloat);
                } else {
                    // float32 MAX/MIN：SetFlag/WaitFlag 位置优化，标量与 MTE 重叠
                    int32_t eventIdMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    int32_t eventIdVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

                    copyParams.blockCount = 1;
                    copyParams.blockLen = this->row_size_bytes;
                    copyParams.srcStride = 0;
                    copyParams.dstStride = 0;
                    int batch_rows = this->max_tile_length / current_tile_length;
                    if (batch_rows > seg_len) batch_rows = seg_len;

                    LocalTensor<T> fwdOutLocal = inQueueFwd.AllocTensor<T>();
                    DataCopyPad(fwdOutLocal, fwdOutGm[grad_read_offset], copyParams, padParams);
                    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
                    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

                    LocalTensor<T> gradOutputLocal = outQueueGradOutput.AllocTensor<T>();

                    int actual_batch = (0 + batch_rows > seg_len) ? (seg_len - 0) : batch_rows;
                    copyParams.blockLen = this->row_size_bytes * (uint32_t)actual_batch;
                    uint64_t data_gm_offset = start_offset;
                    int32_t row = 0;
                    while (row < seg_len) {
                        LocalTensor<T> dataLocal = inQueueData.AllocTensor<T>();
                        LocalTensor<uint8_t> maskLocal = inQueueMask.AllocTensor<uint8_t>();

                        DataCopyPad(dataLocal, dataGm[data_gm_offset], copyParams, padParams);
                        // MTE2 异步搬运中：预算下一轮参数，与 MTE2 重叠
                        int next_row = row + actual_batch;
                        int next_actual_batch = (next_row + batch_rows > seg_len) ? (seg_len - next_row) : batch_rows;
                        uint64_t next_data_gm_offset = start_offset + (uint64_t)next_row * this->n2_dim;
                        uint64_t out_gm_offset = start_offset + (uint64_t)row * this->n2_dim;
                        SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
                        WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

                        uint32_t cur_total = (uint32_t)(current_tile_length * actual_batch);
                        for (uint32_t ub_offset = 0; ub_offset < cur_total; ub_offset += current_tile_length) {
                            AscendC::Compare(maskLocal, dataLocal[ub_offset], fwdOutLocal[0], AscendC::CMPMODE::EQ, this->row_compute_len);
                            AscendC::Select(gradOutputLocal[ub_offset], maskLocal, gradLocal, T(0.0),
                                           AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->row_compute_len);
                        }
                        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

                        DataCopyPad(outputGm[out_gm_offset], gradOutputLocal, copyParams);
                        // MTE3 异步写出中：释放、更新，与 MTE3 重叠
                        inQueueData.FreeTensor(dataLocal);
                        inQueueMask.FreeTensor(maskLocal);
                        row = next_row;
                        actual_batch = next_actual_batch;
                        data_gm_offset = next_data_gm_offset;
                        if (row < seg_len) {
                            copyParams.blockLen = this->row_size_bytes * (uint32_t)actual_batch;
                        }
                    }

                    inQueueFwd.FreeTensor(fwdOutLocal);
                    outQueueGradOutput.FreeTensor(gradOutputLocal);
                }
            }

            inQueueGrad.FreeTensor(gradLocal);
        }
    }

};


// BF16 专用类已合并进通用模板 KernelSegmentReduceGrad<T>（T=bfloat16_t），此处删除旧实现

template<typename T>
class KernelSegmentReduceGradAxis1 {
private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGrad;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueGradOutput;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueData;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueFwd;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueMask;

    GlobalTensor<T> gradGm;
    GlobalTensor<T> dataGm;        // 前向输入 data（用于 MAX/MIN）
    GlobalTensor<T> fwdOutGm;      // 前向输出 output（用于 MAX/MIN）
    GlobalTensor<T> outputGm;
    GlobalTensor<int32_t> lengthsGm;
    GlobalTensor<int32_t> offsetsGm;
    uint32_t N;
    uint32_t N2;
    uint32_t outer_Offset;   // N (tiling_data.outerOffset)
    uint32_t dim_K;       // K (每行的 segment 数量)
    uint32_t inner_Offset;   // N2 (tiling_data.innerOffset, case15中为1)
    uint32_t use_offsets;
    uint32_t reduce_type;

    // 任务调度：根据 total_segments 动态计算每个 segment 拆分的份数，使总任务数尽可能接近但不超过 40
    uint32_t shards_per_seg; // 每个 segment 拆分的份数
    uint32_t task_id;        // 当前 Block 负责的任务 id

    uint32_t tile_length;
    int BlockIdx, BlockNum;

public:
    __aicore__ inline KernelSegmentReduceGradAxis1() {}

    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR output, GM_ADDR data, GM_ADDR lengths, GM_ADDR offsets, GM_ADDR grad_output, GM_ADDR initial,
        SegmentReduceGradTilingData& tiling_data, TPipe* pipeIn) {
        pipe = pipeIn;

        // 核心维度信息映射
        this->N = tiling_data.N;
        this->N2 = tiling_data.N2;
        this->outer_Offset = tiling_data.outerOffset;
        this->inner_Offset = tiling_data.innerOffset;
        this->dim_K = tiling_data.dim_K;
        this->use_offsets = tiling_data.useOffsets;
        this->reduce_type = tiling_data.reduceType;

        this->BlockIdx = GetBlockIdx();
        this->BlockNum = GetBlockNum();

        // 设置 GM 指针
        gradGm.SetGlobalBuffer((__gm__ T*)grad);
        dataGm.SetGlobalBuffer((__gm__ T*)data);
        fwdOutGm.SetGlobalBuffer((__gm__ T*)output);
        outputGm.SetGlobalBuffer((__gm__ T*)grad_output);
        lengthsGm.SetGlobalBuffer((__gm__ int32_t*)lengths);
        offsetsGm.SetGlobalBuffer((__gm__ int32_t*)offsets);

        // 任务调度：根据 total_segments 动态计算每个 segment 拆分的份数
        // 目标：总任务数尽可能接近但不超过 40，充分利用 40 个核
        uint32_t total_segments = this->outer_Offset * this->dim_K;
        const uint32_t MAX_CORES = 40;
        
        // 计算每个 segment 应该拆分成多少份
        // 按照 40/total_segments 向下取整来计算每个 segment 最多拆分的份数
        if (total_segments == 0) {
            this->shards_per_seg = 1;
        } else {
            this->shards_per_seg = MAX_CORES / total_segments;  // 整数除法，自动向下取整
            // 至少拆分成 1 份（不拆分）
            if (this->shards_per_seg == 0) {
                this->shards_per_seg = 1;
            }
        }
        
        uint32_t total_tasks = total_segments * this->shards_per_seg;
        this->task_id = this->BlockIdx;
        
        // 如果 BlockIdx 超出总任务数，直接返回，跳过后续初始化
        if (this->BlockIdx >= total_tasks) {
            return;
        }

        // 根据 reduce_type 选择不同 UB 规划，统一 Sum / Mean / Max / Min
        if (this->reduce_type == REDUCE_SUM || this->reduce_type == REDUCE_MEAN) {
            uint32_t Max_Tile_Length;
            if constexpr (std::is_same_v<T, float>) {
                Max_Tile_Length = 6144;
            } else {
                Max_Tile_Length = 10240;
            }
            // 对 inner_Offset 进行切分
            this->tile_length = Max_Tile_Length;

            uint32_t buffer_size_bytes = this->tile_length * sizeof(T);
            uint32_t aligned_buffer_size = (buffer_size_bytes + 31) / 32 * 32;

            pipe->InitBuffer(inQueueGrad, BUFFER_NUM, aligned_buffer_size);
            pipe->InitBuffer(outQueueGradOutput, BUFFER_NUM, aligned_buffer_size);
        } else {
            // MAX/MIN：需要 data/fwdOut/mask，tile 更小
            uint32_t Max_Tile_Length;
            if constexpr (std::is_same_v<T, float>) {
                Max_Tile_Length = 2944;
            } else {
                Max_Tile_Length = 5760;
            }
            this->tile_length = Max_Tile_Length;

            uint32_t min_buffer_bytes = 256;
            uint32_t calc_buffer_bytes = this->tile_length * sizeof(T);
            uint32_t final_buffer_bytes = (calc_buffer_bytes > min_buffer_bytes) ? calc_buffer_bytes : min_buffer_bytes;
            uint32_t aligned_buffer_size = (final_buffer_bytes + 255) / 256 * 256;

            pipe->InitBuffer(inQueueGrad, BUFFER_NUM, aligned_buffer_size);
            pipe->InitBuffer(inQueueData, BUFFER_NUM, aligned_buffer_size);
            pipe->InitBuffer(inQueueFwd, BUFFER_NUM, aligned_buffer_size);
            pipe->InitBuffer(outQueueGradOutput, BUFFER_NUM, aligned_buffer_size);

            uint32_t mask_size = this->tile_length * sizeof(uint8_t);
            uint32_t aligned_mask_size = (mask_size + 255) / 256 * 256;
            pipe->InitBuffer(inQueueMask, BUFFER_NUM, aligned_mask_size);
        }
    }

    __aicore__ inline void Process() {
        // 检查当前 Block 是否有有效任务
        uint32_t total_segments = this->outer_Offset * this->dim_K;
        uint32_t total_tasks = total_segments * this->shards_per_seg;
        if (this->task_id >= total_tasks) {
            return;  // 当前 Block 没有分配到任务
        }

        // 每个任务对应一个 (seg_idx, shard_idx)：
        // seg_idx  ∈ [0, total_segments)
        // shard_idx ∈ [0, shards_per_seg)，表示该 segment 的第几个 shard
        uint32_t seg_idx  = this->task_id % total_segments;
        uint32_t shard_idx = this->task_id / total_segments;  // [0, shards_per_seg)

        // 解析二维坐标
        uint32_t n_idx = seg_idx / this->dim_K; // 第几行 (Outer)
        uint32_t k_idx = seg_idx % this->dim_K; // 该行第几个 segment

        int32_t seg_len = 0;
        uint32_t start_n2_idx = 0; // 该 segment 在该行内的起始 row 偏移

        // 获取长度和行内起始位置
        if (this->use_offsets == 1) {
            uint32_t offset_base = n_idx * (this->dim_K + 1);
            int32_t o_start = offsetsGm.GetValue(offset_base + k_idx);
            int32_t o_end = offsetsGm.GetValue(offset_base + k_idx + 1);
            seg_len = o_end - o_start;
            start_n2_idx = (uint32_t)o_start;
        }
        else {
            uint32_t length_base = n_idx * this->dim_K;
            seg_len = lengthsGm.GetValue(length_base + k_idx);
            for (uint32_t j = 0; j < k_idx; j++) {
                start_n2_idx += (uint32_t)lengthsGm.GetValue(length_base + j);
            }
        }

        if (seg_len <= 0) {
            return;  // 空 segment，直接返回
        }

        // 根据 shards_per_seg 计算每个 shard 的长度范围
        // 将 seg_len 尽可能均匀地分成 shards_per_seg 份
        uint32_t base_len = seg_len / this->shards_per_seg;  // 每个 shard 的基础长度
        uint32_t rem_len = seg_len % this->shards_per_seg;    // 余数，需要分配给前 rem_len 个 shard

        uint32_t local_offset_in_seg;
        uint32_t local_seg_len;

        if (shard_idx < rem_len) {
            // 前 rem_len 个 shard，每个多分配 1 个元素
            local_offset_in_seg = shard_idx * (base_len + 1);
            local_seg_len = base_len + 1;
        } else {
            // 后面的 shard，每个分配 base_len 个元素
            local_offset_in_seg = rem_len * (base_len + 1) + (shard_idx - rem_len) * base_len;
            local_seg_len = base_len;
        }

        if (local_seg_len == 0) {
            return;  // 空 shard，直接返回
        }

        // 计算该行总长度 N2 (用于计算 outputGm 的地址)
        uint32_t total_L_in_n = this->N2;
        uint32_t shard_start_n2_idx = start_n2_idx + local_offset_in_seg;

        ProcessOneSegment(n_idx, k_idx, static_cast<int32_t>(local_seg_len), static_cast<int32_t>(seg_len), shard_start_n2_idx, total_L_in_n);
    }

    __aicore__ inline void ProcessOneSegment(uint32_t n_idx, uint32_t k_idx, int32_t shard_len, int32_t full_seg_len, uint32_t start_n2_idx, uint32_t total_L) {

        // 1. 计算公共地址
        uint64_t grad_idx = (uint64_t)n_idx * this->dim_K + k_idx;                 // grad 形状 (N, K)
        uint64_t base_idx = (uint64_t)n_idx * total_L + start_n2_idx;              // data/output 形状 (N, N2) 时的起始地址

        if (this->reduce_type == REDUCE_SUM || this->reduce_type == REDUCE_MEAN) {
            // ===== SUM / MEAN：整段广播标量 =====
            uint64_t output_base_idx = base_idx;

            T grad_val = gradGm.GetValue(grad_idx);
            PipeBarrier<PIPE_ALL>();

            if (this->reduce_type == REDUCE_MEAN) {
                // 注意：使用整个 segment 的长度 full_seg_len，而不是切分后的 shard_len
                if constexpr (std::is_same_v<T, half>) {
                    float f_grad = static_cast<float>(grad_val);
                    float f_seg_len = static_cast<float>(full_seg_len);
                    grad_val = static_cast<T>(f_grad / f_seg_len);
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    float f_grad = static_cast<float>(grad_val);
                    float f_seg_len = static_cast<float>(full_seg_len);
                    grad_val = static_cast<T>(f_grad / f_seg_len);
                } else {
                    grad_val = grad_val / static_cast<T>(full_seg_len);
                }
            }

            uint32_t processed = 0;
            while (processed < (uint32_t)shard_len) {
                uint32_t cur_len = ((uint32_t)shard_len - processed > this->tile_length) ?
                    this->tile_length : ((uint32_t)shard_len - processed);

                DataCopyExtParams copyParamsOutput = { 1, (uint32_t)(cur_len * sizeof(T)), 0, 0, 0 };
                LocalTensor<T> outLocal = outQueueGradOutput.AllocTensor<T>();
                Duplicate(outLocal, grad_val, cur_len);
                outQueueGradOutput.EnQue(outLocal);
                outQueueGradOutput.DeQue<T>();

                DataCopyPad(outputGm[output_base_idx + processed], outLocal, copyParamsOutput);
                outQueueGradOutput.FreeTensor(outLocal);

                processed += cur_len;
            }
        } else if (this->reduce_type == REDUCE_MAX || this->reduce_type == REDUCE_MIN) {
            // ===== MAX / MIN：根据前向最值位置生成 mask =====
            uint64_t data_base_idx = base_idx;
            uint64_t output_base_idx = base_idx;
            uint64_t fwd_out_idx = grad_idx;   // fwdOut 形状 (N, K)，存每个 segment 的最值

            T grad_val = gradGm.GetValue(grad_idx);
            T fwd_out_val = fwdOutGm.GetValue(fwd_out_idx);

            uint32_t processed = 0;
            while (processed < (uint32_t)shard_len) {
                uint32_t cur_len = ((uint32_t)shard_len - processed > this->tile_length) ?
                    this->tile_length : ((uint32_t)shard_len - processed);

                DataCopyExtParams copyParams = { 1, (uint32_t)(cur_len * sizeof(T)), 0, 0, 0 };
                DataCopyPadExtParams<T> padParams{ false, 0, 0, 0 };

                // 读取 data 片段
                LocalTensor<T> dataLocal = inQueueData.AllocTensor<T>();
                DataCopyPad(dataLocal, dataGm[data_base_idx + processed], copyParams, padParams);
                inQueueData.EnQue(dataLocal);
                dataLocal = inQueueData.DeQue<T>();

                // 广播 fwd_out_val
                LocalTensor<T> fwdOutLocal = inQueueFwd.AllocTensor<T>();
                Duplicate(fwdOutLocal, fwd_out_val, cur_len);
                inQueueFwd.EnQue(fwdOutLocal);
                fwdOutLocal = inQueueFwd.DeQue<T>();

                // 比较 data == fwdOut，生成 mask
                uint32_t aligned_compute_len = 256 / sizeof(T);
                if (cur_len > aligned_compute_len) {
                    aligned_compute_len = (cur_len * sizeof(T) + 255) / 256 * (256 / sizeof(T));
                }
                LocalTensor<uint8_t> maskLocal = inQueueMask.AllocTensor<uint8_t>();
                AscendC::Compare(maskLocal, dataLocal, fwdOutLocal, AscendC::CMPMODE::EQ, aligned_compute_len);
                inQueueMask.EnQue(maskLocal);
                maskLocal = inQueueMask.DeQue<uint8_t>();

                // 准备梯度：先广播 grad_val，再用 mask 选择性写入
                LocalTensor<T> gradLocal = inQueueGrad.AllocTensor<T>();
                Duplicate(gradLocal, grad_val, aligned_compute_len);
                inQueueGrad.EnQue(gradLocal);
                gradLocal = inQueueGrad.DeQue<T>();

                LocalTensor<T> gradOutputLocal = outQueueGradOutput.AllocTensor<T>();
                Duplicate(gradOutputLocal, (T)0.0, aligned_compute_len);
                Select(gradOutputLocal, maskLocal, gradLocal, gradOutputLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, aligned_compute_len);
                outQueueGradOutput.EnQue(gradOutputLocal);
                gradOutputLocal = outQueueGradOutput.DeQue<T>();

                // 写回结果
                DataCopyPad(outputGm[output_base_idx + processed], gradOutputLocal, copyParams);

                // 释放资源
                inQueueData.FreeTensor(dataLocal);
                inQueueFwd.FreeTensor(fwdOutLocal);
                inQueueMask.FreeTensor(maskLocal);
                inQueueGrad.FreeTensor(gradLocal);
                outQueueGradOutput.FreeTensor(gradOutputLocal);

                processed += cur_len;
            }
        }
    }
};



// 输入顺序: grad, output, data, lengths, offsets, initial, (outputs...) -> grad_output, workspace, tiling
extern "C" __global__ __aicore__ void segment_reduce_grad(GM_ADDR grad, GM_ADDR output, GM_ADDR data, GM_ADDR lengths, GM_ADDR offsets, GM_ADDR initial, GM_ADDR grad_output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(0)) {  
        KernelSegmentReduceGrad<DTYPE_GRAD> op;
        op.Init(grad, output, data, lengths, offsets, grad_output, initial, tiling_data, &pipe);
        op.Process();
    }
    else if(TILING_KEY_IS(1)) {
        KernelSegmentReduceGradAxis1<float> op;
        op.Init(grad, output, data, lengths, offsets, grad_output, initial, tiling_data, &pipe);
        op.Process();
    }
    else if(TILING_KEY_IS(2)) {
        KernelSegmentReduceGradAxis1<half> op;
        op.Init(grad, output, data, lengths, offsets, grad_output, initial, tiling_data, &pipe);
        op.Process();
    }
}
