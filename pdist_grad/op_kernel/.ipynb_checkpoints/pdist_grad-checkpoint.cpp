#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t COL_TILE_SIZE = 32;
constexpr int32_t SCALAR_BATCH_SIZE = 64; 
constexpr int32_t ALIGN_PAD = 8;
constexpr float EPSILON = 1e-20f;
constexpr float P_EPS = 1e-6f;

class KernelPdistGrad
{
public:
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR input, GM_ADDR pdist, GM_ADDR out, GM_ADDR tiling)
    {
        GET_TILING_DATA(tiling_data, tiling);
        n = tiling_data.n;
        total_m = tiling_data.m;
        p = tiling_data.p;

        uint32_t core_idx = GetBlockIdx();
        m_per_core = tiling_data.m_per_core;

        m_start = core_idx * m_per_core;
        m_len = m_per_core;
        if (core_idx == GetBlockNum() - 1)
        {
            m_len = tiling_data.tail_m_core;
        }

        gradGm.SetGlobalBuffer((__gm__ float *)grad);
        inputGm.SetGlobalBuffer((__gm__ float *)input);
        pdistGm.SetGlobalBuffer((__gm__ float *)pdist);
        outGm.SetGlobalBuffer((__gm__ float *)out);

        // UB 元素个数对齐到 32 字节 (8 floats)
        uint32_t ub_ele_num = (COL_TILE_SIZE + 7) / 8 * 8;

        pipe.InitBuffer(inQueue, 1, n * ub_ele_num * sizeof(float));
        pipe.InitBuffer(outQueue, 1, n * ub_ele_num * sizeof(float));
        pipe.InitBuffer(tmpQueue, 2, ub_ele_num * sizeof(float));
        
        // 标量 Buffer 包含 padding 以处理对齐加载
        uint32_t scalar_buf_size = (SCALAR_BATCH_SIZE + ALIGN_PAD) * 2 * sizeof(float);
        pipe.InitBuffer(scalarQueue, 1, scalar_buf_size);
    }

    __aicore__ inline void Process()
    {
        if (m_len == 0) return;

        for (uint32_t m_offset = 0; m_offset < m_len; m_offset += COL_TILE_SIZE)
        {
            uint32_t current_tile_width = COL_TILE_SIZE;
            if (m_offset + current_tile_width > m_len)
            {
                current_tile_width = m_len - m_offset;
            }
            ComputeTile(m_start + m_offset, current_tile_width);
        }
    }

private:
    __aicore__ inline void ComputeTile(uint32_t global_col_offset, uint32_t tile_width)
    {
        uint32_t ub_row_stride = (COL_TILE_SIZE + 7) / 8 * 8;
        uint32_t vec_calc_len = (tile_width + 7) / 8 * 8;

        LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
        LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
        LocalTensor<float> tmpDiff = tmpQueue.AllocTensor<float>();
        LocalTensor<float> tmpRes = tmpQueue.AllocTensor<float>();

        // 1. 初始化 UB (Vector 指令)
        Duplicate(inputLocal, 0.0f, n * ub_row_stride);
        Duplicate(outLocal, 0.0f, n * ub_row_stride);

        // [关键修复] 添加同步屏障
        // 必须确保 Duplicate (Vector pipe) 执行完毕后，才能执行后续的 DataCopy (MTE2 pipe)
        // 否则 DataCopy 搬入的数据可能会被未完成的 Duplicate 覆盖为 0
        PipeBarrier<PIPE_ALL>();

        // 2. 搬运 Input (MTE2 指令)
        for (uint32_t i = 0; i < n; i++)
        {
            uint64_t global_offset = (uint64_t)i * total_m + global_col_offset;
            DataCopy(inputLocal[i * ub_row_stride], inputGm[global_offset], tile_width);
        }

        // [关键修复] 添加同步屏障
        // 确保数据搬运完成，才能被后续 Vector 计算指令 (Sub) 读取
        PipeBarrier<PIPE_ALL>();

        uint64_t total_pairs = (uint64_t)n * (n - 1) / 2;
        uint64_t pair_cnt = 0;
        uint32_t batch_idx = SCALAR_BATCH_SIZE; 
        
        LocalTensor<float> scalars;
        uint32_t ub_offset = 0; 

        for (uint32_t i = 0; i < n - 1; i++)
        {
            for (uint32_t j = i + 1; j < n; j++)
            {
                if (batch_idx >= SCALAR_BATCH_SIZE)
                {
                    if (pair_cnt > 0) scalarQueue.FreeTensor(scalars);

                    scalars = scalarQueue.AllocTensor<float>();
                    
                    uint32_t remain = total_pairs - pair_cnt;
                    uint32_t load_req = (remain > SCALAR_BATCH_SIZE) ? SCALAR_BATCH_SIZE : remain;
                    
                    // 计算对齐加载地址
                    uint64_t aligned_pair_cnt = (pair_cnt / 8) * 8;
                    ub_offset = pair_cnt - aligned_pair_cnt;
                    
                    uint32_t elements_to_load = ub_offset + load_req;
                    uint32_t load_align_len = (elements_to_load * sizeof(float) + 31) / 32 * 8;
                    
                    uint32_t pdist_buffer_offset = SCALAR_BATCH_SIZE + ALIGN_PAD;

                    DataCopy(scalars[0], gradGm[aligned_pair_cnt], load_align_len);
                    DataCopy(scalars[pdist_buffer_offset], pdistGm[aligned_pair_cnt], load_align_len);

                    // EnQue/DeQue 机制隐含了 MTE2 -> Scalar/Vector 的同步
                    scalarQueue.EnQue(scalars);
                    scalars = scalarQueue.DeQue<float>();
                    
                    batch_idx = 0;
                }

                uint32_t read_idx = ub_offset + batch_idx;
                uint32_t pdist_buffer_offset = SCALAR_BATCH_SIZE + ALIGN_PAD;
                
                float g = scalars.GetValue(read_idx);
                float d = scalars.GetValue(pdist_buffer_offset + read_idx);
                
                batch_idx++;
                pair_cnt++;

                float scale = 0.0f;
                // 防止除0及数值异常
                if (d > 1e-12f || d < -1e-12f)
                {
                    if (p > 2.0f - P_EPS && p < 2.0f + P_EPS)
                    {
                        scale = g / d;
                    }
                    else
                    {
                        Duplicate(tmpRes, d, 8); 
                        Ln(tmpRes, tmpRes, 8);
                        Muls(tmpRes, tmpRes, p - 1.0f, 8);
                        Exp(tmpRes, tmpRes, 8);
                        float d_pow = tmpRes.GetValue(0);
                        scale = g / d_pow;
                    }
                }

                Sub(tmpDiff, inputLocal[i * ub_row_stride], inputLocal[j * ub_row_stride], vec_calc_len);

                if (p > 2.0f - P_EPS && p < 2.0f + P_EPS)
                {
                    Muls(tmpRes, tmpDiff, scale, vec_calc_len);
                }
                else
                {
                    // p!=2 时的数值稳定性保护
                    Abs(tmpRes, tmpDiff, vec_calc_len);
                    Maxs(tmpRes, tmpRes, EPSILON, vec_calc_len); // 防止 Ln(0)
                    Ln(tmpRes, tmpRes, vec_calc_len);
                    Muls(tmpRes, tmpRes, p - 2.0f, vec_calc_len);
                    Exp(tmpRes, tmpRes, vec_calc_len);
                    Mul(tmpRes, tmpRes, tmpDiff, vec_calc_len);
                    Muls(tmpRes, tmpRes, scale, vec_calc_len);
                }

                Add(outLocal[i * ub_row_stride], outLocal[i * ub_row_stride], tmpRes, vec_calc_len);
                Sub(outLocal[j * ub_row_stride], outLocal[j * ub_row_stride], tmpRes, vec_calc_len);
            }
        }
        
        if (total_pairs > 0) scalarQueue.FreeTensor(scalars);

        // [关键修复] 计算完成后，必须同步才能进行搬出 (Vector -> MTE2)
        PipeBarrier<PIPE_ALL>();

        // 3. 搬运 Output 回 GM
        for (uint32_t i = 0; i < n; i++)
        {
            uint64_t global_offset = (uint64_t)i * total_m + global_col_offset;
            DataCopy(outGm[global_offset], outLocal[i * ub_row_stride], tile_width);
        }

        inQueue.FreeTensor(inputLocal);
        outQueue.FreeTensor(outLocal);
        tmpQueue.FreeTensor(tmpDiff);
        tmpQueue.FreeTensor(tmpRes);
    }

    GlobalTensor<float> gradGm;
    GlobalTensor<float> inputGm;
    GlobalTensor<float> pdistGm;
    GlobalTensor<float> outGm;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TQue<QuePosition::VECIN, 2> tmpQueue;
    TQue<QuePosition::VECIN, 1> scalarQueue;

    uint32_t n;
    uint32_t total_m;
    float p;
    uint32_t m_per_core;
    uint32_t m_start;
    uint32_t m_len;
};

extern "C" __global__ __aicore__ void pdist_grad(GM_ADDR grad, GM_ADDR input, GM_ADDR pdist, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    KernelPdistGrad op;
    op.Init(grad, input, pdist, out, tiling);
    op.Process();
}