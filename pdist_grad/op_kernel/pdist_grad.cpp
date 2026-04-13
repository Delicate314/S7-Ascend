#include "kernel_operator.h"

using namespace AscendC;

class KernelPdistGrad
{
public:
    __aicore__ inline void Init(GM_ADDR gradAddr, GM_ADDR inputAddr, GM_ADDR pdistAddr, GM_ADDR outAddr, GM_ADDR tiling, TPipe* pipeIn)
    {
        GET_TILING_DATA(tiling_data, tiling);
        n = tiling_data.n;
        m = tiling_data.m;
        p = tiling_data.p;
        tile_m_ub = tiling_data.tile_m_ub;

        pipe = pipeIn;

        uint32_t total_n = n;
        uint32_t core_num = GetBlockNum();
        uint32_t core_idx = GetBlockIdx();

        uint32_t avg_lines = total_n / core_num;
        uint32_t tail_lines = total_n % core_num;

        if (core_idx < tail_lines) {
            start_n = core_idx * (avg_lines + 1);
            num_n = avg_lines + 1;
        } else {
            start_n = tail_lines * (avg_lines + 1) + (core_idx - tail_lines) * avg_lines;
            num_n = avg_lines;
        }

        gradGm.SetGlobalBuffer((__gm__ float *)gradAddr);
        inputGm.SetGlobalBuffer((__gm__ float *)inputAddr);
        pdistGm.SetGlobalBuffer((__gm__ float *)pdistAddr);
        outGm.SetGlobalBuffer((__gm__ float *)outAddr);

        // 计算对齐后的长度
        uint32_t m_align = (tile_m_ub + 7) / 8 * 8;
        uint32_t n_align = (n + 7) / 8 * 8;

        // tmpA 需要同时支持 m 维度的 vec_i 和 n 维度 ReduceSum 的 workTensor
        // 取两者的最大值进行分配
        uint32_t max_align = (m_align > n_align) ? m_align : n_align;
        uint32_t tmpA_size = max_align * sizeof(float);
        
        // input/output 只需要支持 m 维度
        uint32_t io_size = m_align * sizeof(float);

        pipe->InitBuffer(input, 2, io_size);
        pipe->InitBuffer(output, 2, io_size);
        pipe->InitBuffer(tmpA, 1, tmpA_size);

        // scale_size 必须能容纳 n 个 float
        uint32_t scale_size = n_align * sizeof(float);
        if (scale_size < 32) scale_size = 32;
        scale_size += 32; // padding for safety
        pipe->InitBuffer(tmpB, 1, scale_size);
    }

    __aicore__ inline void Process()
    {
        if (num_n == 0) return;

        if (p == 1.0f) {
            for (uint32_t i = start_n; i < start_n + num_n; i++) ComputeRowP1(i);
        } else if (p == 2.0f) {
            for (uint32_t i = start_n; i < start_n + num_n; i++) ComputeRowP2(i);
        } else if (p == 3.0f) {
            for (uint32_t i = start_n; i < start_n + num_n; i++) ComputeRowP3(i);
        } else {
            for (uint32_t i = start_n; i < start_n + num_n; i++) ComputeRowPGeneral(i);
        }
    }

private:
    __aicore__ inline void CalcFullTileParams(uint32_t tile_len, 
                                              uint32_t& align_len, 
                                              DataCopyExtParams& copyParams, 
                                              DataCopyPadExtParams<float>& padParams)
    {
        uint32_t blockLen = tile_len * sizeof(float);
        uint32_t tail = blockLen % 32;
        uint8_t right_pad = (tail == 0) ? 0 : (32 - tail) / sizeof(float);
        align_len = (tile_len + 7) / 8 * 8;

        copyParams.blockCount = 1;
        copyParams.blockLen = blockLen;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = right_pad;
        padParams.paddingValue = 0.0f;
    }

    // === P = 1 保持不变 ===
    __aicore__ inline void ComputeRowP1(uint32_t i)
    {
        LocalTensor<float> scalesLocal = tmpB.AllocTensor<float>();
        
        for (uint32_t j = 0; j <= i; j++) {
            if (i == j) {
                scalesLocal.SetValue(j, 0.0f);
            } else {
                uint32_t r = j; uint32_t c = i;
                uint64_t pair_idx = (uint64_t)n * r - (uint64_t)r * (r + 1) / 2 + c - r - 1;
                scalesLocal.SetValue(j, gradGm.GetValue(pair_idx));
            }
        }
        uint32_t start_j = i + 1;
        if (start_j < n) {
            uint32_t next_aligned_j = (start_j + 7) / 8 * 8;
            if (next_aligned_j > n) next_aligned_j = n;
            for (uint32_t j = start_j; j < next_aligned_j; j++) {
                uint64_t pair_idx = (uint64_t)n * i - (uint64_t)i * (i + 1) / 2 + j - i - 1;
                scalesLocal.SetValue(j, gradGm.GetValue(pair_idx));
            }
            uint32_t remain_vec_len = n - next_aligned_j;
            if (remain_vec_len > 0) {
                uint64_t vec_start_pair_idx = (uint64_t)n * i - (uint64_t)i * (i + 1) / 2 + next_aligned_j - i - 1;
                DataCopyExtParams copyParams{1, (uint32_t)(remain_vec_len * sizeof(float)), 0, 0, 0};
                uint32_t tail = (remain_vec_len * sizeof(float)) % 32;
                uint8_t right_pad = (tail == 0) ? 0 : (32 - tail) / sizeof(float);
                DataCopyPadExtParams<float> padParams{true, 0, right_pad, 0.0f};
                DataCopyPad(scalesLocal[next_aligned_j], gradGm[vec_start_pair_idx], copyParams, padParams);
            }
        }

        uint32_t full_align_len;
        DataCopyExtParams fullCopyParams;
        DataCopyPadExtParams<float> fullPadParams;
        CalcFullTileParams(tile_m_ub, full_align_len, fullCopyParams, fullPadParams);

        for (uint32_t m_offset = 0; m_offset < m; m_offset += tile_m_ub)
        {
            uint32_t cur_len = tile_m_ub;
            bool is_tail = (m_offset + cur_len > m); 
            uint32_t align_len = is_tail ? (m - m_offset + 7) / 8 * 8 : full_align_len;
            DataCopyExtParams copyParams = is_tail ? DataCopyExtParams{1, (uint32_t)((m - m_offset) * sizeof(float)), 0, 0, 0} : fullCopyParams;
            uint32_t tail = is_tail ? ((m - m_offset) * sizeof(float)) % 32 : 0;
            DataCopyPadExtParams<float> padParams = is_tail ? DataCopyPadExtParams<float>{true, 0, (uint8_t)((tail == 0) ? 0 : (32 - tail) / sizeof(float)), 0.0f} : fullPadParams;

            LocalTensor<float> vec_acc = output.AllocTensor<float>();
            Duplicate(vec_acc, 0.0f, align_len);
            LocalTensor<float> vec_i = tmpA.AllocTensor<float>();
            DataCopyPad(vec_i, inputGm[i * m + m_offset], copyParams, padParams);
            tmpA.EnQue(vec_i); vec_i = tmpA.DeQue<float>();

            for (uint32_t j = 0; j < n; j++) {
                if (i == j) continue;
                float scale = scalesLocal.GetValue(j);
                if (scale == 0.0f) continue;
                LocalTensor<float> vec_j = input.AllocTensor<float>();
                DataCopyPad(vec_j, inputGm[j * m + m_offset], copyParams, padParams);
                input.EnQue(vec_j); vec_j = input.DeQue<float>();
                LocalTensor<float> vec_tmp = output.AllocTensor<float>();
                Sub(vec_tmp, vec_i, vec_j, align_len);
                LocalTensor<float> vec_abs = vec_j; 
                Abs(vec_abs, vec_tmp, align_len);
                Div(vec_tmp, vec_tmp, vec_abs, align_len);
                Axpy(vec_acc, vec_tmp, scale, align_len); 
                output.FreeTensor(vec_tmp); input.FreeTensor(vec_j);
            }
            tmpA.FreeTensor(vec_i);
            output.EnQue(vec_acc); vec_acc = output.DeQue<float>();
            DataCopyPad(outGm[i * m + m_offset], vec_acc, copyParams);
            output.FreeTensor(vec_acc);
        }
        tmpB.FreeTensor(scalesLocal);
    }

    // === P = 2 优化路径 (代数化简) ===
    __aicore__ inline void ComputeRowP2(uint32_t i)
    {
        LocalTensor<float> scalesLocal = tmpB.AllocTensor<float>();
        
        // 1. 计算 Scale: S_ij = grad / dist
        for (uint32_t j = 0; j <= i; j++) {
            if (i == j) { scalesLocal.SetValue(j, 0.0f); continue; }
            uint32_t r = j; uint32_t c = i;
            uint64_t pair_idx = (uint64_t)n * r - (uint64_t)r * (r + 1) / 2 + c - r - 1;
            float grad_val = gradGm.GetValue(pair_idx);
            float pdist_val = pdistGm.GetValue(pair_idx);
            scalesLocal.SetValue(j, grad_val / pdist_val);
        }

        uint32_t start_j = i + 1;
        if (start_j < n) {
            uint32_t next_aligned_j = (start_j + 7) / 8 * 8;
            if (next_aligned_j > n) next_aligned_j = n;
            for (uint32_t j = start_j; j < next_aligned_j; j++) {
                uint64_t pair_idx = (uint64_t)n * i - (uint64_t)i * (i + 1) / 2 + j - i - 1;
                scalesLocal.SetValue(j, gradGm.GetValue(pair_idx) / pdistGm.GetValue(pair_idx));
            }
            uint32_t max_align_len = tile_m_ub / 8 * 8;
            if (max_align_len == 0) max_align_len = 8;
            uint32_t vec_j = next_aligned_j;
            while (vec_j < n) {
                uint32_t len = n - vec_j;
                if (len > max_align_len) len = max_align_len;
                uint32_t align_len = (len + 7) / 8 * 8;
                uint64_t vec_start_pair_idx = (uint64_t)n * i - (uint64_t)i * (i + 1) / 2 + vec_j - i - 1;
                uint32_t blockLen = len * sizeof(float);
                uint32_t tail = blockLen % 32;
                uint8_t right_pad = (tail == 0) ? 0 : (32 - tail) / sizeof(float);
                DataCopyExtParams copyParams{1, (uint32_t)blockLen, 0, 0, 0};
                DataCopyPadExtParams<float> padParamsGrad{true, 0, right_pad, 0.0f};
                DataCopyPadExtParams<float> padParamsPdist{true, 0, right_pad, 1.0f};

                LocalTensor<float> vec_grad = input.AllocTensor<float>();
                LocalTensor<float> vec_pdist = output.AllocTensor<float>();
                DataCopyPad(vec_grad, gradGm[vec_start_pair_idx], copyParams, padParamsGrad);
                DataCopyPad(vec_pdist, pdistGm[vec_start_pair_idx], copyParams, padParamsPdist);
                input.EnQue(vec_grad); output.EnQue(vec_pdist);
                vec_grad = input.DeQue<float>(); vec_pdist = output.DeQue<float>();
                Div(scalesLocal[vec_j], vec_grad, vec_pdist, align_len);
                input.FreeTensor(vec_grad); output.FreeTensor(vec_pdist);
                vec_j += len;
            }
        }

        // 2. 优化：计算 Scale 的总和 sum(S_ij)
        // 使用 ReduceSum 代替 Sum
        // tmpA 在 Init 中已确保足够大 (>= n * sizeof(float))
        LocalTensor<float> workTensor = tmpA.AllocTensor<float>();
        LocalTensor<float> sumDst = output.AllocTensor<float>(); 
        
        // 对 scalesLocal 进行规约求和
        ReduceSum(sumDst, scalesLocal, workTensor, n); 
        float total_scale_sum = sumDst.GetValue(0);
        
        output.FreeTensor(sumDst);
        tmpA.FreeTensor(workTensor);

        // 3. 计算行
        uint32_t full_align_len;
        DataCopyExtParams fullCopyParams;
        DataCopyPadExtParams<float> fullPadParams;
        CalcFullTileParams(tile_m_ub, full_align_len, fullCopyParams, fullPadParams);

        for (uint32_t m_offset = 0; m_offset < m; m_offset += tile_m_ub)
        {
            uint32_t cur_len = tile_m_ub;
            bool is_tail = (m_offset + cur_len > m);

            uint32_t align_len;
            DataCopyExtParams copyParams;
            DataCopyPadExtParams<float> padParams;

            if (!is_tail) {
                align_len = full_align_len;
                copyParams = fullCopyParams;
                padParams = fullPadParams;
            } else {
                cur_len = m - m_offset;
                uint32_t blockLen = cur_len * sizeof(float);
                uint32_t tail = blockLen % 32;
                uint8_t right_pad = (tail == 0) ? 0 : (32 - tail) / sizeof(float);
                align_len = (cur_len + 7) / 8 * 8; 
                copyParams = {1, (uint32_t)blockLen, 0, 0, 0};
                padParams = {true, 0, right_pad, 0.0f};
            }

            LocalTensor<float> vec_acc = output.AllocTensor<float>();
            Duplicate(vec_acc, 0.0f, align_len);

            // 复用 tmpA 用于 vec_i (此时 workTensor 已释放)
            LocalTensor<float> vec_i = tmpA.AllocTensor<float>();
            DataCopyPad(vec_i, inputGm[i * m + m_offset], copyParams, padParams);
            tmpA.EnQue(vec_i);
            vec_i = tmpA.DeQue<float>();

            for (uint32_t j = 0; j < n; j++) {
                if (i == j) continue;
                float scale = scalesLocal.GetValue(j);
                if (scale == 0.0f) continue;

                LocalTensor<float> vec_j = input.AllocTensor<float>();
                DataCopyPad(vec_j, inputGm[j * m + m_offset], copyParams, padParams);
                input.EnQue(vec_j); 
                vec_j = input.DeQue<float>();

                // Fused Op: acc = acc - vec_j * scale
                Axpy(vec_acc, vec_j, -scale, align_len);
                input.FreeTensor(vec_j);
            }

            // 最后加上 vec_i * sum(S)
            Axpy(vec_acc, vec_i, total_scale_sum, align_len);

            tmpA.FreeTensor(vec_i);
            output.EnQue(vec_acc);
            vec_acc = output.DeQue<float>();
            DataCopyPad(outGm[i * m + m_offset], vec_acc, copyParams);
            output.FreeTensor(vec_acc);
        }

        tmpB.FreeTensor(scalesLocal);
    }

    // === P = 3 保持不变 ===
    __aicore__ inline void ComputeRowP3(uint32_t i)
    {
        LocalTensor<float> scalesLocal = tmpB.AllocTensor<float>();
        
        for (uint32_t j = 0; j <= i; j++) {
            if (i == j) { scalesLocal.SetValue(j, 0.0f); continue; }
            uint32_t r = j; uint32_t c = i;
            uint64_t pair_idx = (uint64_t)n * r - (uint64_t)r * (r + 1) / 2 + c - r - 1;
            float grad_val = gradGm.GetValue(pair_idx);
            float pdist_val = pdistGm.GetValue(pair_idx);
            scalesLocal.SetValue(j, grad_val / (pdist_val * pdist_val));
        }
        uint32_t start_j = i + 1;
        if (start_j < n) {
            uint32_t next_aligned_j = (start_j + 7) / 8 * 8;
            if (next_aligned_j > n) next_aligned_j = n;
            for (uint32_t j = start_j; j < next_aligned_j; j++) {
                uint64_t pair_idx = (uint64_t)n * i - (uint64_t)i * (i + 1) / 2 + j - i - 1;
                scalesLocal.SetValue(j, gradGm.GetValue(pair_idx) / (pdistGm.GetValue(pair_idx) * pdistGm.GetValue(pair_idx)));
            }
            uint32_t max_align_len = tile_m_ub / 8 * 8;
            if (max_align_len == 0) max_align_len = 8;
            uint32_t vec_j = next_aligned_j;
            while (vec_j < n) {
                uint32_t len = n - vec_j;
                if (len > max_align_len) len = max_align_len;
                uint32_t align_len = (len + 7) / 8 * 8;
                uint64_t vec_start_pair_idx = (uint64_t)n * i - (uint64_t)i * (i + 1) / 2 + vec_j - i - 1;
                DataCopyExtParams copyParams{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
                uint32_t tail = (len * sizeof(float)) % 32;
                uint8_t right_pad = (tail == 0) ? 0 : (32 - tail) / sizeof(float);
                DataCopyPadExtParams<float> padParamsGrad{true, 0, right_pad, 0.0f};
                DataCopyPadExtParams<float> padParamsPdist{true, 0, right_pad, 1.0f};

                LocalTensor<float> vec_grad = input.AllocTensor<float>();
                LocalTensor<float> vec_pdist_sq = tmpA.AllocTensor<float>();
                DataCopyPad(vec_grad, gradGm[vec_start_pair_idx], copyParams, padParamsGrad);
                DataCopyPad(vec_pdist_sq, pdistGm[vec_start_pair_idx], copyParams, padParamsPdist);
                input.EnQue(vec_grad); tmpA.EnQue(vec_pdist_sq);
                vec_grad = input.DeQue<float>(); vec_pdist_sq = tmpA.DeQue<float>();
                Mul(vec_pdist_sq, vec_pdist_sq, vec_pdist_sq, align_len);
                Div(scalesLocal[vec_j], vec_grad, vec_pdist_sq, align_len);
                input.FreeTensor(vec_grad); tmpA.FreeTensor(vec_pdist_sq);
                vec_j += len;
            }
        }
        
        uint32_t full_align_len;
        DataCopyExtParams fullCopyParams;
        DataCopyPadExtParams<float> fullPadParams;
        CalcFullTileParams(tile_m_ub, full_align_len, fullCopyParams, fullPadParams);

        for (uint32_t m_offset = 0; m_offset < m; m_offset += tile_m_ub)
        {
            uint32_t cur_len = tile_m_ub;
            bool is_tail = (m_offset + cur_len > m);
            uint32_t align_len = is_tail ? (m - m_offset + 7) / 8 * 8 : full_align_len;
            DataCopyExtParams copyParams = is_tail ? DataCopyExtParams{1, (uint32_t)((m - m_offset) * sizeof(float)), 0, 0, 0} : fullCopyParams;
            uint32_t tail = is_tail ? ((m - m_offset) * sizeof(float)) % 32 : 0;
            DataCopyPadExtParams<float> padParams = is_tail ? DataCopyPadExtParams<float>{true, 0, (uint8_t)((tail == 0) ? 0 : (32 - tail) / sizeof(float)), 0.0f} : fullPadParams;

            LocalTensor<float> vec_acc = output.AllocTensor<float>();
            Duplicate(vec_acc, 0.0f, align_len);
            LocalTensor<float> vec_i = tmpA.AllocTensor<float>();
            DataCopyPad(vec_i, inputGm[i * m + m_offset], copyParams, padParams);
            tmpA.EnQue(vec_i); vec_i = tmpA.DeQue<float>();

            for (uint32_t j = 0; j < n; j++) {
                if (i == j) continue;
                float scale = scalesLocal.GetValue(j);
                if (scale == 0.0f) continue;
                LocalTensor<float> vec_j = input.AllocTensor<float>();
                DataCopyPad(vec_j, inputGm[j * m + m_offset], copyParams, padParams);
                input.EnQue(vec_j); vec_j = input.DeQue<float>();
                LocalTensor<float> vec_tmp = output.AllocTensor<float>();
                Sub(vec_tmp, vec_i, vec_j, align_len);
                LocalTensor<float> vec_abs = vec_j; 
                Abs(vec_abs, vec_tmp, align_len);
                Mul(vec_tmp, vec_tmp, vec_abs, align_len);
                Axpy(vec_acc, vec_tmp, scale, align_len);
                output.FreeTensor(vec_tmp); input.FreeTensor(vec_j);
            }
            tmpA.FreeTensor(vec_i);
            output.EnQue(vec_acc); vec_acc = output.DeQue<float>();
            DataCopyPad(outGm[i * m + m_offset], vec_acc, copyParams);
            output.FreeTensor(vec_acc);
        }
        tmpB.FreeTensor(scalesLocal);
    }

    // === P = General 保持不变 ===
    __aicore__ inline void ComputeRowPGeneral(uint32_t i)
    {
        LocalTensor<float> scalesLocal = tmpB.AllocTensor<float>();
        LocalTensor<float> tmpBuf = output.AllocTensor<float>();
        float p_minus_1 = p - 1.0f;
        for (uint32_t j = 0; j < n; j++) {
            if (i == j) { scalesLocal.SetValue(j, 0.0f); continue; }
            uint32_t r = (i < j) ? i : j; uint32_t c = (i < j) ? j : i;
            uint64_t pair_idx = (uint64_t)n * r - (uint64_t)r * (r + 1) / 2 + c - r - 1;
            float grad_val = gradGm.GetValue(pair_idx);
            float pdist_val = pdistGm.GetValue(pair_idx);
            Duplicate(tmpBuf, pdist_val, 8); 
            Ln(tmpBuf, tmpBuf, 8); Muls(tmpBuf, tmpBuf, p_minus_1, 8); Exp(tmpBuf, tmpBuf, 8);
            float dist_pow = tmpBuf.GetValue(0);
            scalesLocal.SetValue(j, grad_val / dist_pow);
        }
        output.FreeTensor(tmpBuf);
        uint32_t full_align_len;
        DataCopyExtParams fullCopyParams;
        DataCopyPadExtParams<float> fullPadParams;
        CalcFullTileParams(tile_m_ub, full_align_len, fullCopyParams, fullPadParams);
        float p_minus_2 = p - 2.0f;
        for (uint32_t m_offset = 0; m_offset < m; m_offset += tile_m_ub) {
            uint32_t cur_len = tile_m_ub;
            bool is_tail = (m_offset + cur_len > m);
            uint32_t align_len = is_tail ? (m - m_offset + 7) / 8 * 8 : full_align_len;
            DataCopyExtParams copyParams = is_tail ? DataCopyExtParams{1, (uint32_t)((m - m_offset) * sizeof(float)), 0, 0, 0} : fullCopyParams;
            uint32_t tail = is_tail ? ((m - m_offset) * sizeof(float)) % 32 : 0;
            DataCopyPadExtParams<float> padParams = is_tail ? DataCopyPadExtParams<float>{true, 0, (uint8_t)((tail == 0) ? 0 : (32 - tail) / sizeof(float)), 0.0f} : fullPadParams;

            LocalTensor<float> vec_acc = output.AllocTensor<float>();
            Duplicate(vec_acc, 0.0f, align_len);
            LocalTensor<float> vec_i = tmpA.AllocTensor<float>();
            DataCopyPad(vec_i, inputGm[i * m + m_offset], copyParams, padParams);
            tmpA.EnQue(vec_i); vec_i = tmpA.DeQue<float>();
            for (uint32_t j = 0; j < n; j++) {
                if (i == j) continue;
                float scale = scalesLocal.GetValue(j);
                if (scale == 0.0f) continue;
                LocalTensor<float> vec_j = input.AllocTensor<float>();
                DataCopyPad(vec_j, inputGm[j * m + m_offset], copyParams, padParams);
                input.EnQue(vec_j); vec_j = input.DeQue<float>();
                LocalTensor<float> vec_tmp = output.AllocTensor<float>();
                Sub(vec_tmp, vec_i, vec_j, align_len);
                LocalTensor<float> vec_term = vec_j; 
                Abs(vec_term, vec_tmp, align_len);
                Ln(vec_term, vec_term, align_len); Muls(vec_term, vec_term, p_minus_2, align_len); Exp(vec_term, vec_term, align_len);
                Mul(vec_tmp, vec_tmp, vec_term, align_len);
                Axpy(vec_acc, vec_tmp, scale, align_len);
                output.FreeTensor(vec_tmp); input.FreeTensor(vec_j);
            }
            tmpA.FreeTensor(vec_i);
            output.EnQue(vec_acc); vec_acc = output.DeQue<float>();
            DataCopyPad(outGm[i * m + m_offset], vec_acc, copyParams);
            output.FreeTensor(vec_acc);
        }
        tmpB.FreeTensor(scalesLocal);
    }

    GlobalTensor<float> gradGm;
    GlobalTensor<float> inputGm;
    GlobalTensor<float> pdistGm;
    GlobalTensor<float> outGm;

    TPipe* pipe;
    TQue<QuePosition::VECIN, 2> input;
    TQue<QuePosition::VECOUT, 2> output;
    TQue<QuePosition::VECIN, 1> tmpA;
    TQue<QuePosition::VECIN, 1> tmpB;

    uint32_t n;
    uint32_t m;
    float p;
    uint32_t start_n;
    uint32_t num_n;
    uint32_t tile_m_ub;
};

extern "C" __global__ __aicore__ void pdist_grad(GM_ADDR grad, GM_ADDR input, GM_ADDR pdist, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    KernelPdistGrad op;
    op.Init(grad, input, pdist, out, tiling, &pipe);
    op.Process();
}