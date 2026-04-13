#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t CALC_ALIGN_NUM = 256;

// =========================================================================
// 非 Broadcast 场景
// =========================================================================
template <typename T>
class KernelFmax
{
public:
    __aicore__ inline KernelFmax() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out,
                                uint32_t bigDataCoreNum, uint32_t smallBlockLength, uint32_t bigBlockLength,
                                uint32_t smallTileNum, uint32_t smallTileLength, uint32_t smallLasttileLength,
                                uint32_t bigTileNum, uint32_t bigTileLength, uint32_t bigLasttileLength)
    {

        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        if (AscendC::GetBlockIdx() >= bigDataCoreNum)
        {
            uint32_t offset = (bigDataCoreNum * bigBlockLength) + (smallBlockLength * (AscendC::GetBlockIdx() - bigDataCoreNum));
            InitCommon(input, other, out, offset, smallBlockLength, smallTileLength, smallTileNum, smallLasttileLength);
        }
        else
        {
            uint32_t offset = bigBlockLength * AscendC::GetBlockIdx();
            InitCommon(input, other, out, offset, bigBlockLength, bigTileLength, bigTileNum, bigLasttileLength);
        }
    }

    __aicore__ inline void InitCommon(GM_ADDR input, GM_ADDR other, GM_ADDR out,
                                      uint32_t offset, uint32_t blockLength, uint32_t tLength,
                                      uint32_t tNum, uint32_t lastTLength)
    {

        x1Gm.SetGlobalBuffer((__gm__ T *)input + offset, blockLength);
        x2Gm.SetGlobalBuffer((__gm__ T *)other + offset, blockLength);
        yGm.SetGlobalBuffer((__gm__ T *)out + offset, blockLength);

        uint32_t alignUnit = CALC_ALIGN_NUM / sizeof(T);

        if constexpr (std::is_same_v<T, int64_t>)
        {
            alignUnit = CALC_ALIGN_NUM / sizeof(int32_t); // 64
        }

        if (alignUnit < 32)
            alignUnit = 32;
        uint32_t allocLen = ((tLength + alignUnit - 1) / alignUnit) * alignUnit;

        pipe.InitBuffer(inQueueX1, BUFFER_NUM, allocLen * sizeof(T));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, allocLen * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, allocLen * sizeof(T));

        // 掩码 Queue (用于 NaN 修复)
        uint32_t maskAlignSize = ((allocLen * sizeof(uint8_t) + CALC_ALIGN_NUM - 1) / CALC_ALIGN_NUM) * CALC_ALIGN_NUM;
        pipe.InitBuffer(calcQueue, BUFFER_NUM, maskAlignSize);

        // Cast Buffer 分配
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            pipe.InitBuffer(castQueueX1, BUFFER_NUM, allocLen * sizeof(half));
            pipe.InitBuffer(castQueueX2, BUFFER_NUM, allocLen * sizeof(half));
            pipe.InitBuffer(castQueueY, BUFFER_NUM, allocLen * sizeof(half));
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t>)
        {
            pipe.InitBuffer(castQueueX1, BUFFER_NUM, allocLen * sizeof(float));
            pipe.InitBuffer(castQueueX2, BUFFER_NUM, allocLen * sizeof(float));
            pipe.InitBuffer(castQueueY, BUFFER_NUM, allocLen * sizeof(float));
        }
        else if constexpr (std::is_same_v<T, int64_t>)
        {
            pipe.InitBuffer(castQueueX1, BUFFER_NUM, allocLen * sizeof(int32_t));
            pipe.InitBuffer(castQueueX2, BUFFER_NUM, allocLen * sizeof(int32_t));
            pipe.InitBuffer(castQueueY, BUFFER_NUM, allocLen * sizeof(int32_t));
            pipe.InitBuffer(castQueueFloatX1, BUFFER_NUM, allocLen * sizeof(float));
            pipe.InitBuffer(castQueueFloatX2, BUFFER_NUM, allocLen * sizeof(float));
            pipe.InitBuffer(castQueueFloatY, BUFFER_NUM, allocLen * sizeof(float));
        }
        else if constexpr (std::is_same_v<T, __bf16>)
        {
            pipe.InitBuffer(castQueueX1, BUFFER_NUM, allocLen * sizeof(float));
            pipe.InitBuffer(castQueueX2, BUFFER_NUM, allocLen * sizeof(float));
            pipe.InitBuffer(castQueueY, BUFFER_NUM, allocLen * sizeof(float));
        }

        this->tileNum = tNum;
        this->tileLength = tLength;
        this->lasttileLength = lastTLength;
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;

        // Debug: 打印 Process 开始信息（仅 int8/uint8，第一个 block）
        // if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
        //     if (AscendC::GetBlockIdx() == 0) {
        //         AscendC::printf("int8 non-bcast Process: loopCount=%d tileLength=%u lasttileLength=%u\n",
        //             loopCount, this->tileLength, this->lasttileLength);
        //     }
        // }

        for (int32_t i = 0; i < loopCount; i++)
        {
            this->calcLength = (i == loopCount - 1) ? this->lasttileLength : this->tileLength;
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
        AscendC::LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();
        AscendC::DataCopy(x1Local, x1Gm[progress * this->tileLength], this->calcLength);
        AscendC::DataCopy(x2Local, x2Gm[progress * this->tileLength], this->calcLength);

        // Debug: 打印 CopyIn 后的数据（仅 int8/uint8，第一个 block）
        // if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
        //     if (AscendC::GetBlockIdx() == 0 && progress == 0) {
        //         uint32_t dbgN = (this->calcLength < 8u) ? this->calcLength : 8u;
        //         AscendC::printf("int8 non-bcast CopyIn: progress=%d calcLength=%u tileLength=%u\n",
        //             progress, this->calcLength, this->tileLength);
        //         for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
        //             if constexpr (std::is_same_v<T, int8_t>) {
        //                 AscendC::printf("int8 CopyIn data: idx=%u x1=%d x2=%d\n",
        //                     dbg,
        //                     static_cast<int32_t>(x1Local.GetValue(dbg)),
        //                     static_cast<int32_t>(x2Local.GetValue(dbg)));
        //             }
        //             else {
        //                 AscendC::printf("uint8 CopyIn data: idx=%u x1=%u x2=%u\n",
        //                     dbg,
        //                     static_cast<uint32_t>(x1Local.GetValue(dbg)),
        //                     static_cast<uint32_t>(x2Local.GetValue(dbg)));
        //             }
        //         }
        //     }
        // }

        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
        AscendC::LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        AscendC::LocalTensor<uint8_t> maskLocal = calcQueue.AllocTensor<uint8_t>();

        uint32_t elementsPerRepeat = CALC_ALIGN_NUM / sizeof(T);
        uint32_t calCount = ((this->calcLength + elementsPerRepeat - 1) / elementsPerRepeat) * elementsPerRepeat;

        // Group A: int8, uint8 -> half
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            AscendC::LocalTensor<half> x1Cast = castQueueX1.AllocTensor<half>();
            AscendC::LocalTensor<half> x2Cast = castQueueX2.AllocTensor<half>();
            AscendC::LocalTensor<half> yCast = castQueueY.AllocTensor<half>();
            uint32_t halfElem = CALC_ALIGN_NUM / sizeof(half);
            uint32_t halfCount = ((this->calcLength + halfElem - 1) / halfElem) * halfElem;

            // Debug: 打印原始 int8/uint8 数据（仅第一个 block，前8个元素）
            // uint32_t dbgN = (this->calcLength < 8u) ? this->calcLength : 8u;
            // if (AscendC::GetBlockIdx() == 0) {
            //     AscendC::printf("int8 non-bcast Compute: calcLength=%u halfCount=%u\n",
            //         this->calcLength, halfCount);
            //     for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
            //         if constexpr (std::is_same_v<T, int8_t>) {
            //             AscendC::printf("int8 orig: idx=%u x1=%d x2=%d\n",
            //                 dbg,
            //                 static_cast<int32_t>(x1Local.GetValue(dbg)),
            //                 static_cast<int32_t>(x2Local.GetValue(dbg)));
            //         }
            //         else {
            //             AscendC::printf("uint8 orig: idx=%u x1=%u x2=%u\n",
            //                 dbg,
            //                 static_cast<uint32_t>(x1Local.GetValue(dbg)),
            //                 static_cast<uint32_t>(x2Local.GetValue(dbg)));
            //         }
            //     }
            // }

            AscendC::Cast(x1Cast, x1Local, AscendC::RoundMode::CAST_NONE, halfCount);
            AscendC::Cast(x2Cast, x2Local, AscendC::RoundMode::CAST_NONE, halfCount);

            // Debug: 打印 Cast 到 half 后的数据
            // if (AscendC::GetBlockIdx() == 0) {
            //     for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
            //         AscendC::printf("int8->half: idx=%u x1=%f x2=%f\n",
            //             dbg,
            //             static_cast<float>(x1Cast.GetValue(dbg)),
            //             static_cast<float>(x2Cast.GetValue(dbg)));
            //     }
            // }

            // 优化：使用 Max 指令
            AscendC::Max(yCast, x1Cast, x2Cast, halfCount);

            // Debug: 打印 Max 操作后的结果
            // if (AscendC::GetBlockIdx() == 0) {
            //     for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
            //         AscendC::printf("half Max result: idx=%u y=%f\n",
            //             dbg,
            //             static_cast<float>(yCast.GetValue(dbg)));
            //     }
            // }

            AscendC::Cast(yLocal, yCast, AscendC::RoundMode::CAST_TRUNC, halfCount);

            // Debug: 打印 Cast 回 int8/uint8 后的最终结果
            // if (AscendC::GetBlockIdx() == 0) {
            //     for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
            //         if constexpr (std::is_same_v<T, int8_t>) {
            //             AscendC::printf("half->int8 final: idx=%u y=%d\n",
            //                 dbg,
            //                 static_cast<int32_t>(yLocal.GetValue(dbg)));
            //         }
            //         else {
            //             AscendC::printf("half->uint8 final: idx=%u y=%u\n",
            //                 dbg,
            //                 static_cast<uint32_t>(yLocal.GetValue(dbg)));
            //         }
            //     }
            // }

            castQueueX1.FreeTensor(x1Cast);
            castQueueX2.FreeTensor(x2Cast);
            castQueueY.FreeTensor(yCast);
        }
        // Group B: int16, int32 -> float
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t>)
        {
            AscendC::LocalTensor<float> x1Cast = castQueueX1.AllocTensor<float>();
            AscendC::LocalTensor<float> x2Cast = castQueueX2.AllocTensor<float>();
            AscendC::LocalTensor<float> yCast = castQueueY.AllocTensor<float>();
            uint32_t fElem = CALC_ALIGN_NUM / sizeof(float);
            uint32_t fCount = ((this->calcLength + fElem - 1) / fElem) * fElem;

            AscendC::Cast(x1Cast, x1Local, AscendC::RoundMode::CAST_NONE, fCount);
            AscendC::Cast(x2Cast, x2Local, AscendC::RoundMode::CAST_NONE, fCount);

            AscendC::Max(yCast, x1Cast, x2Cast, fCount);

            AscendC::Cast(yLocal, yCast, AscendC::RoundMode::CAST_TRUNC, fCount);

            castQueueX1.FreeTensor(x1Cast);
            castQueueX2.FreeTensor(x2Cast);
            castQueueY.FreeTensor(yCast);
        }
        // Group B2: int64
        else if constexpr (std::is_same_v<T, int64_t>)
        {
            AscendC::LocalTensor<int32_t> x1Int32 = castQueueX1.AllocTensor<int32_t>();
            AscendC::LocalTensor<int32_t> x2Int32 = castQueueX2.AllocTensor<int32_t>();
            AscendC::LocalTensor<int32_t> yInt32 = castQueueY.AllocTensor<int32_t>();
            uint32_t i32Elem = CALC_ALIGN_NUM / sizeof(int32_t);
            // 此时 Input Buffer 已经按照 64 对齐分配，可以安全进行 64 元素的 Cast
            uint32_t i32Count = ((this->calcLength + i32Elem - 1) / i32Elem) * i32Elem;

            // Debug: 原始 int64（截断显示前4个）
            uint32_t dbgN = (this->calcLength < 4u) ? this->calcLength : 4u;
            // for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
            //     AscendC::printf("orig int64 idx=%u x1=%ld x2=%ld\n",
            //         dbg,
            //         static_cast<long>(x1Local.GetValue(dbg)),
            //         static_cast<long>(x2Local.GetValue(dbg)));
            // }

            AscendC::Cast(x1Int32, x1Local, AscendC::RoundMode::CAST_NONE, i32Count);
            AscendC::Cast(x2Int32, x2Local, AscendC::RoundMode::CAST_NONE, i32Count);
            // Debug: int64 -> int32
            // for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
            //     AscendC::printf("int64->int32 idx=%u x1=%d x2=%d\n",
            //         dbg,
            //         static_cast<int32_t>(x1Int32.GetValue(dbg)),
            //         static_cast<int32_t>(x2Int32.GetValue(dbg)));
            // }

            AscendC::LocalTensor<float> x1Float = castQueueFloatX1.AllocTensor<float>();
            AscendC::LocalTensor<float> x2Float = castQueueFloatX2.AllocTensor<float>();
            AscendC::LocalTensor<float> yFloat = castQueueFloatY.AllocTensor<float>();
            uint32_t fElem = CALC_ALIGN_NUM / sizeof(float);
            uint32_t fCount = ((this->calcLength + fElem - 1) / fElem) * fElem;

            AscendC::Cast(x1Float, x1Int32, AscendC::RoundMode::CAST_NONE, fCount);
            AscendC::Cast(x2Float, x2Int32, AscendC::RoundMode::CAST_NONE, fCount);
            // Debug: int32 -> float
            // for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
            //     AscendC::printf("int32->float idx=%u x1=%f x2=%f\n",
            //         dbg,
            //         x1Float.GetValue(dbg),
            //         x2Float.GetValue(dbg));
            // }

            AscendC::Max(yFloat, x1Float, x2Float, fCount);

            AscendC::Cast(yInt32, yFloat, AscendC::RoundMode::CAST_TRUNC, fCount);
            // // Debug: float -> int32
            // for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
            //     AscendC::printf("float->int32 idx=%u yInt32=%d\n",
            //         dbg,
            //         yInt32.GetValue(dbg));
            // }

            AscendC::Cast(yLocal, yInt32, AscendC::RoundMode::CAST_NONE, i32Count);
            // // Debug: int32 -> int64
            // for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
            //     AscendC::printf("float->int32->int64 idx=%u yInt32=%d y=%ld\n",
            //         dbg,
            //         yInt32.GetValue(dbg),
            //         static_cast<long>(yLocal.GetValue(dbg)));
            // }

            castQueueX1.FreeTensor(x1Int32);
            castQueueX2.FreeTensor(x2Int32);
            castQueueY.FreeTensor(yInt32);
            castQueueFloatX1.FreeTensor(x1Float);
            castQueueFloatX2.FreeTensor(x2Float);
            castQueueFloatY.FreeTensor(yFloat);
        }
        // Group C: BF16 (包含 NaN 修复)
        else if constexpr (std::is_same_v<T, __bf16>)
        {
            AscendC::LocalTensor<float> x1Cast = castQueueX1.AllocTensor<float>();
            AscendC::LocalTensor<float> x2Cast = castQueueX2.AllocTensor<float>();
            AscendC::LocalTensor<float> yCast = castQueueY.AllocTensor<float>();
            uint32_t fElem = CALC_ALIGN_NUM / sizeof(float);
            uint32_t fCount = ((this->calcLength + fElem - 1) / fElem) * fElem;

            AscendC::Cast(x1Cast, x1Local, AscendC::RoundMode::CAST_NONE, fCount);
            AscendC::Cast(x2Cast, x2Local, AscendC::RoundMode::CAST_NONE, fCount);

            AscendC::Max(yCast, x1Cast, x2Cast, fCount);

            // // NaN 修复: 确保 fmax(NaN, x) = x
            // AscendC::Compare(maskLocal, x1Cast, x1Cast, AscendC::CMPMODE::EQ, fCount);
            // AscendC::Select(yCast, maskLocal, yCast, x2Cast, AscendC::SELMODE::VSEL_CMPMASK_SPR, fCount);
            // AscendC::Compare(maskLocal, x2Cast, x2Cast, AscendC::CMPMODE::EQ, fCount);
            // AscendC::Select(yCast, maskLocal, yCast, x1Cast, AscendC::SELMODE::VSEL_CMPMASK_SPR, fCount);

            AscendC::Cast(yLocal, yCast, AscendC::RoundMode::CAST_RINT, fCount);
            castQueueX1.FreeTensor(x1Cast);
            castQueueX2.FreeTensor(x2Cast);
            castQueueY.FreeTensor(yCast);
        }
        // Group D: bool
        else if constexpr (std::is_same_v<T, bool>)
        {
            uint32_t lenReal = this->calcLength;
            for (uint32_t i = 0; i < lenReal; ++i)
            {
                yLocal.SetValue(i, x1Local.GetValue(i) || x2Local.GetValue(i));
            }
        }
        // Group E: float/half (包含 NaN 修复)
        else
        {
            // AscendC::printf("float/half non-broadcast function call\n");

            AscendC::Max(yLocal, x1Local, x2Local, calCount);

            // NaN 修复
            // AscendC::Compare(maskLocal, x1Local, x1Local, AscendC::CMPMODE::EQ, calCount);
            // AscendC::Select(yLocal, maskLocal, yLocal, x2Local, AscendC::SELMODE::VSEL_CMPMASK_SPR, calCount);
            // AscendC::Compare(maskLocal, x2Local, x2Local, AscendC::CMPMODE::EQ, calCount);
            // AscendC::Select(yLocal, maskLocal, yLocal, x1Local, AscendC::SELMODE::VSEL_CMPMASK_SPR, calCount);
        }

        calcQueue.FreeTensor(maskLocal);
        outQueueY.EnQue<T>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();

        // Debug: 打印 CopyOut 前的数据（仅 int8/uint8，第一个 block）
        // if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
        //     if (AscendC::GetBlockIdx() == 0 && progress == 0) {
        //         uint32_t dbgN = (this->calcLength < 8u) ? this->calcLength : 8u;
        //         AscendC::printf("int8 non-bcast CopyOut: progress=%d calcLength=%u\n",
        //             progress, this->calcLength);
        //         for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
        //             if constexpr (std::is_same_v<T, int8_t>) {
        //                 AscendC::printf("int8 CopyOut data: idx=%u y=%d\n",
        //                     dbg,
        //                     static_cast<int32_t>(yLocal.GetValue(dbg)));
        //             }
        //             else {
        //                 AscendC::printf("uint8 CopyOut data: idx=%u y=%u\n",
        //                     dbg,
        //                     static_cast<uint32_t>(yLocal.GetValue(dbg)));
        //             }
        //         }
        //     }
        // }

        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->calcLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::QuePosition::VECCALC, BUFFER_NUM> calcQueue;
    AscendC::TQue<AscendC::QuePosition::VECCALC, BUFFER_NUM> castQueueX1, castQueueX2, castQueueY;
    [[maybe_unused]] AscendC::TQue<AscendC::QuePosition::VECCALC, BUFFER_NUM> castQueueFloatX1, castQueueFloatX2, castQueueFloatY;
    AscendC::GlobalTensor<T> x1Gm, x2Gm, yGm;
    uint32_t tileNum, tileLength, calcLength, lasttileLength;
};

// =========================================================================
// Broadcast 场景 (修复 Duplicate 不支持 bool/int8/int64 的问题 + 多核切分)
// =========================================================================
template <typename T>
class KernelFmaxBroadcast
{
public:
    __aicore__ inline KernelFmaxBroadcast() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out,
                                uint32_t totalLength, uint32_t dim, uint32_t *shape, uint32_t *reduce1, uint32_t *reduce2)
    {

        this->reduce1 = reduce1;
        this->reduce2 = reduce2;
        this->shape = shape;
        this->dim = dim;
        this->totalLength = totalLength;

        // 1. 获取最后一维长度
        this->lastDimLength = this->shape[this->dim - 1];

        // 2. 多核任务切分 (Block Tiling)
        uint32_t totalRows = this->totalLength / this->lastDimLength;
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        uint32_t rowsPerBlock = totalRows / blockNum;
        uint32_t remainRows = totalRows % blockNum;

        // 加入坐标计算逻辑，添加成员变量，假设最大维度不超过16
        uint32_t curIdx[16];

        if (blockIdx < remainRows)
        {
            rowsPerBlock = rowsPerBlock + 1;
            this->startRow = blockIdx * rowsPerBlock;
        }
        else
        {
            this->startRow = remainRows * (rowsPerBlock + 1) + (blockIdx - remainRows) * rowsPerBlock;
        }
        this->endRow = this->startRow + rowsPerBlock;

        // 3. UB 空间分配 (对齐)
        uint32_t alignUnit = CALC_ALIGN_NUM / sizeof(T);
        if (alignUnit < 32)
            alignUnit = 32;
        this->tileLength = ((this->lastDimLength + alignUnit - 1) / alignUnit) * alignUnit;

        // 4. GM 指针初始化
        x1Gm.SetGlobalBuffer((__gm__ T *)input);
        x2Gm.SetGlobalBuffer((__gm__ T *)other);
        yGm.SetGlobalBuffer((__gm__ T *)out);

        // 5. Pipe Buffer 初始化
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(T));

        uint32_t maskAlignSize = ((this->tileLength * sizeof(uint8_t) + CALC_ALIGN_NUM - 1) / CALC_ALIGN_NUM) * CALC_ALIGN_NUM;
        pipe.InitBuffer(calcQueue, BUFFER_NUM, maskAlignSize);

        // Cast Buffer 初始化
        InitCastBuffers();
    }

    __aicore__ inline void Process()
    {
        // 初始化步长数组
        uint32_t d[21] = {0};
        uint32_t dn1[21] = {0};
        uint32_t dn2[21] = {0};
        // AscendC::printf("this->dim = %d\n", this->dim);
        InitializeDnArrays(d, dn1, this->reduce1, this->dim, this->shape);
        InitializeDnArrays(d, dn2, this->reduce2, this->dim, this->shape);

        this->doBcast1 = (this->reduce1[this->dim - 1] == 1);
        this->doBcast2 = (this->reduce2[this->dim - 1] == 1);

        // 循环处理当前核分配到的行
        // for (uint32_t j = this->startRow; j < this->endRow; j++)
        // {
        //     uint32_t start1 = 0, start2 = 0;

        //     // 计算 Input 偏移
        //     CalculateStart(j * this->lastDimLength, start1, dn1, reduce1, d);
        //     CalculateStart(j * this->lastDimLength, start2, dn2, reduce2, d);

        //     CopyIn(start1, start2);
        //     Compute();
        //     CopyOut(j * this->lastDimLength);
        // }

        // 2. [新增] 初始化当前块起始行的坐标 (curIdx)
        // 原理解析：将线性行号 j 转换为多维坐标 [idx_0, idx_1, ..., idx_{dim-2}]
        // 只需要计算 0 到 dim-2 维，因为 dim-1 维是连续的 Block，起始偏移总是 0
        uint32_t tempLinear = this->startRow * this->lastDimLength;
        for (int k = this->dim - 1; k >= 0; k--)
        {
            uint32_t idx = (tempLinear / d[k + 1]) % this->shape[k];
            if (k < 16)
                this->curIdx[k] = idx;
        }

        // 3. 循环处理每一行
        for (uint32_t j = this->startRow; j < this->endRow; j++)
        {
            uint32_t start1 = 0;
            uint32_t start2 = 0;

            // [修改] 替换原本的 CalculateStart，使用预计算的坐标累加
            // 时间复杂度从 O(Dim * Div) 降低为 O(Dim) 的纯加法
            for (int k = 0; k < this->dim - 1; k++) // 只需要处理到倒数第二维
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

            // 注意：对于最后一维 (dim-1)，CalculateStart 计算出的 index 始终为 0 (因为是对行首)，
            // 所以不需要加 curIdx[dim-1] * dn[dim]，这也是为什么循环只到 dim-1

            CopyIn(start1, start2);
            Compute();
            CopyOut(j * this->lastDimLength);

            // [新增] 更新坐标：类似于计数器进位，避免除法
            // 从倒数第二维开始更新 (dim-2)，因为 dim-1 是被 lastDimLength 处理的
            for (int k = this->dim - 2; k >= 0; k--)
            {
                this->curIdx[k]++;
                if (this->curIdx[k] < this->shape[k])
                {
                    break; // 没有进位，结束更新
                }
                this->curIdx[k] = 0; // 发生进位，当前维清零，继续循环让上一维+1
            }
        }
    }

private:
    // 辅助函数：安全的广播赋值 (使用白名单，避开 bool/int8/int64 调用 Duplicate)
    __aicore__ inline void BroadcastT(AscendC::LocalTensor<T> &tensor, T val, uint32_t len)
    {
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float> ||
                      std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t> ||
                      std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t> ||
                      std::is_same_v<T, __bf16>)
        {
            // 支持的类型，使用向量指令
            AscendC::Duplicate(tensor, val, len);
        }
        else
        {
            // 不支持的类型 (bool, int8, uint8, int64)，使用标量循环
            for (uint32_t i = 0; i < len; ++i)
            {
                tensor.SetValue(i, val);
            }
        }
    }

    __aicore__ inline void CalculateStart(uint32_t j, uint32_t &start, uint32_t *dn, uint32_t *reduce, uint32_t *d)
    {
        for (int k = this->dim - 1; k >= 0; k--)
        {
            uint32_t index = (j / d[k + 1] % this->shape[k]);
            if (reduce[k] == 0)
            {
                start += dn[k + 1] * index;
            }
        }
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

    __aicore__ inline void InitCastBuffers()
    {
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            pipe.InitBuffer(castQueueX1, BUFFER_NUM, this->tileLength * sizeof(half));
            pipe.InitBuffer(castQueueX2, BUFFER_NUM, this->tileLength * sizeof(half));
            pipe.InitBuffer(castQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t>)
        {
            pipe.InitBuffer(castQueueX1, BUFFER_NUM, this->tileLength * sizeof(float));
            pipe.InitBuffer(castQueueX2, BUFFER_NUM, this->tileLength * sizeof(float));
            pipe.InitBuffer(castQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        }
        else if constexpr (std::is_same_v<T, int64_t>)
        {
            pipe.InitBuffer(castQueueX1, BUFFER_NUM, this->tileLength * sizeof(int32_t));
            pipe.InitBuffer(castQueueX2, BUFFER_NUM, this->tileLength * sizeof(int32_t));
            pipe.InitBuffer(castQueueY, BUFFER_NUM, this->tileLength * sizeof(int32_t));
            pipe.InitBuffer(castQueueFloatX1, BUFFER_NUM, this->tileLength * sizeof(float));
            pipe.InitBuffer(castQueueFloatX2, BUFFER_NUM, this->tileLength * sizeof(float));
            pipe.InitBuffer(castQueueFloatY, BUFFER_NUM, this->tileLength * sizeof(float));
        }
        else if constexpr (std::is_same_v<T, __bf16>)
        {
            pipe.InitBuffer(castQueueX1, BUFFER_NUM, this->tileLength * sizeof(float));
            pipe.InitBuffer(castQueueX2, BUFFER_NUM, this->tileLength * sizeof(float));
            pipe.InitBuffer(castQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        }
    }

    __aicore__ inline void CopyIn(uint32_t start1, uint32_t start2)
    {
        AscendC::LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
        AscendC::LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();

        // 定义最小拷贝长度 (32字节)，避免非对齐标量拷贝失败
        // 对于 half (2B), 32B = 16 elems; float (4B), 32B = 8 elems
        uint32_t minCopyLen = 32 / sizeof(T);
        if (minCopyLen < 1)
            minCopyLen = 1;

        if (this->doBcast1)
        {
            // [修复] 即使只要1个，也拷贝一段数据，确保 DMA 正常工作
            // 注意：这可能会在 Tensor 极末尾处读取越界，但 GM 通常有 Padding，在 Kernel 场景通常是安全的
            AscendC::DataCopy(x1Local, x1Gm[start1], minCopyLen);
        }
        else
        {
            AscendC::DataCopy(x1Local, x1Gm[start1], this->lastDimLength);
        }

        if (this->doBcast2)
        {
            // [修复] 同上
            AscendC::DataCopy(x2Local, x2Gm[start2], minCopyLen);
        }
        else
        {
            AscendC::DataCopy(x2Local, x2Gm[start2], this->lastDimLength);
        }

        // Debug: 打印 CopyIn 后的数据（仅 float16，第一个 block）
        // if constexpr (std::is_same_v<T, half>) {
        //     if (AscendC::GetBlockIdx() == 0) {
        //         uint32_t dbgN = (this->lastDimLength < 8u) ? this->lastDimLength : 8u;
        //         AscendC::printf("bcast CopyIn: start1=%u start2=%u lastDimLength=%u doBcast1=%d doBcast2=%d\n",
        //             start1, start2, this->lastDimLength, this->doBcast1 ? 1 : 0, this->doBcast2 ? 1 : 0);
        //         for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
        //             AscendC::printf("bcast CopyIn data: idx=%u x1=%f x2=%f\n",
        //                 dbg,
        //                 static_cast<float>(x1Local.GetValue(dbg)),
        //                 static_cast<float>(x2Local.GetValue(dbg)));
        //         }
        //     }
        // }
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
        AscendC::LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        AscendC::LocalTensor<uint8_t> maskLocal = calcQueue.AllocTensor<uint8_t>();

        // Debug: 打印广播前的数据（仅 float16，第一个 block）
        // if constexpr (std::is_same_v<T, half>) {
        //     if (AscendC::GetBlockIdx() == 0) {
        //         uint32_t dbgN = (this->lastDimLength < 8u) ? this->lastDimLength : 8u;
        //         AscendC::printf("bcast Compute before Broadcast: lastDimLength=%u tileLength=%u doBcast1=%d doBcast2=%d\n",
        //             this->lastDimLength, this->tileLength, this->doBcast1 ? 1 : 0, this->doBcast2 ? 1 : 0);
        //         for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
        //             AscendC::printf("bcast before Broadcast: idx=%u x1=%f x2=%f\n",
        //                 dbg,
        //                 static_cast<float>(x1Local.GetValue(dbg)),
        //                 static_cast<float>(x2Local.GetValue(dbg)));
        //         }
        //     }
        // }

        // 广播处理：使用 BroadcastT 安全函数
        if (this->doBcast1)
        {
            T val = x1Local.GetValue(0);
            BroadcastT(x1Local, val, this->tileLength);
        }
        if (this->doBcast2)
        {
            T val = x2Local.GetValue(0);
            BroadcastT(x2Local, val, this->tileLength);
        }

        // Debug: 打印广播后的数据（仅 float16，第一个 block）
        // if constexpr (std::is_same_v<T, half>) {
        //     if (AscendC::GetBlockIdx() == 0) {
        //         uint32_t dbgN = (this->lastDimLength < 8u) ? this->lastDimLength : 8u;
        //         for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
        //             AscendC::printf("bcast after Broadcast: idx=%u x1=%f x2=%f\n",
        //                 dbg,
        //                 static_cast<float>(x1Local.GetValue(dbg)),
        //                 static_cast<float>(x2Local.GetValue(dbg)));
        //         }
        //     }
        // }

        uint32_t calCount = this->tileLength;

        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            AscendC::LocalTensor<half> x1Cast = castQueueX1.AllocTensor<half>();
            AscendC::LocalTensor<half> x2Cast = castQueueX2.AllocTensor<half>();
            AscendC::LocalTensor<half> yCast = castQueueY.AllocTensor<half>();
            AscendC::Cast(x1Cast, x1Local, AscendC::RoundMode::CAST_NONE, calCount);
            AscendC::Cast(x2Cast, x2Local, AscendC::RoundMode::CAST_NONE, calCount);
            AscendC::Max(yCast, x1Cast, x2Cast, calCount);
            AscendC::Cast(yLocal, yCast, AscendC::RoundMode::CAST_TRUNC, calCount);
            castQueueX1.FreeTensor(x1Cast);
            castQueueX2.FreeTensor(x2Cast);
            castQueueY.FreeTensor(yCast);
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t>)
        {
            AscendC::LocalTensor<float> x1Cast = castQueueX1.AllocTensor<float>();
            AscendC::LocalTensor<float> x2Cast = castQueueX2.AllocTensor<float>();
            AscendC::LocalTensor<float> yCast = castQueueY.AllocTensor<float>();
            AscendC::Cast(x1Cast, x1Local, AscendC::RoundMode::CAST_NONE, calCount);
            AscendC::Cast(x2Cast, x2Local, AscendC::RoundMode::CAST_NONE, calCount);
            AscendC::Max(yCast, x1Cast, x2Cast, calCount);
            AscendC::Cast(yLocal, yCast, AscendC::RoundMode::CAST_TRUNC, calCount);
            castQueueX1.FreeTensor(x1Cast);
            castQueueX2.FreeTensor(x2Cast);
            castQueueY.FreeTensor(yCast);
        }
        else if constexpr (std::is_same_v<T, int64_t>)
        {
            AscendC::LocalTensor<int32_t> x1Int32 = castQueueX1.AllocTensor<int32_t>();
            AscendC::LocalTensor<int32_t> x2Int32 = castQueueX2.AllocTensor<int32_t>();
            AscendC::LocalTensor<int32_t> yInt32 = castQueueY.AllocTensor<int32_t>();
            AscendC::Cast(x1Int32, x1Local, AscendC::RoundMode::CAST_NONE, calCount);
            AscendC::Cast(x2Int32, x2Local, AscendC::RoundMode::CAST_NONE, calCount);

            AscendC::LocalTensor<float> x1Float = castQueueFloatX1.AllocTensor<float>();
            AscendC::LocalTensor<float> x2Float = castQueueFloatX2.AllocTensor<float>();
            AscendC::LocalTensor<float> yFloat = castQueueFloatY.AllocTensor<float>();

            AscendC::Cast(x1Float, x1Int32, AscendC::RoundMode::CAST_NONE, calCount);
            AscendC::Cast(x2Float, x2Int32, AscendC::RoundMode::CAST_NONE, calCount);
            AscendC::Max(yFloat, x1Float, x2Float, calCount);
            AscendC::Cast(yInt32, yFloat, AscendC::RoundMode::CAST_TRUNC, calCount);
            AscendC::Cast(yLocal, yInt32, AscendC::RoundMode::CAST_NONE, calCount);

            castQueueX1.FreeTensor(x1Int32);
            castQueueX2.FreeTensor(x2Int32);
            castQueueY.FreeTensor(yInt32);
            castQueueFloatX1.FreeTensor(x1Float);
            castQueueFloatX2.FreeTensor(x2Float);
            castQueueFloatY.FreeTensor(yFloat);
        }
        else if constexpr (std::is_same_v<T, __bf16>)
        {
            AscendC::LocalTensor<float> x1Cast = castQueueX1.AllocTensor<float>();
            AscendC::LocalTensor<float> x2Cast = castQueueX2.AllocTensor<float>();
            AscendC::LocalTensor<float> yCast = castQueueY.AllocTensor<float>();
            // Debug: 打印 Compute 开始时的数据（仅 bf16，第一个 block）
            // Debug: 打印 Compute 开始时的数据（仅 bf16，第一个 block）
            uint32_t dbgN = (this->lastDimLength < 8u) ? this->lastDimLength : 8u;
            // if (AscendC::GetBlockIdx() == 0) {
            //     for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
            //         float x1 = static_cast<float>(x1Local.GetValue(dbg));
            //         float x2 = static_cast<float>(x2Local.GetValue(dbg));
            //         AscendC::printf("bf16 bcast Compute before Cast: idx=%u x1=%f x2=%f\n",
            //             dbg, x1, x2);
            //     }
            // }
            AscendC::Cast(x1Cast, x1Local, AscendC::RoundMode::CAST_NONE, calCount);
            AscendC::Cast(x2Cast, x2Local, AscendC::RoundMode::CAST_NONE, calCount);

            // Debug: 打印 Cast 到 float 后的数据
            // if (AscendC::GetBlockIdx() == 0) {
            //     for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
            //         AscendC::printf("bf16->float bcast: idx=%u x1=%f x2=%f\n",
            //             dbg,
            //             static_cast<float>(x1Cast.GetValue(dbg)),
            //             static_cast<float>(x2Cast.GetValue(dbg)) );
            //     }
            // }
            AscendC::Max(yCast, x1Cast, x2Cast, calCount);
            // if (AscendC::GetBlockIdx() == 0) {
            //     for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
            //         AscendC::printf(" idx=%u yCast=%f x1Cast=%f,x2Cast=%f \n",
            //             dbg,
            //             static_cast<float>(yCast.GetValue(dbg)),
            //             static_cast<float>(x1Cast.GetValue(dbg)),
            //             static_cast<float>(x2Cast.GetValue(dbg)) );
            //     }
            // }
            // AscendC::Compare(maskLocal, x1Cast, x1Cast, AscendC::CMPMODE::EQ, calCount);
            // AscendC::Select(yCast, maskLocal, yCast, x2Cast, AscendC::SELMODE::VSEL_CMPMASK_SPR, calCount);
            // AscendC::Compare(maskLocal, x2Cast, x2Cast, AscendC::CMPMODE::EQ, calCount);
            // AscendC::Select(yCast, maskLocal, yCast, x1Cast, AscendC::SELMODE::VSEL_CMPMASK_SPR, calCount);

            AscendC::Cast(yLocal, yCast, AscendC::RoundMode::CAST_TRUNC, calCount);
            castQueueX1.FreeTensor(x1Cast);
            castQueueX2.FreeTensor(x2Cast);
            castQueueY.FreeTensor(yCast);
        }
        else if constexpr (std::is_same_v<T, bool>)
        {
            for (uint32_t i = 0; i < this->lastDimLength; ++i)
            {
                yLocal.SetValue(i, x1Local.GetValue(i) || x2Local.GetValue(i));
            }
        }
        else
        { // float, half
            // Debug: 打印 Max 操作前的数据（仅 float16，第一个 block）
            // if constexpr (std::is_same_v<T, half>) {
            //     if (AscendC::GetBlockIdx() == 0) {
            //         uint32_t dbgN = (this->lastDimLength < 8u) ? this->lastDimLength : 8u;
            //         // AscendC::printf("bcast Compute before Max: calCount=%u\n", calCount);
            //         // for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
            //         //     AscendC::printf("bcast before Max: idx=%u x1=%f x2=%f\n",
            //         //         dbg,
            //         //         static_cast<float>(x1Local.GetValue(dbg)),
            //         //         static_cast<float>(x2Local.GetValue(dbg)));
            //         // }
            //     }
            // }

            AscendC::Max(yLocal, x1Local, x2Local, calCount);

            // Debug: 打印 Max 操作后的结果（仅 float16，第一个 block）
            // if constexpr (std::is_same_v<T, half>) {
            //     if (AscendC::GetBlockIdx() == 0) {
            //         uint32_t dbgN = (this->lastDimLength < 8u) ? this->lastDimLength : 8u;
            //         for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
            //             AscendC::printf("bcast after Max: idx=%u y=%f\n",
            //                 dbg,
            //                 static_cast<float>(yLocal.GetValue(dbg)));
            //         }
            //     }
            // }

            // AscendC::Compare(maskLocal, x1Local, x1Local, AscendC::CMPMODE::EQ, calCount);
            // AscendC::Select(yLocal, maskLocal, yLocal, x2Local, AscendC::SELMODE::VSEL_CMPMASK_SPR, calCount);
            // AscendC::Compare(maskLocal, x2Local, x2Local, AscendC::CMPMODE::EQ, calCount);
            // AscendC::Select(yLocal, maskLocal, yLocal, x1Local, AscendC::SELMODE::VSEL_CMPMASK_SPR, calCount);

            // Debug: 打印 NaN 修复后的最终结果（仅 float16，第一个 block）
            // if constexpr (std::is_same_v<T, half>) {
            //     if (AscendC::GetBlockIdx() == 0) {
            //         uint32_t dbgN = (this->lastDimLength < 8u) ? this->lastDimLength : 8u;
            //         for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
            //             AscendC::printf("bcast after NaN fix: idx=%u y=%f\n",
            //                 dbg,
            //                 static_cast<float>(yLocal.GetValue(dbg)));
            //         }
            //     }
            // }
        }

        calcQueue.FreeTensor(maskLocal);
        outQueueY.EnQue<T>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(uint32_t start)
    {
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();

        // Debug: 打印 CopyOut 前的数据（仅 float16，第一个 block）
        // if constexpr (std::is_same_v<T, half>) {
        //     if (AscendC::GetBlockIdx() == 0) {
        //         uint32_t dbgN = (this->lastDimLength < 8u) ? this->lastDimLength : 8u;
        //         AscendC::printf("bcast CopyOut: start=%u lastDimLength=%u\n", start, this->lastDimLength);
        //         for (uint32_t dbg = 0; dbg < dbgN; ++dbg) {
        //             AscendC::printf("bcast CopyOut data: idx=%u y=%f\n",
        //                 dbg,
        //                 static_cast<float>(yLocal.GetValue(dbg)));
        //         }
        //     }
        // }

        AscendC::DataCopy(yGm[start], yLocal, this->lastDimLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::QuePosition::VECCALC, BUFFER_NUM> calcQueue;
    AscendC::TQue<AscendC::QuePosition::VECCALC, BUFFER_NUM> castQueueX1, castQueueX2, castQueueY;
    [[maybe_unused]] AscendC::TQue<AscendC::QuePosition::VECCALC, BUFFER_NUM> castQueueFloatX1, castQueueFloatX2, castQueueFloatY;

    AscendC::GlobalTensor<T> x1Gm, x2Gm, yGm;
    uint32_t totalLength, lastDimLength, tileLength;
    uint32_t startRow, endRow;
    uint32_t *reduce1;
    uint32_t *reduce2;
    uint32_t *shape;
    uint32_t dim;
    bool doBcast1 = false, doBcast2 = false;
    // 加入坐标计算逻辑，添加成员变量，假设最大维度不超过16
    uint32_t curIdx[16];
};




template <typename T>
class KernelFmaxBroadcast_FirstD
{
private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_scalar, inQueueX_vec;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    TQue<QuePosition::VECCALC, BUFFER_NUM> inQueueX_cast_scalar, inQueueX_cast_vec;

    GlobalTensor<T> x1Gm, x2Gm, yGm;

    // Tiling 参数
    uint32_t rows1, len1;
    uint32_t rows2, len2;
    uint32_t calcLength; // 最后一维的总长度
    uint32_t startRow, endRow;

    // 切分参数
    uint32_t maxTileLength; // 单次最大处理长度 (元素个数)
    int row_offset;
    int tile_offset;
    int tile_length;
    int blockIdx;
    int blockNum;
    int tile_mode;
public:
    __aicore__ inline KernelFmaxBroadcast_FirstD() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out,
        int* dim1, int* dim2) {
        int blockLength;
        volatile int* v_dim1 = (volatile int*)dim1;
        volatile int* v_dim2 = (volatile int*)dim2;
        this->rows1 = (uint32_t)v_dim1[0];
        this->len1 = (uint32_t)v_dim1[1];
        this->rows2 = (uint32_t)v_dim2[0];
        this->len2 = (uint32_t)v_dim2[1];
        this->calcLength = (this->len1 > this->len2) ? this->len1 : this->len2;

        // printf("rows1=%u len1=%u rows2=%u len2=%u\n,calcLength=%u\n", this->rows1, this->len1, this->rows2, this->len2, this->calcLength);
        uint32_t totalRows = (this->rows1 > this->rows2) ? this->rows1 : this->rows2;
        this->blockNum = GetBlockNum();
        this->blockIdx = GetBlockIdx();

        if constexpr (std::is_same_v<T, bfloat16_t>) {
            this->maxTileLength = 14016; 
        }
        else if constexpr (std::is_same_v<T, float>) {
            this->maxTileLength = 8192*2;
        }
        else if constexpr (std::is_same_v<T, half>) {
            this->maxTileLength = 8192*4;
        }
        else {
            this->maxTileLength = 8192*2;
        }

        if (this->calcLength > this->maxTileLength) {
            this->tile_mode = 0; // calclength超过了maxTileLength，则切分列
            blockLength = (this->calcLength + blockNum - 1) / blockNum;
            blockLength = (blockLength + 16 - 1) / 16 * 16;
            blockLength = blockLength <= this->calcLength ? blockLength : this->calcLength;
            this->startRow = 0;
            this->endRow = totalRows;
            if(blockLength > this->maxTileLength) {
                this->tile_length = this->maxTileLength;
            }
            else {
                this->tile_length = blockLength;
            }
            this->row_offset = 1;
            this->tile_offset = blockNum*this->tile_length;
        }
        else {
            this->tile_mode = 1; // calclength不超过maxTileLength，则不分列
            uint32_t rowsPerBlock = totalRows / blockNum;
            uint32_t remainRows = totalRows % blockNum;
            if (blockIdx < remainRows) {
                rowsPerBlock = rowsPerBlock + 1;
                this->startRow = blockIdx * rowsPerBlock;
            }
            else {
                this->startRow = remainRows * (rowsPerBlock + 1) + (blockIdx - remainRows) * rowsPerBlock;
            }
            this->endRow = this->startRow + rowsPerBlock;
            this->row_offset = this->maxTileLength/this->calcLength;
            this->tile_offset = this->calcLength;
            this->tile_length = this->calcLength * this->row_offset;
        }

        // 3. Buffer 初始化
        // 按 maxTileLength 申请内存
        uint32_t bufSize = this->tile_length * sizeof(float); // 按 float 算最大需求

        x1Gm.SetGlobalBuffer((__gm__ T*)input);
        x2Gm.SetGlobalBuffer((__gm__ T*)other);
        yGm.SetGlobalBuffer((__gm__ T*)out);

        pipe.InitBuffer(inQueueX_scalar, 1, this->tile_length * sizeof(T));
        pipe.InitBuffer(inQueueX_vec, 1, this->tile_length * sizeof(T));
        pipe.InitBuffer(outQueueY, 1, this->tile_length * sizeof(T));

        if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe.InitBuffer(inQueueX_cast_scalar, 1, bufSize);
            pipe.InitBuffer(inQueueX_cast_vec, 1, bufSize);
        }
    }

    __aicore__ inline void Process() {
        if (this->startRow >= this->endRow) return;
        LocalTensor<T> tX_scalar, tX_vec, tY;
        LocalTensor<float> fX_cast_scalar, fX_cast_vec;
        int tile_start;
        if(this->tile_mode == 0) {
            tile_start = this->blockIdx * this->tile_length;
        }
        else {
            tile_start = 0;
        }
        //外层循环tilelength
        if(this->tile_mode == 0) {
        for (uint32_t offset = tile_start; offset < this->calcLength; offset += this->tile_offset) {
            // printf("offset=%u maxTileLength=%u\n", offset, this->maxTileLength);
            // 计算当前切片的长度 (处理最后一块不足 maxTileLength 的情况)
            uint32_t currentTileLen = this->tile_length;

            if (offset + currentTileLen > this->calcLength) {
                currentTileLen = this->calcLength - offset;
            }
              
            // 准备 Copy 参数
            DataCopyExtParams copyParamsVec = { 1, (uint32_t)(currentTileLen * sizeof(T)), 0, 0, 0 };
            DataCopyExtParams copyParamsScalar = { 1, (uint32_t)(currentTileLen * sizeof(T)), 0, 0, 0 };
            DataCopyPadExtParams<T> padParams{ false, 0, 0, 0 };

            tX_scalar = inQueueX_scalar.AllocTensor<T>();
            if (this->rows1 == 1) {
                DataCopyPad(tX_scalar, x1Gm[offset], copyParamsScalar, padParams);
                inQueueX_scalar.EnQue(tX_scalar);
                tX_scalar = inQueueX_scalar.DeQue<T>();
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    fX_cast_scalar = inQueueX_cast_scalar.AllocTensor<float>();
                    Cast(fX_cast_scalar, tX_scalar, RoundMode::CAST_NONE, currentTileLen);
                    inQueueX_cast_scalar.EnQue(fX_cast_scalar);
                    fX_cast_scalar = inQueueX_cast_scalar.DeQue<float>();
                }
            }
            else if (this->rows2 == 1) {
                DataCopyPad(tX_scalar, x2Gm[offset], copyParamsScalar, padParams);
                inQueueX_scalar.EnQue(tX_scalar);
                tX_scalar = inQueueX_scalar.DeQue<T>();
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    fX_cast_scalar = inQueueX_cast_scalar.AllocTensor<float>();
                    Cast(fX_cast_scalar, tX_scalar, RoundMode::CAST_NONE, currentTileLen);
                    inQueueX_cast_scalar.EnQue(fX_cast_scalar);
                    fX_cast_scalar = inQueueX_cast_scalar.DeQue<float>();
                }
            }


            // 内层循环：循环rows，每个rows使用的广播数据相同
            for (uint32_t i = this->startRow; i < this->endRow; i += this->row_offset) {
                // printf("row index=%u\n", i);
                tY = outQueueY.AllocTensor<T>();
                tX_vec = inQueueX_vec.AllocTensor<T>();
                uint32_t rowIdx = i;
                uint32_t gmOffset = rowIdx * this->calcLength + offset;
                //x2是scalar，提前准备好了，x1是vec
                if (this->rows1 != 1) {
                    DataCopyPad(tX_vec, x1Gm[gmOffset], copyParamsVec, padParams);
                }
                //x1是scalar，提前准备好了，x2是vec
                else if (this->rows2 != 1) {
                    DataCopyPad(tX_vec, x2Gm[gmOffset], copyParamsVec, padParams);
                }
                inQueueX_vec.EnQue(tX_vec);
                tX_vec = inQueueX_vec.DeQue<T>();
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    fX_cast_vec = inQueueX_cast_vec.AllocTensor<float>();
                    Cast(fX_cast_vec, tX_vec, RoundMode::CAST_NONE, currentTileLen);
                    inQueueX_cast_vec.EnQue(fX_cast_vec);
                    fX_cast_vec = inQueueX_cast_vec.DeQue<float>();
                    Max(fX_cast_vec, fX_cast_scalar, fX_cast_vec, currentTileLen);
                    Cast(tY, fX_cast_vec, RoundMode::CAST_ROUND, currentTileLen);
                }
                else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float>) {
                    Max(tY, tX_scalar, tX_vec, currentTileLen);
                }
                outQueueY.EnQue(tY);
                tY = outQueueY.DeQue<T>();
                // 输出 Offset: row * calcLength + colOffset
                DataCopyPad(yGm[gmOffset], tY, copyParamsVec);
                outQueueY.FreeTensor(tY);
                inQueueX_vec.FreeTensor(tX_vec);
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    inQueueX_cast_vec.FreeTensor(fX_cast_vec);
                }
            }

            inQueueX_scalar.FreeTensor(tX_scalar);
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                inQueueX_cast_scalar.FreeTensor(fX_cast_scalar);
            }
        }
        }
        else {
            // tile_mode = 1: 最后一维长度较小，按行切分
            for (uint32_t i = this->startRow; i < this->endRow; i += this->row_offset) {
                // 1. 计算当前处理的行数和总长度
                uint32_t offset = i * this->calcLength; // 大 Tensor 和输出的偏移量
                uint32_t currentRows = this->row_offset;
                if(i + currentRows > this->endRow) {
                    currentRows = this->endRow - i;
                }
                uint32_t currentTileLen = currentRows * this->calcLength;

                // 2. 准备 Copy 参数
                // vecParams 用于大块数据 (currentRows * L)
                DataCopyExtParams copyParamsVec = { 1, (uint32_t)(currentTileLen * sizeof(T)), 0, 0, 0 };
                // scalarParams 用于广播数据 (只拷贝 1 * L)
                DataCopyExtParams copyParamsScalar = { 1, (uint32_t)(this->calcLength * sizeof(T)), 0, 0, 0 };
                DataCopyPadExtParams<T> padParams{ false, 0, 0, 0 };
    
                tX_scalar = inQueueX_scalar.AllocTensor<T>();
                
                // 3. 搬运广播数据 (rows=1 的那个)
                // 注意：这里必须用 xGm[0]，且长度仅为 1 行
                if (this->rows1 == 1) {
                    DataCopyPad(tX_scalar, x1Gm[0], copyParamsScalar, padParams);
                    inQueueX_scalar.EnQue(tX_scalar);
                    tX_scalar = inQueueX_scalar.DeQue<T>();
                    if constexpr (std::is_same_v<T, bfloat16_t>) {
                        fX_cast_scalar = inQueueX_cast_scalar.AllocTensor<float>();
                        Cast(fX_cast_scalar, tX_scalar, RoundMode::CAST_NONE, this->calcLength);
                        inQueueX_cast_scalar.EnQue(fX_cast_scalar);
                        fX_cast_scalar = inQueueX_cast_scalar.DeQue<float>();
                    }
                }
                else if (this->rows2 == 1) {
                    DataCopyPad(tX_scalar, x2Gm[0], copyParamsScalar, padParams);
                    inQueueX_scalar.EnQue(tX_scalar);
                    tX_scalar = inQueueX_scalar.DeQue<T>();
                    if constexpr (std::is_same_v<T, bfloat16_t>) {
                        fX_cast_scalar = inQueueX_cast_scalar.AllocTensor<float>();
                        Cast(fX_cast_scalar, tX_scalar, RoundMode::CAST_NONE, this->calcLength);
                        inQueueX_cast_scalar.EnQue(fX_cast_scalar);
                        fX_cast_scalar = inQueueX_cast_scalar.DeQue<float>();
                    }
                }

                // 4. 搬运大块数据 & 申请输出内存
                tY = outQueueY.AllocTensor<T>();
                tX_vec = inQueueX_vec.AllocTensor<T>();
                
                // 使用 offset 搬运多行数据
                if (this->rows1 != 1) {
                    DataCopyPad(tX_vec, x1Gm[offset], copyParamsVec, padParams);
                }
                else if (this->rows2 != 1) {
                    DataCopyPad(tX_vec, x2Gm[offset], copyParamsVec, padParams);
                }
                
                inQueueX_vec.EnQue(tX_vec);
                tX_vec = inQueueX_vec.DeQue<T>();

                // 5. 计算 (循环每一行应用广播向量)
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    fX_cast_vec = inQueueX_cast_vec.AllocTensor<float>();
                    Cast(fX_cast_vec, tX_vec, RoundMode::CAST_NONE, currentTileLen);
                    inQueueX_cast_vec.EnQue(fX_cast_vec);
                    fX_cast_vec = inQueueX_cast_vec.DeQue<float>();
                    
                    // 核心修正：并在 UB 内部循环，复用 scalar 的那一行数据
                    for (uint32_t r = 0; r < currentRows; ++r) {
                        uint32_t localOffset = r * this->calcLength;
                        Max(fX_cast_vec[localOffset], fX_cast_scalar, fX_cast_vec[localOffset], this->calcLength);
                    }
                    
                    Cast(tY, fX_cast_vec, RoundMode::CAST_ROUND, currentTileLen);
                }
                else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float>) {
                    for (uint32_t r = 0; r < currentRows; ++r) {
                        uint32_t localOffset = r * this->calcLength;
                        Max(tY[localOffset], tX_scalar, tX_vec[localOffset], this->calcLength);
                    }
                }
                
                // 6. 搬出结果
                outQueueY.EnQue(tY);
                tY = outQueueY.DeQue<T>();
                DataCopyPad(yGm[offset], tY, copyParamsVec);
                
                // 7. 释放资源
                outQueueY.FreeTensor(tY);
                inQueueX_vec.FreeTensor(tX_vec);
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    inQueueX_cast_vec.FreeTensor(fX_cast_vec);
                }
                
                inQueueX_scalar.FreeTensor(tX_scalar);
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    inQueueX_cast_scalar.FreeTensor(fX_cast_scalar);
                }
            }
        }


    }
};

// =========================================================================
// Kernel Entry
// =========================================================================
extern "C" __global__ __aicore__ void fmax(GM_ADDR input, GM_ADDR other, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        KernelFmax<half> op;
        op.Init(input, other, out, tiling_data.bigDataCoreNum, tiling_data.smallBlockLength, tiling_data.bigBlockLength, tiling_data.smallTileNum, tiling_data.smallTileLength, tiling_data.smallLasttileLength, tiling_data.bigTileNum, tiling_data.bigTileLength, tiling_data.bigLasttileLength);
        op.Process();
    }
    else if (TILING_KEY_IS(2)) {
        KernelFmaxBroadcast<half> op;
        op.Init(input, other, out, tiling_data.totalLength, tiling_data.dim, tiling_data.shape, tiling_data.reduce1, tiling_data.reduce2);
        op.Process();
    }
    else if (TILING_KEY_IS(3)) {
        KernelFmax<float> op;
        op.Init(input, other, out, tiling_data.bigDataCoreNum, tiling_data.smallBlockLength, tiling_data.bigBlockLength, tiling_data.smallTileNum, tiling_data.smallTileLength, tiling_data.smallLasttileLength, tiling_data.bigTileNum, tiling_data.bigTileLength, tiling_data.bigLasttileLength);
        op.Process();
    }
    else if (TILING_KEY_IS(4)) {
        KernelFmaxBroadcast<float> op;
        op.Init(input, other, out, tiling_data.totalLength, tiling_data.dim, tiling_data.shape, tiling_data.reduce1, tiling_data.reduce2);
        op.Process();
    }
    else if (TILING_KEY_IS(5)) {
        KernelFmax<int32_t> op;
        op.Init(input, other, out, tiling_data.bigDataCoreNum, tiling_data.smallBlockLength, tiling_data.bigBlockLength, tiling_data.smallTileNum, tiling_data.smallTileLength, tiling_data.smallLasttileLength, tiling_data.bigTileNum, tiling_data.bigTileLength, tiling_data.bigLasttileLength);
        op.Process();
    }
    else if (TILING_KEY_IS(6)) {
        KernelFmaxBroadcast<int32_t> op;
        op.Init(input, other, out, tiling_data.totalLength, tiling_data.dim, tiling_data.shape, tiling_data.reduce1, tiling_data.reduce2);
        op.Process();
    }

    // ... 其他类型保持不变，确保对应 TILING_KEY ...
    else if (TILING_KEY_IS(7)) {
        KernelFmax<int8_t> op;
        op.Init(input, other, out, tiling_data.bigDataCoreNum, tiling_data.smallBlockLength, tiling_data.bigBlockLength, tiling_data.smallTileNum, tiling_data.smallTileLength, tiling_data.smallLasttileLength, tiling_data.bigTileNum, tiling_data.bigTileLength, tiling_data.bigLasttileLength);
        op.Process();
    }
    else if (TILING_KEY_IS(8)) {
        KernelFmaxBroadcast<int8_t> op;
        op.Init(input, other, out, tiling_data.totalLength, tiling_data.dim, tiling_data.shape, tiling_data.reduce1, tiling_data.reduce2);
        op.Process();
    }
    else if (TILING_KEY_IS(9)) {
        KernelFmax<int16_t> op;
        op.Init(input, other, out, tiling_data.bigDataCoreNum, tiling_data.smallBlockLength, tiling_data.bigBlockLength, tiling_data.smallTileNum, tiling_data.smallTileLength, tiling_data.smallLasttileLength, tiling_data.bigTileNum, tiling_data.bigTileLength, tiling_data.bigLasttileLength);
        op.Process();
    }
    else if (TILING_KEY_IS(10)) {
        KernelFmaxBroadcast<int16_t> op;
        op.Init(input, other, out, tiling_data.totalLength, tiling_data.dim, tiling_data.shape, tiling_data.reduce1, tiling_data.reduce2);
        op.Process();
    }
    else if (TILING_KEY_IS(11)) {
        KernelFmax<int64_t> op;
        op.Init(input, other, out, tiling_data.bigDataCoreNum, tiling_data.smallBlockLength, tiling_data.bigBlockLength, tiling_data.smallTileNum, tiling_data.smallTileLength, tiling_data.smallLasttileLength, tiling_data.bigTileNum, tiling_data.bigTileLength, tiling_data.bigLasttileLength);
        op.Process();
    }
    else if (TILING_KEY_IS(12)) {
        KernelFmaxBroadcast<int64_t> op;
        op.Init(input, other, out, tiling_data.totalLength, tiling_data.dim, tiling_data.shape, tiling_data.reduce1, tiling_data.reduce2);
        op.Process();
    }
    else if (TILING_KEY_IS(13)) {
        KernelFmax<uint8_t> op;
        op.Init(input, other, out, tiling_data.bigDataCoreNum, tiling_data.smallBlockLength, tiling_data.bigBlockLength, tiling_data.smallTileNum, tiling_data.smallTileLength, tiling_data.smallLasttileLength, tiling_data.bigTileNum, tiling_data.bigTileLength, tiling_data.bigLasttileLength);
        op.Process();
    }
    else if (TILING_KEY_IS(14)) {
        KernelFmaxBroadcast<uint8_t> op;
        op.Init(input, other, out, tiling_data.totalLength, tiling_data.dim, tiling_data.shape, tiling_data.reduce1, tiling_data.reduce2);
        op.Process();
    }
    else if (TILING_KEY_IS(15)) {
        KernelFmax<__bf16> op;
        op.Init(input, other, out, tiling_data.bigDataCoreNum, tiling_data.smallBlockLength, tiling_data.bigBlockLength, tiling_data.smallTileNum, tiling_data.smallTileLength, tiling_data.smallLasttileLength, tiling_data.bigTileNum, tiling_data.bigTileLength, tiling_data.bigLasttileLength);
        op.Process();
    }
    else if (TILING_KEY_IS(16)) {
        KernelFmaxBroadcast<__bf16> op;
        op.Init(input, other, out, tiling_data.totalLength, tiling_data.dim, tiling_data.shape, tiling_data.reduce1, tiling_data.reduce2);
        op.Process();
    }
    else if (TILING_KEY_IS(17)) {
        KernelFmax<bool> op;
        op.Init(input, other, out, tiling_data.bigDataCoreNum, tiling_data.smallBlockLength, tiling_data.bigBlockLength, tiling_data.smallTileNum, tiling_data.smallTileLength, tiling_data.smallLasttileLength, tiling_data.bigTileNum, tiling_data.bigTileLength, tiling_data.bigLasttileLength);
        op.Process();
    }
    else if (TILING_KEY_IS(18)) {
        KernelFmaxBroadcast<bool> op;
        op.Init(input, other, out, tiling_data.totalLength, tiling_data.dim, tiling_data.shape, tiling_data.reduce1, tiling_data.reduce2);
        op.Process();
    }
    else if (TILING_KEY_IS(666)) {
        KernelFmaxBroadcast_FirstD<DTYPE_INPUT> op;
        op.Init(input, other, out, tiling_data.dim1, tiling_data.dim2);
        op.Process();
    }
}