/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <functional>
#include <cmath>
#include <vector>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/pipeline>

#include "cuComplex.h"
#include "cfo_ta_est.hpp"
#include "type_convert.hpp"
#include "nvlog.hpp"
#include "cuphy.hpp"

namespace cg = cooperative_groups;

namespace cfo_ta_est
{
// #define ENABLE_PROFILING
// #define ENABLE_DEBUG
// #define USE_MEMCPY_ASYNC

static constexpr uint32_t N_THREADS_PER_WARP = 32; // cudaDeviceProp::warpSize;

/*VCAST_DONT_INSTRUMENT_START*/
template <typename TElem>
struct tensor_ref
{
    TElem*     pAddr;
    const int* strides;

    CUDA_BOTH
    tensor_ref(void* pAddr, const int* pStrides) :
        pAddr(static_cast<TElem*>(pAddr)),
        strides(pStrides)
    {
    }
    CUDA_BOTH int offset(int i0) const
    {
        return (strides[0] * i0);
    }
    CUDA_BOTH int offset(int i0, int i1) const
    {
        return (strides[0] * i0) + (strides[1] * i1);
    }
    CUDA_BOTH int offset(int i0, int i1, int i2) const
    {
        return (strides[0] * i0) + (strides[1] * i1) + (strides[2] * i2);
    };
    CUDA_BOTH int offset(int i0, int i1, int i2, int i3) const
    {
        return (strides[0] * i0) + (strides[1] * i1) + (strides[2] * i2) + (strides[3] * i3);
    };
    // clang-format off
    CUDA_BOTH TElem&       operator()(int i0)                               { return *(pAddr + offset(i0));             }
    CUDA_BOTH TElem&       operator()(int i0, int i1)                       { return *(pAddr + offset(i0, i1));         }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2)               { return *(pAddr + offset(i0, i1, i2));     }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2, int i3)       { return *(pAddr + offset(i0, i1, i2, i3)); }

    CUDA_BOTH const TElem& operator()(int i0) const                         { return *(pAddr + offset(i0));             }
    CUDA_BOTH const TElem& operator()(int i0, int i1) const                 { return *(pAddr + offset(i0, i1));         }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2) const         { return *(pAddr + offset(i0, i1, i2));     }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3) const { return *(pAddr + offset(i0, i1, i2, i3)); }
    // clang-format on
};

template <typename T, int M>
struct block_1D
{
    T         data[M];
    CUDA_BOTH T& operator[](int idx) { return data[idx]; }
};

template <typename T, int M, int N>
struct block_2D
{
    T         data[M * N];
    CUDA_BOTH T& operator()(int m, int n) { return data[(n * M) + m]; }
};

template <typename T, int L, int M, int N>
struct block_3D
{
    T         data[L * M * N];
    CUDA_BOTH T& operator()(int l, int m, int n) { return data[((n * M) + m) * L + l]; }
};

template <typename T, int L, int M, int N, int O>
struct block_4D
{
    T         data[L * M * N * O];
    CUDA_BOTH T& operator()(int l, int m, int n, int o) { return data[((((o * N) + n) * M) + m) * L + l]; }
};

template <typename T, int L, int M, int N, int O, int P>
struct block_5D
{
    T         data[L * M * N * O * P];
    CUDA_BOTH T& operator()(int l, int m, int n, int o, int p) { return data[((((((p * O) + o) * N) + n) * M) + m) * L + l]; }
};


// Partial specialization of block_1D to use shared memory pointers
template <typename T, int M>
struct block_1D<T*, M>
{
    CUDA_BOTH block_1D(T* pData) :
        m_pData(pData){}; // static_assert(std::is_pointer<T>::value, "Must be a pointer type")
    block_1D()                    = delete;
    block_1D(block_1D const& blk) = delete;
    CUDA_BOTH block_1D& operator  =(block_1D const& block) { m_pData = block.m_pData; };
    ~block_1D()                   = default;

    CUDA_BOTH T&               operator[](int idx) { return m_pData[idx]; }
    static constexpr CUDA_BOTH size_t num_elem() { return M; }

private:
    T* m_pData = nullptr;
};

// Partial specialization of block_2D to use shared memory pointers
template <typename T, int M, int N>
struct block_2D<T*, M, N>
{
    CUDA_BOTH block_2D(T* pData) :
        m_pData(pData){};
    block_2D()                    = delete;
    block_2D(block_2D const& blk) = delete;
    CUDA_BOTH block_2D& operator  =(block_2D const& block) { m_pData = block.m_pData; };
    ~block_2D()                   = default;

    CUDA_BOTH T&               operator()(int m, int n) { return m_pData[(n * M) + m]; }
    static constexpr CUDA_BOTH size_t num_elem() { return M * N; }

private:
    T* m_pData = nullptr;
};

// Partial specialization of block_3D to use shared memory pointers
template <typename T, int L, int M, int N>
struct block_3D<T*, L, M, N>
{
    CUDA_BOTH block_3D(T* pData) :
        m_pData(pData){};
    block_3D()                    = delete;
    block_3D(block_3D const& blk) = delete;
    CUDA_BOTH block_3D& operator  =(block_3D const& block) { m_pData = block.m_pData; };
    ~block_3D()                   = default;

    CUDA_BOTH T&               operator()(int l, int m, int n) { return m_pData[((n * M) + m) * L + l]; }
    static constexpr CUDA_BOTH size_t num_elem() { return L * M * N; }

private:
    T* m_pData = nullptr;
};

// Partial specialization of block_4D to use shared memory pointers
template <typename T, int L, int M, int N, int O>
struct block_4D<T*, L, M, N, O>
{
    CUDA_BOTH block_4D(T* pData) :
        m_pData(pData){};
    block_4D()                    = delete;
    block_4D(block_4D const& blk) = delete;
    CUDA_BOTH block_4D& operator  =(block_4D const& block) { m_pData = block.m_pData; };
    ~block_4D()                   = default;

    CUDA_BOTH T&               operator()(int l, int m, int n, int o) { return m_pData[((((o * N) + n) * M) + m) * L + l]; }
    static constexpr CUDA_BOTH size_t num_elem() { return L * M * N * O; }

private:
    T* m_pData = nullptr;
};

// Partial specialization of block_5D to use shared memory pointers
template <typename T, int L, int M, int N, int O, int P>
struct block_5D<T*, L, M, N, O, P>
{
    CUDA_BOTH block_5D(T* pData) :
        m_pData(pData){};
    block_5D()                    = delete;
    block_5D(block_5D const& blk) = delete;
    CUDA_BOTH block_5D& operator  =(block_5D const& block) { m_pData = block.m_pData; };
    ~block_5D()                   = default;

    CUDA_BOTH T&               operator()(int l, int m, int n, int o, int p) { return m_pData[((((((p * O) + o) * N) + n) * M) + m) * L + l]; }
    static constexpr CUDA_BOTH size_t num_elem() { return L * M * N * O * P; }

private:
    T* m_pData = nullptr;
};

// clang-format off
template <typename T> CUDA_BOTH_INLINE constexpr T     cuGet(uint32_t);
template<>            CUDA_BOTH_INLINE constexpr float cuGet(uint32_t x) { return(float(x)); }

template <typename T> CUDA_BOTH_INLINE T         cuGet(int);
template<>            CUDA_BOTH_INLINE float     cuGet(int x) { return(float(x)); }
template<>            CUDA_BOTH_INLINE cuComplex cuGet(int x) { return(make_cuComplex(float(x), 0.0f)); }

template <typename T> CUDA_BOTH_INLINE T         cuGet(float);
template<>            CUDA_BOTH_INLINE cuComplex cuGet(float x) { return(make_cuComplex(x, 0.0f)); }

template <typename T> CUDA_BOTH_INLINE T         cuGet(float,float);
template <>           CUDA_BOTH_INLINE cuComplex cuGet<cuComplex>(float x, float y) { return make_cuComplex(x,y); }

// static CUDA_BOTH_INLINE cuComplex operator/(cuComplex x, float y) { return(make_cuComplex(cuCrealf(x)/y, cuCimagf(x)/y)); }
static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, cuComplex y) { return(cuCmulf(x, y)); }
static CUDA_BOTH_INLINE cuComplex operator+=(cuComplex &x, cuComplex y) { x = cuCaddf(x, y); return x; };

static CUDA_BOTH_INLINE float cuReal(cuComplex x) { return(cuCrealf(x)); }
static CUDA_BOTH_INLINE float cuImag(cuComplex x) { return(cuCimagf(x)); }
static CUDA_BOTH_INLINE cuComplex cuConj(cuComplex x) { return(cuConjf(x)); }
// clang-format on
/*VCAST_DONT_INSTRUMENT_END*/

template <typename TComplexCompute,
          uint32_t THRD_GRP_SIZE>
__device__ __forceinline__
    TComplexCompute
    thrdGrpAllReduceSum(cg::thread_block_tile<THRD_GRP_SIZE> const& thisThrdGrp, TComplexCompute const& val)
{
    uint32_t        thrdGrpSize = thisThrdGrp.size();
    TComplexCompute sum         = val;
    for(int32_t i = thrdGrpSize / 2; i > 0; i /= 2)
    {
        sum.x += thisThrdGrp.shfl_xor(cuReal(sum), i);
        sum.y += thisThrdGrp.shfl_xor(cuImag(sum), i);
    }
    thisThrdGrp.sync();
    return sum;
}

// Direct memory load version
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS,                  // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS,                   // # of layers (# of cols in H matrix)   
          uint32_t N_TIME_CH_EST,              // # of time-domain channel estimates available (must be >= 1)
          uint32_t THRD_GRP_TILE_SIZE,         // # of thread group tiles per layer
          uint32_t N_THRD_GRP_TILES_PER_LAYER, // # of thread group tiles needed to process a layer
          uint32_t N_PRB_PER_THRD_BLK>         // # of PRBs processed by the thread block
__device__ void
cfoTaEstLowMimoKernel_v2(const puschRxCfoTaEstDynDescr_t &dynDescr)
{
    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;
    
    static_assert((N_TIME_CH_EST >= 2) && (N_TIME_CH_EST <= 4), "The number of time domain channel estimates should lie within 2, 4");

    //--------------------------------------------------------------------------------------------------------
    const uint32_t prbGrpIdx = blockIdx.x;
    const uint32_t ueGrpIdx  = blockIdx.y;

    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[ueGrpIdx];
    
    if((!drvdUeGrpPrms.enableCfoCorrection) && (!drvdUeGrpPrms.enableToEstimation)) return;

    // clang-format off
    tensor_ref<const TComplexStorageIn> tHEst           (drvdUeGrpPrms.tInfoHEst.pAddr                   , drvdUeGrpPrms.tInfoHEst.strides                   );// (N_BS_ANTS, N_LAYERS, NF, NH)
    tensor_ref<TComplexStorageOut>      tCfoEst         (drvdUeGrpPrms.tInfoCfoEst.pAddr                 , drvdUeGrpPrms.tInfoCfoEst.strides                 );// (MAX_ND_SUPPORTED, MAX_N_UE_PER_UE_GRP)
    tensor_ref<float>                   tCfoHz          (drvdUeGrpPrms.tInfoCfoHz.pAddr                  , drvdUeGrpPrms.tInfoCfoHz.strides                  );// (MAX_N_UE = MAX_N_UE_PER_UE_GRP*N_MAX_UE_GRPS)
    tensor_ref<TStorageOut>             tTaEst          (drvdUeGrpPrms.tInfoTaEst.pAddr                  , drvdUeGrpPrms.tInfoTaEst.strides                  );// (MAX_N_UE = MAX_N_UE_PER_UE_GRP*N_MAX_UE_GRPS)
    tensor_ref<volatile TComplexCompute>tCfoPhaseRot    (drvdUeGrpPrms.tInfoCfoPhaseRot.pAddr            , drvdUeGrpPrms.tInfoCfoPhaseRot.strides            );// (MAX_N_TIME_CH_EST, N_MAX_LAYERS, N_MAX_UE_GRPS)
    tensor_ref<volatile TComplexCompute>tTaPhaseRot     (drvdUeGrpPrms.tInfoTaPhaseRot.pAddr             , drvdUeGrpPrms.tInfoTaPhaseRot.strides             );// (N_MAX_LAYERS, N_MAX_UE_GRPS)
    tensor_ref<uint32_t>                tInterCtaSyncCnt(drvdUeGrpPrms.tInfoCfoTaEstInterCtaSyncCnt.pAddr, drvdUeGrpPrms.tInfoCfoTaEstInterCtaSyncCnt.strides);// (N_MAX_UE_GRPS)
    // clang-format on

    const uint32_t nRxAnt  = drvdUeGrpPrms.nRxAnt;
    const uint32_t nLayers = drvdUeGrpPrms.nLayers;
    const uint32_t nCHEST  = drvdUeGrpPrms.dmrsAddlnPos+1;
    
    const uint32_t nPrb       = drvdUeGrpPrms.nPrb;
    const uint32_t dmrsMaxLen = drvdUeGrpPrms.dmrsMaxLen;
    const TCompute deltaFKHz  = static_cast<TCompute>(drvdUeGrpPrms.scsKHz);
    const uint8_t  enableWeightedAverageCfo = drvdUeGrpPrms.enableWeightedAverageCfo;

    // Number of thread blocks needed to process this user group
    uint32_t nThrdBlksNeeded = div_round_up(nPrb, N_PRB_PER_THRD_BLK);

    // The number of thread blocks are sized to process UE group with largest PRB allocation
    // Early exit thread blocks which exceed those needed to process PRBs for current UE group
    if(blockIdx.x >= nThrdBlksNeeded) return;

    // N_THRD_GRP_TILES_PER_LAYER tiles per layer, each tile of size N_THREADS_PER_WARP
    cg::thread_block const& thisThrdBlk = cg::this_thread_block();    
    cg::thread_block_tile<THRD_GRP_TILE_SIZE> const& layerThrdTile =
        cg::tiled_partition<THRD_GRP_TILE_SIZE>(thisThrdBlk);
    
    constexpr uint32_t N_THRDS_PER_LAYER = N_THRD_GRP_TILES_PER_LAYER*THRD_GRP_TILE_SIZE; 
    constexpr uint32_t N_SC_PER_ITER     = CUPHY_N_TONES_PER_PRB; // @todo: use a template specialization so that for high MIMO specialization could process fewer subcarriers
    constexpr uint32_t N_ROWS_H          = N_BS_ANTS;
    constexpr uint32_t N_COLS_H          = N_LAYERS;

    __shared__ TComplexCompute shChEstFreqPhaseRotSum[N_LAYERS];
    __shared__ TComplexCompute shChEstTimePhaseRotSum[N_TIME_CH_EST-1][N_LAYERS]; // N_TIME_CH_EST-1 because 1 phase rotation requires 2 time domain channel estimates
    __shared__ bool            shIsLastCtaDone;
    __shared__ TCompute        shAccumCfoPhase[N_LAYERS]; // This array is indexed by ueIdx, sizing it with N_LAYERS (layer count of UE group) is safe since that is max UE count
    __shared__ TCompute        shAccumTaPhase[N_LAYERS];  // This array is indexed by ueIdx, sizing it with N_LAYERS (layer count of UE group) is safe since that is max UE count    
    __shared__ TComplexCompute smemBlk[N_ROWS_H*N_COLS_H*N_SC_PER_ITER*N_TIME_CH_EST];
    block_4D<TComplexCompute*, N_ROWS_H, N_COLS_H, N_SC_PER_ITER, N_TIME_CH_EST> shHEst(smemBlk);    

    // Number of PRBs already processed before this thread block
    const uint32_t nPrbProcessed       = prbGrpIdx * N_PRB_PER_THRD_BLK;
    const uint32_t startPrbThisThrdBlk = nPrbProcessed;

    // Number of PRBs remaining to be processed from this thread block onwards
    const int32_t nPrbRemaining = nPrb - nPrbProcessed;
    // Calculate loop count - PRBs to be processed by this thread block
    const int32_t nPrbThisThrdBlk = (nPrbRemaining <= N_PRB_PER_THRD_BLK) ? nPrbRemaining : N_PRB_PER_THRD_BLK;

    uint8_t* pPilotSymbPos      = &drvdUeGrpPrms.dmrsSymLoc[0];
    uint8_t* pNUeLayers         = &drvdUeGrpPrms.nUeLayers[0];    
    uint8_t* pUeGrpLayerToUeIdx = &drvdUeGrpPrms.ueGrpLayerToUeIdx[0];
    uint16_t* pAbsUeIdxs        = &drvdUeGrpPrms.ueIdxs[0];
    uint8_t nUes = drvdUeGrpPrms.nUes;

    const uint32_t thrdIdx        = thisThrdBlk.thread_rank();
    const uint32_t thrdIdxInTile  = layerThrdTile.thread_rank(); // Thread index within a tile assigned to a layer
    const uint32_t layerThrdIdx   = threadIdx.x; // Thread index within the layer's thread group (layer thread-group can contain multiple layer thread-tiles)
    const uint32_t nThrds         = thisThrdBlk.size();

    uint32_t layerIdx0       = thrdIdx;    
    uint32_t chEstTimeIdx1   = (thrdIdx % (nCHEST-1));
    uint32_t layerIdx1       = thrdIdx / (nCHEST-1);        
    
#ifdef ENABLE_DEBUG
   // if(0 != prbGrpIdx) return;
    if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
    {
        printf("%s\n: nPrb %d nPrbThisThrdBlk %d\n", __PRETTY_FUNCTION__, nPrb, nPrbThisThrdBlk);
        printf("cfoTaEstLowMimoKernel - tHEst           : addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<TComplexStorageIn*>(drvdUeGrpPrms.tInfoHEst.pAddr), drvdUeGrpPrms.tInfoHEst.strides[0], drvdUeGrpPrms.tInfoHEst.strides[1], drvdUeGrpPrms.tInfoHEst.strides[2], drvdUeGrpPrms.tInfoHEst.strides[3]);
        printf("cfoTaEstLowMimoKernel - tCfoEst         : addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<TStorageOut*>(drvdUeGrpPrms.tInfoCfoEst.pAddr), drvdUeGrpPrms.tInfoCfoEst.strides[0], drvdUeGrpPrms.tInfoCfoEst.strides[1], drvdUeGrpPrms.tInfoCfoEst.strides[2], drvdUeGrpPrms.tInfoCfoEst.strides[3]);
        //printf("cfoTaEstLowMimoKernel - tTaEst          : addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<TStorageOut*>(drvdUeGrpPrms.tInfoTaEstArray[ueGrpIdx].pAddr), drvdUeGrpPrms.tInfoTaEstArray[ueGrpIdx].strides[0], drvdUeGrpPrms.tInfoTaEstArray[ueGrpIdx].strides[1], drvdUeGrpPrms.tInfoTaEstArray[ueGrpIdx].strides[2], drvdUeGrpPrms.tInfoTaEstArray[ueGrpIdx].strides[3]);
        printf("cfoTaEstLowMimoKernel - tCfoPhaseRot    : addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<TComplexCompute*>(drvdUeGrpPrms.tInfoCfoPhaseRot.pAddr), drvdUeGrpPrms.tInfoCfoPhaseRot.strides[0], drvdUeGrpPrms.tInfoCfoPhaseRot.strides[1], drvdUeGrpPrms.tInfoCfoPhaseRot.strides[2], drvdUeGrpPrms.tInfoCfoPhaseRot.strides[3]);
        printf("cfoTaEstLowMimoKernel - tTaPhaseRot     : addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<TComplexCompute*>(drvdUeGrpPrms.tInfoTaPhaseRot.pAddr), drvdUeGrpPrms.tInfoTaPhaseRot.strides[0], drvdUeGrpPrms.tInfoTaPhaseRot.strides[1], drvdUeGrpPrms.tInfoTaPhaseRot.strides[2], drvdUeGrpPrms.tInfoTaPhaseRot.strides[3]);
        printf("cfoTaEstLowMimoKernel - tInterCtaSyncCnt: addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<uint32_t*>(drvdUeGrpPrms.tInfoCfoTaEstInterCtaSyncCnt.pAddr), drvdUeGrpPrms.tInfoCfoTaEstInterCtaSyncCnt.strides[0], drvdUeGrpPrms.tInfoCfoTaEstInterCtaSyncCnt.strides[1], drvdUeGrpPrms.tInfoCfoTaEstInterCtaSyncCnt.strides[2], drvdUeGrpPrms.tInfoCfoTaEstInterCtaSyncCnt.strides[3]);        
    }

#if 0    
    if((0 == thrdIdx) && (0 == prbGrpIdx))
    {
        printf("%s\n: ueGrpIdx %d nPrb %d\n", __PRETTY_FUNCTION__, ueGrpIdx, nPrb);
    }
#endif    
#endif
     
    uint32_t& interCtaSyncCnt = tInterCtaSyncCnt(ueGrpIdx);
    // initialize shared memory used for accumulation
    if(thrdIdx < N_LAYERS)
    {
        if(0 == thrdIdx) shIsLastCtaDone = false;
        shAccumCfoPhase[layerIdx0] = cuGet<TCompute>(0);
        shAccumTaPhase[layerIdx0] = cuGet<TCompute>(0);
        shChEstFreqPhaseRotSum[layerIdx0] = cuGet<TComplexCompute>(0);
    }

    if(thrdIdx < N_LAYERS*(N_TIME_CH_EST-1))
    {
        shChEstTimePhaseRotSum[chEstTimeIdx1][layerIdx1] = cuGet<TComplexCompute>(0);
    }

    __syncthreads();

    //--------------------------------------------------------------------------------------------------------
    // Loop over those PRBs of the UE group processed by this thread block
    const uint32_t bsAntIdx  = thrdIdx % nRxAnt;
    const uint32_t toneIdx   = (thrdIdx/nRxAnt) % CUPHY_N_TONES_PER_PRB;
    const uint32_t layerIdx2 = (thrdIdx/N_THRDS_PER_LAYER) % nLayers;
    for(int32_t prbIdx = 0; prbIdx < nPrbThisThrdBlk; ++prbIdx)
    {
        uint32_t startFreqIdx = (startPrbThisThrdBlk + prbIdx)*N_SC_PER_ITER;

        // Fetch data for this iteration
        for(int32_t chEstTimeIdx = 0; chEstTimeIdx < nCHEST; chEstTimeIdx++)
        {
            if((thrdIdxInTile < (CUPHY_N_TONES_PER_PRB*nRxAnt)) && (layerIdx2 < nLayers))
            {
                shHEst(bsAntIdx, layerIdx2, toneIdx, chEstTimeIdx) = type_convert<TComplexCompute>(tHEst(bsAntIdx, layerIdx2, startFreqIdx + toneIdx, chEstTimeIdx));
            }
        }
        __syncthreads();

        // Compute phase rotations along time (CFO) and frequency (TA)
        TComplexCompute chEstTimePhaseRot[N_TIME_CH_EST-1];
        for(int idx=0; idx<N_TIME_CH_EST-1; idx++)
        {
            chEstTimePhaseRot[idx] = cuGet<TComplexCompute>(0);
        }
        TComplexCompute chEstFreqPhaseRotSum = cuGet<TComplexCompute>(0);   

        #pragma unroll
        for(int32_t chEstTimeIdx = 0; chEstTimeIdx < nCHEST; chEstTimeIdx++)
        {    
            // Time phase ramp measurement
            if((thrdIdxInTile < (CUPHY_N_TONES_PER_PRB*nRxAnt)) && (chEstTimeIdx < (nCHEST-1)))            
            {
                TComplexCompute chEst0 = shHEst(bsAntIdx, layerIdx2, toneIdx, chEstTimeIdx + 0);
                TComplexCompute chEst1 = shHEst(bsAntIdx, layerIdx2, toneIdx, chEstTimeIdx + 1);
                chEstTimePhaseRot[chEstTimeIdx] = chEst1 * cuConj(chEst0);

#ifdef ENABLE_DEBUG
                const uint32_t thrdGrpTileIdx = thrdIdx/THRD_GRP_TILE_SIZE;
                const uint32_t layerThrdGrpTileIdx = thrdGrpTileIdx % nLayers;   
                const uint32_t freqIdx = (startPrbThisThrdBlk + prbIdx)*N_SC_PER_ITER + toneIdx;
                printf("cfoTaEstLowMimoKernel: chEstTimePhaseRot[%d][%d][%d][%d] %f+j%f H0 %f+j%f H1 %f+j%f thrdIdx %d toneIdx %d thrdGrpTileIdx %d layerThrdGrpTileIdx %d\n", bsAntIdx, layerIdx2, freqIdx, ueGrpIdx, cuReal(chEstTimePhaseRot[chEstTimeIdx]), cuImag(chEstTimePhaseRot[chEstTimeIdx]), cuReal(chEst0), cuImag(chEst0), cuReal(chEst1), cuImag(chEst1), thrdIdx, toneIdx, thrdGrpTileIdx, layerThrdGrpTileIdx);
#endif            
            }

            // Frequency phase ramp accumulation 1 (across measurements from channel estimates in time)
            // Measures phase rotation across adjacent tones within a PRB, so skip last toneIdx
            if((layerThrdIdx < (CUPHY_N_TONES_PER_PRB*nRxAnt)) && (toneIdx < (CUPHY_N_TONES_PER_PRB-1)))
            {
                TComplexCompute chEst0 = shHEst(bsAntIdx, layerIdx2, toneIdx + 0, chEstTimeIdx);
                TComplexCompute chEst1 = shHEst(bsAntIdx, layerIdx2, toneIdx + 1, chEstTimeIdx);
                chEstFreqPhaseRotSum += (chEst1 * cuConj(chEst0));

#ifdef ENABLE_DEBUG
                const uint32_t thrdGrpTileIdx = thrdIdx/THRD_GRP_TILE_SIZE;
                const uint32_t layerThrdGrpTileIdx = thrdGrpTileIdx % nLayers;   
                const uint32_t freqIdx = (startPrbThisThrdBlk + prbIdx)*N_SC_PER_ITER + toneIdx;
                printf("cfoTaEstLowMimoKernel: chEstFreqPhaseRotSum[%d][%d][%d][%d] %f+j%f H0 %f+j%f H1 %f+j%f thrdIdx %d toneIdx %d thrdGrpTileIdx %d layerThrdGrpTileIdx %d chEstTimeIdx %d startPrbThisThrdBlk %d thrdAbsIdx %d thrdIdxInTile %d\n", bsAntIdx, layerIdx2, freqIdx, ueGrpIdx, cuReal(chEstFreqPhaseRotSum), cuImag(chEstFreqPhaseRotSum), cuReal(chEst0), cuImag(chEst0), cuReal(chEst1), cuImag(chEst1), thrdIdx, toneIdx, thrdGrpTileIdx, layerThrdGrpTileIdx, chEstTimeIdx, startPrbThisThrdBlk, thisThrdBlk.thread_rank(), thrdIdxInTile);
#endif
           }
        }

        // Time phase ramp accumulation 1 (within PRB and across time domain channel estimates)
        // Reduce within a thread group
        TComplexCompute chEstTimePhaseRotSum[N_TIME_CH_EST-1];
        for(int idx=0; idx<N_TIME_CH_EST-1; idx++)
        {
            chEstTimePhaseRotSum[idx] = cuGet<TComplexCompute>(0);
        }
        #pragma unroll
        for(int32_t chEstTimeIdx = 0; chEstTimeIdx < nCHEST-1; chEstTimeIdx++)
        {
            chEstTimePhaseRotSum[chEstTimeIdx] = thrdGrpAllReduceSum<TComplexCompute, THRD_GRP_TILE_SIZE>(layerThrdTile, chEstTimePhaseRot[chEstTimeIdx]);
        }

        // Frequency phase ramp accumulation 2 (within PRB, across BS antenna and across time domain channel estimates)
        // Note: this accumulation assumes THRD_GRP_TILE_SIZE is equal to CUPHY_N_TONES_PER_PRB
        TComplexCompute chEstFreqPhaseRotSum2 = thrdGrpAllReduceSum<TComplexCompute, THRD_GRP_TILE_SIZE>(layerThrdTile, chEstFreqPhaseRotSum);
        
        __syncthreads();
        
        // Time phase ramp accumulation 2 (Reduce across thread groups belonging to the same layer)
        uint32_t layerThrdGrpThrdIdx = layerThrdTile.thread_rank();
        if(layerThrdGrpThrdIdx < nCHEST-1)
        {
            uint32_t chEstTimeIdx2 = layerThrdGrpThrdIdx;
            /* coverity[overrun-local] This is safe because nCHEST <= N_TIME_CH_EST as a consequence of kernel selection */
            atomicAdd(&shChEstTimePhaseRotSum[chEstTimeIdx2][layerIdx2].x, cuReal(chEstTimePhaseRotSum[chEstTimeIdx2]));
            /* coverity[overrun-local] This is safe because nCHEST <= N_TIME_CH_EST as a consequence of kernel selection */
            atomicAdd(&shChEstTimePhaseRotSum[chEstTimeIdx2][layerIdx2].y, cuImag(chEstTimePhaseRotSum[chEstTimeIdx2]));

#ifdef ENABLE_DEBUG
            const uint32_t thrdGrpTileIdx = thrdIdx/THRD_GRP_TILE_SIZE;
            const uint32_t layerThrdGrpTileIdx = thrdGrpTileIdx % nLayers;        
            printf("cfoTaEstLowMimoKernel: chEstTimePhaseRotSum[%d][%d][%d][%d][%d] %f+j%f shChEstTimePhaseRotSum %f+j%f\n", layerThrdGrpTileIdx, layerIdx2, ueGrpIdx, prbIdx, chEstTimeIdx2, cuReal(chEstTimePhaseRotSum[chEstTimeIdx2]), cuImag(chEstTimePhaseRotSum[chEstTimeIdx2]), cuReal(shChEstTimePhaseRotSum[chEstTimeIdx2][layerIdx2]), cuImag(shChEstTimePhaseRotSum[chEstTimeIdx2][layerIdx2]));
#endif            
        }

        // Accumulate frequency phase ramp across thread group tiles
        if(0 == thrdIdxInTile)
        {
            atomicAdd(&shChEstFreqPhaseRotSum[layerIdx2].x, cuReal(chEstFreqPhaseRotSum2));
            atomicAdd(&shChEstFreqPhaseRotSum[layerIdx2].y, cuImag(chEstFreqPhaseRotSum2));
#ifdef ENABLE_DEBUG            
            printf("cfoTaEstLowMimoKernel: shChEstFreqPhaseRotSum[%d][%d] %f+j%f chEstFreqPhaseRotSum2 %f+j%f thrdIdx %d\n", layerIdx2, ueGrpIdx, cuReal(shChEstFreqPhaseRotSum[layerIdx2]), cuImag(shChEstFreqPhaseRotSum[layerIdx2]), cuReal(chEstFreqPhaseRotSum2), cuImag(chEstFreqPhaseRotSum2), thrdIdx);
#endif                        
        }
    }

    // Complete processing of nPrbIter PRBs - wait for the per layer phase rotation sum to complete within the thread block
    __syncthreads();

    //--------------------------------------------------------------------------------------------------------
    // Sum across thread blocks assigned to this user group (i.e. BS antennas, subcarriers within a PRB and across PRBs)

    // For this last stage of time phase ramp compute, only N_LAYERS*(N_TIME_CH_EST-1) threads are active from each thread block    
    if(thrdIdx < nLayers*(nCHEST-1))
    {
        volatile TComplexCompute& timePhaseRotSum = tCfoPhaseRot(chEstTimeIdx1, layerIdx1, ueGrpIdx);

#ifdef ENABLE_DEBUG        
        printf("cfoTaEstLowMimoKernel: shChEstTimePhaseRotSum[%d][%d][%d] %f+j%f\n", chEstTimeIdx1, layerIdx1, ueGrpIdx, cuReal(shChEstTimePhaseRotSum[chEstTimeIdx1][layerIdx1]), cuImag(shChEstTimePhaseRotSum[chEstTimeIdx1][layerIdx1]), layerIdx1, ueGrpIdx);
#endif
        // atomicAdd takes non-volatile pointers but the operation itself is always as-if volatile
        atomicAdd(const_cast<TCompute*>(&timePhaseRotSum.x), cuReal(shChEstTimePhaseRotSum[chEstTimeIdx1][layerIdx1]));
        atomicAdd(const_cast<TCompute*>(&timePhaseRotSum.y), cuImag(shChEstTimePhaseRotSum[chEstTimeIdx1][layerIdx1]));        
    }

    // For this last stage of frequency phase ramp compute, only N_LAYERS threads are active from each thread block    
    if(thrdIdx < nLayers)
    {
        volatile TComplexCompute& freqPhaseRotSum = tTaPhaseRot(layerIdx0, ueGrpIdx);

#ifdef ENABLE_DEBUG        
        printf("cfoTaEstLowMimoKernel: shChEstFreqPhaseRotSum[%d][%d] %f+j%f\n", layerIdx0, ueGrpIdx, cuReal(shChEstFreqPhaseRotSum[layerIdx0]), cuImag(shChEstFreqPhaseRotSum[layerIdx0]), layerIdx0, ueGrpIdx);
#endif
        // atomicAdd takes non-volatile pointers but the operation itself is always as-if volatile
        atomicAdd(const_cast<TCompute*>(&freqPhaseRotSum.x), cuReal(shChEstFreqPhaseRotSum[layerIdx0]));
        atomicAdd(const_cast<TCompute*>(&freqPhaseRotSum.y), cuImag(shChEstFreqPhaseRotSum[layerIdx0]));        
    }

    // Check for last thread block completion
    if(0 == thrdIdx)
    {
        // Ensure interCtaSyncCnt is incremented only after the atomicAdd operation to global memory has been completed.
        // Note that while the global memory access above is atomic, it does not imply ordering constraints for memory operations,
        // hence a threadfence is still needed.
        __threadfence();

        uint32_t syncCnt = atomicInc(const_cast<uint32_t*>(&interCtaSyncCnt), nThrdBlksNeeded);
        
        // Is this the last CTA to be processed for this user group?        
        shIsLastCtaDone = (syncCnt == (nThrdBlksNeeded - 1)) ? true : false;
#ifdef ENABLE_DEBUG
        printf("cfoTaEstLowMimoKernel: shIsLastCtaDone %u nThrdBlksNeeded %d syncCnt %d interCtaSyncCnt %d\n", shIsLastCtaDone, nThrdBlksNeeded, syncCnt, interCtaSyncCnt);
#endif
    }    
    __syncthreads();

    //--------------------------------------------------------------------------------------------------------    
    // Compute CFO estimate with the thread block that completes last for each user group
    if(shIsLastCtaDone)
    {
        // Compute CFO phase ramp
        if(thrdIdx < nLayers*(nCHEST-1))
        {
            volatile TComplexCompute& timePhaseRotSum = tCfoPhaseRot(chEstTimeIdx1, layerIdx1, ueGrpIdx);
            
            // DMRS additional position symbol index corresponding to channel estimate sample
            uint32_t firstDmrsSymbPosIdx  = (chEstTimeIdx1 + 0) * dmrsMaxLen;
            uint32_t secondDmrsSymbPosIdx = (chEstTimeIdx1 + 1) * dmrsMaxLen;

            // constexpr TCompute TWO_PI = (2.0f * M_PI);
            // CFO frequency calculation involves a divide by 2pi and phase calculation a multiply by 2pi.            
            TCompute cfoPhase = -atan2f(cuImag(timePhaseRotSum), cuReal(timePhaseRotSum)) / (static_cast<TCompute>(pPilotSymbPos[secondDmrsSymbPosIdx] - pPilotSymbPos[firstDmrsSymbPosIdx]));
            uint8_t ueIdx = pUeGrpLayerToUeIdx[layerIdx1];
            /* coverity[overrun-local]  ueIdx <= N_LAYERS based on building UE index map in setup() */
            atomicAdd(const_cast<TCompute*>(&shAccumCfoPhase[ueIdx]), cfoPhase);            
            
#ifdef ENABLE_DEBUG
            printf("cfoTaEstLowMimoKernel: CFO[%d][%d][%d] cfoPhase %f, shAccumCfoPhase %f symPos2 %d symPos1 %d\n", ueIdx, layerIdx1, ueGrpIdx, cfoPhase, shAccumCfoPhase[ueIdx], pPilotSymbPos[secondDmrsSymbPosIdx], pPilotSymbPos[firstDmrsSymbPosIdx]);
#endif
        }

        // Compute Timing advance phase ramp
        if(thrdIdx < nLayers)
        {
            volatile TComplexCompute& freqPhaseRotSum = tTaPhaseRot(layerIdx0, ueGrpIdx);
            
            TCompute taPhase = atan2f(cuImag(freqPhaseRotSum), cuReal(freqPhaseRotSum));
            uint8_t ueIdx = pUeGrpLayerToUeIdx[layerIdx0];
            /* coverity[overrun-local]  ueIdx <= N_LAYERS based on building UE index map in setup() */
            atomicAdd(const_cast<TCompute*>(&shAccumTaPhase[ueIdx]), taPhase);            
            
#ifdef ENABLE_DEBUG
            printf("cfoTaEstLowMimoKernel: TA[%d][%d][%d] taPhase %f, shAccumTaPhase %f deltaFKHz %f avgFreqPhaseRot %f+j%f freqPhaseRotSum %f+j%f\n", ueIdx, layerIdx0, ueGrpIdx, taPhase, shAccumTaPhase[ueIdx], deltaFKHz, cuReal(avgFreqPhaseRot), cuImag(avgFreqPhaseRot), cuReal(freqPhaseRotSum), cuImag(freqPhaseRotSum));
#endif
        }

        __syncthreads();
        
        if(enableWeightedAverageCfo)
        {   tensor_ref<TStorageIn>              tSinrPreEq      (drvdUeGrpPrms.tInfoSinrPreEq.pAddr              , drvdUeGrpPrms.tInfoSinrPreEq.strides              );// (N_UE) 
            for(int ueIdx=thrdIdx; ueIdx < nUes; ueIdx+=nThrds)
            {
                float* buffer = static_cast<float*>(dynDescr.pFoCompensationBuffers[pAbsUeIdxs[ueIdx]]);
                float  foForgetCoeff = drvdUeGrpPrms.foForgetCoeff[ueIdx];
                float  cfoAnglecurr  = -1.0*shAccumCfoPhase[ueIdx]/(pNUeLayers[ueIdx]*(nCHEST-1));
                float  coeffCurr = (1.0f-foForgetCoeff)*nPrb*pow(10.0, (float)tSinrPreEq(ueIdx)/10.0f);
                buffer[0] = foForgetCoeff * buffer[0] + coeffCurr * cosf(cfoAnglecurr);
                buffer[1] = foForgetCoeff * buffer[1] + coeffCurr * sinf(cfoAnglecurr);
                shAccumCfoPhase[ueIdx] = -atan2f(buffer[1], buffer[0])*(pNUeLayers[ueIdx]*(nCHEST-1));
            }
            __syncthreads();
        }

        // Compute CFO estimate per symbol
        // @todo: optimization - compute only at enabled data symbol locations
        const uint32_t nCfoEstIter = div_round_up<uint32_t>(nUes*MAX_ND_SUPPORTED, nThrds);
        for(int32_t iterIdx = 0; iterIdx < nCfoEstIter; ++iterIdx)
        {
            uint32_t idx = (iterIdx*nThrds) + thrdIdx;
            if(idx < (nUes*MAX_ND_SUPPORTED))
            {   
                int32_t symbIdx = idx % MAX_ND_SUPPORTED;
                uint8_t ueIdx   = idx / MAX_ND_SUPPORTED;

                // Average over spatial layers of a UE and time estimates (i.e. time domain channel estimates)
                TCompute cfoAngle = shAccumCfoPhase[ueIdx]*(symbIdx - pPilotSymbPos[0])/(pNUeLayers[ueIdx]*(nCHEST-1));
                TComplexCompute cfoEst = cuGet<TComplexCompute>(cosf(cfoAngle), sinf(cfoAngle));
        
                tCfoEst(symbIdx, ueIdx) = type_convert<TComplexStorageOut>(cfoEst);
                
                if((symbIdx - pPilotSymbPos[0]) == 1)
                {
                    tCfoHz(pAbsUeIdxs[ueIdx]) = (float)((-1.0f*(float)cfoAngle*drvdUeGrpPrms.scsKHz*1000000.0f)/(2.0f*M_PI*15.0f*71.35f));
#ifdef ENABLE_DEBUG
                    printf("Block[%d %d %d]Thread[%d %d %d]UE[%d]Angle[%f]CFO[%f]Hz\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, pAbsUeIdxs[ueIdx], (float)cfoAngle, tCfoHz(pAbsUeIdxs[ueIdx]));
#endif
                }
        
#ifdef ENABLE_DEBUG
                printf("cfoTaEstLowMimoKernel: CFO[%d][%d][%d] %f+j%f, cfoAngle %f\n", symbIdx, ueIdx, ueGrpIdx, cuReal(cfoEst), cuImag(cfoEst), cfoAngle);
#endif
            }
        }    

        if(thrdIdx < nUes)
        {
            uint8_t ueIdx = thrdIdx;
            // Average over spatial layers of a UE and time estimates (i.e. time domain channel estimates)
            TCompute avgTaPhase = shAccumTaPhase[ueIdx]/pNUeLayers[ueIdx];
            
            // Compute TA estimate in units of microseconds
            constexpr TCompute TWO_PI = (2.0f * M_PI);
            TCompute taEst = -avgTaPhase * 1000 / (TWO_PI*static_cast<TCompute>(deltaFKHz));
            tTaEst(pAbsUeIdxs[ueIdx]) = type_convert<TStorageOut>(taEst);

#ifdef ENABLE_DEBUG
            printf("cfoTaEstLowMimoKernel: TA[%d][%d][%d] %f, avgTaPhase %f\n", pAbsUeIdxs[ueIdx], ueGrpIdx, ueIdx, taEst, avgTaPhase);
#endif
        }
    }    
}

//---------------------------------

// Carrier Frequency Offset (CFO) estimation
// {N_LAYERS, N_BS_ANTS} = {1,2}, {2,2}, {1,4}, {2,4}, {4,4}, {1,8}, {2,8} and {4,8}
// Inputs and outputs assumed to be column major
// dimBlock: (N_BS_ANTS*N_LAYERS*CUPHY_N_TONES_PER_PRB)
// dimGrid : (ceil(nMaxprb/N_PRB_PER_THRD_BLK), N_UE_GRPS), Note: ceil(nMaxprb/N_PRB_PER_THRD_BLK) => The number of thread blocks are sized to process UE group with largest PRB allocation
// Note it is assumed that the tCfoPhaseRot and tInterCtaSyncCnt are already pre-initialized to zero

// 1. Divvy up the threads in the thread block into thread tiles for each layer. Each layer's thread
//    tile accumulates phase rotation for all BS antenna and subcarriers within the PRB.
// 2. Multiple PRBs (N_PRB_PER_THRD_BLK) are processed via iterations
// 3. Once all threads within thread block complete accumulation, an average is computed and 
//    accumulated (in global memory) across thread blocks processing a given user group
// 4. The last thread block processing a user group computes the CFO estimate
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS,                  // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS,                   // # of layers (# of cols in H matrix)   
          uint32_t N_TIME_CH_EST,              // # of time-domain channel estimates available (must be >= 1)
          uint32_t THRD_GRP_TILE_SIZE,         // # of thread group tiles per layer
          uint32_t N_THRD_GRP_TILES_PER_LAYER, // # of thread group tiles needed to process a layer
          uint32_t N_PRB_PER_THRD_BLK>         // # of PRBs processed by the thread block
__global__ void
cfoTaEstLowMimoKernel(const __grid_constant__ puschRxCfoTaEstDynDescr_t dynDescr)
{
    cfoTaEstLowMimoKernel_v2<TStorageIn, 
                            TStorageOut, 
                            TCompute, 
                            N_BS_ANTS, 
                            N_LAYERS, 
                            N_TIME_CH_EST, 
                            THRD_GRP_TILE_SIZE, 
                            N_THRD_GRP_TILES_PER_LAYER, 
                            N_PRB_PER_THRD_BLK>(dynDescr);
}

template <uint32_t N_LAYERS,                   // # of layers (# of cols in H matrix)
          uint32_t THRD_GRP_TILE_SIZE,         // # of thread group tiles per layer
          uint32_t N_THRD_GRP_TILES_PER_LAYER, // # of thread group tiles needed to process a layer
          uint32_t N_PRB_PER_THRD_BLK>         // # of PRBs processed by the thread block
void puschRxCfoTaEst::cfoTaEstLowMimoKernelLaunchGeo(uint16_t nMaxPrb,
                                                     uint16_t nUeGrps,
                                                     dim3&    gridDim,
                                                     dim3&    blockDim)
{
    constexpr uint32_t N_THRDS_PER_LAYER  = N_THRD_GRP_TILES_PER_LAYER*THRD_GRP_TILE_SIZE;
    const uint32_t nThrdBlksPerUeGrp = static_cast<uint32_t>(std::ceil(static_cast<float>(nMaxPrb) / static_cast<float>(N_PRB_PER_THRD_BLK)));

    gridDim  = dim3(nThrdBlksPerUeGrp, nUeGrps);
    blockDim = dim3(N_THRDS_PER_LAYER, N_LAYERS);

#ifdef ENABLE_DEBUG
    NVLOGI_FMT(NVLOG_PUSCH, "{}: blockDim ({},{},{}), gridDim ({},{},{})", __FUNCTION__, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
#endif
}

template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS,     // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS,      // # of layers (# of cols in H matrix)          
          uint32_t N_TIME_CH_EST> // # of time-domain channel estimates available
void puschRxCfoTaEst::cfoTaEstLowMimo(uint16_t                         nMaxPrb,
                                      uint16_t                         nUeGrps,
                                      cuphyPuschRxCfoTaEstLaunchCfg_t& launchCfg)
{    
    // Each thread block in one iteration processes one PRB (all BS antenna i.e. Rx branches and subcarriers of the PRB)
    // Roundup the thread count to be a multiple of thread group tile size (warp size)
    constexpr uint32_t THRD_GRP_TILE_SIZE         = N_THREADS_PER_WARP;
    constexpr uint32_t N_THRD_GRP_TILES_PER_LAYER = (((N_BS_ANTS * CUPHY_N_TONES_PER_PRB) + THRD_GRP_TILE_SIZE - 1)/THRD_GRP_TILE_SIZE);
    
    // # of PRBs to be processed by each thread block
    constexpr uint32_t N_PRB_PER_THRD_BLK         = 8;

    void* kernelFunc = reinterpret_cast<void*>(cfoTaEstLowMimoKernel<TStorageIn,
                                                                   TStorageOut,
                                                                   TCompute,
                                                                   N_BS_ANTS,
                                                                   N_LAYERS,
                                                                   N_TIME_CH_EST,
                                                                   THRD_GRP_TILE_SIZE,
                                                                   N_THRD_GRP_TILES_PER_LAYER,
                                                                   N_PRB_PER_THRD_BLK>);
    
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;
    {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));}

    dim3 blockDim, gridDim;
    cfoTaEstLowMimoKernelLaunchGeo<N_LAYERS,
                                 THRD_GRP_TILE_SIZE,
                                 N_THRD_GRP_TILES_PER_LAYER,
                                 N_PRB_PER_THRD_BLK>(nMaxPrb, nUeGrps, gridDim, blockDim);

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;
    
    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra          = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = 0;
}

template <typename TStorageIn, typename TStorageOut, typename TCompute, uint32_t N_TIME_CH_EST>
void puschRxCfoTaEst::kernelSelectL0(uint16_t                          nBSAnts,
                                     uint8_t                           nLayers,
                                     uint16_t                          nMaxPrb,
                                     uint16_t                          nUeGrps,
                                     cuphyPuschRxCfoTaEstLaunchCfg_t&  launchCfg)
{
    bool noKernelFound = false;
    // Low MIMO regime
    if((8 >= nBSAnts) && (1 <= nBSAnts))
    {
        switch(nBSAnts)
        {
            // nBSAnts == 8
            case 8:
            {
                constexpr uint32_t N_BS_ANTS = 8; // # of BS antenna (# of rows in H matrix)
                switch(nLayers)
                {
                    // nLayers == 8 not supported since not enough shared memory
                    case 8:
                    {
                        constexpr uint32_t N_LAYERS = 8; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }

                    // nLayers == 7
                    case 7:
                    {
                        constexpr uint32_t N_LAYERS = 7; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }

                    // nLayers == 6
                    case 6:
                    {
                        constexpr uint32_t N_LAYERS = 6; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }

                    // nLayers == 5
                    case 5:
                    {
                        constexpr uint32_t N_LAYERS = 5; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }

                    // nLayers == 4
                    case 4:
                    {
                        constexpr uint32_t N_LAYERS = 4; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }

                    // nLayers == 3
                    case 3:
                    {
                        constexpr uint32_t N_LAYERS = 3; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }

                    // nLayers == 2
                    case 2:
                    {
                        constexpr uint32_t N_LAYERS = 2; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }
                        // nLayers == 1
                    case 1:
                    {
                        constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                      TStorageOut,
                                      TCompute,
                                      N_BS_ANTS,
                                      N_LAYERS,
                                      N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }
                    default:
                    {
                        noKernelFound = true;
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {}", __FUNCTION__, nBSAnts, nLayers);
                        break;
                    }
                }
                break;
            }
            
            // nBSAnts == 7
            case 7:
            {
                constexpr uint32_t N_BS_ANTS = 7; // # of BS antenna (# of rows in H matrix)
                switch(nLayers)
                {
                   // nLayers == 7
                    case 7:
                    {
                        constexpr uint32_t N_LAYERS = 7; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }

                    // nLayers == 6
                    case 6:
                    {
                        constexpr uint32_t N_LAYERS = 6; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }

                    // nLayers == 5
                    case 5:
                    {
                        constexpr uint32_t N_LAYERS = 5; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }

                    // nLayers == 4
                    case 4:
                    {
                        constexpr uint32_t N_LAYERS = 4; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }

                    // nLayers == 3
                    case 3:
                    {
                        constexpr uint32_t N_LAYERS = 3; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }

                    // nLayers == 2
                    case 2:
                    {
                        constexpr uint32_t N_LAYERS = 2; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }
                        // nLayers == 1
                    case 1:
                    {
                        constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                      TStorageOut,
                                      TCompute,
                                      N_BS_ANTS,
                                      N_LAYERS,
                                      N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }
                    default:
                    {
                        noKernelFound = true;
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {}", __FUNCTION__, nBSAnts, nLayers);
                        break;
                    }
                }
                break;
            }
            
            // nBSAnts == 6
            case 6:
            {
                constexpr uint32_t N_BS_ANTS = 6; // # of BS antenna (# of rows in H matrix)
                switch(nLayers)
                {
                    // nLayers == 6
                    case 6:
                    {
                        constexpr uint32_t N_LAYERS = 6; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }

                    // nLayers == 5
                    case 5:
                    {
                        constexpr uint32_t N_LAYERS = 5; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }

                    // nLayers == 4
                    case 4:
                    {
                        constexpr uint32_t N_LAYERS = 4; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }

                    // nLayers == 3
                    case 3:
                    {
                        constexpr uint32_t N_LAYERS = 3; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }

                    // nLayers == 2
                    case 2:
                    {
                        constexpr uint32_t N_LAYERS = 2; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }
                        // nLayers == 1
                    case 1:
                    {
                        constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                      TStorageOut,
                                      TCompute,
                                      N_BS_ANTS,
                                      N_LAYERS,
                                      N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }
                    default:
                    {
                        noKernelFound = true;
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {}", __FUNCTION__, nBSAnts, nLayers);
                        break;
                    }
                }
                break;
            }
            
            // nBSAnts == 5
            case 5:
            {
                constexpr uint32_t N_BS_ANTS = 5; // # of BS antenna (# of rows in H matrix)
                switch(nLayers)
                {                    
                    // nLayers == 5
                    case 5:
                    {
                        constexpr uint32_t N_LAYERS = 5; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }

                    // nLayers == 4
                    case 4:
                    {
                        constexpr uint32_t N_LAYERS = 4; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }

                    // nLayers == 3
                    case 3:
                    {
                        constexpr uint32_t N_LAYERS = 3; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }

                    // nLayers == 2
                    case 2:
                    {
                        constexpr uint32_t N_LAYERS = 2; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }
                        // nLayers == 1
                    case 1:
                    {
                        constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                      TStorageOut,
                                      TCompute,
                                      N_BS_ANTS,
                                      N_LAYERS,
                                      N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }
                    default:
                    {
                        noKernelFound = true;
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {}", __FUNCTION__, nBSAnts, nLayers);
                        break;
                    }
                }
                break;
            }
    
            // nBSAnts == 4
            case 4:
            {
                constexpr uint32_t N_BS_ANTS = 4; // # of BS antenna (# of rows in H matrix)
                switch(nLayers)
                {
                    // nLayers == 4
                    case 4:
                    {
                        constexpr uint32_t N_LAYERS = 4; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);        
                        break;
                    }

                    // nLayers == 3
                    case 3:
                    {
                        constexpr uint32_t N_LAYERS = 3; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }

                    // nLayers == 2
                    case 2:
                    {
                        constexpr uint32_t N_LAYERS = 2; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }
        
                    // nLayers == 1
                    case 1:
                    {
                        constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }
                    default:
                    {
                        noKernelFound = true;
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {}", __FUNCTION__, nBSAnts, nLayers);
                        break;
                    }
                }
                break;
            }
            
            // nBSAnts == 3
            case 3:
            {
                constexpr uint32_t N_BS_ANTS = 3; // # of BS antenna (# of rows in H matrix)
                switch(nLayers)
                {
                    // nLayers == 3
                    case 3:
                    {
                        constexpr uint32_t N_LAYERS = 3; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }

                    // nLayers == 2
                    case 2:
                    {
                        constexpr uint32_t N_LAYERS = 2; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }
        
                    // nLayers == 1
                    case 1:
                    {
                        constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }
                    default:
                    {
                        noKernelFound = true;
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {}", __FUNCTION__, nBSAnts, nLayers);
                        break;
                    }
                }
                break;
            }
    
            // nBSAnts == 2
            case 2:
            {
                constexpr uint32_t N_BS_ANTS = 2; // # of BS antenna (# of rows in H matrix)
                switch(nLayers)
                {
                    // nLayers == 2
                    case 2:
                    {
                        constexpr uint32_t N_LAYERS = 2; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }
        
                    // nLayers == 1
                    case 1:
                    {
                        constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }
    
                    default:
                    {
                        noKernelFound = true;
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {}", __FUNCTION__, nBSAnts, nLayers);
                        break;
                    }
                }
                break;
            }
            
            // nBSAnts == 1
            case 1:
            {
                constexpr uint32_t N_BS_ANTS = 1; // # of BS antenna (# of rows in H matrix)
                switch(nLayers)
                {
                    // nLayers == 1
                    case 1:
                    {
                        constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
                        cfoTaEstLowMimo<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_TIME_CH_EST>(nMaxPrb, nUeGrps, launchCfg);
                        break;
                    }
    
                    default:
                    {  
                        noKernelFound = true;
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {}", __FUNCTION__, nBSAnts, nLayers);
                        break;
                    }
                }
                break;
            }

            /*VCAST_DONT_INSTRUMENT_START*/
            default:
            {
                noKernelFound = true;
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {}", __FUNCTION__, nBSAnts, nLayers);
                break;
            }
	    /*VCAST_DONT_INSTRUMENT_END*/
        }
    }
    else
    {
        noKernelFound = true;
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts", __FUNCTION__, nBSAnts);
    }

    if(noKernelFound)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {}", __FUNCTION__, nBSAnts, nLayers);
    }
}

template <typename TStorageIn, typename TStorageOut, typename TCompute>
void puschRxCfoTaEst::kernelSelectL1(uint16_t                         nBSAnts,
                                     uint8_t                          nLayers,
                                     uint8_t                          nDmrsAddlnPos,
                                     uint16_t                         nMaxPrb,
                                     uint16_t                         nUeGrps,
                                     cuphyPuschRxCfoTaEstLaunchCfg_t& launchCfg)
{
    switch(nDmrsAddlnPos)
    {
        case 1:
        {
            constexpr uint32_t N_TIME_CH_EST = 2;
            kernelSelectL0<TStorageIn, 
                           TStorageOut, 
                           TCompute, 
                           N_TIME_CH_EST>(nBSAnts,
                                          nLayers,
                                          nMaxPrb,
                                          nUeGrps,
                                          launchCfg);
            break; 
        }
        case 2:
        {
            constexpr uint32_t N_TIME_CH_EST = 3;
            kernelSelectL0<TStorageIn, 
                           TStorageOut, 
                           TCompute, 
                           N_TIME_CH_EST>(nBSAnts,
                                          nLayers,
                                          nMaxPrb,
                                          nUeGrps,
                                          launchCfg);
            break; 
        }
        case 3:
        {
            constexpr uint32_t N_TIME_CH_EST = 4;
            kernelSelectL0<TStorageIn, 
                           TStorageOut, 
                           TCompute, 
                           N_TIME_CH_EST>(nBSAnts,
                                          nLayers,
                                          nMaxPrb,
                                          nUeGrps,
                                          launchCfg);
            break; 
        }
        default:
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nDmrsAddlnPos {}", __FUNCTION__, nDmrsAddlnPos);
            break;
        }        
    }
}

void puschRxCfoTaEst::kernelSelectL2(uint16_t                         nBSAnts,
                                     uint8_t                          nLayers,
                                     uint8_t                          nDmrsAddlnPos,
                                     uint16_t                         nMaxPrb,
                                     uint16_t                         nUeGrps,
                                     cuphyDataType_t                  hEstType,
                                     cuphyDataType_t                  cfoEstType,
                                     cuphyPuschRxCfoTaEstLaunchCfg_t& launchCfg)
{
    using TCompute = float;
    if((CUPHY_C_32F == hEstType))
    {
        using TStorageIn = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
        if(CUPHY_C_32F == cfoEstType)
        {
            using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
            kernelSelectL1<TStorageIn, TStorageOut, TCompute>(nBSAnts,
                                                              nLayers,
                                                              nDmrsAddlnPos,
                                                              nMaxPrb,
                                                              nUeGrps,
                                                              launchCfg);
        }
        else if(CUPHY_C_16F == cfoEstType)
        {
            using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
            kernelSelectL1<TStorageIn, TStorageOut, TCompute>(nBSAnts,
                                                              nLayers,
                                                              nDmrsAddlnPos,
                                                              nMaxPrb,
                                                              nUeGrps,
                                                              launchCfg);
        }
        else
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type - HEst {} CfoEst {}", __FUNCTION__, cuphyGetDataTypeString(hEstType), cuphyGetDataTypeString(cfoEstType));
        }
    }
    else if((CUPHY_C_16F == hEstType) && (CUPHY_C_16F == cfoEstType))
    {
        using TStorageIn  = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        kernelSelectL1<TStorageIn, TStorageOut, TCompute>(nBSAnts,
                                                          nLayers,
                                                          nDmrsAddlnPos,
                                                          nMaxPrb,
                                                          nUeGrps,
                                                          launchCfg);
    }
    else
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type - HEst {} CfoEst {}", __FUNCTION__, cuphyGetDataTypeString(hEstType), cuphyGetDataTypeString(cfoEstType));
    }
}

void puschRxCfoTaEst::init(bool                        enableCpuToGpuDescrAsyncCpy,
                           puschRxCfoTaEstStatDescr_t& statDescrCpu,
                           void*                       pStatDescrGpu,
                           cudaStream_t                strm)
{  
}

void puschRxCfoTaEst::getDescrInfo(size_t& statDescrSizeBytes, size_t& statDescrAlignBytes, size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes)
{
    statDescrSizeBytes  = sizeof(puschRxCfoTaEstStatDescr_t);
    statDescrAlignBytes = alignof(puschRxCfoTaEstStatDescr_t);

    dynDescrSizeBytes  = sizeof(puschRxCfoTaEstDynDescrVec_t);
    dynDescrAlignBytes = alignof(puschRxCfoTaEstDynDescrVec_t);
}

cuphyStatus_t 
puschRxCfoTaEst::setup(cuphyPuschRxUeGrpPrms_t*          pDrvdUeGrpPrmsCpu,
                       cuphyPuschRxUeGrpPrms_t*          pDrvdUeGrpPrmsGpu,
                       float**                           pFoCompensationBuffers,
                       uint16_t                          nUeGrps,
                       uint32_t                          nMaxPrb,
                       bool                              enableCpuToGpuDescrAsyncCpy,
                       puschRxCfoTaEstDynDescrVec_t&     dynDescrVecCpu,
                       void*                             pDynDescrsGpu,
                       cuphyPuschRxCfoTaEstLaunchCfgs_t* pLaunchCfgs,
                       cudaStream_t                      strm)
{
    puschRxCfoTaEstDynDescr_t* pDynDescrVecGpu = static_cast<puschRxCfoTaEstDynDescr_t*>(pDynDescrsGpu);

    if(!pDrvdUeGrpPrmsCpu || !pDrvdUeGrpPrmsGpu || !pDynDescrVecGpu || !pLaunchCfgs) return CUPHY_STATUS_INVALID_ARGUMENT;
    
    uint32_t nMaxRxAnt = 1;
    uint32_t nMaxLayers = 0;
    uint32_t maxDmrsAddlnPos = 0;
    
    for(uint32_t idxUeGrps = 0; idxUeGrps < nUeGrps; idxUeGrps++)
    {
        if(nMaxRxAnt < pDrvdUeGrpPrmsCpu[idxUeGrps].nRxAnt)
            nMaxRxAnt = pDrvdUeGrpPrmsCpu[idxUeGrps].nRxAnt;
        if(nMaxLayers < pDrvdUeGrpPrmsCpu[idxUeGrps].nLayers)
            nMaxLayers = pDrvdUeGrpPrmsCpu[idxUeGrps].nLayers;
        if(maxDmrsAddlnPos < pDrvdUeGrpPrmsCpu[idxUeGrps].dmrsAddlnPos)
            maxDmrsAddlnPos = pDrvdUeGrpPrmsCpu[idxUeGrps].dmrsAddlnPos; 
        if(nMaxLayers > nMaxRxAnt)
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Number of layers are greater than number of receive antennas.", __FUNCTION__);
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
    }

    pLaunchCfgs->nCfgs = 0;
    // CUPHY_PUSCH_RX_CFO_EST_N_MAX_HET_CFGS = 1
    for(uint32_t hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_CFO_EST_N_MAX_HET_CFGS; ++hetCfgIdx)
    {
        // Setup descriptor in CPU memory
        puschRxCfoTaEstDynDescr_t& dynDescr = dynDescrVecCpu[hetCfgIdx];

        // TODO: this needs to be per UE group
        // # of time domain channel estimates is equal to the number of DMRS additional positions + 1
        // Need atlest two time domain channel estimates to compute CFO
        //if(0 == pDrvdUeGrpPrmsCpu[0].dmrsAddlnPos) continue;
        if(0 == maxDmrsAddlnPos) continue;

        dynDescr.nUeGrps        = nUeGrps;
        dynDescr.pDrvdUeGrpPrms = pDrvdUeGrpPrmsGpu;
        dynDescr.pFoCompensationBuffers = pFoCompensationBuffers;

        puschRxCfoTaEstKernelArgs_t& kernelArgs = m_kernelArgsArr[hetCfgIdx];
        kernelArgs.pDynDescr = &pDynDescrVecGpu[hetCfgIdx];

        // Because the kernel is now grid constant, we do not need to copy the descriptor to GPU memory

        // Select kernel
        cuphyPuschRxCfoTaEstLaunchCfg_t& launchCfg = pLaunchCfgs->cfgs[hetCfgIdx];
        kernelSelectL2(nMaxRxAnt,
                       nMaxLayers,
                       maxDmrsAddlnPos,
                       nMaxPrb,
                       nUeGrps,
                       pDrvdUeGrpPrmsCpu[0].tInfoHEst.elemType,
                       pDrvdUeGrpPrmsCpu[0].tInfoCfoEst.elemType,
                       launchCfg);

        launchCfg.kernelArgs[0] = &dynDescr;
        
        launchCfg.kernelNodeParamsDriver.kernelParams = &(launchCfg.kernelArgs[0]);

        pLaunchCfgs->nCfgs++;
    }
    return CUPHY_STATUS_SUCCESS;
}

} // namespace cfo_ta_est
