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
#include "cuComplex.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include "cuphy_kernel_util.cuh"
#include "pusch_rssi.hpp"
#include "type_convert.hpp"
#include "nvlog.hpp"
#include "cuphy.hpp"
// using namespace cooperative_groups;
namespace cg = cooperative_groups;

namespace puschRx_rssi
{
// #define ENABLE_PROFILING
// #define ENABLE_DEBUG

/*VCAST_DONT_INSTRUMENT_START*/

static constexpr uint32_t N_THREADS_PER_WARP                = 32;

template <typename TElem>
struct tensor_ref
{
    TElem*         pAddr;
    const int32_t* strides;

    CUDA_BOTH
    tensor_ref(void* pAddr, const int32_t* pStrides) :
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
    CUDA_BOTH int offset(int i0, int i1, int i2, int i3, int i4) const
    {
        return (strides[0] * i0) + (strides[1] * i1) + (strides[2] * i2) + (strides[3] * i3) + (strides[4] * i4);
    };    
    // clang-format off
    CUDA_BOTH TElem&       operator()(int i0)                                 { return *(pAddr + offset(i0));                 }
    CUDA_BOTH TElem&       operator()(int i0, int i1)                         { return *(pAddr + offset(i0, i1));             }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2)                 { return *(pAddr + offset(i0, i1, i2));         }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2, int i3)         { return *(pAddr + offset(i0, i1, i2, i3));     }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2, int i3, int i4) { return *(pAddr + offset(i0, i1, i2, i3, i4)); }

    CUDA_BOTH const TElem& operator()(int i0) const                                 { return *(pAddr + offset(i0));                 }
    CUDA_BOTH const TElem& operator()(int i0, int i1) const                         { return *(pAddr + offset(i0, i1));             }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2) const                 { return *(pAddr + offset(i0, i1, i2));         }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3) const         { return *(pAddr + offset(i0, i1, i2, i3));     }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3, int i4) const { return *(pAddr + offset(i0, i1, i2, i3, i4)); }

    // clang-format on
#if 0    
    if(std::is_const<TElem>::value) {
      CUDA_BOTH const TElem& operator()(int i0) const                         { return __ldg(addr + offset(i0));         }
      CUDA_BOTH const TElem& operator()(int i0, int i1) const                 { return __ldg(addr + offset(i0, i1));     }
      CUDA_BOTH const TElem& operator()(int i0, int i1, int i2) const         { return __ldg(addr + offset(i0, i1, i2)); }
      CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3) const { return __ldg(addr + offset(i0, i1, i2, i3)); }
    }
    else {
      CUDA_BOTH const TElem& operator()(int i0) const                         { return *(addr + offset(i0));         }
      CUDA_BOTH const TElem& operator()(int i0, int i1) const                 { return *(addr + offset(i0, i1));     }
      CUDA_BOTH const TElem& operator()(int i0, int i1, int i2) const         { return *(addr + offset(i0, i1, i2)); }
      CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3) const { return *(addr + offset(i0, i1, i2, i3)); }
    }
#endif
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

static CUDA_BOTH_INLINE float     cuReal(cuComplex x) { return(cuCrealf(x)); }
// static CUDA_BOTH_INLINE float     cuImag(cuComplex x) { return(cuCimagf(x)); }
static CUDA_BOTH_INLINE cuComplex cuConj(cuComplex x) { return(cuConjf(x)); }

static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, cuComplex y)   { return(cuCmulf(x, y)); }
// clang-format on

template <typename TCompute,
          uint32_t THRD_GRP_SIZE>
__device__ __forceinline__
    TCompute
    thrdGrpAllReduceSum(cg::thread_block_tile<THRD_GRP_SIZE> const& thisThrdGrp, TCompute const& val)
{
    uint32_t        thrdGrpSize = thisThrdGrp.size();
    TCompute sum         = val;
    for(int32_t i = thrdGrpSize / 2; i > 0; i /= 2)
    {
        sum += thisThrdGrp.shfl_xor(sum, i);
    }
    if constexpr (THRD_GRP_SIZE == 32) {
        __syncwarp();
    } else {
        thisThrdGrp.sync();
    }
    return sum;
}
/*VCAST_DONT_INSTRUMENT_END*/

// FAPI requirement: "RSSI reported will be total received power summed across all antennas".
// Average across allocated PRBs and measured symbols. Sum across Rx antennas
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t THRD_GRP_TILE_SIZE,            // # of threads in a thread group tile
          uint32_t N_THRD_GRP_TILES_PER_THRD_BLK, // number of thread group tiles in the thread block
          uint32_t N_ITER_PER_THRD_BLK,           // # of iterations in a thread block
          uint8_t  DMRS_SYMBOL_IDX>           
__global__ void rssiMeasKernel(puschRxRssiDynDescr_t* pDynDescr)
 {
    static_assert(((CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS == DMRS_SYMBOL_IDX) || (CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS == DMRS_SYMBOL_IDX) || (CUPHY_PUSCH_RSSI_EST_FIRST_DMRS == DMRS_SYMBOL_IDX)), "Each kernel processes different DMRS symbols");
    
    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;    

    //--------------------------------------------------------------------------------------------------------    
    puschRxRssiDynDescr_t& dynDescr = *(pDynDescr);
    uint32_t ueGrpIdx       = blockIdx.y;    
    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[ueGrpIdx];

#if ENABLE_DEBUG
    if((0 == blockIdx.x) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))                           
    printf("%s\n grid = (%u %u %u), block = (%u %u %u) ueGrpIdx %u\n",
                           __PRETTY_FUNCTION__,
                           gridDim.x, gridDim.y, gridDim.z,
                           blockDim.x, blockDim.y, blockDim.z,
                           ueGrpIdx);
#endif

    // clang-format off
    uint16_t startPrb = drvdUeGrpPrms.startPrb;
    uint16_t nPrb     =  drvdUeGrpPrms.nPrb;
    tensor_ref<const TComplexStorageIn>tDataRx         (drvdUeGrpPrms.tInfoDataRx.pAddr             , drvdUeGrpPrms.tInfoDataRx.strides             );// (NF, ND, N_BS_ANTS)
    tensor_ref<volatile TStorageOut>   tRssiFull       (drvdUeGrpPrms.tInfoRssiFull.pAddr           , drvdUeGrpPrms.tInfoRssiFull.strides           );// (ND, N_BS_ANTS, N_UE_GRPS)
    tensor_ref<volatile TStorageOut>   tRssi           (drvdUeGrpPrms.tInfoRssi.pAddr               , drvdUeGrpPrms.tInfoRssi.strides               );// (N_UE_GRPS)
    tensor_ref<volatile TStorageOut>   tRssiEhq        (drvdUeGrpPrms.tInfoRssiEhq.pAddr            , drvdUeGrpPrms.tInfoRssiEhq.strides            );// (N_UE_GRPS)
    tensor_ref<uint32_t>               tInterCtaSyncCnt(drvdUeGrpPrms.tInfoRssiInterCtaSyncCnt.pAddr, drvdUeGrpPrms.tInfoRssiInterCtaSyncCnt.strides);// (N_UE_GRPS)
    // clang-format on

    const uint32_t nSc      = nPrb*CUPHY_N_TONES_PER_PRB;
    
    cg::thread_block const& thisThrdBlk = cg::this_thread_block();    
    cg::thread_block_tile<THRD_GRP_TILE_SIZE> const& thrdGrpTiles =
        cg::tiled_partition<THRD_GRP_TILE_SIZE>(thisThrdBlk);
    
    uint8_t nSymb = drvdUeGrpPrms.nDmrsSyms;
    if(CUPHY_PUSCH_RSSI_EST_FIRST_DMRS == DMRS_SYMBOL_IDX)
    {
        nSymb = drvdUeGrpPrms.dmrsMaxLen;
    }
    else if(CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS == DMRS_SYMBOL_IDX)
    {
        nSymb = nSymb - drvdUeGrpPrms.dmrsMaxLen;
    }
    const uint16_t nRxAnt = drvdUeGrpPrms.nRxAnt;
    
    const uint32_t thrdIdx  = thisThrdBlk.thread_rank();
    const uint32_t nThrds = thisThrdBlk.size();
      
    uint32_t symbIdx  = blockIdx.z % nSymb;
    uint32_t rxAntIdx = (blockIdx.z / nSymb) % nRxAnt;
    
    uint32_t symbIdxOffset = 0;
    if(CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS == DMRS_SYMBOL_IDX)
    {
        symbIdxOffset = drvdUeGrpPrms.dmrsMaxLen;
    }
    
    const uint32_t symbLocIdx = drvdUeGrpPrms.dmrsSymLoc[symbIdx+symbIdxOffset]; // tSymbLoc(symbIdx);

    const uint32_t startSc = startPrb * CUPHY_N_TONES_PER_PRB;
    uint32_t& interCtaSyncCnt = tInterCtaSyncCnt(ueGrpIdx);    

    __shared__ bool     isLastCtaDone;
    __shared__ TCompute totalPwr;
    
#ifdef ENABLE_DEBUG
    if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
    {
        printf("rssiMeasKernel - startPrb       : %d\n", startPrb);
        printf("rssiMeasKernel - nPrb         : %d\n", nPrb);
        printf("rssiMeasKernel - tSymbLoc        : addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<uint16_t*>(dynDescr.tPrmSymbLoc.pAddr), dynDescr.tPrmSymbLoc.strides[0], dynDescr.tPrmSymbLoc.strides[1], dynDescr.tPrmSymbLoc.strides[2], dynDescr.tPrmSymbLoc.strides[3]);
        printf("rssiMeasKernel - tDataRx         : addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<uint16_t*>(drvdUeGrpPrms.tInfoDataRx.pAddr), dynDescr.[ueGrpIdx].tInfoDataRx.strides[0], drvdUeGrpPrms.tInfoDataRx.strides[1], drvdUeGrpPrms.tInfoDataRx.strides[2], drvdUeGrpPrms.tInfoDataRx.strides[3]);
        printf("rssiMeasKernel - tRssiFull       : addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<uint16_t*>(drvdUeGrpPrms.tInfoRssiFull.pAddr), drvdUeGrpPrms.tInfoRssiFull.strides[0], drvdUeGrpPrms.tInfoRssiFull.strides[1], drvdUeGrpPrms.tInfoRssiFull.strides[2], drvdUeGrpPrms.tInfoRssiFull.strides[3]);
        printf("rssiMeasKernel - tRssi           : addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<uint16_t*>(drvdUeGrpPrms.tInfoRssi.pAddr), drvdUeGrpPrms.tInfoRssi.strides[0], drvdUeGrpPrms.tInfoRssi.strides[1], drvdUeGrpPrms.tInfoRssi.strides[2], drvdUeGrpPrms.tInfoRssi.strides[3]);
        printf("rssiMeasKernel - tInterCtaSyncCnt: addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<uint32_t*>(drvdUeGrpPrms.tInfoRssiInterCtaSyncCnt.pAddr), drvdUeGrpPrms.tInfoRssiInterCtaSyncCnt.strides[0], drvdUeGrpPrms.tInfoRssiInterCtaSyncCnt.strides[1], drvdUeGrpPrms.tInfoRssiInterCtaSyncCnt.strides[2], drvdUeGrpPrms.tInfoRssiInterCtaSyncCnt.strides[3]);        
    }

    if((0 == thrdIdx) && (0 == blockIdx.x))
    {
        printf("%s\n: ueGrpIdx %d nSymb %d nRxAnt %d symbIdx %d symbLocIdx %d rxAntIdx %d startSc %d nSc %d\n", __PRETTY_FUNCTION__, ueGrpIdx, nSymb, nRxAnt, symbIdx, symbLocIdx, rxAntIdx, startSc, nSc);
    }
#endif

    // Initialization
    if(0 == thrdIdx) 
    {
        isLastCtaDone = false;
        totalPwr = cuGet<TCompute>(0);
    }
    __syncthreads();
    
    //--------------------------------------------------------------------------------------------------------                                                                          
    // grid-x stride reduction for pwr  
    if(blockIdx.z < (nSymb*nRxAnt)) 
    {                                                                                                                                               
        TCompute pwr = cuGet<TCompute>(0);
        for( uint32_t scIdx = thrdIdx + blockIdx.x*nThrds; scIdx < nSc; scIdx += nThrds*gridDim.x ) {
          uint32_t absScIdx = startSc + scIdx;
          TComplexCompute rxSample = type_convert<TComplexCompute>(tDataRx(absScIdx, symbLocIdx, rxAntIdx));
          pwr += cuReal(rxSample*cuConj(rxSample));
    
#ifdef ENABLE_DEBUG
          printf("rssiMeasKernel: rxSample[%d][%d][%d][%d] %f+j%f pwr %f\n",
    	     scIdx, symbLocIdx, rxAntIdx, ueGrpIdx, cuReal(rxSample), cuImag(rxSample), cuReal(rxSample*cuConj(rxSample)) );
#endif                   
    
        }
    
        // Accumulate power within a thread block                                                                                                                                           
        TCompute accumPwr = thrdGrpAllReduceSum<TCompute, THRD_GRP_TILE_SIZE>(thrdGrpTiles, pwr);
        if(0 == thrdGrpTiles.thread_rank()) {
          atomicAdd(&totalPwr, accumPwr);
        }
    }

    __syncthreads();

    //--------------------------------------------------------------------------------------------------------                                                                          
    // Logic to ensure all thread blocks in the grid complete                                                                                                                           
    if(0 == thrdIdx) {

      // accumulate contributions from each thread block                                                                                                                                
      atomicAdd(const_cast<TCompute*>(tRssiFull.pAddr + tRssiFull.offset(symbIdx+symbIdxOffset, rxAntIdx, ueGrpIdx)), totalPwr);   // ueGrpIdx = f(block.y) symbIdx & rxAntIdx are f(block.z)         
      atomicAdd(const_cast<TCompute*>(tRssi.pAddr + tRssi.offset(ueGrpIdx))                           , totalPwr);

      // Ensure interCtaSyncCnt is incremented only after the tRssiFull global memory write has been completed.                                                                         
      // Note that while the global memory access above is atomic, it does not imply ordering constraints for memory operations,                                                        
      // hence a threadfence is still needed.                                                                                                                                           
      __threadfence();

      uint32_t syncCnt = atomicInc(const_cast<uint32_t*>(&interCtaSyncCnt), gridDim.x * gridDim.z);

      // Is this the last CTA to be processed for this user group?                                                                                                                      
      isLastCtaDone = (syncCnt == (gridDim.x*gridDim.z - 1)) ? true : false;

#ifdef ENABLE_DEBUG
      printf("rssiMeasKernel: symbIdx %d rxAntIdx %d ueGrpIdx %d isLastCtaDone %u syncCnt %d interCtaSyncCnt %d\n",
	     symbIdx, rxAntIdx, ueGrpIdx, isLastCtaDone, syncCnt, interCtaSyncCnt);
#endif

      if ( isLastCtaDone ) {
        TCompute rssiLin = type_convert<TCompute>(tRssi(ueGrpIdx));
        uint8_t nSymbDivisor = nSymb;
        if(CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS == DMRS_SYMBOL_IDX)
        {
            nSymbDivisor = nSymb + drvdUeGrpPrms.dmrsMaxLen;
        }
        TCompute rssidB = 10*log10f(rssiLin/nSymbDivisor);
        if(CUPHY_PUSCH_RSSI_EST_FIRST_DMRS == DMRS_SYMBOL_IDX)
        {
            tRssiEhq(ueGrpIdx) = type_convert<TStorageOut>(rssidB);
            interCtaSyncCnt = 0; //reset for full-slot processing  
        }
        else
            tRssi(ueGrpIdx) = type_convert<TStorageOut>(rssidB);

#ifdef ENABLE_DEBUG
	printf("rssiMeasKernel: Rssi: db[%d] %07.4f (Lin: %07.4f) nUeGrps %d blockIdx = (%u %u %u), threadIdx = (%u %u %u)\n",
	       ueGrpIdx, rssiDb, rssiLin, nUeGrps, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
#endif

      }

    }

    __syncthreads();

    // Use one thread block per UE group to do the accumulations/averaging for collapsing rssiFull(nDmrsSymb, nRxAnt, nUeGrps) to rssi(nUeGrps)                                         
    // Note: RssiFull is converted from linear to dB here (after its used in rssi calculation)                                                                                          
    // [SR] The point is that only one block per ue sees isLastCtaDone as true.                                                                                                         
    if(isLastCtaDone) {

        for ( uint32_t i = thrdIdx; i<nSymb*nRxAnt; i+=nThrds ) {

          symbIdx = i%nSymb;
          rxAntIdx = i/nSymb;

          // Convert to dB scale                                                                                                                                                        
          TCompute rssiFullLin = type_convert<TCompute>(tRssiFull(symbIdx+symbIdxOffset, rxAntIdx, ueGrpIdx));
          TCompute rssiFullDb = 10*log10f(rssiFullLin);
          tRssiFull(symbIdx+symbIdxOffset, rxAntIdx, ueGrpIdx) = type_convert<TStorageOut>(rssiFullDb);

#ifdef ENABLE_DEBUG
	  printf("rssiMeasKernel: rssiFullDb[%d][%d][%d] %07.4f (Lin: %07.4f)\n", symbIdx, rxAntIdx, ueGrpIdx, rssiFullDb, rssiFullLin);
#endif
	  
        }

    }
 
 }
 

template<uint32_t THRD_GRP_TILE_SIZE,            // # of threads in a thread group tile
         uint32_t N_THRD_GRP_TILES_PER_THRD_BLK, // number of thread group tiles in the thread block
         uint32_t N_ITER_PER_THRD_BLK>           // # of iterations in a thread block
void puschRxRssi::rssiMeasLaunchGeo(uint16_t nMaxPrb, 
                                    uint8_t  nSymb, 
                                    uint16_t nRxAnt,
                                    uint16_t nUeGrps,
                                    dim3&    gridDim,
                                    dim3&    blockDim)
{
    // Max subcarrier count
    uint32_t nMaxSc = nMaxPrb*CUPHY_N_TONES_PER_PRB;
    uint32_t nThrdGrpTiles = N_THRD_GRP_TILES_PER_THRD_BLK;

    // Calculate the number of tiles of size THRD_GRP_TILE_SIZE in the thread block
    // Each thread processes one subcarrier
    uint32_t nThrdGrpTilesNeeded = div_round_up<uint32_t>(nMaxSc, THRD_GRP_TILE_SIZE*N_ITER_PER_THRD_BLK);

    // Number of thread blocks needed to process a UE group with nMaxPrb allocation
    uint32_t nThrdBlks = 1;
    if(nThrdGrpTilesNeeded > nThrdGrpTiles)
    {
        nThrdBlks = div_round_up<uint32_t>(nThrdGrpTilesNeeded, N_THRD_GRP_TILES_PER_THRD_BLK);
    }

    blockDim.x = THRD_GRP_TILE_SIZE;
    blockDim.y = nThrdGrpTiles;
    blockDim.z = 1;

    gridDim.x = nThrdBlks;
    gridDim.y = nUeGrps;
    gridDim.z = nSymb*nRxAnt;

#ifdef ENABLE_DEBUG
    NVLOGI_FMT(NVLOG_PUSCH, "{}: Thread block dim: blockDim ({},{},{}), gridDim ({},{},{})", __FUNCTION__, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
#endif
}

template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute>      
void puschRxRssi::rssiMeas(uint16_t                     nMaxPrb,
                           uint16_t                     nSymb,
                           uint16_t                     nRxAnt,
                           uint16_t                     nUeGrps,
                           uint8_t                      dmrsSymbolIdx,
                           cuphyPuschRxRssiLaunchCfg_t& launchCfg)
{
    static constexpr uint32_t THRD_GRP_TILE_SIZE                = N_THREADS_PER_WARP;
    static constexpr uint32_t N_THRD_GRP_TILES_PER_THRD_BLK     = 4; // 4*32 = 128 threads per thread block
    // # of iterations within a thread block (amortize launch and housekeeping overhead)
    static constexpr uint32_t N_ITER_PER_THRD_BLK               = 4;
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;
    
    if(dmrsSymbolIdx==CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS)
    {
        static constexpr uint8_t DMRS_SYMBOL_IDX    = CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS;
        void* kernelFunc = reinterpret_cast<void*>(rssiMeasKernel<TStorageIn,
                                                                  TStorageOut,
                                                                  TCompute,
                                                                  THRD_GRP_TILE_SIZE,
                                                                  N_THRD_GRP_TILES_PER_THRD_BLK,
                                                                  N_ITER_PER_THRD_BLK,
                                                                  DMRS_SYMBOL_IDX>);
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));}
    }
    else if(dmrsSymbolIdx==CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS)
    {
        static constexpr uint8_t DMRS_SYMBOL_IDX    = CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS;
        void* kernelFunc = reinterpret_cast<void*>(rssiMeasKernel<TStorageIn,
                                                                  TStorageOut,
                                                                  TCompute,
                                                                  THRD_GRP_TILE_SIZE,
                                                                  N_THRD_GRP_TILES_PER_THRD_BLK,
                                                                  N_ITER_PER_THRD_BLK,
                                                                  DMRS_SYMBOL_IDX>);
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));}
    }
    else if(dmrsSymbolIdx==CUPHY_PUSCH_RSSI_EST_FIRST_DMRS)
    {
        static constexpr uint8_t DMRS_SYMBOL_IDX    = CUPHY_PUSCH_RSSI_EST_FIRST_DMRS;
        void* kernelFunc = reinterpret_cast<void*>(rssiMeasKernel<TStorageIn,
                                                                  TStorageOut,
                                                                  TCompute,
                                                                  THRD_GRP_TILE_SIZE,
                                                                  N_THRD_GRP_TILES_PER_THRD_BLK,
                                                                  N_ITER_PER_THRD_BLK,
                                                                  DMRS_SYMBOL_IDX>);
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));}
    }
    
    dim3 blockDim, gridDim;
    rssiMeasLaunchGeo<THRD_GRP_TILE_SIZE,
                      N_THRD_GRP_TILES_PER_THRD_BLK,
                      N_ITER_PER_THRD_BLK>(nMaxPrb, nSymb, nRxAnt, nUeGrps, gridDim, blockDim);

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;
    
    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra          = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = 0;
}

void puschRxRssi::rssiMeasKernelSelect(uint16_t                     nMaxPrb,
                                       uint16_t                     nSymb,
                                       uint16_t                     nRxAnt,
                                       uint16_t                     nUeGrps,
                                       uint8_t                      dmrsSymbolIdx,
                                       cuphyDataType_t              dataRxType,
                                       cuphyDataType_t              rssiFullType,
                                       cuphyDataType_t              rssiType,
                                       cuphyPuschRxRssiLaunchCfg_t& launchCfg)
{
    // Note: Output tensors tRssiFull, tRssi used in global memory accumulation. Hence using single precision
    // clang-format off

    // clang-format off
    using TCompute = float;    
    if(CUPHY_C_32F == dataRxType)
    {
        using TStorageIn = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;        
        if((CUPHY_R_32F == rssiType) && (CUPHY_R_32F == rssiFullType))
        {
            using TStorageOut = data_type_traits<CUPHY_R_32F>::type;
            rssiMeas<TStorageIn, TStorageOut, TCompute>(nMaxPrb,
                                                        nSymb,
                                                        nRxAnt,
                                                        nUeGrps,
                                                        dmrsSymbolIdx,
                                                        launchCfg);
        }
        else
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type - DataRx {} RssiFull {} Rssi {}", __FUNCTION__, cuphyGetDataTypeString(dataRxType), cuphyGetDataTypeString(rssiFullType), cuphyGetDataTypeString(rssiType));
        }
    }
    else if((CUPHY_C_16F == dataRxType) && (CUPHY_R_32F == rssiType) && (CUPHY_R_32F == rssiFullType))
    {
        using TStorageIn  = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        using TStorageOut = data_type_traits<CUPHY_R_32F>::type; 
        rssiMeas<TStorageIn, TStorageOut, TCompute>(nMaxPrb,
                                                    nSymb,
                                                    nRxAnt,
                                                    nUeGrps,
                                                    dmrsSymbolIdx,
                                                    launchCfg);
    }
    else
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type - DataRx {} RssiFull {} Rssi {}", __FUNCTION__, cuphyGetDataTypeString(dataRxType), cuphyGetDataTypeString(rssiFullType), cuphyGetDataTypeString(rssiType));
    }
}

cuphyStatus_t puschRxRssi::setupRssiMeas(cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                         cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                         uint16_t                      nUeGrps,
                                         uint32_t		                   nMaxPrb,
                                         uint8_t                       dmrsSymbolIdx,
                                         uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                         puschRxRssiDynDescrVec_t&     dynDescrVecCpu,
                                         void*                         pDynDescrsGpu,
                                         cuphyPuschRxRssiLaunchCfgs_t* pLaunchCfgs,
                                         cudaStream_t                  strm)
{
    puschRxRssiDynDescr_t* pDynDescrVecGpu = static_cast<puschRxRssiDynDescr_t*>(pDynDescrsGpu);

    if(!pDrvdUeGrpPrmsCpu || !pDrvdUeGrpPrmsGpu || !pDynDescrVecGpu || !pLaunchCfgs) return CUPHY_STATUS_INVALID_ARGUMENT;
    
    if((dmrsSymbolIdx!=CUPHY_PUSCH_RSSI_EST_FIRST_DMRS)&&(dmrsSymbolIdx!=CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS)&&(dmrsSymbolIdx!=CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS))
    {
         NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Invalid dmrsSymbolIdx {}", __FUNCTION__, dmrsSymbolIdx);
         return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    for(uint32_t hetCfgIdx = 0; hetCfgIdx < pLaunchCfgs->nCfgs; ++hetCfgIdx)
    {
        // Setup descriptor in CPU memory
        puschRxRssiDynDescr_t& dynDescr = dynDescrVecCpu[hetCfgIdx];

        uint16_t nMaxRxAnt = 1;
        uint16_t nMaxDmrsSym = 1;
        uint8_t  dmrsMaxLen = pDrvdUeGrpPrmsCpu[0].dmrsMaxLen;
        // Setup measurement symbol location indices
        for(uint16_t ueGrpIdx = 0; ueGrpIdx < nUeGrps; ueGrpIdx++)
        {
            if(nMaxRxAnt < pDrvdUeGrpPrmsCpu[ueGrpIdx].nRxAnt)
            {
                nMaxRxAnt = pDrvdUeGrpPrmsCpu[ueGrpIdx].nRxAnt;
            }
            
            
            uint16_t symbCnt = 0;
            uint32_t symbLocBmsk = pDrvdUeGrpPrmsCpu[ueGrpIdx].rssiSymPosBmsk;
            if(!symbLocBmsk)
                continue;
            for(uint8_t i = 0; i < MAX_ND_SUPPORTED; ++i)
            {
                if((symbLocBmsk >> i) & 0x1)
                {
                    symbCnt++;
                }
            }
            if(nMaxDmrsSym < symbCnt)
            {
                nMaxDmrsSym = symbCnt;
            }
        }
        
        if(dmrsSymbolIdx == CUPHY_PUSCH_RSSI_EST_FIRST_DMRS)
        {
            nMaxDmrsSym = dmrsMaxLen;
        }
        else if(dmrsSymbolIdx == CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS)
        {
            nMaxDmrsSym = nMaxDmrsSym - dmrsMaxLen;
        }
        
        dynDescr.pDrvdUeGrpPrms  = pDrvdUeGrpPrmsGpu;

        puschRxRssiKernelArgs_t& kernelArgs = (dmrsSymbolIdx==CUPHY_PUSCH_RSSI_EST_FIRST_DMRS) ? m_rssiKernelArgsArr[0][hetCfgIdx] :
((dmrsSymbolIdx==CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS) ? m_rssiKernelArgsArr[1][hetCfgIdx] : m_rssiKernelArgsArr[2][hetCfgIdx]);

        kernelArgs.pDynDescr = &pDynDescrVecGpu[hetCfgIdx];

        // Optional descriptor copy to GPU memory
        if(enableCpuToGpuDescrAsyncCpy)
        {
            CUDA_CHECK(cudaMemcpyAsync(&pDynDescrVecGpu[hetCfgIdx], &dynDescr, sizeof(puschRxRssiDynDescr_t), cudaMemcpyHostToDevice, strm));
        }

        // Select kernel
        //TODO: supporting variable antenna counts
        cuphyPuschRxRssiLaunchCfg_t& launchCfg = pLaunchCfgs->cfgs[hetCfgIdx];        
        rssiMeasKernelSelect(nMaxPrb,
                             nMaxDmrsSym,
                             nMaxRxAnt,
                             nUeGrps,
                             dmrsSymbolIdx,
                             pDrvdUeGrpPrmsCpu[0].tInfoDataRx.elemType,
                             pDrvdUeGrpPrmsCpu[0].tInfoRssiFull.elemType,
                             pDrvdUeGrpPrmsCpu[0].tInfoRssi.elemType,
                             launchCfg);    
        
        launchCfg.kernelArgs[0] = &kernelArgs.pDynDescr;        
        launchCfg.kernelNodeParamsDriver.kernelParams = &(launchCfg.kernelArgs[0]);
    }    
    return CUPHY_STATUS_SUCCESS;
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// RSRP measurement
// RSRP - Reference Signal Received Power is computed by averaging the channel estimate allocated PRBs, Rx antenna, time domain estimates and sum across layers
// Kernel also computes post equalization noise variance, pre and post equalization SINR
// #define ENABLE_DEBUG
//                                                                                                                                                                                          
// grid.x -> spans subcarriers                                                                                                                                                              
// grid.y -> spans layers / ueIdx                                                                                                                                                           
// grid.z -> 1 / not used

// Direct global memory loads of channel estimates
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t THRD_GRP_TILE_SIZE,            // # of threads in a thread group tile
          uint32_t N_THRD_GRP_TILES_PER_THRD_BLK, // number of thread group tiles in the thread block
          uint32_t N_SC_PER_THRD_BLK_ITER,        // number of subcarriers processed per thread block iteration
          uint32_t N_ITER_PER_THRD_BLK,           // # of iterations in a thread block
          uint8_t  DMRS_SYMBOL_IDX>
__global__ void rsrpMeasKernel_v1(puschRxRsrpDynDescr_t* pDynDescr)
 {

    KERNEL_PRINT_GRID_ONCE("%s\n grid = (%u %u %u), block = (%u %u %u), THRD_GRP_TILE_SIZE %u, N_THRD_GRP_TILES_PER_THRD_BLK %u, N_SC_PER_THRD_BLK_ITER %u, N_ITER_PER_THRD_BLK %u\n",
                           __PRETTY_FUNCTION__,
                           gridDim.x, gridDim.y, gridDim.z,
                           blockDim.x, blockDim.y, blockDim.z,
                           THRD_GRP_TILE_SIZE,
                           N_THRD_GRP_TILES_PER_THRD_BLK,
                           N_SC_PER_THRD_BLK_ITER,
                           N_ITER_PER_THRD_BLK);

#if 0                           
    if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))                           
    printf("%s\n grid = (%u %u %u), block = (%u %u %u), THRD_GRP_TILE_SIZE %u, N_THRD_GRP_TILES_PER_THRD_BLK %u, N_SC_PER_THRD_BLK_ITER %u, N_ITER_PER_THRD_BLK %u\n",
                           __PRETTY_FUNCTION__,
                           gridDim.x, gridDim.y, gridDim.z,
                           blockDim.x, blockDim.y, blockDim.z,
                           THRD_GRP_TILE_SIZE,
                           N_THRD_GRP_TILES_PER_THRD_BLK,
                           N_SC_PER_THRD_BLK_ITER,
                           N_ITER_PER_THRD_BLK);
#endif

    static_assert(((CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS == DMRS_SYMBOL_IDX) || (CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS == DMRS_SYMBOL_IDX)), "Each kernel processes all DMRS symbols or all DMRS symbols except the first DMRS symbol for full-slot");

    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;    

    //--------------------------------------------------------------------------------------------------------    
    puschRxRsrpDynDescr_t& dynDescr = *(pDynDescr);
    const uint32_t ueGrpIdx         = blockIdx.y;
    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[ueGrpIdx];
    
    // clang-format off
    tensor_ref<const TComplexStorageIn> tHEst              (drvdUeGrpPrms.tInfoHEst.pAddr               , drvdUeGrpPrms.tInfoHEst.strides               );// (N_BS_ANTS, N_LAYERS, NF, NH)
    tensor_ref<const TStorageIn>        tReeDiagInv        (drvdUeGrpPrms.tInfoReeDiagInv.pAddr         , drvdUeGrpPrms.tInfoReeDiagInv.strides         );// (N_SC, N_LAYERS, N_PRB, NH)   // (N_LAYERS, NF, NH)    
    tensor_ref<const TStorageIn>        tNoiseIntfVarPreEq (drvdUeGrpPrms.tInfoNoiseVarPreEq.pAddr      , drvdUeGrpPrms.tInfoNoiseVarPreEq.strides      ); // (N_UE_GRP)
    tensor_ref<volatile TStorageOut>    tRsrp              (drvdUeGrpPrms.tInfoRsrp.pAddr               , drvdUeGrpPrms.tInfoRsrp.strides               );// (N_UE)
    tensor_ref<volatile TStorageOut>    tNoiseIntfVarPostEq(drvdUeGrpPrms.tInfoNoiseVarPostEq.pAddr     , drvdUeGrpPrms.tInfoNoiseVarPostEq.strides     ); // (N_UE_GRP)    
    tensor_ref<TStorageOut>             tSinrPreEq         (drvdUeGrpPrms.tInfoSinrPreEq.pAddr          , drvdUeGrpPrms.tInfoSinrPreEq.strides          );// (N_UE)
    tensor_ref<TStorageOut>             tSinrPostEq        (drvdUeGrpPrms.tInfoSinrPostEq.pAddr         , drvdUeGrpPrms.tInfoSinrPostEq.strides         );// (N_UE)
    tensor_ref<uint32_t>                tInterCtaSyncCnt   (drvdUeGrpPrms.tInfoRsrpInterCtaSyncCnt.pAddr, drvdUeGrpPrms.tInfoRsrpInterCtaSyncCnt.strides);// (N_UE)
    // clang-format on

    uint32_t nUes       = drvdUeGrpPrms.nUes; // Number of UEs in this UE group
    uint16_t nPrb       = drvdUeGrpPrms.nPrb;
    uint32_t nSc        = nPrb*CUPHY_N_TONES_PER_PRB;
    uint16_t nRxAnt     = drvdUeGrpPrms.nRxAnt;  // # of BS antenna (# of rows in H matrix)
    uint16_t nLayers    = drvdUeGrpPrms.nLayers; // # of layers (# of cols in H matrix)   
    uint16_t nTimeChEst = drvdUeGrpPrms.dmrsAddlnPos + 1; // # of time-domain channel estimates available

    uint8_t* pUeGrpLayerToUeIdx = &drvdUeGrpPrms.ueGrpLayerToUeIdx[0];
    uint16_t* pAbsUeIdxs        = &drvdUeGrpPrms.ueIdxs[0];
    uint8_t* pNUeLayers         = &drvdUeGrpPrms.nUeLayers[0];

    cg::thread_block const& thisThrdBlk = cg::this_thread_block();
    cg::thread_block_tile<THRD_GRP_TILE_SIZE> const& thisThrdTile =
      cg::tiled_partition<THRD_GRP_TILE_SIZE>(thisThrdBlk);
    uint32_t thrdIdx  = thisThrdBlk.thread_rank();
    uint32_t nThrds = thisThrdBlk.size();    
    uint32_t totalThreads = nThrds * gridDim.x * gridDim.z;         // number of threads in the grid                                                                                        
    uint32_t thrdGrdIdx = thrdIdx+blockIdx.x*nThrds;                // thread index within the grid                                                                                         

    uint32_t& interCtaSyncCnt = tInterCtaSyncCnt(ueGrpIdx);

    __shared__ bool isLastCtaDone;

    // Flexible shared memory tensor
    extern __shared__ TCompute shAccumInvRee[];
    TCompute *shLayerPwr = shAccumInvRee + nLayers;
    
#ifdef ENABLE_DEBUG
    if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
    {
        printf("%s\n", __PRETTY_FUNCTION__);
	printf("rsrpMeasKernel - nPrb            : %d\n", nPrb);
        printf("rsrpMeasKernel - nRxAnt          : %d\n", nRxAnt);
        printf("rsrpMeasKernel - nLayers         : %d\n", nLayers);
        printf("rsrpMeasKernel - nTimeChEst      : %d\n", nTimeChEst);
        printf("rsrpMeasKernel - tHEst           : addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n",
               static_cast<uint16_t*>(drvdUeGrpPrms.tInfoHEst.pAddr), drvdUeGrpPrms.tInfoHEst.strides[0], drvdUeGrpPrms.tInfoHEst.strides[1], drvdUeGrpPrms.tInfoHEst.strides[2], drvdUeGrp\
Prms.tInfoHEst.strides[3]);
        printf("rsrpMeasKernel - tRsrp           : addr 0x%llx strides[0] %d\n", static_cast<uint16_t*>(drvdUeGrpPrms.tInfoRsrp.pAddr), drvdUeGrpPrms.tInfoRsrp.strides[0]);
        printf("rsrpMeasKernel - tInterCtaSyncCnt: addr 0x%llx strides[0] %d\n", static_cast<uint32_t*>(drvdUeGrpPrms.tInfoRsrpInterCtaSyncCnt.pAddr), drvdUeGrpPrms.tInfoRsrpInterCtaSyncC\
nt.strides[0]);
    }
    // if((0 == thrdIdx) && (0 == blockIdx.x))                                                                                                                                              
    {
        printf("rsrpMeasKernel - blockIdx (%d %d %d) threadIdx (%d %d %d) ueGrpIdx %d nRxAnt %d nLayers %d nTimeChEst %d nSc %d\n",
               blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, ueGrpIdx, nRxAnt, nLayers, nTimeChEst, nSc);
    }
#endif

    // Initialization
    if(0 == thrdIdx)
    {
        isLastCtaDone = false;
    }
    if(thrdIdx < nLayers)
    {
      shLayerPwr[thrdIdx]    = cuGet<TCompute>(0);
      shAccumInvRee[thrdIdx] = cuGet<TCompute>(0);
    }

    __syncthreads();

    
    //--------------------------------------------------------------------------------------------------------                 
    //                                                                                                                                                                                      
    // Reduction is done using a grid.x-stride loop.  As the data is arranged with layer as an intermediate                                                                                 
    // index, tHEst(nRxAnt,nLayer,nSc,nTimeChEst) and tReeDiagInv(nReeSc, nLayers, nReePrb) the approach is to                                                                              
    // use all grid-x threads to span an integer number of 'x-y planes' of data in the structure.  By using                                                                                 
    // that many threads (i.e. integer multiple of first two grid dimensions), and using a grid stride of that                                                                              
    // size, the data can be accessed in a perfectly coalesed mannar, and the Layer index of the data never                                                                                 
    // changes within a thread - so the bulk of the reduction can be performed within the thread.                                                                                           
    //                                                                                                                                                                                      
    // Shared memory is used to reduce layer data between threads (because there is no guarantee that the                                                                                   
    // array dimensions will be convenient powers of 2?).                                                                                                                                   

    // Accumulate power -----                                                                                                                                                               
    uint32_t iLayer;
    TCompute pwr;
    uint32_t stride = (totalThreads / (nLayers*nRxAnt))*nLayers*nRxAnt;

    pwr            = cuGet<TCompute>(0);
    if ( thrdGrdIdx < stride ) {
      
      TComplexCompute signal;
      iLayer         = (thrdGrdIdx/nRxAnt)%nLayers;
      uint32_t loopLimit      = nRxAnt * nLayers * nSc * nTimeChEst;
      const cuComplex* pStart = &tHEst(0,0,0,0);
      
      uint32_t startIdx = 0;
      if(DMRS_SYMBOL_IDX==CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS)
      {
          startIdx = nRxAnt * nLayers * nSc;
      }

      // this is the long loop over all [nRxAnt * nLayers * nSc * nTimeChEst]                                                                                                               
      for ( uint32_t idx=startIdx + thrdGrdIdx; idx<loopLimit; idx+=stride ) 
      {
        signal = type_convert<TComplexCompute>( *(pStart+idx) );
        pwr   += cuReal(signal*cuConj(signal));

#ifdef ENABLE_DEBUG
        uint32_t rxAntIdx = idx%nRxAnt;
        uint32_t layerIdx = (idx/nRxAnt)%nLayers;
        uint32_t scIdx = (idx/(nRxAnt*nLayers))%nSc;
        uint32_t chExtTimeIdx = (idx/(nRxAnt*nLayers*nSc));
        TComplexCompute refSample = shHEst(rxAntIdx, layerIdx, scIdx, chEstTimeIdx, 0);
        printf("rsrpMeasKernel: ueGrpIdx %d thrdIdx %d layerIdx %d rxAntIdx %d chEstTimeIdx %d shRefSample %07.4f + j %07.04f refSample %07.4f + j %07.04f\n",
               ueGrpIdx, thrdIdx, layerIdx, rxAntIdx, chEstTimeIdx, cuReal(refSample), cuImag(refSample),
               cuReal(tHEst(rxAntIdx, layerIdx, startScCurrItr + scIdx, chEstTimeIdx)), cuImag(tHEst(rxAntIdx, layerIdx, startScCurrItr + scIdx, chEstTimeIdx)));
#endif

      }

    }
    
    // Reduce, per-layer, within warp, then ccumulate between warps in smem using atomics
    for ( int jLayer=0; jLayer<nLayers; ++jLayer) {
      TCompute sum = 0;
      if ( iLayer == jLayer ) sum = pwr;
      TCompute sumpwr = cg::reduce(thisThrdTile, sum, cg::plus<TCompute>());
      if ( threadIdx.x == 0 ) {
	atomicAdd(shLayerPwr+jLayer, sumpwr);
      }
    }
    

    // Accumulate Ree -----                                                                                                                                                                 
    stride = (totalThreads / (nLayers*CUPHY_N_TONES_PER_PRB) ) * nLayers * CUPHY_N_TONES_PER_PRB;
    uint16_t dmrsIdx = (0 != drvdUeGrpPrms.enablePuschTdi) ? (nTimeChEst - 1) : 0;  

    TCompute sumInvRee = cuGet<TCompute>(0);
    if ( thrdGrdIdx < stride ) {
      iLayer = thrdGrdIdx / CUPHY_N_TONES_PER_PRB;
      iLayer = iLayer % nLayers;
      uint32_t loopLimit = nSc * nLayers;
      const TCompute* pStart = &tReeDiagInv(0,0,0);

      for ( uint32_t idx=thrdGrdIdx; idx < loopLimit; idx+=stride ) {
        sumInvRee += type_convert<TCompute>(*(pStart+idx+dmrsIdx*loopLimit)); 

#ifdef ENABLE_DEBUG
        uint32_t scIdx = (idx / ( nRxAnt * nLayers ) ) % nSc;
        uint32_t reeScIdx = scIdx % CUPHY_N_TONES_PER_PRB;
        uint32_t layerIdx = (idx / nRxAnt) % nLayers;
        uint32_t reePrbIdx = scIdx / CUPHY_N_TONES_PER_PRB;
        printf("rsrpMeasKernel: Ree[%d][%d][%d] %07.4f accumRee %07.4f\n", reeScIdx, layerIdx, reePrbIdx, tReeDiagInv(reeScIdx, layerIdx, reePrbIdx), sumInvRee);
#endif

      }
    }
      
    // Reduce, per-layer, within warp, then accumulate between threads in smem using atomics
    for ( int jLayer=0; jLayer<nLayers; ++jLayer ) {
      TCompute sum = 0;
      if ( iLayer == jLayer ) sum = sumInvRee;
      TCompute sumInvRee2 = cg::reduce(thisThrdTile, sum, cg::plus<TCompute>());
      if (threadIdx.x == 0 ) {
	atomicAdd(shAccumInvRee + jLayer, sumInvRee2);
      }
    }
      
    
    // ensure all accumulations are complete                                                                                                                                                
    __syncthreads();

    if ( thrdIdx < nLayers ) {

      uint8_t ueIdx      = pUeGrpLayerToUeIdx[thrdIdx];
      uint16_t absUeIdx  = pAbsUeIdxs[ueIdx];

      atomicAdd(const_cast<TCompute*>(tRsrp.pAddr + tRsrp.offset(absUeIdx)), shLayerPwr[thrdIdx]);
      atomicAdd(const_cast<TCompute*>(tSinrPostEq.pAddr + tSinrPostEq.offset(absUeIdx)), shAccumInvRee[thrdIdx]);

    }

    __syncthreads();
    
    //--------------------------------------------------------------------------------------------------------                                                                              
    // Logic to ensure all thread blocks in the grid complete (ending in writes to tRsrp above)                                                                                             
    if(0 == thrdIdx)
    {
        // Ensure interCtaSyncCnt is incremented only after the tRsrp global memory write has been completed.                                                                               
        // Note that while the global memory access above is atomic, it does not imply ordering constraints for memory operations,                                                          
        // hence a threadfence is still needed.                                                                                                                                             
        __threadfence();

        //uint32_t syncCnt = atomicInc(const_cast<uint32_t*>(&interCtaSyncCnt), nUeGrpThrdBlksNeeded);                                                                                      
        // NOTE: the following is atomicInc, vs. atomicAdd.  gridDim.x*gridDim.z is just an upper limit.                                                                                    
        // SR - gridDim.x * gridDim.z (since gridDim.y == ueGrpIdx for this kernel) is just more readable than nUeGrpThrdBlksNeeded                                                         
        //      ... once you know all gridIdx.y's are different reductions.  Note gridDim.z is 1.  So, could just use gridDim.x.                                                            
        // SR - i.e. gridDim.x * gridDim.z is all the blocks, so gridDim.x * gridDim.z - 1 is definitely the last block.                                                                    
        uint32_t syncCnt = atomicInc(const_cast<uint32_t*>(&interCtaSyncCnt), gridDim.x*gridDim.z);

        // Is this the last CTA to be processed for this user group?                                                                                                                        
        isLastCtaDone = (syncCnt == (gridDim.x*gridDim.z - 1)) ? true : false;

#ifdef ENABLE_DEBUG
        printf("rsrpMeasKernel: blockIdx(x y z) %d %d %d threadIdx(x y z) %d %d %d ueGrpIdx %d isLastCtaDone %u nUeGrpThrdBlksNeeded %d syncCnt %d interCtaSyncCnt %d\n",
               blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, ueGrpIdx, isLastCtaDone, gridDim.x*gridDim.z, syncCnt, interCtaSyncCnt);
#endif

    }

    __syncthreads();
    
    // Use one thread block per UE group to compute the average RSRP value per UE group rsrp(nUeGrps), compute SINR                                                                         
    // Note: Rsrp is converted from linear to dB here (after its used in rsrp calculation)                                                                                                  
    // this is worth 2.5 us.                                                                                                                                                                
    if(isLastCtaDone)
    {
      for(uint32_t i = thrdIdx; i < nUes; i+=nThrds)
        {
          // Average the RSRP measurement: average across allocated PRBs, Rx antenna, time domain estimates (and sum across layers)                                                         
          uint16_t absUeIdx = pAbsUeIdxs[i];
          uint8_t nUeLayers = pNUeLayers[i];

          // Note: We don't average across layers                                                                                                                                           
          TCompute rsrpLin  = type_convert<TCompute>(tRsrp(absUeIdx)) / (nSc * nRxAnt * nTimeChEst);
         
          // Convert to dB scale                                                                                                                                                            
          TCompute rsrpDb = 10*log10f(rsrpLin);
          tRsrp(absUeIdx) = type_convert<TStorageOut>(rsrpDb);

          // Load pre-equalizer noise-interference power in dB                                                                                                                              
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
          // noiseVarPreEq per UE                                                                                                                                                           
          TCompute noiseIntfVarPreEqDb = type_convert<TCompute>(tNoiseIntfVarPreEq(absUeIdx));
#else
          // noiseVarPreEq per UEGRP                                                                                                                                                        
          TCompute noiseIntfVarPreEqDb = type_convert<TCompute>(tNoiseIntfVarPreEq(ueGrpIdx));
#endif
          TCompute sinrPreEqDb = rsrpDb - noiseIntfVarPreEqDb;
          tSinrPreEq(absUeIdx) = type_convert<TStorageOut>(sinrPreEqDb);

          // Load accumulated post-equalizer SNR in linear scale:                                                                                                                        
          TCompute accumSnrPostEqLin = type_convert<TCompute>(tSinrPostEq(absUeIdx));
        
          // Compute linear noise + inteference variance:
          TCompute noiseIntVarLin = nSc * nUeLayers / accumSnrPostEqLin;

          // Convert to dB:
          TCompute noiseIntfVarPostEqDb = 10*log10f(noiseIntVarLin);

          // Store results:
          tNoiseIntfVarPostEq(absUeIdx) = type_convert<TStorageOut>(noiseIntfVarPostEqDb);
          TCompute sinrPostEqDb         = -noiseIntfVarPostEqDb;
          tSinrPostEq(absUeIdx)         = type_convert<TStorageOut>(sinrPostEqDb);

#ifdef ENABLE_DEBUG
          printf("rsrpMeasKernel: ue[%d] sinrPreEqDb %010.7f sinrPostEqDb %010.7f rsrpDb %010.7f (Lin: %010.7f) noiseIntfVarPreEqDb %010.7f noiseIntfVarPostEqDb %010.7f accumNoiseVarPostE\
qLin %010.7f nSc %d nUeLayers %d\n",
                 absUeIdx, sinrPreEqDb, sinrPostEqDb, type_convert<TCompute>(tRsrp(absUeIdx)), rsrpLin, noiseIntfVarPreEqDb, noiseIntfVarPostEqDb, accumNoiseVarPostEqLin, nSc, nUeLayers);
#endif

        }
    }

 } // rsrpMeasKernel_v1
 
 template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t THRD_GRP_TILE_SIZE,            // # of threads in a thread group tile
          uint32_t N_THRD_GRP_TILES_PER_THRD_BLK, // number of thread group tiles in the thread block
          uint32_t N_SC_PER_THRD_BLK_ITER,        // number of subcarriers processed per thread block iteration
          uint32_t N_ITER_PER_THRD_BLK,           // # of iterations in a thread block
          uint8_t  DMRS_SYMBOL_IDX>
__global__ void rsrpMeasWithoutPostEqMeasKernel_v1(puschRxRsrpDynDescr_t* pDynDescr)
 {

    static_assert(((CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS == DMRS_SYMBOL_IDX) || (CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS == DMRS_SYMBOL_IDX)), "Each kernel processes all DMRS symbols or all DMRS symbols except the first DMRS symbol for full-slot");

    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;    

    //--------------------------------------------------------------------------------------------------------    
    puschRxRsrpDynDescr_t& dynDescr = *(pDynDescr);
    const uint32_t ueGrpIdx         = blockIdx.y;
    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[ueGrpIdx];
    
    // clang-format off
    tensor_ref<const TComplexStorageIn> tHEst              (drvdUeGrpPrms.tInfoHEst.pAddr               , drvdUeGrpPrms.tInfoHEst.strides               );// (N_BS_ANTS, N_LAYERS, NF, NH)
    tensor_ref<const TStorageIn>        tNoiseIntfVarPreEq (drvdUeGrpPrms.tInfoNoiseVarPreEq.pAddr      , drvdUeGrpPrms.tInfoNoiseVarPreEq.strides      ); // (N_UE_GRP)
    tensor_ref<volatile TStorageOut>    tRsrp              (drvdUeGrpPrms.tInfoRsrp.pAddr               , drvdUeGrpPrms.tInfoRsrp.strides               );// (N_UE)
    tensor_ref<TStorageOut>             tSinrPreEq         (drvdUeGrpPrms.tInfoSinrPreEq.pAddr          , drvdUeGrpPrms.tInfoSinrPreEq.strides          );// (N_UE)
    tensor_ref<uint32_t>                tInterCtaSyncCnt   (drvdUeGrpPrms.tInfoRsrpInterCtaSyncCnt.pAddr, drvdUeGrpPrms.tInfoRsrpInterCtaSyncCnt.strides);// (N_UE)
    // clang-format on

    uint32_t nUes       = drvdUeGrpPrms.nUes; // Number of UEs in this UE group
    uint16_t nPrb       = drvdUeGrpPrms.nPrb;
    uint32_t nSc        = nPrb*CUPHY_N_TONES_PER_PRB;
    uint16_t nRxAnt     = drvdUeGrpPrms.nRxAnt;  // # of BS antenna (# of rows in H matrix)
    uint16_t nLayers    = drvdUeGrpPrms.nLayers; // # of layers (# of cols in H matrix)   
    uint16_t nTimeChEst = drvdUeGrpPrms.dmrsAddlnPos + 1; // # of time-domain channel estimates available

    uint8_t* pUeGrpLayerToUeIdx = &drvdUeGrpPrms.ueGrpLayerToUeIdx[0];
    uint16_t* pAbsUeIdxs        = &drvdUeGrpPrms.ueIdxs[0];

    cg::thread_block const& thisThrdBlk = cg::this_thread_block();
    cg::thread_block_tile<THRD_GRP_TILE_SIZE> const& thisThrdTile =
      cg::tiled_partition<THRD_GRP_TILE_SIZE>(thisThrdBlk);
    uint32_t thrdIdx  = thisThrdBlk.thread_rank();
    uint32_t nThrds = thisThrdBlk.size();    
    uint32_t totalThreads = nThrds * gridDim.x * gridDim.z;         // number of threads in the grid                                                                                        
    uint32_t thrdGrdIdx = thrdIdx+blockIdx.x*nThrds;                // thread index within the grid                                                                                         

    uint32_t& interCtaSyncCnt = tInterCtaSyncCnt(ueGrpIdx);

    __shared__ bool isLastCtaDone;

    // Flexible shared memory tensor
    extern __shared__ TCompute shLayerPwr[];

    // Initialization
    if(0 == thrdIdx)
    {
        isLastCtaDone = false;
    }
    if(thrdIdx < nLayers)
    {
      shLayerPwr[thrdIdx]    = cuGet<TCompute>(0);
    }

    __syncthreads();

    
    //--------------------------------------------------------------------------------------------------------                 
    //                                                                                                                                                                                      
    // Reduction is done using a grid.x-stride loop.  As the data is arranged with layer as an intermediate                                                                                 
    // index, tHEst(nRxAnt,nLayer,nSc,nTimeChEst) and tReeDiagInv(nReeSc, nLayers, nReePrb) the approach is to                                                                              
    // use all grid-x threads to span an integer number of 'x-y planes' of data in the structure.  By using                                                                                 
    // that many threads (i.e. integer multiple of first two grid dimensions), and using a grid stride of that                                                                              
    // size, the data can be accessed in a perfectly coalesed mannar, and the Layer index of the data never                                                                                 
    // changes within a thread - so the bulk of the reduction can be performed within the thread.                                                                                           
    //                                                                                                                                                                                      
    // Shared memory is used to reduce layer data between threads (because there is no guarantee that the                                                                                   
    // array dimensions will be convenient powers of 2?).                                                                                                                                   

    // Accumulate power -----                                                                                                                                                               
    uint32_t iLayer;
    TCompute pwr;
    uint32_t stride = (totalThreads / (nLayers*nRxAnt))*nLayers*nRxAnt;

    pwr            = cuGet<TCompute>(0);
    if ( thrdGrdIdx < stride ) {
      
      TComplexCompute signal;
      iLayer         = (thrdGrdIdx/nRxAnt)%nLayers;
      uint32_t loopLimit      = nRxAnt * nLayers * nSc * nTimeChEst;
      const cuComplex* pStart = &tHEst(0,0,0,0);
      
      uint32_t startIdx = 0;
      if(DMRS_SYMBOL_IDX==CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS)
      {
          startIdx = nRxAnt * nLayers * nSc;
      }

      // this is the long loop over all [nRxAnt * nLayers * nSc * nTimeChEst]                                                                                                               
      for ( uint32_t idx=startIdx + thrdGrdIdx; idx<loopLimit; idx+=stride ) 
      {
        signal = type_convert<TComplexCompute>( *(pStart+idx) );
        pwr   += cuReal(signal*cuConj(signal));
      }

    }
    
    // Reduce, per-layer, within warp, then ccumulate between warps in smem using atomics
    for ( int jLayer=0; jLayer<nLayers; ++jLayer) {
      TCompute sum = 0;
      if ( iLayer == jLayer ) sum = pwr;
      TCompute sumpwr = cg::reduce(thisThrdTile, sum, cg::plus<TCompute>());
      if ( threadIdx.x == 0 ) {
	atomicAdd(shLayerPwr+jLayer, sumpwr);
      }
    }
    
    // ensure all accumulations are complete                                                                                                                                                
    __syncthreads();

    if ( thrdIdx < nLayers ) {

      uint8_t ueIdx      = pUeGrpLayerToUeIdx[thrdIdx];
      uint16_t absUeIdx  = pAbsUeIdxs[ueIdx];

      atomicAdd(const_cast<TCompute*>(tRsrp.pAddr + tRsrp.offset(absUeIdx)), shLayerPwr[thrdIdx]);
    }

    __syncthreads();
    
    //--------------------------------------------------------------------------------------------------------                                                                              
    // Logic to ensure all thread blocks in the grid complete (ending in writes to tRsrp above)                                                                                             
    if(0 == thrdIdx)
    {
        // Ensure interCtaSyncCnt is incremented only after the tRsrp global memory write has been completed.                                                                               
        // Note that while the global memory access above is atomic, it does not imply ordering constraints for memory operations,                                                          
        // hence a threadfence is still needed.                                                                                                                                             
        __threadfence();

        //uint32_t syncCnt = atomicInc(const_cast<uint32_t*>(&interCtaSyncCnt), nUeGrpThrdBlksNeeded);                                                                                      
        // NOTE: the following is atomicInc, vs. atomicAdd.  gridDim.x*gridDim.z is just an upper limit.                                                                                    
        // SR - gridDim.x * gridDim.z (since gridDim.y == ueGrpIdx for this kernel) is just more readable than nUeGrpThrdBlksNeeded                                                         
        //      ... once you know all gridIdx.y's are different reductions.  Note gridDim.z is 1.  So, could just use gridDim.x.                                                            
        // SR - i.e. gridDim.x * gridDim.z is all the blocks, so gridDim.x * gridDim.z - 1 is definitely the last block.                                                                    
        uint32_t syncCnt = atomicInc(const_cast<uint32_t*>(&interCtaSyncCnt), gridDim.x*gridDim.z);

        // Is this the last CTA to be processed for this user group?                                                                                                                        
        isLastCtaDone = (syncCnt == (gridDim.x*gridDim.z - 1)) ? true : false;
    }

    __syncthreads();
    
    // Use one thread block per UE group to compute the average RSRP value per UE group rsrp(nUeGrps), compute SINR                                                                         
    // Note: Rsrp is converted from linear to dB here (after its used in rsrp calculation)                                                                                                  
    // this is worth 2.5 us.                                                                                                                                                                
    if(isLastCtaDone)
    {
      for(uint32_t i = thrdIdx; i < nUes; i+=nThrds)
        {
          // Average the RSRP measurement: average across allocated PRBs, Rx antenna, time domain estimates (and sum across layers)                                                         
          uint16_t absUeIdx = pAbsUeIdxs[i];

          // Note: We don't average across layers                                                                                                                                           
          TCompute rsrpLin  = type_convert<TCompute>(tRsrp(absUeIdx)) / (nSc * nRxAnt * nTimeChEst);
         
          // Convert to dB scale                                                                                                                                                            
          TCompute rsrpDb = 10*log10f(rsrpLin);
          tRsrp(absUeIdx) = type_convert<TStorageOut>(rsrpDb);

          // Load pre-equalizer noise-interference power in dB                                                                                                                              
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
          // noiseVarPreEq per UE                                                                                                                                                           
          TCompute noiseIntfVarPreEqDb = type_convert<TCompute>(tNoiseIntfVarPreEq(absUeIdx));
#else
          // noiseVarPreEq per UEGRP                                                                                                                                                        
          TCompute noiseIntfVarPreEqDb = type_convert<TCompute>(tNoiseIntfVarPreEq(ueGrpIdx));
#endif
          TCompute sinrPreEqDb = rsrpDb - noiseIntfVarPreEqDb;
          tSinrPreEq(absUeIdx) = type_convert<TStorageOut>(sinrPreEqDb);

        }
    }

 } // rsrpMeasWithoutPostEqMeasKernel_v1
 
 
 
// #undef ENABLE_DEBUG

template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t THRD_GRP_TILE_SIZE,            // # of threads in a thread group tile
          uint32_t N_THRD_GRP_TILES_PER_THRD_BLK, // number of thread group tiles in the thread block
          uint32_t N_SC_PER_THRD_BLK_ITER,        // number of subcarriers processed per thread block iteration
          uint32_t N_ITER_PER_THRD_BLK>           // # of iterations in a thread block
__global__ void sinrMeasEarlyHarqKernel_v1(puschRxRsrpDynDescr_t* pDynDescr)
 {
    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;    

    //--------------------------------------------------------------------------------------------------------    
    puschRxRsrpDynDescr_t& dynDescr = *(pDynDescr);
    const uint32_t ueGrpIdx         = blockIdx.y;
    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[ueGrpIdx];
    
    // clang-format off
    tensor_ref<const TComplexStorageIn> tHEst              (drvdUeGrpPrms.tInfoHEst.pAddr               , drvdUeGrpPrms.tInfoHEst.strides               );// (N_BS_ANTS, N_LAYERS, NF, NH)
    tensor_ref<TStorageIn>              tNoiseIntfVarPreEq (drvdUeGrpPrms.tInfoNoiseVarPreEq.pAddr      , drvdUeGrpPrms.tInfoNoiseVarPreEq.strides      ); // (N_UE_GRP)
    tensor_ref<volatile TStorageOut>    tRsrp              (drvdUeGrpPrms.tInfoRsrp.pAddr               , drvdUeGrpPrms.tInfoRsrp.strides               );// (N_UE)  
    tensor_ref<volatile TStorageOut>    tRsrpEhq           (drvdUeGrpPrms.tInfoRsrpEhq.pAddr            , drvdUeGrpPrms.tInfoRsrpEhq.strides            );// (N_UE)  
    tensor_ref<TStorageOut>             tSinrPreEq         (drvdUeGrpPrms.tInfoSinrPreEq.pAddr          , drvdUeGrpPrms.tInfoSinrPreEq.strides          );// (N_UE)
    tensor_ref<uint32_t>                tInterCtaSyncCnt   (drvdUeGrpPrms.tInfoRsrpInterCtaSyncCnt.pAddr, drvdUeGrpPrms.tInfoRsrpInterCtaSyncCnt.strides);// (N_UE)
    // clang-format on

    uint32_t nUes       = drvdUeGrpPrms.nUes; // Number of UEs in this UE group
    uint16_t nPrb       = drvdUeGrpPrms.nPrb;
    uint32_t nSc        = nPrb*CUPHY_N_TONES_PER_PRB;
    uint16_t nRxAnt     = drvdUeGrpPrms.nRxAnt;  // # of BS antenna (# of rows in H matrix)
    uint16_t nLayers    = drvdUeGrpPrms.nLayers; // # of layers (# of cols in H matrix)   
    uint16_t nTimeChEst = 1;//for early-harq processing

    uint8_t* pUeGrpLayerToUeIdx = &drvdUeGrpPrms.ueGrpLayerToUeIdx[0];
    uint16_t* pAbsUeIdxs        = &drvdUeGrpPrms.ueIdxs[0];

    cg::thread_block const& thisThrdBlk = cg::this_thread_block();
    cg::thread_block_tile<THRD_GRP_TILE_SIZE> const& thisThrdTile =
      cg::tiled_partition<THRD_GRP_TILE_SIZE>(thisThrdBlk);
    uint32_t thrdIdx  = thisThrdBlk.thread_rank();
    uint32_t nThrds = thisThrdBlk.size();    
    uint32_t totalThreads = nThrds * gridDim.x * gridDim.z;         // number of threads in the grid                                                                                        
    uint32_t thrdGrdIdx = thrdIdx+blockIdx.x*nThrds;                // thread index within the grid                                                                                         

    uint32_t& interCtaSyncCnt = tInterCtaSyncCnt(ueGrpIdx);

    __shared__ bool isLastCtaDone;

    // Flexible shared memory tensor
    extern __shared__ TCompute shLayerPwr[];

    // Initialization
    if(0 == thrdIdx)
    {
        isLastCtaDone = false;
    }
    if(thrdIdx < nLayers)
    {
      shLayerPwr[thrdIdx]    = cuGet<TCompute>(0);
    }

    __syncthreads();

    
    //--------------------------------------------------------------------------------------------------------                 
    //                                                                                                                                                                                      
    // Reduction is done using a grid.x-stride loop.  As the data is arranged with layer as an intermediate                                                                                 
    // index, tHEst(nRxAnt,nLayer,nSc,nTimeChEst) and tReeDiagInv(nReeSc, nLayers, nReePrb) the approach is to                                                                              
    // use all grid-x threads to span an integer number of 'x-y planes' of data in the structure.  By using                                                                                 
    // that many threads (i.e. integer multiple of first two grid dimensions), and using a grid stride of that                                                                              
    // size, the data can be accessed in a perfectly coalesed mannar, and the Layer index of the data never                                                                                 
    // changes within a thread - so the bulk of the reduction can be performed within the thread.                                                                                           
    //                                                                                                                                                                                      
    // Shared memory is used to reduce layer data between threads (because there is no guarantee that the                                                                                   
    // array dimensions will be convenient powers of 2?).                                                                                                                                   

    // Accumulate power -----                                                                                                                                                               
    uint32_t iLayer;
    TCompute pwr;
    uint32_t stride = (totalThreads / (nLayers*nRxAnt))*nLayers*nRxAnt;

    pwr            = cuGet<TCompute>(0);
    if ( thrdGrdIdx < stride ) {
      
      TComplexCompute signal;
      iLayer         = (thrdGrdIdx/nRxAnt)%nLayers;
      uint32_t loopLimit      = nRxAnt * nLayers * nSc * nTimeChEst;
      const cuComplex* pStart = &tHEst(0,0,0,0);

      // this is the long loop over all [nRxAnt * nLayers * nSc * nTimeChEst]                                                                                                               
      for ( uint32_t idx=thrdGrdIdx; idx<loopLimit; idx+=stride ) {
        signal = type_convert<TComplexCompute>( *(pStart+idx) );
        pwr   += cuReal(signal*cuConj(signal));

      }

    }
    
    // Reduce, per-layer, within warp, then ccumulate between warps in smem using atomics
    for ( int jLayer=0; jLayer<nLayers; ++jLayer) {
      TCompute sum = 0;
      if ( iLayer == jLayer ) sum = pwr;
      TCompute sumpwr = cg::reduce(thisThrdTile, sum, cg::plus<TCompute>());
      if ( threadIdx.x == 0 ) {
	atomicAdd(shLayerPwr+jLayer, sumpwr);
      }
    }
      
    // ensure all accumulations are complete                                                                                                                                                
    __syncthreads();

    if ( thrdIdx < nLayers ) 
    {
      uint8_t ueIdx      = pUeGrpLayerToUeIdx[thrdIdx];
      uint16_t absUeIdx  = pAbsUeIdxs[ueIdx];

      atomicAdd(const_cast<TCompute*>(tRsrp.pAddr + tRsrp.offset(absUeIdx)), shLayerPwr[thrdIdx]);
    }

    __syncthreads();
    
    //--------------------------------------------------------------------------------------------------------                                                                              
    // Logic to ensure all thread blocks in the grid complete (ending in writes to tRsrp above)                                                                                             
    if(0 == thrdIdx)
    {
        // Ensure interCtaSyncCnt is incremented only after the tRsrp global memory write has been completed.                                                                               
        // Note that while the global memory access above is atomic, it does not imply ordering constraints for memory operations,                                                          
        // hence a threadfence is still needed.                                                                                                                                             
        __threadfence();

        //uint32_t syncCnt = atomicInc(const_cast<uint32_t*>(&interCtaSyncCnt), nUeGrpThrdBlksNeeded);                                                                                      
        // NOTE: the following is atomicInc, vs. atomicAdd.  gridDim.x*gridDim.z is just an upper limit.                                                                                    
        // SR - gridDim.x * gridDim.z (since gridDim.y == ueGrpIdx for this kernel) is just more readable than nUeGrpThrdBlksNeeded                                                         
        //      ... once you know all gridIdx.y's are different reductions.  Note gridDim.z is 1.  So, could just use gridDim.x.                                                            
        // SR - i.e. gridDim.x * gridDim.z is all the blocks, so gridDim.x * gridDim.z - 1 is definitely the last block.                                                                    
        uint32_t syncCnt = atomicInc(const_cast<uint32_t*>(&interCtaSyncCnt), gridDim.x*gridDim.z);

        // Is this the last CTA to be processed for this user group?                                                                                                                        
        isLastCtaDone = (syncCnt == (gridDim.x*gridDim.z - 1)) ? true : false;
    }

    __syncthreads();
    
    // Use one thread block per UE group to compute the average RSRP value per UE group rsrp(nUeGrps), compute SINR                                                                         
    // Note: Rsrp is converted from linear to dB here (after its used in rsrp calculation)                                                                                                  
    // this is worth 2.5 us.                                                                                                                                                                
    if(isLastCtaDone)
    {
        for(uint32_t i = thrdIdx; i < nUes; i+=nThrds)
        {
          // Average the RSRP measurement: average across allocated PRBs, Rx antenna, time domain estimates (and sum across layers)                                                         
          uint16_t absUeIdx = pAbsUeIdxs[i];

          // Note: We don't average across layers                                                                                                                                           
          TCompute rsrpLin  = type_convert<TCompute>(tRsrp(absUeIdx)) / (nSc * nRxAnt * nTimeChEst);

          // Convert to dB scale                                                                                                                                                            
          TCompute rsrpDb = 10*log10f(rsrpLin);
          
          tRsrpEhq(absUeIdx) = type_convert<TStorageOut>(rsrpDb);

          // Load pre-equalizer noise-interference power in dB                                                                                                                              
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
          // noiseVarPreEq per UE                                                                                                                                                           
          TCompute noiseIntfVarPreEqDb = type_convert<TCompute>(tNoiseIntfVarPreEq(absUeIdx));
          tNoiseIntfVarPreEq(absUeIdx) = type_convert<TStorageIn>(0);  //reset for full-slot processing
#else
          // noiseVarPreEq per UEGRP                                                                                                                                                        
          TCompute noiseIntfVarPreEqDb = type_convert<TCompute>(tNoiseIntfVarPreEq(ueGrpIdx));
          tNoiseIntfVarPreEq(ueGrpIdx) = type_convert<TStorageIn>(0);  //reset for full-slot processing
#endif
          TCompute sinrPreEqDb = rsrpDb - noiseIntfVarPreEqDb;
          tSinrPreEq(absUeIdx) = type_convert<TStorageOut>(sinrPreEqDb);
        }
        interCtaSyncCnt = 0; //reset for full-slot processing   
    }
 }   // sinrMeasEarlyHarqKernel_v1                      



/*VCAST_DONT_INSTRUMENT_START*/
 // Using memcpy_async to load channel estimates from shared memory
 // @todo: using this version results in large (v2 is 10x versus v1) kernel runtimes and large launch latency. To be debugged
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t THRD_GRP_TILE_SIZE,            // # of threads in a thread group tile
          uint32_t N_THRD_GRP_TILES_PER_THRD_BLK, // number of thread group tiles in the thread block
          uint32_t N_SC_PER_THRD_BLK_ITER,        // number of subcarriers processed per thread block iteration
          uint32_t N_ITER_PER_THRD_BLK>           // # of iterations in a thread block
__global__ void rsrpMeasKernel_v2(puschRxRsrpDynDescr_t* pDynDescr)
 { 
    KERNEL_PRINT_GRID_ONCE("%s\n grid = (%u %u %u), block = (%u %u %u), THRD_GRP_TILE_SIZE %u, N_THRD_GRP_TILES_PER_THRD_BLK %u, N_SC_PER_THRD_BLK_ITER %u, N_ITER_PER_THRD_BLK %u\n",
                           __PRETTY_FUNCTION__,
                           gridDim.x, gridDim.y, gridDim.z,
                           blockDim.x, blockDim.y, blockDim.z,
                           THRD_GRP_TILE_SIZE,
                           N_THRD_GRP_TILES_PER_THRD_BLK,
                           N_SC_PER_THRD_BLK_ITER,
                           N_ITER_PER_THRD_BLK);

    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;    

    //--------------------------------------------------------------------------------------------------------    
    puschRxRsrpDynDescr_t& dynDescr = *(pDynDescr);
    const uint32_t ueGrpIdx         = blockIdx.y;
    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[ueGrpIdx];
    
    // clang-format off
    tensor_ref<const TComplexStorageIn> tHEst           (drvdUeGrpPrms.tInfoHEst.pAddr               , drvdUeGrpPrms.tInfoHEst.strides               );// (N_BS_ANTS, N_LAYERS, NF, NH)
    tensor_ref<volatile TStorageOut>    tRsrp           (drvdUeGrpPrms.tInfoRsrp.pAddr               , drvdUeGrpPrms.tInfoRsrp.strides               );// (N_UE)
    tensor_ref<uint32_t>                tInterCtaSyncCnt(drvdUeGrpPrms.tInfoRsrpInterCtaSyncCnt.pAddr, drvdUeGrpPrms.tInfoRsrpInterCtaSyncCnt.strides);// (N_UE)
    // clang-format on

    // Number of subcarriers processed by a thread group tile, thread block iteration and thread block execution
    static constexpr uint32_t N_SC_PER_THRD_BLK = N_SC_PER_THRD_BLK_ITER*N_ITER_PER_THRD_BLK;
    static constexpr uint32_t N_PIPE_STAGES     = 2; // # of pipeline stages

    // The number of thread blocks are sized to process UE group with largest PRB allocation
    // For a given UE group, not all thread blocks may be needed
    // uint32_t nUeGrps    = gridDim.y;    
    uint32_t nUes       = drvdUeGrpPrms.nUes; // Number of UEs in this UE group
    uint16_t nPrb       = drvdUeGrpPrms.nPrb;
    uint32_t nSc        = nPrb*CUPHY_N_TONES_PER_PRB;
    uint16_t nRxAnt     = drvdUeGrpPrms.nRxAnt;  // # of BS antenna (# of rows in H matrix)
    uint16_t nLayers    = drvdUeGrpPrms.nLayers; // # of layers (# of cols in H matrix)   
    uint16_t nTimeChEst = drvdUeGrpPrms.dmrsAddlnPos + 1; // # of time-domain channel estimates available
    uint16_t nElemHEst  = nRxAnt*nLayers;

    uint8_t* pUeGrpLayerToUeIdx = &drvdUeGrpPrms.ueGrpLayerToUeIdx[0];
    uint16_t* pAbsUeIdxs        = &drvdUeGrpPrms.ueIdxs[0];

    // Number of thread blocks to process each user group are sized for the maximum PRB allocation
    // Compute the thread blocks needed to process the frequency allocation of this user group
    uint32_t nUeGrpFreqAllocThrdBlksNeeded = div_round_up(nSc, N_SC_PER_THRD_BLK);

    // Early exit - thread blocks which exceed those needed to process subcarriers of current UE group
    if(blockIdx.x >= nUeGrpFreqAllocThrdBlksNeeded)  return;

    cg::thread_block const& thisThrdBlk = cg::this_thread_block();

    // Divvy up the thread block into tiles of size THRD_GRP_TILE_SIZE
    cg::thread_block_tile<THRD_GRP_TILE_SIZE> const& layerThrdTile =
        cg::tiled_partition<THRD_GRP_TILE_SIZE>(thisThrdBlk);
    uint32_t thrdIdx  = thisThrdBlk.thread_rank();

    // Each thread tile processes a layer
    uint32_t layerThrdTileIdx = threadIdx.y;
    uint32_t thrdIdxInLayerThrdTile = layerThrdTile.thread_rank();
        
    // Total number of thread blocks needed to process the PRB allocation of this user group
    uint32_t nUeGrpThrdBlksNeeded = nUeGrpFreqAllocThrdBlksNeeded;

    // Compute the number of subcarriers processed until this thread block
    uint32_t ueGrpFreqAllocThrdBlkdx = blockIdx.x;
    uint32_t nScProcessed = ueGrpFreqAllocThrdBlkdx * N_SC_PER_THRD_BLK;

    // Start subcarrier to be processed by this thread block    
    uint32_t startSc = 0; // the channel estimates for each UE group always start at 0 offset into the channel estimate buffer
    uint32_t startScThisThrdBlk = startSc + nScProcessed;

    // Number of PRBs remaining to be processed from this thread block onwards
    int32_t nScRemaining = nSc - nScProcessed;

    // Number of subcarriers to be processed by this thread block    
    int32_t nScThisThrdBlk = (nScRemaining <= N_SC_PER_THRD_BLK) ? nScRemaining : N_SC_PER_THRD_BLK;
    uint32_t nScPerItr     = (nScThisThrdBlk <= N_SC_PER_THRD_BLK_ITER) ? nScThisThrdBlk : N_SC_PER_THRD_BLK_ITER;

    uint32_t& interCtaSyncCnt = tInterCtaSyncCnt(ueGrpIdx);
       
    __shared__ bool isLastCtaDone;

    // Shared memory tensor with runtime dimensioning
    __shared__ int32_t shHEstStrides[5];

    extern __shared__ TComplexCompute smemBlk[];
    tensor_ref<TComplexCompute> shHEst(smemBlk, shHEstStrides);    
    
#ifdef ENABLE_DEBUG
    if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
    {
        printf("%s\n", __PRETTY_FUNCTION__);
        printf("rsrpMeasKernel - nPrb            : %d\n", nPrb);
        printf("rsrpMeasKernel - nRxAnt          : %d\n", nRxAnt);
        printf("rsrpMeasKernel - nLayers         : %d\n", nLayers);
        printf("rsrpMeasKernel - nTimeChEst      : %d\n", nTimeChEst);
        printf("rsrpMeasKernel - tInfoHEst       : addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<uint16_t*>(drvdUeGrpPrms.tInfoHEst.pAddr), drvdUeGrpPrms.tInfoHEst.strides[0], drvdUeGrpPrms.tInfoHEst.strides[1], drvdUeGrpPrms.tInfoHEst.strides[2], drvdUeGrpPrms.tInfoDataRx.strides[3]);
        printf("rsrpMeasKernel - tRsrp           : addr 0x%llx strides[0] %d\n", static_cast<uint16_t*>(drvdUeGrpPrms.tInfoRsrp.pAddr), drvdUeGrpPrms.tInfoRsrp.strides[0]);
        printf("rsrpMeasKernel - tInterCtaSyncCnt: addr 0x%llx strides[0] %d\n", static_cast<uint32_t*>(drvdUeGrpPrms.tInfoRsrpInterCtaSyncCnt.pAddr), drvdUeGrpPrms.tInfoRsrpInterCtaSyncCnt.strides[0]);
    }

    if((0 == thrdIdx) && (0 == blockIdx.x))
    {
        printf("rsrpMeasKernel - blockIdx (%d %d %d) threadIdx (%d %d %d) ueGrpIdx %d nRxAnt %d nLayers %d nTimeChEst %d nSc %d nScRemaining %d nScThisThrdBlk %d nScPerItr %d\n", __PRETTY_FUNCTION__, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, ueGrpIdx, nRxAnt, nLayers, nTimeChEst, nSc, nScRemaining, nScThisThrdBlk, nScPerItr);
    }
#endif

    // Initialization
    if(0 == thrdIdx)
    {
        isLastCtaDone = false;

        // Initialize strides for channel estimation matrix copy in shared memory
        shHEstStrides[0] = 1;
        shHEstStrides[1] = shHEstStrides[0]*nRxAnt;
        shHEstStrides[2] = shHEstStrides[1]*nLayers;
        shHEstStrides[3] = shHEstStrides[2]*N_SC_PER_THRD_BLK_ITER;
        shHEstStrides[4] = shHEstStrides[3]*nTimeChEst;
    }
    thisThrdBlk.sync(); // Ensure all threads see the dimensions in shared memmory before using them in memcpy_async

    // Pipeline for async loads
    #if CUDART_VERSION >= 11060
    #pragma nv_diag_suppress static_var_with_dynamic_init // Disables "shPipeState" initialization warning.
    #else
    #pragma diag_suppress static_var_with_dynamic_init // Disables "shPipeState" initialization warning.
    #endif
 
    // Ref: https://nvidia.github.io/libcudacxx/extended_api/synchronization_primitives/pipeline.html
    // unified pipeline with all the threads performing both producer and consumer actions
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, N_PIPE_STAGES> shPipeState;
    auto pipeline = cuda::make_pipeline(thisThrdBlk, &shPipeState);
 
    uint32_t currPipeStageIdx = 0;
    int32_t cpyBlkLenBytes    = sizeof(TComplexCompute) * nElemHEst * nScPerItr;

    // Prefetch data for first iteration
    pipeline.producer_acquire();
    for(int32_t chEstTimeIdx = 0; chEstTimeIdx < nTimeChEst; ++chEstTimeIdx)
    {        
        cuda::memcpy_async(thisThrdBlk, &shHEst(0, 0, 0, chEstTimeIdx, currPipeStageIdx), &tHEst(0, 0, startScThisThrdBlk, chEstTimeIdx), cpyBlkLenBytes, pipeline);
    }
    pipeline.producer_commit();

    thisThrdBlk.sync();

#if 0
    pipeline.consumer_wait();
    if(thrdIdx < nRxAnt * nScPerItr * nTimeChEst * nLayers)
    {
        uint32_t rxAntIdx     = thrdIdx % nRxAnt;
        uint32_t scIdx        = (thrdIdx / nRxAnt)  % nScPerItr;
        uint32_t chEstTimeIdx = (thrdIdx / (nRxAnt * nScPerItr)) % nTimeChEst;
        uint32_t layerIdx     = (thrdIdx / (nRxAnt * nScPerItr * nTimeChEst));

        if(layerIdx < nLayers)
        {
            volatile TComplexCompute refSample = shHEst(rxAntIdx, layerIdx, scIdx, chEstTimeIdx, currPipeStageIdx);
            printf("rsrpMeasKernel: currPipeStageIdx %d cpyBlkLenBytes %d thrdIdx %d layerIdx %d rxAntIdx %d chEstTimeIdx %d absScIdx %d shRefSample %07.4f + j %07.04f refSample %07.4f + j %07.04f\n", currPipeStageIdx, cpyBlkLenBytes, thrdIdx, layerIdx, rxAntIdx, chEstTimeIdx, startScThisThrdBlk + scIdx, cuReal(refSample), cuImag(refSample), cuReal(tHEst(rxAntIdx, layerIdx, startScThisThrdBlk + scIdx, chEstTimeIdx)), cuImag(tHEst(rxAntIdx, layerIdx, startScThisThrdBlk + scIdx, chEstTimeIdx))); 
        }
    }
    pipeline.consumer_release();
#endif

    // Number of iterations to process all layers of the UE group
    uint16_t nLayerThrdTiles = blockDim.y; // same as N_THRD_GRP_TILES_PER_THRD_BLK
    uint32_t nLayerItr       = div_round_up(nLayers, nLayerThrdTiles);
    
    // Number of elements to be processed (i.e. reduced) for a given layer
    uint32_t nLayerElems = nRxAnt * nScPerItr * nTimeChEst;

    //  Number of iterations needed to reduce the nLayerElems elements in a layer
    uint32_t layerThrdTileSz = layerThrdTile.size(); // same as THRD_GRP_TILE_SIZE
    uint32_t nReductionItr   = div_round_up(nLayerElems, layerThrdTileSz);
        
    //--------------------------------------------------------------------------------------------------------    
    // Each thread block iteration computes RSRP for upto N_SC_PER_THRD_BLK_ITER subcarriers
    for(uint32_t freqItrIdx = 0; freqItrIdx < N_ITER_PER_THRD_BLK; ++freqItrIdx)
    {                        
        //--------------------------------------------------------------------------------------------------------    
        // Load channel estimate for next iteration
        uint32_t nScProcUntilPrevItr = freqItrIdx * nScPerItr;
        uint32_t nScProcUntilCurrItr = nScProcUntilPrevItr + nScPerItr;
        uint32_t nScProcUntilNxtItr  = nScProcUntilCurrItr + nScPerItr;

        // An iteration (very first or last) could process less than N_SC_PER_THRD_BLK_ITER i.e. the number of subcarriers a 
        // thread block iteration can acutally process
        if(nScProcUntilCurrItr > nScThisThrdBlk)
        {
            // Adjust current iteration limits
            nScPerItr     = nScThisThrdBlk - nScProcUntilPrevItr; // number of subcarriers remaining
            nLayerElems   = nRxAnt * nScPerItr * nTimeChEst;
            nReductionItr = div_round_up(nLayerElems, layerThrdTileSz);
        }
        // Check if next iteration needs to process fewer than N_SC_PER_THRD_BLK_ITER
        else if(nScProcUntilNxtItr > nScThisThrdBlk)
        {
            // Setup copy length for next iteration
            cpyBlkLenBytes = sizeof(TComplexCompute) * nElemHEst * (nScThisThrdBlk - nScProcUntilCurrItr);
        }
        
        if(cpyBlkLenBytes > 0)
        {
            uint32_t startScNxtItr = startScThisThrdBlk + nScProcUntilCurrItr;

            // Prefetch data for next iteration
            pipeline.producer_acquire();
            for(int32_t chEstTimeIdx = 0; chEstTimeIdx < nTimeChEst; ++chEstTimeIdx)
            {
                cuda::memcpy_async(thisThrdBlk, &shHEst(0, 0, 0, chEstTimeIdx, currPipeStageIdx^1), &tHEst(0, 0, startScNxtItr, chEstTimeIdx), cpyBlkLenBytes, pipeline);
            }
            pipeline.producer_commit();
        }

        //--------------------------------------------------------------------------------------------------------            
        // Compute RSRP for each layer
        // Thread block can process N_THRD_GRP_TILES_PER_THRD_BLK layers per iteration
        // Loop to ensure all nLayers are covered (when nLayers > N_THRD_GRP_TILES_PER_THRD_BLK)
        pipeline.consumer_wait();
        for(uint32_t layerItrIdx = 0; layerItrIdx < nLayerItr; ++layerItrIdx)
        {
            uint32_t layerIdx = (layerItrIdx * nLayerThrdTiles) + layerThrdTileIdx;
            if(layerIdx >= nLayers) continue;

            // Use a layer thread tile for reduction over rxAnt, subcarriers in PRB, time (i.e. # of time domain channel estimates)
            TCompute pwr = cuGet<TCompute>(0);
            for(uint32_t reductionItrIdx = 0; reductionItrIdx < nReductionItr; ++reductionItrIdx)
            {
                uint32_t idx = (reductionItrIdx * layerThrdTileSz) + thrdIdxInLayerThrdTile;
                uint32_t rxAntIdx     = idx % nRxAnt;
                uint32_t scIdx        = (idx / nRxAnt)  % nScPerItr;
                uint32_t chEstTimeIdx = (idx / (nRxAnt * nScPerItr));
        
                if(chEstTimeIdx >= nTimeChEst) continue;

                TComplexCompute refSample = shHEst(rxAntIdx, layerIdx, scIdx, chEstTimeIdx, currPipeStageIdx);
                pwr += cuReal(refSample*cuConj(refSample));

#ifdef ENABLE_DEBUG
                uint8_t ueIdx      = pUeGrpLayerToUeIdx[layerIdx];
                uint16_t absUeIdx  = pAbsUeIdxs[ueIdx];                
                printf("rsrpMeasKernel: nLayerItr %d nReductionItr %d layerThrdTileSz %d idx %d ueIdx %d absUeIdx %d layerIdx %d rxAntIdx %d chEstTimeIdx %d absScIdx %d pwr %07.4f shRefSample %07.4f + j %07.04f refSample %07.4f + j %07.04f\n", nLayerItr, nReductionItr, layerThrdTileSz, idx, ueIdx, absUeIdx, layerIdx, rxAntIdx, chEstTimeIdx, startScThisThrdBlk + nScProcUntilPrevItr + scIdx, pwr, cuReal(refSample), cuImag(refSample), cuReal(tHEst(rxAntIdx, layerIdx, startScThisThrdBlk + nScProcUntilPrevItr + scIdx, chEstTimeIdx)), cuImag(tHEst(rxAntIdx, layerIdx, startScThisThrdBlk + nScProcUntilPrevItr + scIdx, chEstTimeIdx)));
#endif
            }
 
            // Accumulate power within a thread group corresponding to a layer            
            TCompute accumPwr = cg::reduce(layerThrdTile, pwr, cg::plus<TCompute>());
            
            // Accumulate across layers belonging to the same UE. Pick one thread per layer tile for accumulation
            if(0 == thrdIdxInLayerThrdTile)
            {
                uint8_t ueIdx      = pUeGrpLayerToUeIdx[layerIdx];
                uint16_t absUeIdx  = pAbsUeIdxs[ueIdx];
                atomicAdd(const_cast<TCompute*>(tRsrp.pAddr + tRsrp.offset(absUeIdx)), accumPwr);

#ifdef ENABLE_DEBUG
                printf("rsrpMeasKernel: accumPwr[%d][%d][%d][%d] %010.7f\n", absUeIdx, ueIdx, layerIdx, startScThisThrdBlk + nScProcUntilPrevItr, accumPwr);
#endif                
            }
        }    
        pipeline.consumer_release();

        currPipeStageIdx ^= 1;

        // If the current iteration has processed all the subcarriers then we are done
        if(nScProcUntilCurrItr >= nScThisThrdBlk) break;        
    }
    
    thisThrdBlk.sync();

    //--------------------------------------------------------------------------------------------------------    
    // rsrp(nUeGrps) array - average across allocated PRBs, Rx antenna, time domain estimates (and sum across layers)

    // Logic to ensure all thread blocks in the grid complete (ending in writes to tRsrp above)
    if(0 == thrdIdx)
    {
        // Ensure interCtaSyncCnt is incremented only after the tRsrp global memory write has been completed.
        // Note that while the global memory access above is atomic, it does not imply ordering constraints for memory operations,
        // hence a threadfence is still needed.
        __threadfence();

        uint32_t syncCnt = atomicInc(const_cast<uint32_t*>(&interCtaSyncCnt), nUeGrpThrdBlksNeeded);
        
        // Is this the last CTA to be processed for this user group?        
        isLastCtaDone = (syncCnt == (nUeGrpThrdBlksNeeded - 1)) ? true : false;

#ifdef ENABLE_DEBUG
    printf("rsrpMeasKernel: blockIdx(x y z) %d %d %d threadIdx(x y z) %d %d %d ueGrpFreqAllocThrdBlkdx %d ueGrpIdx %d isLastCtaDone %u nUeGrpThrdBlksNeeded %d syncCnt %d interCtaSyncCnt %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, ueGrpFreqAllocThrdBlkdx, ueGrpIdx, isLastCtaDone, nUeGrpThrdBlksNeeded, syncCnt, interCtaSyncCnt);
#endif
    }
    thisThrdBlk.sync();

    // Use one thread block per UE group to compute the average RSRP value rsrp(nUeGrps) and compute SINR
    // Note: Rsrp is converted from linear to dB here (after its used in rsrp calculation)
    if(isLastCtaDone)
    {
        if(0 == thrdIdx)
        {
            for(uint32_t i = 0; i < nUes; ++i)
            {        
                // Average the RSRP measurement: average across allocated PRBs, Rx antenna, time domain estimates (and sum across layers)
                uint16_t absUeIdx = pAbsUeIdxs[i];

                // Note: We don't average across layers
                TCompute rsrpLin  = type_convert<TCompute>(tRsrp(absUeIdx)) / (nSc * nRxAnt * nTimeChEst);

                // Convert to dB scale
                TCompute rsrpDb = 10*log10f(rsrpLin);
                tRsrp(absUeIdx) = type_convert<TStorageOut>(rsrpDb);
    
#ifdef ENABLE_DEBUG
                printf("rsrpMeasKernel: rsrpDb[%d] %010.7f (Lin: %010.7f)\n", absUeIdx, type_convert<TCompute>(tRsrp(absUeIdx)), rsrpLin);
#endif
            }
        }
    } 
 }
/*VCAST_DONT_INSTRUMENT_END*/

template<typename TCompute,
         uint32_t THRD_GRP_TILE_SIZE,            // # of threads in a thread group tile
         uint32_t N_THRD_GRP_TILES_PER_THRD_BLK, // number of thread group tiles in the thread block
         uint32_t N_SC_PER_THRD_BLK_ITER,        // number of subcarriers processed per thread block iteration
         uint32_t N_ITER_PER_THRD_BLK>           // # of iterations in a thread block
void puschRxRssi::rsrpMeasLaunchGeo(uint16_t         nMaxPrb, 
                                    uint16_t         nRxAnt,
                                    uint16_t         nUeGrps,
                                    dim3&            gridDim,
                                    dim3&            blockDim,
                                    size_t&          sharedMemBytes)
{
    // Max subcarrier count
    uint32_t nMaxSc = nMaxPrb*CUPHY_N_TONES_PER_PRB;
    // static constexpr uint32_t N_PIPE_STAGES     = 1; // N_PIPE_STAGES not needed for v1
    static constexpr uint32_t N_SC_PER_THRD_BLK = N_SC_PER_THRD_BLK_ITER*N_ITER_PER_THRD_BLK;

    // Number of thread blocks needed to process a UE group with nMaxPrb allocation
    uint32_t nThrdBlks = div_round_up<uint32_t>(nMaxSc, N_SC_PER_THRD_BLK);
    // for the updated version of rsrpMeasKernel_v1, a minimum of maxLayers*nRxAnt threads are required.
    uint32_t maxLayers = min(MAX_N_LAYERS_PUSCH, nRxAnt);
    if ( nThrdBlks * THRD_GRP_TILE_SIZE < maxLayers * nRxAnt ) {
      nThrdBlks = div_round_up<uint32_t>(maxLayers*nRxAnt, N_SC_PER_THRD_BLK);
    }
    
    blockDim.x = THRD_GRP_TILE_SIZE;
    blockDim.y = N_THRD_GRP_TILES_PER_THRD_BLK;
    blockDim.z = 1;

    gridDim.x = nThrdBlks;
    gridDim.y = nUeGrps;
    gridDim.z = 1;

    /*
    // @todo: For single kernel processing heterogenous number of antennas this needs to be nMaxRxAnt
    // @todo: Replace CUPHY_C_32F with TCompute type
    uint32_t nMaxHEstElems = nRxAnt*nRxAnt*N_SC_PER_THRD_BLK_ITER*CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST;
    uint32_t nMaxReeElems  = nRxAnt*N_SC_PER_THRD_BLK_ITER;
    sharedMemBytes = nMaxHEstElems*N_PIPE_STAGES*sizeof(data_type_traits<CUPHY_C_32F>::type) + // sizeof(complex_from_scalar<TCompute>::type)
                     nMaxReeElems*sizeof(data_type_traits<CUPHY_R_32F>::type);// sizeof(TCompute::type);
    */
    // For the updated version of rsrpMeasKernel_v1, much fewer shared memory is required.
    sharedMemBytes = maxLayers * 2 * sizeof(TCompute);

#ifdef ENABLE_DEBUG
    NVLOGI_FMT(NVLOG_PUSCH, "{}: Thread block dim: gridDim ({} {} {}) blockDim ({} {} {}) sharedMemBytes {}", __FUNCTION__, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z, sharedMemBytes);
#endif    
}

template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute>      
void puschRxRssi::rsrpMeas(uint16_t                     nMaxPrb,
                           uint16_t                     nRxAnt,
                           uint16_t                     nUeGrps,
                           uint8_t                      dmrsSymbolIdx,
                           uint8_t                      enableWeightedAverageCfo,
                           cuphyPuschRxRsrpLaunchCfg_t& launchCfg)
{
    static constexpr uint32_t THRD_GRP_TILE_SIZE            = N_THREADS_PER_WARP;
    static constexpr uint32_t N_THRD_GRP_TILES_PER_THRD_BLK = 4; // 4*32 = 128 threads per thread block, one thread tile per layer
    static constexpr uint32_t N_SC_PER_THRD_BLK_ITER        = CUPHY_N_TONES_PER_PRB; // @todo: high MIMO specialization could process fewer subcarriers    

    // # of iterations within a thread block (amortize launch and housekeeping overhead)
    static constexpr uint32_t N_ITER_PER_THRD_BLK            = 4;
    
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;
    
    if(dmrsSymbolIdx==CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS)
    {
        static constexpr uint8_t DMRS_SYMBOL_IDX    = CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS;
        
        if(enableWeightedAverageCfo==1)
        {
            void* kernelFunc = reinterpret_cast<void*>(rsrpMeasWithoutPostEqMeasKernel_v1<TStorageIn,
                                                                     TStorageOut,
                                                                     TCompute,
                                                                     THRD_GRP_TILE_SIZE,
                                                                     N_THRD_GRP_TILES_PER_THRD_BLK,
                                                                     N_SC_PER_THRD_BLK_ITER,
                                                                     N_ITER_PER_THRD_BLK,
                                                                     DMRS_SYMBOL_IDX>);
            {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));}
        }
        else
        {
            void* kernelFunc = reinterpret_cast<void*>(rsrpMeasKernel_v1<TStorageIn,
                                                                     TStorageOut,
                                                                     TCompute,
                                                                     THRD_GRP_TILE_SIZE,
                                                                     N_THRD_GRP_TILES_PER_THRD_BLK,
                                                                     N_SC_PER_THRD_BLK_ITER,
                                                                     N_ITER_PER_THRD_BLK,
                                                                     DMRS_SYMBOL_IDX>);
            {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));}
        }
    }
    else if(dmrsSymbolIdx==CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS)
    {
        static constexpr uint8_t DMRS_SYMBOL_IDX    = CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS;
        
        if(enableWeightedAverageCfo==1)
        {
            void* kernelFunc = reinterpret_cast<void*>(rsrpMeasWithoutPostEqMeasKernel_v1<TStorageIn,
                                                                     TStorageOut,
                                                                     TCompute,
                                                                     THRD_GRP_TILE_SIZE,
                                                                     N_THRD_GRP_TILES_PER_THRD_BLK,
                                                                     N_SC_PER_THRD_BLK_ITER,
                                                                     N_ITER_PER_THRD_BLK,
                                                                     DMRS_SYMBOL_IDX>);
            {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));}
        }
        else
        {
            void* kernelFunc = reinterpret_cast<void*>(rsrpMeasKernel_v1<TStorageIn,
                                                                     TStorageOut,
                                                                     TCompute,
                                                                     THRD_GRP_TILE_SIZE,
                                                                     N_THRD_GRP_TILES_PER_THRD_BLK,
                                                                     N_SC_PER_THRD_BLK_ITER,
                                                                     N_ITER_PER_THRD_BLK,
                                                                     DMRS_SYMBOL_IDX>);
            {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));}
        }
    }
    else if(dmrsSymbolIdx==CUPHY_PUSCH_RSRP_EST_FIRST_DMRS)
    {
        void* kernelFunc = reinterpret_cast<void*>(sinrMeasEarlyHarqKernel_v1<TStorageIn,
                                                                              TStorageOut,
                                                                              TCompute,
                                                                              THRD_GRP_TILE_SIZE,
                                                                              N_THRD_GRP_TILES_PER_THRD_BLK,
                                                                              N_SC_PER_THRD_BLK_ITER,
                                                                              N_ITER_PER_THRD_BLK>);
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));}
    }
    
    dim3 blockDim, gridDim;
    size_t sharedMemBytes = 0;
    rsrpMeasLaunchGeo<TCompute,
                      THRD_GRP_TILE_SIZE,
                      N_THRD_GRP_TILES_PER_THRD_BLK,
                      N_SC_PER_THRD_BLK_ITER,
                      N_ITER_PER_THRD_BLK>(nMaxPrb, nRxAnt, nUeGrps, gridDim, blockDim, sharedMemBytes);

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;
    
    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra          = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = sharedMemBytes;
}

void puschRxRssi::rsrpMeasKernelSelect(uint16_t                     nMaxPrb,
                                       uint16_t                     nRxAnt,
                                       uint16_t                     nUeGrps,
                                       uint8_t                      dmrsSymbolIdx,
                                       uint8_t                      enableWeightedAverageCfo,
                                       cuphyDataType_t              hEstType,
                                       cuphyDataType_t              rsrpType,
                                       cuphyPuschRxRsrpLaunchCfg_t& launchCfg)
{
    // Note: Output tensor tRsrp used in global memory accumulation. Hence using single precision

    // clang-format off
    using TCompute = float;    
    if(CUPHY_C_32F == hEstType)
    {
        using TStorageIn = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
        if(CUPHY_R_32F == rsrpType)
        {
            using TStorageOut = data_type_traits<CUPHY_R_32F>::type;
            rsrpMeas<TStorageIn, TStorageOut, TCompute>(nMaxPrb,
                                                        nRxAnt,
                                                        nUeGrps,
                                                        dmrsSymbolIdx,
                                                        enableWeightedAverageCfo,
                                                        launchCfg);
        }      
        else
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type - HEst {} Rsrp {}", __FUNCTION__, cuphyGetDataTypeString(hEstType), cuphyGetDataTypeString(rsrpType));
        }
    }
    else
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type - HEst {} Rsrp {}", __FUNCTION__, cuphyGetDataTypeString(hEstType), cuphyGetDataTypeString(rsrpType));
    }
}

void puschRxRssi::getDescrInfo(size_t& rssiDynDescrSizeBytes, size_t& rssiDynDescrAlignBytes, size_t& rsrpDynDescrSizeBytes, size_t& rsrpDynDescrAlignBytes)
{
    rssiDynDescrSizeBytes  = sizeof(puschRxRssiDynDescrVec_t);
    rssiDynDescrAlignBytes = alignof(puschRxRssiDynDescrVec_t);

    rsrpDynDescrSizeBytes  = sizeof(puschRxRsrpDynDescrVec_t);
    rsrpDynDescrAlignBytes = alignof(puschRxRsrpDynDescrVec_t);
}

cuphyStatus_t puschRxRssi::setupRsrpMeas(cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                         cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                         uint16_t                      nUeGrps,
                                         uint32_t		                   nMaxPrb,
                                         uint8_t                       dmrsSymbolIdx,
                                         uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                         puschRxRsrpDynDescrVec_t&     dynDescrVecCpu,
                                         void*                         pDynDescrsGpu,
                                         cuphyPuschRxRsrpLaunchCfgs_t* pLaunchCfgs,
                                         cudaStream_t                  strm)
{
    puschRxRsrpDynDescr_t* pDynDescrVecGpu = static_cast<puschRxRsrpDynDescr_t*>(pDynDescrsGpu);

    if(!pDrvdUeGrpPrmsCpu || !pDrvdUeGrpPrmsGpu || !pDynDescrVecGpu || !pLaunchCfgs) return CUPHY_STATUS_INVALID_ARGUMENT;
    
    if((dmrsSymbolIdx!=CUPHY_PUSCH_RSRP_EST_FIRST_DMRS)&&(dmrsSymbolIdx!=CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS)&&(dmrsSymbolIdx!=CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS))
    {
         NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Invalid dmrsSymbolIdx {}", __FUNCTION__, dmrsSymbolIdx);
         return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    
    uint32_t nMaxRxAnt = 1;
    uint8_t enableWeightedAverageCfo = 0;
    for(uint32_t idxUeGrps = 0; idxUeGrps < nUeGrps; idxUeGrps++)
    {
        if(nMaxRxAnt < pDrvdUeGrpPrmsCpu[idxUeGrps].nRxAnt)
            nMaxRxAnt = pDrvdUeGrpPrmsCpu[idxUeGrps].nRxAnt;
        
        if((enableWeightedAverageCfo!=1) && (pDrvdUeGrpPrmsCpu[idxUeGrps].enableWeightedAverageCfo==1))
        {
            enableWeightedAverageCfo = 1;
        }
    }

    for(uint32_t hetCfgIdx = 0; hetCfgIdx < pLaunchCfgs->nCfgs; ++hetCfgIdx)
    {
        // Setup descriptor in CPU memory
        puschRxRsrpDynDescr_t& dynDescr = dynDescrVecCpu[hetCfgIdx];

        // RSRP is measured over DMRS symbols
        dynDescr.pDrvdUeGrpPrms  = pDrvdUeGrpPrmsGpu;

        puschRxRsrpKernelArgs_t& kernelArgs = (dmrsSymbolIdx==CUPHY_PUSCH_RSRP_EST_FIRST_DMRS) ? m_rsrpKernelArgsArr[0][hetCfgIdx] :
((dmrsSymbolIdx==CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS) ? m_rsrpKernelArgsArr[1][hetCfgIdx] : m_rsrpKernelArgsArr[2][hetCfgIdx]);
        
        kernelArgs.pDynDescr = &pDynDescrVecGpu[hetCfgIdx];

        // Optional descriptor copy to GPU memory
        if(enableCpuToGpuDescrAsyncCpy)
        {
            CUDA_CHECK(cudaMemcpyAsync(&pDynDescrVecGpu[hetCfgIdx], &dynDescr, sizeof(puschRxRsrpDynDescr_t), cudaMemcpyHostToDevice, strm));
        }

        // Select kernel
        //TODO: supporting variable antenna counts
        cuphyPuschRxRsrpLaunchCfg_t& launchCfg = pLaunchCfgs->cfgs[hetCfgIdx];        
        rsrpMeasKernelSelect(nMaxPrb,
                             nMaxRxAnt,//pDrvdUeGrpPrmsCpu[0].nRxAnt,
                             nUeGrps,
                             dmrsSymbolIdx,
                             enableWeightedAverageCfo,
                             pDrvdUeGrpPrmsCpu[0].tInfoHEst.elemType,
                             pDrvdUeGrpPrmsCpu[0].tInfoRsrp.elemType,
                             launchCfg);    
        
        launchCfg.kernelArgs[0] = &kernelArgs.pDynDescr;        
        launchCfg.kernelNodeParamsDriver.kernelParams = &(launchCfg.kernelArgs[0]);
        // printf("cuPHY::puschRsrpMeas::setup - kernelAddr %p grid(x y z) %d %d %d block(x y z) %d %d %d sharedMemBytes %d kernelParams %p kernelArgs[0] %p gpuDescAddr %p hetCfgIdx %d\n", launchCfg.kernelNodeParamsDriver.func, launchCfg.kernelNodeParamsDriver.gridDimX, launchCfg.kernelNodeParamsDriver.gridDimY,  launchCfg.kernelNodeParamsDriver.gridDimZ,  launchCfg.kernelNodeParamsDriver.blockDimX,  launchCfg.kernelNodeParamsDriver.blockDimY,  launchCfg.kernelNodeParamsDriver.blockDimZ, launchCfg.kernelNodeParamsDriver.sharedMemBytes, &(launchCfg.kernelArgs[0]), &(kernelArgs.pDynDescr), &pDynDescrVecGpu[hetCfgIdx], hetCfgIdx);
    }    
    return CUPHY_STATUS_SUCCESS;
}


} // namespace puschRx_rssi
