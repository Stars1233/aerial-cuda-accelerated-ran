/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//#define CUPHY_DEBUG 1
#include "cuphy_kernel_util.cuh"

#include "soft_demapper.hpp"
#include "soft_demapper.cuh"
#include "soft_demapper_tables.h"
#include "soft_demapper_tables.cuh"
#include "cuphy_context.hpp"
#include "nvlog.hpp"
#include "memfoot_global.h"

namespace
{

////////////////////////////////////////////////////////////////////////
// Texture description, used to create CUDA texture objects for soft
// demapper mipmapped texturers
const cudaTextureDesc s_MipmappedTexDesc =
{
    { cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp}, // addressMode[3]
    cudaFilterModeLinear,                                                // filterMode
    cudaReadModeElementType,                                             // readMode
    0,                                                                   // sRGB
    {0.0f, 0.0f, 0.0f, 0.0f},                                            // borderColor[4]
    1,                                                                   // normalizedCoords
    0,                                                                   // maxAnisotropy
    cudaFilterModePoint,                                                 // mipmapFilterMode
    0,                                                                   // mipmapLevelBias
    0.0f,                                                                // minMipmapLevelClamp
#if CUDART_VERSION >= 11000
    3.0f,                                                                // maxMipmapLevelClamp
    0                                                                    // disableTrilinearOptimization
#else
    3.0f                                                                 // maxMipmapLevelClamp
#endif
};

} // namespace (anonymous)

namespace cuphy_i
{

////////////////////////////////////////////////////////////////////////
// soft_demapper_context::soft_demapper_context()
soft_demapper_context::soft_demapper_context() :
    QAMtex_(channel_format<__half, 4>::desc(), // channel desc
            s_MipmappedTexDesc,                // texture desc
            4,                                 // levels
            32)                                // width
{
    memfoot_add_gpu_size(MF_TAG_CUPHY_OTHER, QAMtex_.get_gpu_size_alloc());
    QAMtex_.copy_to_level(QAM_256_table,
                          sizeof(QAM_256_table),  // pitch
                          0,                      // level
                          cudaMemcpyHostToDevice, // kind
                          0);                     // stream
    QAMtex_.copy_to_level(QAM_64_table,
                          sizeof(QAM_64_table),   // pitch
                          1,                      // level
                          cudaMemcpyHostToDevice, // kind
                          0);                     // stream
    QAMtex_.copy_to_level(QAM_16_table,
                          sizeof(QAM_16_table),   // pitch
                          2,                      // level
                          cudaMemcpyHostToDevice, // kind
                          0);                     // stream
    QAMtex_.copy_to_level(QAM_4_table,
                          sizeof(QAM_4_table),    // pitch
                          3,                      // level
                          cudaMemcpyHostToDevice, // kind
                          0);                     // stream
}

////////////////////////////////////////////////////////////////////////
// soft_demapper_kernel()
template <typename TSymbol, typename TLLR>
__global__ void soft_demapper_kernel(cudaTextureObject_t                   texObj,
                                     float                                 noiseInv,
                                     int                                   QAM_bits,
                                     tensor_ref_t_contig_2D<TLLR>          tLLR,
                                     tensor_ref_t_contig_2D<const TSymbol> tSym)
{
    typedef typename scalar_from_complex<TSymbol>::type             symbol_scalar_t;
    typedef soft_demapper::soft_demapper_any<symbol_scalar_t, TLLR> soft_demapper_t;
    typedef soft_demapper::LLR_group<TLLR, 8>                       llr_group_t;
    typedef soft_demapper::noise_type_map<TLLR>                     noise_type_map_t;

    // LLR output structure. Up to 8 LLRs may be required (for QAM256).
    llr_group_t grp;

    const int SYMBOL_IDX = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int COLUMN_IDX = blockIdx.y;
    (void)COLUMN_IDX;

    if(SYMBOL_IDX >= tSym.layout().dimensions[0])
    {
        return;
    }
    TSymbol softEst = tSym({SYMBOL_IDX, COLUMN_IDX});

    // noiseInv input is the inverse of the (complex, QAM) noise variance
    // PAM_variance = QAM_variance / 2
    // 1 / PAM_variance = inv_PAM_variance = 2 / QAM_variance = 2 * inv_QAM_variance
    soft_demapper_t::symbol_to_LLR_group(grp,                                     // LLR output
                                         softEst,                                 // symbol input
                                         noise_type_map_t::scale(noiseInv, 2.0f), // PAM noise var inverse
                                         QAM_bits,                                // QAM bits
                                         texObj);                                 // CUDA texture object
    KERNEL_PRINT_GRID_ONCE("LLR_tex = (%f, %f), noiseInv = %f --> (%f %f %f %f  %f %f %f %f)\n",
                           (float)softEst.x,
                           (float)softEst.y,
                           noiseInv,
                           grp[0],
                           grp[1],
                           grp[2],
                           grp[3],
                           grp[4],
                           grp[5],
                           grp[6],
                           grp[7]);
    // Write to output (global) memory. Note that as written below, this writes
    // all 8 values, whether valid for the given QAM or not.
    grp.write(tLLR.addr() + tLLR.layout().offset({SYMBOL_IDX * QAM_bits, COLUMN_IDX}), QAM_bits);

}

////////////////////////////////////////////////////////////////////////
// launch_soft_demapper_kernel()
template <typename TSymbol, typename TLLR>
cuphyStatus_t launch_soft_demapper_kernel(const dim3&         grdDim,
                                          const dim3&         blkDim,
                                          cudaTextureObject_t texObj,
                                          tensor_desc&        tLLR,
                                          void*               pLLR,
                                          tensor_desc&        tSym,
                                          const void*         pSym,
                                          int                 log2_QAM,
                                          float               noiseVariance,
                                          cudaStream_t        strm)
{
    //------------------------------------------------------------------
    // C++ data types for symbols and LLR from the cuphyDataType_t parameter
    typedef TSymbol                                     symbol_t;
    typedef typename scalar_from_complex<TSymbol>::type symbol_scalar_t;
    typedef TLLR                                        llr_t;
    //------------------------------------------------------------------
    // Check whether the tensor descriptors can be converted to a 2D
    // contiguous tensor
    cuphy_optional<tensor_ref_t_contig_2D<const TSymbol>> tSymOpt =  tSym.get_ref_contig_rank_t<const TSymbol, 2>(pSym);
    cuphy_optional<tensor_ref_t_contig_2D<TLLR>>          tLLROpt =  tLLR.get_ref_contig_rank_t<TLLR,          2>(pLLR);
    if(!tSymOpt || !tLLROpt)
    {
        return CUPHY_STATUS_UNSUPPORTED_LAYOUT;
    }
    //------------------------------------------------------------------
    soft_demapper_kernel<symbol_t, llr_t><<<grdDim, blkDim, 0, strm>>>(texObj,               // texture object
                                                                       1.0f / noiseVariance, // inv_QAM_variance
                                                                       log2_QAM,             // QAM_bits
                                                                       tLLROpt.value(),      // LLR tensor
                                                                       tSymOpt.value());     // symbols tensor
                                                                       
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        //NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: CUDA Error {} for soft_demapper_kernel", __FUNCTION__, cudaGetErrorString(err));
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// soft_demap()
cuphyStatus_t soft_demap(context&     ctx,
                         tensor_desc& tLLR,
                         void*        pLLR,
                         tensor_desc& tSym,
                         const void*  pSym,
                         int          log2_QAM,
                         float        noiseVariance,
                         cudaStream_t strm)
{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    int NUM_SYMBOLS = tSym.layout().dimensions[0];
    int NUM_LLR     = tLLR.layout().dimensions[0];
    int NUM_COL     = tSym.layout().rank() > 1    ?
                      tSym.layout().dimensions[1] :
                      1;
    //------------------------------------------------------------------
    // Validate inputs
    if(NUM_LLR < (NUM_SYMBOLS * log2_QAM))
    {
        // Not enough space for LLRs given the number of symbols and the
        // QAM
        return CUPHY_STATUS_SIZE_MISMATCH;
    }
    //------------------------------------------------------------------

    dim3 blkDim(1024);
    dim3 grdDim(div_round_up(NUM_SYMBOLS, 1024), NUM_COL);
    
    if(CUPHY_R_16F == tLLR.type())
    {
        if(CUPHY_C_16F == tSym.type())
        {
            s = launch_soft_demapper_kernel<__half2, __half>(grdDim,
                                                             blkDim,
                                                             ctx.soft_demapper_ctx().QAM_tex().tex_obj().handle(),
                                                             tLLR,
                                                             pLLR,
                                                             tSym,
                                                             pSym,
                                                             log2_QAM,
                                                             noiseVariance,
                                                             strm);
        }
        else if(CUPHY_C_32F == tSym.type())
        {
            s = launch_soft_demapper_kernel<cuFloatComplex, __half>(grdDim,
                                                                    blkDim,
                                                                    ctx.soft_demapper_ctx().QAM_tex().tex_obj().handle(),
                                                                    tLLR,
                                                                    pLLR,
                                                                    tSym,
                                                                    pSym,
                                                                    log2_QAM,
                                                                    noiseVariance,
                                                                    strm);
        }
    }
    else if(CUPHY_R_32F == tLLR.type())
    {
        if(CUPHY_C_16F == tSym.type())
        {
            s = launch_soft_demapper_kernel<__half2, float>(grdDim,
                                                            blkDim,
                                                            ctx.soft_demapper_ctx().QAM_tex().tex_obj().handle(),
                                                            tLLR,
                                                            pLLR,
                                                            tSym,
                                                            pSym,
                                                            log2_QAM,
                                                            noiseVariance,
                                                            strm);
        }
        else if(CUPHY_C_32F == tSym.type())
        {
            s = launch_soft_demapper_kernel<cuFloatComplex, float>(grdDim,
                                                                   blkDim,
                                                                   ctx.soft_demapper_ctx().QAM_tex().tex_obj().handle(),
                                                                   tLLR,
                                                                   pLLR,
                                                                   tSym,
                                                                   pSym,
                                                                   log2_QAM,
                                                                   noiseVariance,
                                                                   strm);
        }
    }
    if(CUPHY_STATUS_SUCCESS != s)
    {
        return s;
    }
#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}


} // namespace cuphy_i
