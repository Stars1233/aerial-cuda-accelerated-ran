/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*
 * Fused PDCCH TX pipeline: fusedPdcchTxKernel (polar encode + rate match + scrambling +
 * TF-signal embedding in a single warp-specialized kernel), its warp-scoped prologue
 * helpers, and the fused kernel selector (cuphyPdcchFusedTxKernelSelect).
 * The block-wide legacy kernels live in embed_pdcch_tf_signal.cu; code shared between the
 * two pipelines comes from embed_pdcch_tf_signal.cuh and polar_encoder_pdcch.cuh.
 */

#include <cassert>

#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_api.h"
#include "cuphy_internal.h"
#include "descrambling.hpp"
#include "descrambling.cuh"
#include "polar_encoder.hpp"
#include "polar_encoder.cuh"
#include "polar_encoder_pdcch.cuh"
#include "embed_pdcch_tf_signal.cuh"
#include "tensor_desc.hpp"
#include "nvlog.hpp"

using namespace cuphy_i;
using namespace descrambling;

namespace embedPdcchTx {

//--------------------------------------------------------------------------------------------------------
// Warp-scoped prologue helpers: warp-synchronous variants of the block-wide prologue of
// genPdcchTfSignalBody (generate_dmrs / coreset-map decode / compute_map + bitonicSort), so the
// whole phase-C prologue can run inside a warp-specialized phase 1 with no block-wide barriers.

// Warp-scoped bitonic sort: same comparator network as polar_encoder::bitonicSort, but with
// __syncwarp() between passes. nSortEntries <= 64 (a single warp provides the <= 32 comparators).
template <typename Tentry = uint16_t>
static __device__ inline void bitonicSortWarp(polar_encoder::SortDir_t sortDir, uint32_t nSortEntries, Tentry* pShEntries)
{
    const uint32_t lane       = threadIdx.x & 31;
    const bool     thrdEnable = (lane < nSortEntries / 2);

    for(uint32_t size = 2; size < nSortEntries; size <<= 1)
    {
        polar_encoder::SortDir_t dir = static_cast<polar_encoder::SortDir_t>(((lane & (size / 2)) != 0));
        for(uint32_t stride = size / 2; stride > 0; stride >>= 1)
        {
            if(thrdEnable)
            {
                uint32_t idx = 2 * lane - (lane & (stride - 1));
                polar_encoder::sortComparator<Tentry>(dir, pShEntries[idx], pShEntries[idx + stride]);
            }
            __syncwarp();
        }
    }
    for(uint32_t stride = nSortEntries / 2; stride > 0; stride >>= 1)
    {
        if(thrdEnable)
        {
            uint32_t idx = 2 * lane - (lane & (stride - 1));
            polar_encoder::sortComparator<Tentry>(sortDir, pShEntries[idx], pShEntries[idx + stride]);
        }
        __syncwarp();
    }
}

// Warp-scoped variant of compute_map (same math, lane-strided, warp sort). See compute_map in
// embed_pdcch_tf_signal.cu.
static __device__ inline void compute_map_warp(uint32_t  bundles_per_level,
                                               uint32_t  aggr_level,
                                               uint32_t  cce_index,
                                               bool      interleaved,
                                               uint32_t  interleaver_size,
                                               uint32_t  shift_index,
                                               uint32_t  C,
                                               uint32_t  n_sym,
                                               uint32_t  n_CCEs,
                                               const uint8_t* log_to_physical_map,
                                               uint16_t* phy_bundles)
{
    const uint32_t lane              = threadIdx.x & 31;
    int            N_bundle          = bundles_per_level * n_CCEs;
    uint32_t bundles_per_coreset_bit = n_sym * bundles_per_level;
    int      round_up_elements       = (bundles_per_level == 3) ? aggr_level * 4 : aggr_level * bundles_per_level;

    for(int i = lane; i < round_up_elements; i += 32)
    {
        uint32_t phy_bundle_id;
        if(i >= (aggr_level * bundles_per_level))
        {
            phy_bundle_id = 0xffff; // Fill remaining elements with largest invalid value so they are sorted last
        }
        else
        {
            uint32_t j          = i / bundles_per_level;
            uint32_t log_bundle = bundles_per_level * (cce_index + j) + (i - j * bundles_per_level);

            if(interleaved)
            {
                uint32_t c = log_bundle / interleaver_size;
                uint32_t r = log_bundle - c * interleaver_size;
                log_bundle = (r * C + c + shift_index) % N_bundle;
            }

            int map_index = log_bundle / bundles_per_coreset_bit;
            phy_bundle_id = bundles_per_coreset_bit * log_to_physical_map[map_index] + (log_bundle - map_index * bundles_per_coreset_bit);
        }
        phy_bundles[i] = phy_bundle_id;
    }
    __syncwarp();

    if(interleaved)
    {
        bitonicSortWarp<uint16_t>(polar_encoder::SortDir_t::ASCENDING, round_up_elements, phy_bundles);
    }
}

// Warp-scoped half of generate_dmrs: one warp computes gold words [W/2*half, ...) and the QPSK
// values whose gold bits live in those words. The split is on a gold-WORD boundary, so the two
// halves touch disjoint gold_seqs/dmrs_seqs ranges (no cross-warp synchronization needed; a DMRS
// value never straddles a word because gold_start_bit is even). Formulas match generate_dmrs.
template <typename TComplex>
static __device__ void generate_dmrs_warp_half(TComplex* __restrict__ dmrs_seqs,
                                               uint32_t* __restrict__ gold_seqs,
                                               uint32_t dmrs_id,
                                               uint32_t n_rb, // whole-coreset RBs (= rb_coreset * 6)
                                               uint32_t start_rb,
                                               float    beta_dmrs,
                                               uint32_t slot_number,
                                               uint32_t start_sym,
                                               uint32_t coreset_type,
                                               uint32_t half) // 0 = lower gold words, 1 = upper
{
    const uint32_t lane                 = threadIdx.x & 31;
    const uint32_t gold_start_bit       = (coreset_type == 0) ? 0 : start_rb * 6;
    const uint32_t gold_start_remainder = (gold_start_bit & 0x1F); // even: gold_start_bit is a multiple of 6
    const uint32_t n_gold_seqs          = n_rb * 2 * 3;
    const uint32_t n_gold_seqs_in_B     = n_gold_seqs / 32 + ((n_gold_seqs & 0x1F) != 0) + (gold_start_remainder != 0);
    const uint32_t n_dmrs_seqs          = n_rb * 3;

    const uint32_t Wh      = n_gold_seqs_in_B / 2;
    int            t_split = (static_cast<int>(32 * Wh) - static_cast<int>(gold_start_remainder)) / 2;
    if(t_split < 0)
    {
        t_split = 0;
    }
    if(t_split > static_cast<int>(n_dmrs_seqs))
    {
        t_split = n_dmrs_seqs;
    }

    const uint32_t g_lo = half ? Wh : 0;
    const uint32_t g_hi = half ? n_gold_seqs_in_B : Wh;
    const uint32_t t_lo = half ? static_cast<uint32_t>(t_split) : 0;
    const uint32_t t_hi = half ? n_dmrs_seqs : static_cast<uint32_t>(t_split);

    const float    dmrs_base = 1 / sqrtf(2.f) * beta_dmrs;
    const uint32_t t         = start_sym + blockIdx.x;
    uint32_t       c_init    = (1 << 17) * (OFDM_SYMBOLS_PER_SLOT * slot_number + t + 1) * (2 * dmrs_id + 1) + (2 * dmrs_id);
    c_init &= ~(1 << 31);

    for(uint32_t g = g_lo + lane; g < g_hi; g += 32)
    {
        gold_seqs[g] = gold32(c_init, gold_start_bit + g * 32);
    }
    __syncwarp();

    for(uint32_t tid = t_lo + lane; tid < t_hi; tid += 32)
    {
        const uint32_t gold_seq_bit     = tid * 2 + gold_start_remainder;
        const int      gold_seqs_idx    = gold_seq_bit >> 5;
        const int      gold_seqs_offset = gold_seq_bit & 0x1F;
        const uint32_t vals             = (gold_seqs[gold_seqs_idx] >> gold_seqs_offset) & 0x3; // 2 bits
        const float    r                = (vals == 1 || vals == 3) ? -dmrs_base : dmrs_base;
        const float    j                = (vals == 2 || vals == 3) ? -dmrs_base : dmrs_base;
        dmrs_seqs[tid]                  = make_complex<TComplex>::create(r, j);
    }
}

// Fused PDCCH TX kernel (fully warp-specialized phase 1):
//   warp 0:  polar encode + rate match (warp-synchronous encodeRateMatchPdcchWarp)  -> s_tx
//   warp 1:  scrambling sequence, then coreset-map decode (popc-parallel) and the
//            bundle map + warp-scoped bitonic sort                                  -> s_scramSeq, phy_bundles
//   warps 2/3: whole-coreset DMRS for this symbol, split on a gold-word boundary so
//            the two halves touch fully disjoint shared-memory ranges               -> s_dmrs_seqs
// No branch of phase 1 contains a block-wide barrier; a single reconvergent __syncthreads()
// joins them and phase 2 is only the RE-mapping loop (genPdcchTfSignalMapREs) on all threads.
template <typename TComplex>
__global__ void fusedPdcchTxKernel(
    const uint8_t* __restrict__ d_input_w_crc,   // bit-reversed payload+CRC, CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC stride per DCI
    cuphyPdcchDciPrm_t* __restrict__ params,
    const uint8_t* __restrict__ d_dci_tm_info,
    const uint16_t* __restrict__ d_cidx2uidx_lut,
    PdcchParams* __restrict__ coreset_params,
    cuphyPdcchPmWOneLayer_t* __restrict__ pmw_params,
    const int* __restrict__ d_coreset_idx_of_dci)
{
    const int    DCI_id      = blockIdx.y; // over all coresets
    const int    coreset_idx = d_coreset_idx_of_dci[DCI_id];
    PdcchParams& coreset     = coreset_params[coreset_idx];
    const int    n_sym       = coreset.n_sym & 0x3;

    if(blockIdx.x >= n_sym) return; //early exit as not all DCIs have the same # of symbols

    cuphyPdcchDciPrm_t& dci_params = params[DCI_id];

    const uint32_t coreset_rb = coreset.rb_coreset;

    __shared__ uint32_t s_warpScratch[polar_encoder::N_MAX_CODED_BITS / 32]; // warp-0 private
    __shared__ uint32_t s_tx[CUPHY_PDCCH_MAX_TX_BITS_PER_DCI_WORD_LEN];      // rate-matched tx bits
    __shared__ uint32_t s_scramSeq[CUPHY_PDCCH_MAX_TX_BITS_PER_DCI_WORD_LEN]; // scrambling sequence
    __shared__ uint8_t  log_to_physical_map[64]; // at most 64. It's actually coreset_rb
    __shared__ uint16_t phy_bundles[64];

    extern __shared__ TComplex shmem[];
    TComplex*                  s_dmrs_seqs = shmem;
    uint32_t*                  s_gold_seqs = reinterpret_cast<uint32_t*>(&s_dmrs_seqs[(coreset_rb * 6 * 3)]);

    const uint32_t warp_id = threadIdx.x >> 5;
    const uint32_t lane    = threadIdx.x & 31;

    if(warp_id == 0)
    {
        //--------------------------------------------------------------------------------------------------------
        // Phase A (warp 0): polar encode + rate match for this DCI.
        for(uint32_t i = lane; i < CUPHY_PDCCH_MAX_TX_BITS_PER_DCI_WORD_LEN; i += 32u)
        {
            s_tx[i] = 0;
        }
        __syncwarp();

        const uint32_t in_offset    = DCI_id * CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC;
        const uint8_t  testing_mode = (d_dci_tm_info[DCI_id >> 3] >> (DCI_id & 0x7)) & 0x1;
        if(testing_mode != 0)
        {
            if(lane < polar_encoder::N_MAX_TM_DCI_TX_BYTES)
            {
                reinterpret_cast<uint8_t*>(s_tx)[lane] = d_input_w_crc[in_offset + lane];
            }
        }
        else
        {
            // Compute # of coded bits (see section "5.3.1 Polar coding" in 3GPP TS 38.212);
            // single shared implementation with encodeRateMatchMultipleDCIsKernel.
            const uint32_t nInfoBits  = CUPHY_PDCCH_N_CRC_BITS + dci_params.Npayload;
            const uint32_t nTxBits    = 2 * 9 * 6 * dci_params.aggr_level;
            const uint32_t nCodedBits = polar_encoder::pdcchPolarNumCodedBits(nInfoBits, dci_params.aggr_level);

            const int       log2_AL        = __ffs(static_cast<int>(dci_params.aggr_level)) - 1;
            const int       c2u_lut_offset = polar_encoder::pdcch_polar_cidx2uidx_lut_elem_offset(static_cast<int>(nInfoBits), log2_AL);
            const uint16_t* pCidx2uidx =
                (d_cidx2uidx_lut != nullptr && c2u_lut_offset >= 0 && nInfoBits <= nTxBits) ? (d_cidx2uidx_lut + static_cast<unsigned>(c2u_lut_offset)) : nullptr;

            // A null LUT or invalid (K, AL) combination means there is nothing to encode: s_tx
            // stays zeroed and this DCI's QAM REs are modulated from zero bits, matching the
            // legacy encodeRateMatchMultipleDCIsKernel, which early-exits the same way. This is
            // defense in depth only — cuphyPdcchPipelinePrepare() rejects invalid payload sizes
            // with CUPHY_STATUS_INVALID_ARGUMENT before anything is launched. The assert makes
            // the condition fail fast in debug builds only (this repo's build configurations
            // define NDEBUG in release and never define a DEBUG macro, so NDEBUG is the guard).
#ifndef NDEBUG
            assert(pCidx2uidx != nullptr);
#endif
            if(pCidx2uidx != nullptr)
            {
                polar_encoder::encodeRateMatchPdcchWarp(nInfoBits, nCodedBits, nTxBits, d_input_w_crc + in_offset, pCidx2uidx, s_warpScratch, s_tx);
            }
        }
    }
    else if(warp_id == 1)
    {
        //--------------------------------------------------------------------------------------------------------
        // Phase B (warp 1): scrambling sequence generation.
        {
            const uint32_t c_init       = ((dci_params.rntiBits << 16) + dci_params.dmrs_id) & 0x7fffffffU;
            const int      tx_bits      = 2 * 9 * 6 * dci_params.aggr_level;
            const int      max_elements = tx_bits / 32 + ((tx_bits % 32 != 0) ? 1 : 0);
            for(int i = static_cast<int>(lane); i < max_elements; i += 32)
            {
                s_scramSeq[i] = __brev(gold32(c_init, i * 32));
            }
        }

        // C-prologue part 1 (warp 1): coreset-map decode. Parallel replacement of the serial
        // thread-0 loop in genPdcchTfSignalBody: log_to_physical_map[j] = position, counted from
        // the top of the rb_coreset-bit window, of the j-th set bit of coreset_map.
        const uint64_t coreset_map = coreset.coreset_map;
        for(uint32_t p = lane; p < coreset_rb; p += 32)
        {
            if((coreset_map >> (coreset_rb - 1 - p)) & 0x1)
            {
                // # set bits above position p. Guard p == 0: shifting a uint64_t by coreset_rb
                // would be UB if a malformed freq_domain_resource ever yields rb_coreset == 64.
                const uint32_t rank       = (p == 0) ? 0 : __popcll(coreset_map >> (coreset_rb - p));
                log_to_physical_map[rank] = p;
            }
        }
        __syncwarp();

        // C-prologue part 2 (warp 1): bundle map + sort for this DCI.
        const uint32_t bundle_size       = coreset.bundle_size;
        const bool     interleaved       = coreset.interleaved;
        const uint32_t interleaver_size  = coreset.interleaver_size;
        const uint32_t n_CCEs            = coreset.n_CCE;
        uint32_t       bundles_per_level = (bundle_size == 6) ? 1 : (bundle_size ^ 0x01); // (6 / bundle_size)
        uint32_t       C                 = interleaved ? (n_CCEs * 6 / (bundle_size * interleaver_size)) : 1;

        compute_map_warp(bundles_per_level,
                         dci_params.aggr_level,
                         dci_params.cce_index,
                         interleaved,
                         interleaver_size,
                         coreset.shift_index,
                         C,
                         n_sym,
                         n_CCEs,
                         &log_to_physical_map[0],
                         &phy_bundles[0]);
    }
    else
    {
        //--------------------------------------------------------------------------------------------------------
        // C-prologue (warps 2/3): whole-coreset DMRS for this symbol, one gold-word-aligned half per warp.
        generate_dmrs_warp_half<TComplex>(s_dmrs_seqs,
                                          s_gold_seqs,
                                          dci_params.dmrs_id,
                                          coreset_rb * 6,
                                          coreset.start_rb,
                                          dci_params.beta_dmrs,
                                          coreset.slot_number,
                                          coreset.start_sym,
                                          coreset.coreset_type,
                                          warp_id - 2);
    }

    __syncthreads(); // phase-1 join: s_tx, s_scramSeq, s_dmrs_seqs and phy_bundles complete

    //--------------------------------------------------------------------------------------------------------
    // Phase 2: RE-mapping loop on all threads
    genPdcchTfSignalMapREs<TComplex>(reinterpret_cast<const uint8_t*>(s_tx), s_scramSeq, s_dmrs_seqs, &phy_bundles[0], coreset, dci_params, pmw_params);
}

// NB: parameter order matches the public cuphyPdcchFusedTxKernelSelect (num_coresets first).
void kernelSelectFusedPdcchTx(cuphyGenPdcchTfSgnlLaunchCfg_t* pLaunchCfg,
                              int                             num_coresets,
                              uint32_t                        num_DCIs,
                              const PdcchParams*              h_coreset_params)
{
    if(pLaunchCfg->kernelNodeParamsDriver.func == nullptr)
    {
        // kernel (only one kernel option for now) and without any specialization, so update function only once.
        // func is set to nullptr in PDCCH channel constructor
        void* kernelFunc = reinterpret_cast<void*>(fusedPdcchTxKernel<__half2>);
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));}
    }

    // Same launch geometry and dynamic shared memory as genPdcchTfSignalKernel;
    // the polar/scrambling buffers of the fused kernel are static shared memory.
    pdcchTfSignalLaunchGeometry(pLaunchCfg->kernelNodeParamsDriver, num_DCIs, num_coresets, h_coreset_params);
}

} // namespace embedPdcchTx

/**
 * Select launch configuration for the fused PDCCH TX kernel.
 *
 * Configures the launch parameters for the fused PDCCH TX kernel that combines
 * encode + rate match + scrambling + TF signal embedding in one launch. Call
 * after cuphyPdcchPipelinePrepare as it consumes the derived rb_coreset/n_sym
 * coreset parameters.
 *
 * @param[out] pLaunchCfg    Pointer to launch configuration to populate.
 * @param[in]  num_coresets  Number of coresets in this slot.
 * @param[in]  num_dcis      Number of DCIs across all coresets.
 * @param[in]  params        Array of coreset parameters with derived fields populated.
 *
 * @return CUPHY_STATUS_SUCCESS on success.
 * @return CUPHY_STATUS_INVALID_ARGUMENT if pLaunchCfg or params is nullptr,
 *         or if num_coresets or num_dcis is non-positive.
 */
cuphyStatus_t cuphyPdcchFusedTxKernelSelect(cuphyGenPdcchTfSgnlLaunchCfg_t* pLaunchCfg,
                                            int                             num_coresets,
                                            int                             num_dcis,
                                            const PdcchParams*              params)
{
    if(!pLaunchCfg || !params)
    {
        NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "cuphyPdcchFusedTxKernelSelect() got nullptr(s)");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if(num_coresets <= 0 || num_dcis <= 0)
    {
        NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "cuphyPdcchFusedTxKernelSelect() got invalid counts: num_coresets {}, num_DCIs {}", num_coresets, num_dcis);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    embedPdcchTx::kernelSelectFusedPdcchTx(pLaunchCfg, num_coresets, static_cast<uint32_t>(num_dcis), params);
    return CUPHY_STATUS_SUCCESS;
}
