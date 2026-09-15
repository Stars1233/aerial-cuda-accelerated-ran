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
 * PDCCH TF-signal building blocks shared between embed_pdcch_tf_signal.cu (standalone
 * genPdcchTfSignalKernel and its selector) and pdcch_fused_tx.cu (fusedPdcchTxKernel and
 * its selector), so both compile the same single definition of the RE-mapping loop and
 * of the launch-geometry / dynamic shared-memory contract.
 */

#if !defined(CUPHY_EMBED_PDCCH_TF_SIGNAL_CUH_INCLUDED_)
#define CUPHY_EMBED_PDCCH_TF_SIGNAL_CUH_INCLUDED_

#include <algorithm>

#include "cuphy.h"
#include "tensor_desc.hpp"

namespace embedPdcchTx {

// RE-mapping loop shared by all PDCCH TF-signal variants: scramble+modulate the payload QAM,
// copy DMRS from shared memory, and write both into the slot TF tensor. Inputs s_dmrs_seqs and
// phy_bundles are the prologue outputs (whole-coreset DMRS for this symbol, sorted physical
// bundles of this DCI). All other quantities are re-derived from coreset/DCI params.
// Caller must have performed the blockIdx.x >= n_sym early exit and a barrier making the
// prologue outputs visible to the whole block.
template <typename TComplex>
__device__ __forceinline__ void genPdcchTfSignalMapREs(
    const uint8_t* __restrict__ d_x_tx,
    const uint32_t* __restrict__ d_x_scramSeq,
    const TComplex* __restrict__ s_dmrs_seqs,
    const uint16_t* __restrict__ phy_bundles,
    const PdcchParams& coreset,
    const cuphyPdcchDciPrm_t& dci_params,
    const cuphyPdcchPmWOneLayer_t* __restrict__ pmw_params)
{
    const int      symbol_id   = blockIdx.x;
    const int      n_sym       = coreset.n_sym & 0x3;
    const uint32_t bundle_size = coreset.bundle_size;
    const uint32_t n_f         = coreset.n_f;
    const uint32_t start_rb    = coreset.start_rb;
    const uint32_t start_sym   = coreset.start_sym;

    TComplex* __restrict__ tf_signal = (TComplex*)coreset.slotBufferAddr;

    const uint32_t aggr_level = dci_params.aggr_level;
    const float    beta_qam   = dci_params.beta_qam;

    const uint8_t enablePrcdBf = dci_params.enablePrcdBf;
    uint16_t      pmwPrmIdx    = 0xFFFF;
    uint8_t       nPorts       = 0;
    if(enablePrcdBf)
    {
        pmwPrmIdx = dci_params.pmwPrmIdx;
        nPorts    = pmw_params[pmwPrmIdx].nPorts;
    }
    const uint16_t offset_per_port = n_f * OFDM_SYMBOLS_PER_SLOT;
    const TComplex zeroValue       = make_complex<TComplex>::create(0, 0);

    uint32_t temp = (0x2360 >> (4 * n_sym)) & 0xf; // temp expresses 6 / n_sym
    uint32_t n_rb = temp * aggr_level;

    int pdcch_start_freq = start_rb * 12;
    int total_n_REs      = n_rb * 12;
    int n_qam_per_sym    = n_rb * 9; // For every RB, 9 REs are QAMs and 3 are DMRS. DMRS are in positions 1, 5 and 9 within an RB (0-indexing).

    // Reminder: bundle_size is 2, 3 or 6. n_sym is 1, 2 or 3.
    uint32_t contiguous_rbs = (bundle_size == 6) ? temp : (bundle_size - n_sym + 1); // bundle_size / n_sym

    // Every thread writes one RE. If tid & 0x3 == 1, that's DMRS, everything else is QAM.
    for(int tid = threadIdx.x; tid < total_n_REs; tid += blockDim.x)
    {
        int contiguous_res_chunk_id = tid / (12 * contiguous_rbs);
        int phy_bundle_id           = phy_bundles[contiguous_res_chunk_id];

        int g_w_idx = pdcch_start_freq + (symbol_id + start_sym) * n_f + phy_bundle_id * contiguous_rbs * 12 + (tid % (12 * contiguous_rbs));
        TComplex val;
        if((tid & 0x3) == 1)
        { // map DMRS
            int idxDmrs = phy_bundle_id * contiguous_rbs * 3;
            idxDmrs += (((tid % (12 * contiguous_rbs)) - 1) >> 2);

            val = s_dmrs_seqs[idxDmrs];
        }
        else
        { // map QAM
            int idxQam;
            // find QAM index
            if(tid == 0)
            {
                idxQam = 0 + symbol_id * n_qam_per_sym;
            }
            else
            {
                idxQam = (tid - ((tid - 1) / 4 + 1)) + symbol_id * n_qam_per_sym;
            }
            // scrambling
            int idxBit = 2 * idxQam;
            int x_tx_x = (d_x_tx[idxBit / 8] >> (idxBit % 8)) & 0x1;
            int x_tx_y = (d_x_tx[idxBit / 8] >> ((idxBit + 1) % 8)) & 0x1;

            uint32_t scrambling_val = d_x_scramSeq[idxBit >> 5];
            int      x              = (x_tx_x + (scrambling_val >> (31 - (idxBit & 0x1F)))) & 0x1;
            int      y              = (x_tx_y + (scrambling_val >> (31 - ((idxBit + 1) & 0x1F)))) & 0x1;

            // modulation
            val.x = 0.70710678f * (1 - 2 * x) * beta_qam;
            val.y = 0.70710678f * (1 - 2 * y) * beta_qam;
        }
        if(enablePrcdBf)
        {
            for(int idx = 0; idx < nPorts; idx++)
            {
                tf_signal[g_w_idx + offset_per_port * idx] = __hcmadd(val, pmw_params[pmwPrmIdx].matrix[idx], zeroValue); // uncoalesced writes
            }
        }
        else
        {
            tf_signal[g_w_idx] = val;
        }
    }
}

// Both TF-signal kernels (the standalone genPdcchTfSignalKernel and fusedPdcchTxKernel)
// launch one block per (symbol, DCI) with the same block shape and the same dynamic
// shared-memory layout (whole-coreset DMRS + gold words, sized by slot-wide maxima).
// fusedPdcchTxKernel's warp specialization (warp 0 encode, warp 1 scrambling + bundle map,
// warps 2-3 DMRS halves selected via warp_id - 2) is only correct for exactly 4 warps.
inline constexpr uint32_t PDCCH_TF_BLOCK_THREADS = 128;
static_assert(PDCCH_TF_BLOCK_THREADS == 4 * 32, "fusedPdcchTxKernel's warp specialization requires exactly 4 warps");

// Shared launch-geometry computation for the two TF-signal kernel selectors: this is the
// single definition of the kernel<->host dynamic shared-memory layout contract.
inline void pdcchTfSignalLaunchGeometry(CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver,
                                        uint32_t                 num_DCIs,
                                        int                      num_coresets,
                                        const PdcchParams*       h_coreset_params)
{
    // FIXME temp. get max_rb coreset and max symbols for all coresets
    uint32_t max_rb_coreset = 0;
    uint32_t max_n_sym      = 0;
    for(int coreset_idx = 0; coreset_idx < num_coresets; coreset_idx++)
    {
        max_rb_coreset = std::max(max_rb_coreset, h_coreset_params[coreset_idx].rb_coreset);
        max_n_sym      = std::max(max_n_sym, h_coreset_params[coreset_idx].n_sym);
    }

    // Compute dynamic shared memory size
    size_t s_dmrs_seqs_size = sizeof(__half2) * (max_rb_coreset * 6 * 3);
    size_t s_gold_seqs_size = sizeof(uint32_t) * (((max_rb_coreset * 6 * 6 + 31) / 32) + 2);

    // Max number of symbols is 3. Max number of DCIs is CUPHY_PDCCH_MAX_DCIS_PER_CORESET.
    // Currently, some computations are replicated across symbols for the same DCI.
    kernelNodeParamsDriver.blockDimX = PDCCH_TF_BLOCK_THREADS;
    kernelNodeParamsDriver.blockDimY = 1;
    kernelNodeParamsDriver.blockDimZ = 1;

    kernelNodeParamsDriver.gridDimX = max_n_sym;
    kernelNodeParamsDriver.gridDimY = num_DCIs;
    kernelNodeParamsDriver.gridDimZ = 1;

    kernelNodeParamsDriver.extra          = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = s_dmrs_seqs_size + s_gold_seqs_size;
}

} // namespace embedPdcchTx

#endif // CUPHY_EMBED_PDCCH_TF_SIGNAL_CUH_INCLUDED_
