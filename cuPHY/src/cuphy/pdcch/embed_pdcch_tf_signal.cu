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

#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_api.h"
#include "cuphy_internal.h"
#include "descrambling.hpp"
#include "descrambling.cuh"
#include "crc.hpp"
#include "polar_encoder.hpp"
#include "polar_encoder.cuh"

#include "tensor_desc.hpp"
#include "nvlog.hpp"


using namespace cuphy_i;
using namespace descrambling;

namespace cg = cooperative_groups;

namespace embedPdcchTx {

#define G_CRC_24_C 0x01B2B117
#define Nc 1600
// ToDo define #DBG_PRINT_PDCCH_BUNDLES to allow some host-side info printed about the bundles and freq. allocation

uint32_t find_rightmost_bit(uint64_t val)
{
    return log2((val & (val - 1)) ^ val); // 0-indexing
}

uint32_t count_set_bits(uint64_t val)
{
    uint64_t n   = val;
    uint32_t cnt = 0;
    while(n != 0)
    {
        n = n & (n - 1);
        cnt += 1;
    }
    return cnt;
}

// reverse bit order within each byte
int pdcchReverseBitInByte(uint8_t* inputByte, uint8_t* outputByte, uint32_t Nbyte, int payload_bits)
{
    int payload_bytes = payload_bits / 8;
    int rem_bits      = payload_bits - payload_bytes * 8;
    for(int idxByte = 0; idxByte < Nbyte; idxByte++)
    {
        outputByte[idxByte] = 0;
        for(int idxBit = 0; idxBit < 8; idxBit++)
        {
            outputByte[idxByte] <<= 1;
            // Should not read beyond the bytes corresponding to payload_bits from inputByte.
            if(idxByte < payload_bytes)
            {
                outputByte[idxByte] |= ((inputByte[idxByte] >> idxBit) & 1);
            }
            else if((idxByte == payload_bytes) && (rem_bits != 0))
            {
                // Reminder: if the payload_bits is not evenly divisible by 8
                // and has for example a remainder of rem_bits, these would be the rem_bits
                // most significant bits of the payload's last byte.
                if(idxBit >= (8 - rem_bits))
                {
                    outputByte[idxByte] |= ((inputByte[idxByte] >> idxBit) & 1);
                }
            }
        }
    }
    return 0;
}

// pdcchPrintByte was never used in the codebase
/*VCAST_DONT_INSTRUMENT_START*/
int pdcchPrintByte(uint8_t* inputByte, uint32_t Nbyte, bool rev_bits, int payload_bits)
{
    for(int idxByte = 0; idxByte < Nbyte; idxByte++)
    {
        uint8_t val = inputByte[idxByte];
        uint8_t new_val;
        if(rev_bits)
        { // reverse bit order in byte
            pdcchReverseBitInByte(&val, &new_val, 1, payload_bits);
            val = new_val;
        }
        printf("element %d val %x\n", idxByte, val);
    }
    return 0;
}
/*VCAST_DONT_INSTRUMENT_END*/

// generate CRC bits for pdcch payload
// Same as src/cuphy/crc/crc.hpp except this one can handle sizeBit which is not devisible by 8
template <typename uintCRC_t, size_t uintCRCBitLength>
uintCRC_t computePdcchCRC(const uint8_t* input,
                          uint32_t       sizeBit,
                          uintCRC_t      poly,
                          uintCRC_t      initVal          = 0,
                          uint64_t       stride           = 1,
                          bool           include_crc_ones = true)
{
    //NVLOGI_FMT(NVLOG_PDCCH, "initVal {}, include_crc_ones {}", initVal, include_crc_ones);
    uintCRC_t crc           = initVal;
    uintCRC_t msbMask       = (1 << (uintCRCBitLength - 1));
    uintCRC_t allOnesMask   = static_cast<uintCRC_t>(-1);
    int       extra_bits    = include_crc_ones ? 0 : uintCRCBitLength; // Are uintCRCBitLength one's already prepended in the input? If not count them as extra
    int       new_size_bits = sizeBit + extra_bits;
    int       size_byte     = (new_size_bits + 7) / 8;

    if((sizeof(uintCRC_t) * 8 - uintCRCBitLength) > 0)
        allOnesMask >>= (sizeof(uintCRC_t) * 8 - uintCRCBitLength - 1);

    int extra_bytes = extra_bits / 8; // Assumes extra_bits is evenly disible by 8; it is for 24.
    for(int i = 0; i < size_byte * stride; i += stride)
    {
        uint8_t input_val;
        if(extra_bytes > 0)
        {
            input_val = (i < (extra_bytes * stride)) ? 0xffu : input[i - extra_bytes * stride];
        }
        // Not covered via public API: computePdcchCRC() is called with include_crc_ones=false in production path, so extra_bytes>0 and this branch is not reached.
        /*VCAST_DONT_INSTRUMENT_START*/
        else
        {
            input_val = input[i];
        }
        /*VCAST_DONT_INSTRUMENT_END*/

        //printf("i %d input_val %x\n", i, input_val);
        crc ^= static_cast<uintCRC_t>(input_val << (uintCRCBitLength - 8));
        for(int b = 0; b < 8; b++)
        {
            if(i * 8 + b < new_size_bits)
            {
                uintCRC_t pred = (crc & msbMask) == 0;
                crc <<= 1;
                crc ^= (poly & (pred + allOnesMask));
            }
        }
    }
    return crc;
}

void pdcchAddCrc(uint8_t* dci_payload, uint32_t& dci_crc, const uint32_t rnti_crc, const uint32_t payload_bits)
{
    // Compute the 24-bit CRC for dci_payload. The function itself prepends 24-bits of 1's to the payload.
    dci_crc = computePdcchCRC<uint32_t, 24>(dci_payload, payload_bits, G_CRC_24_C, 0, 1, false);
    //printf("CRC before scrambling is %x\n", dci_crc);

    // Scramble 16 least significant bits of 24-bits CRC with RNTI
    dci_crc ^= (rnti_crc & 0x0FFFFU);
}

// generate gold sequence with given init state and length
// pdcchGenGoldSeq was never used in the codebase
/*VCAST_DONT_INSTRUMENT_START*/
void pdcchGenGoldSeq(uint32_t c_init, uint32_t len, uint32_t* x1, uint32_t* x2, uint32_t* c)
{
    for(int i = 0; i < 32; i++)
    {
        x1[i] = 0;
        x2[i] = (c_init >> i) & 0x1;
    }
    x1[0] = 1;

    for(int i = 0; i < Nc + len - 31; i++)
    {
        x1[i + 31] = (x1[i + 3] + x1[i]) % 2;                         // x1(n + 31) = mod(x1(n + 3) + x1(n),2)
        x2[i + 31] = (x2[i + 3] + x2[i + 2] + x2[i + 1] + x2[i]) % 2; // x2(n + 31) = mod(x2(n + 3) + x2(n + 2) + x2(n + 1) + x2(n),2)
    }

    for(int i = 0; i < len; i += 32)
    {
        //Pack 32 bits together
        uint32_t val = 0;
        for(int offset = 0; ((offset < 32) && ((i + offset) < len)); offset++)
        {
            uint32_t bit = (x1[i + offset + Nc] + x2[i + offset + Nc]) % 2; // c(n) = mod(x1(n + Nc) + x2(n + Nc),2)
            val |= (bit << (31 - offset));
        }
        c[i >> 5] = val;
    }
}
/*VCAST_DONT_INSTRUMENT_END*/

// PDCCH scrambling sequence x_scramSeq used to be generated on the host in cuphyPdcchPipelinePrepare. Now it is done during Run
// genPdcchScramSeq was never used in the codebase
/*VCAST_DONT_INSTRUMENT_START*/
void genPdcchScramSeq(uint32_t dmrs_id, uint32_t rnti_bits, uint32_t Nscram, uint32_t* x_scramSeq)
{
    uint32_t cinit = ((rnti_bits << 16) + dmrs_id) & 0x7fffffffU;

    uint32_t x1[Nc + Nscram];
    uint32_t x2[Nc + Nscram];

    pdcchGenGoldSeq(cinit, Nscram, x1, x2, x_scramSeq);
}
/*VCAST_DONT_INSTRUMENT_END*/

template <typename TComplex, typename Block>
__device__ void generate_dmrs(TComplex* __restrict__ dmrs_seqs, // 3 * n_rb
                              uint32_t* __restrict__ gold_seqs, // 6 * n_rb / 32
                              uint32_t dmrs_id,
                              uint32_t n_rb,
                              uint32_t start_rb,
                              float    beta_dmrs,
                              uint32_t slot_number,
                              uint32_t n_sym,
                              uint32_t start_sym,
                              Block&   block,
                              uint32_t coreset_type)
{
    uint32_t gold_start_bit       = (coreset_type == 0) ? 0 : start_rb * 6;
    uint32_t gold_start_remainder = (gold_start_bit & 0x1F);
    uint32_t n_gold_seqs          = n_rb * 2 * 3;
    uint32_t n_gold_seqs_in_B     = n_gold_seqs / 32 + ((n_gold_seqs & 0x1F) != 0) + (gold_start_remainder != 0);
    uint32_t n_dmrs_seqs          = n_rb * 3;

    float    dmrs_base = 1 / sqrtf(2.f) * beta_dmrs;
    uint32_t symbol_id = blockIdx.x;

    uint32_t t      = start_sym + symbol_id;
    uint32_t c_init = (1 << 17) * (OFDM_SYMBOLS_PER_SLOT * slot_number + t + 1) * (2 * dmrs_id + 1) + (2 * dmrs_id);
    c_init &= ~(1 << 31);

    // Step 1.1. compute gold sequence (in shared memory)
    for(int tid = threadIdx.x; tid < n_gold_seqs_in_B; tid += blockDim.x)
    {
        gold_seqs[tid] = gold32(c_init, gold_start_bit + tid * 32);
        //printf("tid %d, threadIdx.x %d, symbol %d, gold32(c_init %x, n %d) has val %x\n",
        //       tid,  threadIdx.x, symbol_id, c_init, gold_start_bit + tid * 32, gold_seqs[tid]);
    }
    block.sync();

    // Step 1.2. qpsk modulate and scale power (in registers)
    for(int tid = threadIdx.x; tid < n_dmrs_seqs; tid += blockDim.x)
    {
        const uint32_t gold_seq_bit     = tid * 2 + gold_start_remainder;
        const int      gold_seqs_idx    = gold_seq_bit >> 5;
        const int      gold_seqs_offset = gold_seq_bit & 0x1F;
        const uint32_t vals             = (gold_seqs[gold_seqs_idx] >> gold_seqs_offset) & 0x3; // 2 bits
        //printf("threadIdx.x %d, tid %d, gold_seq_idx %d, golds_seqs_offset %d, gold_start_remainder %d\n", threadIdx.x, tid, gold_seqs_idx, gold_seqs_offset, gold_start_remainder);
        const float r  = (vals == 1 || vals == 3) ? -dmrs_base : dmrs_base;
        const float j  = (vals == 2 || vals == 3) ? -dmrs_base : dmrs_base;
        dmrs_seqs[tid] = make_complex<TComplex>::create(r, j);
        //printf("seq %d has %f and %f\n", tid, (float)r, (float)j);
    }
    block.sync();
}

__device__ inline void compute_map(uint32_t  bundles_per_level,
                                   uint32_t  aggr_level,
                                   uint32_t  cce_index,
                                   bool      interleaved,
                                   uint32_t  interleaver_size,
                                   uint32_t  shift_index,
                                   uint32_t  C,
                                   uint32_t  n_sym,
                                   uint32_t  n_CCEs,
                                   uint8_t*  log_to_physical_map,
                                   uint16_t* phy_bundles)
{
    // There are {1, 2, 4, 8, 16} aggregation levels each with {1, 2, 3} bundles.
    // Will need to sort at most 48 elements.
    // bundles_per_level is 6 / bundle_size, values are 1, 2 or 3.
    int      N_bundle                = bundles_per_level * n_CCEs;
    uint32_t bundles_per_coreset_bit = n_sym * bundles_per_level;

    // Round up elements to next power of 2, for the sort. That is only applicable if bundles_per_level == 3.
    int round_up_elements = (bundles_per_level == 3) ? aggr_level * 4 : aggr_level * bundles_per_level;

    for(int i = threadIdx.x; i < round_up_elements; i += blockDim.x)
    {
        uint32_t phy_bundle_id;
        if(i >= (aggr_level * bundles_per_level))
        {
            phy_bundle_id = 0xffff; // Fill remaining elements with largest invalid value so they are sorted last
        }
        else
        {
            uint32_t j          = i / bundles_per_level; // reminder aggregation level is power of 2
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
        //printf("block %d, %d, phy_bundles[%d] = %d\n", blockIdx.x, blockIdx.y, i, phy_bundles[i]);
    }

    __syncthreads();

    // Sort physical bundles. Only needed in interleaved mode.
    if(interleaved)
    {
        //bitonicSort_v2(SortDir_t::ASCENDING, round_up_elements, phy_bundles);
        polar_encoder::bitonicSort<uint16_t>(polar_encoder::SortDir_t::ASCENDING, round_up_elements, phy_bundles);
        __syncthreads();
    }
}

__global__ void genScramblingSeqKernel(uint32_t* __restrict__ d_x_scramSeq_addr,
                                 cuphyPdcchDciPrm_t* __restrict__ params)
{
    const int DCI_id = blockIdx.x; // over all coresets
    const int tid    = threadIdx.x;

    // Every DCI has its own c_init value.
    uint32_t c_init = ((params[DCI_id].rntiBits << 16) + params[DCI_id].dmrs_id) & 0x7fffffffU;

    uint32_t offset       = DCI_id * (CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 32);
    int      tx_bits      = 2 * 9 * 6 * params[DCI_id].aggr_level;
    int      max_elements = tx_bits / 32 + ((tx_bits % 32 != 0) ? 1 : 0);
    // Scrambling seq. length in bits is 2 * 9 * 6 * aggr_level bits, so 54 uint32_t elements worst case for aggr. level 16.
    // There may be some bits that won't be used if tx_bits not evenly divisible by 32, but it's not necessary to mask them out here.
    if(tid < max_elements)
    {
        d_x_scramSeq_addr[offset + tid] = __brev(gold32(c_init, tid * 32));
    }
}

// scrambling payload QAM, generate DMRS and map them to tfSignal
template <typename TComplex>
__global__ void genPdcchTfSignalKernel(
    uint8_t* __restrict__ d_x_tx_addr,
    uint32_t* __restrict__ d_x_scramSeq_addr,
    uint32_t num_coresets,
    PdcchParams* __restrict__ coreset_params,
    cuphyPdcchDciPrm_t* __restrict__ params,
    cuphyPdcchPmWOneLayer_t* __restrict__  pmw_params)
{
    // This kernel launches one thread block per symbol per DCI.
    // Reminder: at most 3 symbols possible.
    const int symbol_id = blockIdx.x;
    const int DCI_id    = blockIdx.y; // over all coresets

    // Temp. FIXME find coreset
#if 1
    int  coreset_idx;
    bool found = false;
    for(int i = 0; (i < num_coresets) && !found; i++)
    {
        if((DCI_id >= coreset_params[i].dciStartIdx) && (DCI_id < (coreset_params[i].dciStartIdx + coreset_params[i].num_dl_dci)))
        {
            found       = true;
            coreset_idx = i;
        }
    }
#else
    __shared__ int coreset_idx;
    for(int i = threadIdx.x; i < num_coresets; i += blockDim.x)
    {
        if((DCI_id >= coreset_params[i].dciStartIdx) && (DCI_id < (coreset_params[i].dciStartIdx + coreset_params[i].num_dl_dci)))
        {                    // Only one coreset will satisfy this condition
            coreset_idx = i; // Each thread block corresponds to a single DCI and a symbol; only one thread in a thread block will update this
        }
    }
    __syncthreads();
#endif

    PdcchParams&   coreset          = coreset_params[coreset_idx];
    const int      n_sym            = coreset.n_sym & 0x3;
    const uint32_t bundle_size      = coreset.bundle_size;
    const uint32_t interleaver_size = coreset.interleaver_size;
    const uint32_t shift_index      = coreset.shift_index;

    if(blockIdx.x >= n_sym) return; //early exit as not all DCIs have the same # of symbols

    const uint32_t n_f         = coreset.n_f;
    const uint32_t slot_number = coreset.slot_number;
    const uint32_t start_rb    = coreset.start_rb;
    const uint32_t start_sym   = coreset.start_sym;

    const bool     interleaved = coreset.interleaved;
    const uint64_t coreset_map = coreset.coreset_map;

    const uint32_t n_CCEs       = coreset.n_CCE;
    const uint32_t coreset_rb   = coreset.rb_coreset;
    const uint32_t coreset_type = coreset.coreset_type;

    TComplex* __restrict__ tf_signal = (TComplex*)coreset.slotBufferAddr;

    // Read per DCI config. parameters
    cuphyPdcchDciPrm_t& dci_params = params[DCI_id];
    uint32_t            dmrs_id    = dci_params.dmrs_id;
    uint32_t            aggr_level = dci_params.aggr_level;
    uint32_t            cce_index  = dci_params.cce_index;
    float               beta_qam   = dci_params.beta_qam;
    float               beta_dmrs  = dci_params.beta_dmrs;

    // Read precoding matrix info
    const uint8_t enablePrcdBf  = dci_params.enablePrcdBf;
    uint16_t pmwPrmIdx = 0xFFFF;
    uint8_t nPorts     = 0;
    if(enablePrcdBf)
    {
        pmwPrmIdx = dci_params.pmwPrmIdx;
        nPorts = pmw_params[pmwPrmIdx].nPorts;
    }
    const uint16_t offset_per_port = n_f * OFDM_SYMBOLS_PER_SLOT;
    const TComplex zeroValue = make_complex<TComplex>::create(0,0);

    //uint32_t n_rb = (6 / n_sym) * aggr_level; // Reminder n_sym values are 1, 2 or 3
    uint32_t temp = (0x2360 >> (4 * n_sym)) & 0xf; // temp expresses 6 / n_sym
    uint32_t n_rb = temp * aggr_level;

    uint32_t* __restrict__ d_x_scramSeq = d_x_scramSeq_addr + DCI_id * (CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 32);
    uint8_t* __restrict__ d_x_tx        = d_x_tx_addr + DCI_id * (CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 8);

    extern __shared__ TComplex shmem[];
    TComplex*                  s_dmrs_seqs = shmem;                                         // (n_rb * 3) * 2 * n_sym
    uint32_t*                  s_gold_seqs = (uint32_t*)&s_dmrs_seqs[(coreset_rb * 6 * 3)]; // ceil(n_rb * 6 / 32) * 2 +

    __shared__ uint8_t log_to_physical_map[64]; // at most 64. It's actually coreset_rb
    if(threadIdx.x == 0)
    {
        uint64_t tmp_map       = coreset_map;
        int      first_set_bit = __clzll(tmp_map); //counting from msb
        tmp_map <<= (first_set_bit + 1);
        first_set_bit -= (64 - coreset_rb);
        log_to_physical_map[0] = first_set_bit;
        int cnt                = 1;
        for(int i = 1; i < coreset_rb; i++)
        {
            if((tmp_map >> 63) == 1)
            {
                log_to_physical_map[cnt] = first_set_bit + i;
                cnt++;
            }
            tmp_map <<= 1;
        }
    }
    __syncwarp();
    cg::thread_block block = cg::this_thread_block();

    generate_dmrs(s_dmrs_seqs, s_gold_seqs, dmrs_id, coreset_rb * 6, start_rb, beta_dmrs, slot_number, n_sym, start_sym, block, coreset_type);

    __shared__ uint16_t phy_bundles[64];

    // Reminder: bundle size can be 2 or 3 or 6
    //uint32_t bundles_per_level = (6 / bundle_size);
    uint32_t bundles_per_level = (bundle_size == 6) ? 1 : (bundle_size ^ 0x01); // (6 / bundle_size)
    //uint32_t bundles_per_level = (bundle_size ^ (bundle_size >> 1)) & 0x3; // (6 / bundle_size)
    uint32_t C = interleaved ? (n_CCEs * 6 / (bundle_size * interleaver_size)) : 1;

    // Compute the physical bundles used for this DCI. These are stored in sorted ascending order in phy_bundles.
    compute_map(bundles_per_level,
                aggr_level,
                cce_index,
                interleaved,
                interleaver_size,
                shift_index,
                C,
                n_sym,
                n_CCEs,
                &log_to_physical_map[0],
                &phy_bundles[0]);

    int pdcch_start_freq = start_rb * 12;
    int total_n_REs      = n_rb * 12;
    int n_qam_per_sym    = n_rb * 9; // For every RB, 9 REs are QAMs and 3 are DMRS. DMRS are in positions 1, 5 and 9 within an RB (0-indexing).

    // Reminder: bundle_size is 2, 3 or 6. n_sym is 1, 2 or 3.
    //uint32_t contiguous_rbs = bundle_size / n_sym;
    uint32_t contiguous_rbs = (bundle_size == 6) ? temp : (bundle_size - n_sym + 1); // bundle_size / n_sym

    // Every thread writes one RE. If tid & 0x3 == 1, that's DMRS, everything else is QAM.
    for(int tid = threadIdx.x; tid < total_n_REs; tid += blockDim.x)
    {
        int contiguous_res_chunk_id = tid / (12 * contiguous_rbs);
        int phy_bundle_id           = phy_bundles[contiguous_res_chunk_id];

        int g_w_idx = pdcch_start_freq + (symbol_id + start_sym) * n_f + phy_bundle_id * contiguous_rbs * 12 + (tid % (12 * contiguous_rbs));
        //printf("threadIdx.x %d, symbol %d, DCI %d, writing phy_bundle_id %d to freq %d RE for that symbol\n",
        //        threadIdx.x, symbol_id + start_sym, blockIdx.y, phy_bundle_id, g_w_idx -  (symbol_id + start_sym) * n_f);
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
                tf_signal[g_w_idx + offset_per_port*idx] = __hcmadd(val, pmw_params[pmwPrmIdx].matrix[idx], zeroValue); // uncoalesced writes
            }
        }
        else
        {
            tf_signal[g_w_idx] = val;
        }
        
    }
}

// dbg_print was never used in the codebase
/*VCAST_DONT_INSTRUMENT_START*/
void dbg_print(PdcchParams* h_coreset_params, cuphyPdcchDciPrm_t* h_dci_params, int num_coresets, int num_dcis)
{
    for(int coreset_idx = 0; coreset_idx < num_coresets; coreset_idx++)
    {
        PdcchParams* params = h_coreset_params + coreset_idx;
        printf("\n");
        printf("Coreset %d\n", coreset_idx);
        uint32_t bundles_per_coreset_bit = 6 * params->n_sym / params->bundle_size;
        uint32_t N_bundle                = params->n_CCE * bundles_per_coreset_bit / params->n_sym; // N_bundle counts all the set bits in coreset map
        uint32_t N_bundle_phy            = params->rb_coreset * bundles_per_coreset_bit;            // N_bundle_phy counts all of them
        printf("# physical bundles %d, # logical bundles %d\n", N_bundle_phy, N_bundle);

        uint32_t bundle_table[N_bundle] = {0};
        uint32_t bundle_map[N_bundle]   = {0};

        // Map all logical bundles (i.e., allocated ones) to physical ones
        int log_bundle_id = 0;
        int phy_bundle_id = 0;
        for(int i = 0; i < params->rb_coreset; i++)
        {
            if((params->coreset_map >> (params->rb_coreset - i - 1)) & 0x1)
            {
                //printf("-----New coreset bit--------\n");
                for(int j = 0; j < bundles_per_coreset_bit; j++)
                {
                    bundle_table[log_bundle_id + j] = phy_bundle_id + j;
                    printf("Logical bundle %d maps to physical bundle %d\n", log_bundle_id + j, phy_bundle_id + j);
                }
                log_bundle_id += bundles_per_coreset_bit;
            }
            phy_bundle_id += bundles_per_coreset_bit;
        }
        printf("\n\n");

        // Map each logical bundle to a different logical bundle if we're in interleaved mode.
        // No mapping change if non-interleaved.
        uint32_t C = (params->interleaved) ? params->n_CCE * 6 / (params->bundle_size * params->interleaver_size) : 1;
        // Sanity check for C
        /*if (params->interleaved && ((params->n_CCE * 6) % (params->bundle_size * params->interleaver_size) != 0)) {
        status = CUPHY_STATUS_INVALID_ARGUMENT;
        return status;
    }*/
        for(int i = 0; i < N_bundle; i++)
        {
            uint32_t new_bundle_id = i;
            if(params->interleaved)
            {
                uint32_t c    = i / params->interleaver_size;
                uint32_t r    = i % params->interleaver_size;
                new_bundle_id = (r * C + c + params->shift_index) % N_bundle;
                printf("Logical bundle %d maps to new logical bundle %d in interleaved mode\n", i, new_bundle_id);
            }
            bundle_map[i] = new_bundle_id;
        }
        printf("\n\n");

        int num_DCIs = params->num_dl_dci;
        for(int i = 0; i < num_DCIs; i++)
        {
            //PdcchDciParams dci_params = params->pDciParams[i];
            cuphyPdcchDciPrm_t dci_params                    = h_dci_params[i + params->dciStartIdx];
            uint32_t           used_bundle_map[N_bundle_phy] = {0};
            for(int j = 0; j < dci_params.aggr_level; j++)
            {
                for(int used_bundle = 0; used_bundle < 6 / params->bundle_size; used_bundle++)
                {
                    uint32_t log_bundle_id                       = bundle_map[(6 / params->bundle_size) * (dci_params.cce_index + j) + used_bundle];
                    used_bundle_map[bundle_table[log_bundle_id]] = 1;
                    printf("DCI %d: physical bundle %d is used, orig. logical bundle %d, new logical bundle  %d. ",
                           i,
                           bundle_table[log_bundle_id],
                           (6 / params->bundle_size) * (dci_params.cce_index + j) + used_bundle,
                           log_bundle_id);
                    // What does it mean to say "physical bundle X" is used?
                    // It coresponds to a bit of coreset_map * 6 RBs * n_sym / bundleSize
                    printf("It will occupy REs from: [%d to %d) per symbol\n",
                           12 * (params->start_rb + bundle_table[log_bundle_id] * params->bundle_size / params->n_sym),
                           12 * (params->start_rb + (bundle_table[log_bundle_id] + 1) * params->bundle_size / params->n_sym));
                }
            }

            printf("nDCI %d will use physical bundles: ", i);
            for(int j = 0; j < N_bundle_phy; j++)
            {
                if(used_bundle_map[j] == 1)
                {
                    printf("%d ", j);
                }
            }
            printf("\n\n");
            // How many RBs are used for a given DCI? (params->bundle_size / params->n_sym) * aggr_level * (6 / params->bundle_size) per OFDM symbol
            // So (6 * aggr_level / n_sym). But only every (params->bundle_size / params->n_sym) ones will be contiguous in the freq. domain.
        }
    }
}
/*VCAST_DONT_INSTRUMENT_END*/

void kernelSelectGenScramblingSeq(cuphyGenScramblingSeqLaunchCfg_t* pLaunchCfg,
                                  uint32_t                          num_DCIs)
{
    if (pLaunchCfg->kernelNodeParamsDriver.func == nullptr) {
        // kernel (only one kernel option for now) and without any specialization, so update function only once. func is set to nullptr in PDCCH channel constructor
        void* kernelFunc = reinterpret_cast<void*>(genScramblingSeqKernel);
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));}
    }

    // launch geometry (can change!)
    dim3 gridDim(num_DCIs);
    dim3 blockDim(64);

    // populate kernel parameters
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;

    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra    = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = 0;
}

void kernelSelectGenTfSignal(cuphyGenPdcchTfSgnlLaunchCfg_t* pLaunchCfg,
                             uint32_t                        num_DCIs,
                             int                             num_coresets,
                             PdcchParams*                    h_coreset_params)
{
    // kernel (only one kernel option for now)
    void* kernelFunc = reinterpret_cast<void*>(genPdcchTfSignalKernel<__half2>);
    {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));}

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
    dim3 gridDim   = dim3(max_n_sym, num_DCIs);
    dim3  blockDim = dim3(128);

    // populate kernel parameters
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;

    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra          = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = s_dmrs_seqs_size + s_gold_seqs_size;
}

} // namespace embedPdcchTx

using namespace embedPdcchTx;

// Generate PDCCH CRC output and scrambling sequence
cuphyStatus_t cuphyPdcchPipelinePrepare(void*                   h_input_w_crc_addr,
                                        cuphyTensorDescriptor_t h_input_w_crc_desc,
                                        const void*             h_input_addr,
                                        cuphyTensorDescriptor_t h_input_desc,
                                        int                     num_coresets,
                                        int                     num_dcis,
                                        PdcchParams*            params, // some derived params are updated.
                                        cuphyPdcchDciPrm_t*     h_dci_params,
                                        uint8_t*                h_dci_tm_info,
                                        cuphyEncoderRateMatchMultiDCILaunchCfg_t* pEncdRMLaunchCfg,
                                        cuphyGenScramblingSeqLaunchCfg_t*         pScrmSeqLaunchCfg,
                                        cuphyGenPdcchTfSgnlLaunchCfg_t*           pTfSignalLaunchCfg,
                                        cudaStream_t            stream)
{
    cuphyStatus_t status = CUPHY_STATUS_SUCCESS;
    if(!h_input_w_crc_addr || !h_input_addr || !pEncdRMLaunchCfg || !pScrmSeqLaunchCfg || !pTfSignalLaunchCfg)
    {
        NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "cuphyPdcchPipelinePrepare() got nullptr(s)");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    /* The encodeRateMatchMultipleDCIsKernel should skip encoding and rate matching
       for those DCIs whose coresets belong to a cell in testing mode, and instead
       copy (D2D) the PN23 DCI payload  to the appropriate location of that kernel's output tx buffer.
       Current design populates a byte array using a bit for the TM of each DCI.
       Another option would be to pass the coreset PdcchParams information, but that would require
       a search over coreset. */

    int current_tm_byte_index = 0; // offset into h_dci_tm_info byte array with a bit per DCI
    uint8_t current_tm_byte = 0;   // initial byte value; only DCIs in TM update (set) relevant bits.
    int current_tm_bit_index = 0;  // next bit (from least significant side) to be written in current_tm_byte; wraps around 8.

    for(int coreset_idx = 0; coreset_idx < num_coresets; coreset_idx++)
    {
        int dci_offset = params[coreset_idx].dciStartIdx;

        // Assume params contain valid values for the various config. parameters. Relevant check happens in PDCCH pipeline.
        if(!params[coreset_idx].interleaved)
        {
            params[coreset_idx].bundle_size = 6;
        }

        // Derive values for coreset_map, n_CCE and rb_coreset and update the params.

        //rb_coreset is the position of the rightmost bit of freq_domain_resource
        params[coreset_idx].rb_coreset = 64 - find_rightmost_bit(params[coreset_idx].freq_domain_resource);

        //n_CCE is the number of set bits in the freq_domain_resource multiplied by number of symbols
        params[coreset_idx].n_CCE = count_set_bits(params[coreset_idx].freq_domain_resource) * params[coreset_idx].n_sym;

        params[coreset_idx].coreset_map = params[coreset_idx].freq_domain_resource >> (64 - params[coreset_idx].rb_coreset);

        //printf("freq domain resource %lx, rb_coreset %d, n_CCE %d, coreset_map %lx \n",
        //       params[coreset_idx].freq_domain_resource, params[coreset_idx].rb_coreset, params[coreset_idx].n_CCE, params[coreset_idx].coreset_map);

        bool coreset_cell_in_testing_mode = (params[coreset_idx].testModel != 0); // The testModel field is expected to be identical for all coresets in the cell; no explicit check in cuPHY
        // When set, we  need to set the next num_dl_dci=1 bits of h_dci_tm_info to the appropriate testing mode value

        // Used in per-DCI payload size check below
        int max_payload_bits = (coreset_cell_in_testing_mode ? polar_encoder::N_MAX_TM_DCI_TX_BYTES : CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES)*8;

        // Go over all DCIs of this coreset
        for(int i = 0; i < params[coreset_idx].num_dl_dci; i++)
        {
            cuphyPdcchDciPrm_t& dci_params = h_dci_params[dci_offset + i];

            // Generate CRC for each DCI
            int payload_bits = dci_params.Npayload;
            int nCrcOutByte  = round_up_to_next(CUPHY_PDCCH_N_CRC_BITS + payload_bits, 32) / 8;

            uint32_t payload_crc        = 0;
            uint32_t input_offset       = (i + dci_offset) * CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES;
            uint32_t input_w_crc_offset = (i + dci_offset) * CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC;

            uint8_t* h_input_w_crc = (uint8_t*)(h_input_w_crc_addr) + input_w_crc_offset;

            // Check for max payload size, to ensure no illegal access (shared memory) in encodeRateMatchMultipleDCIsKernel in non TM mode
            // The kernel already has some checks in place in case of testing mode.
            if(payload_bits > max_payload_bits) {
                NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "Error! PDCCH coreset {} DCI {} (testing mode {}) has invalid payload size of {} bits.",
                           coreset_idx, i, coreset_cell_in_testing_mode, payload_bits);
                return CUPHY_STATUS_INVALID_ARGUMENT;
            }
            if(coreset_cell_in_testing_mode)
            {
                payload_crc = 0; // If the cell this DCI belongs to is in testing mode, skip CRC computation and set CRC payload to 0; acts as padding
                current_tm_byte |= (1 << current_tm_bit_index);
            }
            else
            {
                pdcchAddCrc((uint8_t*)h_input_addr + input_offset, payload_crc, dci_params.rntiCrc, payload_bits);
            }
            // Update current_tm_byte, current_tm_bit_index and current_tm_byte_index
            current_tm_bit_index += 1;
            if (current_tm_bit_index == 8) {
                current_tm_bit_index = 0;
                h_dci_tm_info[current_tm_byte_index] = current_tm_byte;
                current_tm_byte_index += 1;
                current_tm_byte       = 0;
            }


            //printf("DCI %d has payload_crc %x and rnti_crc %x and payload_bits %d\n", i, payload_crc, dci_params.rntiCrc, payload_bits);

            //Need to do a rev. bit order in bytes for h_input_addr buffer and output that to x_input_w_crc_addr buffer.
            //Also need to copy out and add CRC in rev. bit order.
            //TODO Temp change here. Could potentially use a brev in a device function in polar encoder.
            pdcchReverseBitInByte((uint8_t*)h_input_addr + input_offset,
                                  h_input_w_crc,
                                  nCrcOutByte,
                                  payload_bits);

            uint32_t starting_byte = payload_bits / 8;
            uint32_t leftover_bits = 8 - (payload_bits % 8);
            uint8_t  val           = 0;
            uint32_t byte_offset = 0, bit_offset = 0;
            for(int j = 0; j < CUPHY_PDCCH_N_CRC_BITS; j++)
            {
                val = (payload_crc >> (CUPHY_PDCCH_N_CRC_BITS - j - 1)) & 0x1;
                if(j < leftover_bits)
                {
                    h_input_w_crc[starting_byte] |= (val << (8 - leftover_bits + j));
                    if(j == leftover_bits - 1) byte_offset += 1;
                }
                else
                {
                    h_input_w_crc[starting_byte + byte_offset] |= (val << bit_offset);
                    bit_offset += 1;
                    if(bit_offset == 8)
                    {
                        byte_offset += 1;
                        bit_offset = 0;
                    }
                }
            }
        }
    }

    // Handle any leftover h_dci_tm_info writes
    if (current_tm_bit_index != 0)
    {
        h_dci_tm_info[current_tm_byte_index] = current_tm_byte;
    }

    // set kernel parameters
    polar_encoder::kernelSelectEncodeRateMatchMultiDCIs(pEncdRMLaunchCfg, num_dcis);
    kernelSelectGenScramblingSeq(pScrmSeqLaunchCfg, num_dcis);
    kernelSelectGenTfSignal(pTfSignalLaunchCfg, num_dcis, num_coresets, params);

    return status;
}

