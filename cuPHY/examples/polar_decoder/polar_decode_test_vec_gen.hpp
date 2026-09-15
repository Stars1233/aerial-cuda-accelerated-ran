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

#ifndef POLAR_DECODE_TEST_VEC_GEN_HPP
#define POLAR_DECODE_TEST_VEC_GEN_HPP

#include <cstdint>
#include <vector>
#include <cuda_fp16.h>
#include "cuphy.h"

////////////////////////////////////////////////////////////////////////
// PolarGenParams
/**
 * @brief Configuration for the self-contained polar decoder test-vector
 *        generator. One UCI segment (1 or 2 codeblocks) is produced per -w.
 */
struct PolarGenParams
{
    uint16_t A         = 32;    //!< payload bits per segment (does NOT include CRC)
    uint32_t E         = 100;   //!< number of rate-matched bits (nTxBits)
    int      listSz    = 1;     //!< polar decoder list size
    uint16_t nCws      = 16;    //!< number of UCI segments (batch count)
    float    snrDb     = 20.0f; //!< channel SNR in dB
    uint64_t seed      = 0;     //!< base RNG seed
    bool     fullChain = true;  //!< full rate-matched RX chain (default); false (--no-ratematch) = raw N-bit mother code
};

/**
 * @brief Generated test vectors: per-segment tree types (2*N_cw layout from
 *        compCwTreeTypes) plus per-codeblock channel LLRs and source info bits
 *        used for decoder evaluation.
 */
struct PolarGenResult
{
    std::vector<cuphyPolarUciSegPrm_t> segPrms;        //!< per segment
    std::vector<std::vector<uint8_t>>  treeTypes4N;    //!< per segment, PolarCwTreeLayout::sizeBytes(N_cw) bytes
    std::vector<std::vector<__half>>   llrN;           //!< per codeblock, N_cw LLRs
    std::vector<std::vector<uint8_t>>  srcPayloadBits; //!< per codeblock, (K - CRC) info bits (bit-per-byte)
};

/**
 * @brief Encode random payloads, add AWGN to produce per-codeblock LLRs, and
 *        compute the tree-types layout so the decoder can run without a file.
 *
 * @param[in] params generator configuration.
 * @param[in] strm   CUDA stream for the generation work.
 * @return           the generated segment params, tree types, LLRs, and source bits.
 */
PolarGenResult generatePolarTestVectors(const PolarGenParams& params, cudaStream_t strm);

#endif // POLAR_DECODE_TEST_VEC_GEN_HPP
