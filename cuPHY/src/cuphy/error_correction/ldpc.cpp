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

//#define CUPHY_DEBUG 1

#include <cassert>

#include "ldpc.hpp"

#include <gsl-lite/gsl-lite.hpp>

#include "ldpc2_reg_index_fp_desc_dyn.hpp"
#include "ldpc2_reg_index_fp_desc_dyn_small.hpp"
#include "ldpc2_reg_index_fp_x2_desc_dyn.hpp"
#include "ldpc2_reg_index_fp_desc_dyn_row_dep.hpp"
#include "ldpc2_reg_index_fp_dp_desc_dyn_row_dep.hpp"
#include "ldpc2_shm_index_fp_desc_dyn.hpp"
#include "ldpc2_split_index_fp_x2_desc_dyn.hpp"
#include "ldpc2_reg_box_plus.hpp"
#ifdef USE_LDPC_CUBIN
#include "ldpc2_reg_index_fp_desc_dyn_sm80.hpp"
#include "ldpc2_reg_index_fp_desc_dyn_row_dep_sm80.hpp"
#include "ldpc2_reg_index_fp_desc_dyn_row_dep_sm86.hpp"
#include "ldpc2_reg_index_fp_desc_dyn_row_dep_sm90.hpp"
#include "ldpc2_split_index_fp_x2_desc_dyn_sm86.hpp"
#include "ldpc2_split_index_fp_x2_desc_dyn_sm90.hpp"
#endif
#include "cuphy.hpp"

namespace {

////////////////////////////////////////////////////////////////////////
// Compute capabilities
constexpr uint64_t CC_7_0  = ( 7ULL << 32);
constexpr uint64_t CC_7_5  = ( 7ULL << 32) + 5;
constexpr uint64_t CC_8_0  = ( 8ULL << 32);
constexpr uint64_t CC_8_6  = ( 8ULL << 32) + 6;
constexpr uint64_t CC_8_9  = ( 8ULL << 32) + 9;
constexpr uint64_t CC_9_0  = ( 9ULL << 32);
constexpr uint64_t CC_10_0 = (10ULL << 32);
constexpr uint64_t CC_12_0 = (12ULL << 32);
constexpr uint64_t CC_12_1 = (12ULL << 32) + 1;

enum LDPC_ALGO
{
    //LDPC_ALGO_SMALL_FL                      = 1,
    //LDPC_ALGO_MK_FL                         = 4,
    //LDPC_ALGO_MKA_FL                        = 5,
    //LDPC_ALGO_FL                            = 6,
    //LDPC_ALGO_SIMD_FL                       = 7,
    //LDPC_ALGO_SHMEM_FL                      = 8,
    //LDPC_ALGO_MKA_FL_FLAT                   = 9,
    //LDPC_ALGO_SHMEM_LAY                     = 10,
    //LDPC_ALGO_FAST_LAY                      = 11,
    //LDPC_ALGO_SHMEM_LAY_UNROLL              = 12,
    // Layered below here
    //LDPC_ALGO_REG_ADDRESS                   = 13,
    //LDPC_ALGO_GLOB_ADDRESS                  = 14,
    //LDPC_ALGO_REG_INDEX                     = 15,
    //LDPC_ALGO_GLOB_INDEX                    = 16,
    //LDPC_ALGO_SHARED_INDEX                  = 17,
    //LDPC_ALGO_SPLIT_INDEX                   = 18,
    //LDPC_ALGO_SPLIT_DYN                     = 19,
    //LDPC_ALGO_SHARED_DYN                    = 20,
    //LDPC_ALGO_SPLIT_CLUSTER                 = 21,
    //LDPC_ALGO_SHARED_CLUSTER                = 22,
    //LDPC_ALGO_REG_INDEX_FP                  = 23,
    //LDPC_ALGO_REG_INDEX_FP_X2               = 24,
    //LDPC_ALGO_SHARED_INDEX_FP_X2            = 25,
    LDPC_ALGO_REG_INDEX_FP_DESC_DYN              = 26,
    LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN           = 27,
    LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SM80         = 28,
    LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP      = 29,
    LDPC_ALGO_REG_INDEX_FP_DP_DESC_DYN_ROW_DEP   = 30,
    LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP_SM80 = 31,
    LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP_SM86 = 32,
    LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SMALL        = 33,
    LDPC_ALGO_SHM_INDEX_FP_DESC_DYN              = 34,
    LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN         = 35,
    LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN_SM86    = 37,
    LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP_SM90 = 38,
    LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN_SM90    = 39,
    LDPC_ALGO_REG_BOX_PLUS                       = 40,
    LDPC_NUM_ALGO = LDPC_ALGO_REG_BOX_PLUS + 1
};

////////////////////////////////////////////////////////////////////////
// algo_index_map
// Template providing a mapping between the LDPC_ALGO enum value and
// a class implementing the algorithm.
template <LDPC_ALGO TAlgo> struct algo_index_map;
template <> struct algo_index_map<LDPC_ALGO_REG_INDEX_FP_DESC_DYN>              { typedef ldpc2::reg_index_fp_desc_dyn              algo_t; };
template <> struct algo_index_map<LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN>           { typedef ldpc2::reg_index_fp_x2_desc_dyn           algo_t; };
template <> struct algo_index_map<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP>      { typedef ldpc2::reg_index_fp_desc_dyn_row_dep      algo_t; };
template <> struct algo_index_map<LDPC_ALGO_REG_INDEX_FP_DP_DESC_DYN_ROW_DEP>   { typedef ldpc2::reg_index_fp_dp_desc_dyn_row_dep   algo_t; };
template <> struct algo_index_map<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SMALL>        { typedef ldpc2::reg_index_fp_desc_dyn_small        algo_t; };
template <> struct algo_index_map<LDPC_ALGO_SHM_INDEX_FP_DESC_DYN>              { typedef ldpc2::shm_index_fp_desc_dyn              algo_t; };
template <> struct algo_index_map<LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN>         { typedef ldpc2::split_index_fp_x2_desc_dyn         algo_t; };
template <> struct algo_index_map<LDPC_ALGO_REG_BOX_PLUS>                       { typedef ldpc2::reg_box_plus                       algo_t; };
#ifdef USE_LDPC_CUBIN
template <> struct algo_index_map<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SM80>         { typedef ldpc2::reg_index_fp_desc_dyn_sm80         algo_t; };
template <> struct algo_index_map<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP_SM80> { typedef ldpc2::reg_index_fp_desc_dyn_row_dep_sm80 algo_t; };
template <> struct algo_index_map<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP_SM86> { typedef ldpc2::reg_index_fp_desc_dyn_row_dep_sm86 algo_t; };
template <> struct algo_index_map<LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN_SM86>    { typedef ldpc2::split_index_fp_x2_desc_dyn_sm86    algo_t; };
template <> struct algo_index_map<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP_SM90> { typedef ldpc2::reg_index_fp_desc_dyn_row_dep_sm90 algo_t; };
template <> struct algo_index_map<LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN_SM90>    { typedef ldpc2::split_index_fp_x2_desc_dyn_sm90    algo_t; };
#endif

////////////////////////////////////////////////////////////////////////
// algo_factory
// Factory class to create an instance of the algorithm implementation
// class and assign the value to a matching index in a vector of
// unique_ptr instances.
template <LDPC_ALGO TAlgo> struct algo_factory
{
    typedef std::unique_ptr<ldpc::decode_algo>     decode_algo_ptr_t;
    typedef typename algo_index_map<TAlgo>::algo_t algo_t;
    static void create(ldpc::decoder& dec, std::vector<decode_algo_ptr_t>& algos)
    {
        assert(algos.size() > static_cast<int>(TAlgo));
        algos[static_cast<int>(TAlgo)].reset(new algo_t(dec));
    }
};

constexpr float g_min_sum_norm_BG1_Z384[47] =
{
    0.0f,  // 0
    0.0f,  // 1
    0.0f,  // 2
    0.0f,  // 3
    0.79f, // 4
    0.77f, // 5
    0.75f, // 6
    0.73f, // 7
    0.75f, // 8
    0.70f, // 9
    0.67f, // 10
    0.68f, // 11
    0.67f, // 12
    0.67f, // 13
    0.68f, // 14
    0.66f, // 15
    0.65f, // 16
    0.66f, // 17
    0.64f, // 18
    0.65f, // 19
    0.65f, // 20
    0.65f, // 21
    0.65f, // 22
    0.66f, // 23
    0.66f, // 24
    0.66f, // 25
    0.66f, // 26
    0.66f, // 27
    0.66f, // 28
    0.67f, // 29
    0.66f, // 30
    0.65f, // 31
    0.64f, // 32
    0.63f, // 33
    0.63f, // 34
    0.63f, // 35
    0.63f, // 36
    0.63f, // 37
    0.62f, // 38
    0.63f, // 39
    0.63f, // 40
    0.64f, // 41
    0.63f, // 42
    0.63f, // 43
    0.63f, // 44
    0.62f, // 45
    0.63f  // 46
};

constexpr float g_min_sum_norm_BG2_Z384[43] =
{
    0.0f,  // 0
    0.0f,  // 1
    0.0f,  // 2
    0.0f,  // 3
    0.86f, // 4
    0.84f, // 5
    0.80f, // 6
    0.77f, // 7
    0.75f, // 8
    0.75f, // 9
    0.74f, // 10
    0.74f, // 11
    0.74f, // 12
    0.73f, // 13
    0.73f, // 14
    0.73f, // 15 *
    0.73f, // 16
    0.72f, // 17
    0.70f, // 18
    0.71f, // 19 *
    0.71f, // 20
    0.71f, // 21 *
    0.71f, // 22
    0.70f, // 23 *
    0.69f, // 24
    0.70f, // 25
    0.70f, // 26 *
    0.70f, // 27 *
    0.70f, // 28 *
    0.70f, // 29 *
    0.70f, // 30 
    0.70f, // 31 *
    0.70f, // 32
    0.68f, // 33
    0.67f, // 34
    0.67f, // 35
    0.68f, // 36 *
    0.69f, // 37 *
    0.69f, // 38
    0.69f, // 39 *
    0.69f, // 40
    0.69f, // 41 *
    0.69f  // 42
};

[[nodiscard]]
bool flag_choose_throughput(uint32_t flags)
{
    return (0 != (CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT & flags));
}

} // namespace (anonymous)

////////////////////////////////////////////////////////////////////////
// ldpc
namespace ldpc
{

////////////////////////////////////////////////////////////////////////
// decoder::decoder()
decoder::decoder(const cuphy_i::context& ctx) :
    deviceIndex_(ctx.index()),
    cc_(ctx.compute_cap()),
    sharedMemPerBlockOptin_(ctx.max_shmem_per_block_optin()),
    multiProcessorCount_(ctx.sm_count())
{
    //------------------------------------------------------------------
    // Set up algorithm implementation pointers based on the compute
    // capability
    algos_.resize(LDPC_NUM_ALGO);
    DEBUG_PRINTF("ldpc::decoder::decoder() LDPC_NUM_ALGO = %u\n", LDPC_NUM_ALGO);
    try
    {
        switch(cc_)
        {
        default:
        case CC_7_0: // Volta
        case CC_7_5: // Turing
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN>           ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP>   ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DP_DESC_DYN_ROW_DEP>::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SMALL>     ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SHM_INDEX_FP_DESC_DYN>           ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN>      ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS>                    ::create(*this, algos_);
            break;
#ifdef USE_LDPC_CUBIN
        case CC_8_0:
            // Ampere
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN>             ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN>          ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SM80>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP>     ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DP_DESC_DYN_ROW_DEP>  ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP_SM80>::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SMALL>       ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SHM_INDEX_FP_DESC_DYN>             ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS>                      ::create(*this, algos_);
            break;
        case CC_8_6:
            // Ampere (A102)
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN>             ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN>          ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SM80>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP>     ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DP_DESC_DYN_ROW_DEP>  ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP_SM80>::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP_SM86>::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SMALL>       ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SHM_INDEX_FP_DESC_DYN>             ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN_SM86>   ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS>                      ::create(*this, algos_);
            break;
        case CC_8_9:
            // Ampere (AD102)
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN>             ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN>          ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SM80>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP>     ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DP_DESC_DYN_ROW_DEP>  ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP_SM80>::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP_SM86>::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SMALL>       ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SHM_INDEX_FP_DESC_DYN>             ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN_SM86>   ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS>                      ::create(*this, algos_);
            break;
        case CC_9_0:
            // Hopper (H100)
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN>             ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN>          ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP>     ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DP_DESC_DYN_ROW_DEP>  ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SMALL>       ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SHM_INDEX_FP_DESC_DYN>             ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN_SM86>   ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP_SM90>::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN_SM90>   ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS>                      ::create(*this, algos_);
            break;
        case CC_10_0:
        case CC_12_0:
        case CC_12_1:
            // Blackwell (B100)
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN>             ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN>          ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP>     ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DP_DESC_DYN_ROW_DEP>  ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SMALL>       ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SHM_INDEX_FP_DESC_DYN>             ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN>        ::create(*this, algos_);
            // ISA changes require modifications to these kernels
            //algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN_SM86>   ::create(*this, algos_);
            //algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP_SM90>::create(*this, algos_);
            //algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN_SM90>   ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS>                      ::create(*this, algos_);
            break;
#endif
        }
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Error creating algorithm instance for CC {}", cc_);
        throw;
    }
}

////////////////////////////////////////////////////////////////////////
// decoder::choose_algo()
int decoder::choose_algo(const cuphyLDPCDecodeConfigDesc_t& config) const
{
    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Current "fastest" path is FP16.
    // Kernels that decode two codewords at a time (in a single CTA)
    // in general take slightly longer than those that do one
    // codeword at a time, but they do twice as much work.
    //
    // For now, we assume "whole" ownership of the GPU by this LDPC
    // kernel, and thus we will choose 1 codeword per CTA when the
    // number of codewords is less than the number of SMs, and 2
    // codewords per CTA when the number is greater than the number
    // of SMs. In the future, more elaborate criteria may be used.
        
    // TODO: Maybe add a 'hint' flag to inform whether the chosen
    // algorithm should favor say, latency vs. throughput. When the
    // number of codewords is less than the number of SMs, use that
    // hint to choose between regular and x2 kernels.
    switch(cc_)
    {
    default:
    case CC_7_0:  return choose_algo_sm70( config);
#ifdef USE_LDPC_CUBIN
    case CC_7_5:  return choose_algo_sm75( config);
    case CC_8_0:  return choose_algo_sm80( config);
    case CC_8_6:  return choose_algo_sm86( config);
    case CC_8_9:  return choose_algo_sm89( config);
    case CC_9_0:  return choose_algo_sm90( config);
    case CC_10_0:
    case CC_12_0:
    case CC_12_1:
    return choose_algo_sm100(config);
    #endif
    }
}

////////////////////////////////////////////////////////////////////////
// decoder::choose_algo_sm70()
int decoder::choose_algo_sm70(const cuphyLDPCDecodeConfigDesc_t& config) const
{
    //------------------------------------------------------------------
    // Small Z kernel
    if(config.Z <= ldpc2::reg_index_fp_desc_dyn_small::MAX_LIFTING_SIZE)
    {
        return LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SMALL;
    }
    //------------------------------------------------------------------
    if(CUPHY_R_32F == config.llr_type)
    {
        // Convert FP32 to FP16 on load and use the dynamic descriptor
        // algorithm. Other implementations could be modified to do
        // conversion, but we don't expect this to be common.
        return LDPC_ALGO_REG_INDEX_FP_DESC_DYN;
    }
    else if(CUPHY_R_16F == config.llr_type)
    {
        if(flag_choose_throughput(config.flags))
        {
            bool canUseX2 = algos_[LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN]->can_decode_config(*this,
                                                                                          config);
            return canUseX2                           ?
                   LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN :
                   LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP; // LDPC_ALGO_REG_INDEX_FP_DESC_DYN;
        }
        else
        {
            return LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP; // LDPC_ALGO_REG_INDEX_FP_DESC_DYN;
        }
    }
    else
    {
        // Only fp16 and fp32 supported at the moment
        return -1;
    }
}

////////////////////////////////////////////////////////////////////////
// decoder::choose_algo_sm75()
int decoder::choose_algo_sm75(const cuphyLDPCDecodeConfigDesc_t& config) const
{
    //------------------------------------------------------------------
    // Small Z kernel
    if(config.Z <= ldpc2::reg_index_fp_desc_dyn_small::MAX_LIFTING_SIZE)
    {
        return LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SMALL;
    }
    //------------------------------------------------------------------
    if(CUPHY_R_32F == config.llr_type)
    {
        // Convert FP32 to FP16 on load and use the dynamic descriptor
        // algorithm. Other implementations could be modified to do
        // conversion, but we don't expect this to be common.
        return LDPC_ALGO_REG_INDEX_FP_DESC_DYN;
    }
    else if(CUPHY_R_16F == config.llr_type)
    {
        if(flag_choose_throughput(config.flags))
        {
            bool canUseX2 = algos_[LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN]->can_decode_config(*this,
                                                                                          config);
            return canUseX2                           ?
                   LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN :
                   LDPC_ALGO_REG_INDEX_FP_DESC_DYN;
        }
        else
        {
            return LDPC_ALGO_REG_INDEX_FP_DESC_DYN;
        }
    }
    else
    {
        // Only fp16 and fp32 supported at the moment
        return -1;
    }
}

////////////////////////////////////////////////////////////////////////
// decoder::choose_algo_sm80()
int decoder::choose_algo_sm80(const cuphyLDPCDecodeConfigDesc_t& config)
{
    //------------------------------------------------------------------
    // Small Z kernel
    if(config.Z <= ldpc2::reg_index_fp_desc_dyn_small::MAX_LIFTING_SIZE)
    {
        return LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SMALL;
    }
    //------------------------------------------------------------------
    if(CUPHY_R_32F == config.llr_type)
    {
        // Convert FP32 to FP16 on load and use the dynamic descriptor
        // algorithm. Other implementations could be modified to do
        // conversion, but we don't expect this to be common.
        return LDPC_ALGO_REG_INDEX_FP_DESC_DYN;
    }
    else if(CUPHY_R_16F == config.llr_type)
    {
        // Choose throughput kernel (2CW/CTA), supported for all code
        // rates, if the CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT flag is set.
        return flag_choose_throughput(config.flags) ?
               LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN :
               LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP_SM80;
    }
    else
    {
        // Only fp16 and fp32 supported at the moment
        return -1;
    }
}

////////////////////////////////////////////////////////////////////////
// decoder::choose_algo_sm86()
int decoder::choose_algo_sm86(const cuphyLDPCDecodeConfigDesc_t& config) const
{
    //------------------------------------------------------------------
    // Small Z kernel
    if(config.Z <= ldpc2::reg_index_fp_desc_dyn_small::MAX_LIFTING_SIZE)
    {
        return LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SMALL;
    }
    //------------------------------------------------------------------
    if(CUPHY_R_32F == config.llr_type)
    {
        // Convert FP32 to FP16 on load and use the dynamic descriptor
        // algorithm. Other implementations could be modified to do
        // conversion, but we don't expect this to be common.
        return LDPC_ALGO_REG_INDEX_FP_DESC_DYN;
    }
    else if(CUPHY_R_16F == config.llr_type)
    {
        if(flag_choose_throughput(config.flags))
        {
            // SM86-specific x2 kernel upper bound on number of parity nodes is relatively low
            if(algos_[LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN_SM86]->can_decode_config(*this, config))
            {
                return LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN_SM86;
            } else if (algos_[LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN]->can_decode_config(*this, config))
            {
                // Generic x2 kernel upper bound on number of parity nodes is slightly higher
                return LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN;
            }
            else
            {
                // Fall back to x1 (1 codeword per CTA)
                return LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP_SM86;
            }
        }
        else
        {
            return LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP_SM86;
        }
    }
    else
    {
        // Only fp16 and fp32 supported at the moment
        return -1;
    }
}

////////////////////////////////////////////////////////////////////////
// decoder::choose_algo_sm89()
int decoder::choose_algo_sm89(const cuphyLDPCDecodeConfigDesc_t& config) const
{
    //------------------------------------------------------------------
    // Small Z kernel
    if(config.Z <= ldpc2::reg_index_fp_desc_dyn_small::MAX_LIFTING_SIZE)
    {
        return LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SMALL;
    }
    //------------------------------------------------------------------
    if(CUPHY_R_32F == config.llr_type)
    {
        // Convert FP32 to FP16 on load and use the dynamic descriptor
        // algorithm. Other implementations could be modified to do
        // conversion, but we don't expect this to be common.
        return LDPC_ALGO_REG_INDEX_FP_DESC_DYN;
    }
    else if(CUPHY_R_16F == config.llr_type)
    {
        if(flag_choose_throughput(config.flags))
        {
            // SM86-specific x2 kernel upper bound on number of parity nodes is relatively low
            if(algos_[LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN_SM86]->can_decode_config(*this, config))
            {
                return LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN_SM86;
            } else if (algos_[LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN]->can_decode_config(*this, config))
            {
                // Generic x2 kernel upper bound on number of parity nodes is slightly higher
                return LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN;
            }
            else
            {
                // Fall back to x1 (1 codeword per CTA)
                return LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP_SM86;
            }
        }
        else
        {
            return LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP_SM86;
        }
    }
    else
    {
        // Only fp16 and fp32 supported at the moment
        return -1;
    }
}

////////////////////////////////////////////////////////////////////////
// decoder::choose_algo_sm90()
int decoder::choose_algo_sm90(const cuphyLDPCDecodeConfigDesc_t& config) const
{
    //------------------------------------------------------------------
    // Small Z kernel
    if(config.Z <= ldpc2::reg_index_fp_desc_dyn_small::MAX_LIFTING_SIZE)
    {
        return LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SMALL;
    }
    //------------------------------------------------------------------
    if(CUPHY_R_32F == config.llr_type)
    {
        // Convert FP32 to FP16 on load and use the dynamic descriptor
        // algorithm. Other implementations could be modified to do
        // conversion, but we don't expect this to be common.
        return LDPC_ALGO_REG_INDEX_FP_DESC_DYN;
    }
    else if(CUPHY_R_16F == config.llr_type)
    {
        // Choose throughput kernel (2CW/CTA) for supported code rates
        // if the CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT flag is set.
        if(flag_choose_throughput(config.flags))
        {
            // SM86 x2 kernel upper bound on number of parity nodes
            // is relatively low, but it provides the best performance on SM90.
            if(algos_[LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN_SM86]->can_decode_config(*this, config))
            {
                return LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN_SM86;
            } else if (algos_[LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN_SM90]->can_decode_config(*this, config))
            {
                // SM90 x2 kernel uses lots of shared memory and a wider
                // range of parity nodes (up to 40 currently for BG1
                return LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN_SM90;
            }
            else
            {
                // Fall back to x1 (1 codeword per CTA). (Another alternative
                // would be LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN, but the x1
                // kernel uses less shared memory and seems slightly faster.)
                return LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP_SM90;
            }
        }
        else
        {
            return LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP_SM90;
        }
    }
    else
    {
        // Only fp16 and fp32 supported at the moment
        return -1;
    }
}

////////////////////////////////////////////////////////////////////////
// decoder::choose_algo_sm100()
int decoder::choose_algo_sm100(const cuphyLDPCDecodeConfigDesc_t& config)
{
    //------------------------------------------------------------------
    // Small Z kernel
    if(config.Z <= ldpc2::reg_index_fp_desc_dyn_small::MAX_LIFTING_SIZE)
    {
        return LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SMALL;
    }
    //------------------------------------------------------------------
    if(CUPHY_R_32F == config.llr_type)
    {
        // Convert FP32 to FP16 on load and use the dynamic descriptor
        // algorithm. Other implementations could be modified to do
        // conversion, but we don't expect this to be common.
        return LDPC_ALGO_REG_INDEX_FP_DESC_DYN;
    }
    else if(CUPHY_R_16F == config.llr_type)
    {
        // Ignore the throughput flag for now until the available 2CW/CTA
        // kernels can be benchmarked.
        // if the CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT flag is set.
        //if(flag_choose_throughput(config.flags))
        //{
        //}
        //else
        //{
            return LDPC_ALGO_REG_BOX_PLUS;
        //}

    }
    else
    {
        // Only fp16 and fp32 supported at the moment
        return -1;
    }
}

////////////////////////////////////////////////////////////////////////
// decoder::decode()
cuphyStatus_t decoder::decode(const tensor_pair&                 tDst,
                              const_tensor_pair&                 tLLR,
                              const cuphy_optional<tensor_pair>& optSoftOutputs,
                              const cuphyLDPCDecodeConfigDesc_t& config,
                              cudaStream_t                       strm)

{
#if CUPHY_DEBUG
    const int NUM_CW = tLLR.first.get().layout().dimensions[1];
#endif // CUPHY_DEBUG
    //------------------------------------------------------------------
    DEBUG_PRINTF("NCW = %i, BG = %i, N = %i, K = %i, Kb = %i, mb = %i, Z = %i, M = %i, R_trans = %.2f\n",
                 NUM_CW, 
                 config.BG,
                 (config.Kb + config.num_parity_nodes) * config.Z,
                 config.Kb * config.Z,
                 config.Kb,
                 config.num_parity_nodes,
                 config.Z,
                 config.num_parity_nodes * config.Z,
                 static_cast<float>(config.Kb) / (config.Kb + config.num_parity_nodes - 2));
    //------------------------------------------------------------------
    const tensor_desc& tLLRDesc = tLLR.first.get();
    const tensor_desc& tDstDesc = tDst.first.get();
    //------------------------------------------------------------------
    // Validate inputs
    // We currently only support a 2-D tensor for input (i.e. an array
    // of inputs). The output results buffer is currently linear (1-D),
    // and thus only makes sense in that context.
    if(tLLRDesc.layout().rank() > 2)
    {
        return CUPHY_STATUS_UNSUPPORTED_RANK;
    }
    if((tDstDesc.type() != CUPHY_BIT) || (tDstDesc.layout().rank() > 2))
    {
        return CUPHY_STATUS_UNSUPPORTED_TYPE;
    }
    // Create a tensor ref that describes the output layout in 32-bit words.
    tensor_layout_any wordLayout = word_layout_from_bit_layout(tDstDesc.layout());
    LDPC_output_t     tOutWord(tDst.second,                                              // address
                               LDPC_output_t::layout_t(wordLayout.dimensions.begin(),    // layout
                                                       wordLayout.strides.begin() + 1)); // skip unit stride
    //------------------------------------------------------------------
    // If the user doesn't specify an algorithm, choose one
    int algoIndex = config.algo;
    if(0 == algoIndex)
    {
        algoIndex = choose_algo(config);
        DEBUG_PRINTF("ldpc::decoder::decode() algorithm choice: %i\n", algoIndex);
    }

    //------------------------------------------------------------------
    // Forward to the appropriate algorithm handler
    cuphyStatus_t s = CUPHY_STATUS_INTERNAL_ERROR;
    if((algoIndex >= 0) && (algoIndex < algos_.size()) && algos_[algoIndex].get())
    {
        s = algos_[algoIndex]->decode(*this, tOutWord, tLLR, optSoftOutputs, config, strm);
    }
    else
    {
        DEBUG_PRINTF("ldpc::decoder::decode() unexpected algorithm choice: CC = %lu, index = %i\n",
                     cc_,
                     algoIndex);
    }
    return s;
}

////////////////////////////////////////////////////////////////////////
// decoder::decode_tb()
cuphyStatus_t decoder::decode_tb(const cuphyLDPCDecodeDesc_t&  decodeDesc,
                                 cudaStream_t                  strm)
{
    //------------------------------------------------------------------
    DEBUG_PRINTF("NUM_TBS = %i, BG = %i, Z = %i, mb = %i, Kb = %i, max_iterations = %i\n",
                 decodeDesc.num_tbs,
                 decodeDesc.config.BG,
                 decodeDesc.config.Z,
                 decodeDesc.config.num_parity_nodes,
                 decodeDesc.config.Kb,
                 decodeDesc.config.max_iterations);
    //------------------------------------------------------------------
    // If the user doesn't specify an algorithm, choose one
    int algoIndex = decodeDesc.config.algo;

    if(0 == algoIndex)
    {
        algoIndex = choose_algo(decodeDesc.config);
    }
    //------------------------------------------------------------------
    // Forward to the appropriate algorithm handler
    cuphyStatus_t s = CUPHY_STATUS_INTERNAL_ERROR;
    if((algoIndex >= 0) && (algoIndex < algos_.size()) && algos_[algoIndex].get())
    {
        s = algos_[algoIndex]->decode_tb(*this, decodeDesc, strm);
    }
    else
    {
        DEBUG_PRINTF("ldpc::decoder::decode_tb() unexpected algorithm choice: CC = %lu, index = %i\n",
                     cc_,
                     algoIndex);
    }
    return s;
}

////////////////////////////////////////////////////////////////////////
// decoder::workspace_size()
std::pair<bool, size_t> decoder::workspace_size(const cuphyLDPCDecodeConfigDesc_t& config,
                                                int                                numCodeWords) const
{
    //------------------------------------------------------------------
    // If the user doesn't specify an algorithm, choose one
    int algoIndex = config.algo;
    if(0 == algoIndex)
    {
        algoIndex = choose_algo(config);
    }
    //------------------------------------------------------------------
    if(algoIndex < 0)
    {
        // A -1 value for the algorithm indicates that the caller would like
        // a nominal "maximum" size for all algorithms.

        // Only fp16 and fp32 supported at the moment...
        if((config.llr_type != CUPHY_R_32F) && (config.llr_type != CUPHY_R_16F))
            return {false, 0};

        // Return a canonical "maximum" size.
        // At the moment, no kernels require a workspace (now that
        // fp32 inputs are being converted fo fp16).
        return {true, 0};
    }
    //------------------------------------------------------------------
    // Forward to the appropriate algorithm handler
    if(algoIndex < algos_.size() && algos_[algoIndex].get())
    {
        return algos_[algoIndex]->get_workspace_size(*this, config, numCodeWords);
    }
    return {false, 0};
}

////////////////////////////////////////////////////////////////////////
// decoder::set_normalization()
cuphyStatus_t decoder::set_normalization(cuphyLDPCDecodeConfigDesc_t& config)
{
    //------------------------------------------------------------------
    // Validate inputs
    if(((config.llr_type != CUPHY_R_16F) && (config.llr_type != CUPHY_R_32F)) ||
       (config.num_parity_nodes < 4)                                          ||
       ((1 == config.BG) && (config.num_parity_nodes > 46))                   ||
       ((2 == config.BG) && (config.num_parity_nodes > 42)))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    // Fetch from the BG1 or BG2 table
    const float NORM = (1 == config.BG)                                 ?
                       g_min_sum_norm_BG1_Z384[config.num_parity_nodes] :
                       g_min_sum_norm_BG2_Z384[config.num_parity_nodes] ;
    //------------------------------------------------------------------
    // Convert to fp16 if necessary
    if(CUPHY_R_16F == config.llr_type)
    {
        config.norm.f16x2 = __floats2half2_rn(NORM, NORM);
    }
    else
    {
        config.norm.f32 = NORM;
    }
    
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// decoder::get_launch_config()
cuphyStatus_t decoder::get_launch_config(cuphyLDPCDecodeLaunchConfig_t& launchConfig) const
{
    cuphyStatus_t s         = CUPHY_STATUS_INTERNAL_ERROR;
    auto           algoIndex = launchConfig.decode_desc.config.algo;
    //------------------------------------------------------------------
    // If the caller doesn't specify an algorithm, choose one
    if(0 == algoIndex)
    {
        const auto chosenAlgoIndex = choose_algo(launchConfig.decode_desc.config);
        if(chosenAlgoIndex < 0)
        {
            return CUPHY_STATUS_UNSUPPORTED_CONFIG;
        }
        gsl_Expects(chosenAlgoIndex <= std::numeric_limits<int16_t>::max());
        algoIndex = static_cast<int16_t>(chosenAlgoIndex);
        launchConfig.decode_desc.config.algo = algoIndex;
    }
    //------------------------------------------------------------------
    // Forward to the appropriate algorithm handler
    if((algoIndex >= 0) && (algoIndex < algos_.size()) && algos_[algoIndex].get())
    {
        s = algos_[algoIndex]->get_launch_config(*this, launchConfig);
    }
    else
    {
        DEBUG_PRINTF("ldpc::decoder::get_launch_config() unexpected algorithm choice: CC = %lu, index = %i\n",
                     cc_,
                     algoIndex);
    }
    return s;
}

} // namespace ldpc
