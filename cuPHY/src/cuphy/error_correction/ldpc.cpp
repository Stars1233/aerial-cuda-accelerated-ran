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
#include <array>
#include <string>
#include <vector>

#include "ldpc.hpp"

#include <gsl-lite/gsl-lite.hpp>

#include "ldpc2_reg_index_fp_desc_dyn.hpp"
#include "ldpc2_reg_index_fp_desc_dyn_small.hpp"
#include "ldpc2_reg_index_fp_x2_desc_dyn.hpp"
#include "ldpc2_reg_index_fp_desc_dyn_row_dep.hpp"
#include "ldpc2_reg_index_fp_dp_desc_dyn_row_dep.hpp"
#include "ldpc2_shm_index_fp_desc_dyn.hpp"
#include "ldpc2_split_index_fp_x2_desc_dyn.hpp"
#include "ldpc2_index_fp_x2_allreg.hpp"
#include "ldpc2_index_fp_x2_allreg_regapp.hpp"
#include "ldpc2_split_index_bp_x2_desc_dyn.hpp"
#include "ldpc2_index_bp_x2_lowp.hpp"
#include "ldpc2_split_index_fp_dp_x2_desc_dyn.hpp"
#include "ldpc2_split_index_dp_x2_desc_dyn.hpp"
#include "ldpc2_split_index_p_x2_desc_dyn.hpp"
#include "ldpc2_reg_box_plus_fp.hpp"
#include "ldpc2_reg_box_plus_fp_dp.hpp"
#include "ldpc2_reg_box_plus_fp8_fp.hpp"
#include "ldpc2_reg_box_plus_fp8_fp_dp.hpp"
#include "ldpc2_reg_box_plus_spec.hpp"
#include "ldpc2_reg_box_plus_spec_high.hpp"
#include "ldpc2_algo103.hpp"
#include "ldpc2_algo201.hpp"
#include "ldpc2_algo202.hpp"
#include "ldpc2_algo203.hpp"
#include "ldpc2_algo204.hpp"
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

// LDPC_ALGO_REG_INDEX_FP_DESC_DYN
//    - Register storage of C2V values
//    - APP address calculation using floating point instructions
//    - Compressed C2V processing
// LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN
//    - Register storage of C2V values
//    - Decodes 2 codewords per CTA (X2)
//    - Limited to high/medium code rates due to high register storage required
//      for 2 codewords per CTA
//    - Compressed C2V processing
// LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP
//    - Register storage of C2V values
//    - APP address calculation using floating point instructions
//    - Mixed (compressed C2V and box-plus) processing, depending on
//      row degree
// LDPC_ALGO_REG_INDEX_FP_DP_DESC_DYN_ROW_DEP
//    - Register storage of C2V values
//    - APP address calculation using floating point and dot product
//      instructions
//    - Mixed (compressed C2V and box-plus) processing, depending on
//      row degree
// LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SMALL
//    - Register storage of C2V values
//    - Compressed C2V processing
//    - APP address calculation using floating point instructions
//    - Limited to small Z values (less than 32), due to warp handling
//      of hard decision outputs
// LDPC_ALGO_SHM_INDEX_FP_DESC_DYN
//    - Shared memory storage of C2V values
//    - APP address calculation using floating point instructions
// LDPC_ALGO_SPLIT_INDEX_FP_DP_X2_DESC_DYN
//    - "Split" storage of C2V values (registers and shared memory)
//    - Compressed C2V processing
//    - APP address calculation using floating point and dot product
//      instructions
//    - Decodes 2 codewords per CTA (X2)
//    - May not be usable to process low code rates for large Z
// LDPC_ALGO_REG_BOX_PLUS_FP
//    - Register storage of C2V values
//    - Box plus algorithm (instead of compressed C2V)
//    - APP address calculation using floating point instructions
// LDPC_ALGO_REG_BOX_PLUS_FP_DP
//    - Register storage of C2V values
//    - Box plus algorithm (instead of compressed C2V)
//    - APP address calculation using floating point and dot product
//      instructions
// LDPC_ALGO_REG_BOX_PLUS_FP8_FP
//    - Register storage of C2V values
//    - Box plus algorithm (instead of compressed C2V)
//    - APP address calculation using floating point instructions
//    - FP8 internal storage used for C2V messages and APP data
// LDPC_ALGO_REG_BOX_PLUS_FP8_FP_DP
//    - Register storage of C2V values
//    - Box plus algorithm (instead of compressed C2V)
//    - APP address calculation using floating point and dot product
//      instructions
//    - FP8 internal storage used for C2V messages and APP data
// LDPC_ALGO_REG_BOX_PLUS_SPEC
//    - Register storage of C2V values
//    - Box plus algorithm (instead of compressed C2V)
//    - APP address calculation using inline immediates/constants
//    - Kernels are specialized for a single Z value
// LDPC_ALGO_REG_BOX_PLUS_SPEC_HIGH
//    - Register storage of C2V values
//    - Box plus algorithm (instead of compressed C2V)
//    - APP address calculation using inline immediates/constants
//    - Kernels are specialized for a single Z value
//    - Kernels are further specialized to high to medium code rates. (Avoiding
//      lower code rates reduces register usage, in some cases avoiding register
//      spilling and thus improviing performance. However, a different kernel
//      must be used at low code rates.)
// LDPC_ALGO_SPLIT_INDEX_DP_X2_DESC_DYN
//    - "Split" storage of C2V values (registers and shared memory)
//    - Compressed C2V processing
//    - APP address calculation using dot product instructions
// LDPC_ALGO_SPLIT_INDEX_P_X2_DESC_DYN
//    - "Split" storage of C2V values (registers and shared memory)
//    - Compressed C2V processing
//    - APP address calculation using predicate wrap around formulation
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
    LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP      = 29,
    LDPC_ALGO_REG_INDEX_FP_DP_DESC_DYN_ROW_DEP   = 30,
    LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SMALL        = 33,
    LDPC_ALGO_SHM_INDEX_FP_DESC_DYN              = 34,
    LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN         = 35,
    LDPC_ALGO_SPLIT_INDEX_FP_DP_X2_DESC_DYN      = 36,
    LDPC_ALGO_REG_BOX_PLUS_FP                    = 40,
    LDPC_ALGO_REG_BOX_PLUS_FP_DP                 = 41,
    LDPC_ALGO_REG_BOX_PLUS_FP8_FP                = 42,
    LDPC_ALGO_REG_BOX_PLUS_FP8_FP_DP             = 43,
    LDPC_ALGO_REG_BOX_PLUS_SPEC                  = 44,
    LDPC_ALGO_REG_BOX_PLUS_SPEC_HIGH             = 45,
    LDPC_ALGO_SPLIT_INDEX_DP_X2_DESC_DYN         = 46,
    LDPC_ALGO_SPLIT_INDEX_P_X2_DESC_DYN          = 47,
    LDPC_ALGO_INDEX_FP_X2_ALLREG                 = 51,
    LDPC_ALGO_INDEX_FP_X2_ALLREG_REGAPP          = 52,
    LDPC_ALGO_SPLIT_INDEX_BP_X2_DESC_DYN         = 55,
    LDPC_ALGO_INDEX_BP_X2_LOWP                   = 56,
    // ------------------------------------------------------------------
    // Point kernel, BG1 / p=4 / Z=384 only. Kept apart from the 200-series
    // below because it IS a point solution rather than part of the
    // consolidated line: it exists for one commercially weighted config
    // and buys 19% there over the general kernel that also covers it.
    LDPC_ALGO_BG1_Z384_P4_X2                     = 103,
    // ------------------------------------------------------------------
    // 200-SERIES: the consolidated band decoders. Together they cover
    // every legal BG1 (p, Z) with p in 4..46 and Z in {256, 288, 320,
    // 352, 384} -- 215 configs, no gaps -- in four kernels.
    //
    // THE DIGITS CARRY NO MEANING. They are sequential, and every
    // characteristic lives in the NAME instead, because semantic IDs go
    // stale the first time a band edge moves.
    //
    // Name grammar:  bg<BG>_z<zone>_<storage class>_x<CW per CTA>
    //   z256up      the Z zone covered. It names the zone rather than
    //               claiming to be Z-generic because the CTA mapping
    //               (blockDim == Z, one CTA/SM) is what bounds it; a
    //               Z < 256 family needs a different mapping and would
    //               take its own token.
    //   storage     the C2V storage class -- the axis that actually
    //               forces specialization, and the one characteristic
    //               that does not move when a band edge does.
    //   x2          codewords per CTA, as in the 30-/50-series names.
    //
    // The p band is deliberately NOT in the name; it is derived in each
    // kernel's can_decode_config() and stated once there.
    LDPC_ALGO_BG1_Z256UP_BP_ALLREG_X2            = 201,
    LDPC_ALGO_BG1_Z256UP_CMS_SHTAIL_X2           = 202,
    LDPC_ALGO_BG1_Z256UP_BP_SHWIN_X2             = 203,
    LDPC_ALGO_BG1_Z256UP_BP_GMEXT_X2             = 204,
    // Sized to the highest ID + 1: algos_ is a std::vector densely indexed
    // by algo ID, so algos_[204] would otherwise be an out-of-bounds write
    // during registration. The gap leaves ~145 null slots (~1 KB) --
    // harmless, but it is an array, not a map.
    LDPC_NUM_ALGO = LDPC_ALGO_BG1_Z256UP_BP_GMEXT_X2 + 1
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
template <> struct algo_index_map<LDPC_ALGO_SPLIT_INDEX_FP_DP_X2_DESC_DYN>      { typedef ldpc2::split_index_fp_dp_x2_desc_dyn      algo_t; };
template <> struct algo_index_map<LDPC_ALGO_REG_BOX_PLUS_FP>                    { typedef ldpc2::reg_box_plus_fp                    algo_t; };
template <> struct algo_index_map<LDPC_ALGO_REG_BOX_PLUS_FP_DP>                 { typedef ldpc2::reg_box_plus_fp_dp                 algo_t; };
template <> struct algo_index_map<LDPC_ALGO_REG_BOX_PLUS_FP8_FP>                { typedef ldpc2::reg_box_plus_fp8_fp                algo_t; };
template <> struct algo_index_map<LDPC_ALGO_REG_BOX_PLUS_FP8_FP_DP>             { typedef ldpc2::reg_box_plus_fp8_fp_dp             algo_t; };
template <> struct algo_index_map<LDPC_ALGO_REG_BOX_PLUS_SPEC>                  { typedef ldpc2::reg_box_plus_spec                  algo_t; };
template <> struct algo_index_map<LDPC_ALGO_REG_BOX_PLUS_SPEC_HIGH>             { typedef ldpc2::reg_box_plus_spec_high             algo_t; };
template <> struct algo_index_map<LDPC_ALGO_SPLIT_INDEX_DP_X2_DESC_DYN>         { typedef ldpc2::split_index_dp_x2_desc_dyn         algo_t; };
template <> struct algo_index_map<LDPC_ALGO_SPLIT_INDEX_P_X2_DESC_DYN>          { typedef ldpc2::split_index_p_x2_desc_dyn          algo_t; };
template <> struct algo_index_map<LDPC_ALGO_INDEX_FP_X2_ALLREG>                 { typedef ldpc2::index_fp_x2_allreg                 algo_t; };
template <> struct algo_index_map<LDPC_ALGO_INDEX_FP_X2_ALLREG_REGAPP>          { typedef ldpc2::index_fp_x2_allreg_regapp          algo_t; };
template <> struct algo_index_map<LDPC_ALGO_SPLIT_INDEX_BP_X2_DESC_DYN>         { typedef ldpc2::split_index_bp_x2_desc_dyn         algo_t; };
template <> struct algo_index_map<LDPC_ALGO_INDEX_BP_X2_LOWP>                   { typedef ldpc2::index_bp_x2_lowp                   algo_t; };
template <> struct algo_index_map<LDPC_ALGO_BG1_Z384_P4_X2>                     { typedef ldpc2::bg1_z384_p4_x2                     algo_t; };
template <> struct algo_index_map<LDPC_ALGO_BG1_Z256UP_BP_ALLREG_X2>            { typedef ldpc2::bg1_z256up_bp_allreg_x2            algo_t; };
template <> struct algo_index_map<LDPC_ALGO_BG1_Z256UP_CMS_SHTAIL_X2>           { typedef ldpc2::bg1_z256up_cms_shtail_x2           algo_t; };
template <> struct algo_index_map<LDPC_ALGO_BG1_Z256UP_BP_SHWIN_X2>             { typedef ldpc2::bg1_z256up_bp_shwin_x2             algo_t; };
template <> struct algo_index_map<LDPC_ALGO_BG1_Z256UP_BP_GMEXT_X2>             { typedef ldpc2::bg1_z256up_bp_gmext_x2             algo_t; };

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

////////////////////////////////////////////////////////////////////////
// report_algo_unavailable()
// A named algorithm that is not registered for this device's compute
// capability returns CUPHY_STATUS_NOT_SUPPORTED from a null table slot, with
// nothing to distinguish it from an algorithm that ran and declined the
// configuration. Say which it was: the per-CC registration list in the
// decoder constructor is the only thing that decides this, and a caller has
// no way to see it.
void report_algo_unavailable(const char* fn, int algoIndex, uint64_t cc)
{
    const unsigned major = static_cast<unsigned>(cc >> 32);
    const unsigned minor = static_cast<unsigned>(cc & 0xFFFFFFFFULL);
    NVLOGE_FMT(NVLOG_PUSCH,
               AERIAL_CUPHY_EVENT,
               "{}: LDPC algorithm {} is not available on this device "
               "(compute capability {}.{}). The algorithm is not registered for "
               "this architecture -- see the per-CC registration in "
               "ldpc::decoder::decoder().",
               fn, algoIndex, major, minor);
}

////////////////////////////////////////////////////////////////////////
// algo_was_requested_and_refused()
// True when the caller named a specific algorithm and that algorithm
// declined the configuration.
[[nodiscard]]
bool algo_was_requested_and_refused(const cuphyLDPCDecodeConfigDesc_t& config,
                                    cuphyStatus_t                     s)
{
    return (0 != config.algo) &&
           ((CUPHY_STATUS_NOT_SUPPORTED == s) || (CUPHY_STATUS_UNSUPPORTED_CONFIG == s));
}

////////////////////////////////////////////////////////////////////////
// Legal lifting sizes, 3GPP TS 38.212 table 5.3.2-1.
const std::array<int, 51> LDPC_LIFTING_SIZES = {
      2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
     15,  16,  18,  20,  22,  24,  26,  28,  30,  32,  36,  40,  44,
     48,  52,  56,  60,  64,  72,  80,  88,  96, 104, 112, 120, 128,
    144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384
};

////////////////////////////////////////////////////////////////////////
// format_int_set()
// "4..11", "256,288,320,352,384", or "none". Contiguous runs of three or
// more are collapsed so a wide range stays readable.
[[nodiscard]]
std::string format_int_set(const std::vector<int>& v)
{
    if(v.empty())
    {
        return std::string("none");
    }
    std::string out;
    for(size_t i = 0; i < v.size();)
    {
        size_t j = i;
        while((j + 1 < v.size()) && (v[j + 1] == v[j] + 1))
        {
            ++j;
        }
        if(!out.empty())
        {
            out += ",";
        }
        out += std::to_string(v[i]);
        if(j > i + 1)
        {
            out += ".." + std::to_string(v[j]);
        }
        else if(j == i + 1)
        {
            out += "," + std::to_string(v[j]);
        }
        i = j + 1;
    }
    return out;
}

////////////////////////////////////////////////////////////////////////
// probe_accepted()
// Vary ONE field of the caller's configuration over its legal domain,
// holding everything else fixed, and collect the values the algorithm would
// have accepted.
//
// Derived from can_decode_config() itself -- the same predicate that just
// refused -- so it cannot disagree with the code and cannot go stale. That
// matters more than it sounds: every hand-maintained statement of an
// algorithm's applicability in this tree has at some point drifted from what
// the algorithm actually did.
//
// can_decode_config() is a pure host-side predicate (comparisons plus a
// shared-memory calculation), and this runs only on an error path, so ~100
// evaluations cost nothing that matters.
template <typename TSetField, typename TDomain>
[[nodiscard]]
std::vector<int> probe_accepted(ldpc::decode_algo&                 algo,
                                const ldpc::decoder&               dec,
                                const cuphyLDPCDecodeConfigDesc_t& config,
                                const TDomain&                     domain,
                                TSetField                          set_field)
{
    std::vector<int> accepted;
    for(int candidate : domain)
    {
        cuphyLDPCDecodeConfigDesc_t probe = config;   // local copy; caller untouched
        set_field(probe, candidate);
        if(algo.can_decode_config(dec, probe))
        {
            accepted.push_back(candidate);
        }
    }
    return accepted;
}

////////////////////////////////////////////////////////////////////////
// report_algo_refusal()
// A requested algorithm that refuses returns CUPHY_STATUS_NOT_SUPPORTED,
// which does not tell the caller WHICH parameter was out of range. Echo the
// configuration back, together with the algorithm's own statement of what it
// accepts when it supplies one (decode_algo::supported_config_desc()).
//
// Only reached when a named algorithm refused, so it costs nothing on the
// automatic-selection path and nothing on success.
void report_algo_refusal(const char*                        fn,
                         int                                algoIndex,
                         ldpc::decode_algo&                 algo,
                         const ldpc::decoder&               dec,
                         const cuphyLDPCDecodeConfigDesc_t& config)
{
    // NVLOG/libfmt cannot bind a reference to a packed struct field: copy
    // each value into a local first.
    const int   BG    = config.BG;
    const int   Z     = config.Z;
    const int   p     = config.num_parity_nodes;
    const int   Kb    = config.Kb;
    const int   iters = config.max_iterations;
    const char* llr   = cuphyGetDataTypeString(config.llr_type);

    // Which parity-node counts and lifting sizes WOULD this algorithm take,
    // with the rest of the caller's configuration unchanged? Derived, so it
    // stays correct for algorithms that do not exist yet.
    const int MAX_P = (1 == BG) ? 46 : 42;
    std::vector<int> p_domain;
    for(int i = 4; i <= MAX_P; ++i)
    {
        p_domain.push_back(i);
    }
    const std::string p_ok = format_int_set(
        probe_accepted(algo, dec, config, p_domain,
                       [](cuphyLDPCDecodeConfigDesc_t& c, int v)
                       { c.num_parity_nodes = static_cast<int16_t>(v); }));
    const std::string z_ok = format_int_set(
        probe_accepted(algo, dec, config, LDPC_LIFTING_SIZES,
                       [](cuphyLDPCDecodeConfigDesc_t& c, int v)
                       { c.Z = static_cast<int16_t>(v); }));

    NVLOGE_FMT(NVLOG_PUSCH,
               AERIAL_CUPHY_EVENT,
               "{}: LDPC algorithm {} does not support the requested configuration "
               "(BG={}, Z={}, parity nodes={}, Kb={}, iterations={}, LLR={}). "
               "Holding the rest of this configuration fixed, algorithm {} accepts "
               "parity nodes: {}; lifting sizes: {}.",
               fn, algoIndex, BG, Z, p, Kb, iters, llr, algoIndex, p_ok, z_ok);

    // Anything that is not a field of the configuration -- required decode
    // interface, punctured input -- cannot be probed and is stated by the
    // algorithm itself when it chooses to.
    const char* supported = algo.supported_config_desc();
    if(nullptr != supported)
    {
        NVLOGE_FMT(NVLOG_PUSCH,
                   AERIAL_CUPHY_EVENT,
                   "  algorithm {} additionally requires: {}.",
                   algoIndex, supported);
    }
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
            algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_DP_X2_DESC_DYN>   ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_FP>                 ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_FP_DP>              ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_SPEC>               ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_SPEC_HIGH>          ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_DP_X2_DESC_DYN>      ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_P_X2_DESC_DYN>       ::create(*this, algos_);
            algo_factory<LDPC_ALGO_INDEX_FP_X2_ALLREG>              ::create(*this, algos_);
            algo_factory<LDPC_ALGO_INDEX_FP_X2_ALLREG_REGAPP>       ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_BP_X2_DESC_DYN>      ::create(*this, algos_);
            algo_factory<LDPC_ALGO_INDEX_BP_X2_LOWP>                ::create(*this, algos_);
            break;
        case CC_8_0:
            // Ampere
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN>             ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN>          ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP>     ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DP_DESC_DYN_ROW_DEP>  ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SMALL>       ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SHM_INDEX_FP_DESC_DYN>             ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_DP_X2_DESC_DYN>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_DP_X2_DESC_DYN>     ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_P_X2_DESC_DYN>         ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_FP>                   ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_FP_DP>                ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_SPEC>                 ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_SPEC_HIGH>            ::create(*this, algos_);
            algo_factory<LDPC_ALGO_INDEX_FP_X2_ALLREG>                ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_BP_X2_DESC_DYN>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_INDEX_BP_X2_LOWP>                  ::create(*this, algos_);
            break;
        case CC_8_6:
            // Ampere (A102)
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN>             ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN>          ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP>     ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DP_DESC_DYN_ROW_DEP>  ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SMALL>       ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SHM_INDEX_FP_DESC_DYN>             ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_DP_X2_DESC_DYN>     ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_FP>                   ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_FP_DP>                ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_SPEC>                 ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_SPEC_HIGH>            ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_DP_X2_DESC_DYN>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_P_X2_DESC_DYN>         ::create(*this, algos_);
            algo_factory<LDPC_ALGO_INDEX_FP_X2_ALLREG>                ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_BP_X2_DESC_DYN>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_INDEX_BP_X2_LOWP>                  ::create(*this, algos_);
            break;
        case CC_8_9:
            // Ampere (AD102)
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN>             ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN>          ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP>     ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DP_DESC_DYN_ROW_DEP>  ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_INDEX_FP_DESC_DYN_SMALL>       ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SHM_INDEX_FP_DESC_DYN>             ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_DP_X2_DESC_DYN>     ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_FP>                   ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_FP_DP>                ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_SPEC>                 ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_SPEC_HIGH>            ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_DP_X2_DESC_DYN>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_P_X2_DESC_DYN>         ::create(*this, algos_);
            algo_factory<LDPC_ALGO_INDEX_FP_X2_ALLREG>                ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_BP_X2_DESC_DYN>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_INDEX_BP_X2_LOWP>                  ::create(*this, algos_);
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
            algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_DP_X2_DESC_DYN>     ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_FP>                   ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_FP_DP>                ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_FP8_FP>               ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_FP8_FP_DP>            ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_SPEC>                 ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_SPEC_HIGH>            ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_DP_X2_DESC_DYN>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_P_X2_DESC_DYN>         ::create(*this, algos_);
            algo_factory<LDPC_ALGO_INDEX_FP_X2_ALLREG>                ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_BP_X2_DESC_DYN>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_INDEX_BP_X2_LOWP>                  ::create(*this, algos_);
            // Registered, not dispatched: choose_algo_sm90() never returns
            // these. They are reachable only through an explicit algo
            // selection, which is what lets a Hopper part measure them --
            // all of the evidence behind them so far is GB203/sm120.
            algo_factory<LDPC_ALGO_BG1_Z384_P4_X2>                    ::create(*this, algos_);
            algo_factory<LDPC_ALGO_BG1_Z256UP_BP_ALLREG_X2>           ::create(*this, algos_);
            algo_factory<LDPC_ALGO_BG1_Z256UP_CMS_SHTAIL_X2>          ::create(*this, algos_);
            algo_factory<LDPC_ALGO_BG1_Z256UP_BP_SHWIN_X2>            ::create(*this, algos_);
            algo_factory<LDPC_ALGO_BG1_Z256UP_BP_GMEXT_X2>            ::create(*this, algos_);
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
            algo_factory<LDPC_ALGO_SPLIT_INDEX_FP_DP_X2_DESC_DYN>     ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_FP>                   ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_FP_DP>                ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_FP8_FP>               ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_FP8_FP_DP>            ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_SPEC>                 ::create(*this, algos_);
            algo_factory<LDPC_ALGO_REG_BOX_PLUS_SPEC_HIGH>            ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_DP_X2_DESC_DYN>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_P_X2_DESC_DYN>         ::create(*this, algos_);
            algo_factory<LDPC_ALGO_INDEX_FP_X2_ALLREG>                ::create(*this, algos_);
            algo_factory<LDPC_ALGO_INDEX_FP_X2_ALLREG_REGAPP>         ::create(*this, algos_);
            algo_factory<LDPC_ALGO_SPLIT_INDEX_BP_X2_DESC_DYN>        ::create(*this, algos_);
            algo_factory<LDPC_ALGO_INDEX_BP_X2_LOWP>                  ::create(*this, algos_);
            // Registered, not dispatched. choose_algo_sm100() (which
            // choose_algo_sm120() currently delegates to) returns none of
            // these, so nothing reaches them without an explicit algo
            // selection. Wiring them into the dispatch is a separate
            // change, with a separate class of evidence behind it.
            algo_factory<LDPC_ALGO_BG1_Z384_P4_X2>                    ::create(*this, algos_);
            algo_factory<LDPC_ALGO_BG1_Z256UP_BP_ALLREG_X2>           ::create(*this, algos_);
            algo_factory<LDPC_ALGO_BG1_Z256UP_CMS_SHTAIL_X2>          ::create(*this, algos_);
            algo_factory<LDPC_ALGO_BG1_Z256UP_BP_SHWIN_X2>            ::create(*this, algos_);
            algo_factory<LDPC_ALGO_BG1_Z256UP_BP_GMEXT_X2>            ::create(*this, algos_);
            break;
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
    switch(cc_)
    {
    default:
    case CC_7_0:  return choose_algo_sm70( config);
    case CC_7_5:  return choose_algo_sm75( config);
    case CC_8_0:  return choose_algo_sm80( config);
    case CC_8_6:  return choose_algo_sm86( config);
    case CC_8_9:  return choose_algo_sm89( config);
    case CC_9_0:  return choose_algo_sm90( config);
    case CC_10_0: return choose_algo_sm100(config);
    case CC_12_0:
    case CC_12_1:
        return choose_algo_sm120(config);
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
            bool canUseX2 = algos_[LDPC_ALGO_SPLIT_INDEX_FP_DP_X2_DESC_DYN]->can_decode_config(*this,
                                                                                               config);
            return canUseX2                                ?
                   LDPC_ALGO_SPLIT_INDEX_FP_DP_X2_DESC_DYN :
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
        return LDPC_ALGO_SPLIT_INDEX_FP_DP_X2_DESC_DYN;
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
            if (algos_[LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN]->can_decode_config(*this, config))
            {
                // Generic x2 kernel upper bound on number of parity nodes is slightly higher
                return LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN;
            }
        }
        // Use a 1CW/CTA kernel when the throughput flag is not provided
        return LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP;
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
            if (algos_[LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN]->can_decode_config(*this, config))
            {
                // Generic x2 kernel (upper bound on number of parity nodes is slightly higher)
                return LDPC_ALGO_REG_INDEX_FP_X2_DESC_DYN;
            }
        }
        // Fall back to x1 (1 codeword per CTA)
        return LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP;

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
            // GH200 / H100 (227 KB shmem) dispatch tuning. The bulk of the
            // parity range is already best served by the box-plus x2 kernel
            // (ALGO 55) below; these two guards capture the edge regions it
            // loses on H100 (measured Z=384, 2048 CW, 10 iter, FP16):
            //
            //  * BG1 p>=40: once ALGO 55 no longer fits, the box-plus x1
            //    kernel (ALGO 40, whose xhighp40 template fires at p>=40)
            //    beats both the SM90 min-sum x2 kernel (p=40) and the x1
            //    fallback (p=41-46) by 2-5%.
            //  * BG2 p<=9: at low parity the x2 kernel's 2-CW/CTA overhead is
            //    not amortized -- on H100's 132 SMs even x2's 1024 CTAs
            //    (2048 CW / 2) already oversubscribe the GPU, so the box-plus x1
            //    kernel wins. ALGO 40's low-p specialization (the _lowp template,
            //    dispatched internally for p<10) runs at 2 CTA/SM and beats
            //    ALGO 55 by 12-38% across p=4-9 (and the SM90 x1 cubin ALGO 38
            //    by ~25%, keeping this cubin-free). Crossover is at p=10, where
            //    ALGO 55 (box-plus x2) retakes the lead.
            if(1 == config.BG && config.num_parity_nodes >= 40 &&
               algos_[LDPC_ALGO_REG_BOX_PLUS_FP]->can_decode_config(*this, config))
            {
                return LDPC_ALGO_REG_BOX_PLUS_FP;
            }
            if(2 == config.BG && config.num_parity_nodes <= 9 &&
               algos_[LDPC_ALGO_REG_BOX_PLUS_FP]->can_decode_config(*this, config))
            {
                return LDPC_ALGO_REG_BOX_PLUS_FP;
            }
            // Box-plus x2 kernel (ALGO 55) — best performance for supported
            // configs. The bigreg ALGO 55 variant extends its reach to BG1
            // p=31-39, so together with the p>=40 / BG2 p<=9 guards above it
            // covers essentially the whole BG1/BG2 range at Z<=384. The SM90
            // cubin x2 kernels (ALGO 37 SM86, ALGO 39 SM90) that previously
            // sat below this branch are retired and de-registered.
            if(algos_[LDPC_ALGO_SPLIT_INDEX_BP_X2_DESC_DYN]->can_decode_config(*this, config))
            {
                return LDPC_ALGO_SPLIT_INDEX_BP_X2_DESC_DYN;
            }
            else
            {
                // Fall back to the source x1 row-dependent kernel (ALGO 29,
                // 1 codeword per CTA). With BG1 NUM_STORAGE_WORDS=10 / BG2=5
                // it routes the high-degree rows through the fast box-plus
                // path and is bit-exact with, and as fast as, the retired
                // SM90 x1 cubin (ALGO 38) it replaces -- keeping this path
                // cubin-free.
                return LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP;
            }
        }
        else
        {
            // Latency (non-throughput) path: same cubin-free source x1
            // row-dependent kernel (ALGO 29) that replaces the SM90 cubin.
            return LDPC_ALGO_REG_INDEX_FP_DESC_DYN_ROW_DEP;
        }
    }
    else if((CUPHY_R_8F_E4M3 == config.llr_type) || (CUPHY_R_8F_E5M2 == config.llr_type))
    {
        // FP is slightly faster than FP_DP on Hopper
        return LDPC_ALGO_REG_BOX_PLUS_FP8_FP;
    }
    else
    {
        // Only fp8, fp16 and fp32 supported at the moment
        return -1;
    }
}

////////////////////////////////////////////////////////////////////////
// decoder::choose_algo_sm100()
int decoder::choose_algo_sm100(const cuphyLDPCDecodeConfigDesc_t& config) const
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
        // Low-p x2 (ALGO 56) — BG1 p<=5, 2 CTAs/SM for better occupancy.
        if(algos_[LDPC_ALGO_INDEX_BP_X2_LOWP]->can_decode_config(*this, config))
        {
            return LDPC_ALGO_INDEX_BP_X2_LOWP;
        }
        // Box-plus x2 (ALGO 55) — best performance when shmem permits.
        // Internally dispatches between full box-plus core and a hybrid
        // min-sum-core variant depending on per-config shmem requirements.
        if(algos_[LDPC_ALGO_SPLIT_INDEX_BP_X2_DESC_DYN]->can_decode_config(*this, config))
        {
            return LDPC_ALGO_SPLIT_INDEX_BP_X2_DESC_DYN;
        }
        // Compressed min-sum x2 (ALGO 35) — C2V in shmem, faster than
        // allreg when shmem budget permits (up to ~p=35 on GB203).
        if(algos_[LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN]->can_decode_config(*this, config))
        {
            return LDPC_ALGO_SPLIT_INDEX_FP_X2_DESC_DYN;
        }
        // Box-plus xhighp40 variant (ALGO 40) — at p>=40, MIN_PARITY_ROWS=40
        // templating folds IS_LAST_ROW for rows 0..38, opening a wider
        // compiler reorder window than ALGO 51 can match. Measures 2-4%
        // faster than ALGO 51 highp22 in this range on GB203.
        if(config.num_parity_nodes >= 40 &&
           algos_[LDPC_ALGO_REG_BOX_PLUS_FP]->can_decode_config(*this, config))
        {
            return LDPC_ALGO_REG_BOX_PLUS_FP;
        }
        // All-register x2 (ALGO 51) — for high-p where ALGO 35 shmem
        // doesn't fit but APP-only shmem does. All C2V in registers.
        if(algos_[LDPC_ALGO_INDEX_FP_X2_ALLREG]->can_decode_config(*this, config))
        {
            return LDPC_ALGO_INDEX_FP_X2_ALLREG;
        }
        // Box-plus highp22 (ALGO 40) — defensive fallback for p in [22, 39]
        // when neither ALGO 35 nor ALGO 51 fits. Kept for tight-shmem devices.
        if(config.num_parity_nodes >= 22 &&
           algos_[LDPC_ALGO_REG_BOX_PLUS_FP]->can_decode_config(*this, config))
        {
            return LDPC_ALGO_REG_BOX_PLUS_FP;
        }
        // Register-APP x2 (ALGO 52) — BG1 p=4-46. Overflow extension
        // columns in registers. Reachable when ALGO 40 highp doesn't
        // apply (p<22) and ALGO 51 doesn't fit; also testable at any p
        // via -a 52.
        if(algos_[LDPC_ALGO_INDEX_FP_X2_ALLREG_REGAPP]->can_decode_config(*this, config))
        {
            return LDPC_ALGO_INDEX_FP_X2_ALLREG_REGAPP;
        }
        // Unconditional fallback
        return LDPC_ALGO_REG_BOX_PLUS_FP;
    }
    else if((CUPHY_R_8F_E4M3 == config.llr_type) || (CUPHY_R_8F_E5M2 == config.llr_type))
    {
        // FP_DP is slightly faster than FP on Blackwell
        return LDPC_ALGO_REG_BOX_PLUS_FP8_FP_DP;
    }
    else
    {
        // Only fp8, fp16 and fp32 supported at the moment
        return -1;
    }
}

////////////////////////////////////////////////////////////////////////
// decoder::choose_algo_sm120()
int decoder::choose_algo_sm120(const cuphyLDPCDecodeConfigDesc_t& config) const
{
    // Preserve the tuned FP16 dispatch used by develop for CC 12.x while
    // retaining the FP8 selection added to the SM100 path by this branch.
    return choose_algo_sm100(config);
}

////////////////////////////////////////////////////////////////////////
// decoder::decode()
cuphyStatus_t decoder::decode(const tensor_pair&                 tDst,
                              const_tensor_pair&                 tLLR,
                              const cuphy_optional<tensor_pair>& optSoftOutputs,
                              const cuphyLDPCDecodeConfigDesc_t& config,
                              cudaStream_t                       strm)

{
#if !CUPHY_CLMAD_BUILD_ENABLED
    if(config.flags & CUPHY_LDPC_DECODE_EARLY_TERM)
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
#endif

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
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    if((algoIndex >= 0) && (algoIndex < algos_.size()) && algos_[algoIndex].get())
    {
        s = algos_[algoIndex]->decode(*this, tOutWord, tLLR, optSoftOutputs, config, strm);
        if(algo_was_requested_and_refused(config, s))
        {
            report_algo_refusal("decode()",
                                algoIndex,
                                *algos_[algoIndex],
                                *this,
                                config);
        }
    }
    else
    {
        DEBUG_PRINTF("ldpc::decoder::decode() unexpected algorithm choice: CC = %lu, index = %i\n",
                     cc_,
                     algoIndex);
        // Only when the caller named the algorithm; an odd index from
        // automatic selection is a different problem and not the caller's.
        if(0 != config.algo)
        {
            report_algo_unavailable("decode()", algoIndex, cc_);
        }
    }
    return s;
}

////////////////////////////////////////////////////////////////////////
// decoder::decode_tb()
cuphyStatus_t decoder::decode_tb(const cuphyLDPCDecodeDesc_t&  decodeDesc,
                                 cudaStream_t                  strm)
{
#if !CUPHY_CLMAD_BUILD_ENABLED
    if(decodeDesc.config.flags & CUPHY_LDPC_DECODE_EARLY_TERM)
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
#endif

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
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    if((algoIndex >= 0) && (algoIndex < algos_.size()) && algos_[algoIndex].get())
    {
        s = algos_[algoIndex]->decode_tb(*this, decodeDesc, strm);
        if(algo_was_requested_and_refused(decodeDesc.config, s))
        {
            report_algo_refusal("decode_tb()",
                                algoIndex,
                                *algos_[algoIndex],
                                *this,
                                decodeDesc.config);
        }
    }
    else
    {
        DEBUG_PRINTF("ldpc::decoder::decode_tb() unexpected algorithm choice: CC = %lu, index = %i\n",
                     cc_,
                     algoIndex);
        // Only when the caller named the algorithm; an odd index from
        // automatic selection is a different problem and not the caller's.
        if(0 != decodeDesc.config.algo)
        {
            report_algo_unavailable("decode_tb()", algoIndex, cc_);
        }
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
    std::array<cuphyDataType_t, 4> valid_llr_types =
    {
        CUPHY_R_16F,
        CUPHY_R_32F,
        CUPHY_R_8F_E4M3,
        CUPHY_R_8F_E5M2
    };
    //------------------------------------------------------------------
    // Validate inputs
    if((valid_llr_types.end() == std::find(valid_llr_types.begin(),
                                           valid_llr_types.end(),
                                           config.llr_type))                  ||
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
    // Convert to fp16 if necessary. We currently use fp16 for the
    // normalization value for fp8 types as well because we don't
    // currently have fp8 arithmetic support in kernel functions.
    if((CUPHY_R_16F == config.llr_type)     ||
       (CUPHY_R_8F_E4M3 == config.llr_type) ||
       (CUPHY_R_8F_E5M2 == config.llr_type))
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
#if !CUPHY_CLMAD_BUILD_ENABLED
    if(launchConfig.decode_desc.config.flags & CUPHY_LDPC_DECODE_EARLY_TERM)
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
#endif

    cuphyStatus_t s         = CUPHY_STATUS_INTERNAL_ERROR;
    auto           algoIndex = launchConfig.decode_desc.config.algo;
    // Captured BEFORE automatic selection writes its choice back into the
    // descriptor below. Reading config.algo afterwards cannot tell a caller's
    // request from the library's own choice, which would report an
    // auto-selected algorithm as one the caller named.
    const bool     algoRequested = (0 != algoIndex);
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
        if(algoRequested && algo_was_requested_and_refused(launchConfig.decode_desc.config, s))
        {
            report_algo_refusal("get_launch_config()",
                                algoIndex,
                                *algos_[algoIndex],
                                *this,
                                launchConfig.decode_desc.config);
        }
    }
    else
    {
        DEBUG_PRINTF("ldpc::decoder::get_launch_config() unexpected algorithm choice: CC = %lu, index = %i\n",
                     cc_,
                     algoIndex);
        // Only when the caller named the algorithm; an odd index from
        // automatic selection is a different problem and not the caller's.
        if(algoRequested)
        {
            report_algo_unavailable("get_launch_config()", algoIndex, cc_);
        }
    }
    return s;
}

} // namespace ldpc
