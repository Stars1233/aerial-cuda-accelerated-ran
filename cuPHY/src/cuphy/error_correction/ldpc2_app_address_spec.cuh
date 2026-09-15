/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#if !defined(LDPC2_APP_ADDRESS_SPEC_CUH_INCLUDED_)
#define LDPC2_APP_ADDRESS_SPEC_CUH_INCLUDED_

#include "ldpc2_bg_desc.hpp"

namespace ldpc2
{

namespace detail
{
////////////////////////////////////////////////////////////////////////
// adjust_addr_pair()
template <typename T, int Z>
__device__ inline
void adjust_addr_pair(int&     addr0,
                      int&     addr1,
                      uint32_t wrap_thread_index,
                      word_t   tIdx_tIdx)
{
    constexpr int Zsz = sizeof(T) * Z;
    LDPC2_ASM("{\n\t\t"
              ".reg .pred tidx_ge_wrap_0, tidx_ge_wrap_1;\n\t\t"
              "setp.geu.f16x2 tidx_ge_wrap_0|tidx_ge_wrap_1, %2, %3;\n\t\t"  // compare x2 thread idx to wrap values
              "@tidx_ge_wrap_0 sub.s32 %0, %0, %4;\n\t\t"                    // if tidx_0 >= wrap_0, sub (Z * sz) from addr0
              "@tidx_ge_wrap_1 sub.s32 %1, %1, %4;\n\t\t"                    // if tidx_1 >= wrap_1, sub (Z * sz) from addr1
              "}\n"
              : "+r"(addr0), "+r"(addr1)
              : "r"(tIdx_tIdx.u32), "r"(wrap_thread_index), "r"(Zsz));
}

////////////////////////////////////////////////////////////////////////
// adjust_addr_single()
template <typename T, int Z>
__device__ inline
void adjust_addr_single(int&     addr0,
                        uint32_t wrap_thread_index,
                        word_t   tIdx_tIdx)
{
    constexpr int Zsz = sizeof(T) * Z;
    LDPC2_ASM("{\n\t\t"
              ".reg .pred tidx_ge_wrap_0, tidx_ge_wrap_1;\n\t\t"
              "setp.geu.f16x2 tidx_ge_wrap_0|tidx_ge_wrap_1, %1, %2;\n\t\t"  // compare x2 thread idx to wrap values
              "@tidx_ge_wrap_0 sub.s32 %0, %0, %3;\n\t\t"                    // if tidx_0 >= wrap_0, sub (Z * sz) from addr0
              "}\n"
              : "+r"(addr0)
              : "r"(tIdx_tIdx.u32), "r"(wrap_thread_index), "r"(Zsz));
}

template <typename T,
          int BG,
          int Z,
          int CHECK_INDEX,
          int PAIR_IDX>
struct app_address_spec_unroll_pair
{
    __device__
    static void generate(int           (&app_addr)[row_degree<BG, CHECK_INDEX>::value],
                         const word_t& tIdx_tIdx)
    {
        static constexpr int ROW_DEGREE = row_degree<BG, CHECK_INDEX>::value;

        if constexpr(PAIR_IDX > 0)
        {
            app_address_spec_unroll_pair<T, BG, Z, CHECK_INDEX, PAIR_IDX-1>::generate(app_addr, tIdx_tIdx);
        }

        constexpr uint32_t wi_pair = wrap_index_pair<BG, Z, CHECK_INDEX, PAIR_IDX>::value;
        const int          tIdx_sz = sizeof(T) * threadIdx.x;
        if constexpr ((PAIR_IDX * 2) + 1 < ROW_DEGREE)
        {
            app_addr[(PAIR_IDX * 2) + 0]  = vnode_shift_offset<BG, Z, CHECK_INDEX, (PAIR_IDX * 2) + 0>::value * sizeof(T) + tIdx_sz;
            app_addr[(PAIR_IDX * 2) + 1]  = vnode_shift_offset<BG, Z, CHECK_INDEX, (PAIR_IDX * 2) + 1>::value * sizeof(T) + tIdx_sz;
            adjust_addr_pair<T, Z>(app_addr[(PAIR_IDX * 2) + 0],
                                   app_addr[(PAIR_IDX * 2) + 1],
                                   wi_pair,
                                   tIdx_tIdx);
        }
        else
        {
            // Odd row degree
            app_addr[(PAIR_IDX * 2) + 0]  = vnode_shift_offset<BG, Z, CHECK_INDEX, (PAIR_IDX * 2) + 0>::value * sizeof(T) + tIdx_sz;
            adjust_addr_single<T, Z>(app_addr[(PAIR_IDX * 2) + 0], wi_pair, tIdx_tIdx);
        }
    }
};

////////////////////////////////////////////////////////////////////////
// app_address_gen_spec
template <typename T, int BG, int Z, int CHECK_INDEX> struct app_address_gen_spec
{
    __device__
    static void generate(int           (&app_addr)[row_degree<BG, CHECK_INDEX>::value],
                         const word_t& tIdx_tIdx)
    {
        //--------------------------------------------------------------
        // Define a type for a generator with the number of elements in the row
        static constexpr int ROW_DEGREE = row_degree<BG, CHECK_INDEX>::value;
        static constexpr int MAX_PAIR_INDEX = div_round_up_t<ROW_DEGREE, 2>::value - 1;
        typedef app_address_spec_unroll_pair<T, BG, Z, CHECK_INDEX, MAX_PAIR_INDEX> generator_t;
        //--------------------------------------------------------------
        // Generate address values
        generator_t::generate(app_addr, tIdx_tIdx);
    }
};

} // namespace detail

template <typename T, int BG, int Z_>
struct app_loc_address_gen_spec
{
    static constexpr int Z = Z_;
    //------------------------------------------------------------------
    // Base graph descriptor type used by this app address calculator
    using bg_desc_t = null_BG_desc_t<BG>;
    //------------------------------------------------------------------
    // app_loc_address_gen_spec()
    // Constructor using original LDPC_kernel_params struct
    __device__
    app_loc_address_gen_spec(const LDPC_kernel_params& params,
                             const bg_desc_t&          /*bgd*/,
                             unsigned int              t_idx) : tIdx_tIdx(h0_h0(t_idx))
    {
    }
    //------------------------------------------------------------------
    // app_loc_address_fp_desc()
    // Constructor using descriptor config struct
    __device__
    app_loc_address_gen_spec(const cuphyLDPCDecodeConfigDesc_t& config,
                             const bg_desc_t&                   /*bgd*/,
                             unsigned int                       t_idx) : tIdx_tIdx(h0_h0(t_idx))
    {
    }
    //------------------------------------------------------------------
    template <int CHECK_IDX>
    __device__
    void generate(int (&app_addr)[row_degree<BG, CHECK_IDX>::value])
    {
        detail::app_address_gen_spec<T, BG, Z_, CHECK_IDX>::generate(app_addr,
                                                                     tIdx_tIdx);
    }
    word_t tIdx_tIdx;
};

} // namespace ldpc2

#endif // !defined(LDPC2_APP_ADDRESS_SPEC_CUH_INCLUDED_)
