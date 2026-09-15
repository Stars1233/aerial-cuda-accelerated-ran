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

#if !defined(LDPC2_C2V_FP8_CUH_INCLUDED_)
#define LDPC2_C2V_FP8_CUH_INCLUDED_

#include "cuphy.h"
#include "ldpc2_c2v.cuh"

namespace ldpc2
{

// See: FP8 Formats for Deep Learning
// https://arxiv.org/pdf/2209.05433
// The clamp_signed() function in ldpc2.cuh uses an efficient approach
// to clamping a pair of fp16 signed floating point values, with the
// result in the range of +/- the clamp value.
template <typename T> __device__ __half2 finite_clamp_value_as_half2();

template <>
inline __device__
__half2 finite_clamp_value_as_half2<__nv_fp8_e4m3>()
{
    return __float2half2_rn(CUPHY_FP8_E4M3_MAX_FINITE);
}

template <>
inline __device__
__half2 finite_clamp_value_as_half2<__nv_fp8_e5m2>()
{
    return __float2half2_rn(CUPHY_FP8_E5M2_MAX_FINITE);
}

template <int NUM_WORDS_> struct C2V_storage_t<__nv_fp8_e4m3, NUM_WORDS_>
{
    typedef storage_load_store_t<NUM_WORDS_>       load_store_t;
    typedef typename load_store_t::load_store_type load_store_type;
    static constexpr int NUM_WORDS = NUM_WORDS_;
    union value
    {
        word_t          w[NUM_WORDS_];
        load_store_type load_store;
    };
    C2V_storage_t() = default;
    //------------------------------------------------------------------
    // init()
    __device__
    void init()
    {
        #pragma unroll
        for(int i = 0; i < NUM_WORDS; ++i)
        {
            v.w[i].u32 = 0;
        }
    }
    //------------------------------------------------------------------
    // Data
    value v;
};

template <int NUM_WORDS_> struct C2V_storage_t<__nv_fp8_e5m2, NUM_WORDS_>
{
    typedef storage_load_store_t<NUM_WORDS_>       load_store_t;
    typedef typename load_store_t::load_store_type load_store_type;
    static constexpr int NUM_WORDS = NUM_WORDS_;
    union value
    {
        word_t          w[NUM_WORDS_];
        load_store_type load_store;
    };
    C2V_storage_t() = default;
    //------------------------------------------------------------------
    // init()
    __device__
    void init()
    {
        #pragma unroll
        for(int i = 0; i < NUM_WORDS; ++i)
        {
            v.w[i].u32 = 0;
        }
    }
    //------------------------------------------------------------------
    // Data
    value v;
};

template <class TBoxPlusOp, class TLLR>
class box_plus_row_proc<TBoxPlusOp, TLLR, std::enable_if_t<is_fp8<TLLR>::value>>
{
public:
    using fp8_t      = TLLR;
    // fp8_word_t:
    // __nv_fp8_e5m2 --> __nv_fp8x4_e5m2
    // __nv_fp8_e4m3 --> __nv_fp8x4_e4m3
    using fp8_word_t = typename ldpc_traits<fp8_t>::word_union_t;
    //------------------------------------------------------------------
    // process_row()
    template <int ROW_DEGREE,
              int UPDATE_ROW_DEGREE,
              class TStorage>
    __device__
    static void process_row(word_t          (&app_fp8)[row_num_words<TLLR, ROW_DEGREE>::value],
                            TStorage&       row_storage,
                            const __half2&  norm)
    {
        const int NUM_FP16_WORDS = row_num_words<__half, ROW_DEGREE>::value;
        const int NUM_UPDATE_FP16_WORDS = div_round_up_t<UPDATE_ROW_DEGREE, 2>::value;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Convert fp8 app values to fp16
        word_t app_f16[NUM_FP16_WORDS];
        fp8x4_to_half2_array<fp8_t, ROW_DEGREE>(app_f16, app_fp8);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Convert C2V values from the previous iteration fo fp16
        word_t c2v[NUM_UPDATE_FP16_WORDS];
        fp8x4_to_half2_array<fp8_t, UPDATE_ROW_DEGREE>(c2v, row_storage.v.w);

        //print_word_array<uint32_t>(0, "FP8", app_fp8);
        //print_word_array<__half>(0, "APP", app_f16);
        //print_word_array<__half>(0, "C2V", c2v);

        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Update APP and C2V values in f16 precision
        app_sub_prev_iter<UPDATE_ROW_DEGREE>(app_f16, c2v);
        app_update<ROW_DEGREE, UPDATE_ROW_DEGREE>(app_f16, c2v, norm);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Convert C2V values back to fp8 for the next iteration
        half2_to_fp8x4_array<fp8_t, UPDATE_ROW_DEGREE>(row_storage.v.w, c2v);
        half2_to_fp8x4_array<fp8_t, ROW_DEGREE>(app_fp8, app_f16);

        //print_word_array<fp8_t>(0, "CVT", app_fp8, "-----\n");
    }
private:
    //------------------------------------------------------------------
    // app_sub_prev_iter()
    template <int UPDATE_ROW_DEGREE,
              int NUM_APP_WORDS,
              int NUM_C2V_WORDS>
    __device__
    static void app_sub_prev_iter(word_t       (&app)[NUM_APP_WORDS],
                                  const word_t (&c2v)[NUM_C2V_WORDS])
    {
        #pragma unroll
        for(int i = 0; i < div_round_up_t<UPDATE_ROW_DEGREE, 2>::value; ++i)
        {
            // Relying on extension nodes in the high word being
            // set to zero in the storage structure.
            app[i].f16x2 = __hsub2(app[i].f16x2, c2v[i].f16x2);
        }

        //print_word_array<__half>(0, "SUB", app);
    }
    //------------------------------------------------------------------
    // app_update()
    template <int ROW_DEGREE,
              int UPDATE_ROW_DEGREE,
              int NUM_APP_WORDS,
              int NUM_C2V_WORDS>
    __device__
    static void app_update(word_t         (&app)[NUM_APP_WORDS],
                           word_t         (&c2v)[NUM_C2V_WORDS],
                           const __half2& norm)
    {
        word_t bp_update_seq[div_round_up_t<UPDATE_ROW_DEGREE, 2>::value];

        typedef box_plus_seq_gen<__half, TBoxPlusOp, ROW_DEGREE, UPDATE_ROW_DEGREE> box_plus_seq_gen_t;

        box_plus_seq_gen_t::generate(bp_update_seq, app);

        //print_word_array<__half>(0, "SEQ", bp_update_seq);

        #pragma unroll
        for(int i = 0; i < div_round_up_t<UPDATE_ROW_DEGREE, 2>::value; ++i)
        {
            // Handle possible saturation of updated APP values by clamping
            // sum(app + c2v) to the maximum value that can be represented by
            // the fp8 type. Then, subtract the original app value from the
            // clamped sum to determine the ACTUAL contribution of this check
            // node to the APP value.
            // If we simply add the c2v value the app value and clamp to fp8,
            // when the c2v is subtracted (during the next iteration), the v2c
            // value will be inaccurate.
            // Example of the problem:
            //      original app:  400
            //      c2v increment: 300
            //      sum:           700
            //      clamped sum:   448
            //      Next iteration, when subtracting the c2V from the previous
            //      iteration we would subtract 300, which would not be accurate,
            //      as this check node only contributed 48.
            __half2 c2v_new     = __hmul2(bp_update_seq[i].f16x2, norm);
            __half2 app_new     = __hadd2(c2v_new, app[i].f16x2);
            __half2 app_clamped = clamp_signed(app_new, finite_clamp_value_as_half2<fp8_t>());
            c2v[i].f16x2        = __hsub2(app_clamped, app[i].f16x2);
            app[i].f16x2        = app_clamped;
        }

        //print_word_array<__half>(0, "NEW", app);
        //print_word_array<uint16_t>(0, "HEX", app);

    }
};

} // namespace ldpc2

#endif // !defined(LDPC2_C2V_FP8_CUH_INCLUDED_)
