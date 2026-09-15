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

#if !defined(LDPC2_APP_ADDRESS_NZS_UTIL_CUH_INCLUDED_)
#define LDPC2_APP_ADDRESS_NZS_UTIL_CUH_INCLUDED_

#include "ldpc2.hpp"
#include "ldpc2_bg_desc.hpp"

namespace ldpc2
{

// For shift = 0, threadIdx is always less than Z - shift = Z:
// addr_app = [(col_idx * Z * sizeof(T)) + (shift + threadIdx    ) * sizeof(T)]     threadIdx < (Z-shift)    (Z - threadIdx) >  shift
//          = [(col_idx * Z * sizeof(T)) + (         threadIdx   ) * sizeof(T)]
//          = sizeof(T) * [(col_idx * Z) + threadIdx]
template <int BG, int CHECK_IDX, typename TElem> struct zs_generator
{
    static
    __device__
    void generate(int Z,
                  int (&app_addr)[row_degree<BG, CHECK_IDX>::value])
    {
        constexpr int NUM_INFO_NODES = max_info_nodes<BG>::value;
        constexpr int COL_INDEX      = NUM_INFO_NODES + CHECK_IDX;
        constexpr int APP_INDEX      = row_degree<BG, CHECK_IDX>::value - 1;
        app_addr[APP_INDEX] = sizeof(TElem) * ((Z * COL_INDEX) + threadIdx.x);
    }
};

// Specialization for row 0
// Both base graphs (1 & 2) have zero shifts in the last column:
// BG1: column 23 ( = 22 + 1)
// BG2: column 11 ( = 10 + 1)
template <int BG, typename TElem> struct zs_generator<BG, 0, TElem>
{
    static
    __device__
    void generate(int Z,
                  int (&app_addr)[row_degree<BG, 0>::value])
    {
        constexpr int NUM_INFO_NODES = max_info_nodes<BG>::value;
        constexpr int COL_INDEX      = NUM_INFO_NODES + 1;
        constexpr int APP_INDEX      = row_degree<BG, 0>::value - 1;
        app_addr[APP_INDEX] = sizeof(TElem) * ((Z * COL_INDEX) + threadIdx.x);
    }
};

// Specialization for row 1
// Both base graphs (1 & 2) have zero shifts in the last 2 columns:
// BG1: columns 23 & 24
// BG2: columns 11 & 12
template <int BG, typename TElem> struct zs_generator<BG, 1, TElem>
{
    static
    __device__
    void generate(int Z,
                  int (&app_addr)[row_degree<BG, 1>::value])
    {
        constexpr int NUM_INFO_NODES = max_info_nodes<BG>::value;
        constexpr int COL_INDEX_0    = NUM_INFO_NODES + 1;
        constexpr int COL_INDEX_1    = NUM_INFO_NODES + 2;
        constexpr int APP_INDEX_0    = row_degree<BG, 1>::value - 2;
        constexpr int APP_INDEX_1    = row_degree<BG, 1>::value - 1;
        app_addr[APP_INDEX_0] = sizeof(TElem) * ((Z * COL_INDEX_0) + threadIdx.x);
        app_addr[APP_INDEX_1] = sizeof(TElem) * ((Z * COL_INDEX_1) + threadIdx.x);
    }
};

// Specialization for row 2
// Both base graphs (1 & 2) have zero shifts in the last 2 columns:
// BG1: columns 24 & 25
// BG2: columns 12 & 13
template <int BG, typename TElem> struct zs_generator<BG, 2, TElem>
{
    static
    __device__
    void generate(int Z,
                  int (&app_addr)[row_degree<BG, 2>::value])
    {
        constexpr int NUM_INFO_NODES = max_info_nodes<BG>::value;
        constexpr int COL_INDEX_0    = NUM_INFO_NODES + 2;
        constexpr int COL_INDEX_1    = NUM_INFO_NODES + 3;
        constexpr int APP_INDEX_0    = row_degree<BG, 2>::value - 2;
        constexpr int APP_INDEX_1    = row_degree<BG, 2>::value - 1;
        app_addr[APP_INDEX_0] = sizeof(TElem) * ((Z * COL_INDEX_0) + threadIdx.x);
        app_addr[APP_INDEX_1] = sizeof(TElem) * ((Z * COL_INDEX_1) + threadIdx.x);
    }
};

template <int BG, int CHECK_IDX, typename TElem>
struct single_address_generator
{
    static
    __device__
    void generate(int                    Z,
                  const BG_nzs_desc<BG>& bg_desc,
                  int                    (&app_addr)[row_degree<BG, CHECK_IDX>::value])
    {
        if constexpr (0 != (nzs_row_degree<BG, CHECK_IDX>::value % 2))
        {
            const int ROW_PAIR_OFFSET = nzs_row_pair_index<BG, CHECK_IDX>::value;
            const int PAIR_OFFSET     = ROW_PAIR_OFFSET + (nzs_row_degree<BG, CHECK_IDX>::value / 2);
            // Node values are not SIMD hi/lo pairs like they are for other columns
            uint32_t shift_mod        = bg_desc.nodes[PAIR_OFFSET].shift_mod;
            uint32_t col_Z_sz         = bg_desc.nodes[PAIR_OFFSET].col_Z_sz;
            int32_t RD                = shift_mod + threadIdx.x;
            int32_t APP_INDEX         = nzs_row_degree<BG, CHECK_IDX>::value - 1;
            if((Z - threadIdx.x) <= shift_mod)
            {
                RD -= Z;
            }
            app_addr[APP_INDEX] = RD * sizeof(TElem) + col_Z_sz;
        }
    }
    // For adj nodes
    static
    __device__
    void generate(int                        Z,
                  const BG_adj_nzs_desc<BG>& bg_desc,
                  int                        (&app_addr)[row_degree<BG, CHECK_IDX>::value])
    {
        if constexpr (0 != (nzs_row_degree<BG, CHECK_IDX>::value % 2))
        {
            const int ROW_PAIR_OFFSET = nzs_row_pair_index<BG, CHECK_IDX>::value;
            const int PAIR_OFFSET     = ROW_PAIR_OFFSET + (nzs_row_degree<BG, CHECK_IDX>::value / 2);
            // Node values are not SIMD hi/lo pairs like they are for other columns.
            // For the adj base graph nodes, we are storing shift_mode in the wrap_index
            // member, and col_Z_sz in the col_Z_shift_low member.
            uint32_t shift_mod        = bg_desc.nodes[PAIR_OFFSET].wrap_index;
            uint32_t col_Z_sz         = bg_desc.nodes[PAIR_OFFSET].col_Z_shift_low;
            int32_t RD                = shift_mod + threadIdx.x;
            int32_t APP_INDEX         = nzs_row_degree<BG, CHECK_IDX>::value - 1;
            if((Z - threadIdx.x) <= shift_mod)
            {
                RD -= Z;
            }
            app_addr[APP_INDEX] = RD * sizeof(TElem) + col_Z_sz;
        }
    }
};

} // namespace ldpc2

#endif // !defined(LDPC2_APP_ADDRESS_NZS_UTIL_CUH_INCLUDED_)
