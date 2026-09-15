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

#if !defined(LDPC2_BG_DESC_HPP_INCLUDED_)
#define LDPC2_BG_DESC_HPP_INCLUDED_

#include "nrLDPC_templates.cuh"
#include "cuphy_internal.h"
#include "cuda_fp16.h"
#include "cuda_fp8.h"

namespace ldpc2
{
    
////////////////////////////////////////////////////////////////////////
// row_pair_index
// Struct to provide the index of a struct or scalar that provides info
// for the first "row pair" in a given parity check row. (A row pair
// refers to a pair of elements in a parity check row.)
// When paired, rows with odd degree will have a padded last element.
// As an example, in BG1, the first row has 19 elements. The row pair
// index of row 0 will be 0, and the row pair index of row 1 will be
// 10 = (19 + 1) / 2.
template <int BG, int CHECK_IDX> struct row_pair_index
{
    static const int value = row_pair_index<BG, CHECK_IDX-1>::value + div_round_up_t<row_degree<BG, CHECK_IDX-1>::value, 2>::value;
};
template <int BG> struct row_pair_index<BG, 0>
{
    static const int value = 0;
};

////////////////////////////////////////////////////////////////////////
// BG1_ROW_ADDR_PAIR_IDX
// Provides array offsets when parity nodes are processes in PAIRS. For
// example, a row with degree 19 will contain ceil(19/2) nonzero variable
// nodes, and therefore we will allocate 10 "pair" descriptors.
enum BG1_ROW_ADDR_PAIR_IDX
{
    BG1_ADDR_PAIR_IDX_ROW_0  = row_pair_index<1, 0>::value,
    BG1_ADDR_PAIR_IDX_ROW_1  = row_pair_index<1, 1>::value,
    BG1_ADDR_PAIR_IDX_ROW_2  = row_pair_index<1, 2>::value,
    BG1_ADDR_PAIR_IDX_ROW_3  = row_pair_index<1, 3>::value,
    BG1_ADDR_PAIR_IDX_ROW_4  = row_pair_index<1, 4>::value,
    BG1_ADDR_PAIR_IDX_ROW_5  = row_pair_index<1, 5>::value,
    BG1_ADDR_PAIR_IDX_ROW_6  = row_pair_index<1, 6>::value,
    BG1_ADDR_PAIR_IDX_ROW_7  = row_pair_index<1, 7>::value,
    BG1_ADDR_PAIR_IDX_ROW_8  = row_pair_index<1, 8>::value,
    BG1_ADDR_PAIR_IDX_ROW_9  = row_pair_index<1, 9>::value,
    BG1_ADDR_PAIR_IDX_ROW_10 = row_pair_index<1, 10>::value,
    BG1_ADDR_PAIR_IDX_ROW_11 = row_pair_index<1, 11>::value,
    BG1_ADDR_PAIR_IDX_ROW_12 = row_pair_index<1, 12>::value,
    BG1_ADDR_PAIR_IDX_ROW_13 = row_pair_index<1, 13>::value,
    BG1_ADDR_PAIR_IDX_ROW_14 = row_pair_index<1, 14>::value,
    BG1_ADDR_PAIR_IDX_ROW_15 = row_pair_index<1, 15>::value,
    BG1_ADDR_PAIR_IDX_ROW_16 = row_pair_index<1, 16>::value,
    BG1_ADDR_PAIR_IDX_ROW_17 = row_pair_index<1, 17>::value,
    BG1_ADDR_PAIR_IDX_ROW_18 = row_pair_index<1, 18>::value,
    BG1_ADDR_PAIR_IDX_ROW_19 = row_pair_index<1, 19>::value,
    BG1_ADDR_PAIR_IDX_ROW_20 = row_pair_index<1, 20>::value,
    BG1_ADDR_PAIR_IDX_ROW_21 = row_pair_index<1, 21>::value,
    BG1_ADDR_PAIR_IDX_ROW_22 = row_pair_index<1, 22>::value,
    BG1_ADDR_PAIR_IDX_ROW_23 = row_pair_index<1, 23>::value,
    BG1_ADDR_PAIR_IDX_ROW_24 = row_pair_index<1, 24>::value,
    BG1_ADDR_PAIR_IDX_ROW_25 = row_pair_index<1, 25>::value,
    BG1_ADDR_PAIR_IDX_ROW_26 = row_pair_index<1, 26>::value,
    BG1_ADDR_PAIR_IDX_ROW_27 = row_pair_index<1, 27>::value,
    BG1_ADDR_PAIR_IDX_ROW_28 = row_pair_index<1, 28>::value,
    BG1_ADDR_PAIR_IDX_ROW_29 = row_pair_index<1, 29>::value,
    BG1_ADDR_PAIR_IDX_ROW_30 = row_pair_index<1, 30>::value,
    BG1_ADDR_PAIR_IDX_ROW_31 = row_pair_index<1, 31>::value,
    BG1_ADDR_PAIR_IDX_ROW_32 = row_pair_index<1, 32>::value,
    BG1_ADDR_PAIR_IDX_ROW_33 = row_pair_index<1, 33>::value,
    BG1_ADDR_PAIR_IDX_ROW_34 = row_pair_index<1, 34>::value,
    BG1_ADDR_PAIR_IDX_ROW_35 = row_pair_index<1, 35>::value,
    BG1_ADDR_PAIR_IDX_ROW_36 = row_pair_index<1, 36>::value,
    BG1_ADDR_PAIR_IDX_ROW_37 = row_pair_index<1, 37>::value,
    BG1_ADDR_PAIR_IDX_ROW_38 = row_pair_index<1, 38>::value,
    BG1_ADDR_PAIR_IDX_ROW_39 = row_pair_index<1, 39>::value,
    BG1_ADDR_PAIR_IDX_ROW_40 = row_pair_index<1, 40>::value,
    BG1_ADDR_PAIR_IDX_ROW_41 = row_pair_index<1, 41>::value,
    BG1_ADDR_PAIR_IDX_ROW_42 = row_pair_index<1, 42>::value,
    BG1_ADDR_PAIR_IDX_ROW_43 = row_pair_index<1, 43>::value,
    BG1_ADDR_PAIR_IDX_ROW_44 = row_pair_index<1, 44>::value,
    BG1_ADDR_PAIR_IDX_ROW_45 = row_pair_index<1, 45>::value,
    BG1_ADDR_PAIR_COUNT      = row_pair_index<1, 45>::value + div_round_up_t<row_degree<1,45>::value, 2>::value
};

////////////////////////////////////////////////////////////////////////
// BG2_ROW_ADDR_PAIR_IDX
// Provides array offsets when parity nodes are processes in PAIRS. For
// example, a row with degree 19 will contain ceil(19/2) nonzero variable
// nodes, and therefore we will allocate 10 "pair" descriptors.
enum BG2_ROW_ADDR_PAIR_IDX
{
    BG2_ADDR_PAIR_IDX_ROW_0  = row_pair_index<2, 0>::value,
    BG2_ADDR_PAIR_IDX_ROW_1  = row_pair_index<2, 1>::value,
    BG2_ADDR_PAIR_IDX_ROW_2  = row_pair_index<2, 2>::value,
    BG2_ADDR_PAIR_IDX_ROW_3  = row_pair_index<2, 3>::value,
    BG2_ADDR_PAIR_IDX_ROW_4  = row_pair_index<2, 4>::value,
    BG2_ADDR_PAIR_IDX_ROW_5  = row_pair_index<2, 5>::value,
    BG2_ADDR_PAIR_IDX_ROW_6  = row_pair_index<2, 6>::value,
    BG2_ADDR_PAIR_IDX_ROW_7  = row_pair_index<2, 7>::value,
    BG2_ADDR_PAIR_IDX_ROW_8  = row_pair_index<2, 8>::value,
    BG2_ADDR_PAIR_IDX_ROW_9  = row_pair_index<2, 9>::value,
    BG2_ADDR_PAIR_IDX_ROW_10 = row_pair_index<2, 10>::value,
    BG2_ADDR_PAIR_IDX_ROW_11 = row_pair_index<2, 11>::value,
    BG2_ADDR_PAIR_IDX_ROW_12 = row_pair_index<2, 12>::value,
    BG2_ADDR_PAIR_IDX_ROW_13 = row_pair_index<2, 13>::value,
    BG2_ADDR_PAIR_IDX_ROW_14 = row_pair_index<2, 14>::value,
    BG2_ADDR_PAIR_IDX_ROW_15 = row_pair_index<2, 15>::value,
    BG2_ADDR_PAIR_IDX_ROW_16 = row_pair_index<2, 16>::value,
    BG2_ADDR_PAIR_IDX_ROW_17 = row_pair_index<2, 17>::value,
    BG2_ADDR_PAIR_IDX_ROW_18 = row_pair_index<2, 18>::value,
    BG2_ADDR_PAIR_IDX_ROW_19 = row_pair_index<2, 19>::value,
    BG2_ADDR_PAIR_IDX_ROW_20 = row_pair_index<2, 20>::value,
    BG2_ADDR_PAIR_IDX_ROW_21 = row_pair_index<2, 21>::value,
    BG2_ADDR_PAIR_IDX_ROW_22 = row_pair_index<2, 22>::value,
    BG2_ADDR_PAIR_IDX_ROW_23 = row_pair_index<2, 23>::value,
    BG2_ADDR_PAIR_IDX_ROW_24 = row_pair_index<2, 24>::value,
    BG2_ADDR_PAIR_IDX_ROW_25 = row_pair_index<2, 25>::value,
    BG2_ADDR_PAIR_IDX_ROW_26 = row_pair_index<2, 26>::value,
    BG2_ADDR_PAIR_IDX_ROW_27 = row_pair_index<2, 27>::value,
    BG2_ADDR_PAIR_IDX_ROW_28 = row_pair_index<2, 28>::value,
    BG2_ADDR_PAIR_IDX_ROW_29 = row_pair_index<2, 29>::value,
    BG2_ADDR_PAIR_IDX_ROW_30 = row_pair_index<2, 30>::value,
    BG2_ADDR_PAIR_IDX_ROW_31 = row_pair_index<2, 31>::value,
    BG2_ADDR_PAIR_IDX_ROW_32 = row_pair_index<2, 32>::value,
    BG2_ADDR_PAIR_IDX_ROW_33 = row_pair_index<2, 33>::value,
    BG2_ADDR_PAIR_IDX_ROW_34 = row_pair_index<2, 34>::value,
    BG2_ADDR_PAIR_IDX_ROW_35 = row_pair_index<2, 35>::value,
    BG2_ADDR_PAIR_IDX_ROW_36 = row_pair_index<2, 36>::value,
    BG2_ADDR_PAIR_IDX_ROW_37 = row_pair_index<2, 37>::value,
    BG2_ADDR_PAIR_IDX_ROW_38 = row_pair_index<2, 38>::value,
    BG2_ADDR_PAIR_IDX_ROW_39 = row_pair_index<2, 39>::value,
    BG2_ADDR_PAIR_IDX_ROW_40 = row_pair_index<2, 40>::value,
    BG2_ADDR_PAIR_IDX_ROW_41 = row_pair_index<2, 41>::value,
    BG2_ADDR_PAIR_COUNT      = row_pair_index<2, 41>::value + div_round_up_t<row_degree<2,41>::value, 2>::value
};

////////////////////////////////////////////////////////////////////////
// nzs_row_pair_index
// Struct to provide the index of a struct or scalar that provides info
// for the first "row pair" in a given parity check row. (A row pair
// refers to a pair of elements in a parity check row.)
// Compared to the row_pair_index structure above, the nzs structure
// does not use storage for base graph variable nodes that have a zero
// shift value (for all lifting size sets & lifting sizes). Address
// calculation for zero shift values is simpler and can be implemented
// more efficiently.
// Rows with odd non-zero shift row degree will have the last element
// encoded as a single integer value instead of a SIMD pair of hi and
// lo values.
// As an example, in BG1, the first row has 19 elements. The shift
// value of the last column is zero for all lifting sizes. Therefore,
// the non-zero shift row degree is 18. 9 column/shift pairs will be
// stored.
// As another example, in BG2, the second row also has 19 elements, but
// the last 2 columns have zero shifts. Therefore, the non-zero shift
// row degree is 17. 8 pairs will be encoded as SIMD hi/lo pairs, and
// the final node entry will use the full 32-bit integer for the last
// nonzero shift and column.
template <int BG, int CHECK_IDX> struct nzs_row_pair_index
{
    static const int value = nzs_row_pair_index<BG, CHECK_IDX-1>::value + div_round_up_t<nzs_row_degree<BG, CHECK_IDX-1>::value, 2>::value;
};
template <int BG> struct nzs_row_pair_index<BG, 0>
{
    static const int value = 0;
};

////////////////////////////////////////////////////////////////////////
// BG1_NZS_ROW_ADDR_PAIR_IDX
// Provides array offsets when parity nodes are processed in PAIRS. For
// example, a row with degree 19 will contain ceil(19/2) nonzero variable
// nodes, and therefore we will allocate 10 "pair" descriptors.
enum BG1_NZS_ROW_ADDR_PAIR_IDX
{
    BG1_NZS_ADDR_PAIR_IDX_ROW_0  = nzs_row_pair_index<1, 0>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_1  = nzs_row_pair_index<1, 1>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_2  = nzs_row_pair_index<1, 2>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_3  = nzs_row_pair_index<1, 3>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_4  = nzs_row_pair_index<1, 4>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_5  = nzs_row_pair_index<1, 5>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_6  = nzs_row_pair_index<1, 6>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_7  = nzs_row_pair_index<1, 7>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_8  = nzs_row_pair_index<1, 8>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_9  = nzs_row_pair_index<1, 9>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_10 = nzs_row_pair_index<1, 10>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_11 = nzs_row_pair_index<1, 11>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_12 = nzs_row_pair_index<1, 12>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_13 = nzs_row_pair_index<1, 13>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_14 = nzs_row_pair_index<1, 14>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_15 = nzs_row_pair_index<1, 15>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_16 = nzs_row_pair_index<1, 16>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_17 = nzs_row_pair_index<1, 17>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_18 = nzs_row_pair_index<1, 18>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_19 = nzs_row_pair_index<1, 19>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_20 = nzs_row_pair_index<1, 20>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_21 = nzs_row_pair_index<1, 21>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_22 = nzs_row_pair_index<1, 22>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_23 = nzs_row_pair_index<1, 23>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_24 = nzs_row_pair_index<1, 24>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_25 = nzs_row_pair_index<1, 25>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_26 = nzs_row_pair_index<1, 26>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_27 = nzs_row_pair_index<1, 27>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_28 = nzs_row_pair_index<1, 28>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_29 = nzs_row_pair_index<1, 29>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_30 = nzs_row_pair_index<1, 30>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_31 = nzs_row_pair_index<1, 31>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_32 = nzs_row_pair_index<1, 32>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_33 = nzs_row_pair_index<1, 33>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_34 = nzs_row_pair_index<1, 34>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_35 = nzs_row_pair_index<1, 35>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_36 = nzs_row_pair_index<1, 36>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_37 = nzs_row_pair_index<1, 37>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_38 = nzs_row_pair_index<1, 38>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_39 = nzs_row_pair_index<1, 39>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_40 = nzs_row_pair_index<1, 40>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_41 = nzs_row_pair_index<1, 41>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_42 = nzs_row_pair_index<1, 42>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_43 = nzs_row_pair_index<1, 43>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_44 = nzs_row_pair_index<1, 44>::value,
    BG1_NZS_ADDR_PAIR_IDX_ROW_45 = nzs_row_pair_index<1, 45>::value,
    BG1_NZS_ADDR_PAIR_COUNT      = nzs_row_pair_index<1, 45>::value + div_round_up_t<nzs_row_degree<1,45>::value, 2>::value
};

////////////////////////////////////////////////////////////////////////
// BG2_NZS_ROW_ADDR_PAIR_IDX
// Provides array offsets when parity nodes are processed in PAIRS. For
// example, a row with degree 19 will contain ceil(19/2) nonzero variable
// nodes, and therefore we will allocate 10 "pair" descriptors.
enum BG2_NZS_ROW_ADDR_PAIR_IDX
{
    BG2_NZS_ADDR_PAIR_IDX_ROW_0  = nzs_row_pair_index<2, 0>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_1  = nzs_row_pair_index<2, 1>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_2  = nzs_row_pair_index<2, 2>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_3  = nzs_row_pair_index<2, 3>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_4  = nzs_row_pair_index<2, 4>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_5  = nzs_row_pair_index<2, 5>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_6  = nzs_row_pair_index<2, 6>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_7  = nzs_row_pair_index<2, 7>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_8  = nzs_row_pair_index<2, 8>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_9  = nzs_row_pair_index<2, 9>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_10 = nzs_row_pair_index<2, 10>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_11 = nzs_row_pair_index<2, 11>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_12 = nzs_row_pair_index<2, 12>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_13 = nzs_row_pair_index<2, 13>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_14 = nzs_row_pair_index<2, 14>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_15 = nzs_row_pair_index<2, 15>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_16 = nzs_row_pair_index<2, 16>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_17 = nzs_row_pair_index<2, 17>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_18 = nzs_row_pair_index<2, 18>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_19 = nzs_row_pair_index<2, 19>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_20 = nzs_row_pair_index<2, 20>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_21 = nzs_row_pair_index<2, 21>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_22 = nzs_row_pair_index<2, 22>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_23 = nzs_row_pair_index<2, 23>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_24 = nzs_row_pair_index<2, 24>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_25 = nzs_row_pair_index<2, 25>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_26 = nzs_row_pair_index<2, 26>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_27 = nzs_row_pair_index<2, 27>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_28 = nzs_row_pair_index<2, 28>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_29 = nzs_row_pair_index<2, 29>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_30 = nzs_row_pair_index<2, 30>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_31 = nzs_row_pair_index<2, 31>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_32 = nzs_row_pair_index<2, 32>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_33 = nzs_row_pair_index<2, 33>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_34 = nzs_row_pair_index<2, 34>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_35 = nzs_row_pair_index<2, 35>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_36 = nzs_row_pair_index<2, 36>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_37 = nzs_row_pair_index<2, 37>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_38 = nzs_row_pair_index<2, 38>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_39 = nzs_row_pair_index<2, 39>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_40 = nzs_row_pair_index<2, 40>::value,
    BG2_NZS_ADDR_PAIR_IDX_ROW_41 = nzs_row_pair_index<2, 41>::value,
    BG2_NZS_ADDR_PAIR_COUNT      = nzs_row_pair_index<2, 41>::value + div_round_up_t<nzs_row_degree<2,41>::value, 2>::value
};

////////////////////////////////////////////////////////////////////////
// ldpc2::LDPC_node_desc
// 3GPP 5G Base Graph Node Descriptor
struct LDPC_node_desc
{
    uint32_t shift_mod;
    uint32_t col_Z_sz;
};

template <int BG> struct BG_desc;
template <> struct BG_desc<1>
{
    LDPC_node_desc nodes[BG1_ADDR_PAIR_COUNT];
};
template <> struct BG_desc<2>
{
    LDPC_node_desc nodes[BG2_ADDR_PAIR_COUNT];
};

typedef BG_desc<1> BG1_desc_t;
typedef struct BG_desc<2> BG2_desc_t;

// Host descriptor declarations
extern const BG1_desc_t BG1_desc_Z32_8;
extern const BG1_desc_t BG1_desc_Z36_8;
extern const BG1_desc_t BG1_desc_Z40_8;
extern const BG1_desc_t BG1_desc_Z44_8;
extern const BG1_desc_t BG1_desc_Z48_8;
extern const BG1_desc_t BG1_desc_Z52_8;
extern const BG1_desc_t BG1_desc_Z56_8;
extern const BG1_desc_t BG1_desc_Z60_8;
extern const BG1_desc_t BG1_desc_Z64_8;
extern const BG1_desc_t BG1_desc_Z72_8;
extern const BG1_desc_t BG1_desc_Z80_8;
extern const BG1_desc_t BG1_desc_Z88_8;
extern const BG1_desc_t BG1_desc_Z96_8;
extern const BG1_desc_t BG1_desc_Z104_8;
extern const BG1_desc_t BG1_desc_Z112_8;
extern const BG1_desc_t BG1_desc_Z120_8;
extern const BG1_desc_t BG1_desc_Z128_8;
extern const BG1_desc_t BG1_desc_Z144_8;
extern const BG1_desc_t BG1_desc_Z160_8;
extern const BG1_desc_t BG1_desc_Z176_8;
extern const BG1_desc_t BG1_desc_Z192_8;
extern const BG1_desc_t BG1_desc_Z208_8;
extern const BG1_desc_t BG1_desc_Z224_8;
extern const BG1_desc_t BG1_desc_Z240_8;
extern const BG1_desc_t BG1_desc_Z256_8;
extern const BG1_desc_t BG1_desc_Z288_8;
extern const BG1_desc_t BG1_desc_Z320_8;
extern const BG1_desc_t BG1_desc_Z352_8;
extern const BG1_desc_t BG1_desc_Z384_8;

extern const BG1_desc_t BG1_desc_Z32_16;
extern const BG1_desc_t BG1_desc_Z36_16;
extern const BG1_desc_t BG1_desc_Z40_16;
extern const BG1_desc_t BG1_desc_Z44_16;
extern const BG1_desc_t BG1_desc_Z48_16;
extern const BG1_desc_t BG1_desc_Z52_16;
extern const BG1_desc_t BG1_desc_Z56_16;
extern const BG1_desc_t BG1_desc_Z60_16;
extern const BG1_desc_t BG1_desc_Z64_16;
extern const BG1_desc_t BG1_desc_Z72_16;
extern const BG1_desc_t BG1_desc_Z80_16;
extern const BG1_desc_t BG1_desc_Z88_16;
extern const BG1_desc_t BG1_desc_Z96_16;
extern const BG1_desc_t BG1_desc_Z104_16;
extern const BG1_desc_t BG1_desc_Z112_16;
extern const BG1_desc_t BG1_desc_Z120_16;
extern const BG1_desc_t BG1_desc_Z128_16;
extern const BG1_desc_t BG1_desc_Z144_16;
extern const BG1_desc_t BG1_desc_Z160_16;
extern const BG1_desc_t BG1_desc_Z176_16;
extern const BG1_desc_t BG1_desc_Z192_16;
extern const BG1_desc_t BG1_desc_Z208_16;
extern const BG1_desc_t BG1_desc_Z224_16;
extern const BG1_desc_t BG1_desc_Z240_16;
extern const BG1_desc_t BG1_desc_Z256_16;
extern const BG1_desc_t BG1_desc_Z288_16;
extern const BG1_desc_t BG1_desc_Z320_16;
extern const BG1_desc_t BG1_desc_Z352_16;
extern const BG1_desc_t BG1_desc_Z384_16;

extern const BG2_desc_t BG2_desc_Z32_8;
extern const BG2_desc_t BG2_desc_Z36_8;
extern const BG2_desc_t BG2_desc_Z40_8;
extern const BG2_desc_t BG2_desc_Z44_8;
extern const BG2_desc_t BG2_desc_Z48_8;
extern const BG2_desc_t BG2_desc_Z52_8;
extern const BG2_desc_t BG2_desc_Z56_8;
extern const BG2_desc_t BG2_desc_Z60_8;
extern const BG2_desc_t BG2_desc_Z64_8;
extern const BG2_desc_t BG2_desc_Z72_8;
extern const BG2_desc_t BG2_desc_Z80_8;
extern const BG2_desc_t BG2_desc_Z88_8;
extern const BG2_desc_t BG2_desc_Z96_8;
extern const BG2_desc_t BG2_desc_Z104_8;
extern const BG2_desc_t BG2_desc_Z112_8;
extern const BG2_desc_t BG2_desc_Z120_8;
extern const BG2_desc_t BG2_desc_Z128_8;
extern const BG2_desc_t BG2_desc_Z144_8;
extern const BG2_desc_t BG2_desc_Z160_8;
extern const BG2_desc_t BG2_desc_Z176_8;
extern const BG2_desc_t BG2_desc_Z192_8;
extern const BG2_desc_t BG2_desc_Z208_8;
extern const BG2_desc_t BG2_desc_Z224_8;
extern const BG2_desc_t BG2_desc_Z240_8;
extern const BG2_desc_t BG2_desc_Z256_8;
extern const BG2_desc_t BG2_desc_Z288_8;
extern const BG2_desc_t BG2_desc_Z320_8;
extern const BG2_desc_t BG2_desc_Z352_8;
extern const BG2_desc_t BG2_desc_Z384_8;

extern const BG2_desc_t BG2_desc_Z32_16;
extern const BG2_desc_t BG2_desc_Z36_16;
extern const BG2_desc_t BG2_desc_Z40_16;
extern const BG2_desc_t BG2_desc_Z44_16;
extern const BG2_desc_t BG2_desc_Z48_16;
extern const BG2_desc_t BG2_desc_Z52_16;
extern const BG2_desc_t BG2_desc_Z56_16;
extern const BG2_desc_t BG2_desc_Z60_16;
extern const BG2_desc_t BG2_desc_Z64_16;
extern const BG2_desc_t BG2_desc_Z72_16;
extern const BG2_desc_t BG2_desc_Z80_16;
extern const BG2_desc_t BG2_desc_Z88_16;
extern const BG2_desc_t BG2_desc_Z96_16;
extern const BG2_desc_t BG2_desc_Z104_16;
extern const BG2_desc_t BG2_desc_Z112_16;
extern const BG2_desc_t BG2_desc_Z120_16;
extern const BG2_desc_t BG2_desc_Z128_16;
extern const BG2_desc_t BG2_desc_Z144_16;
extern const BG2_desc_t BG2_desc_Z160_16;
extern const BG2_desc_t BG2_desc_Z176_16;
extern const BG2_desc_t BG2_desc_Z192_16;
extern const BG2_desc_t BG2_desc_Z208_16;
extern const BG2_desc_t BG2_desc_Z224_16;
extern const BG2_desc_t BG2_desc_Z240_16;
extern const BG2_desc_t BG2_desc_Z256_16;
extern const BG2_desc_t BG2_desc_Z288_16;
extern const BG2_desc_t BG2_desc_Z320_16;
extern const BG2_desc_t BG2_desc_Z352_16;
extern const BG2_desc_t BG2_desc_Z384_16;

template <int TBitwidth, int BG> const BG_desc<BG>* get_BG_desc_bitwidth(int Z);

template <>
inline
const BG_desc<1>* get_BG_desc_bitwidth<8, 1>(int Z)
{
    const BG1_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG1_desc_Z32_8;  break;
    case 36:  bgd = &BG1_desc_Z36_8;  break;
    case 40:  bgd = &BG1_desc_Z40_8;  break;
    case 44:  bgd = &BG1_desc_Z44_8;  break;
    case 48:  bgd = &BG1_desc_Z48_8;  break;
    case 52:  bgd = &BG1_desc_Z52_8;  break;
    case 56:  bgd = &BG1_desc_Z56_8;  break;
    case 60:  bgd = &BG1_desc_Z60_8;  break;
    case 64:  bgd = &BG1_desc_Z64_8;  break;
    case 72:  bgd = &BG1_desc_Z72_8;  break;
    case 80:  bgd = &BG1_desc_Z80_8;  break;
    case 88:  bgd = &BG1_desc_Z88_8;  break;
    case 96:  bgd = &BG1_desc_Z96_8;  break;
    case 104: bgd = &BG1_desc_Z104_8; break;
    case 112: bgd = &BG1_desc_Z112_8; break;
    case 120: bgd = &BG1_desc_Z120_8; break;
    case 128: bgd = &BG1_desc_Z128_8; break;
    case 144: bgd = &BG1_desc_Z144_8; break;
    case 160: bgd = &BG1_desc_Z160_8; break;
    case 176: bgd = &BG1_desc_Z176_8; break;
    case 192: bgd = &BG1_desc_Z192_8; break;
    case 208: bgd = &BG1_desc_Z208_8; break;
    case 224: bgd = &BG1_desc_Z224_8; break;
    case 240: bgd = &BG1_desc_Z240_8; break;
    case 256: bgd = &BG1_desc_Z256_8; break;
    case 288: bgd = &BG1_desc_Z288_8; break;
    case 320: bgd = &BG1_desc_Z320_8; break;
    case 352: bgd = &BG1_desc_Z352_8; break;
    case 384: bgd = &BG1_desc_Z384_8; break;
    }
    return bgd;
}

template <>
inline
const BG_desc<2>* get_BG_desc_bitwidth<8, 2>(int Z)
{
    const BG2_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG2_desc_Z32_8;  break;
    case 36:  bgd = &BG2_desc_Z36_8;  break;
    case 40:  bgd = &BG2_desc_Z40_8;  break;
    case 44:  bgd = &BG2_desc_Z44_8;  break;
    case 48:  bgd = &BG2_desc_Z48_8;  break;
    case 52:  bgd = &BG2_desc_Z52_8;  break;
    case 56:  bgd = &BG2_desc_Z56_8;  break;
    case 60:  bgd = &BG2_desc_Z60_8;  break;
    case 64:  bgd = &BG2_desc_Z64_8;  break;
    case 72:  bgd = &BG2_desc_Z72_8;  break;
    case 80:  bgd = &BG2_desc_Z80_8;  break;
    case 88:  bgd = &BG2_desc_Z88_8;  break;
    case 96:  bgd = &BG2_desc_Z96_8;  break;
    case 104: bgd = &BG2_desc_Z104_8; break;
    case 112: bgd = &BG2_desc_Z112_8; break;
    case 120: bgd = &BG2_desc_Z120_8; break;
    case 128: bgd = &BG2_desc_Z128_8; break;
    case 144: bgd = &BG2_desc_Z144_8; break;
    case 160: bgd = &BG2_desc_Z160_8; break;
    case 176: bgd = &BG2_desc_Z176_8; break;
    case 192: bgd = &BG2_desc_Z192_8; break;
    case 208: bgd = &BG2_desc_Z208_8; break;
    case 224: bgd = &BG2_desc_Z224_8; break;
    case 240: bgd = &BG2_desc_Z240_8; break;
    case 256: bgd = &BG2_desc_Z256_8; break;
    case 288: bgd = &BG2_desc_Z288_8; break;
    case 320: bgd = &BG2_desc_Z320_8; break;
    case 352: bgd = &BG2_desc_Z352_8; break;
    case 384: bgd = &BG2_desc_Z384_8; break;
    }
    return bgd;
}

template <>
inline
const BG_desc<1>* get_BG_desc_bitwidth<16, 1>(int Z)
{
    const BG1_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG1_desc_Z32_16;  break;
    case 36:  bgd = &BG1_desc_Z36_16;  break;
    case 40:  bgd = &BG1_desc_Z40_16;  break;
    case 44:  bgd = &BG1_desc_Z44_16;  break;
    case 48:  bgd = &BG1_desc_Z48_16;  break;
    case 52:  bgd = &BG1_desc_Z52_16;  break;
    case 56:  bgd = &BG1_desc_Z56_16;  break;
    case 60:  bgd = &BG1_desc_Z60_16;  break;
    case 64:  bgd = &BG1_desc_Z64_16;  break;
    case 72:  bgd = &BG1_desc_Z72_16;  break;
    case 80:  bgd = &BG1_desc_Z80_16;  break;
    case 88:  bgd = &BG1_desc_Z88_16;  break;
    case 96:  bgd = &BG1_desc_Z96_16;  break;
    case 104: bgd = &BG1_desc_Z104_16; break;
    case 112: bgd = &BG1_desc_Z112_16; break;
    case 120: bgd = &BG1_desc_Z120_16; break;
    case 128: bgd = &BG1_desc_Z128_16; break;
    case 144: bgd = &BG1_desc_Z144_16; break;
    case 160: bgd = &BG1_desc_Z160_16; break;
    case 176: bgd = &BG1_desc_Z176_16; break;
    case 192: bgd = &BG1_desc_Z192_16; break;
    case 208: bgd = &BG1_desc_Z208_16; break;
    case 224: bgd = &BG1_desc_Z224_16; break;
    case 240: bgd = &BG1_desc_Z240_16; break;
    case 256: bgd = &BG1_desc_Z256_16; break;
    case 288: bgd = &BG1_desc_Z288_16; break;
    case 320: bgd = &BG1_desc_Z320_16; break;
    case 352: bgd = &BG1_desc_Z352_16; break;
    case 384: bgd = &BG1_desc_Z384_16; break;
    }
    return bgd;
}

template <>
inline
const BG_desc<2>* get_BG_desc_bitwidth<16, 2>(int Z)
{
    const BG2_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG2_desc_Z32_16;  break;
    case 36:  bgd = &BG2_desc_Z36_16;  break;
    case 40:  bgd = &BG2_desc_Z40_16;  break;
    case 44:  bgd = &BG2_desc_Z44_16;  break;
    case 48:  bgd = &BG2_desc_Z48_16;  break;
    case 52:  bgd = &BG2_desc_Z52_16;  break;
    case 56:  bgd = &BG2_desc_Z56_16;  break;
    case 60:  bgd = &BG2_desc_Z60_16;  break;
    case 64:  bgd = &BG2_desc_Z64_16;  break;
    case 72:  bgd = &BG2_desc_Z72_16;  break;
    case 80:  bgd = &BG2_desc_Z80_16;  break;
    case 88:  bgd = &BG2_desc_Z88_16;  break;
    case 96:  bgd = &BG2_desc_Z96_16;  break;
    case 104: bgd = &BG2_desc_Z104_16; break;
    case 112: bgd = &BG2_desc_Z112_16; break;
    case 120: bgd = &BG2_desc_Z120_16; break;
    case 128: bgd = &BG2_desc_Z128_16; break;
    case 144: bgd = &BG2_desc_Z144_16; break;
    case 160: bgd = &BG2_desc_Z160_16; break;
    case 176: bgd = &BG2_desc_Z176_16; break;
    case 192: bgd = &BG2_desc_Z192_16; break;
    case 208: bgd = &BG2_desc_Z208_16; break;
    case 224: bgd = &BG2_desc_Z224_16; break;
    case 240: bgd = &BG2_desc_Z240_16; break;
    case 256: bgd = &BG2_desc_Z256_16; break;
    case 288: bgd = &BG2_desc_Z288_16; break;
    case 320: bgd = &BG2_desc_Z320_16; break;
    case 352: bgd = &BG2_desc_Z352_16; break;
    case 384: bgd = &BG2_desc_Z384_16; break;
    }
    return bgd;
}

template <typename T,
          int      BG>
const BG_desc<BG>* get_BG_desc(int Z)
{
    return get_BG_desc_bitwidth<sizeof(T) * CHAR_BIT, BG>(Z);
}

////////////////////////////////////////////////////////////////////////
// ldpc2::LDPC_adj_node_desc
// 3GPP 5G Base Graph Node Descriptor (alternate "adjusted" data storage
// format)
struct LDPC_adj_node_desc
{
    uint32_t wrap_index;
    int32_t  col_Z_shift_low;
    int32_t  col_Z_shift_high;
};

template <int BG> struct BG_adj_desc;
template <> struct BG_adj_desc<1>
{
    LDPC_adj_node_desc nodes[BG1_ADDR_PAIR_COUNT];
};
template <> struct BG_adj_desc<2>
{
    LDPC_adj_node_desc nodes[BG2_ADDR_PAIR_COUNT];
};

typedef BG_adj_desc<1> BG1_adj_desc_t;
typedef struct BG_adj_desc<2> BG2_adj_desc_t;

// Host descriptor declarations
extern const BG1_adj_desc_t BG1_adj_desc_Z32_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z36_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z40_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z44_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z48_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z52_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z56_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z60_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z64_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z72_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z80_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z88_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z96_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z104_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z112_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z120_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z128_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z144_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z160_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z176_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z192_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z208_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z224_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z240_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z256_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z288_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z320_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z352_8;
extern const BG1_adj_desc_t BG1_adj_desc_Z384_8;

extern const BG1_adj_desc_t BG1_adj_desc_Z2_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z3_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z4_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z5_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z6_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z7_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z8_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z9_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z10_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z11_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z12_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z13_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z14_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z15_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z16_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z18_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z20_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z22_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z24_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z26_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z28_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z30_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z32_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z36_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z40_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z44_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z48_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z52_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z56_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z60_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z64_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z72_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z80_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z88_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z96_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z104_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z112_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z120_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z128_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z144_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z160_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z176_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z192_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z208_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z224_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z240_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z256_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z288_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z320_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z352_16;
extern const BG1_adj_desc_t BG1_adj_desc_Z384_16;

extern const BG1_adj_desc_t BG1_adj_desc_Z32_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z36_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z40_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z44_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z48_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z52_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z56_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z60_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z64_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z72_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z80_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z88_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z96_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z104_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z112_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z120_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z128_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z144_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z160_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z176_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z192_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z208_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z224_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z240_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z256_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z288_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z320_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z352_32;
extern const BG1_adj_desc_t BG1_adj_desc_Z384_32;

extern const BG2_adj_desc_t BG2_adj_desc_Z32_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z36_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z40_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z44_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z48_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z52_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z56_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z60_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z64_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z72_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z80_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z88_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z96_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z104_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z112_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z120_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z128_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z144_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z160_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z176_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z192_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z208_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z224_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z240_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z256_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z288_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z320_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z352_8;
extern const BG2_adj_desc_t BG2_adj_desc_Z384_8;

extern const BG2_adj_desc_t BG2_adj_desc_Z2_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z3_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z4_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z5_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z6_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z7_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z8_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z9_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z10_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z11_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z12_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z13_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z14_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z15_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z16_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z18_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z20_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z22_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z24_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z26_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z28_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z30_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z32_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z36_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z40_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z44_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z48_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z52_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z56_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z60_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z64_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z72_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z80_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z88_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z96_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z104_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z112_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z120_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z128_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z144_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z160_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z176_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z192_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z208_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z224_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z240_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z256_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z288_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z320_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z352_16;
extern const BG2_adj_desc_t BG2_adj_desc_Z384_16;

extern const BG2_adj_desc_t BG2_adj_desc_Z32_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z36_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z40_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z44_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z48_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z52_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z56_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z60_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z64_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z72_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z80_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z88_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z96_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z104_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z112_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z120_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z128_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z144_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z160_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z176_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z192_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z208_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z224_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z240_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z256_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z288_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z320_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z352_32;
extern const BG2_adj_desc_t BG2_adj_desc_Z384_32;

template <int TBitwidth,
          int BG>
const BG_adj_desc<BG>* get_adj_BG_desc_bitwidth(int Z);

template <>
inline
const BG_adj_desc<1>* get_adj_BG_desc_bitwidth<8, 1>(int Z)
{
    const BG1_adj_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG1_adj_desc_Z32_8;  break;
    case 36:  bgd = &BG1_adj_desc_Z36_8;  break;
    case 40:  bgd = &BG1_adj_desc_Z40_8;  break;
    case 44:  bgd = &BG1_adj_desc_Z44_8;  break;
    case 48:  bgd = &BG1_adj_desc_Z48_8;  break;
    case 52:  bgd = &BG1_adj_desc_Z52_8;  break;
    case 56:  bgd = &BG1_adj_desc_Z56_8;  break;
    case 60:  bgd = &BG1_adj_desc_Z60_8;  break;
    case 64:  bgd = &BG1_adj_desc_Z64_8;  break;
    case 72:  bgd = &BG1_adj_desc_Z72_8;  break;
    case 80:  bgd = &BG1_adj_desc_Z80_8;  break;
    case 88:  bgd = &BG1_adj_desc_Z88_8;  break;
    case 96:  bgd = &BG1_adj_desc_Z96_8;  break;
    case 104: bgd = &BG1_adj_desc_Z104_8; break;
    case 112: bgd = &BG1_adj_desc_Z112_8; break;
    case 120: bgd = &BG1_adj_desc_Z120_8; break;
    case 128: bgd = &BG1_adj_desc_Z128_8; break;
    case 144: bgd = &BG1_adj_desc_Z144_8; break;
    case 160: bgd = &BG1_adj_desc_Z160_8; break;
    case 176: bgd = &BG1_adj_desc_Z176_8; break;
    case 192: bgd = &BG1_adj_desc_Z192_8; break;
    case 208: bgd = &BG1_adj_desc_Z208_8; break;
    case 224: bgd = &BG1_adj_desc_Z224_8; break;
    case 240: bgd = &BG1_adj_desc_Z240_8; break;
    case 256: bgd = &BG1_adj_desc_Z256_8; break;
    case 288: bgd = &BG1_adj_desc_Z288_8; break;
    case 320: bgd = &BG1_adj_desc_Z320_8; break;
    case 352: bgd = &BG1_adj_desc_Z352_8; break;
    case 384: bgd = &BG1_adj_desc_Z384_8; break;
    }
    return bgd;
}

template <>
inline
const BG_adj_desc<1>* get_adj_BG_desc_bitwidth<16, 1>(int Z)
{
    const BG1_adj_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG1_adj_desc_Z32_16;  break;
    case 36:  bgd = &BG1_adj_desc_Z36_16;  break;
    case 40:  bgd = &BG1_adj_desc_Z40_16;  break;
    case 44:  bgd = &BG1_adj_desc_Z44_16;  break;
    case 48:  bgd = &BG1_adj_desc_Z48_16;  break;
    case 52:  bgd = &BG1_adj_desc_Z52_16;  break;
    case 56:  bgd = &BG1_adj_desc_Z56_16;  break;
    case 60:  bgd = &BG1_adj_desc_Z60_16;  break;
    case 64:  bgd = &BG1_adj_desc_Z64_16;  break;
    case 72:  bgd = &BG1_adj_desc_Z72_16;  break;
    case 80:  bgd = &BG1_adj_desc_Z80_16;  break;
    case 88:  bgd = &BG1_adj_desc_Z88_16;  break;
    case 96:  bgd = &BG1_adj_desc_Z96_16;  break;
    case 104: bgd = &BG1_adj_desc_Z104_16; break;
    case 112: bgd = &BG1_adj_desc_Z112_16; break;
    case 120: bgd = &BG1_adj_desc_Z120_16; break;
    case 128: bgd = &BG1_adj_desc_Z128_16; break;
    case 144: bgd = &BG1_adj_desc_Z144_16; break;
    case 160: bgd = &BG1_adj_desc_Z160_16; break;
    case 176: bgd = &BG1_adj_desc_Z176_16; break;
    case 192: bgd = &BG1_adj_desc_Z192_16; break;
    case 208: bgd = &BG1_adj_desc_Z208_16; break;
    case 224: bgd = &BG1_adj_desc_Z224_16; break;
    case 240: bgd = &BG1_adj_desc_Z240_16; break;
    case 256: bgd = &BG1_adj_desc_Z256_16; break;
    case 288: bgd = &BG1_adj_desc_Z288_16; break;
    case 320: bgd = &BG1_adj_desc_Z320_16; break;
    case 352: bgd = &BG1_adj_desc_Z352_16; break;
    case 384: bgd = &BG1_adj_desc_Z384_16; break;
    }
    return bgd;
}

template <>
inline
const BG_adj_desc<2>* get_adj_BG_desc_bitwidth<8, 2>(int Z)
{
    const BG2_adj_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG2_adj_desc_Z32_8;  break;
    case 36:  bgd = &BG2_adj_desc_Z36_8;  break;
    case 40:  bgd = &BG2_adj_desc_Z40_8;  break;
    case 44:  bgd = &BG2_adj_desc_Z44_8;  break;
    case 48:  bgd = &BG2_adj_desc_Z48_8;  break;
    case 52:  bgd = &BG2_adj_desc_Z52_8;  break;
    case 56:  bgd = &BG2_adj_desc_Z56_8;  break;
    case 60:  bgd = &BG2_adj_desc_Z60_8;  break;
    case 64:  bgd = &BG2_adj_desc_Z64_8;  break;
    case 72:  bgd = &BG2_adj_desc_Z72_8;  break;
    case 80:  bgd = &BG2_adj_desc_Z80_8;  break;
    case 88:  bgd = &BG2_adj_desc_Z88_8;  break;
    case 96:  bgd = &BG2_adj_desc_Z96_8;  break;
    case 104: bgd = &BG2_adj_desc_Z104_8; break;
    case 112: bgd = &BG2_adj_desc_Z112_8; break;
    case 120: bgd = &BG2_adj_desc_Z120_8; break;
    case 128: bgd = &BG2_adj_desc_Z128_8; break;
    case 144: bgd = &BG2_adj_desc_Z144_8; break;
    case 160: bgd = &BG2_adj_desc_Z160_8; break;
    case 176: bgd = &BG2_adj_desc_Z176_8; break;
    case 192: bgd = &BG2_adj_desc_Z192_8; break;
    case 208: bgd = &BG2_adj_desc_Z208_8; break;
    case 224: bgd = &BG2_adj_desc_Z224_8; break;
    case 240: bgd = &BG2_adj_desc_Z240_8; break;
    case 256: bgd = &BG2_adj_desc_Z256_8; break;
    case 288: bgd = &BG2_adj_desc_Z288_8; break;
    case 320: bgd = &BG2_adj_desc_Z320_8; break;
    case 352: bgd = &BG2_adj_desc_Z352_8; break;
    case 384: bgd = &BG2_adj_desc_Z384_8; break;
    }
    return bgd;
}

template <>
inline
const BG_adj_desc<2>* get_adj_BG_desc_bitwidth<16, 2>(int Z)
{
    const BG2_adj_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG2_adj_desc_Z32_16;  break;
    case 36:  bgd = &BG2_adj_desc_Z36_16;  break;
    case 40:  bgd = &BG2_adj_desc_Z40_16;  break;
    case 44:  bgd = &BG2_adj_desc_Z44_16;  break;
    case 48:  bgd = &BG2_adj_desc_Z48_16;  break;
    case 52:  bgd = &BG2_adj_desc_Z52_16;  break;
    case 56:  bgd = &BG2_adj_desc_Z56_16;  break;
    case 60:  bgd = &BG2_adj_desc_Z60_16;  break;
    case 64:  bgd = &BG2_adj_desc_Z64_16;  break;
    case 72:  bgd = &BG2_adj_desc_Z72_16;  break;
    case 80:  bgd = &BG2_adj_desc_Z80_16;  break;
    case 88:  bgd = &BG2_adj_desc_Z88_16;  break;
    case 96:  bgd = &BG2_adj_desc_Z96_16;  break;
    case 104: bgd = &BG2_adj_desc_Z104_16; break;
    case 112: bgd = &BG2_adj_desc_Z112_16; break;
    case 120: bgd = &BG2_adj_desc_Z120_16; break;
    case 128: bgd = &BG2_adj_desc_Z128_16; break;
    case 144: bgd = &BG2_adj_desc_Z144_16; break;
    case 160: bgd = &BG2_adj_desc_Z160_16; break;
    case 176: bgd = &BG2_adj_desc_Z176_16; break;
    case 192: bgd = &BG2_adj_desc_Z192_16; break;
    case 208: bgd = &BG2_adj_desc_Z208_16; break;
    case 224: bgd = &BG2_adj_desc_Z224_16; break;
    case 240: bgd = &BG2_adj_desc_Z240_16; break;
    case 256: bgd = &BG2_adj_desc_Z256_16; break;
    case 288: bgd = &BG2_adj_desc_Z288_16; break;
    case 320: bgd = &BG2_adj_desc_Z320_16; break;
    case 352: bgd = &BG2_adj_desc_Z352_16; break;
    case 384: bgd = &BG2_adj_desc_Z384_16; break;
    }
    return bgd;
}

template <>
inline
const BG_adj_desc<1>* get_adj_BG_desc_bitwidth<32, 1>(int Z)
{
    const BG1_adj_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG1_adj_desc_Z32_32;  break;
    case 36:  bgd = &BG1_adj_desc_Z36_32;  break;
    case 40:  bgd = &BG1_adj_desc_Z40_32;  break;
    case 44:  bgd = &BG1_adj_desc_Z44_32;  break;
    case 48:  bgd = &BG1_adj_desc_Z48_32;  break;
    case 52:  bgd = &BG1_adj_desc_Z52_32;  break;
    case 56:  bgd = &BG1_adj_desc_Z56_32;  break;
    case 60:  bgd = &BG1_adj_desc_Z60_32;  break;
    case 64:  bgd = &BG1_adj_desc_Z64_32;  break;
    case 72:  bgd = &BG1_adj_desc_Z72_32;  break;
    case 80:  bgd = &BG1_adj_desc_Z80_32;  break;
    case 88:  bgd = &BG1_adj_desc_Z88_32;  break;
    case 96:  bgd = &BG1_adj_desc_Z96_32;  break;
    case 104: bgd = &BG1_adj_desc_Z104_32; break;
    case 112: bgd = &BG1_adj_desc_Z112_32; break;
    case 120: bgd = &BG1_adj_desc_Z120_32; break;
    case 128: bgd = &BG1_adj_desc_Z128_32; break;
    case 144: bgd = &BG1_adj_desc_Z144_32; break;
    case 160: bgd = &BG1_adj_desc_Z160_32; break;
    case 176: bgd = &BG1_adj_desc_Z176_32; break;
    case 192: bgd = &BG1_adj_desc_Z192_32; break;
    case 208: bgd = &BG1_adj_desc_Z208_32; break;
    case 224: bgd = &BG1_adj_desc_Z224_32; break;
    case 240: bgd = &BG1_adj_desc_Z240_32; break;
    case 256: bgd = &BG1_adj_desc_Z256_32; break;
    case 288: bgd = &BG1_adj_desc_Z288_32; break;
    case 320: bgd = &BG1_adj_desc_Z320_32; break;
    case 352: bgd = &BG1_adj_desc_Z352_32; break;
    case 384: bgd = &BG1_adj_desc_Z384_32; break;
    }
    return bgd;
}

template <>
inline
const BG_adj_desc<2>* get_adj_BG_desc_bitwidth<32, 2>(int Z)
{
    const BG2_adj_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG2_adj_desc_Z32_32;  break;
    case 36:  bgd = &BG2_adj_desc_Z36_32;  break;
    case 40:  bgd = &BG2_adj_desc_Z40_32;  break;
    case 44:  bgd = &BG2_adj_desc_Z44_32;  break;
    case 48:  bgd = &BG2_adj_desc_Z48_32;  break;
    case 52:  bgd = &BG2_adj_desc_Z52_32;  break;
    case 56:  bgd = &BG2_adj_desc_Z56_32;  break;
    case 60:  bgd = &BG2_adj_desc_Z60_32;  break;
    case 64:  bgd = &BG2_adj_desc_Z64_32;  break;
    case 72:  bgd = &BG2_adj_desc_Z72_32;  break;
    case 80:  bgd = &BG2_adj_desc_Z80_32;  break;
    case 88:  bgd = &BG2_adj_desc_Z88_32;  break;
    case 96:  bgd = &BG2_adj_desc_Z96_32;  break;
    case 104: bgd = &BG2_adj_desc_Z104_32; break;
    case 112: bgd = &BG2_adj_desc_Z112_32; break;
    case 120: bgd = &BG2_adj_desc_Z120_32; break;
    case 128: bgd = &BG2_adj_desc_Z128_32; break;
    case 144: bgd = &BG2_adj_desc_Z144_32; break;
    case 160: bgd = &BG2_adj_desc_Z160_32; break;
    case 176: bgd = &BG2_adj_desc_Z176_32; break;
    case 192: bgd = &BG2_adj_desc_Z192_32; break;
    case 208: bgd = &BG2_adj_desc_Z208_32; break;
    case 224: bgd = &BG2_adj_desc_Z224_32; break;
    case 240: bgd = &BG2_adj_desc_Z240_32; break;
    case 256: bgd = &BG2_adj_desc_Z256_32; break;
    case 288: bgd = &BG2_adj_desc_Z288_32; break;
    case 320: bgd = &BG2_adj_desc_Z320_32; break;
    case 352: bgd = &BG2_adj_desc_Z352_32; break;
    case 384: bgd = &BG2_adj_desc_Z384_32; break;
    }
    return bgd;
}


template <typename T,
          int      BG>
const BG_adj_desc<BG>* get_adj_BG_desc(int Z)
{
    return get_adj_BG_desc_bitwidth<sizeof(T) * CHAR_BIT, BG>(Z);
}

template <typename T, int BG> const BG_adj_desc<BG>* get_adj_BG_desc_small(int Z);

template <>
inline
const BG_adj_desc<1>* get_adj_BG_desc_small<__half, 1>(int Z)
{
    const BG1_adj_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 2:   bgd = &BG1_adj_desc_Z2_16;   break;
    case 3:   bgd = &BG1_adj_desc_Z3_16;   break;
    case 4:   bgd = &BG1_adj_desc_Z4_16;   break;
    case 5:   bgd = &BG1_adj_desc_Z5_16;   break;
    case 6:   bgd = &BG1_adj_desc_Z6_16;   break;
    case 7:   bgd = &BG1_adj_desc_Z7_16;   break;
    case 8:   bgd = &BG1_adj_desc_Z8_16;   break;
    case 9:   bgd = &BG1_adj_desc_Z9_16;   break;
    case 10:  bgd = &BG1_adj_desc_Z10_16;  break;
    case 11:  bgd = &BG1_adj_desc_Z11_16;  break;
    case 12:  bgd = &BG1_adj_desc_Z12_16;  break;
    case 13:  bgd = &BG1_adj_desc_Z13_16;  break;
    case 14:  bgd = &BG1_adj_desc_Z14_16;  break;
    case 15:  bgd = &BG1_adj_desc_Z15_16;  break;
    case 16:  bgd = &BG1_adj_desc_Z16_16;  break;
    case 18:  bgd = &BG1_adj_desc_Z18_16;  break;
    case 20:  bgd = &BG1_adj_desc_Z20_16;  break;
    case 22:  bgd = &BG1_adj_desc_Z22_16;  break;
    case 24:  bgd = &BG1_adj_desc_Z24_16;  break;
    case 26:  bgd = &BG1_adj_desc_Z26_16;  break;
    case 28:  bgd = &BG1_adj_desc_Z28_16;  break;
    case 30:  bgd = &BG1_adj_desc_Z30_16;  break;
    }
    return bgd;
}

template <>
inline
const BG_adj_desc<2>* get_adj_BG_desc_small<__half, 2>(int Z)
{
    const BG2_adj_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 2:   bgd = &BG2_adj_desc_Z2_16;   break;
    case 3:   bgd = &BG2_adj_desc_Z3_16;   break;
    case 4:   bgd = &BG2_adj_desc_Z4_16;   break;
    case 5:   bgd = &BG2_adj_desc_Z5_16;   break;
    case 6:   bgd = &BG2_adj_desc_Z6_16;   break;
    case 7:   bgd = &BG2_adj_desc_Z7_16;   break;
    case 8:   bgd = &BG2_adj_desc_Z8_16;   break;
    case 9:   bgd = &BG2_adj_desc_Z9_16;   break;
    case 10:  bgd = &BG2_adj_desc_Z10_16;  break;
    case 11:  bgd = &BG2_adj_desc_Z11_16;  break;
    case 12:  bgd = &BG2_adj_desc_Z12_16;  break;
    case 13:  bgd = &BG2_adj_desc_Z13_16;  break;
    case 14:  bgd = &BG2_adj_desc_Z14_16;  break;
    case 15:  bgd = &BG2_adj_desc_Z15_16;  break;
    case 16:  bgd = &BG2_adj_desc_Z16_16;  break;
    case 18:  bgd = &BG2_adj_desc_Z18_16;  break;
    case 20:  bgd = &BG2_adj_desc_Z20_16;  break;
    case 22:  bgd = &BG2_adj_desc_Z22_16;  break;
    case 24:  bgd = &BG2_adj_desc_Z24_16;  break;
    case 26:  bgd = &BG2_adj_desc_Z26_16;  break;
    case 28:  bgd = &BG2_adj_desc_Z28_16;  break;
    case 30:  bgd = &BG2_adj_desc_Z30_16;  break;
    }
    return bgd;
}

template <int BG> struct BG_nzs_desc;
template <> struct BG_nzs_desc<1>
{
    LDPC_node_desc nodes[BG1_NZS_ADDR_PAIR_COUNT];
};
template <> struct BG_nzs_desc<2>
{
    LDPC_node_desc nodes[BG2_NZS_ADDR_PAIR_COUNT];
};

typedef BG_nzs_desc<1> BG1_nzs_desc_t;
typedef BG_nzs_desc<2> BG2_nzs_desc_t;

// Host descriptor declarations
extern const BG1_nzs_desc_t BG1_nzs_desc_Z32_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z36_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z40_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z44_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z48_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z52_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z56_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z60_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z64_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z72_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z80_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z88_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z96_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z104_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z112_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z120_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z128_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z144_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z160_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z176_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z192_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z208_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z224_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z240_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z256_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z288_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z320_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z352_16;
extern const BG1_nzs_desc_t BG1_nzs_desc_Z384_16;

extern const BG2_nzs_desc_t BG2_nzs_desc_Z32_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z36_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z40_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z44_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z48_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z52_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z56_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z60_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z64_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z72_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z80_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z88_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z96_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z104_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z112_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z120_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z128_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z144_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z160_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z176_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z192_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z208_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z224_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z240_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z256_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z288_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z320_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z352_16;
extern const BG2_nzs_desc_t BG2_nzs_desc_Z384_16;

template <int TBitwidth,
          int BG>
const BG_nzs_desc<BG>* get_BG_nzs_desc_bitwidth(int Z);

template <>
inline
const BG_nzs_desc<1>* get_BG_nzs_desc_bitwidth<16, 1>(int Z)
{
    const BG1_nzs_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG1_nzs_desc_Z32_16;  break;
    case 36:  bgd = &BG1_nzs_desc_Z36_16;  break;
    case 40:  bgd = &BG1_nzs_desc_Z40_16;  break;
    case 44:  bgd = &BG1_nzs_desc_Z44_16;  break;
    case 48:  bgd = &BG1_nzs_desc_Z48_16;  break;
    case 52:  bgd = &BG1_nzs_desc_Z52_16;  break;
    case 56:  bgd = &BG1_nzs_desc_Z56_16;  break;
    case 60:  bgd = &BG1_nzs_desc_Z60_16;  break;
    case 64:  bgd = &BG1_nzs_desc_Z64_16;  break;
    case 72:  bgd = &BG1_nzs_desc_Z72_16;  break;
    case 80:  bgd = &BG1_nzs_desc_Z80_16;  break;
    case 88:  bgd = &BG1_nzs_desc_Z88_16;  break;
    case 96:  bgd = &BG1_nzs_desc_Z96_16;  break;
    case 104: bgd = &BG1_nzs_desc_Z104_16; break;
    case 112: bgd = &BG1_nzs_desc_Z112_16; break;
    case 120: bgd = &BG1_nzs_desc_Z120_16; break;
    case 128: bgd = &BG1_nzs_desc_Z128_16; break;
    case 144: bgd = &BG1_nzs_desc_Z144_16; break;
    case 160: bgd = &BG1_nzs_desc_Z160_16; break;
    case 176: bgd = &BG1_nzs_desc_Z176_16; break;
    case 192: bgd = &BG1_nzs_desc_Z192_16; break;
    case 208: bgd = &BG1_nzs_desc_Z208_16; break;
    case 224: bgd = &BG1_nzs_desc_Z224_16; break;
    case 240: bgd = &BG1_nzs_desc_Z240_16; break;
    case 256: bgd = &BG1_nzs_desc_Z256_16; break;
    case 288: bgd = &BG1_nzs_desc_Z288_16; break;
    case 320: bgd = &BG1_nzs_desc_Z320_16; break;
    case 352: bgd = &BG1_nzs_desc_Z352_16; break;
    case 384: bgd = &BG1_nzs_desc_Z384_16; break;
    }
    return bgd;
}

template <>
inline
const BG_nzs_desc<2>* get_BG_nzs_desc_bitwidth<16, 2>(int Z)
{
    const BG2_nzs_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG2_nzs_desc_Z32_16;  break;
    case 36:  bgd = &BG2_nzs_desc_Z36_16;  break;
    case 40:  bgd = &BG2_nzs_desc_Z40_16;  break;
    case 44:  bgd = &BG2_nzs_desc_Z44_16;  break;
    case 48:  bgd = &BG2_nzs_desc_Z48_16;  break;
    case 52:  bgd = &BG2_nzs_desc_Z52_16;  break;
    case 56:  bgd = &BG2_nzs_desc_Z56_16;  break;
    case 60:  bgd = &BG2_nzs_desc_Z60_16;  break;
    case 64:  bgd = &BG2_nzs_desc_Z64_16;  break;
    case 72:  bgd = &BG2_nzs_desc_Z72_16;  break;
    case 80:  bgd = &BG2_nzs_desc_Z80_16;  break;
    case 88:  bgd = &BG2_nzs_desc_Z88_16;  break;
    case 96:  bgd = &BG2_nzs_desc_Z96_16;  break;
    case 104: bgd = &BG2_nzs_desc_Z104_16; break;
    case 112: bgd = &BG2_nzs_desc_Z112_16; break;
    case 120: bgd = &BG2_nzs_desc_Z120_16; break;
    case 128: bgd = &BG2_nzs_desc_Z128_16; break;
    case 144: bgd = &BG2_nzs_desc_Z144_16; break;
    case 160: bgd = &BG2_nzs_desc_Z160_16; break;
    case 176: bgd = &BG2_nzs_desc_Z176_16; break;
    case 192: bgd = &BG2_nzs_desc_Z192_16; break;
    case 208: bgd = &BG2_nzs_desc_Z208_16; break;
    case 224: bgd = &BG2_nzs_desc_Z224_16; break;
    case 240: bgd = &BG2_nzs_desc_Z240_16; break;
    case 256: bgd = &BG2_nzs_desc_Z256_16; break;
    case 288: bgd = &BG2_nzs_desc_Z288_16; break;
    case 320: bgd = &BG2_nzs_desc_Z320_16; break;
    case 352: bgd = &BG2_nzs_desc_Z352_16; break;
    case 384: bgd = &BG2_nzs_desc_Z384_16; break;
    }
    return bgd;
}

template <typename T,
          int      BG>
const BG_nzs_desc<BG>* get_BG_nzs_desc(int Z)
{
    return get_BG_nzs_desc_bitwidth<sizeof(T) * CHAR_BIT, BG>(Z);
}

template <int BG> struct BG_adj_nzs_desc;
template <> struct BG_adj_nzs_desc<1>
{
    LDPC_adj_node_desc nodes[BG1_NZS_ADDR_PAIR_COUNT];
};
template <> struct BG_adj_nzs_desc<2>
{
    LDPC_adj_node_desc nodes[BG2_NZS_ADDR_PAIR_COUNT];
};

typedef BG_adj_nzs_desc<1> BG1_adj_nzs_desc_t;
typedef BG_adj_nzs_desc<2> BG2_adj_nzs_desc_t;

// Host descriptor declarations
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z32_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z36_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z40_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z44_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z48_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z52_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z56_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z60_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z64_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z72_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z80_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z88_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z96_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z104_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z112_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z120_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z128_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z144_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z160_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z176_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z192_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z208_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z224_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z240_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z256_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z288_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z320_16;
//extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z352_16;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z384_16;

//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z32_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z36_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z40_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z44_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z48_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z52_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z56_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z60_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z64_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z72_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z80_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z88_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z96_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z104_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z112_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z120_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z128_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z144_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z160_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z176_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z192_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z208_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z224_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z240_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z256_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z288_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z320_16;
//extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z352_16;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z384_16;

extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z32_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z36_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z40_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z44_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z48_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z52_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z56_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z60_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z64_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z72_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z80_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z88_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z96_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z104_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z112_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z120_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z128_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z144_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z160_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z176_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z192_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z208_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z224_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z240_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z256_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z288_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z320_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z352_32;
extern const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z384_32;

extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z32_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z36_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z40_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z44_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z48_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z52_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z56_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z60_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z64_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z72_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z80_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z88_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z96_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z104_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z112_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z120_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z128_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z144_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z160_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z176_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z192_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z208_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z224_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z240_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z256_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z288_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z320_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z352_32;
extern const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z384_32;

template <int TBitwidth,
          int BG>
const BG_adj_nzs_desc<BG>* get_BG_adj_nzs_desc_bitwidth(int Z);

template <>
inline
const BG_adj_nzs_desc<1>* get_BG_adj_nzs_desc_bitwidth<16, 1>(int Z)
{
    const BG1_adj_nzs_desc_t* bgd = nullptr;
    switch(Z)
    {
    //case 32:  bgd = &BG1_adj_nzs_desc_Z32_16;  break;
    //case 36:  bgd = &BG1_adj_nzs_desc_Z36_16;  break;
    //case 40:  bgd = &BG1_adj_nzs_desc_Z40_16;  break;
    //case 44:  bgd = &BG1_adj_nzs_desc_Z44_16;  break;
    //case 48:  bgd = &BG1_adj_nzs_desc_Z48_16;  break;
    //case 52:  bgd = &BG1_adj_nzs_desc_Z52_16;  break;
    //case 56:  bgd = &BG1_adj_nzs_desc_Z56_16;  break;
    //case 60:  bgd = &BG1_adj_nzs_desc_Z60_16;  break;
    //case 64:  bgd = &BG1_adj_nzs_desc_Z64_16;  break;
    //case 72:  bgd = &BG1_adj_nzs_desc_Z72_16;  break;
    //case 80:  bgd = &BG1_adj_nzs_desc_Z80_16;  break;
    //case 88:  bgd = &BG1_adj_nzs_desc_Z88_16;  break;
    //case 96:  bgd = &BG1_adj_nzs_desc_Z96_16;  break;
    //case 104: bgd = &BG1_adj_nzs_desc_Z104_16; break;
    //case 112: bgd = &BG1_adj_nzs_desc_Z112_16; break;
    //case 120: bgd = &BG1_adj_nzs_desc_Z120_16; break;
    //case 128: bgd = &BG1_adj_nzs_desc_Z128_16; break;
    //case 144: bgd = &BG1_adj_nzs_desc_Z144_16; break;
    //case 160: bgd = &BG1_adj_nzs_desc_Z160_16; break;
    //case 176: bgd = &BG1_adj_nzs_desc_Z176_16; break;
    //case 192: bgd = &BG1_adj_nzs_desc_Z192_16; break;
    //case 208: bgd = &BG1_adj_nzs_desc_Z208_16; break;
    //case 224: bgd = &BG1_adj_nzs_desc_Z224_16; break;
    //case 240: bgd = &BG1_adj_nzs_desc_Z240_16; break;
    //case 256: bgd = &BG1_adj_nzs_desc_Z256_16; break;
    //case 288: bgd = &BG1_adj_nzs_desc_Z288_16; break;
    //case 320: bgd = &BG1_adj_nzs_desc_Z320_16; break;
    //case 352: bgd = &BG1_adj_nzs_desc_Z352_16; break;
    case 384: bgd = &BG1_adj_nzs_desc_Z384_16; break;
    }
    return bgd;
}

template <>
inline
const BG_adj_nzs_desc<2>* get_BG_adj_nzs_desc_bitwidth<16, 2>(int Z)
{
    const BG2_adj_nzs_desc_t* bgd = nullptr;
    switch(Z)
    {
    //case 32:  bgd = &BG2_adj_nzs_desc_Z32_16;  break;
    //case 36:  bgd = &BG2_adj_nzs_desc_Z36_16;  break;
    //case 40:  bgd = &BG2_adj_nzs_desc_Z40_16;  break;
    //case 44:  bgd = &BG2_adj_nzs_desc_Z44_16;  break;
    //case 48:  bgd = &BG2_adj_nzs_desc_Z48_16;  break;
    //case 52:  bgd = &BG2_adj_nzs_desc_Z52_16;  break;
    //case 56:  bgd = &BG2_adj_nzs_desc_Z56_16;  break;
    //case 60:  bgd = &BG2_adj_nzs_desc_Z60_16;  break;
    //case 64:  bgd = &BG2_adj_nzs_desc_Z64_16;  break;
    //case 72:  bgd = &BG2_adj_nzs_desc_Z72_16;  break;
    //case 80:  bgd = &BG2_adj_nzs_desc_Z80_16;  break;
    //case 88:  bgd = &BG2_adj_nzs_desc_Z88_16;  break;
    //case 96:  bgd = &BG2_adj_nzs_desc_Z96_16;  break;
    //case 104: bgd = &BG2_adj_nzs_desc_Z104_16; break;
    //case 112: bgd = &BG2_adj_nzs_desc_Z112_16; break;
    //case 120: bgd = &BG2_adj_nzs_desc_Z120_16; break;
    //case 128: bgd = &BG2_adj_nzs_desc_Z128_16; break;
    //case 144: bgd = &BG2_adj_nzs_desc_Z144_16; break;
    //case 160: bgd = &BG2_adj_nzs_desc_Z160_16; break;
    //case 176: bgd = &BG2_adj_nzs_desc_Z176_16; break;
    //case 192: bgd = &BG2_adj_nzs_desc_Z192_16; break;
    //case 208: bgd = &BG2_adj_nzs_desc_Z208_16; break;
    //case 224: bgd = &BG2_adj_nzs_desc_Z224_16; break;
    //case 240: bgd = &BG2_adj_nzs_desc_Z240_16; break;
    //case 256: bgd = &BG2_adj_nzs_desc_Z256_16; break;
    //case 288: bgd = &BG2_adj_nzs_desc_Z288_16; break;
    //case 320: bgd = &BG2_adj_nzs_desc_Z320_16; break;
    //case 352: bgd = &BG2_adj_nzs_desc_Z352_16; break;
    case 384: bgd = &BG2_adj_nzs_desc_Z384_16; break;
    }
    return bgd;
}

template <>
inline
const BG_adj_nzs_desc<1>* get_BG_adj_nzs_desc_bitwidth<32, 1>(int Z)
{
    const BG1_adj_nzs_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG1_adj_nzs_desc_Z32_32;  break;
    case 36:  bgd = &BG1_adj_nzs_desc_Z36_32;  break;
    case 40:  bgd = &BG1_adj_nzs_desc_Z40_32;  break;
    case 44:  bgd = &BG1_adj_nzs_desc_Z44_32;  break;
    case 48:  bgd = &BG1_adj_nzs_desc_Z48_32;  break;
    case 52:  bgd = &BG1_adj_nzs_desc_Z52_32;  break;
    case 56:  bgd = &BG1_adj_nzs_desc_Z56_32;  break;
    case 60:  bgd = &BG1_adj_nzs_desc_Z60_32;  break;
    case 64:  bgd = &BG1_adj_nzs_desc_Z64_32;  break;
    case 72:  bgd = &BG1_adj_nzs_desc_Z72_32;  break;
    case 80:  bgd = &BG1_adj_nzs_desc_Z80_32;  break;
    case 88:  bgd = &BG1_adj_nzs_desc_Z88_32;  break;
    case 96:  bgd = &BG1_adj_nzs_desc_Z96_32;  break;
    case 104: bgd = &BG1_adj_nzs_desc_Z104_32; break;
    case 112: bgd = &BG1_adj_nzs_desc_Z112_32; break;
    case 120: bgd = &BG1_adj_nzs_desc_Z120_32; break;
    case 128: bgd = &BG1_adj_nzs_desc_Z128_32; break;
    case 144: bgd = &BG1_adj_nzs_desc_Z144_32; break;
    case 160: bgd = &BG1_adj_nzs_desc_Z160_32; break;
    case 176: bgd = &BG1_adj_nzs_desc_Z176_32; break;
    case 192: bgd = &BG1_adj_nzs_desc_Z192_32; break;
    case 208: bgd = &BG1_adj_nzs_desc_Z208_32; break;
    case 224: bgd = &BG1_adj_nzs_desc_Z224_32; break;
    case 240: bgd = &BG1_adj_nzs_desc_Z240_32; break;
    case 256: bgd = &BG1_adj_nzs_desc_Z256_32; break;
    case 288: bgd = &BG1_adj_nzs_desc_Z288_32; break;
    case 320: bgd = &BG1_adj_nzs_desc_Z320_32; break;
    case 352: bgd = &BG1_adj_nzs_desc_Z352_32; break;
    case 384: bgd = &BG1_adj_nzs_desc_Z384_32; break;
    }
    return bgd;
}

template <>
inline
const BG_adj_nzs_desc<2>* get_BG_adj_nzs_desc_bitwidth<32, 2>(int Z)
{
    const BG2_adj_nzs_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG2_adj_nzs_desc_Z32_32;  break;
    case 36:  bgd = &BG2_adj_nzs_desc_Z36_32;  break;
    case 40:  bgd = &BG2_adj_nzs_desc_Z40_32;  break;
    case 44:  bgd = &BG2_adj_nzs_desc_Z44_32;  break;
    case 48:  bgd = &BG2_adj_nzs_desc_Z48_32;  break;
    case 52:  bgd = &BG2_adj_nzs_desc_Z52_32;  break;
    case 56:  bgd = &BG2_adj_nzs_desc_Z56_32;  break;
    case 60:  bgd = &BG2_adj_nzs_desc_Z60_32;  break;
    case 64:  bgd = &BG2_adj_nzs_desc_Z64_32;  break;
    case 72:  bgd = &BG2_adj_nzs_desc_Z72_32;  break;
    case 80:  bgd = &BG2_adj_nzs_desc_Z80_32;  break;
    case 88:  bgd = &BG2_adj_nzs_desc_Z88_32;  break;
    case 96:  bgd = &BG2_adj_nzs_desc_Z96_32;  break;
    case 104: bgd = &BG2_adj_nzs_desc_Z104_32; break;
    case 112: bgd = &BG2_adj_nzs_desc_Z112_32; break;
    case 120: bgd = &BG2_adj_nzs_desc_Z120_32; break;
    case 128: bgd = &BG2_adj_nzs_desc_Z128_32; break;
    case 144: bgd = &BG2_adj_nzs_desc_Z144_32; break;
    case 160: bgd = &BG2_adj_nzs_desc_Z160_32; break;
    case 176: bgd = &BG2_adj_nzs_desc_Z176_32; break;
    case 192: bgd = &BG2_adj_nzs_desc_Z192_32; break;
    case 208: bgd = &BG2_adj_nzs_desc_Z208_32; break;
    case 224: bgd = &BG2_adj_nzs_desc_Z224_32; break;
    case 240: bgd = &BG2_adj_nzs_desc_Z240_32; break;
    case 256: bgd = &BG2_adj_nzs_desc_Z256_32; break;
    case 288: bgd = &BG2_adj_nzs_desc_Z288_32; break;
    case 320: bgd = &BG2_adj_nzs_desc_Z320_32; break;
    case 352: bgd = &BG2_adj_nzs_desc_Z352_32; break;
    case 384: bgd = &BG2_adj_nzs_desc_Z384_32; break;
    }
    return bgd;
}

template <typename T,
          int      BG>
const BG_adj_nzs_desc<BG>* get_BG_adj_nzs_desc(int Z)
{
    return get_BG_adj_nzs_desc_bitwidth<sizeof(T) * CHAR_BIT, BG>(Z);
}

// "Null" descriptor for address calculators that don't use a kernel descriptor
// argument to perform calculations. (Some approaches might, for example, have
// inline constants compiled directly into a specialized kernel.)
template <int BG>
struct null_BG_desc_t
{
};

} // namespace ldpc2


#endif // !defined(LDPC2_BG_DESC_HPP_INCLUDED_)
