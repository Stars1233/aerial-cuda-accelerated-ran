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

#include "ldpc2_bg_desc.hpp"

namespace ldpc2
{

const BG1_adj_nzs_desc_t BG1_adj_nzs_desc_Z384_16 =
{
    {
        { wrap_index_pair<1, 384,  0, 0>::value, vnode_adj_shift_offset<1, 384,  0, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384,  0, 1>::value * 2 }, // Row 0, degree = 19, nzs_row_degree = 18
        { wrap_index_pair<1, 384,  0, 1>::value, vnode_adj_shift_offset<1, 384,  0, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384,  0, 3>::value * 2 },
        { wrap_index_pair<1, 384,  0, 2>::value, vnode_adj_shift_offset<1, 384,  0, 4>::value * 2,  vnode_adj_shift_offset_if<1, 384,  0, 5>::value * 2 },
        { wrap_index_pair<1, 384,  0, 3>::value, vnode_adj_shift_offset<1, 384,  0, 6>::value * 2,  vnode_adj_shift_offset_if<1, 384,  0, 7>::value * 2 },
        { wrap_index_pair<1, 384,  0, 4>::value, vnode_adj_shift_offset<1, 384,  0, 8>::value * 2,  vnode_adj_shift_offset_if<1, 384,  0, 9>::value * 2 },
        { wrap_index_pair<1, 384,  0, 5>::value, vnode_adj_shift_offset<1, 384,  0, 10>::value * 2,  vnode_adj_shift_offset_if<1, 384,  0, 11>::value * 2 },
        { wrap_index_pair<1, 384,  0, 6>::value, vnode_adj_shift_offset<1, 384,  0, 12>::value * 2,  vnode_adj_shift_offset_if<1, 384,  0, 13>::value * 2 },
        { wrap_index_pair<1, 384,  0, 7>::value, vnode_adj_shift_offset<1, 384,  0, 14>::value * 2,  vnode_adj_shift_offset_if<1, 384,  0, 15>::value * 2 },
        { wrap_index_pair<1, 384,  0, 8>::value, vnode_adj_shift_offset<1, 384,  0, 16>::value * 2,  vnode_adj_shift_offset_if<1, 384,  0, 17>::value * 2 },

        { wrap_index_pair<1, 384,  1, 0>::value, vnode_adj_shift_offset<1, 384,  1, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384,  1, 1>::value * 2 }, // Row 1, degree = 19, nzs_row_degree = 17
        { wrap_index_pair<1, 384,  1, 1>::value, vnode_adj_shift_offset<1, 384,  1, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384,  1, 3>::value * 2 },
        { wrap_index_pair<1, 384,  1, 2>::value, vnode_adj_shift_offset<1, 384,  1, 4>::value * 2,  vnode_adj_shift_offset_if<1, 384,  1, 5>::value * 2 },
        { wrap_index_pair<1, 384,  1, 3>::value, vnode_adj_shift_offset<1, 384,  1, 6>::value * 2,  vnode_adj_shift_offset_if<1, 384,  1, 7>::value * 2 },
        { wrap_index_pair<1, 384,  1, 4>::value, vnode_adj_shift_offset<1, 384,  1, 8>::value * 2,  vnode_adj_shift_offset_if<1, 384,  1, 9>::value * 2 },
        { wrap_index_pair<1, 384,  1, 5>::value, vnode_adj_shift_offset<1, 384,  1, 10>::value * 2,  vnode_adj_shift_offset_if<1, 384,  1, 11>::value * 2 },
        { wrap_index_pair<1, 384,  1, 6>::value, vnode_adj_shift_offset<1, 384,  1, 12>::value * 2,  vnode_adj_shift_offset_if<1, 384,  1, 13>::value * 2 },
        { wrap_index_pair<1, 384,  1, 7>::value, vnode_adj_shift_offset<1, 384,  1, 14>::value * 2,  vnode_adj_shift_offset_if<1, 384,  1, 15>::value * 2 },
        { vnode_shift_mod     <1, 384,  1, 16>::value, vnode_base_offset     <1, 384,  1, 16>::value * 2, 0 },

        { wrap_index_pair<1, 384,  2, 0>::value, vnode_adj_shift_offset<1, 384,  2, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384,  2, 1>::value * 2 }, // Row 2, degree = 19, nzs_row_degree = 17
        { wrap_index_pair<1, 384,  2, 1>::value, vnode_adj_shift_offset<1, 384,  2, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384,  2, 3>::value * 2 },
        { wrap_index_pair<1, 384,  2, 2>::value, vnode_adj_shift_offset<1, 384,  2, 4>::value * 2,  vnode_adj_shift_offset_if<1, 384,  2, 5>::value * 2 },
        { wrap_index_pair<1, 384,  2, 3>::value, vnode_adj_shift_offset<1, 384,  2, 6>::value * 2,  vnode_adj_shift_offset_if<1, 384,  2, 7>::value * 2 },
        { wrap_index_pair<1, 384,  2, 4>::value, vnode_adj_shift_offset<1, 384,  2, 8>::value * 2,  vnode_adj_shift_offset_if<1, 384,  2, 9>::value * 2 },
        { wrap_index_pair<1, 384,  2, 5>::value, vnode_adj_shift_offset<1, 384,  2, 10>::value * 2,  vnode_adj_shift_offset_if<1, 384,  2, 11>::value * 2 },
        { wrap_index_pair<1, 384,  2, 6>::value, vnode_adj_shift_offset<1, 384,  2, 12>::value * 2,  vnode_adj_shift_offset_if<1, 384,  2, 13>::value * 2 },
        { wrap_index_pair<1, 384,  2, 7>::value, vnode_adj_shift_offset<1, 384,  2, 14>::value * 2,  vnode_adj_shift_offset_if<1, 384,  2, 15>::value * 2 },
        { vnode_shift_mod     <1, 384,  2, 16>::value, vnode_base_offset     <1, 384,  2, 16>::value * 2, 0 },

        { wrap_index_pair<1, 384,  3, 0>::value, vnode_adj_shift_offset<1, 384,  3, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384,  3, 1>::value * 2 }, // Row 3, degree = 19, nzs_row_degree = 18
        { wrap_index_pair<1, 384,  3, 1>::value, vnode_adj_shift_offset<1, 384,  3, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384,  3, 3>::value * 2 },
        { wrap_index_pair<1, 384,  3, 2>::value, vnode_adj_shift_offset<1, 384,  3, 4>::value * 2,  vnode_adj_shift_offset_if<1, 384,  3, 5>::value * 2 },
        { wrap_index_pair<1, 384,  3, 3>::value, vnode_adj_shift_offset<1, 384,  3, 6>::value * 2,  vnode_adj_shift_offset_if<1, 384,  3, 7>::value * 2 },
        { wrap_index_pair<1, 384,  3, 4>::value, vnode_adj_shift_offset<1, 384,  3, 8>::value * 2,  vnode_adj_shift_offset_if<1, 384,  3, 9>::value * 2 },
        { wrap_index_pair<1, 384,  3, 5>::value, vnode_adj_shift_offset<1, 384,  3, 10>::value * 2,  vnode_adj_shift_offset_if<1, 384,  3, 11>::value * 2 },
        { wrap_index_pair<1, 384,  3, 6>::value, vnode_adj_shift_offset<1, 384,  3, 12>::value * 2,  vnode_adj_shift_offset_if<1, 384,  3, 13>::value * 2 },
        { wrap_index_pair<1, 384,  3, 7>::value, vnode_adj_shift_offset<1, 384,  3, 14>::value * 2,  vnode_adj_shift_offset_if<1, 384,  3, 15>::value * 2 },
        { wrap_index_pair<1, 384,  3, 8>::value, vnode_adj_shift_offset<1, 384,  3, 16>::value * 2,  vnode_adj_shift_offset_if<1, 384,  3, 17>::value * 2 },

        { wrap_index_pair<1, 384,  4, 0>::value, vnode_adj_shift_offset<1, 384,  4, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384,  4, 1>::value * 2 }, // Row 4, degree = 3, nzs_row_degree = 2

        { wrap_index_pair<1, 384,  5, 0>::value, vnode_adj_shift_offset<1, 384,  5, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384,  5, 1>::value * 2 }, // Row 5, degree = 8, nzs_row_degree = 7
        { wrap_index_pair<1, 384,  5, 1>::value, vnode_adj_shift_offset<1, 384,  5, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384,  5, 3>::value * 2 },
        { wrap_index_pair<1, 384,  5, 2>::value, vnode_adj_shift_offset<1, 384,  5, 4>::value * 2,  vnode_adj_shift_offset_if<1, 384,  5, 5>::value * 2 },
        { vnode_shift_mod     <1, 384,  5,  6>::value, vnode_base_offset     <1, 384,  5,  6>::value * 2, 0 },

        { wrap_index_pair<1, 384,  6, 0>::value, vnode_adj_shift_offset<1, 384,  6, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384,  6, 1>::value * 2 }, // Row 6, degree = 9, nzs_row_degree = 8
        { wrap_index_pair<1, 384,  6, 1>::value, vnode_adj_shift_offset<1, 384,  6, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384,  6, 3>::value * 2 },
        { wrap_index_pair<1, 384,  6, 2>::value, vnode_adj_shift_offset<1, 384,  6, 4>::value * 2,  vnode_adj_shift_offset_if<1, 384,  6, 5>::value * 2 },
        { wrap_index_pair<1, 384,  6, 3>::value, vnode_adj_shift_offset<1, 384,  6, 6>::value * 2,  vnode_adj_shift_offset_if<1, 384,  6, 7>::value * 2 },

        { wrap_index_pair<1, 384,  7, 0>::value, vnode_adj_shift_offset<1, 384,  7, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384,  7, 1>::value * 2 }, // Row 7, degree = 7, nzs_row_degree = 6
        { wrap_index_pair<1, 384,  7, 1>::value, vnode_adj_shift_offset<1, 384,  7, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384,  7, 3>::value * 2 },
        { wrap_index_pair<1, 384,  7, 2>::value, vnode_adj_shift_offset<1, 384,  7, 4>::value * 2,  vnode_adj_shift_offset_if<1, 384,  7, 5>::value * 2 },

        { wrap_index_pair<1, 384,  8, 0>::value, vnode_adj_shift_offset<1, 384,  8, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384,  8, 1>::value * 2 }, // Row 8, degree = 10, nzs_row_degree = 9
        { wrap_index_pair<1, 384,  8, 1>::value, vnode_adj_shift_offset<1, 384,  8, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384,  8, 3>::value * 2 },
        { wrap_index_pair<1, 384,  8, 2>::value, vnode_adj_shift_offset<1, 384,  8, 4>::value * 2,  vnode_adj_shift_offset_if<1, 384,  8, 5>::value * 2 },
        { wrap_index_pair<1, 384,  8, 3>::value, vnode_adj_shift_offset<1, 384,  8, 6>::value * 2,  vnode_adj_shift_offset_if<1, 384,  8, 7>::value * 2 },
        { vnode_shift_mod     <1, 384,  8,  8>::value, vnode_base_offset     <1, 384,  8,  8>::value * 2, 0 },

        { wrap_index_pair<1, 384,  9, 0>::value, vnode_adj_shift_offset<1, 384,  9, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384,  9, 1>::value * 2 }, // Row 9, degree = 9, nzs_row_degree = 8
        { wrap_index_pair<1, 384,  9, 1>::value, vnode_adj_shift_offset<1, 384,  9, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384,  9, 3>::value * 2 },
        { wrap_index_pair<1, 384,  9, 2>::value, vnode_adj_shift_offset<1, 384,  9, 4>::value * 2,  vnode_adj_shift_offset_if<1, 384,  9, 5>::value * 2 },
        { wrap_index_pair<1, 384,  9, 3>::value, vnode_adj_shift_offset<1, 384,  9, 6>::value * 2,  vnode_adj_shift_offset_if<1, 384,  9, 7>::value * 2 },

        { wrap_index_pair<1, 384, 10, 0>::value, vnode_adj_shift_offset<1, 384, 10, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 10, 1>::value * 2 }, // Row 10, degree = 7, nzs_row_degree = 6
        { wrap_index_pair<1, 384, 10, 1>::value, vnode_adj_shift_offset<1, 384, 10, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 10, 3>::value * 2 },
        { wrap_index_pair<1, 384, 10, 2>::value, vnode_adj_shift_offset<1, 384, 10, 4>::value * 2,  vnode_adj_shift_offset_if<1, 384, 10, 5>::value * 2 },

        { wrap_index_pair<1, 384, 11, 0>::value, vnode_adj_shift_offset<1, 384, 11, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 11, 1>::value * 2 }, // Row 11, degree = 8, nzs_row_degree = 7
        { wrap_index_pair<1, 384, 11, 1>::value, vnode_adj_shift_offset<1, 384, 11, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 11, 3>::value * 2 },
        { wrap_index_pair<1, 384, 11, 2>::value, vnode_adj_shift_offset<1, 384, 11, 4>::value * 2,  vnode_adj_shift_offset_if<1, 384, 11, 5>::value * 2 },
        { vnode_shift_mod     <1, 384, 11,  6>::value, vnode_base_offset     <1, 384, 11,  6>::value * 2, 0 },

        { wrap_index_pair<1, 384, 12, 0>::value, vnode_adj_shift_offset<1, 384, 12, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 12, 1>::value * 2 }, // Row 12, degree = 7, nzs_row_degree = 6
        { wrap_index_pair<1, 384, 12, 1>::value, vnode_adj_shift_offset<1, 384, 12, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 12, 3>::value * 2 },
        { wrap_index_pair<1, 384, 12, 2>::value, vnode_adj_shift_offset<1, 384, 12, 4>::value * 2,  vnode_adj_shift_offset_if<1, 384, 12, 5>::value * 2 },

        { wrap_index_pair<1, 384, 13, 0>::value, vnode_adj_shift_offset<1, 384, 13, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 13, 1>::value * 2 }, // Row 13, degree = 6, nzs_row_degree = 5
        { wrap_index_pair<1, 384, 13, 1>::value, vnode_adj_shift_offset<1, 384, 13, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 13, 3>::value * 2 },
        { vnode_shift_mod     <1, 384, 13,  4>::value, vnode_base_offset     <1, 384, 13,  4>::value * 2, 0 },

        { wrap_index_pair<1, 384, 14, 0>::value, vnode_adj_shift_offset<1, 384, 14, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 14, 1>::value * 2 }, // Row 14, degree = 7, nzs_row_degree = 6
        { wrap_index_pair<1, 384, 14, 1>::value, vnode_adj_shift_offset<1, 384, 14, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 14, 3>::value * 2 },
        { wrap_index_pair<1, 384, 14, 2>::value, vnode_adj_shift_offset<1, 384, 14, 4>::value * 2,  vnode_adj_shift_offset_if<1, 384, 14, 5>::value * 2 },

        { wrap_index_pair<1, 384, 15, 0>::value, vnode_adj_shift_offset<1, 384, 15, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 15, 1>::value * 2 }, // Row 15, degree = 7, nzs_row_degree = 6
        { wrap_index_pair<1, 384, 15, 1>::value, vnode_adj_shift_offset<1, 384, 15, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 15, 3>::value * 2 },
        { wrap_index_pair<1, 384, 15, 2>::value, vnode_adj_shift_offset<1, 384, 15, 4>::value * 2,  vnode_adj_shift_offset_if<1, 384, 15, 5>::value * 2 },

        { wrap_index_pair<1, 384, 16, 0>::value, vnode_adj_shift_offset<1, 384, 16, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 16, 1>::value * 2 }, // Row 16, degree = 6, nzs_row_degree = 5
        { wrap_index_pair<1, 384, 16, 1>::value, vnode_adj_shift_offset<1, 384, 16, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 16, 3>::value * 2 },
        { vnode_shift_mod     <1, 384, 16,  4>::value, vnode_base_offset     <1, 384, 16,  4>::value * 2, 0 },

        { wrap_index_pair<1, 384, 17, 0>::value, vnode_adj_shift_offset<1, 384, 17, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 17, 1>::value * 2 }, // Row 17, degree = 6, nzs_row_degree = 5
        { wrap_index_pair<1, 384, 17, 1>::value, vnode_adj_shift_offset<1, 384, 17, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 17, 3>::value * 2 },
        { vnode_shift_mod     <1, 384, 17,  4>::value, vnode_base_offset     <1, 384, 17,  4>::value * 2, 0 },

        { wrap_index_pair<1, 384, 18, 0>::value, vnode_adj_shift_offset<1, 384, 18, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 18, 1>::value * 2 }, // Row 18, degree = 6, nzs_row_degree = 5
        { wrap_index_pair<1, 384, 18, 1>::value, vnode_adj_shift_offset<1, 384, 18, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 18, 3>::value * 2 },
        { vnode_shift_mod     <1, 384, 18,  4>::value, vnode_base_offset     <1, 384, 18,  4>::value * 2, 0 },

        { wrap_index_pair<1, 384, 19, 0>::value, vnode_adj_shift_offset<1, 384, 19, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 19, 1>::value * 2 }, // Row 19, degree = 6, nzs_row_degree = 5
        { wrap_index_pair<1, 384, 19, 1>::value, vnode_adj_shift_offset<1, 384, 19, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 19, 3>::value * 2 },
        { vnode_shift_mod     <1, 384, 19,  4>::value, vnode_base_offset     <1, 384, 19,  4>::value * 2, 0 },

        { wrap_index_pair<1, 384, 20, 0>::value, vnode_adj_shift_offset<1, 384, 20, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 20, 1>::value * 2 }, // Row 20, degree = 6, nzs_row_degree = 5
        { wrap_index_pair<1, 384, 20, 1>::value, vnode_adj_shift_offset<1, 384, 20, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 20, 3>::value * 2 },
        { vnode_shift_mod     <1, 384, 20,  4>::value, vnode_base_offset     <1, 384, 20,  4>::value * 2, 0 },

        { wrap_index_pair<1, 384, 21, 0>::value, vnode_adj_shift_offset<1, 384, 21, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 21, 1>::value * 2 }, // Row 21, degree = 6, nzs_row_degree = 5
        { wrap_index_pair<1, 384, 21, 1>::value, vnode_adj_shift_offset<1, 384, 21, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 21, 3>::value * 2 },
        { vnode_shift_mod     <1, 384, 21,  4>::value, vnode_base_offset     <1, 384, 21,  4>::value * 2, 0 },

        { wrap_index_pair<1, 384, 22, 0>::value, vnode_adj_shift_offset<1, 384, 22, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 22, 1>::value * 2 }, // Row 22, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<1, 384, 22, 1>::value, vnode_adj_shift_offset<1, 384, 22, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 22, 3>::value * 2 },

        { wrap_index_pair<1, 384, 23, 0>::value, vnode_adj_shift_offset<1, 384, 23, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 23, 1>::value * 2 }, // Row 23, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<1, 384, 23, 1>::value, vnode_adj_shift_offset<1, 384, 23, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 23, 3>::value * 2 },

        { wrap_index_pair<1, 384, 24, 0>::value, vnode_adj_shift_offset<1, 384, 24, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 24, 1>::value * 2 }, // Row 24, degree = 6, nzs_row_degree = 5
        { wrap_index_pair<1, 384, 24, 1>::value, vnode_adj_shift_offset<1, 384, 24, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 24, 3>::value * 2 },
        { vnode_shift_mod     <1, 384, 24,  4>::value, vnode_base_offset     <1, 384, 24,  4>::value * 2, 0 },

        { wrap_index_pair<1, 384, 25, 0>::value, vnode_adj_shift_offset<1, 384, 25, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 25, 1>::value * 2 }, // Row 25, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<1, 384, 25, 1>::value, vnode_adj_shift_offset<1, 384, 25, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 25, 3>::value * 2 },

        { wrap_index_pair<1, 384, 26, 0>::value, vnode_adj_shift_offset<1, 384, 26, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 26, 1>::value * 2 }, // Row 26, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<1, 384, 26, 1>::value, vnode_adj_shift_offset<1, 384, 26, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 26, 3>::value * 2 },

        { wrap_index_pair<1, 384, 27, 0>::value, vnode_adj_shift_offset<1, 384, 27, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 27, 1>::value * 2 }, // Row 27, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <1, 384, 27,  2>::value, vnode_base_offset     <1, 384, 27,  2>::value * 2, 0 },

        { wrap_index_pair<1, 384, 28, 0>::value, vnode_adj_shift_offset<1, 384, 28, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 28, 1>::value * 2 }, // Row 28, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<1, 384, 28, 1>::value, vnode_adj_shift_offset<1, 384, 28, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 28, 3>::value * 2 },

        { wrap_index_pair<1, 384, 29, 0>::value, vnode_adj_shift_offset<1, 384, 29, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 29, 1>::value * 2 }, // Row 29, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<1, 384, 29, 1>::value, vnode_adj_shift_offset<1, 384, 29, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 29, 3>::value * 2 },

        { wrap_index_pair<1, 384, 30, 0>::value, vnode_adj_shift_offset<1, 384, 30, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 30, 1>::value * 2 }, // Row 30, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<1, 384, 30, 1>::value, vnode_adj_shift_offset<1, 384, 30, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 30, 3>::value * 2 },

        { wrap_index_pair<1, 384, 31, 0>::value, vnode_adj_shift_offset<1, 384, 31, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 31, 1>::value * 2 }, // Row 31, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<1, 384, 31, 1>::value, vnode_adj_shift_offset<1, 384, 31, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 31, 3>::value * 2 },

        { wrap_index_pair<1, 384, 32, 0>::value, vnode_adj_shift_offset<1, 384, 32, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 32, 1>::value * 2 }, // Row 32, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<1, 384, 32, 1>::value, vnode_adj_shift_offset<1, 384, 32, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 32, 3>::value * 2 },

        { wrap_index_pair<1, 384, 33, 0>::value, vnode_adj_shift_offset<1, 384, 33, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 33, 1>::value * 2 }, // Row 33, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<1, 384, 33, 1>::value, vnode_adj_shift_offset<1, 384, 33, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 33, 3>::value * 2 },

        { wrap_index_pair<1, 384, 34, 0>::value, vnode_adj_shift_offset<1, 384, 34, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 34, 1>::value * 2 }, // Row 34, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<1, 384, 34, 1>::value, vnode_adj_shift_offset<1, 384, 34, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 34, 3>::value * 2 },

        { wrap_index_pair<1, 384, 35, 0>::value, vnode_adj_shift_offset<1, 384, 35, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 35, 1>::value * 2 }, // Row 35, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<1, 384, 35, 1>::value, vnode_adj_shift_offset<1, 384, 35, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 35, 3>::value * 2 },

        { wrap_index_pair<1, 384, 36, 0>::value, vnode_adj_shift_offset<1, 384, 36, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 36, 1>::value * 2 }, // Row 36, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<1, 384, 36, 1>::value, vnode_adj_shift_offset<1, 384, 36, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 36, 3>::value * 2 },

        { wrap_index_pair<1, 384, 37, 0>::value, vnode_adj_shift_offset<1, 384, 37, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 37, 1>::value * 2 }, // Row 37, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <1, 384, 37,  2>::value, vnode_base_offset     <1, 384, 37,  2>::value * 2, 0 },

        { wrap_index_pair<1, 384, 38, 0>::value, vnode_adj_shift_offset<1, 384, 38, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 38, 1>::value * 2 }, // Row 38, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<1, 384, 38, 1>::value, vnode_adj_shift_offset<1, 384, 38, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 38, 3>::value * 2 },

        { wrap_index_pair<1, 384, 39, 0>::value, vnode_adj_shift_offset<1, 384, 39, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 39, 1>::value * 2 }, // Row 39, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<1, 384, 39, 1>::value, vnode_adj_shift_offset<1, 384, 39, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 39, 3>::value * 2 },

        { wrap_index_pair<1, 384, 40, 0>::value, vnode_adj_shift_offset<1, 384, 40, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 40, 1>::value * 2 }, // Row 40, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <1, 384, 40,  2>::value, vnode_base_offset     <1, 384, 40,  2>::value * 2, 0 },

        { wrap_index_pair<1, 384, 41, 0>::value, vnode_adj_shift_offset<1, 384, 41, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 41, 1>::value * 2 }, // Row 41, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<1, 384, 41, 1>::value, vnode_adj_shift_offset<1, 384, 41, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 41, 3>::value * 2 },

        { wrap_index_pair<1, 384, 42, 0>::value, vnode_adj_shift_offset<1, 384, 42, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 42, 1>::value * 2 }, // Row 42, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <1, 384, 42,  2>::value, vnode_base_offset     <1, 384, 42,  2>::value * 2, 0 },

        { wrap_index_pair<1, 384, 43, 0>::value, vnode_adj_shift_offset<1, 384, 43, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 43, 1>::value * 2 }, // Row 43, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<1, 384, 43, 1>::value, vnode_adj_shift_offset<1, 384, 43, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 43, 3>::value * 2 },

        { wrap_index_pair<1, 384, 44, 0>::value, vnode_adj_shift_offset<1, 384, 44, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 44, 1>::value * 2 }, // Row 44, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<1, 384, 44, 1>::value, vnode_adj_shift_offset<1, 384, 44, 2>::value * 2,  vnode_adj_shift_offset_if<1, 384, 44, 3>::value * 2 },

        { wrap_index_pair<1, 384, 45, 0>::value, vnode_adj_shift_offset<1, 384, 45, 0>::value * 2,  vnode_adj_shift_offset_if<1, 384, 45, 1>::value * 2 }, // Row 45, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <1, 384, 45,  2>::value, vnode_base_offset     <1, 384, 45,  2>::value * 2, 0 }
    }
};

const BG2_adj_nzs_desc_t BG2_adj_nzs_desc_Z384_16 =
{
    {
        { wrap_index_pair<2, 384,  0, 0>::value, vnode_adj_shift_offset<2, 384,  0, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384,  0, 1>::value * 2 }, // Row 0, degree = 8, nzs_row_degree = 7
        { wrap_index_pair<2, 384,  0, 1>::value, vnode_adj_shift_offset<2, 384,  0, 2>::value * 2,  vnode_adj_shift_offset_if<2, 384,  0, 3>::value * 2 },
        { wrap_index_pair<2, 384,  0, 2>::value, vnode_adj_shift_offset<2, 384,  0, 4>::value * 2,  vnode_adj_shift_offset_if<2, 384,  0, 5>::value * 2 },
        { vnode_shift_mod     <2, 384,  0,  6>::value, vnode_base_offset     <2, 384,  0,  6>::value * 2, 0 },

        { wrap_index_pair<2, 384,  1, 0>::value, vnode_adj_shift_offset<2, 384,  1, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384,  1, 1>::value * 2 }, // Row 1, degree = 10, nzs_row_degree = 8
        { wrap_index_pair<2, 384,  1, 1>::value, vnode_adj_shift_offset<2, 384,  1, 2>::value * 2,  vnode_adj_shift_offset_if<2, 384,  1, 3>::value * 2 },
        { wrap_index_pair<2, 384,  1, 2>::value, vnode_adj_shift_offset<2, 384,  1, 4>::value * 2,  vnode_adj_shift_offset_if<2, 384,  1, 5>::value * 2 },
        { wrap_index_pair<2, 384,  1, 3>::value, vnode_adj_shift_offset<2, 384,  1, 6>::value * 2,  vnode_adj_shift_offset_if<2, 384,  1, 7>::value * 2 },

        { wrap_index_pair<2, 384,  2, 0>::value, vnode_adj_shift_offset<2, 384,  2, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384,  2, 1>::value * 2 }, // Row 2, degree = 8, nzs_row_degree = 6
        { wrap_index_pair<2, 384,  2, 1>::value, vnode_adj_shift_offset<2, 384,  2, 2>::value * 2,  vnode_adj_shift_offset_if<2, 384,  2, 3>::value * 2 },
        { wrap_index_pair<2, 384,  2, 2>::value, vnode_adj_shift_offset<2, 384,  2, 4>::value * 2,  vnode_adj_shift_offset_if<2, 384,  2, 5>::value * 2 },

        { wrap_index_pair<2, 384,  3, 0>::value, vnode_adj_shift_offset<2, 384,  3, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384,  3, 1>::value * 2 }, // Row 3, degree = 10, nzs_row_degree = 9
        { wrap_index_pair<2, 384,  3, 1>::value, vnode_adj_shift_offset<2, 384,  3, 2>::value * 2,  vnode_adj_shift_offset_if<2, 384,  3, 3>::value * 2 },
        { wrap_index_pair<2, 384,  3, 2>::value, vnode_adj_shift_offset<2, 384,  3, 4>::value * 2,  vnode_adj_shift_offset_if<2, 384,  3, 5>::value * 2 },
        { wrap_index_pair<2, 384,  3, 3>::value, vnode_adj_shift_offset<2, 384,  3, 6>::value * 2,  vnode_adj_shift_offset_if<2, 384,  3, 7>::value * 2 },
        { vnode_shift_mod     <2, 384,  3,  8>::value, vnode_base_offset     <2, 384,  3,  8>::value * 2, 0 },

        { wrap_index_pair<2, 384,  4, 0>::value, vnode_adj_shift_offset<2, 384,  4, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384,  4, 1>::value * 2 }, // Row 4, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <2, 384,  4,  2>::value, vnode_base_offset     <2, 384,  4,  2>::value * 2, 0 },

        { wrap_index_pair<2, 384,  5, 0>::value, vnode_adj_shift_offset<2, 384,  5, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384,  5, 1>::value * 2 }, // Row 5, degree = 6, nzs_row_degree = 5
        { wrap_index_pair<2, 384,  5, 1>::value, vnode_adj_shift_offset<2, 384,  5, 2>::value * 2,  vnode_adj_shift_offset_if<2, 384,  5, 3>::value * 2 },
        { vnode_shift_mod     <2, 384,  5,  4>::value, vnode_base_offset     <2, 384,  5,  4>::value * 2, 0 },

        { wrap_index_pair<2, 384,  6, 0>::value, vnode_adj_shift_offset<2, 384,  6, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384,  6, 1>::value * 2 }, // Row 6, degree = 6, nzs_row_degree = 5
        { wrap_index_pair<2, 384,  6, 1>::value, vnode_adj_shift_offset<2, 384,  6, 2>::value * 2,  vnode_adj_shift_offset_if<2, 384,  6, 3>::value * 2 },
        { vnode_shift_mod     <2, 384,  6,  4>::value, vnode_base_offset     <2, 384,  6,  4>::value * 2, 0 },

        { wrap_index_pair<2, 384,  7, 0>::value, vnode_adj_shift_offset<2, 384,  7, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384,  7, 1>::value * 2 }, // Row 7, degree = 6, nzs_row_degree = 5
        { wrap_index_pair<2, 384,  7, 1>::value, vnode_adj_shift_offset<2, 384,  7, 2>::value * 2,  vnode_adj_shift_offset_if<2, 384,  7, 3>::value * 2 },
        { vnode_shift_mod     <2, 384,  7,  4>::value, vnode_base_offset     <2, 384,  7,  4>::value * 2, 0 },

        { wrap_index_pair<2, 384,  8, 0>::value, vnode_adj_shift_offset<2, 384,  8, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384,  8, 1>::value * 2 }, // Row 8, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <2, 384,  8,  2>::value, vnode_base_offset     <2, 384,  8,  2>::value * 2, 0 },

        { wrap_index_pair<2, 384,  9, 0>::value, vnode_adj_shift_offset<2, 384,  9, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384,  9, 1>::value * 2 }, // Row 9, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<2, 384,  9, 1>::value, vnode_adj_shift_offset<2, 384,  9, 2>::value * 2,  vnode_adj_shift_offset_if<2, 384,  9, 3>::value * 2 },

        { wrap_index_pair<2, 384, 10, 0>::value, vnode_adj_shift_offset<2, 384, 10, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 10, 1>::value * 2 }, // Row 10, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<2, 384, 10, 1>::value, vnode_adj_shift_offset<2, 384, 10, 2>::value * 2,  vnode_adj_shift_offset_if<2, 384, 10, 3>::value * 2 },

        { wrap_index_pair<2, 384, 11, 0>::value, vnode_adj_shift_offset<2, 384, 11, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 11, 1>::value * 2 }, // Row 11, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<2, 384, 11, 1>::value, vnode_adj_shift_offset<2, 384, 11, 2>::value * 2,  vnode_adj_shift_offset_if<2, 384, 11, 3>::value * 2 },

        { wrap_index_pair<2, 384, 12, 0>::value, vnode_adj_shift_offset<2, 384, 12, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 12, 1>::value * 2 }, // Row 12, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <2, 384, 12,  2>::value, vnode_base_offset     <2, 384, 12,  2>::value * 2, 0 },

        { wrap_index_pair<2, 384, 13, 0>::value, vnode_adj_shift_offset<2, 384, 13, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 13, 1>::value * 2 }, // Row 13, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<2, 384, 13, 1>::value, vnode_adj_shift_offset<2, 384, 13, 2>::value * 2,  vnode_adj_shift_offset_if<2, 384, 13, 3>::value * 2 },

        { wrap_index_pair<2, 384, 14, 0>::value, vnode_adj_shift_offset<2, 384, 14, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 14, 1>::value * 2 }, // Row 14, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<2, 384, 14, 1>::value, vnode_adj_shift_offset<2, 384, 14, 2>::value * 2,  vnode_adj_shift_offset_if<2, 384, 14, 3>::value * 2 },

        { wrap_index_pair<2, 384, 15, 0>::value, vnode_adj_shift_offset<2, 384, 15, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 15, 1>::value * 2 }, // Row 15, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <2, 384, 15,  2>::value, vnode_base_offset     <2, 384, 15,  2>::value * 2, 0 },

        { wrap_index_pair<2, 384, 16, 0>::value, vnode_adj_shift_offset<2, 384, 16, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 16, 1>::value * 2 }, // Row 16, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<2, 384, 16, 1>::value, vnode_adj_shift_offset<2, 384, 16, 2>::value * 2,  vnode_adj_shift_offset_if<2, 384, 16, 3>::value * 2 },

        { wrap_index_pair<2, 384, 17, 0>::value, vnode_adj_shift_offset<2, 384, 17, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 17, 1>::value * 2 }, // Row 17, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<2, 384, 17, 1>::value, vnode_adj_shift_offset<2, 384, 17, 2>::value * 2,  vnode_adj_shift_offset_if<2, 384, 17, 3>::value * 2 },

        { wrap_index_pair<2, 384, 18, 0>::value, vnode_adj_shift_offset<2, 384, 18, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 18, 1>::value * 2 }, // Row 18, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <2, 384, 18,  2>::value, vnode_base_offset     <2, 384, 18,  2>::value * 2, 0 },

        { wrap_index_pair<2, 384, 19, 0>::value, vnode_adj_shift_offset<2, 384, 19, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 19, 1>::value * 2 }, // Row 19, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <2, 384, 19,  2>::value, vnode_base_offset     <2, 384, 19,  2>::value * 2, 0 },

        { wrap_index_pair<2, 384, 20, 0>::value, vnode_adj_shift_offset<2, 384, 20, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 20, 1>::value * 2 }, // Row 20, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <2, 384, 20,  2>::value, vnode_base_offset     <2, 384, 20,  2>::value * 2, 0 },

        { wrap_index_pair<2, 384, 21, 0>::value, vnode_adj_shift_offset<2, 384, 21, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 21, 1>::value * 2 }, // Row 21, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <2, 384, 21,  2>::value, vnode_base_offset     <2, 384, 21,  2>::value * 2, 0 },

        { wrap_index_pair<2, 384, 22, 0>::value, vnode_adj_shift_offset<2, 384, 22, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 22, 1>::value * 2 }, // Row 22, degree = 3, nzs_row_degree = 2

        { wrap_index_pair<2, 384, 23, 0>::value, vnode_adj_shift_offset<2, 384, 23, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 23, 1>::value * 2 }, // Row 23, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <2, 384, 23,  2>::value, vnode_base_offset     <2, 384, 23,  2>::value * 2, 0 },

        { wrap_index_pair<2, 384, 24, 0>::value, vnode_adj_shift_offset<2, 384, 24, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 24, 1>::value * 2 }, // Row 24, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <2, 384, 24,  2>::value, vnode_base_offset     <2, 384, 24,  2>::value * 2, 0 },

        { wrap_index_pair<2, 384, 25, 0>::value, vnode_adj_shift_offset<2, 384, 25, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 25, 1>::value * 2 }, // Row 25, degree = 3, nzs_row_degree = 2

        { wrap_index_pair<2, 384, 26, 0>::value, vnode_adj_shift_offset<2, 384, 26, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 26, 1>::value * 2 }, // Row 26, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<2, 384, 26, 1>::value, vnode_adj_shift_offset<2, 384, 26, 2>::value * 2,  vnode_adj_shift_offset_if<2, 384, 26, 3>::value * 2 },

        { wrap_index_pair<2, 384, 27, 0>::value, vnode_adj_shift_offset<2, 384, 27, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 27, 1>::value * 2 }, // Row 27, degree = 3, nzs_row_degree = 2

        { wrap_index_pair<2, 384, 28, 0>::value, vnode_adj_shift_offset<2, 384, 28, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 28, 1>::value * 2 }, // Row 28, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <2, 384, 28,  2>::value, vnode_base_offset     <2, 384, 28,  2>::value * 2, 0 },

        { wrap_index_pair<2, 384, 29, 0>::value, vnode_adj_shift_offset<2, 384, 29, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 29, 1>::value * 2 }, // Row 29, degree = 3, nzs_row_degree = 2

        { wrap_index_pair<2, 384, 30, 0>::value, vnode_adj_shift_offset<2, 384, 30, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 30, 1>::value * 2 }, // Row 30, degree = 5, nzs_row_degree = 4
        { wrap_index_pair<2, 384, 30, 1>::value, vnode_adj_shift_offset<2, 384, 30, 2>::value * 2,  vnode_adj_shift_offset_if<2, 384, 30, 3>::value * 2 },

        { wrap_index_pair<2, 384, 31, 0>::value, vnode_adj_shift_offset<2, 384, 31, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 31, 1>::value * 2 }, // Row 31, degree = 3, nzs_row_degree = 2

        { wrap_index_pair<2, 384, 32, 0>::value, vnode_adj_shift_offset<2, 384, 32, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 32, 1>::value * 2 }, // Row 32, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <2, 384, 32,  2>::value, vnode_base_offset     <2, 384, 32,  2>::value * 2, 0 },

        { wrap_index_pair<2, 384, 33, 0>::value, vnode_adj_shift_offset<2, 384, 33, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 33, 1>::value * 2 }, // Row 33, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <2, 384, 33,  2>::value, vnode_base_offset     <2, 384, 33,  2>::value * 2, 0 },

        { wrap_index_pair<2, 384, 34, 0>::value, vnode_adj_shift_offset<2, 384, 34, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 34, 1>::value * 2 }, // Row 34, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <2, 384, 34,  2>::value, vnode_base_offset     <2, 384, 34,  2>::value * 2, 0 },

        { wrap_index_pair<2, 384, 35, 0>::value, vnode_adj_shift_offset<2, 384, 35, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 35, 1>::value * 2 }, // Row 35, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <2, 384, 35,  2>::value, vnode_base_offset     <2, 384, 35,  2>::value * 2, 0 },

        { wrap_index_pair<2, 384, 36, 0>::value, vnode_adj_shift_offset<2, 384, 36, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 36, 1>::value * 2 }, // Row 36, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <2, 384, 36,  2>::value, vnode_base_offset     <2, 384, 36,  2>::value * 2, 0 },

        { wrap_index_pair<2, 384, 37, 0>::value, vnode_adj_shift_offset<2, 384, 37, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 37, 1>::value * 2 }, // Row 37, degree = 3, nzs_row_degree = 2

        { wrap_index_pair<2, 384, 38, 0>::value, vnode_adj_shift_offset<2, 384, 38, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 38, 1>::value * 2 }, // Row 38, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <2, 384, 38,  2>::value, vnode_base_offset     <2, 384, 38,  2>::value * 2, 0 },

        { wrap_index_pair<2, 384, 39, 0>::value, vnode_adj_shift_offset<2, 384, 39, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 39, 1>::value * 2 }, // Row 39, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <2, 384, 39,  2>::value, vnode_base_offset     <2, 384, 39,  2>::value * 2, 0 },

        { wrap_index_pair<2, 384, 40, 0>::value, vnode_adj_shift_offset<2, 384, 40, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 40, 1>::value * 2 }, // Row 40, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <2, 384, 40,  2>::value, vnode_base_offset     <2, 384, 40,  2>::value * 2, 0 },

        { wrap_index_pair<2, 384, 41, 0>::value, vnode_adj_shift_offset<2, 384, 41, 0>::value * 2,  vnode_adj_shift_offset_if<2, 384, 41, 1>::value * 2 }, // Row 41, degree = 4, nzs_row_degree = 3
        { vnode_shift_mod     <2, 384, 41,  2>::value, vnode_base_offset     <2, 384, 41,  2>::value * 2, 0 }
    }
};


} // namespace ldpc2
