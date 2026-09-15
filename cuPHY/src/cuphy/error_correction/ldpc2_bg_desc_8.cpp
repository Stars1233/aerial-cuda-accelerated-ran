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

const BG1_desc_t BG1_desc_Z32_8 =
{
    {
        { vnode_shift_mod_pair<1,  32,  0, 0>::value, vnode_base_offset_pair<1,  32,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1,  32,  0, 1>::value, vnode_base_offset_pair<1,  32,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  0, 2>::value, vnode_base_offset_pair<1,  32,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  0, 3>::value, vnode_base_offset_pair<1,  32,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  0, 4>::value, vnode_base_offset_pair<1,  32,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  0, 5>::value, vnode_base_offset_pair<1,  32,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  0, 6>::value, vnode_base_offset_pair<1,  32,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  0, 7>::value, vnode_base_offset_pair<1,  32,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  0, 8>::value, vnode_base_offset_pair<1,  32,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  0, 9>::value, vnode_base_offset_pair<1,  32,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  32,  1, 0>::value, vnode_base_offset_pair<1,  32,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1,  32,  1, 1>::value, vnode_base_offset_pair<1,  32,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  1, 2>::value, vnode_base_offset_pair<1,  32,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  1, 3>::value, vnode_base_offset_pair<1,  32,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  1, 4>::value, vnode_base_offset_pair<1,  32,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  1, 5>::value, vnode_base_offset_pair<1,  32,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  1, 6>::value, vnode_base_offset_pair<1,  32,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  1, 7>::value, vnode_base_offset_pair<1,  32,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  1, 8>::value, vnode_base_offset_pair<1,  32,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  1, 9>::value, vnode_base_offset_pair<1,  32,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  32,  2, 0>::value, vnode_base_offset_pair<1,  32,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1,  32,  2, 1>::value, vnode_base_offset_pair<1,  32,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  2, 2>::value, vnode_base_offset_pair<1,  32,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  2, 3>::value, vnode_base_offset_pair<1,  32,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  2, 4>::value, vnode_base_offset_pair<1,  32,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  2, 5>::value, vnode_base_offset_pair<1,  32,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  2, 6>::value, vnode_base_offset_pair<1,  32,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  2, 7>::value, vnode_base_offset_pair<1,  32,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  2, 8>::value, vnode_base_offset_pair<1,  32,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  2, 9>::value, vnode_base_offset_pair<1,  32,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  32,  3, 0>::value, vnode_base_offset_pair<1,  32,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1,  32,  3, 1>::value, vnode_base_offset_pair<1,  32,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  3, 2>::value, vnode_base_offset_pair<1,  32,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  3, 3>::value, vnode_base_offset_pair<1,  32,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  3, 4>::value, vnode_base_offset_pair<1,  32,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  3, 5>::value, vnode_base_offset_pair<1,  32,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  3, 6>::value, vnode_base_offset_pair<1,  32,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  3, 7>::value, vnode_base_offset_pair<1,  32,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  3, 8>::value, vnode_base_offset_pair<1,  32,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  3, 9>::value, vnode_base_offset_pair<1,  32,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  32,  4, 0>::value, vnode_base_offset_pair<1,  32,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1,  32,  4, 1>::value, vnode_base_offset_pair<1,  32,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  32,  5, 0>::value, vnode_base_offset_pair<1,  32,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1,  32,  5, 1>::value, vnode_base_offset_pair<1,  32,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  5, 2>::value, vnode_base_offset_pair<1,  32,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  5, 3>::value, vnode_base_offset_pair<1,  32,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  32,  6, 0>::value, vnode_base_offset_pair<1,  32,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1,  32,  6, 1>::value, vnode_base_offset_pair<1,  32,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  6, 2>::value, vnode_base_offset_pair<1,  32,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  6, 3>::value, vnode_base_offset_pair<1,  32,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  6, 4>::value, vnode_base_offset_pair<1,  32,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  32,  7, 0>::value, vnode_base_offset_pair<1,  32,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1,  32,  7, 1>::value, vnode_base_offset_pair<1,  32,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  7, 2>::value, vnode_base_offset_pair<1,  32,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  7, 3>::value, vnode_base_offset_pair<1,  32,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  32,  8, 0>::value, vnode_base_offset_pair<1,  32,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1,  32,  8, 1>::value, vnode_base_offset_pair<1,  32,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  8, 2>::value, vnode_base_offset_pair<1,  32,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  8, 3>::value, vnode_base_offset_pair<1,  32,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  8, 4>::value, vnode_base_offset_pair<1,  32,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  32,  9, 0>::value, vnode_base_offset_pair<1,  32,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1,  32,  9, 1>::value, vnode_base_offset_pair<1,  32,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  9, 2>::value, vnode_base_offset_pair<1,  32,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  9, 3>::value, vnode_base_offset_pair<1,  32,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  32,  9, 4>::value, vnode_base_offset_pair<1,  32,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 10, 0>::value, vnode_base_offset_pair<1,  32, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1,  32, 10, 1>::value, vnode_base_offset_pair<1,  32, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 10, 2>::value, vnode_base_offset_pair<1,  32, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 10, 3>::value, vnode_base_offset_pair<1,  32, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 11, 0>::value, vnode_base_offset_pair<1,  32, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1,  32, 11, 1>::value, vnode_base_offset_pair<1,  32, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 11, 2>::value, vnode_base_offset_pair<1,  32, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 11, 3>::value, vnode_base_offset_pair<1,  32, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 12, 0>::value, vnode_base_offset_pair<1,  32, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1,  32, 12, 1>::value, vnode_base_offset_pair<1,  32, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 12, 2>::value, vnode_base_offset_pair<1,  32, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 12, 3>::value, vnode_base_offset_pair<1,  32, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 13, 0>::value, vnode_base_offset_pair<1,  32, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1,  32, 13, 1>::value, vnode_base_offset_pair<1,  32, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 13, 2>::value, vnode_base_offset_pair<1,  32, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 14, 0>::value, vnode_base_offset_pair<1,  32, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1,  32, 14, 1>::value, vnode_base_offset_pair<1,  32, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 14, 2>::value, vnode_base_offset_pair<1,  32, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 14, 3>::value, vnode_base_offset_pair<1,  32, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 15, 0>::value, vnode_base_offset_pair<1,  32, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1,  32, 15, 1>::value, vnode_base_offset_pair<1,  32, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 15, 2>::value, vnode_base_offset_pair<1,  32, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 15, 3>::value, vnode_base_offset_pair<1,  32, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 16, 0>::value, vnode_base_offset_pair<1,  32, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1,  32, 16, 1>::value, vnode_base_offset_pair<1,  32, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 16, 2>::value, vnode_base_offset_pair<1,  32, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 17, 0>::value, vnode_base_offset_pair<1,  32, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1,  32, 17, 1>::value, vnode_base_offset_pair<1,  32, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 17, 2>::value, vnode_base_offset_pair<1,  32, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 18, 0>::value, vnode_base_offset_pair<1,  32, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1,  32, 18, 1>::value, vnode_base_offset_pair<1,  32, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 18, 2>::value, vnode_base_offset_pair<1,  32, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 19, 0>::value, vnode_base_offset_pair<1,  32, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1,  32, 19, 1>::value, vnode_base_offset_pair<1,  32, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 19, 2>::value, vnode_base_offset_pair<1,  32, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 20, 0>::value, vnode_base_offset_pair<1,  32, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1,  32, 20, 1>::value, vnode_base_offset_pair<1,  32, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 20, 2>::value, vnode_base_offset_pair<1,  32, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 21, 0>::value, vnode_base_offset_pair<1,  32, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1,  32, 21, 1>::value, vnode_base_offset_pair<1,  32, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 21, 2>::value, vnode_base_offset_pair<1,  32, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 22, 0>::value, vnode_base_offset_pair<1,  32, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1,  32, 22, 1>::value, vnode_base_offset_pair<1,  32, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 22, 2>::value, vnode_base_offset_pair<1,  32, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 23, 0>::value, vnode_base_offset_pair<1,  32, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1,  32, 23, 1>::value, vnode_base_offset_pair<1,  32, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 23, 2>::value, vnode_base_offset_pair<1,  32, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 24, 0>::value, vnode_base_offset_pair<1,  32, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1,  32, 24, 1>::value, vnode_base_offset_pair<1,  32, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 24, 2>::value, vnode_base_offset_pair<1,  32, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 25, 0>::value, vnode_base_offset_pair<1,  32, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1,  32, 25, 1>::value, vnode_base_offset_pair<1,  32, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 25, 2>::value, vnode_base_offset_pair<1,  32, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 26, 0>::value, vnode_base_offset_pair<1,  32, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1,  32, 26, 1>::value, vnode_base_offset_pair<1,  32, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 26, 2>::value, vnode_base_offset_pair<1,  32, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 27, 0>::value, vnode_base_offset_pair<1,  32, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1,  32, 27, 1>::value, vnode_base_offset_pair<1,  32, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 28, 0>::value, vnode_base_offset_pair<1,  32, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1,  32, 28, 1>::value, vnode_base_offset_pair<1,  32, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 28, 2>::value, vnode_base_offset_pair<1,  32, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 29, 0>::value, vnode_base_offset_pair<1,  32, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1,  32, 29, 1>::value, vnode_base_offset_pair<1,  32, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 29, 2>::value, vnode_base_offset_pair<1,  32, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 30, 0>::value, vnode_base_offset_pair<1,  32, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1,  32, 30, 1>::value, vnode_base_offset_pair<1,  32, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 30, 2>::value, vnode_base_offset_pair<1,  32, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 31, 0>::value, vnode_base_offset_pair<1,  32, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1,  32, 31, 1>::value, vnode_base_offset_pair<1,  32, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 31, 2>::value, vnode_base_offset_pair<1,  32, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 32, 0>::value, vnode_base_offset_pair<1,  32, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1,  32, 32, 1>::value, vnode_base_offset_pair<1,  32, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 32, 2>::value, vnode_base_offset_pair<1,  32, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 33, 0>::value, vnode_base_offset_pair<1,  32, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1,  32, 33, 1>::value, vnode_base_offset_pair<1,  32, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 33, 2>::value, vnode_base_offset_pair<1,  32, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 34, 0>::value, vnode_base_offset_pair<1,  32, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1,  32, 34, 1>::value, vnode_base_offset_pair<1,  32, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 34, 2>::value, vnode_base_offset_pair<1,  32, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 35, 0>::value, vnode_base_offset_pair<1,  32, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1,  32, 35, 1>::value, vnode_base_offset_pair<1,  32, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 35, 2>::value, vnode_base_offset_pair<1,  32, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 36, 0>::value, vnode_base_offset_pair<1,  32, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1,  32, 36, 1>::value, vnode_base_offset_pair<1,  32, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 36, 2>::value, vnode_base_offset_pair<1,  32, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 37, 0>::value, vnode_base_offset_pair<1,  32, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1,  32, 37, 1>::value, vnode_base_offset_pair<1,  32, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 38, 0>::value, vnode_base_offset_pair<1,  32, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1,  32, 38, 1>::value, vnode_base_offset_pair<1,  32, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 38, 2>::value, vnode_base_offset_pair<1,  32, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 39, 0>::value, vnode_base_offset_pair<1,  32, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1,  32, 39, 1>::value, vnode_base_offset_pair<1,  32, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 39, 2>::value, vnode_base_offset_pair<1,  32, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 40, 0>::value, vnode_base_offset_pair<1,  32, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1,  32, 40, 1>::value, vnode_base_offset_pair<1,  32, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 41, 0>::value, vnode_base_offset_pair<1,  32, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1,  32, 41, 1>::value, vnode_base_offset_pair<1,  32, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 41, 2>::value, vnode_base_offset_pair<1,  32, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 42, 0>::value, vnode_base_offset_pair<1,  32, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1,  32, 42, 1>::value, vnode_base_offset_pair<1,  32, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 43, 0>::value, vnode_base_offset_pair<1,  32, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1,  32, 43, 1>::value, vnode_base_offset_pair<1,  32, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 43, 2>::value, vnode_base_offset_pair<1,  32, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 44, 0>::value, vnode_base_offset_pair<1,  32, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1,  32, 44, 1>::value, vnode_base_offset_pair<1,  32, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  32, 44, 2>::value, vnode_base_offset_pair<1,  32, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  32, 45, 0>::value, vnode_base_offset_pair<1,  32, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1,  32, 45, 1>::value, vnode_base_offset_pair<1,  32, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z36_8 =
{
    {
        { vnode_shift_mod_pair<1,  36,  0, 0>::value, vnode_base_offset_pair<1,  36,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1,  36,  0, 1>::value, vnode_base_offset_pair<1,  36,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  0, 2>::value, vnode_base_offset_pair<1,  36,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  0, 3>::value, vnode_base_offset_pair<1,  36,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  0, 4>::value, vnode_base_offset_pair<1,  36,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  0, 5>::value, vnode_base_offset_pair<1,  36,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  0, 6>::value, vnode_base_offset_pair<1,  36,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  0, 7>::value, vnode_base_offset_pair<1,  36,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  0, 8>::value, vnode_base_offset_pair<1,  36,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  0, 9>::value, vnode_base_offset_pair<1,  36,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  36,  1, 0>::value, vnode_base_offset_pair<1,  36,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1,  36,  1, 1>::value, vnode_base_offset_pair<1,  36,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  1, 2>::value, vnode_base_offset_pair<1,  36,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  1, 3>::value, vnode_base_offset_pair<1,  36,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  1, 4>::value, vnode_base_offset_pair<1,  36,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  1, 5>::value, vnode_base_offset_pair<1,  36,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  1, 6>::value, vnode_base_offset_pair<1,  36,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  1, 7>::value, vnode_base_offset_pair<1,  36,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  1, 8>::value, vnode_base_offset_pair<1,  36,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  1, 9>::value, vnode_base_offset_pair<1,  36,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  36,  2, 0>::value, vnode_base_offset_pair<1,  36,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1,  36,  2, 1>::value, vnode_base_offset_pair<1,  36,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  2, 2>::value, vnode_base_offset_pair<1,  36,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  2, 3>::value, vnode_base_offset_pair<1,  36,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  2, 4>::value, vnode_base_offset_pair<1,  36,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  2, 5>::value, vnode_base_offset_pair<1,  36,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  2, 6>::value, vnode_base_offset_pair<1,  36,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  2, 7>::value, vnode_base_offset_pair<1,  36,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  2, 8>::value, vnode_base_offset_pair<1,  36,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  2, 9>::value, vnode_base_offset_pair<1,  36,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  36,  3, 0>::value, vnode_base_offset_pair<1,  36,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1,  36,  3, 1>::value, vnode_base_offset_pair<1,  36,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  3, 2>::value, vnode_base_offset_pair<1,  36,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  3, 3>::value, vnode_base_offset_pair<1,  36,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  3, 4>::value, vnode_base_offset_pair<1,  36,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  3, 5>::value, vnode_base_offset_pair<1,  36,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  3, 6>::value, vnode_base_offset_pair<1,  36,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  3, 7>::value, vnode_base_offset_pair<1,  36,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  3, 8>::value, vnode_base_offset_pair<1,  36,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  3, 9>::value, vnode_base_offset_pair<1,  36,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  36,  4, 0>::value, vnode_base_offset_pair<1,  36,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1,  36,  4, 1>::value, vnode_base_offset_pair<1,  36,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  36,  5, 0>::value, vnode_base_offset_pair<1,  36,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1,  36,  5, 1>::value, vnode_base_offset_pair<1,  36,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  5, 2>::value, vnode_base_offset_pair<1,  36,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  5, 3>::value, vnode_base_offset_pair<1,  36,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  36,  6, 0>::value, vnode_base_offset_pair<1,  36,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1,  36,  6, 1>::value, vnode_base_offset_pair<1,  36,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  6, 2>::value, vnode_base_offset_pair<1,  36,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  6, 3>::value, vnode_base_offset_pair<1,  36,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  6, 4>::value, vnode_base_offset_pair<1,  36,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  36,  7, 0>::value, vnode_base_offset_pair<1,  36,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1,  36,  7, 1>::value, vnode_base_offset_pair<1,  36,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  7, 2>::value, vnode_base_offset_pair<1,  36,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  7, 3>::value, vnode_base_offset_pair<1,  36,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  36,  8, 0>::value, vnode_base_offset_pair<1,  36,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1,  36,  8, 1>::value, vnode_base_offset_pair<1,  36,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  8, 2>::value, vnode_base_offset_pair<1,  36,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  8, 3>::value, vnode_base_offset_pair<1,  36,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  8, 4>::value, vnode_base_offset_pair<1,  36,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  36,  9, 0>::value, vnode_base_offset_pair<1,  36,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1,  36,  9, 1>::value, vnode_base_offset_pair<1,  36,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  9, 2>::value, vnode_base_offset_pair<1,  36,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  9, 3>::value, vnode_base_offset_pair<1,  36,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  36,  9, 4>::value, vnode_base_offset_pair<1,  36,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 10, 0>::value, vnode_base_offset_pair<1,  36, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1,  36, 10, 1>::value, vnode_base_offset_pair<1,  36, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 10, 2>::value, vnode_base_offset_pair<1,  36, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 10, 3>::value, vnode_base_offset_pair<1,  36, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 11, 0>::value, vnode_base_offset_pair<1,  36, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1,  36, 11, 1>::value, vnode_base_offset_pair<1,  36, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 11, 2>::value, vnode_base_offset_pair<1,  36, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 11, 3>::value, vnode_base_offset_pair<1,  36, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 12, 0>::value, vnode_base_offset_pair<1,  36, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1,  36, 12, 1>::value, vnode_base_offset_pair<1,  36, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 12, 2>::value, vnode_base_offset_pair<1,  36, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 12, 3>::value, vnode_base_offset_pair<1,  36, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 13, 0>::value, vnode_base_offset_pair<1,  36, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1,  36, 13, 1>::value, vnode_base_offset_pair<1,  36, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 13, 2>::value, vnode_base_offset_pair<1,  36, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 14, 0>::value, vnode_base_offset_pair<1,  36, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1,  36, 14, 1>::value, vnode_base_offset_pair<1,  36, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 14, 2>::value, vnode_base_offset_pair<1,  36, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 14, 3>::value, vnode_base_offset_pair<1,  36, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 15, 0>::value, vnode_base_offset_pair<1,  36, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1,  36, 15, 1>::value, vnode_base_offset_pair<1,  36, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 15, 2>::value, vnode_base_offset_pair<1,  36, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 15, 3>::value, vnode_base_offset_pair<1,  36, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 16, 0>::value, vnode_base_offset_pair<1,  36, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1,  36, 16, 1>::value, vnode_base_offset_pair<1,  36, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 16, 2>::value, vnode_base_offset_pair<1,  36, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 17, 0>::value, vnode_base_offset_pair<1,  36, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1,  36, 17, 1>::value, vnode_base_offset_pair<1,  36, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 17, 2>::value, vnode_base_offset_pair<1,  36, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 18, 0>::value, vnode_base_offset_pair<1,  36, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1,  36, 18, 1>::value, vnode_base_offset_pair<1,  36, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 18, 2>::value, vnode_base_offset_pair<1,  36, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 19, 0>::value, vnode_base_offset_pair<1,  36, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1,  36, 19, 1>::value, vnode_base_offset_pair<1,  36, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 19, 2>::value, vnode_base_offset_pair<1,  36, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 20, 0>::value, vnode_base_offset_pair<1,  36, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1,  36, 20, 1>::value, vnode_base_offset_pair<1,  36, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 20, 2>::value, vnode_base_offset_pair<1,  36, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 21, 0>::value, vnode_base_offset_pair<1,  36, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1,  36, 21, 1>::value, vnode_base_offset_pair<1,  36, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 21, 2>::value, vnode_base_offset_pair<1,  36, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 22, 0>::value, vnode_base_offset_pair<1,  36, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1,  36, 22, 1>::value, vnode_base_offset_pair<1,  36, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 22, 2>::value, vnode_base_offset_pair<1,  36, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 23, 0>::value, vnode_base_offset_pair<1,  36, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1,  36, 23, 1>::value, vnode_base_offset_pair<1,  36, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 23, 2>::value, vnode_base_offset_pair<1,  36, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 24, 0>::value, vnode_base_offset_pair<1,  36, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1,  36, 24, 1>::value, vnode_base_offset_pair<1,  36, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 24, 2>::value, vnode_base_offset_pair<1,  36, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 25, 0>::value, vnode_base_offset_pair<1,  36, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1,  36, 25, 1>::value, vnode_base_offset_pair<1,  36, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 25, 2>::value, vnode_base_offset_pair<1,  36, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 26, 0>::value, vnode_base_offset_pair<1,  36, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1,  36, 26, 1>::value, vnode_base_offset_pair<1,  36, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 26, 2>::value, vnode_base_offset_pair<1,  36, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 27, 0>::value, vnode_base_offset_pair<1,  36, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1,  36, 27, 1>::value, vnode_base_offset_pair<1,  36, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 28, 0>::value, vnode_base_offset_pair<1,  36, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1,  36, 28, 1>::value, vnode_base_offset_pair<1,  36, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 28, 2>::value, vnode_base_offset_pair<1,  36, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 29, 0>::value, vnode_base_offset_pair<1,  36, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1,  36, 29, 1>::value, vnode_base_offset_pair<1,  36, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 29, 2>::value, vnode_base_offset_pair<1,  36, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 30, 0>::value, vnode_base_offset_pair<1,  36, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1,  36, 30, 1>::value, vnode_base_offset_pair<1,  36, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 30, 2>::value, vnode_base_offset_pair<1,  36, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 31, 0>::value, vnode_base_offset_pair<1,  36, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1,  36, 31, 1>::value, vnode_base_offset_pair<1,  36, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 31, 2>::value, vnode_base_offset_pair<1,  36, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 32, 0>::value, vnode_base_offset_pair<1,  36, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1,  36, 32, 1>::value, vnode_base_offset_pair<1,  36, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 32, 2>::value, vnode_base_offset_pair<1,  36, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 33, 0>::value, vnode_base_offset_pair<1,  36, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1,  36, 33, 1>::value, vnode_base_offset_pair<1,  36, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 33, 2>::value, vnode_base_offset_pair<1,  36, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 34, 0>::value, vnode_base_offset_pair<1,  36, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1,  36, 34, 1>::value, vnode_base_offset_pair<1,  36, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 34, 2>::value, vnode_base_offset_pair<1,  36, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 35, 0>::value, vnode_base_offset_pair<1,  36, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1,  36, 35, 1>::value, vnode_base_offset_pair<1,  36, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 35, 2>::value, vnode_base_offset_pair<1,  36, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 36, 0>::value, vnode_base_offset_pair<1,  36, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1,  36, 36, 1>::value, vnode_base_offset_pair<1,  36, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 36, 2>::value, vnode_base_offset_pair<1,  36, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 37, 0>::value, vnode_base_offset_pair<1,  36, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1,  36, 37, 1>::value, vnode_base_offset_pair<1,  36, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 38, 0>::value, vnode_base_offset_pair<1,  36, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1,  36, 38, 1>::value, vnode_base_offset_pair<1,  36, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 38, 2>::value, vnode_base_offset_pair<1,  36, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 39, 0>::value, vnode_base_offset_pair<1,  36, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1,  36, 39, 1>::value, vnode_base_offset_pair<1,  36, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 39, 2>::value, vnode_base_offset_pair<1,  36, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 40, 0>::value, vnode_base_offset_pair<1,  36, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1,  36, 40, 1>::value, vnode_base_offset_pair<1,  36, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 41, 0>::value, vnode_base_offset_pair<1,  36, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1,  36, 41, 1>::value, vnode_base_offset_pair<1,  36, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 41, 2>::value, vnode_base_offset_pair<1,  36, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 42, 0>::value, vnode_base_offset_pair<1,  36, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1,  36, 42, 1>::value, vnode_base_offset_pair<1,  36, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 43, 0>::value, vnode_base_offset_pair<1,  36, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1,  36, 43, 1>::value, vnode_base_offset_pair<1,  36, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 43, 2>::value, vnode_base_offset_pair<1,  36, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 44, 0>::value, vnode_base_offset_pair<1,  36, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1,  36, 44, 1>::value, vnode_base_offset_pair<1,  36, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  36, 44, 2>::value, vnode_base_offset_pair<1,  36, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  36, 45, 0>::value, vnode_base_offset_pair<1,  36, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1,  36, 45, 1>::value, vnode_base_offset_pair<1,  36, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z40_8 =
{
    {
        { vnode_shift_mod_pair<1,  40,  0, 0>::value, vnode_base_offset_pair<1,  40,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1,  40,  0, 1>::value, vnode_base_offset_pair<1,  40,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  0, 2>::value, vnode_base_offset_pair<1,  40,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  0, 3>::value, vnode_base_offset_pair<1,  40,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  0, 4>::value, vnode_base_offset_pair<1,  40,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  0, 5>::value, vnode_base_offset_pair<1,  40,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  0, 6>::value, vnode_base_offset_pair<1,  40,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  0, 7>::value, vnode_base_offset_pair<1,  40,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  0, 8>::value, vnode_base_offset_pair<1,  40,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  0, 9>::value, vnode_base_offset_pair<1,  40,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  40,  1, 0>::value, vnode_base_offset_pair<1,  40,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1,  40,  1, 1>::value, vnode_base_offset_pair<1,  40,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  1, 2>::value, vnode_base_offset_pair<1,  40,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  1, 3>::value, vnode_base_offset_pair<1,  40,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  1, 4>::value, vnode_base_offset_pair<1,  40,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  1, 5>::value, vnode_base_offset_pair<1,  40,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  1, 6>::value, vnode_base_offset_pair<1,  40,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  1, 7>::value, vnode_base_offset_pair<1,  40,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  1, 8>::value, vnode_base_offset_pair<1,  40,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  1, 9>::value, vnode_base_offset_pair<1,  40,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  40,  2, 0>::value, vnode_base_offset_pair<1,  40,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1,  40,  2, 1>::value, vnode_base_offset_pair<1,  40,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  2, 2>::value, vnode_base_offset_pair<1,  40,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  2, 3>::value, vnode_base_offset_pair<1,  40,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  2, 4>::value, vnode_base_offset_pair<1,  40,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  2, 5>::value, vnode_base_offset_pair<1,  40,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  2, 6>::value, vnode_base_offset_pair<1,  40,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  2, 7>::value, vnode_base_offset_pair<1,  40,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  2, 8>::value, vnode_base_offset_pair<1,  40,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  2, 9>::value, vnode_base_offset_pair<1,  40,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  40,  3, 0>::value, vnode_base_offset_pair<1,  40,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1,  40,  3, 1>::value, vnode_base_offset_pair<1,  40,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  3, 2>::value, vnode_base_offset_pair<1,  40,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  3, 3>::value, vnode_base_offset_pair<1,  40,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  3, 4>::value, vnode_base_offset_pair<1,  40,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  3, 5>::value, vnode_base_offset_pair<1,  40,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  3, 6>::value, vnode_base_offset_pair<1,  40,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  3, 7>::value, vnode_base_offset_pair<1,  40,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  3, 8>::value, vnode_base_offset_pair<1,  40,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  3, 9>::value, vnode_base_offset_pair<1,  40,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  40,  4, 0>::value, vnode_base_offset_pair<1,  40,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1,  40,  4, 1>::value, vnode_base_offset_pair<1,  40,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  40,  5, 0>::value, vnode_base_offset_pair<1,  40,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1,  40,  5, 1>::value, vnode_base_offset_pair<1,  40,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  5, 2>::value, vnode_base_offset_pair<1,  40,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  5, 3>::value, vnode_base_offset_pair<1,  40,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  40,  6, 0>::value, vnode_base_offset_pair<1,  40,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1,  40,  6, 1>::value, vnode_base_offset_pair<1,  40,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  6, 2>::value, vnode_base_offset_pair<1,  40,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  6, 3>::value, vnode_base_offset_pair<1,  40,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  6, 4>::value, vnode_base_offset_pair<1,  40,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  40,  7, 0>::value, vnode_base_offset_pair<1,  40,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1,  40,  7, 1>::value, vnode_base_offset_pair<1,  40,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  7, 2>::value, vnode_base_offset_pair<1,  40,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  7, 3>::value, vnode_base_offset_pair<1,  40,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  40,  8, 0>::value, vnode_base_offset_pair<1,  40,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1,  40,  8, 1>::value, vnode_base_offset_pair<1,  40,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  8, 2>::value, vnode_base_offset_pair<1,  40,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  8, 3>::value, vnode_base_offset_pair<1,  40,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  8, 4>::value, vnode_base_offset_pair<1,  40,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  40,  9, 0>::value, vnode_base_offset_pair<1,  40,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1,  40,  9, 1>::value, vnode_base_offset_pair<1,  40,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  9, 2>::value, vnode_base_offset_pair<1,  40,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  9, 3>::value, vnode_base_offset_pair<1,  40,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  40,  9, 4>::value, vnode_base_offset_pair<1,  40,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 10, 0>::value, vnode_base_offset_pair<1,  40, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1,  40, 10, 1>::value, vnode_base_offset_pair<1,  40, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 10, 2>::value, vnode_base_offset_pair<1,  40, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 10, 3>::value, vnode_base_offset_pair<1,  40, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 11, 0>::value, vnode_base_offset_pair<1,  40, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1,  40, 11, 1>::value, vnode_base_offset_pair<1,  40, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 11, 2>::value, vnode_base_offset_pair<1,  40, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 11, 3>::value, vnode_base_offset_pair<1,  40, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 12, 0>::value, vnode_base_offset_pair<1,  40, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1,  40, 12, 1>::value, vnode_base_offset_pair<1,  40, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 12, 2>::value, vnode_base_offset_pair<1,  40, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 12, 3>::value, vnode_base_offset_pair<1,  40, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 13, 0>::value, vnode_base_offset_pair<1,  40, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1,  40, 13, 1>::value, vnode_base_offset_pair<1,  40, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 13, 2>::value, vnode_base_offset_pair<1,  40, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 14, 0>::value, vnode_base_offset_pair<1,  40, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1,  40, 14, 1>::value, vnode_base_offset_pair<1,  40, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 14, 2>::value, vnode_base_offset_pair<1,  40, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 14, 3>::value, vnode_base_offset_pair<1,  40, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 15, 0>::value, vnode_base_offset_pair<1,  40, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1,  40, 15, 1>::value, vnode_base_offset_pair<1,  40, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 15, 2>::value, vnode_base_offset_pair<1,  40, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 15, 3>::value, vnode_base_offset_pair<1,  40, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 16, 0>::value, vnode_base_offset_pair<1,  40, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1,  40, 16, 1>::value, vnode_base_offset_pair<1,  40, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 16, 2>::value, vnode_base_offset_pair<1,  40, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 17, 0>::value, vnode_base_offset_pair<1,  40, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1,  40, 17, 1>::value, vnode_base_offset_pair<1,  40, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 17, 2>::value, vnode_base_offset_pair<1,  40, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 18, 0>::value, vnode_base_offset_pair<1,  40, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1,  40, 18, 1>::value, vnode_base_offset_pair<1,  40, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 18, 2>::value, vnode_base_offset_pair<1,  40, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 19, 0>::value, vnode_base_offset_pair<1,  40, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1,  40, 19, 1>::value, vnode_base_offset_pair<1,  40, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 19, 2>::value, vnode_base_offset_pair<1,  40, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 20, 0>::value, vnode_base_offset_pair<1,  40, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1,  40, 20, 1>::value, vnode_base_offset_pair<1,  40, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 20, 2>::value, vnode_base_offset_pair<1,  40, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 21, 0>::value, vnode_base_offset_pair<1,  40, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1,  40, 21, 1>::value, vnode_base_offset_pair<1,  40, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 21, 2>::value, vnode_base_offset_pair<1,  40, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 22, 0>::value, vnode_base_offset_pair<1,  40, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1,  40, 22, 1>::value, vnode_base_offset_pair<1,  40, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 22, 2>::value, vnode_base_offset_pair<1,  40, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 23, 0>::value, vnode_base_offset_pair<1,  40, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1,  40, 23, 1>::value, vnode_base_offset_pair<1,  40, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 23, 2>::value, vnode_base_offset_pair<1,  40, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 24, 0>::value, vnode_base_offset_pair<1,  40, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1,  40, 24, 1>::value, vnode_base_offset_pair<1,  40, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 24, 2>::value, vnode_base_offset_pair<1,  40, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 25, 0>::value, vnode_base_offset_pair<1,  40, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1,  40, 25, 1>::value, vnode_base_offset_pair<1,  40, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 25, 2>::value, vnode_base_offset_pair<1,  40, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 26, 0>::value, vnode_base_offset_pair<1,  40, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1,  40, 26, 1>::value, vnode_base_offset_pair<1,  40, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 26, 2>::value, vnode_base_offset_pair<1,  40, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 27, 0>::value, vnode_base_offset_pair<1,  40, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1,  40, 27, 1>::value, vnode_base_offset_pair<1,  40, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 28, 0>::value, vnode_base_offset_pair<1,  40, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1,  40, 28, 1>::value, vnode_base_offset_pair<1,  40, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 28, 2>::value, vnode_base_offset_pair<1,  40, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 29, 0>::value, vnode_base_offset_pair<1,  40, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1,  40, 29, 1>::value, vnode_base_offset_pair<1,  40, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 29, 2>::value, vnode_base_offset_pair<1,  40, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 30, 0>::value, vnode_base_offset_pair<1,  40, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1,  40, 30, 1>::value, vnode_base_offset_pair<1,  40, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 30, 2>::value, vnode_base_offset_pair<1,  40, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 31, 0>::value, vnode_base_offset_pair<1,  40, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1,  40, 31, 1>::value, vnode_base_offset_pair<1,  40, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 31, 2>::value, vnode_base_offset_pair<1,  40, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 32, 0>::value, vnode_base_offset_pair<1,  40, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1,  40, 32, 1>::value, vnode_base_offset_pair<1,  40, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 32, 2>::value, vnode_base_offset_pair<1,  40, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 33, 0>::value, vnode_base_offset_pair<1,  40, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1,  40, 33, 1>::value, vnode_base_offset_pair<1,  40, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 33, 2>::value, vnode_base_offset_pair<1,  40, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 34, 0>::value, vnode_base_offset_pair<1,  40, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1,  40, 34, 1>::value, vnode_base_offset_pair<1,  40, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 34, 2>::value, vnode_base_offset_pair<1,  40, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 35, 0>::value, vnode_base_offset_pair<1,  40, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1,  40, 35, 1>::value, vnode_base_offset_pair<1,  40, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 35, 2>::value, vnode_base_offset_pair<1,  40, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 36, 0>::value, vnode_base_offset_pair<1,  40, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1,  40, 36, 1>::value, vnode_base_offset_pair<1,  40, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 36, 2>::value, vnode_base_offset_pair<1,  40, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 37, 0>::value, vnode_base_offset_pair<1,  40, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1,  40, 37, 1>::value, vnode_base_offset_pair<1,  40, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 38, 0>::value, vnode_base_offset_pair<1,  40, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1,  40, 38, 1>::value, vnode_base_offset_pair<1,  40, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 38, 2>::value, vnode_base_offset_pair<1,  40, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 39, 0>::value, vnode_base_offset_pair<1,  40, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1,  40, 39, 1>::value, vnode_base_offset_pair<1,  40, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 39, 2>::value, vnode_base_offset_pair<1,  40, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 40, 0>::value, vnode_base_offset_pair<1,  40, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1,  40, 40, 1>::value, vnode_base_offset_pair<1,  40, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 41, 0>::value, vnode_base_offset_pair<1,  40, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1,  40, 41, 1>::value, vnode_base_offset_pair<1,  40, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 41, 2>::value, vnode_base_offset_pair<1,  40, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 42, 0>::value, vnode_base_offset_pair<1,  40, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1,  40, 42, 1>::value, vnode_base_offset_pair<1,  40, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 43, 0>::value, vnode_base_offset_pair<1,  40, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1,  40, 43, 1>::value, vnode_base_offset_pair<1,  40, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 43, 2>::value, vnode_base_offset_pair<1,  40, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 44, 0>::value, vnode_base_offset_pair<1,  40, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1,  40, 44, 1>::value, vnode_base_offset_pair<1,  40, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  40, 44, 2>::value, vnode_base_offset_pair<1,  40, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  40, 45, 0>::value, vnode_base_offset_pair<1,  40, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1,  40, 45, 1>::value, vnode_base_offset_pair<1,  40, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z44_8 =
{
    {
        { vnode_shift_mod_pair<1,  44,  0, 0>::value, vnode_base_offset_pair<1,  44,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1,  44,  0, 1>::value, vnode_base_offset_pair<1,  44,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  0, 2>::value, vnode_base_offset_pair<1,  44,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  0, 3>::value, vnode_base_offset_pair<1,  44,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  0, 4>::value, vnode_base_offset_pair<1,  44,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  0, 5>::value, vnode_base_offset_pair<1,  44,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  0, 6>::value, vnode_base_offset_pair<1,  44,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  0, 7>::value, vnode_base_offset_pair<1,  44,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  0, 8>::value, vnode_base_offset_pair<1,  44,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  0, 9>::value, vnode_base_offset_pair<1,  44,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  44,  1, 0>::value, vnode_base_offset_pair<1,  44,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1,  44,  1, 1>::value, vnode_base_offset_pair<1,  44,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  1, 2>::value, vnode_base_offset_pair<1,  44,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  1, 3>::value, vnode_base_offset_pair<1,  44,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  1, 4>::value, vnode_base_offset_pair<1,  44,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  1, 5>::value, vnode_base_offset_pair<1,  44,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  1, 6>::value, vnode_base_offset_pair<1,  44,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  1, 7>::value, vnode_base_offset_pair<1,  44,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  1, 8>::value, vnode_base_offset_pair<1,  44,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  1, 9>::value, vnode_base_offset_pair<1,  44,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  44,  2, 0>::value, vnode_base_offset_pair<1,  44,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1,  44,  2, 1>::value, vnode_base_offset_pair<1,  44,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  2, 2>::value, vnode_base_offset_pair<1,  44,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  2, 3>::value, vnode_base_offset_pair<1,  44,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  2, 4>::value, vnode_base_offset_pair<1,  44,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  2, 5>::value, vnode_base_offset_pair<1,  44,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  2, 6>::value, vnode_base_offset_pair<1,  44,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  2, 7>::value, vnode_base_offset_pair<1,  44,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  2, 8>::value, vnode_base_offset_pair<1,  44,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  2, 9>::value, vnode_base_offset_pair<1,  44,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  44,  3, 0>::value, vnode_base_offset_pair<1,  44,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1,  44,  3, 1>::value, vnode_base_offset_pair<1,  44,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  3, 2>::value, vnode_base_offset_pair<1,  44,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  3, 3>::value, vnode_base_offset_pair<1,  44,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  3, 4>::value, vnode_base_offset_pair<1,  44,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  3, 5>::value, vnode_base_offset_pair<1,  44,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  3, 6>::value, vnode_base_offset_pair<1,  44,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  3, 7>::value, vnode_base_offset_pair<1,  44,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  3, 8>::value, vnode_base_offset_pair<1,  44,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  3, 9>::value, vnode_base_offset_pair<1,  44,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  44,  4, 0>::value, vnode_base_offset_pair<1,  44,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1,  44,  4, 1>::value, vnode_base_offset_pair<1,  44,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  44,  5, 0>::value, vnode_base_offset_pair<1,  44,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1,  44,  5, 1>::value, vnode_base_offset_pair<1,  44,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  5, 2>::value, vnode_base_offset_pair<1,  44,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  5, 3>::value, vnode_base_offset_pair<1,  44,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  44,  6, 0>::value, vnode_base_offset_pair<1,  44,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1,  44,  6, 1>::value, vnode_base_offset_pair<1,  44,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  6, 2>::value, vnode_base_offset_pair<1,  44,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  6, 3>::value, vnode_base_offset_pair<1,  44,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  6, 4>::value, vnode_base_offset_pair<1,  44,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  44,  7, 0>::value, vnode_base_offset_pair<1,  44,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1,  44,  7, 1>::value, vnode_base_offset_pair<1,  44,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  7, 2>::value, vnode_base_offset_pair<1,  44,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  7, 3>::value, vnode_base_offset_pair<1,  44,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  44,  8, 0>::value, vnode_base_offset_pair<1,  44,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1,  44,  8, 1>::value, vnode_base_offset_pair<1,  44,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  8, 2>::value, vnode_base_offset_pair<1,  44,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  8, 3>::value, vnode_base_offset_pair<1,  44,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  8, 4>::value, vnode_base_offset_pair<1,  44,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  44,  9, 0>::value, vnode_base_offset_pair<1,  44,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1,  44,  9, 1>::value, vnode_base_offset_pair<1,  44,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  9, 2>::value, vnode_base_offset_pair<1,  44,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  9, 3>::value, vnode_base_offset_pair<1,  44,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  44,  9, 4>::value, vnode_base_offset_pair<1,  44,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 10, 0>::value, vnode_base_offset_pair<1,  44, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1,  44, 10, 1>::value, vnode_base_offset_pair<1,  44, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 10, 2>::value, vnode_base_offset_pair<1,  44, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 10, 3>::value, vnode_base_offset_pair<1,  44, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 11, 0>::value, vnode_base_offset_pair<1,  44, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1,  44, 11, 1>::value, vnode_base_offset_pair<1,  44, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 11, 2>::value, vnode_base_offset_pair<1,  44, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 11, 3>::value, vnode_base_offset_pair<1,  44, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 12, 0>::value, vnode_base_offset_pair<1,  44, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1,  44, 12, 1>::value, vnode_base_offset_pair<1,  44, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 12, 2>::value, vnode_base_offset_pair<1,  44, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 12, 3>::value, vnode_base_offset_pair<1,  44, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 13, 0>::value, vnode_base_offset_pair<1,  44, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1,  44, 13, 1>::value, vnode_base_offset_pair<1,  44, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 13, 2>::value, vnode_base_offset_pair<1,  44, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 14, 0>::value, vnode_base_offset_pair<1,  44, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1,  44, 14, 1>::value, vnode_base_offset_pair<1,  44, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 14, 2>::value, vnode_base_offset_pair<1,  44, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 14, 3>::value, vnode_base_offset_pair<1,  44, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 15, 0>::value, vnode_base_offset_pair<1,  44, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1,  44, 15, 1>::value, vnode_base_offset_pair<1,  44, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 15, 2>::value, vnode_base_offset_pair<1,  44, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 15, 3>::value, vnode_base_offset_pair<1,  44, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 16, 0>::value, vnode_base_offset_pair<1,  44, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1,  44, 16, 1>::value, vnode_base_offset_pair<1,  44, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 16, 2>::value, vnode_base_offset_pair<1,  44, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 17, 0>::value, vnode_base_offset_pair<1,  44, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1,  44, 17, 1>::value, vnode_base_offset_pair<1,  44, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 17, 2>::value, vnode_base_offset_pair<1,  44, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 18, 0>::value, vnode_base_offset_pair<1,  44, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1,  44, 18, 1>::value, vnode_base_offset_pair<1,  44, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 18, 2>::value, vnode_base_offset_pair<1,  44, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 19, 0>::value, vnode_base_offset_pair<1,  44, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1,  44, 19, 1>::value, vnode_base_offset_pair<1,  44, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 19, 2>::value, vnode_base_offset_pair<1,  44, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 20, 0>::value, vnode_base_offset_pair<1,  44, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1,  44, 20, 1>::value, vnode_base_offset_pair<1,  44, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 20, 2>::value, vnode_base_offset_pair<1,  44, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 21, 0>::value, vnode_base_offset_pair<1,  44, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1,  44, 21, 1>::value, vnode_base_offset_pair<1,  44, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 21, 2>::value, vnode_base_offset_pair<1,  44, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 22, 0>::value, vnode_base_offset_pair<1,  44, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1,  44, 22, 1>::value, vnode_base_offset_pair<1,  44, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 22, 2>::value, vnode_base_offset_pair<1,  44, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 23, 0>::value, vnode_base_offset_pair<1,  44, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1,  44, 23, 1>::value, vnode_base_offset_pair<1,  44, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 23, 2>::value, vnode_base_offset_pair<1,  44, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 24, 0>::value, vnode_base_offset_pair<1,  44, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1,  44, 24, 1>::value, vnode_base_offset_pair<1,  44, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 24, 2>::value, vnode_base_offset_pair<1,  44, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 25, 0>::value, vnode_base_offset_pair<1,  44, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1,  44, 25, 1>::value, vnode_base_offset_pair<1,  44, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 25, 2>::value, vnode_base_offset_pair<1,  44, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 26, 0>::value, vnode_base_offset_pair<1,  44, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1,  44, 26, 1>::value, vnode_base_offset_pair<1,  44, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 26, 2>::value, vnode_base_offset_pair<1,  44, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 27, 0>::value, vnode_base_offset_pair<1,  44, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1,  44, 27, 1>::value, vnode_base_offset_pair<1,  44, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 28, 0>::value, vnode_base_offset_pair<1,  44, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1,  44, 28, 1>::value, vnode_base_offset_pair<1,  44, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 28, 2>::value, vnode_base_offset_pair<1,  44, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 29, 0>::value, vnode_base_offset_pair<1,  44, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1,  44, 29, 1>::value, vnode_base_offset_pair<1,  44, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 29, 2>::value, vnode_base_offset_pair<1,  44, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 30, 0>::value, vnode_base_offset_pair<1,  44, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1,  44, 30, 1>::value, vnode_base_offset_pair<1,  44, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 30, 2>::value, vnode_base_offset_pair<1,  44, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 31, 0>::value, vnode_base_offset_pair<1,  44, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1,  44, 31, 1>::value, vnode_base_offset_pair<1,  44, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 31, 2>::value, vnode_base_offset_pair<1,  44, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 32, 0>::value, vnode_base_offset_pair<1,  44, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1,  44, 32, 1>::value, vnode_base_offset_pair<1,  44, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 32, 2>::value, vnode_base_offset_pair<1,  44, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 33, 0>::value, vnode_base_offset_pair<1,  44, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1,  44, 33, 1>::value, vnode_base_offset_pair<1,  44, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 33, 2>::value, vnode_base_offset_pair<1,  44, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 34, 0>::value, vnode_base_offset_pair<1,  44, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1,  44, 34, 1>::value, vnode_base_offset_pair<1,  44, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 34, 2>::value, vnode_base_offset_pair<1,  44, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 35, 0>::value, vnode_base_offset_pair<1,  44, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1,  44, 35, 1>::value, vnode_base_offset_pair<1,  44, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 35, 2>::value, vnode_base_offset_pair<1,  44, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 36, 0>::value, vnode_base_offset_pair<1,  44, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1,  44, 36, 1>::value, vnode_base_offset_pair<1,  44, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 36, 2>::value, vnode_base_offset_pair<1,  44, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 37, 0>::value, vnode_base_offset_pair<1,  44, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1,  44, 37, 1>::value, vnode_base_offset_pair<1,  44, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 38, 0>::value, vnode_base_offset_pair<1,  44, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1,  44, 38, 1>::value, vnode_base_offset_pair<1,  44, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 38, 2>::value, vnode_base_offset_pair<1,  44, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 39, 0>::value, vnode_base_offset_pair<1,  44, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1,  44, 39, 1>::value, vnode_base_offset_pair<1,  44, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 39, 2>::value, vnode_base_offset_pair<1,  44, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 40, 0>::value, vnode_base_offset_pair<1,  44, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1,  44, 40, 1>::value, vnode_base_offset_pair<1,  44, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 41, 0>::value, vnode_base_offset_pair<1,  44, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1,  44, 41, 1>::value, vnode_base_offset_pair<1,  44, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 41, 2>::value, vnode_base_offset_pair<1,  44, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 42, 0>::value, vnode_base_offset_pair<1,  44, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1,  44, 42, 1>::value, vnode_base_offset_pair<1,  44, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 43, 0>::value, vnode_base_offset_pair<1,  44, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1,  44, 43, 1>::value, vnode_base_offset_pair<1,  44, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 43, 2>::value, vnode_base_offset_pair<1,  44, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 44, 0>::value, vnode_base_offset_pair<1,  44, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1,  44, 44, 1>::value, vnode_base_offset_pair<1,  44, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  44, 44, 2>::value, vnode_base_offset_pair<1,  44, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  44, 45, 0>::value, vnode_base_offset_pair<1,  44, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1,  44, 45, 1>::value, vnode_base_offset_pair<1,  44, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z48_8 =
{
    {
        { vnode_shift_mod_pair<1,  48,  0, 0>::value, vnode_base_offset_pair<1,  48,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1,  48,  0, 1>::value, vnode_base_offset_pair<1,  48,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  0, 2>::value, vnode_base_offset_pair<1,  48,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  0, 3>::value, vnode_base_offset_pair<1,  48,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  0, 4>::value, vnode_base_offset_pair<1,  48,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  0, 5>::value, vnode_base_offset_pair<1,  48,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  0, 6>::value, vnode_base_offset_pair<1,  48,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  0, 7>::value, vnode_base_offset_pair<1,  48,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  0, 8>::value, vnode_base_offset_pair<1,  48,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  0, 9>::value, vnode_base_offset_pair<1,  48,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  48,  1, 0>::value, vnode_base_offset_pair<1,  48,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1,  48,  1, 1>::value, vnode_base_offset_pair<1,  48,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  1, 2>::value, vnode_base_offset_pair<1,  48,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  1, 3>::value, vnode_base_offset_pair<1,  48,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  1, 4>::value, vnode_base_offset_pair<1,  48,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  1, 5>::value, vnode_base_offset_pair<1,  48,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  1, 6>::value, vnode_base_offset_pair<1,  48,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  1, 7>::value, vnode_base_offset_pair<1,  48,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  1, 8>::value, vnode_base_offset_pair<1,  48,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  1, 9>::value, vnode_base_offset_pair<1,  48,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  48,  2, 0>::value, vnode_base_offset_pair<1,  48,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1,  48,  2, 1>::value, vnode_base_offset_pair<1,  48,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  2, 2>::value, vnode_base_offset_pair<1,  48,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  2, 3>::value, vnode_base_offset_pair<1,  48,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  2, 4>::value, vnode_base_offset_pair<1,  48,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  2, 5>::value, vnode_base_offset_pair<1,  48,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  2, 6>::value, vnode_base_offset_pair<1,  48,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  2, 7>::value, vnode_base_offset_pair<1,  48,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  2, 8>::value, vnode_base_offset_pair<1,  48,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  2, 9>::value, vnode_base_offset_pair<1,  48,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  48,  3, 0>::value, vnode_base_offset_pair<1,  48,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1,  48,  3, 1>::value, vnode_base_offset_pair<1,  48,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  3, 2>::value, vnode_base_offset_pair<1,  48,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  3, 3>::value, vnode_base_offset_pair<1,  48,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  3, 4>::value, vnode_base_offset_pair<1,  48,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  3, 5>::value, vnode_base_offset_pair<1,  48,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  3, 6>::value, vnode_base_offset_pair<1,  48,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  3, 7>::value, vnode_base_offset_pair<1,  48,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  3, 8>::value, vnode_base_offset_pair<1,  48,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  3, 9>::value, vnode_base_offset_pair<1,  48,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  48,  4, 0>::value, vnode_base_offset_pair<1,  48,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1,  48,  4, 1>::value, vnode_base_offset_pair<1,  48,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  48,  5, 0>::value, vnode_base_offset_pair<1,  48,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1,  48,  5, 1>::value, vnode_base_offset_pair<1,  48,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  5, 2>::value, vnode_base_offset_pair<1,  48,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  5, 3>::value, vnode_base_offset_pair<1,  48,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  48,  6, 0>::value, vnode_base_offset_pair<1,  48,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1,  48,  6, 1>::value, vnode_base_offset_pair<1,  48,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  6, 2>::value, vnode_base_offset_pair<1,  48,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  6, 3>::value, vnode_base_offset_pair<1,  48,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  6, 4>::value, vnode_base_offset_pair<1,  48,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  48,  7, 0>::value, vnode_base_offset_pair<1,  48,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1,  48,  7, 1>::value, vnode_base_offset_pair<1,  48,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  7, 2>::value, vnode_base_offset_pair<1,  48,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  7, 3>::value, vnode_base_offset_pair<1,  48,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  48,  8, 0>::value, vnode_base_offset_pair<1,  48,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1,  48,  8, 1>::value, vnode_base_offset_pair<1,  48,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  8, 2>::value, vnode_base_offset_pair<1,  48,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  8, 3>::value, vnode_base_offset_pair<1,  48,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  8, 4>::value, vnode_base_offset_pair<1,  48,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  48,  9, 0>::value, vnode_base_offset_pair<1,  48,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1,  48,  9, 1>::value, vnode_base_offset_pair<1,  48,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  9, 2>::value, vnode_base_offset_pair<1,  48,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  9, 3>::value, vnode_base_offset_pair<1,  48,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  48,  9, 4>::value, vnode_base_offset_pair<1,  48,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 10, 0>::value, vnode_base_offset_pair<1,  48, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1,  48, 10, 1>::value, vnode_base_offset_pair<1,  48, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 10, 2>::value, vnode_base_offset_pair<1,  48, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 10, 3>::value, vnode_base_offset_pair<1,  48, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 11, 0>::value, vnode_base_offset_pair<1,  48, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1,  48, 11, 1>::value, vnode_base_offset_pair<1,  48, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 11, 2>::value, vnode_base_offset_pair<1,  48, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 11, 3>::value, vnode_base_offset_pair<1,  48, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 12, 0>::value, vnode_base_offset_pair<1,  48, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1,  48, 12, 1>::value, vnode_base_offset_pair<1,  48, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 12, 2>::value, vnode_base_offset_pair<1,  48, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 12, 3>::value, vnode_base_offset_pair<1,  48, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 13, 0>::value, vnode_base_offset_pair<1,  48, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1,  48, 13, 1>::value, vnode_base_offset_pair<1,  48, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 13, 2>::value, vnode_base_offset_pair<1,  48, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 14, 0>::value, vnode_base_offset_pair<1,  48, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1,  48, 14, 1>::value, vnode_base_offset_pair<1,  48, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 14, 2>::value, vnode_base_offset_pair<1,  48, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 14, 3>::value, vnode_base_offset_pair<1,  48, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 15, 0>::value, vnode_base_offset_pair<1,  48, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1,  48, 15, 1>::value, vnode_base_offset_pair<1,  48, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 15, 2>::value, vnode_base_offset_pair<1,  48, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 15, 3>::value, vnode_base_offset_pair<1,  48, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 16, 0>::value, vnode_base_offset_pair<1,  48, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1,  48, 16, 1>::value, vnode_base_offset_pair<1,  48, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 16, 2>::value, vnode_base_offset_pair<1,  48, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 17, 0>::value, vnode_base_offset_pair<1,  48, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1,  48, 17, 1>::value, vnode_base_offset_pair<1,  48, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 17, 2>::value, vnode_base_offset_pair<1,  48, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 18, 0>::value, vnode_base_offset_pair<1,  48, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1,  48, 18, 1>::value, vnode_base_offset_pair<1,  48, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 18, 2>::value, vnode_base_offset_pair<1,  48, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 19, 0>::value, vnode_base_offset_pair<1,  48, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1,  48, 19, 1>::value, vnode_base_offset_pair<1,  48, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 19, 2>::value, vnode_base_offset_pair<1,  48, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 20, 0>::value, vnode_base_offset_pair<1,  48, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1,  48, 20, 1>::value, vnode_base_offset_pair<1,  48, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 20, 2>::value, vnode_base_offset_pair<1,  48, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 21, 0>::value, vnode_base_offset_pair<1,  48, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1,  48, 21, 1>::value, vnode_base_offset_pair<1,  48, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 21, 2>::value, vnode_base_offset_pair<1,  48, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 22, 0>::value, vnode_base_offset_pair<1,  48, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1,  48, 22, 1>::value, vnode_base_offset_pair<1,  48, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 22, 2>::value, vnode_base_offset_pair<1,  48, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 23, 0>::value, vnode_base_offset_pair<1,  48, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1,  48, 23, 1>::value, vnode_base_offset_pair<1,  48, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 23, 2>::value, vnode_base_offset_pair<1,  48, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 24, 0>::value, vnode_base_offset_pair<1,  48, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1,  48, 24, 1>::value, vnode_base_offset_pair<1,  48, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 24, 2>::value, vnode_base_offset_pair<1,  48, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 25, 0>::value, vnode_base_offset_pair<1,  48, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1,  48, 25, 1>::value, vnode_base_offset_pair<1,  48, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 25, 2>::value, vnode_base_offset_pair<1,  48, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 26, 0>::value, vnode_base_offset_pair<1,  48, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1,  48, 26, 1>::value, vnode_base_offset_pair<1,  48, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 26, 2>::value, vnode_base_offset_pair<1,  48, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 27, 0>::value, vnode_base_offset_pair<1,  48, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1,  48, 27, 1>::value, vnode_base_offset_pair<1,  48, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 28, 0>::value, vnode_base_offset_pair<1,  48, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1,  48, 28, 1>::value, vnode_base_offset_pair<1,  48, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 28, 2>::value, vnode_base_offset_pair<1,  48, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 29, 0>::value, vnode_base_offset_pair<1,  48, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1,  48, 29, 1>::value, vnode_base_offset_pair<1,  48, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 29, 2>::value, vnode_base_offset_pair<1,  48, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 30, 0>::value, vnode_base_offset_pair<1,  48, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1,  48, 30, 1>::value, vnode_base_offset_pair<1,  48, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 30, 2>::value, vnode_base_offset_pair<1,  48, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 31, 0>::value, vnode_base_offset_pair<1,  48, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1,  48, 31, 1>::value, vnode_base_offset_pair<1,  48, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 31, 2>::value, vnode_base_offset_pair<1,  48, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 32, 0>::value, vnode_base_offset_pair<1,  48, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1,  48, 32, 1>::value, vnode_base_offset_pair<1,  48, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 32, 2>::value, vnode_base_offset_pair<1,  48, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 33, 0>::value, vnode_base_offset_pair<1,  48, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1,  48, 33, 1>::value, vnode_base_offset_pair<1,  48, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 33, 2>::value, vnode_base_offset_pair<1,  48, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 34, 0>::value, vnode_base_offset_pair<1,  48, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1,  48, 34, 1>::value, vnode_base_offset_pair<1,  48, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 34, 2>::value, vnode_base_offset_pair<1,  48, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 35, 0>::value, vnode_base_offset_pair<1,  48, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1,  48, 35, 1>::value, vnode_base_offset_pair<1,  48, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 35, 2>::value, vnode_base_offset_pair<1,  48, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 36, 0>::value, vnode_base_offset_pair<1,  48, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1,  48, 36, 1>::value, vnode_base_offset_pair<1,  48, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 36, 2>::value, vnode_base_offset_pair<1,  48, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 37, 0>::value, vnode_base_offset_pair<1,  48, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1,  48, 37, 1>::value, vnode_base_offset_pair<1,  48, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 38, 0>::value, vnode_base_offset_pair<1,  48, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1,  48, 38, 1>::value, vnode_base_offset_pair<1,  48, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 38, 2>::value, vnode_base_offset_pair<1,  48, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 39, 0>::value, vnode_base_offset_pair<1,  48, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1,  48, 39, 1>::value, vnode_base_offset_pair<1,  48, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 39, 2>::value, vnode_base_offset_pair<1,  48, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 40, 0>::value, vnode_base_offset_pair<1,  48, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1,  48, 40, 1>::value, vnode_base_offset_pair<1,  48, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 41, 0>::value, vnode_base_offset_pair<1,  48, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1,  48, 41, 1>::value, vnode_base_offset_pair<1,  48, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 41, 2>::value, vnode_base_offset_pair<1,  48, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 42, 0>::value, vnode_base_offset_pair<1,  48, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1,  48, 42, 1>::value, vnode_base_offset_pair<1,  48, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 43, 0>::value, vnode_base_offset_pair<1,  48, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1,  48, 43, 1>::value, vnode_base_offset_pair<1,  48, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 43, 2>::value, vnode_base_offset_pair<1,  48, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 44, 0>::value, vnode_base_offset_pair<1,  48, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1,  48, 44, 1>::value, vnode_base_offset_pair<1,  48, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  48, 44, 2>::value, vnode_base_offset_pair<1,  48, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  48, 45, 0>::value, vnode_base_offset_pair<1,  48, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1,  48, 45, 1>::value, vnode_base_offset_pair<1,  48, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z52_8 =
{
    {
        { vnode_shift_mod_pair<1,  52,  0, 0>::value, vnode_base_offset_pair<1,  52,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1,  52,  0, 1>::value, vnode_base_offset_pair<1,  52,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  0, 2>::value, vnode_base_offset_pair<1,  52,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  0, 3>::value, vnode_base_offset_pair<1,  52,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  0, 4>::value, vnode_base_offset_pair<1,  52,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  0, 5>::value, vnode_base_offset_pair<1,  52,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  0, 6>::value, vnode_base_offset_pair<1,  52,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  0, 7>::value, vnode_base_offset_pair<1,  52,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  0, 8>::value, vnode_base_offset_pair<1,  52,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  0, 9>::value, vnode_base_offset_pair<1,  52,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  52,  1, 0>::value, vnode_base_offset_pair<1,  52,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1,  52,  1, 1>::value, vnode_base_offset_pair<1,  52,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  1, 2>::value, vnode_base_offset_pair<1,  52,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  1, 3>::value, vnode_base_offset_pair<1,  52,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  1, 4>::value, vnode_base_offset_pair<1,  52,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  1, 5>::value, vnode_base_offset_pair<1,  52,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  1, 6>::value, vnode_base_offset_pair<1,  52,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  1, 7>::value, vnode_base_offset_pair<1,  52,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  1, 8>::value, vnode_base_offset_pair<1,  52,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  1, 9>::value, vnode_base_offset_pair<1,  52,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  52,  2, 0>::value, vnode_base_offset_pair<1,  52,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1,  52,  2, 1>::value, vnode_base_offset_pair<1,  52,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  2, 2>::value, vnode_base_offset_pair<1,  52,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  2, 3>::value, vnode_base_offset_pair<1,  52,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  2, 4>::value, vnode_base_offset_pair<1,  52,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  2, 5>::value, vnode_base_offset_pair<1,  52,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  2, 6>::value, vnode_base_offset_pair<1,  52,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  2, 7>::value, vnode_base_offset_pair<1,  52,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  2, 8>::value, vnode_base_offset_pair<1,  52,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  2, 9>::value, vnode_base_offset_pair<1,  52,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  52,  3, 0>::value, vnode_base_offset_pair<1,  52,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1,  52,  3, 1>::value, vnode_base_offset_pair<1,  52,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  3, 2>::value, vnode_base_offset_pair<1,  52,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  3, 3>::value, vnode_base_offset_pair<1,  52,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  3, 4>::value, vnode_base_offset_pair<1,  52,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  3, 5>::value, vnode_base_offset_pair<1,  52,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  3, 6>::value, vnode_base_offset_pair<1,  52,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  3, 7>::value, vnode_base_offset_pair<1,  52,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  3, 8>::value, vnode_base_offset_pair<1,  52,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  3, 9>::value, vnode_base_offset_pair<1,  52,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  52,  4, 0>::value, vnode_base_offset_pair<1,  52,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1,  52,  4, 1>::value, vnode_base_offset_pair<1,  52,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  52,  5, 0>::value, vnode_base_offset_pair<1,  52,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1,  52,  5, 1>::value, vnode_base_offset_pair<1,  52,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  5, 2>::value, vnode_base_offset_pair<1,  52,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  5, 3>::value, vnode_base_offset_pair<1,  52,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  52,  6, 0>::value, vnode_base_offset_pair<1,  52,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1,  52,  6, 1>::value, vnode_base_offset_pair<1,  52,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  6, 2>::value, vnode_base_offset_pair<1,  52,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  6, 3>::value, vnode_base_offset_pair<1,  52,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  6, 4>::value, vnode_base_offset_pair<1,  52,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  52,  7, 0>::value, vnode_base_offset_pair<1,  52,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1,  52,  7, 1>::value, vnode_base_offset_pair<1,  52,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  7, 2>::value, vnode_base_offset_pair<1,  52,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  7, 3>::value, vnode_base_offset_pair<1,  52,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  52,  8, 0>::value, vnode_base_offset_pair<1,  52,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1,  52,  8, 1>::value, vnode_base_offset_pair<1,  52,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  8, 2>::value, vnode_base_offset_pair<1,  52,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  8, 3>::value, vnode_base_offset_pair<1,  52,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  8, 4>::value, vnode_base_offset_pair<1,  52,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  52,  9, 0>::value, vnode_base_offset_pair<1,  52,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1,  52,  9, 1>::value, vnode_base_offset_pair<1,  52,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  9, 2>::value, vnode_base_offset_pair<1,  52,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  9, 3>::value, vnode_base_offset_pair<1,  52,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  52,  9, 4>::value, vnode_base_offset_pair<1,  52,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 10, 0>::value, vnode_base_offset_pair<1,  52, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1,  52, 10, 1>::value, vnode_base_offset_pair<1,  52, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 10, 2>::value, vnode_base_offset_pair<1,  52, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 10, 3>::value, vnode_base_offset_pair<1,  52, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 11, 0>::value, vnode_base_offset_pair<1,  52, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1,  52, 11, 1>::value, vnode_base_offset_pair<1,  52, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 11, 2>::value, vnode_base_offset_pair<1,  52, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 11, 3>::value, vnode_base_offset_pair<1,  52, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 12, 0>::value, vnode_base_offset_pair<1,  52, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1,  52, 12, 1>::value, vnode_base_offset_pair<1,  52, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 12, 2>::value, vnode_base_offset_pair<1,  52, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 12, 3>::value, vnode_base_offset_pair<1,  52, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 13, 0>::value, vnode_base_offset_pair<1,  52, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1,  52, 13, 1>::value, vnode_base_offset_pair<1,  52, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 13, 2>::value, vnode_base_offset_pair<1,  52, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 14, 0>::value, vnode_base_offset_pair<1,  52, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1,  52, 14, 1>::value, vnode_base_offset_pair<1,  52, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 14, 2>::value, vnode_base_offset_pair<1,  52, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 14, 3>::value, vnode_base_offset_pair<1,  52, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 15, 0>::value, vnode_base_offset_pair<1,  52, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1,  52, 15, 1>::value, vnode_base_offset_pair<1,  52, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 15, 2>::value, vnode_base_offset_pair<1,  52, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 15, 3>::value, vnode_base_offset_pair<1,  52, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 16, 0>::value, vnode_base_offset_pair<1,  52, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1,  52, 16, 1>::value, vnode_base_offset_pair<1,  52, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 16, 2>::value, vnode_base_offset_pair<1,  52, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 17, 0>::value, vnode_base_offset_pair<1,  52, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1,  52, 17, 1>::value, vnode_base_offset_pair<1,  52, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 17, 2>::value, vnode_base_offset_pair<1,  52, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 18, 0>::value, vnode_base_offset_pair<1,  52, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1,  52, 18, 1>::value, vnode_base_offset_pair<1,  52, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 18, 2>::value, vnode_base_offset_pair<1,  52, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 19, 0>::value, vnode_base_offset_pair<1,  52, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1,  52, 19, 1>::value, vnode_base_offset_pair<1,  52, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 19, 2>::value, vnode_base_offset_pair<1,  52, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 20, 0>::value, vnode_base_offset_pair<1,  52, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1,  52, 20, 1>::value, vnode_base_offset_pair<1,  52, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 20, 2>::value, vnode_base_offset_pair<1,  52, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 21, 0>::value, vnode_base_offset_pair<1,  52, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1,  52, 21, 1>::value, vnode_base_offset_pair<1,  52, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 21, 2>::value, vnode_base_offset_pair<1,  52, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 22, 0>::value, vnode_base_offset_pair<1,  52, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1,  52, 22, 1>::value, vnode_base_offset_pair<1,  52, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 22, 2>::value, vnode_base_offset_pair<1,  52, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 23, 0>::value, vnode_base_offset_pair<1,  52, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1,  52, 23, 1>::value, vnode_base_offset_pair<1,  52, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 23, 2>::value, vnode_base_offset_pair<1,  52, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 24, 0>::value, vnode_base_offset_pair<1,  52, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1,  52, 24, 1>::value, vnode_base_offset_pair<1,  52, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 24, 2>::value, vnode_base_offset_pair<1,  52, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 25, 0>::value, vnode_base_offset_pair<1,  52, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1,  52, 25, 1>::value, vnode_base_offset_pair<1,  52, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 25, 2>::value, vnode_base_offset_pair<1,  52, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 26, 0>::value, vnode_base_offset_pair<1,  52, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1,  52, 26, 1>::value, vnode_base_offset_pair<1,  52, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 26, 2>::value, vnode_base_offset_pair<1,  52, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 27, 0>::value, vnode_base_offset_pair<1,  52, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1,  52, 27, 1>::value, vnode_base_offset_pair<1,  52, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 28, 0>::value, vnode_base_offset_pair<1,  52, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1,  52, 28, 1>::value, vnode_base_offset_pair<1,  52, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 28, 2>::value, vnode_base_offset_pair<1,  52, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 29, 0>::value, vnode_base_offset_pair<1,  52, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1,  52, 29, 1>::value, vnode_base_offset_pair<1,  52, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 29, 2>::value, vnode_base_offset_pair<1,  52, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 30, 0>::value, vnode_base_offset_pair<1,  52, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1,  52, 30, 1>::value, vnode_base_offset_pair<1,  52, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 30, 2>::value, vnode_base_offset_pair<1,  52, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 31, 0>::value, vnode_base_offset_pair<1,  52, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1,  52, 31, 1>::value, vnode_base_offset_pair<1,  52, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 31, 2>::value, vnode_base_offset_pair<1,  52, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 32, 0>::value, vnode_base_offset_pair<1,  52, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1,  52, 32, 1>::value, vnode_base_offset_pair<1,  52, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 32, 2>::value, vnode_base_offset_pair<1,  52, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 33, 0>::value, vnode_base_offset_pair<1,  52, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1,  52, 33, 1>::value, vnode_base_offset_pair<1,  52, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 33, 2>::value, vnode_base_offset_pair<1,  52, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 34, 0>::value, vnode_base_offset_pair<1,  52, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1,  52, 34, 1>::value, vnode_base_offset_pair<1,  52, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 34, 2>::value, vnode_base_offset_pair<1,  52, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 35, 0>::value, vnode_base_offset_pair<1,  52, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1,  52, 35, 1>::value, vnode_base_offset_pair<1,  52, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 35, 2>::value, vnode_base_offset_pair<1,  52, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 36, 0>::value, vnode_base_offset_pair<1,  52, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1,  52, 36, 1>::value, vnode_base_offset_pair<1,  52, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 36, 2>::value, vnode_base_offset_pair<1,  52, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 37, 0>::value, vnode_base_offset_pair<1,  52, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1,  52, 37, 1>::value, vnode_base_offset_pair<1,  52, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 38, 0>::value, vnode_base_offset_pair<1,  52, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1,  52, 38, 1>::value, vnode_base_offset_pair<1,  52, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 38, 2>::value, vnode_base_offset_pair<1,  52, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 39, 0>::value, vnode_base_offset_pair<1,  52, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1,  52, 39, 1>::value, vnode_base_offset_pair<1,  52, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 39, 2>::value, vnode_base_offset_pair<1,  52, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 40, 0>::value, vnode_base_offset_pair<1,  52, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1,  52, 40, 1>::value, vnode_base_offset_pair<1,  52, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 41, 0>::value, vnode_base_offset_pair<1,  52, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1,  52, 41, 1>::value, vnode_base_offset_pair<1,  52, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 41, 2>::value, vnode_base_offset_pair<1,  52, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 42, 0>::value, vnode_base_offset_pair<1,  52, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1,  52, 42, 1>::value, vnode_base_offset_pair<1,  52, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 43, 0>::value, vnode_base_offset_pair<1,  52, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1,  52, 43, 1>::value, vnode_base_offset_pair<1,  52, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 43, 2>::value, vnode_base_offset_pair<1,  52, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 44, 0>::value, vnode_base_offset_pair<1,  52, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1,  52, 44, 1>::value, vnode_base_offset_pair<1,  52, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  52, 44, 2>::value, vnode_base_offset_pair<1,  52, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  52, 45, 0>::value, vnode_base_offset_pair<1,  52, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1,  52, 45, 1>::value, vnode_base_offset_pair<1,  52, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z56_8 =
{
    {
        { vnode_shift_mod_pair<1,  56,  0, 0>::value, vnode_base_offset_pair<1,  56,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1,  56,  0, 1>::value, vnode_base_offset_pair<1,  56,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  0, 2>::value, vnode_base_offset_pair<1,  56,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  0, 3>::value, vnode_base_offset_pair<1,  56,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  0, 4>::value, vnode_base_offset_pair<1,  56,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  0, 5>::value, vnode_base_offset_pair<1,  56,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  0, 6>::value, vnode_base_offset_pair<1,  56,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  0, 7>::value, vnode_base_offset_pair<1,  56,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  0, 8>::value, vnode_base_offset_pair<1,  56,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  0, 9>::value, vnode_base_offset_pair<1,  56,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  56,  1, 0>::value, vnode_base_offset_pair<1,  56,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1,  56,  1, 1>::value, vnode_base_offset_pair<1,  56,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  1, 2>::value, vnode_base_offset_pair<1,  56,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  1, 3>::value, vnode_base_offset_pair<1,  56,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  1, 4>::value, vnode_base_offset_pair<1,  56,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  1, 5>::value, vnode_base_offset_pair<1,  56,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  1, 6>::value, vnode_base_offset_pair<1,  56,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  1, 7>::value, vnode_base_offset_pair<1,  56,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  1, 8>::value, vnode_base_offset_pair<1,  56,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  1, 9>::value, vnode_base_offset_pair<1,  56,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  56,  2, 0>::value, vnode_base_offset_pair<1,  56,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1,  56,  2, 1>::value, vnode_base_offset_pair<1,  56,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  2, 2>::value, vnode_base_offset_pair<1,  56,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  2, 3>::value, vnode_base_offset_pair<1,  56,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  2, 4>::value, vnode_base_offset_pair<1,  56,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  2, 5>::value, vnode_base_offset_pair<1,  56,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  2, 6>::value, vnode_base_offset_pair<1,  56,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  2, 7>::value, vnode_base_offset_pair<1,  56,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  2, 8>::value, vnode_base_offset_pair<1,  56,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  2, 9>::value, vnode_base_offset_pair<1,  56,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  56,  3, 0>::value, vnode_base_offset_pair<1,  56,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1,  56,  3, 1>::value, vnode_base_offset_pair<1,  56,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  3, 2>::value, vnode_base_offset_pair<1,  56,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  3, 3>::value, vnode_base_offset_pair<1,  56,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  3, 4>::value, vnode_base_offset_pair<1,  56,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  3, 5>::value, vnode_base_offset_pair<1,  56,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  3, 6>::value, vnode_base_offset_pair<1,  56,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  3, 7>::value, vnode_base_offset_pair<1,  56,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  3, 8>::value, vnode_base_offset_pair<1,  56,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  3, 9>::value, vnode_base_offset_pair<1,  56,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  56,  4, 0>::value, vnode_base_offset_pair<1,  56,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1,  56,  4, 1>::value, vnode_base_offset_pair<1,  56,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  56,  5, 0>::value, vnode_base_offset_pair<1,  56,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1,  56,  5, 1>::value, vnode_base_offset_pair<1,  56,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  5, 2>::value, vnode_base_offset_pair<1,  56,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  5, 3>::value, vnode_base_offset_pair<1,  56,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  56,  6, 0>::value, vnode_base_offset_pair<1,  56,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1,  56,  6, 1>::value, vnode_base_offset_pair<1,  56,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  6, 2>::value, vnode_base_offset_pair<1,  56,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  6, 3>::value, vnode_base_offset_pair<1,  56,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  6, 4>::value, vnode_base_offset_pair<1,  56,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  56,  7, 0>::value, vnode_base_offset_pair<1,  56,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1,  56,  7, 1>::value, vnode_base_offset_pair<1,  56,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  7, 2>::value, vnode_base_offset_pair<1,  56,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  7, 3>::value, vnode_base_offset_pair<1,  56,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  56,  8, 0>::value, vnode_base_offset_pair<1,  56,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1,  56,  8, 1>::value, vnode_base_offset_pair<1,  56,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  8, 2>::value, vnode_base_offset_pair<1,  56,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  8, 3>::value, vnode_base_offset_pair<1,  56,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  8, 4>::value, vnode_base_offset_pair<1,  56,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  56,  9, 0>::value, vnode_base_offset_pair<1,  56,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1,  56,  9, 1>::value, vnode_base_offset_pair<1,  56,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  9, 2>::value, vnode_base_offset_pair<1,  56,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  9, 3>::value, vnode_base_offset_pair<1,  56,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  56,  9, 4>::value, vnode_base_offset_pair<1,  56,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 10, 0>::value, vnode_base_offset_pair<1,  56, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1,  56, 10, 1>::value, vnode_base_offset_pair<1,  56, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 10, 2>::value, vnode_base_offset_pair<1,  56, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 10, 3>::value, vnode_base_offset_pair<1,  56, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 11, 0>::value, vnode_base_offset_pair<1,  56, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1,  56, 11, 1>::value, vnode_base_offset_pair<1,  56, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 11, 2>::value, vnode_base_offset_pair<1,  56, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 11, 3>::value, vnode_base_offset_pair<1,  56, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 12, 0>::value, vnode_base_offset_pair<1,  56, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1,  56, 12, 1>::value, vnode_base_offset_pair<1,  56, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 12, 2>::value, vnode_base_offset_pair<1,  56, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 12, 3>::value, vnode_base_offset_pair<1,  56, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 13, 0>::value, vnode_base_offset_pair<1,  56, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1,  56, 13, 1>::value, vnode_base_offset_pair<1,  56, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 13, 2>::value, vnode_base_offset_pair<1,  56, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 14, 0>::value, vnode_base_offset_pair<1,  56, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1,  56, 14, 1>::value, vnode_base_offset_pair<1,  56, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 14, 2>::value, vnode_base_offset_pair<1,  56, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 14, 3>::value, vnode_base_offset_pair<1,  56, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 15, 0>::value, vnode_base_offset_pair<1,  56, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1,  56, 15, 1>::value, vnode_base_offset_pair<1,  56, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 15, 2>::value, vnode_base_offset_pair<1,  56, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 15, 3>::value, vnode_base_offset_pair<1,  56, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 16, 0>::value, vnode_base_offset_pair<1,  56, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1,  56, 16, 1>::value, vnode_base_offset_pair<1,  56, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 16, 2>::value, vnode_base_offset_pair<1,  56, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 17, 0>::value, vnode_base_offset_pair<1,  56, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1,  56, 17, 1>::value, vnode_base_offset_pair<1,  56, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 17, 2>::value, vnode_base_offset_pair<1,  56, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 18, 0>::value, vnode_base_offset_pair<1,  56, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1,  56, 18, 1>::value, vnode_base_offset_pair<1,  56, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 18, 2>::value, vnode_base_offset_pair<1,  56, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 19, 0>::value, vnode_base_offset_pair<1,  56, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1,  56, 19, 1>::value, vnode_base_offset_pair<1,  56, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 19, 2>::value, vnode_base_offset_pair<1,  56, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 20, 0>::value, vnode_base_offset_pair<1,  56, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1,  56, 20, 1>::value, vnode_base_offset_pair<1,  56, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 20, 2>::value, vnode_base_offset_pair<1,  56, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 21, 0>::value, vnode_base_offset_pair<1,  56, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1,  56, 21, 1>::value, vnode_base_offset_pair<1,  56, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 21, 2>::value, vnode_base_offset_pair<1,  56, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 22, 0>::value, vnode_base_offset_pair<1,  56, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1,  56, 22, 1>::value, vnode_base_offset_pair<1,  56, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 22, 2>::value, vnode_base_offset_pair<1,  56, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 23, 0>::value, vnode_base_offset_pair<1,  56, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1,  56, 23, 1>::value, vnode_base_offset_pair<1,  56, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 23, 2>::value, vnode_base_offset_pair<1,  56, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 24, 0>::value, vnode_base_offset_pair<1,  56, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1,  56, 24, 1>::value, vnode_base_offset_pair<1,  56, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 24, 2>::value, vnode_base_offset_pair<1,  56, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 25, 0>::value, vnode_base_offset_pair<1,  56, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1,  56, 25, 1>::value, vnode_base_offset_pair<1,  56, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 25, 2>::value, vnode_base_offset_pair<1,  56, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 26, 0>::value, vnode_base_offset_pair<1,  56, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1,  56, 26, 1>::value, vnode_base_offset_pair<1,  56, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 26, 2>::value, vnode_base_offset_pair<1,  56, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 27, 0>::value, vnode_base_offset_pair<1,  56, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1,  56, 27, 1>::value, vnode_base_offset_pair<1,  56, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 28, 0>::value, vnode_base_offset_pair<1,  56, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1,  56, 28, 1>::value, vnode_base_offset_pair<1,  56, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 28, 2>::value, vnode_base_offset_pair<1,  56, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 29, 0>::value, vnode_base_offset_pair<1,  56, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1,  56, 29, 1>::value, vnode_base_offset_pair<1,  56, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 29, 2>::value, vnode_base_offset_pair<1,  56, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 30, 0>::value, vnode_base_offset_pair<1,  56, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1,  56, 30, 1>::value, vnode_base_offset_pair<1,  56, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 30, 2>::value, vnode_base_offset_pair<1,  56, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 31, 0>::value, vnode_base_offset_pair<1,  56, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1,  56, 31, 1>::value, vnode_base_offset_pair<1,  56, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 31, 2>::value, vnode_base_offset_pair<1,  56, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 32, 0>::value, vnode_base_offset_pair<1,  56, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1,  56, 32, 1>::value, vnode_base_offset_pair<1,  56, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 32, 2>::value, vnode_base_offset_pair<1,  56, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 33, 0>::value, vnode_base_offset_pair<1,  56, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1,  56, 33, 1>::value, vnode_base_offset_pair<1,  56, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 33, 2>::value, vnode_base_offset_pair<1,  56, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 34, 0>::value, vnode_base_offset_pair<1,  56, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1,  56, 34, 1>::value, vnode_base_offset_pair<1,  56, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 34, 2>::value, vnode_base_offset_pair<1,  56, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 35, 0>::value, vnode_base_offset_pair<1,  56, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1,  56, 35, 1>::value, vnode_base_offset_pair<1,  56, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 35, 2>::value, vnode_base_offset_pair<1,  56, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 36, 0>::value, vnode_base_offset_pair<1,  56, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1,  56, 36, 1>::value, vnode_base_offset_pair<1,  56, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 36, 2>::value, vnode_base_offset_pair<1,  56, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 37, 0>::value, vnode_base_offset_pair<1,  56, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1,  56, 37, 1>::value, vnode_base_offset_pair<1,  56, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 38, 0>::value, vnode_base_offset_pair<1,  56, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1,  56, 38, 1>::value, vnode_base_offset_pair<1,  56, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 38, 2>::value, vnode_base_offset_pair<1,  56, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 39, 0>::value, vnode_base_offset_pair<1,  56, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1,  56, 39, 1>::value, vnode_base_offset_pair<1,  56, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 39, 2>::value, vnode_base_offset_pair<1,  56, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 40, 0>::value, vnode_base_offset_pair<1,  56, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1,  56, 40, 1>::value, vnode_base_offset_pair<1,  56, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 41, 0>::value, vnode_base_offset_pair<1,  56, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1,  56, 41, 1>::value, vnode_base_offset_pair<1,  56, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 41, 2>::value, vnode_base_offset_pair<1,  56, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 42, 0>::value, vnode_base_offset_pair<1,  56, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1,  56, 42, 1>::value, vnode_base_offset_pair<1,  56, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 43, 0>::value, vnode_base_offset_pair<1,  56, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1,  56, 43, 1>::value, vnode_base_offset_pair<1,  56, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 43, 2>::value, vnode_base_offset_pair<1,  56, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 44, 0>::value, vnode_base_offset_pair<1,  56, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1,  56, 44, 1>::value, vnode_base_offset_pair<1,  56, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  56, 44, 2>::value, vnode_base_offset_pair<1,  56, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  56, 45, 0>::value, vnode_base_offset_pair<1,  56, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1,  56, 45, 1>::value, vnode_base_offset_pair<1,  56, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z60_8 =
{
    {
        { vnode_shift_mod_pair<1,  60,  0, 0>::value, vnode_base_offset_pair<1,  60,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1,  60,  0, 1>::value, vnode_base_offset_pair<1,  60,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  0, 2>::value, vnode_base_offset_pair<1,  60,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  0, 3>::value, vnode_base_offset_pair<1,  60,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  0, 4>::value, vnode_base_offset_pair<1,  60,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  0, 5>::value, vnode_base_offset_pair<1,  60,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  0, 6>::value, vnode_base_offset_pair<1,  60,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  0, 7>::value, vnode_base_offset_pair<1,  60,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  0, 8>::value, vnode_base_offset_pair<1,  60,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  0, 9>::value, vnode_base_offset_pair<1,  60,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  60,  1, 0>::value, vnode_base_offset_pair<1,  60,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1,  60,  1, 1>::value, vnode_base_offset_pair<1,  60,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  1, 2>::value, vnode_base_offset_pair<1,  60,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  1, 3>::value, vnode_base_offset_pair<1,  60,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  1, 4>::value, vnode_base_offset_pair<1,  60,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  1, 5>::value, vnode_base_offset_pair<1,  60,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  1, 6>::value, vnode_base_offset_pair<1,  60,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  1, 7>::value, vnode_base_offset_pair<1,  60,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  1, 8>::value, vnode_base_offset_pair<1,  60,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  1, 9>::value, vnode_base_offset_pair<1,  60,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  60,  2, 0>::value, vnode_base_offset_pair<1,  60,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1,  60,  2, 1>::value, vnode_base_offset_pair<1,  60,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  2, 2>::value, vnode_base_offset_pair<1,  60,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  2, 3>::value, vnode_base_offset_pair<1,  60,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  2, 4>::value, vnode_base_offset_pair<1,  60,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  2, 5>::value, vnode_base_offset_pair<1,  60,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  2, 6>::value, vnode_base_offset_pair<1,  60,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  2, 7>::value, vnode_base_offset_pair<1,  60,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  2, 8>::value, vnode_base_offset_pair<1,  60,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  2, 9>::value, vnode_base_offset_pair<1,  60,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  60,  3, 0>::value, vnode_base_offset_pair<1,  60,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1,  60,  3, 1>::value, vnode_base_offset_pair<1,  60,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  3, 2>::value, vnode_base_offset_pair<1,  60,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  3, 3>::value, vnode_base_offset_pair<1,  60,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  3, 4>::value, vnode_base_offset_pair<1,  60,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  3, 5>::value, vnode_base_offset_pair<1,  60,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  3, 6>::value, vnode_base_offset_pair<1,  60,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  3, 7>::value, vnode_base_offset_pair<1,  60,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  3, 8>::value, vnode_base_offset_pair<1,  60,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  3, 9>::value, vnode_base_offset_pair<1,  60,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  60,  4, 0>::value, vnode_base_offset_pair<1,  60,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1,  60,  4, 1>::value, vnode_base_offset_pair<1,  60,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  60,  5, 0>::value, vnode_base_offset_pair<1,  60,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1,  60,  5, 1>::value, vnode_base_offset_pair<1,  60,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  5, 2>::value, vnode_base_offset_pair<1,  60,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  5, 3>::value, vnode_base_offset_pair<1,  60,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  60,  6, 0>::value, vnode_base_offset_pair<1,  60,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1,  60,  6, 1>::value, vnode_base_offset_pair<1,  60,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  6, 2>::value, vnode_base_offset_pair<1,  60,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  6, 3>::value, vnode_base_offset_pair<1,  60,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  6, 4>::value, vnode_base_offset_pair<1,  60,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  60,  7, 0>::value, vnode_base_offset_pair<1,  60,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1,  60,  7, 1>::value, vnode_base_offset_pair<1,  60,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  7, 2>::value, vnode_base_offset_pair<1,  60,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  7, 3>::value, vnode_base_offset_pair<1,  60,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  60,  8, 0>::value, vnode_base_offset_pair<1,  60,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1,  60,  8, 1>::value, vnode_base_offset_pair<1,  60,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  8, 2>::value, vnode_base_offset_pair<1,  60,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  8, 3>::value, vnode_base_offset_pair<1,  60,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  8, 4>::value, vnode_base_offset_pair<1,  60,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  60,  9, 0>::value, vnode_base_offset_pair<1,  60,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1,  60,  9, 1>::value, vnode_base_offset_pair<1,  60,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  9, 2>::value, vnode_base_offset_pair<1,  60,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  9, 3>::value, vnode_base_offset_pair<1,  60,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  60,  9, 4>::value, vnode_base_offset_pair<1,  60,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 10, 0>::value, vnode_base_offset_pair<1,  60, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1,  60, 10, 1>::value, vnode_base_offset_pair<1,  60, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 10, 2>::value, vnode_base_offset_pair<1,  60, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 10, 3>::value, vnode_base_offset_pair<1,  60, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 11, 0>::value, vnode_base_offset_pair<1,  60, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1,  60, 11, 1>::value, vnode_base_offset_pair<1,  60, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 11, 2>::value, vnode_base_offset_pair<1,  60, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 11, 3>::value, vnode_base_offset_pair<1,  60, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 12, 0>::value, vnode_base_offset_pair<1,  60, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1,  60, 12, 1>::value, vnode_base_offset_pair<1,  60, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 12, 2>::value, vnode_base_offset_pair<1,  60, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 12, 3>::value, vnode_base_offset_pair<1,  60, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 13, 0>::value, vnode_base_offset_pair<1,  60, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1,  60, 13, 1>::value, vnode_base_offset_pair<1,  60, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 13, 2>::value, vnode_base_offset_pair<1,  60, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 14, 0>::value, vnode_base_offset_pair<1,  60, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1,  60, 14, 1>::value, vnode_base_offset_pair<1,  60, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 14, 2>::value, vnode_base_offset_pair<1,  60, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 14, 3>::value, vnode_base_offset_pair<1,  60, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 15, 0>::value, vnode_base_offset_pair<1,  60, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1,  60, 15, 1>::value, vnode_base_offset_pair<1,  60, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 15, 2>::value, vnode_base_offset_pair<1,  60, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 15, 3>::value, vnode_base_offset_pair<1,  60, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 16, 0>::value, vnode_base_offset_pair<1,  60, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1,  60, 16, 1>::value, vnode_base_offset_pair<1,  60, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 16, 2>::value, vnode_base_offset_pair<1,  60, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 17, 0>::value, vnode_base_offset_pair<1,  60, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1,  60, 17, 1>::value, vnode_base_offset_pair<1,  60, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 17, 2>::value, vnode_base_offset_pair<1,  60, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 18, 0>::value, vnode_base_offset_pair<1,  60, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1,  60, 18, 1>::value, vnode_base_offset_pair<1,  60, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 18, 2>::value, vnode_base_offset_pair<1,  60, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 19, 0>::value, vnode_base_offset_pair<1,  60, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1,  60, 19, 1>::value, vnode_base_offset_pair<1,  60, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 19, 2>::value, vnode_base_offset_pair<1,  60, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 20, 0>::value, vnode_base_offset_pair<1,  60, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1,  60, 20, 1>::value, vnode_base_offset_pair<1,  60, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 20, 2>::value, vnode_base_offset_pair<1,  60, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 21, 0>::value, vnode_base_offset_pair<1,  60, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1,  60, 21, 1>::value, vnode_base_offset_pair<1,  60, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 21, 2>::value, vnode_base_offset_pair<1,  60, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 22, 0>::value, vnode_base_offset_pair<1,  60, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1,  60, 22, 1>::value, vnode_base_offset_pair<1,  60, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 22, 2>::value, vnode_base_offset_pair<1,  60, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 23, 0>::value, vnode_base_offset_pair<1,  60, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1,  60, 23, 1>::value, vnode_base_offset_pair<1,  60, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 23, 2>::value, vnode_base_offset_pair<1,  60, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 24, 0>::value, vnode_base_offset_pair<1,  60, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1,  60, 24, 1>::value, vnode_base_offset_pair<1,  60, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 24, 2>::value, vnode_base_offset_pair<1,  60, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 25, 0>::value, vnode_base_offset_pair<1,  60, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1,  60, 25, 1>::value, vnode_base_offset_pair<1,  60, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 25, 2>::value, vnode_base_offset_pair<1,  60, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 26, 0>::value, vnode_base_offset_pair<1,  60, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1,  60, 26, 1>::value, vnode_base_offset_pair<1,  60, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 26, 2>::value, vnode_base_offset_pair<1,  60, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 27, 0>::value, vnode_base_offset_pair<1,  60, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1,  60, 27, 1>::value, vnode_base_offset_pair<1,  60, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 28, 0>::value, vnode_base_offset_pair<1,  60, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1,  60, 28, 1>::value, vnode_base_offset_pair<1,  60, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 28, 2>::value, vnode_base_offset_pair<1,  60, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 29, 0>::value, vnode_base_offset_pair<1,  60, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1,  60, 29, 1>::value, vnode_base_offset_pair<1,  60, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 29, 2>::value, vnode_base_offset_pair<1,  60, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 30, 0>::value, vnode_base_offset_pair<1,  60, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1,  60, 30, 1>::value, vnode_base_offset_pair<1,  60, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 30, 2>::value, vnode_base_offset_pair<1,  60, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 31, 0>::value, vnode_base_offset_pair<1,  60, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1,  60, 31, 1>::value, vnode_base_offset_pair<1,  60, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 31, 2>::value, vnode_base_offset_pair<1,  60, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 32, 0>::value, vnode_base_offset_pair<1,  60, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1,  60, 32, 1>::value, vnode_base_offset_pair<1,  60, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 32, 2>::value, vnode_base_offset_pair<1,  60, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 33, 0>::value, vnode_base_offset_pair<1,  60, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1,  60, 33, 1>::value, vnode_base_offset_pair<1,  60, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 33, 2>::value, vnode_base_offset_pair<1,  60, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 34, 0>::value, vnode_base_offset_pair<1,  60, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1,  60, 34, 1>::value, vnode_base_offset_pair<1,  60, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 34, 2>::value, vnode_base_offset_pair<1,  60, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 35, 0>::value, vnode_base_offset_pair<1,  60, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1,  60, 35, 1>::value, vnode_base_offset_pair<1,  60, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 35, 2>::value, vnode_base_offset_pair<1,  60, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 36, 0>::value, vnode_base_offset_pair<1,  60, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1,  60, 36, 1>::value, vnode_base_offset_pair<1,  60, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 36, 2>::value, vnode_base_offset_pair<1,  60, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 37, 0>::value, vnode_base_offset_pair<1,  60, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1,  60, 37, 1>::value, vnode_base_offset_pair<1,  60, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 38, 0>::value, vnode_base_offset_pair<1,  60, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1,  60, 38, 1>::value, vnode_base_offset_pair<1,  60, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 38, 2>::value, vnode_base_offset_pair<1,  60, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 39, 0>::value, vnode_base_offset_pair<1,  60, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1,  60, 39, 1>::value, vnode_base_offset_pair<1,  60, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 39, 2>::value, vnode_base_offset_pair<1,  60, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 40, 0>::value, vnode_base_offset_pair<1,  60, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1,  60, 40, 1>::value, vnode_base_offset_pair<1,  60, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 41, 0>::value, vnode_base_offset_pair<1,  60, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1,  60, 41, 1>::value, vnode_base_offset_pair<1,  60, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 41, 2>::value, vnode_base_offset_pair<1,  60, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 42, 0>::value, vnode_base_offset_pair<1,  60, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1,  60, 42, 1>::value, vnode_base_offset_pair<1,  60, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 43, 0>::value, vnode_base_offset_pair<1,  60, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1,  60, 43, 1>::value, vnode_base_offset_pair<1,  60, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 43, 2>::value, vnode_base_offset_pair<1,  60, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 44, 0>::value, vnode_base_offset_pair<1,  60, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1,  60, 44, 1>::value, vnode_base_offset_pair<1,  60, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  60, 44, 2>::value, vnode_base_offset_pair<1,  60, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  60, 45, 0>::value, vnode_base_offset_pair<1,  60, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1,  60, 45, 1>::value, vnode_base_offset_pair<1,  60, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z64_8 =
{
    {
        { vnode_shift_mod_pair<1,  64,  0, 0>::value, vnode_base_offset_pair<1,  64,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1,  64,  0, 1>::value, vnode_base_offset_pair<1,  64,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  0, 2>::value, vnode_base_offset_pair<1,  64,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  0, 3>::value, vnode_base_offset_pair<1,  64,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  0, 4>::value, vnode_base_offset_pair<1,  64,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  0, 5>::value, vnode_base_offset_pair<1,  64,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  0, 6>::value, vnode_base_offset_pair<1,  64,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  0, 7>::value, vnode_base_offset_pair<1,  64,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  0, 8>::value, vnode_base_offset_pair<1,  64,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  0, 9>::value, vnode_base_offset_pair<1,  64,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  64,  1, 0>::value, vnode_base_offset_pair<1,  64,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1,  64,  1, 1>::value, vnode_base_offset_pair<1,  64,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  1, 2>::value, vnode_base_offset_pair<1,  64,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  1, 3>::value, vnode_base_offset_pair<1,  64,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  1, 4>::value, vnode_base_offset_pair<1,  64,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  1, 5>::value, vnode_base_offset_pair<1,  64,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  1, 6>::value, vnode_base_offset_pair<1,  64,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  1, 7>::value, vnode_base_offset_pair<1,  64,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  1, 8>::value, vnode_base_offset_pair<1,  64,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  1, 9>::value, vnode_base_offset_pair<1,  64,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  64,  2, 0>::value, vnode_base_offset_pair<1,  64,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1,  64,  2, 1>::value, vnode_base_offset_pair<1,  64,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  2, 2>::value, vnode_base_offset_pair<1,  64,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  2, 3>::value, vnode_base_offset_pair<1,  64,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  2, 4>::value, vnode_base_offset_pair<1,  64,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  2, 5>::value, vnode_base_offset_pair<1,  64,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  2, 6>::value, vnode_base_offset_pair<1,  64,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  2, 7>::value, vnode_base_offset_pair<1,  64,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  2, 8>::value, vnode_base_offset_pair<1,  64,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  2, 9>::value, vnode_base_offset_pair<1,  64,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  64,  3, 0>::value, vnode_base_offset_pair<1,  64,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1,  64,  3, 1>::value, vnode_base_offset_pair<1,  64,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  3, 2>::value, vnode_base_offset_pair<1,  64,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  3, 3>::value, vnode_base_offset_pair<1,  64,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  3, 4>::value, vnode_base_offset_pair<1,  64,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  3, 5>::value, vnode_base_offset_pair<1,  64,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  3, 6>::value, vnode_base_offset_pair<1,  64,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  3, 7>::value, vnode_base_offset_pair<1,  64,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  3, 8>::value, vnode_base_offset_pair<1,  64,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  3, 9>::value, vnode_base_offset_pair<1,  64,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  64,  4, 0>::value, vnode_base_offset_pair<1,  64,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1,  64,  4, 1>::value, vnode_base_offset_pair<1,  64,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  64,  5, 0>::value, vnode_base_offset_pair<1,  64,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1,  64,  5, 1>::value, vnode_base_offset_pair<1,  64,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  5, 2>::value, vnode_base_offset_pair<1,  64,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  5, 3>::value, vnode_base_offset_pair<1,  64,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  64,  6, 0>::value, vnode_base_offset_pair<1,  64,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1,  64,  6, 1>::value, vnode_base_offset_pair<1,  64,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  6, 2>::value, vnode_base_offset_pair<1,  64,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  6, 3>::value, vnode_base_offset_pair<1,  64,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  6, 4>::value, vnode_base_offset_pair<1,  64,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  64,  7, 0>::value, vnode_base_offset_pair<1,  64,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1,  64,  7, 1>::value, vnode_base_offset_pair<1,  64,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  7, 2>::value, vnode_base_offset_pair<1,  64,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  7, 3>::value, vnode_base_offset_pair<1,  64,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  64,  8, 0>::value, vnode_base_offset_pair<1,  64,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1,  64,  8, 1>::value, vnode_base_offset_pair<1,  64,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  8, 2>::value, vnode_base_offset_pair<1,  64,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  8, 3>::value, vnode_base_offset_pair<1,  64,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  8, 4>::value, vnode_base_offset_pair<1,  64,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  64,  9, 0>::value, vnode_base_offset_pair<1,  64,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1,  64,  9, 1>::value, vnode_base_offset_pair<1,  64,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  9, 2>::value, vnode_base_offset_pair<1,  64,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  9, 3>::value, vnode_base_offset_pair<1,  64,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  64,  9, 4>::value, vnode_base_offset_pair<1,  64,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 10, 0>::value, vnode_base_offset_pair<1,  64, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1,  64, 10, 1>::value, vnode_base_offset_pair<1,  64, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 10, 2>::value, vnode_base_offset_pair<1,  64, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 10, 3>::value, vnode_base_offset_pair<1,  64, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 11, 0>::value, vnode_base_offset_pair<1,  64, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1,  64, 11, 1>::value, vnode_base_offset_pair<1,  64, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 11, 2>::value, vnode_base_offset_pair<1,  64, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 11, 3>::value, vnode_base_offset_pair<1,  64, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 12, 0>::value, vnode_base_offset_pair<1,  64, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1,  64, 12, 1>::value, vnode_base_offset_pair<1,  64, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 12, 2>::value, vnode_base_offset_pair<1,  64, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 12, 3>::value, vnode_base_offset_pair<1,  64, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 13, 0>::value, vnode_base_offset_pair<1,  64, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1,  64, 13, 1>::value, vnode_base_offset_pair<1,  64, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 13, 2>::value, vnode_base_offset_pair<1,  64, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 14, 0>::value, vnode_base_offset_pair<1,  64, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1,  64, 14, 1>::value, vnode_base_offset_pair<1,  64, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 14, 2>::value, vnode_base_offset_pair<1,  64, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 14, 3>::value, vnode_base_offset_pair<1,  64, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 15, 0>::value, vnode_base_offset_pair<1,  64, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1,  64, 15, 1>::value, vnode_base_offset_pair<1,  64, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 15, 2>::value, vnode_base_offset_pair<1,  64, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 15, 3>::value, vnode_base_offset_pair<1,  64, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 16, 0>::value, vnode_base_offset_pair<1,  64, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1,  64, 16, 1>::value, vnode_base_offset_pair<1,  64, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 16, 2>::value, vnode_base_offset_pair<1,  64, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 17, 0>::value, vnode_base_offset_pair<1,  64, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1,  64, 17, 1>::value, vnode_base_offset_pair<1,  64, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 17, 2>::value, vnode_base_offset_pair<1,  64, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 18, 0>::value, vnode_base_offset_pair<1,  64, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1,  64, 18, 1>::value, vnode_base_offset_pair<1,  64, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 18, 2>::value, vnode_base_offset_pair<1,  64, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 19, 0>::value, vnode_base_offset_pair<1,  64, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1,  64, 19, 1>::value, vnode_base_offset_pair<1,  64, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 19, 2>::value, vnode_base_offset_pair<1,  64, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 20, 0>::value, vnode_base_offset_pair<1,  64, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1,  64, 20, 1>::value, vnode_base_offset_pair<1,  64, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 20, 2>::value, vnode_base_offset_pair<1,  64, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 21, 0>::value, vnode_base_offset_pair<1,  64, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1,  64, 21, 1>::value, vnode_base_offset_pair<1,  64, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 21, 2>::value, vnode_base_offset_pair<1,  64, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 22, 0>::value, vnode_base_offset_pair<1,  64, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1,  64, 22, 1>::value, vnode_base_offset_pair<1,  64, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 22, 2>::value, vnode_base_offset_pair<1,  64, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 23, 0>::value, vnode_base_offset_pair<1,  64, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1,  64, 23, 1>::value, vnode_base_offset_pair<1,  64, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 23, 2>::value, vnode_base_offset_pair<1,  64, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 24, 0>::value, vnode_base_offset_pair<1,  64, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1,  64, 24, 1>::value, vnode_base_offset_pair<1,  64, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 24, 2>::value, vnode_base_offset_pair<1,  64, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 25, 0>::value, vnode_base_offset_pair<1,  64, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1,  64, 25, 1>::value, vnode_base_offset_pair<1,  64, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 25, 2>::value, vnode_base_offset_pair<1,  64, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 26, 0>::value, vnode_base_offset_pair<1,  64, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1,  64, 26, 1>::value, vnode_base_offset_pair<1,  64, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 26, 2>::value, vnode_base_offset_pair<1,  64, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 27, 0>::value, vnode_base_offset_pair<1,  64, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1,  64, 27, 1>::value, vnode_base_offset_pair<1,  64, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 28, 0>::value, vnode_base_offset_pair<1,  64, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1,  64, 28, 1>::value, vnode_base_offset_pair<1,  64, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 28, 2>::value, vnode_base_offset_pair<1,  64, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 29, 0>::value, vnode_base_offset_pair<1,  64, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1,  64, 29, 1>::value, vnode_base_offset_pair<1,  64, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 29, 2>::value, vnode_base_offset_pair<1,  64, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 30, 0>::value, vnode_base_offset_pair<1,  64, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1,  64, 30, 1>::value, vnode_base_offset_pair<1,  64, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 30, 2>::value, vnode_base_offset_pair<1,  64, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 31, 0>::value, vnode_base_offset_pair<1,  64, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1,  64, 31, 1>::value, vnode_base_offset_pair<1,  64, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 31, 2>::value, vnode_base_offset_pair<1,  64, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 32, 0>::value, vnode_base_offset_pair<1,  64, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1,  64, 32, 1>::value, vnode_base_offset_pair<1,  64, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 32, 2>::value, vnode_base_offset_pair<1,  64, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 33, 0>::value, vnode_base_offset_pair<1,  64, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1,  64, 33, 1>::value, vnode_base_offset_pair<1,  64, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 33, 2>::value, vnode_base_offset_pair<1,  64, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 34, 0>::value, vnode_base_offset_pair<1,  64, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1,  64, 34, 1>::value, vnode_base_offset_pair<1,  64, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 34, 2>::value, vnode_base_offset_pair<1,  64, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 35, 0>::value, vnode_base_offset_pair<1,  64, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1,  64, 35, 1>::value, vnode_base_offset_pair<1,  64, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 35, 2>::value, vnode_base_offset_pair<1,  64, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 36, 0>::value, vnode_base_offset_pair<1,  64, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1,  64, 36, 1>::value, vnode_base_offset_pair<1,  64, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 36, 2>::value, vnode_base_offset_pair<1,  64, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 37, 0>::value, vnode_base_offset_pair<1,  64, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1,  64, 37, 1>::value, vnode_base_offset_pair<1,  64, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 38, 0>::value, vnode_base_offset_pair<1,  64, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1,  64, 38, 1>::value, vnode_base_offset_pair<1,  64, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 38, 2>::value, vnode_base_offset_pair<1,  64, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 39, 0>::value, vnode_base_offset_pair<1,  64, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1,  64, 39, 1>::value, vnode_base_offset_pair<1,  64, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 39, 2>::value, vnode_base_offset_pair<1,  64, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 40, 0>::value, vnode_base_offset_pair<1,  64, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1,  64, 40, 1>::value, vnode_base_offset_pair<1,  64, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 41, 0>::value, vnode_base_offset_pair<1,  64, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1,  64, 41, 1>::value, vnode_base_offset_pair<1,  64, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 41, 2>::value, vnode_base_offset_pair<1,  64, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 42, 0>::value, vnode_base_offset_pair<1,  64, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1,  64, 42, 1>::value, vnode_base_offset_pair<1,  64, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 43, 0>::value, vnode_base_offset_pair<1,  64, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1,  64, 43, 1>::value, vnode_base_offset_pair<1,  64, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 43, 2>::value, vnode_base_offset_pair<1,  64, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 44, 0>::value, vnode_base_offset_pair<1,  64, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1,  64, 44, 1>::value, vnode_base_offset_pair<1,  64, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  64, 44, 2>::value, vnode_base_offset_pair<1,  64, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  64, 45, 0>::value, vnode_base_offset_pair<1,  64, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1,  64, 45, 1>::value, vnode_base_offset_pair<1,  64, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z72_8 =
{
    {
        { vnode_shift_mod_pair<1,  72,  0, 0>::value, vnode_base_offset_pair<1,  72,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1,  72,  0, 1>::value, vnode_base_offset_pair<1,  72,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  0, 2>::value, vnode_base_offset_pair<1,  72,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  0, 3>::value, vnode_base_offset_pair<1,  72,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  0, 4>::value, vnode_base_offset_pair<1,  72,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  0, 5>::value, vnode_base_offset_pair<1,  72,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  0, 6>::value, vnode_base_offset_pair<1,  72,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  0, 7>::value, vnode_base_offset_pair<1,  72,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  0, 8>::value, vnode_base_offset_pair<1,  72,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  0, 9>::value, vnode_base_offset_pair<1,  72,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  72,  1, 0>::value, vnode_base_offset_pair<1,  72,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1,  72,  1, 1>::value, vnode_base_offset_pair<1,  72,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  1, 2>::value, vnode_base_offset_pair<1,  72,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  1, 3>::value, vnode_base_offset_pair<1,  72,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  1, 4>::value, vnode_base_offset_pair<1,  72,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  1, 5>::value, vnode_base_offset_pair<1,  72,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  1, 6>::value, vnode_base_offset_pair<1,  72,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  1, 7>::value, vnode_base_offset_pair<1,  72,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  1, 8>::value, vnode_base_offset_pair<1,  72,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  1, 9>::value, vnode_base_offset_pair<1,  72,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  72,  2, 0>::value, vnode_base_offset_pair<1,  72,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1,  72,  2, 1>::value, vnode_base_offset_pair<1,  72,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  2, 2>::value, vnode_base_offset_pair<1,  72,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  2, 3>::value, vnode_base_offset_pair<1,  72,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  2, 4>::value, vnode_base_offset_pair<1,  72,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  2, 5>::value, vnode_base_offset_pair<1,  72,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  2, 6>::value, vnode_base_offset_pair<1,  72,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  2, 7>::value, vnode_base_offset_pair<1,  72,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  2, 8>::value, vnode_base_offset_pair<1,  72,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  2, 9>::value, vnode_base_offset_pair<1,  72,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  72,  3, 0>::value, vnode_base_offset_pair<1,  72,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1,  72,  3, 1>::value, vnode_base_offset_pair<1,  72,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  3, 2>::value, vnode_base_offset_pair<1,  72,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  3, 3>::value, vnode_base_offset_pair<1,  72,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  3, 4>::value, vnode_base_offset_pair<1,  72,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  3, 5>::value, vnode_base_offset_pair<1,  72,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  3, 6>::value, vnode_base_offset_pair<1,  72,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  3, 7>::value, vnode_base_offset_pair<1,  72,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  3, 8>::value, vnode_base_offset_pair<1,  72,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  3, 9>::value, vnode_base_offset_pair<1,  72,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  72,  4, 0>::value, vnode_base_offset_pair<1,  72,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1,  72,  4, 1>::value, vnode_base_offset_pair<1,  72,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  72,  5, 0>::value, vnode_base_offset_pair<1,  72,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1,  72,  5, 1>::value, vnode_base_offset_pair<1,  72,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  5, 2>::value, vnode_base_offset_pair<1,  72,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  5, 3>::value, vnode_base_offset_pair<1,  72,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  72,  6, 0>::value, vnode_base_offset_pair<1,  72,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1,  72,  6, 1>::value, vnode_base_offset_pair<1,  72,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  6, 2>::value, vnode_base_offset_pair<1,  72,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  6, 3>::value, vnode_base_offset_pair<1,  72,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  6, 4>::value, vnode_base_offset_pair<1,  72,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  72,  7, 0>::value, vnode_base_offset_pair<1,  72,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1,  72,  7, 1>::value, vnode_base_offset_pair<1,  72,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  7, 2>::value, vnode_base_offset_pair<1,  72,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  7, 3>::value, vnode_base_offset_pair<1,  72,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  72,  8, 0>::value, vnode_base_offset_pair<1,  72,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1,  72,  8, 1>::value, vnode_base_offset_pair<1,  72,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  8, 2>::value, vnode_base_offset_pair<1,  72,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  8, 3>::value, vnode_base_offset_pair<1,  72,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  8, 4>::value, vnode_base_offset_pair<1,  72,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  72,  9, 0>::value, vnode_base_offset_pair<1,  72,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1,  72,  9, 1>::value, vnode_base_offset_pair<1,  72,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  9, 2>::value, vnode_base_offset_pair<1,  72,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  9, 3>::value, vnode_base_offset_pair<1,  72,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  72,  9, 4>::value, vnode_base_offset_pair<1,  72,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 10, 0>::value, vnode_base_offset_pair<1,  72, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1,  72, 10, 1>::value, vnode_base_offset_pair<1,  72, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 10, 2>::value, vnode_base_offset_pair<1,  72, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 10, 3>::value, vnode_base_offset_pair<1,  72, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 11, 0>::value, vnode_base_offset_pair<1,  72, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1,  72, 11, 1>::value, vnode_base_offset_pair<1,  72, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 11, 2>::value, vnode_base_offset_pair<1,  72, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 11, 3>::value, vnode_base_offset_pair<1,  72, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 12, 0>::value, vnode_base_offset_pair<1,  72, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1,  72, 12, 1>::value, vnode_base_offset_pair<1,  72, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 12, 2>::value, vnode_base_offset_pair<1,  72, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 12, 3>::value, vnode_base_offset_pair<1,  72, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 13, 0>::value, vnode_base_offset_pair<1,  72, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1,  72, 13, 1>::value, vnode_base_offset_pair<1,  72, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 13, 2>::value, vnode_base_offset_pair<1,  72, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 14, 0>::value, vnode_base_offset_pair<1,  72, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1,  72, 14, 1>::value, vnode_base_offset_pair<1,  72, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 14, 2>::value, vnode_base_offset_pair<1,  72, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 14, 3>::value, vnode_base_offset_pair<1,  72, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 15, 0>::value, vnode_base_offset_pair<1,  72, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1,  72, 15, 1>::value, vnode_base_offset_pair<1,  72, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 15, 2>::value, vnode_base_offset_pair<1,  72, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 15, 3>::value, vnode_base_offset_pair<1,  72, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 16, 0>::value, vnode_base_offset_pair<1,  72, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1,  72, 16, 1>::value, vnode_base_offset_pair<1,  72, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 16, 2>::value, vnode_base_offset_pair<1,  72, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 17, 0>::value, vnode_base_offset_pair<1,  72, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1,  72, 17, 1>::value, vnode_base_offset_pair<1,  72, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 17, 2>::value, vnode_base_offset_pair<1,  72, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 18, 0>::value, vnode_base_offset_pair<1,  72, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1,  72, 18, 1>::value, vnode_base_offset_pair<1,  72, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 18, 2>::value, vnode_base_offset_pair<1,  72, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 19, 0>::value, vnode_base_offset_pair<1,  72, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1,  72, 19, 1>::value, vnode_base_offset_pair<1,  72, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 19, 2>::value, vnode_base_offset_pair<1,  72, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 20, 0>::value, vnode_base_offset_pair<1,  72, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1,  72, 20, 1>::value, vnode_base_offset_pair<1,  72, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 20, 2>::value, vnode_base_offset_pair<1,  72, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 21, 0>::value, vnode_base_offset_pair<1,  72, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1,  72, 21, 1>::value, vnode_base_offset_pair<1,  72, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 21, 2>::value, vnode_base_offset_pair<1,  72, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 22, 0>::value, vnode_base_offset_pair<1,  72, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1,  72, 22, 1>::value, vnode_base_offset_pair<1,  72, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 22, 2>::value, vnode_base_offset_pair<1,  72, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 23, 0>::value, vnode_base_offset_pair<1,  72, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1,  72, 23, 1>::value, vnode_base_offset_pair<1,  72, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 23, 2>::value, vnode_base_offset_pair<1,  72, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 24, 0>::value, vnode_base_offset_pair<1,  72, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1,  72, 24, 1>::value, vnode_base_offset_pair<1,  72, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 24, 2>::value, vnode_base_offset_pair<1,  72, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 25, 0>::value, vnode_base_offset_pair<1,  72, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1,  72, 25, 1>::value, vnode_base_offset_pair<1,  72, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 25, 2>::value, vnode_base_offset_pair<1,  72, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 26, 0>::value, vnode_base_offset_pair<1,  72, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1,  72, 26, 1>::value, vnode_base_offset_pair<1,  72, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 26, 2>::value, vnode_base_offset_pair<1,  72, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 27, 0>::value, vnode_base_offset_pair<1,  72, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1,  72, 27, 1>::value, vnode_base_offset_pair<1,  72, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 28, 0>::value, vnode_base_offset_pair<1,  72, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1,  72, 28, 1>::value, vnode_base_offset_pair<1,  72, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 28, 2>::value, vnode_base_offset_pair<1,  72, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 29, 0>::value, vnode_base_offset_pair<1,  72, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1,  72, 29, 1>::value, vnode_base_offset_pair<1,  72, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 29, 2>::value, vnode_base_offset_pair<1,  72, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 30, 0>::value, vnode_base_offset_pair<1,  72, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1,  72, 30, 1>::value, vnode_base_offset_pair<1,  72, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 30, 2>::value, vnode_base_offset_pair<1,  72, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 31, 0>::value, vnode_base_offset_pair<1,  72, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1,  72, 31, 1>::value, vnode_base_offset_pair<1,  72, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 31, 2>::value, vnode_base_offset_pair<1,  72, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 32, 0>::value, vnode_base_offset_pair<1,  72, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1,  72, 32, 1>::value, vnode_base_offset_pair<1,  72, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 32, 2>::value, vnode_base_offset_pair<1,  72, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 33, 0>::value, vnode_base_offset_pair<1,  72, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1,  72, 33, 1>::value, vnode_base_offset_pair<1,  72, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 33, 2>::value, vnode_base_offset_pair<1,  72, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 34, 0>::value, vnode_base_offset_pair<1,  72, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1,  72, 34, 1>::value, vnode_base_offset_pair<1,  72, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 34, 2>::value, vnode_base_offset_pair<1,  72, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 35, 0>::value, vnode_base_offset_pair<1,  72, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1,  72, 35, 1>::value, vnode_base_offset_pair<1,  72, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 35, 2>::value, vnode_base_offset_pair<1,  72, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 36, 0>::value, vnode_base_offset_pair<1,  72, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1,  72, 36, 1>::value, vnode_base_offset_pair<1,  72, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 36, 2>::value, vnode_base_offset_pair<1,  72, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 37, 0>::value, vnode_base_offset_pair<1,  72, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1,  72, 37, 1>::value, vnode_base_offset_pair<1,  72, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 38, 0>::value, vnode_base_offset_pair<1,  72, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1,  72, 38, 1>::value, vnode_base_offset_pair<1,  72, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 38, 2>::value, vnode_base_offset_pair<1,  72, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 39, 0>::value, vnode_base_offset_pair<1,  72, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1,  72, 39, 1>::value, vnode_base_offset_pair<1,  72, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 39, 2>::value, vnode_base_offset_pair<1,  72, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 40, 0>::value, vnode_base_offset_pair<1,  72, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1,  72, 40, 1>::value, vnode_base_offset_pair<1,  72, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 41, 0>::value, vnode_base_offset_pair<1,  72, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1,  72, 41, 1>::value, vnode_base_offset_pair<1,  72, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 41, 2>::value, vnode_base_offset_pair<1,  72, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 42, 0>::value, vnode_base_offset_pair<1,  72, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1,  72, 42, 1>::value, vnode_base_offset_pair<1,  72, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 43, 0>::value, vnode_base_offset_pair<1,  72, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1,  72, 43, 1>::value, vnode_base_offset_pair<1,  72, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 43, 2>::value, vnode_base_offset_pair<1,  72, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 44, 0>::value, vnode_base_offset_pair<1,  72, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1,  72, 44, 1>::value, vnode_base_offset_pair<1,  72, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  72, 44, 2>::value, vnode_base_offset_pair<1,  72, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  72, 45, 0>::value, vnode_base_offset_pair<1,  72, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1,  72, 45, 1>::value, vnode_base_offset_pair<1,  72, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z80_8 =
{
    {
        { vnode_shift_mod_pair<1,  80,  0, 0>::value, vnode_base_offset_pair<1,  80,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1,  80,  0, 1>::value, vnode_base_offset_pair<1,  80,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  0, 2>::value, vnode_base_offset_pair<1,  80,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  0, 3>::value, vnode_base_offset_pair<1,  80,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  0, 4>::value, vnode_base_offset_pair<1,  80,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  0, 5>::value, vnode_base_offset_pair<1,  80,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  0, 6>::value, vnode_base_offset_pair<1,  80,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  0, 7>::value, vnode_base_offset_pair<1,  80,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  0, 8>::value, vnode_base_offset_pair<1,  80,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  0, 9>::value, vnode_base_offset_pair<1,  80,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  80,  1, 0>::value, vnode_base_offset_pair<1,  80,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1,  80,  1, 1>::value, vnode_base_offset_pair<1,  80,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  1, 2>::value, vnode_base_offset_pair<1,  80,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  1, 3>::value, vnode_base_offset_pair<1,  80,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  1, 4>::value, vnode_base_offset_pair<1,  80,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  1, 5>::value, vnode_base_offset_pair<1,  80,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  1, 6>::value, vnode_base_offset_pair<1,  80,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  1, 7>::value, vnode_base_offset_pair<1,  80,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  1, 8>::value, vnode_base_offset_pair<1,  80,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  1, 9>::value, vnode_base_offset_pair<1,  80,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  80,  2, 0>::value, vnode_base_offset_pair<1,  80,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1,  80,  2, 1>::value, vnode_base_offset_pair<1,  80,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  2, 2>::value, vnode_base_offset_pair<1,  80,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  2, 3>::value, vnode_base_offset_pair<1,  80,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  2, 4>::value, vnode_base_offset_pair<1,  80,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  2, 5>::value, vnode_base_offset_pair<1,  80,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  2, 6>::value, vnode_base_offset_pair<1,  80,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  2, 7>::value, vnode_base_offset_pair<1,  80,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  2, 8>::value, vnode_base_offset_pair<1,  80,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  2, 9>::value, vnode_base_offset_pair<1,  80,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  80,  3, 0>::value, vnode_base_offset_pair<1,  80,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1,  80,  3, 1>::value, vnode_base_offset_pair<1,  80,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  3, 2>::value, vnode_base_offset_pair<1,  80,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  3, 3>::value, vnode_base_offset_pair<1,  80,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  3, 4>::value, vnode_base_offset_pair<1,  80,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  3, 5>::value, vnode_base_offset_pair<1,  80,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  3, 6>::value, vnode_base_offset_pair<1,  80,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  3, 7>::value, vnode_base_offset_pair<1,  80,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  3, 8>::value, vnode_base_offset_pair<1,  80,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  3, 9>::value, vnode_base_offset_pair<1,  80,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  80,  4, 0>::value, vnode_base_offset_pair<1,  80,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1,  80,  4, 1>::value, vnode_base_offset_pair<1,  80,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  80,  5, 0>::value, vnode_base_offset_pair<1,  80,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1,  80,  5, 1>::value, vnode_base_offset_pair<1,  80,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  5, 2>::value, vnode_base_offset_pair<1,  80,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  5, 3>::value, vnode_base_offset_pair<1,  80,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  80,  6, 0>::value, vnode_base_offset_pair<1,  80,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1,  80,  6, 1>::value, vnode_base_offset_pair<1,  80,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  6, 2>::value, vnode_base_offset_pair<1,  80,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  6, 3>::value, vnode_base_offset_pair<1,  80,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  6, 4>::value, vnode_base_offset_pair<1,  80,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  80,  7, 0>::value, vnode_base_offset_pair<1,  80,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1,  80,  7, 1>::value, vnode_base_offset_pair<1,  80,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  7, 2>::value, vnode_base_offset_pair<1,  80,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  7, 3>::value, vnode_base_offset_pair<1,  80,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  80,  8, 0>::value, vnode_base_offset_pair<1,  80,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1,  80,  8, 1>::value, vnode_base_offset_pair<1,  80,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  8, 2>::value, vnode_base_offset_pair<1,  80,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  8, 3>::value, vnode_base_offset_pair<1,  80,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  8, 4>::value, vnode_base_offset_pair<1,  80,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  80,  9, 0>::value, vnode_base_offset_pair<1,  80,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1,  80,  9, 1>::value, vnode_base_offset_pair<1,  80,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  9, 2>::value, vnode_base_offset_pair<1,  80,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  9, 3>::value, vnode_base_offset_pair<1,  80,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  80,  9, 4>::value, vnode_base_offset_pair<1,  80,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 10, 0>::value, vnode_base_offset_pair<1,  80, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1,  80, 10, 1>::value, vnode_base_offset_pair<1,  80, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 10, 2>::value, vnode_base_offset_pair<1,  80, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 10, 3>::value, vnode_base_offset_pair<1,  80, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 11, 0>::value, vnode_base_offset_pair<1,  80, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1,  80, 11, 1>::value, vnode_base_offset_pair<1,  80, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 11, 2>::value, vnode_base_offset_pair<1,  80, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 11, 3>::value, vnode_base_offset_pair<1,  80, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 12, 0>::value, vnode_base_offset_pair<1,  80, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1,  80, 12, 1>::value, vnode_base_offset_pair<1,  80, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 12, 2>::value, vnode_base_offset_pair<1,  80, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 12, 3>::value, vnode_base_offset_pair<1,  80, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 13, 0>::value, vnode_base_offset_pair<1,  80, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1,  80, 13, 1>::value, vnode_base_offset_pair<1,  80, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 13, 2>::value, vnode_base_offset_pair<1,  80, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 14, 0>::value, vnode_base_offset_pair<1,  80, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1,  80, 14, 1>::value, vnode_base_offset_pair<1,  80, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 14, 2>::value, vnode_base_offset_pair<1,  80, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 14, 3>::value, vnode_base_offset_pair<1,  80, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 15, 0>::value, vnode_base_offset_pair<1,  80, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1,  80, 15, 1>::value, vnode_base_offset_pair<1,  80, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 15, 2>::value, vnode_base_offset_pair<1,  80, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 15, 3>::value, vnode_base_offset_pair<1,  80, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 16, 0>::value, vnode_base_offset_pair<1,  80, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1,  80, 16, 1>::value, vnode_base_offset_pair<1,  80, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 16, 2>::value, vnode_base_offset_pair<1,  80, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 17, 0>::value, vnode_base_offset_pair<1,  80, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1,  80, 17, 1>::value, vnode_base_offset_pair<1,  80, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 17, 2>::value, vnode_base_offset_pair<1,  80, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 18, 0>::value, vnode_base_offset_pair<1,  80, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1,  80, 18, 1>::value, vnode_base_offset_pair<1,  80, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 18, 2>::value, vnode_base_offset_pair<1,  80, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 19, 0>::value, vnode_base_offset_pair<1,  80, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1,  80, 19, 1>::value, vnode_base_offset_pair<1,  80, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 19, 2>::value, vnode_base_offset_pair<1,  80, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 20, 0>::value, vnode_base_offset_pair<1,  80, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1,  80, 20, 1>::value, vnode_base_offset_pair<1,  80, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 20, 2>::value, vnode_base_offset_pair<1,  80, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 21, 0>::value, vnode_base_offset_pair<1,  80, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1,  80, 21, 1>::value, vnode_base_offset_pair<1,  80, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 21, 2>::value, vnode_base_offset_pair<1,  80, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 22, 0>::value, vnode_base_offset_pair<1,  80, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1,  80, 22, 1>::value, vnode_base_offset_pair<1,  80, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 22, 2>::value, vnode_base_offset_pair<1,  80, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 23, 0>::value, vnode_base_offset_pair<1,  80, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1,  80, 23, 1>::value, vnode_base_offset_pair<1,  80, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 23, 2>::value, vnode_base_offset_pair<1,  80, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 24, 0>::value, vnode_base_offset_pair<1,  80, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1,  80, 24, 1>::value, vnode_base_offset_pair<1,  80, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 24, 2>::value, vnode_base_offset_pair<1,  80, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 25, 0>::value, vnode_base_offset_pair<1,  80, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1,  80, 25, 1>::value, vnode_base_offset_pair<1,  80, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 25, 2>::value, vnode_base_offset_pair<1,  80, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 26, 0>::value, vnode_base_offset_pair<1,  80, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1,  80, 26, 1>::value, vnode_base_offset_pair<1,  80, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 26, 2>::value, vnode_base_offset_pair<1,  80, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 27, 0>::value, vnode_base_offset_pair<1,  80, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1,  80, 27, 1>::value, vnode_base_offset_pair<1,  80, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 28, 0>::value, vnode_base_offset_pair<1,  80, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1,  80, 28, 1>::value, vnode_base_offset_pair<1,  80, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 28, 2>::value, vnode_base_offset_pair<1,  80, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 29, 0>::value, vnode_base_offset_pair<1,  80, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1,  80, 29, 1>::value, vnode_base_offset_pair<1,  80, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 29, 2>::value, vnode_base_offset_pair<1,  80, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 30, 0>::value, vnode_base_offset_pair<1,  80, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1,  80, 30, 1>::value, vnode_base_offset_pair<1,  80, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 30, 2>::value, vnode_base_offset_pair<1,  80, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 31, 0>::value, vnode_base_offset_pair<1,  80, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1,  80, 31, 1>::value, vnode_base_offset_pair<1,  80, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 31, 2>::value, vnode_base_offset_pair<1,  80, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 32, 0>::value, vnode_base_offset_pair<1,  80, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1,  80, 32, 1>::value, vnode_base_offset_pair<1,  80, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 32, 2>::value, vnode_base_offset_pair<1,  80, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 33, 0>::value, vnode_base_offset_pair<1,  80, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1,  80, 33, 1>::value, vnode_base_offset_pair<1,  80, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 33, 2>::value, vnode_base_offset_pair<1,  80, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 34, 0>::value, vnode_base_offset_pair<1,  80, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1,  80, 34, 1>::value, vnode_base_offset_pair<1,  80, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 34, 2>::value, vnode_base_offset_pair<1,  80, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 35, 0>::value, vnode_base_offset_pair<1,  80, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1,  80, 35, 1>::value, vnode_base_offset_pair<1,  80, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 35, 2>::value, vnode_base_offset_pair<1,  80, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 36, 0>::value, vnode_base_offset_pair<1,  80, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1,  80, 36, 1>::value, vnode_base_offset_pair<1,  80, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 36, 2>::value, vnode_base_offset_pair<1,  80, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 37, 0>::value, vnode_base_offset_pair<1,  80, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1,  80, 37, 1>::value, vnode_base_offset_pair<1,  80, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 38, 0>::value, vnode_base_offset_pair<1,  80, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1,  80, 38, 1>::value, vnode_base_offset_pair<1,  80, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 38, 2>::value, vnode_base_offset_pair<1,  80, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 39, 0>::value, vnode_base_offset_pair<1,  80, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1,  80, 39, 1>::value, vnode_base_offset_pair<1,  80, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 39, 2>::value, vnode_base_offset_pair<1,  80, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 40, 0>::value, vnode_base_offset_pair<1,  80, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1,  80, 40, 1>::value, vnode_base_offset_pair<1,  80, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 41, 0>::value, vnode_base_offset_pair<1,  80, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1,  80, 41, 1>::value, vnode_base_offset_pair<1,  80, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 41, 2>::value, vnode_base_offset_pair<1,  80, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 42, 0>::value, vnode_base_offset_pair<1,  80, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1,  80, 42, 1>::value, vnode_base_offset_pair<1,  80, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 43, 0>::value, vnode_base_offset_pair<1,  80, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1,  80, 43, 1>::value, vnode_base_offset_pair<1,  80, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 43, 2>::value, vnode_base_offset_pair<1,  80, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 44, 0>::value, vnode_base_offset_pair<1,  80, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1,  80, 44, 1>::value, vnode_base_offset_pair<1,  80, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  80, 44, 2>::value, vnode_base_offset_pair<1,  80, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  80, 45, 0>::value, vnode_base_offset_pair<1,  80, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1,  80, 45, 1>::value, vnode_base_offset_pair<1,  80, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z88_8 =
{
    {
        { vnode_shift_mod_pair<1,  88,  0, 0>::value, vnode_base_offset_pair<1,  88,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1,  88,  0, 1>::value, vnode_base_offset_pair<1,  88,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  0, 2>::value, vnode_base_offset_pair<1,  88,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  0, 3>::value, vnode_base_offset_pair<1,  88,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  0, 4>::value, vnode_base_offset_pair<1,  88,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  0, 5>::value, vnode_base_offset_pair<1,  88,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  0, 6>::value, vnode_base_offset_pair<1,  88,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  0, 7>::value, vnode_base_offset_pair<1,  88,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  0, 8>::value, vnode_base_offset_pair<1,  88,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  0, 9>::value, vnode_base_offset_pair<1,  88,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  88,  1, 0>::value, vnode_base_offset_pair<1,  88,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1,  88,  1, 1>::value, vnode_base_offset_pair<1,  88,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  1, 2>::value, vnode_base_offset_pair<1,  88,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  1, 3>::value, vnode_base_offset_pair<1,  88,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  1, 4>::value, vnode_base_offset_pair<1,  88,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  1, 5>::value, vnode_base_offset_pair<1,  88,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  1, 6>::value, vnode_base_offset_pair<1,  88,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  1, 7>::value, vnode_base_offset_pair<1,  88,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  1, 8>::value, vnode_base_offset_pair<1,  88,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  1, 9>::value, vnode_base_offset_pair<1,  88,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  88,  2, 0>::value, vnode_base_offset_pair<1,  88,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1,  88,  2, 1>::value, vnode_base_offset_pair<1,  88,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  2, 2>::value, vnode_base_offset_pair<1,  88,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  2, 3>::value, vnode_base_offset_pair<1,  88,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  2, 4>::value, vnode_base_offset_pair<1,  88,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  2, 5>::value, vnode_base_offset_pair<1,  88,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  2, 6>::value, vnode_base_offset_pair<1,  88,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  2, 7>::value, vnode_base_offset_pair<1,  88,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  2, 8>::value, vnode_base_offset_pair<1,  88,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  2, 9>::value, vnode_base_offset_pair<1,  88,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  88,  3, 0>::value, vnode_base_offset_pair<1,  88,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1,  88,  3, 1>::value, vnode_base_offset_pair<1,  88,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  3, 2>::value, vnode_base_offset_pair<1,  88,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  3, 3>::value, vnode_base_offset_pair<1,  88,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  3, 4>::value, vnode_base_offset_pair<1,  88,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  3, 5>::value, vnode_base_offset_pair<1,  88,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  3, 6>::value, vnode_base_offset_pair<1,  88,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  3, 7>::value, vnode_base_offset_pair<1,  88,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  3, 8>::value, vnode_base_offset_pair<1,  88,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  3, 9>::value, vnode_base_offset_pair<1,  88,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  88,  4, 0>::value, vnode_base_offset_pair<1,  88,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1,  88,  4, 1>::value, vnode_base_offset_pair<1,  88,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  88,  5, 0>::value, vnode_base_offset_pair<1,  88,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1,  88,  5, 1>::value, vnode_base_offset_pair<1,  88,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  5, 2>::value, vnode_base_offset_pair<1,  88,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  5, 3>::value, vnode_base_offset_pair<1,  88,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  88,  6, 0>::value, vnode_base_offset_pair<1,  88,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1,  88,  6, 1>::value, vnode_base_offset_pair<1,  88,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  6, 2>::value, vnode_base_offset_pair<1,  88,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  6, 3>::value, vnode_base_offset_pair<1,  88,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  6, 4>::value, vnode_base_offset_pair<1,  88,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  88,  7, 0>::value, vnode_base_offset_pair<1,  88,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1,  88,  7, 1>::value, vnode_base_offset_pair<1,  88,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  7, 2>::value, vnode_base_offset_pair<1,  88,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  7, 3>::value, vnode_base_offset_pair<1,  88,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  88,  8, 0>::value, vnode_base_offset_pair<1,  88,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1,  88,  8, 1>::value, vnode_base_offset_pair<1,  88,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  8, 2>::value, vnode_base_offset_pair<1,  88,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  8, 3>::value, vnode_base_offset_pair<1,  88,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  8, 4>::value, vnode_base_offset_pair<1,  88,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  88,  9, 0>::value, vnode_base_offset_pair<1,  88,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1,  88,  9, 1>::value, vnode_base_offset_pair<1,  88,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  9, 2>::value, vnode_base_offset_pair<1,  88,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  9, 3>::value, vnode_base_offset_pair<1,  88,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  88,  9, 4>::value, vnode_base_offset_pair<1,  88,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 10, 0>::value, vnode_base_offset_pair<1,  88, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1,  88, 10, 1>::value, vnode_base_offset_pair<1,  88, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 10, 2>::value, vnode_base_offset_pair<1,  88, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 10, 3>::value, vnode_base_offset_pair<1,  88, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 11, 0>::value, vnode_base_offset_pair<1,  88, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1,  88, 11, 1>::value, vnode_base_offset_pair<1,  88, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 11, 2>::value, vnode_base_offset_pair<1,  88, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 11, 3>::value, vnode_base_offset_pair<1,  88, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 12, 0>::value, vnode_base_offset_pair<1,  88, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1,  88, 12, 1>::value, vnode_base_offset_pair<1,  88, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 12, 2>::value, vnode_base_offset_pair<1,  88, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 12, 3>::value, vnode_base_offset_pair<1,  88, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 13, 0>::value, vnode_base_offset_pair<1,  88, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1,  88, 13, 1>::value, vnode_base_offset_pair<1,  88, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 13, 2>::value, vnode_base_offset_pair<1,  88, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 14, 0>::value, vnode_base_offset_pair<1,  88, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1,  88, 14, 1>::value, vnode_base_offset_pair<1,  88, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 14, 2>::value, vnode_base_offset_pair<1,  88, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 14, 3>::value, vnode_base_offset_pair<1,  88, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 15, 0>::value, vnode_base_offset_pair<1,  88, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1,  88, 15, 1>::value, vnode_base_offset_pair<1,  88, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 15, 2>::value, vnode_base_offset_pair<1,  88, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 15, 3>::value, vnode_base_offset_pair<1,  88, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 16, 0>::value, vnode_base_offset_pair<1,  88, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1,  88, 16, 1>::value, vnode_base_offset_pair<1,  88, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 16, 2>::value, vnode_base_offset_pair<1,  88, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 17, 0>::value, vnode_base_offset_pair<1,  88, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1,  88, 17, 1>::value, vnode_base_offset_pair<1,  88, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 17, 2>::value, vnode_base_offset_pair<1,  88, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 18, 0>::value, vnode_base_offset_pair<1,  88, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1,  88, 18, 1>::value, vnode_base_offset_pair<1,  88, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 18, 2>::value, vnode_base_offset_pair<1,  88, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 19, 0>::value, vnode_base_offset_pair<1,  88, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1,  88, 19, 1>::value, vnode_base_offset_pair<1,  88, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 19, 2>::value, vnode_base_offset_pair<1,  88, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 20, 0>::value, vnode_base_offset_pair<1,  88, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1,  88, 20, 1>::value, vnode_base_offset_pair<1,  88, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 20, 2>::value, vnode_base_offset_pair<1,  88, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 21, 0>::value, vnode_base_offset_pair<1,  88, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1,  88, 21, 1>::value, vnode_base_offset_pair<1,  88, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 21, 2>::value, vnode_base_offset_pair<1,  88, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 22, 0>::value, vnode_base_offset_pair<1,  88, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1,  88, 22, 1>::value, vnode_base_offset_pair<1,  88, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 22, 2>::value, vnode_base_offset_pair<1,  88, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 23, 0>::value, vnode_base_offset_pair<1,  88, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1,  88, 23, 1>::value, vnode_base_offset_pair<1,  88, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 23, 2>::value, vnode_base_offset_pair<1,  88, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 24, 0>::value, vnode_base_offset_pair<1,  88, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1,  88, 24, 1>::value, vnode_base_offset_pair<1,  88, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 24, 2>::value, vnode_base_offset_pair<1,  88, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 25, 0>::value, vnode_base_offset_pair<1,  88, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1,  88, 25, 1>::value, vnode_base_offset_pair<1,  88, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 25, 2>::value, vnode_base_offset_pair<1,  88, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 26, 0>::value, vnode_base_offset_pair<1,  88, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1,  88, 26, 1>::value, vnode_base_offset_pair<1,  88, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 26, 2>::value, vnode_base_offset_pair<1,  88, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 27, 0>::value, vnode_base_offset_pair<1,  88, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1,  88, 27, 1>::value, vnode_base_offset_pair<1,  88, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 28, 0>::value, vnode_base_offset_pair<1,  88, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1,  88, 28, 1>::value, vnode_base_offset_pair<1,  88, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 28, 2>::value, vnode_base_offset_pair<1,  88, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 29, 0>::value, vnode_base_offset_pair<1,  88, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1,  88, 29, 1>::value, vnode_base_offset_pair<1,  88, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 29, 2>::value, vnode_base_offset_pair<1,  88, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 30, 0>::value, vnode_base_offset_pair<1,  88, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1,  88, 30, 1>::value, vnode_base_offset_pair<1,  88, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 30, 2>::value, vnode_base_offset_pair<1,  88, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 31, 0>::value, vnode_base_offset_pair<1,  88, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1,  88, 31, 1>::value, vnode_base_offset_pair<1,  88, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 31, 2>::value, vnode_base_offset_pair<1,  88, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 32, 0>::value, vnode_base_offset_pair<1,  88, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1,  88, 32, 1>::value, vnode_base_offset_pair<1,  88, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 32, 2>::value, vnode_base_offset_pair<1,  88, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 33, 0>::value, vnode_base_offset_pair<1,  88, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1,  88, 33, 1>::value, vnode_base_offset_pair<1,  88, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 33, 2>::value, vnode_base_offset_pair<1,  88, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 34, 0>::value, vnode_base_offset_pair<1,  88, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1,  88, 34, 1>::value, vnode_base_offset_pair<1,  88, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 34, 2>::value, vnode_base_offset_pair<1,  88, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 35, 0>::value, vnode_base_offset_pair<1,  88, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1,  88, 35, 1>::value, vnode_base_offset_pair<1,  88, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 35, 2>::value, vnode_base_offset_pair<1,  88, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 36, 0>::value, vnode_base_offset_pair<1,  88, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1,  88, 36, 1>::value, vnode_base_offset_pair<1,  88, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 36, 2>::value, vnode_base_offset_pair<1,  88, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 37, 0>::value, vnode_base_offset_pair<1,  88, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1,  88, 37, 1>::value, vnode_base_offset_pair<1,  88, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 38, 0>::value, vnode_base_offset_pair<1,  88, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1,  88, 38, 1>::value, vnode_base_offset_pair<1,  88, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 38, 2>::value, vnode_base_offset_pair<1,  88, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 39, 0>::value, vnode_base_offset_pair<1,  88, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1,  88, 39, 1>::value, vnode_base_offset_pair<1,  88, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 39, 2>::value, vnode_base_offset_pair<1,  88, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 40, 0>::value, vnode_base_offset_pair<1,  88, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1,  88, 40, 1>::value, vnode_base_offset_pair<1,  88, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 41, 0>::value, vnode_base_offset_pair<1,  88, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1,  88, 41, 1>::value, vnode_base_offset_pair<1,  88, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 41, 2>::value, vnode_base_offset_pair<1,  88, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 42, 0>::value, vnode_base_offset_pair<1,  88, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1,  88, 42, 1>::value, vnode_base_offset_pair<1,  88, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 43, 0>::value, vnode_base_offset_pair<1,  88, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1,  88, 43, 1>::value, vnode_base_offset_pair<1,  88, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 43, 2>::value, vnode_base_offset_pair<1,  88, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 44, 0>::value, vnode_base_offset_pair<1,  88, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1,  88, 44, 1>::value, vnode_base_offset_pair<1,  88, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  88, 44, 2>::value, vnode_base_offset_pair<1,  88, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  88, 45, 0>::value, vnode_base_offset_pair<1,  88, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1,  88, 45, 1>::value, vnode_base_offset_pair<1,  88, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z96_8 =
{
    {
        { vnode_shift_mod_pair<1,  96,  0, 0>::value, vnode_base_offset_pair<1,  96,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1,  96,  0, 1>::value, vnode_base_offset_pair<1,  96,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  0, 2>::value, vnode_base_offset_pair<1,  96,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  0, 3>::value, vnode_base_offset_pair<1,  96,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  0, 4>::value, vnode_base_offset_pair<1,  96,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  0, 5>::value, vnode_base_offset_pair<1,  96,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  0, 6>::value, vnode_base_offset_pair<1,  96,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  0, 7>::value, vnode_base_offset_pair<1,  96,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  0, 8>::value, vnode_base_offset_pair<1,  96,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  0, 9>::value, vnode_base_offset_pair<1,  96,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  96,  1, 0>::value, vnode_base_offset_pair<1,  96,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1,  96,  1, 1>::value, vnode_base_offset_pair<1,  96,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  1, 2>::value, vnode_base_offset_pair<1,  96,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  1, 3>::value, vnode_base_offset_pair<1,  96,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  1, 4>::value, vnode_base_offset_pair<1,  96,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  1, 5>::value, vnode_base_offset_pair<1,  96,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  1, 6>::value, vnode_base_offset_pair<1,  96,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  1, 7>::value, vnode_base_offset_pair<1,  96,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  1, 8>::value, vnode_base_offset_pair<1,  96,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  1, 9>::value, vnode_base_offset_pair<1,  96,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  96,  2, 0>::value, vnode_base_offset_pair<1,  96,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1,  96,  2, 1>::value, vnode_base_offset_pair<1,  96,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  2, 2>::value, vnode_base_offset_pair<1,  96,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  2, 3>::value, vnode_base_offset_pair<1,  96,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  2, 4>::value, vnode_base_offset_pair<1,  96,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  2, 5>::value, vnode_base_offset_pair<1,  96,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  2, 6>::value, vnode_base_offset_pair<1,  96,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  2, 7>::value, vnode_base_offset_pair<1,  96,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  2, 8>::value, vnode_base_offset_pair<1,  96,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  2, 9>::value, vnode_base_offset_pair<1,  96,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  96,  3, 0>::value, vnode_base_offset_pair<1,  96,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1,  96,  3, 1>::value, vnode_base_offset_pair<1,  96,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  3, 2>::value, vnode_base_offset_pair<1,  96,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  3, 3>::value, vnode_base_offset_pair<1,  96,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  3, 4>::value, vnode_base_offset_pair<1,  96,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  3, 5>::value, vnode_base_offset_pair<1,  96,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  3, 6>::value, vnode_base_offset_pair<1,  96,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  3, 7>::value, vnode_base_offset_pair<1,  96,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  3, 8>::value, vnode_base_offset_pair<1,  96,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  3, 9>::value, vnode_base_offset_pair<1,  96,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1,  96,  4, 0>::value, vnode_base_offset_pair<1,  96,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1,  96,  4, 1>::value, vnode_base_offset_pair<1,  96,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  96,  5, 0>::value, vnode_base_offset_pair<1,  96,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1,  96,  5, 1>::value, vnode_base_offset_pair<1,  96,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  5, 2>::value, vnode_base_offset_pair<1,  96,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  5, 3>::value, vnode_base_offset_pair<1,  96,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  96,  6, 0>::value, vnode_base_offset_pair<1,  96,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1,  96,  6, 1>::value, vnode_base_offset_pair<1,  96,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  6, 2>::value, vnode_base_offset_pair<1,  96,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  6, 3>::value, vnode_base_offset_pair<1,  96,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  6, 4>::value, vnode_base_offset_pair<1,  96,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  96,  7, 0>::value, vnode_base_offset_pair<1,  96,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1,  96,  7, 1>::value, vnode_base_offset_pair<1,  96,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  7, 2>::value, vnode_base_offset_pair<1,  96,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  7, 3>::value, vnode_base_offset_pair<1,  96,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  96,  8, 0>::value, vnode_base_offset_pair<1,  96,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1,  96,  8, 1>::value, vnode_base_offset_pair<1,  96,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  8, 2>::value, vnode_base_offset_pair<1,  96,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  8, 3>::value, vnode_base_offset_pair<1,  96,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  8, 4>::value, vnode_base_offset_pair<1,  96,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  96,  9, 0>::value, vnode_base_offset_pair<1,  96,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1,  96,  9, 1>::value, vnode_base_offset_pair<1,  96,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  9, 2>::value, vnode_base_offset_pair<1,  96,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  9, 3>::value, vnode_base_offset_pair<1,  96,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1,  96,  9, 4>::value, vnode_base_offset_pair<1,  96,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 10, 0>::value, vnode_base_offset_pair<1,  96, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1,  96, 10, 1>::value, vnode_base_offset_pair<1,  96, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 10, 2>::value, vnode_base_offset_pair<1,  96, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 10, 3>::value, vnode_base_offset_pair<1,  96, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 11, 0>::value, vnode_base_offset_pair<1,  96, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1,  96, 11, 1>::value, vnode_base_offset_pair<1,  96, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 11, 2>::value, vnode_base_offset_pair<1,  96, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 11, 3>::value, vnode_base_offset_pair<1,  96, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 12, 0>::value, vnode_base_offset_pair<1,  96, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1,  96, 12, 1>::value, vnode_base_offset_pair<1,  96, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 12, 2>::value, vnode_base_offset_pair<1,  96, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 12, 3>::value, vnode_base_offset_pair<1,  96, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 13, 0>::value, vnode_base_offset_pair<1,  96, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1,  96, 13, 1>::value, vnode_base_offset_pair<1,  96, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 13, 2>::value, vnode_base_offset_pair<1,  96, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 14, 0>::value, vnode_base_offset_pair<1,  96, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1,  96, 14, 1>::value, vnode_base_offset_pair<1,  96, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 14, 2>::value, vnode_base_offset_pair<1,  96, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 14, 3>::value, vnode_base_offset_pair<1,  96, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 15, 0>::value, vnode_base_offset_pair<1,  96, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1,  96, 15, 1>::value, vnode_base_offset_pair<1,  96, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 15, 2>::value, vnode_base_offset_pair<1,  96, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 15, 3>::value, vnode_base_offset_pair<1,  96, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 16, 0>::value, vnode_base_offset_pair<1,  96, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1,  96, 16, 1>::value, vnode_base_offset_pair<1,  96, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 16, 2>::value, vnode_base_offset_pair<1,  96, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 17, 0>::value, vnode_base_offset_pair<1,  96, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1,  96, 17, 1>::value, vnode_base_offset_pair<1,  96, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 17, 2>::value, vnode_base_offset_pair<1,  96, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 18, 0>::value, vnode_base_offset_pair<1,  96, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1,  96, 18, 1>::value, vnode_base_offset_pair<1,  96, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 18, 2>::value, vnode_base_offset_pair<1,  96, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 19, 0>::value, vnode_base_offset_pair<1,  96, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1,  96, 19, 1>::value, vnode_base_offset_pair<1,  96, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 19, 2>::value, vnode_base_offset_pair<1,  96, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 20, 0>::value, vnode_base_offset_pair<1,  96, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1,  96, 20, 1>::value, vnode_base_offset_pair<1,  96, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 20, 2>::value, vnode_base_offset_pair<1,  96, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 21, 0>::value, vnode_base_offset_pair<1,  96, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1,  96, 21, 1>::value, vnode_base_offset_pair<1,  96, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 21, 2>::value, vnode_base_offset_pair<1,  96, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 22, 0>::value, vnode_base_offset_pair<1,  96, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1,  96, 22, 1>::value, vnode_base_offset_pair<1,  96, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 22, 2>::value, vnode_base_offset_pair<1,  96, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 23, 0>::value, vnode_base_offset_pair<1,  96, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1,  96, 23, 1>::value, vnode_base_offset_pair<1,  96, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 23, 2>::value, vnode_base_offset_pair<1,  96, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 24, 0>::value, vnode_base_offset_pair<1,  96, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1,  96, 24, 1>::value, vnode_base_offset_pair<1,  96, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 24, 2>::value, vnode_base_offset_pair<1,  96, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 25, 0>::value, vnode_base_offset_pair<1,  96, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1,  96, 25, 1>::value, vnode_base_offset_pair<1,  96, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 25, 2>::value, vnode_base_offset_pair<1,  96, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 26, 0>::value, vnode_base_offset_pair<1,  96, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1,  96, 26, 1>::value, vnode_base_offset_pair<1,  96, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 26, 2>::value, vnode_base_offset_pair<1,  96, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 27, 0>::value, vnode_base_offset_pair<1,  96, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1,  96, 27, 1>::value, vnode_base_offset_pair<1,  96, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 28, 0>::value, vnode_base_offset_pair<1,  96, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1,  96, 28, 1>::value, vnode_base_offset_pair<1,  96, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 28, 2>::value, vnode_base_offset_pair<1,  96, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 29, 0>::value, vnode_base_offset_pair<1,  96, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1,  96, 29, 1>::value, vnode_base_offset_pair<1,  96, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 29, 2>::value, vnode_base_offset_pair<1,  96, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 30, 0>::value, vnode_base_offset_pair<1,  96, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1,  96, 30, 1>::value, vnode_base_offset_pair<1,  96, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 30, 2>::value, vnode_base_offset_pair<1,  96, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 31, 0>::value, vnode_base_offset_pair<1,  96, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1,  96, 31, 1>::value, vnode_base_offset_pair<1,  96, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 31, 2>::value, vnode_base_offset_pair<1,  96, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 32, 0>::value, vnode_base_offset_pair<1,  96, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1,  96, 32, 1>::value, vnode_base_offset_pair<1,  96, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 32, 2>::value, vnode_base_offset_pair<1,  96, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 33, 0>::value, vnode_base_offset_pair<1,  96, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1,  96, 33, 1>::value, vnode_base_offset_pair<1,  96, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 33, 2>::value, vnode_base_offset_pair<1,  96, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 34, 0>::value, vnode_base_offset_pair<1,  96, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1,  96, 34, 1>::value, vnode_base_offset_pair<1,  96, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 34, 2>::value, vnode_base_offset_pair<1,  96, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 35, 0>::value, vnode_base_offset_pair<1,  96, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1,  96, 35, 1>::value, vnode_base_offset_pair<1,  96, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 35, 2>::value, vnode_base_offset_pair<1,  96, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 36, 0>::value, vnode_base_offset_pair<1,  96, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1,  96, 36, 1>::value, vnode_base_offset_pair<1,  96, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 36, 2>::value, vnode_base_offset_pair<1,  96, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 37, 0>::value, vnode_base_offset_pair<1,  96, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1,  96, 37, 1>::value, vnode_base_offset_pair<1,  96, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 38, 0>::value, vnode_base_offset_pair<1,  96, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1,  96, 38, 1>::value, vnode_base_offset_pair<1,  96, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 38, 2>::value, vnode_base_offset_pair<1,  96, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 39, 0>::value, vnode_base_offset_pair<1,  96, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1,  96, 39, 1>::value, vnode_base_offset_pair<1,  96, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 39, 2>::value, vnode_base_offset_pair<1,  96, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 40, 0>::value, vnode_base_offset_pair<1,  96, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1,  96, 40, 1>::value, vnode_base_offset_pair<1,  96, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 41, 0>::value, vnode_base_offset_pair<1,  96, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1,  96, 41, 1>::value, vnode_base_offset_pair<1,  96, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 41, 2>::value, vnode_base_offset_pair<1,  96, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 42, 0>::value, vnode_base_offset_pair<1,  96, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1,  96, 42, 1>::value, vnode_base_offset_pair<1,  96, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 43, 0>::value, vnode_base_offset_pair<1,  96, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1,  96, 43, 1>::value, vnode_base_offset_pair<1,  96, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 43, 2>::value, vnode_base_offset_pair<1,  96, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 44, 0>::value, vnode_base_offset_pair<1,  96, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1,  96, 44, 1>::value, vnode_base_offset_pair<1,  96, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1,  96, 44, 2>::value, vnode_base_offset_pair<1,  96, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1,  96, 45, 0>::value, vnode_base_offset_pair<1,  96, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1,  96, 45, 1>::value, vnode_base_offset_pair<1,  96, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z104_8 =
{
    {
        { vnode_shift_mod_pair<1, 104,  0, 0>::value, vnode_base_offset_pair<1, 104,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1, 104,  0, 1>::value, vnode_base_offset_pair<1, 104,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  0, 2>::value, vnode_base_offset_pair<1, 104,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  0, 3>::value, vnode_base_offset_pair<1, 104,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  0, 4>::value, vnode_base_offset_pair<1, 104,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  0, 5>::value, vnode_base_offset_pair<1, 104,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  0, 6>::value, vnode_base_offset_pair<1, 104,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  0, 7>::value, vnode_base_offset_pair<1, 104,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  0, 8>::value, vnode_base_offset_pair<1, 104,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  0, 9>::value, vnode_base_offset_pair<1, 104,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 104,  1, 0>::value, vnode_base_offset_pair<1, 104,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1, 104,  1, 1>::value, vnode_base_offset_pair<1, 104,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  1, 2>::value, vnode_base_offset_pair<1, 104,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  1, 3>::value, vnode_base_offset_pair<1, 104,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  1, 4>::value, vnode_base_offset_pair<1, 104,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  1, 5>::value, vnode_base_offset_pair<1, 104,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  1, 6>::value, vnode_base_offset_pair<1, 104,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  1, 7>::value, vnode_base_offset_pair<1, 104,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  1, 8>::value, vnode_base_offset_pair<1, 104,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  1, 9>::value, vnode_base_offset_pair<1, 104,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 104,  2, 0>::value, vnode_base_offset_pair<1, 104,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1, 104,  2, 1>::value, vnode_base_offset_pair<1, 104,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  2, 2>::value, vnode_base_offset_pair<1, 104,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  2, 3>::value, vnode_base_offset_pair<1, 104,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  2, 4>::value, vnode_base_offset_pair<1, 104,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  2, 5>::value, vnode_base_offset_pair<1, 104,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  2, 6>::value, vnode_base_offset_pair<1, 104,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  2, 7>::value, vnode_base_offset_pair<1, 104,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  2, 8>::value, vnode_base_offset_pair<1, 104,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  2, 9>::value, vnode_base_offset_pair<1, 104,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 104,  3, 0>::value, vnode_base_offset_pair<1, 104,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1, 104,  3, 1>::value, vnode_base_offset_pair<1, 104,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  3, 2>::value, vnode_base_offset_pair<1, 104,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  3, 3>::value, vnode_base_offset_pair<1, 104,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  3, 4>::value, vnode_base_offset_pair<1, 104,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  3, 5>::value, vnode_base_offset_pair<1, 104,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  3, 6>::value, vnode_base_offset_pair<1, 104,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  3, 7>::value, vnode_base_offset_pair<1, 104,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  3, 8>::value, vnode_base_offset_pair<1, 104,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  3, 9>::value, vnode_base_offset_pair<1, 104,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 104,  4, 0>::value, vnode_base_offset_pair<1, 104,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1, 104,  4, 1>::value, vnode_base_offset_pair<1, 104,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 104,  5, 0>::value, vnode_base_offset_pair<1, 104,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1, 104,  5, 1>::value, vnode_base_offset_pair<1, 104,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  5, 2>::value, vnode_base_offset_pair<1, 104,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  5, 3>::value, vnode_base_offset_pair<1, 104,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 104,  6, 0>::value, vnode_base_offset_pair<1, 104,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1, 104,  6, 1>::value, vnode_base_offset_pair<1, 104,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  6, 2>::value, vnode_base_offset_pair<1, 104,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  6, 3>::value, vnode_base_offset_pair<1, 104,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  6, 4>::value, vnode_base_offset_pair<1, 104,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 104,  7, 0>::value, vnode_base_offset_pair<1, 104,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1, 104,  7, 1>::value, vnode_base_offset_pair<1, 104,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  7, 2>::value, vnode_base_offset_pair<1, 104,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  7, 3>::value, vnode_base_offset_pair<1, 104,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 104,  8, 0>::value, vnode_base_offset_pair<1, 104,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1, 104,  8, 1>::value, vnode_base_offset_pair<1, 104,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  8, 2>::value, vnode_base_offset_pair<1, 104,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  8, 3>::value, vnode_base_offset_pair<1, 104,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  8, 4>::value, vnode_base_offset_pair<1, 104,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 104,  9, 0>::value, vnode_base_offset_pair<1, 104,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1, 104,  9, 1>::value, vnode_base_offset_pair<1, 104,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  9, 2>::value, vnode_base_offset_pair<1, 104,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  9, 3>::value, vnode_base_offset_pair<1, 104,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 104,  9, 4>::value, vnode_base_offset_pair<1, 104,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 10, 0>::value, vnode_base_offset_pair<1, 104, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1, 104, 10, 1>::value, vnode_base_offset_pair<1, 104, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 10, 2>::value, vnode_base_offset_pair<1, 104, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 10, 3>::value, vnode_base_offset_pair<1, 104, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 11, 0>::value, vnode_base_offset_pair<1, 104, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1, 104, 11, 1>::value, vnode_base_offset_pair<1, 104, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 11, 2>::value, vnode_base_offset_pair<1, 104, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 11, 3>::value, vnode_base_offset_pair<1, 104, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 12, 0>::value, vnode_base_offset_pair<1, 104, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1, 104, 12, 1>::value, vnode_base_offset_pair<1, 104, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 12, 2>::value, vnode_base_offset_pair<1, 104, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 12, 3>::value, vnode_base_offset_pair<1, 104, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 13, 0>::value, vnode_base_offset_pair<1, 104, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1, 104, 13, 1>::value, vnode_base_offset_pair<1, 104, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 13, 2>::value, vnode_base_offset_pair<1, 104, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 14, 0>::value, vnode_base_offset_pair<1, 104, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1, 104, 14, 1>::value, vnode_base_offset_pair<1, 104, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 14, 2>::value, vnode_base_offset_pair<1, 104, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 14, 3>::value, vnode_base_offset_pair<1, 104, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 15, 0>::value, vnode_base_offset_pair<1, 104, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1, 104, 15, 1>::value, vnode_base_offset_pair<1, 104, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 15, 2>::value, vnode_base_offset_pair<1, 104, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 15, 3>::value, vnode_base_offset_pair<1, 104, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 16, 0>::value, vnode_base_offset_pair<1, 104, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1, 104, 16, 1>::value, vnode_base_offset_pair<1, 104, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 16, 2>::value, vnode_base_offset_pair<1, 104, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 17, 0>::value, vnode_base_offset_pair<1, 104, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1, 104, 17, 1>::value, vnode_base_offset_pair<1, 104, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 17, 2>::value, vnode_base_offset_pair<1, 104, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 18, 0>::value, vnode_base_offset_pair<1, 104, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1, 104, 18, 1>::value, vnode_base_offset_pair<1, 104, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 18, 2>::value, vnode_base_offset_pair<1, 104, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 19, 0>::value, vnode_base_offset_pair<1, 104, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1, 104, 19, 1>::value, vnode_base_offset_pair<1, 104, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 19, 2>::value, vnode_base_offset_pair<1, 104, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 20, 0>::value, vnode_base_offset_pair<1, 104, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1, 104, 20, 1>::value, vnode_base_offset_pair<1, 104, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 20, 2>::value, vnode_base_offset_pair<1, 104, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 21, 0>::value, vnode_base_offset_pair<1, 104, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1, 104, 21, 1>::value, vnode_base_offset_pair<1, 104, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 21, 2>::value, vnode_base_offset_pair<1, 104, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 22, 0>::value, vnode_base_offset_pair<1, 104, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1, 104, 22, 1>::value, vnode_base_offset_pair<1, 104, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 22, 2>::value, vnode_base_offset_pair<1, 104, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 23, 0>::value, vnode_base_offset_pair<1, 104, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1, 104, 23, 1>::value, vnode_base_offset_pair<1, 104, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 23, 2>::value, vnode_base_offset_pair<1, 104, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 24, 0>::value, vnode_base_offset_pair<1, 104, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1, 104, 24, 1>::value, vnode_base_offset_pair<1, 104, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 24, 2>::value, vnode_base_offset_pair<1, 104, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 25, 0>::value, vnode_base_offset_pair<1, 104, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1, 104, 25, 1>::value, vnode_base_offset_pair<1, 104, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 25, 2>::value, vnode_base_offset_pair<1, 104, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 26, 0>::value, vnode_base_offset_pair<1, 104, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1, 104, 26, 1>::value, vnode_base_offset_pair<1, 104, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 26, 2>::value, vnode_base_offset_pair<1, 104, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 27, 0>::value, vnode_base_offset_pair<1, 104, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1, 104, 27, 1>::value, vnode_base_offset_pair<1, 104, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 28, 0>::value, vnode_base_offset_pair<1, 104, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1, 104, 28, 1>::value, vnode_base_offset_pair<1, 104, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 28, 2>::value, vnode_base_offset_pair<1, 104, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 29, 0>::value, vnode_base_offset_pair<1, 104, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1, 104, 29, 1>::value, vnode_base_offset_pair<1, 104, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 29, 2>::value, vnode_base_offset_pair<1, 104, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 30, 0>::value, vnode_base_offset_pair<1, 104, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1, 104, 30, 1>::value, vnode_base_offset_pair<1, 104, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 30, 2>::value, vnode_base_offset_pair<1, 104, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 31, 0>::value, vnode_base_offset_pair<1, 104, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1, 104, 31, 1>::value, vnode_base_offset_pair<1, 104, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 31, 2>::value, vnode_base_offset_pair<1, 104, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 32, 0>::value, vnode_base_offset_pair<1, 104, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1, 104, 32, 1>::value, vnode_base_offset_pair<1, 104, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 32, 2>::value, vnode_base_offset_pair<1, 104, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 33, 0>::value, vnode_base_offset_pair<1, 104, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1, 104, 33, 1>::value, vnode_base_offset_pair<1, 104, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 33, 2>::value, vnode_base_offset_pair<1, 104, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 34, 0>::value, vnode_base_offset_pair<1, 104, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1, 104, 34, 1>::value, vnode_base_offset_pair<1, 104, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 34, 2>::value, vnode_base_offset_pair<1, 104, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 35, 0>::value, vnode_base_offset_pair<1, 104, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1, 104, 35, 1>::value, vnode_base_offset_pair<1, 104, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 35, 2>::value, vnode_base_offset_pair<1, 104, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 36, 0>::value, vnode_base_offset_pair<1, 104, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1, 104, 36, 1>::value, vnode_base_offset_pair<1, 104, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 36, 2>::value, vnode_base_offset_pair<1, 104, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 37, 0>::value, vnode_base_offset_pair<1, 104, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1, 104, 37, 1>::value, vnode_base_offset_pair<1, 104, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 38, 0>::value, vnode_base_offset_pair<1, 104, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1, 104, 38, 1>::value, vnode_base_offset_pair<1, 104, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 38, 2>::value, vnode_base_offset_pair<1, 104, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 39, 0>::value, vnode_base_offset_pair<1, 104, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1, 104, 39, 1>::value, vnode_base_offset_pair<1, 104, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 39, 2>::value, vnode_base_offset_pair<1, 104, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 40, 0>::value, vnode_base_offset_pair<1, 104, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1, 104, 40, 1>::value, vnode_base_offset_pair<1, 104, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 41, 0>::value, vnode_base_offset_pair<1, 104, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1, 104, 41, 1>::value, vnode_base_offset_pair<1, 104, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 41, 2>::value, vnode_base_offset_pair<1, 104, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 42, 0>::value, vnode_base_offset_pair<1, 104, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1, 104, 42, 1>::value, vnode_base_offset_pair<1, 104, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 43, 0>::value, vnode_base_offset_pair<1, 104, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1, 104, 43, 1>::value, vnode_base_offset_pair<1, 104, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 43, 2>::value, vnode_base_offset_pair<1, 104, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 44, 0>::value, vnode_base_offset_pair<1, 104, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1, 104, 44, 1>::value, vnode_base_offset_pair<1, 104, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 104, 44, 2>::value, vnode_base_offset_pair<1, 104, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 104, 45, 0>::value, vnode_base_offset_pair<1, 104, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1, 104, 45, 1>::value, vnode_base_offset_pair<1, 104, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z112_8 =
{
    {
        { vnode_shift_mod_pair<1, 112,  0, 0>::value, vnode_base_offset_pair<1, 112,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1, 112,  0, 1>::value, vnode_base_offset_pair<1, 112,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  0, 2>::value, vnode_base_offset_pair<1, 112,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  0, 3>::value, vnode_base_offset_pair<1, 112,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  0, 4>::value, vnode_base_offset_pair<1, 112,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  0, 5>::value, vnode_base_offset_pair<1, 112,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  0, 6>::value, vnode_base_offset_pair<1, 112,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  0, 7>::value, vnode_base_offset_pair<1, 112,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  0, 8>::value, vnode_base_offset_pair<1, 112,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  0, 9>::value, vnode_base_offset_pair<1, 112,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 112,  1, 0>::value, vnode_base_offset_pair<1, 112,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1, 112,  1, 1>::value, vnode_base_offset_pair<1, 112,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  1, 2>::value, vnode_base_offset_pair<1, 112,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  1, 3>::value, vnode_base_offset_pair<1, 112,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  1, 4>::value, vnode_base_offset_pair<1, 112,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  1, 5>::value, vnode_base_offset_pair<1, 112,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  1, 6>::value, vnode_base_offset_pair<1, 112,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  1, 7>::value, vnode_base_offset_pair<1, 112,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  1, 8>::value, vnode_base_offset_pair<1, 112,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  1, 9>::value, vnode_base_offset_pair<1, 112,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 112,  2, 0>::value, vnode_base_offset_pair<1, 112,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1, 112,  2, 1>::value, vnode_base_offset_pair<1, 112,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  2, 2>::value, vnode_base_offset_pair<1, 112,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  2, 3>::value, vnode_base_offset_pair<1, 112,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  2, 4>::value, vnode_base_offset_pair<1, 112,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  2, 5>::value, vnode_base_offset_pair<1, 112,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  2, 6>::value, vnode_base_offset_pair<1, 112,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  2, 7>::value, vnode_base_offset_pair<1, 112,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  2, 8>::value, vnode_base_offset_pair<1, 112,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  2, 9>::value, vnode_base_offset_pair<1, 112,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 112,  3, 0>::value, vnode_base_offset_pair<1, 112,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1, 112,  3, 1>::value, vnode_base_offset_pair<1, 112,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  3, 2>::value, vnode_base_offset_pair<1, 112,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  3, 3>::value, vnode_base_offset_pair<1, 112,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  3, 4>::value, vnode_base_offset_pair<1, 112,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  3, 5>::value, vnode_base_offset_pair<1, 112,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  3, 6>::value, vnode_base_offset_pair<1, 112,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  3, 7>::value, vnode_base_offset_pair<1, 112,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  3, 8>::value, vnode_base_offset_pair<1, 112,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  3, 9>::value, vnode_base_offset_pair<1, 112,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 112,  4, 0>::value, vnode_base_offset_pair<1, 112,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1, 112,  4, 1>::value, vnode_base_offset_pair<1, 112,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 112,  5, 0>::value, vnode_base_offset_pair<1, 112,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1, 112,  5, 1>::value, vnode_base_offset_pair<1, 112,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  5, 2>::value, vnode_base_offset_pair<1, 112,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  5, 3>::value, vnode_base_offset_pair<1, 112,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 112,  6, 0>::value, vnode_base_offset_pair<1, 112,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1, 112,  6, 1>::value, vnode_base_offset_pair<1, 112,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  6, 2>::value, vnode_base_offset_pair<1, 112,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  6, 3>::value, vnode_base_offset_pair<1, 112,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  6, 4>::value, vnode_base_offset_pair<1, 112,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 112,  7, 0>::value, vnode_base_offset_pair<1, 112,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1, 112,  7, 1>::value, vnode_base_offset_pair<1, 112,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  7, 2>::value, vnode_base_offset_pair<1, 112,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  7, 3>::value, vnode_base_offset_pair<1, 112,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 112,  8, 0>::value, vnode_base_offset_pair<1, 112,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1, 112,  8, 1>::value, vnode_base_offset_pair<1, 112,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  8, 2>::value, vnode_base_offset_pair<1, 112,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  8, 3>::value, vnode_base_offset_pair<1, 112,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  8, 4>::value, vnode_base_offset_pair<1, 112,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 112,  9, 0>::value, vnode_base_offset_pair<1, 112,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1, 112,  9, 1>::value, vnode_base_offset_pair<1, 112,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  9, 2>::value, vnode_base_offset_pair<1, 112,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  9, 3>::value, vnode_base_offset_pair<1, 112,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 112,  9, 4>::value, vnode_base_offset_pair<1, 112,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 10, 0>::value, vnode_base_offset_pair<1, 112, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1, 112, 10, 1>::value, vnode_base_offset_pair<1, 112, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 10, 2>::value, vnode_base_offset_pair<1, 112, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 10, 3>::value, vnode_base_offset_pair<1, 112, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 11, 0>::value, vnode_base_offset_pair<1, 112, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1, 112, 11, 1>::value, vnode_base_offset_pair<1, 112, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 11, 2>::value, vnode_base_offset_pair<1, 112, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 11, 3>::value, vnode_base_offset_pair<1, 112, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 12, 0>::value, vnode_base_offset_pair<1, 112, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1, 112, 12, 1>::value, vnode_base_offset_pair<1, 112, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 12, 2>::value, vnode_base_offset_pair<1, 112, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 12, 3>::value, vnode_base_offset_pair<1, 112, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 13, 0>::value, vnode_base_offset_pair<1, 112, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1, 112, 13, 1>::value, vnode_base_offset_pair<1, 112, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 13, 2>::value, vnode_base_offset_pair<1, 112, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 14, 0>::value, vnode_base_offset_pair<1, 112, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1, 112, 14, 1>::value, vnode_base_offset_pair<1, 112, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 14, 2>::value, vnode_base_offset_pair<1, 112, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 14, 3>::value, vnode_base_offset_pair<1, 112, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 15, 0>::value, vnode_base_offset_pair<1, 112, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1, 112, 15, 1>::value, vnode_base_offset_pair<1, 112, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 15, 2>::value, vnode_base_offset_pair<1, 112, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 15, 3>::value, vnode_base_offset_pair<1, 112, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 16, 0>::value, vnode_base_offset_pair<1, 112, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1, 112, 16, 1>::value, vnode_base_offset_pair<1, 112, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 16, 2>::value, vnode_base_offset_pair<1, 112, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 17, 0>::value, vnode_base_offset_pair<1, 112, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1, 112, 17, 1>::value, vnode_base_offset_pair<1, 112, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 17, 2>::value, vnode_base_offset_pair<1, 112, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 18, 0>::value, vnode_base_offset_pair<1, 112, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1, 112, 18, 1>::value, vnode_base_offset_pair<1, 112, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 18, 2>::value, vnode_base_offset_pair<1, 112, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 19, 0>::value, vnode_base_offset_pair<1, 112, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1, 112, 19, 1>::value, vnode_base_offset_pair<1, 112, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 19, 2>::value, vnode_base_offset_pair<1, 112, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 20, 0>::value, vnode_base_offset_pair<1, 112, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1, 112, 20, 1>::value, vnode_base_offset_pair<1, 112, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 20, 2>::value, vnode_base_offset_pair<1, 112, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 21, 0>::value, vnode_base_offset_pair<1, 112, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1, 112, 21, 1>::value, vnode_base_offset_pair<1, 112, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 21, 2>::value, vnode_base_offset_pair<1, 112, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 22, 0>::value, vnode_base_offset_pair<1, 112, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1, 112, 22, 1>::value, vnode_base_offset_pair<1, 112, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 22, 2>::value, vnode_base_offset_pair<1, 112, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 23, 0>::value, vnode_base_offset_pair<1, 112, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1, 112, 23, 1>::value, vnode_base_offset_pair<1, 112, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 23, 2>::value, vnode_base_offset_pair<1, 112, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 24, 0>::value, vnode_base_offset_pair<1, 112, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1, 112, 24, 1>::value, vnode_base_offset_pair<1, 112, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 24, 2>::value, vnode_base_offset_pair<1, 112, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 25, 0>::value, vnode_base_offset_pair<1, 112, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1, 112, 25, 1>::value, vnode_base_offset_pair<1, 112, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 25, 2>::value, vnode_base_offset_pair<1, 112, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 26, 0>::value, vnode_base_offset_pair<1, 112, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1, 112, 26, 1>::value, vnode_base_offset_pair<1, 112, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 26, 2>::value, vnode_base_offset_pair<1, 112, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 27, 0>::value, vnode_base_offset_pair<1, 112, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1, 112, 27, 1>::value, vnode_base_offset_pair<1, 112, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 28, 0>::value, vnode_base_offset_pair<1, 112, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1, 112, 28, 1>::value, vnode_base_offset_pair<1, 112, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 28, 2>::value, vnode_base_offset_pair<1, 112, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 29, 0>::value, vnode_base_offset_pair<1, 112, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1, 112, 29, 1>::value, vnode_base_offset_pair<1, 112, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 29, 2>::value, vnode_base_offset_pair<1, 112, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 30, 0>::value, vnode_base_offset_pair<1, 112, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1, 112, 30, 1>::value, vnode_base_offset_pair<1, 112, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 30, 2>::value, vnode_base_offset_pair<1, 112, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 31, 0>::value, vnode_base_offset_pair<1, 112, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1, 112, 31, 1>::value, vnode_base_offset_pair<1, 112, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 31, 2>::value, vnode_base_offset_pair<1, 112, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 32, 0>::value, vnode_base_offset_pair<1, 112, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1, 112, 32, 1>::value, vnode_base_offset_pair<1, 112, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 32, 2>::value, vnode_base_offset_pair<1, 112, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 33, 0>::value, vnode_base_offset_pair<1, 112, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1, 112, 33, 1>::value, vnode_base_offset_pair<1, 112, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 33, 2>::value, vnode_base_offset_pair<1, 112, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 34, 0>::value, vnode_base_offset_pair<1, 112, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1, 112, 34, 1>::value, vnode_base_offset_pair<1, 112, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 34, 2>::value, vnode_base_offset_pair<1, 112, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 35, 0>::value, vnode_base_offset_pair<1, 112, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1, 112, 35, 1>::value, vnode_base_offset_pair<1, 112, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 35, 2>::value, vnode_base_offset_pair<1, 112, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 36, 0>::value, vnode_base_offset_pair<1, 112, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1, 112, 36, 1>::value, vnode_base_offset_pair<1, 112, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 36, 2>::value, vnode_base_offset_pair<1, 112, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 37, 0>::value, vnode_base_offset_pair<1, 112, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1, 112, 37, 1>::value, vnode_base_offset_pair<1, 112, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 38, 0>::value, vnode_base_offset_pair<1, 112, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1, 112, 38, 1>::value, vnode_base_offset_pair<1, 112, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 38, 2>::value, vnode_base_offset_pair<1, 112, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 39, 0>::value, vnode_base_offset_pair<1, 112, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1, 112, 39, 1>::value, vnode_base_offset_pair<1, 112, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 39, 2>::value, vnode_base_offset_pair<1, 112, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 40, 0>::value, vnode_base_offset_pair<1, 112, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1, 112, 40, 1>::value, vnode_base_offset_pair<1, 112, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 41, 0>::value, vnode_base_offset_pair<1, 112, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1, 112, 41, 1>::value, vnode_base_offset_pair<1, 112, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 41, 2>::value, vnode_base_offset_pair<1, 112, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 42, 0>::value, vnode_base_offset_pair<1, 112, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1, 112, 42, 1>::value, vnode_base_offset_pair<1, 112, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 43, 0>::value, vnode_base_offset_pair<1, 112, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1, 112, 43, 1>::value, vnode_base_offset_pair<1, 112, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 43, 2>::value, vnode_base_offset_pair<1, 112, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 44, 0>::value, vnode_base_offset_pair<1, 112, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1, 112, 44, 1>::value, vnode_base_offset_pair<1, 112, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 112, 44, 2>::value, vnode_base_offset_pair<1, 112, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 112, 45, 0>::value, vnode_base_offset_pair<1, 112, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1, 112, 45, 1>::value, vnode_base_offset_pair<1, 112, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z120_8 =
{
    {
        { vnode_shift_mod_pair<1, 120,  0, 0>::value, vnode_base_offset_pair<1, 120,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1, 120,  0, 1>::value, vnode_base_offset_pair<1, 120,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  0, 2>::value, vnode_base_offset_pair<1, 120,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  0, 3>::value, vnode_base_offset_pair<1, 120,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  0, 4>::value, vnode_base_offset_pair<1, 120,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  0, 5>::value, vnode_base_offset_pair<1, 120,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  0, 6>::value, vnode_base_offset_pair<1, 120,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  0, 7>::value, vnode_base_offset_pair<1, 120,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  0, 8>::value, vnode_base_offset_pair<1, 120,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  0, 9>::value, vnode_base_offset_pair<1, 120,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 120,  1, 0>::value, vnode_base_offset_pair<1, 120,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1, 120,  1, 1>::value, vnode_base_offset_pair<1, 120,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  1, 2>::value, vnode_base_offset_pair<1, 120,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  1, 3>::value, vnode_base_offset_pair<1, 120,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  1, 4>::value, vnode_base_offset_pair<1, 120,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  1, 5>::value, vnode_base_offset_pair<1, 120,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  1, 6>::value, vnode_base_offset_pair<1, 120,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  1, 7>::value, vnode_base_offset_pair<1, 120,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  1, 8>::value, vnode_base_offset_pair<1, 120,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  1, 9>::value, vnode_base_offset_pair<1, 120,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 120,  2, 0>::value, vnode_base_offset_pair<1, 120,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1, 120,  2, 1>::value, vnode_base_offset_pair<1, 120,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  2, 2>::value, vnode_base_offset_pair<1, 120,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  2, 3>::value, vnode_base_offset_pair<1, 120,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  2, 4>::value, vnode_base_offset_pair<1, 120,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  2, 5>::value, vnode_base_offset_pair<1, 120,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  2, 6>::value, vnode_base_offset_pair<1, 120,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  2, 7>::value, vnode_base_offset_pair<1, 120,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  2, 8>::value, vnode_base_offset_pair<1, 120,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  2, 9>::value, vnode_base_offset_pair<1, 120,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 120,  3, 0>::value, vnode_base_offset_pair<1, 120,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1, 120,  3, 1>::value, vnode_base_offset_pair<1, 120,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  3, 2>::value, vnode_base_offset_pair<1, 120,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  3, 3>::value, vnode_base_offset_pair<1, 120,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  3, 4>::value, vnode_base_offset_pair<1, 120,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  3, 5>::value, vnode_base_offset_pair<1, 120,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  3, 6>::value, vnode_base_offset_pair<1, 120,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  3, 7>::value, vnode_base_offset_pair<1, 120,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  3, 8>::value, vnode_base_offset_pair<1, 120,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  3, 9>::value, vnode_base_offset_pair<1, 120,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 120,  4, 0>::value, vnode_base_offset_pair<1, 120,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1, 120,  4, 1>::value, vnode_base_offset_pair<1, 120,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 120,  5, 0>::value, vnode_base_offset_pair<1, 120,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1, 120,  5, 1>::value, vnode_base_offset_pair<1, 120,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  5, 2>::value, vnode_base_offset_pair<1, 120,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  5, 3>::value, vnode_base_offset_pair<1, 120,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 120,  6, 0>::value, vnode_base_offset_pair<1, 120,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1, 120,  6, 1>::value, vnode_base_offset_pair<1, 120,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  6, 2>::value, vnode_base_offset_pair<1, 120,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  6, 3>::value, vnode_base_offset_pair<1, 120,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  6, 4>::value, vnode_base_offset_pair<1, 120,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 120,  7, 0>::value, vnode_base_offset_pair<1, 120,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1, 120,  7, 1>::value, vnode_base_offset_pair<1, 120,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  7, 2>::value, vnode_base_offset_pair<1, 120,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  7, 3>::value, vnode_base_offset_pair<1, 120,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 120,  8, 0>::value, vnode_base_offset_pair<1, 120,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1, 120,  8, 1>::value, vnode_base_offset_pair<1, 120,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  8, 2>::value, vnode_base_offset_pair<1, 120,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  8, 3>::value, vnode_base_offset_pair<1, 120,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  8, 4>::value, vnode_base_offset_pair<1, 120,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 120,  9, 0>::value, vnode_base_offset_pair<1, 120,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1, 120,  9, 1>::value, vnode_base_offset_pair<1, 120,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  9, 2>::value, vnode_base_offset_pair<1, 120,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  9, 3>::value, vnode_base_offset_pair<1, 120,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 120,  9, 4>::value, vnode_base_offset_pair<1, 120,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 10, 0>::value, vnode_base_offset_pair<1, 120, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1, 120, 10, 1>::value, vnode_base_offset_pair<1, 120, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 10, 2>::value, vnode_base_offset_pair<1, 120, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 10, 3>::value, vnode_base_offset_pair<1, 120, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 11, 0>::value, vnode_base_offset_pair<1, 120, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1, 120, 11, 1>::value, vnode_base_offset_pair<1, 120, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 11, 2>::value, vnode_base_offset_pair<1, 120, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 11, 3>::value, vnode_base_offset_pair<1, 120, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 12, 0>::value, vnode_base_offset_pair<1, 120, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1, 120, 12, 1>::value, vnode_base_offset_pair<1, 120, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 12, 2>::value, vnode_base_offset_pair<1, 120, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 12, 3>::value, vnode_base_offset_pair<1, 120, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 13, 0>::value, vnode_base_offset_pair<1, 120, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1, 120, 13, 1>::value, vnode_base_offset_pair<1, 120, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 13, 2>::value, vnode_base_offset_pair<1, 120, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 14, 0>::value, vnode_base_offset_pair<1, 120, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1, 120, 14, 1>::value, vnode_base_offset_pair<1, 120, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 14, 2>::value, vnode_base_offset_pair<1, 120, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 14, 3>::value, vnode_base_offset_pair<1, 120, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 15, 0>::value, vnode_base_offset_pair<1, 120, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1, 120, 15, 1>::value, vnode_base_offset_pair<1, 120, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 15, 2>::value, vnode_base_offset_pair<1, 120, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 15, 3>::value, vnode_base_offset_pair<1, 120, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 16, 0>::value, vnode_base_offset_pair<1, 120, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1, 120, 16, 1>::value, vnode_base_offset_pair<1, 120, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 16, 2>::value, vnode_base_offset_pair<1, 120, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 17, 0>::value, vnode_base_offset_pair<1, 120, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1, 120, 17, 1>::value, vnode_base_offset_pair<1, 120, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 17, 2>::value, vnode_base_offset_pair<1, 120, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 18, 0>::value, vnode_base_offset_pair<1, 120, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1, 120, 18, 1>::value, vnode_base_offset_pair<1, 120, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 18, 2>::value, vnode_base_offset_pair<1, 120, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 19, 0>::value, vnode_base_offset_pair<1, 120, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1, 120, 19, 1>::value, vnode_base_offset_pair<1, 120, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 19, 2>::value, vnode_base_offset_pair<1, 120, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 20, 0>::value, vnode_base_offset_pair<1, 120, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1, 120, 20, 1>::value, vnode_base_offset_pair<1, 120, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 20, 2>::value, vnode_base_offset_pair<1, 120, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 21, 0>::value, vnode_base_offset_pair<1, 120, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1, 120, 21, 1>::value, vnode_base_offset_pair<1, 120, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 21, 2>::value, vnode_base_offset_pair<1, 120, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 22, 0>::value, vnode_base_offset_pair<1, 120, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1, 120, 22, 1>::value, vnode_base_offset_pair<1, 120, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 22, 2>::value, vnode_base_offset_pair<1, 120, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 23, 0>::value, vnode_base_offset_pair<1, 120, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1, 120, 23, 1>::value, vnode_base_offset_pair<1, 120, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 23, 2>::value, vnode_base_offset_pair<1, 120, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 24, 0>::value, vnode_base_offset_pair<1, 120, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1, 120, 24, 1>::value, vnode_base_offset_pair<1, 120, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 24, 2>::value, vnode_base_offset_pair<1, 120, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 25, 0>::value, vnode_base_offset_pair<1, 120, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1, 120, 25, 1>::value, vnode_base_offset_pair<1, 120, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 25, 2>::value, vnode_base_offset_pair<1, 120, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 26, 0>::value, vnode_base_offset_pair<1, 120, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1, 120, 26, 1>::value, vnode_base_offset_pair<1, 120, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 26, 2>::value, vnode_base_offset_pair<1, 120, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 27, 0>::value, vnode_base_offset_pair<1, 120, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1, 120, 27, 1>::value, vnode_base_offset_pair<1, 120, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 28, 0>::value, vnode_base_offset_pair<1, 120, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1, 120, 28, 1>::value, vnode_base_offset_pair<1, 120, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 28, 2>::value, vnode_base_offset_pair<1, 120, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 29, 0>::value, vnode_base_offset_pair<1, 120, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1, 120, 29, 1>::value, vnode_base_offset_pair<1, 120, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 29, 2>::value, vnode_base_offset_pair<1, 120, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 30, 0>::value, vnode_base_offset_pair<1, 120, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1, 120, 30, 1>::value, vnode_base_offset_pair<1, 120, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 30, 2>::value, vnode_base_offset_pair<1, 120, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 31, 0>::value, vnode_base_offset_pair<1, 120, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1, 120, 31, 1>::value, vnode_base_offset_pair<1, 120, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 31, 2>::value, vnode_base_offset_pair<1, 120, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 32, 0>::value, vnode_base_offset_pair<1, 120, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1, 120, 32, 1>::value, vnode_base_offset_pair<1, 120, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 32, 2>::value, vnode_base_offset_pair<1, 120, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 33, 0>::value, vnode_base_offset_pair<1, 120, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1, 120, 33, 1>::value, vnode_base_offset_pair<1, 120, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 33, 2>::value, vnode_base_offset_pair<1, 120, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 34, 0>::value, vnode_base_offset_pair<1, 120, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1, 120, 34, 1>::value, vnode_base_offset_pair<1, 120, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 34, 2>::value, vnode_base_offset_pair<1, 120, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 35, 0>::value, vnode_base_offset_pair<1, 120, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1, 120, 35, 1>::value, vnode_base_offset_pair<1, 120, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 35, 2>::value, vnode_base_offset_pair<1, 120, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 36, 0>::value, vnode_base_offset_pair<1, 120, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1, 120, 36, 1>::value, vnode_base_offset_pair<1, 120, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 36, 2>::value, vnode_base_offset_pair<1, 120, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 37, 0>::value, vnode_base_offset_pair<1, 120, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1, 120, 37, 1>::value, vnode_base_offset_pair<1, 120, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 38, 0>::value, vnode_base_offset_pair<1, 120, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1, 120, 38, 1>::value, vnode_base_offset_pair<1, 120, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 38, 2>::value, vnode_base_offset_pair<1, 120, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 39, 0>::value, vnode_base_offset_pair<1, 120, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1, 120, 39, 1>::value, vnode_base_offset_pair<1, 120, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 39, 2>::value, vnode_base_offset_pair<1, 120, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 40, 0>::value, vnode_base_offset_pair<1, 120, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1, 120, 40, 1>::value, vnode_base_offset_pair<1, 120, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 41, 0>::value, vnode_base_offset_pair<1, 120, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1, 120, 41, 1>::value, vnode_base_offset_pair<1, 120, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 41, 2>::value, vnode_base_offset_pair<1, 120, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 42, 0>::value, vnode_base_offset_pair<1, 120, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1, 120, 42, 1>::value, vnode_base_offset_pair<1, 120, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 43, 0>::value, vnode_base_offset_pair<1, 120, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1, 120, 43, 1>::value, vnode_base_offset_pair<1, 120, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 43, 2>::value, vnode_base_offset_pair<1, 120, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 44, 0>::value, vnode_base_offset_pair<1, 120, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1, 120, 44, 1>::value, vnode_base_offset_pair<1, 120, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 120, 44, 2>::value, vnode_base_offset_pair<1, 120, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 120, 45, 0>::value, vnode_base_offset_pair<1, 120, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1, 120, 45, 1>::value, vnode_base_offset_pair<1, 120, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z128_8 =
{
    {
        { vnode_shift_mod_pair<1, 128,  0, 0>::value, vnode_base_offset_pair<1, 128,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1, 128,  0, 1>::value, vnode_base_offset_pair<1, 128,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  0, 2>::value, vnode_base_offset_pair<1, 128,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  0, 3>::value, vnode_base_offset_pair<1, 128,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  0, 4>::value, vnode_base_offset_pair<1, 128,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  0, 5>::value, vnode_base_offset_pair<1, 128,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  0, 6>::value, vnode_base_offset_pair<1, 128,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  0, 7>::value, vnode_base_offset_pair<1, 128,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  0, 8>::value, vnode_base_offset_pair<1, 128,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  0, 9>::value, vnode_base_offset_pair<1, 128,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 128,  1, 0>::value, vnode_base_offset_pair<1, 128,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1, 128,  1, 1>::value, vnode_base_offset_pair<1, 128,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  1, 2>::value, vnode_base_offset_pair<1, 128,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  1, 3>::value, vnode_base_offset_pair<1, 128,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  1, 4>::value, vnode_base_offset_pair<1, 128,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  1, 5>::value, vnode_base_offset_pair<1, 128,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  1, 6>::value, vnode_base_offset_pair<1, 128,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  1, 7>::value, vnode_base_offset_pair<1, 128,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  1, 8>::value, vnode_base_offset_pair<1, 128,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  1, 9>::value, vnode_base_offset_pair<1, 128,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 128,  2, 0>::value, vnode_base_offset_pair<1, 128,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1, 128,  2, 1>::value, vnode_base_offset_pair<1, 128,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  2, 2>::value, vnode_base_offset_pair<1, 128,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  2, 3>::value, vnode_base_offset_pair<1, 128,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  2, 4>::value, vnode_base_offset_pair<1, 128,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  2, 5>::value, vnode_base_offset_pair<1, 128,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  2, 6>::value, vnode_base_offset_pair<1, 128,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  2, 7>::value, vnode_base_offset_pair<1, 128,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  2, 8>::value, vnode_base_offset_pair<1, 128,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  2, 9>::value, vnode_base_offset_pair<1, 128,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 128,  3, 0>::value, vnode_base_offset_pair<1, 128,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1, 128,  3, 1>::value, vnode_base_offset_pair<1, 128,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  3, 2>::value, vnode_base_offset_pair<1, 128,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  3, 3>::value, vnode_base_offset_pair<1, 128,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  3, 4>::value, vnode_base_offset_pair<1, 128,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  3, 5>::value, vnode_base_offset_pair<1, 128,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  3, 6>::value, vnode_base_offset_pair<1, 128,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  3, 7>::value, vnode_base_offset_pair<1, 128,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  3, 8>::value, vnode_base_offset_pair<1, 128,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  3, 9>::value, vnode_base_offset_pair<1, 128,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 128,  4, 0>::value, vnode_base_offset_pair<1, 128,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1, 128,  4, 1>::value, vnode_base_offset_pair<1, 128,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 128,  5, 0>::value, vnode_base_offset_pair<1, 128,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1, 128,  5, 1>::value, vnode_base_offset_pair<1, 128,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  5, 2>::value, vnode_base_offset_pair<1, 128,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  5, 3>::value, vnode_base_offset_pair<1, 128,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 128,  6, 0>::value, vnode_base_offset_pair<1, 128,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1, 128,  6, 1>::value, vnode_base_offset_pair<1, 128,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  6, 2>::value, vnode_base_offset_pair<1, 128,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  6, 3>::value, vnode_base_offset_pair<1, 128,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  6, 4>::value, vnode_base_offset_pair<1, 128,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 128,  7, 0>::value, vnode_base_offset_pair<1, 128,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1, 128,  7, 1>::value, vnode_base_offset_pair<1, 128,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  7, 2>::value, vnode_base_offset_pair<1, 128,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  7, 3>::value, vnode_base_offset_pair<1, 128,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 128,  8, 0>::value, vnode_base_offset_pair<1, 128,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1, 128,  8, 1>::value, vnode_base_offset_pair<1, 128,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  8, 2>::value, vnode_base_offset_pair<1, 128,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  8, 3>::value, vnode_base_offset_pair<1, 128,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  8, 4>::value, vnode_base_offset_pair<1, 128,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 128,  9, 0>::value, vnode_base_offset_pair<1, 128,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1, 128,  9, 1>::value, vnode_base_offset_pair<1, 128,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  9, 2>::value, vnode_base_offset_pair<1, 128,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  9, 3>::value, vnode_base_offset_pair<1, 128,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 128,  9, 4>::value, vnode_base_offset_pair<1, 128,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 10, 0>::value, vnode_base_offset_pair<1, 128, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1, 128, 10, 1>::value, vnode_base_offset_pair<1, 128, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 10, 2>::value, vnode_base_offset_pair<1, 128, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 10, 3>::value, vnode_base_offset_pair<1, 128, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 11, 0>::value, vnode_base_offset_pair<1, 128, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1, 128, 11, 1>::value, vnode_base_offset_pair<1, 128, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 11, 2>::value, vnode_base_offset_pair<1, 128, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 11, 3>::value, vnode_base_offset_pair<1, 128, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 12, 0>::value, vnode_base_offset_pair<1, 128, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1, 128, 12, 1>::value, vnode_base_offset_pair<1, 128, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 12, 2>::value, vnode_base_offset_pair<1, 128, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 12, 3>::value, vnode_base_offset_pair<1, 128, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 13, 0>::value, vnode_base_offset_pair<1, 128, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1, 128, 13, 1>::value, vnode_base_offset_pair<1, 128, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 13, 2>::value, vnode_base_offset_pair<1, 128, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 14, 0>::value, vnode_base_offset_pair<1, 128, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1, 128, 14, 1>::value, vnode_base_offset_pair<1, 128, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 14, 2>::value, vnode_base_offset_pair<1, 128, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 14, 3>::value, vnode_base_offset_pair<1, 128, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 15, 0>::value, vnode_base_offset_pair<1, 128, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1, 128, 15, 1>::value, vnode_base_offset_pair<1, 128, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 15, 2>::value, vnode_base_offset_pair<1, 128, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 15, 3>::value, vnode_base_offset_pair<1, 128, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 16, 0>::value, vnode_base_offset_pair<1, 128, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1, 128, 16, 1>::value, vnode_base_offset_pair<1, 128, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 16, 2>::value, vnode_base_offset_pair<1, 128, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 17, 0>::value, vnode_base_offset_pair<1, 128, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1, 128, 17, 1>::value, vnode_base_offset_pair<1, 128, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 17, 2>::value, vnode_base_offset_pair<1, 128, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 18, 0>::value, vnode_base_offset_pair<1, 128, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1, 128, 18, 1>::value, vnode_base_offset_pair<1, 128, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 18, 2>::value, vnode_base_offset_pair<1, 128, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 19, 0>::value, vnode_base_offset_pair<1, 128, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1, 128, 19, 1>::value, vnode_base_offset_pair<1, 128, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 19, 2>::value, vnode_base_offset_pair<1, 128, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 20, 0>::value, vnode_base_offset_pair<1, 128, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1, 128, 20, 1>::value, vnode_base_offset_pair<1, 128, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 20, 2>::value, vnode_base_offset_pair<1, 128, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 21, 0>::value, vnode_base_offset_pair<1, 128, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1, 128, 21, 1>::value, vnode_base_offset_pair<1, 128, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 21, 2>::value, vnode_base_offset_pair<1, 128, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 22, 0>::value, vnode_base_offset_pair<1, 128, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1, 128, 22, 1>::value, vnode_base_offset_pair<1, 128, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 22, 2>::value, vnode_base_offset_pair<1, 128, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 23, 0>::value, vnode_base_offset_pair<1, 128, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1, 128, 23, 1>::value, vnode_base_offset_pair<1, 128, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 23, 2>::value, vnode_base_offset_pair<1, 128, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 24, 0>::value, vnode_base_offset_pair<1, 128, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1, 128, 24, 1>::value, vnode_base_offset_pair<1, 128, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 24, 2>::value, vnode_base_offset_pair<1, 128, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 25, 0>::value, vnode_base_offset_pair<1, 128, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1, 128, 25, 1>::value, vnode_base_offset_pair<1, 128, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 25, 2>::value, vnode_base_offset_pair<1, 128, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 26, 0>::value, vnode_base_offset_pair<1, 128, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1, 128, 26, 1>::value, vnode_base_offset_pair<1, 128, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 26, 2>::value, vnode_base_offset_pair<1, 128, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 27, 0>::value, vnode_base_offset_pair<1, 128, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1, 128, 27, 1>::value, vnode_base_offset_pair<1, 128, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 28, 0>::value, vnode_base_offset_pair<1, 128, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1, 128, 28, 1>::value, vnode_base_offset_pair<1, 128, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 28, 2>::value, vnode_base_offset_pair<1, 128, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 29, 0>::value, vnode_base_offset_pair<1, 128, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1, 128, 29, 1>::value, vnode_base_offset_pair<1, 128, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 29, 2>::value, vnode_base_offset_pair<1, 128, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 30, 0>::value, vnode_base_offset_pair<1, 128, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1, 128, 30, 1>::value, vnode_base_offset_pair<1, 128, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 30, 2>::value, vnode_base_offset_pair<1, 128, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 31, 0>::value, vnode_base_offset_pair<1, 128, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1, 128, 31, 1>::value, vnode_base_offset_pair<1, 128, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 31, 2>::value, vnode_base_offset_pair<1, 128, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 32, 0>::value, vnode_base_offset_pair<1, 128, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1, 128, 32, 1>::value, vnode_base_offset_pair<1, 128, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 32, 2>::value, vnode_base_offset_pair<1, 128, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 33, 0>::value, vnode_base_offset_pair<1, 128, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1, 128, 33, 1>::value, vnode_base_offset_pair<1, 128, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 33, 2>::value, vnode_base_offset_pair<1, 128, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 34, 0>::value, vnode_base_offset_pair<1, 128, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1, 128, 34, 1>::value, vnode_base_offset_pair<1, 128, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 34, 2>::value, vnode_base_offset_pair<1, 128, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 35, 0>::value, vnode_base_offset_pair<1, 128, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1, 128, 35, 1>::value, vnode_base_offset_pair<1, 128, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 35, 2>::value, vnode_base_offset_pair<1, 128, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 36, 0>::value, vnode_base_offset_pair<1, 128, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1, 128, 36, 1>::value, vnode_base_offset_pair<1, 128, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 36, 2>::value, vnode_base_offset_pair<1, 128, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 37, 0>::value, vnode_base_offset_pair<1, 128, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1, 128, 37, 1>::value, vnode_base_offset_pair<1, 128, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 38, 0>::value, vnode_base_offset_pair<1, 128, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1, 128, 38, 1>::value, vnode_base_offset_pair<1, 128, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 38, 2>::value, vnode_base_offset_pair<1, 128, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 39, 0>::value, vnode_base_offset_pair<1, 128, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1, 128, 39, 1>::value, vnode_base_offset_pair<1, 128, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 39, 2>::value, vnode_base_offset_pair<1, 128, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 40, 0>::value, vnode_base_offset_pair<1, 128, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1, 128, 40, 1>::value, vnode_base_offset_pair<1, 128, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 41, 0>::value, vnode_base_offset_pair<1, 128, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1, 128, 41, 1>::value, vnode_base_offset_pair<1, 128, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 41, 2>::value, vnode_base_offset_pair<1, 128, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 42, 0>::value, vnode_base_offset_pair<1, 128, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1, 128, 42, 1>::value, vnode_base_offset_pair<1, 128, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 43, 0>::value, vnode_base_offset_pair<1, 128, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1, 128, 43, 1>::value, vnode_base_offset_pair<1, 128, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 43, 2>::value, vnode_base_offset_pair<1, 128, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 44, 0>::value, vnode_base_offset_pair<1, 128, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1, 128, 44, 1>::value, vnode_base_offset_pair<1, 128, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 128, 44, 2>::value, vnode_base_offset_pair<1, 128, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 128, 45, 0>::value, vnode_base_offset_pair<1, 128, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1, 128, 45, 1>::value, vnode_base_offset_pair<1, 128, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z144_8 =
{
    {
        { vnode_shift_mod_pair<1, 144,  0, 0>::value, vnode_base_offset_pair<1, 144,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1, 144,  0, 1>::value, vnode_base_offset_pair<1, 144,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  0, 2>::value, vnode_base_offset_pair<1, 144,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  0, 3>::value, vnode_base_offset_pair<1, 144,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  0, 4>::value, vnode_base_offset_pair<1, 144,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  0, 5>::value, vnode_base_offset_pair<1, 144,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  0, 6>::value, vnode_base_offset_pair<1, 144,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  0, 7>::value, vnode_base_offset_pair<1, 144,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  0, 8>::value, vnode_base_offset_pair<1, 144,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  0, 9>::value, vnode_base_offset_pair<1, 144,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 144,  1, 0>::value, vnode_base_offset_pair<1, 144,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1, 144,  1, 1>::value, vnode_base_offset_pair<1, 144,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  1, 2>::value, vnode_base_offset_pair<1, 144,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  1, 3>::value, vnode_base_offset_pair<1, 144,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  1, 4>::value, vnode_base_offset_pair<1, 144,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  1, 5>::value, vnode_base_offset_pair<1, 144,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  1, 6>::value, vnode_base_offset_pair<1, 144,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  1, 7>::value, vnode_base_offset_pair<1, 144,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  1, 8>::value, vnode_base_offset_pair<1, 144,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  1, 9>::value, vnode_base_offset_pair<1, 144,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 144,  2, 0>::value, vnode_base_offset_pair<1, 144,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1, 144,  2, 1>::value, vnode_base_offset_pair<1, 144,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  2, 2>::value, vnode_base_offset_pair<1, 144,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  2, 3>::value, vnode_base_offset_pair<1, 144,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  2, 4>::value, vnode_base_offset_pair<1, 144,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  2, 5>::value, vnode_base_offset_pair<1, 144,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  2, 6>::value, vnode_base_offset_pair<1, 144,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  2, 7>::value, vnode_base_offset_pair<1, 144,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  2, 8>::value, vnode_base_offset_pair<1, 144,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  2, 9>::value, vnode_base_offset_pair<1, 144,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 144,  3, 0>::value, vnode_base_offset_pair<1, 144,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1, 144,  3, 1>::value, vnode_base_offset_pair<1, 144,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  3, 2>::value, vnode_base_offset_pair<1, 144,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  3, 3>::value, vnode_base_offset_pair<1, 144,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  3, 4>::value, vnode_base_offset_pair<1, 144,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  3, 5>::value, vnode_base_offset_pair<1, 144,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  3, 6>::value, vnode_base_offset_pair<1, 144,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  3, 7>::value, vnode_base_offset_pair<1, 144,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  3, 8>::value, vnode_base_offset_pair<1, 144,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  3, 9>::value, vnode_base_offset_pair<1, 144,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 144,  4, 0>::value, vnode_base_offset_pair<1, 144,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1, 144,  4, 1>::value, vnode_base_offset_pair<1, 144,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 144,  5, 0>::value, vnode_base_offset_pair<1, 144,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1, 144,  5, 1>::value, vnode_base_offset_pair<1, 144,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  5, 2>::value, vnode_base_offset_pair<1, 144,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  5, 3>::value, vnode_base_offset_pair<1, 144,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 144,  6, 0>::value, vnode_base_offset_pair<1, 144,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1, 144,  6, 1>::value, vnode_base_offset_pair<1, 144,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  6, 2>::value, vnode_base_offset_pair<1, 144,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  6, 3>::value, vnode_base_offset_pair<1, 144,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  6, 4>::value, vnode_base_offset_pair<1, 144,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 144,  7, 0>::value, vnode_base_offset_pair<1, 144,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1, 144,  7, 1>::value, vnode_base_offset_pair<1, 144,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  7, 2>::value, vnode_base_offset_pair<1, 144,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  7, 3>::value, vnode_base_offset_pair<1, 144,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 144,  8, 0>::value, vnode_base_offset_pair<1, 144,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1, 144,  8, 1>::value, vnode_base_offset_pair<1, 144,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  8, 2>::value, vnode_base_offset_pair<1, 144,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  8, 3>::value, vnode_base_offset_pair<1, 144,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  8, 4>::value, vnode_base_offset_pair<1, 144,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 144,  9, 0>::value, vnode_base_offset_pair<1, 144,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1, 144,  9, 1>::value, vnode_base_offset_pair<1, 144,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  9, 2>::value, vnode_base_offset_pair<1, 144,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  9, 3>::value, vnode_base_offset_pair<1, 144,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 144,  9, 4>::value, vnode_base_offset_pair<1, 144,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 10, 0>::value, vnode_base_offset_pair<1, 144, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1, 144, 10, 1>::value, vnode_base_offset_pair<1, 144, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 10, 2>::value, vnode_base_offset_pair<1, 144, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 10, 3>::value, vnode_base_offset_pair<1, 144, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 11, 0>::value, vnode_base_offset_pair<1, 144, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1, 144, 11, 1>::value, vnode_base_offset_pair<1, 144, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 11, 2>::value, vnode_base_offset_pair<1, 144, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 11, 3>::value, vnode_base_offset_pair<1, 144, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 12, 0>::value, vnode_base_offset_pair<1, 144, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1, 144, 12, 1>::value, vnode_base_offset_pair<1, 144, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 12, 2>::value, vnode_base_offset_pair<1, 144, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 12, 3>::value, vnode_base_offset_pair<1, 144, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 13, 0>::value, vnode_base_offset_pair<1, 144, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1, 144, 13, 1>::value, vnode_base_offset_pair<1, 144, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 13, 2>::value, vnode_base_offset_pair<1, 144, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 14, 0>::value, vnode_base_offset_pair<1, 144, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1, 144, 14, 1>::value, vnode_base_offset_pair<1, 144, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 14, 2>::value, vnode_base_offset_pair<1, 144, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 14, 3>::value, vnode_base_offset_pair<1, 144, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 15, 0>::value, vnode_base_offset_pair<1, 144, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1, 144, 15, 1>::value, vnode_base_offset_pair<1, 144, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 15, 2>::value, vnode_base_offset_pair<1, 144, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 15, 3>::value, vnode_base_offset_pair<1, 144, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 16, 0>::value, vnode_base_offset_pair<1, 144, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1, 144, 16, 1>::value, vnode_base_offset_pair<1, 144, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 16, 2>::value, vnode_base_offset_pair<1, 144, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 17, 0>::value, vnode_base_offset_pair<1, 144, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1, 144, 17, 1>::value, vnode_base_offset_pair<1, 144, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 17, 2>::value, vnode_base_offset_pair<1, 144, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 18, 0>::value, vnode_base_offset_pair<1, 144, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1, 144, 18, 1>::value, vnode_base_offset_pair<1, 144, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 18, 2>::value, vnode_base_offset_pair<1, 144, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 19, 0>::value, vnode_base_offset_pair<1, 144, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1, 144, 19, 1>::value, vnode_base_offset_pair<1, 144, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 19, 2>::value, vnode_base_offset_pair<1, 144, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 20, 0>::value, vnode_base_offset_pair<1, 144, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1, 144, 20, 1>::value, vnode_base_offset_pair<1, 144, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 20, 2>::value, vnode_base_offset_pair<1, 144, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 21, 0>::value, vnode_base_offset_pair<1, 144, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1, 144, 21, 1>::value, vnode_base_offset_pair<1, 144, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 21, 2>::value, vnode_base_offset_pair<1, 144, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 22, 0>::value, vnode_base_offset_pair<1, 144, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1, 144, 22, 1>::value, vnode_base_offset_pair<1, 144, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 22, 2>::value, vnode_base_offset_pair<1, 144, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 23, 0>::value, vnode_base_offset_pair<1, 144, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1, 144, 23, 1>::value, vnode_base_offset_pair<1, 144, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 23, 2>::value, vnode_base_offset_pair<1, 144, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 24, 0>::value, vnode_base_offset_pair<1, 144, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1, 144, 24, 1>::value, vnode_base_offset_pair<1, 144, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 24, 2>::value, vnode_base_offset_pair<1, 144, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 25, 0>::value, vnode_base_offset_pair<1, 144, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1, 144, 25, 1>::value, vnode_base_offset_pair<1, 144, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 25, 2>::value, vnode_base_offset_pair<1, 144, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 26, 0>::value, vnode_base_offset_pair<1, 144, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1, 144, 26, 1>::value, vnode_base_offset_pair<1, 144, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 26, 2>::value, vnode_base_offset_pair<1, 144, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 27, 0>::value, vnode_base_offset_pair<1, 144, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1, 144, 27, 1>::value, vnode_base_offset_pair<1, 144, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 28, 0>::value, vnode_base_offset_pair<1, 144, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1, 144, 28, 1>::value, vnode_base_offset_pair<1, 144, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 28, 2>::value, vnode_base_offset_pair<1, 144, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 29, 0>::value, vnode_base_offset_pair<1, 144, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1, 144, 29, 1>::value, vnode_base_offset_pair<1, 144, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 29, 2>::value, vnode_base_offset_pair<1, 144, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 30, 0>::value, vnode_base_offset_pair<1, 144, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1, 144, 30, 1>::value, vnode_base_offset_pair<1, 144, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 30, 2>::value, vnode_base_offset_pair<1, 144, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 31, 0>::value, vnode_base_offset_pair<1, 144, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1, 144, 31, 1>::value, vnode_base_offset_pair<1, 144, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 31, 2>::value, vnode_base_offset_pair<1, 144, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 32, 0>::value, vnode_base_offset_pair<1, 144, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1, 144, 32, 1>::value, vnode_base_offset_pair<1, 144, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 32, 2>::value, vnode_base_offset_pair<1, 144, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 33, 0>::value, vnode_base_offset_pair<1, 144, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1, 144, 33, 1>::value, vnode_base_offset_pair<1, 144, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 33, 2>::value, vnode_base_offset_pair<1, 144, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 34, 0>::value, vnode_base_offset_pair<1, 144, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1, 144, 34, 1>::value, vnode_base_offset_pair<1, 144, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 34, 2>::value, vnode_base_offset_pair<1, 144, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 35, 0>::value, vnode_base_offset_pair<1, 144, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1, 144, 35, 1>::value, vnode_base_offset_pair<1, 144, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 35, 2>::value, vnode_base_offset_pair<1, 144, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 36, 0>::value, vnode_base_offset_pair<1, 144, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1, 144, 36, 1>::value, vnode_base_offset_pair<1, 144, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 36, 2>::value, vnode_base_offset_pair<1, 144, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 37, 0>::value, vnode_base_offset_pair<1, 144, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1, 144, 37, 1>::value, vnode_base_offset_pair<1, 144, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 38, 0>::value, vnode_base_offset_pair<1, 144, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1, 144, 38, 1>::value, vnode_base_offset_pair<1, 144, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 38, 2>::value, vnode_base_offset_pair<1, 144, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 39, 0>::value, vnode_base_offset_pair<1, 144, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1, 144, 39, 1>::value, vnode_base_offset_pair<1, 144, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 39, 2>::value, vnode_base_offset_pair<1, 144, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 40, 0>::value, vnode_base_offset_pair<1, 144, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1, 144, 40, 1>::value, vnode_base_offset_pair<1, 144, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 41, 0>::value, vnode_base_offset_pair<1, 144, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1, 144, 41, 1>::value, vnode_base_offset_pair<1, 144, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 41, 2>::value, vnode_base_offset_pair<1, 144, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 42, 0>::value, vnode_base_offset_pair<1, 144, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1, 144, 42, 1>::value, vnode_base_offset_pair<1, 144, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 43, 0>::value, vnode_base_offset_pair<1, 144, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1, 144, 43, 1>::value, vnode_base_offset_pair<1, 144, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 43, 2>::value, vnode_base_offset_pair<1, 144, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 44, 0>::value, vnode_base_offset_pair<1, 144, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1, 144, 44, 1>::value, vnode_base_offset_pair<1, 144, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 144, 44, 2>::value, vnode_base_offset_pair<1, 144, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 144, 45, 0>::value, vnode_base_offset_pair<1, 144, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1, 144, 45, 1>::value, vnode_base_offset_pair<1, 144, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z160_8 =
{
    {
        { vnode_shift_mod_pair<1, 160,  0, 0>::value, vnode_base_offset_pair<1, 160,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1, 160,  0, 1>::value, vnode_base_offset_pair<1, 160,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  0, 2>::value, vnode_base_offset_pair<1, 160,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  0, 3>::value, vnode_base_offset_pair<1, 160,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  0, 4>::value, vnode_base_offset_pair<1, 160,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  0, 5>::value, vnode_base_offset_pair<1, 160,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  0, 6>::value, vnode_base_offset_pair<1, 160,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  0, 7>::value, vnode_base_offset_pair<1, 160,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  0, 8>::value, vnode_base_offset_pair<1, 160,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  0, 9>::value, vnode_base_offset_pair<1, 160,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 160,  1, 0>::value, vnode_base_offset_pair<1, 160,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1, 160,  1, 1>::value, vnode_base_offset_pair<1, 160,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  1, 2>::value, vnode_base_offset_pair<1, 160,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  1, 3>::value, vnode_base_offset_pair<1, 160,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  1, 4>::value, vnode_base_offset_pair<1, 160,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  1, 5>::value, vnode_base_offset_pair<1, 160,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  1, 6>::value, vnode_base_offset_pair<1, 160,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  1, 7>::value, vnode_base_offset_pair<1, 160,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  1, 8>::value, vnode_base_offset_pair<1, 160,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  1, 9>::value, vnode_base_offset_pair<1, 160,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 160,  2, 0>::value, vnode_base_offset_pair<1, 160,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1, 160,  2, 1>::value, vnode_base_offset_pair<1, 160,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  2, 2>::value, vnode_base_offset_pair<1, 160,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  2, 3>::value, vnode_base_offset_pair<1, 160,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  2, 4>::value, vnode_base_offset_pair<1, 160,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  2, 5>::value, vnode_base_offset_pair<1, 160,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  2, 6>::value, vnode_base_offset_pair<1, 160,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  2, 7>::value, vnode_base_offset_pair<1, 160,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  2, 8>::value, vnode_base_offset_pair<1, 160,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  2, 9>::value, vnode_base_offset_pair<1, 160,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 160,  3, 0>::value, vnode_base_offset_pair<1, 160,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1, 160,  3, 1>::value, vnode_base_offset_pair<1, 160,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  3, 2>::value, vnode_base_offset_pair<1, 160,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  3, 3>::value, vnode_base_offset_pair<1, 160,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  3, 4>::value, vnode_base_offset_pair<1, 160,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  3, 5>::value, vnode_base_offset_pair<1, 160,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  3, 6>::value, vnode_base_offset_pair<1, 160,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  3, 7>::value, vnode_base_offset_pair<1, 160,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  3, 8>::value, vnode_base_offset_pair<1, 160,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  3, 9>::value, vnode_base_offset_pair<1, 160,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 160,  4, 0>::value, vnode_base_offset_pair<1, 160,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1, 160,  4, 1>::value, vnode_base_offset_pair<1, 160,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 160,  5, 0>::value, vnode_base_offset_pair<1, 160,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1, 160,  5, 1>::value, vnode_base_offset_pair<1, 160,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  5, 2>::value, vnode_base_offset_pair<1, 160,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  5, 3>::value, vnode_base_offset_pair<1, 160,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 160,  6, 0>::value, vnode_base_offset_pair<1, 160,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1, 160,  6, 1>::value, vnode_base_offset_pair<1, 160,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  6, 2>::value, vnode_base_offset_pair<1, 160,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  6, 3>::value, vnode_base_offset_pair<1, 160,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  6, 4>::value, vnode_base_offset_pair<1, 160,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 160,  7, 0>::value, vnode_base_offset_pair<1, 160,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1, 160,  7, 1>::value, vnode_base_offset_pair<1, 160,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  7, 2>::value, vnode_base_offset_pair<1, 160,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  7, 3>::value, vnode_base_offset_pair<1, 160,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 160,  8, 0>::value, vnode_base_offset_pair<1, 160,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1, 160,  8, 1>::value, vnode_base_offset_pair<1, 160,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  8, 2>::value, vnode_base_offset_pair<1, 160,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  8, 3>::value, vnode_base_offset_pair<1, 160,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  8, 4>::value, vnode_base_offset_pair<1, 160,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 160,  9, 0>::value, vnode_base_offset_pair<1, 160,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1, 160,  9, 1>::value, vnode_base_offset_pair<1, 160,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  9, 2>::value, vnode_base_offset_pair<1, 160,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  9, 3>::value, vnode_base_offset_pair<1, 160,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 160,  9, 4>::value, vnode_base_offset_pair<1, 160,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 10, 0>::value, vnode_base_offset_pair<1, 160, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1, 160, 10, 1>::value, vnode_base_offset_pair<1, 160, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 10, 2>::value, vnode_base_offset_pair<1, 160, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 10, 3>::value, vnode_base_offset_pair<1, 160, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 11, 0>::value, vnode_base_offset_pair<1, 160, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1, 160, 11, 1>::value, vnode_base_offset_pair<1, 160, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 11, 2>::value, vnode_base_offset_pair<1, 160, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 11, 3>::value, vnode_base_offset_pair<1, 160, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 12, 0>::value, vnode_base_offset_pair<1, 160, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1, 160, 12, 1>::value, vnode_base_offset_pair<1, 160, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 12, 2>::value, vnode_base_offset_pair<1, 160, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 12, 3>::value, vnode_base_offset_pair<1, 160, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 13, 0>::value, vnode_base_offset_pair<1, 160, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1, 160, 13, 1>::value, vnode_base_offset_pair<1, 160, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 13, 2>::value, vnode_base_offset_pair<1, 160, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 14, 0>::value, vnode_base_offset_pair<1, 160, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1, 160, 14, 1>::value, vnode_base_offset_pair<1, 160, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 14, 2>::value, vnode_base_offset_pair<1, 160, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 14, 3>::value, vnode_base_offset_pair<1, 160, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 15, 0>::value, vnode_base_offset_pair<1, 160, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1, 160, 15, 1>::value, vnode_base_offset_pair<1, 160, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 15, 2>::value, vnode_base_offset_pair<1, 160, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 15, 3>::value, vnode_base_offset_pair<1, 160, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 16, 0>::value, vnode_base_offset_pair<1, 160, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1, 160, 16, 1>::value, vnode_base_offset_pair<1, 160, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 16, 2>::value, vnode_base_offset_pair<1, 160, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 17, 0>::value, vnode_base_offset_pair<1, 160, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1, 160, 17, 1>::value, vnode_base_offset_pair<1, 160, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 17, 2>::value, vnode_base_offset_pair<1, 160, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 18, 0>::value, vnode_base_offset_pair<1, 160, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1, 160, 18, 1>::value, vnode_base_offset_pair<1, 160, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 18, 2>::value, vnode_base_offset_pair<1, 160, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 19, 0>::value, vnode_base_offset_pair<1, 160, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1, 160, 19, 1>::value, vnode_base_offset_pair<1, 160, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 19, 2>::value, vnode_base_offset_pair<1, 160, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 20, 0>::value, vnode_base_offset_pair<1, 160, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1, 160, 20, 1>::value, vnode_base_offset_pair<1, 160, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 20, 2>::value, vnode_base_offset_pair<1, 160, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 21, 0>::value, vnode_base_offset_pair<1, 160, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1, 160, 21, 1>::value, vnode_base_offset_pair<1, 160, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 21, 2>::value, vnode_base_offset_pair<1, 160, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 22, 0>::value, vnode_base_offset_pair<1, 160, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1, 160, 22, 1>::value, vnode_base_offset_pair<1, 160, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 22, 2>::value, vnode_base_offset_pair<1, 160, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 23, 0>::value, vnode_base_offset_pair<1, 160, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1, 160, 23, 1>::value, vnode_base_offset_pair<1, 160, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 23, 2>::value, vnode_base_offset_pair<1, 160, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 24, 0>::value, vnode_base_offset_pair<1, 160, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1, 160, 24, 1>::value, vnode_base_offset_pair<1, 160, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 24, 2>::value, vnode_base_offset_pair<1, 160, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 25, 0>::value, vnode_base_offset_pair<1, 160, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1, 160, 25, 1>::value, vnode_base_offset_pair<1, 160, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 25, 2>::value, vnode_base_offset_pair<1, 160, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 26, 0>::value, vnode_base_offset_pair<1, 160, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1, 160, 26, 1>::value, vnode_base_offset_pair<1, 160, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 26, 2>::value, vnode_base_offset_pair<1, 160, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 27, 0>::value, vnode_base_offset_pair<1, 160, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1, 160, 27, 1>::value, vnode_base_offset_pair<1, 160, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 28, 0>::value, vnode_base_offset_pair<1, 160, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1, 160, 28, 1>::value, vnode_base_offset_pair<1, 160, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 28, 2>::value, vnode_base_offset_pair<1, 160, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 29, 0>::value, vnode_base_offset_pair<1, 160, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1, 160, 29, 1>::value, vnode_base_offset_pair<1, 160, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 29, 2>::value, vnode_base_offset_pair<1, 160, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 30, 0>::value, vnode_base_offset_pair<1, 160, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1, 160, 30, 1>::value, vnode_base_offset_pair<1, 160, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 30, 2>::value, vnode_base_offset_pair<1, 160, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 31, 0>::value, vnode_base_offset_pair<1, 160, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1, 160, 31, 1>::value, vnode_base_offset_pair<1, 160, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 31, 2>::value, vnode_base_offset_pair<1, 160, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 32, 0>::value, vnode_base_offset_pair<1, 160, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1, 160, 32, 1>::value, vnode_base_offset_pair<1, 160, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 32, 2>::value, vnode_base_offset_pair<1, 160, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 33, 0>::value, vnode_base_offset_pair<1, 160, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1, 160, 33, 1>::value, vnode_base_offset_pair<1, 160, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 33, 2>::value, vnode_base_offset_pair<1, 160, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 34, 0>::value, vnode_base_offset_pair<1, 160, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1, 160, 34, 1>::value, vnode_base_offset_pair<1, 160, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 34, 2>::value, vnode_base_offset_pair<1, 160, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 35, 0>::value, vnode_base_offset_pair<1, 160, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1, 160, 35, 1>::value, vnode_base_offset_pair<1, 160, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 35, 2>::value, vnode_base_offset_pair<1, 160, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 36, 0>::value, vnode_base_offset_pair<1, 160, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1, 160, 36, 1>::value, vnode_base_offset_pair<1, 160, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 36, 2>::value, vnode_base_offset_pair<1, 160, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 37, 0>::value, vnode_base_offset_pair<1, 160, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1, 160, 37, 1>::value, vnode_base_offset_pair<1, 160, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 38, 0>::value, vnode_base_offset_pair<1, 160, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1, 160, 38, 1>::value, vnode_base_offset_pair<1, 160, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 38, 2>::value, vnode_base_offset_pair<1, 160, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 39, 0>::value, vnode_base_offset_pair<1, 160, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1, 160, 39, 1>::value, vnode_base_offset_pair<1, 160, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 39, 2>::value, vnode_base_offset_pair<1, 160, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 40, 0>::value, vnode_base_offset_pair<1, 160, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1, 160, 40, 1>::value, vnode_base_offset_pair<1, 160, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 41, 0>::value, vnode_base_offset_pair<1, 160, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1, 160, 41, 1>::value, vnode_base_offset_pair<1, 160, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 41, 2>::value, vnode_base_offset_pair<1, 160, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 42, 0>::value, vnode_base_offset_pair<1, 160, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1, 160, 42, 1>::value, vnode_base_offset_pair<1, 160, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 43, 0>::value, vnode_base_offset_pair<1, 160, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1, 160, 43, 1>::value, vnode_base_offset_pair<1, 160, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 43, 2>::value, vnode_base_offset_pair<1, 160, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 44, 0>::value, vnode_base_offset_pair<1, 160, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1, 160, 44, 1>::value, vnode_base_offset_pair<1, 160, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 160, 44, 2>::value, vnode_base_offset_pair<1, 160, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 160, 45, 0>::value, vnode_base_offset_pair<1, 160, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1, 160, 45, 1>::value, vnode_base_offset_pair<1, 160, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z176_8 =
{
    {
        { vnode_shift_mod_pair<1, 176,  0, 0>::value, vnode_base_offset_pair<1, 176,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1, 176,  0, 1>::value, vnode_base_offset_pair<1, 176,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  0, 2>::value, vnode_base_offset_pair<1, 176,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  0, 3>::value, vnode_base_offset_pair<1, 176,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  0, 4>::value, vnode_base_offset_pair<1, 176,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  0, 5>::value, vnode_base_offset_pair<1, 176,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  0, 6>::value, vnode_base_offset_pair<1, 176,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  0, 7>::value, vnode_base_offset_pair<1, 176,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  0, 8>::value, vnode_base_offset_pair<1, 176,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  0, 9>::value, vnode_base_offset_pair<1, 176,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 176,  1, 0>::value, vnode_base_offset_pair<1, 176,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1, 176,  1, 1>::value, vnode_base_offset_pair<1, 176,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  1, 2>::value, vnode_base_offset_pair<1, 176,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  1, 3>::value, vnode_base_offset_pair<1, 176,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  1, 4>::value, vnode_base_offset_pair<1, 176,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  1, 5>::value, vnode_base_offset_pair<1, 176,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  1, 6>::value, vnode_base_offset_pair<1, 176,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  1, 7>::value, vnode_base_offset_pair<1, 176,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  1, 8>::value, vnode_base_offset_pair<1, 176,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  1, 9>::value, vnode_base_offset_pair<1, 176,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 176,  2, 0>::value, vnode_base_offset_pair<1, 176,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1, 176,  2, 1>::value, vnode_base_offset_pair<1, 176,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  2, 2>::value, vnode_base_offset_pair<1, 176,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  2, 3>::value, vnode_base_offset_pair<1, 176,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  2, 4>::value, vnode_base_offset_pair<1, 176,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  2, 5>::value, vnode_base_offset_pair<1, 176,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  2, 6>::value, vnode_base_offset_pair<1, 176,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  2, 7>::value, vnode_base_offset_pair<1, 176,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  2, 8>::value, vnode_base_offset_pair<1, 176,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  2, 9>::value, vnode_base_offset_pair<1, 176,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 176,  3, 0>::value, vnode_base_offset_pair<1, 176,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1, 176,  3, 1>::value, vnode_base_offset_pair<1, 176,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  3, 2>::value, vnode_base_offset_pair<1, 176,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  3, 3>::value, vnode_base_offset_pair<1, 176,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  3, 4>::value, vnode_base_offset_pair<1, 176,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  3, 5>::value, vnode_base_offset_pair<1, 176,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  3, 6>::value, vnode_base_offset_pair<1, 176,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  3, 7>::value, vnode_base_offset_pair<1, 176,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  3, 8>::value, vnode_base_offset_pair<1, 176,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  3, 9>::value, vnode_base_offset_pair<1, 176,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 176,  4, 0>::value, vnode_base_offset_pair<1, 176,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1, 176,  4, 1>::value, vnode_base_offset_pair<1, 176,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 176,  5, 0>::value, vnode_base_offset_pair<1, 176,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1, 176,  5, 1>::value, vnode_base_offset_pair<1, 176,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  5, 2>::value, vnode_base_offset_pair<1, 176,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  5, 3>::value, vnode_base_offset_pair<1, 176,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 176,  6, 0>::value, vnode_base_offset_pair<1, 176,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1, 176,  6, 1>::value, vnode_base_offset_pair<1, 176,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  6, 2>::value, vnode_base_offset_pair<1, 176,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  6, 3>::value, vnode_base_offset_pair<1, 176,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  6, 4>::value, vnode_base_offset_pair<1, 176,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 176,  7, 0>::value, vnode_base_offset_pair<1, 176,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1, 176,  7, 1>::value, vnode_base_offset_pair<1, 176,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  7, 2>::value, vnode_base_offset_pair<1, 176,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  7, 3>::value, vnode_base_offset_pair<1, 176,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 176,  8, 0>::value, vnode_base_offset_pair<1, 176,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1, 176,  8, 1>::value, vnode_base_offset_pair<1, 176,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  8, 2>::value, vnode_base_offset_pair<1, 176,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  8, 3>::value, vnode_base_offset_pair<1, 176,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  8, 4>::value, vnode_base_offset_pair<1, 176,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 176,  9, 0>::value, vnode_base_offset_pair<1, 176,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1, 176,  9, 1>::value, vnode_base_offset_pair<1, 176,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  9, 2>::value, vnode_base_offset_pair<1, 176,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  9, 3>::value, vnode_base_offset_pair<1, 176,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 176,  9, 4>::value, vnode_base_offset_pair<1, 176,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 10, 0>::value, vnode_base_offset_pair<1, 176, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1, 176, 10, 1>::value, vnode_base_offset_pair<1, 176, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 10, 2>::value, vnode_base_offset_pair<1, 176, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 10, 3>::value, vnode_base_offset_pair<1, 176, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 11, 0>::value, vnode_base_offset_pair<1, 176, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1, 176, 11, 1>::value, vnode_base_offset_pair<1, 176, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 11, 2>::value, vnode_base_offset_pair<1, 176, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 11, 3>::value, vnode_base_offset_pair<1, 176, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 12, 0>::value, vnode_base_offset_pair<1, 176, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1, 176, 12, 1>::value, vnode_base_offset_pair<1, 176, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 12, 2>::value, vnode_base_offset_pair<1, 176, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 12, 3>::value, vnode_base_offset_pair<1, 176, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 13, 0>::value, vnode_base_offset_pair<1, 176, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1, 176, 13, 1>::value, vnode_base_offset_pair<1, 176, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 13, 2>::value, vnode_base_offset_pair<1, 176, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 14, 0>::value, vnode_base_offset_pair<1, 176, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1, 176, 14, 1>::value, vnode_base_offset_pair<1, 176, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 14, 2>::value, vnode_base_offset_pair<1, 176, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 14, 3>::value, vnode_base_offset_pair<1, 176, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 15, 0>::value, vnode_base_offset_pair<1, 176, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1, 176, 15, 1>::value, vnode_base_offset_pair<1, 176, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 15, 2>::value, vnode_base_offset_pair<1, 176, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 15, 3>::value, vnode_base_offset_pair<1, 176, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 16, 0>::value, vnode_base_offset_pair<1, 176, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1, 176, 16, 1>::value, vnode_base_offset_pair<1, 176, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 16, 2>::value, vnode_base_offset_pair<1, 176, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 17, 0>::value, vnode_base_offset_pair<1, 176, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1, 176, 17, 1>::value, vnode_base_offset_pair<1, 176, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 17, 2>::value, vnode_base_offset_pair<1, 176, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 18, 0>::value, vnode_base_offset_pair<1, 176, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1, 176, 18, 1>::value, vnode_base_offset_pair<1, 176, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 18, 2>::value, vnode_base_offset_pair<1, 176, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 19, 0>::value, vnode_base_offset_pair<1, 176, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1, 176, 19, 1>::value, vnode_base_offset_pair<1, 176, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 19, 2>::value, vnode_base_offset_pair<1, 176, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 20, 0>::value, vnode_base_offset_pair<1, 176, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1, 176, 20, 1>::value, vnode_base_offset_pair<1, 176, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 20, 2>::value, vnode_base_offset_pair<1, 176, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 21, 0>::value, vnode_base_offset_pair<1, 176, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1, 176, 21, 1>::value, vnode_base_offset_pair<1, 176, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 21, 2>::value, vnode_base_offset_pair<1, 176, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 22, 0>::value, vnode_base_offset_pair<1, 176, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1, 176, 22, 1>::value, vnode_base_offset_pair<1, 176, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 22, 2>::value, vnode_base_offset_pair<1, 176, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 23, 0>::value, vnode_base_offset_pair<1, 176, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1, 176, 23, 1>::value, vnode_base_offset_pair<1, 176, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 23, 2>::value, vnode_base_offset_pair<1, 176, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 24, 0>::value, vnode_base_offset_pair<1, 176, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1, 176, 24, 1>::value, vnode_base_offset_pair<1, 176, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 24, 2>::value, vnode_base_offset_pair<1, 176, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 25, 0>::value, vnode_base_offset_pair<1, 176, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1, 176, 25, 1>::value, vnode_base_offset_pair<1, 176, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 25, 2>::value, vnode_base_offset_pair<1, 176, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 26, 0>::value, vnode_base_offset_pair<1, 176, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1, 176, 26, 1>::value, vnode_base_offset_pair<1, 176, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 26, 2>::value, vnode_base_offset_pair<1, 176, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 27, 0>::value, vnode_base_offset_pair<1, 176, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1, 176, 27, 1>::value, vnode_base_offset_pair<1, 176, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 28, 0>::value, vnode_base_offset_pair<1, 176, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1, 176, 28, 1>::value, vnode_base_offset_pair<1, 176, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 28, 2>::value, vnode_base_offset_pair<1, 176, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 29, 0>::value, vnode_base_offset_pair<1, 176, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1, 176, 29, 1>::value, vnode_base_offset_pair<1, 176, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 29, 2>::value, vnode_base_offset_pair<1, 176, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 30, 0>::value, vnode_base_offset_pair<1, 176, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1, 176, 30, 1>::value, vnode_base_offset_pair<1, 176, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 30, 2>::value, vnode_base_offset_pair<1, 176, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 31, 0>::value, vnode_base_offset_pair<1, 176, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1, 176, 31, 1>::value, vnode_base_offset_pair<1, 176, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 31, 2>::value, vnode_base_offset_pair<1, 176, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 32, 0>::value, vnode_base_offset_pair<1, 176, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1, 176, 32, 1>::value, vnode_base_offset_pair<1, 176, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 32, 2>::value, vnode_base_offset_pair<1, 176, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 33, 0>::value, vnode_base_offset_pair<1, 176, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1, 176, 33, 1>::value, vnode_base_offset_pair<1, 176, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 33, 2>::value, vnode_base_offset_pair<1, 176, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 34, 0>::value, vnode_base_offset_pair<1, 176, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1, 176, 34, 1>::value, vnode_base_offset_pair<1, 176, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 34, 2>::value, vnode_base_offset_pair<1, 176, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 35, 0>::value, vnode_base_offset_pair<1, 176, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1, 176, 35, 1>::value, vnode_base_offset_pair<1, 176, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 35, 2>::value, vnode_base_offset_pair<1, 176, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 36, 0>::value, vnode_base_offset_pair<1, 176, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1, 176, 36, 1>::value, vnode_base_offset_pair<1, 176, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 36, 2>::value, vnode_base_offset_pair<1, 176, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 37, 0>::value, vnode_base_offset_pair<1, 176, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1, 176, 37, 1>::value, vnode_base_offset_pair<1, 176, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 38, 0>::value, vnode_base_offset_pair<1, 176, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1, 176, 38, 1>::value, vnode_base_offset_pair<1, 176, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 38, 2>::value, vnode_base_offset_pair<1, 176, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 39, 0>::value, vnode_base_offset_pair<1, 176, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1, 176, 39, 1>::value, vnode_base_offset_pair<1, 176, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 39, 2>::value, vnode_base_offset_pair<1, 176, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 40, 0>::value, vnode_base_offset_pair<1, 176, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1, 176, 40, 1>::value, vnode_base_offset_pair<1, 176, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 41, 0>::value, vnode_base_offset_pair<1, 176, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1, 176, 41, 1>::value, vnode_base_offset_pair<1, 176, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 41, 2>::value, vnode_base_offset_pair<1, 176, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 42, 0>::value, vnode_base_offset_pair<1, 176, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1, 176, 42, 1>::value, vnode_base_offset_pair<1, 176, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 43, 0>::value, vnode_base_offset_pair<1, 176, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1, 176, 43, 1>::value, vnode_base_offset_pair<1, 176, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 43, 2>::value, vnode_base_offset_pair<1, 176, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 44, 0>::value, vnode_base_offset_pair<1, 176, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1, 176, 44, 1>::value, vnode_base_offset_pair<1, 176, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 176, 44, 2>::value, vnode_base_offset_pair<1, 176, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 176, 45, 0>::value, vnode_base_offset_pair<1, 176, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1, 176, 45, 1>::value, vnode_base_offset_pair<1, 176, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z192_8 =
{
    {
        { vnode_shift_mod_pair<1, 192,  0, 0>::value, vnode_base_offset_pair<1, 192,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1, 192,  0, 1>::value, vnode_base_offset_pair<1, 192,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  0, 2>::value, vnode_base_offset_pair<1, 192,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  0, 3>::value, vnode_base_offset_pair<1, 192,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  0, 4>::value, vnode_base_offset_pair<1, 192,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  0, 5>::value, vnode_base_offset_pair<1, 192,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  0, 6>::value, vnode_base_offset_pair<1, 192,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  0, 7>::value, vnode_base_offset_pair<1, 192,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  0, 8>::value, vnode_base_offset_pair<1, 192,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  0, 9>::value, vnode_base_offset_pair<1, 192,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 192,  1, 0>::value, vnode_base_offset_pair<1, 192,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1, 192,  1, 1>::value, vnode_base_offset_pair<1, 192,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  1, 2>::value, vnode_base_offset_pair<1, 192,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  1, 3>::value, vnode_base_offset_pair<1, 192,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  1, 4>::value, vnode_base_offset_pair<1, 192,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  1, 5>::value, vnode_base_offset_pair<1, 192,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  1, 6>::value, vnode_base_offset_pair<1, 192,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  1, 7>::value, vnode_base_offset_pair<1, 192,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  1, 8>::value, vnode_base_offset_pair<1, 192,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  1, 9>::value, vnode_base_offset_pair<1, 192,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 192,  2, 0>::value, vnode_base_offset_pair<1, 192,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1, 192,  2, 1>::value, vnode_base_offset_pair<1, 192,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  2, 2>::value, vnode_base_offset_pair<1, 192,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  2, 3>::value, vnode_base_offset_pair<1, 192,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  2, 4>::value, vnode_base_offset_pair<1, 192,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  2, 5>::value, vnode_base_offset_pair<1, 192,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  2, 6>::value, vnode_base_offset_pair<1, 192,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  2, 7>::value, vnode_base_offset_pair<1, 192,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  2, 8>::value, vnode_base_offset_pair<1, 192,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  2, 9>::value, vnode_base_offset_pair<1, 192,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 192,  3, 0>::value, vnode_base_offset_pair<1, 192,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1, 192,  3, 1>::value, vnode_base_offset_pair<1, 192,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  3, 2>::value, vnode_base_offset_pair<1, 192,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  3, 3>::value, vnode_base_offset_pair<1, 192,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  3, 4>::value, vnode_base_offset_pair<1, 192,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  3, 5>::value, vnode_base_offset_pair<1, 192,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  3, 6>::value, vnode_base_offset_pair<1, 192,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  3, 7>::value, vnode_base_offset_pair<1, 192,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  3, 8>::value, vnode_base_offset_pair<1, 192,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  3, 9>::value, vnode_base_offset_pair<1, 192,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 192,  4, 0>::value, vnode_base_offset_pair<1, 192,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1, 192,  4, 1>::value, vnode_base_offset_pair<1, 192,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 192,  5, 0>::value, vnode_base_offset_pair<1, 192,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1, 192,  5, 1>::value, vnode_base_offset_pair<1, 192,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  5, 2>::value, vnode_base_offset_pair<1, 192,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  5, 3>::value, vnode_base_offset_pair<1, 192,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 192,  6, 0>::value, vnode_base_offset_pair<1, 192,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1, 192,  6, 1>::value, vnode_base_offset_pair<1, 192,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  6, 2>::value, vnode_base_offset_pair<1, 192,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  6, 3>::value, vnode_base_offset_pair<1, 192,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  6, 4>::value, vnode_base_offset_pair<1, 192,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 192,  7, 0>::value, vnode_base_offset_pair<1, 192,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1, 192,  7, 1>::value, vnode_base_offset_pair<1, 192,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  7, 2>::value, vnode_base_offset_pair<1, 192,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  7, 3>::value, vnode_base_offset_pair<1, 192,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 192,  8, 0>::value, vnode_base_offset_pair<1, 192,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1, 192,  8, 1>::value, vnode_base_offset_pair<1, 192,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  8, 2>::value, vnode_base_offset_pair<1, 192,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  8, 3>::value, vnode_base_offset_pair<1, 192,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  8, 4>::value, vnode_base_offset_pair<1, 192,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 192,  9, 0>::value, vnode_base_offset_pair<1, 192,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1, 192,  9, 1>::value, vnode_base_offset_pair<1, 192,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  9, 2>::value, vnode_base_offset_pair<1, 192,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  9, 3>::value, vnode_base_offset_pair<1, 192,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 192,  9, 4>::value, vnode_base_offset_pair<1, 192,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 10, 0>::value, vnode_base_offset_pair<1, 192, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1, 192, 10, 1>::value, vnode_base_offset_pair<1, 192, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 10, 2>::value, vnode_base_offset_pair<1, 192, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 10, 3>::value, vnode_base_offset_pair<1, 192, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 11, 0>::value, vnode_base_offset_pair<1, 192, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1, 192, 11, 1>::value, vnode_base_offset_pair<1, 192, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 11, 2>::value, vnode_base_offset_pair<1, 192, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 11, 3>::value, vnode_base_offset_pair<1, 192, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 12, 0>::value, vnode_base_offset_pair<1, 192, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1, 192, 12, 1>::value, vnode_base_offset_pair<1, 192, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 12, 2>::value, vnode_base_offset_pair<1, 192, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 12, 3>::value, vnode_base_offset_pair<1, 192, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 13, 0>::value, vnode_base_offset_pair<1, 192, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1, 192, 13, 1>::value, vnode_base_offset_pair<1, 192, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 13, 2>::value, vnode_base_offset_pair<1, 192, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 14, 0>::value, vnode_base_offset_pair<1, 192, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1, 192, 14, 1>::value, vnode_base_offset_pair<1, 192, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 14, 2>::value, vnode_base_offset_pair<1, 192, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 14, 3>::value, vnode_base_offset_pair<1, 192, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 15, 0>::value, vnode_base_offset_pair<1, 192, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1, 192, 15, 1>::value, vnode_base_offset_pair<1, 192, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 15, 2>::value, vnode_base_offset_pair<1, 192, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 15, 3>::value, vnode_base_offset_pair<1, 192, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 16, 0>::value, vnode_base_offset_pair<1, 192, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1, 192, 16, 1>::value, vnode_base_offset_pair<1, 192, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 16, 2>::value, vnode_base_offset_pair<1, 192, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 17, 0>::value, vnode_base_offset_pair<1, 192, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1, 192, 17, 1>::value, vnode_base_offset_pair<1, 192, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 17, 2>::value, vnode_base_offset_pair<1, 192, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 18, 0>::value, vnode_base_offset_pair<1, 192, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1, 192, 18, 1>::value, vnode_base_offset_pair<1, 192, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 18, 2>::value, vnode_base_offset_pair<1, 192, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 19, 0>::value, vnode_base_offset_pair<1, 192, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1, 192, 19, 1>::value, vnode_base_offset_pair<1, 192, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 19, 2>::value, vnode_base_offset_pair<1, 192, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 20, 0>::value, vnode_base_offset_pair<1, 192, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1, 192, 20, 1>::value, vnode_base_offset_pair<1, 192, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 20, 2>::value, vnode_base_offset_pair<1, 192, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 21, 0>::value, vnode_base_offset_pair<1, 192, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1, 192, 21, 1>::value, vnode_base_offset_pair<1, 192, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 21, 2>::value, vnode_base_offset_pair<1, 192, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 22, 0>::value, vnode_base_offset_pair<1, 192, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1, 192, 22, 1>::value, vnode_base_offset_pair<1, 192, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 22, 2>::value, vnode_base_offset_pair<1, 192, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 23, 0>::value, vnode_base_offset_pair<1, 192, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1, 192, 23, 1>::value, vnode_base_offset_pair<1, 192, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 23, 2>::value, vnode_base_offset_pair<1, 192, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 24, 0>::value, vnode_base_offset_pair<1, 192, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1, 192, 24, 1>::value, vnode_base_offset_pair<1, 192, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 24, 2>::value, vnode_base_offset_pair<1, 192, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 25, 0>::value, vnode_base_offset_pair<1, 192, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1, 192, 25, 1>::value, vnode_base_offset_pair<1, 192, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 25, 2>::value, vnode_base_offset_pair<1, 192, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 26, 0>::value, vnode_base_offset_pair<1, 192, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1, 192, 26, 1>::value, vnode_base_offset_pair<1, 192, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 26, 2>::value, vnode_base_offset_pair<1, 192, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 27, 0>::value, vnode_base_offset_pair<1, 192, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1, 192, 27, 1>::value, vnode_base_offset_pair<1, 192, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 28, 0>::value, vnode_base_offset_pair<1, 192, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1, 192, 28, 1>::value, vnode_base_offset_pair<1, 192, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 28, 2>::value, vnode_base_offset_pair<1, 192, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 29, 0>::value, vnode_base_offset_pair<1, 192, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1, 192, 29, 1>::value, vnode_base_offset_pair<1, 192, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 29, 2>::value, vnode_base_offset_pair<1, 192, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 30, 0>::value, vnode_base_offset_pair<1, 192, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1, 192, 30, 1>::value, vnode_base_offset_pair<1, 192, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 30, 2>::value, vnode_base_offset_pair<1, 192, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 31, 0>::value, vnode_base_offset_pair<1, 192, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1, 192, 31, 1>::value, vnode_base_offset_pair<1, 192, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 31, 2>::value, vnode_base_offset_pair<1, 192, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 32, 0>::value, vnode_base_offset_pair<1, 192, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1, 192, 32, 1>::value, vnode_base_offset_pair<1, 192, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 32, 2>::value, vnode_base_offset_pair<1, 192, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 33, 0>::value, vnode_base_offset_pair<1, 192, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1, 192, 33, 1>::value, vnode_base_offset_pair<1, 192, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 33, 2>::value, vnode_base_offset_pair<1, 192, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 34, 0>::value, vnode_base_offset_pair<1, 192, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1, 192, 34, 1>::value, vnode_base_offset_pair<1, 192, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 34, 2>::value, vnode_base_offset_pair<1, 192, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 35, 0>::value, vnode_base_offset_pair<1, 192, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1, 192, 35, 1>::value, vnode_base_offset_pair<1, 192, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 35, 2>::value, vnode_base_offset_pair<1, 192, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 36, 0>::value, vnode_base_offset_pair<1, 192, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1, 192, 36, 1>::value, vnode_base_offset_pair<1, 192, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 36, 2>::value, vnode_base_offset_pair<1, 192, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 37, 0>::value, vnode_base_offset_pair<1, 192, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1, 192, 37, 1>::value, vnode_base_offset_pair<1, 192, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 38, 0>::value, vnode_base_offset_pair<1, 192, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1, 192, 38, 1>::value, vnode_base_offset_pair<1, 192, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 38, 2>::value, vnode_base_offset_pair<1, 192, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 39, 0>::value, vnode_base_offset_pair<1, 192, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1, 192, 39, 1>::value, vnode_base_offset_pair<1, 192, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 39, 2>::value, vnode_base_offset_pair<1, 192, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 40, 0>::value, vnode_base_offset_pair<1, 192, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1, 192, 40, 1>::value, vnode_base_offset_pair<1, 192, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 41, 0>::value, vnode_base_offset_pair<1, 192, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1, 192, 41, 1>::value, vnode_base_offset_pair<1, 192, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 41, 2>::value, vnode_base_offset_pair<1, 192, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 42, 0>::value, vnode_base_offset_pair<1, 192, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1, 192, 42, 1>::value, vnode_base_offset_pair<1, 192, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 43, 0>::value, vnode_base_offset_pair<1, 192, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1, 192, 43, 1>::value, vnode_base_offset_pair<1, 192, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 43, 2>::value, vnode_base_offset_pair<1, 192, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 44, 0>::value, vnode_base_offset_pair<1, 192, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1, 192, 44, 1>::value, vnode_base_offset_pair<1, 192, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 192, 44, 2>::value, vnode_base_offset_pair<1, 192, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 192, 45, 0>::value, vnode_base_offset_pair<1, 192, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1, 192, 45, 1>::value, vnode_base_offset_pair<1, 192, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z208_8 =
{
    {
        { vnode_shift_mod_pair<1, 208,  0, 0>::value, vnode_base_offset_pair<1, 208,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1, 208,  0, 1>::value, vnode_base_offset_pair<1, 208,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  0, 2>::value, vnode_base_offset_pair<1, 208,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  0, 3>::value, vnode_base_offset_pair<1, 208,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  0, 4>::value, vnode_base_offset_pair<1, 208,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  0, 5>::value, vnode_base_offset_pair<1, 208,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  0, 6>::value, vnode_base_offset_pair<1, 208,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  0, 7>::value, vnode_base_offset_pair<1, 208,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  0, 8>::value, vnode_base_offset_pair<1, 208,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  0, 9>::value, vnode_base_offset_pair<1, 208,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 208,  1, 0>::value, vnode_base_offset_pair<1, 208,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1, 208,  1, 1>::value, vnode_base_offset_pair<1, 208,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  1, 2>::value, vnode_base_offset_pair<1, 208,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  1, 3>::value, vnode_base_offset_pair<1, 208,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  1, 4>::value, vnode_base_offset_pair<1, 208,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  1, 5>::value, vnode_base_offset_pair<1, 208,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  1, 6>::value, vnode_base_offset_pair<1, 208,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  1, 7>::value, vnode_base_offset_pair<1, 208,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  1, 8>::value, vnode_base_offset_pair<1, 208,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  1, 9>::value, vnode_base_offset_pair<1, 208,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 208,  2, 0>::value, vnode_base_offset_pair<1, 208,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1, 208,  2, 1>::value, vnode_base_offset_pair<1, 208,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  2, 2>::value, vnode_base_offset_pair<1, 208,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  2, 3>::value, vnode_base_offset_pair<1, 208,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  2, 4>::value, vnode_base_offset_pair<1, 208,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  2, 5>::value, vnode_base_offset_pair<1, 208,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  2, 6>::value, vnode_base_offset_pair<1, 208,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  2, 7>::value, vnode_base_offset_pair<1, 208,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  2, 8>::value, vnode_base_offset_pair<1, 208,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  2, 9>::value, vnode_base_offset_pair<1, 208,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 208,  3, 0>::value, vnode_base_offset_pair<1, 208,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1, 208,  3, 1>::value, vnode_base_offset_pair<1, 208,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  3, 2>::value, vnode_base_offset_pair<1, 208,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  3, 3>::value, vnode_base_offset_pair<1, 208,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  3, 4>::value, vnode_base_offset_pair<1, 208,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  3, 5>::value, vnode_base_offset_pair<1, 208,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  3, 6>::value, vnode_base_offset_pair<1, 208,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  3, 7>::value, vnode_base_offset_pair<1, 208,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  3, 8>::value, vnode_base_offset_pair<1, 208,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  3, 9>::value, vnode_base_offset_pair<1, 208,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 208,  4, 0>::value, vnode_base_offset_pair<1, 208,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1, 208,  4, 1>::value, vnode_base_offset_pair<1, 208,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 208,  5, 0>::value, vnode_base_offset_pair<1, 208,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1, 208,  5, 1>::value, vnode_base_offset_pair<1, 208,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  5, 2>::value, vnode_base_offset_pair<1, 208,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  5, 3>::value, vnode_base_offset_pair<1, 208,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 208,  6, 0>::value, vnode_base_offset_pair<1, 208,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1, 208,  6, 1>::value, vnode_base_offset_pair<1, 208,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  6, 2>::value, vnode_base_offset_pair<1, 208,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  6, 3>::value, vnode_base_offset_pair<1, 208,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  6, 4>::value, vnode_base_offset_pair<1, 208,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 208,  7, 0>::value, vnode_base_offset_pair<1, 208,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1, 208,  7, 1>::value, vnode_base_offset_pair<1, 208,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  7, 2>::value, vnode_base_offset_pair<1, 208,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  7, 3>::value, vnode_base_offset_pair<1, 208,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 208,  8, 0>::value, vnode_base_offset_pair<1, 208,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1, 208,  8, 1>::value, vnode_base_offset_pair<1, 208,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  8, 2>::value, vnode_base_offset_pair<1, 208,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  8, 3>::value, vnode_base_offset_pair<1, 208,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  8, 4>::value, vnode_base_offset_pair<1, 208,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 208,  9, 0>::value, vnode_base_offset_pair<1, 208,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1, 208,  9, 1>::value, vnode_base_offset_pair<1, 208,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  9, 2>::value, vnode_base_offset_pair<1, 208,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  9, 3>::value, vnode_base_offset_pair<1, 208,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 208,  9, 4>::value, vnode_base_offset_pair<1, 208,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 10, 0>::value, vnode_base_offset_pair<1, 208, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1, 208, 10, 1>::value, vnode_base_offset_pair<1, 208, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 10, 2>::value, vnode_base_offset_pair<1, 208, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 10, 3>::value, vnode_base_offset_pair<1, 208, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 11, 0>::value, vnode_base_offset_pair<1, 208, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1, 208, 11, 1>::value, vnode_base_offset_pair<1, 208, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 11, 2>::value, vnode_base_offset_pair<1, 208, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 11, 3>::value, vnode_base_offset_pair<1, 208, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 12, 0>::value, vnode_base_offset_pair<1, 208, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1, 208, 12, 1>::value, vnode_base_offset_pair<1, 208, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 12, 2>::value, vnode_base_offset_pair<1, 208, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 12, 3>::value, vnode_base_offset_pair<1, 208, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 13, 0>::value, vnode_base_offset_pair<1, 208, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1, 208, 13, 1>::value, vnode_base_offset_pair<1, 208, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 13, 2>::value, vnode_base_offset_pair<1, 208, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 14, 0>::value, vnode_base_offset_pair<1, 208, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1, 208, 14, 1>::value, vnode_base_offset_pair<1, 208, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 14, 2>::value, vnode_base_offset_pair<1, 208, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 14, 3>::value, vnode_base_offset_pair<1, 208, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 15, 0>::value, vnode_base_offset_pair<1, 208, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1, 208, 15, 1>::value, vnode_base_offset_pair<1, 208, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 15, 2>::value, vnode_base_offset_pair<1, 208, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 15, 3>::value, vnode_base_offset_pair<1, 208, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 16, 0>::value, vnode_base_offset_pair<1, 208, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1, 208, 16, 1>::value, vnode_base_offset_pair<1, 208, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 16, 2>::value, vnode_base_offset_pair<1, 208, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 17, 0>::value, vnode_base_offset_pair<1, 208, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1, 208, 17, 1>::value, vnode_base_offset_pair<1, 208, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 17, 2>::value, vnode_base_offset_pair<1, 208, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 18, 0>::value, vnode_base_offset_pair<1, 208, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1, 208, 18, 1>::value, vnode_base_offset_pair<1, 208, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 18, 2>::value, vnode_base_offset_pair<1, 208, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 19, 0>::value, vnode_base_offset_pair<1, 208, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1, 208, 19, 1>::value, vnode_base_offset_pair<1, 208, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 19, 2>::value, vnode_base_offset_pair<1, 208, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 20, 0>::value, vnode_base_offset_pair<1, 208, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1, 208, 20, 1>::value, vnode_base_offset_pair<1, 208, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 20, 2>::value, vnode_base_offset_pair<1, 208, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 21, 0>::value, vnode_base_offset_pair<1, 208, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1, 208, 21, 1>::value, vnode_base_offset_pair<1, 208, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 21, 2>::value, vnode_base_offset_pair<1, 208, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 22, 0>::value, vnode_base_offset_pair<1, 208, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1, 208, 22, 1>::value, vnode_base_offset_pair<1, 208, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 22, 2>::value, vnode_base_offset_pair<1, 208, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 23, 0>::value, vnode_base_offset_pair<1, 208, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1, 208, 23, 1>::value, vnode_base_offset_pair<1, 208, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 23, 2>::value, vnode_base_offset_pair<1, 208, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 24, 0>::value, vnode_base_offset_pair<1, 208, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1, 208, 24, 1>::value, vnode_base_offset_pair<1, 208, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 24, 2>::value, vnode_base_offset_pair<1, 208, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 25, 0>::value, vnode_base_offset_pair<1, 208, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1, 208, 25, 1>::value, vnode_base_offset_pair<1, 208, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 25, 2>::value, vnode_base_offset_pair<1, 208, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 26, 0>::value, vnode_base_offset_pair<1, 208, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1, 208, 26, 1>::value, vnode_base_offset_pair<1, 208, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 26, 2>::value, vnode_base_offset_pair<1, 208, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 27, 0>::value, vnode_base_offset_pair<1, 208, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1, 208, 27, 1>::value, vnode_base_offset_pair<1, 208, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 28, 0>::value, vnode_base_offset_pair<1, 208, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1, 208, 28, 1>::value, vnode_base_offset_pair<1, 208, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 28, 2>::value, vnode_base_offset_pair<1, 208, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 29, 0>::value, vnode_base_offset_pair<1, 208, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1, 208, 29, 1>::value, vnode_base_offset_pair<1, 208, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 29, 2>::value, vnode_base_offset_pair<1, 208, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 30, 0>::value, vnode_base_offset_pair<1, 208, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1, 208, 30, 1>::value, vnode_base_offset_pair<1, 208, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 30, 2>::value, vnode_base_offset_pair<1, 208, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 31, 0>::value, vnode_base_offset_pair<1, 208, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1, 208, 31, 1>::value, vnode_base_offset_pair<1, 208, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 31, 2>::value, vnode_base_offset_pair<1, 208, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 32, 0>::value, vnode_base_offset_pair<1, 208, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1, 208, 32, 1>::value, vnode_base_offset_pair<1, 208, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 32, 2>::value, vnode_base_offset_pair<1, 208, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 33, 0>::value, vnode_base_offset_pair<1, 208, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1, 208, 33, 1>::value, vnode_base_offset_pair<1, 208, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 33, 2>::value, vnode_base_offset_pair<1, 208, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 34, 0>::value, vnode_base_offset_pair<1, 208, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1, 208, 34, 1>::value, vnode_base_offset_pair<1, 208, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 34, 2>::value, vnode_base_offset_pair<1, 208, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 35, 0>::value, vnode_base_offset_pair<1, 208, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1, 208, 35, 1>::value, vnode_base_offset_pair<1, 208, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 35, 2>::value, vnode_base_offset_pair<1, 208, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 36, 0>::value, vnode_base_offset_pair<1, 208, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1, 208, 36, 1>::value, vnode_base_offset_pair<1, 208, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 36, 2>::value, vnode_base_offset_pair<1, 208, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 37, 0>::value, vnode_base_offset_pair<1, 208, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1, 208, 37, 1>::value, vnode_base_offset_pair<1, 208, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 38, 0>::value, vnode_base_offset_pair<1, 208, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1, 208, 38, 1>::value, vnode_base_offset_pair<1, 208, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 38, 2>::value, vnode_base_offset_pair<1, 208, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 39, 0>::value, vnode_base_offset_pair<1, 208, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1, 208, 39, 1>::value, vnode_base_offset_pair<1, 208, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 39, 2>::value, vnode_base_offset_pair<1, 208, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 40, 0>::value, vnode_base_offset_pair<1, 208, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1, 208, 40, 1>::value, vnode_base_offset_pair<1, 208, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 41, 0>::value, vnode_base_offset_pair<1, 208, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1, 208, 41, 1>::value, vnode_base_offset_pair<1, 208, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 41, 2>::value, vnode_base_offset_pair<1, 208, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 42, 0>::value, vnode_base_offset_pair<1, 208, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1, 208, 42, 1>::value, vnode_base_offset_pair<1, 208, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 43, 0>::value, vnode_base_offset_pair<1, 208, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1, 208, 43, 1>::value, vnode_base_offset_pair<1, 208, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 43, 2>::value, vnode_base_offset_pair<1, 208, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 44, 0>::value, vnode_base_offset_pair<1, 208, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1, 208, 44, 1>::value, vnode_base_offset_pair<1, 208, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 208, 44, 2>::value, vnode_base_offset_pair<1, 208, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 208, 45, 0>::value, vnode_base_offset_pair<1, 208, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1, 208, 45, 1>::value, vnode_base_offset_pair<1, 208, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z224_8 =
{
    {
        { vnode_shift_mod_pair<1, 224,  0, 0>::value, vnode_base_offset_pair<1, 224,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1, 224,  0, 1>::value, vnode_base_offset_pair<1, 224,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  0, 2>::value, vnode_base_offset_pair<1, 224,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  0, 3>::value, vnode_base_offset_pair<1, 224,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  0, 4>::value, vnode_base_offset_pair<1, 224,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  0, 5>::value, vnode_base_offset_pair<1, 224,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  0, 6>::value, vnode_base_offset_pair<1, 224,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  0, 7>::value, vnode_base_offset_pair<1, 224,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  0, 8>::value, vnode_base_offset_pair<1, 224,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  0, 9>::value, vnode_base_offset_pair<1, 224,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 224,  1, 0>::value, vnode_base_offset_pair<1, 224,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1, 224,  1, 1>::value, vnode_base_offset_pair<1, 224,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  1, 2>::value, vnode_base_offset_pair<1, 224,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  1, 3>::value, vnode_base_offset_pair<1, 224,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  1, 4>::value, vnode_base_offset_pair<1, 224,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  1, 5>::value, vnode_base_offset_pair<1, 224,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  1, 6>::value, vnode_base_offset_pair<1, 224,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  1, 7>::value, vnode_base_offset_pair<1, 224,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  1, 8>::value, vnode_base_offset_pair<1, 224,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  1, 9>::value, vnode_base_offset_pair<1, 224,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 224,  2, 0>::value, vnode_base_offset_pair<1, 224,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1, 224,  2, 1>::value, vnode_base_offset_pair<1, 224,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  2, 2>::value, vnode_base_offset_pair<1, 224,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  2, 3>::value, vnode_base_offset_pair<1, 224,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  2, 4>::value, vnode_base_offset_pair<1, 224,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  2, 5>::value, vnode_base_offset_pair<1, 224,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  2, 6>::value, vnode_base_offset_pair<1, 224,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  2, 7>::value, vnode_base_offset_pair<1, 224,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  2, 8>::value, vnode_base_offset_pair<1, 224,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  2, 9>::value, vnode_base_offset_pair<1, 224,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 224,  3, 0>::value, vnode_base_offset_pair<1, 224,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1, 224,  3, 1>::value, vnode_base_offset_pair<1, 224,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  3, 2>::value, vnode_base_offset_pair<1, 224,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  3, 3>::value, vnode_base_offset_pair<1, 224,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  3, 4>::value, vnode_base_offset_pair<1, 224,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  3, 5>::value, vnode_base_offset_pair<1, 224,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  3, 6>::value, vnode_base_offset_pair<1, 224,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  3, 7>::value, vnode_base_offset_pair<1, 224,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  3, 8>::value, vnode_base_offset_pair<1, 224,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  3, 9>::value, vnode_base_offset_pair<1, 224,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 224,  4, 0>::value, vnode_base_offset_pair<1, 224,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1, 224,  4, 1>::value, vnode_base_offset_pair<1, 224,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 224,  5, 0>::value, vnode_base_offset_pair<1, 224,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1, 224,  5, 1>::value, vnode_base_offset_pair<1, 224,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  5, 2>::value, vnode_base_offset_pair<1, 224,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  5, 3>::value, vnode_base_offset_pair<1, 224,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 224,  6, 0>::value, vnode_base_offset_pair<1, 224,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1, 224,  6, 1>::value, vnode_base_offset_pair<1, 224,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  6, 2>::value, vnode_base_offset_pair<1, 224,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  6, 3>::value, vnode_base_offset_pair<1, 224,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  6, 4>::value, vnode_base_offset_pair<1, 224,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 224,  7, 0>::value, vnode_base_offset_pair<1, 224,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1, 224,  7, 1>::value, vnode_base_offset_pair<1, 224,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  7, 2>::value, vnode_base_offset_pair<1, 224,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  7, 3>::value, vnode_base_offset_pair<1, 224,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 224,  8, 0>::value, vnode_base_offset_pair<1, 224,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1, 224,  8, 1>::value, vnode_base_offset_pair<1, 224,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  8, 2>::value, vnode_base_offset_pair<1, 224,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  8, 3>::value, vnode_base_offset_pair<1, 224,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  8, 4>::value, vnode_base_offset_pair<1, 224,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 224,  9, 0>::value, vnode_base_offset_pair<1, 224,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1, 224,  9, 1>::value, vnode_base_offset_pair<1, 224,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  9, 2>::value, vnode_base_offset_pair<1, 224,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  9, 3>::value, vnode_base_offset_pair<1, 224,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 224,  9, 4>::value, vnode_base_offset_pair<1, 224,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 10, 0>::value, vnode_base_offset_pair<1, 224, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1, 224, 10, 1>::value, vnode_base_offset_pair<1, 224, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 10, 2>::value, vnode_base_offset_pair<1, 224, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 10, 3>::value, vnode_base_offset_pair<1, 224, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 11, 0>::value, vnode_base_offset_pair<1, 224, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1, 224, 11, 1>::value, vnode_base_offset_pair<1, 224, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 11, 2>::value, vnode_base_offset_pair<1, 224, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 11, 3>::value, vnode_base_offset_pair<1, 224, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 12, 0>::value, vnode_base_offset_pair<1, 224, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1, 224, 12, 1>::value, vnode_base_offset_pair<1, 224, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 12, 2>::value, vnode_base_offset_pair<1, 224, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 12, 3>::value, vnode_base_offset_pair<1, 224, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 13, 0>::value, vnode_base_offset_pair<1, 224, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1, 224, 13, 1>::value, vnode_base_offset_pair<1, 224, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 13, 2>::value, vnode_base_offset_pair<1, 224, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 14, 0>::value, vnode_base_offset_pair<1, 224, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1, 224, 14, 1>::value, vnode_base_offset_pair<1, 224, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 14, 2>::value, vnode_base_offset_pair<1, 224, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 14, 3>::value, vnode_base_offset_pair<1, 224, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 15, 0>::value, vnode_base_offset_pair<1, 224, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1, 224, 15, 1>::value, vnode_base_offset_pair<1, 224, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 15, 2>::value, vnode_base_offset_pair<1, 224, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 15, 3>::value, vnode_base_offset_pair<1, 224, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 16, 0>::value, vnode_base_offset_pair<1, 224, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1, 224, 16, 1>::value, vnode_base_offset_pair<1, 224, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 16, 2>::value, vnode_base_offset_pair<1, 224, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 17, 0>::value, vnode_base_offset_pair<1, 224, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1, 224, 17, 1>::value, vnode_base_offset_pair<1, 224, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 17, 2>::value, vnode_base_offset_pair<1, 224, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 18, 0>::value, vnode_base_offset_pair<1, 224, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1, 224, 18, 1>::value, vnode_base_offset_pair<1, 224, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 18, 2>::value, vnode_base_offset_pair<1, 224, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 19, 0>::value, vnode_base_offset_pair<1, 224, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1, 224, 19, 1>::value, vnode_base_offset_pair<1, 224, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 19, 2>::value, vnode_base_offset_pair<1, 224, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 20, 0>::value, vnode_base_offset_pair<1, 224, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1, 224, 20, 1>::value, vnode_base_offset_pair<1, 224, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 20, 2>::value, vnode_base_offset_pair<1, 224, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 21, 0>::value, vnode_base_offset_pair<1, 224, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1, 224, 21, 1>::value, vnode_base_offset_pair<1, 224, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 21, 2>::value, vnode_base_offset_pair<1, 224, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 22, 0>::value, vnode_base_offset_pair<1, 224, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1, 224, 22, 1>::value, vnode_base_offset_pair<1, 224, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 22, 2>::value, vnode_base_offset_pair<1, 224, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 23, 0>::value, vnode_base_offset_pair<1, 224, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1, 224, 23, 1>::value, vnode_base_offset_pair<1, 224, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 23, 2>::value, vnode_base_offset_pair<1, 224, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 24, 0>::value, vnode_base_offset_pair<1, 224, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1, 224, 24, 1>::value, vnode_base_offset_pair<1, 224, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 24, 2>::value, vnode_base_offset_pair<1, 224, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 25, 0>::value, vnode_base_offset_pair<1, 224, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1, 224, 25, 1>::value, vnode_base_offset_pair<1, 224, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 25, 2>::value, vnode_base_offset_pair<1, 224, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 26, 0>::value, vnode_base_offset_pair<1, 224, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1, 224, 26, 1>::value, vnode_base_offset_pair<1, 224, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 26, 2>::value, vnode_base_offset_pair<1, 224, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 27, 0>::value, vnode_base_offset_pair<1, 224, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1, 224, 27, 1>::value, vnode_base_offset_pair<1, 224, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 28, 0>::value, vnode_base_offset_pair<1, 224, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1, 224, 28, 1>::value, vnode_base_offset_pair<1, 224, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 28, 2>::value, vnode_base_offset_pair<1, 224, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 29, 0>::value, vnode_base_offset_pair<1, 224, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1, 224, 29, 1>::value, vnode_base_offset_pair<1, 224, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 29, 2>::value, vnode_base_offset_pair<1, 224, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 30, 0>::value, vnode_base_offset_pair<1, 224, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1, 224, 30, 1>::value, vnode_base_offset_pair<1, 224, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 30, 2>::value, vnode_base_offset_pair<1, 224, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 31, 0>::value, vnode_base_offset_pair<1, 224, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1, 224, 31, 1>::value, vnode_base_offset_pair<1, 224, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 31, 2>::value, vnode_base_offset_pair<1, 224, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 32, 0>::value, vnode_base_offset_pair<1, 224, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1, 224, 32, 1>::value, vnode_base_offset_pair<1, 224, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 32, 2>::value, vnode_base_offset_pair<1, 224, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 33, 0>::value, vnode_base_offset_pair<1, 224, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1, 224, 33, 1>::value, vnode_base_offset_pair<1, 224, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 33, 2>::value, vnode_base_offset_pair<1, 224, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 34, 0>::value, vnode_base_offset_pair<1, 224, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1, 224, 34, 1>::value, vnode_base_offset_pair<1, 224, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 34, 2>::value, vnode_base_offset_pair<1, 224, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 35, 0>::value, vnode_base_offset_pair<1, 224, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1, 224, 35, 1>::value, vnode_base_offset_pair<1, 224, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 35, 2>::value, vnode_base_offset_pair<1, 224, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 36, 0>::value, vnode_base_offset_pair<1, 224, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1, 224, 36, 1>::value, vnode_base_offset_pair<1, 224, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 36, 2>::value, vnode_base_offset_pair<1, 224, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 37, 0>::value, vnode_base_offset_pair<1, 224, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1, 224, 37, 1>::value, vnode_base_offset_pair<1, 224, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 38, 0>::value, vnode_base_offset_pair<1, 224, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1, 224, 38, 1>::value, vnode_base_offset_pair<1, 224, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 38, 2>::value, vnode_base_offset_pair<1, 224, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 39, 0>::value, vnode_base_offset_pair<1, 224, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1, 224, 39, 1>::value, vnode_base_offset_pair<1, 224, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 39, 2>::value, vnode_base_offset_pair<1, 224, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 40, 0>::value, vnode_base_offset_pair<1, 224, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1, 224, 40, 1>::value, vnode_base_offset_pair<1, 224, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 41, 0>::value, vnode_base_offset_pair<1, 224, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1, 224, 41, 1>::value, vnode_base_offset_pair<1, 224, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 41, 2>::value, vnode_base_offset_pair<1, 224, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 42, 0>::value, vnode_base_offset_pair<1, 224, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1, 224, 42, 1>::value, vnode_base_offset_pair<1, 224, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 43, 0>::value, vnode_base_offset_pair<1, 224, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1, 224, 43, 1>::value, vnode_base_offset_pair<1, 224, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 43, 2>::value, vnode_base_offset_pair<1, 224, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 44, 0>::value, vnode_base_offset_pair<1, 224, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1, 224, 44, 1>::value, vnode_base_offset_pair<1, 224, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 224, 44, 2>::value, vnode_base_offset_pair<1, 224, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 224, 45, 0>::value, vnode_base_offset_pair<1, 224, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1, 224, 45, 1>::value, vnode_base_offset_pair<1, 224, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z240_8 =
{
    {
        { vnode_shift_mod_pair<1, 240,  0, 0>::value, vnode_base_offset_pair<1, 240,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1, 240,  0, 1>::value, vnode_base_offset_pair<1, 240,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  0, 2>::value, vnode_base_offset_pair<1, 240,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  0, 3>::value, vnode_base_offset_pair<1, 240,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  0, 4>::value, vnode_base_offset_pair<1, 240,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  0, 5>::value, vnode_base_offset_pair<1, 240,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  0, 6>::value, vnode_base_offset_pair<1, 240,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  0, 7>::value, vnode_base_offset_pair<1, 240,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  0, 8>::value, vnode_base_offset_pair<1, 240,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  0, 9>::value, vnode_base_offset_pair<1, 240,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 240,  1, 0>::value, vnode_base_offset_pair<1, 240,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1, 240,  1, 1>::value, vnode_base_offset_pair<1, 240,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  1, 2>::value, vnode_base_offset_pair<1, 240,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  1, 3>::value, vnode_base_offset_pair<1, 240,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  1, 4>::value, vnode_base_offset_pair<1, 240,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  1, 5>::value, vnode_base_offset_pair<1, 240,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  1, 6>::value, vnode_base_offset_pair<1, 240,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  1, 7>::value, vnode_base_offset_pair<1, 240,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  1, 8>::value, vnode_base_offset_pair<1, 240,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  1, 9>::value, vnode_base_offset_pair<1, 240,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 240,  2, 0>::value, vnode_base_offset_pair<1, 240,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1, 240,  2, 1>::value, vnode_base_offset_pair<1, 240,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  2, 2>::value, vnode_base_offset_pair<1, 240,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  2, 3>::value, vnode_base_offset_pair<1, 240,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  2, 4>::value, vnode_base_offset_pair<1, 240,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  2, 5>::value, vnode_base_offset_pair<1, 240,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  2, 6>::value, vnode_base_offset_pair<1, 240,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  2, 7>::value, vnode_base_offset_pair<1, 240,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  2, 8>::value, vnode_base_offset_pair<1, 240,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  2, 9>::value, vnode_base_offset_pair<1, 240,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 240,  3, 0>::value, vnode_base_offset_pair<1, 240,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1, 240,  3, 1>::value, vnode_base_offset_pair<1, 240,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  3, 2>::value, vnode_base_offset_pair<1, 240,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  3, 3>::value, vnode_base_offset_pair<1, 240,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  3, 4>::value, vnode_base_offset_pair<1, 240,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  3, 5>::value, vnode_base_offset_pair<1, 240,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  3, 6>::value, vnode_base_offset_pair<1, 240,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  3, 7>::value, vnode_base_offset_pair<1, 240,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  3, 8>::value, vnode_base_offset_pair<1, 240,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  3, 9>::value, vnode_base_offset_pair<1, 240,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 240,  4, 0>::value, vnode_base_offset_pair<1, 240,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1, 240,  4, 1>::value, vnode_base_offset_pair<1, 240,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 240,  5, 0>::value, vnode_base_offset_pair<1, 240,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1, 240,  5, 1>::value, vnode_base_offset_pair<1, 240,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  5, 2>::value, vnode_base_offset_pair<1, 240,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  5, 3>::value, vnode_base_offset_pair<1, 240,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 240,  6, 0>::value, vnode_base_offset_pair<1, 240,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1, 240,  6, 1>::value, vnode_base_offset_pair<1, 240,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  6, 2>::value, vnode_base_offset_pair<1, 240,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  6, 3>::value, vnode_base_offset_pair<1, 240,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  6, 4>::value, vnode_base_offset_pair<1, 240,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 240,  7, 0>::value, vnode_base_offset_pair<1, 240,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1, 240,  7, 1>::value, vnode_base_offset_pair<1, 240,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  7, 2>::value, vnode_base_offset_pair<1, 240,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  7, 3>::value, vnode_base_offset_pair<1, 240,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 240,  8, 0>::value, vnode_base_offset_pair<1, 240,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1, 240,  8, 1>::value, vnode_base_offset_pair<1, 240,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  8, 2>::value, vnode_base_offset_pair<1, 240,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  8, 3>::value, vnode_base_offset_pair<1, 240,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  8, 4>::value, vnode_base_offset_pair<1, 240,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 240,  9, 0>::value, vnode_base_offset_pair<1, 240,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1, 240,  9, 1>::value, vnode_base_offset_pair<1, 240,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  9, 2>::value, vnode_base_offset_pair<1, 240,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  9, 3>::value, vnode_base_offset_pair<1, 240,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 240,  9, 4>::value, vnode_base_offset_pair<1, 240,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 10, 0>::value, vnode_base_offset_pair<1, 240, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1, 240, 10, 1>::value, vnode_base_offset_pair<1, 240, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 10, 2>::value, vnode_base_offset_pair<1, 240, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 10, 3>::value, vnode_base_offset_pair<1, 240, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 11, 0>::value, vnode_base_offset_pair<1, 240, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1, 240, 11, 1>::value, vnode_base_offset_pair<1, 240, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 11, 2>::value, vnode_base_offset_pair<1, 240, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 11, 3>::value, vnode_base_offset_pair<1, 240, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 12, 0>::value, vnode_base_offset_pair<1, 240, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1, 240, 12, 1>::value, vnode_base_offset_pair<1, 240, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 12, 2>::value, vnode_base_offset_pair<1, 240, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 12, 3>::value, vnode_base_offset_pair<1, 240, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 13, 0>::value, vnode_base_offset_pair<1, 240, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1, 240, 13, 1>::value, vnode_base_offset_pair<1, 240, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 13, 2>::value, vnode_base_offset_pair<1, 240, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 14, 0>::value, vnode_base_offset_pair<1, 240, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1, 240, 14, 1>::value, vnode_base_offset_pair<1, 240, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 14, 2>::value, vnode_base_offset_pair<1, 240, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 14, 3>::value, vnode_base_offset_pair<1, 240, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 15, 0>::value, vnode_base_offset_pair<1, 240, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1, 240, 15, 1>::value, vnode_base_offset_pair<1, 240, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 15, 2>::value, vnode_base_offset_pair<1, 240, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 15, 3>::value, vnode_base_offset_pair<1, 240, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 16, 0>::value, vnode_base_offset_pair<1, 240, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1, 240, 16, 1>::value, vnode_base_offset_pair<1, 240, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 16, 2>::value, vnode_base_offset_pair<1, 240, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 17, 0>::value, vnode_base_offset_pair<1, 240, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1, 240, 17, 1>::value, vnode_base_offset_pair<1, 240, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 17, 2>::value, vnode_base_offset_pair<1, 240, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 18, 0>::value, vnode_base_offset_pair<1, 240, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1, 240, 18, 1>::value, vnode_base_offset_pair<1, 240, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 18, 2>::value, vnode_base_offset_pair<1, 240, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 19, 0>::value, vnode_base_offset_pair<1, 240, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1, 240, 19, 1>::value, vnode_base_offset_pair<1, 240, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 19, 2>::value, vnode_base_offset_pair<1, 240, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 20, 0>::value, vnode_base_offset_pair<1, 240, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1, 240, 20, 1>::value, vnode_base_offset_pair<1, 240, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 20, 2>::value, vnode_base_offset_pair<1, 240, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 21, 0>::value, vnode_base_offset_pair<1, 240, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1, 240, 21, 1>::value, vnode_base_offset_pair<1, 240, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 21, 2>::value, vnode_base_offset_pair<1, 240, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 22, 0>::value, vnode_base_offset_pair<1, 240, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1, 240, 22, 1>::value, vnode_base_offset_pair<1, 240, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 22, 2>::value, vnode_base_offset_pair<1, 240, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 23, 0>::value, vnode_base_offset_pair<1, 240, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1, 240, 23, 1>::value, vnode_base_offset_pair<1, 240, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 23, 2>::value, vnode_base_offset_pair<1, 240, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 24, 0>::value, vnode_base_offset_pair<1, 240, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1, 240, 24, 1>::value, vnode_base_offset_pair<1, 240, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 24, 2>::value, vnode_base_offset_pair<1, 240, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 25, 0>::value, vnode_base_offset_pair<1, 240, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1, 240, 25, 1>::value, vnode_base_offset_pair<1, 240, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 25, 2>::value, vnode_base_offset_pair<1, 240, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 26, 0>::value, vnode_base_offset_pair<1, 240, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1, 240, 26, 1>::value, vnode_base_offset_pair<1, 240, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 26, 2>::value, vnode_base_offset_pair<1, 240, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 27, 0>::value, vnode_base_offset_pair<1, 240, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1, 240, 27, 1>::value, vnode_base_offset_pair<1, 240, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 28, 0>::value, vnode_base_offset_pair<1, 240, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1, 240, 28, 1>::value, vnode_base_offset_pair<1, 240, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 28, 2>::value, vnode_base_offset_pair<1, 240, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 29, 0>::value, vnode_base_offset_pair<1, 240, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1, 240, 29, 1>::value, vnode_base_offset_pair<1, 240, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 29, 2>::value, vnode_base_offset_pair<1, 240, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 30, 0>::value, vnode_base_offset_pair<1, 240, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1, 240, 30, 1>::value, vnode_base_offset_pair<1, 240, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 30, 2>::value, vnode_base_offset_pair<1, 240, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 31, 0>::value, vnode_base_offset_pair<1, 240, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1, 240, 31, 1>::value, vnode_base_offset_pair<1, 240, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 31, 2>::value, vnode_base_offset_pair<1, 240, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 32, 0>::value, vnode_base_offset_pair<1, 240, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1, 240, 32, 1>::value, vnode_base_offset_pair<1, 240, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 32, 2>::value, vnode_base_offset_pair<1, 240, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 33, 0>::value, vnode_base_offset_pair<1, 240, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1, 240, 33, 1>::value, vnode_base_offset_pair<1, 240, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 33, 2>::value, vnode_base_offset_pair<1, 240, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 34, 0>::value, vnode_base_offset_pair<1, 240, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1, 240, 34, 1>::value, vnode_base_offset_pair<1, 240, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 34, 2>::value, vnode_base_offset_pair<1, 240, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 35, 0>::value, vnode_base_offset_pair<1, 240, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1, 240, 35, 1>::value, vnode_base_offset_pair<1, 240, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 35, 2>::value, vnode_base_offset_pair<1, 240, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 36, 0>::value, vnode_base_offset_pair<1, 240, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1, 240, 36, 1>::value, vnode_base_offset_pair<1, 240, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 36, 2>::value, vnode_base_offset_pair<1, 240, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 37, 0>::value, vnode_base_offset_pair<1, 240, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1, 240, 37, 1>::value, vnode_base_offset_pair<1, 240, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 38, 0>::value, vnode_base_offset_pair<1, 240, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1, 240, 38, 1>::value, vnode_base_offset_pair<1, 240, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 38, 2>::value, vnode_base_offset_pair<1, 240, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 39, 0>::value, vnode_base_offset_pair<1, 240, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1, 240, 39, 1>::value, vnode_base_offset_pair<1, 240, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 39, 2>::value, vnode_base_offset_pair<1, 240, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 40, 0>::value, vnode_base_offset_pair<1, 240, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1, 240, 40, 1>::value, vnode_base_offset_pair<1, 240, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 41, 0>::value, vnode_base_offset_pair<1, 240, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1, 240, 41, 1>::value, vnode_base_offset_pair<1, 240, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 41, 2>::value, vnode_base_offset_pair<1, 240, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 42, 0>::value, vnode_base_offset_pair<1, 240, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1, 240, 42, 1>::value, vnode_base_offset_pair<1, 240, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 43, 0>::value, vnode_base_offset_pair<1, 240, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1, 240, 43, 1>::value, vnode_base_offset_pair<1, 240, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 43, 2>::value, vnode_base_offset_pair<1, 240, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 44, 0>::value, vnode_base_offset_pair<1, 240, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1, 240, 44, 1>::value, vnode_base_offset_pair<1, 240, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 240, 44, 2>::value, vnode_base_offset_pair<1, 240, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 240, 45, 0>::value, vnode_base_offset_pair<1, 240, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1, 240, 45, 1>::value, vnode_base_offset_pair<1, 240, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z256_8 =
{
    {
        { vnode_shift_mod_pair<1, 256,  0, 0>::value, vnode_base_offset_pair<1, 256,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1, 256,  0, 1>::value, vnode_base_offset_pair<1, 256,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  0, 2>::value, vnode_base_offset_pair<1, 256,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  0, 3>::value, vnode_base_offset_pair<1, 256,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  0, 4>::value, vnode_base_offset_pair<1, 256,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  0, 5>::value, vnode_base_offset_pair<1, 256,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  0, 6>::value, vnode_base_offset_pair<1, 256,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  0, 7>::value, vnode_base_offset_pair<1, 256,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  0, 8>::value, vnode_base_offset_pair<1, 256,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  0, 9>::value, vnode_base_offset_pair<1, 256,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 256,  1, 0>::value, vnode_base_offset_pair<1, 256,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1, 256,  1, 1>::value, vnode_base_offset_pair<1, 256,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  1, 2>::value, vnode_base_offset_pair<1, 256,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  1, 3>::value, vnode_base_offset_pair<1, 256,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  1, 4>::value, vnode_base_offset_pair<1, 256,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  1, 5>::value, vnode_base_offset_pair<1, 256,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  1, 6>::value, vnode_base_offset_pair<1, 256,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  1, 7>::value, vnode_base_offset_pair<1, 256,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  1, 8>::value, vnode_base_offset_pair<1, 256,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  1, 9>::value, vnode_base_offset_pair<1, 256,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 256,  2, 0>::value, vnode_base_offset_pair<1, 256,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1, 256,  2, 1>::value, vnode_base_offset_pair<1, 256,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  2, 2>::value, vnode_base_offset_pair<1, 256,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  2, 3>::value, vnode_base_offset_pair<1, 256,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  2, 4>::value, vnode_base_offset_pair<1, 256,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  2, 5>::value, vnode_base_offset_pair<1, 256,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  2, 6>::value, vnode_base_offset_pair<1, 256,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  2, 7>::value, vnode_base_offset_pair<1, 256,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  2, 8>::value, vnode_base_offset_pair<1, 256,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  2, 9>::value, vnode_base_offset_pair<1, 256,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 256,  3, 0>::value, vnode_base_offset_pair<1, 256,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1, 256,  3, 1>::value, vnode_base_offset_pair<1, 256,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  3, 2>::value, vnode_base_offset_pair<1, 256,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  3, 3>::value, vnode_base_offset_pair<1, 256,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  3, 4>::value, vnode_base_offset_pair<1, 256,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  3, 5>::value, vnode_base_offset_pair<1, 256,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  3, 6>::value, vnode_base_offset_pair<1, 256,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  3, 7>::value, vnode_base_offset_pair<1, 256,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  3, 8>::value, vnode_base_offset_pair<1, 256,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  3, 9>::value, vnode_base_offset_pair<1, 256,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 256,  4, 0>::value, vnode_base_offset_pair<1, 256,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1, 256,  4, 1>::value, vnode_base_offset_pair<1, 256,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 256,  5, 0>::value, vnode_base_offset_pair<1, 256,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1, 256,  5, 1>::value, vnode_base_offset_pair<1, 256,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  5, 2>::value, vnode_base_offset_pair<1, 256,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  5, 3>::value, vnode_base_offset_pair<1, 256,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 256,  6, 0>::value, vnode_base_offset_pair<1, 256,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1, 256,  6, 1>::value, vnode_base_offset_pair<1, 256,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  6, 2>::value, vnode_base_offset_pair<1, 256,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  6, 3>::value, vnode_base_offset_pair<1, 256,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  6, 4>::value, vnode_base_offset_pair<1, 256,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 256,  7, 0>::value, vnode_base_offset_pair<1, 256,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1, 256,  7, 1>::value, vnode_base_offset_pair<1, 256,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  7, 2>::value, vnode_base_offset_pair<1, 256,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  7, 3>::value, vnode_base_offset_pair<1, 256,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 256,  8, 0>::value, vnode_base_offset_pair<1, 256,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1, 256,  8, 1>::value, vnode_base_offset_pair<1, 256,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  8, 2>::value, vnode_base_offset_pair<1, 256,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  8, 3>::value, vnode_base_offset_pair<1, 256,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  8, 4>::value, vnode_base_offset_pair<1, 256,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 256,  9, 0>::value, vnode_base_offset_pair<1, 256,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1, 256,  9, 1>::value, vnode_base_offset_pair<1, 256,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  9, 2>::value, vnode_base_offset_pair<1, 256,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  9, 3>::value, vnode_base_offset_pair<1, 256,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 256,  9, 4>::value, vnode_base_offset_pair<1, 256,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 10, 0>::value, vnode_base_offset_pair<1, 256, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1, 256, 10, 1>::value, vnode_base_offset_pair<1, 256, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 10, 2>::value, vnode_base_offset_pair<1, 256, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 10, 3>::value, vnode_base_offset_pair<1, 256, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 11, 0>::value, vnode_base_offset_pair<1, 256, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1, 256, 11, 1>::value, vnode_base_offset_pair<1, 256, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 11, 2>::value, vnode_base_offset_pair<1, 256, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 11, 3>::value, vnode_base_offset_pair<1, 256, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 12, 0>::value, vnode_base_offset_pair<1, 256, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1, 256, 12, 1>::value, vnode_base_offset_pair<1, 256, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 12, 2>::value, vnode_base_offset_pair<1, 256, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 12, 3>::value, vnode_base_offset_pair<1, 256, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 13, 0>::value, vnode_base_offset_pair<1, 256, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1, 256, 13, 1>::value, vnode_base_offset_pair<1, 256, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 13, 2>::value, vnode_base_offset_pair<1, 256, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 14, 0>::value, vnode_base_offset_pair<1, 256, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1, 256, 14, 1>::value, vnode_base_offset_pair<1, 256, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 14, 2>::value, vnode_base_offset_pair<1, 256, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 14, 3>::value, vnode_base_offset_pair<1, 256, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 15, 0>::value, vnode_base_offset_pair<1, 256, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1, 256, 15, 1>::value, vnode_base_offset_pair<1, 256, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 15, 2>::value, vnode_base_offset_pair<1, 256, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 15, 3>::value, vnode_base_offset_pair<1, 256, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 16, 0>::value, vnode_base_offset_pair<1, 256, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1, 256, 16, 1>::value, vnode_base_offset_pair<1, 256, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 16, 2>::value, vnode_base_offset_pair<1, 256, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 17, 0>::value, vnode_base_offset_pair<1, 256, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1, 256, 17, 1>::value, vnode_base_offset_pair<1, 256, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 17, 2>::value, vnode_base_offset_pair<1, 256, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 18, 0>::value, vnode_base_offset_pair<1, 256, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1, 256, 18, 1>::value, vnode_base_offset_pair<1, 256, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 18, 2>::value, vnode_base_offset_pair<1, 256, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 19, 0>::value, vnode_base_offset_pair<1, 256, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1, 256, 19, 1>::value, vnode_base_offset_pair<1, 256, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 19, 2>::value, vnode_base_offset_pair<1, 256, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 20, 0>::value, vnode_base_offset_pair<1, 256, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1, 256, 20, 1>::value, vnode_base_offset_pair<1, 256, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 20, 2>::value, vnode_base_offset_pair<1, 256, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 21, 0>::value, vnode_base_offset_pair<1, 256, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1, 256, 21, 1>::value, vnode_base_offset_pair<1, 256, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 21, 2>::value, vnode_base_offset_pair<1, 256, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 22, 0>::value, vnode_base_offset_pair<1, 256, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1, 256, 22, 1>::value, vnode_base_offset_pair<1, 256, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 22, 2>::value, vnode_base_offset_pair<1, 256, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 23, 0>::value, vnode_base_offset_pair<1, 256, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1, 256, 23, 1>::value, vnode_base_offset_pair<1, 256, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 23, 2>::value, vnode_base_offset_pair<1, 256, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 24, 0>::value, vnode_base_offset_pair<1, 256, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1, 256, 24, 1>::value, vnode_base_offset_pair<1, 256, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 24, 2>::value, vnode_base_offset_pair<1, 256, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 25, 0>::value, vnode_base_offset_pair<1, 256, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1, 256, 25, 1>::value, vnode_base_offset_pair<1, 256, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 25, 2>::value, vnode_base_offset_pair<1, 256, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 26, 0>::value, vnode_base_offset_pair<1, 256, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1, 256, 26, 1>::value, vnode_base_offset_pair<1, 256, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 26, 2>::value, vnode_base_offset_pair<1, 256, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 27, 0>::value, vnode_base_offset_pair<1, 256, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1, 256, 27, 1>::value, vnode_base_offset_pair<1, 256, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 28, 0>::value, vnode_base_offset_pair<1, 256, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1, 256, 28, 1>::value, vnode_base_offset_pair<1, 256, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 28, 2>::value, vnode_base_offset_pair<1, 256, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 29, 0>::value, vnode_base_offset_pair<1, 256, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1, 256, 29, 1>::value, vnode_base_offset_pair<1, 256, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 29, 2>::value, vnode_base_offset_pair<1, 256, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 30, 0>::value, vnode_base_offset_pair<1, 256, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1, 256, 30, 1>::value, vnode_base_offset_pair<1, 256, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 30, 2>::value, vnode_base_offset_pair<1, 256, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 31, 0>::value, vnode_base_offset_pair<1, 256, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1, 256, 31, 1>::value, vnode_base_offset_pair<1, 256, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 31, 2>::value, vnode_base_offset_pair<1, 256, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 32, 0>::value, vnode_base_offset_pair<1, 256, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1, 256, 32, 1>::value, vnode_base_offset_pair<1, 256, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 32, 2>::value, vnode_base_offset_pair<1, 256, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 33, 0>::value, vnode_base_offset_pair<1, 256, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1, 256, 33, 1>::value, vnode_base_offset_pair<1, 256, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 33, 2>::value, vnode_base_offset_pair<1, 256, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 34, 0>::value, vnode_base_offset_pair<1, 256, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1, 256, 34, 1>::value, vnode_base_offset_pair<1, 256, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 34, 2>::value, vnode_base_offset_pair<1, 256, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 35, 0>::value, vnode_base_offset_pair<1, 256, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1, 256, 35, 1>::value, vnode_base_offset_pair<1, 256, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 35, 2>::value, vnode_base_offset_pair<1, 256, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 36, 0>::value, vnode_base_offset_pair<1, 256, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1, 256, 36, 1>::value, vnode_base_offset_pair<1, 256, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 36, 2>::value, vnode_base_offset_pair<1, 256, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 37, 0>::value, vnode_base_offset_pair<1, 256, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1, 256, 37, 1>::value, vnode_base_offset_pair<1, 256, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 38, 0>::value, vnode_base_offset_pair<1, 256, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1, 256, 38, 1>::value, vnode_base_offset_pair<1, 256, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 38, 2>::value, vnode_base_offset_pair<1, 256, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 39, 0>::value, vnode_base_offset_pair<1, 256, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1, 256, 39, 1>::value, vnode_base_offset_pair<1, 256, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 39, 2>::value, vnode_base_offset_pair<1, 256, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 40, 0>::value, vnode_base_offset_pair<1, 256, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1, 256, 40, 1>::value, vnode_base_offset_pair<1, 256, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 41, 0>::value, vnode_base_offset_pair<1, 256, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1, 256, 41, 1>::value, vnode_base_offset_pair<1, 256, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 41, 2>::value, vnode_base_offset_pair<1, 256, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 42, 0>::value, vnode_base_offset_pair<1, 256, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1, 256, 42, 1>::value, vnode_base_offset_pair<1, 256, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 43, 0>::value, vnode_base_offset_pair<1, 256, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1, 256, 43, 1>::value, vnode_base_offset_pair<1, 256, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 43, 2>::value, vnode_base_offset_pair<1, 256, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 44, 0>::value, vnode_base_offset_pair<1, 256, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1, 256, 44, 1>::value, vnode_base_offset_pair<1, 256, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 256, 44, 2>::value, vnode_base_offset_pair<1, 256, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 256, 45, 0>::value, vnode_base_offset_pair<1, 256, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1, 256, 45, 1>::value, vnode_base_offset_pair<1, 256, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z288_8 =
{
    {
        { vnode_shift_mod_pair<1, 288,  0, 0>::value, vnode_base_offset_pair<1, 288,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1, 288,  0, 1>::value, vnode_base_offset_pair<1, 288,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  0, 2>::value, vnode_base_offset_pair<1, 288,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  0, 3>::value, vnode_base_offset_pair<1, 288,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  0, 4>::value, vnode_base_offset_pair<1, 288,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  0, 5>::value, vnode_base_offset_pair<1, 288,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  0, 6>::value, vnode_base_offset_pair<1, 288,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  0, 7>::value, vnode_base_offset_pair<1, 288,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  0, 8>::value, vnode_base_offset_pair<1, 288,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  0, 9>::value, vnode_base_offset_pair<1, 288,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 288,  1, 0>::value, vnode_base_offset_pair<1, 288,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1, 288,  1, 1>::value, vnode_base_offset_pair<1, 288,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  1, 2>::value, vnode_base_offset_pair<1, 288,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  1, 3>::value, vnode_base_offset_pair<1, 288,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  1, 4>::value, vnode_base_offset_pair<1, 288,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  1, 5>::value, vnode_base_offset_pair<1, 288,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  1, 6>::value, vnode_base_offset_pair<1, 288,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  1, 7>::value, vnode_base_offset_pair<1, 288,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  1, 8>::value, vnode_base_offset_pair<1, 288,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  1, 9>::value, vnode_base_offset_pair<1, 288,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 288,  2, 0>::value, vnode_base_offset_pair<1, 288,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1, 288,  2, 1>::value, vnode_base_offset_pair<1, 288,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  2, 2>::value, vnode_base_offset_pair<1, 288,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  2, 3>::value, vnode_base_offset_pair<1, 288,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  2, 4>::value, vnode_base_offset_pair<1, 288,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  2, 5>::value, vnode_base_offset_pair<1, 288,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  2, 6>::value, vnode_base_offset_pair<1, 288,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  2, 7>::value, vnode_base_offset_pair<1, 288,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  2, 8>::value, vnode_base_offset_pair<1, 288,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  2, 9>::value, vnode_base_offset_pair<1, 288,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 288,  3, 0>::value, vnode_base_offset_pair<1, 288,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1, 288,  3, 1>::value, vnode_base_offset_pair<1, 288,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  3, 2>::value, vnode_base_offset_pair<1, 288,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  3, 3>::value, vnode_base_offset_pair<1, 288,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  3, 4>::value, vnode_base_offset_pair<1, 288,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  3, 5>::value, vnode_base_offset_pair<1, 288,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  3, 6>::value, vnode_base_offset_pair<1, 288,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  3, 7>::value, vnode_base_offset_pair<1, 288,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  3, 8>::value, vnode_base_offset_pair<1, 288,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  3, 9>::value, vnode_base_offset_pair<1, 288,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 288,  4, 0>::value, vnode_base_offset_pair<1, 288,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1, 288,  4, 1>::value, vnode_base_offset_pair<1, 288,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 288,  5, 0>::value, vnode_base_offset_pair<1, 288,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1, 288,  5, 1>::value, vnode_base_offset_pair<1, 288,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  5, 2>::value, vnode_base_offset_pair<1, 288,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  5, 3>::value, vnode_base_offset_pair<1, 288,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 288,  6, 0>::value, vnode_base_offset_pair<1, 288,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1, 288,  6, 1>::value, vnode_base_offset_pair<1, 288,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  6, 2>::value, vnode_base_offset_pair<1, 288,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  6, 3>::value, vnode_base_offset_pair<1, 288,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  6, 4>::value, vnode_base_offset_pair<1, 288,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 288,  7, 0>::value, vnode_base_offset_pair<1, 288,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1, 288,  7, 1>::value, vnode_base_offset_pair<1, 288,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  7, 2>::value, vnode_base_offset_pair<1, 288,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  7, 3>::value, vnode_base_offset_pair<1, 288,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 288,  8, 0>::value, vnode_base_offset_pair<1, 288,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1, 288,  8, 1>::value, vnode_base_offset_pair<1, 288,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  8, 2>::value, vnode_base_offset_pair<1, 288,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  8, 3>::value, vnode_base_offset_pair<1, 288,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  8, 4>::value, vnode_base_offset_pair<1, 288,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 288,  9, 0>::value, vnode_base_offset_pair<1, 288,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1, 288,  9, 1>::value, vnode_base_offset_pair<1, 288,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  9, 2>::value, vnode_base_offset_pair<1, 288,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  9, 3>::value, vnode_base_offset_pair<1, 288,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 288,  9, 4>::value, vnode_base_offset_pair<1, 288,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 10, 0>::value, vnode_base_offset_pair<1, 288, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1, 288, 10, 1>::value, vnode_base_offset_pair<1, 288, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 10, 2>::value, vnode_base_offset_pair<1, 288, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 10, 3>::value, vnode_base_offset_pair<1, 288, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 11, 0>::value, vnode_base_offset_pair<1, 288, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1, 288, 11, 1>::value, vnode_base_offset_pair<1, 288, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 11, 2>::value, vnode_base_offset_pair<1, 288, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 11, 3>::value, vnode_base_offset_pair<1, 288, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 12, 0>::value, vnode_base_offset_pair<1, 288, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1, 288, 12, 1>::value, vnode_base_offset_pair<1, 288, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 12, 2>::value, vnode_base_offset_pair<1, 288, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 12, 3>::value, vnode_base_offset_pair<1, 288, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 13, 0>::value, vnode_base_offset_pair<1, 288, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1, 288, 13, 1>::value, vnode_base_offset_pair<1, 288, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 13, 2>::value, vnode_base_offset_pair<1, 288, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 14, 0>::value, vnode_base_offset_pair<1, 288, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1, 288, 14, 1>::value, vnode_base_offset_pair<1, 288, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 14, 2>::value, vnode_base_offset_pair<1, 288, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 14, 3>::value, vnode_base_offset_pair<1, 288, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 15, 0>::value, vnode_base_offset_pair<1, 288, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1, 288, 15, 1>::value, vnode_base_offset_pair<1, 288, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 15, 2>::value, vnode_base_offset_pair<1, 288, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 15, 3>::value, vnode_base_offset_pair<1, 288, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 16, 0>::value, vnode_base_offset_pair<1, 288, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1, 288, 16, 1>::value, vnode_base_offset_pair<1, 288, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 16, 2>::value, vnode_base_offset_pair<1, 288, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 17, 0>::value, vnode_base_offset_pair<1, 288, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1, 288, 17, 1>::value, vnode_base_offset_pair<1, 288, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 17, 2>::value, vnode_base_offset_pair<1, 288, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 18, 0>::value, vnode_base_offset_pair<1, 288, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1, 288, 18, 1>::value, vnode_base_offset_pair<1, 288, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 18, 2>::value, vnode_base_offset_pair<1, 288, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 19, 0>::value, vnode_base_offset_pair<1, 288, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1, 288, 19, 1>::value, vnode_base_offset_pair<1, 288, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 19, 2>::value, vnode_base_offset_pair<1, 288, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 20, 0>::value, vnode_base_offset_pair<1, 288, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1, 288, 20, 1>::value, vnode_base_offset_pair<1, 288, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 20, 2>::value, vnode_base_offset_pair<1, 288, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 21, 0>::value, vnode_base_offset_pair<1, 288, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1, 288, 21, 1>::value, vnode_base_offset_pair<1, 288, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 21, 2>::value, vnode_base_offset_pair<1, 288, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 22, 0>::value, vnode_base_offset_pair<1, 288, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1, 288, 22, 1>::value, vnode_base_offset_pair<1, 288, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 22, 2>::value, vnode_base_offset_pair<1, 288, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 23, 0>::value, vnode_base_offset_pair<1, 288, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1, 288, 23, 1>::value, vnode_base_offset_pair<1, 288, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 23, 2>::value, vnode_base_offset_pair<1, 288, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 24, 0>::value, vnode_base_offset_pair<1, 288, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1, 288, 24, 1>::value, vnode_base_offset_pair<1, 288, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 24, 2>::value, vnode_base_offset_pair<1, 288, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 25, 0>::value, vnode_base_offset_pair<1, 288, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1, 288, 25, 1>::value, vnode_base_offset_pair<1, 288, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 25, 2>::value, vnode_base_offset_pair<1, 288, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 26, 0>::value, vnode_base_offset_pair<1, 288, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1, 288, 26, 1>::value, vnode_base_offset_pair<1, 288, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 26, 2>::value, vnode_base_offset_pair<1, 288, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 27, 0>::value, vnode_base_offset_pair<1, 288, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1, 288, 27, 1>::value, vnode_base_offset_pair<1, 288, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 28, 0>::value, vnode_base_offset_pair<1, 288, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1, 288, 28, 1>::value, vnode_base_offset_pair<1, 288, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 28, 2>::value, vnode_base_offset_pair<1, 288, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 29, 0>::value, vnode_base_offset_pair<1, 288, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1, 288, 29, 1>::value, vnode_base_offset_pair<1, 288, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 29, 2>::value, vnode_base_offset_pair<1, 288, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 30, 0>::value, vnode_base_offset_pair<1, 288, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1, 288, 30, 1>::value, vnode_base_offset_pair<1, 288, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 30, 2>::value, vnode_base_offset_pair<1, 288, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 31, 0>::value, vnode_base_offset_pair<1, 288, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1, 288, 31, 1>::value, vnode_base_offset_pair<1, 288, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 31, 2>::value, vnode_base_offset_pair<1, 288, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 32, 0>::value, vnode_base_offset_pair<1, 288, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1, 288, 32, 1>::value, vnode_base_offset_pair<1, 288, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 32, 2>::value, vnode_base_offset_pair<1, 288, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 33, 0>::value, vnode_base_offset_pair<1, 288, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1, 288, 33, 1>::value, vnode_base_offset_pair<1, 288, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 33, 2>::value, vnode_base_offset_pair<1, 288, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 34, 0>::value, vnode_base_offset_pair<1, 288, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1, 288, 34, 1>::value, vnode_base_offset_pair<1, 288, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 34, 2>::value, vnode_base_offset_pair<1, 288, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 35, 0>::value, vnode_base_offset_pair<1, 288, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1, 288, 35, 1>::value, vnode_base_offset_pair<1, 288, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 35, 2>::value, vnode_base_offset_pair<1, 288, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 36, 0>::value, vnode_base_offset_pair<1, 288, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1, 288, 36, 1>::value, vnode_base_offset_pair<1, 288, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 36, 2>::value, vnode_base_offset_pair<1, 288, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 37, 0>::value, vnode_base_offset_pair<1, 288, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1, 288, 37, 1>::value, vnode_base_offset_pair<1, 288, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 38, 0>::value, vnode_base_offset_pair<1, 288, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1, 288, 38, 1>::value, vnode_base_offset_pair<1, 288, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 38, 2>::value, vnode_base_offset_pair<1, 288, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 39, 0>::value, vnode_base_offset_pair<1, 288, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1, 288, 39, 1>::value, vnode_base_offset_pair<1, 288, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 39, 2>::value, vnode_base_offset_pair<1, 288, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 40, 0>::value, vnode_base_offset_pair<1, 288, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1, 288, 40, 1>::value, vnode_base_offset_pair<1, 288, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 41, 0>::value, vnode_base_offset_pair<1, 288, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1, 288, 41, 1>::value, vnode_base_offset_pair<1, 288, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 41, 2>::value, vnode_base_offset_pair<1, 288, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 42, 0>::value, vnode_base_offset_pair<1, 288, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1, 288, 42, 1>::value, vnode_base_offset_pair<1, 288, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 43, 0>::value, vnode_base_offset_pair<1, 288, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1, 288, 43, 1>::value, vnode_base_offset_pair<1, 288, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 43, 2>::value, vnode_base_offset_pair<1, 288, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 44, 0>::value, vnode_base_offset_pair<1, 288, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1, 288, 44, 1>::value, vnode_base_offset_pair<1, 288, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 288, 44, 2>::value, vnode_base_offset_pair<1, 288, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 288, 45, 0>::value, vnode_base_offset_pair<1, 288, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1, 288, 45, 1>::value, vnode_base_offset_pair<1, 288, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z320_8 =
{
    {
        { vnode_shift_mod_pair<1, 320,  0, 0>::value, vnode_base_offset_pair<1, 320,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1, 320,  0, 1>::value, vnode_base_offset_pair<1, 320,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  0, 2>::value, vnode_base_offset_pair<1, 320,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  0, 3>::value, vnode_base_offset_pair<1, 320,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  0, 4>::value, vnode_base_offset_pair<1, 320,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  0, 5>::value, vnode_base_offset_pair<1, 320,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  0, 6>::value, vnode_base_offset_pair<1, 320,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  0, 7>::value, vnode_base_offset_pair<1, 320,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  0, 8>::value, vnode_base_offset_pair<1, 320,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  0, 9>::value, vnode_base_offset_pair<1, 320,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 320,  1, 0>::value, vnode_base_offset_pair<1, 320,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1, 320,  1, 1>::value, vnode_base_offset_pair<1, 320,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  1, 2>::value, vnode_base_offset_pair<1, 320,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  1, 3>::value, vnode_base_offset_pair<1, 320,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  1, 4>::value, vnode_base_offset_pair<1, 320,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  1, 5>::value, vnode_base_offset_pair<1, 320,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  1, 6>::value, vnode_base_offset_pair<1, 320,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  1, 7>::value, vnode_base_offset_pair<1, 320,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  1, 8>::value, vnode_base_offset_pair<1, 320,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  1, 9>::value, vnode_base_offset_pair<1, 320,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 320,  2, 0>::value, vnode_base_offset_pair<1, 320,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1, 320,  2, 1>::value, vnode_base_offset_pair<1, 320,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  2, 2>::value, vnode_base_offset_pair<1, 320,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  2, 3>::value, vnode_base_offset_pair<1, 320,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  2, 4>::value, vnode_base_offset_pair<1, 320,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  2, 5>::value, vnode_base_offset_pair<1, 320,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  2, 6>::value, vnode_base_offset_pair<1, 320,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  2, 7>::value, vnode_base_offset_pair<1, 320,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  2, 8>::value, vnode_base_offset_pair<1, 320,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  2, 9>::value, vnode_base_offset_pair<1, 320,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 320,  3, 0>::value, vnode_base_offset_pair<1, 320,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1, 320,  3, 1>::value, vnode_base_offset_pair<1, 320,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  3, 2>::value, vnode_base_offset_pair<1, 320,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  3, 3>::value, vnode_base_offset_pair<1, 320,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  3, 4>::value, vnode_base_offset_pair<1, 320,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  3, 5>::value, vnode_base_offset_pair<1, 320,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  3, 6>::value, vnode_base_offset_pair<1, 320,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  3, 7>::value, vnode_base_offset_pair<1, 320,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  3, 8>::value, vnode_base_offset_pair<1, 320,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  3, 9>::value, vnode_base_offset_pair<1, 320,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 320,  4, 0>::value, vnode_base_offset_pair<1, 320,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1, 320,  4, 1>::value, vnode_base_offset_pair<1, 320,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 320,  5, 0>::value, vnode_base_offset_pair<1, 320,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1, 320,  5, 1>::value, vnode_base_offset_pair<1, 320,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  5, 2>::value, vnode_base_offset_pair<1, 320,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  5, 3>::value, vnode_base_offset_pair<1, 320,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 320,  6, 0>::value, vnode_base_offset_pair<1, 320,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1, 320,  6, 1>::value, vnode_base_offset_pair<1, 320,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  6, 2>::value, vnode_base_offset_pair<1, 320,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  6, 3>::value, vnode_base_offset_pair<1, 320,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  6, 4>::value, vnode_base_offset_pair<1, 320,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 320,  7, 0>::value, vnode_base_offset_pair<1, 320,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1, 320,  7, 1>::value, vnode_base_offset_pair<1, 320,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  7, 2>::value, vnode_base_offset_pair<1, 320,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  7, 3>::value, vnode_base_offset_pair<1, 320,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 320,  8, 0>::value, vnode_base_offset_pair<1, 320,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1, 320,  8, 1>::value, vnode_base_offset_pair<1, 320,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  8, 2>::value, vnode_base_offset_pair<1, 320,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  8, 3>::value, vnode_base_offset_pair<1, 320,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  8, 4>::value, vnode_base_offset_pair<1, 320,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 320,  9, 0>::value, vnode_base_offset_pair<1, 320,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1, 320,  9, 1>::value, vnode_base_offset_pair<1, 320,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  9, 2>::value, vnode_base_offset_pair<1, 320,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  9, 3>::value, vnode_base_offset_pair<1, 320,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 320,  9, 4>::value, vnode_base_offset_pair<1, 320,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 10, 0>::value, vnode_base_offset_pair<1, 320, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1, 320, 10, 1>::value, vnode_base_offset_pair<1, 320, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 10, 2>::value, vnode_base_offset_pair<1, 320, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 10, 3>::value, vnode_base_offset_pair<1, 320, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 11, 0>::value, vnode_base_offset_pair<1, 320, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1, 320, 11, 1>::value, vnode_base_offset_pair<1, 320, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 11, 2>::value, vnode_base_offset_pair<1, 320, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 11, 3>::value, vnode_base_offset_pair<1, 320, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 12, 0>::value, vnode_base_offset_pair<1, 320, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1, 320, 12, 1>::value, vnode_base_offset_pair<1, 320, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 12, 2>::value, vnode_base_offset_pair<1, 320, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 12, 3>::value, vnode_base_offset_pair<1, 320, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 13, 0>::value, vnode_base_offset_pair<1, 320, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1, 320, 13, 1>::value, vnode_base_offset_pair<1, 320, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 13, 2>::value, vnode_base_offset_pair<1, 320, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 14, 0>::value, vnode_base_offset_pair<1, 320, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1, 320, 14, 1>::value, vnode_base_offset_pair<1, 320, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 14, 2>::value, vnode_base_offset_pair<1, 320, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 14, 3>::value, vnode_base_offset_pair<1, 320, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 15, 0>::value, vnode_base_offset_pair<1, 320, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1, 320, 15, 1>::value, vnode_base_offset_pair<1, 320, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 15, 2>::value, vnode_base_offset_pair<1, 320, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 15, 3>::value, vnode_base_offset_pair<1, 320, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 16, 0>::value, vnode_base_offset_pair<1, 320, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1, 320, 16, 1>::value, vnode_base_offset_pair<1, 320, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 16, 2>::value, vnode_base_offset_pair<1, 320, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 17, 0>::value, vnode_base_offset_pair<1, 320, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1, 320, 17, 1>::value, vnode_base_offset_pair<1, 320, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 17, 2>::value, vnode_base_offset_pair<1, 320, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 18, 0>::value, vnode_base_offset_pair<1, 320, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1, 320, 18, 1>::value, vnode_base_offset_pair<1, 320, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 18, 2>::value, vnode_base_offset_pair<1, 320, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 19, 0>::value, vnode_base_offset_pair<1, 320, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1, 320, 19, 1>::value, vnode_base_offset_pair<1, 320, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 19, 2>::value, vnode_base_offset_pair<1, 320, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 20, 0>::value, vnode_base_offset_pair<1, 320, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1, 320, 20, 1>::value, vnode_base_offset_pair<1, 320, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 20, 2>::value, vnode_base_offset_pair<1, 320, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 21, 0>::value, vnode_base_offset_pair<1, 320, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1, 320, 21, 1>::value, vnode_base_offset_pair<1, 320, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 21, 2>::value, vnode_base_offset_pair<1, 320, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 22, 0>::value, vnode_base_offset_pair<1, 320, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1, 320, 22, 1>::value, vnode_base_offset_pair<1, 320, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 22, 2>::value, vnode_base_offset_pair<1, 320, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 23, 0>::value, vnode_base_offset_pair<1, 320, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1, 320, 23, 1>::value, vnode_base_offset_pair<1, 320, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 23, 2>::value, vnode_base_offset_pair<1, 320, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 24, 0>::value, vnode_base_offset_pair<1, 320, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1, 320, 24, 1>::value, vnode_base_offset_pair<1, 320, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 24, 2>::value, vnode_base_offset_pair<1, 320, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 25, 0>::value, vnode_base_offset_pair<1, 320, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1, 320, 25, 1>::value, vnode_base_offset_pair<1, 320, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 25, 2>::value, vnode_base_offset_pair<1, 320, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 26, 0>::value, vnode_base_offset_pair<1, 320, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1, 320, 26, 1>::value, vnode_base_offset_pair<1, 320, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 26, 2>::value, vnode_base_offset_pair<1, 320, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 27, 0>::value, vnode_base_offset_pair<1, 320, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1, 320, 27, 1>::value, vnode_base_offset_pair<1, 320, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 28, 0>::value, vnode_base_offset_pair<1, 320, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1, 320, 28, 1>::value, vnode_base_offset_pair<1, 320, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 28, 2>::value, vnode_base_offset_pair<1, 320, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 29, 0>::value, vnode_base_offset_pair<1, 320, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1, 320, 29, 1>::value, vnode_base_offset_pair<1, 320, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 29, 2>::value, vnode_base_offset_pair<1, 320, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 30, 0>::value, vnode_base_offset_pair<1, 320, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1, 320, 30, 1>::value, vnode_base_offset_pair<1, 320, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 30, 2>::value, vnode_base_offset_pair<1, 320, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 31, 0>::value, vnode_base_offset_pair<1, 320, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1, 320, 31, 1>::value, vnode_base_offset_pair<1, 320, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 31, 2>::value, vnode_base_offset_pair<1, 320, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 32, 0>::value, vnode_base_offset_pair<1, 320, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1, 320, 32, 1>::value, vnode_base_offset_pair<1, 320, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 32, 2>::value, vnode_base_offset_pair<1, 320, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 33, 0>::value, vnode_base_offset_pair<1, 320, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1, 320, 33, 1>::value, vnode_base_offset_pair<1, 320, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 33, 2>::value, vnode_base_offset_pair<1, 320, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 34, 0>::value, vnode_base_offset_pair<1, 320, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1, 320, 34, 1>::value, vnode_base_offset_pair<1, 320, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 34, 2>::value, vnode_base_offset_pair<1, 320, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 35, 0>::value, vnode_base_offset_pair<1, 320, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1, 320, 35, 1>::value, vnode_base_offset_pair<1, 320, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 35, 2>::value, vnode_base_offset_pair<1, 320, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 36, 0>::value, vnode_base_offset_pair<1, 320, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1, 320, 36, 1>::value, vnode_base_offset_pair<1, 320, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 36, 2>::value, vnode_base_offset_pair<1, 320, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 37, 0>::value, vnode_base_offset_pair<1, 320, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1, 320, 37, 1>::value, vnode_base_offset_pair<1, 320, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 38, 0>::value, vnode_base_offset_pair<1, 320, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1, 320, 38, 1>::value, vnode_base_offset_pair<1, 320, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 38, 2>::value, vnode_base_offset_pair<1, 320, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 39, 0>::value, vnode_base_offset_pair<1, 320, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1, 320, 39, 1>::value, vnode_base_offset_pair<1, 320, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 39, 2>::value, vnode_base_offset_pair<1, 320, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 40, 0>::value, vnode_base_offset_pair<1, 320, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1, 320, 40, 1>::value, vnode_base_offset_pair<1, 320, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 41, 0>::value, vnode_base_offset_pair<1, 320, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1, 320, 41, 1>::value, vnode_base_offset_pair<1, 320, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 41, 2>::value, vnode_base_offset_pair<1, 320, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 42, 0>::value, vnode_base_offset_pair<1, 320, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1, 320, 42, 1>::value, vnode_base_offset_pair<1, 320, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 43, 0>::value, vnode_base_offset_pair<1, 320, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1, 320, 43, 1>::value, vnode_base_offset_pair<1, 320, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 43, 2>::value, vnode_base_offset_pair<1, 320, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 44, 0>::value, vnode_base_offset_pair<1, 320, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1, 320, 44, 1>::value, vnode_base_offset_pair<1, 320, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 320, 44, 2>::value, vnode_base_offset_pair<1, 320, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 320, 45, 0>::value, vnode_base_offset_pair<1, 320, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1, 320, 45, 1>::value, vnode_base_offset_pair<1, 320, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z352_8 =
{
    {
        { vnode_shift_mod_pair<1, 352,  0, 0>::value, vnode_base_offset_pair<1, 352,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1, 352,  0, 1>::value, vnode_base_offset_pair<1, 352,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  0, 2>::value, vnode_base_offset_pair<1, 352,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  0, 3>::value, vnode_base_offset_pair<1, 352,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  0, 4>::value, vnode_base_offset_pair<1, 352,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  0, 5>::value, vnode_base_offset_pair<1, 352,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  0, 6>::value, vnode_base_offset_pair<1, 352,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  0, 7>::value, vnode_base_offset_pair<1, 352,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  0, 8>::value, vnode_base_offset_pair<1, 352,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  0, 9>::value, vnode_base_offset_pair<1, 352,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 352,  1, 0>::value, vnode_base_offset_pair<1, 352,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1, 352,  1, 1>::value, vnode_base_offset_pair<1, 352,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  1, 2>::value, vnode_base_offset_pair<1, 352,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  1, 3>::value, vnode_base_offset_pair<1, 352,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  1, 4>::value, vnode_base_offset_pair<1, 352,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  1, 5>::value, vnode_base_offset_pair<1, 352,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  1, 6>::value, vnode_base_offset_pair<1, 352,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  1, 7>::value, vnode_base_offset_pair<1, 352,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  1, 8>::value, vnode_base_offset_pair<1, 352,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  1, 9>::value, vnode_base_offset_pair<1, 352,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 352,  2, 0>::value, vnode_base_offset_pair<1, 352,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1, 352,  2, 1>::value, vnode_base_offset_pair<1, 352,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  2, 2>::value, vnode_base_offset_pair<1, 352,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  2, 3>::value, vnode_base_offset_pair<1, 352,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  2, 4>::value, vnode_base_offset_pair<1, 352,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  2, 5>::value, vnode_base_offset_pair<1, 352,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  2, 6>::value, vnode_base_offset_pair<1, 352,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  2, 7>::value, vnode_base_offset_pair<1, 352,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  2, 8>::value, vnode_base_offset_pair<1, 352,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  2, 9>::value, vnode_base_offset_pair<1, 352,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 352,  3, 0>::value, vnode_base_offset_pair<1, 352,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1, 352,  3, 1>::value, vnode_base_offset_pair<1, 352,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  3, 2>::value, vnode_base_offset_pair<1, 352,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  3, 3>::value, vnode_base_offset_pair<1, 352,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  3, 4>::value, vnode_base_offset_pair<1, 352,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  3, 5>::value, vnode_base_offset_pair<1, 352,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  3, 6>::value, vnode_base_offset_pair<1, 352,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  3, 7>::value, vnode_base_offset_pair<1, 352,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  3, 8>::value, vnode_base_offset_pair<1, 352,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  3, 9>::value, vnode_base_offset_pair<1, 352,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 352,  4, 0>::value, vnode_base_offset_pair<1, 352,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1, 352,  4, 1>::value, vnode_base_offset_pair<1, 352,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 352,  5, 0>::value, vnode_base_offset_pair<1, 352,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1, 352,  5, 1>::value, vnode_base_offset_pair<1, 352,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  5, 2>::value, vnode_base_offset_pair<1, 352,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  5, 3>::value, vnode_base_offset_pair<1, 352,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 352,  6, 0>::value, vnode_base_offset_pair<1, 352,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1, 352,  6, 1>::value, vnode_base_offset_pair<1, 352,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  6, 2>::value, vnode_base_offset_pair<1, 352,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  6, 3>::value, vnode_base_offset_pair<1, 352,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  6, 4>::value, vnode_base_offset_pair<1, 352,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 352,  7, 0>::value, vnode_base_offset_pair<1, 352,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1, 352,  7, 1>::value, vnode_base_offset_pair<1, 352,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  7, 2>::value, vnode_base_offset_pair<1, 352,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  7, 3>::value, vnode_base_offset_pair<1, 352,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 352,  8, 0>::value, vnode_base_offset_pair<1, 352,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1, 352,  8, 1>::value, vnode_base_offset_pair<1, 352,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  8, 2>::value, vnode_base_offset_pair<1, 352,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  8, 3>::value, vnode_base_offset_pair<1, 352,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  8, 4>::value, vnode_base_offset_pair<1, 352,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 352,  9, 0>::value, vnode_base_offset_pair<1, 352,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1, 352,  9, 1>::value, vnode_base_offset_pair<1, 352,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  9, 2>::value, vnode_base_offset_pair<1, 352,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  9, 3>::value, vnode_base_offset_pair<1, 352,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 352,  9, 4>::value, vnode_base_offset_pair<1, 352,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 10, 0>::value, vnode_base_offset_pair<1, 352, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1, 352, 10, 1>::value, vnode_base_offset_pair<1, 352, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 10, 2>::value, vnode_base_offset_pair<1, 352, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 10, 3>::value, vnode_base_offset_pair<1, 352, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 11, 0>::value, vnode_base_offset_pair<1, 352, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1, 352, 11, 1>::value, vnode_base_offset_pair<1, 352, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 11, 2>::value, vnode_base_offset_pair<1, 352, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 11, 3>::value, vnode_base_offset_pair<1, 352, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 12, 0>::value, vnode_base_offset_pair<1, 352, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1, 352, 12, 1>::value, vnode_base_offset_pair<1, 352, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 12, 2>::value, vnode_base_offset_pair<1, 352, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 12, 3>::value, vnode_base_offset_pair<1, 352, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 13, 0>::value, vnode_base_offset_pair<1, 352, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1, 352, 13, 1>::value, vnode_base_offset_pair<1, 352, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 13, 2>::value, vnode_base_offset_pair<1, 352, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 14, 0>::value, vnode_base_offset_pair<1, 352, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1, 352, 14, 1>::value, vnode_base_offset_pair<1, 352, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 14, 2>::value, vnode_base_offset_pair<1, 352, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 14, 3>::value, vnode_base_offset_pair<1, 352, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 15, 0>::value, vnode_base_offset_pair<1, 352, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1, 352, 15, 1>::value, vnode_base_offset_pair<1, 352, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 15, 2>::value, vnode_base_offset_pair<1, 352, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 15, 3>::value, vnode_base_offset_pair<1, 352, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 16, 0>::value, vnode_base_offset_pair<1, 352, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1, 352, 16, 1>::value, vnode_base_offset_pair<1, 352, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 16, 2>::value, vnode_base_offset_pair<1, 352, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 17, 0>::value, vnode_base_offset_pair<1, 352, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1, 352, 17, 1>::value, vnode_base_offset_pair<1, 352, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 17, 2>::value, vnode_base_offset_pair<1, 352, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 18, 0>::value, vnode_base_offset_pair<1, 352, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1, 352, 18, 1>::value, vnode_base_offset_pair<1, 352, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 18, 2>::value, vnode_base_offset_pair<1, 352, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 19, 0>::value, vnode_base_offset_pair<1, 352, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1, 352, 19, 1>::value, vnode_base_offset_pair<1, 352, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 19, 2>::value, vnode_base_offset_pair<1, 352, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 20, 0>::value, vnode_base_offset_pair<1, 352, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1, 352, 20, 1>::value, vnode_base_offset_pair<1, 352, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 20, 2>::value, vnode_base_offset_pair<1, 352, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 21, 0>::value, vnode_base_offset_pair<1, 352, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1, 352, 21, 1>::value, vnode_base_offset_pair<1, 352, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 21, 2>::value, vnode_base_offset_pair<1, 352, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 22, 0>::value, vnode_base_offset_pair<1, 352, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1, 352, 22, 1>::value, vnode_base_offset_pair<1, 352, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 22, 2>::value, vnode_base_offset_pair<1, 352, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 23, 0>::value, vnode_base_offset_pair<1, 352, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1, 352, 23, 1>::value, vnode_base_offset_pair<1, 352, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 23, 2>::value, vnode_base_offset_pair<1, 352, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 24, 0>::value, vnode_base_offset_pair<1, 352, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1, 352, 24, 1>::value, vnode_base_offset_pair<1, 352, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 24, 2>::value, vnode_base_offset_pair<1, 352, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 25, 0>::value, vnode_base_offset_pair<1, 352, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1, 352, 25, 1>::value, vnode_base_offset_pair<1, 352, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 25, 2>::value, vnode_base_offset_pair<1, 352, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 26, 0>::value, vnode_base_offset_pair<1, 352, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1, 352, 26, 1>::value, vnode_base_offset_pair<1, 352, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 26, 2>::value, vnode_base_offset_pair<1, 352, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 27, 0>::value, vnode_base_offset_pair<1, 352, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1, 352, 27, 1>::value, vnode_base_offset_pair<1, 352, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 28, 0>::value, vnode_base_offset_pair<1, 352, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1, 352, 28, 1>::value, vnode_base_offset_pair<1, 352, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 28, 2>::value, vnode_base_offset_pair<1, 352, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 29, 0>::value, vnode_base_offset_pair<1, 352, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1, 352, 29, 1>::value, vnode_base_offset_pair<1, 352, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 29, 2>::value, vnode_base_offset_pair<1, 352, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 30, 0>::value, vnode_base_offset_pair<1, 352, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1, 352, 30, 1>::value, vnode_base_offset_pair<1, 352, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 30, 2>::value, vnode_base_offset_pair<1, 352, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 31, 0>::value, vnode_base_offset_pair<1, 352, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1, 352, 31, 1>::value, vnode_base_offset_pair<1, 352, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 31, 2>::value, vnode_base_offset_pair<1, 352, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 32, 0>::value, vnode_base_offset_pair<1, 352, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1, 352, 32, 1>::value, vnode_base_offset_pair<1, 352, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 32, 2>::value, vnode_base_offset_pair<1, 352, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 33, 0>::value, vnode_base_offset_pair<1, 352, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1, 352, 33, 1>::value, vnode_base_offset_pair<1, 352, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 33, 2>::value, vnode_base_offset_pair<1, 352, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 34, 0>::value, vnode_base_offset_pair<1, 352, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1, 352, 34, 1>::value, vnode_base_offset_pair<1, 352, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 34, 2>::value, vnode_base_offset_pair<1, 352, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 35, 0>::value, vnode_base_offset_pair<1, 352, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1, 352, 35, 1>::value, vnode_base_offset_pair<1, 352, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 35, 2>::value, vnode_base_offset_pair<1, 352, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 36, 0>::value, vnode_base_offset_pair<1, 352, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1, 352, 36, 1>::value, vnode_base_offset_pair<1, 352, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 36, 2>::value, vnode_base_offset_pair<1, 352, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 37, 0>::value, vnode_base_offset_pair<1, 352, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1, 352, 37, 1>::value, vnode_base_offset_pair<1, 352, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 38, 0>::value, vnode_base_offset_pair<1, 352, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1, 352, 38, 1>::value, vnode_base_offset_pair<1, 352, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 38, 2>::value, vnode_base_offset_pair<1, 352, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 39, 0>::value, vnode_base_offset_pair<1, 352, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1, 352, 39, 1>::value, vnode_base_offset_pair<1, 352, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 39, 2>::value, vnode_base_offset_pair<1, 352, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 40, 0>::value, vnode_base_offset_pair<1, 352, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1, 352, 40, 1>::value, vnode_base_offset_pair<1, 352, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 41, 0>::value, vnode_base_offset_pair<1, 352, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1, 352, 41, 1>::value, vnode_base_offset_pair<1, 352, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 41, 2>::value, vnode_base_offset_pair<1, 352, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 42, 0>::value, vnode_base_offset_pair<1, 352, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1, 352, 42, 1>::value, vnode_base_offset_pair<1, 352, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 43, 0>::value, vnode_base_offset_pair<1, 352, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1, 352, 43, 1>::value, vnode_base_offset_pair<1, 352, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 43, 2>::value, vnode_base_offset_pair<1, 352, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 44, 0>::value, vnode_base_offset_pair<1, 352, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1, 352, 44, 1>::value, vnode_base_offset_pair<1, 352, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 352, 44, 2>::value, vnode_base_offset_pair<1, 352, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 352, 45, 0>::value, vnode_base_offset_pair<1, 352, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1, 352, 45, 1>::value, vnode_base_offset_pair<1, 352, 45, 1>::value * 1 }
    }
};

const BG1_desc_t BG1_desc_Z384_8 =
{
    {
        { vnode_shift_mod_pair<1, 384,  0, 0>::value, vnode_base_offset_pair<1, 384,  0, 0>::value * 1 }, // Row 0, degree = 19
        { vnode_shift_mod_pair<1, 384,  0, 1>::value, vnode_base_offset_pair<1, 384,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  0, 2>::value, vnode_base_offset_pair<1, 384,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  0, 3>::value, vnode_base_offset_pair<1, 384,  0, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  0, 4>::value, vnode_base_offset_pair<1, 384,  0, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  0, 5>::value, vnode_base_offset_pair<1, 384,  0, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  0, 6>::value, vnode_base_offset_pair<1, 384,  0, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  0, 7>::value, vnode_base_offset_pair<1, 384,  0, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  0, 8>::value, vnode_base_offset_pair<1, 384,  0, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  0, 9>::value, vnode_base_offset_pair<1, 384,  0, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 384,  1, 0>::value, vnode_base_offset_pair<1, 384,  1, 0>::value * 1 }, // Row 1, degree = 19
        { vnode_shift_mod_pair<1, 384,  1, 1>::value, vnode_base_offset_pair<1, 384,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  1, 2>::value, vnode_base_offset_pair<1, 384,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  1, 3>::value, vnode_base_offset_pair<1, 384,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  1, 4>::value, vnode_base_offset_pair<1, 384,  1, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  1, 5>::value, vnode_base_offset_pair<1, 384,  1, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  1, 6>::value, vnode_base_offset_pair<1, 384,  1, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  1, 7>::value, vnode_base_offset_pair<1, 384,  1, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  1, 8>::value, vnode_base_offset_pair<1, 384,  1, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  1, 9>::value, vnode_base_offset_pair<1, 384,  1, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 384,  2, 0>::value, vnode_base_offset_pair<1, 384,  2, 0>::value * 1 }, // Row 2, degree = 19
        { vnode_shift_mod_pair<1, 384,  2, 1>::value, vnode_base_offset_pair<1, 384,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  2, 2>::value, vnode_base_offset_pair<1, 384,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  2, 3>::value, vnode_base_offset_pair<1, 384,  2, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  2, 4>::value, vnode_base_offset_pair<1, 384,  2, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  2, 5>::value, vnode_base_offset_pair<1, 384,  2, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  2, 6>::value, vnode_base_offset_pair<1, 384,  2, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  2, 7>::value, vnode_base_offset_pair<1, 384,  2, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  2, 8>::value, vnode_base_offset_pair<1, 384,  2, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  2, 9>::value, vnode_base_offset_pair<1, 384,  2, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 384,  3, 0>::value, vnode_base_offset_pair<1, 384,  3, 0>::value * 1 }, // Row 3, degree = 19
        { vnode_shift_mod_pair<1, 384,  3, 1>::value, vnode_base_offset_pair<1, 384,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  3, 2>::value, vnode_base_offset_pair<1, 384,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  3, 3>::value, vnode_base_offset_pair<1, 384,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  3, 4>::value, vnode_base_offset_pair<1, 384,  3, 4>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  3, 5>::value, vnode_base_offset_pair<1, 384,  3, 5>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  3, 6>::value, vnode_base_offset_pair<1, 384,  3, 6>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  3, 7>::value, vnode_base_offset_pair<1, 384,  3, 7>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  3, 8>::value, vnode_base_offset_pair<1, 384,  3, 8>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  3, 9>::value, vnode_base_offset_pair<1, 384,  3, 9>::value * 1 },

        { vnode_shift_mod_pair<1, 384,  4, 0>::value, vnode_base_offset_pair<1, 384,  4, 0>::value * 1 }, // Row 4, degree = 3
        { vnode_shift_mod_pair<1, 384,  4, 1>::value, vnode_base_offset_pair<1, 384,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 384,  5, 0>::value, vnode_base_offset_pair<1, 384,  5, 0>::value * 1 }, // Row 5, degree = 8
        { vnode_shift_mod_pair<1, 384,  5, 1>::value, vnode_base_offset_pair<1, 384,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  5, 2>::value, vnode_base_offset_pair<1, 384,  5, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  5, 3>::value, vnode_base_offset_pair<1, 384,  5, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 384,  6, 0>::value, vnode_base_offset_pair<1, 384,  6, 0>::value * 1 }, // Row 6, degree = 9
        { vnode_shift_mod_pair<1, 384,  6, 1>::value, vnode_base_offset_pair<1, 384,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  6, 2>::value, vnode_base_offset_pair<1, 384,  6, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  6, 3>::value, vnode_base_offset_pair<1, 384,  6, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  6, 4>::value, vnode_base_offset_pair<1, 384,  6, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 384,  7, 0>::value, vnode_base_offset_pair<1, 384,  7, 0>::value * 1 }, // Row 7, degree = 7
        { vnode_shift_mod_pair<1, 384,  7, 1>::value, vnode_base_offset_pair<1, 384,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  7, 2>::value, vnode_base_offset_pair<1, 384,  7, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  7, 3>::value, vnode_base_offset_pair<1, 384,  7, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 384,  8, 0>::value, vnode_base_offset_pair<1, 384,  8, 0>::value * 1 }, // Row 8, degree = 10
        { vnode_shift_mod_pair<1, 384,  8, 1>::value, vnode_base_offset_pair<1, 384,  8, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  8, 2>::value, vnode_base_offset_pair<1, 384,  8, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  8, 3>::value, vnode_base_offset_pair<1, 384,  8, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  8, 4>::value, vnode_base_offset_pair<1, 384,  8, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 384,  9, 0>::value, vnode_base_offset_pair<1, 384,  9, 0>::value * 1 }, // Row 9, degree = 9
        { vnode_shift_mod_pair<1, 384,  9, 1>::value, vnode_base_offset_pair<1, 384,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  9, 2>::value, vnode_base_offset_pair<1, 384,  9, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  9, 3>::value, vnode_base_offset_pair<1, 384,  9, 3>::value * 1 },
        { vnode_shift_mod_pair<1, 384,  9, 4>::value, vnode_base_offset_pair<1, 384,  9, 4>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 10, 0>::value, vnode_base_offset_pair<1, 384, 10, 0>::value * 1 }, // Row 10, degree = 7
        { vnode_shift_mod_pair<1, 384, 10, 1>::value, vnode_base_offset_pair<1, 384, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 10, 2>::value, vnode_base_offset_pair<1, 384, 10, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 10, 3>::value, vnode_base_offset_pair<1, 384, 10, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 11, 0>::value, vnode_base_offset_pair<1, 384, 11, 0>::value * 1 }, // Row 11, degree = 8
        { vnode_shift_mod_pair<1, 384, 11, 1>::value, vnode_base_offset_pair<1, 384, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 11, 2>::value, vnode_base_offset_pair<1, 384, 11, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 11, 3>::value, vnode_base_offset_pair<1, 384, 11, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 12, 0>::value, vnode_base_offset_pair<1, 384, 12, 0>::value * 1 }, // Row 12, degree = 7
        { vnode_shift_mod_pair<1, 384, 12, 1>::value, vnode_base_offset_pair<1, 384, 12, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 12, 2>::value, vnode_base_offset_pair<1, 384, 12, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 12, 3>::value, vnode_base_offset_pair<1, 384, 12, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 13, 0>::value, vnode_base_offset_pair<1, 384, 13, 0>::value * 1 }, // Row 13, degree = 6
        { vnode_shift_mod_pair<1, 384, 13, 1>::value, vnode_base_offset_pair<1, 384, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 13, 2>::value, vnode_base_offset_pair<1, 384, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 14, 0>::value, vnode_base_offset_pair<1, 384, 14, 0>::value * 1 }, // Row 14, degree = 7
        { vnode_shift_mod_pair<1, 384, 14, 1>::value, vnode_base_offset_pair<1, 384, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 14, 2>::value, vnode_base_offset_pair<1, 384, 14, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 14, 3>::value, vnode_base_offset_pair<1, 384, 14, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 15, 0>::value, vnode_base_offset_pair<1, 384, 15, 0>::value * 1 }, // Row 15, degree = 7
        { vnode_shift_mod_pair<1, 384, 15, 1>::value, vnode_base_offset_pair<1, 384, 15, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 15, 2>::value, vnode_base_offset_pair<1, 384, 15, 2>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 15, 3>::value, vnode_base_offset_pair<1, 384, 15, 3>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 16, 0>::value, vnode_base_offset_pair<1, 384, 16, 0>::value * 1 }, // Row 16, degree = 6
        { vnode_shift_mod_pair<1, 384, 16, 1>::value, vnode_base_offset_pair<1, 384, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 16, 2>::value, vnode_base_offset_pair<1, 384, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 17, 0>::value, vnode_base_offset_pair<1, 384, 17, 0>::value * 1 }, // Row 17, degree = 6
        { vnode_shift_mod_pair<1, 384, 17, 1>::value, vnode_base_offset_pair<1, 384, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 17, 2>::value, vnode_base_offset_pair<1, 384, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 18, 0>::value, vnode_base_offset_pair<1, 384, 18, 0>::value * 1 }, // Row 18, degree = 6
        { vnode_shift_mod_pair<1, 384, 18, 1>::value, vnode_base_offset_pair<1, 384, 18, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 18, 2>::value, vnode_base_offset_pair<1, 384, 18, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 19, 0>::value, vnode_base_offset_pair<1, 384, 19, 0>::value * 1 }, // Row 19, degree = 6
        { vnode_shift_mod_pair<1, 384, 19, 1>::value, vnode_base_offset_pair<1, 384, 19, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 19, 2>::value, vnode_base_offset_pair<1, 384, 19, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 20, 0>::value, vnode_base_offset_pair<1, 384, 20, 0>::value * 1 }, // Row 20, degree = 6
        { vnode_shift_mod_pair<1, 384, 20, 1>::value, vnode_base_offset_pair<1, 384, 20, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 20, 2>::value, vnode_base_offset_pair<1, 384, 20, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 21, 0>::value, vnode_base_offset_pair<1, 384, 21, 0>::value * 1 }, // Row 21, degree = 6
        { vnode_shift_mod_pair<1, 384, 21, 1>::value, vnode_base_offset_pair<1, 384, 21, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 21, 2>::value, vnode_base_offset_pair<1, 384, 21, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 22, 0>::value, vnode_base_offset_pair<1, 384, 22, 0>::value * 1 }, // Row 22, degree = 5
        { vnode_shift_mod_pair<1, 384, 22, 1>::value, vnode_base_offset_pair<1, 384, 22, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 22, 2>::value, vnode_base_offset_pair<1, 384, 22, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 23, 0>::value, vnode_base_offset_pair<1, 384, 23, 0>::value * 1 }, // Row 23, degree = 5
        { vnode_shift_mod_pair<1, 384, 23, 1>::value, vnode_base_offset_pair<1, 384, 23, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 23, 2>::value, vnode_base_offset_pair<1, 384, 23, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 24, 0>::value, vnode_base_offset_pair<1, 384, 24, 0>::value * 1 }, // Row 24, degree = 6
        { vnode_shift_mod_pair<1, 384, 24, 1>::value, vnode_base_offset_pair<1, 384, 24, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 24, 2>::value, vnode_base_offset_pair<1, 384, 24, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 25, 0>::value, vnode_base_offset_pair<1, 384, 25, 0>::value * 1 }, // Row 25, degree = 5
        { vnode_shift_mod_pair<1, 384, 25, 1>::value, vnode_base_offset_pair<1, 384, 25, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 25, 2>::value, vnode_base_offset_pair<1, 384, 25, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 26, 0>::value, vnode_base_offset_pair<1, 384, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<1, 384, 26, 1>::value, vnode_base_offset_pair<1, 384, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 26, 2>::value, vnode_base_offset_pair<1, 384, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 27, 0>::value, vnode_base_offset_pair<1, 384, 27, 0>::value * 1 }, // Row 27, degree = 4
        { vnode_shift_mod_pair<1, 384, 27, 1>::value, vnode_base_offset_pair<1, 384, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 28, 0>::value, vnode_base_offset_pair<1, 384, 28, 0>::value * 1 }, // Row 28, degree = 5
        { vnode_shift_mod_pair<1, 384, 28, 1>::value, vnode_base_offset_pair<1, 384, 28, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 28, 2>::value, vnode_base_offset_pair<1, 384, 28, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 29, 0>::value, vnode_base_offset_pair<1, 384, 29, 0>::value * 1 }, // Row 29, degree = 5
        { vnode_shift_mod_pair<1, 384, 29, 1>::value, vnode_base_offset_pair<1, 384, 29, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 29, 2>::value, vnode_base_offset_pair<1, 384, 29, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 30, 0>::value, vnode_base_offset_pair<1, 384, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<1, 384, 30, 1>::value, vnode_base_offset_pair<1, 384, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 30, 2>::value, vnode_base_offset_pair<1, 384, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 31, 0>::value, vnode_base_offset_pair<1, 384, 31, 0>::value * 1 }, // Row 31, degree = 5
        { vnode_shift_mod_pair<1, 384, 31, 1>::value, vnode_base_offset_pair<1, 384, 31, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 31, 2>::value, vnode_base_offset_pair<1, 384, 31, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 32, 0>::value, vnode_base_offset_pair<1, 384, 32, 0>::value * 1 }, // Row 32, degree = 5
        { vnode_shift_mod_pair<1, 384, 32, 1>::value, vnode_base_offset_pair<1, 384, 32, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 32, 2>::value, vnode_base_offset_pair<1, 384, 32, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 33, 0>::value, vnode_base_offset_pair<1, 384, 33, 0>::value * 1 }, // Row 33, degree = 5
        { vnode_shift_mod_pair<1, 384, 33, 1>::value, vnode_base_offset_pair<1, 384, 33, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 33, 2>::value, vnode_base_offset_pair<1, 384, 33, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 34, 0>::value, vnode_base_offset_pair<1, 384, 34, 0>::value * 1 }, // Row 34, degree = 5
        { vnode_shift_mod_pair<1, 384, 34, 1>::value, vnode_base_offset_pair<1, 384, 34, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 34, 2>::value, vnode_base_offset_pair<1, 384, 34, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 35, 0>::value, vnode_base_offset_pair<1, 384, 35, 0>::value * 1 }, // Row 35, degree = 5
        { vnode_shift_mod_pair<1, 384, 35, 1>::value, vnode_base_offset_pair<1, 384, 35, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 35, 2>::value, vnode_base_offset_pair<1, 384, 35, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 36, 0>::value, vnode_base_offset_pair<1, 384, 36, 0>::value * 1 }, // Row 36, degree = 5
        { vnode_shift_mod_pair<1, 384, 36, 1>::value, vnode_base_offset_pair<1, 384, 36, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 36, 2>::value, vnode_base_offset_pair<1, 384, 36, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 37, 0>::value, vnode_base_offset_pair<1, 384, 37, 0>::value * 1 }, // Row 37, degree = 4
        { vnode_shift_mod_pair<1, 384, 37, 1>::value, vnode_base_offset_pair<1, 384, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 38, 0>::value, vnode_base_offset_pair<1, 384, 38, 0>::value * 1 }, // Row 38, degree = 5
        { vnode_shift_mod_pair<1, 384, 38, 1>::value, vnode_base_offset_pair<1, 384, 38, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 38, 2>::value, vnode_base_offset_pair<1, 384, 38, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 39, 0>::value, vnode_base_offset_pair<1, 384, 39, 0>::value * 1 }, // Row 39, degree = 5
        { vnode_shift_mod_pair<1, 384, 39, 1>::value, vnode_base_offset_pair<1, 384, 39, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 39, 2>::value, vnode_base_offset_pair<1, 384, 39, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 40, 0>::value, vnode_base_offset_pair<1, 384, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<1, 384, 40, 1>::value, vnode_base_offset_pair<1, 384, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 41, 0>::value, vnode_base_offset_pair<1, 384, 41, 0>::value * 1 }, // Row 41, degree = 5
        { vnode_shift_mod_pair<1, 384, 41, 1>::value, vnode_base_offset_pair<1, 384, 41, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 41, 2>::value, vnode_base_offset_pair<1, 384, 41, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 42, 0>::value, vnode_base_offset_pair<1, 384, 42, 0>::value * 1 }, // Row 42, degree = 4
        { vnode_shift_mod_pair<1, 384, 42, 1>::value, vnode_base_offset_pair<1, 384, 42, 1>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 43, 0>::value, vnode_base_offset_pair<1, 384, 43, 0>::value * 1 }, // Row 43, degree = 5
        { vnode_shift_mod_pair<1, 384, 43, 1>::value, vnode_base_offset_pair<1, 384, 43, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 43, 2>::value, vnode_base_offset_pair<1, 384, 43, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 44, 0>::value, vnode_base_offset_pair<1, 384, 44, 0>::value * 1 }, // Row 44, degree = 5
        { vnode_shift_mod_pair<1, 384, 44, 1>::value, vnode_base_offset_pair<1, 384, 44, 1>::value * 1 },
        { vnode_shift_mod_pair<1, 384, 44, 2>::value, vnode_base_offset_pair<1, 384, 44, 2>::value * 1 },

        { vnode_shift_mod_pair<1, 384, 45, 0>::value, vnode_base_offset_pair<1, 384, 45, 0>::value * 1 }, // Row 45, degree = 4
        { vnode_shift_mod_pair<1, 384, 45, 1>::value, vnode_base_offset_pair<1, 384, 45, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z32_8 =
{
    {
        { vnode_shift_mod_pair<2,  32,  0, 0>::value, vnode_base_offset_pair<2,  32,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2,  32,  0, 1>::value, vnode_base_offset_pair<2,  32,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  32,  0, 2>::value, vnode_base_offset_pair<2,  32,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  32,  0, 3>::value, vnode_base_offset_pair<2,  32,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  32,  1, 0>::value, vnode_base_offset_pair<2,  32,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2,  32,  1, 1>::value, vnode_base_offset_pair<2,  32,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  32,  1, 2>::value, vnode_base_offset_pair<2,  32,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  32,  1, 3>::value, vnode_base_offset_pair<2,  32,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  32,  1, 4>::value, vnode_base_offset_pair<2,  32,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  32,  2, 0>::value, vnode_base_offset_pair<2,  32,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2,  32,  2, 1>::value, vnode_base_offset_pair<2,  32,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  32,  2, 2>::value, vnode_base_offset_pair<2,  32,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  32,  2, 3>::value, vnode_base_offset_pair<2,  32,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  32,  3, 0>::value, vnode_base_offset_pair<2,  32,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2,  32,  3, 1>::value, vnode_base_offset_pair<2,  32,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  32,  3, 2>::value, vnode_base_offset_pair<2,  32,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  32,  3, 3>::value, vnode_base_offset_pair<2,  32,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  32,  3, 4>::value, vnode_base_offset_pair<2,  32,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  32,  4, 0>::value, vnode_base_offset_pair<2,  32,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2,  32,  4, 1>::value, vnode_base_offset_pair<2,  32,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32,  5, 0>::value, vnode_base_offset_pair<2,  32,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2,  32,  5, 1>::value, vnode_base_offset_pair<2,  32,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  32,  5, 2>::value, vnode_base_offset_pair<2,  32,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  32,  6, 0>::value, vnode_base_offset_pair<2,  32,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2,  32,  6, 1>::value, vnode_base_offset_pair<2,  32,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  32,  6, 2>::value, vnode_base_offset_pair<2,  32,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  32,  7, 0>::value, vnode_base_offset_pair<2,  32,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2,  32,  7, 1>::value, vnode_base_offset_pair<2,  32,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  32,  7, 2>::value, vnode_base_offset_pair<2,  32,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  32,  8, 0>::value, vnode_base_offset_pair<2,  32,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2,  32,  8, 1>::value, vnode_base_offset_pair<2,  32,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32,  9, 0>::value, vnode_base_offset_pair<2,  32,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2,  32,  9, 1>::value, vnode_base_offset_pair<2,  32,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  32,  9, 2>::value, vnode_base_offset_pair<2,  32,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 10, 0>::value, vnode_base_offset_pair<2,  32, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2,  32, 10, 1>::value, vnode_base_offset_pair<2,  32, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  32, 10, 2>::value, vnode_base_offset_pair<2,  32, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 11, 0>::value, vnode_base_offset_pair<2,  32, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2,  32, 11, 1>::value, vnode_base_offset_pair<2,  32, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  32, 11, 2>::value, vnode_base_offset_pair<2,  32, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 12, 0>::value, vnode_base_offset_pair<2,  32, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2,  32, 12, 1>::value, vnode_base_offset_pair<2,  32, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 13, 0>::value, vnode_base_offset_pair<2,  32, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2,  32, 13, 1>::value, vnode_base_offset_pair<2,  32, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  32, 13, 2>::value, vnode_base_offset_pair<2,  32, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 14, 0>::value, vnode_base_offset_pair<2,  32, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2,  32, 14, 1>::value, vnode_base_offset_pair<2,  32, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  32, 14, 2>::value, vnode_base_offset_pair<2,  32, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 15, 0>::value, vnode_base_offset_pair<2,  32, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2,  32, 15, 1>::value, vnode_base_offset_pair<2,  32, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 16, 0>::value, vnode_base_offset_pair<2,  32, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2,  32, 16, 1>::value, vnode_base_offset_pair<2,  32, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  32, 16, 2>::value, vnode_base_offset_pair<2,  32, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 17, 0>::value, vnode_base_offset_pair<2,  32, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2,  32, 17, 1>::value, vnode_base_offset_pair<2,  32, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  32, 17, 2>::value, vnode_base_offset_pair<2,  32, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 18, 0>::value, vnode_base_offset_pair<2,  32, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2,  32, 18, 1>::value, vnode_base_offset_pair<2,  32, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 19, 0>::value, vnode_base_offset_pair<2,  32, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2,  32, 19, 1>::value, vnode_base_offset_pair<2,  32, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 20, 0>::value, vnode_base_offset_pair<2,  32, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2,  32, 20, 1>::value, vnode_base_offset_pair<2,  32, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 21, 0>::value, vnode_base_offset_pair<2,  32, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2,  32, 21, 1>::value, vnode_base_offset_pair<2,  32, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 22, 0>::value, vnode_base_offset_pair<2,  32, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2,  32, 22, 1>::value, vnode_base_offset_pair<2,  32, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 23, 0>::value, vnode_base_offset_pair<2,  32, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2,  32, 23, 1>::value, vnode_base_offset_pair<2,  32, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 24, 0>::value, vnode_base_offset_pair<2,  32, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2,  32, 24, 1>::value, vnode_base_offset_pair<2,  32, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 25, 0>::value, vnode_base_offset_pair<2,  32, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2,  32, 25, 1>::value, vnode_base_offset_pair<2,  32, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 26, 0>::value, vnode_base_offset_pair<2,  32, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2,  32, 26, 1>::value, vnode_base_offset_pair<2,  32, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  32, 26, 2>::value, vnode_base_offset_pair<2,  32, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 27, 0>::value, vnode_base_offset_pair<2,  32, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2,  32, 27, 1>::value, vnode_base_offset_pair<2,  32, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 28, 0>::value, vnode_base_offset_pair<2,  32, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2,  32, 28, 1>::value, vnode_base_offset_pair<2,  32, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 29, 0>::value, vnode_base_offset_pair<2,  32, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2,  32, 29, 1>::value, vnode_base_offset_pair<2,  32, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 30, 0>::value, vnode_base_offset_pair<2,  32, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2,  32, 30, 1>::value, vnode_base_offset_pair<2,  32, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  32, 30, 2>::value, vnode_base_offset_pair<2,  32, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 31, 0>::value, vnode_base_offset_pair<2,  32, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2,  32, 31, 1>::value, vnode_base_offset_pair<2,  32, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 32, 0>::value, vnode_base_offset_pair<2,  32, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2,  32, 32, 1>::value, vnode_base_offset_pair<2,  32, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 33, 0>::value, vnode_base_offset_pair<2,  32, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2,  32, 33, 1>::value, vnode_base_offset_pair<2,  32, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 34, 0>::value, vnode_base_offset_pair<2,  32, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2,  32, 34, 1>::value, vnode_base_offset_pair<2,  32, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 35, 0>::value, vnode_base_offset_pair<2,  32, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2,  32, 35, 1>::value, vnode_base_offset_pair<2,  32, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 36, 0>::value, vnode_base_offset_pair<2,  32, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2,  32, 36, 1>::value, vnode_base_offset_pair<2,  32, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 37, 0>::value, vnode_base_offset_pair<2,  32, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2,  32, 37, 1>::value, vnode_base_offset_pair<2,  32, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 38, 0>::value, vnode_base_offset_pair<2,  32, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2,  32, 38, 1>::value, vnode_base_offset_pair<2,  32, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 39, 0>::value, vnode_base_offset_pair<2,  32, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2,  32, 39, 1>::value, vnode_base_offset_pair<2,  32, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 40, 0>::value, vnode_base_offset_pair<2,  32, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2,  32, 40, 1>::value, vnode_base_offset_pair<2,  32, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  32, 41, 0>::value, vnode_base_offset_pair<2,  32, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2,  32, 41, 1>::value, vnode_base_offset_pair<2,  32, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z36_8 =
{
    {
        { vnode_shift_mod_pair<2,  36,  0, 0>::value, vnode_base_offset_pair<2,  36,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2,  36,  0, 1>::value, vnode_base_offset_pair<2,  36,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  36,  0, 2>::value, vnode_base_offset_pair<2,  36,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  36,  0, 3>::value, vnode_base_offset_pair<2,  36,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  36,  1, 0>::value, vnode_base_offset_pair<2,  36,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2,  36,  1, 1>::value, vnode_base_offset_pair<2,  36,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  36,  1, 2>::value, vnode_base_offset_pair<2,  36,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  36,  1, 3>::value, vnode_base_offset_pair<2,  36,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  36,  1, 4>::value, vnode_base_offset_pair<2,  36,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  36,  2, 0>::value, vnode_base_offset_pair<2,  36,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2,  36,  2, 1>::value, vnode_base_offset_pair<2,  36,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  36,  2, 2>::value, vnode_base_offset_pair<2,  36,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  36,  2, 3>::value, vnode_base_offset_pair<2,  36,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  36,  3, 0>::value, vnode_base_offset_pair<2,  36,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2,  36,  3, 1>::value, vnode_base_offset_pair<2,  36,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  36,  3, 2>::value, vnode_base_offset_pair<2,  36,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  36,  3, 3>::value, vnode_base_offset_pair<2,  36,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  36,  3, 4>::value, vnode_base_offset_pair<2,  36,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  36,  4, 0>::value, vnode_base_offset_pair<2,  36,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2,  36,  4, 1>::value, vnode_base_offset_pair<2,  36,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36,  5, 0>::value, vnode_base_offset_pair<2,  36,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2,  36,  5, 1>::value, vnode_base_offset_pair<2,  36,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  36,  5, 2>::value, vnode_base_offset_pair<2,  36,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  36,  6, 0>::value, vnode_base_offset_pair<2,  36,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2,  36,  6, 1>::value, vnode_base_offset_pair<2,  36,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  36,  6, 2>::value, vnode_base_offset_pair<2,  36,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  36,  7, 0>::value, vnode_base_offset_pair<2,  36,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2,  36,  7, 1>::value, vnode_base_offset_pair<2,  36,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  36,  7, 2>::value, vnode_base_offset_pair<2,  36,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  36,  8, 0>::value, vnode_base_offset_pair<2,  36,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2,  36,  8, 1>::value, vnode_base_offset_pair<2,  36,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36,  9, 0>::value, vnode_base_offset_pair<2,  36,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2,  36,  9, 1>::value, vnode_base_offset_pair<2,  36,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  36,  9, 2>::value, vnode_base_offset_pair<2,  36,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 10, 0>::value, vnode_base_offset_pair<2,  36, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2,  36, 10, 1>::value, vnode_base_offset_pair<2,  36, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  36, 10, 2>::value, vnode_base_offset_pair<2,  36, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 11, 0>::value, vnode_base_offset_pair<2,  36, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2,  36, 11, 1>::value, vnode_base_offset_pair<2,  36, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  36, 11, 2>::value, vnode_base_offset_pair<2,  36, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 12, 0>::value, vnode_base_offset_pair<2,  36, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2,  36, 12, 1>::value, vnode_base_offset_pair<2,  36, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 13, 0>::value, vnode_base_offset_pair<2,  36, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2,  36, 13, 1>::value, vnode_base_offset_pair<2,  36, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  36, 13, 2>::value, vnode_base_offset_pair<2,  36, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 14, 0>::value, vnode_base_offset_pair<2,  36, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2,  36, 14, 1>::value, vnode_base_offset_pair<2,  36, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  36, 14, 2>::value, vnode_base_offset_pair<2,  36, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 15, 0>::value, vnode_base_offset_pair<2,  36, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2,  36, 15, 1>::value, vnode_base_offset_pair<2,  36, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 16, 0>::value, vnode_base_offset_pair<2,  36, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2,  36, 16, 1>::value, vnode_base_offset_pair<2,  36, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  36, 16, 2>::value, vnode_base_offset_pair<2,  36, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 17, 0>::value, vnode_base_offset_pair<2,  36, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2,  36, 17, 1>::value, vnode_base_offset_pair<2,  36, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  36, 17, 2>::value, vnode_base_offset_pair<2,  36, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 18, 0>::value, vnode_base_offset_pair<2,  36, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2,  36, 18, 1>::value, vnode_base_offset_pair<2,  36, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 19, 0>::value, vnode_base_offset_pair<2,  36, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2,  36, 19, 1>::value, vnode_base_offset_pair<2,  36, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 20, 0>::value, vnode_base_offset_pair<2,  36, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2,  36, 20, 1>::value, vnode_base_offset_pair<2,  36, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 21, 0>::value, vnode_base_offset_pair<2,  36, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2,  36, 21, 1>::value, vnode_base_offset_pair<2,  36, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 22, 0>::value, vnode_base_offset_pair<2,  36, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2,  36, 22, 1>::value, vnode_base_offset_pair<2,  36, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 23, 0>::value, vnode_base_offset_pair<2,  36, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2,  36, 23, 1>::value, vnode_base_offset_pair<2,  36, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 24, 0>::value, vnode_base_offset_pair<2,  36, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2,  36, 24, 1>::value, vnode_base_offset_pair<2,  36, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 25, 0>::value, vnode_base_offset_pair<2,  36, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2,  36, 25, 1>::value, vnode_base_offset_pair<2,  36, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 26, 0>::value, vnode_base_offset_pair<2,  36, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2,  36, 26, 1>::value, vnode_base_offset_pair<2,  36, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  36, 26, 2>::value, vnode_base_offset_pair<2,  36, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 27, 0>::value, vnode_base_offset_pair<2,  36, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2,  36, 27, 1>::value, vnode_base_offset_pair<2,  36, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 28, 0>::value, vnode_base_offset_pair<2,  36, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2,  36, 28, 1>::value, vnode_base_offset_pair<2,  36, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 29, 0>::value, vnode_base_offset_pair<2,  36, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2,  36, 29, 1>::value, vnode_base_offset_pair<2,  36, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 30, 0>::value, vnode_base_offset_pair<2,  36, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2,  36, 30, 1>::value, vnode_base_offset_pair<2,  36, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  36, 30, 2>::value, vnode_base_offset_pair<2,  36, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 31, 0>::value, vnode_base_offset_pair<2,  36, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2,  36, 31, 1>::value, vnode_base_offset_pair<2,  36, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 32, 0>::value, vnode_base_offset_pair<2,  36, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2,  36, 32, 1>::value, vnode_base_offset_pair<2,  36, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 33, 0>::value, vnode_base_offset_pair<2,  36, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2,  36, 33, 1>::value, vnode_base_offset_pair<2,  36, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 34, 0>::value, vnode_base_offset_pair<2,  36, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2,  36, 34, 1>::value, vnode_base_offset_pair<2,  36, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 35, 0>::value, vnode_base_offset_pair<2,  36, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2,  36, 35, 1>::value, vnode_base_offset_pair<2,  36, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 36, 0>::value, vnode_base_offset_pair<2,  36, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2,  36, 36, 1>::value, vnode_base_offset_pair<2,  36, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 37, 0>::value, vnode_base_offset_pair<2,  36, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2,  36, 37, 1>::value, vnode_base_offset_pair<2,  36, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 38, 0>::value, vnode_base_offset_pair<2,  36, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2,  36, 38, 1>::value, vnode_base_offset_pair<2,  36, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 39, 0>::value, vnode_base_offset_pair<2,  36, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2,  36, 39, 1>::value, vnode_base_offset_pair<2,  36, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 40, 0>::value, vnode_base_offset_pair<2,  36, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2,  36, 40, 1>::value, vnode_base_offset_pair<2,  36, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  36, 41, 0>::value, vnode_base_offset_pair<2,  36, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2,  36, 41, 1>::value, vnode_base_offset_pair<2,  36, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z40_8 =
{
    {
        { vnode_shift_mod_pair<2,  40,  0, 0>::value, vnode_base_offset_pair<2,  40,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2,  40,  0, 1>::value, vnode_base_offset_pair<2,  40,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  40,  0, 2>::value, vnode_base_offset_pair<2,  40,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  40,  0, 3>::value, vnode_base_offset_pair<2,  40,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  40,  1, 0>::value, vnode_base_offset_pair<2,  40,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2,  40,  1, 1>::value, vnode_base_offset_pair<2,  40,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  40,  1, 2>::value, vnode_base_offset_pair<2,  40,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  40,  1, 3>::value, vnode_base_offset_pair<2,  40,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  40,  1, 4>::value, vnode_base_offset_pair<2,  40,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  40,  2, 0>::value, vnode_base_offset_pair<2,  40,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2,  40,  2, 1>::value, vnode_base_offset_pair<2,  40,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  40,  2, 2>::value, vnode_base_offset_pair<2,  40,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  40,  2, 3>::value, vnode_base_offset_pair<2,  40,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  40,  3, 0>::value, vnode_base_offset_pair<2,  40,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2,  40,  3, 1>::value, vnode_base_offset_pair<2,  40,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  40,  3, 2>::value, vnode_base_offset_pair<2,  40,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  40,  3, 3>::value, vnode_base_offset_pair<2,  40,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  40,  3, 4>::value, vnode_base_offset_pair<2,  40,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  40,  4, 0>::value, vnode_base_offset_pair<2,  40,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2,  40,  4, 1>::value, vnode_base_offset_pair<2,  40,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40,  5, 0>::value, vnode_base_offset_pair<2,  40,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2,  40,  5, 1>::value, vnode_base_offset_pair<2,  40,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  40,  5, 2>::value, vnode_base_offset_pair<2,  40,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  40,  6, 0>::value, vnode_base_offset_pair<2,  40,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2,  40,  6, 1>::value, vnode_base_offset_pair<2,  40,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  40,  6, 2>::value, vnode_base_offset_pair<2,  40,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  40,  7, 0>::value, vnode_base_offset_pair<2,  40,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2,  40,  7, 1>::value, vnode_base_offset_pair<2,  40,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  40,  7, 2>::value, vnode_base_offset_pair<2,  40,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  40,  8, 0>::value, vnode_base_offset_pair<2,  40,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2,  40,  8, 1>::value, vnode_base_offset_pair<2,  40,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40,  9, 0>::value, vnode_base_offset_pair<2,  40,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2,  40,  9, 1>::value, vnode_base_offset_pair<2,  40,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  40,  9, 2>::value, vnode_base_offset_pair<2,  40,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 10, 0>::value, vnode_base_offset_pair<2,  40, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2,  40, 10, 1>::value, vnode_base_offset_pair<2,  40, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  40, 10, 2>::value, vnode_base_offset_pair<2,  40, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 11, 0>::value, vnode_base_offset_pair<2,  40, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2,  40, 11, 1>::value, vnode_base_offset_pair<2,  40, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  40, 11, 2>::value, vnode_base_offset_pair<2,  40, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 12, 0>::value, vnode_base_offset_pair<2,  40, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2,  40, 12, 1>::value, vnode_base_offset_pair<2,  40, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 13, 0>::value, vnode_base_offset_pair<2,  40, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2,  40, 13, 1>::value, vnode_base_offset_pair<2,  40, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  40, 13, 2>::value, vnode_base_offset_pair<2,  40, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 14, 0>::value, vnode_base_offset_pair<2,  40, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2,  40, 14, 1>::value, vnode_base_offset_pair<2,  40, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  40, 14, 2>::value, vnode_base_offset_pair<2,  40, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 15, 0>::value, vnode_base_offset_pair<2,  40, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2,  40, 15, 1>::value, vnode_base_offset_pair<2,  40, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 16, 0>::value, vnode_base_offset_pair<2,  40, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2,  40, 16, 1>::value, vnode_base_offset_pair<2,  40, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  40, 16, 2>::value, vnode_base_offset_pair<2,  40, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 17, 0>::value, vnode_base_offset_pair<2,  40, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2,  40, 17, 1>::value, vnode_base_offset_pair<2,  40, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  40, 17, 2>::value, vnode_base_offset_pair<2,  40, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 18, 0>::value, vnode_base_offset_pair<2,  40, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2,  40, 18, 1>::value, vnode_base_offset_pair<2,  40, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 19, 0>::value, vnode_base_offset_pair<2,  40, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2,  40, 19, 1>::value, vnode_base_offset_pair<2,  40, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 20, 0>::value, vnode_base_offset_pair<2,  40, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2,  40, 20, 1>::value, vnode_base_offset_pair<2,  40, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 21, 0>::value, vnode_base_offset_pair<2,  40, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2,  40, 21, 1>::value, vnode_base_offset_pair<2,  40, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 22, 0>::value, vnode_base_offset_pair<2,  40, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2,  40, 22, 1>::value, vnode_base_offset_pair<2,  40, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 23, 0>::value, vnode_base_offset_pair<2,  40, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2,  40, 23, 1>::value, vnode_base_offset_pair<2,  40, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 24, 0>::value, vnode_base_offset_pair<2,  40, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2,  40, 24, 1>::value, vnode_base_offset_pair<2,  40, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 25, 0>::value, vnode_base_offset_pair<2,  40, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2,  40, 25, 1>::value, vnode_base_offset_pair<2,  40, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 26, 0>::value, vnode_base_offset_pair<2,  40, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2,  40, 26, 1>::value, vnode_base_offset_pair<2,  40, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  40, 26, 2>::value, vnode_base_offset_pair<2,  40, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 27, 0>::value, vnode_base_offset_pair<2,  40, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2,  40, 27, 1>::value, vnode_base_offset_pair<2,  40, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 28, 0>::value, vnode_base_offset_pair<2,  40, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2,  40, 28, 1>::value, vnode_base_offset_pair<2,  40, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 29, 0>::value, vnode_base_offset_pair<2,  40, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2,  40, 29, 1>::value, vnode_base_offset_pair<2,  40, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 30, 0>::value, vnode_base_offset_pair<2,  40, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2,  40, 30, 1>::value, vnode_base_offset_pair<2,  40, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  40, 30, 2>::value, vnode_base_offset_pair<2,  40, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 31, 0>::value, vnode_base_offset_pair<2,  40, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2,  40, 31, 1>::value, vnode_base_offset_pair<2,  40, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 32, 0>::value, vnode_base_offset_pair<2,  40, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2,  40, 32, 1>::value, vnode_base_offset_pair<2,  40, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 33, 0>::value, vnode_base_offset_pair<2,  40, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2,  40, 33, 1>::value, vnode_base_offset_pair<2,  40, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 34, 0>::value, vnode_base_offset_pair<2,  40, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2,  40, 34, 1>::value, vnode_base_offset_pair<2,  40, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 35, 0>::value, vnode_base_offset_pair<2,  40, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2,  40, 35, 1>::value, vnode_base_offset_pair<2,  40, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 36, 0>::value, vnode_base_offset_pair<2,  40, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2,  40, 36, 1>::value, vnode_base_offset_pair<2,  40, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 37, 0>::value, vnode_base_offset_pair<2,  40, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2,  40, 37, 1>::value, vnode_base_offset_pair<2,  40, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 38, 0>::value, vnode_base_offset_pair<2,  40, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2,  40, 38, 1>::value, vnode_base_offset_pair<2,  40, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 39, 0>::value, vnode_base_offset_pair<2,  40, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2,  40, 39, 1>::value, vnode_base_offset_pair<2,  40, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 40, 0>::value, vnode_base_offset_pair<2,  40, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2,  40, 40, 1>::value, vnode_base_offset_pair<2,  40, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  40, 41, 0>::value, vnode_base_offset_pair<2,  40, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2,  40, 41, 1>::value, vnode_base_offset_pair<2,  40, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z44_8 =
{
    {
        { vnode_shift_mod_pair<2,  44,  0, 0>::value, vnode_base_offset_pair<2,  44,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2,  44,  0, 1>::value, vnode_base_offset_pair<2,  44,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  44,  0, 2>::value, vnode_base_offset_pair<2,  44,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  44,  0, 3>::value, vnode_base_offset_pair<2,  44,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  44,  1, 0>::value, vnode_base_offset_pair<2,  44,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2,  44,  1, 1>::value, vnode_base_offset_pair<2,  44,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  44,  1, 2>::value, vnode_base_offset_pair<2,  44,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  44,  1, 3>::value, vnode_base_offset_pair<2,  44,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  44,  1, 4>::value, vnode_base_offset_pair<2,  44,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  44,  2, 0>::value, vnode_base_offset_pair<2,  44,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2,  44,  2, 1>::value, vnode_base_offset_pair<2,  44,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  44,  2, 2>::value, vnode_base_offset_pair<2,  44,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  44,  2, 3>::value, vnode_base_offset_pair<2,  44,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  44,  3, 0>::value, vnode_base_offset_pair<2,  44,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2,  44,  3, 1>::value, vnode_base_offset_pair<2,  44,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  44,  3, 2>::value, vnode_base_offset_pair<2,  44,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  44,  3, 3>::value, vnode_base_offset_pair<2,  44,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  44,  3, 4>::value, vnode_base_offset_pair<2,  44,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  44,  4, 0>::value, vnode_base_offset_pair<2,  44,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2,  44,  4, 1>::value, vnode_base_offset_pair<2,  44,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44,  5, 0>::value, vnode_base_offset_pair<2,  44,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2,  44,  5, 1>::value, vnode_base_offset_pair<2,  44,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  44,  5, 2>::value, vnode_base_offset_pair<2,  44,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  44,  6, 0>::value, vnode_base_offset_pair<2,  44,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2,  44,  6, 1>::value, vnode_base_offset_pair<2,  44,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  44,  6, 2>::value, vnode_base_offset_pair<2,  44,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  44,  7, 0>::value, vnode_base_offset_pair<2,  44,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2,  44,  7, 1>::value, vnode_base_offset_pair<2,  44,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  44,  7, 2>::value, vnode_base_offset_pair<2,  44,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  44,  8, 0>::value, vnode_base_offset_pair<2,  44,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2,  44,  8, 1>::value, vnode_base_offset_pair<2,  44,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44,  9, 0>::value, vnode_base_offset_pair<2,  44,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2,  44,  9, 1>::value, vnode_base_offset_pair<2,  44,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  44,  9, 2>::value, vnode_base_offset_pair<2,  44,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 10, 0>::value, vnode_base_offset_pair<2,  44, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2,  44, 10, 1>::value, vnode_base_offset_pair<2,  44, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  44, 10, 2>::value, vnode_base_offset_pair<2,  44, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 11, 0>::value, vnode_base_offset_pair<2,  44, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2,  44, 11, 1>::value, vnode_base_offset_pair<2,  44, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  44, 11, 2>::value, vnode_base_offset_pair<2,  44, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 12, 0>::value, vnode_base_offset_pair<2,  44, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2,  44, 12, 1>::value, vnode_base_offset_pair<2,  44, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 13, 0>::value, vnode_base_offset_pair<2,  44, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2,  44, 13, 1>::value, vnode_base_offset_pair<2,  44, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  44, 13, 2>::value, vnode_base_offset_pair<2,  44, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 14, 0>::value, vnode_base_offset_pair<2,  44, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2,  44, 14, 1>::value, vnode_base_offset_pair<2,  44, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  44, 14, 2>::value, vnode_base_offset_pair<2,  44, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 15, 0>::value, vnode_base_offset_pair<2,  44, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2,  44, 15, 1>::value, vnode_base_offset_pair<2,  44, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 16, 0>::value, vnode_base_offset_pair<2,  44, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2,  44, 16, 1>::value, vnode_base_offset_pair<2,  44, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  44, 16, 2>::value, vnode_base_offset_pair<2,  44, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 17, 0>::value, vnode_base_offset_pair<2,  44, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2,  44, 17, 1>::value, vnode_base_offset_pair<2,  44, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  44, 17, 2>::value, vnode_base_offset_pair<2,  44, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 18, 0>::value, vnode_base_offset_pair<2,  44, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2,  44, 18, 1>::value, vnode_base_offset_pair<2,  44, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 19, 0>::value, vnode_base_offset_pair<2,  44, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2,  44, 19, 1>::value, vnode_base_offset_pair<2,  44, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 20, 0>::value, vnode_base_offset_pair<2,  44, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2,  44, 20, 1>::value, vnode_base_offset_pair<2,  44, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 21, 0>::value, vnode_base_offset_pair<2,  44, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2,  44, 21, 1>::value, vnode_base_offset_pair<2,  44, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 22, 0>::value, vnode_base_offset_pair<2,  44, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2,  44, 22, 1>::value, vnode_base_offset_pair<2,  44, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 23, 0>::value, vnode_base_offset_pair<2,  44, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2,  44, 23, 1>::value, vnode_base_offset_pair<2,  44, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 24, 0>::value, vnode_base_offset_pair<2,  44, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2,  44, 24, 1>::value, vnode_base_offset_pair<2,  44, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 25, 0>::value, vnode_base_offset_pair<2,  44, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2,  44, 25, 1>::value, vnode_base_offset_pair<2,  44, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 26, 0>::value, vnode_base_offset_pair<2,  44, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2,  44, 26, 1>::value, vnode_base_offset_pair<2,  44, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  44, 26, 2>::value, vnode_base_offset_pair<2,  44, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 27, 0>::value, vnode_base_offset_pair<2,  44, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2,  44, 27, 1>::value, vnode_base_offset_pair<2,  44, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 28, 0>::value, vnode_base_offset_pair<2,  44, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2,  44, 28, 1>::value, vnode_base_offset_pair<2,  44, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 29, 0>::value, vnode_base_offset_pair<2,  44, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2,  44, 29, 1>::value, vnode_base_offset_pair<2,  44, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 30, 0>::value, vnode_base_offset_pair<2,  44, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2,  44, 30, 1>::value, vnode_base_offset_pair<2,  44, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  44, 30, 2>::value, vnode_base_offset_pair<2,  44, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 31, 0>::value, vnode_base_offset_pair<2,  44, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2,  44, 31, 1>::value, vnode_base_offset_pair<2,  44, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 32, 0>::value, vnode_base_offset_pair<2,  44, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2,  44, 32, 1>::value, vnode_base_offset_pair<2,  44, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 33, 0>::value, vnode_base_offset_pair<2,  44, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2,  44, 33, 1>::value, vnode_base_offset_pair<2,  44, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 34, 0>::value, vnode_base_offset_pair<2,  44, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2,  44, 34, 1>::value, vnode_base_offset_pair<2,  44, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 35, 0>::value, vnode_base_offset_pair<2,  44, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2,  44, 35, 1>::value, vnode_base_offset_pair<2,  44, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 36, 0>::value, vnode_base_offset_pair<2,  44, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2,  44, 36, 1>::value, vnode_base_offset_pair<2,  44, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 37, 0>::value, vnode_base_offset_pair<2,  44, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2,  44, 37, 1>::value, vnode_base_offset_pair<2,  44, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 38, 0>::value, vnode_base_offset_pair<2,  44, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2,  44, 38, 1>::value, vnode_base_offset_pair<2,  44, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 39, 0>::value, vnode_base_offset_pair<2,  44, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2,  44, 39, 1>::value, vnode_base_offset_pair<2,  44, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 40, 0>::value, vnode_base_offset_pair<2,  44, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2,  44, 40, 1>::value, vnode_base_offset_pair<2,  44, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  44, 41, 0>::value, vnode_base_offset_pair<2,  44, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2,  44, 41, 1>::value, vnode_base_offset_pair<2,  44, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z48_8 =
{
    {
        { vnode_shift_mod_pair<2,  48,  0, 0>::value, vnode_base_offset_pair<2,  48,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2,  48,  0, 1>::value, vnode_base_offset_pair<2,  48,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  48,  0, 2>::value, vnode_base_offset_pair<2,  48,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  48,  0, 3>::value, vnode_base_offset_pair<2,  48,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  48,  1, 0>::value, vnode_base_offset_pair<2,  48,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2,  48,  1, 1>::value, vnode_base_offset_pair<2,  48,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  48,  1, 2>::value, vnode_base_offset_pair<2,  48,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  48,  1, 3>::value, vnode_base_offset_pair<2,  48,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  48,  1, 4>::value, vnode_base_offset_pair<2,  48,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  48,  2, 0>::value, vnode_base_offset_pair<2,  48,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2,  48,  2, 1>::value, vnode_base_offset_pair<2,  48,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  48,  2, 2>::value, vnode_base_offset_pair<2,  48,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  48,  2, 3>::value, vnode_base_offset_pair<2,  48,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  48,  3, 0>::value, vnode_base_offset_pair<2,  48,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2,  48,  3, 1>::value, vnode_base_offset_pair<2,  48,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  48,  3, 2>::value, vnode_base_offset_pair<2,  48,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  48,  3, 3>::value, vnode_base_offset_pair<2,  48,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  48,  3, 4>::value, vnode_base_offset_pair<2,  48,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  48,  4, 0>::value, vnode_base_offset_pair<2,  48,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2,  48,  4, 1>::value, vnode_base_offset_pair<2,  48,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48,  5, 0>::value, vnode_base_offset_pair<2,  48,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2,  48,  5, 1>::value, vnode_base_offset_pair<2,  48,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  48,  5, 2>::value, vnode_base_offset_pair<2,  48,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  48,  6, 0>::value, vnode_base_offset_pair<2,  48,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2,  48,  6, 1>::value, vnode_base_offset_pair<2,  48,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  48,  6, 2>::value, vnode_base_offset_pair<2,  48,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  48,  7, 0>::value, vnode_base_offset_pair<2,  48,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2,  48,  7, 1>::value, vnode_base_offset_pair<2,  48,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  48,  7, 2>::value, vnode_base_offset_pair<2,  48,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  48,  8, 0>::value, vnode_base_offset_pair<2,  48,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2,  48,  8, 1>::value, vnode_base_offset_pair<2,  48,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48,  9, 0>::value, vnode_base_offset_pair<2,  48,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2,  48,  9, 1>::value, vnode_base_offset_pair<2,  48,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  48,  9, 2>::value, vnode_base_offset_pair<2,  48,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 10, 0>::value, vnode_base_offset_pair<2,  48, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2,  48, 10, 1>::value, vnode_base_offset_pair<2,  48, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  48, 10, 2>::value, vnode_base_offset_pair<2,  48, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 11, 0>::value, vnode_base_offset_pair<2,  48, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2,  48, 11, 1>::value, vnode_base_offset_pair<2,  48, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  48, 11, 2>::value, vnode_base_offset_pair<2,  48, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 12, 0>::value, vnode_base_offset_pair<2,  48, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2,  48, 12, 1>::value, vnode_base_offset_pair<2,  48, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 13, 0>::value, vnode_base_offset_pair<2,  48, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2,  48, 13, 1>::value, vnode_base_offset_pair<2,  48, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  48, 13, 2>::value, vnode_base_offset_pair<2,  48, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 14, 0>::value, vnode_base_offset_pair<2,  48, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2,  48, 14, 1>::value, vnode_base_offset_pair<2,  48, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  48, 14, 2>::value, vnode_base_offset_pair<2,  48, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 15, 0>::value, vnode_base_offset_pair<2,  48, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2,  48, 15, 1>::value, vnode_base_offset_pair<2,  48, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 16, 0>::value, vnode_base_offset_pair<2,  48, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2,  48, 16, 1>::value, vnode_base_offset_pair<2,  48, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  48, 16, 2>::value, vnode_base_offset_pair<2,  48, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 17, 0>::value, vnode_base_offset_pair<2,  48, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2,  48, 17, 1>::value, vnode_base_offset_pair<2,  48, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  48, 17, 2>::value, vnode_base_offset_pair<2,  48, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 18, 0>::value, vnode_base_offset_pair<2,  48, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2,  48, 18, 1>::value, vnode_base_offset_pair<2,  48, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 19, 0>::value, vnode_base_offset_pair<2,  48, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2,  48, 19, 1>::value, vnode_base_offset_pair<2,  48, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 20, 0>::value, vnode_base_offset_pair<2,  48, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2,  48, 20, 1>::value, vnode_base_offset_pair<2,  48, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 21, 0>::value, vnode_base_offset_pair<2,  48, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2,  48, 21, 1>::value, vnode_base_offset_pair<2,  48, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 22, 0>::value, vnode_base_offset_pair<2,  48, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2,  48, 22, 1>::value, vnode_base_offset_pair<2,  48, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 23, 0>::value, vnode_base_offset_pair<2,  48, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2,  48, 23, 1>::value, vnode_base_offset_pair<2,  48, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 24, 0>::value, vnode_base_offset_pair<2,  48, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2,  48, 24, 1>::value, vnode_base_offset_pair<2,  48, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 25, 0>::value, vnode_base_offset_pair<2,  48, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2,  48, 25, 1>::value, vnode_base_offset_pair<2,  48, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 26, 0>::value, vnode_base_offset_pair<2,  48, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2,  48, 26, 1>::value, vnode_base_offset_pair<2,  48, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  48, 26, 2>::value, vnode_base_offset_pair<2,  48, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 27, 0>::value, vnode_base_offset_pair<2,  48, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2,  48, 27, 1>::value, vnode_base_offset_pair<2,  48, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 28, 0>::value, vnode_base_offset_pair<2,  48, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2,  48, 28, 1>::value, vnode_base_offset_pair<2,  48, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 29, 0>::value, vnode_base_offset_pair<2,  48, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2,  48, 29, 1>::value, vnode_base_offset_pair<2,  48, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 30, 0>::value, vnode_base_offset_pair<2,  48, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2,  48, 30, 1>::value, vnode_base_offset_pair<2,  48, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  48, 30, 2>::value, vnode_base_offset_pair<2,  48, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 31, 0>::value, vnode_base_offset_pair<2,  48, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2,  48, 31, 1>::value, vnode_base_offset_pair<2,  48, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 32, 0>::value, vnode_base_offset_pair<2,  48, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2,  48, 32, 1>::value, vnode_base_offset_pair<2,  48, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 33, 0>::value, vnode_base_offset_pair<2,  48, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2,  48, 33, 1>::value, vnode_base_offset_pair<2,  48, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 34, 0>::value, vnode_base_offset_pair<2,  48, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2,  48, 34, 1>::value, vnode_base_offset_pair<2,  48, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 35, 0>::value, vnode_base_offset_pair<2,  48, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2,  48, 35, 1>::value, vnode_base_offset_pair<2,  48, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 36, 0>::value, vnode_base_offset_pair<2,  48, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2,  48, 36, 1>::value, vnode_base_offset_pair<2,  48, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 37, 0>::value, vnode_base_offset_pair<2,  48, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2,  48, 37, 1>::value, vnode_base_offset_pair<2,  48, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 38, 0>::value, vnode_base_offset_pair<2,  48, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2,  48, 38, 1>::value, vnode_base_offset_pair<2,  48, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 39, 0>::value, vnode_base_offset_pair<2,  48, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2,  48, 39, 1>::value, vnode_base_offset_pair<2,  48, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 40, 0>::value, vnode_base_offset_pair<2,  48, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2,  48, 40, 1>::value, vnode_base_offset_pair<2,  48, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  48, 41, 0>::value, vnode_base_offset_pair<2,  48, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2,  48, 41, 1>::value, vnode_base_offset_pair<2,  48, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z52_8 =
{
    {
        { vnode_shift_mod_pair<2,  52,  0, 0>::value, vnode_base_offset_pair<2,  52,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2,  52,  0, 1>::value, vnode_base_offset_pair<2,  52,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  52,  0, 2>::value, vnode_base_offset_pair<2,  52,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  52,  0, 3>::value, vnode_base_offset_pair<2,  52,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  52,  1, 0>::value, vnode_base_offset_pair<2,  52,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2,  52,  1, 1>::value, vnode_base_offset_pair<2,  52,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  52,  1, 2>::value, vnode_base_offset_pair<2,  52,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  52,  1, 3>::value, vnode_base_offset_pair<2,  52,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  52,  1, 4>::value, vnode_base_offset_pair<2,  52,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  52,  2, 0>::value, vnode_base_offset_pair<2,  52,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2,  52,  2, 1>::value, vnode_base_offset_pair<2,  52,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  52,  2, 2>::value, vnode_base_offset_pair<2,  52,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  52,  2, 3>::value, vnode_base_offset_pair<2,  52,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  52,  3, 0>::value, vnode_base_offset_pair<2,  52,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2,  52,  3, 1>::value, vnode_base_offset_pair<2,  52,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  52,  3, 2>::value, vnode_base_offset_pair<2,  52,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  52,  3, 3>::value, vnode_base_offset_pair<2,  52,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  52,  3, 4>::value, vnode_base_offset_pair<2,  52,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  52,  4, 0>::value, vnode_base_offset_pair<2,  52,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2,  52,  4, 1>::value, vnode_base_offset_pair<2,  52,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52,  5, 0>::value, vnode_base_offset_pair<2,  52,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2,  52,  5, 1>::value, vnode_base_offset_pair<2,  52,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  52,  5, 2>::value, vnode_base_offset_pair<2,  52,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  52,  6, 0>::value, vnode_base_offset_pair<2,  52,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2,  52,  6, 1>::value, vnode_base_offset_pair<2,  52,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  52,  6, 2>::value, vnode_base_offset_pair<2,  52,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  52,  7, 0>::value, vnode_base_offset_pair<2,  52,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2,  52,  7, 1>::value, vnode_base_offset_pair<2,  52,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  52,  7, 2>::value, vnode_base_offset_pair<2,  52,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  52,  8, 0>::value, vnode_base_offset_pair<2,  52,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2,  52,  8, 1>::value, vnode_base_offset_pair<2,  52,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52,  9, 0>::value, vnode_base_offset_pair<2,  52,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2,  52,  9, 1>::value, vnode_base_offset_pair<2,  52,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  52,  9, 2>::value, vnode_base_offset_pair<2,  52,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 10, 0>::value, vnode_base_offset_pair<2,  52, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2,  52, 10, 1>::value, vnode_base_offset_pair<2,  52, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  52, 10, 2>::value, vnode_base_offset_pair<2,  52, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 11, 0>::value, vnode_base_offset_pair<2,  52, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2,  52, 11, 1>::value, vnode_base_offset_pair<2,  52, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  52, 11, 2>::value, vnode_base_offset_pair<2,  52, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 12, 0>::value, vnode_base_offset_pair<2,  52, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2,  52, 12, 1>::value, vnode_base_offset_pair<2,  52, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 13, 0>::value, vnode_base_offset_pair<2,  52, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2,  52, 13, 1>::value, vnode_base_offset_pair<2,  52, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  52, 13, 2>::value, vnode_base_offset_pair<2,  52, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 14, 0>::value, vnode_base_offset_pair<2,  52, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2,  52, 14, 1>::value, vnode_base_offset_pair<2,  52, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  52, 14, 2>::value, vnode_base_offset_pair<2,  52, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 15, 0>::value, vnode_base_offset_pair<2,  52, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2,  52, 15, 1>::value, vnode_base_offset_pair<2,  52, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 16, 0>::value, vnode_base_offset_pair<2,  52, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2,  52, 16, 1>::value, vnode_base_offset_pair<2,  52, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  52, 16, 2>::value, vnode_base_offset_pair<2,  52, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 17, 0>::value, vnode_base_offset_pair<2,  52, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2,  52, 17, 1>::value, vnode_base_offset_pair<2,  52, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  52, 17, 2>::value, vnode_base_offset_pair<2,  52, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 18, 0>::value, vnode_base_offset_pair<2,  52, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2,  52, 18, 1>::value, vnode_base_offset_pair<2,  52, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 19, 0>::value, vnode_base_offset_pair<2,  52, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2,  52, 19, 1>::value, vnode_base_offset_pair<2,  52, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 20, 0>::value, vnode_base_offset_pair<2,  52, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2,  52, 20, 1>::value, vnode_base_offset_pair<2,  52, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 21, 0>::value, vnode_base_offset_pair<2,  52, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2,  52, 21, 1>::value, vnode_base_offset_pair<2,  52, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 22, 0>::value, vnode_base_offset_pair<2,  52, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2,  52, 22, 1>::value, vnode_base_offset_pair<2,  52, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 23, 0>::value, vnode_base_offset_pair<2,  52, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2,  52, 23, 1>::value, vnode_base_offset_pair<2,  52, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 24, 0>::value, vnode_base_offset_pair<2,  52, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2,  52, 24, 1>::value, vnode_base_offset_pair<2,  52, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 25, 0>::value, vnode_base_offset_pair<2,  52, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2,  52, 25, 1>::value, vnode_base_offset_pair<2,  52, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 26, 0>::value, vnode_base_offset_pair<2,  52, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2,  52, 26, 1>::value, vnode_base_offset_pair<2,  52, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  52, 26, 2>::value, vnode_base_offset_pair<2,  52, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 27, 0>::value, vnode_base_offset_pair<2,  52, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2,  52, 27, 1>::value, vnode_base_offset_pair<2,  52, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 28, 0>::value, vnode_base_offset_pair<2,  52, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2,  52, 28, 1>::value, vnode_base_offset_pair<2,  52, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 29, 0>::value, vnode_base_offset_pair<2,  52, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2,  52, 29, 1>::value, vnode_base_offset_pair<2,  52, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 30, 0>::value, vnode_base_offset_pair<2,  52, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2,  52, 30, 1>::value, vnode_base_offset_pair<2,  52, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  52, 30, 2>::value, vnode_base_offset_pair<2,  52, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 31, 0>::value, vnode_base_offset_pair<2,  52, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2,  52, 31, 1>::value, vnode_base_offset_pair<2,  52, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 32, 0>::value, vnode_base_offset_pair<2,  52, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2,  52, 32, 1>::value, vnode_base_offset_pair<2,  52, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 33, 0>::value, vnode_base_offset_pair<2,  52, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2,  52, 33, 1>::value, vnode_base_offset_pair<2,  52, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 34, 0>::value, vnode_base_offset_pair<2,  52, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2,  52, 34, 1>::value, vnode_base_offset_pair<2,  52, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 35, 0>::value, vnode_base_offset_pair<2,  52, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2,  52, 35, 1>::value, vnode_base_offset_pair<2,  52, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 36, 0>::value, vnode_base_offset_pair<2,  52, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2,  52, 36, 1>::value, vnode_base_offset_pair<2,  52, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 37, 0>::value, vnode_base_offset_pair<2,  52, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2,  52, 37, 1>::value, vnode_base_offset_pair<2,  52, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 38, 0>::value, vnode_base_offset_pair<2,  52, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2,  52, 38, 1>::value, vnode_base_offset_pair<2,  52, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 39, 0>::value, vnode_base_offset_pair<2,  52, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2,  52, 39, 1>::value, vnode_base_offset_pair<2,  52, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 40, 0>::value, vnode_base_offset_pair<2,  52, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2,  52, 40, 1>::value, vnode_base_offset_pair<2,  52, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  52, 41, 0>::value, vnode_base_offset_pair<2,  52, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2,  52, 41, 1>::value, vnode_base_offset_pair<2,  52, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z56_8 =
{
    {
        { vnode_shift_mod_pair<2,  56,  0, 0>::value, vnode_base_offset_pair<2,  56,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2,  56,  0, 1>::value, vnode_base_offset_pair<2,  56,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  56,  0, 2>::value, vnode_base_offset_pair<2,  56,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  56,  0, 3>::value, vnode_base_offset_pair<2,  56,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  56,  1, 0>::value, vnode_base_offset_pair<2,  56,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2,  56,  1, 1>::value, vnode_base_offset_pair<2,  56,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  56,  1, 2>::value, vnode_base_offset_pair<2,  56,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  56,  1, 3>::value, vnode_base_offset_pair<2,  56,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  56,  1, 4>::value, vnode_base_offset_pair<2,  56,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  56,  2, 0>::value, vnode_base_offset_pair<2,  56,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2,  56,  2, 1>::value, vnode_base_offset_pair<2,  56,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  56,  2, 2>::value, vnode_base_offset_pair<2,  56,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  56,  2, 3>::value, vnode_base_offset_pair<2,  56,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  56,  3, 0>::value, vnode_base_offset_pair<2,  56,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2,  56,  3, 1>::value, vnode_base_offset_pair<2,  56,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  56,  3, 2>::value, vnode_base_offset_pair<2,  56,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  56,  3, 3>::value, vnode_base_offset_pair<2,  56,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  56,  3, 4>::value, vnode_base_offset_pair<2,  56,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  56,  4, 0>::value, vnode_base_offset_pair<2,  56,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2,  56,  4, 1>::value, vnode_base_offset_pair<2,  56,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56,  5, 0>::value, vnode_base_offset_pair<2,  56,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2,  56,  5, 1>::value, vnode_base_offset_pair<2,  56,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  56,  5, 2>::value, vnode_base_offset_pair<2,  56,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  56,  6, 0>::value, vnode_base_offset_pair<2,  56,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2,  56,  6, 1>::value, vnode_base_offset_pair<2,  56,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  56,  6, 2>::value, vnode_base_offset_pair<2,  56,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  56,  7, 0>::value, vnode_base_offset_pair<2,  56,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2,  56,  7, 1>::value, vnode_base_offset_pair<2,  56,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  56,  7, 2>::value, vnode_base_offset_pair<2,  56,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  56,  8, 0>::value, vnode_base_offset_pair<2,  56,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2,  56,  8, 1>::value, vnode_base_offset_pair<2,  56,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56,  9, 0>::value, vnode_base_offset_pair<2,  56,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2,  56,  9, 1>::value, vnode_base_offset_pair<2,  56,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  56,  9, 2>::value, vnode_base_offset_pair<2,  56,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 10, 0>::value, vnode_base_offset_pair<2,  56, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2,  56, 10, 1>::value, vnode_base_offset_pair<2,  56, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  56, 10, 2>::value, vnode_base_offset_pair<2,  56, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 11, 0>::value, vnode_base_offset_pair<2,  56, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2,  56, 11, 1>::value, vnode_base_offset_pair<2,  56, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  56, 11, 2>::value, vnode_base_offset_pair<2,  56, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 12, 0>::value, vnode_base_offset_pair<2,  56, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2,  56, 12, 1>::value, vnode_base_offset_pair<2,  56, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 13, 0>::value, vnode_base_offset_pair<2,  56, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2,  56, 13, 1>::value, vnode_base_offset_pair<2,  56, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  56, 13, 2>::value, vnode_base_offset_pair<2,  56, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 14, 0>::value, vnode_base_offset_pair<2,  56, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2,  56, 14, 1>::value, vnode_base_offset_pair<2,  56, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  56, 14, 2>::value, vnode_base_offset_pair<2,  56, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 15, 0>::value, vnode_base_offset_pair<2,  56, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2,  56, 15, 1>::value, vnode_base_offset_pair<2,  56, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 16, 0>::value, vnode_base_offset_pair<2,  56, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2,  56, 16, 1>::value, vnode_base_offset_pair<2,  56, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  56, 16, 2>::value, vnode_base_offset_pair<2,  56, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 17, 0>::value, vnode_base_offset_pair<2,  56, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2,  56, 17, 1>::value, vnode_base_offset_pair<2,  56, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  56, 17, 2>::value, vnode_base_offset_pair<2,  56, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 18, 0>::value, vnode_base_offset_pair<2,  56, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2,  56, 18, 1>::value, vnode_base_offset_pair<2,  56, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 19, 0>::value, vnode_base_offset_pair<2,  56, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2,  56, 19, 1>::value, vnode_base_offset_pair<2,  56, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 20, 0>::value, vnode_base_offset_pair<2,  56, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2,  56, 20, 1>::value, vnode_base_offset_pair<2,  56, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 21, 0>::value, vnode_base_offset_pair<2,  56, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2,  56, 21, 1>::value, vnode_base_offset_pair<2,  56, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 22, 0>::value, vnode_base_offset_pair<2,  56, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2,  56, 22, 1>::value, vnode_base_offset_pair<2,  56, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 23, 0>::value, vnode_base_offset_pair<2,  56, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2,  56, 23, 1>::value, vnode_base_offset_pair<2,  56, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 24, 0>::value, vnode_base_offset_pair<2,  56, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2,  56, 24, 1>::value, vnode_base_offset_pair<2,  56, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 25, 0>::value, vnode_base_offset_pair<2,  56, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2,  56, 25, 1>::value, vnode_base_offset_pair<2,  56, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 26, 0>::value, vnode_base_offset_pair<2,  56, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2,  56, 26, 1>::value, vnode_base_offset_pair<2,  56, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  56, 26, 2>::value, vnode_base_offset_pair<2,  56, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 27, 0>::value, vnode_base_offset_pair<2,  56, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2,  56, 27, 1>::value, vnode_base_offset_pair<2,  56, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 28, 0>::value, vnode_base_offset_pair<2,  56, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2,  56, 28, 1>::value, vnode_base_offset_pair<2,  56, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 29, 0>::value, vnode_base_offset_pair<2,  56, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2,  56, 29, 1>::value, vnode_base_offset_pair<2,  56, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 30, 0>::value, vnode_base_offset_pair<2,  56, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2,  56, 30, 1>::value, vnode_base_offset_pair<2,  56, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  56, 30, 2>::value, vnode_base_offset_pair<2,  56, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 31, 0>::value, vnode_base_offset_pair<2,  56, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2,  56, 31, 1>::value, vnode_base_offset_pair<2,  56, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 32, 0>::value, vnode_base_offset_pair<2,  56, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2,  56, 32, 1>::value, vnode_base_offset_pair<2,  56, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 33, 0>::value, vnode_base_offset_pair<2,  56, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2,  56, 33, 1>::value, vnode_base_offset_pair<2,  56, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 34, 0>::value, vnode_base_offset_pair<2,  56, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2,  56, 34, 1>::value, vnode_base_offset_pair<2,  56, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 35, 0>::value, vnode_base_offset_pair<2,  56, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2,  56, 35, 1>::value, vnode_base_offset_pair<2,  56, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 36, 0>::value, vnode_base_offset_pair<2,  56, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2,  56, 36, 1>::value, vnode_base_offset_pair<2,  56, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 37, 0>::value, vnode_base_offset_pair<2,  56, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2,  56, 37, 1>::value, vnode_base_offset_pair<2,  56, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 38, 0>::value, vnode_base_offset_pair<2,  56, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2,  56, 38, 1>::value, vnode_base_offset_pair<2,  56, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 39, 0>::value, vnode_base_offset_pair<2,  56, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2,  56, 39, 1>::value, vnode_base_offset_pair<2,  56, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 40, 0>::value, vnode_base_offset_pair<2,  56, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2,  56, 40, 1>::value, vnode_base_offset_pair<2,  56, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  56, 41, 0>::value, vnode_base_offset_pair<2,  56, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2,  56, 41, 1>::value, vnode_base_offset_pair<2,  56, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z60_8 =
{
    {
        { vnode_shift_mod_pair<2,  60,  0, 0>::value, vnode_base_offset_pair<2,  60,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2,  60,  0, 1>::value, vnode_base_offset_pair<2,  60,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  60,  0, 2>::value, vnode_base_offset_pair<2,  60,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  60,  0, 3>::value, vnode_base_offset_pair<2,  60,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  60,  1, 0>::value, vnode_base_offset_pair<2,  60,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2,  60,  1, 1>::value, vnode_base_offset_pair<2,  60,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  60,  1, 2>::value, vnode_base_offset_pair<2,  60,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  60,  1, 3>::value, vnode_base_offset_pair<2,  60,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  60,  1, 4>::value, vnode_base_offset_pair<2,  60,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  60,  2, 0>::value, vnode_base_offset_pair<2,  60,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2,  60,  2, 1>::value, vnode_base_offset_pair<2,  60,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  60,  2, 2>::value, vnode_base_offset_pair<2,  60,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  60,  2, 3>::value, vnode_base_offset_pair<2,  60,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  60,  3, 0>::value, vnode_base_offset_pair<2,  60,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2,  60,  3, 1>::value, vnode_base_offset_pair<2,  60,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  60,  3, 2>::value, vnode_base_offset_pair<2,  60,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  60,  3, 3>::value, vnode_base_offset_pair<2,  60,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  60,  3, 4>::value, vnode_base_offset_pair<2,  60,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  60,  4, 0>::value, vnode_base_offset_pair<2,  60,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2,  60,  4, 1>::value, vnode_base_offset_pair<2,  60,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60,  5, 0>::value, vnode_base_offset_pair<2,  60,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2,  60,  5, 1>::value, vnode_base_offset_pair<2,  60,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  60,  5, 2>::value, vnode_base_offset_pair<2,  60,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  60,  6, 0>::value, vnode_base_offset_pair<2,  60,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2,  60,  6, 1>::value, vnode_base_offset_pair<2,  60,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  60,  6, 2>::value, vnode_base_offset_pair<2,  60,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  60,  7, 0>::value, vnode_base_offset_pair<2,  60,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2,  60,  7, 1>::value, vnode_base_offset_pair<2,  60,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  60,  7, 2>::value, vnode_base_offset_pair<2,  60,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  60,  8, 0>::value, vnode_base_offset_pair<2,  60,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2,  60,  8, 1>::value, vnode_base_offset_pair<2,  60,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60,  9, 0>::value, vnode_base_offset_pair<2,  60,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2,  60,  9, 1>::value, vnode_base_offset_pair<2,  60,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  60,  9, 2>::value, vnode_base_offset_pair<2,  60,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 10, 0>::value, vnode_base_offset_pair<2,  60, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2,  60, 10, 1>::value, vnode_base_offset_pair<2,  60, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  60, 10, 2>::value, vnode_base_offset_pair<2,  60, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 11, 0>::value, vnode_base_offset_pair<2,  60, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2,  60, 11, 1>::value, vnode_base_offset_pair<2,  60, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  60, 11, 2>::value, vnode_base_offset_pair<2,  60, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 12, 0>::value, vnode_base_offset_pair<2,  60, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2,  60, 12, 1>::value, vnode_base_offset_pair<2,  60, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 13, 0>::value, vnode_base_offset_pair<2,  60, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2,  60, 13, 1>::value, vnode_base_offset_pair<2,  60, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  60, 13, 2>::value, vnode_base_offset_pair<2,  60, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 14, 0>::value, vnode_base_offset_pair<2,  60, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2,  60, 14, 1>::value, vnode_base_offset_pair<2,  60, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  60, 14, 2>::value, vnode_base_offset_pair<2,  60, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 15, 0>::value, vnode_base_offset_pair<2,  60, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2,  60, 15, 1>::value, vnode_base_offset_pair<2,  60, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 16, 0>::value, vnode_base_offset_pair<2,  60, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2,  60, 16, 1>::value, vnode_base_offset_pair<2,  60, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  60, 16, 2>::value, vnode_base_offset_pair<2,  60, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 17, 0>::value, vnode_base_offset_pair<2,  60, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2,  60, 17, 1>::value, vnode_base_offset_pair<2,  60, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  60, 17, 2>::value, vnode_base_offset_pair<2,  60, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 18, 0>::value, vnode_base_offset_pair<2,  60, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2,  60, 18, 1>::value, vnode_base_offset_pair<2,  60, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 19, 0>::value, vnode_base_offset_pair<2,  60, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2,  60, 19, 1>::value, vnode_base_offset_pair<2,  60, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 20, 0>::value, vnode_base_offset_pair<2,  60, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2,  60, 20, 1>::value, vnode_base_offset_pair<2,  60, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 21, 0>::value, vnode_base_offset_pair<2,  60, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2,  60, 21, 1>::value, vnode_base_offset_pair<2,  60, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 22, 0>::value, vnode_base_offset_pair<2,  60, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2,  60, 22, 1>::value, vnode_base_offset_pair<2,  60, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 23, 0>::value, vnode_base_offset_pair<2,  60, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2,  60, 23, 1>::value, vnode_base_offset_pair<2,  60, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 24, 0>::value, vnode_base_offset_pair<2,  60, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2,  60, 24, 1>::value, vnode_base_offset_pair<2,  60, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 25, 0>::value, vnode_base_offset_pair<2,  60, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2,  60, 25, 1>::value, vnode_base_offset_pair<2,  60, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 26, 0>::value, vnode_base_offset_pair<2,  60, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2,  60, 26, 1>::value, vnode_base_offset_pair<2,  60, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  60, 26, 2>::value, vnode_base_offset_pair<2,  60, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 27, 0>::value, vnode_base_offset_pair<2,  60, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2,  60, 27, 1>::value, vnode_base_offset_pair<2,  60, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 28, 0>::value, vnode_base_offset_pair<2,  60, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2,  60, 28, 1>::value, vnode_base_offset_pair<2,  60, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 29, 0>::value, vnode_base_offset_pair<2,  60, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2,  60, 29, 1>::value, vnode_base_offset_pair<2,  60, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 30, 0>::value, vnode_base_offset_pair<2,  60, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2,  60, 30, 1>::value, vnode_base_offset_pair<2,  60, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  60, 30, 2>::value, vnode_base_offset_pair<2,  60, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 31, 0>::value, vnode_base_offset_pair<2,  60, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2,  60, 31, 1>::value, vnode_base_offset_pair<2,  60, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 32, 0>::value, vnode_base_offset_pair<2,  60, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2,  60, 32, 1>::value, vnode_base_offset_pair<2,  60, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 33, 0>::value, vnode_base_offset_pair<2,  60, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2,  60, 33, 1>::value, vnode_base_offset_pair<2,  60, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 34, 0>::value, vnode_base_offset_pair<2,  60, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2,  60, 34, 1>::value, vnode_base_offset_pair<2,  60, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 35, 0>::value, vnode_base_offset_pair<2,  60, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2,  60, 35, 1>::value, vnode_base_offset_pair<2,  60, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 36, 0>::value, vnode_base_offset_pair<2,  60, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2,  60, 36, 1>::value, vnode_base_offset_pair<2,  60, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 37, 0>::value, vnode_base_offset_pair<2,  60, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2,  60, 37, 1>::value, vnode_base_offset_pair<2,  60, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 38, 0>::value, vnode_base_offset_pair<2,  60, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2,  60, 38, 1>::value, vnode_base_offset_pair<2,  60, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 39, 0>::value, vnode_base_offset_pair<2,  60, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2,  60, 39, 1>::value, vnode_base_offset_pair<2,  60, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 40, 0>::value, vnode_base_offset_pair<2,  60, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2,  60, 40, 1>::value, vnode_base_offset_pair<2,  60, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  60, 41, 0>::value, vnode_base_offset_pair<2,  60, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2,  60, 41, 1>::value, vnode_base_offset_pair<2,  60, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z64_8 =
{
    {
        { vnode_shift_mod_pair<2,  64,  0, 0>::value, vnode_base_offset_pair<2,  64,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2,  64,  0, 1>::value, vnode_base_offset_pair<2,  64,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  64,  0, 2>::value, vnode_base_offset_pair<2,  64,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  64,  0, 3>::value, vnode_base_offset_pair<2,  64,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  64,  1, 0>::value, vnode_base_offset_pair<2,  64,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2,  64,  1, 1>::value, vnode_base_offset_pair<2,  64,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  64,  1, 2>::value, vnode_base_offset_pair<2,  64,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  64,  1, 3>::value, vnode_base_offset_pair<2,  64,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  64,  1, 4>::value, vnode_base_offset_pair<2,  64,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  64,  2, 0>::value, vnode_base_offset_pair<2,  64,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2,  64,  2, 1>::value, vnode_base_offset_pair<2,  64,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  64,  2, 2>::value, vnode_base_offset_pair<2,  64,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  64,  2, 3>::value, vnode_base_offset_pair<2,  64,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  64,  3, 0>::value, vnode_base_offset_pair<2,  64,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2,  64,  3, 1>::value, vnode_base_offset_pair<2,  64,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  64,  3, 2>::value, vnode_base_offset_pair<2,  64,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  64,  3, 3>::value, vnode_base_offset_pair<2,  64,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  64,  3, 4>::value, vnode_base_offset_pair<2,  64,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  64,  4, 0>::value, vnode_base_offset_pair<2,  64,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2,  64,  4, 1>::value, vnode_base_offset_pair<2,  64,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64,  5, 0>::value, vnode_base_offset_pair<2,  64,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2,  64,  5, 1>::value, vnode_base_offset_pair<2,  64,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  64,  5, 2>::value, vnode_base_offset_pair<2,  64,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  64,  6, 0>::value, vnode_base_offset_pair<2,  64,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2,  64,  6, 1>::value, vnode_base_offset_pair<2,  64,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  64,  6, 2>::value, vnode_base_offset_pair<2,  64,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  64,  7, 0>::value, vnode_base_offset_pair<2,  64,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2,  64,  7, 1>::value, vnode_base_offset_pair<2,  64,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  64,  7, 2>::value, vnode_base_offset_pair<2,  64,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  64,  8, 0>::value, vnode_base_offset_pair<2,  64,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2,  64,  8, 1>::value, vnode_base_offset_pair<2,  64,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64,  9, 0>::value, vnode_base_offset_pair<2,  64,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2,  64,  9, 1>::value, vnode_base_offset_pair<2,  64,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  64,  9, 2>::value, vnode_base_offset_pair<2,  64,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 10, 0>::value, vnode_base_offset_pair<2,  64, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2,  64, 10, 1>::value, vnode_base_offset_pair<2,  64, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  64, 10, 2>::value, vnode_base_offset_pair<2,  64, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 11, 0>::value, vnode_base_offset_pair<2,  64, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2,  64, 11, 1>::value, vnode_base_offset_pair<2,  64, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  64, 11, 2>::value, vnode_base_offset_pair<2,  64, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 12, 0>::value, vnode_base_offset_pair<2,  64, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2,  64, 12, 1>::value, vnode_base_offset_pair<2,  64, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 13, 0>::value, vnode_base_offset_pair<2,  64, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2,  64, 13, 1>::value, vnode_base_offset_pair<2,  64, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  64, 13, 2>::value, vnode_base_offset_pair<2,  64, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 14, 0>::value, vnode_base_offset_pair<2,  64, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2,  64, 14, 1>::value, vnode_base_offset_pair<2,  64, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  64, 14, 2>::value, vnode_base_offset_pair<2,  64, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 15, 0>::value, vnode_base_offset_pair<2,  64, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2,  64, 15, 1>::value, vnode_base_offset_pair<2,  64, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 16, 0>::value, vnode_base_offset_pair<2,  64, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2,  64, 16, 1>::value, vnode_base_offset_pair<2,  64, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  64, 16, 2>::value, vnode_base_offset_pair<2,  64, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 17, 0>::value, vnode_base_offset_pair<2,  64, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2,  64, 17, 1>::value, vnode_base_offset_pair<2,  64, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  64, 17, 2>::value, vnode_base_offset_pair<2,  64, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 18, 0>::value, vnode_base_offset_pair<2,  64, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2,  64, 18, 1>::value, vnode_base_offset_pair<2,  64, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 19, 0>::value, vnode_base_offset_pair<2,  64, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2,  64, 19, 1>::value, vnode_base_offset_pair<2,  64, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 20, 0>::value, vnode_base_offset_pair<2,  64, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2,  64, 20, 1>::value, vnode_base_offset_pair<2,  64, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 21, 0>::value, vnode_base_offset_pair<2,  64, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2,  64, 21, 1>::value, vnode_base_offset_pair<2,  64, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 22, 0>::value, vnode_base_offset_pair<2,  64, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2,  64, 22, 1>::value, vnode_base_offset_pair<2,  64, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 23, 0>::value, vnode_base_offset_pair<2,  64, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2,  64, 23, 1>::value, vnode_base_offset_pair<2,  64, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 24, 0>::value, vnode_base_offset_pair<2,  64, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2,  64, 24, 1>::value, vnode_base_offset_pair<2,  64, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 25, 0>::value, vnode_base_offset_pair<2,  64, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2,  64, 25, 1>::value, vnode_base_offset_pair<2,  64, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 26, 0>::value, vnode_base_offset_pair<2,  64, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2,  64, 26, 1>::value, vnode_base_offset_pair<2,  64, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  64, 26, 2>::value, vnode_base_offset_pair<2,  64, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 27, 0>::value, vnode_base_offset_pair<2,  64, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2,  64, 27, 1>::value, vnode_base_offset_pair<2,  64, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 28, 0>::value, vnode_base_offset_pair<2,  64, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2,  64, 28, 1>::value, vnode_base_offset_pair<2,  64, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 29, 0>::value, vnode_base_offset_pair<2,  64, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2,  64, 29, 1>::value, vnode_base_offset_pair<2,  64, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 30, 0>::value, vnode_base_offset_pair<2,  64, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2,  64, 30, 1>::value, vnode_base_offset_pair<2,  64, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  64, 30, 2>::value, vnode_base_offset_pair<2,  64, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 31, 0>::value, vnode_base_offset_pair<2,  64, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2,  64, 31, 1>::value, vnode_base_offset_pair<2,  64, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 32, 0>::value, vnode_base_offset_pair<2,  64, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2,  64, 32, 1>::value, vnode_base_offset_pair<2,  64, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 33, 0>::value, vnode_base_offset_pair<2,  64, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2,  64, 33, 1>::value, vnode_base_offset_pair<2,  64, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 34, 0>::value, vnode_base_offset_pair<2,  64, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2,  64, 34, 1>::value, vnode_base_offset_pair<2,  64, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 35, 0>::value, vnode_base_offset_pair<2,  64, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2,  64, 35, 1>::value, vnode_base_offset_pair<2,  64, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 36, 0>::value, vnode_base_offset_pair<2,  64, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2,  64, 36, 1>::value, vnode_base_offset_pair<2,  64, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 37, 0>::value, vnode_base_offset_pair<2,  64, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2,  64, 37, 1>::value, vnode_base_offset_pair<2,  64, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 38, 0>::value, vnode_base_offset_pair<2,  64, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2,  64, 38, 1>::value, vnode_base_offset_pair<2,  64, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 39, 0>::value, vnode_base_offset_pair<2,  64, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2,  64, 39, 1>::value, vnode_base_offset_pair<2,  64, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 40, 0>::value, vnode_base_offset_pair<2,  64, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2,  64, 40, 1>::value, vnode_base_offset_pair<2,  64, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  64, 41, 0>::value, vnode_base_offset_pair<2,  64, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2,  64, 41, 1>::value, vnode_base_offset_pair<2,  64, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z72_8 =
{
    {
        { vnode_shift_mod_pair<2,  72,  0, 0>::value, vnode_base_offset_pair<2,  72,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2,  72,  0, 1>::value, vnode_base_offset_pair<2,  72,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  72,  0, 2>::value, vnode_base_offset_pair<2,  72,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  72,  0, 3>::value, vnode_base_offset_pair<2,  72,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  72,  1, 0>::value, vnode_base_offset_pair<2,  72,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2,  72,  1, 1>::value, vnode_base_offset_pair<2,  72,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  72,  1, 2>::value, vnode_base_offset_pair<2,  72,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  72,  1, 3>::value, vnode_base_offset_pair<2,  72,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  72,  1, 4>::value, vnode_base_offset_pair<2,  72,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  72,  2, 0>::value, vnode_base_offset_pair<2,  72,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2,  72,  2, 1>::value, vnode_base_offset_pair<2,  72,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  72,  2, 2>::value, vnode_base_offset_pair<2,  72,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  72,  2, 3>::value, vnode_base_offset_pair<2,  72,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  72,  3, 0>::value, vnode_base_offset_pair<2,  72,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2,  72,  3, 1>::value, vnode_base_offset_pair<2,  72,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  72,  3, 2>::value, vnode_base_offset_pair<2,  72,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  72,  3, 3>::value, vnode_base_offset_pair<2,  72,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  72,  3, 4>::value, vnode_base_offset_pair<2,  72,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  72,  4, 0>::value, vnode_base_offset_pair<2,  72,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2,  72,  4, 1>::value, vnode_base_offset_pair<2,  72,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72,  5, 0>::value, vnode_base_offset_pair<2,  72,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2,  72,  5, 1>::value, vnode_base_offset_pair<2,  72,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  72,  5, 2>::value, vnode_base_offset_pair<2,  72,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  72,  6, 0>::value, vnode_base_offset_pair<2,  72,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2,  72,  6, 1>::value, vnode_base_offset_pair<2,  72,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  72,  6, 2>::value, vnode_base_offset_pair<2,  72,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  72,  7, 0>::value, vnode_base_offset_pair<2,  72,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2,  72,  7, 1>::value, vnode_base_offset_pair<2,  72,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  72,  7, 2>::value, vnode_base_offset_pair<2,  72,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  72,  8, 0>::value, vnode_base_offset_pair<2,  72,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2,  72,  8, 1>::value, vnode_base_offset_pair<2,  72,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72,  9, 0>::value, vnode_base_offset_pair<2,  72,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2,  72,  9, 1>::value, vnode_base_offset_pair<2,  72,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  72,  9, 2>::value, vnode_base_offset_pair<2,  72,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 10, 0>::value, vnode_base_offset_pair<2,  72, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2,  72, 10, 1>::value, vnode_base_offset_pair<2,  72, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  72, 10, 2>::value, vnode_base_offset_pair<2,  72, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 11, 0>::value, vnode_base_offset_pair<2,  72, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2,  72, 11, 1>::value, vnode_base_offset_pair<2,  72, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  72, 11, 2>::value, vnode_base_offset_pair<2,  72, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 12, 0>::value, vnode_base_offset_pair<2,  72, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2,  72, 12, 1>::value, vnode_base_offset_pair<2,  72, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 13, 0>::value, vnode_base_offset_pair<2,  72, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2,  72, 13, 1>::value, vnode_base_offset_pair<2,  72, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  72, 13, 2>::value, vnode_base_offset_pair<2,  72, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 14, 0>::value, vnode_base_offset_pair<2,  72, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2,  72, 14, 1>::value, vnode_base_offset_pair<2,  72, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  72, 14, 2>::value, vnode_base_offset_pair<2,  72, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 15, 0>::value, vnode_base_offset_pair<2,  72, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2,  72, 15, 1>::value, vnode_base_offset_pair<2,  72, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 16, 0>::value, vnode_base_offset_pair<2,  72, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2,  72, 16, 1>::value, vnode_base_offset_pair<2,  72, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  72, 16, 2>::value, vnode_base_offset_pair<2,  72, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 17, 0>::value, vnode_base_offset_pair<2,  72, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2,  72, 17, 1>::value, vnode_base_offset_pair<2,  72, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  72, 17, 2>::value, vnode_base_offset_pair<2,  72, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 18, 0>::value, vnode_base_offset_pair<2,  72, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2,  72, 18, 1>::value, vnode_base_offset_pair<2,  72, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 19, 0>::value, vnode_base_offset_pair<2,  72, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2,  72, 19, 1>::value, vnode_base_offset_pair<2,  72, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 20, 0>::value, vnode_base_offset_pair<2,  72, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2,  72, 20, 1>::value, vnode_base_offset_pair<2,  72, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 21, 0>::value, vnode_base_offset_pair<2,  72, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2,  72, 21, 1>::value, vnode_base_offset_pair<2,  72, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 22, 0>::value, vnode_base_offset_pair<2,  72, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2,  72, 22, 1>::value, vnode_base_offset_pair<2,  72, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 23, 0>::value, vnode_base_offset_pair<2,  72, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2,  72, 23, 1>::value, vnode_base_offset_pair<2,  72, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 24, 0>::value, vnode_base_offset_pair<2,  72, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2,  72, 24, 1>::value, vnode_base_offset_pair<2,  72, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 25, 0>::value, vnode_base_offset_pair<2,  72, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2,  72, 25, 1>::value, vnode_base_offset_pair<2,  72, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 26, 0>::value, vnode_base_offset_pair<2,  72, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2,  72, 26, 1>::value, vnode_base_offset_pair<2,  72, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  72, 26, 2>::value, vnode_base_offset_pair<2,  72, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 27, 0>::value, vnode_base_offset_pair<2,  72, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2,  72, 27, 1>::value, vnode_base_offset_pair<2,  72, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 28, 0>::value, vnode_base_offset_pair<2,  72, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2,  72, 28, 1>::value, vnode_base_offset_pair<2,  72, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 29, 0>::value, vnode_base_offset_pair<2,  72, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2,  72, 29, 1>::value, vnode_base_offset_pair<2,  72, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 30, 0>::value, vnode_base_offset_pair<2,  72, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2,  72, 30, 1>::value, vnode_base_offset_pair<2,  72, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  72, 30, 2>::value, vnode_base_offset_pair<2,  72, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 31, 0>::value, vnode_base_offset_pair<2,  72, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2,  72, 31, 1>::value, vnode_base_offset_pair<2,  72, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 32, 0>::value, vnode_base_offset_pair<2,  72, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2,  72, 32, 1>::value, vnode_base_offset_pair<2,  72, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 33, 0>::value, vnode_base_offset_pair<2,  72, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2,  72, 33, 1>::value, vnode_base_offset_pair<2,  72, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 34, 0>::value, vnode_base_offset_pair<2,  72, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2,  72, 34, 1>::value, vnode_base_offset_pair<2,  72, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 35, 0>::value, vnode_base_offset_pair<2,  72, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2,  72, 35, 1>::value, vnode_base_offset_pair<2,  72, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 36, 0>::value, vnode_base_offset_pair<2,  72, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2,  72, 36, 1>::value, vnode_base_offset_pair<2,  72, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 37, 0>::value, vnode_base_offset_pair<2,  72, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2,  72, 37, 1>::value, vnode_base_offset_pair<2,  72, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 38, 0>::value, vnode_base_offset_pair<2,  72, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2,  72, 38, 1>::value, vnode_base_offset_pair<2,  72, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 39, 0>::value, vnode_base_offset_pair<2,  72, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2,  72, 39, 1>::value, vnode_base_offset_pair<2,  72, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 40, 0>::value, vnode_base_offset_pair<2,  72, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2,  72, 40, 1>::value, vnode_base_offset_pair<2,  72, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  72, 41, 0>::value, vnode_base_offset_pair<2,  72, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2,  72, 41, 1>::value, vnode_base_offset_pair<2,  72, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z80_8 =
{
    {
        { vnode_shift_mod_pair<2,  80,  0, 0>::value, vnode_base_offset_pair<2,  80,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2,  80,  0, 1>::value, vnode_base_offset_pair<2,  80,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  80,  0, 2>::value, vnode_base_offset_pair<2,  80,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  80,  0, 3>::value, vnode_base_offset_pair<2,  80,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  80,  1, 0>::value, vnode_base_offset_pair<2,  80,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2,  80,  1, 1>::value, vnode_base_offset_pair<2,  80,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  80,  1, 2>::value, vnode_base_offset_pair<2,  80,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  80,  1, 3>::value, vnode_base_offset_pair<2,  80,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  80,  1, 4>::value, vnode_base_offset_pair<2,  80,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  80,  2, 0>::value, vnode_base_offset_pair<2,  80,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2,  80,  2, 1>::value, vnode_base_offset_pair<2,  80,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  80,  2, 2>::value, vnode_base_offset_pair<2,  80,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  80,  2, 3>::value, vnode_base_offset_pair<2,  80,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  80,  3, 0>::value, vnode_base_offset_pair<2,  80,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2,  80,  3, 1>::value, vnode_base_offset_pair<2,  80,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  80,  3, 2>::value, vnode_base_offset_pair<2,  80,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  80,  3, 3>::value, vnode_base_offset_pair<2,  80,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  80,  3, 4>::value, vnode_base_offset_pair<2,  80,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  80,  4, 0>::value, vnode_base_offset_pair<2,  80,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2,  80,  4, 1>::value, vnode_base_offset_pair<2,  80,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80,  5, 0>::value, vnode_base_offset_pair<2,  80,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2,  80,  5, 1>::value, vnode_base_offset_pair<2,  80,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  80,  5, 2>::value, vnode_base_offset_pair<2,  80,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  80,  6, 0>::value, vnode_base_offset_pair<2,  80,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2,  80,  6, 1>::value, vnode_base_offset_pair<2,  80,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  80,  6, 2>::value, vnode_base_offset_pair<2,  80,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  80,  7, 0>::value, vnode_base_offset_pair<2,  80,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2,  80,  7, 1>::value, vnode_base_offset_pair<2,  80,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  80,  7, 2>::value, vnode_base_offset_pair<2,  80,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  80,  8, 0>::value, vnode_base_offset_pair<2,  80,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2,  80,  8, 1>::value, vnode_base_offset_pair<2,  80,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80,  9, 0>::value, vnode_base_offset_pair<2,  80,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2,  80,  9, 1>::value, vnode_base_offset_pair<2,  80,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  80,  9, 2>::value, vnode_base_offset_pair<2,  80,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 10, 0>::value, vnode_base_offset_pair<2,  80, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2,  80, 10, 1>::value, vnode_base_offset_pair<2,  80, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  80, 10, 2>::value, vnode_base_offset_pair<2,  80, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 11, 0>::value, vnode_base_offset_pair<2,  80, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2,  80, 11, 1>::value, vnode_base_offset_pair<2,  80, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  80, 11, 2>::value, vnode_base_offset_pair<2,  80, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 12, 0>::value, vnode_base_offset_pair<2,  80, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2,  80, 12, 1>::value, vnode_base_offset_pair<2,  80, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 13, 0>::value, vnode_base_offset_pair<2,  80, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2,  80, 13, 1>::value, vnode_base_offset_pair<2,  80, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  80, 13, 2>::value, vnode_base_offset_pair<2,  80, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 14, 0>::value, vnode_base_offset_pair<2,  80, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2,  80, 14, 1>::value, vnode_base_offset_pair<2,  80, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  80, 14, 2>::value, vnode_base_offset_pair<2,  80, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 15, 0>::value, vnode_base_offset_pair<2,  80, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2,  80, 15, 1>::value, vnode_base_offset_pair<2,  80, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 16, 0>::value, vnode_base_offset_pair<2,  80, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2,  80, 16, 1>::value, vnode_base_offset_pair<2,  80, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  80, 16, 2>::value, vnode_base_offset_pair<2,  80, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 17, 0>::value, vnode_base_offset_pair<2,  80, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2,  80, 17, 1>::value, vnode_base_offset_pair<2,  80, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  80, 17, 2>::value, vnode_base_offset_pair<2,  80, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 18, 0>::value, vnode_base_offset_pair<2,  80, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2,  80, 18, 1>::value, vnode_base_offset_pair<2,  80, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 19, 0>::value, vnode_base_offset_pair<2,  80, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2,  80, 19, 1>::value, vnode_base_offset_pair<2,  80, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 20, 0>::value, vnode_base_offset_pair<2,  80, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2,  80, 20, 1>::value, vnode_base_offset_pair<2,  80, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 21, 0>::value, vnode_base_offset_pair<2,  80, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2,  80, 21, 1>::value, vnode_base_offset_pair<2,  80, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 22, 0>::value, vnode_base_offset_pair<2,  80, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2,  80, 22, 1>::value, vnode_base_offset_pair<2,  80, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 23, 0>::value, vnode_base_offset_pair<2,  80, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2,  80, 23, 1>::value, vnode_base_offset_pair<2,  80, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 24, 0>::value, vnode_base_offset_pair<2,  80, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2,  80, 24, 1>::value, vnode_base_offset_pair<2,  80, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 25, 0>::value, vnode_base_offset_pair<2,  80, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2,  80, 25, 1>::value, vnode_base_offset_pair<2,  80, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 26, 0>::value, vnode_base_offset_pair<2,  80, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2,  80, 26, 1>::value, vnode_base_offset_pair<2,  80, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  80, 26, 2>::value, vnode_base_offset_pair<2,  80, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 27, 0>::value, vnode_base_offset_pair<2,  80, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2,  80, 27, 1>::value, vnode_base_offset_pair<2,  80, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 28, 0>::value, vnode_base_offset_pair<2,  80, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2,  80, 28, 1>::value, vnode_base_offset_pair<2,  80, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 29, 0>::value, vnode_base_offset_pair<2,  80, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2,  80, 29, 1>::value, vnode_base_offset_pair<2,  80, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 30, 0>::value, vnode_base_offset_pair<2,  80, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2,  80, 30, 1>::value, vnode_base_offset_pair<2,  80, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  80, 30, 2>::value, vnode_base_offset_pair<2,  80, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 31, 0>::value, vnode_base_offset_pair<2,  80, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2,  80, 31, 1>::value, vnode_base_offset_pair<2,  80, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 32, 0>::value, vnode_base_offset_pair<2,  80, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2,  80, 32, 1>::value, vnode_base_offset_pair<2,  80, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 33, 0>::value, vnode_base_offset_pair<2,  80, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2,  80, 33, 1>::value, vnode_base_offset_pair<2,  80, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 34, 0>::value, vnode_base_offset_pair<2,  80, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2,  80, 34, 1>::value, vnode_base_offset_pair<2,  80, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 35, 0>::value, vnode_base_offset_pair<2,  80, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2,  80, 35, 1>::value, vnode_base_offset_pair<2,  80, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 36, 0>::value, vnode_base_offset_pair<2,  80, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2,  80, 36, 1>::value, vnode_base_offset_pair<2,  80, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 37, 0>::value, vnode_base_offset_pair<2,  80, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2,  80, 37, 1>::value, vnode_base_offset_pair<2,  80, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 38, 0>::value, vnode_base_offset_pair<2,  80, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2,  80, 38, 1>::value, vnode_base_offset_pair<2,  80, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 39, 0>::value, vnode_base_offset_pair<2,  80, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2,  80, 39, 1>::value, vnode_base_offset_pair<2,  80, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 40, 0>::value, vnode_base_offset_pair<2,  80, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2,  80, 40, 1>::value, vnode_base_offset_pair<2,  80, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  80, 41, 0>::value, vnode_base_offset_pair<2,  80, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2,  80, 41, 1>::value, vnode_base_offset_pair<2,  80, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z88_8 =
{
    {
        { vnode_shift_mod_pair<2,  88,  0, 0>::value, vnode_base_offset_pair<2,  88,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2,  88,  0, 1>::value, vnode_base_offset_pair<2,  88,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  88,  0, 2>::value, vnode_base_offset_pair<2,  88,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  88,  0, 3>::value, vnode_base_offset_pair<2,  88,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  88,  1, 0>::value, vnode_base_offset_pair<2,  88,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2,  88,  1, 1>::value, vnode_base_offset_pair<2,  88,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  88,  1, 2>::value, vnode_base_offset_pair<2,  88,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  88,  1, 3>::value, vnode_base_offset_pair<2,  88,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  88,  1, 4>::value, vnode_base_offset_pair<2,  88,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  88,  2, 0>::value, vnode_base_offset_pair<2,  88,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2,  88,  2, 1>::value, vnode_base_offset_pair<2,  88,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  88,  2, 2>::value, vnode_base_offset_pair<2,  88,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  88,  2, 3>::value, vnode_base_offset_pair<2,  88,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  88,  3, 0>::value, vnode_base_offset_pair<2,  88,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2,  88,  3, 1>::value, vnode_base_offset_pair<2,  88,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  88,  3, 2>::value, vnode_base_offset_pair<2,  88,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  88,  3, 3>::value, vnode_base_offset_pair<2,  88,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  88,  3, 4>::value, vnode_base_offset_pair<2,  88,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  88,  4, 0>::value, vnode_base_offset_pair<2,  88,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2,  88,  4, 1>::value, vnode_base_offset_pair<2,  88,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88,  5, 0>::value, vnode_base_offset_pair<2,  88,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2,  88,  5, 1>::value, vnode_base_offset_pair<2,  88,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  88,  5, 2>::value, vnode_base_offset_pair<2,  88,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  88,  6, 0>::value, vnode_base_offset_pair<2,  88,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2,  88,  6, 1>::value, vnode_base_offset_pair<2,  88,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  88,  6, 2>::value, vnode_base_offset_pair<2,  88,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  88,  7, 0>::value, vnode_base_offset_pair<2,  88,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2,  88,  7, 1>::value, vnode_base_offset_pair<2,  88,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  88,  7, 2>::value, vnode_base_offset_pair<2,  88,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  88,  8, 0>::value, vnode_base_offset_pair<2,  88,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2,  88,  8, 1>::value, vnode_base_offset_pair<2,  88,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88,  9, 0>::value, vnode_base_offset_pair<2,  88,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2,  88,  9, 1>::value, vnode_base_offset_pair<2,  88,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  88,  9, 2>::value, vnode_base_offset_pair<2,  88,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 10, 0>::value, vnode_base_offset_pair<2,  88, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2,  88, 10, 1>::value, vnode_base_offset_pair<2,  88, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  88, 10, 2>::value, vnode_base_offset_pair<2,  88, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 11, 0>::value, vnode_base_offset_pair<2,  88, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2,  88, 11, 1>::value, vnode_base_offset_pair<2,  88, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  88, 11, 2>::value, vnode_base_offset_pair<2,  88, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 12, 0>::value, vnode_base_offset_pair<2,  88, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2,  88, 12, 1>::value, vnode_base_offset_pair<2,  88, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 13, 0>::value, vnode_base_offset_pair<2,  88, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2,  88, 13, 1>::value, vnode_base_offset_pair<2,  88, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  88, 13, 2>::value, vnode_base_offset_pair<2,  88, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 14, 0>::value, vnode_base_offset_pair<2,  88, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2,  88, 14, 1>::value, vnode_base_offset_pair<2,  88, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  88, 14, 2>::value, vnode_base_offset_pair<2,  88, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 15, 0>::value, vnode_base_offset_pair<2,  88, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2,  88, 15, 1>::value, vnode_base_offset_pair<2,  88, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 16, 0>::value, vnode_base_offset_pair<2,  88, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2,  88, 16, 1>::value, vnode_base_offset_pair<2,  88, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  88, 16, 2>::value, vnode_base_offset_pair<2,  88, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 17, 0>::value, vnode_base_offset_pair<2,  88, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2,  88, 17, 1>::value, vnode_base_offset_pair<2,  88, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  88, 17, 2>::value, vnode_base_offset_pair<2,  88, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 18, 0>::value, vnode_base_offset_pair<2,  88, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2,  88, 18, 1>::value, vnode_base_offset_pair<2,  88, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 19, 0>::value, vnode_base_offset_pair<2,  88, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2,  88, 19, 1>::value, vnode_base_offset_pair<2,  88, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 20, 0>::value, vnode_base_offset_pair<2,  88, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2,  88, 20, 1>::value, vnode_base_offset_pair<2,  88, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 21, 0>::value, vnode_base_offset_pair<2,  88, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2,  88, 21, 1>::value, vnode_base_offset_pair<2,  88, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 22, 0>::value, vnode_base_offset_pair<2,  88, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2,  88, 22, 1>::value, vnode_base_offset_pair<2,  88, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 23, 0>::value, vnode_base_offset_pair<2,  88, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2,  88, 23, 1>::value, vnode_base_offset_pair<2,  88, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 24, 0>::value, vnode_base_offset_pair<2,  88, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2,  88, 24, 1>::value, vnode_base_offset_pair<2,  88, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 25, 0>::value, vnode_base_offset_pair<2,  88, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2,  88, 25, 1>::value, vnode_base_offset_pair<2,  88, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 26, 0>::value, vnode_base_offset_pair<2,  88, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2,  88, 26, 1>::value, vnode_base_offset_pair<2,  88, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  88, 26, 2>::value, vnode_base_offset_pair<2,  88, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 27, 0>::value, vnode_base_offset_pair<2,  88, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2,  88, 27, 1>::value, vnode_base_offset_pair<2,  88, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 28, 0>::value, vnode_base_offset_pair<2,  88, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2,  88, 28, 1>::value, vnode_base_offset_pair<2,  88, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 29, 0>::value, vnode_base_offset_pair<2,  88, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2,  88, 29, 1>::value, vnode_base_offset_pair<2,  88, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 30, 0>::value, vnode_base_offset_pair<2,  88, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2,  88, 30, 1>::value, vnode_base_offset_pair<2,  88, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  88, 30, 2>::value, vnode_base_offset_pair<2,  88, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 31, 0>::value, vnode_base_offset_pair<2,  88, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2,  88, 31, 1>::value, vnode_base_offset_pair<2,  88, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 32, 0>::value, vnode_base_offset_pair<2,  88, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2,  88, 32, 1>::value, vnode_base_offset_pair<2,  88, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 33, 0>::value, vnode_base_offset_pair<2,  88, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2,  88, 33, 1>::value, vnode_base_offset_pair<2,  88, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 34, 0>::value, vnode_base_offset_pair<2,  88, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2,  88, 34, 1>::value, vnode_base_offset_pair<2,  88, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 35, 0>::value, vnode_base_offset_pair<2,  88, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2,  88, 35, 1>::value, vnode_base_offset_pair<2,  88, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 36, 0>::value, vnode_base_offset_pair<2,  88, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2,  88, 36, 1>::value, vnode_base_offset_pair<2,  88, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 37, 0>::value, vnode_base_offset_pair<2,  88, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2,  88, 37, 1>::value, vnode_base_offset_pair<2,  88, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 38, 0>::value, vnode_base_offset_pair<2,  88, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2,  88, 38, 1>::value, vnode_base_offset_pair<2,  88, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 39, 0>::value, vnode_base_offset_pair<2,  88, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2,  88, 39, 1>::value, vnode_base_offset_pair<2,  88, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 40, 0>::value, vnode_base_offset_pair<2,  88, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2,  88, 40, 1>::value, vnode_base_offset_pair<2,  88, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  88, 41, 0>::value, vnode_base_offset_pair<2,  88, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2,  88, 41, 1>::value, vnode_base_offset_pair<2,  88, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z96_8 =
{
    {
        { vnode_shift_mod_pair<2,  96,  0, 0>::value, vnode_base_offset_pair<2,  96,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2,  96,  0, 1>::value, vnode_base_offset_pair<2,  96,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  96,  0, 2>::value, vnode_base_offset_pair<2,  96,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  96,  0, 3>::value, vnode_base_offset_pair<2,  96,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  96,  1, 0>::value, vnode_base_offset_pair<2,  96,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2,  96,  1, 1>::value, vnode_base_offset_pair<2,  96,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  96,  1, 2>::value, vnode_base_offset_pair<2,  96,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  96,  1, 3>::value, vnode_base_offset_pair<2,  96,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  96,  1, 4>::value, vnode_base_offset_pair<2,  96,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  96,  2, 0>::value, vnode_base_offset_pair<2,  96,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2,  96,  2, 1>::value, vnode_base_offset_pair<2,  96,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  96,  2, 2>::value, vnode_base_offset_pair<2,  96,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  96,  2, 3>::value, vnode_base_offset_pair<2,  96,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2,  96,  3, 0>::value, vnode_base_offset_pair<2,  96,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2,  96,  3, 1>::value, vnode_base_offset_pair<2,  96,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  96,  3, 2>::value, vnode_base_offset_pair<2,  96,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2,  96,  3, 3>::value, vnode_base_offset_pair<2,  96,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2,  96,  3, 4>::value, vnode_base_offset_pair<2,  96,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2,  96,  4, 0>::value, vnode_base_offset_pair<2,  96,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2,  96,  4, 1>::value, vnode_base_offset_pair<2,  96,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96,  5, 0>::value, vnode_base_offset_pair<2,  96,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2,  96,  5, 1>::value, vnode_base_offset_pair<2,  96,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  96,  5, 2>::value, vnode_base_offset_pair<2,  96,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  96,  6, 0>::value, vnode_base_offset_pair<2,  96,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2,  96,  6, 1>::value, vnode_base_offset_pair<2,  96,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  96,  6, 2>::value, vnode_base_offset_pair<2,  96,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  96,  7, 0>::value, vnode_base_offset_pair<2,  96,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2,  96,  7, 1>::value, vnode_base_offset_pair<2,  96,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  96,  7, 2>::value, vnode_base_offset_pair<2,  96,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  96,  8, 0>::value, vnode_base_offset_pair<2,  96,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2,  96,  8, 1>::value, vnode_base_offset_pair<2,  96,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96,  9, 0>::value, vnode_base_offset_pair<2,  96,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2,  96,  9, 1>::value, vnode_base_offset_pair<2,  96,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  96,  9, 2>::value, vnode_base_offset_pair<2,  96,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 10, 0>::value, vnode_base_offset_pair<2,  96, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2,  96, 10, 1>::value, vnode_base_offset_pair<2,  96, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  96, 10, 2>::value, vnode_base_offset_pair<2,  96, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 11, 0>::value, vnode_base_offset_pair<2,  96, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2,  96, 11, 1>::value, vnode_base_offset_pair<2,  96, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  96, 11, 2>::value, vnode_base_offset_pair<2,  96, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 12, 0>::value, vnode_base_offset_pair<2,  96, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2,  96, 12, 1>::value, vnode_base_offset_pair<2,  96, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 13, 0>::value, vnode_base_offset_pair<2,  96, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2,  96, 13, 1>::value, vnode_base_offset_pair<2,  96, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  96, 13, 2>::value, vnode_base_offset_pair<2,  96, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 14, 0>::value, vnode_base_offset_pair<2,  96, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2,  96, 14, 1>::value, vnode_base_offset_pair<2,  96, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  96, 14, 2>::value, vnode_base_offset_pair<2,  96, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 15, 0>::value, vnode_base_offset_pair<2,  96, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2,  96, 15, 1>::value, vnode_base_offset_pair<2,  96, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 16, 0>::value, vnode_base_offset_pair<2,  96, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2,  96, 16, 1>::value, vnode_base_offset_pair<2,  96, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  96, 16, 2>::value, vnode_base_offset_pair<2,  96, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 17, 0>::value, vnode_base_offset_pair<2,  96, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2,  96, 17, 1>::value, vnode_base_offset_pair<2,  96, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  96, 17, 2>::value, vnode_base_offset_pair<2,  96, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 18, 0>::value, vnode_base_offset_pair<2,  96, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2,  96, 18, 1>::value, vnode_base_offset_pair<2,  96, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 19, 0>::value, vnode_base_offset_pair<2,  96, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2,  96, 19, 1>::value, vnode_base_offset_pair<2,  96, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 20, 0>::value, vnode_base_offset_pair<2,  96, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2,  96, 20, 1>::value, vnode_base_offset_pair<2,  96, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 21, 0>::value, vnode_base_offset_pair<2,  96, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2,  96, 21, 1>::value, vnode_base_offset_pair<2,  96, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 22, 0>::value, vnode_base_offset_pair<2,  96, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2,  96, 22, 1>::value, vnode_base_offset_pair<2,  96, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 23, 0>::value, vnode_base_offset_pair<2,  96, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2,  96, 23, 1>::value, vnode_base_offset_pair<2,  96, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 24, 0>::value, vnode_base_offset_pair<2,  96, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2,  96, 24, 1>::value, vnode_base_offset_pair<2,  96, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 25, 0>::value, vnode_base_offset_pair<2,  96, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2,  96, 25, 1>::value, vnode_base_offset_pair<2,  96, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 26, 0>::value, vnode_base_offset_pair<2,  96, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2,  96, 26, 1>::value, vnode_base_offset_pair<2,  96, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  96, 26, 2>::value, vnode_base_offset_pair<2,  96, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 27, 0>::value, vnode_base_offset_pair<2,  96, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2,  96, 27, 1>::value, vnode_base_offset_pair<2,  96, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 28, 0>::value, vnode_base_offset_pair<2,  96, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2,  96, 28, 1>::value, vnode_base_offset_pair<2,  96, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 29, 0>::value, vnode_base_offset_pair<2,  96, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2,  96, 29, 1>::value, vnode_base_offset_pair<2,  96, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 30, 0>::value, vnode_base_offset_pair<2,  96, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2,  96, 30, 1>::value, vnode_base_offset_pair<2,  96, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2,  96, 30, 2>::value, vnode_base_offset_pair<2,  96, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 31, 0>::value, vnode_base_offset_pair<2,  96, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2,  96, 31, 1>::value, vnode_base_offset_pair<2,  96, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 32, 0>::value, vnode_base_offset_pair<2,  96, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2,  96, 32, 1>::value, vnode_base_offset_pair<2,  96, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 33, 0>::value, vnode_base_offset_pair<2,  96, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2,  96, 33, 1>::value, vnode_base_offset_pair<2,  96, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 34, 0>::value, vnode_base_offset_pair<2,  96, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2,  96, 34, 1>::value, vnode_base_offset_pair<2,  96, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 35, 0>::value, vnode_base_offset_pair<2,  96, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2,  96, 35, 1>::value, vnode_base_offset_pair<2,  96, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 36, 0>::value, vnode_base_offset_pair<2,  96, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2,  96, 36, 1>::value, vnode_base_offset_pair<2,  96, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 37, 0>::value, vnode_base_offset_pair<2,  96, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2,  96, 37, 1>::value, vnode_base_offset_pair<2,  96, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 38, 0>::value, vnode_base_offset_pair<2,  96, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2,  96, 38, 1>::value, vnode_base_offset_pair<2,  96, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 39, 0>::value, vnode_base_offset_pair<2,  96, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2,  96, 39, 1>::value, vnode_base_offset_pair<2,  96, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 40, 0>::value, vnode_base_offset_pair<2,  96, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2,  96, 40, 1>::value, vnode_base_offset_pair<2,  96, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2,  96, 41, 0>::value, vnode_base_offset_pair<2,  96, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2,  96, 41, 1>::value, vnode_base_offset_pair<2,  96, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z104_8 =
{
    {
        { vnode_shift_mod_pair<2, 104,  0, 0>::value, vnode_base_offset_pair<2, 104,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2, 104,  0, 1>::value, vnode_base_offset_pair<2, 104,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 104,  0, 2>::value, vnode_base_offset_pair<2, 104,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 104,  0, 3>::value, vnode_base_offset_pair<2, 104,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 104,  1, 0>::value, vnode_base_offset_pair<2, 104,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2, 104,  1, 1>::value, vnode_base_offset_pair<2, 104,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 104,  1, 2>::value, vnode_base_offset_pair<2, 104,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 104,  1, 3>::value, vnode_base_offset_pair<2, 104,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 104,  1, 4>::value, vnode_base_offset_pair<2, 104,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 104,  2, 0>::value, vnode_base_offset_pair<2, 104,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2, 104,  2, 1>::value, vnode_base_offset_pair<2, 104,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 104,  2, 2>::value, vnode_base_offset_pair<2, 104,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 104,  2, 3>::value, vnode_base_offset_pair<2, 104,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 104,  3, 0>::value, vnode_base_offset_pair<2, 104,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2, 104,  3, 1>::value, vnode_base_offset_pair<2, 104,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 104,  3, 2>::value, vnode_base_offset_pair<2, 104,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 104,  3, 3>::value, vnode_base_offset_pair<2, 104,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 104,  3, 4>::value, vnode_base_offset_pair<2, 104,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 104,  4, 0>::value, vnode_base_offset_pair<2, 104,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2, 104,  4, 1>::value, vnode_base_offset_pair<2, 104,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104,  5, 0>::value, vnode_base_offset_pair<2, 104,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2, 104,  5, 1>::value, vnode_base_offset_pair<2, 104,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 104,  5, 2>::value, vnode_base_offset_pair<2, 104,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 104,  6, 0>::value, vnode_base_offset_pair<2, 104,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2, 104,  6, 1>::value, vnode_base_offset_pair<2, 104,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 104,  6, 2>::value, vnode_base_offset_pair<2, 104,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 104,  7, 0>::value, vnode_base_offset_pair<2, 104,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2, 104,  7, 1>::value, vnode_base_offset_pair<2, 104,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 104,  7, 2>::value, vnode_base_offset_pair<2, 104,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 104,  8, 0>::value, vnode_base_offset_pair<2, 104,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2, 104,  8, 1>::value, vnode_base_offset_pair<2, 104,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104,  9, 0>::value, vnode_base_offset_pair<2, 104,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2, 104,  9, 1>::value, vnode_base_offset_pair<2, 104,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 104,  9, 2>::value, vnode_base_offset_pair<2, 104,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 10, 0>::value, vnode_base_offset_pair<2, 104, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2, 104, 10, 1>::value, vnode_base_offset_pair<2, 104, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 104, 10, 2>::value, vnode_base_offset_pair<2, 104, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 11, 0>::value, vnode_base_offset_pair<2, 104, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2, 104, 11, 1>::value, vnode_base_offset_pair<2, 104, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 104, 11, 2>::value, vnode_base_offset_pair<2, 104, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 12, 0>::value, vnode_base_offset_pair<2, 104, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2, 104, 12, 1>::value, vnode_base_offset_pair<2, 104, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 13, 0>::value, vnode_base_offset_pair<2, 104, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2, 104, 13, 1>::value, vnode_base_offset_pair<2, 104, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 104, 13, 2>::value, vnode_base_offset_pair<2, 104, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 14, 0>::value, vnode_base_offset_pair<2, 104, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2, 104, 14, 1>::value, vnode_base_offset_pair<2, 104, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 104, 14, 2>::value, vnode_base_offset_pair<2, 104, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 15, 0>::value, vnode_base_offset_pair<2, 104, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2, 104, 15, 1>::value, vnode_base_offset_pair<2, 104, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 16, 0>::value, vnode_base_offset_pair<2, 104, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2, 104, 16, 1>::value, vnode_base_offset_pair<2, 104, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 104, 16, 2>::value, vnode_base_offset_pair<2, 104, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 17, 0>::value, vnode_base_offset_pair<2, 104, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2, 104, 17, 1>::value, vnode_base_offset_pair<2, 104, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 104, 17, 2>::value, vnode_base_offset_pair<2, 104, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 18, 0>::value, vnode_base_offset_pair<2, 104, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2, 104, 18, 1>::value, vnode_base_offset_pair<2, 104, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 19, 0>::value, vnode_base_offset_pair<2, 104, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2, 104, 19, 1>::value, vnode_base_offset_pair<2, 104, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 20, 0>::value, vnode_base_offset_pair<2, 104, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2, 104, 20, 1>::value, vnode_base_offset_pair<2, 104, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 21, 0>::value, vnode_base_offset_pair<2, 104, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2, 104, 21, 1>::value, vnode_base_offset_pair<2, 104, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 22, 0>::value, vnode_base_offset_pair<2, 104, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2, 104, 22, 1>::value, vnode_base_offset_pair<2, 104, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 23, 0>::value, vnode_base_offset_pair<2, 104, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2, 104, 23, 1>::value, vnode_base_offset_pair<2, 104, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 24, 0>::value, vnode_base_offset_pair<2, 104, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2, 104, 24, 1>::value, vnode_base_offset_pair<2, 104, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 25, 0>::value, vnode_base_offset_pair<2, 104, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2, 104, 25, 1>::value, vnode_base_offset_pair<2, 104, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 26, 0>::value, vnode_base_offset_pair<2, 104, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2, 104, 26, 1>::value, vnode_base_offset_pair<2, 104, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 104, 26, 2>::value, vnode_base_offset_pair<2, 104, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 27, 0>::value, vnode_base_offset_pair<2, 104, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2, 104, 27, 1>::value, vnode_base_offset_pair<2, 104, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 28, 0>::value, vnode_base_offset_pair<2, 104, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2, 104, 28, 1>::value, vnode_base_offset_pair<2, 104, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 29, 0>::value, vnode_base_offset_pair<2, 104, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2, 104, 29, 1>::value, vnode_base_offset_pair<2, 104, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 30, 0>::value, vnode_base_offset_pair<2, 104, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2, 104, 30, 1>::value, vnode_base_offset_pair<2, 104, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 104, 30, 2>::value, vnode_base_offset_pair<2, 104, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 31, 0>::value, vnode_base_offset_pair<2, 104, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2, 104, 31, 1>::value, vnode_base_offset_pair<2, 104, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 32, 0>::value, vnode_base_offset_pair<2, 104, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2, 104, 32, 1>::value, vnode_base_offset_pair<2, 104, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 33, 0>::value, vnode_base_offset_pair<2, 104, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2, 104, 33, 1>::value, vnode_base_offset_pair<2, 104, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 34, 0>::value, vnode_base_offset_pair<2, 104, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2, 104, 34, 1>::value, vnode_base_offset_pair<2, 104, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 35, 0>::value, vnode_base_offset_pair<2, 104, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2, 104, 35, 1>::value, vnode_base_offset_pair<2, 104, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 36, 0>::value, vnode_base_offset_pair<2, 104, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2, 104, 36, 1>::value, vnode_base_offset_pair<2, 104, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 37, 0>::value, vnode_base_offset_pair<2, 104, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2, 104, 37, 1>::value, vnode_base_offset_pair<2, 104, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 38, 0>::value, vnode_base_offset_pair<2, 104, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2, 104, 38, 1>::value, vnode_base_offset_pair<2, 104, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 39, 0>::value, vnode_base_offset_pair<2, 104, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2, 104, 39, 1>::value, vnode_base_offset_pair<2, 104, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 40, 0>::value, vnode_base_offset_pair<2, 104, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2, 104, 40, 1>::value, vnode_base_offset_pair<2, 104, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 104, 41, 0>::value, vnode_base_offset_pair<2, 104, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2, 104, 41, 1>::value, vnode_base_offset_pair<2, 104, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z112_8 =
{
    {
        { vnode_shift_mod_pair<2, 112,  0, 0>::value, vnode_base_offset_pair<2, 112,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2, 112,  0, 1>::value, vnode_base_offset_pair<2, 112,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 112,  0, 2>::value, vnode_base_offset_pair<2, 112,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 112,  0, 3>::value, vnode_base_offset_pair<2, 112,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 112,  1, 0>::value, vnode_base_offset_pair<2, 112,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2, 112,  1, 1>::value, vnode_base_offset_pair<2, 112,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 112,  1, 2>::value, vnode_base_offset_pair<2, 112,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 112,  1, 3>::value, vnode_base_offset_pair<2, 112,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 112,  1, 4>::value, vnode_base_offset_pair<2, 112,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 112,  2, 0>::value, vnode_base_offset_pair<2, 112,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2, 112,  2, 1>::value, vnode_base_offset_pair<2, 112,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 112,  2, 2>::value, vnode_base_offset_pair<2, 112,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 112,  2, 3>::value, vnode_base_offset_pair<2, 112,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 112,  3, 0>::value, vnode_base_offset_pair<2, 112,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2, 112,  3, 1>::value, vnode_base_offset_pair<2, 112,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 112,  3, 2>::value, vnode_base_offset_pair<2, 112,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 112,  3, 3>::value, vnode_base_offset_pair<2, 112,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 112,  3, 4>::value, vnode_base_offset_pair<2, 112,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 112,  4, 0>::value, vnode_base_offset_pair<2, 112,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2, 112,  4, 1>::value, vnode_base_offset_pair<2, 112,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112,  5, 0>::value, vnode_base_offset_pair<2, 112,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2, 112,  5, 1>::value, vnode_base_offset_pair<2, 112,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 112,  5, 2>::value, vnode_base_offset_pair<2, 112,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 112,  6, 0>::value, vnode_base_offset_pair<2, 112,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2, 112,  6, 1>::value, vnode_base_offset_pair<2, 112,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 112,  6, 2>::value, vnode_base_offset_pair<2, 112,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 112,  7, 0>::value, vnode_base_offset_pair<2, 112,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2, 112,  7, 1>::value, vnode_base_offset_pair<2, 112,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 112,  7, 2>::value, vnode_base_offset_pair<2, 112,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 112,  8, 0>::value, vnode_base_offset_pair<2, 112,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2, 112,  8, 1>::value, vnode_base_offset_pair<2, 112,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112,  9, 0>::value, vnode_base_offset_pair<2, 112,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2, 112,  9, 1>::value, vnode_base_offset_pair<2, 112,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 112,  9, 2>::value, vnode_base_offset_pair<2, 112,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 10, 0>::value, vnode_base_offset_pair<2, 112, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2, 112, 10, 1>::value, vnode_base_offset_pair<2, 112, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 112, 10, 2>::value, vnode_base_offset_pair<2, 112, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 11, 0>::value, vnode_base_offset_pair<2, 112, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2, 112, 11, 1>::value, vnode_base_offset_pair<2, 112, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 112, 11, 2>::value, vnode_base_offset_pair<2, 112, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 12, 0>::value, vnode_base_offset_pair<2, 112, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2, 112, 12, 1>::value, vnode_base_offset_pair<2, 112, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 13, 0>::value, vnode_base_offset_pair<2, 112, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2, 112, 13, 1>::value, vnode_base_offset_pair<2, 112, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 112, 13, 2>::value, vnode_base_offset_pair<2, 112, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 14, 0>::value, vnode_base_offset_pair<2, 112, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2, 112, 14, 1>::value, vnode_base_offset_pair<2, 112, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 112, 14, 2>::value, vnode_base_offset_pair<2, 112, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 15, 0>::value, vnode_base_offset_pair<2, 112, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2, 112, 15, 1>::value, vnode_base_offset_pair<2, 112, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 16, 0>::value, vnode_base_offset_pair<2, 112, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2, 112, 16, 1>::value, vnode_base_offset_pair<2, 112, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 112, 16, 2>::value, vnode_base_offset_pair<2, 112, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 17, 0>::value, vnode_base_offset_pair<2, 112, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2, 112, 17, 1>::value, vnode_base_offset_pair<2, 112, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 112, 17, 2>::value, vnode_base_offset_pair<2, 112, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 18, 0>::value, vnode_base_offset_pair<2, 112, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2, 112, 18, 1>::value, vnode_base_offset_pair<2, 112, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 19, 0>::value, vnode_base_offset_pair<2, 112, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2, 112, 19, 1>::value, vnode_base_offset_pair<2, 112, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 20, 0>::value, vnode_base_offset_pair<2, 112, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2, 112, 20, 1>::value, vnode_base_offset_pair<2, 112, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 21, 0>::value, vnode_base_offset_pair<2, 112, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2, 112, 21, 1>::value, vnode_base_offset_pair<2, 112, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 22, 0>::value, vnode_base_offset_pair<2, 112, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2, 112, 22, 1>::value, vnode_base_offset_pair<2, 112, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 23, 0>::value, vnode_base_offset_pair<2, 112, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2, 112, 23, 1>::value, vnode_base_offset_pair<2, 112, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 24, 0>::value, vnode_base_offset_pair<2, 112, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2, 112, 24, 1>::value, vnode_base_offset_pair<2, 112, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 25, 0>::value, vnode_base_offset_pair<2, 112, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2, 112, 25, 1>::value, vnode_base_offset_pair<2, 112, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 26, 0>::value, vnode_base_offset_pair<2, 112, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2, 112, 26, 1>::value, vnode_base_offset_pair<2, 112, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 112, 26, 2>::value, vnode_base_offset_pair<2, 112, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 27, 0>::value, vnode_base_offset_pair<2, 112, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2, 112, 27, 1>::value, vnode_base_offset_pair<2, 112, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 28, 0>::value, vnode_base_offset_pair<2, 112, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2, 112, 28, 1>::value, vnode_base_offset_pair<2, 112, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 29, 0>::value, vnode_base_offset_pair<2, 112, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2, 112, 29, 1>::value, vnode_base_offset_pair<2, 112, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 30, 0>::value, vnode_base_offset_pair<2, 112, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2, 112, 30, 1>::value, vnode_base_offset_pair<2, 112, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 112, 30, 2>::value, vnode_base_offset_pair<2, 112, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 31, 0>::value, vnode_base_offset_pair<2, 112, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2, 112, 31, 1>::value, vnode_base_offset_pair<2, 112, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 32, 0>::value, vnode_base_offset_pair<2, 112, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2, 112, 32, 1>::value, vnode_base_offset_pair<2, 112, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 33, 0>::value, vnode_base_offset_pair<2, 112, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2, 112, 33, 1>::value, vnode_base_offset_pair<2, 112, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 34, 0>::value, vnode_base_offset_pair<2, 112, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2, 112, 34, 1>::value, vnode_base_offset_pair<2, 112, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 35, 0>::value, vnode_base_offset_pair<2, 112, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2, 112, 35, 1>::value, vnode_base_offset_pair<2, 112, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 36, 0>::value, vnode_base_offset_pair<2, 112, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2, 112, 36, 1>::value, vnode_base_offset_pair<2, 112, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 37, 0>::value, vnode_base_offset_pair<2, 112, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2, 112, 37, 1>::value, vnode_base_offset_pair<2, 112, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 38, 0>::value, vnode_base_offset_pair<2, 112, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2, 112, 38, 1>::value, vnode_base_offset_pair<2, 112, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 39, 0>::value, vnode_base_offset_pair<2, 112, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2, 112, 39, 1>::value, vnode_base_offset_pair<2, 112, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 40, 0>::value, vnode_base_offset_pair<2, 112, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2, 112, 40, 1>::value, vnode_base_offset_pair<2, 112, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 112, 41, 0>::value, vnode_base_offset_pair<2, 112, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2, 112, 41, 1>::value, vnode_base_offset_pair<2, 112, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z120_8 =
{
    {
        { vnode_shift_mod_pair<2, 120,  0, 0>::value, vnode_base_offset_pair<2, 120,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2, 120,  0, 1>::value, vnode_base_offset_pair<2, 120,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 120,  0, 2>::value, vnode_base_offset_pair<2, 120,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 120,  0, 3>::value, vnode_base_offset_pair<2, 120,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 120,  1, 0>::value, vnode_base_offset_pair<2, 120,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2, 120,  1, 1>::value, vnode_base_offset_pair<2, 120,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 120,  1, 2>::value, vnode_base_offset_pair<2, 120,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 120,  1, 3>::value, vnode_base_offset_pair<2, 120,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 120,  1, 4>::value, vnode_base_offset_pair<2, 120,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 120,  2, 0>::value, vnode_base_offset_pair<2, 120,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2, 120,  2, 1>::value, vnode_base_offset_pair<2, 120,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 120,  2, 2>::value, vnode_base_offset_pair<2, 120,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 120,  2, 3>::value, vnode_base_offset_pair<2, 120,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 120,  3, 0>::value, vnode_base_offset_pair<2, 120,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2, 120,  3, 1>::value, vnode_base_offset_pair<2, 120,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 120,  3, 2>::value, vnode_base_offset_pair<2, 120,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 120,  3, 3>::value, vnode_base_offset_pair<2, 120,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 120,  3, 4>::value, vnode_base_offset_pair<2, 120,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 120,  4, 0>::value, vnode_base_offset_pair<2, 120,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2, 120,  4, 1>::value, vnode_base_offset_pair<2, 120,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120,  5, 0>::value, vnode_base_offset_pair<2, 120,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2, 120,  5, 1>::value, vnode_base_offset_pair<2, 120,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 120,  5, 2>::value, vnode_base_offset_pair<2, 120,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 120,  6, 0>::value, vnode_base_offset_pair<2, 120,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2, 120,  6, 1>::value, vnode_base_offset_pair<2, 120,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 120,  6, 2>::value, vnode_base_offset_pair<2, 120,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 120,  7, 0>::value, vnode_base_offset_pair<2, 120,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2, 120,  7, 1>::value, vnode_base_offset_pair<2, 120,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 120,  7, 2>::value, vnode_base_offset_pair<2, 120,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 120,  8, 0>::value, vnode_base_offset_pair<2, 120,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2, 120,  8, 1>::value, vnode_base_offset_pair<2, 120,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120,  9, 0>::value, vnode_base_offset_pair<2, 120,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2, 120,  9, 1>::value, vnode_base_offset_pair<2, 120,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 120,  9, 2>::value, vnode_base_offset_pair<2, 120,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 10, 0>::value, vnode_base_offset_pair<2, 120, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2, 120, 10, 1>::value, vnode_base_offset_pair<2, 120, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 120, 10, 2>::value, vnode_base_offset_pair<2, 120, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 11, 0>::value, vnode_base_offset_pair<2, 120, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2, 120, 11, 1>::value, vnode_base_offset_pair<2, 120, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 120, 11, 2>::value, vnode_base_offset_pair<2, 120, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 12, 0>::value, vnode_base_offset_pair<2, 120, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2, 120, 12, 1>::value, vnode_base_offset_pair<2, 120, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 13, 0>::value, vnode_base_offset_pair<2, 120, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2, 120, 13, 1>::value, vnode_base_offset_pair<2, 120, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 120, 13, 2>::value, vnode_base_offset_pair<2, 120, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 14, 0>::value, vnode_base_offset_pair<2, 120, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2, 120, 14, 1>::value, vnode_base_offset_pair<2, 120, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 120, 14, 2>::value, vnode_base_offset_pair<2, 120, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 15, 0>::value, vnode_base_offset_pair<2, 120, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2, 120, 15, 1>::value, vnode_base_offset_pair<2, 120, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 16, 0>::value, vnode_base_offset_pair<2, 120, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2, 120, 16, 1>::value, vnode_base_offset_pair<2, 120, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 120, 16, 2>::value, vnode_base_offset_pair<2, 120, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 17, 0>::value, vnode_base_offset_pair<2, 120, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2, 120, 17, 1>::value, vnode_base_offset_pair<2, 120, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 120, 17, 2>::value, vnode_base_offset_pair<2, 120, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 18, 0>::value, vnode_base_offset_pair<2, 120, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2, 120, 18, 1>::value, vnode_base_offset_pair<2, 120, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 19, 0>::value, vnode_base_offset_pair<2, 120, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2, 120, 19, 1>::value, vnode_base_offset_pair<2, 120, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 20, 0>::value, vnode_base_offset_pair<2, 120, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2, 120, 20, 1>::value, vnode_base_offset_pair<2, 120, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 21, 0>::value, vnode_base_offset_pair<2, 120, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2, 120, 21, 1>::value, vnode_base_offset_pair<2, 120, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 22, 0>::value, vnode_base_offset_pair<2, 120, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2, 120, 22, 1>::value, vnode_base_offset_pair<2, 120, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 23, 0>::value, vnode_base_offset_pair<2, 120, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2, 120, 23, 1>::value, vnode_base_offset_pair<2, 120, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 24, 0>::value, vnode_base_offset_pair<2, 120, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2, 120, 24, 1>::value, vnode_base_offset_pair<2, 120, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 25, 0>::value, vnode_base_offset_pair<2, 120, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2, 120, 25, 1>::value, vnode_base_offset_pair<2, 120, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 26, 0>::value, vnode_base_offset_pair<2, 120, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2, 120, 26, 1>::value, vnode_base_offset_pair<2, 120, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 120, 26, 2>::value, vnode_base_offset_pair<2, 120, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 27, 0>::value, vnode_base_offset_pair<2, 120, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2, 120, 27, 1>::value, vnode_base_offset_pair<2, 120, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 28, 0>::value, vnode_base_offset_pair<2, 120, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2, 120, 28, 1>::value, vnode_base_offset_pair<2, 120, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 29, 0>::value, vnode_base_offset_pair<2, 120, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2, 120, 29, 1>::value, vnode_base_offset_pair<2, 120, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 30, 0>::value, vnode_base_offset_pair<2, 120, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2, 120, 30, 1>::value, vnode_base_offset_pair<2, 120, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 120, 30, 2>::value, vnode_base_offset_pair<2, 120, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 31, 0>::value, vnode_base_offset_pair<2, 120, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2, 120, 31, 1>::value, vnode_base_offset_pair<2, 120, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 32, 0>::value, vnode_base_offset_pair<2, 120, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2, 120, 32, 1>::value, vnode_base_offset_pair<2, 120, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 33, 0>::value, vnode_base_offset_pair<2, 120, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2, 120, 33, 1>::value, vnode_base_offset_pair<2, 120, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 34, 0>::value, vnode_base_offset_pair<2, 120, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2, 120, 34, 1>::value, vnode_base_offset_pair<2, 120, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 35, 0>::value, vnode_base_offset_pair<2, 120, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2, 120, 35, 1>::value, vnode_base_offset_pair<2, 120, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 36, 0>::value, vnode_base_offset_pair<2, 120, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2, 120, 36, 1>::value, vnode_base_offset_pair<2, 120, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 37, 0>::value, vnode_base_offset_pair<2, 120, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2, 120, 37, 1>::value, vnode_base_offset_pair<2, 120, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 38, 0>::value, vnode_base_offset_pair<2, 120, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2, 120, 38, 1>::value, vnode_base_offset_pair<2, 120, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 39, 0>::value, vnode_base_offset_pair<2, 120, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2, 120, 39, 1>::value, vnode_base_offset_pair<2, 120, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 40, 0>::value, vnode_base_offset_pair<2, 120, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2, 120, 40, 1>::value, vnode_base_offset_pair<2, 120, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 120, 41, 0>::value, vnode_base_offset_pair<2, 120, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2, 120, 41, 1>::value, vnode_base_offset_pair<2, 120, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z128_8 =
{
    {
        { vnode_shift_mod_pair<2, 128,  0, 0>::value, vnode_base_offset_pair<2, 128,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2, 128,  0, 1>::value, vnode_base_offset_pair<2, 128,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 128,  0, 2>::value, vnode_base_offset_pair<2, 128,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 128,  0, 3>::value, vnode_base_offset_pair<2, 128,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 128,  1, 0>::value, vnode_base_offset_pair<2, 128,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2, 128,  1, 1>::value, vnode_base_offset_pair<2, 128,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 128,  1, 2>::value, vnode_base_offset_pair<2, 128,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 128,  1, 3>::value, vnode_base_offset_pair<2, 128,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 128,  1, 4>::value, vnode_base_offset_pair<2, 128,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 128,  2, 0>::value, vnode_base_offset_pair<2, 128,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2, 128,  2, 1>::value, vnode_base_offset_pair<2, 128,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 128,  2, 2>::value, vnode_base_offset_pair<2, 128,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 128,  2, 3>::value, vnode_base_offset_pair<2, 128,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 128,  3, 0>::value, vnode_base_offset_pair<2, 128,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2, 128,  3, 1>::value, vnode_base_offset_pair<2, 128,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 128,  3, 2>::value, vnode_base_offset_pair<2, 128,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 128,  3, 3>::value, vnode_base_offset_pair<2, 128,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 128,  3, 4>::value, vnode_base_offset_pair<2, 128,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 128,  4, 0>::value, vnode_base_offset_pair<2, 128,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2, 128,  4, 1>::value, vnode_base_offset_pair<2, 128,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128,  5, 0>::value, vnode_base_offset_pair<2, 128,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2, 128,  5, 1>::value, vnode_base_offset_pair<2, 128,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 128,  5, 2>::value, vnode_base_offset_pair<2, 128,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 128,  6, 0>::value, vnode_base_offset_pair<2, 128,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2, 128,  6, 1>::value, vnode_base_offset_pair<2, 128,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 128,  6, 2>::value, vnode_base_offset_pair<2, 128,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 128,  7, 0>::value, vnode_base_offset_pair<2, 128,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2, 128,  7, 1>::value, vnode_base_offset_pair<2, 128,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 128,  7, 2>::value, vnode_base_offset_pair<2, 128,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 128,  8, 0>::value, vnode_base_offset_pair<2, 128,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2, 128,  8, 1>::value, vnode_base_offset_pair<2, 128,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128,  9, 0>::value, vnode_base_offset_pair<2, 128,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2, 128,  9, 1>::value, vnode_base_offset_pair<2, 128,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 128,  9, 2>::value, vnode_base_offset_pair<2, 128,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 10, 0>::value, vnode_base_offset_pair<2, 128, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2, 128, 10, 1>::value, vnode_base_offset_pair<2, 128, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 128, 10, 2>::value, vnode_base_offset_pair<2, 128, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 11, 0>::value, vnode_base_offset_pair<2, 128, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2, 128, 11, 1>::value, vnode_base_offset_pair<2, 128, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 128, 11, 2>::value, vnode_base_offset_pair<2, 128, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 12, 0>::value, vnode_base_offset_pair<2, 128, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2, 128, 12, 1>::value, vnode_base_offset_pair<2, 128, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 13, 0>::value, vnode_base_offset_pair<2, 128, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2, 128, 13, 1>::value, vnode_base_offset_pair<2, 128, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 128, 13, 2>::value, vnode_base_offset_pair<2, 128, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 14, 0>::value, vnode_base_offset_pair<2, 128, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2, 128, 14, 1>::value, vnode_base_offset_pair<2, 128, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 128, 14, 2>::value, vnode_base_offset_pair<2, 128, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 15, 0>::value, vnode_base_offset_pair<2, 128, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2, 128, 15, 1>::value, vnode_base_offset_pair<2, 128, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 16, 0>::value, vnode_base_offset_pair<2, 128, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2, 128, 16, 1>::value, vnode_base_offset_pair<2, 128, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 128, 16, 2>::value, vnode_base_offset_pair<2, 128, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 17, 0>::value, vnode_base_offset_pair<2, 128, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2, 128, 17, 1>::value, vnode_base_offset_pair<2, 128, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 128, 17, 2>::value, vnode_base_offset_pair<2, 128, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 18, 0>::value, vnode_base_offset_pair<2, 128, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2, 128, 18, 1>::value, vnode_base_offset_pair<2, 128, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 19, 0>::value, vnode_base_offset_pair<2, 128, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2, 128, 19, 1>::value, vnode_base_offset_pair<2, 128, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 20, 0>::value, vnode_base_offset_pair<2, 128, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2, 128, 20, 1>::value, vnode_base_offset_pair<2, 128, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 21, 0>::value, vnode_base_offset_pair<2, 128, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2, 128, 21, 1>::value, vnode_base_offset_pair<2, 128, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 22, 0>::value, vnode_base_offset_pair<2, 128, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2, 128, 22, 1>::value, vnode_base_offset_pair<2, 128, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 23, 0>::value, vnode_base_offset_pair<2, 128, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2, 128, 23, 1>::value, vnode_base_offset_pair<2, 128, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 24, 0>::value, vnode_base_offset_pair<2, 128, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2, 128, 24, 1>::value, vnode_base_offset_pair<2, 128, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 25, 0>::value, vnode_base_offset_pair<2, 128, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2, 128, 25, 1>::value, vnode_base_offset_pair<2, 128, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 26, 0>::value, vnode_base_offset_pair<2, 128, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2, 128, 26, 1>::value, vnode_base_offset_pair<2, 128, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 128, 26, 2>::value, vnode_base_offset_pair<2, 128, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 27, 0>::value, vnode_base_offset_pair<2, 128, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2, 128, 27, 1>::value, vnode_base_offset_pair<2, 128, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 28, 0>::value, vnode_base_offset_pair<2, 128, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2, 128, 28, 1>::value, vnode_base_offset_pair<2, 128, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 29, 0>::value, vnode_base_offset_pair<2, 128, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2, 128, 29, 1>::value, vnode_base_offset_pair<2, 128, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 30, 0>::value, vnode_base_offset_pair<2, 128, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2, 128, 30, 1>::value, vnode_base_offset_pair<2, 128, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 128, 30, 2>::value, vnode_base_offset_pair<2, 128, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 31, 0>::value, vnode_base_offset_pair<2, 128, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2, 128, 31, 1>::value, vnode_base_offset_pair<2, 128, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 32, 0>::value, vnode_base_offset_pair<2, 128, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2, 128, 32, 1>::value, vnode_base_offset_pair<2, 128, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 33, 0>::value, vnode_base_offset_pair<2, 128, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2, 128, 33, 1>::value, vnode_base_offset_pair<2, 128, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 34, 0>::value, vnode_base_offset_pair<2, 128, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2, 128, 34, 1>::value, vnode_base_offset_pair<2, 128, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 35, 0>::value, vnode_base_offset_pair<2, 128, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2, 128, 35, 1>::value, vnode_base_offset_pair<2, 128, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 36, 0>::value, vnode_base_offset_pair<2, 128, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2, 128, 36, 1>::value, vnode_base_offset_pair<2, 128, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 37, 0>::value, vnode_base_offset_pair<2, 128, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2, 128, 37, 1>::value, vnode_base_offset_pair<2, 128, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 38, 0>::value, vnode_base_offset_pair<2, 128, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2, 128, 38, 1>::value, vnode_base_offset_pair<2, 128, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 39, 0>::value, vnode_base_offset_pair<2, 128, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2, 128, 39, 1>::value, vnode_base_offset_pair<2, 128, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 40, 0>::value, vnode_base_offset_pair<2, 128, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2, 128, 40, 1>::value, vnode_base_offset_pair<2, 128, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 128, 41, 0>::value, vnode_base_offset_pair<2, 128, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2, 128, 41, 1>::value, vnode_base_offset_pair<2, 128, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z144_8 =
{
    {
        { vnode_shift_mod_pair<2, 144,  0, 0>::value, vnode_base_offset_pair<2, 144,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2, 144,  0, 1>::value, vnode_base_offset_pair<2, 144,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 144,  0, 2>::value, vnode_base_offset_pair<2, 144,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 144,  0, 3>::value, vnode_base_offset_pair<2, 144,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 144,  1, 0>::value, vnode_base_offset_pair<2, 144,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2, 144,  1, 1>::value, vnode_base_offset_pair<2, 144,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 144,  1, 2>::value, vnode_base_offset_pair<2, 144,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 144,  1, 3>::value, vnode_base_offset_pair<2, 144,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 144,  1, 4>::value, vnode_base_offset_pair<2, 144,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 144,  2, 0>::value, vnode_base_offset_pair<2, 144,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2, 144,  2, 1>::value, vnode_base_offset_pair<2, 144,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 144,  2, 2>::value, vnode_base_offset_pair<2, 144,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 144,  2, 3>::value, vnode_base_offset_pair<2, 144,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 144,  3, 0>::value, vnode_base_offset_pair<2, 144,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2, 144,  3, 1>::value, vnode_base_offset_pair<2, 144,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 144,  3, 2>::value, vnode_base_offset_pair<2, 144,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 144,  3, 3>::value, vnode_base_offset_pair<2, 144,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 144,  3, 4>::value, vnode_base_offset_pair<2, 144,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 144,  4, 0>::value, vnode_base_offset_pair<2, 144,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2, 144,  4, 1>::value, vnode_base_offset_pair<2, 144,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144,  5, 0>::value, vnode_base_offset_pair<2, 144,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2, 144,  5, 1>::value, vnode_base_offset_pair<2, 144,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 144,  5, 2>::value, vnode_base_offset_pair<2, 144,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 144,  6, 0>::value, vnode_base_offset_pair<2, 144,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2, 144,  6, 1>::value, vnode_base_offset_pair<2, 144,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 144,  6, 2>::value, vnode_base_offset_pair<2, 144,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 144,  7, 0>::value, vnode_base_offset_pair<2, 144,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2, 144,  7, 1>::value, vnode_base_offset_pair<2, 144,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 144,  7, 2>::value, vnode_base_offset_pair<2, 144,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 144,  8, 0>::value, vnode_base_offset_pair<2, 144,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2, 144,  8, 1>::value, vnode_base_offset_pair<2, 144,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144,  9, 0>::value, vnode_base_offset_pair<2, 144,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2, 144,  9, 1>::value, vnode_base_offset_pair<2, 144,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 144,  9, 2>::value, vnode_base_offset_pair<2, 144,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 10, 0>::value, vnode_base_offset_pair<2, 144, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2, 144, 10, 1>::value, vnode_base_offset_pair<2, 144, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 144, 10, 2>::value, vnode_base_offset_pair<2, 144, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 11, 0>::value, vnode_base_offset_pair<2, 144, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2, 144, 11, 1>::value, vnode_base_offset_pair<2, 144, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 144, 11, 2>::value, vnode_base_offset_pair<2, 144, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 12, 0>::value, vnode_base_offset_pair<2, 144, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2, 144, 12, 1>::value, vnode_base_offset_pair<2, 144, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 13, 0>::value, vnode_base_offset_pair<2, 144, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2, 144, 13, 1>::value, vnode_base_offset_pair<2, 144, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 144, 13, 2>::value, vnode_base_offset_pair<2, 144, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 14, 0>::value, vnode_base_offset_pair<2, 144, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2, 144, 14, 1>::value, vnode_base_offset_pair<2, 144, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 144, 14, 2>::value, vnode_base_offset_pair<2, 144, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 15, 0>::value, vnode_base_offset_pair<2, 144, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2, 144, 15, 1>::value, vnode_base_offset_pair<2, 144, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 16, 0>::value, vnode_base_offset_pair<2, 144, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2, 144, 16, 1>::value, vnode_base_offset_pair<2, 144, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 144, 16, 2>::value, vnode_base_offset_pair<2, 144, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 17, 0>::value, vnode_base_offset_pair<2, 144, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2, 144, 17, 1>::value, vnode_base_offset_pair<2, 144, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 144, 17, 2>::value, vnode_base_offset_pair<2, 144, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 18, 0>::value, vnode_base_offset_pair<2, 144, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2, 144, 18, 1>::value, vnode_base_offset_pair<2, 144, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 19, 0>::value, vnode_base_offset_pair<2, 144, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2, 144, 19, 1>::value, vnode_base_offset_pair<2, 144, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 20, 0>::value, vnode_base_offset_pair<2, 144, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2, 144, 20, 1>::value, vnode_base_offset_pair<2, 144, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 21, 0>::value, vnode_base_offset_pair<2, 144, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2, 144, 21, 1>::value, vnode_base_offset_pair<2, 144, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 22, 0>::value, vnode_base_offset_pair<2, 144, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2, 144, 22, 1>::value, vnode_base_offset_pair<2, 144, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 23, 0>::value, vnode_base_offset_pair<2, 144, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2, 144, 23, 1>::value, vnode_base_offset_pair<2, 144, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 24, 0>::value, vnode_base_offset_pair<2, 144, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2, 144, 24, 1>::value, vnode_base_offset_pair<2, 144, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 25, 0>::value, vnode_base_offset_pair<2, 144, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2, 144, 25, 1>::value, vnode_base_offset_pair<2, 144, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 26, 0>::value, vnode_base_offset_pair<2, 144, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2, 144, 26, 1>::value, vnode_base_offset_pair<2, 144, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 144, 26, 2>::value, vnode_base_offset_pair<2, 144, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 27, 0>::value, vnode_base_offset_pair<2, 144, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2, 144, 27, 1>::value, vnode_base_offset_pair<2, 144, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 28, 0>::value, vnode_base_offset_pair<2, 144, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2, 144, 28, 1>::value, vnode_base_offset_pair<2, 144, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 29, 0>::value, vnode_base_offset_pair<2, 144, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2, 144, 29, 1>::value, vnode_base_offset_pair<2, 144, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 30, 0>::value, vnode_base_offset_pair<2, 144, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2, 144, 30, 1>::value, vnode_base_offset_pair<2, 144, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 144, 30, 2>::value, vnode_base_offset_pair<2, 144, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 31, 0>::value, vnode_base_offset_pair<2, 144, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2, 144, 31, 1>::value, vnode_base_offset_pair<2, 144, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 32, 0>::value, vnode_base_offset_pair<2, 144, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2, 144, 32, 1>::value, vnode_base_offset_pair<2, 144, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 33, 0>::value, vnode_base_offset_pair<2, 144, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2, 144, 33, 1>::value, vnode_base_offset_pair<2, 144, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 34, 0>::value, vnode_base_offset_pair<2, 144, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2, 144, 34, 1>::value, vnode_base_offset_pair<2, 144, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 35, 0>::value, vnode_base_offset_pair<2, 144, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2, 144, 35, 1>::value, vnode_base_offset_pair<2, 144, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 36, 0>::value, vnode_base_offset_pair<2, 144, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2, 144, 36, 1>::value, vnode_base_offset_pair<2, 144, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 37, 0>::value, vnode_base_offset_pair<2, 144, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2, 144, 37, 1>::value, vnode_base_offset_pair<2, 144, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 38, 0>::value, vnode_base_offset_pair<2, 144, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2, 144, 38, 1>::value, vnode_base_offset_pair<2, 144, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 39, 0>::value, vnode_base_offset_pair<2, 144, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2, 144, 39, 1>::value, vnode_base_offset_pair<2, 144, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 40, 0>::value, vnode_base_offset_pair<2, 144, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2, 144, 40, 1>::value, vnode_base_offset_pair<2, 144, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 144, 41, 0>::value, vnode_base_offset_pair<2, 144, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2, 144, 41, 1>::value, vnode_base_offset_pair<2, 144, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z160_8 =
{
    {
        { vnode_shift_mod_pair<2, 160,  0, 0>::value, vnode_base_offset_pair<2, 160,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2, 160,  0, 1>::value, vnode_base_offset_pair<2, 160,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 160,  0, 2>::value, vnode_base_offset_pair<2, 160,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 160,  0, 3>::value, vnode_base_offset_pair<2, 160,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 160,  1, 0>::value, vnode_base_offset_pair<2, 160,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2, 160,  1, 1>::value, vnode_base_offset_pair<2, 160,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 160,  1, 2>::value, vnode_base_offset_pair<2, 160,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 160,  1, 3>::value, vnode_base_offset_pair<2, 160,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 160,  1, 4>::value, vnode_base_offset_pair<2, 160,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 160,  2, 0>::value, vnode_base_offset_pair<2, 160,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2, 160,  2, 1>::value, vnode_base_offset_pair<2, 160,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 160,  2, 2>::value, vnode_base_offset_pair<2, 160,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 160,  2, 3>::value, vnode_base_offset_pair<2, 160,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 160,  3, 0>::value, vnode_base_offset_pair<2, 160,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2, 160,  3, 1>::value, vnode_base_offset_pair<2, 160,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 160,  3, 2>::value, vnode_base_offset_pair<2, 160,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 160,  3, 3>::value, vnode_base_offset_pair<2, 160,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 160,  3, 4>::value, vnode_base_offset_pair<2, 160,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 160,  4, 0>::value, vnode_base_offset_pair<2, 160,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2, 160,  4, 1>::value, vnode_base_offset_pair<2, 160,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160,  5, 0>::value, vnode_base_offset_pair<2, 160,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2, 160,  5, 1>::value, vnode_base_offset_pair<2, 160,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 160,  5, 2>::value, vnode_base_offset_pair<2, 160,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 160,  6, 0>::value, vnode_base_offset_pair<2, 160,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2, 160,  6, 1>::value, vnode_base_offset_pair<2, 160,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 160,  6, 2>::value, vnode_base_offset_pair<2, 160,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 160,  7, 0>::value, vnode_base_offset_pair<2, 160,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2, 160,  7, 1>::value, vnode_base_offset_pair<2, 160,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 160,  7, 2>::value, vnode_base_offset_pair<2, 160,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 160,  8, 0>::value, vnode_base_offset_pair<2, 160,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2, 160,  8, 1>::value, vnode_base_offset_pair<2, 160,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160,  9, 0>::value, vnode_base_offset_pair<2, 160,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2, 160,  9, 1>::value, vnode_base_offset_pair<2, 160,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 160,  9, 2>::value, vnode_base_offset_pair<2, 160,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 10, 0>::value, vnode_base_offset_pair<2, 160, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2, 160, 10, 1>::value, vnode_base_offset_pair<2, 160, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 160, 10, 2>::value, vnode_base_offset_pair<2, 160, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 11, 0>::value, vnode_base_offset_pair<2, 160, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2, 160, 11, 1>::value, vnode_base_offset_pair<2, 160, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 160, 11, 2>::value, vnode_base_offset_pair<2, 160, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 12, 0>::value, vnode_base_offset_pair<2, 160, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2, 160, 12, 1>::value, vnode_base_offset_pair<2, 160, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 13, 0>::value, vnode_base_offset_pair<2, 160, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2, 160, 13, 1>::value, vnode_base_offset_pair<2, 160, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 160, 13, 2>::value, vnode_base_offset_pair<2, 160, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 14, 0>::value, vnode_base_offset_pair<2, 160, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2, 160, 14, 1>::value, vnode_base_offset_pair<2, 160, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 160, 14, 2>::value, vnode_base_offset_pair<2, 160, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 15, 0>::value, vnode_base_offset_pair<2, 160, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2, 160, 15, 1>::value, vnode_base_offset_pair<2, 160, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 16, 0>::value, vnode_base_offset_pair<2, 160, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2, 160, 16, 1>::value, vnode_base_offset_pair<2, 160, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 160, 16, 2>::value, vnode_base_offset_pair<2, 160, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 17, 0>::value, vnode_base_offset_pair<2, 160, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2, 160, 17, 1>::value, vnode_base_offset_pair<2, 160, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 160, 17, 2>::value, vnode_base_offset_pair<2, 160, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 18, 0>::value, vnode_base_offset_pair<2, 160, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2, 160, 18, 1>::value, vnode_base_offset_pair<2, 160, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 19, 0>::value, vnode_base_offset_pair<2, 160, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2, 160, 19, 1>::value, vnode_base_offset_pair<2, 160, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 20, 0>::value, vnode_base_offset_pair<2, 160, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2, 160, 20, 1>::value, vnode_base_offset_pair<2, 160, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 21, 0>::value, vnode_base_offset_pair<2, 160, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2, 160, 21, 1>::value, vnode_base_offset_pair<2, 160, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 22, 0>::value, vnode_base_offset_pair<2, 160, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2, 160, 22, 1>::value, vnode_base_offset_pair<2, 160, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 23, 0>::value, vnode_base_offset_pair<2, 160, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2, 160, 23, 1>::value, vnode_base_offset_pair<2, 160, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 24, 0>::value, vnode_base_offset_pair<2, 160, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2, 160, 24, 1>::value, vnode_base_offset_pair<2, 160, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 25, 0>::value, vnode_base_offset_pair<2, 160, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2, 160, 25, 1>::value, vnode_base_offset_pair<2, 160, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 26, 0>::value, vnode_base_offset_pair<2, 160, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2, 160, 26, 1>::value, vnode_base_offset_pair<2, 160, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 160, 26, 2>::value, vnode_base_offset_pair<2, 160, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 27, 0>::value, vnode_base_offset_pair<2, 160, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2, 160, 27, 1>::value, vnode_base_offset_pair<2, 160, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 28, 0>::value, vnode_base_offset_pair<2, 160, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2, 160, 28, 1>::value, vnode_base_offset_pair<2, 160, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 29, 0>::value, vnode_base_offset_pair<2, 160, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2, 160, 29, 1>::value, vnode_base_offset_pair<2, 160, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 30, 0>::value, vnode_base_offset_pair<2, 160, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2, 160, 30, 1>::value, vnode_base_offset_pair<2, 160, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 160, 30, 2>::value, vnode_base_offset_pair<2, 160, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 31, 0>::value, vnode_base_offset_pair<2, 160, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2, 160, 31, 1>::value, vnode_base_offset_pair<2, 160, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 32, 0>::value, vnode_base_offset_pair<2, 160, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2, 160, 32, 1>::value, vnode_base_offset_pair<2, 160, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 33, 0>::value, vnode_base_offset_pair<2, 160, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2, 160, 33, 1>::value, vnode_base_offset_pair<2, 160, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 34, 0>::value, vnode_base_offset_pair<2, 160, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2, 160, 34, 1>::value, vnode_base_offset_pair<2, 160, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 35, 0>::value, vnode_base_offset_pair<2, 160, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2, 160, 35, 1>::value, vnode_base_offset_pair<2, 160, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 36, 0>::value, vnode_base_offset_pair<2, 160, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2, 160, 36, 1>::value, vnode_base_offset_pair<2, 160, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 37, 0>::value, vnode_base_offset_pair<2, 160, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2, 160, 37, 1>::value, vnode_base_offset_pair<2, 160, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 38, 0>::value, vnode_base_offset_pair<2, 160, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2, 160, 38, 1>::value, vnode_base_offset_pair<2, 160, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 39, 0>::value, vnode_base_offset_pair<2, 160, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2, 160, 39, 1>::value, vnode_base_offset_pair<2, 160, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 40, 0>::value, vnode_base_offset_pair<2, 160, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2, 160, 40, 1>::value, vnode_base_offset_pair<2, 160, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 160, 41, 0>::value, vnode_base_offset_pair<2, 160, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2, 160, 41, 1>::value, vnode_base_offset_pair<2, 160, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z176_8 =
{
    {
        { vnode_shift_mod_pair<2, 176,  0, 0>::value, vnode_base_offset_pair<2, 176,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2, 176,  0, 1>::value, vnode_base_offset_pair<2, 176,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 176,  0, 2>::value, vnode_base_offset_pair<2, 176,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 176,  0, 3>::value, vnode_base_offset_pair<2, 176,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 176,  1, 0>::value, vnode_base_offset_pair<2, 176,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2, 176,  1, 1>::value, vnode_base_offset_pair<2, 176,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 176,  1, 2>::value, vnode_base_offset_pair<2, 176,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 176,  1, 3>::value, vnode_base_offset_pair<2, 176,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 176,  1, 4>::value, vnode_base_offset_pair<2, 176,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 176,  2, 0>::value, vnode_base_offset_pair<2, 176,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2, 176,  2, 1>::value, vnode_base_offset_pair<2, 176,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 176,  2, 2>::value, vnode_base_offset_pair<2, 176,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 176,  2, 3>::value, vnode_base_offset_pair<2, 176,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 176,  3, 0>::value, vnode_base_offset_pair<2, 176,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2, 176,  3, 1>::value, vnode_base_offset_pair<2, 176,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 176,  3, 2>::value, vnode_base_offset_pair<2, 176,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 176,  3, 3>::value, vnode_base_offset_pair<2, 176,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 176,  3, 4>::value, vnode_base_offset_pair<2, 176,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 176,  4, 0>::value, vnode_base_offset_pair<2, 176,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2, 176,  4, 1>::value, vnode_base_offset_pair<2, 176,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176,  5, 0>::value, vnode_base_offset_pair<2, 176,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2, 176,  5, 1>::value, vnode_base_offset_pair<2, 176,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 176,  5, 2>::value, vnode_base_offset_pair<2, 176,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 176,  6, 0>::value, vnode_base_offset_pair<2, 176,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2, 176,  6, 1>::value, vnode_base_offset_pair<2, 176,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 176,  6, 2>::value, vnode_base_offset_pair<2, 176,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 176,  7, 0>::value, vnode_base_offset_pair<2, 176,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2, 176,  7, 1>::value, vnode_base_offset_pair<2, 176,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 176,  7, 2>::value, vnode_base_offset_pair<2, 176,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 176,  8, 0>::value, vnode_base_offset_pair<2, 176,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2, 176,  8, 1>::value, vnode_base_offset_pair<2, 176,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176,  9, 0>::value, vnode_base_offset_pair<2, 176,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2, 176,  9, 1>::value, vnode_base_offset_pair<2, 176,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 176,  9, 2>::value, vnode_base_offset_pair<2, 176,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 10, 0>::value, vnode_base_offset_pair<2, 176, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2, 176, 10, 1>::value, vnode_base_offset_pair<2, 176, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 176, 10, 2>::value, vnode_base_offset_pair<2, 176, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 11, 0>::value, vnode_base_offset_pair<2, 176, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2, 176, 11, 1>::value, vnode_base_offset_pair<2, 176, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 176, 11, 2>::value, vnode_base_offset_pair<2, 176, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 12, 0>::value, vnode_base_offset_pair<2, 176, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2, 176, 12, 1>::value, vnode_base_offset_pair<2, 176, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 13, 0>::value, vnode_base_offset_pair<2, 176, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2, 176, 13, 1>::value, vnode_base_offset_pair<2, 176, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 176, 13, 2>::value, vnode_base_offset_pair<2, 176, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 14, 0>::value, vnode_base_offset_pair<2, 176, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2, 176, 14, 1>::value, vnode_base_offset_pair<2, 176, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 176, 14, 2>::value, vnode_base_offset_pair<2, 176, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 15, 0>::value, vnode_base_offset_pair<2, 176, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2, 176, 15, 1>::value, vnode_base_offset_pair<2, 176, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 16, 0>::value, vnode_base_offset_pair<2, 176, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2, 176, 16, 1>::value, vnode_base_offset_pair<2, 176, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 176, 16, 2>::value, vnode_base_offset_pair<2, 176, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 17, 0>::value, vnode_base_offset_pair<2, 176, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2, 176, 17, 1>::value, vnode_base_offset_pair<2, 176, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 176, 17, 2>::value, vnode_base_offset_pair<2, 176, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 18, 0>::value, vnode_base_offset_pair<2, 176, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2, 176, 18, 1>::value, vnode_base_offset_pair<2, 176, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 19, 0>::value, vnode_base_offset_pair<2, 176, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2, 176, 19, 1>::value, vnode_base_offset_pair<2, 176, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 20, 0>::value, vnode_base_offset_pair<2, 176, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2, 176, 20, 1>::value, vnode_base_offset_pair<2, 176, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 21, 0>::value, vnode_base_offset_pair<2, 176, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2, 176, 21, 1>::value, vnode_base_offset_pair<2, 176, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 22, 0>::value, vnode_base_offset_pair<2, 176, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2, 176, 22, 1>::value, vnode_base_offset_pair<2, 176, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 23, 0>::value, vnode_base_offset_pair<2, 176, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2, 176, 23, 1>::value, vnode_base_offset_pair<2, 176, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 24, 0>::value, vnode_base_offset_pair<2, 176, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2, 176, 24, 1>::value, vnode_base_offset_pair<2, 176, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 25, 0>::value, vnode_base_offset_pair<2, 176, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2, 176, 25, 1>::value, vnode_base_offset_pair<2, 176, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 26, 0>::value, vnode_base_offset_pair<2, 176, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2, 176, 26, 1>::value, vnode_base_offset_pair<2, 176, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 176, 26, 2>::value, vnode_base_offset_pair<2, 176, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 27, 0>::value, vnode_base_offset_pair<2, 176, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2, 176, 27, 1>::value, vnode_base_offset_pair<2, 176, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 28, 0>::value, vnode_base_offset_pair<2, 176, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2, 176, 28, 1>::value, vnode_base_offset_pair<2, 176, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 29, 0>::value, vnode_base_offset_pair<2, 176, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2, 176, 29, 1>::value, vnode_base_offset_pair<2, 176, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 30, 0>::value, vnode_base_offset_pair<2, 176, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2, 176, 30, 1>::value, vnode_base_offset_pair<2, 176, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 176, 30, 2>::value, vnode_base_offset_pair<2, 176, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 31, 0>::value, vnode_base_offset_pair<2, 176, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2, 176, 31, 1>::value, vnode_base_offset_pair<2, 176, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 32, 0>::value, vnode_base_offset_pair<2, 176, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2, 176, 32, 1>::value, vnode_base_offset_pair<2, 176, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 33, 0>::value, vnode_base_offset_pair<2, 176, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2, 176, 33, 1>::value, vnode_base_offset_pair<2, 176, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 34, 0>::value, vnode_base_offset_pair<2, 176, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2, 176, 34, 1>::value, vnode_base_offset_pair<2, 176, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 35, 0>::value, vnode_base_offset_pair<2, 176, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2, 176, 35, 1>::value, vnode_base_offset_pair<2, 176, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 36, 0>::value, vnode_base_offset_pair<2, 176, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2, 176, 36, 1>::value, vnode_base_offset_pair<2, 176, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 37, 0>::value, vnode_base_offset_pair<2, 176, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2, 176, 37, 1>::value, vnode_base_offset_pair<2, 176, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 38, 0>::value, vnode_base_offset_pair<2, 176, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2, 176, 38, 1>::value, vnode_base_offset_pair<2, 176, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 39, 0>::value, vnode_base_offset_pair<2, 176, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2, 176, 39, 1>::value, vnode_base_offset_pair<2, 176, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 40, 0>::value, vnode_base_offset_pair<2, 176, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2, 176, 40, 1>::value, vnode_base_offset_pair<2, 176, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 176, 41, 0>::value, vnode_base_offset_pair<2, 176, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2, 176, 41, 1>::value, vnode_base_offset_pair<2, 176, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z192_8 =
{
    {
        { vnode_shift_mod_pair<2, 192,  0, 0>::value, vnode_base_offset_pair<2, 192,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2, 192,  0, 1>::value, vnode_base_offset_pair<2, 192,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 192,  0, 2>::value, vnode_base_offset_pair<2, 192,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 192,  0, 3>::value, vnode_base_offset_pair<2, 192,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 192,  1, 0>::value, vnode_base_offset_pair<2, 192,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2, 192,  1, 1>::value, vnode_base_offset_pair<2, 192,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 192,  1, 2>::value, vnode_base_offset_pair<2, 192,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 192,  1, 3>::value, vnode_base_offset_pair<2, 192,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 192,  1, 4>::value, vnode_base_offset_pair<2, 192,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 192,  2, 0>::value, vnode_base_offset_pair<2, 192,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2, 192,  2, 1>::value, vnode_base_offset_pair<2, 192,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 192,  2, 2>::value, vnode_base_offset_pair<2, 192,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 192,  2, 3>::value, vnode_base_offset_pair<2, 192,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 192,  3, 0>::value, vnode_base_offset_pair<2, 192,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2, 192,  3, 1>::value, vnode_base_offset_pair<2, 192,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 192,  3, 2>::value, vnode_base_offset_pair<2, 192,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 192,  3, 3>::value, vnode_base_offset_pair<2, 192,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 192,  3, 4>::value, vnode_base_offset_pair<2, 192,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 192,  4, 0>::value, vnode_base_offset_pair<2, 192,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2, 192,  4, 1>::value, vnode_base_offset_pair<2, 192,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192,  5, 0>::value, vnode_base_offset_pair<2, 192,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2, 192,  5, 1>::value, vnode_base_offset_pair<2, 192,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 192,  5, 2>::value, vnode_base_offset_pair<2, 192,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 192,  6, 0>::value, vnode_base_offset_pair<2, 192,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2, 192,  6, 1>::value, vnode_base_offset_pair<2, 192,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 192,  6, 2>::value, vnode_base_offset_pair<2, 192,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 192,  7, 0>::value, vnode_base_offset_pair<2, 192,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2, 192,  7, 1>::value, vnode_base_offset_pair<2, 192,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 192,  7, 2>::value, vnode_base_offset_pair<2, 192,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 192,  8, 0>::value, vnode_base_offset_pair<2, 192,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2, 192,  8, 1>::value, vnode_base_offset_pair<2, 192,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192,  9, 0>::value, vnode_base_offset_pair<2, 192,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2, 192,  9, 1>::value, vnode_base_offset_pair<2, 192,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 192,  9, 2>::value, vnode_base_offset_pair<2, 192,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 10, 0>::value, vnode_base_offset_pair<2, 192, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2, 192, 10, 1>::value, vnode_base_offset_pair<2, 192, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 192, 10, 2>::value, vnode_base_offset_pair<2, 192, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 11, 0>::value, vnode_base_offset_pair<2, 192, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2, 192, 11, 1>::value, vnode_base_offset_pair<2, 192, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 192, 11, 2>::value, vnode_base_offset_pair<2, 192, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 12, 0>::value, vnode_base_offset_pair<2, 192, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2, 192, 12, 1>::value, vnode_base_offset_pair<2, 192, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 13, 0>::value, vnode_base_offset_pair<2, 192, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2, 192, 13, 1>::value, vnode_base_offset_pair<2, 192, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 192, 13, 2>::value, vnode_base_offset_pair<2, 192, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 14, 0>::value, vnode_base_offset_pair<2, 192, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2, 192, 14, 1>::value, vnode_base_offset_pair<2, 192, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 192, 14, 2>::value, vnode_base_offset_pair<2, 192, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 15, 0>::value, vnode_base_offset_pair<2, 192, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2, 192, 15, 1>::value, vnode_base_offset_pair<2, 192, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 16, 0>::value, vnode_base_offset_pair<2, 192, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2, 192, 16, 1>::value, vnode_base_offset_pair<2, 192, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 192, 16, 2>::value, vnode_base_offset_pair<2, 192, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 17, 0>::value, vnode_base_offset_pair<2, 192, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2, 192, 17, 1>::value, vnode_base_offset_pair<2, 192, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 192, 17, 2>::value, vnode_base_offset_pair<2, 192, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 18, 0>::value, vnode_base_offset_pair<2, 192, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2, 192, 18, 1>::value, vnode_base_offset_pair<2, 192, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 19, 0>::value, vnode_base_offset_pair<2, 192, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2, 192, 19, 1>::value, vnode_base_offset_pair<2, 192, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 20, 0>::value, vnode_base_offset_pair<2, 192, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2, 192, 20, 1>::value, vnode_base_offset_pair<2, 192, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 21, 0>::value, vnode_base_offset_pair<2, 192, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2, 192, 21, 1>::value, vnode_base_offset_pair<2, 192, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 22, 0>::value, vnode_base_offset_pair<2, 192, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2, 192, 22, 1>::value, vnode_base_offset_pair<2, 192, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 23, 0>::value, vnode_base_offset_pair<2, 192, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2, 192, 23, 1>::value, vnode_base_offset_pair<2, 192, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 24, 0>::value, vnode_base_offset_pair<2, 192, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2, 192, 24, 1>::value, vnode_base_offset_pair<2, 192, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 25, 0>::value, vnode_base_offset_pair<2, 192, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2, 192, 25, 1>::value, vnode_base_offset_pair<2, 192, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 26, 0>::value, vnode_base_offset_pair<2, 192, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2, 192, 26, 1>::value, vnode_base_offset_pair<2, 192, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 192, 26, 2>::value, vnode_base_offset_pair<2, 192, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 27, 0>::value, vnode_base_offset_pair<2, 192, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2, 192, 27, 1>::value, vnode_base_offset_pair<2, 192, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 28, 0>::value, vnode_base_offset_pair<2, 192, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2, 192, 28, 1>::value, vnode_base_offset_pair<2, 192, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 29, 0>::value, vnode_base_offset_pair<2, 192, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2, 192, 29, 1>::value, vnode_base_offset_pair<2, 192, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 30, 0>::value, vnode_base_offset_pair<2, 192, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2, 192, 30, 1>::value, vnode_base_offset_pair<2, 192, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 192, 30, 2>::value, vnode_base_offset_pair<2, 192, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 31, 0>::value, vnode_base_offset_pair<2, 192, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2, 192, 31, 1>::value, vnode_base_offset_pair<2, 192, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 32, 0>::value, vnode_base_offset_pair<2, 192, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2, 192, 32, 1>::value, vnode_base_offset_pair<2, 192, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 33, 0>::value, vnode_base_offset_pair<2, 192, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2, 192, 33, 1>::value, vnode_base_offset_pair<2, 192, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 34, 0>::value, vnode_base_offset_pair<2, 192, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2, 192, 34, 1>::value, vnode_base_offset_pair<2, 192, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 35, 0>::value, vnode_base_offset_pair<2, 192, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2, 192, 35, 1>::value, vnode_base_offset_pair<2, 192, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 36, 0>::value, vnode_base_offset_pair<2, 192, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2, 192, 36, 1>::value, vnode_base_offset_pair<2, 192, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 37, 0>::value, vnode_base_offset_pair<2, 192, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2, 192, 37, 1>::value, vnode_base_offset_pair<2, 192, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 38, 0>::value, vnode_base_offset_pair<2, 192, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2, 192, 38, 1>::value, vnode_base_offset_pair<2, 192, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 39, 0>::value, vnode_base_offset_pair<2, 192, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2, 192, 39, 1>::value, vnode_base_offset_pair<2, 192, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 40, 0>::value, vnode_base_offset_pair<2, 192, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2, 192, 40, 1>::value, vnode_base_offset_pair<2, 192, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 192, 41, 0>::value, vnode_base_offset_pair<2, 192, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2, 192, 41, 1>::value, vnode_base_offset_pair<2, 192, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z208_8 =
{
    {
        { vnode_shift_mod_pair<2, 208,  0, 0>::value, vnode_base_offset_pair<2, 208,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2, 208,  0, 1>::value, vnode_base_offset_pair<2, 208,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 208,  0, 2>::value, vnode_base_offset_pair<2, 208,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 208,  0, 3>::value, vnode_base_offset_pair<2, 208,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 208,  1, 0>::value, vnode_base_offset_pair<2, 208,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2, 208,  1, 1>::value, vnode_base_offset_pair<2, 208,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 208,  1, 2>::value, vnode_base_offset_pair<2, 208,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 208,  1, 3>::value, vnode_base_offset_pair<2, 208,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 208,  1, 4>::value, vnode_base_offset_pair<2, 208,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 208,  2, 0>::value, vnode_base_offset_pair<2, 208,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2, 208,  2, 1>::value, vnode_base_offset_pair<2, 208,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 208,  2, 2>::value, vnode_base_offset_pair<2, 208,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 208,  2, 3>::value, vnode_base_offset_pair<2, 208,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 208,  3, 0>::value, vnode_base_offset_pair<2, 208,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2, 208,  3, 1>::value, vnode_base_offset_pair<2, 208,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 208,  3, 2>::value, vnode_base_offset_pair<2, 208,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 208,  3, 3>::value, vnode_base_offset_pair<2, 208,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 208,  3, 4>::value, vnode_base_offset_pair<2, 208,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 208,  4, 0>::value, vnode_base_offset_pair<2, 208,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2, 208,  4, 1>::value, vnode_base_offset_pair<2, 208,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208,  5, 0>::value, vnode_base_offset_pair<2, 208,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2, 208,  5, 1>::value, vnode_base_offset_pair<2, 208,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 208,  5, 2>::value, vnode_base_offset_pair<2, 208,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 208,  6, 0>::value, vnode_base_offset_pair<2, 208,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2, 208,  6, 1>::value, vnode_base_offset_pair<2, 208,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 208,  6, 2>::value, vnode_base_offset_pair<2, 208,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 208,  7, 0>::value, vnode_base_offset_pair<2, 208,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2, 208,  7, 1>::value, vnode_base_offset_pair<2, 208,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 208,  7, 2>::value, vnode_base_offset_pair<2, 208,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 208,  8, 0>::value, vnode_base_offset_pair<2, 208,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2, 208,  8, 1>::value, vnode_base_offset_pair<2, 208,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208,  9, 0>::value, vnode_base_offset_pair<2, 208,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2, 208,  9, 1>::value, vnode_base_offset_pair<2, 208,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 208,  9, 2>::value, vnode_base_offset_pair<2, 208,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 10, 0>::value, vnode_base_offset_pair<2, 208, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2, 208, 10, 1>::value, vnode_base_offset_pair<2, 208, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 208, 10, 2>::value, vnode_base_offset_pair<2, 208, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 11, 0>::value, vnode_base_offset_pair<2, 208, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2, 208, 11, 1>::value, vnode_base_offset_pair<2, 208, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 208, 11, 2>::value, vnode_base_offset_pair<2, 208, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 12, 0>::value, vnode_base_offset_pair<2, 208, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2, 208, 12, 1>::value, vnode_base_offset_pair<2, 208, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 13, 0>::value, vnode_base_offset_pair<2, 208, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2, 208, 13, 1>::value, vnode_base_offset_pair<2, 208, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 208, 13, 2>::value, vnode_base_offset_pair<2, 208, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 14, 0>::value, vnode_base_offset_pair<2, 208, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2, 208, 14, 1>::value, vnode_base_offset_pair<2, 208, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 208, 14, 2>::value, vnode_base_offset_pair<2, 208, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 15, 0>::value, vnode_base_offset_pair<2, 208, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2, 208, 15, 1>::value, vnode_base_offset_pair<2, 208, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 16, 0>::value, vnode_base_offset_pair<2, 208, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2, 208, 16, 1>::value, vnode_base_offset_pair<2, 208, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 208, 16, 2>::value, vnode_base_offset_pair<2, 208, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 17, 0>::value, vnode_base_offset_pair<2, 208, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2, 208, 17, 1>::value, vnode_base_offset_pair<2, 208, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 208, 17, 2>::value, vnode_base_offset_pair<2, 208, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 18, 0>::value, vnode_base_offset_pair<2, 208, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2, 208, 18, 1>::value, vnode_base_offset_pair<2, 208, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 19, 0>::value, vnode_base_offset_pair<2, 208, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2, 208, 19, 1>::value, vnode_base_offset_pair<2, 208, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 20, 0>::value, vnode_base_offset_pair<2, 208, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2, 208, 20, 1>::value, vnode_base_offset_pair<2, 208, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 21, 0>::value, vnode_base_offset_pair<2, 208, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2, 208, 21, 1>::value, vnode_base_offset_pair<2, 208, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 22, 0>::value, vnode_base_offset_pair<2, 208, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2, 208, 22, 1>::value, vnode_base_offset_pair<2, 208, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 23, 0>::value, vnode_base_offset_pair<2, 208, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2, 208, 23, 1>::value, vnode_base_offset_pair<2, 208, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 24, 0>::value, vnode_base_offset_pair<2, 208, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2, 208, 24, 1>::value, vnode_base_offset_pair<2, 208, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 25, 0>::value, vnode_base_offset_pair<2, 208, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2, 208, 25, 1>::value, vnode_base_offset_pair<2, 208, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 26, 0>::value, vnode_base_offset_pair<2, 208, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2, 208, 26, 1>::value, vnode_base_offset_pair<2, 208, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 208, 26, 2>::value, vnode_base_offset_pair<2, 208, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 27, 0>::value, vnode_base_offset_pair<2, 208, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2, 208, 27, 1>::value, vnode_base_offset_pair<2, 208, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 28, 0>::value, vnode_base_offset_pair<2, 208, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2, 208, 28, 1>::value, vnode_base_offset_pair<2, 208, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 29, 0>::value, vnode_base_offset_pair<2, 208, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2, 208, 29, 1>::value, vnode_base_offset_pair<2, 208, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 30, 0>::value, vnode_base_offset_pair<2, 208, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2, 208, 30, 1>::value, vnode_base_offset_pair<2, 208, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 208, 30, 2>::value, vnode_base_offset_pair<2, 208, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 31, 0>::value, vnode_base_offset_pair<2, 208, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2, 208, 31, 1>::value, vnode_base_offset_pair<2, 208, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 32, 0>::value, vnode_base_offset_pair<2, 208, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2, 208, 32, 1>::value, vnode_base_offset_pair<2, 208, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 33, 0>::value, vnode_base_offset_pair<2, 208, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2, 208, 33, 1>::value, vnode_base_offset_pair<2, 208, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 34, 0>::value, vnode_base_offset_pair<2, 208, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2, 208, 34, 1>::value, vnode_base_offset_pair<2, 208, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 35, 0>::value, vnode_base_offset_pair<2, 208, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2, 208, 35, 1>::value, vnode_base_offset_pair<2, 208, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 36, 0>::value, vnode_base_offset_pair<2, 208, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2, 208, 36, 1>::value, vnode_base_offset_pair<2, 208, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 37, 0>::value, vnode_base_offset_pair<2, 208, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2, 208, 37, 1>::value, vnode_base_offset_pair<2, 208, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 38, 0>::value, vnode_base_offset_pair<2, 208, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2, 208, 38, 1>::value, vnode_base_offset_pair<2, 208, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 39, 0>::value, vnode_base_offset_pair<2, 208, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2, 208, 39, 1>::value, vnode_base_offset_pair<2, 208, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 40, 0>::value, vnode_base_offset_pair<2, 208, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2, 208, 40, 1>::value, vnode_base_offset_pair<2, 208, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 208, 41, 0>::value, vnode_base_offset_pair<2, 208, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2, 208, 41, 1>::value, vnode_base_offset_pair<2, 208, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z224_8 =
{
    {
        { vnode_shift_mod_pair<2, 224,  0, 0>::value, vnode_base_offset_pair<2, 224,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2, 224,  0, 1>::value, vnode_base_offset_pair<2, 224,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 224,  0, 2>::value, vnode_base_offset_pair<2, 224,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 224,  0, 3>::value, vnode_base_offset_pair<2, 224,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 224,  1, 0>::value, vnode_base_offset_pair<2, 224,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2, 224,  1, 1>::value, vnode_base_offset_pair<2, 224,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 224,  1, 2>::value, vnode_base_offset_pair<2, 224,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 224,  1, 3>::value, vnode_base_offset_pair<2, 224,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 224,  1, 4>::value, vnode_base_offset_pair<2, 224,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 224,  2, 0>::value, vnode_base_offset_pair<2, 224,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2, 224,  2, 1>::value, vnode_base_offset_pair<2, 224,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 224,  2, 2>::value, vnode_base_offset_pair<2, 224,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 224,  2, 3>::value, vnode_base_offset_pair<2, 224,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 224,  3, 0>::value, vnode_base_offset_pair<2, 224,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2, 224,  3, 1>::value, vnode_base_offset_pair<2, 224,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 224,  3, 2>::value, vnode_base_offset_pair<2, 224,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 224,  3, 3>::value, vnode_base_offset_pair<2, 224,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 224,  3, 4>::value, vnode_base_offset_pair<2, 224,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 224,  4, 0>::value, vnode_base_offset_pair<2, 224,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2, 224,  4, 1>::value, vnode_base_offset_pair<2, 224,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224,  5, 0>::value, vnode_base_offset_pair<2, 224,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2, 224,  5, 1>::value, vnode_base_offset_pair<2, 224,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 224,  5, 2>::value, vnode_base_offset_pair<2, 224,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 224,  6, 0>::value, vnode_base_offset_pair<2, 224,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2, 224,  6, 1>::value, vnode_base_offset_pair<2, 224,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 224,  6, 2>::value, vnode_base_offset_pair<2, 224,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 224,  7, 0>::value, vnode_base_offset_pair<2, 224,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2, 224,  7, 1>::value, vnode_base_offset_pair<2, 224,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 224,  7, 2>::value, vnode_base_offset_pair<2, 224,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 224,  8, 0>::value, vnode_base_offset_pair<2, 224,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2, 224,  8, 1>::value, vnode_base_offset_pair<2, 224,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224,  9, 0>::value, vnode_base_offset_pair<2, 224,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2, 224,  9, 1>::value, vnode_base_offset_pair<2, 224,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 224,  9, 2>::value, vnode_base_offset_pair<2, 224,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 10, 0>::value, vnode_base_offset_pair<2, 224, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2, 224, 10, 1>::value, vnode_base_offset_pair<2, 224, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 224, 10, 2>::value, vnode_base_offset_pair<2, 224, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 11, 0>::value, vnode_base_offset_pair<2, 224, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2, 224, 11, 1>::value, vnode_base_offset_pair<2, 224, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 224, 11, 2>::value, vnode_base_offset_pair<2, 224, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 12, 0>::value, vnode_base_offset_pair<2, 224, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2, 224, 12, 1>::value, vnode_base_offset_pair<2, 224, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 13, 0>::value, vnode_base_offset_pair<2, 224, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2, 224, 13, 1>::value, vnode_base_offset_pair<2, 224, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 224, 13, 2>::value, vnode_base_offset_pair<2, 224, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 14, 0>::value, vnode_base_offset_pair<2, 224, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2, 224, 14, 1>::value, vnode_base_offset_pair<2, 224, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 224, 14, 2>::value, vnode_base_offset_pair<2, 224, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 15, 0>::value, vnode_base_offset_pair<2, 224, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2, 224, 15, 1>::value, vnode_base_offset_pair<2, 224, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 16, 0>::value, vnode_base_offset_pair<2, 224, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2, 224, 16, 1>::value, vnode_base_offset_pair<2, 224, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 224, 16, 2>::value, vnode_base_offset_pair<2, 224, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 17, 0>::value, vnode_base_offset_pair<2, 224, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2, 224, 17, 1>::value, vnode_base_offset_pair<2, 224, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 224, 17, 2>::value, vnode_base_offset_pair<2, 224, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 18, 0>::value, vnode_base_offset_pair<2, 224, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2, 224, 18, 1>::value, vnode_base_offset_pair<2, 224, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 19, 0>::value, vnode_base_offset_pair<2, 224, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2, 224, 19, 1>::value, vnode_base_offset_pair<2, 224, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 20, 0>::value, vnode_base_offset_pair<2, 224, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2, 224, 20, 1>::value, vnode_base_offset_pair<2, 224, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 21, 0>::value, vnode_base_offset_pair<2, 224, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2, 224, 21, 1>::value, vnode_base_offset_pair<2, 224, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 22, 0>::value, vnode_base_offset_pair<2, 224, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2, 224, 22, 1>::value, vnode_base_offset_pair<2, 224, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 23, 0>::value, vnode_base_offset_pair<2, 224, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2, 224, 23, 1>::value, vnode_base_offset_pair<2, 224, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 24, 0>::value, vnode_base_offset_pair<2, 224, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2, 224, 24, 1>::value, vnode_base_offset_pair<2, 224, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 25, 0>::value, vnode_base_offset_pair<2, 224, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2, 224, 25, 1>::value, vnode_base_offset_pair<2, 224, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 26, 0>::value, vnode_base_offset_pair<2, 224, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2, 224, 26, 1>::value, vnode_base_offset_pair<2, 224, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 224, 26, 2>::value, vnode_base_offset_pair<2, 224, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 27, 0>::value, vnode_base_offset_pair<2, 224, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2, 224, 27, 1>::value, vnode_base_offset_pair<2, 224, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 28, 0>::value, vnode_base_offset_pair<2, 224, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2, 224, 28, 1>::value, vnode_base_offset_pair<2, 224, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 29, 0>::value, vnode_base_offset_pair<2, 224, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2, 224, 29, 1>::value, vnode_base_offset_pair<2, 224, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 30, 0>::value, vnode_base_offset_pair<2, 224, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2, 224, 30, 1>::value, vnode_base_offset_pair<2, 224, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 224, 30, 2>::value, vnode_base_offset_pair<2, 224, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 31, 0>::value, vnode_base_offset_pair<2, 224, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2, 224, 31, 1>::value, vnode_base_offset_pair<2, 224, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 32, 0>::value, vnode_base_offset_pair<2, 224, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2, 224, 32, 1>::value, vnode_base_offset_pair<2, 224, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 33, 0>::value, vnode_base_offset_pair<2, 224, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2, 224, 33, 1>::value, vnode_base_offset_pair<2, 224, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 34, 0>::value, vnode_base_offset_pair<2, 224, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2, 224, 34, 1>::value, vnode_base_offset_pair<2, 224, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 35, 0>::value, vnode_base_offset_pair<2, 224, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2, 224, 35, 1>::value, vnode_base_offset_pair<2, 224, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 36, 0>::value, vnode_base_offset_pair<2, 224, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2, 224, 36, 1>::value, vnode_base_offset_pair<2, 224, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 37, 0>::value, vnode_base_offset_pair<2, 224, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2, 224, 37, 1>::value, vnode_base_offset_pair<2, 224, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 38, 0>::value, vnode_base_offset_pair<2, 224, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2, 224, 38, 1>::value, vnode_base_offset_pair<2, 224, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 39, 0>::value, vnode_base_offset_pair<2, 224, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2, 224, 39, 1>::value, vnode_base_offset_pair<2, 224, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 40, 0>::value, vnode_base_offset_pair<2, 224, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2, 224, 40, 1>::value, vnode_base_offset_pair<2, 224, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 224, 41, 0>::value, vnode_base_offset_pair<2, 224, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2, 224, 41, 1>::value, vnode_base_offset_pair<2, 224, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z240_8 =
{
    {
        { vnode_shift_mod_pair<2, 240,  0, 0>::value, vnode_base_offset_pair<2, 240,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2, 240,  0, 1>::value, vnode_base_offset_pair<2, 240,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 240,  0, 2>::value, vnode_base_offset_pair<2, 240,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 240,  0, 3>::value, vnode_base_offset_pair<2, 240,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 240,  1, 0>::value, vnode_base_offset_pair<2, 240,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2, 240,  1, 1>::value, vnode_base_offset_pair<2, 240,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 240,  1, 2>::value, vnode_base_offset_pair<2, 240,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 240,  1, 3>::value, vnode_base_offset_pair<2, 240,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 240,  1, 4>::value, vnode_base_offset_pair<2, 240,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 240,  2, 0>::value, vnode_base_offset_pair<2, 240,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2, 240,  2, 1>::value, vnode_base_offset_pair<2, 240,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 240,  2, 2>::value, vnode_base_offset_pair<2, 240,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 240,  2, 3>::value, vnode_base_offset_pair<2, 240,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 240,  3, 0>::value, vnode_base_offset_pair<2, 240,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2, 240,  3, 1>::value, vnode_base_offset_pair<2, 240,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 240,  3, 2>::value, vnode_base_offset_pair<2, 240,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 240,  3, 3>::value, vnode_base_offset_pair<2, 240,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 240,  3, 4>::value, vnode_base_offset_pair<2, 240,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 240,  4, 0>::value, vnode_base_offset_pair<2, 240,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2, 240,  4, 1>::value, vnode_base_offset_pair<2, 240,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240,  5, 0>::value, vnode_base_offset_pair<2, 240,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2, 240,  5, 1>::value, vnode_base_offset_pair<2, 240,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 240,  5, 2>::value, vnode_base_offset_pair<2, 240,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 240,  6, 0>::value, vnode_base_offset_pair<2, 240,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2, 240,  6, 1>::value, vnode_base_offset_pair<2, 240,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 240,  6, 2>::value, vnode_base_offset_pair<2, 240,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 240,  7, 0>::value, vnode_base_offset_pair<2, 240,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2, 240,  7, 1>::value, vnode_base_offset_pair<2, 240,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 240,  7, 2>::value, vnode_base_offset_pair<2, 240,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 240,  8, 0>::value, vnode_base_offset_pair<2, 240,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2, 240,  8, 1>::value, vnode_base_offset_pair<2, 240,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240,  9, 0>::value, vnode_base_offset_pair<2, 240,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2, 240,  9, 1>::value, vnode_base_offset_pair<2, 240,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 240,  9, 2>::value, vnode_base_offset_pair<2, 240,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 10, 0>::value, vnode_base_offset_pair<2, 240, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2, 240, 10, 1>::value, vnode_base_offset_pair<2, 240, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 240, 10, 2>::value, vnode_base_offset_pair<2, 240, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 11, 0>::value, vnode_base_offset_pair<2, 240, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2, 240, 11, 1>::value, vnode_base_offset_pair<2, 240, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 240, 11, 2>::value, vnode_base_offset_pair<2, 240, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 12, 0>::value, vnode_base_offset_pair<2, 240, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2, 240, 12, 1>::value, vnode_base_offset_pair<2, 240, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 13, 0>::value, vnode_base_offset_pair<2, 240, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2, 240, 13, 1>::value, vnode_base_offset_pair<2, 240, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 240, 13, 2>::value, vnode_base_offset_pair<2, 240, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 14, 0>::value, vnode_base_offset_pair<2, 240, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2, 240, 14, 1>::value, vnode_base_offset_pair<2, 240, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 240, 14, 2>::value, vnode_base_offset_pair<2, 240, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 15, 0>::value, vnode_base_offset_pair<2, 240, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2, 240, 15, 1>::value, vnode_base_offset_pair<2, 240, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 16, 0>::value, vnode_base_offset_pair<2, 240, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2, 240, 16, 1>::value, vnode_base_offset_pair<2, 240, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 240, 16, 2>::value, vnode_base_offset_pair<2, 240, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 17, 0>::value, vnode_base_offset_pair<2, 240, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2, 240, 17, 1>::value, vnode_base_offset_pair<2, 240, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 240, 17, 2>::value, vnode_base_offset_pair<2, 240, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 18, 0>::value, vnode_base_offset_pair<2, 240, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2, 240, 18, 1>::value, vnode_base_offset_pair<2, 240, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 19, 0>::value, vnode_base_offset_pair<2, 240, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2, 240, 19, 1>::value, vnode_base_offset_pair<2, 240, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 20, 0>::value, vnode_base_offset_pair<2, 240, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2, 240, 20, 1>::value, vnode_base_offset_pair<2, 240, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 21, 0>::value, vnode_base_offset_pair<2, 240, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2, 240, 21, 1>::value, vnode_base_offset_pair<2, 240, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 22, 0>::value, vnode_base_offset_pair<2, 240, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2, 240, 22, 1>::value, vnode_base_offset_pair<2, 240, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 23, 0>::value, vnode_base_offset_pair<2, 240, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2, 240, 23, 1>::value, vnode_base_offset_pair<2, 240, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 24, 0>::value, vnode_base_offset_pair<2, 240, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2, 240, 24, 1>::value, vnode_base_offset_pair<2, 240, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 25, 0>::value, vnode_base_offset_pair<2, 240, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2, 240, 25, 1>::value, vnode_base_offset_pair<2, 240, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 26, 0>::value, vnode_base_offset_pair<2, 240, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2, 240, 26, 1>::value, vnode_base_offset_pair<2, 240, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 240, 26, 2>::value, vnode_base_offset_pair<2, 240, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 27, 0>::value, vnode_base_offset_pair<2, 240, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2, 240, 27, 1>::value, vnode_base_offset_pair<2, 240, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 28, 0>::value, vnode_base_offset_pair<2, 240, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2, 240, 28, 1>::value, vnode_base_offset_pair<2, 240, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 29, 0>::value, vnode_base_offset_pair<2, 240, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2, 240, 29, 1>::value, vnode_base_offset_pair<2, 240, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 30, 0>::value, vnode_base_offset_pair<2, 240, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2, 240, 30, 1>::value, vnode_base_offset_pair<2, 240, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 240, 30, 2>::value, vnode_base_offset_pair<2, 240, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 31, 0>::value, vnode_base_offset_pair<2, 240, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2, 240, 31, 1>::value, vnode_base_offset_pair<2, 240, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 32, 0>::value, vnode_base_offset_pair<2, 240, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2, 240, 32, 1>::value, vnode_base_offset_pair<2, 240, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 33, 0>::value, vnode_base_offset_pair<2, 240, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2, 240, 33, 1>::value, vnode_base_offset_pair<2, 240, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 34, 0>::value, vnode_base_offset_pair<2, 240, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2, 240, 34, 1>::value, vnode_base_offset_pair<2, 240, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 35, 0>::value, vnode_base_offset_pair<2, 240, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2, 240, 35, 1>::value, vnode_base_offset_pair<2, 240, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 36, 0>::value, vnode_base_offset_pair<2, 240, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2, 240, 36, 1>::value, vnode_base_offset_pair<2, 240, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 37, 0>::value, vnode_base_offset_pair<2, 240, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2, 240, 37, 1>::value, vnode_base_offset_pair<2, 240, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 38, 0>::value, vnode_base_offset_pair<2, 240, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2, 240, 38, 1>::value, vnode_base_offset_pair<2, 240, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 39, 0>::value, vnode_base_offset_pair<2, 240, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2, 240, 39, 1>::value, vnode_base_offset_pair<2, 240, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 40, 0>::value, vnode_base_offset_pair<2, 240, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2, 240, 40, 1>::value, vnode_base_offset_pair<2, 240, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 240, 41, 0>::value, vnode_base_offset_pair<2, 240, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2, 240, 41, 1>::value, vnode_base_offset_pair<2, 240, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z256_8 =
{
    {
        { vnode_shift_mod_pair<2, 256,  0, 0>::value, vnode_base_offset_pair<2, 256,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2, 256,  0, 1>::value, vnode_base_offset_pair<2, 256,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 256,  0, 2>::value, vnode_base_offset_pair<2, 256,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 256,  0, 3>::value, vnode_base_offset_pair<2, 256,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 256,  1, 0>::value, vnode_base_offset_pair<2, 256,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2, 256,  1, 1>::value, vnode_base_offset_pair<2, 256,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 256,  1, 2>::value, vnode_base_offset_pair<2, 256,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 256,  1, 3>::value, vnode_base_offset_pair<2, 256,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 256,  1, 4>::value, vnode_base_offset_pair<2, 256,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 256,  2, 0>::value, vnode_base_offset_pair<2, 256,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2, 256,  2, 1>::value, vnode_base_offset_pair<2, 256,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 256,  2, 2>::value, vnode_base_offset_pair<2, 256,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 256,  2, 3>::value, vnode_base_offset_pair<2, 256,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 256,  3, 0>::value, vnode_base_offset_pair<2, 256,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2, 256,  3, 1>::value, vnode_base_offset_pair<2, 256,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 256,  3, 2>::value, vnode_base_offset_pair<2, 256,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 256,  3, 3>::value, vnode_base_offset_pair<2, 256,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 256,  3, 4>::value, vnode_base_offset_pair<2, 256,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 256,  4, 0>::value, vnode_base_offset_pair<2, 256,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2, 256,  4, 1>::value, vnode_base_offset_pair<2, 256,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256,  5, 0>::value, vnode_base_offset_pair<2, 256,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2, 256,  5, 1>::value, vnode_base_offset_pair<2, 256,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 256,  5, 2>::value, vnode_base_offset_pair<2, 256,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 256,  6, 0>::value, vnode_base_offset_pair<2, 256,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2, 256,  6, 1>::value, vnode_base_offset_pair<2, 256,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 256,  6, 2>::value, vnode_base_offset_pair<2, 256,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 256,  7, 0>::value, vnode_base_offset_pair<2, 256,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2, 256,  7, 1>::value, vnode_base_offset_pair<2, 256,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 256,  7, 2>::value, vnode_base_offset_pair<2, 256,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 256,  8, 0>::value, vnode_base_offset_pair<2, 256,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2, 256,  8, 1>::value, vnode_base_offset_pair<2, 256,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256,  9, 0>::value, vnode_base_offset_pair<2, 256,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2, 256,  9, 1>::value, vnode_base_offset_pair<2, 256,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 256,  9, 2>::value, vnode_base_offset_pair<2, 256,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 10, 0>::value, vnode_base_offset_pair<2, 256, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2, 256, 10, 1>::value, vnode_base_offset_pair<2, 256, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 256, 10, 2>::value, vnode_base_offset_pair<2, 256, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 11, 0>::value, vnode_base_offset_pair<2, 256, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2, 256, 11, 1>::value, vnode_base_offset_pair<2, 256, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 256, 11, 2>::value, vnode_base_offset_pair<2, 256, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 12, 0>::value, vnode_base_offset_pair<2, 256, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2, 256, 12, 1>::value, vnode_base_offset_pair<2, 256, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 13, 0>::value, vnode_base_offset_pair<2, 256, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2, 256, 13, 1>::value, vnode_base_offset_pair<2, 256, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 256, 13, 2>::value, vnode_base_offset_pair<2, 256, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 14, 0>::value, vnode_base_offset_pair<2, 256, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2, 256, 14, 1>::value, vnode_base_offset_pair<2, 256, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 256, 14, 2>::value, vnode_base_offset_pair<2, 256, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 15, 0>::value, vnode_base_offset_pair<2, 256, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2, 256, 15, 1>::value, vnode_base_offset_pair<2, 256, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 16, 0>::value, vnode_base_offset_pair<2, 256, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2, 256, 16, 1>::value, vnode_base_offset_pair<2, 256, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 256, 16, 2>::value, vnode_base_offset_pair<2, 256, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 17, 0>::value, vnode_base_offset_pair<2, 256, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2, 256, 17, 1>::value, vnode_base_offset_pair<2, 256, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 256, 17, 2>::value, vnode_base_offset_pair<2, 256, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 18, 0>::value, vnode_base_offset_pair<2, 256, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2, 256, 18, 1>::value, vnode_base_offset_pair<2, 256, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 19, 0>::value, vnode_base_offset_pair<2, 256, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2, 256, 19, 1>::value, vnode_base_offset_pair<2, 256, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 20, 0>::value, vnode_base_offset_pair<2, 256, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2, 256, 20, 1>::value, vnode_base_offset_pair<2, 256, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 21, 0>::value, vnode_base_offset_pair<2, 256, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2, 256, 21, 1>::value, vnode_base_offset_pair<2, 256, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 22, 0>::value, vnode_base_offset_pair<2, 256, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2, 256, 22, 1>::value, vnode_base_offset_pair<2, 256, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 23, 0>::value, vnode_base_offset_pair<2, 256, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2, 256, 23, 1>::value, vnode_base_offset_pair<2, 256, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 24, 0>::value, vnode_base_offset_pair<2, 256, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2, 256, 24, 1>::value, vnode_base_offset_pair<2, 256, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 25, 0>::value, vnode_base_offset_pair<2, 256, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2, 256, 25, 1>::value, vnode_base_offset_pair<2, 256, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 26, 0>::value, vnode_base_offset_pair<2, 256, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2, 256, 26, 1>::value, vnode_base_offset_pair<2, 256, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 256, 26, 2>::value, vnode_base_offset_pair<2, 256, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 27, 0>::value, vnode_base_offset_pair<2, 256, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2, 256, 27, 1>::value, vnode_base_offset_pair<2, 256, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 28, 0>::value, vnode_base_offset_pair<2, 256, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2, 256, 28, 1>::value, vnode_base_offset_pair<2, 256, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 29, 0>::value, vnode_base_offset_pair<2, 256, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2, 256, 29, 1>::value, vnode_base_offset_pair<2, 256, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 30, 0>::value, vnode_base_offset_pair<2, 256, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2, 256, 30, 1>::value, vnode_base_offset_pair<2, 256, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 256, 30, 2>::value, vnode_base_offset_pair<2, 256, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 31, 0>::value, vnode_base_offset_pair<2, 256, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2, 256, 31, 1>::value, vnode_base_offset_pair<2, 256, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 32, 0>::value, vnode_base_offset_pair<2, 256, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2, 256, 32, 1>::value, vnode_base_offset_pair<2, 256, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 33, 0>::value, vnode_base_offset_pair<2, 256, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2, 256, 33, 1>::value, vnode_base_offset_pair<2, 256, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 34, 0>::value, vnode_base_offset_pair<2, 256, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2, 256, 34, 1>::value, vnode_base_offset_pair<2, 256, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 35, 0>::value, vnode_base_offset_pair<2, 256, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2, 256, 35, 1>::value, vnode_base_offset_pair<2, 256, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 36, 0>::value, vnode_base_offset_pair<2, 256, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2, 256, 36, 1>::value, vnode_base_offset_pair<2, 256, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 37, 0>::value, vnode_base_offset_pair<2, 256, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2, 256, 37, 1>::value, vnode_base_offset_pair<2, 256, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 38, 0>::value, vnode_base_offset_pair<2, 256, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2, 256, 38, 1>::value, vnode_base_offset_pair<2, 256, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 39, 0>::value, vnode_base_offset_pair<2, 256, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2, 256, 39, 1>::value, vnode_base_offset_pair<2, 256, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 40, 0>::value, vnode_base_offset_pair<2, 256, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2, 256, 40, 1>::value, vnode_base_offset_pair<2, 256, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 256, 41, 0>::value, vnode_base_offset_pair<2, 256, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2, 256, 41, 1>::value, vnode_base_offset_pair<2, 256, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z288_8 =
{
    {
        { vnode_shift_mod_pair<2, 288,  0, 0>::value, vnode_base_offset_pair<2, 288,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2, 288,  0, 1>::value, vnode_base_offset_pair<2, 288,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 288,  0, 2>::value, vnode_base_offset_pair<2, 288,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 288,  0, 3>::value, vnode_base_offset_pair<2, 288,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 288,  1, 0>::value, vnode_base_offset_pair<2, 288,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2, 288,  1, 1>::value, vnode_base_offset_pair<2, 288,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 288,  1, 2>::value, vnode_base_offset_pair<2, 288,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 288,  1, 3>::value, vnode_base_offset_pair<2, 288,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 288,  1, 4>::value, vnode_base_offset_pair<2, 288,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 288,  2, 0>::value, vnode_base_offset_pair<2, 288,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2, 288,  2, 1>::value, vnode_base_offset_pair<2, 288,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 288,  2, 2>::value, vnode_base_offset_pair<2, 288,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 288,  2, 3>::value, vnode_base_offset_pair<2, 288,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 288,  3, 0>::value, vnode_base_offset_pair<2, 288,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2, 288,  3, 1>::value, vnode_base_offset_pair<2, 288,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 288,  3, 2>::value, vnode_base_offset_pair<2, 288,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 288,  3, 3>::value, vnode_base_offset_pair<2, 288,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 288,  3, 4>::value, vnode_base_offset_pair<2, 288,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 288,  4, 0>::value, vnode_base_offset_pair<2, 288,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2, 288,  4, 1>::value, vnode_base_offset_pair<2, 288,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288,  5, 0>::value, vnode_base_offset_pair<2, 288,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2, 288,  5, 1>::value, vnode_base_offset_pair<2, 288,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 288,  5, 2>::value, vnode_base_offset_pair<2, 288,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 288,  6, 0>::value, vnode_base_offset_pair<2, 288,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2, 288,  6, 1>::value, vnode_base_offset_pair<2, 288,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 288,  6, 2>::value, vnode_base_offset_pair<2, 288,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 288,  7, 0>::value, vnode_base_offset_pair<2, 288,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2, 288,  7, 1>::value, vnode_base_offset_pair<2, 288,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 288,  7, 2>::value, vnode_base_offset_pair<2, 288,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 288,  8, 0>::value, vnode_base_offset_pair<2, 288,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2, 288,  8, 1>::value, vnode_base_offset_pair<2, 288,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288,  9, 0>::value, vnode_base_offset_pair<2, 288,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2, 288,  9, 1>::value, vnode_base_offset_pair<2, 288,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 288,  9, 2>::value, vnode_base_offset_pair<2, 288,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 10, 0>::value, vnode_base_offset_pair<2, 288, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2, 288, 10, 1>::value, vnode_base_offset_pair<2, 288, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 288, 10, 2>::value, vnode_base_offset_pair<2, 288, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 11, 0>::value, vnode_base_offset_pair<2, 288, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2, 288, 11, 1>::value, vnode_base_offset_pair<2, 288, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 288, 11, 2>::value, vnode_base_offset_pair<2, 288, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 12, 0>::value, vnode_base_offset_pair<2, 288, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2, 288, 12, 1>::value, vnode_base_offset_pair<2, 288, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 13, 0>::value, vnode_base_offset_pair<2, 288, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2, 288, 13, 1>::value, vnode_base_offset_pair<2, 288, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 288, 13, 2>::value, vnode_base_offset_pair<2, 288, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 14, 0>::value, vnode_base_offset_pair<2, 288, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2, 288, 14, 1>::value, vnode_base_offset_pair<2, 288, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 288, 14, 2>::value, vnode_base_offset_pair<2, 288, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 15, 0>::value, vnode_base_offset_pair<2, 288, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2, 288, 15, 1>::value, vnode_base_offset_pair<2, 288, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 16, 0>::value, vnode_base_offset_pair<2, 288, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2, 288, 16, 1>::value, vnode_base_offset_pair<2, 288, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 288, 16, 2>::value, vnode_base_offset_pair<2, 288, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 17, 0>::value, vnode_base_offset_pair<2, 288, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2, 288, 17, 1>::value, vnode_base_offset_pair<2, 288, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 288, 17, 2>::value, vnode_base_offset_pair<2, 288, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 18, 0>::value, vnode_base_offset_pair<2, 288, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2, 288, 18, 1>::value, vnode_base_offset_pair<2, 288, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 19, 0>::value, vnode_base_offset_pair<2, 288, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2, 288, 19, 1>::value, vnode_base_offset_pair<2, 288, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 20, 0>::value, vnode_base_offset_pair<2, 288, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2, 288, 20, 1>::value, vnode_base_offset_pair<2, 288, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 21, 0>::value, vnode_base_offset_pair<2, 288, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2, 288, 21, 1>::value, vnode_base_offset_pair<2, 288, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 22, 0>::value, vnode_base_offset_pair<2, 288, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2, 288, 22, 1>::value, vnode_base_offset_pair<2, 288, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 23, 0>::value, vnode_base_offset_pair<2, 288, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2, 288, 23, 1>::value, vnode_base_offset_pair<2, 288, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 24, 0>::value, vnode_base_offset_pair<2, 288, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2, 288, 24, 1>::value, vnode_base_offset_pair<2, 288, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 25, 0>::value, vnode_base_offset_pair<2, 288, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2, 288, 25, 1>::value, vnode_base_offset_pair<2, 288, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 26, 0>::value, vnode_base_offset_pair<2, 288, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2, 288, 26, 1>::value, vnode_base_offset_pair<2, 288, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 288, 26, 2>::value, vnode_base_offset_pair<2, 288, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 27, 0>::value, vnode_base_offset_pair<2, 288, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2, 288, 27, 1>::value, vnode_base_offset_pair<2, 288, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 28, 0>::value, vnode_base_offset_pair<2, 288, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2, 288, 28, 1>::value, vnode_base_offset_pair<2, 288, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 29, 0>::value, vnode_base_offset_pair<2, 288, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2, 288, 29, 1>::value, vnode_base_offset_pair<2, 288, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 30, 0>::value, vnode_base_offset_pair<2, 288, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2, 288, 30, 1>::value, vnode_base_offset_pair<2, 288, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 288, 30, 2>::value, vnode_base_offset_pair<2, 288, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 31, 0>::value, vnode_base_offset_pair<2, 288, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2, 288, 31, 1>::value, vnode_base_offset_pair<2, 288, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 32, 0>::value, vnode_base_offset_pair<2, 288, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2, 288, 32, 1>::value, vnode_base_offset_pair<2, 288, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 33, 0>::value, vnode_base_offset_pair<2, 288, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2, 288, 33, 1>::value, vnode_base_offset_pair<2, 288, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 34, 0>::value, vnode_base_offset_pair<2, 288, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2, 288, 34, 1>::value, vnode_base_offset_pair<2, 288, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 35, 0>::value, vnode_base_offset_pair<2, 288, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2, 288, 35, 1>::value, vnode_base_offset_pair<2, 288, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 36, 0>::value, vnode_base_offset_pair<2, 288, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2, 288, 36, 1>::value, vnode_base_offset_pair<2, 288, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 37, 0>::value, vnode_base_offset_pair<2, 288, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2, 288, 37, 1>::value, vnode_base_offset_pair<2, 288, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 38, 0>::value, vnode_base_offset_pair<2, 288, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2, 288, 38, 1>::value, vnode_base_offset_pair<2, 288, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 39, 0>::value, vnode_base_offset_pair<2, 288, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2, 288, 39, 1>::value, vnode_base_offset_pair<2, 288, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 40, 0>::value, vnode_base_offset_pair<2, 288, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2, 288, 40, 1>::value, vnode_base_offset_pair<2, 288, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 288, 41, 0>::value, vnode_base_offset_pair<2, 288, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2, 288, 41, 1>::value, vnode_base_offset_pair<2, 288, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z320_8 =
{
    {
        { vnode_shift_mod_pair<2, 320,  0, 0>::value, vnode_base_offset_pair<2, 320,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2, 320,  0, 1>::value, vnode_base_offset_pair<2, 320,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 320,  0, 2>::value, vnode_base_offset_pair<2, 320,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 320,  0, 3>::value, vnode_base_offset_pair<2, 320,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 320,  1, 0>::value, vnode_base_offset_pair<2, 320,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2, 320,  1, 1>::value, vnode_base_offset_pair<2, 320,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 320,  1, 2>::value, vnode_base_offset_pair<2, 320,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 320,  1, 3>::value, vnode_base_offset_pair<2, 320,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 320,  1, 4>::value, vnode_base_offset_pair<2, 320,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 320,  2, 0>::value, vnode_base_offset_pair<2, 320,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2, 320,  2, 1>::value, vnode_base_offset_pair<2, 320,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 320,  2, 2>::value, vnode_base_offset_pair<2, 320,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 320,  2, 3>::value, vnode_base_offset_pair<2, 320,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 320,  3, 0>::value, vnode_base_offset_pair<2, 320,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2, 320,  3, 1>::value, vnode_base_offset_pair<2, 320,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 320,  3, 2>::value, vnode_base_offset_pair<2, 320,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 320,  3, 3>::value, vnode_base_offset_pair<2, 320,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 320,  3, 4>::value, vnode_base_offset_pair<2, 320,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 320,  4, 0>::value, vnode_base_offset_pair<2, 320,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2, 320,  4, 1>::value, vnode_base_offset_pair<2, 320,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320,  5, 0>::value, vnode_base_offset_pair<2, 320,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2, 320,  5, 1>::value, vnode_base_offset_pair<2, 320,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 320,  5, 2>::value, vnode_base_offset_pair<2, 320,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 320,  6, 0>::value, vnode_base_offset_pair<2, 320,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2, 320,  6, 1>::value, vnode_base_offset_pair<2, 320,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 320,  6, 2>::value, vnode_base_offset_pair<2, 320,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 320,  7, 0>::value, vnode_base_offset_pair<2, 320,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2, 320,  7, 1>::value, vnode_base_offset_pair<2, 320,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 320,  7, 2>::value, vnode_base_offset_pair<2, 320,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 320,  8, 0>::value, vnode_base_offset_pair<2, 320,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2, 320,  8, 1>::value, vnode_base_offset_pair<2, 320,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320,  9, 0>::value, vnode_base_offset_pair<2, 320,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2, 320,  9, 1>::value, vnode_base_offset_pair<2, 320,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 320,  9, 2>::value, vnode_base_offset_pair<2, 320,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 10, 0>::value, vnode_base_offset_pair<2, 320, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2, 320, 10, 1>::value, vnode_base_offset_pair<2, 320, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 320, 10, 2>::value, vnode_base_offset_pair<2, 320, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 11, 0>::value, vnode_base_offset_pair<2, 320, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2, 320, 11, 1>::value, vnode_base_offset_pair<2, 320, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 320, 11, 2>::value, vnode_base_offset_pair<2, 320, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 12, 0>::value, vnode_base_offset_pair<2, 320, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2, 320, 12, 1>::value, vnode_base_offset_pair<2, 320, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 13, 0>::value, vnode_base_offset_pair<2, 320, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2, 320, 13, 1>::value, vnode_base_offset_pair<2, 320, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 320, 13, 2>::value, vnode_base_offset_pair<2, 320, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 14, 0>::value, vnode_base_offset_pair<2, 320, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2, 320, 14, 1>::value, vnode_base_offset_pair<2, 320, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 320, 14, 2>::value, vnode_base_offset_pair<2, 320, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 15, 0>::value, vnode_base_offset_pair<2, 320, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2, 320, 15, 1>::value, vnode_base_offset_pair<2, 320, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 16, 0>::value, vnode_base_offset_pair<2, 320, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2, 320, 16, 1>::value, vnode_base_offset_pair<2, 320, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 320, 16, 2>::value, vnode_base_offset_pair<2, 320, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 17, 0>::value, vnode_base_offset_pair<2, 320, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2, 320, 17, 1>::value, vnode_base_offset_pair<2, 320, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 320, 17, 2>::value, vnode_base_offset_pair<2, 320, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 18, 0>::value, vnode_base_offset_pair<2, 320, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2, 320, 18, 1>::value, vnode_base_offset_pair<2, 320, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 19, 0>::value, vnode_base_offset_pair<2, 320, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2, 320, 19, 1>::value, vnode_base_offset_pair<2, 320, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 20, 0>::value, vnode_base_offset_pair<2, 320, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2, 320, 20, 1>::value, vnode_base_offset_pair<2, 320, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 21, 0>::value, vnode_base_offset_pair<2, 320, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2, 320, 21, 1>::value, vnode_base_offset_pair<2, 320, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 22, 0>::value, vnode_base_offset_pair<2, 320, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2, 320, 22, 1>::value, vnode_base_offset_pair<2, 320, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 23, 0>::value, vnode_base_offset_pair<2, 320, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2, 320, 23, 1>::value, vnode_base_offset_pair<2, 320, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 24, 0>::value, vnode_base_offset_pair<2, 320, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2, 320, 24, 1>::value, vnode_base_offset_pair<2, 320, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 25, 0>::value, vnode_base_offset_pair<2, 320, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2, 320, 25, 1>::value, vnode_base_offset_pair<2, 320, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 26, 0>::value, vnode_base_offset_pair<2, 320, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2, 320, 26, 1>::value, vnode_base_offset_pair<2, 320, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 320, 26, 2>::value, vnode_base_offset_pair<2, 320, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 27, 0>::value, vnode_base_offset_pair<2, 320, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2, 320, 27, 1>::value, vnode_base_offset_pair<2, 320, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 28, 0>::value, vnode_base_offset_pair<2, 320, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2, 320, 28, 1>::value, vnode_base_offset_pair<2, 320, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 29, 0>::value, vnode_base_offset_pair<2, 320, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2, 320, 29, 1>::value, vnode_base_offset_pair<2, 320, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 30, 0>::value, vnode_base_offset_pair<2, 320, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2, 320, 30, 1>::value, vnode_base_offset_pair<2, 320, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 320, 30, 2>::value, vnode_base_offset_pair<2, 320, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 31, 0>::value, vnode_base_offset_pair<2, 320, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2, 320, 31, 1>::value, vnode_base_offset_pair<2, 320, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 32, 0>::value, vnode_base_offset_pair<2, 320, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2, 320, 32, 1>::value, vnode_base_offset_pair<2, 320, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 33, 0>::value, vnode_base_offset_pair<2, 320, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2, 320, 33, 1>::value, vnode_base_offset_pair<2, 320, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 34, 0>::value, vnode_base_offset_pair<2, 320, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2, 320, 34, 1>::value, vnode_base_offset_pair<2, 320, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 35, 0>::value, vnode_base_offset_pair<2, 320, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2, 320, 35, 1>::value, vnode_base_offset_pair<2, 320, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 36, 0>::value, vnode_base_offset_pair<2, 320, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2, 320, 36, 1>::value, vnode_base_offset_pair<2, 320, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 37, 0>::value, vnode_base_offset_pair<2, 320, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2, 320, 37, 1>::value, vnode_base_offset_pair<2, 320, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 38, 0>::value, vnode_base_offset_pair<2, 320, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2, 320, 38, 1>::value, vnode_base_offset_pair<2, 320, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 39, 0>::value, vnode_base_offset_pair<2, 320, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2, 320, 39, 1>::value, vnode_base_offset_pair<2, 320, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 40, 0>::value, vnode_base_offset_pair<2, 320, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2, 320, 40, 1>::value, vnode_base_offset_pair<2, 320, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 320, 41, 0>::value, vnode_base_offset_pair<2, 320, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2, 320, 41, 1>::value, vnode_base_offset_pair<2, 320, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z352_8 =
{
    {
        { vnode_shift_mod_pair<2, 352,  0, 0>::value, vnode_base_offset_pair<2, 352,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2, 352,  0, 1>::value, vnode_base_offset_pair<2, 352,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 352,  0, 2>::value, vnode_base_offset_pair<2, 352,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 352,  0, 3>::value, vnode_base_offset_pair<2, 352,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 352,  1, 0>::value, vnode_base_offset_pair<2, 352,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2, 352,  1, 1>::value, vnode_base_offset_pair<2, 352,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 352,  1, 2>::value, vnode_base_offset_pair<2, 352,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 352,  1, 3>::value, vnode_base_offset_pair<2, 352,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 352,  1, 4>::value, vnode_base_offset_pair<2, 352,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 352,  2, 0>::value, vnode_base_offset_pair<2, 352,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2, 352,  2, 1>::value, vnode_base_offset_pair<2, 352,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 352,  2, 2>::value, vnode_base_offset_pair<2, 352,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 352,  2, 3>::value, vnode_base_offset_pair<2, 352,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 352,  3, 0>::value, vnode_base_offset_pair<2, 352,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2, 352,  3, 1>::value, vnode_base_offset_pair<2, 352,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 352,  3, 2>::value, vnode_base_offset_pair<2, 352,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 352,  3, 3>::value, vnode_base_offset_pair<2, 352,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 352,  3, 4>::value, vnode_base_offset_pair<2, 352,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 352,  4, 0>::value, vnode_base_offset_pair<2, 352,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2, 352,  4, 1>::value, vnode_base_offset_pair<2, 352,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352,  5, 0>::value, vnode_base_offset_pair<2, 352,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2, 352,  5, 1>::value, vnode_base_offset_pair<2, 352,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 352,  5, 2>::value, vnode_base_offset_pair<2, 352,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 352,  6, 0>::value, vnode_base_offset_pair<2, 352,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2, 352,  6, 1>::value, vnode_base_offset_pair<2, 352,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 352,  6, 2>::value, vnode_base_offset_pair<2, 352,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 352,  7, 0>::value, vnode_base_offset_pair<2, 352,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2, 352,  7, 1>::value, vnode_base_offset_pair<2, 352,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 352,  7, 2>::value, vnode_base_offset_pair<2, 352,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 352,  8, 0>::value, vnode_base_offset_pair<2, 352,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2, 352,  8, 1>::value, vnode_base_offset_pair<2, 352,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352,  9, 0>::value, vnode_base_offset_pair<2, 352,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2, 352,  9, 1>::value, vnode_base_offset_pair<2, 352,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 352,  9, 2>::value, vnode_base_offset_pair<2, 352,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 10, 0>::value, vnode_base_offset_pair<2, 352, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2, 352, 10, 1>::value, vnode_base_offset_pair<2, 352, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 352, 10, 2>::value, vnode_base_offset_pair<2, 352, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 11, 0>::value, vnode_base_offset_pair<2, 352, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2, 352, 11, 1>::value, vnode_base_offset_pair<2, 352, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 352, 11, 2>::value, vnode_base_offset_pair<2, 352, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 12, 0>::value, vnode_base_offset_pair<2, 352, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2, 352, 12, 1>::value, vnode_base_offset_pair<2, 352, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 13, 0>::value, vnode_base_offset_pair<2, 352, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2, 352, 13, 1>::value, vnode_base_offset_pair<2, 352, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 352, 13, 2>::value, vnode_base_offset_pair<2, 352, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 14, 0>::value, vnode_base_offset_pair<2, 352, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2, 352, 14, 1>::value, vnode_base_offset_pair<2, 352, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 352, 14, 2>::value, vnode_base_offset_pair<2, 352, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 15, 0>::value, vnode_base_offset_pair<2, 352, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2, 352, 15, 1>::value, vnode_base_offset_pair<2, 352, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 16, 0>::value, vnode_base_offset_pair<2, 352, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2, 352, 16, 1>::value, vnode_base_offset_pair<2, 352, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 352, 16, 2>::value, vnode_base_offset_pair<2, 352, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 17, 0>::value, vnode_base_offset_pair<2, 352, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2, 352, 17, 1>::value, vnode_base_offset_pair<2, 352, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 352, 17, 2>::value, vnode_base_offset_pair<2, 352, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 18, 0>::value, vnode_base_offset_pair<2, 352, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2, 352, 18, 1>::value, vnode_base_offset_pair<2, 352, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 19, 0>::value, vnode_base_offset_pair<2, 352, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2, 352, 19, 1>::value, vnode_base_offset_pair<2, 352, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 20, 0>::value, vnode_base_offset_pair<2, 352, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2, 352, 20, 1>::value, vnode_base_offset_pair<2, 352, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 21, 0>::value, vnode_base_offset_pair<2, 352, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2, 352, 21, 1>::value, vnode_base_offset_pair<2, 352, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 22, 0>::value, vnode_base_offset_pair<2, 352, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2, 352, 22, 1>::value, vnode_base_offset_pair<2, 352, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 23, 0>::value, vnode_base_offset_pair<2, 352, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2, 352, 23, 1>::value, vnode_base_offset_pair<2, 352, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 24, 0>::value, vnode_base_offset_pair<2, 352, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2, 352, 24, 1>::value, vnode_base_offset_pair<2, 352, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 25, 0>::value, vnode_base_offset_pair<2, 352, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2, 352, 25, 1>::value, vnode_base_offset_pair<2, 352, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 26, 0>::value, vnode_base_offset_pair<2, 352, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2, 352, 26, 1>::value, vnode_base_offset_pair<2, 352, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 352, 26, 2>::value, vnode_base_offset_pair<2, 352, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 27, 0>::value, vnode_base_offset_pair<2, 352, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2, 352, 27, 1>::value, vnode_base_offset_pair<2, 352, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 28, 0>::value, vnode_base_offset_pair<2, 352, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2, 352, 28, 1>::value, vnode_base_offset_pair<2, 352, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 29, 0>::value, vnode_base_offset_pair<2, 352, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2, 352, 29, 1>::value, vnode_base_offset_pair<2, 352, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 30, 0>::value, vnode_base_offset_pair<2, 352, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2, 352, 30, 1>::value, vnode_base_offset_pair<2, 352, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 352, 30, 2>::value, vnode_base_offset_pair<2, 352, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 31, 0>::value, vnode_base_offset_pair<2, 352, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2, 352, 31, 1>::value, vnode_base_offset_pair<2, 352, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 32, 0>::value, vnode_base_offset_pair<2, 352, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2, 352, 32, 1>::value, vnode_base_offset_pair<2, 352, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 33, 0>::value, vnode_base_offset_pair<2, 352, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2, 352, 33, 1>::value, vnode_base_offset_pair<2, 352, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 34, 0>::value, vnode_base_offset_pair<2, 352, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2, 352, 34, 1>::value, vnode_base_offset_pair<2, 352, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 35, 0>::value, vnode_base_offset_pair<2, 352, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2, 352, 35, 1>::value, vnode_base_offset_pair<2, 352, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 36, 0>::value, vnode_base_offset_pair<2, 352, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2, 352, 36, 1>::value, vnode_base_offset_pair<2, 352, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 37, 0>::value, vnode_base_offset_pair<2, 352, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2, 352, 37, 1>::value, vnode_base_offset_pair<2, 352, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 38, 0>::value, vnode_base_offset_pair<2, 352, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2, 352, 38, 1>::value, vnode_base_offset_pair<2, 352, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 39, 0>::value, vnode_base_offset_pair<2, 352, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2, 352, 39, 1>::value, vnode_base_offset_pair<2, 352, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 40, 0>::value, vnode_base_offset_pair<2, 352, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2, 352, 40, 1>::value, vnode_base_offset_pair<2, 352, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 352, 41, 0>::value, vnode_base_offset_pair<2, 352, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2, 352, 41, 1>::value, vnode_base_offset_pair<2, 352, 41, 1>::value * 1 }
    }
};

const BG2_desc_t BG2_desc_Z384_8 =
{
    {
        { vnode_shift_mod_pair<2, 384,  0, 0>::value, vnode_base_offset_pair<2, 384,  0, 0>::value * 1 }, // Row 0, degree = 8
        { vnode_shift_mod_pair<2, 384,  0, 1>::value, vnode_base_offset_pair<2, 384,  0, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 384,  0, 2>::value, vnode_base_offset_pair<2, 384,  0, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 384,  0, 3>::value, vnode_base_offset_pair<2, 384,  0, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 384,  1, 0>::value, vnode_base_offset_pair<2, 384,  1, 0>::value * 1 }, // Row 1, degree = 10
        { vnode_shift_mod_pair<2, 384,  1, 1>::value, vnode_base_offset_pair<2, 384,  1, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 384,  1, 2>::value, vnode_base_offset_pair<2, 384,  1, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 384,  1, 3>::value, vnode_base_offset_pair<2, 384,  1, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 384,  1, 4>::value, vnode_base_offset_pair<2, 384,  1, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 384,  2, 0>::value, vnode_base_offset_pair<2, 384,  2, 0>::value * 1 }, // Row 2, degree = 8
        { vnode_shift_mod_pair<2, 384,  2, 1>::value, vnode_base_offset_pair<2, 384,  2, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 384,  2, 2>::value, vnode_base_offset_pair<2, 384,  2, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 384,  2, 3>::value, vnode_base_offset_pair<2, 384,  2, 3>::value * 1 },

        { vnode_shift_mod_pair<2, 384,  3, 0>::value, vnode_base_offset_pair<2, 384,  3, 0>::value * 1 }, // Row 3, degree = 10
        { vnode_shift_mod_pair<2, 384,  3, 1>::value, vnode_base_offset_pair<2, 384,  3, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 384,  3, 2>::value, vnode_base_offset_pair<2, 384,  3, 2>::value * 1 },
        { vnode_shift_mod_pair<2, 384,  3, 3>::value, vnode_base_offset_pair<2, 384,  3, 3>::value * 1 },
        { vnode_shift_mod_pair<2, 384,  3, 4>::value, vnode_base_offset_pair<2, 384,  3, 4>::value * 1 },

        { vnode_shift_mod_pair<2, 384,  4, 0>::value, vnode_base_offset_pair<2, 384,  4, 0>::value * 1 }, // Row 4, degree = 4
        { vnode_shift_mod_pair<2, 384,  4, 1>::value, vnode_base_offset_pair<2, 384,  4, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384,  5, 0>::value, vnode_base_offset_pair<2, 384,  5, 0>::value * 1 }, // Row 5, degree = 6
        { vnode_shift_mod_pair<2, 384,  5, 1>::value, vnode_base_offset_pair<2, 384,  5, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 384,  5, 2>::value, vnode_base_offset_pair<2, 384,  5, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 384,  6, 0>::value, vnode_base_offset_pair<2, 384,  6, 0>::value * 1 }, // Row 6, degree = 6
        { vnode_shift_mod_pair<2, 384,  6, 1>::value, vnode_base_offset_pair<2, 384,  6, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 384,  6, 2>::value, vnode_base_offset_pair<2, 384,  6, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 384,  7, 0>::value, vnode_base_offset_pair<2, 384,  7, 0>::value * 1 }, // Row 7, degree = 6
        { vnode_shift_mod_pair<2, 384,  7, 1>::value, vnode_base_offset_pair<2, 384,  7, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 384,  7, 2>::value, vnode_base_offset_pair<2, 384,  7, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 384,  8, 0>::value, vnode_base_offset_pair<2, 384,  8, 0>::value * 1 }, // Row 8, degree = 4
        { vnode_shift_mod_pair<2, 384,  8, 1>::value, vnode_base_offset_pair<2, 384,  8, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384,  9, 0>::value, vnode_base_offset_pair<2, 384,  9, 0>::value * 1 }, // Row 9, degree = 5
        { vnode_shift_mod_pair<2, 384,  9, 1>::value, vnode_base_offset_pair<2, 384,  9, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 384,  9, 2>::value, vnode_base_offset_pair<2, 384,  9, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 10, 0>::value, vnode_base_offset_pair<2, 384, 10, 0>::value * 1 }, // Row 10, degree = 5
        { vnode_shift_mod_pair<2, 384, 10, 1>::value, vnode_base_offset_pair<2, 384, 10, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 384, 10, 2>::value, vnode_base_offset_pair<2, 384, 10, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 11, 0>::value, vnode_base_offset_pair<2, 384, 11, 0>::value * 1 }, // Row 11, degree = 5
        { vnode_shift_mod_pair<2, 384, 11, 1>::value, vnode_base_offset_pair<2, 384, 11, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 384, 11, 2>::value, vnode_base_offset_pair<2, 384, 11, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 12, 0>::value, vnode_base_offset_pair<2, 384, 12, 0>::value * 1 }, // Row 12, degree = 4
        { vnode_shift_mod_pair<2, 384, 12, 1>::value, vnode_base_offset_pair<2, 384, 12, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 13, 0>::value, vnode_base_offset_pair<2, 384, 13, 0>::value * 1 }, // Row 13, degree = 5
        { vnode_shift_mod_pair<2, 384, 13, 1>::value, vnode_base_offset_pair<2, 384, 13, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 384, 13, 2>::value, vnode_base_offset_pair<2, 384, 13, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 14, 0>::value, vnode_base_offset_pair<2, 384, 14, 0>::value * 1 }, // Row 14, degree = 5
        { vnode_shift_mod_pair<2, 384, 14, 1>::value, vnode_base_offset_pair<2, 384, 14, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 384, 14, 2>::value, vnode_base_offset_pair<2, 384, 14, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 15, 0>::value, vnode_base_offset_pair<2, 384, 15, 0>::value * 1 }, // Row 15, degree = 4
        { vnode_shift_mod_pair<2, 384, 15, 1>::value, vnode_base_offset_pair<2, 384, 15, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 16, 0>::value, vnode_base_offset_pair<2, 384, 16, 0>::value * 1 }, // Row 16, degree = 5
        { vnode_shift_mod_pair<2, 384, 16, 1>::value, vnode_base_offset_pair<2, 384, 16, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 384, 16, 2>::value, vnode_base_offset_pair<2, 384, 16, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 17, 0>::value, vnode_base_offset_pair<2, 384, 17, 0>::value * 1 }, // Row 17, degree = 5
        { vnode_shift_mod_pair<2, 384, 17, 1>::value, vnode_base_offset_pair<2, 384, 17, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 384, 17, 2>::value, vnode_base_offset_pair<2, 384, 17, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 18, 0>::value, vnode_base_offset_pair<2, 384, 18, 0>::value * 1 }, // Row 18, degree = 4
        { vnode_shift_mod_pair<2, 384, 18, 1>::value, vnode_base_offset_pair<2, 384, 18, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 19, 0>::value, vnode_base_offset_pair<2, 384, 19, 0>::value * 1 }, // Row 19, degree = 4
        { vnode_shift_mod_pair<2, 384, 19, 1>::value, vnode_base_offset_pair<2, 384, 19, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 20, 0>::value, vnode_base_offset_pair<2, 384, 20, 0>::value * 1 }, // Row 20, degree = 4
        { vnode_shift_mod_pair<2, 384, 20, 1>::value, vnode_base_offset_pair<2, 384, 20, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 21, 0>::value, vnode_base_offset_pair<2, 384, 21, 0>::value * 1 }, // Row 21, degree = 4
        { vnode_shift_mod_pair<2, 384, 21, 1>::value, vnode_base_offset_pair<2, 384, 21, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 22, 0>::value, vnode_base_offset_pair<2, 384, 22, 0>::value * 1 }, // Row 22, degree = 3
        { vnode_shift_mod_pair<2, 384, 22, 1>::value, vnode_base_offset_pair<2, 384, 22, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 23, 0>::value, vnode_base_offset_pair<2, 384, 23, 0>::value * 1 }, // Row 23, degree = 4
        { vnode_shift_mod_pair<2, 384, 23, 1>::value, vnode_base_offset_pair<2, 384, 23, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 24, 0>::value, vnode_base_offset_pair<2, 384, 24, 0>::value * 1 }, // Row 24, degree = 4
        { vnode_shift_mod_pair<2, 384, 24, 1>::value, vnode_base_offset_pair<2, 384, 24, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 25, 0>::value, vnode_base_offset_pair<2, 384, 25, 0>::value * 1 }, // Row 25, degree = 3
        { vnode_shift_mod_pair<2, 384, 25, 1>::value, vnode_base_offset_pair<2, 384, 25, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 26, 0>::value, vnode_base_offset_pair<2, 384, 26, 0>::value * 1 }, // Row 26, degree = 5
        { vnode_shift_mod_pair<2, 384, 26, 1>::value, vnode_base_offset_pair<2, 384, 26, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 384, 26, 2>::value, vnode_base_offset_pair<2, 384, 26, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 27, 0>::value, vnode_base_offset_pair<2, 384, 27, 0>::value * 1 }, // Row 27, degree = 3
        { vnode_shift_mod_pair<2, 384, 27, 1>::value, vnode_base_offset_pair<2, 384, 27, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 28, 0>::value, vnode_base_offset_pair<2, 384, 28, 0>::value * 1 }, // Row 28, degree = 4
        { vnode_shift_mod_pair<2, 384, 28, 1>::value, vnode_base_offset_pair<2, 384, 28, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 29, 0>::value, vnode_base_offset_pair<2, 384, 29, 0>::value * 1 }, // Row 29, degree = 3
        { vnode_shift_mod_pair<2, 384, 29, 1>::value, vnode_base_offset_pair<2, 384, 29, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 30, 0>::value, vnode_base_offset_pair<2, 384, 30, 0>::value * 1 }, // Row 30, degree = 5
        { vnode_shift_mod_pair<2, 384, 30, 1>::value, vnode_base_offset_pair<2, 384, 30, 1>::value * 1 },
        { vnode_shift_mod_pair<2, 384, 30, 2>::value, vnode_base_offset_pair<2, 384, 30, 2>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 31, 0>::value, vnode_base_offset_pair<2, 384, 31, 0>::value * 1 }, // Row 31, degree = 3
        { vnode_shift_mod_pair<2, 384, 31, 1>::value, vnode_base_offset_pair<2, 384, 31, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 32, 0>::value, vnode_base_offset_pair<2, 384, 32, 0>::value * 1 }, // Row 32, degree = 4
        { vnode_shift_mod_pair<2, 384, 32, 1>::value, vnode_base_offset_pair<2, 384, 32, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 33, 0>::value, vnode_base_offset_pair<2, 384, 33, 0>::value * 1 }, // Row 33, degree = 4
        { vnode_shift_mod_pair<2, 384, 33, 1>::value, vnode_base_offset_pair<2, 384, 33, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 34, 0>::value, vnode_base_offset_pair<2, 384, 34, 0>::value * 1 }, // Row 34, degree = 4
        { vnode_shift_mod_pair<2, 384, 34, 1>::value, vnode_base_offset_pair<2, 384, 34, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 35, 0>::value, vnode_base_offset_pair<2, 384, 35, 0>::value * 1 }, // Row 35, degree = 4
        { vnode_shift_mod_pair<2, 384, 35, 1>::value, vnode_base_offset_pair<2, 384, 35, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 36, 0>::value, vnode_base_offset_pair<2, 384, 36, 0>::value * 1 }, // Row 36, degree = 4
        { vnode_shift_mod_pair<2, 384, 36, 1>::value, vnode_base_offset_pair<2, 384, 36, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 37, 0>::value, vnode_base_offset_pair<2, 384, 37, 0>::value * 1 }, // Row 37, degree = 3
        { vnode_shift_mod_pair<2, 384, 37, 1>::value, vnode_base_offset_pair<2, 384, 37, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 38, 0>::value, vnode_base_offset_pair<2, 384, 38, 0>::value * 1 }, // Row 38, degree = 4
        { vnode_shift_mod_pair<2, 384, 38, 1>::value, vnode_base_offset_pair<2, 384, 38, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 39, 0>::value, vnode_base_offset_pair<2, 384, 39, 0>::value * 1 }, // Row 39, degree = 4
        { vnode_shift_mod_pair<2, 384, 39, 1>::value, vnode_base_offset_pair<2, 384, 39, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 40, 0>::value, vnode_base_offset_pair<2, 384, 40, 0>::value * 1 }, // Row 40, degree = 4
        { vnode_shift_mod_pair<2, 384, 40, 1>::value, vnode_base_offset_pair<2, 384, 40, 1>::value * 1 },

        { vnode_shift_mod_pair<2, 384, 41, 0>::value, vnode_base_offset_pair<2, 384, 41, 0>::value * 1 }, // Row 41, degree = 4
        { vnode_shift_mod_pair<2, 384, 41, 1>::value, vnode_base_offset_pair<2, 384, 41, 1>::value * 1 }
    }
};


} // namespace ldpc2
