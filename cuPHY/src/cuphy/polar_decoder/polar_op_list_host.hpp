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

#ifndef POLAR_OP_LIST_HOST_HPP
#define POLAR_OP_LIST_HOST_HPP

#include <bit>
#include <cstdint>
#include <vector>

// Host reference for the polar successive-cancellation traversal operation lists.
//
// This is the single source of truth that mirrors the compCwTreeTypes device
// kernel (comp_cwTreeTypes.cu) opList emission. The component example and the
// unit-test fixture both call these instead of re-implementing the algorithm, so
// the three no longer drift; a dedicated unit test byte-compares the device
// kernel output against this reference.
//
// `treeTypes` holds the pruned binary tree for a length-N (power-of-two)
// codeword: internal node types at [2, N), the N leaf types at [N, 2N) -- the
// tree-types region of the buffer described in polar_cw_tree_layout.hpp. Node
// types are 0 = frozen, 1 = info, 2 = parity, 3 = mixed. Each returned byte is
// one visited node in leaf order, bits[3:0] = node stage, bits[5:4] = node type.
namespace cuphy {
namespace polar {

// Plain successive-cancellation opList: a leaf is visited as its highest
// ancestor whose subtree is uniform (type != 3).
inline std::vector<uint8_t> buildPolarOpList(const uint8_t* treeTypes, uint16_t N)
{
    // N is a power of two, so the tree depth is the index of its set bit. Kept
    // as uint8_t to match the kernel's n_cw; the stages it bounds reach 9 at
    // the largest supported N.
    const uint8_t n = static_cast<uint8_t>(std::bit_width(N) - 1U);

    std::vector<uint8_t> opList;
    opList.reserve(N);
    uint32_t j = 0;
    while(j < N)
    {
        uint8_t stage = 0;
        uint8_t type  = treeTypes[N + j];
        while(stage < n - 1)
        {
            const uint8_t parentType = treeTypes[(1u << (n - stage - 1)) + (j >> (stage + 1))];
            if(parentType == 3) { break; }
            type = parentType;
            stage++;
        }
        opList.push_back(static_cast<uint8_t>(stage | (type << 4)));
        j += (1u << stage);
    }
    return opList;
}

// Fast-SSC opList: REP subtrees (all leaves frozen except the last, class 4)
// and SPC subtrees (all info except the first, class 5) stay fused as single
// entries. Parity leaves (class 6) block fusion and decode as type-2 leaves.
// The classes reach 5, so each byte packs bits[3:0] = node stage and
// bits[6:4] = node class (wider than the SC list's bits[5:4]).
inline std::vector<uint8_t> buildPolarFssOpList(const uint8_t* treeTypes, uint16_t N)
{
    const uint8_t n = static_cast<uint8_t>(std::bit_width(N) - 1U);

    std::vector<uint8_t> cls(2u * static_cast<size_t>(N), 0);
    for(uint32_t k = 0; k < N; k++)
    {
        const uint8_t t = treeTypes[N + k];
        cls[N + k] = (t == 2) ? 6 : t;
    }
    for(uint16_t s = 1; s < n; s++)
    {
        const uint32_t rowStart = 1u << (n - s);
        for(uint32_t k = 0; k < rowStart; k++)
        {
            const uint8_t cL = cls[2 * (rowStart + k)];
            const uint8_t cR = cls[2 * (rowStart + k) + 1];
            uint8_t c;
            if(cL == 0 && cR == 0)                                { c = 0; }
            else if(cL == 1 && cR == 1)                           { c = 1; }
            else if(cL == 0 && ((s == 1 && cR == 1) || cR == 4))  { c = 4; }
            else if(cR == 1 && ((s == 2 && cL == 4) || cL == 5))  { c = 5; }
            else                                                  { c = 3; }
            cls[rowStart + k] = c;
        }
    }

    std::vector<uint8_t> opList;
    opList.reserve(N);
    uint32_t j = 0;
    while(j < N)
    {
        uint8_t stage = 0;
        uint8_t c     = cls[N + j];
        if(c == 6) { c = 2; }
        while(stage < n - 1)
        {
            const uint8_t pc = cls[(1u << (n - stage - 1)) + (j >> (stage + 1))];
            if(pc == 3 || pc == 6) { break; }
            c = pc;
            stage++;
        }
        opList.push_back(static_cast<uint8_t>(stage | (c << 4)));
        j += (1u << stage);
    }
    return opList;
}

} // namespace polar
} // namespace cuphy

#endif // POLAR_OP_LIST_HOST_HPP
