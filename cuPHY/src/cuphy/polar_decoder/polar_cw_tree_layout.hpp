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

#ifndef POLAR_CW_TREE_LAYOUT_HPP
#define POLAR_CW_TREE_LAYOUT_HPP

#include <cstddef>

// Layout of the per-UCI-segment codeword-tree buffer that the compCwTreeTypes
// kernel (comp_cwTreeTypes.cu) fills and the polar decoder consumes. For a
// length-N codeword it is four contiguous N-byte regions:
//
//   0               N               2N              3N              4N
//   +---------------+---------------+---------------+---------------+
//   | internal      | leaf          | SC            | fast-SSC      |
//   | node types    | types         | op list       | op list       |
//   +---------------+---------------+---------------+---------------+
//   |<----- treeTypesBytes(N) ----->|< opListBytes >|< opListBytes >|
//
//   [0,  2N)  pruned tree types, heap-indexed: the stage-s node with sub-index
//             k sits at (1 << (n - s)) + k, so internal node types occupy
//             [2, N) and the N leaf types [N, 2N). Entries 0 and 1 are both
//             unused -- index 1 is the root, whose type is never queried
//             (the walk stops at stage n-1), and index 0 has no node at all
//   [2N, 3N)  SC operation list, consumed by the list decoder. Prunes only
//             uniform subtrees (all-frozen or all-info), i.e. simplified SC
//   [3N, 4N)  fast-SSC operation list, consumed by the SC decoder. As above
//             but REP and SPC subtrees stay fused as single entries
//
// Both lists are emitted one byte per visited node in leaf order and are at
// most N entries long, which is why one N-byte region each is always enough.
//
// Every allocation, offset and readback of this buffer goes through the struct
// below, so the four-region contract has a single definition instead of a
// literal 4 (and a copy of this comment) at each site. The accessors are
// constexpr and the CUDA build enables --expt-relaxed-constexpr, so device code
// can call them on a run-time N as well.
namespace cuphy {
namespace polar {

struct PolarCwTreeLayout final
{
    // ---- region sizes, in bytes, for a length-N codeword ----

    // Pruned tree types: internal nodes plus leaves.
    [[nodiscard]] static constexpr std::size_t treeTypesBytes(std::size_t N) noexcept
    {
        return 2 * N;
    }

    // Either operation list; both are capped at one entry per leaf.
    [[nodiscard]] static constexpr std::size_t opListBytes(std::size_t N) noexcept
    {
        return N;
    }

    // The whole buffer.
    [[nodiscard]] static constexpr std::size_t sizeBytes(std::size_t N) noexcept
    {
        return treeTypesBytes(N) + 2 * opListBytes(N);
    }

    // ---- region start offsets, in bytes, derived from the sizes above ----

    // Start of the SC operation list (the tree types occupy the buffer head).
    [[nodiscard]] static constexpr std::size_t scOpListOffset(std::size_t N) noexcept
    {
        return treeTypesBytes(N);
    }

    // Start of the fast-SSC operation list.
    [[nodiscard]] static constexpr std::size_t fssOpListOffset(std::size_t N) noexcept
    {
        return scOpListOffset(N) + opListBytes(N);
    }
};

} // namespace polar
} // namespace cuphy

#endif // POLAR_CW_TREE_LAYOUT_HPP
