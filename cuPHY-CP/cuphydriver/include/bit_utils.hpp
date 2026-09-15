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

#ifndef BIT_UTILS_HPP
#define BIT_UTILS_HPP

#include <cstddef>
#include <cstdint>

namespace nv {

/**
 * Lowest @p n bits set, saturating at 64 (avoids the `1 << 64` UB at n >= 64).
 *
 * @param[in] n Number of low bits to set.
 * @return Bitmask with bits [0, min(n, 64)) set.
 */
[[nodiscard]] inline constexpr std::uint64_t lowBitsMask(const std::size_t n) noexcept
{
    constexpr std::size_t MASK_BITS = 64;
    return (n >= MASK_BITS) ? ~std::uint64_t{0} : (std::uint64_t{1} << n) - 1U;
}

static_assert(lowBitsMask(0) == 0x0000000000000000ULL);
static_assert(lowBitsMask(4) == 0x000000000000000FULL);
static_assert(lowBitsMask(63) == 0x7FFFFFFFFFFFFFFFULL);
static_assert(lowBitsMask(64) == 0xFFFFFFFFFFFFFFFFULL);
static_assert(lowBitsMask(65) == 0xFFFFFFFFFFFFFFFFULL);

} // namespace nv

#endif // BIT_UTILS_HPP
