/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ENUM_UTILS_HPP_INCLUDED
#define ENUM_UTILS_HPP_INCLUDED

#include <type_traits>
#include <utility>  // For std::to_underlying (C++23)

// =============================================================================
// Enum to underlying type conversion helper (C++17/20 → C++23 compatibility)
// =============================================================================
// Provides std::to_underlying functionality for C++17/20 codebases.
// Automatically uses std::to_underlying when compiling with C++23 or later.
//
// Purpose:
// - Cleanly convert enum class values to their underlying integer types
// - Safer and more explicit than static_cast<int>(enum_value)
// - Forward compatible with C++23 standard library
// - Reusable across all cuPHY modules (nvlog, examples, cuphydriver, etc.)
//
// Implementation:
// - C++23+: Uses std::to_underlying from <utility>
// - C++17/20: Provides inline function template (works with NVCC -std=c++17)
//
// Usage:
//   enum class Status : uint8_t { OK = 0, ERROR = 1 };
//   Status s = Status::OK;
//   auto value = to_underlying(s);  // Returns uint8_t(0)
// =============================================================================

#if __cplusplus >= 202302L && defined(__cpp_lib_to_underlying)
    // C++23 or later: Use standard library version
    using std::to_underlying;
#else
    // C++17/C++20: Provide portable implementation
    /**
     * Convert enum to its underlying integer type
     * 
     * Provides C++23 std::to_underlying compatibility for earlier standards.
     * This is a zero-cost abstraction - generates identical code to static_cast.
     * 
     * @tparam Enum Enum or enum class type
     * @param e The enum value to convert
     * @return The underlying integer value
     * 
     * @note inline ensures ODR compliance when included in multiple translation units
     * @note constexpr allows compile-time evaluation
     * @note noexcept guarantees no exceptions (enables optimizations)
     * 
     * Example:
     * @code
     *   enum class Color : uint8_t { RED = 1, GREEN = 2, BLUE = 3 };
     *   Color c = Color::RED;
     *   uint8_t value = to_underlying(c);  // value == 1
     * @endcode
     */
    template <typename Enum>
    [[nodiscard]] inline constexpr std::underlying_type_t<Enum> to_underlying(Enum e) noexcept
    {
        return static_cast<std::underlying_type_t<Enum>>(e);
    }
#endif

#endif // ENUM_UTILS_HPP_INCLUDED

