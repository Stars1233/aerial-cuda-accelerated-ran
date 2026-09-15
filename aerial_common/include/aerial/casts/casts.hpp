// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef AERIAL_CASTS_CASTS_HPP
#define AERIAL_CASTS_CASTS_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <gsl-lite/gsl-lite.hpp>

namespace aerial::casts {

/**
 * @brief Cast a void* to T* with compile-time type safety and runtime alignment validation.
 *
 * Centralizes reinterpret_cast so linter suppressions are in one place. T must be
 * standard-layout and trivial; the caller must ensure the pointer points to
 * sufficient, correctly-initialized memory.
 *
 * @note For packed structs where alignof(T) == 1, the runtime alignment check
 *       reduces to a trivially-satisfied byte-alignment check (ptr % 1 == 0).
 *       The compile-time static_assert constraints still apply.
 * @warning Technical UB: reinterpret_cast on memory that may not hold a properly
 *          constructed C++ object violates strict aliasing. This pattern is
 *          accepted in systems/network code and recognised by modern compilers;
 *          alignment validation catches the most common real-world failures.
 *
 * @tparam T  Target type — must be standard-layout, trivial, and alignof(T) <= alignof(std::max_align_t).
 * @param[in] ptr Source void pointer; caller must ensure it points to memory
 *                suitably sized and aligned for T, or is null.
 * @return Pointer of type T* at the same address as ptr.
 */
template <typename T>
// cppcheck-suppress constParameterPointer
[[nodiscard]] inline T *assume_cast(void *ptr)
{
    static_assert(
            alignof(T) <= alignof(std::max_align_t),
            "Type T has alignment requirements that may not be satisfied by arbitrary pointers");
    static_assert(std::is_standard_layout_v<T>, "Type T must be standard layout for safe casting");
    static_assert(std::is_trivial_v<T>, "Type T must be trivial for safe casting");

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto ptr_value = reinterpret_cast<std::uintptr_t>(ptr);
    gsl_Expects(ptr_value % alignof(T) == 0);

    return static_cast<T *>(ptr);
}

/**
 * @brief Cast a const void* to const T* with compile-time type safety and runtime alignment validation.
 *
 * Const-qualified variant of assume_cast for read-only access patterns.
 *
 * @tparam T  Target type — must be standard-layout, trivial, and alignof(T) <= alignof(std::max_align_t).
 * @param[in] ptr Source const void pointer; caller must ensure it points to memory
 *                suitably sized and aligned for T, or is null.
 * @return Const pointer of type const T* at the same address as ptr.
 */
template <typename T>
[[nodiscard]] inline const T *assume_cast(const void *ptr)
{
    static_assert(
            alignof(T) <= alignof(std::max_align_t),
            "Type T has alignment requirements that may not be satisfied by arbitrary pointers");
    static_assert(std::is_standard_layout_v<T>, "Type T must be standard layout for safe casting");
    static_assert(std::is_trivial_v<T>, "Type T must be trivial for safe casting");

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto ptr_value = reinterpret_cast<std::uintptr_t>(ptr);
    gsl_Expects(ptr_value % alignof(T) == 0);

    return static_cast<const T *>(ptr);
}

/**
 * @brief Reinterpret a reference to U as a reference to T with compile-time and runtime alignment validation.
 *
 * The compile-time static_assert (alignof(T) <= alignof(U)) proves that any valid U reference
 * is sufficiently aligned for T. The runtime gsl_Expects adds defense-in-depth for cases where
 * the caller obtained the reference via a prior unsafe cast that bypassed alignment guarantees.
 *
 * @tparam T  Target type — must be standard-layout, trivial, and alignof(T) <= alignof(U).
 * @tparam U  Source type — determines the compile-time alignment guarantee.
 * @param[in] ref Reference to a U object whose storage will be reinterpreted as T.
 * @return Mutable reference to T at the same address as ref.
 */
template <typename T, typename U>
[[nodiscard]] inline T &assume_cast_ref(U &ref)
{
    static_assert(
            alignof(T) <= alignof(U),
            "Type T requires stricter alignment than source type U provides");
    static_assert(std::is_standard_layout_v<T>, "Type T must be standard layout for safe casting");
    static_assert(std::is_trivial_v<T>, "Type T must be trivial for safe casting");

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto ptr_value = reinterpret_cast<std::uintptr_t>(&ref);
    gsl_Expects(ptr_value % alignof(T) == 0);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return *reinterpret_cast<T *>(&ref);
}

/**
 * @brief Reinterpret a const reference to U as a const reference to T with compile-time and runtime alignment validation.
 *
 * Const-qualified variant of assume_cast_ref for read-only access patterns.
 *
 * @tparam T  Target type — must be standard-layout, trivial, and alignof(T) <= alignof(U).
 * @tparam U  Source type — determines the compile-time alignment guarantee.
 * @param[in] ref Const reference to a U object whose storage will be reinterpreted as T.
 * @return Const reference to T at the same address as ref.
 */
template <typename T, typename U>
[[nodiscard]] inline const T &assume_cast_ref(const U &ref)
{
    static_assert(
            alignof(T) <= alignof(U),
            "Type T requires stricter alignment than source type U provides");
    static_assert(std::is_standard_layout_v<T>, "Type T must be standard layout for safe casting");
    static_assert(std::is_trivial_v<T>, "Type T must be trivial for safe casting");

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto ptr_value = reinterpret_cast<std::uintptr_t>(&ref);
    gsl_Expects(ptr_value % alignof(T) == 0);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return *reinterpret_cast<const T *>(&ref);
}

} // namespace aerial::casts

#endif // AERIAL_CASTS_CASTS_HPP
