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

#ifndef TESTBENCHES_GPU3GPPCHAN_PYTHON_NANOBIND_CUDA_ARRAY_INTERFACE_HPP
#define TESTBENCHES_GPU3GPPCHAN_PYTHON_NANOBIND_CUDA_ARRAY_INTERFACE_HPP

#include <nanobind/nanobind.h>
#include <vector>
#include <complex>
#include <cstdint>
#include <string>

namespace gpu3gppchan_bindings {

/**
 * Convert a vector of dimensions or strides to a Python tuple.
 *
 * @param[in] vec Values to copy into the tuple.
 * @return Python tuple containing the vector values.
 */
[[nodiscard]] nanobind::tuple as_tuple(const std::vector<size_t>& vec);

/**
 * Non-owning CUDA-array view with optional producer lifetime retention.
 *
 * Implements version 3 of `__cuda_array_interface__`.
 *
 * @tparam T Element type described by the CUDA array interface.
 */
template <typename T>
class cuda_array_t final
{
public:
    /**
     * Construct a view from a raw device address and explicit layout.
     *
     * @param[in] addr Device address. Must be nonzero for a nonempty array.
     * @param[in] shape Array dimensions. Must not be empty.
     * @param[in] strides Byte strides, or an empty vector for contiguous layout.
     * @param[in] readonly Whether writes through the view are forbidden.
     * @throws std::runtime_error If `shape` is empty.
     * @throws std::invalid_argument If a nonempty array has a null address.
     */
    explicit cuda_array_t(intptr_t addr,
                          const std::vector<size_t> &shape,
                          const std::vector<size_t> &strides,
                          bool readonly = false);

    /**
     * Construct a view from an object exposing `__cuda_array_interface__`.
     *
     * Retains `array` so its device allocation remains valid for this view.
     *
     * @param[in] array CUDA-array producer object.
     * @throws std::runtime_error If shape or dtype metadata is invalid.
     * @throws std::invalid_argument If a nonempty array has a null address.
     */
    explicit cuda_array_t(const nanobind::object& array);

    /** @return Device data pointer, or `nullptr` for an empty array. */
    [[nodiscard]] void* get_device_ptr() const { return device_ptr; }
    /** @return Copy of the array dimensions. */
    [[nodiscard]] auto get_shape() const { return shape_; }
    /** @return Number of array dimensions. */
    [[nodiscard]] auto get_ndim() const { return shape_.size(); }
    /** @return Copy of the byte strides, or an empty vector if unspecified. */
    [[nodiscard]] auto get_strides() const { return strides_; }
    /** @return Whether writes through the view are forbidden. */
    [[nodiscard]] bool is_readonly() const { return readonly; }
    /** @return Whether explicit stride metadata is present. */
    [[nodiscard]] bool has_stride_info() const { return has_strides; }
    /** @return Whether the view has a C-contiguous layout. */
    [[nodiscard]] bool is_c_contiguous() const;
    /** @return Whether the view has a Fortran-contiguous layout. */
    [[nodiscard]] bool is_f_contiguous() const;
    /** @return Product of all shape dimensions. */
    [[nodiscard]] size_t get_size() const;
    /** @return Version 3 CUDA-array-interface dictionary for this view. */
    [[nodiscard]] nanobind::dict get_interface_dict() const;

private:
    void* device_ptr{};
    nanobind::object owner_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    bool readonly{};
    bool has_strides{};

    void verify_dtype(const std::string &typestr) const;
    [[nodiscard]] std::string get_typestr() const;
};

using cuda_array_float = cuda_array_t<float>;
using cuda_array_complex_float = cuda_array_t<std::complex<float>>;
using cuda_array_uint32 = cuda_array_t<uint32_t>;
using cuda_array_uint8 = cuda_array_t<uint8_t>;


}  // namespace gpu3gppchan_bindings

#endif  // TESTBENCHES_GPU3GPPCHAN_PYTHON_NANOBIND_CUDA_ARRAY_INTERFACE_HPP
