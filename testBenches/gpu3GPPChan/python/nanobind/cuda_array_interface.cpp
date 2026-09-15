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

// CUDA array-interface support for the standalone nanobind module.
#include <vector>
#include <complex>
#include <limits>
#include <numeric>
#include <cuda_fp16.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "cuda_array_interface.hpp"

namespace nb = nanobind;

namespace gpu3gppchan_bindings {

// nb::tuple is immutable, so build a list and convert it to a tuple.
nb::tuple as_tuple(const std::vector<size_t>& vec) {
    nb::list result;
    for (size_t i = 0; i < vec.size(); ++i)
    {
        result.append(vec[i]);
    }
    return nb::steal<nb::tuple>(PyList_AsTuple(result.ptr()));
}

template <typename T>
cuda_array_t<T>::cuda_array_t(intptr_t addr,
                              const std::vector<size_t> &shape,
                              const std::vector<size_t> &strides,
                              bool isReadonly):
device_ptr(reinterpret_cast<void *>(addr)),
shape_(shape),
strides_(strides),
readonly(isReadonly),
has_strides(!strides.empty()) {
    if (shape_.empty())
    {
        throw std::runtime_error("Shape cannot be empty!");
    }
    if (get_size() != 0 && device_ptr == nullptr)
    {
        throw std::invalid_argument(
            "non-empty CUDA array must expose a non-null data pointer");
    }
}

template <typename T>
cuda_array_t<T>::cuda_array_t(const nb::object& array): owner_(array) {
    nb::dict interface = nb::cast<nb::dict>(array.attr("__cuda_array_interface__"));

    // shape (required)
    auto shape_tuple = nb::cast<nb::tuple>(interface["shape"]);
    shape_.clear();
    for (const auto &dim : shape_tuple)
    {
        shape_.push_back(nb::cast<size_t>(dim));
    }
    if (shape_.empty())
    {
        throw std::runtime_error("Shape cannot be empty!");
    }

    // data pointer (required)
    auto data_tuple = nb::cast<nb::tuple>(interface["data"]);
    void* raw_ptr{reinterpret_cast<void*>(nb::cast<uintptr_t>(data_tuple[0]))};
    if (get_size() != 0 && raw_ptr == nullptr)
    {
        throw std::invalid_argument(
            "non-empty CUDA array must expose a non-null data pointer");
    }
    // for zero-size arrays, pointer should be nullptr
    device_ptr = (get_size() == 0) ? nullptr : raw_ptr;
    readonly = nb::cast<bool>(data_tuple[1]);

    // strides (optional)
    has_strides = false;
    if (interface.contains("strides"))
    {
        nb::object strides_obj = nb::cast<nb::object>(interface["strides"]);
        if (!strides_obj.is_none())
        {
            auto strides_tuple = nb::cast<nb::tuple>(strides_obj);
            strides_.clear();
            for (const auto &stride : strides_tuple)
            {
                strides_.push_back(nb::cast<size_t>(stride));
            }
            has_strides = true;
        }
    }

    auto typestr = nb::cast<std::string>(interface["typestr"]);
    verify_dtype(typestr);
}

template <typename T>
size_t cuda_array_t<T>::get_size() const {
    size_t size{1};
    for (const size_t dimension : shape_)
    {
        if (dimension != 0 && size > std::numeric_limits<size_t>::max() / dimension)
        {
            throw std::overflow_error("CUDA array shape product overflows size_t");
        }
        size *= dimension;
    }
    return size;
}

template <typename T>
bool cuda_array_t<T>::is_c_contiguous() const {
    if (!has_strides)
    {
        return true;
    }
    if (strides_.size() != shape_.size())
    {
        return false;
    }

    size_t expected_stride{sizeof(T)};
    for (size_t dimension = shape_.size(); dimension-- > 0;)
    {
        if (shape_[dimension] > 1 && strides_[dimension] != expected_stride)
        {
            return false;
        }
        if (shape_[dimension] != 0 &&
            expected_stride > std::numeric_limits<size_t>::max() / shape_[dimension])
        {
            return false;
        }
        expected_stride *= shape_[dimension];
    }
    return true;
}

template <typename T>
bool cuda_array_t<T>::is_f_contiguous() const {
    if (!has_strides)
    {
        return shape_.size() <= 1;
    }
    if (strides_.size() != shape_.size())
    {
        return false;
    }

    size_t expected_stride{sizeof(T)};
    for (size_t dimension = 0; dimension < shape_.size(); ++dimension)
    {
        if (shape_[dimension] > 1 && strides_[dimension] != expected_stride)
        {
            return false;
        }
        if (shape_[dimension] != 0 &&
            expected_stride > std::numeric_limits<size_t>::max() / shape_[dimension])
        {
            return false;
        }
        expected_stride *= shape_[dimension];
    }
    return true;
}

template <typename T>
nb::dict cuda_array_t<T>::get_interface_dict() const {
    nb::dict interface;

    // required fields
    interface["shape"] = as_tuple(shape_);
    interface["typestr"] = get_typestr();
    interface["data"] = nb::make_tuple(reinterpret_cast<uintptr_t>(device_ptr), readonly);
    interface["version"] = 3;

    // optional fields
    if (has_strides) {
        interface["strides"] = as_tuple(strides_);
    }
    else {
        interface["strides"] = nb::none();
    }

    // we don't currently support mask or descr
    interface["mask"] = nb::none();
    interface["descr"] = nb::none();

    return interface;
}

template <typename T>
void cuda_array_t<T>::verify_dtype(const std::string &typestr) const
{
    if constexpr (std::is_same_v<T, float>)
    {
        if (typestr != "<f4")
        {
            throw std::runtime_error("Type mismatch: expected float32!");
        }
    }
    else if constexpr (std::is_same_v<T, int>)
    {
        if (typestr != "<i4")
        {
            throw std::runtime_error("Type mismatch: expected int!");
        }
    }
    else if constexpr (std::is_same_v<T, uint8_t>)
    {
        if (typestr != "|u1" && typestr != "<u1")
        {
            throw std::runtime_error("Type mismatch: expected uint8!");
        }
    }
    else if constexpr (std::is_same_v<T, uint16_t>)
    {
        if (typestr != "<u2")
        {
            throw std::runtime_error("Type mismatch: expected uint16!");
        }
    }
    else if constexpr (std::is_same_v<T, uint32_t>)
    {
        if (typestr != "<u4")
        {
            throw std::runtime_error("Type mismatch: expected uint32!");
        }
    }
    else if constexpr (std::is_same_v<T, __half>)
    {
        if (typestr != "<f2")
        {
            throw std::runtime_error("Type mismatch: expected float16!");
        }
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        if (typestr != "<f8")
        {
            throw std::runtime_error("Type mismatch: expected float64!");
        }
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>)
    {
        if (typestr != "<c8")
        {
            throw std::runtime_error("Type mismatch: expected complex64!");
        }
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>)
    {
        if (typestr != "<c16")
        {
            throw std::runtime_error("Type mismatch: expected complex128!");
        }
    }
}

template <typename T>
std::string cuda_array_t<T>::get_typestr() const
{
    if constexpr (std::is_same_v<T, float>)
    {
        return "<f4";
    }
    else if constexpr (std::is_same_v<T, int>)
    {
        return "<i4";
    }
    else if constexpr (std::is_same_v<T, uint8_t>)
    {
        return "<u1";
    }
    else if constexpr (std::is_same_v<T, uint16_t>)
    {
        return "<u2";
    }
    else if constexpr (std::is_same_v<T, uint32_t>)
    {
        return "<u4";
    }
    else if constexpr (std::is_same_v<T, __half>)
    {
        return "<f2";
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        return "<f8";
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>)
    {
        return "<c8";
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>)
    {
        return "<c16";
    }
    throw std::runtime_error("Unsupported dtype!");
}

template class cuda_array_t<int>;
template class cuda_array_t<uint8_t>;
template class cuda_array_t<uint16_t>;
template class cuda_array_t<uint32_t>;
template class cuda_array_t<float>;
template class cuda_array_t<__half>;
template class cuda_array_t<std::complex<float>>;

}  // namespace gpu3gppchan_bindings
