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

#ifndef AERIAL_FH_UTILS_HPP__
#define AERIAL_FH_UTILS_HPP__

#include "memfoot_global.h"
#include "defaults.hpp"
#include "output_formatting.hpp"
#include "nvlog.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <time.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include "aerial-fh-driver/fh_mutex.hpp"
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#ifndef likely
#define likely(x) __builtin_expect((x), 1)      //!< Branch prediction hint: likely true
#endif
#ifndef unlikely
#define unlikely(x) __builtin_expect((x), 0)    //!< Branch prediction hint: unlikely true
#endif

#define NEXT_VALUE(x, y) (((x) + (y) - 1) / (y) * (y))  //!< Round x up to next multiple of y
#define ROUND_UP(x, y) 1 + ((x - 1) / y);               //!< Round x up by dividing by y

namespace aerial_fh
{

typedef void* socket_handle;  //!< Opaque socket handle

/**
 * String builder for efficient string concatenation
 *
 * Used for building error messages and logging output.
 * Wraps std::stringstream with operator<< overloading.
 */
class StringBuilder {
public:
    template <class T>
    StringBuilder& operator<<(T const& x)
    {
        ss_ << x;
        return *this;
    }
    operator std::string()
    {
        return ss_.str();
    }

protected:
    std::stringstream ss_;
};

/**
 * Fronthaul driver exception class
 *
 * Extends std::runtime_error with additional context:
 * - Error code (Linux errno-style)
 * - Source file, function, and line number
 */
class FronthaulException : public std::runtime_error {
public:
    /**
     * Constructor
     * @param err_code Linux error code (e.g., EINVAL, EIO)
     * @param what Error description
     * @param file Source file where exception occurred
     * @param func Function name where exception occurred
     * @param lineno Line number where exception occurred
     */
    FronthaulException(int err_code, const std::string& what, const char* file, const char* func, int lineno) :
        std::runtime_error{what},
        err_code_{err_code},
        file_{file},
        func_{func},
        lineno_{lineno} {}
    int         err_code() const { return err_code_; }  //!< Get error code
    const char* file() const { return file_; }          //!< Get source file
    const char* func() const { return func_; }          //!< Get function name
    int         lineno() const { return lineno_; }      //!< Get line number

protected:
    int         err_code_;  //!< Linux error code
    const char* file_;      //!< Source file name
    const char* func_;      //!< Function name
    int         lineno_;    //!< Line number
};

/**
 * Thread-safe round-robin iterator
 *
 * Iterates through a vector of items in round-robin fashion.
 * Thread-safe using mutex protection for the index.
 *
 * \tparam T Type of items to iterate over (typically pointers)
 */
template <class T>
class Iterator {
public:
    /**
     * Get next item in round-robin fashion
     * @return Next item (thread-safe)
     */
    T next()
    {
        size_t                            current_index;
        const std::lock_guard<aerial_fh::FHMutex> lock(mtx_);
        current_index = index_;
        index_        = (index_ + 1) % items_.size();
        return items_[current_index];
    }

    /**
     * Add item to iterator
     * @param element Item to add
     * @return New size of items vector
     */
    size_t add(T element)
    {
        items_.push_back(element);
        return items_.size();
    }

    /**
     * Get const reference to items vector
     * @return Items vector
     */
    std::vector<T> const& items() const
    {
        return items_;
    }

    /**
     * Clear all items
     */
    void clear()
    {
        items_.clear();
    }

protected:
    std::vector<T> items_;        //!< Vector of items
    aerial_fh::FHMutex     mtx_;  //!< Mutex for thread-safe access
    size_t         index_{0};     //!< Current round-robin index
};

#define FILE_BNAME (__builtin_strrchr(__FILE__, '/') ? __builtin_strrchr(__FILE__, '/') + 1 : __FILE__)  //!< Extract filename from path

// FH exception throw macro
#pragma GCC diagnostic push
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wterminate"
#endif
/**
 * Internal function for throwing fronthaul exceptions
 */
inline void throw_fh_func(int err_code, const std::string& what, const char* filename, const char* funcname, int line)
{
    throw FronthaulException(err_code, what, filename, funcname, line);
}
#define THROW_FH(err_code, what) throw_fh_func(err_code, what, FILE_BNAME, __FUNCTION__, __LINE__);  //!< Throw fronthaul exception with context
#pragma GCC diagnostic pop

#define STR_(x) #x          //!< Stringize helper macro
#define STR(x) STR_(x)      //!< Stringize macro (converts x to string literal)

/**
 * Check CUDA call and throw exception on error
 *
 * Wraps CUDA API calls and converts errors to FronthaulException.
 * Includes CUDA error code, name, and description in exception message.
 */
#define CHECK_CUDA_THROW(expr)                                                                                                                                                        \
    do                                                                                                                                                                                \
    {                                                                                                                                                                                 \
        cudaError_t err = (expr);                                                                                                                                                     \
        if(err != cudaSuccess)                                                                                                                                                        \
        {                                                                                                                                                                             \
            THROW_FH(EIO, StringBuilder() << "CUDA call failed with " << err << "(" << cudaGetErrorName(err) << "):" << cudaGetErrorString(err) << ". Failed CUDA call: " STR(expr)); \
        }                                                                                                                                                                             \
    } while(0)

typedef std::unique_ptr<void, decltype(&free)> UniquePtr;  //!< Unique pointer with free() deleter

/**
 * Calculate PRB size in bytes
 *
 * Computes the size of a Physical Resource Block (PRB) based on:
 * - IQ sample bit width
 * - User data compression method
 *
 * @param iq_sample_size IQ sample size in bits (1-16)
 * @param compression_method Compression method (affects overhead)
 * @return PRB size in bytes
 * @throws FronthaulException if parameters are invalid
 */
static size_t get_prb_size(size_t iq_sample_size, UserDataCompressionMethod compression_method)
{
    auto                  method             = static_cast<size_t>(compression_method);
    std::array<size_t, 7> ud_comp_param_size = {0, 1, 1, 1, 0, 2, 2};  // Compression param sizes per method

    if(unlikely((iq_sample_size > UD_IQ_WIDH_MAX) || (iq_sample_size == 0)))
        THROW_FH(EINVAL, StringBuilder() << "Invalid user data IQ sample size: " << iq_sample_size);

    if(unlikely(method >= ud_comp_param_size.size()))
        THROW_FH(EINVAL, "Invalid user data IQ compression method");

    // BFP with max bit width has no compression overhead
    auto comp_param_sz = (compression_method == UserDataCompressionMethod::BLOCK_FLOATING_POINT && iq_sample_size == BFP_NO_COMPRESSION) ? 0 : ud_comp_param_size[method];
    return PRB_SIZE(iq_sample_size) + comp_param_sz;
}

/**
 * Get C-plane message common header size
 *
 * @param section_type ORAN C-plane section type (0, 1, 3, or 5)
 * @return Common header size in bytes
 */
inline uint16_t get_cmsg_common_hdr_size(uint8_t section_type)
{
    std::array<uint16_t, ORAN_CMSG_SECTION_TYPE_5 + 1> common_hdr_sizes = {
        sizeof(oran_cmsg_sect0_common_hdr),
        sizeof(oran_cmsg_sect1_common_hdr),
        0,
        sizeof(oran_cmsg_sect3_common_hdr),
        0,
        sizeof(oran_cmsg_sect5_common_hdr),
    };

    return common_hdr_sizes[section_type];
}

/**
 * Get C-plane message section size
 *
 * @param section_type ORAN C-plane section type (0, 1, 3, or 5)
 * @return Section size in bytes
 */
inline uint16_t get_cmsg_section_size(uint8_t section_type)
{
    std::array<size_t, ORAN_CMSG_SECTION_TYPE_5 + 1> section_sizes = {
        sizeof(oran_cmsg_sect0),
        sizeof(oran_cmsg_sect1),
        0,
        sizeof(oran_cmsg_sect3),
        0,
        sizeof(oran_cmsg_sect5),
    };

    return section_sizes[section_type];
}

} // namespace aerial_fh

#endif //ifndef AERIAL_FH_UTILS_HPP__
