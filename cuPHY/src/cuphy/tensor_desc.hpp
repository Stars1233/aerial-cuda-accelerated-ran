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

#if !defined(TENSOR_DESC_HPP_INCLUDED_)
#define TENSOR_DESC_HPP_INCLUDED_

#include <array>
#include <experimental/optional>
#include <functional>
#include <numeric>
#include <algorithm>
#include <type_traits>
#include "cuComplex.h"
#include "cuda_fp16.h"
#include "cuphy.h"
#include "cuphy_internal.h"

template<typename T>
[[nodiscard]] CUDA_BOTH_INLINE bool device_equal(const T* a, const T* a_end, const T* b) {
    while (a != a_end) {
        if (*a++ != *b++) return false;
    }
    return true;
}


// Empty struct - used as base class for internal tensor descriptor
struct cuphyTensorDescriptor
{};

////////////////////////////////////////////////////////////////////////////
// get_cuphy_type_storage_element_size()
// Returns the size of the storage element for a given type. In general,
// this is the size of the type used to store the given cuphyDataType_t.
// However, for sub-byte types, multiple elements are stored in a machine
// word. For these types, the returned size is the size of a machine word,
// which stores multiple elements.
// Returns 0 for CUPHY_VOID.
int get_cuphy_type_storage_element_size(cuphyDataType_t type);

////////////////////////////////////////////////////////////////////////////
// type_is_sub_byte()
inline bool type_is_sub_byte(cuphyDataType_t type)
{
    return (CUPHY_BIT == type);
}

////////////////////////////////////////////////////////////////////////////
// type_is_complex()
inline bool type_is_complex(cuphyDataType_t type)
{
  switch(type)
  {
  case CUPHY_VOID:
  case CUPHY_BIT:
  case CUPHY_R_8I:
  case CUPHY_R_8U:
  case CUPHY_R_16I:
  case CUPHY_R_16U:
  case CUPHY_R_32I:
  case CUPHY_R_32U:
  case CUPHY_R_16F:
  case CUPHY_R_32F:
  case CUPHY_R_64F:
  default:
      return false;
  case CUPHY_C_8I:
  case CUPHY_C_8U:
  case CUPHY_C_16I:
  case CUPHY_C_16U:
  case CUPHY_C_32I:
  case CUPHY_C_32U:
  case CUPHY_C_16F:
  case CUPHY_C_32F:
  case CUPHY_C_64F:
      return true;
  }
}

////////////////////////////////////////////////////////////////////////////
// scalar_type_from_complex_type()
inline cuphyDataType_t scalar_type_from_complex_type(cuphyDataType_t type)
{
  switch(type)
  {
  case CUPHY_VOID:
  case CUPHY_BIT:
  case CUPHY_R_8I:
  case CUPHY_R_8U:
  case CUPHY_R_16I:
  case CUPHY_R_16U:
  case CUPHY_R_32I:
  case CUPHY_R_32U:
  case CUPHY_R_16F:
  case CUPHY_R_32F:
  case CUPHY_R_64F:
  default:
      return CUPHY_VOID;
  case CUPHY_C_8I:
      return CUPHY_R_8I;
  case CUPHY_C_8U:
      return CUPHY_R_8U;
  case CUPHY_C_16I:
      return CUPHY_R_16I;
  case CUPHY_C_16U:
      return CUPHY_R_16U;
  case CUPHY_C_32I:
      return CUPHY_R_32I;
  case CUPHY_C_32U:
      return CUPHY_R_32U;
  case CUPHY_C_16F:
      return CUPHY_R_16F;
  case CUPHY_C_32F:
      return CUPHY_R_32F;
  case CUPHY_C_64F:
      return CUPHY_R_64F;
  }
}

////////////////////////////////////////////////////////////////////////////
// get_element_multiple_for_alignment()
// Returns the number of elements required to populate the given byte
// alignment. This can be used to round up tensor dimensions so that
// higher dimensions begin at a given alignment.
// Example: For byteAlign = 128 and type = CUPHY_R_32F (which has a size
// of 4 bytes), this function will return 128 / 4 = 32.
// Example: For byteAlign = 4 and type = CUPHY_BIT, this function will
// return 4 / (1/8) = 32.
int get_element_multiple_for_alignment(int byte_align, cuphyDataType_t type);

////////////////////////////////////////////////////////////////////////////
// data_type_traits
// clang-format off
template <cuphyDataType_t T> struct data_type_traits;
template <>                  struct data_type_traits<CUPHY_VOID>      { typedef void type;            };
template <>                  struct data_type_traits<CUPHY_BIT>       { typedef uint32_t type;        };
template <>                  struct data_type_traits<CUPHY_R_8I>      { typedef signed char type;     };
template <>                  struct data_type_traits<CUPHY_C_8I>      { typedef char2 type;           };
template <>                  struct data_type_traits<CUPHY_R_8U>      { typedef unsigned char type;   };
template <>                  struct data_type_traits<CUPHY_C_8U>      { typedef uchar2 type;          };
template <>                  struct data_type_traits<CUPHY_R_16I>     { typedef short type;           };
template <>                  struct data_type_traits<CUPHY_C_16I>     { typedef short2 type;          };
template <>                  struct data_type_traits<CUPHY_R_16U>     { typedef unsigned short type;  };
template <>                  struct data_type_traits<CUPHY_C_16U>     { typedef ushort2 type;         };
template <>                  struct data_type_traits<CUPHY_R_32I>     { typedef int type;             };
template <>                  struct data_type_traits<CUPHY_C_32I>     { typedef int2 type;            };
template <>                  struct data_type_traits<CUPHY_R_32U>     { typedef unsigned int type;    };
template <>                  struct data_type_traits<CUPHY_C_32U>     { typedef uint2 type;           };
template <>                  struct data_type_traits<CUPHY_R_16F>     { typedef __half type;          };
template <>                  struct data_type_traits<CUPHY_C_16F>     { typedef __half2 type;         };
template <>                  struct data_type_traits<CUPHY_R_32F>     { typedef float type;           };
template <>                  struct data_type_traits<CUPHY_C_32F>     { typedef cuComplex type;       };
template <>                  struct data_type_traits<CUPHY_R_64F>     { typedef double type;          };
template <>                  struct data_type_traits<CUPHY_C_64F>     { typedef cuDoubleComplex type; };
template <>                  struct data_type_traits<CUPHY_R_8F_E4M3> { typedef __nv_fp8_e4m3 type;   };
template <>                  struct data_type_traits<CUPHY_R_8F_E5M2> { typedef __nv_fp8_e5m2 type;   };

// clang-format on

////////////////////////////////////////////////////////////////////////////
// type_to_cuphy_type
// clang-format off
template <typename T> struct type_to_cuphy_type;
template <>           struct type_to_cuphy_type<void>            { static constexpr cuphyDataType_t value = CUPHY_VOID;      };
//template <>           struct type_to_cuphy_type<CUPHY_BIT>     { typedef uint32_t type;        };
template <>           struct type_to_cuphy_type<signed char>     { static constexpr cuphyDataType_t value = CUPHY_R_8I;      };
template <>           struct type_to_cuphy_type<char2>           { static constexpr cuphyDataType_t value = CUPHY_C_8I;      };
template <>           struct type_to_cuphy_type<unsigned char>   { static constexpr cuphyDataType_t value = CUPHY_R_8U;      };
template <>           struct type_to_cuphy_type<uchar2>          { static constexpr cuphyDataType_t value = CUPHY_C_8U;      };
template <>           struct type_to_cuphy_type<short>           { static constexpr cuphyDataType_t value = CUPHY_R_16I;     };
template <>           struct type_to_cuphy_type<short2>          { static constexpr cuphyDataType_t value = CUPHY_C_16I;     };
template <>           struct type_to_cuphy_type<unsigned short>  { static constexpr cuphyDataType_t value = CUPHY_R_16U;     };
template <>           struct type_to_cuphy_type<ushort2>         { static constexpr cuphyDataType_t value = CUPHY_C_16U;     };
template <>           struct type_to_cuphy_type<int>             { static constexpr cuphyDataType_t value = CUPHY_R_32I;     };
template <>           struct type_to_cuphy_type<int2>            { static constexpr cuphyDataType_t value = CUPHY_C_32I;     };
template <>           struct type_to_cuphy_type<unsigned int>    { static constexpr cuphyDataType_t value = CUPHY_R_32U;     };
template <>           struct type_to_cuphy_type<uint2>           { static constexpr cuphyDataType_t value = CUPHY_C_32U;     };
template <>           struct type_to_cuphy_type<__half>          { static constexpr cuphyDataType_t value = CUPHY_R_16F;     };
template <>           struct type_to_cuphy_type<__half2>         { static constexpr cuphyDataType_t value = CUPHY_C_16F;     };
template <>           struct type_to_cuphy_type<float>           { static constexpr cuphyDataType_t value = CUPHY_R_32F;     };
template <>           struct type_to_cuphy_type<cuComplex>       { static constexpr cuphyDataType_t value = CUPHY_C_32F;     };
template <>           struct type_to_cuphy_type<double>          { static constexpr cuphyDataType_t value = CUPHY_R_64F;     };
template <>           struct type_to_cuphy_type<cuDoubleComplex> { static constexpr cuphyDataType_t value = CUPHY_C_64F;     };
template <>           struct type_to_cuphy_type<__nv_fp8_e4m3>   { static constexpr cuphyDataType_t value = CUPHY_R_8F_E4M3; };
template <>           struct type_to_cuphy_type<__nv_fp8_e5m2>   { static constexpr cuphyDataType_t value = CUPHY_R_8F_E5M2; };
// clang-format on

////////////////////////////////////////////////////////////////////////
// complex_from_scalar
// clang-format off
template <typename T> struct complex_from_scalar;
template <>           struct complex_from_scalar<signed char>    { typedef char2 type;           };
template <>           struct complex_from_scalar<unsigned char>  { typedef uchar2 type;          };
template <>           struct complex_from_scalar<short>          { typedef short2 type;          };
template <>           struct complex_from_scalar<unsigned short> { typedef ushort2 type;         };
template <>           struct complex_from_scalar<int>            { typedef int2 type;            };
template <>           struct complex_from_scalar<unsigned int>   { typedef uint2 type;           };
template <>           struct complex_from_scalar<__half>         { typedef __half2 type;         };
template <>           struct complex_from_scalar<float>          { typedef cuComplex type;       };
template <>           struct complex_from_scalar<double>         { typedef cuDoubleComplex type; };
// clang-format on

////////////////////////////////////////////////////////////////////////
// scalar_from_complex
// clang-format off
template <typename T> struct scalar_from_complex;
template <>           struct scalar_from_complex<char2>           { typedef signed char type;    };
template <>           struct scalar_from_complex<uchar2>          { typedef unsigned char type;  };
template <>           struct scalar_from_complex<short2>          { typedef short type;          };
template <>           struct scalar_from_complex<ushort2>         { typedef unsigned short type; };
template <>           struct scalar_from_complex<int2>            { typedef int type;            };
template <>           struct scalar_from_complex<uint2>           { typedef unsigned int type;   };
template <>           struct scalar_from_complex<__half2>         { typedef __half type;         };
template <>           struct scalar_from_complex<cuComplex>       { typedef float type;          };
template <>           struct scalar_from_complex<cuDoubleComplex> { typedef double type;         };
// clang-format on

////////////////////////////////////////////////////////////////////////
// make_complex
// clang-format off
template <typename Tcomplex> struct make_complex;
template <>                  struct make_complex<char2>
{
    typedef typename scalar_from_complex<char2>::type Tscalar;
    static CUDA_BOTH char2   create(Tscalar r, Tscalar i)  { return make_char2(r, i); }
};
template <>                  struct make_complex<uchar2>
{
    typedef typename scalar_from_complex<uchar2>::type Tscalar;
    static CUDA_BOTH uchar2  create(Tscalar r, Tscalar i) {return make_uchar2(r, i); }
};
template <>                  struct make_complex<short2>
{
    typedef typename scalar_from_complex<short2>::type Tscalar;
    static CUDA_BOTH short2  create(Tscalar r, Tscalar i) { return make_short2(r, i); }
};
template <>                  struct make_complex<ushort2>
{
    typedef typename scalar_from_complex<ushort2>::type Tscalar;
    static CUDA_BOTH ushort2 create(Tscalar r, Tscalar i) { return make_ushort2(r, i); }
};
template <>                  struct make_complex<int2>
{
    typedef typename scalar_from_complex<int2>::type Tscalar;
    static CUDA_BOTH int2   create(Tscalar r, Tscalar i) { return make_int2(r, i); }
};
template <>                 struct make_complex<uint2>
{
    typedef typename scalar_from_complex<uint2>::type Tscalar;
    static CUDA_BOTH uint2  create(Tscalar r, Tscalar i) { return make_uint2(r, i); }
};
#if defined(__CUDACC__)
template <>                 struct make_complex<struct __half2>
{
    typedef typename scalar_from_complex<struct __half2>::type Tscalar;
    static CUDA_INLINE struct __half2 create(Tscalar r, Tscalar i) { return __halves2half2(r, i); }
};
#endif
template <>                 struct make_complex<cuComplex>
{
    typedef typename scalar_from_complex<cuComplex>::type Tscalar;
    static CUDA_BOTH cuComplex create(Tscalar r, Tscalar i) { return make_cuFloatComplex(r, i); }
};
template <>                 struct make_complex<cuDoubleComplex>
{
    typedef typename scalar_from_complex<cuDoubleComplex>::type Tscalar;
    static CUDA_BOTH cuDoubleComplex create(Tscalar r, Tscalar i) { return make_cuDoubleComplex(r, i); }
};
// clang-format on

namespace cuphy
{
////////////////////////////////////////////////////////////////////////
// vec
// Array of elements with a size that is fixed at compile time. Similar
// to std::array, but this class differs in that it has function
// decorations for operation on a GPU device as well as the host.
// Defined as an aggregate type so that initialization can occur using
// a braced-init-list.
// https://en.cppreference.com/w/cpp/language/aggregate_initialization
template <typename T, int Dim>
class vec final
{
public:
    vec() = default;
    template <std::size_t N>
    CUDA_BOTH_INLINE
    explicit vec(const T (&arr)[N])
    {
        static_assert(N == Dim, "Initializer list does not have enough dimensions");
        for(int i = 0; i < N; ++i) { elem_[i] = arr[i]; }
    }
    CUDA_BOTH_INLINE
    void fill(T val)
    {
        for(int i = 0; i < Dim; ++i) { elem_[i] = val; }
    }
    [[nodiscard]] CUDA_BOTH_INLINE
    T& operator[](int idx) { return elem_[idx]; }
    [[nodiscard]] CUDA_BOTH_INLINE
    const T& operator[](int idx) const { return elem_[idx]; }
    [[nodiscard]] CUDA_BOTH_INLINE
    T*       begin() { return elem_; }
    [[nodiscard]]
    T*       end() { return elem_ + Dim; }
    [[nodiscard]]
    const T* begin() const { return elem_; }
    [[nodiscard]]
    const T* end() const { return elem_ + Dim; }

    [[nodiscard]] CUDA_BOTH_INLINE friend bool operator==(const vec& a, const vec& b)
    {
        return device_equal(a.elem_, a.elem_ + Dim, b.elem_);
    }

    [[nodiscard]] CUDA_BOTH_INLINE friend bool operator!=(const vec& a, const vec& b)
    {
        return !(a == b);
    }

private:
    T elem_[Dim]{};
};
} //namespace cuphy

template <typename TResult, typename TVector, int Dim>
TResult dot(const cuphy::vec<TVector, Dim>& a, const cuphy::vec<TVector, Dim>& b)
{
    TResult sum = static_cast<TResult>(0);
#pragma unroll
    for(size_t i = 0; i < Dim; ++i)
    {
        sum += (a[i] * b[i]);
    }
    return sum;
}

////////////////////////////////////////////////////////////////////////
// tensor_layout_N
// Data type-independent description of tensor layout for a fixed (at
// compile time) number of dimensions. Supports both host and device
// usage.
//
// Tensor indexing:
// A tensor contains N dimensions. The size of each dimension is
// given by an integer d(i), and the stride for each dimension is
// given by s(i). We refer to a specific element of the tensor
// using a set of integers n(i), where n(i) < d(i) for each
// i = 0..N.
// Here we are assuming that all sizes d(i) > 0, strides s(i) > 0,
// and s(i) >= d(i) for i = 0..(N-1).
// For a give set of N indices, the offset from the beginning of the
// data (in elements) is given by the dot product:
//
// offset(n) = sum(i=0..(N-1)){ n(i) * s(i) }
//
// The maximum offset can be obtained using the maximum index in each
// dimension n(i)_max = d(i) - 1.
//
// offset_max = sum(i=0..(N-1)){ (d(i) - 1) * s(i) }
template <int TRank>
struct tensor_layout_N
{
    cuphy::vec<int, TRank> dimensions;
    cuphy::vec<int, TRank> strides;
    CUDA_BOTH_INLINE
    int rank() const { return TRank; }
    //------------------------------------------------------------------
    // tensor_layout_N()
    CUDA_BOTH_INLINE
    tensor_layout_N() noexcept
    {
        dimensions.fill(0);
        strides.fill(0);
    }
    //------------------------------------------------------------------
    // tensor_layout_N()
    CUDA_BOTH_INLINE
    tensor_layout_N(const int* dim, const int* str) noexcept
    {
        for(int i = 0; i < TRank; ++i)
        {
            dimensions[i] = dim[i];
            strides[i]    = str[i];
        }
    }
    //------------------------------------------------------------------
    // offset()
    CUDA_BOTH_INLINE
    int offset(const cuphy::vec<int, TRank>& indices) const
    {
        return dot<int, int, TRank>(indices, strides);
    }
    //------------------------------------------------------------------
    CUDA_BOTH_INLINE
    int offset(const int (&n)[TRank]) const
    {
        int idx = 0;
#pragma unroll
        for(int i = 0; i < TRank; ++i)
        {
            idx += n[i] * strides[i];
        }
        return idx;
    }
    //------------------------------------------------------------------
    // get_num_elements()
    CUDA_BOTH_INLINE
    int get_num_elements() const noexcept
    {
        // Return the product of the dimensions d(0) * d(1) * ... * d(N-1)
        return dimensions.product();
    }
    //------------------------------------------------------------------
    // get_num_elements_with_strides()
    CUDA_BOTH_INLINE
    size_t get_num_elements_with_strides() const noexcept
    {
        size_t max_size = 0;
        for(int i = 0; i < TRank; ++i)
        {
            max_size = std::max(max_size, static_cast<size_t>(dimensions[i] * strides[i]));
        }
        return max_size;
    }
    //------------------------------------------------------------------
    // swap_dimensions()
    CUDA_BOTH_INLINE
    void swap_dimensions(int a, int b)
    {
        int tmpD = dimensions[a]; dimensions[a] = dimensions[b]; dimensions[b] = tmpD;
        int tmpS = strides[a];    strides[a]    = strides[b];    strides[b]    = tmpS;
    }
};

template <int Dim>
CUDA_BOTH_INLINE bool operator==(const tensor_layout_N<Dim>& a, const tensor_layout_N<Dim>& b)
{
    return (a.dimensions == b.dimensions) && (a.strides == b.strides);
}

template <typename T, int Dim>
CUDA_BOTH_INLINE bool operator!=(const tensor_layout_N<Dim>& a, const tensor_layout_N<Dim>& b)
{
    return !(a == b);
}

////////////////////////////////////////////////////////////////////////
// tensor_layout_any
// Data type-independent description of tensor layout. Supports any
// number of dimensions, up to the maximum supported by the library.
// Useful for representing "generic" layouts for kernels and functions
// that handle any number of (legal) dimensions, as well as layouts
// before optional conversion to dimension-specific handling.
struct tensor_layout_any : tensor_layout_N<CUPHY_DIM_MAX>
{
    //------------------------------------------------------------------
    // tensor_layout_any()
    CUDA_BOTH_INLINE
    tensor_layout_any() noexcept :
        rank_(0) {}
    //------------------------------------------------------------------
    // tensor_layout_any()
    CUDA_BOTH_INLINE
    tensor_layout_any(int nrank, const int* dim, const int* str) noexcept :
        rank_(nrank)
    {
        // Explicitly initialize base class dimensions and strides, using
        // sensible values for unused dimensions (1) and strides (0)
        for(int i = 0; i < CUPHY_DIM_MAX; ++i)
        {
            dimensions[i] = (i < nrank) ? dim[i] : 1;
        }
        if(str)
        {
            for(int i = 0; i < CUPHY_DIM_MAX; ++i)
            {
                // Set unused strides to 0
                strides[i] = (i < nrank) ? str[i] : 0;
            }
        }
        else
        {
            // Default to tightly packed when no strides are provided
            strides[0] = 1;
            for(int i = 1; i < CUPHY_DIM_MAX; ++i)
            {
                strides[i] = (i < nrank) ? (dim[i - 1] * strides[i - 1]) : 0;
            }
        }
    }
    //------------------------------------------------------------------
    // validate()
    static bool validate(int             rank,
                         const int*      dim,
                         const int*      str,
                         cuphyDataType_t type) noexcept;
    //------------------------------------------------------------------
    // has_same_size()
    bool has_same_size(const tensor_layout_any& layout) const noexcept
    {
        return (rank() == layout.rank()) &&
               (dimensions == layout.dimensions);
    }
    //------------------------------------------------------------------
    // has_same_strides()
    bool has_same_strides(const tensor_layout_any& layout) const noexcept
    {
        return (rank() == layout.rank()) &&
               (strides == layout.strides);
    }
    int rank() const { return rank_; }

private:
    int rank_; // Explicit count of number of "valid" dimensions
};

////////////////////////////////////////////////////////////////////////
// tensor_layout_contiguous_N
// Data type-independent description of tensor layout for a fixed (at
// compile time) number of dimensions, with a first dimension stride of
// 1.
// Some API functions might only support input tensors with a
// contiguous layout, or we might have specialized kernels for the
// common contiguous case.
//
// This class supports both host and device usage.
//
// Note specialization for size 1, to avoid complications for a zero
// length stride vector.
template <int TRank>
struct tensor_layout_contiguous_N
{
    cuphy::vec<int, TRank>     dimensions;
    cuphy::vec<int, TRank - 1> strides;
    int                 rank() const { return TRank; }
    //------------------------------------------------------------------
    // tensor_layout_contiguous_N()
    CUDA_BOTH_INLINE
    tensor_layout_contiguous_N() noexcept
    {
        dimensions.fill(0);
        strides.fill(0);
    }
    //------------------------------------------------------------------
    // tensor_layout_contiguous_N()
    CUDA_BOTH_INLINE
    tensor_layout_contiguous_N(const int* dim, const int* str) noexcept
    {
        // clang-format off
        for(int i = 0; i < TRank;     ++i)   { dimensions[i] = dim[i]; }
        for(int i = 0; i < (TRank-1); ++i)   { strides[i]    = str[i]; }
        // clang-format on
    }
    //------------------------------------------------------------------
    // tensor_layout_contiguous_N()
    CUDA_BOTH_INLINE
    explicit tensor_layout_contiguous_N(const tensor_layout_any& layout_any) noexcept
    {
        // clang-format off
        for(int i = 0; i < TRank;     ++i)   { dimensions[i] = layout_any.dimensions[i];  }
        for(int i = 0; i < (TRank-1); ++i)   { strides[i]    = layout_any.strides[i + 1]; } // skip first stride (contiguous)
        // clang-format on
    }
    //------------------------------------------------------------------
    // offset()
    CUDA_BOTH_INLINE
    int offset(const cuphy::vec<int, TRank>& indices) const
    {
        // Add offset for first index, which has stride 1
        int idx = indices[0];
#pragma unroll
        for(int i = 1; i < TRank; ++i)
        {
            idx += (indices[i] * strides[i - 1]);
        }
        return idx;
    }
    //------------------------------------------------------------------
    CUDA_BOTH_INLINE
    int offset(const int (&n)[TRank]) const
    {
        int idx = n[0];
#pragma unroll
        for(int i = 1; i < TRank; ++i)
        {
            idx += (n[i] * strides[i - 1]);
        }
        return idx;
    }
    //------------------------------------------------------------------
    // get_num_elements()
    CUDA_BOTH_INLINE
    int get_num_elements() const noexcept
    {
        // Return the product of the dimensions d(0) * d(1) * ... * d(N-1)
        return dimensions.product();
    }
    //------------------------------------------------------------------
    // get_num_elements_with_strides()
    CUDA_BOTH_INLINE
    int get_num_elements_with_strides() const noexcept
    {
        // Account for first index, which has stride 1
        int max_size = dimensions[0];
        for(int i = 1; i < TRank; ++i)
        {
            max_size = std::max(max_size, static_cast<int>(dimensions[i] * strides[i - 1]));
        }
        return max_size;
    }
};

template <>
struct tensor_layout_contiguous_N<1>
{
    cuphy::vec<int, 1> dimensions;
    int         rank() const { return 1; }
    //------------------------------------------------------------------
    // tensor_layout_contiguous_N()
    CUDA_BOTH_INLINE
    tensor_layout_contiguous_N() noexcept
    {
        dimensions[0] = 0;
    }
    //------------------------------------------------------------------
    // tensor_layout_contiguous_N()
    CUDA_BOTH_INLINE
    tensor_layout_contiguous_N(const int* dim, const int* /*str*/) noexcept
    {
        dimensions[0] = dim[0];
    }
    //------------------------------------------------------------------
    // tensor_layout_contiguous_N()
    CUDA_BOTH_INLINE
    explicit tensor_layout_contiguous_N(const tensor_layout_any& layout_any) noexcept
    {
        dimensions[0] = layout_any.dimensions[0];
    }
    //------------------------------------------------------------------
    // offset()
    CUDA_BOTH_INLINE
    int offset(const cuphy::vec<int, 1>& indices) const
    {
        return indices[0];
    }
    //------------------------------------------------------------------
    CUDA_BOTH_INLINE
    int offset(const int (&n)[1]) const
    {
        return n[0];
    }
    //------------------------------------------------------------------
    // get_num_elements()
    CUDA_BOTH_INLINE
    int get_num_elements() const noexcept
    {
        // Return the product of the dimensions d(0) * d(1) * ... * d(N-1)
        return dimensions[0];
    }
    //------------------------------------------------------------------
    // get_num_elements_with_strides()
    CUDA_BOTH_INLINE
    int get_num_elements_with_strides() const noexcept
    {
        // Account for first index, which has stride 1
        return dimensions[0];
    }
};

namespace cuphy
{
// clang-format off
// //////////////////////////////////////////////////////////////////////
// cuphy::tensor_layout_t
/**
 * @brief Represents the layout of a tensor in memory.
 *
 * This class is a template class that allows for tensors of any rank (up to the maximum supported by the cuPHY library).
 * The layout is defined by the dimensions and strides of the tensor.
 *
 * @tparam DIM The maximum number of dimensions in the tensor.
 */
template <std::size_t DIM = CUPHY_DIM_MAX>
class tensor_layout_t
{
public:
    tensor_layout_t()
    {
        dimensions_.fill(0);
        strides_.fill(0);
    }
    tensor_layout_t(const tensor_layout_t& t) :
      rank_(t.rank_),
      dimensions_(t.dimensions_),
      strides_(t.strides_)
    {
        // GCC Optimizer bug without this explicit definition?
    }

    tensor_layout_t(tensor_layout_t&&) noexcept = default;
    tensor_layout_t& operator=(tensor_layout_t&&) noexcept = default;
    tensor_layout_t& operator=(const tensor_layout_t&) noexcept = default;

    tensor_layout_t(int nrank, const int* dims, const int* str) : rank_(nrank)
    {
        for(size_t i = 0; i < DIM; ++i) { dimensions_[i] = (i < nrank) ? dims[i] : 1; }
        if(str)
        {
            // Set unused strides to zero
            for(size_t i = 0; i < DIM; ++i) { strides_[i] = (i < nrank) ? str[i] : 0; }
        }
        else
        {
            strides_[0] = 1;
            for(size_t i = 1; i < DIM; ++i) { strides_[i] = (i < nrank) ? (strides_[i - 1] * dimensions_[i -1]) : 0; }
        }

        // Initialize remaining strides to zero
        for(size_t i = nrank; i < DIM; ++i) { strides_[i] = 0; }
    }
    template <int N>
    tensor_layout_t(const int(&dims)[N], const int(&strides)[N]) : rank_(N)
    {
        static_assert(N <= DIM, "Layout initialization must have <= template's DIM dimensions");
        for(size_t i = 0;     i < DIM; ++i) { dimensions_[i] = (i < N) ? dims[i]         : 1; }
        for(size_t i = 0;     i < DIM; ++i) { strides_[i]    = (i < N) ? strides[i]      : 0; }
    }
    template <int N>
    tensor_layout_t(const vec<int, N>& dims, const vec<int, N>& strides) : rank_(N)
    {
        static_assert(N <= DIM, "Layout initialization must have <= template's DIM dimensions");
        for(size_t i = 0;     i < DIM; ++i) { dimensions_[i] = (i < N) ? dims[i]         : 1; }
        for(size_t i = 0;     i < DIM; ++i) { strides_[i]    = (i < N) ? strides[i]      : 0; }
    }

    template <size_t N>
    [[nodiscard]]
    int offset(const int (&indices)[N]) const {
        static_assert(N <= DIM, "Layout initialization must have <= template's DIM dimensions");
        return offset_impl(indices, N);
    }

    template <size_t N>
    [[nodiscard]]
    int offset(const std::array<int, N>& indices) const {
        static_assert(N <= DIM, "Layout initialization must have <= template's DIM dimensions");
        return offset_impl(indices.data(), N);
    }
    template <int N>
    void check_bounds(const int (&indices)[N]) const
    {
        for(size_t i = 0; i < N; ++i)
        {
            if(indices[i] >= dimensions_[i])
            {
                throw std::runtime_error("Index exceeds tensor dimension");
            }
        }

    }

    template <typename... Indices>
    [[nodiscard]]
    size_t get_offset(Indices... indices) const {
        static_assert(sizeof...(Indices) <= DIM, "Too many indices for tensor layout");
        const int idx[] = {indices...};
        std::size_t offset = 0;
        for (std::size_t i = 0; i < sizeof...(Indices); ++i) {
            offset += idx[i] * strides_[i];
        }
        return offset;
    }

    [[nodiscard]] int rank() const { return rank_; }
    vec<int, DIM>& dimensions()             { return dimensions_; }
    vec<int, DIM>& strides()                { return strides_;    }
    [[nodiscard]] const vec<int, DIM>& dimensions() const { return dimensions_; }
    [[nodiscard]] const vec<int, DIM>& strides()    const { return strides_;    }
private:
    // Common implementation for offset calculation
    template<typename T>
    [[nodiscard]]
    int offset_impl(const T* indices, const int ndims) const {
        int offset = 0;
        for (int i = 0; i < ndims; i++) {
            offset += indices[i] * strides_[i];
        }
        return offset;
    }

    int rank_{};
    vec<int, DIM> dimensions_; // (Only rank elements are valid)
    vec<int, DIM> strides_;    // stride in elements (Only rank elements are valid)
};

using tensor_layout = tensor_layout_t<>;

// clang-format on
} // namespace cuphy

////////////////////////////////////////////////////////////////////////
// Function to translate the layout for a tensor of type CUPHY_BIT to
// an equivalent tensor with 32-bit words.
tensor_layout_any word_layout_from_bit_layout(const tensor_layout_any& kLayout);

////////////////////////////////////////////////////////////////////////////
// bit_column_end_mask()
inline uint32_t bit_column_end_mask(int num_rows)
{
    const int      LAST_ROW_BITS = (0 == (num_rows % 32)) ? 32 : (num_rows % 32);
    const uint32_t COL_MASK      = static_cast<uint32_t>((1UL << LAST_ROW_BITS) - 1);
    return COL_MASK;
}

template <class T>
using cuphy_optional = std::experimental::optional<T>;

////////////////////////////////////////////////////////////////////////
// tensor_ref
// Combination of:
// 1.) a storage element type
// 2.) a layout class
// 3.) an address
// to allow host and device code to access elements of the tensor.
// The TLayout class provides calculation of the address of a specific
// element of the tensor.
template <cuphyDataType_t TType,
          int             NDim,
          bool            TIsConst     = false,
          template <int> class TLayout = tensor_layout_N>
class tensor_ref //
{
public:
    typedef typename std::conditional<TIsConst,
                                      const typename data_type_traits<TType>::type,
                                      typename data_type_traits<TType>::type>::type element_t;
    typedef typename data_type_traits<TType>::type                                  non_const_element_t;
    typedef TLayout<NDim>                                                           layout_t;

    tensor_ref(void* addr, const int* dim, const int* str) :
        addr_(static_cast<element_t*>(addr)),
        layout_(dim, str)
    {
        static_assert(!TIsConst, "Cannot initialize const tensor with non-const address");
    }
    tensor_ref(const void* addr, const int* dim, const int* str) :
        addr_(static_cast<element_t*>(addr)),
        layout_(dim, str)
    {
    }
    tensor_ref(void* addr, const layout_t& layout) :
        addr_(static_cast<element_t*>(addr)),
        layout_(layout)
    {
        static_assert(!TIsConst, "Cannot initialize const tensor with non-const address");
    }
    CUDA_BOTH_INLINE
    element_t* addr() { return addr_; }
    CUDA_BOTH_INLINE
    element_t* addr_at(const int (&idx)[NDim]) { return &addr_[layout_.offset(idx)]; }    
    CUDA_BOTH_INLINE
    element_t& operator()(const int (&idx)[NDim]) { return addr_[layout_.offset(idx)]; }
    CUDA_BOTH_INLINE
    const element_t& operator()(const int (&idx)[NDim]) const { return addr_[layout_.offset(idx)]; }
    //------------------------------------------------------------------
    // Generate a tensor_ref for a specific number of dimensions
    template <int TRank>
    cuphy_optional<tensor_ref<TType, TRank>> get_ref_rank()
    {
        if(layout_.rank() == TRank)
        {
            return tensor_ref<TType, TRank, TIsConst>(addr_,
                                                      layout_.dimensions.begin(),
                                                      layout_.strides.begin());
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    //------------------------------------------------------------------
    // Generate a contiguous tensor_ref for a specific rank
    template <int TRank>
    cuphy_optional<tensor_ref<TType, TRank, TIsConst, tensor_layout_contiguous_N>>
    get_ref_contig_rank()
    {
        if(1 == layout_.strides[0])
        {
            return tensor_ref<TType, TRank, TIsConst, tensor_layout_contiguous_N>(addr_,
                                                                                  layout_.dimensions.begin(),
                                                                                  layout_.strides.begin() + 1); // skip unit stride
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    //------------------------------------------------------------------
    // Generate a contiguous tensor_ref for a specific rank (const)
    template <int TRank>
    cuphy_optional<tensor_ref<TType, TRank, TIsConst, tensor_layout_contiguous_N>>
    get_ref_contig_rank() const
    {
        if(1 == layout_.strides[0])
        {
            return tensor_ref<TType, TRank, TIsConst, tensor_layout_contiguous_N>(addr_,
                                                                                  layout_.dimensions.begin(),
                                                                                  layout_.strides.begin() + 1); // skip unit stride
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    CUDA_BOTH_INLINE
    const layout_t& layout() const { return layout_; }

private:
    element_t* addr_;
    layout_t   layout_;
};

// clang-format off
template <cuphyDataType_t T, int NDim = CUPHY_DIM_MAX>     using optional_tensor_ref        = std::experimental::optional<tensor_ref<T, NDim>>;
template <cuphyDataType_t T, bool TIsConst = false>        using tensor_ref_any             = tensor_ref<T, CUPHY_DIM_MAX, TIsConst, tensor_layout_N>;
template <cuphyDataType_t T, bool TIsConst = false>        using tensor_ref_contig_any      = tensor_ref<T, CUPHY_DIM_MAX, TIsConst, tensor_layout_contiguous_N>;
template <cuphyDataType_t T, int N, bool TIsConst = false> using tensor_ref_contig_N        = tensor_ref<T, N, TIsConst, tensor_layout_contiguous_N>;
template <cuphyDataType_t T, bool TIsConst = false>        using tensor_ref_contig_1D       = tensor_ref<T, 1, TIsConst, tensor_layout_contiguous_N>;
template <cuphyDataType_t T, bool TIsConst = false>        using tensor_ref_contig_2D       = tensor_ref<T, 2, TIsConst, tensor_layout_contiguous_N>;
template <cuphyDataType_t T, bool TIsConst = false>        using tensor_ref_contig_3D       = tensor_ref<T, 3, TIsConst, tensor_layout_contiguous_N>;
template <cuphyDataType_t T>                               using const_tensor_ref_contig_1D = tensor_ref<T, 1, true, tensor_layout_contiguous_N>;
template <cuphyDataType_t T>                               using const_tensor_ref_contig_2D = tensor_ref<T, 2, true, tensor_layout_contiguous_N>;
template <cuphyDataType_t T>                               using const_tensor_ref_contig_3D = tensor_ref<T, 3, true, tensor_layout_contiguous_N>;
// clang-format on

////////////////////////////////////////////////////////////////////////
// tensor_ref_t
// Combination of:
// 1.) a storage element type
// 2.) a layout class
// 3.) an address
// to allow host and device code to access elements of the tensor.
// The TLayout class provides calculation of the address of a specific
// element of the tensor.
template <typename        T,
          int             NDim,
          template <int> class TLayout = tensor_layout_N>
class tensor_ref_t //
{
public:
    typedef T              element_t;
    typedef TLayout<NDim>  layout_t;

    tensor_ref_t(void* addr, const int* dim, const int* str) :
        addr_(static_cast<element_t*>(addr)),
        layout_(dim, str)
    {
    }
    tensor_ref_t(const void* addr, const int* dim, const int* str) :
        addr_(static_cast<element_t*>(addr)),
        layout_(dim, str)
    {
    }
    tensor_ref_t(void* addr, const layout_t& layout) :
        addr_(static_cast<element_t*>(addr)),
        layout_(layout)
    {
    }
    CUDA_BOTH_INLINE
    element_t* addr() { return addr_; }
    CUDA_BOTH_INLINE
    element_t& operator()(const int (&idx)[NDim]) { return addr_[layout_.offset(idx)]; }
    CUDA_BOTH_INLINE
    const element_t& operator()(const int (&idx)[NDim]) const { return addr_[layout_.offset(idx)]; }
    //------------------------------------------------------------------
    // Generate a tensor_ref for a specific number of dimensions
    template <int TRank>
    cuphy_optional<tensor_ref_t<T, TRank>> get_ref_rank()
    {
        if(layout_.rank() == TRank)
        {
            return tensor_ref_t<T, TRank>(addr_,
                                          layout_.dimensions.begin(),
                                          layout_.strides.begin());
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    //------------------------------------------------------------------
    // Generate a contiguous tensor_ref for a specific rank
    template <int TRank>
    cuphy_optional<tensor_ref_t<T, TRank, tensor_layout_contiguous_N>>
    get_ref_contig_rank()
    {
        if(1 == layout_.strides[0])
        {
            return tensor_ref_t<T, TRank, tensor_layout_contiguous_N>(addr_,
                                                                      layout_.dimensions.begin(),
                                                                      layout_.strides.begin() + 1); // skip unit stride
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    //------------------------------------------------------------------
    // Generate a contiguous tensor_ref for a specific rank (const)
    template <int TRank>
    cuphy_optional<tensor_ref_t<T, TRank, tensor_layout_contiguous_N>>
    get_ref_contig_rank() const
    {
        if(1 == layout_.strides[0])
        {
            return tensor_ref_t<T, TRank, tensor_layout_contiguous_N>(addr_,
                                                                      layout_.dimensions.begin(),
                                                                      layout_.strides.begin() + 1); // skip unit stride
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    CUDA_BOTH_INLINE
    const layout_t& layout() const { return layout_; }

private:
    element_t* addr_;
    layout_t   layout_;
};

// clang-format off
template <typename T, int NDim = CUPHY_DIM_MAX>     using optional_tensor_ref_t        = std::experimental::optional<tensor_ref_t<T, NDim>>;
template <typename T>                               using tensor_ref_t_any             = tensor_ref_t<T, CUPHY_DIM_MAX, tensor_layout_N>;
template <typename T>                               using tensor_ref_t_contig_any      = tensor_ref_t<T, CUPHY_DIM_MAX, tensor_layout_contiguous_N>;
template <typename T, int N>                        using tensor_ref_t_contig_N        = tensor_ref_t<T, N, tensor_layout_contiguous_N>;
template <typename T>                               using tensor_ref_t_contig_1D       = tensor_ref_t<T, 1, tensor_layout_contiguous_N>;
template <typename T>                               using tensor_ref_t_contig_2D       = tensor_ref_t<T, 2, tensor_layout_contiguous_N>;
template <typename T>                               using tensor_ref_t_contig_3D       = tensor_ref_t<T, 3, tensor_layout_contiguous_N>;
// clang-format on


////////////////////////////////////////////////////////////////////////
// tensor_desc
// The tensor_desc class combines a layout specification (i.e dimension
// sizes and strides) along with a specific element type enumeration
// (e.g. 32-bit floating point complex values).
// Upon construction, the element type is CUPHY_VOID. In this state,
// the tensor_desc is invalid, and no operations can make use of the
// tensor_desc.
// Users must call the set() function to place the tensor_desc into a
// usable state. If the arguments to the set() function do not
// represent a valid descriptor, the set() function will return false,
// and the descriptor will remain in its previous state.
class tensor_desc : public cuphyTensorDescriptor //
{
public:
    tensor_desc() noexcept :
        type_(CUPHY_VOID),
        size_bytes_(0) {}
    ~tensor_desc() noexcept = default;
    // Returns true for success, false if the type and/or layout are
    // invalid.
    bool                     set(cuphyDataType_t t,
                                 int             numDim,
                                 const int*      dim,
                                 const int*      strides) noexcept;
    cuphyDataType_t          type() const noexcept { return type_; }
    const tensor_layout_any& layout() const noexcept { return layout_; }
    size_t                   get_size_in_bytes() const noexcept { return size_bytes_; }
    //------------------------------------------------------------------
    // Generate a tensor_ref for a specific type and number of dimensions
    template <cuphyDataType_t TType, int TRank>
    cuphy_optional<tensor_ref<TType, TRank, false>> get_ref_rank(void* pv) const
    {
        if((TType == type()) && (layout().rank() == TRank))
        {
            return tensor_ref<TType, TRank, false>(pv,
                                                   layout().dimensions.begin(),
                                                   layout().strides.begin());
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    //------------------------------------------------------------------
    // Generate a tensor_ref for a specific type and number of dimensions
    template <typename T, int TRank>
    cuphy_optional<tensor_ref_t<T, TRank>> get_ref_t_rank(void* pv) const
    {
        if((type_to_cuphy_type<T>::value == type()) && (layout().rank() == TRank))
        {
            return tensor_ref_t<T, TRank>(pv,
                                          layout().dimensions.begin(),
                                          layout().strides.begin());
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    //------------------------------------------------------------------
    // Generate a tensor_ref for a specific type and number of dimensions
    // (const address)
    template <cuphyDataType_t TType, int TRank>
    cuphy_optional<tensor_ref<TType, TRank, true>> get_ref_rank(const void* pv) const
    {
        if((TType == type()) && (layout().rank() == TRank))
        {
            return tensor_ref<TType, TRank, true>(pv,
                                                  layout().dimensions.begin(),
                                                  layout().strides.begin());
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    //------------------------------------------------------------------
    // Generate a tensor_ref for a specific type, allowing any number of
    // dimensions by promoting to the maximum rank (CUPHY_DIM_MAX).
    template <cuphyDataType_t TType>
    cuphy_optional<tensor_ref<TType, CUPHY_DIM_MAX>> get_ref(void* pv) const
    {
        if(TType == type())
        {
            return tensor_ref<TType, CUPHY_DIM_MAX>(pv,
                                                    layout_.dimensions.begin(),
                                                    layout_.strides.begin());
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    //------------------------------------------------------------------
    // Generate a tensor_ref for a specific type, allowing any number of
    // dimensions by promoting to the maximum rank (CUPHY_DIM_MAX).
    template <typename T>
    cuphy_optional<tensor_ref_t<T, CUPHY_DIM_MAX>> get_ref_t(void* pv) const
    {
        if(type_to_cuphy_type<T>::value == type())
        {
            return tensor_ref_t<T, CUPHY_DIM_MAX>(pv,
                                                  layout_.dimensions.begin(),
                                                  layout_.strides.begin());
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    //------------------------------------------------------------------
    // Generate a tensor_ref for a specific type, allowing any number of
    // dimensions by promoting to the maximum rank (CUPHY_DIM_MAX)
    // (const address)
    template <cuphyDataType_t TType>
    cuphy_optional<tensor_ref<TType, CUPHY_DIM_MAX>> get_ref(const void* pv) const
    {
        if(TType == type())
        {
            return tensor_ref<TType, CUPHY_DIM_MAX, true>(pv,
                                                          layout_.dimensions.begin(),
                                                          layout_.strides.begin());
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    //------------------------------------------------------------------
    // Generate a tensor_ref that is contiguous in the first dimension
    // for a specific type.
    template <cuphyDataType_t TType, int TRank>
    cuphy_optional<tensor_ref<TType, TRank, false, tensor_layout_contiguous_N>>
    get_ref_contig_rank(void* pv) const
    {
        if((TType == type()) && (1 == layout_.strides[0]))
        {
            return tensor_ref<TType, TRank, false, tensor_layout_contiguous_N>(pv,
                                                                               layout_.dimensions.begin(),
                                                                               layout_.strides.begin() + 1); // skip unit stride
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    //------------------------------------------------------------------
    // Generate a tensor_ref that is contiguous in the first dimension
    // for a specific type.
    template <typename T, int TRank>
    cuphy_optional<tensor_ref_t<T, TRank, tensor_layout_contiguous_N>>
    get_ref_contig_rank_t(void* pv) const
    {
        if((type_to_cuphy_type<T>::value == type()) && (1 == layout_.strides[0]))
        {
            return tensor_ref_t<T, TRank, tensor_layout_contiguous_N>(pv,
                                                                      layout_.dimensions.begin(),
                                                                      layout_.strides.begin() + 1); // skip unit stride
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    //------------------------------------------------------------------
    // Generate a tensor_ref that is contiguous in the first dimension
    // for a specific type.
    template <typename T, int TRank>
    cuphy_optional<tensor_ref_t<T, TRank, tensor_layout_contiguous_N>>
    get_ref_contig_rank_t(const void* pv) const
    {
        if((type_to_cuphy_type<typename std::remove_const<T>::type>::value == type()) && (1 == layout_.strides[0]))
        {
            return tensor_ref_t<T, TRank, tensor_layout_contiguous_N>(pv,
                                                                      layout_.dimensions.begin(),
                                                                      layout_.strides.begin() + 1); // skip unit stride
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    //------------------------------------------------------------------
    // Generate a tensor_ref that is contiguous in the first dimension
    // for a specific type.
    // (const address)
    template <cuphyDataType_t TType, int TRank>
    cuphy_optional<tensor_ref<TType, TRank, true, tensor_layout_contiguous_N>>
    get_ref_contig_rank(const void* pv) const
    {
        if((TType == type()) && (1 == layout_.strides[0]))
        {
            return tensor_ref<TType, TRank, true, tensor_layout_contiguous_N>(pv,
                                                                              layout_.dimensions.begin(),
                                                                              layout_.strides.begin() + 1); // skip unit stride
        }
        else
        {
            return std::experimental::nullopt;
        }
    }

private:
    cuphyDataType_t   type_;
    tensor_layout_any layout_;
    size_t            size_bytes_; // includes padding induced by strides
};

typedef std::pair<std::reference_wrapper<const tensor_desc>, void*>       tensor_pair;
typedef std::pair<std::reference_wrapper<const tensor_desc>, const void*> const_tensor_pair;


template <typename T>
void copyTensorPrm2Info(cuphyTensorPrm_t const& tPrm, T& tInfo)
{
    tInfo.pAddr              = tPrm.pAddr;
    const tensor_desc& tDesc = static_cast<const tensor_desc&>(*tPrm.desc);
    tInfo.elemType           = tDesc.type();
    std::copy_n(tDesc.layout().strides.begin(), std::extent<decltype(tInfo.strides)>::value, tInfo.strides);
    // printf("elemType %d\n", tInfo.elemType);
}

#endif // !defined(TENSOR_DESC_HPP_INCLUDED_)
