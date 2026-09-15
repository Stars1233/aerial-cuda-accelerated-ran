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

#include <gtest/gtest.h>
#include "cuphy.hpp"
#include "convert_tensor.cuh"

namespace
{

// Helper functions for type traits and size calculations
namespace detail {
    // Get size in bytes for a type
    constexpr size_t get_size_in_bytes(cuphyDataType_t type) {
        switch(type) {
            case CUPHY_BIT: return sizeof(uint32_t);
            case CUPHY_R_8I:
            case CUPHY_R_8U: return 1;
            case CUPHY_C_8I:
            case CUPHY_C_8U: return 2;
            case CUPHY_R_16I:
            case CUPHY_R_16U:
            case CUPHY_R_16F: return 2;
            case CUPHY_C_16I:
            case CUPHY_C_16U:
            case CUPHY_C_16F: return 4;
            case CUPHY_R_32I:
            case CUPHY_R_32U:
            case CUPHY_R_32F: return 4;
            case CUPHY_C_32I:
            case CUPHY_C_32U:
            case CUPHY_C_32F: return 8;
            case CUPHY_R_64F: return 8;
            case CUPHY_C_64F: return 16;
            case CUPHY_R_8F_E4M3:
            case CUPHY_R_8F_E5M2: return 1;
            default: return 0;
        }
    }

    // Check if type is complex
    template<cuphyDataType_t T>
    struct is_complex_type {
        static constexpr bool value =
            T == CUPHY_C_8I || T == CUPHY_C_8U ||
            T == CUPHY_C_16I || T == CUPHY_C_16U || T == CUPHY_C_16F ||
            T == CUPHY_C_32I || T == CUPHY_C_32U || T == CUPHY_C_32F ||
            T == CUPHY_C_64F;
    };

    // Get base type for complex numbers
    template<cuphyDataType_t T>
    struct complex_base_type {
        using type = void;  // Default case
    };

    // Specializations for each complex type
    template<> struct complex_base_type<CUPHY_C_8I> { using type = int8_t; };
    template<> struct complex_base_type<CUPHY_C_8U> { using type = uint8_t; };
    template<> struct complex_base_type<CUPHY_C_16I> { using type = int16_t; };
    template<> struct complex_base_type<CUPHY_C_16U> { using type = uint16_t; };
    template<> struct complex_base_type<CUPHY_C_16F> { using type = __half; };
    template<> struct complex_base_type<CUPHY_C_32I> { using type = int32_t; };
    template<> struct complex_base_type<CUPHY_C_32U> { using type = uint32_t; };
    template<> struct complex_base_type<CUPHY_C_32F> { using type = float; };
    template<> struct complex_base_type<CUPHY_C_64F> { using type = double; };

    // Get type for real numbers
    template<cuphyDataType_t T>
    struct real_type {
        using type = void;  // Default case
    };

    // Specializations for each real type
    template<> struct real_type<CUPHY_R_8I>      { using type = int8_t; };
    template<> struct real_type<CUPHY_R_8U>      { using type = uint8_t; };
    template<> struct real_type<CUPHY_R_16I>     { using type = int16_t; };
    template<> struct real_type<CUPHY_R_16U>     { using type = uint16_t; };
    template<> struct real_type<CUPHY_R_16F>     { using type = __half; };
    template<> struct real_type<CUPHY_R_32I>     { using type = int32_t; };
    template<> struct real_type<CUPHY_R_32U>     { using type = uint32_t; };
    template<> struct real_type<CUPHY_R_32F>     { using type = float; };
    template<> struct real_type<CUPHY_R_64F>     { using type = double; };
    template<> struct real_type<CUPHY_R_8F_E4M3> { using type = __nv_fp8_e4m3; };
    template<> struct real_type<CUPHY_R_8F_E5M2> { using type = __nv_fp8_e5m2; };
}

////////////////////////////////////////////////////////////////////////
// do_convert_test_to_bits()
template <cuphyDataType_t TType>
void do_convert_test_to_bits(int NUM_ROWS,
                             int NUM_COLS)
{
    typedef cuphy::typed_tensor<TType,     cuphy::pinned_alloc> tensor_p;
    typedef cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc> tensor_bit_p;

    //const std::array<int, 2> SRC_DIMS = {{NUM_ROWS, NUM_COLS}};
    //------------------------------------------------------------------
    // Allocate tensors
    tensor_p     tSrc(NUM_ROWS, NUM_COLS);
    tensor_bit_p tDst(NUM_ROWS, NUM_COLS, cuphy::tensor_flags::align_coalesce);
    //------------------------------------------------------------------
    // Initialize the source tensor with random values
    cuphy::rng rng;
    rng.uniform(tSrc, 0, 1);
    //------------------------------------------------------------------
    // Convert to the CUPHY_BIT type
    cuphy::tensor_convert(tDst, tSrc);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    
    //printf("Input:\n");
    //for(int i = 0; i < NUM_ROWS; ++i)
    //{
    //    printf("[%2i]: ", i);
    //    for(int j = 0; j < NUM_COLS; ++j)
    //    {
    //        printf("%i ", tSrc(i, j));
    //    }
    //    printf("\n");
    //}
    const int NUM_WORDS    = (NUM_ROWS + 31) / 32;
    for(int i = 0; i < NUM_WORDS; ++i)
    {
      //printf("[%2i]: ", i);
        for(int j = 0; j < NUM_COLS; ++j)
        {
            for(int k = 0; k < 32; ++k)
            {
                if(((i * 32) + k) < NUM_ROWS)
                {
                    uint32_t convert_bit = (tDst(i, j) >> k) & 0x1;
                    uint32_t expected_bit = (0 == tSrc(i * 32 + k, j)) ? 0 : 1;
                    EXPECT_EQ(convert_bit, expected_bit)
                      << "ROW = "   << (i * 32 + k)
                      << ", COL = " << j
                      <<", BIT = "  << k
                      <<", WORD = " << std::hex << tDst(i, j) << std::dec
                      <<", SRC = "  << tSrc(i*32+k, j)
                      << std::endl;
                }
                //printf("0x%X ", tDst(i, j));
            }
        }
        //printf("\n");
    }
}

////////////////////////////////////////////////////////////////////////
// do_convert_test_copy_bits()
void do_convert_test_copy_bits(int SRC_NUM_ROWS,
                               int DST_NUM_ROWS,
                               int NUM_COLS)
{
    typedef cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc> tensor_bit_p;

    //const std::array<int, 2> SRC_DIMS = {{NUM_ROWS, NUM_COLS}};
    //------------------------------------------------------------------
    // Allocate tensors
    tensor_bit_p tSrc(SRC_NUM_ROWS, NUM_COLS);
    tensor_bit_p tDst(DST_NUM_ROWS, NUM_COLS);
    //------------------------------------------------------------------
    // Initialize the source tensor with random values
    cuphy::rng rng;
    rng.uniform(tSrc, 0, 1);
    //------------------------------------------------------------------
    // Copy from source to destination
    tensor_copy_range(tDst,
                      tSrc,
                      cuphy::index_group(cuphy::index_range(0, DST_NUM_ROWS),
                                         cuphy::dim_all()));
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    
    //printf("Input:\n");
    //for(int i = 0; i < NUM_ROWS; ++i)
    //{
    //    printf("[%2i]: ", i);
    //    for(int j = 0; j < NUM_COLS; ++j)
    //    {
    //        printf("%i ", tSrc(i, j));
    //    }
    //    printf("\n");
    //}
    const int DST_NUM_WORDS    = (DST_NUM_ROWS + 31) / 32;
    for(int i = 0; i < DST_NUM_WORDS; ++i)
    {
        for(int j = 0; j < NUM_COLS; ++j)
        {
            const uint32_t SRC_WORD = tSrc(i, j);
            const uint32_t DST_WORD = tDst(i, j);
            //printf("[%i, %i]: SRC = 0x%X, DST = 0x%X\n", i, j, SRC_WORD, DST_WORD);
            for(int k = 0; k < 32; ++k)
            {
                const uint32_t DST_BIT = (DST_WORD >> k) & 0x1U;
                if(((i * 32) + k) < DST_NUM_ROWS)
                {
                    const uint32_t SRC_BIT = (SRC_WORD >> k) & 0x1U;
                    EXPECT_EQ(DST_BIT, SRC_BIT)
                      << "ROW = "   << (i * 32 + k)
                      << ", COL = " << j
                      <<", BIT = "  << k
                      <<", WORD = " << std::hex << DST_WORD << std::dec
                      <<", SRC = "  << std::hex << SRC_WORD << std::dec
                      << std::endl;
                }
                else
                {
                    EXPECT_EQ(DST_BIT, 0U)
                      << "ROW = "   << (i * 32 + k)
                      << ", COL = " << j
                      <<", BIT = "  << k
                      <<", WORD = " << std::hex << DST_WORD << std::dec
                      <<", SRC = "  << std::hex << SRC_WORD << std::dec
                      << std::endl;
                }
            }
        }
        //printf("\n");
    }
}

template<cuphyDataType_t Type>
constexpr double get_tolerance() {
    if constexpr (Type == CUPHY_R_8F_E4M3 || Type == CUPHY_R_8F_E5M2) {
        return 10.0f;
    } else if constexpr (Type == CUPHY_R_16F || Type == CUPHY_C_16F) {
        return 1e-3;  // Looser tolerance for float16
    } else if constexpr (Type == CUPHY_R_32F || Type == CUPHY_C_32F) {
        return 1e-5;  // Standard tolerance for float32
    } else if constexpr (Type == CUPHY_R_64F || Type == CUPHY_C_64F) {
        return 1e-10; // Tighter tolerance for float64
    } else {
        return 0.0;   // Exact comparison for integer types
    }
}

template <typename T>
bool is_zero(T t)
{
    return t == 0;
}
template <>
bool is_zero<__half>(__half t)
{
    return (__half2float(t) == 0.0f);
}

template <>
bool is_zero<__nv_fp8_e4m3>(__nv_fp8_e4m3 t)
{
    return static_cast<float>(t) == 0.0f;
}
template <>
bool is_zero<__nv_fp8_e5m2>(__nv_fp8_e5m2 t)
{
    return static_cast<float>(t) == 0.0f;
}

template <typename T>
struct display_value
{
    T value_;
    display_value(T t) : value_(t) {}
    T value() const
    {
        return value_;
    }
};
template <>
struct display_value<__half>
{
    display_value(__half t) : value_(t) {}
    float value() const
    {
        return __half2float(value_);
    }
    __half value_;
};
template <>
struct display_value<__nv_fp8_e4m3>
{
    display_value(__nv_fp8_e4m3 t) : value_(t) {}
    float value() const
    {
        return static_cast<float>(value_);
    }
    __nv_fp8_e4m3 value_;
};
template <>
struct display_value<__nv_fp8_e5m2>
{
    display_value(__nv_fp8_e5m2 t) : value_(t) {}
    float value() const
    {
        return static_cast<float>(value_);
    }
    __nv_fp8_e5m2 value_;
};

template <cuphyDataType_t TDst, cuphyDataType_t TSrc>
void do_convert_test_types(int NUM_ROWS, int NUM_COLS) {
    typedef cuphy::typed_tensor<TSrc, cuphy::pinned_alloc> tensor_src;
    typedef cuphy::typed_tensor<TDst, cuphy::pinned_alloc> tensor_dst;

    // Allocate tensors
    tensor_src tSrc(NUM_ROWS, NUM_COLS);
    tensor_dst tDst(NUM_ROWS, NUM_COLS);

    // Initialize source tensor with random values
    cuphy::rng rng;
    if constexpr (TSrc == CUPHY_BIT) {
        rng.uniform(tSrc, 0, 1);
    } else if constexpr (TSrc == CUPHY_R_32F || TSrc == CUPHY_R_64F || TSrc == CUPHY_R_8F_E4M3 || TSrc == CUPHY_R_8F_E5M2) {
        rng.uniform(tSrc, -1.0f, 1.0f);
    } else {
        rng.uniform(tSrc, 0, 255);
    }

    // Convert between types
    cuphy::tensor_convert(tDst, tSrc);

    // Wait for results
    cudaStreamSynchronize(0);

    // Get appropriate tolerance based on destination type
    const double tolerance = get_tolerance<TDst>();

    if constexpr (TDst == CUPHY_BIT) {
        const int NUM_WORDS = (NUM_ROWS + 31) / 32;
        for(int i = 0; i < NUM_WORDS; ++i) {
            for(int j = 0; j < NUM_COLS; ++j) {
                for(int k = 0; k < 32; ++k) {
                    if(((i * 32) + k) < NUM_ROWS) {
                        // verification of conversion from BIT type
                        uint32_t convert_bit = (tDst(i, j) >> k) & 0x1;
                        uint32_t expected_bit = is_zero(tSrc(i * 32 + k, j)) ? 0 : 1;
                        if (convert_bit != expected_bit) {
                            std::cout << "Debug info:" << std::endl
                                     << "  Word index: " << i << std::endl
                                     << "  Column: " << j << std::endl
                                     << "  Bit index: " << k << std::endl
                                     << "  Row: " << (i * 32 + k) << std::endl
                                     << "  Word value: 0x" << std::hex << tDst(i, j) << std::dec << std::endl
                                     << "  Source value: " << display_value{tSrc(i * 32 + k, j)}.value() << std::endl;
                        }
                        EXPECT_EQ(convert_bit, expected_bit)
                            << "ROW = " << (i * 32 + k)
                            << ", COL = " << j
                            << ", BIT = " << k
                            << ", WORD = " << std::hex << tDst(i, j) << std::dec
                            << ", SRC = " << display_value{tSrc(i * 32 + k, j)}.value();
                    }
                }
            }
        }
    } else if constexpr ((TSrc == CUPHY_BIT)) {
        // verification of conversion from BIT type
        const int NUM_WORDS = (NUM_ROWS + 31) / 32;
        for(int i = 0; i < NUM_ROWS; ++i) {
            for(int j = 0; j < NUM_COLS; ++j) {
                int word_idx = i / 32;
                int bit_idx = i % 32;
                uint32_t src_bit = (tSrc(word_idx, j) >> bit_idx) & 0x1;

                if constexpr (TDst == CUPHY_R_16F) {
                    float dst_val = __half2float(tDst(i, j));
                    float expected_val = src_bit ? 1.0f : 0.0f;
                    EXPECT_FLOAT_EQ(dst_val, expected_val)
                        << "Row = " << i << ", Col = " << j;
                } else {
                    double dst_val = static_cast<double>(tDst(i, j));
                    double expected_val = src_bit ? 1.0 : 0.0;
                    EXPECT_DOUBLE_EQ(dst_val, expected_val)
                        << "Row = " << i << ", Col = " << j;
                }
            }
        }
    } else {
        for(int i = 0; i < NUM_ROWS; ++i) {
            for(int j = 0; j < NUM_COLS; ++j) {
                if constexpr (TDst == CUPHY_R_32F || TDst == CUPHY_R_64F ||
                             TDst == CUPHY_R_16F || TDst == CUPHY_C_16F ||
                             TSrc == CUPHY_R_16F || TSrc == CUPHY_C_16F ||
                             TSrc == CUPHY_R_32F || TSrc == CUPHY_R_64F) {
                    EXPECT_NEAR(static_cast<double>(tDst(i, j)),
                               static_cast<double>(tSrc(i, j)),
                               tolerance)
                        << "Row = " << i << ", Col = " << j;
                } else {
                    EXPECT_EQ(static_cast<double>(tDst(i, j)),
                             static_cast<double>(tSrc(i, j)))
                        << "Row = " << i << ", Col = " << j;
                }
            }
        }
    }
}

template <cuphyDataType_t TDst, cuphyDataType_t TSrc>
void do_convert_test_complex(int NUM_ROWS, int NUM_COLS) {
    typedef cuphy::typed_tensor<TSrc, cuphy::pinned_alloc> tensor_src;
    typedef cuphy::typed_tensor<TDst, cuphy::pinned_alloc> tensor_dst;

    // Allocate tensors
    tensor_src tSrc(NUM_ROWS, NUM_COLS);
    tensor_dst tDst(NUM_ROWS, NUM_COLS);

    // Initialize source tensor with random values
    //cuphy::rng rng;
    //rng.uniform(tSrc, 0, 1);

    // Initialize source tensor manually
    for(int i = 0; i < NUM_ROWS; ++i) {
        for(int j = 0; j < NUM_COLS; ++j) {
            if constexpr (TSrc == CUPHY_C_8I) {
                tSrc(i, j).x = static_cast<int8_t>(i % 127);
                tSrc(i, j).y = static_cast<int8_t>(j % 127);
            } else if constexpr (TSrc == CUPHY_C_8U) {
                tSrc(i, j).x = static_cast<uint8_t>(i % 255);
                tSrc(i, j).y = static_cast<uint8_t>(j % 255);
            } else if constexpr (TSrc == CUPHY_C_16I) {
                tSrc(i, j).x = static_cast<int16_t>(i % 32767);
                tSrc(i, j).y = static_cast<int16_t>(j % 32767);
            } else if constexpr (TSrc == CUPHY_C_16U) {
                tSrc(i, j).x = static_cast<uint16_t>(i % 65535);
                tSrc(i, j).y = static_cast<uint16_t>(j % 65535);
            } else if constexpr (TSrc == CUPHY_C_16F) {
                tSrc(i, j).x = __float2half(i * 0.1f);
                tSrc(i, j).y = __float2half(j * 0.1f);
            } else if constexpr (TSrc == CUPHY_C_32F) {
                tSrc(i, j).x = static_cast<float>(i * 0.1f);
                tSrc(i, j).y = static_cast<float>(j * 0.1f);
            } else if constexpr (TSrc == CUPHY_C_64F) {
                tSrc(i, j).x = static_cast<double>(i * 0.1);
                tSrc(i, j).y = static_cast<double>(j * 0.1);
            }
        }
    }

    // Convert between types
    cuphy::tensor_convert(tDst, tSrc);

    // Wait for results
    cudaStreamSynchronize(0);

    // Get appropriate tolerance based on destination type
    const double tolerance = get_tolerance<TDst>();

    // Verify results based on expected type conversion behavior
    for(int i = 0; i < NUM_ROWS; ++i) {
        for(int j = 0; j < NUM_COLS; ++j) {
            // Add appropriate comparison based on types
            // This is a basic example - you'll need to adjust based on type
            double dst_real, dst_imag, src_real, src_imag;
            if constexpr (TDst == CUPHY_C_16F) {
                dst_real = static_cast<double>(__half2float(tDst(i, j).x));
                dst_imag = static_cast<double>(__half2float(tDst(i, j).y));
            } else if constexpr (TDst == CUPHY_C_32F) {
                dst_real = static_cast<double>(tDst(i, j).x);
                dst_imag = static_cast<double>(tDst(i, j).y);
            } else if constexpr (TDst == CUPHY_C_64F) {
                dst_real = static_cast<double>(tDst(i, j).x);
                dst_imag = static_cast<double>(tDst(i, j).y);
            } else {
                dst_real = static_cast<double>(tDst(i, j).x);
                dst_imag = static_cast<double>(tDst(i, j).y);
            }

            if constexpr (TSrc == CUPHY_C_16F) {
                src_real = static_cast<double>(__half2float(tSrc(i, j).x));
                src_imag = static_cast<double>(__half2float(tSrc(i, j).y));
            } else if constexpr (TSrc == CUPHY_C_32F) {
                src_real = static_cast<double>(tSrc(i, j).x);
                src_imag = static_cast<double>(tSrc(i, j).y);
            } else if constexpr (TSrc == CUPHY_C_64F) {
                src_real = static_cast<double>(tSrc(i, j).x);
                src_imag = static_cast<double>(tSrc(i, j).y);
            } else {
                src_real = static_cast<double>(tSrc(i, j).x);
                src_imag = static_cast<double>(tSrc(i, j).y);
            }

            EXPECT_NEAR(dst_real, src_real, tolerance)
                << "Row = " << i << ", Col = " << j;
            EXPECT_NEAR(dst_imag, src_imag, tolerance)
                << "Row = " << i << ", Col = " << j;
        }
    }
}

// Test invalid conversion attempts
template<cuphyDataType_t DstType, cuphyDataType_t SrcType>
void test_invalid_type_conversion() {
    // use conversion from real to complex to trigger invalid conversion
    typedef cuphy::typed_tensor<SrcType, cuphy::pinned_alloc> tensor_src;
    typedef cuphy::typed_tensor<DstType, cuphy::pinned_alloc> tensor_dst;

    tensor_src tSrc(10, 10);
    tensor_dst tDst(10, 10);

    try {
        cuphy::tensor_convert(tDst, tSrc);
        FAIL() << "Expected exception for invalid conversion";
    } catch (const std::exception& e) {
        SUCCEED() << "Caught expected exception for conversion" << e.what();
    } catch (...) {
        SUCCEED() << "Caught expected exception for conversion";
    }
}

// Test direct same type conversions with convert_tensor_layout
template <cuphyDataType_t TType>
void do_convert_test_layout_same_type(int NUM_ROWS, int NUM_COLS) {
    // Create source and destination tensors with different layouts
    const int src_dims[2] = {NUM_ROWS, NUM_COLS};
    const int dst_dims[2] = {NUM_ROWS, NUM_COLS};

    // Source: row-major layout
    const int src_strides[2] = {NUM_COLS, 1};
    // Destination: column-major layout
    const int dst_strides[2] = {1, NUM_ROWS};

    // Create tensor descriptors
    tensor_desc srcDesc;
    tensor_desc dstDesc;

    // Initialize tensor descriptors using set() method
    bool success = srcDesc.set(TType, 2, src_dims, src_strides);
    EXPECT_TRUE(success) << "Failed to set source tensor descriptor";

    success = dstDesc.set(TType, 2, dst_dims, dst_strides);
    EXPECT_TRUE(success) << "Failed to set destination tensor descriptor";

    // Allocate memory for source and destination
    size_t elem_size = detail::get_size_in_bytes(TType);
    size_t total_size = NUM_ROWS * NUM_COLS * elem_size;

    void* src_data = nullptr;
    void* dst_data = nullptr;
    EXPECT_EQ(cudaMalloc(&src_data, total_size), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dst_data, total_size), cudaSuccess);

    // Initialize source data with pattern
    std::vector<uint8_t> host_src(total_size);
    std::vector<uint8_t> host_dst(total_size);

    if constexpr (detail::is_complex_type<TType>::value) {
        // For complex types
        for(int i = 0; i < NUM_ROWS; ++i) {
            for(int j = 0; j < NUM_COLS; ++j) {
                size_t idx = (i * src_strides[0] + j * src_strides[1]) * elem_size;
                if constexpr (TType == CUPHY_C_8I) {
                    reinterpret_cast<int8_t*>(host_src.data() + idx)[0] = static_cast<int8_t>((i + j) % 127);
                    reinterpret_cast<int8_t*>(host_src.data() + idx)[1] = static_cast<int8_t>((i * j) % 127);
                } else if constexpr (TType == CUPHY_C_8U) {
                    reinterpret_cast<uint8_t*>(host_src.data() + idx)[0] = static_cast<uint8_t>((i + j) % 255);
                    reinterpret_cast<uint8_t*>(host_src.data() + idx)[1] = static_cast<uint8_t>((i * j) % 255);
                } else if constexpr (TType == CUPHY_C_16I) {
                    reinterpret_cast<int16_t*>(host_src.data() + idx)[0] = static_cast<int16_t>((i + j) % 32767);
                    reinterpret_cast<int16_t*>(host_src.data() + idx)[1] = static_cast<int16_t>((i * j) % 32767);
                } else if constexpr (TType == CUPHY_C_16U) {
                    reinterpret_cast<uint16_t*>(host_src.data() + idx)[0] = static_cast<uint16_t>((i + j) % 65535);
                    reinterpret_cast<uint16_t*>(host_src.data() + idx)[1] = static_cast<uint16_t>((i * j) % 65535);
                } else if constexpr (TType == CUPHY_C_16F) {
                    reinterpret_cast<__half*>(host_src.data() + idx)[0] = __float2half(i * 0.1f + j * 0.2f);
                    reinterpret_cast<__half*>(host_src.data() + idx)[1] = __float2half(i * 0.3f + j * 0.1f);
                } else if constexpr (TType == CUPHY_C_32F) {
                    reinterpret_cast<float*>(host_src.data() + idx)[0] = i * 0.1f + j * 0.2f;
                    reinterpret_cast<float*>(host_src.data() + idx)[1] = i * 0.3f + j * 0.1f;
                } else if constexpr (TType == CUPHY_C_32I) {
                    reinterpret_cast<int32_t*>(host_src.data() + idx)[0] = static_cast<int32_t>((i + j) % INT32_MAX);
                    reinterpret_cast<int32_t*>(host_src.data() + idx)[1] = static_cast<int32_t>((i * j) % INT32_MAX);
                } else if constexpr (TType == CUPHY_C_32U) {
                    reinterpret_cast<uint32_t*>(host_src.data() + idx)[0] = static_cast<uint32_t>((i + j) % UINT32_MAX);
                    reinterpret_cast<uint32_t*>(host_src.data() + idx)[1] = static_cast<uint32_t>((i * j) % UINT32_MAX);
                } else if constexpr (TType == CUPHY_C_64F) {
                    reinterpret_cast<double*>(host_src.data() + idx)[0] = i * 0.1 + j * 0.2;
                    reinterpret_cast<double*>(host_src.data() + idx)[1] = i * 0.3 + j * 0.1;
                }
            }
        }
    } else {
        // For real types
        for(int i = 0; i < NUM_ROWS; ++i) {
            for(int j = 0; j < NUM_COLS; ++j) {
                size_t idx = (i * src_strides[0] + j * src_strides[1]) * elem_size;
                if constexpr (TType == CUPHY_R_8I) {
                    reinterpret_cast<int8_t*>(host_src.data())[i * src_strides[0] + j] = static_cast<int8_t>((i + j) % 127);
                } else if constexpr (TType == CUPHY_R_8U) {
                    reinterpret_cast<uint8_t*>(host_src.data())[i * src_strides[0] + j] = static_cast<uint8_t>((i + j) % 255);
                } else if constexpr (TType == CUPHY_R_16I) {
                    reinterpret_cast<int16_t*>(host_src.data() + idx)[0] = static_cast<int16_t>((i + j) % 32767);
                } else if constexpr (TType == CUPHY_R_16U) {
                    reinterpret_cast<uint16_t*>(host_src.data() + idx)[0] = static_cast<uint16_t>((i + j) % 65535);
                } else if constexpr (TType == CUPHY_R_16F) {
                    reinterpret_cast<__half*>(host_src.data() + idx)[0] = __float2half(i * 0.1f + j * 0.2f);
                } else if constexpr (TType == CUPHY_R_32F) {
                    reinterpret_cast<float*>(host_src.data() + idx)[0] = i * 0.1f + j * 0.2f;
                } else if constexpr (TType == CUPHY_R_32I) {
                    reinterpret_cast<int32_t*>(host_src.data() + idx)[0] = static_cast<int32_t>((i + j) % INT32_MAX);
                } else if constexpr (TType == CUPHY_R_32U) {
                    reinterpret_cast<uint32_t*>(host_src.data() + idx)[0] = static_cast<uint32_t>((i + j) % UINT32_MAX);
                } else if constexpr (TType == CUPHY_R_64F) {
                    reinterpret_cast<double*>(host_src.data() + idx)[0] = i * 0.1 + j * 0.2;
                }
            }
        }
    }

    // Copy data to device
    EXPECT_EQ(cudaMemcpy(src_data, host_src.data(), total_size, cudaMemcpyHostToDevice), cudaSuccess);

    // Perform conversion
    cuphyStatus_t status = convert_tensor_layout(dstDesc, dst_data, srcDesc, src_data, 0);
    EXPECT_EQ(status, CUPHY_STATUS_SUCCESS) << "convert_tensor_layout failed";

    // Copy result back to host
    EXPECT_EQ(cudaMemcpy(host_dst.data(), dst_data, total_size, cudaMemcpyDeviceToHost), cudaSuccess);

    // Verify results
    const double tolerance = get_tolerance<TType>();

    for(int i = 0; i < NUM_ROWS; ++i) {
        for(int j = 0; j < NUM_COLS; ++j) {
            size_t src_idx = (i * src_strides[0] + j * src_strides[1]) * elem_size;
            size_t dst_idx = (i * dst_strides[0] + j * dst_strides[1]) * elem_size;

            if constexpr (detail::is_complex_type<TType>::value) {
                double src_real, src_imag, dst_real, dst_imag;

                if constexpr (TType == CUPHY_C_16F) {
                    src_real = static_cast<double>(__half2float(reinterpret_cast<__half*>(host_src.data() + src_idx)[0]));
                    src_imag = static_cast<double>(__half2float(reinterpret_cast<__half*>(host_src.data() + src_idx)[1]));
                    dst_real = static_cast<double>(__half2float(reinterpret_cast<__half*>(host_dst.data() + dst_idx)[0]));
                    dst_imag = static_cast<double>(__half2float(reinterpret_cast<__half*>(host_dst.data() + dst_idx)[1]));
                } else {
                    src_real = static_cast<double>(reinterpret_cast<typename detail::complex_base_type<TType>::type*>(host_src.data() + src_idx)[0]);
                    src_imag = static_cast<double>(reinterpret_cast<typename detail::complex_base_type<TType>::type*>(host_src.data() + src_idx)[1]);
                    dst_real = static_cast<double>(reinterpret_cast<typename detail::complex_base_type<TType>::type*>(host_dst.data() + dst_idx)[0]);
                    dst_imag = static_cast<double>(reinterpret_cast<typename detail::complex_base_type<TType>::type*>(host_dst.data() + dst_idx)[1]);
                }

                EXPECT_NEAR(dst_real, src_real, tolerance)
                    << "Mismatch at i=" << i << ", j=" << j << " (real part)";
                EXPECT_NEAR(dst_imag, src_imag, tolerance)
                    << "Mismatch at i=" << i << ", j=" << j << " (imaginary part)";
            } else {
                double src_val, dst_val;

                if constexpr (TType == CUPHY_R_16F) {
                    src_val = static_cast<double>(__half2float(reinterpret_cast<__half*>(host_src.data() + src_idx)[0]));
                    dst_val = static_cast<double>(__half2float(reinterpret_cast<__half*>(host_dst.data() + dst_idx)[0]));
                } else {
                    src_val = static_cast<double>(reinterpret_cast<typename detail::real_type<TType>::type*>(host_src.data() + src_idx)[0]);
                    dst_val = static_cast<double>(reinterpret_cast<typename detail::real_type<TType>::type*>(host_dst.data() + dst_idx)[0]);
                }

                EXPECT_NEAR(dst_val, src_val, tolerance)
                    << "Mismatch at i=" << i << ", j=" << j;
            }
        }
    }

    // Cleanup
    cudaFree(src_data);
    cudaFree(dst_data);
}

} // namespace

////////////////////////////////////////////////////////////////////////
// Convert.ToBits
// Test conversion of scalar types to bits
TEST(Convert, ToBits)
{
    do_convert_test_to_bits<CUPHY_R_32U>(32, 8);
    do_convert_test_to_bits<CUPHY_R_8U>(100, 3);

}

////////////////////////////////////////////////////////////////////////
// Convert.BitCopySubset
// Test copy of subsets of bit tensor - in particular, the "zeroing" of
// bits in the last word of each column
TEST(Convert, BitCopySubset)
{
    do_convert_test_copy_bits(64,    33, 17);
    do_convert_test_copy_bits(64,    64, 1);
    do_convert_test_copy_bits(91,    51, 3);
    do_convert_test_copy_bits(1024, 511, 44);
}

// Test conversion of scalar types to bits
TEST(Convert, ToBitsset) {
    do_convert_test_types<CUPHY_BIT, CUPHY_R_8I>(32, 8);
    do_convert_test_types<CUPHY_BIT, CUPHY_R_8U>(32, 8);
    do_convert_test_types<CUPHY_BIT, CUPHY_R_16F>(32, 8);
    do_convert_test_types<CUPHY_BIT, CUPHY_R_16I>(32, 8);
    do_convert_test_types<CUPHY_BIT, CUPHY_R_16U>(32, 8);
    do_convert_test_types<CUPHY_BIT, CUPHY_R_32F>(32, 8);
    do_convert_test_types<CUPHY_BIT, CUPHY_R_32I>(32, 8);
    do_convert_test_types<CUPHY_BIT, CUPHY_R_32U>(32, 8);
    do_convert_test_types<CUPHY_BIT, CUPHY_R_64F>(32, 8);
    do_convert_test_types<CUPHY_BIT, CUPHY_R_8F_E4M3>(1, 8);
    do_convert_test_types<CUPHY_BIT, CUPHY_R_8F_E5M2>(32, 8);
    // do_convert_test_types<CUPHY_BIT, CUPHY_VOID>(32, 8);
}

template <cuphyDataType_t TType>
void do_convert_signed_zeros_to_bits()
{
    using source_tensor_t = cuphy::typed_tensor<TType, cuphy::pinned_alloc>;
    using bit_tensor_t    = cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc>;

    source_tensor_t tSrc(32, 8);
    bit_tensor_t    tDst(32, 8, cuphy::tensor_flags::align_coalesce);

    for(int i = 0; i < 32; ++i)
    {
        tSrc(i, 0) = static_cast<typename detail::real_type<TType>::type>(1.0f);
    }
    tSrc(0, 0) = static_cast<typename detail::real_type<TType>::type>(0.0f);
    tSrc(1, 0) = static_cast<typename detail::real_type<TType>::type>(-0.0f);

    CUstream stream = nullptr;
    ASSERT_EQ(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
    cuphy::tensor_convert(tDst, tSrc, stream);
    const CUresult sync_status    = cuStreamSynchronize(stream);
    const CUresult destroy_status = cuStreamDestroy(stream);
    ASSERT_EQ(sync_status, CUDA_SUCCESS);
    ASSERT_EQ(destroy_status, CUDA_SUCCESS);

    EXPECT_EQ(tDst(0, 0) & 0x3U, 0U);
    EXPECT_EQ((tDst(0, 0) >> 2U) & 0x1U, 1U);
}

TEST(Convert, FloatingPointSignedZerosConvertToZeroBits)
{
    do_convert_signed_zeros_to_bits<CUPHY_R_16F>();
    do_convert_signed_zeros_to_bits<CUPHY_R_32F>();
    do_convert_signed_zeros_to_bits<CUPHY_R_64F>();
    do_convert_signed_zeros_to_bits<CUPHY_R_8F_E4M3>();
    do_convert_signed_zeros_to_bits<CUPHY_R_8F_E5M2>();
}

TEST(Convert, FloatingPointSignedZerosConvertToZeroVariants)
{
    auto expect_zero_bit = [](cuphyVariant_t value) {
        ASSERT_EQ(cuphyConvertVariant(&value, CUPHY_BIT), CUPHY_STATUS_SUCCESS);
        EXPECT_EQ(value.value.b1, 0U);
    };

    cuphyVariant_t half_value{};
    half_value.type       = CUPHY_R_16F;
    half_value.value.r16f = static_cast<__half_raw>(__float2half(-0.0f));
    expect_zero_bit(half_value);

    cuphyVariant_t e4m3_value{};
    e4m3_value.type            = CUPHY_R_8F_E4M3;
    e4m3_value.value.r8f_e4m3 = static_cast<__nv_fp8_e4m3>(-0.0f);
    expect_zero_bit(e4m3_value);

    cuphyVariant_t e5m2_value{};
    e5m2_value.type            = CUPHY_R_8F_E5M2;
    e5m2_value.value.r8f_e5m2 = static_cast<__nv_fp8_e5m2>(-0.0f);
    expect_zero_bit(e5m2_value);
}

// Test all conversions from CUPHY_BIT
TEST(Convert, FromBits) {
    do_convert_test_types<CUPHY_R_8I,      CUPHY_BIT>(32, 8);
    do_convert_test_types<CUPHY_R_8U,      CUPHY_BIT>(32, 8);
    do_convert_test_types<CUPHY_R_16I,     CUPHY_BIT>(32, 8);
    do_convert_test_types<CUPHY_R_16U,     CUPHY_BIT>(32, 8);
    do_convert_test_types<CUPHY_R_16F,     CUPHY_BIT>(32, 8);
    do_convert_test_types<CUPHY_R_32I,     CUPHY_BIT>(32, 8);
    do_convert_test_types<CUPHY_R_32U,     CUPHY_BIT>(32, 8);
    do_convert_test_types<CUPHY_R_32F,     CUPHY_BIT>(32, 8);
    do_convert_test_types<CUPHY_R_64F,     CUPHY_BIT>(32, 8);
    do_convert_test_types<CUPHY_R_8F_E4M3, CUPHY_BIT>(32, 8);
    do_convert_test_types<CUPHY_R_8F_E5M2, CUPHY_BIT>(32, 8);
}

// Test real number conversions
TEST(Convert, RealNumbers) {
    // 8-bit integer conversions
    do_convert_test_types<CUPHY_R_8I, CUPHY_R_8I>(32, 8);
    do_convert_test_types<CUPHY_R_8I, CUPHY_R_8I>(128, 64);  // Additional test with larger dimensions
    do_convert_test_types<CUPHY_R_8U, CUPHY_R_8U>(32, 8);
    do_convert_test_types<CUPHY_R_8U, CUPHY_R_8U>(128, 64);  // Additional test with larger dimensions

    // 16-bit integer conversions
    do_convert_test_types<CUPHY_R_16I, CUPHY_R_8I>(32, 8);   // 8-bit to 16-bit
    do_convert_test_types<CUPHY_R_16I, CUPHY_R_8I>(128, 64); // Additional test with larger dimensions
    do_convert_test_types<CUPHY_R_16I, CUPHY_R_16I>(32, 8);
    do_convert_test_types<CUPHY_R_16I, CUPHY_R_16I>(128, 64);
    do_convert_test_types<CUPHY_R_16U, CUPHY_R_8U>(32, 8);   // 8-bit to 16-bit unsigned
    do_convert_test_types<CUPHY_R_16U, CUPHY_R_16U>(32, 8);
    do_convert_test_types<CUPHY_R_16U, CUPHY_R_16U>(32, 16);

    // 8-bit floating point conversions
    do_convert_test_types<CUPHY_R_8F_E4M3, CUPHY_R_16F>(1, 8);
    do_convert_test_types<CUPHY_R_8F_E5M2, CUPHY_R_16F>(1, 8);

    // 16-bit floating point conversions
    do_convert_test_types<CUPHY_R_16F, CUPHY_R_16F>(32, 8);
    do_convert_test_types<CUPHY_R_16F, CUPHY_R_32F>(32, 64);

    // 32-bit integer conversions
    do_convert_test_types<CUPHY_R_32I, CUPHY_R_8I>(32, 8);   // 8-bit to 32-bit
    do_convert_test_types<CUPHY_R_32I, CUPHY_R_16I>(32, 8);  // 16-bit to 32-bit
    do_convert_test_types<CUPHY_R_32I, CUPHY_R_16I>(64, 32); // Additional test with different dimensions
    do_convert_test_types<CUPHY_R_32I, CUPHY_R_32I>(32, 8);
    do_convert_test_types<CUPHY_R_32I, CUPHY_R_32I>(32, 16);

    // 32-bit unsigned integer conversions
    do_convert_test_types<CUPHY_R_32U, CUPHY_R_8U>(32, 8);   // 8-bit to 32-bit unsigned
    do_convert_test_types<CUPHY_R_32U, CUPHY_R_16U>(32, 8);  // 16-bit to 32-bit unsigned
    do_convert_test_types<CUPHY_R_32U, CUPHY_R_16U>(32, 16); // Additional test with different dimensions
    do_convert_test_types<CUPHY_R_32U, CUPHY_R_32U>(32, 8);
    do_convert_test_types<CUPHY_R_32U, CUPHY_R_32U>(32, 16);

    // 32-bit floating point conversions
    do_convert_test_types<CUPHY_R_32F, CUPHY_R_16F>(32, 8);      // 16-bit to 32-bit float
    do_convert_test_types<CUPHY_R_32F, CUPHY_R_16F>(64, 64);     // Additional test with different dimensions
    do_convert_test_types<CUPHY_R_32F, CUPHY_R_32F>(32, 8);
    do_convert_test_types<CUPHY_R_32F, CUPHY_R_64F>(16, 16);     // 64-bit to 32-bit float
    do_convert_test_types<CUPHY_R_32F, CUPHY_R_8F_E4M3>(16, 16); // fp8 to 32-bit float
    do_convert_test_types<CUPHY_R_32F, CUPHY_R_8F_E5M2>(64, 64); // fp8 to 32-bit float

    // 64-bit floating point conversions
    do_convert_test_types<CUPHY_R_64F, CUPHY_R_16F>(16, 16); // 16-bit to 64-bit float
    do_convert_test_types<CUPHY_R_64F, CUPHY_R_32F>(32, 32); // 32-bit to 64-bit float
    do_convert_test_types<CUPHY_R_64F, CUPHY_R_64F>(64, 64);

}

// Test complex number conversions
TEST(Convert, ComplexNumbers) {
    // 8-bit complex conversions
    do_convert_test_complex<CUPHY_C_8I, CUPHY_C_8I>(32, 8);
    do_convert_test_complex<CUPHY_C_8U, CUPHY_C_8U>(32, 8);
    do_convert_test_complex<CUPHY_C_8I, CUPHY_C_8I>(128, 64);  // Additional test with larger dimensions

    // 16-bit complex conversions
    do_convert_test_complex<CUPHY_C_16I, CUPHY_C_8I>(32, 8);
    do_convert_test_complex<CUPHY_C_16I, CUPHY_C_16I>(32, 8);
    do_convert_test_complex<CUPHY_C_16U, CUPHY_C_8U>(32, 8);
    do_convert_test_complex<CUPHY_C_16U, CUPHY_C_16U>(32, 8);
    do_convert_test_complex<CUPHY_C_16F, CUPHY_C_16F>(32, 8);
    do_convert_test_complex<CUPHY_C_16F, CUPHY_C_32F>(32, 8);

    // 32-bit complex conversions
    do_convert_test_complex<CUPHY_C_32I, CUPHY_C_8I>(32, 8);
    do_convert_test_complex<CUPHY_C_32I, CUPHY_C_16I>(32, 8);
    do_convert_test_complex<CUPHY_C_32I, CUPHY_C_32I>(32, 8);
    do_convert_test_complex<CUPHY_C_32F, CUPHY_C_16F>(32, 8);
    do_convert_test_complex<CUPHY_C_32F, CUPHY_C_32F>(32, 8);
    do_convert_test_complex<CUPHY_C_32F, CUPHY_C_16F>(32, 32);  // Additional test with different dimensions
    do_convert_test_complex<CUPHY_C_32F, CUPHY_C_32F>(32, 32);  // Additional test with different dimensions
    do_convert_test_complex<CUPHY_C_32F, CUPHY_C_64F>(32, 32);  // Additional test with different dimensions

    // 32-bit unsigned complex conversions
    do_convert_test_complex<CUPHY_C_32U, CUPHY_C_8U>(32, 32);
    do_convert_test_complex<CUPHY_C_32U, CUPHY_C_16U>(32, 32);
    do_convert_test_complex<CUPHY_C_32U, CUPHY_C_32U>(32, 32);

    // 64-bit complex conversions
    do_convert_test_complex<CUPHY_C_64F, CUPHY_C_16F>(32, 8);
    do_convert_test_complex<CUPHY_C_64F, CUPHY_C_32F>(32, 8);
    do_convert_test_complex<CUPHY_C_64F, CUPHY_C_64F>(32, 8);
    do_convert_test_complex<CUPHY_C_64F, CUPHY_C_32F>(16, 16);  // Additional test with different dimensions
    do_convert_test_complex<CUPHY_C_64F, CUPHY_C_64F>(32, 32);
}

// Test invalid conversions between complex and real types
TEST(Convert, ErrorCases) {
    // Complex to Complex invalid conversions
    test_invalid_type_conversion<CUPHY_C_16F, CUPHY_C_16U>();  // Complex unsigned to complex float

    // Bit to Complex conversions
    test_invalid_type_conversion<CUPHY_BIT, CUPHY_C_8I>();     // Bit to complex int8

    // Real to Complex conversions
    test_invalid_type_conversion<CUPHY_R_8I, CUPHY_C_16F>();   // Real int8 to complex float16
    test_invalid_type_conversion<CUPHY_R_8U, CUPHY_C_32F>();   // Real uint8 to complex float32
    test_invalid_type_conversion<CUPHY_R_16I, CUPHY_C_8U>();   // Real int16 to complex uint8
    test_invalid_type_conversion<CUPHY_R_16U, CUPHY_C_32I>();  // Real uint16 to complex int32
    test_invalid_type_conversion<CUPHY_R_16F, CUPHY_C_32U>();  // Real float16 to complex uint32
    test_invalid_type_conversion<CUPHY_R_32I, CUPHY_C_16I>();  // Real int32 to complex int16
    test_invalid_type_conversion<CUPHY_R_32U, CUPHY_C_64F>();  // Real uint32 to complex float64
    test_invalid_type_conversion<CUPHY_R_32F, CUPHY_C_16U>();  // Real float32 to complex uint16
    test_invalid_type_conversion<CUPHY_R_64F, CUPHY_C_8I>();   // Real float64 to complex int8

    // Complex to Real conversions
    test_invalid_type_conversion<CUPHY_C_8I, CUPHY_R_32F>();   // Complex int8 to real float32
    test_invalid_type_conversion<CUPHY_C_8U, CUPHY_R_64F>();   // Complex uint8 to real float64
    test_invalid_type_conversion<CUPHY_C_16I, CUPHY_R_16F>();  // Complex int16 to real float16
    test_invalid_type_conversion<CUPHY_C_16U, CUPHY_R_32U>();  // Complex uint16 to real uint32
    test_invalid_type_conversion<CUPHY_C_16F, CUPHY_R_8U>();   // Complex float16 to real uint8
    test_invalid_type_conversion<CUPHY_C_32I, CUPHY_R_8I>();   // Complex int32 to real int8
    test_invalid_type_conversion<CUPHY_C_32U, CUPHY_R_16U>();  // Complex uint32 to real uint16
    test_invalid_type_conversion<CUPHY_C_32F, CUPHY_R_32I>();  // Complex float32 to real int32
    test_invalid_type_conversion<CUPHY_C_64F, CUPHY_R_16I>();  // Complex float64 to real int16
}

TEST(Convert, ErrorCases_invalid_type) {
    // Test invalid conversion with CUPHY_VOID type to trigger default case at line 539
    const int dims[2] = {16, 16};
    const int strides[2] = {16, 1};  // row-major layout

    tensor_desc srcDesc, dstDesc;

    // Set both descriptors to use CUPHY_VOID type to trigger default case
    srcDesc.set(CUPHY_VOID, 2, dims, strides);
    dstDesc.set(CUPHY_VOID, 2, dims, strides);

    // Allocate some dummy memory - won't be used but needed for API
    void* src_data = nullptr;
    void* dst_data = nullptr;
    EXPECT_EQ(cudaMalloc(&src_data, 16 * 16 * sizeof(float)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dst_data, 16 * 16 * sizeof(float)), cudaSuccess);

    // This should return CUPHY_STATUS_INVALID_CONVERSION since CUPHY_VOID
    // is not a supported conversion type
    EXPECT_EQ(convert_tensor_layout(dstDesc, dst_data, srcDesc, src_data, 0),
              CUPHY_STATUS_INVALID_CONVERSION);

    // Cleanup
    cudaFree(src_data);
    cudaFree(dst_data);
}

TEST(Convert, SameTypeComplex) {
    // Test complex types with different dimensions
    const int test_dims[][2] = {
        {3, 5},      // Small prime dimensions
        {16, 16},    // Power of 2 dimensions
        {13, 17},    // Larger prime dimensions
        {32, 8},     // Mixed dimensions
        {7, 11}      // More prime dimensions
    };

    for (const auto& dims : test_dims) {
        // Complex types
        do_convert_test_layout_same_type<CUPHY_C_8I>(dims[0], dims[1]);
        do_convert_test_layout_same_type<CUPHY_C_8U>(dims[0], dims[1]);
        do_convert_test_layout_same_type<CUPHY_C_16I>(dims[0], dims[1]);
        do_convert_test_layout_same_type<CUPHY_C_16U>(dims[0], dims[1]);
        do_convert_test_layout_same_type<CUPHY_C_16F>(dims[0], dims[1]);
        do_convert_test_layout_same_type<CUPHY_C_32I>(dims[0], dims[1]);
        do_convert_test_layout_same_type<CUPHY_C_32U>(dims[0], dims[1]);
        do_convert_test_layout_same_type<CUPHY_C_32F>(dims[0], dims[1]);
        do_convert_test_layout_same_type<CUPHY_C_64F>(dims[0], dims[1]);
    }
}

TEST(Convert, SameTypeReal) {
    // Test real types with different dimensions
    const int test_dims[][2] = {
        {3, 5},      // Small prime dimensions
        {16, 16},    // Power of 2 dimensions
        {13, 17},    // Larger prime dimensions
        {32, 8},     // Mixed dimensions
        {7, 11}      // More prime dimensions
    };

    for (const auto& dims : test_dims) {
        // Real types
        do_convert_test_layout_same_type<CUPHY_R_8I>(dims[0], dims[1]);
        do_convert_test_layout_same_type<CUPHY_R_8U>(dims[0], dims[1]);
        do_convert_test_layout_same_type<CUPHY_R_16I>(dims[0], dims[1]);
        do_convert_test_layout_same_type<CUPHY_R_16U>(dims[0], dims[1]);
        do_convert_test_layout_same_type<CUPHY_R_16F>(dims[0], dims[1]);
        do_convert_test_layout_same_type<CUPHY_R_32I>(dims[0], dims[1]);
        do_convert_test_layout_same_type<CUPHY_R_32U>(dims[0], dims[1]);
        do_convert_test_layout_same_type<CUPHY_R_32F>(dims[0], dims[1]);
        do_convert_test_layout_same_type<CUPHY_R_64F>(dims[0], dims[1]);
    }
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    return result;
}
