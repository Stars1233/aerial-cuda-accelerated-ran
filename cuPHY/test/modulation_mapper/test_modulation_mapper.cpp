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

#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <algorithm>
#include <random>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <type_traits>
// Ensure host-side definitions for __half2 and cuComplex
#include <cuda_fp16.h>
#include <cuComplex.h>

#include "cuphy.h"
#include "cuphy.hpp"

namespace
{

struct ModKernelParams
{
    int   log2_qam;
    int   num_symbols;
    float beta_qam;
    bool  tensorized; // when true, exercise params->num_Rbs != 0 branch
};

// CPU reference mapper for BPSK and QAM-4/16/64/256 (log2(M) = 1/2/4/6/8)
static inline void cpu_symbol_from_bits(int      log2_qam,
                                        uint32_t bits,
                                        float    beta,
                                        float&   out_x,
                                        float&   out_y)
{
    if(log2_qam == CUPHY_QAM_4)
    {
        float a = 1.0f / std::sqrt(2.0f);
        a *= beta;
        out_x = (0 == (bits & 0x1)) ? a : -a;
        out_y = (0 == (bits & 0x2)) ? a : -a;
    }
    else if(log2_qam == CUPHY_QAM_16)
    {
        // 3GPP Gray mapping, scaled by 1/sqrt(10)
        float s  = 1.0f / std::sqrt(10.0f);
        int   b0 = (bits >> 0) & 1;
        int   b1 = (bits >> 1) & 1;
        int   b2 = (bits >> 2) & 1;
        int   b3 = (bits >> 3) & 1;
        float xr = static_cast<float>((1 - 2 * b0) * (1 + 2 * b2));
        float yi = static_cast<float>((1 - 2 * b1) * (1 + 2 * b3));
        out_x    = xr * s * beta;
        out_y    = yi * s * beta;
    }
    else if(log2_qam == CUPHY_QAM_64)
    {
        // 1/sqrt(42)
        float s  = 1.0f / std::sqrt(42.0f);
        int   b0 = (bits >> 0) & 1;
        int   b1 = (bits >> 1) & 1;
        int   b2 = (bits >> 2) & 1;
        int   b3 = (bits >> 3) & 1;
        int   b4 = (bits >> 4) & 1;
        int   b5 = (bits >> 5) & 1;
        float xr = static_cast<float>((1 - 2 * b0) * (4 - (1 - 2 * b2) * (1 + 2 * b4)));
        float yi = static_cast<float>((1 - 2 * b1) * (4 - (1 - 2 * b3) * (1 + 2 * b5)));
        out_x    = xr * s * beta;
        out_y    = yi * s * beta;
    }
    else if(log2_qam == CUPHY_QAM_256)
    {
        // 1/sqrt(170)
        float s  = 1.0f / std::sqrt(170.0f);
        int   b0 = (bits >> 0) & 1;
        int   b1 = (bits >> 1) & 1;
        int   b2 = (bits >> 2) & 1;
        int   b3 = (bits >> 3) & 1;
        int   b4 = (bits >> 4) & 1;
        int   b5 = (bits >> 5) & 1;
        int   b6 = (bits >> 6) & 1;
        int   b7 = (bits >> 7) & 1;
        float xr = static_cast<float>((1 - 2 * b0) * (8 - (1 - 2 * b2) * (4 - (1 - 2 * b4) * (1 + 2 * b6))));
        float yi = static_cast<float>((1 - 2 * b1) * (8 - (1 - 2 * b3) * (4 - (1 - 2 * b5) * (1 + 2 * b7))));
        out_x    = xr * s * beta;
        out_y    = yi * s * beta;
    }
    else if(log2_qam == 1)
    {
        // Utility kernel maps BPSK to (a, a)
        float a = 1.0f / std::sqrt(2.0f);
        a *= beta;
        out_x = (bits == 0) ? a : -a;
        out_y = out_x;
    }
    else
    {
        out_x = 0.0f;
        out_y = 0.0f;
    }
}

// Extract symbol bits for a given symbol index from a packed bitstream (flat mode)
static inline uint32_t extract_symbol_bits_flat(const std::vector<uint32_t>& words,
                                                int                          log2_qam,
                                                int                          symbol_index)
{
    const int bit_idx     = symbol_index * log2_qam;
    const int word_idx    = bit_idx / 32;
    const int word_offset = bit_idx % 32;
    uint32_t  value       = (words[word_idx] >> word_offset);
    if(log2_qam == CUPHY_QAM_64)
    {
        if(word_offset == 28)
        {
            value &= 0x0FU;
            value |= ((words[word_idx + 1] & 0x03U) << 4);
        }
        else if(word_offset == 30)
        {
            value &= 0x03U;
            value |= ((words[word_idx + 1] & 0x0FU) << 2);
        }
    }
    return value & ((1u << log2_qam) - 1u);
}

template <typename TComplex>
static inline float abs_diff_real(const TComplex& a, const TComplex& b)
{
    return std::abs(static_cast<float>(a.x) - static_cast<float>(b.x));
}

template <typename TComplex>
static inline float abs_diff_imag(const TComplex& a, const TComplex& b)
{
    return std::abs(static_cast<float>(a.y) - static_cast<float>(b.y));
}

// Launch device kernel via public API and validate against CPU reference
static void run_and_validate_kernel(const ModKernelParams& prm, int launch_max_symbols_override = -1)
{
    cudaStream_t strm = 0;

    const int   log2_qam    = prm.log2_qam;
    const int   num_symbols = prm.num_symbols;
    const float beta_qam    = prm.beta_qam;
    const int   num_TBs     = 1;
    const int   num_layers  = 1;
    const int   num_bits    = num_symbols * log2_qam;

    const int                          input_elements = (num_bits + 31) / 32 + ((log2_qam == CUPHY_QAM_64) ? 1 : 0);
    cuphy::unique_device_ptr<uint32_t> d_input        = cuphy::make_unique_device<uint32_t>(input_elements);
    std::vector<uint32_t>              h_input(input_elements);

    std::mt19937                            rng(1234);
    std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFFu);
    for(int i = 0; i < input_elements; ++i) h_input[i] = dist(rng);
    CUDA_CHECK(cudaMemcpy(d_input.get(), h_input.data(), input_elements * sizeof(uint32_t), cudaMemcpyHostToDevice));

    std::vector<PdschPerTbParams> h_workspace(num_TBs);
    for(int i = 0; i < num_TBs; ++i)
    {
        h_workspace[i]    = {};
        h_workspace[i].G  = num_bits;
        h_workspace[i].Qm = log2_qam;
    }
    cuphy::unique_device_ptr<PdschPerTbParams> d_workspace = cuphy::make_unique_device<PdschPerTbParams>(num_TBs);
    CUDA_CHECK(cudaMemcpy(d_workspace.get(), h_workspace.data(), num_TBs * sizeof(PdschPerTbParams), cudaMemcpyHostToDevice));

    std::vector<PdschDmrsParams> h_params(num_TBs);
    for(int i = 0; i < num_TBs; ++i)
    {
        h_params[i]            = {};
        h_params[i].beta_qam   = beta_qam;
        h_params[i].num_layers = num_layers;
        if(prm.tensorized)
        {
            h_params[i].num_Rbs          = 1; // minimal to exercise tensorized branch
            h_params[i].start_Rb         = 0;
            h_params[i].num_data_symbols = 1;
            h_params[i].symbol_number    = 0;
            h_params[i].num_dmrs_symbols = 0;
            h_params[i].data_sym_loc     = 0; // symbol position 0
            h_params[i].port_ids[0]      = 0;
            h_params[i].n_scid           = 0;
        }
        else
        {
            h_params[i].num_Rbs = 0; // flat mode
        }
    }
    cuphy::unique_device_ptr<PdschDmrsParams> d_params = cuphy::make_unique_device<PdschDmrsParams>(num_TBs);
    CUDA_CHECK(cudaMemcpy(d_params.get(), h_params.data(), num_TBs * sizeof(PdschDmrsParams), cudaMemcpyHostToDevice));

    cuphyTensorDescriptor_t input_desc, output_desc;
    cuphyCreateTensorDescriptor(&input_desc);
    cuphyCreateTensorDescriptor(&output_desc);
    int in_dims[1]  = {input_elements};
    int out_dims[1] = {num_symbols};
    cuphySetTensorDescriptor(input_desc, CUPHY_R_32U, 1, in_dims, nullptr, 0);
    cuphySetTensorDescriptor(output_desc, CUPHY_C_16F, 1, out_dims, nullptr, 0);

    cuphy::unique_device_ptr<__half2> d_output = cuphy::make_unique_device<__half2>(num_symbols);

    // Setup launch config and descriptors
    std::unique_ptr<cuphyModulationLaunchConfig> hLaunch   = std::make_unique<cuphyModulationLaunchConfig>();
    size_t                                       desc_size = 0, alloc_size = 0;
    cuphyStatus_t                                st = cuphyModulationGetDescrInfo(&desc_size, &alloc_size);
    ASSERT_EQ(st, CUPHY_STATUS_SUCCESS);
    cuphy::unique_pinned_ptr<uint8_t> h_desc = cuphy::make_unique_pinned<uint8_t>(desc_size);
    cuphy::unique_device_ptr<uint8_t> d_desc = cuphy::make_unique_device<uint8_t>(desc_size);

    int max_bits_per_layer = num_bits; // single layer
    // Always pass non-null params (beta_qam is read even in flat mode)
    PdschDmrsParams* pParamsDev        = d_params.get();
    const int        max_symbols_param = (launch_max_symbols_override > 0) ? launch_max_symbols_override : num_symbols;
    st                                 = cuphySetupModulation(hLaunch.get(),
                              pParamsDev,
                              input_desc,
                              d_input.get(),
                              max_symbols_param,
                              max_bits_per_layer,
                              num_TBs,
                              d_workspace.get(),
                              output_desc,
                              d_output.get(),
                              h_desc.get(),
                              d_desc.get(),
                              1,
                              strm);
    ASSERT_EQ(st, CUPHY_STATUS_SUCCESS);

    // Warm-up and run once
    ASSERT_EQ(launch_kernel(hLaunch->m_kernelNodeParams, strm), CUDA_SUCCESS);
    ASSERT_EQ(launch_kernel(hLaunch->m_kernelNodeParams, strm), CUDA_SUCCESS);
    CUDA_CHECK(cudaStreamSynchronize(strm));

    std::vector<__half2> h_output(num_symbols);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output.get(), num_symbols * sizeof(__half2), cudaMemcpyDeviceToHost));

    const float tol             = beta_qam * 1.5e-3f;
    const int   compare_symbols = (launch_max_symbols_override > 0) ? std::min(num_symbols, launch_max_symbols_override) : num_symbols;

    if(!prm.tensorized)
    {
        // Flat mode: validate with CPU bit extraction
        for(int s = 0; s < compare_symbols; ++s)
        {
            uint32_t bits = extract_symbol_bits_flat(h_input, log2_qam, s);
            float    xr = 0.0f, yi = 0.0f;
            cpu_symbol_from_bits(log2_qam, bits, beta_qam, xr, yi);
            __half2 ref;
            ref.x = __float2half(xr);
            ref.y = __float2half(yi);
            EXPECT_LE(abs_diff_real(ref, h_output[s]), tol) << "sym=" << s;
            EXPECT_LE(abs_diff_imag(ref, h_output[s]), tol) << "sym=" << s;
        }
    }
    else
    {
        // Tensorized mode: run a second pass with flat indexing (num_Rbs=0)
        std::vector<PdschDmrsParams> h_params_flat = h_params;
        for(auto& p : h_params_flat) p.num_Rbs = 0;
        cuphy::unique_device_ptr<PdschDmrsParams> d_params_flat = cuphy::make_unique_device<PdschDmrsParams>(num_TBs);
        CUDA_CHECK(cudaMemcpy(d_params_flat.get(), h_params_flat.data(), num_TBs * sizeof(PdschDmrsParams), cudaMemcpyHostToDevice));

        cuphy::unique_device_ptr<__half2>            d_output_ref = cuphy::make_unique_device<__half2>(num_symbols);
        std::unique_ptr<cuphyModulationLaunchConfig> hLaunchRef   = std::make_unique<cuphyModulationLaunchConfig>();
        // Reuse descriptors; only change params and output pointer
        cuphy::unique_pinned_ptr<uint8_t> h_desc_ref = cuphy::make_unique_pinned<uint8_t>(desc_size);
        cuphy::unique_device_ptr<uint8_t> d_desc_ref = cuphy::make_unique_device<uint8_t>(desc_size);
        cuphyStatus_t                     st2        = cuphySetupModulation(hLaunchRef.get(),
                                                 d_params_flat.get(),
                                                 input_desc,
                                                 d_input.get(),
                                                 max_symbols_param,
                                                 max_bits_per_layer,
                                                 num_TBs,
                                                 d_workspace.get(),
                                                 output_desc,
                                                 d_output_ref.get(),
                                                 h_desc_ref.get(),
                                                 d_desc_ref.get(),
                                                 1,
                                                 strm);
        ASSERT_EQ(st2, CUPHY_STATUS_SUCCESS);
        ASSERT_EQ(launch_kernel(hLaunchRef->m_kernelNodeParams, strm), CUDA_SUCCESS);
        CUDA_CHECK(cudaStreamSynchronize(strm));
        std::vector<__half2> h_output_ref(num_symbols);
        CUDA_CHECK(cudaMemcpy(h_output_ref.data(), d_output_ref.get(), num_symbols * sizeof(__half2), cudaMemcpyDeviceToHost));

        for(int s = 0; s < compare_symbols; ++s)
        {
            EXPECT_LE(abs_diff_real(h_output_ref[s], h_output[s]), tol) << "sym=" << s;
            EXPECT_LE(abs_diff_imag(h_output_ref[s], h_output[s]), tol) << "sym=" << s;
        }
    }

    EXPECT_EQ(cuphyDestroyTensorDescriptor(input_desc), CUPHY_STATUS_SUCCESS);
    EXPECT_EQ(cuphyDestroyTensorDescriptor(output_desc), CUPHY_STATUS_SUCCESS);
}

template <typename TOut>
static void run_symbol_modulate_util(int LOG2_QAM, int NUM_BITS, int NUM_COLS)
{
    const int                                                   NUM_SYMBOLS = NUM_BITS / LOG2_QAM;
    const cuphyDataType_t                                       SYM_TYPE    = cuphy::type_to_cuphy_type<TOut>::value;
    typedef cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc> tensor_bit_p;
    typedef cuphy::typed_tensor<SYM_TYPE, cuphy::pinned_alloc>  tensor_sym_p;

    const std::array<int, 2> SRC_DIMS = {{NUM_BITS, NUM_COLS}};
    const std::array<int, 2> DST_DIMS = {{NUM_SYMBOLS, NUM_COLS}};
    tensor_bit_p             tSrc(cuphy::tensor_layout(SRC_DIMS.size(), SRC_DIMS.data(), nullptr), cuphy::tensor_flags::align_coalesce);
    tensor_sym_p             tDst(cuphy::tensor_layout(DST_DIMS.size(), DST_DIMS.data(), nullptr), cuphy::tensor_flags::align_coalesce);

    // Initialize bits randomly to 0/1
    cuphy::rng rng;
    rng.uniform(tSrc, 0, 1);

    // Modulate
    cuphy::modulate_symbol(tDst, tSrc, LOG2_QAM);
    cudaStreamSynchronize(0);

    // Overloads for __half2 and cuComplex
    struct RefBuilder
    {
        static __half2 make(float xr, float yi)
        {
            __half2 v;
            v.x = __float2half(xr);
            v.y = __float2half(yi);
            return v;
        }
        static cuComplex make_c(float xr, float yi)
        {
            cuComplex v;
            v.x = xr;
            v.y = yi;
            return v;
        }
    };

    // Validate a subset by recomputing reference on host
    int      NUM_WORDS = (NUM_BITS + 31) / 32;
    uint32_t MASK      = (1u << LOG2_QAM) - 1u;
    for(int sym = 0; sym < NUM_SYMBOLS; ++sym)
    {
        int WORD_IDX    = (sym * LOG2_QAM) / 32;
        int WORD_OFFSET = (sym * LOG2_QAM) % 32;
        for(int col = 0; col < SRC_DIMS[1]; ++col)
        {
            uint32_t lo   = tSrc(WORD_IDX, col);
            uint32_t hi   = tSrc(std::min(WORD_IDX + 1, NUM_WORDS - 1), col);
            uint64_t hilo = (static_cast<uint64_t>(hi) << 32) | lo;
            uint32_t bits = static_cast<uint32_t>((hilo >> WORD_OFFSET) & MASK);
            float    xr = 0.0f, yi = 0.0f;
            cpu_symbol_from_bits(LOG2_QAM, bits, 1.0f, xr, yi);
            TOut ref;
            if constexpr(std::is_same<TOut, __half2>::value)
            {
                ref = RefBuilder::make(xr, yi);
            }
            else
            {
                ref = RefBuilder::make_c(xr, yi);
            }
            EXPECT_LE(abs_diff_real(ref, tDst(sym, col)), 1.0e-3f);
            EXPECT_LE(abs_diff_imag(ref, tDst(sym, col)), 1.0e-3f);
        }
    }
}

} // namespace

//-----------------------------------------------------------------------------
// Kernel-path tests (flat and tensorized) for QAM-4/16/64/256
//-----------------------------------------------------------------------------

TEST(ModulationMapperKernel, Flat_QAM_4_16_64_256)
{
    run_and_validate_kernel({CUPHY_QAM_4, 1024, 1.0f, false});
    run_and_validate_kernel({CUPHY_QAM_16, 512, 1.0f, false});
    run_and_validate_kernel({CUPHY_QAM_64, 256, 1.0f, false});
    run_and_validate_kernel({CUPHY_QAM_256, 256, 1.0f, false});
}

TEST(ModulationMapperKernel, Tensorized_Minimal_QAM_Orders)
{
    // Minimal symbols to hit QAM64 boundary cases (sym 5 and 10 => offsets 30, 28).
    // Tensorized results are compared against a flat-mode run to validate symbol placement.
    run_and_validate_kernel({CUPHY_QAM_4, 12, 0.75f, true});
    run_and_validate_kernel({CUPHY_QAM_16, 12, 0.5f, true});
    run_and_validate_kernel({CUPHY_QAM_64, 12, 1.0f, true});
    run_and_validate_kernel({CUPHY_QAM_256, 12, 1.25f, true});
}

// Cover tid >= num_symbols early return in device kernels by limiting max symbols at setup
TEST(ModulationMapperKernel, KernelEarlyExit_WhenGridOverlaunches)
{
    // Overlaunch threads relative to actual symbols so extra threads take the
    // device early-return path: if (tid >= num_symbols) return;
    run_and_validate_kernel({CUPHY_QAM_4, 16, 1.0f, false}, /*launch_max_symbols_override=*/8);
    run_and_validate_kernel({CUPHY_QAM_16, 16, 1.0f, true}, /*launch_max_symbols_override=*/7);
    run_and_validate_kernel({CUPHY_QAM_256, 8, 1.0f, false}, /*launch_max_symbols_override=*/32);
}

//-----------------------------------------------------------------------------
// Utility symbol_modulate tests to exercise host/device utility path
//-----------------------------------------------------------------------------

TEST(ModulationMapperUtil, ModulateSymbol_Half2_And_CuComplex)
{
    // BPSK to cover bits_to_symbol<1, TOut>::map
    run_symbol_modulate_util<__half2>(1, 1 * 1024, 7);
    run_symbol_modulate_util<cuComplex>(1, 1 * 373, 3);

    run_symbol_modulate_util<__half2>(CUPHY_QAM_4, CUPHY_QAM_4 * 1024, 3);
    run_symbol_modulate_util<__half2>(CUPHY_QAM_16, CUPHY_QAM_16 * 256, 2);
    run_symbol_modulate_util<__half2>(CUPHY_QAM_64, CUPHY_QAM_64 * 64, 1);
    run_symbol_modulate_util<__half2>(CUPHY_QAM_256, CUPHY_QAM_256 * 64, 1);

    run_symbol_modulate_util<cuComplex>(CUPHY_QAM_4, CUPHY_QAM_4 * 313, 4);
    run_symbol_modulate_util<cuComplex>(CUPHY_QAM_16, CUPHY_QAM_16 * 63, 5);
    run_symbol_modulate_util<cuComplex>(CUPHY_QAM_64, CUPHY_QAM_64 * 16, 2);
    run_symbol_modulate_util<cuComplex>(CUPHY_QAM_256, CUPHY_QAM_256 * 11, 3);
}

//-----------------------------------------------------------------------------
// Additional branch coverage for cuphySetupModulation
//-----------------------------------------------------------------------------

static void setup_invalid_args_tests()
{
    cudaStream_t strm           = 0;
    const int    num_symbols    = 8;
    const int    num_bits       = num_symbols * CUPHY_QAM_4;
    const int    input_elements = (num_bits + 31) / 32;

    cuphyTensorDescriptor_t input_desc, output_desc;
    cuphyCreateTensorDescriptor(&input_desc);
    cuphyCreateTensorDescriptor(&output_desc);
    int in_dims[1]  = {input_elements};
    int out_dims[1] = {num_symbols};
    cuphySetTensorDescriptor(input_desc, CUPHY_R_32U, 1, in_dims, nullptr, 0);
    cuphySetTensorDescriptor(output_desc, CUPHY_C_16F, 1, out_dims, nullptr, 0);

    cuphy::unique_device_ptr<uint32_t> d_input  = cuphy::make_unique_device<uint32_t>(input_elements);
    cuphy::unique_device_ptr<__half2>  d_output = cuphy::make_unique_device<__half2>(num_symbols);

    std::unique_ptr<cuphyModulationLaunchConfig> hLaunch   = std::make_unique<cuphyModulationLaunchConfig>();
    size_t                                       desc_size = 0, alloc_size = 0;
    ASSERT_EQ(cuphyModulationGetDescrInfo(&desc_size, &alloc_size), CUPHY_STATUS_SUCCESS);
    cuphy::unique_pinned_ptr<uint8_t> h_desc = cuphy::make_unique_pinned<uint8_t>(desc_size);
    cuphy::unique_device_ptr<uint8_t> d_desc = cuphy::make_unique_device<uint8_t>(desc_size);

    // Invalid: workspace == nullptr
    {
        cuphyStatus_t st = cuphySetupModulation(hLaunch.get(),
                                                nullptr, // d_params
                                                input_desc,
                                                d_input.get(),
                                                num_symbols,
                                                num_bits,
                                                1,       // num_TBs
                                                nullptr, // workspace
                                                output_desc,
                                                d_output.get(),
                                                h_desc.get(),
                                                d_desc.get(),
                                                1,
                                                strm);
        EXPECT_EQ(st, CUPHY_STATUS_INVALID_ARGUMENT);
    }

    // Invalid: modulation_output == nullptr
    {
        PdschPerTbParams tb{};
        tb.G                                                   = num_bits;
        tb.Qm                                                  = CUPHY_QAM_4;
        cuphy::unique_device_ptr<PdschPerTbParams> d_workspace = cuphy::make_unique_device<PdschPerTbParams>(1);
        CUDA_CHECK(cudaMemcpy(d_workspace.get(), &tb, sizeof(tb), cudaMemcpyHostToDevice));

        cuphyStatus_t st = cuphySetupModulation(hLaunch.get(),
                                                nullptr,
                                                input_desc,
                                                d_input.get(),
                                                num_symbols,
                                                num_bits,
                                                1,
                                                d_workspace.get(),
                                                output_desc,
                                                nullptr, // invalid output
                                                h_desc.get(),
                                                d_desc.get(),
                                                1,
                                                strm);
        EXPECT_EQ(st, CUPHY_STATUS_INVALID_ARGUMENT);
    }

    EXPECT_EQ(cuphyDestroyTensorDescriptor(input_desc), CUPHY_STATUS_SUCCESS);
    EXPECT_EQ(cuphyDestroyTensorDescriptor(output_desc), CUPHY_STATUS_SUCCESS);
}

TEST(ModulationMapperSetup, InvalidArgumentBranches)
{
    setup_invalid_args_tests();
}

TEST(ModulationMapperSetup, EnableDescAsyncCopy_Branches)
{
    // First call with enable=0 will set func (func==nullptr condition true)
    // Second call with enable=0 will skip setting func, covering the OR branch false path
    cudaStream_t strm           = 0;
    const int    log2_qam       = CUPHY_QAM_4;
    const int    num_symbols    = 64;
    const int    num_bits       = num_symbols * log2_qam;
    const int    input_elements = (num_bits + 31) / 32;

    std::vector<uint32_t>              h_input(input_elements, 0xA5A5A5A5u);
    cuphy::unique_device_ptr<uint32_t> d_input = cuphy::make_unique_device<uint32_t>(input_elements);
    CUDA_CHECK(cudaMemcpy(d_input.get(), h_input.data(), input_elements * sizeof(uint32_t), cudaMemcpyHostToDevice));

    PdschPerTbParams tb{};
    tb.G                                                   = num_bits;
    tb.Qm                                                  = log2_qam;
    cuphy::unique_device_ptr<PdschPerTbParams> d_workspace = cuphy::make_unique_device<PdschPerTbParams>(1);
    CUDA_CHECK(cudaMemcpy(d_workspace.get(), &tb, sizeof(tb), cudaMemcpyHostToDevice));

    PdschDmrsParams prm{};
    prm.beta_qam                                       = 1.0f;
    prm.num_layers                                     = 1;
    prm.num_Rbs                                        = 0;
    cuphy::unique_device_ptr<PdschDmrsParams> d_params = cuphy::make_unique_device<PdschDmrsParams>(1);
    CUDA_CHECK(cudaMemcpy(d_params.get(), &prm, sizeof(prm), cudaMemcpyHostToDevice));

    cuphyTensorDescriptor_t input_desc, output_desc;
    cuphyCreateTensorDescriptor(&input_desc);
    cuphyCreateTensorDescriptor(&output_desc);
    int in_dims[1]  = {input_elements};
    int out_dims[1] = {num_symbols};
    cuphySetTensorDescriptor(input_desc, CUPHY_R_32U, 1, in_dims, nullptr, 0);
    cuphySetTensorDescriptor(output_desc, CUPHY_C_16F, 1, out_dims, nullptr, 0);

    cuphy::unique_device_ptr<__half2> d_output = cuphy::make_unique_device<__half2>(num_symbols);

    std::unique_ptr<cuphyModulationLaunchConfig> hLaunch1  = std::make_unique<cuphyModulationLaunchConfig>();
    std::unique_ptr<cuphyModulationLaunchConfig> hLaunch2  = std::make_unique<cuphyModulationLaunchConfig>();
    size_t                                       desc_size = 0, alloc_size = 0;
    ASSERT_EQ(cuphyModulationGetDescrInfo(&desc_size, &alloc_size), CUPHY_STATUS_SUCCESS);
    cuphy::unique_pinned_ptr<uint8_t> h_desc1 = cuphy::make_unique_pinned<uint8_t>(desc_size);
    cuphy::unique_device_ptr<uint8_t> d_desc1 = cuphy::make_unique_device<uint8_t>(desc_size);
    cuphy::unique_pinned_ptr<uint8_t> h_desc2 = cuphy::make_unique_pinned<uint8_t>(desc_size);
    cuphy::unique_device_ptr<uint8_t> d_desc2 = cuphy::make_unique_device<uint8_t>(desc_size);

    cuphyStatus_t st1 = cuphySetupModulation(hLaunch1.get(),
                                             d_params.get(),
                                             input_desc,
                                             d_input.get(),
                                             num_symbols,
                                             num_bits,
                                             1,
                                             d_workspace.get(),
                                             output_desc,
                                             d_output.get(),
                                             h_desc1.get(),
                                             d_desc1.get(),
                                             0, // enable_desc_async_copy = 0
                                             strm);
    ASSERT_EQ(st1, CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(launch_kernel(hLaunch1->m_kernelNodeParams, strm), CUDA_SUCCESS);

    cuphyStatus_t st2 = cuphySetupModulation(hLaunch2.get(),
                                             d_params.get(),
                                             input_desc,
                                             d_input.get(),
                                             num_symbols,
                                             num_bits,
                                             1,
                                             d_workspace.get(),
                                             output_desc,
                                             d_output.get(),
                                             h_desc2.get(),
                                             d_desc2.get(),
                                             0, // second call still 0, func already set
                                             strm);
    ASSERT_EQ(st2, CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(launch_kernel(hLaunch2->m_kernelNodeParams, strm), CUDA_SUCCESS);
    CUDA_CHECK(cudaStreamSynchronize(strm));

    EXPECT_EQ(cuphyDestroyTensorDescriptor(input_desc), CUPHY_STATUS_SUCCESS);
    EXPECT_EQ(cuphyDestroyTensorDescriptor(output_desc), CUPHY_STATUS_SUCCESS);
}

TEST(ModulationMapperUtil, SymbolModulate_UnsupportedType_ReturnsError)
{
    const int LOG2_QAM    = CUPHY_QAM_4;
    const int NUM_BITS    = LOG2_QAM * 128;
    const int NUM_COLS    = 3;
    const int NUM_SYMBOLS = NUM_BITS / LOG2_QAM;
    const int NUM_WORDS   = (NUM_BITS + 31) / 32;

    cuphy::buffer<uint32_t, cuphy::device_alloc> dBits(NUM_WORDS * NUM_COLS);
    cuphy::buffer<float, cuphy::device_alloc>    dSym(NUM_SYMBOLS * NUM_COLS);

    cuphyTensorDescriptor_t tBits = nullptr;
    cuphyTensorDescriptor_t tSym  = nullptr;
    ASSERT_EQ(cuphyCreateTensorDescriptor(&tBits), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphyCreateTensorDescriptor(&tSym), CUPHY_STATUS_SUCCESS);

    int src_dims[2] = {NUM_BITS, NUM_COLS};
    int dst_dims[2] = {NUM_SYMBOLS, NUM_COLS};
    ASSERT_EQ(cuphySetTensorDescriptor(tBits, CUPHY_BIT, 2, src_dims, nullptr, 0), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphySetTensorDescriptor(tSym, CUPHY_R_32F, 2, dst_dims, nullptr, 0), CUPHY_STATUS_SUCCESS);

    cuphyStatus_t st = cuphyModulateSymbol(tSym, dSym.addr(), tBits, dBits.addr(), LOG2_QAM, 0);
    EXPECT_EQ(st, CUPHY_STATUS_UNSUPPORTED_TYPE);

    EXPECT_EQ(cuphyDestroyTensorDescriptor(tBits), CUPHY_STATUS_SUCCESS);
    EXPECT_EQ(cuphyDestroyTensorDescriptor(tSym), CUPHY_STATUS_SUCCESS);
}

//-----------------------------------------------------------------------------
// Exercise public API validation for invalid modulation order
//-----------------------------------------------------------------------------
TEST(ModulationMapperUtil, SymbolModulate_InvalidLogQam_ReturnsError)
{
    const int LOG2_QAM    = 3;   // invalid per public API
    const int NUM_BITS    = 96;
    const int NUM_COLS    = 2;
    const int NUM_SYMBOLS = 32;
    const int NUM_WORDS   = (NUM_BITS + 31) / 32;

    cuphy::buffer<uint32_t, cuphy::device_alloc> dBits(NUM_WORDS * NUM_COLS);
    cuphy::buffer<__half2, cuphy::device_alloc>  dSym(NUM_SYMBOLS * NUM_COLS);

    cuphyTensorDescriptor_t tBits = nullptr;
    cuphyTensorDescriptor_t tSym  = nullptr;
    ASSERT_EQ(cuphyCreateTensorDescriptor(&tBits), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphyCreateTensorDescriptor(&tSym), CUPHY_STATUS_SUCCESS);

    int src_dims[2] = {NUM_BITS, NUM_COLS};
    int dst_dims[2] = {NUM_SYMBOLS, NUM_COLS};
    ASSERT_EQ(cuphySetTensorDescriptor(tBits, CUPHY_BIT, 2, src_dims, nullptr, 0), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphySetTensorDescriptor(tSym, CUPHY_C_16F, 2, dst_dims, nullptr, 0), CUPHY_STATUS_SUCCESS);

    cuphyStatus_t st = cuphyModulateSymbol(tSym, dSym.addr(), tBits, dBits.addr(), LOG2_QAM, 0);
    EXPECT_EQ(st, CUPHY_STATUS_INVALID_ARGUMENT);

    EXPECT_EQ(cuphyDestroyTensorDescriptor(tBits), CUPHY_STATUS_SUCCESS);
    EXPECT_EQ(cuphyDestroyTensorDescriptor(tSym), CUPHY_STATUS_SUCCESS);
}

//-----------------------------------------------------------------------------
// Main with VectorCAST coverage dump
//-----------------------------------------------------------------------------

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    return result;
}
