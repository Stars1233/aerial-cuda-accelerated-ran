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

// Decrease N_TEST_VALUES before enabling this...
//#define CUPHY_DEBUG 1

#include <gtest/gtest.h>
#include <random>
#include <limits>
#include <type_traits>
#include "ldpc2.hpp"
#include "ldpc2.cuh"
#include "ldpc2_c2v_x2.cuh"
#include "cuphy.hpp"
#include "ldpc2_llr_loader.cuh"
#include "ldpc/ldpc_api.hpp"

using namespace ldpc2;

////////////////////////////////////////////////////////////////////////
// kernel_loader_test()
// Kernel to instantiate a loader object that works with 2 codewords at
// a time.
template <typename T, class TLoader>
__global__
void kernel_loader_test(LDPC_kernel_params params,
                        T*                 out,
                        size_t             out_stride_elements)
{
    //typedef llr_loader_fixed<__half2, Z, V>   llr_loader_t;
    //typedef llr_loader_fixed_debug<__half2, Z, V>   llr_loader_t;
    typedef TLoader                           llr_loader_t;
    typedef typename llr_loader_t::app_buf_t  app_buf_t;
    typedef typename llr_loader_t::app_elem_t app_elem_t;

    extern __shared__ char smem[];
    //__shared__ char smem[llr_loader_t::LLR_BUFFER_SIZE];

    app_buf_t*   app_smem = reinterpret_cast<app_buf_t*>(smem);
    __half2_raw  hInf_hInf;
    hInf_hInf.x = hInf_hInf.y = 0x7C00; /* inf(fp16) */

    //------------------------------------------------------------------
    // Initialize shared memory values
    for(int i = threadIdx.x; i < params.Z_var; i += blockDim.x)
    {
        app_smem[i] = hInf_hInf;
    }
    __syncthreads();
    //------------------------------------------------------------------
    llr_loader_t loader;
    loader.load_sync(ldpc_dec_loader_params<T>(smem, params, blockIdx.x));

    T* wordOutput = out + (blockIdx.x * out_stride_elements);
    for(int i = 0; i < params.num_var_nodes; ++i)
    {
        //if(0 == threadIdx.x)
        //{
        //    float2 fCheck = __half22float2(app_smem[(i * params.Z) + threadIdx.x]);
        //    //printf("[%u]: HI = %f, LO = %f\n", (i * params.Z) + threadIdx.x, fCheck.y, fCheck.x);
        //
        //}
        wordOutput[(i * params.Z) + threadIdx.x] = app_smem[(i * params.Z) + threadIdx.x];
    }
}

////////////////////////////////////////////////////////////////////////
// Compare the output of a min-sum row context to a host reference
// implementation. Two separate codewords are represented by the hi and
// lo values in the sequence, and the row context generates a pair of
// output sequences.
TEST(LDPCInternalLoader, Loader_fp16x2)
{
    const int BG     = 1;
    const int mb     = 4; // max_parity_nodes<1>::value;
    const int Kb     = max_info_nodes<1>::value;
#if 1
    const int Z      = 384;
    const int NUM_CW = 79;
#else
    const int Z      = 32;
    const int NUM_CW = 3;
#endif
    const int N      = (mb + Kb) * Z;
    const int K      = Kb * Z;
    //const int V      = mb + Kb;
    //------------------------------------------------------------------
    // Allocate input, output, and "test" tensors. Input will be a set
    // of NUM_CW codewords, and output will be a tensor of output bits.
    // The "test" tensor output will be a set of NUM_CW/2 "paired"
    // codewords, with values from 2 codewords combined into the high
    // and low values of a fp16x2 word.
    cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc> tLLR(N, NUM_CW);
    cuphy::typed_tensor<CUPHY_C_16F, cuphy::pinned_alloc> tTest(N, (NUM_CW + 1) / 2);
    cuphy::tensor_device                                  tOut(CUPHY_BIT, K, NUM_CW);
    //------------------------------------------------------------------
    // Generate random data
    // Random number generation. Choose an interval that will not
    // overflow fp16
    //std::random_device rd;
    //std::mt19937       e2(rd());
    std::mt19937                     e2;
    std::uniform_real_distribution<> dist(-32.0f, 32.0f);
    for(int iCW = 0; iCW < NUM_CW; ++iCW)
    {
        for(int iLLR = 0; iLLR < N; ++iLLR)
        {
            tLLR(iLLR, iCW) = __float2half(dist(e2));
            //if(0 == (iLLR % Z)) printf("tLLR(%i, %i) = %f\n", idx[0], idx[1], __half2float(tLLR(idx)));
        }
    }
    //------------------------------------------------------------------
    // Initialize data structures describing the LDPC configuration
    cuphy::LDPC_decode_config config(CUPHY_R_16F, // LLR type
                                     mb,          // num parity nodes
                                     Z,           // lifting size
                                     10,          // num iterations
                                     32.0f,       // clamp value
                                     Kb,          // num info nodes
                                     1.0f,        // normalization
                                     0,           // flags
                                     BG,          // base graph
                                     0,           // algorithm index
                                     nullptr);    // workspace
    LDPC_kernel_params params(config,               // LDPC config
                              tLLR.strides()[1],    // input_stride_elem
                              tLLR.addr(),          // input_addr
                              tOut.strides()[1]/32, // output_stride_words
                              tOut.addr(),          // output_addr
                              0,                    // soft_out_stride_elem
                              nullptr,              // soft_out_addr
                              NUM_CW);
    dim3 blkDim(Z);
    dim3 grdDim((NUM_CW + 1) / 2);
#if 1
    // dynamic shared memory specification in execution configuration
    const size_t SHMEM_SIZE = sizeof(__half2) * N;
#else
    // shared memory statically allocated in kernel
    const size_t SHMEM_SIZE = 0;
#endif
    //printf("SHMEM_SIZE = %lu\n", SHMEM_SIZE);
    //typedef llr_loader_fixed<__half2, Z, mb + Kb>            llr_loader_t;
    typedef llr_loader_variable_batch<__half2, 4, llr_op_none> llr_loader_t;

    kernel_loader_test<__half2, llr_loader_t><<<grdDim, blkDim, SHMEM_SIZE>>>(params,
                                                                              tTest.addr(),
                                                                              tTest.strides()[1]);
    //------------------------------------------------------------------
    // Check for kernel errors
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if(e != cudaSuccess)
    {
        throw cuphy_i::cuda_exception(e);
    }
    //------------------------------------------------------------------
    // Compare results to source data
    for(int iCW = 0; iCW < NUM_CW; ++iCW)
    {
        for(int iLLR = 0; iLLR < N; ++iLLR)
        {
            __half  src  = tLLR(iLLR, iCW);
            __half2 out = tTest(iLLR, iCW / 2);
            // Exact comparision should be OK here - no computations
            float value = (0 == (iCW % 2)) ? __low2float(out) : __high2float(out);
            EXPECT_EQ(__half2float(src), value) << "iCW = " << iCW << ", iLLR = " << iLLR << std::endl;
            //printf("src = %f, out = %f\n",
            //        __half2float(src),
            //        (0 == (iCW % 2)) ? __low2float(out) : __high2float(out));
        }
    }
}

////////////////////////////////////////////////////////////////////////
// generate_random_scan_set()
// Populates desc, tb_num_cw, and block_idx with values from a randomly
// generated distribution of codewords for a given number of transport
// blocks (num_tb).
template <class TRand>
int generate_random_scan_set(cuphyLDPCDecodeDesc_t& desc,
                             std::vector<int32_t>&  tb_num_cw,
                             std::vector<tb_token>& block_tokens,
                             int                    num_tb,
                             TRand                  rng,
                             int                    cw_per_cta)
{
    tb_num_cw.resize(num_tb);
    desc.num_tbs = num_tb;
    for(int i = 0; i < num_tb; ++i)
    {
        tb_num_cw[i] = rng();
        //tb_num_cw[i] = 1;
        desc.llr_input[i].num_codewords = tb_num_cw[i];
        //printf("TB[%i]: %i codewords\n", i, tb_num_cw[i]);
    }
    //------------------------------------------------------------------
    // Determine the number of blocks required. This will be equal to
    // the number of codewords, unless there are multiple codewords per
    // CTA. In that case, we assume that a "batch" cannot span two TBs,
    // and that a CTA at the end of a TB may have a "partial" batch.
    int sum = 0;
    for(auto num_cw : tb_num_cw)
    {
        sum += (num_cw + cw_per_cta - 1) / cw_per_cta;
    }
    block_tokens.resize(0);
    // Assemble the values that each launched CTA should determine as
    // its token (tb + index)
    for(int i = 0; i < num_tb; ++i)
    {
        int num_CTA_per_TB = (tb_num_cw[i] + cw_per_cta - 1) / cw_per_cta;
        for(int j = 0; j < num_CTA_per_TB; ++j)
        {
            bool partial = ((j * cw_per_cta) + 1) >= tb_num_cw[i];
            if(1 == cw_per_cta)
            {
                block_tokens.push_back(to_token<1>(i, j, false));
            }
            else
            {
                //printf("ref: TB = %i, j = %i, offset = %i, partial = %s\n",
                //       i, j, j * cw_per_cta, partial ? "true" : "false");
                block_tokens.push_back(to_token<2>(i, j * cw_per_cta, partial));
            }

        }
    }
    return sum;
}

////////////////////////////////////////////////////////////////////////
// kernel_tb_scan()
// Test kernel for the device function that determines which transport
// block (and index within that transport block) a given CTA is assigned
// to. The LDPC decoder "transport block" interface uses this function.
// In contrast, the tensor interface uses a "regular" layout and has no
// need for search.
template <int CW_PER_CTA>
__global__ void kernel_tb_scan(cuphyLDPCDecodeDesc_t decode_desc,
                               tb_token*             tokens)
{
    __shared__ tb_token tok;
    warp_find_block_tb_token<CW_PER_CTA>(decode_desc, blockIdx.x, &tok);
    __syncthreads();

    if(0 == threadIdx.x)
    {
        tokens[blockIdx.x] = tok;
    }
}

////////////////////////////////////////////////////////////////////////
// LDPCInternalLoader.TransportBlockInterfaceScan
TEST(LDPCInternalLoader, TransportBlockInterfaceScan)
{
    cuphyLDPCDecodeDesc_t                  decode_desc{};   // value-init: zero all optional/debug arrays
    std::mt19937                           rd;
    std::uniform_int_distribution<int32_t> dist(1, 32);
    std::vector<int32_t>                   tb_num_cw;
    std::vector<tb_token>                  block_tokens;
    auto                                   rng = [&]() { return dist(rd); };
    std::array<int, 4>                     test_num_tbs{1,
                                                        7,
                                                        CUPHY_LDPC_DECODE_DESC_MAX_TB-1,
                                                        CUPHY_LDPC_DECODE_DESC_MAX_TB};
    std::array<int, 2>                     test_cw_per_ctas{1, 2};
    for(auto cw_per_cta : test_cw_per_ctas)
    {
        for(auto num_tbs : test_num_tbs)
        {
            //------------------------------------------------------------------
            // Generate a random set of codeword counts - 1 for each transport
            // block.
            int num_blocks = generate_random_scan_set(decode_desc,
                                                      tb_num_cw,
                                                      block_tokens,
                                                      num_tbs,
                                                      rng,
                                                      cw_per_cta);
            //printf("num_blocks = %i\n", num_blocks);
            //------------------------------------------------------------------
            // Allocate a pinned buffer that the kernel will use for output
            cuphy::buffer<tb_token, cuphy::pinned_alloc> o_tokens(num_blocks);
            //------------------------------------------------------------------
            // Launch the kernel
            if(1 == cw_per_cta)
            {
                kernel_tb_scan<1><<<num_blocks, 32>>>(decode_desc, o_tokens.addr());
            }
            else
            {
                kernel_tb_scan<2><<<num_blocks, 32>>>(decode_desc, o_tokens.addr());
            }
            cudaDeviceSynchronize();
            cudaError_t e = cudaGetLastError();
            if(e != cudaSuccess)
            {
                throw cuphy_i::cuda_exception(e);
            }
            //------------------------------------------------------------------
            // Compare results
            for(int i = 0; i < num_blocks; ++i)
            {
                EXPECT_EQ(o_tokens[i], block_tokens[i])
                    << "block = " << i << ", GPU = " << o_tokens[i]
                    << ", REF = " << block_tokens[i] << std::endl;
               //printf("TB[%3i]: 0x%08X  0x%08X : REF TB = %3i, OFFSET = %3i, IS_PARTIAL = %s\n",
               //       i,
               //       o_tokens[i],
               //       block_tokens[i],
               //       tb_from_token(o_tokens[i]),
               //       offset_from_token(o_tokens[i]),
               //       is_partial_from_token(o_tokens[i]) ? "true" : "false");
            }
        }
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
