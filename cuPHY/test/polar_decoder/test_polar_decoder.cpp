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
#include <cuda.h> // for cuLaunchKernel / CUresult
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuphy_api.h"
#include "cuphy.h"
#include <vector>
#include <memory>
#include <string>
#include <cstring>  // for memset
#include <cstdlib>

// Define CUDA_CHECK macro for error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return CUPHY_STATUS_INTERNAL_ERROR; \
        } \
    } while(0)

// Test configuration structure
struct PolarTestConfig {
    const char* description;
    uint16_t nPolCws;
    uint8_t nPolLists;
    uint16_t N_cw;
    uint16_t A_cw;
    uint8_t nCrcBits;
    uint8_t nCbsInUciSeg;
    bool enableAsync;
    bool zeroInsertFlag;
    uint8_t exitFlag;
    uint8_t en_CrcStatus;
    cuphyStatus_t expectedStatus;
};

// LLR pattern types
enum class LLRPattern {
    REALISTIC,      // Mix of positive/negative realistic values
    WEAK,          // Very weak values to trigger list decoder
    STRONG,        // Strong values for reliable decoding
    ALTERNATING,   // Alternating strong/weak pattern
    ULTRA_WEAK,    // Extremely weak values
    CORRUPTED      // Pseudo-random weak values
};

// Tree type patterns
enum class TreePattern {
    STANDARD,      // Standard frozen/info/mixed pattern
    COMPLEX,       // Complex mix with all types
    INFO_HEAVY,    // Many info bits
    MINIMAL,       // Minimal pattern for small sizes
    ALTERNATING    // Alternating pattern
};

// Test fixture for Polar Decoder testing
class PolarDecoderTest : public ::testing::Test {
protected:
    // cuPHY polar decoder indexes pCwTreeTypes like a binary tree array:
    // node_idx = (1 << (n - stage)) + sub_idx, where n = floor(log2(N_cw)).
    // NOTE: in `singlePolarDecoder()` the code decrements `stage` before first `get_type()`,
    // so the top-most query uses `stage = n-1`, which maps to `node_idx = 2`.
    // This implies indices [2 .. 2*N_tree-1] are used (0 and 1 are unused),
    // with leaves at [N_tree .. 2*N_tree-1], where N_tree = 1<<n.
    static uint16_t ComputeTreeLeafCount(uint16_t N_cw) {
        if (N_cw <= 1) return 1;
        uint16_t n = 0;
        uint16_t tmp = N_cw;
        while (tmp > 1) {
            tmp >>= 1;
            ++n;
        }
        return static_cast<uint16_t>(1u << n);
    }

    void SetUp() override {
        // Initialize CUDA
        int deviceCount;
        ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&deviceCount));
        ASSERT_GT(deviceCount, 0);
        ASSERT_EQ(cudaSuccess, cudaSetDevice(0));

        // Create CUDA stream
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&cuStream));

        // Get descriptor sizes
        ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphyPolarDecoderGetDescrInfo(&dynDescrSizeBytes, &dynDescrAlignBytes));
        
        // Allocate buffers for descriptors
        ASSERT_EQ(cudaSuccess, cudaMallocHost(&dynDescrBufCpu, dynDescrSizeBytes));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&dynDescrBufGpu, dynDescrSizeBytes));

        // Create polar decoder handle
        ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphyCreatePolarDecoder(&polarDecoderHndl));

        // Initialize pointers to nullptr
        ResetTestData();
    }

    static inline uint8_t GetPackedBit(const uint8_t* bytes, uint32_t bitIdx)
    {
        return static_cast<uint8_t>((bytes[bitIdx >> 3] >> (bitIdx & 7u)) & 1u);
    }

    static inline uint8_t GetPackedBitMsb(const uint8_t* bytes, uint32_t bitIdx)
    {
        const uint32_t byteIdx = bitIdx >> 3;
        const uint32_t bitInByte = bitIdx & 7u;
        return static_cast<uint8_t>((bytes[byteIdx] >> (7u - bitInByte)) & 1u);
    }

    static inline void SetPackedBit(uint8_t* bytes, uint32_t bitIdx, uint8_t bit)
    {
        const uint32_t byteIdx = bitIdx >> 3;
        const uint32_t shift   = bitIdx & 7u;
        const uint8_t  mask    = static_cast<uint8_t>(1u << shift);
        if(bit) bytes[byteIdx] = static_cast<uint8_t>(bytes[byteIdx] | mask);
        else bytes[byteIdx] = static_cast<uint8_t>(bytes[byteIdx] & ~mask);
    }

    static inline uint8_t GetWordBit(const uint32_t* words, uint32_t bitIdx)
    {
        return static_cast<uint8_t>((words[bitIdx >> 5] >> (bitIdx & 31u)) & 1u);
    }

    static inline uint8_t GetWordBitMsb(const uint32_t* words, uint32_t bitIdx)
    {
        const uint32_t wordIdx = bitIdx >> 5;
        const uint32_t bitInWord = bitIdx & 31u;
        return static_cast<uint8_t>((words[wordIdx] >> (31u - bitInWord)) & 1u);
    }

    void TearDown() override {
        FreeResources();
        if(cuStream) {
            cudaStreamSynchronize(cuStream);
            cudaStreamDestroy(cuStream);
            cuStream = nullptr;
        }
        cudaDeviceSynchronize();
    }

    // Reset all test data pointers
    void ResetTestData() {
        cwPrmsCpu = nullptr;
        cwPrmsGpu = nullptr;
        cwTreeLLRsAddrs = nullptr;
        polCbEstAddrs = nullptr;
        listPolScratchAddrs = nullptr;
        polCrcErrorFlags = nullptr;
    }

    // Generate LLR values based on pattern type
    std::vector<__half> GenerateLLRs(size_t size, LLRPattern pattern, int seed = 0) {
        std::vector<__half> llrs(size);
        
        switch(pattern) {
            case LLRPattern::REALISTIC:
                for(size_t i = 0; i < size; i++) {
                    float val = (i % 2 == 0) ? 2.5f : -1.8f;
                    llrs[i] = __float2half(val);
                }
                break;

            case LLRPattern::WEAK:
                for(size_t i = 0; i < size; i++) {
                    float val = ((i + seed) % 3 == 0) ? 0.1f : -0.1f;
                    llrs[i] = __float2half(val);
                }
                break;

            case LLRPattern::STRONG:
                for(size_t i = 0; i < size; i++) {
                    float val = (i % 2 == 0) ? 5.0f : -5.0f;
                    llrs[i] = __float2half(val);
                }
                break;

            case LLRPattern::ALTERNATING:
                for(size_t i = 0; i < size; i++) {
                    if(i % 4 < 2) {
                        float val = (i % 2 == 0) ? 5.0f : -5.0f;  // Strong
                        llrs[i] = __float2half(val);
                    } else {
                        float val = (i % 2 == 0) ? 0.2f : -0.2f;  // Weak
                        llrs[i] = __float2half(val);
                    }
                }
                break;

            case LLRPattern::ULTRA_WEAK:
                for(size_t i = 0; i < size; i++) {
                    float val = ((i * 7 + seed) % 13 < 6) ? 0.01f : -0.01f;
                    llrs[i] = __float2half(val);
                }
                break;

            case LLRPattern::CORRUPTED:
                for(size_t i = 0; i < size; i++) {
                    float val = ((i * 17 + seed) % 7 == 0) ? 0.05f : -0.05f;
                    llrs[i] = __float2half(val);
                }
                break;
        }

        return llrs;
    }

    // Generate tree types based on pattern
    std::vector<uint8_t> GenerateTreeTypes(size_t size, TreePattern pattern) {
        // 'size' is passed as N_cw in this test. The device code uses a power-of-two
        // leaf count derived from floor(log2(N_cw)).
        const uint16_t N_cw = static_cast<uint16_t>(size);
        const uint16_t N_tree = ComputeTreeLeafCount(N_cw);

        // Full tree array layout: indices [2 .. 2*N_tree-1] used, 0 and 1 unused.
        std::vector<uint8_t> types(2u * static_cast<size_t>(N_tree), 0);

        // Build leaf types at indices [N_tree .. 2*N_tree-1].
        // Keep leaves mostly {0,1}. Internal nodes are derived bottom-up to be consistent.
        switch(pattern) {
            case TreePattern::STANDARD:
                for(uint16_t i = 0; i < N_tree; i++) {
                    if(i < N_tree/4) types[N_tree + i] = 0;           // frozen
                    else if(i < (3*N_tree)/4) types[N_tree + i] = 1;  // info
                    else types[N_tree + i] = static_cast<uint8_t>((i & 1) ? 0 : 1); // mixed tail
                }
                break;

            case TreePattern::COMPLEX:
                for(uint16_t i = 0; i < N_tree; i++) {
                    // leaves: 0/1 mix, with occasional 1 to create diversity
                    if(i < N_tree/8) types[N_tree + i] = 0;
                    else if(i < N_tree/4) types[N_tree + i] = 1;
                    else types[N_tree + i] = static_cast<uint8_t>((i * 3u) & 1u);
                }
                break;

            case TreePattern::INFO_HEAVY:
                for(uint16_t i = 0; i < N_tree; i++) {
                    if(i < N_tree/8) types[N_tree + i] = 0;
                    else if(i < (7*N_tree)/8) types[N_tree + i] = 1;
                    else types[N_tree + i] = static_cast<uint8_t>((i & 1) ? 0 : 1);
                }
                break;

            case TreePattern::MINIMAL:
                for(uint16_t i = 0; i < N_tree; i++) {
                    if(i < N_tree/3) types[N_tree + i] = 0;
                    else if(i < (2*N_tree)/3) types[N_tree + i] = 1;
                    else types[N_tree + i] = static_cast<uint8_t>((i & 1) ? 0 : 1);
                }
                break;

            case TreePattern::ALTERNATING:
                // This pattern is intentionally constructed so that:
                // - leaf 0 is an info bit (type 1), enabling the (type==1 && stage==0) path
                // - all ancestors remain mixed (type 3), preventing early pruning at stage>0
                //   and ensuring we reach stage 0 during traversal.
                for(uint16_t i = 0; i < N_tree; i++) {
                    types[N_tree + i] = static_cast<uint8_t>((i & 1) ? 0 : 1); // 1,0,1,0,...
                }
                break;
        }

        // Derive internal node types bottom-up:
        // 0 if both children are 0, 1 if both children are 1, otherwise 3 (mixed).
        for(int32_t node = static_cast<int32_t>(N_tree) - 1; node >= 2; --node) {
            const uint8_t l = types[static_cast<size_t>(2 * node)];
            const uint8_t r = types[static_cast<size_t>(2 * node + 1)];
            types[static_cast<size_t>(node)] = (l == r && (l == 0 || l == 1)) ? l : 3;
        }

        return types;
    }

    // Setup test configuration with specified parameters
    void SetupTestConfiguration(const PolarTestConfig& config) {
        FreeTestData();

        // Allocate memory for codeword parameters
        ASSERT_EQ(cudaSuccess, cudaMallocHost(&cwPrmsCpu, config.nPolCws * sizeof(cuphyPolarCwPrm_t)));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&cwPrmsGpu, config.nPolCws * sizeof(cuphyPolarCwPrm_t)));

        // Allocate memory for LLR addresses
        ASSERT_EQ(cudaSuccess, cudaMallocHost(&cwTreeLLRsAddrs, config.nPolCws * sizeof(__half*)));
        
        // Allocate memory for estimated codeblocks
        ASSERT_EQ(cudaSuccess, cudaMallocHost(&polCbEstAddrs, config.nPolCws * sizeof(uint32_t*)));

        // Allocate memory for list polar scratch (only if list size > 1)
        if(config.nPolLists > 1) {
            ASSERT_EQ(cudaSuccess, cudaMallocHost(&listPolScratchAddrs, config.nPolCws * sizeof(bool*)));
        }

        // Allocate memory for CRC error flags
        ASSERT_EQ(cudaSuccess, cudaMalloc(&polCrcErrorFlags, config.nPolCws * sizeof(uint8_t)));
        ASSERT_EQ(cudaSuccess, cudaMemset(polCrcErrorFlags, 0, config.nPolCws * sizeof(uint8_t)));

        // Initialize each codeword
        for(uint16_t i = 0; i < config.nPolCws; i++) {
            SetupSingleCodeword(i, config);
        }

        // Copy initialized parameters to GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpy(cwPrmsGpu, cwPrmsCpu, config.nPolCws * sizeof(cuphyPolarCwPrm_t), cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    }

    // Setup a single codeword with specified configuration
    void SetupSingleCodeword(uint16_t cwIdx, const PolarTestConfig& config) {
        // Clear the structure
        memset(&cwPrmsCpu[cwIdx], 0, sizeof(cuphyPolarCwPrm_t));

        // Set basic parameters
        cwPrmsCpu[cwIdx].N_cw = config.N_cw;
        cwPrmsCpu[cwIdx].nCrcBits = config.nCrcBits;
        cwPrmsCpu[cwIdx].A_cw = config.A_cw;
        cwPrmsCpu[cwIdx].exitFlag = config.exitFlag;
        cwPrmsCpu[cwIdx].nCbsInUciSeg = config.nCbsInUciSeg;
        cwPrmsCpu[cwIdx].cbIdxWithinUciSeg = cwIdx;
        cwPrmsCpu[cwIdx].zeroInsertFlag = config.zeroInsertFlag ? (cwIdx == 0 ? 1 : 0) : 0;
        cwPrmsCpu[cwIdx].en_CrcStatus = config.en_CrcStatus;

        // Allocate and setup LLR buffer
        size_t llrSize = 2 * config.N_cw;
        ASSERT_EQ(cudaSuccess, cudaMalloc(&cwTreeLLRsAddrs[cwIdx], llrSize * sizeof(__half)));

        // Allocate estimation buffer
        size_t estSize = (config.A_cw + 31) / 32; // Round up to words
        ASSERT_EQ(cudaSuccess, cudaMalloc(&polCbEstAddrs[cwIdx], estSize * sizeof(uint32_t)));
        ASSERT_EQ(cudaSuccess, cudaMemset(polCbEstAddrs[cwIdx], 0, estSize * sizeof(uint32_t)));

        // Allocate list scratch if needed
        if(listPolScratchAddrs && config.nPolLists > 1) {
            ASSERT_EQ(cudaSuccess, cudaMalloc(&listPolScratchAddrs[cwIdx], 8192 * sizeof(bool)));
            ASSERT_EQ(cudaSuccess, cudaMemset(listPolScratchAddrs[cwIdx], 0, 8192 * sizeof(bool)));
        }

        // Set pointers
        cwPrmsCpu[cwIdx].pCwTreeLLRs = cwTreeLLRsAddrs[cwIdx];
        cwPrmsCpu[cwIdx].pCwLLRs = cwTreeLLRsAddrs[cwIdx];
        cwPrmsCpu[cwIdx].pCbEst = polCbEstAddrs[cwIdx];

        // Allocate CRC status buffer
        uint8_t* tempCrcStatus;
        ASSERT_EQ(cudaSuccess, cudaMalloc(&tempCrcStatus, sizeof(uint8_t)));
        ASSERT_EQ(cudaSuccess, cudaMemset(tempCrcStatus, 0, sizeof(uint8_t)));
        cwPrmsCpu[cwIdx].pCrcStatus = tempCrcStatus;

        // Allocate CRC status 1 buffer if needed
        if(config.en_CrcStatus & 0x80) {
            uint8_t* tempCrcStatus1;
            ASSERT_EQ(cudaSuccess, cudaMalloc(&tempCrcStatus1, sizeof(uint8_t)));
            ASSERT_EQ(cudaSuccess, cudaMemset(tempCrcStatus1, 0, sizeof(uint8_t)));
            cwPrmsCpu[cwIdx].pCrcStatus1 = tempCrcStatus1;
        } else {
            cwPrmsCpu[cwIdx].pCrcStatus1 = nullptr;
        }

        // Allocate UCI segment estimation buffer if needed
        if(config.nCbsInUciSeg > 1) {
            uint32_t* tempUciSegEst;
            size_t uciSize = config.nCbsInUciSeg * estSize;
            ASSERT_EQ(cudaSuccess, cudaMalloc(&tempUciSegEst, uciSize * sizeof(uint32_t)));
            ASSERT_EQ(cudaSuccess, cudaMemset(tempUciSegEst, 0, uciSize * sizeof(uint32_t)));
            cwPrmsCpu[cwIdx].pUciSegEst = tempUciSegEst;
        } else {
            cwPrmsCpu[cwIdx].pUciSegEst = nullptr;
        }

        // Allocate and setup tree types
        uint8_t* tempTreeTypes;
        auto hostTreeTypes = GenerateTreeTypes(config.N_cw, TreePattern::STANDARD);
        ASSERT_EQ(cudaSuccess, cudaMalloc(&tempTreeTypes, hostTreeTypes.size() * sizeof(uint8_t)));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(tempTreeTypes, hostTreeTypes.data(), hostTreeTypes.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
        cwPrmsCpu[cwIdx].pCwTreeTypes = tempTreeTypes;
    }

    // Setup LLRs with specified pattern
    void SetupLLRs(const PolarTestConfig& config, LLRPattern pattern, int seed = 0) {
        for(uint16_t i = 0; i < config.nPolCws; i++) {
            size_t llrSize = 2 * config.N_cw;
            auto hostLLRs = GenerateLLRs(llrSize, pattern, seed + i);
            ASSERT_EQ(cudaSuccess, cudaMemcpy(cwTreeLLRsAddrs[i], hostLLRs.data(), llrSize * sizeof(__half), cudaMemcpyHostToDevice));
        }
    }

    // Setup tree types with specified pattern
    void SetupTreeTypes(const PolarTestConfig& config, TreePattern pattern) {
        for(uint16_t i = 0; i < config.nPolCws; i++) {
            auto hostTreeTypes = GenerateTreeTypes(config.N_cw, pattern);
            ASSERT_EQ(cudaSuccess, cudaMemcpy((void*)cwPrmsCpu[i].pCwTreeTypes, hostTreeTypes.data(), 
                hostTreeTypes.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
        }
    }

    // Run polar decoder with current configuration
    cuphyStatus_t RunPolarDecoder(const PolarTestConfig& config) {
        cuphyPolarDecoderLaunchCfg_t launchCfg = {};

        cuphyStatus_t status = cuphySetupPolarDecoder(
            polarDecoderHndl,
            config.nPolCws,
            cwTreeLLRsAddrs,
            cwPrmsGpu,
            cwPrmsCpu,
            polCbEstAddrs,
            listPolScratchAddrs,
            config.nPolLists,
            polCrcErrorFlags,
            config.enableAsync,
            dynDescrBufCpu,
            dynDescrBufGpu,
            cwTreeLLRsAddrs,
            polCbEstAddrs,
            listPolScratchAddrs,
            &launchCfg,
            cuStream);

        if(status == CUPHY_STATUS_SUCCESS) {
            if(!config.enableAsync) {
                CUDA_CHECK(cudaMemcpyAsync(cwPrmsGpu, cwPrmsCpu, 
                    config.nPolCws * sizeof(cuphyPolarCwPrm_t), 
                    cudaMemcpyHostToDevice, cuStream));
                
                CUDA_CHECK(cudaMemcpyAsync(dynDescrBufGpu, dynDescrBufCpu, 
                    dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStream));

                CUDA_CHECK(cudaStreamSynchronize(cuStream));
            }

            if(launchCfg.kernelNodeParamsDriver.func != nullptr) {
                CUresult runStatus = cuLaunchKernel(
                    launchCfg.kernelNodeParamsDriver.func,
                    launchCfg.kernelNodeParamsDriver.gridDimX,
                    launchCfg.kernelNodeParamsDriver.gridDimY,
                    launchCfg.kernelNodeParamsDriver.gridDimZ,
                    launchCfg.kernelNodeParamsDriver.blockDimX,
                    launchCfg.kernelNodeParamsDriver.blockDimY,
                    launchCfg.kernelNodeParamsDriver.blockDimZ,
                    launchCfg.kernelNodeParamsDriver.sharedMemBytes,
                    static_cast<CUstream>(cuStream),
                    launchCfg.kernelNodeParamsDriver.kernelParams,
                    launchCfg.kernelNodeParamsDriver.extra);

                if(runStatus != CUDA_SUCCESS) {
                    const char* errorStr;
                    cuGetErrorString(runStatus, &errorStr);
                    fprintf(stderr, "CUDA driver error in kernel launch: %s\n", errorStr);
                    return CUPHY_STATUS_INTERNAL_ERROR;
                }

                CUDA_CHECK(cudaStreamSynchronize(cuStream));
            }
        }

        return status;
    }

    void VerifyOutputs(const PolarTestConfig& config)
    {
        // Basic sanity invariants on decoded output + CRC flag buffers.
        // This does not assume any particular correctness for synthetic LLR/tree patterns,
        // but it ensures we read back results and that buffers are consistent/clean.

        std::vector<uint8_t> crcErr(config.nPolCws, 0);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(crcErr.data(), polCrcErrorFlags, config.nPolCws * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        for(uint16_t cw = 0; cw < config.nPolCws; ++cw)
        {
            EXPECT_TRUE(crcErr[cw] == 0u || crcErr[cw] == 1u) << "CRC error flag not boolean at cw=" << cw;
        }

        for(uint16_t cw = 0; cw < config.nPolCws; ++cw)
        {
            const uint32_t nWords = static_cast<uint32_t>((config.A_cw + 31u) / 32u);
            std::vector<uint32_t> est(nWords, 0xFFFFFFFFu);
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy(est.data(), polCbEstAddrs[cw], nWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));

            // For exitFlag tests, the kernel should early-return and leave outputs unchanged (we memset to 0).
            if(config.exitFlag != 0)
            {
                for(uint32_t w = 0; w < nWords; ++w)
                {
                    EXPECT_EQ(0u, est[w]) << "exitFlag set but output changed, cw=" << cw << " word=" << w;
                }
            }

            // CRC status bytes should remain within a small range when enabled.
            if(config.en_CrcStatus != 0 && cwPrmsCpu && cwPrmsCpu[cw].pCrcStatus)
            {
                uint8_t st = 0xFF;
                ASSERT_EQ(cudaSuccess, cudaMemcpy(&st, cwPrmsCpu[cw].pCrcStatus, sizeof(uint8_t), cudaMemcpyDeviceToHost));
                EXPECT_LE(st, 3u) << "Unexpected CRC status value, cw=" << cw;
            }
            if(cwPrmsCpu && cwPrmsCpu[cw].pCrcStatus1)
            {
                uint8_t st1 = 0xFF;
                ASSERT_EQ(cudaSuccess, cudaMemcpy(&st1, cwPrmsCpu[cw].pCrcStatus1, sizeof(uint8_t), cudaMemcpyDeviceToHost));
                EXPECT_LE(st1, 3u) << "Unexpected CRC status1 value, cw=" << cw;
            }
        }
    }

    // Test a configuration and verify result
    void TestConfiguration(const PolarTestConfig& config, LLRPattern llrPattern = LLRPattern::REALISTIC, 
                          TreePattern treePattern = TreePattern::STANDARD) {
        printf("Testing: %s\n", config.description);
        
        SetupTestConfiguration(config);
        SetupLLRs(config, llrPattern);
        if(treePattern != TreePattern::STANDARD) {
            SetupTreeTypes(config, treePattern);
        }

        cuphyStatus_t status = RunPolarDecoder(config);
        EXPECT_EQ(config.expectedStatus, status) << "Failed for: " << config.description;

        if(status == CUPHY_STATUS_SUCCESS)
        {
            VerifyOutputs(config);
        }

        FreeTestData();
    }

    void FreeTestData() {
        if(cwTreeLLRsAddrs) {
            for(uint16_t i = 0; i < maxPolCws; i++) {
                if(cwTreeLLRsAddrs[i]) {
                    cudaFree(cwTreeLLRsAddrs[i]);
                    cwTreeLLRsAddrs[i] = nullptr;
                }
            }
            cudaFreeHost(cwTreeLLRsAddrs);
            cwTreeLLRsAddrs = nullptr;
        }

        if(polCbEstAddrs) {
            for(uint16_t i = 0; i < maxPolCws; i++) {
                if(polCbEstAddrs[i]) {
                    cudaFree(polCbEstAddrs[i]);
                    polCbEstAddrs[i] = nullptr;
                }
            }
            cudaFreeHost(polCbEstAddrs);
            polCbEstAddrs = nullptr;
        }

        if(listPolScratchAddrs) {
            for(uint16_t i = 0; i < maxPolCws; i++) {
                if(listPolScratchAddrs[i]) {
                    cudaFree(listPolScratchAddrs[i]);
                    listPolScratchAddrs[i] = nullptr;
                }
            }
            cudaFreeHost(listPolScratchAddrs);
            listPolScratchAddrs = nullptr;
        }

        if(polCrcErrorFlags) {
            cudaFree(polCrcErrorFlags);
            polCrcErrorFlags = nullptr;
        }

        if(cwPrmsCpu) {
            // Free individual allocated buffers
            for(uint16_t i = 0; i < maxPolCws; i++) {
                if(cwPrmsCpu[i].pCrcStatus) cudaFree((void*)cwPrmsCpu[i].pCrcStatus);
                if(cwPrmsCpu[i].pCrcStatus1) cudaFree((void*)cwPrmsCpu[i].pCrcStatus1);
                if(cwPrmsCpu[i].pUciSegEst) cudaFree((void*)cwPrmsCpu[i].pUciSegEst);
                if(cwPrmsCpu[i].pCwTreeTypes) cudaFree((void*)cwPrmsCpu[i].pCwTreeTypes);
            }
            cudaFreeHost(cwPrmsCpu);
            cwPrmsCpu = nullptr;
        }

        if(cwPrmsGpu) {
            cudaFree(cwPrmsGpu);
            cwPrmsGpu = nullptr;
        }
    }

    void FreeResources() {
        FreeTestData();

        if(polarDecoderHndl) {
            cuphyDestroyPolarDecoder(polarDecoderHndl);
            polarDecoderHndl = nullptr;
        }

        if(dynDescrBufCpu) {
            cudaFreeHost(dynDescrBufCpu);
            dynDescrBufCpu = nullptr;
        }
        if(dynDescrBufGpu) {
            cudaFree(dynDescrBufGpu);
            dynDescrBufGpu = nullptr;
        }
    }

    // Test parameters and resources
    cudaStream_t cuStream;
    const uint16_t maxPolCws = 273;

    // Descriptor information
    size_t dynDescrSizeBytes = 0;
    size_t dynDescrAlignBytes = 0;

    // Memory resources
    void* dynDescrBufCpu = nullptr;
    void* dynDescrBufGpu = nullptr;
    cuphyPolarCwPrm_t* cwPrmsCpu = nullptr;
    cuphyPolarCwPrm_t* cwPrmsGpu = nullptr;
    __half** cwTreeLLRsAddrs = nullptr;
    uint32_t** polCbEstAddrs = nullptr;
    bool** listPolScratchAddrs = nullptr;
    uint8_t* polCrcErrorFlags = nullptr;

    // Handle
    cuphyPolarDecoderHndl_t polarDecoderHndl = nullptr;
};

// Comprehensive basic functionality test
TEST_F(PolarDecoderTest, ComprehensiveBasicFunctionalityTest) {
    std::vector<PolarTestConfig> configs = {
        {"Basic functionality", 1, 1, 1024, 512, 24, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"Multiple codewords", 4, 1, 1024, 512, 24, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"List decoding size 2", 1, 2, 1024, 512, 24, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"List decoding size 4", 1, 4, 1024, 512, 24, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"List decoding size 8", 1, 8, 1024, 512, 24, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"Async copy enabled", 1, 1, 1024, 512, 24, 1, true, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"Maximum codewords", maxPolCws, 1, 1024, 512, 24, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"Small codeword", 1, 1, 64, 32, 8, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"Large codeword", 1, 1, 1024, 768, 16, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"Minimum parameters", 1, 1, 32, 16, 0, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        // Exit-flag early return in the non-list kernel
        {"Exit flag set", 1, 1, 1024, 512, 24, 1, false, false, 1, 1, CUPHY_STATUS_SUCCESS},
        // Exit-flag early return in the list-decoder kernel (`listPolarDecoderKernel`)
        {"Exit flag set (list size 2)", 1, 2, 1024, 512, 24, 1, false, false, 1, 1, CUPHY_STATUS_SUCCESS}
    };

    for(const auto& config : configs) {
        TestConfiguration(config);
    }
}

// Comprehensive CRC and validation test
TEST_F(PolarDecoderTest, ComprehensiveCrcAndValidationTest) {
    std::vector<uint8_t> crcSizes = {0, 2, 4, 6, 8, 11, 16, 24};
    std::vector<uint8_t> crcStatusFlags = {1, 0xC0, 0xFF};
    
    for(uint8_t crcSize : crcSizes) {
        for(uint8_t statusFlag : crcStatusFlags) {
            PolarTestConfig config = {
                "CRC test", 1, 1, 1024, 512, crcSize, 1, false, false, 0, statusFlag, CUPHY_STATUS_SUCCESS
            };
            TestConfiguration(config, LLRPattern::REALISTIC);
        }
    }

    // Test CRC success path with strong LLRs
    PolarTestConfig strongConfig = {
        "CRC success path", 1, 1, 32, 16, 8, 1, false, false, 0, 0xC0, CUPHY_STATUS_SUCCESS
    };
    TestConfiguration(strongConfig, LLRPattern::STRONG);
}

// Deterministic correctness check for existing decoder cases:
// If *all* leaf LLRs are strongly positive, the hard decision is 0 for every visited bit, so the
// decoded payload must be all-zeros (independent of tree pruning details).
TEST_F(PolarDecoderTest, Deterministic_StrongPositiveLLRs_ProduceZeroPayload) {
    constexpr uint16_t N_cw     = 512;
    constexpr uint16_t A_cw     = 58;
    constexpr uint8_t  nCrcBits = 6;
    constexpr uint16_t K_cw     = static_cast<uint16_t>(A_cw + nCrcBits);

    PolarTestConfig cfg = {
        "Deterministic zeros from +LLR", // description
        1,                                // nPolCws
        1,                                // nPolLists
        N_cw,                              // N_cw
        A_cw,                              // A_cw
        nCrcBits,                          // nCrcBits
        1,                                // nCbsInUciSeg
        false,                            // enableAsync
        false,                            // zeroInsertFlag
        0,                                // exitFlag
        0,                                // en_CrcStatus
        CUPHY_STATUS_SUCCESS              // expectedStatus
    };

    SetupTestConfiguration(cfg);

    // Compute cwTreeTypes using compCwTreeTypes kernel for a single UCI segment.
    cuphyPolarUciSegPrm_t hSegPrm{};
    hSegPrm.nCbs           = 1;
    hSegPrm.childCbIdxs[0] = 0;
    hSegPrm.childCbIdxs[1] = 0;
    hSegPrm.zeroInsertFlag = 0;
    hSegPrm.nCrcBits       = nCrcBits;
    hSegPrm.K_cw           = K_cw;
    hSegPrm.N_cw           = N_cw;
    hSegPrm.E_seg          = 0;
    hSegPrm.E_cw           = 0;
    hSegPrm.n_cw           = 9; // log2(512)

    cuphyPolarUciSegPrm_t* dSegPrm = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dSegPrm, sizeof(cuphyPolarUciSegPrm_t)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dSegPrm, &hSegPrm, sizeof(cuphyPolarUciSegPrm_t), cudaMemcpyHostToDevice));

    size_t compDynSz = 0, compDynAlign = 0;
    ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphyCompCwTreeTypesGetDescrInfo(&compDynSz, &compDynAlign));
    (void)compDynAlign;

    void* compDynCpu = nullptr;
    void* compDynGpu = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMallocHost(&compDynCpu, compDynSz));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&compDynGpu, compDynSz));

    void* compDynCpuTreeAddrs = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMallocHost(&compDynCpuTreeAddrs, sizeof(uint8_t*) * 1));

    cuphyCompCwTreeTypesHndl_t compHndl = nullptr;
    ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphyCreateCompCwTreeTypes(&compHndl));

    cuphyCompCwTreeTypesLaunchCfg_t compLaunchCfg{};
    std::vector<uint8_t*> cwTreeTypesAddrsCpu(1);
    cwTreeTypesAddrsCpu[0] = reinterpret_cast<uint8_t*>(cwPrmsCpu[0].pCwTreeTypes);

    const bool enableDescrAsync = false;
    ASSERT_EQ(CUPHY_STATUS_SUCCESS,
              cuphySetupCompCwTreeTypes(compHndl,
                                       /*nPolUciSegs*/ 1,
                                       &hSegPrm,
                                       dSegPrm,
                                       cwTreeTypesAddrsCpu.data(),
                                       compDynCpu,
                                       compDynGpu,
                                       compDynCpuTreeAddrs,
                                       enableDescrAsync,
                                       &compLaunchCfg,
                                       cuStream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(compDynGpu, compDynCpu, compDynSz, cudaMemcpyHostToDevice, cuStream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(cuStream));

    {
        const CUDA_KERNEL_NODE_PARAMS& p = compLaunchCfg.kernelNodeParamsDriver;
        ASSERT_NE(nullptr, p.func);
        const CUresult runStatus = cuLaunchKernel(p.func,
                                                  p.gridDimX, p.gridDimY, p.gridDimZ,
                                                  p.blockDimX, p.blockDimY, p.blockDimZ,
                                                  p.sharedMemBytes,
                                                  static_cast<CUstream>(cuStream),
                                                  p.kernelParams,
                                                  p.extra);
        ASSERT_EQ(CUDA_SUCCESS, runStatus);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(cuStream));
    }

    ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphyDestroyCompCwTreeTypes(compHndl));
    cudaFree(dSegPrm);
    cudaFreeHost(compDynCpu);
    cudaFree(compDynGpu);
    cudaFreeHost(compDynCpuTreeAddrs);

    // Leaf LLRs: all strongly positive => decoded payload must be all zeros.
    constexpr float L = 8.0f;
    std::vector<__half> hLLRs(N_cw, __float2half(+L));
    ASSERT_EQ(cudaSuccess, cudaMemset(cwTreeLLRsAddrs[0], 0, 2u * static_cast<size_t>(N_cw) * sizeof(__half)));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(reinterpret_cast<void*>(cwTreeLLRsAddrs[0] + N_cw),
                         hLLRs.data(),
                         N_cw * sizeof(__half),
                         cudaMemcpyHostToDevice));

    ASSERT_EQ(CUPHY_STATUS_SUCCESS, RunPolarDecoder(cfg));

    const uint32_t outWords = (A_cw + 31u) / 32u;
    std::vector<uint32_t> hEst(outWords, 0xFFFFFFFFu);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(hEst.data(), polCbEstAddrs[0], outWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for(uint32_t w = 0; w < outWords; ++w)
    {
        ASSERT_EQ(0u, hEst[w]) << "Expected all-zero payload word, word=" << w;
    }

    FreeTestData();
}

// Targeted test to cover `validate_crc()` CRC16 path (nCrcBits == 11) in the list-decoder finalize stage.
// `validate_crc()` is only executed when list decoding runs, which requires:
// - nPolLists > 1 (launch list kernel)
// - initial single-pass CRC check fails (so list decoder is invoked)
//
// We use multiple codewords and a corrupted LLR pattern to make CRC failure extremely likely for at least
// one codeword (sufficient for statement/branch coverage).
TEST_F(PolarDecoderTest, CoversValidateCrcCrc16PathInListDecoder) {
    PolarTestConfig cfg = {
        "validate_crc CRC16 (list decoder)", // description
        64,                                  // nPolCws
        2,                                   // nPolLists
        128,                                 // N_cw
        63,                                  // A_cw
        11,                                  // nCrcBits (CRC16 LUT path + maskCrc=0xFFF)
        1,                                   // nCbsInUciSeg
        false,                               // enableAsync
        false,                               // zeroInsertFlag
        0,                                   // exitFlag
        1,                                   // en_CrcStatus
        CUPHY_STATUS_SUCCESS                 // expectedStatus
    };
    TestConfiguration(cfg, LLRPattern::CORRUPTED, TreePattern::INFO_HEAVY);
}

// Targeted test to exercise the tail-word path in `xorBits()` (bit copy that crosses a 32-bit boundary).
// This happens in `append_and_validate_crc()` when appending CRC bits at a payload bit index that is not
// 32-bit aligned and the CRC length exceeds the remaining head bits in that destination word.
TEST_F(PolarDecoderTest, CoversXorBitsTailWordWhenAppendingCrc) {
    PolarTestConfig cfg = {
        "Misaligned payload crosses word", // description
        1,                                 // nPolCws
        1,                                 // nPolLists
        128,                               // N_cw
        63,                                // A_cw (bitIdxDst=63 => bIdx_d=31 => only 1 head bit left)
        13,                                // nCrcBits (sz=13 > numHeadBits=1 => triggers tail-word copy)
        1,                                 // nCbsInUciSeg
        false,                             // enableAsync
        false,                             // zeroInsertFlag
        0,                                 // exitFlag
        1,                                 // en_CrcStatus
        CUPHY_STATUS_SUCCESS               // expectedStatus
    };
    TestConfiguration(cfg, LLRPattern::STRONG, TreePattern::INFO_HEAVY);
}

// Targeted test to exercise the "partial tail-word" block in `resetBits()`.
// `resetBits(estBits, A_cw, nCrcBits)` is called in the list-polar decoder finalize stage.
// We choose A_cw such that (A_cw % 32) is large (few head bits left) and nCrcBits spills into
// the next word but not a full word, which forces:
//   num_head_bits < sz  and  num_tail_bits > 0
// inside `resetBits()`.
TEST_F(PolarDecoderTest, CoversResetBitsPartialTailWordWhenClearingCrc) {
    PolarTestConfig cfg = {
        "Misaligned CRC clear crosses word", // description
        1,                                   // nPolCws
        4,                                   // nPolLists (enable list decoder path)
        128,                                 // N_cw (> 31 required by list decoder)
        63,                                  // A_cw (bitIdx=63 => bIdx=31 => num_head_bits=1)
        13,                                  // nCrcBits (13 > 1 and 12 tail bits => hits partial tail-word)
        1,                                   // nCbsInUciSeg
        false,                               // enableAsync
        false,                               // zeroInsertFlag
        0,                                   // exitFlag
        1,                                   // en_CrcStatus
        CUPHY_STATUS_SUCCESS                 // expectedStatus
    };

    // Using nCrcBits != {6,11} makes CRC checking unconditionally fail in the current implementation,
    // guaranteeing the list-decoder finalize path (and `resetBits()`) executes deterministically.
    TestConfiguration(cfg, LLRPattern::STRONG, TreePattern::INFO_HEAVY);
}

// Comprehensive UCI segment test
TEST_F(PolarDecoderTest, ComprehensiveUciSegmentTest) {
    std::vector<PolarTestConfig> configs = {
        {"Two CBs in UCI", 2, 1, 1024, 512, 24, 2, false, true, 0, 1, CUPHY_STATUS_SUCCESS},
        // Covers the `zeroInsertFlag == 0` branch in `updateUciSegEstForTwoCbsInUciSeg()` (CB0 path).
        {"Two CBs in UCI (no zero insert)", 2, 1, 1024, 512, 24, 2, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        // Covers the `updateUciSegEstForTwoCbsInUciSeg()` call site in the list-decoder path
        // (`listPolarDecoder()`), which is only executed when list decoding is enabled and the
        // initial single-pass CRC check fails.
        // Using nCrcBits != {6,11} makes the current CRC check deterministically fail, so the
        // list decoder runs every time.
        {"Two CBs in UCI (list decode)", 2, 2, 1024, 512, 13, 2, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"Three CBs in UCI", 3, 1, 1024, 512, 24, 3, false, true, 0, 1, CUPHY_STATUS_SUCCESS},
        {"Zero insertion flag", 2, 1, 1024, 512, 24, 2, false, true, 0, 1, CUPHY_STATUS_SUCCESS}
    };
    
    for(const auto& config : configs) {
        TestConfiguration(config);
    }
}

// Comprehensive list decoder test
TEST_F(PolarDecoderTest, ComprehensiveListDecoderTest) {
    std::vector<uint8_t> listSizes = {2, 4, 8};
    std::vector<LLRPattern> patterns = {LLRPattern::WEAK, LLRPattern::ULTRA_WEAK, LLRPattern::CORRUPTED, LLRPattern::ALTERNATING};
    
    for(uint8_t listSize : listSizes) {
        for(LLRPattern pattern : patterns) {
            PolarTestConfig config = {
                "List decoder test", 1, listSize, 1024, 512, 24, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS
            };
            TestConfiguration(config, pattern);
        }
    }
    
    // Test with maximum codewords and list decoding
    PolarTestConfig maxConfig = {
        "Max codewords list", maxPolCws, 2, 1024, 512, 24, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS
    };
    TestConfiguration(maxConfig, LLRPattern::WEAK);
}

// Comprehensive kernel function test
TEST_F(PolarDecoderTest, ComprehensiveKernelFunctionTest) {
    std::vector<PolarTestConfig> configs = {
        // Test different sizes to trigger various kernel functions
        {"Very small kernel", 1, 4, 4, 2, 1, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"Small kernel", 1, 2, 16, 8, 4, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"Medium kernel", 1, 4, 64, 32, 8, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"Large kernel", 1, 8, 256, 128, 16, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"Cooperative kernel", 1, 8, 128, 64, 8, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS}
    };
    
    std::vector<TreePattern> treePatterns = {TreePattern::COMPLEX, TreePattern::INFO_HEAVY, TreePattern::ALTERNATING};
    
    for(const auto& config : configs) {
        for(TreePattern pattern : treePatterns) {
            TestConfiguration(config, LLRPattern::WEAK, pattern);
        }
    }
}

// Comprehensive bit manipulation test
TEST_F(PolarDecoderTest, ComprehensiveBitManipulationTest) {
    std::vector<PolarTestConfig> configs = {
        {"Bit manipulation 8", 1, 2, 8, 4, 2, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"Bit manipulation 32", 1, 2, 32, 16, 4, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"Bit manipulation 64", 1, 2, 64, 32, 8, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"Word extraction", 1, 1, 96, 48, 12, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"Bit shifting", 1, 1, 80, 40, 10, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS}
    };
    
    for(const auto& config : configs) {
        TestConfiguration(config, LLRPattern::ULTRA_WEAK, TreePattern::INFO_HEAVY);
    }
}

// Comprehensive edge case test
TEST_F(PolarDecoderTest, ComprehensiveEdgeCaseTest) {
    std::vector<PolarTestConfig> configs = {
        {"Minimal size", 1, 1, 2, 1, 0, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"Single bit", 1, 4, 4, 2, 1, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"Edge size 48", 1, 1, 48, 24, 6, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS},
        {"Complex tree", 1, 8, 64, 32, 6, 1, false, false, 0, 1, CUPHY_STATUS_SUCCESS}
    };
    
    for(const auto& config : configs) {
        TestConfiguration(config, LLRPattern::ULTRA_WEAK, TreePattern::COMPLEX);
    }
}

// Error handling test
TEST_F(PolarDecoderTest, ErrorHandlingTest) {
    cuphyPolarDecoderLaunchCfg_t launchCfg = {};
    
    // Test zero codewords
    cuphyStatus_t status = cuphySetupPolarDecoder(
        polarDecoderHndl, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 1, nullptr, false,
        dynDescrBufCpu, dynDescrBufGpu, nullptr, nullptr, nullptr, &launchCfg, cuStream);
    EXPECT_NE(CUPHY_STATUS_SUCCESS, status);

    // Test null handle
    status = cuphySetupPolarDecoder(
        nullptr, 1, nullptr, nullptr, nullptr, nullptr, nullptr, 1, nullptr, false,
        dynDescrBufCpu, dynDescrBufGpu, nullptr, nullptr, nullptr, &launchCfg, cuStream);
    EXPECT_EQ(CUPHY_STATUS_INVALID_ARGUMENT, status);

    // Test descriptor info with null parameter
    size_t sizeBytes = 0;
    status = cuphyPolarDecoderGetDescrInfo(&sizeBytes, nullptr);
    EXPECT_NE(CUPHY_STATUS_SUCCESS, status);

    // Test descriptor info normal case
    size_t alignBytes = 0;
    status = cuphyPolarDecoderGetDescrInfo(&sizeBytes, &alignBytes);
    EXPECT_EQ(CUPHY_STATUS_SUCCESS, status);
    EXPECT_GT(sizeBytes, 0);
    EXPECT_GT(alignBytes, 0);
}

// main()
int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    return result;
}