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
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include "cuphy.h"
#include "cuphy_api.h"
#include "common_utils.hpp"
#include <vector>
#include <memory>
#include <string>
#include <cstring>  // for memset
#include <random>
#include <algorithm>
#include <sstream>

// CUDA_CHECK macro is already defined in common_utils.hpp

namespace {
// Compare byte buffers and report first mismatch.
void ExpectBuffersEqual(const uint8_t* exp,
                        const uint8_t* act,
                        size_t         nBytes,
                        const char*    what,
                        uint32_t       idx)
{
    for(size_t i = 0; i < nBytes; ++i)
    {
        if(exp[i] != act[i])
        {
            std::ostringstream oss;
            oss << what << " mismatch at idx=" << idx << " byte=" << i << " exp=0x" << std::hex
                << static_cast<int>(exp[i]) << " act=0x" << static_cast<int>(act[i]) << std::dec;
            ADD_FAILURE() << oss.str();
            return;
        }
    }
}
} // namespace

// Test configuration structure for polar encoder
struct PolarEncoderTestConfig {
    const char* description;
    uint32_t nInfoBits;
    uint32_t nTxBits;
    uint32_t procModeBmsk;  // 0 for DL, 1 for UL
    cuphyStatus_t expectedStatus;
    bool testRateMatching;
    bool testMultiDCI;
    bool testMultiSSB;
    uint32_t numDCIs;
    uint16_t numSSBs;
};

// Information bit patterns for testing
enum class InfoBitPattern {
    ALL_ZEROS,      // All information bits are 0
    ALL_ONES,       // All information bits are 1
    ALTERNATING,    // Alternating 0s and 1s
    RANDOM,         // Random bit pattern
    SPARSE,         // Mostly zeros with few ones
    DENSE,          // Mostly ones with few zeros
    BURST,          // Bursts of ones and zeros
    EDGE_CASE       // Edge case patterns
};

// Test fixture for Polar Encoder testing
class PolarEncoderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA
        int deviceCount;
        ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&deviceCount));
        ASSERT_GT(deviceCount, 0);
        ASSERT_EQ(cudaSuccess, cudaSetDevice(0));

        // Create CUDA stream
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&cuStream));
        
        // Initialize test data
        ResetTestData();
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
        d_infoBits = nullptr;
        d_codedBits = nullptr;
        d_txBits = nullptr;
        h_infoBits = nullptr;
        h_codedBits = nullptr;
        h_txBits = nullptr;
        nCodedBits = 0;
    }

    // Generate information bits based on pattern type
    std::vector<uint8_t> GenerateInfoBits(uint32_t nBits, InfoBitPattern pattern, int seed = 0) {
        std::vector<uint8_t> bits((nBits + 7) / 8, 0);  // Byte-aligned
        std::mt19937 rng(seed);
        
        switch(pattern) {
            case InfoBitPattern::ALL_ZEROS:
                // Already initialized to zeros
                break;

            case InfoBitPattern::ALL_ONES:
                for(size_t i = 0; i < bits.size(); i++) {
                    bits[i] = 0xFF;
                }
                // Clear extra bits in last byte
                if(nBits % 8 != 0) {
                    uint8_t mask = (1 << (nBits % 8)) - 1;
                    bits.back() &= mask;
                }
                break;

            case InfoBitPattern::ALTERNATING:
                for(uint32_t i = 0; i < nBits; i++) {
                    if(i % 2 == 1) {
                        bits[i / 8] |= (1 << (i % 8));
                    }
                }
                break;

            case InfoBitPattern::RANDOM:
                for(size_t i = 0; i < bits.size(); i++) {
                    bits[i] = rng() & 0xFF;
                }
                // Clear extra bits in last byte
                if(nBits % 8 != 0) {
                    uint8_t mask = (1 << (nBits % 8)) - 1;
                    bits.back() &= mask;
                }
                break;

            case InfoBitPattern::SPARSE:
                // Set approximately 10% of bits to 1
                for(uint32_t i = 0; i < nBits; i++) {
                    if(rng() % 10 == 0) {
                        bits[i / 8] |= (1 << (i % 8));
                    }
                }
                break;

            case InfoBitPattern::DENSE:
                // Set approximately 90% of bits to 1
                for(uint32_t i = 0; i < nBits; i++) {
                    if(rng() % 10 != 0) {
                        bits[i / 8] |= (1 << (i % 8));
                    }
                }
                break;

            case InfoBitPattern::BURST:
                // Create bursts of 8 ones followed by 8 zeros
                for(uint32_t i = 0; i < nBits; i++) {
                    if((i / 8) % 2 == 0) {
                        bits[i / 8] |= (1 << (i % 8));
                    }
                }
                break;

            case InfoBitPattern::EDGE_CASE:
                // Set first and last bits, and some middle bits
                if(nBits > 0) {
                    bits[0] |= 1;  // First bit
                    if(nBits > 1) {
                        bits[(nBits - 1) / 8] |= (1 << ((nBits - 1) % 8));  // Last bit
                    }
                    if(nBits > 2) {
                        bits[(nBits / 2) / 8] |= (1 << ((nBits / 2) % 8));  // Middle bit
                    }
                }
                break;
        }

        return bits;
    }

    // Setup test buffers for encoding
    void SetupTestBuffers(const PolarEncoderTestConfig& config) {
        FreeTestData();

        // Calculate buffer sizes
        uint32_t infoBytesSize = (config.nInfoBits + 7) / 8;
        uint32_t codedBytesSize = (CUPHY_POLAR_ENC_MAX_CODED_BITS + 7) / 8;
        uint32_t txBytesSize = (config.nTxBits + 7) / 8;

        // Ensure 32-bit alignment as required by the encoder
        infoBytesSize = ((infoBytesSize + 3) / 4) * 4;
        codedBytesSize = ((codedBytesSize + 3) / 4) * 4;
        txBytesSize = ((txBytesSize + 3) / 4) * 4;

        // Allocate host memory
        ASSERT_EQ(cudaSuccess, cudaMallocHost(&h_infoBits, infoBytesSize));
        ASSERT_EQ(cudaSuccess, cudaMallocHost(&h_codedBits, codedBytesSize));
        ASSERT_EQ(cudaSuccess, cudaMallocHost(&h_txBits, txBytesSize));

        // Allocate device memory
        ASSERT_EQ(cudaSuccess, cudaMalloc(&d_infoBits, infoBytesSize));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&d_codedBits, codedBytesSize));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&d_txBits, txBytesSize));

        // Initialize memory to zero
        memset(h_infoBits, 0, infoBytesSize);
        memset(h_codedBits, 0, codedBytesSize);
        memset(h_txBits, 0, txBytesSize);

        ASSERT_EQ(cudaSuccess, cudaMemset(d_infoBits, 0, infoBytesSize));
        ASSERT_EQ(cudaSuccess, cudaMemset(d_codedBits, 0, codedBytesSize));
        ASSERT_EQ(cudaSuccess, cudaMemset(d_txBits, 0, txBytesSize));
    }

    // Setup information bits with specified pattern
    void SetupInfoBits(const PolarEncoderTestConfig& config, InfoBitPattern pattern, int seed = 0) {
        auto hostInfoBits = GenerateInfoBits(config.nInfoBits, pattern, seed);
        uint32_t infoBytesSize = (config.nInfoBits + 7) / 8;
        infoBytesSize = ((infoBytesSize + 3) / 4) * 4;  // 32-bit alignment
        
        memcpy(h_infoBits, hostInfoBits.data(), std::min(infoBytesSize, (uint32_t)hostInfoBits.size()));

        ASSERT_EQ(cudaSuccess, cudaMemcpy(d_infoBits, h_infoBits, infoBytesSize, cudaMemcpyHostToDevice));
    }

    // Run basic polar encoder test
    cuphyStatus_t RunPolarEncoder(const PolarEncoderTestConfig& config) {
        cuphyStatus_t status = cuphyPolarEncRateMatch(
            config.nInfoBits,
            config.nTxBits,
            d_infoBits,
            &nCodedBits,
            d_codedBits,
            d_txBits,
            config.procModeBmsk,
            cuStream);

        if(status == CUPHY_STATUS_SUCCESS) {
            CUDA_CHECK(cudaStreamSynchronize(cuStream));
            
            // Copy results back to host for verification
            uint32_t codedBytesSize = (nCodedBits + 7) / 8;
            uint32_t txBytesSize = (config.nTxBits + 7) / 8;
            
            CUDA_CHECK(cudaMemcpy(h_codedBits, d_codedBits, codedBytesSize, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_txBits, d_txBits, txBytesSize, cudaMemcpyDeviceToHost));
        }

        return status;
    }

    // Test a configuration and verify result
    void TestConfiguration(const PolarEncoderTestConfig& config, InfoBitPattern pattern = InfoBitPattern::RANDOM, int seed = 0) {
        printf("Testing: %s\n", config.description);

        SetupTestBuffers(config);
        SetupInfoBits(config, pattern, seed);
        
        cuphyStatus_t status = RunPolarEncoder(config);
        EXPECT_EQ(config.expectedStatus, status) << "Failed for: " << config.description;

        if(status == CUPHY_STATUS_SUCCESS) {
            VerifyEncodingResults(config);
        }

        FreeTestData();
    }

    // Verify encoding results
    void VerifyEncodingResults(const PolarEncoderTestConfig& config) {
        // Basic sanity checks
        EXPECT_GT(nCodedBits, 0) << "Number of coded bits should be positive";
        EXPECT_GE(nCodedBits, config.nInfoBits) << "Coded bits should be >= info bits";
        EXPECT_LE(nCodedBits, CUPHY_POLAR_ENC_MAX_CODED_BITS) << "Coded bits should not exceed maximum";

        // Check that coded bits is a power of 2 and within valid range
        EXPECT_TRUE(IsPowerOfTwo(nCodedBits)) << "Coded bits should be power of 2";
        EXPECT_GE(nCodedBits, 32) << "Coded bits should be >= 32";

        // Verify output buffers are not all zeros (unless input was all zeros)
        bool allZerosCoded = true;
        bool allZerosTx = true;

        uint32_t codedBytesSize = (nCodedBits + 7) / 8;
        uint32_t txBytesSize = (config.nTxBits + 7) / 8;
        
        for(uint32_t i = 0; i < codedBytesSize; i++) {
            if(h_codedBits[i] != 0) {
                allZerosCoded = false;
                break;
            }
        }

        for(uint32_t i = 0; i < txBytesSize; i++) {
            if(h_txBits[i] != 0) {
                allZerosTx = false;
                break;
            }
        }

        // Check if input was all zeros
        bool allZerosInput = true;
        uint32_t infoBytesSize = (config.nInfoBits + 7) / 8;
        for(uint32_t i = 0; i < infoBytesSize; i++) {
            if(h_infoBits[i] != 0) {
                allZerosInput = false;
                break;
            }
        }

        if(!allZerosInput) {
            EXPECT_FALSE(allZerosCoded) << "Coded output should not be all zeros for non-zero input";
            EXPECT_FALSE(allZerosTx) << "TX output should not be all zeros for non-zero input";
        }
    }

    // Check if number is power of 2
    bool IsPowerOfTwo(uint32_t n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    void FreeTestData() {
        if(d_infoBits) { cudaFree(d_infoBits); d_infoBits = nullptr; }
        if(d_codedBits) { cudaFree(d_codedBits); d_codedBits = nullptr; }
        if(d_txBits) { cudaFree(d_txBits); d_txBits = nullptr; }
        if(h_infoBits) { cudaFreeHost(h_infoBits); h_infoBits = nullptr; }
        if(h_codedBits) { cudaFreeHost(h_codedBits); h_codedBits = nullptr; }
        if(h_txBits) { cudaFreeHost(h_txBits); h_txBits = nullptr; }
    }

    void FreeResources() {
        FreeTestData();
    }

    // Test parameters and resources
    cudaStream_t cuStream = nullptr;
    
    // Memory resources
    uint8_t* d_infoBits = nullptr;
    uint8_t* d_codedBits = nullptr;
    uint8_t* d_txBits = nullptr;
    uint8_t* h_infoBits = nullptr;
    uint8_t* h_codedBits = nullptr;
    uint8_t* h_txBits = nullptr;
    uint32_t nCodedBits = 0;
};

// Test fixture for Multi-DCI polar encoder testing
class PolarEncoderMultiDCITest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA
        int deviceCount;
        ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&deviceCount));
        ASSERT_GT(deviceCount, 0);
        ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
        
        // Create CUDA stream
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&cuStream));

        ResetTestData();
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

    void ResetTestData() {
        d_infoBits = nullptr;
        d_codedBits = nullptr;
        d_txBits = nullptr;
        h_dciParams = nullptr;
        d_dciParams = nullptr;
        h_dciTmInfo = nullptr;
        d_dciTmInfo = nullptr;
    }

    void SetupMultiDCIBuffers(uint32_t numDCIs) {
        FreeTestData();
        
        // Allocate buffers for multiple DCIs
        uint32_t totalInfoSize = numDCIs * CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC;
        uint32_t totalCodedSize = numDCIs * (CUPHY_POLAR_ENC_MAX_CODED_BITS / 8);
        uint32_t totalTxSize = numDCIs * (CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 8);
        
        ASSERT_EQ(cudaSuccess, cudaMalloc(&d_infoBits, totalInfoSize));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&d_codedBits, totalCodedSize));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&d_txBits, totalTxSize));
        
        // Allocate DCI parameters
        ASSERT_EQ(cudaSuccess, cudaMallocHost(&h_dciParams, numDCIs * sizeof(cuphyPdcchDciPrm_t)));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&d_dciParams, numDCIs * sizeof(cuphyPdcchDciPrm_t)));
        
        // Allocate testing mode info
        uint32_t tmInfoSize = (numDCIs + 7) / 8;  // One bit per DCI
        ASSERT_EQ(cudaSuccess, cudaMallocHost(&h_dciTmInfo, tmInfoSize));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&d_dciTmInfo, tmInfoSize));
        
        // Initialize memory
        ASSERT_EQ(cudaSuccess, cudaMemset(d_infoBits, 0, totalInfoSize));
        ASSERT_EQ(cudaSuccess, cudaMemset(d_codedBits, 0, totalCodedSize));
        ASSERT_EQ(cudaSuccess, cudaMemset(d_txBits, 0, totalTxSize));
        memset(h_dciParams, 0, numDCIs * sizeof(cuphyPdcchDciPrm_t));
        memset(h_dciTmInfo, 0, tmInfoSize);
    }

    void SetupDCIParameters(uint32_t numDCIs, bool enableTestingMode = false) {
        std::vector<uint16_t> aggrLevels = {1, 2, 4, 8, 16};
        std::vector<uint16_t> payloadSizes = {12, 24, 48, 64, 108};
        
        for(uint32_t i = 0; i < numDCIs; i++) {
            h_dciParams[i].aggr_level = aggrLevels[i % aggrLevels.size()];
            h_dciParams[i].Npayload = payloadSizes[i % payloadSizes.size()];
            
            // Set testing mode for some DCIs if enabled
            if(enableTestingMode && (i % 3 == 0)) {
                h_dciTmInfo[i / 8] |= (1 << (i % 8));
            }
        }

        // Copy to device
        ASSERT_EQ(cudaSuccess, cudaMemcpy(d_dciParams, h_dciParams, numDCIs * sizeof(cuphyPdcchDciPrm_t), cudaMemcpyHostToDevice));
        uint32_t tmInfoSize = (numDCIs + 7) / 8;
        ASSERT_EQ(cudaSuccess, cudaMemcpy(d_dciTmInfo, h_dciTmInfo, tmInfoSize, cudaMemcpyHostToDevice));
    }

    void FreeTestData() {
        if(d_infoBits) { cudaFree(d_infoBits); d_infoBits = nullptr; }
        if(d_codedBits) { cudaFree(d_codedBits); d_codedBits = nullptr; }
        if(d_txBits) { cudaFree(d_txBits); d_txBits = nullptr; }
        if(h_dciParams) { cudaFreeHost(h_dciParams); h_dciParams = nullptr; }
        if(d_dciParams) { cudaFree(d_dciParams); d_dciParams = nullptr; }
        if(h_dciTmInfo) { cudaFreeHost(h_dciTmInfo); h_dciTmInfo = nullptr; }
        if(d_dciTmInfo) { cudaFree(d_dciTmInfo); d_dciTmInfo = nullptr; }
    }

    void FreeResources() {
        FreeTestData();
    }

    // Test parameters and resources
    cudaStream_t cuStream = nullptr;
    
    // Memory resources
    uint8_t* d_infoBits = nullptr;
    uint8_t* d_codedBits = nullptr;
    uint8_t* d_txBits = nullptr;
    cuphyPdcchDciPrm_t* h_dciParams = nullptr;
    cuphyPdcchDciPrm_t* d_dciParams = nullptr;
    uint8_t* h_dciTmInfo = nullptr;
    uint8_t* d_dciTmInfo = nullptr;
};

// Test fixture for Multi-SSB polar encoder testing
class PolarEncoderMultiSSBTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA
        int deviceCount;
        ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&deviceCount));
        ASSERT_GT(deviceCount, 0);
        ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
        
        // Create CUDA stream
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&cuStream));
        
        ResetTestData();
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

    void ResetTestData() {
        d_infoBits = nullptr;
        d_codedBits = nullptr;
        d_txBits = nullptr;
        launchCfg = {};
        ssbMapperLaunchCfg = {};
    }

    void SetupMultiSSBBuffers(uint16_t numSSBs) {
        FreeTestData();
        
        // Calculate buffer sizes for SSBs
        uint32_t totalInfoSize = numSSBs * 8;  // CUPHY_SSB_N_PBCH_SEQ_W_CRC_BITS rounded to bytes
        uint32_t totalCodedSize = numSSBs * (CUPHY_SSB_N_PBCH_POLAR_ENCODED_BITS / 8);
        uint32_t totalTxSize = numSSBs * (CUPHY_SSB_N_PBCH_SCRAMBLING_SEQ_BITS / 8);
        
        ASSERT_EQ(cudaSuccess, cudaMalloc(&d_infoBits, totalInfoSize));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&d_codedBits, totalCodedSize));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&d_txBits, totalTxSize));
        
        // Initialize memory
        ASSERT_EQ(cudaSuccess, cudaMemset(d_infoBits, 0, totalInfoSize));
        ASSERT_EQ(cudaSuccess, cudaMemset(d_codedBits, 0, totalCodedSize));
        ASSERT_EQ(cudaSuccess, cudaMemset(d_txBits, 0, totalTxSize));
    }

    cuphyStatus_t RunMultiSSBEncoder(uint16_t numSSBs) {
        // Setup launch configuration using exported API.
        cuphyStatus_t status = cuphySSBsKernelSelect(&launchCfg, &ssbMapperLaunchCfg, numSSBs);

        if(status == CUPHY_STATUS_SUCCESS) {
            // Launch kernel using driver API parameters provided by kernelSelect...().
            // (This TU is compiled with g++, so we cannot use <<< >>> syntax here.)
            launchCfg.kernelArgs[0]                       = reinterpret_cast<void*>(&d_infoBits);
            launchCfg.kernelArgs[1]                       = reinterpret_cast<void*>(&d_codedBits);
            launchCfg.kernelArgs[2]                       = reinterpret_cast<void*>(&d_txBits);
            launchCfg.kernelNodeParamsDriver.kernelParams = &(launchCfg.kernelArgs[0]);

            const CUDA_KERNEL_NODE_PARAMS& k            = launchCfg.kernelNodeParamsDriver;
            CUresult                       launchResult = cuLaunchKernel(
                k.func,
                k.gridDimX,
                k.gridDimY,
                k.gridDimZ,
                k.blockDimX,
                k.blockDimY,
                k.blockDimZ,
                k.sharedMemBytes,
                static_cast<CUstream>(cuStream),
                k.kernelParams,
                k.extra);
            EXPECT_EQ(launchResult, CUDA_SUCCESS);
            CUDA_CHECK(cudaStreamSynchronize(cuStream));
        }

        return status;
    }

    void FreeTestData() {
        if(d_infoBits) { cudaFree(d_infoBits); d_infoBits = nullptr; }
        if(d_codedBits) { cudaFree(d_codedBits); d_codedBits = nullptr; }
        if(d_txBits) { cudaFree(d_txBits); d_txBits = nullptr; }
    }

    void FreeResources() {
        FreeTestData();
    }

    // Test parameters and resources
    cudaStream_t cuStream = nullptr;
    cuphyEncoderRateMatchMultiSSBLaunchCfg_t launchCfg = {};
    cuphySsbMapperLaunchCfg_t                ssbMapperLaunchCfg = {};
    
    // Memory resources
    uint8_t* d_infoBits = nullptr;
    uint8_t* d_codedBits = nullptr;
    uint8_t* d_txBits = nullptr;
};

//-----------------------------------------------------------------------------
// Kernel-select API coverage tests

TEST(PolarEncoderKernelSelect, MultiDCI_InvalidArgs_ReturnsInvalidArgument)
{
    EXPECT_EQ(CUPHY_STATUS_INVALID_ARGUMENT,
              cuphyPdcchPipelinePrepare(nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        /*num_coresets*/ 0,
                                        /*num_DCIs*/ 0,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr));
}

// NOTE: Line 1144 in `polar_encoder::kernelSelectEncodeRateMatchMultiDCIs()` is an internal null-check:
// `if (pLaunchCfg == nullptr) return CUPHY_STATUS_INVALID_ARGUMENT;`
//
// That line can only be covered if the internal C++ symbol is reachable from the test binary.
// In Release builds, `libcuphy.so` is typically linked with a version script that hides internal
// symbols, so we attempt to resolve the mangled symbol dynamically and skip the test if it is not
// exported.
TEST(PolarEncoderKernelSelect, MultiDCI_InternalNullLaunchCfg_CoversLine1144_WhenExported)
{
   using Fn = cuphyStatus_t (*)(cuphyEncoderRateMatchMultiDCILaunchCfg_t*, uint32_t);

   // Mangled name from the toolchain for:
   // polar_encoder::kernelSelectEncodeRateMatchMultiDCIs(cuphyEncoderRateMatchMultiDCILaunchCfg_t*, uint32_t)
   constexpr const char* kMangled =
       "_ZN13polar_encoder36kernelSelectEncodeRateMatchMultiDCIsEP40cuphyEncoderRateMatchMultiDCILaunchCfg_tj";

   // Clear any stale error and try to resolve from already-loaded DSOs.
   (void)dlerror();
   void* sym = dlsym(RTLD_DEFAULT, kMangled);
   if(sym == nullptr)
   {
       const char* err = dlerror();
       GTEST_SKIP() << "Internal symbol not exported; cannot cover line 1144 via test-only change. dlerror="
                    << (err ? err : "(none)");
   }

   auto fn = reinterpret_cast<Fn>(sym);
   EXPECT_EQ(CUPHY_STATUS_INVALID_ARGUMENT, fn(nullptr, /*num_DCIs*/ 1));
}

TEST(PolarEncoderKernelSelect, MultiSSB_InvalidArgs_ReturnsInvalidArgument)
{
    cuphySsbMapperLaunchCfg_t ssbMapperCfg = {};
    EXPECT_EQ(CUPHY_STATUS_INVALID_ARGUMENT,
              cuphySSBsKernelSelect(nullptr, &ssbMapperCfg, /*num_SSBs*/ 1));
}


// Comprehensive basic functionality test
TEST_F(PolarEncoderTest, ComprehensiveBasicFunctionalityTest) {
    std::vector<PolarEncoderTestConfig> configs = {
        // Basic functionality tests
        {"Basic DL encoding", 100, 200, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Basic UL encoding", 100, 200, 1, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Minimum info bits", 1, 32, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Maximum info bits", CUPHY_POLAR_ENC_MAX_INFO_BITS, CUPHY_POLAR_ENC_MAX_TX_BITS, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},

        // Rate matching scenarios - valid combinations respecting min code rate (1/8)
        {"Repetition case", 50, 200, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Valid puncturing case", 50, 100, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Valid shortening case", 25, 100, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},

        // Different code rates - all respecting minimum code rate constraint
        {"Code rate 1/2", 100, 200, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Code rate 1/4", 50, 200, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Code rate 1/8", 32, 256, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},

        // Edge cases
        {"Small sizes", 8, 64, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Medium sizes", 100, 400, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Large sizes", 150, 1000, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0}
    };

    for(const auto& config : configs) {
        TestConfiguration(config, InfoBitPattern::RANDOM);
    }
}

// Comprehensive information bit pattern test
TEST_F(PolarEncoderTest, ComprehensiveInfoBitPatternTest) {
    std::vector<InfoBitPattern> patterns = {
        InfoBitPattern::ALL_ZEROS,
        InfoBitPattern::ALL_ONES,
        InfoBitPattern::ALTERNATING,
        InfoBitPattern::RANDOM,
        InfoBitPattern::SPARSE,
        InfoBitPattern::DENSE,
        InfoBitPattern::BURST,
        InfoBitPattern::EDGE_CASE
    };

    PolarEncoderTestConfig baseConfig = {
        "Pattern test", 100, 200, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0
    };

    for(InfoBitPattern pattern : patterns) {
        TestConfiguration(baseConfig, pattern);
    }
}

// Comprehensive DL vs UL encoding test
TEST_F(PolarEncoderTest, ComprehensiveDLvsULTest) {
    std::vector<PolarEncoderTestConfig> configs = {
        {"DL small", 50, 100, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"UL small", 50, 100, 1, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"DL medium", 100, 400, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"UL medium", 100, 400, 1, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"DL large", 150, 800, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"UL large", 150, 800, 1, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0}
    };

    for(const auto& config : configs) {
        TestConfiguration(config, InfoBitPattern::RANDOM);
    }
}

// Comprehensive rate matching scenarios test
TEST_F(PolarEncoderTest, ComprehensiveRateMatchingTest) {
    std::vector<PolarEncoderTestConfig> configs = {
        // Repetition scenarios (nTxBits > nCodedBits)
        {"Repetition 2x", 64, 256, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Repetition 4x", 32, 256, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Repetition 8x", 16, 256, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        // Force the "marshal remaining bits" masking path in rateMatch() (repetition + nTxBits % 8 != 0).
        // This targets the device branch that applies a bitmask to the last partial TX byte.
        {"Repetition rembits mask path", 1, 201, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},

        // Puncturing scenarios (nTxBits < nCodedBits, respecting min code rate 1/8)
        {"Puncturing valid high", 50, 100, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Puncturing valid medium", 40, 100, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        
        // Shortening scenarios (nTxBits < nCodedBits, low info/tx ratio)
        {"Shortening low ratio", 50, 400, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Shortening very low ratio", 25, 400, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},

        // Edge cases around minimum code rate (1/8 = 0.125)
        {"Min code rate edge 1", 32, 256, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Min code rate edge 2", 64, 512, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0}
    };

    for(const auto& config : configs) {
        // Use a deterministic non-zero pattern for the rembits mask-path case to avoid all-zero outputs.
        if(std::string(config.description) == "Repetition rembits mask path") {
            TestConfiguration(config, InfoBitPattern::EDGE_CASE);
        } else {
            TestConfiguration(config, InfoBitPattern::RANDOM);
        }
    }
}

// Comprehensive code size boundary test
TEST_F(PolarEncoderTest, ComprehensiveCodeSizeBoundaryTest) {
    std::vector<PolarEncoderTestConfig> configs = {
        // Test around minimum code size (32)
        {"Min code boundary 1", 1, 16, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Min code boundary 2", 8, 32, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Min code boundary 3", 16, 64, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},

        // Test around power-of-2 boundaries
        {"Power of 2 - 64", 32, 64, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Power of 2 - 128", 64, 128, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Power of 2 - 256", 128, 256, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Power of 2 - 512", 64, 512, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},

        // Test around maximum code size (512) - respecting min code rate 1/8
        {"Max code boundary 1", 64, 512, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Max code boundary 2", 125, 1000, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Max code boundary 3", CUPHY_POLAR_ENC_MAX_INFO_BITS, CUPHY_POLAR_ENC_MAX_TX_BITS, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0}
    };

    for(const auto& config : configs) {
        TestConfiguration(config, InfoBitPattern::RANDOM);
    }
}

// Comprehensive sub-block interleaving test
TEST_F(PolarEncoderTest, ComprehensiveSubBlockInterleavingTest) {
    std::vector<PolarEncoderTestConfig> configs = {
        // Test different code sizes to trigger different scale factors
        {"Interleaving 32 bits", 16, 32, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Interleaving 64 bits", 32, 64, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Interleaving 128 bits", 64, 128, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Interleaving 256 bits", 128, 256, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Interleaving 512 bits", 64, 512, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0}
    };

    for(const auto& config : configs) {
        TestConfiguration(config, InfoBitPattern::ALTERNATING);
    }
}

// Multi-DCI encoder test
TEST_F(PolarEncoderMultiDCITest, MultiDCIEncodingTest) {
    std::vector<uint32_t> dciCounts = {1, 2, 4, 8, 16};
    
    for(uint32_t numDCIs : dciCounts) {
        printf("Testing Multi-DCI encoding with %u DCIs\n", numDCIs);
        
        SetupMultiDCIBuffers(numDCIs);
        SetupDCIParameters(numDCIs, /*enableTestingMode*/ false);

        // Prepare host payload input (no CRC) and let cuPHY prepare CRC + bit-reversal and kernel launch params.
        std::vector<uint8_t> h_payload(numDCIs * CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES, 0);
        std::mt19937         rng(1234);
        for (auto& b : h_payload) b = static_cast<uint8_t>(rng() & 0xFF);

        std::vector<uint8_t> h_input_w_crc(numDCIs * CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC, 0);

        PdcchParams coresetParams = {};
        coresetParams.interleaved         = 0;
        coresetParams.bundle_size         = 6;
        coresetParams.interleaver_size    = 0;
        coresetParams.shift_index         = 0;
        coresetParams.n_sym               = 1;
        coresetParams.num_dl_dci          = numDCIs;
        coresetParams.dciStartIdx         = 0;
        coresetParams.testModel           = 0;
        coresetParams.freq_domain_resource = 0x8000000000000000ull; // non-zero, single-bit mask

        const uint32_t tmInfoSize = (numDCIs + 7) / 8;
        memset(h_dciTmInfo, 0, tmInfoSize);

        cuphyEncoderRateMatchMultiDCILaunchCfg_t encLaunchCfg = {};
        cuphyGenScramblingSeqLaunchCfg_t         scrLaunchCfg = {};
        cuphyGenPdcchTfSgnlLaunchCfg_t           tfLaunchCfg  = {};

        cuphyStatus_t status = cuphyPdcchPipelinePrepare(h_input_w_crc.data(),
                                                         nullptr,
                                                         h_payload.data(),
                                                         nullptr,
                                                         /*num_coresets*/ 1,
                                                         /*num_dcis*/ static_cast<int>(numDCIs),
                                                         &coresetParams,
                                                         h_dciParams,
                                                         h_dciTmInfo,
                                                         &encLaunchCfg,
                                                         &scrLaunchCfg,
                                                         &tfLaunchCfg,
                                                         cuStream);
        EXPECT_EQ(CUPHY_STATUS_SUCCESS, status) << "Failed for " << numDCIs << " DCIs";
        // Call again to cover the "func already initialized" path.
        status = cuphyPdcchPipelinePrepare(h_input_w_crc.data(),
                                           nullptr,
                                           h_payload.data(),
                                           nullptr,
                                           /*num_coresets*/ 1,
                                           /*num_dcis*/ static_cast<int>(numDCIs),
                                           &coresetParams,
                                           h_dciParams,
                                           h_dciTmInfo,
                                           &encLaunchCfg,
                                           &scrLaunchCfg,
                                           &tfLaunchCfg,
                                           cuStream);
        EXPECT_EQ(CUPHY_STATUS_SUCCESS, status) << "Second call failed for " << numDCIs << " DCIs";

        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(d_infoBits, h_input_w_crc.data(), h_input_w_crc.size(), cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(d_dciParams, h_dciParams, numDCIs * sizeof(cuphyPdcchDciPrm_t), cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(d_dciTmInfo, h_dciTmInfo, tmInfoSize, cudaMemcpyHostToDevice));

        // Launch kernel using the driver API parameters provided by kernelSelect...().
        // (This TU is compiled with g++, so we cannot use <<< >>> syntax here.)
        encLaunchCfg.kernelArgs[0]                       = reinterpret_cast<void*>(&d_infoBits);
        encLaunchCfg.kernelArgs[1]                       = reinterpret_cast<void*>(&d_codedBits);
        encLaunchCfg.kernelArgs[2]                       = reinterpret_cast<void*>(&d_txBits);
        encLaunchCfg.kernelArgs[3]                       = reinterpret_cast<void*>(&d_dciParams);
        encLaunchCfg.kernelArgs[4]                       = reinterpret_cast<void*>(&d_dciTmInfo);
        encLaunchCfg.kernelNodeParamsDriver.kernelParams = &(encLaunchCfg.kernelArgs[0]);

        const CUDA_KERNEL_NODE_PARAMS& k            = encLaunchCfg.kernelNodeParamsDriver;
        CUresult                       launchResult = cuLaunchKernel(
            k.func,
            k.gridDimX,
            k.gridDimY,
            k.gridDimZ,
            k.blockDimX,
            k.blockDimY,
            k.blockDimZ,
            k.sharedMemBytes,
            static_cast<CUstream>(cuStream),
            k.kernelParams,
            k.extra);
        EXPECT_EQ(launchResult, CUDA_SUCCESS);
        CUDA_CHECK(cudaStreamSynchronize(cuStream));

        // Basic sanity: expect at least one byte in TX output to be non-zero.
        std::vector<uint8_t> h_tx(numDCIs * (CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 8), 0);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(h_tx.data(), d_txBits, h_tx.size(), cudaMemcpyDeviceToHost));
        bool anyNonZero = false;
        for (auto b : h_tx) {
            if (b != 0) {
                anyNonZero = true;
                break;
            }
        }
        EXPECT_TRUE(anyNonZero) << "Expected non-zero TX output for Multi-DCI encoding";
        
        FreeTestData();
    }
}

// Multi-DCI with testing mode test
TEST_F(PolarEncoderMultiDCITest, MultiDCITestingModeTest) {
    uint32_t numDCIs = 6;

    printf("Testing Multi-DCI encoding with testing mode\n");

    SetupMultiDCIBuffers(numDCIs);
    // Split DCIs across two coresets: first non-TM, second TM.
    memset(h_dciParams, 0, numDCIs * sizeof(cuphyPdcchDciPrm_t));
    for (uint32_t dci = 0; dci < numDCIs; ++dci) {
        h_dciParams[dci].aggr_level = 1;
        h_dciParams[dci].Npayload   = (dci < 3) ? 24u : 108u; // TM DCIs use PN23 payload length
        h_dciParams[dci].rntiCrc    = 0;
    }

    // Deterministic payload pattern per DCI.
    std::vector<uint8_t> h_payload(numDCIs * CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES, 0);
    for (uint32_t dci = 0; dci < numDCIs; ++dci) {
        for (uint32_t i = 0; i < CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES; ++i) {
            h_payload[dci * CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES + i] = static_cast<uint8_t>((dci * 17u + i) & 0xFF);
        }
    }
    std::vector<uint8_t> h_input_w_crc(numDCIs * CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC, 0);

    PdcchParams coresetParams[2] = {};
    // Coreset 0: non-TM DCIs [0..2]
    coresetParams[0].interleaved          = 0;
    coresetParams[0].bundle_size          = 6;
    coresetParams[0].interleaver_size     = 0;
    coresetParams[0].shift_index          = 0;
    coresetParams[0].n_sym                = 1;
    coresetParams[0].num_dl_dci           = 3;
    coresetParams[0].dciStartIdx          = 0;
    coresetParams[0].testModel            = 0;
    coresetParams[0].freq_domain_resource = 0x8000000000000000ull;
    // Coreset 1: TM DCIs [3..5]
    coresetParams[1].interleaved          = 0;
    coresetParams[1].bundle_size          = 6;
    coresetParams[1].interleaver_size     = 0;
    coresetParams[1].shift_index          = 0;
    coresetParams[1].n_sym                = 1;
    coresetParams[1].num_dl_dci           = 3;
    coresetParams[1].dciStartIdx          = 3;
    coresetParams[1].testModel            = 1;
    coresetParams[1].freq_domain_resource = 0x8000000000000000ull;

    const uint32_t tmInfoSize = (numDCIs + 7) / 8;
    memset(h_dciTmInfo, 0, tmInfoSize);

    cuphyEncoderRateMatchMultiDCILaunchCfg_t encLaunchCfg = {};
    cuphyGenScramblingSeqLaunchCfg_t         scrLaunchCfg = {};
    cuphyGenPdcchTfSgnlLaunchCfg_t           tfLaunchCfg  = {};

    cuphyStatus_t status = cuphyPdcchPipelinePrepare(h_input_w_crc.data(),
                                                     nullptr,
                                                     h_payload.data(),
                                                     nullptr,
                                                     /*num_coresets*/ 2,
                                                     /*num_dcis*/ static_cast<int>(numDCIs),
                                                     &coresetParams[0],
                                                     h_dciParams,
                                                     h_dciTmInfo,
                                                     &encLaunchCfg,
                                                     &scrLaunchCfg,
                                                     &tfLaunchCfg,
                                                     cuStream);
    EXPECT_EQ(CUPHY_STATUS_SUCCESS, status) << "Failed for Multi-DCI with testing mode";
    // Call again to cover the "func already initialized" path.
    status = cuphyPdcchPipelinePrepare(h_input_w_crc.data(),
                                       nullptr,
                                       h_payload.data(),
                                       nullptr,
                                       /*num_coresets*/ 2,
                                       /*num_dcis*/ static_cast<int>(numDCIs),
                                       &coresetParams[0],
                                       h_dciParams,
                                       h_dciTmInfo,
                                       &encLaunchCfg,
                                       &scrLaunchCfg,
                                       &tfLaunchCfg,
                                       cuStream);
    EXPECT_EQ(CUPHY_STATUS_SUCCESS, status) << "Second call failed for Multi-DCI with testing mode";

    // Launch kernel and validate testing-mode (TM) path copies payload bytes into TX buffer.
    constexpr uint32_t kTmCopyBytes  = 14; // polar_encoder::N_MAX_TM_DCI_TX_BYTES (internal), bytes copied in TM mode
    static_assert(kTmCopyBytes > 0, "Invalid TM copy size");

    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(d_infoBits, h_input_w_crc.data(), h_input_w_crc.size(), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(d_dciParams, h_dciParams, numDCIs * sizeof(cuphyPdcchDciPrm_t), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(d_dciTmInfo, h_dciTmInfo, tmInfoSize, cudaMemcpyHostToDevice));

    encLaunchCfg.kernelArgs[0]                       = reinterpret_cast<void*>(&d_infoBits);
    encLaunchCfg.kernelArgs[1]                       = reinterpret_cast<void*>(&d_codedBits);
    encLaunchCfg.kernelArgs[2]                       = reinterpret_cast<void*>(&d_txBits);
    encLaunchCfg.kernelArgs[3]                       = reinterpret_cast<void*>(&d_dciParams);
    encLaunchCfg.kernelArgs[4]                       = reinterpret_cast<void*>(&d_dciTmInfo);
    encLaunchCfg.kernelNodeParamsDriver.kernelParams = &(encLaunchCfg.kernelArgs[0]);

    const CUDA_KERNEL_NODE_PARAMS& k            = encLaunchCfg.kernelNodeParamsDriver;
    CUresult                       launchResult = cuLaunchKernel(
        k.func,
        k.gridDimX,
        k.gridDimY,
        k.gridDimZ,
        k.blockDimX,
        k.blockDimY,
        k.blockDimZ,
        k.sharedMemBytes,
        static_cast<CUstream>(cuStream),
        k.kernelParams,
        k.extra);
    EXPECT_EQ(launchResult, CUDA_SUCCESS);
    CUDA_CHECK(cudaStreamSynchronize(cuStream));

    // Copy back TX and TM bitmask to validate behavior on TM DCIs.
    const uint32_t txStrideBytes = (CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 8);
    std::vector<uint8_t> h_tx(numDCIs * txStrideBytes, 0);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(h_tx.data(), d_txBits, h_tx.size(), cudaMemcpyDeviceToHost));
    std::vector<uint8_t> h_tm((numDCIs + 7) / 8, 0);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(h_tm.data(), d_dciTmInfo, h_tm.size(), cudaMemcpyDeviceToHost));

    for (uint32_t dci = 0; dci < numDCIs; ++dci) {
        const bool isTm = ((h_tm[dci >> 3] >> (dci & 0x7)) & 0x1) != 0;
        if (!isTm) continue;

        const uint32_t infoOff = dci * CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC;
        const uint32_t txOff   = dci * txStrideBytes;
        for (uint32_t i = 0; i < kTmCopyBytes; ++i) {
            EXPECT_EQ(h_input_w_crc[infoOff + i], h_tx[txOff + i])
                << "TM copy mismatch at DCI=" << dci << " byte=" << i;
        }
    }

    FreeTestData();
}

// `cuphyPdcchPipelinePrepare()` rejects invalid payload sizes (guarding the multi-DCI kernel from
// illegal shared-memory accesses).
TEST_F(PolarEncoderMultiDCITest, MultiDCI_MinCodedBitsClampPath) {
    const uint32_t numDCIs = 1;

    SetupMultiDCIBuffers(numDCIs);

   // Cover the clamp in `encodeRateMatchMultipleDCIsKernel`:
   //   if (nCodedBits < N_MIN_CODED_BITS) nCodedBits = N_MIN_CODED_BITS;
   //
   // We cannot pass a "wrapped" Npayload into cuphyPdcchPipelinePrepare() because it casts Npayload
   // to `int` and may index out of bounds. Instead:
   // 1) Call cuphyPdcchPipelinePrepare() once with a valid payload size to initialize the launch config.
   // 2) Overwrite the device-side DCI param Npayload with a value that makes nInfoBits wrap to 1
   //    inside the kernel (uint32 addition), which makes nMin2CodedBits = 8 and forces the clamp.
    memset(h_dciParams, 0, numDCIs * sizeof(cuphyPdcchDciPrm_t));
    h_dciParams[0].aggr_level = 1;
   h_dciParams[0].Npayload   = 24;
    h_dciParams[0].rntiCrc    = 0;

   // DCI payload input without CRC (host). Just needs to be non-zero.
   std::vector<uint8_t> h_payload(CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES, 0);
   h_payload[0] = 0x1;
    std::vector<uint8_t> h_input_w_crc(CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC, 0);

    PdcchParams coresetParams = {};
    coresetParams.interleaved          = 0;
    coresetParams.bundle_size          = 6;
    coresetParams.interleaver_size     = 0;
    coresetParams.shift_index          = 0;
    coresetParams.n_sym                = 1;
    coresetParams.num_dl_dci           = 1;
    coresetParams.dciStartIdx          = 0;
    coresetParams.testModel            = 0;
    coresetParams.freq_domain_resource = 0x8000000000000000ull;

    uint8_t tmInfo = 0;
    cuphyEncoderRateMatchMultiDCILaunchCfg_t encLaunchCfg = {};
    cuphyGenScramblingSeqLaunchCfg_t         scrLaunchCfg = {};
    cuphyGenPdcchTfSgnlLaunchCfg_t           tfLaunchCfg  = {};

   // Initialize kernel launch configuration with a valid payload size.
   cuphyStatus_t status = cuphyPdcchPipelinePrepare(h_input_w_crc.data(),
                                                    nullptr,
                                                    h_payload.data(),
                                                    nullptr,
                                                    /*num_coresets*/ 1,
                                                    /*num_dcis*/ 1,
                                                    &coresetParams,
                                                    h_dciParams,
                                                    &tmInfo,
                                                    &encLaunchCfg,
                                                    &scrLaunchCfg,
                                                    &tfLaunchCfg,
                                                    cuStream);
   ASSERT_EQ(CUPHY_STATUS_SUCCESS, status);
   ASSERT_NE(encLaunchCfg.kernelNodeParamsDriver.func, nullptr);

   // Copy prepared DCI bytes-with-CRC to device (kernel expects this format).
   ASSERT_EQ(cudaSuccess,
             cudaMemcpy(d_infoBits, h_input_w_crc.data(), h_input_w_crc.size(), cudaMemcpyHostToDevice));

   // Prefill TX buffer with a sentinel so we can confirm the kernel overwrote it.
   ASSERT_EQ(cudaSuccess,
             cudaMemset(d_txBits, 0xA5, CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 8));

   // Now force nInfoBits wrap-around in the kernel:
   // nInfoBits = 24 + Npayload => 1 (mod 2^32)
   // which yields nMin2CodedBits=8 and triggers the nCodedBits<32 clamp.
   h_dciParams[0].Npayload = 0xFFFFFFE9u; // 2^32 - 23, so 24 + Npayload wraps to 1
   ASSERT_EQ(cudaSuccess,
             cudaMemcpy(d_dciParams, h_dciParams, sizeof(cuphyPdcchDciPrm_t), cudaMemcpyHostToDevice));
   ASSERT_EQ(cudaSuccess, cudaMemcpy(d_dciTmInfo, &tmInfo, 1, cudaMemcpyHostToDevice));

   // Launch Multi-DCI kernel via driver API.
   encLaunchCfg.kernelArgs[0]                       = reinterpret_cast<void*>(&d_infoBits);
   encLaunchCfg.kernelArgs[1]                       = reinterpret_cast<void*>(&d_codedBits);
   encLaunchCfg.kernelArgs[2]                       = reinterpret_cast<void*>(&d_txBits);
   encLaunchCfg.kernelArgs[3]                       = reinterpret_cast<void*>(&d_dciParams);
   encLaunchCfg.kernelArgs[4]                       = reinterpret_cast<void*>(&d_dciTmInfo);
   encLaunchCfg.kernelNodeParamsDriver.kernelParams = &(encLaunchCfg.kernelArgs[0]);

   const CUDA_KERNEL_NODE_PARAMS& k            = encLaunchCfg.kernelNodeParamsDriver;
   const CUresult                 launchResult = cuLaunchKernel(
       k.func,
       k.gridDimX,
       k.gridDimY,
       k.gridDimZ,
       k.blockDimX,
       k.blockDimY,
       k.blockDimZ,
       k.sharedMemBytes,
       static_cast<CUstream>(cuStream),
       k.kernelParams,
       k.extra);
   ASSERT_EQ(launchResult, CUDA_SUCCESS);
   CUDA_CHECK(cudaStreamSynchronize(cuStream));

   // Basic sanity: kernel ran and overwrote at least one TX byte.
   std::vector<uint8_t> h_tx(CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 8, 0);
   ASSERT_EQ(cudaSuccess, cudaMemcpy(h_tx.data(), d_txBits, h_tx.size(), cudaMemcpyDeviceToHost));
   bool anyChanged = false;
   for (auto b : h_tx) {
       if (b != 0xA5) { anyChanged = true; break; }
   }
   EXPECT_TRUE(anyChanged);

    FreeTestData();
}

// Bit-accurate correctness check for the Multi-DCI kernel:
// Compare each DCI's coded bits and TX bits against the reference single-DCI API.
TEST_F(PolarEncoderMultiDCITest, MultiDCI_CorrectnessAgainstSingleDCIReference)
{
    const uint32_t numDCIs = 4;
    SetupMultiDCIBuffers(numDCIs);

    // Hand-pick params to cover different aggregation levels (including 1 -> non-byte-aligned nTxBits).
    memset(h_dciParams, 0, numDCIs * sizeof(cuphyPdcchDciPrm_t));
    const uint32_t aggrLevels[numDCIs]   = {1, 2, 4, 8};
    const uint32_t payloadBits[numDCIs]  = {12, 24, 48, 64};
    for(uint32_t i = 0; i < numDCIs; ++i)
    {
        h_dciParams[i].aggr_level = aggrLevels[i];
        h_dciParams[i].Npayload   = payloadBits[i];
    }
    memset(h_dciTmInfo, 0, (numDCIs + 7) / 8); // all non-TM

    // Random-but-deterministic DCI payload bytes (no CRC).
    std::vector<uint8_t> h_payload(numDCIs * CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES, 0);
    std::mt19937         rng(2468);
    for(auto& b : h_payload) b = static_cast<uint8_t>(rng() & 0xFF);

    std::vector<uint8_t> h_input_w_crc(numDCIs * CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC, 0);
    PdcchParams coresetParams = {};
    coresetParams.interleaved          = 0;
    coresetParams.bundle_size          = 6;
    coresetParams.interleaver_size     = 0;
    coresetParams.shift_index          = 0;
    coresetParams.n_sym                = 1;
    coresetParams.num_dl_dci           = numDCIs;
    coresetParams.dciStartIdx          = 0;
    coresetParams.testModel            = 0;
    coresetParams.freq_domain_resource = 0x8000000000000000ull;

    cuphyEncoderRateMatchMultiDCILaunchCfg_t encLaunchCfg = {};
    cuphyGenScramblingSeqLaunchCfg_t         scrLaunchCfg = {};
    cuphyGenPdcchTfSgnlLaunchCfg_t           tfLaunchCfg  = {};

    cuphyStatus_t status = cuphyPdcchPipelinePrepare(h_input_w_crc.data(),
                                                     nullptr,
                                                     h_payload.data(),
                                                     nullptr,
                                                     /*num_coresets*/ 1,
                                                     /*num_dcis*/ static_cast<int>(numDCIs),
                                                     &coresetParams,
                                                     h_dciParams,
                                                     h_dciTmInfo,
                                                     &encLaunchCfg,
                                                     &scrLaunchCfg,
                                                     &tfLaunchCfg,
                                                     cuStream);
    ASSERT_EQ(CUPHY_STATUS_SUCCESS, status);

    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_infoBits, h_input_w_crc.data(), h_input_w_crc.size(), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(d_dciParams, h_dciParams, numDCIs * sizeof(cuphyPdcchDciPrm_t), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(d_dciTmInfo, h_dciTmInfo, (numDCIs + 7) / 8, cudaMemcpyHostToDevice));

    // Launch Multi-DCI kernel via driver API.
    encLaunchCfg.kernelArgs[0]                       = reinterpret_cast<void*>(&d_infoBits);
    encLaunchCfg.kernelArgs[1]                       = reinterpret_cast<void*>(&d_codedBits);
    encLaunchCfg.kernelArgs[2]                       = reinterpret_cast<void*>(&d_txBits);
    encLaunchCfg.kernelArgs[3]                       = reinterpret_cast<void*>(&d_dciParams);
    encLaunchCfg.kernelArgs[4]                       = reinterpret_cast<void*>(&d_dciTmInfo);
    encLaunchCfg.kernelNodeParamsDriver.kernelParams = &(encLaunchCfg.kernelArgs[0]);

    const CUDA_KERNEL_NODE_PARAMS& k            = encLaunchCfg.kernelNodeParamsDriver;
    CUresult                       launchResult = cuLaunchKernel(
        k.func, k.gridDimX, k.gridDimY, k.gridDimZ, k.blockDimX, k.blockDimY, k.blockDimZ, k.sharedMemBytes,
        static_cast<CUstream>(cuStream), k.kernelParams, k.extra);
    ASSERT_EQ(launchResult, CUDA_SUCCESS);
    CUDA_CHECK(cudaStreamSynchronize(cuStream));

    // Pull Multi-DCI outputs to host.
    const uint32_t codedStrideBytes = (CUPHY_POLAR_ENC_MAX_CODED_BITS / 8);
    const uint32_t txStrideBytes    = (CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 8);
    std::vector<uint8_t> h_coded_multi(numDCIs * codedStrideBytes, 0);
    std::vector<uint8_t> h_tx_multi(numDCIs * txStrideBytes, 0);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(h_coded_multi.data(), d_codedBits, h_coded_multi.size(), cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(h_tx_multi.data(), d_txBits, h_tx_multi.size(), cudaMemcpyDeviceToHost));

    // Reference buffers reused per DCI.
    uint8_t* d_info_ref  = nullptr;
    uint8_t* d_coded_ref = nullptr;
    uint8_t* d_tx_ref    = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_info_ref, CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_coded_ref, codedStrideBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_tx_ref, txStrideBytes));

    std::vector<uint8_t> h_coded_ref(codedStrideBytes, 0);
    std::vector<uint8_t> h_tx_ref(txStrideBytes, 0);

    for(uint32_t dci = 0; dci < numDCIs; ++dci)
    {
        const uint32_t nInfoBits = CUPHY_PDCCH_N_CRC_BITS + h_dciParams[dci].Npayload;
        const uint32_t nTxBits   = 2 * 9 * 6 * h_dciParams[dci].aggr_level;

        ASSERT_EQ(cudaSuccess, cudaMemset(d_coded_ref, 0, codedStrideBytes));
        ASSERT_EQ(cudaSuccess, cudaMemset(d_tx_ref, 0, txStrideBytes));
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(d_info_ref,
                             &h_input_w_crc[dci * CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC],
                             CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC,
                             cudaMemcpyHostToDevice));

        uint32_t nCodedBitsRef = 0;
        cuphyStatus_t st = cuphyPolarEncRateMatch(
            nInfoBits, nTxBits, d_info_ref, &nCodedBitsRef, d_coded_ref, d_tx_ref, /*procModeBmsk*/ 0, cuStream);
        ASSERT_EQ(CUPHY_STATUS_SUCCESS, st) << "Reference API failed for DCI=" << dci;
        CUDA_CHECK(cudaStreamSynchronize(cuStream));

        const uint32_t codedBytesRef = (nCodedBitsRef + 7) / 8;
        const uint32_t txBytesRef    = (nTxBits + 7) / 8;

        ASSERT_EQ(cudaSuccess, cudaMemcpy(h_coded_ref.data(), d_coded_ref, codedBytesRef, cudaMemcpyDeviceToHost));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(h_tx_ref.data(), d_tx_ref, txBytesRef, cudaMemcpyDeviceToHost));

        const uint8_t* coded_multi_ptr = &h_coded_multi[dci * codedStrideBytes];
        const uint8_t* tx_multi_ptr    = &h_tx_multi[dci * txStrideBytes];

        ExpectBuffersEqual(h_coded_ref.data(), coded_multi_ptr, codedBytesRef, "codedBits", dci);
        ExpectBuffersEqual(h_tx_ref.data(), tx_multi_ptr, txBytesRef, "txBits", dci);
    }

    cudaFree(d_info_ref);
    cudaFree(d_coded_ref);
    cudaFree(d_tx_ref);

    FreeTestData();
}

// Multi-SSB encoder test
TEST_F(PolarEncoderMultiSSBTest, MultiSSBEncodingTest) {
    std::vector<uint16_t> ssbCounts = {1, 2, 4, 8, 16, 32, 64};
    
    for(uint16_t numSSBs : ssbCounts) {
        printf("Testing Multi-SSB encoding with %u SSBs\n", numSSBs);

        SetupMultiSSBBuffers(numSSBs);

        // Fill SSB info bits with non-zero data to ensure non-trivial output.
        const uint32_t totalInfoSize = static_cast<uint32_t>(numSSBs) * 8;
        std::vector<uint8_t> h_info(totalInfoSize, 0);
        std::mt19937         rng(5678);
        for (auto& b : h_info) b = static_cast<uint8_t>(rng() & 0xFF);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(d_infoBits, h_info.data(), h_info.size(), cudaMemcpyHostToDevice));

        cuphyStatus_t status = RunMultiSSBEncoder(numSSBs);
        EXPECT_EQ(CUPHY_STATUS_SUCCESS, status) << "Failed for " << numSSBs << " SSBs";

        if(status == CUPHY_STATUS_SUCCESS) {
            const uint32_t totalTxSize = static_cast<uint32_t>(numSSBs) * (CUPHY_SSB_N_PBCH_SCRAMBLING_SEQ_BITS / 8);
            std::vector<uint8_t> h_tx(totalTxSize, 0);
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy(h_tx.data(), d_txBits, h_tx.size(), cudaMemcpyDeviceToHost));
            bool anyNonZero = false;
            for (auto b : h_tx) {
                if (b != 0) { anyNonZero = true; break; }
            }
            EXPECT_TRUE(anyNonZero) << "Expected non-zero TX output for Multi-SSB encoding";
        }

        FreeTestData();
    }
}

// Bit-accurate correctness check for the Multi-SSB kernel:
// Compare a few SSBs against the reference single-encoder API.
TEST_F(PolarEncoderMultiSSBTest, MultiSSB_CorrectnessAgainstSingleReference)
{
    const uint16_t numSSBs = 4;
    SetupMultiSSBBuffers(numSSBs);

    // Fill per-SSB info bytes.
    const uint32_t totalInfoSize = static_cast<uint32_t>(numSSBs) * 8;
    std::vector<uint8_t> h_info(totalInfoSize, 0);
    std::mt19937         rng(97531);
    for(auto& b : h_info) b = static_cast<uint8_t>(rng() & 0xFF);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_infoBits, h_info.data(), h_info.size(), cudaMemcpyHostToDevice));

    cuphyStatus_t status = RunMultiSSBEncoder(numSSBs);
    ASSERT_EQ(CUPHY_STATUS_SUCCESS, status);

    const uint32_t codedStrideBytes = (CUPHY_SSB_N_PBCH_POLAR_ENCODED_BITS / 8);
    const uint32_t txStrideBytes    = (CUPHY_SSB_N_PBCH_SCRAMBLING_SEQ_BITS / 8);
    std::vector<uint8_t> h_coded_multi(static_cast<uint32_t>(numSSBs) * codedStrideBytes, 0);
    std::vector<uint8_t> h_tx_multi(static_cast<uint32_t>(numSSBs) * txStrideBytes, 0);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(h_coded_multi.data(), d_codedBits, h_coded_multi.size(), cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(h_tx_multi.data(), d_txBits, h_tx_multi.size(), cudaMemcpyDeviceToHost));

    // Reference buffers reused per SSB.
    uint8_t* d_info_ref  = nullptr;
    uint8_t* d_coded_ref = nullptr;
    uint8_t* d_tx_ref    = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_info_ref, 8));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_coded_ref, codedStrideBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_tx_ref, txStrideBytes));

    std::vector<uint8_t> h_coded_ref(codedStrideBytes, 0);
    std::vector<uint8_t> h_tx_ref(txStrideBytes, 0);

    for(uint16_t ssb = 0; ssb < numSSBs; ++ssb)
    {
        ASSERT_EQ(cudaSuccess, cudaMemset(d_coded_ref, 0, codedStrideBytes));
        ASSERT_EQ(cudaSuccess, cudaMemset(d_tx_ref, 0, txStrideBytes));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(d_info_ref, &h_info[ssb * 8], 8, cudaMemcpyHostToDevice));

        uint32_t nCodedBitsRef = 0;
        cuphyStatus_t st = cuphyPolarEncRateMatch(
            CUPHY_SSB_N_PBCH_SEQ_W_CRC_BITS,
            CUPHY_SSB_N_PBCH_SCRAMBLING_SEQ_BITS,
            d_info_ref,
            &nCodedBitsRef,
            d_coded_ref,
            d_tx_ref,
            /*procModeBmsk*/ 0,
            cuStream);
        ASSERT_EQ(CUPHY_STATUS_SUCCESS, st) << "Reference API failed for SSB=" << ssb;
        CUDA_CHECK(cudaStreamSynchronize(cuStream));

        const uint32_t codedBytesRef = (nCodedBitsRef + 7) / 8;
        ASSERT_EQ(cudaSuccess, cudaMemcpy(h_coded_ref.data(), d_coded_ref, codedBytesRef, cudaMemcpyDeviceToHost));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(h_tx_ref.data(), d_tx_ref, txStrideBytes, cudaMemcpyDeviceToHost));

        const uint8_t* coded_multi_ptr = &h_coded_multi[static_cast<uint32_t>(ssb) * codedStrideBytes];
        const uint8_t* tx_multi_ptr    = &h_tx_multi[static_cast<uint32_t>(ssb) * txStrideBytes];

        ExpectBuffersEqual(h_coded_ref.data(), coded_multi_ptr, codedBytesRef, "ssb codedBits", ssb);
        ExpectBuffersEqual(h_tx_ref.data(), tx_multi_ptr, txStrideBytes, "ssb txBits", ssb);
    }

    cudaFree(d_info_ref);
    cudaFree(d_coded_ref);
    cudaFree(d_tx_ref);

    FreeTestData();
}

// Error handling test
TEST_F(PolarEncoderTest, ErrorHandlingTest) {
    // Test invalid arguments
    uint32_t nCodedBits = 0;

    // Test with null pointers
    cuphyStatus_t status = cuphyPolarEncRateMatch(
        100, 200, nullptr, &nCodedBits, nullptr, nullptr, 0, cuStream);
    EXPECT_NE(CUPHY_STATUS_SUCCESS, status);

    // Test with zero info bits
    SetupTestBuffers({"Test", 100, 200, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0});
    status = cuphyPolarEncRateMatch(
        0, 200, d_infoBits, &nCodedBits, d_codedBits, d_txBits, 0, cuStream);
    EXPECT_NE(CUPHY_STATUS_SUCCESS, status);

    // Test with zero tx bits
    status = cuphyPolarEncRateMatch(
        100, 0, d_infoBits, &nCodedBits, d_codedBits, d_txBits, 0, cuStream);
    EXPECT_NE(CUPHY_STATUS_SUCCESS, status);

    // Test with excessive info bits
    status = cuphyPolarEncRateMatch(
        CUPHY_POLAR_ENC_MAX_INFO_BITS + 1, 200, d_infoBits, &nCodedBits, d_codedBits, d_txBits, 0, cuStream);
    EXPECT_NE(CUPHY_STATUS_SUCCESS, status);

    // Test with excessive tx bits
    status = cuphyPolarEncRateMatch(
        100, CUPHY_POLAR_ENC_MAX_TX_BITS + 1, d_infoBits, &nCodedBits, d_codedBits, d_txBits, 0, cuStream);
    EXPECT_NE(CUPHY_STATUS_SUCCESS, status);

    FreeTestData();
}

// Alignment test
TEST_F(PolarEncoderTest, AlignmentTest) {
    // Test with properly aligned buffers
    PolarEncoderTestConfig config = {
        "Alignment test", 100, 200, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0
    };

    SetupTestBuffers(config);
    SetupInfoBits(config, InfoBitPattern::RANDOM);

    // Verify buffer alignment
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(d_infoBits) & 0x3) << "Info bits buffer not 32-bit aligned";
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(d_codedBits) & 0x3) << "Coded bits buffer not 32-bit aligned";
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(d_txBits) & 0x3) << "TX bits buffer not 32-bit aligned";

    cuphyStatus_t status = RunPolarEncoder(config);
    EXPECT_EQ(CUPHY_STATUS_SUCCESS, status);

    FreeTestData();
}

// Stress test with maximum parameters
TEST_F(PolarEncoderTest, StressTest) {
    std::vector<PolarEncoderTestConfig> stressConfigs = {
        {"Stress max info", CUPHY_POLAR_ENC_MAX_INFO_BITS, CUPHY_POLAR_ENC_MAX_TX_BITS, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Stress max info UL", CUPHY_POLAR_ENC_MAX_INFO_BITS, CUPHY_POLAR_ENC_MAX_TX_BITS, 1, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Stress high repetition", 64, CUPHY_POLAR_ENC_MAX_TX_BITS, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0},
        {"Stress high puncturing", CUPHY_POLAR_ENC_MAX_INFO_BITS, 600, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0}
    };

    for(const auto& config : stressConfigs) {
        TestConfiguration(config, InfoBitPattern::RANDOM);
    }
}

// Performance consistency test
TEST_F(PolarEncoderTest, PerformanceConsistencyTest) {
    PolarEncoderTestConfig config = {
        "Performance test", 64, 512, 0, CUPHY_STATUS_SUCCESS, true, false, false, 0, 0
    };

    // Run the same configuration multiple times to check consistency
    for(int i = 0; i < 10; i++) {
        TestConfiguration(config, InfoBitPattern::RANDOM, i);  // Different seed each time
    }
}

// main()
int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    return result;
}
