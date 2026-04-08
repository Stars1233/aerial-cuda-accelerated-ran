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
#include <vector>
#include <memory>
#include <string>

// Test fixture for CFO TA EST testing
class CfoTaEstTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        // Create CUDA stream
        ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&cuStream, cudaStreamNonBlocking));
    }

    void TearDown() override
    {
        // Free any remaining resources
        FreeResources();

        // Destroy CUDA stream
        cudaStreamDestroy(cuStream);
        cudaDeviceSynchronize();
    }

    // Helper method to test specific CFO TA EST configurations
    void TestConfiguration(const char* testName, uint8_t nLayers, uint8_t nRxAnts, uint8_t dmrsAddlnPos, cuphyDataType_t hEstType, cuphyDataType_t cfoEstType, cuphyStatus_t expectedStatus)
    {
        const uint16_t nUeGrps = 1;
        const uint32_t nMaxPrb = 273; // MAX_N_PRBS_SUPPORTED

        printf("Running configuration test: %s\n", testName);

        // Setup CFO TA EST
        SetupCfoTaEst(nUeGrps, nMaxPrb);

        // Configure UE group parameters
        ConfigureUeGroup(0, 1, nLayers, 50, dmrsAddlnPos);
        ConfigureBsAntennas(0, nRxAnts);
        SetElementTypes(0, hEstType, cfoEstType);

        // Run CFO TA EST setup
        cuphyStatus_t status = RunCfoTaEst(nUeGrps, nMaxPrb);

        // Verify expected status
        EXPECT_EQ(expectedStatus, status);
        // Free resources for this configuration
        FreeResources();
    }

    // Helper function to test a range of configurations with different layers and RX antennas
    void TestConfigurationRange(const std::vector<uint8_t>& nLayersNum,
                                const std::vector<uint8_t>& nRxAntsNum,
                                uint8_t                     dmrsAddlnPos   = 1,
                                cuphyDataType_t             hEstType       = CUPHY_C_32F,
                                cuphyDataType_t             cfoEstType     = CUPHY_C_32F,
                                cuphyStatus_t               expectedStatus = CUPHY_STATUS_SUCCESS)
    {
        for(const auto& nRxAnt : nRxAntsNum)
        {
            for(const auto& nLayer : nLayersNum)
            {
                // Skip invalid configurations where layers exceed RX antennas
                if(nLayer > nRxAnt)
                {
                    continue;
                }

                std::string testName = " nBSAnts == " + std::to_string(nRxAnt) +
                                       " && nLayers == " + std::to_string(nLayer);

                TestConfiguration(testName.c_str(), nLayer, nRxAnt, dmrsAddlnPos, hEstType, cfoEstType, expectedStatus);
            }
        }
    }

    // Common setup function for CFO TA EST tests
    void SetupCfoTaEst(uint16_t nUeGrps, uint32_t nMaxPrb)
    {
        // Get descriptor sizes
        ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphyPuschRxCfoTaEstGetDescrInfo(&statDescrSizeBytes, &statDescrAlignBytes, &dynDescrSizeBytes, &dynDescrAlignBytes));

        // Allocate buffers for descriptors
        ASSERT_EQ(cudaSuccess, cudaMallocHost(&statDescrBufCpu, statDescrSizeBytes));
        ASSERT_EQ(cudaSuccess, cudaMallocHost(&dynDescrBufCpu, dynDescrSizeBytes));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&statDescrBufGpu, statDescrSizeBytes));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&dynDescrBufGpu, dynDescrSizeBytes));

        // Create handle
        ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphyCreatePuschRxCfoTaEst(&puschRxCfoTaEstHndl, static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy), statDescrBufCpu, statDescrBufGpu, cuStream));

        // Copy static descriptors from CPU to GPU if not using async copy
        if(!enableCpuToGpuDescrAsyncCpy)
        {
            ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(statDescrBufGpu, statDescrBufCpu, statDescrSizeBytes, cudaMemcpyHostToDevice, cuStream));
            ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(cuStream));
        }

        // Create UeGrp parameters
        ASSERT_EQ(cudaSuccess, cudaMallocHost(&drvdUeGrpPrmsCpu, nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t)));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&drvdUeGrpPrmsGpu, nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t)));

        // Initialize UeGrp parameters to zero
        memset(drvdUeGrpPrmsCpu, 0, nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t));

        // Allocate a minimal, non-null FO compensation buffer pointer array.
        // Note: the public API wrapper rejects a null pFoCompensationBuffers argument.
        ASSERT_EQ(cudaSuccess, cudaMallocHost(&foCompensationBuffersCpu, sizeof(float*)));
        foCompensationBuffersCpu[0] = nullptr;
    }

    // Configure UE group parameters for tests
    void ConfigureUeGroup(int index, uint8_t nUes, uint8_t nLayers, uint16_t nPrb, uint8_t dmrsAddlnPos = 0)
    {
        drvdUeGrpPrmsCpu[index].nUes         = nUes;
        drvdUeGrpPrmsCpu[index].nLayers      = nLayers;
        drvdUeGrpPrmsCpu[index].nPrb         = nPrb;
        drvdUeGrpPrmsCpu[index].dmrsAddlnPos = dmrsAddlnPos;
    }

    // Configure RX antennas
    void ConfigureRxAntennas(int index, uint8_t nRxAnt)
    {
        drvdUeGrpPrmsCpu[index].nRxAnt = nRxAnt;
    }

    // Set tensor element types
    void SetElementTypes(int index, cuphyDataType_t hEstType, cuphyDataType_t cfoEstType)
    {
        drvdUeGrpPrmsCpu[index].tInfoHEst.elemType   = hEstType;
        drvdUeGrpPrmsCpu[index].tInfoCfoEst.elemType = cfoEstType;
    }

    // Configure BS (base station) antennas
    void ConfigureBsAntennas(int index, uint8_t nBSAnts)
    {
        drvdUeGrpPrmsCpu[index].nRxAnt = nBSAnts;
    }

    // Run CFO TA EST setup
    cuphyStatus_t RunCfoTaEst(uint16_t nUeGrps, uint32_t nMaxPrb, uint8_t subSlotStage = 0)
    {
        // Initialize launch configs
        cfoEstLaunchCfgs.nCfgs = 1;

        // Run the setup function
        cuphyStatus_t status = cuphySetupPuschRxCfoTaEst(
            puschRxCfoTaEstHndl,
            drvdUeGrpPrmsCpu,
            drvdUeGrpPrmsGpu,
            foCompensationBuffersCpu, // pFoCompensationBuffers (must be non-null)
            nUeGrps,
            nMaxPrb,
            nullptr, // pDbg parameter - set to nullptr
            enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
            dynDescrBufCpu,
            dynDescrBufGpu,
            &cfoEstLaunchCfgs,
            cuStream);

        // If setup was successful and we have launch configurations, launch the kernel
        if(status == CUPHY_STATUS_SUCCESS && cfoEstLaunchCfgs.nCfgs > 0)
        {
            // Copy data to GPU if not using async copy
            if(!enableCpuToGpuDescrAsyncCpy)
            {
                cudaError_t cudaStatus;

                // Copy UE group parameters to GPU
                cudaStatus = cudaMemcpyAsync(drvdUeGrpPrmsGpu, drvdUeGrpPrmsCpu, nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t), cudaMemcpyHostToDevice, cuStream);
                if(cudaStatus != cudaSuccess)
                {
                    return CUPHY_STATUS_INTERNAL_ERROR;
                }

                // Copy dynamic descriptor buffer to GPU
                cudaStatus = cudaMemcpyAsync(dynDescrBufGpu, dynDescrBufCpu, dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStream);
                if(cudaStatus != cudaSuccess)
                {
                    return CUPHY_STATUS_INTERNAL_ERROR;
                }

                // Synchronize stream to ensure memory transfers are complete
                cudaStatus = cudaStreamSynchronize(cuStream);
                if(cudaStatus != cudaSuccess)
                {
                    return CUPHY_STATUS_INTERNAL_ERROR;
                }
            }

            // Launch kernel using the CUDA driver API
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = cfoEstLaunchCfgs.cfgs[0].kernelNodeParamsDriver;
            CUresult                       cfoEstRunStatus        = cuLaunchKernel(
                kernelNodeParamsDriver.func,
                kernelNodeParamsDriver.gridDimX,
                kernelNodeParamsDriver.gridDimY,
                kernelNodeParamsDriver.gridDimZ,
                kernelNodeParamsDriver.blockDimX,
                kernelNodeParamsDriver.blockDimY,
                kernelNodeParamsDriver.blockDimZ,
                kernelNodeParamsDriver.sharedMemBytes,
                static_cast<CUstream>(cuStream),
                kernelNodeParamsDriver.kernelParams,
                kernelNodeParamsDriver.extra);

            // Check if kernel launch was successful
            if(cfoEstRunStatus != CUDA_SUCCESS)
            {
                return CUPHY_STATUS_INTERNAL_ERROR;
            }

            // Synchronize to ensure kernel execution is complete
            cudaError_t cudaStatus = cudaStreamSynchronize(cuStream);
            if(cudaStatus != cudaSuccess)
            {
                return CUPHY_STATUS_INTERNAL_ERROR;
            }
        }

        return status;
    }

    // Cleanup and free resources
    void FreeResources()
    {
        // Destroy handle if created
        if(puschRxCfoTaEstHndl)
        {
            cuphyDestroyPuschRxCfoTaEst(puschRxCfoTaEstHndl);
            puschRxCfoTaEstHndl = nullptr;
        }

        // Free CPU memory
        if(statDescrBufCpu)
        {
            cudaFreeHost(statDescrBufCpu);
            statDescrBufCpu = nullptr;
        }

        if(dynDescrBufCpu)
        {
            cudaFreeHost(dynDescrBufCpu);
            dynDescrBufCpu = nullptr;
        }

        if(drvdUeGrpPrmsCpu)
        {
            cudaFreeHost(drvdUeGrpPrmsCpu);
            drvdUeGrpPrmsCpu = nullptr;
        }

        if(foCompensationBuffersCpu)
        {
            cudaFreeHost(foCompensationBuffersCpu);
            foCompensationBuffersCpu = nullptr;
        }

        // Free GPU memory
        if(statDescrBufGpu)
        {
            cudaFree(statDescrBufGpu);
            statDescrBufGpu = nullptr;
        }

        if(dynDescrBufGpu)
        {
            cudaFree(dynDescrBufGpu);
            dynDescrBufGpu = nullptr;
        }

        if(drvdUeGrpPrmsGpu)
        {
            cudaFree(drvdUeGrpPrmsGpu);
            drvdUeGrpPrmsGpu = nullptr;
        }

        // Synchronize stream
        cudaStreamSynchronize(cuStream);
        cudaDeviceSynchronize();
    }

    // Test parameters and resources
    cudaStream_t cuStream;
    bool         enableCpuToGpuDescrAsyncCpy = false;

    // Descriptor information
    size_t statDescrSizeBytes  = 0;
    size_t statDescrAlignBytes = 0;
    size_t dynDescrSizeBytes   = 0;
    size_t dynDescrAlignBytes  = 0;

    // Memory resources
    void*                    statDescrBufCpu  = nullptr;
    void*                    dynDescrBufCpu   = nullptr;
    void*                    statDescrBufGpu  = nullptr;
    void*                    dynDescrBufGpu   = nullptr;
    cuphyPuschRxUeGrpPrms_t* drvdUeGrpPrmsCpu = nullptr;
    cuphyPuschRxUeGrpPrms_t* drvdUeGrpPrmsGpu = nullptr;
    float**                  foCompensationBuffersCpu = nullptr;

    // Handle and launch configs
    cuphyPuschRxCfoTaEstHndl_t       puschRxCfoTaEstHndl = nullptr;
    cuphyPuschRxCfoTaEstLaunchCfgs_t cfoEstLaunchCfgs;
};

// Basic test for CFO TA EST functionality
TEST_F(CfoTaEstTest, BasicCfoTaEstTest)
{
    TestConfiguration("Basic CFO TA EST functionality", 1, 4, 0, CUPHY_C_32F, CUPHY_C_32F, CUPHY_STATUS_SUCCESS);
}

// Test for various data type formats in CFO TA EST
TEST_F(CfoTaEstTest, DataTypeFormatTest)
{
    // Test case 1: FP16 output with FP32 input (CUPHY_C_32F == hEstType) && (CUPHY_C_16F == cfoEstType)
    TestConfiguration("FP16 output with FP32 input", 1, 4, 1, CUPHY_C_32F, CUPHY_C_16F, CUPHY_STATUS_SUCCESS);

    // Test case 2: Both HEst and CfoEst in FP16 format (CUPHY_C_16F == hEstType) && (CUPHY_C_16F == cfoEstType)
    TestConfiguration("Both HEst and CfoEst in FP16 format", 1, 4, 1, CUPHY_C_16F, CUPHY_C_16F, CUPHY_STATUS_SUCCESS);

    // Test case 3: Both HEst and CfoEst in FP32 format (CUPHY_C_32F == hEstType) && (CUPHY_C_32F == cfoEstType)
    TestConfiguration("Both HEst and CfoEst in FP32 format", 1, 4, 1, CUPHY_C_32F, CUPHY_C_32F, CUPHY_STATUS_SUCCESS);
}

// Test for error case when number of layers exceeds number of RX antennas
TEST_F(CfoTaEstTest, LayersExceedRxAntennasErrorTest)
{
    TestConfiguration("Layers exceed RX antennas", 4, 2, 1, CUPHY_C_32F, CUPHY_C_32F, CUPHY_STATUS_INTERNAL_ERROR);
}

// Test for case when nDmrsAddlnPos is not 1, 2, or 3 for kernelSelectL1 function
TEST_F(CfoTaEstTest, DmrsAddlnPosErrorTest)
{
    // Test name | nLayers | nRxAnts | dmrsAddlnPos | hEstType | cfoEstType | expectedStatus
    TestConfiguration("nDmrsAddlnPos is not 1, 2, or 3", 1, 4, 4, CUPHY_C_32F, CUPHY_C_32F, CUPHY_STATUS_SUCCESS);
}

// Test for case when nBSAnts > 8 or nBSAnts < 1 for kernelSelectL0 function
TEST_F(CfoTaEstTest, BsAntsErrorTest)
{
    // Test name | nLayers | nRxAnts | dmrsAddlnPos | hEstType | cfoEstType | expectedStatus
    TestConfiguration("nBSAnts > 8", 1, 16, 1, CUPHY_C_32F, CUPHY_C_32F, CUPHY_STATUS_SUCCESS);
    TestConfiguration("nBSAnts < 1", 1, 0, 1, CUPHY_C_32F, CUPHY_C_32F, CUPHY_STATUS_SUCCESS);
}

// Test for case when nBSAnts == 8 with different numbers of layers for kernelSelectL0 function
TEST_F(CfoTaEstTest, TestNBSAnts8WithDifferentLayers)
{
    // Test for nBSAnts from 1 to 8 with all valid layer combinations
    std::vector<uint8_t> allAntennas = {8, 7, 6, 5, 4, 3, 2, 1};
    
    for (const auto& nRxAnts : allAntennas) {
        // For each antenna count, test all valid layer counts (0 to nRxAnts)
        std::vector<uint8_t> layerValues;
        for (uint8_t i = 0; i <= nRxAnts; i++) {
            layerValues.push_back(i);
        }
        
        TestConfigurationRange(layerValues, {nRxAnts});
    }
}

// Comprehensive test case to cover all branches and statements in cfoTaEstLowMimoKernel_v1
TEST_F(CfoTaEstTest, CfoTaEstLowMimoKernelComprehensiveTest)
{
    // Test name | nLayers | nRxAnts | dmrsAddlnPos | hEstType | cfoEstType | expectedStatus

    // 1. Test CFO estimation with different number of time domain channel estimates (N_TIME_CH_EST)
    // This is controlled by dmrsAddlnPos (1, 2, or 3)
    TestConfiguration("N_TIME_CH_EST=2 (dmrsAddlnPos=1)", 1, 4, 1, CUPHY_C_32F, CUPHY_C_32F, CUPHY_STATUS_SUCCESS);
    TestConfiguration("N_TIME_CH_EST=3 (dmrsAddlnPos=2)", 1, 4, 2, CUPHY_C_32F, CUPHY_C_32F, CUPHY_STATUS_SUCCESS);
    TestConfiguration("N_TIME_CH_EST=4 (dmrsAddlnPos=3)", 1, 4, 3, CUPHY_C_32F, CUPHY_C_32F, CUPHY_STATUS_SUCCESS);

    // 2. Test different BS antenna and layer combinations to cover thread tile configurations
    // Various N_BS_ANTS and N_LAYERS combinations
    // Covering cases from 1 to 8 BS antennas with various layer configurations

    // Full 8x8 MIMO
    TestConfiguration("8x8 MIMO configuration", 8, 8, 1, CUPHY_C_32F, CUPHY_C_32F, CUPHY_STATUS_SUCCESS);

    // Test all possible combinations of BS antennas and layers
    // These cover different thread blocks, tiles, and memory access patterns in the kernel
    TestConfiguration("1x1 MIMO configuration", 1, 1, 1, CUPHY_C_32F, CUPHY_C_32F, CUPHY_STATUS_SUCCESS);
    TestConfiguration("4x2 MIMO configuration", 2, 4, 1, CUPHY_C_32F, CUPHY_C_32F, CUPHY_STATUS_SUCCESS);
    TestConfiguration("4x4 MIMO configuration", 4, 4, 1, CUPHY_C_32F, CUPHY_C_32F, CUPHY_STATUS_SUCCESS);
    TestConfiguration("8x4 MIMO configuration", 4, 8, 1, CUPHY_C_32F, CUPHY_C_32F, CUPHY_STATUS_SUCCESS);

    // 3. Test with FP16 types to cover data type conversions in the kernel
    // Tests expansion from __half2 to full precision types
    TestConfiguration("FP16 expansion test", 2, 4, 1, CUPHY_C_16F, CUPHY_C_16F, CUPHY_STATUS_SUCCESS);

    // 4. Cover edge cases for shared memory and thread synchronization
    // Maximum PRB allocation to stress the kernel's pipeline and memory access patterns
    TestConfiguration("Maximum PRB allocation", 4, 8, 1, CUPHY_C_32F, CUPHY_C_32F, CUPHY_STATUS_SUCCESS);

    // 5. Test cases to maximize parallel execution and thread divergence
    // Odd number of antennas and layers to test thread divergence paths
    TestConfiguration("Thread divergence test 7x5", 5, 7, 1, CUPHY_C_32F, CUPHY_C_32F, CUPHY_STATUS_SUCCESS);
    TestConfiguration("Thread divergence test 5x3", 3, 5, 1, CUPHY_C_32F, CUPHY_C_32F, CUPHY_STATUS_SUCCESS);

    // 6. Testing with both CFO and TA estimation enabled
    // This covers the full execution path in the kernel
    TestConfiguration("Full CFO and TA estimation", 2, 4, 2, CUPHY_C_32F, CUPHY_C_32F, CUPHY_STATUS_SUCCESS);
}

// main()
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    return result;
}
