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

#include "ldpc/ldpc_cb_kernel_api.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr uint16_t kZc = 384;
constexpr uint16_t kKb = 22;
constexpr uint16_t kInfoBits = kKb * kZc;
constexpr uint32_t kLlrsPerCodeblock = 66U * kZc;
constexpr uint32_t kLlrBytesPerCodeblock = kLlrsPerCodeblock * sizeof(uint16_t);
constexpr uint32_t kBitsWordsPerCodeblock = (kInfoBits + 31U) / 32U;
constexpr uint32_t kBitsBytesPerCodeblock = kBitsWordsPerCodeblock * sizeof(uint32_t);

cuphyLdpcCbSubgroupKey_t valid_key()
{
    return cuphyLdpcCbSubgroupKey_t{
        CUPHY_LDPC_CB_BG1,
        kZc,
        8,
        kInfoBits,
        CUPHY_LDPC_CB_CRC_NONE,
        10,
        0,
    };
}

cuphyLdpcCbLaunchStaticConfig_t valid_static_config()
{
    return cuphyLdpcCbLaunchStaticConfig_t{
        CUPHY_R_16F,
        31.0f,
        CUPHY_LDPC_CB_THROUGHPUT_MODE,
    };
}

cuphyLdpcCbPreparedLaunchConfig_t valid_prepared_config()
{
    return cuphyLdpcCbPreparedLaunchConfig_t{
        1,
        8,
    };
}

cuphyLdpcCbBufferSpan_t valid_span()
{
    alignas(16) static std::array<uint8_t, kLlrBytesPerCodeblock * 2> llrs{};
    alignas(16) static std::array<uint32_t, kBitsWordsPerCodeblock * 2> bits{};

    return cuphyLdpcCbBufferSpan_t{
        llrs.data(),
        kLlrBytesPerCodeblock,
        bits.data(),
        0,
        kBitsBytesPerCodeblock,
        nullptr,
        0,
        2,
    };
}

cuphyLdpcCbSubgroupDesc_t valid_subgroup(const cuphyLdpcCbBufferSpan_t* spans)
{
    return cuphyLdpcCbSubgroupDesc_t{
        valid_key(),
        spans,
        1,
    };
}

cuphyLdpcCbLaunchBatchDesc_t valid_batch(const cuphyLdpcCbSubgroupDesc_t* subgroup)
{
    return cuphyLdpcCbLaunchBatchDesc_t{
        subgroup,
        1,
        nullptr,
    };
}

bool has_cuda_device()
{
    int device_count = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count == 0)
    {
        return false;
    }
    return true;
}

void* align_pointer(void* ptr, const size_t alignment)
{
    const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    const auto aligned = (addr + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1);
    return reinterpret_cast<void*>(aligned);
}

class LdpcCbPrepareLaunchCudaTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!has_cuda_device())
        {
            GTEST_SKIP() << "CUDA device required for LDPC prepare-launch setup";
        }

        const auto config = valid_static_config();
        ASSERT_EQ(cuphyCreateLdpcCbLaunchPreparer(&preparer_, &config), CUPHY_STATUS_SUCCESS);
        const auto key = valid_key();
        ASSERT_EQ(cuphyLdpcCbQueryLaunchFamily(preparer_, &key, &family_), CUPHY_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if (launch_ != nullptr)
        {
            EXPECT_EQ(cuphyLdpcCbPreparedLaunchDeinit(launch_), CUPHY_STATUS_SUCCESS);
            launch_ = nullptr;
        }
        if (preparer_ != nullptr)
        {
            EXPECT_EQ(cuphyDestroyLdpcCbLaunchPreparer(preparer_), CUPHY_STATUS_SUCCESS);
            preparer_ = nullptr;
        }
    }

    cuphyStatus_t init_launch(const cuphyLdpcCbPreparedLaunchConfig_t& config)
    {
        size_t workspace_size = 0;
        size_t workspace_alignment = 0;
        const auto size_status = cuphyLdpcCbPreparedLaunchGetWorkspaceSize(preparer_,
                                                                           &family_,
                                                                           &config,
                                                                           &workspace_size,
                                                                           &workspace_alignment);
        if (size_status != CUPHY_STATUS_SUCCESS)
        {
            return size_status;
        }

        workspace_.assign(workspace_size + workspace_alignment, 0);
        void* workspace = align_pointer(workspace_.data(), workspace_alignment);
        return cuphyLdpcCbPreparedLaunchInitInPlace(preparer_,
                                                    &family_,
                                                    &config,
                                                    workspace,
                                                    workspace_size,
                                                    &launch_);
    }

    cuphyLdpcCbLaunchPreparer_t preparer_{};
    cuphyLdpcCbLaunchFamily_t family_{};
    cuphyLdpcCbPreparedLaunch_t launch_{};
    std::vector<uint8_t> workspace_;
};

} // namespace

TEST(LdpcCbPrepareLaunchContract, CreateRejectsNullOutputHandle)
{
    const auto config = valid_static_config();
    EXPECT_EQ(cuphyCreateLdpcCbLaunchPreparer(nullptr, &config), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbPrepareLaunchContract, CreateRejectsNullConfig)
{
    cuphyLdpcCbLaunchPreparer_t preparer = nullptr;
    EXPECT_EQ(cuphyCreateLdpcCbLaunchPreparer(&preparer, nullptr), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbPrepareLaunchContract, CreateRejectsInvalidConfig)
{
    auto config = valid_static_config();
    config.llr_type = static_cast<cuphyDataType_t>(999);

    cuphyLdpcCbLaunchPreparer_t preparer = nullptr;
    EXPECT_EQ(cuphyCreateLdpcCbLaunchPreparer(&preparer, &config), CUPHY_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(preparer, nullptr);
}

TEST(LdpcCbPrepareLaunchContract, CreateRejectsUnsupportedEarlyTerminationFlag)
{
    auto config = valid_static_config();
    config.config_flags |= CUPHY_LDPC_CB_EARLY_TERM;

    cuphyLdpcCbLaunchPreparer_t preparer = nullptr;
    EXPECT_EQ(cuphyCreateLdpcCbLaunchPreparer(&preparer, &config), CUPHY_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(preparer, nullptr);
}

TEST(LdpcCbPrepareLaunchContract, CreateRejectsUnknownConfigFlags)
{
    auto config = valid_static_config();
    config.config_flags |= (1U << 31);

    cuphyLdpcCbLaunchPreparer_t preparer = nullptr;
    EXPECT_EQ(cuphyCreateLdpcCbLaunchPreparer(&preparer, &config), CUPHY_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(preparer, nullptr);
}

TEST(LdpcCbPrepareLaunchContract, DestroyRejectsNullPreparer)
{
    EXPECT_EQ(cuphyDestroyLdpcCbLaunchPreparer(nullptr), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbPrepareLaunchContract, QueryLaunchFamilyRejectsNulls)
{
    const auto key = valid_key();
    cuphyLdpcCbLaunchFamily_t family{};

    EXPECT_EQ(cuphyLdpcCbQueryLaunchFamily(nullptr, &key, &family), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbPrepareLaunchContract, PreparedLaunchDeinitRejectsNullLaunch)
{
    EXPECT_EQ(cuphyLdpcCbPreparedLaunchDeinit(nullptr), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST_F(LdpcCbPrepareLaunchCudaTest, WorkspaceSizeRejectsNullOutput)
{
    const auto config = valid_prepared_config();
    size_t workspace_size = 0;
    size_t workspace_alignment = 0;

    EXPECT_EQ(cuphyLdpcCbPreparedLaunchGetWorkspaceSize(preparer_,
                                                        &family_,
                                                        &config,
                                                        nullptr,
                                                        &workspace_alignment),
              CUPHY_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(cuphyLdpcCbPreparedLaunchGetWorkspaceSize(preparer_,
                                                        &family_,
                                                        &config,
                                                        &workspace_size,
                                                        nullptr),
              CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST_F(LdpcCbPrepareLaunchCudaTest, QueryLaunchFamilyRejectsNullKeyAndFamily)
{
    const auto key = valid_key();
    cuphyLdpcCbLaunchFamily_t family{};

    EXPECT_EQ(cuphyLdpcCbQueryLaunchFamily(preparer_, nullptr, &family), CUPHY_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(cuphyLdpcCbQueryLaunchFamily(preparer_, &key, nullptr), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST_F(LdpcCbPrepareLaunchCudaTest, WorkspaceSizeRejectsZeroCapacity)
{
    auto config = valid_prepared_config();
    config.max_total_codeblocks = 0;

    size_t workspace_size = 0;
    size_t workspace_alignment = 0;
    EXPECT_EQ(cuphyLdpcCbPreparedLaunchGetWorkspaceSize(preparer_,
                                                        &family_,
                                                        &config,
                                                        &workspace_size,
                                                        &workspace_alignment),
              CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST_F(LdpcCbPrepareLaunchCudaTest, InitRejectsTooSmallWorkspace)
{
    const auto config = valid_prepared_config();
    size_t workspace_size = 0;
    size_t workspace_alignment = 0;
    ASSERT_EQ(cuphyLdpcCbPreparedLaunchGetWorkspaceSize(preparer_,
                                                        &family_,
                                                        &config,
                                                        &workspace_size,
                                                        &workspace_alignment),
              CUPHY_STATUS_SUCCESS);

    workspace_.assign(workspace_size + workspace_alignment, 0);
    void* workspace = align_pointer(workspace_.data(), workspace_alignment);
    EXPECT_EQ(cuphyLdpcCbPreparedLaunchInitInPlace(preparer_,
                                                   &family_,
                                                   &config,
                                                   workspace,
                                                   workspace_size - 1,
                                                   &launch_),
              CUPHY_STATUS_INSUFFICIENT_RESOURCES);
}

TEST_F(LdpcCbPrepareLaunchCudaTest, InitRejectsNullWorkspaceAndOutput)
{
    const auto config = valid_prepared_config();
    size_t workspace_size = 0;
    size_t workspace_alignment = 0;
    ASSERT_EQ(cuphyLdpcCbPreparedLaunchGetWorkspaceSize(preparer_,
                                                        &family_,
                                                        &config,
                                                        &workspace_size,
                                                        &workspace_alignment),
              CUPHY_STATUS_SUCCESS);

    workspace_.assign(workspace_size + workspace_alignment, 0);
    void* workspace = align_pointer(workspace_.data(), workspace_alignment);

    EXPECT_EQ(cuphyLdpcCbPreparedLaunchInitInPlace(preparer_,
                                                   &family_,
                                                   &config,
                                                   nullptr,
                                                   workspace_size,
                                                   &launch_),
              CUPHY_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(cuphyLdpcCbPreparedLaunchInitInPlace(preparer_,
                                                   &family_,
                                                   &config,
                                                   workspace,
                                                   workspace_size,
                                                   nullptr),
              CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST_F(LdpcCbPrepareLaunchCudaTest, InitRejectsUnalignedWorkspace)
{
    const auto config = valid_prepared_config();
    size_t workspace_size = 0;
    size_t workspace_alignment = 0;
    ASSERT_EQ(cuphyLdpcCbPreparedLaunchGetWorkspaceSize(preparer_,
                                                        &family_,
                                                        &config,
                                                        &workspace_size,
                                                        &workspace_alignment),
              CUPHY_STATUS_SUCCESS);

    workspace_.assign(workspace_size + workspace_alignment + 1, 0);
    void* workspace = static_cast<void*>(static_cast<uint8_t*>(align_pointer(workspace_.data(), workspace_alignment)) + 1);
    EXPECT_EQ(cuphyLdpcCbPreparedLaunchInitInPlace(preparer_,
                                                   &family_,
                                                   &config,
                                                   workspace,
                                                   workspace_size,
                                                   &launch_),
              CUPHY_STATUS_UNSUPPORTED_ALIGNMENT);
}

TEST_F(LdpcCbPrepareLaunchCudaTest, PrepareLaunchRejectsNullBatchDesc)
{
    ASSERT_EQ(init_launch(valid_prepared_config()), CUPHY_STATUS_SUCCESS);

    EXPECT_EQ(cuphyLdpcCbPrepareLaunch(launch_, nullptr, nullptr), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST_F(LdpcCbPrepareLaunchCudaTest, DestroyPreparerRejectsActivePreparedLaunch)
{
    ASSERT_EQ(init_launch(valid_prepared_config()), CUPHY_STATUS_SUCCESS);

    const auto status = cuphyDestroyLdpcCbLaunchPreparer(preparer_);
    EXPECT_EQ(status, CUPHY_STATUS_INVALID_ARGUMENT);

    // Avoid invoking the known invalid lifetime path in the pre-fix implementation.
    if (status == CUPHY_STATUS_SUCCESS)
    {
        launch_   = nullptr;
        preparer_ = nullptr;
    }
}

TEST_F(LdpcCbPrepareLaunchCudaTest, GetKernelNodeParamsRejectsNullAndUnprepared)
{
    ASSERT_EQ(init_launch(valid_prepared_config()), CUPHY_STATUS_SUCCESS);

    CUDA_KERNEL_NODE_PARAMS params{};
    EXPECT_EQ(cuphyLdpcCbGetKernelNodeParams(nullptr, &params), CUPHY_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(cuphyLdpcCbGetKernelNodeParams(launch_, nullptr), CUPHY_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(cuphyLdpcCbGetKernelNodeParams(launch_, &params), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST_F(LdpcCbPrepareLaunchCudaTest, PrepareLaunchRejectsExcessCodeblocks)
{
    auto config = valid_prepared_config();
    config.max_total_codeblocks = 1;
    ASSERT_EQ(init_launch(config), CUPHY_STATUS_SUCCESS);

    auto span = valid_span();
    auto subgroup = valid_subgroup(&span);
    auto batch = valid_batch(&subgroup);

    EXPECT_EQ(cuphyLdpcCbPrepareLaunch(launch_, &batch, nullptr), CUPHY_STATUS_INSUFFICIENT_RESOURCES);
}

TEST_F(LdpcCbPrepareLaunchCudaTest, PrepareLaunchReturnsPersistentNodeParams)
{
    ASSERT_EQ(init_launch(valid_prepared_config()), CUPHY_STATUS_SUCCESS);

    auto span = valid_span();
    auto subgroup = valid_subgroup(&span);
    auto batch = valid_batch(&subgroup);
    cuphyLdpcCbPreparedLaunchInfo_t info{};

    ASSERT_EQ(cuphyLdpcCbPrepareLaunch(launch_, &batch, &info), CUPHY_STATUS_SUCCESS);

    CUDA_KERNEL_NODE_PARAMS first{};
    CUDA_KERNEL_NODE_PARAMS second{};
    EXPECT_EQ(cuphyLdpcCbGetKernelNodeParams(launch_, &first), CUPHY_STATUS_SUCCESS);
    EXPECT_EQ(cuphyLdpcCbGetKernelNodeParams(launch_, &second), CUPHY_STATUS_SUCCESS);

    EXPECT_EQ(first.func, second.func);
    EXPECT_EQ(first.kernelParams, second.kernelParams);
    EXPECT_EQ(first.gridDimX, info.grid_dim_x);
    EXPECT_EQ(info.total_codewords, span.num_cb);
    EXPECT_EQ(info.num_subgroups, 1);
}

TEST_F(LdpcCbPrepareLaunchCudaTest, PrepareLaunchAcceptsPreviouslyQueriedChoice)
{
    ASSERT_EQ(init_launch(valid_prepared_config()), CUPHY_STATUS_SUCCESS);

    auto span = valid_span();
    auto subgroup = valid_subgroup(&span);
    auto batch = valid_batch(&subgroup);
    cuphyLdpcCbKernelChoice_t choice{};
    ASSERT_EQ(cuphyLdpcCbQueryKernelChoice(preparer_, &subgroup.key, &choice), CUPHY_STATUS_SUCCESS);

    EXPECT_EQ(cuphyLdpcCbPrepareLaunchWithChoice(launch_, &batch, &choice, nullptr), CUPHY_STATUS_SUCCESS);
}

TEST_F(LdpcCbPrepareLaunchCudaTest, PrepareLaunchCopiesTransientDescriptors)
{
    ASSERT_EQ(init_launch(valid_prepared_config()), CUPHY_STATUS_SUCCESS);

    auto span = valid_span();
    const void* expected_llr_base = span.llr_base;
    auto subgroup = valid_subgroup(&span);
    auto batch = valid_batch(&subgroup);

    ASSERT_EQ(cuphyLdpcCbPrepareLaunch(launch_, &batch, nullptr), CUPHY_STATUS_SUCCESS);

    span.llr_base = nullptr;
    subgroup.spans = nullptr;
    batch.subgroups = nullptr;

    CUDA_KERNEL_NODE_PARAMS params{};
    ASSERT_EQ(cuphyLdpcCbGetKernelNodeParams(launch_, &params), CUPHY_STATUS_SUCCESS);
    auto** kernel_args = static_cast<void**>(params.kernelParams);
    const auto* decode_desc = static_cast<const cuphyLDPCDecodeDesc_t*>(kernel_args[0]);
    ASSERT_NE(decode_desc, nullptr);
    EXPECT_EQ(decode_desc->llr_input[0].addr, expected_llr_base);
}

TEST_F(LdpcCbPrepareLaunchCudaTest, PrepareLaunchMatchesSpiNodeParamsForSameBatch)
{
    ASSERT_EQ(init_launch(valid_prepared_config()), CUPHY_STATUS_SUCCESS);

    auto span = valid_span();
    auto subgroup = valid_subgroup(&span);
    auto batch = valid_batch(&subgroup);

    ASSERT_EQ(cuphyLdpcCbPrepareLaunch(launch_, &batch, nullptr), CUPHY_STATUS_SUCCESS);
    CUDA_KERNEL_NODE_PARAMS prepared_params{};
    ASSERT_EQ(cuphyLdpcCbGetKernelNodeParams(launch_, &prepared_params), CUPHY_STATUS_SUCCESS);

    const auto chooser_config = cuphyLdpcCbKernelChooserConfig_t{
        valid_static_config().llr_type,
        valid_static_config().clamp_value,
        valid_static_config().config_flags,
    };
    cuphyLdpcCbKernelChooser_t chooser = nullptr;
    ASSERT_EQ(cuphyCreateLdpcCbKernelChooser(&chooser, &chooser_config), CUPHY_STATUS_SUCCESS);

    cuphyLdpcCbKernelChoice_t choice{};
    ASSERT_EQ(cuphyLdpcCbChooseKernel(chooser, &subgroup.key, &choice), CUPHY_STATUS_SUCCESS);
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{&choice, &batch};
    cuphyLdpcCbKernelNodeParams_t spi_node_params{};
    ASSERT_EQ(cuphyLdpcCbBuildKernelNodeParams(&launch_batch, &spi_node_params), CUPHY_STATUS_SUCCESS);

    EXPECT_EQ(prepared_params.func, spi_node_params.params.func);
    EXPECT_EQ(prepared_params.gridDimX, spi_node_params.params.gridDimX);
    EXPECT_EQ(prepared_params.blockDimX, spi_node_params.params.blockDimX);
    EXPECT_EQ(prepared_params.sharedMemBytes, spi_node_params.params.sharedMemBytes);

    ASSERT_NE(prepared_params.kernelParams, nullptr);
    ASSERT_NE(spi_node_params.params.kernelParams, nullptr);
    auto** prepared_args = static_cast<void**>(prepared_params.kernelParams);
    auto** spi_args = static_cast<void**>(spi_node_params.params.kernelParams);
    ASSERT_NE(prepared_args[0], nullptr);
    ASSERT_NE(spi_args[0], nullptr);
    EXPECT_EQ(prepared_args[1], spi_args[1]);

    const auto* prepared_desc = static_cast<const cuphyLDPCDecodeDesc_t*>(prepared_args[0]);
    const auto* spi_desc = static_cast<const cuphyLDPCDecodeDesc_t*>(spi_args[0]);
    EXPECT_EQ(prepared_desc->num_tbs, spi_desc->num_tbs);
    EXPECT_EQ(prepared_desc->config.llr_type, spi_desc->config.llr_type);
    EXPECT_EQ(prepared_desc->config.num_parity_nodes, spi_desc->config.num_parity_nodes);
    EXPECT_EQ(prepared_desc->config.Z, spi_desc->config.Z);
    EXPECT_EQ(prepared_desc->config.max_iterations, spi_desc->config.max_iterations);
    EXPECT_EQ(prepared_desc->config.Kb, spi_desc->config.Kb);
    EXPECT_EQ(prepared_desc->config.flags, spi_desc->config.flags);
    EXPECT_EQ(prepared_desc->config.BG, spi_desc->config.BG);
    EXPECT_EQ(prepared_desc->config.algo, spi_desc->config.algo);
    EXPECT_EQ(prepared_desc->config.clamp_value, spi_desc->config.clamp_value);

    for (int i = 0; i < prepared_desc->num_tbs; ++i)
    {
        EXPECT_EQ(prepared_desc->llr_input[i].addr, spi_desc->llr_input[i].addr);
        EXPECT_EQ(prepared_desc->llr_input[i].stride_elements, spi_desc->llr_input[i].stride_elements);
        EXPECT_EQ(prepared_desc->llr_input[i].num_codewords, spi_desc->llr_input[i].num_codewords);
        EXPECT_EQ(prepared_desc->tb_output[i].addr, spi_desc->tb_output[i].addr);
        EXPECT_EQ(prepared_desc->tb_output[i].stride_words, spi_desc->tb_output[i].stride_words);
        EXPECT_EQ(prepared_desc->tb_output[i].num_codewords, spi_desc->tb_output[i].num_codewords);
        EXPECT_EQ(prepared_desc->llr_output[i].addr, spi_desc->llr_output[i].addr);
        EXPECT_EQ(prepared_desc->llr_output[i].stride_elements, spi_desc->llr_output[i].stride_elements);
        EXPECT_EQ(prepared_desc->llr_output[i].num_codewords, spi_desc->llr_output[i].num_codewords);
    }

    EXPECT_EQ(cuphyDestroyLdpcCbKernelChooser(chooser), CUPHY_STATUS_SUCCESS);
}
