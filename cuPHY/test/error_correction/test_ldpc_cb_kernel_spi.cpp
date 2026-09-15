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
#include "ldpc/ldpc_cb_common.hpp"
#include "ldpc/ldpc_params.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace {

TEST(LdpcParams, Bg2UsesEightInfoNodesThrough560Bits)
{
    EXPECT_EQ(cuphy::ldpc::derive_Kb(2, 560), 8);
    EXPECT_EQ(cuphy::ldpc::derive_Kb(2, 561), 9);
}

cuphyLdpcCbSubgroupKey_t valid_key()
{
    return cuphyLdpcCbSubgroupKey_t{
        CUPHY_LDPC_CB_BG1,
        384,
        8,
        static_cast<uint16_t>(22 * 384),
        CUPHY_LDPC_CB_CRC_24B,
        10,
        0,
    };
}

cuphyLdpcCbBufferSpan_t valid_span()
{
    static std::array<uint8_t, 1024>  llrs{};
    static std::array<uint32_t, 256> bits{};

    return cuphyLdpcCbBufferSpan_t{
        llrs.data(),
        1024,
        bits.data(),
        0,
        128,
        nullptr,
        0,
        1,
    };
}

cuphyLDPCDecodeConfigDesc_t valid_decode_config()
{
    cuphyLDPCDecodeConfigDesc_t config{};
    config.llr_type         = CUPHY_R_16F;
    config.num_parity_nodes = 8;
    config.Z                = 384;
    config.max_iterations   = 10;
    config.Kb               = 22;
    config.flags            = CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT;
    config.BG               = CUPHY_LDPC_CB_BG1;
    config.algo             = 4;
    config.clamp_value      = 31.0f;
    config.workspace        = nullptr;
    return config;
}

cuphyLdpcCbKernelChoice_t valid_choice()
{
    cuphyLdpcCbKernelChoice_t choice{};
    choice.func                                     = reinterpret_cast<CUfunction>(0x1);
    choice.traits.block_dim_x                       = 128;
    choice.traits.block_dim_y                       = 1;
    choice.traits.block_dim_z                       = 1;
    choice.traits.shared_mem_bytes                  = 256;
    choice.traits.codewords_per_cta                 = 2;
    choice.traits.kernel_arg_count                  = 2;
    choice.traits.llr_base_alignment_bytes          = 1;
    choice.traits.bits_base_alignment_bytes         = alignof(uint32_t);
    choice.effective_algo                           = 4;
    choice.compatibility_class_id                   = 1234;
    choice.supports_heterogeneous_batch             = 0;
    choice.max_subgroups                            = 1;
    choice.key                                      = valid_key();
    choice.decode_config                            = valid_decode_config();
    choice.static_kernel_arg                        = reinterpret_cast<void*>(0x2);
    return choice;
}

cuphyLdpcCbSubgroupDesc_t valid_subgroup(const cuphyLdpcCbBufferSpan_t* spans, uint16_t num_spans)
{
    return cuphyLdpcCbSubgroupDesc_t{
        valid_key(),
        spans,
        num_spans,
    };
}

cuphyLdpcCbKernelChooserConfig_t valid_chooser_config()
{
    return cuphyLdpcCbKernelChooserConfig_t{
        CUPHY_R_16F,
        31.0f,
        CUPHY_LDPC_CB_THROUGHPUT_MODE,
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

const cuphyLDPCDecodeLaunchConfig_t& payload_launch_config(const cuphyLdpcCbKernelNodeParams_t& node_params)
{
    return *reinterpret_cast<const cuphyLDPCDecodeLaunchConfig_t*>(node_params.payload);
}

class LdpcCbKernelChooserCudaTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!has_cuda_device())
        {
            GTEST_SKIP() << "CUDA device required for LDPC kernel chooser setup";
        }

        const auto config = valid_chooser_config();
        ASSERT_EQ(cuphyCreateLdpcCbKernelChooser(&chooser_, &config), CUPHY_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if (chooser_ != nullptr)
        {
            EXPECT_EQ(cuphyDestroyLdpcCbKernelChooser(chooser_), CUPHY_STATUS_SUCCESS);
            chooser_ = nullptr;
        }
    }

    cuphyLdpcCbKernelChooser_t chooser_{};
};

} // namespace

static_assert(CUPHY_LDPC_CB_KERNEL_SPI_CONTRACT_VERSION == 1,
              "LDPC codeblock kernel SPI contract version changed");
static_assert(std::is_trivially_copyable<cuphyLdpcCbKernelChoice_t>::value,
              "Kernel choices must remain cacheable by value");

TEST(LdpcCbKernelSpiContract, DescriptorsAreAggregateInitializable)
{
    cuphyLdpcCbSubgroupKey_t key{
        CUPHY_LDPC_CB_BG1,
        384,
        8,
        static_cast<uint16_t>(22 * 384),
        CUPHY_LDPC_CB_CRC_24B,
        10,
        0,
    };

    cuphyLdpcCbBufferSpan_t span{
        nullptr,
        0,
        nullptr,
        0,
        0,
        nullptr,
        0,
        1,
    };

    cuphyLdpcCbSubgroupDesc_t subgroup{
        key,
        &span,
        1,
    };

    cuphyLdpcCbLaunchBatchDesc_t batch{
        &subgroup,
        1,
        nullptr,
    };

    EXPECT_EQ(batch.num_subgroups, 1);
    EXPECT_EQ(batch.subgroups->num_spans, 1);
    EXPECT_EQ(batch.subgroups->spans->num_cb, 1);
    EXPECT_EQ(batch.subgroups->key.bg, CUPHY_LDPC_CB_BG1);
}

TEST(LdpcCbCommon, ValidateKeyRejectsUnsupportedBaseGraph)
{
    auto key = valid_key();
    key.bg   = static_cast<cuphyLdpcCbBaseGraph_t>(3);

    EXPECT_EQ(cuphy::ldpc::cb::validate_key(key), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbCommon, ValidateKeyRejectsZeroShapeFields)
{
    auto key = valid_key();

    key.Zc = 0;
    EXPECT_EQ(cuphy::ldpc::cb::validate_key(key), CUPHY_STATUS_INVALID_ARGUMENT);

    key              = valid_key();
    key.parity_nodes = 0;
    EXPECT_EQ(cuphy::ldpc::cb::validate_key(key), CUPHY_STATUS_INVALID_ARGUMENT);

    key   = valid_key();
    key.k = 0;
    EXPECT_EQ(cuphy::ldpc::cb::validate_key(key), CUPHY_STATUS_INVALID_ARGUMENT);

    key           = valid_key();
    key.max_iters = 0;
    EXPECT_EQ(cuphy::ldpc::cb::validate_key(key), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbCommon, ValidateKeyRejectsParityNodesBelowMinimum)
{
    auto key          = valid_key();
    key.parity_nodes = CUPHY_LDPC_MIN_PARITY_NODES - 1;

    EXPECT_EQ(cuphy::ldpc::cb::validate_key(key), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbCommon, ValidateKeyRejectsParityNodesAboveBaseGraphMax)
{
    auto key          = valid_key();
    key.bg           = CUPHY_LDPC_CB_BG1;
    key.parity_nodes = CUPHY_LDPC_MAX_BG1_PARITY_NODES + 1;
    EXPECT_EQ(cuphy::ldpc::cb::validate_key(key), CUPHY_STATUS_INVALID_ARGUMENT);

    key              = valid_key();
    key.bg           = CUPHY_LDPC_CB_BG2;
    key.parity_nodes = CUPHY_LDPC_MAX_BG2_PARITY_NODES + 1;
    EXPECT_EQ(cuphy::ldpc::cb::validate_key(key), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbCommon, ValidateKeyRejectsNonIntegralKb)
{
    auto key = valid_key();
    key.k += 1;

    EXPECT_EQ(cuphy::ldpc::cb::validate_key(key), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbCommon, ValidateKeyRejectsNonStandardLiftingSize)
{
    auto key = valid_key();
    key.Zc  = 17;
    key.k   = static_cast<uint16_t>(22 * key.Zc);

    EXPECT_EQ(cuphy::ldpc::cb::validate_key(key), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbCommon, ValidateKeyRejectsIncompatibleKbForBaseGraph)
{
    auto key = valid_key();
    key.bg  = CUPHY_LDPC_CB_BG1;
    key.k   = static_cast<uint16_t>(10 * key.Zc);
    EXPECT_EQ(cuphy::ldpc::cb::validate_key(key), CUPHY_STATUS_INVALID_ARGUMENT);

    key     = valid_key();
    key.bg  = CUPHY_LDPC_CB_BG2;
    key.k   = static_cast<uint16_t>(22 * key.Zc);
    EXPECT_EQ(cuphy::ldpc::cb::validate_key(key), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbCommon, ValidateKeyRejectsInvalidCrcType)
{
    auto key     = valid_key();
    key.crc_type = static_cast<cuphyLdpcCbCrc_t>(4);

    EXPECT_EQ(cuphy::ldpc::cb::validate_key(key), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbCommon, ValidateBufferSpanRejectsNullLlrBase)
{
    auto span     = valid_span();
    span.llr_base = nullptr;

    EXPECT_EQ(cuphy::ldpc::cb::validate_buffer_span(span), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbCommon, ValidateBufferSpanRejectsNullBitsBase)
{
    auto span      = valid_span();
    span.bits_base = nullptr;

    EXPECT_EQ(cuphy::ldpc::cb::validate_buffer_span(span), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbCommon, ValidateBufferSpanRejectsUnalignedBitsBase)
{
    static std::array<uint8_t, 64> storage{};
    alignas(16) std::array<uint8_t, 4096> llrs{};
    std::array<uint32_t, 256> bits{};
    cuphyLdpcCbBufferSpan_t span{
        llrs.data(),
        1024,
        bits.data(),
        0,
        128,
        nullptr,
        0,
        1,
    };
    for (std::size_t offset = 0; offset < alignof(uint32_t); ++offset)
    {
        auto* candidate = storage.data() + offset;
        if (reinterpret_cast<std::uintptr_t>(candidate) % alignof(uint32_t) != 0)
        {
            span.bits_base = reinterpret_cast<uint32_t*>(candidate);
            break;
        }
    }

    EXPECT_EQ(cuphy::ldpc::cb::validate_buffer_span(span), CUPHY_STATUS_UNSUPPORTED_ALIGNMENT);
}

TEST(LdpcCbCommon, ValidateBufferSpanAcceptsWordAlignedByteOffset)
{
    auto span               = valid_span();
    span.bits_offset_bytes  = sizeof(uint32_t);
    span.bits_stride_bytes += sizeof(uint32_t);

    EXPECT_EQ(cuphy::ldpc::cb::validate_buffer_span(span), CUPHY_STATUS_SUCCESS);
}

TEST(LdpcCbCommon, ValidateBufferSpanRejectsZeroStridesForMultiCodeblockSpan)
{
    auto span               = valid_span();
    span.num_cb            = 2;
    span.llr_stride_bytes  = 0;
    span.bits_stride_bytes = 64;
    EXPECT_EQ(cuphy::ldpc::cb::validate_buffer_span(span), CUPHY_STATUS_INVALID_ARGUMENT);

    span                   = valid_span();
    span.num_cb            = 2;
    span.llr_stride_bytes  = 256;
    span.bits_stride_bytes = 0;
    EXPECT_EQ(cuphy::ldpc::cb::validate_buffer_span(span), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbCommon, ValidateBufferSpanRejectsZeroSoftOutputStrideForMultiCodeblockSpan)
{
    static std::array<uint8_t, 1024> llr_out{};
    auto span                    = valid_span();
    span.num_cb                 = 2;
    span.llr_stride_bytes       = 256;
    span.bits_stride_bytes      = 64;
    span.llr_out                = llr_out.data();
    span.llr_out_stride_elems   = 0;

    EXPECT_EQ(cuphy::ldpc::cb::validate_buffer_span(span), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbCommon, ValidateSubgroupDescRejectsZeroCodeblocks)
{
    auto span     = valid_span();
    span.num_cb   = 0;
    auto subgroup = valid_subgroup(&span, 1);

    EXPECT_EQ(cuphy::ldpc::cb::validate_subgroup_desc(subgroup), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbCommon, ValidateSubgroupDescRejectsMissingSpans)
{
    auto subgroup = valid_subgroup(nullptr, 1);
    EXPECT_EQ(cuphy::ldpc::cb::validate_subgroup_desc(subgroup), CUPHY_STATUS_INVALID_ARGUMENT);

    alignas(16) std::array<uint8_t, 4096> llrs{};
    std::array<uint32_t, 256> bits{};
    cuphyLdpcCbBufferSpan_t span{
        llrs.data(),
        1024,
        bits.data(),
        0,
        128,
        nullptr,
        0,
        1,
    };
    subgroup  = valid_subgroup(&span, 0);
    EXPECT_EQ(cuphy::ldpc::cb::validate_subgroup_desc(subgroup), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbCommon, ValidateLaunchBatchDescRejectsMissingSubgroups)
{
    cuphyLdpcCbLaunchBatchDesc_t batch{
        nullptr,
        1,
        nullptr,
    };
    EXPECT_EQ(cuphy::ldpc::cb::validate_launch_batch_desc(batch), CUPHY_STATUS_INVALID_ARGUMENT);

    auto span     = valid_span();
    auto subgroup = valid_subgroup(&span, 1);
    batch         = cuphyLdpcCbLaunchBatchDesc_t{
        &subgroup,
        0,
        nullptr,
    };
    EXPECT_EQ(cuphy::ldpc::cb::validate_launch_batch_desc(batch), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbCommon, ValidateLaunchBatchDescRejectsMultipleSubgroupsInV1)
{
    alignas(16) std::array<uint8_t, 4096> aligned_llrs{};
    std::array<uint32_t, 256> bits{};
    cuphyLdpcCbBufferSpan_t span{
        aligned_llrs.data(),
        1024,
        bits.data(),
        0,
        128,
        nullptr,
        0,
        1,
    };
    std::array<cuphyLdpcCbSubgroupDesc_t, 2> subgroups{
        valid_subgroup(&span, 1),
        valid_subgroup(&span, 1),
    };
    cuphyLdpcCbLaunchBatchDesc_t batch{
        subgroups.data(),
        static_cast<uint16_t>(subgroups.size()),
        nullptr,
    };

    EXPECT_EQ(cuphy::ldpc::cb::validate_launch_batch_desc(batch), CUPHY_STATUS_NOT_SUPPORTED);
}

TEST(LdpcCbCommon, CountCodeblocksSumsSpans)
{
    auto span0      = valid_span();
    auto span1      = valid_span();
    span0.num_cb    = 2;
    span1.num_cb    = 3;
    span0.llr_stride_bytes  = 256;
    span0.bits_stride_bytes = 64;
    span1.llr_stride_bytes  = 256;
    span1.bits_stride_bytes = 64;
    std::array<cuphyLdpcCbBufferSpan_t, 2> spans{span0, span1};
    const auto subgroup = valid_subgroup(spans.data(), static_cast<uint16_t>(spans.size()));

    EXPECT_EQ(cuphy::ldpc::cb::count_codeblocks(subgroup), 5U);
}

TEST(LdpcCbCommon, CountCodeblocksSumsBatchSubgroups)
{
    auto span0   = valid_span();
    auto span1   = valid_span();
    auto span2   = valid_span();
    span0.num_cb = 2;
    span1.num_cb = 3;
    span2.num_cb = 4;

    std::array<cuphyLdpcCbBufferSpan_t, 2> first_subgroup_spans{span0, span1};
    std::array<cuphyLdpcCbBufferSpan_t, 1> second_subgroup_spans{span2};
    std::array<cuphyLdpcCbSubgroupDesc_t, 2> subgroups{
        valid_subgroup(first_subgroup_spans.data(), static_cast<uint16_t>(first_subgroup_spans.size())),
        valid_subgroup(second_subgroup_spans.data(), static_cast<uint16_t>(second_subgroup_spans.size())),
    };
    cuphyLdpcCbLaunchBatchDesc_t batch{
        subgroups.data(),
        static_cast<uint16_t>(subgroups.size()),
        nullptr,
    };

    EXPECT_EQ(cuphy::ldpc::cb::count_codeblocks(batch), 9U);
}

TEST(LdpcCbCommon, ValidateLaunchBatchRejectsNullChoiceOrBatchDesc)
{
    auto span     = valid_span();
    auto subgroup = valid_subgroup(&span, 1);
    cuphyLdpcCbLaunchBatchDesc_t batch_desc{
        &subgroup,
        1,
        nullptr,
    };
    cuphyLdpcCbKernelChoice_t choice{};
    choice.max_subgroups = 1;

    cuphyLdpcCbKernelLaunchBatch_t launch_batch{
        nullptr,
        &batch_desc,
    };
    EXPECT_EQ(cuphy::ldpc::cb::validate_launch_batch(launch_batch), CUPHY_STATUS_INVALID_ARGUMENT);

    launch_batch = cuphyLdpcCbKernelLaunchBatch_t{
        &choice,
        nullptr,
    };
    EXPECT_EQ(cuphy::ldpc::cb::validate_launch_batch(launch_batch), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbCommon, ValidateLaunchBatchRejectsChoiceLlrAlignmentViolation)
{
    alignas(64) static std::array<uint8_t, 1024> llrs{};
    auto span     = valid_span();
    span.llr_base = llrs.data() + 1;
    auto subgroup = valid_subgroup(&span, 1);
    cuphyLdpcCbLaunchBatchDesc_t batch_desc{
        &subgroup,
        1,
        nullptr,
    };
    cuphyLdpcCbKernelChoice_t choice{};
    choice.max_subgroups                   = 1;
    choice.traits.llr_base_alignment_bytes = 64;
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{
        &choice,
        &batch_desc,
    };

    EXPECT_EQ(cuphy::ldpc::cb::validate_launch_batch(launch_batch), CUPHY_STATUS_UNSUPPORTED_ALIGNMENT);
}

TEST(LdpcCbCommon, ValidateLaunchBatchRejectsChoiceBitsAlignmentViolation)
{
    alignas(64) static std::array<uint32_t, 256> bits{};
    auto span      = valid_span();
    span.bits_base = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(bits.data()) + alignof(uint32_t));
    auto subgroup  = valid_subgroup(&span, 1);
    cuphyLdpcCbLaunchBatchDesc_t batch_desc{
        &subgroup,
        1,
        nullptr,
    };
    cuphyLdpcCbKernelChoice_t choice{};
    choice.max_subgroups                    = 1;
    choice.traits.bits_base_alignment_bytes = 64;
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{
        &choice,
        &batch_desc,
    };

    EXPECT_EQ(cuphy::ldpc::cb::validate_launch_batch(launch_batch), CUPHY_STATUS_UNSUPPORTED_ALIGNMENT);
}

TEST(LdpcCbCommon, ValidateLaunchBatchRejectsChoiceBatchMismatch)
{
    auto span = valid_span();
    std::array<cuphyLdpcCbSubgroupDesc_t, 2> subgroups{
        valid_subgroup(&span, 1),
        valid_subgroup(&span, 1),
    };
    cuphyLdpcCbLaunchBatchDesc_t batch_desc{
        subgroups.data(),
        static_cast<uint16_t>(subgroups.size()),
        nullptr,
    };
    cuphyLdpcCbKernelChoice_t choice{};
    choice.supports_heterogeneous_batch = 1;
    choice.max_subgroups                = 1;
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{
        &choice,
        &batch_desc,
    };

    EXPECT_EQ(cuphy::ldpc::cb::validate_launch_batch(launch_batch), CUPHY_STATUS_UNSUPPORTED_CONFIG);
}

TEST(LdpcCbCommon, ValidateLaunchBatchAppliesMaxSubgroupsBeforeHeterogeneityRejection)
{
    auto span = valid_span();
    std::array<cuphyLdpcCbSubgroupDesc_t, 2> subgroups{
        valid_subgroup(&span, 1),
        valid_subgroup(&span, 1),
    };
    cuphyLdpcCbLaunchBatchDesc_t batch_desc{
        subgroups.data(),
        static_cast<uint16_t>(subgroups.size()),
        nullptr,
    };
    cuphyLdpcCbKernelChoice_t choice{};
    choice.supports_heterogeneous_batch = 0;
    choice.max_subgroups                = 1;
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{
        &choice,
        &batch_desc,
    };

    EXPECT_EQ(cuphy::ldpc::cb::validate_launch_batch(launch_batch), CUPHY_STATUS_UNSUPPORTED_CONFIG);
}

TEST(LdpcCbKernelSpi, BuildSubgroupDescRejectsNullArguments)
{
    const auto key = valid_key();
    std::array<cuphyLdpcCbData_t, 1> cb_data{};
    std::array<cuphyLdpcCbBufferSpan_t, 1> spans{};
    uint16_t num_spans = static_cast<uint16_t>(spans.size());
    cuphyLdpcCbSubgroupDesc_t subgroup{};

    EXPECT_EQ(cuphyLdpcCbBuildSubgroupDesc(nullptr, cb_data.data(), 1, spans.data(), &num_spans, &subgroup),
              CUPHY_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(cuphyLdpcCbBuildSubgroupDesc(&key, nullptr, 1, spans.data(), &num_spans, &subgroup),
              CUPHY_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(cuphyLdpcCbBuildSubgroupDesc(&key, cb_data.data(), 1, nullptr, &num_spans, &subgroup),
              CUPHY_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(cuphyLdpcCbBuildSubgroupDesc(&key, cb_data.data(), 1, spans.data(), nullptr, &subgroup),
              CUPHY_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(cuphyLdpcCbBuildSubgroupDesc(&key, cb_data.data(), 1, spans.data(), &num_spans, nullptr),
              CUPHY_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(cuphyLdpcCbBuildSubgroupDesc(&key, cb_data.data(), 0, spans.data(), &num_spans, &subgroup),
              CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbKernelSpi, BuildSubgroupDescCopiesKeyAndSingleSpan)
{
    alignas(16) std::array<uint8_t, 1024> llrs{};
    std::array<uint32_t, 64> bits{};
    std::array<cuphyLdpcCbData_t, 1> cb_data{{
        {llrs.data(), bits.data(), nullptr, CUPHY_LDPC_CB_GROUP_ID_NONE},
    }};
    std::array<cuphyLdpcCbBufferSpan_t, 1> spans{};
    uint16_t num_spans = static_cast<uint16_t>(spans.size());
    cuphyLdpcCbSubgroupDesc_t subgroup{};
    const auto key = valid_key();

    ASSERT_EQ(cuphyLdpcCbBuildSubgroupDesc(&key, cb_data.data(), static_cast<uint16_t>(cb_data.size()),
                                           spans.data(), &num_spans, &subgroup),
              CUPHY_STATUS_SUCCESS);

    EXPECT_EQ(subgroup.key.bg, key.bg);
    EXPECT_EQ(subgroup.key.Zc, key.Zc);
    EXPECT_EQ(subgroup.key.parity_nodes, key.parity_nodes);
    EXPECT_EQ(subgroup.key.k, key.k);
    EXPECT_EQ(subgroup.key.crc_type, key.crc_type);
    EXPECT_EQ(subgroup.key.max_iters, key.max_iters);
    EXPECT_EQ(subgroup.key.algo, key.algo);
    EXPECT_EQ(subgroup.spans, spans.data());
    ASSERT_EQ(subgroup.num_spans, 1);
    EXPECT_EQ(num_spans, 1);
    EXPECT_EQ(spans[0].llr_base, llrs.data());
    EXPECT_EQ(spans[0].llr_stride_bytes, 0);
    EXPECT_EQ(spans[0].bits_base, bits.data());
    EXPECT_EQ(spans[0].bits_offset_bytes, 0);
    EXPECT_EQ(spans[0].bits_stride_bytes, 0);
    EXPECT_EQ(spans[0].llr_out, nullptr);
    EXPECT_EQ(spans[0].llr_out_stride_elems, 0);
    EXPECT_EQ(spans[0].num_cb, 1);
}

TEST(LdpcCbKernelSpi, BuildSubgroupDescGroupsContiguousCodeblocks)
{
    alignas(16) std::array<uint8_t, 4096> llrs{};
    std::array<uint32_t, 256> bits{};
    std::array<uint8_t, 4096> soft{};
    std::array<cuphyLdpcCbData_t, 3> cb_data{{
        {llrs.data(), bits.data(), soft.data(), 7},
        {llrs.data() + 256, bits.data() + 16, soft.data() + 512, 7},
        {llrs.data() + 512, bits.data() + 32, soft.data() + 1024, 7},
    }};
    std::array<cuphyLdpcCbBufferSpan_t, 1> spans{};
    uint16_t num_spans = static_cast<uint16_t>(spans.size());
    cuphyLdpcCbSubgroupDesc_t subgroup{};
    const auto key = valid_key();

    ASSERT_EQ(cuphyLdpcCbBuildSubgroupDesc(&key, cb_data.data(), static_cast<uint16_t>(cb_data.size()),
                                           spans.data(), &num_spans, &subgroup),
              CUPHY_STATUS_SUCCESS);

    ASSERT_EQ(num_spans, 1);
    EXPECT_EQ(subgroup.num_spans, 1);
    EXPECT_EQ(spans[0].llr_base, cb_data[0].llr_in);
    EXPECT_EQ(spans[0].llr_stride_bytes, 256);
    EXPECT_EQ(spans[0].bits_base, cb_data[0].bits_out);
    EXPECT_EQ(spans[0].bits_stride_bytes, 16U * sizeof(uint32_t));
    EXPECT_EQ(spans[0].llr_out, cb_data[0].llr_out);
    EXPECT_EQ(spans[0].llr_out_stride_elems, 256);
    EXPECT_EQ(spans[0].num_cb, 3);
}

TEST(LdpcCbKernelSpi, BuildSubgroupDescRejectsTooSmallSpanStorage)
{
    alignas(16) std::array<uint8_t, 2048> llrs{};
    std::array<uint32_t, 128> bits{};
    std::array<cuphyLdpcCbData_t, 2> cb_data{{
        {llrs.data(), bits.data(), nullptr, CUPHY_LDPC_CB_GROUP_ID_NONE},
        {llrs.data() + 256, bits.data() + 16, nullptr, CUPHY_LDPC_CB_GROUP_ID_NONE},
    }};
    std::array<cuphyLdpcCbBufferSpan_t, 1> spans{};
    uint16_t num_spans = static_cast<uint16_t>(spans.size());
    cuphyLdpcCbSubgroupDesc_t subgroup{};
    const auto key = valid_key();

    EXPECT_EQ(cuphyLdpcCbBuildSubgroupDesc(&key, cb_data.data(), static_cast<uint16_t>(cb_data.size()),
                                           spans.data(), &num_spans, &subgroup),
              CUPHY_STATUS_INSUFFICIENT_RESOURCES);
}

TEST(LdpcCbKernelSpi, BuildKernelNodeParamsRejectsNullArguments)
{
    cuphyLdpcCbKernelNodeParams_t node_params{};
    EXPECT_EQ(cuphyLdpcCbBuildKernelNodeParams(nullptr, &node_params), CUPHY_STATUS_INVALID_ARGUMENT);

    auto span     = valid_span();
    auto subgroup = valid_subgroup(&span, 1);
    cuphyLdpcCbLaunchBatchDesc_t batch_desc{&subgroup, 1, nullptr};
    auto choice = valid_choice();
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{&choice, &batch_desc};

    EXPECT_EQ(cuphyLdpcCbBuildKernelNodeParams(&launch_batch, nullptr), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbKernelSpi, BuildKernelNodeParamsRejectsChoiceBatchMismatch)
{
    auto span     = valid_span();
    auto subgroup = valid_subgroup(&span, 1);
    cuphyLdpcCbLaunchBatchDesc_t batch_desc{&subgroup, 1, nullptr};
    auto choice = valid_choice();
    choice.key.Zc += 8;
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{&choice, &batch_desc};
    cuphyLdpcCbKernelNodeParams_t node_params{};

    EXPECT_EQ(cuphyLdpcCbBuildKernelNodeParams(&launch_batch, &node_params), CUPHY_STATUS_UNSUPPORTED_CONFIG);
}

TEST(LdpcCbKernelSpi, BuildKernelNodeParamsRejectsTooManySpans)
{
    std::array<cuphyLdpcCbBufferSpan_t, CUPHY_LDPC_DECODE_DESC_MAX_TB + 1> spans{};
    auto span = valid_span();
    for (auto& item : spans)
    {
        item = span;
    }
    cuphyLdpcCbSubgroupDesc_t subgroup{valid_key(), spans.data(), static_cast<uint16_t>(spans.size())};
    cuphyLdpcCbLaunchBatchDesc_t batch_desc{&subgroup, 1, nullptr};
    auto choice = valid_choice();
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{&choice, &batch_desc};
    cuphyLdpcCbKernelNodeParams_t node_params{};

    EXPECT_EQ(cuphyLdpcCbBuildKernelNodeParams(&launch_batch, &node_params), CUPHY_STATUS_INSUFFICIENT_RESOURCES);
}

TEST(LdpcCbKernelSpi, BuildKernelNodeParamsRejectsNonWordBitByteOffset)
{
    auto span              = valid_span();
    span.bits_offset_bytes = 1;
    auto subgroup          = valid_subgroup(&span, 1);
    cuphyLdpcCbLaunchBatchDesc_t batch_desc{&subgroup, 1, nullptr};
    auto choice = valid_choice();
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{&choice, &batch_desc};
    cuphyLdpcCbKernelNodeParams_t node_params{};

    EXPECT_EQ(cuphyLdpcCbBuildKernelNodeParams(&launch_batch, &node_params), CUPHY_STATUS_NOT_SUPPORTED);
}

TEST(LdpcCbKernelSpi, BuildKernelNodeParamsRejectsNonWordBitByteStride)
{
    auto span             = valid_span();
    span.num_cb           = 2;
    span.bits_stride_bytes = sizeof(uint32_t) + 1;
    auto subgroup         = valid_subgroup(&span, 1);
    cuphyLdpcCbLaunchBatchDesc_t batch_desc{&subgroup, 1, nullptr};
    auto choice = valid_choice();
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{&choice, &batch_desc};
    cuphyLdpcCbKernelNodeParams_t node_params{};

    EXPECT_EQ(cuphyLdpcCbBuildKernelNodeParams(&launch_batch, &node_params), CUPHY_STATUS_NOT_SUPPORTED);
}

TEST(LdpcCbKernelSpi, BuildKernelNodeParamsRejectsUnsupportedStatusOutput)
{
    auto span     = valid_span();
    auto subgroup = valid_subgroup(&span, 1);
    uint32_t status_out{};
    auto choice = valid_choice();
    cuphyLdpcCbKernelNodeParams_t node_params{};

    cuphyLdpcCbLaunchBatchDesc_t batch_desc{&subgroup, 1, &status_out};
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{&choice, &batch_desc};
    EXPECT_EQ(cuphyLdpcCbBuildKernelNodeParams(&launch_batch, &node_params), CUPHY_STATUS_NOT_SUPPORTED);
}

TEST(LdpcCbKernelSpi, BuildKernelNodeParamsRejectsInvalidKernelArgCount)
{
    auto span     = valid_span();
    auto subgroup = valid_subgroup(&span, 1);
    cuphyLdpcCbLaunchBatchDesc_t batch_desc{&subgroup, 1, nullptr};
    auto choice = valid_choice();
    choice.traits.kernel_arg_count = 1;
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{&choice, &batch_desc};
    cuphyLdpcCbKernelNodeParams_t node_params{};

    EXPECT_EQ(cuphyLdpcCbBuildKernelNodeParams(&launch_batch, &node_params), CUPHY_STATUS_UNSUPPORTED_CONFIG);
}

TEST(LdpcCbKernelSpi, BuildKernelNodeParamsRejectsZeroCodewordsPerCta)
{
    auto span     = valid_span();
    auto subgroup = valid_subgroup(&span, 1);
    cuphyLdpcCbLaunchBatchDesc_t batch_desc{&subgroup, 1, nullptr};
    auto choice = valid_choice();
    choice.traits.codewords_per_cta = 0;
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{&choice, &batch_desc};
    cuphyLdpcCbKernelNodeParams_t node_params{};

    EXPECT_EQ(cuphyLdpcCbBuildKernelNodeParams(&launch_batch, &node_params), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbKernelSpi, BuildKernelNodeParamsPopulatesLaunchConfigAndNodeParams)
{
    alignas(16) std::array<uint8_t, 4096> llrs{};
    std::array<uint32_t, 256> bits{};
    std::array<cuphyLdpcCbBufferSpan_t, 2> spans{{
        {llrs.data(), 768, bits.data(), 4 * sizeof(uint32_t), 24 * sizeof(uint32_t), nullptr, 0, 2},
        {llrs.data() + 2048, 0, bits.data() + 96, 0, 0, nullptr, 0, 1},
    }};
    cuphyLdpcCbSubgroupDesc_t subgroup{valid_key(), spans.data(), static_cast<uint16_t>(spans.size())};
    cuphyLdpcCbLaunchBatchDesc_t batch_desc{&subgroup, 1, nullptr};
    auto choice = valid_choice();
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{&choice, &batch_desc};
    cuphyLdpcCbKernelNodeParams_t node_params{};

    ASSERT_EQ(cuphyLdpcCbBuildKernelNodeParams(&launch_batch, &node_params), CUPHY_STATUS_SUCCESS);

    const auto& launch_config = payload_launch_config(node_params);
    const auto& decode_desc = launch_config.decode_desc;
    EXPECT_EQ(std::memcmp(&decode_desc.config, &choice.decode_config, sizeof(choice.decode_config)), 0);
    EXPECT_EQ(decode_desc.num_tbs, 2);
    EXPECT_EQ(decode_desc.llr_input[0].addr, const_cast<uint8_t*>(spans[0].llr_base));
    EXPECT_EQ(decode_desc.llr_input[0].stride_elements, 384);
    EXPECT_EQ(decode_desc.llr_input[0].num_codewords, 2);
    EXPECT_EQ(decode_desc.tb_output[0].addr, bits.data() + 4);
    EXPECT_EQ(decode_desc.tb_output[0].stride_words, 24);
    EXPECT_EQ(decode_desc.tb_output[0].num_codewords, 2);
    EXPECT_EQ(decode_desc.llr_input[1].stride_elements, 66 * valid_key().Zc);
    EXPECT_EQ(decode_desc.tb_output[1].stride_words, (valid_key().k + 31) / 32);
    EXPECT_EQ(decode_desc.llr_output[0].addr, nullptr);
    EXPECT_EQ(decode_desc.llr_output[0].stride_elements, 0);
    EXPECT_EQ(decode_desc.llr_output[0].num_codewords, 0);

    EXPECT_GE(CUPHY_LDPC_CB_KERNEL_NODE_PAYLOAD_BYTES, sizeof(cuphyLDPCDecodeLaunchConfig_t));
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(node_params.payload) % alignof(cuphyLDPCDecodeLaunchConfig_t), 0U);
    EXPECT_EQ(node_params.kernel_arg_count, 2U);
    EXPECT_EQ(node_params.kernel_args[0], &launch_config.decode_desc);
    EXPECT_EQ(node_params.kernel_args[1], choice.static_kernel_arg);
    EXPECT_EQ(node_params.params.func, choice.func);
    EXPECT_EQ(node_params.params.gridDimX, 2U);
    EXPECT_EQ(node_params.params.gridDimY, 1U);
    EXPECT_EQ(node_params.params.gridDimZ, 1U);
    EXPECT_EQ(node_params.params.blockDimX, choice.traits.block_dim_x);
    EXPECT_EQ(node_params.params.blockDimY, choice.traits.block_dim_y);
    EXPECT_EQ(node_params.params.blockDimZ, choice.traits.block_dim_z);
    EXPECT_EQ(node_params.params.sharedMemBytes, choice.traits.shared_mem_bytes);
    EXPECT_EQ(node_params.params.kernelParams, node_params.kernel_args);
    EXPECT_EQ(node_params.params.extra, nullptr);
    EXPECT_EQ(launch_config.kernel_args[0], node_params.kernel_args[0]);
    EXPECT_EQ(launch_config.kernel_args[1], node_params.kernel_args[1]);
    EXPECT_EQ(launch_config.kernel_node_params_driver.kernelParams, node_params.kernel_args);
}

TEST(LdpcCbKernelSpi, BuildKernelNodeParamsRoundsX2GridPerSpan)
{
    alignas(16) std::array<uint8_t, 4096> llrs{};
    std::array<uint32_t, 512> bits{};
    std::array<cuphyLdpcCbBufferSpan_t, 2> spans{{
        {llrs.data(), 768, bits.data(), 0, 24 * sizeof(uint32_t), nullptr, 0, 5},
        {llrs.data() + 2048, 768, bits.data() + 128, 0, 24 * sizeof(uint32_t), nullptr, 0, 5},
    }};
    cuphyLdpcCbSubgroupDesc_t subgroup{valid_key(), spans.data(), static_cast<uint16_t>(spans.size())};
    cuphyLdpcCbLaunchBatchDesc_t batch_desc{&subgroup, 1, nullptr};
    auto choice = valid_choice();
    choice.traits.codewords_per_cta = 2;
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{&choice, &batch_desc};
    cuphyLdpcCbKernelNodeParams_t node_params{};

    ASSERT_EQ(cuphyLdpcCbBuildKernelNodeParams(&launch_batch, &node_params), CUPHY_STATUS_SUCCESS);

    EXPECT_EQ(node_params.params.gridDimX, 6U);
}

TEST(LdpcCbKernelSpi, BuildKernelNodeParamsRejectsMissingSoftOutputWhenEnabled)
{
    auto span     = valid_span();
    auto subgroup = valid_subgroup(&span, 1);
    cuphyLdpcCbLaunchBatchDesc_t batch_desc{&subgroup, 1, nullptr};
    auto choice = valid_choice();
    choice.decode_config.flags |= CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS;
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{&choice, &batch_desc};
    cuphyLdpcCbKernelNodeParams_t node_params{};

    EXPECT_EQ(cuphyLdpcCbBuildKernelNodeParams(&launch_batch, &node_params), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbKernelSpi, BuildKernelNodeParamsPopulatesSoftOutput)
{
    alignas(16) std::array<uint8_t, 2048> llrs{};
    std::array<uint32_t, 256> bits{};
    std::array<uint8_t, 4096> soft{};
    std::array<cuphyLdpcCbBufferSpan_t, 1> spans{{
        {llrs.data(), 768, bits.data(), 0, 24 * sizeof(uint32_t), soft.data(), 384, 2},
    }};
    cuphyLdpcCbSubgroupDesc_t subgroup{valid_key(), spans.data(), static_cast<uint16_t>(spans.size())};
    cuphyLdpcCbLaunchBatchDesc_t batch_desc{&subgroup, 1, nullptr};
    auto choice = valid_choice();
    choice.decode_config.flags |= CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS;
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{&choice, &batch_desc};
    cuphyLdpcCbKernelNodeParams_t node_params{};

    ASSERT_EQ(cuphyLdpcCbBuildKernelNodeParams(&launch_batch, &node_params), CUPHY_STATUS_SUCCESS);

    const auto& llr_output = payload_launch_config(node_params).decode_desc.llr_output[0];
    EXPECT_EQ(llr_output.addr, spans[0].llr_out);
    EXPECT_EQ(llr_output.stride_elements, 384);
    EXPECT_EQ(llr_output.num_codewords, 2);
}

TEST(LdpcCbKernelSpi, BuildKernelNodeParamsRejectsSoftOutputStrideOutOfRange)
{
    alignas(16) std::array<uint8_t, 2048> llrs{};
    std::array<uint32_t, 256> bits{};
    std::array<uint8_t, 4096> soft{};
    std::array<cuphyLdpcCbBufferSpan_t, 1> spans{{
        {llrs.data(),
         768,
         bits.data(),
         0,
         24 * sizeof(uint32_t),
         soft.data(),
         static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) + 1U,
         2},
    }};
    cuphyLdpcCbSubgroupDesc_t subgroup{valid_key(), spans.data(), static_cast<uint16_t>(spans.size())};
    cuphyLdpcCbLaunchBatchDesc_t batch_desc{&subgroup, 1, nullptr};
    auto choice = valid_choice();
    choice.decode_config.flags |= CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS;
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{&choice, &batch_desc};
    cuphyLdpcCbKernelNodeParams_t node_params{};

    EXPECT_EQ(cuphyLdpcCbBuildKernelNodeParams(&launch_batch, &node_params), CUPHY_STATUS_VALUE_OUT_OF_RANGE);
}

TEST(LdpcCbKernelSpi, BuildKernelNodeParamsRejectsFp32Input)
{
    alignas(16) std::array<uint8_t, 4096> llrs{};
    std::array<uint32_t, 256> bits{};
    std::array<uint8_t, 4096> soft{};
    std::array<cuphyLdpcCbBufferSpan_t, 1> spans{{
        {llrs.data(), 1536, bits.data(), 0, 24 * sizeof(uint32_t), soft.data(), 384, 2},
    }};
    cuphyLdpcCbSubgroupDesc_t subgroup{valid_key(), spans.data(), static_cast<uint16_t>(spans.size())};
    cuphyLdpcCbLaunchBatchDesc_t batch_desc{&subgroup, 1, nullptr};
    auto choice = valid_choice();
    choice.decode_config.llr_type = CUPHY_R_32F;
    choice.decode_config.flags |= CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS;
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{&choice, &batch_desc};
    cuphyLdpcCbKernelNodeParams_t node_params{};

    EXPECT_EQ(cuphyLdpcCbBuildKernelNodeParams(&launch_batch, &node_params), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbKernelSpi, BuildKernelNodeParamsRejectsUnsupportedEarlyTermination)
{
    alignas(16) std::array<uint8_t, 4096> llrs{};
    std::array<uint32_t, 256> bits{};
    std::array<cuphyLdpcCbBufferSpan_t, 1> spans{{
        {llrs.data(), 1536, bits.data(), 0, 24 * sizeof(uint32_t), nullptr, 0, 2},
    }};
    cuphyLdpcCbSubgroupDesc_t subgroup{valid_key(), spans.data(), static_cast<uint16_t>(spans.size())};
    cuphyLdpcCbLaunchBatchDesc_t batch_desc{&subgroup, 1, nullptr};
    auto choice = valid_choice();
    choice.decode_config.flags |= CUPHY_LDPC_DECODE_EARLY_TERM;
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{&choice, &batch_desc};
    cuphyLdpcCbKernelNodeParams_t node_params{};

    EXPECT_EQ(cuphyLdpcCbBuildKernelNodeParams(&launch_batch, &node_params), CUPHY_STATUS_NOT_SUPPORTED);
}

TEST_F(LdpcCbKernelChooserCudaTest, BuildKernelNodeParamsDoesNotRequireChooserForDescriptorPacking)
{
    const auto key = valid_key();
    cuphyLdpcCbKernelChoice_t choice{};
    ASSERT_EQ(cuphyLdpcCbChooseKernel(chooser_, &key, &choice), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphyDestroyLdpcCbKernelChooser(chooser_), CUPHY_STATUS_SUCCESS);
    chooser_ = nullptr;

    alignas(16) std::array<uint8_t, 4096> aligned_llrs{};
    std::array<uint32_t, 256> bits{};
    cuphyLdpcCbBufferSpan_t span{
        aligned_llrs.data(),
        1024,
        bits.data(),
        0,
        128,
        nullptr,
        0,
        1,
    };
    auto subgroup = valid_subgroup(&span, 1);
    cuphyLdpcCbLaunchBatchDesc_t batch_desc{&subgroup, 1, nullptr};
    cuphyLdpcCbKernelLaunchBatch_t launch_batch{&choice, &batch_desc};
    cuphyLdpcCbKernelNodeParams_t node_params{};

    // This verifies host-side descriptor packing only. The chooser must still
    // outlive launches and graph nodes that use choices it produced.
    EXPECT_EQ(cuphyLdpcCbBuildKernelNodeParams(&launch_batch, &node_params), CUPHY_STATUS_SUCCESS);
    EXPECT_EQ(node_params.kernel_args[1], choice.static_kernel_arg);
}

TEST(LdpcCbKernelChooser, CreateKernelChooserRejectsNullOutput)
{
    const auto config = valid_chooser_config();
    EXPECT_EQ(cuphyCreateLdpcCbKernelChooser(nullptr, &config), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbKernelChooser, CreateKernelChooserRejectsNullConfig)
{
    cuphyLdpcCbKernelChooser_t chooser{};
    EXPECT_EQ(cuphyCreateLdpcCbKernelChooser(&chooser, nullptr), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(LdpcCbKernelChooser, CreateKernelChooserRejectsUnsupportedLlrType)
{
    cuphyLdpcCbKernelChooser_t chooser{};
    auto config      = valid_chooser_config();
    config.llr_type  = CUPHY_R_8I;

    EXPECT_EQ(cuphyCreateLdpcCbKernelChooser(&chooser, &config), CUPHY_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(chooser, nullptr);
}

TEST(LdpcCbKernelChooser, CreateKernelChooserRejectsFp32Input)
{
    cuphyLdpcCbKernelChooser_t chooser{};
    auto config      = valid_chooser_config();
    config.llr_type  = CUPHY_R_32F;

    EXPECT_EQ(cuphyCreateLdpcCbKernelChooser(&chooser, &config), CUPHY_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(chooser, nullptr);
}

TEST(LdpcCbKernelChooser, CreateKernelChooserRejectsNonPositiveClamp)
{
    cuphyLdpcCbKernelChooser_t chooser{};
    auto config         = valid_chooser_config();
    config.clamp_value  = 0.0f;
    EXPECT_EQ(cuphyCreateLdpcCbKernelChooser(&chooser, &config), CUPHY_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(chooser, nullptr);

    config.clamp_value = -1.0f;
    EXPECT_EQ(cuphyCreateLdpcCbKernelChooser(&chooser, &config), CUPHY_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(chooser, nullptr);
}

TEST(LdpcCbKernelChooser, CreateKernelChooserRejectsUnsupportedEarlyTerminationFlag)
{
    cuphyLdpcCbKernelChooser_t chooser{};
    auto config = valid_chooser_config();
    config.config_flags |= CUPHY_LDPC_CB_EARLY_TERM;

    EXPECT_EQ(cuphyCreateLdpcCbKernelChooser(&chooser, &config), CUPHY_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(chooser, nullptr);
}

TEST(LdpcCbKernelChooser, CreateKernelChooserRejectsUnknownConfigFlags)
{
    cuphyLdpcCbKernelChooser_t chooser{};
    auto config = valid_chooser_config();
    config.config_flags |= (1U << 31);

    EXPECT_EQ(cuphyCreateLdpcCbKernelChooser(&chooser, &config), CUPHY_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(chooser, nullptr);
}

TEST(LdpcCbKernelChooser, ChooseKernelRejectsNullChooser)
{
    const auto key = valid_key();
    cuphyLdpcCbKernelChoice_t choice{};

    EXPECT_EQ(cuphyLdpcCbChooseKernel(nullptr, &key, &choice), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST_F(LdpcCbKernelChooserCudaTest, ChooseKernelRejectsNullKey)
{
    cuphyLdpcCbKernelChoice_t choice{};
    EXPECT_EQ(cuphyLdpcCbChooseKernel(chooser_, nullptr, &choice), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST_F(LdpcCbKernelChooserCudaTest, ChooseKernelRejectsUnsupportedBaseGraph)
{
    auto key = valid_key();
    key.bg   = static_cast<cuphyLdpcCbBaseGraph_t>(3);
    cuphyLdpcCbKernelChoice_t choice{};

    EXPECT_EQ(cuphyLdpcCbChooseKernel(chooser_, &key, &choice), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST_F(LdpcCbKernelChooserCudaTest, ChooseKernelRejectsNullChoice)
{
    const auto key = valid_key();
    EXPECT_EQ(cuphyLdpcCbChooseKernel(chooser_, &key, nullptr), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST_F(LdpcCbKernelChooserCudaTest, ChooseKernelRejectsNonIntegralKb)
{
    auto key = valid_key();
    key.k += 1;
    cuphyLdpcCbKernelChoice_t choice{};

    EXPECT_EQ(cuphyLdpcCbChooseKernel(chooser_, &key, &choice), CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST_F(LdpcCbKernelChooserCudaTest, ChooseKernelRejectsInvalidExplicitAlgorithm)
{
    auto key  = valid_key();
    key.algo  = 255;
    cuphyLdpcCbKernelChoice_t choice{};

    EXPECT_EQ(cuphyLdpcCbChooseKernel(chooser_, &key, &choice), CUPHY_STATUS_UNSUPPORTED_CONFIG);
}

TEST_F(LdpcCbKernelChooserCudaTest, ChooseKernelReturnsStableChoiceForSameKey)
{
    const auto key = valid_key();
    cuphyLdpcCbKernelChoice_t first{};
    cuphyLdpcCbKernelChoice_t second{};

    ASSERT_EQ(cuphyLdpcCbChooseKernel(chooser_, &key, &first), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphyLdpcCbChooseKernel(chooser_, &key, &second), CUPHY_STATUS_SUCCESS);

    EXPECT_EQ(std::memcmp(&first, &second, sizeof(first)), 0);
}

TEST_F(LdpcCbKernelChooserCudaTest, ChooseKernelPopulatesChoiceTraits)
{
    const auto key = valid_key();
    cuphyLdpcCbKernelChoice_t choice{};

    ASSERT_EQ(cuphyLdpcCbChooseKernel(chooser_, &key, &choice), CUPHY_STATUS_SUCCESS);

    EXPECT_NE(choice.func, nullptr);
    EXPECT_GT(choice.traits.block_dim_x, 0);
    EXPECT_GT(choice.traits.block_dim_y, 0);
    EXPECT_GT(choice.traits.block_dim_z, 0);
    EXPECT_TRUE(choice.traits.codewords_per_cta == 1 || choice.traits.codewords_per_cta == 2);
    EXPECT_EQ(choice.traits.kernel_arg_count, 2);
    EXPECT_EQ(choice.traits.llr_base_alignment_bytes, 16);
    EXPECT_EQ(choice.traits.bits_base_alignment_bytes, alignof(uint32_t));
    EXPECT_NE(choice.effective_algo, 0);
    EXPECT_NE(choice.compatibility_class_id, 0U);
    EXPECT_EQ(choice.supports_heterogeneous_batch, 0);
    EXPECT_EQ(choice.max_subgroups, 1);
}

TEST_F(LdpcCbKernelChooserCudaTest, ChooseKernelReportsSmallZCodewordsPerCta)
{
    auto key = valid_key();
    key.Zc   = 2;
    key.k    = static_cast<uint16_t>(22 * key.Zc);
    cuphyLdpcCbKernelChoice_t choice{};

    ASSERT_EQ(cuphyLdpcCbChooseKernel(chooser_, &key, &choice), CUPHY_STATUS_SUCCESS);
    EXPECT_EQ(choice.traits.codewords_per_cta, 16);
}
