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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

struct cuphyLdpcCbLaunchPreparer_s
{
    cuphyLdpcCbLaunchStaticConfig_t config;
    cuphyLdpcCbKernelChooser_t      chooser;
    uint32_t                         active_prepared_launches;

    cuphyLdpcCbLaunchPreparer_s(const cuphyLdpcCbLaunchStaticConfig_t& cfg,
                                cuphyLdpcCbKernelChooser_t             kernel_chooser) :
        config(cfg),
        chooser(kernel_chooser),
        active_prepared_launches(0)
    {
    }

    ~cuphyLdpcCbLaunchPreparer_s()
    {
        if (chooser != nullptr)
        {
            cuphyDestroyLdpcCbKernelChooser(chooser);
        }
    }

    cuphyLdpcCbLaunchPreparer_s(const cuphyLdpcCbLaunchPreparer_s&) = delete;
    cuphyLdpcCbLaunchPreparer_s& operator=(const cuphyLdpcCbLaunchPreparer_s&) = delete;
    cuphyLdpcCbLaunchPreparer_s(cuphyLdpcCbLaunchPreparer_s&&) = delete;
    cuphyLdpcCbLaunchPreparer_s& operator=(cuphyLdpcCbLaunchPreparer_s&&) = delete;
};

struct cuphyLdpcCbPreparedLaunch_s
{
    cuphyLdpcCbLaunchPreparer_t        preparer;
    cuphyLdpcCbLaunchFamily_t          family;
    cuphyLdpcCbPreparedLaunchConfig_t  capacity;
    cuphyLdpcCbKernelChoice_t          choice;
    cuphyLdpcCbLaunchBatchDesc_t       batch_desc;
    cuphyLdpcCbKernelLaunchBatch_t     launch_batch;
    cuphyLdpcCbKernelNodeParams_t      spi_node_params;
    cuphyLdpcCbSubgroupDesc_t*         subgroups;
    cuphyLdpcCbBufferSpan_t*           spans;
    uint16_t                           prepared_num_subgroups;
    uint32_t                           prepared_total_codeblocks;
    bool                               prepared;
};

namespace {

struct WorkspaceLayout
{
    size_t alignment;
    size_t object_offset;
    size_t subgroups_offset;
    size_t spans_offset;
    size_t total_size;
};

constexpr size_t workspace_alignment()
{
    return std::max({alignof(cuphyLdpcCbPreparedLaunch_s),
                     alignof(cuphyLdpcCbSubgroupDesc_t),
                     alignof(cuphyLdpcCbBufferSpan_t)});
}

bool align_offset(const size_t offset, const size_t alignment, size_t& aligned_offset)
{
    const size_t mask = alignment - 1;
    if (offset > std::numeric_limits<size_t>::max() - mask)
    {
        return false;
    }

    aligned_offset = (offset + mask) & ~mask;
    return true;
}

template <typename T>
bool append_array(const size_t count, size_t& offset, size_t& array_offset)
{
    if (!align_offset(offset, alignof(T), array_offset))
    {
        return false;
    }

    if (count > std::numeric_limits<size_t>::max() / sizeof(T))
    {
        return false;
    }

    const size_t bytes = count * sizeof(T);
    if (array_offset > std::numeric_limits<size_t>::max() - bytes)
    {
        return false;
    }

    offset = array_offset + bytes;
    return true;
}

bool compute_workspace_layout(const cuphyLdpcCbPreparedLaunchConfig_t& config, WorkspaceLayout& layout)
{
    size_t offset = 0;
    layout = WorkspaceLayout{};
    layout.alignment = workspace_alignment();

    if (!append_array<cuphyLdpcCbPreparedLaunch_s>(1, offset, layout.object_offset))
    {
        return false;
    }
    if (!append_array<cuphyLdpcCbSubgroupDesc_t>(config.max_subgroups, offset, layout.subgroups_offset))
    {
        return false;
    }
    if (!append_array<cuphyLdpcCbBufferSpan_t>(config.max_total_codeblocks, offset, layout.spans_offset))
    {
        return false;
    }
    if (!align_offset(offset, layout.alignment, layout.total_size))
    {
        return false;
    }

    return true;
}

cuphyStatus_t validate_family(const cuphyLdpcCbLaunchFamily_t& family)
{
    if (family.compatibility_class_id == 0 || family.max_subgroups == 0)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t validate_prepared_config(const cuphyLdpcCbLaunchFamily_t&         family,
                                       const cuphyLdpcCbPreparedLaunchConfig_t& config)
{
    const auto family_status = validate_family(family);
    if (family_status != CUPHY_STATUS_SUCCESS)
    {
        return family_status;
    }

    if (config.max_subgroups == 0 || config.max_total_codeblocks == 0)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if (config.max_subgroups > family.max_subgroups)
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }

    return CUPHY_STATUS_SUCCESS;
}

cuphyLdpcCbKernelChooserConfig_t make_kernel_chooser_config(const cuphyLdpcCbLaunchStaticConfig_t& config)
{
    return cuphyLdpcCbKernelChooserConfig_t{
        config.llr_type,
        config.clamp_value,
        config.config_flags,
    };
}

cuphyStatus_t choose_for_batch(cuphyLdpcCbPreparedLaunch_s&     launch,
                               const cuphyLdpcCbLaunchBatchDesc_t& batch_desc,
                               cuphyLdpcCbKernelChoice_t&       choice)
{
    const auto status = cuphyLdpcCbChooseKernel(launch.preparer->chooser, &batch_desc.subgroups[0].key, &choice);
    if (status != CUPHY_STATUS_SUCCESS)
    {
        return status;
    }

    if (choice.compatibility_class_id != launch.family.compatibility_class_id ||
        choice.effective_algo != launch.family.effective_algo)
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }

    return CUPHY_STATUS_SUCCESS;
}

bool same_subgroup_key(const cuphyLdpcCbSubgroupKey_t& lhs, const cuphyLdpcCbSubgroupKey_t& rhs)
{
    return lhs.bg == rhs.bg && lhs.Zc == rhs.Zc && lhs.parity_nodes == rhs.parity_nodes && lhs.k == rhs.k &&
           lhs.crc_type == rhs.crc_type && lhs.max_iters == rhs.max_iters && lhs.algo == rhs.algo;
}

cuphyStatus_t validate_choice_for_batch(const cuphyLdpcCbPreparedLaunch_s&       launch,
                                        const cuphyLdpcCbLaunchBatchDesc_t&      batch_desc,
                                        const cuphyLdpcCbKernelChoice_t&         choice)
{
    if (!same_subgroup_key(choice.key, batch_desc.subgroups[0].key))
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    if (choice.compatibility_class_id != launch.family.compatibility_class_id ||
        choice.effective_algo != launch.family.effective_algo)
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t count_batch_codeblocks_checked(const cuphyLdpcCbLaunchBatchDesc_t& batch_desc,
                                             uint32_t&                            total_codeblocks,
                                             uint32_t&                            total_spans)
{
    uint64_t codeblocks = 0;
    uint64_t spans = 0;

    for (uint16_t subgroup_idx = 0; subgroup_idx < batch_desc.num_subgroups; ++subgroup_idx)
    {
        const auto& subgroup = batch_desc.subgroups[subgroup_idx];
        spans += subgroup.num_spans;
        if (spans > std::numeric_limits<uint32_t>::max())
        {
            return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
        }

        for (uint16_t span_idx = 0; span_idx < subgroup.num_spans; ++span_idx)
        {
            codeblocks += subgroup.spans[span_idx].num_cb;
            if (codeblocks > std::numeric_limits<uint32_t>::max())
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
    }

    if (codeblocks == 0 || spans == 0)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    total_codeblocks = static_cast<uint32_t>(codeblocks);
    total_spans = static_cast<uint32_t>(spans);
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t copy_batch_desc(cuphyLdpcCbPreparedLaunch_s&           launch,
                              const cuphyLdpcCbLaunchBatchDesc_t&    batch_desc,
                              const uint32_t                         total_spans)
{
    if (total_spans > launch.capacity.max_total_codeblocks)
    {
        return CUPHY_STATUS_INSUFFICIENT_RESOURCES;
    }

    uint32_t span_write_idx = 0;
    for (uint16_t subgroup_idx = 0; subgroup_idx < batch_desc.num_subgroups; ++subgroup_idx)
    {
        const auto& src_subgroup = batch_desc.subgroups[subgroup_idx];
        auto&       dst_subgroup = launch.subgroups[subgroup_idx];

        dst_subgroup.key       = src_subgroup.key;
        dst_subgroup.num_spans = src_subgroup.num_spans;
        dst_subgroup.spans     = &launch.spans[span_write_idx];

        std::memcpy(&launch.spans[span_write_idx],
                    src_subgroup.spans,
                    static_cast<size_t>(src_subgroup.num_spans) * sizeof(cuphyLdpcCbBufferSpan_t));
        span_write_idx += src_subgroup.num_spans;
    }

    launch.batch_desc.subgroups     = launch.subgroups;
    launch.batch_desc.num_subgroups = batch_desc.num_subgroups;
    launch.batch_desc.status_out    = batch_desc.status_out;
    return CUPHY_STATUS_SUCCESS;
}

} // namespace

cuphyStatus_t cuphyCreateLdpcCbLaunchPreparer(cuphyLdpcCbLaunchPreparer_t*           preparer,
                                              const cuphyLdpcCbLaunchStaticConfig_t* config)
{
    if (preparer == nullptr || config == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    *preparer = nullptr;

    cuphyLdpcCbKernelChooser_t chooser = nullptr;
    const auto chooser_config = make_kernel_chooser_config(*config);
    auto status = cuphyCreateLdpcCbKernelChooser(&chooser, &chooser_config);
    if (status != CUPHY_STATUS_SUCCESS)
    {
        return status;
    }

    try
    {
        *preparer = new cuphyLdpcCbLaunchPreparer_s(*config, chooser);
        return CUPHY_STATUS_SUCCESS;
    }
    catch (const std::bad_alloc&)
    {
        cuphyDestroyLdpcCbKernelChooser(chooser);
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch (...)
    {
        cuphyDestroyLdpcCbKernelChooser(chooser);
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
}

cuphyStatus_t cuphyDestroyLdpcCbLaunchPreparer(cuphyLdpcCbLaunchPreparer_t preparer)
{
    if (preparer == nullptr || preparer->active_prepared_launches != 0)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    delete preparer;
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t cuphyLdpcCbQueryLaunchFamily(cuphyLdpcCbLaunchPreparer_t     preparer,
                                           const cuphyLdpcCbSubgroupKey_t* key,
                                           cuphyLdpcCbLaunchFamily_t*      family)
{
    if (preparer == nullptr || key == nullptr || family == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    cuphyLdpcCbKernelChoice_t choice{};
    const auto status = cuphyLdpcCbQueryKernelChoice(preparer, key, &choice);
    if (status != CUPHY_STATUS_SUCCESS)
    {
        return status;
    }

    *family = cuphyLdpcCbLaunchFamily_t{
        choice.compatibility_class_id,
        choice.effective_algo,
        choice.supports_heterogeneous_batch,
        choice.max_subgroups,
    };
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t cuphyLdpcCbQueryKernelChoice(cuphyLdpcCbLaunchPreparer_t     preparer,
                                           const cuphyLdpcCbSubgroupKey_t* key,
                                           cuphyLdpcCbKernelChoice_t*      choice)
{
    if (preparer == nullptr || key == nullptr || choice == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphyLdpcCbChooseKernel(preparer->chooser, key, choice);
}

cuphyStatus_t cuphyLdpcCbPreparedLaunchGetWorkspaceSize(
    cuphyLdpcCbLaunchPreparer_t              preparer,
    const cuphyLdpcCbLaunchFamily_t*         family,
    const cuphyLdpcCbPreparedLaunchConfig_t* config,
    size_t*                                  workspace_size_bytes,
    size_t*                                  workspace_alignment_bytes)
{
    if (preparer == nullptr || family == nullptr || config == nullptr || workspace_size_bytes == nullptr ||
        workspace_alignment_bytes == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    const auto config_status = validate_prepared_config(*family, *config);
    if (config_status != CUPHY_STATUS_SUCCESS)
    {
        return config_status;
    }

    WorkspaceLayout layout{};
    if (!compute_workspace_layout(*config, layout))
    {
        return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
    }

    *workspace_size_bytes = layout.total_size;
    *workspace_alignment_bytes = layout.alignment;
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t cuphyLdpcCbPreparedLaunchInitInPlace(
    cuphyLdpcCbLaunchPreparer_t              preparer,
    const cuphyLdpcCbLaunchFamily_t*         family,
    const cuphyLdpcCbPreparedLaunchConfig_t* config,
    void*                                    workspace,
    size_t                                   workspace_size_bytes,
    cuphyLdpcCbPreparedLaunch_t*             launch)
{
    if (preparer == nullptr || family == nullptr || config == nullptr || workspace == nullptr || launch == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    *launch = nullptr;

    size_t required_size = 0;
    size_t required_alignment = 0;
    auto status = cuphyLdpcCbPreparedLaunchGetWorkspaceSize(preparer,
                                                            family,
                                                            config,
                                                            &required_size,
                                                            &required_alignment);
    if (status != CUPHY_STATUS_SUCCESS)
    {
        return status;
    }

    if (workspace_size_bytes < required_size)
    {
        return CUPHY_STATUS_INSUFFICIENT_RESOURCES;
    }
    if (!cuphy::ldpc::cb::is_aligned_to(workspace, static_cast<uint32_t>(required_alignment)))
    {
        return CUPHY_STATUS_UNSUPPORTED_ALIGNMENT;
    }

    WorkspaceLayout layout{};
    if (!compute_workspace_layout(*config, layout))
    {
        return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
    }

    auto* base = static_cast<uint8_t*>(workspace);
    auto* prepared = new (base + layout.object_offset) cuphyLdpcCbPreparedLaunch_s{};
    prepared->preparer = preparer;
    ++preparer->active_prepared_launches;
    prepared->family = *family;
    prepared->capacity = *config;
    prepared->subgroups = reinterpret_cast<cuphyLdpcCbSubgroupDesc_t*>(base + layout.subgroups_offset);
    prepared->spans = reinterpret_cast<cuphyLdpcCbBufferSpan_t*>(base + layout.spans_offset);
    prepared->prepared_num_subgroups = 0;
    prepared->prepared_total_codeblocks = 0;
    prepared->prepared = false;

    *launch = prepared;
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t cuphyLdpcCbPrepareLaunchWithChoice(cuphyLdpcCbPreparedLaunch_t         launch,
                                                 const cuphyLdpcCbLaunchBatchDesc_t* batch_desc,
                                                 const cuphyLdpcCbKernelChoice_t*    choice,
                                                 cuphyLdpcCbPreparedLaunchInfo_t*    info)
{
    if (launch == nullptr || batch_desc == nullptr || choice == nullptr || launch->preparer == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    launch->prepared = false;

    if (batch_desc->subgroups == nullptr || batch_desc->num_subgroups == 0)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if (batch_desc->num_subgroups > launch->capacity.max_subgroups)
    {
        return CUPHY_STATUS_INSUFFICIENT_RESOURCES;
    }

    const auto batch_status = cuphy::ldpc::cb::validate_launch_batch_desc(*batch_desc);
    if (batch_status != CUPHY_STATUS_SUCCESS)
    {
        return batch_status;
    }

    uint32_t total_codeblocks = 0;
    uint32_t total_spans = 0;
    auto status = count_batch_codeblocks_checked(*batch_desc, total_codeblocks, total_spans);
    if (status != CUPHY_STATUS_SUCCESS)
    {
        return status;
    }
    if (total_codeblocks > launch->capacity.max_total_codeblocks)
    {
        return CUPHY_STATUS_INSUFFICIENT_RESOURCES;
    }

    status = validate_choice_for_batch(*launch, *batch_desc, *choice);
    if (status != CUPHY_STATUS_SUCCESS)
    {
        return status;
    }

    launch->choice = *choice;

    status = copy_batch_desc(*launch, *batch_desc, total_spans);
    if (status != CUPHY_STATUS_SUCCESS)
    {
        return status;
    }

    launch->launch_batch.choice = &launch->choice;
    launch->launch_batch.batch_desc = &launch->batch_desc;

    status = cuphyLdpcCbBuildKernelNodeParams(&launch->launch_batch, &launch->spi_node_params);
    if (status != CUPHY_STATUS_SUCCESS)
    {
        return status;
    }

    launch->prepared_num_subgroups = batch_desc->num_subgroups;
    launch->prepared_total_codeblocks = total_codeblocks;
    launch->prepared = true;

    if (info != nullptr)
    {
        *info = cuphyLdpcCbPreparedLaunchInfo_t{
            launch->choice.compatibility_class_id,
            launch->choice.effective_algo,
            launch->choice.traits.codewords_per_cta,
            static_cast<uint8_t>(batch_desc->num_subgroups > 1 ? 1 : 0),
            batch_desc->num_subgroups,
            total_codeblocks,
            static_cast<uint32_t>(launch->spi_node_params.params.gridDimX),
        };
    }

    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t cuphyLdpcCbPrepareLaunch(cuphyLdpcCbPreparedLaunch_t         launch,
                                       const cuphyLdpcCbLaunchBatchDesc_t* batch_desc,
                                       cuphyLdpcCbPreparedLaunchInfo_t*    info)
{
    if (launch == nullptr || batch_desc == nullptr || launch->preparer == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if (batch_desc->subgroups == nullptr || batch_desc->num_subgroups == 0)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    cuphyLdpcCbKernelChoice_t choice{};
    const auto status = choose_for_batch(*launch, *batch_desc, choice);
    if (status != CUPHY_STATUS_SUCCESS)
    {
        return status;
    }
    return cuphyLdpcCbPrepareLaunchWithChoice(launch, batch_desc, &choice, info);
}

cuphyStatus_t cuphyLdpcCbGetKernelNodeParams(cuphyLdpcCbPreparedLaunch_t launch,
                                             CUDA_KERNEL_NODE_PARAMS*    node_params)
{
    if (launch == nullptr || node_params == nullptr || !launch->prepared)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    *node_params = launch->spi_node_params.params;
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t cuphyLdpcCbPreparedLaunchDeinit(cuphyLdpcCbPreparedLaunch_t launch)
{
    if (launch == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    auto* const preparer = launch->preparer;
    launch->~cuphyLdpcCbPreparedLaunch_s();
    --preparer->active_prepared_launches;
    return CUPHY_STATUS_SUCCESS;
}
