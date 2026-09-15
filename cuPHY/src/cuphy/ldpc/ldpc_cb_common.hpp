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

#ifndef CUPHY_LDPC_CB_COMMON_HPP
#define CUPHY_LDPC_CB_COMMON_HPP

#include "ldpc/ldpc_cb_kernel_api.h"

#include <cstdint>

namespace cuphy::ldpc::cb {

inline bool is_valid_crc_type(const cuphyLdpcCbCrc_t crc_type)
{
    return crc_type == CUPHY_LDPC_CB_CRC_NONE || crc_type == CUPHY_LDPC_CB_CRC_16 ||
           crc_type == CUPHY_LDPC_CB_CRC_24A || crc_type == CUPHY_LDPC_CB_CRC_24B;
}

inline bool is_valid_lifting_size(const uint16_t zc)
{
    constexpr uint16_t valid_lifting_sizes[] = {
        2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
        15,  16,  18,  20,  22,  24,  26,  28,  30,  32,  36,  40,  44,
        48,  52,  56,  60,  64,  72,  80,  88,  96,  104, 112, 120, 128,
        144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384,
    };

    for (const auto valid_zc : valid_lifting_sizes) {
        if (zc == valid_zc) {
            return true;
        }
    }
    return false;
}

inline bool is_valid_kb_for_base_graph(const cuphyLdpcCbBaseGraph_t bg, const uint16_t kb)
{
    if (bg == CUPHY_LDPC_CB_BG1) {
        return kb == CUPHY_LDPC_BG1_INFO_NODES;
    }

    constexpr uint16_t valid_bg2_kb[] = {6, 8, 9, CUPHY_LDPC_MAX_BG2_INFO_NODES};
    for (const auto valid_kb : valid_bg2_kb) {
        if (kb == valid_kb) {
            return true;
        }
    }
    return false;
}

inline bool is_aligned_to(const void* ptr, const uint32_t alignment_bytes)
{
    return alignment_bytes <= 1 || reinterpret_cast<std::uintptr_t>(ptr) % alignment_bytes == 0;
}

inline cuphyStatus_t validate_key(const cuphyLdpcCbSubgroupKey_t& key)
{
    if (key.bg != CUPHY_LDPC_CB_BG1 && key.bg != CUPHY_LDPC_CB_BG2) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if (key.Zc == 0 || key.parity_nodes == 0 || key.k == 0 || key.max_iters == 0) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if (key.Zc > CUPHY_LDPC_MAX_LIFTING_SIZE || !is_valid_lifting_size(key.Zc) ||
        key.parity_nodes < CUPHY_LDPC_MIN_PARITY_NODES || (key.k % key.Zc) != 0) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if (!is_valid_kb_for_base_graph(key.bg, static_cast<uint16_t>(key.k / key.Zc))) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if ((key.bg == CUPHY_LDPC_CB_BG1 && key.parity_nodes > CUPHY_LDPC_MAX_BG1_PARITY_NODES) ||
        (key.bg == CUPHY_LDPC_CB_BG2 && key.parity_nodes > CUPHY_LDPC_MAX_BG2_PARITY_NODES)) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if (!is_valid_crc_type(key.crc_type)) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return CUPHY_STATUS_SUCCESS;
}

inline cuphyStatus_t validate_buffer_span(const cuphyLdpcCbBufferSpan_t& span)
{
    if (span.llr_base == nullptr || span.bits_base == nullptr || span.num_cb == 0) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if (!is_aligned_to(span.bits_base, alignof(uint32_t))) {
        return CUPHY_STATUS_UNSUPPORTED_ALIGNMENT;
    }

    if (span.num_cb > 1 && (span.llr_stride_bytes == 0 || span.bits_stride_bytes == 0)) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if (span.num_cb > 1 && span.llr_out != nullptr && span.llr_out_stride_elems == 0) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return CUPHY_STATUS_SUCCESS;
}

inline cuphyStatus_t validate_subgroup_desc(const cuphyLdpcCbSubgroupDesc_t& subgroup)
{
    const auto key_status = validate_key(subgroup.key);
    if (key_status != CUPHY_STATUS_SUCCESS) {
        return key_status;
    }

    if (subgroup.spans == nullptr || subgroup.num_spans == 0) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    for (uint16_t i = 0; i < subgroup.num_spans; ++i) {
        const auto span_status = validate_buffer_span(subgroup.spans[i]);
        if (span_status != CUPHY_STATUS_SUCCESS) {
            return span_status;
        }
    }

    return CUPHY_STATUS_SUCCESS;
}

inline cuphyStatus_t validate_launch_batch_desc(const cuphyLdpcCbLaunchBatchDesc_t& batch_desc)
{
    if (batch_desc.subgroups == nullptr || batch_desc.num_subgroups == 0) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if (batch_desc.num_subgroups > 1) {
        return CUPHY_STATUS_NOT_SUPPORTED;
    }

    for (uint16_t i = 0; i < batch_desc.num_subgroups; ++i) {
        const auto subgroup_status = validate_subgroup_desc(batch_desc.subgroups[i]);
        if (subgroup_status != CUPHY_STATUS_SUCCESS) {
            return subgroup_status;
        }
    }

    return CUPHY_STATUS_SUCCESS;
}

inline cuphyStatus_t validate_launch_batch(const cuphyLdpcCbKernelLaunchBatch_t& launch_batch)
{
    if (launch_batch.choice == nullptr || launch_batch.batch_desc == nullptr) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    const auto& choice     = *launch_batch.choice;
    const auto& batch_desc = *launch_batch.batch_desc;

    if (batch_desc.subgroups == nullptr || batch_desc.num_subgroups == 0) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if (batch_desc.num_subgroups > choice.max_subgroups) {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }

    if (choice.supports_heterogeneous_batch == 0 && batch_desc.num_subgroups > 1) {
        return CUPHY_STATUS_NOT_SUPPORTED;
    }

    const auto batch_desc_status = validate_launch_batch_desc(batch_desc);
    if (batch_desc_status != CUPHY_STATUS_SUCCESS) {
        return batch_desc_status;
    }

    for (uint16_t subgroup_idx = 0; subgroup_idx < batch_desc.num_subgroups; ++subgroup_idx) {
        const auto& subgroup = batch_desc.subgroups[subgroup_idx];
        for (uint16_t span_idx = 0; span_idx < subgroup.num_spans; ++span_idx) {
            const auto& span = subgroup.spans[span_idx];
            if (!is_aligned_to(span.llr_base, choice.traits.llr_base_alignment_bytes) ||
                !is_aligned_to(span.bits_base, choice.traits.bits_base_alignment_bytes)) {
                return CUPHY_STATUS_UNSUPPORTED_ALIGNMENT;
            }
        }
    }

    return CUPHY_STATUS_SUCCESS;
}

inline uint32_t count_codeblocks(const cuphyLdpcCbSubgroupDesc_t& subgroup)
{
    uint32_t total_codeblocks = 0;
    for (uint16_t i = 0; subgroup.spans != nullptr && i < subgroup.num_spans; ++i) {
        total_codeblocks += subgroup.spans[i].num_cb;
    }
    return total_codeblocks;
}

inline uint32_t count_codeblocks(const cuphyLdpcCbLaunchBatchDesc_t& batch_desc)
{
    uint32_t total_codeblocks = 0;
    for (uint16_t i = 0; batch_desc.subgroups != nullptr && i < batch_desc.num_subgroups; ++i) {
        total_codeblocks += count_codeblocks(batch_desc.subgroups[i]);
    }
    return total_codeblocks;
}

} // namespace cuphy::ldpc::cb

#endif /* CUPHY_LDPC_CB_COMMON_HPP */
