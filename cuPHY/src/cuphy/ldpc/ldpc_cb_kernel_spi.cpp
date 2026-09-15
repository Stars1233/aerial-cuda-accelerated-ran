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

#include "cuphy_context.hpp"
#include "ldpc.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <utility>

static_assert(sizeof(cuphyLDPCDecodeLaunchConfig_t) <= CUPHY_LDPC_CB_KERNEL_NODE_PAYLOAD_BYTES,
              "LDPC CB kernel node payload is too small for V1 launch config");
static_assert(alignof(cuphyLDPCDecodeLaunchConfig_t) <= CUPHY_LDPC_CB_KERNEL_NODE_PAYLOAD_ALIGNMENT,
              "LDPC CB kernel node payload alignment is too small for V1 launch config");

struct cuphyLdpcCbKernelChooser_s
{
    cuphyLdpcCbKernelChooserConfig_t config;
    std::unique_ptr<ldpc::decoder>   decoder;

    cuphyLdpcCbKernelChooser_s(const cuphyLdpcCbKernelChooserConfig_t& cfg, std::unique_ptr<ldpc::decoder> dec) :
        config(cfg),
        decoder(std::move(dec))
    {
    }

    ~cuphyLdpcCbKernelChooser_s() = default;
    cuphyLdpcCbKernelChooser_s(const cuphyLdpcCbKernelChooser_s&) = delete;
    cuphyLdpcCbKernelChooser_s& operator=(const cuphyLdpcCbKernelChooser_s&) = delete;
    cuphyLdpcCbKernelChooser_s(cuphyLdpcCbKernelChooser_s&&) = delete;
    cuphyLdpcCbKernelChooser_s& operator=(cuphyLdpcCbKernelChooser_s&&) = delete;
};

namespace {

constexpr uint32_t kKnownConfigFlags = CUPHY_LDPC_CB_EARLY_TERM |
                                       CUPHY_LDPC_CB_THROUGHPUT_MODE |
                                       CUPHY_LDPC_CB_WRITE_SOFT_BITS;

cuphyStatus_t validate_static_config(const cuphyLdpcCbKernelChooserConfig_t& config)
{
    if (config.llr_type != CUPHY_R_16F)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if (!std::isfinite(config.clamp_value) || config.clamp_value <= 0.0f)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if ((config.config_flags & ~kKnownConfigFlags) != 0)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if ((config.config_flags & CUPHY_LDPC_CB_EARLY_TERM) != 0)
    {
        return CUPHY_STATUS_NOT_SUPPORTED;
    }

    return CUPHY_STATUS_SUCCESS;
}

uint32_t fnv1a_mix_bytes(uint32_t hash, const void* data, const size_t size)
{
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i)
    {
        hash ^= bytes[i];
        hash *= 16777619U;
    }
    return hash;
}

template <typename T>
uint32_t fnv1a_mix(uint32_t hash, const T& value)
{
    return fnv1a_mix_bytes(hash, &value, sizeof(value));
}

uint32_t compatibility_class_id(const cuphyLdpcCbKernelChooserConfig_t& config,
                                const cuphyLdpcCbSubgroupKey_t&         key,
                                const uint8_t                           effective_algo)
{
    uint32_t hash = 2166136261U;
    hash = fnv1a_mix(hash, config.llr_type);
    hash = fnv1a_mix(hash, config.clamp_value);
    hash = fnv1a_mix(hash, config.config_flags);
    hash = fnv1a_mix(hash, key.bg);
    hash = fnv1a_mix(hash, key.Zc);
    hash = fnv1a_mix(hash, key.parity_nodes);
    hash = fnv1a_mix(hash, key.k);
    hash = fnv1a_mix(hash, key.crc_type);
    hash = fnv1a_mix(hash, key.max_iters);
    hash = fnv1a_mix(hash, effective_algo);
    return hash == 0 ? 1U : hash;
}

uint32_t map_decode_flags(const uint32_t cb_flags)
{
    uint32_t decode_flags = 0;
    if ((cb_flags & CUPHY_LDPC_CB_THROUGHPUT_MODE) != 0)
    {
        decode_flags |= CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT;
    }
    if ((cb_flags & CUPHY_LDPC_CB_WRITE_SOFT_BITS) != 0)
    {
        decode_flags |= CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS;
    }
    if ((cb_flags & CUPHY_LDPC_CB_EARLY_TERM) != 0)
    {
        decode_flags |= CUPHY_LDPC_DECODE_EARLY_TERM;
    }
    return decode_flags;
}

size_t llr_element_size(const cuphyDataType_t llr_type)
{
    if (llr_type == CUPHY_R_16F)
    {
        return sizeof(uint16_t);
    }
    return 0;
}

uint32_t llrs_per_codeblock(const cuphyLdpcCbSubgroupKey_t& key)
{
    return static_cast<uint32_t>(((key.bg == CUPHY_LDPC_CB_BG1) ? 66 : 50) * key.Zc);
}

uint32_t bits_words_per_codeblock(const cuphyLdpcCbSubgroupKey_t& key)
{
    return static_cast<uint32_t>((key.k + 31) / 32);
}

bool compatible_with_choice_key(const cuphyLdpcCbKernelChoice_t& choice,
                                const cuphyLdpcCbSubgroupKey_t&  subgroup_key)
{
    const auto& choice_key = choice.key;
    return subgroup_key.bg == choice_key.bg &&
           subgroup_key.Zc == choice_key.Zc &&
           subgroup_key.parity_nodes == choice_key.parity_nodes &&
           subgroup_key.k == choice_key.k &&
           subgroup_key.crc_type == choice_key.crc_type &&
           subgroup_key.max_iters == choice_key.max_iters &&
           (subgroup_key.algo == 0 || subgroup_key.algo == choice.effective_algo);
}

std::uintptr_t ptr_value(const void* ptr)
{
    return reinterpret_cast<std::uintptr_t>(ptr);
}

bool can_extend_span(const cuphyLdpcCbData_t& first,
                     const cuphyLdpcCbData_t& next,
                     uint32_t&                llr_stride_bytes,
                     uint32_t&                bits_stride_bytes,
                     uint32_t&                llr_out_stride_elems)
{
    if (first.group_id == CUPHY_LDPC_CB_GROUP_ID_NONE || next.group_id != first.group_id)
    {
        return false;
    }

    const auto first_llr = ptr_value(first.llr_in);
    const auto next_llr  = ptr_value(next.llr_in);
    const auto first_bits = ptr_value(first.bits_out);
    const auto next_bits  = ptr_value(next.bits_out);
    if (next_llr <= first_llr || next_bits <= first_bits)
    {
        return false;
    }

    const auto llr_stride  = next_llr - first_llr;
    const auto bits_stride = next_bits - first_bits;
    if (llr_stride > UINT32_MAX || bits_stride > UINT32_MAX)
    {
        return false;
    }

    uint32_t soft_stride = 0;
    if (first.llr_out != nullptr || next.llr_out != nullptr)
    {
        if (first.llr_out == nullptr || next.llr_out == nullptr)
        {
            return false;
        }
        const auto first_soft = ptr_value(first.llr_out);
        const auto next_soft  = ptr_value(next.llr_out);
        if (next_soft <= first_soft)
        {
            return false;
        }
        const auto soft_stride_bytes = next_soft - first_soft;
        if (soft_stride_bytes > UINT32_MAX || (soft_stride_bytes % sizeof(uint16_t)) != 0)
        {
            return false;
        }
        soft_stride = static_cast<uint32_t>(soft_stride_bytes / sizeof(uint16_t));
    }

    llr_stride_bytes   = static_cast<uint32_t>(llr_stride);
    bits_stride_bytes  = static_cast<uint32_t>(bits_stride);
    llr_out_stride_elems = soft_stride;
    return true;
}

bool cb_matches_span_stride(const cuphyLdpcCbData_t& cb,
                            const cuphyLdpcCbData_t& first,
                            const uint16_t           offset,
                            const uint32_t           llr_stride_bytes,
                            const uint32_t           bits_stride_bytes,
                            const uint32_t           llr_out_stride_elems)
{
    if (cb.group_id != first.group_id)
    {
        return false;
    }
    if (ptr_value(cb.llr_in) != ptr_value(first.llr_in) + static_cast<std::uintptr_t>(offset) * llr_stride_bytes)
    {
        return false;
    }
    if (ptr_value(cb.bits_out) != ptr_value(first.bits_out) + static_cast<std::uintptr_t>(offset) * bits_stride_bytes)
    {
        return false;
    }
    if (first.llr_out == nullptr)
    {
        return cb.llr_out == nullptr;
    }
    return cb.llr_out != nullptr &&
           ptr_value(cb.llr_out) == ptr_value(first.llr_out) +
                                    static_cast<std::uintptr_t>(offset) * llr_out_stride_elems * sizeof(uint16_t);
}

cuphyStatus_t build_synthetic_launch_config(const cuphyLdpcCbKernelChooserConfig_t& config,
                                            const cuphyLdpcCbSubgroupKey_t&         key,
                                            const uint16_t                          num_codewords,
                                            cuphyLDPCDecodeLaunchConfig_t&          launch_config)
{
    std::memset(&launch_config, 0, sizeof(launch_config));

    if (key.Zc == 0 || (key.k % key.Zc) != 0)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    auto& decode_desc = launch_config.decode_desc;
    auto& decode_cfg  = decode_desc.config;

    decode_cfg.llr_type         = config.llr_type;
    decode_cfg.num_parity_nodes = static_cast<int16_t>(key.parity_nodes);
    decode_cfg.Z                = static_cast<int16_t>(key.Zc);
    decode_cfg.max_iterations   = static_cast<int16_t>(key.max_iters);
    decode_cfg.Kb               = static_cast<int16_t>(key.k / key.Zc);
    decode_cfg.flags            = map_decode_flags(config.config_flags);
    decode_cfg.BG               = static_cast<int16_t>(key.bg);
    decode_cfg.algo             = static_cast<int16_t>(key.algo);
    decode_cfg.clamp_value      = config.clamp_value;
    decode_cfg.workspace        = nullptr;

    const auto norm_status = ldpc::decoder::set_normalization(decode_cfg);
    if (norm_status != CUPHY_STATUS_SUCCESS)
    {
        return norm_status;
    }

    const int llrs_per_codeword = ((key.bg == CUPHY_LDPC_CB_BG1) ? 66 : 50) * key.Zc;
    const int bits_words        = (key.k + 31) / 32;

    decode_desc.num_tbs                        = 1;
    decode_desc.llr_input[0].addr              = reinterpret_cast<void*>(0x1000);
    decode_desc.llr_input[0].stride_elements   = llrs_per_codeword;
    decode_desc.llr_input[0].num_codewords     = num_codewords;
    decode_desc.tb_output[0].addr              = reinterpret_cast<uint32_t*>(0x2000);
    decode_desc.tb_output[0].stride_words      = bits_words;
    decode_desc.tb_output[0].num_codewords     = num_codewords;
    decode_desc.llr_output[0].addr             = ((config.config_flags & CUPHY_LDPC_CB_WRITE_SOFT_BITS) != 0) ?
                                                 reinterpret_cast<void*>(0x3000) :
                                                 nullptr;
    decode_desc.llr_output[0].stride_elements  = ((config.config_flags & CUPHY_LDPC_CB_WRITE_SOFT_BITS) != 0) ?
                                                 llrs_per_codeword :
                                                 0;
    decode_desc.llr_output[0].num_codewords    = ((config.config_flags & CUPHY_LDPC_CB_WRITE_SOFT_BITS) != 0) ?
                                                 num_codewords :
                                                 0;

    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t normalize_explicit_algo_status(const cuphyLdpcCbSubgroupKey_t& key,
                                             const cuphyStatus_t             status)
{
    if (key.algo != 0 && status == CUPHY_STATUS_INTERNAL_ERROR)
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    return status;
}

cuphyStatus_t infer_codewords_per_cta(cuphyLdpcCbKernelChooser_s&       chooser,
                                      const cuphyLdpcCbSubgroupKey_t&   key,
                                      const uint8_t                     effective_algo,
                                      uint8_t&                          codewords_per_cta)
{
    constexpr uint16_t max_probe_codewords = 255;

    for (uint16_t num_codewords = 1; num_codewords <= max_probe_codewords; ++num_codewords)
    {
        cuphyLDPCDecodeLaunchConfig_t launch_config{};
        auto status = build_synthetic_launch_config(chooser.config, key, num_codewords, launch_config);
        if (status != CUPHY_STATUS_SUCCESS)
        {
            return status;
        }

        launch_config.decode_desc.config.algo = static_cast<int16_t>(effective_algo);
        status = chooser.decoder->get_launch_config(launch_config);
        if (status != CUPHY_STATUS_SUCCESS)
        {
            return normalize_explicit_algo_status(key, status);
        }

        if (launch_config.kernel_node_params_driver.gridDimX > 1)
        {
            codewords_per_cta = static_cast<uint8_t>(num_codewords - 1);
            return CUPHY_STATUS_SUCCESS;
        }
    }

    codewords_per_cta = static_cast<uint8_t>(max_probe_codewords);
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t count_codeblocks_checked(const cuphyLdpcCbLaunchBatchDesc_t& batch_desc,
                                       uint32_t&                            total_codeblocks)
{
    uint64_t total = 0;
    for (uint16_t subgroup_idx = 0; subgroup_idx < batch_desc.num_subgroups; ++subgroup_idx)
    {
        const auto& subgroup = batch_desc.subgroups[subgroup_idx];
        for (uint16_t span_idx = 0; span_idx < subgroup.num_spans; ++span_idx)
        {
            total += subgroup.spans[span_idx].num_cb;
            if (total > std::numeric_limits<uint32_t>::max())
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
    }

    if (total == 0)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    total_codeblocks = static_cast<uint32_t>(total);
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t compute_grid_dim_x(const cuphyLdpcCbLaunchBatchDesc_t& batch_desc,
                                 const uint8_t                       codewords_per_cta,
                                 uint32_t&                           grid_dim_x)
{
    if (codewords_per_cta != 2)
    {
        uint32_t total_codeblocks = 0;
        const auto count_status = count_codeblocks_checked(batch_desc, total_codeblocks);
        if (count_status != CUPHY_STATUS_SUCCESS)
        {
            return count_status;
        }

        grid_dim_x = (total_codeblocks + codewords_per_cta - 1) / codewords_per_cta;
        return CUPHY_STATUS_SUCCESS;
    }

    uint64_t total_ctas = 0;
    for (uint16_t subgroup_idx = 0; subgroup_idx < batch_desc.num_subgroups; ++subgroup_idx)
    {
        const auto& subgroup = batch_desc.subgroups[subgroup_idx];
        for (uint16_t span_idx = 0; span_idx < subgroup.num_spans; ++span_idx)
        {
            const auto& span = subgroup.spans[span_idx];
            total_ctas += (span.num_cb + codewords_per_cta - 1) / codewords_per_cta;
            if (total_ctas > std::numeric_limits<uint32_t>::max())
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
    }

    if (total_ctas == 0)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    grid_dim_x = static_cast<uint32_t>(total_ctas);
    return CUPHY_STATUS_SUCCESS;
}

} // namespace

cuphyStatus_t cuphyCreateLdpcCbKernelChooser(cuphyLdpcCbKernelChooser_t*             chooser,
                                             const cuphyLdpcCbKernelChooserConfig_t* config)
{
    if (chooser == nullptr || config == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    *chooser = nullptr;

    const auto config_status = validate_static_config(*config);
    if (config_status != CUPHY_STATUS_SUCCESS)
    {
        return config_status;
    }

    try
    {
        cuphy_i::context ctx;
        auto dec = std::make_unique<ldpc::decoder>(ctx);
        auto* new_chooser = new cuphyLdpcCbKernelChooser_s(*config, std::move(dec));
        *chooser = new_chooser;
        return CUPHY_STATUS_SUCCESS;
    }
    catch (const std::bad_alloc&)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch (...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
}

cuphyStatus_t cuphyDestroyLdpcCbKernelChooser(cuphyLdpcCbKernelChooser_t chooser)
{
    if (chooser == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    delete chooser;
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t cuphyLdpcCbChooseKernel(cuphyLdpcCbKernelChooser_t      chooser,
                                      const cuphyLdpcCbSubgroupKey_t* key,
                                      cuphyLdpcCbKernelChoice_t*      choice)
{
    if (chooser == nullptr || key == nullptr || choice == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    const auto key_status = cuphy::ldpc::cb::validate_key(*key);
    if (key_status != CUPHY_STATUS_SUCCESS)
    {
        return key_status;
    }

    try
    {
        cuphyLDPCDecodeLaunchConfig_t launch_config{};
        auto status = build_synthetic_launch_config(chooser->config, *key, 1, launch_config);
        if (status != CUPHY_STATUS_SUCCESS)
        {
            return status;
        }

        status = chooser->decoder->get_launch_config(launch_config);
        if (status != CUPHY_STATUS_SUCCESS)
        {
            return normalize_explicit_algo_status(*key, status);
        }

        uint8_t codewords_per_cta = 1;
        status = infer_codewords_per_cta(*chooser,
                                         *key,
                                         static_cast<uint8_t>(launch_config.decode_desc.config.algo),
                                         codewords_per_cta);
        if (status != CUPHY_STATUS_SUCCESS)
        {
            return status;
        }

        const auto& params = launch_config.kernel_node_params_driver;
        std::memset(choice, 0, sizeof(*choice));
        choice->func                                     = params.func;
        choice->traits.shared_mem_bytes                  = static_cast<uint32_t>(params.sharedMemBytes);
        choice->traits.block_dim_x                       = static_cast<uint16_t>(params.blockDimX);
        choice->traits.block_dim_y                       = static_cast<uint16_t>(params.blockDimY);
        choice->traits.block_dim_z                       = static_cast<uint16_t>(params.blockDimZ);
        choice->traits.codewords_per_cta                 = codewords_per_cta;
        choice->traits.kernel_arg_count                  = 2;
        choice->traits.llr_base_alignment_bytes          = 16;
        choice->traits.bits_base_alignment_bytes         = alignof(uint32_t);
        choice->effective_algo                           = static_cast<uint8_t>(launch_config.decode_desc.config.algo);
        choice->compatibility_class_id                   = compatibility_class_id(chooser->config, *key, choice->effective_algo);
        choice->supports_heterogeneous_batch             = 0;
        choice->max_subgroups                            = 1;
        choice->key                                      = *key;
        choice->decode_config                            = launch_config.decode_desc.config;
        // Existing LDPC launch families use process-lifetime base graph descriptors
        // for arg 1, so caching this pointer does not borrow chooser-owned storage.
        choice->static_kernel_arg                        = launch_config.kernel_args[1];
        return CUPHY_STATUS_SUCCESS;
    }
    catch (const std::bad_alloc&)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch (...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
}

cuphyStatus_t cuphyLdpcCbBuildSubgroupDesc(const cuphyLdpcCbSubgroupKey_t* key,
                                           const cuphyLdpcCbData_t*        cb_data,
                                           uint16_t                        num_cb,
                                           cuphyLdpcCbBufferSpan_t*        span_storage,
                                           uint16_t*                       num_spans,
                                           cuphyLdpcCbSubgroupDesc_t*      subgroup_desc)
{
    if (key == nullptr || cb_data == nullptr || span_storage == nullptr || num_spans == nullptr ||
        subgroup_desc == nullptr || num_cb == 0)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    const auto key_status = cuphy::ldpc::cb::validate_key(*key);
    if (key_status != CUPHY_STATUS_SUCCESS)
    {
        return key_status;
    }

    const uint16_t span_capacity = *num_spans;
    uint16_t span_count = 0;
    uint16_t cb_idx = 0;

    while (cb_idx < num_cb)
    {
        if (span_count >= span_capacity)
        {
            return CUPHY_STATUS_INSUFFICIENT_RESOURCES;
        }

        const auto& first = cb_data[cb_idx];
        cuphyLdpcCbBufferSpan_t span{};
        span.llr_base           = first.llr_in;
        span.bits_base          = first.bits_out;
        span.bits_offset_bytes  = 0;
        span.llr_out            = first.llr_out;
        span.num_cb             = 1;

        uint32_t llr_stride_bytes = 0;
        uint32_t bits_stride_bytes = 0;
        uint32_t llr_out_stride_elems = 0;

        if (cb_idx + 1 < num_cb &&
            can_extend_span(first, cb_data[cb_idx + 1], llr_stride_bytes, bits_stride_bytes, llr_out_stride_elems))
        {
            span.llr_stride_bytes     = llr_stride_bytes;
            span.bits_stride_bytes    = bits_stride_bytes;
            span.llr_out_stride_elems = llr_out_stride_elems;

            uint16_t next_idx = cb_idx + 1;
            while (next_idx < num_cb &&
                   cb_matches_span_stride(cb_data[next_idx],
                                          first,
                                          static_cast<uint16_t>(next_idx - cb_idx),
                                          llr_stride_bytes,
                                          bits_stride_bytes,
                                          llr_out_stride_elems))
            {
                ++span.num_cb;
                ++next_idx;
            }
        }

        span_storage[span_count] = span;
        ++span_count;
        cb_idx = static_cast<uint16_t>(cb_idx + span.num_cb);
    }

    subgroup_desc->key       = *key;
    subgroup_desc->spans     = span_storage;
    subgroup_desc->num_spans = span_count;
    *num_spans               = span_count;

    return cuphy::ldpc::cb::validate_subgroup_desc(*subgroup_desc);
}

cuphyStatus_t cuphyLdpcCbBuildKernelNodeParams(const cuphyLdpcCbKernelLaunchBatch_t* launch_batch,
                                               cuphyLdpcCbKernelNodeParams_t*        node_params)
{
    if (launch_batch == nullptr || node_params == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    const auto batch_status = cuphy::ldpc::cb::validate_launch_batch(*launch_batch);
    if (batch_status != CUPHY_STATUS_SUCCESS)
    {
        return batch_status;
    }

    const auto& choice     = *launch_batch->choice;
    const auto& batch_desc = *launch_batch->batch_desc;

    if (batch_desc.status_out != nullptr)
    {
        return CUPHY_STATUS_NOT_SUPPORTED;
    }

    if (choice.traits.kernel_arg_count > 4)
    {
        return CUPHY_STATUS_INSUFFICIENT_RESOURCES;
    }
    if (choice.traits.kernel_arg_count != 2 || choice.static_kernel_arg == nullptr)
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    if (choice.traits.codewords_per_cta == 0)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if (batch_desc.num_subgroups != 1)
    {
        return CUPHY_STATUS_NOT_SUPPORTED;
    }

    const auto& subgroup = batch_desc.subgroups[0];
    if (!compatible_with_choice_key(choice, subgroup.key))
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    if (subgroup.num_spans > CUPHY_LDPC_DECODE_DESC_MAX_TB)
    {
        return CUPHY_STATUS_INSUFFICIENT_RESOURCES;
    }

    const size_t llr_elem_size = llr_element_size(choice.decode_config.llr_type);
    if (llr_elem_size == 0)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if ((choice.decode_config.flags & CUPHY_LDPC_DECODE_EARLY_TERM) != 0)
    {
        return CUPHY_STATUS_NOT_SUPPORTED;
    }

    std::memset(node_params, 0, sizeof(*node_params));
    auto* launch_config_ptr = new (node_params->payload) cuphyLDPCDecodeLaunchConfig_t{};
    auto& launch_config = *launch_config_ptr;
    auto& decode_desc   = launch_config.decode_desc;
    decode_desc.config  = choice.decode_config;
    decode_desc.num_tbs = subgroup.num_spans;

    const uint32_t default_llrs_per_cb  = llrs_per_codeblock(subgroup.key);
    const uint32_t default_bits_words   = bits_words_per_codeblock(subgroup.key);
    const bool write_soft_outputs =
        (choice.decode_config.flags & CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS) != 0;

    for (uint16_t span_idx = 0; span_idx < subgroup.num_spans; ++span_idx)
    {
        const auto& span = subgroup.spans[span_idx];
        auto& llr_input  = decode_desc.llr_input[span_idx];
        auto& tb_output  = decode_desc.tb_output[span_idx];
        auto& llr_output = decode_desc.llr_output[span_idx];

        llr_input.addr = const_cast<void*>(static_cast<const void*>(span.llr_base));
        if (span.num_cb > 1)
        {
            if ((span.llr_stride_bytes % llr_elem_size) != 0)
            {
                return CUPHY_STATUS_INVALID_ARGUMENT;
            }
            llr_input.stride_elements = static_cast<int32_t>(span.llr_stride_bytes / llr_elem_size);
        }
        else
        {
            llr_input.stride_elements = static_cast<int32_t>(default_llrs_per_cb);
        }
        llr_input.num_codewords = span.num_cb;

        if ((span.bits_offset_bytes % sizeof(uint32_t)) != 0 ||
            (span.bits_stride_bytes % sizeof(uint32_t)) != 0)
        {
            return CUPHY_STATUS_NOT_SUPPORTED;
        }
        tb_output.addr = reinterpret_cast<uint32_t*>(
            reinterpret_cast<uint8_t*>(span.bits_base) + span.bits_offset_bytes);
        tb_output.stride_words = (span.num_cb > 1) ?
                                 static_cast<int32_t>(span.bits_stride_bytes / sizeof(uint32_t)) :
                                 static_cast<int32_t>(default_bits_words);
        tb_output.num_codewords = span.num_cb;

        if (write_soft_outputs)
        {
            if (span.llr_out == nullptr)
            {
                return CUPHY_STATUS_INVALID_ARGUMENT;
            }
            if (span.num_cb > 1 && (span.llr_out_stride_elems % 2) != 0)
            {
                return CUPHY_STATUS_UNSUPPORTED_ALIGNMENT;
            }
            if (span.num_cb > 1 && span.llr_out_stride_elems > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
            llr_output.addr = span.llr_out;
            llr_output.stride_elements = (span.num_cb > 1) ?
                                         static_cast<int32_t>(span.llr_out_stride_elems) :
                                         static_cast<int32_t>(default_llrs_per_cb);
            llr_output.num_codewords = span.num_cb;
        }
        else
        {
            llr_output.addr             = nullptr;
            llr_output.stride_elements  = 0;
            llr_output.num_codewords    = 0;
        }
    }

    uint32_t total_codeblocks = 0;
    const auto count_status = count_codeblocks_checked(batch_desc, total_codeblocks);
    if (count_status != CUPHY_STATUS_SUCCESS)
    {
        return count_status;
    }

    const uint8_t codewords_per_cta = choice.traits.codewords_per_cta;
    uint32_t grid_dim_x = 0;
    const auto grid_status = compute_grid_dim_x(batch_desc, codewords_per_cta, grid_dim_x);
    if (grid_status != CUPHY_STATUS_SUCCESS)
    {
        return grid_status;
    }

    node_params->kernel_arg_count = choice.traits.kernel_arg_count;
    node_params->kernel_args[0]   = &launch_config.decode_desc;
    node_params->kernel_args[1]   = choice.static_kernel_arg;

    node_params->params.func           = choice.func;
    node_params->params.gridDimX       = grid_dim_x;
    node_params->params.gridDimY       = 1;
    node_params->params.gridDimZ       = 1;
    node_params->params.blockDimX      = choice.traits.block_dim_x;
    node_params->params.blockDimY      = choice.traits.block_dim_y;
    node_params->params.blockDimZ      = choice.traits.block_dim_z;
    node_params->params.sharedMemBytes = choice.traits.shared_mem_bytes;
    node_params->params.kernelParams   = node_params->kernel_args;
    node_params->params.extra          = nullptr;

    launch_config.kernel_args[0] = node_params->kernel_args[0];
    launch_config.kernel_args[1] = node_params->kernel_args[1];
    launch_config.kernel_node_params_driver = node_params->params;

    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t cuphyLdpcCbComputeNormalization(cuphyLdpcCbBaseGraph_t    bg,
                                              uint16_t                  parity_nodes,
                                              cuphyDataType_t           llr_type,
                                              cuphyLDPCNormalization_t* norm)
{
    if (norm == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    cuphyLDPCDecodeConfigDesc_t config{};
    config.BG               = static_cast<int16_t>(bg);
    config.num_parity_nodes = static_cast<int16_t>(parity_nodes);
    config.llr_type         = llr_type;

    const auto status = ldpc::decoder::set_normalization(config);
    if (status != CUPHY_STATUS_SUCCESS)
    {
        return status;
    }

    *norm = config.norm;
    return CUPHY_STATUS_SUCCESS;
}
