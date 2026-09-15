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

#include "scf_5g_fapi_config_decode.hpp"
#include "scf_5g_fapi_tags.hpp"
#include "aerial/casts/casts.hpp"
#include "nvlog.hpp"

#define TAG (NVLOG_TAG_BASE_SCF_L2_ADAPTER + 3) // "SCF.PHY"

namespace scf_5g_fapi
{

bool decode_reconfig_tlvs(const scf_fapi_config_request_msg_t& config_request,
                          nv::cell_update_config&              out)
{
    const uint8_t* body_ptr = &config_request.msg_body.tlvs[0];
    const auto     num_tlvs = config_request.msg_body.num_tlvs;

    int32_t  prach_fd_index                 = -1;
    uint32_t prach_root_seq_unused_seq_index = 0;
    bool     dbt_present                    = false;

    for(auto remaining = num_tlvs; remaining != 0; --remaining)
    {
        auto* hdr = aerial::casts::assume_cast<scf_fapi_tl_t>(body_ptr);
        switch(hdr->tag)
        {
            case CONFIG_TLV_DL_BANDWIDTH:
                out.carrier_config_.dl_bandwidth = hdr->AsValue<uint16_t>();
                NVLOGI_FMT(TAG, "{} config request: Carrier DL Bandwidth (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), out.carrier_config_.dl_bandwidth);
                break;
            case CONFIG_TLV_UL_BANDWIDTH:
                out.carrier_config_.ul_bandwidth = hdr->AsValue<uint16_t>();
                NVLOGI_FMT(TAG, "{} config request: Carrier UL Bandwidth (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), out.carrier_config_.ul_bandwidth);
                break;
            case CONFIG_TLV_PHY_CELL_ID:
                out.cell_config_.phy_cell_id = hdr->AsValue<uint16_t>();
                NVLOGI_FMT(TAG, "{} config request: Physical Cell ID (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), out.cell_config_.phy_cell_id);
                break;
            case CONFIG_TLV_NUM_PRACH_FD_OCCASIONS:
                out.prach_config_.num_prach_fd_occasions = hdr->AsValue<uint8_t>();
                NVLOGI_FMT(TAG, "{} config request: Number of PRACH FD Occasions (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), out.prach_config_.num_prach_fd_occasions);
                break;
            case CONFIG_TLV_PRACH_ROOT_SEQ_INDEX:
                ++prach_fd_index;
                prach_root_seq_unused_seq_index = 0;
                if(prach_fd_index >= 0 && prach_fd_index < nv::NV_MAX_PRACH_FD_OCCASION_NUM)
                {
                    auto& root = out.prach_config_.root_sequence[prach_fd_index];
                    root.seq_index = hdr->AsValue<uint16_t>();
                    NVLOGI_FMT(TAG, "{} config request: PRACH Root Sequence Index (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), root.seq_index);
                }
                break;
            case CONFIG_TLV_NUM_ROOT_SEQ:
                if(prach_fd_index >= 0 && prach_fd_index < nv::NV_MAX_PRACH_FD_OCCASION_NUM)
                {
                    auto& root = out.prach_config_.root_sequence[prach_fd_index];
                    root.number_root_sequence = hdr->AsValue<uint8_t>();
                    NVLOGI_FMT(TAG, "{} config request: Number of Root Sequence (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), root.number_root_sequence);
                }
                break;
            case CONFIG_TLV_K1:
                if(prach_fd_index >= 0 && prach_fd_index < nv::NV_MAX_PRACH_FD_OCCASION_NUM)
                {
                    auto& root = out.prach_config_.root_sequence[prach_fd_index];
                    root.k1 = hdr->AsValue<uint16_t>();
                    NVLOGI_FMT(TAG, "{} config request: Frequency Offset K1 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), root.k1);
                }
                break;
            case CONFIG_TLV_PRACH_ZERO_CORR_CONF:
                if(prach_fd_index >= 0 && prach_fd_index < nv::NV_MAX_PRACH_FD_OCCASION_NUM)
                {
                    auto& root = out.prach_config_.root_sequence[prach_fd_index];
                    root.zero_conf = hdr->AsValue<uint8_t>();
                    NVLOGI_FMT(TAG, "{} config request: PRACH Zero Correlation Config (message ID {:X}) value {} prach_fd_index {}", __FUNCTION__, static_cast<int>(hdr->tag), root.zero_conf, prach_fd_index);
                }
                break;
            case CONFIG_TLV_NUM_UNUSED_ROOT_SEQ:
                if(prach_fd_index >= 0 && prach_fd_index < nv::NV_MAX_PRACH_FD_OCCASION_NUM)
                {
                    auto& root = out.prach_config_.root_sequence[prach_fd_index];
                    root.number_unused_sequence = hdr->AsValue<uint16_t>();
                    NVLOGI_FMT(TAG, "{} config request: Number of Unused Root Sequence (message ID {:X}) value {} prach_fd_index {}", __FUNCTION__, static_cast<int>(hdr->tag), root.number_unused_sequence, prach_fd_index);
                }
                break;
            case CONFIG_TLV_UNUSED_ROOT_SEQ:
                if(prach_fd_index >= 0 && prach_fd_index < nv::NV_MAX_PRACH_FD_OCCASION_NUM && prach_root_seq_unused_seq_index < nv::NV_MAX_UNUSED_ROOT_SEQUENCE_NUM)
                {
                    auto& root = out.prach_config_.root_sequence[prach_fd_index];
                    root.unused_sequence[prach_root_seq_unused_seq_index] = hdr->AsValue<uint16_t>();
                    NVLOGI_FMT(TAG, "{} config request: Unused Root Sequence (message ID {:X}) value {} prach_fd_index {}", __FUNCTION__, static_cast<int>(hdr->tag), root.unused_sequence[prach_root_seq_unused_seq_index], prach_fd_index);
                }
                ++prach_root_seq_unused_seq_index;
                break;
            case CONFIG_TLV_PRACH_CONFIG_INDEX:
                out.prach_config_.prach_conf_index = hdr->AsValue<uint8_t>();
                NVLOGI_FMT(TAG, "{} config request: PRACH Config Index (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), out.prach_config_.prach_conf_index);
                break;
            case CONFIG_TLV_RESTRICTED_SET_CONFIG:
                out.prach_config_.restricted_set_config = hdr->AsValue<uint8_t>();
                NVLOGI_FMT(TAG, "{} config request: PRACH Restricted Set Config (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), out.prach_config_.restricted_set_config);
                break;
            case CONFIG_TLV_VENDOR_DIGITAL_BEAM_TABLE_PDU:
                // Flag only; the caller stores the DBT PDU if the cell update succeeds.
                dbt_present = true;
                break;
        }
        // Round up TLV length to 4-byte boundary according to specs
        body_ptr += sizeof(scf_fapi_tl_t) + ((hdr->length + 3) / 4) * 4;
    }
    return dbt_present;
}

} // namespace scf_5g_fapi
