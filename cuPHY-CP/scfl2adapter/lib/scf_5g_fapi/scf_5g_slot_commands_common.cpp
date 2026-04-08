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

#include "scf_5g_slot_commands_common.hpp"
#include "nv_phy_driver_proxy.hpp"
#include "nvlog.h"
#include "nvlog_fmt.hpp"

#define TAG (NVLOG_TAG_BASE_SCF_L2_ADAPTER + 4) // "SCF.SLOTCMD"

namespace scf_5g_fapi {

    /**
     * Check if the beamforming parameters are valid
     * @param numPrg Number of PRGs
     * @param numDigBFI Number of DigBFIs
     * @param mmimo_enabled Whether MIMO is enabled
     * @return True if the beamforming parameters are valid, false otherwise
     */
    bool check_bf_pc_params(int numPrg, int numDigBFI, bool mmimo_enabled)
    {
        if(!mmimo_enabled && (numPrg > MAX_NUM_PRGS))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "Num PRG received {} larger than MAX {} prgs allocated", numPrg, MAX_NUM_PRGS);
            return false;
        }
        if(!mmimo_enabled && (numDigBFI > MAX_NUM_DIGBFI))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "Num DigBFIs received {} larger than MAX {} DigBFIs allocated", numDigBFI, MAX_NUM_DIGBFI);
            return false;
        }
        return true;
    }

    void check_prb_info_size(size_t& prb_info_size)
    {
        if(prb_info_size >= MAX_PRB_INFO)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "Number of prb_infos reached the max of {}, overwriting the last entry", MAX_PRB_INFO);
            prb_info_size--;
        }
    }

    void update_beam_list(beamid_array_t& array, size_t& array_size, scf_fapi_tx_precoding_beamforming_t& pmi_bf_pdu, bool mmimo_enabled, prb_info_t& prb_info, int32_t cell_idx)
    {
        const uint8_t dig_bf_interfaces = pmi_bf_pdu.dig_bf_interfaces;
        const uint16_t num_prgs = pmi_bf_pdu.num_prgs;
        const uint16_t prg_size = pmi_bf_pdu.prg_size;
        const uint16_t max_prgs = mmimo_enabled ? MAX_NUM_PRGS_DBF : MAX_NUM_PRGS;
        if (num_prgs > max_prgs || (array_size + num_prgs * dig_bf_interfaces) > MAX_BEAMS)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "update_beam_list: num_prgs={} exceeds max_prgs={} or would overflow beamid array (size={}, needed={}) - PDU skipped",
                static_cast<uint16_t>(num_prgs), max_prgs, array_size, array_size + num_prgs * dig_bf_interfaces);
            return;
        }

        std::size_t offset = sizeof(scf_fapi_tx_precoding_beamforming_t);
        //NVLOGD_FMT(TAG, "{} TX BeamForming PDU num_prgs={} prg_size={} dig_bf_interfaces={}", __FUNCTION__, num_prgs, prg_size, dig_bf_interfaces);

        uint8_t* buf = reinterpret_cast<uint8_t*>(&pmi_bf_pdu);
        if(dig_bf_interfaces)
        {
            for (uint i = 0; i < num_prgs ; i++)
            {
                uint16_t pmi = *reinterpret_cast<uint16_t*>(buf + offset);
                offset+= sizeof(uint16_t);
                //NVLOGI_FMT(TAG, "{} Offset= {} pmi= {}", __FUNCTION__, offset,pmi);
                uint16_t* beam_indexes = reinterpret_cast<uint16_t*>(buf + offset);
                if(mmimo_enabled)
                {
                    for (int j = 0; j < dig_bf_interfaces; j++)
                    {
                        //NVLOGI_FMT(TAG, "beam-id = {}", beam_indexes[j]);
                        array[array_size++] = beam_indexes[j];
                        update_static_bf_wt(cell_idx, pmi_bf_pdu, prb_info, beam_indexes[j]);
                    }
                }
                else
                {
                    for (int j = 0; j < dig_bf_interfaces; j++)
                    {
                        //NVLOGI_FMT(TAG, "beam-id = {}", beam_indexes[j]);
                        array[array_size++] = beam_indexes[j];
                    }
                }
                offset += sizeof(uint16_t) * dig_bf_interfaces;
            }
        }
#if 0
        NVLOGI_FMT(TAG, "{} TX BeamForming beam list size = {}", __FUNCTION__, array_size);
        for (std::size_t i = 0; i < array_size; i++) {
            NVLOGI_FMT(TAG, "{} beam [{}] = {}", __FUNCTION__, i, array[i]);
        }
#endif
    }

    void update_beam_list_uniq(beamid_array_t& array, size_t& array_size, scf_fapi_tx_precoding_beamforming_t& pmi_bf_pdu, prb_info_t& prb_info, bool mmimo_enabled, int32_t cell_idx) {
        update_beam_list(array, array_size, pmi_bf_pdu, mmimo_enabled, prb_info, cell_idx);
    }

    void update_beam_list(beamid_array_t& array, size_t& array_size, const scf_fapi_rx_beamforming_t& pmi_bf_pdu, bool  mmimo_enabled, prb_info_t& prb_info, int32_t cell_idx) {

        const uint8_t dig_bf_interfaces = pmi_bf_pdu.dig_bf_interfaces;
        const uint16_t num_prgs = pmi_bf_pdu.num_prgs;
        const uint16_t prg_size = pmi_bf_pdu.prg_size;
        const uint16_t max_prgs = mmimo_enabled ? MAX_NUM_PRGS_DBF : MAX_NUM_PRGS;
        if (num_prgs > max_prgs || (array_size + num_prgs * dig_bf_interfaces) > MAX_BEAMS)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "update_beam_list(RX): num_prgs={} exceeds max_prgs={} or would overflow beamid array (size={}, needed={}) - PDU skipped",
                static_cast<uint16_t>(num_prgs), max_prgs, array_size, array_size + num_prgs * dig_bf_interfaces);
            return;
        }
        //NVLOGD_FMT(TAG, "{} RX BeamForming PDU num_prgs = {} dig_bf_interfaces ={}", __FUNCTION__, num_prgs, dig_bf_interfaces);
        uint16_t offset = sizeof(scf_fapi_rx_beamforming_t);

        if (dig_bf_interfaces)
        {
            const uint8_t* buf = reinterpret_cast<const uint8_t*>(&pmi_bf_pdu);
            for (uint i = 0; i < num_prgs; i++)
            {
                //NVLOGI_FMT(TAG, "{} Offset= {} ", __FUNCTION__, offset);
                const uint16_t* beam_indexes = reinterpret_cast<const uint16_t*>(buf + offset);
                if(mmimo_enabled)
                {
                    for (int j = 0; j < dig_bf_interfaces; j++) {
                        //NVLOGI_FMT(TAG, "beam-id = {}", beam_indexes[j];);
                        array[array_size++] = beam_indexes[j];
                        update_static_bf_wt(cell_idx, pmi_bf_pdu, prb_info, beam_indexes[j]);
                    }
                }
                else
                {
                    for (int j = 0; j < dig_bf_interfaces; j++) {
                        //NVLOGI_FMT(TAG, "beam-id = {}", beam_indexes[j];);
                        array[array_size++] = beam_indexes[j];
                    }
                }
                offset += sizeof(uint16_t) * dig_bf_interfaces;
            }
        }
#if 0
         NVLOGI_FMT(TAG, "{} RX BeamForming beam list size = {}", __FUNCTION__, array_size);
        for (std::size_t i = 0; i < array_size; i++) {
            NVLOGI_FMT(TAG, "{} beam [{}] = {}", __FUNCTION__, i, array[i]);
        }
#endif
    }
}
