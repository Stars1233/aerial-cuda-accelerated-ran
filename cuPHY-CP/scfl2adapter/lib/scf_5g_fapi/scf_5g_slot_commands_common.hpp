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

#pragma once
#include <concepts>
#include <unordered_map>

#include "slot_command/slot_command.hpp"
using namespace slot_command_api;
#include "scf_5g_fapi.h"
#include "nv_phy_utils.hpp"
#include "cuphydriver_api.hpp"

#include "nv_phy_config_option.hpp"
#include "scf_5g_slot_commands_mod_comp.hpp"
#include "cuphy_internal.h"

#include "nv_phy_driver_proxy.hpp"

#define SCF_SLTCMD_TAG (NVLOG_TAG_BASE_SCF_L2_ADAPTER + 4) // "SCF.SLOTCMD"
namespace scf_5g_fapi {

    using pm_weight_map_t = std::unordered_map<uint32_t, slot_command_api::pm_weights_t>;
    using static_digBeam_weight_map_t = std::unordered_map<uint16_t, slot_command_api::digBeam_t>;

    #ifndef likely
    #define likely(x) __builtin_expect(!!(x), 1)
    #endif

    #ifndef unlikely
    #define unlikely(x) __builtin_expect(!!(x), 0)
    #endif

    #define getName(var)  #var

    const uint16_t DL_CHANNEL_MASK = (1 << channel_type::PDSCH_CSIRS) |
                                     (1 << channel_type::PDSCH) |
                                     (1 << channel_type::CSI_RS) |
                                     (1 << channel_type::PDSCH_DMRS) |
                                     (1 << channel_type::SSB_PBCH_DMRS) |
                                     (1 << channel_type::PDCCH_DL) |
                                     (1 << channel_type::PDCCH_UL) |
                                     (1 << channel_type::PDCCH_DMRS);

    static constexpr uint8_t CSI_RS_NUM_SYM[4] = { 1,  // NO_CDM
                                                   1 , // CDM2_FD
                                                   2 , // CDM4_FD2_TD2
                                                   4 };// CDM8_FD2_TD4

    static constexpr uint8_t CSI_RS_L0_NUM_SYM[19] = {      0, // ROW 0 invalid
                                                            1, // 1
                                                            1, // 2
                                                            1, // 3
                                                            1, // 4
                                                            2, // 5
                                                            1, // 6
                                                            2, // 7
                                                            1, // 8
                                                            1, // 9
                                                            1, // 10
                                                            2, // 11
                                                            1, // 12
                                                            2, // 13
                                                            1, // 14
                                                            1, // 15
                                                            2, // 16
                                                            1, // 17
                                                            1  // 18
                                                            };

    static constexpr uint8_t CSI_RS_L1_NUM_SYM[19] = {      0, // ROW 0 invalid
                                                            0, // 1
                                                            0, // 2
                                                            0, // 3
                                                            0, // 4
                                                            0, // 5
                                                            0, // 6
                                                            0, // 7
                                                            0, // 8
                                                            0, // 9
                                                            0, // 10
                                                            0, // 11
                                                            0, // 12
                                                            2, // 13
                                                            1, // 14
                                                            0, // 15
                                                            2, // 16
                                                            1, // 17
                                                            0  // 18
                                                            };

    static constexpr uint8_t CSI_RS_L_PRIME_NUM_SYM[19] = {     0, // ROW 0 invalid
                                                                1, // 1
                                                                1, // 2
                                                                1, // 3
                                                                1, // 4
                                                                1, // 5
                                                                1, // 6
                                                                1, // 7
                                                                2, // 8
                                                                1, // 9
                                                                2, // 10
                                                                1, // 11
                                                                2, // 12
                                                                1, // 13
                                                                2, // 14
                                                                4, // 15
                                                                1, // 16
                                                                2, // 17
                                                                4  // 18
                                                                };

    static constexpr std::array<uint8_t, 4> csirs_symL1_rows = {13, 14, 16, 17};

    inline uint16_t compute_N_prb_lbrm(uint16_t maxNumPrb)
    {
        uint16_t value = 0;
        if (maxNumPrb < 33) {
            value = 32;
        } else if (maxNumPrb >= 33 && maxNumPrb <= 66) {
            value = 66;
        } else if (maxNumPrb >= 67 && maxNumPrb <= 107) {
            value = 107;
        } else if (maxNumPrb >= 108 && maxNumPrb <= 135) {
            value = 135;
        } else if (maxNumPrb >= 136 && maxNumPrb <= 162) {
            value = 162;
        } else if (maxNumPrb >= 163 && maxNumPrb <= 217) {
            value = 217;
        } else if (maxNumPrb > 217) {
            value = 273;
        }
        return value;
    }

    inline bool ifAnySymbolPresent(sym_info_list_t& symbols, uint16_t channelMask)
    {
        if(!channelMask || symbols.size() == 0)
        {
            return  false;
        }
        else
        {
            for(int i = 0; i < channel_type::CHANNEL_MAX; i++)
            {
                auto j = (1 << i);
                if(((channelMask & j) == j) && symbols[0][i].size() > 0)
                    return true;
            }
        }

        return false;
    }

    inline int find_rightmost_bit64(uint64_t val)
    {
        return __builtin_ffsll(val) - 1;
    }

    inline int find_rightmost_bit32(uint32_t val)
    {
        return __builtin_ffs(val) - 1;
    }

    inline uint32_t count_set_bits(uint64_t val)
    {
        uint64_t n   = val;
        uint32_t cnt = 0;
        while(n != 0)
        {
            n = n & (n - 1);
            cnt += 1;
        }

        return cnt;
    }


    /**
     * setPrbReMask
     * n = number of Bits to Set
     * k = Offset in CUPHY_N_TONES_PER_PRB 
     */
  

    inline uint16_t setPrbReMask(int n, int k) {
        uint16_t num = 0;
        // Create a mask with n MSB bits set to 1
        uint16_t mask = ((1 << n) - 1) << (CUPHY_N_TONES_PER_PRB - n - k);
        
        // Apply the mask to the number
        num |= mask;

        // Return the modified number
        return num;
    }


    /**
     * Calculate the DMRS port mask from port bitmask, scrambling ID, and layer extension
     * 
     * This function computes the final DMRS port mask by applying bit shifts based on:
     * - Scrambling ID (scid): shifts by 0 or 8 bits (for scid 0 or 1)
     * - Layer extension (nlAbove16): shifts by 0 or 16 bits (for layers <=16 or >16)
     * 
     * @param[in] dmrsPortBmsk DMRS port bitmask (12-bit value for Type 1)
     * @param[in] scid Scrambling ID (0 or 1)
     * @param[in] nlAbove16 Indicator for layers above 16 (0 or 1)
     * @return Calculated DMRS port mask with applied shifts
     */
    inline uint64_t calculate_dmrs_port_mask(
        const uint16_t dmrsPortBmsk,
        const uint8_t scid,
        const uint8_t nlAbove16)
    {
        // Apply shifts in sequence: first scid shift, then nlAbove16 shift
        // Parentheses added for clarity (left-to-right associativity ensures correct order)
        return (static_cast<uint64_t>(dmrsPortBmsk) << (scid * 8)) << (16 * nlAbove16);
    }

    /**
     * @brief Tracks DMRS port bitmask positions and updates ap_index for UE groups
     * 
     * This template function works with both PDSCH and PUSCH UE parameter types.
     * It analyzes the DMRS port bitmask to determine which antenna ports are active
     * and updates the ap_index accordingly for proper UE group management.
     * 
     * @tparam UeType Type of UE parameters (cuphyPdschUePrm_t or cuphyPuschUePrm_t)
     * @param[in] ue Reference to UE parameters containing dmrsPortBmsk, scid, and rnti
     * @param[in,out] common Reference to PRB common info structure to update ap_index and active_eaxc_ids
     * 
     * @note The function expects UeType to have: dmrsPortBmsk (uint16_t), scid (uint8_t), rnti (uint16_t)
     * @note scid can only be 0 or 1, and dmrsPortBmsk can only have bits 0-11 set (max 12 bits)
     * @note This function is called after DMRS port bitmask is used to update portMask
     */
    template<typename UeType>
    inline void track_eaxcids_fh(UeType& ue, prb_info_common_t& common)
    {
        // Create shifted mask based on UE's scid (spatial correlation ID)
        // Each UE can have scid 0 or 1, which shifts the DMRS port bitmask by 0 or 8 bits
        uint64_t tempMask = calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
        
        while (tempMask != 0) {
            // Find the position of the least significant set bit (trailing zeros + 1)
            const uint32_t bit_pos = static_cast<uint32_t>(__builtin_ctzll(tempMask));

            // This bit position represents an active antenna port
            // Store the current ap_index in the active_eaxc_ids array at this bit position
            common.active_eaxc_ids[bit_pos] = common.ap_index;

            // Log the mapping for debugging
            //NVLOGD_FMT(SCF_SLTCMD_TAG, "ue.rnti {} common.active_eaxc_ids[{}] = {}",
            //           ue.rnti, bit_pos, static_cast<uint32_t>(common.ap_index));

            // Increment ap_index for the next UE in this group
            common.ap_index++;

            // Clear the least significant set bit for next iteration
            tempMask &= (tempMask - 1);
        }
    }

    inline void update_pm_weights_fh(cuphyPdschUePrm_t& ue, const pm_weight_map_t & pm_weight_map,
        scf_fapi_tx_precoding_beamforming_t& pmi_bf_pdu, prb_info_common_t& prbs, int32_t cell_index, bool mmimo_enabled)
    {
        uint16_t offset = 0;

        auto restore_defaults = [&ue, &prbs, mmimo_enabled] () {
            if(ue.rnti == UINT16_MAX)
            {
                //prbs.numApIndices += 2*ue.nUeLayers;
                prbs.portMask |= (1 <<  (2*ue.nUeLayers)) - 1 ;
            }
            else
            {
                prbs.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                if(mmimo_enabled)
                {
                    //NVLOGD_FMT(SCF_SLTCMD_TAG, "{}:{} update_pm_weights_fh: ue.rnti {} ue.dmrsPortBmsk {} ue.scid {}", __FILE__, __LINE__, ue.rnti, ue.dmrsPortBmsk, ue.scid);
                    track_eaxcids_fh(ue, prbs);
                }
            }
        };

        for (uint16_t i = 0; i < pmi_bf_pdu.num_prgs; i++) {
            if (pmi_bf_pdu.pm_idx_and_beam_idx[i + offset] == 0 || pmi_bf_pdu.dig_bf_interfaces == 0) {  // dig_bf_interfaces == 0 means dynamic beamforming or ZP-CSI-RS, which doesn't need precoding
                offset+= (pmi_bf_pdu.dig_bf_interfaces + 1);
                restore_defaults();
                continue;
            }
            uint32_t pmi = pmi_bf_pdu.pm_idx_and_beam_idx[i + offset] | cell_index << 16; /// PMI Unused
            offset+= (pmi_bf_pdu.dig_bf_interfaces + 1);
            auto pmw_iter = pm_weight_map.find(pmi);
            if(pmw_iter != pm_weight_map.end()) {
                prbs.portMask |= (1 << pmw_iter->second.ports) - 1;
            } else {
                restore_defaults();
                continue;
            }
        }
    }

    inline bool is_latest_bfw_coff_avail ( uint16_t currSfn, uint16_t currSlot, uint16_t prevSfn, uint16_t prevSlot)
    {
        // This is the most likely case, add for quick return
        if (currSlot == prevSlot)
        {
            return 0;
        }

        int slot_per_frame = nv::mu_to_slot_in_sf(1);

        uint32_t new_slots = currSfn * slot_per_frame + currSlot;
        uint32_t old_slots = prevSfn * slot_per_frame + prevSlot;

        // Old SFN/SLOT should not latter than new SFN/SLOT
        if (old_slots > new_slots)
        {
            new_slots += FAPI_SFN_MAX * slot_per_frame;
        }
        return (((new_slots - old_slots) == 1) ?NV_TRUE:NV_FALSE);
    }


    inline void update_prb_sym_list(slot_info_t& list, std::size_t prb_index, uint8_t startSym, uint8_t numSym, channel_type channel, enum ru_type ru) {
        auto begin = list.symbols.begin();
        auto iter_start = begin + startSym;
        auto iter_end = begin + startSym + numSym;
        std::for_each(iter_start , iter_end, [&prb_index, &channel, &ru](channel_info_list_t& channel_info_list) {
            if(channel != channel_type::PRACH && ru == SINGLE_SECT_MODE && channel_info_list[channel].size() > 0)
            {   
                return;
            }
            channel_info_list[channel].push_back(prb_index);
        });

        if(ru ==SINGLE_SECT_MODE)
        {
            if(channel == channel_type::PUCCH || channel == channel_type::PUSCH || channel == channel_type::SRS)
            {
                list.start_symbol_ul = startSym;
            }
            else if (channel == channel_type::PRACH)
            {
                list.start_symbol_ul = 0;  
            }
            else
            {
                list.start_symbol_dl = startSym;
            }
        }
    }

    inline void update_prb_sym_list(slot_info_t* list, std::size_t prb_index, uint8_t startSym, uint8_t numSym, channel_type channel, enum ru_type ru) {
        auto begin = list->symbols.begin();
        auto iter_start = begin + startSym;
        auto iter_end = begin + startSym + numSym;
        std::for_each(iter_start , iter_end, [&prb_index, &channel, &ru](channel_info_list_t& channel_info_list) {
            if(channel != channel_type::PRACH && ru == SINGLE_SECT_MODE && channel_info_list[channel].size() > 0)
            {
                return;
            }
            channel_info_list[channel].push_back(prb_index);
        });
        
        if(ru ==SINGLE_SECT_MODE)
        {
            if(channel == channel_type::PUCCH || channel == channel_type::PUSCH || channel == channel_type::SRS)
            {
                list->start_symbol_ul = startSym;
            }
            else if (channel == channel_type::PRACH)
            {
                list->start_symbol_ul = 0;  
            }
            else
            {
                list->start_symbol_dl = startSym;
            }
        }
    }

    inline void copy_prb_beam_list(prb_info_t & prb) {
        std::copy(prb.beams_array.begin(), prb.beams_array.begin() + prb.beams_array_size, prb.beams_array2.begin());
        prb.beams_array_size2 = prb.beams_array_size;        
    }

    /**
     * @brief Check if the beamforming parameters are valid
     * @param numPrg Number of PRGs
     * @param numDigBFI Number of DigBFIs
     * @param mmimo_enabled Whether MIMO is enabled
     * @return True if the beamforming parameters are valid, false otherwise
     */
    bool check_bf_pc_params(int numPrg, int numDigBFI, bool mmimo_enabled);

    void check_prb_info_size(size_t& prb_info_size);
    void update_beam_list(beamid_array_t& array, size_t& array_size, scf_fapi_tx_precoding_beamforming_t& pmi_bf_pdu, bool mmimo_enabled, prb_info_t& prb_info, int32_t cell_idx);
    void update_beam_list_uniq(beamid_array_t& array, size_t& array_size, scf_fapi_tx_precoding_beamforming_t& pmi_bf_pdu, prb_info_t& prb_info, bool mmimo_enabled, int32_t cell_idx);
    void update_beam_list(beamid_array_t& array, size_t& array_size, const scf_fapi_rx_beamforming_t& pmi_bf_pdu, bool mmimo_enabled, prb_info_t& prb_info, int32_t cell_idx);

    /**
     * @brief C++20 concept constraining types for update_static_bf_wt template
     *
     * Requires types to have three fields with exact types to prevent narrowing:
     * - num_prgs (uint16_t)
     * - prg_size (uint16_t)
     * - dig_bf_interfaces (uint8_t)
     */
    template<typename T>
    concept BeamformingPduType = requires(T t) {
        { t.num_prgs } -> std::same_as<uint16_t&>;
        { t.prg_size } -> std::same_as<uint16_t&>;
        { t.dig_bf_interfaces } -> std::same_as<uint8_t&>;
    };

    /**
     * @brief Update static beamforming weights for a beamforming PDU
     *
     * This function is called when a Tx Precoding/Beamforming or Rx Beamforming PDU is received from L2
     * for UEs and channels where L2 requests static beamforming.
     *
     * @tparam BeamformingPdu Beamforming PDU type; must satisfy BeamformingPduType
     * @param[in]     cell_index Logical cell index associated with the PDU
     * @param[in]     bf_pdu     Beamforming PDU providing num_prgs, prg_size, and dig_bf_interfaces
     * @param[in,out] prb_info   PRB info structure to be updated with static beamforming buffer metadata
     * @param[in]     beam_id    Beam identifier used to look up static DBT entries
     */
    template<BeamformingPduType BeamformingPdu>
    inline void update_static_bf_wt(const int32_t cell_index,
                                    const BeamformingPdu& bf_pdu,
                                    prb_info_t& prb_info,
                                    const uint16_t beam_id)
    {
        nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
        if ((bf_pdu.dig_bf_interfaces == 0) || !phyDriver.l1_staticBFWConfigured(cell_index))
        {
            NVLOGD_FMT(SCF_SLTCMD_TAG, "{} Static Beamforming is disabled for this PDU prb_info={}", __func__, reinterpret_cast<void*>(&prb_info.common));
            return;
        }

        NVLOGD_FMT(SCF_SLTCMD_TAG, "num_prgs={}, dig_bf_interfaces={}", static_cast<uint16_t>(bf_pdu.num_prgs), static_cast<uint8_t>(bf_pdu.dig_bf_interfaces));

        // Check if beamId is part of static DBT table
        // Return values: -1 = not in DBT table (predefined beam), 0 = in DBT table but not sent, 1 = in DBT table and sent
        const int isbeamInDBT = phyDriver.l1_getBeamWeightsSentFlag(cell_index, beam_id);

        if (isbeamInDBT == -1)
        {
            NVLOGD_FMT(SCF_SLTCMD_TAG, "Beamidx={} is a predefined beam ID (not in static DBT table)", beam_id);
            return;
        }

        NVLOGD_FMT(SCF_SLTCMD_TAG, "Beamidx={} is in static DBT table", beam_id);

        // Check if static beamforming weights should be sent once per cell (first time only) or every time
        const bool sendOncePerBeam = !phyDriver.l1_get_send_static_bfw_wt_all_cplane();
        if (sendOncePerBeam)
        {
            if (isbeamInDBT == 1)
            {
                // Beam weights already sent, skip
                return;
            }
            // Beam weights not sent yet (isbeamInDBT == 0), mark as sent
            phyDriver.l1_setBeamWeightsSentFlag(cell_index, beam_id);
        }

        NVLOGD_FMT(SCF_SLTCMD_TAG, "Beamidx={} entry available in DBT PDU IQ sent using extType=11 prb_info={}", beam_id, reinterpret_cast<void*>(&prb_info.common));
        prb_info.common.extType = 11;
        prb_info.common.isStaticBfwEncoded = true;
        prb_info.static_bfwCoeff_buf_info.num_prgs = bf_pdu.num_prgs;
        prb_info.static_bfwCoeff_buf_info.prg_size = bf_pdu.prg_size;
        prb_info.static_bfwCoeff_buf_info.dig_bf_interfaces = bf_pdu.dig_bf_interfaces;
        //prb_info.static_bfwCoeff_buf_info.nGnbAnt = bf_pdu.nGnbAnt;
    }

    inline void print_sym_prb_info(int sfn, int slot, const slot_info_t* slot_info, int cell_idx)
    {
        for(int symbol_id = 0; symbol_id < slot_info->symbols.size(); symbol_id++)
        {
            for (int channel_type = slot_command_api::channel_type::PDSCH_CSIRS; channel_type < slot_command_api::channel_type::CHANNEL_MAX; channel_type++)
            {
                for (auto prb_info_idx : slot_info->symbols[symbol_id][channel_type])
                {
                    const auto& prb_info              = slot_info->prbs[prb_info_idx];
                    NVLOGC_FMT(SCF_SLTCMD_TAG, "{}.{} Cell {}  Sym:{} Chan:{} idx:{} startPrbc:{} numPrbc:{} reMask:{:x} rbInc:{} numApIndices:{} portMask:{:x} numSymbols:{} freqOffset {} pdschPortMask {}",
                        sfn,slot,
                        cell_idx,
                        symbol_id,
                        channel_type,
                        prb_info_idx,
                        prb_info.common.startPrbc,
                        prb_info.common.numPrbc,
                        prb_info.common.reMask,
                        prb_info.common.useAltPrb,
                        prb_info.common.numApIndices,
                        prb_info.common.portMask,
                        prb_info.common.numSymbols,
                        prb_info.common.freqOffset,
                        prb_info.common.pdschPortMask
                        );
                }
            }
        }
    }

    inline comp_method get_comp_method(int32_t cell_index) {
        auto* phyDriver = nv::PHYDriverProxy::getInstancePtr();
        if (phyDriver && phyDriver->driver_exist()) {
            const auto & mplane_info = phyDriver->getMPlaneConfig(cell_index);
            return mplane_info.dl_comp_meth;
        }
        return comp_method::BLOCK_FLOATING_POINT;
    }

}
