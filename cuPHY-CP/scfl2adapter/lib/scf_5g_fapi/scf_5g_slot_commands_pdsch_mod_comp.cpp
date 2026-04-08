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

#include "scf_5g_slot_commands_pdsch_mod_comp.hpp"
#include <array>

#define TAG (NVLOG_TAG_BASE_SCF_L2_ADAPTER + 4) // "SCF.SLOTCMD"

namespace scf_5g_fapi {

    static constexpr uint8_t DEFAULT_DMRS_OFFSET_MASK = 0x33;
    static constexpr uint16_t DEFAULT_DMRS_MASK_1 = 0xAAA;
    static constexpr uint16_t DEFAULT_DMRS_MASK_2 = ~DEFAULT_DMRS_MASK_1 & 0xFFF;
    
    // QAM modulation order array size: max QAM is 8 (CUPHY_QAM_256), divided by 2 gives index 4, so we need size 5
    static constexpr std::size_t QAM_MOD_ORDER_ARRAY_SIZE = 5;

    enum HandlerOptions : uint8_t{
        GROUP = 0,
        PRECODING = 1,
        BEAM = 2,
        MIMO = 3,
        COMP = 4
    };

    static auto no_op = [] (uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType, uint8_t qamModOrder) {};
    static auto no_op_pdsch_csirs = [] (uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType, uint8_t num_csirs_eaxcids, uint8_t qamModOrder) {};
    static constexpr uint8_t MAX_PDSCH_ONLY_FUNC_HANDLERS = 5;
    static PdschFunc funcHandlers [MAX_PDSCH_ONLY_FUNC_HANDLERS] = {no_op, no_op, no_op, no_op, no_op};
    static constexpr uint8_t MAX_PDSCH_CSIRS_FUNC_HANDLERS = MAX_PDSCH_ONLY_FUNC_HANDLERS;
    static PdschCsirsFunc funcPdschCsirsHandlers [MAX_PDSCH_CSIRS_FUNC_HANDLERS] = {no_op_pdsch_csirs, no_op_pdsch_csirs, no_op_pdsch_csirs, no_op_pdsch_csirs, no_op_pdsch_csirs};

    inline uint16_t get_dmrs_remask(uint16_t dmrsPorts) {
        uint16_t mask = 0;
        if ((DEFAULT_DMRS_OFFSET_MASK & dmrsPorts)) {
            mask = DEFAULT_DMRS_MASK_1;
        } else {
            mask = DEFAULT_DMRS_MASK_2;
        }
        return mask;
    }

    inline uint16_t get_data_remask(uint16_t dmrsPorts, bool is_data_muxed) {
        auto dmrs_remask = get_dmrs_remask(dmrsPorts);
        return (is_data_muxed? ~dmrs_remask: dmrs_remask) & 0xFFF;
    }

    inline void updatePdschPortMask(prb_info_t& prb_info, uint64_t portMask) {
        prb_info.common.portMask |= portMask;
        prb_info.common.pdschPortMask |= portMask;
    }

    // Updates the active eaxc ids in the prb_info.common.active_eaxc_ids array based on the pdschGroupPortMask and the record_active_eaxc_ids array
    inline void update_eaxcids_from_portmask(slot_command_api::prb_info_t& prb_info, uint64_t pdschGroupPortMask, std::array<int, MAX_DL_EAXCIDS>& record_active_eaxc_ids) {
        while (pdschGroupPortMask != 0) {
            // Find the position of the least significant set bit
            const uint32_t bit_pos = static_cast<uint32_t>(__builtin_ctzll(pdschGroupPortMask));

            if (bit_pos >= MAX_DL_EAXCIDS) {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}:{} pdschGroupPortMask {} bit_pos {} >= MAX_DL_EAXCIDS {}", 
                    __FILE__, __LINE__, pdschGroupPortMask, bit_pos, MAX_DL_EAXCIDS);
                break;
            }

            prb_info.common.active_eaxc_ids[bit_pos] = record_active_eaxc_ids[bit_pos];
            //NVLOGD_FMT(TAG, "update_eaxcids_from_portmask: prb_info.common.active_eaxc_ids[{}] = {}", bit_pos, record_active_eaxc_ids[bit_pos]);

            // Clear the least significant set bit for next iteration
            pdschGroupPortMask &= (pdschGroupPortMask - 1);
        }
    }
    /**
     * Create a new PDSCH PRB entry in the symbol PRB information structure
     * 
     * Creates a new Physical Resource Block (PRB) entry for PDSCH channel and initializes
     * it with the specified start PRB and number of PRBs. The function adds the PRB to the
     * prbs array, increments the size counter, and sets default downlink parameters.
     * 
     * @param[in,out] sym_prbs Pointer to symbol PRB information structure where new PRB will be added
     * @param[in] startPrb Starting PRB index for this allocation
     * @param[in] nPrb Number of contiguous PRBs in this allocation
     * 
     * @note The function checks PRB info size limits before adding
     * @note Sets direction to FH_DIR_DL and numApIndices to 0 by default
     */
    void createNewPdschPrbs(slot_info_t* sym_prbs, uint16_t startPrb, uint16_t nPrb) {

        check_prb_info_size(sym_prbs->prbs_size);
        auto& prbs{sym_prbs->prbs};

        prbs[sym_prbs->prbs_size] =
            prb_info_t(startPrb, nPrb);
        sym_prbs->prbs_size++;

        std::size_t index{sym_prbs->prbs_size - 1};
        prb_info_t& prb_info{prbs[index]};
        prb_info.common.direction = fh_dir_t::FH_DIR_DL;
        prb_info.common.numApIndices = 0;
        // NVLOGD_FMT(TAG, "{} startPrb {} nPrb {} index {} prb_info.common.portMask {}", __FUNCTION__, startPrb, nPrb, index, +prb_info.common.portMask);
    }
    
    /** Build qamModOrder to port mask mapping for all UEs in a group
     *
     * @param[in] cell_grp Cell group containing UE parameters
     * @param[in] nUes Number of UEs in the group
     * @param[in] ueIdxs Array of UE parameter indices
     * @param[in] pdsch_info PDSCH information containing codeword details
     * @param[in] mmimo_enabled Flag indicating if mMIMO is enabled
     * @return Array indexed by qamModOrder/2 containing aggregated port masks
     */
    static const std::array<uint64_t, QAM_MOD_ORDER_ARRAY_SIZE>& buildQamModOrderToPortMask(
        const auto& cell_grp,
        const uint16_t nUes,
        const uint16_t* ueIdxs,
        const IFhCallbackContext& fh_context,
        const bool mmimo_enabled)
    {
        static std::array<uint64_t, QAM_MOD_ORDER_ARRAY_SIZE> qamModOrderToPortMask{};
        
        // Initialize array with zeroes
        qamModOrderToPortMask.fill(0);
        
        // Iterate through all UEs in the group
        for (uint16_t ueIdx = 0; ueIdx < nUes; ++ueIdx) {
            const auto& ueParams = cell_grp.pUePrms[ueIdxs[ueIdx]];
            
            // Get the port mask for this UE
            uint64_t uePortMask{};
            if (ueParams.rnti == UINT16_MAX && !mmimo_enabled) {
                uePortMask = ueParams.nUeLayers > 0 ? 1ULL << (2 * ueParams.nUeLayers - 1) : 0;
            } else {
                uePortMask = calculate_dmrs_port_mask(ueParams.dmrsPortBmsk, ueParams.scid, ueParams.nlAbove16);
            }
            
            // Each UE can have multiple codewords; aggregate ports by qamModOrder
            for (uint8_t cwIdx = 0; cwIdx < ueParams.nCw; ++cwIdx) {
                const auto cwGlobalIdx = ueParams.pCwIdxs[cwIdx];
                const auto modOrder = fh_context.pdsch_ue_cw_qam_mod_order(cwGlobalIdx);
                
                // Aggregate the port mask for this qamModOrder (use modOrder >> 1 as array index)
                qamModOrderToPortMask[modOrder >> 1] |= uePortMask;
            }
        }
        
        return qamModOrderToPortMask;
    }

    /** Build qamModOrder to DMRS port mask mapping with common/non-common separation
     *
     * @param[in] cell_grp Cell group containing UE parameters
     * @param[in] nUes Number of UEs in the group
     * @param[in] ueIdxs Array of UE parameter indices
     * @param[in] pdsch_info PDSCH information containing codeword details
     * @param[in] mmimo_enabled Flag indicating if mMIMO is enabled
     * @return Pair of arrays: first for common DMRS ports, second for non-common DMRS ports
     */
    static std::pair<std::array<uint64_t, QAM_MOD_ORDER_ARRAY_SIZE>, std::array<uint64_t, QAM_MOD_ORDER_ARRAY_SIZE>> 
    buildQamModOrderToDmrsPortMasks(
        const auto& cell_grp,
        const uint16_t nUes,
        const uint16_t* ueIdxs,
        const IFhCallbackContext& fh_context)
    {
        std::array<uint64_t, QAM_MOD_ORDER_ARRAY_SIZE> qamModOrderToCommonMask{};
        std::array<uint64_t, QAM_MOD_ORDER_ARRAY_SIZE> qamModOrderToNonCommonMask{};
        
        // Initialize arrays with zeroes
        qamModOrderToCommonMask.fill(0);
        qamModOrderToNonCommonMask.fill(0);
        
        // Iterate through all UEs in the group
        for (uint16_t ueIdx = 0; ueIdx < nUes; ++ueIdx) {
            const auto& ueParams = cell_grp.pUePrms[ueIdxs[ueIdx]];
            
            // Separate common and non-common DMRS ports
            const auto mask1 = ueParams.dmrsPortBmsk;
            const auto common = mask1 & DEFAULT_DMRS_OFFSET_MASK;
            const auto non_common = mask1 & ~(common);
            
            // Calculate port masks for common and non-common separately
            const uint64_t commonPortMask = calculate_dmrs_port_mask(common, ueParams.scid, ueParams.nlAbove16);
            const uint64_t nonCommonPortMask = calculate_dmrs_port_mask(non_common, ueParams.scid, ueParams.nlAbove16);
            
            // Each UE can have multiple codewords; aggregate ports by qamModOrder
            for (uint8_t cwIdx = 0; cwIdx < ueParams.nCw; ++cwIdx) {
                const auto cwGlobalIdx = ueParams.pCwIdxs[cwIdx];
                const auto modOrder = fh_context.pdsch_ue_cw_qam_mod_order(cwGlobalIdx);
                const auto arrayIdx = modOrder >> 1;
                
                // Aggregate the port masks for this qamModOrder
                qamModOrderToCommonMask[arrayIdx] |= commonPortMask;
                qamModOrderToNonCommonMask[arrayIdx] |= nonCommonPortMask;
            }
        }
        
        return std::make_pair(qamModOrderToCommonMask, qamModOrderToNonCommonMask);
    }

    /** Helper function to create a PRB with port mask and call function handlers
     *
     * @param[in,out] sym_prbs Symbol PRB information
     * @param[in,out] prbs PRB array
     * @param[in] startPrb Starting PRB index
     * @param[in] nPrb Number of PRBs
     * @param[in] portMask Port mask to apply
     * @param[in] reMask Resource element mask to apply
     * @param[in] mmimo_enabled Flag indicating if mMIMO is enabled
     * @param[in,out] record_active_eaxc_ids Array to record active eaxc IDs
     * @param[in] tempPdschSym PDSCH symbol index
     * @param[in] tempPdschNum PDSCH number
     * @param[in] pm_map PM weight map
     * @param[in,out] pdsch_fh_param PDSCH FH parameters
     * @param[in,out] slot_detail Slot detail information
     * @param[in] ru RU type
     * @param[in] chType Channel type
     * @param[in] qamModOrder QAM modulation order
     * @param[in] Cell Group command 
     */
    inline void createPrbWithPortMask(
        slot_info_t* sym_prbs,
        prb_info_t* prbs,
        uint16_t startPrb,
        uint16_t nPrb,
        uint64_t portMask,
        uint16_t reMask,
        bool mmimo_enabled,
        std::array<int, MAX_DL_EAXCIDS>& record_active_eaxc_ids,
        uint8_t tempPdschSym,
        uint8_t tempPdschNum,
        const pm_weight_map_t& pm_map,
        const IFhCallbackContext& fh_context,
        const PdschFhParamsView& pdsch_fh_param,
        nv::slot_detail_t* slot_detail,
        ru_type ru,
        channel_type chType,
        uint8_t qamModOrder)
    {
        createNewPdschPrbs(sym_prbs, startPrb, nPrb);
        std::size_t index{sym_prbs->prbs_size - 1};
        prb_info_t& prb_info{prbs[index]};
        
        prb_info.common.reMask = reMask;
        updatePdschPortMask(prb_info, portMask);
        
        if (mmimo_enabled) {
            update_eaxcids_from_portmask(prb_info, portMask, record_active_eaxc_ids);
        }
        
        // Call function handlers
        for (auto hdlIndex = 2; hdlIndex < MAX_PDSCH_ONLY_FUNC_HANDLERS; hdlIndex++) {
            funcHandlers[hdlIndex](tempPdschSym, tempPdschNum, pm_map, fh_context, pdsch_fh_param, slot_detail, ru, chType, qamModOrder);
        }
    }

    template <bool modcomp_enabled = false>
    void addDisjointCsirsPrbs(slot_info_t* sym_prbs, prb_info_t& csirs_prb_info, uint16_t overlap_start_prb, uint16_t overlap_num_prb, uint16_t symbol_id , uint16_t num_dl_prb);

    inline void updatePortMask(prb_info_t& prb_info, uint64_t portMask) {
        prb_info.common.portMask |= portMask;
    }

    uint8_t getDmrsPorts(const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param) {
        auto& grp = pdsch_fh_param.grp();
        const auto& cell_grp = fh_context.pdsch_cell_grp_info();
        auto nUes = grp.nUes;
        auto ueIdxs = grp.pUePrmIdxs;
        
        uint8_t dmrsPorts = 0;
        for (uint16_t ueIdx = 0; ueIdx < nUes; ueIdx++) {
            auto& ueParams = cell_grp.pUePrms[ueIdxs[ueIdx]];
            dmrsPorts |= ueParams.dmrsPortBmsk;
        }
        return dmrsPorts;
    }

    /**
     * Get the maximum QAM modulation order for all UEs in a UE group
     *
     * @param[in] pdsch_fh_param PDSCH FH preparation parameters containing UE group info
     * @return Maximum QAM modulation order (2=QPSK, 4=16QAM, 6=64QAM, 8=256QAM, 10=1024QAM)
     */
    uint8_t getMaxModOrderInUeGroup(const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param) {
        auto& grp = pdsch_fh_param.grp();
        const auto& cell_grp = fh_context.pdsch_cell_grp_info();
        const auto nUes = grp.nUes;
        const auto ueIdxs = grp.pUePrmIdxs;
        
        uint8_t maxModOrder{};
        for (uint16_t ueIdx = 0; ueIdx < nUes; ++ueIdx) {
            const auto& ueParams = cell_grp.pUePrms[ueIdxs[ueIdx]];
            // Each UE can have multiple codewords; check all of them
            for (uint8_t cwIdx = 0; cwIdx < ueParams.nCw; ++cwIdx) {
                const auto cwGlobalIdx = ueParams.pCwIdxs[cwIdx];
                const auto modOrder = fh_context.pdsch_ue_cw_qam_mod_order(cwGlobalIdx);
                maxModOrder = std::max(maxModOrder, modOrder);
            }
        }
        return maxModOrder;
    }
    // returns total PDSCH Ports set per group;
    // Also records the active eaxc ids in the record_active_eaxc_ids array
    uint64_t getPdschPortMask(const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, std::array<int, MAX_DL_EAXCIDS>& record_active_eaxc_ids) {
        auto& grp = pdsch_fh_param.grp();
        const auto& cell_grp = fh_context.pdsch_cell_grp_info();
        auto nUes = grp.nUes;
        auto ueIdxs = grp.pUePrmIdxs;
        uint64_t pdschPortPerGroup = 0ULL;
        uint32_t ap_index = 0;
        for (uint16_t ueIdx = 0; ueIdx < nUes; ueIdx++) {
            auto& ueParams = cell_grp.pUePrms[ueIdxs[ueIdx]];
            if (ueParams.rnti == UINT16_MAX && !pdsch_fh_param.mmimo_enabled()) {  // For 4T4R O-RUs, SI-RNTI date with a single layer transmission will be replicated for the second antenna
                pdschPortPerGroup |= 1ULL << (2 * ueParams.nUeLayers - 1);
            } else {
                uint64_t tempMask = calculate_dmrs_port_mask(ueParams.dmrsPortBmsk, ueParams.scid, ueParams.nlAbove16);
                pdschPortPerGroup |= tempMask;
                while (pdsch_fh_param.mmimo_enabled() && tempMask != 0) {
                    // Find the position of the least significant set bit (trailing zeros + 1) for 32DL
                    const uint32_t bit_pos = static_cast<uint32_t>(__builtin_ctzll(tempMask));

                    if (bit_pos >= MAX_DL_EAXCIDS) {
                        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}:{} tempMask {} bit_pos {} >= MAX_DL_EAXCIDS {}", 
                            __FILE__, __LINE__, tempMask, bit_pos, MAX_DL_EAXCIDS);
                        break;
                    }

                    // This bit position represents an active antenna port
                    // Store the current ap_index in the active_eaxc_ids array at this bit position
                    record_active_eaxc_ids[bit_pos] = ap_index;
                    // Log the mapping for debugging
                    //NVLOGD_FMT(TAG, "getPdschPortMask: ue.rnti {} record_active_eaxc_ids[{}] = {}",
                    //            ueParams.rnti, bit_pos, ap_index);
                    // Increment ap_index for the next UE in this group
                    ap_index++;

                    // Clear the least significant set bit for next iteration
                    tempMask &= (tempMask - 1);
                }
            }
        }
        return pdschPortPerGroup;
    }

    std::pair<uint64_t, uint64_t> getCsirsPortMask(prb_info_idx_list_t& csi_rs_prb_info, prb_info_t* prbs, uint8_t num_csirs_eaxcids) {
        uint16_t prb_index = 0;
        auto& csirs_prb = prbs[csi_rs_prb_info[prb_index]];
        uint64_t portMask = 0;
        uint64_t flowMask = 0;
        std::for_each(csi_rs_prb_info.begin(), csi_rs_prb_info.end(), [&prbs, &portMask, &flowMask, &num_csirs_eaxcids](const auto &elem) {
            uint64_t tempPortMask = prbs[elem].common.portMask;
            uint64_t msb = 63 - __builtin_clzll(tempPortMask);
            if (msb >= num_csirs_eaxcids) {
                uint64_t newFlowMask = 0;
                while (tempPortMask) {
                    uint8_t bit_index = __builtin_ctzll(tempPortMask);
                    newFlowMask |= (1ULL << (bit_index % num_csirs_eaxcids));
                    tempPortMask &= (tempPortMask - 1);
                }
                flowMask |= newFlowMask;
                portMask |= prbs[elem].common.portMask;
            } else {
                portMask |= prbs[elem].common.portMask;
                flowMask |= prbs[elem].common.portMask;
            }
        });

        return std::make_pair(flowMask, portMask);
    }
    inline uint64_t generateNonCommonCsirsPorts(uint64_t csirsPortMask, uint64_t csirsFlowMask, uint64_t nonCommonCsirsFlows, uint8_t num_csirs_eaxcids) {
        uint64_t nonCommonCsirsPorts = 0;
        if(csirsPortMask == csirsFlowMask)
        {
            nonCommonCsirsPorts = nonCommonCsirsFlows;
        }
        else
        {
            for (int bit_idx = 0; bit_idx < 64; bit_idx++) {
                if (csirsPortMask & (1ULL << bit_idx)) {
                    uint8_t resultant_idx = bit_idx % num_csirs_eaxcids;
                    if (nonCommonCsirsFlows & (1ULL << resultant_idx)) {
                        nonCommonCsirsPorts |= (1ULL << bit_idx);
                    }
                }
            }
        }
        return nonCommonCsirsPorts;
    }
    
    inline void handleNewGroup(uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, slot_command_api::channel_type chType, uint8_t qamModOrder) {
        auto& grp = pdsch_fh_param.grp();
        auto ue_grp_index = pdsch_fh_param.ue_grp_index();
        auto mmimo_enabled = pdsch_fh_param.mmimo_enabled();
        auto sym_prbs{pdsch_fh_param.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};
        const auto& cell_grp = fh_context.pdsch_cell_grp_info();
        const auto nUes = grp.nUes;
        const auto ueIdxs = grp.pUePrmIdxs;
        //NVLOGD_FMT( TAG, "{}:{} tempPdschSym {} tempPdschNum {} chType {} \n", __FUNCTION__, __LINE__, +tempPdschSym, +tempPdschNum, +chType);
        uint32_t raTypeSize = 0;
        switch (grp.resourceAlloc) {
            case 0:
                raTypeSize = fh_context.pdsch_num_ra_type0_info(ue_grp_index);
                break;
            case 1:
                raTypeSize = 1;
                break;
        }

        // NVLOGD_FMT(TAG, "{} raTypeSize {} resourceAlloc {}", __FUNCTION__, raTypeSize, +grp.resourceAlloc);

        for (uint32_t i = 0; i < raTypeSize; i++) {
            uint16_t startPrb = 0, nPrb = 0;
            switch (grp.resourceAlloc) {
                case 0:
                    startPrb = fh_context.pdsch_ra_type0_info(i, ue_grp_index).start_prb + grp.startPrb;
                    nPrb =  fh_context.pdsch_ra_type0_info(i, ue_grp_index).num_prb;
                    break;
                case 1:
                    startPrb = grp.startPrb;
                    nPrb = grp.nPrb;
                    break;
            }
            std::array<int, MAX_DL_EAXCIDS> record_active_eaxc_ids{};
            if(mmimo_enabled)
            {
                record_active_eaxc_ids.fill(-1); // Initialize with -1 to indicate no active eaxc ids
            }
            auto pdschGroupPortMask = getPdschPortMask(fh_context, pdsch_fh_param, record_active_eaxc_ids);

            // select from 1 - >end handler functions
                // NVLOGD_FMT(TAG, "{} hdlIndex {} start", __FUNCTION__, hdlIndex);

                switch (chType) {
                    case PDSCH_DMRS: {
                        // Build qamModOrder to DMRS port mask mapping with common/non-common separation
                        const auto [qamModOrderToCommonMask, qamModOrderToNonCommonMask] = 
                            buildQamModOrderToDmrsPortMasks(cell_grp, nUes, ueIdxs, fh_context);
                        
                        // Create PRBs for each unique qamModOrder with aggregated port masks
                        for (std::size_t idx = 0; idx < qamModOrderToCommonMask.size(); ++idx) {
                            const auto commonPortMask = qamModOrderToCommonMask[idx];
                            const auto nonCommonPortMask = qamModOrderToNonCommonMask[idx];
                            
                            // Skip if no ports for this qamModOrder
                            if (commonPortMask == 0 && nonCommonPortMask == 0) continue;
                            
                            const auto modOrder = idx << 1;  // Reconstruct qamModOrder from array index
                            
                            // Create PRB for common DMRS ports if present
                            if (commonPortMask != 0) {
                                createPrbWithPortMask(sym_prbs, prbs, startPrb, nPrb, commonPortMask, 
                                    DEFAULT_DMRS_MASK_1, mmimo_enabled, record_active_eaxc_ids,
                                    tempPdschSym, tempPdschNum, pm_map, fh_context, pdsch_fh_param, 
                                    slot_detail, ru, chType, modOrder);
                            }
                            
                            // Create PRB for non-common DMRS ports if present
                            if (nonCommonPortMask != 0) {
                                createPrbWithPortMask(sym_prbs, prbs, startPrb, nPrb, nonCommonPortMask, 
                                    DEFAULT_DMRS_MASK_2, mmimo_enabled, record_active_eaxc_ids,
                                    tempPdschSym, tempPdschNum, pm_map, fh_context, pdsch_fh_param, 
                                    slot_detail, ru, chType, modOrder);
                            }
                        }
                    }
                        break;
                        
                    case PDSCH: {                        
                        // Build qamModOrder to port mask mapping
                        const auto& qamModOrderToPortMask = buildQamModOrderToPortMask(
                            cell_grp, nUes, ueIdxs, fh_context, mmimo_enabled);
                        
                        // Create PRBs for each unique qamModOrder with aggregated port masks
                        for (std::size_t idx = 0; idx < qamModOrderToPortMask.size(); ++idx) {
                            const auto portMask = qamModOrderToPortMask[idx];
                            if (portMask == 0) continue;
                            
                            const auto modOrder = idx << 1;  // Reconstruct qamModOrder from array index
                            
                            createPrbWithPortMask(sym_prbs, prbs, startPrb, nPrb, portMask, 
                                0, mmimo_enabled, record_active_eaxc_ids,
                                tempPdschSym, tempPdschNum, pm_map, fh_context, pdsch_fh_param, 
                                slot_detail, ru, chType, modOrder);
                        }
                    }                      
                        break;

                }
                // NVLOGD_FMT(TAG, "{} hdlIndex {} end", __FUNCTION__, hdlIndex);
        }
        // NVLOGD_FMT(TAG, "{} end", __FUNCTION__, raTypeSize, +grp.resourceAlloc);

    }   

    inline void handleNonNewGroup(uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType, uint8_t qamModOrder) {
        auto& grp = pdsch_fh_param.grp();
        auto ue_grp_index = pdsch_fh_param.ue_grp_index();
        auto& ue = pdsch_fh_param.ue();
        auto& pmi_bf_pdu = pdsch_fh_param.pc_bf();
        auto pm_enabled = pdsch_fh_param.pm_enabled();
        auto mmimo_enabled = pdsch_fh_param.mmimo_enabled();
        auto cell_index = pdsch_fh_param.cell_index();
        auto sym_prbs{pdsch_fh_param.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};

        auto dl_comp_method = get_comp_method(pdsch_fh_param.cell_index());
        auto& idxlist{sym_prbs->symbols[grp.pdschStartSym][channel_type::PDSCH]};
        uint32_t raTypeSize = 0;
        //NVLOGD_FMT( TAG, "{}:{} tempPdschSym {} tempPdschNum {} chType {} mmimo_enabled {}", __FUNCTION__, __LINE__, +tempPdschSym, +tempPdschNum, +chType, +mmimo_enabled);
        switch (grp.resourceAlloc) {
            case 0:
                raTypeSize = fh_context.pdsch_num_ra_type0_info(ue_grp_index);
                break;
            case 1:
                raTypeSize = 1;
                break;
        }

        for (uint32_t i = 0; i < raTypeSize; i++) {
            uint16_t startPrb = 0, nPrb = 0;
            switch (grp.resourceAlloc) {
                case 0:
                    startPrb = fh_context.pdsch_ra_type0_info(i, ue_grp_index).start_prb + grp.startPrb;
                    nPrb =  fh_context.pdsch_ra_type0_info(i, ue_grp_index).num_prb;
                    break;
                case 1:
                    startPrb = grp.startPrb;
                    nPrb = grp.nPrb;
                    break;
            }
           
           auto func = [&pm_map, pm_enabled, cell_index, mmimo_enabled](prb_info_t & prb, scf_fapi_tx_precoding_beamforming_t& pmi_bf_pdu, uint16_t dmrsPortBmsk, cuphyPdschUePrm_t& ue) {
            if (likely(!pm_enabled)) {
                    prb.common.portMask |= calculate_dmrs_port_mask(dmrsPortBmsk, ue.scid, ue.nlAbove16);
                    if(mmimo_enabled)
                    {
                        //NVLOGD_FMT(TAG, "{}:{} not pm_enabled prb_info.common.portMask {}, ue.dmrsPortBmsk {}, ue.scid {}", __FILE__, __LINE__, static_cast<uint32_t>(prb.common.portMask), static_cast<uint64_t>(dmrsPortBmsk), ue.scid);
                        track_eaxcids_fh(ue, prb.common);
                    }
                } else {
                    update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb.common, cell_index, mmimo_enabled);
                }
                NVLOGD_FMT(TAG, "PDSCH old Grp dig_bf_interfaces={}, prb_info.common.portMask {}",
                    pmi_bf_pdu.dig_bf_interfaces, static_cast<uint32_t>(prb.common.portMask));
           };
            auto iter = std::find_if(idxlist.begin(), idxlist.end(),[&sym_prbs, &prbs, startPrb, nPrb](const auto& e){
                bool retval = (e < sym_prbs->prbs_size);
                if (retval) {
                    auto& prb{prbs[e]};
                    retval = (prb.common.startPrbc == startPrb && prb.common.numPrbc == nPrb);
                }
                return retval;
            });

        
            if (iter != idxlist.end()){
                auto& prb{prbs[*iter]};
                switch (chType) {
                    case PDSCH_DMRS: {
                        if (dl_comp_method == comp_method::MODULATION_COMPRESSION) {
                            auto mask1 = ue.dmrsPortBmsk;
                            auto common = mask1 & DEFAULT_DMRS_OFFSET_MASK;
                            auto non_common = mask1 & ~(common);
                            func(prb, pmi_bf_pdu, DEFAULT_DMRS_MASK_1, ue);
                            if (!!non_common) {
                                iter++;
                                if (iter != idxlist.end()) {
                                    auto& prb2{prbs[*iter]};
                                    func(prb2, pmi_bf_pdu, DEFAULT_DMRS_MASK_2, ue);
                                }
                            }
                        } else {
                            func(prb, pmi_bf_pdu, ue.dmrsPortBmsk, ue);
                        }
                    }
                    break;
                    case PDSCH:
                    default:
                        func(prb, pmi_bf_pdu, ue.dmrsPortBmsk, ue);
                        break;

                }
            }
        }
    }

    inline void handleBfEnabled(uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType, uint8_t qamModOrder) {
        bool mmimo_enabled = pdsch_fh_param.mmimo_enabled();
        auto bfwCoeff_mem_info = pdsch_fh_param.bfw_coeff_mem_info();
        auto& pmi_bf_pdu = pdsch_fh_param.pc_bf();
        auto cell_index = pdsch_fh_param.cell_index();
        auto ue_grp_bfw_index_per_cell = pdsch_fh_param.ue_grp_bfw_index_per_cell();
        auto& grp = pdsch_fh_param.grp();
        auto& slot_3gpp = pdsch_fh_param.slot_3gpp();
        auto sym_prbs{pdsch_fh_param.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};
        std::size_t index{sym_prbs->prbs_size - 1};
        prb_info_t& prb_info{prbs[index]};
       // NVLOGD_FMT( TAG, "{} tempPdschSym {} tempPdschNum {} chType {} \n", __FUNCTION__, +tempPdschSym, +tempPdschNum, +chType);

        update_beam_list(prb_info.beams_array, prb_info.beams_array_size, pmi_bf_pdu, mmimo_enabled, prb_info, cell_index);

        if (!grp.resourceAlloc) {
            return;
        }

        if (bfwCoeff_mem_info != NULL)
        {
            NVLOGD_FMT(TAG, "Header {} SFN {} SLOT {} BFW_SFN {} BFW_SLOT {}", *bfwCoeff_mem_info->header, slot_3gpp.sfn_, slot_3gpp.slot_, bfwCoeff_mem_info->sfn, bfwCoeff_mem_info->slot);
        }
        else
        {
            NVLOGD_FMT(TAG, "bfwCoeff_mem_info is NULL");
        }
        if (bfwCoeff_mem_info != NULL && *bfwCoeff_mem_info->header == BFW_COFF_MEM_BUSY)
        {
            if ((is_latest_bfw_coff_avail(slot_3gpp.sfn_ , slot_3gpp.slot_,
                    bfwCoeff_mem_info->sfn, bfwCoeff_mem_info->slot)) && (pmi_bf_pdu.dig_bf_interfaces==0))
            {
                NVLOGD_FMT(TAG, "PDSCH new Grp :: Header {} SFN {} SLOT {} BFW_SFN {} BFW_SLOT {}",
                                *bfwCoeff_mem_info->header, slot_3gpp.sfn_, slot_3gpp.slot_,
                                bfwCoeff_mem_info->sfn, bfwCoeff_mem_info->slot);
                prb_info.common.extType = ORAN_CMSG_SECTION_EXT_TYPE_11;
                //prb_info.common.portMask = (1 << pmi_bf_pdu.dig_bf_interfaces) -1;
                prb_info.bfwCoeff_buf_info.num_prgs = pmi_bf_pdu.num_prgs;
                prb_info.bfwCoeff_buf_info.prg_size = pmi_bf_pdu.prg_size;
                prb_info.bfwCoeff_buf_info.dig_bf_interfaces = pmi_bf_pdu.dig_bf_interfaces;
                prb_info.bfwCoeff_buf_info.nGnbAnt = bfwCoeff_mem_info->nGnbAnt;
                prb_info.bfwCoeff_buf_info.header = bfwCoeff_mem_info->header;
                prb_info.bfwCoeff_buf_info.p_buf_bfwCoef_h = bfwCoeff_mem_info->buff_addr_chunk_h[ue_grp_bfw_index_per_cell];
                prb_info.bfwCoeff_buf_info.p_buf_bfwCoef_d = bfwCoeff_mem_info->buff_addr_chunk_d[ue_grp_bfw_index_per_cell];
                NVLOGD_FMT(TAG, "PDSCH new Grp dig_bf_interfaces={}, prb_info.common.portMask {}, num_prgs={}, prg_size={}, buff_addr_chunk_h[{}]={}",
                                pmi_bf_pdu.dig_bf_interfaces, static_cast<uint32_t>(prb_info.common.portMask), 
                                static_cast<uint16_t>(pmi_bf_pdu.num_prgs), static_cast<uint16_t>(pmi_bf_pdu.prg_size), ue_grp_bfw_index_per_cell, reinterpret_cast<void*>(bfwCoeff_mem_info->buff_addr_chunk_h[ue_grp_bfw_index_per_cell]));

                #if 0
                // 35072
                uint8_t* buffer_ptr = (uint8_t*)bfwCoeff_mem_info->buff_addr_chunk_h[ue_grp_index];
                float2* value = (float2*)buffer_ptr;
                uint32_t count = 0;
                for (uint8_t i=0; i < 1 ; i++)
                {
                    for ( uint8_t j = 0; j < 1; j++ )
                    {
                        for ( uint8_t k = 0; k < 1; k++ )
                        {
                            NVLOGD_FMT(TAG, "####Port {} GnBlayer {} PRBG {} IQ = {} + j{} ", i, j, k, value[count].x, value[count].y);
                            count++;
                        }
                    }
                }
                #endif
            }
        }
        if (prb_info.common.extType == ORAN_CMSG_SECTION_EXT_TYPE_11) {
            prb_info.common.numSymbols = tempPdschNum;
            // prb_info.common.numSymbols = 1;
        }
    }

    inline void handlePmEnabled(uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType, uint8_t qamModOrder) {
        auto& ue = pdsch_fh_param.ue();
        auto& pmi_bf_pdu = pdsch_fh_param.pc_bf();
        auto cell_index = pdsch_fh_param.cell_index();
        auto sym_prbs{pdsch_fh_param.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};
        std::size_t index{sym_prbs->prbs_size - 1};
        prb_info_t& prb_info{prbs[index]};
       // NVLOGD_FMT( TAG, "{} tempPdschSym {} tempPdschNum {} chType {} \n", __FUNCTION__, +tempPdschSym, +tempPdschNum, +chType);

        update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb_info.common, cell_index, pdsch_fh_param.mmimo_enabled());
    }

    inline void handlePmDisabled(uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType, uint8_t qamModOrder) {
        auto& ue = pdsch_fh_param.ue();
        auto cell_index = pdsch_fh_param.cell_index();
        auto& grp = pdsch_fh_param.grp();
        auto mmimo_enabled = pdsch_fh_param.mmimo_enabled();
        auto sym_prbs{pdsch_fh_param.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};
        std::size_t index{sym_prbs->prbs_size - 1};
        prb_info_t& prb_info{prbs[index]};
       // NVLOGD_FMT( TAG, "{} tempPdschSym {} tempPdschNum {} chType {} \n", __FUNCTION__, +tempPdschSym, +tempPdschNum, +chType);

        switch (grp.resourceAlloc) {
            case 0:
                // prb_info.common.portMask |= static_cast<uint64_t>(ue.dmrsPortBmsk) << ue.scid * 8;
                updatePdschPortMask(prb_info, calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16));
                if(mmimo_enabled)
                {
                    //NVLOGD_FMT(TAG, "{}:{} ue.rnti {} ue.dmrsPortBmsk {} ue.scid {}", __FILE__, __LINE__, ue.rnti, ue.dmrsPortBmsk, ue.scid);
                    track_eaxcids_fh(ue, prb_info.common);
                }
                break;
            case 1: {
                if(ue.rnti == UINT16_MAX && !pdsch_fh_param.mmimo_enabled())  // For 4T4R O-RUs, SI-RNTI date with a single layer transmission will be replicated for the second antenna
                {
                    // prb_info.common.portMask |= (1 << (2 * ue.nUeLayers)) - 1;
                    updatePdschPortMask(prb_info, (1 << (2 * ue.nUeLayers)) - 1);
                }
                else
                {
                    // prb_info.common.portMask |= static_cast<uint64_t>(ue.dmrsPortBmsk) << ue.scid * 8;
                    updatePdschPortMask(prb_info, calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16));
                    if(mmimo_enabled)
                    {
                        //NVLOGD_FMT(TAG, "{}:{} ue.rnti {} ue.dmrsPortBmsk {} ue.scid {}", __FILE__, __LINE__, ue.rnti, ue.dmrsPortBmsk, ue.scid);
                        track_eaxcids_fh(ue, prb_info.common);
                    }
                }
            }
        }
    }

    inline void handleMimoEnabled(uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType, uint8_t qamModOrder) {
        auto& grp = pdsch_fh_param.grp();
        auto sym_prbs{pdsch_fh_param.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};
        std::size_t index{sym_prbs->prbs_size - 1};
        prb_info_t& prb_info{prbs[index]};
        // NVLOGD_FMT( TAG, "{} tempPdschSym {} tempPdschNum {} chType {} \n", __FUNCTION__, +tempPdschSym, +tempPdschNum, +chType);

        switch (grp.resourceAlloc) {
            case 0:
                prb_info.common.numSymbols = 1;
                break;
            case 1:
                prb_info.common.numSymbols = tempPdschNum;
                // prb_info.common.numSymbols = 1;
            break;
            default:
                break;
        }
    }
    inline void handleModCompEnabled(uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType, uint8_t qamModOrder) {
        auto& ue = pdsch_fh_param.ue();
        auto ue_grp_index = pdsch_fh_param.ue_grp_index();
        const cuphyPdschDmrsPrm_t& ue_dmrs = fh_context.pdsch_ue_dmrs_info(ue_grp_index);
        bool is_data_muxed = ue_dmrs.nDmrsCdmGrpsNoData == 1;
        uint16_t num_dl_prb = pdsch_fh_param.num_dl_prb();


        auto sym_prbs{pdsch_fh_param.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};
        std::size_t index{sym_prbs->prbs_size - 1};
        prb_info_t& prb_info{prbs[index]};
        // auto maxModOrder = getMaxModOrderInUeGroup(pdsch_fh_param);
        // NVLOGD_FMT( TAG, "{} tempPdschSym {} tempPdschNum {} chType {} \n", __FUNCTION__, +tempPdschSym, +tempPdschNum, +chType);

        // NVLOGD_FMT(TAG, "{} prb remask 0x{:x} portmask 0x{:x}", __FUNCTION__, static_cast<uint16_t>(prb_info.common.reMask), static_cast<uint64_t>(prb_info.common.portMask));
        // NVLOGD_FMT(TAG, "{} ue beta_dmrs {}, beta_qam {}, qamOrder {} is_data_muxed {}", __FUNCTION__, ue.beta_dmrs, ue.beta_qam, qamModOrder, is_data_muxed);
        update_mod_comp_info_common_pdsch(prb_info,  getBwScaler(num_dl_prb));

        switch (chType) {

            case channel_type::PDSCH_DMRS:
                // prb_info.common.reMask = get_dmrs_remask(ue.dmrsPortBmsk);
                if (!is_data_muxed) {
                    update_mod_comp_info_section(prb_info, prb_info.common.reMask, ue.beta_dmrs * SQRT_2, CUPHY_QAM_4, DEFAULT_CSF);
                } else {
                    update_mod_comp_info_section(prb_info, prb_info.common.reMask, ue.beta_dmrs, CUPHY_QAM_4, DEFAULT_CSF);
                    update_mod_comp_info_section(prb_info, ~prb_info.common.reMask, ue.beta_qam, qamModOrder, DEFAULT_CSF);
                    copy_prb_beam_list(prb_info);
                }
                break;
            case channel_type::PDSCH:
                update_mod_comp_info_section(prb_info, 0xFFF, ue.beta_qam, qamModOrder, DEFAULT_CSF);
                break;
        }
    }

    void handleNewPdschSegment(uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType) {
        bool is_new_grp = pdsch_fh_param.is_new_grp();
        bool bf_enabled = pdsch_fh_param.bf_enabled();
        bool pm_enabled = pdsch_fh_param.pm_enabled();
        bool mmimo_enabled = pdsch_fh_param.mmimo_enabled();
        uint8_t funcIndex = 0;
        auto dl_comp_method = get_comp_method(pdsch_fh_param.cell_index());
        //NVLOGC_FMT( TAG, "{}:{} is_new_grp {} pm_enabled {} bf_enabled {} mmimo_enabled {} tempPdschSym {} tempPdschNum {} chType {} dl_comp_method {}", 
        //    __FUNCTION__, __LINE__, is_new_grp, pm_enabled, bf_enabled, mmimo_enabled, +tempPdschSym, +tempPdschNum, +chType, static_cast<uint32_t>(dl_comp_method));
        if (is_new_grp) {
            funcHandlers[HandlerOptions::GROUP] = handleNewGroup; 
        } else {
            funcHandlers[HandlerOptions::GROUP] = no_op;
        }

        if (!pm_enabled) {
            switch(chType) {
                case channel_type::PDSCH_DMRS:
                    if (dl_comp_method == comp_method::MODULATION_COMPRESSION) {
                        funcHandlers[HandlerOptions::PRECODING] = no_op;
                    } else {
                        funcHandlers[HandlerOptions::PRECODING] = handlePmDisabled; 
                    }
                    break;
                case channel_type::PDSCH:
                    funcHandlers[HandlerOptions::PRECODING] = handlePmDisabled; 
                    break;

            }
        } else {
            funcHandlers[HandlerOptions::PRECODING] = handlePmEnabled; 
        }
        if (bf_enabled) {
            funcHandlers[HandlerOptions::BEAM] = handleBfEnabled; 
        } else {
            funcHandlers[HandlerOptions::BEAM] = no_op;
        }

        if (mmimo_enabled) {
            funcHandlers[HandlerOptions::MIMO] = handleMimoEnabled; 

        } else {
            funcHandlers[HandlerOptions::MIMO] = no_op;
        }
        switch (dl_comp_method) {
            case comp_method::MODULATION_COMPRESSION:
                funcHandlers[HandlerOptions::COMP] = handleModCompEnabled; 

            break;
            case comp_method::BLOCK_FLOATING_POINT:
            case comp_method::NO_COMPRESSION:
                funcHandlers[HandlerOptions::COMP] = no_op;

            default:
                break;

        }
        funcHandlers[HandlerOptions::GROUP](tempPdschSym, tempPdschNum, pm_map, fh_context, pdsch_fh_param, slot_detail, ru, chType, 0);
    }

    uint16_t getPdschCsirsPrbOverlap(uint16_t csirs_remap, bool use_alt_prb, uint16_t startPdschPrbOffset, uint16_t nPdschPrb, uint16_t* reMap, int32_t cell_idx) {
        // Create a range to search over
        auto start = reMap + startPdschPrbOffset;
        auto end = start + nPdschPrb;
        
        // Define a lambda to check if the current position matches our criteria
        auto matches_criteria = [csirs_remap, use_alt_prb](uint16_t val) {
            return (csirs_remap == (val & 0xFFF)) || 
                   (use_alt_prb && (csirs_remap == ((val + 1) & 0xFFF)));
        };
        
        // Use lower_bound to find the first position that doesn't match our criteria
        // Since the array is sorted, we can use binary search
        auto it = std::lower_bound(start, end, csirs_remap, 
            [&matches_criteria](uint16_t val, uint16_t target) {
                return matches_criteria(val);
            });
        // Return the offset from startPdschPrbOffset
        return std::distance(start, it) + startPdschPrbOffset;
    }

    // uint16_t getPdschCsirsPrbOverlap(uint16_t csirs_remap, bool use_alt_prb, uint16_t startPdschPrbOffset, uint16_t nPdschPrb, uint16_t*reMap, int32_t cell_idx ) {
    //     uint16_t counter = nPdschPrb;
    //     auto rb_idx = startPdschPrbOffset;
    //     while(((csirs_remap == (reMap[rb_idx] & 0xFFF)) ||
    //         (use_alt_prb && (csirs_remap == (reMap[rb_idx + 1] & 0xFFF))))
    //         && (counter > 0)) {
    //             //NVLOGD_FMT(TAG, "csirs_remap={} csirs_info->reMap={}",csirs_remap, (csirs_info->reMap[cell_index][pdschSym*num_dl_prb + rb_idx] & 0xFFF));
    //             // NVLOGD_FMT(TAG, "csirs_remap={} csirs_info->reMap={} rb_idx {} counter {}",csirs_remap, (reMap[rb_idx]& 0xFFF), rb_idx, counter);
    //             rb_idx++; counter--;
    //         continue;
    //     }
    //     return rb_idx;
    // }

    void createNewPdschCsirsPrbs(slot_info_t* sym_prbs, uint16_t startPrb, uint16_t nPrb, bool use_alt_prb, uint16_t csirs_remap, bool isPdschSplitAcrossPrbInfo) {

        auto& prbs{sym_prbs->prbs};
        auto tempnPrb = use_alt_prb? ((nPrb + 1) >> 1) : nPrb;
        createNewPdschPrbs(sym_prbs, startPrb, tempnPrb);
        std::size_t index_pdsch_csirs{sym_prbs->prbs_size - 1};
        prb_info_t& prb_info{prbs[sym_prbs->prbs_size - 1]};
        prb_info.common.reMask = ~csirs_remap & 0xFFF;
        prb_info.common.useAltPrb = use_alt_prb;
        prb_info.common.isPdschSplitAcrossPrbInfo = isPdschSplitAcrossPrbInfo;
    }

    template <bool modcomp_enabled = false>
    inline void handlePdschCsirsNewGroup(uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType, uint8_t num_csirs_eaxcids, uint8_t qamModOrder) {
        auto& grp = pdsch_fh_param.grp();
        const auto& cell_grp = fh_context.pdsch_cell_grp_info();
        auto ue_grp_index = pdsch_fh_param.ue_grp_index();
        auto mmimo_enabled = pdsch_fh_param.mmimo_enabled();
        auto cell_index = pdsch_fh_param.cell_index();
        uint16_t num_dl_prb = pdsch_fh_param.num_dl_prb();
        auto sym_prbs{pdsch_fh_param.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};
        //prbs_info.common.ap_index = 0; //Init AP_INDEX as this is the first UE in the new UE-Group

        NVLOGD_FMT( TAG, "{} tempPdschSym {} tempPdschNum {} chType {} \n", __FUNCTION__, +tempPdschSym, +tempPdschNum, +chType);

        prb_info_idx_list_t & csi_rs_prb_info = sym_prbs->symbols[tempPdschSym][channel_type::CSI_RS];
        auto csi_rs_prb_info_size = csi_rs_prb_info.size();

        std::array<int, MAX_DL_EAXCIDS> record_active_eaxc_ids;
        if(mmimo_enabled)
        {
            record_active_eaxc_ids.fill(-1); // Initialize with -1 to indicate no active eaxc ids
        }
        auto pdschGrpPortMask =  getPdschPortMask(fh_context, pdsch_fh_param, record_active_eaxc_ids);
        auto [csirsFlowMask, csirsPortMask] = getCsirsPortMask(csi_rs_prb_info, prbs, num_csirs_eaxcids);
        
        // Build qamModOrder to port mask mapping for all UEs in the group
        const auto nUes = grp.nUes;
        const auto ueIdxs = grp.pUePrmIdxs;
        
        const auto qamModOrderToPortMask = buildQamModOrderToPortMask(
            cell_grp, nUes, ueIdxs, fh_context, mmimo_enabled);
        
        auto commonPorts = pdschGrpPortMask & csirsFlowMask;
        auto nonCommonPdschGrpPorts = pdschGrpPortMask & ~(commonPorts);
        auto nonCommonCsirsflows = csirsFlowMask & ~(commonPorts);
        auto nonCommonCsirsPorts = generateNonCommonCsirsPorts(csirsPortMask, csirsFlowMask, nonCommonCsirsflows, num_csirs_eaxcids);
        uint32_t raTypeSize = 0;
        switch (grp.resourceAlloc) {
            case 0:
                raTypeSize = fh_context.pdsch_num_ra_type0_info(ue_grp_index);
                break;
            case 1:
                raTypeSize = 1;
                break;
        }
    
        for (uint32_t i = 0; i < raTypeSize; i++) {
            uint16_t startPrb = 0, nPrb = 0;
            switch (grp.resourceAlloc) {
                case 0:
                    startPrb = fh_context.pdsch_ra_type0_info(i, ue_grp_index).start_prb + grp.startPrb;
                    nPrb = fh_context.pdsch_ra_type0_info(i, ue_grp_index).num_prb;
                    break;
                case 1:
                    startPrb = grp.startPrb;
                    nPrb = grp.nPrb;
                    break;
            }
            uint16_t rb_idx = grp.startPrb + tempPdschSym*num_dl_prb;
            uint16_t csirs_remap = fh_context.csirs_remap(cell_index, rb_idx) & 0xFFF;
            bool use_alt_prb = (fh_context.csirs_remap(cell_index, rb_idx) & (1 << 15)) ? true : false;
            uint32_t tempnPrb = 0;
            uint32_t tempStartPrb = rb_idx;
            uint32_t counter = grp.nPrb;
            bool isPdschSplitAcrossPrbInfo = false;

            while (counter) {
                rb_idx = getPdschCsirsPrbOverlap(csirs_remap, use_alt_prb, rb_idx, counter, fh_context.csirs_remap_row(cell_index), cell_index);
                tempnPrb = rb_idx - tempStartPrb;
                counter -= tempnPrb;
                if (counter > 0) {
                    isPdschSplitAcrossPrbInfo = true;
                }
                uint16_t startPrb = tempStartPrb >= tempPdschSym*num_dl_prb? tempStartPrb-tempPdschSym*num_dl_prb : tempStartPrb;
                    NVLOGD_FMT(TAG, "csirs_remap 0x{:x} pdschGrpPortMask {} csirsPortMask {} commonPorts {} nonCommonPdschGrpPorts {} nonCommonCsirsPorts {} counter {}",csirs_remap, pdschGrpPortMask, csirsPortMask, commonPorts, nonCommonPdschGrpPorts, nonCommonCsirsPorts, counter);
                if (!csirs_remap) {
                    // Create PRBs for each unique qamModOrder
                    for (std::size_t idx = 0; idx < qamModOrderToPortMask.size(); ++idx) {
                        const auto portMask = qamModOrderToPortMask[idx];
                        if (portMask == 0) continue;
                        
                        const auto modOrder = idx << 1;  // Reconstruct qamModOrder from array index
                        
                        createNewPdschPrbs(sym_prbs, startPrb, tempnPrb);
                        std::size_t index_pdsch{sym_prbs->prbs_size - 1};
                        prb_info_t& prb_info{prbs[index_pdsch]};
                        prb_info.common.reMask = 0xFFF;
                        funcPdschCsirsHandlers[HandlerOptions::PRECODING](tempPdschSym, tempPdschNum, pm_map, fh_context, pdsch_fh_param, slot_detail, ru, channel_type::PDSCH, 0, modOrder);
                        updatePdschPortMask(prb_info, portMask);
                        if(mmimo_enabled)
                        {
                            //NVLOGD_FMT(TAG, "{}:{} update_eaxcids_from_portmask => prb_info for modOrder {}",__FUNCTION__, __LINE__, modOrder);
                            update_eaxcids_from_portmask(prb_info, portMask, record_active_eaxc_ids);
                        }
                        funcPdschCsirsHandlers[HandlerOptions::BEAM](tempPdschSym, tempPdschNum, pm_map, fh_context, pdsch_fh_param, slot_detail, ru, channel_type::PDSCH, 0, modOrder);
                        funcPdschCsirsHandlers[HandlerOptions::COMP](tempPdschSym, tempPdschNum, pm_map, fh_context, pdsch_fh_param, slot_detail, ru, channel_type::PDSCH, 0, modOrder);
                        update_prb_sym_list(*sym_prbs, index_pdsch, tempPdschSym, 1, channel_type::PDSCH, ru);
                    }
                } else {
                        // Handle common ports (PDSCH+CSI-RS overlap) per qamModOrder
                        for (std::size_t idx = 0; idx < qamModOrderToPortMask.size(); ++idx) {
                            const auto portMask = qamModOrderToPortMask[idx];
                            if (portMask == 0) continue;
                            
                            const auto modOrder = idx << 1;  // Reconstruct qamModOrder from array index
                            const auto commonPortsForModOrder = portMask & csirsFlowMask;
                            if (commonPortsForModOrder) {
                                /// PDSCH+CSIRS there 
                                uint64_t mask = commonPortsForModOrder;
                                std::size_t last_index_pdsch_csirs{};
                                while (mask) {
                                    uint8_t bit_index = __builtin_ctzll(mask); // Get index of least significant set bit
                                    auto portId = 1ULL << bit_index;
                                    createNewPdschCsirsPrbs(sym_prbs, startPrb, tempnPrb, use_alt_prb, csirs_remap, isPdschSplitAcrossPrbInfo);
                                    funcPdschCsirsHandlers[HandlerOptions::BEAM](tempPdschSym, tempPdschNum, pm_map, fh_context, pdsch_fh_param, slot_detail, ru, chType, 0, modOrder);
                                    last_index_pdsch_csirs = sym_prbs->prbs_size - 1;
                                    prb_info_t& prb_info{prbs[last_index_pdsch_csirs]};
                                    updatePdschPortMask(prb_info, portId);
                                    if(mmimo_enabled)
                                    {
                                        //NVLOGD_FMT(TAG, "{}:{} update_eaxcids_from_portmask => prb_info for modOrder {}",__FUNCTION__, __LINE__, modOrder);
                                        update_eaxcids_from_portmask(prb_info, portId, record_active_eaxc_ids);
                                    }
                                    funcPdschCsirsHandlers[HandlerOptions::COMP](tempPdschSym, tempPdschNum, pm_map, fh_context, pdsch_fh_param, slot_detail, ru, chType, num_csirs_eaxcids, modOrder);
                                    update_prb_sym_list(*sym_prbs, last_index_pdsch_csirs, tempPdschSym, 1, channel_type::PDSCH_CSIRS, ru);
                                    mask &= (mask - 1); // Clear least significant set bit
                                }

                                if(use_alt_prb && tempnPrb > 2) {
                                    createNewPdschPrbs(sym_prbs, startPrb + 1, (tempnPrb >> 1));
                                    std::size_t index_pdsch{sym_prbs->prbs_size - 1};
                                    prb_info_t& prb_info{prbs[index_pdsch]};
                                    prb_info_t& prb_info_ref{prbs[last_index_pdsch_csirs]};
                                    prb_info.common.direction = prb_info_ref.common.direction;
                                    prb_info.common.numApIndices = prb_info_ref.common.numApIndices;
                                    prb_info.common.useAltPrb = prb_info_ref.common.useAltPrb;
                                    prb_info.common.portMask = prb_info_ref.common.portMask;
                                    prb_info.beams_array = prb_info_ref.beams_array;
                                    prb_info.beams_array_size = prb_info_ref.beams_array_size;
                                    prb_info.common.reMask = 0xFFF;
                                    funcPdschCsirsHandlers[HandlerOptions::COMP](tempPdschSym, tempPdschNum, pm_map, fh_context, pdsch_fh_param, slot_detail, ru, channel_type::PDSCH, 0, modOrder);
                                    update_prb_sym_list(*sym_prbs, index_pdsch, tempPdschSym, 1, channel_type::PDSCH, ru);
                                    NVLOGD_FMT(TAG, "PDSCH with CSIRS new Grp but PDSCH only: at symbol={} added index={} startPrb={} numPrb={} nSym=1 modOrder={}",
                                        tempPdschSym, index_pdsch, startPrb+1, tempnPrb>>1, modOrder);
                                }
                            }
                        }

                        // Create PRBs for non-common PDSCH ports per qamModOrder
                        for (std::size_t idx = 0; idx < qamModOrderToPortMask.size(); ++idx) {
                            const auto portMask = qamModOrderToPortMask[idx];
                            if (portMask == 0) continue;
                            
                            const auto modOrder = idx << 1;  // Reconstruct qamModOrder from array index
                            const auto nonCommonPortsForModOrder = portMask & ~(portMask & csirsFlowMask);
                            if (nonCommonPortsForModOrder) {
                                createNewPdschPrbs(sym_prbs, startPrb, tempnPrb);
                                std::size_t index_pdsch{sym_prbs->prbs_size - 1};
                                prb_info_t& prb_info{prbs[index_pdsch]};
                                prb_info.common.reMask = ~csirs_remap & 0xFFF;
                                updatePdschPortMask(prb_info, nonCommonPortsForModOrder);
                                if(mmimo_enabled)
                                {
                                    //NVLOGD_FMT(TAG, "{}:{} update_eaxcids_from_portmask => prb_info for modOrder {}",__FUNCTION__, __LINE__, modOrder);
                                    update_eaxcids_from_portmask(prb_info, nonCommonPortsForModOrder, record_active_eaxc_ids);
                                }
                                funcPdschCsirsHandlers[HandlerOptions::BEAM](tempPdschSym, tempPdschNum, pm_map, fh_context, pdsch_fh_param, slot_detail, ru, chType, 0, modOrder);
                                funcPdschCsirsHandlers[HandlerOptions::COMP](tempPdschSym, tempPdschNum, pm_map, fh_context, pdsch_fh_param, slot_detail, ru, channel_type::PDSCH, 0, modOrder);
                                NVLOGD_FMT(TAG, "nonCommonPdschGrpPorts modOrder {} portMask {} pdschPortMask {} reMask {}", modOrder, +prb_info.common.portMask, +prb_info.common.pdschPortMask, +prb_info.common.reMask);
                                update_prb_sym_list(*sym_prbs, index_pdsch, tempPdschSym, 1, channel_type::PDSCH, ru);
                            }
                        }

                        if (nonCommonCsirsPorts) {
                            uint64_t mask = nonCommonCsirsPorts;
                            while (mask) {
                                uint8_t bit_index = __builtin_ctzll(mask); // Get index of least significant set bit
                                auto csirsPort = 1ULL << bit_index;
                                auto iter = std::find_if(csi_rs_prb_info.begin(), csi_rs_prb_info.end(),[&csirsPort, &prbs](const auto& e){ 
                                    return (prbs[e].common.portMask == csirsPort);
                                });
                                if (iter != csi_rs_prb_info.end()) {
                                    auto& csirs_prb = prbs[*iter];
                                    NVLOGD_FMT(TAG, "nonCommonCsirsPorts portMask {} pdschPortMask {} reMask {} ", +csirs_prb.common.portMask, +csirs_prb.common.pdschPortMask, +csirs_prb.common.reMask);
                                    addDisjointCsirsPrbs<modcomp_enabled>(sym_prbs, csirs_prb, startPrb, tempnPrb, tempPdschSym, num_dl_prb);
                                }
                                mask &= (~csirsPort); // Clear least significant set bit
                            }
                        }
                }
                tempStartPrb = rb_idx;
                tempnPrb = 0;
                csirs_remap = fh_context.csirs_remap(cell_index, rb_idx) & 0xFFF;
                use_alt_prb = (fh_context.csirs_remap(cell_index, rb_idx) & (1 << 15)) ? true : false;
            }
        }
    }

    inline void handlePdschCsirsNonNewGroup(uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType, uint8_t num_csirs_eaxcids, uint8_t qamModOrder) {
        auto& grp = pdsch_fh_param.grp();
        auto ue_grp_index = pdsch_fh_param.ue_grp_index();
        auto& ue = pdsch_fh_param.ue();
        auto& pmi_bf_pdu = pdsch_fh_param.pc_bf();
        auto cell_index = pdsch_fh_param.cell_index();
        uint16_t num_dl_prb = pdsch_fh_param.num_dl_prb();
        auto sym_prbs{pdsch_fh_param.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};

        NVLOGD_FMT( TAG, "{} tempPdschSym {} tempPdschNum {} chType {} qamModOrder {} \n", __FUNCTION__, +tempPdschSym, +tempPdschNum, +chType, +qamModOrder);

        uint32_t raTypeSize = 0;
        switch (grp.resourceAlloc) {
            case 0:
                raTypeSize = fh_context.pdsch_num_ra_type0_info(ue_grp_index);
                break;
            case 1:
                raTypeSize = 1;
                break;
        }
    
        for (uint32_t i = 0; i < raTypeSize; i++) {
            uint16_t startPrb = 0, nPrb = 0;
            switch (grp.resourceAlloc) {
                case 0:
                    startPrb = fh_context.pdsch_ra_type0_info(i, ue_grp_index).start_prb + grp.startPrb;
                    nPrb =  fh_context.pdsch_ra_type0_info(i, ue_grp_index).num_prb;
                    break;
                case 1:
                    startPrb = grp.startPrb;
                    nPrb = grp.nPrb;
                    break;
            }

            prb_info_idx_list_t & csi_rs_prb_info = sym_prbs->symbols[tempPdschSym][channel_type::CSI_RS];
            auto csi_rs_prb_info_size = csi_rs_prb_info.size();
            
            uint16_t rb_idx = grp.startPrb;
            uint16_t csirs_remap = fh_context.csirs_remap(cell_index, tempPdschSym * num_dl_prb + rb_idx) & 0xFFF;
            bool use_alt_prb = (fh_context.csirs_remap(cell_index, tempPdschSym * num_dl_prb + rb_idx) & (1 << 15)) ? true : false;
            uint32_t tempnPrb = 0;
            uint32_t tempStartPrb = rb_idx;
            uint32_t counter = grp.nPrb;
            bool isPdschSplitAcrossPrbInfo = false;
            auto& idxlist{sym_prbs->symbols[tempPdschSym][channel_type::PDSCH]};

            prb_info_idx_list_t::iterator iter = idxlist.end();

            while (counter) {
                rb_idx = getPdschCsirsPrbOverlap(csirs_remap, use_alt_prb, tempPdschSym * num_dl_prb + rb_idx, counter, fh_context.csirs_remap_row(cell_index), cell_index);
                tempnPrb = rb_idx - tempPdschSym*num_dl_prb - tempStartPrb;
                counter -= tempnPrb;
                switch (grp.resourceAlloc) {
                    case 0: {
                        iter = std::find_if(idxlist.begin(), idxlist.end(),[&sym_prbs, &prbs, startPrb, nPrb](const auto& e){
                            bool retval = (e < sym_prbs->prbs_size);
                            if (retval) {
                                auto& prb{prbs[e]};
                                retval = (prb.common.startPrbc == startPrb && prb.common.numPrbc == nPrb);
                            }
                            return retval;
                        });
                    }
                    break;
                    case 1: {
                            iter = std::find_if(idxlist.begin(), idxlist.end(),[&sym_prbs, &prbs, &grp, &tempStartPrb, &tempnPrb](const auto& e){
                            bool retval = (e < sym_prbs->prbs_size);
                            if (retval) {
                                auto& prb{prbs[e]};
                                if(prb.common.useAltPrb)
                                {
                                    retval = (prb.common.startPrbc == grp.startPrb && prb.common.numPrbc == ((grp.nPrb + 1) >> 2 ));
                                }
                                else
                                {
                                    retval = (prb.common.startPrbc == tempStartPrb && prb.common.numPrbc == tempnPrb);
                                }
                            }
                            return retval;
                        });
                    }
                    break;
                }

                tempStartPrb = rb_idx;
                tempnPrb = 0;
                if (iter != idxlist.end()){
                    auto& prb{prbs[*iter]};
                        if (likely(!pdsch_fh_param.pm_enabled())) {
                        updatePdschPortMask(prb,calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16));
                    } else {
                        update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb.common, cell_index, pdsch_fh_param.mmimo_enabled());
                    }
                }
            }

        }
    }

    inline void handlePdschCsirsPmDisabled(uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType, uint8_t num_csirs_eaxcids, uint8_t qamModOrder) {
        auto sym_prbs{pdsch_fh_param.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};
        handlePmDisabled(tempPdschSym, tempPdschNum, pm_map, fh_context, pdsch_fh_param, slot_detail, ru, chType, 0);
        std::size_t index_pdsch_csirs{sym_prbs->prbs_size - 1};
        prb_info_t& prb_info{prbs[sym_prbs->prbs_size - 1]};
        prb_info.common.pdschPortMask = prb_info.common.portMask;
    }

    inline void handlePdschCsirsPmEnabled(uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType, uint8_t num_csirs_eaxcids, uint8_t qamModOrder) {
        auto sym_prbs{pdsch_fh_param.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};
        handlePmEnabled(tempPdschSym, tempPdschNum, pm_map, fh_context, pdsch_fh_param, slot_detail, ru, chType, 0);
        std::size_t index_pdsch_csirs{sym_prbs->prbs_size - 1};
        prb_info_t& prb_info{prbs[sym_prbs->prbs_size - 1]};
        prb_info.common.pdschPortMask = prb_info.common.portMask;
    }

    inline void handlePdschCsirsBfEnabled(uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType, uint8_t num_csirs_eaxcids, uint8_t qamModOrder) {
        auto sym_prbs{pdsch_fh_param.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};
        handleBfEnabled(tempPdschSym, tempPdschNum, pm_map, fh_context, pdsch_fh_param, slot_detail, ru, chType, 0);
        std::size_t index_pdsch_csirs{sym_prbs->prbs_size - 1};
        prb_info_t& prb_info{prbs[sym_prbs->prbs_size - 1]};
        prb_info.common.pdschPortMask = prb_info.common.portMask;
    }

    inline void handlePdschCsirsMimoEnabled(uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType, uint8_t num_csirs_eaxcids, uint8_t qamModOrder) {
    
    }

    template <bool modcomp_enabled>
    void addDisjointCsirsPrbs(slot_info_t* sym_prbs, prb_info_t& csirs_prb_info, uint16_t overlap_start_prb, uint16_t overlap_num_prb, uint16_t symbol_id , uint16_t num_dl_prb) {
        auto& alt_csirs_prb_info_list{sym_prbs->alt_csirs_prb_info_list};
        auto& alt_csirs_prb_info_idx_list{sym_prbs->alt_csirs_prb_info_idx_list};
        auto& prbs{sym_prbs->prbs};

        alt_csirs_prb_info_list[sym_prbs->alt_csirs_prb_info_list_size] = prb_info_t(overlap_start_prb, overlap_num_prb);
        auto &prb_info(alt_csirs_prb_info_list[sym_prbs->alt_csirs_prb_info_list_size]);
        prb_info.common.direction = csirs_prb_info.common.direction;
        prb_info.common.numApIndices = csirs_prb_info.common.numApIndices;
        prb_info.common.portMask = csirs_prb_info.common.portMask;
        prb_info.common.useAltPrb = csirs_prb_info.common.useAltPrb;
        prb_info.beams_array = csirs_prb_info.beams_array;
        prb_info.beams_array_size = csirs_prb_info.beams_array_size;
        prb_info.common.reMask = csirs_prb_info.common.reMask;
        if(modcomp_enabled) {
            update_mod_comp_info_common(prb_info, getBwScaler(num_dl_prb));
            prb_info.common.reMask = csirs_prb_info.common.reMask; // This is done to ensure that the reMask is not modified by the modcomp compression 
            // as update_mod_comp_info_common function sets the reMask =0..
            NVLOGD_FMT(TAG, "addDisjointCsirsPrbs symbol {} portMask {} reMask {}", symbol_id, +prb_info.common.portMask, +prb_info.common.reMask);
            prb_info.comp_info.common = csirs_prb_info.comp_info.common; 
            NVLOGD_FMT(TAG, "addDisjointCsirsPrbs nSections {} udIqWidth {}", +prb_info.comp_info.common.nSections, +prb_info.comp_info.common.udIqWidth);
            for (uint8_t i = 0; i < prb_info.comp_info.common.nSections; i++) {
                auto& section = prb_info.comp_info.sections[i];
                auto& csirs_section = csirs_prb_info.comp_info.sections[i];
                section = csirs_section; 
                prb_info.comp_info.modCompScalingValue[i] = csirs_prb_info.comp_info.modCompScalingValue[i];
            }
        }
        alt_csirs_prb_info_idx_list[symbol_id].emplace_back(sym_prbs->alt_csirs_prb_info_list_size++);
    }

    inline void handlePdschCsirsModCompEnabled(uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType, uint8_t num_csirs_eaxcids, uint8_t qamModOrder) {

        auto& ue = pdsch_fh_param.ue();
        auto ue_grp_index = pdsch_fh_param.ue_grp_index();
        auto& grp = pdsch_fh_param.grp();
        auto sym_prbs{pdsch_fh_param.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};
        std::size_t index{sym_prbs->prbs_size - 1};
        prb_info_t& prb_info{prbs[index]};
        overlap_csirs_port_info_t& overlap_csirs_port_info = sym_prbs->overlap_csirs_port_info[index];
        prb_info_idx_list_t & csi_rs_prb_info = sym_prbs->symbols[tempPdschSym][channel_type::CSI_RS];
        auto csi_rs_prb_info_size = csi_rs_prb_info.size();
        auto cell_index = pdsch_fh_param.cell_index();
        uint16_t num_dl_prb = pdsch_fh_param.num_dl_prb();
        uint16_t rb_idx = grp.startPrb;
        uint16_t csirs_remap = fh_context.csirs_remap(cell_index, tempPdschSym * num_dl_prb + rb_idx) & 0xFFF;
        // auto maxModOrder = getMaxModOrderInUeGroup(pdsch_fh_param);

        update_mod_comp_info_common_pdsch(prb_info, getBwScaler(num_dl_prb));
        switch (chType) {
            case PDSCH:
                update_mod_comp_info_section(prb_info, prb_info.common.reMask,  ue.beta_qam, qamModOrder, DEFAULT_CSF);
                break;
            case PDSCH_CSIRS: {
                auto common_port_mask = prb_info.common.pdschPortMask;
                NVLOGD_FMT(TAG, "symbol {} common_port_mask 0x{:x} reMask 0x{:x}", tempPdschSym, common_port_mask, +prb_info.common.reMask);
                update_mod_comp_info_section(prb_info, prb_info.common.reMask, ue.beta_qam, qamModOrder, DEFAULT_CSF);

                auto iter = std::find_if(csi_rs_prb_info.begin(), csi_rs_prb_info.end(),[&common_port_mask, &prbs, num_csirs_eaxcids](const auto& e){ 
                        uint64_t portMask = prbs[e].common.portMask;
                        if (__builtin_clzll(portMask) < (64 - num_csirs_eaxcids)) {
                            uint64_t newMask = 0;
                            uint64_t tempMask = portMask;
                            while (tempMask) {
                                uint8_t bit_index = __builtin_ctzll(tempMask);
                                uint8_t new_bit_index = bit_index % num_csirs_eaxcids;
                                newMask |= (1ULL << new_bit_index);
                                tempMask &= (tempMask - 1);
                            }
                            portMask = newMask;
                        }
                            return ((portMask & common_port_mask) != 0);
                                });
                if (iter != csi_rs_prb_info.end()) {
                    
                    auto& csirs_prb = prbs[*iter];
                    auto next_iter = iter + num_csirs_eaxcids;
                    auto& csirs_section = csirs_prb.comp_info.sections[csirs_prb.comp_info.common.nSections - 1]; 
                    float beta = csirs_prb.comp_info.modCompScalingValue[csirs_prb.comp_info.common.nSections - 1];
                    uint16_t csirs_reMask = csirs_section.mcScaleReMask;
                    overlap_csirs_port_info.reMask_ap_idx_pairs[overlap_csirs_port_info.num_overlap_ports].first = csirs_reMask;
                    overlap_csirs_port_info.reMask_ap_idx_pairs[overlap_csirs_port_info.num_overlap_ports].second = csirs_prb.common.ap_index;
                    overlap_csirs_port_info.num_overlap_ports++;
                    //Use the num_ports from the index at which CSI-RS with matching port mask is found.
                    //We had saved the num_ports in the overlap_csirs_port_info when the CSI-RS was created
                    //for this purpose. As we want to know what is the total number of ports assigned by CSI-RS
                    //configuration. We need this information for correct beam ID assignment to overlapping CSI-RS sections.
                    overlap_csirs_port_info.num_ports = sym_prbs->overlap_csirs_port_info[*iter].num_ports;
                    //Accumulate reMask for all ports of csirs mapped to same pdsch port.
                    while(next_iter < csi_rs_prb_info.end()){
                    
                        auto& next_csirs_prb = prbs[*next_iter];
                        uint8_t bit_index = __builtin_ctzll(next_csirs_prb.common.portMask);
                        uint8_t new_bit_index = bit_index % num_csirs_eaxcids;
                        uint64_t newMask = (1ULL << new_bit_index);
                        if(!(newMask & common_port_mask)) {
                            break;
                        }
                        auto& csirs_prb = prbs[*next_iter];
                        csirs_reMask |= csirs_prb.common.reMask;
                        overlap_csirs_port_info.reMask_ap_idx_pairs[overlap_csirs_port_info.num_overlap_ports].first = csirs_prb.common.reMask;
                        overlap_csirs_port_info.reMask_ap_idx_pairs[overlap_csirs_port_info.num_overlap_ports].second = csirs_prb.common.ap_index;
                        overlap_csirs_port_info.num_overlap_ports++;
                        next_iter+= num_csirs_eaxcids;
                    }
                    NVLOGD_FMT(TAG, "csirs_section.mcScaleReMask.get() {} beta {} reMask {} portMask {}", csirs_reMask, beta, +csirs_prb.common.reMask, +csirs_prb.common.portMask);
                    update_mod_comp_info_section(prb_info, csirs_reMask, beta, CUPHY_QAM_4, csirs_section.csf);
                    // We are assuming that all other values like beam ids modcomp related values are same for different ports of csirs.
                    std::copy(csirs_prb.beams_array.begin(), csirs_prb.beams_array.begin() + csirs_prb.beams_array_size, prb_info.beams_array2.begin());
                    prb_info.beams_array_size2 = csirs_prb.beams_array_size;       
                }
            }
            break;
            default:
                break;
        }

    }

    void handleNewPdschCsirsSegment(uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType, uint8_t num_csirs_eaxcids, bool csirs_compact_mode) {
        NVLOGD_FMT( TAG, "{} tempPdschSym {} tempPdschNum {} chType {} \n", __FUNCTION__, +tempPdschSym, +tempPdschNum, +chType);

        bool is_new_grp = pdsch_fh_param.is_new_grp();
        bool bf_enabled = pdsch_fh_param.bf_enabled();
        bool pm_enabled = pdsch_fh_param.pm_enabled();
        bool mmimo_enabled = pdsch_fh_param.mmimo_enabled();
        uint8_t funcIndex = 0;
        auto dl_comp_method = get_comp_method(pdsch_fh_param.cell_index());
        if(csirs_compact_mode) {
            dl_comp_method = comp_method::MODULATION_COMPRESSION;
        }
        if (is_new_grp) {
            if(dl_comp_method == comp_method::MODULATION_COMPRESSION) {
                funcPdschCsirsHandlers[HandlerOptions::GROUP] = handlePdschCsirsNewGroup<true>; 
            } else {
                funcPdschCsirsHandlers[HandlerOptions::GROUP] = handlePdschCsirsNewGroup<false>; 
            }
        } else {
            funcPdschCsirsHandlers[HandlerOptions::GROUP] = no_op_pdsch_csirs;
        }

        if (!pm_enabled) {
            funcPdschCsirsHandlers[HandlerOptions::PRECODING] = handlePdschCsirsPmDisabled; 
        } else {
            funcPdschCsirsHandlers[HandlerOptions::PRECODING] = handlePdschCsirsPmEnabled; 
        }

        if (bf_enabled) {
            funcPdschCsirsHandlers[HandlerOptions::BEAM] = handlePdschCsirsBfEnabled; 
        } else {
            funcPdschCsirsHandlers[HandlerOptions::BEAM] = no_op_pdsch_csirs;
        }

        if (mmimo_enabled) {
            funcPdschCsirsHandlers[HandlerOptions::MIMO] = handlePdschCsirsMimoEnabled; 
        } else {
            funcPdschCsirsHandlers[HandlerOptions::MIMO] = no_op_pdsch_csirs;
        }

        switch (dl_comp_method) {
            case comp_method::MODULATION_COMPRESSION:
                funcPdschCsirsHandlers[HandlerOptions::COMP] = handlePdschCsirsModCompEnabled; 
            break;
            case comp_method::BLOCK_FLOATING_POINT:
            case comp_method::NO_COMPRESSION:
                funcPdschCsirsHandlers[HandlerOptions::COMP] = no_op_pdsch_csirs;

            default:
                break;

        }
        funcPdschCsirsHandlers[HandlerOptions::GROUP](tempPdschSym, tempPdschNum, pm_map, fh_context, pdsch_fh_param, slot_detail, ru, chType, num_csirs_eaxcids, 0);
    }

}
