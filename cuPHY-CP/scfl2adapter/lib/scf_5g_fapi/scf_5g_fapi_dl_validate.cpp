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

#include "scf_5g_fapi_dl_validate.hpp"
#include <aerial/casts/casts.hpp>
#include <climits>

#define TAG (NVLOG_TAG_BASE_SCF_L2_ADAPTER + 10) // "SCF.DL_FAPI_VALIDATE"

static int validate_csirs_pdu(scf_fapi_csi_rsi_pdu_t csi_rs_pdu)
{
    if((csi_rs_pdu.bwp.bwp_size < 1) || (csi_rs_pdu.bwp.bwp_size > MAX_N_PRBS_SUPPORTED))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "CSI-RS PDU BWP size {} out of range, it can be from 1 to {}", 
        static_cast<uint16_t>(csi_rs_pdu.bwp.bwp_size), static_cast<uint16_t>(MAX_N_PRBS_SUPPORTED));
        return INVALID_FAPI_PDU;
    }

    if(csi_rs_pdu.bwp.bwp_start > (MAX_N_PRBS_SUPPORTED-1))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "CSI-RS PDU BWP start index {} out of range, it should be less than {}", 
        static_cast<uint16_t>(csi_rs_pdu.bwp.bwp_start), static_cast<uint16_t>(MAX_N_PRBS_SUPPORTED));
        return INVALID_FAPI_PDU;
    }

    if((csi_rs_pdu.bwp.cyclic_prefix != 0)&&(csi_rs_pdu.bwp.cyclic_prefix != 1))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "CSI-RS PDU BWP cyclic prefix {} out of range, it can be 0 or 1", static_cast<uint16_t>(csi_rs_pdu.bwp.cyclic_prefix));
        return INVALID_FAPI_PDU;
    }

    if(csi_rs_pdu.bwp.scs > 4)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "CSI-RS PDU BWP subcarrier spacing format {} out of range, it cannot be greater than 4", 
        static_cast<uint16_t>(csi_rs_pdu.bwp.scs));
        return INVALID_FAPI_PDU;
    }

    if(csi_rs_pdu.start_rb > (MAX_N_PRBS_SUPPORTED-1))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "CSI-RS PDU prb start index {} out of range, it should be less than or equal to {}", 
        static_cast<uint16_t>(csi_rs_pdu.start_rb), static_cast<uint16_t>(MAX_N_PRBS_SUPPORTED-1));
        return INVALID_FAPI_PDU;
    }
    
    if(csi_rs_pdu.num_of_rbs > MAX_N_PRBS_SUPPORTED)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "CSI-RS PDU number of RBs {} out of range, it should be less than or equal to {}", 
        static_cast<uint16_t>(csi_rs_pdu.num_of_rbs), static_cast<uint16_t>(MAX_N_PRBS_SUPPORTED));
        return INVALID_FAPI_PDU;
    }

    if(csi_rs_pdu.start_rb + csi_rs_pdu.num_of_rbs > MAX_N_PRBS_SUPPORTED)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "CSI-RS PDU end PRB index {} exceeds the max limit, it should be less than or equal to {}",
        static_cast<uint16_t>(csi_rs_pdu.start_rb + csi_rs_pdu.num_of_rbs), static_cast<uint16_t>(MAX_N_PRBS_SUPPORTED));
        return INVALID_FAPI_PDU;
    }

    if((csi_rs_pdu.csi_type != cuphyCsiType_t::TRS)&&(csi_rs_pdu.csi_type != cuphyCsiType_t::NZP_CSI_RS)&&(csi_rs_pdu.csi_type != cuphyCsiType_t::ZP_CSI_RS))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "CSI-RS PDU CSI type {} out of range, it can be 0,1 or 2", static_cast<uint16_t>(csi_rs_pdu.csi_type));
        return INVALID_FAPI_PDU;
    }

    if((csi_rs_pdu.csi_type == cuphyCsiType_t::TRS)&&((csi_rs_pdu.row != 1)||(csi_rs_pdu.freq_density != 3)))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "CSI-RS PDU incorrect TRS configuration, the row value should be 1 and freq density should be 3 for TRS");
        return INVALID_FAPI_PDU;

    }

    if((csi_rs_pdu.row < 1)||(csi_rs_pdu.row > CUPHY_CSIRS_SYMBOL_LOCATION_TABLE_LENGTH))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "CSI-RS PDU row {} out of range, it should be within [1:{}]", 
        static_cast<uint16_t>(csi_rs_pdu.row), static_cast<uint16_t>(CUPHY_CSIRS_SYMBOL_LOCATION_TABLE_LENGTH));
        return INVALID_FAPI_PDU;
    }

    //skipping freq domain bitmap check. All values within the LSB 12 bits are valid.

    if(csi_rs_pdu.sym_l0 > (OFDM_SYMBOLS_PER_SLOT-1))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "CSI-RS PDU first OFDM symbol index l0 {} out of range, it cannot be greater than {}", 
        static_cast<uint16_t>(csi_rs_pdu.sym_l0), static_cast<uint16_t>(OFDM_SYMBOLS_PER_SLOT-1));
        return INVALID_FAPI_PDU;
    }

    if((csi_rs_pdu.sym_l1 < 2)||(csi_rs_pdu.sym_l1 > 12))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "CSI-RS PDU first OFDM symbol index l1 {} out of range, it should be within [2:12]", 
        static_cast<uint16_t>(csi_rs_pdu.sym_l1));
        return INVALID_FAPI_PDU;
    }

    if((csi_rs_pdu.cdm_type != cuphyCdmType_t::NO_CDM)&&(csi_rs_pdu.cdm_type != cuphyCdmType_t::CDM2_FD)&&
    (csi_rs_pdu.cdm_type != cuphyCdmType_t::CDM4_FD2_TD2)&&(csi_rs_pdu.cdm_type != cuphyCdmType_t::CDM8_FD2_TD4))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "CSI-RS PDU CDM type {} out of range, it can be 0,1,2 or 3", static_cast<uint16_t>(csi_rs_pdu.cdm_type));
        return INVALID_FAPI_PDU;
    }

    if((csi_rs_pdu.freq_density != 0)&&(csi_rs_pdu.freq_density != 1)&&(csi_rs_pdu.freq_density != 2)&&(csi_rs_pdu.freq_density != 3))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "CSI-RS PDU freq density {} out of range, it can be 0,1,2 or 3", static_cast<uint16_t>(csi_rs_pdu.freq_density));
        return INVALID_FAPI_PDU;
    }

    if(csi_rs_pdu.scrambling_id > 1023)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "CSI-RS PDU scrambling ID {} out of range, it cannot be greater than 1023", 
        static_cast<uint16_t>(csi_rs_pdu.scrambling_id));
        return INVALID_FAPI_PDU;
    }

    if(csi_rs_pdu.tx_power.power_control_offset > 23)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "CSI-RS PDU power control offset {} out of range, it cannot be greater than 23", 
        static_cast<uint16_t>(csi_rs_pdu.tx_power.power_control_offset));
        return INVALID_FAPI_PDU;
    }

    if((csi_rs_pdu.tx_power.power_control_offset_ss != 0)&&(csi_rs_pdu.tx_power.power_control_offset_ss != 1)
    &&(csi_rs_pdu.tx_power.power_control_offset_ss != 2)&&(csi_rs_pdu.tx_power.power_control_offset_ss != 3))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "CSI-RS PDU power control offset SS {} out of range, it can be 0,1,2 or 3", 
        static_cast<uint16_t>(csi_rs_pdu.tx_power.power_control_offset_ss));

        return INVALID_FAPI_PDU;
    }

    return VALID_FAPI_PDU;

}

int validate_pdcch_pdu(scf_fapi_pdcch_pdu_t& pdcch_pdu)
{
        // Bandwidth part size [3GPP TS 38.213 [4], sec 12]
        // It can take values in [1:275] (supported max value is 273)
        if ((pdcch_pdu.bwp.bwp_size < 1) || (pdcch_pdu.bwp.bwp_size > MAX_N_PRBS_SUPPORTED))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDCCH Invalid BWP size={} the max  value is {}", 
            static_cast<int>(pdcch_pdu.bwp.bwp_size), static_cast<int>(MAX_N_PRBS_SUPPORTED));
            return INVALID_FAPI_PDU;
        }
        
        // Bandwidth part start index, values in range [0:274]
        if (pdcch_pdu.bwp.bwp_start > (MAX_N_PRBS_SUPPORTED-1))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDCCH Invalid BWP start index ={}, it can not be greater than {}", 
            static_cast<int>(pdcch_pdu.bwp.bwp_start), static_cast<int>(MAX_N_PRBS_SUPPORTED-1));
            return INVALID_FAPI_PDU;
        }
        
        // SubcarrierSpacing configuration index [3GPP TS 38.211 [2], sec 4.2],
        // values in range [0:4]
        if (pdcch_pdu.bwp.scs  > 4)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDCCH Invalid SCS={}, it can not be greater than 4", static_cast<int>(pdcch_pdu.bwp.scs));
            return INVALID_FAPI_PDU;
        }
         
         //Cyclic prefix type [3GPP TS 38.211 [2], sec 4.2]
         //0: Normal; 1: Extended
         // TODO: Do we support both options?
        if ((pdcch_pdu.bwp.cyclic_prefix  != 0)&&(pdcch_pdu.bwp.cyclic_prefix  != 1))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDCCH Invalid cyclic prefix={}, it can only be 0 or 1", static_cast<int>(pdcch_pdu.bwp.cyclic_prefix));
            return INVALID_FAPI_PDU;
        }

        if (pdcch_pdu.start_sym_index > (OFDM_SYMBOLS_PER_SLOT-1))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDCCH Invalid start symbol index ={}, it can not be greater than {}", 
            static_cast<int>(pdcch_pdu.start_sym_index), static_cast<int>(OFDM_SYMBOLS_PER_SLOT-1));
            return INVALID_FAPI_PDU;
        }

        if((pdcch_pdu.duration_sym != 1)&&(pdcch_pdu.duration_sym != 2)&&(pdcch_pdu.duration_sym != 3))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDCCH Invalid symbol duration ={}, it can only be 1,2 or 3", 
            static_cast<int>(pdcch_pdu.duration_sym));
            return INVALID_FAPI_PDU;
        }

        //skipping freq domain resource map

        if((pdcch_pdu.cce_reg_mapping_type != 0)&&(pdcch_pdu.cce_reg_mapping_type != 1))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDCCH Invalid cce to reg mapping type ={}, it can only be 0 or 1", 
            static_cast<int>(pdcch_pdu.cce_reg_mapping_type));
            return INVALID_FAPI_PDU;
        }

        if((pdcch_pdu.cce_reg_mapping_type == 0)&&(pdcch_pdu.reg_bundle_size != 6))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDCCH Invalid REG bundle size ={}, it should be 6 for non-interleaved mapping type", 
            static_cast<int>(pdcch_pdu.reg_bundle_size));
            return INVALID_FAPI_PDU;
        }

        if(pdcch_pdu.cce_reg_mapping_type == 1)
        {
            if((pdcch_pdu.duration_sym == 1)||(pdcch_pdu.duration_sym == 2))
            {
                if((pdcch_pdu.reg_bundle_size != 2)&&(pdcch_pdu.reg_bundle_size != 6))
                {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDCCH Invalid REG bundle size ={}, it should be 2 or 6 for \
                    interleaved mapping type with symbol duration 1 or 2", 
                    static_cast<int>(pdcch_pdu.reg_bundle_size));
                    return INVALID_FAPI_PDU;
                }
            }
            
            if((pdcch_pdu.duration_sym == 3)&&(pdcch_pdu.reg_bundle_size != 3)&&(pdcch_pdu.reg_bundle_size != 6))
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDCCH Invalid REG bundle size ={}, it should be 3 or 6 for \
                interleaved mapping type with symbol duration 3", 
                static_cast<int>(pdcch_pdu.reg_bundle_size));
                return INVALID_FAPI_PDU;
            }

            if((pdcch_pdu.interleaver_size != 2)&&(pdcch_pdu.interleaver_size != 3)&&(pdcch_pdu.interleaver_size != 6))
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDCCH Invalid interleaver size ={}, it should be 2,3 or 6 for interleaved mapping type", 
                static_cast<int>(pdcch_pdu.interleaver_size));
                return INVALID_FAPI_PDU;
            }

            if(pdcch_pdu.shift_index > MAX_N_PRBS_SUPPORTED)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDCCH Invalid shift index ={} for interleaved mapping , it cannot be greater than {}", 
                static_cast<int>(pdcch_pdu.interleaver_size), static_cast<int>(MAX_N_PRBS_SUPPORTED));
                return INVALID_FAPI_PDU;
            }
        }

        if((pdcch_pdu.coreset_type != 0)&&(pdcch_pdu.coreset_type != 1))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDCCH Invalid coreset type ={}, it can be 0 or 1", 
            static_cast<int>(pdcch_pdu.coreset_type));
            return INVALID_FAPI_PDU;
        }

        if((pdcch_pdu.precoder_granularity != 0)&&(pdcch_pdu.precoder_granularity != 1))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDCCH Invalid precoder granularity ={}, it can be 0 or 1", 
            static_cast<int>(pdcch_pdu.precoder_granularity));
            return INVALID_FAPI_PDU;
        }

        
        // Bound the DCI count before any iteration over the DCI list below; num_dl_dci is
        // L2-supplied (uint16_t) and an oversized value would drive the loops past the buffer.
        if (pdcch_pdu.num_dl_dci > CUPHY_PDCCH_MAX_DCIS_PER_CORESET) [[unlikely]]
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                "PDCCH num_dl_dci={} exceeds max {} per coreset; rejecting PDU",
                static_cast<int>(pdcch_pdu.num_dl_dci),
                static_cast<int>(CUPHY_PDCCH_MAX_DCIS_PER_CORESET));
            return INVALID_FAPI_PDU;
        }

        auto ptr_dci = reinterpret_cast<scf_fapi_dl_dci_t*>(&pdcch_pdu.dl_dci[0]);

        for (uint8_t dci_idx = 0; dci_idx < pdcch_pdu.num_dl_dci; dci_idx++)
        {
            //Commenting out for now because our default TVs have rnti = 0 
            
            //if(ptr_dci[dci_idx].rnti == 0)
            //{
            //    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "DCI Invalid RNTI value ={}, it can not be 0", 
            //    static_cast<int>(ptr_dci[dci_idx].rnti));
            //    return INVALID_FAPI_PDU;
            //}

            if(ptr_dci[dci_idx].cce_index > 135)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "DCI Invalid CCE index ={}, it can not be greater than 135", 
                static_cast<int>(ptr_dci[dci_idx].cce_index));
                return INVALID_FAPI_PDU;
            }

            if((ptr_dci[dci_idx].aggregation_level != 1) && (ptr_dci[dci_idx].aggregation_level != 2) && (ptr_dci[dci_idx].aggregation_level != 4)
            && (ptr_dci[dci_idx].aggregation_level != 8) && (ptr_dci[dci_idx].aggregation_level != 16))
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "DCI Invalid aggregation level ={}, it can be 1,2,4,8 and 16", 
                static_cast<int>(ptr_dci[dci_idx].aggregation_level));
                return INVALID_FAPI_PDU;
            }
        }

        // Variable-stride walk to validate per-DCI payload_size_bits (CWE-787).
        // Each DCI entry layout: scf_fapi_dl_dci_t | bf_size bytes | tx_power_info | scf_fapi_pdcch_dci_payload_t | payload bytes
        const uint8_t* dci_ptr = reinterpret_cast<const uint8_t*>(&pdcch_pdu.dl_dci[0]);
        for (uint16_t dci_idx = 0; dci_idx < pdcch_pdu.num_dl_dci; ++dci_idx)
        {
            const auto& msg_dci    = *aerial::casts::assume_cast<scf_fapi_dl_dci_t>(dci_ptr);
            const auto& bf         = *aerial::casts::assume_cast<scf_fapi_tx_precoding_beamforming_t>(&msg_dci.payload[0]);
            const uint32_t bf_size = static_cast<uint32_t>(bf.num_prgs) * sizeof(uint16_t)
                                   + static_cast<uint32_t>(bf.num_prgs) * bf.dig_bf_interfaces * sizeof(uint16_t)
                                   + sizeof(bf);
            const auto tx_pwr_size = sizeof(scf_fapi_pdcch_tx_power_info_t);
            const auto& dci_end    = *aerial::casts::assume_cast<scf_fapi_pdcch_dci_payload_t>(
                                         &msg_dci.payload[bf_size + tx_pwr_size]);
            if (dci_end.payload_size_bits > CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES * CHAR_BIT) [[unlikely]]
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                    "PDCCH DCI[{}] payload_size_bits={} exceeds max {} bits; rejecting PDU",
                    static_cast<int>(dci_idx),
                    static_cast<int>(dci_end.payload_size_bits),
                    CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES * CHAR_BIT);
                return INVALID_FAPI_PDU;
            }
            const auto dci_payload_len = (static_cast<uint32_t>(dci_end.payload_size_bits) + (CHAR_BIT - 1U)) / CHAR_BIT;
            dci_ptr += sizeof(scf_fapi_dl_dci_t) + bf_size + tx_pwr_size
                     + sizeof(scf_fapi_pdcch_dci_payload_t) + dci_payload_len;
        }

    return VALID_FAPI_PDU;
}


static int validate_ssb_pdu(scf_fapi_ssb_pdu_t& ssb_pdu)
{
    if(ssb_pdu.phys_cell_id > 1007)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SSB PDU physical cell ID {} out of range, it can not be greater than 1007", 
        static_cast<uint16_t>(ssb_pdu.phys_cell_id));
        return INVALID_FAPI_PDU;
    }

    if((ssb_pdu.beta_pss != 0)&&(ssb_pdu.beta_pss != 1))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SSB PDU beta PSS {} out of range, it can be 0 or 1", static_cast<uint16_t>(ssb_pdu.beta_pss));
        return INVALID_FAPI_PDU;
    }

    if(ssb_pdu.ssb_block_index > 63)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SSB PDU block index {} out of range, it cannot be greater than 63", 
        static_cast<uint16_t>(ssb_pdu.ssb_block_index));
        return INVALID_FAPI_PDU;
    }

    if(ssb_pdu.ssb_subcarrier_offset > 31)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SSB PDU subcarrier offset {} out of range, it cannot be greater than 31", 
        static_cast<uint16_t>(ssb_pdu.ssb_subcarrier_offset));
        return INVALID_FAPI_PDU;
    }

    if(ssb_pdu.ssb_offset_point_a > 2199)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SSB PDU offset point A {} out of range, it can not be greater than 2199", 
        static_cast<uint16_t>(ssb_pdu.ssb_offset_point_a));
        return INVALID_FAPI_PDU;
    }

    if(ssb_pdu.bch_payload_flag != 1)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SSB PDU payload flag {} is not supported, it should be equal to 1", 
        static_cast<uint16_t>(ssb_pdu.bch_payload_flag));
        return INVALID_FAPI_PDU;
    }

    return VALID_FAPI_PDU;
}

static int validate_pdsch_pdu(scf_fapi_pdsch_pdu_t& pdu_info)
{    

    // We do not support any of the optional PDU fields, bitmap should be = 0
    if (pdu_info.pdu_bitmap != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid pdu bitmap={}", static_cast<int>(pdu_info.pdu_bitmap));
        return INVALID_FAPI_PDU;
    }

    // Bandwidth part size [3GPP TS 38.213 [4], sec 12]
    // It can take values in [1:275] (supported max value is 273)
    if((pdu_info.bwp.bwp_size < 1) || (pdu_info.bwp.bwp_size > MAX_N_PRBS_SUPPORTED))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid BWP size={}", static_cast<int>(pdu_info.bwp.bwp_size));
        return INVALID_FAPI_PDU;
    }

    // Bandwidth part start index, values in range [0:274]
    if(pdu_info.bwp.bwp_start > (MAX_N_PRBS_SUPPORTED - 1))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid BWP start={}", static_cast<int>(pdu_info.bwp.bwp_start));
        return INVALID_FAPI_PDU;
    }

    // SubcarrierSpacing configuration index [3GPP TS 38.211 [2], sec 4.2],
    // values in range [0:4]
    // TODO: are all scs configurations supported?
    if(pdu_info.bwp.scs > 4)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid SCS={}", static_cast<int>(pdu_info.bwp.scs));
        return INVALID_FAPI_PDU;
    }

    //Cyclic prefix type [3GPP TS 38.211 [2], sec 4.2]
    //0: Normal; 1: Extended
    // TODO: Do we support both options?
    if(pdu_info.bwp.cyclic_prefix > 1)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid cyclic prefix={}", static_cast<int>(pdu_info.bwp.cyclic_prefix));
        return INVALID_FAPI_PDU;
    }

    // There should be at least a single codeword and not exceed the maximum
    if(pdu_info.num_codewords != 1)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid number of codewords={}", static_cast<int>(pdu_info.num_codewords));
        return INVALID_FAPI_PDU;
    }

    auto ptr_codeword = reinterpret_cast<scf_fapi_pdsch_codeword_t*>(&pdu_info.codewords[0]);

    //There is currently no need for this loop since only a single codeword is supported.
    //Keeping it for future enhancements.
    for(uint8_t cw_idx = 0; cw_idx < pdu_info.num_codewords; cw_idx++)
    {
        // 3GPP TS 38.214 section 5.1.3.1 maximum target code rate is 948
        // It should also not be 0.
        if((ptr_codeword[cw_idx].target_code_rate == 0) || (ptr_codeword[cw_idx].target_code_rate > 9480))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Codeword Idx={} PDSCH Invalid target code rate={}", static_cast<int>(cw_idx), static_cast<int>(ptr_codeword[cw_idx].target_code_rate));
            return INVALID_FAPI_PDU;
        }

        // 3GPP TS 38.214 section 5.1.3.1, qam mod can only take values [2,4,6,8]
        uint8_t qam_mod = ptr_codeword[cw_idx].qam_mod_order;
        if((qam_mod != CUPHY_QAM_4) && (qam_mod != CUPHY_QAM_16) && (qam_mod != CUPHY_QAM_64) && (qam_mod != CUPHY_QAM_256))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Codeword Idx={} PDSCH Invalid QAM mod order={}", static_cast<int>(cw_idx), static_cast<int>(qam_mod));
            return INVALID_FAPI_PDU;
        }

        // 3GPP TS 38.214 section 5.1.3.1, mcs index cannot be greater than 31
        if(ptr_codeword[cw_idx].mcs_index > 31)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Codeword Idx={} PDSCH Invalid MCS index={}", static_cast<int>(cw_idx), static_cast<int>(ptr_codeword[cw_idx].mcs_index));
            return INVALID_FAPI_PDU;
        }

        // 3GPP TS 38.214 section 5.1.3.1, there are 3 MCS tables indexed by [0:2]
        if(ptr_codeword[cw_idx].mcs_table > 2)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Codeword Idx={} PDSCH Invalid MCS table={}", static_cast<int>(cw_idx), static_cast<int>(ptr_codeword[cw_idx].mcs_index));
            return INVALID_FAPI_PDU;
        }

        //3GPP TS 38.211 section 7.3.1.1, rv indices can take values [0:3]. This is applicable to both PDSCH and PUSCH
        if((ptr_codeword[cw_idx].rv_index != 0) && (ptr_codeword[cw_idx].rv_index != 1) &&
           (ptr_codeword[cw_idx].rv_index != 2) && (ptr_codeword[cw_idx].rv_index != 3))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Codeword Idx={} PDSCH Invalid rx index={}", static_cast<int>(cw_idx), static_cast<int>(ptr_codeword[cw_idx].rv_index));
            return INVALID_FAPI_PDU;
        }

        // TODO : Is TB size check needed since it is already done in cuPHY?
    }

    auto end_cw = reinterpret_cast<scf_fapi_pdsch_pdu_end_t*>(&pdu_info.codewords[pdu_info.num_codewords]);

    //dataScramblingIdentityPdsch [TS38.211, sec 7.3.1.1], it is represented by uint16.
    if(end_cw->data_scrambling_id > UINT16_MAX)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid data scrambling id={}", static_cast<int>(end_cw->data_scrambling_id));
        return INVALID_FAPI_PDU;
    }

    //Maximum DL layers per transport block
    if(end_cw->num_of_layers > MAX_DL_LAYERS_PER_TB)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid number of layers={}", static_cast<int>(end_cw->num_of_layers));
        return INVALID_FAPI_PDU;
    }

    //PDSCH DMRS reference point "k" - used for tone mapping [3GPP TS 38.211 [2], sec 7.4.1.1.2]
    // it can take values 0 and 1
    if((end_cw->ref_point != 0) && (end_cw->ref_point != 1))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid DMRS reference point(k)={}", static_cast<int>(end_cw->ref_point));
        return INVALID_FAPI_PDU;
    }

    //We need to at least have one DMRS symbol
    if(end_cw->dl_dmrs_sym_pos == 0)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH no DMRS is allocated!");
        return INVALID_FAPI_PDU;
    }

    //TODO: The following check assumes all bits are set, i.e. 14 DMRS symbols.
    //This check may not be necessary since setting bits beyond the 14 LSBs may not
    //impact the functionality.
    //What is the maximum number of DMRS symbols we actually support?
    if(end_cw->dl_dmrs_sym_pos > ((1 << OFDM_SYMBOLS_PER_SLOT) - 1))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid DMRS positions={}", static_cast<int>(end_cw->dl_dmrs_sym_pos));
        return INVALID_FAPI_PDU;
    }

    // DL DMRS config type [3GPP TS 38.211 [2], sec 7.4.1.1.2], it can only take values 0 and 1
    if((end_cw->dmrs_config_type != 0) && (end_cw->dmrs_config_type != 1))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid DMRS config type={}", static_cast<int>(end_cw->dmrs_config_type));
        return INVALID_FAPI_PDU;
    }

    //PDSCH DMRS scrambling ID [3GPP TS 38.211, section 7.4.1.1.1] is represented by uint16
    if(end_cw->dl_dmrs_scrambling_id > UINT16_MAX)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid DMRS scramb id={}", static_cast<int>(end_cw->dl_dmrs_scrambling_id));
        return INVALID_FAPI_PDU;
    }

    // DMRS sequence initialization [TS38.211, sec 7.4.1.1.2]. It can only take values 0 and 1
    if((end_cw->sc_id != 0) && (end_cw->sc_id != 1))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid SCID={}", static_cast<int>(end_cw->sc_id));
        return INVALID_FAPI_PDU;
    }

    // Number of DM-RS CDM groups without data [3GPP TS 38.212 [3], sec 7.3.1.2.2] [3GPP TS 38.214 [5], Table 4.1-1]
    // It determines the ratio of PDSCH EPRE to DM-RS EPRE. It can take values in the range [1:3]
    // We don't support =3 for type I DMRS
    uint8_t cdm_no_data = end_cw->num_dmrs_cdm_grps_no_data;
    if((cdm_no_data < 1) || (cdm_no_data > 2))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid num DMRS grp no data={}", static_cast<int>(cdm_no_data));
        return INVALID_FAPI_PDU;
    }

    //There should be at least one DMRS port.
    if(end_cw->dmrs_ports == 0)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH no DMRS ports are defined!");
        return INVALID_FAPI_PDU;
    }

    // Bitmap that indicates the DMRS port ids as defined in [3GPP TS 38.212 [3], 7.3.1.2.2]
    // Only LSB 12 bits are used.
    if(end_cw->dmrs_ports > 0xFFF)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid DMRS ports={}", static_cast<int>(end_cw->dmrs_ports));
        return INVALID_FAPI_PDU;
    }

    // Resource Allocation Type [3GPP TS 38.214 [5], sec 5.1.2.2]. It can be 0 or 1.
    if(end_cw->resource_alloc > 1)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid resource alloc type={}", static_cast<int>(end_cw->resource_alloc));
        return INVALID_FAPI_PDU;
    }

    //TODO: skipped rbBitmap checking. Is it needed?

    if(end_cw->rb_start > (MAX_N_PRBS_SUPPORTED - 1))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid rb start={}", static_cast<int>(end_cw->rb_start));
        return INVALID_FAPI_PDU;
    }

    if((end_cw->rb_size < 1) || (end_cw->rb_size > MAX_N_PRBS_SUPPORTED))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid rb size={}", static_cast<int>(end_cw->rb_size));
        return INVALID_FAPI_PDU;
    }

    // VRB-to-PRB-mapping [3GPP TS 38.211 [2], sec 7.3.1.6]. It can take values in [0:2]
    // // This is currently not used by cuPHY
    // if(end_cw->vrb_to_prb_mapping > 2)
    // {
    //     NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid vrb to prb mapping={}", static_cast<int>(end_cw->vrb_to_prb_mapping));
    //     return INVALID_FAPI_PDU;
    // }

    // Start symbol index of PDSCH mapping from the start of the slot, S. [3GPP TS 38.214 [5], Table 5.1.2.1-1]
    // with 0 indexing.
    if(end_cw->start_sym_index >= OFDM_SYMBOLS_PER_SLOT)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid start symbol index={}", static_cast<int>(end_cw->start_sym_index));
        return INVALID_FAPI_PDU;
    }

    // Number of OFDM symbols per slot
    if((end_cw->num_symbols < 1) || (end_cw->num_symbols > OFDM_SYMBOLS_PER_SLOT))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PDSCH Invalid number of symbols{}", static_cast<int>(end_cw->num_symbols));
        return INVALID_FAPI_PDU;
    }

    //completed all checks
    return VALID_FAPI_PDU;
}


int validate_dl_tti_req(scf_fapi_dl_tti_req_t& msg, uint64_t validate_mask, bool& pdsch_pdu_check)
{
    int ret = VALID_FAPI_PDU;
    
    uint64_t pbch_mask = (1 << channel_type::PBCH)|(1 << channel_type::SSB_PBCH_DMRS);
    pbch_mask = validate_mask & pbch_mask;

    uint64_t pdsch_mask = (1 << channel_type::PDSCH)|(1 << channel_type::PDSCH_DMRS)
    |(1 << channel_type::PDSCH_CSIRS);
    pdsch_mask = validate_mask & pdsch_mask;

    uint64_t csi_rs_mask = (1 << channel_type::PDSCH_CSIRS)|(1 << channel_type::CSI_RS);
    csi_rs_mask = validate_mask & csi_rs_mask;

    uint64_t pdcch_mask = (1 << channel_type::PDCCH_DL);
    pdcch_mask = validate_mask & pdcch_mask;

    uint offset = 0;
    uint16_t msg_len = 0;
    msg_len += sizeof(scf_fapi_dl_tti_req_t) + sizeof(scf_fapi_header_t);
    uint num_pdu_rx = msg.num_pdus;
    uint8_t* data = reinterpret_cast<uint8_t*>(msg.payload);
    for (uint i = 0 ; i < num_pdu_rx; i++) 
    {
        auto &pdu = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(data + offset));
        
        if((pdu.pdu_type == DL_TTI_PDU_TYPE_SSB)&&(pbch_mask))
        {
            auto &pdu_dat = *reinterpret_cast<scf_fapi_ssb_pdu_t*>(&pdu.pdu_config[0]);
            auto ret = validate_ssb_pdu(pdu_dat);
            if(ret == INVALID_FAPI_PDU)
            {
                return ret;
            }
        }
        
        if((pdu.pdu_type == DL_TTI_PDU_TYPE_PDSCH)&&(pdsch_mask))
        {
            auto &pdu_dat = *reinterpret_cast<scf_fapi_pdsch_pdu_t*>(&pdu.pdu_config[0]);
            auto ret = validate_pdsch_pdu(pdu_dat);
            if(ret == INVALID_FAPI_PDU)
            {
                pdsch_pdu_check = false;
                return ret;
            }
        }

        if((pdu.pdu_type == DL_TTI_PDU_TYPE_CSI_RS)&&(csi_rs_mask))
        {
            auto &pdu_dat = *reinterpret_cast<scf_fapi_csi_rsi_pdu_t*>(&pdu.pdu_config[0]);
            auto ret = validate_csirs_pdu(pdu_dat);
            if(ret == INVALID_FAPI_PDU)
            {
                return ret;
            }
        }

        if((pdu.pdu_type == DL_TTI_PDU_TYPE_PDCCH)&&(pdcch_mask))
        {
            auto &pdu_dat = *reinterpret_cast<scf_fapi_pdcch_pdu_t*>(&pdu.pdu_config[0]);
            auto ret = validate_pdcch_pdu(pdu_dat);
            if(ret == INVALID_FAPI_PDU)
            {
                return ret;
            }
        }

        offset += pdu.pdu_size;
    }

    return ret;
}

#ifdef ENABLE_L2_SLT_RSP
// L1 limits validation for PDCCH PDU
int validate_pdcch_pdu_l1_limits(const scf_fapi_pdcch_pdu_t& pdu, nv::pdcch_limit_error_t& error)
{
    // Check if coreset count is within limit
    if (error.coreset_parsed < CUPHY_PDCCH_N_MAX_CORESETS_PER_CELL) {
        error.coreset_parsed++;
        
        // Check if total DCIs across all coresets is within limit
        if (error.dci_parsed + pdu.num_dl_dci < CUPHY_PDCCH_MAX_DCIS_PER_CORESET * CUPHY_PDCCH_N_MAX_CORESETS_PER_CELL) {
            error.dci_parsed += pdu.num_dl_dci;
        } else {
            error.dci_errors+=pdu.num_dl_dci;
            return VALID_FAPI_PDU; /// Drop the DCIs when preparing the slot command
        }
    } else {
        error.coreset_errors++;
        error.dci_errors+=pdu.num_dl_dci;
        return INVALID_FAPI_PDU;
    }
    
    return VALID_FAPI_PDU;
}

/**
 * @brief Update PDCCH error contexts with DCI information
 * @param[in] dci The DCI information from FAPI
 * @param[in,out] pdcch_error The PDCCH limit error structure to update
 */
void update_pdcch_error_contexts(const scf_fapi_dl_dci_t& dci, nv::pdcch_limit_error_t& pdcch_error, const uint8_t& index) {
    // Find the first available slot in the error contexts array
    pdcch_error.pdu_error_contexts[index].rnti = dci.rnti;
}

// L1 limits validation for SSB PDU
int validate_ssb_pdu_l1_limits(const scf_fapi_ssb_pdu_t& pdu, nv::ssb_pbch_limit_error_t & error)
{
    // Check if SSB count is within limit
    if (error.parsed < CUPHY_SSB_MAX_SSBS_PER_CELL_PER_SLOT) {
        error.parsed++;
    } else {
        error.errors++;
        return INVALID_FAPI_PDU;
    }
    
    return VALID_FAPI_PDU;
}

// L1 limits validation for CSI-RS PDU
int validate_csirs_pdu_l1_limits(const scf_fapi_csi_rsi_pdu_t& pdu, nv::csirs_limit_error_t& error)
{
    // Check if CSI-RS count is within limit
    if (error.parsed < CUPHY_CSIRS_MAX_NUM_PARAMS) {
        error.parsed++;
    } else {
        error.errors++;
        return INVALID_FAPI_PDU;
    }
    
    return VALID_FAPI_PDU;
}

// L1 limits validation for PDSCH PDU
int validate_pdsch_pdu_l1_limits(const scf_fapi_pdsch_pdu_t& pdu, nv::pdsch_limit_error_t& error, nv::pdsch_pdu_error_ctxts_info_t& pdsch_pdu_error_contexts_info)
{
    // Check if PDSCH count is within limit
    if (error.parsed < MAX_N_TBS_SUPPORTED) {
        error.parsed++;
    } else {
        error.errors++;
        if (nv::TxNotificationHelper::getEnableTxNotification() && !!error.errors) {
            pdsch_pdu_error_contexts_info.pdsch_pdu_error_contexts[pdsch_pdu_error_contexts_info.pdsch_pdu_error_ctxt_num].rnti = pdu.rnti;
            pdsch_pdu_error_contexts_info.pdsch_pdu_error_contexts[pdsch_pdu_error_contexts_info.pdsch_pdu_error_ctxt_num].pduIndex = pdu.pdu_index;
            pdsch_pdu_error_contexts_info.pdsch_pdu_error_ctxt_num++;
        }
        return INVALID_FAPI_PDU;
    }
    return VALID_FAPI_PDU;
}

// Check DL L1 limit errors and return total errors and error mask
error_pair check_dl_tti_l1_limit_errors(const nv::slot_limit_cell_error_t& cell_error, const nv::slot_limit_group_error_t& group_error)
{
    uint32_t total_errors = 0;
    uint64_t error_mask = 0;

    // Check SSB errors
    if (cell_error.ssb_pbch_errors.errors > 0) {
        total_errors += cell_error.ssb_pbch_errors.errors;
        error_mask |= SCF_FAPI_SSB_PBCH_L1_LIMIT_EXCEEDED;
    }

    // Check PDCCH errors
    if (cell_error.pdcch_errors.coreset_errors > 0 || cell_error.pdcch_errors.dci_errors > 0) {
        total_errors += cell_error.pdcch_errors.coreset_errors + cell_error.pdcch_errors.dci_errors;
        error_mask |= SCF_FAPI_PDCCH_L1_LIMIT_EXCEEDED;
    }

    // Check CSIRS errors
    if (cell_error.csirs_errors.errors > 0) {
        total_errors += cell_error.csirs_errors.errors;
        error_mask |= SCF_FAPI_CSIRS_L1_LIMIT_EXCEEDED;
    }

    // Check PDSCH errors
    if (group_error.pdsch_errors.errors > 0) {
        total_errors += group_error.pdsch_errors.errors;
        error_mask |= SCF_FAPI_PDSCH_L1_LIMIT_EXCEEDED;
    }

    return error_pair(total_errors, error_mask);
}

// Check UL DCI L1 limit errors and return total errors and error mask
error_pair check_ul_dci_l1_limit(const nv::slot_limit_cell_error_t& error)
{
    uint32_t total_errors = 0;
    uint64_t error_mask = 0;

    // Check PDCCH errors for UL DCI
    if (error.pdcch_errors.coreset_errors > 0 || error.pdcch_errors.dci_errors > 0) {
        total_errors += error.pdcch_errors.coreset_errors + error.pdcch_errors.dci_errors;
        error_mask |=  SCF_FAPI_PDCCH_L1_LIMIT_EXCEEDED;
    }

    return error_pair(total_errors, error_mask);
}
#endif
