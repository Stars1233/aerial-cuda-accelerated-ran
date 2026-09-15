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

#include "scf_5g_fapi_ul_validate.hpp"

#include "aerial/casts/casts.hpp"

#include <cstddef>

#define TAG (NVLOG_TAG_BASE_SCF_L2_ADAPTER + 9) // "SCF.UL_FAPI_VALIDATE"

static constexpr uint8_t srs_symb_idx_to_numSymb[]={1,2,4};

/**
 * @brief Validates the SRS (Sounding Reference Signal) PDU parameters
 * 
 * @param srs_pdu Reference to the SRS PDU structure to validate
 * @return int Returns VALID_FAPI_PDU if validation passes, INVALID_FAPI_PDU otherwise
 * 
 * This function performs validation checks on various SRS PDU parameters including:
 * - RNTI range check
 * - BWP size and configuration
 * - Number of antenna ports
 * - Number of symbols and repetitions
 * - Time start position
 * - Configuration index
 * - Sequence ID
 * - Bandwidth index
 * - Comb size and offset
 * - Frequency position and shift
 * - And other SRS-specific parameters
 */
static int validate_srs_pdu(scf_fapi_srs_pdu_t& srs_pdu)
{
    if(srs_pdu.rnti > 65535)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU rnti {} out of range", static_cast<uint16_t>(srs_pdu.rnti));
        return INVALID_FAPI_PDU;
    }
    if((srs_pdu.bwp.bwp_size < 1) || (srs_pdu.bwp.bwp_size > 275))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU BWP size {} out of range", static_cast<uint16_t>(srs_pdu.bwp.bwp_size));
        return INVALID_FAPI_PDU;
    }
    if(srs_pdu.bwp.scs > 4)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU SCS {} out of range", static_cast<uint8_t>(srs_pdu.bwp.scs));
        return INVALID_FAPI_PDU;
    }
    if((srs_pdu.bwp.cyclic_prefix != 0) && (srs_pdu.bwp.cyclic_prefix != 1))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU cyclic_prefix {} out of range", static_cast<uint8_t>(srs_pdu.bwp.cyclic_prefix));
        return INVALID_FAPI_PDU;
    }
    if(srs_pdu.num_ant_ports > 2)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU num_ant_ports {} out of range", static_cast<uint8_t>(srs_pdu.num_ant_ports));
        return INVALID_FAPI_PDU;
    }
    if(srs_pdu.num_symbols > 2)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU num_symbols {} out of range", static_cast<uint8_t>(srs_pdu.num_symbols));
        return INVALID_FAPI_PDU;
    }
    if(srs_pdu.num_repetitions > 2)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU num_repetitions {} out of range", static_cast<uint8_t>(srs_pdu.num_repetitions));
        return INVALID_FAPI_PDU;
    }
    if(srs_pdu.time_start_position > 13)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU time_start_position {} out of range", static_cast<uint8_t>(srs_pdu.time_start_position));
        return INVALID_FAPI_PDU;
    }
    if(srs_pdu.config_index > 63)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU config_index {} out of range", static_cast<uint8_t>(srs_pdu.config_index));
        return INVALID_FAPI_PDU;
    }
    if(srs_pdu.sequenceId > 1023)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU sequenceId {} out of range", static_cast<uint16_t>(srs_pdu.sequenceId));
        return INVALID_FAPI_PDU;
    }
    if(srs_pdu.bandwidth_index > 3)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU bandwidth_index {} out of range", static_cast<uint8_t>(srs_pdu.bandwidth_index));
        return INVALID_FAPI_PDU;
    }
#ifdef SCF_FAPI_10_04_SRS
    if(srs_pdu.comb_size > 2)
#else
    if(srs_pdu.comb_size > 1)
#endif
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU comb_size {} out of range", static_cast<uint8_t>(srs_pdu.comb_size));
        return INVALID_FAPI_PDU;
    }
    else
    {
        if(srs_pdu.comb_size == 0)
        {
            if(srs_pdu.comb_offset > 1)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU comb_size(0) comb_offset {} out of range", static_cast<uint8_t>(srs_pdu.comb_offset));
                return INVALID_FAPI_PDU;
            }
            if(srs_pdu.cyclic_shift > 7)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU comb_size(0) cyclic_shift {} out of range", static_cast<uint8_t>(srs_pdu.cyclic_shift));
                return INVALID_FAPI_PDU;
            }
        }
        else if(srs_pdu.comb_size == 1)
        {
            if(srs_pdu.comb_offset > 3)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU comb_size(1) comb_offset {} out of range", static_cast<uint8_t>(srs_pdu.comb_offset));
                return INVALID_FAPI_PDU;
            }
            if(srs_pdu.cyclic_shift > 11)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU comb_size(1) cyclic_shift {} out of range", static_cast<uint8_t>(srs_pdu.cyclic_shift));
                return INVALID_FAPI_PDU;
            }
        }
#ifdef SCF_FAPI_10_04_SRS
        else if(srs_pdu.comb_size == 2)
        {
            if(srs_pdu.comb_offset > 7)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU comb_size(2) comb_offset {} out of range", static_cast<uint8_t>(srs_pdu.comb_offset));
                return INVALID_FAPI_PDU;
            }
            if(srs_pdu.cyclic_shift > 5)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU comb_size(2) cyclic_shift {} out of range", static_cast<uint8_t>(srs_pdu.cyclic_shift));
                return INVALID_FAPI_PDU;
            }
        }
#endif
    }
    if(srs_pdu.frequency_position > 67)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU frequency_position {} out of range", static_cast<uint8_t>(srs_pdu.frequency_position));
        return INVALID_FAPI_PDU;
    }
    if(srs_pdu.frequency_shift > 268)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU frequency_shift {} out of range", static_cast<uint16_t>(srs_pdu.frequency_shift));
        return INVALID_FAPI_PDU;
    }
    if(srs_pdu.frequency_hopping > 3)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU frequency_hopping {} out of range", static_cast<uint8_t>(srs_pdu.frequency_hopping));
        return INVALID_FAPI_PDU;
    }
    if(srs_pdu.group_or_sequence_hopping > 2)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU group_or_sequence_hopping {} out of range", static_cast<uint8_t>(srs_pdu.group_or_sequence_hopping));
        return INVALID_FAPI_PDU;
    }
    if(srs_pdu.resource_type > 2)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU resource_type {} out of range", static_cast<uint8_t>(srs_pdu.resource_type));
        return INVALID_FAPI_PDU;
    }
    if((srs_pdu.t_srs != 1) && (srs_pdu.t_srs != 2) && (srs_pdu.t_srs != 3) && (srs_pdu.t_srs != 4) && (srs_pdu.t_srs != 5) && 
       (srs_pdu.t_srs != 8) && (srs_pdu.t_srs != 10) && (srs_pdu.t_srs != 16) && (srs_pdu.t_srs != 20) && (srs_pdu.t_srs != 32) &&
       (srs_pdu.t_srs != 40) && (srs_pdu.t_srs != 64) && (srs_pdu.t_srs != 80) && (srs_pdu.t_srs != 160) && (srs_pdu.t_srs != 320) &&
       (srs_pdu.t_srs != 640) && (srs_pdu.t_srs != 1280) && (srs_pdu.t_srs != 2560))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU t_srs {} out of range", static_cast<uint16_t>(srs_pdu.t_srs));
        return INVALID_FAPI_PDU;
    }
    if(srs_pdu.t_offset > 2559)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU t_offset {} out of range", static_cast<uint16_t>(srs_pdu.t_offset));
        return INVALID_FAPI_PDU;
    }
#ifdef SCF_FAPI_10_04_SRS
    uint8_t* next = &srs_pdu.payload[0];
    scs_fapi_v4_srs_params_t* srs_v4_parms = aerial::casts::assume_cast<scs_fapi_v4_srs_params_t>(next);
    if((srs_v4_parms->srs_bw_size < 4)||(srs_v4_parms->srs_bw_size > 272))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS V4 PDU srs_bw_size {} out of range", static_cast<uint16_t>(srs_v4_parms->srs_bw_size));
        return INVALID_FAPI_PDU;
    }
    for(uint32_t i = 0; i < srs_symb_idx_to_numSymb[srs_pdu.num_symbols]; i++)
    {
        if(srs_v4_parms->srs_bw_sq_info[i].srs_bandwidth_start > 268)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS V4 PDU srs_bandwidth_start {} out of range", static_cast<uint16_t>(srs_v4_parms->srs_bw_sq_info[i].srs_bandwidth_start));
            return INVALID_FAPI_PDU;
        }
        if(srs_v4_parms->srs_bw_sq_info[i].sequence_group > 29)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS V4 PDU sequence_group {} out of range", static_cast<uint16_t>(srs_v4_parms->srs_bw_sq_info[i].sequence_group));
            return INVALID_FAPI_PDU;
        }
        if(srs_v4_parms->srs_bw_sq_info[i].sequence_number > 1)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS V4 PDU sequence_number {} out of range", static_cast<uint16_t>(srs_v4_parms->srs_bw_sq_info[i].sequence_number));
            return INVALID_FAPI_PDU;
        }
    }
    if(!(srs_v4_parms->usage & (SRS_REPORT_FOR_BEAM_MANAGEMENT|SRS_REPORT_FOR_CODEBOOK|SRS_REPORT_FOR_NON_CODEBOOK)))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS V4 PDU usage {} unsupported", static_cast<uint32_t>(srs_v4_parms->usage));
        return INVALID_FAPI_PDU;
    }
#if 1
    if(srs_v4_parms->report_type != 1)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS V4 PDU report_type {} unsupported", static_cast<uint8_t>(srs_v4_parms->report_type));
        return INVALID_FAPI_PDU;
    }
#else
    if (srs_v4_parms->usage & SRS_REPORT_FOR_BEAM_MANAGEMENT)
    {
        if(srs_v4_parms->report_type[SRS_USAGE_FOR_BEAM_MANAGEMENT] != 1)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS V4 PDU report_type {} unsupported", static_cast<uint8_t>(srs_v4_parms->report_type[SRS_USAGE_FOR_BEAM_MANAGEMENT]));
            return INVALID_FAPI_PDU;
        }
    }
    if (srs_v4_parms->usage & SRS_REPORT_FOR_CODEBOOK)
    {
        if(srs_v4_parms->report_type[SRS_USAGE_FOR_CODEBOOK] != 1)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS V4 PDU report_type {} unsupported", static_cast<uint8_t>(srs_v4_parms->report_type[SRS_USAGE_FOR_CODEBOOK]));
            return INVALID_FAPI_PDU;
        }
    }
    if (srs_v4_parms->usage & SRS_REPORT_FOR_NON_CODEBOOK)
    {
        if(srs_v4_parms->report_type[SRS_USAGE_FOR_NON_CODEBOOK] != 1)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS V4 PDU report_type {} unsupported", static_cast<uint8_t>(srs_v4_parms->report_type[SRS_USAGE_FOR_NON_CODEBOOK]));
            return INVALID_FAPI_PDU;
        }
    }
#endif
    if(srs_v4_parms->sing_val_rep != 255)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS V4 PDU sing_val_rep {} unsupported", static_cast<uint8_t>(srs_v4_parms->sing_val_rep));
        return INVALID_FAPI_PDU;
    }
    if(srs_v4_parms->iq_repr != 2)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS V4 PDU iq_repr {} unsupported", static_cast<uint8_t>(srs_v4_parms->iq_repr));
        return INVALID_FAPI_PDU;
    }
    if((srs_v4_parms->prg_size < 1) || (srs_v4_parms->prg_size > 272))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS V4 PDU prg_size {} unsupported", static_cast<uint16_t>(srs_v4_parms->prg_size));
        return INVALID_FAPI_PDU;
    }
    if((srs_v4_parms->num_of_tot_ue_ant < 1) || (srs_v4_parms->num_of_tot_ue_ant > 16))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS V4 PDU num_of_tot_ue_ant {} out of range", static_cast<uint16_t>(srs_v4_parms->num_of_tot_ue_ant));
        return INVALID_FAPI_PDU;
    }
    
    //TODO: srs_v4_parms->ue_ant_in_this_srs_res_set
    //TODO: srs_v4_parms->samp_ue_ant
    
    if (srs_v4_parms->rep_scope != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS V4 PDU reportScope = {}, only reportScope sampledUeAntennas is supported. Hence, SRS.IND failed!", srs_v4_parms->rep_scope);
        return  INVALID_FAPI_PDU; /* ERROR */
    }
    if (srs_v4_parms->num_ul_spat_strm_ports != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS V4 PDU num_ul_spat_strm_ports = {}, un-supported", srs_v4_parms->num_ul_spat_strm_ports);
        return  INVALID_FAPI_PDU; /* ERROR */
    }
#endif
    return VALID_FAPI_PDU;
}

/**
 * @brief Validates the PUCCH (Physical Uplink Control Channel) PDU parameters
 * 
 * @param pucch_pdu Reference to the PUCCH PDU structure to validate
 * @return int Returns VALID_FAPI_PDU if validation passes, INVALID_FAPI_PDU otherwise
 * 
 * This function performs validation checks on various PUCCH PDU parameters including:
 * - BWP size and configuration
 * - Format type
 * - PRB start and size
 * - Symbol configuration
 * - Frequency hopping parameters
 * - Cyclic shift and scrambling
 * - DMRS configuration
 * - HARQ and CSI bit lengths
 */
static int validate_pucch_pdu(scf_fapi_pucch_pdu_t& pucch_pdu)
{
    if((pucch_pdu.bwp.bwp_size < 1) || (pucch_pdu.bwp.bwp_size > MAX_N_PRBS_SUPPORTED))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU BWP size {} out of range", static_cast<uint16_t>(pucch_pdu.bwp.bwp_size));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "BWP size can be from 1 to {}", static_cast<uint16_t>(MAX_N_PRBS_SUPPORTED));
        return INVALID_FAPI_PDU;
    }

    if(pucch_pdu.bwp.bwp_start > (MAX_N_PRBS_SUPPORTED-1))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU BWP start index {} out of range", static_cast<uint16_t>(pucch_pdu.bwp.bwp_start));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "BWP start index should be less than {}", static_cast<uint16_t>(MAX_N_PRBS_SUPPORTED));
        return INVALID_FAPI_PDU;
    }

    if((pucch_pdu.bwp.cyclic_prefix != 0)&&(pucch_pdu.bwp.cyclic_prefix != 1))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU BWP cyclic prefix {} out of range", static_cast<uint16_t>(pucch_pdu.bwp.cyclic_prefix));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "BWP cyclic prefix type can be 0 or 1");
        return INVALID_FAPI_PDU;
    }

    if(pucch_pdu.bwp.scs > 4)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU BWP subcarrier spacing format {} out of range", static_cast<uint16_t>(pucch_pdu.bwp.scs));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "BWP subcarrier spacing can not be greater than 4");
        return INVALID_FAPI_PDU;
    }

    if(pucch_pdu.format_type > UL_TTI_PUCCH_FORMAT_4)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU unsupported format {}", static_cast<uint16_t>(pucch_pdu.format_type));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH format can not be greater than {}", static_cast<uint16_t>(UL_TTI_PUCCH_FORMAT_4));
        return INVALID_FAPI_PDU;
    }

    if(pucch_pdu.prb_start > (MAX_N_PRBS_SUPPORTED-1))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU prb start index {} out of range", static_cast<uint16_t>(pucch_pdu.prb_start));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH prb start index should be less than {}", static_cast<uint16_t>(MAX_N_PRBS_SUPPORTED));
        return INVALID_FAPI_PDU;
    }

    if((pucch_pdu.prb_size < 1) ||(pucch_pdu.prb_size > 16))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU prb size {} out of range", static_cast<uint16_t>(pucch_pdu.prb_size));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH prb size can be in the range from 1 to 16");
        return INVALID_FAPI_PDU;
    }

    if(pucch_pdu.start_symbol_index > (OFDM_SYMBOLS_PER_SLOT-1))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU start symbol index {} out of range", static_cast<uint16_t>(pucch_pdu.start_symbol_index));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH start symbol index should be less than {}", static_cast<uint16_t>(OFDM_SYMBOLS_PER_SLOT));
        return INVALID_FAPI_PDU;
    }

    if((pucch_pdu.format_type == UL_TTI_PUCCH_FORMAT_0) || (pucch_pdu.format_type == UL_TTI_PUCCH_FORMAT_2))
    {
        if((pucch_pdu.num_of_symbols != 1)&&(pucch_pdu.num_of_symbols != 2))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU number of symbols {} out of range for pucch format {}", 
            static_cast<uint16_t>(pucch_pdu.num_of_symbols), static_cast<uint16_t>(pucch_pdu.format_type));
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH number of symbols can be 1 or 2 for this PUCCH format");
            return INVALID_FAPI_PDU;
        }
    }
    
    if((pucch_pdu.format_type == UL_TTI_PUCCH_FORMAT_1) || (pucch_pdu.format_type == UL_TTI_PUCCH_FORMAT_3) || (pucch_pdu.format_type == UL_TTI_PUCCH_FORMAT_4))
    {
        if((pucch_pdu.num_of_symbols < 4) || (pucch_pdu.num_of_symbols > OFDM_SYMBOLS_PER_SLOT))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU number of symbols {} out of range for pucch format {}", 
            static_cast<uint16_t>(pucch_pdu.num_of_symbols), static_cast<uint16_t>(pucch_pdu.format_type));
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH number of symbols can be in the range from 4 to {} for this PUCCH format", 
            static_cast<uint16_t>(OFDM_SYMBOLS_PER_SLOT));
            return INVALID_FAPI_PDU;
        }
    }

    if(pucch_pdu.freq_hop_flag > 1)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU freq hopping flag {} out of range", static_cast<uint16_t>(pucch_pdu.freq_hop_flag));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH freq hopping flag can be 0 or 1 (disable or enable)");
        return INVALID_FAPI_PDU;
    }

    if(pucch_pdu.second_hop_prb > (MAX_N_PRBS_SUPPORTED-1))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU second hop prb index {} out of range", static_cast<uint16_t>(pucch_pdu.second_hop_prb));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH second hop prb index should be less than {}", static_cast<uint16_t>(MAX_N_PRBS_SUPPORTED));
        return INVALID_FAPI_PDU;
    }

    if(pucch_pdu.hopping_id > 1023)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU hopping id {} out of range", static_cast<uint16_t>(pucch_pdu.hopping_id));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH cell specific scrambling ID should be within the range [0:1023]");
        return INVALID_FAPI_PDU;
    }

     // This check is not needed for PUCCH format 2
    if((pucch_pdu.format_type != UL_TTI_PUCCH_FORMAT_2) && (pucch_pdu.initial_cyclic_shift > 11))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU initial cyclic shift {} out of range", static_cast<uint16_t>(pucch_pdu.initial_cyclic_shift));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH initial cyclic shift should be within the range [0:11]");
        return INVALID_FAPI_PDU;
    }

    if(pucch_pdu.data_scrambling_id > 1023)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU data scrambling identity {} out of range", static_cast<uint16_t>(pucch_pdu.data_scrambling_id));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH data scrambling ID should be within the range [0:1023]");
        return INVALID_FAPI_PDU;
    }
     
     // Index of orthogonal cover code can be with the range [0:6] for PUCCH format 1
    if((pucch_pdu.format_type != UL_TTI_PUCCH_FORMAT_1)&&(pucch_pdu.time_domain_occ_idx > 6))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU index of orthogonal cover code {} out of range", static_cast<uint16_t>(pucch_pdu.time_domain_occ_idx));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH index of orthogonal cover code should be within the range [0:6] for format 1");
        return INVALID_FAPI_PDU;
    }
     
    if (pucch_pdu.format_type == UL_TTI_PUCCH_FORMAT_4)
    {
        if(pucch_pdu.pre_dft_occ_idx > 3)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU index of orthogonal cover code {} out of range", static_cast<uint16_t>(pucch_pdu.pre_dft_occ_idx));
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH index of orthogonal cover code should be within the range [0:3] for format 4");
            return INVALID_FAPI_PDU;
        }
        
        if((pucch_pdu.pre_dft_occ_len != 2) && (pucch_pdu.pre_dft_occ_len != 4))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU length of orthogonal cover code {} out of range", static_cast<uint16_t>(pucch_pdu.pre_dft_occ_len));
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH length of orthogonal cover code can be either 2 or 4 for format 4");
            return INVALID_FAPI_PDU;
        }

    }

    if(pucch_pdu.add_dmrs_flag > 1)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU additional DMRS flag {} out of range", static_cast<uint16_t>(pucch_pdu.add_dmrs_flag));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH additional DMRS flag can be 0 or 1 (disable or enable)");
        return INVALID_FAPI_PDU;
    }
     
     //skipping the check for pucch_pdu.dmrs_scrambling_id, is it really needed? It can be any value within the range of uin16_t
     
     //Needs to be checked for PUCCH format 4
    if((pucch_pdu.format_type == UL_TTI_PUCCH_FORMAT_4) && (pucch_pdu.dmrs_cyclic_shift > 9))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU cyclic shift index of DMRS {} out of range for pucch format {}", 
        static_cast<uint16_t>(pucch_pdu.dmrs_cyclic_shift), static_cast<uint16_t>(pucch_pdu.format_type));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH cyclic shift index of DMRS can not be greater than 9 for this PUCCH format");
        return INVALID_FAPI_PDU;
    }

    if(pucch_pdu.sr_flag > 1)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU SR flag {} out of range", static_cast<uint16_t>(pucch_pdu.sr_flag));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH SR flag can be 0 or 1 (disable or enable)");
        return INVALID_FAPI_PDU;
    }

    if((pucch_pdu.format_type == UL_TTI_PUCCH_FORMAT_0) || (pucch_pdu.format_type == UL_TTI_PUCCH_FORMAT_1))
    {
        if(pucch_pdu.bit_len_harq > 2)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU number of HARQ bits {} out of range for pucch format {}", 
            static_cast<uint16_t>(pucch_pdu.bit_len_harq), static_cast<uint16_t>(pucch_pdu.format_type));
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH number of HARQ bits can not be greater than 2 for this PUCCH format");
            return INVALID_FAPI_PDU;
        }
    }
    
    if((pucch_pdu.format_type == UL_TTI_PUCCH_FORMAT_2) || (pucch_pdu.format_type == UL_TTI_PUCCH_FORMAT_3) || (pucch_pdu.format_type == UL_TTI_PUCCH_FORMAT_4))
    {
        if((pucch_pdu.bit_len_harq < 2) || (pucch_pdu.bit_len_harq > 1706))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU number of HARQ bits {} out of range for pucch format {}", 
            static_cast<uint16_t>(pucch_pdu.bit_len_harq), static_cast<uint16_t>(pucch_pdu.format_type));
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH number of HARQ bits should be within the range [2:1706] for this PUCCH format");
            return INVALID_FAPI_PDU;
        }

        if(pucch_pdu.bit_len_csi_part_1 > 1706)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU number of csi part 1 bits {} out of range for pucch format {}", 
            static_cast<uint16_t>(pucch_pdu.bit_len_csi_part_1), static_cast<uint16_t>(pucch_pdu.format_type));
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH number of csi part 1 bits can not be greater than 1706 for this PUCCH format");
            return INVALID_FAPI_PDU;
        }

        if(pucch_pdu.bit_len_csi_part_2 > 1706)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH PDU number of csi part 2 bits {} out of range for pucch format {}", 
            static_cast<uint16_t>(pucch_pdu.bit_len_csi_part_2), static_cast<uint16_t>(pucch_pdu.format_type));
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PUCCH number of csi part 2 bits can not be greater than 1706 for this PUCCH format");
            return INVALID_FAPI_PDU;
        }
    }

    return VALID_FAPI_PDU;
}

/**
 * @brief Validates the PRACH (Physical Random Access Channel) PDU parameters
 * 
 * @param prach_pdu Reference to the PRACH PDU structure to validate
 * @return int Returns VALID_FAPI_PDU if validation passes, INVALID_FAPI_PDU otherwise
 * 
 * This function performs validation checks on various PRACH PDU parameters including:
 * - Physical cell ID
 * - Number of time-domain RACH occasions
 * - Frequency domain occasion index
 * - Starting symbol
 * - Zero correlation zone configuration
 */
static int validate_prach_pdu(scf_fapi_prach_pdu_t& prach_pdu){
    
    //physical cell ID can be in the range [0:1007]
    if(prach_pdu.phys_cell_id > 1007)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PRACH PDU phys cell ID {} out of range", static_cast<uint16_t>(prach_pdu.phys_cell_id));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "Physical cell ID can not be greater than 1007.");
        return INVALID_FAPI_PDU;
    }
     
     //number of time-domain rach occasions can be in the range [1:7]
    if((prach_pdu.num_prach_ocas < 1)||(prach_pdu.num_prach_ocas > 7))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PRACH PDU number of time domain rach occasions {} out of range", static_cast<uint16_t>(prach_pdu.num_prach_ocas));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "Number of RACH occasions should be with in [1:7].");
        return INVALID_FAPI_PDU;
    }

    //Supported PRACH format type is already checked in cuPHY. Skipping this field.
 
    //frequency domain occasion index can be in the range [0:M-1], where M is configured by msg1-FDM 
    // and it can take values in {1,2,4,8}. The corresponding FAPI values are in the range [0:7]
    if(prach_pdu.num_ra > 7)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PRACH PDU frequency domain occasion index {} out of range", static_cast<uint16_t>(prach_pdu.num_ra));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "Frequency domain RACH occasions should be with in [0:7].");
        return INVALID_FAPI_PDU;
    }
     
    //Starting symbol for the first PRACH TD occasion in the current PRACH FD occasion.
    //It can take values in the range [0:13]
    if(prach_pdu.prach_start_symbol>(OFDM_SYMBOLS_PER_SLOT-1))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PRACH PDU frequency domain occasion index {} out of range", static_cast<uint16_t>(prach_pdu.prach_start_symbol));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "Starting OFDM symbol of PRACH time domain occasion cannot be more than {}.", OFDM_SYMBOLS_PER_SLOT-1);
        return INVALID_FAPI_PDU;
    }

     //Zero correlation zone configuration number, corresponds to the parameter Ncs
     //It can be in the range [0:419]
    if(prach_pdu.num_cs > 419)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PRACH PDU Ncs {} out of range", static_cast<uint16_t>(prach_pdu.num_cs));
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "Zero correlation zone config number cannot be greater than 419.");
        return INVALID_FAPI_PDU;
    }

    //TODO check number of beam indices to match number of antenna ports as defined in cuphycontroller yaml file
    return VALID_FAPI_PDU;
}

/**
 * @brief Validates the UL TTI (Uplink Transmission Time Interval) request
 * 
 * @param msg Reference to the UL TTI request structure to validate
 * @param validate_mask Bit mask indicating which channel types to validate
 * @return int Returns VALID_FAPI_PDU if validation passes, INVALID_FAPI_PDU otherwise
 * 
 * This function validates the UL TTI request by checking:
 * - SRS PDUs if enabled in validate_mask
 * - PUCCH PDUs if enabled in validate_mask
 * - PRACH PDUs if enabled in validate_mask
 */
int validate_ul_tti_req(scf_fapi_ul_tti_req_t& msg, uint64_t validate_mask)
{
    int ret = VALID_FAPI_PDU;
    
    uint64_t srs_mask = 1 << channel_type::SRS;
    srs_mask = validate_mask & srs_mask;
    
    uint64_t pucch_mask = 1 << channel_type::PUCCH;
    pucch_mask = validate_mask & pucch_mask;

    uint64_t prach_mask = 1 << channel_type::PRACH;
    prach_mask = validate_mask & prach_mask;

    uint offset = 0;
    uint16_t msg_len = 0;
    msg_len += sizeof(scf_fapi_ul_tti_req_t) + sizeof(scf_fapi_header_t);
    uint num_pdu_rx = msg.num_pdus;
    uint8_t* data = reinterpret_cast<uint8_t*>(msg.payload);
    for (uint i = 0 ; i < num_pdu_rx; i++) 
    {
        auto &pdu = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(data + offset));
            
        if((pdu.pdu_type == UL_TTI_PDU_TYPE_SRS)&&(srs_mask))
        {
            auto &pdu_dat = *reinterpret_cast<scf_fapi_srs_pdu_t*>(&pdu.pdu_config[0]);
            auto ret = validate_srs_pdu(pdu_dat);
            if(ret == INVALID_FAPI_PDU)
            {
                return ret;
            }
        } else if ((pdu.pdu_type == UL_TTI_PDU_TYPE_PUCCH)&&(pucch_mask)) {
            auto &pdu_dat = *reinterpret_cast<scf_fapi_pucch_pdu_t*>(&pdu.pdu_config[0]);
            auto ret = validate_pucch_pdu(pdu_dat);
            if(ret == INVALID_FAPI_PDU)
            {
                return ret;
            }
        } else if ((pdu.pdu_type == UL_TTI_PDU_TYPE_PRACH)&&(prach_mask)) {
            auto &pdu_dat = *reinterpret_cast<scf_fapi_prach_pdu_t*>(&pdu.pdu_config[0]);
            auto ret = validate_prach_pdu(pdu_dat);
            if(ret == INVALID_FAPI_PDU)
            {
                return ret;
            }
        } else {
            continue;
        }
        offset += pdu.pdu_size;
    }
    return ret;
}

/**
 * @brief Validates the UL/DL beamforming CVI (Channel Vector Information) request
 * 
 * @param msg Reference to the beamforming CVI request structure to validate
 * @param is_ul_bfw Flag indicating if this is an UL beamforming request
 * @param validate_mask Bit mask indicating which channel types to validate
 * @return int Returns VALID_FAPI_PDU if validation passes, INVALID_FAPI_PDU otherwise
 * 
 * This function validates beamforming parameters including:
 * - Resource block start and size
 * - Number of PRGs and PRG size
 * - Number of UEs
 * - RNTI and PDU index
 * - Antenna indices
 */
int validate_ul_dl_bfw_cvi_req(scf_fapi_ul_bfw_cvi_request_t& msg, uint8_t is_ul_bfw, uint64_t validate_mask)
{
    int ret = VALID_FAPI_PDU;

    uint64_t bfw_mask = 1 << channel_type::BFW;
    bfw_mask = validate_mask & bfw_mask;

    NVLOGD_FMT(TAG, "validate_mask={}, bfw_mask={}", validate_mask, bfw_mask);

    if (bfw_mask)
    {
        uint num_pdu_rx = msg.npdus;
        uint8_t* data = reinterpret_cast<uint8_t*>(msg.config_pdu);

        if(is_ul_bfw)
        {
            NVLOGD_FMT(TAG, "Validating PHY_UL_BFW_CVI_REQUEST received with NUMBER of UL PDU ={}", num_pdu_rx);
        }
        else
        {
            NVLOGD_FMT(TAG, "Validating PHY_DL_BFW_CVI_REQUEST received with NUMBER of DL PDU ={}", num_pdu_rx);
        }

        std::size_t offset = 0;
        uint8_t ueIdx = 0;
        uint8_t ueAntIdx = 0;

        for (uint i = 0 ; i < num_pdu_rx; i++)
        {
            auto &bfw_msg = *(reinterpret_cast<scf_fapi_ul_bfw_group_config_t*>(data + offset));
            constexpr std::size_t bfw_msg_fixed_bytes = sizeof(scf_fapi_ul_bfw_group_config_t);
            const uint16_t bfw_pdu_size = bfw_msg.pdu_size;
            if (bfw_pdu_size < bfw_msg_fixed_bytes)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                           "BFW PDU pdu_size {} smaller than fixed header {}",
                           bfw_pdu_size, bfw_msg_fixed_bytes);
                return INVALID_FAPI_PDU;
            }

            if(bfw_msg.dl_bfw_cvi_config.rb_start > 274)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "BFW PDU rbStart {} out of range", static_cast<uint16_t>(bfw_msg.dl_bfw_cvi_config.rb_start));
                return INVALID_FAPI_PDU;
            }

            if(bfw_msg.dl_bfw_cvi_config.rb_size < 1 || bfw_msg.dl_bfw_cvi_config.rb_size > 275)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "BFW PDU rbSize {} out of range", static_cast<uint16_t>(bfw_msg.dl_bfw_cvi_config.rb_size));
                return INVALID_FAPI_PDU;
            }

            if(bfw_msg.dl_bfw_cvi_config.num_prgs < 1 || bfw_msg.dl_bfw_cvi_config.num_prgs > 275)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "BFW PDU numPRGs {} out of range", static_cast<uint16_t>(bfw_msg.dl_bfw_cvi_config.num_prgs));
                return INVALID_FAPI_PDU;
            }

            if(bfw_msg.dl_bfw_cvi_config.prg_size < 1 || bfw_msg.dl_bfw_cvi_config.prg_size > 275)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "BFW PDU prgSize {} out of range", static_cast<uint16_t>(bfw_msg.dl_bfw_cvi_config.prg_size));
                return INVALID_FAPI_PDU;
            }

            if(bfw_msg.dl_bfw_cvi_config.nUes < 1 || bfw_msg.dl_bfw_cvi_config.nUes > 12)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "BFW PDU nUes {} out of range", static_cast<uint16_t>(bfw_msg.dl_bfw_cvi_config.nUes));
                return INVALID_FAPI_PDU;
            }

            uint8_t* next = &bfw_msg.dl_bfw_cvi_config.payload[0];
            std::size_t remaining_bytes = static_cast<std::size_t>(bfw_pdu_size) - bfw_msg_fixed_bytes;

            for(ueIdx = 0; ueIdx < bfw_msg.dl_bfw_cvi_config.nUes; ueIdx++)
            {
                if (remaining_bytes < sizeof(scf_dl_bfw_config_start_t))
                {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                               "BFW PDU ue[{}] header truncated, remaining={} required={}",
                               ueIdx, remaining_bytes, sizeof(scf_dl_bfw_config_start_t));
                    return INVALID_FAPI_PDU;
                }
                scf_dl_bfw_config_start_t& bfw_config_start  = *reinterpret_cast<scf_dl_bfw_config_start_t*>(next);
                const std::size_t ue_config_bytes =
                    sizeof(scf_dl_bfw_config_start_t) + static_cast<std::size_t>(bfw_config_start.num_ue_ants);
                if (remaining_bytes < ue_config_bytes)
                {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                               "BFW PDU ue[{}] payload truncated, remaining={} required={} num_ue_ants={}",
                               ueIdx, remaining_bytes, ue_config_bytes,
                               static_cast<uint8_t>(bfw_config_start.num_ue_ants));
                    return INVALID_FAPI_PDU;
                }

                if(bfw_config_start.rnti > 65535)
                {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "BFW PDU rnti {} out of range", static_cast<uint16_t>(bfw_config_start.rnti));
                    return INVALID_FAPI_PDU;
                }

                if(bfw_config_start.pduIndex > 65535)
                {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "BFW PDU pduIndex {} out of range", static_cast<uint16_t>(bfw_config_start.pduIndex));
                    return INVALID_FAPI_PDU;
                }

                if(bfw_config_start.gnb_ant_index_start > 0)
                {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "BFW PDU gnbAntIdxStart {} out of range", static_cast<uint16_t>(bfw_config_start.gnb_ant_index_start));
                    return INVALID_FAPI_PDU;
                }
                if(bfw_config_start.gnb_ant_index_end > 31)
                {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "BFW PDU gnbAntIdxEnd {} out of range", static_cast<uint16_t>(bfw_config_start.gnb_ant_index_end));
                    return INVALID_FAPI_PDU;
                }

                if(bfw_config_start.num_ue_ants > 4)
                {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "BFW PDU numOfUeAnt {} out of range", static_cast<uint16_t>(bfw_config_start.num_ue_ants));
                    return INVALID_FAPI_PDU;
                }

                uint8_t* ue_ant_idx = &bfw_config_start.payload[0];
                for (ueAntIdx = 0; ueAntIdx < bfw_config_start.num_ue_ants; ueAntIdx++)
                {
                    if(*(ue_ant_idx+ueAntIdx) > 3)
                    {
                        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "BFW PDU ueAntIdx {} at index {} out of range", static_cast<uint16_t>(*(ue_ant_idx+ueAntIdx)), ueAntIdx);
                        return INVALID_FAPI_PDU;
                    }
                }
                next += ue_config_bytes;
                remaining_bytes -= ue_config_bytes;
            }
            offset += bfw_pdu_size;
        }
    }
    return ret;
}

#ifdef ENABLE_L2_SLT_RSP
/**
 * @brief Validates PUSCH PDU against L1 limits
 * 
 * @param pdu Reference to the PUSCH PDU structure to validate
 * @param error Reference to the error tracking structure
 * @return int Returns VALID_FAPI_PDU if validation passes, INVALID_FAPI_PDU otherwise
 * 
 * This function checks if the number of PUSCH PDUs is within the maximum supported limit
 */
int validate_pusch_pdu_l1_limits(const scf_fapi_pusch_pdu_t& pdu, nv::pusch_limit_error_t& error)
{
    // NVLOGD_FMT(TAG, "validate_pusch_pdu_l1_limits: before error.parsed={}, error.errors={}, MAX_N_TBS_SUPPORTED={}", error.parsed, error.errors, MAX_N_TBS_SUPPORTED);

    auto value = VALID_FAPI_PDU;
    if (error.parsed < MAX_N_TBS_SUPPORTED) {
        error.parsed++;
    } else {
        error.errors++;
        value = INVALID_FAPI_PDU;
    }
    // NVLOGD_FMT(TAG, "validate_pusch_pdu_l1_limits: error.parsed={}, error.errors={}, MAX_N_TBS_SUPPORTED={}", error.parsed, error.errors, MAX_N_TBS_SUPPORTED);

    return value;
}

/**
 * @brief Validates SRS PDU against L1 limits
 * 
 * @param pdu Reference to the SRS PDU structure to validate
 * @param error Reference to the error tracking structure
 * @return int Returns VALID_FAPI_PDU if validation passes, INVALID_FAPI_PDU otherwise
 * 
 * This function checks if the number of SRS PDUs is within the maximum supported limit
 */
int validate_srs_pdu_l1_limits(const scf_fapi_srs_pdu_t& pdu, nv::srs_limit_error_t& error)
{
    // Check if SRS count is within limit
    if (error.parsed < slot_command_api::MAX_SRS_PDU_PER_SLOT) {
        error.parsed++;
    } else {
        error.errors++;
        return INVALID_FAPI_PDU;
    }
    
    return VALID_FAPI_PDU;
}

/**
 * @brief Validates PUCCH PDU against L1 limits
 * 
 * @param pdu Reference to the PUCCH PDU structure to validate
 * @param error Reference to the error tracking structure
 * @return int Returns VALID_FAPI_PDU if validation passes, INVALID_FAPI_PDU otherwise
 * 
 * This function checks PUCCH format-specific limits for:
 * - Format 0
 * - Format 1
 * - Format 2
 * - Format 3
 * - Format 4
 */
int validate_pucch_pdu_l1_limits(const scf_fapi_pucch_pdu_t& pdu, nv::pucch_limit_error_t& error)
{
    int ret = VALID_FAPI_PDU;
    switch (pdu.format_type) {
        case UL_TTI_PUCCH_FORMAT_0:
            if (error.pf0_parsed < CUPHY_PUCCH_F0_MAX_UCI_PER_GRP * CUPHY_PUCCH_F0_MAX_GRPS) {
                error.pf0_parsed++;
            } else {
                error.pf0_errors++;
                ret = INVALID_FAPI_PDU;
            }
            break;
        case UL_TTI_PUCCH_FORMAT_1:
            if (error.pf1_parsed < CUPHY_PUCCH_F1_MAX_UCI_PER_GRP * CUPHY_PUCCH_F1_MAX_GRPS) {
                error.pf1_parsed++;
            } else {
                error.pf1_errors++;
                ret = INVALID_FAPI_PDU;
            }
            break;

        case UL_TTI_PUCCH_FORMAT_2:
            if (error.pf2_parsed < CUPHY_PUCCH_F2_MAX_UCI) {
                error.pf2_parsed++;
            } else {
                error.pf2_errors++;
                ret = INVALID_FAPI_PDU;
            }
            break;

        case UL_TTI_PUCCH_FORMAT_3:
            if (error.pf3_parsed < CUPHY_PUCCH_F3_MAX_UCI) {
                error.pf3_parsed++;
            } else {
                error.pf3_errors++;
                ret = INVALID_FAPI_PDU;
            }
            break;

        // TODO: Add support for PUCCH format 4 using CUPHY_PUCCH_F3_MAX_UCI PF3 limits for PF4
        case UL_TTI_PUCCH_FORMAT_4:
            if (error.pf4_parsed < CUPHY_PUCCH_F3_MAX_UCI) {
                error.pf4_parsed++;
            } else {
                error.pf4_errors++;
                ret = INVALID_FAPI_PDU;
            }
            break;

        default:
            break;
    }

    return ret;
}

/**
 * @brief Validates PRACH PDU against L1 limits
 * 
 * @param pdu Reference to the PRACH PDU structure to validate
 * @param error Reference to the error tracking structure
 * @return int Returns VALID_FAPI_PDU if validation passes, INVALID_FAPI_PDU otherwise
 * 
 * This function checks if the number of PRACH PDUs is within the maximum supported limit
 */
int validate_prach_pdu_l1_limits(const scf_fapi_prach_pdu_t& pdu, nv::prach_limit_error_t& error)
{
    // Check if PRACH count is within limit
    if (error.parsed < MAX_PRACH_MAX_OCCASIONS_PER_CELL) {
        error.parsed++;
    } else {
        error.errors++;
        return INVALID_FAPI_PDU;
    }
    
    return VALID_FAPI_PDU;
}

/**
 * @brief Checks UL TTI L1 limit errors and returns total errors and error mask
 * 
 * @param cell_error Reference to the cell error tracking structure
 * @param group_error Reference to the group error tracking structure
 * @return error_pair Returns a pair containing total error count and error mask
 * 
 * This function aggregates errors from:
 * - PUSCH limit errors
 * - SRS limit errors
 * - PUCCH format-specific limit errors
 */
error_pair check_ul_tti_l1_limit_errors(const nv::slot_limit_cell_error_t& cell_error, const nv::slot_limit_group_error_t& group_error)
{
    uint32_t total_errors = 0;
    uint64_t error_mask = 0;

    // Check PUSCH errors
    if (group_error.pusch_errors.errors > 0) {
        total_errors += group_error.pusch_errors.errors;
        error_mask |= SCF_FAPI_PUSCH_L1_LIMIT_EXCEEDED;
    }

    // // Check SRS errors
    // if (cell_error.srs_errors.errors > 0) {
    //     total_errors += cell_error.srs_errors.errors;
    //     error_mask |= SCF_FAPI_SRS_L1_LIMIT_EXCEEDED;
    // }

    // Check PRACH errors
    if (cell_error.prach_errors.errors > 0) {
        total_errors += cell_error.prach_errors.errors;
        error_mask |= SCF_FAPI_PRACH_L1_LIMIT_EXCEEDED;
    }

    // Check PUCCH errors
    if (group_error.pucch_errors.pf0_errors > 0 || group_error.pucch_errors.pf1_errors > 0 || group_error.pucch_errors.pf2_errors > 0 || group_error.pucch_errors.pf3_errors > 0 || group_error.pucch_errors.pf4_errors > 0) {
        total_errors += group_error.pucch_errors.pf0_errors + group_error.pucch_errors.pf1_errors + group_error.pucch_errors.pf2_errors + group_error.pucch_errors.pf3_errors + group_error.pucch_errors.pf4_errors;
        error_mask |= SCF_FAPI_PUCCH_L1_LIMIT_EXCEEDED;
    }

    return error_pair(total_errors, error_mask);
}

#endif
