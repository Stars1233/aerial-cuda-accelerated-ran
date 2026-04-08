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

#include "ru_emulator.hpp"
#include <vector_types.h>
#include <cuda_fp16.h>
#include <math.h>

int RU_Emulator::validate_pdcch(uint8_t cell_index, const struct oran_packet_header_info& header_info, void* buffer, dl_channel channel_type)
{
    uint16_t pkt_pdcch_startPrb;
    uint16_t pkt_pdcch_endPrb;
    uint16_t pkt_pdcch_startPrb_offset;
    uint32_t tv_index;
    unsigned char* slot_buf;
    uint32_t slot_data_size;
    void* mbuf_payload;
    uint16_t payload_len;
    uint32_t flow_offset;
    uint32_t symbol_offset;
    uint32_t prb_offset;
    uint32_t buffer_index;
    uint32_t size_count = 0;
    uint32_t received_checksum = 0;//Not used at the moment, initialized to 0 to pass coverity scan
    uint32_t tv_checksum = 0;//Not used at the moment, initialized to 0 to pass coverity scan
    int comp;
    uint16_t startPrb = header_info.startPrb;
    uint16_t numPrb = header_info.numPrb;
    uint8_t flowId = header_info.flowValue;
    uint8_t launch_pattern_slot = header_info.launch_pattern_slot;
    uint8_t symbolId = header_info.symbolId;
    uint8_t flow_index = header_info.flow_index;
    uint16_t sectionId = header_info.sectionId;
    const struct fssId& fss = header_info.fss;
    struct dl_tv_object* tv_object;
    bool complete = false;
    int dl_prb_size = cell_configs[cell_index].dl_prb_size; // used to for the received buffer; for the TV we'll use tv_dl_prb_size
    std::string ch_type;
    if(channel_type == dl_channel::PDCCH_UL)
    {
        ch_type = "PDCCH_UL";
        tv_object = &pdcch_ul_object;
    }
    else
    {
        tv_object = &pdcch_dl_object;
        ch_type = "PDCCH_DL";
    }

    if(tv_object->initialization_phase[cell_index].load() == RE_ENABLED)
    {
        if(tv_object->init_launch_pattern[launch_pattern_slot].size() == 0)
        {
            return 0;
        }

        if(tv_object->init_launch_pattern[launch_pattern_slot].find(cell_index) == tv_object->init_launch_pattern[launch_pattern_slot].end())
        {
            return 0;
        }
        tv_index = tv_object->init_launch_pattern[launch_pattern_slot][cell_index];
    }
    else
    {
        if(tv_object->launch_pattern[launch_pattern_slot].size() == 0)
        {
            return 0;
        }

        if(tv_object->launch_pattern[launch_pattern_slot].find(cell_index) == tv_object->launch_pattern[launch_pattern_slot].end())
        {
            return 0;
        }
        tv_index = tv_object->launch_pattern[launch_pattern_slot][cell_index];
    }
    struct dl_tv_info& dl_tv_info = tv_object->tv_info[tv_index];

    if(flow_index >= dl_tv_info.numFlows)
    {
        return 0;
    }

    if (cell_configs[cell_index].dl_comp_meth == aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION)
    {
        auto &mod_comp_data = tv_object->mod_comp_data[tv_index];
        auto &hdr = mod_comp_data.mod_comp_header;
        int mod_comp_msg_idx = find_modcomp_msg_id(mod_comp_data, symbolId, flow_index, startPrb, numPrb);
        //if (hdr.find(symbolId) != hdr.end() && hdr[symbolId].find(flow_index) != hdr[symbolId].end() && c_plane_channel_type_checking(c_plane_info, cell_index, pdsch_object))
        // if (hdr.find(symbolId) != hdr.end() && hdr[symbolId].find(flow_index) != hdr[symbolId].end())
        // {
        //     auto &msg_idx_mp = mod_comp_data.fss_mod_comp_payload_idx[cell_index][fss.frameId][fss.subframeId * ORAN_MAX_SLOT_ID + fss.slotId][symbolId][flow_index];
        //     if (msg_idx_mp.find(sectionId) == msg_idx_mp.end())
        //     {
        //         return 0;
        //     }
        //     auto mod_comp_msg_idx = msg_idx_mp[sectionId];
        if (mod_comp_msg_idx != -1)
        {
            auto tv_data_idx = mod_comp_data.global_msg_idx_to_tv_idx[mod_comp_msg_idx];
            slot_buf = (unsigned char *)mod_comp_data.mod_comp_payload[tv_data_idx].data.get();
            auto &payload_params = mod_comp_data.mod_comp_payload_params[tv_data_idx];

            if (buffer != nullptr)
            {
                auto actual_iq_width = oran_umsg_get_iq_width_from_section_buf(static_cast<uint8_t*>(buffer));
                if (actual_iq_width != payload_params[2])
                {
                    re_cons("{} ERROR udIqWidth mismatch: Cell {} 3GPP slot {} F{} S{} S{} Flow {} symbolId {} sectionId {} "
                            "startPrb {} numPrb {} actual {} expected {}",
                            ch_type, cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId,
                            flow_index, symbolId, sectionId, startPrb, numPrb, actual_iq_width, payload_params[2]);
                    tv_object->invalid_flag[cell_index][launch_pattern_slot] = true;
                }
            }

            dl_prb_size = payload_params[2] * 3;
            auto buffer_index = dl_prb_size * (startPrb - payload_params[0]);
            payload_len = numPrb * dl_prb_size;

            if (opt_dlc_tb)
            {
                comp = 0; // Skip IQ comparison when dlc_tb is enabled
            }
            else
            {
                auto byte_buffer = static_cast<uint8_t*>(buffer);
                mbuf_payload = &byte_buffer[ORAN_IQ_COMPRESSED_SECTION_OVERHEAD];
                comp = memcmp(mbuf_payload, &slot_buf[buffer_index], payload_len);
            }
            if (comp != 0)
            {
                re_cons("{} ERROR Invalid byte-match: Cell {} 3GPP slot {} F{} S{} S{} Flow {} symbolId {} startPrb {} numPrb {} buffer_index {}", ch_type,
                        cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId, flow_index, symbolId, startPrb, numPrb, buffer_index);
                re_cons("{} modComp info: msg_idx {} startPrb {} numPrb {} iqWidth {}", ch_type, mod_comp_msg_idx + 1, payload_params[0], payload_params[1], payload_params[2]);
                tv_object->invalid_flag[cell_index][launch_pattern_slot] = true;

#if 0
                for (int prb_idx = 0; prb_idx < numPrb; ++prb_idx)
                {
                    char buffer[MAX_PRINT_LOG_LENGTH];
                    int string_buffer_index = 0;
                    string_buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "F%dS%dS%d Cell %d Flow %d symbolId %d Prb %d RX: ", fss.frameId, fss.subframeId, fss.slotId, cell_index, flow_index, symbolId, prb_idx + startPrb);
                    for (int idx = prb_idx * dl_prb_size; idx < (prb_idx + 1) * dl_prb_size; ++idx)
                    {
                        string_buffer_index += snprintf(buffer + string_buffer_index, MAX_PRINT_LOG_LENGTH - string_buffer_index, "%02X", ((unsigned char *)(mbuf_payload))[idx]);
                    }
                    re_cons("{}", buffer);
                    string_buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "F%dS%dS%d Cell %d Flow %d symbolId %d Prb %d TV: ", fss.frameId, fss.subframeId, fss.slotId, cell_index, flow_index, symbolId, prb_idx + startPrb);

                    for (int idx = prb_idx * dl_prb_size; idx < (prb_idx + 1) * dl_prb_size; ++idx)
                    {
                        string_buffer_index += snprintf(buffer + string_buffer_index, MAX_PRINT_LOG_LENGTH - string_buffer_index, "%02X", ((unsigned char *)(&slot_buf[buffer_index]))[idx]);
                    }
                    re_cons("{}", buffer);
                    string_buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "F%dS%dS%d Cell %d Flow %d symbolId %d Prb %d   : ", fss.frameId, fss.subframeId, fss.slotId, cell_index, flow_index, symbolId, prb_idx + startPrb);

                    for (int idx = prb_idx * dl_prb_size; idx < (prb_idx + 1) * dl_prb_size; ++idx)
                    {
                        string_buffer_index += snprintf(buffer + string_buffer_index, MAX_PRINT_LOG_LENGTH - string_buffer_index, "%s", (((unsigned char *)(&slot_buf[buffer_index]))[idx] == ((unsigned char *)(mbuf_payload))[idx]) ? "__" : "^^");
                    }
                    re_cons("{}", buffer);
                }
#endif
            }
            tv_object->atomic_received_prbs[launch_pattern_slot][cell_index] += numPrb;
            tv_object->mtx[cell_index].lock();
            if (tv_object->atomic_received_prbs[launch_pattern_slot][cell_index].load() >= dl_tv_info.modCompNumPrb)
            {
                tv_object->atomic_received_prbs[launch_pattern_slot][cell_index].store(0);
                complete = true;
            }
            tv_object->mtx[cell_index].unlock();
        }
    }
    else
    {
        /* Select which slot_buf buffer (TV dataset) to use for validation. When approx. validation is enabled (expected when precoding is enabled and we don't use identity
           precoding matrices), we use the uncompressed dataset directly for the TV. */
        bool approx_validation_w_compression = (opt_dl_approx_validation == RE_ENABLED);
        int qams_buffer_index = approx_validation_w_compression ? 16 : cell_configs[cell_index].dl_bit_width;
        if (cell_configs[cell_index].dl_comp_meth == aerial_fh::UserDataCompressionMethod::NO_COMPRESSION)
        {
            qams_buffer_index = FIXED_POINT_16_BITS;
        }
        slot_buf = (unsigned char *)tv_object->qams[qams_buffer_index][tv_index].data.get();
        int tv_dl_prb_size = approx_validation_w_compression ? PRB_SIZE_16F : dl_prb_size;

        // Compute beta_dl using the PRB BWP size for the current cell, if fix_beta_dl isn't set
        if (cell_configs[cell_index].beta_dl == BETA_DL_NOT_SET)
        {
            cell_configs[cell_index].beta_dl = sqrt(cell_configs[cell_index].numerator / (SUBCARRIERS_PER_PRB * dl_tv_info.nPrbDlBwp));
        }

        for (auto &pdu : dl_tv_info.pdu_infos)
        {
            if (symbolId < pdu.startSym)
            {
                continue;
            }

            if (symbolId >= pdu.startSym + pdu.numSym)
            {
                continue;
            }

            // Set PDCCH startPrb
            if (startPrb < pdu.startPrb)
            {
                pkt_pdcch_startPrb = pdu.startPrb;
                pkt_pdcch_startPrb_offset = pdu.startPrb - startPrb;
            }
            else
            {
                pkt_pdcch_startPrb = startPrb;
                pkt_pdcch_startPrb_offset = 0;
            }

            // Set PDCCH endPrb
            if (startPrb + numPrb > pdu.startPrb + pdu.numPrb)
            {
                pkt_pdcch_endPrb = pdu.startPrb + pdu.numPrb;
            }
            else
            {
                pkt_pdcch_endPrb = startPrb + numPrb;
            }
            if (pkt_pdcch_startPrb >= pkt_pdcch_endPrb)
            {
                continue;
            }

            numPrb = pkt_pdcch_endPrb - pkt_pdcch_startPrb;

            payload_len = numPrb * dl_prb_size;
            flow_offset = flow_index * (OFDM_SYMBOLS_PER_SLOT * cell_configs[cell_index].dlGridSize);
            symbol_offset = symbolId * (cell_configs[cell_index].dlGridSize);
            prb_offset = pkt_pdcch_startPrb;
            buffer_index = tv_dl_prb_size * (flow_offset + symbol_offset + prb_offset);

            if (opt_dlc_tb)
            {
                comp = 0; // Skip IQ comparison when dlc_tb is enabled
            }
            else
            {
                auto byte_buffer = static_cast<uint8_t *>(buffer);
                mbuf_payload = &byte_buffer[ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD + (uint32_t)pkt_pdcch_startPrb_offset * (uint32_t)dl_prb_size];

                if (opt_dl_approx_validation == RE_ENABLED)
                {
                    comp = decompress_and_compare_approx_buffer((uint8_t *)mbuf_payload, &slot_buf[buffer_index], payload_len, cell_configs[cell_index].dl_bit_width, cell_configs[cell_index].beta_dl, flow_index, symbolId, startPrb);
                }
                else
                {
                    comp = memcmp((uint8_t *)mbuf_payload, &slot_buf[buffer_index], payload_len);
                }
            }

            if (comp != 0)
            {
                tv_object->invalid_flag[cell_index][launch_pattern_slot] = true;
                re_dbg("ERROR Invalid checksum: Cell {} 3GPP slot {} F{} S{} S{} Flow {} Packet symbolId {} startPrb {} numPrb {} PDCCH startPrb {} numPrb {} pkt_pdcch_startPrb_offset {} rx {:X} exp {:X}",
                       cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId, flow_index, symbolId, header_info.startPrb, header_info.numPrb, pkt_pdcch_startPrb, numPrb, pkt_pdcch_startPrb_offset, received_checksum, tv_checksum);

#if 0
            if (opt_dl_approx_validation != RE_ENABLED) // If we do approx. validation one buffer is uncompressed so prints won't be aligned.
            for(int prb_idx = 0; prb_idx < numPrb; ++prb_idx)
            {
                char buffer[MAX_PRINT_LOG_LENGTH];
                int string_buffer_index = 0;
                string_buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "F%dS%dS%d PDCCH Cell %d Flow %d symbolId %d Prb %d RX: ", fss.frameId, fss.subframeId, fss.slotId, cell_index, flow_index, symbolId, prb_idx + startPrb);
                for(int idx = prb_idx * dl_prb_size; idx < (prb_idx + 1) * dl_prb_size; ++idx)
                {
                    string_buffer_index += snprintf(buffer + string_buffer_index, MAX_PRINT_LOG_LENGTH - string_buffer_index, "%02X", ((unsigned char*)(mbuf_payload))[idx]);
                }
                re_cons("{}", buffer);
                string_buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "F%dS%dS%d PDCCH Cell %d Flow %d symbolId %d Prb %d TV: ", fss.frameId, fss.subframeId, fss.slotId, cell_index, flow_index, symbolId, prb_idx + startPrb);

                for(int idx = prb_idx * dl_prb_size; idx < (prb_idx + 1) * dl_prb_size; ++idx)
                {
                    string_buffer_index += snprintf(buffer + string_buffer_index, MAX_PRINT_LOG_LENGTH - string_buffer_index, "%02X",  ((unsigned char*)(&slot_buf[buffer_index]))[idx]);
                }
                re_cons("{}", buffer);
                string_buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "F%dS%dS%d PDCCH Cell %d Flow %d symbolId %d Prb %d   : ", fss.frameId, fss.subframeId, fss.slotId, cell_index, flow_index, symbolId, prb_idx + startPrb);

                for(int idx = prb_idx * dl_prb_size; idx < (prb_idx + 1) * dl_prb_size; ++idx)
                {
                    string_buffer_index += snprintf(buffer + string_buffer_index, MAX_PRINT_LOG_LENGTH - string_buffer_index, "%s",  (((unsigned char*)(&slot_buf[buffer_index]))[idx] == ((unsigned char*)(mbuf_payload))[idx]) ? "__" : "^^");
                }
                re_cons("{}", buffer);
            }
#endif
            }

            tv_object->atomic_received_prbs[launch_pattern_slot][cell_index] += numPrb;
            tv_object->mtx[cell_index].lock();
            if (tv_object->atomic_received_prbs[launch_pattern_slot][cell_index].load() >= dl_tv_info.numPrb * std::min(dl_tv_info.numFlows, (uint8_t)cell_configs[cell_index].eAxC_DL.size()))
            {
                tv_object->atomic_received_prbs[launch_pattern_slot][cell_index].store(0);
                complete = true;
            }
            tv_object->mtx[cell_index].unlock();
            if (complete)
            {
                break;
            }
        }
    }

    if (complete)
    {
        if (tv_object->invalid_flag[cell_index][launch_pattern_slot])
        {
            re_cons("{} Complete Cell {} 3GPP slot {} F{} S{} S{} validation error", dl_channel_string[channel_type].c_str(), cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId);
            ++tv_object->error_slot_counters[cell_index];
        }
        else
        {
            re_info("{} Complete Cell {} 3GPP slot {} F{} S{} S{} OK", dl_channel_string[channel_type].c_str(), cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId);
            ++tv_object->good_slot_counters[cell_index];
            ++tv_object->throughput_slot_counters[cell_index];
        }
        ++tv_object->total_slot_counters[cell_index];
        tv_object->invalid_flag[cell_index][launch_pattern_slot] = false;

        if (tv_object->init_slot_counters[cell_index].load() <= tv_object->total_slot_counters[cell_index].load())
        {
            tv_object->initialization_phase[cell_index].store(0);
        }
        return 1;
    }
    return 0;
}
