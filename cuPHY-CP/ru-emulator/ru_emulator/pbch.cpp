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

int RU_Emulator::validate_pbch(uint8_t cell_index, const struct oran_packet_header_info& header_info, void* buffer, uint64_t& prev_pbch_time)
{
    //PBCH Validation
    uint16_t pbch_startPrb;
    uint16_t pbch_endPrb;
    uint16_t pdu_startPrb;
    uint16_t pdu_numPrb;
    uint16_t pdu_startSym;
    uint16_t pdu_numSym;
    uint16_t pbch_startPrb_offset;
    uint16_t endPrb;
    int comp;
    bool complete = false;
    uint32_t tv_index;
    unsigned char* slot_buf;
    uint32_t slot_data_size;
    uint8_t* mbuf_payload;
    uint16_t payload_len;
    uint32_t flow_offset;
    uint32_t symbol_offset;
    uint32_t prb_offset;
    uint32_t buffer_index;
    uint32_t size_count = 0;
    uint32_t received_checksum;
    uint32_t tv_checksum;
    uint64_t curr_pbch_time;

    uint16_t startPrb = header_info.startPrb;
    uint16_t numPrb = header_info.numPrb;
    uint8_t flowId = header_info.flowValue;
    uint8_t launch_pattern_slot = header_info.launch_pattern_slot;
    uint8_t symbolId = header_info.symbolId;
    uint8_t flow_index = header_info.flow_index;
    uint16_t sectionId = header_info.sectionId;
    const struct fssId& fss = header_info.fss;
    int dl_prb_size = cell_configs[cell_index].dl_prb_size; // used to for the received buffer; for the TV we'll use tv_dl_prb_size
    //Only check the first Antenna
    // if(flow_index != 0)
    // {
    //     return 0;
    // }

    if(pbch_object.initialization_phase[cell_index].load() == RE_ENABLED)
    {
        if(pbch_object.init_launch_pattern[launch_pattern_slot].size() == 0)
        {
            return 0;
        }
        tv_index = pbch_object.init_launch_pattern[launch_pattern_slot][cell_index];
    }
    else
    {
        if (pbch_object.launch_pattern[launch_pattern_slot].size() == 0 || pbch_object.launch_pattern[launch_pattern_slot].find(cell_index) == pbch_object.launch_pattern[launch_pattern_slot].end())
        {
            return 0;
        }
        tv_index = pbch_object.launch_pattern[launch_pattern_slot][cell_index];
    }
    struct dl_tv_info& dl_tv_info = pbch_object.tv_info[tv_index];
    if(flow_index >= dl_tv_info.numFlows)
    {
        return 0;
    }

    if (cell_configs[cell_index].dl_comp_meth == aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION)
    {
        auto &mod_comp_data = pbch_object.mod_comp_data[tv_index];
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
                    re_cons("PBCH ERROR udIqWidth mismatch: Cell {} 3GPP slot {} F{} S{} S{} Flow {} symbolId {} sectionId {} "
                            "startPrb {} numPrb {} actual {} expected {}",
                            cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId,
                            flow_index, symbolId, sectionId, startPrb, numPrb, actual_iq_width, payload_params[2]);
                    pbch_object.invalid_flag[cell_index][launch_pattern_slot] = true;
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
                re_cons("PBCH ERROR Invalid byte-match: Cell {} 3GPP slot {} F{} S{} S{} Flow {} symbolId {} startPrb {} numPrb {} buffer_index {}",
                        cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId, flow_index, symbolId, startPrb, numPrb, buffer_index);
                re_cons("PBCH modComp info: msg_idx {} startPrb {} numPrb {} iqWidth {}", mod_comp_msg_idx + 1, payload_params[0], payload_params[1], payload_params[2]);
                pbch_object.invalid_flag[cell_index][launch_pattern_slot] = true;

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

            pbch_object.atomic_received_prbs[launch_pattern_slot][cell_index] += numPrb;
        }
    }
    else
    {
        // re_cons("PBCH checking packet: Cell {} 3GPP slot {} F{} S{} S{} Flow {} symbolId {} startPrb {} numPrb {}", cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId, flowId, symbolId, startPrb, numPrb);

        /* Select which slot_buf buffer (TV dataset) to use for validation. When approx. validation is enabled (expected when precoding is enabled and we don't use identity
    +       precoding matrices), we use the uncompressed dataset directly for the TV. */
        bool approx_validation_w_compression = (opt_dl_approx_validation == RE_ENABLED);
        int qams_buffer_index = approx_validation_w_compression ? 16 : cell_configs[cell_index].dl_bit_width;
        if (cell_configs[cell_index].dl_comp_meth == aerial_fh::UserDataCompressionMethod::NO_COMPRESSION)
        {
            qams_buffer_index = FIXED_POINT_16_BITS;
        }
        slot_buf = (unsigned char *)pbch_object.qams[qams_buffer_index][tv_index].data.get();
        int tv_dl_prb_size = approx_validation_w_compression ? PRB_SIZE_16F : dl_prb_size;

        // Compute beta_dl using the PRB BWP size for the current cell, if fix_beta_dl isn't set
        if (cell_configs[cell_index].beta_dl == BETA_DL_NOT_SET)
        {
            cell_configs[cell_index].beta_dl = sqrt(cell_configs[cell_index].numerator / (SUBCARRIERS_PER_PRB * dl_tv_info.nPrbDlBwp));
        }

        for (int i = 0; i < dl_tv_info.pdu_infos.size(); ++i)
        {
            pdu_startSym = dl_tv_info.pdu_infos[i].startSym;
            pdu_numSym = dl_tv_info.pdu_infos[i].numSym;
            pdu_startPrb = dl_tv_info.pdu_infos[i].startPrb;
            pdu_numPrb = dl_tv_info.pdu_infos[i].numPrb;
            // re_cons("PBCH checking packet: Cell {} 3GPP slot {} F{} S{} S{} Flow {} symbolId {} startPrb {} numPrb {}", cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId, flowId, symbolId, startPrb, numPrb);
            // re_cons("PBCH checking packet: Cell {} 3GPP slot {} F{} S{} S{} Flow {} PDU startSym {} numSym {} startPrb {} numPrb {}", cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId, flowId, pdu_startSym, pdu_numSym, pdu_startPrb, pdu_numPrb);

            if (symbolId < pdu_startSym || symbolId >= pdu_startSym + pdu_numSym)
            {
                continue;
            }
            // Set PBCH startPrb
            if (startPrb < pdu_startPrb)
            {
                pbch_startPrb = pdu_startPrb;
                pbch_startPrb_offset = pbch_startPrb - startPrb;
            }
            else
            {
                pbch_startPrb = startPrb;
                pbch_startPrb_offset = 0;
            }

            // Set PBCH endPrb
            if (startPrb + numPrb > pdu_startPrb + pdu_numPrb)
            {
                pbch_endPrb = pdu_startPrb + pdu_numPrb;
            }
            else
            {
                pbch_endPrb = startPrb + numPrb;
            }

            if (pbch_startPrb > pbch_endPrb)
            {
                continue;
            }

            numPrb = pbch_endPrb - pbch_startPrb;

            payload_len = numPrb * dl_prb_size;
            flow_offset = flow_index * (OFDM_SYMBOLS_PER_SLOT * cell_configs[cell_index].dlGridSize);
            symbol_offset = symbolId * (cell_configs[cell_index].dlGridSize);
            prb_offset = (pbch_startPrb);
            buffer_index = tv_dl_prb_size * (flow_offset + symbol_offset + prb_offset);

            if (opt_dlc_tb)
            {
                comp = 0; // Skip IQ comparison when dlc_tb is enabled
            }
            else
            {
                // static cast to uint_8* and then access offset
                auto byte_buffer = static_cast<uint8_t *>(buffer);
                mbuf_payload = &byte_buffer[ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD + (uint32_t)pbch_startPrb_offset * dl_prb_size];
                
                if (opt_dl_approx_validation == RE_ENABLED)
                {
                    comp = decompress_and_compare_approx_buffer((uint8_t *)mbuf_payload, &slot_buf[buffer_index], payload_len, cell_configs[cell_index].dl_bit_width, cell_configs[cell_index].beta_dl, flow_index, symbolId, startPrb);
                }
                else
                {
                    comp = memcmp(mbuf_payload, &slot_buf[buffer_index], payload_len);
                }
            }

            if (comp != 0)
            {
                pbch_object.invalid_flag[cell_index][launch_pattern_slot] = true;
                re_dbg("ERR Invalid checksum: Cell {} 3GPP slot {} F{} S{} S{} Flow {} Packet symbolId {} startPrb {} numPrb {} PBCH startPrb {} numPrb {}", cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId, flow_index, symbolId, header_info.startPrb, header_info.numPrb, pbch_startPrb, numPrb);
#if 0
            if (opt_dl_approx_validation != RE_ENABLED) // If we do approx. validation one buffer is uncompressed so prints won't be aligned.
            for(int prb_idx = 0; prb_idx < numPrb; ++prb_idx)
            {
                char buffer[MAX_PRINT_LOG_LENGTH];
                int string_buffer_index = 0;
                string_buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "F%dS%dS%d SSB Cell %d Flow %d symbolId %d Prb %d RX: ", fss.frameId, fss.subframeId, fss.slotId, cell_index, flow_index, symbolId, prb_idx + startPrb);
                for(int idx = prb_idx * dl_prb_size; idx < (prb_idx + 1) * dl_prb_size; ++idx)
                {
                    string_buffer_index += snprintf(buffer + string_buffer_index, MAX_PRINT_LOG_LENGTH - string_buffer_index, "%02X", ((unsigned char*)(mbuf_payload))[idx]);
                }
                re_cons("{}", buffer);
                string_buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "F%dS%dS%d SSB Cell %d Flow %d symbolId %d Prb %d TV: ", fss.frameId, fss.subframeId, fss.slotId, cell_index, flow_index, symbolId, prb_idx + startPrb);

                for(int idx = prb_idx * dl_prb_size; idx < (prb_idx + 1) * dl_prb_size; ++idx)
                {
                    string_buffer_index += snprintf(buffer + string_buffer_index, MAX_PRINT_LOG_LENGTH - string_buffer_index, "%02X",  ((unsigned char*)(&slot_buf[buffer_index]))[idx]);
                }
                re_cons("{}", buffer);
                string_buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "F%dS%dS%d SSB Cell %d Flow %d symbolId %d Prb %d   : ", fss.frameId, fss.subframeId, fss.slotId, cell_index, flow_index, symbolId, prb_idx + startPrb);

                for(int idx = prb_idx * dl_prb_size; idx < (prb_idx + 1) * dl_prb_size; ++idx)
                {
                    string_buffer_index += snprintf(buffer + string_buffer_index, MAX_PRINT_LOG_LENGTH - string_buffer_index, "%s",  (((unsigned char*)(&slot_buf[buffer_index]))[idx] == ((unsigned char*)(mbuf_payload))[idx]) ? "__" : "^^");
                }
                re_cons("{}", buffer);
            }
#endif
            }
            pbch_object.atomic_received_prbs[launch_pattern_slot][cell_index] += numPrb;
        }
    }

    int expected_num_prbs = 0;
    if (cell_configs[cell_index].dl_comp_meth != aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION)
    {
        expected_num_prbs = dl_tv_info.numPrb * std::min(static_cast<int>(cell_configs[cell_index].eAxC_DL.size()), static_cast<int>(dl_tv_info.numFlows));
    }
    else
    {
        expected_num_prbs = dl_tv_info.modCompNumPrb;
    }

    pbch_object.mtx[cell_index].lock();
    if(pbch_object.atomic_received_prbs[launch_pattern_slot][cell_index].load() >= expected_num_prbs)
    {
        pbch_object.atomic_received_prbs[launch_pattern_slot][cell_index].store(0);
        complete = true;
    }
    pbch_object.mtx[cell_index].unlock();

    if(complete)
    {
        curr_pbch_time = get_ns();
        if(pbch_object.invalid_flag[cell_index][launch_pattern_slot])
        {
            re_cons("PBCH Cell {} 3GPP slot {} F{} S{} S{} validation error {:4.2f} us from prev PBCH",  cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId, (double)(curr_pbch_time - prev_pbch_time)/ NS_X_US);
            ++pbch_object.error_slot_counters[cell_index];
        }
        else
        {
            re_info("PBCH Complete Cell {} 3GPP slot {} F{} S{} S{} {:4.2f} us from prev PBCH", cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId, (double)(curr_pbch_time - prev_pbch_time)/ NS_X_US);
            ++pbch_object.good_slot_counters[cell_index];
            ++pbch_object.throughput_slot_counters[cell_index];
        }
        prev_pbch_time = curr_pbch_time;
        ++pbch_object.total_slot_counters[cell_index];
        pbch_object.invalid_flag[cell_index][launch_pattern_slot] = false;
        if(pbch_object.init_slot_counters[cell_index].load() <= pbch_object.total_slot_counters[cell_index].load())
        {
            pbch_object.initialization_phase[cell_index].store(0);
        }
        return 1;
    }
    return 0;
}
