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

bool RU_Emulator::does_slot_have_channel_for_cell(uint8_t cell_index, struct dl_tv_object* tv_object, uint8_t launch_pattern_slot)
{
    if(tv_object->launch_pattern[launch_pattern_slot].find(cell_index) != tv_object->launch_pattern[launch_pattern_slot].end())
    {
        return true;
    }
    return false;
}

bool RU_Emulator::u_plane_channel_type_checking(const struct oran_packet_header_info& header_info, uint16_t cell_index, struct dl_tv_object& tv_obj)
{
    if (header_info.launch_pattern_slot < tv_obj.launch_pattern.size() && tv_obj.launch_pattern[header_info.launch_pattern_slot].find(cell_index) != tv_obj.launch_pattern[header_info.launch_pattern_slot].end())
    {
        int tv_idx = tv_obj.launch_pattern[header_info.launch_pattern_slot][cell_index];
        if (tv_idx < tv_obj.tv_info.size())
        {
            auto& prb_map  = tv_obj.tv_info[tv_idx].prb_map;
            return prb_map[header_info.symbolId][header_info.startPrb];
        }
    }
    return false;
}

int RU_Emulator::validate_csirs(uint8_t cell_index, const struct oran_packet_header_info& header_info, void* buffer)
{
    uint32_t tv_index;
    uint32_t buffer_index;
    uint8_t reSize = sizeof(uint32_t);
    uint16_t startPrb = header_info.startPrb;
    uint16_t numPrb = header_info.numPrb;
    uint8_t flowId = header_info.flowValue;
    uint8_t flow_index = header_info.flow_index;
    uint8_t launch_pattern_slot = header_info.launch_pattern_slot;
    uint8_t symbolId = header_info.symbolId;
    uint16_t sectionId = header_info.sectionId;
    const struct fssId& fss = header_info.fss;
    int comp;
    uint32_t numConsecutiveREs;

    uint16_t reOffset;
    uint16_t startPrbOffset;
    uint16_t prbOffset;
    uint32_t flow_offset;
    uint32_t symbol_offset;
    uint32_t prb_offset;
    uint16_t numREsInPrb;
    uint16_t reCount = 0;
    uint32_t prbIndex;
    uint16_t prbSize = cell_configs[cell_index].dl_prb_size;
    uint8_t* mbuf_payload;
    uint16_t payload_len;
    unsigned char* slot_buf;
    struct dl_tv_object* tv_object = &csirs_object;
    bool complete = false;

    int dl_prb_size = cell_configs[cell_index].dl_prb_size; // used to for the received buffer; for the TV we'll use tv_dl_prb_size
    if(!does_slot_have_channel_for_cell(cell_index, tv_object, launch_pattern_slot))
    {
        re_info("Slot {} does not have CSI_RS channel", launch_pattern_slot);
        return 0;
    }
    tv_index = tv_object->launch_pattern[launch_pattern_slot][cell_index];
    struct dl_tv_info& tv_info = tv_object->tv_info[tv_index];

    if (cell_configs[cell_index].dl_comp_meth == aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION)
    {
        auto &mod_comp_data = csirs_object.mod_comp_data[tv_index];
        auto &hdr = mod_comp_data.mod_comp_header;
        int mod_comp_msg_idx = -1;
        for (int port_idx = flow_index; mod_comp_msg_idx == -1 && port_idx < tv_object->tv_info[tv_index].csirsMaxPortNum; port_idx += cell_configs[cell_index].num_dl_flows)
        {
            mod_comp_msg_idx = find_modcomp_msg_id(mod_comp_data, symbolId, port_idx, startPrb, numPrb);
        }
        // if (hdr.find(symbolId) != hdr.end() && hdr[symbolId].find(flow_index) != hdr[symbolId].end() && c_plane_channel_type_checking(c_plane_info, cell_index, pdsch_object))
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
                    re_cons("CSI_RS ERROR udIqWidth mismatch: Cell {} 3GPP slot {} F{} S{} S{} Flow {} symbolId {} sectionId {} "
                            "startPrb {} numPrb {} actual {} expected {}",
                            cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId,
                            flow_index, symbolId, sectionId, startPrb, numPrb, actual_iq_width, payload_params[2]);
                    csirs_object.invalid_flag[cell_index][launch_pattern_slot] = true;
                }
            }

            if (!payload_params[3])
            {
                dl_prb_size = payload_params[2] * 3;
                buffer_index = dl_prb_size * (startPrb - payload_params[0]);
                payload_len = numPrb * dl_prb_size;

                if (opt_dlc_tb)
                {
                    comp = 0; // Skip IQ comparison when dlc_tb is enabled
                }
                else
                {
                    auto byte_buffer = static_cast<uint8_t *>(buffer);
                    mbuf_payload = &byte_buffer[ORAN_IQ_COMPRESSED_SECTION_OVERHEAD];
                    comp = memcmp(mbuf_payload, &slot_buf[buffer_index], payload_len);
                }
            }
            else
            {//csirs section of multiplexed pdsch+csirs, iq validation will be performed with pdsch
                comp = 0;
            }

            if (comp != 0)
            {
                re_cons("CSI_RS ERROR Invalid byte-match: Cell {} 3GPP slot {} F{} S{} S{} Flow {} symbolId {} sectionId {} startPrb {} numPrb {} buffer_index {}",
                        cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId, flow_index, symbolId, sectionId, startPrb, numPrb, buffer_index);
                re_cons("CSI_RS modComp info: msg_idx {} startPrb {} numPrb {} iqWidth {} skip_validation {}", mod_comp_msg_idx + 1, payload_params[0], payload_params[1], payload_params[2],  payload_params[3]);
                csirs_object.invalid_flag[cell_index][launch_pattern_slot] = true;

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
            //re_dbg("Received REs {}", tv_object->fss_atomic_received_res[cell_index][fss.frameId][fss.subframeId * ORAN_MAX_SLOT_ID + fss.slotId].load());
            tv_object->mtx[cell_index].lock();
            // if (tv_object->atomic_received_prbs[launch_pattern_slot][cell_index].load() >= tv_info.numPrb * std::min(tv_info.numFlows, (uint8_t)cell_configs[cell_index].num_dl_flows) / tv_info.numFlows && tv_info.csirsNumREs != 0)
            if (tv_object->atomic_received_prbs[launch_pattern_slot][cell_index].load() >= tv_info.modCompNumPrb)
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
            cell_configs[cell_index].beta_dl = sqrt(cell_configs[cell_index].numerator / (SUBCARRIERS_PER_PRB * tv_info.nPrbDlBwp));
        }

        if (csi_rs_optimized_validation)
        {
            if (flow_index < tv_info.numFlowsArray[symbolId])
            {
                for (const auto &pdu_info : tv_info.pdu_infos)
                {
                    if (symbolId < pdu_info.startSym || symbolId >= pdu_info.startSym + pdu_info.numSym)
                    {
                        continue;
                    }
                    else if (startPrb > pdu_info.numPrb + pdu_info.startPrb)
                    {
                        continue;
                    }
                    else if (startPrb + numPrb < pdu_info.startPrb)
                    {
                        continue;
                    }
                    else
                    {
                        startPrbOffset = 0;
                        if (startPrb + numPrb > pdu_info.numPrb + pdu_info.startPrb)
                        {
                            numPrb -= (startPrb + numPrb) - (pdu_info.numPrb + pdu_info.startPrb);
                        }
                        if (startPrb < pdu_info.startPrb)
                        {
                            startPrbOffset = pdu_info.startPrb - startPrb;
                            numPrb -= startPrbOffset;
                            startPrb = pdu_info.startPrb;
                        }

                        payload_len = numPrb * dl_prb_size;
                        flow_offset = flow_index * (SLOT_NUM_SYMS * cell_configs[cell_index].dlGridSize);
                        symbol_offset = (symbolId) * (cell_configs[cell_index].dlGridSize);
                        prb_offset = (startPrb);

                        buffer_index = tv_dl_prb_size * (flow_offset + symbol_offset + prb_offset);

                        if (opt_dlc_tb)
                        {
                            comp = 0; // Skip IQ comparison when dlc_tb is enabled
                        }
                        else
                        {
                            auto byte_buffer = static_cast<uint8_t *>(buffer);
                            mbuf_payload = &byte_buffer[ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD + startPrbOffset * dl_prb_size];

                            if (opt_dl_approx_validation == RE_ENABLED)
                            {
                                comp = decompress_and_compare_approx_buffer((uint8_t *)mbuf_payload, &slot_buf[buffer_index], payload_len, cell_configs[cell_index].dl_bit_width, cell_configs[cell_index].beta_dl, flow_index, symbolId, startPrb);
                                // comp = compare_approx_buffer(mbuf_payload, &slot_buf[buffer_index], payload_len);
                            }
                            else
                            {
                                comp = memcmp(mbuf_payload, &slot_buf[buffer_index], payload_len);
                            }
                        }
                        if (comp != 0)
                        {
                            // re_info("ERROR Invalid byte-match: Cell {} 3GPP slot {} F{} S{} S{} Flow {} symbolId {} startPrb {} numPrb {} buffer_index {} flow_offset {} symbol_offset {} prb_offset {}",
                            //     cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId, flow_index, symbolId, startPrb, numPrb, buffer_index, flow_offset, symbol_offset, prb_offset);
                            tv_object->invalid_flag[cell_index][launch_pattern_slot] = true;
                        }
                        tv_object->atomic_received_prbs[launch_pattern_slot][cell_index] += numPrb;
                    }
                }
            }
            else if (opt_enable_beam_forming)
            {
#if 0
            if (u_plane_channel_type_checking(header_info, cell_index, csirs_object))
            {
                re_cons("CSI-RS UP pkt is not supposed to be on this flow!");
                sleep(1);
                do_throw(sb() << "CSI-RS UP pkt is not supposed to be on this flow!\n");
            }
#endif
            }
            tv_object->mtx[cell_index].lock();
            // if (tv_object->atomic_received_prbs[launch_pattern_slot][cell_index].load() >= tv_info.numPrb * std::min(tv_info.numFlows, (uint8_t)cell_configs[cell_index].num_dl_flows) / tv_info.numFlows && tv_info.csirsNumREs != 0)
            if (tv_object->atomic_received_prbs[launch_pattern_slot][cell_index].load() >= tv_info.numPrb && tv_info.csirsNumREs != 0)
            {
                tv_object->atomic_received_prbs[launch_pattern_slot][cell_index].store(0);
                complete = true;
            }
            tv_object->mtx[cell_index].unlock();
        }
        else
        {
            // All antennas have the same CSI_RS REs
            uint32_t startReIndex = symbolId * cell_configs[cell_index].dlGridSize * PRB_NUM_RE + startPrb * PRB_NUM_RE;
            uint32_t numRE = header_info.numPrb * PRB_NUM_RE;
            if (header_info.rb)
            {
                numRE <<= 1;
            }
            payload_len = dl_prb_size; // checking one PRB at a time
            if (flow_index < tv_info.numFlowsArray[symbolId])
            {
                for (int i = 0; i < tv_info.csirsREsToValidate.size(); ++i)
                {
                    uint32_t reIndex = tv_info.csirsREsToValidate[i][0];
                    if (reIndex < startReIndex || reIndex >= startReIndex + numRE)
                    {
                        continue;
                    }
                    if (reIndex >= startReIndex && reIndex < startReIndex + numRE)
                    {
                        reOffset = reIndex - startReIndex;
                        reCount = 1;

                        if (opt_dlc_tb)
                        {
                            comp = 0; // Skip IQ comparison when dlc_tb is enabled
                        }
                        else
                        {
                            auto byte_buffer = static_cast<uint8_t *>(buffer);
                            mbuf_payload = &byte_buffer[ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD];

                            if (cell_configs[cell_index].dl_bit_width == BFP_NO_COMPRESSION)
                            {
                                if (header_info.rb)
                                {
                                    prbOffset = (reOffset / PRB_NUM_RE) >> 1;
                                    reOffset = prbOffset * PRB_NUM_RE + (reOffset % PRB_NUM_RE);
                                }

                                mbuf_payload = &(mbuf_payload[reOffset * reSize]);

                                buffer_index = flow_index * ORAN_ALL_SYMBOLS * cell_configs[cell_index].dlGridSize * PRB_NUM_RE * reSize + reIndex * reSize;

                                if (opt_dl_approx_validation == RE_ENABLED)
                                {
                                    // Comparing a __half2, i.e., an RE,  at a time
                                    comp = compare_approx_buffer(mbuf_payload, &slot_buf[buffer_index], reSize);
                                }
                                else
                                {
                                    comp = memcmp(mbuf_payload, &slot_buf[buffer_index], reSize);
                                }
                            }
                            else if (cell_configs[cell_index].dl_bit_width == BFP_COMPRESSION_14_BITS || cell_configs[cell_index].ul_bit_width == BFP_COMPRESSION_9_BITS)
                            {
                                prbOffset = reOffset / PRB_NUM_RE;
                                reCount = tv_info.csirsREsToValidate[i][1];
                                prbIndex = reIndex / PRB_NUM_RE;
                                mbuf_payload = &(mbuf_payload[(header_info.rb ? (prbOffset >> 1) : prbOffset) * prbSize]);

                                buffer_index = (flow_index * ORAN_ALL_SYMBOLS * cell_configs[cell_index].dlGridSize + prbIndex) * tv_dl_prb_size;

                                if (opt_dl_approx_validation == RE_ENABLED)
                                {
                                    comp = decompress_and_compare_approx_buffer((uint8_t *)mbuf_payload, &slot_buf[buffer_index], payload_len, cell_configs[cell_index].dl_bit_width, cell_configs[cell_index].beta_dl, flow_index, symbolId, startPrb);
                                }
                                else
                                {
                                    comp = memcmp(mbuf_payload, &slot_buf[buffer_index], prbSize);
                                }
                            }
                            else
                            {
                                do_throw(sb() << "Compression other than 9,14,16 not supported!\n");
                            }
                        }

                        if (comp != 0)
                        {
                            re_info("ERROR Invalid byte-match: Cell {} 3GPP slot {} F{} S{} S{} Flow {} symbolId {} startPrb {} numPrb {} buffer_index {}",
                                    cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId, flow_index, symbolId, startPrb, numPrb, buffer_index);
                            tv_object->invalid_flag[cell_index][launch_pattern_slot] = true;
#if 0
                    if(cell_configs[cell_index].dl_bit_width == BFP_NO_COMPRESSION)
                    {
                        for(int re_i = 0; re_i < numConsecutiveREs; ++re_i)
                        {
                            re_cons("Symbol {} RE {} RX: {:02X}{:02X}{:02X}{:02X}", symbolId, reIndex + re_i, mbuf_payload[0], mbuf_payload[1], mbuf_payload[2], mbuf_payload[3]);
                            re_cons("Symbol {} RE {} TV: {:02X}{:02X}{:02X}{:02X}", symbolId, reIndex + re_i, slot_buf[buffer_index], slot_buf[buffer_index+1],slot_buf[buffer_index+2], slot_buf[buffer_index+3]);
                        }
                    }
                    else
                    {
                        if (opt_dl_approx_validation != RE_ENABLED) // If we do approx. validation one buffer is uncompressed so prints won't be aligned.
                        {
                            printf("\nFlow %d symbolId %d Prb %d RX: ", flow_index, symbolId, prbOffset + startPrb);
                            for(int idx = 0; idx < prbSize; ++idx)
                            {
                                printf("%02X", ((unsigned char*)(mbuf_payload))[idx]);
                            }
                            printf("\nFlow %d symbolId %d Prb %d TV: ", flow_index, symbolId, prbOffset + startPrb);
                            for(int idx = 0; idx < prbSize; ++idx)
                            {
                                printf("%02X", ((unsigned char*)(&slot_buf[buffer_index]))[idx]);
                            }
                            printf("\nFlow %d symbolId %d Prb %d   : ", flow_index, symbolId, prbOffset + startPrb);
                            for(int idx = 0; idx < prbSize; ++idx)
                            {
                                printf("%s", (((unsigned char*)(&slot_buf[buffer_index]))[idx] == ((unsigned char*)(mbuf_payload))[idx]) ? "__" : "^^");
                            }
                            printf("\n");
                        }
                    }
#endif
                        }
                        tv_object->fss_atomic_received_res[cell_index][fss.frameId][fss.subframeId * ORAN_MAX_SLOT_ID + fss.slotId] += reCount;
                        i += reCount - 1;
                    }
                }
            }
            else if (opt_enable_beam_forming)
            {
#if 0
            if (u_plane_channel_type_checking(header_info, cell_index, csirs_object))
            {
                re_cons("CSI-RS UP pkt is not supposed to be on this flow!");
                sleep(1);
                do_throw(sb() << "CSI-RS UP pkt is not supposed to be on this flow!\n");
            }
#endif
            }
            re_dbg("Received REs {}", tv_object->fss_atomic_received_res[cell_index][fss.frameId][fss.subframeId * ORAN_MAX_SLOT_ID + fss.slotId].load());
            tv_object->mtx[cell_index].lock();
            // if(tv_object->fss_atomic_received_res[cell_index][fss.frameId][fss.subframeId * ORAN_MAX_SLOT_ID + fss.slotId].load() >= (tv_info.csirsNumREs + tv_info.csirsNumREsSkipped) * std::min(tv_info.numFlows, (uint8_t)cell_configs[cell_index].num_dl_flows) && tv_info.csirsNumREs != 0)
            if (tv_object->fss_atomic_received_res[cell_index][fss.frameId][fss.subframeId * ORAN_MAX_SLOT_ID + fss.slotId].load() >= tv_info.csirsExpectedNumREs && tv_info.csirsNumREs != 0)
            {
                tv_object->fss_atomic_received_res[cell_index][fss.frameId][fss.subframeId * ORAN_MAX_SLOT_ID + fss.slotId].store(0);
                complete = true;
            }
            tv_object->mtx[cell_index].unlock();
        }
    }

    if(complete)
    {
        if(tv_object->invalid_flag[cell_index][launch_pattern_slot])
        {
            re_warn("CSI-RS Complete Cell {} 3GPP slot {} F{} S{} S{} Validation {}",
                cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId,
                tv_object->invalid_flag[cell_index][launch_pattern_slot]? "ERROR" : "OK");
            ++tv_object->error_slot_counters[cell_index];
        }
        else
        {
            re_info("CSI-RS Complete Cell {} 3GPP slot {} F{} S{} S{} OK", cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId);
            ++tv_object->throughput_slot_counters[cell_index];
            ++tv_object->good_slot_counters[cell_index];
        }
        ++tv_object->total_slot_counters[cell_index];
        tv_object->invalid_flag[cell_index][launch_pattern_slot] = false;
        return 1;
    }
    return 0;
}
