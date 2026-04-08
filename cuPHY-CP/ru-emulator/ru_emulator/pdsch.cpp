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

int RU_Emulator::validate_pdsch(uint8_t cell_index, const struct oran_packet_header_info& header_info, void* buffer)
{
    uint8_t* mbuf_payload;
    uint16_t payload_len;
    uint32_t tv_index;
    uint32_t flow_offset;
    uint32_t symbol_offset;
    uint32_t prb_offset;
    uint32_t buffer_index;
    int comp;
    uint8_t startDataSym;
    uint8_t numDataSym;
    unsigned char* slot_buf;
    uint16_t startPrb = header_info.startPrb;
    uint16_t numPrb = header_info.numPrb;
    uint8_t flowId = header_info.flowValue;
    uint8_t flow_index = header_info.flow_index;
    uint8_t launch_pattern_slot = header_info.launch_pattern_slot;
    uint8_t symbolId = header_info.symbolId;
    uint16_t sectionId = header_info.sectionId;
    uint16_t startPrbOffset = 0;
    const struct fssId& fss = header_info.fss;
    struct dl_tv_object* tv_object = &pdsch_object;
    bool complete = false;
    int dl_prb_size = cell_configs[cell_index].dl_prb_size; // used to for the received buffer; for the TV we'll use tv_dl_prb_size
    if(pdsch_object.initialization_phase[cell_index].load() == RE_ENABLED)
    {
        tv_index = pdsch_object.init_launch_pattern[launch_pattern_slot][cell_index];
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

    struct dl_tv_info& dl_tv_info = pdsch_object.tv_info[tv_index];

    if(flow_index >= dl_tv_info.numFlows)
    {
        return 0;
    }

    if (cell_configs[cell_index].dl_comp_meth == aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION)
    {
        auto &mod_comp_data = pdsch_object.mod_comp_data[tv_index];
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
                    re_cons("PDSCH ERROR udIqWidth mismatch: Cell {} 3GPP slot {} F{} S{} S{} Flow {} symbolId {} sectionId {} "
                            "startPrb {} numPrb {} actual {} expected {}",
                            cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId,
                            flow_index, symbolId, sectionId, startPrb, numPrb, actual_iq_width, payload_params[2]);
                    pdsch_object.invalid_flag[cell_index][launch_pattern_slot] = true;
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
                auto byte_buffer = static_cast<uint8_t *>(buffer);
                mbuf_payload = &byte_buffer[ORAN_IQ_COMPRESSED_SECTION_OVERHEAD];
                comp = memcmp(mbuf_payload, &slot_buf[buffer_index], payload_len);
            }
            if (comp != 0)
            {
                re_cons("PDSCH ERROR Invalid byte-match: Cell {} 3GPP slot {} F{} S{} S{} Flow {} symbolId {} sectionId {} startPrb {} numPrb {} buffer_index {}",
                        cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId, flow_index, symbolId, sectionId, startPrb, numPrb, buffer_index);
                re_cons("PDSCH modComp info: msg_idx {} startPrb {} numPrb {} iqWidth {}", mod_comp_msg_idx + 1, payload_params[0], payload_params[1], payload_params[2]);
                pdsch_object.invalid_flag[cell_index][launch_pattern_slot] = true;

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

            pdsch_object.atomic_received_prbs[launch_pattern_slot][cell_index] += numPrb;

            re_dbg("F{} S{} S{} Flow {} symbolId {} atomic_received_prbs {} dl_tv_info.modCompNumPrb {}",
                   fss.frameId, fss.subframeId, fss.slotId, flow_index, symbolId,
                   pdsch_object.atomic_received_prbs[launch_pattern_slot][cell_index].load(), dl_tv_info.modCompNumPrb);
            {
                const std::lock_guard<aerial_fh::FHMutex> lock(pdsch_object.mtx[cell_index]);
                if (pdsch_object.atomic_received_prbs[launch_pattern_slot][cell_index].load() >= dl_tv_info.modCompNumPrb)
                {
                    pdsch_object.atomic_received_prbs[launch_pattern_slot][cell_index].store(0);
                    complete = true;
                }
            }
        }
    }
    else
    {
        // slot_buf = (unsigned char *)pdsch_object.qams_fp16[tv_index].data.get();
        // printf("cell %d compression bits %d\n", cell_index, cell_configs[cell_index].dl_bit_width);

        /* Select which slot_buf buffer (TV dataset) to use for validation. When approx. validation is enabled (expected when precoding is enabled and we don't use identity
           precoding matrices), we use the uncompressed dataset directly for the TV. */
        bool approx_validation_w_compression = (opt_dl_approx_validation == RE_ENABLED);
        int qams_buffer_index = approx_validation_w_compression ? 16 : cell_configs[cell_index].dl_bit_width;
        if (cell_configs[cell_index].dl_comp_meth == aerial_fh::UserDataCompressionMethod::NO_COMPRESSION)
        {
            qams_buffer_index = FIXED_POINT_16_BITS;
        }
        slot_buf = (unsigned char *)pdsch_object.qams[qams_buffer_index][tv_index].data.get();
        int tv_dl_prb_size = approx_validation_w_compression ? PRB_SIZE_16F : dl_prb_size;

        // Compute beta_dl using the PRB BWP size for the current cell, if fix_beta_dl isn't set
        if (cell_configs[cell_index].beta_dl == BETA_DL_NOT_SET)
        {
            cell_configs[cell_index].beta_dl = sqrt(cell_configs[cell_index].numerator / (SUBCARRIERS_PER_PRB * dl_tv_info.nPrbDlBwp));
        }
        // re_cons("new_beta_dl {}, nPrbDlBwp {}, beta_dl {}", new_beta_dl, dl_tv_info.nPrbDlBwp, cell_configs[cell_index].beta_dl);

        for (int i = 0; i < dl_tv_info.combined_pdu_infos.size(); ++i)
        {
            startPrb = header_info.startPrb;
            numPrb = header_info.numPrb;
            flowId = header_info.flowValue;
            launch_pattern_slot = header_info.launch_pattern_slot;
            symbolId = header_info.symbolId;
            flow_index = header_info.flow_index;
            auto &pdu_info = dl_tv_info.combined_pdu_infos[i];
            startDataSym = pdu_info.startDataSym;
            numDataSym = pdu_info.numDataSym;

            // The startDataSym and numDataSym specify the first symbol of a UE group allocation and total
            // symbols in that allocation respectively. Thy include DMRS symbols too.
            if (symbolId < startDataSym || symbolId >= startDataSym + numDataSym)
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
                auto it = std::find(pdu_info.flow_indices.begin(), pdu_info.flow_indices.end(), flow_index);
                if (it == pdu_info.flow_indices.end())
                {
                    continue;
                }

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

                re_dbg("PDU {} Flow {} symbolId {} startPrb {} numPrb {} startPrbOffset {}", i, flow_index, symbolId, startPrb, numPrb, startPrbOffset);

                if (find(pdu_info.flow_indices.begin(), pdu_info.flow_indices.end(), flow_index) == pdu_info.flow_indices.end())
                {
                    continue;
                }

                flow_offset = flow_index * (SLOT_NUM_SYMS * cell_configs[cell_index].dlGridSize);
                symbol_offset = (symbolId) * (cell_configs[cell_index].dlGridSize);
                prb_offset = (startPrb);

                buffer_index = tv_dl_prb_size * (flow_offset + symbol_offset + prb_offset);
                comp = 0;
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
                        // comp = compare_approx_buffer(mbuf_payload, &slot_buf[buffer_index], payload_len);
                        // if ((flow_index == 0) && (symbolId == 3) && (startPrb == 20))
                        comp = decompress_and_compare_approx_buffer((uint8_t *)mbuf_payload, &slot_buf[buffer_index], payload_len, cell_configs[cell_index].dl_bit_width, cell_configs[cell_index].beta_dl, flow_index, symbolId, startPrb);
                    }
                    else
                    {
                        if (header_info.rb)
                        {
                            uint8_t *dst = mbuf_payload, *src = &slot_buf[buffer_index];
                            for (int i = 0; i < numPrb; i++)
                            {
                                comp |= memcmp(dst, src, dl_prb_size);
                                if (comp)
                                    break;
                                dst += dl_prb_size;
                                src += (dl_prb_size << 1);
                            }
                        }
                        else
                        {
                            comp = memcmp(mbuf_payload, &slot_buf[buffer_index], payload_len);
                        }
                        // comp = 0;
                    }
                }
#if 0
	    // PRINT 5 REs at symbol 3 (0-indexing) starting at PRB 20 from TV
	    if ((symbolId == 3) && (flow_index == 0) && (startPrb == 20)) {
                 char buffer[MAX_PRINT_LOG_LENGTH];
                 printf("----> byte-match: Cell %d 3GPP slot %d F%d S%d S%d Flow %d symbolId %d startPrb %d numPrb %d buffer_index %d flow_offset %d symbol_offset %d prb_offset %d\n",
                     cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId, flow_index, symbolId, startPrb, numPrb, buffer_index, flow_offset, symbol_offset, prb_offset);
                 snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%02X",  ((unsigned char*)(&slot_buf[buffer_index]))[0]);
	    }
#endif

                if (comp != 0)
                {
                    re_info("ERROR Invalid byte-match: Cell {} 3GPP slot {} F{} S{} S{} Flow {} symbolId {} startPrb {} numPrb {} buffer_index {} flow_offset {} symbol_offset {} prb_offset {}",
                            cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId, flow_index, symbolId, startPrb, numPrb, buffer_index, flow_offset, symbol_offset, prb_offset);
                    pdsch_object.invalid_flag[cell_index][launch_pattern_slot] = true;
#if 0
                if (opt_dl_approx_validation != RE_ENABLED) // If we do approx. validation one buffer is uncompressed so prints won't be aligned.
                for(int prb_idx = 0; prb_idx < numPrb; ++prb_idx)
                {
                    char buffer[MAX_PRINT_LOG_LENGTH];
                    int string_buffer_index = 0;
                    int rx_prb_idx = prb_idx;
                    int tv_prb_idx = (header_info.rb ? (prb_idx << 1) : prb_idx);
                    int rx_prb_buf_idx = rx_prb_idx * dl_prb_size;
                    int tv_prb_buf_idx = tv_prb_idx * dl_prb_size;

                    string_buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "F%dS%dS%d PDSCH Cell %d Flow %d symbolId %d Prb %d RX: ", fss.frameId, fss.subframeId, fss.slotId, cell_index, flow_index, symbolId, tv_prb_idx + startPrb);
                    for(int idx = 0; idx < dl_prb_size; ++idx)
                    {
                        string_buffer_index += snprintf(buffer + string_buffer_index, MAX_PRINT_LOG_LENGTH - string_buffer_index, "%02X", ((unsigned char*)(mbuf_payload))[rx_prb_buf_idx + idx]);
                    }
                    re_cons("{}", buffer);
                    string_buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "F%dS%dS%d PDSCH Cell %d Flow %d symbolId %d Prb %d TV: ", fss.frameId, fss.subframeId, fss.slotId, cell_index, flow_index, symbolId, tv_prb_idx + startPrb);

                    for(int idx = 0; idx < dl_prb_size; ++idx)
                    {
                        string_buffer_index += snprintf(buffer + string_buffer_index, MAX_PRINT_LOG_LENGTH - string_buffer_index, "%02X",  ((unsigned char*)(&slot_buf[buffer_index]))[tv_prb_buf_idx + idx]);
                    }
                    re_cons("{}", buffer);
                    string_buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "F%dS%dS%d PDSCH Cell %d Flow %d symbolId %d Prb %d   : ", fss.frameId, fss.subframeId, fss.slotId, cell_index, flow_index, symbolId, tv_prb_idx + startPrb);

                    for(int idx = 0; idx < dl_prb_size; ++idx)
                    {
                        string_buffer_index += snprintf(buffer + string_buffer_index, MAX_PRINT_LOG_LENGTH - string_buffer_index, "%s",  (((unsigned char*)(&slot_buf[buffer_index]))[tv_prb_buf_idx + idx] == ((unsigned char*)(mbuf_payload))[rx_prb_buf_idx + idx]) ? "__" : "^^");
                    }
                    re_cons("{}", buffer);
                }
#endif
                }

                pdsch_object.atomic_received_prbs[launch_pattern_slot][cell_index] += numPrb;
            }

            re_dbg("F{} S{} S{} Flow {} symbolId {} atomic_received_prbs {} dl_tv_info.numPrb {}",
                   fss.frameId, fss.subframeId, fss.slotId, flow_index, symbolId,
                   pdsch_object.atomic_received_prbs[launch_pattern_slot][cell_index].load(), dl_tv_info.numPrb);
            {
                const std::lock_guard<aerial_fh::FHMutex> lock(pdsch_object.mtx[cell_index]);
                if (pdsch_object.atomic_received_prbs[launch_pattern_slot][cell_index].load() >= dl_tv_info.numPrb)
                {
                    pdsch_object.atomic_received_prbs[launch_pattern_slot][cell_index].store(0);
                    complete = true;
                }
            }
            if (complete)
            {
                break;
            }
        }
    }
    if (complete)
    {
        if (pdsch_object.invalid_flag[cell_index][launch_pattern_slot])
        {
            re_cons("PDSCH Complete Cell {} 3GPP slot {} F{} S{} S{} Payload Validation {}",
                    cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId,
                    pdsch_object.invalid_flag[cell_index][launch_pattern_slot] ? "ERROR" : "OK");
            ++pdsch_object.error_slot_counters[cell_index];
        }
        else
        {
            re_info("PDSCH Complete Cell {} 3GPP slot {} F{} S{} S{} OK", cell_index, launch_pattern_slot, fss.frameId, fss.subframeId, fss.slotId);
            pdsch_object.throughput_counters[cell_index] += pdsch_object.tv_info[tv_index].tb_size;
            ++pdsch_object.throughput_slot_counters[cell_index];
            ++pdsch_object.good_slot_counters[cell_index];
        }

        if (pdsch_object.tv_info[tv_index].hasZPCsirsPdu)
        {
// Disable ZP CSI-RS slot counters according to nvbug 4041331
#if 0
                ++csirs_object.throughput_slot_counters[cell_index];
                ++csirs_object.good_slot_counters[cell_index];
                ++csirs_object.total_slot_counters[cell_index];
#endif
        }
        else if (cell_configs[cell_index].dl_comp_meth != aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION && pdsch_object.tv_info[tv_index].numOverlappingCsirs)
        {
            if (pdsch_object.tv_info[tv_index].fullyOverlappingCsirs)
            {
                ++csirs_object.throughput_slot_counters[cell_index];
                ++csirs_object.total_slot_counters[cell_index];
                if (pdsch_object.invalid_flag[cell_index][launch_pattern_slot])
                {
                    ++csirs_object.error_slot_counters[cell_index];
                }
                else
                {
                    ++csirs_object.good_slot_counters[cell_index];
                }
            }
            else
            {
                csirs_object.fss_atomic_received_res[cell_index][fss.frameId][fss.subframeId * ORAN_MAX_SLOT_ID + fss.slotId] += pdsch_object.tv_info[tv_index].numOverlappingCsirs * dl_tv_info.numFlows;
                if (pdsch_object.invalid_flag[cell_index][launch_pattern_slot])
                {
                    csirs_object.invalid_flag[cell_index][launch_pattern_slot] = true;
                }
            }
            re_dbg("PDSCH added Received REs {} current total {}", pdsch_object.tv_info[tv_index].numOverlappingCsirs * dl_tv_info.numFlows, csirs_object.fss_atomic_received_res[cell_index][fss.frameId][fss.subframeId * ORAN_MAX_SLOT_ID + fss.slotId].load());
        }

        ++pdsch_object.total_slot_counters[cell_index];
        pdsch_object.invalid_flag[cell_index][launch_pattern_slot] = false;
        complete = false;
        if (pdsch_object.init_slot_counters[cell_index].load() <= pdsch_object.total_slot_counters[cell_index].load())
        {
            pdsch_object.initialization_phase[cell_index].store(0);
        }
        return 1;
    }
    return 0;
}
