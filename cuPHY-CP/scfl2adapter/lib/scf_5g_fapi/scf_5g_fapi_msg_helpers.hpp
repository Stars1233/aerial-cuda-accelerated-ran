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

#include "nv_ipc.h"
#include "scf_5g_fapi.h"

#include "aerial/casts/casts.hpp"

#include <algorithm>
#include <tuple>

namespace scf_5g_fapi {

template <typename T>
static scf_fapi_body_header_t *add_scf_fapi_hdr(nv_ipc_msg_t& msg, int msg_id, int cell_id, bool data)
{
    scf_fapi_header_t *hdr;
    if (data) {
        hdr = reinterpret_cast<scf_fapi_header_t*>(msg.data_buf);
    } 
    else {
        hdr = reinterpret_cast<scf_fapi_header_t*>(msg.msg_buf);
    }

    hdr->message_count     = 1;
    hdr->handle_id         = cell_id;

    auto *body = aerial::casts::assume_cast<scf_fapi_body_header_t>(hdr->payload);
    body->type_id          = msg_id;
    body->length           = static_cast<uint32_t>(sizeof(T) - sizeof(scf_fapi_body_header_t));

    msg.msg_id = msg_id;
    msg.cell_id = cell_id;
    msg.msg_len = sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t) + body->length;
    msg.data_len = 0;

    return body;
}

static void PrintFAPIMsg(nv_ipc_msg_t& msg, bool dump = false, bool data = false)
{
    std::unordered_map<uint16_t, std::string> id2str = 
    {
        {0x00, "SCF_FAPI_PARAM_REQUEST"},
        {0x01, "SCF_FAPI_PARAM_RESPONSE"},
        {0x02, "SCF_FAPI_CONFIG_REQUEST"},
        {0x03, "SCF_FAPI_CONFIG_RESPONSE"},
        {0x04, "SCF_FAPI_START_REQUEST"},
        {0x05, "SCF_FAPI_STOP_REQUEST"},
        {0x06, "SCF_FAPI_STOP_INDICATION"},
        {0x07, "SCF_FAPI_ERROR_INDICATION"},
        {0x08, "SCF_FAPI_VS_DLBFW_CVI_REQUEST"},
        {0x09, "SCF_FAPI_VS_ULBFW_CVI_REQUEST"},
        {0x80, "SCF_FAPI_DL_TTI_REQUEST"},
        {0x81, "SCF_FAPI_UL_TTI_REQUEST"},
        {0x82, "SCF_FAPI_SLOT_INDICATION"},
        {0x83, "SCF_FAPI_UL_DCI_REQUEST"},
        {0x84, "SCF_FAPI_TX_DATA_REQUEST"},
        {0x85, "SCF_FAPI_RX_DATA_INDICATION"},
        {0x86, "SCF_FAPI_CRC_INDICATION"},
        {0x87, "SCF_FAPI_UCI_INDICATION"},
        {0x88, "SCF_FAPI_SRS_INDICATION"},
        {0x89, "SCF_FAPI_RACH_INDICATION"},
        {0x8b, "SCF_FAPI_RX_PE_NOISE_VARIANCE_INDICATION)"}
    };

    auto fapi = reinterpret_cast<scf_fapi_header_t*>(data ? msg.data_buf : msg.msg_buf);
    printf("FAPI Header (%s buffer):\n", data ? "data" : "msg");
    printf("  msg_count = %d\n", fapi->message_count);
    printf("  handle_id = %d\n", fapi->handle_id);

    auto body = reinterpret_cast<scf_fapi_body_header_t*>(&fapi->payload[0]);
    printf("Body Header:\n");
    printf("  type_id = %s (%d)\n", id2str[body->type_id].c_str(), body->type_id);
    printf("  length = %d\n", body->length);

    switch (body->type_id) {
        case SCF_FAPI_CONFIG_REQUEST:
        {
            printf("CONFIG_REQ TLVs:\n");
            auto ts = reinterpret_cast<scf_fapi_config_request_body_t*>(body->data);
            auto tlv = reinterpret_cast<scf_fapi_tl_t*>(ts->tlvs);

            for (uint32_t t = 0; t < ts->num_tlvs; t++) {
                printf("  ");
                tlv = reinterpret_cast<scf_fapi_tl_t*>(tlv->Print());
                printf("\n");
            }
            break;
        }
        case SCF_FAPI_DL_TTI_REQUEST:
        {
            printf("DL TTI REQ\n");
            auto tti = reinterpret_cast<scf_fapi_dl_tti_req_t*>(&fapi->payload[0]);
            printf("  sfn=%d\n", tti->sfn);
            printf("  slot=%d\n", tti->slot);
            printf("  num_pdus=%d\n", tti->num_pdus);
#ifdef ENABLE_CONFORMANCE_TM_PDSCH_PDCCH
            printf("  testMode=%d\n", tti->testMode);
#endif
            printf("  ngroup=%d\n", tti->ngroup);

            auto req = reinterpret_cast<scf_fapi_generic_pdu_info_t*>(tti->payload);
            printf("  Payload:\n");

            static const char *pstr[] = {"PDCCH", "PDSCH", "CSI-RS", "SSB"};
            for (uint32_t n = 0; n < tti->num_pdus; n++) {
                printf("   PDU %d, type=%s (%d), size=%u\n", n, pstr[req->pdu_type], req->pdu_type, req->pdu_size);
                switch (req->pdu_type)
                {
                    case 0: break;
                    case 1:
                    {
                        printf("    PDSCH info:\n");
                        auto pdsch = reinterpret_cast<scf_fapi_pdsch_pdu_t*>(&req->pdu_config[0]);
                        printf("     pdu_bitmap=%u\n", pdsch->pdu_bitmap);
                        printf("     rnti=%u\n", pdsch->rnti);
                        printf("     pdu_index=%u\n", pdsch->pdu_index);
                        printf("     bwp.bwp_size=%u\n", pdsch->bwp.bwp_size);
                        printf("     bwp.bwp_start=%u\n", pdsch->bwp.bwp_start);
                        printf("     bwp.scs=%u\n", pdsch->bwp.scs);
                        printf("     bwp.cyclic_prefix=%u\n", pdsch->bwp.cyclic_prefix);
                        printf("     num_codewords=%u\n", pdsch->num_codewords);
                        printf("     Codewords:\n");
                        auto cwp = reinterpret_cast<scf_fapi_pdsch_codeword_t*>(&pdsch->codewords[0]);
                        for (uint32_t cw = 0; cw < pdsch->num_codewords; cw++) {
                            printf("      Codeword %u\n", cw);
                            printf("       target_code_rate=%u\n", pdsch->codewords[cw].target_code_rate);
                            printf("       qam_mod_order=%u\n", pdsch->codewords[cw].qam_mod_order);
                            printf("       mcs_index=%u\n", pdsch->codewords[cw].mcs_index);
                            printf("       mcs_table=%u\n", pdsch->codewords[cw].mcs_table);
                            printf("       rv_index=%u\n", pdsch->codewords[cw].rv_index);
                            printf("       tb_size=%u\n", pdsch->codewords[cw].tb_size);
                        }

                        auto end = reinterpret_cast<scf_fapi_pdsch_pdu_end_t*>(&pdsch->codewords[pdsch->num_codewords]);
                        printf("     data_scrambling_id=%u\n", end->data_scrambling_id);
                        printf("     num_of_layers=%u\n", end->num_of_layers);
                        printf("     transmission_scheme=%u\n", end->transmission_scheme);
                        printf("     ref_point=%u\n", end->ref_point);
                        printf("     dl_dmrs_sym_pos=%u\n", end->dl_dmrs_sym_pos);
                        printf("     dmrs_config_type=%u\n", end->dmrs_config_type);
                        printf("     dl_dmrs_scrambling_id=%u\n", end->dl_dmrs_scrambling_id);
                        printf("     sc_id=%u\n", end->sc_id);
                        printf("     num_dmrs_cdm_grps_no_data=%u\n", end->num_dmrs_cdm_grps_no_data);
                        printf("     dmrs_ports=%u\n", end->dmrs_ports);
                        printf("     resource_alloc=%u\n", end->resource_alloc);
                        printf("     rb_start=%u\n", end->rb_start);
                        printf("     rb_size=%u\n", end->rb_size);
                        printf("     vrb_to_prb_mapping=%u\n", end->vrb_to_prb_mapping);
                        printf("     start_sym_index=%u\n", end->start_sym_index);
                        printf("     num_symbols=%u\n", end->num_symbols);
                        // printf("     ptrs_port_index=%u\n", end->ptrs_port_index);
                        // printf("     ptrs_time_density=%u\n", end->ptrs_time_density);
                        // printf("     ptrs_freq_density=%u\n", end->ptrs_freq_density);
                        // printf("     ptrs_re_offset=%u\n", end->ptrs_re_offset);
                        // printf("     n_epre_ratio_of_pdsch_to_ptrs=%u\n", end->n_epre_ratio_of_pdsch_to_ptrs);

                    }
                }
            }
        }
    }

    if (dump) {
        printf("Data dump:\n");
        for (uint32_t i = 0; i < std::min(64, int(sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t) + body->length)); i++) {
            if ((i % 16) == 0)
                printf("\n%04x: ", i);

            printf("%02x ", reinterpret_cast<uint8_t*>(msg.msg_buf)[i]);
        }
    }

    printf("\n");
}


};

