/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <pthread.h>
#include <cstdint>
#include <string>
#include <vector>

// A struct that holds all total slot counters processed. 
typedef struct {
    // DL channels
    uint32_t pdsch{};
    uint32_t pdcch_dl{};
    uint32_t pdcch_ul{};
    uint32_t pbch{};
    uint32_t csi_rs{};
    uint32_t bfw_dl{};
    // UL channels
    uint32_t pusch{};
    uint32_t pucch{};
    uint32_t prach{};
    uint32_t srs{};
    uint32_t bfw_ul{};
} ru_emulator_total_slot_counters_t; 

// Opaque wrapper interface for RU_Emulator to avoid header conflicts
// Create/destroy
void* create_ru_emulator();
void destroy_ru_emulator(void* ru_emulator);

// RU_Emulator method wrappers
void ru_emulator_init(void* ru_emulator, int argc, char** argv);
int ru_emulator_start(void* ru_emulator);
int ru_emulator_finalize(void* ru_emulator);

// Configuration methods
void ru_emulator_set_default_configs(void* ru_emulator);
void ru_emulator_parse_yaml(void* ru_emulator, const std::string& yaml_file);
void ru_emulator_verify_and_apply_configs(void* ru_emulator);
void ru_emulator_print_configs(void* ru_emulator);

// TV and setup methods
void ru_emulator_load_tvs(void* ru_emulator);
void ru_emulator_setup_slots(void* ru_emulator);
void ru_emulator_add_flows(void* ru_emulator);
void ru_emulator_setup_rings(void* ru_emulator);
void ru_emulator_oam_init(void* ru_emulator);
void ru_emulator_verify_dl_cplane_content(void* ru_emulator, uint8_t *mbuf_payload, size_t buffer_length, int cell_index);
void ru_emulator_verify_ul_cplane_content(void* ru_emulator, uint8_t *mbuf_payload, size_t buffer_length, int cell_index);

// One parsed BFW PDU: host pointer to the parsed IQ samples plus the metadata
// needed to build a framework DynamicBfwCompletionRecord. Mirrors the RU
// emulator's per-PDU bfw_info; host_bfws aliases RU-emulator-owned memory that
// stays valid for the bench lifetime.
struct ru_emulator_bfw_pdu_t {
    uint8_t* host_bfws{};         //!< Parsed BFW IQ samples (RU-emulator owned)
    uint16_t rb_start{};          //!< bfw_info.rbStart
    uint16_t rb_size{};           //!< bfw_info.rbSize
    uint16_t num_prgs{};          //!< bfw_info.numPRGs
    uint16_t prg_size{};          //!< bfw_info.prgSize
    uint8_t  bfw_iq_bitwidth{};   //!< bfw_info.compressBitWidth
    uint16_t num_antenna_ports{}; //!< nGnbAnt (L_TRX)
};

// Collect the DL BFW PDUs scheduled for (sfn, slot) from get_bfw_dl_object().
void ru_emulator_construct_bfw_dl(void* ru_emulator, int sfn, int slot, int cell_idx,
                                  std::vector<ru_emulator_bfw_pdu_t>& out);

// Collect the UL BFW PDUs scheduled for (sfn, slot) from get_bfw_ul_object().
void ru_emulator_construct_bfw_ul(void* ru_emulator, int sfn, int slot, int cell_idx,
                                  std::vector<ru_emulator_bfw_pdu_t>& out);

// Query this at the end of a slot processing to get the total # of slots processed till that slot.
void ru_emulator_get_total_slt_counters (void *ru_emulator, int cell_idx, ru_emulator_total_slot_counters_t &counters); 
void ru_emulator_get_cplane_err_sections (void *ru_emulator, int cell_idx, uint64_t &err_sections);
void ru_emulator_get_cplane_dl_sections  (void *ru_emulator, int cell_idx, uint64_t &dl_sections);
void ru_emulator_get_cplane_ul_sections  (void *ru_emulator, int cell_idx, uint64_t &ul_sections);

// Framework optimization: PDCCH is sent on a single eAxC port.
// Adjust PDCCH TV numFlows to 1 so the RU emulator completion threshold matches.
void ru_emulator_set_pdcch_single_port(void* ru_emulator);
