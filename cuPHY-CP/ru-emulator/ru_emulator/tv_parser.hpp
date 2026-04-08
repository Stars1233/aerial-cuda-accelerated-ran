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

#ifndef TVPARSER_HPP__
#define TVPARSER_HPP__

#include <signal.h>
#include <sched.h>
#include <fstream>
#include <atomic>
#include <array>
#include <memory>
#include <unordered_map>

#include "ru_emulator.hpp"
#include "slot_command/slot_command.hpp"
#include "slot_command/csirs_lookup.hpp"

struct tv_parsing_timers
{
    uint64_t compute;
    uint64_t load;
};

void do_throw(std::string const& what);
// Convert a DataRx dataset to a slot, pre-filling all the pointers
Slot dataset_to_slot(Dataset d, size_t num_ante, size_t start_symbol, size_t num_symbols, size_t prb_sz, size_t tv_prbs_per_symbol, size_t start_prb);

/* Loads test vectors' requested single datasets into dataset */
Dataset load_tv_datasets_single(hdf5hpp::hdf5_file& hdf5file, std::string const& file, std::string const dataset);
int load_first_dimension_dataset(hdf5hpp::hdf5_file& hdf5file, std::string const& file, std::string dset_name);
Slot buffer_to_slot(uint8_t * buffer, size_t buffer_size, size_t num_ante, size_t start_symbol, size_t num_symbols, size_t prb_sz);

int load_ul_num_antenna_from_tv(hdf5hpp::hdf5_file& hdf5file, std::string const& file);
int load_ul_tb_size_from_tv(hdf5hpp::hdf5_file& hdf5file, std::string const& file);
bool is_nr_tv(hdf5hpp::hdf5_file& hdf5file);
int load_num_antenna_from_nr_tv(hdf5hpp::hdf5_file& hdf5file);
int load_num_antenna_from_nr_tv_srs(hdf5hpp::hdf5_file& hdf5file);
int load_num_antenna_from_nr_tv_zp_csi_rs(hdf5hpp::hdf5_file& hdf5file);
int load_num_antenna_from_nr_prach_tv(hdf5hpp::hdf5_file& hdf5file, std::string dset);
void load_dl_qams(hdf5hpp::hdf5_file& hdf5file, dl_tv_object& tv_object, dl_tv_info& dl_tv_info, bool mod_comp_enabled, bool non_mod_comp_enabled, const std::vector<struct cell_config>& cell_configs, bool selective_load);
void load_bfw_qams(hdf5hpp::hdf5_file& hdf5file, dl_tv_object& tv_object);
void parse_coreset(cuphyPdcchCoresetDynPrm_t& coreset, dci_param_list& dci, std::vector<std::pair<u_int16_t, u_int16_t>>& prb_pair);
bool merge_pdu_if_adjacent(std::vector<pdu_info>& existing_pdus, pdu_info& new_pdu);
bool merge_pdu_if_identical(std::vector<pdu_info>& existing_pdus, pdu_info& new_pdu);
#endif //ifndef TVPARSER_HPP__
