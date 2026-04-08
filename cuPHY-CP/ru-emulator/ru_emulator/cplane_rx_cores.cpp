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
#include "nvlog.h"
#include "shm_logger.h"
#include "perf_metrics/perf_metrics_accumulator.hpp"
#include "utils.hpp"
// #define RU_EM_UL_TIMERS_ENABLE 1

#define HALFBW (-(100 * 1000)/2)
#define GUARDBW 845
#define SCSBW 30

#undef TAG_TX_TIMINGS_SUM
#define TAG_TX_TIMINGS_SUM (NVLOG_TAG_BASE_RU_EMULATOR + 6) // "RU.TX_TIMINGS_SUM"

/**
 * @brief Validate beam IDs in a C-plane message against expected values from test vectors.
 *
 * Shared helper for both DL and UL 4T4R beam ID validation. Computes the expected
 * beam ID for the given eAxC index using the beam_repeat_interval mapping, then
 * compares each section's beamId against it.
 *
 * @param[in]     expected_beam_ids Pointer to the expected beam ID vector from the TV
 * @param[in]     num_flows         Number of antenna-carrier flows (DL or UL)
 * @param[in,out] c_plane_info      C-plane info with section data to validate
 * @param[in]     num_sections      Number of sections to check
 * @param[in]     cell_index        Cell identifier
 * @param[in,out] total_counter     Atomic counter for total checks
 * @param[in,out] error_counter     Atomic counter for mismatches
 * @param[in]     set_error_status  Whether to mark error_status on mismatch
 * @param[in]     direction_label   "DL" or "UL" for log messages
 */
static void validate_beamids(const std::vector<uint16_t>& expected_beam_ids,
                             size_t num_flows,
                             oran_c_plane_info_t& c_plane_info,
                             int num_sections,
                             int cell_index,
                             std::atomic<uint64_t>& total_counter,
                             std::atomic<uint64_t>& error_counter,
                             bool set_error_status,
                             const char* direction_label)
{
    if (expected_beam_ids.empty() || num_flows == 0)
    {
        return;
    }
    size_t beams_array_size = std::min(expected_beam_ids.size(), num_flows);
    size_t beam_repeat_interval = (beams_array_size > 0) ? (num_flows / beams_array_size) : 1;

    uint16_t expected = expected_beam_ids[0];
    if (c_plane_info.eaxcId_index >= 0 && static_cast<size_t>(c_plane_info.eaxcId_index) < num_flows)
    {
        size_t beam_idx = static_cast<size_t>(c_plane_info.eaxcId_index) / beam_repeat_interval;
        if (beam_idx < expected_beam_ids.size())
        {
            expected = expected_beam_ids[beam_idx];
        }
    }

    for (int sec_idx = 0; sec_idx < num_sections; ++sec_idx)
    {
        total_counter++;
        if (c_plane_info.section_infos[sec_idx].beamId != expected)
        {
            error_counter++;
            re_warn("{} C-plane beam ID mismatch: cell {} eAxC {} F{}S{}S{} section {} beamId {} expected {}",
                    direction_label, cell_index, c_plane_info.eaxcId,
                    c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId,
                    sec_idx, c_plane_info.section_infos[sec_idx].beamId, expected);
            if (set_error_status)
            {
                c_plane_info.section_infos[sec_idx].error_status = -1;
            }
        }
    }
}
#define TAG_UL_LATE_TX (NVLOG_TAG_BASE_RU_EMULATOR + 8) // "RU.UL_LATE_TX"
#define TAG_CP_WORKER_TRACING (NVLOG_TAG_BASE_RU_EMULATOR + 9) // "RU.CP_WORKER_TRACING"

int check_flow_exists(std::vector<int> flow_list, int flowId) {
    for(uint i = 0; i < flow_list.size(); ++i) {
        if(flowId == flow_list[i]) {
            return 0;
        }
    }
    return -1;
}

inline void generate_pcap_filename(char* output_filename, size_t max_len, uint8_t cell_id, const char* threadname)
{
    // Get current time
    struct timeval tv;
    struct tm timeinfo;
    char timestamp[32];

    gettimeofday(&tv, NULL);
    localtime_r(&tv.tv_sec, &timeinfo);

    // Format timestamp YYYYMMDD_HHMMSS
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &timeinfo);

    // Generate filename with format: YYYYMMDD_HHMMSS_CellId_ThreadName.pcap
    snprintf(output_filename, max_len,
             "%s_C%d_%s",
             timestamp, cell_id, threadname);
}

inline void generate_pcap_file(uint8_t cell_id, aerial_fh::MsgReceiveInfo* info, int nb_rx, const char* threadname)
{
    char output_filename[512];
    char shmlogger_name[64];
    snprintf(shmlogger_name, sizeof(shmlogger_name), "debug_ru_%s_pcap", threadname);

    shmlogger_config_t shm_cfg{};
    shm_cfg.save_to_file = 1;               // Start a background thread to save SHM cache to file before overflow
    shm_cfg.shm_cache_size = (1L << 27);    // 128MB, shared memory size, a SHM file will be created at /dev/shm/${name}_pcap
    shm_cfg.max_file_size = (1L << 28);     // 256MB Max file size, a disk file will be created at /var/log/aerial/${name}_pcap
    shm_cfg.file_saving_core = -1;          // CPU core ID for the background file saving if enabled.
    shm_cfg.shm_caching_core = -1;          // CPU core ID for the background copying to shared memory if enabled.
    shm_cfg.max_data_size = 8000;
    auto shmlogger = shmlogger_open(1, shmlogger_name, &shm_cfg);

    if (shmlogger == nullptr) {
        re_warn("Failed to open shmlogger for thread %s", threadname);
        return;
    }

    for(int i = 0; i < nb_rx; i++)
    {
        auto pkt = (uint8_t*)info[i].buffer;
        auto pkt_ts = info[i].rx_timestamp;
        auto ecpri_len = oran_umsg_get_ecpri_payload(pkt);
        shmlogger_save_fh_buffer(shmlogger, (const char*)pkt, ecpri_len + ORAN_ETH_HDR_SIZE + 4, 0, pkt_ts); // +4 for padding
    }

    generate_pcap_filename(output_filename, sizeof(output_filename), cell_id, threadname);

    char collect_prefix[64];
    snprintf(collect_prefix, sizeof(collect_prefix), "debug_ru_%s", threadname);
    shmlog_collect_params_t params = {
        .prefix = collect_prefix,
        .type = "pcap",
        .path = "/tmp",
        .fh_collect = 1,
        .output_filename = output_filename
    };
    shmlogger_collect_ex(&params);

    // Close the shmlogger to release resources and stop background threads
    shmlogger_close(shmlogger);
}


void RU_Emulator::increment_section_rx_counter(oran_c_plane_info_t& c_plane_info, uint16_t cell_index, int sym)
{
    if(c_plane_info.valid_eaxcId)
    {
        for(int sec = 0; sec < c_plane_info.numberOfSections; ++sec)
        {
            auto &section_info = c_plane_info.section_infos[sec];
            if (sym < c_plane_info.startSym || sym >= c_plane_info.startSym + section_info.numSymbol)
            {
                continue;
            }
            // Resolve per-section tv_object when sections carry different channel types.
            auto* tv_object = c_plane_info.is_mixed_channel ? section_info.tv_object : c_plane_info.tv_object;
            if (!tv_object) {
                re_warn("increment_section_rx_counter: null tv_object for section {}, cell {}, sym {} — skipping", sec, cell_index, sym);
                continue;
            }
            auto sec_channel = c_plane_info.is_mixed_channel ? section_info.channel_type : c_plane_info.channel_type;
            section_info.tv_index = tv_object->launch_pattern[c_plane_info.launch_pattern_slot][cell_index];
            auto & tv_info = tv_object->tv_info[section_info.tv_index];

            uint16_t numPrb = section_info.numPrbc;
            if (section_info.numPrbc == 0)
            {
                numPrb = cell_configs[cell_index].ru_type == ru_type::SINGLE_SECT_MODE ? (int)tv_object->tv_info[section_info.tv_index].numPrb : std::min((int)tv_object->tv_info[section_info.tv_index].numPrb, cell_configs[cell_index].ulGridSize);
            }
            int counter_index = c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId;

            ++tv_object->c_plane_rx[cell_index];
            ++tv_object->c_plane_rx_tot[cell_index];

            int ul_flows = sec_channel == ul_channel::SRS ? cell_configs[cell_index].num_valid_SRS_flows : cell_configs[cell_index].num_ul_flows;
            int prb_num = sec_channel == ul_channel::SRS ? tv_info.fss_numPrb[c_plane_info.fss.frameId][counter_index] : tv_info.numPrb;
            int expected_prbs = prb_num * ul_flows;
            if (opt_enable_mmimo && sec_channel != ul_channel::SRS)
            {
                expected_prbs = prb_num;
            }

            bool complete = false;
            {
                // Atomically increment and get the previous value
                const uint32_t prev_count = tv_object->prb_rx_counters[cell_index][counter_index].fetch_add(
                    numPrb, std::memory_order_acq_rel);
                const uint32_t new_count = prev_count + numPrb;
                
                // Check if we reached the expected threshold
                if (new_count == expected_prbs)
                {
                    tv_object->prb_rx_counters[cell_index][counter_index].store(0, std::memory_order_release);
                    complete = true;
                }
                else if (new_count > expected_prbs)
                {
                    re_warn("PRB counter overflow: cell_index={}, counter_index={}, new_count={}, expected_prbs={}",
                            cell_index, counter_index, new_count, expected_prbs);
                    tv_object->prb_rx_counters[cell_index][counter_index].store(0, std::memory_order_release);
                }
            }
            if(complete)
            {
                ++tv_object->throughput_slot_counters[cell_index];
                ++tv_object->total_slot_counters[cell_index];
                tv_object->throughput_counters[cell_index] += tv_object->tv_info[section_info.tv_index].tb_size;
                tv_object->section_rx_counters[cell_index][counter_index].store(0);
            }
        }
    }
}

void RU_Emulator::increment_section_rx_counter_v2(oran_c_plane_info_t& c_plane_info, uint16_t cell_index)
{
    if(c_plane_info.valid_eaxcId)
    {
        for(int sec = 0; sec < c_plane_info.numberOfSections; ++sec)
        {
            auto& section_info = c_plane_info.section_infos[sec];
            // Resolve per-section tv_object when sections carry different channel types.
            auto* tv_object = c_plane_info.is_mixed_channel ? section_info.tv_object : c_plane_info.tv_object;
            if (!tv_object) {
                re_warn("increment_section_rx_counter_v2: null tv_object for section {}, cell {} — skipping", sec, cell_index);
                continue;
            }
            auto sec_channel = c_plane_info.is_mixed_channel ? section_info.channel_type : c_plane_info.channel_type;
            section_info.tv_index = tv_object->launch_pattern[c_plane_info.launch_pattern_slot][cell_index];
            auto & tv_info = tv_object->tv_info[section_info.tv_index];

            uint16_t numPrb = section_info.numPrbc;
            if (section_info.numPrbc == 0)
            {
                numPrb = cell_configs[cell_index].ru_type == ru_type::SINGLE_SECT_MODE ? (int)tv_object->tv_info[section_info.tv_index].numPrb : std::min((int)tv_object->tv_info[section_info.tv_index].numPrb, cell_configs[cell_index].ulGridSize);
            }
            if (cell_configs[cell_index].ru_type != ru_type::SINGLE_SECT_MODE)
            {
                numPrb *= section_info.numSymbol;
            }
            int counter_index = c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID +
                    c_plane_info.fss.slotId;
            tv_object->prb_rx_counters[cell_index][counter_index] += numPrb;

            ++tv_object->c_plane_rx[cell_index];
            ++tv_object->c_plane_rx_tot[cell_index];

            int ul_flows = sec_channel == ul_channel::SRS ? cell_configs[cell_index].num_valid_SRS_flows : cell_configs[cell_index].num_ul_flows;
            int prb_num = sec_channel == ul_channel::SRS ? tv_info.fss_numPrb[c_plane_info.fss.frameId][counter_index] : tv_info.numPrb;
            int expected_prbs = prb_num * ul_flows;
            if (opt_enable_mmimo && sec_channel != ul_channel::SRS)
            {
                expected_prbs = prb_num;
            }
            if(tv_object->prb_rx_counters[cell_index][counter_index].load() == expected_prbs)
            {
                ++tv_object->throughput_slot_counters[cell_index];
                ++tv_object->total_slot_counters[cell_index];
                tv_object->throughput_counters[cell_index] += tv_object->tv_info[section_info.tv_index].tb_size;
                tv_object->section_rx_counters[cell_index][counter_index].store(0);
                tv_object->prb_rx_counters[cell_index][counter_index].store(0);
            }
        }
    }
}

void RU_Emulator::prepare_cplane_info(ul_tv_object& pusch_object, const std::vector<struct cell_config>& cell_configs,
                                     slot_tx_info& slot_tx, const struct fssId& fss, uint8_t cell_index,
                                     int slot_idx, int64_t next_slot_time, int64_t first_f0s0s0_time,
                                     int64_t frame_cycle_time_ns, int max_slot_id, int opt_tti_us, int64_t& slot_t0)
{
    auto& tv_info = pusch_object.tv_info[pusch_object.launch_pattern[slot_idx][cell_index]];
    auto num_ul_flows = cell_configs[cell_index].num_ul_flows;

    auto t0_toa = calculate_t0_toa(get_ns(), first_f0s0s0_time, frame_cycle_time_ns,
        fss.frameId, fss.subframeId,
        fss.slotId, tv_info.startSym,
        max_slot_id, opt_tti_us);
    slot_t0 = t0_toa.slot_t0;

    for(int i = 0; i < num_ul_flows; ++i)
    {
        int section_id = 0;
        auto& c_plane_info = slot_tx.c_plane_infos[slot_tx.c_plane_infos_size];
        c_plane_info.fss = fss;
        c_plane_info.slot_t0 = slot_t0;
        c_plane_info.eaxcId = cell_configs[cell_index].eAxC_UL[i];
        c_plane_info.eaxcId_index = i;
        c_plane_info.dir = oran_pkt_dir::DIRECTION_UPLINK;
        c_plane_info.section_type = ORAN_CMSG_SECTION_TYPE_1;
        c_plane_info.section_id = 0;
        c_plane_info.tv_object = &pusch_object;
        c_plane_info.channel_type = ul_channel::PUSCH;
        c_plane_info.startSym = tv_info.startSym;
        c_plane_info.section_infos_size = 0;
        c_plane_info.rx_time = next_slot_time + tv_info.startSym * (opt_tti_us * NS_X_US / ORAN_ALL_SYMBOLS);
        c_plane_info.numberOfSections = tv_info.pdu_infos.size();
        c_plane_info.valid_eaxcId = true;
        for(const auto& pdu_info: tv_info.pdu_infos)
        {
            auto& section_info = c_plane_info.section_infos[c_plane_info.section_infos_size];
            section_info.section_id = section_id++;
            section_info.numSymbol = pdu_info.numSym;
            section_info.startPrbc = pdu_info.startPrb;
            section_info.numPrbc = pdu_info.numPrb;
            section_info.tv_object = &pusch_object;
            section_info.channel_type = ul_channel::PUSCH;
            ++c_plane_info.section_infos_size;
        }
        ++slot_tx.ul_c_plane_infos_size;
        ++slot_tx.c_plane_infos_size;
    }
}

static void tx_complete_callback(void* addr, void* opaque)
{
}

aerial_fh::UPlaneTxCompleteNotification tx_complete_notification
{
    .callback     = tx_complete_callback,
    .callback_arg = nullptr,
};

// ---------------------------------------------------------------------------
// Pre-computed UL TX cache: build + fast-path
// ---------------------------------------------------------------------------

const srs_slot_type_info* get_srs_info(const int launch_pattern_slot,
                                        const srs_slot_type_info& srs_s3_info,
                                        const srs_slot_type_info& srs_s4_info,
                                        const srs_slot_type_info& srs_s5_info);

int get_window_index(const int launch_pattern_slot, const int symbol, const int16_t eaxcId_index,
                     const srs_slot_type_info& srs_s3_info, const srs_slot_type_info& srs_s4_info, const srs_slot_type_info& srs_s5_info,
                     const int srs_pacing_eaxcids_per_symbol, const int srs_pacing_eaxcids_per_tx_window);

/**
 * @brief Pre-populates precomputed_tx_cache with PrecomputedEaxcTx entries
 *        for every launch-pattern slot, cell, and UL eAxC flow.
 *
 * Iterates the launch-pattern (launch_pattern_slot_size x opt_num_cells) and,
 * for each enabled UL channel (PUSCH / PUCCH / SRS, gated by
 * opt_pusch_enabled, opt_pucch_enabled, opt_srs_enabled), looks up the
 * corresponding ul_tv_object slot data to build per-symbol section and
 * IQ-pointer information.
 *
 * @note PRACH (ul_channel::PRACH) is not pre-computed by this function;
 *       only PUSCH, PUCCH, and SRS are covered.
 *
 * Key side effects:
 *  - Resizes precomputed_tx_cache to [launch_pattern_slot_size][opt_num_cells].
 *  - Reads IQ pointers from ul_tv_object::slots indexed by compression mode.
 *  - Computes per-symbol tx_time_offset_ns from the UL U-Plane TX offset and
 *    (ORAN_ALL_SYMBOLS-derived) symbol duration.
 *  - Resolves TX queue indices via get_txq_index().
 *  - Logs the total entry count through re_cons.
 *
 * @note Relies on cell_configs for eAxC IDs, compression, grid size, and
 *       RU type.  Flow handles are drawn from peer_flow_map_srs (SRS) or
 *       ul_peer_flow_map (PUSCH/PUCCH).  Callers must ensure tv_obj->slots
 *       indices are valid for the chosen compression / TV index.
 * @note Modifies the class member precomputed_tx_cache (no parameters).
 *
 * @retval void
 *
 * @see PrecomputedEaxcTx
 * @see ul_tv_object
 * @see ul_channel
 * @see get_txq_index
 */
void RU_Emulator::precompute_ul_tx_cache()
{
    if (!opt_enable_precomputed_tx)
        return;

    const int lp_size = launch_pattern_slot_size;
    const int num_cells = opt_num_cells;

    precomputed_tx_cache.resize(lp_size);
    for (int slot_idx = 0; slot_idx < lp_size; ++slot_idx)
        precomputed_tx_cache[slot_idx].resize(num_cells);

    struct UlTvEntry {
        ul_tv_object* tv_obj;
        ul_channel    ch;
    };
    std::vector<UlTvEntry> ul_tvs;
    if (opt_pusch_enabled) ul_tvs.push_back({&pusch_object, ul_channel::PUSCH});
    if (opt_pucch_enabled) ul_tvs.push_back({&pucch_object, ul_channel::PUCCH});
    if (opt_srs_enabled)   ul_tvs.push_back({&srs_object,   ul_channel::SRS});

    int64_t symbol_duration_ns = (int64_t)opt_tti_us * NS_X_US / ORAN_ALL_SYMBOLS;

    size_t total_entries = 0;

    for (int slot_idx = 0; slot_idx < lp_size; ++slot_idx)
    {
        for (int cell = 0; cell < num_cells; ++cell)
        {
            auto& slot_cell = precomputed_tx_cache[slot_idx][cell];

            for (auto& [tv_obj, ch_type] : ul_tvs)
            {
                if (slot_idx >= (int)tv_obj->launch_pattern.size())
                    continue;
                auto it = tv_obj->launch_pattern[slot_idx].find(cell);
                if (it == tv_obj->launch_pattern[slot_idx].end())
                    continue;

                int tv_idx = it->second;
                if (tv_idx >= (int)tv_obj->tv_info.size())
                    continue;

                auto& tvi = tv_obj->tv_info[tv_idx];
                auto loaded_idx = cell_configs[cell].ul_comp_meth == aerial_fh::UserDataCompressionMethod::NO_COMPRESSION
                                  ? FIXED_POINT_16_BITS : cell_configs[cell].ul_bit_width;
                if (loaded_idx >= (int)tv_obj->slots.size() ||
                    tv_idx >= (int)tv_obj->slots[loaded_idx].size())
                    continue;

                auto& slot_data = tv_obj->slots[loaded_idx][tv_idx];
                int num_flows = (ch_type == ul_channel::SRS)
                               ? cell_configs[cell].num_valid_SRS_flows
                               : cell_configs[cell].num_ul_flows;

                for (int eaxc = 0; eaxc < num_flows; ++eaxc)
                {
                    PrecomputedEaxcTx entry{};
                    entry.channel_type = ch_type;
                    entry.tv_object = tv_obj;
                    entry.eaxcId_index = (int16_t)eaxc;
                    entry.eaxcId = (ch_type == ul_channel::SRS)
                                   ? cell_configs[cell].eAxC_SRS_list[eaxc]
                                   : cell_configs[cell].eAxC_UL[eaxc];
                    entry.start_sym = tvi.startSym;
                    entry.num_sym = tvi.numSym;

                    if (ch_type == ul_channel::SRS)
                        entry.flow = (cell < (int)peer_flow_map_srs.size() && eaxc < (int)peer_flow_map_srs[cell].size())
                                     ? peer_flow_map_srs[cell][eaxc] : aerial_fh::FlowHandle{};
                    else
                        entry.flow = (cell < (int)ul_peer_flow_map.size() && eaxc < (int)ul_peer_flow_map[cell].size())
                                     ? ul_peer_flow_map[cell][eaxc] : aerial_fh::FlowHandle{};

                    uint64_t tx_offset = 0;
                    if (ch_type != ul_channel::SRS)
                        tx_offset = oran_timing_info.ul_u_plane_tx_offset * NS_X_US;
                    else
                        tx_offset = oran_timing_info.ul_u_plane_tx_offset_srs * NS_X_US;

                    for (int sym = 0; sym < ORAN_ALL_SYMBOLS; ++sym)
                    {
                        if (ch_type == ul_channel::SRS && opt_enable_srs_eaxcid_pacing)
                        {
                            const int win_idx = get_window_index(slot_idx, sym, eaxc,
                                srs_s3_info, srs_s4_info, srs_s5_info,
                                opt_srs_pacing_eaxcids_per_symbol, opt_srs_pacing_eaxcids_per_tx_window);
                            if (win_idx >= 0)
                            {
                                const auto* srs_info = get_srs_info(slot_idx, srs_s3_info, srs_s4_info, srs_s5_info);
                                int64_t slot_time_offset_ns = (srs_info != nullptr) ? srs_info->slot_time_offset_ns : 0;
                                entry.tx_time_offset_ns[sym] = slot_time_offset_ns + (int64_t)tx_offset
                                    + (int64_t)win_idx * symbol_duration_ns;
                            }
                            else
                            {
                                entry.tx_time_offset_ns[sym] = (int64_t)tx_offset + (int64_t)sym * symbol_duration_ns;
                            }
                        }
                        else
                        {
                            entry.tx_time_offset_ns[sym] = (int64_t)tx_offset + (int64_t)sym * symbol_duration_ns;
                        }

                        entry.txq_index[sym] = get_txq_index(ch_type, slot_idx, sym, eaxc,
                                                              opt_split_srs_txq, opt_enable_srs_eaxcid_pacing,
                                                              srs_s3_info, srs_s4_info, srs_s5_info,
                                                              opt_srs_pacing_eaxcids_per_symbol, opt_srs_pacing_eaxcids_per_tx_window);
                    }

                    slot_cell.eaxc_entries.push_back(std::move(entry));
                    ++total_entries;
                }
            }
        }
    }

    re_cons("Pre-computed UL TX cache: {} entries across {} launch pattern slots x {} cells",
            total_entries, lp_size, num_cells);
}

inline void RU_Emulator::update_ul_throughput_counters(oran_c_plane_info_t& c_plane_info, uint16_t cell_index)
{
    if (cell_configs[cell_index].ru_type == ru_type::SINGLE_SECT_MODE)
    {
        for (int sec = 0; sec < c_plane_info.numberOfSections; ++sec)
        {
            auto& section_info = c_plane_info.section_infos[sec];
            if (opt_pucch_enabled && prb_range_matching(c_plane_info, section_info, cell_index, pucch_object))
            {
                section_info.tv_object = &pucch_object;
                section_info.channel_type = ul_channel::PUCCH;
                c_plane_info.tv_object = &pucch_object;
                c_plane_info.channel_type = ul_channel::PUCCH;
                if (opt_enable_mmimo) {
                    for (int sym = 0; sym < ORAN_ALL_SYMBOLS; ++sym)
                        increment_section_rx_counter(c_plane_info, cell_index, sym);
                } else {
                    increment_section_rx_counter_v2(c_plane_info, cell_index);
                }
            }
            if (opt_srs_enabled && prb_range_matching(c_plane_info, section_info, cell_index, srs_object))
            {
                section_info.tv_object = &srs_object;
                section_info.channel_type = ul_channel::SRS;
                c_plane_info.tv_object = &srs_object;
                c_plane_info.channel_type = ul_channel::SRS;
                if (opt_enable_mmimo) {
                    for (int sym = 0; sym < ORAN_ALL_SYMBOLS; ++sym)
                        increment_section_rx_counter(c_plane_info, cell_index, sym);
                } else {
                    increment_section_rx_counter_v2(c_plane_info, cell_index);
                }
            }
            if (opt_pusch_enabled && prb_range_matching(c_plane_info, section_info, cell_index, pusch_object))
            {
                section_info.tv_object = &pusch_object;
                section_info.channel_type = ul_channel::PUSCH;
                c_plane_info.tv_object = &pusch_object;
                c_plane_info.channel_type = ul_channel::PUSCH;
                if (opt_enable_mmimo) {
                    for (int sym = 0; sym < ORAN_ALL_SYMBOLS; ++sym)
                        increment_section_rx_counter(c_plane_info, cell_index, sym);
                } else {
                    increment_section_rx_counter_v2(c_plane_info, cell_index);
                }
            }
        }
    }
    else
    {
        if (opt_enable_mmimo) {
            for (int sym = 0; sym < ORAN_ALL_SYMBOLS; ++sym)
                increment_section_rx_counter(c_plane_info, cell_index, sym);
        } else {
            increment_section_rx_counter_v2(c_plane_info, cell_index);
        }
    }
}

/**
 * @brief Transmits UL U-Plane packets for one slot/cell using the
 *        pre-computed TX cache, falling back to handle_sect1_c_plane()
 *        (mMIMO, per-symbol) or handle_sect1_c_plane_v2() (non-mMIMO)
 *        when no cache match exists.
 *
 *  **Phase 1 – Cache resolution.**  Iterates over all UL section-type-1
 *  C-Plane info entries in @p slot_tx and looks up
 *  precomputed_tx_cache[launch_pattern_slot][cell] for a matching
 *  PrecomputedEaxcTx (by channel_type and eaxcId_index).  Matched entries
 *  are collected into a fixed-size buffer; unmatched entries are sent
 *  immediately via the appropriate legacy handler.
 *
 *  **Phase 2 – Mode-dependent transmit ordering.**  When mMIMO is
 *  enabled, uses sym-outer / CPI-inner ordering matching legacy
 *  tx_slot(enable_mmimo=true) so all channel types (PUSCH, PUCCH, SRS)
 *  for a given symbol are sent before the next symbol.  When mMIMO is
 *  disabled, uses CPI-outer / sym-inner ordering matching legacy
 *  handle_sect1_c_plane_v2() which completes all symbols per CPI.
 *  Each CPI is prepared and sent individually (no cross-CPI batching)
 *  to keep NIC TX burst sizes small.
 *
 *  **Phase 3 – Deferred counter update.**  After all symbols are sent,
 *  accumulates u_plane_tx / u_plane_tx_tot on each matched CPI's
 *  tv_object, logs TAG_TX_TIMINGS_SUM, and calls
 *  update_ul_throughput_counters().
 *
 *  **Phase 4 – BFW extension verification.**  Iterates over all C-Plane
 *  info entries and calls verify_extensions() (mMIMO) or
 *  verify_extType11() (non-mMIMO) on sections carrying BFW extensions.
 *
 * @param[in,out] slot_tx     Slot TX descriptor holding the array of
 *                            C-Plane info entries to transmit.
 * @param[in]     cell_index  Zero-based cell index into precomputed_tx_cache,
 *                            cell_configs, peer_list, and packet counters.
 * @param[in,out] timers      Per-symbol timing helpers (passed through to
 *                            handle_sect1_c_plane / handle_sect1_c_plane_v2
 *                            on fallback).
 * @param[in]     txqs        Array of TX queue handles indexed by txq_index;
 *                            caller retains ownership.
 * @param[in,out] tx_request  Pre-allocated TX request handle reused across
 *                            prepare/send cycles; caller retains ownership.
 * @param[in,out] profiler    Optional performance profiler (may be nullptr).
 *                            Sections "packet_preparation" and "packet_send"
 *                            are timed when non-null.
 *
 * @return Total number of U-Plane packets transmitted (sum of all send
 *         calls, including fallback path).
 *
 * @note This function runs on the real-time TX path.  @p txqs and
 *       @p tx_request are owned by the caller and must remain valid for the
 *       duration of the call.  @p tx_request is mutated by each
 *       prepare/send cycle but is not freed.
 * @note precomputed_tx_cache must have been populated by
 *       precompute_ul_tx_cache() before this function is called.
 *       cpi.eaxcId_index must be non-negative for a cache hit; entries
 *       with negative values fall through to the appropriate handler.
 *
 * @see PrecomputedEaxcTx
 * @see precompute_ul_tx_cache
 * @see handle_sect1_c_plane
 * @see handle_sect1_c_plane_v2
 * @see ul_channel
 * @see get_txq_index
 */
int RU_Emulator::tx_slot_precomputed(slot_tx_info& slot_tx, int cell_index,
                                     tx_symbol_timers& timers,
                                     aerial_fh::TxqHandle* txqs,
                                     aerial_fh::TxRequestHandle* tx_request,
                                     perf_metrics::PerfMetricsAccumulator* profiler)
{
    int total_tx = 0;

    // --- Phase 1: Resolve cache matches ---
    struct CPlaneInfoMatch {
        int                c_plane_info_idx;
        PrecomputedEaxcTx* match;
        size_t             nb_tx;
    };
    constexpr int kMaxMatchedCpis = 32;
    CPlaneInfoMatch matched_buf[kMaxMatchedCpis];
    int num_matched = 0;

    /// Fallback: transmit one C-Plane info entry through the legacy
    /// (non-precomputed) handler and update total_tx / per-TV counters.
    /// Uses handle_sect1_c_plane (per-symbol) for mMIMO, or
    /// handle_sect1_c_plane_v2 (all symbols at once) for non-mMIMO.
    auto send_via_legacy = [&](oran_c_plane_info_t& cpi) -> size_t
    {
        size_t nb = 0;
        if (opt_enable_mmimo) {
            for (int sym = 0; sym < ORAN_ALL_SYMBOLS; ++sym)
                nb += handle_sect1_c_plane(cpi, cell_index, timers, txqs, tx_request, sym, nullptr);
        } else {
            nb = handle_sect1_c_plane_v2(cpi, cell_index, timers, txqs, tx_request);
        }
        total_tx += nb;
        cpi.tv_object->u_plane_tx[cell_index] += nb;
        cpi.tv_object->u_plane_tx_tot[cell_index] += nb;
        return nb;
    };

    for (int i = 0; i < slot_tx.c_plane_infos_size; ++i)
    {
        auto& c_plane_info = slot_tx.c_plane_infos[i];
        if (c_plane_info.dir == DIRECTION_DOWNLINK)
            continue;
        if (c_plane_info.section_type != ORAN_CMSG_SECTION_TYPE_1)
            continue;
        if (c_plane_info.eaxcId_index < 0)
        {
            send_via_legacy(c_plane_info);
            continue;
        }

        int lp_slot = c_plane_info.launch_pattern_slot;
        if (lp_slot >= (int)precomputed_tx_cache.size() ||
            cell_index >= (int)precomputed_tx_cache[lp_slot].size())
        {
            send_via_legacy(c_plane_info);
            continue;
        }

        auto& slot_cell = precomputed_tx_cache[lp_slot][cell_index];

        PrecomputedEaxcTx* match = nullptr;
        for (auto& entry : slot_cell.eaxc_entries)
        {
            if (entry.channel_type == c_plane_info.channel_type &&
                entry.eaxcId_index == c_plane_info.eaxcId_index)
            {
                match = &entry;
                break;
            }
        }

        if (!match)
        {
            send_via_legacy(c_plane_info);
            continue;
        }

        if (num_matched < kMaxMatchedCpis)
        {
            matched_buf[num_matched++] = {.c_plane_info_idx = i, .match = match, .nb_tx = 0};
        }
        else
        {
            static bool warned = false;
            if (!warned) {
                re_warn("Precomputed TX matched-CPI overflow ({}/{}), falling back to legacy",
                        num_matched, kMaxMatchedCpis);
                warned = true;
            }
            send_via_legacy(c_plane_info);
        }
    }

    // --- Phase 2: Transmit matched CPIs with mode-appropriate ordering ---
    aerial_fh::UPlaneMsgMultiSectionSendInfo uplane_msg = {};
    if (opt_ecpri_hdr_cfg_test)
        uplane_msg.ecpri_hdr_cfg = &ecpri_hdr_cfg;

    auto send_one_cpi_symbol = [&](int mc, int sym)
    {
        auto& c_plane_info = slot_tx.c_plane_infos[matched_buf[mc].c_plane_info_idx];
        auto* match = matched_buf[mc].match;

        uplane_msg.section_num = 0;
        int num_sections_added = 0;

        int lp_slot = c_plane_info.launch_pattern_slot;
        auto loaded_idx = cell_configs[cell_index].ul_comp_meth == aerial_fh::UserDataCompressionMethod::NO_COMPRESSION
                          ? FIXED_POINT_16_BITS : cell_configs[cell_index].ul_bit_width;
        int tv_idx = match->tv_object->launch_pattern[lp_slot][cell_index];
        auto& slot_data = match->tv_object->slots[loaded_idx][tv_idx];

        for (int sec = 0; sec < c_plane_info.numberOfSections; ++sec)
        {
            auto& section_info = c_plane_info.section_infos[sec];
            if (sym < c_plane_info.startSym || sym >= c_plane_info.startSym + section_info.numSymbol)
                continue;

            if (uplane_msg.section_num >= aerial_fh::kMaxSectionNum)
                break;

            auto& dst = uplane_msg.section_infos[uplane_msg.section_num++];

            uint16_t prb_idx = section_info.startPrbc;
            if (c_plane_info.eaxcId_index >= 0 &&
                c_plane_info.eaxcId_index < (int)slot_data.ptrs.size() &&
                sym < (int)slot_data.ptrs[c_plane_info.eaxcId_index].size() &&
                prb_idx < (int)slot_data.ptrs[c_plane_info.eaxcId_index][sym].size())
            {
                dst.iq_data_buffer = slot_data.ptrs[c_plane_info.eaxcId_index][sym][prb_idx];
            }
            else
            {
                dst.iq_data_buffer = match->tv_object->blank_prbs.get();
            }

            dst.section_id = section_info.section_id;
            dst.rb = 0;
            dst.sym_inc = 0;
            dst.start_prbu = section_info.startPrbc;

            uint16_t numPrbc = section_info.numPrbc;
            if (numPrbc == 0)
            {
                auto& tv_info = match->tv_object->tv_info[tv_idx];
                numPrbc = cell_configs[cell_index].ru_type == ru_type::SINGLE_SECT_MODE
                          ? cell_configs[cell_index].ulGridSize
                          : std::min((int)tv_info.numPrb, cell_configs[cell_index].ulGridSize);
            }
            if (numPrbc == 0)
                numPrbc = cell_configs[cell_index].ulGridSize;
            dst.num_prbu = numPrbc;

            ++num_sections_added;
        }

        if (num_sections_added > 0 && !check_if_drop((uint16_t)cell_index, c_plane_info.channel_type, c_plane_info.fss))
        {
            auto& hdr = uplane_msg.radio_app_hdr;
            hdr.frameId    = c_plane_info.fss.frameId;
            hdr.subframeId = c_plane_info.fss.subframeId;
            hdr.slotId     = c_plane_info.fss.slotId;
            hdr.symbolId   = sym;

            uplane_msg.flow = match->flow;

            uint64_t tx_time = (uint64_t)c_plane_info.slot_t0 + match->tx_time_offset_ns[sym];
            if (is_cx6_nic)
            {
                if (opt_afh_accu_tx_sched_res_ns != 0)
                    uplane_msg.tx_window.tx_window_start = tx_time;
            }
            else
            {
                uplane_msg.tx_window.tx_window_start = tx_time;
            }

            int txq_idx = match->txq_index[sym];

            if (profiler) profiler->startSection("packet_preparation");
            aerial_fh::prepare_uplane_with_preallocated_tx_request(
                peer_list[cell_index], &uplane_msg, tx_complete_notification, tx_request, txq_idx);
            if (profiler) profiler->stopSection("packet_preparation");

            if (profiler) profiler->startSection("packet_send");
            auto tx_cnt = aerial_fh::send_uplane_without_freeing_tx_request(*tx_request, txqs[txq_idx]);
            if (profiler) profiler->stopSection("packet_send");

            auto now = get_ns();
            bool is_late = (tx_time < now && opt_afh_accu_tx_sched_res_ns != 0);

            if (is_late)
            {
                NVLOGI_FMT(TAG_UL_LATE_TX,
                    "Cell {} F{}S{}S{} Scheduling section type 1 packets in the past rx time {} tx_time {}, now {}, now - tx_time {}",
                    cell_index, c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId,
                    c_plane_info.rx_time, tx_time, now, now - tx_time);
            }

            NVLOGI_FMT(TAG_TX_TIMINGS,
                "[ST1] {} F{}S{}S{} Cell {} TX Time {} Enqueue Time {} Sym {} Num Packets {} {} Queue {}",
                ul_channel_to_string(c_plane_info.channel_type),
                c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId,
                cell_index, tx_time, now, sym,
                num_sections_added, tx_cnt, txq_idx);

            matched_buf[mc].nb_tx += tx_cnt;

            if (oran_packet_counters.ul_c_plane[cell_index].total_slot.load() >= opt_ul_warmup_slots)
            {
                auto timing = is_late ? PacketCounterTiming::LATE : PacketCounterTiming::ONTIME;
                if (c_plane_info.channel_type == ul_channel::PUCCH)
                    ul_u_pucch_packet_stats.increment_counters(cell_index, timing, lp_slot, tx_cnt);
                else if (c_plane_info.channel_type == ul_channel::PUSCH)
                    ul_u_pusch_packet_stats.increment_counters(cell_index, timing, lp_slot, tx_cnt);
                else if (c_plane_info.channel_type == ul_channel::SRS)
                    ul_u_srs_packet_stats.increment_counters(cell_index, timing, lp_slot, tx_cnt);
            }
        }
    };

    if (opt_enable_mmimo)
    {
        // sym-outer, CPI-inner — matches legacy mMIMO tx_slot() so all
        // channel types for a given symbol are sent before the next symbol.
        for (int sym = 0; sym < ORAN_ALL_SYMBOLS; ++sym)
            for (int mc = 0; mc < num_matched; ++mc)
                send_one_cpi_symbol(mc, sym);
    }
    else
    {
        // CPI-outer, sym-inner — matches legacy non-mMIMO
        // handle_sect1_c_plane_v2() which completes all symbols per CPI.
        for (int mc = 0; mc < num_matched; ++mc)
            for (int sym = 0; sym < ORAN_ALL_SYMBOLS; ++sym)
                send_one_cpi_symbol(mc, sym);
    }

    // --- Phase 3: Deferred counter updates ---
    for (int mc = 0; mc < num_matched; ++mc)
    {
        auto& c_plane_info = slot_tx.c_plane_infos[matched_buf[mc].c_plane_info_idx];
        size_t nb_tx = matched_buf[mc].nb_tx;

        auto now = get_ns();
        NVLOGI_FMT(TAG_TX_TIMINGS_SUM, "[ST1] {} F{}S{}S{} Cell {} Enqueue Time {}",
                    ul_channel_to_string(c_plane_info.channel_type),
                    c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId,
                    cell_index, now);

        total_tx += nb_tx;

        c_plane_info.tv_object->u_plane_tx[cell_index] += nb_tx;
        c_plane_info.tv_object->u_plane_tx_tot[cell_index] += nb_tx;

        update_ul_throughput_counters(c_plane_info, cell_index);
    }

    // --- Phase 4: BFW extension verification ---
    for (int i = 0; i < slot_tx.c_plane_infos_size; ++i)
    {
        auto& c_plane_info = slot_tx.c_plane_infos[i];
        if (opt_enable_mmimo)
        {
            if (c_plane_info.dir == DIRECTION_DOWNLINK)
                continue;
            for (auto section_idx = 0; section_idx < c_plane_info.section_infos_size; section_idx++)
            {
                auto& section_info = c_plane_info.section_infos[section_idx];
                if (section_info.ext_infos_size > 0)
                {
                    verify_extensions(c_plane_info, section_info, cell_index);
                }
            }
        }
        else
        {
            for (auto section_idx = 0; section_idx < c_plane_info.section_infos_size; section_idx++)
            {
                auto& section_info = c_plane_info.section_infos[section_idx];
                if (section_info.ext11_ptr != nullptr)
                {
                    verify_extType11(section_info.ext11_ptr, c_plane_info, section_info, cell_index);
                }
            }
        }
    }

    return total_tx;
}

bool RU_Emulator::prb_range_check(oran_c_plane_info_t &c_plane_info, oran_c_plane_section_info_t &section_info, std::vector<pdu_info>& pdu_infos)
{
    auto range_check = [](uint16_t s1, uint16_t n1, uint16_t s2, uint16_t n2)
    { return s1 >= s2 && (s1 + n1) <= (s2 + n2); };
    int symbol_id = c_plane_info.startSym;
    for (int pdu = 0; pdu < pdu_infos.size(); ++pdu)
    {
        auto &pdu_info = pdu_infos[pdu];
        if (symbol_id >= pdu_info.startSym && symbol_id < (pdu_info.startSym + pdu_info.numSym) && (range_check(section_info.startPrbc, section_info.numPrbc, pdu_info.startPrb, pdu_info.numPrb) || (pdu_info.freqHopFlag > 0 && range_check(section_info.startPrbc, section_info.numPrbc, pdu_info.secondHopPrb, pdu_info.numPrb))))
        {
            return true;
        }
    }
    return false;
}

bool RU_Emulator::prb_range_check(oran_c_plane_info_t &c_plane_info, std::vector<pdu_info>& pdu_infos)
{
    auto range_check = [](uint16_t s1, uint16_t n1, uint16_t s2, uint16_t n2)
    { return s1 >= s2 && (s1 + n1) <= (s2 + n2); };
    int symbol_id = c_plane_info.startSym;
    for (int sec = 0; sec < c_plane_info.numberOfSections; ++sec)
    {
        auto &section_info = c_plane_info.section_infos[sec];
        // Refer to section 5.4.5.3 of ORAN-WG4.CUS.0
        if (section_info.symInc != 0)
        {
            symbol_id += c_plane_info.section_infos[sec - 1].numSymbol;
        }

        for (int pdu = 0; pdu < pdu_infos.size(); ++pdu)
        {
            auto &pdu_info = pdu_infos[pdu];
            if(pdu_info.rb && (!section_info.rb || (pdu_info.startPrb&1) != (section_info.startPrbc&1)))
            {
                continue;
            }

            if (symbol_id >= pdu_info.startSym && symbol_id < (pdu_info.startSym + pdu_info.numSym) && (range_check(section_info.startPrbc, section_info.numPrbc, pdu_info.startPrb, pdu_info.numPrb) || (pdu_info.freqHopFlag > 0 && range_check(section_info.startPrbc, section_info.numPrbc, pdu_info.secondHopPrb, pdu_info.numPrb))))
            {
                return true;
            }
        }
    }
    return false;
}

bool RU_Emulator::prb_range_matching(oran_c_plane_info_t &c_plane_info, oran_c_plane_section_info_t &section_info, uint16_t cell_index, struct ul_tv_object& tv_obj)
{
    if (c_plane_info.launch_pattern_slot < tv_obj.launch_pattern.size() && tv_obj.launch_pattern[c_plane_info.launch_pattern_slot].find(cell_index) != tv_obj.launch_pattern[c_plane_info.launch_pattern_slot].end())
    {
        int tv_idx = tv_obj.launch_pattern[c_plane_info.launch_pattern_slot][cell_index];
        if (tv_idx < tv_obj.tv_info.size())
        {
            if(cell_configs[cell_index].ru_type == ru_type::SINGLE_SECT_MODE)
            {
                return true;
            }
            auto pdu_infos  = &tv_obj == &srs_object ? tv_obj.tv_info[tv_idx].fss_pdu_infos[c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId] : tv_obj.tv_info[tv_idx].pdu_infos;
            return prb_range_check(c_plane_info, section_info, pdu_infos);
        }
    }
    return false;
}

bool RU_Emulator::prb_range_matching(oran_c_plane_info_t &c_plane_info, uint16_t cell_index, struct ul_tv_object& tv_obj)
{
    if (c_plane_info.launch_pattern_slot < tv_obj.launch_pattern.size() && tv_obj.launch_pattern[c_plane_info.launch_pattern_slot].find(cell_index) != tv_obj.launch_pattern[c_plane_info.launch_pattern_slot].end())
    {
        int tv_idx = tv_obj.launch_pattern[c_plane_info.launch_pattern_slot][cell_index];
        if (tv_idx < tv_obj.tv_info.size())
        {
            auto pdu_infos  = &tv_obj == &srs_object ? tv_obj.tv_info[tv_idx].fss_pdu_infos[c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId] : tv_obj.tv_info[tv_idx].pdu_infos;
            return prb_range_check(c_plane_info, pdu_infos);
        }
    }
    return false;
}

bool RU_Emulator::prb_range_matching(oran_c_plane_info_t &c_plane_info, uint16_t cell_index, struct dl_tv_object& tv_obj)
{
    if (c_plane_info.launch_pattern_slot < tv_obj.launch_pattern.size() && tv_obj.launch_pattern[c_plane_info.launch_pattern_slot].find(cell_index) != tv_obj.launch_pattern[c_plane_info.launch_pattern_slot].end())
    {
        int tv_idx = tv_obj.launch_pattern[c_plane_info.launch_pattern_slot][cell_index];
        if (tv_idx < tv_obj.tv_info.size())
        {
            return prb_range_check(c_plane_info, tv_obj.tv_info[tv_idx].csirc_pdu_infos.size() > 0 ? tv_obj.tv_info[tv_idx].csirc_pdu_infos : tv_obj.tv_info[tv_idx].pdu_infos);
        }
    }
    return false;
}

/**
 * @brief Validates modulation compression for a given C-plane section
 *
 * This function searches through DL TV objects to find matching modulation compression
 * messages for the given section. It validates that the section's PRB range and
 * resource element mask match with available modulation compression data.
 *
 * @param[in] c_plane_info C-plane information containing frame/subframe/slot and eAxC details
 * @param[in] section_info Section information containing PRB range, RE mask, and symbol details
 * @param[in] cell_index Cell identifier for the validation
 * @return true if matching modulation compression message is found, false otherwise
 */
bool RU_Emulator::validate_modulation_compression(const oran_c_plane_info_t& c_plane_info, const oran_c_plane_section_info_t& section_info, int cell_index)
{
    const auto portIdx = c_plane_info.eaxcId_index;
    const auto section_reMask = section_info.reMask;
    bool found = false;
    int matched_msg_idx = -1;
    tv_mod_comp_object* matched_mod_comp = nullptr;
    uint32_t matched_tv_reMask = 0;

    for (auto tv_obj_p : dl_tv_objs)
    {
        if (c_plane_info.launch_pattern_slot < tv_obj_p->launch_pattern.size() &&
            tv_obj_p->launch_pattern[c_plane_info.launch_pattern_slot].find(cell_index) != tv_obj_p->launch_pattern[c_plane_info.launch_pattern_slot].end())
        {
            int tv_idx = tv_obj_p->launch_pattern[c_plane_info.launch_pattern_slot][cell_index];
            if (tv_idx >= tv_obj_p->tv_info.size())
                continue;

            auto &mod_comp_data = tv_obj_p->mod_comp_data[tv_idx];
            auto &hdr = mod_comp_data.mod_comp_header;

            for (int sym = c_plane_info.startSym; sym < c_plane_info.startSym + section_info.numSymbol; sym++)
            {
                for (int msg_port_idx = portIdx;; msg_port_idx += cell_configs[cell_index].num_dl_flows)
                {
                    if (tv_obj_p == &csirs_object && msg_port_idx >= tv_obj_p->tv_info[tv_idx].csirsMaxPortNum)
                    {
                        break;
                    }

                    const auto sym_it = hdr.find(sym);
                    if (sym_it != hdr.end())
                    {
                        auto &port_map = sym_it->second;
                        const auto port_it = port_map.find(msg_port_idx);
                        if (port_it != port_map.end())
                        {
                            auto &mask_map = port_it->second;

                            // Fast path: exact mask match
                            const auto exact_it = mask_map.find(section_reMask);
                            if (exact_it != mask_map.end())
                            {
                                for (const auto &e : exact_it->second)
                                {
                                    if (section_info.startPrbc >= e[0] &&
                                        section_info.startPrbc + section_info.numPrbc <= e[0] + e[1])
                                    {
                                        found = true;
                                        matched_msg_idx = e[2];
                                        matched_mod_comp = &mod_comp_data;
                                        matched_tv_reMask = section_reMask;
                                        break;
                                    }
                                }

                                if (found)
                                {
                                    break;
                                }
                            }

                            // Fallback: allow matches where tv_reMask is a superset of section_reMask
                            for (const auto &mask_entry : mask_map)
                            {
                                const std::uint32_t tv_reMask = mask_entry.first;

                                // Skip exact match key which was already checked above
                                if (tv_reMask == section_reMask)
                                {
                                    continue;
                                }

                                // Match when all bits set in section_reMask are also set in tv_reMask
                                if ((section_reMask & tv_reMask) != section_reMask)
                                {
                                    continue;
                                }

                                for (const auto &e : mask_entry.second)
                                {
                                    if (section_info.startPrbc >= e[0] &&
                                        section_info.startPrbc + section_info.numPrbc <= e[0] + e[1])
                                    {
                                        found = true;
                                        matched_msg_idx = e[2];
                                        matched_mod_comp = &mod_comp_data;
                                        matched_tv_reMask = tv_reMask;
                                        break;
                                    }
                                }

                                if (found)
                                {
                                    break;
                                }
                            }
                        }
                    }

                    if (found || tv_obj_p != &csirs_object)
                    {
                        break;
                    }
                }
            }
        }
    }

    if (!found)
    {
        re_cons("No modComp MSG found for Cell {} SEC1 C-Plane: F{}S{}S{} sym {} eAxC {} port {} sec_id {} remask 0x{:x} startPrbc {} numPrbc {} numSymbol {} ",
                (uint32_t)cell_index,
                (uint32_t)c_plane_info.fss.frameId,
                (uint32_t)c_plane_info.fss.subframeId,
                (uint32_t)c_plane_info.fss.slotId,
                (uint32_t)c_plane_info.startSym,
                (uint32_t)c_plane_info.eaxcId,
                (uint32_t)c_plane_info.eaxcId_index,
                (uint32_t)section_info.section_id,
                (uint32_t)section_info.reMask,
                (uint32_t)section_info.startPrbc,
                (uint32_t)section_info.numPrbc,
                (uint32_t)section_info.numSymbol);
        return false;
    }

    // --- Cross-check superset-fallback reMask against known patterns ---
    // When the exact reMask match failed and the superset fallback was used,
    // verify that the wire reMask is legitimate.  Accepted cases:
    //   1. CSI-RS puncturing: wire == (~csirsREMask) & 0xFFF
    //   2. SE5 RE sub-grouping: wire == mc_scale_re_mask[g]  (from ext_info)
    //   3. Both combined: wire == mc_scale_re_mask[g] & (~csirsREMask) & 0xFFF
    //   4. SE5 sub-grouping without ext_info: TV reMask != 0xFFF and wire is
    //      a subset -- the TV already carries a grouped mask, so the DU splitting
    //      it into per-group sections is expected.  Optionally combined with
    //      CSI-RS puncturing (wire is a subset of TV & csirs_pdsch_mask).
    if (found && matched_tv_reMask != section_reMask)
    {
        bool mismatch_explained = false;

        uint16_t csirs_pdsch_mask = 0xFFF;
        if (c_plane_info.launch_pattern_slot < csirs_object.launch_pattern.size())
        {
            auto csirs_lp_it = csirs_object.launch_pattern[c_plane_info.launch_pattern_slot].find(cell_index);
            if (csirs_lp_it != csirs_object.launch_pattern[c_plane_info.launch_pattern_slot].end())
            {
                int csirs_tv_idx = csirs_lp_it->second;
                if (csirs_tv_idx < static_cast<int>(csirs_object.tv_info.size()))
                {
                    const auto& csirs_tv = csirs_object.tv_info[csirs_tv_idx];
                    int remask_idx = c_plane_info.startSym * cell_configs[cell_index].ulGridSize
                                   + section_info.startPrbc;
                    if (remask_idx >= 0 &&
                        remask_idx < static_cast<int>(csirs_tv.csirsREMaskArray.size()))
                    {
                        uint16_t csirs_mask = csirs_tv.csirsREMaskArray[remask_idx];
                        csirs_pdsch_mask = (~csirs_mask) & 0xFFF;
                    }
                    if (csirs_pdsch_mask == 0xFFF && remask_idx >= 0 &&
                        remask_idx < static_cast<int>(csirs_tv.csirsREMaskArrayTRSNZP.size()))
                    {
                        uint16_t csirs_mask = csirs_tv.csirsREMaskArrayTRSNZP[remask_idx];
                        csirs_pdsch_mask = (~csirs_mask) & 0xFFF;
                    }
                }
            }
        }

        // Check 1: CSI-RS puncturing only (full mask TV, wire equals PDSCH complement)
        mismatch_explained = (section_reMask == csirs_pdsch_mask);

        // Try precise ext_info-based checks (Checks 2 & 3)
        uint32_t n_groups = 0;
        uint16_t group_mask[2] = {0, 0};
        bool has_ext = false;
        if (!mismatch_explained && matched_mod_comp != nullptr && matched_msg_idx >= 0)
        {
            auto tv_it = matched_mod_comp->global_msg_idx_to_tv_idx.find(matched_msg_idx);
            if (tv_it != matched_mod_comp->global_msg_idx_to_tv_idx.end())
            {
                int idx = tv_it->second;
                if (idx >= 0 && idx < static_cast<int>(matched_mod_comp->mod_comp_ext_info.size()))
                {
                    const auto& ext = matched_mod_comp->mod_comp_ext_info[idx];
                    if (ext.valid)
                    {
                        has_ext = true;
                        n_groups = std::min(ext.n_mask, 2u);
                        for (uint32_t g = 0; g < n_groups; g++)
                            group_mask[g] = ext.mc_scale_re_mask[g];
                    }
                }
            }
        }

        if (!mismatch_explained && has_ext)
        {
            // Check 2: SE5 sub-grouping only (wire matches a per-group mask)
            for (uint32_t g = 0; g < n_groups; g++)
            {
                if (section_reMask == group_mask[g])
                {
                    mismatch_explained = true;
                    break;
                }
            }
            // Check 3: combined CSI-RS puncturing + SE5 sub-grouping
            if (!mismatch_explained && csirs_pdsch_mask != 0xFFF)
            {
                for (uint32_t g = 0; g < n_groups; g++)
                {
                    uint16_t expected = group_mask[g] & csirs_pdsch_mask;
                    if (section_reMask == expected)
                    {
                        mismatch_explained = true;
                        break;
                    }
                }
            }
        }

        // Check 4: sub-grouping fallback when ext_info is unavailable.
        // If the TV reMask is not 0xFFF, it already represents a grouped mask
        // (e.g. SE4 CSI-RS per-port masks, or SE5 combined group masks).  The DU
        // may split it into per-group sections, each a subset.  The superset
        // fallback already guarantees wire is a subset of TV, so any non-zero
        // wire reMask is accepted here.
        if (!mismatch_explained && !has_ext && matched_tv_reMask != 0xFFF)
        {
            mismatch_explained = (section_reMask != 0);
        }

        if (!mismatch_explained)
        {
            re_cons("ModComp reMask mismatch: Cell {} F{}S{}S{} sym {} eAxC {} "
                    "sec_id {} wire=0x{:03x} TV=0x{:03x} startPrbc {} numPrbc {}",
                    (uint32_t)cell_index,
                    (uint32_t)c_plane_info.fss.frameId,
                    (uint32_t)c_plane_info.fss.subframeId,
                    (uint32_t)c_plane_info.fss.slotId,
                    (uint32_t)c_plane_info.startSym,
                    (uint32_t)c_plane_info.eaxcId,
                    (uint32_t)section_info.section_id,
                    (uint32_t)section_reMask,
                    (uint32_t)matched_tv_reMask,
                    (uint32_t)section_info.startPrbc,
                    (uint32_t)section_info.numPrbc);
            return false;
        }
    }

    // --- TV-based comparison of SE4/SE5 extension values ---
    if (matched_mod_comp == nullptr || matched_msg_idx < 0)
        return true;

    auto tv_it = matched_mod_comp->global_msg_idx_to_tv_idx.find(matched_msg_idx);
    if (tv_it == matched_mod_comp->global_msg_idx_to_tv_idx.end())
        return true;

    int internal_idx = tv_it->second;
    if (internal_idx < 0 || internal_idx >= static_cast<int>(matched_mod_comp->mod_comp_ext_info.size()))
        return true;

    const auto& expected = matched_mod_comp->mod_comp_ext_info[internal_idx];
    if (!expected.valid)
        return true;

    bool comparison_ok = true;

    for (int ei = 0; ei < section_info.ext_infos_size; ei++)
    {
        const auto& ext = section_info.ext_infos[ei];

        if (ext.ext_type == ORAN_CMSG_SECTION_EXT_TYPE_4 && expected.ext_type == ORAN_CMSG_SECTION_EXT_TYPE_4)
        {
            auto* se4_hdr = reinterpret_cast<oran_cmsg_sect_ext_type_4*>(ext.ext_ptr + sizeof(oran_cmsg_ext_hdr));
            uint16_t wire_scaler = se4_hdr->modCompScalor.get();
            uint16_t wire_csf = se4_hdr->csf.get();

            if (wire_scaler != expected.mc_scale_offset_encoded[0])
            {
                re_cons("F{}S{}S{} Sym {} eAxCID {} SE4 modCompScaler mismatch: wire=0x{:04x} TV=0x{:04x}",
                        c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId,
                        c_plane_info.startSym, c_plane_info.eaxcId,
                        wire_scaler, expected.mc_scale_offset_encoded[0]);
                comparison_ok = false;
            }
            if (wire_csf != expected.csf[0])
            {
                re_cons("F{}S{}S{} Sym {} eAxCID {} SE4 csf mismatch: wire={} TV={}",
                        c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId,
                        c_plane_info.startSym, c_plane_info.eaxcId,
                        wire_csf, expected.csf[0]);
                comparison_ok = false;
            }
        }
        else if (ext.ext_type == ORAN_CMSG_SECTION_EXT_TYPE_5 && expected.ext_type == ORAN_CMSG_SECTION_EXT_TYPE_5)
        {
            oran_cmsg_sect_ext_type_5 se5_copy;
            memcpy(&se5_copy, ext.ext_ptr + sizeof(oran_cmsg_ext_hdr), sizeof(oran_cmsg_sect_ext_type_5));
            uint64_t se5_bitfield_val;
            memcpy(&se5_bitfield_val, reinterpret_cast<uint8_t*>(&se5_copy) + sizeof(se5_copy.extLen), sizeof(uint64_t));
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
            se5_bitfield_val = __builtin_bswap64(se5_bitfield_val);
#endif
            memcpy(reinterpret_cast<uint8_t*>(&se5_copy) + sizeof(se5_copy.extLen), &se5_bitfield_val, sizeof(uint64_t));

            struct {
                uint16_t re_mask;
                uint16_t offset;
                uint32_t csf;
            } wire_groups[2] = {
                {static_cast<uint16_t>(se5_copy.mcScaleReMask_1), static_cast<uint16_t>(se5_copy.mcScaleOffset_1), static_cast<uint32_t>(se5_copy.csf_1)},
                {static_cast<uint16_t>(se5_copy.mcScaleReMask_2), static_cast<uint16_t>(se5_copy.mcScaleOffset_2), static_cast<uint32_t>(se5_copy.csf_2)}
            };

            for (uint32_t g = 0; g < expected.n_mask && g < 2; g++)
            {
                if (wire_groups[g].re_mask != expected.mc_scale_re_mask[g])
                {
                    re_cons("F{}S{}S{} Sym {} eAxCID {} SE5 mcScaleReMask_{} mismatch: wire=0x{:03x} TV=0x{:03x}",
                            c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId,
                            c_plane_info.startSym, c_plane_info.eaxcId, g + 1,
                            wire_groups[g].re_mask, expected.mc_scale_re_mask[g]);
                    comparison_ok = false;
                }

                if (wire_groups[g].offset != expected.mc_scale_offset_encoded[g])
                {
                    re_cons("F{}S{}S{} Sym {} eAxCID {} SE5 mcScaleOffset_{} mismatch: wire=0x{:04x} TV=0x{:04x}",
                            c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId,
                            c_plane_info.startSym, c_plane_info.eaxcId, g + 1,
                            wire_groups[g].offset, expected.mc_scale_offset_encoded[g]);
                    comparison_ok = false;
                }

                if (wire_groups[g].csf != expected.csf[g])
                {
                    re_cons("F{}S{}S{} Sym {} eAxCID {} SE5 csf_{} mismatch: wire={} TV={}",
                            c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId,
                            c_plane_info.startSym, c_plane_info.eaxcId, g + 1,
                            wire_groups[g].csf, expected.csf[g]);
                    comparison_ok = false;
                }
            }
        }
        // When wire and TV use different extension types (SE4 vs SE5), the
        // scaler computation semantics differ, so value comparison is skipped.
    }

    return comparison_ok;
}

/**
 * @brief Validates reMask for C-plane sections based on CSIRS and PDSCH inclusion
 *
 * This function performs comprehensive reMask validation for different scenarios:
 * - mMIMO enabled: CSIRS only, PDSCH only, or both CSIRS and PDSCH
 * - mMIMO disabled: PDSCH with CSIRS pairing, or CSIRS only
 *
 * The validation checks that the received reMask values match expected values
 * based on the TV object data and channel type inclusion.
 *
 * @param[in,out] c_plane_info C-plane information containing sections to validate
 * @param[in] cell_index Cell identifier for the validation
 * @param[in] is_pdsch_included Pre-computed flag indicating PDSCH PRB overlap with this C-plane message
 */
void RU_Emulator::validate_remask(oran_c_plane_info_t& c_plane_info, int cell_index, bool is_pdsch_included)
{
    uint32_t csirs_remask_mismatch = 0, pdsch_remask_mismatch = 0, missing_pdsch_section = 0;
    uint16_t expected_reMask = 0xFFF;

    int remask_idx_base = c_plane_info.startSym * cell_configs[cell_index].ulGridSize;
    c_plane_info.tv_index = csirs_object.launch_pattern[c_plane_info.launch_pattern_slot][cell_index];
    auto &tv_info = csirs_object.tv_info[c_plane_info.tv_index];

    if (opt_enable_mmimo)
    {
        int tv_idx = csirs_object.launch_pattern[c_plane_info.launch_pattern_slot][cell_index];
        bool is_csirs_included = c_plane_info.eaxcId_index < csirs_object.tv_info[tv_idx].numFlowsArray[c_plane_info.startSym];

        if (!is_csirs_included && is_pdsch_included)
        {
            // PDSCH only scenario
            if (c_plane_info.section_infos_size > 1)
            {
                expected_reMask = 0xFFF;
                for (int sec_idx = 0; sec_idx < c_plane_info.section_infos_size; sec_idx++)
                {
                    int remask_idx = remask_idx_base + c_plane_info.section_infos[sec_idx].startPrbc;
                    if (c_plane_info.section_infos[sec_idx].reMask != expected_reMask)
                    {
                        pdsch_remask_mismatch++;
                        c_plane_info.section_infos[sec_idx].error_status = -1;
                    }
                }
                if (pdsch_remask_mismatch)
                {
                    re_cons("pdsch_remask_mismatch: {}", pdsch_remask_mismatch);
                    re_cons("cell_index: {}, symbol: {}, c_plane_info.tv_index {} , remask_idx_base {} ", cell_index, c_plane_info.startSym, c_plane_info.tv_index, remask_idx_base);
                    for (int sec_idx = 0; sec_idx < c_plane_info.section_infos_size; sec_idx++)
                    {
                        int remask_idx = remask_idx_base + c_plane_info.section_infos[sec_idx].startPrbc;
                        re_cons("section_id: {}, startPrbc: {}, numPrbc: {}, expected pdsch_remask 0x{:x}, received pdsch_remask 0x{:x} ", c_plane_info.section_infos[sec_idx].section_id, c_plane_info.section_infos[sec_idx].startPrbc, c_plane_info.section_infos[sec_idx].numPrbc, expected_reMask, c_plane_info.section_infos[sec_idx].reMask);
                    }
                }
            }
        }
        else if (is_csirs_included && is_pdsch_included)
        {
            // Both CSIRS and PDSCH scenario
            if (c_plane_info.section_infos_size > 1)
            {
                auto pdsch_tv_index = pdsch_object.launch_pattern[c_plane_info.launch_pattern_slot][cell_index];
                auto &pdsch_tv_info = pdsch_object.tv_info[pdsch_tv_index];
                uint16_t invalid_sections[2][10][3]; // channel_type:invalid_sect_num:invalid_info

                for (int sec_idx = 0; sec_idx < c_plane_info.section_infos_size; sec_idx++)
                {
                    auto start_prb = c_plane_info.section_infos[sec_idx].startPrbc;
                    auto num_prbs = c_plane_info.section_infos[sec_idx].numPrbc;
                    int remask_idx = remask_idx_base + start_prb;
                    expected_reMask = tv_info.csirsREMaskArray[remask_idx];

                    bool pdsch_section = pdsch_tv_info.prb_num_flow_map[c_plane_info.startSym][start_prb] & ((uint64_t)1 << c_plane_info.eaxcId_index);
                    bool csirs_section = csirs_object.tv_info[tv_idx].prb_map[c_plane_info.startSym][start_prb];

                    if (csirs_section)
                    {
                        // CSIRS section validation -- subset match for per-port modcomp reMasks
                        uint16_t wire_csirs = c_plane_info.section_infos[sec_idx].reMask;
                        if (wire_csirs == 0 || (wire_csirs & expected_reMask) != wire_csirs)
                        {
                            invalid_sections[1][csirs_remask_mismatch][0] = c_plane_info.section_infos[sec_idx].section_id;
                            invalid_sections[1][csirs_remask_mismatch][1] = expected_reMask;
                            invalid_sections[1][csirs_remask_mismatch][2] = wire_csirs;
                            csirs_remask_mismatch++;
                            c_plane_info.section_infos[sec_idx].error_status = -1;
                        }
                        if (pdsch_section && (sec_idx + 1 >= c_plane_info.section_infos_size || c_plane_info.section_infos[sec_idx].section_id != c_plane_info.section_infos[sec_idx + 1].section_id))
                        {
                            missing_pdsch_section++;
                        }
                        else if(pdsch_section)
                        {
                            sec_idx++;
                        }
                    }

                    if (pdsch_section)
                    {
                        // PDSCH section validation
                        if (c_plane_info.section_infos[sec_idx].reMask != ((~expected_reMask) & 0xFFF))
                        {
                            invalid_sections[0][pdsch_remask_mismatch][0] = c_plane_info.section_infos[sec_idx].section_id;
                            invalid_sections[0][pdsch_remask_mismatch][1] = ((~expected_reMask) & 0xFFF);
                            invalid_sections[0][pdsch_remask_mismatch][2] = c_plane_info.section_infos[sec_idx].reMask;
                            pdsch_remask_mismatch++;
                            c_plane_info.section_infos[sec_idx].error_status = -1;
                        }
                    }
                }

                if (csirs_remask_mismatch || pdsch_remask_mismatch || missing_pdsch_section)
                {
                    re_cons("pdsch_remask_mismatch: {}, csirs_remask_mismatch: {}, missing_pdsch_section: {}", pdsch_remask_mismatch, csirs_remask_mismatch, missing_pdsch_section);
                    re_cons("cell_index: {}, symbol: {}, c_plane_info.tv_index {} , remask_idx_base {} ", cell_index, c_plane_info.startSym, c_plane_info.tv_index, remask_idx_base);
                    for (int i = 0; i < pdsch_remask_mismatch; i++)
                    {
                        re_cons("section_id: {}, expected pdsch_remask 0x{:x}, received pdsch_remask 0x{:x} ", invalid_sections[0][i][0], invalid_sections[0][i][1], invalid_sections[0][i][2]);
                    }
                    for (int i = 0; i < csirs_remask_mismatch; i++)
                    {
                        re_cons("section_id: {}, expected csirs_remask 0x{:x}, received csirs_remask 0x{:x} ", invalid_sections[1][i][0], invalid_sections[1][i][1], invalid_sections[1][i][2]);
                    }
                    for (int sec_idx = 0; sec_idx < c_plane_info.section_infos_size; sec_idx++)
                    {
                        re_cons("Cell {} SEC1 C-Plane: F{}S{}S{} sym {} eAxC {} sec_id {} remask 0x{:x} startPrbc {} numPrbc {} numSymbol {} beamId {}",
                                cell_index,
                                c_plane_info.fss.frameId,
                                c_plane_info.fss.subframeId,
                                c_plane_info.fss.slotId,
                                c_plane_info.startSym,
                                c_plane_info.eaxcId,
                                c_plane_info.section_infos[sec_idx].section_id,
                                c_plane_info.section_infos[sec_idx].reMask,
                                c_plane_info.section_infos[sec_idx].startPrbc,
                                c_plane_info.section_infos[sec_idx].numPrbc,
                                c_plane_info.section_infos[sec_idx].numSymbol,
                                c_plane_info.section_infos[sec_idx].beamId);
                    }
                }
            }
        }
        else if (is_csirs_included && !is_pdsch_included)
        {
            // CSIRS only scenario -- subset match for per-port modcomp reMasks
            for (int sec_idx = 0; sec_idx < c_plane_info.section_infos_size; sec_idx++)
            {
                int remask_idx = remask_idx_base + c_plane_info.section_infos[sec_idx].startPrbc;
                expected_reMask = tv_info.csirsREMaskArrayTRSNZP[remask_idx];
                uint16_t wire_csirs = c_plane_info.section_infos[sec_idx].reMask;
                if (wire_csirs == 0 || (wire_csirs & expected_reMask) != wire_csirs)
                {
                    csirs_remask_mismatch++;
                    c_plane_info.section_infos[sec_idx].error_status = -1;
                }
            }
            if (csirs_remask_mismatch)
            {
                re_cons("csirs_remask_mismatch: {}", csirs_remask_mismatch);
                re_cons("cell_index: {}, symbol: {}, c_plane_info.tv_index {} , remask_idx_base {} ", cell_index, c_plane_info.startSym, c_plane_info.tv_index, remask_idx_base);
                for (int sec_idx = 0; sec_idx < c_plane_info.section_infos_size; sec_idx++)
                {
                    int remask_idx = remask_idx_base + c_plane_info.section_infos[sec_idx].startPrbc;
                    expected_reMask = tv_info.csirsREMaskArrayTRSNZP[remask_idx];
                    re_cons("section_id: {}, startPrbc: {}, numPrbc: {}, expected csirs_remask 0x{:x}, received csirs_remask 0x{:x}", c_plane_info.section_infos[sec_idx].section_id, c_plane_info.section_infos[sec_idx].startPrbc, c_plane_info.section_infos[sec_idx].numPrbc, expected_reMask, c_plane_info.section_infos[sec_idx].reMask);
                }
            }
        }
    }
    else
    {
        // mMIMO disabled scenarios
        if (is_pdsch_included)
        {
            // PDSCH with CSIRS pairing
            if (c_plane_info.section_infos_size > 1)
            {
                for (int sec_idx = 0; sec_idx < c_plane_info.section_infos_size; sec_idx += 2)
                {
                    int remask_idx = remask_idx_base + c_plane_info.section_infos[sec_idx + 1].startPrbc;
                    expected_reMask = tv_info.csirsREMaskArray[remask_idx];
                    if (c_plane_info.section_infos[sec_idx].reMask != ((~expected_reMask) & 0xFFF))
                    {
                        pdsch_remask_mismatch++;
                        c_plane_info.section_infos[sec_idx].error_status = -1;
                    }
                    // Subset match for per-port modcomp reMasks
                    uint16_t wire_csirs = c_plane_info.section_infos[sec_idx + 1].reMask;
                    if (wire_csirs == 0 || (wire_csirs & expected_reMask) != wire_csirs)
                    {
                        csirs_remask_mismatch++;
                        c_plane_info.section_infos[sec_idx + 1].error_status = -1;
                    }
                }
                if (csirs_remask_mismatch || pdsch_remask_mismatch)
                {
                    re_cons("pdsch_remask_mismatch: {}, csirs_remask_mismatch: {}", pdsch_remask_mismatch, csirs_remask_mismatch);
                    re_cons("cell_index: {}, symbol: {}, c_plane_info.tv_index {} , remask_idx_base {} ", cell_index, c_plane_info.startSym, c_plane_info.tv_index, remask_idx_base);
                    for (int sec_idx = 0; sec_idx < c_plane_info.section_infos_size; sec_idx += 2)
                    {
                        int remask_idx = remask_idx_base + c_plane_info.section_infos[sec_idx + 1].startPrbc;
                        expected_reMask = tv_info.csirsREMaskArray[remask_idx];
                        re_cons("section_id: {}, startPrbc: {}, numPrbc: {}, expected csirs_remask 0x{:x}, received csirs_remask 0x{:x}, expected pdsch_remask 0x{:x}, received pdsch_remask 0x{:x} ", c_plane_info.section_infos[sec_idx + 1].section_id, c_plane_info.section_infos[sec_idx + 1].startPrbc, c_plane_info.section_infos[sec_idx + 1].numPrbc, expected_reMask, c_plane_info.section_infos[sec_idx + 1].reMask, ((~expected_reMask) & 0xFFF), c_plane_info.section_infos[sec_idx].reMask);
                    }
                }
            }
        }
        else
        {
            // CSIRS only scenario (mMIMO disabled) -- subset match for per-port modcomp reMasks
            for (int sec_idx = 0; sec_idx < c_plane_info.section_infos_size; sec_idx++)
            {
                int remask_idx = remask_idx_base + c_plane_info.section_infos[sec_idx].startPrbc;
                expected_reMask = tv_info.csirsREMaskArrayTRSNZP[remask_idx];
                uint16_t wire_csirs = c_plane_info.section_infos[sec_idx].reMask;
                if (wire_csirs == 0 || (wire_csirs & expected_reMask) != wire_csirs)
                {
                    csirs_remask_mismatch++;
                    c_plane_info.section_infos[sec_idx].error_status = -1;
                }
            }
            if (csirs_remask_mismatch)
            {
                re_cons("csirs_remask_mismatch: {}", csirs_remask_mismatch);
                re_cons("cell_index: {}, symbol: {}, c_plane_info.tv_index {} , remask_idx_base {} ", cell_index, c_plane_info.startSym, c_plane_info.tv_index, remask_idx_base);
                for (int sec_idx = 0; sec_idx < c_plane_info.section_infos_size; sec_idx++)
                {
                    int remask_idx = remask_idx_base + c_plane_info.section_infos[sec_idx].startPrbc;
                    expected_reMask = tv_info.csirsREMaskArrayTRSNZP[remask_idx];
                    re_cons("section_id: {}, startPrbc: {}, numPrbc: {}, expected csirs_remask 0x{:x}, received csirs_remask 0x{:x}", c_plane_info.section_infos[sec_idx].section_id, c_plane_info.section_infos[sec_idx].startPrbc, c_plane_info.section_infos[sec_idx].numPrbc, expected_reMask, c_plane_info.section_infos[sec_idx].reMask);
                }
            }
        }
    }
}

inline void find_DL_eAxC_index(oran_c_plane_info_t& c_plane_info, cell_config& cell_config)
{
    c_plane_info.eaxcId_index = -1;
    c_plane_info.valid_eaxcId = false;
    auto it = find(cell_config.eAxC_DL.begin(), cell_config.eAxC_DL.end(), c_plane_info.eaxcId);
    c_plane_info.eaxcId_index = it - cell_config.eAxC_DL.begin();
    if (c_plane_info.eaxcId_index < cell_config.num_dl_flows)
    {
            c_plane_info.valid_eaxcId = true;
    }
    else
    {
        c_plane_info.eaxcId_index = -1;
        c_plane_info.valid_eaxcId = false;
    }
}

/**
 * @brief Validate a single CSI-RS section's beam ID against non-ZP CSI-RS beam ID sets.
 *
 * Uses modulo indexing (ap_idx % beams_array_size) matching cuphydriver's CSI-RS
 * beam assignment. Accepts a match from any set (TRS or NZP CSI-RS).
 */
void RU_Emulator::validate_dl_beamid_csirs(const std::vector<std::vector<uint16_t>>* csirs_beam_id_sets,
                                            oran_c_plane_info_t& c_plane_info, int sec_idx,
                                            size_t num_dl_flows, int cell_index)
{
    if (csirs_beam_id_sets == nullptr || csirs_beam_id_sets->empty())
    {
        return;
    }
    uint16_t actual = c_plane_info.section_infos[sec_idx].beamId;
    bool matched = false;
    for (const auto& beam_set : *csirs_beam_id_sets)
    {
        if (beam_set.empty())
        {
            continue;
        }
        // CSI-RS: cuphydriver uses beam_id = beams_array[ap_idx % beams_array_size]
        uint16_t expected_val = beam_set[0];
        if (c_plane_info.eaxcId_index >= 0 &&
            static_cast<size_t>(c_plane_info.eaxcId_index) < num_dl_flows)
        {
            size_t beam_idx = static_cast<size_t>(c_plane_info.eaxcId_index) % beam_set.size();
            expected_val = beam_set[beam_idx];
        }
        if (actual == expected_val)
        {
            matched = true;
            break;
        }
    }
    beamid_dl_total_counters[cell_index]++;
    if (!matched)
    {
        beamid_dl_error_counters[cell_index]++;
        re_warn("DL C-plane beam ID mismatch: cell {} eAxC {} F{}S{}S{} section {} beamId {} (CSI-RS, no matching beam set)",
                cell_index, c_plane_info.eaxcId,
                c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId,
                sec_idx, actual);
        if (opt_dlc_tb)
        {
            c_plane_info.section_infos[sec_idx].error_status = -1;
        }
    }
}

/**
 * @brief Validate a single non-CSI-RS section's beam ID (PDSCH/PBCH/PDCCH).
 *
 * PBCH/SSB uses membership check against all SSB PDU beam IDs.
 * PDSCH/PDCCH uses positional division mapping (beam_repeat_interval).
 */
void RU_Emulator::validate_dl_beamid_non_csirs(const std::vector<uint16_t>* non_csirs_beam_ids,
                                                bool non_csirs_is_pbch,
                                                oran_c_plane_info_t& c_plane_info, int sec_idx,
                                                size_t num_dl_flows, int cell_index)
{
    if (non_csirs_beam_ids == nullptr || non_csirs_beam_ids->empty())
    {
        return;
    }

    uint16_t actual = c_plane_info.section_infos[sec_idx].beamId;
    bool beam_ok = false;

    if (non_csirs_is_pbch)
    {
        // PBCH/SSB: multiple SSB beams share the same eAxC with different
        // beam IDs (portMask-based assignment in cuphydriver). Validate by
        // checking if the actual beam ID is among any SSB PDU's beam IDs.
        beam_ok = std::find(non_csirs_beam_ids->begin(),
                            non_csirs_beam_ids->end(), actual) != non_csirs_beam_ids->end();
    }
    else
    {
        // PDSCH/PDCCH: positional beam_repeat_interval mapping
        uint16_t expected = (*non_csirs_beam_ids)[0];
        if (c_plane_info.eaxcId_index >= 0 &&
            static_cast<size_t>(c_plane_info.eaxcId_index) < num_dl_flows)
        {
            size_t beams_array_size = std::min(non_csirs_beam_ids->size(), num_dl_flows);
            size_t beam_repeat_interval = (beams_array_size > 0) ? (num_dl_flows / beams_array_size) : 1;

            size_t beam_idx = static_cast<size_t>(c_plane_info.eaxcId_index) / beam_repeat_interval;
            if (beam_idx < non_csirs_beam_ids->size())
            {
                expected = (*non_csirs_beam_ids)[beam_idx];
            }
        }
        beam_ok = (actual == expected);
        if (!beam_ok)
        {
            re_warn("DL C-plane beam ID mismatch: cell {} eAxC {} F{}S{}S{} section {} beamId {} expected {}",
                    cell_index, c_plane_info.eaxcId,
                    c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId,
                    sec_idx, actual, expected);
        }
    }

    beamid_dl_total_counters[cell_index]++;
    if (!beam_ok)
    {
        beamid_dl_error_counters[cell_index]++;
        if (non_csirs_is_pbch)
        {
            re_warn("DL C-plane beam ID mismatch: cell {} eAxC {} F{}S{}S{} section {} beamId {} (PBCH, not in SSB beam set)",
                    cell_index, c_plane_info.eaxcId,
                    c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId,
                    sec_idx, actual);
        }
        if (opt_dlc_tb)
        {
            c_plane_info.section_infos[sec_idx].error_status = -1;
        }
    }
}

// Template sectionId validation functions moved to sectionid_validation.cpp (cold)

/**
 * @brief Verifies downlink C-plane messages with section validation and compression handling
 *
 * This function handles the complete processing of downlink C-plane messages including:
 * - Finding DL eAxC index
 * - Managing mMIMO PRB tracking
 * - Handling beamforming if enabled
 * - Validating section types and processing sections
 * - Validating sectionId (range 0-4095 and duplicate consistency) for default coupling
 * - Validating modulation compression for applicable sections
 * - Validating beam IDs and extensions
 * - Performing reMask validation for CSIRS when enabled
 *
 * @param[in,out] c_plane_info C-plane information containing the message to process
 * @param[in] cell_index Cell identifier for the processing
 * @param[in] mbuf_payload Pointer to the message buffer payload
 * @param[in] msg_info Message receive information for logging
 * @param[in,out] fss_pdsch_prb_seen Array of nested maps tracking PRB usage for mMIMO
 */
void RU_Emulator::verify_dl_cplane_content(oran_c_plane_info_t& c_plane_info,
                                     const int cell_index,
                                     uint8_t* mbuf_payload,
                                     aerial_fh::MsgReceiveInfo& msg_info,
                                     FssPdschPrbSeenArray& fss_pdsch_prb_seen)
{
    
    find_DL_eAxC_index(c_plane_info, cell_configs[cell_index]);

    if (opt_enable_mmimo && fss_pdsch_prb_seen[cell_index][c_plane_info.eaxcId_index].find(c_plane_info.fss.frameId) == fss_pdsch_prb_seen[cell_index][c_plane_info.eaxcId_index].end())
    {
        fss_pdsch_prb_seen[cell_index][c_plane_info.eaxcId_index].clear();
    }

    if (opt_beamforming == RE_ENABLED)
    {
        c_plane_info.numberOfSections = oran_cmsg_get_number_of_sections(mbuf_payload);
        print_c_plane_beamid(msg_info, c_plane_info.numberOfSections, ORAN_CMSG_SECTION_TYPE_1);
    }

    if(c_plane_info.section_type != ORAN_CMSG_SECTION_TYPE_1)
    {
        if (opt_dlc_tb)
        {
            // Counters gated by opt_dlc_tb: the else branch throws an error, so counters
            // are only meaningful on the graceful-handling path.
            error_dl_section_counters[cell_index] += c_plane_info.numberOfSections;
            total_dl_section_counters[cell_index] += c_plane_info.numberOfSections;
            re_warn("DL C-plane section type error: %d (cell %d)", (int)c_plane_info.section_type, cell_index);
            return;
        }
        else
        {
            do_throw(sb() << "Section type error: "<< (int)c_plane_info.section_type);
        }
    }

    if (unlikely(opt_sectionid_validation == RE_ENABLED))
        sid_dl_validate(c_plane_info, cell_index);

    uint8_t* next = mbuf_payload;
    next += ORAN_CMSG_SECT1_FIELDS_OFFSET;

    struct oran_packet_header_info header_info;
    uint64_t prev_pbch_time;

    if (opt_dlc_tb)
    {
        header_info.fss = c_plane_info.fss;
        header_info.launch_pattern_slot = c_plane_info.launch_pattern_slot;
        header_info.flowValue = c_plane_info.eaxcId;
        header_info.flow_index = c_plane_info.eaxcId_index;
        header_info.payload_len = 0;  // Not applicable for C-plane validation
    }

    for(int sec_idx = 0; sec_idx < c_plane_info.numberOfSections; ++sec_idx)
    {
        oran_c_plane_section_info_t& section_info = c_plane_info.section_infos[sec_idx];
        next += sizeof(oran_cmsg_sect1);

        if (section_info.numPrbc == 0)
        {
            section_info.numPrbc = cell_configs[cell_index].dlGridSize;
        }

        if(cell_configs[cell_index].dl_comp_meth == aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION)
        {
            if (validate_modulation_compression(c_plane_info, section_info, cell_index) == false)
            {
                section_info.error_status = -1;
            }

            if (!section_info.ef)
            {
                if (opt_dlc_tb)
                {
                    section_info.error_status = -1;
                    re_warn("DL C-plane: no section extension detected for modulation compression (cell %d, section %d)", cell_index, sec_idx);
                    continue;
                }
                else
                {
                    do_throw(sb() << "Error: no section extension detected for modulation compression" << "\n");
                }
            }
        }

        if (section_info.ef)
        {
            auto seen = cell_configs[cell_index].dbt_cfg.static_beamIdx_seen;
            if (seen.find(section_info.beamId) != seen.end() && seen[section_info.beamId] == false)
            {
                if (opt_dlc_tb)
                {
                    section_info.error_status = -1;
                    re_warn("DL C-plane: static beam id in section header not seen in SE11 bundle (cell %d, beam %d)", cell_index, section_info.beamId);
                    continue;
                }
                else
                {
                    do_throw(sb() << "Static beam id in section header, but not seen in SE11 bundle before" << "\n");
                }
            }
            next += verify_extensions(c_plane_info, section_info, cell_index);
        }

        if(opt_dlc_tb)
        {
            if(sec_idx && section_info.section_id == c_plane_info.section_infos[sec_idx-1].section_id)
            {
                continue;
            }

            header_info.sectionId = section_info.section_id;
            header_info.rb = section_info.rb;
            header_info.startPrb = section_info.startPrbc;
            header_info.numPrb = section_info.numPrbc;

            for (int sym = c_plane_info.startSym; sym < c_plane_info.startSym + section_info.numSymbol; sym++)
            {
                header_info.symbolId = sym;
                validate_dl_channels(cell_index, c_plane_info.launch_pattern_slot, header_info, nullptr, prev_pbch_time);
            }
        }
    }

    // Shared section classification for reMask and beam ID validation.
    // Computed once and reused by both validate_remask() and beam ID validation
    // to avoid redundant c_plane_channel_type_checking() calls and reMask scans.
    bool has_csirs_sections = false;
    bool is_pdsch_included = false;
    if (opt_csirs_validation || (opt_beamid_validation == RE_ENABLED))
    {
        for (int sec_idx = 0; sec_idx < c_plane_info.numberOfSections; ++sec_idx)
        {
            if (c_plane_info.section_infos[sec_idx].reMask != 0xFFF)
            {
                has_csirs_sections = true;
                break;
            }
        }
        is_pdsch_included = c_plane_channel_type_checking(c_plane_info, cell_index, pdsch_object);
    }

    if (opt_csirs_validation &&
        cell_configs[cell_index].dl_comp_meth != aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION)
    {
        bool remask_validation = c_plane_channel_type_checking(c_plane_info, cell_index, csirs_object);
        if (remask_validation)
        {
            validate_remask(c_plane_info, cell_index, is_pdsch_included);
        }
    }

    // DL section-header beam ID validation (4T4R and mMIMO non-SE11 sections).
    // Handles three message types:
    //   1. non-CSI-RS (all reMask == 0xFFF): validate all sections against PDSCH/PBCH/PDCCH TV
    //   2. Paired PDSCH+CSI-RS (even=PDSCH, odd=CSI-RS): validate each section against its channel's TV
    //   3. Only CSI-RS (all reMask != 0xFFF, no PDSCH): validate all sections against CSI-RS TV
    //
    // In mMIMO mode, sections carrying SE11 extensions use beamId=0x7FFF as a
    // placeholder; their beam IDs are validated inside verify_extType11() instead.
    // Only non-SE11 sections (beamId != 0x7FFF) are checked here.
    if (opt_beamid_validation == RE_ENABLED)
    {
        // Resolve expected beam IDs for non-CSI-RS channels (PDSCH/PBCH/PDCCH)
        const std::vector<uint16_t>* non_csirs_beam_ids = nullptr;
        const std::vector<tv_info::pdu_beam_entry>* non_csirs_per_pdu_beams = nullptr;
        bool non_csirs_is_pbch = false;
        for (auto* tv_obj : dl_tv_objs)
        {
            if (tv_obj == &csirs_object)
            {
                continue;
            }
            if (!c_plane_channel_type_checking(c_plane_info, cell_index, *tv_obj))
            {
                continue;
            }
            int tv_idx = tv_obj->launch_pattern[c_plane_info.launch_pattern_slot].at(cell_index);
            if (tv_idx < tv_obj->tv_info.size() && !tv_obj->tv_info[tv_idx].expected_beam_ids.empty())
            {
                if (non_csirs_beam_ids == nullptr)
                {
                    non_csirs_beam_ids = &tv_obj->tv_info[tv_idx].expected_beam_ids;
                    non_csirs_is_pbch = (tv_obj == &pbch_object);
                    if (!tv_obj->tv_info[tv_idx].per_pdu_beam_ids.empty())
                    {
                        non_csirs_per_pdu_beams = &tv_obj->tv_info[tv_idx].per_pdu_beam_ids;
                    }
                }
                break;
            }
        }

        // Resolve expected beam IDs for CSI-RS (only when CSI-RS sections are present)
        // Multiple non-ZP CSI-RS types (TRS, NZP) may coexist in a single TV,
        // each with different beam IDs. Store all sets for multi-set validation.
        const std::vector<std::vector<uint16_t>>* csirs_beam_id_sets = nullptr;
        if (has_csirs_sections && c_plane_channel_type_checking(c_plane_info, cell_index, csirs_object))
        {
            int tv_idx = csirs_object.launch_pattern[c_plane_info.launch_pattern_slot].at(cell_index);
            if (tv_idx < csirs_object.tv_info.size() && !csirs_object.tv_info[tv_idx].csirs_beam_id_sets.empty())
            {
                csirs_beam_id_sets = &csirs_object.tv_info[tv_idx].csirs_beam_id_sets;
            }
        }

        // Determine message type using shared classification:
        // paired = CSI-RS sections present AND PDSCH PRB overlap
        bool is_paired = has_csirs_sections && is_pdsch_included;

        size_t num_dl_flows = cell_configs[cell_index].eAxC_DL.size();

        // Per-section beam ID validation
        for (int sec_idx = 0; sec_idx < c_plane_info.numberOfSections; ++sec_idx)
        {
            // In mMIMO, skip SE11 sections — their beam IDs live inside the
            // extension bundles and are validated by verify_extType11().
            if (opt_enable_mmimo && c_plane_info.section_infos[sec_idx].beamId == 0x7FFF)
            {
                continue;
            }

            // Determine if this section is CSI-RS based on channel ownership
            bool is_csirs_section = false;
            if (!has_csirs_sections)
                is_csirs_section = false;
            else if (is_paired)
                is_csirs_section = (sec_idx % 2 != 0);
            else
                is_csirs_section = true;

            if (is_csirs_section)
            {
                validate_dl_beamid_csirs(csirs_beam_id_sets, c_plane_info, sec_idx, num_dl_flows, cell_index);
            }
            else
            {
                // For multi-PDU messages, look up per-PDU beam IDs by PRB range
                const std::vector<uint16_t>* section_beam_ids = non_csirs_beam_ids;
                if (non_csirs_per_pdu_beams != nullptr)
                {
                    uint16_t sec_start = c_plane_info.section_infos[sec_idx].startPrbc;
                    for (auto& entry : *non_csirs_per_pdu_beams)
                    {
                        if (sec_start >= entry.startPrb &&
                            sec_start < entry.startPrb + entry.numPrb)
                        {
                            section_beam_ids = &entry.beam_ids;
                            break;
                        }
                    }
                }
                validate_dl_beamid_non_csirs(section_beam_ids, non_csirs_is_pbch, c_plane_info, sec_idx, num_dl_flows, cell_index);
            }
        }
    }

    if (opt_dlc_tb)
    {
        total_dl_section_counters[cell_index] += c_plane_info.numberOfSections;
        for (int sec_idx = 0; sec_idx < c_plane_info.numberOfSections; ++sec_idx)
        {
            if (c_plane_info.section_infos[sec_idx].error_status)
            {
                error_dl_section_counters[cell_index]++;
            }
        }
    }
}

/**
 * @brief Verifies uplink C-plane messages with section validation and processing
 *
 * This function handles the complete processing of uplink C-plane messages including:
 * - UL processing start timing
 * - Section type validation and offset calculation (TYPE_1 and TYPE_3)
 * - Section verification loop with extension handling
 * - Beam ID validation for TYPE_1 sections
 * - Extension verification
 * - UL C-plane info size tracking
 * - Verification timing
 *
 * @param[in,out] c_plane_info C-plane information containing the message to process
 * @param[in] cell_index Cell identifier for the processing
 * @param[in] mbuf_payload Pointer to the message buffer payload
 * @param[in,out] slot_tx Slot transmission info for UL counter tracking
 */
void RU_Emulator::verify_ul_cplane_content(oran_c_plane_info_t& c_plane_info,
                                           const int cell_index,
                                           uint8_t* mbuf_payload,
                                           slot_tx_info& slot_tx)
{
    c_plane_info.ul_processing_start = get_ns();
    uint8_t* next = mbuf_payload;

    switch(c_plane_info.section_type)
    {
        case ORAN_CMSG_SECTION_TYPE_1:
        {
            next += ORAN_CMSG_SECT1_FIELDS_OFFSET;
        }
        break;
        case ORAN_CMSG_SECTION_TYPE_3:
        {
            next += ORAN_CMSG_SECT3_FIELDS_OFFSET;
        }
        break;
        default:
        {
            if (opt_dlc_tb)
            {
                error_ul_section_counters[cell_index] += c_plane_info.numberOfSections;
                total_ul_section_counters[cell_index] += c_plane_info.numberOfSections;
                re_warn("UL C-plane section type not supported: %d (cell %d)", (int)c_plane_info.section_type, cell_index);
                return;
            }
            else
            {
                do_throw(sb() << "Section type not supported: "<< (int)c_plane_info.section_type);
            }
        }
        break;
    }

    c_plane_info.verification_section_start = get_ns();

    if (unlikely(opt_sectionid_validation == RE_ENABLED) && !sid_real_ul(c_plane_info, cell_index))
        return;

    ++slot_tx.ul_c_plane_infos_size;
    c_plane_info.verification_section_end = get_ns();

    if (opt_dlc_tb)
    {
        total_ul_section_counters[cell_index] += c_plane_info.numberOfSections;
        for (auto section_idx = 0; section_idx < c_plane_info.section_infos_size; section_idx++)
        {
            auto &section_info = c_plane_info.section_infos[section_idx];
            if (section_info.ext_infos_size > 0)
            {
                verify_extensions(c_plane_info, section_info, cell_index);
            }

            if (section_info.error_status)
            {
                error_ul_section_counters[cell_index]++;
            }
        }

        for(int sym = 0; sym < ORAN_ALL_SYMBOLS; ++sym)
        {
            increment_section_rx_counter(c_plane_info, cell_index, sym);
        }
    }

    // 4T4R beam ID validation for UL (non-mMIMO path, Section Type 1 only)
    if (opt_beamid_validation == RE_ENABLED && !opt_enable_mmimo &&
        c_plane_info.section_type == ORAN_CMSG_SECTION_TYPE_1 && c_plane_info.tv_object != nullptr)
    {
        for (int sec_idx = 0; sec_idx < c_plane_info.section_infos_size; ++sec_idx)
        {
            // Resolve per-section tv_object when sections carry different channel types.
            auto* ul_tv_obj = c_plane_info.is_mixed_channel ?
                c_plane_info.section_infos[sec_idx].tv_object : c_plane_info.tv_object;
            if (!ul_tv_obj) {
                re_warn("beam_id_validation: null tv_object for section {}, cell {} — skipping", sec_idx, cell_index);
                continue;
            }

            // Resolve per-section flow count and eAxC index: SRS sections use
            // the SRS eAxC list/flow count, all others use the standard UL list.
            auto sec_channel = c_plane_info.is_mixed_channel ?
                c_plane_info.section_infos[sec_idx].channel_type : c_plane_info.channel_type;
            size_t num_flows;
            int16_t eaxc_idx;
            if (sec_channel == ul_channel::SRS)
            {
                num_flows = cell_configs[cell_index].num_valid_SRS_flows;
                auto it = find(cell_configs[cell_index].eAxC_SRS_list.begin(),
                               cell_configs[cell_index].eAxC_SRS_list.end(),
                               c_plane_info.eaxcId);
                eaxc_idx = (it != cell_configs[cell_index].eAxC_SRS_list.end())
                    ? static_cast<int16_t>(it - cell_configs[cell_index].eAxC_SRS_list.begin()) : -1;
            }
            else
            {
                num_flows = cell_configs[cell_index].eAxC_UL.size();
                eaxc_idx = c_plane_info.eaxcId_index;
            }

            if (c_plane_info.launch_pattern_slot >= ul_tv_obj->launch_pattern.size() ||
                ul_tv_obj->launch_pattern[c_plane_info.launch_pattern_slot].find(cell_index) ==
                ul_tv_obj->launch_pattern[c_plane_info.launch_pattern_slot].end())
                continue;

            int tv_idx = ul_tv_obj->launch_pattern[c_plane_info.launch_pattern_slot].at(cell_index);
            if (tv_idx >= static_cast<int>(ul_tv_obj->tv_info.size()) ||
                ul_tv_obj->tv_info[tv_idx].expected_beam_ids.empty())
                continue;

            const auto& tv = ul_tv_obj->tv_info[tv_idx];
            const std::vector<uint16_t>* beam_ids = &tv.expected_beam_ids;

            if (!tv.per_pdu_beam_ids.empty())
            {
                uint16_t sec_start = c_plane_info.section_infos[sec_idx].startPrbc;
                for (auto& entry : tv.per_pdu_beam_ids)
                {
                    if (sec_start >= entry.startPrb &&
                        sec_start < entry.startPrb + entry.numPrb)
                    {
                        beam_ids = &entry.beam_ids;
                        break;
                    }
                }
            }

            size_t beams_sz = std::min(beam_ids->size(), num_flows);
            // Beam repeat interval: how many consecutive eAxC flows share the
            // same beam ID (e.g. 4 flows / 2 beams → beam_repeat_interval 2, so eAxC 0-1 use
            // beam_ids[0] and eAxC 2-3 use beam_ids[1]).
            size_t beam_repeat_interval = (beams_sz > 0) ? (num_flows / beams_sz) : 1;
            uint16_t expected = (*beam_ids)[0];
            if (eaxc_idx >= 0 &&
                static_cast<size_t>(eaxc_idx) < num_flows)
            {
                size_t bidx = static_cast<size_t>(eaxc_idx) / beam_repeat_interval;
                if (bidx < beam_ids->size())
                    expected = (*beam_ids)[bidx];
            }

            beamid_ul_total_counters[cell_index]++;
            if (c_plane_info.section_infos[sec_idx].beamId != expected)
            {
                beamid_ul_error_counters[cell_index]++;
                re_warn("UL C-plane beam ID mismatch: cell {} eAxC {} F{}S{}S{} section {} beamId {} expected {}",
                        cell_index, c_plane_info.eaxcId,
                        c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId,
                        sec_idx, c_plane_info.section_infos[sec_idx].beamId, expected);
                if (opt_dlc_tb)
                {
                    c_plane_info.section_infos[sec_idx].error_status = -1;
                }
            }
        }
    }
}

bool RU_Emulator::c_plane_channel_type_checking(oran_c_plane_info_t &c_plane_info, uint16_t cell_index, struct ul_tv_object& tv_obj)
{
    if (c_plane_info.launch_pattern_slot < tv_obj.launch_pattern.size() && tv_obj.launch_pattern[c_plane_info.launch_pattern_slot].find(cell_index) != tv_obj.launch_pattern[c_plane_info.launch_pattern_slot].end())
    {
        int tv_idx = tv_obj.launch_pattern[c_plane_info.launch_pattern_slot][cell_index];
        if (tv_idx < tv_obj.tv_info.size())
        {
            auto& prb_map  = &tv_obj == &srs_object ? tv_obj.tv_info[tv_idx].fss_prb_map[c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId] : tv_obj.tv_info[tv_idx].prb_map;
            return prb_map[c_plane_info.startSym][c_plane_info.section_infos[0].startPrbc];
        }
    }
    return false;
}

/**
 * @brief Resolve the PRB map for a given UL TV object, cell, and slot.
 *
 * @param[in] c_plane_info  C-plane info containing launch-pattern slot,
 *                          frame/subframe/slot indices, and symbol context.
 * @param[in] cell_index    Cell index used to look up the test-vector entry
 *                          in the launch pattern.
 * @param[in] tv_obj        UL test-vector object (e.g. PUSCH, PUCCH, or SRS).
 *
 * @return Pointer to the matching prb_map_t, or nullptr if the launch-pattern
 *         slot or cell index is out of range.
 *
 * @note When @p tv_obj is the SRS object (&tv_obj == &srs_object) the
 *       function returns the frame-specific fss_prb_map keyed by
 *       frameId / subframeId / slotId; for all other TV objects it returns
 *       the standard prb_map.
 */
const RU_Emulator::prb_map_t* RU_Emulator::resolve_prb_map(oran_c_plane_info_t &c_plane_info, uint16_t cell_index, struct ul_tv_object& tv_obj)
{
    if (c_plane_info.launch_pattern_slot < tv_obj.launch_pattern.size())
    {
        auto it = tv_obj.launch_pattern[c_plane_info.launch_pattern_slot].find(cell_index);
        if (it != tv_obj.launch_pattern[c_plane_info.launch_pattern_slot].end())
        {
            int tv_idx = it->second;
            if (tv_idx < tv_obj.tv_info.size())
            {
                if (&tv_obj == &srs_object)
                {
                    // Use .find() to avoid unordered_map::operator[] inserting
                    // a default all-false prb_map for unseen frame/slot keys.
                    auto frame_it = tv_obj.tv_info[tv_idx].fss_prb_map.find(c_plane_info.fss.frameId);
                    if (frame_it == tv_obj.tv_info[tv_idx].fss_prb_map.end())
                        return nullptr;
                    auto slot_it = frame_it->second.find(c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId);
                    if (slot_it == frame_it->second.end())
                        return nullptr;
                    return &slot_it->second;
                }
                else
                    return &tv_obj.tv_info[tv_idx].prb_map;
            }
        }
    }
    return nullptr;
}

inline bool RU_Emulator::c_plane_channel_type_checking(oran_c_plane_info_t &c_plane_info, uint16_t cell_index, struct dl_tv_object& tv_obj)
{
    if (c_plane_info.launch_pattern_slot < tv_obj.launch_pattern.size() && tv_obj.launch_pattern[c_plane_info.launch_pattern_slot].find(cell_index) != tv_obj.launch_pattern[c_plane_info.launch_pattern_slot].end())
    {
        int tv_idx = tv_obj.launch_pattern[c_plane_info.launch_pattern_slot][cell_index];
        if (tv_idx < tv_obj.tv_info.size())
        {
            auto &prb_map = tv_obj.tv_info[tv_idx].prb_map;
            return prb_map[c_plane_info.startSym][c_plane_info.section_infos[0].startPrbc];
        }
    }
    return false;
}

static inline uint64_t fine_timers_get_ns()
{
#if RU_EM_UL_TIMERS_ENABLE
    return get_ns();
#endif
    return 0;
}

bool RU_Emulator::check_if_drop(uint16_t cell_id, ul_channel ch, const struct fssId& fss)
{
    bool drop = false;
    if(ul_pkts_drop_test[cell_id][(int)ch].single_drop.load())
    {
        //if not drop slot, then just drop this instance
        if(!ul_pkts_drop_test[cell_id][(int)ch].drop_slot.load())
        {
            drop = true;
            ul_pkts_drop_test[cell_id][(int)ch].single_drop.store(false);
        }
        else // drop slot: if it is the first time encountering this fss, set the start_ts and end_ts, else drop packets with this fss if it is before the end_ts
        {
            auto frame_id = ul_pkts_drop_test[cell_id][(int)ch].drop_frame_id.load();
            auto subframe_id = ul_pkts_drop_test[cell_id][(int)ch].drop_subframe_id.load();
            auto slot_id = ul_pkts_drop_test[cell_id][(int)ch].drop_slot_id.load();
            if(fss.frameId == frame_id && fss.subframeId == subframe_id && fss.slotId == slot_id)
            {
                if(!ul_pkts_drop_test[cell_id][(int)ch].drop_slot_ts_set.load())
                {
                    ul_pkts_drop_test[cell_id][(int)ch].drop_slot_start_ts.store(get_ns());
                    ul_pkts_drop_test[cell_id][(int)ch].drop_slot_end_ts.store(get_ns() + 1000000000);
                    ul_pkts_drop_test[cell_id][(int)ch].drop_slot_ts_set.store(true);
                    drop = true;
                }
                else
                {
                    auto now = get_ns();
                    //drop packets if it is before end_ts
                    if(now < ul_pkts_drop_test[cell_id][(int)ch].drop_slot_end_ts.load())
                    {
                        drop = true;
                    }
                    else
                    {
                        ul_pkts_drop_test[cell_id][(int)ch].drop_slot_ts_set.store(false);
                        ul_pkts_drop_test[cell_id][(int)ch].drop_slot_start_ts.store(0);
                        ul_pkts_drop_test[cell_id][(int)ch].drop_slot_end_ts.store(0);
                        ul_pkts_drop_test[cell_id][(int)ch].drop_slot.store(false);
                        ul_pkts_drop_test[cell_id][(int)ch].single_drop.store(false);
                        ul_pkts_drop_test[cell_id][(int)ch].drop_frame_id.store(0);
                        ul_pkts_drop_test[cell_id][(int)ch].drop_subframe_id.store(0);
                        ul_pkts_drop_test[cell_id][(int)ch].drop_slot_id.store(0);
                    }
                }
            }
        }
    }
    else if(ul_pkts_drop_test[cell_id][(int)ch].enabled.load())
    {
        if(ul_pkts_drop_test[cell_id][(int)ch].enabled.load())
        {
            auto& drop_set = ul_pkts_drop_test[cell_id][(int)ch].drop_set;
            if(drop_set.find(ul_pkts_drop_test[cell_id][(int)ch].cnt++) != drop_set.end())
            {
                drop = true;
            }
            ul_pkts_drop_test[cell_id][(int)ch].cnt %= 100;
        }
    }
    return drop;
}

const srs_slot_type_info* get_srs_info(const int launch_pattern_slot,
                                        const srs_slot_type_info& srs_s3_info, 
                                        const srs_slot_type_info& srs_s4_info, 
                                        const srs_slot_type_info& srs_s5_info)
{
    const int slot_type = launch_pattern_slot % 10;
    
    if (slot_type == 3) {
        return &srs_s3_info;
    } else if (slot_type == 4) {
        return &srs_s4_info;
    } else if (slot_type == 5) {
        return &srs_s5_info;
    }
    
    return nullptr;
}

int get_window_index(const int launch_pattern_slot, const int symbol, const int16_t eaxcId_index,
                     const srs_slot_type_info& srs_s3_info, const srs_slot_type_info& srs_s4_info, const srs_slot_type_info& srs_s5_info,
                     const int srs_pacing_eaxcids_per_symbol, const int srs_pacing_eaxcids_per_tx_window)
{
    const srs_slot_type_info* srs_info = get_srs_info(launch_pattern_slot, srs_s3_info, srs_s4_info, srs_s5_info);
    
    if (srs_info != nullptr && srs_info->num_symbols > 0)
    {
        const int txqs_per_symbol = srs_pacing_eaxcids_per_symbol / srs_pacing_eaxcids_per_tx_window;
        const int logical_symbol = symbol - srs_info->first_symbol;
        const int window_within_slot = logical_symbol * txqs_per_symbol + (eaxcId_index / srs_pacing_eaxcids_per_tx_window);
        const int total_window_index = srs_info->txq_base_offset + window_within_slot;
        return total_window_index;
    }
    
    // Invalid slot or no SRS symbols configured
    return -1;
}

int get_txq_index(const ul_channel channel_type, const int launch_pattern_slot, const int symbol, const int16_t eaxcId_index,
                  const int split_srs_txq, const int enable_srs_eaxcid_pacing,
                  const srs_slot_type_info& srs_s3_info, const srs_slot_type_info& srs_s4_info, const srs_slot_type_info& srs_s5_info,
                  const int srs_pacing_eaxcids_per_symbol, const int srs_pacing_eaxcids_per_tx_window)
{
    //RU_TXQ_COUNT = NONSRS count (28) + SRS count (dynamic)
    //RU_TXQ_COUNT = ORAN_ALL_SYMBOLS * 2 + num_srs_txqs
    int txq_index = 0;
    if((channel_type == ul_channel::SRS) && split_srs_txq == 1)
    {
        if(enable_srs_eaxcid_pacing)
        {
            const int total_window_index = get_window_index(launch_pattern_slot, symbol, eaxcId_index,
                                                           srs_s3_info, srs_s4_info, srs_s5_info,
                                                           srs_pacing_eaxcids_per_symbol, srs_pacing_eaxcids_per_tx_window);
            if (total_window_index >= 0)
            {
                txq_index = 2*ORAN_ALL_SYMBOLS + total_window_index;
            }
            else
            {
                // Should not happen if error checking is done properly
                txq_index = 2*ORAN_ALL_SYMBOLS;
            }
        }
        else
        {
            txq_index = 2*ORAN_ALL_SYMBOLS + symbol;
        }
    }
    else
    {
        // Even launch_pattern_slots use TXQs 0-13, odd use 14-27
        txq_index = (launch_pattern_slot & 1) * ORAN_ALL_SYMBOLS + symbol;
    }

    //NVLOGI_FMT(TAG_CP_WORKER_TRACING, "{}, symbol: {}, eaxcId_index: {}, txq_index: {}", channel_type==ul_channel::SRS ? "SRS" : "UL", symbol, eaxcId_index, txq_index);

    return txq_index;
}

uint64_t calculate_tx_time(const int64_t slot_t0, const uint64_t tx_offset, const ul_channel channel_type, const int symbol, 
                           const int16_t eaxcId_index, const int opt_tti_us, const int enable_srs_eaxcid_pacing,
                           const int launch_pattern_slot, const srs_slot_type_info& srs_s3_info, 
                           const srs_slot_type_info& srs_s4_info, const srs_slot_type_info& srs_s5_info,
                           const int srs_pacing_eaxcids_per_symbol, const int srs_pacing_eaxcids_per_tx_window)
{
    if(channel_type == ul_channel::SRS && enable_srs_eaxcid_pacing) {
        const int total_window_index = get_window_index(launch_pattern_slot, symbol, eaxcId_index,
                                                       srs_s3_info, srs_s4_info, srs_s5_info,
                                                       srs_pacing_eaxcids_per_symbol, srs_pacing_eaxcids_per_tx_window);
        
        if (total_window_index >= 0)
        {
            // Get slot-specific time offset
            const srs_slot_type_info* srs_info = get_srs_info(launch_pattern_slot, srs_s3_info, srs_s4_info, srs_s5_info);
            const int64_t slot_time_offset_ns = (srs_info != nullptr) ? srs_info->slot_time_offset_ns : 0;
            
            // Schedule relative to s3's t0, accounting for slot offset and window timing
            return slot_t0 + slot_time_offset_ns + tx_offset + (total_window_index * (opt_tti_us * NS_X_US) / ORAN_ALL_SYMBOLS);
        }
        else
        {
            // Fallback if no valid slot info (should not happen with proper error checking)
            return slot_t0 + tx_offset + (symbol * (opt_tti_us * NS_X_US) / ORAN_ALL_SYMBOLS);
        }
    } else {
        return slot_t0 + tx_offset + (symbol * (opt_tti_us * NS_X_US) / ORAN_ALL_SYMBOLS);
    }
}

int RU_Emulator::handle_sect1_c_plane(oran_c_plane_info_t &c_plane_info, uint16_t cell_index, tx_symbol_timers &timers, aerial_fh::TxqHandle *txqs, aerial_fh::TxRequestHandle* tx_request, int sym, perf_metrics::PerfMetricsAccumulator* profiler)
{
    size_t nb_tx = 0;
    tx_symbol_helper tx_symbol_info;

    timers.tx_symbol_info_copy_start_t = fine_timers_get_ns();
    tx_symbol_info.cell_index = cell_index;
    tx_symbol_info.eaxcId = c_plane_info.eaxcId;
    tx_symbol_info.eaxcId_index = c_plane_info.eaxcId_index;
    tx_symbol_info.valid_eaxcId = c_plane_info.valid_eaxcId;
    tx_symbol_info.fss.frameId = c_plane_info.fss.frameId;
    tx_symbol_info.fss.subframeId = c_plane_info.fss.subframeId;
    tx_symbol_info.fss.slotId = c_plane_info.fss.slotId;
    timers.tx_symbol_info_copy_end_t = fine_timers_get_ns();
    timers.tx_symbol_info_copy_t += timers.tx_symbol_info_copy_end_t - timers.tx_symbol_info_copy_start_t;

    // Error checking for SRS pacing: validate slot and symbols
    if (opt_enable_srs_eaxcid_pacing && c_plane_info.channel_type == ul_channel::SRS)
    {
        const srs_slot_type_info* srs_info = get_srs_info(c_plane_info.launch_pattern_slot, srs_s3_info, srs_s4_info, srs_s5_info);
        const int slot_type = c_plane_info.launch_pattern_slot % 10;

        if (srs_info == nullptr || srs_info->num_symbols == 0)
        {
            NVLOGW_FMT(NVLOG_TAG_BASE_RU_EMULATOR, "SRS encountered in slot {} (slot type {}) but no SRS expected - skipping transmission",
                       c_plane_info.launch_pattern_slot, slot_type);
            return 0;
        }

        // Check if symbols are in expected range for the single symbol being processed
        if (sym < srs_info->first_symbol || sym >= ORAN_ALL_SYMBOLS)
        {
            NVLOGW_FMT(NVLOG_TAG_BASE_RU_EMULATOR, "SRS symbol {} outside expected range [{}, {}) for slot {} - skipping transmission",
                       sym, srs_info->first_symbol, ORAN_ALL_SYMBOLS, c_plane_info.launch_pattern_slot);
            return 0;
        }
    }

    timers.umsg_alloc_start_t = fine_timers_get_ns();
    aerial_fh::UPlaneMsgMultiSectionSendInfo uplane_msg = {};
    if(opt_ecpri_hdr_cfg_test)
    {
        uplane_msg.ecpri_hdr_cfg = &ecpri_hdr_cfg;
    }
    timers.umsg_alloc_end_t = fine_timers_get_ns();

    // for (int i = 0; i < ORAN_ALL_SYMBOLS; ++i)
    {
        if (profiler) profiler->startSection("section_processing");
        uplane_msg.section_num = 0;
        int num_tx_pkts = 0;

        for (int sec = 0; sec < c_plane_info.numberOfSections; ++sec)
        {
            auto &section_info = c_plane_info.section_infos[sec];
            if (sym < c_plane_info.startSym || sym >= c_plane_info.startSym + section_info.numSymbol)
            {
                continue;
            }
            timers.tx_symbol_start_t = fine_timers_get_ns();

            tx_symbol_info.section_id = section_info.section_id;
            tx_symbol_info.startPrbc = section_info.startPrbc;
            tx_symbol_info.numPrbc = section_info.numPrbc;
            tx_symbol_info.rb = section_info.rb;
            tx_symbol_info.symInc = section_info.symInc;
            tx_symbol_info.reMask = section_info.reMask;
            tx_symbol_info.ef = section_info.ef;
            tx_symbol_info.beamId = section_info.beamId;
            tx_symbol_info.freqOffset = section_info.freqOffset;
            // Resolve per-section tv_object when sections carry different channel types.
            auto* sec_tv_object = c_plane_info.is_mixed_channel ? section_info.tv_object : c_plane_info.tv_object;
            if (!sec_tv_object) {
                re_warn("handle_sect1_c_plane: null tv_object for section {}, cell {}, sym {} — skipping", sec, cell_index, sym);
                continue;
            }
            auto sec_channel = c_plane_info.is_mixed_channel ? section_info.channel_type : c_plane_info.channel_type;
            tx_symbol_info.channel_type = sec_channel;
            tx_symbol_info.startSym = sym;
            tx_symbol_info.tx_time = calculate_tx_time(c_plane_info.slot_t0, c_plane_info.tx_offset, sec_channel, sym, 
                                                       c_plane_info.eaxcId_index, opt_tti_us, opt_enable_srs_eaxcid_pacing,
                                                       c_plane_info.launch_pattern_slot, srs_s3_info, srs_s4_info, srs_s5_info,
                                                       opt_srs_pacing_eaxcids_per_symbol, opt_srs_pacing_eaxcids_per_tx_window);

            section_info.tv_index = sec_tv_object->launch_pattern[c_plane_info.launch_pattern_slot][cell_index];
            auto &tv_info = sec_tv_object->tv_info[section_info.tv_index];

            if (tx_symbol_info.numPrbc == 0)
            {
                tx_symbol_info.numPrbc = cell_configs[cell_index].ru_type == ru_type::SINGLE_SECT_MODE ? cell_configs[cell_index].ulGridSize : std::min((int)sec_tv_object->tv_info[section_info.tv_index].numPrb, cell_configs[cell_index].ulGridSize);
            }

            auto loaded_idx = cell_configs[cell_index].ul_comp_meth == aerial_fh::UserDataCompressionMethod::NO_COMPRESSION ? FIXED_POINT_16_BITS : cell_configs[cell_index].ul_bit_width;
            tx_symbol(sec_tv_object->slots[loaded_idx][section_info.tv_index], tx_symbol_info, sec_tv_object->blank_prbs.get(), timers, ORAN_CMSG_SECTION_TYPE_1, uplane_msg);
            ++num_tx_pkts;

            if (num_tx_pkts >= MAX_NUM_PACKETS_PER_C_PLANE)
            {
                do_throw(sb() << "Too many U-plane msgs to prepare " << (int)num_tx_pkts);
            }
            timers.tx_symbol_end_t = fine_timers_get_ns();
            timers.tx_symbol_t += timers.tx_symbol_end_t - timers.tx_symbol_start_t;
        }
        if (profiler) profiler->stopSection("section_processing");

        c_plane_info.packet_prepare_start = get_ns();
        bool should_drop = check_if_drop((uint16_t)cell_index, c_plane_info.channel_type, tx_symbol_info.fss);

        if(num_tx_pkts > 0 && !should_drop)
        {
            if (profiler) profiler->startSection("packet_preparation");
            timers.prepare_start_t = fine_timers_get_ns();
            const int txq_index = (c_plane_info.eaxcId_index == -1) ? 0 : get_txq_index(c_plane_info.channel_type, c_plane_info.launch_pattern_slot, tx_symbol_info.startSym, c_plane_info.eaxcId_index,
                                                                                          opt_split_srs_txq, opt_enable_srs_eaxcid_pacing,
                                                                                          srs_s3_info, srs_s4_info, srs_s5_info,
                                                                                          opt_srs_pacing_eaxcids_per_symbol, opt_srs_pacing_eaxcids_per_tx_window);
            aerial_fh::prepare_uplane_with_preallocated_tx_request(peer_list[cell_index], &uplane_msg, tx_complete_notification, tx_request, txq_index);
            timers.send_start_t = timers.prepare_end_t = fine_timers_get_ns();
            auto now = get_ns();
            bool is_late = false;
            if (tx_symbol_info.tx_time < now && opt_afh_accu_tx_sched_res_ns != 0)
            {
                NVLOGI_FMT(TAG_UL_LATE_TX, "Cell {} F{}S{}S{} Scheduling section type 1 channel {} nb_rx {} packets idx {} in the past rx time {} to rte_rx_time {} to packet_processing_start {} to ul_processing_start {} to verification_section_start {} to verification_section_end {} to tx_slot_start {} to packet_prepare_start {} to now {} tx_time - rx_time {}, now {}, now - rx_time {} now - tx_time {}",
                        cell_index,
                        tx_symbol_info.fss.frameId,
                        tx_symbol_info.fss.subframeId,
                        tx_symbol_info.fss.slotId,
                        (int)c_plane_info.channel_type,
                        c_plane_info.nb_rx,
                        c_plane_info.rx_index,
                        c_plane_info.rx_time,
                        c_plane_info.rte_rx_time - c_plane_info.rx_time,
                        c_plane_info.packet_processing_start - c_plane_info.rte_rx_time,
                        c_plane_info.ul_processing_start - c_plane_info.packet_processing_start,
                        c_plane_info.verification_section_start - c_plane_info.ul_processing_start,
                        c_plane_info.verification_section_end - c_plane_info.verification_section_start,
                        c_plane_info.tx_slot_start - c_plane_info.verification_section_end,
                        c_plane_info.packet_prepare_start - c_plane_info.tx_slot_start,
                        now - c_plane_info.packet_prepare_start,
                        tx_symbol_info.tx_time - c_plane_info.rx_time,
                        now,
                        now - c_plane_info.rx_time,
                        now - tx_symbol_info.tx_time);
                is_late = true;
            }
            if (profiler) profiler->stopSection("packet_preparation");

            if (profiler) profiler->startSection("packet_send");
            aerial_fh::TxqSendTiming tx_timing;
            auto tx_cnt = aerial_fh::send_uplane_without_freeing_tx_request(*tx_request, txqs[txq_index], &tx_timing);
            timers.send_end_t = fine_timers_get_ns();
            now = get_ns();
            if (profiler) {
                profiler->stopSection("packet_send");
                profiler->addSectionDuration("lock_wait", tx_timing.lock_wait_ns);
                profiler->addSectionDuration("tx_burst_loop", tx_timing.tx_burst_loop_ns);
            }

            NVLOGI_FMT(TAG_TX_TIMINGS,"[ST1] {} F{}S{}S{} Cell {} TX Time {} Enqueue Time {} Sym {} Num Packets {} {} Queue {}",
                        ul_channel_to_string(c_plane_info.channel_type),
                        tx_symbol_info.fss.frameId,
                        tx_symbol_info.fss.subframeId,
                        tx_symbol_info.fss.slotId,
                        cell_index,
                        tx_symbol_info.tx_time,
                        now,
                        tx_symbol_info.startSym,
                        num_tx_pkts,
                        tx_cnt,
                        txq_index);

            if (profiler) profiler->startSection("packet_stats_update");
            nb_tx += tx_cnt;
            if(oran_packet_counters.ul_c_plane[cell_index].total_slot.load() >= opt_ul_warmup_slots)
            {
                if(is_late)
                {
                    if(c_plane_info.channel_type == ul_channel::PUCCH)
                    {
                        ul_u_pucch_packet_stats.increment_counters(cell_index, PacketCounterTiming::LATE, c_plane_info.launch_pattern_slot, tx_cnt);
                    }
                    else if(c_plane_info.channel_type == ul_channel::PUSCH)
                    {
                        ul_u_pusch_packet_stats.increment_counters(cell_index, PacketCounterTiming::LATE, c_plane_info.launch_pattern_slot, tx_cnt);
                    }
                    else if(c_plane_info.channel_type == ul_channel::SRS)
                    {
                        ul_u_srs_packet_stats.increment_counters(cell_index, PacketCounterTiming::LATE, c_plane_info.launch_pattern_slot, tx_cnt);
                    }
                }
                else
                {
                    if(c_plane_info.channel_type == ul_channel::PUCCH)
                    {
                        ul_u_pucch_packet_stats.increment_counters(cell_index, PacketCounterTiming::ONTIME, c_plane_info.launch_pattern_slot, tx_cnt);
                    }
                    else if(c_plane_info.channel_type == ul_channel::PUSCH)
                    {
                        ul_u_pusch_packet_stats.increment_counters(cell_index, PacketCounterTiming::ONTIME, c_plane_info.launch_pattern_slot, tx_cnt);
                    }
                    else if(c_plane_info.channel_type == ul_channel::SRS)
                    {
                        ul_u_srs_packet_stats.increment_counters(cell_index, PacketCounterTiming::ONTIME, c_plane_info.launch_pattern_slot, tx_cnt);
                    }
                }
            }

            timers.prepare_t += timers.prepare_end_t - timers.prepare_start_t;
            timers.send_t += timers.send_end_t - timers.send_start_t;

            timers.prepare_sum_t += timers.prepare_end_t - timers.prepare_start_t;
            timers.send_sum_t += timers.send_end_t - timers.send_start_t;
            if (profiler) profiler->stopSection("packet_stats_update");
            // tv_object->u_plane_tx_tot[cell_index] += nb_tx;
        }
    }

    if (profiler) profiler->startSection("counter_updates");
    timers.counter_inc_start_t = fine_timers_get_ns();

    if (cell_configs[cell_index].ru_type == ru_type::SINGLE_SECT_MODE)
    {
        for (int sec = 0; sec < c_plane_info.numberOfSections; ++sec)
        {
            auto &section_info = c_plane_info.section_infos[sec];
            if (opt_pucch_enabled && prb_range_matching(c_plane_info, section_info, cell_index, pucch_object))
            {
                section_info.tv_object = &pucch_object;
                section_info.channel_type = ul_channel::PUCCH;
                c_plane_info.tv_object = &pucch_object;
                c_plane_info.channel_type = ul_channel::PUCCH;
                increment_section_rx_counter(c_plane_info, cell_index, sym);
            }

            if (opt_srs_enabled && prb_range_matching(c_plane_info, section_info, cell_index, srs_object))
            {
                section_info.tv_object = &srs_object;
                section_info.channel_type = ul_channel::SRS;
                c_plane_info.tv_object = &srs_object;
                c_plane_info.channel_type = ul_channel::SRS;
                increment_section_rx_counter(c_plane_info, cell_index, sym);
            }

            if (opt_pusch_enabled && prb_range_matching(c_plane_info, section_info, cell_index, pusch_object))
            {
                section_info.tv_object = &pusch_object;
                section_info.channel_type = ul_channel::PUSCH;
                c_plane_info.tv_object = &pusch_object;
                c_plane_info.channel_type = ul_channel::PUSCH;
                increment_section_rx_counter(c_plane_info, cell_index, sym);
            }
        }
    }
    else
    {
        increment_section_rx_counter(c_plane_info, cell_index, sym);
    }

    timers.counter_inc_end_t = fine_timers_get_ns();
    timers.counter_inc_t += timers.counter_inc_end_t - timers.counter_inc_start_t;
    timers.umsg_alloc_t += timers.umsg_alloc_end_t - timers.umsg_alloc_start_t;
    if (profiler) profiler->stopSection("counter_updates");

    return nb_tx;
}

int RU_Emulator::handle_sect3_c_plane(oran_c_plane_info_t& c_plane_info, uint16_t cell_index, tx_symbol_timers& timers, aerial_fh::TxqHandle* txqs, aerial_fh::TxRequestHandle* tx_request, int sym)
{
    size_t nb_tx = 0;
    tx_symbol_helper tx_symbol_info;
    auto& tv_object = c_plane_info.tv_object;
    c_plane_info.tv_index = tv_object->launch_pattern[c_plane_info.launch_pattern_slot][cell_index];
    auto& tv_info = tv_object->tv_info[c_plane_info.tv_index];
    // int num_tx_pkts = 0;
    int numSym = tv_object->tv_info[c_plane_info.tv_index].numSym;
    int startSym = tv_object->tv_info[c_plane_info.tv_index].startSym;
    int txq_index = 0;
    tx_symbol_info.cell_index       = cell_index;
    tx_symbol_info.eaxcId           = c_plane_info.eaxcId;
    tx_symbol_info.eaxcId_index     = c_plane_info.eaxcId_index;
    tx_symbol_info.fss.frameId      = c_plane_info.fss.frameId;
    tx_symbol_info.fss.subframeId   = c_plane_info.fss.subframeId;
    tx_symbol_info.fss.slotId       = c_plane_info.fss.slotId;
    tx_symbol_info.channel_type     = c_plane_info.channel_type;
    aerial_fh::UPlaneMsgMultiSectionSendInfo uplane_msg = {};
    if(opt_ecpri_hdr_cfg_test)
    {
        uplane_msg.ecpri_hdr_cfg = &ecpri_hdr_cfg;
    }
    if(sym < startSym || sym >= startSym + numSym)
    {
        return 0;
    }

    uplane_msg.section_num = 0;
    int num_tx_pkts = 0;
    // c_plane_info.startSym = i;
    // c_plane_info.numSym = 1;
    for(int sec = 0; sec < c_plane_info.numberOfSections; ++sec)
    {
        auto& section_info = c_plane_info.section_infos[sec];
        tx_symbol_info.cell_index       = cell_index;
        tx_symbol_info.eaxcId           = c_plane_info.eaxcId;
        tx_symbol_info.eaxcId_index     = c_plane_info.eaxcId_index;
        tx_symbol_info.section_id       = section_info.section_id;
        tx_symbol_info.rb               = section_info.rb;
        tx_symbol_info.symInc           = section_info.symInc;
        tx_symbol_info.startPrbc        = section_info.startPrbc;
        tx_symbol_info.numPrbc          = section_info.numPrbc;
        tx_symbol_info.reMask           = section_info.reMask;
        tx_symbol_info.ef               = section_info.ef;
        tx_symbol_info.beamId           = section_info.beamId;
        tx_symbol_info.freqOffset       = section_info.freqOffset;
        tx_symbol_info.startSym         = sym;
        tx_symbol_info.tx_time          = calculate_tx_time(c_plane_info.slot_t0, c_plane_info.tx_offset, c_plane_info.channel_type, sym, 
                                                           c_plane_info.eaxcId_index, opt_tti_us, opt_enable_srs_eaxcid_pacing,
                                                           c_plane_info.launch_pattern_slot, srs_s3_info, srs_s4_info, srs_s5_info,
                                                           opt_srs_pacing_eaxcids_per_symbol, opt_srs_pacing_eaxcids_per_tx_window);
        tx_symbol_info.startPrbc = getPRACHStartPRB(tx_symbol_info.freqOffset, SCSBW, cell_configs[cell_index].ulBandwidth);
        bool found = false;
        for(int pdu = 0; pdu < tv_info.pdu_infos.size(); ++pdu)
        {
            if(tv_info.pdu_infos[pdu].startPrb == tx_symbol_info.startPrbc)
            {
                found = true;
                section_info.prach_pdu_index = pdu;
            }
        }
        tx_symbol_info.valid_eaxcId  = (found && c_plane_info.valid_eaxcId);
        if(!found)
            re_warn("Section type 3 frequency offset {} received does not match the startPrb of the TVs in this launch pattern, sending all zeros", tx_symbol_info.freqOffset);
#if 0
        re_cons("SEC3 SYM {}: F{}S{}S{} eAxC {} sec_id {} startPrbc {} numPrbc {} freqOffset {}", sym,
                    tx_symbol_info.fss.frameId,
                    tx_symbol_info.fss.subframeId,
                    tx_symbol_info.fss.slotId,
                    tx_symbol_info.eaxcId,
                    tx_symbol_info.section_id,
                    tx_symbol_info.startPrbc,
                    tx_symbol_info.numPrbc,
                    tx_symbol_info.freqOffset
                );
#endif
        auto loaded_idx = cell_configs[cell_index].ul_comp_meth == aerial_fh::UserDataCompressionMethod::NO_COMPRESSION ? FIXED_POINT_16_BITS : cell_configs[cell_index].ul_bit_width;
        tx_symbol(prach_object.prach_slots[loaded_idx][c_plane_info.tv_index][section_info.prach_pdu_index], tx_symbol_info, prach_object.blank_prbs.get(), timers, ORAN_CMSG_SECTION_TYPE_3, uplane_msg);
        ++num_tx_pkts;
    }

    if (num_tx_pkts > 0 && !check_if_drop((uint16_t)cell_index, c_plane_info.channel_type, tx_symbol_info.fss))
    {
        // PRACH TXQ = PUSCH + PRACH.eaxcId_index
        const int txq_index = (c_plane_info.eaxcId_index == -1) ? 0 : get_txq_index(c_plane_info.channel_type, c_plane_info.launch_pattern_slot, tx_symbol_info.startSym, c_plane_info.eaxcId_index,
                                                                                      opt_split_srs_txq, opt_enable_srs_eaxcid_pacing,
                                                                                      srs_s3_info, srs_s4_info, srs_s5_info,
                                                                                      opt_srs_pacing_eaxcids_per_symbol, opt_srs_pacing_eaxcids_per_tx_window);
        aerial_fh::prepare_uplane_with_preallocated_tx_request(peer_list[cell_index], &uplane_msg, tx_complete_notification, tx_request, txq_index);
        timers.prepare_end_t = timers.send_start_t = fine_timers_get_ns();
        auto now = get_ns();

        bool is_late = false;
        if (tx_symbol_info.tx_time < now && opt_afh_accu_tx_sched_res_ns != 0)
        {
            NVLOGI_FMT(TAG_UL_LATE_TX, "Cell {} F{}S{}S{} Scheduling section type 3 packets in the past rx time {} tx_time {}, now {}, now - tx_time {}",
                    cell_index, tx_symbol_info.fss.frameId, tx_symbol_info.fss.subframeId, tx_symbol_info.fss.slotId,
                    c_plane_info.rx_time, tx_symbol_info.tx_time, now, now - tx_symbol_info.tx_time);
            is_late = true;
        }
        size_t current_nb_tx = aerial_fh::send_uplane_without_freeing_tx_request(*tx_request, txqs[txq_index]);
        nb_tx += current_nb_tx;
        if(oran_packet_counters.ul_c_plane[cell_index].total_slot.load() >= opt_ul_warmup_slots)
        {
            if(is_late)
            {
                ul_u_prach_packet_stats.increment_counters(cell_index, PacketCounterTiming::LATE, c_plane_info.launch_pattern_slot, current_nb_tx);
            }
            else
            {
                ul_u_prach_packet_stats.increment_counters(cell_index, PacketCounterTiming::ONTIME, c_plane_info.launch_pattern_slot, current_nb_tx);
            }
        }
        now = get_ns();
        NVLOGI_FMT(TAG_TX_TIMINGS, "[ST3] {} F{}S{}S{} Cell {} TX Time {} Enqueue Time {} Sym {} Num Packets {} {} Queue {}",
                    ul_channel_to_string(c_plane_info.channel_type),
                    tx_symbol_info.fss.frameId,
                    tx_symbol_info.fss.subframeId,
                    tx_symbol_info.fss.slotId,
                    cell_index,
                    tx_symbol_info.tx_time,
                    now,
                    tx_symbol_info.startSym,
                    num_tx_pkts,
                    current_nb_tx,
                    txq_index);
    }
    return nb_tx;
}

int RU_Emulator::handle_sect1_c_plane_v2(oran_c_plane_info_t &c_plane_info, uint16_t cell_index, tx_symbol_timers &timers, aerial_fh::TxqHandle *txqs, aerial_fh::TxRequestHandle* tx_request)
{
    size_t nb_tx = 0;
    tx_symbol_helper tx_symbol_info;

    timers.tx_symbol_info_copy_start_t = fine_timers_get_ns();
    tx_symbol_info.cell_index = cell_index;
    tx_symbol_info.eaxcId = c_plane_info.eaxcId;
    tx_symbol_info.eaxcId_index = c_plane_info.eaxcId_index;
    tx_symbol_info.valid_eaxcId = c_plane_info.valid_eaxcId;
    tx_symbol_info.fss.frameId = c_plane_info.fss.frameId;
    tx_symbol_info.fss.subframeId = c_plane_info.fss.subframeId;
    tx_symbol_info.fss.slotId = c_plane_info.fss.slotId;
    timers.tx_symbol_info_copy_end_t = fine_timers_get_ns();
    timers.tx_symbol_info_copy_t += timers.tx_symbol_info_copy_end_t - timers.tx_symbol_info_copy_start_t;
    
    // Error checking for SRS pacing: validate slot and symbols
    if (opt_enable_srs_eaxcid_pacing && c_plane_info.channel_type == ul_channel::SRS)
    {
        const srs_slot_type_info* srs_info = get_srs_info(c_plane_info.launch_pattern_slot, srs_s3_info, srs_s4_info, srs_s5_info);
        const int slot_type = c_plane_info.launch_pattern_slot % 10;
        
        if (srs_info == nullptr || srs_info->num_symbols == 0)
        {
            NVLOGW_FMT(NVLOG_TAG_BASE_RU_EMULATOR, "SRS encountered in slot {} (slot type {}) but no SRS expected - skipping transmission",
                       c_plane_info.launch_pattern_slot, slot_type);
            return 0;
        }
        
        // Check if symbols are in expected range
        const int start_sym = c_plane_info.startSym;
        const int end_sym = start_sym + c_plane_info.section_infos[0].numSymbol;
        if (start_sym < srs_info->first_symbol || end_sym > ORAN_ALL_SYMBOLS)
        {
            NVLOGW_FMT(NVLOG_TAG_BASE_RU_EMULATOR, "SRS symbols [{}, {}) outside expected range [{}, {}) for slot {} - skipping transmission",
                       start_sym, end_sym, srs_info->first_symbol, ORAN_ALL_SYMBOLS, c_plane_info.launch_pattern_slot);
            return 0;
        }
    }
    
    timers.umsg_alloc_start_t = fine_timers_get_ns();
    aerial_fh::UPlaneMsgMultiSectionSendInfo uplane_msg = {};
    if(opt_ecpri_hdr_cfg_test)
    {
        uplane_msg.ecpri_hdr_cfg = &ecpri_hdr_cfg;
    }
    timers.umsg_alloc_end_t = fine_timers_get_ns();
    for (int i = 0; i < ORAN_ALL_SYMBOLS; ++i)
    {
        uplane_msg.section_num = 0;
        int num_tx_pkts = 0;

        for (int sec = 0; sec < c_plane_info.numberOfSections; ++sec)
        {
            auto &section_info = c_plane_info.section_infos[sec];
            if (i < c_plane_info.startSym || i >= c_plane_info.startSym + section_info.numSymbol)
            {
                continue;
            }
            timers.tx_symbol_start_t = fine_timers_get_ns();

            tx_symbol_info.section_id = section_info.section_id;
            tx_symbol_info.startPrbc = section_info.startPrbc;
            tx_symbol_info.numPrbc = section_info.numPrbc;
            tx_symbol_info.rb = section_info.rb;
            tx_symbol_info.symInc = section_info.symInc;
            tx_symbol_info.reMask = section_info.reMask;
            tx_symbol_info.ef = section_info.ef;
            tx_symbol_info.beamId = section_info.beamId;
            tx_symbol_info.freqOffset = section_info.freqOffset;
            // Resolve per-section tv_object when sections carry different channel types.
            auto* sec_tv_object = c_plane_info.is_mixed_channel ? section_info.tv_object : c_plane_info.tv_object;
            if (!sec_tv_object) {
                re_warn("handle_sect1_c_plane_v2: null tv_object for section {}, cell {}, sym {} — skipping", sec, cell_index, i);
                continue;
            }
            auto sec_channel = c_plane_info.is_mixed_channel ? section_info.channel_type : c_plane_info.channel_type;
            tx_symbol_info.channel_type = sec_channel;
            tx_symbol_info.startSym = i;
            tx_symbol_info.tx_time = calculate_tx_time(c_plane_info.slot_t0, c_plane_info.tx_offset, sec_channel, i, 
                                                       c_plane_info.eaxcId_index, opt_tti_us, opt_enable_srs_eaxcid_pacing,
                                                       c_plane_info.launch_pattern_slot, srs_s3_info, srs_s4_info, srs_s5_info,
                                                       opt_srs_pacing_eaxcids_per_symbol, opt_srs_pacing_eaxcids_per_tx_window);
            section_info.tv_index = sec_tv_object->launch_pattern[c_plane_info.launch_pattern_slot][cell_index];
            auto &tv_info = sec_tv_object->tv_info[section_info.tv_index];

            if (tx_symbol_info.numPrbc == 0)
            {
                tx_symbol_info.numPrbc = cell_configs[cell_index].ru_type == ru_type::SINGLE_SECT_MODE ? cell_configs[cell_index].ulGridSize : std::min((int)sec_tv_object->tv_info[section_info.tv_index].numPrb, cell_configs[cell_index].ulGridSize);
            }

            auto loaded_idx = cell_configs[cell_index].ul_comp_meth == aerial_fh::UserDataCompressionMethod::NO_COMPRESSION ? FIXED_POINT_16_BITS : cell_configs[cell_index].ul_bit_width;
            tx_symbol(sec_tv_object->slots[loaded_idx][section_info.tv_index], tx_symbol_info, sec_tv_object->blank_prbs.get(), timers, ORAN_CMSG_SECTION_TYPE_1, uplane_msg);
            ++num_tx_pkts;

            if (num_tx_pkts >= MAX_NUM_PACKETS_PER_C_PLANE)
            {
                do_throw(sb() << "Too many U-plane msgs to prepare " << (int)num_tx_pkts);
            }
            timers.tx_symbol_end_t = fine_timers_get_ns();
            timers.tx_symbol_t += timers.tx_symbol_end_t - timers.tx_symbol_start_t;
        }

        if(num_tx_pkts > 0 && !check_if_drop((uint16_t)cell_index, c_plane_info.channel_type, tx_symbol_info.fss))
        {
            timers.prepare_start_t = fine_timers_get_ns();
            const int txq_index = (c_plane_info.eaxcId_index == -1) ? 0 : get_txq_index(c_plane_info.channel_type, c_plane_info.launch_pattern_slot, tx_symbol_info.startSym, c_plane_info.eaxcId_index,
                                                                                          opt_split_srs_txq, opt_enable_srs_eaxcid_pacing,
                                                                                          srs_s3_info, srs_s4_info, srs_s5_info,
                                                                                          opt_srs_pacing_eaxcids_per_symbol, opt_srs_pacing_eaxcids_per_tx_window);
            aerial_fh::prepare_uplane_with_preallocated_tx_request(peer_list[cell_index], &uplane_msg, tx_complete_notification, tx_request, txq_index);
            timers.send_start_t = timers.prepare_end_t = fine_timers_get_ns();
            auto now = get_ns();
            bool is_late = (tx_symbol_info.tx_time < now && opt_afh_accu_tx_sched_res_ns != 0);
            if (is_late)
            {
                NVLOGI_FMT(TAG_UL_LATE_TX, "Cell {} F{}S{}S{} Scheduling section type 1 packets in the past rx time {} tx_time {}, now {}, now - tx_time {}",
                        cell_index, tx_symbol_info.fss.frameId, tx_symbol_info.fss.subframeId, tx_symbol_info.fss.slotId,
                        c_plane_info.rx_time, tx_symbol_info.tx_time, now, now - tx_symbol_info.tx_time);
            }
            auto tx_cnt = aerial_fh::send_uplane_without_freeing_tx_request(*tx_request, txqs[txq_index]);
            timers.send_end_t = fine_timers_get_ns();
            now = get_ns();
            NVLOGI_FMT(TAG_TX_TIMINGS,"[ST1] {} F{}S{}S{} Cell {} TX Time {} Enqueue Time {} Sym {} Num Packets {} {} Queue {}",
                        ul_channel_to_string(c_plane_info.channel_type),
                        tx_symbol_info.fss.frameId,
                        tx_symbol_info.fss.subframeId,
                        tx_symbol_info.fss.slotId,
                        cell_index,
                        tx_symbol_info.tx_time,
                        now,
                        tx_symbol_info.startSym,
                        num_tx_pkts,
                        tx_cnt,
                        txq_index);

            nb_tx += tx_cnt;
            if(oran_packet_counters.ul_c_plane[cell_index].total_slot.load() >= opt_ul_warmup_slots)
            {
                if(is_late)
                {
                    if(c_plane_info.channel_type == ul_channel::PUCCH)
                        ul_u_pucch_packet_stats.increment_counters(cell_index, PacketCounterTiming::LATE, c_plane_info.launch_pattern_slot, tx_cnt);
                    else if(c_plane_info.channel_type == ul_channel::PUSCH)
                        ul_u_pusch_packet_stats.increment_counters(cell_index, PacketCounterTiming::LATE, c_plane_info.launch_pattern_slot, tx_cnt);
                    else if(c_plane_info.channel_type == ul_channel::SRS)
                        ul_u_srs_packet_stats.increment_counters(cell_index, PacketCounterTiming::LATE, c_plane_info.launch_pattern_slot, tx_cnt);
                }
                else
                {
                    if(c_plane_info.channel_type == ul_channel::PUCCH)
                        ul_u_pucch_packet_stats.increment_counters(cell_index, PacketCounterTiming::ONTIME, c_plane_info.launch_pattern_slot, tx_cnt);
                    else if(c_plane_info.channel_type == ul_channel::PUSCH)
                        ul_u_pusch_packet_stats.increment_counters(cell_index, PacketCounterTiming::ONTIME, c_plane_info.launch_pattern_slot, tx_cnt);
                    else if(c_plane_info.channel_type == ul_channel::SRS)
                        ul_u_srs_packet_stats.increment_counters(cell_index, PacketCounterTiming::ONTIME, c_plane_info.launch_pattern_slot, tx_cnt);
                }
            }

            timers.prepare_t += timers.prepare_end_t - timers.prepare_start_t;
            timers.send_t += timers.send_end_t - timers.send_start_t;

            timers.prepare_sum_t += timers.prepare_end_t - timers.prepare_start_t;
            timers.send_sum_t += timers.send_end_t - timers.send_start_t;
            // tv_object->u_plane_tx_tot[cell_index] += nb_tx;
        }
    }
    auto now = get_ns();
    NVLOGI_FMT(TAG_TX_TIMINGS_SUM,"[ST1] {} F{}S{}S{} Cell {} Enqueue Time {}",
               ul_channel_to_string(c_plane_info.channel_type),
               tx_symbol_info.fss.frameId,
               tx_symbol_info.fss.subframeId,
               tx_symbol_info.fss.slotId,
               cell_index,
               now);

    timers.counter_inc_start_t = fine_timers_get_ns();

    update_ul_throughput_counters(c_plane_info, cell_index);

    timers.counter_inc_end_t = fine_timers_get_ns();
    timers.counter_inc_t += timers.counter_inc_end_t - timers.counter_inc_start_t;
    timers.umsg_alloc_t += timers.umsg_alloc_end_t - timers.umsg_alloc_start_t;
    return nb_tx;
}

int RU_Emulator::handle_sect3_c_plane_v2(oran_c_plane_info_t& c_plane_info, uint16_t cell_index, tx_symbol_timers& timers, aerial_fh::TxqHandle* txqs, aerial_fh::TxRequestHandle* tx_request)
{
    size_t nb_tx = 0;
    tx_symbol_helper tx_symbol_info;
    auto& tv_object = c_plane_info.tv_object;
    c_plane_info.tv_index = tv_object->launch_pattern[c_plane_info.launch_pattern_slot][cell_index];
    auto& tv_info = tv_object->tv_info[c_plane_info.tv_index];
    // int num_tx_pkts = 0;
    int numSym = tv_object->tv_info[c_plane_info.tv_index].numSym;
    int startSym = tv_object->tv_info[c_plane_info.tv_index].startSym;
    int txq_index = 0;
    tx_symbol_info.cell_index       = cell_index;
    tx_symbol_info.eaxcId           = c_plane_info.eaxcId;
    tx_symbol_info.eaxcId_index     = c_plane_info.eaxcId_index;
    tx_symbol_info.fss.frameId      = c_plane_info.fss.frameId;
    tx_symbol_info.fss.subframeId   = c_plane_info.fss.subframeId;
    tx_symbol_info.fss.slotId       = c_plane_info.fss.slotId;
    tx_symbol_info.channel_type     = c_plane_info.channel_type;
    aerial_fh::UPlaneMsgMultiSectionSendInfo uplane_msg = {};
    if(opt_ecpri_hdr_cfg_test)
    {
        uplane_msg.ecpri_hdr_cfg = &ecpri_hdr_cfg;
    }
    for(int i = startSym; i < startSym + numSym; ++i)
    {
        uplane_msg.section_num = 0;
        int num_tx_pkts = 0;
        c_plane_info.startSym = i;
        c_plane_info.numSym = 1;
        for(int sec = 0; sec < c_plane_info.numberOfSections; ++sec)
        {
            auto& section_info = c_plane_info.section_infos[sec];
            tx_symbol_info.cell_index       = cell_index;
            tx_symbol_info.eaxcId           = c_plane_info.eaxcId;
            tx_symbol_info.eaxcId_index     = c_plane_info.eaxcId_index;
            tx_symbol_info.section_id       = section_info.section_id;
            tx_symbol_info.rb               = section_info.rb;
            tx_symbol_info.symInc           = section_info.symInc;
            tx_symbol_info.startPrbc        = section_info.startPrbc;
            tx_symbol_info.numPrbc          = section_info.numPrbc;
            tx_symbol_info.reMask           = section_info.reMask;
            tx_symbol_info.ef               = section_info.ef;
            tx_symbol_info.beamId           = section_info.beamId;
            tx_symbol_info.freqOffset       = section_info.freqOffset;
            tx_symbol_info.startSym         = c_plane_info.startSym;
            tx_symbol_info.tx_time          = calculate_tx_time(c_plane_info.slot_t0, c_plane_info.tx_offset, c_plane_info.channel_type, i, 
                                                               c_plane_info.eaxcId_index, opt_tti_us, opt_enable_srs_eaxcid_pacing,
                                                               c_plane_info.launch_pattern_slot, srs_s3_info, srs_s4_info, srs_s5_info,
                                                               opt_srs_pacing_eaxcids_per_symbol, opt_srs_pacing_eaxcids_per_tx_window);
            tx_symbol_info.startPrbc = getPRACHStartPRB(tx_symbol_info.freqOffset, SCSBW, cell_configs[cell_index].ulBandwidth);
            bool found = false;
            for(int pdu = 0; pdu < tv_info.pdu_infos.size(); ++pdu)
            {
                if(tv_info.pdu_infos[pdu].startPrb == tx_symbol_info.startPrbc)
                {
                    found = true;
                    section_info.prach_pdu_index = pdu;
                }
            }
            tx_symbol_info.valid_eaxcId  = (found && c_plane_info.valid_eaxcId);
            if(!found)
                re_warn("Section type 3 frequency offset {} received does not match the startPrb of the TVs in this launch pattern, sending all zeros", tx_symbol_info.freqOffset);
#if 0
            re_cons("SEC3 SYM {}: F{}S{}S{} eAxC {} sec_id {} startPrbc {} numPrbc {} freqOffset {}", i,
                        tx_symbol_info.fss.frameId,
                        tx_symbol_info.fss.subframeId,
                        tx_symbol_info.fss.slotId,
                        tx_symbol_info.eaxcId,
                        tx_symbol_info.section_id,
                        tx_symbol_info.startPrbc,
                        tx_symbol_info.numPrbc,
                        tx_symbol_info.freqOffset
                    );
#endif
            auto loaded_idx = cell_configs[cell_index].ul_comp_meth == aerial_fh::UserDataCompressionMethod::NO_COMPRESSION ? FIXED_POINT_16_BITS : cell_configs[cell_index].ul_bit_width;
            tx_symbol(prach_object.prach_slots[loaded_idx][c_plane_info.tv_index][section_info.prach_pdu_index], tx_symbol_info, prach_object.blank_prbs.get(), timers, ORAN_CMSG_SECTION_TYPE_3, uplane_msg);
            ++num_tx_pkts;
        }

        if (num_tx_pkts > 0 && !check_if_drop((uint16_t)cell_index, c_plane_info.channel_type, tx_symbol_info.fss))
        {
            // PRACH TXQ = PUSCH + PRACH.eaxcId_index
            const int txq_index = (c_plane_info.eaxcId_index == -1) ? 0 : get_txq_index(c_plane_info.channel_type, c_plane_info.launch_pattern_slot, tx_symbol_info.startSym, c_plane_info.eaxcId_index,
                                                                                          opt_split_srs_txq, opt_enable_srs_eaxcid_pacing,
                                                                                          srs_s3_info, srs_s4_info, srs_s5_info,
                                                                                          opt_srs_pacing_eaxcids_per_symbol, opt_srs_pacing_eaxcids_per_tx_window);
            aerial_fh::prepare_uplane_with_preallocated_tx_request(peer_list[cell_index], &uplane_msg, tx_complete_notification, tx_request, txq_index);
            timers.prepare_end_t = timers.send_start_t = fine_timers_get_ns();
            auto now = get_ns();

            if (tx_symbol_info.tx_time < now && opt_afh_accu_tx_sched_res_ns != 0)
            {
                NVLOGI_FMT(TAG_UL_LATE_TX, "Cell {} F{}S{}S{} Scheduling section type 3 packets in the past rx time {} tx_time {}, now {}, now - tx_time {}",
                        cell_index, tx_symbol_info.fss.frameId, tx_symbol_info.fss.subframeId, tx_symbol_info.fss.slotId,
                        c_plane_info.rx_time, tx_symbol_info.tx_time, now, now - tx_symbol_info.tx_time);
            }
            size_t current_nb_tx = aerial_fh::send_uplane_without_freeing_tx_request(*tx_request, txqs[txq_index]);
            nb_tx += current_nb_tx;
            bool is_late = (tx_symbol_info.tx_time < now && opt_afh_accu_tx_sched_res_ns != 0);
            if(oran_packet_counters.ul_c_plane[cell_index].total_slot.load() >= opt_ul_warmup_slots)
            {
                if(is_late)
                    ul_u_prach_packet_stats.increment_counters(cell_index, PacketCounterTiming::LATE, c_plane_info.launch_pattern_slot, current_nb_tx);
                else
                    ul_u_prach_packet_stats.increment_counters(cell_index, PacketCounterTiming::ONTIME, c_plane_info.launch_pattern_slot, current_nb_tx);
            }

            now = get_ns();
            NVLOGI_FMT(TAG_TX_TIMINGS, "[ST3] {} F{}S{}S{} Cell {} TX Time {} Enqueue Time {} Sym {} Num Packets {} {} Queue {}",
                       ul_channel_to_string(c_plane_info.channel_type),
                       tx_symbol_info.fss.frameId,
                       tx_symbol_info.fss.subframeId,
                       tx_symbol_info.fss.slotId,
                       cell_index,
                       tx_symbol_info.tx_time,
                       now,
                       tx_symbol_info.startSym,
                       num_tx_pkts,
                       current_nb_tx,
                       txq_index);
        }
    }
    auto now = get_ns();
    NVLOGI_FMT(TAG_TX_TIMINGS_SUM,"[ST3] {} F{}S{}S{} Cell {} Enqueue Time {}",
                ul_channel_to_string(c_plane_info.channel_type),
                tx_symbol_info.fss.frameId,
                tx_symbol_info.fss.subframeId,
                tx_symbol_info.fss.slotId,
                cell_index,
                now);
    return nb_tx;
}

void RU_Emulator::tx_symbol(Slot& slot, tx_symbol_helper& tx_symbol_info, void * blank_prbs, tx_symbol_timers& timers, uint8_t section_type, aerial_fh::UPlaneMsgMultiSectionSendInfo& uplane_msg)
{
    void * ext_ptr = nullptr;
    uint8_t eaxcId_index = tx_symbol_info.eaxcId_index;
    uint8_t sym_idx = tx_symbol_info.startSym;
    uint8_t cell_index = tx_symbol_info.cell_index;
    uint16_t prb_idx = tx_symbol_info.startPrbc;
    auto channel_id = tx_symbol_info.channel_type;
    if(tx_symbol_info.valid_eaxcId && !ul_pkts_zero_uplane_test.at(cell_index).at((int)channel_id).enabled.load())
    {
        ext_ptr = slot.ptrs.at(eaxcId_index).at(sym_idx).at(prb_idx);
    }
    else
    {
        ext_ptr = blank_prbs;
        ul_pkts_zero_uplane_test.at(cell_index).at((int)channel_id).enabled.store(false);
    }
    aerial_fh::UPlaneSectionInfo& uplane_section = uplane_msg.section_infos[uplane_msg.section_num++];
    aerial_fh::MsgSendWindow& msg_send_window = uplane_msg.tx_window;

    uplane_section.iq_data_buffer = ext_ptr;
    uplane_section.section_id = tx_symbol_info.section_id;
    uplane_section.rb = 0;
    uplane_section.sym_inc = 0;
    if(section_type == ORAN_CMSG_SECTION_TYPE_3)
    {
        uplane_section.start_prbu = 0;// start_prbu = 0 for all PRACH occasions
    }
    else
    {
        uplane_section.start_prbu = tx_symbol_info.startPrbc;
    }
    uplane_section.num_prbu = tx_symbol_info.numPrbc;

#if 0
        if(tx_symbol_info.eaxcId_index == 0)
        {
            re_cons("Pkt {} sym {} flow {} prb {}", i, sym_idx, tx_symbol_info.eaxcId_index,  prb_idx);
            uint8_t* tmp = (uint8_t*)ext_ptr;
            for(int b=0; b < num_prbs; b++) {
                // printf("TV PRB %5d: ", b);
                for(int c=0; c<48; c++)
                {
                    printf("%02X", tmp[c+ b*48]);
                }
                printf("\n");
            }
        }
#endif
    if(is_cx6_nic) //Gate the opt_afh_accu_tx_sched_res_ns check only for CX-6 NIC
    {
        if(opt_afh_accu_tx_sched_res_ns != 0)
        {
            msg_send_window.tx_window_start = tx_symbol_info.tx_time;
        }
    }
    else
    {
        msg_send_window.tx_window_start = tx_symbol_info.tx_time;
    }

    if(uplane_msg.section_num == 1)
    {
        struct oran_umsg_iq_hdr& iq_df = uplane_msg.radio_app_hdr;
        iq_df.frameId          = tx_symbol_info.fss.frameId;
        iq_df.subframeId       = tx_symbol_info.fss.subframeId;
        iq_df.slotId           = tx_symbol_info.fss.slotId;
        iq_df.symbolId         = tx_symbol_info.startSym;

        if(tx_symbol_info.valid_eaxcId)
        {
            if(section_type == ORAN_CMSG_SECTION_TYPE_1)
            {
                if (tx_symbol_info.channel_type == ul_channel::SRS)
                {
                    if (cell_index >= opt_num_cells || eaxcId_index >= peer_flow_map_srs[cell_index].size())
                    {
                        fprintf(stderr, "ST1 flow not found cell_index %d eaxcID_index %d\n", cell_index, eaxcId_index);
                    }

                    uplane_msg.flow = peer_flow_map_srs[cell_index][eaxcId_index];
                }
                else
                {
                    if (cell_index >= opt_num_cells || eaxcId_index >= ul_peer_flow_map[cell_index].size())
                    {
                        fprintf(stderr, "ST1 flow not found cell_index %d eaxcID_index %d\n", cell_index, eaxcId_index);
                    }

                    uplane_msg.flow = ul_peer_flow_map[cell_index][eaxcId_index];
                }
            }
            else if(section_type == ORAN_CMSG_SECTION_TYPE_3)
            {
                if(cell_index >= opt_num_cells || eaxcId_index >= peer_flow_map_prach[cell_index].size())
                {
                    fprintf(stderr,"ST3 flow not found cell_index %d eaxcID_index %d\n", cell_index, eaxcId_index);
                }
                uplane_msg.flow = peer_flow_map_prach[cell_index][eaxcId_index];
            }
        }
        else
        {
            if(section_type == ORAN_CMSG_SECTION_TYPE_1)
            {
                uplane_msg.flow = ul_peer_flow_map[cell_index][0];
            }
            else if(section_type == ORAN_CMSG_SECTION_TYPE_3)
            {
                uplane_msg.flow = peer_flow_map_prach[cell_index][0];
            }
        }
    }
}

int RU_Emulator::tx_slot(slot_tx_info& slot_tx, int cell_index, tx_symbol_timers& timers,  aerial_fh::TxqHandle* txqs, aerial_fh::TxRequestHandle* tx_request, bool enable_mmimo, perf_metrics::PerfMetricsAccumulator* profiler)
{
    if(enable_mmimo)
    {
        int num_tx_pkts = 0;
        for(int sym = 0; sym < ORAN_ALL_SYMBOLS; ++sym)
        {
            for(int i = 0; i < slot_tx.c_plane_infos_size; ++i)
            {
                auto& c_plane_info = slot_tx.c_plane_infos[i];
                if(c_plane_info.dir == DIRECTION_DOWNLINK)
                {
                    continue;
                }

                c_plane_info.tx_slot_start = get_ns();
                if(c_plane_info.startSym > sym)
                {
                    continue;
                }

                switch(slot_tx.c_plane_infos[i].section_type)
                {
                    //If section 1 send PUSCH/PUCCH/SRS symbols
                    case ORAN_CMSG_SECTION_TYPE_1:
                    {
                        if (profiler) profiler->startSection("handle_sect1");
                        auto num_tx_local = handle_sect1_c_plane(c_plane_info, cell_index, timers, txqs, tx_request, sym, profiler);
                        if (profiler) profiler->stopSection("handle_sect1");
                        num_tx_pkts += num_tx_local;
                        auto &tv_object = c_plane_info.tv_object;
                        tv_object->u_plane_tx[cell_index] += num_tx_local;
                        tv_object->u_plane_tx_tot[cell_index] += num_tx_local;
                    }
                    break;
                    //If section 3, prepare and send all PRACH symbols
                    case ORAN_CMSG_SECTION_TYPE_3:
                    {
                        if (profiler) profiler->startSection("handle_sect3");
                        auto num_tx_local = handle_sect3_c_plane(c_plane_info, cell_index, timers, txqs, tx_request, sym);
                        if (profiler) profiler->stopSection("handle_sect3");
                        num_tx_pkts += num_tx_local;
                        auto &tv_object = c_plane_info.tv_object;
                        tv_object->u_plane_tx[cell_index] += num_tx_local;
                        tv_object->u_plane_tx_tot[cell_index] += num_tx_local;

                        ++tv_object->c_plane_rx[cell_index];
                        ++tv_object->c_plane_rx_tot[cell_index];
                    }
                    break;
                    default:
                    break;
                }

                if(sym == ORAN_ALL_SYMBOLS - 1) {
                    auto now = get_ns();
                    NVLOGI_FMT(TAG_TX_TIMINGS_SUM, "[{}] {} F{}S{}S{} Cell {} Enqueue Time {}",
                        (slot_tx.c_plane_infos[i].section_type == ORAN_CMSG_SECTION_TYPE_3 ? "ST3" : "ST1"),
                        ul_channel_to_string(c_plane_info.channel_type),
                        c_plane_info.fss.frameId,
                        c_plane_info.fss.subframeId,
                        c_plane_info.fss.slotId,
                        cell_index,
                        now
                    );
                }
            }
        }


        for(int i = 0; i < slot_tx.c_plane_infos_size; ++i)
        {
            auto& c_plane_info = slot_tx.c_plane_infos[i];
            if(c_plane_info.dir == DIRECTION_DOWNLINK)
            {
                continue;
            }
            if (profiler) profiler->startSection("tput_counters");
            if(c_plane_info.section_type == ORAN_CMSG_SECTION_TYPE_3)
            {
                auto &tv_object = c_plane_info.tv_object;
                if (c_plane_info.valid_eaxcId)
                {
                    int counter_index = c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId;
                    bool complete = false;
                    {
                        // Atomically increment and get the previous value
                        const uint16_t prev_count = tv_object->section_rx_counters[cell_index][counter_index].fetch_add(
                            1, std::memory_order_acq_rel);
                        const uint16_t new_count = prev_count + 1;
                        const uint16_t expected_count = cell_configs[cell_index].num_valid_PRACH_flows;
                        
                        // Check if we reached the expected threshold
                        if (new_count == expected_count)
                        {
                            tv_object->section_rx_counters[cell_index][counter_index].store(0, std::memory_order_release);
                            complete = true;
                        }
                        else if (new_count > expected_count)
                        {
                            re_warn("Section counter overflow: cell_index={}, counter_index={}, new_count={}, expected_count={}",
                                    cell_index, counter_index, new_count, expected_count);
                            tv_object->section_rx_counters[cell_index][counter_index].store(0, std::memory_order_release);
                        }
                    }
                    if(complete)
                    {
                        ++tv_object->throughput_slot_counters[cell_index];
                        ++tv_object->total_slot_counters[cell_index];
                    }
                }
            }
            if (profiler) profiler->stopSection("tput_counters");

            if (profiler) profiler->startSection("verify");
            for (auto section_idx = 0; section_idx < c_plane_info.section_infos_size; section_idx++)
            {
                auto &section_info = c_plane_info.section_infos[section_idx];
                // VERIFY BFW
                if(section_info.ext_infos_size > 0)
                {
                    verify_extensions(c_plane_info, section_info, cell_index);
                }
            }
            if (profiler) profiler->stopSection("verify");
        }
        return num_tx_pkts;
    }
    else //legacy for non-mmimo
    {
        int num_tx_pkts = 0;

        for(int i = 0; i < slot_tx.c_plane_infos_size; ++i)
        {
            auto& c_plane_info = slot_tx.c_plane_infos[i];
            if(c_plane_info.dir == DIRECTION_DOWNLINK)
            {
                continue;
            }

            switch(slot_tx.c_plane_infos[i].section_type)
            {
                //If section 1 send PUSCH/PUCCH/SRS symbols
                case ORAN_CMSG_SECTION_TYPE_1:
                {
                    auto num_tx_local = handle_sect1_c_plane_v2(c_plane_info, cell_index, timers, txqs, tx_request);
                    num_tx_pkts += num_tx_local;
                    auto &tv_object = c_plane_info.tv_object;
                    tv_object->u_plane_tx[cell_index] += num_tx_local;
                    tv_object->u_plane_tx_tot[cell_index] += num_tx_local;
                }
                break;
                //If section 3, prepare and send all PRACH symbols
                case ORAN_CMSG_SECTION_TYPE_3:
                {
                    auto num_tx_local = handle_sect3_c_plane_v2(c_plane_info, cell_index, timers, txqs, tx_request);
                    num_tx_pkts += num_tx_local;
                    auto &tv_object = c_plane_info.tv_object;
                    tv_object->u_plane_tx[cell_index] += num_tx_local;
                    tv_object->u_plane_tx_tot[cell_index] += num_tx_local;

                    ++tv_object->c_plane_rx[cell_index];
                    ++tv_object->c_plane_rx_tot[cell_index];
                    if (c_plane_info.valid_eaxcId)
                    {
                        int counter_index = c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID +
                                            c_plane_info.fss.slotId;
                        ++tv_object->section_rx_counters[cell_index][counter_index];
                        if (tv_object->section_rx_counters[cell_index][counter_index].load() == cell_configs[cell_index].num_valid_PRACH_flows)
                        {
                            ++tv_object->throughput_slot_counters[cell_index];
                            ++tv_object->total_slot_counters[cell_index];
                            tv_object->section_rx_counters[cell_index][counter_index].store(0);
                        }
                    }
                }
                break;
                default:
                break;
            }
        }

        for(int i = 0; i < slot_tx.c_plane_infos_size; ++i)
        {
            auto& c_plane_info = slot_tx.c_plane_infos[i];
            for (auto section_idx = 0; section_idx < c_plane_info.section_infos_size; section_idx++)
            {
                auto &section_info = c_plane_info.section_infos[section_idx];
                // VERIFY BFW
                if (section_info.ext11_ptr != nullptr)
                {
                    verify_extType11(section_info.ext11_ptr, c_plane_info, section_info, cell_index);
                }
            }
        }
        return num_tx_pkts;
    }
}

void RU_Emulator::find_channel_type_for_each_section(oran_c_plane_info_t& c_plane_info, uint16_t cell_index) {
    if (c_plane_info.numberOfSections == 0)
    {
        re_warn("find_channel_type_for_each_section: numberOfSections is 0, cell {} — defaulting to PUSCH", cell_index);
        c_plane_info.tv_object = &pusch_object;
        c_plane_info.channel_type = ul_channel::PUSCH;
        c_plane_info.is_mixed_channel = false;
        return;
    }

    const prb_map_t* pucch_prb_map = opt_pucch_enabled ? resolve_prb_map(c_plane_info, cell_index, pucch_object) : nullptr;
    const prb_map_t* srs_prb_map = opt_srs_enabled ? resolve_prb_map(c_plane_info, cell_index, srs_object) : nullptr;
    int sym = c_plane_info.startSym;

    for (int sec = 0; sec < c_plane_info.numberOfSections; ++sec)
    {
        auto &section_info = c_plane_info.section_infos[sec];
        // Treat out-of-range sym/startPrbc as "not present" so PUSCH is chosen.
        bool sym_in_range = sym >= 0 && sym < OFDM_SYMBOLS_PER_SLOT;
        bool prb_in_range = section_info.startPrbc < MAX_NUM_PRBS_PER_SYMBOL;

        if (sym_in_range && prb_in_range &&
            pucch_prb_map && (*pucch_prb_map)[sym][section_info.startPrbc])
        {
            section_info.tv_object = &pucch_object;
            section_info.channel_type = ul_channel::PUCCH;
        }
        else if (sym_in_range && prb_in_range &&
                 srs_prb_map && (*srs_prb_map)[sym][section_info.startPrbc])
        {
            section_info.tv_object = &srs_object;
            section_info.channel_type = ul_channel::SRS;
        }
        else
        {
            section_info.tv_object = &pusch_object;
            section_info.channel_type = ul_channel::PUSCH;
        }
    }

    c_plane_info.tv_object = c_plane_info.section_infos[0].tv_object;
    c_plane_info.channel_type = c_plane_info.section_infos[0].channel_type;
    c_plane_info.is_mixed_channel = false;

    for (int sec = 1; sec < c_plane_info.numberOfSections; ++sec)
    {
        if (c_plane_info.section_infos[sec].channel_type != c_plane_info.channel_type)
        {
            c_plane_info.is_mixed_channel = true;
            break;
        }
    }
}

inline void RU_Emulator::find_channel_type(oran_c_plane_info_t& c_plane_info, uint16_t cell_index) {
    if(opt_pucch_enabled && c_plane_channel_type_checking(c_plane_info, cell_index, pucch_object))
    {
        c_plane_info.tv_object = &pucch_object;
        c_plane_info.channel_type = ul_channel::PUCCH;
    }
    else if(opt_srs_enabled && c_plane_channel_type_checking(c_plane_info, cell_index, srs_object))
    {
        c_plane_info.tv_object = &srs_object;
        c_plane_info.channel_type = ul_channel::SRS;
    }
    else
    {
        c_plane_info.tv_object = &pusch_object;
        c_plane_info.channel_type = ul_channel::PUSCH;
    }
}

inline void find_eAxC_index(oran_c_plane_info_t& c_plane_info, cell_config& cell_config)
{
    c_plane_info.eaxcId_index = -1;
    c_plane_info.valid_eaxcId = false;
    switch(c_plane_info.section_type)
    {
        case ORAN_CMSG_SECTION_TYPE_1:
        {
            c_plane_info.eaxcId_index = -1;
            if (c_plane_info.channel_type == ul_channel::SRS)
            {
                auto it = find(cell_config.eAxC_SRS_list.begin(), cell_config.eAxC_SRS_list.end(), c_plane_info.eaxcId);
                c_plane_info.eaxcId_index = it - cell_config.eAxC_SRS_list.begin();
                if (c_plane_info.eaxcId_index < cell_config.num_valid_SRS_flows)
                {
                        c_plane_info.valid_eaxcId = true;
                }
                else
                {
                    c_plane_info.eaxcId_index = -1;
                    c_plane_info.valid_eaxcId = false;
                }
            }
            else
            {
                auto it = find(cell_config.eAxC_UL.begin(), cell_config.eAxC_UL.end(), c_plane_info.eaxcId);
                c_plane_info.eaxcId_index = it - cell_config.eAxC_UL.begin();
                if (c_plane_info.eaxcId_index < cell_config.num_ul_flows)
                {
                        c_plane_info.valid_eaxcId = true;
                }
                else
                {
                    c_plane_info.eaxcId_index = -1;
                    c_plane_info.valid_eaxcId = false;
                }
            }
        }
        break;
        case ORAN_CMSG_SECTION_TYPE_3:
        {
            c_plane_info.eaxcId_index = -1;
            auto it = find(cell_config.eAxC_PRACH_list.begin(), cell_config.eAxC_PRACH_list.end(), c_plane_info.eaxcId);
            c_plane_info.eaxcId_index = it - cell_config.eAxC_PRACH_list.begin();
            if (c_plane_info.eaxcId_index < cell_config.num_valid_PRACH_flows)
            {
                c_plane_info.valid_eaxcId = true;
            }
            else
            {
                c_plane_info.eaxcId_index = -1;
                c_plane_info.valid_eaxcId = false;
            }
        }
        break;
        default:
        break;
    }
}

int RU_Emulator::verify_extensions(oran_c_plane_info_t &c_plane_info, oran_c_plane_section_info_t &section_info, int cell_index)
{
    uint16_t total_ext_len = 0;
    bool se4_seen = false;
    bool se5_seen = false;
    bool se11_seen = false;

    for (int i = 0; i < section_info.ext_infos_size; i++)
    {
        auto &ext = section_info.ext_infos[i];
        if (ext.ext_type == ORAN_CMSG_SECTION_EXT_TYPE_11)
        {
            if (i != section_info.ext_infos_size - 1)
            {
                re_cons("SE11 should be the last extension with current implementation");
                if (opt_dlc_tb)
                {
                    section_info.error_status = -1;
                }
                else
                {
                    do_throw(sb() << "SE11 should be the last extension with current implementation!");
                    sleep(1);
                }
            }
            total_ext_len += verify_extType11(ext.ext_ptr, c_plane_info, section_info, cell_index);
        }
        else if (ext.ext_type == ORAN_CMSG_SECTION_EXT_TYPE_4)
        {
            if (se5_seen)
            {
                re_cons("SE5 already seen in the same section");
                if (opt_dlc_tb)
                {
                    section_info.error_status = -1;
                    continue;
                }
                else
                {
                    do_throw(sb() << "SE5 already seen in the same section!");
                    sleep(1);
                }
            }
            if (c_plane_info.dir != oran_pkt_dir::DIRECTION_DOWNLINK)
            {
                re_cons("SE4 only supported with DL");
                if (opt_dlc_tb)
                {
                    section_info.error_status = -1;
                }
                else
                {
                    do_throw(sb() << "SE4 only supported with DL!");
                    sleep(1);
                }
            }
            auto ext_sz = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_4);
            if (ext.ext_len != ext_sz)
            {
                re_cons("F{}S{}S{} Sym {} eAxCID {} eAxCID idx {} ext_info_size {} ext_type {} ext4 len not correct! expected len {} ext_info.ext_len {}", c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, c_plane_info.eaxcId, c_plane_info.eaxcId_index, section_info.ext_infos_size, ext.ext_type, ext_sz, ext.ext_len);
                if (opt_dlc_tb)
                {
                    section_info.error_status = -1;
                }
                else
                {
                    do_throw(sb() << "ext4 len not correct!");
                    sleep(1);
                }
            }

            total_ext_len += ext_sz;
            se4_seen = true;
        }
        else if (ext.ext_type == ORAN_CMSG_SECTION_EXT_TYPE_5)
        {
            if (se4_seen)
            {
                re_cons("SE4 already seen in the same section");
                if (opt_dlc_tb)
                {
                    section_info.error_status = -1;
                }
                else
                {
                    do_throw(sb() << "SE4 already seen in the same section!");
                    sleep(1);
                }
            }

            if (c_plane_info.dir != oran_pkt_dir::DIRECTION_DOWNLINK)
            {
                re_cons("SE5 only supported with DL");
                if (opt_dlc_tb)
                {
                    section_info.error_status = -1;
                }
                else
                {
                    do_throw(sb() << "SE5 only supported with DL!");
                    sleep(1);
                }
            }
            auto ext_sz = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_5);
            if (ext.ext_len != ext_sz)
            {
                re_cons("F{}S{}S{} Sym {} eAxCID {} eAxCID idx {} ext_info_size {} ext_type {} ext5 len not correct! expected len {} ext_info.ext_len {}", c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, c_plane_info.eaxcId, c_plane_info.eaxcId_index, section_info.ext_infos_size, ext.ext_type, ext_sz, ext.ext_len);
                if (opt_dlc_tb)
                {
                    section_info.error_status = -1;
                }
                else
                {
                    do_throw(sb() << "ext5 len not correct!");
                    sleep(1);
                }
            }

            total_ext_len += ext_sz;
            se5_seen = true;
        }
        else
        {
            re_cons("F{}S{}S{} Sym {} eAxCID {} eAxCID idx {} ext_info_size {} ext_type {}", c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, c_plane_info.eaxcId, c_plane_info.eaxcId_index, section_info.ext_infos_size, ext.ext_type);
            re_cons("Unsupported extType detected!");
            if (opt_dlc_tb)
            {
                section_info.error_status = -1;
            }
            else
            {
                do_throw(sb() << "Unsupported extType detected!");
                sleep(1);
            }
        }
    }

    if (c_plane_info.dir == oran_pkt_dir::DIRECTION_DOWNLINK && cell_configs[cell_index].dl_comp_meth == aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION && !se4_seen && !se5_seen)
    {
        re_cons("Error: no ext4/5 detected for modulation compression. F{}S{}S{} Sym {} eAxCID {} eAxCID idx {} ext_info_size {}", c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, c_plane_info.eaxcId, c_plane_info.eaxcId_index, section_info.ext_infos_size);
        if (opt_dlc_tb)
        {
            section_info.error_status = -1;
        }
        else
        {
            sleep(1);
            do_throw(sb() << "Error: no ext4/5 detected for modulation compression");
        }
    }
    return total_ext_len;
}

inline void RU_Emulator::dynamic_beamid_validation(oran_c_plane_info_t &c_plane_info, int cell_index)
{
    auto eaxcId_index =  c_plane_info.eaxcId_index;
    auto slot_3gpp = c_plane_info.launch_pattern_slot;
    if (eaxcId_index < 0 || slot_3gpp >= launch_pattern_slot_size)
    {
        re_cons("Invalid params!  eaxcId_index {} slot_3gpp {}", eaxcId_index, slot_3gpp);
        return;
    }
    auto &disabled_bfw_dyn_bfw_beam_id = fss_disabled_bfw_dyn_bfw_beam_id[cell_index][eaxcId_index][slot_3gpp];
    auto &beam_id_cache = fss_dyn_bfw_beam_id[cell_index][eaxcId_index][slot_3gpp];

    bool dump = false;
    for (int sym = 0; sym < SLOT_NUM_SYMS; sym++)
    {
        if (dump)
            break;
        auto &beam_id_cnt = fss_disabled_bfw_dyn_bfw_beam_id_cnt[cell_index][eaxcId_index][slot_3gpp][sym];
        for (int i = 0; i < beam_id_cnt.load(); i++)
        {
            auto cur_prb = disabled_bfw_dyn_bfw_beam_id[sym][i][0];
            auto beamId = disabled_bfw_dyn_bfw_beam_id[sym][i][1];
            bool found = false;
            uint16_t expected_beam_id = 0;
            if (beam_id_cache[cur_prb][0] >= oran_beam_id_info.dynamic_beam_id_start)
            {
                found = true;
                expected_beam_id = beam_id_cache[cur_prb][0];
            }
            else
            {
                for (int prb = 0; prb < ORAN_MAX_PRB_X_SLOT; prb++)
                {
                    if (beam_id_cache[prb][0] >= oran_beam_id_info.dynamic_beam_id_start)
                    {
                        if (cur_prb >= prb && cur_prb <= prb + beam_id_cache[prb][1])
                        {
                            found = true;
                            expected_beam_id = beam_id_cache[prb][0];
                            break;
                        }
                    }
                }
            }

            if (!found)
            {
                auto &beam_id_set_full_bw = cell_configs[cell_index].dyn_bfw_beam_id_with_full_bw[eaxcId_index];
                if (beam_id_set_full_bw.count(beamId))
                {
                    found = true;
                    expected_beam_id = beamId;
                }
            }

            if (beamId < oran_beam_id_info.dynamic_beam_id_start || !found || expected_beam_id != beamId)
            {
                auto cur_prb = disabled_bfw_dyn_bfw_beam_id[sym][i][0];
                auto beamId = disabled_bfw_dyn_bfw_beam_id[sym][i][1];
                auto ts = disabled_bfw_dyn_bfw_beam_id[sym][i][2];
                auto cnt = disabled_bfw_dyn_bfw_beam_id[sym][i][3];
                re_cons("Erroneous dynamic beamId detected: disableBFWs = 1, Cell {} Flow {} current: F{}S{}S{} slot_3gpp {} Symbol {} cur_prb {} beamId {}  ts {} cnt {}", cell_index, eaxcId_index, c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, slot_3gpp, sym, cur_prb, beamId, ts, cnt);
                if (found)
                {
                    re_cons("Expected beamId: {}, received beamId: {}", expected_beam_id, beamId);
                }
                else
                {
                    re_cons("Expected beamId not found!");
                }
                dump = true;
                break;
            }
        }
        // beam_id_cnt.store(0);
    }

    if (dump)
    {
        re_cons("*************************************BEAM ID DUMP*********************************************");
        for (int sym = 0; sym < SLOT_NUM_SYMS; sym++)
        {
            auto &beam_id_cnt = fss_disabled_bfw_dyn_bfw_beam_id_cnt[cell_index][eaxcId_index][slot_3gpp][sym];
            for (int i = 0; i < beam_id_cnt.load(); i++)
            {
                auto cur_prb = disabled_bfw_dyn_bfw_beam_id[sym][i][0];
                auto beamId = disabled_bfw_dyn_bfw_beam_id[sym][i][1];
                auto ts = disabled_bfw_dyn_bfw_beam_id[sym][i][2];
                auto cnt = disabled_bfw_dyn_bfw_beam_id[sym][i][3];
                re_cons("Cell {} Flow {} current: F{}S{}S{} slot_3gpp {} Symbol {} cur_prb {} beamId {}  ts {} cnt {}", cell_index, eaxcId_index, c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, slot_3gpp, sym, cur_prb, beamId, ts, cnt);
            }
        }
        for (int prb = 0; prb < ORAN_MAX_PRB_X_SLOT; prb++)
        {
            re_cons("Cell {} Flow {} current: F{}S{}S{} slot_3gpp {} prb {} beamId {} grpsz {}", cell_index, eaxcId_index, c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, slot_3gpp, prb, beam_id_cache[prb][0], beam_id_cache[prb][1]);
        }
        sleep(1);
        do_throw(sb() << "Erroneous dynamic beam Id detected, disableBFWs = 1 !");
    }
    for (int sym = 0; sym < SLOT_NUM_SYMS; sym++)
    {
        fss_disabled_bfw_dyn_bfw_beam_id_cnt[cell_index][eaxcId_index][slot_3gpp][sym].store(0);
    }
}

int RU_Emulator::verify_extType11(uint8_t *section_ptr, oran_c_plane_info_t &c_plane_info, oran_c_plane_section_info_t &section_info, int cell_index)
{
    uint8_t *curr_ptr = section_ptr;
    uint8_t bfw_compression_bits = 9;
    uint16_t extLen = 0;
    uint16_t numBundPrb;
    int numPrbBundles = 0;
    auto ext_hdr = reinterpret_cast<oran_cmsg_ext_hdr *>(curr_ptr);
    auto ef = ext_hdr->ef.get();
    curr_ptr += sizeof(oran_cmsg_ext_hdr);

    auto ext_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_11 *>(curr_ptr);
    extLen = (((ext_ptr->extLen & 0xFF00) >> 8) | ((ext_ptr->extLen & 0x00FF) << 8)) << 2;
    curr_ptr += sizeof(oran_cmsg_sect_ext_type_11);

    if ((opt_bfw_dl_validation == 0 && c_plane_info.dir == DIRECTION_DOWNLINK) || (opt_bfw_ul_validation == 0 && c_plane_info.dir == DIRECTION_UPLINK))
    {
        return extLen;
    }

    if (opt_enable_mmimo)
    {
        // Check which TV
        if (c_plane_info.dir == DIRECTION_DOWNLINK)
        {
            c_plane_info.dl_tv_object = &bfw_dl_object;
        }
        else
        {
            c_plane_info.dl_tv_object = &bfw_ul_object;
        }
        auto &tv_object = c_plane_info.dl_tv_object;

        auto temp_comphdr_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr *>(curr_ptr);

        if (static_cast<aerial_fh::UserDataCompressionMethod>(temp_comphdr_ptr->bfwCompMeth.get()) != aerial_fh::UserDataCompressionMethod::NO_COMPRESSION)
        {
            if (c_plane_info.launch_pattern_slot < tv_object->launch_pattern.size())
            {
                // Check map contains key before accessing
                auto& map = tv_object->launch_pattern[c_plane_info.launch_pattern_slot];
                if (map.find(cell_index) != map.end())
                {
                    c_plane_info.tv_index = map.at(cell_index);
                }
                else
                {
                    re_warn("cell_index {} Invalid F{}S{}S{} Sym {} eAxC ID {} ", cell_index, c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, c_plane_info.eaxcId);
                    return extLen;
                }
            }
        }

        static const uint64_t delayed_ns = static_cast<uint64_t>(NS_X_US) * opt_tti_us * (launch_pattern_slot_size - 5);
        auto last_validation_ts = fss_dyn_bfw_beam_id_last_validation_ts[cell_index][c_plane_info.eaxcId_index][c_plane_info.launch_pattern_slot].load();
        if (last_validation_ts != 0 && (get_ns() - last_validation_ts) > delayed_ns)
        {
            const std::lock_guard<aerial_fh::FHMutex> lock(fss_dyn_bfw_beam_id_mtx[cell_index][c_plane_info.eaxcId_index][c_plane_info.launch_pattern_slot]);
            last_validation_ts = fss_dyn_bfw_beam_id_last_validation_ts[cell_index][c_plane_info.eaxcId_index][c_plane_info.launch_pattern_slot].load();
            if (last_validation_ts != 0 && (get_ns() - last_validation_ts) > delayed_ns)
            {
                dynamic_beamid_validation(c_plane_info, cell_index);
            }
            fss_dyn_bfw_beam_id_last_validation_ts[cell_index][c_plane_info.eaxcId_index][c_plane_info.launch_pattern_slot].store(get_ns());
        }
        else
        {
            fss_dyn_bfw_beam_id_last_validation_ts[cell_index][c_plane_info.eaxcId_index][c_plane_info.launch_pattern_slot].store(get_ns());
        }

        auto &tv_info = tv_object->tv_info[c_plane_info.tv_index];

        uint64_t portMask = 0;
        auto disableBFWs = oran_cmsg_get_ext_11_disableBFWs(ext_ptr);
        if (unlikely(ext_ptr->reserved.get() != 0))
        {
            re_warn("Ext11 header reserved bits not set to 0. F{}S{}S{} Sym {} eAxC ID {} ", c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, c_plane_info.eaxcId);
        }
        if (!disableBFWs)
        {
            auto comphdr_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr *>(curr_ptr);
            curr_ptr += sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr);

            // Calculate the number of bundles from prg size and numPrbs
            numBundPrb = ext_ptr->numBundPrb;
            if (numBundPrb == 0)
            {
                re_cons("numBundPrb is 0");
                if (opt_dlc_tb)
                {
                    section_info.error_status = -1;
                    return extLen;
                }
                else
                {
                    do_throw(sb() << "numBundPrb is 0");
                }
            }
            auto &startPrbc = section_info.startPrbc;
            auto numPrbc = section_info.numPrbc;
            if(numPrbc == 0) //all PRBs are used
            {
                numPrbc = cell_configs[cell_index].dlGridSize;
            }

            if (static_cast<aerial_fh::UserDataCompressionMethod>(comphdr_ptr->bfwCompMeth.get()) == aerial_fh::UserDataCompressionMethod::NO_COMPRESSION)
            {
                // static bfw validation
                int compressBitWidth = 16; // 9
                int L_TRX = cell_configs[cell_index].dbt_cfg.num_TRX_beamforming;
                int bundleIQByteSize = L_TRX * compressBitWidth * 2 / 8;

                uint8_t *packet_bundle_ptr = curr_ptr;
                int numPrbBundles = (numPrbc + numBundPrb - 1) / numBundPrb;
                auto seen = cell_configs[cell_index].dbt_cfg.static_beamIdx_seen;

                for (int i = 0; i < numPrbBundles; ++i)
                {
                    auto bfwCompParam_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_11_disableBFWs_0_bundle_uncompressed *>(packet_bundle_ptr);
                    if (unlikely(bfwCompParam_ptr->reserved.get()))
                    {
                        re_warn("Ext11 header(static bfw) reserved bits not set to 0. F{}S{}S{} Sym {} eAxC ID {} ", c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, c_plane_info.eaxcId);
                    }
                    uint16_t beam_id = bfwCompParam_ptr->beamId.get();
                    packet_bundle_ptr += 2;

                    if (seen.find(beam_id) != seen.end())
                    {

                        if (opt_dlc_tb)
                        {
                            re_cons("static beam id : {} seen before!", beam_id);
                            section_info.error_status = -1;
                            return extLen;
                        }
                        else
                        {
                            do_throw(sb() << "static beam id :" << beam_id << " seen before!");
                        }
                    }
                    seen[beam_id] = true;

                    for (int j = 0; j < L_TRX; j++)
                    {
                        if (cell_configs[cell_index].dbt_cfg.dbt_data_buf[(beam_id - 1) * L_TRX + j].re != *(int16_t *)(packet_bundle_ptr) || cell_configs[cell_index].dbt_cfg.dbt_data_buf[(beam_id - 1) * L_TRX + j].im != *(int16_t *)(packet_bundle_ptr + 2))
                        {
                            re_cons("Cell {} Flow {} F{}S{}S{} Symbol {} static bfw Bundle {} mismatch!", cell_index, c_plane_info.eaxcId_index, c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, i);
                            re_cons("tv:  re: {} im: {}", cell_configs[cell_index].dbt_cfg.dbt_data_buf[(beam_id - 1) * L_TRX + j].re, cell_configs[cell_index].dbt_cfg.dbt_data_buf[(beam_id - 1) * L_TRX + j].im);
                            re_cons("pkt: re: {} im: {}", *(int16_t *)(packet_bundle_ptr), *(int16_t *)(packet_bundle_ptr + 2));

                            if (opt_dlc_tb)
                            {
                                section_info.error_status = -1;
                                return extLen;
                            }
                            else
                            {
                                sleep(1);
                                do_throw(sb() << "beam_id : " << beam_id << " IQ sample idx: " << j << " Static beam bfw mismatch detected!");
                            }
                        }
                        packet_bundle_ptr += 4;
                    }
                }
                return extLen;
            }

            // TV sanity check
            if (c_plane_info.dir == DIRECTION_DOWNLINK)
            {
                if (opt_bfw_dl_validation != RE_ENABLED)
                {
                    re_warn("F{}S{}S{} Sym {} eAxC ID {} ", c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, c_plane_info.eaxcId);
                    re_warn("No DL BFW TV found!");
                    section_info.error_status = -1;
                    // sleep(1);
                    // do_throw(sb() << "No DL BFW TV found!");
                    return extLen;
                }
            }
            else if (opt_bfw_ul_validation != RE_ENABLED)
            {
                re_warn("F{}S{}S{} Sym {} eAxC ID {} ", c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, c_plane_info.eaxcId);
                re_warn("No UL BFW TV found!");
                section_info.error_status = -1;
                // sleep(1);
                // do_throw(sb() << "No UL BFW TV found!");
                return extLen;
            }

            // Check eAxC ID index DONE
            // FIXME Following assumptions are made:
            // - only one BFW TV is used per slot
            // - BFP 9 compression of BFWs

            int bfw_info_index = -1;
            for (int i = 0; i < tv_info.bfw_infos.size(); ++i)
            {
                auto bfw_start_rb = tv_info.bfw_infos[i].rbStart;
                auto bfw_end_rb = tv_info.bfw_infos[i].rbStart + tv_info.bfw_infos[i].rbSize;
                if (section_info.startPrbc >= bfw_start_rb && section_info.startPrbc + numPrbc <= bfw_end_rb)
                {
                    bfw_info_index = i;
                }
            }

            if (bfw_info_index == -1)
            {
                if (opt_dlc_tb)
                {
                    re_cons("Slot {} BFW received is not found in TV name: {} startPrbc: {} numPrbc: {} ", c_plane_info.launch_pattern_slot, tv_object->tv_names[c_plane_info.tv_index].c_str(), section_info.startPrbc, section_info.numPrbc);
                    section_info.error_status = -1;
                    return extLen;
                }
                else
                {
                    do_throw(sb() << "Slot " << c_plane_info.launch_pattern_slot << " BFW received is not found in TV name: " << tv_object->tv_names[c_plane_info.tv_index].c_str() << " startPrbc: " << section_info.startPrbc << " numPrbc: " << section_info.numPrbc << "\n");
                }
            }
            // else
            // {
            //     re_cons("F{}S{}S{} Sym {} bfw tv {} bfw_info_index {} ", c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, tv_object->tv_names[c_plane_info.tv_index].c_str(), bfw_info_index);
            // }

            auto &bfw_info = tv_info.bfw_infos[bfw_info_index];
            portMask = bfw_info.portMask[cell_index][c_plane_info.launch_pattern_slot];
            int prgSize = bfw_info.prgSize;
            int bfwPrbGrpSize = bfw_info.bfwPrbGrpSize;
            int rbStart = bfw_info.rbStart;
            int rbSize = bfw_info.rbSize;
            int numPRGs = bfw_info.numPRGs;
            int compressBitWidth = bfw_info.compressBitWidth; // 9

            // Invalid eAxC ID
            if (!c_plane_info.valid_eaxcId)
            {
                re_cons("F{}S{}S{} Sym {} Invalid eAxC ID seen {} ", c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, c_plane_info.eaxcId);
                if (opt_dlc_tb)
                {
                    section_info.error_status = -1;
                    return extLen;
                }
            }
            //re_cons("{}:{} portMask {} eaxcId_index {}", __FILE__, __LINE__, portMask, c_plane_info.eaxcId_index);
            // If the portMask does not have a
            if ((portMask & (1 << c_plane_info.eaxcId_index)) == 0)
            {
                re_cons("F{}S{}S{} Sym {} Unexpected BFW antenna received on antenna {} eaxc ID {} portMask {} c_plane_info.valid_eaxcId {}", c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, c_plane_info.eaxcId_index, c_plane_info.eaxcId, portMask, c_plane_info.valid_eaxcId ? "T" : "F");
                if (opt_dlc_tb)
                {
                    section_info.error_status = -1;
                    return extLen;
                }
                else
                {
                    sleep(1);
                    do_throw(sb() << " Unexpected BFW antenna received\n");
                }
            }
            //re_cons("F{}S{}S{} Sym {} BFW antenna received on antenna {} eaxc ID {} portMask {} c_plane_info.valid_eaxcId {}",
            //    c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, c_plane_info.eaxcId_index, c_plane_info.eaxcId, portMask, c_plane_info.valid_eaxcId ? "T" : "F");
            // Find the antenna offset out of the on bits in the portMask to use as the offset in the BFW buffer

            int bfw_ant_idx = bfw_info.active_eaxc_ids[cell_index][c_plane_info.launch_pattern_slot][c_plane_info.eaxcId_index];
            //re_cons("{}:{} cell_index {} slot {} eaxcId_index {} bfw_ant_idx {}", __FILE__, __LINE__, cell_index, c_plane_info.launch_pattern_slot, c_plane_info.eaxcId_index, bfw_ant_idx);
            int L_TRX = ORAN_SECT_EXT_11_L_TRX;
            if (c_plane_info.dir == DIRECTION_DOWNLINK)
            {
                L_TRX = pdsch_object.tv_info[c_plane_info.tv_index].numGnbAnt;
            }
            else
            {
                L_TRX = pusch_object.tv_info[c_plane_info.tv_index].numGnbAnt;
            }
            int bundleIQByteSize = L_TRX * compressBitWidth * 2 / 8;

            int qams_buffer_index = compressBitWidth;
            int sample_index = 0;
            for(int i = 0; i < c_plane_info.tv_index; ++i)
            {
                sample_index += tv_info.bfw_infos.size();
            }
            sample_index += bfw_info_index;
            auto slot_buf = (unsigned char *)tv_object->qams[qams_buffer_index][sample_index].data.get();
            int offset = 0;

            offset += bfw_ant_idx * numPRGs * (bundleIQByteSize + 1); // + 1 for exponent
            numBundPrb = bfwPrbGrpSize;
            int start_bundle_index = (startPrbc - rbStart) / numBundPrb;
            if(bfwPrbGrpSize == MAX_NUM_PRBS_PER_SYMBOL)
            {
                start_bundle_index = 0;
            }
            numPrbBundles = (numPrbc + numBundPrb - 1) / numBundPrb;
            offset += start_bundle_index * (bundleIQByteSize + 1); // + 1 for exponent
            uint8_t *tv_bundle_ptr = &slot_buf[offset];

            uint8_t *packet_bundle_ptr = curr_ptr;

            auto cur_prb = startPrbc;
            for (int i = 0; i < numPrbBundles; ++i)
            {
                // todo validate bfwCompParam
                auto bfwCompParam_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_11_disableBFWs_0_bfp_compressed_bundle_hdr *>(packet_bundle_ptr);
                uint8_t rx_exp = bfwCompParam_ptr->bfwCompParam.exponent.get();

                if (unlikely(bfwCompParam_ptr->bfwCompParam.reserved.get()))
                {
                    re_warn("Ext11 header(disableBFWs_0) reserved bits not set to 0. F{}S{}S{} Sym {} eAxC ID {} ", c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, c_plane_info.eaxcId);
                }

                auto beamId = bfwCompParam_ptr->beamId.get();
                if ((beamId < oran_beam_id_info.dynamic_beam_id_start) || (beamId > oran_beam_id_info.dynamic_beam_id_end))
                {
                    re_cons("Erroneous dynamic beam Id detected, disableBFWs = 0, beamId: {}", beamId);
                    if (opt_dlc_tb)
                    {
                        section_info.error_status = -1;
                        return extLen;
                    }
                    else
                    {
                        sleep(1);
                        do_throw(sb() << "Erroneous dynamic beam Id detected, 'disableBFWs = 0 !");
                    }
                }

                if (bfwPrbGrpSize == MAX_NUM_PRBS_PER_SYMBOL)
                {
                    cell_configs[cell_index].dyn_bfw_beam_id_with_full_bw[c_plane_info.eaxcId_index].insert(beamId);
                }
                else
                {
                    fss_dyn_bfw_beam_id[cell_index][c_plane_info.eaxcId_index][c_plane_info.launch_pattern_slot][cur_prb][0] = beamId;
                    fss_dyn_bfw_beam_id[cell_index][c_plane_info.eaxcId_index][c_plane_info.launch_pattern_slot][cur_prb][1] = (uint16_t)bfwPrbGrpSize;
                }

                cur_prb += numBundPrb;

                bfwCompParam_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_11_disableBFWs_0_bfp_compressed_bundle_hdr *>(tv_bundle_ptr);
                uint8_t tv_exp = bfwCompParam_ptr->bfwCompParam.exponent.get();
                ++tv_bundle_ptr;

                packet_bundle_ptr += 1;
                // todo validate beamID TV does not include beamID
                packet_bundle_ptr += 2;
                // int comp = decompress_and_compare_approx_bfw_bundle_buffer(packet_bundle_ptr, rx_exp, tv_bundle_ptr, tv_exp, bundleIQByteSize, compressBitWidth, 2048,
                //     c_plane_info.eaxcId_index, c_plane_info.startSym, start_bundle_index + i, true);
                int comp = memcmp(packet_bundle_ptr, tv_bundle_ptr, bundleIQByteSize);
                if (comp != 0)
                {
                    int skip_ul_comp = 1;
#if 1
                    if ((c_plane_info.dir == DIRECTION_UPLINK) && (L_TRX == ORAN_SECT_EXT_11_L_TRX))
                    {
                        // Check if packet_bundle_ptr contains all zeros
                        const bool all_zeros = std::all_of(
                            packet_bundle_ptr, 
                            packet_bundle_ptr + bundleIQByteSize,
                            [](uint8_t byte) { return byte == 0; }
                        );
                        skip_ul_comp = all_zeros ? 0 : 1;
                        if (skip_ul_comp != 0)
                        {
                            re_info("UL BFW mismatch comp {} skip_ul_comp {}!", comp, skip_ul_comp);
                        }
                        else
                        {
                            comp = 0;
                            re_dbg("SKIP UL BFW COMPARISION AS NO WEIGHTS ARE ADDED FOR THIS EAXC_ID!");
                        }
                    }
#endif

                    if (skip_ul_comp != 0)
                    {
                        comp = fixedpt_bundle_compare(packet_bundle_ptr, tv_bundle_ptr, bundleIQByteSize, rx_exp, tv_exp);
                        // comp = decompress_and_compare_approx_bfw_bundle_buffer(packet_bundle_ptr, rx_exp, tv_bundle_ptr, tv_exp, bundleIQByteSize, compressBitWidth, 2048, c_plane_info.eaxcId_index, c_plane_info.startSym, start_bundle_index + i, true, L_TRX);
                        if (comp != 0)
                        {
                            re_info("Cell {} Flow {} F{}S{}S{} Symbol {} Bundle {} compressBitWidth {} mismatch!", cell_index, c_plane_info.eaxcId_index, c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, start_bundle_index + i, compressBitWidth);
                            // Below two lines can be enabled to debug the mismatch
                            //sleep(2);
                            //raise(SIGINT);
                        }
                    }
                }
                tv_object->invalid_flag[cell_index][c_plane_info.launch_pattern_slot] = (tv_object->invalid_flag[cell_index][c_plane_info.launch_pattern_slot] || comp);
                packet_bundle_ptr += bundleIQByteSize;
                tv_bundle_ptr += bundleIQByteSize; // +1 for exponent
            }
            tv_object->atomic_received_prbs[c_plane_info.launch_pattern_slot][cell_index] += numPrbc;
        }
        else
        {
            // Calculate the number of bundles from prg size and numPrbs
            numBundPrb = ext_ptr->numBundPrb;

            auto &startPrbc = section_info.startPrbc;
            auto numPrbc = section_info.numPrbc;
            if(numPrbc == 0) //all PRBs are used
            {
                numPrbc = cell_configs[cell_index].dlGridSize;
            }

            // Check eAxC ID index DONE
            // FIXME Following assumptions are made:
            // - only one BFW TV is used per slot
            // - BFP 9 compression of BFWs

            numPrbBundles = (numPrbc + numBundPrb - 1) / numBundPrb;
            uint8_t *packet_bundle_ptr = curr_ptr;

            auto & disabled_bfw_dyn_bfw_beam_id = fss_disabled_bfw_dyn_bfw_beam_id[cell_index][c_plane_info.eaxcId_index][c_plane_info.launch_pattern_slot][c_plane_info.startSym];
            auto cur_prb = startPrbc;
            {
                const std::lock_guard<aerial_fh::FHMutex> lock(fss_dyn_bfw_beam_id_mtx[cell_index][c_plane_info.eaxcId_index][c_plane_info.launch_pattern_slot]);
                auto& beam_id_cnt = fss_disabled_bfw_dyn_bfw_beam_id_cnt[cell_index][c_plane_info.eaxcId_index][c_plane_info.launch_pattern_slot][c_plane_info.startSym];
                for (int i = 0; i < numPrbBundles; ++i)
                {
                    auto bundle_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_11_disableBFWs_1_bundle *>(packet_bundle_ptr);
                    packet_bundle_ptr += sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_1_bundle);

                    if (unlikely(bundle_ptr->reserved.get()))
                    {
                        re_warn("Ext11 header(disableBFWs_1) reserved bits not set to 0. F{}S{}S{} Sym {} eAxC ID {} ", c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, c_plane_info.eaxcId);
                    }

                    auto beamId = bundle_ptr->beamId.get();

                    int cnt = beam_id_cnt.fetch_add(1);
                    disabled_bfw_dyn_bfw_beam_id[cnt][0] = cur_prb;
                    disabled_bfw_dyn_bfw_beam_id[cnt][1] = beamId;
                    //disabled_bfw_dyn_bfw_beam_id[cnt][2] = get_ns();
                    //disabled_bfw_dyn_bfw_beam_id[cnt][3] = cnt;
                    cur_prb += numBundPrb;
                }
            }

            return extLen;
        }

        bool complete = false;
        {
            const std::lock_guard<aerial_fh::FHMutex> lock(tv_object->mtx[cell_index]);
            //re_cons("BFW received_prbs  {} , total_expected_prbs {}", tv_object->atomic_received_prbs[c_plane_info.launch_pattern_slot][cell_index].load(), tv_info.total_expected_prbs[cell_index][c_plane_info.launch_pattern_slot]);
            //Why would 'tv_info.total_expected_prbs[cell_index][c_plane_info.launch_pattern_slot]' become 0 ?
            if (tv_info.total_expected_prbs[cell_index][c_plane_info.launch_pattern_slot] > 0 && tv_object->atomic_received_prbs[c_plane_info.launch_pattern_slot][cell_index].load() >= tv_info.total_expected_prbs[cell_index][c_plane_info.launch_pattern_slot])
            {
                tv_object->atomic_received_prbs[c_plane_info.launch_pattern_slot][cell_index].store(0);
                complete = true;
            }
        }

        if (complete)
        {
            if (tv_object->invalid_flag[cell_index][c_plane_info.launch_pattern_slot])
            {
                re_cons("BFW Complete Cell {} 3GPP slot {} F{} S{} S{} Payload Validation {}",
                        cell_index, c_plane_info.launch_pattern_slot, c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId,
                        tv_object->invalid_flag[cell_index][c_plane_info.launch_pattern_slot] ? "ERROR" : "OK");
                ++tv_object->error_slot_counters[cell_index];
            }
            else
            {
                re_info("BFW Complete Cell {} 3GPP slot {} F{} S{} S{} OK", cell_index, c_plane_info.launch_pattern_slot, c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId);
                ++tv_object->throughput_slot_counters[cell_index];
                ++tv_object->good_slot_counters[cell_index];
            }
            ++tv_object->total_slot_counters[cell_index];
            tv_object->invalid_flag[cell_index][c_plane_info.launch_pattern_slot] = false;
        }
    }
    else
    {
        if (unlikely(ext_ptr->reserved.get() != 0))
        {
            re_warn("Ext11 header reserved bits not set to 0. F{}S{}S{} Sym {} eAxC ID {} ", c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, c_plane_info.eaxcId);
        }
        if (oran_cmsg_get_ext_11_disableBFWs(ext_ptr))
        {
            re_cons("Unsupported disableBFWs detected!");
            do_throw(sb() << "Unsupported disableBFWs detected!");
        }

        auto comphdr_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr *>(curr_ptr);
        curr_ptr += sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr);

        // Calculate the number of bundles from prg size and numPrbs
        numBundPrb = ext_ptr->numBundPrb;

        auto &startPrbc = section_info.startPrbc;
        auto &numPrbc = section_info.numPrbc;

        if (static_cast<aerial_fh::UserDataCompressionMethod>(comphdr_ptr->bfwCompMeth.get()) == aerial_fh::UserDataCompressionMethod::NO_COMPRESSION)
        {
            // static bfw validation
            int compressBitWidth = 16; // 9
            int L_TRX = cell_configs[cell_index].dbt_cfg.num_TRX_beamforming;
            int bundleIQByteSize = L_TRX * compressBitWidth * 2 / 8;

            uint8_t *packet_bundle_ptr = curr_ptr;
            int numPrbBundles = (numPrbc + numBundPrb - 1) / numBundPrb;
            auto seen = cell_configs[cell_index].dbt_cfg.static_beamIdx_seen;

            for (int i = 0; i < numPrbBundles; ++i)
            {
                auto bfwCompParam_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_11_disableBFWs_0_bundle_uncompressed *>(packet_bundle_ptr);
                if (unlikely(bfwCompParam_ptr->reserved.get()))
                {
                    re_warn("Ext11 header(static bfw) reserved bits not set to 0. F{}S{}S{} Sym {} eAxC ID {} ", c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, c_plane_info.eaxcId);
                }
                uint16_t beam_id = bfwCompParam_ptr->beamId.get();
                packet_bundle_ptr += 2;

                if (seen.find(beam_id) != seen.end())
                {
                    do_throw(sb() << "static beam id :" << beam_id << " seen before!");
                }
                seen[beam_id] = true;

                for (int j = 0; j < L_TRX; j++)
                {
                    if (cell_configs[cell_index].dbt_cfg.dbt_data_buf[(beam_id - 1) * L_TRX + j].re != *(int16_t *)(packet_bundle_ptr) || cell_configs[cell_index].dbt_cfg.dbt_data_buf[(beam_id - 1) * L_TRX + j].im != *(int16_t *)(packet_bundle_ptr + 2))
                    {
                        re_cons("Cell {} Flow {} F{}S{}S{} Symbol {} static bfw Bundle {} mismatch!", cell_index, c_plane_info.eaxcId_index, c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, i);
                        re_cons("tv:  re: {} im: {}", cell_configs[cell_index].dbt_cfg.dbt_data_buf[(beam_id - 1) * L_TRX + j].re, cell_configs[cell_index].dbt_cfg.dbt_data_buf[(beam_id - 1) * L_TRX + j].im);
                        re_cons("pkt: re: {} im: {}", *(int16_t *)(packet_bundle_ptr), *(int16_t *)(packet_bundle_ptr + 2));
                        sleep(1);
                        do_throw(sb() << "beam_id : " << beam_id << " IQ sample idx: " << j << " Static beam bfw mismatch detected!");
                    }
                    packet_bundle_ptr += 4;
                }
            }
            return extLen;
        }

        // Check eAxC ID index DONE

        // Check which TV
        if (c_plane_info.dir == DIRECTION_DOWNLINK)
        {
            c_plane_info.dl_tv_object = &bfw_dl_object;
        }
        else
        {
            c_plane_info.dl_tv_object = &bfw_ul_object;
        }

        // FIXME Following assumptions are made:
        // - only one BFW TV is used per slot
        // - disableBFW = 0
        // - BFP 9 compression of BFWs
        auto &tv_object = c_plane_info.dl_tv_object;
        c_plane_info.tv_index = tv_object->launch_pattern[c_plane_info.launch_pattern_slot][cell_index];

        auto &tv_info = tv_object->tv_info[c_plane_info.tv_index];
        int prgSize = tv_info.bfw_infos[0].prgSize;
        int rbStart = tv_info.bfw_infos[0].rbStart;
        int rbSize = tv_info.bfw_infos[0].rbSize;
        int numPRGs = tv_info.bfw_infos[0].numPRGs;
        int compressBitWidth = tv_info.bfw_infos[0].compressBitWidth; // 9
        int L_TRX = ORAN_SECT_EXT_11_L_TRX;
        if (c_plane_info.dir == DIRECTION_DOWNLINK)
        {
            L_TRX = pdsch_object.tv_info[c_plane_info.tv_index].numGnbAnt;
        }
        else
        {
            L_TRX = pusch_object.tv_info[c_plane_info.tv_index].numGnbAnt;
        }
        int bundleIQByteSize = L_TRX * compressBitWidth * 2 / 8;

        int qams_buffer_index = compressBitWidth;
        auto slot_buf = (unsigned char *)tv_object->qams[qams_buffer_index][c_plane_info.tv_index].data.get();
        int offset = 0;
        offset += c_plane_info.eaxcId_index * numPRGs * (bundleIQByteSize + 1); // + 1 for exponent

        int start_bundle_index = (startPrbc - rbStart) / numBundPrb;
        int numPrbBundles = (numPrbc + numBundPrb - 1) / numBundPrb;
        offset += start_bundle_index * (bundleIQByteSize + 1); // + 1 for exponent
        uint8_t *tv_bundle_ptr = &slot_buf[offset];

        uint8_t *packet_bundle_ptr = curr_ptr;
        for (int i = 0; i < numPrbBundles; ++i)
        {
            // todo validate bfwCompParam
            auto bfwCompParam_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_11_disableBFWs_0_bfp_compressed_bundle_hdr *>(packet_bundle_ptr);
            if (unlikely(bfwCompParam_ptr->bfwCompParam.reserved.get()))
            {
                re_warn("Ext11 header(disableBFWs_0) reserved bits not set to 0. F{}S{}S{} Sym {} eAxC ID {} ", c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, c_plane_info.eaxcId);
            }

            uint8_t rx_exp = bfwCompParam_ptr->bfwCompParam.exponent.get();
            bfwCompParam_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_11_disableBFWs_0_bfp_compressed_bundle_hdr *>(tv_bundle_ptr);
            uint8_t tv_exp = bfwCompParam_ptr->bfwCompParam.exponent.get();
            ++tv_bundle_ptr;

            packet_bundle_ptr += 1;
            // todo validate beamID TV does not include beamID
            packet_bundle_ptr += 2;
            // int comp = decompress_and_compare_approx_bfw_bundle_buffer(packet_bundle_ptr, rx_exp, tv_bundle_ptr, tv_exp, bundleIQByteSize, compressBitWidth, 2048,
            //     c_plane_info.eaxcId_index, c_plane_info.startSym, start_bundle_index + i, true);
            int comp = memcmp(packet_bundle_ptr, tv_bundle_ptr, bundleIQByteSize);
            if (comp != 0)
            {
                int skip_ul_comp = 1;
#if 1
                if ((c_plane_info.dir == DIRECTION_UPLINK) && (L_TRX == ORAN_SECT_EXT_11_L_TRX))
                {
                    // Check if packet_bundle_ptr contains all zeros
                    const bool all_zeros = std::all_of(
                        packet_bundle_ptr, 
                        packet_bundle_ptr + bundleIQByteSize,
                        [](uint8_t byte) { return byte == 0; }
                    );
                    skip_ul_comp = all_zeros ? 0 : 1;
                    if (skip_ul_comp != 0)
                    {
                        re_info("UL BFW mismatch comp {} skip_ul_comp {}!", comp, skip_ul_comp);
                    }
                    else
                    {
                        comp = 0;
                        re_dbg("SKIP UL BFW COMPARISION AS NO WEIGHTS ARE ADDED FOR THIS EAXC_ID!");
                    }
                }
#endif

                if (skip_ul_comp != 0)
                {
                    comp = fixedpt_bundle_compare(packet_bundle_ptr, tv_bundle_ptr, bundleIQByteSize, rx_exp, tv_exp);
                    // comp = decompress_and_compare_approx_bfw_bundle_buffer(packet_bundle_ptr, rx_exp, tv_bundle_ptr, tv_exp, bundleIQByteSize, compressBitWidth, 2048, c_plane_info.eaxcId_index, c_plane_info.startSym, start_bundle_index + i, true, L_TRX);
                    if (comp != 0)
                    {
                        re_cons("Cell {} Flow {} F{}S{}S{} Symbol {} Bundle {} mismatch!", cell_index, c_plane_info.eaxcId_index, c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, start_bundle_index + i);
                        // Below two lines can be enabled to debug the mismatch
                        //sleep(2);
                        //raise(SIGINT);
                    }
                }
            }
            tv_object->invalid_flag[cell_index][c_plane_info.launch_pattern_slot] = (tv_object->invalid_flag[cell_index][c_plane_info.launch_pattern_slot] || comp);
            packet_bundle_ptr += bundleIQByteSize;
            tv_bundle_ptr += bundleIQByteSize; // +1 for exponent
        }
        tv_object->atomic_received_prbs[c_plane_info.launch_pattern_slot][cell_index] += numPrbBundles;
        bool complete = false;
        {
            const std::lock_guard<aerial_fh::FHMutex> lock(tv_object->mtx[cell_index]);
            int num_flows = c_plane_info.dir == oran_pkt_dir::DIRECTION_DOWNLINK ? cell_configs[cell_index].num_dl_flows : cell_configs[cell_index].num_ul_flows;
            if (tv_object->atomic_received_prbs[c_plane_info.launch_pattern_slot][cell_index].load() >= tv_info.bfw_infos[0].numPRGs * num_flows)
            {
                tv_object->atomic_received_prbs[c_plane_info.launch_pattern_slot][cell_index].store(0);
                complete = true;
            }
        }

        if (complete)
        {
            if (tv_object->invalid_flag[cell_index][c_plane_info.launch_pattern_slot])
            {
                re_cons("BFW Complete Cell {} 3GPP slot {} F{} S{} S{} Payload Validation {}",
                        cell_index, c_plane_info.launch_pattern_slot, c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId,
                        tv_object->invalid_flag[cell_index][c_plane_info.launch_pattern_slot] ? "ERROR" : "OK");
                ++tv_object->error_slot_counters[cell_index];
            }
            else
            {
                re_info("BFW Complete Cell {} 3GPP slot {} F{} S{} S{} OK", cell_index, c_plane_info.launch_pattern_slot, c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId);
                ++tv_object->throughput_slot_counters[cell_index];
                ++tv_object->good_slot_counters[cell_index];
            }
            ++tv_object->total_slot_counters[cell_index];
            tv_object->invalid_flag[cell_index][c_plane_info.launch_pattern_slot] = false;
        }
    }
    return extLen;
}

/**
 * @brief Receives packets from the control plane interface
 * @param[in,out] cell_index Current cell index
 * @param[in,out] info Array to store received packet information
 * @param[in,out] nb_rx Number of received packets
 * @param[in,out] rte_rx_time Receive call completion time
 * @param[in] threadname Thread name for logging
 * @param[in,out] current_state Current worker state
 * @param[in,out] previous_state Previous worker state
 * @param[in] first_f0s0s0_time First frame/subframe/slot time anchor
 * @param[in,out] round_robin_counter Round robin counter for multi-cell processing
 * @param[in] start_cell_index Starting cell index
 * @param[in] num_cells_per_core Number of cells per core
 * @param[in,out] tx_request TX request handle
 * @param[in] tx_request_init Whether TX request is initialized
 * @param[in] is_srs Whether this is an SRS thread
 * @return Number of packets received
 */
size_t RU_Emulator::receive_packets(uint8_t& cell_index, aerial_fh::MsgReceiveInfo* info, size_t& nb_rx, uint64_t& rte_rx_time,
                                   const char* threadname, WorkerState& current_state, WorkerState& previous_state,
                                   uint64_t first_f0s0s0_time, uint8_t& round_robin_counter,
                                   int start_cell_index, int num_cells_per_core,
                                   aerial_fh::TxRequestHandle& tx_request, bool tx_request_init, bool is_srs)
{
    // Helper function to convert WorkerState to string
    auto workerStateToString = [](WorkerState state) -> const char* {
        switch (state) {
            case WorkerState::RECEIVING:    return "RECEIVING";
            case WorkerState::SPINNING:     return "SPINNING";
            case WorkerState::PROCESSING:   return "PROCESSING";
            case WorkerState::TRANSMITTING: return "TRANSMITTING";
            default:                        return "UNKNOWN";
        }
    };

    while (!check_force_quit() && (nb_rx == 0) && (cell_index < opt_num_cells))
    {
        if(opt_enable_cplane_worker_tracing && current_state != WorkerState::RECEIVING && current_state != WorkerState::SPINNING)
        {
            previous_state = current_state;
            current_state = WorkerState::RECEIVING;
            uint64_t state_transition_time = get_ns();
            NVLOGI_FMT(TAG_CP_WORKER_TRACING, "Worker {} Cell {} state transition: {} -> RECEIVING | Transition time: {} | Delta from time ref: {}",
                        threadname,
                        cell_index,
                        workerStateToString(previous_state),
                        state_transition_time,
                        state_transition_time - first_f0s0s0_time);
        }

        if(opt_enable_mmimo)
        {
            nb_rx = CPLANE_DEQUEUE_BURST_SIZE_MMIMO;
        }
        else
        {
            nb_rx = CPLANE_DEQUEUE_BURST_SIZE;
        }

#ifdef STANDALONE
        void* standalone_mbufs[CPLANE_DEQUEUE_BURST_SIZE];
        nb_rx = aerial_fh::ring_dequeue_burst_mbufs_payload_offset(standalone_c_plane_rings[cell_index], standalone_mbufs, &info[0], nb_rx);
#else
        aerial_fh::receive(peer_list[cell_index],  &info[0], &nb_rx, is_srs);
        rte_rx_time = get_ns();
        if(nb_rx == 0)
        {
            // State transition to SPINNING
            if (opt_enable_cplane_worker_tracing && current_state != WorkerState::SPINNING)
            {
                previous_state = current_state;
                current_state = WorkerState::SPINNING;
                uint64_t state_transition_time = get_ns();

                NVLOGI_FMT(TAG_CP_WORKER_TRACING, "Worker {} Cell {} state transition: {} -> SPINNING | Transition time: {} | Delta from time ref: {} | No packets to process",
                           threadname,
                           cell_index,
                           workerStateToString(previous_state),
                           state_transition_time,
                           state_transition_time - first_f0s0s0_time);
            }

            // Preallocate mbuf for TX when nb_rx == 0, we assume it is a down time for the thread, so it can spend some cycles to pre-allocate the mbufs
            // needed for UL TX, 512 is a heuristic for the UL peak patterns for now.
            if (tx_request_init == true)
            {
                aerial_fh::preallocate_mbufs(peer_list[cell_index], &tx_request, 1024);
            }

            if (num_cells_per_core > 1)
            {
                ++round_robin_counter;
                round_robin_counter = round_robin_counter % (int)num_cells_per_core;
            }
            cell_index = start_cell_index + round_robin_counter;
            usleep(10); // Busy spinning has caused other stalls in the system when running with other applications.
            // Issue can be reproduced by running the loopback test, DU cores will see "random" stalls when the usleep is commented out.
        }
#endif
    }

    return nb_rx;
}

size_t RU_Emulator::parse_c_plane_section_extension(oran_c_plane_section_ext_info_t& section_ext_info, uint8_t* section_ext_ptr)
{
    uint8_t* next = section_ext_ptr;
    auto ext_hdr = reinterpret_cast<oran_cmsg_ext_hdr*>(next);
    section_ext_info.ext_ptr = next;
    next += sizeof(oran_cmsg_ext_hdr);

    switch(ext_hdr->extType.get())
    {
        case ORAN_CMSG_SECTION_EXT_TYPE_11:
        {
            auto ext_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_11 *>(next);
            auto ext_len = (((ext_ptr->extLen & 0xFF00) >> 8) | ((ext_ptr->extLen & 0x00FF) << 8)) << 2;
            section_ext_info.ext_len = ext_len;
            section_ext_info.ext_type = ORAN_CMSG_SECTION_EXT_TYPE_11;
            break;
        }
        case ORAN_CMSG_SECTION_EXT_TYPE_4:
        {
            auto ext_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_4 *>(next);
            auto ext_len = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_4);
            section_ext_info.ext_len = ext_len;
            section_ext_info.ext_type = ORAN_CMSG_SECTION_EXT_TYPE_4;
            break;
        }
        case ORAN_CMSG_SECTION_EXT_TYPE_5:
        {
            auto ext_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_5 *>(next);
            auto ext_len = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_5);
            section_ext_info.ext_len = ext_len;
            section_ext_info.ext_type = ORAN_CMSG_SECTION_EXT_TYPE_5;
            break;
        }
        default:
            do_throw(sb() << "Unsupported extension type: " << (int)ext_hdr->extType.get());
            break;
    }

    section_ext_info.ef = ext_hdr->ef.get();
    return section_ext_info.ext_len;
}

size_t RU_Emulator::parse_c_plane_section_extensions(oran_c_plane_section_info_t& section_info, uint8_t* section_ext_ptr)
{
    int ef = section_info.ef;
    section_info.ext_infos_size = 0;
    size_t total_ext_len = 0;
    uint8_t* next = section_ext_ptr;
    while(ef)
    {
        oran_c_plane_section_ext_info_t& section_ext_info = section_info.ext_infos[section_info.ext_infos_size++];
        size_t ext_len = parse_c_plane_section_extension(section_ext_info, next);
        total_ext_len += ext_len;
        next += ext_len;
        ef = section_ext_info.ef;
    }
    return total_ext_len;
}

size_t RU_Emulator::parse_c_plane_section(oran_c_plane_section_info_t& section_info, int section_type, uint8_t* section_ptr)
{
    size_t section_len = 0;
    if(section_type != ORAN_CMSG_SECTION_TYPE_1 && section_type != ORAN_CMSG_SECTION_TYPE_3)
    {
        do_throw(sb() << "Section type error: "<< (int)section_type);
    }
    uint8_t* next = section_ptr;
    if(section_type == ORAN_CMSG_SECTION_TYPE_1)
    {
        auto section = reinterpret_cast<oran_cmsg_sect1*>(next);
        section_info.section_id     = section->sectionId;
        section_info.rb             = section->rb;
        section_info.symInc         = section->symInc;
        section_info.startPrbc      = section->startPrbc;
        section_info.numPrbc        = section->numPrbc;
        section_info.reMask         = section->reMask;
        section_info.numSymbol      = section->numSymbol;
        section_info.ef             = oran_cmsg_get_section_1_ef(section) ? 1 : 0;
        section_info.beamId         = section->beamId;
        section_info.error_status   = 0;
        next += sizeof(oran_cmsg_sect1);
        section_len += sizeof(oran_cmsg_sect1);
        auto ext_len = parse_c_plane_section_extensions(section_info, next);
        section_len += ext_len;
        next += ext_len;
    }
    else
    {
        auto section = reinterpret_cast<oran_cmsg_sect3*>(next);
        section_info.section_id     = section->sectionId;
        section_info.rb             = section->rb;
        section_info.symInc         = section->symInc;
        section_info.startPrbc      = section->startPrbc;
        section_info.numPrbc        = section->numPrbc;
        section_info.reMask         = section->reMask;
        section_info.numSymbol      = section->numSymbol;
        section_info.ef             = section->ef;
        section_info.beamId         = section->beamId;
        section_info.error_status   = 0;
        section_info.freqOffset     = section->freqOffset;
        section_info.reserved       = section->reserved.get();
        // 2s complement conversion from 24 bits to 32 bits
        if(section_info.freqOffset >> 23 == 1)
        {
            section_info.freqOffset |= 0b11111111000000000000000000000000;
        }
        next += sizeof(oran_cmsg_sect3);
        section_len += sizeof(oran_cmsg_sect3);

        auto ext_len = parse_c_plane_section_extensions(section_info, next);
        section_len += ext_len;
        next += ext_len;
    }
    return section_len;
}

void RU_Emulator::parse_c_plane(oran_c_plane_info_t& c_plane_info, int nb_rx, int index_rx, uint64_t rte_rx_time, uint8_t* mbuf_payload, size_t buffer_length, int cell_index)
{
    if (buffer_length < ORAN_CMSG_HDR_OFFSET + sizeof(oran_cmsg_radio_app_hdr))
    {
        do_throw(sb() << "Buffer too small for C-plane header");
    }
    c_plane_info.nb_rx              = nb_rx;
    c_plane_info.rx_index           = index_rx;
    c_plane_info.rte_rx_time        = rte_rx_time;
    c_plane_info.packet_processing_start = get_ns();
    c_plane_info.section_type       = oran_cmsg_get_section_type(mbuf_payload);
    c_plane_info.fss.frameId        = oran_cmsg_get_frame_id(mbuf_payload);
    c_plane_info.fss.subframeId     = oran_cmsg_get_subframe_id(mbuf_payload);
    c_plane_info.fss.slotId         = oran_cmsg_get_slot_id(mbuf_payload);
    c_plane_info.numberOfSections   = oran_cmsg_get_number_of_sections(mbuf_payload);
    c_plane_info.eaxcId             = oran_msg_get_flowid(mbuf_payload);
    c_plane_info.startSym           = oran_cmsg_get_startsymbol_id(mbuf_payload);
    c_plane_info.launch_pattern_slot = fss_to_launch_pattern_slot(c_plane_info.fss, launch_pattern_slot_size);
    c_plane_info.dir                = oran_msg_get_data_direction(mbuf_payload);

    // Needed for 4TR single section
    c_plane_info.numSym             = oran_cmsg_get_numsymbol(mbuf_payload, c_plane_info.section_type);
    c_plane_info.startPrbc          = oran_cmsg_get_startprbc(mbuf_payload, c_plane_info.section_type);
    c_plane_info.numPrbc            = oran_cmsg_get_numprbc(mbuf_payload, c_plane_info.section_type);
    if(unlikely(c_plane_info.startSym >= ORAN_ALL_SYMBOLS))
    {
        do_throw(sb() << "C Plane pkt startSym exceeds 14... ");
    }

    uint8_t* next = mbuf_payload;
    if(c_plane_info.section_type == ORAN_CMSG_SECTION_TYPE_1)
    {
        if (unlikely(opt_sectionid_validation == RE_ENABLED))
        {
            auto* common_hdr = reinterpret_cast<oran_cmsg_sect1_common_hdr*>(mbuf_payload + ORAN_CMSG_HDR_OFFSET);
            c_plane_info.udCompHdr = common_hdr->udCompHdr;
        }
        auto reserved_field = oran_msg_get_sect1_common_hdr_reserved_field(mbuf_payload);
        if (unlikely(reserved_field))
        {
            re_warn("CPlane section type 1 header reserved bits not set to 0. F{}S{}S{} Sym {} eAxC ID {} ", c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, c_plane_info.eaxcId);
        }
        next += ORAN_CMSG_SECT1_FIELDS_OFFSET;
    }
    else if(c_plane_info.section_type == ORAN_CMSG_SECTION_TYPE_3)
    {
        if (unlikely(opt_sectionid_validation == RE_ENABLED))
        {
            auto* common_hdr = reinterpret_cast<oran_cmsg_sect3_common_hdr*>(mbuf_payload + ORAN_CMSG_HDR_OFFSET);
            c_plane_info.udCompHdr = common_hdr->udCompHdr;
        }
        next += ORAN_CMSG_SECT3_FIELDS_OFFSET;
    }
    else
    {
        do_throw(sb() << "Section type error: "<< (int)c_plane_info.section_type);
    }

    c_plane_info.section_infos_size = 0;
    for(int i = 0; i < c_plane_info.numberOfSections; ++i)
    {
        oran_c_plane_section_info_t& section_info = c_plane_info.section_infos[c_plane_info.section_infos_size++];
        size_t section_len = parse_c_plane_section(section_info, c_plane_info.section_type, next);
        next += section_len;
        if (unlikely(c_plane_info.section_type == ORAN_CMSG_SECTION_TYPE_3 && section_info.reserved))
        {
            re_warn("CPlane section type 3 header reserved bits not set to 0. F{}S{}S{} Sym {} eAxC ID {} ", c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId, c_plane_info.startSym, c_plane_info.eaxcId);
        }
    }

    if(c_plane_info.dir == DIRECTION_UPLINK)
    {
        if (c_plane_info.section_type == ORAN_CMSG_SECTION_TYPE_1)
        {
            find_channel_type_for_each_section(c_plane_info, cell_index);
        }
        else
        {
            c_plane_info.channel_type = ul_channel::PRACH;
            c_plane_info.tv_object = &prach_object;
        }
        find_eAxC_index(c_plane_info, cell_configs[cell_index]);

        if(c_plane_info.channel_type != ul_channel::SRS)
        {
            c_plane_info.tx_offset = oran_timing_info.ul_u_plane_tx_offset * NS_X_US;
        }
        else
        {
            c_plane_info.tx_offset = oran_timing_info.ul_u_plane_tx_offset_srs * NS_X_US;
        }
    }
}

void RU_Emulator::track_stats(oran_c_plane_info_t& c_plane_info, uint16_t cell_index)
{
    if(opt_max_sect_stats)
    {
        const std::lock_guard<aerial_fh::FHMutex> lock(cell_mtx[section_type::C_SECTION_TYPE_AGGR][cell_index]);
        auto cur_t = get_ns();
        if (cur_t - fss_received_sections_prev_ts[section_type::C_SECTION_TYPE_AGGR][cell_index][c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId] > 10000000)
        {
            fss_received_sections[section_type::C_SECTION_TYPE_AGGR][cell_index][c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId] = 0;
        }
        fss_received_sections_prev_ts[section_type::C_SECTION_TYPE_AGGR][cell_index][c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId] = cur_t;
        fss_received_sections[section_type::C_SECTION_TYPE_AGGR][cell_index][c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId] += c_plane_info.numberOfSections;
        max_sections_per_slot[section_type::C_SECTION_TYPE_AGGR][cell_index] = std::max(max_sections_per_slot[section_type::C_SECTION_TYPE_AGGR][cell_index], fss_received_sections[section_type::C_SECTION_TYPE_AGGR][cell_index][c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId]);
    }

    if(c_plane_info.dir == DIRECTION_DOWNLINK)
    {
        if(opt_max_sect_stats)
        {
            const std::lock_guard<aerial_fh::FHMutex> lock(cell_mtx[section_type::DL_C_SECTION_TYPE_1][cell_index]);
            auto cur_t = get_ns();
            if (cur_t - fss_received_sections_prev_ts[section_type::DL_C_SECTION_TYPE_1][cell_index][c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId] > 10000000)
            {
                fss_received_sections[section_type::DL_C_SECTION_TYPE_1][cell_index][c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId] = 0;
            }
            fss_received_sections_prev_ts[section_type::DL_C_SECTION_TYPE_1][cell_index][c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId] = cur_t;
            fss_received_sections[section_type::DL_C_SECTION_TYPE_1][cell_index][c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId] += c_plane_info.numberOfSections;
            max_sections_per_slot[section_type::DL_C_SECTION_TYPE_1][cell_index] = std::max(max_sections_per_slot[section_type::DL_C_SECTION_TYPE_1][cell_index], fss_received_sections[section_type::DL_C_SECTION_TYPE_1][cell_index][c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId]);
        }
    }
    else if(c_plane_info.dir == DIRECTION_UPLINK)
    {
        if(c_plane_info.section_type == ORAN_CMSG_SECTION_TYPE_1)
        {
            if(opt_max_sect_stats)
            {
                const std::lock_guard<aerial_fh::FHMutex> lock(cell_mtx[section_type::UL_C_SECTION_TYPE_1][cell_index]);
                auto cur_t = get_ns();
                if (cur_t - fss_received_sections_prev_ts[section_type::UL_C_SECTION_TYPE_1][cell_index][c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId] > 10000000)
                {
                    fss_received_sections[section_type::UL_C_SECTION_TYPE_1][cell_index][c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId] = 0;
                }
                fss_received_sections_prev_ts[section_type::UL_C_SECTION_TYPE_1][cell_index][c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId] = cur_t;
                fss_received_sections[section_type::UL_C_SECTION_TYPE_1][cell_index][c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId] += c_plane_info.numberOfSections;
                max_sections_per_slot[section_type::UL_C_SECTION_TYPE_1][cell_index] = std::max(max_sections_per_slot[section_type::UL_C_SECTION_TYPE_1][cell_index], fss_received_sections[section_type::UL_C_SECTION_TYPE_1][cell_index][c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId]);
            }
        }
        else if(c_plane_info.section_type == ORAN_CMSG_SECTION_TYPE_3)
        {
            if(opt_max_sect_stats)
            {
                const std::lock_guard<aerial_fh::FHMutex> lock(cell_mtx[section_type::UL_C_SECTION_TYPE_3][cell_index]);
                auto cur_t = get_ns();
                if (cur_t - fss_received_sections_prev_ts[section_type::UL_C_SECTION_TYPE_3][cell_index][c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId] > 10000000)
                {
                    fss_received_sections[section_type::UL_C_SECTION_TYPE_3][cell_index][c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId] = 0;
                }
                fss_received_sections_prev_ts[section_type::UL_C_SECTION_TYPE_3][cell_index][c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId] = cur_t;
                fss_received_sections[section_type::UL_C_SECTION_TYPE_3][cell_index][c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId] += c_plane_info.numberOfSections;
                max_sections_per_slot[section_type::UL_C_SECTION_TYPE_3][cell_index] = std::max(max_sections_per_slot[section_type::UL_C_SECTION_TYPE_3][cell_index], fss_received_sections[section_type::UL_C_SECTION_TYPE_3][cell_index][c_plane_info.fss.frameId][c_plane_info.fss.subframeId * ORAN_MAX_SLOT_ID + c_plane_info.fss.slotId]);
            }
        }
    }
}

void *cplane_core_wrapper(void *arg)
{
    if (!arg) {
        do_throw(sb() << "Error: arg == nullptr with cplane_core_wrapper");
    }
    auto params = reinterpret_cast<struct RU_Emulator::cplane_core_param *>(arg);
    if (!params->rue) {
        do_throw(sb() << "Error: rue == nullptr with cplane_core_wrapper");
    }
    RU_Emulator *rue = static_cast<RU_Emulator*>(params->rue);
    return rue->cplane_core(arg);
}

void* RU_Emulator::cplane_core(void *arg) {
    nvlog_fmtlog_thread_init();
    char threadname[30];
    auto params = reinterpret_cast<struct RU_Emulator::cplane_core_param *>(arg);
    int thread_id = params->thread_id;
    int start_cell_index = params->start_cell_index;
    int num_cells_per_core = params->num_cells_per_core;
    if(params->is_srs)
    {
        sprintf(threadname, "srs%s%u", __FUNCTION__, thread_id);
    }
    else
    {
        sprintf(threadname, "%s%u", __FUNCTION__, thread_id);
    }
    SET_THREAD_NAME(threadname);
    uint8_t cell_index;
    uint8_t round_robin_counter = 0;
    oran_c_plane_info_t rx_c_plane_infos[CPLANE_DEQUEUE_BURST_SIZE];
    int rx_c_plane_info_size = 0;
    re_cons("Thread {} started on {} thread ID {} CPU {} start_cell_index {} num_cells_per_core {}", threadname, params->is_srs ? "SRS" : "UL", thread_id, params->cpu_id, start_cell_index, num_cells_per_core);

    // State tracking for worker tracing
    // Helper function to convert WorkerState to string
    auto workerStateToString = [](WorkerState state) -> const char* {
        switch (state) {
            case WorkerState::RECEIVING:    return "RECEIVING";
            case WorkerState::SPINNING:     return "SPINNING";
            case WorkerState::PROCESSING:   return "PROCESSING";
            case WorkerState::TRANSMITTING: return "TRANSMITTING";
            default:                        return "UNKNOWN";
        }
    };
    WorkerState current_state = WorkerState::SPINNING;
    WorkerState previous_state = WorkerState::SPINNING;

    size_t nb_rx=0;
    uint64_t start_t, end_t;
    struct slot_tx_info slot_tx;
    aerial_fh::MsgReceiveInfo info[CPLANE_DEQUEUE_BURST_SIZE]{}; // Assume non-mmimo has bigger burst size
    if(CPLANE_DEQUEUE_BURST_SIZE_MMIMO > CPLANE_DEQUEUE_BURST_SIZE)
    {
        do_throw(sb() << "CPLANE_DEQUEUE_BURST_SIZE_MMIMO " << CPLANE_DEQUEUE_BURST_SIZE_MMIMO << " > CPLANE_DEQUEUE_BURST_SIZE " << CPLANE_DEQUEUE_BURST_SIZE);
    }

    tx_symbol_timers timers;
    uint64_t sum_tx_pkts_per_cell = 0;
    uint8_t* mbuf_payload;
    void* standalone_mbufs[CPLANE_DEQUEUE_BURST_SIZE];
    int64_t toa;
    uint64_t packet_time = 0;

    // Dynamically allocate TXQs based on actual need
    size_t num_txqs = ORAN_ALL_SYMBOLS * 2;
    if(opt_split_srs_txq)
    {
        num_txqs += num_srs_txqs;
    }
    std::vector<aerial_fh::TxqHandle> txqs(num_txqs);
    NVLOGI_FMT(TAG_CP_WORKER_TRACING, "allocated {} txqs (non-SRS: {}, SRS: {}), opt_split_srs_txq {}",
               num_txqs, ORAN_ALL_SYMBOLS * 2, opt_split_srs_txq ? num_srs_txqs : 0, opt_split_srs_txq);
    int64_t slot_t0;
    int64_t frame_cycle_time_ns = get_frame_cycle_time_ns(max_slot_id, opt_tti_us);
    int64_t first_f0s0s0_time = get_first_f0s0s0_time();

    aerial_fh::TxRequestHandle tx_request;
    bool tx_request_init = false;

    bool pdsch_section = true;

    // Create profiler object once outside the main loop (reused for all iterations)
    perf_metrics::PerfMetricsAccumulator* profiler_ptr = nullptr;
    perf_metrics::PerfMetricsAccumulator tx_slot_profiler{
        "all_tx",
        "handle_sect1",
        "handle_sect3",
        "section_processing",
        "packet_preparation",
        "packet_send",
        "lock_wait",
        "tx_burst_loop",
        "packet_stats_update",
        "counter_updates",
        "tput_counters",
        "verify"
    };

    if (opt_enable_cplane_worker_tracing) {
        profiler_ptr = &tx_slot_profiler;
    }

    try
    {
    while (!check_force_quit()) {
        cell_index = start_cell_index + round_robin_counter;

        NVLOGI_FMT(TAG_CP_WORKER_TRACING, "cell_index {}", cell_index);

        if (num_cells_per_core > 1 && cell_index >= opt_num_cells)
        {
            ++round_robin_counter;
            round_robin_counter = round_robin_counter % (int)num_cells_per_core;
            continue;
        }

        // TX Request can be allocated for each TX thread as well, this used to be in the critical path for prepare_uplane
        // it is moved out so it does not contribute to TX latencies.
        if (tx_request_init == false)
        {
            aerial_fh::alloc_tx_request(peer_list[cell_index], &tx_request);
            tx_request_init = true;
        }

        ////////////////////////////////////////////////
        //// Receive C-msg
        ////////////////////////////////////////////////

        //Note - we do not clear the c_plane_infos here, just reset the sizes
        slot_tx.c_plane_infos_size = 0;
        slot_tx.ul_c_plane_infos_size = 0;

        uint64_t rte_rx_time = 0;
        nb_rx = 0;

        // Call the receive_packets function to handle packet reception
        nb_rx = receive_packets(cell_index, info, nb_rx, rte_rx_time, threadname, current_state, previous_state,
                               first_f0s0s0_time, round_robin_counter, start_cell_index, num_cells_per_core,
                               tx_request, tx_request_init, params->is_srs);

        // State transition to PROCESSING when we have packets
        if (nb_rx > 0 && opt_enable_cplane_worker_tracing) //Note - logging processing for every loop, not just on state transitions
        {
            previous_state = current_state;
            current_state = WorkerState::PROCESSING;
            uint64_t state_transition_time = get_ns();

            // Extract first and last packet frame/subframe/slot info
            // First packet (index 0)
            uint8_t* first_payload = (uint8_t*)info[0].buffer;
            uint8_t first_frame = oran_cmsg_get_frame_id(first_payload);
            uint8_t first_subframe = oran_cmsg_get_subframe_id(first_payload);
            uint8_t first_slot = oran_cmsg_get_slot_id(first_payload);

            // Last packet (index nb_rx-1)
            uint8_t* last_payload = (uint8_t*)info[nb_rx-1].buffer;
            uint8_t last_frame = oran_cmsg_get_frame_id(last_payload);
            uint8_t last_subframe = oran_cmsg_get_subframe_id(last_payload);
            uint8_t last_slot = oran_cmsg_get_slot_id(last_payload);

            NVLOGI_FMT(TAG_CP_WORKER_TRACING, "Worker {} Cell {} state transition: {} -> PROCESSING | Transition time: {} | Delta from time ref: {} | Packets to process: {} | First: F{}S{}S{} | Last: F{}S{}S{}",
                       threadname,
                       cell_index,
                       workerStateToString(previous_state),
                       state_transition_time,
                       state_transition_time - first_f0s0s0_time,
                       nb_rx,
                       first_frame, first_subframe, first_slot,
                       last_frame, last_subframe, last_slot);
        }

        timers.clear();
        start_t = timers.packet_parsing_start_t = get_ns();

        //Go through each c-msg to read all of the unique flowIDs -> need improvement, may read from the wrong TTI
        re_dbg("Cell {} dequeued {} C-msgs:", cell_index, nb_rx);
        for(int index_rx = 0; index_rx < nb_rx; ++index_rx)
        {
#ifdef STANDALONE
            mbuf_payload = (uint8_t*)info[index_rx].buffer;
#else
            mbuf_payload = (uint8_t*)info[index_rx].buffer;
#endif
            oran_c_plane_info_t& c_plane_info = slot_tx.c_plane_infos[slot_tx.c_plane_infos_size];
            c_plane_info.rx_time = info[index_rx].rx_timestamp;
            ++slot_tx.c_plane_infos_size;

            parse_c_plane(c_plane_info, nb_rx, index_rx, rte_rx_time, mbuf_payload, info[index_rx].buffer_length, cell_index);
            track_stats(c_plane_info, cell_index);

            packet_time = info[index_rx].rx_timestamp;
            auto t0_toa = calculate_t0_toa(packet_time, first_f0s0s0_time, frame_cycle_time_ns,
                                        c_plane_info.fss.frameId, c_plane_info.fss.subframeId,
                                        c_plane_info.fss.slotId, c_plane_info.startSym,
                                        max_slot_id, opt_tti_us);
            slot_t0 = t0_toa.slot_t0;
            toa = t0_toa.toa;

            c_plane_info.slot_t0 = slot_t0;

            if (c_plane_info.dir == oran_pkt_dir::DIRECTION_DOWNLINK)
            {
                verify_dl_cplane_content(c_plane_info, cell_index, mbuf_payload, info[index_rx], fss_pdsch_prb_seen);
            }
            else
            {
                verify_ul_cplane_content(c_plane_info, cell_index, mbuf_payload, slot_tx);
            }

            auto& packet_timer = (c_plane_info.dir == oran_pkt_dir::DIRECTION_DOWNLINK) ? oran_packet_slot_timers.timers[DLPacketCounterType::DLC][cell_index][c_plane_info.launch_pattern_slot] : oran_packet_slot_timers.timers[DLPacketCounterType::ULC][cell_index][c_plane_info.launch_pattern_slot];
            process_cplane_timing(cell_index, packet_timer, c_plane_info, packet_time, toa, slot_t0);
        }
        timers.packet_parsing_end_t = fine_timers_get_ns();
        timers.packet_parsing_t += timers.packet_parsing_end_t - timers.packet_parsing_start_t;
        ////////////////////////////////////////////////
        //// Send UMSG
        ////////////////////////////////////////////////

        int num_tx_pkts = 0;
        static std::atomic<uint64_t> sum_tx_pkts;
        if(slot_tx.ul_c_plane_infos_size > 0)
        {
            // Declare variables for frame/subframe/slot info (populated conditionally below)
            uint8_t first_frame{}, first_subframe{}, first_slot{};
            uint8_t last_frame{}, last_subframe{}, last_slot{};

            // State transition to TRANSMITTING when we have packets to transmit
            if (opt_enable_cplane_worker_tracing) //Note - logging processing for every loop, not just on state transitions
            {
                // Extract first and last packet frame/subframe/slot info from processed c_plane_infos
                // First packet (index 0)
                first_frame = slot_tx.c_plane_infos[0].fss.frameId;
                first_subframe = slot_tx.c_plane_infos[0].fss.subframeId;
                first_slot = slot_tx.c_plane_infos[0].fss.slotId;

                // Last packet (index slot_tx.c_plane_infos_size-1)
                last_frame = slot_tx.c_plane_infos[slot_tx.c_plane_infos_size-1].fss.frameId;
                last_subframe = slot_tx.c_plane_infos[slot_tx.c_plane_infos_size-1].fss.subframeId;
                last_slot = slot_tx.c_plane_infos[slot_tx.c_plane_infos_size-1].fss.slotId;

                previous_state = current_state;
                current_state = WorkerState::TRANSMITTING;
                uint64_t state_transition_time = get_ns();

                NVLOGI_FMT(TAG_CP_WORKER_TRACING, "Worker {} Cell {} state transition: {} -> TRANSMITTING | Transition time: {} | Delta from time ref: {} | Packets to transmit: {} | First: F{}S{}S{} | Last: F{}S{}S{}",
                           threadname,
                           cell_index,
                           workerStateToString(previous_state),
                           state_transition_time,
                           state_transition_time - first_f0s0s0_time,
                           slot_tx.c_plane_infos_size,
                           first_frame, first_subframe, first_slot,
                           last_frame, last_subframe, last_slot);
            }

            try {
                aerial_fh::get_uplane_txqs(peer_list[cell_index], txqs.data(), &num_txqs);
                if(num_txqs==0){ // 2
                    do_throw(sb() << "[FH] Failed to get uplane txqs" << "\n");
                }

                // Declare timing variables (populated conditionally below)
                int64_t start_time{}, completion_time{};

                if (opt_enable_cplane_worker_tracing) {
                    start_time = get_ns();
                    profiler_ptr->startSection("all_tx");
                }

                if (opt_enable_precomputed_tx)
                {
                    num_tx_pkts = tx_slot_precomputed(slot_tx, cell_index,
                                                     timers, txqs.data(), &tx_request, profiler_ptr);

                    // PRACH (section type 3) is not precomputed; handle via original path
                    for (int ci = 0; ci < slot_tx.c_plane_infos_size; ++ci) {
                        auto& cpi = slot_tx.c_plane_infos[ci];
                        if (cpi.dir == DIRECTION_DOWNLINK || cpi.section_type != ORAN_CMSG_SECTION_TYPE_3)
                            continue;
                        int prach_tx = 0;
                        if (opt_enable_mmimo) {
                            for (int sym = 0; sym < ORAN_ALL_SYMBOLS; ++sym)
                                prach_tx += handle_sect3_c_plane(cpi, cell_index, timers, txqs.data(), &tx_request, sym);
                        } else {
                            prach_tx = handle_sect3_c_plane_v2(cpi, cell_index, timers, txqs.data(), &tx_request);
                        }
                        num_tx_pkts += prach_tx;
                        auto& tv_object = cpi.tv_object;
                        tv_object->u_plane_tx[cell_index] += prach_tx;
                        tv_object->u_plane_tx_tot[cell_index] += prach_tx;
                        if (opt_enable_mmimo) {
                            // Legacy mMIMO tx_slot() increments once per symbol
                            // iteration; the sym loop skips when startSym > sym.
                            const int n = ORAN_ALL_SYMBOLS - cpi.startSym;
                            tv_object->c_plane_rx[cell_index] += n;
                            tv_object->c_plane_rx_tot[cell_index] += n;
                        } else {
                            ++tv_object->c_plane_rx[cell_index];
                            ++tv_object->c_plane_rx_tot[cell_index];
                        }

                        if (cpi.valid_eaxcId)
                        {
                            int counter_index = cpi.fss.subframeId * ORAN_MAX_SLOT_ID + cpi.fss.slotId;
                            if (opt_enable_mmimo)
                            {
                                const uint16_t prev_count = tv_object->section_rx_counters[cell_index][counter_index].fetch_add(
                                    1, std::memory_order_acq_rel);
                                const uint16_t new_count = prev_count + 1;
                                const uint16_t expected_count = cell_configs[cell_index].num_valid_PRACH_flows;
                                if (new_count == expected_count)
                                {
                                    tv_object->section_rx_counters[cell_index][counter_index].store(0, std::memory_order_release);
                                    ++tv_object->throughput_slot_counters[cell_index];
                                    ++tv_object->total_slot_counters[cell_index];
                                }
                                else if (new_count > expected_count)
                                {
                                    re_warn("Section counter overflow: cell_index={}, counter_index={}, new_count={}, expected_count={}",
                                            cell_index, counter_index, new_count, expected_count);
                                    tv_object->section_rx_counters[cell_index][counter_index].store(0, std::memory_order_release);
                                }
                            }
                            else
                            {
                                ++tv_object->section_rx_counters[cell_index][counter_index];
                                if (tv_object->section_rx_counters[cell_index][counter_index].load() == cell_configs[cell_index].num_valid_PRACH_flows)
                                {
                                    ++tv_object->throughput_slot_counters[cell_index];
                                    ++tv_object->total_slot_counters[cell_index];
                                    tv_object->section_rx_counters[cell_index][counter_index].store(0);
                                }
                            }
                        }
                    }
                }
                else
                {
                    num_tx_pkts = tx_slot(slot_tx, cell_index, timers, txqs.data(), &tx_request, opt_enable_mmimo, profiler_ptr);
                }

                if (opt_enable_cplane_worker_tracing) {
                    profiler_ptr->stopSection("all_tx");
                    completion_time = get_ns();
                    int64_t t0_time = calculate_t0(
                        completion_time,
                        first_frame, first_subframe, first_slot,
                        max_slot_id,
                        opt_tti_us
                    );

                    // Create prefix with frame/subframe/slot range information
                    char profiler_prefix[256];
                    snprintf(profiler_prefix, sizeof(profiler_prefix), 
                             "tx_slot %li %li %li %li Cell %d First: F%dS%dS%d | Last: F%dS%dS%d | UL_Infos: %d",
                             t0_time,
                             completion_time - start_time,
                             start_time - t0_time,
                             completion_time - t0_time,
                             cell_index,
                             first_frame, first_subframe, first_slot,
                             last_frame, last_subframe, last_slot,
                             slot_tx.ul_c_plane_infos_size);

                    // Log profiling results with frame/slot context
                    profiler_ptr->logDurations<TAG_CP_WORKER_TRACING>(profiler_prefix);

                    // Reset profiler for next iteration
                    profiler_ptr->reset();
                }

                sum_tx_pkts += num_tx_pkts;
                sum_tx_pkts_per_cell += num_tx_pkts;
            } catch (std::runtime_error &e) {
                do_throw(sb() << e.what() << "\n Slot counter: "
                    << (int)ul_slot_counters[cell_index].load() << ".");
            }
            end_t = get_ns();
            timers.overall_sum_t += end_t - start_t;
            if (num_tx_pkts > 0)
            {
#if RU_EM_UL_TIMERS_ENABLE
                NVLOGI_FMT(TAG_TX_TIMINGS, "==> Cell {} First Packet F{}S{}S{} Finished handling {} C-Plane Sent {} pkts, Overall: {} ({}) ns Parsing {} ns ({} ns) Counter {} ns ({} ns) umsg alloc {} ns ({} ns) prb match {} ns ({} ns) tx_sym copy time {} ns ({} ns) tx_sym {} ns ({} ns) Prepare {} ns ({} ns) Send time {} ns ({} ns) total pkts so far {} Prepare avg {} ns Send time avg {} ns overall avg per pkt {} ns",
                        cell_index, slot_tx.c_plane_infos[0].fss.frameId, slot_tx.c_plane_infos[0].fss.subframeId, slot_tx.c_plane_infos[0].fss.slotId,
                        slot_tx.c_plane_infos_size, num_tx_pkts, end_t - start_t, (end_t - start_t) / num_tx_pkts,
                        timers.packet_parsing_t, timers.packet_parsing_t / num_tx_pkts,
                        timers.counter_inc_t, timers.counter_inc_t / num_tx_pkts,
                        timers.umsg_alloc_t, timers.umsg_alloc_t / num_tx_pkts,
                        timers.prb_match_t, timers.prb_match_t / num_tx_pkts,
                        timers.tx_symbol_info_copy_t, timers.tx_symbol_info_copy_t / num_tx_pkts,
                        timers.tx_symbol_t, timers.tx_symbol_t / num_tx_pkts,
                        timers.prepare_t, timers.prepare_t / num_tx_pkts,
                        timers.send_t, timers.send_t / num_tx_pkts,
                        sum_tx_pkts_per_cell, timers.prepare_sum_t / sum_tx_pkts_per_cell, timers.send_sum_t / sum_tx_pkts_per_cell, timers.overall_sum_t / sum_tx_pkts_per_cell);
#else

                re_info("==> Cell {} First Packet F{}S{}S{} Finished handling {} C-Plane Sent {} pkts, Overall: {} ({}) ns total pkts so far {} overall avg per pkt {} ns",
                        cell_index, slot_tx.c_plane_infos[0].fss.frameId, slot_tx.c_plane_infos[0].fss.subframeId, slot_tx.c_plane_infos[0].fss.slotId,
                        slot_tx.c_plane_infos_size, num_tx_pkts, end_t - start_t, (end_t - start_t) / num_tx_pkts,
                        sum_tx_pkts_per_cell, timers.overall_sum_t / sum_tx_pkts_per_cell);
#endif
            } else {
                re_info("==> Cell {} First Packet F{}S{}S{} Finished handling {} C-Plane Sent {} pkts",
                        cell_index, slot_tx.c_plane_infos[0].fss.frameId, slot_tx.c_plane_infos[0].fss.subframeId, slot_tx.c_plane_infos[0].fss.slotId,
                        slot_tx.c_plane_infos_size, num_tx_pkts);
            }
        }

        aerial_fh::free_rx_messages(&info[0],nb_rx);

        if(opt_forever == RE_DISABLED)
        {
            if(ul_slot_counters[cell_index].load() >= opt_num_slots_ul)
            {
                break;
            }
        }

        if (num_cells_per_core > 1)
        {
            ++round_robin_counter;
            round_robin_counter = round_robin_counter % (int)num_cells_per_core;
        }
    }
    }
    catch(std::exception& e)
    {
        re_cons("Exception in cplane_cores: {}", e.what());
        {
            std::lock_guard<aerial_fh::FHMutex> lock(cplane_pcap_mutex);
            generate_pcap_file(cell_index, &info[0], nb_rx, threadname);
        }
    }
    catch(...)
    {
        re_cons("Unknown exception in cplane_cores");
        {
            std::lock_guard<aerial_fh::FHMutex> lock(cplane_pcap_mutex);
            generate_pcap_file(cell_index, &info[0], nb_rx, threadname);
        }
    }
    if(pdsch_object.total_slot_counters[cell_index].load() >= opt_num_slots_dl)
    {
        set_force_quit();
    }

    // All cells use the same tx_request pool, use peer 0 for placeholder, will fix with the UL core assignment
    if (tx_request_init == true)
    {
        aerial_fh::free_preallocated_mbufs(peer_list[0], &tx_request);
    }
    re_info("Cell {} Exiting after {} UL slots", cell_index, ul_slot_counters[cell_index].load());
    // return EXIT_SUCCESS;
    re_cons("Thread {} exiting", threadname);

    usleep(2000000);
    return NULL;
}

void *uplane_tx_only_core_wrapper(void *arg)
{
    if (!arg) {
        do_throw(sb() << "Error: arg == nullptr with uplane_tx_only_core_wrapper");
    }
    auto params = reinterpret_cast<struct RU_Emulator::cplane_core_param *>(arg);
    if (!params->rue) {
        do_throw(sb() << "Error: rue == nullptr with uplane_tx_only_core_wrapper");
    }
    RU_Emulator *rue = static_cast<RU_Emulator*>(params->rue);
    return rue->uplane_tx_only_core(arg);
}

void *RU_Emulator::uplane_tx_only_core(void *arg)
{
    // opt_ul_only enabled
    // Only handles single cell
    // Only handles PUSCH
    // Starts at next F0S0S0

    nvlog_fmtlog_thread_init();
    char threadname[30];
    auto params = reinterpret_cast<struct RU_Emulator::cplane_core_param *>(arg);
    int thread_id = params->thread_id;
    int start_cell_index = params->start_cell_index;
    int num_cells_per_core = params->num_cells_per_core;
    if(params->is_srs)
    {
        sprintf(threadname, "srs%s%u", __FUNCTION__, thread_id);
    }
    else
    {
        sprintf(threadname, "%s%u", __FUNCTION__, thread_id);
    }
    SET_THREAD_NAME(threadname);
    uint8_t cell_index = start_cell_index;
    uint8_t round_robin_counter = 0;
    oran_c_plane_info_t rx_c_plane_infos[CPLANE_DEQUEUE_BURST_SIZE];
    int rx_c_plane_info_size = 0;
    re_cons("Thread {} started on {} thread ID {} CPU {} start_cell_index {} num_cells_per_core {}", threadname, params->is_srs ? "SRS" : "UL", thread_id, params->cpu_id, start_cell_index, num_cells_per_core);

    OranSlotIterator oran_slot_iterator(OranSlotNumber{0, 0, 0, 0});

    int64_t slot_t0;
    int64_t frame_cycle_time_ns = get_frame_cycle_time_ns(max_slot_id, opt_tti_us);
    int64_t first_f0s0s0_time = get_first_f0s0s0_time();
    int64_t next_slot_time = first_f0s0s0_time + frame_cycle_time_ns;
    uint64_t slot_counter = 0;
    uint64_t ul_enqueue_advance_ns = 500000;
    struct slot_tx_info slot_tx;
    struct fssId fss{};
    tx_symbol_timers timers{};
    aerial_fh::TxqHandle txqs[RU_TXQ_COUNT];
    size_t num_txqs = RU_TXQ_COUNT;
    aerial_fh::TxRequestHandle tx_request;
    bool tx_request_init = false;
    if (tx_request_init == false)
    {
        aerial_fh::alloc_tx_request(peer_list[cell_index], &tx_request);
        tx_request_init = true;
    }

    while(!check_force_quit())
    {
        auto oran_slot_number = oran_slot_iterator.get_next();
        fss.frameId = oran_slot_number.frame_id;
        fss.subframeId = oran_slot_number.subframe_id;
        fss.slotId = oran_slot_number.slot_id;
        slot_tx.ul_c_plane_infos_size = 0;
        slot_tx.c_plane_infos_size = 0;

        while(!check_force_quit() && get_ns() < next_slot_time - ul_enqueue_advance_ns)
        {
            usleep(10);
        }
        int slot_idx = fss_to_launch_pattern_slot(fss, launch_pattern_slot_size);
        if(pusch_object.launch_pattern[slot_idx].find(cell_index) != pusch_object.launch_pattern[slot_idx].end())
        {
            prepare_cplane_info(pusch_object, cell_configs, slot_tx, fss, cell_index, slot_idx,
                               next_slot_time, first_f0s0s0_time, frame_cycle_time_ns,
                               max_slot_id, opt_tti_us, slot_t0);
        }

        if(slot_tx.ul_c_plane_infos_size > 0)
        {
            try {
                aerial_fh::get_uplane_txqs(peer_list[cell_index], txqs, &num_txqs);
                auto num_tx_pkts = tx_slot(slot_tx, cell_index, timers, txqs, &tx_request, 0, nullptr);
            } catch (std::runtime_error &e) {
                do_throw(sb() << e.what() << "\n Slot counter: "
                    << (int)ul_slot_counters[cell_index].load() << ".");
            }
        }
        next_slot_time += 500 * NS_X_US;
        ++slot_counter;
    }
    re_cons("Thread {} exiting after {} slots", threadname, slot_counter);

    return NULL;
}
