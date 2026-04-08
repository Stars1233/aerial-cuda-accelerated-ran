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

#include "ru_emulator.hpp"

// ============================================================
// All functions in this translation unit are __attribute__((cold))
// so the linker places them in .text.unlikely, keeping the hot
// C-plane / U-plane processing functions compact in icache.
// ============================================================

/**
 * @brief Validates DL C-plane section IDs for default coupling (sectionId range and duplicate consistency).
 * @param[in,out] c_plane_info C-plane message info to validate; contains sections and eAxC index used for tracker lookup.
 * @param[in] cell_index Cell index being validated (selects tracker and counters).
 * @return void
 */
__attribute__((cold))
void RU_Emulator::sid_dl_validate(oran_c_plane_info_t& c_plane_info, int cell_index)
{
    if (!c_plane_info.valid_eaxcId)
    {
        error_dl_section_counters[cell_index] += c_plane_info.numberOfSections;
        total_dl_section_counters[cell_index] += c_plane_info.numberOfSections;
        re_warn("Invalid eAxC ID {} (cell {})", c_plane_info.eaxcId, cell_index);
        return;
    }

    const std::lock_guard<aerial_fh::FHMutex> lock(cell_mtx[section_type::DL_C_SECTION_TYPE_1][cell_index]);
    auto& dl_tracker = section_id_trackers->dl[cell_index][c_plane_info.eaxcId_index];
    validate_section_id_default_coupling(c_plane_info, cell_index, dl_tracker);
}

/**
 * @brief Validates UL C-plane section IDs by dispatching to the appropriate channel-specific tracker.
 * @param[in,out] c_plane_info C-plane message info containing sections, eAxC, and channel type to validate.
 * @param[in]     cell_index   Cell index selecting the per-cell tracker and error counters.
 * @return true if all sections pass validation (or there are none), false if the eAxC ID is invalid
 *         or a section fails range/consistency checks.
 */
__attribute__((cold))
bool RU_Emulator::sid_real_ul(oran_c_plane_info_t& c_plane_info, int cell_index)
{
    if (!c_plane_info.valid_eaxcId)
    {
        error_ul_section_counters[cell_index] += c_plane_info.numberOfSections;
        total_ul_section_counters[cell_index] += c_plane_info.numberOfSections;
        re_warn("Invalid eAxC ID {} in UL C-plane (cell {})", c_plane_info.eaxcId, cell_index);
        c_plane_info.verification_section_end = get_ns();
        return false;
    }

    if (c_plane_info.numberOfSections == 0)
    {
        c_plane_info.verification_section_end = get_ns();
        return true;
    }

    // Channel type (PRACH / SRS / PUSCH-PUCCH) and eaxcId_index come from find_channel_type + find_eAxC_index upstream.
    uint16_t first_sid = c_plane_info.section_infos[0].section_id;
    return sid_real_ul_one_channel(c_plane_info, cell_index, first_sid);
}

/**
 * @brief Validates UL C-plane section IDs for a single channel type against its
 *        per-eAxC slot tracker, advancing the slot and checking range consistency.
 *
 * Acquires the per-cell mutex (when mMIMO is enabled), advances the tracker on
 * a new slot boundary, verifies that every sectionId in @p c_plane_info falls
 * within the tracker's [BASE, BASE+N) range, and delegates per-section
 * duplicate/consistency checks to validate_section_id_sections_only_impl().
 *
 * @tparam N     Number of sectionIds covered by the tracker (e.g.
 *               UL_PRACH_SECTION_ID_SIZE, UL_SRS_SECTION_ID_SIZE,
 *               UL_PUSCH_PUCCH_SECTION_ID_SIZE).
 * @tparam BASE  Lowest valid sectionId in the range (0 for PUSCH/PUCCH,
 *               UL_PRACH_SECTION_ID_BASE for PRACH, UL_SRS_SECTION_ID_BASE
 *               for SRS).
 *
 * @param[in,out] c_plane_info  Parsed C-plane message; sections are validated
 *                              and @c verification_section_end is stamped on
 *                              early-exit paths.
 * @param[in]     cell_index    Cell index selecting per-cell counters and mutex.
 * @param[in]     first_sid     SectionId of the first section, used in mixed-channel
 *                              warning messages when a subsequent section falls
 *                              outside the expected range.
 * @param[in,out] vec           Per-eAxC tracker vector for the channel type;
 *                              indexed by @c c_plane_info.eaxcId_index.
 * @param[in]     mtx_type      Mutex selector (section_type enum) for
 *                              @c cell_mtx[mtx_type][cell_index].
 * @param[in]     channel_name  Human-readable channel label for log messages
 *                              (e.g. "PRACH", "SRS", "PUSCH/PUCCH").
 *
 * @return true if all sections pass validation, false if @c eaxcId_index is out
 *         of range, the slot is stale, or any sectionId falls outside [BASE, BASE+N).
 *
 * @pre  @p c_plane_info.eaxcId_index must be a valid index into @p vec; otherwise
 *       the function logs a warning and returns false.
 * @pre  @p vec is sized to the number of configured eAxC flows for this channel
 *       type and cell (set up by sectionid_validation_init()).
 */
template <size_t N, uint16_t BASE>
bool RU_Emulator::sid_real_ul_one_channel_impl(oran_c_plane_info_t& c_plane_info, int cell_index,
                                               uint16_t first_sid, std::vector<SlotSectionIdTrackerRange<N, BASE>>& vec,
                                               int mtx_type, const char* channel_name)
{
    if (static_cast<size_t>(c_plane_info.eaxcId_index) >= vec.size())
    {
        error_ul_section_counters[cell_index] += c_plane_info.numberOfSections;
        total_ul_section_counters[cell_index] += c_plane_info.numberOfSections;
        re_warn("UL C-plane {} eaxcId_index out of range: cell {} eaxcId {} eaxcId_index {} vector size {}",
                channel_name, cell_index, c_plane_info.eaxcId, c_plane_info.eaxcId_index, vec.size());
        c_plane_info.verification_section_end = get_ns();
        return false;
    }
    std::unique_lock<aerial_fh::FHMutex> lock(cell_mtx[mtx_type][cell_index], std::defer_lock);
    if (opt_enable_mmimo)
        lock.lock();
    auto& tracker = vec[c_plane_info.eaxcId_index];
    if (!tracker.is_same_slot(c_plane_info.fss))
    {
        if (tracker.has_prev && !tracker.is_forward_slot(c_plane_info.fss))
        {
            c_plane_info.verification_section_end = get_ns();
            return false;
        }
        tracker.advance_slot(c_plane_info.fss);
    }
    for (int i = 0; i < c_plane_info.numberOfSections; ++i)
    {
        if (!tracker.in_range(c_plane_info.section_infos[i].section_id))
        {
            re_warn("UL C-plane mixed channel types: cell {} section {} sid {} does not match first sid {} range",
                    cell_index, i, c_plane_info.section_infos[i].section_id, first_sid);
            c_plane_info.verification_section_end = get_ns();
            return false;
        }
    }
    validate_section_id_sections_only_impl(c_plane_info, cell_index, tracker);
    return true;
}

/**
 * @brief Dispatches UL C-plane section-ID validation to the channel-specific
 *        tracker (PRACH, SRS, or PUSCH/PUCCH) based on section and channel type.
 *
 * @param[in,out] c_plane_info  Parsed C-plane message whose section/channel type
 *                              determines the dispatch target.
 * @param[in]     cell_index    Cell index selecting the per-cell tracker vector.
 * @param[in]     first_sid     SectionId of the first section, used to choose
 *                              the matching tracker range and for diagnostic logs.
 *
 * @return true if all sections pass validation, false on any range or
 *         consistency failure.
 */
bool RU_Emulator::sid_real_ul_one_channel(oran_c_plane_info_t& c_plane_info, int cell_index, uint16_t first_sid)
{
    if (c_plane_info.section_type == ORAN_CMSG_SECTION_TYPE_3)
        return sid_real_ul_one_channel_impl(c_plane_info, cell_index, first_sid,
                                           section_id_trackers->ul_prach[cell_index],
                                           section_type::UL_C_SECTION_TYPE_3, "PRACH");
    if (c_plane_info.channel_type == ul_channel::SRS)
        return sid_real_ul_one_channel_impl(c_plane_info, cell_index, first_sid,
                                           section_id_trackers->ul_srs[cell_index],
                                           section_type::UL_C_SECTION_TYPE_1, "SRS");
    return sid_real_ul_one_channel_impl(c_plane_info, cell_index, first_sid,
                                        section_id_trackers->ul_pusch_pucch[cell_index],
                                        section_type::UL_C_SECTION_TYPE_1, "PUSCH/PUCCH");
}

template bool RU_Emulator::sid_real_ul_one_channel_impl<UL_PRACH_SECTION_ID_SIZE, UL_PRACH_SECTION_ID_BASE>(
    oran_c_plane_info_t&, int, uint16_t, std::vector<SlotSectionIdTrackerRange<UL_PRACH_SECTION_ID_SIZE, UL_PRACH_SECTION_ID_BASE>>&,
    int, const char*);
template bool RU_Emulator::sid_real_ul_one_channel_impl<UL_SRS_SECTION_ID_SIZE, UL_SRS_SECTION_ID_BASE>(
    oran_c_plane_info_t&, int, uint16_t, std::vector<SlotSectionIdTrackerRange<UL_SRS_SECTION_ID_SIZE, UL_SRS_SECTION_ID_BASE>>&,
    int, const char*);
template bool RU_Emulator::sid_real_ul_one_channel_impl<UL_PUSCH_PUCCH_SECTION_ID_SIZE, 0>(
    oran_c_plane_info_t&, int, uint16_t, std::vector<SlotSectionIdTrackerRange<UL_PUSCH_PUCCH_SECTION_ID_SIZE, 0>>&,
    int, const char*);

/**
 * @brief Records a received DL U-plane sectionId in the per-eAxC slot tracker
 *        so that orphan detection can identify C-plane-only sectionIds at slot advance.
 *
 * @param[in] cell_index   Cell index; silently ignored if out of range.
 * @param[in] header_info  Parsed U-plane packet header containing flow_index,
 *                         slot id (fss), and sectionId to record.
 * @return void
 */
__attribute__((cold))
void RU_Emulator::validate_uplane_section_id_match(uint8_t cell_index, const struct oran_packet_header_info &header_info)
{
    if (cell_index >= opt_num_cells)
        return;

    if (header_info.flow_index >= section_id_trackers->dl[cell_index].size())
        return;

    const std::lock_guard<aerial_fh::FHMutex> lock(cell_mtx[section_type::DL_C_SECTION_TYPE_1][cell_index]);
    auto& tracker = section_id_trackers->dl[cell_index][header_info.flow_index];
    if (!tracker.is_same_slot(header_info.fss))
        return;

    tracker.record_uplane_sid(header_info.sectionId);
}

// ============================================================
// Template implementations (cold) -- moved from cplane_rx_cores.cpp
// ============================================================

/**
 * @brief Cold-path wrapper that delegates DL default-coupling sectionId
 *        validation to validate_section_id_default_coupling_impl().
 *
 * @param[in,out] c_plane_info  Parsed C-plane message to validate.
 * @param[in]     cell_index    Cell index selecting the per-cell counters and mutex.
 * @param[in,out] tracker       Per-eAxC DL slot tracker (SlotSectionIdTracker typedef).
 *
 * @see validate_section_id_default_coupling_impl
 */
__attribute__((cold))
void RU_Emulator::validate_section_id_default_coupling(oran_c_plane_info_t& c_plane_info, int cell_index,
                                                       SlotSectionIdTracker& tracker)
{
    validate_section_id_default_coupling_impl(c_plane_info, cell_index, tracker);
}

/**
 * @brief Validates C-plane sectionId default-coupling semantics for one eAxC within a slot.
 *
 * On a new slot, advances the tracker and — for DL only — reports any U-plane
 * sectionIds that were received but never announced by C-plane (orphan detection).
 * After the slot boundary is reconciled, delegates per-section validation
 * (range check, duplicate consistency) to validate_section_id_sections_only_impl().
 *
 * Marked `__attribute__((cold))` so the linker places the body in `.text.unlikely`,
 * keeping the hot C-plane / U-plane fast paths compact in icache.
 *
 * @tparam N     Number of sectionIds covered by the tracker (e.g. DL_SECTION_ID_SIZE,
 *               UL_PUSCH_PUCCH_SECTION_ID_SIZE).
 * @tparam BASE  Lowest sectionId in the tracker's valid range (0 for DL/PUSCH+PUCCH,
 *               UL_PRACH_SECTION_ID_BASE for PRACH, UL_SRS_SECTION_ID_BASE for SRS).
 *
 * @param[in,out] c_plane_info  Parsed C-plane message; read for direction, slot id (fss),
 *                              eAxC, and section list.  Section entries may have their
 *                              @c error_status set to -1 on validation failure.
 * @param[in]     cell_index    Cell index used to select the per-cell tracker vector and
 *                              the per-cell error counter.
 * @param[in,out] tracker       Per-eAxC slot tracker that records announced/received
 *                              sectionIds and is advanced on slot boundaries.
 *
 * @pre  @p c_plane_info.valid_eaxcId is true and @p c_plane_info.eaxcId_index is a
 *       valid index into the tracker vector for the corresponding channel type.
 * @pre  The caller holds any required mutex (e.g. cell_mtx for DL).
 *
 * @post For DL direction: @c section_id_trackers->error_dl_uplane[cell_index] is
 *       incremented once per orphan U-plane sectionId detected during slot advance,
 *       and a warning is logged for each.
 * @post Each section in @p c_plane_info is validated for range and duplicate
 *       consistency via validate_section_id_sections_only_impl().
 *
 * @note If the incoming slot is not forward-monotonic and the tracker already has
 *       history, the function returns early without validation (stale packet).
 */
template <size_t N, uint16_t BASE>
__attribute__((cold))
void RU_Emulator::validate_section_id_default_coupling_impl(oran_c_plane_info_t& c_plane_info, int cell_index,
                                                           SlotSectionIdTrackerRange<N, BASE>& tracker)
{
    if (!tracker.is_same_slot(c_plane_info.fss))
    {
        if (tracker.has_prev && !tracker.is_forward_slot(c_plane_info.fss))
            return;

        if (c_plane_info.dir == oran_pkt_dir::DIRECTION_DOWNLINK)
        {
            tracker.advance_slot(c_plane_info.fss, [&, eaxc_copy = c_plane_info.eaxcId, cell_index](uint16_t sid, const struct fssId& fss) {
                section_id_trackers->error_dl_uplane[cell_index]++;
                re_warn("DL U-plane sectionId not announced in C-plane: cell {} F{}S{}S{} eAxC {} sectionId {}",
                        cell_index, fss.frameId, fss.subframeId, fss.slotId, eaxc_copy, sid);
            });
        }
        else
        {
            tracker.advance_slot(c_plane_info.fss);
        }
    }

    validate_section_id_sections_only_impl(c_plane_info, cell_index, tracker);
}

/**
 * @brief Validates every C-plane sectionId in @p c_plane_info against the
 *        tracker's valid range [BASE, BASE+N) and enforces duplicate-citation
 *        consistency within the current slot.
 *
 * For each section in @p c_plane_info the function:
 *   -# Checks that the sectionId falls within the tracker's range via
 *      @c tracker.in_range(); out-of-range IDs are flagged and logged.
 *   -# Converts the sectionId to a tracker-internal index with
 *      @c tracker.to_index() and marks it in @c tracker.cplane_sids_announced.
 *   -# On the first citation in this slot (generation mismatch), records the
 *      section parameters (rb, startPrbc, numPrbc, numSymbol, udCompHdr) in
 *      @c tracker.entries and stamps the current generation.
 *   -# On a duplicate citation, verifies that the parameters match the first
 *      occurrence; mismatches are flagged and logged.
 *
 * Marked @c __attribute__((cold)) — placed in @c .text.unlikely so it does not
 * pollute the icache of the hot C-plane / U-plane receive paths.
 *
 * @tparam N     Size of the sectionId range tracked (e.g. DL_SECTION_ID_SIZE,
 *               UL_PUSCH_PUCCH_SECTION_ID_SIZE, UL_PRACH_SECTION_ID_SIZE,
 *               UL_SRS_SECTION_ID_SIZE).
 * @tparam BASE  First valid sectionId in the range (0 for DL / PUSCH+PUCCH,
 *               UL_PRACH_SECTION_ID_BASE for PRACH, UL_SRS_SECTION_ID_BASE
 *               for SRS).
 *
 * @param[in,out] c_plane_info  Parsed C-plane message whose @c section_infos
 *                              are iterated.  On error, the affected section's
 *                              @c error_status is set to -1.
 * @param[in]     cell_index    Cell index; used only for diagnostic log messages.
 * @param[in,out] tracker       Per-eAxC slot tracker updated with announced
 *                              sectionIds and section parameters.
 *
 * @return void
 *
 * @note Side effects:
 *       - Sets @c sec.error_status = -1 for out-of-range or inconsistent sections.
 *       - Sets bits in @c tracker.cplane_sids_announced for every in-range sectionId.
 *       - Writes rb/startPrbc/numPrbc/numSymbol/udCompHdr and generation into
 *         @c tracker.entries on first citation per slot.
 *       - Emits @c re_warn log lines for every validation failure.
 */
template <size_t N, uint16_t BASE>
__attribute__((cold))
void RU_Emulator::validate_section_id_sections_only_impl(oran_c_plane_info_t& c_plane_info, int cell_index,
                                                         SlotSectionIdTrackerRange<N, BASE>& tracker)
{
    const char* dir_str = (c_plane_info.dir == oran_pkt_dir::DIRECTION_DOWNLINK) ? "DL" : "UL";
    const uint32_t gen = tracker.current_generation;

    for (int sec_idx = 0; sec_idx < c_plane_info.numberOfSections; ++sec_idx)
    {
        oran_c_plane_section_info_t& sec = c_plane_info.section_infos[sec_idx];
        uint16_t sid = sec.section_id;

        if (!tracker.in_range(sid))
        {
            sec.error_status = -1;
            re_warn("{} C-plane sectionId out of range: cell {} F{}S{}S{} eAxC {} section {} sectionId {} (valid {}-{})",
                    dir_str, cell_index, c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId,
                    c_plane_info.eaxcId, sec_idx, sid, static_cast<unsigned>(BASE), static_cast<unsigned>(tracker.MAX_SECTION_ID));
            continue;
        }

        const size_t idx = tracker.to_index(sid);
        tracker.cplane_sids_announced.set(idx);
        auto& entry = tracker.entries[idx];

        if (entry.generation != gen)
        {
            entry.generation = gen;
            entry.rb         = sec.rb;
            entry.startPrbc  = sec.startPrbc;
            entry.numPrbc    = sec.numPrbc;
            entry.numSymbol  = sec.numSymbol;
            entry.udCompHdr  = c_plane_info.udCompHdr;
            continue;
        }

        if (sec.rb != entry.rb || sec.startPrbc != entry.startPrbc ||
            sec.numPrbc != entry.numPrbc || sec.numSymbol != entry.numSymbol ||
            c_plane_info.udCompHdr != entry.udCompHdr)
        {
            sec.error_status = -1;
            re_warn("{} C-plane duplicate sectionId with inconsistent data: cell {} F{}S{}S{} eAxC {} sectionId {} section {} "
                    "(rb/startPrbc/numPrbc/numSymbol/udCompHdr must match first occurrence in this slot)",
                    dir_str, cell_index, c_plane_info.fss.frameId, c_plane_info.fss.subframeId, c_plane_info.fss.slotId,
                    c_plane_info.eaxcId, sid, sec_idx);
        }
    }
}

// Explicit template instantiations
template void RU_Emulator::validate_section_id_default_coupling_impl(oran_c_plane_info_t&, int, SlotSectionIdTrackerRange<DL_SECTION_ID_SIZE, 0>&);
template void RU_Emulator::validate_section_id_sections_only_impl(oran_c_plane_info_t&, int, SlotSectionIdTrackerRange<UL_PUSCH_PUCCH_SECTION_ID_SIZE, 0>&);
template void RU_Emulator::validate_section_id_sections_only_impl(oran_c_plane_info_t&, int, SlotSectionIdTrackerRange<UL_PRACH_SECTION_ID_SIZE, UL_PRACH_SECTION_ID_BASE>&);
template void RU_Emulator::validate_section_id_sections_only_impl(oran_c_plane_info_t&, int, SlotSectionIdTrackerRange<UL_SRS_SECTION_ID_SIZE, UL_SRS_SECTION_ID_BASE>&);

// ============================================================
// Initialization -- called from apply_configs()
// ============================================================
void RU_Emulator::sectionid_validation_init()
{
    if (opt_sectionid_validation == RE_ENABLED)
    {
        // opt_num_cells <= MAX_CELLS_PER_SLOT is enforced by verify_configs()
        // which runs before apply_configs() in verify_and_apply_configs().
        for (int cell = 0; cell < opt_num_cells; ++cell)
        {
            section_id_trackers->dl[cell].resize(cell_configs[cell].num_dl_flows);
            section_id_trackers->ul_pusch_pucch[cell].resize(cell_configs[cell].num_ul_flows);
            section_id_trackers->ul_prach[cell].resize(cell_configs[cell].num_valid_PRACH_flows);
            section_id_trackers->ul_srs[cell].resize(cell_configs[cell].num_valid_SRS_flows);
        }
    }
}
