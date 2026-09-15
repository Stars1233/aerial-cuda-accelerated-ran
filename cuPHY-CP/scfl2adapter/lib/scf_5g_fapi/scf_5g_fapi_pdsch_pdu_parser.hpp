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

#if !defined(SCF_5G_FAPI_PDSCH_PDU_PARSER_HPP_INCLUDED_)
#define SCF_5G_FAPI_PDSCH_PDU_PARSER_HPP_INCLUDED_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <span>
#include <utility>

#include <gsl-lite/gsl-lite.hpp>

#include "scf_5g_fapi_dl_stats.hpp"
#include "scf_5g_fapi_slot_concepts.hpp"
#include "scf_5g_fapi_pdsch_ue_grouping.hpp"
#include "scf_5g_fapi_pdsch_fh_fill.hpp"
#include "scf_5g_fapi_errors.hpp"
#include "scf_5g_fapi_lbrm.hpp"
#include "scf_5g_fapi_tags.hpp"
#include "scf_5g_fapi.h"
#include "slot_command/slot_command.hpp"
#include "nvlog.h"   // also provides AERIAL_L2ADAPTER_EVENT via aerial_event_code.h
#include "nvlog_fmt.hpp"
#include "aerial_event_code.h"

#include <tl/expected.hpp>
#include <wise_enum/wise_enum.h>

namespace scf_5g_fapi
{


// Forward declaration: defined in scf_5g_slot_commands_common.cpp (production)
// and stubbed in tests/test_stubs.cpp (unit tests).
bool check_bf_pc_params(int numPrg, int numDigBFI, bool mmimo_enabled);

/**
 * Success payload of @c PdschPduParser::setup_ue_group(): the UE-group
 * context that downstream @c setup_beamforming() and @c setup_fh_params()
 * need to populate @c fh_params.
 *
 * @c is_new   True when this PDU created the UE group (vs. joining an existing one).
 * @c grp_idx  UE group index within @c params.ue_grp_info.
 * @c bfw_idx  Per-cell BFW group index (@c ue_grp_idx_bfw_id_map entry).
 */
struct UeGroupContext
{
    bool     is_new{};   //!< Whether this PDU created its UE group.
    uint16_t grp_idx{};  //!< UE group index within @c params.ue_grp_info.
    uint32_t bfw_idx{};  //!< @c ue_grp_idx_bfw_id_map entry for this group.
};

/**
 * Result of @c PdschPduParser::setup_ue_group(): success carries a
 * @c UeGroupContext; failure carries a @c UeGroupErrorCode identifying
 * which early-return guard fired. Caller checks via @c has_value() and
 * accesses the context via @c operator-> / @c operator*. Error enumerator
 * names are available via @c wise_enum::to_string().
 */
using UeGroupResult = tl::expected<UeGroupContext, UeGroupErrorCode>;

/**
 * Parses a single PDSCH PDU from a DL_TTI.request message.
 *
 * Calls update_cell_command for PDSCH, forwarding group_command,
 * cell_sub_command, pm_map, pm_enabled, bf_enabled, bfw_coeff_mem_info,
 * mmimo_enabled, and cell_view (num_dl_prb, slot_detail) from the module view.
 *
 * Before dispatching individual PDSCH PDUs, call setup_cell() once per
 * DL_TTI.request to initialize cuphyPdschCellDynPrm_t, advance the
 * global CSI-RS parameter offset in pdsch_params, and reset the per-slot
 * codeword byte offset accumulator.
 *
 * pdsch_params is pre-allocated by PhyModule; setup_cell() only populates
 * cuphyPdschCellDynPrm_t and advances the CSI-RS offset counter.
 *
 * @tparam V  Type satisfying DlModuleView.
 * @tparam G  UE-grouping policy; defaults to DefaultUeGroupingPolicy which
 *             replicates update_cell_command() behaviour (RA-Type 0 / 1 at runtime).
 */
template<DlModuleView V, UeGroupingPolicy G = DefaultUeGroupingPolicy>
class PdschPduParser final
{
public:
    using pdu_t = scf_fapi_pdsch_pdu_t;
    static constexpr scf_fapi_dl_tti_pdu_type_t pdu_type = DL_TTI_PDU_TYPE_PDSCH;

    explicit PdschPduParser(V& v) noexcept : view_{&v} {}

    /**
     * Per-message setup: initialise the cell dynamic PDSCH params, advance
     * the global CSI-RS offset in pdsch_params, and reset the codeword byte
     * offset accumulator.
     *
     * Must be called once per DL_TTI.request before any parse() calls for
     * that message.
     *
     * @param[in] req The DL TTI request for the current message.
     * @param[in] msg_cell_id Logical cell id from the DL_TTI message header.
     */
    void setup_cell(const scf_fapi_dl_tti_req_t& req, uint32_t msg_cell_id) noexcept;

    void set_stats_batch(DlPdschStatsBatch* batch) noexcept
    {
        stats_batch_ = batch;
        if (batch == nullptr) {
            stats_message_active_ = false;
            stats_cell_id_ = 0U;
        }
    }

    void begin_stats_message(std::uint32_t cell_id) noexcept
    {
        stats_cell_id_ = cell_id;
        stats_message_active_ = (stats_batch_ != nullptr);
    }

    void end_stats_message() noexcept
    {
        stats_message_active_ = false;
    }

    /**
     * Process one PDSCH PDU for the given sfn/slot.
     *
     * @param[in] sfn   System Frame Number.
     * @param[in] slot  Slot number.
     * @param[in] pdu   PDSCH PDU to process.
     * @return true on success; false if processing could not be completed.
     *         Return value must be checked.
     * @note Marked noexcept: the body wraps view_->slot_command() and all
     *       subsequent setup_* calls in a try-catch.  The throwing path is
     *       view_->slot_command() → slot_detail() → getMPlaneConfig(), which
     *       throws std::runtime_error on misconfiguration.  Any std::exception
     *       (or unknown exception) is logged and mapped to a false return so
     *       that the noexcept contract of the dispatch infrastructure is upheld.
     */
    [[nodiscard]] bool parse(uint16_t sfn, uint16_t slot, const pdu_t& pdu) noexcept;

private:
    gsl_lite::not_null<V*> view_; //!< Non-owning; must outlive this parser.
    uint32_t cw_byte_offset_{}; //!< Running TB byte offset; reset by setup_cell() each slot.
    bool fh_enabled_for_msg_{}; //!< Runtime FH gate cached in setup_cell() — drives NullFhCallable dispatch.
    uint32_t msg_cell_id_{}; //!< Cell index from the current DL_TTI message descriptor.
    bool fh_committed_for_pdu_{}; //!< True after setup_fh_params() committed an entry; consulted by rollback_ue().
    bool cell_setup_valid_{}; //!< True only when setup_cell() completed successfully for the current message.
    DlPdschStatsBatch* stats_batch_{}; //!< Non-owning batch set by DLSlotProcessor for one process() call.
    std::uint32_t stats_cell_id_{}; //!< Current message cell id for DL throughput/slot counters.
    bool stats_message_active_{}; //!< True only while DLSlotProcessor is dispatching a PDSCH message.

    // ------------------------------------------------------------------
    // Compile-time constants
    // ------------------------------------------------------------------

    /// Bit position in dmrs_ports that signals > 16-layer transmission (TS 38.212).
    static constexpr uint8_t k_nlabove16_bit_loc = 15;

    /// Max SCF FAPI powerControlOffset value (0–23 → −8..+15 dB, SCF 222.10.02 §3.4.3.1).
    static constexpr uint8_t k_max_power_ctrl_offset = 23;

    /// Max SCF FAPI powerControlOffsetSS value (0–3 → −3/0/+3/+6 dB, SCF 222.10.02 §3.4.3.1).
    static constexpr uint8_t k_max_power_ctrl_offset_ss = 3;

    // ------------------------------------------------------------------
    // PDU tail helpers
    // ------------------------------------------------------------------

    /**
     * Aggregates the three pointer/reference values derived from the
     * variable-length PDU layout that multiple parse() sub-steps share.
     * Computed once in parse() by derive_pdu_tail() and passed down to
     * eliminate redundant reinterpret_cast chains per PDU.
     */
    struct PduTail {
        const scf_fapi_pdsch_pdu_end_t&            end;       //!< Fixed PDU tail after codewords.
        const uint8_t*                              pm_bf_raw; //!< Byte pointer past optional PTRS (for stride arithmetic).
        const scf_fapi_tx_precoding_beamforming_t& pm_bf;     //!< TxPrecoding+Beamforming PDU.
    };

    /**
     * Walk the variable-length codeword tail of @p pdu once and derive the
     * end/pm_bf_raw/pm_bf triplet used by setup_ue_group(), setup_beamforming(),
     * and setup_power_control().  All returned references are valid for the
     * lifetime of @p pdu.
     *
     * @param[in] pdu  PDSCH PDU to walk.
     * @return Tail aggregating end, pm_bf_raw, and pm_bf.
     *         Return value must be checked.
     */
    [[nodiscard]] static PduTail derive_pdu_tail(const pdu_t& pdu) noexcept;

    // ------------------------------------------------------------------
    // setup_cell helpers
    // ------------------------------------------------------------------

    /**
     * Populate cuphyPdschCellDynPrm_t for the current cell.
     *
     * Sets the per-cell dynamic params (cellPrmStatIdx, cellPrmDynIdx, slotNum,
     * testModel) and registers the cell in the cell index lists, then bumps
     * cell_grp_info.nCells.
     *
     * Note: CSI-RS counters (cell_dyn.nCsiRsPrms, cell_dyn.csiRsPrmsOffset,
     * cell_grp_info.nCsiRsPrms) are intentionally NOT written here. That field
     * family is owned exclusively by process_aggr_csirs_channel to keep a single
     * writer per field and avoid races on the shared cell_group_command.
     *
     * @param[in,out] params  Active pdsch_params for this slot.
     * @param[in]     req     DL TTI request containing nPDUsOfEachType.
     */
    void init_cell_dyn_prm(slot_command_api::pdsch_params& params,
                            const scf_fapi_dl_tti_req_t&   req,
                            uint32_t                       msg_cell_id) noexcept;

    // ------------------------------------------------------------------
    // parse() helpers
    // ------------------------------------------------------------------

    /**
     * Walk the variable-length codeword array and the fixed PDU tail, then
     * populate the UE and codeword entries in pdsch_params.
     *
     * Covers steps 1–3 of update_cell_command:
     *   - Capture UE index (cell_grp_info.nUes) and codeword base index (nCws).
     *   - For each codeword in the SCF FAPI variable-length embedded array:
     *     fill cuphyPdschCwPrm_t (MCS, TB size, RV, maxLayers, maxQm, n_PRB_LBRM)
     *     and advance nCws / cw_byte_offset_.
     *   - Parse scf_fapi_pdsch_pdu_end_t that follows the codeword array and
     *     populate the matching cuphyPdschUePrm_t fields (RNTI, DMRS ports,
     *     BWP start, scrambling IDs, layer count, reference point).
     *
     * Note: cell_grp_info.nUes is NOT incremented here; that is deferred to
     * setup_ue_group() after all error checks succeed.
     *
     * @param[in,out] params  Active pdsch_params for this slot.
     * @param[in]     pdu     PDSCH PDU being processed.
     */
    [[nodiscard]] std::uint64_t setup_ue_and_codeword_indices(
        slot_command_api::pdsch_params& params,
        const pdu_t&                    pdu) noexcept;

    /**
     * Match or create a UE group for this PDU using policy G, then register
     * the UE in the group and increment cell_grp_info.nUes.
     *
     * Searches ue_grp_info[cell_ue_group_idx_start .. MAX_PDSCH_UE_GROUPS) via
     * G::matches().  On a miss, allocates a new group and calls G::init_new_group()
     * to populate its PRB fields (RA-Type 0 or RA-Type 1 depending on the PDU).
     *
     * Also maintains @c params.ue_grp_idx_bfw_id_map (P2 from the design plan):
     *   - Existing group: reads @c ue_grp_idx_bfw_id_map[grp_idx] into @c bfw_idx.
     *   - New group: reads @c nue_grps_per_cell[local_cell_ordinal] into @c bfw_idx,
     *     writes the map entry, then increments @c nue_grps_per_cell.
     *
     * @note setup_beamforming() may roll back nUes and nUeGrps on a later
     *       check_bf_pc_params() failure.
     *
     * @param[in,out] params  Active pdsch_params for this slot.
     * @param[in]     pdu     PDSCH PDU being processed.
     * @param[in]     tail    Pre-computed PDU tail from derive_pdu_tail().
     * @return UeGroupResult holding a UeGroupContext on success; an unexpected
     *         UeGroupErrorCode on failure (NoCells, UeGroupsFull, or
     *         UeGroupAtCapacity). On failure, nCws and cw_byte_offset_ have
     *         NOT been rolled back — parse() must do so.
     *         Return value must be checked.
     */
    [[nodiscard]] UeGroupResult setup_ue_group(
        slot_command_api::pdsch_params& params,
        const pdu_t&                    pdu,
        const PduTail&                  tail) noexcept;

    /**
     * Populate DMRS symbols and UE-group DMRS configuration.
     *
     * @param[in,out] params  Active pdsch_params for this slot.
     * @param[in]     pdu     PDSCH PDU being processed.
     */
    void setup_dmrs(
        slot_command_api::pdsch_params& params,
        const pdu_t&                    pdu) noexcept;

    /**
     * Roll back all state written by setup_ue_and_codeword_indices() and
     * setup_ue_group() for the current (last) UE, keeping pdsch_params consistent.
     *
     * Called by setup_beamforming() when check_bf_pc_params() fails.
     *
     * @param[in,out] params  Active pdsch_params to roll back.
     * @param[in]     pdu     PDU whose codewords drove the writes being undone.
     */
    void rollback_ue(slot_command_api::pdsch_params& params,
                     const pdu_t&                    pdu) noexcept;

    /**
     * Fill beamforming and BFW-coefficient parameters when bf/mmimo enabled,
     * then populate fh_params (or skip via NullFhCallable when direct mode is on).
     *
     * Calls rollback_ue() and returns tl::unexpected(BfParamsInvalid) when
     * check_bf_pc_params() fails, so parse() can skip setup_power_control()
     * and avoid corrupting the previous UE's beta values.
     *
     * After BF/PC validation succeeds, dispatches FH population at the call
     * site between the real fill lambda and @c NullFhCallable based on
     * @c fh_enabled_for_msg_ (cached in @c setup_cell() from
     * @c view_->is_fapi_to_cplane_direct_enabled()).
     *
     * @param[in]     sfn     System frame number (carried into the error value).
     * @param[in]     slot    Slot number (carried into the error value).
     * @param[in,out] params  Active pdsch_params for this slot.
     * @param[in]     pdu     PDSCH PDU being processed.
     * @param[in]     tail    Pre-computed PDU tail from derive_pdu_tail().
     * @param[in]     ue_grp  UE-group context from setup_ue_group() (drives fh_params fill).
     * @return empty expected on success; tl::unexpected(SlotParseError) with
     *         Code::BfParamsInvalid if check_bf_pc_params() failed and state
     *         was rolled back.
     *         Return value must be checked.
     *
     * @note parse() checks this return value: on failure it returns false and
     *       skips setup_power_control() to avoid corrupting the previous UE's
     *       beta values.
     */
    [[nodiscard]] SlotParseResult setup_beamforming(
        uint16_t                        sfn,
        uint16_t                        slot,
        slot_command_api::pdsch_params& params,
        const pdu_t&                    pdu,
        const PduTail&                  tail,
        const UeGroupContext&           ue_grp) noexcept;

    /**
     * Fill power-control parameters (PDSCH EPRE, ratio, etc.).
     *
     * @param[in,out] params  Active pdsch_params for this slot.
     * @param[in]     tail    Pre-computed PDU tail from derive_pdu_tail().
     */
    void setup_power_control(
        slot_command_api::pdsch_params& params,
        const PduTail&                  tail) noexcept;

    /**
     * Build FhFillArgs from the parser state and dispatch the FH fill.
     *
     * Dispatches at the call site between the real lambda (which populates one
     * @c pdsch_fh_prepare_params + @c pc_bf_arr entry) and @c NullFhCallable
     * (which the compiler eliminates) based on @c fh_enabled_for_msg_.
     *
     * Applies the mMIMO null-BFW guard (P4): passes @c bfw = nullptr to
     * @c FhFillArgs when @c mmimo_enabled && pm_bf.dig_bf_interfaces == 0.
     *
     * @param[in,out] params  Active pdsch_params for this slot.
     * @param[in]     tail    Pre-computed PDU tail from derive_pdu_tail().
     * @param[in]     ue_grp  UE-group context from setup_ue_group().
     */
    void setup_fh_params(slot_command_api::pdsch_params& params,
                          const PduTail&                  tail,
                          const UeGroupContext&           ue_grp) noexcept;

    /**
     * Update precoding-matrix weight indices for the current UE.
     *
     * Re-implements update_pm_weights_cuphy()
     * (scf_5g_slot_commands_pdsch_csirs.cpp:582) using params.pm_info and
     * view_->pm_map().  Only called when pm_enabled() is true and
     * mmimo_enabled() is false (digital precoding path).
     *
     * For each PRG in pm_bf: looks up the PMI in pm_weight_map, inserts the
     * precoding matrix into params.pm_info on a cache miss, and sets
     * ue.pmwPrmIdx / cell_grp.pPmwPrms.  Disables precoded BF for this UE
     * when a PRG has pmi == 0 or dig_bf_interfaces == 0 (dynamic BFW).
     *
     * @param[in,out] params  Active pdsch_params for this slot.
     * @param[in]     pm_bf   Parsed TxPrecoding+Beamforming PDU for this UE.
     */
    void apply_pm_weights(slot_command_api::pdsch_params&             params,
                          const scf_fapi_tx_precoding_beamforming_t&  pm_bf) noexcept;
};

// ---------------------------------------------------------------------------
// Inline definitions
// ---------------------------------------------------------------------------

template<DlModuleView V, UeGroupingPolicy G>
bool PdschPduParser<V, G>::parse(uint16_t sfn,
                               uint16_t slot,
                               const pdu_t& pdu) noexcept
{
    if (!cell_setup_valid_) [[unlikely]] { return false; }

    // view_->slot_command() calls slot_detail() → getMPlaneConfig(), which can
    // throw std::runtime_error on misconfiguration.  Catch here so the noexcept
    // contract of the surrounding dispatch infrastructure is upheld.
    try
    {
        auto* params = view_->slot_command().cell_groups.get_pdsch_params();
        if (!params) [[unlikely]] { return false; }

        if (params->cell_grp_info.nUes >= static_cast<uint16_t>(slot_command_api::MAX_PDSCH_UE_PER_TTI)) [[unlikely]] {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "PdschPduParser: sfn={} slot={} nUes={} >= MAX_PDSCH_UE_PER_TTI={}; dropping PDU",
                       sfn, slot, params->cell_grp_info.nUes, slot_command_api::MAX_PDSCH_UE_PER_TTI);
            return false;
        }
        if (static_cast<uint32_t>(params->cell_grp_info.nCws) + static_cast<uint32_t>(pdu.num_codewords)
                > static_cast<uint32_t>(slot_command_api::MAX_PDSCH_UE_CW_PER_TTI)) [[unlikely]] {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "PdschPduParser: sfn={} slot={} nCws={} + num_cw={} > MAX_PDSCH_UE_CW_PER_TTI={}; dropping PDU",
                       sfn, slot,
                       params->cell_grp_info.nCws, pdu.num_codewords,
                       slot_command_api::MAX_PDSCH_UE_CW_PER_TTI);
            return false;
        }

        const std::uint64_t pdu_tb_bytes = setup_ue_and_codeword_indices(*params, pdu);
        const PduTail tail = derive_pdu_tail(pdu);
        fh_committed_for_pdu_ = false;  // Reset before setup_beamforming() may commit an FH entry.
        const auto& ue_grp_result = setup_ue_group(*params, pdu, tail);
        if (!ue_grp_result.has_value()) [[unlikely]] {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "PdschPduParser: sfn={} slot={} setup_ue_group failed: {}",
                       sfn, slot,
                       wise_enum::to_string(ue_grp_result.error()));
            for (uint8_t cw = 0u; cw < pdu.num_codewords && params->cell_grp_info.nCws > 0u; ++cw) {
                cw_byte_offset_ -= params->ue_cw_info[params->cell_grp_info.nCws - 1u].tbSize;
                --params->cell_grp_info.nCws;
            }
            return false;
        }
        const UeGroupContext& ue_grp_ctx = *ue_grp_result;
        setup_dmrs(*params, pdu);
        // BF failure triggers rollback (nUes decremented) — skip power-control to
        // avoid corrupting the previous UE's beta_qam / beta_dmrs.
        if (!setup_beamforming(sfn, slot, *params, pdu, tail, ue_grp_ctx).has_value()) [[unlikely]] {
            return false;
        }
        setup_power_control(*params, tail);
        if (stats_message_active_ && stats_batch_ != nullptr) {
            stats_batch_->add_pdsch_bytes(stats_cell_id_, pdu_tb_bytes);
        }

        return true;
    }
    catch (const std::exception& ex)
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PdschPduParser::parse sfn={} slot={} threw: {}",
                   sfn, slot, ex.what());
        return false;
    }
    catch (...)
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PdschPduParser::parse sfn={} slot={} threw unknown exception",
                   sfn, slot);
        return false;
    }
}

template<DlModuleView V, UeGroupingPolicy G>
std::uint64_t PdschPduParser<V, G>::setup_ue_and_codeword_indices(
    slot_command_api::pdsch_params& params,
    const pdu_t&                    pdu) noexcept
{
    // nUes and nCws are reset by pdsch_params::reset() (via grp_cmd->reset() in
    // reset_cell_command) — no local shadow copies needed; use them directly.
    auto& ue = params.ue_info[params.cell_grp_info.nUes];
    ue.nCw   = pdu.num_codewords;

    // Hoist pdu fields in ascending offset order (rnti@2, bwp_start@8) before
    // the variable-length codeword loop.
    ue.rnti     = pdu.rnti;           // pdu@2
    ue.BWPStart = pdu.bwp.bwp_start;  // pdu@8

    // SCF FAPI embeds a variable-length codeword array inside the PDU struct.
    // Pointer arithmetic is required; reinterpret_cast is the only portable
    // mechanism for walking this layout (SCF 222.10.02 §3.4.3.1).
    const auto* cw_ptr = reinterpret_cast<const uint8_t*>(&pdu.codewords[0]);

    // Capture the base index once before the loop — pCwIdxs must point to the
    // first codeword entry for this UE, not the last one written.
    ue.pCwIdxs = &params.ue_cw_index_info[params.cell_grp_info.nCws];

    std::uint64_t tb_bytes{};
    // Prefer CONFIG.request nTxAnt (via the cell registered by setup_cell).
    // Fall back to k_max_layers when no cell is registered yet.
    uint16_t num_dl_ant = lbrm::k_max_layers;
    if (params.cell_grp_info.nCells > 0u && !params.cell_index_list.empty())
    {
        const auto carrier = static_cast<uint32_t>(params.cell_index_list.back());
        const slot_command_api::slot_indication slot_ind{};
        num_dl_ant = view_->cell_view(carrier, slot_ind).cell_params().nTxAnt;
    }
    for (uint8_t cw = 0; cw < pdu.num_codewords; ++cw) {
        const auto* p_cw  = reinterpret_cast<const scf_fapi_pdsch_codeword_t*>(cw_ptr);
        auto&       ue_cw = params.ue_cw_info[params.cell_grp_info.nCws];

        ue_cw.pUePrm         = &ue;
        ue_cw.mcsIndex       = p_cw->mcs_index;
        ue_cw.mcsTableIndex  = p_cw->mcs_table;
        ue_cw.targetCodeRate = p_cw->target_code_rate;
        ue_cw.qamModOrder    = p_cw->qam_mod_order;
        ue_cw.rv             = p_cw->rv_index;
        ue_cw.tbSize         = p_cw->tb_size;
        ue_cw.tbStartOffset  = cw_byte_offset_;
        ue_cw.maxLayers      = lbrm::compute_max_layers(num_dl_ant);
        ue_cw.maxQm          = lbrm::compute_max_qm(p_cw->mcs_table);
        ue_cw.n_PRB_LBRM     = lbrm::compute_n_prb_lbrm(pdu.bwp.bwp_size);

        params.ue_cw_index_info[params.cell_grp_info.nCws] = params.cell_grp_info.nCws;
        cw_byte_offset_ += ue_cw.tbSize;
        tb_bytes += ue_cw.tbSize;
        ++params.cell_grp_info.nCws;
        cw_ptr += sizeof(scf_fapi_pdsch_codeword_t);
    }

    // The fixed PDU tail immediately follows the variable-length codeword array.
    const auto* end = reinterpret_cast<const scf_fapi_pdsch_pdu_end_t*>(cw_ptr);

    // Access end fields in ascending offset order (d-cache line efficiency).
    ue.dataScramId  = end->data_scrambling_id;    // end@0
    ue.nUeLayers    = end->num_of_layers;          // end@2
    ue.refPoint     = end->ref_point;              // end@4
    ue.dmrsScrmId   = end->dl_dmrs_scrambling_id;  // end@8
    ue.scid         = end->sc_id;                  // end@10
    const auto dmrs_ports_raw = end->dmrs_ports;   // end@12 — single load, two consumers
    ue.dmrsPortBmsk = static_cast<uint16_t>(dmrs_ports_raw & 0xFFFu);
    ue.nlAbove16    = static_cast<uint8_t>((dmrs_ports_raw >> k_nlabove16_bit_loc) & 0x1u);
    return tb_bytes;
}

template<DlModuleView V, UeGroupingPolicy G>
typename PdschPduParser<V, G>::PduTail
PdschPduParser<V, G>::derive_pdu_tail(const pdu_t& pdu) noexcept
{
    const auto* cw_ptr = reinterpret_cast<const uint8_t*>(&pdu.codewords[0])
                         + pdu.num_codewords * sizeof(scf_fapi_pdsch_codeword_t);
    const auto& end    = *reinterpret_cast<const scf_fapi_pdsch_pdu_end_t*>(cw_ptr);
    // pdu_bitmap bit 0: optional scf_fapi_pdsch_ptrs_t extension (SCF 222.10.02 §3.4.3.1).
    const auto* pm_bf_raw = ((pdu.pdu_bitmap & 0x1u) != 0u)
        ? reinterpret_cast<const scf_fapi_pdsch_ptrs_t*>(end.next)->next
        : end.next;
    return {end, pm_bf_raw,
            *reinterpret_cast<const scf_fapi_tx_precoding_beamforming_t*>(pm_bf_raw)};
}

template<DlModuleView V, UeGroupingPolicy G>
UeGroupResult PdschPduParser<V, G>::setup_ue_group(
    slot_command_api::pdsch_params& params,
    const pdu_t&                    pdu,
    const PduTail&                  tail) noexcept
{
    // Guard against unsigned underflow when setup_cell() was never called
    // (e.g. SCF_FAPI_10_04 path skips setup_cell if nPDUsOfEachType[PDSCH]==0
    // but a malformed payload still carries a PDSCH PDU).
    if (params.cell_grp_info.nCells == 0u) [[unlikely]] {
        return tl::make_unexpected(UeGroupErrorCode::NoCells);
    }

    const auto& end = tail.end;
    const auto local_cell_idx = static_cast<uint32_t>(params.cell_grp_info.nCells) - 1u;

    // nUes has not been incremented yet; use it as the current UE's index.
    const auto ue_idx = params.cell_grp_info.nUes;
    auto&      ue     = params.ue_info[ue_idx];

    // Search for an existing compatible UE group using the policy predicate.
    // Only groups starting at cell_ue_group_idx_start belong to the current cell.
    const auto grp_span = std::span{
        params.ue_grp_info + params.cell_ue_group_idx_start,
        params.ue_grp_info + params.cell_grp_info.nUeGrps};

    auto iter = std::ranges::find_if(grp_span,
        [&](const cuphyPdschUeGrpPrm_t& g) noexcept {
            return G::matches(g, end, pdu);
        });

    const bool      is_new_grp = (iter == grp_span.end());
    const std::size_t grp_idx  = is_new_grp
        ? static_cast<std::size_t>(params.cell_grp_info.nUeGrps)
        : static_cast<std::size_t>(iter - grp_span.begin()) + params.cell_ue_group_idx_start;

    if (is_new_grp && params.cell_grp_info.nUeGrps
            >= static_cast<uint16_t>(slot_command_api::MAX_PDSCH_UE_GROUPS)) [[unlikely]] {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PdschPduParser: rnti=0x{:04X} nUeGrps={} >= MAX_PDSCH_UE_GROUPS={}; dropping PDU",
                   pdu.rnti, params.cell_grp_info.nUeGrps, slot_command_api::MAX_PDSCH_UE_GROUPS);
        return tl::make_unexpected(UeGroupErrorCode::UeGroupsFull);
    }

    auto& ue_grp = params.ue_grp_info[grp_idx];

    if (ue_grp.nUes >= static_cast<uint16_t>(slot_command_api::MAX_PDSCH_UE_PER_TTI)) [[unlikely]] {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PdschPduParser: rnti=0x{:04X} grp_idx={} nUes={} >= MAX_PDSCH_UE_PER_TTI={}; dropping UE from group",
                   pdu.rnti, grp_idx, ue_grp.nUes, slot_command_api::MAX_PDSCH_UE_PER_TTI);
        return tl::make_unexpected(UeGroupErrorCode::UeGroupAtCapacity);
    }

    // Link UE → group; register UE index in the group's member list.
    ue.pUeGrpPrm                     = &ue_grp;
    ue_grp.pUePrmIdxs[ue_grp.nUes]   = static_cast<uint16_t>(ue_idx);
    ++ue_grp.nUes;

    // Update DMRS info (applies to both new and existing groups).
    auto& ue_dmrs              = params.ue_dmrs_info[grp_idx];
    ue_dmrs.nDmrsCdmGrpsNoData = end.num_dmrs_cdm_grps_no_data;

    // P2 — Maintain ue_grp_idx_bfw_id_map:
    //   Existing group: read map entry into bfw_idx.
    //   New group:      bfw_idx = nue_grps_per_cell[local_cell_idx]; write map entry,
    //                   then increment nue_grps_per_cell.
    // Mirrors legacy scf_5g_slot_commands_pdsch_csirs.cpp L310-317.
    uint32_t bfw_idx{};
    if (is_new_grp) {
        bfw_idx = params.nue_grps_per_cell[local_cell_idx];
        params.ue_grp_idx_bfw_id_map[grp_idx] = bfw_idx;

        ++params.cell_grp_info.nUeGrps;
        ++params.nue_grps_per_cell[local_cell_idx];

        // Read end@5 before G::init_new_group() reads end@14+ (ascending offset order).
        ue_grp.dmrsSymLocBmsk = end.dl_dmrs_sym_pos;               // end@5
        // Delegate PRB-allocation initialisation to the policy (RA-Type 0 / 1).
        G::init_new_group(ue_grp, params, grp_idx, end, pdu);      // end@14+

        ue_grp.pDmrsDynPrm   = &ue_dmrs;
        ue_grp.pdschStartSym  = end.start_sym_index;                // end@56
        ue_grp.nPdschSym      = end.num_symbols;                    // end@57
        // Link to the current cell's dynamic params (last cell added this slot).
        ue_grp.pCellPrm       = &params.cell_dyn_info[local_cell_idx];

        NVLOGD_FMT(detail::k_tag,
                   "setup_ue_group: new group rnti=0x{:04X} grp_idx={} "
                   "local_cell_idx={} bfw_idx={} nUeGrps={}",
                   pdu.rnti, grp_idx, local_cell_idx, bfw_idx,
                   params.cell_grp_info.nUeGrps);
    } else {
        bfw_idx = params.ue_grp_idx_bfw_id_map[grp_idx];
        NVLOGD_FMT(detail::k_tag,
                   "setup_ue_group: reuse group rnti=0x{:04X} grp_idx={} bfw_idx={}",
                   pdu.rnti, grp_idx, bfw_idx);
    }

    // Increment nUes now that the UE is fully registered in its group.
    // setup_beamforming() will roll this back on check_bf_pc_params() failure.
    ++params.cell_grp_info.nUes;

    return UeGroupContext{
        .is_new  = is_new_grp,
        .grp_idx = static_cast<uint16_t>(grp_idx),
        .bfw_idx = bfw_idx,
    };
}

template<DlModuleView V, UeGroupingPolicy G>
void PdschPduParser<V, G>::setup_dmrs(
    [[maybe_unused]] slot_command_api::pdsch_params& params,
    [[maybe_unused]] const pdu_t& pdu) noexcept
{
    // TODO
}

template<DlModuleView V, UeGroupingPolicy G>
void PdschPduParser<V, G>::rollback_ue(
    slot_command_api::pdsch_params& params,
    const pdu_t&                    pdu) noexcept
{
    auto& ue     = params.ue_info[params.cell_grp_info.nUes - 1u];
    auto& ue_grp = *ue.pUeGrpPrm;

    // Snapshot the carrier id for the current cell BEFORE any nCells mutation
    // below — eliminates the underflow path (carrier_id(nCells-1) when nCells
    // becomes 0 in the cell-pop case). nCells > 0 is implied here by the
    // function's precondition (nUes > 0 was checked at the first line above,
    // and setup_ue_group only increments nUes after asserting nCells > 0).
    // Computed only when an FH entry was committed; otherwise unused.
    uint16_t fh_cell_index = 0u;
    if (fh_enabled_for_msg_ && fh_committed_for_pdu_) {
        fh_cell_index = static_cast<uint16_t>(view_->carrier_id(msg_cell_id_));
    }

    // nUes==1 in the group means this UE created it during this parse() call.
    const bool was_new_grp = (ue_grp.nUes == 1u);
    --ue_grp.nUes;

    for (uint8_t cw = 0u; cw < pdu.num_codewords; ++cw) {
        if (params.cell_grp_info.nCws == 0u) [[unlikely]] { break; }
        cw_byte_offset_ -= params.ue_cw_info[params.cell_grp_info.nCws - 1u].tbSize;
        --params.cell_grp_info.nCws;
    }

    if (was_new_grp) {
        --params.cell_grp_info.nUeGrps;
        --params.nue_grps_per_cell[params.cell_grp_info.nCells - 1u];
        if (params.cell_grp_info.nUeGrps == 0u
                && params.cell_grp_info.nCells > 0u) {
            --params.cell_grp_info.nCells;
            if (!params.cell_index_list.empty()) [[unlikely]] { params.cell_index_list.pop_back(); }
            if (!params.phy_cell_index_list.empty()) [[unlikely]] { params.phy_cell_index_list.pop_back(); }
        }
    }

    --params.cell_grp_info.nUes;

    // Roll back the FH entry only when fill_pdsch_fh_entry committed one this
    // PDU. Uses fh_cell_index snapshotted above (no underflow possible).
    // Skip entirely when fh_enabled_for_msg_ was false (direct mode).
    if (fh_enabled_for_msg_ && fh_committed_for_pdu_) {
        auto* grp_cmd = view_->group_command();
        if (grp_cmd != nullptr && grp_cmd->fh_params.total_num_pdsch_pdus > 0u) {
            auto& fh = grp_cmd->fh_params;
            --fh.total_num_pdsch_pdus;
            if (fh.num_pdsch_fh_params[fh_cell_index] > 0u) {
                --fh.num_pdsch_fh_params[fh_cell_index];
            }
            NVLOGD_FMT(detail::k_tag,
                       "rollback_ue: rolled back FH entry cell_index={} "
                       "total_num_pdsch_pdus={} num_pdsch_fh_params[cell]={}",
                       fh_cell_index, fh.total_num_pdsch_pdus,
                       fh.num_pdsch_fh_params[fh_cell_index]);
        }
        fh_committed_for_pdu_ = false;
    }
}

template<DlModuleView V, UeGroupingPolicy G>
SlotParseResult PdschPduParser<V, G>::setup_beamforming(
    uint16_t                        sfn,
    uint16_t                        slot,
    slot_command_api::pdsch_params& params,
    const pdu_t&                    pdu,
    const PduTail&                  tail,
    const UeGroupContext&           ue_grp) noexcept
{
    const auto& pm_bf = tail.pm_bf;

    const uint16_t num_prgs      = pm_bf.num_prgs;
    const uint8_t  dig_bf_ifaces = pm_bf.dig_bf_interfaces;
    const bool     mmimo         = view_->mmimo_enabled();

    if (!check_bf_pc_params(static_cast<int>(num_prgs),
                             static_cast<int>(dig_bf_ifaces),
                             mmimo)) [[unlikely]]
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PdschPduParser: check_bf_pc_params failed: "
                   "numPRGs={} digBFInterfaces={} mmimo_enabled={}",
                   num_prgs, dig_bf_ifaces, mmimo);
        rollback_ue(params, pdu);
        return tl::unexpected(SlotParseError{
            SlotParseError::Code::BfParamsInvalid, sfn, slot});
    }

    params.tb_data.pBufferType = cuphyPdschDataIn_t::CPU_BUFFER;

    auto& ue        = params.ue_info[params.cell_grp_info.nUes - 1u];
    ue.enablePrcdBf = static_cast<uint8_t>(view_->pm_enabled());

    // Fill precoding-matrix weights when digital precoding is active and
    // MU-MIMO is disabled (mirrors update_cell_command lines 496–499 with
    // explicit !mmimo guard).
    if (view_->pm_enabled() && !mmimo) {
        apply_pm_weights(params, pm_bf);
    }

    // P5: populate fh_params via NullFhCallable dispatch when direct mode is on.
    // setup_fh_params() handles the cell_index lookup, mmimo null-BFW guard,
    // and the call-site dispatch between real fill and NullFhCallable.
    setup_fh_params(params, tail, ue_grp);
    return {};
}

template<DlModuleView V, UeGroupingPolicy G>
void PdschPduParser<V, G>::apply_pm_weights(
    slot_command_api::pdsch_params&            params,
    const scf_fapi_tx_precoding_beamforming_t& pm_bf) noexcept
{
    auto& ue       = params.ue_info[params.cell_grp_info.nUes - 1u];
    auto& cell_grp = params.cell_grp_info;
    auto& cache    = params.pmw_idx_cache;
    auto& list     = params.pm_info;
    const auto& pm_map = view_->pm_map();

    // carrier_id is already recorded in cell_index_list during init_cell_dyn_prm.
    // The PM weight map is keyed by (pm_idx | carrier_id << 16) — the same format
    // used at CONFIG time in scf_5g_fapi_phy.cpp (pm_pdu.pmi_idx | cell_id << 16)
    // and copied to all cells via copy_precoding_configs_to().
    const auto k_cell_index = view_->carrier_id(msg_cell_id_);
    uint16_t offset = 0u;

    for (uint16_t i = 0u; i < pm_bf.num_prgs; ++i) {
        const uint16_t pm_idx = pm_bf.pm_idx_and_beam_idx[i + offset];

        // pm_idx == 0 → no precoding; dig_bf_interfaces == 0 → dynamic BFW.
        if (pm_idx == 0u || pm_bf.dig_bf_interfaces == 0u) {
            offset += static_cast<uint16_t>(pm_bf.dig_bf_interfaces + 1u);
            ue.enablePrcdBf = 0u;
            continue;
        }

        // Upper 16 bits encode the cell; lower 16 bits encode the PMI.
        const uint32_t pmi = static_cast<uint32_t>(pm_idx)
                             | (static_cast<uint32_t>(k_cell_index) << 16u);
        offset += static_cast<uint16_t>(pm_bf.dig_bf_interfaces + 1u);

        const auto hit = std::ranges::find(cache, pmi);
        uint16_t matrix_index = std::numeric_limits<uint16_t>::max();

        if (hit != cache.end()) {
            matrix_index = static_cast<uint16_t>(
                std::ranges::distance(cache.begin(), hit));
        } else {
            const auto pmw_it = pm_map.find(pmi);
            if (pmw_it == pm_map.end()) [[unlikely]] {
                ue.enablePrcdBf = 0u;
                continue;
            }

            const uint16_t layers = pmw_it->second.layers;
            const uint16_t ports  = pmw_it->second.ports;

            if (layers == 0u || ports == 0u) [[unlikely]] {
                NVLOGW_FMT(detail::k_tag,
                           "apply_pm_weights: pmi=0x{:08x} layers={} ports={} — "
                           "zero dimension, skipping entry",
                           pmi, layers, ports);
                ue.enablePrcdBf = 0u;
                continue;
            }

            const uint32_t n = static_cast<uint32_t>(layers)
                               * static_cast<uint32_t>(ports);
            static constexpr uint32_t k_matrix_capacity =
                MAX_DL_LAYERS_PER_TB * MAX_DL_PORTS;

            if (n > k_matrix_capacity) [[unlikely]] {
                NVLOGW_FMT(detail::k_tag,
                           "apply_pm_weights: pmi=0x{:08x} n=layers*ports={}*{}={} "
                           "exceeds matrix capacity {}, skipping entry",
                           pmi, layers, ports, n, k_matrix_capacity);
                ue.enablePrcdBf = 0u;
                continue;
            }

            if (pmw_it->second.weights.nPorts != static_cast<uint8_t>(ports)) [[unlikely]] {
                NVLOGW_FMT(detail::k_tag,
                           "apply_pm_weights: pmi=0x{:08x} weights.nPorts={} != "
                           "ports={} — inconsistent, skipping entry",
                           pmi, pmw_it->second.weights.nPorts, ports);
                ue.enablePrcdBf = 0u;
                continue;
            }

            if (cell_grp.nPrecodingMatrices >= static_cast<uint32_t>(list.size())) [[unlikely]] {
                NVLOGW_FMT(detail::k_tag,
                           "apply_pm_weights: pmi=0x{:08x} pm_info size ({}) exhausted, "
                           "skipping entry",
                           pmi, list.size());
                ue.enablePrcdBf = 0u;
                continue;
            }

            auto& slot   = list[cell_grp.nPrecodingMatrices];
            matrix_index = static_cast<uint16_t>(cell_grp.nPrecodingMatrices);
            cache.push_back(pmi);
            slot.nPorts  = pmw_it->second.weights.nPorts;
            std::ranges::copy(
                std::span{pmw_it->second.weights.matrix, n},
                slot.matrix);
            ++cell_grp.nPrecodingMatrices;
        }

        if (matrix_index != std::numeric_limits<uint16_t>::max()) {
            ue.pmwPrmIdx      = matrix_index;
            cell_grp.pPmwPrms = list.data();
        }
    }
}

template<DlModuleView V, UeGroupingPolicy G>
void PdschPduParser<V, G>::setup_fh_params(
    slot_command_api::pdsch_params& params,
    const PduTail&                  tail,
    const UeGroupContext&           ue_grp) noexcept
{
    const auto cell_index = static_cast<uint16_t>(view_->carrier_id(msg_cell_id_));
    const auto& pm_bf         = tail.pm_bf;
    const bool  mmimo         = view_->mmimo_enabled();
    const bool  bf_enabled    = view_->bf_enabled();
    const bool  pm_enabled    = view_->pm_enabled();

    // BFW pointer is non-null only for the mmimo dynamic-BFW path
    // (mmimo_enabled && dig_bf_interfaces == 0). Mirrors the legacy contract:
    //   - !mmimo:                 caller never fetched (prepare_dl_slot_command L2308–2320)
    //                             → nullptr.
    //   - mmimo && dig == 0:      dynamic BFW → real ring pointer.
    //   - mmimo && dig != 0:      static BFW → caller fetches, then
    //                             update_cell_command L441 nulls it out.
    // BFW slot index argument is a placeholder (0) until the previous-slot
    // ring index is threaded through DLSlotProcessor (see plan P5 / deferred).
    slot_command_api::bfw_coeff_mem_info_t* bfw = nullptr;
    if (mmimo && pm_bf.dig_bf_interfaces == 0u) {
        bfw = view_->bfw_coeff_mem_info(msg_cell_id_, /*slot_index*/ 0u);
    }

    auto& ue     = params.ue_info[params.cell_grp_info.nUes - 1u];
    auto& ue_grp_prm = params.ue_grp_info[ue_grp.grp_idx];

    auto* grp_cmd = view_->group_command();
    if (grp_cmd == nullptr) [[unlikely]] {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "setup_fh_params: group_command() == nullptr; skipping FH fill");
        return;
    }
    auto& fh = grp_cmd->fh_params;

    const FhFillArgs args{
        .cell_cmd     = view_->cell_sub_command(msg_cell_id_),
        .ue_grp       = ue_grp_prm,
        .ue           = ue,
        .pm_bf        = pm_bf,
        .bfw          = bfw,
        .bfw_idx      = ue_grp.bfw_idx,
        .ue_grp_index = ue_grp.grp_idx,
        .num_dl_prb   = view_->num_dl_prb(msg_cell_id_),
        .cell_index   = cell_index,
        .flags        = make_fh_flags(/*is_new_grp*/    ue_grp.is_new,
                                      /*bf_enabled*/    bf_enabled,
                                      /*pm_enabled*/    pm_enabled,
                                      /*mmimo_enabled*/ mmimo),
    };

    // Call-site dispatch: real lambda when populate is enabled, NullFhCallable
    // (compile-time eliminated) when direct mode is on. append_fh_entry stays
    // generic; the runtime branch picks the callable.
    if (fh_enabled_for_msg_) [[likely]] {
        fh_committed_for_pdu_ = fill_pdsch_fh_entry(fh, args);
        NVLOGD_FMT(detail::k_tag, "setup_fh_params: fill_pdsch_fh_entry returned {}", fh_committed_for_pdu_);
        if (!fh_committed_for_pdu_) [[unlikely]] {
            NVLOGW_FMT(detail::k_tag,
                       "setup_fh_params: fill_pdsch_fh_entry returned false "
                       "(capacity exhausted) cell_index={} total={}",
                       cell_index, fh.total_num_pdsch_pdus);
        }
    } else {
        // Direct mode — NullFhCallable specialization collapses to a ret.
        std::ignore = append_fh_entry(fh, cell_index, NullFhCallable{});
        fh_committed_for_pdu_ = false;
    }
}

template<DlModuleView V, UeGroupingPolicy G>
void PdschPduParser<V, G>::setup_power_control(
    slot_command_api::pdsch_params& params,
    const PduTail&                  tail) noexcept
{
    const auto* pm_bf_raw = tail.pm_bf_raw;
    const auto& pm_bf     = tail.pm_bf;

    // Compute the byte footprint of the variable-length pm_bf struct so we can
    // advance past it to the scf_fapi_tx_power_info_t that immediately follows.
    //
    // Non-dynamic (not mmimo, or has dig interfaces):
    //   fixed header (num_prgs:2 + prg_size:2 + dig_bf_interfaces:1 = 5 bytes)
    //   + numPRGs * (1 + digBFInterfaces) * sizeof(uint16_t) variable entries.
    // Dynamic (mmimo && dig_bf_interfaces == 0): fixed header only (5 bytes).
    const bool     mmimo    = view_->mmimo_enabled();
    const uint32_t bf_fixed = static_cast<uint32_t>(
        sizeof(pm_bf.num_prgs) + sizeof(pm_bf.prg_size) + sizeof(pm_bf.dig_bf_interfaces));
    const uint32_t bf_var   = (!mmimo || pm_bf.dig_bf_interfaces != 0u)
        ? static_cast<uint32_t>(pm_bf.num_prgs)
              * (1u + static_cast<uint32_t>(pm_bf.dig_bf_interfaces))
              * static_cast<uint32_t>(sizeof(uint16_t))
        : 0u;

    const auto& tx_power = *reinterpret_cast<const scf_fapi_tx_power_info_t*>(
        pm_bf_raw + bf_fixed + bf_var);

    // LUT: beta[offset][offset_ss] = 10^((offset-8 + (offset_ss-1)*3) / 20)
    //
    // SCF 222.10.02 bounds: power_control_offset ∈ [0,23] (−8..+15 dB, 1 dB step),
    // power_control_offset_ss ∈ [0,3] (−3, 0, +3, +6 dB via (ss−1)*3).
    // 24 × 4 = 96 floats = 384 bytes — fits in L1 on both x86-64 and Grace/Neoverse V2.
    //
    // Initialized once at first call (C++11 guaranteed thread-safe static init).
    // Uses expf(x * ln10/20) rather than pow(10, x/20): ~3–5× faster on both targets
    // (x86-64: SVML __svml_expf; ARM: SVE FEXPA+FSCALE pathway).
    static const auto k_beta_lut = []() noexcept {
        constexpr float k_ln10_over20 = 0.11512925464970228f; // ln(10)/20
        std::array<std::array<float, k_max_power_ctrl_offset_ss + 1u>,
                                     k_max_power_ctrl_offset    + 1u> t{};
        for (int i = 0; i <= k_max_power_ctrl_offset; ++i) {
            for (int j = 0; j <= k_max_power_ctrl_offset_ss; ++j) {
                t[i][j] = std::exp(
                    static_cast<float>((i - 8) + (j - 1) * 3) * k_ln10_over20);
            }
        }
        return t;
    }();

    // Clamp to the SCF-specified range before indexing; std::min emits UMIN on
    // ARM and CMOV/VMINPS on x86 — branchless on both targets.
    const auto o  = std::min(tx_power.power_control_offset,    k_max_power_ctrl_offset);
    const auto ss = std::min(tx_power.power_control_offset_ss, k_max_power_ctrl_offset_ss);

    auto& ue = params.ue_info[params.cell_grp_info.nUes - 1u];
    ue.beta_qam = ue.beta_dmrs = k_beta_lut[o][ss];
}

// ---------------------------------------------------------------------------
// Warm path — called once per DL_TTI.request message, not per PDU
// ---------------------------------------------------------------------------

template<DlModuleView V, UeGroupingPolicy G>
void PdschPduParser<V, G>::setup_cell(const scf_fapi_dl_tti_req_t& req,
                                      uint32_t msg_cell_id) noexcept
{
    cw_byte_offset_ = 0;
    msg_cell_id_ = msg_cell_id;
    cell_setup_valid_ = false;
    // P5: read the runtime FH gate once per setup_cell (== once per cell per slot).
    // When direct mode is on, the parser still parses PDSCH but does not populate
    // fh_params — setup_fh_params() routes through NullFhCallable to a no-op.
    fh_enabled_for_msg_ = !view_->is_fapi_to_cplane_direct_enabled();
    fh_committed_for_pdu_ = false;
    try
    {
        auto* params = view_->slot_command().cell_groups.get_pdsch_params();
        if (!params) [[unlikely]] { return; }
        if (msg_cell_id_ >= static_cast<uint32_t>(slot_command_api::MAX_CELLS_PER_CELL_GROUP)) [[unlikely]] {
            NVLOGW_FMT(detail::k_tag,
                       "PdschPduParser::setup_cell: msg_cell_id={} >= MAX_CELLS_PER_CELL_GROUP; skipping",
                       msg_cell_id_);
            return;
        }
        params->cell_ue_group_idx_start = static_cast<uint16_t>(params->cell_grp_info.nUeGrps);
        const auto prev_ncells = params->cell_grp_info.nCells;
        init_cell_dyn_prm(*params, req, msg_cell_id_);
        // Require this call to register a cell; nCells > 0 alone can be true from
        // an earlier message after init_cell_dyn_prm fails (group already full).
        cell_setup_valid_ = (params->cell_grp_info.nCells > prev_ncells);
        if (!cell_setup_valid_) [[unlikely]] {
            return;
        }

        // Per-cell FH-counter init (lifted from legacy per-PDU L461-464). Only
        // runs in legacy / populate mode. Idempotent: skipped if FH entries for
        // this cell already exist (handles repeated setup_cell calls).
        if (fh_enabled_for_msg_) {
            auto* grp_cmd = view_->group_command();
            if (grp_cmd != nullptr && params->cell_grp_info.nCells > 0u) {
                auto& fh = grp_cmd->fh_params;
                const auto cell_index =
                    static_cast<uint16_t>(view_->carrier_id(msg_cell_id_));
                if (cell_index < fh.num_pdsch_fh_params.size()
                    && fh.num_pdsch_fh_params[cell_index] == 0u) {
                    fh.start_index_pdsch_fh_params[cell_index] = fh.total_num_pdsch_pdus;
                    NVLOGD_FMT(detail::k_tag,
                               "setup_cell: sfn={} slot={} cell_index={} "
                               "start_index_pdsch_fh_params={} fh_enabled_for_msg=true",
                               req.sfn, req.slot, cell_index,
                               fh.start_index_pdsch_fh_params[cell_index]);
                }
            }
        } else {
            NVLOGD_FMT(detail::k_tag,
                       "setup_cell: sfn={} slot={} direct mode "
                       "(is_fapi_to_cplane_direct_enabled=true); FH fill will be no-op",
                       req.sfn, req.slot);
        }
    }
    catch (const std::exception& ex)
    {
        cell_setup_valid_ = false;
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PdschPduParser::setup_cell sfn={} slot={} threw: {}",
                   req.sfn, req.slot, ex.what());
    }
    catch (...)
    {
        cell_setup_valid_ = false;
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PdschPduParser::setup_cell sfn={} slot={} threw unknown exception",
                   req.sfn, req.slot);
    }
}

template<DlModuleView V, UeGroupingPolicy G>
void PdschPduParser<V, G>::init_cell_dyn_prm(slot_command_api::pdsch_params& params,
                                            const scf_fapi_dl_tti_req_t&   req,
                                            uint32_t                       msg_cell_id) noexcept
{
    const auto cell_idx = static_cast<uint32_t>(params.cell_grp_info.nCells);
    if (cell_idx >= static_cast<uint32_t>(slot_command_api::MAX_CELLS_PER_CELL_GROUP)) [[unlikely]] {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "init_cell_dyn_prm: nCells={} >= MAX_CELLS_PER_CELL_GROUP={}; skipping",
                   cell_idx, slot_command_api::MAX_CELLS_PER_CELL_GROUP);
        return;
    }
    auto& cell_dyn = params.cell_dyn_info[cell_idx];

#ifdef SCF_FAPI_10_04
    const auto csirs_count = req.nPDUsOfEachType[DL_TTI_NPDUS_IDX_CSI_RS];
#else
    // nPDUsOfEachType not available without SCF_FAPI_10_04; CSI-RS count unknown.
    constexpr uint16_t csirs_count = 0u;
#endif
    // cell_dyn.nCsiRsPrms      = csirs_count;
    // cell_dyn.csiRsPrmsOffset = static_cast<uint16_t>(params.cell_grp_info.nCsiRsPrms);
    cell_dyn.cellPrmStatIdx  = view_->cell_stat_prm_idx(msg_cell_id);
    cell_dyn.cellPrmDynIdx   = static_cast<uint16_t>(cell_idx);
    cell_dyn.slotNum         = static_cast<uint16_t>(
        (view_->staticPdschSlotNum() > -1) ? view_->staticPdschSlotNum() : req.slot);
    cell_dyn.pdschStartSym   = 0u;
    cell_dyn.nPdschSym       = 0u;
    cell_dyn.dmrsSymLocBmsk  = 0u;
#if !ENABLE_CONFORMANCE_TM_PDSCH_PDCCH
    cell_dyn.testModel       = 0;
#else
    cell_dyn.testModel       = req.testMode;
#endif
    // params.cell_grp_info.nCsiRsPrms += csirs_count;
    params.cell_index_list.push_back(view_->carrier_id(msg_cell_id));
    params.phy_cell_index_list.push_back(static_cast<int32_t>(view_->phy_cell_id(msg_cell_id)));
    params.cell_grp_info.nCells++;
}

} // namespace scf_5g_fapi

#endif // SCF_5G_FAPI_PDSCH_PDU_PARSER_HPP_INCLUDED_
