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

#if !defined(SCF_5G_FAPI_PDCCH_PDU_PARSER_HPP_INCLUDED_)
#define SCF_5G_FAPI_PDCCH_PDU_PARSER_HPP_INCLUDED_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <tuple>

#include <gsl-lite/gsl-lite.hpp>

#include "aerial/casts/casts.hpp"
#include "scf_5g_fapi_slot_concepts.hpp"
#include "scf_5g_fapi_tags.hpp"
#include "scf_5g_slot_commands_pdcch.hpp"
#include "scf_5g_fapi.h"
#include "slot_command/slot_command.hpp"
#include "nvlog.h"
#include "nvlog_fmt.hpp"
#include "aerial_event_code.h"

namespace scf_5g_fapi
{

/**
 * Parses a single PDCCH PDU from a DL_TTI.request message.
 *
 * Populates the PDCCH-specific cell-group aggregate parameters. Common
 * slot/channel registration is handled by the message thread before workers
 * run. A temporary sym_prb_info compatibility update is emitted after the
 * PDCCH-local state is populated for non-direct, PDCCH-only bring-up.
 *
 * @tparam V  Type satisfying DlModuleView.
 */
template<DlModuleView V>
class PdcchPduParser final
{
public:
    using pdu_t = scf_fapi_pdcch_pdu_t;
    static constexpr scf_fapi_dl_tti_pdu_type_t pdu_type = DL_TTI_PDU_TYPE_PDCCH;

    explicit PdcchPduParser(V& v) noexcept : view_{&v} {}

    /**
     * Per-message setup: capture cell context, the @c testMode flag for
     * CORESET dynamic parameters, and the PDCCH-only validation bridge.
     *
     * Used by both DL PDCCH carriers (DL_TTI.request) and UL grant PDCCH
     * carriers (UL_DCI.request) — the PHY-layer PDCCH setup is identical
     * for both; only the @c test_mode and @c temporary_sym_prb_info_enabled
     * derivation differs, and that derivation lives at the call site.
     *
     * @param[in] msg_cell_id                     Logical cell id from the FAPI message descriptor.
     * @param[in] test_mode                       DL_TTI testMode (0 on the UL_DCI path).
     * @param[in] temporary_sym_prb_info_enabled  PDCCH-only validation bridge.
     */
    void setup_cell(uint32_t msg_cell_id,
                    uint8_t test_mode,
                    bool temporary_sym_prb_info_enabled) noexcept;

    /**
     * Process one PDCCH PDU for the given sfn/slot.
     *
     * @param[in] sfn   System Frame Number.
     * @param[in] slot  Slot number.
     * @param[in] pdu   PDCCH PDU to process.
     * @return true on success; false if processing could not be completed.
     *         Return value must be checked.
     */
    [[nodiscard]] bool parse(uint16_t sfn, uint16_t slot, const pdu_t& pdu) noexcept;

private:
    gsl_lite::not_null<V*> view_; //!< Non-owning; must outlive this parser.
    uint8_t test_mode_{};         //!< Captured from the current DL_TTI.request.
    uint32_t logical_cell_id_{};   //!< Logical cell id from phy_mac_msg_desc::cell_id.
    int32_t carrier_id_{-1};       //!< Carrier id resolved during setup_cell; consumed by parse().
    bool temporary_sym_prb_info_enabled_{}; //!< PDCCH-only non-direct validation bridge.
    bool setup_valid_{};           //!< True after setup resolves a valid carrier id.

    [[nodiscard]] static constexpr uint32_t ceil_div(uint32_t numerator,
                                                     uint32_t denominator) noexcept
    {
        return (numerator + denominator - 1u) / denominator;
    }

    [[nodiscard]] static constexpr uint16_t dci_count_to_process(const pdu_t& pdu) noexcept
    {
        // One PDCCH PDU maps to a single CORESET, which can carry at most
        // CUPHY_PDCCH_MAX_DCIS_PER_CORESET DCIs. Cap unconditionally so the count
        // is bounded by the per-CORESET maximum in every build config; this also
        // lets dci_offsets in parse() be sized to the per-CORESET maximum instead
        // of the (much larger) per-cell-group dci_param_list size.
        return std::min<uint16_t>(pdu.num_dl_dci, CUPHY_PDCCH_MAX_DCIS_PER_CORESET);
    }

    [[nodiscard]] static constexpr uint64_t freq_domain_resource(const pdu_t& pdu) noexcept
    {
        uint64_t resource = 0u;
        for (int i = 0; i < 6; ++i) {
            resource |= static_cast<uint64_t>(pdu.freq_domain_resource[i]) << (56 - (i * 8));
        }
        return resource;
    }

    [[nodiscard]] static std::size_t tx_precoding_size(
        const scf_fapi_tx_precoding_beamforming_t& pc_bf) noexcept
    {
        return sizeof(scf_fapi_tx_precoding_beamforming_t)
            + (static_cast<std::size_t>(pc_bf.num_prgs) * sizeof(uint16_t))
            + (static_cast<std::size_t>(pc_bf.num_prgs)
                * static_cast<std::size_t>(pc_bf.dig_bf_interfaces) * sizeof(uint16_t));
    }

    [[nodiscard]] static const scf_fapi_tx_precoding_beamforming_t& tx_precoding(
        const scf_fapi_dl_dci_t& dci) noexcept
    {
        return *aerial::casts::assume_cast<scf_fapi_tx_precoding_beamforming_t>(&dci.payload[0]);
    }

    [[nodiscard]] static const scf_fapi_pdcch_tx_power_info_t& tx_power_info(
        const scf_fapi_dl_dci_t& dci,
        const scf_fapi_tx_precoding_beamforming_t& pc_bf) noexcept
    {
        return *aerial::casts::assume_cast<scf_fapi_pdcch_tx_power_info_t>(
            &dci.payload[0] + tx_precoding_size(pc_bf));
    }

    [[nodiscard]] static const scf_fapi_pdcch_dci_payload_t& dci_payload_info(
        const scf_fapi_dl_dci_t& dci,
        const scf_fapi_tx_precoding_beamforming_t& pc_bf) noexcept
    {
        return *aerial::casts::assume_cast<scf_fapi_pdcch_dci_payload_t>(
            &dci.payload[0] + tx_precoding_size(pc_bf) + sizeof(scf_fapi_pdcch_tx_power_info_t));
    }

    [[nodiscard]] static std::size_t dci_wire_size(const scf_fapi_dl_dci_t& dci) noexcept
    {
        const auto& pc_bf   = tx_precoding(dci);
        const auto& payload = dci_payload_info(dci, pc_bf);
        return sizeof(scf_fapi_dl_dci_t)
            + tx_precoding_size(pc_bf)
            + sizeof(scf_fapi_pdcch_tx_power_info_t)
            + sizeof(scf_fapi_pdcch_dci_payload_t)
            + ceil_div(static_cast<uint32_t>(payload.payload_size_bits), 8u);
    }

    [[nodiscard]] static float beta_from_power(const scf_fapi_pdcch_tx_power_info_t& power) noexcept
    {
#ifdef SCF_FAPI_10_04
        return static_cast<float>(std::pow(10.0, power.power_control_offset_ss_profile_nr / 20.0));
#else
        return static_cast<float>(std::pow(10.0, (power.power_control_offset_ss - 1) * 3.0 / 20.0));
#endif
    }

    void setup_coreset(cuphyPdcchCoresetDynPrm_t& coreset,
                       const pdu_t&               pdu,
                       uint16_t                   slot,
                       int32_t                    carrier_id,
                       const cuphyCellStatPrm_t&  cell_params) const noexcept;

    [[nodiscard]] bool setup_dci(cuphyPdcchDciPrm_t&     dci,
                                 slot_command_api::dci_payload_t& payload,
                                 const scf_fapi_dl_dci_t& msg_dci) const noexcept;

    [[nodiscard]] static bool validate_dci_payloads(uint16_t    sfn,
                                                    uint16_t    slot,
                                                    const pdu_t& pdu,
                                                    uint16_t    n_dcis_to_process) noexcept;

    void apply_pm_weights(cuphyPdcchDciPrm_t&      dci,
                          const scf_fapi_dl_dci_t& msg_dci,
                          slot_command_api::pm_group* pm_group,
                          int32_t                  carrier_id) const noexcept;
};

template<DlModuleView V>
void PdcchPduParser<V>::setup_cell(uint32_t msg_cell_id,
                                   uint8_t test_mode,
                                   bool temporary_sym_prb_info_enabled) noexcept
{
    test_mode_ = test_mode;
    logical_cell_id_ = msg_cell_id;
    carrier_id_ = -1;
    const auto carrier_id = view_->carrier_id(msg_cell_id);
    if (carrier_id < 0) [[unlikely]] {
        setup_valid_ = false;
        temporary_sym_prb_info_enabled_ = temporary_sym_prb_info_enabled;
        return;
    }

    auto* cell_group = view_->group_command();
    if (!cell_group) [[unlikely]] {
        setup_valid_ = false;
        temporary_sym_prb_info_enabled_ = temporary_sym_prb_info_enabled;
        return;
    }

    auto* params = cell_group->get_pdcch_params();
    if (!params) [[unlikely]] {
        setup_valid_ = false;
        temporary_sym_prb_info_enabled_ = temporary_sym_prb_info_enabled;
        return;
    }

    carrier_id_ = carrier_id;
    setup_valid_ = true;
    temporary_sym_prb_info_enabled_ = temporary_sym_prb_info_enabled;
}

template<DlModuleView V>
bool PdcchPduParser<V>::parse(uint16_t sfn, uint16_t slot,
                              const pdu_t& pdu) noexcept
{
    try
    {
        if (!setup_valid_) [[unlikely]] { return false; }

        auto* cell_group = view_->group_command();
        if (!cell_group || !cell_group->pdcch) [[unlikely]] { return false; }

        auto& params = *cell_group->pdcch;
        auto& group  = params.csets_group;
        if (carrier_id_ < 0) [[unlikely]] { return false; }
        const auto cell_index = carrier_id_;

        const auto n_dcis_to_process = dci_count_to_process(pdu);
        if (group.nCoresets >= group.csets.size()) [[unlikely]] {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "PdcchPduParser: sfn={} slot={} nCoresets={} >= {}; dropping PDU",
                       sfn, slot, group.nCoresets, group.csets.size());
            return false;
        }
        if ((group.nDcis + n_dcis_to_process) > group.dcis.size()) [[unlikely]] {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "PdcchPduParser: sfn={} slot={} nDcis={} + num_dl_dci={} > {}; dropping PDU",
                       sfn, slot, group.nDcis, n_dcis_to_process, group.dcis.size());
            return false;
        }
        if (!validate_dci_payloads(sfn, slot, pdu, n_dcis_to_process)) [[unlikely]] {
            return false;
        }

        const slot_command_api::slot_indication slot_ind{sfn, slot, 0u};
        const auto  cell_view  = view_->cell_view(logical_cell_id_, slot_ind);
        const auto& cell_params = cell_view.cell_params();

        // PDCCH keeps these lists per parsed PDU/CORESET, not per unique cell.
        const auto pdcch_entry_capacity =
            std::min(params.cell_index_list.capacity(), params.phy_cell_index_list.capacity());
        if (params.cell_index_list.size() >= pdcch_entry_capacity
            || params.phy_cell_index_list.size() >= pdcch_entry_capacity) [[unlikely]] {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "PdcchPduParser::parse pdcch_entry_count={} phy_entry_count={} >= capacity={}; skipping",
                       params.cell_index_list.size(),
                       params.phy_cell_index_list.size(),
                       pdcch_entry_capacity);
            return false;
        }

        // PhyPdcchAggr indexes phy_cell_index_list by CORESET index, so keep
        // these lists aligned one entry per parsed PDCCH PDU/CORESET.
        params.cell_index_list.push_back(cell_index);
        params.phy_cell_index_list.push_back(static_cast<int32_t>(cell_params.phyCellId));

        auto& coreset = group.csets[group.nCoresets];
        setup_coreset(coreset, pdu, slot, cell_index, cell_params);
        ++group.nCoresets;

        coreset.dciStartIdx   = group.nDcis;
        coreset.slotBufferIdx = static_cast<uint32_t>(cell_index);

        std::size_t dci_offset = 0u;
        const auto* dci_bytes  = reinterpret_cast<const uint8_t*>(pdu.dl_dci);
        auto*       pm_group   = cell_group->get_pm_group();
        // Sized to the per-CORESET DCI maximum (one PDU == one CORESET), not the
        // per-cell-group dci_param_list size. dci_count_to_process() bounds the
        // write index below by CUPHY_PDCCH_MAX_DCIS_PER_CORESET, so this is safe.
        std::array<std::size_t, CUPHY_PDCCH_MAX_DCIS_PER_CORESET> dci_offsets{};

        for (uint16_t i = 0u; i < pdu.num_dl_dci; ++i) {
            const auto& msg_dci = *aerial::casts::assume_cast<scf_fapi_dl_dci_t>(dci_bytes + dci_offset);
            const auto  size    = dci_wire_size(msg_dci);
            if (i < n_dcis_to_process) {
                dci_offsets[i] = dci_offset;
                auto& dci     = group.dcis[group.nDcis];
                auto& payload = group.payloads[group.nDcis];
                if (!setup_dci(dci, payload, msg_dci)) [[unlikely]] {
                    --group.nCoresets;
                    group.nDcis = coreset.dciStartIdx;
                    params.cell_index_list.pop_back();
                    params.phy_cell_index_list.pop_back();
                    return false;
                }
                apply_pm_weights(dci, msg_dci, pm_group, cell_index);
                ++group.nDcis;
            }
            dci_offset += size;
        }

        if (temporary_sym_prb_info_enabled_) {
            auto& cell_cmd = view_->cell_sub_command(logical_cell_id_);
            // const_cast is safe here: update_pdcch_sym_prb_info_for_pdcch_only_validation
            // and its downstream chain only read from the PDU. The cast is required solely
            // because that helper is a shared legacy entry point declared with a non-const
            // scf_fapi_pdcch_pdu_t&. This whole call is a temporary interim validation bridge
            // (gated by temporary_sym_prb_info_enabled_) and will be removed once the
            // PDCCH-only sym/PRB info path is handled natively.
            update_pdcch_sym_prb_info_for_pdcch_only_validation(
                cell_cmd,
                coreset,
                group.dcis,
                cell_params.nPrbDlBwp,
                pm_group,
                const_cast<pdu_t&>(pdu),
                dci_offsets.data(),
                view_->config_options(),
                cell_view.slot_detail(),
                view_->mmimo_enabled(),
                cell_index);
        }

        return true;
    }
    catch (const std::exception& ex)
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PdcchPduParser::parse cell_id={} sfn={} slot={} threw: {}",
                   logical_cell_id_, sfn, slot, ex.what());
        return false;
    }
    catch (...)
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PdcchPduParser::parse cell_id={} sfn={} slot={} threw unknown exception",
                   logical_cell_id_, sfn, slot);
        return false;
    }
}

template<DlModuleView V>
void PdcchPduParser<V>::setup_coreset(cuphyPdcchCoresetDynPrm_t& coreset,
                                      const pdu_t&               pdu,
                                      uint16_t                   slot,
                                      int32_t                    carrier_id,
                                      const cuphyCellStatPrm_t&  cell_params) const noexcept
{
    coreset.n_f                  = static_cast<uint32_t>(cell_params.nPrbDlBwp) * 12u;
    coreset.slot_number          = static_cast<uint32_t>(
        (view_->staticPdcchSlotNum() != -1) ? view_->staticPdcchSlotNum() : slot);
    coreset.start_rb             = pdu.bwp.bwp_start;
    coreset.start_sym            = pdu.start_sym_index;
    coreset.n_sym                = pdu.duration_sym;
    coreset.bundle_size          = pdu.reg_bundle_size;
    coreset.interleaver_size     = pdu.interleaver_size;
    coreset.shift_index          = pdu.shift_index;
    coreset.interleaved          = pdu.cce_reg_mapping_type;
    coreset.freq_domain_resource = freq_domain_resource(pdu);
    coreset.coreset_type         = pdu.coreset_type;
    coreset.testModel            = test_mode_;
    coreset.nDci                 = static_cast<uint8_t>(dci_count_to_process(pdu));
    coreset.slotBufferIdx        = static_cast<uint32_t>(carrier_id);
}

template<DlModuleView V>
bool PdcchPduParser<V>::setup_dci(cuphyPdcchDciPrm_t&      dci,
                                  slot_command_api::dci_payload_t& payload,
                                  const scf_fapi_dl_dci_t& msg_dci) const noexcept
{
    const auto& pc_bf       = tx_precoding(msg_dci);
    const auto& power       = tx_power_info(msg_dci, pc_bf);
    const auto& dci_payload = dci_payload_info(msg_dci, pc_bf);
    const auto  payload_len = ceil_div(static_cast<uint32_t>(dci_payload.payload_size_bits), 8u);

    if (payload_len > payload.size()) [[unlikely]] {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PdcchPduParser: DCI payload bytes={} > max={}; dropping PDU",
                   payload_len, payload.size());
        return false;
    }

    dci.rntiCrc     = msg_dci.rnti;
    dci.rntiBits    = msg_dci.scrambling_rnti;
    dci.dmrs_id     = msg_dci.scrambling_id;
    dci.aggr_level  = msg_dci.aggregation_level;
    dci.cce_index   = msg_dci.cce_index;
    dci.beta_qam    = beta_from_power(power);
    dci.beta_dmrs   = dci.beta_qam;
    dci.Npayload    = dci_payload.payload_size_bits;
    dci.enablePrcdBf = false;
    dci.pmwPrmIdx    = 0u;

    std::memcpy(payload.data(), dci_payload.payload, payload_len);
    return true;
}

template<DlModuleView V>
bool PdcchPduParser<V>::validate_dci_payloads(uint16_t    sfn,
                                              uint16_t    slot,
                                              const pdu_t& pdu,
                                              uint16_t    n_dcis_to_process) noexcept
{
    std::size_t offset = 0u;
    const auto* bytes  = reinterpret_cast<const uint8_t*>(pdu.dl_dci);
    for (uint16_t i = 0u; i < pdu.num_dl_dci; ++i) {
        const auto& msg_dci = *aerial::casts::assume_cast<scf_fapi_dl_dci_t>(bytes + offset);
        const auto& pc_bf   = tx_precoding(msg_dci);
        const auto& payload = dci_payload_info(msg_dci, pc_bf);
        const auto  payload_len = ceil_div(static_cast<uint32_t>(payload.payload_size_bits), 8u);
        if (i < n_dcis_to_process
            && payload_len > static_cast<uint32_t>(CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES)) [[unlikely]] {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "PdcchPduParser: sfn={} slot={} DCI payload bytes={} > max={}; dropping PDU",
                       sfn, slot, payload_len, CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES);
            return false;
        }
        offset += dci_wire_size(msg_dci);
    }
    return true;
}

template<DlModuleView V>
void PdcchPduParser<V>::apply_pm_weights(cuphyPdcchDciPrm_t&      dci,
                                         const scf_fapi_dl_dci_t& msg_dci,
                                         slot_command_api::pm_group* pm_group,
                                         int32_t                  carrier_id) const noexcept
{
    const auto& pc_bf = tx_precoding(msg_dci);
    dci.enablePrcdBf  = view_->config_options().precoding_enabled;

    auto disable_precoding = [&dci]() noexcept {
        dci.enablePrcdBf = false;
    };

    if (!pm_group) [[unlikely]] {
        disable_precoding();
        return;
    }

    uint16_t offset = 0u;
    for (uint16_t prg = 0u; prg < pc_bf.num_prgs; ++prg) {
        const uint16_t pmi = pc_bf.pm_idx_and_beam_idx[prg + offset];
        // pmi == 0 → no precoding; dig_bf_interfaces == 0 → dynamic beamforming
        // (the value is not a valid PM weight index), matching the PDSCH parser
        // and legacy PDCCH path conventions.
        dci.enablePrcdBf   = dci.enablePrcdBf && (pmi != 0u)
                             && (pc_bf.dig_bf_interfaces != 0u);
        if (dci.enablePrcdBf) {
            const auto cache_pmi = static_cast<uint32_t>(pmi)
                | (static_cast<uint32_t>(carrier_id) << 16u);
            const auto pmw_iter = view_->pm_map().find(cache_pmi);
            if (pmw_iter == view_->pm_map().end() || pmw_iter->second.layers != 1u) {
                disable_precoding();
                offset += static_cast<uint16_t>(pc_bf.dig_bf_interfaces + 1u);
                continue;
            }

            auto cache_iter = std::ranges::find_if(
                pm_group->pdcch_pmw_idx_cache,
                [cache_pmi](const auto& entry) noexcept {
                    return entry.pmwIdx == cache_pmi;
                });
            if (cache_iter == pm_group->pdcch_pmw_idx_cache.end()) {
                if (pm_group->nPmPdcch >= pm_group->pdcch_list.size()
                    || pm_group->nPmPdcch >= pm_group->pdcch_pmw_idx_cache.size()) [[unlikely]] {
                    disable_precoding();
                    return;
                }

                const auto pmw_index = pm_group->nPmPdcch;
                auto& cache_entry    = pm_group->pdcch_pmw_idx_cache[pmw_index];
                cache_entry.pmwIdx   = cache_pmi;
                cache_entry.nIndex   = pmw_index;
                dci.pmwPrmIdx        = pmw_index;

                auto& dst = pm_group->pdcch_list[pmw_index];
                dst.nPorts = pmw_iter->second.weights.nPorts;
                std::copy_n(pmw_iter->second.weights.matrix,
                            pmw_iter->second.layers * pmw_iter->second.ports,
                            dst.matrix);
                ++pm_group->nPmPdcch;
            }
            else {
                dci.pmwPrmIdx = static_cast<uint16_t>(cache_iter->nIndex);
            }
        }
        offset += static_cast<uint16_t>(pc_bf.dig_bf_interfaces + 1u);
    }
}

} // namespace scf_5g_fapi

#endif // SCF_5G_FAPI_PDCCH_PDU_PARSER_HPP_INCLUDED_
