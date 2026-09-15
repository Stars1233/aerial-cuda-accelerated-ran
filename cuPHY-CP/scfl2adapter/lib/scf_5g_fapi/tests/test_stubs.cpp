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

/**
 * @file test_stubs.cpp
 * @brief Stub implementations for scf_5g_fapi functions called by the parsers.
 *
 * check_bf_pc_params(), update_beam_list(), update_prb_sym_list() are defined
 * in scf_5g_slot_commands_common.cpp, which transitively pulls in cuphy/CUDA
 * kernels. This file provides minimal stubs so unit tests can link without
 * that heavy dependency chain.
 *
 * Do NOT add this file to the production scf_5g_fapi target_sources.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "nv_phy_fapi_msg_common.hpp"   // nv::slot_detail_t
#include "scf_5g_fapi.h"
#include "scf_5g_fapi_ssb_pdu_parser.hpp"
#include "scf_5g_fapi_ul_validate.hpp"
#include "nv_phy_config_option.hpp"
#include "nv_phy_limit_errors.hpp"
#include "nv_phy_mac_transport.hpp"
#include "slot_command/slot_command.hpp"
#include "scf_5g_slot_commands_pdcch.hpp"
#include "aerial-fh-driver/oran.hpp"

int validate_srs_pdu_l1_limits(const scf_fapi_srs_pdu_t&,
                               nv::srs_limit_error_t& error)
{
    if (error.parsed < slot_command_api::MAX_SRS_PDU_PER_SLOT)
    {
        ++error.parsed;
        return VALID_FAPI_PDU;
    }

    ++error.errors;
    return INVALID_FAPI_PDU;
}

// 4 KiB is large enough for every nv_ipc message the parser tests construct
// (FAPI header + UL_TTI request + SRS PDU < 4 KiB). The real nvIPC pool size
// varies per pool id; these tests never exercise pool-size-dependent behavior,
// so a fixed test value is sufficient.
inline constexpr int k_test_ipc_buf_size = 4096;

int nv_ipc_get_buf_size(const nv_ipc_config_t*, nv_ipc_mempool_id_t)
{
    return k_test_ipc_buf_size;
}

namespace nv
{

void phy_mac_transport::cleanup_nv_ipc(nv_ipc_t*)
{
}

phy_mac_transport::phy_mac_transport(nv_ipc_config_t& config, uint32_t cell_num_in)
    : ipc_(nullptr, &phy_mac_transport::cleanup_nv_ipc),
      config_(config),
      cell_num(cell_num_in)
{
}

int phy_mac_transport::tx_alloc(phy_mac_msg_desc& msg_desc, uint32_t)
{
    msg_desc.msg_buf  = new uint8_t[k_test_ipc_buf_size];
    msg_desc.data_buf = new uint8_t[k_test_ipc_buf_size];
    msg_desc.msg_len  = 0;
    msg_desc.data_len = 0;
    return 0;
}

void phy_mac_transport::tx_release(phy_mac_msg_desc& msg_desc)
{
    delete[] static_cast<uint8_t*>(msg_desc.msg_buf);
    delete[] static_cast<uint8_t*>(msg_desc.data_buf);
    msg_desc.reset();
}

} // namespace nv

namespace
{

uint32_t    g_pdcch_sym_prb_info_call_count{};
uint16_t    g_pdcch_sym_prb_info_bandwidth{};
int32_t     g_pdcch_sym_prb_info_cell_index{};
bool        g_pdcch_sym_prb_info_mmimo_enabled{};
uint16_t    g_pdcch_sym_prb_info_num_dl_dci{};
uint8_t     g_pdcch_sym_prb_info_coreset_n_dci{};
uint32_t    g_pdcch_sym_prb_info_coreset_dci_start_idx{};
uint16_t    g_pdcch_sym_prb_info_first_dci_rnti{};
uint32_t    g_pdcch_sym_prb_info_first_dci_payload_bits{};
std::size_t g_pdcch_sym_prb_info_first_dci_offset{};

} // namespace

namespace scf_5g_fapi
{

void update_prb_sym_list(slot_command_api::slot_info_t&       list,
                         std::size_t                         prb_index,
                         uint8_t                             startSym,
                         uint8_t                             numSym,
                         slot_command_api::channel_type      channel,
                         ru_type                             ru);

namespace test
{

void reset_pdcch_sym_prb_info_stub_state() noexcept
{
    g_pdcch_sym_prb_info_call_count = 0u;
    g_pdcch_sym_prb_info_bandwidth = 0u;
    g_pdcch_sym_prb_info_cell_index = 0;
    g_pdcch_sym_prb_info_mmimo_enabled = false;
    g_pdcch_sym_prb_info_num_dl_dci = 0u;
    g_pdcch_sym_prb_info_coreset_n_dci = 0u;
    g_pdcch_sym_prb_info_coreset_dci_start_idx = 0u;
    g_pdcch_sym_prb_info_first_dci_rnti = 0u;
    g_pdcch_sym_prb_info_first_dci_payload_bits = 0u;
    g_pdcch_sym_prb_info_first_dci_offset = 0u;
}

[[nodiscard]] uint32_t pdcch_sym_prb_info_stub_call_count() noexcept
{
    return g_pdcch_sym_prb_info_call_count;
}

[[nodiscard]] uint16_t pdcch_sym_prb_info_stub_bandwidth() noexcept
{
    return g_pdcch_sym_prb_info_bandwidth;
}

[[nodiscard]] int32_t pdcch_sym_prb_info_stub_cell_index() noexcept
{
    return g_pdcch_sym_prb_info_cell_index;
}

[[nodiscard]] bool pdcch_sym_prb_info_stub_mmimo_enabled() noexcept
{
    return g_pdcch_sym_prb_info_mmimo_enabled;
}

[[nodiscard]] uint16_t pdcch_sym_prb_info_stub_num_dl_dci() noexcept
{
    return g_pdcch_sym_prb_info_num_dl_dci;
}

[[nodiscard]] uint8_t pdcch_sym_prb_info_stub_coreset_n_dci() noexcept
{
    return g_pdcch_sym_prb_info_coreset_n_dci;
}

[[nodiscard]] uint32_t pdcch_sym_prb_info_stub_coreset_dci_start_idx() noexcept
{
    return g_pdcch_sym_prb_info_coreset_dci_start_idx;
}

[[nodiscard]] uint16_t pdcch_sym_prb_info_stub_first_dci_rnti() noexcept
{
    return g_pdcch_sym_prb_info_first_dci_rnti;
}

[[nodiscard]] uint32_t pdcch_sym_prb_info_stub_first_dci_payload_bits() noexcept
{
    return g_pdcch_sym_prb_info_first_dci_payload_bits;
}

[[nodiscard]] std::size_t pdcch_sym_prb_info_stub_first_dci_offset() noexcept
{
    return g_pdcch_sym_prb_info_first_dci_offset;
}

} // namespace test

// PUCCH parser instrumentation.
//
// The parser builds its cuPHY params through pucch::populate_slot_command(), an
// inline helper in scf_5g_fapi_pucch_pdu_parser_common.hpp, so that step cannot be
// intercepted by a link-time stub — tests assert its output on the slot command
// directly.  append_pucch_order_prbs() below is the one remaining out-of-line seam
// on the accept path, and it receives the values the parser only forwards (carrier
// id, UL bandwidth, mMIMO flag, RU, slot detail).  Recording there gives tests both
// a per-PDU call count and those pass-through arguments.
namespace test_support
{
int      pucch_order_append_count = 0;
int32_t  pucch_last_cell_index = -1;
uint16_t pucch_last_cell_stat_prm_idx = 0;
uint16_t pucch_last_ul_bandwidth = 0;
uint16_t pucch_last_prb_size = 0;
uint8_t  pucch_last_format = 0;
uint16_t pucch_last_rnti = 0;
bool     pucch_last_mmimo_enabled = false;
bool     pucch_last_slot_detail_present = false;

void reset_pucch_stub_state() noexcept
{
    pucch_order_append_count = 0;
    pucch_last_cell_index = -1;
    pucch_last_cell_stat_prm_idx = 0;
    pucch_last_ul_bandwidth = 0;
    pucch_last_prb_size = 0;
    pucch_last_format = 0;
    pucch_last_rnti = 0;
    pucch_last_mmimo_enabled = false;
    pucch_last_slot_detail_present = false;
}
} // namespace test_support

/**
 * Stub: always reports BF/PC parameters as valid.
 *
 * The production implementation rejects configurations where
 * !mmimo && numPrg > MAX_NUM_PRGS.  Tests that need to exercise the
 * failure path should set up PDUs with dig_bf_interfaces == 0 and
 * mmimo_enabled == true so the stub result matches the expected outcome.
 */
bool check_bf_pc_params(const int  /*numPrg*/,
                        const int  /*numDigBFI*/,
                        const bool /*mmimo_enabled*/)
{
    return true;
}

/**
 * Stub: records temporary PDCCH-only sym_prb_info bridge invocations without
 * pulling the production fronthaul implementation into the parser unit target.
 */
void update_pdcch_sym_prb_info_for_pdcch_only_validation(
    cell_sub_command& /*cell_cmd*/,
    cuphyPdcchCoresetDynPrm_t& coreset,
    dci_param_list& dci,
    uint16_t bandwidth,
    pm_group* /*pm_grp*/,
    scf_fapi_pdcch_pdu_t& msg,
    std::size_t* fapiDciOffsets,
    nv::phy_config_option& /*config_option*/,
    nv::slot_detail_t* /*slot_detail*/,
    bool mmimo_enabled,
    int32_t cell_index)
{
    ++g_pdcch_sym_prb_info_call_count;
    g_pdcch_sym_prb_info_bandwidth = bandwidth;
    g_pdcch_sym_prb_info_cell_index = cell_index;
    g_pdcch_sym_prb_info_mmimo_enabled = mmimo_enabled;
    g_pdcch_sym_prb_info_num_dl_dci = msg.num_dl_dci;
    g_pdcch_sym_prb_info_coreset_n_dci = coreset.nDci;
    g_pdcch_sym_prb_info_coreset_dci_start_idx = coreset.dciStartIdx;
    g_pdcch_sym_prb_info_first_dci_rnti = dci[coreset.dciStartIdx].rntiCrc;
    g_pdcch_sym_prb_info_first_dci_payload_bits = dci[coreset.dciStartIdx].Npayload;
    g_pdcch_sym_prb_info_first_dci_offset = fapiDciOffsets ? fapiDciOffsets[0] : 0u;
}

// Mirrors the field-population contract of the production PBCH helper for
// parser tests. Keep this stub in sync when PBCH cell/block fields change.
void update_cell_command(slot_command_api::cell_group_command* cell_grp_cmd,
                         slot_command_api::cell_sub_command& cell_cmd,
                         const scf_fapi_ssb_pdu_t& cmd,
                         int32_t cell_index,
                         slot_command_api::slot_indication& slotinfo,
                         const nv::phy_config& cell_params,
                         uint8_t l_max,
                         const uint16_t* lmax_symbols,
                         nv::phy_config_option& config_options,
                         [[maybe_unused]] pm_weight_map_t& pm_map,
                         [[maybe_unused]] nv::slot_detail_t* slot_detail,
                         [[maybe_unused]] bool mmimo_enabled)
{
    auto* grp_params = cell_grp_cmd->get_pbch_params();
    auto& cell_dyn_params = grp_params->pbch_dyn_cell_params[grp_params->ncells];
    auto& block_params = grp_params->pbch_dyn_block_params[grp_params->nSsbBlocks];
    auto& mib_data = grp_params->pbch_dyn_mib_data[grp_params->nSsbBlocks];
    cell_cmd.slot.set_downlink(slotinfo);
    cell_grp_cmd->slot.set_downlink(slotinfo);

    const auto it = std::find(grp_params->cell_index_list.begin(),
                              grp_params->cell_index_list.end(),
                              cell_index);
    const bool new_cell = (it == grp_params->cell_index_list.end());
    if (new_cell)
    {
        grp_params->cell_index_list.push_back(cell_index);
        grp_params->phy_cell_index_list.push_back(cell_cmd.cell);
        ++grp_params->ncells;
    }

    uint16_t ssb_slot = slotinfo.slot_;
    if (config_options.staticSsbSlotNum != -1)
    {
        ssb_slot = static_cast<uint16_t>(config_options.staticSsbSlotNum);
    }

    if (new_cell)
    {
        cell_dyn_params.NID = (config_options.staticSsbPcid != -1)
            ? static_cast<uint16_t>(config_options.staticSsbPcid)
            : cell_params.cell_config_.phy_cell_id;
        cell_dyn_params.nHF = ssb_slot / (5u << cell_params.ssb_config_.sub_c_common);
        cell_dyn_params.Lmax = l_max;
        cell_dyn_params.SFN = config_options.enableTickDynamicSfnSlot
            ? slotinfo.sfn_
            : static_cast<uint16_t>((config_options.staticSsbSFN != -1)
                ? config_options.staticSsbSFN
                : 0);
        cell_dyn_params.k_SSB = cmd.ssb_subcarrier_offset;
        cell_dyn_params.nF =
            cell_params.carrier_config_.dl_grid_size[cell_params.ssb_config_.sub_c_common] *
            CUPHY_N_TONES_PER_PRB;
        cell_dyn_params.slotBufferIdx =
            static_cast<uint16_t>(grp_params->cell_index_list.size() - 1u);
    }

    const auto f0 = detail::calc_ssb_f0(cmd, cell_params.ssb_config_.sub_c_common);
    if (!f0.has_value())
    {
        // Mirror the parser: drop the PDU on unsupported numerology so negative-path
        // tests see a clean rejection instead of half-filled block state.
        if (new_cell)
        {
            grp_params->cell_index_list.pop_back();
            grp_params->phy_cell_index_list.pop_back();
            --grp_params->ncells;
        }
        return;
    }
    block_params.blockIndex = cmd.ssb_block_index;
    block_params.t0 = lmax_symbols[block_params.blockIndex] % OFDM_SYMBOLS_PER_SLOT;
    block_params.f0 = *f0;
    block_params.beta_pss = (cmd.beta_pss == 1u) ? detail::k_beta_pss_3db : 1.0F;
    block_params.beta_sss = 1.0F;
    block_params.cell_index = static_cast<uint16_t>(
        std::distance(grp_params->cell_index_list.begin(),
                      std::find(grp_params->cell_index_list.begin(),
                                grp_params->cell_index_list.end(),
                                cell_index)));
    block_params.enablePrcdBf = false;

    mib_data = cmd.mib_pdu.agg;
    ++grp_params->nSsbBlocks;
}

void append_pucch_order_prbs(cuphyPucchUciPrm_t& uci_info,
                             const uint16_t prb_size,
                             const scf_fapi_rx_beamforming_t& /*pmi_bf_pdu*/,
                             slot_command_api::slot_info_t& sym_prbs,
                             const bool /*bf_enabled*/,
                             const enum ru_type /*ru*/,
                             nv::slot_detail_t* const slot_detail,
                             const bool mmimo_enabled,
                             const int32_t cell_index,
                             const uint16_t ul_bandwidth)
{
    // Recorded before the capacity check so the count reflects parser accepts, not
    // how much room the destination slot_info_t happened to have left.
    ++test_support::pucch_order_append_count;
    test_support::pucch_last_cell_index = cell_index;
    test_support::pucch_last_cell_stat_prm_idx = uci_info.cellPrmStatIdx;
    test_support::pucch_last_ul_bandwidth = ul_bandwidth;
    test_support::pucch_last_prb_size = prb_size;
    test_support::pucch_last_format = uci_info.formatType;
    test_support::pucch_last_rnti = uci_info.rnti;
    test_support::pucch_last_mmimo_enabled = mmimo_enabled;
    test_support::pucch_last_slot_detail_present = (slot_detail != nullptr);

    if (sym_prbs.prbs_size >= MAX_PRB_INFO)
    {
        return;
    }
    const auto index = sym_prbs.prbs_size++;
    sym_prbs.prbs[index] = slot_command_api::prb_info_t(uci_info.startPrb, prb_size);
    update_prb_sym_list(sym_prbs, index, uci_info.startSym, 1, slot_command_api::channel_type::PUCCH, OTHER_MODE);
}

/**
 * Stub: no-op beam-list append. Production walks beams[] and pushes unique
 * beam IDs into prb_info.beams_array. PRACH FH-shell tests with bf_enabled=true
 * would exercise this; current tests set bf_enabled=false so the stub is harmless.
 */
void update_beam_list(slot_command_api::beamid_array_t& /*array*/,
                      std::size_t&                      /*array_size*/,
                      const scf_fapi_rx_beamforming_t&  /*pmi_bf_pdu*/,
                      const bool                        /*mmimo_enabled*/,
                      slot_command_api::prb_info_t&     /*prb_info*/,
                      const int32_t                     /*cell_idx*/)
{
    // intentionally empty — see file-level comment.
}

/**
 * Stub: minimal PRB-symbol registration. Production has extra RU-specific
 * handling; parser tests only need the symbol/channel index populated.
 */
void update_prb_sym_list(slot_command_api::slot_info_t&       list,
                         const std::size_t                    prb_index,
                         const uint8_t                        startSym,
                         const uint8_t                        numSym,
                         const slot_command_api::channel_type channel,
                         const ru_type                        /*ru*/)
{
    for (uint8_t sym = startSym; sym < (startSym + numSym) && sym < OFDM_SYMBOLS_PER_SLOT; ++sym)
    {
        list.symbols[sym][channel].push_back(prb_index);
    }
}

} // namespace scf_5g_fapi

// Declared at global scope in scf_5g_fapi_ul_validate.hpp (gated by
// ENABLE_L2_SLT_RSP). Production impl lives in scf_5g_fapi_ul_validate.cpp;
// tests provide this minimal stub so the per-format counter accounting in
// parse_pucch_pdu_common can be exercised without linking the validator TU.
int validate_pucch_pdu_l1_limits(const scf_fapi_pucch_pdu_t& pdu, nv::pucch_limit_error_t& error)
{
    switch (pdu.format_type)
    {
        case UL_TTI_PUCCH_FORMAT_0:
            if (error.pf0_parsed == 0xFFu) { ++error.pf0_errors; return INVALID_FAPI_PDU; }
            ++error.pf0_parsed;
            break;
        case UL_TTI_PUCCH_FORMAT_1:
            if (error.pf1_parsed == 0xFFu) { ++error.pf1_errors; return INVALID_FAPI_PDU; }
            ++error.pf1_parsed;
            break;
        case UL_TTI_PUCCH_FORMAT_2:
            if (error.pf2_parsed == 0xFFFFu) { ++error.pf2_errors; return INVALID_FAPI_PDU; }
            ++error.pf2_parsed;
            break;
        case UL_TTI_PUCCH_FORMAT_3:
            if (error.pf3_parsed == 0xFFFFu) { ++error.pf3_errors; return INVALID_FAPI_PDU; }
            ++error.pf3_parsed;
            break;
        case UL_TTI_PUCCH_FORMAT_4:
            if (error.pf4_parsed == 0xFFFFu) { ++error.pf4_errors; return INVALID_FAPI_PDU; }
            ++error.pf4_parsed;
            break;
        default:
            break;
    }
    return VALID_FAPI_PDU;


}
