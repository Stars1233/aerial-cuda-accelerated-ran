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

#if !defined(SCF_5G_SLOT_COMMAND_DEBUG_HPP_INCLUDED_)
#define SCF_5G_SLOT_COMMAND_DEBUG_HPP_INCLUDED_

#include <cstdint>

#include "nvlog.h"
#include "nvlog_fmt.hpp"
#include "scf_5g_fapi_tags.hpp"
#include "slot_command/slot_command.hpp"

namespace slot_command_api
{

// ---------------------------------------------------------------------------
// print_pdsch_params
// ---------------------------------------------------------------------------

/**
 * Log PDSCH dynamic parameters from a cell_group_command at debug level.
 *
 * Only enabled when @p Channel is one of the three PDSCH channel types
 * (PDSCH_CSIRS, PDSCH, PDSCH_DMRS), which all share the same pdsch_params
 * payload in cell_group_command.
 *
 * Output is gated at debug log level (fmtlog::DBG) and is a compile-time
 * no-op when NVIPC_FMTLOG_ENABLE is not defined.
 *
 * The output hierarchy mirrors the cuphy driver layout:
 *   Cell-group summary → per-cell → per-UE-group → per-UE → per-CW → per-UE-group DMRS
 *
 * @tparam Channel  Must be channel_type::PDSCH_CSIRS, PDSCH, or PDSCH_DMRS.
 * @param[in] sfn   System frame number at the point of logging.
 * @param[in] slot  Slot number at the point of logging.
 * @param[in] cmd   Cell group command whose pdsch field will be printed.
 */
template<channel_type Channel>
    requires (Channel == channel_type::PDSCH_CSIRS ||
              Channel == channel_type::PDSCH       ||
              Channel == channel_type::PDSCH_DMRS)
void print_pdsch_params(uint16_t sfn, uint16_t slot,
                        const cell_group_command& cmd) noexcept
{
    if (!cmd.pdsch) { return; }
    const pdsch_params& p   = *cmd.pdsch;
    const auto&         grp = p.cell_grp_info;

    NVLOGD_FMT(scf_5g_fapi::detail::k_tag,
               "PDSCH [{}.{}] channel={} nCells={} nUeGrps={} nUes={} nCws={}"
               " nPrecodingMatrices={} ueGrpIdxStart={}",
               sfn, slot,
               static_cast<uint32_t>(Channel),
               grp.nCells, grp.nUeGrps, grp.nUes, grp.nCws,
               grp.nPrecodingMatrices, p.cell_ue_group_idx_start);

    // Per-cell dynamic parameters
    for (uint16_t c = 0u; c < grp.nCells; ++c)
    {
        const auto& cd         = p.cell_dyn_info[c];
        const auto  carrier_id = (static_cast<std::size_t>(c) < p.cell_index_list.size())
                                     ? p.cell_index_list[c] : -1;
        const auto  phy_cid    = (static_cast<std::size_t>(c) < p.phy_cell_index_list.size())
                                     ? p.phy_cell_index_list[c] : -1;
        NVLOGD_FMT(scf_5g_fapi::detail::k_tag,
                   "  Cell[{}] carrierId={} phyCellId={} statIdx={} dynIdx={} slotNum={}"
                   " nCsiRsPrms={} csiRsOffset={} testModel={}"
                   " pdschStartSym={} nPdschSym={} dmrsSymLocBmsk=0x{:04x} nUeGrps={}",
                   c, carrier_id, phy_cid,
                   cd.cellPrmStatIdx, cd.cellPrmDynIdx, cd.slotNum,
                   cd.nCsiRsPrms, cd.csiRsPrmsOffset, cd.testModel,
                   cd.pdschStartSym, cd.nPdschSym, cd.dmrsSymLocBmsk,
                   p.nue_grps_per_cell[c]);
    }

    // Per-UE-group parameters
    for (uint16_t g = 0u; g < grp.nUeGrps; ++g)
    {
        const auto& ug = p.ue_grp_info[g];
        NVLOGD_FMT(scf_5g_fapi::detail::k_tag,
                   "  UeGrp[{}] resourceAlloc={} startPrb={} nPrb={} nUes={}"
                   " pdschStartSym={} nPdschSym={} dmrsSymLocBmsk=0x{:04x}",
                   g, ug.resourceAlloc, ug.startPrb, ug.nPrb, ug.nUes,
                   ug.pdschStartSym, ug.nPdschSym, ug.dmrsSymLocBmsk);
    }

    // Per-UE parameters
    for (uint16_t u = 0u; u < grp.nUes; ++u)
    {
        const auto& ue = p.ue_info[u];
        NVLOGD_FMT(scf_5g_fapi::detail::k_tag,
                   "  UE[{}] rnti=0x{:04x} nCw={} nLayers={} scid={} BWPStart={}"
                   " dmrsScrmId={} dmrsPortBmsk=0x{:03x} dataScramId={} refPoint={}"
                   " nlAbove16={} enablePrcdBf={} pmwPrmIdx={}"
                   " beta_qam={:.4f} beta_dmrs={:.4f}",
                   u, ue.rnti, ue.nCw, ue.nUeLayers, ue.scid, ue.BWPStart,
                   ue.dmrsScrmId, ue.dmrsPortBmsk, ue.dataScramId, ue.refPoint,
                   ue.nlAbove16, ue.enablePrcdBf, ue.pmwPrmIdx,
                   ue.beta_qam, ue.beta_dmrs);
    }

    // Per-codeword parameters
    for (uint16_t w = 0u; w < grp.nCws; ++w)
    {
        const auto& cw = p.ue_cw_info[w];
        NVLOGD_FMT(scf_5g_fapi::detail::k_tag,
                   "  CW[{}] mcsTable={} mcsIdx={} targetCodeRate={} qamModOrder={}"
                   " rv={} tbStartOffset={} tbSize={} n_PRB_LBRM={} maxLayers={} maxQm={}",
                   w, cw.mcsTableIndex, cw.mcsIndex,
                   cw.targetCodeRate, cw.qamModOrder,
                   cw.rv, cw.tbStartOffset, cw.tbSize, cw.n_PRB_LBRM, cw.maxLayers, cw.maxQm);
    }

    // Per-UE-group DMRS parameters
    for (uint16_t g = 0u; g < grp.nUeGrps; ++g)
    {
        const auto& dm = p.ue_dmrs_info[g];
        NVLOGD_FMT(scf_5g_fapi::detail::k_tag,
                   "  DMRS[{}] nDmrsCdmGrpsNoData={}",
                   g, dm.nDmrsCdmGrpsNoData);
    }

    // Precoding matrix index cache: one line per entry.
    // pmw_idx_cache[m] holds the PMI key (carrier_id<<16 | pm_idx) for pm_info[m].
    for (uint32_t m = 0u; m < grp.nPrecodingMatrices; ++m)
    {
        const auto  key   = (m < p.pmw_idx_cache.size()) ? p.pmw_idx_cache[m] : 0u;
        const auto  nPorts = (m < p.pm_info.size()) ? p.pm_info[m].nPorts : 0u;
        NVLOGD_FMT(scf_5g_fapi::detail::k_tag,
                   "  PmW[{}] pmi=0x{:08x} (carrierId={} pmIdx={}) nPorts={}",
                   m, key, key >> 16u, key & 0xFFFFu, nPorts);
    }
}

} // namespace slot_command_api

#endif // SCF_5G_SLOT_COMMAND_DEBUG_HPP_INCLUDED_
