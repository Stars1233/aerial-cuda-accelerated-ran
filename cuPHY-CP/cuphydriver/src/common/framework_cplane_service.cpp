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

#include "framework_cplane_service.hpp"

#include "direct_uplane_modcomp.hpp"
#include "context.hpp"
#include "cell.hpp"
#include "slot_map_dl.hpp"
#include "slot_map_ul.hpp"
#include "nv_phy_mac_transport.hpp"
#include "nv_phy_utils.hpp"
#include "app_config.hpp"
#include "constant.hpp"
#include "time.hpp"
#include "nvlog.hpp"
#include "aerial-fh-driver/api.hpp"
#include "aerial-fh-driver/partial_uplane_slot_info.hpp"
#include "cuphydriver_api.hpp"
#include <aerial/casts/casts.hpp>

#include <oran/cplane_generator.hpp>
#include <oran/dpdk_buf.hpp>
#include <oran/numerology.hpp>
#include <oran/ssb_utils.hpp>
#include <oran/prach_utils.hpp>
#include <oran/pmi_beam_utils.hpp>
#include <oran/fapi_to_cplane.hpp>
#if 0  // enable along with the call_once block in init() to route Oran logs to /tmp/phy.log
    #include <log/rt_log.hpp>
    #include <log/components.hpp>
#endif
#include "ran_common.hpp"

#include <aerial/containers/static_vector.hpp>
#include <oran/bfw_utils.hpp>

#include <rte_ethdev.h>
#include <rte_memory.h>

#include <algorithm>
#include <array>
#include <limits>
#include <span>
#include <string_view>
#include <unordered_map>
#include <fmt/format.h>

#undef  TAG
#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 4) // DRV.API

// --------------------------------------------------------------------------
// Record-store types and pure helpers (used by the Impl record store below)
// --------------------------------------------------------------------------

namespace {

// Prior-slot BFW completions / coefficient headers per cell-ring entry. Bounded
// by the framework's per-slot dynamic-BFW group reserve; static_vector keeps the
// producer recording path allocation-free.
using BfwCompletionVec =
    aerial::static_vector<ran::oran::DynamicBfwCompletionRecord,
                          slot_command_api::MAX_DL_UL_BF_UE_GROUPS>;
using BfwHeaderVec =
    aerial::static_vector<uint8_t*, slot_command_api::MAX_DL_UL_BF_UE_GROUPS>;

// One (sfn,slot) ring entry: the BFW completions recorded by the producer in
// slot N-1, consumed by send_dl/ul_cplane in slot N. Lives in the per-cell ring
// at index (slot % MAX_BFW_COFF_STORE_INDEX); overwritten when a new (sfn,slot)
// reuses the same ring index (see record_bfw_cvi).
struct DirectBfwRecordSlot {
    // 3GPP frame/slot this entry was recorded for. Together they are the ring
    // entry's identity: records_for_slot() returns this entry only when both
    // match the requested (sfn,slot), so a stale slot sharing the ring index is
    // never mistaken for the current one. 0xFFFF = "never written".
    uint16_t          sfn{0xFFFF};
    uint16_t          slot{0xFFFF};
    // True once populated for (sfn,slot); cleared by release_headers() after the
    // consumer (send_dl/ul_cplane) has consumed and released this slot. A false
    // value makes records_for_slot() return nullptr even on an (sfn,slot) match.
    bool              valid{false};
    // Absolute dynamic beam-ID counter base for this slot (from ue.beamIdOffset
    // at record time). Copied into BfwSlotParams.slot_dynamic_beam_id_start so
    // the generator allocates bundle beam IDs from the same base the cuPHY BFW
    // kernel used; 0 means "no dynamic base / unused this slot".
    uint16_t          slot_dynamic_beam_id_start{};
    // One DynamicBfwCompletionRecord per recorded UE group (capacity-bounded,
    // no heap). host_bfws/device_bfws inside each record are NON-OWNING pointers
    // into the cuPHY bfwCoeff_mem_info ring; that backing memory must stay valid
    // until generate_*_packets() has consumed it (chaining holds it longer than
    // the inline copy path). Passed to prepare_dl/ul as a std::span (no copy).
    BfwCompletionVec  completion_records;
    // Unique BFW_COFF_MEM headers for the buffers referenced above (deduped via
    // append_unique_header so each backing buffer is freed exactly once).
    // release_headers() flips each from BUSY->FREE via fh_bfw_coeff_usage_done_fn
    // at consumer scope exit; never released on the stale-overwrite path, since
    // the producer has already reused (and re-marked BUSY) the ring entry.
    BfwHeaderVec      coeff_headers;
};

// Per-transaction C-plane send scratch (reused every slot, no per-slot alloc).
// Match legacy fhproxy limit (ORAN_ALL_SYMBOLS * MAX_AP_PER_SLOT); mMIMO BFW
// chaining can exceed 160 packets/slot/cell (observed 161 in phase-4 logs).
inline constexpr std::size_t MAX_CPLANE_PACKETS_PER_SLOT = MAX_CPLANE_MSGS_PER_SLOT;
inline constexpr std::size_t MAX_CPLANE_CHAIN_MBUFS_PER_SLOT =
    MAX_CPLANE_PACKETS_PER_SLOT * ran::oran::MAX_CHAIN_MBUFS_PER_PACKET;

using ChainMbufVec = aerial::static_vector<rte_mbuf*, MAX_CPLANE_CHAIN_MBUFS_PER_SLOT>;

struct CPlaneSendBuffers {
    aerial::static_vector<rte_mbuf*,       MAX_CPLANE_PACKETS_PER_SLOT> mbufs{};        ///< Packet head mbufs alloc'd from the sender
    aerial::static_vector<ran::oran::MBuf, MAX_CPLANE_PACKETS_PER_SLOT> mbuf_wrappers{}; ///< ORAN wrappers over mbufs[], handed to generate_*_packets
    ChainMbufVec                                                        chain_mbufs{};   ///< ext11 BFW chain tails (mMIMO only); empty on the 4T4R path
};

// Resolve a virtual address to its IOVA, or 0 when not translatable. 0 tells the
// serializer to resolve the address lazily at attach time.
[[nodiscard]] uint64_t direct_bfw_iova_or_zero(const void* ptr) noexcept
{
    if (ptr == nullptr) {
        return 0;
    }
    const rte_iova_t iova = rte_mem_virt2iova(ptr);
    return iova == RTE_BAD_IOVA ? 0 : static_cast<uint64_t>(iova);
}

// Append a coefficient header once; duplicate headers within a slot are skipped
// so release_headers frees each backing buffer exactly once.
void append_unique_header(BfwHeaderVec& headers, uint8_t* header)
{
    if (header == nullptr) {
        return;
    }
    if (std::find(headers.begin(), headers.end(), header) == headers.end()) {
        headers.push_back(header);
    }
}

} // namespace

// --------------------------------------------------------------------------
// Impl definition
// --------------------------------------------------------------------------

struct FrameworkCPlaneService::Impl {
    Config config;
    bool active{false};
    PhyDriverCtx* pdctx{};
    std::size_t max_concurrent_transactions{1};

    std::unique_ptr<ran::oran::CPlaneGenerator> cplane_gen;
    std::unordered_map<uint32_t, std::size_t> cell_index_by_phy_id;
    std::vector<std::vector<aerial_fh::CplaneSender>> dl_senders;
    std::vector<std::vector<aerial_fh::CplaneSender>> ul_senders;

    // Per-cell DL/UL BFW record rings, keyed slot % MAX_BFW_COFF_STORE_INDEX.
    // Outer vector sized once in init() from the runtime cell count; inner array
    // + static_vector storage is allocation-free per slot.
    std::vector<std::array<DirectBfwRecordSlot,
                           slot_command_api::MAX_BFW_COFF_STORE_INDEX>> dl_bfw_records_;
    std::vector<std::array<DirectBfwRecordSlot,
                           slot_command_api::MAX_BFW_COFF_STORE_INDEX>> ul_bfw_records_;

    // Per-transaction send scratch, sized once in init() to
    // max_concurrent_transactions; each in-flight tx_id owns one slot exclusively.
    std::vector<CPlaneSendBuffers> send_buffers_;

    // Record store API (only access path to the rings — DOP/Tell-Don't-Ask).
    /// Validate + convert a producer record and append it to its (sfn,slot) ring
    /// entry. @return 0 on success, -1 on unknown cell, capacity, or bad fields.
    [[nodiscard]] int record_bfw_cvi(uint16_t cell_id, const direct_bfw_cvi_record& record);
    /// Look up the ring entry for (cell,dir,sfn,slot). @return the entry only when
    /// valid and the slot identity matches, else nullptr (stale / never written).
    [[nodiscard]] DirectBfwRecordSlot* records_for_slot(uint16_t cell_id,
                                                        direct_bfw_direction direction,
                                                        uint16_t sfn, uint16_t slot);
    /// Hand the coefficient headers back to the FH (BUSY->FREE) and reset the
    /// entry. Called once the consumer is done with the slot (via the RAII guard).
    void release_headers(direct_bfw_direction direction, DirectBfwRecordSlot& records);
};

// --------------------------------------------------------------------------
// Impl record-store method definitions
// --------------------------------------------------------------------------

int FrameworkCPlaneService::Impl::record_bfw_cvi(const uint16_t cell_id,
                                                 const direct_bfw_cvi_record& record)
{
    auto& table = record.direction == direct_bfw_direction::dl
        ? dl_bfw_records_
        : ul_bfw_records_;

    if (cell_id >= table.size()) {
        return -1;
    }

    auto& slot_records =
        table[cell_id][record.slot % slot_command_api::MAX_BFW_COFF_STORE_INDEX];

    if (!slot_records.valid ||
        slot_records.sfn != record.sfn ||
        slot_records.slot != record.slot) {
        // New (sfn,slot) reusing this ring entry: the producer has already reused
        // the coefficient ring slot and marked its header BUSY, so we only drop
        // the stale metadata here (no header release — that would free the live
        // slot's BFW output before C-plane consumes it).
        slot_records.sfn = record.sfn;
        slot_records.slot = record.slot;
        slot_records.valid = true;
        slot_records.slot_dynamic_beam_id_start = record.slot_dynamic_beam_id_start;
        slot_records.completion_records.clear();
        slot_records.coeff_headers.clear();
    } else if (slot_records.slot_dynamic_beam_id_start !=
               record.slot_dynamic_beam_id_start) {
        // All groups of one slot must share a beam-ID base; a mismatch means the
        // producer disagrees with itself within the slot, so reject the record.
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
            "direct BFW record beam base mismatch cell={} sfn={} slot={} existing={} new={}",
            cell_id, record.sfn, record.slot,
            slot_records.slot_dynamic_beam_id_start, record.slot_dynamic_beam_id_start);
        return -1;
    }

    // A header is "new" only if we haven't seen this exact buffer in the slot yet
    // (one header can back several UE groups).
    const bool new_header =
        record.header != nullptr &&
        std::find(slot_records.coeff_headers.begin(),
                  slot_records.coeff_headers.end(),
                  record.header) == slot_records.coeff_headers.end();

    // Capacity guard: bail before push_back so static_vector never fail-fasts.
    if (slot_records.completion_records.size() ==
            slot_records.completion_records.capacity() ||
        (new_header &&
         slot_records.coeff_headers.size() == slot_records.coeff_headers.capacity())) {
        return -1;
    }

    // Which coefficient buffer the serializer consumes depends only on the
    // chaining mode (BFW chaining is supported for both DL and UL): GPU chaining
    // consumes the device buffer (the host buffer is legitimately null in that
    // mode), while NO/CPU chaining consume the host buffer. Validation below then
    // requires whichever buffer is actually used.
    const auto chaining_mode = pdctx->getFhProxy()->getBfwCPlaneChainingMode();
    const bool use_device_bfws =
        chaining_mode == aerial_fh::BfwCplaneChainingMode::GPU_CHAINING;
    const bool have_usable_bfws =
        use_device_bfws ? (record.device_bfws != nullptr)
                        : (record.host_bfws != nullptr);

    // Reject malformed producer records up front (no usable coeff buffer, zero
    // geometry, wrong IQ width, or a layer span that overflows the shared group).
    // Report the exact offending field + value so a rejected record is diagnosable.
    if (!have_usable_bfws ||
        record.num_prgs == 0 ||
        record.prg_size == 0 ||
        record.n_gnb_ant == 0 ||
        record.bfw_iq_bitwidth != DIRECT_BFW_DYNAMIC_IQ_BITWIDTH ||
        record.group_num_layers == 0 ||
        record.num_layers == 0 ||
        record.num_layers > DIRECT_BFW_MAX_LAYERS ||
        record.layer_base + record.num_layers > record.group_num_layers) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
            "record_bfw_cvi reject cell={} sfn={} slot={} pdu={} rnti={}: "
            "host_present={} device_present={} use_device={} num_prgs={} prg_size={} "
            "n_gnb_ant={} iq_bw={} group_num_layers={} num_layers={} layer_base={}",
            cell_id, record.sfn, record.slot, record.target_pdu_index, record.rnti,
            (record.host_bfws != nullptr) ? 1 : 0,
            (record.device_bfws != nullptr) ? 1 : 0,
            use_device_bfws ? 1 : 0,
            record.num_prgs, record.prg_size,
            record.n_gnb_ant, static_cast<uint16_t>(record.bfw_iq_bitwidth),
            static_cast<uint16_t>(record.group_num_layers),
            static_cast<uint16_t>(record.num_layers),
            static_cast<uint16_t>(record.layer_base));
        return -1;
    }

    ran::oran::DynamicBfwCompletionRecord completion{};
    completion.target_pdu_index = record.target_pdu_index;
    completion.rnti = record.rnti;
    completion.rb_start = record.rb_start;
    completion.rb_size = record.rb_size;
    completion.num_prgs = record.num_prgs;
    completion.prg_size = record.prg_size;
    completion.layer_base = record.layer_base;
    completion.num_layers = record.num_layers;
    completion.group_num_layers = record.group_num_layers;
    completion.host_bfws = record.host_bfws;

    // The device buffer is attached only on the GPU-chaining path; when chaining
    // is active (CPU or GPU) we resolve the IOVA now so the serializer can attach
    // the buffer to the mbuf.
    completion.device_bfws = use_device_bfws ? record.device_bfws : nullptr;
    const bool bfw_chaining =
        chaining_mode != aerial_fh::BfwCplaneChainingMode::NO_CHAINING;
    if (bfw_chaining) {
        if (completion.device_bfws != nullptr) {
            completion.device_bfws_iova = direct_bfw_iova_or_zero(completion.device_bfws);
        } else {
            completion.host_bfws_iova = direct_bfw_iova_or_zero(completion.host_bfws);
        }
    }
    completion.bfw_iq_bitwidth = record.bfw_iq_bitwidth;
    completion.num_antenna_ports = record.n_gnb_ant;

    std::copy_n(record.ue_layer_indices.begin(),
                static_cast<std::size_t>(record.num_layers),
                completion.ue_layer_indices.begin());

    slot_records.completion_records.push_back(completion);
    append_unique_header(slot_records.coeff_headers, record.header);
    return 0;
}

DirectBfwRecordSlot* FrameworkCPlaneService::Impl::records_for_slot(
    const uint16_t cell_id, const direct_bfw_direction direction, const uint16_t sfn, const uint16_t slot)
{
    auto& table = direction == direct_bfw_direction::dl
        ? dl_bfw_records_
        : ul_bfw_records_;

    if (cell_id >= table.size()) {
        return nullptr;
    }

    auto& records = table[cell_id][slot % slot_command_api::MAX_BFW_COFF_STORE_INDEX];
    if (!records.valid || records.sfn != sfn || records.slot != slot) {
        return nullptr;
    }
    return &records;
}

void FrameworkCPlaneService::Impl::release_headers(const direct_bfw_direction direction,
                                                   DirectBfwRecordSlot& records)
{
    slot_command_api::dl_slot_callbacks dl_cb_local{};
    slot_command_api::ul_slot_callbacks ul_cb_local{};
    const bool have_dl_cb =
        direction == direct_bfw_direction::dl && pdctx->getDlCb(dl_cb_local);
    const bool have_ul_cb =
        direction == direct_bfw_direction::ul && pdctx->getUlCb(ul_cb_local);

    for (auto* header : records.coeff_headers) {
        if (header == nullptr) {
            continue;
        }
        if (have_dl_cb && dl_cb_local.fh_bfw_coeff_usage_done_fn != nullptr) {
            dl_cb_local.fh_bfw_coeff_usage_done_fn(
                dl_cb_local.fh_bfw_coeff_usage_done_fn_context, header);
        } else if (have_ul_cb && ul_cb_local.fh_bfw_coeff_usage_done_fn != nullptr) {
            ul_cb_local.fh_bfw_coeff_usage_done_fn(
                ul_cb_local.fh_bfw_coeff_usage_done_fn_context, header);
        }
    }
    records.coeff_headers.clear();
    records.completion_records.clear();
    records.valid = false;
    records.sfn = 0xFFFF;
    records.slot = 0xFFFF;
}

// --------------------------------------------------------------------------
// Anonymous-namespace helpers (extracted from PoC context.cpp)
// --------------------------------------------------------------------------

namespace {

[[nodiscard]] std::array<uint8_t, RTE_ETHER_ADDR_LEN>
resolve_direct_cplane_src_mac(const Cell& cell)
{
    auto src_mac = cell.getSrcEthAddr();
    const bool use_nic_mac = std::all_of(
        src_mac.begin(), src_mac.end(), [](const uint8_t octet) { return octet == 0; });
    if (!use_nic_mac) {
        return src_mac;
    }

    uint16_t port_id{};
    const auto nic_name = cell.getNicName();
    const int port_ret = rte_eth_dev_get_port_by_name(nic_name.c_str(), &port_id);
    if (port_ret != 0) {
        throw std::runtime_error(fmt::format(
            "CPlaneGenerator: cannot resolve NIC {} port for zero source MAC: ret={}",
            nic_name, port_ret));
    }

    rte_ether_addr nic_mac{};
    const int mac_ret = rte_eth_macaddr_get(port_id, &nic_mac);
    if (mac_ret != 0) {
        throw std::runtime_error(fmt::format(
            "CPlaneGenerator: cannot read NIC {} MAC for zero source MAC: ret={}",
            nic_name, mac_ret));
    }

    std::ranges::copy(nic_mac.addr_bytes, src_mac.begin());
    return src_mac;
}

std::string mac_to_string(const std::array<uint8_t, 6>& mac)
{
    return fmt::format("{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                       mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

std::vector<ran::common::AntennaPortId>
to_antenna_port_ids(const std::vector<uint16_t>& ids)
{
    std::vector<ran::common::AntennaPortId> result;
    result.reserve(ids.size());
    for (auto id : ids) {
        result.emplace_back(ran::common::AntennaPortId{id});
    }
    return result;
}

// Massive-MIMO vs regular port-mask policy, driven by the YAML mMIMO flag.
[[nodiscard]] ran::oran::MimoMode to_oran_mimo_mode(const PhyDriverCtx& pdctx) noexcept
{
    return pdctx.getmMIMO_enable() != 0 ? ran::oran::MimoMode::MassiveMimo
                                        : ran::oran::MimoMode::Regular;
}

// Port count for one flow category = its eAxC vector size (mMIMO: layer count;
// 4T4R: physical antennas, since the eAxC vector size equals nTxAnt there).
// Throws on an empty or oversized vector so misconfiguration fails loudly at init.
[[nodiscard]] uint16_t checked_eaxc_count(std::span<const uint16_t> ids,
                                          std::string_view category,
                                          uint32_t cell_id)
{
    if (ids.empty()) {
        throw std::runtime_error(fmt::format(
            "CPlaneGenerator: {} eAxC vector is empty for cell {}", category, cell_id));
    }
    if (ids.size() > std::numeric_limits<uint16_t>::max()) {
        throw std::runtime_error(fmt::format(
            "CPlaneGenerator: {} eAxC count {} exceeds uint16_t for cell {}",
            category, ids.size(), cell_id));
    }
    return static_cast<uint16_t>(ids.size());
}

[[nodiscard]] bool vectors_equal(const std::vector<uint16_t>& lhs,
                                 const std::vector<uint16_t>& rhs) noexcept
{
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

// Map the FAPI DL compression method to the ORAN generator compression enum.
// Returns nullopt for methods the generator does not model (leaves default None).
[[nodiscard]] std::optional<ran::oran::CompressionMethod>
to_oran_dl_compression(uint16_t method) noexcept
{
    using aerial_fh::UserDataCompressionMethod;
    switch (static_cast<UserDataCompressionMethod>(method)) {
    case UserDataCompressionMethod::NO_COMPRESSION:
        return ran::oran::CompressionMethod::None;
    case UserDataCompressionMethod::BLOCK_FLOATING_POINT:
        return ran::oran::CompressionMethod::BlockFloatingPoint;
    case UserDataCompressionMethod::MODULATION_COMPRESSION:
        return ran::oran::CompressionMethod::ModulationCompression;
    default:
        return std::nullopt;
    }
}

// mMIMO-only generator config: DL modcomp, BFW beam-ID ranges/send policy, and
// owned DBT PDU payloads. Also enforces the framework's single-UL-flow-vector
// constraint (PUCCH eAxC == PUSCH eAxC). 4T4R never calls this.
void configure_mmimo(ran::oran::CPlaneGeneratorCellConfig& cc,
                     const Cell& cell,
                     PhyDriverCtx& pdctx)
{
    if (!vectors_equal(cell.geteAxCIdsPucch(), cell.geteAxCIdsPusch())) {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT,
            "CPlaneGenerator mMIMO requires PUCCH eAxC IDs to match PUSCH eAxC IDs "
            "for cell {} (framework carries a single UL flow vector)", cell.getIdx());
        throw std::runtime_error(fmt::format(
            "CPlaneGenerator: mMIMO PUCCH/PUSCH eAxC mismatch for cell {}", cell.getIdx()));
    }

    if (const auto dl_compression = to_oran_dl_compression(cell.getDLCompMeth())) {
        cc.dl_compression_method = *dl_compression;
    }

    FhProxy* fh = pdctx.getFhProxy();
    cc.bfw_config.static_beam_id_start  = fh->getStaticBeamIdStart();
    cc.bfw_config.static_beam_id_end    = fh->getStaticBeamIdEnd();
    cc.bfw_config.dynamic_beam_id_start = fh->getDynamicBeamIdStart();
    cc.bfw_config.dynamic_beam_id_end   = fh->getDynamicBeamIdEnd();
    cc.bfw_config.send_once_per_beam    = (pdctx.get_send_static_bfw_wt_all_cplane() == 0);

    cc.bfw_dbt_config.pdus.clear();
    cc.bfw_dbt_config.pdus.reserve(cell.getBfwDbtPduPayloads().size());
    for (const auto& pdu : cell.getBfwDbtPduPayloads()) {
        cc.bfw_dbt_config.pdus.push_back(ran::oran::BfwDbtPduConfig{.payload = pdu});
    }
}

ran::oran::CPlaneGeneratorConfig build_cplane_gen_config(
        PhyDriverCtx* pdctx,
        const std::vector<Cell*>& cells,
        bool bf_enabled,
        bool precoding_enabled)
{
    const auto mimo_mode = to_oran_mimo_mode(*pdctx);

    ran::oran::CPlaneGeneratorConfig config{};
    config.max_concurrent_transactions = pdctx->getCplaneMaxConcurrentTransactions();
    config.max_dynamic_bfw_groups_per_slot = slot_command_api::MAX_DL_UL_BF_UE_GROUPS;
    config.cells.resize(cells.size());

    for (std::size_t i = 0; i < cells.size(); ++i) {
        Cell* cell = cells[i];
        auto& cc = config.cells[i];

        const auto resolved_src_mac = resolve_direct_cplane_src_mac(*cell);
        cc.src_mac  = mac_to_string(resolved_src_mac);
        cc.dst_mac  = mac_to_string(cell->getDstEthAddr());
        cc.vlan_tci = cell->getVlanTci();

        cc.eaxc.dl    = to_antenna_port_ids(cell->geteAxCIdsPdsch());
        cc.eaxc.ul    = to_antenna_port_ids(cell->geteAxCIdsPusch());
        cc.eaxc.prach = to_antenna_port_ids(cell->geteAxCIdsPrach());
        cc.eaxc.srs   = to_antenna_port_ids(cell->geteAxCIdsSrs());

        cc.mtu            = pdctx->getNicMtu(cell->getNicIndex());
        cc.bandwidth_prbs = cell->getPrbDlBwp();
        const uint8_t mu  = cell->getMu();
        cc.numerology     = ran::oran::from_scs_khz(15u * (1u << mu));

        const auto* prach_stat = cell->getPrachCellStatConfig();
        const uint32_t dl_freq_mhz = static_cast<uint32_t>(cell->getDlFreqAbsAKhz() / 1000);
        const uint32_t ul_freq_mhz = dl_freq_mhz;
        const nv::ssb_case nv_ssb = nv::getSSBCase(dl_freq_mhz, ul_freq_mhz, mu);
        tl::expected<ran::oran::SsbCase, ran::oran::SsbError> ssb_case_result = 
            static_cast<ran::oran::SsbCase>(static_cast<uint8_t>(nv_ssb));
        if (nv_ssb == nv::ssb_case::CASE_UNKNOWN) {
            const auto dl_khz = ran::common::FrequencyKhz{cell->getDlFreqAbsAKhz()};
            ssb_case_result = ran::oran::get_ssb_case(dl_khz, dl_khz, mu);
        }
        if (!ssb_case_result.has_value()) {
            NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT,
                "CPlaneGenerator: invalid SSB case for mu={}", mu);
            throw std::runtime_error(fmt::format(
                "CPlaneGenerator: invalid SSB case for mu={}", mu));
        }
        cc.ssb_case = ssb_case_result.value();

        const auto dl_freq_abs_a_khz = cell->getDlFreqAbsAKhz();
        const auto l_max_result = ran::oran::get_ssb_l_max(
            cc.ssb_case, ran::common::FrequencyKhz{dl_freq_abs_a_khz});
        if (!l_max_result.has_value()) {
            NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT,
                "CPlaneGenerator: invalid L_max for ssb_case={}, freq={}kHz",
                static_cast<int>(cc.ssb_case), dl_freq_abs_a_khz);
            throw std::runtime_error(fmt::format(
                "CPlaneGenerator: invalid L_max for freq={}kHz", dl_freq_abs_a_khz));
        }
        cc.ssb_l_max = l_max_result.value();

        cc.prach_config.mu = mu;
        cc.prach_config.num_prach_fd_occasions =
            prach_stat ? prach_stat->nFdmOccasions : 0;
        for (uint8_t fd = 0; fd < cc.prach_config.num_prach_fd_occasions; ++fd) {
            const auto prach_freq_offset = cell->getPrachFreqOffset(fd);
            if (!prach_freq_offset.has_value()) {
                NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT,
                    "CPlaneGenerator: invalid PRACH freq offset index fd={} num_fd={} phy_idx={}",
                    fd, cc.prach_config.num_prach_fd_occasions, cell->getIdx());
                throw std::runtime_error(fmt::format(
                    "CPlaneGenerator: invalid PRACH freq offset index fd={}", fd));
            }
            cc.prach_config.prach_freq_offsets[fd] = *prach_freq_offset;
        }
        cc.prach_config.sect3_time_offset = cell->getSection3TimeOffset();
        const uint32_t prach_nfft =
            (cell->getPrachSeqLength() == 1) ? 256U : 1024U;
        cc.prach_config.sect3_frame_structure =
            ran::oran::compute_frame_structure(prach_nfft, mu);
        cc.prach_config.sect3_cp_length = 0;

        ran::oran::PmiConfig pmi_config{};
        for (const auto& entry : cell->getPmiEntries()) {
            pmi_config.add_entry(entry.pmi_idx, entry.num_ant_ports);
        }
        cc.pmi_config = std::move(pmi_config);

        // Port counts come from eAxC vector sizes, not getTxAnt(): in mMIMO these
        // are layer counts; for 4T4R the eAxC vector size equals nTxAnt so this is
        // byte-identical to the previous getTxAnt() form (asserted in validation).
        const auto& dl_eaxc    = cell->geteAxCIdsPdsch();
        const auto& pusch_eaxc = cell->geteAxCIdsPusch();
        const auto& prach_eaxc = cell->geteAxCIdsPrach();
        const auto& srs_eaxc   = cell->geteAxCIdsSrs();
        const uint32_t cell_id = cell->getIdx();
        cc.port_mask_config = ran::oran::PortMaskConfig{
            .bf_enabled        = bf_enabled,
            .precoding_enabled = precoding_enabled,
            .mimo_mode         = mimo_mode,
            .num_ul_ports      = checked_eaxc_count(pusch_eaxc, "UL/PUSCH", cell_id),
            .num_dl_ports      = checked_eaxc_count(dl_eaxc, "DL/PDSCH", cell_id),
            .num_prach_ports   = checked_eaxc_count(prach_eaxc, "PRACH", cell_id),
            .num_srs_ports     = checked_eaxc_count(srs_eaxc, "SRS", cell_id),
        };

        if (mimo_mode == ran::oran::MimoMode::MassiveMimo) {
            configure_mmimo(cc, *cell, *pdctx);
        }
    }

    return config;
}

std::optional<std::vector<aerial_fh::CplaneSender>> create_cplane_senders(
    aerial_fh::NicHandle nic_handle, oran_pkt_dir direction,
    int num_queues, std::string_view nic_name)
{
    std::vector<aerial_fh::CplaneSender> out;
    out.reserve(num_queues);
    for (int w = 0; w < num_queues; w++) {
        auto sender = aerial_fh::make_cplane_sender(nic_handle, direction);
        if (!sender) {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                "create_cplane_senders: failed for NIC {} dir={} worker {}",
                nic_name, direction, w);
            return std::nullopt;
        }
        out.push_back(*sender);
    }
    return out;
}

template <typename T>
const T* stored_msg_body(const nv::phy_mac_msg_desc* const msg_desc)
{
    if ((msg_desc == nullptr) || (msg_desc->msg_buf == nullptr)) {
        return nullptr;
    }
    static_assert(alignof(T) == 1, "stored_msg_body requires packed FAPI message body type");
    const auto* const payload =
        static_cast<const uint8_t*>(msg_desc->msg_buf) + sizeof(scf_fapi_header_t);
    return aerial::casts::assume_cast<T>(payload);
}

std::size_t stored_msg_body_len(const nv::phy_mac_msg_desc* const msg_desc)
{
    if (msg_desc == nullptr) {
        return 0;
    }
    if (msg_desc->msg_len < (sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t))) {
        return 0;
    }
    return static_cast<std::size_t>(msg_desc->msg_len - sizeof(scf_fapi_header_t) - sizeof(scf_fapi_body_header_t));
}

[[nodiscard]] Cell* resolve_direct_cplane_cell(PhyDriverCtx* const pdctx, const int msg_cell_id)
{
    if ((pdctx == nullptr) || (msg_cell_id < 0))
    {
        return nullptr;
    }

    Cell* cell = pdctx->getCellByIdx(static_cast<uint32_t>(msg_cell_id));
    if (cell != nullptr)
    {
        return cell;
    }

    return pdctx->getCellByPhyId(static_cast<uint16_t>(msg_cell_id));
}

// Previous slot in the (sfn,slot) ring; wraps slot 0 to the prior frame.
struct SlotPoint {
    uint16_t sfn{};
    uint16_t slot{};
};

[[nodiscard]] SlotPoint previous_slot_point(uint16_t sfn, uint16_t slot,
                                            uint16_t slots_per_frame) noexcept
{
    if (slot != 0) {
        return {.sfn = sfn, .slot = static_cast<uint16_t>(slot - 1)};
    }
    return {.sfn  = static_cast<uint16_t>((sfn + 1024 - 1) % 1024),
            .slot = static_cast<uint16_t>(slots_per_frame - 1)};
}

// Spread mMIMO BFW C-plane TX across the C-plane transmit window
// when the divide-per-cell flag is on. Returns 0 whenever disabled,
// the cell is not part of the current aggregated batch (sm_pos < 0), or the
// batch is empty; also returns 0 on a degenerate T1a window.
[[nodiscard]] uint64_t compute_bfw_cell_stagger_ns(bool     enabled,
                                                   int      sm_pos,
                                                   int      num_cells,
                                                   uint32_t t1a_max_cp_ns,
                                                   uint32_t t1a_min_cp_ns) noexcept
{
    if (!enabled || sm_pos < 0 || num_cells <= 0) {
        return 0U;
    }
    if (t1a_max_cp_ns <= t1a_min_cp_ns) {
        return 0U;
    }
    const uint64_t window_ns = static_cast<uint64_t>(t1a_max_cp_ns - t1a_min_cp_ns);
    return static_cast<uint64_t>(sm_pos) * (window_ns / static_cast<uint64_t>(num_cells));
}

// Releases the consumed prior-slot BFW coefficient headers at scope exit, so a
// send error path cannot leave headers BUSY and poison later slot-command state.
// Templated on the impl type so this anon-namespace helper never names the
// private FrameworkCPlaneService::Impl; CTAD deduces it at the (member) call site.
template <typename ImplT>
class DirectBfwReleaseGuard final {
public:
    DirectBfwReleaseGuard(ImplT& impl,
                          direct_bfw_direction direction,
                          DirectBfwRecordSlot* records) noexcept
        : impl_(impl), direction_(direction), records_(records) {}

    ~DirectBfwReleaseGuard()
    {
        if (records_ != nullptr) {
            impl_.release_headers(direction_, *records_);
        }
    }

    DirectBfwReleaseGuard(const DirectBfwReleaseGuard&) = delete;
    DirectBfwReleaseGuard& operator=(const DirectBfwReleaseGuard&) = delete;
    DirectBfwReleaseGuard(DirectBfwReleaseGuard&&) = delete;
    DirectBfwReleaseGuard& operator=(DirectBfwReleaseGuard&&) = delete;

private:
    ImplT&               impl_;
    direct_bfw_direction direction_{};
    DirectBfwRecordSlot* records_{};
};

// Allocate chain mbufs to back ext11 BFW chaining and wire them to the packet
// heads via one shared cursor (the count is a slot-level upper bound).
[[nodiscard]] int alloc_cplane_chain_mbufs(aerial_fh::CplaneSender& sender,
                                           std::span<ran::oran::MBuf> heads,
                                           ChainMbufVec& chain_mbufs,
                                           std::size_t chain_count,
                                           std::size_t& chain_mbufs_used)
{
    chain_mbufs_used = 0;
    chain_mbufs.resize(chain_count);
    if (sender.alloc(chain_mbufs.data(), static_cast<unsigned>(chain_count)) != 0) {
        chain_mbufs.clear();
        return -1;
    }
    ran::oran::assign_shared_chain_pool(
        heads, std::span<rte_mbuf*>{chain_mbufs.data(), chain_mbufs.size()}, chain_mbufs_used);
    return 0;
}

// Free chain mbufs the serializer never claimed (send() only owns the prefix
// linked into generated packet heads).
void free_unattached_chain_tails(aerial_fh::CplaneSender& sender,
                                 ChainMbufVec& chain_mbufs,
                                 std::size_t chain_mbufs_used)
{
    if (chain_mbufs_used >= chain_mbufs.size()) {
        return;
    }
    sender.free(chain_mbufs.data() + chain_mbufs_used,
                static_cast<unsigned>(chain_mbufs.size() - chain_mbufs_used));
    chain_mbufs.resize(chain_mbufs_used);
}

// Free all head + chain mbufs on the generate error path, where ownership has
// not transferred to TX and ->next links are not trustworthy.
void free_cplane_send_buffers_on_error(aerial_fh::CplaneSender& sender,
                                       CPlaneSendBuffers& bufs)
{
    if (!bufs.mbufs.empty()) {
        sender.free(bufs.mbufs.data(), bufs.mbufs.size());
    }
    if (!bufs.chain_mbufs.empty()) {
        sender.free(bufs.chain_mbufs.data(), bufs.chain_mbufs.size());
        bufs.chain_mbufs.clear();
    }
}

} // anonymous namespace

// --------------------------------------------------------------------------
// FrameworkCPlaneService public interface
// --------------------------------------------------------------------------

FrameworkCPlaneService::FrameworkCPlaneService(Config cfg)
    : impl_(std::make_unique<Impl>())
{
    impl_->config = cfg;
}

FrameworkCPlaneService::~FrameworkCPlaneService() = default;

bool FrameworkCPlaneService::is_active() const
{
    return impl_ && impl_->active;
}

std::optional<std::size_t> FrameworkCPlaneService::resolve_cell_index(uint32_t phy_id) const
{
    if (!impl_) {
        return std::nullopt;
    }
    auto it = impl_->cell_index_by_phy_id.find(phy_id);
    if (it == impl_->cell_index_by_phy_id.end()) {
        return std::nullopt;
    }
    return it->second;
}

int FrameworkCPlaneService::record_bfw_cvi(const uint16_t cell_id,
                                           const direct_bfw_cvi_record& record)
{
    if (!impl_) {
        return -1;
    }
    return impl_->record_bfw_cvi(cell_id, record);
}

int FrameworkCPlaneService::init(PhyDriverCtx* pdctx,
                                 const std::vector<Cell*>& cells,
                                 bool bf_enabled,
                                 bool precoding_enabled)
{
    impl_->pdctx = pdctx;

    NVLOGC_FMT(TAG, "FrameworkCPlaneService::init: enabled={}, bf_enabled={}, precoding_enabled={}",
        impl_->config.enabled, bf_enabled, precoding_enabled);

    if (!impl_->config.enabled) {
        NVLOGC_FMT(TAG, "FrameworkCPlaneService::init: disabled, skipping CPlaneGenerator construction");
        return 0;
    }

    if (cells.empty()) {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT,
            "FrameworkCPlaneService::init: no cells configured");
        return -1;
    }

    NVLOGC_FMT(TAG, "FrameworkCPlaneService::init: building config for {} cell(s)", cells.size());
    for (std::size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        NVLOGC_FMT(TAG, "  cell[{}]: mu={} bw_prb={} tx_ant={} dl_freq_khz={} prach_seq_len={} pmi_entries={}",
            i, c->getMu(), c->getPrbDlBwp(), c->getTxAnt(),
            c->getDlFreqAbsAKhz(), c->getPrachSeqLength(), c->getPmiEntries().size());
    }

    try {
        auto config = build_cplane_gen_config(pdctx,
                                              cells,
                                              bf_enabled,
                                              precoding_enabled);
        impl_->max_concurrent_transactions = config.max_concurrent_transactions;
        impl_->cplane_gen = std::make_unique<ran::oran::CPlaneGenerator>(std::move(config));
        NVLOGC_FMT(TAG,
                   "CPlaneGenerator constructed successfully with {} cell(s), max_concurrent_transactions={}",
                   cells.size(),
                   impl_->max_concurrent_transactions);
    } catch (const std::runtime_error& e) {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT,
            "FrameworkCPlaneService::init: failed to build C-plane generator config: {}",
            e.what());
        impl_->cplane_gen.reset();
        return -1;
    } catch (...) {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT,
            "FrameworkCPlaneService::init: failed to build C-plane generator config: unknown exception");
        impl_->cplane_gen.reset();
        return -1;
    }

#if 0  // enable along with the <log/*.hpp> includes above to route Oran logs
    static std::once_flag fw_log_once;
    std::call_once(fw_log_once, [] {
        framework::log::register_component<ran::oran::Oran>(
            framework::log::LogLevel::Debug);
    });
#endif

    impl_->cell_index_by_phy_id.clear();
    for (std::size_t i = 0; i < cells.size(); ++i) {
        impl_->cell_index_by_phy_id[cells[i]->getIdx()] = i;
    }

    // Size the per-cell BFW record rings once; static_vector storage inside each
    // ring entry is inline, so no per-slot allocation occurs after this point.
    impl_->dl_bfw_records_.resize(cells.size());
    impl_->ul_bfw_records_.resize(cells.size());

    // One send-scratch slot per concurrent transaction id; reused every slot.
    impl_->send_buffers_.resize(impl_->max_concurrent_transactions);

    FhProxy* fhproxy = pdctx->getFhProxy();
    auto nic_list = fhproxy->getNicList();
    const int num_cplane_cells = static_cast<int>(cells.size());

    impl_->dl_senders.resize(nic_list.size());
    impl_->ul_senders.resize(nic_list.size());
    for (std::size_t nic_idx = 0; nic_idx < nic_list.size(); nic_idx++) {
        auto nic_handle = fhproxy->getNic(nic_list[nic_idx]);
        auto dl = create_cplane_senders(nic_handle, DIRECTION_DOWNLINK, num_cplane_cells, nic_list[nic_idx]);
        auto ul = create_cplane_senders(nic_handle, DIRECTION_UPLINK, num_cplane_cells, nic_list[nic_idx]);
        if (!dl || !ul) {
            impl_->cplane_gen.reset();
            return -1;
        }
        impl_->dl_senders[nic_idx] = std::move(*dl);
        impl_->ul_senders[nic_idx] = std::move(*ul);
        NVLOGC_FMT(TAG, "CplaneSenders created for NIC {} (index {}): {} DL and {} UL queues (one per cell)",
            nic_list[nic_idx], nic_idx, num_cplane_cells, num_cplane_cells);
    }

    for (auto* cell : cells) {
        for (std::size_t nic_idx = 0; nic_idx < nic_list.size(); nic_idx++) {
            if (cell->getNicName() == nic_list[nic_idx]) {
                cell->setCplaneNicIndex(static_cast<int>(nic_idx));
                break;
            }
        }
    }

    impl_->active = true;
    return 0;
}

// --------------------------------------------------------------------------
// send_dl_cplane — extracted from PhyDriverCtx::sendDlCPlaneFromStoredMsg
// Thread-safety: send scratch is per-transaction (impl_->send_buffers_[tx_id]);
// the tx_id is the concurrency token, so each in-flight call owns its slot.
// --------------------------------------------------------------------------

int FrameworkCPlaneService::send_dl_cplane(const nv::phy_mac_msg_desc* const dl_tti,
                                           const nv::phy_mac_msg_desc* const ul_dci,
                                           std::size_t transaction_id,
                                           SlotMapDl* slot_map_dl)
{
    const int early_cell_id = (dl_tti != nullptr) ? dl_tti->cell_id : ((ul_dci != nullptr) ? ul_dci->cell_id : -1);
    NVLOGD_FMT(TAG, "[DL_CP_SEND] ENTER cell_id={} dl_tti={} ul_dci={} slot_map={}",
               early_cell_id, dl_tti ? "valid" : "NULL", ul_dci ? "valid" : "NULL",
               slot_map_dl ? "valid" : "NULL");

    if (!impl_->cplane_gen) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[DL_CP_SEND] cell_id={} EARLY_EXIT cplane_gen=null", early_cell_id);
        return -1;
    }
    if (transaction_id >= impl_->max_concurrent_transactions) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[DL_CP_SEND] cell_id={} EARLY_EXIT tx_id={} out of range [0,{})",
                   early_cell_id, transaction_id, impl_->max_concurrent_transactions);
        return -1;
    }
    if (slot_map_dl == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[DL_CP_SEND] cell_id={} EARLY_EXIT slot_map=null", early_cell_id);
        return -1;
    }

    const int cell_id = (dl_tti != nullptr) ? dl_tti->cell_id : ((ul_dci != nullptr) ? ul_dci->cell_id : -1);
    if (cell_id < 0) {
        NVLOGW_FMT(TAG, "[DL_CP_SEND] EARLY_EXIT cell_id<0 ({})", cell_id);
        return 1;
    }
    int sm_pos = -1;
    for (int cell_idx = 0, num_cells = static_cast<int>(slot_map_dl->aggr_cell_list.size());
         cell_idx < num_cells; ++cell_idx) {
        if (slot_map_dl->aggr_cell_list[cell_idx] &&
            slot_map_dl->aggr_cell_list[cell_idx]->getIdx() == static_cast<uint32_t>(cell_id)) {
            sm_pos = cell_idx;
            break;
        }
    }

    if (sm_pos >= 0) {
        slot_map_dl->timings.start_t_dl_cplane[sm_pos] = Time::nowNs();
    }

    // Signal peer-ready on every exit once sm_pos is known (including failures).
    // Prefer U-plane progress over waitPeerUpdateDone hanging on a partial send.
    struct DirectDlPeerReadyGuard {
        SlotMapDl* slot_map{};
        int sm_pos{-1};
        bool signaled{false};

        DirectDlPeerReadyGuard(SlotMapDl* map, int pos) noexcept
            : slot_map(map), sm_pos(pos) {}

        ~DirectDlPeerReadyGuard() { signal(); }

        DirectDlPeerReadyGuard(const DirectDlPeerReadyGuard&) = delete;
        DirectDlPeerReadyGuard& operator=(const DirectDlPeerReadyGuard&) = delete;
        DirectDlPeerReadyGuard(DirectDlPeerReadyGuard&&) = delete;
        DirectDlPeerReadyGuard& operator=(DirectDlPeerReadyGuard&&) = delete;

        void signal()
        {
            if (signaled || slot_map == nullptr || sm_pos < 0) {
                return;
            }
            slot_map->timings.end_t_dl_cplane[sm_pos] = Time::nowNs();
            if (slot_map->aggr_slot_info[sm_pos]) {
                slot_map->aggr_slot_info[sm_pos]->section_id_ready.store(true);
            }
            slot_map->atom_dl_cplane_info_for_uplane_rdy_count.fetch_add(1);
            signaled = true;
        }
    };
    DirectDlPeerReadyGuard peer_ready_guard{slot_map_dl, sm_pos};

    Cell* cell = resolve_direct_cplane_cell(impl_->pdctx, cell_id);
    if (cell == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[DL_CP_SEND] cell_id={} EARLY_EXIT cell=null (resolve failed)", cell_id);
        return -1;
    }

    const auto it = impl_->cell_index_by_phy_id.find(cell->getIdx());
    if (it == impl_->cell_index_by_phy_id.end()) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[DL_CP_SEND] cell_id={} getIdx={} NOT FOUND in cell_index_by_phy_id",
                   cell_id, cell->getIdx());
        return -1;
    }
    const std::size_t cell_index = it->second;

    const bool single_section_mode = (cell->getRUType() == ru_type::SINGLE_SECT_MODE);
    const int64_t tti_ns = static_cast<int64_t>(Cell::getTtiNsFromMu(cell->getMu()));
    const int64_t tai_offset = static_cast<int64_t>(AppConfig::getInstance().getTaiOffset());

    PartialUplaneSlotInfo_t* partial_info = nullptr;
    UplaneConversionParams conv_params{};
    DLOutputBuffer* dlbuf = nullptr;
    if (sm_pos >= 0 && sm_pos < static_cast<int>(slot_map_dl->aggr_dlbuf_list.size())) {
        dlbuf = slot_map_dl->aggr_dlbuf_list[sm_pos];
    }
    if (dlbuf != nullptr) {
        FhProxy* fhproxy = impl_->pdctx->getFhProxy();
        auto slot_ind = slot_map_dl->getSlot3GPP();
        auto slot_oran_ind = slot_command_api::to_oran_slot_format(slot_ind);
        bool commViaCpu = impl_->pdctx->gpuCommEnabledViaCpu();

        t_ns start_tx = slot_map_dl->getSlotRefTs()
            + t_ns(tti_ns * static_cast<int64_t>(cell->getSlotAhead()))
            - t_ns(static_cast<int64_t>(cell->getT1aMaxUpNs()))
            + t_ns(tai_offset);

        const auto& eaxc_ids = cell->geteAxCIdsPdsch();
        auto ret = fhproxy->setupUPlaneGpuComm(cell->getPeerId(), slot_oran_ind,
            dlbuf->getTxMsgContainer(), cell->getDLGridSize(),
            start_tx, commViaCpu, &partial_info,
            &conv_params, eaxc_ids.data(),
            static_cast<uint16_t>(eaxc_ids.size()));
        if (ret == -1) {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                      "setupUPlaneGpuComm failed. ret:{}", ret); 
            return -1; 
        }
    } else {
        NVLOGW_FMT(TAG, "[DL_CP_SEND] cell_id={} dlbuf=NULL — setupUPlaneGpuComm SKIPPED", cell_id);
    }

    const int64_t tx_win_start = slot_map_dl->getSlotRefTs().count()
                                 + (tti_ns * static_cast<int64_t>(cell->getSlotAhead()))
                                 - static_cast<int64_t>(cell->getTcpAdvDlNs())
                                 - static_cast<int64_t>(cell->getT1aMaxUpNs())
                                 + tai_offset;
    const int64_t tx_win_end = tx_win_start
                               + (static_cast<int64_t>(cell->getT1aMaxCpDlNs())
                                  - static_cast<int64_t>(cell->getT1aMinCpDlNs()));
    if ((tx_win_start < 0) || (tx_win_end < 0)) {
        return -1;
    }

    // Reject this slot if we have already missed (or are about to miss) the DL
    // C-plane timing window. Threshold is yaml-driven via sendCPlane_timing_error_th_ns 
    // (default 0 disables).
    //   - tx_win_start already includes tai_offset; 
    //   - On miss, zero partial_info to mirror this function's existing
    //     post-failure cleanup (lines for prepare_dl-failed / no-packets /
    //     convert-failed). Downstream GPU comm Tx tolerates an all-zero
    //     partial_info: host-side dispatcher skips per-symbol copies when
    //     max_pkts == 0 (gpu_comm.cpp), and the pre_prepare / prepare_send
    //     kernels early-exit on num_messages == 0 / tot_pkts == 0
    //     (gpu_comm_doca.cu). This is the same zero-state the FH driver
    //     itself seeds in setupUPlaneGpuComm (peer.cpp).
    const int dl_timing_ret = impl_->pdctx->getFhProxy()->sendCPlane_timingCheck(
        t_ns(tx_win_start - tai_offset), t_ns(0), DIRECTION_DOWNLINK);
    if (dl_timing_ret != SEND_CPLANE_NO_ERROR) {
        if (partial_info) {
            reset_partial_uplane_preserving_modcomp_backing(partial_info);
        }
        const auto dl_slot_ind = slot_map_dl->getSlot3GPP();
        NVLOGW_FMT(TAG,
                   "[DL_CP_SEND] sfn={} slot={} cell_id={} sm_pos={} TIMING SKIP ret={}",
                   dl_slot_ind.sfn_, dl_slot_ind.slot_,
                   cell_id, sm_pos, dl_timing_ret);
        return dl_timing_ret;
    }

    const auto* const dl_tti_msg = (dl_tti != nullptr) ? stored_msg_body<scf_fapi_dl_tti_req_t>(dl_tti) : nullptr;
    const auto* const ul_dci_msg = (ul_dci != nullptr) ? stored_msg_body<scf_fapi_ul_dci_t>(ul_dci) : nullptr;
    const std::size_t dl_tti_len = stored_msg_body_len(dl_tti);
    const std::size_t ul_dci_len = stored_msg_body_len(ul_dci);
    if ((dl_tti_msg == nullptr) && (ul_dci_msg == nullptr)) {
        return 1;
    }

    scf_fapi_dl_tti_req_t empty_dl_tti{};
    scf_fapi_ul_dci_t empty_ul_dci{};
    const auto* dl_msg = (dl_tti_msg != nullptr) ? dl_tti_msg : &empty_dl_tti;
    const auto* ul_msg = (ul_dci_msg != nullptr) ? ul_dci_msg : &empty_ul_dci;

    // mMIMO: bind the prior-slot DL BFW completions for this slot. 4T4R leaves
    // bfw_params default-empty (no dynamic BFW, no chaining), byte-identical.
    ran::oran::BfwSlotParams bfw_params{};
    DirectBfwRecordSlot* bfw_records = nullptr;
    if (impl_->pdctx->getmMIMO_enable()) {
        impl_->cplane_gen->reset_bfw_sent_flags(cell_index);
        const auto dl_slot_ind = slot_map_dl->getSlot3GPP();
        const auto slots_per_frame = static_cast<uint16_t>(10U * (1U << cell->getMu()));
        const auto prev = previous_slot_point(dl_slot_ind.sfn_, dl_slot_ind.slot_, slots_per_frame);
        bfw_records = impl_->records_for_slot(static_cast<uint16_t>(cell_index),
                                              direct_bfw_direction::dl, prev.sfn, prev.slot);
        // Seed from slot-map offset (legacy sendDlCPlaneDirect path); override when
        // prior-slot BFW records carry a non-zero absolute base.
        uint16_t slot_dynamic_beam_id_start =
            static_cast<uint16_t>(slot_map_dl->getDynBeamIdOffset());
        if (bfw_records != nullptr) {
            // Only attach dynamic records if at least one coeff buffer is still
            // BUSY (i.e. its weights are live). All-FREE means the producer's
            // output was already reclaimed, so treat this slot as static-only.
            bool any_busy = false;
            for (auto* header : bfw_records->coeff_headers) {
                if (header != nullptr && *header == slot_command_api::BFW_COFF_MEM_BUSY) {
                    any_busy = true;
                    break;
                }
            }
            if (any_busy) {
                bfw_params.completion_records =
                    std::span<const ran::oran::DynamicBfwCompletionRecord>{
                        bfw_records->completion_records.data(),
                        bfw_records->completion_records.size()};
                if (bfw_records->slot_dynamic_beam_id_start != 0) {
                    slot_dynamic_beam_id_start = bfw_records->slot_dynamic_beam_id_start;
                }
            }
        }
        bfw_params.slot_dynamic_beam_id_start = slot_dynamic_beam_id_start;
        bfw_params.chaining_enabled =
            impl_->pdctx->getFhProxy()->getBfwCPlaneChainingMode() !=
            aerial_fh::BfwCplaneChainingMode::NO_CHAINING;
    }
    // Release the consumed coefficient headers when this scope exits (after send),
    // mirroring the legacy C-plane lifecycle so stale BUSY state cannot persist.
    DirectBfwReleaseGuard bfw_guard{*impl_, direct_bfw_direction::dl, bfw_records};

    const uint64_t bfw_stagger_ns = compute_bfw_cell_stagger_ns(
        impl_->pdctx->getmMIMO_enable()
            && impl_->pdctx->getFhProxy()->getDlcBfwEnableDividePerCell(),
        sm_pos,
        slot_map_dl->getNumCells(),
        cell->getT1aMaxCpDlNs(),
        cell->getT1aMinCpDlNs());

    const ran::oran::OranTxWindows tx_windows{
        .tx_window_start = static_cast<uint64_t>(tx_win_start),
        .tx_window_bfw_start = static_cast<uint64_t>(tx_win_start) + bfw_stagger_ns,
        .tx_window_end = static_cast<uint64_t>(tx_win_end),
    };
    ran::oran::RuConfig ru_config{
        .section_mode = single_section_mode ? ran::oran::RuSectionMode::SingleSection
                                            : ran::oran::RuSectionMode::MultiSection,
    };
    if (impl_->pdctx->getmMIMO_enable()) {
        if (const auto dl_compression = to_oran_dl_compression(cell->getDLCompMeth())) {
            ru_config.dl_compression_method = *dl_compression;
        }
    }
    const auto tx_id = ran::oran::CPlaneTransactionId{transaction_id};

    auto pkt_count = impl_->cplane_gen->prepare_dl(
        tx_id,
        ran::oran::CPlaneDlRequestView{
            .dl_tti = *dl_msg,
            .dl_body_len = dl_tti_len,
            .ul_dci = *ul_msg,
            .ul_dci_body_len = ul_dci_len,
        },
        cell_index,
        tx_windows,
        ru_config,
        bfw_params);
    if (!pkt_count) {
        if (partial_info) {
            reset_partial_uplane_preserving_modcomp_backing(partial_info);
        }
        const auto ec = pkt_count.error();
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[DL_CP_SEND] cell_id={} EARLY_EXIT prepare_dl failed: value={} message={}",
                   cell_id, ec.value(), ec.message());
        return -1;
    }
    if (pkt_count->num_packets == 0) {
        if (partial_info) {
            reset_partial_uplane_preserving_modcomp_backing(partial_info);
        }
        NVLOGW_FMT(TAG, "[DL_CP_SEND] cell_id={} EARLY_EXIT prepare_dl returned no packets", cell_id);
        return 0;
    }

    const int nic_index = cell->getCplaneNicIndex();
    if ((nic_index < 0) || (static_cast<std::size_t>(nic_index) >= impl_->dl_senders.size())) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[DL_CP_SEND] cell_id={} EARLY_EXIT bad nic_index={}", cell_id, nic_index);
        return -1;
    }
    auto& senders = impl_->dl_senders[static_cast<std::size_t>(nic_index)];
    if (senders.empty() || cell_index >= senders.size()) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[DL_CP_SEND] cell_id={} EARLY_EXIT cell_index={} senders.size()={}",
                   cell_id, cell_index, senders.size());
        return -1;
    }
    auto& sender = senders[cell_index];

    // Per-transaction scratch (reused each slot; tx_id is the concurrency token,
    // so this slot is owned exclusively by the calling worker).
    CPlaneSendBuffers& send_bufs = impl_->send_buffers_[transaction_id];
    if (pkt_count->num_packets > send_bufs.mbufs.capacity()) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[DL_CP_SEND] cell_id={} EARLY_EXIT num_packets={} exceeds scratch capacity={}",
                   cell_id, pkt_count->num_packets, send_bufs.mbufs.capacity());
        return -1;
    }
    if (pkt_count->num_chain_mbufs > send_bufs.chain_mbufs.capacity()) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[DL_CP_SEND] cell_id={} EARLY_EXIT num_chain_mbufs={} exceeds scratch capacity={}",
                   cell_id, pkt_count->num_chain_mbufs, send_bufs.chain_mbufs.capacity());
        return -1;
    }

    send_bufs.mbufs.resize(pkt_count->num_packets);
    if (sender.alloc(send_bufs.mbufs.data(), static_cast<unsigned>(pkt_count->num_packets)) != 0) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[DL_CP_SEND] cell_id={} EARLY_EXIT mbuf alloc failed pkt_count={}",
                   cell_id, pkt_count->num_packets);
        return -1;
    }
    send_bufs.mbuf_wrappers.clear();
    for (auto* m : send_bufs.mbufs) {
        static_cast<void>(send_bufs.mbuf_wrappers.emplace_back(m));
    }

    send_bufs.chain_mbufs.clear();
    std::size_t chain_mbufs_used = 0;
    if (pkt_count->num_chain_mbufs > 0 &&
        alloc_cplane_chain_mbufs(sender,
                                 std::span<ran::oran::MBuf>{send_bufs.mbuf_wrappers.data(),
                                                            send_bufs.mbuf_wrappers.size()},
                                 send_bufs.chain_mbufs,
                                 pkt_count->num_chain_mbufs,
                                 chain_mbufs_used) != 0) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[DL_CP_SEND] cell_id={} EARLY_EXIT chain mbuf alloc failed num_packets={}",
                   cell_id, pkt_count->num_packets);
        sender.free(send_bufs.mbufs.data(), send_bufs.mbufs.size());
        return -1;
    }

    auto prepared = impl_->cplane_gen->generate_dl_packets(
        tx_id, std::span<ran::oran::MBuf>{send_bufs.mbuf_wrappers.data(),
                                          send_bufs.mbuf_wrappers.size()});
    if (!prepared) {
        free_cplane_send_buffers_on_error(sender, send_bufs);
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[DL_CP_SEND] cell_id={} EARLY_EXIT generate_dl_packets failed", cell_id);
        return -1;
    }

    free_unattached_chain_tails(sender, send_bufs.chain_mbufs, chain_mbufs_used);

    const auto send_ec = sender.send(send_bufs.mbufs.data(), *prepared);
    if (*prepared < pkt_count->num_packets) {
        sender.free(send_bufs.mbufs.data() + *prepared,
                    static_cast<unsigned>(pkt_count->num_packets - *prepared));
    }
    // send() transferred chain-mbuf ownership to TX; drop our tracking handles.
    send_bufs.chain_mbufs.clear();
    const int send_ret = send_ec ? -send_ec.value() : 0;

    if (send_ret == 0 && partial_info) {
        const auto tx_id = ran::oran::CPlaneTransactionId{transaction_id};
        auto conv_result =
            impl_->cplane_gen->convert_dl_cplane_to_uplane(tx_id, conv_params, *partial_info);
        if (!conv_result) {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                "convert_dl_cplane_to_uplane failed: {}", conv_result.error().message());
            reset_partial_uplane_preserving_modcomp_backing(partial_info);
        } else if (dlbuf != nullptr) {
            const bool dl_modcomp_enabled =
                cell->getDLCompMeth() == static_cast<int>(
                    aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION);
            const auto active_pdsch_antennas =
                static_cast<std::uint16_t>(cell->geteAxCNumPdsch());
            if (active_pdsch_antennas > API_MAX_ANTENNAS) {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                    "[DL_CP_SEND] cell_id={} active_pdsch_antennas={} exceeds API_MAX_ANTENNAS={}",
                    cell_id, active_pdsch_antennas, API_MAX_ANTENNAS);
                reset_partial_uplane_preserving_modcomp_backing(partial_info);
                static_cast<void>(clear_direct_modcomp_config(
                    dl_modcomp_enabled, dlbuf->getModCompressionTempConfig()));
            } else if (const auto modcomp_ec = fill_direct_modcomp_config(
                           dl_modcomp_enabled,
                           *conv_result,
                           active_pdsch_antennas,
                           dlbuf->getModCompressionTempConfig())) {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                    "[DL_CP_SEND] cell_id={} modcomp adapter failed: {}",
                    cell_id, modcomp_ec.message());
                reset_partial_uplane_preserving_modcomp_backing(partial_info);
                static_cast<void>(clear_direct_modcomp_config(
                    dl_modcomp_enabled, dlbuf->getModCompressionTempConfig()));
            }
        }
    } else if (partial_info) {
        reset_partial_uplane_preserving_modcomp_backing(partial_info);
    }

    peer_ready_guard.signal();
    return send_ret;
}

// --------------------------------------------------------------------------
// send_ul_cplane — extracted from PhyDriverCtx::sendUlCPlaneFromStoredMsg
// Thread-safety: send scratch is per-transaction (impl_->send_buffers_[tx_id]);
// the tx_id is the concurrency token, so each in-flight call owns its slot.
// --------------------------------------------------------------------------

int FrameworkCPlaneService::send_ul_cplane(const nv::phy_mac_msg_desc* const ul_tti,
                                           std::size_t transaction_id,
                                           SlotMapUl* slot_map_ul)
{
    const int early_cell_id = (ul_tti != nullptr) ? ul_tti->cell_id : -1;
    NVLOGD_FMT(TAG, "[UL_CP_SEND] ENTER cell_id={} ul_tti={} slot_map={}",
               early_cell_id, ul_tti ? "valid" : "NULL", slot_map_ul ? "valid" : "NULL");

    if ((ul_tti == nullptr) || !impl_->cplane_gen) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[UL_CP_SEND] cell_id={} EARLY_EXIT ul_tti={} cplane_gen={}",
                   early_cell_id, ul_tti ? "valid" : "NULL", impl_->cplane_gen ? "valid" : "NULL");
        return -1;
    }
    if (transaction_id >= impl_->max_concurrent_transactions) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[UL_CP_SEND] cell_id={} EARLY_EXIT tx_id={} out of range [0,{})",
                   early_cell_id, transaction_id, impl_->max_concurrent_transactions);
        return -1;
    }
    if (slot_map_ul == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[UL_CP_SEND] cell_id={} EARLY_EXIT slot_map=null", early_cell_id);
        return -1;
    }
    const auto* const ul_tti_msg = stored_msg_body<scf_fapi_ul_tti_req_t>(ul_tti);
    if (ul_tti_msg == nullptr) {
        NVLOGW_FMT(TAG, "[UL_CP_SEND] cell_id={} EARLY_EXIT ul_tti_msg body=null", early_cell_id);
        return 1;
    }

    const int cell_id = ul_tti->cell_id;
    Cell* cell = resolve_direct_cplane_cell(impl_->pdctx, cell_id);
    if (cell == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[UL_CP_SEND] cell_id={} EARLY_EXIT cell=null (resolve failed)", cell_id);
        return -1;
    }
    const auto it = impl_->cell_index_by_phy_id.find(cell->getIdx());
    if (it == impl_->cell_index_by_phy_id.end()) {
        return -1;
    }
    const std::size_t cell_index = it->second;

    int sm_pos = -1;
    {
        const auto uid = static_cast<uint32_t>(cell_id);
        for (int cell_idx = 0, num_cells = static_cast<int>(slot_map_ul->aggr_cell_list.size());
             cell_idx < num_cells; ++cell_idx) {
            if (slot_map_ul->aggr_cell_list[cell_idx] &&
                slot_map_ul->aggr_cell_list[cell_idx]->getIdx() == uid) {
                sm_pos = cell_idx;
                break;
            }
        }
    }

    const bool single_section_mode = (cell->getRUType() == ru_type::SINGLE_SECT_MODE);
    const int exec_slot_ahead = cell->getSlotAhead() - 1;
    const int64_t tti_ns = static_cast<int64_t>(Cell::getTtiNsFromMu(cell->getMu()));
    const int64_t tai_offset = static_cast<int64_t>(AppConfig::getInstance().getTaiOffset());
    const int64_t tx_win_start = slot_map_ul->getSlotRefTs().count()
                                 + (tti_ns * static_cast<int64_t>(exec_slot_ahead))
                                 - static_cast<int64_t>(cell->getT1aMaxCpUlNs())
                                 + tai_offset;

    // Reject this slot if we have already missed (or are about to miss) the C-plane
    // timing window. Threshold is yaml-driven via
    // sendCPlane_timing_error_th_ns (default 0 disables the check).
    //   - tx_win_start already includes tai_offset; the legacy check compares
    //     against Time::nowNs() (local clock), so subtract tai_offset to align
    //     time domains.
    //   - start_ch_task_time is unused by the legacy body; pass t_ns(0).
    //   - direction selects the legacy log line and return code.
    const int timing_ret = impl_->pdctx->getFhProxy()->sendCPlane_timingCheck(
        t_ns(tx_win_start - tai_offset), t_ns(0), DIRECTION_UPLINK);
    if (timing_ret != SEND_CPLANE_NO_ERROR) {
        const auto ul_slot_ind = slot_map_ul->getSlot3GPP();
        NVLOGW_FMT(TAG,
                   "[UL_CP_SEND] sfn={} slot={} cell_id={} sm_pos={} TIMING SKIP ret={}",
                   ul_slot_ind.sfn_, ul_slot_ind.slot_,
                   cell_id, sm_pos, timing_ret);
        return timing_ret;
    }

    const int64_t tx_win_end = tx_win_start
                               + (static_cast<int64_t>(cell->getT1aMaxCpUlNs())
                                  - static_cast<int64_t>(cell->getT1aMinCpUlNs()));
    if ((tx_win_start < 0) || (tx_win_end < 0)) {
        return -1;
    }

    const uint64_t bfw_stagger_ns = compute_bfw_cell_stagger_ns(
        impl_->pdctx->getmMIMO_enable()
            && impl_->pdctx->getFhProxy()->getUlcBfwEnableDividePerCell(),
        sm_pos,
        slot_map_ul->getNumCells(),
        cell->getT1aMaxCpUlNs(),
        cell->getT1aMinCpUlNs());

    const ran::oran::OranTxWindows tx_windows{
        .tx_window_start = static_cast<uint64_t>(tx_win_start),
        .tx_window_bfw_start = static_cast<uint64_t>(tx_win_start) + bfw_stagger_ns,
        .tx_window_end = static_cast<uint64_t>(tx_win_end),
    };
    const ran::oran::RuConfig ru_config{
        .section_mode = single_section_mode ? ran::oran::RuSectionMode::SingleSection
                                            : ran::oran::RuSectionMode::MultiSection,
    };

    // mMIMO: bind the prior-slot UL BFW completions for this slot. 4T4R leaves
    // bfw_params default-empty (no dynamic BFW, no chaining), byte-identical.
    ran::oran::BfwSlotParams bfw_params{};
    DirectBfwRecordSlot* bfw_records = nullptr;
    if (impl_->pdctx->getmMIMO_enable()) {
        impl_->cplane_gen->reset_bfw_sent_flags(cell_index);
        const auto ul_slot_ind = slot_map_ul->getSlot3GPP();
        const auto slots_per_frame = static_cast<uint16_t>(10U * (1U << cell->getMu()));
        const auto prev = previous_slot_point(ul_slot_ind.sfn_, ul_slot_ind.slot_, slots_per_frame);
        bfw_records = impl_->records_for_slot(static_cast<uint16_t>(cell_index),
                                              direct_bfw_direction::ul, prev.sfn, prev.slot);
        uint16_t slot_dynamic_beam_id_start =
            static_cast<uint16_t>(slot_map_ul->getDynBeamIdOffset());
        if (bfw_records != nullptr) {
            // Only attach dynamic records if at least one coeff buffer is still
            // BUSY (i.e. its weights are live). All-FREE means the producer's
            // output was already reclaimed, so treat this slot as static-only.
            bool any_busy = false;
            for (auto* header : bfw_records->coeff_headers) {
                if (header != nullptr && *header == slot_command_api::BFW_COFF_MEM_BUSY) {
                    any_busy = true;
                    break;
                }
            }
            if (any_busy) {
                bfw_params.completion_records =
                    std::span<const ran::oran::DynamicBfwCompletionRecord>{
                        bfw_records->completion_records.data(),
                        bfw_records->completion_records.size()};
                if (bfw_records->slot_dynamic_beam_id_start != 0) {
                    slot_dynamic_beam_id_start = bfw_records->slot_dynamic_beam_id_start;
                }
            }
        }
        bfw_params.slot_dynamic_beam_id_start = slot_dynamic_beam_id_start;
        bfw_params.chaining_enabled =
            impl_->pdctx->getFhProxy()->getBfwCPlaneChainingMode() !=
            aerial_fh::BfwCplaneChainingMode::NO_CHAINING;
    }
    DirectBfwReleaseGuard bfw_guard{*impl_, direct_bfw_direction::ul, bfw_records};

    const auto tx_id = ran::oran::CPlaneTransactionId{transaction_id};
    if (sm_pos >= 0) {
        slot_map_ul->timings.start_t_ul_cplane[sm_pos] = Time::nowNs();
    }
    auto pkt_count = impl_->cplane_gen->prepare_ul(
        tx_id,
        ran::oran::CPlaneUlRequestView{
            .ul_tti = *ul_tti_msg,
            .body_len = stored_msg_body_len(ul_tti),
        },
        cell_index,
        tx_windows,
        ru_config,
        bfw_params);
    if (!pkt_count) {
        const auto ec = pkt_count.error();
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[UL_CP_SEND] cell_id={} EARLY_EXIT prepare_ul failed: value={} message={}",
                   cell_id, ec.value(), ec.message());
        return -1;
    }
    if (pkt_count->num_packets == 0) {
        NVLOGW_FMT(TAG, "[UL_CP_SEND] cell_id={} EARLY_EXIT prepare_ul returned no packets", cell_id);
        return 0;
    }

    const int nic_index = cell->getCplaneNicIndex();
    if ((nic_index < 0) || (static_cast<std::size_t>(nic_index) >= impl_->ul_senders.size())) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[UL_CP_SEND] cell_id={} EARLY_EXIT bad nic_index={}", cell_id, nic_index);
        return -1;
    }
    auto& senders = impl_->ul_senders[static_cast<std::size_t>(nic_index)];
    if (senders.empty() || cell_index >= senders.size()) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[UL_CP_SEND] cell_id={} EARLY_EXIT cell_index={} senders.size()={}",
                   cell_id, cell_index, senders.size());
        return -1;
    }
    auto& sender = senders[cell_index];

    // Per-transaction scratch (reused each slot; tx_id is the concurrency token).
    CPlaneSendBuffers& send_bufs = impl_->send_buffers_[transaction_id];
    if (pkt_count->num_packets > send_bufs.mbufs.capacity()) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[UL_CP_SEND] cell_id={} EARLY_EXIT num_packets={} exceeds scratch capacity={}",
                   cell_id, pkt_count->num_packets, send_bufs.mbufs.capacity());
        return -1;
    }
    if (pkt_count->num_chain_mbufs > send_bufs.chain_mbufs.capacity()) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[UL_CP_SEND] cell_id={} EARLY_EXIT num_chain_mbufs={} exceeds scratch capacity={}",
                   cell_id, pkt_count->num_chain_mbufs, send_bufs.chain_mbufs.capacity());
        return -1;
    }

    send_bufs.mbufs.resize(pkt_count->num_packets);
    if (sender.alloc(send_bufs.mbufs.data(), static_cast<unsigned>(pkt_count->num_packets)) != 0) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[UL_CP_SEND] cell_id={} EARLY_EXIT mbuf alloc failed pkt_count={}",
                   cell_id, pkt_count->num_packets);
        return -1;
    }
    send_bufs.mbuf_wrappers.clear();
    for (auto* m : send_bufs.mbufs) {
        static_cast<void>(send_bufs.mbuf_wrappers.emplace_back(m));
    }

    send_bufs.chain_mbufs.clear();
    std::size_t chain_mbufs_used = 0;
    if (pkt_count->num_chain_mbufs > 0 &&
        alloc_cplane_chain_mbufs(sender,
                                 std::span<ran::oran::MBuf>{send_bufs.mbuf_wrappers.data(),
                                                            send_bufs.mbuf_wrappers.size()},
                                 send_bufs.chain_mbufs,
                                 pkt_count->num_chain_mbufs,
                                 chain_mbufs_used) != 0) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[UL_CP_SEND] cell_id={} EARLY_EXIT chain mbuf alloc failed num_packets={}",
                   cell_id, pkt_count->num_packets);
        sender.free(send_bufs.mbufs.data(), send_bufs.mbufs.size());
        return -1;
    }

    auto prepared = impl_->cplane_gen->generate_ul_packets(
        tx_id, std::span<ran::oran::MBuf>{send_bufs.mbuf_wrappers.data(),
                                          send_bufs.mbuf_wrappers.size()});
    if (!prepared) {
        free_cplane_send_buffers_on_error(sender, send_bufs);
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "[UL_CP_SEND] cell_id={} EARLY_EXIT generate_ul_packets failed", cell_id);
        return -1;
    }

    free_unattached_chain_tails(sender, send_bufs.chain_mbufs, chain_mbufs_used);

    const auto send_ec = sender.send(send_bufs.mbufs.data(), *prepared);
    if (*prepared < pkt_count->num_packets) {
        sender.free(send_bufs.mbufs.data() + *prepared,
                    static_cast<unsigned>(pkt_count->num_packets - *prepared));
    }
    // send() transferred chain-mbuf ownership to TX; drop our tracking handles.
    send_bufs.chain_mbufs.clear();
    const int send_ret = send_ec ? -send_ec.value() : 0;

    if (sm_pos >= 0) {
        slot_map_ul->timings.end_t_ul_cplane[sm_pos] = Time::nowNs();
    }

    return send_ret;
}

// --------------------------------------------------------------------------
// convert_dl_cplane_to_uplane — extracted from PhyDriverCtx::convertDlCplaneToUplane
// --------------------------------------------------------------------------

int FrameworkCPlaneService::convert_dl_cplane_to_uplane(
    std::size_t transaction_id,
    const UplaneConversionParams& conv_params,
    PartialUplaneSlotInfo_t& partial_info)
{
    if (!impl_->cplane_gen) {
        return -1;
    }
    const auto tx_id = ran::oran::CPlaneTransactionId{transaction_id};
    // convert_dl_cplane_to_uplane returns tl::expected; a missing value is the
    // error case (operator bool / has_value() is true on success).
    auto result = impl_->cplane_gen->convert_dl_cplane_to_uplane(tx_id, conv_params, partial_info);
    if (!result) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
            "convert_dl_cplane_to_uplane failed: {}", result.error().message());
        return -1;
    }
    return 0;
}
