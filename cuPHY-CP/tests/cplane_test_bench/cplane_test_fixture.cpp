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

// Include yamlparser first (brings cuphydriver constant.hpp with constexpr SLOTS_PER_FRAME),
// then undef the symbol before testMAC headers redefine it as a macro.
#include "yamlparser.hpp"
#ifdef SLOTS_PER_FRAME
#undef SLOTS_PER_FRAME
#endif

#include "cplane_test_fixture.hpp"
#include "cplane_test_bench.hpp"
#include "ru_emulator_wrapper.hpp"
#include <cstdio>
#include <cstring>
#include <ctime>
#include <unistd.h>  // access()
#include <rte_eal.h>
#include <rte_errno.h>
#include <rte_lcore.h>
#include <rte_mbuf.h>
#include <rte_mempool.h>

#include "pcap_writer.h"

#include "nvlog_fmt.hpp"

#include "aerial/casts/casts.hpp"
#include "oran/bfw_store.hpp"
#include "oran/bfw_utils.hpp"
#include "oran/cplane_generator.hpp"
#include "oran/cplane_types.hpp"
#include "oran/fapi_to_cplane.hpp"
#include "oran/numerology.hpp"
#include "oran/pmi_beam_utils.hpp"
#include "oran/prach_utils.hpp"
#include "oran/vec_buf.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>
#include <fmt/format.h>

namespace cplane_tb {

constexpr std::size_t kMtuSize = 9000;

// Static member definitions
std::unique_ptr<ran::oran::CPlaneGenerator> CPlaneTestFixture::generator_;
std::unique_ptr<FapiTestMacSource> CPlaneTestFixture::fapi_source_;
void* CPlaneTestFixture::ru_emulator_{nullptr};
std::vector<ran::oran::VecBuf> CPlaneTestFixture::packet_buffers_;
rte_mempool* CPlaneTestFixture::pcap_mbuf_pool_{nullptr};
ran::oran::RuConfig CPlaneTestFixture::ru_config_;
ran::oran::OranTxWindows CPlaneTestFixture::tx_windows_;
bool CPlaneTestFixture::mmimo_enabled_{false};
ran::oran::BfwConfig CPlaneTestFixture::bfw_config_{};

// Reserve for per-transaction dynamic BFW descriptors (= CUPHY max UE groups).
static constexpr std::size_t kMaxDynamicBfwGroupsPerSlot = 72;

// ---------------------------------------------------------------------------
// Helper: parse eAxC IDs from YAML node into a vector of AntennaPortId
// ---------------------------------------------------------------------------
static std::vector<ran::common::AntennaPortId> parse_eaxc_ids(yaml::node node)
{
    std::vector<ran::common::AntennaPortId> ids;
    for (size_t i = 0; i < node.length(); ++i) {
        ids.push_back(ran::common::AntennaPortId{static_cast<uint16_t>(static_cast<int>(node[i]))});
    }
    return ids;
}

// ---------------------------------------------------------------------------
// mMIMO dynamic-BFW helpers (file-local). Bind loaded BFW PDUs to current-slot
// FAPI allocations and produce framework DynamicBfwCompletionRecords.
// ---------------------------------------------------------------------------
namespace {

// Current-slot allocation identity used to bind a BFW PDU. ordinal counts only
// PDSCH (DL) or PUSCH (UL) PDUs, matching the framework's pdu ordinal used for
// dynamic-BFW completion-record matching.
struct PduAlloc {
    std::uint16_t ordinal{};
    std::uint16_t rnti{};
    std::uint16_t rb_start{};
    std::uint16_t rb_size{};
    std::uint8_t  num_layers{};
};

// Collect PDSCH allocations from a DL_TTI.request in PDU iteration order.
std::vector<PduAlloc> collect_pdsch_allocs(const scf_fapi_dl_tti_req_t& dl_tti)
{
    std::vector<PduAlloc> allocs;
    std::size_t offset = 0;
    std::uint16_t ordinal = 0;
    for (std::uint16_t i = 0; i < dl_tti.num_pdus; ++i) {
        const auto& pdu =
            *aerial::casts::assume_cast<scf_fapi_generic_pdu_info_t>(&dl_tti.payload[offset]);
        if (pdu.pdu_type == DL_TTI_PDU_TYPE_PDSCH) {
            const auto& pdsch =
                *aerial::casts::assume_cast<scf_fapi_pdsch_pdu_t>(&pdu.pdu_config[0]);
            // The allocation fields live in the trailing "end" struct, after the
            // variable-length codeword array.
            const auto* end = aerial::casts::assume_cast<scf_fapi_pdsch_pdu_end_t>(
                aerial::casts::assume_cast<std::uint8_t>(&pdsch.codewords[0]) +
                static_cast<std::size_t>(pdsch.num_codewords) * sizeof(scf_fapi_pdsch_codeword_t));
            allocs.push_back(PduAlloc{ordinal, pdsch.rnti, end->rb_start, end->rb_size,
                                      end->num_of_layers});
            ++ordinal;
        }
        offset += pdu.pdu_size;
    }
    return allocs;
}

// Collect PUSCH allocations from an UL_TTI.request in PDU iteration order.
std::vector<PduAlloc> collect_pusch_allocs(const scf_fapi_ul_tti_req_t& ul_tti)
{
    std::vector<PduAlloc> allocs;
    std::size_t offset = 0;
    std::uint16_t ordinal = 0;
    for (std::uint16_t i = 0; i < ul_tti.num_pdus; ++i) {
        const auto& pdu =
            *aerial::casts::assume_cast<scf_fapi_generic_pdu_info_t>(&ul_tti.payload[offset]);
        if (pdu.pdu_type == UL_TTI_PDU_TYPE_PUSCH) {
            const auto& pusch =
                *aerial::casts::assume_cast<scf_fapi_pusch_pdu_t>(&pdu.pdu_config[0]);
            allocs.push_back(PduAlloc{ordinal, pusch.rnti, pusch.rb_start, pusch.rb_size,
                                      pusch.num_of_layers});
            ++ordinal;
        }
        offset += pdu.pdu_size;
    }
    return allocs;
}

// Test if a PDU allocation's PRB range is covered by a BFW PDU's PRB range.
bool bfw_covers_alloc(const ru_emulator_bfw_pdu_t& bfw, const PduAlloc& a)
{
    return a.rb_start >= bfw.rb_start &&
           (static_cast<int>(a.rb_start) + a.rb_size) <=
                   (static_cast<int>(bfw.rb_start) + bfw.rb_size);
}

// Emit completion records for one BFW UE-group. A single BFW buffer is shared by
// all co-scheduled UEs whose PRB range it covers (MU-MIMO): group_num_layers is
// the sum of those UEs' layers, and each UE gets one record into the same buffer
// at a running layer_base. ue_layer_indices map UE-local layers to the UE's
// active-port slots (0..num_layers-1).
void emit_group_records(const ru_emulator_bfw_pdu_t& bfw,
                        const std::vector<PduAlloc>& allocs,
                        std::vector<ran::oran::DynamicBfwCompletionRecord>& out)
{
    std::uint16_t group_num_layers = 0;
    for (const auto& a : allocs) {
        if (bfw_covers_alloc(bfw, a)) {
            group_num_layers = static_cast<std::uint16_t>(group_num_layers + a.num_layers);
        }
    }
    if (group_num_layers == 0) {
        return;
    }

    std::uint8_t layer_base = 0;
    for (const auto& a : allocs) {
        if (!bfw_covers_alloc(bfw, a) || a.num_layers == 0) {
            continue;
        }
        ran::oran::DynamicBfwCompletionRecord rec{};
        rec.target_pdu_index = a.ordinal;
        rec.rnti = a.rnti;
        rec.rb_start = bfw.rb_start;
        rec.rb_size = bfw.rb_size;
        rec.num_prgs = bfw.num_prgs;
        rec.prg_size = bfw.prg_size;
        rec.layer_base = layer_base;
        rec.num_layers = a.num_layers;
        rec.group_num_layers = static_cast<std::uint8_t>(group_num_layers);
        for (std::uint8_t l = 0; l < a.num_layers && l < rec.ue_layer_indices.size(); ++l) {
            rec.ue_layer_indices[l] = l;
        }
        rec.host_bfws = bfw.host_bfws;
        rec.num_antenna_ports = bfw.num_antenna_ports;
        rec.bfw_iq_bitwidth = bfw.bfw_iq_bitwidth;
        out.push_back(rec);
        layer_base = static_cast<std::uint8_t>(layer_base + a.num_layers);
    }
}

// Build per-slot dynamic-BFW completion records. The bench loads pre-parsed BFW
// samples, so every loaded PDU is treated as already complete (no busy/
// eligibility gating). PDUs with invalid metadata are skipped.
std::vector<ran::oran::DynamicBfwCompletionRecord>
build_completion_records(const std::vector<ru_emulator_bfw_pdu_t>& bfws,
                         const std::vector<PduAlloc>& allocs)
{
    std::vector<ran::oran::DynamicBfwCompletionRecord> records;
    for (const auto& bfw : bfws) {
        const bool valid_meta = bfw.host_bfws != nullptr && bfw.num_prgs != 0 &&
                                bfw.prg_size != 0 && bfw.num_antenna_ports != 0 &&
                                bfw.bfw_iq_bitwidth != 0;
        if (valid_meta) {
            emit_group_records(bfw, allocs, records);
        }
    }
    return records;
}

// Write a 16-bit value little-endian (= host order on supported LE targets,
// matching the testMAC-built CONFIG.request DBT layout the framework parses).
static void write_le16(std::vector<std::uint8_t>& buf, std::size_t at, std::uint16_t v)
{
    buf[at] = static_cast<std::uint8_t>(v & 0xFFU);
    buf[at + 1] = static_cast<std::uint8_t>((v >> 8U) & 0xFFU);
}

// Populate mMIMO BFW config: beam-ID ranges + send policy from the controller
// YAML, and static DBT payload from the launch pattern (4-byte header + per-beam
// records, identical to the testMAC CONFIG.request DBT wire layout).
void configure_mmimo_bfw(ran::oran::CPlaneGeneratorCellConfig& cell,
                         yaml::node cuphy_cfg, launch_pattern* lp, int cell_idx)
{
    auto read_u16 = [&](const char* key, std::uint16_t dflt) -> std::uint16_t {
        return cuphy_cfg.has_key(key) ? static_cast<std::uint16_t>(cuphy_cfg[key].as<uint>()) : dflt;
    };
    cell.bfw_config.static_beam_id_start =
        read_u16("static_beam_id_start", cell.bfw_config.static_beam_id_start);
    cell.bfw_config.static_beam_id_end =
        read_u16("static_beam_id_end", cell.bfw_config.static_beam_id_end);
    cell.bfw_config.dynamic_beam_id_start =
        read_u16("dynamic_beam_id_start", cell.bfw_config.dynamic_beam_id_start);
    cell.bfw_config.dynamic_beam_id_end =
        read_u16("dynamic_beam_id_end", cell.bfw_config.dynamic_beam_id_end);
    const bool send_all = cuphy_cfg.has_key("send_static_bfw_wt_all_cplane")
            ? (cuphy_cfg["send_static_bfw_wt_all_cplane"].as<uint>() > 0)
            : true;
    cell.bfw_config.send_once_per_beam = !send_all;

    auto* dbt = (lp != nullptr) ? lp->get_dbt_info(cell_idx) : nullptr;
    if (dbt == nullptr || dbt->num_static_beamIdx == 0 || dbt->num_TRX_beamforming == 0) {
        return;
    }
    const std::uint16_t num_beams = dbt->num_static_beamIdx;
    const std::uint16_t num_txrus = dbt->num_TRX_beamforming;

    std::vector<std::uint8_t> payload(
        4U + static_cast<std::size_t>(num_beams) * (2U + static_cast<std::size_t>(num_txrus) * 4U));
    write_le16(payload, 0, num_beams);
    write_le16(payload, 2, num_txrus);
    std::size_t off = 4;
    for (std::uint16_t b = 0; b < num_beams; ++b) {
        write_le16(payload, off, static_cast<std::uint16_t>(b + 1)); // beam IDs are 1-based
        off += 2;
        for (std::uint16_t t = 0; t < num_txrus; ++t) {
            const auto& w = dbt->dbt_data_buf[static_cast<std::size_t>(b) * num_txrus + t];
            write_le16(payload, off, static_cast<std::uint16_t>(w.re));
            off += 2;
            write_le16(payload, off, static_cast<std::uint16_t>(w.im));
            off += 2;
        }
    }
    cell.bfw_dbt_config.pdus.push_back(ran::oran::BfwDbtPduConfig{std::move(payload)});
}

} // namespace

// ---------------------------------------------------------------------------
// SetUpTestSuite
// ---------------------------------------------------------------------------
void CPlaneTestFixture::SetUpTestSuite()
{
    auto& cfg = TestConfig::instance();
    NVLOGC_FMT(TAG_CPTB_COMMON, "=== C-Plane Test Bench SetUpTestSuite (pattern {}) ===", cfg.pattern_number);

    // -----------------------------------------------------------------------
    // 0. Initialize minimal DPDK EAL (required by CPlaneGenerator for memory allocation)
    // -----------------------------------------------------------------------
    // DPDK EAL is required because CPlaneGenerator's OranFlow uses rte_malloc
    // for packet header templates (via aerial_fh::allocate_memory).
    // Match Fronthaul::eal_init() args to avoid aarch64 tailq issues.
    const int EAL_ARGS = 8;
    const char* eal_argv[EAL_ARGS] = {"cplane_test_bench",
                                       "--file-prefix=cptb",
                                       "-l", "0",
                                       "--main-lcore=0",
                                       "-a", "0000:00:00.0",
                                       "--"};
    int eal_ret = rte_eal_init(EAL_ARGS, const_cast<char**>(eal_argv));
    if (eal_ret < 0) {
        NVLOGE_FMT(TAG_CPTB_COMMON, AERIAL_INVALID_PARAM_EVENT, "rte_eal_init failed (ret={})", eal_ret);
        throw std::runtime_error("rte_eal_init failed");
    }
    NVLOGC_FMT(TAG_CPTB_COMMON, "DPDK EAL initialized");

    // -----------------------------------------------------------------------
    // 1. Create FAPI source from testMAC launch pattern
    // -----------------------------------------------------------------------
    fapi_source_ = std::make_unique<FapiTestMacSource>(cfg.pattern_number, cfg.is_nrsim());

    // -----------------------------------------------------------------------
    // 2. Parse cuphycontroller YAML for cell config -> CPlaneGeneratorConfig
    //
    // Probe for a pattern-specific nrSim cuphycontroller YAML — some sub-90xxx
    // patterns (e.g. 0103) are actually nrSim test vectors that fall outside the
    // 90000-99999 range that TestConfig::is_nrsim() uses. If the nrSim yaml exists,
    // it MUST be used: falling back to cuphycontroller_F08_CG1.yaml will hand
    // CPlaneGenerator the 4T4R eAxC IDs which won't match the pattern-specific RU
    // emulator YAML eAxC_DL — every section then fails find_DL_eAxC_index and the
    // DL slot counters never fire.
    // -----------------------------------------------------------------------
    std::string cuphy_yaml_name;
    {
        const std::string probe_nrsim_cuphy =
            "cuphycontroller_nrSim_SCF_CG1_" + cfg.pattern_number + ".yaml";
        char probe_cuphy_path[MAX_PATH_LEN];
        get_full_path_file(probe_cuphy_path, CONFIG_YAML_FILE_PATH,
                           probe_nrsim_cuphy.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
        if (cfg.is_nrsim() || access(probe_cuphy_path, R_OK) == 0) {
            cuphy_yaml_name = probe_nrsim_cuphy;
        } else {
            cuphy_yaml_name = "cuphycontroller_F08_CG1.yaml";
        }
    }
    char yaml_path[MAX_PATH_LEN];
    get_full_path_file(yaml_path, CONFIG_YAML_FILE_PATH, cuphy_yaml_name.c_str(),
                       CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);

    yaml::file_parser parser(yaml_path);
    yaml::document doc = parser.next_document();
    yaml::node root = doc.root();

    mmimo_enabled_ = (root["cuphydriver_config"]["mMIMO_enable"].as<uint>() > 0);
    NVLOGC_FMT(TAG_CPTB_COMMON, "mMIMO_enable = {}", mmimo_enabled_);

    ran::oran::CPlaneGeneratorConfig gen_config{};
    const int num_cells = 1; // TODO: multi-cell support
    // Captured from the cell config (gen_config is moved into the generator below)
    // to drive the per-call RuConfig modcomp method.
    ran::oran::CompressionMethod dl_comp_method = ran::oran::CompressionMethod::None;

    for (int cell_idx = 0; cell_idx < num_cells; ++cell_idx) {
        yaml::node yaml_cell = root["cuphydriver_config"]["cells"][static_cast<size_t>(cell_idx)];

        ran::oran::CPlaneGeneratorCellConfig cell{};

        // MAC addresses — read from YAML if available, else use defaults
        // The DLC test bench hardcodes these; we do the same for now.
        cell.src_mac = "00:11:22:33:44:00";
        cell.dst_mac = fmt::format("20:04:9B:9E:27:{:02X}", cell_idx);
        cell.vlan_tci = 0;

        // eAxC IDs — map from cuphycontroller channel-based eAxC to framework flow categories
        // DL category: PDSCH eAxC IDs (also covers PDCCH, SSB, CSI-RS in the DL path)
        cell.eaxc.dl = parse_eaxc_ids(yaml_cell["eAxC_id_pdsch"]);
        // UL category: PUSCH eAxC IDs (also covers PUCCH)
        cell.eaxc.ul = parse_eaxc_ids(yaml_cell["eAxC_id_pusch"]);
        // PRACH
        cell.eaxc.prach = parse_eaxc_ids(yaml_cell["eAxC_id_prach"]);
        // SRS
        cell.eaxc.srs = parse_eaxc_ids(yaml_cell["eAxC_id_srs"]);

        // ORAN parameters
        cell.mtu = 8192; // Match DLC test bench NIC MTU
        cell.bandwidth_prbs = 273; // TODO: from config
        cell.numerology = ran::oran::from_scs(ran::oran::SubcarrierSpacing::Scs30Khz);
        cell.ssb_case = ran::oran::SsbCase::CaseC;
        cell.ssb_l_max = 8;

        // Port mask config — per-channel port counts from each eAxC list size.
        // Cannot use make_uniform_port_mask_config(DL size) because some patterns
        // (e.g. nrSim 90045) have asymmetric DL/UL eAxC counts, and CPlaneGenerator
        // validates num_*_ports against cell.eaxc.*.size().
        cell.port_mask_config = ran::oran::PortMaskConfig{
            .bf_enabled        = false,  // overridden by l2_adapter YAML below
            .precoding_enabled = false,  // overridden by l2_adapter YAML below
            .num_ul_ports      = static_cast<std::uint16_t>(cell.eaxc.ul.size()),
            .num_dl_ports      = static_cast<std::uint16_t>(cell.eaxc.dl.size()),
            .num_prach_ports   = static_cast<std::uint16_t>(cell.eaxc.prach.size()),
            .num_srs_ports     = static_cast<std::uint16_t>(cell.eaxc.srs.size()),
        };

        // Read precoding/BF config from L2 adapter YAML (same as DLC test bench).
        // Use same nrSim probe as above — sub-90xxx nrSim patterns must NOT fall
        // through to the F08 L2A yaml.
        {
            std::string l2a_yaml_name;
            const std::string probe_nrsim_l2a = "l2_adapter_config_nrSim_SCF_CG1.yaml";
            char probe_l2a_path[MAX_PATH_LEN];
            get_full_path_file(probe_l2a_path, CONFIG_YAML_FILE_PATH,
                               probe_nrsim_l2a.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
            if (cfg.is_nrsim() || access(probe_l2a_path, R_OK) == 0) {
                l2a_yaml_name = probe_nrsim_l2a;
            } else {
                l2a_yaml_name = "l2_adapter_config_F08_CG1.yaml";
            }
            // Tie L2A selection to the cuphycontroller selection: if we chose the F08
            // cuphy yaml, stick with the F08 L2A (don't mix).
            if (cuphy_yaml_name == "cuphycontroller_F08_CG1.yaml") {
                l2a_yaml_name = "l2_adapter_config_F08_CG1.yaml";
            }
            char l2a_path[MAX_PATH_LEN];
            get_full_path_file(l2a_path, CONFIG_YAML_FILE_PATH, l2a_yaml_name.c_str(),
                               CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
            yaml::file_parser l2a_parser(l2a_path);
            yaml::document l2a_doc = l2a_parser.next_document();
            yaml::node l2a_root = l2a_doc.root();

            if (l2a_root.has_key("enable_precoding")) {
                cell.port_mask_config.precoding_enabled = (l2a_root["enable_precoding"].as<uint>() > 0);
            }
            if (l2a_root.has_key("enable_beam_forming")) {
                cell.port_mask_config.bf_enabled = (l2a_root["enable_beam_forming"].as<uint>() > 0);
            }
            NVLOGC_FMT(TAG_CPTB_COMMON, "Cell {} precoding_enabled={} bf_enabled={} (from {})",
                       cell_idx, cell.port_mask_config.precoding_enabled,
                       cell.port_mask_config.bf_enabled, l2a_yaml_name);
        }

        // Compression method from YAML
        auto comp_meth = static_cast<int>(yaml_cell["dl_iq_data_fmt"]["comp_meth"]);
        cell.dl_compression_method = static_cast<ran::oran::CompressionMethod>(comp_meth);
        dl_comp_method = cell.dl_compression_method;

        // PMI config — populate from launch pattern precoding matrices
        // Maps pmi_idx -> num_ant_ports so CPlaneGenerator knows port counts per PDSCH PDU
        auto* lp = fapi_source_->get_launch_pattern();
        auto matrix_vec = lp->get_precoding_matrix_v(cell_idx);
        for (const auto& matrix : matrix_vec) {
            cell.pmi_config.add_entry(
                static_cast<uint16_t>(matrix.PMidx),
                static_cast<uint8_t>(matrix.numAntPorts));
        }
        NVLOGC_FMT(TAG_CPTB_COMMON, "Cell {} PMI config: {} entries", cell_idx, cell.pmi_config.num_entries());

        // PRACH config — read from launch pattern (same source as testMAC CONFIG.request)
        {
            auto& prach_cfgs = lp->get_prach_configs(cell_idx);
            cell.prach_config.mu = prach_cfgs.prachSubCSpacing;
            cell.prach_config.num_prach_fd_occasions = prach_cfgs.numPrachFdOccasions;
            for (uint8_t fd = 0; fd < prach_cfgs.numPrachFdOccasions && fd < ran::oran::MAX_PRACH_FD_OCCASIONS; ++fd) {
                cell.prach_config.prach_freq_offsets[fd] = prach_cfgs.prachFdOccasions[fd].k1;
            }
            NVLOGC_FMT(TAG_CPTB_COMMON, "Cell {} PRACH config: mu={} fd_occasions={}",
                       cell_idx, cell.prach_config.mu, cell.prach_config.num_prach_fd_occasions);
        }

        // mMIMO: massive-MIMO port-mask policy + per-cell BFW beam-ID ranges and
        // static DBT. 
        if (mmimo_enabled_) {
            cell.port_mask_config.mimo_mode = ran::oran::MimoMode::MassiveMimo;
            configure_mmimo_bfw(cell, root["cuphydriver_config"], lp, cell_idx);
            bfw_config_ = cell.bfw_config;
            NVLOGC_FMT(TAG_CPTB_COMMON,
                       "Cell {} mMIMO BFW: static_beam=[{},{}] dynamic_beam=[{},{}] "
                       "send_once_per_beam={} dbt_pdus={}",
                       cell_idx, cell.bfw_config.static_beam_id_start,
                       cell.bfw_config.static_beam_id_end, cell.bfw_config.dynamic_beam_id_start,
                       cell.bfw_config.dynamic_beam_id_end, cell.bfw_config.send_once_per_beam,
                       cell.bfw_dbt_config.pdus.size());
        }

        gen_config.cells.push_back(std::move(cell));
    }

    // Reserve per-transaction dynamic-BFW descriptor capacity (mMIMO only).
    if (mmimo_enabled_) {
        gen_config.max_dynamic_bfw_groups_per_slot = kMaxDynamicBfwGroupsPerSlot;
    }

    // -----------------------------------------------------------------------
    // 3. Create CPlaneGenerator
    // -----------------------------------------------------------------------
    generator_ = std::make_unique<ran::oran::CPlaneGenerator>(std::move(gen_config));
    NVLOGC_FMT(TAG_CPTB_COMMON, "CPlaneGenerator created ({} cells)", num_cells);

    // -----------------------------------------------------------------------
    // 4. Configure RU and TX windows
    // -----------------------------------------------------------------------
    // Promote modulation compression from the cell config so the framework emits
    // the SE5 (ext4/5) modcomp extension the RU emulator requires for modcomp
    // patterns. Other methods keep the bench's uncompressed DL behavior so the
    // non-modcomp 4T4R/nrSim paths are unchanged.
    ru_config_ = ran::oran::RuConfig{
        .section_mode = ran::oran::RuSectionMode::MultiSection,
        .dl_compression_method =
            (dl_comp_method == ran::oran::CompressionMethod::ModulationCompression)
                ? dl_comp_method
                : ran::oran::CompressionMethod::None,
    };

    // TX windows — use generous test values (not timing-critical in test bench)
    tx_windows_ = ran::oran::OranTxWindows{
        .tx_window_start = 1000,
        .tx_window_bfw_start = 1000,
        .tx_window_end = 500000,
    };

    // -----------------------------------------------------------------------
    // 5. Pre-allocate VecBuf pool
    // -----------------------------------------------------------------------
    constexpr size_t MAX_PACKETS_PER_SLOT = 512;
    packet_buffers_.reserve(MAX_PACKETS_PER_SLOT);
    for (size_t i = 0; i < MAX_PACKETS_PER_SLOT; ++i) {
        packet_buffers_.emplace_back(kMtuSize);
    }

    // -----------------------------------------------------------------------
    // 6. Initialize RU emulator
    // -----------------------------------------------------------------------
    ru_emulator_ = create_ru_emulator();

    // Probe the launch-pattern file to decide between nrSim and F08_1C argv tokens —
    // mirrors FapiTestMacSource so sub-90xxx patterns (e.g. 0103) that actually have
    // launch_pattern_nrSim_<p>.yaml take the nrSim arg path.
    const std::string probe_nrsim_lp = "launch_pattern_nrSim_" + cfg.pattern_number + ".yaml";
    char probe_lp_path[MAX_PATH_LEN];
    get_full_path_file(probe_lp_path, "testVectors/multi-cell/", probe_nrsim_lp.c_str(),
                       CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
    const bool use_nrsim_args = cfg.is_nrsim() || (access(probe_lp_path, R_OK) == 0);

    if (use_nrsim_args) {
        // The pattern-specific nrSim RU yaml is required. The generic config.yaml
        // is 4T4R with dlc_tb=0 which silently disables every RU-emulator counter —
        // falling through to it hides real failures as false passes. Fail hard.
        const std::string nrsim_yaml_name =
            "ru_emulator_config_nrSim_SCF_CG1_" + cfg.pattern_number + ".yaml";
        char yaml_probe[MAX_PATH_LEN];
        get_full_path_file(yaml_probe, "cuPHY-CP/ru-emulator/config/",
                           nrsim_yaml_name.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
        if (access(yaml_probe, R_OK) != 0) {
            NVLOGE_FMT(TAG_CPTB_COMMON, AERIAL_INVALID_PARAM_EVENT,
                       "Missing required nrSim RU emulator config for pattern {}: {} "
                       "(searched in cuPHY-CP/ru-emulator/config/). Generate it with "
                       "test_config_nrSim.sh before running this pattern.",
                       cfg.pattern_number, nrsim_yaml_name);
            throw std::runtime_error(
                "Missing nrSim RU emulator config: " + nrsim_yaml_name);
        }
        const char* argv[5] = {"ru_emulator", "nrSim", cfg.pattern_number.c_str(),
                               "--config", nrsim_yaml_name.c_str()};
        ru_emulator_init(ru_emulator_, 5, const_cast<char**>(argv));
    } else {
        const char* argv[6] = {"ru_emulator", "F08", "1C", cfg.pattern_number.c_str(),
                               "--config", "config.yaml"};
        ru_emulator_init(ru_emulator_, 6, const_cast<char**>(argv));
    }
    NVLOGC_FMT(TAG_CPTB_COMMON, "RU emulator initialized with pattern {}", cfg.pattern_number);

    // -----------------------------------------------------------------------
    // 7. Optional mbuf pool for PCAP capture (pcap_writer lib takes rte_mbuf**)
    // -----------------------------------------------------------------------
    if (cfg.enable_pcap) {
        constexpr unsigned POOL_SIZE = 1024;
        constexpr unsigned MBUF_DATA_SIZE = 9216;  // covers 9000-byte VecBuf capacity + headroom
        pcap_mbuf_pool_ = rte_pktmbuf_pool_create("cptb_pcap_pool", POOL_SIZE,
                                                   0 /*cache*/, 0 /*priv*/,
                                                   MBUF_DATA_SIZE, SOCKET_ID_ANY);
        if (pcap_mbuf_pool_ == nullptr) {
            NVLOGE_FMT(TAG_CPTB_COMMON, AERIAL_INVALID_PARAM_EVENT,
                "PCAP mbuf pool allocation failed: {}", rte_strerror(rte_errno));
            throw std::runtime_error("PCAP mbuf pool allocation failed");
        }
        NVLOGC_FMT(TAG_CPTB_COMMON, "PCAP mbuf pool created (size={}, data={})",
                   POOL_SIZE, MBUF_DATA_SIZE);
    }

    // Note: PDCCH port count is determined by PMI config (numAntPorts).
    // With correct PMI populated, framework sends PDCCH on all configured ports.

    NVLOGC_FMT(TAG_CPTB_COMMON, "=== SetUpTestSuite complete ({} slots to process) ===",
               fapi_source_->get_total_slots());
}

// ---------------------------------------------------------------------------
// TearDownTestSuite
// ---------------------------------------------------------------------------
void CPlaneTestFixture::TearDownTestSuite()
{
    auto& cfg = TestConfig::instance();
    NVLOGC_FMT(TAG_CPTB_COMMON, "=== C-Plane Test Bench TearDownTestSuite ===");

    if (cfg.verify_cplane && ru_emulator_) {
        ru_emulator_finalize(ru_emulator_);

        const int cell_idx = 0;

        // Check slot counters against launch pattern expectations
        ru_emulator_total_slot_counters_t counters{};
        ru_emulator_get_total_slt_counters(ru_emulator_, cell_idx, counters);

        auto* lp = fapi_source_->get_launch_pattern();
        const auto& expected = lp->get_expected_values().at(cell_idx);

        NVLOGC_FMT(TAG_CPTB_COMMON, "Slot counters (expected:completed) [completed counts only error-free slots]:");
        NVLOGC_FMT(TAG_CPTB_COMMON, "  [DL] PDSCH    {}:{}", expected.lp_slots[channel_type_t::PDSCH].load(), counters.pdsch);
        NVLOGC_FMT(TAG_CPTB_COMMON, "  [DL] PDCCH_DL {}:{}", expected.lp_slots[channel_type_t::PDCCH_DL].load(), counters.pdcch_dl);
        NVLOGC_FMT(TAG_CPTB_COMMON, "  [DL] PDCCH_UL {}:{}", expected.lp_slots[channel_type_t::PDCCH_UL].load(), counters.pdcch_ul);
        NVLOGC_FMT(TAG_CPTB_COMMON, "  [DL] PBCH     {}:{}", expected.lp_slots[channel_type_t::PBCH].load(), counters.pbch);
        NVLOGC_FMT(TAG_CPTB_COMMON, "  [DL] CSI_RS   {}:{}", expected.lp_slots[channel_type_t::CSI_RS].load(), counters.csi_rs);
        NVLOGC_FMT(TAG_CPTB_COMMON, "  [DL] BFW_DL   {}:{}", expected.lp_slots[channel_type_t::BFW_DL].load(), counters.bfw_dl);
        NVLOGC_FMT(TAG_CPTB_COMMON, "  [UL] PUSCH    {}:{}", expected.lp_slots[channel_type_t::PUSCH].load(), counters.pusch);
        NVLOGC_FMT(TAG_CPTB_COMMON, "  [UL] PUCCH    {}:{}", expected.lp_slots[channel_type_t::PUCCH].load(), counters.pucch);
        NVLOGC_FMT(TAG_CPTB_COMMON, "  [UL] PRACH    {}:{}", expected.lp_slots[channel_type_t::PRACH].load(), counters.prach);
        NVLOGC_FMT(TAG_CPTB_COMMON, "  [UL] SRS      {}:{}", expected.lp_slots[channel_type_t::SRS].load(), counters.srs);
        NVLOGC_FMT(TAG_CPTB_COMMON, "  [UL] BFW_UL   {}:{}", expected.lp_slots[channel_type_t::BFW_UL].load(), counters.bfw_ul);

        // completed now counts only error-free slots, so a channel that suffered
        // any section mismatch (modcomp SE4/SE5 or otherwise) shows completed < expected.
        EXPECT_EQ(expected.lp_slots[channel_type_t::PDSCH].load(), counters.pdsch);
        EXPECT_EQ(expected.lp_slots[channel_type_t::PDCCH_DL].load(), counters.pdcch_dl);
        EXPECT_EQ(expected.lp_slots[channel_type_t::PDCCH_UL].load(), counters.pdcch_ul);
        EXPECT_EQ(expected.lp_slots[channel_type_t::PBCH].load(), counters.pbch);
        EXPECT_EQ(expected.lp_slots[channel_type_t::CSI_RS].load(), counters.csi_rs);
        EXPECT_EQ(expected.lp_slots[channel_type_t::BFW_DL].load(), counters.bfw_dl);

        // Check for C-Plane section errors (DL side only — UL section errors
        // surface via separate channel counters upstream).
        uint64_t err_sections = 0xFFF;
        ru_emulator_get_cplane_err_sections(ru_emulator_, cell_idx, err_sections);
        EXPECT_EQ(err_sections, 0u) << "RU emulator detected " << err_sections << " C-Plane section errors";

        // Sanity check: at least some C-Plane traffic must have reached the RU
        // emulator. Sum DL + UL so UL-only launch patterns (e.g. nrSim 90041)
        // also clear the assertion.
        uint64_t dl_sections = 0;
        uint64_t ul_sections = 0;
        ru_emulator_get_cplane_dl_sections(ru_emulator_, cell_idx, dl_sections);
        ru_emulator_get_cplane_ul_sections(ru_emulator_, cell_idx, ul_sections);
        EXPECT_GT(dl_sections + ul_sections, 0u) << "No C-Plane sections were processed";
        NVLOGC_FMT(TAG_CPTB_COMMON, "C-Plane sections: DL={} UL={} (errors: {})",
                   dl_sections, ul_sections, err_sections);
    }

    // Cleanup
    if (ru_emulator_) {
        destroy_ru_emulator(ru_emulator_);
        ru_emulator_ = nullptr;
    }
    generator_.reset();
    fapi_source_.reset();

    if (pcap_mbuf_pool_ != nullptr) {
        rte_mempool_free(pcap_mbuf_pool_);
        pcap_mbuf_pool_ = nullptr;
    }
    packet_buffers_.clear();

    // Clean up DPDK EAL so shared memory doesn't persist across runs
    rte_eal_cleanup();

    NVLOGC_FMT(TAG_CPTB_COMMON, "=== TearDownTestSuite complete ===");
}

// ---------------------------------------------------------------------------
// RunSlotDl — core per-slot test logic
// ---------------------------------------------------------------------------
void CPlaneTestFixture::RunSlotDl(size_t slot_index, size_t cell_index)
{
    auto slot_msgs = fapi_source_->get_slot(slot_index, cell_index);

    if (!slot_msgs.has_dl) {
        return; // Nothing to do for this slot
    }

    // Phase 1: prepare DL
    scf_fapi_ul_dci_t empty_ul_dci{};
    const auto& dl_req = *slot_msgs.dl_tti_req;
    const auto& ul_dci = slot_msgs.ul_dci_req ? *slot_msgs.ul_dci_req : empty_ul_dci;

    const size_t dl_body_len = slot_msgs.dl_body_len;
    const size_t ul_dci_body_len = slot_msgs.ul_dci_body_len;

    // mMIMO: bind loaded DL BFW samples to this slot's PDSCH PDUs and feed them
    // as previous-slot dynamic-BFW completions. The BFW for slot N is delivered
    // at N-1; in this single-threaded replay we build it inline before prepare.
    // Buffers (host_bfws) alias RU-emulator memory and outlive generate.
    ran::oran::BfwSlotParams bfw_params{};
    std::vector<ran::oran::DynamicBfwCompletionRecord> bfw_records;
    if (mmimo_enabled_) {
        std::vector<ru_emulator_bfw_pdu_t> bfw_pdus;
        ru_emulator_construct_bfw_dl(ru_emulator_, slot_msgs.sfn, slot_msgs.slot,
                                     static_cast<int>(cell_index), bfw_pdus);
        bfw_records = build_completion_records(bfw_pdus, collect_pdsch_allocs(dl_req));
        bfw_params.completion_records = bfw_records;
        bfw_params.slot_dynamic_beam_id_start = bfw_config_.dynamic_beam_id_start;
        bfw_params.chaining_enabled = false; // host inline path (no NIC chaining in the bench)
    }

    const auto tx_id = ran::oran::CPlaneTransactionId{0};
    auto pkt_count = generator_->prepare_dl(tx_id,
                                            {dl_req, dl_body_len, ul_dci, ul_dci_body_len},
                                            cell_index,
                                            tx_windows_,
                                            ru_config_,
                                            bfw_params);
    if (!pkt_count.has_value()) {
        NVLOGE_FMT(TAG_CPTB_COMMON, AERIAL_INVALID_PARAM_EVENT,
                   "prepare_dl failed for slot {}: {}", slot_index, pkt_count.error().message());
        FAIL() << "prepare_dl failed: " << pkt_count.error().message();
        return;
    }

    const auto num_packets = pkt_count->num_packets;
    if (num_packets == 0) {
        return; // No packets for this slot (e.g., empty DL)
    }

    // Ensure we have enough buffers
    while (packet_buffers_.size() < num_packets) {
        packet_buffers_.emplace_back(kMtuSize);
    }

    // Phase 2: generate DL packets into VecBuf
    auto gen_result = generator_->generate_dl_packets(
        tx_id,
        std::span<ran::oran::VecBuf>{packet_buffers_.data(), num_packets});

    if (!gen_result.has_value()) {
        NVLOGE_FMT(TAG_CPTB_COMMON, AERIAL_INVALID_PARAM_EVENT,
                   "generate_dl_packets failed for slot {}: {}", slot_index, gen_result.error().message());
        FAIL() << "generate_dl_packets failed: " << gen_result.error().message();
        return;
    }

    NVLOGD_FMT(TAG_CPTB_COMMON, "Slot {} (SFN={}.SLT={}): {} packets generated",
               slot_index, slot_msgs.sfn, slot_msgs.slot, *gen_result);

    // Phase 3: feed each packet to RU emulator for verification
    for (size_t i = 0; i < *gen_result; ++i) {
        auto& buf = packet_buffers_[i];

        if (TestConfig::instance().verify_cplane) {
            ru_emulator_verify_dl_cplane_content(
                ru_emulator_, buf.data(), buf.size(), static_cast<int>(cell_index));
        }
    }

    // Phase 4: optional PCAP capture (same pcap_writer library as DLC test bench)
    capture_packets(packet_buffers_.data(), *gen_result);
}

// ---------------------------------------------------------------------------
// capture_packets — bridge VecBuf output into rte_mbufs and hand to pcap_writer.
// Mirrors DLC test bench's MockPeer::capture_packets pattern so both benches share
// the same pcap_writer library entry points.
// ---------------------------------------------------------------------------
void CPlaneTestFixture::capture_packets(const ran::oran::VecBuf* bufs, size_t num_packets)
{
    if (!TestConfig::instance().enable_pcap || num_packets == 0 || pcap_mbuf_pool_ == nullptr) {
        return;
    }

    static bool first_time = true;
    const std::string pcap_path = "/tmp/" + TestConfig::instance().pcap_file_name;

    std::vector<rte_mbuf*> mbufs(num_packets, nullptr);
    size_t allocated = 0;
    for (; allocated < num_packets; ++allocated) {
        rte_mbuf* m = rte_pktmbuf_alloc(pcap_mbuf_pool_);
        if (m == nullptr) {
            NVLOGW_FMT(TAG_CPTB_COMMON, "rte_pktmbuf_alloc failed at packet {} of {}", allocated, num_packets);
            break;
        }
        const auto& src = bufs[allocated];
        const auto len = static_cast<uint16_t>(src.size());
        uint8_t* dst = reinterpret_cast<uint8_t*>(rte_pktmbuf_append(m, len));
        if (dst == nullptr) {
            NVLOGW_FMT(TAG_CPTB_COMMON, "rte_pktmbuf_append failed for packet {} (size {})", allocated, len);
            rte_pktmbuf_free(m);
            break;
        }
        std::memcpy(dst, src.data(), len);
        mbufs[allocated] = m;
    }

    if (allocated > 0) {
        const auto count = static_cast<uint32_t>(allocated);
        const int rc = first_time
            ? pcap_write_mbufs(pcap_path.c_str(), mbufs.data(), count, false /*use_timestamps*/)
            : pcap_append_mbufs(pcap_path.c_str(), mbufs.data(), count, false /*use_timestamps*/);
        first_time = false;
        if (rc != 0) {
            NVLOGW_FMT(TAG_CPTB_COMMON, "pcap write/append returned {} for {} packets", rc, allocated);
        }
    }

    for (size_t i = 0; i < allocated; ++i) {
        rte_pktmbuf_free(mbufs[i]);
    }
}

// ---------------------------------------------------------------------------
// Test: run all DL slots through the pipeline
// ---------------------------------------------------------------------------
TEST_F(CPlaneTestFixture, AllSlotsDlCplane)
{
    const size_t total_slots = fapi_source_->get_total_slots();
    NVLOGC_FMT(TAG_CPTB_COMMON, "Running {} DL slots through CPlaneGenerator -> RU Emulator", total_slots);

    for (size_t slot = 0; slot < total_slots; ++slot) {
        SCOPED_TRACE(fmt::format("slot_index={} SFN={} SLT={}", slot, slot / 20, slot % 20));
        RunSlotDl(slot, 0);
    }
}

// ---------------------------------------------------------------------------
// RunSlotUl — core per-slot UL test logic
// ---------------------------------------------------------------------------
void CPlaneTestFixture::RunSlotUl(size_t slot_index, size_t cell_index)
{
    auto slot_msgs = fapi_source_->get_slot(slot_index, cell_index);

    if (!slot_msgs.has_ul) {
        return; // No UL for this slot
    }

    // Phase 1: prepare UL
    const auto& ul_req = *slot_msgs.ul_tti_req;
    const size_t ul_body_len = slot_msgs.ul_body_len;

    // mMIMO: bind loaded UL BFW samples to this slot's PUSCH PDUs (see RunSlotDl).
    ran::oran::BfwSlotParams bfw_params{};
    std::vector<ran::oran::DynamicBfwCompletionRecord> bfw_records;
    if (mmimo_enabled_) {
        std::vector<ru_emulator_bfw_pdu_t> bfw_pdus;
        ru_emulator_construct_bfw_ul(ru_emulator_, slot_msgs.sfn, slot_msgs.slot,
                                     static_cast<int>(cell_index), bfw_pdus);
        bfw_records = build_completion_records(bfw_pdus, collect_pusch_allocs(ul_req));
        bfw_params.completion_records = bfw_records;
        bfw_params.slot_dynamic_beam_id_start = bfw_config_.dynamic_beam_id_start;
        bfw_params.chaining_enabled = false;
    }

    const auto tx_id = ran::oran::CPlaneTransactionId{0};
    auto pkt_count = generator_->prepare_ul(tx_id,
                                            {ul_req, ul_body_len},
                                            cell_index,
                                            tx_windows_,
                                            ru_config_,
                                            bfw_params);
    if (!pkt_count.has_value()) {
        NVLOGE_FMT(TAG_CPTB_COMMON, AERIAL_INVALID_PARAM_EVENT,
                   "prepare_ul failed for slot {}: {}", slot_index, pkt_count.error().message());
        FAIL() << "prepare_ul failed: " << pkt_count.error().message();
        return;
    }

    const auto num_packets = pkt_count->num_packets;
    if (num_packets == 0) {
        return;
    }

    // Ensure we have enough buffers
    while (packet_buffers_.size() < num_packets) {
        packet_buffers_.emplace_back(kMtuSize);
    }

    // Phase 2: generate UL packets into VecBuf
    auto gen_result = generator_->generate_ul_packets(
        tx_id,
        std::span<ran::oran::VecBuf>{packet_buffers_.data(), num_packets});

    if (!gen_result.has_value()) {
        NVLOGE_FMT(TAG_CPTB_COMMON, AERIAL_INVALID_PARAM_EVENT,
                   "generate_ul_packets failed for slot {}: {}", slot_index, gen_result.error().message());
        FAIL() << "generate_ul_packets failed: " << gen_result.error().message();
        return;
    }

    NVLOGD_FMT(TAG_CPTB_COMMON, "Slot {} (SFN={}.SLT={}): {} UL packets generated",
               slot_index, slot_msgs.sfn, slot_msgs.slot, *gen_result);

    // Phase 3: feed each packet to RU emulator for UL verification
    for (size_t i = 0; i < *gen_result; ++i) {
        auto& buf = packet_buffers_[i];

        if (TestConfig::instance().verify_cplane) {
            ru_emulator_verify_ul_cplane_content(
                ru_emulator_, buf.data(), buf.size(), static_cast<int>(cell_index));
        }
    }

    // Phase 4: optional PCAP capture (same pcap_writer library as DLC test bench)
    capture_packets(packet_buffers_.data(), *gen_result);
}

// ---------------------------------------------------------------------------
// Test: run all UL slots through the pipeline
// ---------------------------------------------------------------------------
TEST_F(CPlaneTestFixture, AllSlotsUlCplane)
{
    const size_t total_slots = fapi_source_->get_total_slots();
    NVLOGC_FMT(TAG_CPTB_COMMON, "Running {} UL slots through CPlaneGenerator -> RU Emulator", total_slots);

    for (size_t slot = 0; slot < total_slots; ++slot) {
        SCOPED_TRACE(fmt::format("slot_index={} SFN={} SLT={}", slot, slot / 20, slot % 20));
        RunSlotUl(slot, 0);
    }
}

} // namespace cplane_tb
