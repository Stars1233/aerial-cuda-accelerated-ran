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

#include "test_config.hpp" // for cuphy::test_config
#include "cuphy_testWrkr.hpp"
#include "cumac_testWrkr.hpp"

#define CURRENT_MAX_GREEN_CTXS 10 // excessive but ok; currently 6 are used
// #define CUBB_GPU_TESTBENCH_POWER_ITERATION_BATCHING // to control power iteration mode

// NOTE: cannot define vectors of cuPHYTestWorker or cuMACTestWorker yet (need copy constructors). Until then, need use separate function to finish simulation after creation.
typedef std::vector<std::vector<std::vector<std::string>>> string3Dvec_t;

// Global start-delay config (defaults from testbench_common.hpp; overridden from vectors YAML if provided).
start_delay_cfg_us g_start_delay_cfg_us;

// Global TV parameter override config (optional; overridden from vectors YAML if provided).
tv_override_cfg g_tv_override_cfg;

static std::optional<int64_t> parse_i64_loose(const yaml::node& n);
static std::optional<ul_anchor_mode_t> parse_ul_anchor_mode(const yaml::node& n);

/** Read optional cumac_options from YAML (root["cumac_options"] or root["config"]["cumac_options"]). Leaves out unchanged on absence. */
static void apply_cumac_options_from_yaml(const yaml::node& root, CumacOptions& out)
{
    auto parse_cumac = [&out](const yaml::node& co) {
        auto read_uint8_arr = [&co](const char* key, uint8_t* arr, size_t n) {
            try {
                for(size_t i = 0; i < n; i++) {
                    yaml::node elem = co[key][i];
                    arr[i] = static_cast<uint8_t>(elem.as<int>());
                }
            } catch(...) {}
        };
        auto read_int_arr = [&co](const char* key, int* arr, size_t n) {
            try {
                for(size_t i = 0; i < n; i++) {
                    yaml::node elem = co[key][i];
                    arr[i] = elem.as<int>();
                }
            } catch(...) {}
        };
        auto read_float_arr = [&co](const char* key, float* arr, size_t n) {
            try {
                for(size_t i = 0; i < n; i++) {
                    yaml::node elem = co[key][i];
                    arr[i] = elem.as<float>();
                }
            } catch(...) {}
        };
        read_uint8_arr("modules_called", out.modules_called, 4);
        read_int_arr("cumac_light_weight_flag", out.cumac_light_weight_flag, kMaxMacSlotsPerPattern);
        read_float_arr("perc_sm_num_thrd_blk", out.perc_sm_num_thrd_blk, kMaxMacSlotsPerPattern);
        try {
            yaml::node seq = co["cumac_light_weight_flag"];
            auto len = seq.length();
            if (len > kMaxMacSlotsPerPattern) {
                NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT, "ERROR: cumac_light_weight_flag array length ({}) exceeds kMaxMacSlotsPerPattern ({})", len, kMaxMacSlotsPerPattern);
                exit(1);
            }
            out.config_slot_count = static_cast<uint8_t>(len);

            if (co.has_key("perc_sm_num_thrd_blk")) {
                auto pLen = co["perc_sm_num_thrd_blk"].length();
                if (pLen != len) {
                    NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT, "ERROR: perc_sm_num_thrd_blk length ({}) does not match cumac_light_weight_flag length ({})", pLen, len);
                    exit(1);
                }
            }
        } catch(...) {}
        try { out.half_precision = static_cast<uint8_t>(co["half_precision"].template as<int>()); } catch(...) {}
        try { out.sch_alg = static_cast<uint8_t>(co["sch_alg"].template as<int>()); } catch(...) {}
        try { out.hetero_ue_sel_cells = static_cast<uint8_t>(co["hetero_ue_sel_cells"].template as<int>()); } catch(...) {}
    };
    try
    {
        if(root.has_key("cumac_options"))
            parse_cumac(root["cumac_options"]);
        else if(root.has_key("config"))
        {
            yaml::node cfg = root["config"];
            if(cfg.has_key("cumac_options"))
                parse_cumac(cfg["cumac_options"]);
        }
    }
    catch(...)
    {}
}

static void apply_start_delay_overrides_from_yaml(const yaml::node& root)
{
    try
    {
        yaml::node sd = root["start_delay"];

        auto set_u32 = [&](const char* k, uint32_t& dst) {
            try
            {
                dst = sd[k].as<unsigned int>();
            }
            catch(...)
            {}
        };
        auto set_i32 = [&](const char* k, int32_t& dst) {
            try
            {
                auto opt = parse_i64_loose(sd[k]);
                if(opt.has_value())
                    dst = static_cast<int32_t>(*opt);
            }
            catch(...)
            {}
        };
        auto set_ul_anchor = [&](const char* k, ul_anchor_mode_t& dst) {
            try
            {
                auto opt = parse_ul_anchor_mode(sd[k]);
                if(opt.has_value())
                {
                    dst = *opt;
                    g_start_delay_cfg_us.ul_anchor_from_yaml = true;
                }
            }
            catch(...)
            {}
        };

        // Generic keys (apply to both u5/u6 where applicable)
        set_u32("SRS", g_start_delay_cfg_us.srs_u5);
        g_start_delay_cfg_us.srs_u6 = g_start_delay_cfg_us.srs_u5;
        set_u32("SRS2", g_start_delay_cfg_us.srs2_u5);
        g_start_delay_cfg_us.srs2_u6 = g_start_delay_cfg_us.srs2_u5;
        set_ul_anchor("UL_ANCHOR", g_start_delay_cfg_us.ul_anchor_mode);

        // Keys matching phase3_test_config.yaml conventions
        set_i32("PUSCH1", g_start_delay_cfg_us.pusch_u5);
        g_start_delay_cfg_us.pusch_u6 = g_start_delay_cfg_us.pusch_u5;
        set_u32("PUCCH1", g_start_delay_cfg_us.pucch_u5);
        g_start_delay_cfg_us.pucch_u6 = g_start_delay_cfg_us.pucch_u5;
        if(sd.has_key("PRACH"))
        {
            set_u32("PRACH", g_start_delay_cfg_us.prach_u5);
            g_start_delay_cfg_us.prach_u6 = g_start_delay_cfg_us.prach_u5;
            g_start_delay_cfg_us.prach_delay_from_yaml = true;
        }
        set_i32("PUSCH2", g_start_delay_cfg_us.pusch2_u5);
        g_start_delay_cfg_us.pusch2_u6 = g_start_delay_cfg_us.pusch2_u5;
        set_i32("PUCCH2", g_start_delay_cfg_us.pucch2_u6);
        set_u32("PDSCH_DLBFW", g_start_delay_cfg_us.dlbfw_slot0_u5);
        g_start_delay_cfg_us.dlbfw_u6 = g_start_delay_cfg_us.dlbfw_slot0_u5;
        set_u32("PDSCH", g_start_delay_cfg_us.pdsch_no_bfw_u6);
        set_i32("PDCCH", g_start_delay_cfg_us.pdcch_u6);
        set_i32("PDCCH_CSIRS", g_start_delay_cfg_us.pdcch_csirs_u6);
        set_u32("SSB", g_start_delay_cfg_us.ssb_u5);
        g_start_delay_cfg_us.ssb_u6 = g_start_delay_cfg_us.ssb_u5;

        // Optional explicit u6-only keys
        set_u32("ULBFW", g_start_delay_cfg_us.ulbfw_u6);
        set_u32("ULBFW2", g_start_delay_cfg_us.ulbfw2_u6);
        set_u32("DLBFW", g_start_delay_cfg_us.dlbfw_u6);
    }
    catch(...)
    {
        // start_delay not present: keep defaults
    }
}

static std::optional<int64_t> parse_i64_loose(const yaml::node& n)
{
    try
    {
        return static_cast<int64_t>(const_cast<yaml::node&>(n).as<int>());
    }
    catch(...)
    {}
    try
    {
        std::string s = const_cast<yaml::node&>(n).as<std::string>();
        std::string sl = s;
        for(size_t idx = 0; idx < sl.size(); ++idx)
        {
            if(sl[idx] >= 'A' && sl[idx] <= 'Z')
                sl[idx] = static_cast<char>(sl[idx] - 'A' + 'a');
        }
        if(sl == "true")
            return 1;
        if(sl == "false")
            return 0;

        size_t start = 0;
        while(start < s.size() && (s[start] == ' ' || s[start] == '\t'))
            ++start;
        size_t end = s.size();
        while(end > start && (s[end - 1] == ' ' || s[end - 1] == '\t'))
            --end;
        if(start >= end)
            return std::nullopt;

        try
        {
            return std::stoll(s.substr(start, end - start));
        }
        catch(const std::invalid_argument&)
        {
            return std::nullopt;
        }
        catch(const std::out_of_range&)
        {
            return std::nullopt;
        }
    }
    catch(...)
    {
        return std::nullopt;
    }
}

static std::optional<ul_anchor_mode_t> parse_ul_anchor_mode(const yaml::node& n)
{
    try
    {
        std::string s = const_cast<yaml::node&>(n).as<std::string>();
        for(size_t idx = 0; idx < s.size(); ++idx)
        {
            if(s[idx] >= 'a' && s[idx] <= 'z')
                s[idx] = static_cast<char>(s[idx] - 'a' + 'A');
        }
        size_t start = 0;
        while(start < s.size() && (s[start] == ' ' || s[start] == '\t'))
            ++start;
        size_t end = s.size();
        while(end > start && (s[end - 1] == ' ' || s[end - 1] == '\t'))
            --end;
        if(start >= end)
            return std::nullopt;

        std::string mode = s.substr(start, end - start);
        if(mode == "PUSCH")
            return ul_anchor_mode_t::PUSCH;
        if(mode == "PRACH")
            return ul_anchor_mode_t::PRACH;
        if(mode == "PUCCH")
            return ul_anchor_mode_t::PUCCH;
    }
    catch(...)
    {}
    return std::nullopt;
}

static void print_tv_overrides_config()
{
    if(!g_tv_override_cfg.enable)
    {
        NVLOGI_FMT(NVLOG_TESTBENCH_PHY, "[TV-OVERRIDE] disabled");
        return;
    }

    NVLOGI_FMT(NVLOG_TESTBENCH_PHY, "[TV-OVERRIDE] enabled");
    auto print_u8 = [](const char* chan, const char* key, bool has, uint8_t v) {
        if(has) NVLOGI_FMT(NVLOG_TESTBENCH_PHY, "[TV-OVERRIDE][{}] {}={}", chan, key, static_cast<unsigned int>(v));
    };
    auto print_u16 = [](const char* chan, const char* key, bool has, uint16_t v) {
        if(has) NVLOGI_FMT(NVLOG_TESTBENCH_PHY, "[TV-OVERRIDE][{}] {}={}", chan, key, static_cast<unsigned int>(v));
    };
    auto print_u32 = [](const char* chan, const char* key, bool has, uint32_t v) {
        if(has) NVLOGI_FMT(NVLOG_TESTBENCH_PHY, "[TV-OVERRIDE][{}] {}={}", chan, key, static_cast<unsigned int>(v));
    };

    const auto& po = g_tv_override_cfg.pusch;
    print_u8("PUSCH", "list_length", po.has_polar_list_length, po.polar_list_length);
    print_u8("PUSCH", "enable_cfo_correction", po.has_enable_cfo_correction, po.enable_cfo_correction);
    print_u8("PUSCH", "enable_weighted_average_cfo", po.has_enable_weighted_average_cfo, po.enable_weighted_average_cfo);
    print_u8("PUSCH", "enable_to_estimation", po.has_enable_to_estimation, po.enable_to_estimation);
    print_u8("PUSCH", "tdi_mode", po.has_tdi_mode, po.tdi_mode);
    print_u8("PUSCH", "enable_dft_sofdm", po.has_enable_dft_sofdm, po.enable_dft_sofdm);
    print_u8("PUSCH", "enable_rssi_measurement", po.has_enable_rssi_measurement, po.enable_rssi_measurement);
    print_u8("PUSCH", "enable_sinr_measurement", po.has_enable_sinr_measurement, po.enable_sinr_measurement);
    print_u8("PUSCH", "enable_static_dynamic_beamforming", po.has_enable_static_dynamic_beamforming, po.enable_static_dynamic_beamforming);
    print_u8("PUSCH", "enable_early_harq", po.has_enable_early_harq, po.enable_early_harq);
    print_u8("PUSCH", "ldpc_early_termination", po.has_ldpc_early_termination, po.ldpc_early_termination);
    print_u16("PUSCH", "ldpc_algorithm_index", po.has_ldpc_algorithm_index, po.ldpc_algorithm_index);
    print_u32("PUSCH", "ldpc_flags", po.has_ldpc_flags, po.ldpc_flags);
    print_u8("PUSCH", "ldpc_use_half", po.has_ldpc_use_half, po.ldpc_use_half);
    print_u8("PUSCH", "ldpc_max_num_iterations", po.has_ldpc_max_num_iterations, po.ldpc_max_num_iterations);
    print_u8("PUSCH", "ldpc_max_num_iterations_algorithm_index", po.has_ldpc_max_num_iterations_algorithm_index, po.ldpc_max_num_iterations_algorithm_index);
    print_u8("PUSCH", "dmrs_channel_estimation_algorithm_index", po.has_dmrs_channel_estimation_algorithm_index, po.dmrs_channel_estimation_algorithm_index);
    print_u8("PUSCH", "enable_per_prg_channel_estimation", po.has_enable_per_prg_channel_estimation, po.enable_per_prg_channel_estimation);
    print_u8("PUSCH", "eq_coefficient_algorithm_index", po.has_eq_coefficient_algorithm_index, po.eq_coefficient_algorithm_index);

    const auto& co = g_tv_override_cfg.pucch;
    print_u8("PUCCH", "list_length", co.has_polar_list_length, co.polar_list_length);

    const auto& so = g_tv_override_cfg.srs;
    print_u8("SRS", "chEst_alg_selector", so.has_chest_alg_index, so.chest_alg_index);
}

template<typename T>
static void read_uint(const yaml::node& parent, const char* key, bool& has, T& dst)
{
    has = false;
    try
    {
        auto opt = parse_i64_loose(parent[key]);
        if(!opt.has_value() || *opt == -1)
            return;
        dst = static_cast<T>(*opt);
        has = true;
    }
    catch(...)
    {}
}

/**
 * @brief Apply optional test-vector field overrides from YAML into global override state.
 *
 * Reads the `override_test_vectors` section from @p root and updates `g_tv_override_cfg`
 * only for fields explicitly set to values other than `-1`. Missing fields or sentinel
 * `-1` values are treated as no-op, parse failures are ignored, and existing try/catch
 * boundaries keep behavior non-fatal for malformed or absent override content.
 *
 * @param root Root YAML node parsed from the input vectors/config file.
 */
static void apply_tv_overrides_from_yaml(const yaml::node& root)
{
    try
    {
        yaml::node ov = root["override_test_vectors"];
        try
        {
            auto opt = parse_i64_loose(ov["enable_override"]);
            g_tv_override_cfg.enable = (opt.has_value() && *opt != 0);
        }
        catch(...)
        {
            g_tv_override_cfg.enable = false;
        }

        if(!g_tv_override_cfg.enable)
        {
            print_tv_overrides_config();
            return;
        }

        // PUSCH
        try
        {
            yaml::node p = ov["PUSCH"];
            read_uint<uint8_t>(p, "list_length", g_tv_override_cfg.pusch.has_polar_list_length, g_tv_override_cfg.pusch.polar_list_length);
            read_uint<uint8_t>(p, "enable_cfo_correction", g_tv_override_cfg.pusch.has_enable_cfo_correction, g_tv_override_cfg.pusch.enable_cfo_correction);
            read_uint<uint8_t>(p, "enable_weighted_average_cfo", g_tv_override_cfg.pusch.has_enable_weighted_average_cfo, g_tv_override_cfg.pusch.enable_weighted_average_cfo);
            read_uint<uint8_t>(p, "enable_to_estimation", g_tv_override_cfg.pusch.has_enable_to_estimation, g_tv_override_cfg.pusch.enable_to_estimation);
            read_uint<uint8_t>(p, "tdi_mode", g_tv_override_cfg.pusch.has_tdi_mode, g_tv_override_cfg.pusch.tdi_mode);
            read_uint<uint8_t>(p, "enable_dft_sofdm", g_tv_override_cfg.pusch.has_enable_dft_sofdm, g_tv_override_cfg.pusch.enable_dft_sofdm);
            read_uint<uint8_t>(p, "enable_rssi_measurement", g_tv_override_cfg.pusch.has_enable_rssi_measurement, g_tv_override_cfg.pusch.enable_rssi_measurement);
            read_uint<uint8_t>(p, "enable_sinr_measurement", g_tv_override_cfg.pusch.has_enable_sinr_measurement, g_tv_override_cfg.pusch.enable_sinr_measurement);
            read_uint<uint8_t>(p, "enable_static_dynamic_beamforming", g_tv_override_cfg.pusch.has_enable_static_dynamic_beamforming, g_tv_override_cfg.pusch.enable_static_dynamic_beamforming);
            read_uint<uint8_t>(p, "enable_early_harq", g_tv_override_cfg.pusch.has_enable_early_harq, g_tv_override_cfg.pusch.enable_early_harq);

            read_uint<uint8_t>(p, "ldpc_early_termination", g_tv_override_cfg.pusch.has_ldpc_early_termination, g_tv_override_cfg.pusch.ldpc_early_termination);
            read_uint<uint16_t>(p, "ldpc_algorithm_index", g_tv_override_cfg.pusch.has_ldpc_algorithm_index, g_tv_override_cfg.pusch.ldpc_algorithm_index);
            read_uint<uint32_t>(p, "ldpc_flags", g_tv_override_cfg.pusch.has_ldpc_flags, g_tv_override_cfg.pusch.ldpc_flags);
            read_uint<uint8_t>(p, "ldpc_use_half", g_tv_override_cfg.pusch.has_ldpc_use_half, g_tv_override_cfg.pusch.ldpc_use_half);
            read_uint<uint8_t>(p, "ldpc_max_num_iterations", g_tv_override_cfg.pusch.has_ldpc_max_num_iterations, g_tv_override_cfg.pusch.ldpc_max_num_iterations);
            read_uint<uint8_t>(p, "ldpc_max_num_iterations_algorithm_index", g_tv_override_cfg.pusch.has_ldpc_max_num_iterations_algorithm_index, g_tv_override_cfg.pusch.ldpc_max_num_iterations_algorithm_index);

            read_uint<uint8_t>(p, "dmrs_channel_estimation_algorithm_index", g_tv_override_cfg.pusch.has_dmrs_channel_estimation_algorithm_index, g_tv_override_cfg.pusch.dmrs_channel_estimation_algorithm_index);
            read_uint<uint8_t>(p, "enable_per_prg_channel_estimation", g_tv_override_cfg.pusch.has_enable_per_prg_channel_estimation, g_tv_override_cfg.pusch.enable_per_prg_channel_estimation);
            read_uint<uint8_t>(p, "eq_coefficient_algorithm_index", g_tv_override_cfg.pusch.has_eq_coefficient_algorithm_index, g_tv_override_cfg.pusch.eq_coefficient_algorithm_index);
        }
        catch(...)
        {}

        // PUCCH
        try
        {
            yaml::node p = ov["PUCCH"];
            read_uint<uint8_t>(p, "list_length", g_tv_override_cfg.pucch.has_polar_list_length, g_tv_override_cfg.pucch.polar_list_length);
        }
        catch(...)
        {}

        // SRS
        try
        {
            yaml::node p = ov["SRS"];
            read_uint<uint8_t>(p, "chEst_alg_selector", g_tv_override_cfg.srs.has_chest_alg_index, g_tv_override_cfg.srs.chest_alg_index);
        }
        catch(...)
        {}

        print_tv_overrides_config();
    }
    catch(...)
    {
        // No overrides present.
    }
}

// finishPschSim for run with uldl = 4
void finishPschSim(std::vector<cuPHYTestWorker*>& pCuphyTestWorkers, string3Dvec_t& inCtxFileNamesPuschRx, string3Dvec_t& inCtxFileNamesPdschTx, uint32_t nTimingItrs, uint32_t nPowerItrs, uint32_t num_patterns, uint32_t nCtxts, uint32_t nCuphyCtxts, bool printCbErrors, cuphy::stream& mainStream, int32_t gpuId, uint32_t delayUs, uint32_t powerDelayUs, bool ref_check_pdsch, bool identical_ldpc_configs, cuphyPdschProcMode_t pdsch_proc_mode, uint64_t pusch_proc_mode, uint32_t fp16Mode, int descramblingOn, uint32_t nCellsPerCtxt, uint32_t nStrmsPerCtxt, uint32_t nPschItrsPerStrm, cuphy::event_timer& slotPatternTimer, float tot_slotPattern_time, float avg_slotPattern_time_us, std::shared_ptr<cuphy::buffer<uint32_t, cuphy::pinned_alloc>>& shPtrGpuStartSyncFlag, std::map<std::string, int>& cuStrmPrioMap, uint32_t syncUpdateIntervalCnt, bool enableLdpcThroughputMode, bool printCellMetrics, bool pdsch_group_cells, uint32_t pdsch_cells_per_stream, bool pusch_group_cells, maxPDSCHPrms pdschPrms, maxPUSCHPrms puschPrms, uint32_t ldpcLaunchMode, uint32_t pdsch_nItrsPerStrm, uint8_t* pdschSlotRunFlag, uint8_t* puschSubslotProcFlag);

/**
 * Integration function for power iterations using reference vectors
 * 
 * @param[in] cuphyTestWorkerVec Vector of cuPHY test workers (references)
 * @param[in] nPowerItrs Number of power iterations to run
 * @param[in] powerDelayUs Delay in microseconds between power iterations
 * @param[in] mainStream CUDA stream for main execution
 * @param[in] gpuId GPU device ID
 * @param[in,out] shPtrGpuStartSyncFlag Shared pointer to GPU synchronization flag
 * @param[in] syncUpdateIntervalCnt Synchronization update interval count
 * @param[in] startEvent CUDA event for timing
 * @param[in] cumacTestWorkerVec Vector of cuMAC test workers (references)
 * @param[in] shPtrStopEvents Vector of shared pointers to stop events
 * @param[in] channelWorkerMap Unordered map of channel workers
 * @param[in] macWorkerMap Unordered map of MAC workers
 * @param[in] cfg_process_mode Processing mode configuration (0=stream, 1=graph)
 * @param[in] uldl UL/DL mode configuration
 * @param[in] nCuphyCtxts Number of cuPHY contexts
 * @param[in] nCumacCtxts Number of cuMAC contexts
 * @param[in] nCtxts Total number of contexts
 * @param[in] nSlotsPerPattern Number of slots per pattern
 * @param[in] pdschCtx Enable PDSCH context
 * @param[in] pdcchCtx Enable PDCCH context
 * @param[in] puschCtx Enable PUSCH context
 * @param[in] prachCtx Enable PRACH context
 * @param[in] pucchCtx Enable PUCCH context
 * @param[in] srsCtx Enable SRS context
 * @param[in] ssbCtx Enable SSB context
 * @param[in] macCtx Enable MAC context
 * @param[in] mac2Ctx Enable second MAC context
 * @param[in] macInternalTimer Use internal timer for MAC
 * @param[in] mac2InternalTimer Use internal timer for second MAC
 */
void runPowerIterations(
    std::vector<cuPHYTestWorker>& cuphyTestWorkerVec,
    uint32_t nPowerItrs, uint32_t powerDelayUs,
    cuphy::stream& mainStream, int32_t gpuId,
    std::shared_ptr<cuphy::buffer<uint32_t, cuphy::pinned_alloc>>& shPtrGpuStartSyncFlag,
    uint32_t syncUpdateIntervalCnt,
    cuphy::event& startEvent,
    std::vector<cuMACTestWorker>& cumacTestWorkerVec,
    std::vector<std::shared_ptr<cuphy::event>>& shPtrStopEvents,
    std::unordered_map<std::string, int>& channelWorkerMap,
    std::unordered_map<std::string, int>& macWorkerMap,
    int cfg_process_mode, int uldl,
    int nCuphyCtxts, int nCumacCtxts, int nCtxts,
    uint32_t nSlotsPerPattern,
    bool pdschCtx, bool pdcchCtx, bool puschCtx, bool prachCtx,
    bool pucchCtx, bool srsCtx, bool ssbCtx, bool macCtx, bool mac2Ctx,
    bool macInternalTimer, bool mac2InternalTimer);

// read SM Ids for cuPHY and cuMAC test workers
template<typename testWorkerType>
void readSmIds(std::vector<testWorkerType*>& ptestWorkerVec, cuphy::stream& strm, int gpuId);
//----------------------------------------------------------------------------------------------------------
// usage
//----------------------------------------------------------------------------------------------------------
void usage()
{
    printf("cuphy_ex_sch_rx_tx [options]\n");
    printf("  Options:\n");
    printf("    -h                     Display usage information\n");
    printf("    -i  yaml input_file    Input yaml filename\n");
    printf("    -d                     Debug message setting: Disable(0), Enable(> 0)\n");
    printf("    -m  process mode       streams (0), graphs (1).\n");
    printf("    -u                     uldl mode: -u 3 uses one UL and one DL context at a minimum, DL context processes 4 slots in serial\n");
    printf("                           uldl mode: -u 4 uses -C UL/DL contexts. Per slot each context has -S pipelines run -I times sequentially\n");
    printf("                           uldl mode: -u 5 long pattern DDDSUUDDDD\n");
    printf("                           uldl mode: -u 6 long pattern DDDSUUDDDD with new mMIMO timeline\n");
    printf("    -c  CPU Id             CPU Id used to run the first pipeline (default 0), (cpuIdPipeline[ii] = CPU Id, or cpuIdPipeline[ii] = CPU Id + ii if multi-core enabled)\n");
    printf("    -t                     Enables multi-core threading\n");
    printf("    -g  GPU Id             GPU Id used to run all the pipelines\n");
    printf("    -r  # of iterations    Number of run iterations to run (set to 1 internally for power measurements mode -P)\n");
    printf("    -w  delayUs            Set the GPU delay before running each pattern in microseconds (default: 10000)\n");
    printf("    -b                     Option to print codeblock errors\n");
    printf("    -l                     Force separate LDPC kernels, one per transport block, for PDSCH\n");
    printf("    -k                     Reference check for PDSCH (can also be enabled by --c PDSCH)\n");
    printf("    -S  <streams>          Number of streams per worker\n");
    printf("    -I  <iterations>       Number of iterations per stream\n");
    printf("    -B                     Enable PUSCH cascade (PUSCH2 must wait unitl PUSCH1 finishes) for DDDSUUDDDD (u5/u6) (default disabled) \n");
    printf("    -C  <contexts>         Number of contexts \n");
    printf("    -P  <iterations>       if > 0 enables power measurement mode (default: disabled). When enabled, min iteration count forced to 100)\n");
    printf("    -W  <delay (us)>       Inter slot pattern workload delay in microseconds for power measurement mode (default: 30)\n");
    printf("    -L                     Force selection of throughput mode for PUSCH LDPC decoder\n");
    printf("    -K                     LDPC kernel launch mode: -K 0, uses single stream using driver api (default);-K 1, uses multi-stream launch;\n");
    printf("                           -K 2, uses single-stream launch via tensor interface; -K 3, single stream opt;\n");
    printf("                           Note: option '-K' will be ignored if graph mode selected.\n");
    printf("    --H                    Use half precision (FP16) for back-end\n");
    printf("    --U                    Enable context for PUSCH\n");
    printf("    --D                    Enable context for PDSCH\n");
    printf("    --P                    Use separate context for PRACH\n");
    printf("    --Q                    Use separate context for PDCCH\n");
    printf("    --X                    Use separate context for PUCCH\n");
    printf("    --B                    Use separate context for SSB\n");
    printf("    --Z                    Use separate context for SRS\n");
    printf("    --S                    Split SRS: run first half of SRS cells at the beginning and the remaining half after PUSCH\n");
    printf("    --G                    Group together all PDSCH cells in a context in a slot and execute via single PDSCH pipeline object; same for PDCCH if present\n");
    printf("    --g                    Group together all PUSCH cells in a context in a slot and execute via single PUSCH pipeline object; same for PUCCH and PRACH if present\n");
    printf("    --T                    Enable first cuMAC workload (first cuMAC always runs in a separate context)\n");
    printf("    --V                    Enable second cuMAC workload (second cuMAC always runs in a separate context)\n");
    printf("    --R                    Use internal timer for cuMAC workload (otherwise use PDSCH timer)\n");
    printf("    --M <ctx1,..,ctxtN>    Max number of SMs used per context in comma separated list. Order is as follows: [PRACH if --P], [PDCCH if --Q], [PUCCH if --X], PDSCH, PUSCH, [SSB if SSB TVs exist in yaml], [SRS if --Z], [MAC if --T] \n");
    printf("    --k                    Reference check for PDCCH (can also be enabled by --c PDCCH)\n");
    printf("    --c <ch_name_1,...,ch_name_N> enable reference checks for channels whose names are provided as a comma separated list\n");
    printf("    --b                    [Deprecated] Inter-cell batching for PDSCH. No effect as inter-cell batching is always enabled with --G\n");
    printf("    -n                     Use green contexts\n");
    printf("    -v                     Enable cudaProfilerStart/Stop around each pattern run\n");
}

//----------------------------------------------------------------------------------------------------------
// main - Instantiates one or more workers objects. Responsible for accepting commands from user,
// orchestrating the tests
//----------------------------------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int returnValue = 0;
    testBenchNvlogFmtHelper nvlog_fmt("cubb_gpu_test_bench.log");

#if 0
    // temp. print args
    for (int n = 1; n < argc; n++) {
        std::cout << argv[n] << std::endl;
    }
#endif

    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int                iArg          = 1;
        std::string        inputFileName;

        int32_t              nMaxConcurrentThrds       = std::thread::hardware_concurrency();
        int32_t              baseCpuId                 = 0;
        bool                 enableMultiCore           = false;
        int32_t              gpuId                     = 0;
        uint32_t             nTimingItrs               = 1;
        uint32_t             nPowerItrs                = 0;
        int                  dbgMsgLevel               = 0; // no debug messages
        int                  descramblingOn            = 1;
        bool                 printCbErrors             = false; // enabled with --c ..., PUSCH, ...: PUSCH checks 
        bool                 ref_check_dlbfw           = false; // enabled with --c ...,DLBFW,...
        bool                 ref_check_ulbfw           = false; // enabled with --c ...,ULBFW,...
        bool                 ref_check_pucch           = false; // enabled with --c ...,PUCCH,...
        bool                 ref_check_prach           = false; // enabled with --c ...,PRACH,...
        bool                 ref_check_srs             = false; // enabled with --c ...,SRS,...
        bool                 ref_check_pdsch           = false; // enabled with -k or --c ...,PDSCH,...
        bool                 ref_check_pdcch           = false; // enabled with --k or --c ...,PDCCH,...
        bool                 ref_check_ssb             = false; // enabled with --c ...,SSB,...
        bool                 ref_check_csirs           = false; // enabled with --c ...,CSIRS,...
        bool                 ref_check_mac             = false; // enabled with --c ...,MAC,..., also used for MAC2
        int                  cfg_process_mode          = 0;
        bool                 identical_ldpc_configs    = true;
        bool                 enableLdpcThroughputMode  = false;
        cuphyPdschProcMode_t pdsch_proc_mode           = PDSCH_PROC_MODE_NO_GRAPHS;
        uint64_t             pusch_proc_mode           = 0;
        uint32_t             ldpcLaunchMode            = 1;
        uint32_t             delayUs                   = 10000;
        uint32_t             powerDelayUs              = 30; // emperically measured values from nsight-sys indicate a lower bound of around 15-20us
        uint32_t             fp16Mode                  = 1;
        bool                 enableHighPdschCuStrmPrio = false;
        uint32_t             nCtxts                    = 0; // total subcontexts = cuPHY subcontexts + cuMAC subcontexts or overwitten by -C
        uint32_t             nCuphyCtxts               = 0; // cuPHY subcontexts
        uint32_t             nCumacCtxts               = 0; // cuMAC subcontexts
        // place holder for other cuBB contexts
        uint32_t             nStrmsPerCtxt             = 0;
        uint32_t             nPuschStrmsPerCtxt        = 0;
        uint32_t             nUlBfwStrmsPerCtxt        = 0;
        uint32_t             nPucchStrmsPerCtxt        = 0;
        uint32_t             nPrachStrmsPerCtxt        = 0;
        uint32_t             nSrsStrmsPerCtxt          = 0;
        uint32_t             nPdschStrmsPerCtxt        = 0;
        uint32_t             nDlBfwStrmsPerCtxt        = 0;
        uint32_t             nPdcchStrmsPerCtxt        = 0;
        uint32_t             nSSBStrmsPerCtxt          = 0;
        uint32_t             nCsiRsStrmsPerCtxt        = 0;
        uint32_t             nPdschCellsPerStrm        = 1; // Number of cells per pipeline object per context. 1 unless group_pdsch_cells is set.
        uint32_t             nPdcchCellsPerStrm        = 1; // Number of cells per pipeline object per context. 1 unless group_pdsch_cells is set. Could be different than nPdschCellsPerStrm
        uint32_t             nPrachCellsPerStrm        = 1; // Number of cells per pipeline object per context. 1 unless group_pusch_cells is set.
        uint32_t             nPschItrsPerStrm          = 0; // used in u4 mode
        int                  pipelineExec              = 0;
        int                  uldl                      = 3;
        bool                 printCellMetrics          = false;
        uint32_t             num_patterns              = 0;
        uint32_t             nSlotsPerPattern          = 0;
        uint8_t              puschSubslotProcFlag[2]     = {0}; // puschSubslotProcFlag[0] for PUSCH1, puschSubslotProcFlag[1] for PUSCH2
        uint32_t             pusch_nItrsPerStrm   = 0;
        uint32_t             pucch_nItrsPerStrm   = 0;
        uint32_t             ulbfw_nItrsPerStrm   = 0;
        uint32_t             prach_nItrsPerStrm   = 0;
        uint32_t             srs_nItrsPerStrm     = 0;
        uint32_t             pdsch_nItrsPerStrm   = 0;
        uint32_t             pdcch_nItrsPerStrm   = 0;
        uint32_t             csirs_nItrsPerStrm   = 0;
        uint32_t             dlbfw_nItrsPerStrm   = 0;
        uint32_t             ssb_nItrsPerStrm     = 0;
        uint32_t             nPdschCellsPerPattern     = 0;
        uint32_t             nPuschCellsPerPattern     = 0;
        uint32_t             nPschCellsPerPattern      = 0;
        int                  mpsVals                   = 0;
        bool                 heterogenousMpsPartitions = false;
        std::vector<int32_t> ctxSmCounts;
        bool                 group_pdsch_cells         = false;
        bool                 group_pusch_cells         = false;
        bool                 group_pucch_cells         = false;
        bool                 splitSRScells50_50        = false;
        bool                 puschCtx                  = false;
        bool                 pdschCtx                  = false;
        bool                 prachCtx                  = false;
        bool                 pdcchCtx                  = false;
        bool                 pucchCtx                  = false;
        bool                 srsCtx                    = false; // default SRS runs in PUSCH subcontext
        bool                 ssbCtx                    = false; // default SSB runs in PDCCH subcontext
        bool                 pdsch_inter_cell_batching = true; // deprecated
        uint32_t             mode                      = 0; // 0 is mode A (default), 1 is mode B
        bool                 useGreenContexts          = false; // Set if '-n' is used
        bool                 enableNvprof              = false;

        // mac paramters
        bool                 macCtx                    = false; // enable subcontext for first MAC workload
        bool                 mac2Ctx                   = false; // enable subcontext for second MAC workload
        // mac run configs, same for both MAC and MAC2
        uint32_t             nStrmsMac                 = 1; // number of streams to run cuMAC workload; TODO: currently hardcoded to 1
        uint32_t             nMacItrsPerStrm           = 0; // number of cuMAC iterations per stream; nStrmsMac * nMacItrsPerStrm = nMacSlots
        uint32_t             nMacSlots     = 0; // number of slots per pattern for cuMAC workload; TODO: currently only support nMacSlots = 8 for DDDSUUDDDD
        uint32_t             nStrmsMac2                = 1; // number of streams to run cuMAC2 workload; TODO: currently hardcoded to 1
        uint32_t             nMac2ItrsPerStrm          = 0; // number of cuMAC2 iterations per stream; nStrmsMac2 * nMac2ItrsPerStrm = nMacSlots2
        uint32_t             nMac2Slots    = 0; // number of slots per pattern for cuMAC2 workload; TODO: currently only support nMac2Slots = 8 for DDDSUUDDDD
        bool                 enableMacInternalTimer    = false; // enable internal timer for MAC and/or MAC2

        std::unordered_map<std::string, int> chNameMap = {{"PUSCH", 0}, {"PDSCH", 1}, {"PDCCH", 2}, {"PUCCH", 3}, {"PRACH", 4}, {"DLBFW", 5}, {"ULBFW", 6}, {"SSB", 7}, {"CSIRS", 8}, {"SRS", 9}, {"MAC", 10}, {"MAC2", 11}};

        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                switch(argv[iArg][1])
                {
                case 'i':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: No filename provided");
                    }
                    inputFileName.assign(argv[iArg++]);
                    break;
                case 'h':
                    usage();
                    exit(0);
                    break;
                case 'a':
                    enableHighPdschCuStrmPrio = true;
                    ++iArg;
                    break;
                case 'k':
                    ref_check_pdsch = true;
                    ++iArg;
                    break;
                case 'l':
                    identical_ldpc_configs = false; // Launch separate LDPC kernels, one per TB.
                    ++iArg;
                    break;
                case 'L':
                    enableLdpcThroughputMode = true;
                    ++iArg;
                    break;
                case 'm':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &cfg_process_mode)) || ((cfg_process_mode < 0)) || ((cfg_process_mode > 1)))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: Invalid process mode");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'K':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%u", &ldpcLaunchMode)) || (3 < ldpcLaunchMode))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: Invalid LDPC kernel launch mode ({})", ldpcLaunchMode);
                        exit(1);
                    }
                    ldpcLaunchMode = 1 << ldpcLaunchMode;
                    ++iArg;
                    break;
                case 'b':
                    printCbErrors = true;
                    ++iArg;
                    break;
                case 'r':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nTimingItrs)) || ((nTimingItrs <= 0)))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: Invalid number of run iterations");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'B':
                    mode = 1;
                    ++iArg;
                    break;
                case 'C':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nCtxts)) || (nCtxts <= 0))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: Invalid number of contexts");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'S':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nStrmsPerCtxt)) || ((nStrmsPerCtxt <= 0)))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: Invalid number of streams per context");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'I':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nPschItrsPerStrm)) || ((nPschItrsPerStrm <= 0)))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: Invalid number of iterations per stream");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'P':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nPowerItrs)) || ((nPowerItrs <= 0)))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: Invalid number of iterations");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'W':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%u", &powerDelayUs)) || (powerDelayUs < 0))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: Invalid power delay value (should be atleast 30us)");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'g':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &gpuId)) ||
                       ((gpuId < 0)))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: Invalid GPU Id {}", gpuId);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'w':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%u", &delayUs))) {
                        NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: Invalid delay");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'c':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &baseCpuId)) ||
                       ((baseCpuId < 0) || (baseCpuId >= nMaxConcurrentThrds)))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: Invalid base CPU Id (should be within [0,{}])", nMaxConcurrentThrds-1);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 't':
                    enableMultiCore = true;
                    ++iArg;
                    break;
                case 'd':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &dbgMsgLevel)) || ((dbgMsgLevel < 0)))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: Invalid debug message level {}, disabling debug messages", dbgMsgLevel);
                        dbgMsgLevel = 0;
                    }
                    ++iArg;
                    break;
                case 'v':
                    enableNvprof = true;
                    ++iArg;
                    break;
                case 'n':
                    useGreenContexts = true;
                    ++iArg;
                    break;
                case 'u':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &uldl)) ||
                       ((uldl < 3) || (uldl > 6)))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: Invalid uldl mode {}", uldl);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case '-':
                    switch(argv[iArg][2])
                    {
                    case 'H':
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &fp16Mode)) || (1 < fp16Mode))
                        {
                            NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: Invalid FP16 mode {}\n", fp16Mode);
                            exit(1);
                        }
                        ++iArg;
                        break;
                    case 'c':
                        if(++iArg >= argc)
                        {
                            NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "List of channel names not provided for reference check option --c");
                        }
                        else
                        {
                            // Input -> std::string
                            std::string chanelNameListStr(argv[iArg++]);
                            // std::string -> std::stringstream
                            std::stringstream channelNameListStrStrm;
                            channelNameListStrStrm.str(chanelNameListStr);

                            // Extract from std::stringsream into ctxSmCounts
                            while(channelNameListStrStrm.good())
                            {
                                std::string subStr;
                                getline(channelNameListStrStrm, subStr, ',');
                                switch(chNameMap[subStr])
                                {
                                case 0:
                                    printCbErrors = true;
                                    NVLOGI_FMT(NVLOG_TESTBENCH, "Reference checks for PUSCH enabled");
                                    break;
                                case 1:
                                    ref_check_pdsch = true;
                                    NVLOGI_FMT(NVLOG_TESTBENCH, "Reference checks for PDSCH enabled");
                                    break;
                                case 2:
                                    ref_check_pdcch = true;
                                    NVLOGI_FMT(NVLOG_TESTBENCH, "Reference checks for PDCCH enabled");
                                    break;
                                case 3:
                                    ref_check_pucch = true;
                                    NVLOGI_FMT(NVLOG_TESTBENCH, "Reference checks for PUCCH enabled");
                                    break;
                                case 4:
                                    ref_check_prach = true;
                                    NVLOGI_FMT(NVLOG_TESTBENCH, "Reference checks for PRACH enabled");
                                    break;
                                case 5:
                                    ref_check_dlbfw = true;
                                    NVLOGI_FMT(NVLOG_TESTBENCH, "Reference checks for DLBFW enabled");
                                    break;
                                case 6:
                                    ref_check_ulbfw = true;
                                    NVLOGI_FMT(NVLOG_TESTBENCH, "Reference checks for ULBFW enabled");
                                    break;
                                case 7:
                                    ref_check_ssb = true;
                                    NVLOGI_FMT(NVLOG_TESTBENCH, "Reference checks for SSB enabled");
                                    break;
                                case 8:
                                    ref_check_csirs = true;
                                    NVLOGI_FMT(NVLOG_TESTBENCH, "Reference checks for CSIRS enabled");
                                    break;
                                case 9:
                                    ref_check_srs = true;
                                    NVLOGI_FMT(NVLOG_TESTBENCH, "Reference checks for SRS enabled");
                                    break;
                                case 10:
                                    ref_check_mac = true;
                                    NVLOGI_FMT(NVLOG_TESTBENCH, "Reference checks for MAC and MAC2 enabled");
                                    break;
                                default:
                                    break;
                                }
                            }
                        }
                        break;

                    case 'M':
                        heterogenousMpsPartitions = true;
                        if(++iArg >= argc || (1 != sscanf(argv[iArg], "%i", &mpsVals)))
                        {
                            NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "Using default SM counts for all sub-contexts");
                        }
                        else
                        {
                            // Input -> std::string
                            std::string ctxSmCountsStr(argv[iArg++]);
                            // std::string -> std::stringstream
                            std::stringstream ctxSmCountsStrStrm;
                            ctxSmCountsStrStrm.str(ctxSmCountsStr);

                            // Extract from std::stringsream into ctxSmCounts
                            int i = 0;
                            while(ctxSmCountsStrStrm.good())
                            {
                                std::string subStr;
                                getline(ctxSmCountsStrStrm, subStr, ',');
                                ctxSmCounts.push_back(std::stoul(subStr));
                                //printf("ctxSmCounts[%d] str %s subStr %s int %d\n", i++, ctxSmCountsStrStrm.str().c_str(), subStr.c_str(), ctxSmCounts.back());
                            }
                            // printf("MPS sub-context SM counts:\n");
                            // for(auto &ctxSmCount : ctxSmCounts) {printf("%d\n", ctxSmCount);}
                        }
                        break;
                    case 'U':
                        puschCtx = true;
                        nCuphyCtxts ++;
                        ++iArg;
                        break;
                    case 'D':
                        pdschCtx = true;
                        nCuphyCtxts ++;
                        ++iArg;
                        break;
                    case 'P':
                        prachCtx = true;
                        nCuphyCtxts++;
                        ++iArg;
                        break;
                    case 'Q':
                        pdcchCtx = true;
                        nCuphyCtxts++;
                        ++iArg;
                        break;
                    case 'X':
                        pucchCtx = true;
                        nCuphyCtxts++;
                        ++iArg;
                        break;
                    case 'B':
                        ssbCtx = true;
                        nCuphyCtxts++;
                        ++iArg;
                        break;
                    case 'Z':
                        srsCtx = true;
                        nCuphyCtxts++;
                        ++iArg;
                        break;
                    case 'S':
                        splitSRScells50_50 = true;
                        ++iArg;
                        break;
                    case 'T':
#ifdef AERIAL_CUMAC_ENABLE
                        macCtx = true;
                        nCumacCtxts ++;
#else
                        NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT, "ERROR: cuMAC support disabled. Cannot use --T option. Rebuild with -DENABLE_CUMAC=ON to enable cuMAC support.");
                        exit(1);
#endif
                        ++iArg;
                        break;
                    case 'V':
#ifdef AERIAL_CUMAC_ENABLE
                        mac2Ctx = true;
                        nCumacCtxts ++;
#else
                        NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT, "ERROR: cuMAC support disabled. Cannot use --V option. Rebuild with -DENABLE_CUMAC=ON to enable cuMAC support.");
                        exit(1);
#endif
                        ++iArg;
                        break;
                    case 'R':
                        enableMacInternalTimer = true;
                        ++iArg;
                        break;
                    case 'g':
                        group_pusch_cells = group_pucch_cells = true;
                        ++iArg;
                        break;
                    case 'G':
                        group_pdsch_cells = true;
                        ++iArg;
                        break;
                    case 'b':
                        pdsch_inter_cell_batching = true; // no effect, true by default
                        ++iArg;
                        break;
                    case 'k':
                        ref_check_pdcch = true;
                        ++iArg;
                        break;
                    default:
                        usage();
                        exit(1);
                        break;
                    }
                    break;
                default:
                    NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                    usage();
                    exit(1);
                    break;
                }
            }

            else // if('-' == argv[iArg][0])
            {
                NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
                exit(1);
            }
        } // while (iArg < argc)

        // make sure the total contexts match
        if(nCtxts == 0)
        {
            nCtxts = nCuphyCtxts + nCumacCtxts;
            if(nCtxts == 0) // if still 0, issue error and quit
            {
                NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: nCtxts = 0, no subcontexts are defined, please check input arguments");
                usage();
                exit(1);
            }
        }
        if(uldl == 4) // overwritten nCuphyCtxts by nCtxts, used in finishPschSim(*); cuMAC does not supported in u4 yet
            nCuphyCtxts = nCtxts;

        if(useGreenContexts)
        {
            // Hardcode max connections for now or otherwise work like PUCCH and SSB could execute at the end of a pattern, not respecting expected timeline
            // Needs to happen before any other CUDA API calls.
            setenv("CUDA_DEVICE_MAX_CONNECTIONS", "12", 1);

#if CUDA_VERSION < 12040
            NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: CUDA_VERSION {}, which is before 12.4, does not support green contexts. Run in MPS mode, i.e., without -n.", CUDA_VERSION);
            exit(1);
#endif
        }
        const char* dev_max_connections_env_var = getenv("CUDA_DEVICE_MAX_CONNECTIONS");
        NVLOGC_FMT(NVLOG_TAG_BASE_TESTBENCH, "CUDA_DEVICE_MAX_CONNECTIONS {}", dev_max_connections_env_var);

        if(inputFileName.empty())
        {
            usage();
            exit(1);
        }

        if(prachCtx && !(uldl == 3 || uldl == 5 || uldl == 6))
        {
            NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: PRACH context mode --P can only be used in conjunction with -u 3 or -u 5 or -u 6");
            exit(1);
        }
        if(pdcchCtx && !(uldl == 3 || uldl == 5 || uldl == 6))
        {
            NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: PDCCH context mode --Q can only be used in conjunction with -u 3 or -u 5 or -u 6");
            exit(1);
        }
        if(pucchCtx && !(uldl == 3 || uldl == 5 || uldl == 6))
        {
            NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: PUCCH context mode --X can only be used in conjunction with -u 3 or -u 5 or -u 6");
            exit(1);
        }
        if(uldl != 5 && nCumacCtxts > 0) // TODO: cuMAC only support -u 5
        {
            NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: only -u 5 supports cuMAC contexts");
            exit(1);
        }

        //---------------------------------------------------------------------------
        // input files

        // yaml parsing
        cuphy::test_config testCfg(inputFileName.c_str());
        // testCfg.print();
        uint32_t cells_per_slot   = static_cast<uint32_t>(testCfg.num_cells());
        uint32_t num_slots        = static_cast<uint32_t>(testCfg.num_slots());
        uint32_t slotsPerPatternFromYaml = 0;
        CumacOptions cumacOptions;

        // Optional YAML start-delay overrides (generated by perf Python scripts).
        {
            yaml::file_parser fp(inputFileName.c_str());
            yaml::document    d = fp.next_document();
            yaml::node        r = d.root();
            apply_start_delay_overrides_from_yaml(r);
            apply_tv_overrides_from_yaml(r);
            apply_cumac_options_from_yaml(r, cumacOptions);
            if(r.has_key("slots_per_pattern"))
            {
                slotsPerPatternFromYaml = r["slots_per_pattern"].as<unsigned int>();
                if(slotsPerPatternFromYaml == 0)
                {
                    NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "Error! slots_per_pattern in YAML must be greater than zero");
                    exit(1);
                }
            }
        }

        uint32_t nPUSCHCells      = 0;
        uint32_t nUlbfwCells      = 0;
        uint32_t nPUCCHCells      = 0;
        uint32_t nPRACHCells      = 0;
        uint32_t nSRSCells        = 0;
        uint32_t nPDSCHCells      = 0;    
        uint32_t nDlbfwCells      = 0;
        uint32_t nPDCCHCells      = 0;  
        uint32_t nSSBCells        = 0;
        uint32_t nCSIRSCells      = 0;
        uint32_t nMACWorkers      = 0; // number of cuMAC test workers, configured by number of cuMAC TVs in the yaml file
        uint32_t nMAC2Workers     = 0; // number of cuMAC2 test workers, configed by number of cuMAC2 TVs in the yaml file

        // convert slots to slot patterns
        switch(uldl) // input argument from -u option
        {
            case 3: {
                if(num_slots % 4 != 0)
                {
                    NVLOGW_FMT(NVLOG_TAG_BASE_TESTBENCH, "Warning! For F13 mode (u = 4) the number of slots in YAML must be a multiple of four");
                    num_patterns              = num_slots;
                    nSlotsPerPattern          = 1;
                }
                else
                {
                    num_patterns              = num_slots / 4;
                    nSlotsPerPattern          = 4;
                }
                break;
            }

            case 4: {
                num_patterns            = num_slots;
                nSlotsPerPattern        = 1;
                break;
            }

            case 5: {
                nSlotsPerPattern = (slotsPerPatternFromYaml > 0) ? slotsPerPatternFromYaml : 10;
                if(slotsPerPatternFromYaml == 0)
                {
                    NVLOGW_FMT(NVLOG_TAG_BASE_TESTBENCH, "Warning! slots_per_pattern is not provided for u=5; using default 10-slot DDDSUUDDDD pattern");
                }
                if(num_slots % nSlotsPerPattern != 0)
                {
                    NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "Error! For DDDSUUDDDD mode (u = 5) the number of slots in YAML must be a multiple of slots_per_pattern ({})", nSlotsPerPattern);
                    exit(1);
                }

                num_patterns            = num_slots / nSlotsPerPattern;
                break;
            }

            case 6: {
                if(num_slots % 15 != 0)
                {
                    NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "Error! For DDDSUUDDDD mode (u = 6) the number of slots in YAML must be a multiple of fifteen");
                    exit(1);
                }
                // TODO: need to check with Harsha
                num_patterns            = num_slots / 15;
                nSlotsPerPattern        = 15;
                break;
            }
        }

        bool graphs_mode = (cfg_process_mode >= 1); // input argument from -m option
        pdsch_proc_mode  = (graphs_mode) ? PDSCH_PROC_MODE_GRAPHS : PDSCH_PROC_MODE_NO_GRAPHS;
        if(pdsch_inter_cell_batching) // always true
        {
            pdsch_proc_mode = (cuphyPdschProcMode_t)((uint64_t)pdsch_proc_mode | (uint64_t)PDSCH_INTER_CELL_BATCHING);
        }
        pusch_proc_mode = (graphs_mode) ? 1 : 0;

        // read H5 files
        const std::string puschChannelName  = "PUSCH";
        const std::string pusch2ChannelName = "PUSCH2";
        const std::string pdschChannelName  = "PDSCH";
        const std::string ulbfwChannelName  = "ULBFW";
        const std::string dlbfwChannelName  = "DLBFW";
        const std::string srsChannelName    = "SRS";
        const std::string prachChannelName  = "PRACH";
        const std::string pdcchChannelName  = "PDCCH";
        const std::string pucchChannelName  = "PUCCH";
        const std::string pucch2ChannelName = "PUCCH2";
        const std::string ssbChannelName    = "SSB";
        const std::string csirsChannelName  = "CSIRS";
        const std::string macChannelName    = "MAC";
        const std::string mac2ChannelName   = "MAC2";

        std::vector<std::vector<std::string>> inFileNamesPuschRx(num_patterns); // Dim: num_patterns x (num_cells * pusch_nItrsPerStrm)
        std::vector<std::vector<std::string>> inFileNamesPdschTx(num_patterns); // Dim: num_patterns x (num_cells * pdsch_nItrsPerStrm)
        std::vector<std::vector<std::string>> inFileNamesDlbfw(num_patterns);     // Dim: num_patterns x (num_cells * dlbfw_nItrsPerStrm)
        std::vector<std::vector<std::string>> inFileNamesUlbfw(num_patterns);     // Dim: num_patterns x (num_cells * ulbfw_nItrsPerStrm)
        std::vector<std::vector<std::string>> inFileNamesSRS(num_patterns);     // Dim: num_patterns x (num_cells * srs_nItrsPerStrm)
        std::vector<std::vector<std::string>> inFileNamesPRACH(num_patterns);   // Dim: num_patterns x (num_cells * prach_nItrsPerStrm)
        std::vector<std::vector<std::string>> inFileNamesPdcchTx(num_patterns); // Dim: num_patterns x (num_cells * pdcch_nItrsPerStrm)
        std::vector<std::vector<std::string>> inFileNamesPucchRx(num_patterns); // Dim: num_patterns x (num_cells * pusch_nItrsPerStrm)
        std::vector<std::vector<std::string>> inFileNamesSSB(num_patterns);     // Dim: num_patterns x (num_cells * ssb_nItrsPerStrm)
        std::vector<std::vector<std::string>> inFileNamesCSIRS(num_patterns);   // Dim: num_patterns x (num_cells * csirs_nItrsPerStrm)
        std::vector<std::vector<std::string>> inFileNamesMac(num_patterns);     // Dim: num_patterns x (num_cells * nMacSlots)
        std::vector<std::vector<std::string>> inFileNamesMac2(num_patterns);    // Dim: num_patterns x (num_cells * nMac2Slots)

        std::vector<bool> runSRSVec(num_patterns, false);
        std::vector<bool> runPRACHVec(num_patterns, false);
        std::vector<bool> runPUSCHVec(num_patterns, false);
        std::vector<bool> runPDSCHVec(num_patterns, false);
        std::vector<bool> runPDCCHVec(num_patterns, false);
        std::vector<bool> runPUCCHVec(num_patterns, false);
        std::vector<bool> runULBFWVec(num_patterns, false);
        std::vector<bool> runDLBFWVec(num_patterns, false);
        std::vector<bool> runSSBVec(num_patterns, false);
        std::vector<bool> runCSIRSVec(num_patterns, false);
        std::vector<bool> runMACVec(num_patterns, false);
        std::vector<bool> runMAC2Vec(num_patterns, false);

        std::vector<uint32_t> patternMode(num_patterns, 0); // not u5 by default

        int leastPriority, greatestPriority;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

#ifndef ENABLE_F01_STREAM_PRIO
        if(enableHighPdschCuStrmPrio && (4 == uldl))
        {
            NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: F01 mode with priorities is not supported");
            exit(1);
        }
#endif

        std::map<std::string, int> cuStrmPrioMap;
        // No effect since the priorities are overridden by the ones specified in the yaml file
        // if(enableHighPdschCuStrmPrio)
        // {
        //     cuStrmPrioMap["PUSCH"]  = greatestPriority + 1 + prachCtx;
        //     cuStrmPrioMap["PUSCH2"] = greatestPriority + 2 + prachCtx;
        //     cuStrmPrioMap["PUCCH"]  = greatestPriority + 1 + prachCtx;
        //     cuStrmPrioMap["PUCCH2"] = greatestPriority + 2 + prachCtx;
        //     cuStrmPrioMap["PDSCH"]  = greatestPriority + prachCtx;
        //     cuStrmPrioMap["PDCCH"]  = greatestPriority + prachCtx;
        //     cuStrmPrioMap["CSIRS"]  = greatestPriority + prachCtx;
        //     cuStrmPrioMap["SRS"]    = greatestPriority + 3 + prachCtx;
        //     cuStrmPrioMap["PRACH"]  = greatestPriority + 4 * (!prachCtx);
        //     cuStrmPrioMap["SSB"]    = greatestPriority + 1 + prachCtx;
        //     cuStrmPrioMap["MAC"]    = greatestPriority + 1 + prachCtx;
        //     cuStrmPrioMap["MAC2"]   = greatestPriority + 1 + prachCtx;
        // }

        // try reading priorities from yaml file; the priorities there are all relative to this
        // GPU's greatest priority (reminder greatest priority is negative, e.g., -5)
        {
            yaml::file_parser fp(inputFileName.c_str());
            yaml::document    d = fp.next_document();
            yaml::node        r = d.root();
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // number of cells (scalar)
            try
            {
                cuStrmPrioMap["PUSCH"] = greatestPriority + r["PUSCH_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["PUSCH2"] = greatestPriority + r["PUSCH2_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["PUCCH"] = greatestPriority + r["PUCCH_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["PUCCH2"] = greatestPriority + r["PUCCH2_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["ULBFW"] = greatestPriority + r["ULBFW_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["PDSCH"] = greatestPriority + r["PDSCH_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["DLBFW"] = greatestPriority + r["DLBFW_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["PDCCH"] = greatestPriority + r["PDCCH_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["SRS"] = greatestPriority + r["SRS_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["PRACH"] = greatestPriority + r["PRACH_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["SSB"] = greatestPriority + r["SSB_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["CSIRS"] = greatestPriority + r["CSIRS_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["MAC"] = greatestPriority + r["MAC_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["MAC2"] = greatestPriority + r["MAC2_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
        }

        // parse maximum parameters for memory allocation
        maxPDSCHPrms pdschPrms;
        maxPUSCHPrms puschPrms;

        pdschPrms.maxNTbs      = 0;
        pdschPrms.maxNCbs      = 0;
        pdschPrms.maxNCbsPerTb = 0;
        pdschPrms.maxNPrbs     = 0;
        pdschPrms.maxNTx       = 0;

        puschPrms.maxNTbs          = 0;
        puschPrms.maxNCbs          = 0;
        puschPrms.maxNCbsPerTb     = 0;
        puschPrms.maxNPrbs         = 0;
        puschPrms.maxNRx           = 0;
        puschPrms.maxNCellsPerSlot = 1; // no cell-groups by default

        yaml::file_parser fp(inputFileName.c_str());
        yaml::document    d           = fp.next_document();
        yaml::node        r           = d.root();
        bool              prmsPresent = false;
        try
        {
            r           = r["parameters"];
            prmsPresent = true;
        }
        catch(...)
        {
            NVLOGW_FMT(NVLOG_TAG_BASE_TESTBENCH, "WARNING: Parameters field not found in YAML");
        }

        if(prmsPresent)
        {
            try
            {
                yaml::node p = r["PDSCH"];
                try
                {
                    pdschPrms.maxNTbs = group_pdsch_cells ? p["Max #TB per slot"] : p["Max #TB per slot per cell"];
                }
                catch(...)
                {
                    pdschPrms.maxNTbs = 0;
                }
                try
                {
                    pdschPrms.maxNCbs = group_pdsch_cells ? p["Max #CB per slot"] : p["Max #CB per slot per cell"];
                }
                catch(...)
                {
                    pdschPrms.maxNCbs = 0;
                }
                try
                {
                    pdschPrms.maxNCbsPerTb = p["Max #CB per slot per cell per TB"];
                }
                catch(...)
                {
                    pdschPrms.maxNCbsPerTb = 0;
                }
                try
                {
                    pdschPrms.maxNPrbs = p["Max #PRB per cell"];
                }
                catch(...)
                {
                    pdschPrms.maxNPrbs = 0;
                }
                try
                {
                    pdschPrms.maxNTx = p["Max #TX per cell"];
                }
                catch(...)
                {
                    pdschPrms.maxNTx = 0;
                }
            }
            catch(...)
            {
                ("WARNING: Parameters / PDSCH field not found in YAML\n");
            }

            try
            {
                yaml::node p = r["PUSCH"];
                try
                {
                    puschPrms.maxNTbs = group_pusch_cells ? p["Max #TB per slot"] : p["Max #TB per slot per cell"];
                }
                catch(...)
                {
                    puschPrms.maxNTbs = 0;
                }
                try
                {
                    puschPrms.maxNCbs = group_pusch_cells ? p["Max #CB per slot"] : p["Max #CB per slot per cell"];
                }
                catch(...)
                {
                    puschPrms.maxNCbs = 0;
                }
                try
                {
                    puschPrms.maxNCbsPerTb = p["Max #CB per slot per cell per TB"];
                }
                catch(...)
                {
                    puschPrms.maxNCbsPerTb = 0;
                }
                try
                {
                    puschPrms.maxNPrbs = p["Max #PRB per cell"];
                }
                catch(...)
                {
                    puschPrms.maxNPrbs = 0;
                }
                try
                {
                    puschPrms.maxNRx = p["Max #RX per cell"];
                }
                catch(...)
                {
                    puschPrms.maxNRx = 0;
                }
                try
                {
                    puschSubslotProcFlag[0] = int(p["PUSCH subslot proc flag"]);
                }
                catch(...)
                {
                    puschSubslotProcFlag[0] = 0;
                }
                try
                {
                    puschSubslotProcFlag[1] = int(p["PUSCH2 subslot proc flag"]);
                }
                catch(...)
                {
                    puschSubslotProcFlag[1] = 0;
                }
            }
            catch(...)
            {
                ("WARNING: Parameters / PUSCH field not found in YAML\n");
            }
        }

        // dl slot config and iters per strm only configured by the TVs in the first pattern; assuming all patterns has the same number of workloads per channel
        uint8_t pdschSlotRunFlag[nSlotsPerPattern] = {0};
        uint8_t pdcchSlotRunFlag[nSlotsPerPattern] = {0};
        uint8_t csirsSlotRunFlag[nSlotsPerPattern] = {0};
        uint8_t pbchSlotRunFlag[nSlotsPerPattern] = {0};
        uint8_t macSlotRunFlag[nSlotsPerPattern] = {0};
        
        ssb_nItrsPerStrm = 0;
        prach_nItrsPerStrm = 0;
        pusch_nItrsPerStrm = 0;
        pucch_nItrsPerStrm = 0;
        pdsch_nItrsPerStrm = 0;
        dlbfw_nItrsPerStrm = 0;
        ulbfw_nItrsPerStrm = 0;
        pdcch_nItrsPerStrm = 0;
        csirs_nItrsPerStrm = 0;
        nMacSlots = 0;
        nMac2Slots = 0;

        // read channels and TVs from yaml file
        for(uint32_t patternIdx = 0; patternIdx < num_patterns; patternIdx++)
        {
            nSRSCells        = 0;
            nPRACHCells      = 0;
            nPDSCHCells      = 0;
            nPUSCHCells      = 0;
            nPUCCHCells      = 0;
            nSSBCells        = 0;
            nCSIRSCells      = 0;
            nMACWorkers      = 0;
            nMAC2Workers     = 0;
            nUlbfwCells      = 0;
            nDlbfwCells      = 0;
            
            for(uint32_t slotIdx = patternIdx * nSlotsPerPattern; slotIdx < (patternIdx + 1) * nSlotsPerPattern; slotIdx++)
            {
                std::map<std::string, std::vector<std::string>> currentSlotTVs = testCfg.slots()[slotIdx];

                // read the channel if exists
                for (auto it = currentSlotTVs.begin(); it != currentSlotTVs.end(); it++)
                {
                    switch (chNameMap[it -> first])
                    // std::unordered_map<std::string, int> chNameMap = {{"PUSCH", 0}, {"PDSCH", 1}, {"PDCCH", 2}, {"PUCCH", 3}, {"PRACH", 4}, {"DLBFW", 5}, {"ULBFW", 6}, {"SSB", 7}, {"CSIRS", 8}, {"SRS", 9}, {"MAC", 10}, {"MAC2", 11}};
                    {
                    case 0: // chNameMap["PUSCH"]
                        try
                        {
                            nPUSCHCells = currentSlotTVs.at(puschChannelName).size();
                            if(nPUSCHCells == 0)
                            {
                                throw std::out_of_range("PUSCH #cells == 0");
                            }
                            runPUSCHVec[patternIdx] = true;
                            if(patternIdx == 0)
                                pusch_nItrsPerStrm++;

                            for(uint32_t cellIdx = 0; cellIdx < nPUSCHCells; ++cellIdx)
                            {
                                std::string pusch_tv_filename = currentSlotTVs.at(puschChannelName)[cellIdx];
                                inFileNamesPuschRx[patternIdx].emplace_back(pusch_tv_filename);
                            }
                        }
                        catch(const std::out_of_range& ex)
                        {
                            runPUSCHVec[patternIdx] = false;
                            printf("NO PUSCH detected for pattern %d\n", patternIdx);
                        }
                        break;

                    case 1: // chNameMap["PDSCH"]
                        try
                        {
                            nPDSCHCells = currentSlotTVs.at(pdschChannelName).size();
                            if(nPDSCHCells == 0)
                            {
                                throw std::out_of_range("PDSCH #cells == 0");
                            }
                            runPDSCHVec[patternIdx] = true;
                            if(patternIdx == 0)
                            {
                                pdsch_nItrsPerStrm++;
                                pdschSlotRunFlag[slotIdx] = 1; // only set slot run flag at pattern 0
                            }

                            // PDSCH TVs
                            for(int cellIdx = 0; cellIdx < nPDSCHCells; ++cellIdx)
                            {
                                std::string pdsch_tv_filename = currentSlotTVs.at(pdschChannelName)[cellIdx];
                                inFileNamesPdschTx[patternIdx].emplace_back(pdsch_tv_filename);
                            }
                        }
                        catch(const std::out_of_range& ex)
                        {
                            runPDSCHVec[patternIdx] = false;
                            printf("NO PDSCH detected for pattern %d\n", patternIdx);
                        }
                        break;

                    case 2: // chNameMap["PDCCH"]
                        try
                        {
                            nPDCCHCells = currentSlotTVs.at(pdcchChannelName).size();
                            if(nPDCCHCells == 0)
                            {
                                throw std::out_of_range("PDCCH #cells == 0");
                            }
                            runPDCCHVec[patternIdx] = true;
                            if(patternIdx == 0)
                            {
                                pdcch_nItrsPerStrm++;
                                pdcchSlotRunFlag[slotIdx] = 1; // only set slot run flag at pattern 0
                            }

                            for(int cellIdx = 0; cellIdx < nPDCCHCells; ++cellIdx)
                            {
                                std::string pdcch_tv_filename = currentSlotTVs.at(pdcchChannelName)[cellIdx];
                                inFileNamesPdcchTx[patternIdx].emplace_back(pdcch_tv_filename);
                            }
                        }
                        catch(const std::out_of_range& ex)
                        {
                            runPDCCHVec[patternIdx] = false;
                            printf("NO PDCCH detected for pattern %d\n", patternIdx);
                        }
                        
                        if(pdcchCtx && nPDCCHCells == 0)
                        {
                            printf("ERROR: PDCCH context enabled with --Q but no PDCCH test vectors in yaml for pattern %d\n", patternIdx);
                            exit(-1);
                        }
                        break;

                    case 3: // chNameMap["PUCCH"]
                        try
                        {
                            nPUCCHCells = currentSlotTVs.at(pucchChannelName).size();
                            if(nPUCCHCells == 0)
                            {
                                throw std::out_of_range("PUCCH #cells == 0");
                            }
                            runPUCCHVec[patternIdx] = true;
                            if(patternIdx == 0)
                                pucch_nItrsPerStrm++;

                            for(uint32_t cellIdx = 0; cellIdx < nPUCCHCells; ++cellIdx)
                            {
                                std::string pucch_tv_filename = currentSlotTVs.at(pucchChannelName)[cellIdx];
                                inFileNamesPucchRx[patternIdx].emplace_back(pucch_tv_filename);
                            }
                        }
                        catch(const std::out_of_range& ex)
                        {
                            runPUCCHVec[patternIdx] = false;
                            printf("NO PUCCH detected for pattern %d\n", patternIdx);

                            if(pucchCtx && nPUCCHCells == 0)
                            {
                                printf("ERROR: PUCCH context enabled with --X but no PUCCH test vectors in yaml for pattern %d\n", patternIdx);
                                exit(-1);
                            }
                        }
                        break;

                    case 4: // chNameMap["PRACH"]
                        try
                        {
                            nPRACHCells = currentSlotTVs.at(prachChannelName).size();
                            if(nPRACHCells == 0)
                            {
                                throw std::out_of_range("PRACH #cells == 0");
                            }
                            runPRACHVec[patternIdx] = true;
                            if(patternIdx == 0)
                                prach_nItrsPerStrm++;

                            // prach TVs
                            for(uint32_t cellIdx = 0; cellIdx < nPRACHCells; ++cellIdx)
                            {
                                std::string prach_tv_filename = currentSlotTVs.at(prachChannelName)[cellIdx];
                                inFileNamesPRACH[patternIdx].emplace_back(prach_tv_filename);
                            }
                        }
                        catch(const std::out_of_range& ex)
                        {
                            runPRACHVec[patternIdx] = false;
                            printf("NO PRACH detected for pattern %d\n", patternIdx);
                        }

                        if(prachCtx && nPRACHCells == 0)
                        {
                            printf("ERROR: PRACH context enabled with --P but no PRACH test vectors in yaml for pattern %d\n", patternIdx);
                            exit(-1);
                        }
                        break;

                    case 5: // chNameMap["DLBFW"]
                        try
                        {
                            nDlbfwCells = currentSlotTVs.at(dlbfwChannelName).size();
                            if(nDlbfwCells == 0)
                            {
                                throw std::out_of_range("DL BFW #cells == 0");
                            }
                            runDLBFWVec[patternIdx] = true;
                            if(patternIdx == 0)
                                dlbfw_nItrsPerStrm++;

                            if(runDLBFWVec[patternIdx])
                            {
                                for(int cellIdx = 0; cellIdx < nDlbfwCells; ++cellIdx)
                                {
                                    std::string dlbfw_tv_filename = currentSlotTVs.at(dlbfwChannelName)[cellIdx];
                                    inFileNamesDlbfw[patternIdx].emplace_back(dlbfw_tv_filename);
                                }
                            }
                        }
                        catch(const std::out_of_range& ex)
                        {
                            printf("NO DL BFW detected for pattern %d\n", patternIdx);
                        }
                        break;

                    case 6: // chNameMap["ULBFW"]
                        try
                        {
                            nUlbfwCells = currentSlotTVs.at(ulbfwChannelName).size();
                            if(nUlbfwCells == 0)
                            {
                                throw std::out_of_range("UL BFW #cells == 0");
                            }
                            runULBFWVec[patternIdx] = true;
                            if(patternIdx == 0)
                                ulbfw_nItrsPerStrm++;

                            for(int cellIdx = 0; cellIdx < nUlbfwCells; ++cellIdx)
                            {
                                std::string ulbfw_tv_filename = currentSlotTVs.at(ulbfwChannelName)[cellIdx];
                                inFileNamesUlbfw[patternIdx].emplace_back(ulbfw_tv_filename);
                            }
                        }
                        catch(const std::out_of_range& ex)
                        {
                            printf("NO UL BFW detected for pattern %d\n", patternIdx);
                        }
                        break;

                    case 7: //chNameMap["SSB"]
                        try
                        {
                            nSSBCells = currentSlotTVs.at(ssbChannelName).size();
                            if(nSSBCells == 0)
                            {
                                throw std::out_of_range("SSB #cells == 0");
                            }
                            runSSBVec[patternIdx] = true;
                            if(patternIdx == 0)
                            {
                                ssb_nItrsPerStrm++;
                                pbchSlotRunFlag[slotIdx] = 1; // only set slot run flag at pattern 0
                            }

                            // ssb TVs
                            for(uint32_t cellIdx = 0; cellIdx < nSSBCells; ++cellIdx)
                            {
                                std::string ssb_tv_filename = currentSlotTVs.at(ssbChannelName)[cellIdx];
                                inFileNamesSSB[patternIdx].emplace_back(ssb_tv_filename);
                            }
                        }
                        catch(const std::out_of_range& ex)
                        {
                            runSSBVec[patternIdx] = false;
                            printf("NO SSB detected for pattern %d\n", patternIdx);
                        }
                        break;

                    case 8: // chNameMap["CSIRS"]
                        try
                        {
                            nCSIRSCells = currentSlotTVs.at(csirsChannelName).size();
                            if(nCSIRSCells == 0)
                            {
                                throw std::out_of_range("CSIRS #cells == 0");
                            }
                            runCSIRSVec[patternIdx] = true;
                            if(patternIdx == 0)
                            {
                                csirs_nItrsPerStrm++;
                                csirsSlotRunFlag[slotIdx] = 1; // only set slot run flag at pattern 0
                            }

                            for(int cellIdx = 0; cellIdx < nCSIRSCells; ++cellIdx)
                            {
                                std::string csirs_tv_filename = currentSlotTVs.at(csirsChannelName)[cellIdx];
                                inFileNamesCSIRS[patternIdx].emplace_back(csirs_tv_filename);
                            }
                        }
                        catch(const std::out_of_range& ex)
                        {
                            runCSIRSVec[patternIdx] = false;
                            printf("NO CSIRS detected for pattern %d\n", patternIdx);
                        }
                        break;

                    case 9: // chNameMap["SRS"]
                        try
                        {
                            nSRSCells = currentSlotTVs.at(srsChannelName).size();
                            if(nSRSCells == 0)
                            {
                                throw std::out_of_range("SRS #cells == 0");
                            }
                            runSRSVec[patternIdx] = true;
                            if(patternIdx == 0)
                                srs_nItrsPerStrm++;

                            // srs TVs
                            for(uint32_t cellIdx = 0; cellIdx < nSRSCells; ++cellIdx)
                            {
                                std::string srs_tv_filename = currentSlotTVs.at(srsChannelName)[cellIdx];
                                inFileNamesSRS[patternIdx].emplace_back(srs_tv_filename);
                            }
                        }
                        catch(const std::out_of_range& ex)
                        {
                            runSRSVec[patternIdx] = false;
                            NVLOGI_FMT(NVLOG_SRS, "NO SRS detected for pattern {}", patternIdx);
                            if(srsCtx && nSRSCells == 0)
                            {
                                NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: SRS context enabled with --Z but no SRS test vectors in yaml for pattern {}", patternIdx);
                                exit(-1);
                            }
                        }
                        break;

                    case 10: // chNameMap["MAC"]
                        try
                        {
                            nMACWorkers = currentSlotTVs.at(macChannelName).size(); // TODO: currenty one cuMAC TV contains the channel for all cells, thus nMACWorkers = 1 and only one cuMAC worker is created
                            if(nMACWorkers == 0)
                            {
                                throw std::out_of_range("MAC #cells == 0");
                            }
                            else if(nMACWorkers > 1)
                            {
                                throw std::out_of_range("MAC #cells > 1");
                            }
                            runMACVec[patternIdx] = true;
                            if(patternIdx == 0)
                            {
                                nMacSlots++;
                                macSlotRunFlag[slotIdx] = 1; // only set slot run flag at pattern 0
                            }

                            for(int macWrkrIdx = 0; macWrkrIdx < nMACWorkers; ++macWrkrIdx)
                            {
                                std::string mac_tv_filename = currentSlotTVs.at(macChannelName)[macWrkrIdx]; // cuMAC TV should be only 1 that contains the channels for all TVs
                                inFileNamesMac[patternIdx].emplace_back(mac_tv_filename);
                            }
                        }
                        catch(const std::out_of_range& ex)
                        {
                            runMACVec[patternIdx] = false;
                            printf("MAC TV error %s for pattern %d\n", ex.what(), patternIdx);
                            if(macCtx && nMACWorkers == 0)
                            {
                                NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: MAC context enabled with --T but no MAC test vectors in yaml for pattern {}", patternIdx);
                                exit(-1);
                            }
                        }
                        break;

                    case 11: // chNameMap["MAC2"]
                        try
                        {
                            nMAC2Workers = currentSlotTVs.at(mac2ChannelName).size(); // TODO: currenlty one cuMAC TV contains the channel for all cells, thus nMAC2Workers = 1 and only one cuMAC worker is created for MAC2
                            if(nMAC2Workers == 0)
                            {
                                throw std::out_of_range("MAC2 #cells == 0");
                            }
                            else if(nMAC2Workers > 1)
                            {
                                throw std::out_of_range("MAC2 #cells > 1");
                            }
                            runMAC2Vec[patternIdx] = true;
                            if(patternIdx == 0)
                            {
                                nMac2Slots++;
                                macSlotRunFlag[slotIdx] = 1; // only set slot run flag at pattern 0 (shared with MAC)
                            }

                            for(int mac2WrkrIdx = 0; mac2WrkrIdx < nMAC2Workers; ++mac2WrkrIdx)
                            {
                                std::string mac2_tv_filename = currentSlotTVs.at(mac2ChannelName)[mac2WrkrIdx]; // cuMAC TV should be only 1 that contains the channels for all TVs
                                inFileNamesMac2[patternIdx].emplace_back(mac2_tv_filename);
                            }
                        }
                        catch(const std::out_of_range& ex)
                        {
                            runMAC2Vec[patternIdx] = false;
                            printf("MAC2 TV error %s for pattern %d\n", ex.what(), patternIdx);

                            if(mac2Ctx && nMAC2Workers == 0)
                            {
                                NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "ERROR: MAC2 context enabled with --V but no MAC2 test vectors in yaml for pattern {}", patternIdx);
                                exit(-1);
                            }
                        }
                        break;

                    default:
                        NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "Error! Unknown channel name {} for input TV", it -> first);
                        exit(1);
                    }
                }
            }

            // set patternMode based on input arguments for u3 and u5
            // patternMode[*] = 0 for u4
            if(uldl == 6)
            {
                if(mode == 0) // mode A, PUSCH2 do not need to wait until PUSCH1 ends, only wait for delay ends
                {
                    patternMode[patternIdx] = 7; 
                    printf("Pattern %d: DDDSUUDDDD mode C0\n", patternIdx);
                }
                else if(mode == 1) // mode B, PUSCH2 must wait until PUSCH1 ends and delay stops
                {
                    patternMode[patternIdx] = 8;
                    printf("Pattern %d: DDDSUUDDDD mode C1\n", patternIdx);
                }
            }
            else if(uldl == 5)
            {             
                if(mode == 0) // mode A
                {
                    if(runULBFWVec[patternIdx] || runDLBFWVec[patternIdx])
                    {
                        patternMode[patternIdx] = 3;
                        printf("Pattern %d: DDDSUUDDDD mode A2\n", patternIdx);
                    }
                    else if(runPDCCHVec[patternIdx] || runPRACHVec[patternIdx])
                    {
                        patternMode[patternIdx] = 2;
                        printf("Pattern %d: DDDSUUDDDD mode A1\n", patternIdx);
                    }
                    else
                    {
                        printf("Pattern %d: DDDSUUDDDD mode A0\n", patternIdx);
                        patternMode[patternIdx] = 1;
                    }
                }
                else if(mode == 1) // mode B
                {
                    if(runULBFWVec[patternIdx] || runDLBFWVec[patternIdx])
                    {
                        patternMode[patternIdx] = 6;
                        printf("Pattern %d: DDDSUUDDDD mode B2\n", patternIdx);
                    }
                    else if(runPDCCHVec[patternIdx] || runPRACHVec[patternIdx])
                    {
                        patternMode[patternIdx] = 5;
                        printf("Pattern %d: DDDSUUDDDD mode B1\n", patternIdx);
                    }
                    else
                    {
                        patternMode[patternIdx] = 4;
                        printf("Pattern %d: DDDSUUDDDD mode B0\n", patternIdx);
                    }
                }
            }
            else if(uldl == 3)
            {
                if(runPDCCHVec[patternIdx])
                {
                    if(runULBFWVec[patternIdx] || runDLBFWVec[patternIdx])
                        printf("Pattern %d: DDDSU mode A5\n", patternIdx);
                    else
                        printf("Pattern %d: DDDSU mode A4\n", patternIdx);
                }
                else
                {
                    if(runULBFWVec[patternIdx] || runDLBFWVec[patternIdx])
                    {
                        if(runPRACHVec[patternIdx])
                            printf("Pattern %d: DDDSU mode A3\n", patternIdx);
                        else
                            printf("Pattern %d: DDDSU mode A2\n", patternIdx);
                    }
                    else
                    {
                        if(runPRACHVec[patternIdx])
                            printf("Pattern %d: DDDSU mode A1\n", patternIdx);
                        else
                            printf("Pattern %d: DDDSU mode A0\n", patternIdx);
                    }
                }
            }
        }

        // Beamforming
        dlbfw_nItrsPerStrm      = pdsch_nItrsPerStrm;
        ulbfw_nItrsPerStrm      = pusch_nItrsPerStrm;

        nPdschCellsPerPattern   = cells_per_slot * pdsch_nItrsPerStrm;
        nPuschCellsPerPattern   = cells_per_slot * pusch_nItrsPerStrm;
        if(uldl == 4)
            nPschCellsPerPattern  = cells_per_slot; // in u4, pdsch_nItrsPerStrm = 1, pusch_nItrsPerStrm = 1, 

        //---------------------------------------------------------------------------
        // worker run configurations

        if(uldl == 4)
        {
            // use a default cfg if not specified at input
            if(nStrmsPerCtxt == 0)
            {
                nStrmsPerCtxt    = cells_per_slot / nCtxts;
                nPschItrsPerStrm = 1;
            }

            nPdschCellsPerStrm         = group_pdsch_cells ? nStrmsPerCtxt : 1;
            nPdschStrmsPerCtxt         = group_pdsch_cells ? 1 : nStrmsPerCtxt;
            nPuschStrmsPerCtxt         = group_pusch_cells ? 1 : nStrmsPerCtxt;
            puschPrms.maxNCellsPerSlot = group_pusch_cells ? nStrmsPerCtxt : 1;
            // check that run configuration is legal
            if((nCtxts * nStrmsPerCtxt * nPschItrsPerStrm) != nPschCellsPerPattern)
            {
                NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "Error! nCtxts {} * nStrmsPerCtxt {} * nPschItrsPerStrm {} must be equal to number of cells {}", nCtxts, nStrmsPerCtxt, nPschItrsPerStrm, nPschCellsPerPattern);
                exit(1);
            }
        }

        if(uldl != 4)
        {
            nPdschCellsPerStrm         = group_pdsch_cells ? nPDSCHCells : 1;
            nPdcchCellsPerStrm         = group_pdsch_cells ? nPDCCHCells : 1; // group_pdsch_cells used intentionally
            nPrachCellsPerStrm         = group_pusch_cells ? nPRACHCells : 1;
            nStrmsPerCtxt              = cells_per_slot;
            nPdschStrmsPerCtxt         = group_pdsch_cells ? 1 : nPDSCHCells;
            nSSBStrmsPerCtxt           = group_pdsch_cells ? 1 : nSSBCells;
            nPuschStrmsPerCtxt         = group_pusch_cells ? (uldl == 5 || uldl == 6) ? 2 : 1 : cells_per_slot;
            puschPrms.maxNCellsPerSlot = group_pusch_cells ? cells_per_slot : 1;
        }
        if(group_pdsch_cells && (nPdschCellsPerStrm > PDSCH_MAX_CELLS_PER_CELL_GROUP))
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Error! nPdschCellsPerStrm {} > PDSCH_MAX_CELLS_PER_CELL_GROUP {}. Update the max. limit in the header file", nPdschCellsPerStrm, PDSCH_MAX_CELLS_PER_CELL_GROUP);
            exit(1);
        }

        if (uldl == 5 || uldl == 6) // for long pattern, we use separate streams for PUSCH/PUCCH and PUSCH2/PUCCH2 workloads
        {
            pusch_nItrsPerStrm /= 2;
            pucch_nItrsPerStrm /= 2;
        }
        nMacItrsPerStrm = nMacSlots / nStrmsMac;
        nMac2ItrsPerStrm = nMac2Slots / nStrmsMac2;
        // mac timer configurations
        // if enableMacInternalTimer = true, try to use the timer in first cuMAC subcontext, this reduces the overhead introduced by timers
        // if cuPHY is run, recommend to use the timer from PDSCH worker (without --R)
        bool macInternalTimer  = enableMacInternalTimer && macCtx;
        bool mac2InternalTimer = enableMacInternalTimer && (!macCtx) && mac2Ctx; // only if macCtx is disabled but mac2Ctx is enabled
        //---------------------------------------------------------------------------
        // Partition GPU and CPU

        if(nCtxts > CURRENT_MAX_GREEN_CTXS)
        {
            printf("ERROR! Update MAX_GREEN_CTX_POSSIBLE to at least %d (number of contexts)\n", nCtxts);
            exit(-1);
        }

        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, gpuId));

#if CUDA_VERSION >= 12040
        int mpsEnabled = 0;
        CU_CHECK(cuDeviceGetAttribute(&mpsEnabled, CU_DEVICE_ATTRIBUTE_MPS_ENABLED, device));
        if (useGreenContexts) {
            if (mpsEnabled == 1) {
                NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "MPS is enabled. Heads-up that currently using green contexts with MPS enabled can have unintended side effects. Will run regardless.\n");
                //exit(1);
            } else {
                NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "MPS service is not running.");
            }
        }
#endif

        int32_t gpuMaxSmCount = 0;
        CU_CHECK(cuDeviceGetAttribute(&gpuMaxSmCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));

        if(heterogenousMpsPartitions)
        {
            if(nCtxts != ctxSmCounts.size())
            {
                NVLOGE_FMT(NVLOG_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "Number of input MPS sub-context SM counts ({}) does not match # of worker instances {}", ctxSmCounts.size(), nCtxts);
                exit(1);
            }

            bool     validSmCount = true;
            uint32_t subCtxIdx    = 0;
            std::for_each(ctxSmCounts.begin(), ctxSmCounts.end(), [&gpuMaxSmCount, &subCtxIdx, &validSmCount](auto& mpsSubCtxSmCount) {
                if(mpsSubCtxSmCount > gpuMaxSmCount)
                {
                    validSmCount = false;
                    NVLOGE_FMT(NVLOG_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "SM count ({}) for sub-context {} exceeds device limit ({})\n", mpsSubCtxSmCount, subCtxIdx, gpuMaxSmCount);
                }
                subCtxIdx++;
            });
            if(!validSmCount)
            {
                exit(1);
            }
        }
        else if(!heterogenousMpsPartitions)
        {
            ctxSmCounts.clear();
            for(uint32_t ctxIdx = 0; ctxIdx < nCtxts; ++ctxIdx)
            {
                ctxSmCounts.emplace_back(gpuMaxSmCount);
            }
        }

        // GPU and CPU Ids
        std::vector<int> gpuIds(nCtxts, gpuId);
        std::vector<int> cpuIds(nCtxts, baseCpuId);
        if(enableMultiCore) {
            for(int ii=0; ii<nCtxts; ii++) {
                int current_core = baseCpuId + ii;
                int max_core = nMaxConcurrentThrds-1;
                cpuIds[ii] = std::min(current_core, max_core);
                if(current_core > max_core) {
                    NVLOGW_FMT(NVLOG_TESTBENCH, "WARNING!  Not enough cores available to fit thread {}.  Aliasing cpu {}",current_core,max_core);
                }
            }
        }

        int              cpuThrdSchdPolicy = SCHED_RR; // SCHED_FIFO // SCHED_RR
        int              cpuThrdPrio       = sched_get_priority_max(cpuThrdSchdPolicy);
        std::vector<int> cpuThrdPrios(nCtxts, cpuThrdPrio);

        // Main cuda stream and events for synchronizing GPU work
        cuphy::stream mainStream(cudaStreamNonBlocking);

        // Power test bench operates in two distinct modes controlled by CUBB_GPU_TESTBENCH_POWER_ITERATION_BATCHING macro:
        //
        // BATCHING MODE (macro defined):
        //   - GPU commands are batched and ungated every syncUpdateIntervalCnt iterations
        //   - Periodically synchronize CPU-GPU execution every syncUpdateIntervalCnt iterations to prevent CPU being too far ahead of the GPU
        uint32_t syncUpdateIntervalCnt = graphs_mode ? 20 : 5;
        //
        // NON-BATCHING MODE (macro undefined):
        //   - GPU commands for next pattern launched immediately during current pattern execution
        //   - CPU launches commands during the middle of current pattern (either PDSCH event or PUSCH event)
        //   - Continuous pipeline operation with overlapped CPU/GPU work
        //
        if(0 != nPowerItrs)
        {
            nPowerItrs  = (nPowerItrs >= 100) ? nPowerItrs : 100;
            nTimingItrs = 1; // Force timing iterations to 1 for power measurements
        }
        // Create and initialize (clear) CPU to GPU start sync flag
        auto shPtrGpuStartSyncFlag  = std::make_shared<cuphy::buffer<uint32_t, cuphy::pinned_alloc>>(1);
        (*shPtrGpuStartSyncFlag)[0] = 0;

        // Command/Response queues for communication between main (orchestration) thread and worker thread(s)
        std::vector<sharedPtrTestWrkrCmdQ> cmdQVec;
        std::vector<sharedPtrTestWrkrRspQ> rspQVec;

        for(uint32_t ctxIdx = 0; ctxIdx < nCtxts; ++ctxIdx)
        {
            std::string commandStr;
            commandStr = "CommandQueue" + std::to_string(ctxIdx);
            cmdQVec.emplace_back(std::make_shared<testWrkrCmdQ>(commandStr.c_str()));

            std::string responseStr;
            responseStr = "ResponseQueue" + std::to_string(ctxIdx);
            rspQVec.emplace_back(std::make_shared<testWrkrRspQ>(responseStr.c_str()));
        }

        //---------------------------------------------------------------------------
        // Timing objects
        cuphy::event_timer slotPatternTimer;
        float              tot_slotPattern_time    = 0;
        float              avg_slotPattern_time_us = 0;

        //--------------------------------------------------------------------------
        // Print simulation setup

        // printf("\n ----------------------------------------------------------");
        // printf("\n Notes on  simulation setup:");
        // printf("\n--> %d cells", cells_per_slot);
        // if(uldl == 3){
        //     printf("\n--> TDD slot pattern. Consists of %d PDSCH and %d PUSCH slots", nPdschCellsPerPattern, nPuschCellsPerPattern);
        // }else{
        //     printf("\n--> FDD slot pattern. Consists of 1 PUSCH + PDSCH slot");
        // }


        //----------\--------------------------------------------------
        //PDSCH + PUSCH
        if((uldl == 3) || (uldl == 5) || (uldl == 6))
        {
            //------------------------------------------------------------------------------------
            // create PHY workers
            // uint32_t ctxtIdx = 0;
            //cuphyTestWorkerVec.reserve(nCtxts); // needed to avoid call to copy constructor
            std::vector<std::string> phyTestWorkerStringVec;
            //= {"PdschTxTestWorker", "PuschRxTestWorker", "PrachTestWorker", "PdcchTestWorker", "PucchTestWorker"};

            std::unordered_map<std::string, int> channelWorkerMap;

            int c = 0; // index of subcontexts for PHY workers
            if(prachCtx)
            {
                channelWorkerMap.insert({"PRACH", c});
                phyTestWorkerStringVec.push_back("PrachTestWorker");
                c++;
            }

            if(pdcchCtx)
            {
                channelWorkerMap.insert({"PDCCH", c});
                phyTestWorkerStringVec.push_back("PdcchTestWorker");
                if(!ssbCtx)
                    channelWorkerMap.insert({"SSB", c});
                c++;
            }

            if(pucchCtx)
            {
                channelWorkerMap.insert({"PUCCH", c});
                phyTestWorkerStringVec.push_back("PucchTestWorker");
                c++;
            }

            if(pdschCtx)
            {
                channelWorkerMap.insert({"PDSCH", c});
                phyTestWorkerStringVec.push_back("PdschTxTestWorker");
                if(!pdcchCtx)
                    channelWorkerMap.insert({"PDCCH", c});
                if(!pdcchCtx && !ssbCtx)
                    channelWorkerMap.insert({"SSB", c});
                c++;
            }

            if(puschCtx)
            {
                channelWorkerMap.insert({"PUSCH", c});
                phyTestWorkerStringVec.push_back("PuschRxTestWorker");
                if(!prachCtx)
                    channelWorkerMap.insert({"PRACH", c});
                if(!pucchCtx)
                    channelWorkerMap.insert({"PUCCH", c});
                if(!srsCtx)
                    channelWorkerMap.insert({"SRS", c});
                c++;
            }

            if(ssbCtx)
            {
                channelWorkerMap.insert({"SSB", c});
                phyTestWorkerStringVec.push_back("SsbTestWorker");
                c++;
            }

            if (srsCtx)
            {
                channelWorkerMap.insert({"SRS", c});
                phyTestWorkerStringVec.push_back("SrsTestWorker");
                c++;
            }

            // create MAC workers
            std::vector<std::string> macTestWorkerStringVec;
            std::unordered_map<std::string, int> macWorkerMap;

            int mac_c = 0; // index of subcontexts for PHY workers

            if(macCtx)
            {
                macWorkerMap.insert({"MAC", mac_c});
                macTestWorkerStringVec.push_back("MacTestWorker");
                mac_c ++;
            }

            if(mac2Ctx)
            {
                macWorkerMap.insert({"MAC2", mac_c});
                macTestWorkerStringVec.push_back("Mac2TestWorker");
                mac_c ++;
            }
            //--------------------------------------------------------------------------
            // Print run setup
            std::string ctxt_run_string            = "parallel";
            std::string graphs_streams_mode_string = (graphs_mode) ? "Graphs" : "Streams";

            printf("\n----------------------------------------------------------");
            printf("\nNotes on run setup:");
            printf("\n--> %d CUDA contexts (workers) are run in %s and in %s mode: %d cuPHY CUDA contexts, %d cuMAC contexts \n\n", nCtxts, ctxt_run_string.c_str(), graphs_streams_mode_string.c_str(), nCuphyCtxts, nCumacCtxts);
            for(int ctxtIdx = 0; ctxtIdx < nCtxts; ctxtIdx++)
            {
                if(ctxtIdx < nCuphyCtxts)
                {
                    printf("requested SMs for context [%d] %-20s : %d\n", ctxtIdx, phyTestWorkerStringVec[ctxtIdx].c_str(), ctxSmCounts[ctxtIdx]);
                }
                else
                {
                    if(macCtx || mac2Ctx)
                    {
                        printf("requested SMs for context [%d] %-20s : %d\n", ctxtIdx, macTestWorkerStringVec[ctxtIdx - nCuphyCtxts].c_str(), ctxSmCounts[ctxtIdx]);
                    }
                }                
            }

            CUcontext primaryCtx;
            CU_CHECK(cuCtxGetCurrent(&primaryCtx));
            printf("\nPrimary Context Id 0x%0lx\n", reinterpret_cast<uint64_t>(primaryCtx));
            printf("----------------------------------------------------------\n");

            // test worker vectors
            // PHY workers
            std::vector<cuPHYTestWorker> cuphyTestWorkerVec;
            cuphyTestWorkerVec.reserve(nCuphyCtxts);
            // MAC workers
            std::vector<cuMACTestWorker> cumacTestWorkerVec;
            cumacTestWorkerVec.reserve(nCumacCtxts);

//---------------------------------------------------------------------------------------------------------------
/* Thinking out loud
   - Either do the split + resource descr + green ctx creation here and then just pass it along to the cuPHYTestWorker ctor
   - Or ensure the ctor is called in a way that all splits from initial resources happen first
     and resplits happen later. This has to be known with an arg.
     If this is a resplit, then that worker needs to have access to the greenCtx created for that resplit.
    Following option 1, as it appears to be simpler/better.
*/
            std::array<cuphy::cudaGreenContext, CURRENT_MAX_GREEN_CTXS> greenContexts = {};
#if CUDA_VERSION >= 12040
            CUdevResource initial_device_GPU_resources = {};
            CUdevResourceType default_resource_type = CU_DEV_RESOURCE_TYPE_SM; // other alternative is CU_DEV_RESOURCE_TYPE_INVALID
            unsigned int use_flags = 0;
            std::vector<CUdevResource> devResources;
            std::vector<unsigned int> actual_split_groups;
            std::vector<unsigned int> min_sm_counts;
            // On every cuDevSmResourceSplitByCount() API call, we create two CUdevResource(s): the resulting one and the remaining one.
            // Even though the remaining one may not always be used, and thus one could pass a nullptr to that API call, we keep the 2x notation for convenience
            // as theoretically, every devResource could be used to create a green context.
            // This is the reason why all resource_index.*split variables have even values (0, 2, 4, etc.).
            devResources.resize(2*CURRENT_MAX_GREEN_CTXS);
            actual_split_groups.resize(2*CURRENT_MAX_GREEN_CTXS);
            min_sm_counts.resize(2*CURRENT_MAX_GREEN_CTXS);
            //std::array<cuphy::cudaGreenContext, CURRENT_MAX_GREEN_CTXS> greenContexts = {};
            std::array<cuphy::cudaGreenContext, 2 /* hardcoded for now; update as needed*/> tmpGreenContextsForResplit = {}; // these are for resplits. 2 is hardcoded for now

            // If using green contexts, create the requested splits and cuphy::cudaGreenContext objects
            if (useGreenContexts)
            {
                // Get minor and major compute capability numbers, to determine the currently supported granularity of SM splits.
                int minor_cc = 0;
                int major_cc = 0;
                CU_CHECK(cuDeviceGetAttribute(&minor_cc, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
                CU_CHECK(cuDeviceGetAttribute(&major_cc, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
                
                // Hardcode SM granularity for green context splits. Based on driver API documentation for now.
                int SM_granularity = 0;
                if(major_cc == 8)
                {
                    SM_granularity = 2;
                }
                else if (major_cc == 9)
                {
                    SM_granularity = 8;
                }
                else
                {
                    NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "Running in untested compute capability ({}.{}). Granularity of SM splits for green contexts unknown.", major_cc, minor_cc);
                    exit(1);
                }
                NVLOGC_FMT(NVLOG_TESTBENCH, "compute capability {}.{}. SM granularity for green contexts splits is {}.", major_cc, minor_cc, SM_granularity);

                CU_CHECK(cuDeviceGetDevResource(device, &initial_device_GPU_resources, default_resource_type));
                printf("Initial GPU resources retrieved via cuDeviceGetDevResource() have type %d and SM count %d.\n",  initial_device_GPU_resources.type, initial_device_GPU_resources.sm.smCount);

#if 0
                // Do multiple splits of initial GPU resources; always select the first N SMs for the green context of each channel.
                // This is solely for experimental purposes. Intention is to show the risks (perf. impact) of poorly chosen overlapping SM partitions.

                // PUSCH
                unsigned int resource_index_pusch_split = 0;  //index into devResources
                actual_split_groups[resource_index_pusch_split] = 1;
                min_sm_counts[resource_index_pusch_split] = ctxSmCounts[channelWorkerMap["PUSCH"]];
                CU_CHECK(cuDevSmResourceSplitByCount(&devResources[resource_index_pusch_split], &actual_split_groups[resource_index_pusch_split], &initial_device_GPU_resources, &devResources[resource_index_pusch_split+1], use_flags, min_sm_counts[resource_index_pusch_split]));
                greenContexts[channelWorkerMap["PUSCH"]].create(gpuId, &devResources[resource_index_pusch_split]);

                // PUCCH
                unsigned int resource_index_pucch_split = 2;  //index into devResources
                actual_split_groups[resource_index_pucch_split] = 1;
                min_sm_counts[resource_index_pucch_split] = ctxSmCounts[channelWorkerMap["PUCCH"]];
                CU_CHECK(cuDevSmResourceSplitByCount(&devResources[resource_index_pucch_split], &actual_split_groups[resource_index_pucch_split], &initial_device_GPU_resources, &devResources[resource_index_pucch_split+1], use_flags, min_sm_counts[resource_index_pucch_split]));
                greenContexts[channelWorkerMap["PUCCH"]].create(gpuId, &devResources[resource_index_pucch_split]);

                // PRACH
                unsigned int resource_index_prach_split = 4;  //index into devResources
                actual_split_groups[resource_index_prach_split] = 1;
                min_sm_counts[resource_index_prach_split] = ctxSmCounts[channelWorkerMap["PRACH"]];
                CU_CHECK(cuDevSmResourceSplitByCount(&devResources[resource_index_prach_split], &actual_split_groups[resource_index_prach_split], &initial_device_GPU_resources, &devResources[resource_index_prach_split+1], use_flags, min_sm_counts[resource_index_prach_split]));
                greenContexts[channelWorkerMap["PRACH"]].create(gpuId, &devResources[resource_index_prach_split]);

                // PDSCH
                unsigned int resource_index_pdsch_split = 6;  //index into devResources
                actual_split_groups[resource_index_pdsch_split] = 1;
                min_sm_counts[resource_index_pdsch_split] = ctxSmCounts[channelWorkerMap["PDSCH"]];
                CU_CHECK(cuDevSmResourceSplitByCount(&devResources[resource_index_pdsch_split], &actual_split_groups[resource_index_pdsch_split], &initial_device_GPU_resources, &devResources[resource_index_pdsch_split+1], use_flags, min_sm_counts[resource_index_pdsch_split]));
                greenContexts[channelWorkerMap["PDSCH"]].create(gpuId, &devResources[resource_index_pdsch_split]);

                // PDCCH
                unsigned int resource_index_pdcch_split = 8;  //index into devResources
                actual_split_groups[resource_index_pdcch_split] = 1;
                min_sm_counts[resource_index_pdcch_split] = ctxSmCounts[channelWorkerMap["PDCCH"]];
                CU_CHECK(cuDevSmResourceSplitByCount(&devResources[resource_index_pdcch_split], &actual_split_groups[resource_index_pdcch_split], &initial_device_GPU_resources, &devResources[resource_index_pdcch_split+1], use_flags, min_sm_counts[resource_index_pdcch_split]));
                greenContexts[channelWorkerMap["PDCCH"]].create(gpuId, &devResources[resource_index_pdcch_split]);

                // SSB
                if(ssbCtx)
                {
                    unsigned int resource_index_ssb_split = 10;  //index into devResources
                    actual_split_groups[resource_index_ssb_split] = 1;
                    min_sm_counts[resource_index_ssb_split] = ctxSmCounts[channelWorkerMap["SSB"]];
                    CU_CHECK(cuDevSmResourceSplitByCount(&devResources[resource_index_ssb_split], &actual_split_groups[resource_index_ssb_split], &initial_device_GPU_resources, &devResources[resource_index_ssb_split+1], use_flags, min_sm_counts[resource_index_ssb_split]));
                    greenContexts[channelWorkerMap["SSB"]].create(gpuId, &devResources[resource_index_ssb_split]);
                }

                // MAC
                if (nMACWorkers > 0)
                {
                    unsigned int resource_index_cumac_split = ssbCtx ? 12 : 10;  //index into devResources
                    actual_split_groups[resource_index_cumac_split] = 1;
                    min_sm_counts[resource_index_cumac_split] = ctxSmCounts[channelWorkerMap["MAC"]];
                    CU_CHECK(cuDevSmResourceSplitByCount(&devResources[resource_index_cumac_split], &actual_split_groups[resource_index_cumac_split], &initial_device_GPU_resources, &devResources[resource_index_cumac_split+1], use_flags, min_sm_counts[resource_index_cumac_split]));
                    greenContexts[channelWorkerMap["MAC"]].create(gpuId, &devResources[resource_index_cumac_split]);
                }

#else
                // Current SM split strategy is to take into consideration the execution duration, timing requirements and scheduling pattern of different channels and establish,
                // as much as possible, min. SM overlap when different channels contend for the same SMs due to GPU oversubscription.
                // For example, if PDSCH gets 102 SMs and PUSCH 66, the goal is to have as little overlap between them as possible given these SM counts and also have some SMs only be used
                // by PDSCH and some only by PUSCH. Similarly, because DL (downlink channels) have tighter timing requirements than UL, their green contexts' SM allocations should have as
                // little overhead as possible with those from UL channels.

                // Assume you have all available SMs of a GPU visualized as [0..max_SMs-1]. Current splits could potentially be visualized as follows:
                // (1) [     ][      Y SMs      ]  PUSCH runs on a green context with Y SMs
                // (2) [        ][     A + B SMs]
                // (3) [        ][ A SMs][ B SMs]  PUCCH's green context can use A SMs; PRACH's GC B SMs. Both PUCCH and PRACH will contend with some PUSCH SMs, but not with each other.
                // (4) [       N SMs       ][   ]  PDSCH can use N SMs. Partial overlap with PUSCH SMs depending on values of N, Y and max_SMs of the GPU.
                // (5) [X SMs][                 ]  PDCCH can use X SMs.
                // (6)        [ Z SMs][         ]  SSB can use Z SMs. No overlap with PDCCH, but can contend with PDSCH SMs.
                // (7) [            ][ M1       ]  cuMAC can use M1 SMs. TODO overlap of SMs in flux
                // (8) [            ][ M2       ]  cuMAC2 can use M2 SMs. TODO overlap of SMs in flux
                // Please note that these are all subject to change and the schematic is not up to scale. The actual values of Y, A, B, N, X, Z, M and the GPU's max_SM count also matter.

                // Split for PUSCH first from initial_device GPU resources. Split into these ranges: [0, max SMs - PUSCH SMs-1] and [max SMs - PUSCH SMs, max SMs)
                unsigned int resource_index_pusch_split = 0;  //index into devResources
                actual_split_groups[resource_index_pusch_split] = 1;
                min_sm_counts[resource_index_pusch_split] = puschCtx ? gpuMaxSmCount - ctxSmCounts[channelWorkerMap["PUSCH"]] : 0;
                // Since the resource split will round up every split but the remaining one to SM_granularity, update first split accordingly to ensure we have at least the
                // user specified SM count for PUSCH, as long as there are at least SM_granularity SMs left.
                int first_split_modulo_SM_granularity = (min_sm_counts[resource_index_pusch_split] % SM_granularity);
                if ((min_sm_counts[resource_index_pusch_split] > SM_granularity) && (first_split_modulo_SM_granularity != 0)) {
                    min_sm_counts[resource_index_pusch_split] -= first_split_modulo_SM_granularity;
                }
                if(puschCtx)
                {
                    CU_CHECK(cuDevSmResourceSplitByCount(&devResources[resource_index_pusch_split], &actual_split_groups[resource_index_pusch_split], &initial_device_GPU_resources, &devResources[resource_index_pusch_split+1], use_flags, min_sm_counts[resource_index_pusch_split]));
                    //printf("Split <some val | PUSCH> result with %d actual groups has SMs %d and remaining %d and initial SMs %d\n", actual_split_groups[resource_index_pusch_split], devResources[resource_index_pusch_split].sm.smCount, devResources[resource_index_pusch_split+1].sm.smCount, initial_device_GPU_resources.sm.smCount);
                    greenContexts[channelWorkerMap["PUSCH"]].create(gpuId, &devResources[resource_index_pusch_split+1]);
                }

                // Do another split of the initial GPU resources; will then resplit the remaining split for PUCCH and PRACH
                unsigned int resource_index_prep_for_ul_ctrl_split = 2;  //index into devResources
                actual_split_groups[resource_index_prep_for_ul_ctrl_split] = 1;
                int rounded_up_PUCCH = pucchCtx ? round_up_to_next(ctxSmCounts[channelWorkerMap["PUCCH"]], (pucchCtx && prachCtx) ? SM_granularity: 1) : 0;
                // remaining split, so only round up if there is no PUCCH
                //int rounded_up_PRACH = (prachCtx && !pucchCtx) ? round_up_to_next(ctxSmCounts[channelWorkerMap["PRACH"]], SM_granularity) : ctxSmCounts[channelWorkerMap["PRACH"]];
                int rounded_up_PRACH = prachCtx ? ctxSmCounts[channelWorkerMap["PRACH"]] : 0;
                min_sm_counts[resource_index_prep_for_ul_ctrl_split] = gpuMaxSmCount - rounded_up_PUCCH - rounded_up_PRACH;

                int second_split_modulo_SM_granularity = min_sm_counts[resource_index_prep_for_ul_ctrl_split] % SM_granularity;
                if ((min_sm_counts[resource_index_prep_for_ul_ctrl_split] > SM_granularity) && (second_split_modulo_SM_granularity != 0)) {
                    min_sm_counts[resource_index_prep_for_ul_ctrl_split] -= second_split_modulo_SM_granularity;
                }
                if (pucchCtx || prachCtx)
                {
                    CU_CHECK(cuDevSmResourceSplitByCount(&devResources[resource_index_prep_for_ul_ctrl_split], &actual_split_groups[resource_index_prep_for_ul_ctrl_split], &initial_device_GPU_resources, &devResources[resource_index_prep_for_ul_ctrl_split+1], use_flags, min_sm_counts[resource_index_prep_for_ul_ctrl_split]));
                    tmpGreenContextsForResplit[0].create(gpuId, &devResources[resource_index_prep_for_ul_ctrl_split+1]);
                    //printf("Split <some val | PUCCH/PRACH> result with %d actual groups has SMs %d and remaining %d and initial SMs %d\n", actual_split_groups[resource_index_prep_for_ul_ctrl_split], devResources[resource_index_prep_for_ul_ctrl_split].sm.smCount, devResources[resource_index_prep_for_ul_ctrl_split+1].sm.smCount, initial_device_GPU_resources.sm.smCount);
                }

                // Now resplit the remaining split. It'll be PUCCH | PRACH unless there is no PUCCH so it'd be PRACH (if one exists)
                unsigned int resource_index_pucch_split = 4;  //index into devResources
                actual_split_groups[resource_index_pucch_split] = 1;
                min_sm_counts[resource_index_pucch_split] = (pucchCtx) ? rounded_up_PUCCH : rounded_up_PRACH;
                CUdevResource resource_to_split = {};
                if (pucchCtx && !prachCtx)
                {
                    greenContexts[channelWorkerMap["PUCCH"]].create(gpuId, &devResources[resource_index_prep_for_ul_ctrl_split+1]);
                }
                else if (prachCtx && !pucchCtx)
                {
                    greenContexts[channelWorkerMap["PRACH"]].create(gpuId, &devResources[resource_index_prep_for_ul_ctrl_split+1]);
                }
                else if (pucchCtx && prachCtx)
                {
                    tmpGreenContextsForResplit[0].getResources(&resource_to_split);
                    CU_CHECK(cuDevSmResourceSplitByCount(&devResources[resource_index_pucch_split], &actual_split_groups[resource_index_pucch_split], &resource_to_split, &devResources[resource_index_pucch_split+1], use_flags, min_sm_counts[resource_index_pucch_split]));
                    greenContexts[channelWorkerMap["PUCCH"]].create(gpuId, &devResources[resource_index_pucch_split]);
                    greenContexts[channelWorkerMap["PRACH"]].create(gpuId, &devResources[resource_index_pucch_split+1]);
                }

                // Do another split of the initial GPU resources to get PDSCH resource
                unsigned int resource_index_pdsch_split = 6;  //index into devResources
                actual_split_groups[resource_index_pdsch_split] = 1;
                min_sm_counts[resource_index_pdsch_split] = pdschCtx ? ctxSmCounts[channelWorkerMap["PDSCH"]] : 0;

                int pdsch_split_modulo_SM_granularity = min_sm_counts[resource_index_pdsch_split] % SM_granularity;
                // Round up (default behavior of API) unless this would cause PDSCH to take all SMs
                if (pdsch_split_modulo_SM_granularity != 0) {
                    min_sm_counts[resource_index_pdsch_split] += (SM_granularity - pdsch_split_modulo_SM_granularity);
                    if (min_sm_counts[resource_index_pdsch_split] >= gpuMaxSmCount) {
                        min_sm_counts[resource_index_pdsch_split] -= SM_granularity;
                    }
                }

                if(pdschCtx)
                {
                    CU_CHECK(cuDevSmResourceSplitByCount(&devResources[resource_index_pdsch_split], &actual_split_groups[resource_index_pdsch_split], &initial_device_GPU_resources, &devResources[resource_index_pdsch_split+1], use_flags, min_sm_counts[resource_index_pdsch_split]));
                    greenContexts[channelWorkerMap["PDSCH"]].create(gpuId, &devResources[resource_index_pdsch_split]);
                }

                // Do another split of the initial GPU resources to get PDCCH resource
                unsigned int resource_index_pdcch_split = 8;  //index into devResources
                actual_split_groups[resource_index_pdcch_split] = 1;
                min_sm_counts[resource_index_pdcch_split] = ctxSmCounts[channelWorkerMap["PDCCH"]];

                if(pdcchCtx)
                {
                    CU_CHECK(cuDevSmResourceSplitByCount(&devResources[resource_index_pdcch_split], &actual_split_groups[resource_index_pdcch_split], &initial_device_GPU_resources, &devResources[resource_index_pdcch_split+1], use_flags, min_sm_counts[resource_index_pdcch_split]));
                    greenContexts[channelWorkerMap["PDCCH"]].create(gpuId, &devResources[resource_index_pdcch_split]);
                }

                CUdevResource another_resource_to_split = {};
                if(pdcchCtx)
                {
                    tmpGreenContextsForResplit[1].create(gpuId, &devResources[resource_index_pdcch_split+1]);
                    tmpGreenContextsForResplit[1].getResources(&another_resource_to_split);
                }
                // Resplit the remaining split of PDCCH to get SSB SMs

                if(ssbCtx)
                {
                    unsigned int resource_index_ssb_split = 10;  //index into devResources
                    actual_split_groups[resource_index_ssb_split] = 1;
                    min_sm_counts[resource_index_ssb_split] = ctxSmCounts[channelWorkerMap["SSB"]];
                    // If there was no PDCCH context, just resplit the initial GPU resources
                    CU_CHECK(cuDevSmResourceSplitByCount(&devResources[resource_index_ssb_split], &actual_split_groups[resource_index_ssb_split], (pdcchCtx) ? &another_resource_to_split : &initial_device_GPU_resources, &devResources[resource_index_ssb_split+1], use_flags, min_sm_counts[resource_index_ssb_split]));
                    greenContexts[channelWorkerMap["SSB"]].create(gpuId, &devResources[resource_index_ssb_split]);
                }

                if (macCtx)
                {
                    // Do another split of the initial GPU resources to get cuMAC resources
                    unsigned int resource_index_cumac_split = ssbCtx ? 12 : 10;  //index into devResources
                    actual_split_groups[resource_index_cumac_split] = 1;
                    min_sm_counts[resource_index_cumac_split] = gpuMaxSmCount - ctxSmCounts[nCuphyCtxts]; // single cuMAC worker for now
                    // Since the resource split will round up every split but the remaining one to SM_granularity, update first split to ensure we have at least the
                    // user specified SM count for cuMAC, as long as there are at least SM_granularity SMs left.
                    int first_split_modulo_SM_granularity = (min_sm_counts[resource_index_cumac_split] % SM_granularity);
                    if ((min_sm_counts[resource_index_cumac_split] > SM_granularity) && (first_split_modulo_SM_granularity != 0)) {
                        min_sm_counts[resource_index_cumac_split] -= first_split_modulo_SM_granularity;
                    }
                    CU_CHECK(cuDevSmResourceSplitByCount(&devResources[resource_index_cumac_split], &actual_split_groups[resource_index_cumac_split], &initial_device_GPU_resources, &devResources[resource_index_cumac_split+1], use_flags, min_sm_counts[resource_index_cumac_split]));
                    greenContexts[nCuphyCtxts].create(gpuId, &devResources[resource_index_cumac_split+1]);
                }

                if (mac2Ctx)
                {
                    // Do another split of the initial GPU resources to get cumac2 resources
                    unsigned int resource_index_cumac2_split = ssbCtx ? 14 : 12;  //index into devResources
                    actual_split_groups[resource_index_cumac2_split] = 1;
                    min_sm_counts[resource_index_cumac2_split] = gpuMaxSmCount - ctxSmCounts[nCuphyCtxts+1]; // single cumac2 worker for now
                    // Since the resource split will round up every split but the remaining one to SM_granularity, update first split to ensure we have at least the
                    // user specified SM count for cumac2, as long as there are at least SM_granularity SMs left.
                    int first_split_modulo_SM_granularity = (min_sm_counts[resource_index_cumac2_split] % SM_granularity);
                    if ((min_sm_counts[resource_index_cumac2_split] > SM_granularity) && (first_split_modulo_SM_granularity != 0)) {
                        min_sm_counts[resource_index_cumac2_split] -= first_split_modulo_SM_granularity;
                    }
                    CU_CHECK(cuDevSmResourceSplitByCount(&devResources[resource_index_cumac2_split], &actual_split_groups[resource_index_cumac2_split], &initial_device_GPU_resources, &devResources[resource_index_cumac2_split+1], use_flags, min_sm_counts[resource_index_cumac2_split]));
                    greenContexts[nCuphyCtxts+1].create(gpuId, &devResources[resource_index_cumac2_split+1]);
                }
#endif
                // print green context configs, only run if green context is used
                std::vector<std::string> tmp_channel_names = {"PUSCH", "PUCCH", "PRACH", "PDSCH", "PDCCH", "SSB", "SRS"};
                for (int j = 0; j < tmp_channel_names.size(); j++) 
                {
                    printf("channel %s has SM count %d and channelWorkerMap %d\n", tmp_channel_names[j].c_str(), greenContexts[channelWorkerMap[tmp_channel_names[j]]].getSmCount(), channelWorkerMap[tmp_channel_names[j]]);
                }

                if (macCtx)
                {
                    printf("MAC has SM count %d and channelWorkerMap %d\n", greenContexts[nCuphyCtxts].getSmCount(), nCuphyCtxts);
                }
                if (mac2Ctx)
                {
                    printf("MAC2 has SM count %d and channelWorkerMap %d\n", greenContexts[nCuphyCtxts+1].getSmCount(), nCuphyCtxts+1);
                }
            }
#endif
//---------------------------------------------------------------------------------------------------------------
            for(int ctxtIdx = 0; ctxtIdx < nCtxts; ctxtIdx++)
            {
                if(ctxtIdx < nCuphyCtxts)
                {
                    // create cuPHY test worker
                    //printf("will call emplace for %s and ctx %d\n", testWorkerStringVec[ctxtIdx].c_str(), ctxtIdx);
                    //here the cuPHYTestWorker ctor is called, which calls cuPHYTestWorker::run() which in turn calls createCuCtx or createCuGreenCtx.
                    cuphyTestWorkerVec.emplace_back(phyTestWorkerStringVec[ctxtIdx].c_str(), ctxtIdx, cpuIds[ctxtIdx], gpuIds[ctxtIdx], cpuThrdSchdPolicy, cpuThrdPrios[ctxtIdx], ctxSmCounts[ctxtIdx], cmdQVec[ctxtIdx], rspQVec[ctxtIdx], uldl, dbgMsgLevel, useGreenContexts, std::cref(greenContexts[ctxtIdx]));
                }
                else
                {
                    // create cuMAC test worker
                    // Check that at most two cuMAC test workers exist if green contexts enabled
                    if (useGreenContexts && (ctxtIdx > nCuphyCtxts + 1))
                    {
                        printf("At most two cuMAC workers are supported in case of green contexts.");
                        exit(1);
                    }
                    cumacTestWorkerVec.emplace_back(macTestWorkerStringVec[ctxtIdx - nCuphyCtxts].c_str(), ctxtIdx, cpuIds[ctxtIdx], gpuIds[ctxtIdx], cpuThrdSchdPolicy, cpuThrdPrios[ctxtIdx], ctxSmCounts[ctxtIdx], cmdQVec[ctxtIdx], rspQVec[ctxtIdx], uldl, dbgMsgLevel, useGreenContexts, std::cref(greenContexts[ctxtIdx]));
                }
            }

            cuphy::event                               startEvent(cudaEventDisableTiming);
            std::vector<std::shared_ptr<cuphy::event>> shPtrStopEvents;
            shPtrStopEvents.resize(nCtxts);

            //------------------------------------------------------------------------------------
            // initialize workers
            numPhyCells_t numPhyCells; // default 0 for all channels
            if(prachCtx)
            {
                numPhyCells.nPRACHCells = nPRACHCells;
                cuphyTestWorkerVec[channelWorkerMap["PRACH"]].init(nStrmsPerCtxt, nSlotsPerPattern, prach_nItrsPerStrm, nTimingItrs, cuStrmPrioMap, shPtrGpuStartSyncFlag, numPhyCells, false, true, patternMode[0], false);
                // {.nPRACHCells = nPRACHCells, .longPattern = patternMode[0]}
                numPhyCells.nPRACHCells = 0;
            }
            if(pdcchCtx) // It's possible to have PDCCH in the yaml but not a separate PDCCH context. In this case PDCCH will share PDSCH's context, see PDSCH init below
            {
                // Note nPDCCHCells need not be identical to nPDSCHCells
                numPhyCells.nPDCCHCells = nPDCCHCells;
                numPhyCells.nCSIRSCells = nCSIRSCells;
                numPhyCells.nSSBCells   = ssbCtx? 0 : nSSBCells;
                cuphyTestWorkerVec[channelWorkerMap["PDCCH"]].init(nStrmsPerCtxt, nSlotsPerPattern, pdcch_nItrsPerStrm, nTimingItrs, cuStrmPrioMap, shPtrGpuStartSyncFlag, numPhyCells, false, true, patternMode[0], false);
                // {.nPDCCHCells = nPDCCHCells, .nCSIRSCells = nCSIRSCells, .nSSBCells = ssbCtx ? 0 : nSSBCells, .longPattern = patternMode[0]}
                numPhyCells.nPDCCHCells = 0;
                numPhyCells.nCSIRSCells = 0;
                numPhyCells.nSSBCells   = 0;
            }
            if(pucchCtx)
            {
                numPhyCells.nPUCCHCells = nPUCCHCells;
                cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].init(nStrmsPerCtxt, nSlotsPerPattern, pucch_nItrsPerStrm, nTimingItrs, cuStrmPrioMap, shPtrGpuStartSyncFlag, numPhyCells, false, true, patternMode[0], false);
                // {.nPUCCHCells = nPUCCHCells, .longPattern = patternMode[0]}
                numPhyCells.nPUCCHCells = 0;
            }
            if(srsCtx)
            {
                numPhyCells.nSRSCells = nSRSCells;
                cuphyTestWorkerVec[channelWorkerMap["SRS"]].init(nStrmsPerCtxt, nSlotsPerPattern, srs_nItrsPerStrm, nTimingItrs, cuStrmPrioMap, shPtrGpuStartSyncFlag, numPhyCells, splitSRScells50_50, true, patternMode[0], srsCtx);
                // {.nSRSCells = nSRSCells, .longPattern = patternMode[0], .srsSplit = splitSRScells50_50, .srsCtx = srsCtx}
                numPhyCells.nSRSCells = 0;
            }

            if(ssbCtx)
            {
                numPhyCells.nSSBCells = nSSBCells;
                cuphyTestWorkerVec[channelWorkerMap["SSB"]].init(nSSBStrmsPerCtxt, nSlotsPerPattern, ssb_nItrsPerStrm, nTimingItrs, cuStrmPrioMap, shPtrGpuStartSyncFlag, numPhyCells, false, true, patternMode[0], false);
                // {.nSSBCells = nSSBCells, .longPattern = patternMode[0]}
                numPhyCells.nSSBCells = 0;
            }

            if(pdschCtx) // PDSCH
            { 
                numPhyCells.nPDSCHCells = nPDSCHCells;
                numPhyCells.nDlbfwCells = nDlbfwCells;
                numPhyCells.nPDCCHCells = pdcchCtx ? 0 : nPDCCHCells;
                numPhyCells.nCSIRSCells = pdcchCtx ? 0 : nCSIRSCells;
                numPhyCells.nSSBCells   = (pdcchCtx || ssbCtx) ? 0 : nSSBCells;
                cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].init(nPdschStrmsPerCtxt, nSlotsPerPattern, pdsch_nItrsPerStrm, nTimingItrs, cuStrmPrioMap, shPtrGpuStartSyncFlag, numPhyCells, false, true, patternMode[0], false);
                // {.nPDSCHCells = nPDSCHCells, .nPDCCHCells = pdcchCtx ? 0 : nPDCCHCells, .nCSIRSCells = pdcchCtx ? 0 : nCSIRSCells, .nSSBCells = (pdcchCtx || ssbCtx) ? 0 : nSSBCells, .longPattern = patternMode[0]}
                numPhyCells.nPDSCHCells = 0;
                numPhyCells.nDlbfwCells = 0;
                numPhyCells.nPDCCHCells = 0;
                numPhyCells.nCSIRSCells = 0;
                numPhyCells.nSSBCells   = 0;
            }
            
            if(puschCtx) // PUSCH
            { 
                numPhyCells.nPUSCHCells = nPUSCHCells;
                numPhyCells.nPUCCHCells = pucchCtx? 0 : nPUCCHCells;
                numPhyCells.nUlbfwCells = nUlbfwCells;
                numPhyCells.nSRSCells   = srsCtx ? 0 : nSRSCells;
                numPhyCells.nPRACHCells = prachCtx? 0 : nPRACHCells;
                cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].init(nPuschStrmsPerCtxt, nSlotsPerPattern, pusch_nItrsPerStrm, nTimingItrs, cuStrmPrioMap, shPtrGpuStartSyncFlag, numPhyCells, splitSRScells50_50, true, patternMode[0], srsCtx);
                // {.nPUSCHCells = nPUSCHCells, .nPUCCHCells = pucchCtx ? 0 : nPUCCHCells, .nUlbfwCells = nUlbfwCells, .nPRACHCells = prachCtx ? 0 : nPRACHCells, .longPattern = patternMode[0], .srsSplit = splitSRScells50_50, .srsCtx = srsCtx}
                numPhyCells.nPUSCHCells = 0;
                numPhyCells.nPUCCHCells = 0;
                numPhyCells.nUlbfwCells = 0;
                numPhyCells.nSRSCells   = 0;
                numPhyCells.nPRACHCells = 0;
            }

            //------------------------------------------------------------------------------------
            // initialize channel functions
            if(! cuphyTestWorkerVec.empty())
            {
                cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].pdschTxInit(inFileNamesPdschTx[0], pdsch_nItrsPerStrm, ref_check_pdsch, identical_ldpc_configs, pdsch_proc_mode, group_pdsch_cells, nPdschCellsPerStrm, pdschPrms);

                cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].dlbfwInit(inFileNamesDlbfw[0], dlbfw_nItrsPerStrm, ref_check_dlbfw);

                cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].puschRxInit(inFileNamesPuschRx[0], fp16Mode, descramblingOn, printCbErrors, pusch_proc_mode, enableLdpcThroughputMode, group_pusch_cells, puschPrms, ldpcLaunchMode, puschSubslotProcFlag);

                cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].ulbfwInit(inFileNamesUlbfw[0], ref_check_ulbfw, pdsch_proc_mode); // TODO: ULBFW processing mode using the same with DLBFW

                cuphyTestWorkerVec[channelWorkerMap["SRS"]].srsInit(inFileNamesSRS[0], ref_check_srs, pusch_proc_mode, splitSRScells50_50);

                cuphyTestWorkerVec[channelWorkerMap["PRACH"]].prachInit(inFileNamesPRACH[0], pusch_proc_mode, ref_check_prach, group_pusch_cells, nPrachCellsPerStrm);

                cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].pucchRxInit(inFileNamesPucchRx[0], ref_check_pucch, group_pucch_cells, pusch_proc_mode);

                cuphyTestWorkerVec[channelWorkerMap["PDCCH"]].pdcchTxInit(inFileNamesPdcchTx[0], pdcch_nItrsPerStrm, group_pdsch_cells, nPdcchCellsPerStrm /* can be different than nPdschCellsPerStrm */, ref_check_pdcch, (uint64_t)(pdsch_proc_mode & 0x1));
                cuphyTestWorkerVec[channelWorkerMap["PDCCH"]].csirsInit(inFileNamesCSIRS[0], csirs_nItrsPerStrm, ref_check_csirs, group_pdsch_cells, (uint64_t)(pdsch_proc_mode & 0x1));

                cuphyTestWorkerVec[channelWorkerMap["SSB"]].ssbInit(inFileNamesSSB[0], ssb_nItrsPerStrm, ref_check_ssb, group_pdsch_cells, (uint64_t)(pdsch_proc_mode & 0x1));
            }

            // cuMAC
            if(macCtx)
            {
                if(cumacOptions.config_slot_count != 0 && static_cast<uint32_t>(cumacOptions.config_slot_count) != nSlotsPerPattern)
                {
                    NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT, "Error! cumac_options per-slot array length ({}) does not match total slot config length ({}). Set cumac_light_weight_flag and perc_sm_num_thrd_blk length to {} or omit cumac_options to use defaults.", cumacOptions.config_slot_count, nSlotsPerPattern, nSlotsPerPattern);
                    exit(1);
                }
                cumacTestWorkerVec[macWorkerMap["MAC"]].init(nStrmsMac, nMacItrsPerStrm, nTimingItrs, cuStrmPrioMap, shPtrGpuStartSyncFlag, true /*waitRsp*/, patternMode[0], macInternalTimer, 0 /*lightWeightFlagOffset*/); // init cumac worker
                cumacTestWorkerVec[macWorkerMap["MAC"]].setCumacOptions(cumacOptions);

                cumacTestWorkerVec[macWorkerMap["MAC"]].macInit(inFileNamesMac[0], nMacSlots, ref_check_mac, true /* waitRsp */); // init cumac worker
            }
            // cuMAC2
            if(mac2Ctx)
            {
                cumacTestWorkerVec[macWorkerMap["MAC2"]].init(nStrmsMac2, nMac2ItrsPerStrm, nTimingItrs, cuStrmPrioMap, shPtrGpuStartSyncFlag, true /*waitRsp*/, patternMode[0], mac2InternalTimer, 2 /*lightWeightFlagOffset*/); // init cumac2 worker, hardcoded 2 slot gap

                cumacTestWorkerVec[macWorkerMap["MAC2"]].macInit(inFileNamesMac2[0], nMac2Slots, ref_check_mac, true /* waitRsp */); // init cumac2 worker
            }

            //------------------------------------------------------------------------------------
            // read SM ids for all test workers
            std::vector<cuPHYTestWorker*> pCuphyTestWorkers; //{&cuphyTestWorkerVec[0], &cuphyTestWorkerVec[1], &cuphyTestWorkerVec[2]};
            for(int i = 0; i < nCuphyCtxts; i++)
            {
                pCuphyTestWorkers.push_back(&cuphyTestWorkerVec[i]);
            }
            readSmIds<cuPHYTestWorker>(pCuphyTestWorkers, mainStream, gpuId);

            std::vector<cuMACTestWorker*> pCumacTestWorkers; //{&cumacTestWorkerVec[0], &cumacTestWorkerVec[1], &cumacTestWorkerVec[2]};
            for(int i = 0; i < nCumacCtxts; i++)
            {
                pCumacTestWorkers.push_back(&cumacTestWorkerVec[i]);
            }
            readSmIds<cuMACTestWorker>(pCumacTestWorkers, mainStream, gpuId);

            //------------------------------------------------------------------------------------
            // Loop over slot-patterns
            //
            for(int patternIdx = 0; patternIdx < num_patterns; patternIdx++)
            {
                // timing iterations
                // setup pipelines
	    	    if(! cuphyTestWorkerVec.empty()) // no need to run cuPHY workload
                {
                    cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].pdschTxSetup(inFileNamesPdschTx[patternIdx], pdschSlotRunFlag);

                    cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].dlbfwSetup(inFileNamesDlbfw[patternIdx]);

                    cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].puschRxSetup(inFileNamesPuschRx[patternIdx]);

                    cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].ulbfwSetup(inFileNamesUlbfw[patternIdx]);

                    cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].pucchRxSetup(inFileNamesPucchRx[patternIdx]);

                    cuphyTestWorkerVec[channelWorkerMap["PRACH"]].prachSetup(inFileNamesPRACH[patternIdx]);

                    cuphyTestWorkerVec[channelWorkerMap["PDCCH"]].pdcchTxSetup(inFileNamesPdcchTx[patternIdx], pdcchSlotRunFlag);

                    cuphyTestWorkerVec[channelWorkerMap["PDCCH"]].csirsSetup(inFileNamesCSIRS[patternIdx], csirsSlotRunFlag);
                    
                    cuphyTestWorkerVec[channelWorkerMap["SRS"]].srsSetup(inFileNamesSRS[patternIdx]);

                    cuphyTestWorkerVec[channelWorkerMap["SSB"]].ssbSetup(inFileNamesSSB[patternIdx], pbchSlotRunFlag);
                }
                
                if(macCtx)
                {
                    std::vector<uint8_t> macSlotConfig(macSlotRunFlag, macSlotRunFlag + nSlotsPerPattern);
                    cumacTestWorkerVec[macWorkerMap["MAC"]].macSetup(inFileNamesMac[patternIdx], macSlotConfig); // cumac worker setup, using per timeslot parameter
                }
                if(mac2Ctx)
                {
                    std::vector<uint8_t> macSlotConfig(macSlotRunFlag, macSlotRunFlag + nSlotsPerPattern);
                    cumacTestWorkerVec[macWorkerMap["MAC2"]].macSetup(inFileNamesMac2[patternIdx], macSlotConfig); // cumac2 worker setup, using per timeslot parameter
                }
                // start profiler after setup so capture range excludes setup
                if(enableNvprof)
                {
                    cudaProfilerStart();
                    NVLOGI_FMT(NVLOG_TAG_BASE_TESTBENCH, "Profiler started");
                }
#if USE_NVTX
                nvtxRangePush("PATTERN");
#endif
                for(uint32_t itrIdx = 0; itrIdx < nTimingItrs; ++itrIdx)
                {
                    if(0 != nPowerItrs)
                    {
                        // Unified power iteration function - supports both graph and stream modes
                        // Graph mode enabled with -m 1, stream mode with -m 0
                        runPowerIterations(
                            cuphyTestWorkerVec,             // Worker vectors (references)
                            nPowerItrs, powerDelayUs,       // Power iteration config
                            mainStream, gpuId,              // CUDA resources
                            shPtrGpuStartSyncFlag, syncUpdateIntervalCnt, // Sync config
                            startEvent,                     // Events
                            cumacTestWorkerVec, shPtrStopEvents, // More workers/events
                            channelWorkerMap, macWorkerMap, // Worker maps
                            cfg_process_mode, uldl,         // Mode and config
                            nCuphyCtxts, nCumacCtxts, nCtxts, // Context counts
                            nSlotsPerPattern,               // Slots per pattern
                            pdschCtx, pdcchCtx, puschCtx, prachCtx, pucchCtx, srsCtx, ssbCtx, macCtx, mac2Ctx, // Channel enables
                            macInternalTimer, mac2InternalTimer // Timer configs
                        );
                    }

                    // Launch delay kernel on main stream on every iteration to ensure timeline is preserved
                    gpu_us_delay(delayUs, gpuId, mainStream.handle(), 1);

                    // Drop event on main stream for kernels to queue behind
                    startEvent.record(mainStream.handle());
                    // start timer
                    slotPatternTimer.record_begin(mainStream.handle());

                    // run pipeline
                    if(pdschCtx)
                    {
                        cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].pdschTxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PDSCH"]], true, pdcchCtx ? cuphyTestWorkerVec[channelWorkerMap["PDCCH"]].getpdcchCsirsInterSlotEndEventVec() : nullptr);
                    }

                    if(pdcchCtx)
                    {
                        cuphyTestWorkerVec[channelWorkerMap["PDCCH"]].pdschTxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PDCCH"]], true, nullptr, cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].getSlotBoundaryEventVecPtr());
                    }

                    if(g_start_delay_cfg_us.ul_anchor_from_yaml)
                    {
                        if(g_start_delay_cfg_us.ul_anchor_mode == ul_anchor_mode_t::PRACH)
                        {
                            if(prachCtx)
                                cuphyTestWorkerVec[channelWorkerMap["PRACH"]].puschRxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PRACH"]], startEvent.handle());

                            cudaEvent_t anchorEvt = (prachCtx && cuphyTestWorkerVec[channelWorkerMap["PRACH"]].getPrachStartEvent())
                                ? cuphyTestWorkerVec[channelWorkerMap["PRACH"]].getPrachStartEvent()
                                : startEvent.handle();

                            if(puschCtx)
                                cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].puschRxRun(anchorEvt, shPtrStopEvents[channelWorkerMap["PUSCH"]]);
                            if(pucchCtx)
                                cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].puschRxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PUCCH"]], anchorEvt, anchorEvt);
                        }
                        else if(g_start_delay_cfg_us.ul_anchor_mode == ul_anchor_mode_t::PUCCH)
                        {
                            if(pucchCtx)
                                cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].puschRxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PUCCH"]], startEvent.handle(), startEvent.handle());

                            cudaEvent_t anchorEvt = (pucchCtx && cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].getPucchStartEvent())
                                ? cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].getPucchStartEvent()
                                : startEvent.handle();

                            if(puschCtx)
                                cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].puschRxRun(anchorEvt, shPtrStopEvents[channelWorkerMap["PUSCH"]]);
                            if(prachCtx)
                                cuphyTestWorkerVec[channelWorkerMap["PRACH"]].puschRxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PRACH"]], anchorEvt);
                        }
                        else
                        {
                            if(puschCtx)
                                cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].puschRxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PUSCH"]]);

                            cudaEvent_t puschStartEvt = (puschCtx && cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getPuschStartEvent())
                                ? cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getPuschStartEvent()
                                : startEvent.handle();
                            cudaEvent_t puschPucch2Evt = (puschCtx && cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getPucch2DelayStopEvent())
                                ? cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getPucch2DelayStopEvent()
                                : puschStartEvt;

                            if(prachCtx)
                                cuphyTestWorkerVec[channelWorkerMap["PRACH"]].puschRxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PRACH"]], puschStartEvt);
                            if(pucchCtx)
                                cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].puschRxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PUCCH"]], puschPucch2Evt, puschStartEvt);
                        }
                    }
                    else
                    {
                        if(puschCtx)
                        {
                            cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].puschRxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PUSCH"]]);
                        }

                        if(prachCtx)
                        {
                            // If PRACH delay from YAML: relative to PUSCH1 start; else relative to PUSCH1 end
                            cudaEvent_t prachStart = startEvent.handle();
                            if(puschCtx)
                            {
                                cudaEvent_t puschEvent = g_start_delay_cfg_us.prach_delay_from_yaml
                                    ? cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getPuschStartEvent()
                                    : cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getPusch1EndEvent();
                                if(puschEvent)
                                {
                                    prachStart = puschEvent;
                                }
                            }
                            cuphyTestWorkerVec[channelWorkerMap["PRACH"]].puschRxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PRACH"]], prachStart);
                        }

                        if(pucchCtx)
                        {
                            cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].puschRxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PUCCH"]], cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getPucch2DelayStopEvent(), cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getPuschStartEvent());
                        }
                    }

                    if(srsCtx)
                    {
                        cuphyTestWorkerVec[channelWorkerMap["SRS"]].puschRxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["SRS"]]);
                    }

                    if(ssbCtx)
                    {
                        cuphyTestWorkerVec[channelWorkerMap["SSB"]].pdschTxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["SSB"]], true /*waitRsp*/, nullptr, uldl == 3 ? nullptr : cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].getSlotBoundaryEventVecPtr() /*uldl == 5:  start at 5th slot*/);
                    }

                    // cuMAC run scheduler
                    if(macCtx)
                    {
                        cumacTestWorkerVec[macWorkerMap["MAC"]].macRun(startEvent.handle(), shPtrStopEvents[macWorkerMap["MAC"] + nCuphyCtxts], true /*waitRsp*/, (uldl == 3 || macInternalTimer) ? nullptr : cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].getSlotBoundaryEventVecPtr()); // cumac run scheduler
                        // if uldl == 3 or macInternalTimer = true, use internal timer for sync on 500
                        // else use pdsch interslot for sync on 500 us
                    }
                    // cuMAC2 run scheduler
                    if(mac2Ctx)
                    {
                        std::vector<cuphy::event>* eventVecPtr = nullptr;
                        if (!(uldl == 3 || mac2InternalTimer)) {
                            if (macInternalTimer) {
                                auto macEventVecOpt = cumacTestWorkerVec[macWorkerMap["MAC"]].getMacInterSlotEventVecPtr();
                                eventVecPtr = macEventVecOpt.has_value() ? *macEventVecOpt : nullptr;
                            } else {
                                eventVecPtr = cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].getSlotBoundaryEventVecPtr();
                            }
                        }
                        cumacTestWorkerVec[macWorkerMap["MAC2"]].macRun(startEvent.handle(), shPtrStopEvents[macWorkerMap["MAC2"] + nCuphyCtxts], true /*waitRsp*/, eventVecPtr); // cumac2 run scheduler
                        // if uldl == 3 or mac2InternalTimer = true, use internal timer for sync on 500
                        // if macInternalTimer = true (MAC already has a timer), use it for sync
                        // else use pdsch interslot for sync on 500 us
                    }

                    for(int i = 0; i < nCtxts; i++) // wait for cuPHY workers and cuMAC workers to finish
                    {
                        CUDA_CHECK(cudaStreamWaitEvent(mainStream.handle(), shPtrStopEvents[i]->handle()));
                    }

                    // end timer
                    slotPatternTimer.record_end(mainStream.handle());
                    mainStream.synchronize();
                    float et = slotPatternTimer.elapsed_time_ms();
                    tot_slotPattern_time += et;

                    // cuPHY evaluation results
                    if(pdschCtx)
                        cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].eval();

                    if(puschCtx)
                        cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].eval(printCbErrors);

                    if(prachCtx)
                        cuphyTestWorkerVec[channelWorkerMap["PRACH"]].eval();

                    if(pdcchCtx)
                        cuphyTestWorkerVec[channelWorkerMap["PDCCH"]].eval();

                    if(pucchCtx)
                        cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].eval();

                    if(srsCtx)
                        cuphyTestWorkerVec[channelWorkerMap["SRS"]].eval();

                    if(ssbCtx)
                        cuphyTestWorkerVec[channelWorkerMap["SSB"]].eval();

                    // cuMAC evaluation results
                    if(macCtx)
                        cumacTestWorkerVec[macWorkerMap["MAC"]].eval();
                    if(mac2Ctx)
                        cumacTestWorkerVec[macWorkerMap["MAC2"]].eval();

                }
                // stop profiler
                if(enableNvprof)
                {
                    cudaProfilerStop();
                    NVLOGI_FMT(NVLOG_TAG_BASE_TESTBENCH, "Profiler stopped");
                }
                
                // NOTE: The txt log will be used in python scripts for performance analysis. Changes to the log format below may also need to be propagated to:
                // - testBenches/perf/measure/TDD/DDDSU/parser/standard.py
                // - testBenches/perf/measure/TDD/DDDSUUDDDD/parser/standard.py
                // - testBenches/perf/measure/FDD/sweep.py
                // - testBenches/perf/jitter.py
                
                // print results
                avg_slotPattern_time_us = tot_slotPattern_time * 1000 / static_cast<float>(nTimingItrs);
                printf("\n-----------------------------------------------------------\n");
                printf("Slot pattern # %d\n", patternIdx);
                printf("average slot pattern run time: %.2f us (averaged over %d iterations) \n", avg_slotPattern_time_us, nTimingItrs);

                if(runPUSCHVec[patternIdx])
                {
                    float startTimePusch = cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getTotPuschStartTime() * 1000 / static_cast<float>(nTimingItrs);
                    float subslotProcPusch = cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getTotPuschSubslotProcRunTime() * 1000 / static_cast<float>(nTimingItrs);
                    float timePusch      = cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getTotPuschRunTime() * 1000 / static_cast<float>(nTimingItrs);
                    if(subslotProcPusch > 0)
                    {
                        printf("Average PUSCH_subslotProc run time: %.2f us from %.2f (averaged over %d iterations) \n", subslotProcPusch, startTimePusch, nTimingItrs);
                    }
                    printf("Average PUSCH run time: %.2f us from %.2f (averaged over %d iterations) \n", timePusch, startTimePusch, nTimingItrs);
                    if(uldl == 5 || uldl == 6)
                    {
                        float startTimePusch2 = cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getTotPusch2StartTime() * 1000 / static_cast<float>(nTimingItrs);
                        float subslotProcPusch2 = cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getTotPusch2SubslotProcRunTime() * 1000 / static_cast<float>(nTimingItrs);
                        float timePusch2      = cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getTotPusch2RunTime() * 1000 / static_cast<float>(nTimingItrs);
                        if(subslotProcPusch2 > 0)
                        {
                            printf("Average PUSCH2_subslotProc run time: %.2f us from %.2f (averaged over %d iterations) \n", subslotProcPusch2, startTimePusch2, nTimingItrs);
                        }
                        printf("Average PUSCH2 run time: %.2f us from %.2f (averaged over %d iterations) \n", timePusch2, startTimePusch2, nTimingItrs);
                    }
                }
                if(runPUCCHVec[patternIdx])
                {
                    float startTimePUCCH = cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].getTotPucchStartTime() * 1000 / static_cast<float>(nTimingItrs);
                    float timePUCCH      = cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].getTotPucchRunTime() * 1000 / static_cast<float>(nTimingItrs);
                    printf("Average PUCCH run time: %.2f us from %.2f (averaged over %d iterations) \n", timePUCCH, startTimePUCCH, nTimingItrs);
                    if(uldl == 5 || uldl == 6)
                    {
                        float startTimePucch2 = cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].getTotPucch2StartTime() * 1000 / static_cast<float>(nTimingItrs);
                        float timePucch2      = cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].getTotPucch2RunTime() * 1000 / static_cast<float>(nTimingItrs);
                        printf("Average PUCCH2 run time: %.2f us from %.2f (averaged over %d iterations) \n", timePucch2, startTimePucch2, nTimingItrs);
                    }
                }
                if(runULBFWVec[patternIdx])
                {
                    float startTimeUlbfw = cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getTotUlbfwStartTime() * 1000 / static_cast<float>(nTimingItrs);
                    float timeUlbfw      = cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getTotUlbfwRunTime() * 1000 / static_cast<float>(nTimingItrs);
                    printf("Average ULBFW run time: %.2f us from %.2f (averaged over %d iterations) \n", timeUlbfw, startTimeUlbfw, nTimingItrs);
                    if(uldl == 5 || uldl == 6)
                    {
                        float startTimeUlbfw2 = cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getTotUlbfw2StartTime() * 1000 / static_cast<float>(nTimingItrs);
                        float timeUlbfw2      = cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getTotUlbfw2RunTime() * 1000 / static_cast<float>(nTimingItrs);
                        printf("Average ULBFW2 run time: %.2f us from %.2f (averaged over %d iterations) \n", timeUlbfw2, startTimeUlbfw2, nTimingItrs);
                    }
                }
                if(runSRSVec[patternIdx])
                {
                    float startTimeSRS = cuphyTestWorkerVec[channelWorkerMap["SRS"]].getTotSRSStartTime() * 1000 / static_cast<float>(nTimingItrs);
                    float timeSRS      = cuphyTestWorkerVec[channelWorkerMap["SRS"]].getTotSRSRunTime() * 1000 / static_cast<float>(nTimingItrs);
                    printf("Average SRS1  run time: %.2f us from %.2f (averaged over %d iterations) \n", timeSRS, startTimeSRS, nTimingItrs);

                    if(splitSRScells50_50)
                    {
                        float startTimeSRS2 = cuphyTestWorkerVec[channelWorkerMap["SRS"]].getTotSRS2StartTime() * 1000 / static_cast<float>(nTimingItrs);
                        float timeSRS2      = cuphyTestWorkerVec[channelWorkerMap["SRS"]].getTotSRS2RunTime() * 1000 / static_cast<float>(nTimingItrs);
                        printf("Average SRS2  run time: %.2f us from %.2f (averaged over %d iterations) \n", timeSRS2, startTimeSRS2, nTimingItrs);
                    }
                }
                if(runPRACHVec[patternIdx])
                {
                    float startTimePRACH = cuphyTestWorkerVec[channelWorkerMap["PRACH"]].getTotPrachStartTime() * 1000 / static_cast<float>(nTimingItrs);
                    float timePRACH      = cuphyTestWorkerVec[channelWorkerMap["PRACH"]].getTotPrachRunTime() * 1000 / static_cast<float>(nTimingItrs);
                    printf("Average PRACH run time: %.2f us from %.2f (averaged over %d iterations) \n", timePRACH, startTimePRACH, nTimingItrs);
                }

                if(runPDCCHVec[patternIdx])
                {
                    uint8_t pdcchRunSlotIdx = 0;
                    for(int sl = 0; sl < nSlotsPerPattern; sl++)
                    {
                        if(pdcchSlotRunFlag[sl])
                        {
                            float startTimePdcch = cuphyTestWorkerVec[channelWorkerMap["PDCCH"]].getPdcchStartTimes()[pdcchRunSlotIdx] * 1000 / static_cast<float>(nTimingItrs);
                            float timePdcch      = cuphyTestWorkerVec[channelWorkerMap["PDCCH"]].getPdcchIterTimes()[pdcchRunSlotIdx] * 1000 / static_cast<float>(nTimingItrs);
                            printf("Slot # %d: average PDCCH run time: %.2f us from %.2f (averaged over %d iterations) \n", sl, timePdcch, startTimePdcch, nTimingItrs);
                            pdcchRunSlotIdx++;
                        }
                    }
                }

                if(runCSIRSVec[patternIdx])
                {
                    uint8_t csirsRunSlotIdx = 0;
                    for(int sl = 0; sl < nSlotsPerPattern; sl++)
                    {
                        if(csirsSlotRunFlag[sl])
                        {
                            float startTimeCSIRS = cuphyTestWorkerVec[channelWorkerMap["PDCCH"]].getCSIRSStartTimes()[csirsRunSlotIdx] * 1000 / static_cast<float>(nTimingItrs);
                            float timeCSIRS      = cuphyTestWorkerVec[channelWorkerMap["PDCCH"]].getCSIRSIterTimes()[csirsRunSlotIdx] * 1000 / static_cast<float>(nTimingItrs);
                            printf("Slot # %d: average CSIRS run time: %.2f us from %.2f (averaged over %d iterations) \n", sl, timeCSIRS, startTimeCSIRS, nTimingItrs);
                            csirsRunSlotIdx++;
                        }
                    }
                }

                if(runPDSCHVec[patternIdx])
                {
                    uint8_t pdschRunSlotIdx = 0;
                    for(int sl = 0; sl < nSlotsPerPattern; sl++)
                    {
                        if(pdschSlotRunFlag[sl])
                        {
                            if(uldl == 5 || uldl == 6)
                            {
                                if(runDLBFWVec[patternIdx])
                                {
                                    float  timeDlbfwStart = cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].getDlbfwIterStartTimes()[pdschRunSlotIdx] * 1000 / static_cast<float>(nTimingItrs);                                
                                    float timeDlbfw            = cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].getDlbfwIterTimes()[pdschRunSlotIdx] * 1000 / static_cast<float>(nTimingItrs);
                                    printf("Slot # %d: average DLBFW run time: %.2f us from %.2f us (averaged over %d iterations) \n", sl, timeDlbfw, timeDlbfwStart, nTimingItrs);
                                }

                                float timePdsch          = cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].getPdschIterTimes()[pdschRunSlotIdx] * 1000 / static_cast<float>(nTimingItrs);
                                float timePdschNextStart = cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].getPdschSlotStartTimes()[pdschRunSlotIdx] * 1000 / static_cast<float>(nTimingItrs);
                                printf("Slot # %d: average PDSCH run time: %.2f us from %.2f (averaged over %d iterations) \n", sl, timePdsch, timePdschNextStart, nTimingItrs);
                            }
                            else
                            {
                                float timePdsch          = cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].getPdschIterTimes()[pdschRunSlotIdx] * 1000 / static_cast<float>(nTimingItrs);
                                float timePdschNextStart = cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].getPdschSlotStartTimes()[pdschRunSlotIdx] * 1000 / static_cast<float>(nTimingItrs);
                                printf("Slot # %d: average PDSCH run time: %.2f us from %.2f (averaged over %d iterations) \n", sl, timePdsch, timePdschNextStart, nTimingItrs);

                                if(runDLBFWVec[patternIdx])
                                {
                                    float timePdschNextStart = cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].getPdschSlotStartTimes()[pdschRunSlotIdx] * 1000 / static_cast<float>(nTimingItrs);
                                    float timeDlbfw            = cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].getDlbfwIterTimes()[pdschRunSlotIdx] * 1000 / static_cast<float>(nTimingItrs);
                                    printf("Slot # %d: average DL BFW run time: %.2f us from %.2f (averaged over %d iterations) \n", sl, timeDlbfw, timePdschNextStart, nTimingItrs);
                                }
                            }
                            pdschRunSlotIdx++;
                        }
                        
                    }
                }

                if(runSSBVec[patternIdx])
                {
                    uint8_t pbchRunSlotIdx = 0;
                    for(uint32_t sl = 0; sl < nSlotsPerPattern; sl++)
                    {
                        if(pbchSlotRunFlag[sl])
                        {
                            float startTimeSSB = cuphyTestWorkerVec[channelWorkerMap["SSB"]].getTotSSBStartTime()[pbchRunSlotIdx] * 1000 / static_cast<float>(nTimingItrs);
                            float timeSSB      = cuphyTestWorkerVec[channelWorkerMap["SSB"]].getTotSSBRunTime()[pbchRunSlotIdx] * 1000 / static_cast<float>(nTimingItrs);
                            printf("Slot # %d: average SSB   run time: %.2f us from %.2f (averaged over %d iterations) \n", sl, timeSSB, startTimeSSB, nTimingItrs);
                            pbchRunSlotIdx++;
                        }
                    }
                }

                if(runMACVec[patternIdx]) // print cuMAC timing per slot
                {
                    uint8_t macRunSlotIdx = 0;
                    for(int sl = 0; sl < nSlotsPerPattern; sl++)
                    {
                        if(macSlotRunFlag[sl])
                        {
                            auto macSlotStartTimeOpt = cumacTestWorkerVec[macWorkerMap["MAC"]].getTotMACSlotStartTime();
                            auto macSlotEndTimeOpt = cumacTestWorkerVec[macWorkerMap["MAC"]].getTotMacSlotEndTime();
                            
                            if (macSlotStartTimeOpt.has_value() && macSlotEndTimeOpt.has_value() && 
                                macRunSlotIdx < static_cast<int>(macSlotStartTimeOpt->size()) && 
                                macRunSlotIdx < static_cast<int>(macSlotEndTimeOpt->size()))
                            {
                                float timeMacSlotStart = (*macSlotStartTimeOpt)[macRunSlotIdx] * 1000 / static_cast<float>(nTimingItrs);
                                float timeMacSlotEnd = (*macSlotEndTimeOpt)[macRunSlotIdx] * 1000 / static_cast<float>(nTimingItrs);
                                printf("Slot # %d: average MAC   run time: %.2f us from %.2f (averaged over %d iterations) \n", sl, timeMacSlotEnd, timeMacSlotStart, nTimingItrs);
                            }
                            macRunSlotIdx++;
                        }
                    }
                }

                if (runMAC2Vec[patternIdx]) // print cuMAC2 timing per slot
                {
                    uint8_t mac2RunSlotIdx = 0;
                    for(int sl = 0; sl < nSlotsPerPattern; sl++)
                    {
                        if(macSlotRunFlag[sl])
                        {
                            auto mac2SlotStartTimeOpt = cumacTestWorkerVec[macWorkerMap["MAC2"]].getTotMACSlotStartTime();
                            auto mac2SlotEndTimeOpt = cumacTestWorkerVec[macWorkerMap["MAC2"]].getTotMacSlotEndTime();
                            
                            if (mac2SlotStartTimeOpt.has_value() && mac2SlotEndTimeOpt.has_value() && 
                                mac2RunSlotIdx < static_cast<int>(mac2SlotStartTimeOpt->size()) && 
                                mac2RunSlotIdx < static_cast<int>(mac2SlotEndTimeOpt->size()))
                            {
                                float timeMacSlotStart = (*mac2SlotStartTimeOpt)[mac2RunSlotIdx] * 1000 / static_cast<float>(nTimingItrs);
                                float timeMacSlotEnd = (*mac2SlotEndTimeOpt)[mac2RunSlotIdx] * 1000 / static_cast<float>(nTimingItrs);
                                printf("Slot # %d: average MAC2   run time: %.2f us from %.2f (averaged over %d iterations) \n", sl, timeMacSlotEnd, timeMacSlotStart, nTimingItrs);
                            }
                            mac2RunSlotIdx++;
                        }
                    }
                }

                if(printCellMetrics)
                {
                    cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].print();
                    cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].print(printCbErrors);
                    if(prachCtx)
                        cuphyTestWorkerVec[channelWorkerMap["PRACH"]].print();
                    if(pdcchCtx)
                        cuphyTestWorkerVec[channelWorkerMap["PDCCH"]].print();
                    if(pucchCtx)
                        cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].print();
                    if(srsCtx)
                        cuphyTestWorkerVec[channelWorkerMap["SRS"]].print();
                    if(ssbCtx)
                        cuphyTestWorkerVec[channelWorkerMap["SSB"]].print();
                    if(macCtx)
                        cumacTestWorkerVec[macWorkerMap["MAC"]].print();
                    if(mac2Ctx)
                        cumacTestWorkerVec[macWorkerMap["MAC2"]].print();
                }

                // clean up params
                if(pdschCtx)
                {
                    cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].pdschTxClean();
                    cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].resetEvalBuffers();
                }
                if(prachCtx)
                    cuphyTestWorkerVec[channelWorkerMap["PRACH"]].resetEvalBuffers();
                if(pdcchCtx)
                    cuphyTestWorkerVec[channelWorkerMap["PDCCH"]].resetEvalBuffers();
                if(pucchCtx)
                    cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].resetEvalBuffers();
                if(srsCtx)
                    cuphyTestWorkerVec[channelWorkerMap["SRS"]].resetEvalBuffers();
                if(puschCtx)
                    cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].resetEvalBuffers(printCbErrors);
                if(ssbCtx)
                    cuphyTestWorkerVec[channelWorkerMap["SSB"]].resetEvalBuffers();
                if(macCtx)
                    cumacTestWorkerVec[macWorkerMap["MAC"]].resetEvalBuffers();
                if(mac2Ctx)
                    cumacTestWorkerVec[macWorkerMap["MAC2"]].resetEvalBuffers();
                
                // reset time
                tot_slotPattern_time    = 0;
                avg_slotPattern_time_us = 0;
            }

            // deinitialize workers
            for(int i = 0; i < nCuphyCtxts; i++)
            {
                cuphyTestWorkerVec[i].deinit();
            }
#if USE_NVTX
            nvtxRangePop();
#endif

            //shPtrStopEvents.clear(); // if the testWorkerVec is not cleared only when green contexts are used, add this to avoid a "context is destroyed" error later
                                       // So only uncomment, if the if condition is removed, i.e., becomes if(true).
            if (useGreenContexts)
            {

                // Delete all cuPHYTestWorker objects. Expectation is that these are deleted *before* the various green contexts are explicitly destroyed.
                // Otherwise, we'll get "context is destroyed" error when the cuphy::~stream (destructor) is called from the ~cuPHYTestWorker,
                // as at that point the contexts these streams belong to will have been destroyed
                cuphyTestWorkerVec.clear();

                cumacTestWorkerVec.clear();

                // Explicitly call destroy on every cudaGreenContext object from greenContexts.
                // Same not needed for tmpGreenContextsForResplit as these have no stream or other resources created under them
                for(int i = 0; i < CURRENT_MAX_GREEN_CTXS; i++) {
                   greenContexts[i].destroy();
                }
            }
        }
        //------------------------------------------------------------
        //PSCH
        else if(uldl == 4) // TODO: not adding cuMAC workloads since we focus on the long pattern
        {

            std::array<cuphy::cudaGreenContext, CURRENT_MAX_GREEN_CTXS> greenContexts = {};
            //------------------------------------------------------------------------------------
            // Divide cells between contexts

            printf("\n\n nPschCellsPerPattern: %d, num_patterns: %d\n\n", nPschCellsPerPattern, num_patterns);

            uint32_t      nCellsPerCtxt = nPschCellsPerPattern / nCuphyCtxts;
            string3Dvec_t inCtxFileNamesPuschRx(num_patterns); // Dim: num_patterns x nCuphyCtxts x nCellsPerCtxt
            string3Dvec_t inCtxFileNamesPdschTx(num_patterns); // Dim: num_patterns x nCuphyCtxts x nCellsPerCtxt

            for(uint32_t patternIdx = 0; patternIdx < num_patterns; ++patternIdx)
            {
                inCtxFileNamesPuschRx[patternIdx].resize(nCuphyCtxts);
                inCtxFileNamesPdschTx[patternIdx].resize(nCuphyCtxts);

                for(uint32_t ctxtIdx = 0; ctxtIdx < nCuphyCtxts; ++ctxtIdx)
                {
                    for(uint32_t i = 0; i < nCellsPerCtxt; ++i)
                    {
                        uint32_t cellIdx = ctxtIdx * nCellsPerCtxt + i;
                        inCtxFileNamesPuschRx[patternIdx][ctxtIdx].emplace_back(inFileNamesPuschRx[patternIdx][cellIdx]);
                        inCtxFileNamesPdschTx[patternIdx][ctxtIdx].emplace_back(inFileNamesPdschTx[patternIdx][cellIdx]);
                    }
                }
            }

            //-----------------------------------------------------------------------------------
            // Create workers
            std::vector<cuPHYTestWorker*> pCuphyTestWorkers(nCuphyCtxts);

            for(uint32_t ii = 0; ii < nCuphyCtxts; ii++)
            {
                pCuphyTestWorkers[ii] = new cuPHYTestWorker(std::string("PschTxRxTestWorker"), ii, cpuIds[ii], gpuIds[ii], cpuThrdSchdPolicy, cpuThrdPrios[ii], ctxSmCounts[ii], cmdQVec[ii], rspQVec[ii], uldl, dbgMsgLevel, false /*useGreenContexts is ignored*/,  std::cref(greenContexts[ii]));  //FIXME check that nCtxts <= CURRENT_MAX_GREEN_CTXS. Also ensure not accessed
            }
            finishPschSim(pCuphyTestWorkers, inCtxFileNamesPuschRx, inCtxFileNamesPdschTx, nTimingItrs, nPowerItrs, num_patterns, nCtxts, nCuphyCtxts, printCbErrors, mainStream, gpuId, delayUs, powerDelayUs, ref_check_pdsch, identical_ldpc_configs, pdsch_proc_mode, pusch_proc_mode, fp16Mode, descramblingOn, nCellsPerCtxt, nStrmsPerCtxt, nPschItrsPerStrm, slotPatternTimer, tot_slotPattern_time, avg_slotPattern_time_us, shPtrGpuStartSyncFlag, cuStrmPrioMap, syncUpdateIntervalCnt, enableLdpcThroughputMode, printCellMetrics, group_pdsch_cells, nPdschCellsPerStrm, group_pusch_cells, pdschPrms, puschPrms, ldpcLaunchMode, pdsch_nItrsPerStrm, pdschSlotRunFlag, puschSubslotProcFlag);

            for(uint32_t ii = 0; ii < nCuphyCtxts; ii++)
            {
                delete pCuphyTestWorkers[ii];
            }
        }
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    return returnValue;
}

void pschRunCore(std::vector<cuPHYTestWorker*>& pCuphyTestWorkers, uint32_t nCuphyCtxts, cuphy::stream& mainStream, cuphy::event& startEvent, std::vector<std::shared_ptr<cuphy::event>>& shPtrStopEvents)
{
    // Drop event on main stream for kernels to queue behind
    startEvent.record(mainStream.handle());

    // run pipeline
    for(uint32_t ctxIdx = 0; ctxIdx < nCuphyCtxts; ++ctxIdx)
    {
        pCuphyTestWorkers[ctxIdx]->pschTxRxRun(startEvent.handle(), shPtrStopEvents[ctxIdx]);
    }

    for(uint32_t ctxIdx = 0; ctxIdx < nCuphyCtxts; ++ctxIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(mainStream.handle(), shPtrStopEvents[ctxIdx]->handle()));
    }
}

void finishPschSim(std::vector<cuPHYTestWorker*>& pCuphyTestWorkers, string3Dvec_t& inCtxFileNamesPuschRx, string3Dvec_t& inCtxFileNamesPdschTx, uint32_t nTimingItrs, uint32_t nPowerItrs, uint32_t num_patterns, uint32_t nCtxts, uint32_t nCuphyCtxts, bool printCbErrors, cuphy::stream& mainStream, int32_t gpuId, uint32_t delayUs, uint32_t powerDelayUs, bool ref_check_pdsch, bool identical_ldpc_configs, cuphyPdschProcMode_t pdsch_proc_mode, uint64_t pusch_proc_mode, uint32_t fp16Mode, int descramblingOn, uint32_t nCellsPerCtxt, uint32_t nStrmsPerCtxt, uint32_t nPschItrsPerStrm, cuphy::event_timer& slotPatternTimer, float tot_slotPattern_time, float avg_slotPattern_time_us, std::shared_ptr<cuphy::buffer<uint32_t, cuphy::pinned_alloc>>& shPtrGpuStartSyncFlag, std::map<std::string, int>& cuStrmPrioMap, uint32_t syncUpdateIntervalCnt, bool enableLdpcThroughputMode, bool printCellMetrics, bool pdsch_group_cells, uint32_t pdsch_cells_per_stream, bool pusch_group_cells, maxPDSCHPrms pdschPrms, maxPUSCHPrms puschPrms, uint32_t ldpcLaunchMode, uint32_t pdsch_nItrsPerStrm, uint8_t* pdschSlotRunFlag, uint8_t* puschSubslotProcFlag)
{
    //------------------------------------------------------------------------------------
    // initialize workers
    numPhyCells_t numPhyCells; // default 0 for all channels
    for(uint32_t ctxIdx = 0; ctxIdx < nCuphyCtxts; ++ctxIdx)
    {
        pCuphyTestWorkers[ctxIdx]->init(nStrmsPerCtxt, nPschItrsPerStrm, nPschItrsPerStrm, nTimingItrs, cuStrmPrioMap, shPtrGpuStartSyncFlag, numPhyCells);
        pCuphyTestWorkers[ctxIdx]->pdschTxInit(inCtxFileNamesPdschTx[0][ctxIdx], pdsch_nItrsPerStrm, ref_check_pdsch, identical_ldpc_configs, pdsch_proc_mode, pdsch_group_cells, pdsch_cells_per_stream, pdschPrms);
        pCuphyTestWorkers[ctxIdx]->puschRxInit(inCtxFileNamesPuschRx[0][ctxIdx], 1, 1, true, pusch_proc_mode, enableLdpcThroughputMode, pusch_group_cells, puschPrms, ldpcLaunchMode, puschSubslotProcFlag);
    }

    //------------------------------------------------------------------------------------
    // Loop over slot patterns

    cuphy::event                               startEvent(cudaEventDisableTiming);
    std::vector<std::shared_ptr<cuphy::event>> shPtrStopEvents;
    for(uint32_t i = 0; i < nCtxts; ++i) shPtrStopEvents.emplace_back(std::make_shared<cuphy::event>(cudaEventDisableTiming));

    bool isPschTxRx = true;
    for(int patternIdx = 0; patternIdx < num_patterns; patternIdx++)
    {
        // timing iterations
        for(uint32_t itrIdx = 0; itrIdx < nTimingItrs; ++itrIdx)
        {
            // setup workers
            for(uint32_t ctxIdx = 0; ctxIdx < nCuphyCtxts; ++ctxIdx)
            {
                pCuphyTestWorkers[ctxIdx]->pdschTxSetup(inCtxFileNamesPdschTx[patternIdx][ctxIdx], pdschSlotRunFlag);
                pCuphyTestWorkers[ctxIdx]->puschRxSetup(inCtxFileNamesPuschRx[patternIdx][ctxIdx]);
            }

            if(0 != nPowerItrs)
            {
                // Initialize CPU-GPU start sync flag
                uint32_t syncFlagVal        = 1;
                (*shPtrGpuStartSyncFlag)[0] = 0;
                for(uint32_t ctxIdx = 0; ctxIdx < nCuphyCtxts; ++ctxIdx)
                {
                    pCuphyTestWorkers[ctxIdx]->setWaitVal(syncFlagVal);
                }

                // Initialize GPU batch start events
                std::vector<cuphy::event> gpuBatchStartEvents;
                for (int i = 0; i < (nPowerItrs + syncUpdateIntervalCnt - 1) / syncUpdateIntervalCnt; i++)
                    gpuBatchStartEvents.emplace_back(cudaEventDisableTiming);

                for(uint32_t powerItrIdx = 1; powerItrIdx <= nPowerItrs; ++powerItrIdx)
                {
                    if(powerDelayUs > 0) gpu_us_delay(powerDelayUs, gpuId, mainStream.handle(), 1);
                    pschRunCore(pCuphyTestWorkers, nCuphyCtxts, mainStream, startEvent, shPtrStopEvents);

                    // **BATCHING LOGIC** - Periodically synchronize CPU-GPU execution every syncUpdateIntervalCnt iterations to prevent CPU being too far ahead of the GPU
                    if(0 == (powerItrIdx % syncUpdateIntervalCnt))
                    {
                        // Signal GPU to get started
                        // CU_CHECK(cuStreamWriteValue32(mainStream.handle(), reinterpret_cast<CUdeviceptr>(shPtrGpuStartSyncFlag->addr()), syncFlagVal, CU_STREAM_WRITE_VALUE_DEFAULT));
                        (*shPtrGpuStartSyncFlag)[0] = syncFlagVal;

                        // Setup wait for next set of workloads
                        syncFlagVal++;
                        for(uint32_t ctxIdx = 0; ctxIdx < nCuphyCtxts; ++ctxIdx)
                        {
                            pCuphyTestWorkers[ctxIdx]->setWaitVal(syncFlagVal);
                        }
                    }
                    // CPU-GPU synchronization in the middle of the batch
                    if ((syncUpdateIntervalCnt / 2 ) == (powerItrIdx % syncUpdateIntervalCnt))
                    {
                        CUDA_CHECK(cudaEventRecord(gpuBatchStartEvents[powerItrIdx / syncUpdateIntervalCnt].handle(), mainStream.handle()));
                        if (powerItrIdx > syncUpdateIntervalCnt)
                        {
                            // sync CPU and GPU to prevent CPU being too far ahead of the GPU
                            CUDA_CHECK(cudaEventSynchronize(gpuBatchStartEvents[powerItrIdx / syncUpdateIntervalCnt - 1].handle()));
                        }
                    }
                }
                // Ungate the last wait request
                (*shPtrGpuStartSyncFlag)[0] = syncFlagVal;
            }
            else
            {
                // Launch delay kernel on main stream
                gpu_us_delay(delayUs, gpuId, mainStream.handle(), 1);
            }

            // start timer
            slotPatternTimer.record_begin(mainStream.handle());

            pschRunCore(pCuphyTestWorkers, nCuphyCtxts, mainStream, startEvent, shPtrStopEvents);

            // end timer
            slotPatternTimer.record_end(mainStream.handle());
            mainStream.synchronize();
            tot_slotPattern_time += slotPatternTimer.elapsed_time_ms();

            // Evaluate result
            for(uint32_t ctxIdx = 0; ctxIdx < nCuphyCtxts; ++ctxIdx)
            {
                pCuphyTestWorkers[ctxIdx]->eval(printCbErrors, isPschTxRx);
            }
        }
        // print results
        avg_slotPattern_time_us = tot_slotPattern_time * 1000 / static_cast<float>(nTimingItrs);
        printf("\n-----------------------------------------------------------\n");
        printf("Slot # %d\n", patternIdx);
        printf("average slot run time: %.2f us (averaged over %d iterations) \n", avg_slotPattern_time_us, nTimingItrs);

        for(uint32_t ctxIdx = 0; ctxIdx < nCuphyCtxts; ++ctxIdx)
        {
            float timePdsch = pCuphyTestWorkers[ctxIdx]->getPdschIterTimes()[0] * 1000 / static_cast<float>(nTimingItrs);
            printf("Ctx # %d: average PDSCH run time: %.2f us (averaged over %d iterations) \n", ctxIdx, timePdsch, nTimingItrs);

            float timePusch = pCuphyTestWorkers[ctxIdx]->getTotPuschRunTime() * 1000 / static_cast<float>(nTimingItrs);
            printf("Ctx # %d: average PUSCH run time: %.2f us (averaged over %d iterations) \n", ctxIdx, timePusch, nTimingItrs);
        }
        if(printCellMetrics)
        {
            for(uint32_t ctxIdx = 0; ctxIdx < nCuphyCtxts; ++ctxIdx)
            {
                pCuphyTestWorkers[ctxIdx]->print(printCbErrors, isPschTxRx);
            }
        }

        // clean up params
        for(uint32_t ctxIdx = 0; ctxIdx < nCuphyCtxts; ++ctxIdx)
        {
            pCuphyTestWorkers[ctxIdx]->pdschTxClean();
            pCuphyTestWorkers[ctxIdx]->resetEvalBuffers(printCbErrors);
        }

        // reset time
        tot_slotPattern_time    = 0;
        avg_slotPattern_time_us = 0;
    }

    // deinitialize workers
    for(uint32_t ctxIdx = 0; ctxIdx < nCuphyCtxts; ++ctxIdx)
    {
        pCuphyTestWorkers[ctxIdx]->deinit();
    }
}

// Unified power iteration function - works for both graph and stream modes
void runPowerIterations(
    // Core test parameters with reference vectors
    std::vector<cuPHYTestWorker>& cuphyTestWorkerVec,
    uint32_t nPowerItrs, uint32_t powerDelayUs,
    cuphy::stream& mainStream, int32_t gpuId,
    std::shared_ptr<cuphy::buffer<uint32_t, cuphy::pinned_alloc>>& shPtrGpuStartSyncFlag,
    uint32_t syncUpdateIntervalCnt,
    
    // Additional parameters needed for power iterations
    cuphy::event& startEvent,
    std::vector<cuMACTestWorker>& cumacTestWorkerVec,
    std::vector<std::shared_ptr<cuphy::event>>& shPtrStopEvents,
    std::unordered_map<std::string, int>& channelWorkerMap,
    std::unordered_map<std::string, int>& macWorkerMap,
    int cfg_process_mode, int uldl,
    int nCuphyCtxts, int nCumacCtxts, int nCtxts,
    const uint32_t nSlotsPerPattern,
    bool pdschCtx, bool pdcchCtx, bool puschCtx, bool prachCtx,
    bool pucchCtx, bool srsCtx, bool ssbCtx, bool macCtx, bool mac2Ctx,
    bool macInternalTimer, bool mac2InternalTimer) {
    
    if (nPowerItrs == 0) return;

    // Initialize GPU batch start events
    std::vector<cuphy::event> gpuBatchStartEvents;
    for (int i = 0; i < (nPowerItrs + syncUpdateIntervalCnt - 1) / syncUpdateIntervalCnt; i++)
        gpuBatchStartEvents.emplace_back(cudaEventDisableTiming);
    
    printf("Starting power iterations: mode=%s, nPowerItrs=%u, batch_size=%u\n",
           (cfg_process_mode == 1) ? "GRAPH" : "STREAM", nPowerItrs, syncUpdateIntervalCnt);

#ifdef CUBB_GPU_TESTBENCH_POWER_ITERATION_BATCHING
    // Initialize CPU-GPU start sync flag
    uint32_t syncFlagVal = 1;
    (*shPtrGpuStartSyncFlag)[0] = 0;

    // Initialize sync flag with correct sequence to avoid race condition
    // Set what GPU should wait for FIRST
    for(int i = 0; i < nCuphyCtxts; i++)
        cuphyTestWorkerVec[i].setWaitVal(syncFlagVal);
    for(int i = 0; i < nCumacCtxts; i++)
        cumacTestWorkerVec[i].setWaitVal(syncFlagVal);
#endif 

    // Main power iteration loop
    for(uint32_t powerItrIdx = 1; powerItrIdx <= nPowerItrs; ++powerItrIdx) {
        
        // Always signal startEvent for CPU-GPU sync coordination
        gpu_us_delay(powerDelayUs, gpuId, mainStream.handle(), 1);
        startEvent.record(mainStream.handle());
        
        // run pipeline
        if(pdschCtx) {
            cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].pdschTxRun(
                startEvent.handle(), shPtrStopEvents[channelWorkerMap["PDSCH"]], 
                true /*waitRsp*/, 
                pdcchCtx ? cuphyTestWorkerVec[channelWorkerMap["PDCCH"]].getpdcchCsirsInterSlotEndEventVec() : nullptr);
        }
        if(pdcchCtx) {
            cuphyTestWorkerVec[channelWorkerMap["PDCCH"]].pdschTxRun(
                startEvent.handle(), shPtrStopEvents[channelWorkerMap["PDCCH"]], 
                true, nullptr, 
                cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].getSlotBoundaryEventVecPtr());
        }
        if(g_start_delay_cfg_us.ul_anchor_from_yaml) {
            if(g_start_delay_cfg_us.ul_anchor_mode == ul_anchor_mode_t::PRACH) {
                if(prachCtx) {
                    cuphyTestWorkerVec[channelWorkerMap["PRACH"]].puschRxRun(
                        startEvent.handle(), shPtrStopEvents[channelWorkerMap["PRACH"]], startEvent.handle());
                }
                cudaEvent_t anchorEvt = (prachCtx && cuphyTestWorkerVec[channelWorkerMap["PRACH"]].getPrachStartEvent())
                    ? cuphyTestWorkerVec[channelWorkerMap["PRACH"]].getPrachStartEvent()
                    : startEvent.handle();

                if(puschCtx) {
                    cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].puschRxRun(
                        anchorEvt, shPtrStopEvents[channelWorkerMap["PUSCH"]]);
                }
                if(pucchCtx) {
                    cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].puschRxRun(
                        startEvent.handle(), shPtrStopEvents[channelWorkerMap["PUCCH"]], anchorEvt, anchorEvt);
                }
            } else if(g_start_delay_cfg_us.ul_anchor_mode == ul_anchor_mode_t::PUCCH) {
                if(pucchCtx) {
                    cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].puschRxRun(
                        startEvent.handle(), shPtrStopEvents[channelWorkerMap["PUCCH"]], startEvent.handle(), startEvent.handle());
                }
                cudaEvent_t anchorEvt = (pucchCtx && cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].getPucchStartEvent())
                    ? cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].getPucchStartEvent()
                    : startEvent.handle();

                if(puschCtx) {
                    cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].puschRxRun(
                        anchorEvt, shPtrStopEvents[channelWorkerMap["PUSCH"]]);
                }
                if(prachCtx) {
                    cuphyTestWorkerVec[channelWorkerMap["PRACH"]].puschRxRun(
                        startEvent.handle(), shPtrStopEvents[channelWorkerMap["PRACH"]], anchorEvt);
                }
            } else {
                if(puschCtx) {
                    cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].puschRxRun(
                        startEvent.handle(), shPtrStopEvents[channelWorkerMap["PUSCH"]]);
                }
                cudaEvent_t puschStartEvt = (puschCtx && cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getPuschStartEvent())
                    ? cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getPuschStartEvent()
                    : startEvent.handle();
                cudaEvent_t puschPucch2Evt = (puschCtx && cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getPucch2DelayStopEvent())
                    ? cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getPucch2DelayStopEvent()
                    : puschStartEvt;

                if(prachCtx) {
                    cuphyTestWorkerVec[channelWorkerMap["PRACH"]].puschRxRun(
                        startEvent.handle(), shPtrStopEvents[channelWorkerMap["PRACH"]], puschStartEvt);
                }
                if(pucchCtx) {
                    cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].puschRxRun(
                        startEvent.handle(), shPtrStopEvents[channelWorkerMap["PUCCH"]], puschPucch2Evt, puschStartEvt);
                }
            }
        } else {
            if(puschCtx) {
                cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].puschRxRun(
                    startEvent.handle(), shPtrStopEvents[channelWorkerMap["PUSCH"]]);
            }
            if(prachCtx) {
                // If PRACH delay from YAML: relative to PUSCH1 start; else relative to PUSCH1 end
                cudaEvent_t prachStart = startEvent.handle();
                if(puschCtx) {
                    cudaEvent_t puschEvent = g_start_delay_cfg_us.prach_delay_from_yaml
                        ? cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getPuschStartEvent()
                        : cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getPusch1EndEvent();
                    if(puschEvent) {
                        prachStart = puschEvent;
                    }
                }
                cuphyTestWorkerVec[channelWorkerMap["PRACH"]].puschRxRun(
                    startEvent.handle(), shPtrStopEvents[channelWorkerMap["PRACH"]], prachStart);
            }
            if(pucchCtx) {
                cuphyTestWorkerVec[channelWorkerMap["PUCCH"]].puschRxRun(
                    startEvent.handle(), shPtrStopEvents[channelWorkerMap["PUCCH"]], 
                    cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getPucch2DelayStopEvent(), 
                    cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getPuschStartEvent());
            }
        }
        if(srsCtx) {
            cuphyTestWorkerVec[channelWorkerMap["SRS"]].puschRxRun(
                startEvent.handle(), shPtrStopEvents[channelWorkerMap["SRS"]]);
        }
        if(ssbCtx) {
            cuphyTestWorkerVec[channelWorkerMap["SSB"]].pdschTxRun(
                startEvent.handle(), shPtrStopEvents[channelWorkerMap["SSB"]], 
                true /*waitRsp*/, nullptr, 
                uldl == 3 ? nullptr : cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].getSlotBoundaryEventVecPtr());
        }
        
        // cuMAC run scheduler
        if(macCtx) {
            cumacTestWorkerVec[macWorkerMap["MAC"]].macRun(
                startEvent.handle(), shPtrStopEvents[macWorkerMap["MAC"] + nCuphyCtxts], 
                true /*waitRsp*/, 
                (uldl == 3 || macInternalTimer) ? nullptr : cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].getSlotBoundaryEventVecPtr());
        }
        
        // cuMAC2 run scheduler
        if(mac2Ctx) {
            std::vector<cuphy::event>* eventVecPtr = nullptr;
            if (!(uldl == 3 || mac2InternalTimer)) {
                if (macInternalTimer) {
                    auto macEventVecOpt = cumacTestWorkerVec[macWorkerMap["MAC"]].getMacInterSlotEventVecPtr();
                    eventVecPtr = macEventVecOpt.has_value() ? *macEventVecOpt : nullptr;
                } else {
                    eventVecPtr = cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].getSlotBoundaryEventVecPtr();
                }
            }
            cumacTestWorkerVec[macWorkerMap["MAC2"]].macRun(
                startEvent.handle(), shPtrStopEvents[macWorkerMap["MAC2"] + nCuphyCtxts], 
                true /*waitRsp*/, eventVecPtr);
        }

        // Wait for all contexts to complete
        for(int i = 0; i < nCtxts; i++)
            CUDA_CHECK(cudaStreamWaitEvent(mainStream.handle(), shPtrStopEvents[i]->handle()));

#ifdef CUBB_GPU_TESTBENCH_POWER_ITERATION_BATCHING
        // **BATCHING LOGIC** - Periodically synchronize CPU-GPU execution every syncUpdateIntervalCnt iterations to prevent CPU being too far ahead of the GPU
        if(0 == (powerItrIdx % syncUpdateIntervalCnt)) {
            // Signal GPU to get started
            (*shPtrGpuStartSyncFlag)[0] = syncFlagVal;

            // Setup wait for next set of workloads
            syncFlagVal++;

            for(int i = 0; i < nCuphyCtxts; i++)
                cuphyTestWorkerVec[i].setWaitVal(syncFlagVal);
            for(int i = 0; i < nCumacCtxts; i++)
                cumacTestWorkerVec[i].setWaitVal(syncFlagVal);
        }
        // CPU-GPU synchronization in the middle of the batch
        if ((syncUpdateIntervalCnt / 2 ) == (powerItrIdx % syncUpdateIntervalCnt))
        {
            CUDA_CHECK(cudaEventRecord(gpuBatchStartEvents[powerItrIdx / syncUpdateIntervalCnt].handle(), mainStream.handle()));
            if (powerItrIdx > syncUpdateIntervalCnt)
            {
                // sync CPU and GPU to prevent CPU being too far ahead of the GPU
                CUDA_CHECK(cudaEventSynchronize(gpuBatchStartEvents[powerItrIdx / syncUpdateIntervalCnt - 1].handle()));
            }
        }
    }

    // Ungate the last wait request
    (*shPtrGpuStartSyncFlag)[0] = syncFlagVal;
#else
        // set sync event before launching next pattern
        cudaEvent_t syncEvent = nullptr;
        if(pdschCtx) { // if PDSCH run, use the middle slot for sync
            auto temp = cuphyTestWorkerVec[channelWorkerMap["PDSCH"]].getSlotBoundaryEventVecPtr();
            syncEvent = (*temp)[nSlotsPerPattern / 2].handle();
        }
        else if(puschCtx) { // if PUSCH run, use the end of PUSCH1 for sync
            syncEvent = cuphyTestWorkerVec[channelWorkerMap["PUSCH"]].getPusch1EndEvent();
        }
        else { // ungated mode if either PDSCH or PUSCH is not run
            syncEvent = nullptr;
        }
        if(syncEvent != nullptr) {
            CUDA_CHECK(cudaEventSynchronize(syncEvent));
        }
    }
#endif
    
    printf("Power iterations completed successfully\n");
}

// read SM Ids for cuPHY and cuMAC test workers
template<typename testWorkerType>
void readSmIds(std::vector<testWorkerType*>& ptestWorkerVec, cuphy::stream& strm, int gpuId)
{
    // check worker type
    std::string workerType;
    if constexpr (std::is_same<testWorkerType, cuPHYTestWorker>::value)
    {
        workerType = "cuPHY";
    } 
    else if constexpr (std::is_same<testWorkerType, cuMACTestWorker>::value) 
    {
        workerType = "cuMAC";
    } 
    else 
    {
        workerType = "Unknown";
    }

    std::vector<std::shared_ptr<cuphy::event>> shPtrRdSmIdWaitEvents(ptestWorkerVec.size());
    
    int maxSmCount = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&maxSmCount, cudaDevAttrMultiProcessorCount, gpuId));

    uint32_t wrkrIdx = 0;
    for(auto& pTestWorker : ptestWorkerVec)
    {
        pTestWorker->readSmIds(shPtrRdSmIdWaitEvents[wrkrIdx++]);
    }

    for(auto& shPtrRdSmIdWaitEvent : shPtrRdSmIdWaitEvents)
    {
        CUDA_CHECK(cudaStreamWaitEvent(strm.handle(), shPtrRdSmIdWaitEvent->handle()));
    }

    std::vector<std::vector<uint32_t>> smIds(ptestWorkerVec.size());
    wrkrIdx = 0;

    std::vector<uint32_t> workers_per_SM(maxSmCount, 0);
    std::vector<std::string> channel_for_worker(ptestWorkerVec.size());

    for(auto& pTestWorker : ptestWorkerVec)
    {
        uint32_t  nSmIds    = 0;
        uint32_t* pSmIdsGpu = pTestWorker->getSmIdsGpu(nSmIds);
        channel_for_worker[wrkrIdx] = pTestWorker->getName();
        smIds[wrkrIdx]      = std::move(std::vector<uint32_t>(nSmIds));

        CUDA_CHECK(cudaMemcpyAsync(smIds[wrkrIdx].data(), pSmIdsGpu, sizeof(uint32_t) * nSmIds, cudaMemcpyDeviceToHost, strm.handle()));
        strm.synchronize();
        std::sort(smIds[wrkrIdx].begin(), smIds[wrkrIdx].end(), std::less<uint32_t>());

        printf("%s Test Worker[%d]: SM Id counts %lu\n", workerType.c_str(), wrkrIdx, smIds[wrkrIdx].size());
        auto it = std::unique(smIds[wrkrIdx].begin(), smIds[wrkrIdx].end());
        smIds[wrkrIdx].resize(std::distance(smIds[wrkrIdx].begin(), it));
        for(int i = 0; i < smIds[wrkrIdx].size(); ++i)
        {
            printf("%02u ", smIds[wrkrIdx][i]);
            workers_per_SM[smIds[wrkrIdx][i]] |= (1 << wrkrIdx);
        }
        printf("\n");
        wrkrIdx++;
    }
#if 0
    printf("for_grep,sm #");
    for (int j = 0; j < ptestWorkerVec.size(); j++) printf(",worker-%d-%s",j,channel_for_worker[j].c_str());
    printf("\n");
    for (int i = 0; i < maxSmCount; i++) {
        printf("for_grep,%d", i);
        for (int j = 0; j < ptestWorkerVec.size(); j++) {
            printf(",%d", (workers_per_SM[i] >> j) & 0x1);
        }
        printf("\n");
    }
#endif
}
