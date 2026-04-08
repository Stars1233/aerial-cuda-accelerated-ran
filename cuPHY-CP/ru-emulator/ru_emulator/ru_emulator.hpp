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

#ifndef RU_EMULATOR_H__
#define RU_EMULATOR_H__
// #define STANDALONE
#include <getopt.h>
#include <signal.h>

#include <cinttypes>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <memory>
#include <map>
#include <unordered_set>
#include <mutex>
#include "aerial-fh-driver/fh_mutex.hpp"
#include "aerial-fh-driver/api.hpp"
#include "aerial-fh-driver/packet_stats.hpp"

#include "utils.hpp"
#include "aerial-fh-driver/oran.hpp"
#include "perf_metrics/perf_metrics_accumulator.hpp"
#include "oran_structs.hpp"
#include "defines.hpp"
#include "cuphyoam.hpp"
#include "hdf5hpp.hpp"
#include "timing_utils.hpp"

using FssDynBfwBeamIdArray = std::array<std::array<std::array<std::array<std::array<uint16_t, 2>, ORAN_MAX_PRB_X_SLOT>, MAX_LAUNCH_PATTERN_SLOTS>, MAX_AP_PER_SLOT>, MAX_CELLS_PER_SLOT>;
using FssDisabledBfwDynBfwBeamIdArray = std::array<std::array<std::array<std::array<std::array<std::array<uint64_t, 4>, ORAN_MAX_PRB_X_SLOT>, SLOT_NUM_SYMS>, MAX_LAUNCH_PATTERN_SLOTS>, MAX_AP_PER_SLOT>, MAX_CELLS_PER_SLOT>;
using FssDisabledBfwDynBfwBeamIdCntArray = std::array<std::array<std::array<std::array<std::atomic<int>, SLOT_NUM_SYMS>, MAX_LAUNCH_PATTERN_SLOTS>, MAX_AP_PER_SLOT>, MAX_CELLS_PER_SLOT>;
using FssDynBfwBeamIdLastValidationTsArray = std::array<std::array<std::array<std::atomic<uint64_t>, MAX_LAUNCH_PATTERN_SLOTS>, MAX_AP_PER_SLOT>, MAX_CELLS_PER_SLOT>;
using FssDynBfwBeamIdMtxArray = std::array<std::array<std::array<aerial_fh::FHMutex, MAX_LAUNCH_PATTERN_SLOTS>, MAX_AP_PER_SLOT>, MAX_CELLS_PER_SLOT>;
using FssPdschPrbSeenArray = std::array<std::array<std::unordered_map<int, std::unordered_map<int, std::array<std::unordered_map<uint16_t, uint16_t>, OFDM_SYMBOLS_PER_SLOT>>>, MAX_FLOWS_PER_DL_CORE>, MAX_CELLS_PER_SLOT>;

#define fss_dyn_bfw_beam_id (*RU_Emulator::fss_dyn_bfw_beam_id_ptr)
#define fss_disabled_bfw_dyn_bfw_beam_id (*RU_Emulator::fss_disabled_bfw_dyn_bfw_beam_id_ptr)
#define fss_disabled_bfw_dyn_bfw_beam_id_cnt (*RU_Emulator::fss_disabled_bfw_dyn_bfw_beam_id_cnt_ptr)
#define fss_dyn_bfw_beam_id_last_validation_ts (*RU_Emulator::fss_dyn_bfw_beam_id_last_validation_ts_ptr)
#define fss_dyn_bfw_beam_id_mtx (*RU_Emulator::fss_dyn_bfw_beam_id_mtx_ptr)

void *cplane_core_wrapper(void *arg);
void *uplane_proc_rx_core_wrapper(void *arg);
void *uplane_proc_validate_core_wrapper(void *arg);
void *uplane_proc_core_wrapper(void *arg);
void *uplane_tx_only_core_wrapper(void *arg);
#ifdef STANDALONE
void* standalone_core_wrapper(void *arg);
#endif
void* oam_thread_func_wrapper(void* arg);

struct srs_slot_type_info {
    int first_symbol{};
    int num_symbols{};
    int txq_base_offset{};
    int64_t slot_time_offset_ns{};  //!< Time offset from s3's t0 (0 for s3, -500us for s4, -1000us for s5)
};

int get_txq_index(const ul_channel channel_type, const int launch_pattern_slot, const int symbol, const int16_t eaxcId_index,
                  const int split_srs_txq, const int enable_srs_eaxcid_pacing,
                  const srs_slot_type_info& srs_s3_info, const srs_slot_type_info& srs_s4_info, const srs_slot_type_info& srs_s5_info,
                  const int srs_pacing_eaxcids_per_symbol, const int srs_pacing_eaxcids_per_tx_window);

class RU_Emulator
{
    public:
        /**
         * Initialize the RU Emulator
         *
         * @param[in] argc Command line argument count
         * @param[in] argv Command line argument vector
         * @return Thread ID of the logging thread, or -1 on failure
         */
        pthread_t init(int argc, char ** argv);

        /**
         * Perform minimal initialization of the RU Emulator without starting logging
         *
         * This is a lightweight initialization that parses configuration and sets up
         * basic structures but does not start the logging thread. Useful for testing
         * or when only configuration parsing is needed.
         *
         * @param[in] argc Command line argument count
         * @param[in] argv Command line argument vector
         */
        void init_minimal(int argc, char ** argv);

        /**
         * Start the RU Emulator worker threads and main processing loop
         *
         * @return RE_OK on success, RE_ERR on failure
         */
        int start();

        /**
         * Finalize and cleanup the RU Emulator
         *
         * Prints statistics, flushes counters, and cleans up resources
         *
         * @return RE_OK on success, RE_ERR on failure
         */
        int finalize();
        int finalize_dlc_tb();

        /////////////////////////////////////////////////////
        //// INIT FUNCTIONS
        /////////////////////////////////////////////////////
        /**
         * Set default configuration values for all RU Emulator options
         */
        void set_default_configs();

        /**
         * Parse command line arguments
         *
         * @param[in] argc Argument count
         * @param[in] argv Argument vector
         */
        void get_args(int argc, char ** argv);

        /**
         * Parse the main YAML configuration file
         *
         * @param[in] yaml_file Path to the YAML configuration file
         */
        void parse_yaml(std::string yaml_file);
        void initialize_srs_slot_info();

        /**
         * Parse the launch pattern YAML file
         *
         * @param[in] yaml_file Path to the launch pattern YAML file
         */
        void parse_launch_pattern(std::string yaml_file);

        /**
         * Pre-process launch pattern v2 test vectors
         *
         * @param[in,out] root YAML root node containing launch pattern configuration
         */
        void launch_pattern_v2_tv_pre_processing(yaml::node& root);

        /**
         * Extract cell configurations from launch pattern YAML
         *
         * @param[in] lp_yaml_file Path to launch pattern YAML file
         */
        void cell_configs_from_lp_yaml(std::string lp_yaml_file);

        /**
         * Parse digital beamforming table (DBT) configurations from HDF5 file
         *
         * @param[in] file HDF5 file handle
         * @param[in] cell_id Cell identifier
         * @return 0 on success, negative error code on failure
         */
        int  parse_dbt_configs(hdf5hpp::hdf5_file& file, int cell_id);

        /**
         * Load H5 configuration parameters for a specific cell
         *
         * @param[in] cell_id Cell identifier
         * @param[in] config_params_h5_file Path to H5 config file
         * @return 0 on success, negative error code on failure
         */
        int  load_h5_config_params(int cell_id, const char* config_params_h5_file);

        /**
         * Extract uplink port information from PDU parameters
         *
         * @param[in] pdu_pars PDU parameters dataset element
         * @param[out] pdu_info PDU information structure to populate
         */
        void get_ul_ports(hdf5hpp::hdf5_dataset_elem &pdu_pars, pdu_info &pdu_info);

        /**
         * Parse a specific channel from the launch pattern
         *
         * @param[in] root YAML root node
         * @param[in] key Channel key to parse
         * @param[in,out] tv_object Test vector object to populate
         */
        void parse_launch_pattern_channel(yaml::node& root, std::string key, tv_object& tv_object);

        /**
         * Assign test vectors from launch pattern YAML
         *
         * @param[in] root YAML root node
         * @param[in] key Key to look up in YAML
         * @param[out] tvs Vector to store test vector names
         * @param[out] tv_map Map from TV name to index
         */
        void yaml_assign_launch_pattern_tv(yaml::node root, std::string key, std::vector<std::string>& tvs, std::unordered_map<std::string, int>& tv_map);

        /**
         * Assign launch pattern from YAML configuration
         *
         * @param[in] root YAML root node
         * @param[in] channel Channel name
         * @param[out] launch_pattern Launch pattern matrix to populate
         * @param[in] tv_map Map from TV name to index
         * @param[in] num_cells Number of cells
         */
        void yaml_assign_launch_pattern(yaml::node root, std::string channel, launch_pattern_matrix& launch_pattern, std::unordered_map<std::string, int>& tv_map, int num_cells);

        /**
         * Verify and apply parsed configurations
         */
        void verify_and_apply_configs();

        /**
         * Verify configuration validity
         */
        void verify_configs();

        /**
         * Apply verified configurations to the system
         */
        void apply_configs();

        /**
         * Print current configuration values
         */
        void print_configs();

        /**
         * Load all test vectors (UL and DL)
         */
        void load_tvs();

        /**
         * Load uplink test vectors for all UL channels
         */
        void load_ul_tvs();

        /**
         * Load PUSCH (Physical Uplink Shared Channel) test vectors
         */
        void load_pusch_tvs();

        /**
         * Load PRACH (Physical Random Access Channel) test vectors
         */
        void load_prach_tvs();

        /**
         * Load PUCCH (Physical Uplink Control Channel) test vectors
         */
        void load_pucch_tvs();

        /**
         * Load SRS (Sounding Reference Signal) test vectors
         */
        void load_srs_tvs();

        /**
         * Load downlink test vectors for all DL channels
         */
        void load_dl_tvs();

        /**
         * Load PDSCH (Physical Downlink Shared Channel) test vectors
         */
        void load_pdsch_tvs();

        /**
         * Load PBCH (Physical Broadcast Channel) test vectors
         */
        void load_pbch_tvs();

        /**
         * Load PDCCH (Physical Downlink Control Channel) UL test vectors
         */
        void load_pdcch_ul_tvs();

        /**
         * Load PDCCH (Physical Downlink Control Channel) DL test vectors
         */
        void load_pdcch_dl_tvs();

        /**
         * Load PDCCH test vectors into specified TV object
         *
         * @param[in,out] tv_object Test vector object to populate with PDCCH data
         */
        void load_pdcch_tvs(struct dl_tv_object& tv_object);

        /**
         * Check if a CSI-RS resource element is within PDSCH allocation
         *
         * @param[in] filename Test vector filename
         * @param[in] re_index Resource element index
         * @return true if RE is in PDSCH, false otherwise
         */
        bool is_csirs_re_in_pdsch(std::string filename, int re_index);

        /**
         * Check if CSI-RS test vector has overlapping PDSCH
         *
         * @param[in] filename Test vector filename
         * @return true if PDSCH overlaps with CSI-RS, false otherwise
         */
        bool does_csirs_tv_have_pdsch(std::string filename);

        /**
         * Flag PDSCH test vector that has zero-power CSI-RS
         *
         * @param[in] filename Test vector filename
         */
        void flag_pdsch_tv_with_zp_csirs(std::string filename);

        /**
         * Read cell configuration from test vector HDF5 file
         *
         * @param[in] hdf5file HDF5 file handle
         * @param[out] tv_info Test vector information structure
         * @param[in] tv_name Test vector name
         */
        void read_cell_cfg_from_tv(hdf5hpp::hdf5_file & hdf5file, struct tv_info & tv_info, std::string & tv_name);

        /**
         * Add count of overlapping non-zero-power CSI-RS to PDSCH TV
         *
         * @param[in] filename Test vector filename
         * @param[in] numOverlappingCsirs Number of overlapping CSI-RS
         * @param[in] fullyOverlapping Whether CSI-RS fully overlaps PDSCH
         */
        void add_num_overlapping_nzp_csirs_pdsch(std::string filename, uint32_t numOverlappingCsirs, bool fullyOverlapping);

        /**
         * Load CSI-RS (Channel State Information Reference Signal) test vectors
         */
        void load_csirs_tvs();

        /**
         * Load beamforming weight (BFW) test vectors
         *
         * @param[in] dirDL true for downlink BFW, false for uplink BFW
         */
        void load_bfw_tvs(bool dirDL);

        /**
         * Generate PRB (Physical Resource Block) map from TV info and PDU info
         *
         * @param[in,out] tv_info_ Test vector information
         * @param[in] pdu_infos Vector of PDU information
         */
        void generate_prb_map(struct tv_info &tv_info_, std::vector<pdu_info> &pdu_infos);

        /**
         * Apply test vector configurations to the system
         */
        void apply_tv_configs();

        /**
         * Setup slot data structures for UL and DL
         */
        void setup_slots();

        /**
         * Setup uplink slot data structures
         */
        void setup_ul_slots();

        /**
         * Setup uplink counters for a TV object
         *
         * @param[in,out] tv_object Test vector object to initialize counters for
         */
        void setup_ul_counters(struct ul_tv_object& tv_object);

        /**
         * Setup downlink slot data structures
         */
        void setup_dl_slots();

        /**
         * Setup downlink counters for a TV object
         *
         * @param[in,out] tv_object Test vector object to initialize counters for
         */
        void setup_dl_counters(struct dl_tv_object& tv_object);

        /**
         * Setup downlink receive byte counters
         *
         * @param[in,out] tv_object Test vector object to initialize byte counters for
         */
        void setup_dl_receive_bytes(struct dl_tv_object& tv_object);

        /**
         * Setup downlink atomic counters for thread-safe counting
         *
         * @param[in,out] tv_object Test vector object to initialize atomic counters for
         */
        void setup_dl_atomic_counters(struct dl_tv_object& tv_object);

        /**
         * Setup ring buffers for packet communication
         */
        void setup_rings();

        /**
         * Setup uplink transmit ring buffers
         */
        void setup_ul_tx_rings();

        /**
         * Initialize OAM (Operations, Administration, and Maintenance) subsystem
         */
        void oam_init();

        /**
         * OAM thread function for handling OAM commands
         *
         * @param[in] arg Thread argument (RU_Emulator instance)
         * @return nullptr on thread exit
         */
        void* oam_thread_func(void* arg);

        /////////////////////////////////////////////////////
        //// FH DRIVER INIT FUNCTIONS
        /////////////////////////////////////////////////////
        /**
         * Initialize fronthaul driver
         */
        void init_fh();

        /**
         * Open and configure the fronthaul driver
         */
        void open_fh_driver();

        /**
         * Add network interface cards (NICs) to fronthaul driver
         */
        void add_nics();

        /**
         * Add peer network endpoints to fronthaul driver
         */
        void add_peers();

        /**
         * Add packet flows for all channels
         */
        void add_flows();

        /**
         * Start the fronthaul driver processing
         */
        void start_fh_driver();

        /////////////////////////////////////////////////////
        //// CPLANE
        /////////////////////////////////////////////////////
        /**
         * Main C-plane processing core worker thread
         *
         * @param[in] arg Thread argument containing core parameters
         * @return nullptr on thread exit
         */
        void *cplane_core(void *arg);

        /**
         * Transmit a complete slot of uplink data
         *
         * @param[in] slot_tx Slot transmission information
         * @param[in] cell_index Cell identifier
         * @param[in,out] timers Timing metrics for this slot
         * @param[in] txqs Transmit queue handles
         * @param[in] tx_request Transmit request handle
         * @param[in] enable_mmimo Whether mMIMO is enabled
         * @param[in] profiler Performance metrics accumulator
         * @return 0 on success, negative error code on failure
         */
        int tx_slot(slot_tx_info & slot_tx, int cell_index, tx_symbol_timers& timers, aerial_fh::TxqHandle* txqs, aerial_fh::TxRequestHandle* tx_request, bool enable_mmimo, perf_metrics::PerfMetricsAccumulator* profiler);

        /**
         * C-plane-gated pre-computed TX path.  Iterates C-plane infos
         * (like legacy tx_slot) but uses the pre-built cache for
         * IQ pointer resolution, flow handles, TXQ indices, and TX
         * timing offsets.  Section IDs, FSS, and slot_t0 come from
         * the live C-plane messages, preserving correct DU timing.
         *
         * @param[in,out] slot_tx           Slot TX info with C-plane messages
         * @param[in] cell_index            Cell identifier
         * @param[in,out] timers            Timing metrics
         * @param[in] txqs                  Transmit queue handles
         * @param[in] tx_request            Transmit request handle
         * @param[in] profiler              Performance metrics accumulator
         * @return number of U-plane packets transmitted
         */
        int tx_slot_precomputed(slot_tx_info& slot_tx, int cell_index,
                                tx_symbol_timers& timers,
                                aerial_fh::TxqHandle* txqs,
                                aerial_fh::TxRequestHandle* tx_request,
                                perf_metrics::PerfMetricsAccumulator* profiler);

        /**
         * Build the pre-computed UL TX cache from launch patterns and TVs.
         * Called once after test vectors are loaded and peers are configured.
         */
        void precompute_ul_tx_cache();

        /**
         * Handle section type 1 C-plane message (UL/DL data channels)
         *
         * @param[in,out] c_plane_info C-plane packet information
         * @param[in] cell_index Cell identifier
         * @param[in,out] timers Timing metrics
         * @param[in] txqs Transmit queue handles
         * @param[in] tx_request Transmit request handle
         * @param[in] sym Symbol index
         * @param[in] profiler Performance metrics accumulator
         * @return 0 on success, negative error code on failure
         */
        int handle_sect1_c_plane(oran_c_plane_info_t& c_plane_info, uint16_t cell_index, tx_symbol_timers& timers, aerial_fh::TxqHandle* txqs, aerial_fh::TxRequestHandle* tx_request, int sym, perf_metrics::PerfMetricsAccumulator* profiler);

        /**
         * Handle section type 3 C-plane message (PRACH)
         *
         * @param[in,out] c_plane_info C-plane packet information
         * @param[in] cell_index Cell identifier
         * @param[in,out] timers Timing metrics
         * @param[in] txqs Transmit queue handles
         * @param[in] tx_request Transmit request handle
         * @param[in] sym Symbol index
         * @return 0 on success, negative error code on failure
         */
        int handle_sect3_c_plane(oran_c_plane_info_t& c_plane_info, uint16_t cell_index, tx_symbol_timers& timers, aerial_fh::TxqHandle* txqs, aerial_fh::TxRequestHandle* tx_request, int sym);

        /**
         * Handle section type 1 C-plane message (legacy, non-mMIMO)
         *
         * @param[in,out] c_plane_info C-plane packet information
         * @param[in] cell_index Cell identifier
         * @param[in,out] timers Timing metrics
         * @param[in] txqs Transmit queue handles
         * @param[in] tx_request Transmit request handle
         * @return 0 on success, negative error code on failure
         */
        int handle_sect1_c_plane_v2(oran_c_plane_info_t& c_plane_info, uint16_t cell_index, tx_symbol_timers& timers, aerial_fh::TxqHandle* txqs, aerial_fh::TxRequestHandle* tx_request);

        /**
         * Handle section type 3 C-plane message (legacy, non-mMIMO)
         *
         * @param[in,out] c_plane_info C-plane packet information
         * @param[in] cell_index Cell identifier
         * @param[in,out] timers Timing metrics
         * @param[in] txqs Transmit queue handles
         * @param[in] tx_request Transmit request handle
         * @return 0 on success, negative error code on failure
         */
        int handle_sect3_c_plane_v2(oran_c_plane_info_t& c_plane_info, uint16_t cell_index, tx_symbol_timers& timers, aerial_fh::TxqHandle* txqs, aerial_fh::TxRequestHandle* tx_request);

        /**
         * Transmit a single symbol of uplink data
         *
         * @param[in] slot Slot data containing IQ samples
         * @param[in,out] tx_symbol_info Symbol transmission helper data
         * @param[in] blank_prbs Pointer to blank PRB data for filling gaps
         * @param[in,out] timers Timing metrics for this symbol
         * @param[in] section_type ORAN section type
         * @param[in,out] uplane_msg U-plane message to populate and send
         */
        void tx_symbol(Slot& slot, tx_symbol_helper& tx_symbol_info, void * blank_prbs, tx_symbol_timers& timers, uint8_t section_type, aerial_fh::UPlaneMsgMultiSectionSendInfo& uplane_msg);

        /**
         * Increment section receive counter for a specific symbol
         *
         * @param[in] c_plane_info C-plane packet information
         * @param[in] cell_index Cell identifier
         * @param[in] sym Symbol index
         */
        void increment_section_rx_counter(oran_c_plane_info_t& c_plane_info, uint16_t cell_index, int sym);

        /**
         * Increment section receive counter (version 2, symbol-agnostic)
         *
         * @param[in] c_plane_info C-plane packet information
         * @param[in] cell_index Cell identifier
         */
        void increment_section_rx_counter_v2(oran_c_plane_info_t& c_plane_info, uint16_t cell_index);

        /**
         * @brief Classifies sections by PRB-range matching in SINGLE_SECT_MODE
         *        and updates section RX counters.
         *
         * For SINGLE_SECT_MODE RU types, iterates each section in @p c_plane_info
         * and reclassifies it as PUCCH / SRS / PUSCH via prb_range_matching(),
         * updating both section_info and c_plane_info tv_object / channel_type
         * before calling increment_section_rx_counter_v2().  For other RU types,
         * calls increment_section_rx_counter_v2() directly.
         *
         * @param[in,out] c_plane_info  C-plane info whose sections may be reclassified.
         * @param[in]     cell_index    Cell identifier.
         *
         * @see increment_section_rx_counter_v2
         * @see prb_range_matching
         */
        inline void update_ul_throughput_counters(oran_c_plane_info_t& c_plane_info, uint16_t cell_index);

        /**
         * Process C-plane packet timing and update timing statistics
         *
         * @param[in] cell_index Cell identifier
         * @param[in,out] packet_timer Packet timing structure to update
         * @param[in] c_plane_info C-plane packet information
         * @param[in] packet_time Packet timestamp
         * @param[in] toa Time of arrival relative to slot boundary
         * @param[in] slot_t0 Slot start time
         */
        void process_cplane_timing(uint8_t cell_index, struct packet_timer_per_slot& packet_timer, oran_c_plane_info_t& c_plane_info, uint64_t packet_time, int64_t toa, int64_t slot_t0);

        /**
         * Prepare C-plane information structure for slot transmission
         *
         * @param[in] pusch_object PUSCH test vector object
         * @param[in] cell_configs Vector of cell configurations
         * @param[out] slot_tx Slot transmission info to populate
         * @param[in] fss Frame/subframe/slot identifier
         * @param[in] cell_index Cell identifier
         * @param[in] slot_idx Slot index in launch pattern
         * @param[in] next_slot_time Next slot start time
         * @param[in] first_f0s0s0_time Time of first frame 0 subframe 0 slot 0
         * @param[in] frame_cycle_time_ns Frame cycle duration in nanoseconds
         * @param[in] max_slot_id Maximum slot ID value
         * @param[in] opt_tti_us TTI duration in microseconds
         * @param[out] slot_t0 Computed slot start time
         */
        void prepare_cplane_info(ul_tv_object& pusch_object, const std::vector<struct cell_config>& cell_configs, 
                                       slot_tx_info& slot_tx, const struct fssId& fss, uint8_t cell_index, 
                                       int slot_idx, int64_t next_slot_time, int64_t first_f0s0s0_time, 
                                       int64_t frame_cycle_time_ns, int max_slot_id, int opt_tti_us, int64_t& slot_t0);

        /**
         * Worker thread for transmit-only mode (UL only, no DL processing)
         *
         * @param[in] arg Thread argument containing core parameters
         * @return nullptr on thread exit
         */
        void *uplane_tx_only_core(void *arg);

        /**
         * Worker state tracking for performance tracing
         */
        enum class WorkerState {
            RECEIVING,      //!< Worker is receiving packets
            SPINNING,       //!< Worker is spinning/waiting
            PROCESSING,     //!< Worker is processing packets
            TRANSMITTING    //!< Worker is transmitting packets
        };

        /**
         * Receive packets from fronthaul driver with round-robin cell selection
         *
         * @param[out] cell_index Cell index that received packets
         * @param[out] info Array of receive message information structures
         * @param[out] nb_rx Number of packets received
         * @param[out] rte_rx_time Receive timestamp
         * @param[in] threadname Thread name for logging
         * @param[in,out] current_state Current worker state
         * @param[in,out] previous_state Previous worker state
         * @param[in] first_f0s0s0_time Time of first frame 0 subframe 0 slot 0
         * @param[in,out] round_robin_counter Round-robin cell selection counter
         * @param[in] start_cell_index Starting cell index for this worker
         * @param[in] num_cells_per_core Number of cells handled by this worker
         * @param[in,out] tx_request Transmit request handle
         * @param[in] tx_request_init Whether TX request is initialized
         * @param[in] is_srs Whether processing SRS channel
         * @return Number of packets received
         */
        size_t receive_packets(uint8_t& cell_index, aerial_fh::MsgReceiveInfo* info, size_t& nb_rx, uint64_t& rte_rx_time, 
                                     const char* threadname, WorkerState& current_state, WorkerState& previous_state, 
                                     uint64_t first_f0s0s0_time, uint8_t& round_robin_counter, 
                                     int start_cell_index, int num_cells_per_core, 
                                     aerial_fh::TxRequestHandle& tx_request, bool tx_request_init, bool is_srs);

        /**
         * Parse C-plane packet header and metadata
         *
         * @param[out] c_plane_info C-plane information structure to populate
         * @param[in] nb_rx Number of received packets in burst
         * @param[in] index_rx Index of current packet in burst
         * @param[in] rte_rx_time Receive timestamp
         * @param[in] mbuf_payload Pointer to packet payload
         * @param[in] buffer_length Length of @p mbuf_payload in bytes
         * @param[in] cell_index Cell identifier
         */
        void parse_c_plane(oran_c_plane_info_t& c_plane_info, int nb_rx, int index_rx, uint64_t rte_rx_time, uint8_t* mbuf_payload, size_t buffer_length, int cell_index);

        /**
         * Parse all sections within a C-plane packet
         *
         * @param[in,out] c_plane_info C-plane information structure
         * @param[in] nb_rx Number of received packets in burst
         * @param[in] index_rx Index of current packet in burst
         * @param[in] rte_rx_time Receive timestamp
         * @param[in] mbuf_payload Pointer to packet payload
         */
        void parse_c_plane_sections(oran_c_plane_info_t& c_plane_info, int nb_rx, int index_rx, uint64_t rte_rx_time, uint8_t* mbuf_payload);

        /**
         * Parse a single C-plane section
         *
         * @param[out] section_info Section information structure to populate
         * @param[in] section_type ORAN section type (1 or 3)
         * @param[in] section_ptr Pointer to section data
         * @return Number of bytes parsed
         */
        size_t parse_c_plane_section(oran_c_plane_section_info_t& section_info, int section_type, uint8_t* section_ptr);

        /**
         * Parse all section extensions within a C-plane section
         *
         * @param[in,out] section_info Section information structure
         * @param[in] section_ext_ptr Pointer to section extension data
         * @return Number of bytes parsed
         */
        size_t parse_c_plane_section_extensions(oran_c_plane_section_info_t& section_info, uint8_t* section_ext_ptr);

        /**
         * Parse a single C-plane section extension
         *
         * @param[out] section_ext_info Section extension information structure
         * @param[in] section_ext_ptr Pointer to section extension data
         * @return Number of bytes parsed
         */
        size_t parse_c_plane_section_extension(oran_c_plane_section_ext_info_t& section_ext_info, uint8_t* section_ext_ptr);

        /**
         * Track and update packet statistics
         *
         * @param[in] c_plane_info C-plane packet information
         * @param[in] cell_index Cell identifier
         */
        void track_stats(oran_c_plane_info_t& c_plane_info, uint16_t cell_index);

        /////////////////////////////////////////////////////
        //// UPLANE
        /////////////////////////////////////////////////////
        /**
         * U-plane receive core worker thread
         *
         * Receives and buffers U-plane packets for validation
         *
         * @param[in] arg Thread argument containing core info
         * @return nullptr on thread exit
         */
        void *uplane_proc_rx_core(void *arg);

        /**
         * U-plane validation core worker thread
         *
         * Validates received U-plane packets against test vectors
         *
         * @param[in] arg Thread argument containing core info
         * @return nullptr on thread exit
         */
        void *uplane_proc_validate_core(void *arg);

        /**
         * U-plane processing core worker thread (combined receive and validate)
         *
         * @param[in] arg Thread argument containing core info
         * @return nullptr on thread exit
         */
        void *uplane_proc_core(void *arg);

        /**
         * Find eAxC (antenna-carrier) index from flow value
         *
         * @param[in] cell_config Cell configuration containing eAxC mappings
         * @param[in] flowVal Flow identifier value
         * @param[out] index Resolved eAxC index
         * @return 0 on success, negative error code if not found
         */
        int find_eAxC_index_from_flowVal(cell_config& cell_config,uint16_t flowVal,uint8_t& index);
#ifdef STANDALONE
        /////////////////////////////////////////////////////
        //// STANDALONE
        /////////////////////////////////////////////////////
        /**
         * Standalone mode core worker thread
         *
         * @param[in] arg Thread argument
         * @return nullptr on thread exit
         */
        void* standalone_core(void *arg);

        /**
         * Check if a slot has uplink data for a specific cell
         *
         * @param[in] launch_pattern_slot Slot index in launch pattern
         * @param[in] cell_index Cell identifier
         * @param[in] tv_object UL test vector object to check
         * @return true if slot has UL data, false otherwise
         */
        bool has_ul_for_slot(int launch_pattern_slot, int cell_index, ul_tv_object& tv_object);
#endif

        /////////////////////////////////////////////////////
        //// HELPER FUNCTIONS
        /////////////////////////////////////////////////////
        /**
         * Check if a slot has a specific DL channel for a cell
         *
         * @param[in] cell_index Cell identifier
         * @param[in] tv_object DL test vector object
         * @param[in] launch_pattern_slot Slot index in launch pattern
         * @return true if slot has the channel, false otherwise
         */
        bool does_slot_have_channel_for_cell(uint8_t cell_index, struct dl_tv_object* tv_object, uint8_t launch_pattern_slot);

        /**
         * Generate throughput log message
         *
         * @param[out] buffer Buffer to write log message to
         * @param[in] seconds_count Number of seconds elapsed
         */
        void generate_throughput_log(char* buffer, uint64_t seconds_count);

        /**
         * Generate results log for a DL test vector object
         *
         * @param[in] tv_object DL test vector object with counters
         */
        void generate_results_log(struct dl_tv_object& tv_object);

        /**
         * Generate results string from uint32_t counter array
         *
         * @param[in] base Base string to prepend
         * @param[in] counter Array of atomic counters
         */
        void generate_results_string(char* base, std::array<std::atomic<uint32_t>, MAX_CELLS_PER_SLOT>& counter);

        /**
         * Generate results string from uint64_t counter array
         *
         * @param[in] base Base string to prepend
         * @param[in] counter Array of atomic counters
         */
        void generate_results_string(char* base, std::array<std::atomic<uint64_t>, MAX_CELLS_PER_SLOT>& counter);

        /**
         * Generate and log slot count statistics
         */
        void generate_slot_count_log();

        /**
         * Generate and log maximum sections per slot statistics
         */
        void generate_max_sections_count_log();
    
        /**
         * Generate timing results table with packet and slot timing statistics
         */
        void generate_timing_results_table();

        /**
         * Generate slot-level timing results table
         */
        void generate_slot_level_timing_results_table();

        /**
         * Reset all counters for a specific cell
         *
         * @param[in] cell_index Cell identifier
         */
        void reset_cell_counters(uint16_t cell_index);

        /**
         * Flush remaining slot timing counters at end of run
         */
        void flush_slot_timing_counters();

        /**
         * Print divider line for result tables
         *
         * @param[in] num_cells Number of cells to size divider for
         */
        void print_divider(int num_cells);

        /**
         * Print aggregate timing statistics
         *
         * @param[in] packet_type Packet type string (e.g., "DL C", "UL C")
         * @param[in] metric Metric name (e.g., "AVG", "MIN")
         * @param[in] value Metric value to print
         */
        void print_aggr_times(std::string packet_type, std::string metric, float value);

        /**
         * Get expected frame/subframe/slot from timestamp
         */
        void get_expected_fss_from_ts();

        /**
         * Check if PRB range in C-plane matches expected PDU configuration
         *
         * @param[in] c_plane_info C-plane packet information
         * @param[in] pdu_infos Vector of expected PDU configurations
         * @return true if PRB range is valid, false otherwise
         */
        bool prb_range_check(oran_c_plane_info_t &c_plane_info, std::vector<pdu_info>& pdu_infos);

        /**
         * Check if PRB range in C-plane section matches expected PDU configuration
         *
         * @param[in] c_plane_info C-plane packet information
         * @param[in] section_info Section information to check
         * @param[in] pdu_infos Vector of expected PDU configurations
         * @return true if PRB range is valid, false otherwise
         */
        bool prb_range_check(oran_c_plane_info_t &c_plane_info, oran_c_plane_section_info_t &section_info, std::vector<pdu_info>& pdu_infos);

        /**
         * Verify PRB range in C-plane matches uplink test vector
         *
         * @param[in] c_plane_info C-plane packet information
         * @param[in] cell_index Cell identifier
         * @param[in] tv_obj UL test vector object
         * @return true if PRB range matches, false otherwise
         */
        bool prb_range_matching(oran_c_plane_info_t& c_plane_info, uint16_t cell_index, struct ul_tv_object& tv_obj);

        /**
         * Verify PRB range in C-plane matches downlink test vector
         *
         * @param[in] c_plane_info C-plane packet information
         * @param[in] cell_index Cell identifier
         * @param[in] tv_obj DL test vector object
         * @return true if PRB range matches, false otherwise
         */
        bool prb_range_matching(oran_c_plane_info_t& c_plane_info, uint16_t cell_index, struct dl_tv_object& tv_obj);

        /**
         * Verify PRB range in C-plane section matches uplink test vector
         *
         * @param[in] c_plane_info C-plane packet information
         * @param[in] section_info Section information to check
         * @param[in] cell_index Cell identifier
         * @param[in] tv_obj UL test vector object
         * @return true if PRB range matches, false otherwise
         */
        bool prb_range_matching(oran_c_plane_info_t& c_plane_info, oran_c_plane_section_info_t &section_info, uint16_t cell_index, struct ul_tv_object& tv_obj);

        /**
         * Check if C-plane channel type matches expected uplink channel
         *
         * @param[in] c_plane_info C-plane packet information
         * @param[in] cell_index Cell identifier
         * @param[in] tv_obj UL test vector object
         * @return true if channel type is correct, false otherwise
         */
        bool c_plane_channel_type_checking(oran_c_plane_info_t &c_plane_info, uint16_t cell_index, struct ul_tv_object& tv_obj);

        /**
         * Check if C-plane channel type matches expected downlink channel
         *
         * @param[in] c_plane_info C-plane packet information
         * @param[in] cell_index Cell identifier
         * @param[in] tv_obj DL test vector object
         * @return true if channel type is correct, false otherwise
         */
        bool c_plane_channel_type_checking(oran_c_plane_info_t &c_plane_info, uint16_t cell_index, struct dl_tv_object& tv_obj);

        using prb_map_t = std::array<std::array<bool, MAX_NUM_PRBS_PER_SYMBOL>, OFDM_SYMBOLS_PER_SLOT>;

        /**
         * Resolve the PRB allocation map for a given UL channel in the current slot/cell.
         * Used to hoist the launch pattern lookup out of per-section loops.
         *
         * @param[in] c_plane_info C-plane packet information
         * @param[in] cell_index Cell identifier
         * @param[in] tv_obj UL test vector object
         * @return Pointer to the prb_map, or nullptr if channel has no allocation in this slot/cell
         */
        const prb_map_t* resolve_prb_map(oran_c_plane_info_t &c_plane_info, uint16_t cell_index, struct ul_tv_object& tv_obj);

        /**
         * Determine channel type from C-plane packet properties
         *
         * @param[in,out] c_plane_info C-plane packet information
         * @param[in] cell_index Cell identifier
         */
        void find_channel_type(oran_c_plane_info_t& c_plane_info, uint16_t cell_index);

        /**
         * Determine channel type for each section in C-plane packet
         *
         * @param[in,out] c_plane_info C-plane packet information
         * @param[in] cell_index Cell identifier
         */
        void find_channel_type_for_each_section(oran_c_plane_info_t& c_plane_info, uint16_t cell_index);

        /**
         * Check if packet should be dropped for testing purposes
         *
         * @param[in] cell_index Cell identifier
         * @param[in] ch UL channel type
         * @param[in] fss Frame/subframe/slot identifier
         * @return true if packet should be dropped, false otherwise
         */
        bool check_if_drop(uint16_t cell_index, ul_channel ch, const struct fssId& fss);

        /**
         * Print C-plane beam IDs for debugging
         *
         * @param[in] info Message receive information
         * @param[in] numberOfSections Number of sections in packet
         * @param[in] section_type ORAN section type
         */
        void print_c_plane_beamid(aerial_fh::MsgReceiveInfo &info, uint8_t numberOfSections, uint8_t section_type);

        /**
         * Normalize counters by dividing by number of DL cores per cell
         *
         * @param[in,out] tv_object DL test vector object with counters to normalize
         */
        void normalize_counters(struct dl_tv_object& tv_object);

        /**
         * Validate received PBCH (Physical Broadcast Channel) U-plane data
         *
         * @param[in] cell_index Cell identifier
         * @param[in] header_info U-plane packet header information
         * @param[in] buffer Pointer to received IQ data
         * @param[in,out] prev_pbch_time Previous PBCH receive time for duplicate detection
         * @return 0 on success, negative error code on validation failure
         */
        int validate_pbch(uint8_t cell_index, const struct oran_packet_header_info &header_info, void *buffer, uint64_t &prev_pbch_time);

        /**
         * Validate received PDSCH (Physical Downlink Shared Channel) U-plane data
         *
         * @param[in] cell_index Cell identifier
         * @param[in] header_info U-plane packet header information
         * @param[in] buffer Pointer to received IQ data
         * @return 0 on success, negative error code on validation failure
         */
        int validate_pdsch(uint8_t cell_index, const struct oran_packet_header_info &header_info, void *buffer);

        /**
         * Validate received PDCCH (Physical Downlink Control Channel) U-plane data
         *
         * @param[in] cell_index Cell identifier
         * @param[in] header_info U-plane packet header information
         * @param[in] buffer Pointer to received IQ data
         * @param[in] channel_type DL channel type (PDCCH_UL or PDCCH_DL)
         * @return 0 on success, negative error code on validation failure
         */
        int validate_pdcch(uint8_t cell_index, const struct oran_packet_header_info &header_info, void *buffer, dl_channel channel_type);

        /**
         * Validate received CSI-RS (Channel State Information Reference Signal) U-plane data
         *
         * @param[in] cell_index Cell identifier
         * @param[in] header_info U-plane packet header information
         * @param[in] buffer Pointer to received IQ data
         * @return 0 on success, negative error code on validation failure
         */
        int validate_csirs(uint8_t cell_index, const struct oran_packet_header_info &header_info, void *buffer);

        /**
         * Check if U-plane channel type matches expected DL channel
         *
         * @param[in] header_info U-plane packet header information
         * @param[in] cell_index Cell identifier
         * @param[in] tv_obj DL test vector object
         * @return true if channel type matches, false otherwise
         */
        void validate_dl_channels(uint8_t cell_index, uint8_t curr_launch_pattern_slot, const struct oran_packet_header_info &header_info, void *section_buffer, uint64_t &prev_pbch_time);
        bool u_plane_channel_type_checking(const struct oran_packet_header_info &header_info, uint16_t cell_index, struct dl_tv_object &tv_obj);

        /** Validates that U-plane sectionId was announced in C-plane for the same slot/eAxC.
         *
         * Looks up the DL SlotSectionIdTracker for the packet's eAxC and verifies
         * that the sectionId carried in the U-plane header was previously recorded
         * by C-plane processing for the current slot. Silently skips the check when
         * no tracker exists for the eAxC or when the tracker's slot does not match
         * the packet's FSS (C-plane may not have arrived yet).
         *
         * @param[in] cell_index Cell identifier used for tracker lookup and logging.
         * @param[in] header_info Parsed U-plane packet header containing sectionId,
         *                        flowValue (eAxC ID), and FSS.
         */
        void validate_uplane_section_id_match(uint8_t cell_index, const struct oran_packet_header_info &header_info);

        /**
         * Verify extension type 11 (beamforming weights) in C-plane section
         *
         * @param[in] section_ptr Pointer to section data
         * @param[in] c_plane_info C-plane packet information
         * @param[in] section_info Section information
         * @param[in] cell_index Cell identifier
         * @return 0 on success, negative error code on validation failure
         */
        int verify_extType11(uint8_t *section_ptr, oran_c_plane_info_t &c_plane_info, oran_c_plane_section_info_t &section_info, int cell_index);

        /**
         * Verify all extensions in a C-plane section
         *
         * @param[in] c_plane_info C-plane packet information
         * @param[in,out] section_info Section information
         * @param[in] cell_index Cell identifier
         * @return 0 on success, negative error code on validation failure
         */
        int verify_extensions(oran_c_plane_info_t &c_plane_info, oran_c_plane_section_info_t &section_info, int cell_index);

        /**
         * Validate dynamic beam IDs in C-plane packet
         *
         * @param[in] c_plane_info C-plane packet information
         * @param[in] cell_index Cell identifier
         */
        void dynamic_beamid_validation(oran_c_plane_info_t &c_plane_info, int cell_index);

        /**
         * Validates C-plane section IDs for default coupling (range and duplicate consistency) using the given DL tracker.
         * @param[in,out] c_plane_info C-plane message info; sections and eAxC index used for lookup.
         * @param[in] cell_index Cell index for tracker and counters.
         * @param[in,out] tracker Slot section ID tracker for this eAxC (DL).
         */
        void validate_section_id_default_coupling(oran_c_plane_info_t& c_plane_info, int cell_index,
                                                  SlotSectionIdTracker& tracker);

        /**
         * Template implementation: validates C-plane section IDs for default coupling against a range-based tracker.
         * @tparam N Number of section IDs in the tracker range.
         * @tparam BASE Base (start) section ID of the tracker range.
         * @param[in,out] c_plane_info C-plane message info; sections and eAxC index used for lookup.
         * @param[in] cell_index Cell index for tracker and counters.
         * @param[in,out] tracker Slot section ID tracker (range [BASE, BASE+N)) for this eAxC.
         */
        template <size_t N, uint16_t BASE>
        void validate_section_id_default_coupling_impl(oran_c_plane_info_t& c_plane_info, int cell_index,
                                                       SlotSectionIdTrackerRange<N, BASE>& tracker);
        /**
         * Validates every C-plane section ID in c_plane_info against the tracker range and duplicate-citation consistency.
         * @tparam N Number of section IDs in the tracker range.
         * @tparam BASE Base (start) section ID of the tracker range.
         * @param[in,out] c_plane_info C-plane message info; section list and eAxC/slot data.
         * @param[in] cell_index Cell index for logging and counters.
         * @param[in,out] tracker Slot section ID tracker (range [BASE, BASE+N)) for this eAxC.
         */
        template <size_t N, uint16_t BASE>
        void validate_section_id_sections_only_impl(oran_c_plane_info_t& c_plane_info, int cell_index,
                                                    SlotSectionIdTrackerRange<N, BASE>& tracker);

        /**
         * Validates DL C-plane section IDs (default coupling): range check and duplicate consistency.
         * @param[in,out] c_plane_info C-plane message info; must have valid eAxC index.
         * @param[in] cell_index Cell index for tracker and counters.
         */
        void sid_dl_validate(oran_c_plane_info_t& c_plane_info, int cell_index);
        /**
         * Validates UL C-plane section IDs per channel (PRACH/SRS/PUSCH-PUCCH) using upstream channel_type and trackers.
         * @param[in,out] c_plane_info C-plane message info; channel_type and eaxcId_index set by find_channel_type/find_eAxC_index.
         * @param[in] cell_index Cell index for tracker and counters.
         * @return true if validation passed or packet has 0 sections; false on invalid eAxC, out-of-range, or section ID mismatch.
         */
        bool sid_real_ul(oran_c_plane_info_t& c_plane_info, int cell_index);

        template <size_t N, uint16_t BASE>
        bool sid_real_ul_one_channel_impl(oran_c_plane_info_t& c_plane_info, int cell_index, uint16_t first_sid,
                                          std::vector<SlotSectionIdTrackerRange<N, BASE>>& vec,
                                          int mtx_type, const char* channel_name);
        bool sid_real_ul_one_channel(oran_c_plane_info_t& c_plane_info, int cell_index, uint16_t first_sid);

        /**
         * Initializes section ID validation state (DL and UL tracker vectors and sizes); called from apply_configs().
         */
        void sectionid_validation_init();

        /**
         * Validate modulation and compression parameters in C-plane section
         *
         * @param[in] c_plane_info C-plane packet information
         * @param[in] section_info Section information
         * @param[in] cell_index Cell identifier
         * @return true if parameters are valid, false otherwise
         */
        bool validate_modulation_compression(const oran_c_plane_info_t& c_plane_info, const oran_c_plane_section_info_t& section_info, int cell_index);

        /**
         * Validate RE (Resource Element) mask in C-plane packet
         *
         * @param[in] c_plane_info C-plane packet information
         * @param[in] cell_index Cell identifier
         * @param[in] is_pdsch_included Pre-computed flag indicating PDSCH PRB overlap with this C-plane message
         */
        void validate_remask(oran_c_plane_info_t& c_plane_info, int cell_index, bool is_pdsch_included);

        /**
         * Validate a single CSI-RS section's beam ID against non-ZP CSI-RS beam ID sets.
         * Uses modulo indexing (ap_idx % beams_array_size) to derive the expected beam ID.
         *
         * @param[in] csirs_beam_id_sets Pointer to resolved CSI-RS beam ID sets (TRS + NZP); nullptr skips validation
         * @param[in,out] c_plane_info C-plane info with section data
         * @param[in] sec_idx Section index to validate
         * @param[in] num_dl_flows Number of DL eAxC flows
         * @param[in] cell_index Cell identifier
         */
        void validate_dl_beamid_csirs(const std::vector<std::vector<uint16_t>>* csirs_beam_id_sets,
                                      oran_c_plane_info_t& c_plane_info, int sec_idx,
                                      size_t num_dl_flows, int cell_index);

        /**
         * Validate a single non-CSI-RS section's beam ID (PDSCH/PBCH/PDCCH).
         * PBCH uses membership check; PDSCH/PDCCH uses positional division mapping.
         *
         * @param[in] non_csirs_beam_ids Pointer to the expected beam ID vector
         * @param[in] non_csirs_is_pbch Whether the matched channel is PBCH/SSB
         * @param[in,out] c_plane_info C-plane info with section data
         * @param[in] sec_idx Section index to validate
         * @param[in] num_dl_flows Number of DL eAxC flows
         * @param[in] cell_index Cell identifier
         */
        void validate_dl_beamid_non_csirs(const std::vector<uint16_t>* non_csirs_beam_ids,
                                          bool non_csirs_is_pbch,
                                          oran_c_plane_info_t& c_plane_info, int sec_idx,
                                          size_t num_dl_flows, int cell_index);

        /**
         * Verify downlink C-plane packet content against test vectors
         *
         * @param[in,out] c_plane_info C-plane packet information
         * @param[in] cell_index Cell identifier
         * @param[in] mbuf_payload Packet payload buffer
         * @param[in,out] msg_info Message receive information
         * @param[in,out] fss_pdsch_prb_seen Array tracking seen PDSCH PRBs per FSS
         */
        void verify_dl_cplane_content(oran_c_plane_info_t& c_plane_info, const int cell_index, uint8_t* mbuf_payload, aerial_fh::MsgReceiveInfo& msg_info, FssPdschPrbSeenArray& fss_pdsch_prb_seen);

        /**
         * Verify uplink C-plane packet content against test vectors
         *
         * @param[in,out] c_plane_info C-plane packet information
         * @param[in] cell_index Cell identifier
         * @param[in] mbuf_payload Packet payload buffer
         * @param[in,out] slot_tx Slot transmission information
         */
        void verify_ul_cplane_content(oran_c_plane_info_t& c_plane_info, const int cell_index, uint8_t* mbuf_payload, slot_tx_info& slot_tx);

        /**
         * Flow information for packet routing
         */
        struct flow_info
        {
            int flowId;       //!< Flow identifier
            int flowValue;    //!< Flow value (eAxC ID)
            int cell_index;   //!< Associated cell index
        };

        /**
         * DL core worker thread information
         */
        struct dl_core_info
        {
            std::array<struct flow_info, MAX_FLOWS_PER_DL_CORE * MAX_CELLS_PER_SLOT> flow_infos;  //!< Array of flow information
            int core_index;      //!< Core index
            float flow_count;    //!< Number of flows assigned to this core
            RU_Emulator* rue;    //!< Pointer to RU_Emulator instance
        };

        /**
         * C-plane core worker thread parameters
         */
        struct cplane_core_param
        {
            int thread_id;              //!< Thread ID in the thread group (SRS or UL/DL C)
            int cpu_id;                 //!< CPU core ID to pin thread to
            int start_cell_index;       //!< Starting cell index for this worker
            int num_cells_per_core;     //!< Number of cells per core (negative if multiple threads per cell)
            bool is_srs;                //!< Whether this is an SRS processing thread
            RU_Emulator* rue;           //!< Pointer to RU_Emulator instance
        };


        /**
         * Get reference to PDSCH test vector object
         *
         * @return Const reference to PDSCH test vectors and validation counters
         */
        [[nodiscard]] const dl_tv_object& get_pdsch_object() const { return pdsch_object; }

        /**
         * Get reference to PBCH test vector object
         *
         * @return Const reference to PBCH test vectors and validation counters
         */
        [[nodiscard]] const dl_tv_object& get_pbch_object() const { return pbch_object; }

        /**
         * Get reference to PDCCH DL test vector object
         *
         * @return Const reference to PDCCH DL test vectors and validation counters
         */
        [[nodiscard]] const dl_tv_object& get_pdcch_dl_object() const { return pdcch_dl_object; }

        /**
         * Get reference to PDCCH UL test vector object
         *
         * @return Const reference to PDCCH UL test vectors and validation counters
         */
        [[nodiscard]] const dl_tv_object& get_pdcch_ul_object() const { return pdcch_ul_object; }

        /**
         * Get reference to CSI-RS test vector object
         *
         * @return Const reference to CSI-RS test vectors and validation counters
         */
        [[nodiscard]] const dl_tv_object& get_csirs_object() const { return csirs_object; }

        /**
         * Get reference to DL beamforming weight test vector object
         *
         * @return Const reference to DL BFW test vectors and validation counters
         */
        [[nodiscard]] const dl_tv_object& get_bfw_dl_object() const { return bfw_dl_object; }

        /**
         * Get reference to UL beamforming weight test vector object
         *
         * @return Const reference to UL BFW test vectors and validation counters
         */
        [[nodiscard]] const dl_tv_object& get_bfw_ul_object() const { return bfw_ul_object; }

        /**
         * Get reference to PUSCH test vector object
         *
         * @return Const reference to PUSCH test vectors and validation counters
         */
        [[nodiscard]] const ul_tv_object& get_pusch_object() const { return pusch_object; }

        /**
         * Get reference to PRACH test vector object
         *
         * @return Const reference to PRACH test vectors and validation counters
         */
        [[nodiscard]] const ul_tv_object& get_prach_object() const { return prach_object; }

        /**
         * Get reference to PUCCH test vector object
         *
         * @return Const reference to PUCCH test vectors and validation counters
         */
        [[nodiscard]] const ul_tv_object& get_pucch_object() const { return pucch_object; }

        /**
         * Get reference to SRS test vector object
         *
         * @return Const reference to SRS test vectors and validation counters
         */
        [[nodiscard]] const ul_tv_object& get_srs_object() const { return srs_object; }

        /**
         * Get total DL section counter for a specific cell
         *
         * @param[in] cell_index Cell identifier
         * @return Total number of DL C-plane sections processed (opt_dlc_tb mode)
         */
        [[nodiscard]] uint64_t get_total_dl_section_count(const int cell_index) const
        {
            return total_dl_section_counters[cell_index].load();
        }

        /**
         * Get error DL section counter for a specific cell
         *
         * @param[in] cell_index Cell identifier
         * @return Number of DL C-plane sections with validation errors (opt_dlc_tb mode)
         */
        [[nodiscard]] uint64_t get_error_dl_section_count(const int cell_index) const
        {
            return error_dl_section_counters[cell_index].load();
        }

        /**
         * Get total UL section counter for a specific cell
         *
         * @param[in] cell_index Cell identifier
         * @return Total number of UL C-plane sections processed (opt_dlc_tb mode)
         */
        [[nodiscard]] uint64_t get_total_ul_section_count(const int cell_index) const
        {
            return total_ul_section_counters[cell_index].load();
        }

        /**
         * Get error UL section counter for a specific cell
         *
         * @param[in] cell_index Cell identifier
         * @return Number of UL C-plane sections with validation errors (opt_dlc_tb mode)
         */
        [[nodiscard]] uint64_t get_error_ul_section_count(const int cell_index) const
        {
            return error_ul_section_counters[cell_index].load();
        }

        const dl_tv_object &get_bfw_dl_obj() { return bfw_dl_object; }

    private:

        /////////////////////////////////////////////////////
        //// CONFIGURATION OPTIONS (from YAML and command line)
        /////////////////////////////////////////////////////

        // Channel enable/disable flags
        int opt_ul_enabled;                              //!< Enable uplink data processing (PUSCH/PUCCH/PRACH/SRS)
        int opt_prach_enabled;                           //!< Enable PRACH channel processing and transmission
        int opt_pucch_enabled;                           //!< Enable PUCCH channel processing and transmission
        int opt_pusch_enabled;                           //!< Enable PUSCH channel processing and transmission
        int opt_srs_enabled;                             //!< Enable SRS channel processing and transmission
        int opt_dl_enabled;                              //!< Enable downlink data processing (PDSCH/PBCH/PDCCH/CSI-RS)
        int opt_dlc_tb;                                  //!< Enable DL C-plane transport block mode (skip IQ validation, buffer is nullptr)
        int opt_mod_comp_enabled;                        //!< Enable modulation compression for IQ data
        int opt_non_mod_comp_enabled;                    //!< Enable BFP/FIX_POINT for IQ data
        
        // Core and thread assignment
        int opt_low_priority_core;                       //!< CPU core ID for low-priority background tasks
        int opt_oam_cell_ctrl_cmd;                       //!< Enable OAM cell control command processing
        
        // C-plane configuration
        int opt_multi_section_ul;                        //!< Enable multi-section uplink C-plane messages
        int opt_c_interval_us;                           //!< C-plane message transmission interval in microseconds
        int opt_c_plane_per_symbol;                      //!< Number of C-plane messages per OFDM symbol
        int opt_prach_c_plane_per_symbol;                //!< Number of PRACH C-plane messages per symbol
        
        // System parameters
        int opt_num_cells;                               //!< Number of cells to emulate
        int opt_tti_us;                                  //!< Transmission Time Interval in microseconds (500 or 1000)
        int opt_num_slots_ul;                            //!< Number of uplink slots to process (for testing)
        int opt_num_slots_dl;                            //!< Number of downlink slots to process (for testing)
        int opt_forever;                                 //!< Run indefinitely, ignore num_slots limit (0=finite, 1=infinite)
        int opt_send_slot;                               //!< Slot index to start transmission
        
        // Timing and synchronization
        int opt_timer_level;                             //!< Performance timing granularity (0=none, 1=slot, 2=symbol)
        int opt_timer_offset_us;                         //!< Timing offset in microseconds for packet transmission
        int opt_symbol_offset_us;                        //!< Symbol timing offset in microseconds
        int opt_validate_dl_timing;                      //!< Enable downlink packet timing validation against ORAN windows
        int opt_dl_warmup_slots;                         //!< Number of warmup slots before starting DL validation
        int opt_ul_warmup_slots;                         //!< Number of warmup slots before starting UL validation
        int opt_timing_histogram;                        //!< Enable timing histogram collection for distribution analysis
        int opt_timing_histogram_bin_size;               //!< Histogram bin size in nanoseconds
        
        // Validation and debugging
        int opt_dl_up_sanity_check;                      //!< Enable DL U-plane packet sanity checking
        int opt_max_sect_stats;                          //!< Maximum number of section statistics to track
        int opt_pdsch_validation;                        //!< Enable PDSCH IQ sample validation against test vectors
        int opt_pbch_validation;                         //!< Enable PBCH IQ sample validation against test vectors
        int opt_pdcch_ul_validation;                     //!< Enable PDCCH UL grant validation against test vectors
        int opt_pdcch_dl_validation;                     //!< Enable PDCCH DL grant validation against test vectors
        int opt_csirs_validation;                        //!< Enable CSI-RS validation against test vectors
        int opt_bfw_dl_validation;                       //!< Enable downlink beamforming weight validation
        int opt_bfw_ul_validation;                       //!< Enable uplink beamforming weight validation
        int opt_beamforming;                             //!< Enable beamforming weight processing
        int opt_beamid_validation;                       //!< Enable 4T4R beam ID validation against test vectors
        int opt_sectionid_validation;                    //!< Enable C-plane/U-plane sectionId cross-validation
        int opt_dl_approx_validation;                    //!< Enable approximate IQ comparison with tolerance (vs exact match)
        int opt_debug_u_plane_threshold;                 //!< U-plane debug print threshold for packet count
        int opt_debug_u_plane_prints;                    //!< Enable detailed U-plane debug prints
        
        // Advanced features
        int opt_enable_mmimo;                            //!< Enable massive MIMO mode (multiple cores per cell)
        int opt_min_ul_cores_per_cell_mmimo;             //!< Minimum number of UL cores per cell for mMIMO (default: 3)
        int opt_enable_beam_forming;                     //!< Enable beamforming weight processing and validation
        int opt_enable_cplane_worker_tracing;            //!< Enable detailed C-plane worker thread tracing
        int opt_drop_packet_every_ten_secs;              //!< Intentionally drop packets every 10 seconds (for robustness testing)
        int opt_enable_dl_proc_mt;                       //!< Enable multi-threaded downlink processing
        int opt_ul_only;                                 //!< Uplink-only mode (no DL processing, TX-only)
        int opt_enable_precomputed_tx = 0;               //!< Pre-compute UL TX messages from launch pattern at init

        // DL processing configuration
        float opt_num_flows_per_dl_thread;               //!< Number of flows (eAxC IDs) per DL processing thread
        
        // File paths
        std::string opt_config_file;                     //!< Path to main YAML configuration file
        std::string opt_launch_pattern_file;             //!< Path to launch pattern YAML file
        int opt_launch_pattern_version;                  //!< Launch pattern file format version
        bool selective_tv_load;                          //!< Load only test vectors referenced in launch pattern (not all)
        
        // Aerial FH driver configuration
        int opt_afh_txq_size;                            //!< Aerial FH TX queue size (number of packets)
        int opt_afh_rxq_size;                            //!< Aerial FH RX queue size (number of packets)
        int opt_afh_txq_request_num;                     //!< Number of TX request objects to allocate
        int opt_afh_dpdk_thread;                         //!< CPU core ID for DPDK polling thread
        int opt_afh_pdump_client_thread;                 //!< CPU core ID for packet dump client thread
        int opt_afh_accu_tx_sched_res_ns;                //!< Accumulated TX scheduler resolution in nanoseconds
        int opt_aerial_fh_per_rxq_mempool;               //!< Use separate memory pool per RX queue (0=shared, 1=per-queue)
        int opt_afh_cpu_mbuf_pool_rx_size;               //!< CPU mbuf pool size for RX (number of mbufs)
        int opt_afh_cpu_mbuf_pool_tx_size;               //!< CPU mbuf pool size for TX (number of mbufs)
        int opt_afh_cpu_mbuf_pool_size_per_rxq;          //!< CPU mbuf pool size per RX queue
        int opt_afh_split_rx_tx_mp;                      //!< Use split RX/TX memory pools (0=shared, 1=split)
        std::string opt_afh_dpdk_file_prefix;            //!< DPDK file prefix for shared memory files
        int opt_afh_mtu;                                 //!< MTU (Maximum Transmission Unit) for network interface
        
        // SRS configuration
        int opt_split_srs_txq;                           //!< Use split TX queues for SRS symbols (0=single queue, 1=split)
        int opt_enable_srs_eaxcid_pacing;                //!< Enable SRS eAxC ID pacing to limit simultaneous transmissions
        int opt_srs_pacing_s3_srs_symbols;               //!< Number of SRS symbols for scenario 3 pacing
        int opt_srs_pacing_s4_srs_symbols;               //!< Number of SRS symbols for scenario 4 pacing
        int opt_srs_pacing_s5_srs_symbols;               //!< Number of SRS symbols for scenario 5 pacing
        int opt_srs_pacing_eaxcids_per_tx_window;        //!< Maximum eAxC IDs per TX window for SRS pacing
        int opt_srs_pacing_eaxcids_per_symbol;           //!< Maximum eAxC IDs per symbol for SRS pacing
        int num_srs_txqs;                                //!< Number of SRS TX queues allocated
        srs_slot_type_info srs_s3_info;                  //!< SRS slot type 3 configuration parameters
        srs_slot_type_info srs_s4_info;                  //!< SRS slot type 4 configuration parameters
        srs_slot_type_info srs_s5_info;                  //!< SRS slot type 5 configuration parameters
        
        // Testing and debug options
        int opt_ecpri_hdr_cfg_test;                      //!< Enable eCPRI header configuration testing mode
        
        // Derived parameters
        int max_slot_id;                                 //!< Maximum slot ID value (1 for 500us TTI, 0 for 1ms TTI)
        int launch_pattern_slot_size;                    //!< Total number of slots in launch pattern
        int dl_cores_per_cell;                           //!< Number of DL processing cores per cell
        int csi_rs_optimized_validation;                 //!< Enable optimized CSI-RS validation algorithm

        /////////////////////////////////////////////////////
        //// LOGGING
        /////////////////////////////////////////////////////
        std::string opt_log_name;                //!< NVLog instance name for logging

        /////////////////////////////////////////////////////
        //// CELL AND NETWORK CONFIGURATION
        /////////////////////////////////////////////////////
        std::vector<struct cell_config> cell_configs;           //!< Per-cell configuration (indexed by eth addr)
        std::vector<std::string> nic_interfaces;                //!< List of NIC interface names (PCIe addresses)
        std::vector<int> ul_core_list;                          //!< CPU core IDs for uplink worker threads
        std::vector<int> ul_srs_core_list;                      //!< CPU core IDs for SRS uplink worker threads
        std::vector<int> dl_core_list;                          //!< CPU core IDs for downlink worker threads
        std::vector<int> dl_rx_core_list;                       //!< CPU core IDs for DL RX worker threads (multi-threaded mode)
        int opt_standalone_core_id;                             //!< CPU core ID for standalone mode

        std::array<cplane_core_param, MAX_RU_THREADS> cplane_core_params;  //!< C-plane worker thread parameters
        aerial_fh::EcpriHdrConfig ecpri_hdr_cfg;                //!< eCPRI header configuration for testing

        /////////////////////////////////////////////////////
        //// FRONTHAUL DRIVER STATE
        /////////////////////////////////////////////////////
        aerial_fh::FronthaulHandle fronthaul;                   //!< Main fronthaul driver handle
        std::vector<aerial_fh::NicHandle> nic_list;             //!< List of registered NIC handles
        std::vector<aerial_fh::NicInfo> nic_info_list;          //!< NIC information structures
        std::unordered_map<aerial_fh::PeerId, aerial_fh::PeerHandle> peer_list;  //!< Map of peer ID to peer handle

        std::unordered_map<int, std::vector<aerial_fh::FlowHandle>> dl_peer_flow_map;  //!< DL flows per peer (cell_index -> flows)
        std::vector<std::vector<aerial_fh::FlowHandle>> ul_peer_flow_map;              //!< UL flows per peer (PUSCH/PUCCH)
        std::vector<std::vector<aerial_fh::FlowHandle>> peer_flow_map_prach;           //!< PRACH flows per peer
        std::vector<std::vector<aerial_fh::FlowHandle>> peer_flow_map_srs;             //!< SRS flows per peer

        /////////////////////////////////////////////////////
        //// TEST VECTOR MANAGEMENT
        /////////////////////////////////////////////////////
        std::unordered_map<dl_channel, std::string> dl_channel_string;   //!< DL channel enum to string mapping
        std::unordered_map<ul_channel, std::string> ul_channel_string;   //!< UL channel enum to string mapping

        std::unordered_map<std::string, std::unordered_map<std::string, std::string>> channel_to_tv_map;  //!< Channel to test vector mapping
        std::unordered_map<std::string, std::unordered_set<std::string>> tv_to_channel_map;               //!< Test vector to channel mapping
        std::string user_defined_tv_base_path;                  //!< User-specified test vector base directory path
        std::string user_defined_lp_base_path;                  //!< User-specified launch pattern base directory path
        bool is_cx6_nic;                                        //!< True if using ConnectX-6 NIC (affects TX scheduling)
        /**
         * Initialize channel enum to string mappings for logging
         */
        void channel_string_setup()
        {
            dl_channel_string[dl_channel::NONE] = "NONE";
            dl_channel_string[dl_channel::PDSCH] = "PDSCH";
            dl_channel_string[dl_channel::PBCH] = "PBCH";
            dl_channel_string[dl_channel::PDCCH_DL] = "PDCCH_DL";
            dl_channel_string[dl_channel::PDCCH_UL] = "PDCCH_UL";
            dl_channel_string[dl_channel::CSI_RS] = "CSI_RS";
            dl_channel_string[dl_channel::BFW_DL] = "BFW_DL";
            dl_channel_string[dl_channel::BFW_UL] = "BFW_UL";
            ul_channel_string[ul_channel::NONE] = "NONE";
            ul_channel_string[ul_channel::PUSCH] = "PUSCH";
            ul_channel_string[ul_channel::PRACH] = "PRACH";
            ul_channel_string[ul_channel::PUCCH] = "PUCCH";
            ul_channel_string[ul_channel::SRS] = "SRS";
        }

        /**
         * Initialize large arrays used for dynamic beamforming validation
         *
         * Allocates and zero-initializes arrays for tracking beam IDs across
         * frames, subframes, slots, and symbols for dynamic beamforming validation
         */
        void initialize_arrays()
        {
            fss_dyn_bfw_beam_id_ptr = std::make_unique<FssDynBfwBeamIdArray>();
            fss_disabled_bfw_dyn_bfw_beam_id_ptr = std::make_unique<FssDisabledBfwDynBfwBeamIdArray>();
            fss_dyn_bfw_beam_id_last_validation_ts_ptr = std::make_unique<FssDynBfwBeamIdLastValidationTsArray>();
            for (auto& cell : *fss_dyn_bfw_beam_id_last_validation_ts_ptr)
              for (auto& ap : cell)
                for (auto& slot : ap)
                  slot.store(0, std::memory_order_relaxed);

            fss_disabled_bfw_dyn_bfw_beam_id_cnt_ptr = std::make_unique<FssDisabledBfwDynBfwBeamIdCntArray>();
            for (auto& cell : *fss_disabled_bfw_dyn_bfw_beam_id_cnt_ptr)
              for (auto& ap : cell)
                for (auto& slot : ap)
                  for (auto& sym : slot)
                     sym.store(0, std::memory_order_relaxed);
            fss_dyn_bfw_beam_id_mtx_ptr = std::make_unique<FssDynBfwBeamIdMtxArray>();

            dl_tv_objs = {&pdsch_object, &pbch_object, &pdcch_ul_object, &pdcch_dl_object, &pdsch_object, &csirs_object};
        }

        /////////////////////////////////////////////////////
        //// ORAN TIMING AND STATISTICS
        /////////////////////////////////////////////////////

        /**
         * Packet type classification for timing statistics
         */
        enum rx_packet_type
        {
            UL_C_PLANE = 0,    //!< Uplink C-plane packets
            DL_C_PLANE,        //!< Downlink C-plane packets
            DL_U_PLANE,        //!< Downlink U-plane packets
            ALL_PACKET_TYPES   //!< Count of all packet types
        };

        /**
         * Section type classification for statistics
         */
        enum section_type
        {
            UL_C_SECTION_TYPE_1 = 0,   //!< UL C-plane section type 1 (data channels)
            UL_C_SECTION_TYPE_3,       //!< UL C-plane section type 3 (PRACH)
            DL_C_SECTION_TYPE_1,       //!< DL C-plane section type 1 (data channels)
            C_SECTION_TYPE_AGGR,       //!< Aggregate section statistics
            ALL_SECTION_TYPES          //!< Count of all section types
        };

        /**
         * Unit for counter aggregation
         */
        enum counter_unit
        {
            PACKET = 0,    //!< Per-packet counters
            SLOT           //!< Per-slot counters
        };

        /**
         * Timing category for packet arrival classification
         */
        enum timing_category
        {
            EARLY = 0,     //!< Packet arrived early (before window)
            ONTIME,        //!< Packet arrived on-time (within window)
            LATE,          //!< Packet arrived late (after window)
            TOTAL,         //!< Total packet count
            ALL_CATEGORIES //!< Count of all timing categories
        };
        /**
         * ORAN timing parameters for packet windows and delays
         */
        struct ru_oran_timing_info
        {
            int dl_c_plane_timing_delay;     //!< DL C-plane timing delay in microseconds (T2a)
            int ul_c_plane_timing_delay;     //!< UL C-plane timing delay in microseconds (T1a)
            int dl_c_plane_window_size;      //!< DL C-plane acceptance window size in microseconds
            int ul_c_plane_window_size;      //!< UL C-plane acceptance window size in microseconds
            int dl_u_plane_timing_delay;     //!< DL U-plane timing delay in microseconds (T2a)
            int dl_u_plane_window_size;      //!< DL U-plane acceptance window size in microseconds
            int ul_u_plane_tx_offset;        //!< UL U-plane TX timing offset in microseconds (Ta3)
            int ul_u_plane_tx_offset_srs;    //!< UL U-plane TX timing offset for SRS in microseconds
        };
        ru_oran_timing_info oran_timing_info;    //!< ORAN timing configuration

        /**
         * Beam ID range configuration for validation
         */
        struct ru_oran_beam_id_info
        {
            int static_beam_id_start;        //!< Start of static beam ID range
            int static_beam_id_end;          //!< End of static beam ID range
            int dynamic_beam_id_start;       //!< Start of dynamic beam ID range
            int dynamic_beam_id_end;         //!< End of dynamic beam ID range
        };
        ru_oran_beam_id_info oran_beam_id_info;  //!< Beam ID range configuration

        /**
         * Packet and slot counters for timing category tracking
         */
        struct ru_packet_counter
        {
            std::atomic<uint64_t> early_packet;    //!< Count of packets that arrived early
            std::atomic<uint64_t> ontime_packet;   //!< Count of packets that arrived on-time
            std::atomic<uint64_t> late_packet;     //!< Count of packets that arrived late
            std::atomic<uint64_t> early_slot;      //!< Count of slots with early packets
            std::atomic<uint64_t> ontime_slot;     //!< Count of slots with all on-time packets
            std::atomic<uint64_t> late_slot;       //!< Count of slots with late packets
            std::atomic<uint64_t> total_slot;      //!< Total slot count
            std::array<std::atomic<uint64_t>, MAX_LAUNCH_PATTERN_SLOTS> early_slots_for_slot_num;   //!< Early slots per launch pattern slot
            std::array<std::atomic<uint64_t>, MAX_LAUNCH_PATTERN_SLOTS> ontime_slots_for_slot_num;  //!< On-time slots per launch pattern slot
            std::array<std::atomic<uint64_t>, MAX_LAUNCH_PATTERN_SLOTS> late_slots_for_slot_num;    //!< Late slots per launch pattern slot
            std::array<std::atomic<uint64_t>, MAX_LAUNCH_PATTERN_SLOTS> total_slots_for_slot_num;   //!< Total slots per launch pattern slot
        };

        /**
         * Packet counters organized by packet type (UL C, DL C, DL U)
         */
        struct ru_oran_packet_counters
        {
            std::array<struct ru_packet_counter, MAX_CELLS_PER_SLOT> dl_c_plane;   //!< DL C-plane counters per cell
            std::array<struct ru_packet_counter, MAX_CELLS_PER_SLOT> dl_u_plane;   //!< DL U-plane counters per cell
            std::array<struct ru_packet_counter, MAX_CELLS_PER_SLOT> ul_c_plane;   //!< UL C-plane counters per cell
        };

        ru_oran_packet_counters oran_packet_counters;   //!< Packet timing counters

        /**
         * Configuration for UL packet drop testing
         *
         * Used for testing robustness to packet loss
         */
        struct ul_pkts_drop_test_cfg
        {
            std::atomic<bool> enabled;                 //!< Enable packet drop testing
            uint16_t drop_rate;                        //!< Drop rate percentage (0~50%)
            uint16_t cnt;                              //!< Packet counter for drop rate calculation
            std::unordered_set<uint16_t> drop_set;     //!< Set of packet indices to drop
            std::atomic<bool> single_drop{0};          //!< Drop single packet flag
            std::atomic<bool> drop_slot{0};            //!< Drop entire slot flag
            std::atomic<bool> drop_slot_ts_set{0};     //!< Drop slot timestamp set flag
            std::atomic<uint64_t> drop_slot_start_ts{0};  //!< Drop slot start timestamp
            std::atomic<uint64_t> drop_slot_end_ts{0};    //!< Drop slot end timestamp
            std::atomic<uint8_t> drop_frame_id{0};     //!< Frame ID to drop
            std::atomic<uint8_t> drop_subframe_id{0};  //!< Subframe ID to drop
            std::atomic<uint8_t> drop_slot_id{0};      //!< Slot ID to drop
        };

        /**
         * Configuration for zero U-plane testing
         *
         * Used for testing TX without actual IQ data
         */
        struct zero_uplane_test_cfg
        {
            std::atomic<bool> enabled{0};              //!< Enable zero U-plane testing
        };

        std::array<std::array<struct ul_pkts_drop_test_cfg, MAX_UL_CHANNELS>, MAX_CELLS_PER_SLOT> ul_pkts_drop_test;  //!< Packet drop test config per cell per channel
        std::array<std::array<struct zero_uplane_test_cfg, MAX_UL_CHANNELS>, MAX_CELLS_PER_SLOT> ul_pkts_zero_uplane_test;  //!< Zero U-plane test config per cell per channel

        FssPdschPrbSeenArray fss_pdsch_prb_seen;    //!< Tracks PDSCH PRBs seen per FSS for validation

        using slot_count_array = std::array<std::array<std::atomic<uint64_t>, MAX_CELLS_PER_SLOT>, ALL_PACKET_TYPES>;
        slot_count_array slot_count;                //!< Slot counters per packet type per cell

        struct packet_slot_timers oran_packet_slot_timers;  //!< Per-slot packet timing accumulators

        /////////////////////////////////////////////////////
        //// DL PACKET RECEIVE BUFFERS
        /////////////////////////////////////////////////////
        std::array<aerial_fh::MsgReceiveInfo, MAX_RUE_DL_CORE_COUNT*PACKET_RX_BUFFER_COUNT*MAX_PACKET_PER_RX_BURST> mbuf_info;  //!< Mbuf info ring buffer
        std::array<std::atomic<uint16_t>, MAX_RUE_DL_CORE_COUNT> pkt_buf_write_idx;     //!< Write index per DL core
        std::array<std::atomic<uint16_t>, MAX_RUE_DL_CORE_COUNT> pkt_buf_read_idx;      //!< Read index per DL core
        std::array<std::atomic<size_t>, MAX_RUE_DL_CORE_COUNT*PACKET_RX_BUFFER_COUNT> num_mbufs_rx;  //!< Number of mbufs received per buffer
        std::array<std::atomic<uint16_t>, MAX_RUE_DL_CORE_COUNT*PACKET_RX_BUFFER_COUNT> pkt_flow_counter;  //!< Packet flow counter per buffer

        /////////////////////////////////////////////////////
        //// PACKET TIMING MANAGEMENT
        /////////////////////////////////////////////////////
        /**
         * Flush packet timing accumulators and update statistics
         *
         * @param[in] dir Direction (UL/DL)
         * @param[in] type Packet type
         * @param[in] cell_index Cell identifier
         * @param[in,out] packet_timer Packet timer structure to flush
         */
        void flush_packet_timers(uint8_t dir, uint8_t type, uint8_t cell_index, struct packet_timer_per_slot& packet_timer);

        /**
         * Increment ORAN packet timing counters based on packet arrival time
         *
         * @param[in] type Packet type
         * @param[in] cell_index Cell identifier
         * @param[in] packet_timer Packet timer with timing information
         * @param[in] curr_launch_pattern_slot Current launch pattern slot
         */
        void increment_oran_packet_counters(uint8_t type, uint8_t cell_index, struct packet_timer_per_slot& packet_timer, uint8_t curr_launch_pattern_slot);

        // Packet statistics objects for each channel/direction
        Packet_Statistics ul_c_packet_stats;         //!< UL C-plane packet statistics
        Packet_Statistics dl_c_packet_stats;         //!< DL C-plane packet statistics
        Packet_Statistics dl_u_packet_stats;         //!< DL U-plane packet statistics
        Packet_Statistics ul_u_prach_packet_stats;   //!< UL PRACH packet statistics
        Packet_Statistics ul_u_pucch_packet_stats;   //!< UL PUCCH packet statistics
        Packet_Statistics ul_u_pusch_packet_stats;   //!< UL PUSCH packet statistics
        Packet_Statistics ul_u_srs_packet_stats;     //!< UL SRS packet statistics

        /////////////////////////////////////////////////////
        //// UPLINK TEST VECTOR OBJECTS
        /////////////////////////////////////////////////////
        ul_tv_object pusch_object;                   //!< PUSCH test vectors and counters
        ul_tv_object prach_object;                   //!< PRACH test vectors and counters
        ul_tv_object pucch_object;                   //!< PUCCH test vectors and counters
        ul_tv_object srs_object;                     //!< SRS test vectors and counters
        void * zero_prbs;                            //!< Buffer of zeros for filling unused PRBs
        bool enable_srs;                             //!< SRS enabled flag

        /////////////////////////////////////////////////////
        //// PRE-COMPUTED UL TX CACHE
        /////////////////////////////////////////////////////
        // precomputed_tx_cache[launch_pattern_slot][cell_index] -> list of eAxC TX entries
        std::vector<std::vector<PrecomputedSlotCellTx>> precomputed_tx_cache;

        /////////////////////////////////////////////////////
        //// DOWNLINK TEST VECTOR OBJECTS
        /////////////////////////////////////////////////////
        dl_tv_object pdsch_object;                   //!< PDSCH test vectors and validation counters
        dl_tv_object pbch_object;                    //!< PBCH test vectors and validation counters
        dl_tv_object pdcch_dl_object;                //!< PDCCH DL test vectors and validation counters
        dl_tv_object pdcch_ul_object;                //!< PDCCH UL test vectors and validation counters
        dl_tv_object csirs_object;                   //!< CSI-RS test vectors and validation counters
        dl_tv_object bfw_dl_object;                  //!< DL beamforming weight test vectors and counters
        dl_tv_object bfw_ul_object;                  //!< UL beamforming weight test vectors and counters

        std::array<dl_core_info, MAX_CELLS_PER_SLOT*MAX_FLOWS_PER_DL_CORE>  dl_core_info;  //!< DL worker thread parameters

        std::vector<dl_tv_object *> dl_tv_objs;      //!< DL TV objects vector

        /////////////////////////////////////////////////////
        //// NETWORK AND PEER CONFIGURATION
        /////////////////////////////////////////////////////
        struct dpdk_info dpdk;                       //!< DPDK and peer configuration

#ifdef STANDALONE
        std::vector<aerial_fh::RingBufferHandle> standalone_c_plane_rings;  //!< Ring buffers for standalone mode C-plane
#endif

        /////////////////////////////////////////////////////
        //// UPLINK THROUGHPUT TRACKING
        /////////////////////////////////////////////////////
        std::array<std::atomic<uint64_t>, MAX_CELLS_PER_SLOT> ul_throughput_counters;        //!< UL throughput byte counters per cell
        std::array<std::atomic<uint32_t>, MAX_CELLS_PER_SLOT> ul_slot_counters;              //!< UL total slot counters per cell
        std::array<std::atomic<uint16_t>, MAX_CELLS_PER_SLOT> ul_throughput_slot_counters;   //!< UL throughput slot counters per cell

        /////////////////////////////////////////////////////
        //// SECTION AND BEAMFORMING TRACKING
        /////////////////////////////////////////////////////
        std::array<std::array<std::array<std::array<uint32_t, ORAN_MAX_SLOT_X_SUBFRAME_ID>, ORAN_MAX_FRAME_ID>, MAX_CELLS_PER_SLOT>, 4> fss_received_sections;  //!< Received section count per FSS per cell per packet type
        std::array<std::array<std::array<std::array<uint64_t, ORAN_MAX_SLOT_X_SUBFRAME_ID>, ORAN_MAX_FRAME_ID>, MAX_CELLS_PER_SLOT>, 4> fss_received_sections_prev_ts;  //!< Previous timestamp per FSS per cell per packet type
        std::array<std::array<uint32_t, MAX_CELLS_PER_SLOT>, 4> max_sections_per_slot;  //!< Maximum sections seen per slot per cell per packet type
        std::array<std::array<aerial_fh::FHMutex, MAX_CELLS_PER_SLOT>, 4> cell_mtx;     //!< Mutexes for cell-level synchronization

        // Dynamic beamforming validation arrays (heap-allocated due to size)
        std::unique_ptr<FssDynBfwBeamIdArray> fss_dyn_bfw_beam_id_ptr;                           //!< Dynamic beam IDs per FSS
        std::unique_ptr<FssDisabledBfwDynBfwBeamIdArray> fss_disabled_bfw_dyn_bfw_beam_id_ptr;   //!< Disabled BFW dynamic beam IDs
        std::unique_ptr<FssDisabledBfwDynBfwBeamIdCntArray> fss_disabled_bfw_dyn_bfw_beam_id_cnt_ptr;  //!< Disabled BFW beam ID counts
        std::unique_ptr<FssDynBfwBeamIdLastValidationTsArray> fss_dyn_bfw_beam_id_last_validation_ts_ptr;  //!< Last validation timestamp per beam ID
        std::unique_ptr<FssDynBfwBeamIdMtxArray> fss_dyn_bfw_beam_id_mtx_ptr;                    //!< Mutexes for beam ID validation

        /////////////////////////////////////////////////////
        //// GLOBAL COUNTERS AND STATISTICS
        /////////////////////////////////////////////////////
        uint64_t global_slot_counter;                                      //!< Global slot counter across all cells
        uint32_t slots_per_second;                                         //!< Slots per second (based on TTI)

        std::array<std::atomic<uint64_t>, STATS_MAX_BINS> timing_bins;    //!< Timing histogram bins for distribution analysis

        /////////////////////////////////////////////////////
        //// DLC TESTBENCH SECTION VERIFICATION COUNTERS
        /////////////////////////////////////////////////////
        std::array<std::atomic<uint64_t>, MAX_CELLS_PER_SLOT> total_dl_section_counters;   //!< Total DL C-plane sections processed (opt_dlc_tb mode)
        std::array<std::atomic<uint64_t>, MAX_CELLS_PER_SLOT> error_dl_section_counters;   //!< DL C-plane sections with validation errors (opt_dlc_tb mode)
        std::array<std::atomic<uint64_t>, MAX_CELLS_PER_SLOT> total_ul_section_counters;   //!< Total UL C-plane sections processed (opt_dlc_tb mode)
        std::array<std::atomic<uint64_t>, MAX_CELLS_PER_SLOT> error_ul_section_counters;   //!< UL C-plane sections with validation errors (opt_dlc_tb mode)

        /** UL uses separate tracker vectors per channel so section ID ranges/sizes match DU-side
         *  (cuphydriver / FH) channel-specific start indices and limited section counts. */
        struct SectionIdTrackerStorage
        {
            std::array<std::vector<SlotSectionIdTracker>, MAX_CELLS_PER_SLOT> dl;
            std::array<std::vector<SlotSectionIdTrackerPuschPucch>, MAX_CELLS_PER_SLOT> ul_pusch_pucch;
            std::array<std::vector<SlotSectionIdTrackerPrach>, MAX_CELLS_PER_SLOT> ul_prach;
            std::array<std::vector<SlotSectionIdTrackerSrs>, MAX_CELLS_PER_SLOT> ul_srs;
            std::array<std::atomic<uint64_t>, MAX_CELLS_PER_SLOT> error_dl_uplane{};
        };
        std::unique_ptr<SectionIdTrackerStorage> section_id_trackers;

        // 4T4R beam ID validation counters
        std::array<std::atomic<uint64_t>, MAX_CELLS_PER_SLOT> beamid_dl_error_counters{};  //!< DL beam ID mismatch errors per cell
        std::array<std::atomic<uint64_t>, MAX_CELLS_PER_SLOT> beamid_ul_error_counters{};  //!< UL beam ID mismatch errors per cell
        std::array<std::atomic<uint64_t>, MAX_CELLS_PER_SLOT> beamid_dl_total_counters{};  //!< Total DL beam IDs checked per cell
        std::array<std::atomic<uint64_t>, MAX_CELLS_PER_SLOT> beamid_ul_total_counters{};  //!< Total UL beam IDs checked per cell

        /////////////////////////////////////////////////////
        //// eCPRI SEQUENCE ID TRACKING
        /////////////////////////////////////////////////////
        std::vector<std::map<uint16_t, uint8_t>> ecpriSeqid_vectormap;    //!< eCPRI sequence ID per PC ID per cell (key: ecpriPcid, value: ecpriSeqid)

        /**
         * Get and increment eCPRI sequence ID for a given PC ID
         *
         * @param[in] ecpriPcid eCPRI PC ID
         * @param[in] cell_index Cell identifier
         * @return Next eCPRI sequence ID
         */
        uint8_t next_ecpriSeqid(uint16_t ecpriPcid, int cell_index) {
            auto it = ecpriSeqid_vectormap[cell_index].find(ecpriPcid);
            if (it == ecpriSeqid_vectormap[cell_index].end())
                ecpriSeqid_vectormap[cell_index][ecpriPcid] = 0;
            return ecpriSeqid_vectormap[cell_index][ecpriPcid]++;
        }

        aerial_fh::FHMutex cplane_pcap_mutex;                              //!< Mutex for C-plane PCAP capture synchronization
};

#endif
