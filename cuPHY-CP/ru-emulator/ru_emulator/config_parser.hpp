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

#ifndef CONFIG_PARSER_H__
#define CONFIG_PARSER_H__
#include <cstring>
#include <libgen.h>
#include "ru_emulator.hpp"
#include "hdf5hpp.hpp"

// RU Emulator YAML configuration keys
#define YAML_RU_EMULATOR        "ru_emulator"                    //!< Root key for RU emulator configuration section
#define YAML_NIC_INTERFACE      "nic_interface"                  //!< Network interface card name (e.g., "enp1s0f0")
#define YAML_UL_CORE_LIST       "ul_core_list"                   //!< CPU core list for uplink processing threads
#define YAML_UL_SRS_CORE_LIST   "ul_srs_core_list"              //!< CPU core list for SRS uplink processing threads
#define YAML_DL_CORE_LIST       "dl_core_list"                   //!< CPU core list for downlink processing threads
#define YAML_DL_RX_CORE_LIST    "dl_rx_core_list"                //!< CPU core list for downlink RX worker threads
#define YAML_STANDALONE_CORE_ID "standalone_core_id"             //!< CPU core ID for standalone mode operation
#define YAML_NVLOG_NAME         "nvlog_name"                     //!< NVLOG logger name for this RU emulator instance
#define YAML_PEER_ETH_ADDR      "peerethaddr"                    //!< Peer Ethernet MAC address for ORAN fronthaul
#define YAML_VLAN               "vlan"                           //!< VLAN ID for ORAN traffic tagging
#define YAML_PCP                "pcp"                            //!< Priority Code Point for VLAN QoS
#define YAML_TV_UPLINK          "tv_uplink"                      //!< Uplink test vector directory path
#define YAML_TV_DOWNLINK        "tv_downlink"                    //!< Downlink test vector directory path
#define YAML_NUM_SLOTS          "num_slots"                      //!< Total number of slots to process (for testing)
#define YAML_NUM_SLOTS_UL       "num_slots_ul"                   //!< Number of uplink slots to process
#define YAML_NUM_SLOTS_DL       "num_slots_dl"                   //!< Number of downlink slots to process
#define YAML_NUM_CELLS          "num_cells"                      //!< Number of cells to emulate
#define YAML_TTI                "tti"                            //!< Transmission Time Interval in microseconds
#define YAML_DL_UP_SANITY_CHECK "dl_up_sanity_check"             //!< Enable DL U-plane packet sanity checking
#define YAML_MAX_SECT_STATS     "max_sect_stats"                 //!< Maximum number of section statistics to track
#define YAML_DL_BFW_VALIDATION  "dl_bfw_validation"              //!< Enable downlink beamforming weight validation
#define YAML_UL_BFW_VALIDATION  "ul_bfw_validation"              //!< Enable uplink beamforming weight validation
#define YAML_BEAMID_VALIDATION  "beamid_validation"              //!< Enable 4T4R beam ID validation against test vectors
#define YAML_SECTIONID_VALIDATION "sectionid_validation"          //!< Enable C-plane/U-plane sectionId cross-validation
#define YAML_C_INTERVAL         "c_interval"                     //!< C-plane message transmission interval
#define YAML_C_PLANE_PER_SYMBOL "c_plane_per_symbol"             //!< Number of C-plane messages per OFDM symbol
#define YAML_PRACH_C_PLANE_PER_SYMBOL "prach_c_plane_per_symbol" //!< Number of PRACH C-plane messages per symbol
#define YAML_FLOW_ID_METHOD     "flow_ident_method"              //!< Method for identifying flows (eAxC ID mapping)
#define YAML_TIMER_LEVEL        "timer_level"                    //!< Granularity of performance timing (0=none, 1=slot, 2=symbol)
#define YAML_TIMER_OFFSET_US    "timer_offset_us"                //!< Timing offset in microseconds for packet transmission
#define YAML_SYMBOL_OFFSET_US   "symbol_offset_us"               //!< Symbol timing offset in microseconds
#define YAML_CELL_CONFIGS       "cell_configs"                   //!< Array of per-cell configuration objects
#define YAML_CELL_NAME          "name"                           //!< Cell identifier name (e.g., "cell_0")
#define YAML_CELL_ETH           "eth"                            //!< Cell-specific Ethernet MAC address
#define YAML_CELL_EAXC_PRACH_LIST "eAxC_prach_list"              //!< List of PRACH eAxC IDs for this cell
#define YAML_CELL_EAXC_SRS_LIST "eAxC_srs_list"                  //!< List of SRS eAxC IDs for this cell
#define YAML_CELL_NUMANTS       "numAnts"                        //!< Number of antenna ports for this cell
#define YAML_CELL_RU_TYPE       "ru_type"                        //!< Radio unit type identifier
#define YAML_CELL_DL_IQ_DATA_FMT "dl_iq_data_fmt"                //!< Downlink IQ data format configuration object
#define YAML_CELL_UL_IQ_DATA_FMT "ul_iq_data_fmt"                //!< Uplink IQ data format configuration object
#define YAML_CELL_COMP_METH "comp_meth"                          //!< IQ compression method (1=BFP, 4=modulation compression)
#define YAML_CELL_BIT_WIDTH "bit_width"                          //!< IQ sample bit width (e.g., 9, 14, 16)
#define YAML_CELL_FS_OFFSET_DL      "fs_offset_dl"               //!< Frequency shift offset for downlink
#define YAML_CELL_EXPONENT_DL       "exponent_dl"                //!< Block floating point exponent for downlink
#define YAML_CELL_REF_DL            "ref_dl"                     //!< Reference value for downlink compression
#define YAML_STATIC_BEAM_ID_START "static_beam_id_start"         //!< Starting beam ID for static beamforming
#define YAML_STATIC_BEAM_ID_END "static_beam_id_end"             //!< Ending beam ID for static beamforming
#define YAML_DYNAMIC_BEAM_ID_START "dynamic_beam_id_start"       //!< Starting beam ID for dynamic beamforming
#define YAML_DYNAMIC_BEAM_ID_END "dynamic_beam_id_end"           //!< Ending beam ID for dynamic beamforming
#define YAML_LOW_PRIORITY_CORE          "low_priority_core"      //!< CPU core for low-priority background tasks
#define YAML_OAM_CELL_CTRL_CMD          "oam_cell_ctrl_cmd"      //!< Enable OAM cell control command processing
#define YAML_FIX_BETA_DL       "fix_beta_dl"                     //!< Override beta_dl from test vectors with fixed value
#define YAML_SEND_SLOT          "send_slot"                      //!< Slot index to start transmission
#define YAML_LAUNCH_PATTERN     "launch_pattern"                 //!< Launch pattern configuration file path
#define YAML_UL_ENABLED         "ul_enabled"                     //!< Enable uplink data processing
#define YAML_DL_ENABLED         "dl_enabled"                     //!< Enable downlink data processing
#define YAML_DLC_TB             "dlc_tb"                         //!< Enable DL C-plane transport block mode (skip IQ validation)
#define YAML_FOREVER            "forever"                        //!< Run indefinitely (ignore num_slots limit)
#define YAML_VALIDATE_TIMING    "validate_dl_timing"             //!< Enable downlink packet timing validation
#define YAML_DL_WARMUP_SLOTS    "dl_warmup_slots"                //!< Number of warmup slots before starting DL validation
#define YAML_UL_WARMUP_SLOTS    "ul_warmup_slots"                //!< Number of warmup slots before starting UL validation
#define YAML_TIMING_HISTOGRAM   "timing_histogram"               //!< Enable timing histogram collection
#define YAML_TIMING_HISTOGRAM_BIN_SIZE "timing_histogram_bin_size" //!< Histogram bin size in nanoseconds
#define YAML_DL_COMPRESS_BITS   "dl_compress_bits"               //!< Downlink compression bit width
#define YAML_UL_COMPRESS_BITS   "ul_compress_bits"               //!< Uplink compression bit width
#define YAML_PBCH_PARAMS        "pbch_params"                    //!< PBCH channel optional parameters (startSym, numSym, startPrb, numPrb)
#define YAML_PRACH_PARAMS       "prach_params"                   //!< PRACH channel optional parameters
#define YAML_PDCCH_UL_PARAMS       "pdcch_ul_params"             //!< PDCCH UL grant optional parameters
#define YAML_PDCCH_DL_PARAMS       "pdcch_dl_params"             //!< PDCCH DL grant optional parameters
#define YAML_STARTSYM      "startSym"                            //!< Starting OFDM symbol index (for channel params)
#define YAML_NUMSYM        "numSym"                              //!< Number of OFDM symbols (for channel params)
#define YAML_STARTPRB      "startPrb"                            //!< Starting PRB index (for channel params)
#define YAML_NUMPRB        "numPrb"                              //!< Number of PRBs (for channel params)
#define YAML_PRACH_ENABLED "prach_enabled"                       //!< Enable PRACH channel processing
#define YAML_SRS_ENABLED "srs_enabled"                           //!< Enable SRS channel processing
#define YAML_PDSCH_VALIDATION "pdsch_validation"                 //!< Enable PDSCH IQ sample validation
#define YAML_PBCH_VALIDATION "pbch_validation"                   //!< Enable PBCH IQ sample validation
#define YAML_PDCCH_UL_VALIDATION "pdcch_ul_validation"           //!< Enable PDCCH UL grant validation
#define YAML_PDCCH_DL_VALIDATION "pdcch_dl_validation"           //!< Enable PDCCH DL grant validation
#define YAML_DL_APPROX_VALIDATION "dl_approx_validation"         //!< Enable approximate IQ comparison for DL (with tolerance)
#define YAML_ENABLE_MMIMO "enable_mmimo"                         //!< Enable massive MIMO mode (multi-core per cell)
#define YAML_MIN_UL_CORES_PER_CELL_MMIMO "min_ul_cores_per_cell_mmimo" //!< Minimum number of UL cores per cell for mMIMO
#define YAML_ENABLE_BEAM_FORMING "enable_beam_forming"           //!< Enable beamforming weight processing
#define YAML_ENABLE_CPLANE_WORKER_TRACING "enable_cplane_worker_tracing" //!< Enable detailed C-plane worker thread tracing
#define YAML_DROP_PACKET_EVERY_TEN_SECS "drop_packet_every_ten_secs" //!< Intentionally drop packets every 10 seconds (for testing)
#define YAML_MULTI_SECTION_UL "multi_section_ul"                 //!< Enable multi-section uplink C-plane messages
#define YAML_ENABLE_DL_PROC_MT "enable_dl_proc_mt"               //!< Enable multi-threaded downlink processing
#define YAML_SPLIT_SRS_TXQ "split_srs_txq"                       //!< Use split TX queues for SRS symbols
#define YAML_UL_ONLY "ul_only"                                   //!< Uplink-only mode (no DL processing)
#define YAML_ENABLE_PRECOMPUTED_TX "enable_precomputed_tx"       //!< Pre-compute UL TX messages from launch pattern
#define YAML_ENABLE_SRS_EAXCID_PACING  "enable_srs_eaxcid_pacing" //!< Enable SRS eAxC ID pacing to limit simultaneous transmissions
#define YAML_SRS_PACING_S3_SRS_SYMBOLS "srs_pacing_s3_srs_symbols" //!< Number of SRS symbols for scenario 3 pacing
#define YAML_SRS_PACING_S4_SRS_SYMBOLS "srs_pacing_s4_srs_symbols" //!< Number of SRS symbols for scenario 4 pacing
#define YAML_SRS_PACING_S5_SRS_SYMBOLS "srs_pacing_s5_srs_symbols" //!< Number of SRS symbols for scenario 5 pacing
#define YAML_SRS_PACING_EAXCIDS_PER_TX_WINDOW "srs_pacing_eaxcids_per_tx_window" //!< Maximum eAxC IDs per TX window for SRS pacing
#define YAML_SRS_PACING_EAXCIDS_PER_SYMBOL "srs_pacing_eaxcids_per_symbol" //!< Maximum eAxC IDs per symbol for SRS pacing

// Aerial FH (Fronthaul) driver configuration keys
#define YAML_AFH_TXQ_SIZE                       "aerial_fh_txq_size"                      //!< TX queue size
#define YAML_AFH_RXQ_SIZE                       "aerial_fh_rxq_size"                      //!< RX queue size
#define YAML_AFH_TX_REQUEST_NUM                 "aerial_fh_tx_request_num"                //!< Number of TX requests
#define YAML_AFH_DPDK_THREAD                    "aerial_fh_dpdk_thread"                   //!< DPDK thread CPU core
#define YAML_AFH_PDUMP_CLIENT_THREAD            "aerial_fh_pdump_client_thread"           //!< Packet dump client thread
#define YAML_AFH_ACCU_TX_SCHED_RES_NS           "aerial_fh_accu_tx_sched_res_ns"          //!< TX scheduler resolution in nanoseconds
#define YAML_AFH_DPDK_FILE_PREFIX               "aerial_fh_dpdk_file_prefix"              //!< DPDK file prefix
#define YAML_AFH_PER_RXQ_MEMPOOL                "aerial_fh_per_rxq_mempool"               //!< Per-RXQ memory pool
#define YAML_AFH_CPU_MBUF_POOL_SIZE_PER_RXQ     "aerial_fh_cpu_mbuf_pool_size_per_rxq"    //!< CPU mbuf pool size per RXQ
#define YAML_AFH_CPU_MBUF_POOL_TX_SIZE          "aerial_fh_cpu_mbuf_pool_tx_size"         //!< CPU mbuf pool TX size
#define YAML_AFH_CPU_MBUF_POOL_RX_SIZE          "aerial_fh_cpu_mbuf_pool_rx_size"         //!< CPU mbuf pool RX size
#define YAML_AFH_SPLIT_MP                       "aerial_fh_split_rx_tx_mempool"           //!< Split RX/TX memory pools
#define YAML_AFH_MTU                            "aerial_fh_mtu"                           //!< MTU (Maximum Transmission Unit)

// Launch pattern YAML configuration keys
#define YAML_LP_TV              "TV"                     //!< Test vector array
#define YAML_LP_NAME            "name"                   //!< Test vector name
#define YAML_LP_PATH            "path"                   //!< Test vector file path
#define YAML_LP_INIT            "INIT"                   //!< Initialization section
#define YAML_LP_SCHED           "SCHED"                  //!< Schedule section
#define YAML_LP_PDSCH           "PDSCH"                  //!< PDSCH channel key
#define YAML_LP_PUSCH           "PUSCH"                  //!< PUSCH channel key
#define YAML_LP_PRACH           "PRACH"                  //!< PRACH channel key
#define YAML_LP_PBCH            "PBCH"                   //!< PBCH channel key
#define YAML_LP_PDCCH_DL        "PDCCH_DL"               //!< PDCCH DL channel key
#define YAML_LP_PDCCH_UL        "PDCCH_UL"               //!< PDCCH UL channel key
#define YAML_LP_PUCCH           "PUCCH"                  //!< PUCCH channel key
#define YAML_LP_SRS             "SRS"                    //!< SRS channel key
#define YAML_LP_CSIRS           "CSI_RS"                 //!< CSI-RS channel key
#define YAML_LP_BFW_DL          "BFW_DL"                 //!< DL beamforming weights key
#define YAML_LP_BFW_UL          "BFW_UL"                 //!< UL beamforming weights key
#define YAML_LP_SLOT            "slot"                   //!< Slot index
#define YAML_LP_CONFIG          "config"                 //!< Configuration key
#define YAML_LP_CELL_INDEX      "cell_index"             //!< Cell index
#define YAML_LP_CHANNELS        "channels"               //!< Channels array
#define YAML_LP_CHANNEL_TYPE    "type"                   //!< Channel type
#define YAML_LP_CHANNEL_TV      "tv"                     //!< Test vector reference
#define YAML_LP_CHANNEL_BEAM_IDS    "beam_ids"           //!< Beam ID list
#define YAML_LP_NUM_CELLS       "Num_Cells"              //!< Number of cells
#define YAML_LP_CELL_CONFIGS       "Cell_Configs"        //!< Cell configurations array

// Deprecated launch pattern keys
#define YAML_LP_CELL_TV         "tv"                     //!< (Deprecated) Cell test vector
#define YAML_LP_DOWNLINK        "DOWNLINK"               //!< (Deprecated) Downlink section
#define YAML_LP_UPLINK          "UPLINK"                 //!< (Deprecated) Uplink section

#define MAX_PATH_LEN 1024   //!< Maximum path length for file paths

// Configuration file path macros
#ifdef SUBMODULE_BUILD
    #define CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM 2        //!< Relative directory depth for CUBB_HOME (submodule build)
    #define CONFIG_RE_YAML_FILE_PATH "config/"         //!< RU emulator config file path (submodule build)
    #define CONFIG_LAUNCH_PATTERN_PATH "config/"       //!< Launch pattern path (submodule build)
    #define CONFIG_TEST_VECTOR_PATH "testMAC_tvs/"     //!< Test vector path (submodule build)
#else
    #define CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM 4        //!< Relative directory depth for CUBB_HOME (standalone build)
    #define CONFIG_RE_YAML_FILE_PATH "cuPHY-CP/ru-emulator/config/"   //!< RU emulator config file path (standalone)
    #define CONFIG_LAUNCH_PATTERN_PATH "testVectors/multi-cell/"      //!< Launch pattern path (standalone)
    #define CONFIG_TEST_VECTOR_PATH "testVectors/"     //!< Test vector path (standalone)
#endif

/**
 * Parse and assign cell configurations from YAML
 *
 * @param[in] root YAML root node
 * @param[out] cell_configs Vector to populate with cell configurations
 * @param[out] num_cells Number of cells parsed
 * @param[in] lp_yaml_file Launch pattern YAML file path
 */
void yaml_assign_cell_configs(yaml::node root, std::vector<struct cell_config>& cell_configs, int& num_cells, std::string lp_yaml_file);

/**
 * Parse and assign CPU core list from YAML
 *
 * @param[in] root YAML root node
 * @param[out] core_list Vector to populate with core IDs
 * @param[in] UL Whether this is for uplink cores
 * @param[in] DL_proc Whether this is for DL processing cores
 */
void yaml_assign_core_list(yaml::node root, std::vector<int>& core_list, bool UL,bool DL_proc);

/**
 * Assign core list from YAML using a custom key
 *
 * @param[in] root YAML root node
 * @param[out] core_list Vector to populate with core IDs
 * @param[in] key YAML key to look up
 */
void yaml_assign_core_list(yaml::node root, std::vector<int>& core_list, const char* key);

/**
 * Assign test vector paths from YAML (legacy method from config.yaml)
 *
 * @param[in] root YAML root node
 * @param[in] key YAML key to look up
 * @param[out] tvs Vector to populate with test vector paths
 */
void yaml_assign_tv(yaml::node root, std::string key, std::vector<std::string>& tvs);

/**
 * Parse Ethernet address string and assign to ORAN address structure
 *
 * @param[in] eth Ethernet address string (format: "XX:XX:XX:XX:XX:XX")
 * @param[out] addr ORAN Ethernet address structure to populate
 */
void yaml_assign_eth(std::string eth, struct oran_ether_addr& addr);

/**
 * Try to assign float value from YAML node, with default if not found
 *
 * @param[in] parent Parent YAML node
 * @param[in] key Key to look up
 * @param[out] dest Destination float variable
 */
void try_yaml_assign_float(yaml::node& parent, std::string key, float& dest);

/**
 * Try to assign integer value from YAML node, with default if not found
 *
 * @param[in] parent Parent YAML node
 * @param[in] key Key to look up
 * @param[out] dest Destination integer variable
 */
void try_yaml_assign_int(yaml::node& parent, std::string key, int& dest);

/**
 * Try to assign string value from YAML node, with default if not found
 *
 * @param[in] parent Parent YAML node
 * @param[in] key Key to look up
 * @param[out] dest Destination string variable
 */
void try_yaml_assign_string(yaml::node& parent, std::string key, std::string& dest);

#endif
