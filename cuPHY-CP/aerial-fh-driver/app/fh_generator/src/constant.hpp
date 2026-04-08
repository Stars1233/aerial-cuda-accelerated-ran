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

#ifndef FH_GENERATOR_DEFAULTS_HPP__
#define FH_GENERATOR_DEFAULTS_HPP__

#include "aerial-fh-driver/api.hpp"
namespace fh_gen
{
constexpr aerial_fh::Ns kNsPerSec             = 1000 * 1000 * 1000;
constexpr int           kCpuIqBufferMagicChar = 0x01;
constexpr size_t        kMinIqBufferSize      = 32;
constexpr size_t        kMaxLogLength = 1024;
constexpr int      kGpuIqBufferMagicChar = 0x20;
constexpr uint16_t kRxqSize              = 1 << 6;
constexpr uint8_t  kNvGpuPageShift       = 16;
constexpr uint32_t kNvGpuPageSize        = (1UL << kNvGpuPageShift);
constexpr uint16_t kMaxSectionCount      = 1024;
constexpr uint16_t kMaxMsgSendInfoCount  = 4096;
constexpr uint32_t kMaxCells = API_MAX_NUM_CELLS; //max number of cells

constexpr uint32_t kMaxPrbsPerSymbol = 273;
constexpr uint32_t kMaxSymbols = 14;
constexpr uint32_t kMaxSlots = 160;
constexpr uint32_t kMaxAntennas = 16;
constexpr uint32_t kMaxPrbSize = 48;
constexpr uint32_t kMaxSlotCount = ORAN_MAX_SLOT_X_SUBFRAME_ID * 4;
constexpr uint32_t kMaxSectionNum = 273;
constexpr uint8_t  kMaxNicsSupported = 2;
constexpr uint8_t  kOrderEntityNum = 16;

static constexpr uint32_t ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM = 50; //50 is a safe number to assume?
static constexpr uint32_t MAX_UPLANE_MSGS_PER_SLOT = (kMaxSectionNum * kMaxAntennas * ORAN_ALL_SYMBOLS);


// Config file YAML keys
constexpr char kNicsYaml[]          = "nics";
constexpr char kRuNicsYaml[]        = "ru_nics";
constexpr char kIqDataBuffersYaml[] = "iq_data_buffers";
constexpr char kPeersYaml[]         = "cells";
constexpr char kFlowsYaml[]         = "flows";
constexpr char kCPlaneTxYaml[]      = "cplane_tx";
constexpr char kUPlaneTxYaml[]      = "uplane_tx";
constexpr char kUPlaneRxYaml[]      = "uplane_rx";
constexpr char kDuCpusYaml[]        = "du_cpus";
constexpr char kRuCpusYaml[]        = "ru_cpus";
constexpr char kSfnSlotSyncYaml[]   = "sfn_slot_sync";

constexpr char kFhDpdkThreadYaml[]              = "dpdk_thread";
constexpr char kFhDpdkVerboseLogsYaml[]         = "dpdk_verbose_logs";
constexpr char kFhAccuTxSchedResNsYaml[]        = "accu_tx_sched_res_ns";
constexpr char kFhStartupDelayYaml[]            = "startup_delay_sec";
constexpr char kWorkerThreadSchedFifoPrioYaml[] = "worker_thread_sched_fifo_prio";
constexpr char kFhSendUtcAnchorYaml[]           = "send_utc_anchor";
constexpr char kValidateIqDataBufferSizeYaml[]  = "validate_iq_data_buffer_size";
constexpr char kRandomSeedYaml[]                = "random_seed";
constexpr char kShuffleCplaneTxYaml[]           = "shuffle_cplane_tx";
constexpr char kShuffleUplaneTxYaml[]           = "shuffle_uplane_tx";
constexpr char kMaxTxTimestampDiffYaml[]        = "max_tx_timestamp_diff_ns";
constexpr char kStartFrameIdYaml[]              = "frame_id";
constexpr char kStartSubframeIdYaml[]           = "subframe_id";
constexpr char kStartSectionIdYaml[]            = "slot_id";
constexpr char kSlotDurationYaml[]              = "slot_duration_ns";
constexpr char kSlotCountYaml[]                 = "slot_count";
constexpr char kEnableULUYaml[]                 = "enable_ulu";
constexpr char kEnableDLUYaml[]                 = "enable_dlu";
constexpr char kEnableULCYaml[]                 = "enable_ulc";
constexpr char kEnableDLCYaml[]                 = "enable_dlc";
constexpr char kTestSlotsYaml[]                 = "test_slots";
constexpr char kDLUEnqTimeAdvanceYaml[]         = "dlu_enq_time_advance_ns";
constexpr char kDLCEnqTimeAdvanceYaml[]         = "dlc_enq_time_advance_ns";
constexpr char kULUTxTimeAdvanceYaml[]          = "ulu_tx_time_advance_ns";
constexpr char kULUEnqTimeAdvanceYaml[]         = "ulu_enq_time_advance_ns";
constexpr char kULCOntimePassPctYaml[]          = "ulc_ontime_pass_percentage";
constexpr char kULUOntimePassPctYaml[]          = "ulu_ontime_pass_percentage";
constexpr char kDLCOntimePassPctYaml[]          = "dlc_ontime_pass_percentage";
constexpr char kDLUOntimePassPctYaml[]          = "dlu_ontime_pass_percentage";

constexpr char kNicNameYaml[]            = "nic";
constexpr char kRuNicNameYaml[]          = "ru_nic";
constexpr char kNicMtuYaml[]             = "mtu";
constexpr char kNicCpuMbufsYaml[]        = "cpu_mbufs";
constexpr char kNicUplaneTxHandlesYaml[] = "uplane_tx_handles";
constexpr char kNicTxqCountYaml[]        = "txq_count";
constexpr char kNicTxqSizeYaml[]         = "txq_size";
constexpr char kNicRxqCountYaml[]        = "rxq_count";
constexpr char kNicRxqSizeYaml[]         = "rxq_size";
constexpr char kNicCudaDeviceIdYaml[]    = "cuda_device_id";

constexpr char kIqDataBufferIdYaml[]   = "id";
constexpr char kIqDataBufferSizeYaml[] = "buffer_size";
constexpr char kIqCudaDeviceIdYaml[]   = "cuda_device_id";
constexpr char kIqInputFileYaml[]      = "input_file";

constexpr char kPeerIdYaml[]             = "cell_id";
constexpr char kPeerSrcMacAddrYaml[]     = "src_mac_addr";
constexpr char kPeerDstMacAddrYaml[]     = "dst_mac_addr";
constexpr char kPeerVlanIdYaml[]         = "vlan";
constexpr char kPeerVlanPcpYaml[]        = "pcp";
constexpr char kPeerTxqCountUplaneYaml[] = "txq_count_uplane";
constexpr char kPeerUdIqWithYaml[]       = "ud_iq_width";
constexpr char kPeerUdCompMethYaml[]     = "ud_comp_meth";
constexpr char kPeerTcpAdvDlYaml[]       = "tcp_adv_dl_ns";
constexpr char kPeerT1aMaxCpUlYaml[]     = "t1a_max_cp_ul_ns";
constexpr char kPeerT1aMaxUpYaml[]       = "t1a_max_up_ns";
constexpr char kPeerTa4MinYaml[]         = "ta4_min_ns";
constexpr char kPeerTa4MaxYaml[]         = "ta4_max_ns";
constexpr char kPeerWindowEndYaml[]      = "window_end_ns";

constexpr char kFlowEaxcYaml[]    = "eAxC";
constexpr char kFlowVlanIdYaml[]  = "vlan";
constexpr char kFlowVlanPcpYaml[] = "pcp";
constexpr char kFlowPeerIdYaml[]  = "cell_id";

constexpr char kTxSlotIdYaml[]        = "slot_id";
constexpr char kTxSymbolIdYaml[]      = "symbol_id";
constexpr char kTxSectionCount[]      = "section_count";
constexpr char kTxSections[]          = "sections";
constexpr char kTxDataDirectionYaml[] = "data_direction";
constexpr char kTxStartPrbYaml[]      = "start_prb";
constexpr char kTxNumPrbYaml[]        = "num_prb";
constexpr char kTxIqDataBufferYaml[]  = "iq_data_buffer";
constexpr char kTxSectionIdYaml[]     = "section_id";
constexpr char kTxSectionStartSymYaml[]     = "start_sym";
constexpr char kTxSectionNumSymYaml[]       = "num_sym";
constexpr char kTxSectionStartPrbYaml[]     = "start_prb";
constexpr char kTxSectionNumPrbYaml[]       = "num_prb";

} // namespace fh_gen

#endif //ifndef FH_GENERATOR_DEFAULTS_HPP__
