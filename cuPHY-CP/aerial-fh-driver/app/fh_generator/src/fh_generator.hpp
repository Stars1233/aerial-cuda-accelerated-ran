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

#ifndef FH_GENERATOR_HPP__
#define FH_GENERATOR_HPP__

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "aerial-fh-driver/api.hpp"
#include "utils.hpp"
#include "gpudevice.hpp"
#include "yaml_parser.hpp"
#include "cuphyoam.hpp"
#include "order_entity.hpp"
#include "cell.hpp"
#include <doca_gpunetio.h>
#include <doca_log.h>

namespace fh_gen
{
class YamlParser;
class Worker;

void cuda_deallocator(void* addr);

using NicMap          = std::unordered_map<std::string, aerial_fh::NicHandle>;
using IqDataBufferMap = std::unordered_map<MemId, std::pair<void*, size_t>>;
using PeerMap         = std::unordered_map<aerial_fh::PeerId, aerial_fh::PeerHandle>;
using FlowMap         = std::map<aerial_fh::PeerId, std::map<aerial_fh::FlowId, aerial_fh::FlowHandle>>;
using CellList        = std::vector<std::unique_ptr<Cell>>;

using UniqueFronthaulHandle = std::unique_ptr<void, decltype(&aerial_fh::close)>;
using UniqueNicHandles      = std::vector<std::unique_ptr<void, decltype(&aerial_fh::remove_nic)>>;
using UniquePeerHandles     = std::vector<std::unique_ptr<void, decltype(&aerial_fh::remove_peer)>>;
using UniqueFlowHandles     = std::vector<std::unique_ptr<void, decltype(&aerial_fh::remove_flow)>>;
using UniqueMemRegHandles   = std::vector<std::unique_ptr<void, decltype(&aerial_fh::unregister_memory)>>;
using UniqueCpuBuffers      = std::vector<std::unique_ptr<void, decltype(&aerial_fh::free_memory)>>;
using UniqueGpuBuffers      = std::vector<std::unique_ptr<void, decltype(&cuda_deallocator)>>;
using UniqueWorkers         = std::vector<std::unique_ptr<Worker>>;
using UniqueGpu             = std::vector<std::unique_ptr<GpuDevice>>;
using UniqueOrderEntities   = std::vector<std::unique_ptr<OrderEntity>>;
using GDR             = std::vector<struct gpinned_buffer*>;

struct FHGenResources
{
    UniqueNicHandles  nics;
    UniquePeerHandles peers;
    UniqueFlowHandles flows;
    UniqueFlowHandles memregs;
    UniqueCpuBuffers  cpu_buffers;
    UniqueGpuBuffers  gpu_buffers;
};

struct UtcAnchor
{
    uint8_t       signature[8]{'t', 'c', 'a', 'n', 'c', 'h', 'o', 'r'};
    aerial_fh::Ns anchor;
};


// counters[cell][packet_type][E/O/L][slot][symbol]
using per_packet_counter = std::atomic<uint64_t>;
using per_symbol_counter = std::array<per_packet_counter, kMaxSymbols>;
using per_slot_counter = std::array<per_symbol_counter, kMaxSlots>;
using per_timing_counter = std::array<per_slot_counter, PacketCounterTiming::CounterTimingMax>;
using per_type_counter = std::array<per_timing_counter, DLPacketCounterType::DLCounterTypeMax>;
using per_cell_counter = std::array<per_type_counter, kMaxCells>;

struct PacketExpectation
{
    uint32_t count[DLPacketCounterType::DLCounterTypeMax + ULPacketCounterType::ULCounterTypeMax][STAT_ARRAY_CELL_SIZE][MAX_LAUNCH_PATTERN_SLOTS][ORAN_ALL_SYMBOLS];
    uint64_t total[DLPacketCounterType::DLCounterTypeMax + ULPacketCounterType::ULCounterTypeMax][STAT_ARRAY_CELL_SIZE];
};

class FhGenerator {
public:
    FhGenerator(const std::string& config_file, FhGenType type);
    ~FhGenerator();

    bool is_ru(){return fh_gen_type_ == RU;};
    bool is_du(){return fh_gen_type_ == DU;};
    void print_periodic_counters();
    void increment_dl_counter(int cell_index, int type, int slot, int timing);
    uint64_t get_cell_type_timing_count(int cell_index, int type, int timing);
    uint64_t get_cell_total_count(int cell_index, int type);
    uint64_t get_ulu_counter_value(int cell_index, int timing);
    float get_cell_timing_percentage(int cell_index, int type, int timing);
    float get_ulu_cell_timing_percentage(int cell_index, int type, int timing);
    void start_ul_tx() {start_ul_tx_ = true;};
    bool check_start_ul_tx() {return start_ul_tx_;};
    packet_slot_timers* get_packet_timer() {return &oran_packet_slot_timers;};
    void set_workers_exit_signal();
    volatile bool   exit_signal() const;
    void            set_exit_signal();
    std::array<std::atomic<uint64_t>, kMaxSlotCount> late_packet_counters;
    aerial_fh::NicHandle        get_nic_handle_from_name(std::string nic_name);
    NicMap                      nics_;
    CellList& get_cell_list(){return cells;};
    Packet_Statistics*   getULCPacketStatistics() {return &ulc_stats;};
    Packet_Statistics*   getULUPacketStatistics() {return &ulu_stats;};
    Packet_Statistics*   getDLCPacketStatistics() {return &dlc_stats;};
    Packet_Statistics*   getDLUPacketStatistics() {return &dlu_stats;};
protected:
    aerial_fh::FronthaulHandle  fhi_;
    FhGenType                   fh_gen_type_;
    // NicMap                      nics_;
    IqDataBufferMap             iq_data_buffers_;
    PeerMap                     peers_;
    FlowMap                     cplane_flows_;
    FlowMap                     uplane_flows_;
    packet_slot_timers oran_packet_slot_timers;

    std::default_random_engine  random_engine_;
    YamlParser                  yaml_parser_;
    UniqueWorkers               workers_;
    UniqueGpu                   gpus_;
    GDR                         buffer_ready_gdr; //GpuComm required
    GDR                         exit_flag;
    UniqueOrderEntities         order_entities;
    CellList                    cells;
    FHGenResources              resources_to_free_;
    std::optional<PrimaryCtxGuard> ctx_guard_;  //!< Primary context RAII guard (DU mode only)
    bool                        start_ul_tx_ = false;
    bool                        cuphy_pti_initialized_ = false;
    std::atomic<bool>           exit_signal_{false};
    std::string                 peer_oam_addr;
    Packet_Statistics           ulu_stats;
    Packet_Statistics           ulc_stats;
    Packet_Statistics           dlc_stats;
    Packet_Statistics           dlu_stats;
    static std::atomic<bool>           synced_with_peer;
    static std::atomic<bool>           ru_send_ack;
    static std::atomic<aerial_fh::Ns>  time_anchor_;
    PacketExpectation pkt_exp;
    void setup_fh_driver();
    void add_nics();
    void add_iq_data_buffers();
    void add_peers();
    void add_cells();
    void add_flows();
    void add_gpu_comm_ready_flags();
    void add_doca_rx_kernel_exit_flag();
    void create_workers();
    void create_ru_workers();
    void create_du_workers();
    void initialize_random_number_generator();
    void print_ru_summary_stats();
    void print_late_packet_stats();
    void print_cell_stats(int cell);
    void free_resources();
    void setup_oam();
    void oam_init();
    void synchronize_peer();
    bool check_packet_count_pass_criteria();
    void calculate_expected_packet_counts();
    static void* du_sfn_slot_sync_cmd_thread_func(void* arg);
    static void* ru_sfn_slot_sync_cmd_thread_func(void* arg);
    int send_sfn_slot_sync_grpc_command();
    void initialize_order_entities();
};

} // namespace fh_gen

#endif //ifndef FH_GENERATOR_HPP__
