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

#ifndef ORDER_KERNEL_TB_HPP__
#define ORDER_KERNEL_TB_HPP__

#include <iostream>
#include <string>
#include <vector>
#include "yaml_parser.hpp"
#include "utils.hpp"
#include "gpudevice.hpp"
#include "hdf5hpp.hpp"
#include "../../../../cuPHY/examples/common/cuphy.hpp" // for cudaGreenContext
#include <map>

// Launch pattern keys
#define YAML_LP_TV              "TV"
#define YAML_LP_NAME            "name"
#define YAML_LP_PATH            "path"
#define YAML_LP_INIT            "INIT"
#define YAML_LP_SCHED           "SCHED"
#define YAML_LP_PDSCH           "PDSCH"
#define YAML_LP_PUSCH           "PUSCH"
#define YAML_LP_PRACH           "PRACH"
#define YAML_LP_PBCH            "PBCH"
#define YAML_LP_PDCCH_DL        "PDCCH_DL"
#define YAML_LP_PDCCH_UL        "PDCCH_UL"
#define YAML_LP_PUCCH           "PUCCH"
#define YAML_LP_SRS             "SRS"
#define YAML_LP_CSIRS           "CSI_RS"
#define YAML_LP_BFW_DL          "BFW_DL"
#define YAML_LP_BFW_UL          "BFW_UL"
#define YAML_LP_SLOT            "slot"
#define YAML_LP_CONFIG          "config"
#define YAML_LP_CELL_INDEX      "cell_index"
#define YAML_LP_CHANNELS        "channels"
#define YAML_LP_CHANNEL_TYPE    "type"
#define YAML_LP_CHANNEL_TV      "tv"
#define YAML_LP_CHANNEL_BEAM_IDS    "beam_ids"
#define YAML_LP_NUM_CELLS       "Num_Cells"
#define YAML_LP_CELL_CCONFIGS       "Cell_Configs"

#define MAX_PATH_LEN 1024
#define CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM 4 // Set CUBB_HOME to N level parent directory of this process. Example: 4 means "../../../../"
#define CONFIG_TEST_VECTOR_PATH "testVectors/"

#define NUM_SLOTS_LP_40     40
#define NUM_SLOTS_LP_80     80

namespace order_kernel_tb
{

    void* allocate_memory(size_t size);
    int free_memory(void* ptr);
}

struct pdu_info
{
    uint8_t startSym;
    uint8_t numSym;
    uint8_t startDataSym;
    uint8_t numDataSym;
    uint8_t dmrsMaxLength;
    uint16_t startPrb;
    uint16_t numPrb;
    uint8_t rb;
    uint8_t freqHopFlag;
    uint16_t secondHopPrb;
    uint8_t numFlows;
    uint32_t tb_size;
    uint32_t dmrsPorts;
    uint64_t freqDomainResource;
    uint8_t scid;
    std::vector<uint8_t> flow_indices;
};

struct tv_info
{
    uint32_t tb_size;
    uint8_t numFlows;
    uint16_t startPrb;
    uint16_t numPrb;
    uint16_t modCompNumPrb;
    uint16_t endPrb;
    uint8_t startSym;
    uint8_t numSym;
    bool is_nr_tv = false;
    std::vector<pdu_info> pdu_infos;
    std::vector<pdu_info> combined_pdu_infos;
    std::array<std::array<bool, MAX_N_PRBS_SUPPORTED>, OFDM_SYMBOLS_PER_SLOT> prb_map{};
    std::array<std::array<uint64_t, MAX_N_PRBS_SUPPORTED>, OFDM_SYMBOLS_PER_SLOT> prb_num_flow_map{};
    std::unordered_map<int, std::unordered_map<int, std::vector<pdu_info>>> fss_pdu_infos;
    std::unordered_map<int, std::unordered_map<int, uint16_t>> fss_numPrb;
    std::unordered_map<int, std::unordered_map<int, std::array<std::array<bool, MAX_N_PRBS_SUPPORTED>, OFDM_SYMBOLS_PER_SLOT>>> fss_prb_map;
    uint16_t nPrbDlBwp = 0;
    uint16_t nPrbUlBwp = 0;
    uint16_t numGnbAnt = 0;
};

struct ul_tv_info : tv_info
{
    uint8_t numSections = 0;
};

namespace nrsim_tv_type{
    enum nrsim_tv_type {
        SSB     = 1,
        PDCCH   = 2,
        PDSCH   = 3,
        CSI_RS  = 4,
        PRACH   = 6,
        PUCCH   = 7,
        PUSCH   = 8,
        SRS     = 9,
        BFW     = 10,
    };
}

struct tv_object
{
    std::string channel_string;
    std::vector<std::string> tv_names;
    std::unordered_map<std::string, int> tv_map;
};

typedef std::unique_ptr<void, decltype(&order_kernel_tb::free_memory)> unique_void_ptr;
struct Dataset {
    Dataset() : size(0), data(nullptr, order_kernel_tb::free_memory) {};
    size_t size;
    unique_void_ptr data;
};

struct Slot {
    Slot(size_t size) {
        for(int ant_idx = 0; ant_idx < size; ++ant_idx)
        {
            ptrs.emplace_back(std::vector<std::vector<void*>>());
            for(int sym_idx = 0; sym_idx < SLOT_NUM_SYMS; ++sym_idx)
            {
                ptrs[ant_idx].emplace_back(std::vector<void*>());
                for(int prb_idx = 0; prb_idx < MAX_N_PRBS_SUPPORTED; ++prb_idx)
                {
                    ptrs[ant_idx][sym_idx].emplace_back(nullptr);
                }
            }
        }
        for(int buf_idx=0;buf_idx<PRACH_MAX_NUM_SEC;buf_idx++)
        {
            for(int ant_idx = 0; ant_idx < size; ++ant_idx)
            {
                ptrs_prach[buf_idx].emplace_back(std::vector<std::vector<void*>>());
                for(int sym_idx = 0; sym_idx < 12; ++sym_idx)
                {
                    ptrs_prach[buf_idx][ant_idx].emplace_back(std::vector<void*>());
                    for(int prb_idx = 0; prb_idx < 12; ++prb_idx)
                    {
                        ptrs_prach[buf_idx][ant_idx][sym_idx].emplace_back(nullptr);
                    }
                }
            }
        }
    }
    /* Pointers for quickly accessing pointers to PRBs */
    std::vector<std::vector<std::vector<void *>>> ptrs;
    std::array<std::vector<std::vector<std::vector<void *>>>,PRACH_MAX_NUM_SEC> ptrs_prach;

    size_t data_sz = 0;
    size_t antenna_sz = 0;
    size_t symbol_sz = 0;
    size_t prb_sz = 0;
    size_t prbs_per_symbol = 0;
    size_t prbs_per_slot = 0;
    int pkts_per_slot = 0;
    Dataset raw_data;
    Dataset raw_data_prach_0;
    Dataset raw_data_prach_1;
    Dataset raw_data_prach_2;
    Dataset raw_data_prach_3;
};

struct ul_tv_object : tv_object
{
    std::array<std::vector<Slot>, IQ_DATA_FMT_MAX> slots; //index by fmt type
    std::array<std::vector<std::vector<Slot> >, IQ_DATA_FMT_MAX> prach_slots;

    std::vector<struct ul_tv_info> tv_info;
};

struct doca_gpu_semaphore_packet {
	enum doca_gpu_semaphore_status status;
	uint32_t num_packets;
	uint64_t doca_buf_idx_start;
};

typedef struct doca_gpu_semaphore_gpu {
	struct doca_gpu_semaphore_packet *pkt_info_gpu; /* Packet info, GPU pointer */
	enum doca_gpu_mem_type pkt_info_mtype;		/* Semaphore memory type */
	uint32_t num_items;				/* Number of items in semaphore */
	uintptr_t custom_info_gpu;			/* User info list, GPU pointer */
	uint32_t custom_info_size;			/* User info item size */
}doca_gpu_semaphore_gpu_t;

struct doca_gpu_buf {
	uintptr_t addr;
	uint32_t size;
	struct doca_gpu_buf_arr *buf_arr;
};

struct doca_gpu_buf_arr {
	uintptr_t addr;
	uint32_t total_bytes;
	uint32_t elem_bytes;
	uint32_t elem_num;
	uint32_t mkey[DOCA_GPUNETIO_BUF_ARR_MAX_DEV];
	uint32_t doca_dev_num;
	struct doca_gpu_buf *buf_list;
};

struct doca_gpu_eth_rxq {
	/* Packets' info */
	struct doca_gpu_buf_arr *buf_arr;
	uint8_t striding_rq;
	uint32_t num_packets;
	uint16_t max_pkt_sz;

	/* RxQ WQE Produce Index. */
	uint32_t wqe_pi;
	uint32_t wqe_num;
	/* RxQ WQE mask. */
	uint32_t wqe_mask;
	/* Number of strides per WQE */
	uint32_t wqe_strides_num;
	/* Last WQE used with striding RQ */
	uint32_t wqe_id_last;
	/* Update next striding WQE */
	uint32_t wqe_id_update;
	/* RxQ WQE buffer. */
	uintptr_t wqe_addr;
	/* RxQ DBREC. */
	uint32_t rq_dbrec_off;
	uintptr_t rq_db_rec;
	uint32_t rq_umem_size;
	uint8_t wqe_on_gpu; /* WQE on GMEM */

	/* CQ CQE Consumer Index. */
	uint32_t cqe_ci;
	uint32_t cqe_num;
	/* CQ CQE mask. */
	uint32_t cqe_mask;
	/* CQ CQE buffer. */
	uintptr_t cqe_addr;
	/* CQ DBREC. */
	uintptr_t cq_db_rec;
	uint32_t cq_umem_size;

	/* RxQ Flush WQE index. */
	uint32_t flush_wqe_pi;
	uint32_t flush_wqe_num;
	/* RxQ Flush WQE mask. */
	uint32_t flush_wqe_mask;
	/* RxQ Flush WQE buffer. */
	uintptr_t flush_wqe_addr;
	/* RxQ Flush DBREC. */
	uintptr_t flush_qp_db_rec;
	/* RxQ Flush SQ DB. */
	uintptr_t flush_qp_db;
	uint32_t flush_dbrec_off;

	/* RxQ Flush CQE index. */
	uint32_t flush_cqe_ci;
	uint32_t flush_cqe_num;
	/* RxQ Flush CQ CQE mask. */
	uint32_t flush_cqe_mask;
	/* RxQ Flush CQ CQE buffer CPU memory pointer. */
	uintptr_t flush_cqe_addr;
	/* RxQ Flush CQ DBREC. */
	uintptr_t flush_cq_db_rec;
	uint32_t *flush_mem_addr;
	uint32_t flush_mkey;
	uint32_t flush_cq_umem_size;
};

struct mlx5_cqe {
	uint8_t pkt_info;
	uint8_t rsvd0;
	uint16_t wqe_id;
	uint8_t lro_tcppsh_abort_dupack;
	uint8_t lro_min_ttl;
	uint16_t lro_tcp_win;
	uint32_t lro_ack_seq_num;
	uint32_t rx_hash_res;
	uint8_t rx_hash_type;
	uint8_t rsvd1[3];
	uint16_t csum;
	uint8_t rsvd2[6];
	uint16_t hdr_type_etc;
	uint16_t vlan_info;
	uint8_t lro_num_seg;
	uint8_t user_index_bytes[3];
	uint32_t flow_table_metadata;
	uint8_t rsvd4[4];
	uint32_t byte_cnt;
	uint64_t timestamp;
	uint32_t sop_drop_qpn;
	uint16_t wqe_counter;
	uint8_t rsvd5;
	uint8_t op_own;
};


typedef struct orderKerneltvParams{
    uint32_t num_cells;
    int launch_pattern_version;
    std::array<std::multimap<std::string,int>,UL_MAX_CELLS_PER_SLOT> tv_to_slot_map;
    std::array<std::unordered_map<int,int>,UL_MAX_CELLS_PER_SLOT> slot_num_to_slot_idx_map;
    std::array<std::unordered_map<int,int>,UL_MAX_CELLS_PER_SLOT> prach_slot_num_to_slot_idx_map;
    std::unordered_map<std::string, std::unordered_set<std::string>> tv_to_channel_map;
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> channel_to_tv_map;
    struct ul_tv_object pusch_object;
    struct ul_tv_object prach_object;
    struct ul_tv_object pucch_object;
    struct ul_tv_object srs_object;
}orderKerneltvParams_t;

typedef struct orderKernelTbConfigParams{
int                   cell_id[UL_MAX_CELLS_PER_SLOT];
uint32_t		      *exit_cond_d[UL_MAX_CELLS_PER_SLOT];
uint16_t         sem_order_num[UL_MAX_CELLS_PER_SLOT];
int		 ru_type[UL_MAX_CELLS_PER_SLOT];
bool*    cell_health;
uint32_t* start_cuphy_d[UL_MAX_CELLS_PER_SLOT];

/* ORAN */
uint8_t		frameId;
uint8_t		subframeId;
uint8_t		slotId;

int		prb_size;
int		comp_meth[UL_MAX_CELLS_PER_SLOT];
int		bit_width[UL_MAX_CELLS_PER_SLOT];
float		beta[UL_MAX_CELLS_PER_SLOT];
uint32_t		*last_sem_idx_order_h[UL_MAX_CELLS_PER_SLOT];

uint32_t       rx_pkt_num_slot[UL_MAX_CELLS_PER_SLOT];
uint8_t*       tb_fh_buf[UL_MAX_CELLS_PER_SLOT];
uint32_t  max_pkt_size;

/* Timer */
uint32_t* early_rx_packets[UL_MAX_CELLS_PER_SLOT];
uint32_t* on_time_rx_packets[UL_MAX_CELLS_PER_SLOT];
uint32_t* late_rx_packets[UL_MAX_CELLS_PER_SLOT];
uint32_t* next_slot_early_rx_packets[UL_MAX_CELLS_PER_SLOT];
uint32_t* next_slot_on_time_rx_packets[UL_MAX_CELLS_PER_SLOT];
uint32_t* next_slot_late_rx_packets[UL_MAX_CELLS_PER_SLOT];
uint32_t* rx_packets_dropped_count[UL_MAX_CELLS_PER_SLOT];

/* Sub-slot processing*/
uint32_t* 			   sym_ord_done_sig_arr;
uint32_t*              sym_ord_done_mask_arr;
uint32_t*              pusch_prb_symbol_map;
uint32_t* 			   num_order_cells_sym_mask_arr;    

/*PUSCH*/
uint8_t*       pusch_buffer[UL_MAX_CELLS_PER_SLOT];
uint16_t*	   pusch_eAxC_map[UL_MAX_CELLS_PER_SLOT];
int			pusch_eAxC_num[UL_MAX_CELLS_PER_SLOT];    
int			pusch_symbols_x_slot;
uint32_t			pusch_prb_x_port_x_symbol[UL_MAX_CELLS_PER_SLOT];
uint32_t*		pusch_ordered_prbs[UL_MAX_CELLS_PER_SLOT];
int			pusch_prb_x_slot[UL_MAX_CELLS_PER_SLOT];

/*PRACH*/
uint16_t 	*prach_eAxC_map[UL_MAX_CELLS_PER_SLOT];
int		prach_eAxC_num[UL_MAX_CELLS_PER_SLOT];
uint8_t		*prach_buffer_0[UL_MAX_CELLS_PER_SLOT];
uint8_t		*prach_buffer_1[UL_MAX_CELLS_PER_SLOT];
uint8_t 	*prach_buffer_2[UL_MAX_CELLS_PER_SLOT];
uint8_t 	*prach_buffer_3[UL_MAX_CELLS_PER_SLOT];
int	    prach_prb_x_slot[UL_MAX_CELLS_PER_SLOT];
int			prach_symbols_x_slot;
uint32_t   prach_prb_x_port_x_symbol[UL_MAX_CELLS_PER_SLOT];
uint32_t	*prach_ordered_prbs[UL_MAX_CELLS_PER_SLOT];
uint16_t	prach_section_id_0;
uint16_t	prach_section_id_1;
uint16_t	prach_section_id_2;
uint16_t	prach_section_id_3;
uint8_t num_order_cells;

/*SRS*/
uint16_t	*srs_eAxC_map[UL_MAX_CELLS_PER_SLOT];
int			srs_eAxC_num[UL_MAX_CELLS_PER_SLOT];
uint8_t		*srs_buffer[UL_MAX_CELLS_PER_SLOT];
int			srs_prb_x_slot[UL_MAX_CELLS_PER_SLOT];
uint32_t              srs_prb_stride[UL_MAX_CELLS_PER_SLOT];	
uint32_t		*srs_ordered_prbs[UL_MAX_CELLS_PER_SLOT];
uint8_t srs_start_sym[UL_MAX_CELLS_PER_SLOT];

/*Receive CTA params*/
uint32_t	timeout_no_pkt_ns;
uint32_t	timeout_first_pkt_ns;
uint32_t  timeout_log_interval_ns;
uint8_t   timeout_log_enable;
uint64_t  *order_kernel_last_timeout_error_time[UL_MAX_CELLS_PER_SLOT];
uint32_t  *last_sem_idx_rx_h[UL_MAX_CELLS_PER_SLOT];
bool            commViaCpu;
struct doca_gpu_eth_rxq *doca_rxq[UL_MAX_CELLS_PER_SLOT];
uint32_t  max_rx_pkts;
uint32_t  rx_pkts_timeout_ns;
doca_gpu_semaphore_gpu_t *sem_gpu[UL_MAX_CELLS_PER_SLOT];
struct aerial_fh_gpu_semaphore_gpu *sem_gpu_aerial_fh[UL_MAX_CELLS_PER_SLOT];
uint64_t		slot_start[UL_MAX_CELLS_PER_SLOT];
uint64_t		ta4_min_ns[UL_MAX_CELLS_PER_SLOT];
uint64_t		ta4_max_ns[UL_MAX_CELLS_PER_SLOT];
uint64_t		slot_duration[UL_MAX_CELLS_PER_SLOT];
uint8_t                ul_rx_pkt_tracing_level;
uint8_t                ul_order_kernel_mode;
}orderKernelTbConfigParams_t;

typedef struct orderKernelTbInputParams{
    uint32_t		      *exit_cond_d[UL_MAX_CELLS_PER_SLOT];
    uint32_t		*last_sem_idx_order_h[UL_MAX_CELLS_PER_SLOT];
    uint32_t* early_rx_packets[UL_MAX_CELLS_PER_SLOT];
    uint32_t* on_time_rx_packets[UL_MAX_CELLS_PER_SLOT];
    uint32_t* late_rx_packets[UL_MAX_CELLS_PER_SLOT];
    uint32_t* next_slot_early_rx_packets[UL_MAX_CELLS_PER_SLOT];
    uint32_t* next_slot_on_time_rx_packets[UL_MAX_CELLS_PER_SLOT];
    uint32_t* next_slot_late_rx_packets[UL_MAX_CELLS_PER_SLOT];
    uint32_t* rx_packets_dropped_count[UL_MAX_CELLS_PER_SLOT];
    bool*     cell_health;
    uint32_t* start_cuphy_d[UL_MAX_CELLS_PER_SLOT];
    uint8_t*       pusch_buffer[UL_MAX_CELLS_PER_SLOT];
    uint8_t*       pusch_buffer_h[UL_MAX_CELLS_PER_SLOT];
    uint16_t*	   pusch_eAxC_map[UL_MAX_CELLS_PER_SLOT];
    uint32_t*		pusch_ordered_prbs[UL_MAX_CELLS_PER_SLOT];
    uint16_t 	*prach_eAxC_map[UL_MAX_CELLS_PER_SLOT];
    uint8_t		*prach_buffer_0[UL_MAX_CELLS_PER_SLOT];
    uint8_t		*prach_buffer_1[UL_MAX_CELLS_PER_SLOT];
    uint8_t 	*prach_buffer_2[UL_MAX_CELLS_PER_SLOT];
    uint8_t 	*prach_buffer_3[UL_MAX_CELLS_PER_SLOT];
    uint8_t		*prach_buffer_0_h[UL_MAX_CELLS_PER_SLOT];
    uint8_t		*prach_buffer_1_h[UL_MAX_CELLS_PER_SLOT];
    uint8_t 	*prach_buffer_2_h[UL_MAX_CELLS_PER_SLOT];
    uint8_t 	*prach_buffer_3_h[UL_MAX_CELLS_PER_SLOT];
    uint32_t	*prach_ordered_prbs[UL_MAX_CELLS_PER_SLOT];
    uint16_t	*srs_eAxC_map[UL_MAX_CELLS_PER_SLOT];
    uint8_t		*srs_buffer[UL_MAX_CELLS_PER_SLOT];
    uint8_t		*srs_buffer_h[UL_MAX_CELLS_PER_SLOT];
    uint32_t		*srs_ordered_prbs[UL_MAX_CELLS_PER_SLOT];
    uint32_t* 			   sym_ord_done_sig_arr;
    uint32_t*              sym_ord_done_mask_arr;
    uint32_t*              pusch_prb_symbol_map;
    uint32_t* 			   num_order_cells_sym_mask_arr;
    /*Receive CTA params*/
    struct doca_gpu_eth_rxq *doca_rxq[UL_MAX_CELLS_PER_SLOT];
    doca_gpu_semaphore_gpu_t *sem_gpu[UL_MAX_CELLS_PER_SLOT];
    struct doca_gpu_semaphore_packet *pkt_info[UL_MAX_CELLS_PER_SLOT];
    struct aerial_fh_gpu_semaphore_gpu *sem_gpu_aerial_fh[UL_MAX_CELLS_PER_SLOT];
    struct mlx5_cqe *cqe_addr[UL_MAX_CELLS_PER_SLOT];
    uint32_t* cq_db_rec[UL_MAX_CELLS_PER_SLOT];
    uint32_t* rq_db_rec[UL_MAX_CELLS_PER_SLOT];    
    uint64_t  *order_kernel_last_timeout_error_time[UL_MAX_CELLS_PER_SLOT];
    uint32_t  *last_sem_idx_rx_h[UL_MAX_CELLS_PER_SLOT];
}orderKernelTbInputParams_t;


typedef struct orderKernelTbConfigFileParams{
    uint32_t cell_id[MAX_UL_SLOTS_OK_TB];
    uint32_t num_rx_packets[MAX_UL_SLOTS_OK_TB];
    uint32_t num_pusch_prbs[MAX_UL_SLOTS_OK_TB];
    uint32_t num_prach_prbs[MAX_UL_SLOTS_OK_TB];
    uint32_t num_srs_prbs[MAX_UL_SLOTS_OK_TB];
    uint8_t  frameId[MAX_UL_SLOTS_OK_TB];
    uint8_t  subframeId[MAX_UL_SLOTS_OK_TB];       
    uint8_t  slotId[MAX_UL_SLOTS_OK_TB];
    int pusch_eAxC_num[MAX_UL_SLOTS_OK_TB];
    std::array<std::vector<uint16_t>, MAX_UL_SLOTS_OK_TB> pusch_eAxC_map;
    int prach_eAxC_num[MAX_UL_SLOTS_OK_TB];
    std::array<std::vector<uint16_t>, MAX_UL_SLOTS_OK_TB> prach_eAxC_map;
    int srs_eAxC_num[MAX_UL_SLOTS_OK_TB];
    std::array<std::vector<uint16_t>, MAX_UL_SLOTS_OK_TB> srs_eAxC_map;    
    uint32_t    pusch_prb_symbol_map[MAX_UL_SLOTS_OK_TB][ORAN_PUSCH_SYMBOLS_X_SLOT];
    uint32_t    num_order_cells_sym_mask[MAX_UL_SLOTS_OK_TB][ORAN_PUSCH_SYMBOLS_X_SLOT];
    uint8_t*       pusch_buffer_tv[MAX_UL_SLOTS_OK_TB];
}orderKernelTbConfigFileParams_t;

typedef struct slotInfo{
    int frameId;
    int subframeId;
    int slotId;
    int pusch_pucch_mismatch_count;
    int prach_mismatch_count;
    int srs_mismatch_count;
}slotInfo_t;

namespace order_kernel_tb
{
// using UniqueGpu             = std::vector<std::unique_ptr<GpuDevice>>;
// using GDR             = std::vector<struct gpinned_buffer*>;

class OrderKernelTestBench {
private:
    std::string config_file_;
    std::array<std::string,UL_MAX_CELLS_PER_SLOT> binary_file_;
    std::string output_file_;
    std::string launch_pattern_file_;
    YamlParser yaml_parser_;
    // UniqueGpu                   gpus_;
    // GDR                         buffer_ready_gdr; //GpuComm required
    std::array<uint8_t*,UL_MAX_CELLS_PER_SLOT> fh_buf_ok_tb;
    uint32_t start_test_slot;
    uint32_t num_test_slots;
    uint32_t num_test_cells;
    uint8_t  same_test_slot;
    uint32_t num_mps_sms;
    uint32_t num_gc_sms;
    int enable_mimo;
    int enable_srs;
    uint32_t max_rx_ant;
    uint32_t num_ant_ports; //Applicable only for mMIMO
    uint32_t num_ant_ports_prach; //Applicable only for mMIMO
    orderKernelTbConfigParams_t* ok_tb_config_params;
    orderKernelTbInputParams_t ok_tb_input_params;
    orderKerneltvParams_t      ok_tb_tv_params;
    std::array<orderKernelTbConfigFileParams_t,UL_MAX_CELLS_PER_SLOT> ok_tb_config_file_params;
    std::array<float,MAX_TEST_SLOTS_OK_TB> process_dur_us;
    slotInfo_t* slot_info;
    uint32_t ok_tb_max_packet_size;
    uint32_t ok_tb_num_valid_slots;

    cuphy::cudaGreenContext   greenCtx_oktb;
    CUcontext                 cuCtx_oktb;
    cudaStream_t              stream_oktb;
    uint32_t                  gpuId;
    CUdevice                  cuDev;
    int                       initStatus; //0: success, 1: failed

    cudaEvent_t start_ok_tb_process;
    cudaEvent_t end_ok_tb_process;
    struct doca_gpu_eth_rxq doca_rxq_info[UL_MAX_CELLS_PER_SLOT];    
public:
    // Constructor to initialize the class
  OrderKernelTestBench(
      std::string config_file,
      std::array<std::string, UL_MAX_CELLS_PER_SLOT> &binary_file,
      std::string launch_pattern_file,
      std::string output_file, uint32_t start_slot, uint32_t num_slots,
      uint32_t num_cells, uint8_t same_slot, uint32_t mps_sm_count, uint32_t gc_count, uint8_t mimo, uint8_t srs_enabled);
  ~OrderKernelTestBench();

    int run_test();
    std::string get_config_file() const {
        return config_file_;
    }
    int initialize();
    void add_gpu_comm_ready_flags();
    void setup_input_params();
    void setup_config_params();
    void read_ok_tb_config_file_params();
    void get_process_kernel_run_duration(int slot_count);
    void write_output_file();
    void parse_launch_pattern(std::string& yaml_file);
    void load_pusch_tvs();
    void load_prach_tvs();
    void load_srs_tvs();
    void launch_pattern_v2_tv_pre_processing(yaml::node& root);
    void parse_launch_pattern_channel(yaml::node& root, std::string key, tv_object& tv_object);
    void yaml_assign_launch_pattern_tv(yaml::node root, std::string key, std::vector<std::string>& tvs, std::unordered_map<std::string, int>& tv_map);
    void read_cell_cfg_from_tv(hdf5hpp::hdf5_file & hdf5file, struct tv_info & tv_info, std::string & tv_name);
    int load_num_antenna_from_nr_tv(hdf5hpp::hdf5_file& hdf5file);
    int load_num_antenna_from_nr_tv_srs(hdf5hpp::hdf5_file& hdf5file);
    int load_num_antenna_from_nr_prach_tv(hdf5hpp::hdf5_file& hdf5file, std::string dset);
    void load_ul_qams(hdf5hpp::hdf5_file& hdf5file, ul_tv_object& tv_object, ul_tv_info& tv_info, std::string& tv_name);
    void load_ul_qams_prach(hdf5hpp::hdf5_file& hdf5file, ul_tv_object& tv_object, ul_tv_info& tv_info, std::string& tv_name);
    Dataset load_tv_datasets_single(hdf5hpp::hdf5_file& hdf5file, std::string const& file, std::string const dataset);
    Slot dataset_to_slot(Dataset d, size_t num_ante, size_t start_symbol, size_t num_symbols, size_t prb_sz, size_t tv_prbs_per_symbol, size_t start_prb);
    Slot dataset_to_slot_prach(Dataset d_1,Dataset d_2,Dataset d_3,Dataset d_4, size_t num_ante, size_t start_symbol, size_t num_symbols, size_t prb_sz, size_t tv_prbs_per_symbol, size_t start_prb);
    int compare_iq(int test_slot_idx,int frameId,int subframeId,int slotId,std::array<uint32_t,UL_MAX_CELLS_PER_SLOT>& mis_match_counter,bool is_prach);
    bool compare_approx(__half& a,__half& b,float tolf);
    void save_process_kernel_run_info(int slot_count,slotInfo_t& slot_info_curr);
    void OrderKernelSetupDocaParams(uint32_t cell_idx);
    // Additional setters and getters can be added here as needed
};
}

#endif