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

#ifndef MLX5_H
#define MLX5_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat"
#pragma nv_diag_suppress 1217 // warning #1217-D: unrecognized format function type "gnu_printf" ignored
#include <rte_common.h>
#include <infiniband/mlx5dv.h>
#pragma nv_diag_default 1217
#pragma GCC diagnostic pop

/* WQE Segment sizes in bytes. */
#define MLX5_WSEG_SIZE 16u

/* The alignment needed for WQ buffer. */
#define MLX5_WQE_BUF_ALIGNMENT rte_mem_page_size()

/* The alignment needed for CQ buffer. */
#define MLX5_CQE_BUF_ALIGNMENT rte_mem_page_size()

/* How many WQEs before asking for CQE */
#define MLX5_TX_COMP_THRESH 1024

/* The completion mode offset in the WQE control segment line 2. */
#define MLX5_COMP_MODE_OFFSET 2

/* Amount of data bytes in minimal inline data segment. */
#define MLX5_DSEG_MIN_INLINE_SIZE 12u

/* Amount of data bytes after eth data segment. */
#define MLX5_ESEG_EXTRA_DATA_SIZE 32u

#ifndef HAVE_MLX5_OPCODE_SEND_EN
#define MLX5_OPCODE_SEND_EN 0x17u
#endif

#ifndef HAVE_MLX5_OPCODE_WAIT
#define MLX5_OPCODE_WAIT 0x0fu
#endif

#define MLX5_OPC_MOD_WAIT_CQ_PI 0u
#define MLX5_OPC_MOD_WAIT_DATA 1u
#define MLX5_OPC_MOD_WAIT_TIME 2u

/* MLNX OFED 5.4 */
#if 0
enum {
	MLX5_OPCODE_NOP			= 0x00,
	MLX5_OPCODE_SEND_INVAL		= 0x01,
	MLX5_OPCODE_RDMA_WRITE		= 0x08,
	MLX5_OPCODE_RDMA_WRITE_IMM	= 0x09,
	MLX5_OPCODE_SEND		= 0x0a,
	MLX5_OPCODE_SEND_IMM		= 0x0b,
	MLX5_OPCODE_LSO			= 0x0e,
	MLX5_OPCODE_RDMA_READ		= 0x10,
	MLX5_OPCODE_ATOMIC_CS		= 0x11,
	MLX5_OPCODE_ATOMIC_FA		= 0x12,
	MLX5_OPCODE_ATOMIC_MASKED_CS	= 0x14,
	MLX5_OPCODE_ATOMIC_MASKED_FA	= 0x15,
	MLX5_OPCODE_BIND_MW		= 0x18,
	MLX5_OPCODE_CONFIG_CMD		= 0x1f,
	MLX5_OPCODE_ENHANCED_MPSW	= 0x29,

	MLX5_RECV_OPCODE_RDMA_WRITE_IMM	= 0x00,
	MLX5_RECV_OPCODE_SEND		= 0x01,
	MLX5_RECV_OPCODE_SEND_IMM	= 0x02,
	MLX5_RECV_OPCODE_SEND_INVAL	= 0x03,

	MLX5_CQE_OPCODE_ERROR		= 0x1e,
	MLX5_CQE_OPCODE_RESIZE		= 0x16,

	MLX5_OPCODE_SET_PSV		= 0x20,
	MLX5_OPCODE_GET_PSV		= 0x21,
	MLX5_OPCODE_CHECK_PSV		= 0x22,
	MLX5_OPCODE_DUMP		= 0x23,
	MLX5_OPCODE_RGET_PSV		= 0x26,
	MLX5_OPCODE_RCHECK_PSV		= 0x27,

	MLX5_OPCODE_UMR			= 0x25,

	MLX5_OPCODE_FLOW_TBL_ACCESS	= 0x2c,
	MLX5_OPCODE_ACCESS_ASO		= 0x2d,
};
#endif

/* Get CQE owner bit. */
#define MLX5_CQE_OWNER(op_own) ((op_own) & MLX5_CQE_OWNER_MASK)

/* Get CQE format. */
#define MLX5_CQE_FORMAT(op_own) (((op_own) & MLX5E_CQE_FORMAT_MASK) >> 2)

/* Get CQE opcode. */
#define MLX5_CQE_OPCODE(op_own) (((op_own) & 0xf0) >> 4)

/* Get CQE solicited event. */
#define MLX5_CQE_SE(op_own) (((op_own) >> 1) & 1)

/* Invalidate a CQE. */
#define MLX5_CQE_INVALIDATE (MLX5_CQE_INVALID << 4)

/* Completion mode. */
enum mlx5_completion_mode {
	MLX5_COMP_ONLY_ERR = 0x0,
	MLX5_COMP_ONLY_FIRST_ERR = 0x1,
	MLX5_COMP_ALWAYS = 0x2,
	MLX5_COMP_CQE_AND_EQE = 0x3,
};

/* CQE status. */
enum mlx5_cqe_status {
	MLX5_CQE_STATUS_SW_OWN = -1,
	MLX5_CQE_STATUS_HW_OWN = -2,
	MLX5_CQE_STATUS_ERR = -3,
};

/* WQE Control segment. */
struct mlx5_wqe_cseg {
	uint32_t opcode;
	uint32_t sq_ds;
	uint32_t flags;
	uint32_t misc;
} __attribute__((__packed__)) __rte_aligned(MLX5_WSEG_SIZE);

/* Header of data segment. Minimal size Data Segment */
struct mlx5_wqe_dseg {
	uint32_t bcount;
	union {
		uint8_t inline_data[MLX5_DSEG_MIN_INLINE_SIZE];
		struct {
			uint32_t lkey;
			uint64_t pbuf;
		} __attribute__((__packed__));
	};
} __attribute__((__packed__));

/* Subset of struct WQE Ethernet Segment. */
struct mlx5_wqe_eseg {
	union {
		struct {
			uint32_t swp_offs;
			uint8_t	cs_flags;
			uint8_t	swp_flags;
			uint16_t mss;
			uint32_t metadata;
			uint16_t inline_hdr_sz;
			union {
				uint16_t inline_data;
				uint16_t vlan_tag;
			};
		} __attribute__((__packed__));
		struct {
			uint32_t offsets;
			uint32_t flags;
			uint32_t flow_metadata;
			uint32_t inline_hdr;
		} __attribute__((__packed__));
	};
} __attribute__((__packed__));

struct mlx5_wqe_qseg {
	uint32_t reserved0;
	uint32_t reserved1;
	uint32_t max_index;
	uint32_t qpn_cqn;
} __attribute__((__packed__));

/* The title WQEBB, header of WQE. */
struct mlx5_wqe {
	union {
		struct mlx5_wqe_cseg cseg;
		uint32_t ctrl[4];
	};
	struct mlx5_wqe_eseg eseg;
	union {
		struct mlx5_wqe_dseg dseg[2];
		uint8_t data[MLX5_ESEG_EXTRA_DATA_SIZE];
	};
} __attribute__((__packed__));

struct mlx5_ds_wqe {
	struct mlx5_wqe_dseg dseg[4];
} __attribute__((__packed__));

#define MLX5_WQE_CSEG_SIZE sizeof(struct mlx5_wqe_cseg)
#define MLX5_WQE_DSEG_SIZE sizeof(struct mlx5_wqe_dseg)
#define MLX5_WQE_ESEG_SIZE sizeof(struct mlx5_wqe_eseg)

/* WQE/WQEBB size in bytes. */
#define MLX5_WQE_SIZE sizeof(struct mlx5_wqe)

/* WQE Segment sizes in bytes. */
#define MLX5_WSEG_SIZE 16u
#define MLX5_WQE_CSEG_SIZE sizeof(struct mlx5_wqe_cseg)
#define MLX5_WQE_DSEG_SIZE sizeof(struct mlx5_wqe_dseg)
#define MLX5_WQE_ESEG_SIZE sizeof(struct mlx5_wqe_eseg)

/*
 * Max size of a WQE session.
 * Absolute maximum size is 63 (MLX5_DSEG_MAX) segments,
 * the WQE size field in Control Segment is 6 bits wide.
 */
#define MLX5_WQE_SIZE_MAX (60 * MLX5_WSEG_SIZE)

#define MLX5_TX_COMP_MAX_CQE 32

/* CQ element structure - should be equal to the cache line size */
struct mlx5_cqe {
#if (RTE_CACHE_LINE_SIZE == 128)
	uint8_t padding[64];
#endif
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
	union {
		uint8_t user_index_bytes[3];
		struct {
			uint8_t user_index_hi;
			uint16_t user_index_low;
		} __attribute__((__packed__));
	};
	uint32_t flow_table_metadata;
	uint8_t rsvd4[4];
	uint32_t byte_cnt;
	uint64_t timestamp;
	uint32_t sop_drop_qpn;
	uint16_t wqe_counter;
	uint8_t rsvd5;
	uint8_t op_own;
};

#endif
