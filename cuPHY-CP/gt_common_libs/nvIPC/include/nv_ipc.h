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

#ifndef NV_IPC_H_INCLUDED_
#define NV_IPC_H_INCLUDED_

#include <stdint.h>
#include <stddef.h>

#define NV_NAME_MAX_LEN 32

#define NV_IPV4_STRING_LEN 16

/**
 * IPC module types
 *
 * PRIMARY and SECONDARY are for generic IPC use cases.
 */
typedef enum
{
    NV_IPC_MODULE_MAC       = 0, //!< MAC module, same with SECONDARY
    NV_IPC_MODULE_PHY       = 1, //!< PHY module, same with PRIMARY
    NV_IPC_MODULE_SECONDARY = 2, //!< Secondary module for generic IPC
    NV_IPC_MODULE_PRIMARY   = 3, //!< Primary module for generic IPC
    NV_IPC_MODULE_IPC_DUMP  = 4, //!< IPC dump module
    NV_IPC_MODULE_MAX       = 5  //!< Maximum module count
} nv_ipc_module_t;

/** IPC transport types */
typedef enum
{
    NV_IPC_TRANSPORT_UDP  = 0, //!< UDP transport
    NV_IPC_TRANSPORT_SHM  = 1, //!< Shared memory transport
    NV_IPC_TRANSPORT_DPDK = 2, //!< DPDK transport
    NV_IPC_TRANSPORT_DOCA = 3, //!< DOCA transport
    NV_IPC_TRANSPORT_MAX  = 4  //!< Maximum transport type count
} nv_ipc_transport_t;

/** Memory pool identifiers */
typedef enum
{
    NV_IPC_MEMPOOL_CPU_MSG   = 0, //!< CPU message pool
    NV_IPC_MEMPOOL_CPU_DATA  = 1, //!< CPU data pool
    NV_IPC_MEMPOOL_CPU_LARGE = 2, //!< CPU large data pool
    NV_IPC_MEMPOOL_CUDA_DATA = 3, //!< CUDA data pool (Shared GPU memory buffer)
    NV_IPC_MEMPOOL_GPU_DATA  = 4, //!< GPU data pool (GDR copy buffer)
    NV_IPC_MEMPOOL_NUM       = 5  //!< Total number of memory pools
} nv_ipc_mempool_id_t;

/** IPC message structure */
typedef struct
{
    int32_t msg_id;    //!< IPC message ID
    int32_t cell_id;   //!< Cell ID
    int32_t msg_len;   //!< Message part length
    int32_t data_len;  //!< Data part length
    int32_t data_pool; //!< Data memory pool ID
    void*   msg_buf;   //!< Message buffer pointer
    void*   data_buf;  //!< Data buffer pointer
} nv_ipc_msg_t;

/** UDP transport configuration */
typedef struct
{
    int  local_port;                        //!< Local UDP port
    int  remote_port;                       //!< Remote UDP port
    char local_addr[NV_IPV4_STRING_LEN];    //!< Local IPv4 address
    char remote_addr[NV_IPV4_STRING_LEN];   //!< Remote IPv4 address
    int32_t msg_buf_size;                   //!< Message buffer size
    int32_t data_buf_size;                  //!< Data buffer size
} nv_ipc_config_udp_t;

/** Memory pool size configuration */
typedef struct
{
    int32_t buf_size; //!< Buffer size in bytes
    int32_t pool_len; //!< Number of buffers in pool
} nv_ipc_mempool_size_t;

/** Shared memory IPC configuration */
typedef struct
{
    int cuda_device_id;   //!< CUDA device ID for CUDA/GPU DATA memory pool
    int32_t ring_len;     //!< Ring queue length for TX and RX queues
    /*!< Size of all memory pools: MSG, CPU_DATA, CUDA_DATA, GPU_DATA */
    nv_ipc_mempool_size_t mempool_size[NV_IPC_MEMPOOL_NUM];
    char prefix[NV_NAME_MAX_LEN]; //!< Unique name prefix for nv_ipc_t instance
} nv_ipc_config_shm_t;

/** DPDK transport configuration */
typedef struct
{
    int primary;              //!< Primary process flag (initiates shared memory and semaphores)
    int cuda_device_id;       //!< CUDA device ID for CUDA memory pool
    uint16_t need_eal_init;   //!< DPDK EAL initialization flag
    uint16_t lcore_id;        //!< DPDK logical core ID
    uint16_t nic_mtu;         //!< NIC MTU size
    /*!< Size of all memory pools */
    nv_ipc_mempool_size_t mempool_size[NV_IPC_MEMPOOL_NUM];
    char prefix[NV_NAME_MAX_LEN];         //!< Instance name prefix
    char local_nic_pci[NV_NAME_MAX_LEN];  //!< Local NIC PCI address
    char peer_nic_mac[NV_NAME_MAX_LEN];   //!< Peer NIC MAC address
} nv_ipc_config_dpdk_t;

/** DOCA transport configuration */
typedef struct
{
    int primary;           //!< Primary process flag (initiates shared memory and semaphores)
    int cuda_device_id;    //!< CUDA device ID for CUDA memory pool
    uint16_t cpu_core;     //!< CPU core ID
    uint16_t nic_mtu;      //!< NIC MTU size
    /*!< Size of all memory pools */
    nv_ipc_mempool_size_t mempool_size[NV_IPC_MEMPOOL_NUM];
    char prefix[NV_NAME_MAX_LEN];    //!< Instance name prefix
    char host_pci[NV_NAME_MAX_LEN];  //!< Host PCI address
    char dpu_pci[NV_NAME_MAX_LEN];   //!< DPU PCI address
} nv_ipc_config_doca_t;

#define NVIPC_MAX_MSG_ID (256)
#define NVIPC_MAX_CELL_ID (256)

/** Debug and PCAP configuration */
typedef struct
{
    int32_t nvipc_version;             //!< NVIPC version
    int32_t fapi_tb_loc;               //!< FAPI TB data location
    int32_t grpc_forward;              //!< Enable forwarding IPC messages to another queue
    int32_t debug_timing;              //!< Enable timing debug
    int32_t pcap_enable;               //!< Enable PCAP capture
    int32_t pcap_sync_save;            //!< PCAP save mode: 0=async (Save in SHM caching thread), 1=sync (Save in caller thread)
    int32_t pcap_shm_caching_cpu_core; //!< CPU core for PCAP SHM caching thread
    int32_t pcap_file_saving_cpu_core; //!< CPU core for PCAP file saving thread
    int32_t pcap_cache_size_bits;      //!< PCAP cache size in bits for /dev/shm/${prefix}_pcap
    int32_t pcap_file_size_bits;       //!< PCAP file size in bits for /var/log/aerial/${prefix}_pcap, must be > cache_size
    int32_t pcap_max_msg_size;         //!< Maximum message size to capture
    int32_t pcap_max_data_size;        //!< Maximum data size to capture
    uint8_t msg_filter[NVIPC_MAX_MSG_ID];   //!< Message ID filter array
    uint8_t cell_filter[NVIPC_MAX_CELL_ID]; //!< Cell ID filter array
} nv_ipc_debug_config_t;

/** Main IPC configuration structure */
typedef struct
{
    nv_ipc_module_t    module_type;    //!< IPC module type
    nv_ipc_transport_t ipc_transport;  //!< Transport type
    union
    {
        nv_ipc_config_udp_t  udp;      //!< UDP configuration
        nv_ipc_config_shm_t  shm;      //!< Shared memory configuration
        nv_ipc_config_dpdk_t dpdk;     //!< DPDK configuration
        nv_ipc_config_doca_t doca;     //!< DOCA configuration
    } transport_config;                //!< Transport-specific configuration
    nv_ipc_debug_config_t debug_configs; //!< Debug configs (primary process only)
} nv_ipc_config_t;

/**
 * IPC interface structure
 *
 * Provides function pointers for all IPC operations. All functions return < 0 on failure.
 */
typedef struct nv_ipc_t nv_ipc_t;
struct nv_ipc_t
{
    int (*ipc_destroy)(nv_ipc_t* ipc);        //!< Destroy the IPC instance
    int (*tx_allocate)(nv_ipc_t* ipc, nv_ipc_msg_t* msg, uint32_t options);  //!< Allocate TX memory
    int (*tx_release)(nv_ipc_t* ipc, nv_ipc_msg_t* msg);                     //!< Release TX memory
    int (*rx_allocate)(nv_ipc_t* ipc, nv_ipc_msg_t* msg, uint32_t options);  //!< Allocate RX memory
    int (*rx_release)(nv_ipc_t* ipc, nv_ipc_msg_t* msg);                     //!< Release RX memory
    int (*tx_send_msg)(nv_ipc_t* ipc, nv_ipc_msg_t* msg);                    //!< Send message
    int (*tx_tti_sem_post)(nv_ipc_t* ipc);                                   //!< Post semaphore for TX
    int (*rx_tti_sem_wait)(nv_ipc_t* ipc);                                   //!< Wait for semaphore for RX
    int (*rx_recv_msg)(nv_ipc_t* ipc, nv_ipc_msg_t* msg);                    //!< Receive message
    int (*get_fd)(nv_ipc_t* ipc);                                            //!< Get file descriptor (epoll mode only)
    int (*notify)(nv_ipc_t* ipc, int value);                                 //!< Notify event (epoll mode only)
    int (*get_value)(nv_ipc_t* ipc);                                         //!< Read and clear event (epoll mode only)
    int (*cuda_memcpy_to_host)(nv_ipc_t* ipc, void* host, const void* device, size_t size);    //!< CUDA copy device to host
    int (*cuda_memcpy_to_device)(nv_ipc_t* ipc, void* device, const void* host, size_t size);  //!< CUDA copy host to device
    int (*gdr_memcpy_to_host)(nv_ipc_t* ipc, void* host, const void* device, size_t size);     //!< GDR copy device to host
    int (*gdr_memcpy_to_device)(nv_ipc_t* ipc, void* device, const void* host, size_t size);   //!< GDR copy host to device
};

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * Set default configuration for IPC
 *
 * @param[out] cfg Configuration structure to populate
 * @param[in] module_type Module type for this IPC instance
 * @return 0 on success, -1 on failure
 */
int set_nv_ipc_default_config(nv_ipc_config_t* cfg, nv_ipc_module_t module_type);

/**
 * Load IPC configuration from YAML file
 *
 * For SHM IPC:
 * - Primary app: all fields must be present in YAML
 * - Secondary app: only transport.type and shm_config.prefix required
 *
 * @param[out] cfg Configuration structure to populate
 * @param[in] yaml_path Path to YAML configuration file
 * @param[in] module_type Module type for this IPC instance
 * @return 0 on success, -1 on failure
 */
int load_nv_ipc_yaml_config(nv_ipc_config_t* cfg, const char* yaml_path, nv_ipc_module_t module_type);

/**
 * Create IPC interface instance
 *
 * For SHM IPC:
 * - Recommend calling load_nv_ipc_yaml_config() to populate cfg first
 * - Primary app: saves configuration to shared memory
 * - Secondary app: only ipc_transport and prefix used, updates cfg with primary's saved config
 *
 * @param[in,out] cfg Configuration (updated for secondary app in SHM mode)
 * @return Pointer to IPC instance on success, NULL on failure
 */
nv_ipc_t* create_nv_ipc_interface(nv_ipc_config_t* cfg);

/**
 * Get IPC instance by prefix name
 *
 * @param[in] prefix Instance name prefix
 * @return Pointer to IPC instance on success, NULL if not found
 */
nv_ipc_t* nv_ipc_get_instance(const char* prefix);

/**
 * Get buffer size for memory pool
 *
 * @param[in] cfg Configuration structure
 * @param[in] pool_id Memory pool identifier
 * @return Buffer size in bytes on success, -1 on failure
 */
int nv_ipc_get_buf_size(const nv_ipc_config_t* cfg, nv_ipc_mempool_id_t pool_id);

/**
 * Get number of buffers in memory pool
 *
 * @param[in] cfg Configuration structure
 * @param[in] pool_id Memory pool identifier
 * @return Number of buffers on success, -1 on failure
 */
int nv_ipc_get_pool_len(const nv_ipc_config_t* cfg, nv_ipc_mempool_id_t pool_id);


#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* NV_IPC_H_INCLUDED_ */
