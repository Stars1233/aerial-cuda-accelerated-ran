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

#include "cumac_muUeGrp_test.h"
#include <csignal>

// #define CUMAC_L1_MEM_SHARE_TEST

int CUMAC_MAIN_THREAD_CORE;
int CUMAC_L2_RECV_THREAD_CORE;
int CUMAC_L1_RECV_THREAD_CORE;

int NUM_TIME_SLOTS;
int NUM_CELL;

volatile sig_atomic_t g_shutdown = 0;

static void signal_handler(int signum)
{
    g_shutdown = 1;
}

constexpr int LEN_LOCKFREE_RING_POOL = 10;

// global variables (accessible to all threads)
nv_ipc_t* ipc_cumac_l2 = nullptr;
sem_t task_sem;
nv::lock_free_ring_pool<test_task_t>* test_task_ring;

// CUDA stream for cuMAC
cudaStream_t cuStrmCumac;

void* cumac_l2_blocking_recv_task(void* arg)
{
    // Set thread name, max string length < 16
    pthread_setname_np(pthread_self(), "cumac_l2_recv");
    // Set thread schedule policy to SCHED_FIFO and set priority to 80
    nv_set_sched_fifo_priority(80);
    // Set the thread CPU core
    nv_assign_thread_cpu_core(CUMAC_L2_RECV_THREAD_CORE);

    nv_ipc_msg_t recv_msg;

    int num_slot = 0;
    
    while(num_slot < NUM_TIME_SLOTS && !g_shutdown) {
        NVLOGI(MU_TEST_TAG, "%s: wait for incoming messages notification ...", __func__);

        // Wait for notification of incoming cuMAC scheduling message
        ipc_cumac_l2->rx_tti_sem_wait(ipc_cumac_l2);
        if (g_shutdown) break;
        num_slot++;

        struct timespec msg_recv_start, msg_recv_end;
        clock_gettime(CLOCK_REALTIME, &msg_recv_start);

        test_task_t* task;

        if ((task = test_task_ring->alloc()) == nullptr) {
            NVLOGW(MU_TEST_TAG, "RECV: task process can't catch up with enqueue, drop slot");
            continue;
        }
        task->num_cell = 0;
        task->num_srs_ue = 0;
        task->strm = cuStrmCumac;

        // enqueue the incoming NVIPC message
        while(ipc_cumac_l2->rx_recv_msg(ipc_cumac_l2, &recv_msg) >= 0) {
            cumac_muUeGrp_req_msg_t* req = (cumac_muUeGrp_req_msg_t*)recv_msg.msg_buf;
            cumac_muUeGrp_req_info_t* req_info = (cumac_muUeGrp_req_info_t*) recv_msg.data_buf;
            // NVLOGC(MU_TEST_TAG, "cuMAC L2 RECV: SFN = %u.%u, cell ID = %d, msg_id = 0x%02X %s, msg_len = %d, data_len = %d",
                    //req->sfn, req->slot, recv_msg.cell_id, recv_msg.msg_id, get_cumac_msg_name(recv_msg.msg_id), recv_msg.msg_len, recv_msg.data_len);
            req_info->srsInfoMsh = (cumac_muUeGrp_req_srs_info_msh_t*) (req_info->payload);
            
            task->sfn = req->sfn;
            task->slot = req->slot;
            task->num_cell++;
            task->num_srs_ue += req_info->numSrsInfo;
            task->recv_msg[recv_msg.cell_id] = recv_msg;
        }

        test_task_ring->enqueue(task);
        sem_post(&task_sem);

        clock_gettime(CLOCK_REALTIME, &msg_recv_end);
        int64_t msg_recv_duration = nvlog_timespec_interval(&msg_recv_start, &msg_recv_end);

        NVLOGC(MU_TEST_TAG, "cuMAC-L2 RECV: NVIPC message receive duration: %f microseconds", msg_recv_duration/1000.0);

        NVLOGC(MU_TEST_TAG, "cuMAC-L2 RECV: time slot %d, received messages of %d cells", num_slot-1, task->num_cell);
    }

    NVLOGC(MU_TEST_TAG, "cuMAC-L2 RECV: test completed successfully");
    return NULL;
}

int main(int argc, char** argv)
{
    struct sigaction sa = {};
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    // load simulation parameters
    sys_param_t sys_param(YAML_PARAM_CONFIG_PATH);
    NUM_TIME_SLOTS = sys_param.num_time_slots;
    NUM_CELL = sys_param.num_cell;

    // Set CUDA device (must be same device as producer)
    const uint32_t cuda_device_id = sys_param.cuda_device_id;
    cudaSetDevice(cuda_device_id);

    // SRS Shared Memory Bank
    TestConfig config;
    if (!config.loadFromYaml(YAML_PARAM_CONFIG_PATH)) {
        std::cerr << "Warning: Could not load config from " << YAML_PARAM_CONFIG_PATH << ", using default values" << std::endl;
    }

    uint32_t total_num_buffers = std::min(config.num_srs_buffers, static_cast<uint32_t>(slot_command_api::MAX_SRS_CHEST_BUFFERS));
    uint32_t buffer_size = config.num_prg * config.num_gnb_ant * config.num_ue_layer * sizeof(uint32_t);

    nv_ipc_sem_t* l1_cumac_sem = nv_ipc_sem_open(CUMAC_L1_SECONDARY_PROCESS, L1_SEM_NAME);

    // Create shared memory pool in GPU
    nv_ipc_mempool_t* cpu_mem_pool = nv_ipc_mempool_open(CUMAC_L1_SECONDARY_PROCESS, L1_CPU_MEM_POOL_NAME, (sizeof(CVSrsChestBuff_contMemAlloc)*total_num_buffers + sizeof(l1_cumac_message_t)*MAX_NUM_UE_SRS_INFO_PER_SLOT*NUM_CELL), 1, NV_IPC_MEMPOOL_NO_CUDA_DEV); // CPU memory pool to be shared between L1 and cuMAC
    nv_ipc_mempool_t* gpu_mem_pool = nv_ipc_mempool_open(CUMAC_L1_SECONDARY_PROCESS, L1_GPU_MEM_POOL_NAME, buffer_size, total_num_buffers, config.cuda_device_id); // GPU memory pool to be shared between L1 and cuMAC
    
    // Get CPU/GPU shared memory pool start address
    void* cpu_pool_start_addr = cpu_mem_pool->get_addr(cpu_mem_pool, 0); 
    void* gpu_pool_start_addr = gpu_mem_pool->get_addr(gpu_mem_pool, 0);

    if (!is_aligned_for_type<__half2>(gpu_pool_start_addr)) {
        throw std::runtime_error("GPU memory base address is not aligned for __half2");
    }
    __half2* cubb_srs_gpu_buff_base_addr = reinterpret_cast<__half2*>(gpu_pool_start_addr);

    if (!is_aligned_for_type<CVSrsChestBuff_contMemAlloc>(cpu_pool_start_addr)) {
        throw std::runtime_error("CPU memory base address is not aligned for CVSrsChestBuff_contMemAlloc");
    }
    CVSrsChestBuff_contMemAlloc* arr_cv_srs_chest_buff_base_addr = reinterpret_cast<CVSrsChestBuff_contMemAlloc*>(cpu_pool_start_addr);
    l1_cumac_message_t* arr_l1_cumac_msg = reinterpret_cast<l1_cumac_message_t*>(arr_cv_srs_chest_buff_base_addr + total_num_buffers);

    // cuMAC worker (sender) thread
    NVLOGC(MU_TEST_TAG, "cuMAC-MAIN: parameter config YAML file: %s", YAML_PARAM_CONFIG_PATH);
    NVLOGC(MU_TEST_TAG, "cuMAC-MAIN: cuMAC/L2 primary NVIPC config YAML file: %s", YAML_CUMAC_L2_NVIPC_CONFIG_PATH);
    
    // Load nvipc configuration from YAML file
    nv_ipc_config_t cumac_l2_config;
    load_nv_ipc_yaml_config(&cumac_l2_config, YAML_CUMAC_L2_NVIPC_CONFIG_PATH, NV_IPC_MODULE_PRIMARY);

    // Initialize NVIPC interface and connect to the cuMAC-CP
    if ((ipc_cumac_l2 = create_nv_ipc_interface(&cumac_l2_config)) == NULL)
    {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s: create cuMAC/L2 primary IPC interface failed", __func__);
        return -1;
    }

    // Sleep 1 seconds for NVIPC connection
    usleep(1000000);

    // Initialize semaphore for GPU task finish notification
    sem_init(&task_sem, 0, 0);

    // Initialize CUDA stream for cuMAC
    CHECK_CUDA_ERR(cudaStreamCreate(&cuStrmCumac));

    // Initialize GPU buffers
    uint8_t* srs_chan_est_buf; // GPU buffer for storting SRS channel estimates
    float* srs_snr_buf; // GPU buffer for storting SRS SNRs
    float* chan_orth_mat_buf; // GPU buffer for storing computed channel correlation values (lower triangular matrix)
    uint8_t* gpu_out_buf; // GPU buffer for storing UE grouping results
    uint8_t* gpu_sol_buf_host; // host buffer for storing UE pairing solution
    uint32_t out_buf_len_per_cell = sizeof(cumac_muUeGrp_resp_info_t);
    CHECK_CUDA_ERR(cudaMalloc(&srs_chan_est_buf, sizeof(__half2)*sys_param.num_bs_ant_port*MAX_NUM_UE_ANT_PORT*sys_param.num_subband*sys_param.num_prg_samp_per_subband*MAX_NUM_SRS_UE_PER_CELL*sys_param.num_cell));
    CHECK_CUDA_ERR(cudaMalloc(&srs_snr_buf, sizeof(float)*MAX_NUM_SRS_UE_PER_CELL*sys_param.num_cell));
    CHECK_CUDA_ERR(cudaMalloc(&chan_orth_mat_buf, sizeof(float)*MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT*(MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2*sys_param.num_subband*sys_param.num_prg_samp_per_subband*sys_param.num_cell));
    CHECK_CUDA_ERR(cudaMalloc(&gpu_out_buf, out_buf_len_per_cell*sys_param.num_cell));
    CHECK_CUDA_ERR(cudaMallocHost(&gpu_sol_buf_host, out_buf_len_per_cell*sys_param.num_cell));

    // Initialize CPU buffers for verification
    uint8_t*    srs_chan_est_buf_host;
    float*      srs_snr_buf_host;
    float*      chan_orth_mat_buf_host;
    uint8_t*    cpu_out_buf_host;
    __half2*    cubb_srs_gpu_buff_host_copy_base_addr;
    uint8_t*    task_in_buf_host;
    uint32_t    task_in_buf_size = sys_param.enable_l1_l2_mem_sharing ? sizeof(cumac_muUeGrp_req_info_t) + sizeof(cumac_muUeGrp_req_srs_info_msh_t)*MAX_NUM_UE_SRS_INFO_PER_SLOT + sizeof(cumac_muUeGrp_req_ue_info_t)*MAX_NUM_SRS_UE_PER_CELL : sizeof(cumac_muUeGrp_req_info_t) + sizeof(cumac_muUeGrp_req_srs_info_t)*MAX_NUM_UE_SRS_INFO_PER_SLOT + sizeof(cumac_muUeGrp_req_ue_info_t)*MAX_NUM_SRS_UE_PER_CELL;
    CHECK_CUDA_ERR(cudaMallocHost(&srs_chan_est_buf_host, sizeof(__half2)*sys_param.num_bs_ant_port*MAX_NUM_UE_ANT_PORT*sys_param.num_subband*sys_param.num_prg_samp_per_subband*MAX_NUM_SRS_UE_PER_CELL*sys_param.num_cell));
    CHECK_CUDA_ERR(cudaMallocHost(&srs_snr_buf_host, sizeof(float)*MAX_NUM_SRS_UE_PER_CELL*sys_param.num_cell));
    CHECK_CUDA_ERR(cudaMallocHost(&chan_orth_mat_buf_host, sizeof(float)*MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT*(MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2*sys_param.num_subband*sys_param.num_prg_samp_per_subband*sys_param.num_cell));
    CHECK_CUDA_ERR(cudaMallocHost(&cpu_out_buf_host, out_buf_len_per_cell*sys_param.num_cell));
    CHECK_CUDA_ERR(cudaMallocHost(&cubb_srs_gpu_buff_host_copy_base_addr, buffer_size*total_num_buffers));
    CHECK_CUDA_ERR(cudaMallocHost(&task_in_buf_host, task_in_buf_size*sys_param.num_cell));

    // Initialize lock-free ring pool for tasks
    test_task_ring = new nv::lock_free_ring_pool<test_task_t>("test_task", LEN_LOCKFREE_RING_POOL, sizeof(test_task_t));
    uint32_t ring_len = test_task_ring->get_ring_len();
    for (int i = 0; i < ring_len; i++)
    {
        test_task_t *task = test_task_ring->get_buf_addr(i);
        if (task == nullptr)
        {
            NVLOGE(MU_TEST_TAG, AERIAL_CUMAC_CP_EVENT, "Error cumac_task ring lengh: i=%d length=%d", i, ring_len);
            return -1;
        }

        task->alloc_mem(sys_param.enable_l1_l2_mem_sharing, sys_param.num_cell);
    }

    // ********* Create cuMAC/L2 receiver thread *********
    pthread_t cumac_l2_recv_thread_id;
    CUMAC_L2_RECV_THREAD_CORE = sys_param.cumac_l2_recv_thread_core;
    if (pthread_create(&cumac_l2_recv_thread_id, NULL, cumac_l2_blocking_recv_task, NULL) != 0) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s failed", __func__);
        return -1;
    }
    // ************************************************

    // Set cuMAC worker thread name, max string length < 16
    pthread_setname_np(pthread_self(), "cumac_main");
    // Set thread schedule policy to SCHED_FIFO and set priority to 80
    nv_set_sched_fifo_priority(80);
    // Set the thread CPU core
    CUMAC_MAIN_THREAD_CORE = sys_param.cumac_main_thread_core;
    nv_assign_thread_cpu_core(CUMAC_MAIN_THREAD_CORE);

    // ********* Create cuMAC MU-MIMO UE pairing module object and task structure object *********
    cumac::muMimoUserPairing* muMimoUserPairingObj = new cumac::muMimoUserPairing(srs_chan_est_buf, srs_snr_buf, chan_orth_mat_buf, cubb_srs_gpu_buff_base_addr, sys_param.num_cell, sys_param.num_prg_per_cell, sys_param.num_subband, sys_param.num_prg_samp_per_subband, sys_param.num_bs_ant_port);
    cumac::muUePairTask* muMimoUserPairingTask = new cumac::muUePairTask();
    muMimoUserPairingTask->task_out_buf = gpu_out_buf;
    muMimoUserPairingTask->strm = cuStrmCumac;
    muMimoUserPairingTask->is_mem_sharing = sys_param.enable_l1_l2_mem_sharing;

    // create UE pairing module for CPU verification
    cumac::muMimoUserPairing* muMimoUserPairingObjCpu = new cumac::muMimoUserPairing(srs_chan_est_buf_host, srs_snr_buf_host, chan_orth_mat_buf_host, cubb_srs_gpu_buff_host_copy_base_addr, sys_param.num_cell, sys_param.num_prg_per_cell, sys_param.num_subband, sys_param.num_prg_samp_per_subband, sys_param.num_bs_ant_port);
    cumac::muUePairTask* muMimoUserPairingTaskCpu = new cumac::muUePairTask();
    muMimoUserPairingTaskCpu->task_in_buf = task_in_buf_host;
    muMimoUserPairingTaskCpu->task_out_buf = cpu_out_buf_host;
    muMimoUserPairingTaskCpu->strm = cuStrmCumac;
    muMimoUserPairingTaskCpu->is_mem_sharing = sys_param.enable_l1_l2_mem_sharing;

    int if_failed_task = 0;
    int task_count = 0;
    while(task_count < NUM_TIME_SLOTS && !g_shutdown) {
        // Wait for GPU processing finish notification by semaphore task_sem
        sem_wait(&task_sem);
        if (g_shutdown) break;
 
        const auto cumac_prepare_start = std::chrono::high_resolution_clock::now();

        // Dequeue task from lock-free task ring
        test_task_t* task = test_task_ring->dequeue();
        if (task == nullptr)
        {
            // No task to process
            NVLOGI(MU_TEST_TAG, "%s: no task to process", __func__);
            continue;
        }

        // update task structure for the MU-MIMO user pairing module
        muMimoUserPairingTask->task_in_buf = task->task_in_buf;
        muMimoUserPairingTask->num_srs_ue_per_slot_cell = sys_param.num_srs_ue_per_slot;
        muMimoUserPairingTask->num_blocks_per_row_chanOrtMat = sys_param.num_blocks_per_row_chanOrtMat;
        muMimoUserPairingTask->kernel_launch_flags = sys_param.kernel_launch_flags;

        // for CPU verification
        muMimoUserPairingTaskCpu->num_srs_ue_per_slot_cell = sys_param.num_srs_ue_per_slot;
        muMimoUserPairingTaskCpu->num_blocks_per_row_chanOrtMat = sys_param.num_blocks_per_row_chanOrtMat;
        muMimoUserPairingTaskCpu->kernel_launch_flags = sys_param.kernel_launch_flags;

        // prepare response message and allocate NVIPC buffer
        // Alloc NVIPC buffer: msg_buf will be allocated by default. Set data_pool to get data_buf
        std::vector<uint8_t*> out_data_buf(task->num_cell);
        nv_ipc_msg_t send_msg[task->num_cell];
        for (int cIdx = 0; cIdx < task->num_cell; cIdx++) {
            send_msg[cIdx].data_pool = NV_IPC_MEMPOOL_CPU_DATA;

            // Allocate NVIPC buffer which contains MSG part and DATA part
            if(ipc_cumac_l2->tx_allocate(ipc_cumac_l2, &send_msg[cIdx], 0) != 0) {
                NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s error: NVIPC memory pool is full", __func__);
                return -1;
            }

            // MSG part
            cumac_muUeGrp_resp_msg_t* resp = (cumac_muUeGrp_resp_msg_t*) send_msg[cIdx].msg_buf;
            out_data_buf[cIdx] = (uint8_t*) send_msg[cIdx].data_buf;

            resp->sfn = task->sfn;
            resp->slot = task->slot;
            resp->offsetData = 0;

            // Update the msg_len and data_len of the NVIPC message header
            send_msg[cIdx].msg_id = CUMAC_SCH_TTI_RESPONSE;
            send_msg[cIdx].cell_id = task->recv_msg[cIdx].cell_id;
            send_msg[cIdx].msg_len = sizeof(cumac_muUeGrp_resp_msg_t);
            send_msg[cIdx].data_len = sizeof(cumac_muUeGrp_resp_info_t);
        }

        const auto cumac_prepare_end = std::chrono::high_resolution_clock::now();
        const auto cumac_prepare_duration = std::chrono::duration_cast<std::chrono::microseconds>(cumac_prepare_end - cumac_prepare_start).count();
        NVLOGC(MU_TEST_TAG, "cuMAC-MAIN: response messages preparation duration: %d microseconds", cumac_prepare_duration);

        // Wait for L1 semaphore signal to notify cuMAC that the SRS channel estimation is done
        l1_cumac_sem->sem_wait(l1_cumac_sem);

        // for CPU verification
        CHECK_CUDA_ERR(cudaMemcpy(cubb_srs_gpu_buff_host_copy_base_addr, cubb_srs_gpu_buff_base_addr, buffer_size*total_num_buffers, cudaMemcpyDeviceToHost));

        // Update SRS info structures in the request messages based on the shared L1 SRS memory bank
        update_req_srs_info_msh(arr_cv_srs_chest_buff_base_addr, arr_l1_cumac_msg, task->recv_msg, task->num_srs_ue);
  
        // Copy data from NVIPC buffer to GPU buffer
        // Create CUDA events
        cudaEvent_t startCopyH2D, stopCopyH2D;
        cudaEventCreate(&startCopyH2D);
        cudaEventCreate(&stopCopyH2D);

        cudaEventRecord(startCopyH2D);            
        for (int cIdx = 0; cIdx < task->num_cell; cIdx++) {
            CHECK_CUDA_ERR(cudaMemcpyAsync(task->task_in_buf + cIdx*task->task_in_buf_len_per_cell, task->recv_msg[cIdx].data_buf, task->recv_msg[cIdx].data_len, cudaMemcpyHostToDevice, task->strm));
        }
        cudaEventRecord(stopCopyH2D);

        // setup and run MU-MIMO UE pairing algorithm
        muMimoUserPairingObj->setup(muMimoUserPairingTask);

        muMimoUserPairingObj->run(gpu_sol_buf_host);

        CHECK_CUDA_ERR(cudaStreamSynchronize(task->strm));
        NVLOGC(MU_TEST_TAG, "cuMAC-MAIN: GPU MU-MIMO UE pairing done");

        if (sys_param.print_ue_pairing_solution) {
            print_ue_pairing_sol("cuMAC", task->sfn, task->slot, gpu_sol_buf_host, task->num_cell);
        }

        // for CPU verification
        CHECK_CUDA_ERR(cudaMemcpy(task_in_buf_host, task->task_in_buf, task_in_buf_size*task->num_cell, cudaMemcpyDeviceToHost));
        muMimoUserPairingObjCpu->setup(muMimoUserPairingTaskCpu);
        muMimoUserPairingObjCpu->run_cpu();
        NVLOGC(MU_TEST_TAG, "cuMAC-MAIN: CPU MU-MIMO UE pairing done");

        bool pass = compare_gpu_cpu_results(gpu_sol_buf_host, cpu_out_buf_host, task->num_cell);
        if (pass) {
            NVLOGC(MU_TEST_TAG, "cuMAC-MAIN: GPU and CPU MU-MIMO UE pairing solutions MATCH");
        } else {
            NVLOGC(MU_TEST_TAG, "cuMAC-MAIN: GPU and CPU MU-MIMO UE pairing solutions DO NOT match");
            if_failed_task = 1;
        }

        // Post L1 semaphore signal to notify L1 that the cuMAC UE pairing is done
        l1_cumac_sem->sem_post(l1_cumac_sem);

        // Copy UE pairing solution from pinned host memory to NVIPC buffers
        const auto cumac_h2h_sol_copy_start = std::chrono::high_resolution_clock::now();
        for (int cIdx = 0; cIdx < task->num_cell; cIdx++) {
            std::memcpy(out_data_buf[cIdx], gpu_sol_buf_host + cIdx*out_buf_len_per_cell, out_buf_len_per_cell);
        }
        const auto cumac_h2h_sol_copy_end = std::chrono::high_resolution_clock::now();

        // calculate H2D data transfer duration
        float timeH2D;
        cudaEventElapsedTime(&timeH2D, startCopyH2D, stopCopyH2D);
        NVLOGC(MU_TEST_TAG, "cuMAC-MAIN: H2D data transfer duration: %f microseconds", timeH2D*1000.0);

        // calculate H2H solution copy duration
        const auto cumac_h2h_sol_copy_duration = std::chrono::duration_cast<std::chrono::microseconds>(cumac_h2h_sol_copy_end - cumac_h2h_sol_copy_start).count();
        NVLOGC(MU_TEST_TAG, "cuMAC-MAIN: H2H solution copy duration: %d microseconds", cumac_h2h_sol_copy_duration);

        const auto msg_send_start = std::chrono::high_resolution_clock::now();
        for (int cIdx = 0; cIdx < task->num_cell; cIdx++) {
            // send cuMAC schedule result to L2
            // Send the message
            //NVLOGC(MU_TEST_TAG, "cuMAC L2 SEND: SFN = %u.%u, cell ID = %u, msg_id = 0x%02X %s, msg_len = %d, data_len = %d",
                //task->sfn, task->slot, task->recv_msg[cIdx].cell_id, send_msg[cIdx].msg_id, get_cumac_msg_name(send_msg[cIdx].msg_id), send_msg[cIdx].msg_len, send_msg[cIdx].data_len);
            
            if(ipc_cumac_l2->tx_send_msg(ipc_cumac_l2, &send_msg[cIdx]) < 0) {
                NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s error: send message failed", __func__);
                return -1;
            }
        }

        if(ipc_cumac_l2->tx_tti_sem_post(ipc_cumac_l2) < 0) {
            NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s error: tx notification failed", __func__);
            return -1;
        }

        const auto msg_send_end = std::chrono::high_resolution_clock::now();
        const auto msg_send_duration = std::chrono::duration_cast<std::chrono::microseconds>(msg_send_end - msg_send_start).count();
        NVLOGC(MU_TEST_TAG, "cuMAC-MAIN: NVIPC message sending duration: %d microseconds", msg_send_duration);

        // Release the NVIPC message buffer
        for (int cIdx = 0; cIdx < task->num_cell; cIdx++) {
            ipc_cumac_l2->rx_release(ipc_cumac_l2, &task->recv_msg[cIdx]);
        }

        // Task is finished, free the task buffer
        test_task_ring->free(task);
        task_count++;

        cudaEventDestroy(startCopyH2D);
        cudaEventDestroy(stopCopyH2D);
    }

    if (g_shutdown) {
        NVLOGC(MU_TEST_TAG, "cuMAC-MAIN: received shutdown signal, cleaning up...");
    }

    delete muMimoUserPairingObj;
    delete muMimoUserPairingTask;

    CHECK_CUDA_ERR(cudaFree(srs_chan_est_buf));
    CHECK_CUDA_ERR(cudaFree(srs_snr_buf));
    CHECK_CUDA_ERR(cudaFree(chan_orth_mat_buf));
    CHECK_CUDA_ERR(cudaFree(gpu_out_buf));
    CHECK_CUDA_ERR(cudaFreeHost(gpu_sol_buf_host));
    CHECK_CUDA_ERR(cudaFreeHost(srs_chan_est_buf_host));
    CHECK_CUDA_ERR(cudaFreeHost(srs_snr_buf_host));
    CHECK_CUDA_ERR(cudaFreeHost(chan_orth_mat_buf_host));
    CHECK_CUDA_ERR(cudaFreeHost(cpu_out_buf_host));
    CHECK_CUDA_ERR(cudaFreeHost(cubb_srs_gpu_buff_host_copy_base_addr));
    CHECK_CUDA_ERR(cudaFreeHost(task_in_buf_host));
    for (int i = 0; i < ring_len; i++) {
        CHECK_CUDA_ERR(cudaFree(test_task_ring->get_buf_addr(i)->task_in_buf));
    }

    // Wait for the cuMAC/L2 receiver thread to exit
    int ret = pthread_join(cumac_l2_recv_thread_id, NULL);
    if(ret != 0)
    {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s pthread_join failed, stderr=%s", __func__, strerror(ret));
        return -1;
    }

    ipc_cumac_l2->ipc_destroy(ipc_cumac_l2);

    l1_cumac_sem->close(l1_cumac_sem);
    cpu_mem_pool->close(cpu_mem_pool);
    gpu_mem_pool->close(gpu_mem_pool);

    NVLOGC(MU_TEST_TAG, "cuMAC-MAIN: test completed successfully, ENABLE_L1_L2_MEM_SHARING: %s", (sys_param.enable_l1_l2_mem_sharing ? "true" : "false"));

    if (if_failed_task == 0) {
        NVLOGC(MU_TEST_TAG, "cuMAC-MAIN: UE pairing solution CPU verification - test completed successfully");
    } else {
        NVLOGC(MU_TEST_TAG, "cuMAC-MAIN: UE pairing solution CPU verification - test completed with failures");
    }

    return if_failed_task;
}