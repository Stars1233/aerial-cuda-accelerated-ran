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

#include "muMimoUserPairing.cuh"

//#define SCHEDULER_KERNEL_TIME_MEASURE_ 
#ifdef SCHEDULER_KERNEL_TIME_MEASURE_
constexpr uint16_t numRunSchKnlTimeMsr = 1;
#endif

// cuMAC namespace
namespace cumac {
    muMimoUserPairing::muMimoUserPairing(uint8_t*      srs_chan_est_buf_base_addr,
                                         float*        srs_snr_buf_base_addr,
                                         float*        chan_orth_mat_buf_base_addr,
                                         __half2*      cubb_srs_gpu_buf_base_addr,
                                         uint16_t      num_cell,
                                         uint16_t      num_prg,
                                         uint16_t      num_subband,
                                         uint16_t      num_prg_samp_per_subband,
                                         uint16_t      num_bs_ant_port)
    : m_srs_chan_est_buf(srs_chan_est_buf_base_addr),
      m_srs_snr_buf(srs_snr_buf_base_addr),
      m_chan_orth_mat_buf(chan_orth_mat_buf_base_addr),
      m_cubb_srs_gpu_buf(cubb_srs_gpu_buf_base_addr),
      m_num_cell(num_cell),
      m_num_prg(num_prg),
      m_num_subband(num_subband),
      m_num_prg_samp_per_subband(num_prg_samp_per_subband),
      m_num_bs_ant_port(num_bs_ant_port)
    {
        CUDA_CHECK_ERR(cudaMallocHost((void **)&m_cpuDynDescrChanCorr, sizeof(muUePairChanCorrDynDescr)));
        CUDA_CHECK_ERR(cudaMallocHost((void **)&m_cpuDynDescrUePair, sizeof(muUePairAlgDynDescr)));
        CUDA_CHECK_ERR(cudaMalloc((void **)&m_gpuDynDescrChanCorr, sizeof(muUePairChanCorrDynDescr)));
        CUDA_CHECK_ERR(cudaMalloc((void **)&m_gpuDynDescrUePair, sizeof(muUePairAlgDynDescr)));

        m_launchCfgChanCorr = std::make_unique<launchCfg_t>();
        m_launchCfgUePair = std::make_unique<launchCfg_t>();

        m_task_out_buf_len_per_cell = sizeof(cumac_muUeGrp_resp_info_t);
    }

    muMimoUserPairing::~muMimoUserPairing()
    {
        CUDA_CHECK_ERR(cudaFreeHost(m_cpuDynDescrChanCorr));
        CUDA_CHECK_ERR(cudaFreeHost(m_cpuDynDescrUePair));
        CUDA_CHECK_ERR(cudaFree(m_gpuDynDescrChanCorr));
        CUDA_CHECK_ERR(cudaFree(m_gpuDynDescrUePair));
    }

    void muMimoUserPairing::chanCorrKernelSelect()
    {
        void* kernelFunc;
        
        if (m_is_mem_sharing) { // cuBB SRS memory sharing is enabled
            kernelFunc = reinterpret_cast<void*>(muUePairChanCorrKernel_memSharing);
        } else { // cuBB SRS memory sharing is disabled
            kernelFunc = reinterpret_cast<void*>(muUePairChanCorrKernel);
        }

        CUDA_CHECK_ERR(cudaGetFuncBySymbol(&m_launchCfgChanCorr->kernelNodeParamsDriver.func, kernelFunc));

        // launch geometry
        m_blockDimChanCorr = {m_numThrdPerBlkChanCorr, 1, 1};
        m_gridDimChanCorr = {m_numThrdBlkChanCorr, 1, 1};

        CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver   = m_launchCfgChanCorr->kernelNodeParamsDriver;
  
        kernelNodeParamsDriver.blockDimX                  = m_blockDimChanCorr.x;
        kernelNodeParamsDriver.blockDimY                  = m_blockDimChanCorr.y;
        kernelNodeParamsDriver.blockDimZ                  = m_blockDimChanCorr.z;
  
        kernelNodeParamsDriver.gridDimX                   = m_gridDimChanCorr.x;
        kernelNodeParamsDriver.gridDimY                   = m_gridDimChanCorr.y;
        kernelNodeParamsDriver.gridDimZ                   = m_gridDimChanCorr.z;

        kernelNodeParamsDriver.extra                      = nullptr;
        kernelNodeParamsDriver.sharedMemBytes             = 0;
    }

    void muMimoUserPairing::uePairKernelSelect()
    {
        void* kernelFunc;
        
        if (m_is_mem_sharing) { // cuBB SRS memory sharing is enabled
            kernelFunc = reinterpret_cast<void*>(muUePairAlgKernel_memSharing);
        } else { // cuBB SRS memory sharing is disabled
            kernelFunc = reinterpret_cast<void*>(muUePairAlgKernel);
        }

        CUDA_CHECK_ERR(cudaGetFuncBySymbol(&m_launchCfgUePair->kernelNodeParamsDriver.func, kernelFunc));

        m_blockDimUePair = {m_numThrdPerBlkUePair, 1, 1};
        m_gridDimUePair = {m_numThrdBlkUePair, 1, 1};

        CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_launchCfgUePair->kernelNodeParamsDriver;

        kernelNodeParamsDriver.blockDimX = m_blockDimUePair.x;
        kernelNodeParamsDriver.blockDimY = m_blockDimUePair.y;
        kernelNodeParamsDriver.blockDimZ = m_blockDimUePair.z;

        kernelNodeParamsDriver.gridDimX = m_gridDimUePair.x;
        kernelNodeParamsDriver.gridDimY = m_gridDimUePair.y;
        kernelNodeParamsDriver.gridDimZ = m_gridDimUePair.z;

        kernelNodeParamsDriver.extra = nullptr;
        kernelNodeParamsDriver.sharedMemBytes = 0;
    }

    void muMimoUserPairing::setup(muUePairTask* task)
    {
        m_strm = task->strm;

        m_kernel_launch_flags = task->kernel_launch_flags;
        m_is_mem_sharing = task->is_mem_sharing;

        if (m_is_mem_sharing) {
            m_task_in_buf_len_per_cell = sizeof(cumac_muUeGrp_req_info_t) + sizeof(cumac_muUeGrp_req_srs_info_msh_t)*MAX_NUM_UE_SRS_INFO_PER_SLOT + sizeof(cumac_muUeGrp_req_ue_info_t)*MAX_NUM_SRS_UE_PER_CELL;
        } else {
            m_task_in_buf_len_per_cell = sizeof(cumac_muUeGrp_req_info_t) + sizeof(cumac_muUeGrp_req_srs_info_t)*MAX_NUM_UE_SRS_INFO_PER_SLOT + sizeof(cumac_muUeGrp_req_ue_info_t)*MAX_NUM_SRS_UE_PER_CELL;
        }

        if ((m_kernel_launch_flags & 0x01) > 0) { // channel correlation computation kernel is to be launched
            // kernel dimensioning
            m_numBlocksPerRowChanCorr = task->num_blocks_per_row_chanOrtMat;
            m_numThrdPerBlkChanCorr = MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT/m_numBlocksPerRowChanCorr;
            m_numThrdPerBlkChanCorr = m_numThrdPerBlkChanCorr > 1024 ? 1024 : m_numThrdPerBlkChanCorr;

            m_numBlocksPerPrgChanCorr = task->num_srs_ue_per_slot_cell*MAX_NUM_UE_ANT_PORT*m_numBlocksPerRowChanCorr;
            m_numBlocksPerCellChanCorr = m_num_subband*m_num_prg_samp_per_subband*m_numBlocksPerPrgChanCorr;
            m_numThrdBlkChanCorr = m_numBlocksPerCellChanCorr*m_num_cell;

            // dynamic descriptor
            m_cpuDynDescrChanCorr->srs_chan_est_buf = m_srs_chan_est_buf;
            m_cpuDynDescrChanCorr->srs_snr_buf = m_srs_snr_buf;
            m_cpuDynDescrChanCorr->task_in_buf = task->task_in_buf;
            m_cpuDynDescrChanCorr->chan_orth_mat_buf = m_chan_orth_mat_buf;
            m_cpuDynDescrChanCorr->cubb_srs_gpu_buf = m_cubb_srs_gpu_buf;
            m_cpuDynDescrChanCorr->task_in_buf_len_per_cell = m_task_in_buf_len_per_cell;
            m_cpuDynDescrChanCorr->num_cell = m_num_cell;
            m_cpuDynDescrChanCorr->num_bs_ant_port = m_num_bs_ant_port;
            m_cpuDynDescrChanCorr->num_prg = m_num_prg;
            m_cpuDynDescrChanCorr->num_subband = m_num_subband;
            m_cpuDynDescrChanCorr->num_prg_samp_per_subband = m_num_prg_samp_per_subband;
            m_cpuDynDescrChanCorr->num_blocks_per_prg = m_numBlocksPerPrgChanCorr;
            m_cpuDynDescrChanCorr->num_blocks_per_cell = m_numBlocksPerCellChanCorr;
            m_cpuDynDescrChanCorr->num_blocks_per_row_chanOrtMat = m_numBlocksPerRowChanCorr;

            CUDA_CHECK_ERR(cudaMemcpyAsync(m_gpuDynDescrChanCorr, m_cpuDynDescrChanCorr, sizeof(muUePairChanCorrDynDescr), cudaMemcpyHostToDevice, m_strm));

            // select kernel 
            chanCorrKernelSelect();
 
            m_launchCfgChanCorr->kernelArgs[0]                       = &m_gpuDynDescrChanCorr;
            m_launchCfgChanCorr->kernelNodeParamsDriver.kernelParams = &(m_launchCfgChanCorr->kernelArgs[0]);
        }

        if ((m_kernel_launch_flags & 0x02) > 0) { // UE pairing algorithm kernel is to be launched
            // kernel dimensioning
            m_numThrdPerBlkUePair = 1024;
            m_numThrdBlkUePair = m_num_cell;

            // dynamic descriptor
            m_cpuDynDescrUePair->srs_snr_buf = m_srs_snr_buf;
            m_cpuDynDescrUePair->task_in_buf = task->task_in_buf;
            m_cpuDynDescrUePair->task_out_buf = task->task_out_buf;
            m_solution_out_buf = task->task_out_buf;
            m_cpuDynDescrUePair->chan_orth_mat_buf = m_chan_orth_mat_buf;
            m_cpuDynDescrUePair->task_in_buf_len_per_cell = m_task_in_buf_len_per_cell;
            m_cpuDynDescrUePair->task_out_buf_len_per_cell = m_task_out_buf_len_per_cell;
            m_cpuDynDescrUePair->num_cell = m_num_cell;
            m_cpuDynDescrUePair->num_bs_ant_port = m_num_bs_ant_port;
            m_cpuDynDescrUePair->num_subband = m_num_subband;
            m_cpuDynDescrUePair->num_prg_samp_per_subband = m_num_prg_samp_per_subband;

            CUDA_CHECK_ERR(cudaMemcpyAsync(m_gpuDynDescrUePair, m_cpuDynDescrUePair, sizeof(muUePairAlgDynDescr), cudaMemcpyHostToDevice, m_strm));

            // select kernel 
            uePairKernelSelect();

            m_launchCfgUePair->kernelArgs[0]                       = &m_gpuDynDescrUePair;
            m_launchCfgUePair->kernelNodeParamsDriver.kernelParams = &(m_launchCfgUePair->kernelArgs[0]);
        }
    }

    void muMimoUserPairing::run(uint8_t* hSolAddr)
    {
        const CUDA_KERNEL_NODE_PARAMS& chanCorrKernelNodeParamsDriver = m_launchCfgChanCorr->kernelNodeParamsDriver;
        const CUDA_KERNEL_NODE_PARAMS& uePairKernelNodeParamsDriver = m_launchCfgUePair->kernelNodeParamsDriver;

#ifdef SCHEDULER_KERNEL_TIME_MEASURE_
        cudaEvent_t start, stop;
        CUDA_CHECK_ERR(cudaEventCreate(&start));
        CUDA_CHECK_ERR(cudaEventCreate(&stop));
        float milliseconds = 0;
        CUDA_CHECK_ERR(cudaEventRecord(start));
        for (int exeIdx = 0; exeIdx < numRunSchKnlTimeMsr; exeIdx++) {
#endif

        if ((m_kernel_launch_flags & 0x01) > 0) { // launch channel correlation computation kernel
            CUDA_CHECK_RES(cuLaunchKernel(chanCorrKernelNodeParamsDriver.func,
                                          chanCorrKernelNodeParamsDriver.gridDimX,
                                          chanCorrKernelNodeParamsDriver.gridDimY, 
                                          chanCorrKernelNodeParamsDriver.gridDimZ,
                                          chanCorrKernelNodeParamsDriver.blockDimX, 
                                          chanCorrKernelNodeParamsDriver.blockDimY, 
                                          chanCorrKernelNodeParamsDriver.blockDimZ,
                                          chanCorrKernelNodeParamsDriver.sharedMemBytes,
                                          m_strm,
                                          chanCorrKernelNodeParamsDriver.kernelParams,
                                          chanCorrKernelNodeParamsDriver.extra));   
        }

        if ((m_kernel_launch_flags & 0x02) > 0) { // launch UE pairing algorithm kernel
            CUDA_CHECK_RES(cuLaunchKernel(uePairKernelNodeParamsDriver.func,
                                          uePairKernelNodeParamsDriver.gridDimX,
                                          uePairKernelNodeParamsDriver.gridDimY, 
                                          uePairKernelNodeParamsDriver.gridDimZ,
                                          uePairKernelNodeParamsDriver.blockDimX, 
                                          uePairKernelNodeParamsDriver.blockDimY, 
                                          uePairKernelNodeParamsDriver.blockDimZ,
                                          uePairKernelNodeParamsDriver.sharedMemBytes,
                                          m_strm,
                                          uePairKernelNodeParamsDriver.kernelParams,
                                          uePairKernelNodeParamsDriver.extra));
        }

#ifdef SCHEDULER_KERNEL_TIME_MEASURE_
        }
        CUDA_CHECK_ERR(cudaEventRecord(stop));
        CUDA_CHECK_ERR(cudaEventSynchronize(stop));
        CUDA_CHECK_ERR(cudaEventElapsedTime(&milliseconds, start, stop));
        printf("MU-MIMO UE pairing CUDA kernel(s) execution time = %f ms\n", milliseconds/static_cast<float>(numRunSchKnlTimeMsr));

        CUDA_CHECK_ERR(cudaEventDestroy(start));
        CUDA_CHECK_ERR(cudaEventDestroy(stop));
#endif
        if ((m_kernel_launch_flags & 0x02) > 0) {
            CUDA_CHECK_ERR(cudaMemcpyAsync(hSolAddr, m_solution_out_buf, m_task_out_buf_len_per_cell * m_num_cell, cudaMemcpyDeviceToHost, m_strm));
        }
    }

    void muMimoUserPairing::run_cpu()
    {
        if (m_is_mem_sharing) { // cuBB SRS memory sharing is enabled
            if ((m_kernel_launch_flags & 0x01) > 0) { // launch channel correlation computation kernel
                muUePairChanCorrKernel_memSharing_cpu(m_cpuDynDescrChanCorr);
            }

            if ((m_kernel_launch_flags & 0x02) > 0) { // launch UE pairing algorithm kernel
                muUePairAlgKernel_memSharing_cpu(m_cpuDynDescrUePair);
            }
        } else { // cuBB SRS memory sharing is disabled
            if ((m_kernel_launch_flags & 0x01) > 0) { // launch channel correlation computation kernel
                muUePairChanCorrKernel_cpu(m_cpuDynDescrChanCorr);
            }

            if ((m_kernel_launch_flags & 0x02) > 0) { // launch UE pairing algorithm kernel
                muUePairAlgKernel_cpu(m_cpuDynDescrUePair);
            }
        }
    }

    constexpr uint32_t dir = 0; // controls direction of comparator in sorting

    template<typename T1, typename T2>
    inline __device__ void bitonicSort(T1* valueArr, T2* uidArr, uint32_t n)
    {
        for (uint32_t size = 2; size < n; size *= 2) {
            uint32_t d = dir ^ ((threadIdx.x & (size / 2)) != 0);
       
            for (uint32_t stride = size / 2; stride > 0; stride /= 2) {
                if(threadIdx.x < n/2) {
                    uint32_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

                    T1 t;
                    T2 t_uid;
                    if (((valueArr[pos] > valueArr[pos + stride]) || 
                         (valueArr[pos] == valueArr[pos + stride] && uidArr[pos] < uidArr[pos + stride])) == d) {
                        t = valueArr[pos];
                        valueArr[pos] = valueArr[pos + stride];
                        valueArr[pos + stride] = t;
                        t_uid = uidArr[pos];
                        uidArr[pos] = uidArr[pos + stride];
                        uidArr[pos + stride] = t_uid;
                    }
                }
                __syncthreads(); 
            }
        }
    
        for (uint32_t stride = n / 2; stride > 0; stride /= 2) {
            if(threadIdx.x < n/2) {
                uint32_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

                T1 t;
                T2 t_uid;
                if (((valueArr[pos] > valueArr[pos + stride]) || 
                     (valueArr[pos] == valueArr[pos + stride] && uidArr[pos] < uidArr[pos + stride])) == dir) {
                    t = valueArr[pos];
                    valueArr[pos] = valueArr[pos + stride];
                    valueArr[pos + stride] = t;
             
                    t_uid = uidArr[pos];
                    uidArr[pos] = uidArr[pos + stride];
                    uidArr[pos + stride] = t_uid;
                }
            }
            __syncthreads(); 
        }
    }
    __global__ void muUePairChanCorrKernel_memSharing(muUePairChanCorrDynDescr* pDynDescr)
    {
        uint16_t cellId = blockIdx.x / pDynDescr->num_blocks_per_cell;
/*
        //! test
        cumac_muUeGrp_req_info_t* req_info_ptr = (cumac_muUeGrp_req_info_t*) (pDynDescr->task_in_buf + cellId*pDynDescr->task_in_buf_len_per_cell);
        if (cellId == 1 && blockIdx.x % pDynDescr->num_blocks_per_cell == 0 && threadIdx.x == 0) {
            printf("req_info_ptr->numSrsInfo = %d\n", req_info_ptr->numSrsInfo);
            for (int srs_info_idx = 0; srs_info_idx < req_info_ptr->numSrsInfo; srs_info_idx++) {
                printf("srsInfo_msh_ptr[%d].realBuffIdx = %d\n", srs_info_idx, req_info_ptr->srsInfoMsh[srs_info_idx].realBuffIdx);
                printf("srsInfo_msh_ptr[%d].srsStartPrg = %d\n", srs_info_idx, req_info_ptr->srsInfoMsh[srs_info_idx].srsStartPrg);
                printf("srsInfo_msh_ptr[%d].srsStartValidPrg = %d\n", srs_info_idx, req_info_ptr->srsInfoMsh[srs_info_idx].srsStartValidPrg);
                printf("srsInfo_msh_ptr[%d].srsNValidPrg = %d\n", srs_info_idx, req_info_ptr->srsInfoMsh[srs_info_idx].srsNValidPrg);
            }
        }
        //! test
*/
        uint16_t blockInd_in_cell = blockIdx.x - cellId*pDynDescr->num_blocks_per_cell;
        uint16_t prgIdx = blockInd_in_cell/pDynDescr->num_blocks_per_prg;
        uint16_t subbandIdx = prgIdx/pDynDescr->num_prg_samp_per_subband;
        uint16_t prgIdx_in_subband = prgIdx - subbandIdx*pDynDescr->num_prg_samp_per_subband;
        uint16_t real_prg_idx = pDynDescr->num_prg/pDynDescr->num_subband*subbandIdx + (prgIdx_in_subband+1)*((pDynDescr->num_prg/pDynDescr->num_subband-1)/(pDynDescr->num_prg_samp_per_subband+1));
        uint16_t blockIdx_in_prg = blockInd_in_cell - prgIdx*pDynDescr->num_blocks_per_prg;
        uint16_t row_idx_chanOrtMat = blockIdx_in_prg / pDynDescr->num_blocks_per_row_chanOrtMat;
        uint16_t blockIdx_in_row_chanOrtMat = blockIdx_in_prg - row_idx_chanOrtMat*pDynDescr->num_blocks_per_row_chanOrtMat;
        uint16_t srs_info_idx = row_idx_chanOrtMat / MAX_NUM_UE_ANT_PORT;
        uint16_t srs_info_ant_port = row_idx_chanOrtMat - srs_info_idx*MAX_NUM_UE_ANT_PORT;

        cumac_muUeGrp_req_info_t* req_info_ptr = (cumac_muUeGrp_req_info_t*) (pDynDescr->task_in_buf + cellId*pDynDescr->task_in_buf_len_per_cell);
        cumac_muUeGrp_req_srs_info_msh_t* srsInfo_msh_ptr = (cumac_muUeGrp_req_srs_info_msh_t*) (req_info_ptr->payload);
        cumac_muUeGrp_req_ue_info_t* ueInfo_ptr = (cumac_muUeGrp_req_ue_info_t*) (srsInfo_msh_ptr + req_info_ptr->numSrsInfo);

        if (srs_info_idx >= req_info_ptr->numSrsInfo) {
            return;
        }

        uint16_t num_ue_ant_port = srsInfo_msh_ptr[srs_info_idx].nUeAnt;

        if (srs_info_ant_port >= num_ue_ant_port) {
            return;
        }

        uint16_t ue_info_per_block = req_info_ptr->numUeInfo/pDynDescr->num_blocks_per_row_chanOrtMat;
        uint16_t first_ue_info_idx_in_block = ue_info_per_block*blockIdx_in_row_chanOrtMat;
        uint16_t last_ue_info_idx_in_block;
        if (blockIdx_in_row_chanOrtMat == pDynDescr->num_blocks_per_row_chanOrtMat - 1) { // last block
            last_ue_info_idx_in_block = req_info_ptr->numUeInfo - 1;
        } else {
            last_ue_info_idx_in_block = first_ue_info_idx_in_block + ue_info_per_block - 1;
        }

        uint16_t portIdx = threadIdx.x;
        uint16_t tx_port_idx = portIdx%MAX_NUM_UE_ANT_PORT;
        uint16_t num_rnd = ceil(static_cast<float>((last_ue_info_idx_in_block - first_ue_info_idx_in_block + 1)*MAX_NUM_UE_ANT_PORT)/blockDim.x);
    
        __shared__ cuComplex srs_info_chanEst[MAX_NUM_BS_ANT_PORT];

        uint16_t row_idx = srsInfo_msh_ptr[srs_info_idx].id*MAX_NUM_UE_ANT_PORT + srs_info_ant_port;

        // update SRS SNRs
        if (srs_info_ant_port == 0 && blockIdx_in_row_chanOrtMat == 0 && threadIdx.x == 0) {
            pDynDescr->srs_snr_buf[cellId*MAX_NUM_SRS_UE_PER_CELL + srsInfo_msh_ptr[srs_info_idx].id] = __uint_as_float(srsInfo_msh_ptr[srs_info_idx].srsWbSnr);
        }
    
        __half2* chanEst_main_buf_ptr = (__half2*) (pDynDescr->srs_chan_est_buf + sizeof(__half2)*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT*pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband*MAX_NUM_SRS_UE_PER_CELL*cellId);

        float* gpu_srsChanOrt_ptr = pDynDescr->chan_orth_mat_buf + (cellId*pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband + prgIdx)*MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT*(MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2;

        __half2* srsChanEst_srs_info = pDynDescr->cubb_srs_gpu_buf + srsInfo_msh_ptr[srs_info_idx].realBuffIdx * pDynDescr->num_prg * pDynDescr->num_bs_ant_port * MAX_NUM_UE_ANT_PORT;
        
        for (int idx = threadIdx.x; idx < pDynDescr->num_bs_ant_port; idx += blockDim.x) {
            srs_info_chanEst[idx] = __half22float2(srsChanEst_srs_info[real_prg_idx*num_ue_ant_port*pDynDescr->num_bs_ant_port + srs_info_ant_port*pDynDescr->num_bs_ant_port + idx]);
        }
        __syncthreads();
/*
        //! test
        if (cellId == 0 && prgIdx == 0 && srs_info_idx == 0 && srs_info_ant_port == 0 && blockIdx_in_row_chanOrtMat == 0 && threadIdx.x == 0) {
            printf("threadIdx.x = %d, srs_info_chanEst: \n", threadIdx.x);
            for (int idx = 0; idx < num_bs_ant_port; idx++) {
                printf("%f + %fj\n", srs_info_chanEst[idx].x, srs_info_chanEst[idx].y);
            }
            printf("\n");
        }
        //! test
*/
        for (int rIdx = 0; rIdx < num_rnd; rIdx++) {
            cuComplex innerProduct = make_cuComplex(0.0f, 0.0f);
            float corrVal = -1.0f;

            uint16_t real_uIdx_in_block = portIdx/MAX_NUM_UE_ANT_PORT + rIdx*(blockDim.x/MAX_NUM_UE_ANT_PORT);
        
            if ((real_uIdx_in_block + first_ue_info_idx_in_block) <= last_ue_info_idx_in_block) {
                uint16_t num_ue_ant_port_real_uIdx = ueInfo_ptr[real_uIdx_in_block + first_ue_info_idx_in_block].nUeAnt;

                if (tx_port_idx < num_ue_ant_port_real_uIdx) {
                    uint8_t flags_ue_info = ueInfo_ptr[real_uIdx_in_block + first_ue_info_idx_in_block].flags;

                    if ((flags_ue_info & 0x01) > 0 && (flags_ue_info & 0x04) > 0) { // is a valid UE info and SRS chanEst available
                        uint16_t real_ue_id = ueInfo_ptr[real_uIdx_in_block + first_ue_info_idx_in_block].id;

                        if ((flags_ue_info & 0x08) > 0) { // has updated SRS info in the current slot
                            uint16_t srsInfoIdx = ueInfo_ptr[real_uIdx_in_block + first_ue_info_idx_in_block].srsInfoIdx;

                            __half2* srsChanEst_srsInfoIdx = pDynDescr->cubb_srs_gpu_buf + srsInfo_msh_ptr[srsInfoIdx].realBuffIdx * pDynDescr->num_prg * pDynDescr->num_bs_ant_port * MAX_NUM_UE_ANT_PORT;
                            if (row_idx_chanOrtMat == 0) {
                                for (int idx = 0; idx < pDynDescr->num_bs_ant_port; idx++) {
                                    __half2 tmp1 = srsChanEst_srsInfoIdx[real_prg_idx*num_ue_ant_port_real_uIdx*pDynDescr->num_bs_ant_port + tx_port_idx*pDynDescr->num_bs_ant_port + idx];
                                    cuComplex tmp1_complex = __half22float2(tmp1);

                                    innerProduct.x += srs_info_chanEst[idx].x*tmp1_complex.x + srs_info_chanEst[idx].y*tmp1_complex.y;
                                    innerProduct.y += srs_info_chanEst[idx].x*tmp1_complex.y - srs_info_chanEst[idx].y*tmp1_complex.x;
    
                                    chanEst_main_buf_ptr[real_ue_id*pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT + prgIdx*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT + tx_port_idx*pDynDescr->num_bs_ant_port + idx] = tmp1;
                                }
                            } else {
                                for (int idx = 0; idx < pDynDescr->num_bs_ant_port; idx++) {
                                    __half2 tmp1 = srsChanEst_srsInfoIdx[real_prg_idx*num_ue_ant_port_real_uIdx*pDynDescr->num_bs_ant_port + tx_port_idx*pDynDescr->num_bs_ant_port + idx];
                                    cuComplex tmp1_complex = __half22float2(tmp1);
                                
                                    innerProduct.x += srs_info_chanEst[idx].x*tmp1_complex.x + srs_info_chanEst[idx].y*tmp1_complex.y;
                                    innerProduct.y += srs_info_chanEst[idx].x*tmp1_complex.y - srs_info_chanEst[idx].y*tmp1_complex.x;
                                }
                            }
                        } else {    
                            for (int idx = 0; idx < pDynDescr->num_bs_ant_port; idx++) {
                                __half2 tmp1 = chanEst_main_buf_ptr[real_ue_id*pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT + prgIdx*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT + tx_port_idx*pDynDescr->num_bs_ant_port + idx];
                                cuComplex tmp1_complex = __half22float2(tmp1);

                                innerProduct.x += srs_info_chanEst[idx].x*tmp1_complex.x + srs_info_chanEst[idx].y*tmp1_complex.y;
                                innerProduct.y += srs_info_chanEst[idx].x*tmp1_complex.y - srs_info_chanEst[idx].y*tmp1_complex.x;
                            }
                        }
    
                        uint16_t col_idx = real_ue_id*MAX_NUM_UE_ANT_PORT + tx_port_idx;

                        corrVal = sqrt(innerProduct.x*innerProduct.x + innerProduct.y*innerProduct.y);
                
                        if (row_idx >= col_idx) {
                            gpu_srsChanOrt_ptr[row_idx * (row_idx + 1) / 2 + col_idx] = corrVal;
                        } else {
                            gpu_srsChanOrt_ptr[col_idx * (col_idx + 1) / 2 + row_idx] = corrVal;
                        }    
                    } 
                }
            }
        }
    }

    __global__ void muUePairChanCorrKernel(muUePairChanCorrDynDescr* pDynDescr)
    {
        uint16_t cellId = blockIdx.x / pDynDescr->num_blocks_per_cell;
        uint16_t blockInd_in_cell = blockIdx.x - cellId*pDynDescr->num_blocks_per_cell;
        uint16_t prgIdx = blockInd_in_cell/pDynDescr->num_blocks_per_prg;
        uint16_t blockIdx_in_prg = blockInd_in_cell - prgIdx*pDynDescr->num_blocks_per_prg;
        uint16_t row_idx_chanOrtMat = blockIdx_in_prg / pDynDescr->num_blocks_per_row_chanOrtMat;
        uint16_t blockIdx_in_row_chanOrtMat = blockIdx_in_prg - row_idx_chanOrtMat*pDynDescr->num_blocks_per_row_chanOrtMat;
        uint16_t srs_info_idx = row_idx_chanOrtMat / MAX_NUM_UE_ANT_PORT;
        uint16_t srs_info_ant_port = row_idx_chanOrtMat - srs_info_idx*MAX_NUM_UE_ANT_PORT;

        cumac_muUeGrp_req_info_t* req_info_ptr = (cumac_muUeGrp_req_info_t*) (pDynDescr->task_in_buf + cellId*pDynDescr->task_in_buf_len_per_cell);

        cumac_muUeGrp_req_srs_info_t* srsInfo_ptr = (cumac_muUeGrp_req_srs_info_t*) (req_info_ptr->payload);

        cumac_muUeGrp_req_ue_info_t* ueInfo_ptr = (cumac_muUeGrp_req_ue_info_t*) (srsInfo_ptr + req_info_ptr->numSrsInfo);

        if (srs_info_idx >= req_info_ptr->numSrsInfo) {
            return;
        }

        uint16_t num_ue_ant_port = srsInfo_ptr[srs_info_idx].nUeAnt;

        if (srs_info_ant_port >= num_ue_ant_port) {
            return;
        }

        uint16_t ue_info_per_block = req_info_ptr->numUeInfo/pDynDescr->num_blocks_per_row_chanOrtMat;
        uint16_t first_ue_info_idx_in_block = ue_info_per_block*blockIdx_in_row_chanOrtMat;
        uint16_t last_ue_info_idx_in_block;
        if (blockIdx_in_row_chanOrtMat == pDynDescr->num_blocks_per_row_chanOrtMat - 1) { // last block
            last_ue_info_idx_in_block = req_info_ptr->numUeInfo - 1;
        } else {
            last_ue_info_idx_in_block = first_ue_info_idx_in_block + ue_info_per_block - 1;
        }

        uint16_t portIdx = threadIdx.x;
        uint16_t tx_port_idx = portIdx%MAX_NUM_UE_ANT_PORT;
        uint16_t num_rnd = ceil(static_cast<float>((last_ue_info_idx_in_block - first_ue_info_idx_in_block + 1)*MAX_NUM_UE_ANT_PORT)/blockDim.x);
    
        __shared__ cuComplex srs_info_chanEst[MAX_NUM_BS_ANT_PORT];

        uint16_t row_idx = srsInfo_ptr[srs_info_idx].id*MAX_NUM_UE_ANT_PORT + srs_info_ant_port;

        // update SRS SNRs
        if (srs_info_ant_port == 0 && blockIdx_in_row_chanOrtMat == 0 && threadIdx.x == 0) {
            pDynDescr->srs_snr_buf[cellId*MAX_NUM_SRS_UE_PER_CELL + srsInfo_ptr[srs_info_idx].id] = __uint_as_float(srsInfo_ptr[srs_info_idx].srsWbSnr);
        }

        __half2* chanEst_main_buf_ptr = (__half2*) (pDynDescr->srs_chan_est_buf + sizeof(__half2)*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT*pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband*MAX_NUM_SRS_UE_PER_CELL*cellId);

        float* gpu_srsChanOrt_ptr = pDynDescr->chan_orth_mat_buf + (cellId*pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband + prgIdx)*MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT*(MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2;
    
        for (int idx = threadIdx.x; idx < pDynDescr->num_bs_ant_port; idx += blockDim.x) {
            srs_info_chanEst[idx] = __half22float2(srsInfo_ptr[srs_info_idx].srsChanEst[prgIdx*num_ue_ant_port*pDynDescr->num_bs_ant_port + srs_info_ant_port*pDynDescr->num_bs_ant_port + idx]);
        }
        __syncthreads();

        for (int rIdx = 0; rIdx < num_rnd; rIdx++) {
            cuComplex innerProduct = make_cuComplex(0.0f, 0.0f);
            float corrVal = -1.0f;

            uint16_t real_uIdx_in_block = portIdx/MAX_NUM_UE_ANT_PORT + rIdx*(blockDim.x/MAX_NUM_UE_ANT_PORT);
        
            if ((real_uIdx_in_block + first_ue_info_idx_in_block) <= last_ue_info_idx_in_block) {
                uint16_t num_ue_ant_port_real_uIdx = ueInfo_ptr[real_uIdx_in_block + first_ue_info_idx_in_block].nUeAnt;

                if (tx_port_idx < num_ue_ant_port_real_uIdx) {
                    uint8_t flags_ue_info = ueInfo_ptr[real_uIdx_in_block + first_ue_info_idx_in_block].flags;

                    if ((flags_ue_info & 0x01) > 0 && (flags_ue_info & 0x04) > 0) { // is a valid UE info and SRS chanEst available
                        uint16_t real_ue_id = ueInfo_ptr[real_uIdx_in_block + first_ue_info_idx_in_block].id;

                        if ((flags_ue_info & 0x08) > 0) { // has updated SRS info in the current slot
                            uint16_t srsInfoIdx = ueInfo_ptr[real_uIdx_in_block + first_ue_info_idx_in_block].srsInfoIdx;

                            if (row_idx_chanOrtMat == 0) {
                                for (int idx = 0; idx < pDynDescr->num_bs_ant_port; idx++) {
                                    __half2 tmp1 = srsInfo_ptr[srsInfoIdx].srsChanEst[prgIdx*num_ue_ant_port_real_uIdx*pDynDescr->num_bs_ant_port + tx_port_idx*pDynDescr->num_bs_ant_port + idx];
                                    cuComplex tmp1_complex = __half22float2(tmp1);

                                    innerProduct.x += srs_info_chanEst[idx].x*tmp1_complex.x + srs_info_chanEst[idx].y*tmp1_complex.y;
                                    innerProduct.y += srs_info_chanEst[idx].x*tmp1_complex.y - srs_info_chanEst[idx].y*tmp1_complex.x;
    
                                    chanEst_main_buf_ptr[real_ue_id*pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT + prgIdx*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT + tx_port_idx*pDynDescr->num_bs_ant_port + idx] = tmp1;
                                }
                            } else {
                                for (int idx = 0; idx < pDynDescr->num_bs_ant_port; idx++) {
                                    __half2 tmp1 = srsInfo_ptr[srsInfoIdx].srsChanEst[prgIdx*num_ue_ant_port_real_uIdx*pDynDescr->num_bs_ant_port + tx_port_idx*pDynDescr->num_bs_ant_port + idx];
                                    cuComplex tmp1_complex = __half22float2(tmp1);
                                
                                    innerProduct.x += srs_info_chanEst[idx].x*tmp1_complex.x + srs_info_chanEst[idx].y*tmp1_complex.y;
                                    innerProduct.y += srs_info_chanEst[idx].x*tmp1_complex.y - srs_info_chanEst[idx].y*tmp1_complex.x;
                                }
                            }
                        } else {    
                            for (int idx = 0; idx < pDynDescr->num_bs_ant_port; idx++) {
                                __half2 tmp1 = chanEst_main_buf_ptr[real_ue_id*pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT + prgIdx*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT + tx_port_idx*pDynDescr->num_bs_ant_port + idx];
                                cuComplex tmp1_complex = __half22float2(tmp1);

                                innerProduct.x += srs_info_chanEst[idx].x*tmp1_complex.x + srs_info_chanEst[idx].y*tmp1_complex.y;
                                innerProduct.y += srs_info_chanEst[idx].x*tmp1_complex.y - srs_info_chanEst[idx].y*tmp1_complex.x;
                            }
                        }
    
                        uint16_t col_idx = real_ue_id*MAX_NUM_UE_ANT_PORT + tx_port_idx;

                        corrVal = sqrt(innerProduct.x*innerProduct.x + innerProduct.y*innerProduct.y);
                
                        if (row_idx >= col_idx) {
                            gpu_srsChanOrt_ptr[row_idx * (row_idx + 1) / 2 + col_idx] = corrVal;
                        } else {
                            gpu_srsChanOrt_ptr[col_idx * (col_idx + 1) / 2 + row_idx] = corrVal;
                        }    
                    } 
                }
            }
        }
    }

    __global__ void muUePairAlgKernel(muUePairAlgDynDescr* pDynDescr)
    {
        uint32_t                        cellId                  = blockIdx.x;
        cumac_muUeGrp_req_info_t*       req_info_ptr            = (cumac_muUeGrp_req_info_t*) (pDynDescr->task_in_buf + cellId*pDynDescr->task_in_buf_len_per_cell);
        cumac_muUeGrp_req_srs_info_t*   srsInfo_ptr             = (cumac_muUeGrp_req_srs_info_t*) (req_info_ptr->payload);
        float*                          gpu_srsChanOrt_cell_ptr = pDynDescr->chan_orth_mat_buf + 
                                                                  cellId*pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT*
                                                                  (MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2;
        cumac_muUeGrp_resp_info_t*      resp_info_ptr           = (cumac_muUeGrp_resp_info_t*) (pDynDescr->task_out_buf + cellId*pDynDescr->task_out_buf_len_per_cell);

        __shared__ float    weights[512];
        __shared__ uint16_t ueIds[512];
        __shared__ uint16_t ueRnti[MAX_NUM_SRS_UE_PER_CELL];
        __shared__ uint16_t muMimoInd[MAX_NUM_SRS_UE_PER_CELL];
        __shared__ uint16_t nUeAnt[MAX_NUM_SRS_UE_PER_CELL];

        if (threadIdx.x < 512) {
            weights[threadIdx.x] = -1.0;
            ueIds[threadIdx.x] = 0xFFFF;
        }
    
        cumac_muUeGrp_req_ue_info_t* ueInfo_ptr = (cumac_muUeGrp_req_ue_info_t*) (srsInfo_ptr + req_info_ptr->numSrsInfo);

        if (threadIdx.x < req_info_ptr->numUeInfo) {
            uint8_t flags   = ueInfo_ptr[threadIdx.x].flags;
            uint16_t ue_id  = ueInfo_ptr[threadIdx.x].id;
            nUeAnt[ue_id]   = ueInfo_ptr[threadIdx.x].nUeAnt;
            ueRnti[ue_id]   = ueInfo_ptr[threadIdx.x].rnti;
            if ((flags & 0x01) > 0 && ueInfo_ptr[threadIdx.x].bufferSize > 0) { // valid UE info
                weights[threadIdx.x] = powf(__uint_as_float(ueInfo_ptr[threadIdx.x].currRate), __uint_as_float(req_info_ptr->betaCoeff))/__uint_as_float(ueInfo_ptr[threadIdx.x].avgRate);
                ueIds[threadIdx.x] = ue_id;

                if ((flags & 0x02) > 0) { // new TX indication
                    if ((flags & 0x04) > 0) { // SRS chanEst available
                        if (pDynDescr->srs_snr_buf[cellId*MAX_NUM_SRS_UE_PER_CELL + ue_id] >= __uint_as_float(req_info_ptr->srsSnrThr)) {
                            muMimoInd[ue_id] = 1;
                            weights[threadIdx.x] *= __uint_as_float(req_info_ptr->muCoeff);
                        } else {
                            muMimoInd[ue_id] = 0;
                        }
                    } else {
                        muMimoInd[ue_id] = 0;
                    }
                } else { // re-TX
                    muMimoInd[ue_id] = 0;
                }
            } else {
                muMimoInd[ue_id] = 0;
            }
        }
        __syncthreads();

        // Sorting
        uint32_t minPow2 = 2;
        while (minPow2 < req_info_ptr->numUeInfo) {
            minPow2 *= 2;
        }
        bitonicSort<float, uint16_t>(weights, ueIds, minPow2);

        // load correlation values from global memory to shared memory
        __shared__ uint8_t orth_ind[MAX_NUM_UE_FOR_GRP_PER_CELL*MAX_NUM_UE_ANT_PORT*(MAX_NUM_UE_FOR_GRP_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2];
        int totNumElem = req_info_ptr->numUeForGrpPerCell*MAX_NUM_UE_ANT_PORT*(req_info_ptr->numUeForGrpPerCell*MAX_NUM_UE_ANT_PORT+1)/2;

        __shared__ uint16_t schdUeIdPerLayerInGrp[MAX_NUM_LAYER_PER_GRP];
        __shared__ uint8_t  schdAntIdxPerLayerInGrp[MAX_NUM_LAYER_PER_GRP];

        __shared__ int numSchdUeg;
        __shared__ int numUeSchd;

        if (threadIdx.x == 0) {
            numSchdUeg = 0;
            numUeSchd = 0;
        }

        for (uint32_t subbandIdx = 0; subbandIdx < pDynDescr->num_subband; subbandIdx++) {
            for (uint32_t idx = threadIdx.x; idx < totNumElem; idx += blockDim.x) {
                orth_ind[idx] = 0xFF;

                uint32_t row_idx = 0;
                while (row_idx*(row_idx+1)/2 <= idx) {row_idx++;}
                row_idx--;
                uint32_t col_idx = idx-row_idx*(row_idx+1)/2;

                uint32_t ue_idx_row = row_idx/MAX_NUM_UE_ANT_PORT;
                uint32_t ue_idx_col = col_idx/MAX_NUM_UE_ANT_PORT;
                uint32_t ant_port_row = row_idx - ue_idx_row*MAX_NUM_UE_ANT_PORT;
                uint32_t ant_port_col = col_idx - ue_idx_col*MAX_NUM_UE_ANT_PORT;
                uint32_t ue_id_row = ueIds[ue_idx_row];
                uint32_t ue_id_col = ueIds[ue_idx_col];

                if (ue_id_row != 0xFFFF && ue_id_col != 0xFFFF && ant_port_row < nUeAnt[ue_id_row] && ant_port_col < nUeAnt[ue_id_col]) {
                    uint32_t row_idx_main = ue_id_row*MAX_NUM_UE_ANT_PORT + ant_port_row;
                    uint32_t col_idx_main = ue_id_col*MAX_NUM_UE_ANT_PORT + ant_port_col;

                    float corrValues;

                    for (uint32_t prgIdx = 0; prgIdx < pDynDescr->num_prg_samp_per_subband; prgIdx++) {
                        float* gpu_srsChanOrt_ptr = gpu_srsChanOrt_cell_ptr + 
                                                    (subbandIdx*pDynDescr->num_prg_samp_per_subband + prgIdx)*
                                                    MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT*(MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2;

                        if (row_idx_main >= col_idx_main) {
                            corrValues = gpu_srsChanOrt_ptr[row_idx_main*(row_idx_main+1)/2 + col_idx_main];
                        } else {
                            corrValues = gpu_srsChanOrt_ptr[col_idx_main*(col_idx_main+1)/2 + row_idx_main];
                        }
                        corrValues /= sqrt(gpu_srsChanOrt_ptr[row_idx_main*(row_idx_main+1)/2 + row_idx_main]*gpu_srsChanOrt_ptr[col_idx_main*(col_idx_main+1)/2 + col_idx_main]);

                        if (corrValues == 1.0f) {
                            orth_ind[idx] = 1;
                        } else if (corrValues > __uint_as_float(req_info_ptr->chanCorrThr)) {
                            orth_ind[idx] = 0;
                            break;
                        } else {
                            orth_ind[idx] = 1;
                        }
                    }
                }
            }

            if (threadIdx.x < MAX_NUM_LAYER_PER_GRP) {
                schdUeIdPerLayerInGrp[threadIdx.x] = 0xFFFF;
                schdAntIdxPerLayerInGrp[threadIdx.x] = 0xFF;
            }
            __syncthreads();

            // determinie UE pairing solution based on the computed channel orthogonality indication matrix orth_ind
            if (threadIdx.x == 0) {
                uint16_t num_ue_schd_in_grp = 0;
                uint16_t num_layer_schd_in_grp = 0;

                for (int uIdx = 0; uIdx < req_info_ptr->numUeForGrpPerCell; uIdx++) {
                    uint16_t ue_id = ueIds[uIdx];
                    if (ue_id != 0xFFFF) {
                        if (num_ue_schd_in_grp > 0 && muMimoInd[ue_id] == 0) { // not the first UE in the UEG and not feasible for MU-MIMO
                            continue;
                        }

                        uint8_t layerSel = 0x00;
                        uint8_t nLayerSchdUe = 0;
                        for (int aIdx = 0; aIdx < nUeAnt[ue_id]; aIdx++) {
                            bool orthogonal = true;
                            for (int lIdx = 0; lIdx < num_layer_schd_in_grp; lIdx++) { // go through the scheduled layers in the UEG
                                uint32_t row_idx = schdUeIdPerLayerInGrp[lIdx]*MAX_NUM_UE_ANT_PORT + schdAntIdxPerLayerInGrp[lIdx];
                                uint32_t col_idx = uIdx*MAX_NUM_UE_ANT_PORT + aIdx;
                                
                                uint8_t ifOrtho;
                                if (row_idx >= col_idx) {
                                    ifOrtho = orth_ind[row_idx*(row_idx+1)/2 + col_idx];
                                } else {
                                    ifOrtho = orth_ind[col_idx*(col_idx+1)/2 + row_idx];
                                }
                                if (ifOrtho != 1) {
                                    orthogonal = false;
                                    break;
                                }
                            }

                            if (orthogonal) { // if the UE layer is orthogonal to the scheduled layers in the UEG
                                layerSel |= 0x01 << aIdx;

                                schdUeIdPerLayerInGrp[num_layer_schd_in_grp] = uIdx;
                                schdAntIdxPerLayerInGrp[num_layer_schd_in_grp] = aIdx;
                                num_layer_schd_in_grp++;
                                nLayerSchdUe++;

                                if (muMimoInd[ue_id] == 0) { // SU-MIMO UE (first in group)
                                    if (nLayerSchdUe >= req_info_ptr->nMaxLayerPerUeSu) {break;}
                                } else { // MU-MIMO UE
                                    if (nLayerSchdUe >= req_info_ptr->nMaxLayerPerUeMu) {break;}
                                }

                                if (num_layer_schd_in_grp >= req_info_ptr->nMaxLayerPerGrp) {break;}
                            }
                        }

                        if (layerSel != 0x00) { // layer selection found
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].layerSel = layerSel;
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].rnti = ueRnti[ue_id];
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].id = ue_id;
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].ueOrderInGrp = num_ue_schd_in_grp;
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].nSCID = 0xFF;
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].flags = 0x01;

                            ueIds[uIdx] = 0xFFFF;

                            if (muMimoInd[ue_id] == 0) { // the first UE in the UEG and not feasible for MU-MIMO
                                num_ue_schd_in_grp++;
                                numUeSchd++;
                                break;
                            } else {
                                resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].flags |= 0x02;
                                num_ue_schd_in_grp++;
                                numUeSchd++;
                            }

                            if (num_ue_schd_in_grp >= req_info_ptr->nMaxUePerGrp || numUeSchd >= req_info_ptr->nMaxUeSchdPerCellTTI) {break;}
                        }
                    }

                    if (num_layer_schd_in_grp >= req_info_ptr->nMaxLayerPerGrp) {break;}
                }

                if (num_ue_schd_in_grp > 0) {
                    resp_info_ptr->schdUegInfo[numSchdUeg].numUeInGrp = num_ue_schd_in_grp;
                    resp_info_ptr->schdUegInfo[numSchdUeg].flags = 0x01;
                    // PRB allocation
                    resp_info_ptr->schdUegInfo[numSchdUeg].allocPrgStart = req_info_ptr->nPrbGrp/pDynDescr->num_subband*subbandIdx;
                    resp_info_ptr->schdUegInfo[numSchdUeg].allocPrgEnd = subbandIdx == (pDynDescr->num_subband-1) ? req_info_ptr->nPrbGrp : (req_info_ptr->nPrbGrp/pDynDescr->num_subband)*(subbandIdx + 1);
                    
                    numSchdUeg++;
                }   
            }
            __syncthreads();

            if (numSchdUeg >= req_info_ptr->nMaxUegPerCell || numUeSchd >= req_info_ptr->nMaxUeSchdPerCellTTI) {break;}
        }

        if (threadIdx.x == 0) {
            resp_info_ptr->numSchdUeg = numSchdUeg;
        }
    }

    static __global__ void muUePairAlgKernel_memSharing(muUePairAlgDynDescr* pDynDescr)
    {
        uint32_t                            cellId                  = blockIdx.x;
        cumac_muUeGrp_req_info_t*           req_info_ptr            = (cumac_muUeGrp_req_info_t*) (pDynDescr->task_in_buf + cellId*pDynDescr->task_in_buf_len_per_cell);
        cumac_muUeGrp_req_srs_info_msh_t*   srsInfo_msh_ptr         = (cumac_muUeGrp_req_srs_info_msh_t*) (req_info_ptr->payload);
        float*                              gpu_srsChanOrt_cell_ptr = pDynDescr->chan_orth_mat_buf + 
                                                                      cellId*pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT*
                                                                      (MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2;
        cumac_muUeGrp_resp_info_t*          resp_info_ptr           = (cumac_muUeGrp_resp_info_t*) (pDynDescr->task_out_buf + cellId*pDynDescr->task_out_buf_len_per_cell);

        __shared__ float    weights[512];
        __shared__ uint16_t ueIds[512];
        __shared__ uint16_t ueRnti[MAX_NUM_SRS_UE_PER_CELL];
        __shared__ uint16_t muMimoInd[MAX_NUM_SRS_UE_PER_CELL];
        __shared__ uint16_t nUeAnt[MAX_NUM_SRS_UE_PER_CELL];

        if (threadIdx.x < 512) {
            weights[threadIdx.x] = -1.0;
            ueIds[threadIdx.x] = 0xFFFF;
        }
    
        cumac_muUeGrp_req_ue_info_t* ueInfo_ptr = (cumac_muUeGrp_req_ue_info_t*) (srsInfo_msh_ptr + req_info_ptr->numSrsInfo);

        if (threadIdx.x < req_info_ptr->numUeInfo) {
            uint8_t flags   = ueInfo_ptr[threadIdx.x].flags;
            uint16_t ue_id  = ueInfo_ptr[threadIdx.x].id;
            nUeAnt[ue_id]   = ueInfo_ptr[threadIdx.x].nUeAnt;
            ueRnti[ue_id]   = ueInfo_ptr[threadIdx.x].rnti;
            if ((flags & 0x01) > 0 && ueInfo_ptr[threadIdx.x].bufferSize > 0) { // valid UE info
                weights[threadIdx.x] = powf(__uint_as_float(ueInfo_ptr[threadIdx.x].currRate), __uint_as_float(req_info_ptr->betaCoeff))/__uint_as_float(ueInfo_ptr[threadIdx.x].avgRate);
                ueIds[threadIdx.x] = ue_id;

                if ((flags & 0x02) > 0) { // new TX indication
                    if ((flags & 0x04) > 0) { // SRS chanEst available
                        if (pDynDescr->srs_snr_buf[cellId*MAX_NUM_SRS_UE_PER_CELL + ue_id] >= __uint_as_float(req_info_ptr->srsSnrThr)) {
                            muMimoInd[ue_id] = 1;
                            weights[threadIdx.x] *= __uint_as_float(req_info_ptr->muCoeff);
                        } else {
                            muMimoInd[ue_id] = 0;
                        }
                    } else {
                        muMimoInd[ue_id] = 0;
                    }
                } else { // re-TX
                    muMimoInd[ue_id] = 0;
                }
            } else {
                muMimoInd[ue_id] = 0;
            }
        }
        __syncthreads();

        // Sorting
        uint32_t minPow2 = 2;
        while (minPow2 < req_info_ptr->numUeInfo) {
            minPow2 *= 2;
        }
        bitonicSort<float, uint16_t>(weights, ueIds, minPow2);

        // load correlation values from global memory to shared memory
        __shared__ uint8_t orth_ind[MAX_NUM_UE_FOR_GRP_PER_CELL*MAX_NUM_UE_ANT_PORT*(MAX_NUM_UE_FOR_GRP_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2];
        int totNumElem = req_info_ptr->numUeForGrpPerCell*MAX_NUM_UE_ANT_PORT*(req_info_ptr->numUeForGrpPerCell*MAX_NUM_UE_ANT_PORT+1)/2;

        __shared__ uint16_t schdUeIdPerLayerInGrp[MAX_NUM_LAYER_PER_GRP];
        __shared__ uint8_t  schdAntIdxPerLayerInGrp[MAX_NUM_LAYER_PER_GRP];

        __shared__ int numSchdUeg;
        __shared__ int numUeSchd;

        if (threadIdx.x == 0) {
            numSchdUeg = 0;
            numUeSchd = 0;
        }

        for (uint32_t subbandIdx = 0; subbandIdx < pDynDescr->num_subband; subbandIdx++) {
            for (uint32_t idx = threadIdx.x; idx < totNumElem; idx += blockDim.x) {
                orth_ind[idx] = 0xFF;

                uint32_t row_idx = 0;
                while (row_idx*(row_idx+1)/2 <= idx) {row_idx++;}
                row_idx--;
                uint32_t col_idx = idx-row_idx*(row_idx+1)/2;

                uint32_t ue_idx_row = row_idx/MAX_NUM_UE_ANT_PORT;
                uint32_t ue_idx_col = col_idx/MAX_NUM_UE_ANT_PORT;
                uint32_t ant_port_row = row_idx - ue_idx_row*MAX_NUM_UE_ANT_PORT;
                uint32_t ant_port_col = col_idx - ue_idx_col*MAX_NUM_UE_ANT_PORT;
                uint32_t ue_id_row = ueIds[ue_idx_row];
                uint32_t ue_id_col = ueIds[ue_idx_col];

                if (ue_id_row != 0xFFFF && ue_id_col != 0xFFFF && ant_port_row < nUeAnt[ue_id_row] && ant_port_col < nUeAnt[ue_id_col]) {
                    uint32_t row_idx_main = ue_id_row*MAX_NUM_UE_ANT_PORT + ant_port_row;
                    uint32_t col_idx_main = ue_id_col*MAX_NUM_UE_ANT_PORT + ant_port_col;

                    float corrValues;

                    for (uint32_t prgIdx = 0; prgIdx < pDynDescr->num_prg_samp_per_subband; prgIdx++) {
                        float* gpu_srsChanOrt_ptr = gpu_srsChanOrt_cell_ptr + 
                                                    (subbandIdx*pDynDescr->num_prg_samp_per_subband + prgIdx)*
                                                    MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT*(MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2;

                        if (row_idx_main >= col_idx_main) {
                            corrValues = gpu_srsChanOrt_ptr[row_idx_main*(row_idx_main+1)/2 + col_idx_main];
                        } else {
                            corrValues = gpu_srsChanOrt_ptr[col_idx_main*(col_idx_main+1)/2 + row_idx_main];
                        }
                        corrValues /= sqrt(gpu_srsChanOrt_ptr[row_idx_main*(row_idx_main+1)/2 + row_idx_main]*gpu_srsChanOrt_ptr[col_idx_main*(col_idx_main+1)/2 + col_idx_main]);

                        if (corrValues == 1.0f) {
                            orth_ind[idx] = 1;
                        } else if (corrValues > __uint_as_float(req_info_ptr->chanCorrThr)) {
                            orth_ind[idx] = 0;
                            break;
                        } else {
                            orth_ind[idx] = 1;
                        }
                    }
                }
            }

            if (threadIdx.x < MAX_NUM_LAYER_PER_GRP) {
                schdUeIdPerLayerInGrp[threadIdx.x] = 0xFFFF;
                schdAntIdxPerLayerInGrp[threadIdx.x] = 0xFF;
            }
            __syncthreads();

            // determinie UE pairing solution based on the computed channel orthogonality indication matrix orth_ind
            if (threadIdx.x == 0) {
                uint16_t num_ue_schd_in_grp = 0;
                uint16_t num_layer_schd_in_grp = 0;

                for (int uIdx = 0; uIdx < req_info_ptr->numUeForGrpPerCell; uIdx++) {
                    uint16_t ue_id = ueIds[uIdx];
                    if (ue_id != 0xFFFF) {
                        if (num_ue_schd_in_grp > 0 && muMimoInd[ue_id] == 0) { // not the first UE in the UEG and not feasible for MU-MIMO
                            continue;
                        }

                        uint8_t layerSel = 0x00;
                        uint8_t nLayerSchdUe = 0;
                        for (int aIdx = 0; aIdx < nUeAnt[ue_id]; aIdx++) {
                            bool orthogonal = true;
                            for (int lIdx = 0; lIdx < num_layer_schd_in_grp; lIdx++) { // go through the scheduled layers in the UEG
                                uint32_t row_idx = schdUeIdPerLayerInGrp[lIdx]*MAX_NUM_UE_ANT_PORT + schdAntIdxPerLayerInGrp[lIdx];
                                uint32_t col_idx = uIdx*MAX_NUM_UE_ANT_PORT + aIdx;
                                
                                uint8_t ifOrtho;
                                if (row_idx >= col_idx) {
                                    ifOrtho = orth_ind[row_idx*(row_idx+1)/2 + col_idx];
                                } else {
                                    ifOrtho = orth_ind[col_idx*(col_idx+1)/2 + row_idx];
                                }
                                if (ifOrtho != 1) {
                                    orthogonal = false;
                                    break;
                                }
                            }

                            if (orthogonal) { // if the UE layer is orthogonal to the scheduled layers in the UEG
                                layerSel |= 0x01 << aIdx;

                                schdUeIdPerLayerInGrp[num_layer_schd_in_grp] = uIdx;
                                schdAntIdxPerLayerInGrp[num_layer_schd_in_grp] = aIdx;
                                num_layer_schd_in_grp++;
                                nLayerSchdUe++;

                                if (muMimoInd[ue_id] == 0) { // SU-MIMO UE (first in group)
                                    if (nLayerSchdUe >= req_info_ptr->nMaxLayerPerUeSu) {break;}
                                } else { // MU-MIMO UE
                                    if (nLayerSchdUe >= req_info_ptr->nMaxLayerPerUeMu) {break;}
                                }

                                if (num_layer_schd_in_grp >= req_info_ptr->nMaxLayerPerGrp) {break;}
                            }
                        }

                        if (layerSel != 0x00) { // layer selection found
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].layerSel = layerSel;
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].rnti = ueRnti[ue_id];
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].id = ue_id;
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].ueOrderInGrp = num_ue_schd_in_grp;
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].nSCID = 0xFF;
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].flags = 0x01;

                            ueIds[uIdx] = 0xFFFF;

                            if (muMimoInd[ue_id] == 0) { // the first UE in the UEG and not feasible for MU-MIMO
                                num_ue_schd_in_grp++;
                                numUeSchd++;
                                break;
                            } else {
                                resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].flags |= 0x02;
                                num_ue_schd_in_grp++;
                                numUeSchd++;
                            }

                            if (num_ue_schd_in_grp >= req_info_ptr->nMaxUePerGrp || numUeSchd >= req_info_ptr->nMaxUeSchdPerCellTTI) {break;}
                        }
                    }

                    if (num_layer_schd_in_grp >= req_info_ptr->nMaxLayerPerGrp) {break;}
                }

                if (num_ue_schd_in_grp > 0) {
                    resp_info_ptr->schdUegInfo[numSchdUeg].numUeInGrp = num_ue_schd_in_grp;
                    resp_info_ptr->schdUegInfo[numSchdUeg].flags = 0x01;
                    // PRB allocation
                    resp_info_ptr->schdUegInfo[numSchdUeg].allocPrgStart = req_info_ptr->nPrbGrp/pDynDescr->num_subband*subbandIdx;
                    resp_info_ptr->schdUegInfo[numSchdUeg].allocPrgEnd = subbandIdx == (pDynDescr->num_subband-1) ? req_info_ptr->nPrbGrp : (req_info_ptr->nPrbGrp/pDynDescr->num_subband)*(subbandIdx + 1);
                    
                    numSchdUeg++;
                }   
            }
            __syncthreads();

            if (numSchdUeg >= req_info_ptr->nMaxUegPerCell || numUeSchd >= req_info_ptr->nMaxUeSchdPerCellTTI) {break;}
        }

        if (threadIdx.x == 0) {
            resp_info_ptr->numSchdUeg = numSchdUeg;
        }
    }

    /// CPU version functions of the CUDA kernels for debugging purposes
    void muUePairChanCorrKernel_memSharing_cpu(muUePairChanCorrDynDescr* pDynDescr)
    {
        for (uint16_t cellId = 0; cellId < pDynDescr->num_cell; cellId++) {
            cumac_muUeGrp_req_info_t* req_info_ptr = (cumac_muUeGrp_req_info_t*) (pDynDescr->task_in_buf + cellId*pDynDescr->task_in_buf_len_per_cell);
            
            cumac_muUeGrp_req_srs_info_msh_t* srsInfo_msh_ptr = (cumac_muUeGrp_req_srs_info_msh_t*) (req_info_ptr->payload);
            
            cumac_muUeGrp_req_ue_info_t* ueInfo_ptr = (cumac_muUeGrp_req_ue_info_t*) (srsInfo_msh_ptr + req_info_ptr->numSrsInfo);

            __half2* chanEst_main_buf_ptr = (__half2*) (pDynDescr->srs_chan_est_buf + sizeof(__half2)*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT*pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband*MAX_NUM_SRS_UE_PER_CELL*cellId);

            for (uint16_t prgIdx = 0; prgIdx < pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband; prgIdx++) {
                uint16_t subbandIdx = prgIdx/pDynDescr->num_prg_samp_per_subband;
                uint16_t prgIdx_in_subband = prgIdx - subbandIdx*pDynDescr->num_prg_samp_per_subband;
                uint16_t real_prg_idx = pDynDescr->num_prg/pDynDescr->num_subband*subbandIdx + (prgIdx_in_subband+1)*((pDynDescr->num_prg/pDynDescr->num_subband-1)/(pDynDescr->num_prg_samp_per_subband+1));
                
                float* gpu_srsChanOrt_ptr = pDynDescr->chan_orth_mat_buf + (cellId*pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband + prgIdx)*MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT*(MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2;
                
                for (uint16_t srs_info_idx = 0; srs_info_idx < req_info_ptr->numSrsInfo; srs_info_idx++) {
                    uint16_t num_ue_ant_port = srsInfo_msh_ptr[srs_info_idx].nUeAnt;

                    float snr_f;
                    std::memcpy(&snr_f, &(srsInfo_msh_ptr[srs_info_idx].srsWbSnr), sizeof(float));
                    pDynDescr->srs_snr_buf[cellId*MAX_NUM_SRS_UE_PER_CELL + srsInfo_msh_ptr[srs_info_idx].id] = snr_f;

                    __half2* srsChanEst_srs_info = pDynDescr->cubb_srs_gpu_buf + srsInfo_msh_ptr[srs_info_idx].realBuffIdx * pDynDescr->num_prg * pDynDescr->num_bs_ant_port * MAX_NUM_UE_ANT_PORT;

                    for (uint16_t  srs_info_ant_port = 0; srs_info_ant_port < num_ue_ant_port; srs_info_ant_port++) {
                        cuComplex srs_info_chanEst[MAX_NUM_BS_ANT_PORT];
                        for (uint16_t idx = 0; idx < pDynDescr->num_bs_ant_port; idx++) {
                            srs_info_chanEst[idx] = __half22float2(srsChanEst_srs_info[real_prg_idx*num_ue_ant_port*pDynDescr->num_bs_ant_port + srs_info_ant_port*pDynDescr->num_bs_ant_port + idx]);
                        }

                        uint16_t row_idx = srsInfo_msh_ptr[srs_info_idx].id*MAX_NUM_UE_ANT_PORT + srs_info_ant_port;

                        for (uint16_t uIdx = 0; uIdx < req_info_ptr->numUeInfo; uIdx++) {
                            uint16_t num_ue_ant_port_real_uIdx = ueInfo_ptr[uIdx].nUeAnt;
                            uint8_t flags_ue_info = ueInfo_ptr[uIdx].flags;
                            for (uint16_t tx_port_idx = 0; tx_port_idx < num_ue_ant_port_real_uIdx; tx_port_idx++) {
                                cuComplex innerProduct = make_cuComplex(0.0f, 0.0f);
                                float corrVal = -1.0f;

                                if ((flags_ue_info & 0x01) > 0 && (flags_ue_info & 0x04) > 0) { // is a valid UE info and SRS chanEst available
                                    uint16_t real_ue_id = ueInfo_ptr[uIdx].id;

                                    if ((flags_ue_info & 0x08) > 0) { // has updated SRS info in the current slot
                                        uint16_t srsInfoIdx = ueInfo_ptr[uIdx].srsInfoIdx;

                                        __half2* srsChanEst_srsInfoIdx = pDynDescr->cubb_srs_gpu_buf + srsInfo_msh_ptr[srsInfoIdx].realBuffIdx * pDynDescr->num_prg * pDynDescr->num_bs_ant_port * MAX_NUM_UE_ANT_PORT;

                                        for (int idx = 0; idx < pDynDescr->num_bs_ant_port; idx++) {
                                            __half2 tmp1 = srsChanEst_srsInfoIdx[real_prg_idx*num_ue_ant_port_real_uIdx*pDynDescr->num_bs_ant_port + tx_port_idx*pDynDescr->num_bs_ant_port + idx];
                                            cuComplex tmp1_complex = __half22float2(tmp1);

                                            innerProduct.x += srs_info_chanEst[idx].x*tmp1_complex.x + srs_info_chanEst[idx].y*tmp1_complex.y;
                                            innerProduct.y += srs_info_chanEst[idx].x*tmp1_complex.y - srs_info_chanEst[idx].y*tmp1_complex.x;
    
                                            chanEst_main_buf_ptr[real_ue_id*pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT + prgIdx*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT + tx_port_idx*pDynDescr->num_bs_ant_port + idx] = tmp1;
                                        }
                                    } else {    
                                        for (int idx = 0; idx < pDynDescr->num_bs_ant_port; idx++) {
                                            __half2 tmp1 = chanEst_main_buf_ptr[real_ue_id*pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT + prgIdx*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT + tx_port_idx*pDynDescr->num_bs_ant_port + idx];
                                            cuComplex tmp1_complex = __half22float2(tmp1);

                                            innerProduct.x += srs_info_chanEst[idx].x*tmp1_complex.x + srs_info_chanEst[idx].y*tmp1_complex.y;
                                            innerProduct.y += srs_info_chanEst[idx].x*tmp1_complex.y - srs_info_chanEst[idx].y*tmp1_complex.x;
                                        }
                                    }
    
                                    uint16_t col_idx = real_ue_id*MAX_NUM_UE_ANT_PORT + tx_port_idx;

                                    corrVal = sqrt(innerProduct.x*innerProduct.x + innerProduct.y*innerProduct.y);
                
                                    if (row_idx >= col_idx) {
                                        gpu_srsChanOrt_ptr[row_idx * (row_idx + 1) / 2 + col_idx] = corrVal;
                                    } else {
                                        gpu_srsChanOrt_ptr[col_idx * (col_idx + 1) / 2 + row_idx] = corrVal;
                                    }    
                                } 
                            }
                        }
                    }
                }
            }
        }
    }

    void muUePairChanCorrKernel_cpu(muUePairChanCorrDynDescr* pDynDescr)
    {
        for (uint16_t cellId = 0; cellId < pDynDescr->num_cell; cellId++) {
            cumac_muUeGrp_req_info_t* req_info_ptr = (cumac_muUeGrp_req_info_t*) (pDynDescr->task_in_buf + cellId*pDynDescr->task_in_buf_len_per_cell);

            cumac_muUeGrp_req_srs_info_t* srsInfo_ptr = (cumac_muUeGrp_req_srs_info_t*) (req_info_ptr->payload);

            cumac_muUeGrp_req_ue_info_t* ueInfo_ptr = (cumac_muUeGrp_req_ue_info_t*) (srsInfo_ptr + req_info_ptr->numSrsInfo);

            __half2* chanEst_main_buf_ptr = (__half2*) (pDynDescr->srs_chan_est_buf + sizeof(__half2)*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT*pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband*MAX_NUM_SRS_UE_PER_CELL*cellId);

            for (uint16_t prgIdx = 0; prgIdx < pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband; prgIdx++) {
                float* gpu_srsChanOrt_ptr = pDynDescr->chan_orth_mat_buf + (cellId*pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband + prgIdx)*MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT*(MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2;
                
                for (uint16_t srs_info_idx = 0; srs_info_idx < req_info_ptr->numSrsInfo; srs_info_idx++) {
                    uint16_t num_ue_ant_port = srsInfo_ptr[srs_info_idx].nUeAnt;

                    float snr_f;
                    std::memcpy(&snr_f, &(srsInfo_ptr[srs_info_idx].srsWbSnr), sizeof(float));
                    pDynDescr->srs_snr_buf[cellId*MAX_NUM_SRS_UE_PER_CELL + srsInfo_ptr[srs_info_idx].id] = snr_f;

                    for (uint16_t  srs_info_ant_port = 0; srs_info_ant_port < num_ue_ant_port; srs_info_ant_port++) {
                        cuComplex srs_info_chanEst[MAX_NUM_BS_ANT_PORT];
                        for (uint16_t idx = 0; idx < pDynDescr->num_bs_ant_port; idx++) {
                            srs_info_chanEst[idx] = __half22float2(srsInfo_ptr[srs_info_idx].srsChanEst[prgIdx*num_ue_ant_port*pDynDescr->num_bs_ant_port + srs_info_ant_port*pDynDescr->num_bs_ant_port + idx]);
                        }

                        uint16_t row_idx = srsInfo_ptr[srs_info_idx].id*MAX_NUM_UE_ANT_PORT + srs_info_ant_port;
                        
                        for (uint16_t uIdx = 0; uIdx < req_info_ptr->numUeInfo; uIdx++) {
                            uint16_t num_ue_ant_port_real_uIdx = ueInfo_ptr[uIdx].nUeAnt;
                            uint8_t flags_ue_info = ueInfo_ptr[uIdx].flags;
                            for (uint16_t tx_port_idx = 0; tx_port_idx < num_ue_ant_port_real_uIdx; tx_port_idx++) {
                                cuComplex innerProduct = make_cuComplex(0.0f, 0.0f);
                                float corrVal = -1.0f;

                                if ((flags_ue_info & 0x01) > 0 && (flags_ue_info & 0x04) > 0) { // is a valid UE info and SRS chanEst available
                                    uint16_t real_ue_id = ueInfo_ptr[uIdx].id;

                                    if ((flags_ue_info & 0x08) > 0) { // has updated SRS info in the current slot
                                        uint16_t srsInfoIdx = ueInfo_ptr[uIdx].srsInfoIdx;

                                        for (int idx = 0; idx < pDynDescr->num_bs_ant_port; idx++) {
                                            __half2 tmp1 = srsInfo_ptr[srsInfoIdx].srsChanEst[prgIdx*num_ue_ant_port_real_uIdx*pDynDescr->num_bs_ant_port + tx_port_idx*pDynDescr->num_bs_ant_port + idx];
                                            cuComplex tmp1_complex = __half22float2(tmp1);

                                            innerProduct.x += srs_info_chanEst[idx].x*tmp1_complex.x + srs_info_chanEst[idx].y*tmp1_complex.y;
                                            innerProduct.y += srs_info_chanEst[idx].x*tmp1_complex.y - srs_info_chanEst[idx].y*tmp1_complex.x;
    
                                            chanEst_main_buf_ptr[real_ue_id*pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT + prgIdx*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT + tx_port_idx*pDynDescr->num_bs_ant_port + idx] = tmp1;
                                        }
                                    } else {    
                                        for (int idx = 0; idx < pDynDescr->num_bs_ant_port; idx++) {
                                            __half2 tmp1 = chanEst_main_buf_ptr[real_ue_id*pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT + prgIdx*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT + tx_port_idx*pDynDescr->num_bs_ant_port + idx];
                                            cuComplex tmp1_complex = __half22float2(tmp1);

                                            innerProduct.x += srs_info_chanEst[idx].x*tmp1_complex.x + srs_info_chanEst[idx].y*tmp1_complex.y;
                                            innerProduct.y += srs_info_chanEst[idx].x*tmp1_complex.y - srs_info_chanEst[idx].y*tmp1_complex.x;
                                        }
                                    }
    
                                    uint16_t col_idx = real_ue_id*MAX_NUM_UE_ANT_PORT + tx_port_idx;

                                    corrVal = sqrt(innerProduct.x*innerProduct.x + innerProduct.y*innerProduct.y);
                
                                    if (row_idx >= col_idx) {
                                        gpu_srsChanOrt_ptr[row_idx * (row_idx + 1) / 2 + col_idx] = corrVal;
                                    } else {
                                        gpu_srsChanOrt_ptr[col_idx * (col_idx + 1) / 2 + row_idx] = corrVal;
                                    }    
                                } 
                            }
                        }
                    }
                }
            }
        }
    }

    void muUePairAlgKernel_memSharing_cpu(muUePairAlgDynDescr* pDynDescr)
    {
        for (uint16_t cellId = 0; cellId < pDynDescr->num_cell; cellId++) {
            cumac_muUeGrp_req_info_t*           req_info_ptr            = (cumac_muUeGrp_req_info_t*) (pDynDescr->task_in_buf + cellId*pDynDescr->task_in_buf_len_per_cell);
            cumac_muUeGrp_req_srs_info_msh_t*   srsInfo_msh_ptr         = (cumac_muUeGrp_req_srs_info_msh_t*) (req_info_ptr->payload);
            float*                              gpu_srsChanOrt_cell_ptr = pDynDescr->chan_orth_mat_buf + 
                                                                          cellId*pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT*
                                                                          (MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2;
            cumac_muUeGrp_resp_info_t*          resp_info_ptr           = (cumac_muUeGrp_resp_info_t*) (pDynDescr->task_out_buf + cellId*pDynDescr->task_out_buf_len_per_cell);
        
            cumac_muUeGrp_req_ue_info_t*        ueInfo_ptr              = (cumac_muUeGrp_req_ue_info_t*) (srsInfo_msh_ptr + req_info_ptr->numSrsInfo);
        
            std::vector<pfMetricUePair> pf;

            uint16_t ueRnti[MAX_NUM_SRS_UE_PER_CELL];
            uint16_t muMimoInd[MAX_NUM_SRS_UE_PER_CELL];
            uint16_t nUeAnt[MAX_NUM_SRS_UE_PER_CELL];

            for (size_t i = 0; i < 512; i++) {
                pfMetricUePair pfTemp;
                pfTemp.first = -1.0;
                pfTemp.second = 0xFFFF;
                pf.push_back(pfTemp);
            }

            float betaCoeff_f;
            std::memcpy(&betaCoeff_f, &(req_info_ptr->betaCoeff), sizeof(float));
            float muCoeff_f;
            std::memcpy(&muCoeff_f, &(req_info_ptr->muCoeff), sizeof(float));

            for (size_t i = 0; i < req_info_ptr->numUeInfo; i++) {
                uint8_t flags   = ueInfo_ptr[i].flags;
                uint16_t ue_id  = ueInfo_ptr[i].id;
                nUeAnt[ue_id]   = ueInfo_ptr[i].nUeAnt;
                ueRnti[ue_id]   = ueInfo_ptr[i].rnti;
                if ((flags & 0x01) > 0 && ueInfo_ptr[i].bufferSize > 0) { // valid UE info
                    float currRate_f;
                    std::memcpy(&currRate_f, &(ueInfo_ptr[i].currRate), sizeof(float));
                    float avgRate_f;
                    std::memcpy(&avgRate_f, &(ueInfo_ptr[i].avgRate), sizeof(float));
                    pf[i].first = powf(currRate_f, betaCoeff_f)/avgRate_f;
                    pf[i].second = ue_id;

                    if ((flags & 0x02) > 0) { // new TX indication
                        if ((flags & 0x04) > 0) { // SRS chanEst available
                            float srsSnrThr_f;
                            std::memcpy(&srsSnrThr_f, &(req_info_ptr->srsSnrThr), sizeof(float));
                            if (pDynDescr->srs_snr_buf[cellId*MAX_NUM_SRS_UE_PER_CELL + ue_id] >= srsSnrThr_f) {
                                muMimoInd[ue_id] = 1;
                                pf[i].first *= muCoeff_f;
                            } else {
                                muMimoInd[ue_id] = 0;
                            }
                        } else {
                            muMimoInd[ue_id] = 0;
                        }
                    } else { // re-TX
                        muMimoInd[ue_id] = 0;
                    }
                } else { // invalid UE info
                    muMimoInd[ue_id] = 0;
                }
            }

            // sorting
            std::sort(pf.begin(), pf.end(), [](pfMetricUePair a, pfMetricUePair b)
                                  {
                                      return (a.first > b.first) || (a.first == b.first && a.second < b.second);
                                  });

            // load correlation values from global memory to shared memory
            uint8_t orth_ind[MAX_NUM_UE_FOR_GRP_PER_CELL*MAX_NUM_UE_ANT_PORT*(MAX_NUM_UE_FOR_GRP_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2];
            int totNumElem = req_info_ptr->numUeForGrpPerCell*MAX_NUM_UE_ANT_PORT*(req_info_ptr->numUeForGrpPerCell*MAX_NUM_UE_ANT_PORT+1)/2;

            uint16_t schdUeIdPerLayerInGrp[MAX_NUM_LAYER_PER_GRP];
            uint8_t  schdAntIdxPerLayerInGrp[MAX_NUM_LAYER_PER_GRP];

            int numSchdUeg = 0;
            int numUeSchd = 0;

            for (uint32_t subbandIdx = 0; subbandIdx < pDynDescr->num_subband; subbandIdx++) {
                for (uint32_t idx = 0; idx < totNumElem; idx++) {
                    orth_ind[idx] = 0xFF;

                    uint32_t row_idx = 0;
                    while (row_idx*(row_idx+1)/2 <= idx) {row_idx++;}
                    row_idx--;
                    uint32_t col_idx = idx-row_idx*(row_idx+1)/2;

                    uint32_t ue_idx_row = row_idx/MAX_NUM_UE_ANT_PORT;
                    uint32_t ue_idx_col = col_idx/MAX_NUM_UE_ANT_PORT;
                    uint32_t ant_port_row = row_idx - ue_idx_row*MAX_NUM_UE_ANT_PORT;
                    uint32_t ant_port_col = col_idx - ue_idx_col*MAX_NUM_UE_ANT_PORT;
                    uint32_t ue_id_row = pf[ue_idx_row].second;
                    uint32_t ue_id_col = pf[ue_idx_col].second;

                    if (ue_id_row != 0xFFFF && ue_id_col != 0xFFFF && ant_port_row < nUeAnt[ue_id_row] && ant_port_col < nUeAnt[ue_id_col]) {
                        uint32_t row_idx_main = ue_id_row*MAX_NUM_UE_ANT_PORT + ant_port_row;
                        uint32_t col_idx_main = ue_id_col*MAX_NUM_UE_ANT_PORT + ant_port_col;

                        float corrValues;

                        for (uint32_t prgIdx = 0; prgIdx < pDynDescr->num_prg_samp_per_subband; prgIdx++) {
                            float* gpu_srsChanOrt_ptr = gpu_srsChanOrt_cell_ptr + 
                                                        (subbandIdx*pDynDescr->num_prg_samp_per_subband + prgIdx)*
                                                        MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT*(MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2;

                            if (row_idx_main >= col_idx_main) {
                                corrValues = gpu_srsChanOrt_ptr[row_idx_main*(row_idx_main+1)/2 + col_idx_main];
                            } else {
                                corrValues = gpu_srsChanOrt_ptr[col_idx_main*(col_idx_main+1)/2 + row_idx_main];
                            }
                            corrValues /= sqrt(gpu_srsChanOrt_ptr[row_idx_main*(row_idx_main+1)/2 + row_idx_main]*gpu_srsChanOrt_ptr[col_idx_main*(col_idx_main+1)/2 + col_idx_main]);

                            float chanCorrThr_f;
                            std::memcpy(&chanCorrThr_f, &(req_info_ptr->chanCorrThr), sizeof(float));
                            if (corrValues == 1.0f) {
                                orth_ind[idx] = 1;
                            } else if (corrValues > chanCorrThr_f) {
                                orth_ind[idx] = 0;
                                break;
                            } else {
                                orth_ind[idx] = 1;
                            }
                        }
                    }
                }

                for (size_t i = 0; i < MAX_NUM_LAYER_PER_GRP; i++) {
                    schdUeIdPerLayerInGrp[i] = 0xFFFF;
                    schdAntIdxPerLayerInGrp[i] = 0xFF;
                }
                
                uint16_t num_ue_schd_in_grp = 0;
                uint16_t num_layer_schd_in_grp = 0;

                for (int uIdx = 0; uIdx < req_info_ptr->numUeForGrpPerCell; uIdx++) {
                    uint16_t ue_id = pf[uIdx].second;
                    if (ue_id != 0xFFFF) {
                        if (num_ue_schd_in_grp > 0 && muMimoInd[ue_id] == 0) { // not the first UE in the UEG and not feasible for MU-MIMO
                            continue;
                        }

                        uint8_t layerSel = 0x00;
                        uint8_t nLayerSchdUe = 0;
                        for (int aIdx = 0; aIdx < nUeAnt[ue_id]; aIdx++) {
                            bool orthogonal = true;
                            for (int lIdx = 0; lIdx < num_layer_schd_in_grp; lIdx++) { // go through the scheduled layers in the UEG
                                uint32_t row_idx = schdUeIdPerLayerInGrp[lIdx]*MAX_NUM_UE_ANT_PORT + schdAntIdxPerLayerInGrp[lIdx];
                                uint32_t col_idx = uIdx*MAX_NUM_UE_ANT_PORT + aIdx;
                                
                                uint8_t ifOrtho;
                                if (row_idx >= col_idx) {
                                    ifOrtho = orth_ind[row_idx*(row_idx+1)/2 + col_idx];
                                } else {
                                    ifOrtho = orth_ind[col_idx*(col_idx+1)/2 + row_idx];
                                }
                                if (ifOrtho != 1) {
                                    orthogonal = false;
                                    break;
                                }
                            }

                            if (orthogonal) { // if the UE layer is orthogonal to the scheduled layers in the UEG
                                layerSel |= 0x01 << aIdx;

                                schdUeIdPerLayerInGrp[num_layer_schd_in_grp] = uIdx;
                                schdAntIdxPerLayerInGrp[num_layer_schd_in_grp] = aIdx;
                                num_layer_schd_in_grp++;
                                nLayerSchdUe++;

                                if (muMimoInd[ue_id] == 0) { // SU-MIMO UE (first in group)
                                    if (nLayerSchdUe >= req_info_ptr->nMaxLayerPerUeSu) {break;}
                                } else { // MU-MIMO UE
                                    if (nLayerSchdUe >= req_info_ptr->nMaxLayerPerUeMu) {break;}
                                }

                                if (num_layer_schd_in_grp >= req_info_ptr->nMaxLayerPerGrp) {break;}
                            }
                        }

                        if (layerSel != 0x00) { // layer selection found
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].layerSel = layerSel;
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].rnti = ueRnti[ue_id];
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].id = ue_id;
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].ueOrderInGrp = num_ue_schd_in_grp;
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].nSCID = 0xFF;
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].flags = 0x01;

                            pf[uIdx].second = 0xFFFF;

                            if (muMimoInd[ue_id] == 0) { // the first UE in the UEG and not feasible for MU-MIMO
                                num_ue_schd_in_grp++;
                                numUeSchd++;
                                break;
                            } else {
                                resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].flags |= 0x02;
                                num_ue_schd_in_grp++;
                                numUeSchd++;
                            }

                            if (num_ue_schd_in_grp >= req_info_ptr->nMaxUePerGrp || numUeSchd >= req_info_ptr->nMaxUeSchdPerCellTTI) {break;}
                        }
                    }

                    if (num_layer_schd_in_grp >= req_info_ptr->nMaxLayerPerGrp) {break;}
                }

                if (num_ue_schd_in_grp > 0) {
                    resp_info_ptr->schdUegInfo[numSchdUeg].numUeInGrp = num_ue_schd_in_grp;
                    resp_info_ptr->schdUegInfo[numSchdUeg].flags = 0x01;
                    // PRB allocation
                    resp_info_ptr->schdUegInfo[numSchdUeg].allocPrgStart = req_info_ptr->nPrbGrp/pDynDescr->num_subband*subbandIdx;
                    resp_info_ptr->schdUegInfo[numSchdUeg].allocPrgEnd = subbandIdx == (pDynDescr->num_subband-1) ? req_info_ptr->nPrbGrp : (req_info_ptr->nPrbGrp/pDynDescr->num_subband)*(subbandIdx + 1);
                    
                    numSchdUeg++;
                } 
                
                if (numSchdUeg >= req_info_ptr->nMaxUegPerCell || numUeSchd >= req_info_ptr->nMaxUeSchdPerCellTTI) {break;}
            }

            resp_info_ptr->numSchdUeg = numSchdUeg;
        }
    }

    void muUePairAlgKernel_cpu(muUePairAlgDynDescr* pDynDescr)
    {
        for (uint16_t cellId = 0; cellId < pDynDescr->num_cell; cellId++) {
            cumac_muUeGrp_req_info_t*           req_info_ptr            = (cumac_muUeGrp_req_info_t*) (pDynDescr->task_in_buf + cellId*pDynDescr->task_in_buf_len_per_cell);
            cumac_muUeGrp_req_srs_info_t*       srsInfo_ptr             = (cumac_muUeGrp_req_srs_info_t*) (req_info_ptr->payload);
            float*                              gpu_srsChanOrt_cell_ptr = pDynDescr->chan_orth_mat_buf + 
                                                                          cellId*pDynDescr->num_subband*pDynDescr->num_prg_samp_per_subband*pDynDescr->num_bs_ant_port*MAX_NUM_UE_ANT_PORT*
                                                                          (MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2;
            cumac_muUeGrp_resp_info_t*          resp_info_ptr           = (cumac_muUeGrp_resp_info_t*) (pDynDescr->task_out_buf + cellId*pDynDescr->task_out_buf_len_per_cell);
        
            cumac_muUeGrp_req_ue_info_t*        ueInfo_ptr              = (cumac_muUeGrp_req_ue_info_t*) (srsInfo_ptr + req_info_ptr->numSrsInfo);
        
            std::vector<pfMetricUePair> pf;

            uint16_t ueRnti[MAX_NUM_SRS_UE_PER_CELL];
            uint16_t muMimoInd[MAX_NUM_SRS_UE_PER_CELL];
            uint16_t nUeAnt[MAX_NUM_SRS_UE_PER_CELL];

            for (size_t i = 0; i < 512; i++) {
                pfMetricUePair pfTemp;
                pfTemp.first = -1.0;
                pfTemp.second = 0xFFFF;
                pf.push_back(pfTemp);
            }

            float betaCoeff_f;
            std::memcpy(&betaCoeff_f, &(req_info_ptr->betaCoeff), sizeof(float));
            float muCoeff_f;
            std::memcpy(&muCoeff_f, &(req_info_ptr->muCoeff), sizeof(float));

            for (size_t i = 0; i < req_info_ptr->numUeInfo; i++) {
                uint8_t flags   = ueInfo_ptr[i].flags;
                uint16_t ue_id  = ueInfo_ptr[i].id;
                nUeAnt[ue_id]   = ueInfo_ptr[i].nUeAnt;
                ueRnti[ue_id]   = ueInfo_ptr[i].rnti;
                if ((flags & 0x01) > 0 && ueInfo_ptr[i].bufferSize > 0) { // valid UE info
                    float currRate_f;
                    std::memcpy(&currRate_f, &(ueInfo_ptr[i].currRate), sizeof(float));
                    float avgRate_f;
                    std::memcpy(&avgRate_f, &(ueInfo_ptr[i].avgRate), sizeof(float));
                    pf[i].first = powf(currRate_f, betaCoeff_f)/avgRate_f;
                    pf[i].second = ue_id;

                    if ((flags & 0x02) > 0) { // new TX indication
                        if ((flags & 0x04) > 0) { // SRS chanEst available
                            float srsSnrThr_f;
                            std::memcpy(&srsSnrThr_f, &(req_info_ptr->srsSnrThr), sizeof(float));
                            if (pDynDescr->srs_snr_buf[cellId*MAX_NUM_SRS_UE_PER_CELL + ue_id] >= srsSnrThr_f) {
                                muMimoInd[ue_id] = 1;
                                pf[i].first *= muCoeff_f;
                            } else {
                                muMimoInd[ue_id] = 0;
                            }
                        } else {
                            muMimoInd[ue_id] = 0;
                        }
                    } else { // re-TX
                        muMimoInd[ue_id] = 0;
                    }
                } else { // invalid UE info
                    muMimoInd[ue_id] = 0;
                }
            }

            // sorting
            std::sort(pf.begin(), pf.end(), [](pfMetricUePair a, pfMetricUePair b)
                                  {
                                      return (a.first > b.first) || (a.first == b.first && a.second < b.second);
                                  });

            // load correlation values from global memory to shared memory
            uint8_t orth_ind[MAX_NUM_UE_FOR_GRP_PER_CELL*MAX_NUM_UE_ANT_PORT*(MAX_NUM_UE_FOR_GRP_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2];
            int totNumElem = req_info_ptr->numUeForGrpPerCell*MAX_NUM_UE_ANT_PORT*(req_info_ptr->numUeForGrpPerCell*MAX_NUM_UE_ANT_PORT+1)/2;

            uint16_t schdUeIdPerLayerInGrp[MAX_NUM_LAYER_PER_GRP];
            uint8_t  schdAntIdxPerLayerInGrp[MAX_NUM_LAYER_PER_GRP];

            int numSchdUeg = 0;
            int numUeSchd = 0;

            for (uint32_t subbandIdx = 0; subbandIdx < pDynDescr->num_subband; subbandIdx++) {
                for (uint32_t idx = 0; idx < totNumElem; idx++) {
                    orth_ind[idx] = 0xFF;

                    uint32_t row_idx = 0;
                    while (row_idx*(row_idx+1)/2 <= idx) {row_idx++;}
                    row_idx--;
                    uint32_t col_idx = idx-row_idx*(row_idx+1)/2;

                    uint32_t ue_idx_row = row_idx/MAX_NUM_UE_ANT_PORT;
                    uint32_t ue_idx_col = col_idx/MAX_NUM_UE_ANT_PORT;
                    uint32_t ant_port_row = row_idx - ue_idx_row*MAX_NUM_UE_ANT_PORT;
                    uint32_t ant_port_col = col_idx - ue_idx_col*MAX_NUM_UE_ANT_PORT;
                    uint32_t ue_id_row = pf[ue_idx_row].second;
                    uint32_t ue_id_col = pf[ue_idx_col].second;

                    if (ue_id_row != 0xFFFF && ue_id_col != 0xFFFF && ant_port_row < nUeAnt[ue_id_row] && ant_port_col < nUeAnt[ue_id_col]) {
                        uint32_t row_idx_main = ue_id_row*MAX_NUM_UE_ANT_PORT + ant_port_row;
                        uint32_t col_idx_main = ue_id_col*MAX_NUM_UE_ANT_PORT + ant_port_col;

                        float corrValues;

                        for (uint32_t prgIdx = 0; prgIdx < pDynDescr->num_prg_samp_per_subband; prgIdx++) {
                            float* gpu_srsChanOrt_ptr = gpu_srsChanOrt_cell_ptr + 
                                                        (subbandIdx*pDynDescr->num_prg_samp_per_subband + prgIdx)*
                                                        MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT*(MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2;

                            if (row_idx_main >= col_idx_main) {
                                corrValues = gpu_srsChanOrt_ptr[row_idx_main*(row_idx_main+1)/2 + col_idx_main];
                            } else {
                                corrValues = gpu_srsChanOrt_ptr[col_idx_main*(col_idx_main+1)/2 + row_idx_main];
                            }
                            corrValues /= sqrt(gpu_srsChanOrt_ptr[row_idx_main*(row_idx_main+1)/2 + row_idx_main]*gpu_srsChanOrt_ptr[col_idx_main*(col_idx_main+1)/2 + col_idx_main]);

                            float chanCorrThr_f;
                            std::memcpy(&chanCorrThr_f, &(req_info_ptr->chanCorrThr), sizeof(float));
                            if (corrValues == 1.0f) {
                                orth_ind[idx] = 1;
                            } else if (corrValues > chanCorrThr_f) {
                                orth_ind[idx] = 0;
                                break;
                            } else {
                                orth_ind[idx] = 1;
                            }
                        }
                    }
                }

                for (size_t i = 0; i < MAX_NUM_LAYER_PER_GRP; i++) {
                    schdUeIdPerLayerInGrp[i] = 0xFFFF;
                    schdAntIdxPerLayerInGrp[i] = 0xFF;
                }
                
                uint16_t num_ue_schd_in_grp = 0;
                uint16_t num_layer_schd_in_grp = 0;

                for (int uIdx = 0; uIdx < req_info_ptr->numUeForGrpPerCell; uIdx++) {
                    uint16_t ue_id = pf[uIdx].second;
                    if (ue_id != 0xFFFF) {
                        if (num_ue_schd_in_grp > 0 && muMimoInd[ue_id] == 0) { // not the first UE in the UEG and not feasible for MU-MIMO
                            continue;
                        }

                        uint8_t layerSel = 0x00;
                        uint8_t nLayerSchdUe = 0;
                        for (int aIdx = 0; aIdx < nUeAnt[ue_id]; aIdx++) {
                            bool orthogonal = true;
                            for (int lIdx = 0; lIdx < num_layer_schd_in_grp; lIdx++) { // go through the scheduled layers in the UEG
                                uint32_t row_idx = schdUeIdPerLayerInGrp[lIdx]*MAX_NUM_UE_ANT_PORT + schdAntIdxPerLayerInGrp[lIdx];
                                uint32_t col_idx = uIdx*MAX_NUM_UE_ANT_PORT + aIdx;
                                
                                uint8_t ifOrtho;
                                if (row_idx >= col_idx) {
                                    ifOrtho = orth_ind[row_idx*(row_idx+1)/2 + col_idx];
                                } else {
                                    ifOrtho = orth_ind[col_idx*(col_idx+1)/2 + row_idx];
                                }
                                if (ifOrtho != 1) {
                                    orthogonal = false;
                                    break;
                                }
                            }

                            if (orthogonal) { // if the UE layer is orthogonal to the scheduled layers in the UEG
                                layerSel |= 0x01 << aIdx;

                                schdUeIdPerLayerInGrp[num_layer_schd_in_grp] = uIdx;
                                schdAntIdxPerLayerInGrp[num_layer_schd_in_grp] = aIdx;
                                num_layer_schd_in_grp++;
                                nLayerSchdUe++;

                                if (muMimoInd[ue_id] == 0) { // SU-MIMO UE (first in group)
                                    if (nLayerSchdUe >= req_info_ptr->nMaxLayerPerUeSu) {break;}
                                } else { // MU-MIMO UE
                                    if (nLayerSchdUe >= req_info_ptr->nMaxLayerPerUeMu) {break;}
                                }

                                if (num_layer_schd_in_grp >= req_info_ptr->nMaxLayerPerGrp) {break;}
                            }
                        }

                        if (layerSel != 0x00) { // layer selection found
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].layerSel = layerSel;
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].rnti = ueRnti[ue_id];
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].id = ue_id;
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].ueOrderInGrp = num_ue_schd_in_grp;
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].nSCID = 0xFF;
                            resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].flags = 0x01;

                            pf[uIdx].second = 0xFFFF;

                            if (muMimoInd[ue_id] == 0) { // the first UE in the UEG and not feasible for MU-MIMO
                                num_ue_schd_in_grp++;
                                numUeSchd++;
                                break;
                            } else {
                                resp_info_ptr->schdUegInfo[numSchdUeg].ueInfo[num_ue_schd_in_grp].flags |= 0x02;
                                num_ue_schd_in_grp++;
                                numUeSchd++;
                            }

                            if (num_ue_schd_in_grp >= req_info_ptr->nMaxUePerGrp || numUeSchd >= req_info_ptr->nMaxUeSchdPerCellTTI) {break;}
                        }
                    }

                    if (num_layer_schd_in_grp >= req_info_ptr->nMaxLayerPerGrp) {break;}
                }

                if (num_ue_schd_in_grp > 0) {
                    resp_info_ptr->schdUegInfo[numSchdUeg].numUeInGrp = num_ue_schd_in_grp;
                    resp_info_ptr->schdUegInfo[numSchdUeg].flags = 0x01;
                    // PRB allocation
                    resp_info_ptr->schdUegInfo[numSchdUeg].allocPrgStart = req_info_ptr->nPrbGrp/pDynDescr->num_subband*subbandIdx;
                    resp_info_ptr->schdUegInfo[numSchdUeg].allocPrgEnd = subbandIdx == (pDynDescr->num_subband-1) ? req_info_ptr->nPrbGrp : (req_info_ptr->nPrbGrp/pDynDescr->num_subband)*(subbandIdx + 1);
                    
                    numSchdUeg++;
                } 
                
                if (numSchdUeg >= req_info_ptr->nMaxUegPerCell || numUeSchd >= req_info_ptr->nMaxUeSchdPerCellTTI) {break;}
            }

            resp_info_ptr->numSchdUeg = numSchdUeg;
        }
    }
}

/*
//! backup CUDA kernel for channel correlation computation
__global__ void ueGrpKernel_64T64R_chanCorr_v1(uint8_t*    main_in_buf, 
                                               uint8_t*    task_in_buf, 
                                               float*      gpu_srsChanOrt,
                                               uint32_t    task_in_buf_len_per_cell,
                                               int         num_cell,
                                               int         num_bs_ant_port,
                                               int         num_subband,
                                               int         num_prg_samp_per_subband,
                                               int         num_blocks_per_cell,
                                               int         num_blocks_per_row_chanOrtMat)
{
    uint16_t cellId = blockIdx.x / num_blocks_per_cell;
    uint16_t row_idx_chanOrtMat = (blockIdx.x - cellId*num_blocks_per_cell) / num_blocks_per_row_chanOrtMat;
    uint16_t blockIdx_in_row_chanOrtMat = blockIdx.x - cellId*num_blocks_per_cell - row_idx_chanOrtMat*num_blocks_per_row_chanOrtMat;
    uint16_t srs_info_idx = row_idx_chanOrtMat / MAX_NUM_UE_ANT_PORT;
    uint16_t srs_info_ant_port = row_idx_chanOrtMat % MAX_NUM_UE_ANT_PORT;

    cumac_muUeGrp_req_info_t* req_info_ptr = (cumac_muUeGrp_req_info_t*) (task_in_buf + cellId*task_in_buf_len_per_cell);

    cumac_muUeGrp_req_srs_info_t* srsInfo_ptr = (cumac_muUeGrp_req_srs_info_t*) (req_info_ptr->payload);

    cumac_muUeGrp_req_ue_info_t* ueInfo_ptr = (cumac_muUeGrp_req_ue_info_t*) (srsInfo_ptr + req_info_ptr->numSrsInfo);

    if (srs_info_idx >= req_info_ptr->numSrsInfo) {
        return;
    }

    uint16_t num_ue_ant_port = srsInfo_ptr[srs_info_idx].nUeAnt;

    if (srs_info_ant_port >= num_ue_ant_port) {
        return;
    }

    uint16_t ue_info_per_block = req_info_ptr->numUeInfo/num_blocks_per_row_chanOrtMat;
    uint16_t first_ue_info_idx_in_block = ue_info_per_block*blockIdx_in_row_chanOrtMat;
    uint16_t last_ue_info_idx_in_block;
    if (blockIdx_in_row_chanOrtMat == num_blocks_per_row_chanOrtMat - 1) { // last block
        last_ue_info_idx_in_block = req_info_ptr->numUeInfo - 1;
    } else {
        last_ue_info_idx_in_block = first_ue_info_idx_in_block + ue_info_per_block - 1;
    }

    uint16_t portIdx = threadIdx.x/(num_bs_ant_port/2);
    uint16_t tx_port_idx = portIdx%MAX_NUM_UE_ANT_PORT;
    uint16_t rx_port_idx = threadIdx.x%(num_bs_ant_port/2);
    uint16_t num_rnd = ceil(static_cast<float>((last_ue_info_idx_in_block - first_ue_info_idx_in_block + 1)*MAX_NUM_UE_ANT_PORT)/(blockDim.x/(num_bs_ant_port/2)));

    __shared__ cuComplex    srs_info_chanEst[MAX_NUM_BS_ANT_PORT];
    __shared__ cuComplex    ue_info_chanEst[2048];
    __shared__ uint16_t     corrIdx[MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT];
    __shared__ float        corrValues[MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT];

    // update SRS SNRs
    if (srs_info_ant_port == 0 && blockIdx_in_row_chanOrtMat == 0 && threadIdx.x == 0) {
        float* srsSnr_main_buf_ptr = (float*) (main_in_buf + sizeof(__half2)*num_bs_ant_port*MAX_NUM_UE_ANT_PORT*num_subband*num_prg_samp_per_subband*MAX_NUM_SRS_UE_PER_CELL*num_cell + sizeof(float)*MAX_NUM_SRS_UE_PER_CELL*cellId);

        srsSnr_main_buf_ptr[srsInfo_ptr[srs_info_idx].id] = __uint_as_float(srsInfo_ptr[srs_info_idx].srsWbSnr);
    }

    __half2* chanEst_main_buf_ptr = (__half2*) (main_in_buf + sizeof(__half2)*num_bs_ant_port*MAX_NUM_UE_ANT_PORT*num_subband*num_prg_samp_per_subband*MAX_NUM_SRS_UE_PER_CELL*cellId);

    for (int prgIdx = 0; prgIdx < num_subband*num_prg_samp_per_subband; prgIdx++) {
        for (int idx = threadIdx.x; idx < MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT; idx += blockDim.x) {
            corrIdx[idx] = 0x0000FFFF;
        }
    
        float* gpu_srsChanOrt_ptr = gpu_srsChanOrt + (cellId*num_subband*num_prg_samp_per_subband + prgIdx)*MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT*(MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2;
            
        if (threadIdx.x < num_bs_ant_port) {
            srs_info_chanEst[threadIdx.x] = __half22float2(srsInfo_ptr[srs_info_idx].srsChanEst[prgIdx*num_ue_ant_port*num_bs_ant_port + srs_info_ant_port*num_bs_ant_port + threadIdx.x]);
        }
        
        for (int rIdx = 0; rIdx < num_rnd; rIdx++) {
            ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx] = make_cuComplex(0.0f, 0.0f);
            ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)] = make_cuComplex(0.0f, 0.0f);
    
            uint16_t real_uIdx_in_block = portIdx/MAX_NUM_UE_ANT_PORT + rIdx*(blockDim.x/(MAX_NUM_UE_ANT_PORT*num_bs_ant_port/2));
        
            uint16_t real_ue_id = 0xFFFF;
    
            if ((real_uIdx_in_block + first_ue_info_idx_in_block) <= last_ue_info_idx_in_block) {
                uint16_t num_ue_ant_port_real_uIdx = ueInfo_ptr[real_uIdx_in_block + first_ue_info_idx_in_block].nUeAnt;
    
                if (tx_port_idx < num_ue_ant_port_real_uIdx) {
                    uint8_t flags_ue_info = ueInfo_ptr[real_uIdx_in_block + first_ue_info_idx_in_block].flags;
    
                    if ((flags_ue_info & 0x04) > 0) { // SRS chanEst available
                        real_ue_id = ueInfo_ptr[real_uIdx_in_block + first_ue_info_idx_in_block].id;
                        
                        if ((flags_ue_info & 0x08) > 0) { // has updated SRS info in the current slot
                            uint16_t srsInfoIdx = ueInfo_ptr[real_uIdx_in_block + first_ue_info_idx_in_block].srsInfoIdx;
                            __half2 tmp1 = srsInfo_ptr[srsInfoIdx].srsChanEst[prgIdx*num_ue_ant_port_real_uIdx*num_bs_ant_port + tx_port_idx*num_bs_ant_port + rx_port_idx];
                            __half2 tmp2 = srsInfo_ptr[srsInfoIdx].srsChanEst[prgIdx*num_ue_ant_port_real_uIdx*num_bs_ant_port + tx_port_idx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)];
                            ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx] = __half22float2(tmp1);
                            ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)] = __half22float2(tmp2);
                            if (row_idx_chanOrtMat == 0) {
                                chanEst_main_buf_ptr[real_ue_id*num_subband*num_prg_samp_per_subband*num_bs_ant_port*MAX_NUM_UE_ANT_PORT + prgIdx*num_bs_ant_port*MAX_NUM_UE_ANT_PORT + tx_port_idx*num_bs_ant_port + rx_port_idx] = tmp1;
                                chanEst_main_buf_ptr[real_ue_id*num_subband*num_prg_samp_per_subband*num_bs_ant_port*MAX_NUM_UE_ANT_PORT + prgIdx*num_bs_ant_port*MAX_NUM_UE_ANT_PORT + tx_port_idx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)] = tmp2;
                            }
                        } else {    
                            ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx] = __half22float2(chanEst_main_buf_ptr[real_ue_id*num_subband*num_prg_samp_per_subband*num_bs_ant_port*MAX_NUM_UE_ANT_PORT + prgIdx*num_bs_ant_port*MAX_NUM_UE_ANT_PORT + tx_port_idx*num_bs_ant_port + rx_port_idx]);
                            ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)] = __half22float2(chanEst_main_buf_ptr[real_ue_id*num_subband*num_prg_samp_per_subband*num_bs_ant_port*MAX_NUM_UE_ANT_PORT + prgIdx*num_bs_ant_port*MAX_NUM_UE_ANT_PORT + tx_port_idx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)]);
                        }

                        cuComplex innerProduct;
                        innerProduct.x = srs_info_chanEst[rx_port_idx].x*ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx].x + srs_info_chanEst[rx_port_idx].y*ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx].y;
                        innerProduct.y = srs_info_chanEst[rx_port_idx].x*ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx].y - srs_info_chanEst[rx_port_idx].y*ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx].x;

                        ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx] = innerProduct;

                        innerProduct.x = srs_info_chanEst[rx_port_idx + (num_bs_ant_port/2)].x*ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)].x + srs_info_chanEst[rx_port_idx + (num_bs_ant_port/2)].y*ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)].y;
                        innerProduct.y = srs_info_chanEst[rx_port_idx + (num_bs_ant_port/2)].x*ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)].y - srs_info_chanEst[rx_port_idx + (num_bs_ant_port/2)].y*ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)].x;

                        ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)] = innerProduct;
                    }
                }
            }
            __syncthreads();
    
            // parallel reduction
            uint16_t h = num_bs_ant_port;
            uint16_t s = ceilf(h*0.5f);

            while(s > 1) {
                if(rx_port_idx < (h - s)) {
                    ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx].x += ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + s].x;
                    ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx].y += ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + s].y;
                }
                h = s; 
                s = ceilf(h*0.5f);
    
                __syncthreads();
            }
    
            if (rx_port_idx == 0) {
                ue_info_chanEst[portIdx*num_bs_ant_port].x += ue_info_chanEst[portIdx*num_bs_ant_port + 1].x;
                ue_info_chanEst[portIdx*num_bs_ant_port].y += ue_info_chanEst[portIdx*num_bs_ant_port + 1].y;
    
                if (real_ue_id != 0xFFFF) {
                    uint16_t col_idx = real_ue_id*MAX_NUM_UE_ANT_PORT + tx_port_idx;
                    uint16_t row_idx = srsInfo_ptr[srs_info_idx].id*MAX_NUM_UE_ANT_PORT + srs_info_ant_port;
    
                    if (row_idx >= col_idx) {
                        corrIdx[real_uIdx_in_block*MAX_NUM_UE_ANT_PORT + tx_port_idx] = row_idx * (row_idx + 1) / 2 + col_idx;
                    } else {
                        corrIdx[real_uIdx_in_block*MAX_NUM_UE_ANT_PORT + tx_port_idx] = col_idx * (col_idx + 1) / 2 + row_idx;
                    }
    
                    corrValues[real_uIdx_in_block*MAX_NUM_UE_ANT_PORT + tx_port_idx] = sqrt((ue_info_chanEst[portIdx*num_bs_ant_port].x*ue_info_chanEst[portIdx*num_bs_ant_port].x + ue_info_chanEst[portIdx*num_bs_ant_port].y*ue_info_chanEst[portIdx*num_bs_ant_port].y));
                }
            }
        }
        __syncthreads();
    
        for (int idx = threadIdx.x; idx < MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT; idx += blockDim.x) {
            if (corrIdx[idx] != 0x0000FFFF) {
                gpu_srsChanOrt_ptr[corrIdx[idx]] = corrValues[idx];
            }
        }
        __syncthreads();
    }
}
*/