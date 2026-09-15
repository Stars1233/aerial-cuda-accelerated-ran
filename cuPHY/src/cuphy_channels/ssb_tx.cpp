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

#include "ssb_tx.hpp"
#include "cuphy_internal.h"
#include <mutex>
#include <unordered_map>
#include <memory>

using namespace cuphy;

// File-scoped mutex system for HDF5 file access synchronization
namespace {
    // Global map to store per-filename mutexes for thread-safe HDF5 file access
    std::unordered_map<std::string, std::shared_ptr<std::mutex>> g_fileMutexMap;
    
    // Mutex to protect access to the global mutex map itself
    std::mutex g_mapMutex;
    
    /**
     * @brief Get or create a mutex for the specified filename
     * @param[in] filename: The HDF5 filename to get a mutex for
     * @return Shared pointer to the mutex for this filename
     */
    std::shared_ptr<std::mutex> getFileMutex(const std::string& filename) {
        std::lock_guard<std::mutex> mapLock(g_mapMutex);
        
        auto it = g_fileMutexMap.find(filename);
        if (it != g_fileMutexMap.end()) {
            return it->second;
        }
        
        // Create new mutex for this filename
        auto newMutex = std::make_shared<std::mutex>();
        g_fileMutexMap[filename] = newMutex;
        return newMutex;
    }
}

// Define the static counter
std::atomic<int> SsbTx::s_ssbIdCounter{0};

cuphyStatus_t CUPHYWINAPI cuphyCreateSsbTx(cuphySsbTxHndl_t* pSsbTxHndl, cuphySsbStatPrms_t const* pStatPrms)
{
    if((pSsbTxHndl == nullptr) || (pStatPrms == nullptr))
    {
        NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "cuphyCreateSsbTx called with cuphySsbTxHndl_t ptr ({:p}) or for cuphySsbStatPrms_t ptr ({:p}) nullptr.",
                   static_cast<void*>(pSsbTxHndl), const_cast<void*>(static_cast<void const*>(pStatPrms)));
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    return cuphy::tryCallableAndCatch([&]
    {
        if (pStatPrms->pDbgPrms != nullptr && pStatPrms->pDbgPrms->enableApiLogging) { 
            SsbTx::printStaticApiPrms<fmtlog::DBG>(pStatPrms);
        }
        SsbTx* new_pipeline = new SsbTx(pStatPrms);

        if(new_pipeline == nullptr)
        {
            return CUPHY_STATUS_ALLOC_FAILED;
        }
        *pSsbTxHndl = new_pipeline;

        return CUPHY_STATUS_SUCCESS;
    });
}

#if 0
const void* cuphyGetMemoryFootprintTrackerSsbTx(cuphySsbTxHndl_t ssbTxHndl)
{
    if(ssbTxHndl == nullptr)
    {
        return nullptr;
    }
    SsbTx* pipeline_ptr  = static_cast<SsbTx*>(ssbTxHndl);
    return pipeline_ptr->getMemoryTracker();
}
#endif

const void* SsbTx::getMemoryTracker()
{
    return &memory_footprint;
}


cuphyStatus_t CUPHYWINAPI cuphySetupSsbTx(cuphySsbTxHndl_t ssbTxHndl, cuphySsbDynPrms_t* pDynPrms)
{
    PUSH_RANGE("SSB_SETUP", 1);
    if((pDynPrms == nullptr) || (ssbTxHndl == nullptr))
    {
        NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "cuphySetupSsbTx called with nullptr for cuphySsbTxHndl_t ({:p}) or for cuphySsbDynPrms_t ptr ({:p}).",
                   static_cast<void*>(ssbTxHndl), static_cast<void*>(pDynPrms));
        POP_RANGE
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        SsbTx* pipeline_ptr  = static_cast<SsbTx*>(ssbTxHndl);
        cuphyStatus_t status = pipeline_ptr->setup(pDynPrms);
        POP_RANGE
        return status;
    }, CUPHY_STATUS_INVALID_ARGUMENT);
}

cuphyStatus_t CUPHYWINAPI cuphyRunSsbTx(cuphySsbTxHndl_t ssbTxHndl, uint64_t procModeBmsk /* not used */)
{
    if(ssbTxHndl == nullptr)
    {
        NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "cuphyRunSsbTx error: cuphySsbTxHndl_t is nullptr!");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    return cuphy::tryCallableAndCatch([&]
    {
        SsbTx* pipeline_ptr  = static_cast<SsbTx*>(ssbTxHndl);

        cuphyStatus_t status = pipeline_ptr->run(pipeline_ptr->dynamic_params->cuStream);
        return status;
    });
}

cuphyStatus_t CUPHYWINAPI cuphyDestroySsbTx(cuphySsbTxHndl_t ssbTxHndl)
{
    if(ssbTxHndl == nullptr)
    {
        NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "cuphyDestroySsbTx error: cuphySsbTxHndl_t is nullptr!");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    SsbTx* pipeline_ptr  = static_cast<SsbTx*>(ssbTxHndl);
    delete pipeline_ptr;
    return CUPHY_STATUS_SUCCESS;
}

void SsbTx::writeDbgBufSynch(cudaStream_t cuStream)
{
    if(m_outputPrms.debugOutputFlag)
    {
        NVLOGD_FMT(NVLOG_SSB, "Writing SSB debug data for {} cells to HDF5 file", 
                   dynamic_params ? dynamic_params->nCells : 0);
        
        // Get file-specific mutex for thread-safe HDF5 file access
        auto fileMutex = getFileMutex(m_outputPrms.outFileName);
        std::lock_guard<std::mutex> fileLock(*fileMutex);
        
        NVLOGD_FMT(NVLOG_SSB, "Acquired file lock for HDF5 file: {}", m_outputPrms.outFileName);
        
        try {
            // Create unique identifier for this H5 write operation using static counter
            int writeId = s_ssbIdCounter.fetch_add(1);
            std::string ssbId = "SSB_Call" + std::to_string(writeId);
            
            // Open H5 file for this write operation (now thread-safe with file-scoped mutex)
            hdf5hpp::hdf5_file outHdf5File;
            if (access(m_outputPrms.outFileName.c_str(), F_OK) == 0) {
                outHdf5File = hdf5hpp::hdf5_file::open(m_outputPrms.outFileName.c_str(), H5F_ACC_RDWR);
                NVLOGD_FMT(NVLOG_SSB, "Opened existing HDF5 file: {}", m_outputPrms.outFileName);
            } else {
                outHdf5File = hdf5hpp::hdf5_file::create(m_outputPrms.outFileName.c_str());
                NVLOGD_FMT(NVLOG_SSB, "Created new HDF5 file: {}", m_outputPrms.outFileName);
            }
            
            // Write static parameters (only if file is new or dataset doesn't exist)
            std::string static_params_name = ssbId + "_StaticParams_nMaxCellsPerSlot";
            if (!outHdf5File.is_valid_dataset(static_params_name.c_str())) {
                uint16_t nMaxCellsPerSlot_copy = static_params->nMaxCellsPerSlot;
                cuphy::write_HDF5_dataset(outHdf5File, static_params_name.c_str(), CUPHY_R_16U, 1, 
                                         const_cast<void*>(reinterpret_cast<const void*>(&nMaxCellsPerSlot_copy)));
            }
            
            if(dynamic_params != nullptr)
            {
                // Write basic dynamic parameters with unique names
                std::string cells_name = ssbId + "_nCells";
                std::string blocks_name = ssbId + "_nSSBlocks";
                std::string precoding_matrices_name = ssbId + "_nPrecodingMatrices";
                std::string proc_mode_name = ssbId + "_procModeBmsk";
                
                NVLOGD_FMT(NVLOG_SSB, "Writing SSB dynamic parameters: nCells={}, nSSBlocks={}, nPrecodingMatrices={}, procModeBmsk={}", 
                           dynamic_params->nCells, dynamic_params->nSSBlocks, dynamic_params->nPrecodingMatrices, dynamic_params->procModeBmsk);
                
                // Create non-const copies for writing
                uint32_t nCells_copy = dynamic_params->nCells;
                uint32_t nSSBlocks_copy = dynamic_params->nSSBlocks;
                uint32_t nPrecodingMatrices_copy = dynamic_params->nPrecodingMatrices;
                uint32_t procModeBmsk_copy = dynamic_params->procModeBmsk;
                
                cuphy::write_HDF5_dataset(outHdf5File, cells_name.c_str(), CUPHY_R_32U, 1, const_cast<void*>(reinterpret_cast<const void*>(&nCells_copy)));
                cuphy::write_HDF5_dataset(outHdf5File, blocks_name.c_str(), CUPHY_R_32U, 1, const_cast<void*>(reinterpret_cast<const void*>(&nSSBlocks_copy)));
                cuphy::write_HDF5_dataset(outHdf5File, precoding_matrices_name.c_str(), CUPHY_R_32U, 1, const_cast<void*>(reinterpret_cast<const void*>(&nPrecodingMatrices_copy)));
                cuphy::write_HDF5_dataset(outHdf5File, proc_mode_name.c_str(), CUPHY_R_32U, 1, const_cast<void*>(reinterpret_cast<const void*>(&procModeBmsk_copy)));
                
                // Write per-cell SSB dynamic parameters
                if(dynamic_params->pPerCellSsbDynParams != nullptr && dynamic_params->nCells > 0)
                {
                    std::string cell_params_name = ssbId + "_PerCellSsbParams";
                    
                    // Create a buffer for per-cell parameters to write as array
                    cuphy::buffer<cuphyPerCellSsbDynPrms_t, cuphy::pinned_alloc> cell_params_buffer(dynamic_params->nCells);
                    std::copy(dynamic_params->pPerCellSsbDynParams, dynamic_params->pPerCellSsbDynParams + dynamic_params->nCells, cell_params_buffer.addr());
                    
                    cuphy::write_HDF5_dataset(outHdf5File, cell_params_name.c_str(), CUPHY_R_8U, 
                                             static_cast<int32_t>(dynamic_params->nCells * sizeof(cuphyPerCellSsbDynPrms_t)), 
                                             const_cast<void*>(reinterpret_cast<const void*>(cell_params_buffer.addr())));
                    
                    NVLOGD_FMT(NVLOG_SSB, "Written {} per-cell SSB parameters to dataset {}", dynamic_params->nCells, cell_params_name);
                }
                
                // Write per-SS block dynamic parameters
                if(dynamic_params->pPerSsBlockParams != nullptr && dynamic_params->nSSBlocks > 0)
                {
                    std::string ss_block_params_name = ssbId + "_PerSSBlockParams";
                    
                    // Create a buffer for per-SS block parameters to write as array
                    cuphy::buffer<cuphyPerSsBlockDynPrms_t, cuphy::pinned_alloc> ss_block_params_buffer(dynamic_params->nSSBlocks);
                    std::copy(dynamic_params->pPerSsBlockParams, dynamic_params->pPerSsBlockParams + dynamic_params->nSSBlocks, ss_block_params_buffer.addr());
                    
                    cuphy::write_HDF5_dataset(outHdf5File, ss_block_params_name.c_str(), CUPHY_R_8U, 
                                             static_cast<int32_t>(dynamic_params->nSSBlocks * sizeof(cuphyPerSsBlockDynPrms_t)), 
                                             const_cast<void*>(reinterpret_cast<const void*>(ss_block_params_buffer.addr())));
                    
                    NVLOGD_FMT(NVLOG_SSB, "Written {} per-SS block parameters to dataset {}", dynamic_params->nSSBlocks, ss_block_params_name);
                }
                
                // Write precoding matrices
                if(dynamic_params->pPmwParams != nullptr && dynamic_params->nPrecodingMatrices > 0)
                {
                    std::string precoding_matrix_name = ssbId + "_PrecodingMatrices";
                    
                    // Create a buffer for precoding matrices to write as array
                    cuphy::buffer<cuphyPmWOneLayer_t, cuphy::pinned_alloc> precoding_buffer(dynamic_params->nPrecodingMatrices);
                    std::copy(dynamic_params->pPmwParams, dynamic_params->pPmwParams + dynamic_params->nPrecodingMatrices, precoding_buffer.addr());
                    
                    cuphy::write_HDF5_dataset(outHdf5File, precoding_matrix_name.c_str(), CUPHY_R_8U, 
                                             static_cast<int32_t>(dynamic_params->nPrecodingMatrices * sizeof(cuphyPmWOneLayer_t)), 
                                             const_cast<void*>(reinterpret_cast<const void*>(precoding_buffer.addr())));
                    
                    NVLOGD_FMT(NVLOG_SSB, "Written {} precoding matrices to dataset {}", dynamic_params->nPrecodingMatrices, precoding_matrix_name);
                }
                
                // Write input data (MIB bits)
                if(dynamic_params->pDataIn != nullptr && dynamic_params->pDataIn->pMibInput != nullptr && dynamic_params->nSSBlocks > 0)
                {
                    std::string input_data_name = ssbId + "_InputMIBData";
                    
                    // Write the MIB input data
                    cuphy::write_HDF5_dataset(outHdf5File, input_data_name.c_str(), CUPHY_R_32U, 
                                             static_cast<int32_t>(dynamic_params->nSSBlocks), 
                                             const_cast<void*>(reinterpret_cast<const void*>(dynamic_params->pDataIn->pMibInput)));
                    
                    NVLOGD_FMT(NVLOG_SSB, "Written MIB input data ({} elements) to dataset {}", dynamic_params->nSSBlocks, input_data_name);
                }
                
                // Write SSB output tensor data for each cell
                if(dynamic_params->pDataOut != nullptr && dynamic_params->pDataOut->pTDataTx != nullptr)
                {
                    for (int cell_idx = 0; cell_idx < dynamic_params->nCells; cell_idx++) {
                        // Create unique dataset name for this cell and SSB object
                        std::string dataset_name = ssbId + "_Cell" + std::to_string(cell_idx) + "_Output";
                        
                        // Write the tensor data to HDF5 file using the correct cuPHY API
                        if(dynamic_params->pDataOut->pTDataTx[cell_idx].pAddr != nullptr && 
                           dynamic_params->pDataOut->pTDataTx[cell_idx].desc != nullptr)
                        {
                            try {
                                // Get tensor dimensions from descriptor
                                int num_dims;
                                int dims[5];
                                CUPHY_CHECK(cuphyGetTensorDescriptor(dynamic_params->pDataOut->pTDataTx[cell_idx].desc, 5, nullptr, &num_dims, dims, nullptr));
                                
                                // Check that tensor has at least 3 dimensions before accessing dims[1] and dims[2]
                                if (num_dims < 3) {
                                    NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "Tensor for cell {} has insufficient dimensions ({} < 3)", cell_idx, num_dims);
                                    continue; // Skip this cell and continue with the next one
                                }
                                
                                // Create a tensor device wrapper for the GPU tensor data using the same pattern as PDSCH
                                cuphy::tensor_device gpu_tensor_wrapper = cuphy::tensor_device(dynamic_params->pDataOut->pTDataTx[cell_idx].pAddr,
                                                                                             CUPHY_C_16F,
                                                                                             dims[0],
                                                                                             dims[1],
                                                                                             dims[2],
                                                                                             cuphy::tensor_flags::align_tight);
                                
                                // Allocate CPU tensor with pinned memory using same layout
                                cuphy::typed_tensor<CUPHY_C_16F, cuphy::pinned_alloc> cpu_tensor(gpu_tensor_wrapper.layout());
                                
                                // Copy data from GPU to CPU
                                cpu_tensor.convert(gpu_tensor_wrapper, cuStream);
                                CUDA_CHECK(cudaStreamSynchronize(cuStream));
                                
                                // Write the CPU tensor data to HDF5 file using the tensor-aware API
                                cuphy::write_HDF5_dataset(outHdf5File, cpu_tensor, dataset_name.c_str(), cuStream);
                                
                                // Safe logging with dimension bounds checking
                                NVLOGD_FMT(NVLOG_SSB, "Written SSB output data for cell {} to dataset {} (dims: {}x{}x{})", 
                                          cell_idx, dataset_name, dims[0], dims[1], dims[2]);
                            }
                            catch(const std::exception& e) {
                                NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "Failed to write tensor data for cell {}: {}", cell_idx, e.what());
                            }
                        }
                    }
                }
            }
            
            // File will be automatically closed when outHdf5File goes out of scope
            NVLOGD_FMT(NVLOG_SSB, "Successfully written SSB debug data to HDF5 file and closed file (SSB ID: {})", ssbId);
        }
        catch(const std::exception& e) {
            NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "Error writing SSB debug data: {}", e.what());
        }
        
        // File lock will be automatically released when fileLock goes out of scope
        NVLOGD_FMT(NVLOG_SSB, "Released file lock for HDF5 file: {}", m_outputPrms.outFileName);
    }
}

template <fmtlog::LogLevel log_level>
void SsbTx::printStaticApiPrms(cuphySsbStatPrms_t const* pStaticPrms)
{
    NVLOG_FMT(log_level, NVLOG_SSB, "=== SSB TX Static Parameters ===");
    NVLOG_FMT(log_level, NVLOG_SSB, "nMaxCellsPerSlot: {}", pStaticPrms->nMaxCellsPerSlot);
    
    if(pStaticPrms->pDbgPrms != nullptr)
    {
        NVLOG_FMT(log_level, NVLOG_SSB, "Debug parameters:");
        NVLOG_FMT(log_level, NVLOG_SSB, "  enableApiLogging: {}", pStaticPrms->pDbgPrms->enableApiLogging);
        NVLOG_FMT(log_level, NVLOG_SSB, "  pOutFileName: {}", 
                  pStaticPrms->pDbgPrms->pOutFileName ? pStaticPrms->pDbgPrms->pOutFileName : "NULL");
    }
    NVLOG_FMT(log_level, NVLOG_SSB, "==============================");
}

template <fmtlog::LogLevel log_level>
void SsbTx::printDynApiPrms(cuphySsbDynPrms_t const* pDynPrms)
{
    NVLOG_FMT(log_level, NVLOG_SSB, "=== SSB TX Dynamic Parameters ===");
    NVLOG_FMT(log_level, NVLOG_SSB, "nCells: {}", pDynPrms->nCells);
    NVLOG_FMT(log_level, NVLOG_SSB, "nSSBlocks: {}", pDynPrms->nSSBlocks);
    NVLOG_FMT(log_level, NVLOG_SSB, "nPrecodingMatrices: {}", pDynPrms->nPrecodingMatrices);
    NVLOG_FMT(log_level, NVLOG_SSB, "procModeBmsk: 0x{:x}", pDynPrms->procModeBmsk);

    // Detailed per-cell SSB parameters
    NVLOG_FMT(log_level, NVLOG_SSB, "SSB TX pipeline: per cell SSB parameters");
    for (int i = 0; i < pDynPrms->nCells; i++) {
        const cuphyPerCellSsbDynPrms_t& per_cell_SSB_params = pDynPrms->pPerCellSsbDynParams[i];
        NVLOG_FMT(log_level, NVLOG_SSB, "SSB parameters for cell {:5d}: ", i);
        NVLOG_FMT(log_level, NVLOG_SSB, "------------------------------------------------------");
        NVLOG_FMT(log_level, NVLOG_SSB, "NID:            {:5d}", per_cell_SSB_params.NID);
        NVLOG_FMT(log_level, NVLOG_SSB, "nHF:            {:5d}", per_cell_SSB_params.nHF);
        NVLOG_FMT(log_level, NVLOG_SSB, "Lmax:           {:5d}", per_cell_SSB_params.Lmax);
        NVLOG_FMT(log_level, NVLOG_SSB, "SFN:            {:5d}", per_cell_SSB_params.SFN);
        NVLOG_FMT(log_level, NVLOG_SSB, "k_SSB:          {:5d}", per_cell_SSB_params.k_SSB);
        NVLOG_FMT(log_level, NVLOG_SSB, "nF:             {:5d}", per_cell_SSB_params.nF);
        NVLOG_FMT(log_level, NVLOG_SSB, "slotBufferIdx   {:5d}", per_cell_SSB_params.slotBufferIdx);
    }

    // Detailed per-SS block parameters
    NVLOG_FMT(log_level, NVLOG_SSB, "SSB TX pipeline: SSB parameters across all {} SS blocks: ", pDynPrms->nSSBlocks);
    for (int i = 0; i < pDynPrms->nSSBlocks; i++) {
        NVLOG_FMT(log_level, NVLOG_SSB, "SSB parameters for SS block {}: ", i);
        NVLOG_FMT(log_level, NVLOG_SSB, "------------------------------------------------------");
        const cuphyPerSsBlockDynPrms_t& per_SS_block_params = pDynPrms->pPerSsBlockParams[i];
        NVLOG_FMT(log_level, NVLOG_SSB, "f0:            {:5d}", per_SS_block_params.f0);
        NVLOG_FMT(log_level, NVLOG_SSB, "t0:            {:5d}", per_SS_block_params.t0);
        NVLOG_FMT(log_level, NVLOG_SSB, "blockIndex:    {:5d}", per_SS_block_params.blockIndex);
        NVLOG_FMT(log_level, NVLOG_SSB, "beta_pss:      {:f}",  per_SS_block_params.beta_pss);
        NVLOG_FMT(log_level, NVLOG_SSB, "beta_sss:      {:f}",  per_SS_block_params.beta_sss);
        NVLOG_FMT(log_level, NVLOG_SSB, "cell_index:    {:5d}", per_SS_block_params.cell_index);
        NVLOG_FMT(log_level, NVLOG_SSB, "enablePrcdBf:  {:5d}", per_SS_block_params.enablePrcdBf);
        if(per_SS_block_params.enablePrcdBf)
        {
            NVLOG_FMT(log_level, NVLOG_SSB, "pmwPrmIdx:     {:5d}", per_SS_block_params.pmwPrmIdx);
            NVLOG_FMT(log_level, NVLOG_SSB, "nPorts :       {:5d}", pDynPrms->pPmwParams[per_SS_block_params.pmwPrmIdx].nPorts);
            std::stringstream matrix_row;
            matrix_row.precision(5);
            matrix_row << "Precoding Matrix:      ";
            for(int idx = 0; idx < pDynPrms->pPmwParams[per_SS_block_params.pmwPrmIdx].nPorts; idx++)
            {
                if (idx != 0) matrix_row << ", ";
                matrix_row << std::fixed << "{" << (float)pDynPrms->pPmwParams[per_SS_block_params.pmwPrmIdx].matrix[idx].x << ", " <<  (float)pDynPrms->pPmwParams[per_SS_block_params.pmwPrmIdx].matrix[idx].y << "}";
            }
            NVLOG_FMT(log_level, NVLOG_SSB, "{}", matrix_row.str());
        }
    }

    // PBCH MIB data for each SS block
    NVLOG_FMT(log_level, NVLOG_SSB, "SSB TX pipeline: PBCH MIB across all {} SSBs: ", pDynPrms->nSSBlocks);
    for (int i = 0; i < pDynPrms->nSSBlocks; i++) {
        // 24 (Nmib) most significant bits are valid; SSBs in the same slot of a cell share the same MIB but it is still separately stored //FIXME
        // cuPHY does not check that this is the case.
        NVLOG_FMT(log_level, NVLOG_SSB, "PBCH MIB for SS block {}: {:#x}", i, pDynPrms->pDataIn[0].pMibInput[i]);
    }
    
    NVLOG_FMT(log_level, NVLOG_SSB, "===============================");
}

// Explicit instantiation for common log levels
template void SsbTx::printStaticApiPrms<fmtlog::DBG>(cuphySsbStatPrms_t const* pStaticPrms);
template void SsbTx::printStaticApiPrms<fmtlog::INF>(cuphySsbStatPrms_t const* pStaticPrms);
template void SsbTx::printDynApiPrms<fmtlog::DBG>(cuphySsbDynPrms_t const* pDynPrms);
template void SsbTx::printDynApiPrms<fmtlog::INF>(cuphySsbDynPrms_t const* pDynPrms);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cuphyWriteDbgBufSynchSsb()

cuphyStatus_t CUPHYWINAPI cuphyWriteDbgBufSynchSsb(cuphySsbTxHndl_t ssbTxHndl, cudaStream_t cuStream)
{
    if(!ssbTxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        auto* p = static_cast<SsbTx*>(ssbTxHndl);
        p->writeDbgBufSynch(cuStream);
        return CUPHY_STATUS_SUCCESS;
    });
}


SsbTx::SsbTx(const cuphySsbStatPrms_t* cfg_static_params):
    dynamic_params(nullptr),
    static_params(cfg_static_params),
    m_component_descrs("SsbDescr"),
    bulk_desc_async_copy(true),
    /* the following members will be updated on every setup() */
    num_cells(0),
    num_SSBs(0),
    m_encdRmSSBArgs{0},
    m_ssbMapperArgs{0},
    m_cudaGraphModeEnabled(true)
{
    cfg_static_params->pOutInfo->pMemoryFootprint = &memory_footprint; // update  static parameter field that points to the cuphyMemoryFootprintTracker object for this channel
    max_cells_per_slot   = static_params->nMaxCellsPerSlot;
    max_SSBs_per_slot    = max_cells_per_slot * CUPHY_SSB_MAX_SSBS_PER_CELL_PER_SLOT;
    allocateBuffers();
    allocateDescr(); //Allocate Descriptors.

    //Memset all launch configs structs to 0; they will be properly updated on every setup() as needed
    memset(&m_encodeRmMultiSsbLaunchCfg, 0, sizeof(cuphyEncoderRateMatchMultiSSBLaunchCfg_t));
    memset(& m_ssbMapperLaunchCfg, 0, sizeof(cuphySsbMapperLaunchCfg_t));

    // Superfluous given the memset above, but left it to make it clear this was to be set to nullptr.
    // Set kernel functions to nullptr to ensure call to cudaFuncGetSymbol happens only once.
    m_encodeRmMultiSsbLaunchCfg.kernelNodeParamsDriver.func = nullptr;
    // func for m_ssbMapperLaunchCfg not set here because it's templated

    // the following call to create graph also helps (by calling cudaGetFuncBySymbol) to move CUDA runtime initialization overhead into channel constructor
    createGraph();
#if CUDA_VERSION >= 12000
    CU_CHECK_EXCEPTION(cuGraphInstantiate(&m_graphExec, m_graph, 0));
#else
    CU_CHECK_EXCEPTION(cuGraphInstantiate(&m_graphExec, m_graph, 0, 0, 0));
#endif
    // Debug parameters
    m_outputPrms.debugOutputFlag = false;
    NVLOGD_FMT(NVLOG_SSB, "SSB Constructor: pDbgPrms = {}", (void*)static_params->pDbgPrms);
    if(nullptr != static_params->pDbgPrms)
    {
        NVLOGD_FMT(NVLOG_SSB, "SSB Constructor: pOutFileName = {}", 
                   static_params->pDbgPrms->pOutFileName ? static_params->pDbgPrms->pOutFileName : "NULL");
        if(static_params->pDbgPrms->pOutFileName != nullptr)
        {
            m_outputPrms.debugOutputFlag = true;
            // Store filename instead of creating file immediately
            m_outputPrms.outFileName = std::string(static_params->pDbgPrms->pOutFileName);
            NVLOGD_FMT(NVLOG_SSB, "SSB Constructor: Debug enabled, filename stored: {}", static_params->pDbgPrms->pOutFileName);
        }
    }

    if (PRINT_GPU_MEMORY_CUPHY_CHANNEL == 1) 
    {
        memory_footprint.printMemoryFootprint(this, "SSB");
    }
}



void SsbTx::allocateBuffers()
{
    // allocate device tensors
    d_x_coded    = make_unique_device<uint8_t>(max_SSBs_per_slot * (CUPHY_SSB_N_PBCH_POLAR_ENCODED_BITS/8), &memory_footprint); // output sequence of polar encoder
    d_x_tx       = make_unique_device<uint8_t>(max_SSBs_per_slot * (round_up_to_next(CUPHY_SSB_N_PBCH_SCRAMBLING_SEQ_BITS, 32)/8), &memory_footprint); // output sequence of rate match, 32-bit aligned and multiple of 4 bytes
}

void SsbTx::allocateDescr()
{
    std::array<size_t, N_SSB_COMPONENTS> dynDescrSizeBytes{};
    std::array<size_t, N_SSB_COMPONENTS> dynDescrAlignBytes{};

    size_t* pDynDescrSizeBytes  = dynDescrSizeBytes.data();
    size_t* pDynDescrAlignBytes = dynDescrAlignBytes.data();

    //Allocate workspace memory to take advantage of the bulk async copy of the descriptors.

    // Allocate memory for parameters
    pDynDescrSizeBytes[SSB_PER_CELL_PARAMS]  = sizeof(cuphyPerCellSsbDynPrms_t) * max_cells_per_slot;
    pDynDescrAlignBytes[SSB_PER_CELL_PARAMS] = alignof(cuphyPerCellSsbDynPrms_t);

    pDynDescrSizeBytes[PER_SS_BLOCK_PARAMS]  = max_SSBs_per_slot * sizeof(cuphyPerSsBlockDynPrms_t);
    pDynDescrAlignBytes[PER_SS_BLOCK_PARAMS] = alignof(cuphyPerSsBlockDynPrms_t);

    pDynDescrSizeBytes[SSB_PRECODING_MATRIX]  = max_SSBs_per_slot * sizeof(cuphyPmWOneLayer_t);
    pDynDescrAlignBytes[SSB_PRECODING_MATRIX] = alignof(cuphyPmWOneLayer_t);

    // Allocate memory for input array after CRC attachment. TODO This computation is currently done on the host.
    // FIXME  x_crc per SSB has to be uint32_t aligned so use aligned_x_crc size
    int aligned_x_crc_size = round_up_to_next(CUPHY_SSB_N_PBCH_SEQ_W_CRC_BITS, 32)/ 8;
    pDynDescrSizeBytes[SSB_INPUT_W_CRC]  = max_SSBs_per_slot * aligned_x_crc_size * sizeof(uint8_t);
    pDynDescrAlignBytes[SSB_INPUT_W_CRC] = alignof(uint32_t);

    pDynDescrSizeBytes[CELL_OUTPUT_ADDR]  = max_cells_per_slot * sizeof(__half2*);
    pDynDescrAlignBytes[CELL_OUTPUT_ADDR] = alignof(__half2*);

    m_component_descrs.alloc(dynDescrSizeBytes, dynDescrAlignBytes, &memory_footprint);
    //m_component_descrs.displayDescrSizes();

    // Added for convenience
    h_x_crc = (uint8_t*)m_component_descrs.getCpuStartAddrs()[SSB_INPUT_W_CRC];
    d_x_crc = (uint8_t*)m_component_descrs.getGpuStartAddrs()[SSB_INPUT_W_CRC];

}


// generate gold sequence with given init state and length
void genGoldSeq(uint32_t c_init, uint32_t len, uint32_t *x1, uint32_t *x2, uint32_t *c)
{
    for(int i = 0; i < 32; i++) {
        x1[i] = 0;
        x2[i] = (c_init >> i) & 0x1;
    }
    x1[0] = 1;

    for(int i = 0; i < Nc + len - 31; i++) {
        x1[i + 31] = (x1[i + 3] + x1[i]) % 2;                         // x1(n + 31) = mod(x1(n + 3) + x1(n),2)
        x2[i + 31] = (x2[i + 3] + x2[i + 2] + x2[i + 1] + x2[i]) % 2; // x2(n + 31) = mod(x2(n + 3) + x2(n + 2) + x2(n + 1) + x2(n),2)
    }

    for(int i = 0; i < len; i++) {
        c[i] = (x1[i + Nc] + x2[i + Nc]) % 2; // c(n) = mod(x1(n + Nc) + x2(n + Nc),2)
    }
}

// reverse bits order within each byte
int reverseBitInByte(uint8_t * inputByte, uint8_t * outputByte, uint32_t Nbyte)
{
    for (int idxByte = 0; idxByte < Nbyte; idxByte++) {
        outputByte[idxByte] = 0;
        for (int idxBit = 0; idxBit < 8; idxBit ++) {
            outputByte[idxByte] <<= 1;
            outputByte[idxByte] += (inputByte[idxByte] >> idxBit) & 1;
        }
    }
    return 0;
}


// genereate crc bits for pbch payload
template <typename uintCRC_t, size_t uintCRCBitLength>
uintCRC_t computeCRC(const uint8_t* input,
                     uint32_t       size,
                     uintCRC_t      poly,
                     uintCRC_t      initVal = 0,
                     uint64_t       stride  = 1)
{
    uintCRC_t crc         = initVal;
    uintCRC_t msbMask     = (1 << (uintCRCBitLength - 1));
    uintCRC_t allOnesMask = static_cast<uintCRC_t>(-1);
    if((sizeof(uintCRC_t) * 8 - uintCRCBitLength) > 0)
        allOnesMask >>= (sizeof(uintCRC_t) * 8 - uintCRCBitLength - 1);
    for(int i = 0; i < size * stride; i += stride)
    {
        crc ^= static_cast<uintCRC_t>(input[i] << (uintCRCBitLength - 8));
        for(int b = 0; b < 8; b++)
        {
            uintCRC_t pred = (crc & msbMask) == 0;
            crc <<= 1;
            crc ^= (poly & (pred + allOnesMask));
        }
    }
    return crc;
}


//Assume x_mib is packed bits per uint32_t
int pbchGenPayload_per_SSB(uint32_t* x_mib,
                           uint32_t* x_payload,
                           uint8_t   SSB_block_index,
                           const cuphyPerCellSsbDynPrms_t*   cell_params)
{
   // The CUPHY_SSB_N_PBCH_PAYLOAD_BITS 32-bit wide sequence has always bits 0, 6 and 24 set to 1, whereas bits 2, 3 and 5 are only set if Lmax==64.
   // Can change bit order if more convenient

   const uint32_t no_scrambling_Lmax_ne_64 = 0x01000041;
   const uint32_t no_scrambling_Lmax_eq_64 = (no_scrambling_Lmax_ne_64 | 0x0000001c);
   const uint32_t no_scrambling_val = (cell_params->Lmax == 64) ? no_scrambling_Lmax_eq_64 : no_scrambling_Lmax_ne_64; // Reminder Lmax is per cell/slot

   const int a_from_abar_mapping[32] =
   {28, 0, 31, 30, 7, 29, 25, 27,
     5, 8, 24,  9, 10, 11, 12, 13,
     1, 4,  3, 14, 15, 16, 17, 2,
    26, 18, 19, 20, 21, 22, 6, 23};

   uint32_t a0_3 = cell_params->SFN & 0xf; // cell
   uint32_t a5_7 = 0;
   uint32_t M = 0;

    if (cell_params->Lmax == 64) {
        a5_7 = (SSB_block_index & 0x3f) >> 3;
        M = 32-6;
    }
    else if ((cell_params->Lmax == 4) || (cell_params->Lmax == 8)) {
        uint32_t msb_k_ssb = cell_params->k_SSB >> 4; // cell
        a5_7 = msb_k_ssb << 2;
        M = 32-3;
    }
    else {
        NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "Invalid Lmax value {}. L_max can only be 4, 8 or 64.", cell_params->Lmax);
        return 1;
    }

    uint32_t a_bar = x_mib[0] << (32 - CUPHY_SSB_N_MIB_BITS); // Assumption that MIB occupies the least significant bits and all  unused MIB bits set to 0; can also mask to be safe
    a_bar |= (a0_3 << 4);
    a_bar |= ((cell_params->nHF & 0x1) << 3);
    a_bar |= (a5_7 & 0x7);
    //all 32-bits of a_bar are used.

    uint32_t v = (a0_3 & 6) >> 1;
    uint32_t len = v*M + 32;

    // FIXME gold seq. gen and x_payload below are one bit per uint32_t element could just pack them and then do bitwise operations
    // Gold seq. generation
    uint32_t x1[Nc + len], x2[Nc + len], c[len];
    genGoldSeq(cell_params->NID, len, x1, x2, c); // populates array c, with one valid bit per uint32_t.

    uint32_t j = 0;
    x_payload[0] = 0;
    for (int i = 0; i < CUPHY_SSB_N_PBCH_PAYLOAD_BITS; i++) {
         uint32_t a = (a_bar >> (31-a_from_abar_mapping[i])) & 0x1;

         if (((no_scrambling_val >> i) & 0x1) == 0) {
             x_payload[0] |= (((a + c[j + v*M]) & 1) << (31 - i));
             j++;
         } else {
             x_payload[0] |= ((a & 1) << (31 - i));
         }
    }
    return 0;
}

cuphyStatus_t SsbTx::preparePBCH(uint32_t*                                 h_x_mib,
                                 const cuphyPerSsBlockDynPrms_t*           h_ssb_params,
                                 const cuphyPerCellSsbDynPrms_t*           h_per_cell_params,
                                 cuphyEncoderRateMatchMultiSSBLaunchCfg_t* pEncdRMLaunchCfg,
                                 cuphySsbMapperLaunchCfg_t*                pSsbMapperLaunchCfg)
{
    if ((h_x_mib == nullptr) || (h_ssb_params == nullptr) || (h_per_cell_params == nullptr) ||
        (num_SSBs == 0) || (num_cells == 0))
    {
        NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "SSB setup error! Received nullptr for one of pMibInput {:p}, pPerSsBlockParams {:p} or pPerCellSsbDynParams {:p} pointers, or zero for for SSBs ({}) or cells ({}).",
                   static_cast<void*>(h_x_mib), const_cast<void*>(static_cast<void const*>(h_ssb_params)),
                   const_cast<void*>(static_cast<void const*>(h_per_cell_params)), num_SSBs, num_cells);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    cuphyStatus_t status;

    // Go over all SSBs and populate the h_x_crc host buffer based on the SSB and cell parameters and the SSB MIB input data.
    // The h_x_crc buffer will be copied to the GPU as part of the SSB pipeline before calling Run.
    for (int i = 0; i < num_SSBs; i++) {

        uint32_t* x_mib_for_SSB = h_x_mib + i; // The MIB input buffer contains one uint32_t element per SSB

        // Generate 32-bit payload, x_payload_packed, for this SSB. The following parameter fields are used:
        //  -  SSB specific: blockIndex
        //  -  cell specific: SFN, Nhf, NID, Lmax, k_SSB

        uint32_t x_payload_packed; // packed; old x_payload[0] is at least significant x_payload bit
        uint16_t cell_index = h_ssb_params[i].cell_index;
        if (cell_index > num_cells) {
            NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "SSB error! Received cell_index {} for SSB {}, but current number of cells is {}.", cell_index, i, num_cells);
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
        int e = pbchGenPayload_per_SSB(x_mib_for_SSB, &x_payload_packed, h_ssb_params[i].blockIndex, h_per_cell_params + cell_index);  // Note x_payload is packed 32 bits
        if (e != 0) { // Possible if L_max was invalid
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }

        // add CRC to PBCH payload
        uint8_t addCrc[CUPHY_SSB_N_PBCH_SEQ_W_CRC_BITS/8];
        uint32_t totalOutTBByteSize = CUPHY_SSB_N_PBCH_SEQ_W_CRC_BITS/8;
        for (int idxByte = 0; idxByte < 4; idxByte ++) { // 4 = CUPHY_SSB_N_PBCH_PAYLOAD_BITS/8
            addCrc[idxByte] = (x_payload_packed >> (3-idxByte)*8);
        }

        // Compute 24-bit CRC
        uint32_t crc = computeCRC<uint32_t, 24>(addCrc, 4, G_CRC_24_C, 0, 1);

        // Fill in 24-bit CRC into bytes 4-6 of addCrc
        addCrc[4] = (crc >> 16) & 0xFF;
        addCrc[5] = (crc >> 8) & 0xFF;
        addCrc[6] = (crc >> 0) & 0xFF;

        // Reverse bits in byte in addCRC and store output in h_x_crc buffer to match with expected polar encoder input format
        int aligned_x_crc_size = 8; //round_up_to_next(CUPHY_SSB_N_PBCH_SEQ_W_CRC_BITS, 32)/ 8;
        reverseBitInByte(addCrc, h_x_crc + i * aligned_x_crc_size, totalOutTBByteSize);
    }

    // set SSB kernels launch configs
    status = cuphySSBsKernelSelect(pEncdRMLaunchCfg, pSsbMapperLaunchCfg, num_SSBs);

    return status;
}


cuphyStatus_t SsbTx::expandParameters(cuphySsbDynPrms_t* dyn_params, cudaStream_t cuda_strm)
{
    num_cells = dyn_params->nCells;
    num_SSBs  = dyn_params->nSSBlocks;
    
    if (num_cells > max_cells_per_slot || num_SSBs > max_SSBs_per_slot ||  dyn_params->nPrecodingMatrices > max_SSBs_per_slot) [[unlikely]]
    {
        NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "{}: nCells ({}) or nSSBlocks ({}) or nPrecodingMatrices ({}) exceeds allocated capacity ({} {})",
                   __FUNCTION__, num_cells, num_SSBs, dyn_params->nPrecodingMatrices, max_cells_per_slot, max_SSBs_per_slot);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    //The extra host to host copy is needed if we want the respective H2D copy to be part of the single bulk async. copy.
    //FIXME can skip and have separate H2D copies instead
    memcpy(m_component_descrs.getCpuStartAddrs()[SSB_PER_CELL_PARAMS],
           dyn_params->pPerCellSsbDynParams,
           sizeof(cuphyPerCellSsbDynPrms_t) * num_cells); // reminder the bulk copy will copy max_cells_per_slot

    memcpy(m_component_descrs.getCpuStartAddrs()[PER_SS_BLOCK_PARAMS],
           dyn_params->pPerSsBlockParams,
           sizeof(cuphyPerSsBlockDynPrms_t) * num_SSBs);

    if(dyn_params->nPrecodingMatrices>0)
    {
        memcpy(m_component_descrs.getCpuStartAddrs()[SSB_PRECODING_MATRIX],
               dyn_params->pPmwParams, sizeof(cuphyPmWOneLayer_t) * dyn_params->nPrecodingMatrices);
    }

    cuphyStatus_t status = preparePBCH(dyn_params->pDataIn->pMibInput,
                                       dyn_params->pPerSsBlockParams,
                                       dyn_params->pPerCellSsbDynParams,
                                       &m_encodeRmMultiSsbLaunchCfg,
                                       &m_ssbMapperLaunchCfg);

    if (status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "Error during SSB setup.");
        return status;
    }

    __half2** h_output_addr = (__half2**)m_component_descrs.getCpuStartAddrs()[CELL_OUTPUT_ADDR];
    for (int i = 0; i < num_cells; i++) {
        h_output_addr[i] = (__half2*)dyn_params->pDataOut->pTDataTx[i].pAddr;
    }

    // Bulk async H2D copy for all workspaces
    if(bulk_desc_async_copy)
    {
        m_component_descrs.asyncCpuToGpuCpy(cuda_strm);
    }
    return CUPHY_STATUS_SUCCESS;
}


void SsbTx::createGraph()
{
    CU_CHECK_EXCEPTION(cuGraphCreate(&m_graph, 0));
    // Add node(s). Initially start with some kernel parameters, and at setup do the updating

    // Set empty graph kernel nodes with the appropriate argument count (all pointers) to avoid dynamic
    // memory allocation during graph kernel node update. If the number of kernel parameters changes, the calls below should be updated.
    void* arg;
    void* kernelParams[5] = {&arg, &arg, &arg, &arg, &arg}; // use max. number of kernel args for array size
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode5ParamsDriver, 5, &kernelParams[0]));
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode3ParamsDriver, 3, &kernelParams[0]));

    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_encdRmMultiSsbNode, m_graph, nullptr, 0, &m_emptyNode3ParamsDriver));
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_ssbMapperNode, m_graph, &m_encdRmMultiSsbNode, 1, &m_emptyNode5ParamsDriver));
}

void SsbTx::updateGraph()
{
    CU_CHECK_EXCEPTION_W_TAG(NVLOG_SSB, cuGraphExecKernelNodeSetParams(m_graphExec, m_encdRmMultiSsbNode, &(m_encodeRmMultiSsbLaunchCfg.kernelNodeParamsDriver)));
    CU_CHECK_EXCEPTION_W_TAG(NVLOG_SSB, cuGraphExecKernelNodeSetParams(m_graphExec, m_ssbMapperNode, &(m_ssbMapperLaunchCfg.kernelNodeParamsDriver)));
}

cuphyStatus_t SsbTx::setup(cuphySsbDynPrms_t* dyn_params)
{
    dynamic_params = dyn_params; // Needed to retrieve CUDA stream etc.
    //printDynApiPrms<fmtlog::DBG>(dynamic_params); //FIXME comment out or add a single check if log level < comp. nvlog level and return immediately to avoid runtime checks overhead
    cuphyStatus_t status = expandParameters(dyn_params, dyn_params->cuStream);
    if (status != CUPHY_STATUS_SUCCESS) {
        return status;
    }
    dyn_params->chan_graph = &m_graph;
    //---------------------------------------------
    // set kernel args for encodeRateMatchMultipleSSBsKernel()
    m_encdRmSSBArgs[0] = d_x_crc;
    m_encdRmSSBArgs[1] = d_x_coded.get();
    m_encdRmSSBArgs[2] = d_x_tx.get();

    m_encodeRmMultiSsbLaunchCfg.kernelArgs[0] = &m_encdRmSSBArgs[0];
    m_encodeRmMultiSsbLaunchCfg.kernelArgs[1] = &m_encdRmSSBArgs[1];
    m_encodeRmMultiSsbLaunchCfg.kernelArgs[2] = &m_encdRmSSBArgs[2];

    m_encodeRmMultiSsbLaunchCfg.kernelNodeParamsDriver.kernelParams = m_encodeRmMultiSsbLaunchCfg.kernelArgs;

    // set kernel args for ssbModTfSigKernel()
    m_ssbMapperArgs[0] = m_component_descrs.getGpuStartAddrs()[CELL_OUTPUT_ADDR];
    m_ssbMapperArgs[1] = d_x_tx.get();
    m_ssbMapperArgs[2] = m_component_descrs.getGpuStartAddrs()[PER_SS_BLOCK_PARAMS];
    m_ssbMapperArgs[3] = m_component_descrs.getGpuStartAddrs()[SSB_PER_CELL_PARAMS];
    m_ssbMapperArgs[4] = m_component_descrs.getGpuStartAddrs()[SSB_PRECODING_MATRIX];

    m_ssbMapperLaunchCfg.kernelArgs[0] = &m_ssbMapperArgs[0];
    m_ssbMapperLaunchCfg.kernelArgs[1] = &m_ssbMapperArgs[1];
    m_ssbMapperLaunchCfg.kernelArgs[2] = &m_ssbMapperArgs[2];
    m_ssbMapperLaunchCfg.kernelArgs[3] = &m_ssbMapperArgs[3];
    m_ssbMapperLaunchCfg.kernelArgs[4] = &m_ssbMapperArgs[4];

    m_ssbMapperLaunchCfg.kernelNodeParamsDriver.kernelParams = m_ssbMapperLaunchCfg.kernelArgs;
    //---------------------------------------------

    //executable graph setup
    m_cudaGraphModeEnabled = (dynamic_params->procModeBmsk & CSIRS_PROC_MODE_GRAPHS) ? true : false;
    if(m_cudaGraphModeEnabled)
    {
        updateGraph();
    }
    return CUPHY_STATUS_SUCCESS;
}


cuphyStatus_t SsbTx::run(const cudaStream_t& cuda_strm)
{

    if(m_cudaGraphModeEnabled)
    {
        MemtraceDisableScope md; // Disable temporarily
        CUresult e = cuGraphLaunch(m_graphExec, cuda_strm);
        if (e != CUDA_SUCCESS) {
            const char* pErrStr;
            CUresult str_e = cuGetErrorString(e, &pErrStr);
            NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "Invalid graph launch for SSB ({}, {}).", e, (str_e == CUDA_SUCCESS) ? pErrStr : "CUDA error");
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
    }
    else
    {
        // Polar encoder and rate matching kernel processing multiple SSBs.
        // Offsets into d_x_crc and d_x_coded and d_x_tx buffers per SSB are hard-coded internally.
        cuphyStatus_t polar_enc_status = cuphyRunPolarEncRateMatchSSBs(&m_encodeRmMultiSsbLaunchCfg,
                                                                       d_x_crc,
                                                                       d_x_coded.get(),
                                                                       d_x_tx.get(),
                                                                       num_SSBs,
                                                                       cuda_strm);


        // modulate PBCH, DMRS, PSS and SSS sequences and map to time frequency domain subcarriers
        cuphyStatus_t ssb_mapper_status = cuphyRunSsbMapper(d_x_tx.get(),
                                                 (__half2**)m_component_descrs.getGpuStartAddrs()[CELL_OUTPUT_ADDR],
                                                 (const cuphyPerSsBlockDynPrms_t*)m_component_descrs.getGpuStartAddrs()[PER_SS_BLOCK_PARAMS],
                                                 (const cuphyPerCellSsbDynPrms_t*)m_component_descrs.getGpuStartAddrs()[SSB_PER_CELL_PARAMS],
                                                 (const cuphyPmWOneLayer_t*)m_component_descrs.getGpuStartAddrs()[SSB_PRECODING_MATRIX],
                                                 num_SSBs,
                                                 num_cells,
                                                 cuda_strm,
                                                 &m_ssbMapperLaunchCfg);


        if ((ssb_mapper_status != CUPHY_STATUS_SUCCESS) || (polar_enc_status != CUPHY_STATUS_SUCCESS)) {
            return CUPHY_STATUS_INTERNAL_ERROR;
            //throw std::runtime_error("\nError for cuphySsbTxPipeline\n");
        }
    }

    return CUPHY_STATUS_SUCCESS;
}

SsbTx::~SsbTx()
{
    CUDA_CHECK_NO_THROW(cudaGraphDestroy(m_graph));
    CUDA_CHECK_NO_THROW(cudaGraphExecDestroy(m_graphExec));
}
