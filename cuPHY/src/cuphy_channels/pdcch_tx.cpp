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

#include "pdcch_tx.hpp"
#include "cuphy_internal.h"
#include "nvlog.hpp"
#include "pdcch_polar_cidx2uidx_lut_cpu.h"

#include <cstring>

#define PRINT_CONFIG 0          // Set to 1 to print PDCCH common and per-DCI config. params.
#define MEMSET_IN_FIRST_SETUP 0 // when 0, memset happens in constructor; see comment in the code
#define MANUAL_WORKSPACE 1      // setting to 0 will use descriptors, but note H2D copy of the descriptors copies more data than needed
#define PDCCH_FUSED_KERNEL 1    // PDCCH_ALL mode kernel selection: 1 = fused single-kernel pipeline (encode + scrambling + TF signal, warp-specialized); 0 = legacy 3-kernel pipeline (fallback)

using namespace cuphy;

cuphyStatus_t CUPHYWINAPI cuphyCreatePdcchTx(cuphyPdcchTxHndl_t* pPdcchTxHndl, cuphyPdcchStatPrms_t const* pStatPrms)
{
    if((pPdcchTxHndl == nullptr) || (pStatPrms == nullptr))
    {
        NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "cuphyCreatePdcchTx called with cuphyPdcchTxHndl_t ptr ({:p}) or cuphyPdcchStatPrms_t ptr ({:p}) nullptr.", static_cast<void*>(pPdcchTxHndl), const_cast<void*>(static_cast<void const*>(pStatPrms)));
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    return cuphy::tryCallableAndCatch([&] {
        PdcchTx* new_pipeline = new PdcchTx(pStatPrms);

        if(new_pipeline == nullptr)
        {
            return CUPHY_STATUS_ALLOC_FAILED;
        }
        *pPdcchTxHndl = new_pipeline;
        return CUPHY_STATUS_SUCCESS;
    });
}
#if 0
const void* cuphyGetMemoryFootprintTrackerPdcchTx(cuphyPdcchTxHndl_t pdcchTxHndl)
{
    if(pdcchTxHndl == nullptr)
    {
        return nullptr;
    }
    PdcchTx* pipeline_ptr  = static_cast<PdcchTx*>(pdcchTxHndl);
    return pipeline_ptr->getMemoryTracker();
}
#endif

const void* PdcchTx::getMemoryTracker()
{
    return &memory_footprint;
}

cuphyStatus_t CUPHYWINAPI cuphySetupPdcchTx(cuphyPdcchTxHndl_t pdcchTxHndl, cuphyPdcchDynPrms_t* pDynPrms)
{
    PUSH_RANGE("PDCCH_SETUP", 1);
    //    NVLOGI_FMT(NVLOG_PDCCH, "PDCCH_SETUP {}", 1);
    if((pDynPrms == nullptr) || (pdcchTxHndl == nullptr))
    {
        NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "cuphySetupPdcchTx called with nullptr for cuphyPdcchTxHndl_t ({:p}) or for cuphyPdcchDynPrms_t ptr ({:p}).", static_cast<void*>(pdcchTxHndl), static_cast<void*>(pDynPrms));
        POP_RANGE
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    return cuphy::tryCallableAndCatch([&] {
        PdcchTx*      pipeline_ptr = static_cast<PdcchTx*>(pdcchTxHndl);
        cuphyStatus_t status       = pipeline_ptr->setup(pDynPrms);
        POP_RANGE
        return status;
    },
                                      CUPHY_STATUS_INVALID_ARGUMENT);
}

cuphyStatus_t CUPHYWINAPI cuphyRunPdcchTx(cuphyPdcchTxHndl_t pdcchTxHndl, uint64_t procModeBmsk /* not used */)
{
    PUSH_RANGE("PDCCH_RUN", 1);
    if(pdcchTxHndl == nullptr)
    {
        NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "cuphyRunPdcchTx error: cuphyPdcchTxHndl_t is nullptr!");
        POP_RANGE
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    return cuphy::tryCallableAndCatch([&] {
        PdcchTx* pipeline_ptr = static_cast<PdcchTx*>(pdcchTxHndl);

        cuphyStatus_t status = pipeline_ptr->run(pipeline_ptr->dynamic_params->cuStream);
        POP_RANGE
        return status;
    });
}

cuphyStatus_t CUPHYWINAPI cuphyDestroyPdcchTx(cuphyPdcchTxHndl_t pdcchTxHndl)
{
    if(pdcchTxHndl == nullptr)
    {
        NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "cuphyDestroyPdcchTx error: cuphyPdcchTxHndl_t is nullptr!");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    PdcchTx* pipeline_ptr = static_cast<PdcchTx*>(pdcchTxHndl);
    delete pipeline_ptr;
    return CUPHY_STATUS_SUCCESS;
}

PdcchTx::PdcchTx(const cuphyPdcchStatPrms_t* cfg_static_params) :
    static_params(cfg_static_params),
    dynamic_params(nullptr), // also updated in every setup()
    m_LinearAlloc(getBufferSize(cfg_static_params), &memory_footprint),
    m_component_descrs("PdcchDescr"),
    bulk_desc_async_copy(true),
    /* following parameters are properly updated in every setup() */
    num_coresets(0),
    num_DCIs(1),
    m_encodeRateMatchMultiDCIsArgs{0},
    m_genScramblingSeqArgs{0},
    m_genTfSignalArgs{0},
    h_input_bytes_ptr(nullptr),
    h_x_input_bytes_ptr(nullptr),
    h_c_input_ints_ptr(nullptr),
    m_cudaGraphModeEnabled(true),
    m_kernelSelOption(cfg_static_params->kernelSelOption),
    m_delayUs(cfg_static_params->delayUs)
{

    if(m_kernelSelOption >= PDCCH_MAX_KERNEL_SEL_MODES)
    {
        NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT,  "ERROR: unsupported PDCCH kernel selection mode {}.", m_kernelSelOption);
    }
    
    //Memset all launch configs structs to 0; they will be properly updated on every setup() as needed
    memset(&m_encodeRateMatchMultiDCIsLaunchCfg, 0, sizeof(cuphyEncoderRateMatchMultiDCILaunchCfg_t));
    memset(&m_genScramblingSeqLaunchCfg, 0, sizeof(cuphyGenScramblingSeqLaunchCfg_t));
    memset(&m_genTfSignalLaunchCfg, 0, sizeof(cuphyGenPdcchTfSgnlLaunchCfg_t));
    // m_fusedPdcchTx (cfg/args/node) is zero-initialized at declaration (KernelNodeSlot).

    cfg_static_params->pOutInfo->pMemoryFootprint = &memory_footprint; // update  static parameter field that points to the cuphyMemoryFootprintTracker object for this channel

    // Set kernel functions to nullptr to ensure call to cudaFuncGetSymbol happens only once.
    m_encodeRateMatchMultiDCIsLaunchCfg.kernelNodeParamsDriver.func = nullptr;
    m_genScramblingSeqLaunchCfg.kernelNodeParamsDriver.func         = nullptr;
    //Did not update m_genTfSignalLaunchCfg as it is templated.

    allocateBuffers();
    
    if(m_kernelSelOption != PDCCH_ALL && m_delayUs > 0)
    {
        m_delayKernelArgs[0] = &m_delayUs;
        CUPHY_CHECK(cuphySetDelayKernelNodeParams(&m_delayKernelParamsDriver, &m_delayKernelArgs[0]));
    }
    
#if MANUAL_WORKSPACE
    int pdcch_coreset_params_size     = round_up_to_next<int>(sizeof(PdcchParams) * max_coresets_per_slot, max_alignment);
    int pdcch_dci_params_size         = round_up_to_next<int>(max_DCIs_per_slot * sizeof(cuphyPdcchDciPrm_t), max_alignment);
    int pdcch_input_w_crc_size        = round_up_to_next<int>(CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC * max_DCIs_per_slot, max_alignment);
    int pdcch_pmw_params_size         = round_up_to_next<int>(max_DCIs_per_slot * sizeof(cuphyPdcchPmWOneLayer_t), max_alignment);
    int pdcch_tm_params_size          = round_up_to_next<int>(div_round_up<int>(max_DCIs_per_slot, BITS_PER_UINT8), max_alignment);
    int pdcch_coreset_idx_of_dci_size = round_up_to_next<int>(max_DCIs_per_slot * static_cast<int>(sizeof(int)), max_alignment);
    int total_workspace_size = pdcch_coreset_params_size + pdcch_dci_params_size + pdcch_input_w_crc_size + pdcch_pmw_params_size +
                               pdcch_tm_params_size + pdcch_coreset_idx_of_dci_size;

    d_workspace = make_unique_device<uint8_t>(total_workspace_size, &memory_footprint);
    h_workspace = make_unique_pinned<uint8_t>(total_workspace_size);

    // Properly updated as part of every setup()
    h_coreset_params = nullptr;
    d_coreset_params = nullptr;
    h_dci_params     = nullptr;
    d_dci_params     = nullptr;
    h_pmw_params     = nullptr;
    d_pmw_params     = nullptr;

    h_input_w_crc_bytes = nullptr;
    d_input_w_crc_bytes = nullptr;

    h_dci_tm_info = nullptr;
    d_dci_tm_info = nullptr;

    h_coreset_idx_of_dci = nullptr;
    d_coreset_idx_of_dci = nullptr;
#else
    allocateDescr(); //Allocate Descriptors.
#endif

    d_cidx2uidx_lut = make_unique_device<uint16_t>(PDCCH_POLAR_CIDX2UIDX_LUT_SIZE, &memory_footprint);
    CUDA_CHECK(cudaMemcpy(d_cidx2uidx_lut.get(), PDCCH_POLAR_CIDX2UIDX_LUT_CPU, PDCCH_POLAR_CIDX2UIDX_LUT_SIZE * sizeof(uint16_t), cudaMemcpyHostToDevice));

    if(!MEMSET_IN_FIRST_SETUP)
    {
        CUDA_CHECK(cudaMemset(d_x_tx_bytes, 0, max_tx_bytes * max_DCIs_per_slot));
    }

    // the following call to create graph also helps (by calling cudaGetFuncBySymbol) to move CUDA runtime initialization overhead into channel constructor
    if(m_kernelSelOption==PDCCH_ALL)
    {
        createGraph();
    }
    else if(m_delayUs==0)
    {
        createOffloadingGraph();
    }
    else
    {
        createOffloadingDelayGraph();
    }
#if CUDA_VERSION >= 12000
    CU_CHECK_EXCEPTION(cuGraphInstantiate(&m_graphExec, m_graph, 0));
#else
    CU_CHECK_EXCEPTION(cuGraphInstantiate(&m_graphExec, m_graph, 0, 0, 0));
#endif

    if(PRINT_GPU_MEMORY_CUPHY_CHANNEL == 1)
    {
        memory_footprint.printMemoryFootprint(this, "PDCCH");
    }
}

size_t PdcchTx::getBufferSize(const cuphyPdcchStatPrms_t* pStatParams)
{
    size_t nBytesBuffer = 0;

    max_coresets_per_slot = pStatParams->nMaxCellsPerSlot * CUPHY_PDCCH_N_MAX_CORESETS_PER_CELL;
    max_DCIs_per_slot     = max_coresets_per_slot * CUPHY_PDCCH_MAX_DCIS_PER_CORESET;

    // Reminder: overprovisioned buffer allocation happens only once, in the constructor.
    max_tx_bytes    = CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / BITS_PER_UINT8;
    max_coded_bytes = CUPHY_POLAR_ENC_MAX_CODED_BITS / BITS_PER_UINT8; // max number of coded bits from polar encoder for memory allocation (512 bits = 64B)

    // Buffer allocations for polar encoder (intermediate workspace and output).
    nBytesBuffer += max_coded_bytes * max_DCIs_per_slot;
    nBytesBuffer = round_up_to_next(nBytesBuffer, 128LU); // ensure 128-byte alignment in cuphy::linear_alloc<128, cuphy::device_alloc> m_LinearAlloc

    // The polar encoder currently supports one call per DCI, so the max DCIs multiplier is not needed when run on the same CUDA stream.
    nBytesBuffer += max_tx_bytes * max_DCIs_per_slot;
    nBytesBuffer = round_up_to_next(nBytesBuffer, 128LU); // ensure 128-byte alignment

    // Allocate buffer for scrambling sequence on the GPU
    // Reminder max. scrambling seq. elements (32-bit) per DCI = CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 32; // = (2 * 9 * 6 * 16) /32 where 16 is the max. aggregation level.
    nBytesBuffer += (CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 32) * max_DCIs_per_slot * 4;
    nBytesBuffer = round_up_to_next(nBytesBuffer, 128LU); // ensure 128-byte alignment

    return nBytesBuffer;
}

void PdcchTx::allocateBuffers()
{
    // Reminder: overprovisioned buffer allocation happens only once, in the constructor.

    max_tx_bytes    = CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / BITS_PER_UINT8;
    max_coded_bytes = CUPHY_POLAR_ENC_MAX_CODED_BITS / BITS_PER_UINT8; // max number of coded bits from polar encoder for memory allocation (512 bits = 64B)

    // Buffer allocations for polar encoder (intermediate workspace and output).
    //d_x_coded_bytes = make_unique_device<uint8_t>(max_coded_bytes  * max_DCIs_per_slot);
    d_x_coded_bytes = static_cast<uint8_t*>(m_LinearAlloc.alloc(max_coded_bytes * max_DCIs_per_slot));

    // The polar encoder currently supports one call per DCI, so the max DCIs multiplier is not needed when run on the same CUDA stream.
    d_x_tx_bytes = static_cast<uint8_t*>(m_LinearAlloc.alloc(max_tx_bytes * max_DCIs_per_slot));

    // Allocate buffer for scrambling sequence on the GPU
    // Reminder max. scrambling seq. elements (32-bit) per DCI = CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 32; // = (2 * 9 * 6 * 16) /32 where 16 is the max. aggregation level.
    d_scrambling_seq = static_cast<uint32_t*>(m_LinearAlloc.alloc((CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 32) * max_DCIs_per_slot * 4));
}

void PdcchTx::updateWorkspaceOffsets(int coreset_cnt, int dci_cnt)
{
    workspace_offsets[PDCCH_PARAMS]          = 0;
    workspace_offsets[PDCCH_DCI_PARAMS]      = workspace_offsets[PDCCH_PARAMS] + round_up_to_next<int>(coreset_cnt * sizeof(PdcchParams), max_alignment);
    workspace_offsets[PDCCH_INPUT_W_CRC]     = workspace_offsets[PDCCH_DCI_PARAMS] + round_up_to_next<int>(dci_cnt * sizeof(cuphyPdcchDciPrm_t), max_alignment);
    workspace_offsets[PDCCH_PMW_PARAMS]      = workspace_offsets[PDCCH_INPUT_W_CRC] + round_up_to_next<int>(CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC * dci_cnt, max_alignment);
    workspace_offsets[PDCCH_TM_PARAMS]       = workspace_offsets[PDCCH_PMW_PARAMS] + round_up_to_next<int>(dci_cnt * sizeof(cuphyPdcchPmWOneLayer_t), max_alignment);
    workspace_offsets[PDCCH_DCI_CORESET_IDX] = workspace_offsets[PDCCH_TM_PARAMS] + round_up_to_next<int>(div_round_up<int>(dci_cnt, BITS_PER_UINT8), max_alignment);
    workspace_offsets[N_PDCCH_COMPONENTS]    = workspace_offsets[PDCCH_DCI_CORESET_IDX] + round_up_to_next<int>(dci_cnt * static_cast<int>(sizeof(int)), max_alignment);

    // curently unused
    //int h_input_dims[2] = { CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES, (int)dci_cnt};
    //h_input_bytes_desc = tensor_desc(CUPHY_R_8U, tensor_layout(2, h_input_dims, nullptr));
}

void PdcchTx::updateWorkspacePtrs()
{
    // Workspace layout as follows: h_coreset_params, h_dci_params, h_input_w_crc_bytes, h_pmw_params, h_dci_tm_info following the Component enum order

    h_coreset_params = (PdcchParams*)(h_workspace.get());
    d_coreset_params = (PdcchParams*)(d_workspace.get());

    h_dci_params = (cuphyPdcchDciPrm_t*)(h_workspace.get() + workspace_offsets[PDCCH_DCI_PARAMS]);
    d_dci_params = (cuphyPdcchDciPrm_t*)(d_workspace.get() + workspace_offsets[PDCCH_DCI_PARAMS]);

    h_input_w_crc_bytes = (uint8_t*)(h_workspace.get() + workspace_offsets[PDCCH_INPUT_W_CRC]);
    d_input_w_crc_bytes = (uint8_t*)(d_workspace.get() + workspace_offsets[PDCCH_INPUT_W_CRC]);

    h_pmw_params = (cuphyPdcchPmWOneLayer_t*)(h_workspace.get() + workspace_offsets[PDCCH_PMW_PARAMS]);
    d_pmw_params = (cuphyPdcchPmWOneLayer_t*)(d_workspace.get() + workspace_offsets[PDCCH_PMW_PARAMS]);

    h_dci_tm_info = (uint8_t*)(h_workspace.get() + workspace_offsets[PDCCH_TM_PARAMS]);
    d_dci_tm_info = (uint8_t*)(d_workspace.get() + workspace_offsets[PDCCH_TM_PARAMS]);

    h_coreset_idx_of_dci = reinterpret_cast<int*>(h_workspace.get() + workspace_offsets[PDCCH_DCI_CORESET_IDX]);
    d_coreset_idx_of_dci = reinterpret_cast<int*>(d_workspace.get() + workspace_offsets[PDCCH_DCI_CORESET_IDX]);

    /*NVLOGD_FMT(NVLOG_PDCCH, "h_workspace = {:p}, d_workspace {:p}", (void*)h_workspace.get(), (void*)d_workspace.get());
    for (int i = 0; i < N_PDCCH_COMPONENTS + 1; i++) {
        NVLOGD_FMT(NVLOG_PDCCH, "offset[{}] = {}", i, workspace_offsets[i]);
    }*/
}

void PdcchTx::allocateDescr()
{
    // currently unused
    //int h_input_dims[2] = { CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES, (int)max_DCIs_per_slot};
    //h_input_bytes_desc = tensor_desc(CUPHY_R_8U, tensor_layout(2, h_input_dims, nullptr));

    std::array<size_t, N_PDCCH_COMPONENTS> dynDescrSizeBytes{};
    std::array<size_t, N_PDCCH_COMPONENTS> dynDescrAlignBytes{};

    size_t* pDynDescrSizeBytes  = dynDescrSizeBytes.data();
    size_t* pDynDescrAlignBytes = dynDescrAlignBytes.data();

    //Allocate workspace memory to take advantage of the bulk async copy of the descriptors.

    // Allocate memory for PdcchParams.
    pDynDescrSizeBytes[PDCCH_PARAMS]  = sizeof(PdcchParams) * max_coresets_per_slot;
    pDynDescrAlignBytes[PDCCH_PARAMS] = alignof(PdcchParams);

    pDynDescrSizeBytes[PDCCH_DCI_PARAMS]  = max_DCIs_per_slot * sizeof(cuphyPdcchDciPrm_t);
    pDynDescrAlignBytes[PDCCH_DCI_PARAMS] = alignof(cuphyPdcchDciPrm_t);

    // Allocate memory for input array after CRC attachment. TODO This is currently done on the host.
    pDynDescrSizeBytes[PDCCH_INPUT_W_CRC]  = (CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC)*max_DCIs_per_slot;
    pDynDescrAlignBytes[PDCCH_INPUT_W_CRC] = alignof(uint32_t);

    pDynDescrSizeBytes[PDCCH_PMW_PARAMS]  = max_DCIs_per_slot * sizeof(cuphyPdcchPmWOneLayer_t);
    pDynDescrAlignBytes[PDCCH_PMW_PARAMS] = alignof(cuphyPdcchPmWOneLayer_t);

    // PDCCH_TM_PARAMS is using a bit per byte.
    pDynDescrSizeBytes[PDCCH_TM_PARAMS]  = div_round_up<int>(max_DCIs_per_slot, BITS_PER_UINT8);
    pDynDescrAlignBytes[PDCCH_TM_PARAMS] = alignof(uint8_t);

    pDynDescrSizeBytes[PDCCH_DCI_CORESET_IDX]  = max_DCIs_per_slot * sizeof(int);
    pDynDescrAlignBytes[PDCCH_DCI_CORESET_IDX] = alignof(int);

    // currently unused
    //int input_w_crc_desc_dims[2] = { CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC, (int)max_DCIs_per_slot};
    //input_w_crc_desc = tensor_desc(CUPHY_R_8U, tensor_layout(2, input_w_crc_desc_dims, nullptr));

    m_component_descrs.alloc(dynDescrSizeBytes, dynDescrAlignBytes, &memory_footprint);
    //m_component_descrs.displayDescrSizes();

    // Added for convenience
    h_coreset_params = (PdcchParams*)m_component_descrs.getCpuStartAddrs()[PDCCH_PARAMS];
    d_coreset_params = (PdcchParams*)m_component_descrs.getGpuStartAddrs()[PDCCH_PARAMS];

    h_dci_params = (cuphyPdcchDciPrm_t*)m_component_descrs.getCpuStartAddrs()[PDCCH_DCI_PARAMS];
    d_dci_params = (cuphyPdcchDciPrm_t*)m_component_descrs.getGpuStartAddrs()[PDCCH_DCI_PARAMS];

    h_pmw_params = (cuphyPdcchPmWOneLayer_t*)m_component_descrs.getCpuStartAddrs()[PDCCH_PMW_PARAMS];
    d_pmw_params = (cuphyPdcchPmWOneLayer_t*)m_component_descrs.getGpuStartAddrs()[PDCCH_PMW_PARAMS];

    h_input_w_crc_bytes = (uint8_t*)m_component_descrs.getCpuStartAddrs()[PDCCH_INPUT_W_CRC];
    d_input_w_crc_bytes = (uint8_t*)m_component_descrs.getGpuStartAddrs()[PDCCH_INPUT_W_CRC];

    h_dci_tm_info = (uint8_t*)m_component_descrs.getCpuStartAddrs()[PDCCH_TM_PARAMS];
    d_dci_tm_info = (uint8_t*)m_component_descrs.getGpuStartAddrs()[PDCCH_TM_PARAMS];

    h_coreset_idx_of_dci = reinterpret_cast<int*>(m_component_descrs.getCpuStartAddrs()[PDCCH_DCI_CORESET_IDX]);
    d_coreset_idx_of_dci = reinterpret_cast<int*>(m_component_descrs.getGpuStartAddrs()[PDCCH_DCI_CORESET_IDX]);

} // some derived params are updated.

void PdcchTx::createGraph()
{
    CU_CHECK_EXCEPTION(cuGraphCreate(&m_graph, 0));
    // Add node(s). Initially start with some kernel parameters, and at setup do the updating
    // Set empty graph kernel nodes with the appropriate argument count (all pointers) to avoid dynamic
    // memory allocation during graph kernel node update. encodeRateMatchMultipleDCIsKernel uses 6 parameters,
    // genPdcchTfSignalKernel and fusedPdcchTxKernel use 7 (hence kernelParams[7]).
    void* arg;
    void* kernelParams[7] = {&arg, &arg, &arg, &arg, &arg, &arg, &arg}; // use max. number of kernel args for array size

#if PDCCH_FUSED_KERNEL
    // Single fused kernel node: encode + rate match + scrambling + TF signal embedding.
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode7ParamsDriver, 7, &kernelParams[0]));
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_fusedPdcchTx.node, m_graph, nullptr, 0, &m_emptyNode7ParamsDriver));
#else
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode6ParamsDriver, 6, &kernelParams[0]));
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode2ParamsDriver, 2, &kernelParams[0]));
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode7ParamsDriver, 7, &kernelParams[0]));

    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_encodeRateMatchMultiDCIsNode, m_graph, nullptr, 0, &m_emptyNode6ParamsDriver));
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_genScramblingSeqNode, m_graph, &m_encodeRateMatchMultiDCIsNode, 1, &m_emptyNode2ParamsDriver));
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_genTfSignalNode, m_graph, &m_genScramblingSeqNode, 1, &m_emptyNode7ParamsDriver));
    // The 3rd arg of the last kernel is uint32_t but using void* (whose size is greater), so the code is more generalizable.
#endif
}

void PdcchTx::updateGraph()
{
#if PDCCH_FUSED_KERNEL
    CU_CHECK_EXCEPTION_W_TAG(NVLOG_PDCCH, cuGraphExecKernelNodeSetParams(m_graphExec, m_fusedPdcchTx.node, &(m_fusedPdcchTx.cfg.kernelNodeParamsDriver)));
#else
    CU_CHECK_EXCEPTION_W_TAG(NVLOG_PDCCH, cuGraphExecKernelNodeSetParams(m_graphExec, m_encodeRateMatchMultiDCIsNode, &(m_encodeRateMatchMultiDCIsLaunchCfg.kernelNodeParamsDriver)));
    CU_CHECK_EXCEPTION_W_TAG(NVLOG_PDCCH, cuGraphExecKernelNodeSetParams(m_graphExec, m_genScramblingSeqNode, &(m_genScramblingSeqLaunchCfg.kernelNodeParamsDriver)));
    CU_CHECK_EXCEPTION_W_TAG(NVLOG_PDCCH, cuGraphExecKernelNodeSetParams(m_graphExec, m_genTfSignalNode, &(m_genTfSignalLaunchCfg.kernelNodeParamsDriver)));
#endif
}

void PdcchTx::createOffloadingGraph()
{
    CU_CHECK_EXCEPTION(cuGraphCreate(&m_graph, 0));
    // Add node(s). Initially start with some kernel parameters, and at setup do the updating
    // Set empty graph kernel nodes with the appropriate argument count (all pointers) to avoid dynamic
    // memory allocation during graph kernel node update. encodeRateMatchMultipleDCIsKernel uses 6 parameters.
    void* arg;
    void* kernelParams[7] = {&arg, &arg, &arg, &arg, &arg, &arg, &arg}; // use max. number of kernel args for array size
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode2ParamsDriver, 2, &kernelParams[0]));
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode6ParamsDriver, 6, &kernelParams[0]));
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode7ParamsDriver, 7, &kernelParams[0]));

    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_genTfSignalNode, m_graph, nullptr, 0, &m_emptyNode7ParamsDriver));
    // The 3rd arg of the last kernel is uint32_t but using void* (whose size is greater), so the code is more generalizable.
}

void PdcchTx::updateOffloadingGraph()
{
    CU_CHECK_EXCEPTION_W_TAG(NVLOG_PDCCH, cuGraphExecKernelNodeSetParams(m_graphExec, m_genTfSignalNode, &(m_genTfSignalLaunchCfg.kernelNodeParamsDriver)));
}

void PdcchTx::createOffloadingDelayGraph()
{
    CU_CHECK_EXCEPTION(cuGraphCreate(&m_graph, 0));
    // Add node(s). Initially start with some kernel parameters, and at setup do the updating
    // Set empty graph kernel nodes with the appropriate argument count (all pointers) to avoid dynamic
    // memory allocation during graph kernel node update. encodeRateMatchMultipleDCIsKernel uses 6 parameters.
    void* arg;
    void* kernelParams[7] = {&arg, &arg, &arg, &arg, &arg, &arg, &arg}; // use max. number of kernel args for array size
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode2ParamsDriver, 2, &kernelParams[0]));
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode6ParamsDriver, 6, &kernelParams[0]));
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode7ParamsDriver, 7, &kernelParams[0]));
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode1ParamsDriver, 1, &kernelParams[0]));

    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_delayNode, m_graph, nullptr, 0, &m_emptyNode1ParamsDriver));
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_genTfSignalNode, m_graph, &m_delayNode, 1, &m_emptyNode7ParamsDriver));
    // The 3rd arg of the last kernel is uint32_t but using void* (whose size is greater), so the code is more generalizable.
}

void PdcchTx::updateOffloadingDelayGraph()
{
    CU_CHECK_EXCEPTION_W_TAG(NVLOG_PDCCH, cuGraphExecKernelNodeSetParams(m_graphExec, m_delayNode, &m_delayKernelParamsDriver));
    CU_CHECK_EXCEPTION_W_TAG(NVLOG_PDCCH, cuGraphExecKernelNodeSetParams(m_graphExec, m_genTfSignalNode, &(m_genTfSignalLaunchCfg.kernelNodeParamsDriver)));
}

template <fmtlog::LogLevel log_level>
void PdcchTx::printPdcchConfig(const cuphyPdcchDynPrms_t& params)
{
    //FIXME could move to cuphy_hdf5 potentially. Could consider adding functions in a way that makes it easier to see which DCIs belong to which CORESET
    NVLOG_FMT(log_level, NVLOG_PDCCH, "PDCCH TX pipeline with {} cells", params.nCells);
    NVLOG_FMT(log_level, NVLOG_PDCCH, "PDCCH TX pipeline with {} precoded DCIs", params.nPrecodingMatrices);
    NVLOG_FMT(log_level, NVLOG_PDCCH, "PDCCH TX pipeline: DCI parameters across all {} DCIs: ", params.nDci);
    for(int i = 0; i < params.nDci; i++)
    {
        const cuphyPdcchDciPrm_t& dci_params = params.pDciPrms[i];
        NVLOG_FMT(log_level, NVLOG_PDCCH, "PDCCH parameters for DCI {}: ", i);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "------------------------------------------------------");
        NVLOG_FMT(log_level, NVLOG_PDCCH, "Npayload:     {:5d}", dci_params.Npayload);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "rntiCrc:      {:5d}", dci_params.rntiCrc);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "rntiBits:     {:5d}", dci_params.rntiBits);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "dmrsId:       {:5d}", dci_params.dmrs_id);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "aggrL:        {:5d}", dci_params.aggr_level);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "cceIdx:       {:5d}", dci_params.cce_index);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "beta_qam:     {: 5.3f}", dci_params.beta_qam);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "beta_dmrs:    {: 5.3f}", dci_params.beta_dmrs);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "enablePrcdBf: {:5d}", dci_params.enablePrcdBf);
        if(dci_params.enablePrcdBf)
        {
            NVLOG_FMT(log_level, NVLOG_PDCCH, "pmwPrmIdx:    {:5d}", dci_params.pmwPrmIdx);
            NVLOG_FMT(log_level, NVLOG_PDCCH, "nPorts :      {:5d}", params.pPmwParams[dci_params.pmwPrmIdx].nPorts);

            std::stringstream matrix_row;
            matrix_row.precision(5);
            matrix_row << "Precoding Matrix:      ";
            for(int idx = 0; idx < params.pPmwParams[dci_params.pmwPrmIdx].nPorts; idx++)
            {
                if(idx != 0) matrix_row << ", ";
                matrix_row << std::fixed << "{" << (float)params.pPmwParams[dci_params.pmwPrmIdx].matrix[idx].x << ", " << (float)params.pPmwParams[dci_params.pmwPrmIdx].matrix[idx].y << "}";
            }
            NVLOG_FMT(log_level, NVLOG_PDCCH, "{}", matrix_row.str());
        }
    }

    NVLOG_FMT(log_level, NVLOG_PDCCH, "PDCCH TX pipeline: Coreset parameters across all {} coresets: ", params.nCoresets);
    for(int i = 0; i < params.nCoresets; i++)
    {
        NVLOG_FMT(log_level, NVLOG_PDCCH, "PDCCH parameters for Coreset {}: ", i);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "------------------------------------------------------");
        const cuphyPdcchCoresetDynPrm_t& coreset_params = params.pCoresetDynPrm[i];
        NVLOG_FMT(log_level, NVLOG_PDCCH, "slot_number:     {:5d}", coreset_params.slot_number);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "start_rb:        {:5d}", coreset_params.start_rb);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "start_sym:       {:5d}", coreset_params.start_sym);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "n_sym:           {:5d}", coreset_params.n_sym);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "bundle_size:     {:5d}", coreset_params.bundle_size);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "interleaver_size:{:5d}", coreset_params.interleaver_size);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "shift_index:     {:5d}", coreset_params.shift_index);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "interleaved:     {:5d}", coreset_params.interleaved);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "n_f:             {:5d}", coreset_params.n_f);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "freq_domain_resource: {:#x}", coreset_params.freq_domain_resource);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "Coreset type:    {:5d}", coreset_params.coreset_type);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "Test model:      {:5d}", coreset_params.testModel);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "nDci:            {:5d}", coreset_params.nDci);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "dciStartIdx:     {:5d}", coreset_params.dciStartIdx);
        NVLOG_FMT(log_level, NVLOG_PDCCH, "slotBufferIdx:   {:5d}", coreset_params.slotBufferIdx);
    }
}

cuphyStatus_t PdcchTx::checkConfig(PdcchParams& params)
{
    //TODO Add sanity check about values of specific config params
    if(params.start_sym >= OFDM_SYMBOLS_PER_SLOT)
    {
        NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "Unsupported start symbol {}!", params.start_sym);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if((params.n_sym == 0) || (params.n_sym > 3))
    {
        NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "Unsupported number of symbols {}! Should be 1, 2 or 3.", params.n_sym);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if(params.interleaved > 1)
    {
        NVLOGD_FMT(NVLOG_PDCCH, "Interleaved mode cannot be greater than 1. Mapping to 1.");
        params.interleaved = 1;
    }

    if(params.interleaved == 0)
    { // non-interleaved mode
        if(params.bundle_size != 6)
        {
            NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "Bundle size has to be 6 for non-interleaved mode, but was {}!.", params.bundle_size);
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
        if(params.interleaver_size != 0)
        {
            NVLOGD_FMT(NVLOG_PDCCH, "interleaver size is N/A in non-interleaved mode. Setting to 0.");
            params.interleaver_size = 0;
        }
        if(params.shift_index != 0)
        {
            NVLOGD_FMT(NVLOG_PDCCH, "shift index size is N/A in non-interleaved mode. Setting to 0.");
            params.shift_index = 0;
        }
    }
    else
    { // interleaved mode
        if((params.n_sym == 3) && ((params.bundle_size != 3) && (params.bundle_size != 6)))
        {
            NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "Bundle size has to be 3 or 6 for 3 symbols in interleaved mode, but was {}!", params.bundle_size);
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
        else if((params.n_sym != 3) && ((params.bundle_size != 2) && (params.bundle_size != 6)))
        {
            NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "Bundle size has to be 2 or 6 for 1 or 2 symbols in interleaved mode, but was {}!", params.bundle_size);
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
        if((params.interleaver_size != 2) && (params.interleaver_size != 3) && (params.interleaver_size != 6))
        {
            NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "interleaver size must be 2, 3 or 6 in interleaved mode, but was {}.", params.interleaver_size);
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
    }
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t PdcchTx::preparePdcch(cudaStream_t cuda_strm)
{
    // Generate PDCCH CRC output and scrambling sequence
    cuphyStatus_t status = cuphyPdcchPipelinePrepare(h_input_w_crc_bytes,
                                                     nullptr /*input_w_crc_desc.handle()*/,
                                                     h_input_bytes_ptr,
                                                     nullptr /*h_input_bytes_desc.handle()*/,
                                                     num_coresets,
                                                     num_DCIs,
                                                     h_coreset_params,
                                                     h_dci_params,
                                                     h_dci_tm_info,
                                                     &m_encodeRateMatchMultiDCIsLaunchCfg,
                                                     &m_genScramblingSeqLaunchCfg,
                                                     &m_genTfSignalLaunchCfg,
                                                     cuda_strm);
#if PDCCH_FUSED_KERNEL
    if((status == CUPHY_STATUS_SUCCESS) && (m_kernelSelOption == PDCCH_ALL))
    {
        // Needs the derived coreset params (rb_coreset, n_sym) computed by the prepare call above.
        status = cuphyPdcchFusedTxKernelSelect(&m_fusedPdcchTx.cfg, num_coresets, num_DCIs, h_coreset_params);
    }
#endif
    return status;
}

cuphyStatus_t PdcchTx::expandParameters(cuphyPdcchDynPrms_t* dyn_params,
                                        cudaStream_t         cuda_strm)
{
    if(dyn_params == nullptr)
    {
        NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "cuphySetupPdcchTx() error: cuphyPdcchDynPrms_t nullptr!");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if(dyn_params->pCoresetDynPrm == nullptr)
    {
        NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "cuphySetupPdcchTx() error: cuphyPdcchDynPrms_t's pCoresetDynPrm is nullptr!");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    num_DCIs = dyn_params->nDci; // Number of DCIs across all coresets. Originally set to 1 in the constructor
    if(num_DCIs > max_DCIs_per_slot)
    {
        NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "PDCCH setup: Buffer allocation was for fewer DCIs ({}) than the received {}. Update CUPHY_PDCCH_MAX_DCIS_PER_CORESET, CUPHY_PDCCH_N_MAX_CORESET_PER_CELL or cuphyPdcchStatPrms_t.nMaxCellsPerSlot field.", max_DCIs_per_slot, num_DCIs);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    num_coresets = dyn_params->nCoresets;
    if(num_coresets > max_coresets_per_slot)
    {
        NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "PDCCH setup: Buffer allocation was for fewer coresets ({}) than the received {}. Update CUPHY_PDCCH_N_MAX_CORESET_PER_CELL or cuphyPdcchStatPrms_t.nMaxCellsPerSlot field.", max_coresets_per_slot, num_coresets);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if(dyn_params->nCells > static_params->nMaxCellsPerSlot)
    {
        NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "PDCCH setup: Buffer allocation was for fewer max. cells per slot ({}) than the received {}. Update the cuphyPdcchStatPrms_t.nMaxCellsPerSlot field.", static_params->nMaxCellsPerSlot, dyn_params->nCells);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    // The buffer allocation, workspace offset computation (for MANUAL_WORKSPACE) assume nPrecodingMatrices <= nDci. These buffers are involved in an h2h copy too.
    if(dyn_params->nPrecodingMatrices > dyn_params->nDci)
    {
        NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "PDCCH setup: nPrecodingMatrices ({}) are expected to be <= nDci ({})", dyn_params->nPrecodingMatrices, dyn_params->nDci);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

#if MANUAL_WORKSPACE
    updateWorkspaceOffsets(num_coresets, num_DCIs);
    updateWorkspacePtrs();
#endif

#if MEMSET_IN_FIRST_SETUP
    /* Reset the transmission bits for all DCIs, since there are some atomicAnd operations taking place as part of the rate-matching function in polar encoder
       when nTxBits < nCodedBits. This memset operation is not necessary for functional correctness, as uninitialized contents will not affect computation,
       but avoids a compute-sanitizer initcheck error.
       If resetting on every slot, one can simply memset num_DCIs max_tx_bytes, otherwise if we reset only once (as is the case below),
       then we need to memset max_DCIs_per_slot max_tx_bytes bytes.
       Can also move this operation to the constructor but would need to use the synchronous memset version as no stream is known there. */
    if(first_setup)
    {
        CUDA_CHECK(cudaMemsetAsync(d_x_tx_bytes, 0, max_tx_bytes * max_DCIs_per_slot, cuda_strm));
        first_setup = false;
    }
#endif

#if PRINT_CONFIG
    printPdcchConfig<fmtlog::DBG>(*dyn_params);
#endif
    //The extra host to host copy is needed if we want the respective H2D copy to be part of the single bulk async. copy.
    memcpy(h_dci_params, dyn_params->pDciPrms, sizeof(cuphyPdcchDciPrm_t) * num_DCIs);
    if(dyn_params->nPrecodingMatrices)
    {
        memcpy(h_pmw_params, dyn_params->pPmwParams, sizeof(cuphyPdcchPmWOneLayer_t) * dyn_params->nPrecodingMatrices);
    }

    //Copy config params to PDCCH params to be able to fill in some extra fields
    for(int coreset = 0; coreset < num_coresets; coreset++)
    {
        h_coreset_params[coreset].n_f                  = dyn_params->pCoresetDynPrm[coreset].n_f;
        h_coreset_params[coreset].slot_number          = dyn_params->pCoresetDynPrm[coreset].slot_number;
        h_coreset_params[coreset].start_rb             = dyn_params->pCoresetDynPrm[coreset].start_rb;
        h_coreset_params[coreset].start_sym            = dyn_params->pCoresetDynPrm[coreset].start_sym;
        h_coreset_params[coreset].n_sym                = dyn_params->pCoresetDynPrm[coreset].n_sym;
        h_coreset_params[coreset].bundle_size          = dyn_params->pCoresetDynPrm[coreset].bundle_size;
        h_coreset_params[coreset].interleaver_size     = dyn_params->pCoresetDynPrm[coreset].interleaver_size;
        h_coreset_params[coreset].shift_index          = dyn_params->pCoresetDynPrm[coreset].shift_index;
        h_coreset_params[coreset].interleaved          = dyn_params->pCoresetDynPrm[coreset].interleaved;
        h_coreset_params[coreset].num_dl_dci           = dyn_params->pCoresetDynPrm[coreset].nDci;
        h_coreset_params[coreset].freq_domain_resource = dyn_params->pCoresetDynPrm[coreset].freq_domain_resource;
        h_coreset_params[coreset].coreset_type         = dyn_params->pCoresetDynPrm[coreset].coreset_type;

        h_coreset_params[coreset].slotBufferIdx  = dyn_params->pCoresetDynPrm[coreset].slotBufferIdx;
        h_coreset_params[coreset].slotBufferAddr = dyn_params->pDataOut->pTDataTx[dyn_params->pCoresetDynPrm[coreset].slotBufferIdx].pAddr;
        h_coreset_params[coreset].dciStartIdx    = dyn_params->pCoresetDynPrm[coreset].dciStartIdx;
        h_coreset_params[coreset].testModel      = dyn_params->pCoresetDynPrm[coreset].testModel;
    }

    // Populate coreset_idx_of_dci array. Do it here to avoid modifying API of "cuphyPdcchPipelinePrepare"
    memset(h_coreset_idx_of_dci, 0, num_DCIs * sizeof(h_coreset_idx_of_dci[0]));
    for(int coreset_idx = 0; coreset_idx < num_coresets; coreset_idx++)
    {
        for(int k = 0; k < h_coreset_params[coreset_idx].num_dl_dci; k++)
        {
            const int DCI_id = h_coreset_params[coreset_idx].dciStartIdx + k;
            if(DCI_id >= 0 && DCI_id < num_DCIs)
            {
                h_coreset_idx_of_dci[DCI_id] = coreset_idx;
            }
            else
            {
                NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "Invalid DCI_id {}! (range: [{},{}]), coreset_idx {}, dciStartIdx {}, k {}.", 
                    DCI_id, 0, num_DCIs - 1, coreset_idx, h_coreset_params[coreset_idx].dciStartIdx, k);
                return CUPHY_STATUS_INVALID_ARGUMENT;
            }
        }
    }

    // NB Some derived params, e.g., rb_coreset etc. are computed in Setup and the h_coreset_params are updated there.

    //Sanity checks for config. params
#if 0
    cuphyStatus_t check_config_status = checkConfig(*h_coreset_params);
    if (check_config_status != CUPHY_STATUS_SUCCESS) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
#endif

    if((dyn_params->pDataIn == nullptr) || (dyn_params->pDataIn->pDciInput == nullptr))
    {
        NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "cuphySetupPdcchTx() error: input buffer pDciInput nullptr!");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    h_input_bytes_ptr = dyn_params->pDataIn->pDciInput; // pointer to uint8_t buffer. Single buffer for all DCIs across all coresets.
    h_x_input_bytes_ptr = dyn_params->pDataIn->pXInput;
    h_c_input_ints_ptr  = dyn_params->pDataIn->pCInput;
    if(m_kernelSelOption == PDCCH_NO_POLAR_ENCODER)
    {
        if((h_x_input_bytes_ptr == nullptr) || (h_c_input_ints_ptr == nullptr))
        {
            NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "cuphySetupPdcchTx() error: pXInput or pCInput nullptr in PDCCH_NO_POLAR_ENCODER mode!");
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
    }


    // Prepare PDCCH
    cuphyStatus_t status = preparePdcch(cuda_strm);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "Error for cuphyPdcchPipelinePrepare");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

#if MANUAL_WORKSPACE
    CUDA_CHECK(cudaMemcpyAsync(d_workspace.get(), h_workspace.get(), workspace_offsets[N_PDCCH_COMPONENTS], cudaMemcpyHostToDevice, dynamic_params->cuStream));
    if(m_kernelSelOption==PDCCH_NO_POLAR_ENCODER)
    {
        CUDA_CHECK(cudaMemcpyAsync(d_x_tx_bytes, h_x_input_bytes_ptr, CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / BITS_PER_UINT8 * num_DCIs, cudaMemcpyHostToDevice, dynamic_params->cuStream));
        CUDA_CHECK(cudaMemcpyAsync(d_scrambling_seq, h_c_input_ints_ptr, CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 32 * num_DCIs *4, cudaMemcpyHostToDevice, dynamic_params->cuStream));
    }
#else
    // Bulk async H2D copy for all workspaces
    if(bulk_desc_async_copy)
    {
        m_component_descrs.asyncCpuToGpuCpy(cuda_strm);
    }
    
#endif
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t PdcchTx::setup(cuphyPdcchDynPrms_t* pDynPrms)
{
    dynamic_params       = pDynPrms; // Needed to retrieve CUDA stream etc.
    cuphyStatus_t status = expandParameters(pDynPrms, pDynPrms->cuStream);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        return status;
    }

    pDynPrms->chan_graph = &m_graph;

    //---------------------------------------------
    // set kernel args for encodeRateMatchMultipleDCIsKernel()
    // The polar encoder first encodes and the rate-matches the d_input_w_crc input for each DCI.
    // Its output for all num_DCIs DCIs is in d_x_tx buffer.
    // The d_x_coded is an intermediate buffer used internally by the polar encoder.
    // The d_input_w_crc assumes the 24-bit CRC has already been computed on each DCI's payload and that
    m_encodeRateMatchMultiDCIsArgs[0] = d_input_w_crc_bytes;
    m_encodeRateMatchMultiDCIsArgs[1] = d_x_coded_bytes;
    m_encodeRateMatchMultiDCIsArgs[2] = d_x_tx_bytes;
    m_encodeRateMatchMultiDCIsArgs[3] = d_dci_params;
    m_encodeRateMatchMultiDCIsArgs[4] = d_dci_tm_info;
    m_encodeRateMatchMultiDCIsArgs[5] = d_cidx2uidx_lut.get();

    m_encodeRateMatchMultiDCIsLaunchCfg.kernelArgs[0] = &m_encodeRateMatchMultiDCIsArgs[0];
    m_encodeRateMatchMultiDCIsLaunchCfg.kernelArgs[1] = &m_encodeRateMatchMultiDCIsArgs[1];
    m_encodeRateMatchMultiDCIsLaunchCfg.kernelArgs[2] = &m_encodeRateMatchMultiDCIsArgs[2];
    m_encodeRateMatchMultiDCIsLaunchCfg.kernelArgs[3] = &m_encodeRateMatchMultiDCIsArgs[3];
    m_encodeRateMatchMultiDCIsLaunchCfg.kernelArgs[4] = &m_encodeRateMatchMultiDCIsArgs[4];
    m_encodeRateMatchMultiDCIsLaunchCfg.kernelArgs[5] = &m_encodeRateMatchMultiDCIsArgs[5];

    m_encodeRateMatchMultiDCIsLaunchCfg.kernelNodeParamsDriver.kernelParams = m_encodeRateMatchMultiDCIsLaunchCfg.kernelArgs;

    // set kernel args for genScramblingSeqKernel()
    m_genScramblingSeqArgs[0] = d_scrambling_seq;
    m_genScramblingSeqArgs[1] = d_dci_params;

    m_genScramblingSeqLaunchCfg.kernelArgs[0] = &m_genScramblingSeqArgs[0];
    m_genScramblingSeqLaunchCfg.kernelArgs[1] = &m_genScramblingSeqArgs[1];

    m_genScramblingSeqLaunchCfg.kernelNodeParamsDriver.kernelParams = m_genScramblingSeqLaunchCfg.kernelArgs;

    // set kernel args for genPdcchTfSignalKernel()
    // The kernel populates the appropriate REs in the corresponding output tensors (addr. in d_coreset_params).
    // d_x_tx is the polar encoder's output that includes the transmit bits for all DCIs;
    // d_scrambling_seq is the scrambling sequence generated via the previous kernel.
    // For both d_x_tx and d_scrambling_seq the order of DCIs is the same as in d_dci_params.
    m_genTfSignalArgs[0] = d_x_tx_bytes;
    m_genTfSignalArgs[1] = d_scrambling_seq;
    m_genTfSignalArgs[2] = &num_coresets;
    m_genTfSignalArgs[3] = d_coreset_params;
    m_genTfSignalArgs[4] = d_dci_params;
    m_genTfSignalArgs[5] = d_pmw_params;
    m_genTfSignalArgs[6] = d_coreset_idx_of_dci;

    m_genTfSignalLaunchCfg.kernelArgs[0] = &m_genTfSignalArgs[0];
    m_genTfSignalLaunchCfg.kernelArgs[1] = &m_genTfSignalArgs[1];
    m_genTfSignalLaunchCfg.kernelArgs[2] = m_genTfSignalArgs[2]; // &num_coresets
    m_genTfSignalLaunchCfg.kernelArgs[3] = &m_genTfSignalArgs[3];
    m_genTfSignalLaunchCfg.kernelArgs[4] = &m_genTfSignalArgs[4];
    m_genTfSignalLaunchCfg.kernelArgs[5] = &m_genTfSignalArgs[5];
    m_genTfSignalLaunchCfg.kernelArgs[6] = &m_genTfSignalArgs[6];

    m_genTfSignalLaunchCfg.kernelNodeParamsDriver.kernelParams = m_genTfSignalLaunchCfg.kernelArgs;

#if PDCCH_FUSED_KERNEL
    // set kernel args for fusedPdcchTxKernel(): encode+rate-match inputs plus the TF signal
    // embedding params. The tx bits and scrambling sequence stay in shared memory, so
    // d_x_tx_bytes / d_scrambling_seq / d_x_coded_bytes are not passed.
    m_fusedPdcchTx.args = {d_input_w_crc_bytes,
                           d_dci_params,
                           d_dci_tm_info,
                           d_cidx2uidx_lut.get(),
                           d_coreset_params,
                           d_pmw_params,
                           d_coreset_idx_of_dci};
    m_fusedPdcchTx.bindArgs();
#endif
    //---------------------------------------------

    //executable graph setup
    m_cudaGraphModeEnabled = (pDynPrms->procModeBmsk & PDCCH_PROC_MODE_GRAPHS) ? true : false;
    if(m_cudaGraphModeEnabled)
    {
        if(m_kernelSelOption==PDCCH_ALL)
        {
            updateGraph();
        }
        else if(m_delayUs==0)
        {
            updateOffloadingGraph();
        }
        else
        {
            updateOffloadingDelayGraph();
        }
    }
    return CUPHY_STATUS_SUCCESS;
}

PdcchTx::~PdcchTx()
{
    CUDA_CHECK_NO_THROW(cudaGraphDestroy(m_graph));
    CUDA_CHECK_NO_THROW(cudaGraphExecDestroy(m_graphExec));
}

// Generate PDCCH QAM and DMRS symbols and map them to subcarriers
cuphyStatus_t PdcchTx::run(const cudaStream_t& cuda_strm)
{
    if(m_cudaGraphModeEnabled)
    {
        MemtraceDisableScope md; // Disable temporarily
        CUresult             e = cuGraphLaunch(m_graphExec, cuda_strm);
        if(e != CUDA_SUCCESS)
        {
            const char* pErrStr;
            CUresult    str_e = cuGetErrorString(e, &pErrStr);
            NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "Invalid graph launch for PDCCH ({}, {}).", e, (str_e == CUDA_SUCCESS) ? pErrStr : "CUDA error");
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
    }
    else
    {
#if PDCCH_FUSED_KERNEL
        if(m_kernelSelOption == PDCCH_ALL)
        {
            // Single fused kernel launch (encode + rate match + scrambling + TF signal) for all DCIs across all coresets.
            CUresult e = launch_kernel(m_fusedPdcchTx.cfg.kernelNodeParamsDriver, cuda_strm);
            if(e != CUDA_SUCCESS)
            {
                NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "PDCCH fused kernel launch failed ({}) for {} coresets / {} DCIs.", static_cast<int>(e), num_coresets, num_DCIs);
                return CUPHY_STATUS_INTERNAL_ERROR;
            }
            return CUPHY_STATUS_SUCCESS;
        }
#endif
        // Encoding and scrambling-sequence generation run on the GPU only in PDCCH_ALL mode.
        // In PDCCH_NO_POLAR_ENCODER (offload) mode the tx bits and scrambling sequence were
        // H2D-copied from the host in expandParameters(); launching the kernels here would
        // overwrite the offloaded data (mirrors createOffloadingGraph(), which contains only
        // the TF-signal node).
        if(m_kernelSelOption == PDCCH_ALL)
        {
            // Single kernel launch (encodeRateMatchMultipleDCIsKernel) for all DCIs in the slot (across all coresets).
            // the bit order within a byte has been reversed compared to the original input buffer.
            // This work is done in cuphyPdcchPipelinePrepare on the host. TODO We could absorb some of it on the GPU too.
            CUresult e_enc = launch_kernel(m_encodeRateMatchMultiDCIsLaunchCfg.kernelNodeParamsDriver, cuda_strm);
            if(e_enc != CUDA_SUCCESS) 
            {
                return CUPHY_STATUS_INTERNAL_ERROR;
            }

            // Kernel that generates the scrambling sequence in device memory. Moved here from setup.
            // Currently a separate kernel, but could be fused with another one.
            // Every thread block corresponds to a DCI. Every thread in a thread block fills in an uint32_t element
            // of the scrambling sequence. The max. number of tx uint32_t elements (rounded up) is 54, thus the 2-warps thread block.
            e_enc = launch_kernel(m_genScramblingSeqLaunchCfg.kernelNodeParamsDriver, cuda_strm);
            if(e_enc != CUDA_SUCCESS) 
            {
                return CUPHY_STATUS_INTERNAL_ERROR;
            }
        }

        // Single kernel launch (genPdcchTfSignal) for all DCIs across all coresets.
        CUresult e = launch_kernel(m_genTfSignalLaunchCfg.kernelNodeParamsDriver, cuda_strm);
        if(e != CUDA_SUCCESS) return CUPHY_STATUS_INTERNAL_ERROR;
    }

    return CUPHY_STATUS_SUCCESS;
}