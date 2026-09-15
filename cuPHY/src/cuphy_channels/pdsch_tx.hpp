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

#if !defined(PDSCH_TX_HPP_INCLUDED_)
#define PDSCH_TX_HPP_INCLUDED_

#include "cuphy.h"
#include "cuphy_api.h"
#include "cuphy.hpp"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "pdsch_dmrs/pdsch_dmrs.hpp"
#include "util.hpp"
#include <iostream>
#include <cstdlib>
#include <string>
#include "empty_kernels.hpp"

#define PDSCH_USE_BATCHED_MEMCPY 1 // Enables batches async memcpy along with a static parameter field.  Set to 0, to disable just this specific batched memcpy.
                                   // Provided as a separate knob for convenience, as a single yaml parameter controls the relevant static parameters for all channels/batched mem. copies in cuBB tests.

struct cuphyPdschTx{};

class PdschTx : public cuphyPdschTx {

public:
    enum Component
    {
        //PDSCH COMPONENTS
        PDSCH_CSIRS_PREP    = 0,
        PDSCH_CRC_PREP      = PDSCH_CSIRS_PREP + 1,
        PDSCH_CRC           = PDSCH_CRC_PREP + 1,
        PDSCH_LDPC          = PDSCH_CRC + 1,
        PDSCH_RM            = PDSCH_LDPC + 1,
        PDSCH_MODULATION    = PDSCH_RM + 1,
        PDSCH_DMRS          = PDSCH_MODULATION + 1,

        //PDSCH WORKSPACES
        PDSCH_PER_TB_PARAMS = PDSCH_DMRS + 1,
        PDSCH_RM_WORKSPACE  = PDSCH_PER_TB_PARAMS + 1,
        PDSCH_DMRS_PARAMS   = PDSCH_RM_WORKSPACE + 1,
        PDSCH_LDPC_WORKSPACE  = PDSCH_DMRS_PARAMS + 1,
        PDSCH_TB_CRCS       = PDSCH_LDPC_WORKSPACE + 1,
        PDSCH_UE_GRP_WORKSPACE = PDSCH_TB_CRCS + 1,
        N_PDSCH_COMPONENTS  = PDSCH_UE_GRP_WORKSPACE + 1
    };

    enum CsirsComponent
    {
        N_CSIRS_COMPONENTS = 3
    };

    /**
     * @brief: Construct PdschTx object. Allocate overprovisioned buffers, descriptors etc.
     * @param[in] cfg_static_params: pointer to PDSCH static parameters.
     * @param[in] cfg_strm: CUDA stream for kernel launches. Default is the default stream.
     * @param[in] cfg_aas_mode: set pipeline in AAS mode (no layer mapping, modulation, DMRS).
     */
    PdschTx(const cuphyPdschStatPrms_t* cfg_static_params, cudaStream_t cfg_strm=0, bool read_TB_CRC = false, bool cfg_aas_mode=false);

    /**
     * @brief: PdschTx cleanup.
     */
    ~PdschTx();

    /**
     * @brief: Run all PDSCH pipeline components, i.e., CRC + LDPC encoder + Rate Matching
     *         + Modulation + DMRS, once. Should have called expandParametersMultipleCells() first.
     * @param[in] cuda_strm: CUDA stream for kernel launches.
     * @param[in] ref_check: If set, compare the output of each pipeline component with
     *                       the reference output from the HDF5 file that drives the pipeline.
     * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INTERNAL_ERROR
     */
    cuphyStatus_t RunMultipleCells(const cudaStream_t& cuda_strm, bool ref_check=false);


    /**
     * @brief: Populate PdschPerTbParams and PdschDmrsParams arrays based on dyn_params.
     * @param[in] dyn_params: pointer to the dynamic parameters for the cell group the pipeline will process next.
     * @param[in] cuda_strm: CUDA stream for memory copies.
     * @return CUPHY_STATUS_SUCCESS or relevant error status
     */
    cuphyStatus_t expandParametersMultipleCells(cuphyPdschDynPrms_t* dyn_params,
                                                cudaStream_t cuda_strm=0);

    /**
     * @brief: Update output tensor addr. in kernel launch configs with the dynamic parameters (updated via cuphyFallbackBufferSetupPdschTx) in case of a single cell per cell group.
     */
    void updateOutputTensorBuffer(cudaStream_t cuda_strm=0);

    /**
     * @brief: Update output tensor addr. in kernel launch configs with the dynamic parameters (updated via cuphyFallbackBufferSetupPdschTx) in case of multiple cells per cell group.
     */
    void updateOutputTensorBuffers(int fallback_cells, cudaStream_t cuda_strm=0);

    /**
     * @brief: Compare CRC output with reference.
     * @param[in] cuda_strm: CUDA stream used for DtoH memory copies.
     * @return number of mismatched uint32_t elements.
     */
    int refCheckCRCMultipleCells(cudaStream_t cuda_strm);

    /**
     * @brief: Compare LDPC output with reference.
     * @param[in] cuda_strm: CUDA stream used for DtoH memory copies.
     * @return number of mismatched uint32_t elements.
     */
    int refCheckLDPCMultipleCells(cudaStream_t cuda_strm);

    /**
     * @brief: Compare Rate Matching output with reference.
     * @param[in] cuda_strm: CUDA stream used for DtoH memory copies.
     * @return number of mismatched bits.
     */
    int refCheckRateMatchingMultipleCells(cudaStream_t cuda_strm);

    /**
     * @brief: Compare Modulation output (only the data symbols) with reference.
     * @param[in] cuda_strm: CUDA stream used for DtoH memory copies.
     * @return number of mismatched modulation symbols.
     */
    int refCheckModulationMultipleCells(cudaStream_t cuda_strm);

    /**
     * @brief: Compare final output tensor, incl. modulation and DMRS components' output,
     *         with reference.
     * @param[in] cuda_strm: CUDA stream used for DtoH memory copies.
     * @return number of mismatched symbols.
     */
    int refCheckTxDataMultipleCells(cudaStream_t cuda_strm);
    int refCheckDmrsMultipleCells(cudaStream_t cuda_strm);

    /**
     * @brief: Copy the CRC output to a host buffer. No synchronization within this method.
     * @param[in] cuda_strm: CUDA stream for async. mem. copy.
     * @return CRC output as code blocks packed in uint32_t elements after per-TB and per-CB CRC attachments.
     */
    std::vector<uint32_t> getHostOutputCRCMultipleCells(uint16_t cell, cudaStream_t cuda_strm);

    /**
     * @brief: Copy the LDPC output of the cell_TB_id TB of a given cell to a host buffer. Includes stream synchronization.
     * @param[in] cell: index of cell whose TB will be copied; 0 for single-cell cases
     * @param[in] cell_TB_id: index of TB within cell to be copied
     * @param[in,out] num_CBs: pointer to number of CBs (code blocks) in this TB
     * @param[in,out] N: pointer to number of coded bits filled in for this TB
     * @param[in] cuda_strm: CUDA stream for async. mem. copy.
     * @return LDPC output as a bit tensor with {N rounded up to be div. by 32, num_CBs} layout.
     */
    cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc> getHostOutputLDPCTBPerCell(uint16_t cell, int cell_TB_id, int* num_CBs, int* N, cudaStream_t cuda_strm);
    cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc> getHostOutputLDPCMultipleCells(uint16_t cell, cudaStream_t cuda_strm);

    /**
     * @brief: Copy the Rate Matching output, before it got restructured for modulation,
     *         to a host buffer. No synchronization within this method.
     * @param[in] cuda_strm: CUDA stream for async. mem. copy.
     * @return Rate Matching output packed in uint32_t elements.
     */
    std::vector<uint32_t> getHostOutputRateMatching(cudaStream_t cuda_strm);

    const uint8_t * getGPUOutputCRC();
    cuphy::tensor_ref const& getGPUOutputLDPC();
    const uint32_t* getGPUOutputRateMatching();

    /**
     * @brief: Set pipeline's reverse bits flag to value.
     *         If set, will reverse bit order per byte, when parsing pipeline input.
     * @params[in] value: reverse bits option.
     */
    void setReverseBits(bool value);

    /**
     * @brief: Set pipeline's graph mode and inter-cell batching mode.
     *         If graph mode is set the pipeline uses a CUDA graph.
     *         If inter-cell batching mode is set the pipeline could batch work across cells in a cell group. Non inter-cell batching mode is deprecated and will default to true irrespective of this value.
     * @params[in] cfg_graph_mode: use CUDA graphs, a graph per pipeline (cell) if set.
     * @params[in] cfg_inter_cell_batching_mode: enable batching across cells in a cell group if set.
     */
    void setProcMode(bool cfg_graph_mode, bool cfg_inter_cell_batching_mode=true);
    /**
     * @brief: Return pipeline's graph mode.
     * @return true if pipeline's mode is graphs, false otherwise.
     */
    bool getGraphMode();

    const cuphyPdschStatPrms_t* static_params;
    const cuphyPdschDynPrms_t*  dynamic_params;

    size_t getGpuBufferSize();
    const void* getMemoryTracker();

private:

    cuphyMemoryFootprint memory_footprint;

    CUDA_KERNEL_NODE_PARAMS m_emptyNodeParamsDriver;
    void disableLDPCNodesMultipleCells();
    void disableCsirsPrepNodesMultipleCells();

    // For kernels with descriptor passed as __grid_constant__ arg
    CUDA_KERNEL_NODE_PARAMS m_emptyNodeParamsDriver_32B, m_emptyNodeParamsDriver_48B;
    struct testDescr_sz<32> arg_32B;
    struct testDescr_sz<48> arg_48B;

    // Parameters for delay kernel launched before PDSCH-run GPU work in cuphyPdschPipelineMode_t::PDSCH_POST_FEC_PROCESSING mode
    void* delayKernelParams[1]{};
    CUDA_KERNEL_NODE_PARAMS m_delayKernelParamsDriver;

    static constexpr int BITS_PER_U32 = sizeof(uint32_t) * 8; // in bits

    //DL Pipeline is driven by HDF5 input or h_pipeline_input_bytes if no HDF5 file is available
    std::string hdf5_filename;
    //TODO: change members dimensioned on PDSCH_MAX_CELLS_PER_CELL_GROUP to std::vectors for
    //more flexibility, especially if we do allocations etc. based on some static parameter field
    //rather than the PDSCH_MAX_CELLS_PER_CELL_GROUP.
    std::vector<std::string> per_cell_hdf5_filename;
    std::vector<std::unique_ptr<hdf5hpp::hdf5_file>> per_cell_input_file;

    std::vector<const uint8_t *> per_cell_pipeline_input_bytes;
    std::vector<const uint8_t *> h_per_cell_pipeline_input_tb_crcs;
    std::vector<int> per_cell_pipeline_input_size_bytes; // Size of pTbInput[i] buffer for cell i in bytes, which includes TB sizes along with supported padding. Used in H2D copy if buffer type is CPU_BUFFER.
    std::vector<int> per_cell_pipeline_input_padding_size_bytes; // Some of all padding bytes in the pTBInput[i] buffer for cell i. Determined by comparing the end of one TB with the tbStartOffset start of the subsequent one.
    bool pdsch_data_in_is_gpu_buffer = false;

    int num_TBs, num_cells;
    PdschPerTbParams* kernel_params = nullptr;
    cudaStream_t strm;

    std::vector<int> per_cell_num_TBs;
    std::vector<int> per_cell_TB_params_offset;

    bool scrambling;
    bool layer_mapping;
    bool read_TB_CRC; // if true, the TB CRC is part of the input buffer that drives the pipeline
    bool aas_mode; // if true, layer_mapping, modulation, DMRS should not be included.
    bool graph_mode; // if true, a CUDA graph is run.
    bool inter_cell_batching_mode; //if true, enable batching of kernels across cells. Only relevant if there are multiple cells per cell group.
    bool post_fec_processing{false};   // when set, indicates this pipeline will always run in cuphyPdschPipelineMode_t::PDSCH_POST_FEC_PROCESSING mode or PDSCH_POST_FEC_RM_SCRAMBLING_PROCESSING mode (i.e., execute post LDPC)
    bool post_fec_rm_scrambling_processing{false};   // subset of post_fec_processing; when set, indicates this pipeline will always run in cuphyPdschPipelineMode_t::PDSCH_POST_FEC_RM_SCRAMBLING_PROCESSING mode (i.e., execute post RM scrambling)
    uint32_t fec_delay_usec{0}; // if delay is not zero and post_fec_processing is enabled, launch delay kernel with said delay before actual PDSCH-run work

    uint32_t Emax; // reset to 0 in prepareRateMatching()
    uint32_t rounded_Emax; // rounded up to nearest 32-bit

    uint32_t prev_first_unused_ldpc_node;
    bool prev_had_CSIRS_params;
    bool rev_bit_order; //Should be set to true in E2E

    // Vector (one element per TB) of the number of parity nodes the LDPC encoder should compute.
    // Valid values are in [CUPHY_LDPC_MIN_PARITY_NODES, CUPHY_LDPC_MAX_BG{1,2}_PARITY_NODES] range.
    // A value of 0 (special case) signifies to the LDPC encoder that all parity nodes for that TB's base graph should be computed.
    // Computing a subset of the parity nodes (i.e., skipping nodes that will be later skipped by the fused rate matching kernel) is an optimization only relevant for rv=0.
    std::vector<uint32_t> max_ldpc_parity_nodes;

    bool ran_pipeline = false; // Set to true from Run method

    //Max supported config. values. Used in pool computation
    int max_CBs_per_TB;
    int max_K_per_CB;
    int max_N_per_CB;
    int max_Emax;
    int max_layers;
    int max_cells; // Use either static param. nMaxCellsPerSlot field or PDSCH_MAX_CELLS_PER_CELL_GROUP if the former is 0.

    int per_cell_max_TBs;
    int per_cell_group_max_TBs; // Use either static parameter nMaxUesPerCellGroup field or PDSCH_MAX_UES_PER_CELL_GROUP if the former is 0. Should be declared before ldpc_stream_pool for proper init. list order.
    int max_PRB_BWP; // max PRBs in downlink bandwidth part. Used for [h,d]_xtf_re_maps allocation. Default is 273.

    int cell_group_Emax; // max. number of rate-matched bits in a cell group

    __half pdsch_comp_tol;    // tolerance used in reference check evaluations

    void setMaxVals();
    void setMaxVals(int cfg_cell_max_TBs, int cfg_max_K_per_CB,
                    int cfg_max_N_per_CB, int cfg_max_Emax, int cfg_max_layers);

    // Number of PDSCH pipeline components with failed reference checks.
    // Reset via expandParameters call. Return via Run()
    int failed_ref_checks = 0;

    cuphy::kernelDescrs<N_PDSCH_COMPONENTS> m_component_descrs; // component descriptors
    bool bulk_desc_async_copy;
    size_t ldpc_descr_offset;
    size_t ldpc_workspace_offset, ldpc_workspace_bytes;

    //Per component launch config structs. All should include a cudaKernelNodeParams field.
    std::unique_ptr<cuphyLDPCEncodeLaunchConfig[]>    m_ldpc_encode_launch_cfg;
    std::unique_ptr<cuphyCrcEncodeLaunchConfig>       m_crc_encode_launch_cfg;
    std::unique_ptr<cuphyDlRateMatchingLaunchConfig>  m_rate_matching_launch_cfg;
    std::unique_ptr<cuphyPdschDmrsLaunchConfig>       m_dmrs_launch_cfg;
    std::unique_ptr<cuphyPdschCsirsPrepLaunchConfig>  m_csirs_prep_launch_cfg;
    std::unique_ptr<cuphyPrepareCrcEncodeLaunchConfig>  m_prepare_crc_encode_launch_cfg;

    std::vector<uint16_t> cell_idx_for_TB; //FIXME dynamic or static?
    uint32_t LDPC_signature(uint32_t BG, uint32_t Zc, uint32_t num_CBs, uint32_t F=0);


    void allocateDescr();

    void h_compute_re_map(CsirsTables* h_csirs_tables, uint16_t* computed_re_map,  _cuphyCsirsRrcDynPrm* csi_rs_params, uint16_t csirs_params_offset, int num_params, uint16_t cell);
    cuphyStatus_t d_compute_re_maps();

    //Output + workspace buffers for pipeline components

    // Xtf RE (resource element) maps for all cells in a cell group
    size_t per_cell_xtf_re_map_elements; // Number of uint16_t element per RE map per cell

    uint8_t* d_ldpc_w_ptr;
    uint8_t* h_ldpc_w_ptr;

    //CRC
    cuphy::unique_device_ptr<uint32_t> d_CB_CRCs; // standalone CB-CRCs
    cuphy::unique_device_ptr<uint32_t> d_TB_CRCs; // standalone TB-CRCs
    cuphy::unique_device_ptr<uint8_t>  d_code_blocks; // CRC output consumed by LDPC encoder
    PdschPerTbParams* d_tbPrmsArray;
    cuphy::unique_device_ptr<uint32_t> d_crc_workspace;

    //Moved vectors used only in prepareCRC and prepareCRCMultipleCells to avoid alloc/dealloc cost during setup
    std::vector<uint32_t> total_CB_byte_sizes; // first num_TBs elements used
    std::vector<uint32_t> CB_data_byte_sizes;  // first num_TBs elements used per setup. number of data bytes per CB (i.e, prior to CRC attachment and filler bits)
    std::vector<int>      tb_size;            // in bits; first num_TBs elements used
    std::vector<int>      padding_bytes;      // first num_TBs elements used.

    cuphy::tensor_device crc_d_in_tensor;
    uint32_t CRC_output_bytes; //size of CRC output buffer
    uint32_t max_CBs; // updated in prepareCRC
    uint32_t CRC_max_TB_padded_bytes; // reset in prepareCRC
    uint32_t max_tb_size_bytes;
    int crc_h_in_tensor_bytes;


    std::vector<uint32_t> per_cell_CRC_output_bytes; //size of CRC output buffer
    std::vector<uint32_t> per_cell_max_CBs; // updated in prepareCRC
    std::vector<uint32_t> per_cell_CRC_max_TB_padded_bytes; // reset in prepareCRC

    std::vector<uint32_t> per_cell_max_tb_size_bytes;
    std::vector<int> per_cell_crc_h_in_tensor_bytes;
    std::vector<int> per_cell_crc_h_in_tensor_bytes_offset;

    size_t max_per_cell_crc_workspace_elements;
    size_t max_per_cell_CB_CRCs_elements;
    size_t max_per_cell_code_blocks_bytes;

    uint32_t cell_max_CBs; // updated in prepareCRC
    uint32_t cell_CRC_max_TB_padded_bytes; // reset in prepareCRC

    int cell_group_crc_h_in_tensor_bytes;
    uint32_t cell_group_CRC_output_bytes;

    cuphy::tensor_ref cell_group_crc_d_in_tensor_ref;

    std::vector<uint32_t> TB_padded_byte_sizes;
    std::vector<uint32_t> TB_sizes; // tb_size read (in bits)

    struct LDPC_batch_info {
        uint32_t signature; // LDPC signature of all TBs in this batch
        uint16_t num_TBs;    // number of TBs in this batch; they will be processed by a single kernel
        uint8_t TB_idxs[PDSCH_MAX_UES_PER_CELL_GROUP]; //the TBs indices from the cell group included in the batch; first num_TBs elements are valid.
        //FIXME need to revisit design as we scale. Also, added runtime check if we have more than 256 UEs and thus data type should change to uint16_t
    };

    uint32_t total_LDPC_kernel_configs; // in a cell group

    std::vector<uint32_t> CRC_dst_offset; // per-TB offsets used in LDPC batching
    std::vector<uint32_t> LDPC_dst_offset; // per-TB offsets used in LDPC batching

    std::vector<struct LDPC_batch_info> LDPC_batches; //LDPC batches
    std::vector<void*> LDPC_input_addr;	// input addr. used in LDPC batching
    std::vector<void*> LDPC_output_addr; // output addr. used in LDPC batching


    // LDPC
    cuphy::tensor_ref d_ldpc_out_tensor_ref;
    std::vector<cuphy::tensor_ref> d_per_cell_ldpc_out_tensor_ref;

    cuphy::unique_device_ptr<uint32_t> d_ldpc_workspace;
    uint32_t* d_ldpc_workspace_ptr{nullptr}; // owned by PdschTx pipeline caller in cuphyPdschPipelineMode_t::PDSCH_POST_FEC_PROCESSING mode; otherwise same as d_ldpc_workspace.get()
    std::vector<cuphy::tensor_desc> single_TB_d_ldpc_in_tensor_desc;
    std::vector<cuphy::tensor_desc> single_TB_d_ldpc_out_tensor_desc;
    std::vector<cuphy::tensor_device> single_TB_d_ldpc_out_tensor;


    uint32_t LDPC_N_max; // per cell group values; reset in prepareRateMatching (called before prepareLDPC*)
    std::vector<uint32_t> per_cell_LDPC_K_max; // reset in prepareRateMatching (called before prepareLDPC*)

    size_t max_per_TB_LDPC_workspace_size;

    std::vector<cudaStream_t> ldpc_streams;
    cuphy::stream_pool ldpc_stream_pool; // Only relevant if AVOID_STREAM_CREATION_IN_SETUP is 1.
    size_t ldpc_stream_elements; // Reset to zero in every setup; counts # LDPC streams used in a given setup/run
    size_t created_ldpc_streams; // Only relevant if AVOID_STREAM_CREATION_IN_SETUP is 0.
    /* When a single pipeline object processes multiple cells, it is possible that
     * the testbench or controller has queued up some work in that object's CUDA stream
     * after setup and before calling run. This event will ensure all CUDA streams from the
     * PdschTx internal pools wait until that work has been completed.
     * One use case is measurememnts etc. */
    cuphy::event start_run_event;

    cuphy::event crc_event;
    std::vector<cudaEvent_t> ldpc_complete_events;

    // Rate matching
    uint32_t* h_rm_workspace, *d_rm_workspace;
    cuphy::tensor_device d_rate_matching_output; // RM's output; only valid in case of AAS mode
    size_t rm_output_elements;
    size_t rm_allocated_workspace_size;

    bool has_cells_in_TM; // at least one cell in cell group is in testing mode if set

    // DMRS
    PdschDmrsParams* h_dmrs_params;
    PdschDmrsParams* d_dmrs_params; // used in modulation too

    // UE groups for CSI-RS paramters
    PdschUeGrpParams* h_ue_grp_params;
    PdschUeGrpParams* d_ue_grp_params;

    // Modulation
    cuphy::tensor_device d_modulation_in_tensor;
    uint32_t max_symbols;
    uint32_t max_bits_per_layer;

    // This method is called in the constructor. It allocates overprovisioned buffers needed.
    void allocateBuffers();

    // This method is called in expandParameters*() and  calls the corresponding prepare* methods.
    cuphyStatus_t prepareBuffersMultipleCells(cudaStream_t cuda_strm);

    // Methods that do needed preparation for the corresponding components for multiple cells.
    cuphyStatus_t prepareCRCMultipleCells(cudaStream_t cuda_strm);
    cuphyStatus_t prepareCRC_step2MultipleCells(cudaStream_t cuda_strm);
    cuphyStatus_t prepareLDPCBatching(cudaStream_t cuda_strm);
    cuphyStatus_t prepareRateMatchingMultipleCells(cudaStream_t cuda_strm);
    cuphyStatus_t prepareDmrsMultipleCells(cudaStream_t cuda_strm);


    // Methods to run pipeline components; called in order from RunMultipleCells.
    void runCRCMultipleCells(cudaStream_t cuda_strm, bool ref_check);
    void runLDPCBatching(cudaStream_t cuda_strm, bool ref_check);
    void runRateMatchingMultipleCells(cudaStream_t cuda_strm, bool ref_check);
    void runDmrsMultipleCells(cudaStream_t cuda_strm, bool ref_check);
    void runPdschCsirsPrepMultipleCells(cudaStream_t cuda_strm);


    void print_csv_friendly_config();

    // Graph
    CUgraph m_graph;
    CUgraphExec exec_graph;
    void createAndInstantiateGraph();

    std::vector<CUgraphNode> m_crcEncodeNodesMultipleCells;
    std::vector<CUgraphNode> m_ldpcEncodeNodesMultipleCells;
    std::vector<CUgraphNode> m_rmNodesMultipleCells;
    std::vector<CUgraphNode> m_dmrsNodeMultipleCells;
    std::vector<CUgraphNode> m_csirsPrepNodesMultipleCells;
    std::vector<CUgraphNode> m_prepareCRCEncodeNodeMultipleCells;

#if CUDA_VERSION >= 13020
    // Attribute value for kernel nodes that may need dynamic shared memory opt-in
    CUkernelNodeAttrValue m_kernelNodeAttrValue{.sharedMemoryMode = CU_SHARED_MEMORY_MODE_ALLOW_NON_PORTABLE};
#endif

    CUresult addDependenciesHelper(CUgraph hGraph, const CUgraphNode* from, const CUgraphNode* to, size_t numDependencies);
    void addDependenciesMultipleCells();
    void updateNodeParamsMultipleCells();

    // CSI-RS & xtf-re-map related
    // Offset of the various host/device buffers, in bytes, from the start of the workspace
    std::array<int, N_CSIRS_COMPONENTS+1> workspace_offsets;

    // Update workspace_offsets for all buffers for each new config of cells and CSI-RS params
    void updateCsirsWorkspaceOffsets(int cells, int params);
    // Update the workspace pointers
    void updateCsirsWorkspacePtrs();

    // Offset array used to map starting thread index for each parameter in the batch
    uint32_t* h_offsets;
    uint32_t* d_offsets;

    uint32_t* h_csirs_cell_index;
    uint32_t* d_csirs_cell_index;

    // CSI-RS input parameter list
    _cuphyCsirsRrcDynPrm* h_csirs_params;
    _cuphyCsirsRrcDynPrm* d_csirs_params;

    // Workspace buffers that enables single, not overprovisioned H2D copy, per setup instead of the
    // overprovisioned one with descriptors.
    cuphy::unique_device_ptr<uint8_t> d_workspace;
    cuphy::unique_pinned_ptr<uint8_t> h_workspace;

    // Xtf RE (resource element) maps for all cells in a cell group
    cuphy::unique_device_ptr<uint16_t> d_xtf_re_maps_v2;
    cuphy::unique_pinned_ptr<uint16_t> h_xtf_re_maps_v2;

    cuphy::unique_device_ptr<uint8_t> d_prepare_crc_input_buffer; // temp. used for H2D copies when REVERT_TO_ASYNC_COPIES is 1 in pdsch_tx.cpp

    //batched memory copy
    cuphyBatchedMemcpyHelper m_batchedMemcpyHelper;
};

void updateRefCheck(cuphyPdschTxHndl_t pdschTxHndl, bool ref_check);
void updateRefCheckMultipleCells(cuphyPdschTxHndl_t pdschTxHndl, bool ref_check);
void updateFileName(cuphyPdschTxHndl_t pdschTxHndl, const char* file_name);
void updateFileNameMultipleCells(cuphyPdschTxHndl_t pdschTxHndl, uint32_t cell_id, const char* file_name);

#endif // !defined(PDSCH_TX_HPP_INCLUDED_)
