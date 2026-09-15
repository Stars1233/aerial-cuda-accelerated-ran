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

#ifndef CUPHY_LDPC_CB_KERNEL_API_H
#define CUPHY_LDPC_CB_KERNEL_API_H

#ifndef __cplusplus
#error "ldpc_cb_kernel_api.h requires C++ because it depends on existing cuPHY C++ headers."
#endif

#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

#include "cuphy.h"

#define CUPHY_LDPC_CB_KERNEL_SPI_CONTRACT_VERSION 1
#define CUPHY_LDPC_CB_KERNEL_NODE_PAYLOAD_BYTES 4096U
#define CUPHY_LDPC_CB_KERNEL_NODE_PAYLOAD_ALIGNMENT 16U

/*******************************************************************************
 * Common Codeblock Type Definitions
 ******************************************************************************/

#define CUPHY_LDPC_CB_EARLY_TERM       (1U << 0)  /*!< Enable early termination on CRC pass.
                                                       Reserved for a future V1 update; the
                                                       current reference implementation rejects it. */
#define CUPHY_LDPC_CB_THROUGHPUT_MODE  (1U << 1)  /*!< Optimize for throughput over latency. */
#define CUPHY_LDPC_CB_WRITE_SOFT_BITS  (1U << 2)  /*!< Write soft output to llr_out buffers. */

#define CUPHY_LDPC_CB_GROUP_ID_NONE    (0xFFFFU)  /*!< Sentinel value indicating codeblock is
                                                       not part of a group. */

/**
 * @brief CRC type for codeblock decoding.
 */
typedef enum cuphyLdpcCbCrc_e {
    CUPHY_LDPC_CB_CRC_NONE = 0, /*!< No CRC check. */
    CUPHY_LDPC_CB_CRC_16   = 1, /*!< 16-bit CRC. */
    CUPHY_LDPC_CB_CRC_24A  = 2, /*!< 24-bit CRC type A. */
    CUPHY_LDPC_CB_CRC_24B  = 3  /*!< 24-bit CRC type B. */
} cuphyLdpcCbCrc_t;

/**
 * @brief LDPC base graph selection.
 */
typedef enum cuphyLdpcCbBaseGraph_e {
    CUPHY_LDPC_CB_BG1 = 1, /*!< Base graph 1. */
    CUPHY_LDPC_CB_BG2 = 2  /*!< Base graph 2. */
} cuphyLdpcCbBaseGraph_t;

/**
 * @brief Per-codeblock data pointers and grouping hint used by the subgroup builder.
 */
typedef struct cuphyLdpcCbData_s {
    const uint8_t* llr_in;    /*!< [Device memory] Pointer to input LLRs for this codeblock. */
    uint32_t*      bits_out;  /*!< [Device memory] Hard decision packed decoded bits. */
    uint8_t*       llr_out;   /*!< [Device memory] Optional output LLR buffer. */
    uint16_t       group_id;  /*!< Optional grouping hint for contiguous/strided spans. */
} cuphyLdpcCbData_t;

/*******************************************************************************
 * SPI Type Definitions
 ******************************************************************************/

/**
 * @brief Opaque handle for the LDPC kernel chooser.
 */
typedef struct cuphyLdpcCbKernelChooser_s* cuphyLdpcCbKernelChooser_t;

/**
 * @brief Opaque handle for the LDPC prepare-launch object.
 */
typedef struct cuphyLdpcCbLaunchPreparer_s* cuphyLdpcCbLaunchPreparer_t;

/**
 * @brief Opaque handle for a reusable prepared-launch object.
 */
typedef struct cuphyLdpcCbPreparedLaunch_s* cuphyLdpcCbPreparedLaunch_t;

/**
 * @brief Static configuration for kernel selection.
 */
typedef struct cuphyLdpcCbKernelChooserConfig_s {
    cuphyDataType_t llr_type;     /*!< Type of LLR input data. Only ::CUPHY_R_16F is supported. */
    float           clamp_value;  /*!< LLR saturation clamp value. */
    uint32_t        config_flags; /*!< Decode policy flags. V1 supports throughput mode
                                       and optional soft-output behavior. */
} cuphyLdpcCbKernelChooserConfig_t;

/**
 * @brief Static configuration for launch preparation.
 */
typedef struct cuphyLdpcCbLaunchStaticConfig_s {
    cuphyDataType_t llr_type;     /*!< Type of LLR input data. Only ::CUPHY_R_16F is supported. */
    float           clamp_value;  /*!< LLR saturation clamp value. */
    uint32_t        config_flags; /*!< Decode policy flags. V1 supports throughput mode
                                       and optional soft-output behavior. */
} cuphyLdpcCbLaunchStaticConfig_t;

/**
 * @brief Homogeneous subgroup shape chosen by the caller.
 */
typedef struct cuphyLdpcCbSubgroupKey_s {
    cuphyLdpcCbBaseGraph_t bg;           /*!< Base graph (1 or 2) for all codeblocks in the subgroup. */
    uint16_t               Zc;           /*!< Lifting size for all codeblocks in the subgroup. */
    uint16_t               parity_nodes; /*!< Number of parity-check nodes for the subgroup. */
    uint16_t               k;            /*!< Number of information bits per codeblock in the subgroup. */
    cuphyLdpcCbCrc_t       crc_type;     /*!< CRC type used for early termination. */
    uint16_t               max_iters;    /*!< Maximum number of decoding iterations for the subgroup. */
    uint8_t                algo;         /*!< Algorithm selection (0 for automatic, non-zero for explicit override). */
} cuphyLdpcCbSubgroupKey_t;

/**
 * @brief Contiguous or strided span of codeblocks inside one homogeneous subgroup.
 */
typedef struct cuphyLdpcCbBufferSpan_s {
    const uint8_t* llr_base;             /*!< [Device memory] Base pointer to the input LLR buffer
                                              for the first codeblock in the span. */
    uint32_t       llr_stride_bytes;     /*!< Byte distance between successive codeblocks' input
                                              LLR buffers in this span. */
    uint32_t*      bits_base;            /*!< [Device memory] Non-NULL, at least 32-bit aligned
                                              base pointer for packed hard-output storage. */
    uint32_t       bits_offset_bytes;    /*!< Byte offset from `bits_base` to the first hard-output
                                              byte of the first codeblock in the span. */
    uint32_t       bits_stride_bytes;    /*!< Byte distance between successive codeblocks'
                                              hard-output regions in this span. */
    uint8_t*       llr_out;              /*!< [Device memory] Base pointer to optional soft-output
                                              storage for the first codeblock in the span. May be
                                              NULL if soft output is disabled. */
    uint32_t       llr_out_stride_elems; /*!< Element distance between successive codeblocks'
                                              soft-output buffers in this span. */
    uint16_t       num_cb;               /*!< Number of codeblocks described by this span. */
} cuphyLdpcCbBufferSpan_t;

/**
 * @brief Homogeneous subgroup descriptor.
 */
typedef struct cuphyLdpcCbSubgroupDesc_s {
    cuphyLdpcCbSubgroupKey_t       key;       /*!< Homogeneous LDPC parameters for this subgroup. */
    const cuphyLdpcCbBufferSpan_t* spans;     /*!< Array of one or more memory-layout spans for the
                                                   subgroup. */
    uint16_t                       num_spans; /*!< Number of valid span descriptors in `spans`. */
} cuphyLdpcCbSubgroupDesc_t;

/**
 * @brief Batch-oriented input descriptor for one LDPC codeblock launch.
 */
typedef struct cuphyLdpcCbLaunchBatchDesc_s {
    const cuphyLdpcCbSubgroupDesc_t* subgroups;     /*!< [Host memory] Array of homogeneous
                                                         subgroup descriptors to combine
                                                         into one launch. */
    uint16_t                         num_subgroups; /*!< Number of valid subgroup descriptors
                                                         in `subgroups`. */
    uint32_t*                        status_out;    /*!< [Device memory] Optional per-codeblock
                                                         status words in flattened batch order.
                                                         Bits [7:0] are reserved for the iteration
                                                         count. Bits [31:8] are reserved for status
                                                         flags. The V1 canonical packer returns
                                                         CUPHY_STATUS_NOT_SUPPORTED when this field
                                                         is non-NULL. */
} cuphyLdpcCbLaunchBatchDesc_t;

/**
 * @brief Minimum traits required to construct a kernel launch.
 */
typedef struct cuphyLdpcCbKernelTraits_s {
    uint32_t shared_mem_bytes;          /*!< Dynamic shared-memory bytes required per CTA. */
    uint16_t block_dim_x;               /*!< CUDA block dimension X for this launch family. */
    uint16_t block_dim_y;               /*!< CUDA block dimension Y for this launch family. */
    uint16_t block_dim_z;               /*!< CUDA block dimension Z for this launch family. */
    uint16_t llr_base_alignment_bytes;  /*!< Required alignment for `llr_base` pointers. */
    uint16_t bits_base_alignment_bytes; /*!< Required alignment for `bits_base` pointers. */
    uint8_t  codewords_per_cta;         /*!< Number of codewords processed per CTA. */
    uint8_t  kernel_arg_count;          /*!< Number of kernel arguments used by this family. */
} cuphyLdpcCbKernelTraits_t;

/**
 * @brief Selected launch family and kernel implementation.
 */
typedef struct cuphyLdpcCbKernelChoice_s {
    CUfunction                func;                         /*!< Opaque CUDA driver function handle for
                                                                 the selected launch family. Callers may
                                                                 pass this handle to CUDA launch APIs but
                                                                 must not infer batching compatibility
                                                                 from raw handle equality. For the shipped
                                                                 chooser implementation, this handle may
                                                                 be backed by chooser-owned CUDA module
                                                                 state; keep the chooser alive while any
                                                                 launch or graph node using this choice
                                                                 may execute. */
    cuphyLdpcCbKernelTraits_t traits;                       /*!< Launch traits needed to construct valid
                                                                 kernel node params. */
    uint8_t                   effective_algo;               /*!< Resolved algorithm after automatic
                                                                 selection. */
    uint32_t                  compatibility_class_id;       /*!< Family-level compatibility tag for
                                                                 batching decisions. */
    uint8_t                   supports_heterogeneous_batch; /*!< Nonzero if the family can accept
                                                                 multiple subgroup keys in one batch. */
    uint16_t                  max_subgroups;                /*!< Maximum subgroup descriptors accepted
                                                                 by this family. */
    cuphyLdpcCbSubgroupKey_t  key;                          /*!< Reference-helper compatibility key
                                                                 captured at choice time so launch packing
                                                                 is self-contained. */
    cuphyLDPCDecodeConfigDesc_t decode_config;               /*!< Reference-helper resolved static decode
                                                                 config template for launch packing. */
    void*                     static_kernel_arg;             /*!< Reference-helper opaque family-specific
                                                                 static kernel argument. This must point
                                                                 to process-lifetime kernel metadata, not
                                                                 chooser-owned storage. Callers should
                                                                 treat this pointer as opaque. */
} cuphyLdpcCbKernelChoice_t;

/**
 * @brief Kernel choice paired with one launch batch descriptor.
 */
typedef struct cuphyLdpcCbKernelLaunchBatch_s {
    const cuphyLdpcCbKernelChoice_t*    choice;     /*!< Selected launch-family choice for this
                                                         whole batch. */
    const cuphyLdpcCbLaunchBatchDesc_t* batch_desc; /*!< Batch descriptor to pack for launch. */
} cuphyLdpcCbKernelLaunchBatch_t;

/**
 * @brief Launch-ready CUDA kernel-node params produced by the canonical SPI helper.
 */
typedef struct cuphyLdpcCbKernelNodeParams_s {
    CUDA_KERNEL_NODE_PARAMS      params;         /*!< Launch-ready CUDA kernel-node params.
                                                     The function handle follows the lifetime
                                                     rules of the choice used to build it. */
    void*                        kernel_args[4]; /*!< Reference-helper storage for kernel
                                                     argument pointers. */
    uint32_t                     kernel_arg_count; /*!< Number of valid entries in `kernel_args`. */
    alignas(CUPHY_LDPC_CB_KERNEL_NODE_PAYLOAD_ALIGNMENT)
    uint8_t payload[CUPHY_LDPC_CB_KERNEL_NODE_PAYLOAD_BYTES]; /*!< Reference-helper opaque payload
                                                                   storage so kernel args remain
                                                                   valid until this object is reused
                                                                   or destroyed. */
} cuphyLdpcCbKernelNodeParams_t;

/**
 * @brief Launch-family metadata returned by the prepare-launch API.
 */
typedef struct cuphyLdpcCbLaunchFamily_s {
    uint32_t compatibility_class_id;       /*!< Family-level compatibility tag for batching decisions. */
    uint8_t  effective_algo;               /*!< Resolved algorithm after automatic selection. */
    uint8_t  supports_heterogeneous_batch; /*!< Nonzero if the family can accept heterogeneous batches. */
    uint16_t max_subgroups;                /*!< Maximum subgroup descriptors accepted by this family. */
} cuphyLdpcCbLaunchFamily_t;

/**
 * @brief Capacity configuration for one reusable prepared-launch object.
 */
typedef struct cuphyLdpcCbPreparedLaunchConfig_s {
    uint16_t max_subgroups;        /*!< Maximum number of subgroup descriptors that may be
                                        bound into one batch. */
    uint32_t max_total_codeblocks; /*!< Maximum total number of codeblocks across all
                                        subgroups in one prepared batch. This also bounds
                                        the number of buffer spans, each of which must
                                        describe at least one codeblock. */
} cuphyLdpcCbPreparedLaunchConfig_t;

/**
 * @brief Optional prepared-launch metadata for logging, caching, and debugging.
 */
typedef struct cuphyLdpcCbPreparedLaunchInfo_s {
    uint32_t compatibility_class_id; /*!< Family-level compatibility tag used for the launch. */
    uint8_t  effective_algo;         /*!< Resolved algorithm used for the launch. */
    uint8_t  codewords_per_cta;      /*!< Number of codewords processed per CTA. */
    uint8_t  heterogeneous_batch;    /*!< Nonzero if the prepared batch contains multiple subgroup keys. */
    uint16_t num_subgroups;          /*!< Number of subgroup descriptors in the prepared batch. */
    uint32_t total_codewords;        /*!< Total codeblocks across all subgroups in the prepared batch. */
    uint32_t grid_dim_x;             /*!< CUDA grid dimension X selected for the launch. */
} cuphyLdpcCbPreparedLaunchInfo_t;

/*******************************************************************************
 * SPI Functions
 ******************************************************************************/

/**
 * Creates a codeblock LDPC kernel chooser.
 * @param[out] chooser Receives the created chooser handle.
 * @param[in] config Static chooser configuration.
 * @return CUPHY_STATUS_SUCCESS on success; otherwise an error status.
 */
cuphyStatus_t cuphyCreateLdpcCbKernelChooser(
    cuphyLdpcCbKernelChooser_t*             chooser,
    const cuphyLdpcCbKernelChooserConfig_t* config);

/**
 * Destroys a codeblock LDPC kernel chooser.
 * @param[in] chooser Chooser handle to destroy.
 * @return CUPHY_STATUS_SUCCESS on success; otherwise an error status.
 */
cuphyStatus_t cuphyDestroyLdpcCbKernelChooser(
    cuphyLdpcCbKernelChooser_t chooser);

/**
 * Selects a kernel implementation for a homogeneous subgroup.
 * @param[in] chooser Kernel chooser handle.
 * @param[in] key Homogeneous subgroup shape to select.
 * @param[out] choice Receives the selected kernel choice.
 * @return CUPHY_STATUS_SUCCESS on success; otherwise an error status.
 */
cuphyStatus_t cuphyLdpcCbChooseKernel(
    cuphyLdpcCbKernelChooser_t      chooser,
    const cuphyLdpcCbSubgroupKey_t* key,
    cuphyLdpcCbKernelChoice_t*      choice);

/**
 * Builds a subgroup descriptor from codeblock data.
 * @param[in] key Homogeneous subgroup shape.
 * @param[in] cb_data Codeblock data to pack into spans.
 * @param[in] num_cb Number of codeblocks in cb_data.
 * @param[out] span_storage Caller-owned storage for generated spans.
 * @param[in,out] num_spans On input, span_storage capacity; on output, generated span count.
 * @param[out] subgroup_desc Receives the subgroup descriptor.
 * @return CUPHY_STATUS_SUCCESS on success; otherwise an error status.
 */
cuphyStatus_t cuphyLdpcCbBuildSubgroupDesc(
    const cuphyLdpcCbSubgroupKey_t* key,
    const cuphyLdpcCbData_t*        cb_data,
    uint16_t                        num_cb,
    cuphyLdpcCbBufferSpan_t*        span_storage,
    uint16_t*                       num_spans,
    cuphyLdpcCbSubgroupDesc_t*      subgroup_desc);

/**
 * Builds CUDA kernel node parameters for a chosen launch batch.
 * @param[in] launch_batch Kernel choice and batch descriptor to pack.
 * @param[out] node_params Receives the driver kernel node parameters.
 * @return CUPHY_STATUS_SUCCESS on success; otherwise an error status.
 */
cuphyStatus_t cuphyLdpcCbBuildKernelNodeParams(
    const cuphyLdpcCbKernelLaunchBatch_t* launch_batch,
    cuphyLdpcCbKernelNodeParams_t*        node_params);

/**
 * Computes the LDPC normalization mode for a subgroup configuration.
 * @param[in] bg Base graph.
 * @param[in] parity_nodes Number of parity nodes.
 * @param[in] llr_type Input LLR data type.
 * @param[out] norm Receives the normalization mode.
 * @return CUPHY_STATUS_SUCCESS on success; otherwise an error status.
 */
cuphyStatus_t cuphyLdpcCbComputeNormalization(
    cuphyLdpcCbBaseGraph_t    bg,
    uint16_t                  parity_nodes,
    cuphyDataType_t           llr_type,
    cuphyLDPCNormalization_t* norm);

/*******************************************************************************
 * Prepare-Launch Functions
 ******************************************************************************/

/**
 * Creates a preparer for reusable codeblock LDPC launches.
 * @param[out] preparer Receives the created preparer handle.
 * @param[in] config Static launch configuration.
 * @return CUPHY_STATUS_SUCCESS on success; otherwise an error status.
 */
cuphyStatus_t cuphyCreateLdpcCbLaunchPreparer(
    cuphyLdpcCbLaunchPreparer_t*           preparer,
    const cuphyLdpcCbLaunchStaticConfig_t* config);

/**
 * Destroys a launch preparer when it has no active prepared launches.
 * @param[in] preparer Preparer handle to destroy.
 * @return CUPHY_STATUS_SUCCESS on success; CUPHY_STATUS_INVALID_ARGUMENT if launches remain active.
 */
cuphyStatus_t cuphyDestroyLdpcCbLaunchPreparer(
    cuphyLdpcCbLaunchPreparer_t preparer);

/**
 * Queries the compatible launch family for a subgroup key.
 * @param[in] preparer Launch preparer handle.
 * @param[in] key Homogeneous subgroup key.
 * @param[out] family Receives the compatible launch family.
 * @return CUPHY_STATUS_SUCCESS on success; otherwise an error status.
 */
cuphyStatus_t cuphyLdpcCbQueryLaunchFamily(
    cuphyLdpcCbLaunchPreparer_t     preparer,
    const cuphyLdpcCbSubgroupKey_t* key,
    cuphyLdpcCbLaunchFamily_t*      family);

/**
 * Queries the kernel choice for a subgroup key.
 * The returned choice remains valid while its preparer is alive.
 * @param[in] preparer Launch preparer handle.
 * @param[in] key Homogeneous subgroup key.
 * @param[out] choice Receives the selected kernel choice.
 * @return CUPHY_STATUS_SUCCESS on success; otherwise an error status.
 */
cuphyStatus_t cuphyLdpcCbQueryKernelChoice(
    cuphyLdpcCbLaunchPreparer_t     preparer,
    const cuphyLdpcCbSubgroupKey_t* key,
    cuphyLdpcCbKernelChoice_t*      choice);

/**
 * Queries workspace size and alignment for a reusable prepared launch.
 * @param[in] preparer Launch preparer handle.
 * @param[in] family Compatible launch family.
 * @param[in] config Prepared-launch capacity configuration.
 * @param[out] workspace_size_bytes Receives required workspace size.
 * @param[out] workspace_alignment Receives required workspace alignment.
 * @return CUPHY_STATUS_SUCCESS on success; otherwise an error status.
 */
cuphyStatus_t cuphyLdpcCbPreparedLaunchGetWorkspaceSize(
    cuphyLdpcCbLaunchPreparer_t              preparer,
    const cuphyLdpcCbLaunchFamily_t*         family,
    const cuphyLdpcCbPreparedLaunchConfig_t* config,
    size_t*                                  workspace_size_bytes,
    size_t*                                  workspace_alignment);

/**
 * Initializes a prepared launch in caller-owned workspace.
 * The preparer remains in use until cuphyLdpcCbPreparedLaunchDeinit is called.
 * @param[in] preparer Launch preparer handle.
 * @param[in] family Compatible launch family.
 * @param[in] config Prepared-launch capacity configuration.
 * @param[in] workspace Aligned caller-owned workspace.
 * @param[in] workspace_size_bytes Size of workspace in bytes.
 * @param[out] launch Receives the prepared launch handle.
 * @return CUPHY_STATUS_SUCCESS on success; otherwise an error status.
 */
cuphyStatus_t cuphyLdpcCbPreparedLaunchInitInPlace(
    cuphyLdpcCbLaunchPreparer_t              preparer,
    const cuphyLdpcCbLaunchFamily_t*         family,
    const cuphyLdpcCbPreparedLaunchConfig_t* config,
    void*                                    workspace,
    size_t                                   workspace_size_bytes,
    cuphyLdpcCbPreparedLaunch_t*             launch);

/**
 * Packs a batch descriptor into a prepared launch.
 * @param[in] launch Prepared launch handle.
 * @param[in] batch_desc Batch descriptor to pack.
 * @param[out] info Optional prepared-launch metadata.
 * @return CUPHY_STATUS_SUCCESS on success; otherwise an error status.
 */
cuphyStatus_t cuphyLdpcCbPrepareLaunch(
    cuphyLdpcCbPreparedLaunch_t         launch,
    const cuphyLdpcCbLaunchBatchDesc_t* batch_desc,
    cuphyLdpcCbPreparedLaunchInfo_t*    info);

/**
 * Packs a batch descriptor into a prepared launch using a previously queried choice.
 * @param[in] launch Prepared launch handle.
 * @param[in] batch_desc Batch descriptor to pack.
 * @param[in] choice Choice returned by the launch preparer used to initialize launch.
 * @param[out] info Optional prepared-launch metadata.
 * @return CUPHY_STATUS_SUCCESS on success; otherwise an error status.
 */
cuphyStatus_t cuphyLdpcCbPrepareLaunchWithChoice(
    cuphyLdpcCbPreparedLaunch_t         launch,
    const cuphyLdpcCbLaunchBatchDesc_t* batch_desc,
    const cuphyLdpcCbKernelChoice_t*    choice,
    cuphyLdpcCbPreparedLaunchInfo_t*    info);

/**
 * Retrieves CUDA kernel node parameters from a prepared launch.
 * @param[in] launch Prepared launch handle.
 * @param[out] node_params Receives the driver kernel node parameters.
 * @return CUPHY_STATUS_SUCCESS on success; otherwise an error status.
 */
cuphyStatus_t cuphyLdpcCbGetKernelNodeParams(
    cuphyLdpcCbPreparedLaunch_t launch,
    CUDA_KERNEL_NODE_PARAMS*    node_params);

/**
 * Deinitializes a prepared launch and releases its preparer reference.
 * @param[in] launch Prepared launch handle to deinitialize.
 * @return CUPHY_STATUS_SUCCESS on success; otherwise an error status.
 */
cuphyStatus_t cuphyLdpcCbPreparedLaunchDeinit(
    cuphyLdpcCbPreparedLaunch_t launch);

#endif /* CUPHY_LDPC_CB_KERNEL_API_H */
