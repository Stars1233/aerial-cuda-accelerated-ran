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

//#define CUPHY_DEBUG 1

#include "cuphy.h"
#include "cuphy_internal.h"
#include <vector>
#include <iostream>
#include "modulation_mapper.hpp"
#include <assert.h>
#include "tensor_desc.hpp"
#include "cuphy_kernel_util.cuh"

#define FLOAT_COMP 1

__device__ __constant__ float rev_qam_16[4] = {
    0.316227766,
    -0.316227766,
    0.948683298,
    -0.948683298,
};

__device__ __constant__ float rev_qam_16_long[8] = {
    0.316227766,
    -0.316227766,
    0.316227766,
    -0.316227766,
    0.948683298,
    -0.948683298,
    0.948683298,
    -0.94868329
};

/* Indexed as {bit 4, bit 2, bit 0}, thus the reverse in the name. */
__device__ __constant__ float rev_qam_64[8] = {
    0.462910049886276,
    -0.462910049886276,
    0.77151674981046,
    -0.77151674981046,
    0.154303349962092,
    -0.154303349962092,
    1.08012344973464,
    -1.08012344973464
};

__device__ __constant__ float rev_qam_256[16] = {
    0.383482494,
    -0.383482494,
    0.843661488,
    -0.843661488,
    0.230089497,
    -0.230089497,
    0.997054486,
    -0.997054486,
    0.536875492,
    -0.536875492,
    0.69026849,
    -0.69026849,
    0.076696499,
    -0.076696499,
    1.150447483,
    -1.150447483
};

__device__ __inline__ uint32_t map_index_8bits(uint32_t index) {
    uint32_t masked_index = (index & 0x1) | ((index & 0x4) >> 1) |
                   ((index & 0x10) >> 2)  | ((index & 0x40) >> 3);
    return masked_index;
}

__device__ __inline__ uint32_t map_index_6bits(uint32_t index) {
    uint32_t masked_index = (index & 0x1) | ((index & 0x4) >> 1) |
                   ((index & 0x10) >> 2);
    return masked_index;
}

//blockIdx.y is TB_id; a TB can map to multiple layers.
__device__ __inline__ uint32_t output_index_calc(int allocated_Rbs, int start_data_symbol,
                                            int start_Rb, int data_symbols_per_layer,
                                            const PdschDmrsParams * __restrict__ params) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x; //symbols within a TB
    int Rbs_per_symbol = allocated_Rbs * CUPHY_N_TONES_PER_PRB;

    int symbol_id = tid / Rbs_per_symbol; // This is the symbol # within all the layer(s) TB blockIdx.y maps to.
    int layer_cnt = symbol_id / data_symbols_per_layer;
    int layer_id = calculate_layer_id(params[blockIdx.y].port_ids[layer_cnt], params[blockIdx.y].n_scid, params[blockIdx.y].nlAbove16);
    int per_layer_symbol_id = symbol_id % data_symbols_per_layer;
    int symbol_pos = (params->data_sym_loc >> (4*per_layer_symbol_id)) & 0xF;

    int all_Rbs_symbols = 273 * CUPHY_N_TONES_PER_PRB;
    //uint32_t output_index = all_Rbs_symbols * (layer_id * OFDM_SYMBOLS_PER_SLOT + start_data_symbol + per_layer_symbol_id) \
    //                        + (start_Rb * CUPHY_N_TONES_PER_PRB) + (tid % Rbs_per_symbol);
    uint32_t output_index = all_Rbs_symbols * (layer_id * OFDM_SYMBOLS_PER_SLOT + symbol_pos) \
                            + (start_Rb * CUPHY_N_TONES_PER_PRB) + (tid % Rbs_per_symbol);
    return output_index;
}


template<uint8_t Tqam>
__device__ uint32_t input_index_calc(int max_bits_per_layer, int allocated_Rbs, int data_symbols_per_layer,
                                     const PdschDmrsParams * __restrict__ params,
                                     const uint32_t* __restrict__ modulation_input,
                                     const struct PdschPerTbParams * workspace) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // symbol within a TB
    int num_symbols_per_layer_per_TB = (workspace[blockIdx.y].G / params[blockIdx.y].num_layers) / Tqam;
    int num_TBs = gridDim.y;

    //Have each layer in the restructured output start at uint32_t aligned boundary
    uint32_t rounded_num_elements_per_layer = div_round_up<uint32_t>(max_bits_per_layer, 32);

    int TB_id = blockIdx.y;
    int layer_cnt = tid / num_symbols_per_layer_per_TB; //check layer cnt is valid
    int symbol_id_within_layer = tid % num_symbols_per_layer_per_TB;
    int layer_id = calculate_layer_id(params[TB_id].port_ids[layer_cnt], params[TB_id].n_scid, params[TB_id].nlAbove16);

    uint32_t input_index = (layer_id * num_TBs + TB_id)* rounded_num_elements_per_layer;
    input_index += (symbol_id_within_layer * Tqam / 32);

    int symbol_start_bit = (symbol_id_within_layer * Tqam) % 32;
    uint32_t bit_values = (modulation_input[input_index] >> symbol_start_bit);

    if (Tqam == CUPHY_QAM_64) {
        if (symbol_start_bit == 28) { // Read 2 bits from next element. Symbol 10 (every 16)
            bit_values &= 0x0FU;
            bit_values |= ((modulation_input[input_index + 1] & 0x03U) << 4);

        } else if (symbol_start_bit == 30) { // Read 4 bits from next element. Symbol 5 (every 16)
            bit_values &= 0x03U;
            bit_values |= ((modulation_input[input_index + 1] & 0x0FU) << 2);
        }
    }
    return bit_values;
}


__device__ uint32_t flat_input_index_calc_256QAM(const uint32_t* __restrict__ modulation_input) {

    int input_index = (blockIdx.x << 6) + (threadIdx.x >> 2);
    int symbol_start_bit = ((threadIdx.x & 0x3) << 3);
    uint32_t bit_values = (modulation_input[input_index] >> symbol_start_bit);
    return bit_values;
}

__device__ uint32_t flat_input_index_calc_64QAM(const uint32_t* __restrict__ modulation_input) {

    const int element_size = sizeof(uint32_t) * 8;
    int  input_index = blockIdx.x * blockDim.x * CUPHY_QAM_64 / element_size +  threadIdx.x * CUPHY_QAM_64 / element_size;
    int symbol_start_bit = (threadIdx.x * CUPHY_QAM_64) % element_size;
    uint32_t bit_values = 0;

    bit_values = (modulation_input[input_index] >> symbol_start_bit);
    // Handle 2 misaligned cases. Every 3 32-bit elements, i.e., every 16 symbols,
    // the 5th and 10th symbols (0-based indexing) are crossing 32-bit element boundaries.
    if (symbol_start_bit == 28) { // Read 2 bits from next element. Symbol 10 (every 16)
        bit_values &= 0x0FU;
        bit_values |= ((modulation_input[input_index + 1] & 0x03U) << 4);
    } else if (symbol_start_bit == 30) { // Read 4 bits from next element. Symbol 5 (every 16)
        bit_values &= 0x03U;
        bit_values |= ((modulation_input[input_index + 1] & 0x0FU) << 2);
    }
    return bit_values;
}

__device__ uint32_t flat_input_index_calc_16QAM(const uint32_t* __restrict__ modulation_input) {

    int input_index = (blockIdx.x << 5) + (threadIdx.x >> 3); // Hardcoded for blockDim.x of 256
    int symbol_start_bit = ((threadIdx.x & 0x7) << 2); // multiply by 4 within a block; modulation order
    uint32_t bit_values = (modulation_input[input_index] >> symbol_start_bit);
    return bit_values;
}

__device__ uint32_t flat_input_index_calc_4QAM(const uint32_t* __restrict__ modulation_input) {

    // Some values are hardcoded based on blockDim.x
    // threadIdx.x >> 4 is divided by (sizeof(uint32_t)*8 / modulation_order) => only influenced by modulation order
    // blockIdx.x << 4 is multiplied by (blockDim.x * modulation_order) / (sizeof(uint32_t)*8)
    int input_index = (blockIdx.x << 4) + (threadIdx.x >> 4); // Hardcoded for blockDim.x of 256
    int symbol_start_bit = ((threadIdx.x & 0xF) << 1); // multiply by 2 within a block, the modulation order
    uint32_t bit_values = (modulation_input[input_index] >> symbol_start_bit);
    return bit_values;
}

__device__ void modulation_256QAM(const PdschDmrsParams * __restrict__ params,
                                  const uint32_t* __restrict__ modulation_input,
                                  __half2 * __restrict__ modulation_output,
                                  const struct PdschPerTbParams * workspace,
                                  int max_bits_per_layer) {


    __shared__ __half  shmem_qam_256[16];
    if (threadIdx.x < 16) {
        assert(params != nullptr);
        shmem_qam_256[threadIdx.x] = (__half) (rev_qam_256[threadIdx.x] * params[blockIdx.y].beta_qam);
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_symbols = (workspace[blockIdx.y].G >> 3); // divide by CUPHY_QAM_256
    if (tid >= num_symbols) {
        return;
    }

    int output_index = tid;
    uint32_t bit_values;
    if (params[blockIdx.y].num_Rbs != 0) {
        output_index = output_index_calc(params[blockIdx.y].num_Rbs, params[blockIdx.y].symbol_number + params[blockIdx.y].num_dmrs_symbols,
                       params[blockIdx.y].start_Rb, params[blockIdx.y].num_data_symbols, params);
        bit_values = input_index_calc<CUPHY_QAM_256>(max_bits_per_layer, params[blockIdx.y].num_Rbs, params[blockIdx.y].num_data_symbols, params, modulation_input, workspace);
    } else {
        bit_values = flat_input_index_calc_256QAM(modulation_input);
    }

    int x_index = map_index_8bits(bit_values);
    int y_index = map_index_8bits(bit_values >> 1);

    modulation_output[output_index] = make_complex<__half2>::create(shmem_qam_256[x_index],
                                                                    shmem_qam_256[y_index]);
}

__device__ void modulation_64QAM(const PdschDmrsParams * __restrict__ params, const uint32_t* __restrict__ modulation_input,
                                 __half2 * __restrict__ modulation_output,
                                  const struct PdschPerTbParams * workspace,
                                 int max_bits_per_layer) {


    __shared__ __half shmem_qam_64[8];
    if (threadIdx.x < 8) {
        assert(params != nullptr);
        shmem_qam_64[threadIdx.x] = (__half) (rev_qam_64[threadIdx.x] * params[blockIdx.y].beta_qam);
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_symbols = (workspace[blockIdx.y].G / CUPHY_QAM_64);
    if (tid >= num_symbols) {
        return;
    }

    int output_index = tid;
    uint32_t bit_values;
    if (params[blockIdx.y].num_Rbs != 0) {
        output_index = output_index_calc(params[blockIdx.y].num_Rbs, params[blockIdx.y].symbol_number + params[blockIdx.y].num_dmrs_symbols,
                       params[blockIdx.y].start_Rb, params[blockIdx.y].num_data_symbols, params);
        bit_values = input_index_calc<CUPHY_QAM_64>(max_bits_per_layer, params[blockIdx.y].num_Rbs, params[blockIdx.y].num_data_symbols, params, modulation_input, workspace);
    } else {
        bit_values = flat_input_index_calc_64QAM(modulation_input);
    }

    int x_index = map_index_6bits(bit_values);
    int y_index = map_index_6bits(bit_values >> 1);

    modulation_output[output_index] = make_complex<__half2>::create(shmem_qam_64[x_index],
                                                                    shmem_qam_64[y_index]);

}


__device__ void modulation_16QAM(const PdschDmrsParams * __restrict__ params,
                                 const uint32_t* __restrict__ modulation_input,
                                 __half2 * __restrict__ modulation_output,
                                  const struct PdschPerTbParams * workspace,
                                 int max_bits_per_layer) {

    __shared__ __half shmem_qam_16[8];
    if (threadIdx.x < 8) {
        assert(params != nullptr);
        shmem_qam_16[threadIdx.x] = (__half) (rev_qam_16_long[threadIdx.x] * params[blockIdx.y].beta_qam);
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x; //symbols within a TB
    int num_symbols = (workspace[blockIdx.y].G >> 2); // divide by CUPHY_QAM_16
    if (tid >= num_symbols) {
        return;
    }

    int output_index = tid;
    uint32_t bit_values;
    if (params[blockIdx.y].num_Rbs != 0) {
        output_index = output_index_calc(params[blockIdx.y].num_Rbs, params[blockIdx.y].symbol_number + params[blockIdx.y].num_dmrs_symbols,
                                         params[blockIdx.y].start_Rb, params[blockIdx.y].num_data_symbols, params);
        bit_values = input_index_calc<CUPHY_QAM_16>(max_bits_per_layer, params[blockIdx.y].num_Rbs, params[blockIdx.y].num_data_symbols, params, modulation_input, workspace);
    } else {
        bit_values = flat_input_index_calc_16QAM(modulation_input);
    }

    modulation_output[output_index] = make_complex<__half2>::create(shmem_qam_16[bit_values & 0x05],
                                                                    shmem_qam_16[(bit_values >> 1) & 0x05]);
}

__device__ void modulation_QPSK(const PdschDmrsParams * __restrict__ params,
                                const uint32_t* __restrict__ modulation_input,
                                __half2 * __restrict__ modulation_output,
                                const struct PdschPerTbParams * workspace,
                                int max_bits_per_layer) {

    assert(params != nullptr);
#if FLOAT_COMP
    float reciprocal_sqrt2 = 0.707106781186547f * params[blockIdx.y].beta_qam;
#else
    __half reciprocal_sqrt2 = __hmul(hrsqrt(2), __float2half(params[blockIdx.y].beta_qam)); //0.707106781186547;
#endif

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_symbols = (workspace[blockIdx.y].G >> 1); // divide by CUPHY_QAM_4
    if (tid >= num_symbols) {
        return;
    }

    int output_index = tid;
    uint32_t bit_values;
    if (params[blockIdx.y].num_Rbs != 0) {
        output_index = output_index_calc(params[blockIdx.y].num_Rbs, params[blockIdx.y].symbol_number + params[blockIdx.y].num_dmrs_symbols,
                       params[blockIdx.y].start_Rb, params[blockIdx.y].num_data_symbols, params);
        bit_values = input_index_calc<CUPHY_QAM_4>(max_bits_per_layer, params[blockIdx.y].num_Rbs, params[blockIdx.y].num_data_symbols, params, modulation_input, workspace);
    } else {
        bit_values = flat_input_index_calc_4QAM(modulation_input);
    }

    __half2 tmp_val;
    tmp_val.x = ((bit_values & 0x1) == 0) ? reciprocal_sqrt2 : -reciprocal_sqrt2;
    tmp_val.y = ((bit_values & 0x2) == 0) ? reciprocal_sqrt2 : -reciprocal_sqrt2;
    modulation_output[output_index] = tmp_val;
}


/* For now, when d_params of the modulationDescr_t is nullptr, QAM symbols are allocated contiguously in the modulation_output buffer,
   assuming (i.e., 1 TB) w/ num_bits bits, where num_bits % modulation_order = 0.

   If d_params is non nullptr, then modulation_output is the {273*12, 14, 16} tensor and
   the data symbols are allocated in the appropriate location (i.e., correct data symbol, startRb, etc.
   for the allocated Rbs.
*/
__global__ void modulation_mapper(modulationDescr_t* p_desc) {

    modulationDescr_t& desc = *p_desc;
    const PdschDmrsParams* params = desc.d_params;
    const uint32_t* modulation_input = desc.modulation_input;
    const struct PdschPerTbParams* workspace = desc.workspace;
    __half2* modulation_output = desc.modulation_output;
    int max_bits_per_layer = desc.max_bits_per_layer;

    if (modulation_input == nullptr) return;

    int TB_id = blockIdx.y;
    int modulation_order = workspace[TB_id].Qm;
    if (modulation_order == CUPHY_QAM_4) {
	modulation_QPSK(params, modulation_input, modulation_output, workspace, max_bits_per_layer);
    } else if (modulation_order == CUPHY_QAM_16) {
	modulation_16QAM(params, modulation_input, modulation_output, workspace, max_bits_per_layer);
    } else if (modulation_order == CUPHY_QAM_64) {
	modulation_64QAM(params, modulation_input, modulation_output, workspace, max_bits_per_layer);
    } else if (modulation_order == CUPHY_QAM_256) {
	modulation_256QAM(params, modulation_input, modulation_output, workspace, max_bits_per_layer);
    } 
}

cuphyStatus_t CUPHYWINAPI cuphySetupModulation(cuphyModulationLaunchConfig_t modulationLaunchConfig,
                             PdschDmrsParams * d_params,
                             const cuphyTensorDescriptor_t input_desc, /* not used */
                             const void* modulation_input,
                             int max_num_symbols,
                             int max_bits_per_layer,
                             int num_TBs,
                             PdschPerTbParams* workspace, /* num_TBs entries; the only fields used are G (# rate matched bits) and Qm (modulation_order) */
                             cuphyTensorDescriptor_t output_desc, /* not used */
                             void* modulation_output,
                             void* cpu_desc,
                             void* gpu_desc,
                             uint8_t enable_desc_async_copy,
                             cudaStream_t strm)
{

    // Calling cuphySetupModulation with d_params == nullptr is permitted and assumes
    // output is a contiguous buffer rather than a 3D {3276, 14, 4} tensor as in the PdschTx example.
    if ((workspace == nullptr)|| (modulation_output == nullptr)) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    modulationLaunchConfig->m_kernelArgs[0] = &(modulationLaunchConfig->m_desc);
    modulationLaunchConfig->m_kernelNodeParams.extra = nullptr;
    modulationLaunchConfig->m_kernelNodeParams.kernelParams = &(modulationLaunchConfig->m_kernelArgs[0]);

    //Set up CPU descriptor. Assumes it has been pre-allocated. Easier for a pipeline, than for a standalone component setting.
    modulationDescr_t& desc = *(static_cast<modulationDescr_t*>(cpu_desc));
    desc.d_params = d_params; // device pointer
    desc.modulation_input = (uint32_t*)modulation_input;
    desc.workspace = workspace;
    desc.modulation_output = (__half2*)modulation_output;
    desc.max_bits_per_layer = max_bits_per_layer;

    // Optional descriptor copy to GPU memory
    // When running as part of a pipeline, it's better to do a single copy of all descriptors in the pipeline.
    if (enable_desc_async_copy) {
        CUDA_CHECK(cudaMemcpyAsync(gpu_desc, cpu_desc, sizeof(modulationDescr_t), cudaMemcpyHostToDevice, strm));
    }
    modulationLaunchConfig->m_desc = static_cast<modulationDescr_t*>(gpu_desc);

    // When running modulation as part of the PDSCH pipeline (!enable_desc_async_copy),
    // update the kernel function only during the first call to cuphySetupModulation.
    // For standalone components, update it on every setup call.
    cudaFunction_t modulation_device_function;
    if (enable_desc_async_copy || (modulationLaunchConfig->m_kernelNodeParams.func == nullptr)) {
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&modulation_device_function, reinterpret_cast<void*>(modulation_mapper)));}
        modulationLaunchConfig->m_kernelNodeParams.func = modulation_device_function;
    }

    // Have a thread per symbol. NB: some values in the kernels are hardcoded for 256 threads per block.
    int num_threads = (max_num_symbols >= 256) ? 256 : max_num_symbols;
    modulationLaunchConfig->m_kernelNodeParams.blockDimX = num_threads;
    modulationLaunchConfig->m_kernelNodeParams.blockDimY = 1;
    modulationLaunchConfig->m_kernelNodeParams.blockDimZ = 1;
    modulationLaunchConfig->m_kernelNodeParams.gridDimX = div_round_up(max_num_symbols, num_threads);
    modulationLaunchConfig->m_kernelNodeParams.gridDimY = num_TBs;
    modulationLaunchConfig->m_kernelNodeParams.gridDimZ = 1;
    modulationLaunchConfig->m_kernelNodeParams.sharedMemBytes = 0;

    return CUPHY_STATUS_SUCCESS;

}

namespace
{

////////////////////////////////////////////////////////////////////////
// QAM_traits
// See 3GPP 38.211 Section 5.1
template <int TLog2QAM> struct QAM_traits;
//template <> struct QAM_traits<8> { static constexpr float A = 0.076696498884737; /* 1.0 / sqrt(170.0) */ };
//template <> struct QAM_traits<6> { static constexpr float A = 0.154303349962092; /* 1.0 / sqrt(42.0)  */ };
//template <> struct QAM_traits<4> { static constexpr float A = 0.316227766016838; /* 1.0 / sqrt(10.0)  */ };
template <> struct QAM_traits<2> { static constexpr float A = 0.707106781186547; /* 1.0 / sqrt(2.0)   */ };
template <> struct QAM_traits<1> { static constexpr float A = 0.707106781186547; /* 1.0 / sqrt(2.0)   */ };

////////////////////////////////////////////////////////////////////////
// The lowest common multiple of 6 (the number of bits in QAM64) and
// 32 (the number of threads in a warp) is 96. (All other modulations
// correspond to numbers of bits that are powers of 2, and thus divide
// 32 evenly.) Batching words in multiples of 3 guarantees that no input
// bit sequence will cross batch boundaries.
const int WORDS_PER_THREAD = 3;
const int BITS_PER_WORD    = 32;

////////////////////////////////////////////////////////////////////////
// extract_bits()
// Retrieves a set of bits from the source buffer ('bits') from a symbol
// location given by 'idx'.
template <int TLOG2_QAM>
__device__ uint32_t extract_bits(const uint32_t* bits, int symbolIndex)
{
    const uint32_t MASK        = (1 << TLOG2_QAM) - 1;
    const int      BIT_IDX     = symbolIndex * TLOG2_QAM;
    const int      WORD_IDX    = BIT_IDX / BITS_PER_WORD;
    const int      WORD_OFFSET = BIT_IDX % BITS_PER_WORD;
    uint32_t       value       = ((bits[WORD_IDX] >> WORD_OFFSET) & MASK);
    //KERNEL_PRINT("threadIdx.x = %u, symbolIndex = %i, WORD_IDX = %i, WORD_OFFSET = %i, value = 0x%02X\n",
    //              threadIdx.x,
    //              symbolIndex,
    //              WORD_IDX,
    //              WORD_OFFSET,
    //              value);
    return value;
}
// Specialization for QAM64, because the source bits may reside in
// different source words.
// symbol start
// 0      0
// 1      6
// 2      12
// 3      18
// 4      24
// 5      30
//        --------- word boundary at 32
// 6      36
// etc.
template <>
__device__ uint32_t extract_bits<6>(const uint32_t* bits, int symbolIndex)
{
    const int      LOG2_QAM    = 6;
    const uint32_t MASK        = (1 << LOG2_QAM) - 1;
    const int      BIT_IDX     = symbolIndex * LOG2_QAM;
    const int      WORD_IDX    = BIT_IDX / 32;
    const int      WORD_OFFSET = BIT_IDX % 32;
    // Indexing to WORD_IDX + 1, requires an "extra" word of shared mem.
    // We could instead conditionally load...
    uint32_t       value       = MASK & __funnelshift_r(bits[WORD_IDX],
                                                        bits[WORD_IDX + 1],
                                                        WORD_OFFSET);
    //KERNEL_PRINT("threadIdx.x, symbolIndex = %i, WORD_IDX = %i, WORD_OFFSET = %i, value = 0x%02X\n",
    //             threadIdx.x,
    //             symbolIndex,
    //             WORD_IDX,
    //             WORD_OFFSET,
    //             value);
    return value;
}

////////////////////////////////////////////////////////////////////////
// mod_table_none
struct mod_table_none
{
};

////////////////////////////////////////////////////////////////////////
// mod_table_QAM16
template <typename TOut> struct mod_table_QAM16
{
    typedef typename scalar_from_complex<TOut>::type scalar_t;
    
    static const int SZ = sizeof(rev_qam_16_long) / sizeof(rev_qam_16_long[0]);
    scalar_t         qam16_values[SZ];
    //-------------------------------------------------------------------
    // init()
    // Initialize table with const memory
    __device__ void init()
    {
        if (threadIdx.x < SZ)
        {
            qam16_values[threadIdx.x] = static_cast<scalar_t>(rev_qam_16_long[threadIdx.x]);
        }
    }
};

////////////////////////////////////////////////////////////////////////
// mod_table_QAM64
template <typename TOut> struct mod_table_QAM64
{
    typedef typename scalar_from_complex<TOut>::type scalar_t;
    
    static const int SZ = sizeof(rev_qam_64) / sizeof(rev_qam_64[0]);
    scalar_t         qam64_values[SZ];
    //-------------------------------------------------------------------
    // init()
    // Initialize table with const memory
    __device__ void init()
    {
        if (threadIdx.x < SZ)
        {
            qam64_values[threadIdx.x] = static_cast<scalar_t>(rev_qam_64[threadIdx.x]);
        }
    }
};

////////////////////////////////////////////////////////////////////////
// mod_table_QAM256
template <typename TOut> struct mod_table_QAM256
{
    typedef typename scalar_from_complex<TOut>::type scalar_t;
    
    static const int SZ = sizeof(rev_qam_256) / sizeof(rev_qam_256[0]);
    scalar_t         qam256_values[SZ];
    //-------------------------------------------------------------------
    // init()
    // Initialize table with const memory
    __device__ void init()
    {
        if (threadIdx.x < SZ)
        {
            qam256_values[threadIdx.x] = static_cast<scalar_t>(rev_qam_256[threadIdx.x]);
        }
    }
};

////////////////////////////////////////////////////////////////////////
// bits_to_symbol()
// QAM-specific mapping from bits to a complex symbol. (Some QAMs may
// use a table lookup, others may not.)
template <int TLOG2_QAM, typename TOut> struct bits_to_symbol;
// BPSK
template <typename TOut> struct bits_to_symbol<1, TOut>
{
    __device__
    static void map(TOut* dst, uint32_t bits, const mod_table_none&)
    {
        *dst = make_complex<TOut>::create((0 == bits) ? QAM_traits<1>::A : -QAM_traits<1>::A,
                                          (0 == bits) ? QAM_traits<1>::A : -QAM_traits<1>::A);
    }
};
// QPSK
template <typename TOut> struct bits_to_symbol<2, TOut>
{
    __device__
    static void map(TOut* dst, uint32_t bits, const mod_table_none&)
    {
        *dst = make_complex<TOut>::create((0 == (bits & 0x1)) ? QAM_traits<2>::A : -QAM_traits<2>::A,
                                          (0 == (bits & 0x2)) ? QAM_traits<2>::A : -QAM_traits<2>::A);
    }
};
// QAM16
template <typename TOut> struct bits_to_symbol<4, TOut>
{
    __device__
    static void map(TOut* dst, uint32_t bits, const mod_table_QAM16<TOut>& tbl)
    {
        *dst = make_complex<TOut>::create(tbl.qam16_values[bits        & 0x05],
                                          tbl.qam16_values[(bits >> 1) & 0x05]);
    }
};
// QAM64
template <typename TOut> struct bits_to_symbol<6, TOut>
{
    __device__
    static void map(TOut* dst, uint32_t bits, const mod_table_QAM64<TOut>& tbl)
    {
        int x_index = map_index_6bits(bits);
        int y_index = map_index_6bits(bits >> 1);

        *dst = make_complex<TOut>::create(tbl.qam64_values[x_index],
                                          tbl.qam64_values[y_index]);
    }
};
// QAM256
template <typename TOut> struct bits_to_symbol<8, TOut>
{
    __device__
    static void map(TOut* dst, uint32_t bits, const mod_table_QAM256<TOut>& tbl)
    {
        int x_index = map_index_8bits(bits);
        int y_index = map_index_8bits(bits >> 1);

        *dst = make_complex<TOut>::create(tbl.qam256_values[x_index],
                                          tbl.qam256_values[y_index]);
    }
};


////////////////////////////////////////////////////////////////////////
// block_symbol_modulator
// Structure to perform modulation on a block of source bits in shared
// memory. (Loading the bits is indepedent of modulation, and assumed
// to be performed elsewhere.)
template <int TLOG2_QAM, typename TOut, int THREADS_PER_CTA> struct block_symbol_modulator
{
    static const int SYMBOLS_PER_BATCH = (WORDS_PER_THREAD * BITS_PER_WORD) / TLOG2_QAM;

    template <class TTable>
    __device__ static void modulate(TOut* sym, const TOut* symEnd, const uint32_t* srcBits, const TTable& t)
    {
        for(int i = 0; i < SYMBOLS_PER_BATCH; ++i)
        {
            const int SYMBOL_IDX = threadIdx.x + (i * THREADS_PER_CTA);
            uint32_t bits        = extract_bits<TLOG2_QAM>(srcBits, SYMBOL_IDX);
            //KERNEL_PRINT_GRID_ONCE("SYMBOL_IDX = %i, value = 0x%X\n", SYMBOL_IDX, bits);
            if((sym + SYMBOL_IDX) < symEnd)
            {
                bits_to_symbol<TLOG2_QAM, TOut>::map(sym + SYMBOL_IDX, bits, t);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////
// sym_mod_util()
// Simplified "utility function" kernel for modulating symbols in
// offline/example program situations.
template <unsigned int THREADS_PER_CTA, typename TOut>
__global__ void sym_mod_util(const tensor_layout_any tDstLayout,
                             TOut*                   dst,
                             const tensor_layout_any tSrcLayout,
                             const uint32_t*         src,
                             int                     log2_QAM,
                             int                     symbolsPerBatch)
{
    // "+1" required for current implementation that uses funnel shift
    // for QAM64. Can be avoided if necessary...
    __shared__ uint32_t               block_src[THREADS_PER_CTA * WORDS_PER_THREAD + 1];
    __shared__ mod_table_QAM16<TOut>  tbl_QAM16;
    __shared__ mod_table_QAM64<TOut>  tbl_QAM64;
    __shared__ mod_table_QAM256<TOut> tbl_QAM256;
    
    mod_table_none no_table;
    
    typedef block_symbol_modulator<1, TOut, THREADS_PER_CTA> mod_BPSK_t;
    typedef block_symbol_modulator<2, TOut, THREADS_PER_CTA> mod_QPSK_t;
    typedef block_symbol_modulator<4, TOut, THREADS_PER_CTA> mod_QAM16_t;
    typedef block_symbol_modulator<6, TOut, THREADS_PER_CTA> mod_QAM64_t;
    typedef block_symbol_modulator<8, TOut, THREADS_PER_CTA> mod_QAM256_t;

    tbl_QAM16.init();
    tbl_QAM64.init();
    tbl_QAM256.init();

    //------------------------------------------------------------------
    // Each CTA will process a "column" of input data, one "batch" at a
    // time.
    const int       COL_IDX      = blockIdx.x;
    const uint32_t* blockAddr    = src + tSrcLayout.offset({0, COL_IDX});
    const uint32_t* SRC_COL_END  = blockAddr + tSrcLayout.dimensions[0];
    TOut*           symAddr      = dst + tDstLayout.offset({0, COL_IDX});
    const TOut*     SYM_ADDR_END = symAddr + tDstLayout.dimensions[0];
    //------------------------------------------------------------------
    while(blockAddr < SRC_COL_END)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Copy data from global to shared and syncthreads
        block_copy_N_sync_check<uint32_t, WORDS_PER_THREAD>(block_src,
                                                            blockAddr,
                                                            SRC_COL_END);
        //print_array_sync("SRC", "0x%X", block_src, 32);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Dispatch to the approprate QAM function
        switch(log2_QAM)
        {
        // Coverage note: this default path is intentionally unreachable via public API.
        // cuphyModulateSymbol() validates log2_QAM to {1,2,4,6,8} before launching sym_mod_util,
        // so invalid values never reach this switch in normal/tested flows.
        /*VCAST_DONT_INSTRUMENT_START*/
        default:
        /*VCAST_DONT_INSTRUMENT_END*/
        case 1:  mod_BPSK_t::modulate  (symAddr, SYM_ADDR_END, block_src, no_table);   break;
        case 2:  mod_QPSK_t::modulate  (symAddr, SYM_ADDR_END, block_src, no_table);   break;
        case 4:  mod_QAM16_t::modulate (symAddr, SYM_ADDR_END, block_src, tbl_QAM16);  break;
        case 6:  mod_QAM64_t::modulate (symAddr, SYM_ADDR_END, block_src, tbl_QAM64);  break;
        case 8:  mod_QAM256_t::modulate(symAddr, SYM_ADDR_END, block_src, tbl_QAM256); break;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Advance to the next batch
        blockAddr += (THREADS_PER_CTA * WORDS_PER_THREAD);
        symAddr   += symbolsPerBatch;
    }


}

} // namespace

namespace cuphy_i
{

////////////////////////////////////////////////////////////////////////
// symbol_modulate()
cuphyStatus_t symbol_modulate(const tensor_desc& tSym,
                              void*              pSym,
                              const tensor_desc& tBits,
                              const void*        pBits,
                              int                log2_QAM,
                              cudaStream_t       strm)
{
    const int NUM_COLUMNS     = tSym.layout().dimensions[1];
    const int THREADS_PER_CTA = 32;
    const int SYMBOLS_PER_BATCH = (THREADS_PER_CTA * WORDS_PER_THREAD * BITS_PER_WORD) / log2_QAM;
    //------------------------------------------------------------------
    // Get a uint32_t word layout from the source CUPHY_BIT tensors
    tensor_layout_any layoutSrcWords = word_layout_from_bit_layout(tBits.layout());
    //printf("source dim[0]: %i\n", layoutSrcWords.dimensions[0]);
    //printf("symbolsPerBatch: %i\n", (3 * 32) / log2_QAM);

    if(CUPHY_C_32F == tSym.type())
    {
        sym_mod_util<32, cuComplex><<<NUM_COLUMNS, THREADS_PER_CTA, 0, strm>>>(tSym.layout(),
                                                                               static_cast<cuComplex*>(pSym),
                                                                               layoutSrcWords,
                                                                               static_cast<const uint32_t*>(pBits),
                                                                               log2_QAM,
                                                                               SYMBOLS_PER_BATCH);
        
    }
    else if(CUPHY_C_16F == tSym.type())
    {
        sym_mod_util<32, __half2><<<NUM_COLUMNS, THREADS_PER_CTA, 0, strm>>>(tSym.layout(),
                                                                             static_cast<__half2*>(pSym),
                                                                             layoutSrcWords,
                                                                             static_cast<const uint32_t*>(pBits),
                                                                             log2_QAM,
                                                                             SYMBOLS_PER_BATCH);
    }

    // Coverage note: this branch is guarded by the public API.
    // cuphyModulateSymbol() performs the same type validation and returns
    // CUPHY_STATUS_UNSUPPORTED_TYPE before reaching cuphy_i::symbol_modulate().
    /*VCAST_DONT_INSTRUMENT_START*/
    else
    {
        return CUPHY_STATUS_UNSUPPORTED_TYPE;
    }
    /*VCAST_DONT_INSTRUMENT_END*/

    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

} // namespace cuphy_i
