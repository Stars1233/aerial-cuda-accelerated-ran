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

#if !defined(LDPC_2_HPP_INCLUDED_)
#define LDPC_2_HPP_INCLUDED_

#include "ldpc.hpp"
#include "ldpc2_et_context.cuh"
#include "nrLDPC_templates.cuh"

////////////////////////////////////////////////////////////////////////
// Functions specific to the "LDPC2" family of implementations
namespace ldpc2
{

union word_t
{
    float                 f32;
    uint32_t              u32;
    int32_t               i32;
    __half_raw            f16;
    __half2_raw           f16x2;
    ushort2               u16x2;
    short2                i16x2;
    __nv_fp8x4_e5m2       fp8x4_e5m2;
    __nv_fp8x4_e4m3       fp8x4_e4m3;
};

template <typename T> __device__ T& word_t_as(word_t& w);

template <> inline __device__ float&           word_t_as<float>          (word_t& w) { return w.f32; }
template <> inline __device__ uint32_t&        word_t_as<uint32_t>       (word_t& w) { return w.u32; }
template <> inline __device__ int32_t&         word_t_as<int32_t>        (word_t& w) { return w.i32; }
template <> inline __device__ __half_raw&      word_t_as<__half_raw>     (word_t& w) { return w.f16; }
template <> inline __device__ __half2_raw&     word_t_as<__half2_raw>    (word_t& w) { return w.f16x2; }
template <> inline __device__ ushort2&         word_t_as<ushort2>        (word_t& w) { return w.u16x2; }
template <> inline __device__ short2&          word_t_as<short2>         (word_t& w) { return w.i16x2; }
template <> inline __device__ __nv_fp8x4_e5m2& word_t_as<__nv_fp8x4_e5m2>(word_t& w) { return w.fp8x4_e5m2; }
template <> inline __device__ __nv_fp8x4_e4m3& word_t_as<__nv_fp8x4_e4m3>(word_t& w) { return w.fp8x4_e4m3; }

template <typename T> __device__ const T& word_t_as(const word_t& w);

template <> inline __device__ const float&           word_t_as<float>          (const word_t& w) { return w.f32; }
template <> inline __device__ const uint32_t&        word_t_as<uint32_t>       (const word_t& w) { return w.u32; }
template <> inline __device__ const int32_t&         word_t_as<int32_t>        (const word_t& w) { return w.i32; }
template <> inline __device__ const __half_raw&      word_t_as<__half_raw>     (const word_t& w) { return w.f16; }
template <> inline __device__ const __half2_raw&     word_t_as<__half2_raw>    (const word_t& w) { return w.f16x2; }
template <> inline __device__ const ushort2&         word_t_as<ushort2>        (const word_t& w) { return w.u16x2; }
template <> inline __device__ const short2&          word_t_as<short2>         (const word_t& w) { return w.i16x2; }
template <> inline __device__ const __nv_fp8x4_e5m2& word_t_as<__nv_fp8x4_e5m2>(const word_t& w) { return w.fp8x4_e5m2; }
template <> inline __device__ const __nv_fp8x4_e4m3& word_t_as<__nv_fp8x4_e4m3>(const word_t& w) { return w.fp8x4_e4m3; }

////////////////////////////////////////////////////////////////////////
// LDPC_kernel_params
struct LDPC_kernel_params
{
    const char* input_llr;
    char*       out;
    int         input_llr_stride_elements;
    int         output_stride_words;
    int         max_iterations;
    int         outputs_per_codeword;       // The number of outputs/ints per codeword.
    word_t      norm;
    void*       workspace;
    int         Z;
    int         z2;
    int         z4;
    int         z8;
    int         z16;
    int         mbz8;
    int         mbz16;
    int         num_parity_nodes;
    int         num_var_nodes;             // (1 == BG) ? (22 + mb) : (10 + mb)
    int         K;                         // number of bits: (1 == BG) ? (22 * Z) : (10 * Z)
    int         Kb;                        // num info nodes (22 for BG 1, {6, 8, 9, 10} for BG2)
    int         KbZ;
    int         Z_var;                     // Z * num_var_nodes
    int         Z_var_szelem;              // Z * num_var_nodes * sizeof(app_t)
    int         num_codewords;
    int         soft_out_stride_elements;
    float       clamp_value;
    void*       soft_out;
    LDPC_kernel_params(const cuphyLDPCDecodeConfigDesc_t& cfg,
                       const_tensor_pair&                 tLLR,
                       LDPC_output_t&                     tDst,
                       const cuphy_optional<tensor_pair>& optSoftOutputs) :
        input_llr((const char*)tLLR.second),
        out((char*)tDst.addr()),
        input_llr_stride_elements(tLLR.first.get().layout().strides[1]),
        output_stride_words(tDst.layout().strides[0]),
        max_iterations(cfg.max_iterations),
        outputs_per_codeword(((cfg.Kb * cfg.Z) + 31) / 32),
        workspace(cfg.workspace),
        Z(cfg.Z),
        z2(cfg.Z * 2),
        z4(cfg.Z * 4),
        z8(cfg.Z * 8),
        z16(cfg.Z * 16),
        mbz8(cfg.num_parity_nodes * cfg.Z * 8),
        mbz16(cfg.num_parity_nodes * cfg.Z * 16),
        num_parity_nodes(cfg.num_parity_nodes),
        num_var_nodes(cfg.num_parity_nodes + ((1 == cfg.BG) ? 22 : 10)),
        K((1 == cfg.BG) ? (22 * cfg.Z) : (10 * cfg.Z)),
        Kb(cfg.Kb),
        KbZ(cfg.Kb*cfg.Z),
        Z_var(cfg.Z * num_var_nodes),
        Z_var_szelem(cfg.Z * num_var_nodes * ((CUPHY_R_16F == cfg.llr_type) ?  2 : 4)),
        num_codewords(tLLR.first.get().layout().dimensions[1]),
        soft_out_stride_elements(static_cast<bool>(optSoftOutputs) ? optSoftOutputs.value().first.get().layout().strides[1] : 0),
        clamp_value(cfg.clamp_value),
        soft_out(static_cast<bool>(optSoftOutputs) ? optSoftOutputs.value().second : nullptr)
    {
        // When the LLR type is F32, we convert LLR values to FP16 when
        // loading.
        // The cuPHY API dictates that the config normalization should match
        // the LLR type, so we convert to FP16 here if necessary.
        norm.f16x2 = (CUPHY_R_32F == cfg.llr_type)                            ?
                     static_cast<__half2_raw>(__float2half2_rn(cfg.norm.f32)) :
                     cfg.norm.f16x2;
    }
    LDPC_kernel_params(const cuphyLDPCDecodeConfigDesc_t& cfg,
                       int                                input_stride_elem,
                       const void*                        input_addr,
                       int                                out_stride_words,
                       void*                              out_addr,
                       int                                soft_out_stride_elem,
                       void*                              soft_out_addr,
                       int                                num_cw) :
        input_llr((const char*)input_addr),
        out((char*)out_addr),
        input_llr_stride_elements(input_stride_elem),
        output_stride_words(out_stride_words),
        max_iterations(cfg.max_iterations),
        outputs_per_codeword(((cfg.Kb * cfg.Z) + 31) / 32),
        workspace(cfg.workspace),
        Z(cfg.Z),
        z2(cfg.Z * 2),
        z4(cfg.Z * 4),
        z8(cfg.Z * 8),
        z16(cfg.Z * 16),
        mbz8(cfg.num_parity_nodes * cfg.Z * 8),
        mbz16(cfg.num_parity_nodes * cfg.Z * 16),
        num_parity_nodes(cfg.num_parity_nodes),
        num_var_nodes(cfg.num_parity_nodes + ((1 == cfg.BG) ? 22 : 10)),
        K((1 == cfg.BG) ? (22 * cfg.Z) : (10 * cfg.Z)),
        Kb(cfg.Kb),
        KbZ(cfg.Kb*cfg.Z),
        Z_var(cfg.Z * num_var_nodes),
        Z_var_szelem(cfg.Z * num_var_nodes * ((CUPHY_R_16F == cfg.llr_type) ?  2 : 4)),
        num_codewords(num_cw),
        soft_out_stride_elements(soft_out_stride_elem),
        clamp_value(cfg.clamp_value),
        soft_out(soft_out_addr)
    {
        // When the LLR type is F32, we convert LLR values to FP16 when
        // loading.
        // The cuPHY API dictates that the config normalization should match
        // the LLR type, so we convert to FP16 here if necessary.
        norm.f16x2 = (CUPHY_R_32F == cfg.llr_type)                            ?
                     static_cast<__half2_raw>(__float2half2_rn(cfg.norm.f32)) :
                     cfg.norm.f16x2;
    }
};

////////////////////////////////////////////////////////////////////////
// shmem_llr_buffer_size()
// Returns the size required to store LLR/APP values in shared memory only.
// For total shared memory including early-termination context, use
// shmem_size_with_et_context(shmem_llr_buffer_size(...)) or add the ET
// context to your kernel's base size (e.g. LLR+C2V) via shmem_size_with_et_context().
CUDA_BOTH_INLINE
uint32_t shmem_llr_buffer_size(uint32_t vnodes, uint32_t Z, uint32_t elem_size)
{
    return vnodes * Z * elem_size;
}

////////////////////////////////////////////////////////////////////////
// shmem_size_with_et_context()
// Given a base shared memory size (e.g. LLR buffer only, or LLR+C2V),
// returns the total size including the early-termination context when
template <typename EtContextT = ldpc_et_context_t>
CUDA_BOTH_INLINE
uint32_t shmem_size_with_et_context(uint32_t base_size)
{
    base_size = round_up_to_next(base_size,
        static_cast<uint32_t>(alignof(EtContextT)));
    base_size += sizeof(EtContextT);
    return base_size;
}

constexpr uint32_t LDPC_ACCESSORY_FEATURE_FLAGS =
    CUPHY_LDPC_DECODE_EARLY_TERM |
    CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS |
    CUPHY_LDPC_DECODE_WRITE_ITER_COUNT |
    CUPHY_LDPC_DECODE_ET_LATENCY_DEBUG;

////////////////////////////////////////////////////////////////////////
// use_accessory_kernel()
// Selects the full TB kernel whenever an accessory is requested. The force
// flag selects that kernel without enabling an accessory, for A/B timing.
[[nodiscard]] CUDA_BOTH_INLINE
bool use_accessory_kernel(uint32_t flags)
{
    return 0 != (flags &
        (LDPC_ACCESSORY_FEATURE_FLAGS | CUPHY_LDPC_DECODE_FORCE_ET_KERNEL));
}

template <bool ENABLE_ACCESSORY_FEATURES>
[[nodiscard]] CUDA_BOTH_INLINE
bool accessory_flag_enabled(uint32_t flags, uint32_t flag)
{
    return ENABLE_ACCESSORY_FEATURES && (0 != (flags & flag));
}

template <typename EtContextT = ldpc_et_context_t>
[[nodiscard]] CUDA_BOTH_INLINE
uint32_t shmem_size_for_accessory_kernel(uint32_t base_size, uint32_t flags)
{
    return use_accessory_kernel(flags) ?
        shmem_size_with_et_context<EtContextT>(base_size) : base_size;
}

// TB launch paths already calculate a size that includes the ET context.
// The lean kernel needs only the aligned base region preceding that context.
template <typename EtContextT = ldpc_et_context_t>
[[nodiscard]] CUDA_BOTH_INLINE
uint32_t shmem_size_for_selected_kernel(uint32_t size_with_et_context, uint32_t flags)
{
    return use_accessory_kernel(flags) ?
        size_with_et_context : size_with_et_context - sizeof(EtContextT);
}

#define LDPC_LAUNCH_TB_KERNEL(KERNEL, FLAGS, GRID, BLOCK, SHMEM, STREAM, ...) \
    do                                                                        \
    {                                                                         \
        const uint32_t ldpc_kernel_shmem =                                    \
            ldpc2::shmem_size_for_selected_kernel(SHMEM, FLAGS);              \
        if(ldpc2::use_accessory_kernel(FLAGS))                                \
        {                                                                     \
            KERNEL<<<GRID, BLOCK, ldpc_kernel_shmem, STREAM>>>(__VA_ARGS__);  \
        }                                                                     \
        else                                                                  \
        {                                                                     \
            KERNEL##_no_accessories<<<GRID, BLOCK, ldpc_kernel_shmem,         \
                STREAM>>>(__VA_ARGS__);                                       \
        }                                                                     \
    } while(false)

#define LDPC_LAUNCH_TB_KERNEL_X2(KERNEL, FLAGS, GRID, BLOCK, SHMEM, STREAM, ...) \
    do                                                                           \
    {                                                                            \
        const uint32_t ldpc_kernel_shmem =                                       \
            ldpc2::shmem_size_for_selected_kernel<ldpc_et_context_x2_t>(         \
                SHMEM, FLAGS);                                                   \
        if(ldpc2::use_accessory_kernel(FLAGS))                                   \
        {                                                                        \
            KERNEL<<<GRID, BLOCK, ldpc_kernel_shmem, STREAM>>>(__VA_ARGS__);     \
        }                                                                        \
        else                                                                     \
        {                                                                        \
            KERNEL##_no_accessories<<<GRID, BLOCK, ldpc_kernel_shmem,            \
                STREAM>>>(__VA_ARGS__);                                          \
        }                                                                        \
    } while(false)

#define LDPC_TB_KERNEL_SYMBOL(KERNEL, FLAGS)                                   \
    (ldpc2::use_accessory_kernel(FLAGS) ?                                      \
        reinterpret_cast<const void*>(KERNEL) :                               \
        reinterpret_cast<const void*>(KERNEL##_no_accessories))

#define LDPC_GET_TB_KERNEL_FUNCTION(FUNCTION, KERNEL, FLAGS)                   \
    cudaGetFuncBySymbol(&(FUNCTION), LDPC_TB_KERNEL_SYMBOL(KERNEL, FLAGS))

////////////////////////////////////////////////////////////////////////
// shmem_size_with_experimental_et_context()
// Same, for decoders whose ET path is compiled only under
// CUPHY_EXPERIMENTAL_LDPC_ET: reserve the context only when that path
// actually exists in the binary.
//
// Decoders that ship an accessory twin must NOT use this. Their ET path is
// compiled unconditionally and selected at runtime by flags, so they size
// through shmem_size_for_accessory_kernel() / shmem_size_for_selected_kernel().
[[nodiscard]] CUDA_BOTH_INLINE
uint32_t shmem_size_with_experimental_et_context(uint32_t base_size)
{
#ifdef CUPHY_EXPERIMENTAL_LDPC_ET
    return shmem_size_with_et_context(base_size);
#else
    return base_size;
#endif
}

////////////////////////////////////////////////////////////////////////
// get_llr_buffer_size_from_parity_nodes()
// Returns the size required to store LLR/APP values in shared memory
template <int BG, typename T>
CUDA_BOTH_INLINE
uint32_t get_llr_buffer_size_from_parity_nodes(uint32_t pnodes, uint32_t Z)
{
    return (max_info_nodes<BG>::value + pnodes) * Z * sizeof(T);
}

template <typename T>
CUDA_BOTH_INLINE
uint32_t get_llr_buffer_size_from_parity_nodes(int BG,
                                               int num_parity_nodes,
                                               int Z)
{
    int info_nodes = (1 == BG) ? max_info_nodes<1>::value : max_info_nodes<2>::value;
    return (info_nodes + num_parity_nodes) * Z * sizeof(T);
}

template <int BG, typename T>
CUDA_BOTH_INLINE
uint32_t get_llr_buffer_size(const LDPC_kernel_params& params)
{
    return (max_info_nodes<BG>::value + params.num_parity_nodes) * params.Z * sizeof(T);
}

template <int BG, typename T>
CUDA_BOTH_INLINE
uint32_t get_llr_buffer_size(const cuphyLDPCDecodeConfigDesc_t& cfg)
{
    return (max_info_nodes<BG>::value + cfg.num_parity_nodes) * cfg.Z * sizeof(T);
}

template <typename T>
CUDA_BOTH_INLINE
uint32_t get_llr_buffer_size(int BG, const LDPC_kernel_params& params)
{
    int info_nodes = (1 == BG) ? max_info_nodes<1>::value : max_info_nodes<2>::value;
    return (info_nodes + params.num_parity_nodes) * params.Z * sizeof(T);
}

template <typename T>
CUDA_BOTH_INLINE
uint32_t get_llr_buffer_size(const cuphyLDPCDecodeConfigDesc_t& cfg)
{
    int info_nodes = (1 == cfg.BG) ? max_info_nodes<1>::value : max_info_nodes<2>::value;
    return (info_nodes + cfg.num_parity_nodes) * cfg.Z * sizeof(T);
}

////////////////////////////////////////////////////////////////////////
// shmem_llr_buffer_size_padded()
// Returns the size required to store LLR/APP values in shared memory,
// including end padding that may be required by the shared memory
// storage type used by the loader. (For example, if the loader loads
// a uint4 from global memory and stores that to shared memory, we need
// to round up the shared memory so that the end uint4 store targets a
// valid shared memory address.) Note, however, that shared memory is
// allocated in increments of the shared memory unit allocation size,
// which according to the CUDA Occupancy Calculator, is 256. Therefore,
// rounding up as this function does may not be necessary.
template <typename T, typename TStore>
inline
uint32_t shmem_llr_buffer_size_padded(uint32_t vnodes, uint32_t Z)
{
    return round_up_to_next(vnodes * Z * sizeof(T), sizeof(TStore));
}

////////////////////////////////////////////////////////////////////////
// get_device_max_shmem_per_block_option()
// Returns the maximum shared memory per block (optin) values as would
// be returned via a query of the CUDA device properties. Returns -1
// on error.
int32_t get_device_max_shmem_per_block_optin();

} // namespace ldpc2

#endif // !defined(LDPC_2_HPP_INCLUDED_)
