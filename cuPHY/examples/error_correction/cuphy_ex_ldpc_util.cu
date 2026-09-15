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

#include "cuphy_ex_ldpc_util.hpp"

#include "dl_rate_matching/dl_rate_matching.cuh"
#include "ldpc/ldpc_params.hpp"
#include "rate_matching/derate_matching_modulo.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace cuphy_ex_ldpc_rm
{
namespace
{

constexpr uint32_t CRC24A_POLY = 0x864CFB;
constexpr uint32_t CRC24B_POLY = 0x800063;
constexpr uint32_t CRC16_POLY  = 0x1021;

template <typename T>
T div_round_up(T a, T b)
{
    return (a + b - 1) / b;
}

int q_m_from_mcs(int mcs_table, int mcs)
{
    static constexpr int table1[29] = {
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        4, 4, 4, 4, 4, 4, 4,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6
    };
    static constexpr int table2[28] = {
        2, 2, 2, 2, 2,
        4, 4, 4, 4, 4, 4,
        6, 6, 6, 6, 6, 6, 6, 6, 6,
        8, 8, 8, 8, 8, 8, 8, 8
    };
    static constexpr int table3[28] = {
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2,
        4, 4, 4, 4, 4, 4,
        6, 6, 6, 6, 6, 6, 6
    };

    if(mcs_table == 1 && mcs >= 0 && mcs < 29) { return table1[mcs]; }
    if(mcs_table == 2 && mcs >= 0 && mcs < 28) { return table2[mcs]; }
    if(mcs_table == 3 && mcs >= 0 && mcs < 28) { return table3[mcs]; }
    throw std::runtime_error("unsupported MCS table/index");
}

int target_rate_x10240_from_mcs(int mcs_table, int mcs)
{
    static constexpr int table1[29] = {
        1200, 1570, 1930, 2510, 3080, 3790, 4490, 5260, 6020, 6790,
        3400, 3780, 4340, 4900, 5530, 6160, 6580,
        4380, 4660, 5170, 5670, 6160, 6660, 7190, 7720, 8220, 8730, 9100, 9480
    };
    static constexpr int table2[28] = {
        1200, 1930, 3080, 4490, 6020,
        3780, 4340, 4900, 5530, 6160, 6580,
        4660, 5170, 5670, 6160, 6660, 7190, 7720, 8220, 8730,
        6825, 7110, 7540, 7970, 8410, 8850, 9165, 9480
    };
    static constexpr int table3[28] = {
        300, 400, 500, 640, 780, 990, 1200, 1570, 1930, 2510,
        3080, 3790, 4490, 5260, 6020,
        3400, 3780, 4340, 4900, 5530, 6160,
        4380, 4660, 5170, 5670, 6160, 6660, 7190
    };

    if(mcs_table == 1 && mcs >= 0 && mcs < 29) { return table1[mcs]; }
    if(mcs_table == 2 && mcs >= 0 && mcs < 28) { return table2[mcs]; }
    if(mcs_table == 3 && mcs >= 0 && mcs < 28) { return table3[mcs]; }
    throw std::runtime_error("unsupported MCS table/index");
}

uint32_t compute_crc(const uint8_t* bits, int num_bits, int crc_len, uint32_t poly)
{
    const uint32_t msb_mask = 1u << (crc_len - 1);
    const uint32_t all_mask = (1u << crc_len) - 1u;
    uint32_t crc = 0;
    for(int i = 0; i < num_bits; ++i)
    {
        if(bits[i] & 1u)
        {
            crc ^= msb_mask;
        }
        const bool pred = (crc & msb_mask) != 0;
        crc = (crc << 1) & all_mask;
        if(pred)
        {
            crc ^= poly;
        }
    }
    return crc & all_mask;
}

void append_crc(std::vector<uint8_t>& bits, int crc_len, uint32_t poly)
{
    const uint32_t crc = compute_crc(bits.data(), static_cast<int>(bits.size()), crc_len, poly);
    for(int i = crc_len - 1; i >= 0; --i)
    {
        bits.push_back(static_cast<uint8_t>((crc >> i) & 1u));
    }
}

__device__ void rm_locate_cb(int tb_pos, int G, int C, int nl, int Qm, int& cb, int& local_idx, int& E)
{
    const int llrs_per_layer_qam = nl * Qm;
    const int q1 = G / llrs_per_layer_qam;
    const int q = q1 / C;
    const int rr = C - (q1 - q * C) - 1;
    const int El = llrs_per_layer_qam * q;
    const int Eh = El + ((q * llrs_per_layer_qam * C < G) ? llrs_per_layer_qam : 0);
    const int small_count = rr + 1;
    const int small_bits = small_count * El;
    if(tb_pos < small_bits)
    {
        cb = tb_pos / El;
        local_idx = tb_pos - cb * El;
        E = El;
    }
    else
    {
        const int tail = tb_pos - small_bits;
        cb = small_count + tail / Eh;
        local_idx = tail - (cb - small_count) * Eh;
        E = Eh;
    }
}

__global__ void tx_rm_kernel(const uint8_t* __restrict__ encoded,
                             int                       encoded_stride,
                             uint8_t* __restrict__     rm_bits,
                             int                       num_tbs,
                             int                       G,
                             int                       C,
                             int                       nl,
                             int                       Qm,
                             int                       K,
                             int                       F,
                             int                       Z,
                             int                       Ncb,
                             int                       k0)
{
    const int total_bits = num_tbs * G;
    for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_bits; idx += blockDim.x * gridDim.x)
    {
        const int tb = idx / G;
        const int tb_pos = idx - tb * G;
        int cb = 0;
        int local_idx = 0;
        int E = 0;
        rm_locate_cb(tb_pos, G, C, nl, Qm, cb, local_idx, E);

        const int EoverQm = E / Qm;
        const int j = local_idx / Qm;
        const int k = local_idx - j * Qm;
        const int inIdx = k * EoverQm + j;
        const int Kd = K - F - 2 * Z;
        const int outIdx = derate_match_fast_calc_modulo(inIdx, Kd, F, k0, Ncb);
        const int code_idx = 2 * Z + outIdx;
        const int global_cb = tb * C + cb;
        rm_bits[idx] = encoded[global_cb * encoded_stride + code_idx] & 1u;
    }
}

__global__ void repeat_modulated_symbols_kernel(const __half2* __restrict__ source_symbols,
                                                __half2* __restrict__       repeated_symbols,
                                                int                         symbols_per_tb,
                                                int                         source_num_tbs,
                                                int                         num_tbs)
{
    const int total_symbols = num_tbs * symbols_per_tb;
    for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_symbols; idx += blockDim.x * gridDim.x)
    {
        const int tb = idx / symbols_per_tb;
        const int symbol = idx - tb * symbols_per_tb;
        const int source_tb = tb % source_num_tbs;
        repeated_symbols[idx] = source_symbols[source_tb * symbols_per_tb + symbol];
    }
}

template <typename T>
__device__ T from_float(float x);

template <>
__device__ __half from_float<__half>(float x)
{
    return __float2half(x);
}

template <>
__device__ float from_float<float>(float x)
{
    return x;
}

template <typename T>
__device__ float to_float(T x);

template <>
__device__ float to_float<__half>(__half x)
{
    return __half2float(x);
}

template <>
__device__ float to_float<float>(float x)
{
    return x;
}

template <typename T>
__device__ void atomic_add_t(T* ptr, T val);

template <>
__device__ void atomic_add_t<float>(float* ptr, float val)
{
    atomicAdd(ptr, val);
}

template <>
__device__ void atomic_add_t<__half>(__half* ptr, __half val)
{
    atomicAdd(ptr, val);
}

template <typename T>
__global__ void init_dec_llr_kernel(T* __restrict__ out,
                                    int             llr_stride,
                                    int             total_cbs,
                                    int             llr_len,
                                    int             K,
                                    int             F)
{
    const int total = total_cbs * llr_len;
    for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x)
    {
        const int cb = idx / llr_len;
        const int n = idx - cb * llr_len;
        const bool is_filler = (n >= K - F) && (n < K);
        out[cb * llr_stride + n] = from_float<T>(is_filler ? INFINITY : 0.0f);
    }
}

template <typename T>
__global__ void rx_derm_kernel(const T* __restrict__ rm_llr,
                               T* __restrict__       dec_llr,
                               int                   llr_stride,
                               int                   num_tbs,
                               int                   G,
                               int                   C,
                               int                   nl,
                               int                   Qm,
                               int                   K,
                               int                   F,
                               int                   Z,
                               int                   Ncb,
                               int                   k0,
                               float                 clamp_value)
{
    const int total_bits = num_tbs * G;
    for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_bits; idx += blockDim.x * gridDim.x)
    {
        const int tb = idx / G;
        const int tb_pos = idx - tb * G;
        int cb = 0;
        int local_idx = 0;
        int E = 0;
        rm_locate_cb(tb_pos, G, C, nl, Qm, cb, local_idx, E);

        const int EoverQm = E / Qm;
        const int j = local_idx / Qm;
        const int k = local_idx - j * Qm;
        const int inIdx = k * EoverQm + j;
        const int Kd = K - F - 2 * Z;
        const int outIdx = derate_match_fast_calc_modulo(inIdx, Kd, F, k0, Ncb);
        const int code_idx = 2 * Z + outIdx;
        const int global_cb = tb * C + cb;
        float v = to_float<T>(rm_llr[idx]);
        v = fminf(fmaxf(v, -clamp_value), clamp_value);
        atomic_add_t(dec_llr + global_cb * llr_stride + code_idx, from_float<T>(v));
    }
}

template <typename T>
void launch_derm_typed(const pusch_ldpc_case&    c,
                       const cuphy::tensor_device& rm_llr,
                       cuphy::tensor_device&       dec_llr,
                       int                         num_tbs,
                       float                       clamp_value,
                       cudaStream_t                strm)
{
    const int total_cbs = num_tbs * c.C;
    const int init_threads = 256;
    const int init_blocks = std::min(div_round_up(total_cbs * c.llr_len, init_threads), 4096);
    init_dec_llr_kernel<T><<<init_blocks, init_threads, 0, strm>>>(
        static_cast<T*>(dec_llr.addr()),
        dec_llr.layout().strides()[1],
        total_cbs,
        c.llr_len,
        c.K,
        c.F);

    const int rm_threads = 256;
    const int rm_blocks = std::min(div_round_up(num_tbs * c.G, rm_threads), 4096);
    rx_derm_kernel<T><<<rm_blocks, rm_threads, 0, strm>>>(
        static_cast<const T*>(rm_llr.addr()),
        static_cast<T*>(dec_llr.addr()),
        dec_llr.layout().strides()[1],
        num_tbs,
        c.G,
        c.C,
        c.num_layers,
        c.Qm,
        c.K,
        c.F,
        c.Z,
        c.Ncb,
        c.k0,
        clamp_value);
}

} // namespace

const char* mod_from_Qm(int Qm)
{
    switch(Qm)
    {
    case 2: return "QPSK";
    case 4: return "QAM16";
    case 6: return "QAM64";
    case 8: return "QAM256";
    default: return "UNKNOWN";
    }
}

cuphyDataType_t llr_type_from_string(const std::string& fptype)
{
    if(fptype == "fp16") { return CUPHY_R_16F; }
    if(fptype == "fp32") { return CUPHY_R_32F; }
    throw std::runtime_error("unsupported --fptype");
}

pusch_ldpc_case derive_pusch_ldpc_case(int num_prb, int num_layers, int mcs, int mcs_table, int rv,
                                       int num_dmrs, int cdm_no_data)
{
    static constexpr uint32_t TBS_table[93] = {
        24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 208, 224,
        240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 408, 432, 456, 480, 504, 528, 552, 576, 608, 640, 672, 704,
        736, 768, 808, 848, 888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 1256, 1288, 1320, 1352, 1416, 1480, 1544,
        1608, 1672, 1736, 1800, 1864, 1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472, 2536, 2600, 2664, 2728, 2792,
        2856, 2976, 3104, 3240, 3368, 3496, 3624, 3752, 3824
    };

    pusch_ldpc_case c;
    c.num_prb = num_prb;
    c.num_layers = num_layers;
    c.mcs = mcs;
    c.mcs_table = mcs_table;
    c.rv = rv;
    c.Qm = q_m_from_mcs(mcs_table, mcs);
    c.target_code_rate_x10240 = target_rate_x10240_from_mcs(mcs_table, mcs);
    c.code_rate = static_cast<float>(c.target_code_rate_x10240) / 10240.0f;
    c.mod = mod_from_Qm(c.Qm);

    // 14-symbol PUSCH, no UCI; DMRS config type 1 (2 comb).
    // Matches genTV_ldpc.m -- G and TBS use different RE counts:
    //  - data_re_per_prb: real data RE/PRB.
    //  - G (coded bits): uses data_re_per_prb, the real count.
    //  - TBS: uses min(156, data_re_per_prb) per PRB -- the 3GPP cap (38.214 5.1.3.2).
    //  - num_dmrs and cdm_no_data each support 1 or 2; defaults 2, 2 -> 144 RE/PRB.
    const int data_re_per_prb = 12 * 14 - 6 * cdm_no_data * num_dmrs;
    c.G = data_re_per_prb * num_prb * c.Qm * num_layers;
    const int Nre_tbs = std::min(156, data_re_per_prb) * num_prb;
    const float Ninfo = static_cast<float>(Nre_tbs) * c.code_rate * c.Qm * num_layers;

    if(Ninfo <= 3824.0f)
    {
        const int n = std::max(3, static_cast<int>(std::floor(std::log2(Ninfo))) - 6);
        const int Ninfo_prime = std::max(24, static_cast<int>(std::pow(2.0, n) * std::floor(Ninfo / std::pow(2.0, n))));
        for(uint32_t tbs : TBS_table)
        {
            if(Ninfo_prime <= static_cast<int>(tbs))
            {
                c.tb_size = static_cast<int>(tbs);
                break;
            }
        }
        c.C = 1;
    }
    else
    {
        const int n = static_cast<int>(std::floor(std::log2(Ninfo - 24.0f))) - 5;
        const int Ninfo_prime = std::max(3840, static_cast<int>(std::pow(2.0, n) * std::round((Ninfo - 24.0f) / std::pow(2.0, n))));
        if(c.code_rate < 0.25f)
        {
            const int C = div_round_up(Ninfo_prime + 24, 3816);
            c.tb_size = 8 * C * div_round_up(Ninfo_prime + 24, 8 * C) - 24;
        }
        else if(Ninfo_prime > 8424)
        {
            const int C = div_round_up(Ninfo_prime + 24, 8424);
            c.tb_size = 8 * C * div_round_up(Ninfo_prime + 24, 8 * C) - 24;
        }
        else
        {
            c.tb_size = 8 * div_round_up(Ninfo_prime + 24, 8) - 24;
        }
    }

    c.bg = ((c.tb_size <= 292) || ((c.tb_size <= 3824) && (c.code_rate <= 0.67f)) || (c.code_rate <= 0.25f)) ? 2 : 1;
    c.K_cb = (c.bg == 1) ? 8448 : 3840;
    c.tb_crc_len = (c.tb_size <= 3824) ? 16 : 24;

    int B = 0;
    int B_prime = 0;
    if(c.tb_size <= c.K_cb)
    {
        B = c.tb_size + c.tb_crc_len;
        c.C = 1;
        c.cb_crc_len = 0;
        B_prime = B;
    }
    else
    {
        c.tb_crc_len = 24;
        B = c.tb_size + 24;
        c.C = div_round_up(B, c.K_cb - 24);
        c.cb_crc_len = 24;
        B_prime = B + c.C * 24;
    }

    c.K_prime = B_prime / c.C;
    if(c.bg == 1)
    {
        c.Kb = 22;
    }
    else if(B > 640)
    {
        c.Kb = 10;
    }
    else if(B > 560)
    {
        c.Kb = 9;
    }
    else if(B > 192)
    {
        c.Kb = 8;
    }
    else
    {
        c.Kb = 6;
    }

    c.Z = cuphy::ldpc::derive_Zc(c.Kb, static_cast<uint32_t>(c.K_prime));
    if(c.Z == 0)
    {
        throw std::runtime_error("unable to select LDPC lifting size");
    }
    c.K = (c.bg == 1) ? c.Z * 22 : c.Z * 10;
    c.F = c.K - c.K_prime;
    c.Ncb = (c.bg == 1) ? c.Z * 66 : c.Z * 50;
    c.Ncb_padded = 8 * div_round_up(c.Ncb + 2 * c.Z, 8);
    c.k0 = ::compute_k0(rv, c.bg, c.Ncb, c.Z);

    const int Kd = c.K - c.F - 2 * c.Z;
    const int Ncb_for_parity = std::min(c.G / c.C + c.k0, c.Ncb);
    const int max_p = (c.bg == 1) ? CUPHY_LDPC_MAX_BG1_PARITY_NODES : CUPHY_LDPC_MAX_BG2_PARITY_NODES;
    c.p = std::max(4, std::min(max_p, div_round_up(Ncb_for_parity - Kd, c.Z)));
    c.llr_len = c.K + c.p * c.Z;

    const int llrs_per_layer_qam = c.num_layers * c.Qm;
    const int q1 = c.G / llrs_per_layer_qam;
    const int q = q1 / c.C;
    const int rr = c.C - (q1 - q * c.C) - 1;
    const int El = llrs_per_layer_qam * q;
    const int Eh = El + ((q * llrs_per_layer_qam * c.C < c.G) ? llrs_per_layer_qam : 0);
    c.E.resize(c.C);
    c.cb_rm_offset.resize(c.C);
    int offset = 0;
    for(int r = 0; r < c.C; ++r)
    {
        c.cb_rm_offset[r] = offset;
        c.E[r] = (r <= rr) ? El : Eh;
        offset += c.E[r];
    }
    c.valid = true;
    return c;
}

uint32_t cb_crc_type(const pusch_ldpc_case& c)
{
    if(c.C > 1) { return CUPHY_LDPC_CRC_24B; }
    if(c.tb_crc_len == 16) { return CUPHY_LDPC_CRC_16; }
    return CUPHY_LDPC_CRC_24A;
}

void build_random_cb_inputs(const pusch_ldpc_case& c,
                            int                   num_tbs,
                            uint64_t              seed,
                            std::vector<uint8_t>& cb_bits)
{
    cb_bits.assign(static_cast<size_t>(num_tbs) * c.C * c.K, 0);
    std::mt19937_64 rng(seed);
    std::bernoulli_distribution bit_dist(0.5);

    for(int tb = 0; tb < num_tbs; ++tb)
    {
        std::vector<uint8_t> tb_bits;
        tb_bits.reserve(c.tb_size + c.tb_crc_len);
        for(int i = 0; i < c.tb_size; ++i)
        {
            tb_bits.push_back(static_cast<uint8_t>(bit_dist(rng)));
        }
        append_crc(tb_bits, c.tb_crc_len, (c.tb_crc_len == 16) ? CRC16_POLY : CRC24A_POLY);

        for(int r = 0; r < c.C; ++r)
        {
            std::vector<uint8_t> cb_payload;
            if(c.C == 1)
            {
                cb_payload = tb_bits;
            }
            else
            {
                const int payload_len = c.K_prime - 24;
                cb_payload.reserve(c.K_prime);
                const int start = r * payload_len;
                for(int i = 0; i < payload_len; ++i)
                {
                    cb_payload.push_back(tb_bits[start + i]);
                }
                append_crc(cb_payload, 24, CRC24B_POLY);
            }

            const int global_cb = tb * c.C + r;
            uint8_t* dst = cb_bits.data() + static_cast<size_t>(global_cb) * c.K;
            std::copy(cb_payload.begin(), cb_payload.end(), dst);
        }
    }
}

void launch_pusch_tx_rate_match(const pusch_ldpc_case&    c,
                                const cuphy::tensor_device& encoded_u8,
                                cuphy::tensor_device&       rm_bits_u8,
                                int                         num_tbs,
                                cudaStream_t                strm)
{
    const int threads = 256;
    const int blocks = std::min(div_round_up(num_tbs * c.G, threads), 4096);
    tx_rm_kernel<<<blocks, threads, 0, strm>>>(
        static_cast<const uint8_t*>(encoded_u8.addr()),
        encoded_u8.layout().strides()[1],
        static_cast<uint8_t*>(rm_bits_u8.addr()),
        num_tbs,
        c.G,
        c.C,
        c.num_layers,
        c.Qm,
        c.K,
        c.F,
        c.Z,
        c.Ncb,
        c.k0);
}

void launch_repeat_modulated_symbols(const cuphy::tensor_device& source_symbols,
                                     cuphy::tensor_device&       repeated_symbols,
                                     int                         symbols_per_tb,
                                     int                         source_num_tbs,
                                     int                         num_tbs,
                                     cudaStream_t                strm)
{
    const int threads = 256;
    const int blocks = std::min(div_round_up(num_tbs * symbols_per_tb, threads), 4096);
    repeat_modulated_symbols_kernel<<<blocks, threads, 0, strm>>>(
        static_cast<const __half2*>(source_symbols.addr()),
        static_cast<__half2*>(repeated_symbols.addr()),
        symbols_per_tb,
        source_num_tbs,
        num_tbs);
}

void launch_pusch_rx_derate_match_fp16(const pusch_ldpc_case&    c,
                                       const cuphy::tensor_device& rm_llr,
                                       cuphy::tensor_device&       dec_llr,
                                       int                         num_tbs,
                                       float                       clamp_value,
                                       cudaStream_t                strm)
{
    launch_derm_typed<__half>(c, rm_llr, dec_llr, num_tbs, clamp_value, strm);
}

void launch_pusch_rx_derate_match_fp32(const pusch_ldpc_case&    c,
                                       const cuphy::tensor_device& rm_llr,
                                       cuphy::tensor_device&       dec_llr,
                                       int                         num_tbs,
                                       float                       clamp_value,
                                       cudaStream_t                strm)
{
    launch_derm_typed<float>(c, rm_llr, dec_llr, num_tbs, clamp_value, strm);
}

error_stats compare_decoded_bits(const pusch_ldpc_case&      c,
                                 int                         num_tbs,
                                 const cuphy::tensor_device& decoded_bits,
                                 const std::vector<uint8_t>& cb_bits,
                                 bool                        reuse_tb)
{
    const int total_cbs = num_tbs * c.C;
    cuphy::typed_tensor<CUPHY_R_8U, cuphy::pinned_alloc> decoded_u8(c.K, total_cbs);
    decoded_u8.convert(decoded_bits);
    cudaStreamSynchronize(0);

    error_stats stats;
    stats.cb_count = total_cbs;
    stats.tb_count = num_tbs;
    const int valid_bits_per_cb = c.K - c.F;
    stats.bit_count = static_cast<uint64_t>(valid_bits_per_cb) * total_cbs;

    for(int tb = 0; tb < num_tbs; ++tb)
    {
        bool tb_error = false;
        for(int r = 0; r < c.C; ++r)
        {
            const int global_cb = tb * c.C + r;
            const int ref_cb = reuse_tb ? r : global_cb;
            bool cb_error = false;
            const uint8_t* ref = cb_bits.data() + static_cast<size_t>(ref_cb) * c.K;
            for(int k = 0; k < valid_bits_per_cb; ++k)
            {
                const uint8_t got = decoded_u8(k, global_cb) & 1u;
                const uint8_t exp = ref[k] & 1u;
                if(got != exp)
                {
                    ++stats.bit_errors;
                    cb_error = true;
                    tb_error = true;
                }
            }
            if(cb_error)
            {
                ++stats.cb_errors;
            }
        }
        if(tb_error)
        {
            ++stats.tb_errors;
        }
    }
    return stats;
}

} // namespace cuphy_ex_ldpc_rm
