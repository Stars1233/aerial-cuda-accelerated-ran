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

#if !defined(LDPC2_CRC_DISPATCH_CUH_INCLUDED_)
#define LDPC2_CRC_DISPATCH_CUH_INCLUDED_



#include "cuphy.h"
#include "cuphy_internal.h"
#include "cuphy_clmad_util.cuh"
#include "ldpc2_et_context.cuh"

#include "G_CRC_16_P_LUT.h"
#include "G_CRC_24_A_P_LUT.h"
#include "G_CRC_24_B_P_LUT.h"
constexpr uint32_t G_CRC_16 = 0x011021;
constexpr uint32_t G_CRC_24_A = 0x01864CFB;
constexpr uint32_t G_CRC_24_B = 0x01800063;

// Barrett-reduction constants for CLMAD-based CRC.
//
// For a CRC polynomial p of degree T (e.g. CRC-24B has T=24, p = G_CRC_24_B):
//   gstar    = p mod x^T               = G_CRC_xxx with the leading x^T bit cleared
//                                        (i.e. G_CRC_xxx & ((1u << T) - 1))
//   qplusX   = floor(x^(2T)   / p)     (degree T,  used for the S=T  postadjust)
//   qplusCRC = floor(x^(32+T) / p)     (degree 32, used for the S=32 preadjust)
//
// opt_reduction<S, T, QPLUS, GSTAR>(c) computes c * x^T mod p when
// QPLUS = floor(x^(S+T)/p).
//
// The gstar values are derived from G_CRC_* by masking off the leading bit.
// The qplus values are outputs of polynomial long-division of x^k by p in
// GF(2)[x] -- they can't be expressed in closed form from G_CRC_* alone, so
// they are precomputed and hardcoded below.

constexpr uint32_t gcrc16astrX   = G_CRC_16   & 0x0000FFFFu; // = 0x1021
constexpr uint32_t gcrc24aastrX  = G_CRC_24_A & 0x00FFFFFFu; // = 0x864CFB
constexpr uint32_t gcrc24bastrX  = G_CRC_24_B & 0x00FFFFFFu; // = 0x800063

constexpr uint32_t qplusX_16     = 0x00011130u;             // floor(x^32 / G_CRC_16)
constexpr uint32_t qplusX_24_A   = 0x01F845FEu;             // floor(x^48 / G_CRC_24_A)
constexpr uint32_t qplusX_24_B   = 0x01FFFF83u;             // floor(x^48 / G_CRC_24_B)

constexpr uint64_t qplusCRC_16   = 0x0000000111303471ull;   // floor(x^48 / G_CRC_16)
constexpr uint64_t qplusCRC_24_A = 0x00000001F845FE24ull;   // floor(x^56 / G_CRC_24_A)
constexpr uint64_t qplusCRC_24_B = 0x00000001FFFF83FFull;   // floor(x^56 / G_CRC_24_B)

namespace ldpc2
{

constexpr uint32_t CRC_FAILURE_WORD = 0xBAD;
constexpr uint64_t CRC_FAILURE_WORD_X2 = (static_cast<uint64_t>(CRC_FAILURE_WORD) << 32) | CRC_FAILURE_WORD;

template <typename T> struct et_crc_traits;

template <> struct et_crc_traits<__nv_fp8_e4m3>
{
    using crc_t = uint32_t;
    using prev_t = uint32_t;
    using context_t = ldpc_et_context_t;
    static constexpr crc_t failure = CRC_FAILURE_WORD;
    static constexpr prev_t prev_init = 0u;
    static constexpr int cw_per_cta = 1;
};

template <> struct et_crc_traits<__nv_fp8_e5m2>
{
    using crc_t = uint32_t;
    using prev_t = uint32_t;
    using context_t = ldpc_et_context_t;
    static constexpr crc_t failure = CRC_FAILURE_WORD;
    static constexpr prev_t prev_init = 0u;
    static constexpr int cw_per_cta = 1;
};

template <> struct et_crc_traits<__half>
{
    using crc_t = uint32_t;
    using prev_t = uint32_t;
    using context_t = ldpc_et_context_t;
    static constexpr crc_t failure = CRC_FAILURE_WORD;
    static constexpr prev_t prev_init = 0u;
    static constexpr int cw_per_cta = 1;
};

template <> struct et_crc_traits<__half2>
{
    using crc_t = uint64_t;
    using prev_t = uint64_t;
    using context_t = ldpc_et_context_x2_t;
    static constexpr crc_t failure = CRC_FAILURE_WORD_X2;
    static constexpr prev_t prev_init = 0ull;
    static constexpr int cw_per_cta = 2;
};

union u64_u32x2 {
    uint64_t u64;
    uint32_t u32[2];
};



__device__ __inline__ uint64_t clmul_lo(uint64_t a, uint64_t b) {
#if CUPHY_CLMAD_AVAILABLE
  uint64_t clm;
  asm volatile ("clmad.lo.u64 %0, %1, %2, 0x0000000000000000;" : "=l"(clm) : "l"(a), "l"(b));

  return clm;
#else
  (void)a;
  (void)b;
  return 0;
#endif
}

////////////////////////////////////////////////////////////////////////
// decode_desc_token()
// Return the transport-block token for a decode CTA index.
template <int CW_PER_CTA>
__device__ __forceinline__
tb_token decode_desc_token(const cuphyLDPCDecodeDesc_t& decodeDesc, int decodeIndex)
{
    int blkIndex = decodeIndex;
    tb_token tok = to_token<CW_PER_CTA>(0, 0, true);

    #pragma unroll
    for(int i = 0; i < CUPHY_LDPC_DECODE_DESC_MAX_TB; ++i)
    {
        if(i < decodeDesc.num_tbs)
        {
            const int num_cw = decodeDesc.llr_input[i].num_codewords;
            const int blocks_claimed = (num_cw + CW_PER_CTA - 1) / CW_PER_CTA;
            if(blkIndex < blocks_claimed)
            {
                const int offset = blkIndex * CW_PER_CTA;
                const bool partial = (CW_PER_CTA == 2) && ((offset + 1) >= num_cw);
                tok = to_token<CW_PER_CTA>(i, offset, partial);
                break;
            }
            blkIndex -= blocks_claimed;
        }
    }

    return tok;
}

__device__ __forceinline__
bool token_has_codeword(tb_token tok, int cta_cw)
{
    return (cta_cw == 0) || !is_partial_from_token(tok);
}

////////////////////////////////////////////////////////////////////////
// decode_desc_iter_output_addr()
// Helper to get the iteration output address for a token/codeword lane.
__device__ __forceinline__
int32_t* decode_desc_iter_output_addr(const cuphyLDPCDecodeDesc_t& decodeDesc, tb_token tok, int cta_cw)
{
    if(!token_has_codeword(tok, cta_cw))
    {
        return nullptr;
    }

    const int tb = tb_from_token(tok);
    const int offset = offset_from_token(tok) + cta_cw;
    if((tb >= decodeDesc.num_tbs) || (offset >= decodeDesc.iter_output[tb].num_codewords) ||
       (decodeDesc.iter_output[tb].addr == nullptr))
    {
        return nullptr;
    }

    return decodeDesc.iter_output[tb].addr + offset;
}

////////////////////////////////////////////////////////////////////////
// decode_desc_crc_addr()
// Helper to get the CRC output address for a token/codeword lane.
__device__ __forceinline__
uint32_t* decode_desc_crc_addr(const cuphyLDPCDecodeDesc_t& decodeDesc, tb_token tok, int cta_cw)
{
    if(!token_has_codeword(tok, cta_cw))
    {
        return nullptr;
    }

    const int tb = tb_from_token(tok);
    const int offset = offset_from_token(tok) + cta_cw;
    if((tb >= decodeDesc.num_tbs) || (offset >= decodeDesc.tb_output[tb].num_codewords) ||
       (decodeDesc.tb_output[tb].crc == nullptr))
    {
        return nullptr;
    }

    return decodeDesc.tb_output[tb].crc + offset;
}

////////////////////////////////////////////////////////////////////////
// decode_desc_crc_type()
// Returns CUPHY_LDPC_CRC_NONE if CRC type is not configured.
__device__ __forceinline__
cuphyLDPCCrcType_t decode_desc_crc_type(const cuphyLDPCDecodeDesc_t& decodeDesc, tb_token tok, int cta_cw)
{
    if(!token_has_codeword(tok, cta_cw))
    {
        return CUPHY_LDPC_CRC_NONE;
    }

    const int tb = tb_from_token(tok);
    const int offset = offset_from_token(tok) + cta_cw;
    if((tb >= decodeDesc.num_tbs) || (offset >= decodeDesc.llr_input[tb].num_codewords) ||
       (decodeDesc.llr_input[tb].crc_type == nullptr))
    {
        return CUPHY_LDPC_CRC_NONE;
    }

    return static_cast<cuphyLDPCCrcType_t>(decodeDesc.llr_input[tb].crc_type[offset]);
}

template <int CW_PER_CTA>
__device__ __forceinline__
cuphyLDPCCrcType_t decode_desc_crc_type(const cuphyLDPCDecodeDesc_t& decodeDesc, int decodeIndex)
{
    return decode_desc_crc_type(decodeDesc, decode_desc_token<CW_PER_CTA>(decodeDesc, decodeIndex), 0);
}

////////////////////////////////////////////////////////////////////////
// get_et_ctx_ptr()
// Returns a pointer to the early-termination context in shared memory.
// size_before_et is the number of bytes of shared memory used for APPs
// and any other regions (e.g. C2V, tb_token) before the ET context.
// The ET context is placed at smem + round_up(size_before_et, alignof(EtContextT)).
// Callers must pass the correct size for their kernel layout (LLR-only vs LLR+C2V+token).
template <typename EtContextT = ldpc_et_context_t>
__device__ __forceinline__
EtContextT* get_et_ctx_ptr(char* smem, uint32_t size_before_et)
{
    const uint32_t offset = round_up_to_next(size_before_et,
                                            static_cast<uint32_t>(alignof(EtContextT)));
    return reinterpret_cast<EtContextT*>(smem + offset);
}

// opt_reduction<S, T, QPLUS, GSTAR>(c) computes c * x^T mod p (low T bits) when
// QPLUS = floor(x^(S+T) / p) and GSTAR = p mod x^T (= p with leading x^T bit
// cleared). Used both to bake the x^T factor into the per-word LUT (during
// early_term_initialize) and to fold the high half of clmul products back
// into the low T bits (during compute_crc_clmad).
template <uint32_t S, uint32_t T, uint64_t QPLUS, uint32_t GSTAR>
__device__ inline uint32_t opt_reduction(uint32_t c)
{
  u64_u32x2 step1;
  step1.u64 = clmul_lo(c, QPLUS);

  u64_u32x2 step2;
  step2.u32[0] = __funnelshift_rc(step1.u32[0], step1.u32[1], S);
  step2.u32[1] = 0;

  u64_u32x2 step3;
  constexpr uint32_t lsb_mask = (1u << T) - 1u;
  step3.u64 = clmul_lo(step2.u64, GSTAR);

  return step3.u32[0] & lsb_mask;
}

////////////////////////////////////////////////////////////////////////
// early_term_initialize()
// Loads the per-codeword CRC LUT into shared memory (et_ctx->two_k_mod_p),
// pre-multiplied by x^T so the per-iteration CLMAD path only needs a single
// reduction. Call before the per-iteration loop; ends with a __syncthreads
// so the smem writes are visible to the iteration loop's CRC checks.
template <int CW_PER_CTA, typename EtContextT>
__device__ __forceinline__
void early_term_initialize(EtContextT* et_ctx, const cuphyLDPCDecodeDesc_t& decodeDesc, int decodeIndex)
{
    if(0 == (CUPHY_LDPC_DECODE_EARLY_TERM & decodeDesc.config.flags))
    {
        return;
    }

    cuphyLDPCCrcType_t crcType = decode_desc_crc_type<CW_PER_CTA>(decodeDesc, decodeIndex);

    // For each CRC variant, load G_CRC_xxx_P_LUT[k] * x^T mod p into smem.
    // The pre-multiply by x^T lets compute_crc_clmad skip its preadjust step
    // (since the LUT carries the x^T factor that the LUT-path's outer
    // partial_crc = a * x^T * b mod p formula requires).
    switch(crcType)
    {
        case CUPHY_LDPC_CRC_16:
            for (int k = threadIdx.x; k < EtContextT::MAX_NUM_WORDS; k += blockDim.x)
            {
                et_ctx->two_k_mod_p[k] =
                    opt_reduction<16, 16, qplusX_16, gcrc16astrX>(G_CRC_16_P_LUT[k]);
            }
            break;
        case CUPHY_LDPC_CRC_24A:
            for (int k = threadIdx.x; k < EtContextT::MAX_NUM_WORDS; k += blockDim.x)
            {
                et_ctx->two_k_mod_p[k] =
                    opt_reduction<24, 24, qplusX_24_A, gcrc24aastrX>(G_CRC_24_A_P_LUT[k]);
            }
            break;
        case CUPHY_LDPC_CRC_24B:
            for (int k = threadIdx.x; k < EtContextT::MAX_NUM_WORDS; k += blockDim.x)
            {
                et_ctx->two_k_mod_p[k] =
                    opt_reduction<24, 24, qplusX_24_B, gcrc24bastrX>(G_CRC_24_B_P_LUT[k]);
            }
            break;
        default:
            return;
    }
    __syncthreads();
}

template <bool ENABLE_ACCESSORY_FEATURES, int CW_PER_CTA, typename EtContextT>
__device__ __forceinline__
void early_term_initialize_if_enabled(
    EtContextT*             et_ctx,
    const cuphyLDPCDecodeDesc_t&   decodeDesc,
    int                            decodeIndex)
{
    if constexpr (ENABLE_ACCESSORY_FEATURES)
    {
        early_term_initialize<CW_PER_CTA>(et_ctx, decodeDesc, decodeIndex);
    }
}

template <cuphyLDPCCrcType_t CRCVariant>
__device__ __forceinline__
uint32_t crc_static_lut_value(int word_idx)
{
  if constexpr (CRCVariant == CUPHY_LDPC_CRC_16)
  {
    return opt_reduction<16, 16, qplusX_16, gcrc16astrX>(G_CRC_16_P_LUT[word_idx]);
  }
  else if constexpr (CRCVariant == CUPHY_LDPC_CRC_24A)
  {
    return opt_reduction<24, 24, qplusX_24_A, gcrc24aastrX>(G_CRC_24_A_P_LUT[word_idx]);
  }
  else
  {
    return opt_reduction<24, 24, qplusX_24_B, gcrc24bastrX>(G_CRC_24_B_P_LUT[word_idx]);
  }
}

template <cuphyLDPCCrcType_t CRCVariant, uint32_t T, uint64_t QPLUSCRC, uint32_t GSTAR>
__device__ __forceinline__
uint32_t crc_word_partial(uint32_t selected_word, uint32_t lut_value)
{
  constexpr uint32_t T_MASK = (1u << T) - 1u;
  u64_u32x2 AxB;
  AxB.u64 = clmul_lo(selected_word, lut_value);
  const uint32_t AxB_Tlsb = AxB.u32[0] & T_MASK;
  const uint32_t AxB_Tmsb = __funnelshift_rc(AxB.u32[0], AxB.u32[1], T);
  return opt_reduction<32, T, QPLUSCRC, GSTAR>(AxB_Tmsb) ^ AxB_Tlsb;
}

template <uint32_t T, uint64_t QPLUSCRC, uint32_t GSTAR>
__device__ __forceinline__
uint32_t reduce_clmad_crc_words_runtime(uint32_t selected_word,
                                         uint32_t lut_value,
                                         bool active_word,
                                         uint32_t* partial_crcs)
{
  constexpr uint32_t T_MASK = (1u << T) - 1u;
  constexpr int WARP_SIZE = 32;
  const int lane_idx = threadIdx.x & (WARP_SIZE - 1);
  const int warp_idx = threadIdx.x / WARP_SIZE;

  uint32_t partial_crc = 0u;
  if(active_word)
  {
    u64_u32x2 AxB;
    AxB.u64 = clmul_lo(selected_word, lut_value);
    const uint32_t AxB_Tlsb = AxB.u32[0] & T_MASK;
    const uint32_t AxB_Tmsb = __funnelshift_rc(AxB.u32[0], AxB.u32[1], T);

    partial_crc = opt_reduction<32, T, QPLUSCRC, GSTAR>(AxB_Tmsb);
    partial_crc ^= AxB_Tlsb;
  }

  const uint32_t active_mask = __ballot_sync(__activemask(), active_word);
  if(active_mask != 0u)
  {
    partial_crc = __reduce_xor_sync(active_mask, partial_crc);
    if(lane_idx == (__ffs(active_mask) - 1))
    {
      partial_crcs[warp_idx] = partial_crc;
    }
  }
  else if(lane_idx == 0)
  {
    partial_crcs[warp_idx] = 0u;
  }
  __syncthreads();

  const int NUM_WARPS = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  const uint32_t FINAL_WARP_REDUCE_MASK =
      (NUM_WARPS >= WARP_SIZE) ? 0xFFFFFFFFu : ((1u << NUM_WARPS) - 1u);
  if(warp_idx == 0)
  {
    uint32_t crc = (lane_idx < NUM_WARPS) ? partial_crcs[lane_idx] : 0u;
    if(lane_idx < NUM_WARPS)
    {
      crc = __reduce_xor_sync(FINAL_WARP_REDUCE_MASK, crc);
      if(lane_idx == 0)
      {
        partial_crcs[0] = crc;
      }
    }
  }
  __syncthreads();

  return partial_crcs[0];
}


template <cuphyLDPCCrcType_t CRCVariant, uint32_t T, uint64_t QPLUSCRC, uint32_t GSTAR>
__device__ __forceinline__
uint2 reduce_clmad_crc_words_runtime_x2(uint32_t selected_word_lo,
                                        uint32_t selected_word_hi,
                                        uint32_t lut_value,
                                        bool active_word,
                                        bool has_upper_codeword,
                                        uint2* partial_crcs)
{
  constexpr int WARP_SIZE = 32;
  const int lane_idx = threadIdx.x & (WARP_SIZE - 1);
  const int warp_idx = threadIdx.x / WARP_SIZE;

  uint2 partial_crc = make_uint2(0u, 0u);
  if(active_word)
  {
    partial_crc.x = crc_word_partial<CRCVariant, T, QPLUSCRC, GSTAR>(selected_word_lo,
                                                                    lut_value);
    if(has_upper_codeword)
    {
      partial_crc.y = crc_word_partial<CRCVariant, T, QPLUSCRC, GSTAR>(selected_word_hi,
                                                                      lut_value);
    }
  }

  const uint32_t active_mask = __ballot_sync(__activemask(), active_word);
  if(active_mask != 0u)
  {
    partial_crc.x = __reduce_xor_sync(active_mask, partial_crc.x);
    partial_crc.y = __reduce_xor_sync(active_mask, partial_crc.y);
    if(lane_idx == (__ffs(active_mask) - 1))
    {
      partial_crcs[warp_idx] = partial_crc;
    }
  }
  else if(lane_idx == 0)
  {
    partial_crcs[warp_idx] = make_uint2(0u, 0u);
  }
  __syncthreads();

  const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  const uint32_t final_warp_reduce_mask =
      (num_warps >= WARP_SIZE) ? 0xFFFFFFFFu : ((1u << num_warps) - 1u);
  if(warp_idx == 0)
  {
    uint2 crc = (lane_idx < num_warps) ? partial_crcs[lane_idx] : make_uint2(0u, 0u);
    if(lane_idx < num_warps)
    {
      crc.x = __reduce_xor_sync(final_warp_reduce_mask, crc.x);
      crc.y = __reduce_xor_sync(final_warp_reduce_mask, crc.y);
      if(lane_idx == 0)
      {
        partial_crcs[0] = crc;
      }
    }
  }
  __syncthreads();

  return partial_crcs[0];
}
template <cuphyLDPCCrcType_t CRCVariant, typename AppT, typename Layout>
__device__ __forceinline__
uint32_t crc_after_bitflip_gate(const AppT*  apps,
                                int           iter,
                                uint32_t&     prev_packed_word,
                                const Layout& layout,
                                bool          force_crc,
                                bool          force_bit_flip)
{
  constexpr uint32_t T = (CRCVariant == CUPHY_LDPC_CRC_16) ? 16u : 24u;
  constexpr uint32_t GSTAR  = (CRCVariant == CUPHY_LDPC_CRC_16)  ? gcrc16astrX  :
                              (CRCVariant == CUPHY_LDPC_CRC_24A) ? gcrc24aastrX :
                                                                   gcrc24bastrX;
  constexpr uint64_t QPLUSCRC = (CRCVariant == CUPHY_LDPC_CRC_16)  ? qplusCRC_16   :
                                (CRCVariant == CUPHY_LDPC_CRC_24A) ? qplusCRC_24_A :
                                                                     qplusCRC_24_B;

  const int total_bits = layout.total_bits();
  if(total_bits < static_cast<int>(T))
  {
    return CRC_FAILURE_WORD;
  }

  const int word_idx = layout.word_index();
  const bool active_word = layout.active_word();
  uint32_t packed_word = 0u;

  if(active_word)
  {
    #pragma unroll
    for(int bit_idx = 0; bit_idx < 32; ++bit_idx)
    {
      const int linear_bit = word_idx * 32 + bit_idx;
      uint8_t bit = 0u;
      if(linear_bit < total_bits)
      {
        bit = llr_hard_decision(apps[linear_bit]);
      }
      packed_word |= static_cast<uint32_t>(bit) << (31 - bit_idx);
    }
  }

  const bool bit_flip = active_word && (iter != 0) && (packed_word != prev_packed_word);
  if(active_word)
  {
    prev_packed_word = packed_word;
  }

  if((iter == 0) && !force_crc)
  {
    if constexpr (Layout::needs_app_read_barrier)
    {
      // Complete APP reads before the next decoder iteration can update them.
      __syncthreads();
    }

    return CRC_FAILURE_WORD;
  }

  if(!force_crc && layout.any_bit_flip(bit_flip || force_bit_flip))
  {
    return CRC_FAILURE_WORD;
  }

  return layout.template reduce_crc<CRCVariant, T, QPLUSCRC, GSTAR>(packed_word,
                                                                    active_word,
                                                                    word_idx);
}

struct ldpc_crc_packed_layout
{
  static constexpr bool needs_app_read_barrier = true;

  ldpc_et_context_t* et_ctx;
  int total_bits_;

  __device__ __forceinline__ int total_bits() const
  {
    return total_bits_;
  }

  __device__ __forceinline__ int total_words() const
  {
    return (total_bits_ + 31) >> 5;
  }

  __device__ __forceinline__ int word_index() const
  {
    return threadIdx.x;
  }

  __device__ __forceinline__ bool active_word() const
  {
    return word_index() < total_words();
  }

  __device__ __forceinline__ bool any_bit_flip(bool bit_flip) const
  {
    return __syncthreads_count(bit_flip) != 0u;
  }

  template <cuphyLDPCCrcType_t CRCVariant, uint32_t T, uint64_t QPLUSCRC, uint32_t GSTAR>
  __device__ __forceinline__ uint32_t reduce_crc(uint32_t packed_word,
                                                  bool     active_word,
                                                  int      word_idx) const
  {
    (void)CRCVariant;
    const uint32_t lut_value = active_word ? et_ctx->two_k_mod_p[word_idx] : 0u;
    return reduce_clmad_crc_words_runtime<T, QPLUSCRC, GSTAR>(packed_word,
                                                              lut_value,
                                                              active_word,
                                                              et_ctx->partial_crcs);
  }
};



template <cuphyLDPCCrcType_t CRCVariant, typename AppT, typename Layout>
__device__ __forceinline__
uint32_t crc_without_bitflip_gate(const AppT* apps,
                                  const Layout& layout)
{
  constexpr uint32_t T = (CRCVariant == CUPHY_LDPC_CRC_16) ? 16u : 24u;
  constexpr uint32_t GSTAR  = (CRCVariant == CUPHY_LDPC_CRC_16)  ? gcrc16astrX  :
                              (CRCVariant == CUPHY_LDPC_CRC_24A) ? gcrc24aastrX :
                                                                   gcrc24bastrX;
  constexpr uint64_t QPLUSCRC = (CRCVariant == CUPHY_LDPC_CRC_16)  ? qplusCRC_16   :
                                (CRCVariant == CUPHY_LDPC_CRC_24A) ? qplusCRC_24_A :
                                                                     qplusCRC_24_B;

  const int total_bits = layout.total_bits();
  if(total_bits < static_cast<int>(T))
  {
    return CRC_FAILURE_WORD;
  }

  const int word_idx = layout.word_index();
  const bool active_word = layout.active_word();
  uint32_t packed_word = 0u;

  if(active_word)
  {
    #pragma unroll
    for(int bit_idx = 0; bit_idx < 32; ++bit_idx)
    {
      const int linear_bit = word_idx * 32 + bit_idx;
      uint8_t bit = 0u;
      if(linear_bit < total_bits)
      {
        bit = llr_hard_decision(apps[linear_bit]);
      }
      packed_word |= static_cast<uint32_t>(bit) << (31 - bit_idx);
    }
  }

  return layout.template reduce_crc<CRCVariant, T, QPLUSCRC, GSTAR>(packed_word,
                                                                    active_word,
                                                                    word_idx);
}

template <cuphyLDPCCrcType_t CRCVariant, int BG, typename AppT>
__device__ __forceinline__ uint32_t compute_crc_clmad_packed(const AppT* apps,
                                                              ldpc_et_context_t* et_ctx,
                                                              int iter,
                                                              uint32_t& prev_packed_word,
                                                              int Z,
                                                              int num_info_nodes,
                                                              bool force_crc,
                                                              bool force_bit_flip)
{
  (void)BG;
  const ldpc_crc_packed_layout layout{et_ctx, num_info_nodes * Z};
  return crc_after_bitflip_gate<CRCVariant>(apps, iter, prev_packed_word, layout, force_crc, force_bit_flip);
}

template <cuphyLDPCCrcType_t CRCVariant, int BG>
__device__ __forceinline__ uint64_t compute_crc_clmad_packed(const __half2* apps,
                                                              ldpc_et_context_x2_t* et_ctx,
                                                              int iter,
                                                              uint64_t& prev_packed_word,
                                                              int Z,
                                                              int num_info_nodes,
                                                              bool force_crc,
                                                              bool force_bit_flip,
                                                              bool has_upper_codeword)
{
  constexpr uint32_t T = (CRCVariant == CUPHY_LDPC_CRC_16) ? 16u : 24u;
  constexpr uint32_t GSTAR  = (CRCVariant == CUPHY_LDPC_CRC_16)  ? gcrc16astrX  :
                              (CRCVariant == CUPHY_LDPC_CRC_24A) ? gcrc24aastrX :
                                                                   gcrc24bastrX;
  constexpr uint64_t QPLUSCRC = (CRCVariant == CUPHY_LDPC_CRC_16)  ? qplusCRC_16   :
                                (CRCVariant == CUPHY_LDPC_CRC_24A) ? qplusCRC_24_A :
                                                                     qplusCRC_24_B;

  constexpr int WARP_SIZE = 32;
  const int total_bits = num_info_nodes * Z;
  const int total_words = (total_bits + 31) >> 5;
  const int lane_idx = threadIdx.x & (WARP_SIZE - 1);
  const int warp_idx = threadIdx.x / WARP_SIZE;
  int word_idx = threadIdx.x;
  bool active_word = false;
  uint32_t selected_word_lo = 0u;
  uint32_t selected_word_hi = 0u;

  const bool warp_aligned_layout = ((Z & (WARP_SIZE - 1)) == 0) && (blockDim.x == Z);
  if(warp_aligned_layout)
  {
    // For warp-aligned Z, transpose the APP loads as in the original CLMAD
    // implementation. Each warp reads one contiguous 32-bit slice from every
    // information node; lane k retains the word for information node k.
    const int num_warps = Z / WARP_SIZE;
    const int reversed_thread_idx =
        (threadIdx.x & ~(WARP_SIZE - 1)) | ((WARP_SIZE - 1) - lane_idx);

    #pragma unroll
    for(int node_idx = 0; node_idx < max_info_nodes<BG>::value; ++node_idx)
    {
      if(node_idx < num_info_nodes)
      {
        const int app_idx = node_idx * Z + reversed_thread_idx;
        const __half2 app = apps[app_idx];
        const uint8_t bit_lo = llr_hard_decision(__low2half(app));
        const uint8_t bit_hi =
            has_upper_codeword ? llr_hard_decision(__high2half(app)) : 0u;
        const uint32_t word_lo = __ballot_sync(0xFFFFFFFFu, bit_lo);
        const uint32_t word_hi = __ballot_sync(0xFFFFFFFFu, bit_hi);

        if(lane_idx == node_idx)
        {
          selected_word_lo = word_lo;
          selected_word_hi = word_hi;
          word_idx = node_idx * num_warps + warp_idx;
        }
      }
    }
    active_word = lane_idx < num_info_nodes;
  }
  else
  {
    // Unlike the aligned transpose, this path can read APPs most recently
    // written by another thread during row processing.
    __syncthreads();

    // A partial final warp cannot form a 32-bit ballot in one operation.
    // Assign consecutive CRC words to each warp and have its available lanes
    // cover each word in chunks. Every load remains contiguous across lanes.
    const uint32_t warp_mask = __activemask();
    const int active_lanes = __popc(warp_mask);
    const int warp_word_base = warp_idx * WARP_SIZE;
    int words_in_warp = total_words - warp_word_base;
    words_in_warp = (words_in_warp > active_lanes) ? active_lanes : words_in_warp;
    words_in_warp = (words_in_warp > 0) ? words_in_warp : 0;

    for(int word_lane = 0; word_lane < words_in_warp; ++word_lane)
    {
      uint32_t word_lo = 0u;
      uint32_t word_hi = 0u;
      for(int bit_base = 0; bit_base < WARP_SIZE; bit_base += active_lanes)
      {
        const int bit_idx = bit_base + lane_idx;
        const int linear_bit = (warp_word_base + word_lane) * WARP_SIZE + bit_idx;
        uint8_t bit_lo = 0u;
        uint8_t bit_hi = 0u;
        if((bit_idx < WARP_SIZE) && (linear_bit < total_bits))
        {
          const __half2 app = apps[linear_bit];
          bit_lo = llr_hard_decision(__low2half(app));
          bit_hi = has_upper_codeword ? llr_hard_decision(__high2half(app)) : 0u;
        }
        const uint32_t vote_lo = __ballot_sync(warp_mask, bit_lo);
        const uint32_t vote_hi = __ballot_sync(warp_mask, bit_hi);
        word_lo |= __brev(vote_lo) >> bit_base;
        word_hi |= __brev(vote_hi) >> bit_base;
      }

      if(lane_idx == word_lane)
      {
        selected_word_lo = word_lo;
        selected_word_hi = word_hi;
      }
    }
    active_word = lane_idx < words_in_warp;
    word_idx = warp_word_base + lane_idx;

    // Finish all cross-thread APP reads before row processing can resume.
    __syncthreads();
  }

  bool bit_flip = false;
  const bool check_bit_flip = !force_crc;
  if(check_bit_flip && active_word)
  {
    u64_u32x2 prev;
    prev.u64 = prev_packed_word;
    bit_flip = (iter != 0) && ((selected_word_lo != prev.u32[0]) ||
                               (has_upper_codeword && (selected_word_hi != prev.u32[1])));
    prev.u32[0] = selected_word_lo;
    prev.u32[1] = has_upper_codeword ? selected_word_hi : 0u;
    prev_packed_word = prev.u64;
  }

  const uint64_t failure = has_upper_codeword ? CRC_FAILURE_WORD_X2 : CRC_FAILURE_WORD;
  if((iter == 0) && !force_crc)
  {
    if(warp_aligned_layout)
    {
      // The unaligned path already synchronizes after its cross-thread reads.
      __syncthreads();
    }

    return failure;
  }
  if(!force_crc && (__syncthreads_count(bit_flip || force_bit_flip) != 0u))
  {
    return failure;
  }

  const uint32_t lut_value = active_word ? et_ctx->two_k_mod_p[word_idx] : 0u;
  const uint2 crc_pair = reduce_clmad_crc_words_runtime_x2<CRCVariant, T, QPLUSCRC, GSTAR>(
      selected_word_lo,
      selected_word_hi,
      lut_value,
      active_word,
      has_upper_codeword,
      et_ctx->partial_crcs);
  const uint32_t crc_lo = crc_pair.x;
  const uint32_t crc_hi = crc_pair.y;

  const uint64_t crc = (static_cast<uint64_t>(crc_hi) << 32) | crc_lo;
  return force_bit_flip ? failure : crc;
}

template <cuphyLDPCCrcType_t CRCVariant, int BG, typename AppT>
__device__ __forceinline__ uint32_t compute_crc_clmad(const AppT* apps,
                                                       ldpc_et_context_t* et_ctx,
                                                       int                iter,
                                                       uint32_t&          prev_packed_word,
                                                       int                Z,
                                                       int                num_info_nodes,
                                                       bool               force_crc,
                                                       bool               force_bit_flip)
{
  return compute_crc_clmad_packed<CRCVariant, BG>(apps,
                                                  et_ctx,
                                                  iter,
                                                  prev_packed_word,
                                                  Z,
                                                  num_info_nodes,
                                                  force_crc,
                                                  force_bit_flip);
}

template <cuphyLDPCCrcType_t CRCVariant, int BG, typename AppT>
__device__ __forceinline__ uint32_t compute_crc_clmad_no_bitflip(const AppT* apps,
                                                                  ldpc_et_context_t* et_ctx,
                                                                  int Z,
                                                                  int num_info_nodes)
{
  (void)BG;
  const ldpc_crc_packed_layout layout{et_ctx, num_info_nodes * Z};
  return crc_without_bitflip_gate<CRCVariant>(apps, layout);
}

template <cuphyLDPCCrcType_t CRCVariant, int BG>
__device__ __forceinline__ uint64_t compute_crc_clmad_no_bitflip(const __half2* apps,
                                                                 ldpc_et_context_x2_t* et_ctx,
                                                                 int Z,
                                                                 int num_info_nodes,
                                                                 bool has_upper_codeword)
{
  uint64_t prev_packed_word = 0u;
  return compute_crc_clmad_packed<CRCVariant, BG>(apps,
                                                  et_ctx,
                                                  1,
                                                  prev_packed_word,
                                                  Z,
                                                  num_info_nodes,
                                                  true,
                                                  false,
                                                  has_upper_codeword);
}

template <cuphyLDPCCrcType_t CRCVariant, int BG>
__device__ __forceinline__ uint64_t compute_crc_clmad(const __half2*     apps,
                                                       ldpc_et_context_x2_t* et_ctx,
                                                       int                iter,
                                                       uint64_t&          prev_packed_word,
                                                       int                Z,
                                                       int                num_info_nodes,
                                                       bool               force_crc,
                                                       bool               force_bit_flip,
                                                       bool               has_upper_codeword)
{
  return compute_crc_clmad_packed<CRCVariant, BG>(apps,
                                                  et_ctx,
                                                  iter,
                                                  prev_packed_word,
                                                  Z,
                                                  num_info_nodes,
                                                  force_crc,
                                                  force_bit_flip,
                                                  has_upper_codeword);
}

////////////////////////////////////////////////////////////////////////
// should_terminate_early_crc<T, BG>()
// Check if early termination should occur based on CRC.
// Dispatches to the appropriate CRC function based on the configured
// CRC type for this codeword. BG is the base graph (1 or 2), known at
// compile time from the calling kernel.
//
// @param decodeDesc The LDPC decode descriptor
// @param cwIndex    The codeword index (typically blockIdx.x)
// @param apps      Pointer to APP values in shared memory
// @param et_ctx    Early-termination context
// @return 0 if CRC passes (terminate early), non-zero if CRC fails (continue)
template <int BG, typename T>
__device__ __forceinline__
typename et_crc_traits<T>::crc_t should_terminate_early_crc(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                            int                          decodeIndex,
                                                            const T*                     apps,
                                                            typename et_crc_traits<T>::context_t* et_ctx,
                                                            int                          iter,
                                                            typename et_crc_traits<T>::prev_t& prev_packed_word)
{
    using traits = et_crc_traits<T>;
#if !CUPHY_CLMAD_AVAILABLE
    return traits::failure;
#else
    tb_token tok = decode_desc_token<traits::cw_per_cta>(decodeDesc, decodeIndex);
    cuphyLDPCCrcType_t crcType = decode_desc_crc_type(decodeDesc, tok, 0);
    const bool final_iteration = (iter >= (decodeDesc.config.max_iterations - 1));
    const bool et_latency_debug = (0 != (decodeDesc.config.flags & CUPHY_LDPC_DECODE_ET_LATENCY_DEBUG));
    const bool force_crc = final_iteration;
    const bool force_bit_flip = et_latency_debug && !final_iteration;

    if constexpr (traits::cw_per_cta == 2)
    {
        cuphyLDPCCrcType_t crcTypeUpper = decode_desc_crc_type(decodeDesc, tok, 1);
        if(token_has_codeword(tok, 1) && (crcTypeUpper != crcType))
        {
            return traits::failure;
        }
    }

    // Not every row scheduler ends an iteration with a CTA barrier. Complete all
    // shared APP updates before the CRC pack reads values written by other warps.
    __syncthreads();

    switch(crcType)
    {
        case CUPHY_LDPC_CRC_16:
            if constexpr (traits::cw_per_cta == 2)
            {
                return compute_crc_clmad<CUPHY_LDPC_CRC_16, BG>(apps, et_ctx, iter, prev_packed_word, decodeDesc.config.Z, decodeDesc.config.Kb, force_crc, force_bit_flip, token_has_codeword(tok, 1));
            }
            else
            {
                return compute_crc_clmad<CUPHY_LDPC_CRC_16, BG>(apps, et_ctx, iter, prev_packed_word, decodeDesc.config.Z, decodeDesc.config.Kb, force_crc, force_bit_flip);
            }

        case CUPHY_LDPC_CRC_24A:
            if constexpr (traits::cw_per_cta == 2)
            {
                return compute_crc_clmad<CUPHY_LDPC_CRC_24A, BG>(apps, et_ctx, iter, prev_packed_word, decodeDesc.config.Z, decodeDesc.config.Kb, force_crc, force_bit_flip, token_has_codeword(tok, 1));
            }
            else
            {
                return compute_crc_clmad<CUPHY_LDPC_CRC_24A, BG>(apps, et_ctx, iter, prev_packed_word, decodeDesc.config.Z, decodeDesc.config.Kb, force_crc, force_bit_flip);
            }

        case CUPHY_LDPC_CRC_24B:
            if constexpr (traits::cw_per_cta == 2)
            {
                return compute_crc_clmad<CUPHY_LDPC_CRC_24B, BG>(apps, et_ctx, iter, prev_packed_word, decodeDesc.config.Z, decodeDesc.config.Kb, force_crc, force_bit_flip, token_has_codeword(tok, 1));
            }
            else
            {
                return compute_crc_clmad<CUPHY_LDPC_CRC_24B, BG>(apps, et_ctx, iter, prev_packed_word, decodeDesc.config.Z, decodeDesc.config.Kb, force_crc, force_bit_flip);
            }

        case CUPHY_LDPC_CRC_NONE:
        default:
            return traits::failure;
    }
#endif
}

template <int BG, typename T, typename SchedT>
__device__ __forceinline__
int32_t run_iterations_with_early_termination(
    SchedT&                                  sched,
    const cuphyLDPCDecodeDesc_t&             decodeDesc,
    int                                      decodeIndex,
    const T*                                 apps,
    typename et_crc_traits<T>::context_t*     et_ctx,
    typename et_crc_traits<T>::crc_t&         crc)
{
    using traits = et_crc_traits<T>;
    auto prev_packed_word = traits::prev_init;
    int32_t iter = 0;
    crc = traits::failure;

    while(iter < decodeDesc.config.max_iterations)
    {
        sched.do_iteration();
        ++iter;

        if(0 != (CUPHY_LDPC_DECODE_EARLY_TERM & decodeDesc.config.flags))
        {
            crc = should_terminate_early_crc<BG>(
                decodeDesc, decodeIndex, apps, et_ctx, iter - 1, prev_packed_word);
            if(crc == 0) break;
        }
    }

    return iter;
}

template <bool ENABLE_ACCESSORY_FEATURES, int BG, typename T, typename SchedT>
__device__ __forceinline__
int32_t run_tb_iterations_with_early_termination(
    SchedT&                                  sched,
    const cuphyLDPCDecodeDesc_t&             decodeDesc,
    int                                      decodeIndex,
    const T*                                 apps,
    char*                                    smem,
    uint32_t                                 size_before_et,
    typename et_crc_traits<T>::crc_t&         crc)
{
    if constexpr (!ENABLE_ACCESSORY_FEATURES)
    {
        int32_t iter = 0;
        while(iter < decodeDesc.config.max_iterations)
        {
            sched.do_iteration();
            ++iter;
        }
        return iter;
    }

    using context_t = typename et_crc_traits<T>::context_t;
    context_t* et_ctx = get_et_ctx_ptr<context_t>(smem, size_before_et);
    early_term_initialize<et_crc_traits<T>::cw_per_cta>(
        et_ctx, decodeDesc, decodeIndex);
    return run_iterations_with_early_termination<BG>(
        sched,
        decodeDesc,
        decodeIndex,
        apps,
        et_ctx,
        crc);
}

template <int BG, typename T>
__device__ __forceinline__
typename et_crc_traits<T>::crc_t should_terminate_early_crc_no_bitflip(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                                       int                          decodeIndex,
                                                                       const T*                     apps,
                                                                       typename et_crc_traits<T>::context_t* et_ctx,
                                                                        int                          iter)
{
  using traits = et_crc_traits<T>;
#if !CUPHY_CLMAD_AVAILABLE
  return traits::failure;
#else
  tb_token tok = decode_desc_token<traits::cw_per_cta>(decodeDesc, decodeIndex);
  const cuphyLDPCCrcType_t crcType = decode_desc_crc_type(decodeDesc, tok, 0);
  if constexpr (traits::cw_per_cta == 2)
  {
    const cuphyLDPCCrcType_t crcTypeUpper = decode_desc_crc_type(decodeDesc, tok, 1);
    if(token_has_codeword(tok, 1) && (crcTypeUpper != crcType))
    {
      return traits::failure;
    }
  }

  // Order the preceding decoder iteration's shared APP writes before packing.
  __syncthreads();

  switch(crcType)
  {
    case CUPHY_LDPC_CRC_16:
      if constexpr (traits::cw_per_cta == 2)
      {
        return compute_crc_clmad_no_bitflip<CUPHY_LDPC_CRC_16, BG>(apps, et_ctx, decodeDesc.config.Z, decodeDesc.config.Kb, token_has_codeword(tok, 1));
      }
      else
      {
        return compute_crc_clmad_no_bitflip<CUPHY_LDPC_CRC_16, BG>(apps, et_ctx, decodeDesc.config.Z, decodeDesc.config.Kb);
      }
    case CUPHY_LDPC_CRC_24A:
      if constexpr (traits::cw_per_cta == 2)
      {
        return compute_crc_clmad_no_bitflip<CUPHY_LDPC_CRC_24A, BG>(apps, et_ctx, decodeDesc.config.Z, decodeDesc.config.Kb, token_has_codeword(tok, 1));
      }
      else
      {
        return compute_crc_clmad_no_bitflip<CUPHY_LDPC_CRC_24A, BG>(apps, et_ctx, decodeDesc.config.Z, decodeDesc.config.Kb);
      }
    case CUPHY_LDPC_CRC_24B:
      if constexpr (traits::cw_per_cta == 2)
      {
        return compute_crc_clmad_no_bitflip<CUPHY_LDPC_CRC_24B, BG>(apps, et_ctx, decodeDesc.config.Z, decodeDesc.config.Kb, token_has_codeword(tok, 1));
      }
      else
      {
        return compute_crc_clmad_no_bitflip<CUPHY_LDPC_CRC_24B, BG>(apps, et_ctx, decodeDesc.config.Z, decodeDesc.config.Kb);
      }
    case CUPHY_LDPC_CRC_NONE:
    default:
      return traits::failure;
  }
#endif
}

// ldpc_dec_iter_output()
// Write the iteration count to host pinned memory.
template <typename T>
__device__ __forceinline__
void ldpc_dec_iter_output(const cuphyLDPCDecodeDesc_t& decodeDesc,
                          int                          decodeIndex,
                          int32_t                      iter)
{
    if(threadIdx.x == 0)
    {
        tb_token tok = decode_desc_token<et_crc_traits<T>::cw_per_cta>(decodeDesc, decodeIndex);
        int32_t* addr0 = decode_desc_iter_output_addr(decodeDesc, tok, 0);
        if(addr0 != nullptr)
        {
            *addr0 = iter;
        }
        if constexpr (et_crc_traits<T>::cw_per_cta == 2)
        {
            int32_t* addr1 = decode_desc_iter_output_addr(decodeDesc, tok, 1);
            if(addr1 != nullptr)
            {
                *addr1 = iter;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_crc_output()
// Write CRC result(s) to GPU memory.
__device__ __forceinline__
void ldpc_dec_crc_output(const cuphyLDPCDecodeDesc_t& decodeDesc,
                         int                          decodeIndex,
                         uint32_t                     crc)
{
    if(threadIdx.x == 0)
    {
        tb_token tok = decode_desc_token<1>(decodeDesc, decodeIndex);
        uint32_t* addr = decode_desc_crc_addr(decodeDesc, tok, 0);
        if(addr != nullptr)
        {
            *addr = crc;
        }
    }
}

__device__ __forceinline__
void ldpc_dec_crc_output(const cuphyLDPCDecodeDesc_t& decodeDesc,
                         int                          decodeIndex,
                         uint64_t                     crc)
{
    if(threadIdx.x == 0)
    {
        tb_token tok = decode_desc_token<2>(decodeDesc, decodeIndex);
        u64_u32x2 packed;
        packed.u64 = crc;
        uint32_t* addr0 = decode_desc_crc_addr(decodeDesc, tok, 0);
        if(addr0 != nullptr)
        {
            *addr0 = packed.u32[0];
        }
        uint32_t* addr1 = decode_desc_crc_addr(decodeDesc, tok, 1);
        if(addr1 != nullptr)
        {
            *addr1 = packed.u32[1];
        }
    }
}

template <bool ENABLE_ACCESSORY_FEATURES, typename T>
__device__ __forceinline__
void write_early_termination_outputs(
    const cuphyLDPCDecodeDesc_t&         decodeDesc,
    int                                  decodeIndex,
    typename et_crc_traits<T>::crc_t      crc,
    int32_t                              iter)
{
    if constexpr (!ENABLE_ACCESSORY_FEATURES)
    {
        return;
    }

    if(0 != (CUPHY_LDPC_DECODE_EARLY_TERM & decodeDesc.config.flags))
    {
        ldpc_dec_crc_output(decodeDesc, decodeIndex, crc);
    }
    if(0 != (CUPHY_LDPC_DECODE_WRITE_ITER_COUNT & decodeDesc.config.flags))
    {
        ldpc_dec_iter_output<T>(decodeDesc, decodeIndex, iter);
    }
}

} // namespace ldpc2

#endif // !defined(LDPC2_CRC_DISPATCH_CUH_INCLUDED_)
