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

#if !defined(CUPHY_ERROR_CORRECTION_LDPC_HPP_INCLUDED_)
#define CUPHY_ERROR_CORRECTION_LDPC_HPP_INCLUDED_

#include <vector>
#include "tensor_desc.hpp"
#include "cuphy_context.hpp"

typedef tensor_ref_contig_2D<CUPHY_R_32U> LDPC_output_t;

////////////////////////////////////////////////////////////////////////
// cuphyLDPCDecoder
// Empty base class for internal context class, used by forward
// declaration in public-facing cuphy.h.
struct cuphyLDPCDecoder
{
};

////////////////////////////////////////////////////////////////////////
// ldpc
namespace ldpc
{

class decoder;
    
////////////////////////////////////////////////////////////////////////
// ldpc::decode_algo
// Base class for different LDPC decoder implementations
class decode_algo
{
public:
    //------------------------------------------------------------------
    // destructor
    virtual ~decode_algo() = default;
    //------------------------------------------------------------------
    // decode()
    // Decode the input codewords ("legacy" tensor interface)
    [[nodiscard]]
    virtual cuphyStatus_t decode(decoder&                           dec,
                                 LDPC_output_t&                     tDst,
                                 const_tensor_pair&                 tLLR,
                                 const cuphy_optional<tensor_pair>& optSoftOutputs,
                                 const cuphyLDPCDecodeConfigDesc_t& config,
                                 cudaStream_t                       strm) = 0;
    //------------------------------------------------------------------
    // decode_tb()
    // Decode the input codewords (transport block interface)
    [[nodiscard]]
    virtual cuphyStatus_t decode_tb(decoder&                     dec,
                                    const cuphyLDPCDecodeDesc_t& decodeDesc,
                                    cudaStream_t                 strm) = 0;
    //------------------------------------------------------------------
    // init()
    // One-time (per cuPHY context) initialization for the algorithm
    [[nodiscard]]
    virtual cuphyStatus_t init(decoder& desc)  { return CUPHY_STATUS_SUCCESS; }
    //------------------------------------------------------------------
    // get_workspace_size()
    // Return the workspace size required for a given configuration
    [[nodiscard]]
    virtual std::pair<bool, size_t> get_workspace_size(const decoder&                     dec,
                                                       const cuphyLDPCDecodeConfigDesc_t& config,
                                                       int                                num_cw) = 0;
    //------------------------------------------------------------------
    // can_decode_config()
    [[nodiscard]]
    virtual bool can_decode_config(const decoder&                     dec,
                                   const cuphyLDPCDecodeConfigDesc_t& config)
    {
        return true;
    }
    //------------------------------------------------------------------
    // supported_config_desc()
    // Requirements that are NOT fields of cuphyLDPCDecodeConfigDesc_t, and
    // so cannot be discovered by probing can_decode_config(): which decode
    // interface the algorithm implements, whether it needs punctured input,
    // and so on. Used only to explain a refusal.
    //
    // Do NOT restate parity-node counts, lifting sizes or anything else the
    // configuration carries -- those are derived from can_decode_config()
    // itself when a refusal is reported, which keeps them correct for
    // algorithms that do not exist yet and stops them going stale here.
    //
    // Returns nullptr by default; overriding is optional.
    [[nodiscard]]
    virtual const char* supported_config_desc() const { return nullptr; }
    //------------------------------------------------------------------
    // requirements()
    // The machine-readable form of supported_config_desc(): preconditions a
    // CALLER must satisfy that the configuration descriptor cannot express,
    // so can_decode_config() has no way to check them. Currently punctured
    // input (cuphyLDPCDecodeConfigDesc_t carries no puncturing indicator) and
    // transport-block-only support.
    //
    // Declared here so a caller can ask instead of keeping its own table of
    // which algorithms need what -- such a table is a second source of truth
    // and goes stale silently.
    //
    // Returns 0 by default; overriding is optional.
    [[nodiscard]]
    virtual uint32_t requirements() const { return 0; }
    //------------------------------------------------------------------
    // get_launch_config()
    [[nodiscard]]
    virtual cuphyStatus_t get_launch_config(const decoder&                 dec,
                                            cuphyLDPCDecodeLaunchConfig_t& launchConfig) = 0;
};

////////////////////////////////////////////////////////////////////////
// decoder
class decoder final : public cuphyLDPCDecoder
{
public:
    //------------------------------------------------------------------
    // decoder()
    explicit decoder(const cuphy_i::context& ctx);
    //------------------------------------------------------------------
    // decode()
    // LDPC decode using the tensor descriptor interface
    [[nodiscard]]
    cuphyStatus_t decode(const tensor_pair&                 tDst,
                         const_tensor_pair&                 tLLR,
                         const cuphy_optional<tensor_pair>& optSoftOutputs,
                         const cuphyLDPCDecodeConfigDesc_t& config,
                         cudaStream_t                       strm);
    //------------------------------------------------------------------
    // decode_tb()
    // LDPC decode using the transport block interface
    [[nodiscard]]
    cuphyStatus_t decode_tb(const cuphyLDPCDecodeDesc_t&  decodeDesc,
                            cudaStream_t                  strm);
    //------------------------------------------------------------------
    // workspace_size()
    [[nodiscard]]
    std::pair<bool, size_t> workspace_size(const cuphyLDPCDecodeConfigDesc_t& config,
                                           int                                numCodeWords) const;
    //------------------------------------------------------------------
    // choose_algo()
    [[nodiscard]]
    int choose_algo(const cuphyLDPCDecodeConfigDesc_t& config) const;
    //------------------------------------------------------------------
    // device index
    [[nodiscard]]
    int index() const { return deviceIndex_; }
    //------------------------------------------------------------------
    // compute capability
    [[nodiscard]]
    uint64_t compute_cap() const { return cc_; }
    //------------------------------------------------------------------
    // algo_requirements()
    // Caller-side preconditions declared by one algorithm. Returns 0 for an
    // index that is out of range or not registered on this device, which is
    // the same answer as "no requirements": a caller that cannot reach the
    // algorithm has nothing to satisfy.
    [[nodiscard]] uint32_t algo_requirements(int algo) const
    {
        if((algo < 0) || (static_cast<size_t>(algo) >= algos_.size()) || !algos_[algo])
        {
            return 0;
        }
        return algos_[algo]->requirements();
    }
    //------------------------------------------------------------------
    // maximum shared mem per block (optin)
    [[nodiscard]]
    int max_shmem_per_block_optin() const { return sharedMemPerBlockOptin_; }
    //------------------------------------------------------------------
    // SM count
    [[nodiscard]]
    int sm_count() const { return multiProcessorCount_; }
    //------------------------------------------------------------------
    // get_total_num_codewords()
    [[nodiscard]]
    static int get_total_num_codewords(const cuphyLDPCDecodeDesc_t& decodeDesc)
    {
        int num = 0;
        for(int i = 0; i < decodeDesc.num_tbs; ++i)
        {
            num += decodeDesc.llr_input[i].num_codewords;
        }
        return num;
    }
    //------------------------------------------------------------------
    // get_total_num_codeword_pairs()
    [[nodiscard]]
    static int get_total_num_codeword_pairs(const cuphyLDPCDecodeDesc_t& decodeDesc)
    {
        // Determine the number of codeword pairs. Note that if there
        // are multiple transport blocks, and if any of the transport
        // blocks have an ODD number of codewords, we may have one or
        // more non-full "interior" pairs.
        int num = 0;
        for(int i = 0; i < decodeDesc.num_tbs; ++i)
        {
            num += (decodeDesc.llr_input[i].num_codewords + 1) / 2;
        }
        return num;
    }
    //------------------------------------------------------------------
    // set_normalization()
    [[nodiscard]]
    static cuphyStatus_t set_normalization(cuphyLDPCDecodeConfigDesc_t& config);
    //------------------------------------------------------------------
    // get_launch_config()
    [[nodiscard]]
    cuphyStatus_t get_launch_config(cuphyLDPCDecodeLaunchConfig_t& launchConfig) const;
    //------------------------------------------------------------------
    static int get_num_variable_nodes(int BG, int num_parity)
    {
        return num_parity + ((1 == BG) ? 22 : 10);
    }
private:
    [[nodiscard]] int choose_algo_sm70(const cuphyLDPCDecodeConfigDesc_t& config) const;
    [[nodiscard]] int choose_algo_sm75(const cuphyLDPCDecodeConfigDesc_t& config) const;
    [[nodiscard]] static int choose_algo_sm80(const cuphyLDPCDecodeConfigDesc_t& config);
    [[nodiscard]] int choose_algo_sm86(const cuphyLDPCDecodeConfigDesc_t& config) const;
    [[nodiscard]] int choose_algo_sm89(const cuphyLDPCDecodeConfigDesc_t& config) const;
    [[nodiscard]] int choose_algo_sm90(const cuphyLDPCDecodeConfigDesc_t& config) const;
    [[nodiscard]] int choose_algo_sm100(const cuphyLDPCDecodeConfigDesc_t& config) const;
    [[nodiscard]] int choose_algo_sm120(const cuphyLDPCDecodeConfigDesc_t& config) const;

    typedef std::unique_ptr<decode_algo> decode_algo_ptr_t;
    //------------------------------------------------------------------
    // Data
    // Note: copying required device data from cuPHY context to avoid
    //       issues if cuphy context is destroyed before decoder object.
    int      deviceIndex_{};            // index of device associated with context
    uint64_t cc_{};                     // compute capability (major << 32) | minor
    int      sharedMemPerBlockOptin_{}; // maximum shared memory per block usable by option
    int      multiProcessorCount_{};    // number of multiprocessors on device

    // Pointers to different LDPC decoder implementations
    std::vector<decode_algo_ptr_t> algos_;
};

} // namespace ldpc

// ldpcEncodeDescr is passed as a __grid_constant__ and is currently 32 bytes.
// If adding fields, note any size increases. Some of the fields below could
// have their widths shortened if necessary.
struct ldpcEncodeDescr final
{
    // Swapped old overprovisioned arrays of PDSCH_MAX_UES_PER_CELL_GROUP with pointers to buffers. Note that
    // these would need to be allocated by the caller; OK for PDSCH not as user-friendly for component level API.
    LDPC_output_t* input{};
    LDPC_output_t* output{};
    int num_TBs{}; // Number of input or output elements that are valid for each buffer. The size of the buffers themselves can be different.
    uint16_t BG{};
    uint16_t Kb{};
    uint16_t Z{};
    uint16_t num_rows{};
    char H_type{};
    bool puncture{};
};
typedef ldpcEncodeDescr ldpcEncodeDescr_t;
typedef ldpcEncodeDescr ldpcEncodeDescr_t_array[PDSCH_MAX_HET_LDPC_CONFIGS_SUPPORTED];

#endif // !defined(CUPHY_ERROR_CORRECTION_LDPC_HPP_INCLUDED_)
