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

#ifndef LDPC_API_HPP
#define LDPC_API_HPP

#include "driver_types.h"
#include "cuphy.h"
#include "cuphy.hpp"

namespace cuphy {

////////////////////////////////////////////////////////////////////////
// ldpc_encode()
template <class TDst, class TSrc>
void ldpc_encode(TDst&        dst,
                 TSrc&        src,
                 int          BG,
                 int          Z,
                 bool         puncture = false,
                 int          maxParityNodes = 0,
                 int          rv = 0,
                 cudaStream_t strm = nullptr)
{
    size_t                      desc_size  = 0;
    size_t                      alloc_size = 0;
    size_t                      workspace_size = 0; // in bytes
    static constexpr int        max_UEs    = PDSCH_MAX_UES_PER_CELL_GROUP;
    cuphyStatus_t               s          = cuphyLDPCEncodeGetDescrInfo(&desc_size,
                                                                         &alloc_size,
                                                                         max_UEs,
                                                                         &workspace_size);
    if(s != CUPHY_STATUS_SUCCESS)
    {
        throw cuphy_fn_exception(s, "cuphyLDPCEncodeGetDescrInfo()");
    }
    unique_device_ptr<uint8_t> d_ldpc_desc = make_unique_device<uint8_t>(desc_size);
    unique_pinned_ptr<uint8_t> h_ldpc_desc = make_unique_pinned<uint8_t>(desc_size);

    unique_device_ptr<uint8_t> d_workspace = make_unique_device<uint8_t>(workspace_size);
    unique_pinned_ptr<uint8_t> h_workspace = make_unique_pinned<uint8_t>(workspace_size);

    cuphyLDPCEncodeLaunchConfig launchConfig{};
    s = cuphySetupLDPCEncode(&launchConfig,       // launch config (output)
                             src.desc().handle(), // source descriptor
                             src.addr(),          // source address
                             dst.desc().handle(), // destination descriptor
                             dst.addr(),          // destination address
                             BG,                  // base graph
                             Z,                   // lifting size
                             puncture,            // puncture output bits
                             maxParityNodes,      // max parity nodes
                             rv,                  // redundancy version
                             0,
                             1,
                             nullptr,
                             nullptr,
                             h_workspace.get(),
                             d_workspace.get(),
                             h_ldpc_desc.get(),   // host descriptor
                             d_ldpc_desc.get(),   // device descriptor
                             1,                   // do async copy during setup
                             strm);
    if(CUPHY_STATUS_SUCCESS != s)
    {
        throw cuphy_fn_exception(s, "cuphySetupLDPCEncode()");
    }
    // Launch LDPC encoder kernel
    const CUDA_KERNEL_NODE_PARAMS& kernelNodeParams = launchConfig.m_kernelNodeParams;
    CU_CHECK_EXCEPTION(cuLaunchKernel(kernelNodeParams.func,
                                      kernelNodeParams.gridDimX,
                                      kernelNodeParams.gridDimY,
                                      kernelNodeParams.gridDimZ,
                                      kernelNodeParams.blockDimX,
                                      kernelNodeParams.blockDimY,
                                      kernelNodeParams.blockDimZ,
                                      kernelNodeParams.sharedMemBytes,
                                      strm,
                                      kernelNodeParams.kernelParams,
                                      kernelNodeParams.extra));
    // Synchronization required, as launchConfig is local
    // to this function.
    CUDA_CHECK_EXCEPTION(cudaStreamSynchronize(strm));
}

////////////////////////////////////////////////////////////////////////
// LDPC_decode_config
// C++ wrapper for a cuphyLDPCDecodeConfigDesc_t, which contains LDPC
// configuration info.
class LDPC_decode_config final : public cuphyLDPCDecodeConfigDesc_t
{
public:
    explicit LDPC_decode_config(cuphyDataType_t llr_type_in         = CUPHY_R_16F, // Type of LLR input data (CUPHY_R_16F or CUPHY_R_32F)
                                int16_t         num_parity_nodes_in = 4,           // Number of parity nodes
                                int16_t         Z_in                = 384,         // Lifting size
                                int16_t         max_iterations_in   = 10,          // Maximum number of iterations
                                float           clamp_value_in      = 32.0f,        // Clamp value
                                int16_t         Kb_in               = 22,          // Number of "information" variable nodes
                                float           norm_in             = 0.8125f,     // Normalization (for normalized min-sum)
                                uint32_t        flags_in            = 0,           // Flags
                                int16_t         BG_in               = 1,           // Base graph (1 or 2)
                                int16_t         algo_in             = 0,           // Algorithm (0 for automatic choice)
                                void*           workspace_in        = nullptr);     // Workspace area
    [[nodiscard]] float get_norm() const
    {
        return (llr_type == CUPHY_R_32F) ? norm.f32 : __low2float(static_cast<__half2>(norm.f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// LDPC_decode_desc
// C++ wrapper for a cuphyLDPCDecodeDesc_t, which contains LDPC
// configuration info and addresses for a finite number of transport
// blocks. All transport blocks references by a cuphyLDPCDecodeDesc_t
// have the same LDPC configuration (BG, lifting size, code rate).
class LDPC_decode_desc final : public cuphyLDPCDecodeDesc_t
{
public:
    //------------------------------------------------------------------
    // Constructor
    explicit LDPC_decode_desc(const cuphyLDPCDecodeConfigDesc_t& config_in, uint8_t max_tbs_per_desc_in) : cuphyLDPCDecodeDesc_t{}, max_tbs_per_desc(max_tbs_per_desc_in)
    {
        config  = config_in;
        num_tbs = 0;
        assert(max_tbs_per_desc > 0);
        assert(CUPHY_LDPC_DECODE_DESC_MAX_TB >= max_tbs_per_desc);
    }
    //------------------------------------------------------------------
    // Constructor
    LDPC_decode_desc(uint8_t max_tbs_per_desc_in) : cuphyLDPCDecodeDesc_t{}, max_tbs_per_desc(max_tbs_per_desc_in)
    {
        num_tbs = 0;
        assert(max_tbs_per_desc > 0);
        assert(CUPHY_LDPC_DECODE_DESC_MAX_TB >= max_tbs_per_desc);
    }
    //------------------------------------------------------------------
    // add_tensor_as_tb()
    // Use the address and layout of the given tensors as if they belong
    // to a single transport block. This overload is used when the caller
    // does not require soft output values.
    void add_tensor_as_tb(const tensor_desc& llrTensorDesc,
                          void*              llrAddr,
                          const tensor_desc& decodeTensorDesc,
                          void*              decodeAddr);
    //------------------------------------------------------------------
    // add_tensor_as_tb()
    // Use the address and layout of the given tensors as if they belong
    // to a single transport block. This overload is used when the caller
    // requires soft output values to be stored in the provided tensor.
    void add_tensor_as_tb(const tensor_desc& llrTensorDesc,
                          void*              llrAddr,
                          const tensor_desc& decodeTensorDesc,
                          void*              decodeAddr,
                          const tensor_desc& softOutputsTensorDesc,
                          void*              softOutputsAddr);
    //------------------------------------------------------------------
    // reset()
    // Sets the number of valid transport blocks to zero
    void reset() { num_tbs = 0; }
    //------------------------------------------------------------------
    // has_config()
    [[nodiscard]] bool has_config(const int16_t BG_, const int Z_, const int parity_nodes, const cuphyLdpcMaxItrAlgoType_t ldpcMaxNumItrAlgo, const uint8_t ldpcMaxNumItrPerUe) const
    {   if(ldpcMaxNumItrAlgo == LDPC_MAX_NUM_ITR_ALGO_TYPE_PER_UE)
            return ((BG_ == config.BG) && (Z_ == config.Z) && (parity_nodes == config.num_parity_nodes) && (ldpcMaxNumItrPerUe == config.max_iterations));
        return ((BG_ == config.BG) && (Z_ == config.Z) && (parity_nodes == config.num_parity_nodes)); 
    }
    //------------------------------------------------------------------
    // is_full()
    [[nodiscard]] bool is_full() const
    {
        return (num_tbs == max_tbs_per_desc);
    }
    private:
    uint8_t max_tbs_per_desc;
};

////////////////////////////////////////////////////////////////////////
// LDPC_decode_desc_set
// C++ class to represent a set of LDPC decode descriptors, with each
// descriptor referencing transport blocks that have the same LDPC
// configuration.
class LDPC_decode_desc_set final
{
public:
    LDPC_decode_desc_set() : count_(0), max_tbs_per_desc_(CUPHY_LDPC_DECODE_DESC_MAX_TB) { }
    explicit LDPC_decode_desc_set(uint8_t max_tbs_per_desc) : count_(0), max_tbs_per_desc_(max_tbs_per_desc) { }
    LDPC_decode_desc& operator[](size_t idx) { return descs_[idx]; }
    [[nodiscard]] unsigned int count() const { return count_; }
    //------------------------------------------------------------------
    // find()
    // Locate a decode descriptor that matches the given configuration
    // and return a reference. If find() was previously called with the
    // same configuration, and that descriptor is not "full", that
    // descriptor is returned. If not, a new descriptor reference will
    // be returned, with the BG, Z, and num_parity_nodes fields of the
    // descriptor configuration set. If all descriptors of the
    // LDPC_decode_desc_set are used, an exception is thrown.
    [[nodiscard]]
    LDPC_decode_desc& find(int16_t BG, int Z, int num_parity, cuphyLdpcMaxItrAlgoType_t ldpcMaxNumItrAlgo, uint8_t ldpcMaxNumItrPerUe);

    void resize(size_t maxSize)
    {
        max_count_ = maxSize;
        const LDPC_decode_desc default_desc(max_tbs_per_desc_);
        descs_.resize(maxSize, default_desc);
    }
    //------------------------------------------------------------------
    // reset()
    // Resets all underlying decode descriptors and sets the valid count
    // to zero.
    void reset()
    {
        for(unsigned int i = 0; i < count_; ++i) { descs_[i].reset(); }
        count_ = 0;
    }
private:
    unsigned int                            count_{};
    unsigned int                            max_count_{};
    uint8_t                                 max_tbs_per_desc_{};
    std::vector<LDPC_decode_desc> descs_;
};

////////////////////////////////////////////////////////////////////////////
// LDPC_decoder_deleter
struct LDPC_decoder_deleter final
{
    typedef cuphyLDPCDecoder_t ptr_t;
    void operator()(const ptr_t p) const
    {
        cuphyDestroyLDPCDecoder(p);
    }
};

////////////////////////////////////////////////////////////////////////////
// unique_LDPC_decoder_ptr
using unique_LDPC_decoder_ptr = std::unique_ptr<cuphyLDPCDecoder, LDPC_decoder_deleter>;

////////////////////////////////////////////////////////////////////////////
// LDPC_decode_tensor_params
// Collection of API parameters for the LDPC decoder, using the tensor
// decoder interface
struct LDPC_decode_tensor_params final
{
    LDPC_decode_tensor_params(const cuphyLDPCDecodeConfigDesc_t& cfg,
                              cuphyTensorDescriptor_t            dst_desc_,
                              void*                              dst_addr_,
                              cuphyTensorDescriptor_t            LLR_desc_,
                              const void*                        LLR_addr_,
                              cuphyTensorDescriptor_t            softOut_desc_ = nullptr,
                              void*                              softOut_addr_ = nullptr) :
      config(cfg),
      dst_desc(dst_desc_),
      dst_addr(dst_addr_),
      LLR_desc(LLR_desc_),
      LLR_addr(LLR_addr_),
      softOutputs_desc(softOut_desc_),
      softOutputs_addr(softOut_addr_)
    {
    }
    cuphyLDPCDecodeConfigDesc_t   config{};
    cuphyTensorDescriptor_t       dst_desc{};
    void*                         dst_addr{};
    cuphyTensorDescriptor_t       LLR_desc{};
    const void*                   LLR_addr{};
    cuphyTensorDescriptor_t       softOutputs_desc{};
    void*                         softOutputs_addr{};
};

class LDPC_decoder final
{
public:
    //----------------------------------------------------------------------
    // LDPC_decoder()
    explicit LDPC_decoder(context& ctx, unsigned int flags = 0);
    //----------------------------------------------------------------------
    // get_workspace_size()
    [[nodiscard]] size_t get_workspace_size(const cuphyLDPCDecodeConfigDesc_t& cfg,
                                            int                                numCodeWords) const;
    //----------------------------------------------------------------------
    // decode()
    void decode(const LDPC_decode_tensor_params& params,
                cudaStream_t                     strm = nullptr) const;
    //----------------------------------------------------------------------
    // decode() (transport block interface)
    void decode(const cuphyLDPCDecodeDesc_t& desc,
                cudaStream_t                 strm = nullptr) const;
    //----------------------------------------------------------------------
    // set_normalization()
    void set_normalization(cuphyLDPCDecodeConfigDesc_t& config) const;
    //----------------------------------------------------------------------
    // get_launch_config
    void get_launch_config(cuphyLDPCDecodeLaunchConfig_t& cfg) const;
    //----------------------------------------------------------------------
    // handle()
    [[nodiscard]] cuphyLDPCDecoder_t handle() const { return dec_.get(); }

    LDPC_decoder(const LDPC_decoder&)            = delete;
    LDPC_decoder& operator=(const LDPC_decoder&) = delete;
private:
    //----------------------------------------------------------------------
    // Data
    unique_LDPC_decoder_ptr dec_;
};

} // namespace cuphy

#endif //LDPC_API_HPP
