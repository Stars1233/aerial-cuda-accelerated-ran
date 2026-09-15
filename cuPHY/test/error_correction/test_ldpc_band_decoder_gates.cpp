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

// Preconditions of the BG1 band decoders (103, 201-204).
//
// These kernels emit hard decisions through ldpc_dec_output_x2_all_warps<22>.
// Because they are all blockDim == Z with one CTA per SM, that writer's
// words-per-warp template argument IS Kb -- it writes 22 * Z bits regardless
// of what the descriptor says. A BG1 descriptor carrying a smaller Kb would
// therefore overrun an output buffer the caller sized from it, so Kb == 22 is
// a precondition of these kernels rather than a preference, and each one
// rejects anything else in can_decode_config().
//
// 3GPP gives BG1 Kb = 22 unconditionally and every in-tree caller derives it,
// so this is a guard on a value no current caller can get wrong. It is here
// because the consequence if one ever does is an out-of-bounds write, and
// because the gate was missing from three of the five kernels.

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "cuphy.h"

namespace
{

// Algorithm ids, and a (p, Z) each one accepts.
struct band_decoder
{
    int         algo;
    int         num_parity_nodes;
    int         Z;
    bool        writes_soft_outputs;
    // How many of cuphyLDPCDecodeLaunchConfig_t::kernel_args the kernel
    // actually declares. These five span all three widths the struct allows,
    // and the width is what the CUDA driver reads when the graph node is
    // updated, so it is a property worth stating rather than assuming:
    //   103           decode descriptor only (base graph is in immediates)
    //   201/202/203   decode descriptor + base graph descriptor
    //   204           the above + %smid-sliced extension-column scratch
    int         num_kernel_args;
    // A second lifting size in the kernel's zone, or 0 if it serves only one.
    // 103 is a Z=384 point kernel; the rest cover Z in {256..384}.
    int         z_alt;
    // Whether a SEPARATE kernel is compiled for Z == the zone maximum, read
    // from each get_launch_config() body: 202 and 203 select on
    // (CFG_Z_MAX == Z) and have two variants; 201 and 204 compile one
    // Z-generic kernel and select on nothing.
    bool        has_z_pinned_variant;
    const char* name;
};

// 103 and 201 have no soft-LLR writer AND repurpose llr_output[]'s scalar
// fields as a pair-CTA index; 202/203/204 implement soft outputs normally.
constexpr band_decoder kBandDecoders[] = {
    {103,  4, 384, false, 1,   0, false, "bg1_z384_p4_x2"},
    {201,  8, 384, false, 2, 256, false, "bg1_z256up_bp_allreg_x2"},
    {202, 18, 384, true,  2, 256, true,  "bg1_z256up_cms_shtail_x2"},
    {203, 28, 384, true,  2, 256, true,  "bg1_z256up_bp_shwin_x2"},
    {204, 40, 384, true,  3, 256, false, "bg1_z256up_bp_gmext_x2"},
};

class LdpcBandDecoderGates : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphyCreateContext(&ctx_, 0));
        ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphyCreateLDPCDecoder(ctx_, &dec_, 0));
    }

    void TearDown() override
    {
        if(dec_) cuphyDestroyLDPCDecoder(dec_);
        if(ctx_) cuphyDestroyContext(ctx_);
    }

    // Ask for a launch descriptor from one specific band decoder. A refusal
    // surfaces as CUPHY_STATUS_UNSUPPORTED_CONFIG via can_decode_config().
    cuphyStatus_t launch_status(const band_decoder& d, int Kb, uint32_t flags = 0) const
    {
        cuphyLDPCDecodeLaunchConfig_t launchConfig{};
        cuphyLDPCDecodeConfigDesc_t&  config = launchConfig.decode_desc.config;
        config.llr_type         = CUPHY_R_16F;
        config.num_parity_nodes = static_cast<int16_t>(d.num_parity_nodes);
        config.Z                = static_cast<int16_t>(d.Z);
        config.max_iterations   = 10;
        config.Kb               = static_cast<int16_t>(Kb);
        config.BG               = 1;
        config.algo             = static_cast<int16_t>(d.algo);
        config.flags            = flags;
        launchConfig.decode_desc.num_tbs = 0;
        return cuphyErrorCorrectionLDPCDecodeGetLaunchDescriptor(dec_, &launchConfig);
    }

    // A launch config for one band decoder carrying a multi-TB workload, so
    // the codeword-pair index actually has something to index. cw_per_tb may
    // hold odd counts: a transport block with an odd number of codewords
    // produces a non-full "interior" pair, which is the partial-token case.
    cuphyLDPCDecodeLaunchConfig_t make_launch_config(const band_decoder&     d,
                                                     const std::vector<int>& cw_per_tb,
                                                     int                     Z = 0) const
    {
        cuphyLDPCDecodeLaunchConfig_t launchConfig{};
        cuphyLDPCDecodeConfigDesc_t&  config = launchConfig.decode_desc.config;
        config.llr_type         = CUPHY_R_16F;
        config.num_parity_nodes = static_cast<int16_t>(d.num_parity_nodes);
        config.Z                = static_cast<int16_t>(Z ? Z : d.Z);
        config.max_iterations   = 10;
        config.Kb               = 22;
        config.BG               = 1;
        config.algo             = static_cast<int16_t>(d.algo);
        // llr_input[] is a fixed-size table in the descriptor. Fail the test
        // rather than scribble past it if a workload above ever outgrows it.
        if(cw_per_tb.size() > static_cast<size_t>(CUPHY_LDPC_DECODE_DESC_MAX_TB))
        {
            ADD_FAILURE() << "workload has " << cw_per_tb.size() << " transport blocks, "
                          << "but the descriptor holds at most "
                          << CUPHY_LDPC_DECODE_DESC_MAX_TB;
            return launchConfig;
        }
        launchConfig.decode_desc.num_tbs = static_cast<int32_t>(cw_per_tb.size());
        for(size_t i = 0; i < cw_per_tb.size(); ++i)
        {
            launchConfig.decode_desc.llr_input[i].num_codewords = cw_per_tb[i];
        }
        return launchConfig;
    }

    cuphyContext_t     ctx_ = nullptr;
    cuphyLDPCDecoder_t dec_ = nullptr;
};

TEST_F(LdpcBandDecoderGates, AcceptTheirOwnBandAtKb22)
{
    for(const auto& d : kBandDecoders)
    {
        SCOPED_TRACE(d.name);
        EXPECT_EQ(CUPHY_STATUS_SUCCESS, launch_status(d, 22));
    }
}

// The gate that matters: <22> would write 22 * Z bits into a buffer the
// caller sized for Kb * Z.
TEST_F(LdpcBandDecoderGates, RejectBG1WithKbOtherThan22)
{
    for(const auto& d : kBandDecoders)
    {
        SCOPED_TRACE(d.name);
        EXPECT_NE(CUPHY_STATUS_SUCCESS, launch_status(d, 10));
        EXPECT_NE(CUPHY_STATUS_SUCCESS, launch_status(d, 6));
        EXPECT_NE(CUPHY_STATUS_SUCCESS, launch_status(d, 0));
    }
}

// A hard-output-only decoder must refuse a soft-output request rather than
// silently dropping it. For 103/201 it would also corrupt the descriptor:
// prepare_ldpc_tb_pair_index() overwrites llr_output[].stride_elements and
// .num_codewords, which are exactly the fields the soft-output writer
// addresses through.
TEST_F(LdpcBandDecoderGates, RefuseSoftOutputsUnlessImplemented)
{
    for(const auto& d : kBandDecoders)
    {
        SCOPED_TRACE(d.name);
        const cuphyStatus_t s =
            launch_status(d, 22, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS);
        if(d.writes_soft_outputs)
        {
            EXPECT_EQ(CUPHY_STATUS_SUCCESS, s);
        }
        else
        {
            EXPECT_NE(CUPHY_STATUS_SUCCESS, s);
        }
    }
}

// get_launch_config() writes into the CALLER's descriptor, where decode_tb()
// works on a private copy. For 103 and 201 that write is
// prepare_ldpc_tb_pair_index(), which rewrites llr_output[]'s two scalar
// fields in all 32 slots into a codeword-pair index.
//
// That is safe only if the write is a pure function of the input fields, so
// that asking twice cannot drift. It is, by inspection --- the function reads
// only num_tbs and llr_input[].num_codewords and writes only llr_output[] ---
// but "by inspection" is what this whole descriptor hazard already survived
// once, so it is pinned here instead.
//
// The production caller (PuschRx::setupCmnPhase2()) additionally re-assigns
// decode_desc from its own decode_desc_set entry before every call, so even a
// non-idempotent write could not accumulate there. This test is the guarantee
// that does not depend on that caller staying written the way it is.
TEST_F(LdpcBandDecoderGates, LaunchConfigIsIdempotent)
{
    // Mixed even and odd codeword counts: the odd ones are what produce
    // partial codeword pairs.
    const std::vector<int> kWorkload = {8, 1, 7, 2, 13};

    for(const auto& d : kBandDecoders)
    {
        SCOPED_TRACE(d.name);
        cuphyLDPCDecodeLaunchConfig_t lc = make_launch_config(d, kWorkload);

        ASSERT_EQ(CUPHY_STATUS_SUCCESS,
                  cuphyErrorCorrectionLDPCDecodeGetLaunchDescriptor(dec_, &lc));
        const cuphyLDPCDecodeDesc_t after_first = lc.decode_desc;

        ASSERT_EQ(CUPHY_STATUS_SUCCESS,
                  cuphyErrorCorrectionLDPCDecodeGetLaunchDescriptor(dec_, &lc));
        EXPECT_EQ(0, std::memcmp(&after_first, &lc.decode_desc, sizeof(after_first)))
            << "get_launch_config() is not idempotent on the caller's descriptor";

        // The grid must be the codeword-pair count, and it must be derived
        // from llr_input[] --- not from the llr_output[] fields the pair index
        // just overwrote. Those two agree only if nothing reads back its own
        // scratch: 8+1+7+2+13 codewords is 4+1+4+1+7 = 17 pairs.
        EXPECT_EQ(17u, lc.kernel_node_params_driver.gridDimX);
        EXPECT_EQ(static_cast<unsigned>(d.Z), lc.kernel_node_params_driver.blockDimX);
    }
}

// Same call on a single transport block holding a single codeword: the
// one-CTA, one-lone-codeword corner, where the pair index degenerates to a
// single partial pair.
TEST_F(LdpcBandDecoderGates, LaunchConfigForOneLoneCodeword)
{
    for(const auto& d : kBandDecoders)
    {
        SCOPED_TRACE(d.name);
        cuphyLDPCDecodeLaunchConfig_t lc = make_launch_config(d, {1});
        ASSERT_EQ(CUPHY_STATUS_SUCCESS,
                  cuphyErrorCorrectionLDPCDecodeGetLaunchDescriptor(dec_, &lc));
        EXPECT_EQ(1u, lc.kernel_node_params_driver.gridDimX);
    }
}

// Every band decoder must leave a usable kernel and a shared memory request
// the device can actually satisfy. A launch descriptor that names no function
// fails only at launch, which until '--graph' existed was nowhere in-tree.
TEST_F(LdpcBandDecoderGates, LaunchConfigNamesAKernelAndFitsSharedMemory)
{
    int optin_shmem = 0;
    ASSERT_EQ(cudaSuccess,
              cudaDeviceGetAttribute(&optin_shmem,
                                     cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                     0));
    for(const auto& d : kBandDecoders)
    {
        SCOPED_TRACE(d.name);
        cuphyLDPCDecodeLaunchConfig_t lc = make_launch_config(d, {8});
        ASSERT_EQ(CUPHY_STATUS_SUCCESS,
                  cuphyErrorCorrectionLDPCDecodeGetLaunchDescriptor(dec_, &lc));
        EXPECT_NE(nullptr, lc.kernel_node_params_driver.func);
        EXPECT_EQ(lc.kernel_args, lc.kernel_node_params_driver.kernelParams);
        // Slot 0 must point at the launch config's OWN descriptor copy, not at
        // whatever the caller passed: that copy is what outlives the call and
        // what cuGraphExecKernelNodeSetParams reads through.
        EXPECT_EQ(&lc.decode_desc, lc.kernel_args[0]);
        // Every slot the kernel declares must be set, and no more is required.
        for(int i = 1; i < d.num_kernel_args; ++i)
        {
            EXPECT_NE(nullptr, lc.kernel_args[i]) << "kernel_args[" << i << "]";
        }
        EXPECT_GT(lc.kernel_node_params_driver.sharedMemBytes, 0u);
        EXPECT_LE(lc.kernel_node_params_driver.sharedMemBytes,
                  static_cast<unsigned>(optin_shmem));
    }
}

// Every accepted lifting must resolve a base graph descriptor.
//
// can_decode_config() gates Z with (256 <= Z <= 384 && Z % 32 == 0) rather
// than by looking the lifting up in get_adj_BG_desc()'s table, and
// get_launch_config() then publishes that table entry as kernel_args[1]
// WITHOUT a null check (decode_tb() does check). The two agree only because
// the multiples of 32 in [256, 384] are exactly the 38.212 liftings in that
// range -- {256, 288, 320, 352, 384} -- so the arithmetic test and the table
// happen to have the same domain. That is a coincidence of where the zone
// boundaries fall, not something either side states, and widening the zone
// downward would break it silently: 224 and 240 are legal liftings that are
// NOT multiples of 32, and 208..255 contains multiples of 32 that are not
// legal liftings. Pin the equivalence over the whole neighbourhood.
TEST_F(LdpcBandDecoderGates, EveryAcceptedLiftingHasABaseGraphDescriptor)
{
    int num_accepted = 0;
    for(const auto& d : kBandDecoders)
    {
        SCOPED_TRACE(d.name);
        for(int Z = 192; Z <= 448; ++Z)
        {
            cuphyLDPCDecodeLaunchConfig_t lc = make_launch_config(d, {8}, Z);
            if(CUPHY_STATUS_SUCCESS !=
               cuphyErrorCorrectionLDPCDecodeGetLaunchDescriptor(dec_, &lc))
            {
                continue;  // refused: nothing to check
            }
            ++num_accepted;
            // 103 carries its base graph in instruction immediates and takes
            // the descriptor alone, so it has no kernel_args[1] to resolve.
            if(d.num_kernel_args >= 2)
            {
                EXPECT_NE(nullptr, lc.kernel_args[1])
                    << "Z=" << Z << " was accepted but resolved a NULL base graph "
                       "descriptor, which the kernel dereferences";
            }
            EXPECT_EQ(0, Z % 32) << "Z=" << Z << " accepted but not a multiple of 32";
        }
    }
    // 201/202/203/204 each take the five liftings in {256..384}; 103 is a
    // Z=384 point kernel. A drop here means a lifting silently stopped being
    // served, which no other test in this file would notice.
    EXPECT_EQ(4 * 5 + 1, num_accepted);
}

// The Z-pinned variant selection, which is the highest-risk line in
// get_launch_config() and the one a null check cannot reach.
//
// 202 and 203 compile TWO kernels each -- one with the lifting size baked in
// at the zone maximum, one taking Z at runtime -- and pick between them on
// (CFG_Z_MAX == Z). 201 and 204 compile one Z-generic kernel and pick nothing.
// Handing back the Z-pinned kernel for a Z it was not compiled for would
// satisfy every structural assertion above (the function pointer is non-null,
// the shared memory request is in range, the arguments are all set) and decode
// garbage. Only comparing the selection across two lifting sizes catches it,
// and doing it that way needs no GPU work and no decoded output.
//
// The base graph descriptor is checked the same way: it is a per-Z table, so
// two lifting sizes must not resolve to the same one.
TEST_F(LdpcBandDecoderGates, ZPinnedVariantSelection)
{
    for(const auto& d : kBandDecoders)
    {
        if(0 == d.z_alt)
        {
            continue;  // serves a single lifting size; nothing to select between
        }
        SCOPED_TRACE(d.name);

        cuphyLDPCDecodeLaunchConfig_t hi = make_launch_config(d, {8}, d.Z);
        cuphyLDPCDecodeLaunchConfig_t lo = make_launch_config(d, {8}, d.z_alt);
        ASSERT_EQ(CUPHY_STATUS_SUCCESS,
                  cuphyErrorCorrectionLDPCDecodeGetLaunchDescriptor(dec_, &hi));
        ASSERT_EQ(CUPHY_STATUS_SUCCESS,
                  cuphyErrorCorrectionLDPCDecodeGetLaunchDescriptor(dec_, &lo));

        if(d.has_z_pinned_variant)
        {
            EXPECT_NE(hi.kernel_node_params_driver.func, lo.kernel_node_params_driver.func)
                << "Z=" << d.Z << " and Z=" << d.z_alt << " resolved to the SAME kernel, "
                   "but this decoder compiles a Z-pinned variant";
        }
        else
        {
            EXPECT_EQ(hi.kernel_node_params_driver.func, lo.kernel_node_params_driver.func)
                << "Z=" << d.Z << " and Z=" << d.z_alt << " resolved to DIFFERENT kernels, "
                   "but this decoder compiles one Z-generic kernel";
        }

        // Block dimension is the lifting size in this family (blockDim == Z).
        EXPECT_EQ(static_cast<unsigned>(d.Z),     hi.kernel_node_params_driver.blockDimX);
        EXPECT_EQ(static_cast<unsigned>(d.z_alt), lo.kernel_node_params_driver.blockDimX);

        // Shared memory is monotone in Z for these kernels, so the smaller
        // lifting must ask for strictly less. Equality would mean the request
        // is not actually a function of Z.
        EXPECT_LT(lo.kernel_node_params_driver.sharedMemBytes,
                  hi.kernel_node_params_driver.sharedMemBytes);

        // Per-Z base graph table: two liftings must not share one.
        if(d.num_kernel_args >= 2)
        {
            EXPECT_NE(hi.kernel_args[1], lo.kernel_args[1])
                << "both liftings resolved to the same base graph descriptor";
        }
    }
}

// kBandDecoders carries ONE representative configuration per decoder -- the
// other tests iterate it and count on that -- so p=45/46 get their own case
// rather than extra rows.
//
// They are worth their own case because they take a different extension
// capture path: the fixed 66-word shared block holds exactly 66 columns
// (p=44), so at p>=45 the deepest extension columns are captured straight
// from global memory instead. p=40 never reaches that code.
TEST_F(LdpcBandDecoderGates, DeepExtensionBandAcceptedAtEveryLifting)
{
    const band_decoder& gmext = kBandDecoders[4];
    ASSERT_EQ(204, gmext.algo) << "kBandDecoders[4] is expected to be algo 204";

    for(int p : {45, 46})
    {
        for(int Z : {256, 288, 320, 352, 384})
        {
            SCOPED_TRACE("p=" + std::to_string(p) + " Z=" + std::to_string(Z));
            band_decoder d = gmext;
            d.num_parity_nodes = p;
            d.Z                = Z;

            cuphyLDPCDecodeLaunchConfig_t lc = make_launch_config(d, {2});
            ASSERT_EQ(CUPHY_STATUS_SUCCESS,
                      cuphyErrorCorrectionLDPCDecodeGetLaunchDescriptor(dec_, &lc));

            // Same contract the shared cases assert: a usable kernel, a shared
            // memory request the device can satisfy, and the third argument
            // slot this decoder alone uses for its extension scratch.
            EXPECT_NE(nullptr, lc.kernel_node_params_driver.func);
            EXPECT_EQ(static_cast<unsigned>(Z), lc.kernel_node_params_driver.blockDimX);
            EXPECT_NE(nullptr, lc.kernel_args[2])
                << "the p=35..46 decoder passes its extension scratch in kernel_args[2]";
        }
    }
}

} // namespace
