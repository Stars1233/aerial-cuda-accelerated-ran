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

/**
 * @file cuphy_ex_ldpc_cb.cpp
 * @brief Example demonstrating the codeblock-centric LDPC decoder SPI
 *
 * This example shows how to use the LDPC codeblock kernel SPI reference path
 * to decode multiple homogeneous batches of codeblocks grouped by configuration.
 * The API shape can represent heterogeneous batches, but the V1 reference path
 * emits one subgroup per launch batch.
 */

#include "cuphy.h"
#include "ldpc/ldpc_cb_kernel_api.h"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include "CLI/CLI.hpp"
#include "cuphy.hpp"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "ldpc_decode_test_vec_file.hpp"
#include "ldpc_decode_test_vec_gen.hpp"
#include "ldpc_decode_test_vec_pusch.hpp"

using namespace cuphy;

enum class LaunchPath
{
    Spi,
    PrepareLaunch
};

void* alignPointer(void* ptr, size_t alignment)
{
    const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    const auto aligned = (addr + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1);
    return reinterpret_cast<void*>(aligned);
}

////////////////////////////////////////////////////////////////////////
// LDPC_decode_error_stats (reused from cuphy_ex_ldpc.cpp)
class LDPC_decode_error_stats
{
public:
    typedef cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> err_count_tensor_t;

    LDPC_decode_error_stats() :
        bit_error_count_(0),
        bit_count_(0),
        block_error_count_(0),
        block_count_(0)
    {
    }

    template <class TSrc, class TDecoded>
    void update(TSrc& src, TDecoded& decoded, cudaStream_t stream)
    {
        const int             B      = src.desc().get_info().layout().dimensions()[0];
        const int             NUM_CW = decoded.dimensions()[1];
        cuphy::tensor_device  xor_results(CUPHY_BIT, B, NUM_CW);
        err_count_tensor_t    err_count(1, NUM_CW);

        cuphy::tensor_ref tDecodeB = decoded.subset(cuphy::index_group(cuphy::index_range(0, B),
                                                                       cuphy::dim_all()));
        cuphy::tensor_xor(xor_results, tDecodeB, src, stream);
        cuphy::tensor_reduction_sum(err_count, xor_results, 0, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int NUM_CW_actual = err_count.dimensions()[1];
        for(int i = 0; i < NUM_CW_actual; ++i)
        {
            uint32_t cwBitErrors = err_count(0, i);
            bit_error_count_ += cwBitErrors;
            if(cwBitErrors > 0)
            {
                ++block_error_count_;
            }
        }
        bit_count_   += (B * NUM_CW_actual);
        block_count_ += NUM_CW_actual;
    }

    uint64_t bit_error_count()   const { return bit_error_count_;   }
    uint64_t bit_count()         const { return bit_count_;         }
    uint32_t block_error_count() const { return block_error_count_; }
    uint32_t block_count()       const { return block_count_;       }
    float    BER()               const { return static_cast<float>(bit_error_count_)   / bit_count_;   }
    float    BLER()              const { return static_cast<float>(block_error_count_) / block_count_; }

private:
    uint64_t bit_error_count_;
    uint64_t bit_count_;
    uint32_t block_error_count_;
    uint32_t block_count_;
};

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;

    cuphyNvlogFmtHelper nvlog_fmt("ldpc_cb_decoder.log");

    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments using CLI11
        CLI::App app{"LDPC Codeblock-Centric Decoder Example"};

        std::string  inputFilename;
        int          numIterations        = 10;
        float        clampValue           = 32.0f;
        bool         useHalf              = true;
        int          parityNodes          = 8;
        int          algoIndex            = 0;
        bool         compareDecodeOutput  = true;
        unsigned int numRuns              = 1;
        int          numCBLimit           = -1;
        bool         doWarmup             = true;
        int          BG                   = 1;
        int          Zi                   = 384;
        float        SNR                  = 10.0f;
        int          maxBins              = 0;
        bool         useGraph             = false;
        int          tbIndex              = -1;
        std::string  launchPathOption     = "spi";

        app.add_option("-i", inputFilename, "Input HDF5 file name")
            ->group("Input");

        app.add_option("-n", numIterations, "Maximum number of LDPC iterations (default: 10)")
            ->check(CLI::Range(1, static_cast<int>(std::numeric_limits<uint16_t>::max())))
            ->group("Decoder Options");

        app.add_option("-C", clampValue, "Clamp value for LLR values (default: 32.0)")
            ->group("Decoder Options");

        app.add_flag("-f", useHalf, "Use half precision (default: true)")
            ->group("Decoder Options");

        app.add_option("-p", parityNodes, "Number of parity nodes (default: 8)")
            ->check(CLI::Range(4, 46))
            ->group("Decoder Options");

        app.add_option("-a", algoIndex, "Algorithm index (0 = auto, default: 0)")
            ->check(CLI::Range(0, static_cast<int>(std::numeric_limits<uint8_t>::max())))
            ->group("Decoder Options");

        app.add_option("-g", BG, "Base graph (1 or 2, default: 1)")
            ->check(CLI::Range(1, 2))
            ->group("Input");

        app.add_option("-Z", Zi, "Lifting size for generated data (default: 384)")
            ->check(CLI::Range(1, static_cast<int>(std::numeric_limits<uint16_t>::max())))
            ->group("Input");

        app.add_option("-S", SNR, "SNR in dB for generated noise (default: 10)")
            ->group("Input");

        app.add_option("-w", numCBLimit, "Limit number of codeblocks to decode")
            ->check(CLI::PositiveNumber)
            ->group("Input");

        app.add_option("-r", numRuns, "Number of runs for timing (default: 1)")
            ->check(CLI::PositiveNumber)
            ->group("Execution");

        app.add_flag("-s,!--compare", compareDecodeOutput, "Compare decoder output (default: true)")
            ->group("Execution");

        app.add_flag("-k,!--warmup", doWarmup, "Perform warmup run (default: true)")
            ->group("Execution");

        CLI::Option* maxBinsOption =
            app.add_option("-b", maxBins, "Maximum number of configuration bins (default: auto-sized)")
            ->check(CLI::PositiveNumber)
            ->group("CB Launch Options");

        app.add_flag("--graph", useGraph, "Exercise external graph execution with LDPC CB kernel node params (default: false)")
            ->group("CB Launch Options");

        app.add_option("--launch-path", launchPathOption, "LDPC CB launch path: spi or prepare (default: spi)")
            ->group("CB Launch Options");

        app.add_option("-t", tbIndex, "Transport block index for PUSCH test vectors (-1 = all, default: -1)")
            ->check(CLI::Range(-1, std::numeric_limits<int>::max()))
            ->group("Input");

        CLI11_PARSE(app, argc, argv);
        const bool maxBinsSpecified = (maxBinsOption->count() > 0);

        LaunchPath launchPath = LaunchPath::Spi;
        if (launchPathOption == "spi")
        {
            launchPath = LaunchPath::Spi;
        }
        else if (launchPathOption == "prepare")
        {
            launchPath = LaunchPath::PrepareLaunch;
        }
        else
        {
            printf("ERROR: Unsupported --launch-path value '%s' (expected 'spi' or 'prepare')\n",
                   launchPathOption.c_str());
            return 1;
        }

        //--------------------------------------------------------------
        // Display device info
        printf("*********************************************************************\n");
        cuphy::device gpuDevice;
        printf("%s\n", gpuDevice.desc().c_str());

        //--------------------------------------------------------------
        // Create cuPHY context and RNG (needed for test vector generation)
        cuphy::context ctx;
        cuphy::stream  cuStrmMain;
        cuphy::rng rng_gen;

        //--------------------------------------------------------------
        // Initialize test data
        if (!useHalf)
        {
            printf("ERROR: codeblock-centric LDPC interface supports only half-precision LLR input\n");
            return 1;
        }
        cuphyDataType_t LLR_type = CUPHY_R_16F;
        std::vector<std::unique_ptr<ldpc_decode_test_vec>> tb_test_vecs;
        tb_test_vecs.reserve(8);

        if(!inputFilename.empty())
        {
            // Try to detect if this is a PUSCH test vector file by checking for gnb_pars dataset
            hdf5hpp::hdf5_file fTest = hdf5hpp::hdf5_file::open(inputFilename.c_str());
            bool isPuschFile = fTest.is_valid_dataset("gnb_pars");

            if(isPuschFile)
            {
                hdf5hpp::hdf5_dataset gnb_pars_ds = fTest.open_dataset("gnb_pars");
                uint32_t numTb = gnb_pars_ds[0]["numTb"].as<uint32_t>();

                if (tbIndex < 0)
                {
                    printf("Detected PUSCH test vector file, loading all TBs (%u)\n", numTb);
                    for (uint32_t tb = 0; tb < numTb; ++tb)
                    {
                        tb_test_vecs.emplace_back(new ldpc_decode_test_vec_pusch(
                            test_vec_pusch_params(inputFilename.c_str(),
                                                  LLR_type,
                                                  static_cast<int>(tb),
                                                  numCBLimit)));
                    }
                }
                else
                {
                    if (static_cast<uint32_t>(tbIndex) >= numTb)
                    {
                        printf("ERROR: Requested TB index %d is out of range (file contains %u TB%s)\n",
                               tbIndex,
                               numTb,
                               numTb == 1 ? "" : "s");
                        return 1;
                    }
                    printf("Detected PUSCH test vector file, loading TB %d\n", tbIndex);
                    tb_test_vecs.emplace_back(new ldpc_decode_test_vec_pusch(
                        test_vec_pusch_params(inputFilename.c_str(),
                                              LLR_type,
                                              tbIndex,
                                              numCBLimit)));
                }
            }
            else
            {
                tb_test_vecs.emplace_back(new ldpc_decode_test_vec_file(
                    test_vec_file_params(inputFilename.c_str(),
                                         LLR_type,
                                         BG,
                                         parityNodes,
                                         numCBLimit)));
            }
        }
        else
        {
            int numCW = (numCBLimit > 0) ? numCBLimit : 80;
            tb_test_vecs.emplace_back(new ldpc_decode_test_vec_gen(
                ctx,
                rng_gen,
                test_vec_gen_params(LLR_type,
                                    BG,
                                    Zi,
                                    parityNodes,
                                    numCW,
                                    -1,    // blockSize
                                    0.0f,  // codeRate
                                    -1,    // modulatedBits
                                    CUPHY_QAM_4,
                                    SNR,
                                    false,                // puncture
                                    CUPHY_LDPC_CRC_NONE))); // CRC type
        }
        struct TbContext
        {
            ldpc_decode_test_vec* tv = nullptr;
            const ldpc_decode_test_vec_config* cfg = nullptr;
            cuphy::tensor_device tDecode;
            int num_cw = 0;
            int K = 0;
            int Kb = 0;
            int Z = 0;
            int mb = 0;
            int BG = 0;
            size_t llrStride = 0;
            size_t llrElemSize = 0;
            size_t bitsStride = 0;
        };

        std::vector<TbContext> tb_contexts;
        tb_contexts.reserve(tb_test_vecs.size());

        for (auto& tv_ptr : tb_test_vecs)
        {
            ldpc_decode_test_vec& tv = *tv_ptr;
            const ldpc_decode_test_vec_config& tv_cfg = tv.config();

            tv.print_config();

            const int NUM_CW  = tv_cfg.num_cw;
            const int K       = tv_cfg.K;
            const int Kb      = tv_cfg.Kb;
            const int Z       = tv_cfg.Z;
            const int mb      = tv_cfg.mb;
            const int tv_BG   = tv_cfg.BG;

            cuphy::tensor_device tDecode(CUPHY_BIT, K, NUM_CW, cuphy::tensor_flags::align_coalesce);

            const size_t llrStride = tv.LLR_desc().get_stride(1);
            const size_t llrElemSize = (LLR_type == CUPHY_R_16F) ? 2 : 4;
            const size_t bitsStride = tDecode.desc().get_stride(1) / 32; // Convert bits to uint32 words

            tb_contexts.push_back(TbContext{&tv, &tv_cfg, std::move(tDecode),
                                            NUM_CW, K, Kb, Z, mb, tv_BG,
                                            llrStride, llrElemSize, bitsStride});
        }

        if (tb_contexts.empty())
        {
            printf("ERROR: No test vectors available\n");
            return 1;
        }

        //--------------------------------------------------------------
        // Prepare codeblock data structures
        size_t totalCbs = 0;
        for (const auto& ctx : tb_contexts)
        {
            if (ctx.num_cw <= 0)
            {
                printf("ERROR: Invalid codeblock count %d in test vector\n", ctx.num_cw);
                return 1;
            }
            if (static_cast<size_t>(ctx.num_cw) > std::numeric_limits<size_t>::max() - totalCbs)
            {
                printf("ERROR: Total codeblock count overflow\n");
                return 1;
            }
            totalCbs += static_cast<size_t>(ctx.num_cw);
        }
        if (totalCbs > std::numeric_limits<uint32_t>::max())
        {
            printf("ERROR: Total codeblock count %zu exceeds supported range\n", totalCbs);
            return 1;
        }

        const uint32_t cbConfigFlags = 0; // Could add CUPHY_LDPC_CB_THROUGHPUT_MODE.

        cuphyLdpcCbKernelChooser_t chooser = nullptr;
        cuphyLdpcCbLaunchPreparer_t preparer = nullptr;
        std::vector<cuphyLdpcCbPreparedLaunch_t> preparedLaunches;
        auto cleanupLaunchState = [&]()
        {
            for (auto& launch : preparedLaunches)
            {
                if (launch != nullptr)
                {
                    cuphyLdpcCbPreparedLaunchDeinit(launch);
                    launch = nullptr;
                }
            }
            if (chooser != nullptr)
            {
                cuphyDestroyLdpcCbKernelChooser(chooser);
                chooser = nullptr;
            }
            if (preparer != nullptr)
            {
                cuphyDestroyLdpcCbLaunchPreparer(preparer);
                preparer = nullptr;
            }
        };

        cuphyStatus_t status = CUPHY_STATUS_SUCCESS;
        if (launchPath == LaunchPath::Spi)
        {
            cuphyLdpcCbKernelChooserConfig_t chooserConfig{};
            chooserConfig.llr_type     = LLR_type;
            chooserConfig.clamp_value  = clampValue;
            chooserConfig.config_flags = cbConfigFlags;

            status = cuphyCreateLdpcCbKernelChooser(&chooser, &chooserConfig);
            if (status != CUPHY_STATUS_SUCCESS)
            {
                printf("ERROR: Failed to create LDPC CB SPI kernel chooser (status=%d)\n", status);
                cleanupLaunchState();
                return 1;
            }
        }
        else
        {
            cuphyLdpcCbLaunchStaticConfig_t preparerConfig{};
            preparerConfig.llr_type     = LLR_type;
            preparerConfig.clamp_value  = clampValue;
            preparerConfig.config_flags = cbConfigFlags;

            status = cuphyCreateLdpcCbLaunchPreparer(&preparer, &preparerConfig);
            if (status != CUPHY_STATUS_SUCCESS)
            {
                printf("ERROR: Failed to create LDPC CB prepare-launch object (status=%d)\n", status);
                cleanupLaunchState();
                return 1;
            }
        }

        // Group codeblocks by configuration to minimize batches.
        struct CbConfigKey
        {
            int BG;
            int Z;
            int mb;
            int k;
            int max_iters;
            int algo;
        };

        struct CbConfigGroup
        {
            CbConfigKey key{};
            std::vector<int> tb_indices;
        };

        std::vector<CbConfigGroup> config_groups;
        config_groups.reserve(tb_contexts.size());

        for (size_t tb_idx = 0; tb_idx < tb_contexts.size(); ++tb_idx)
        {
            const auto& ctx = tb_contexts[tb_idx];
            const int64_t kBits = static_cast<int64_t>(ctx.Kb) * static_cast<int64_t>(ctx.Z);
            if (kBits < 0 || kBits > std::numeric_limits<int>::max())
            {
                printf("ERROR: Transport block %zu has unsupported K=%lld\n",
                       tb_idx,
                       static_cast<long long>(kBits));
                cleanupLaunchState();
                return 1;
            }
            CbConfigKey key{ctx.BG, ctx.Z, ctx.mb, static_cast<int>(kBits), numIterations, algoIndex};

            bool matched = false;
            for (auto& group : config_groups)
            {
                if (group.key.BG == key.BG &&
                    group.key.Z == key.Z &&
                    group.key.mb == key.mb &&
                    group.key.k == key.k &&
                    group.key.max_iters == key.max_iters &&
                    group.key.algo == key.algo)
                {
                    group.tb_indices.push_back(static_cast<int>(tb_idx));
                    matched = true;
                    break;
                }
            }

            if (!matched)
            {
                CbConfigGroup group;
                group.key = key;
                group.tb_indices.push_back(static_cast<int>(tb_idx));
                config_groups.push_back(std::move(group));
            }
        }

        if (maxBinsSpecified && config_groups.size() > static_cast<size_t>(maxBins))
        {
            printf("ERROR: Need %zu configuration bins, but maxBins=%d\n",
                   config_groups.size(), maxBins);
            cleanupLaunchState();
            return 1;
        }
        if (config_groups.size() > CUPHY_LDPC_CB_GROUP_ID_NONE)
        {
            printf("ERROR: Need %zu configuration bins, but only %u group IDs are available\n",
                   config_groups.size(), CUPHY_LDPC_CB_GROUP_ID_NONE);
            cleanupLaunchState();
            return 1;
        }

        std::vector<cuphyLdpcCbData_t> cbDataHost(totalCbs);
        std::vector<cuphyLdpcCbKernelChoice_t> choices(config_groups.size());
        std::vector<std::vector<cuphyLdpcCbBufferSpan_t>> spanStorage(config_groups.size());
        std::vector<cuphyLdpcCbSubgroupDesc_t> subgroups(config_groups.size());
        std::vector<cuphyLdpcCbLaunchBatchDesc_t> batchDescs(config_groups.size());
        std::vector<cuphyLdpcCbKernelLaunchBatch_t> launchBatches(config_groups.size());
        std::vector<cuphyLdpcCbKernelNodeParams_t> spiNodeParams(config_groups.size());
        std::vector<CUDA_KERNEL_NODE_PARAMS> kernelNodeParams(config_groups.size());
        std::vector<cuphyLdpcCbLaunchFamily_t> launchFamilies(config_groups.size());
        std::vector<cuphyLdpcCbPreparedLaunchInfo_t> preparedLaunchInfo(config_groups.size());
        std::vector<std::vector<uint8_t>> preparedWorkspaces(config_groups.size());
        preparedLaunches.assign(config_groups.size(), nullptr);
        std::vector<uint16_t> groupCounts(config_groups.size());
        std::vector<cuphyLdpcCbSubgroupKey_t> subgroupKeys(config_groups.size());

        size_t cbOffset = 0;
        for (size_t cfg_idx = 0; cfg_idx < config_groups.size(); ++cfg_idx)
        {
            const auto& group = config_groups[cfg_idx];

            const size_t groupStart = cbOffset;
            for (int tb_idx : group.tb_indices)
            {
                const auto& ctx = tb_contexts[static_cast<size_t>(tb_idx)];
                const uint8_t* llrBase = static_cast<const uint8_t*>(ctx.tv->LLR_addr());
                uint32_t* bitsBase = static_cast<uint32_t*>(ctx.tDecode.addr());

                for (int i = 0; i < ctx.num_cw; ++i)
                {
                    cbDataHost[cbOffset].llr_in   = llrBase + i * ctx.llrStride * ctx.llrElemSize;
                    cbDataHost[cbOffset].bits_out = bitsBase + i * ctx.bitsStride;
                    cbDataHost[cbOffset].llr_out  = nullptr;  // No soft output
                    cbDataHost[cbOffset].group_id = static_cast<uint16_t>(cfg_idx);
                    ++cbOffset;
                }
            }

            const size_t groupCount = cbOffset - groupStart;
            if (groupCount > std::numeric_limits<uint16_t>::max())
            {
                printf("ERROR: Configuration group %zu has too many codeblocks (%zu)\n",
                       cfg_idx, groupCount);
                cleanupLaunchState();
                return 1;
            }

            if (group.key.Z < 0 ||
                group.key.Z > static_cast<int>(std::numeric_limits<uint16_t>::max()) ||
                group.key.mb < 0 ||
                group.key.mb > static_cast<int>(std::numeric_limits<uint16_t>::max()) ||
                group.key.k < 0 ||
                group.key.k > static_cast<int>(std::numeric_limits<uint16_t>::max()))
            {
                printf("ERROR: Configuration group %zu has values outside SPI uint16_t range\n", cfg_idx);
                cleanupLaunchState();
                return 1;
            }

            groupCounts[cfg_idx] = static_cast<uint16_t>(groupCount);
            subgroupKeys[cfg_idx].bg           = (group.key.BG == 1) ? CUPHY_LDPC_CB_BG1 : CUPHY_LDPC_CB_BG2;
            subgroupKeys[cfg_idx].Zc           = static_cast<uint16_t>(group.key.Z);
            subgroupKeys[cfg_idx].parity_nodes = static_cast<uint16_t>(group.key.mb);
            subgroupKeys[cfg_idx].k            = static_cast<uint16_t>(group.key.k);
            subgroupKeys[cfg_idx].crc_type     = CUPHY_LDPC_CB_CRC_NONE;
            subgroupKeys[cfg_idx].max_iters    = static_cast<uint16_t>(numIterations);
            subgroupKeys[cfg_idx].algo         = static_cast<uint8_t>(algoIndex);

            if (launchPath == LaunchPath::Spi)
            {
                status = cuphyLdpcCbChooseKernel(chooser, &subgroupKeys[cfg_idx], &choices[cfg_idx]);
                if (status != CUPHY_STATUS_SUCCESS)
                {
                    printf("ERROR: cuphyLdpcCbChooseKernel failed for cfg %zu (status=%d)\n",
                           cfg_idx, status);
                    cleanupLaunchState();
                    return 1;
                }
            }
            else
            {
                status = cuphyLdpcCbQueryLaunchFamily(preparer, &subgroupKeys[cfg_idx], &launchFamilies[cfg_idx]);
                if (status != CUPHY_STATUS_SUCCESS)
                {
                    printf("ERROR: cuphyLdpcCbQueryLaunchFamily failed for cfg %zu (status=%d)\n",
                           cfg_idx, status);
                    cleanupLaunchState();
                    return 1;
                }
            }

            spanStorage[cfg_idx].resize(groupCount);
            uint16_t numSpans = static_cast<uint16_t>(spanStorage[cfg_idx].size());
            status = cuphyLdpcCbBuildSubgroupDesc(&subgroupKeys[cfg_idx],
                                                  cbDataHost.data() + groupStart,
                                                  groupCounts[cfg_idx],
                                                  spanStorage[cfg_idx].data(),
                                                  &numSpans,
                                                  &subgroups[cfg_idx]);
            if (status != CUPHY_STATUS_SUCCESS)
            {
                printf("ERROR: cuphyLdpcCbBuildSubgroupDesc failed for cfg %zu (status=%d)\n",
                       cfg_idx, status);
                cleanupLaunchState();
                return 1;
            }

            batchDescs[cfg_idx] = cuphyLdpcCbLaunchBatchDesc_t{&subgroups[cfg_idx], 1, nullptr};

            if (launchPath == LaunchPath::Spi)
            {
                launchBatches[cfg_idx] = cuphyLdpcCbKernelLaunchBatch_t{&choices[cfg_idx], &batchDescs[cfg_idx]};

                status = cuphyLdpcCbBuildKernelNodeParams(&launchBatches[cfg_idx], &spiNodeParams[cfg_idx]);
                if (status != CUPHY_STATUS_SUCCESS)
                {
                    printf("ERROR: cuphyLdpcCbBuildKernelNodeParams failed for cfg %zu (status=%d)\n",
                           cfg_idx, status);
                    cleanupLaunchState();
                    return 1;
                }
                kernelNodeParams[cfg_idx] = spiNodeParams[cfg_idx].params;
            }
            else
            {
                const cuphyLdpcCbPreparedLaunchConfig_t preparedConfig{
                    1,
                    static_cast<uint32_t>(groupCount),
                };
                size_t workspaceSize = 0;
                size_t workspaceAlignment = 0;
                status = cuphyLdpcCbPreparedLaunchGetWorkspaceSize(preparer,
                                                                   &launchFamilies[cfg_idx],
                                                                   &preparedConfig,
                                                                   &workspaceSize,
                                                                   &workspaceAlignment);
                if (status != CUPHY_STATUS_SUCCESS)
                {
                    printf("ERROR: cuphyLdpcCbPreparedLaunchGetWorkspaceSize failed for cfg %zu (status=%d)\n",
                           cfg_idx, status);
                    cleanupLaunchState();
                    return 1;
                }

                preparedWorkspaces[cfg_idx].assign(workspaceSize + workspaceAlignment, 0);
                void* workspace = alignPointer(preparedWorkspaces[cfg_idx].data(), workspaceAlignment);
                status = cuphyLdpcCbPreparedLaunchInitInPlace(preparer,
                                                              &launchFamilies[cfg_idx],
                                                              &preparedConfig,
                                                              workspace,
                                                              workspaceSize,
                                                              &preparedLaunches[cfg_idx]);
                if (status != CUPHY_STATUS_SUCCESS)
                {
                    printf("ERROR: cuphyLdpcCbPreparedLaunchInitInPlace failed for cfg %zu (status=%d)\n",
                           cfg_idx, status);
                    cleanupLaunchState();
                    return 1;
                }

                status = cuphyLdpcCbPrepareLaunch(preparedLaunches[cfg_idx],
                                                  &batchDescs[cfg_idx],
                                                  &preparedLaunchInfo[cfg_idx]);
                if (status != CUPHY_STATUS_SUCCESS)
                {
                    printf("ERROR: cuphyLdpcCbPrepareLaunch failed for cfg %zu (status=%d)\n",
                           cfg_idx, status);
                    cleanupLaunchState();
                    return 1;
                }

                status = cuphyLdpcCbGetKernelNodeParams(preparedLaunches[cfg_idx], &kernelNodeParams[cfg_idx]);
                if (status != CUPHY_STATUS_SUCCESS)
                {
                    printf("ERROR: cuphyLdpcCbGetKernelNodeParams failed for cfg %zu (status=%d)\n",
                           cfg_idx, status);
                    cleanupLaunchState();
                    return 1;
                }
            }
        }

        printf("\nLDPC CB launch path: %s, batches=%zu, total codeblocks=%zu\n",
               (launchPath == LaunchPath::Spi) ? "spi" : "prepare",
               kernelNodeParams.size(),
               totalCbs);
        printf("Codeblock configuration(s):\n");
        for (size_t i = 0; i < config_groups.size(); ++i)
        {
            const auto& key = subgroupKeys[i];
            printf("  [cfg %zu] num_cb=%u spans=%u BG=%d Zc=%d parity_nodes=%d k=%d max_iters=%d\n",
                   i,
                   groupCounts[i],
                   subgroups[i].num_spans,
                   key.bg,
                   key.Zc,
                   key.parity_nodes,
                   key.k,
                   key.max_iters);
        }
        printf("\n");

        //--------------------------------------------------------------
        // Get graph if requested for external graph execution path
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graphExec = nullptr;
        auto cleanupGraph = [&]()
        {
            if (graphExec)
            {
                cudaGraphExecDestroy(graphExec);
                graphExec = nullptr;
            }
            if (graph)
            {
                cudaGraphDestroy(graph);
                graph = nullptr;
            }
        };
        if (useGraph)
        {
            cudaError_t cuda_err = cudaGraphCreate(&graph, 0);
            if (cuda_err != cudaSuccess)
            {
                printf("WARNING: Failed to create CUDA graph, falling back to stream execution\n");
                useGraph = false;
            }
            else
            {
                for (size_t i = 0; i < kernelNodeParams.size(); ++i)
                {
                    cudaGraphNode_t node = nullptr;
                    CUresult cu_err = cuGraphAddKernelNode(
                        &node,
                        graph,
                        nullptr,
                        0,
                        &kernelNodeParams[i]);
                    if (cu_err != CUDA_SUCCESS)
                    {
                        printf("WARNING: Failed to add LDPC CB kernel node %zu to CUDA graph (CUresult=%d), falling back to stream execution\n",
                               i, cu_err);
                        cleanupGraph();
                        useGraph = false;
                        break;
                    }
                }

                if (useGraph)
                {
                    cuda_err = cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
                    if (cuda_err != cudaSuccess)
                    {
                        printf("WARNING: Failed to instantiate CUDA graph, falling back to stream execution\n");
                        cleanupGraph();
                        useGraph = false;
                    }
                    else
                    {
                        printf("Using externally launched CUDA graph execution path\n");
                    }
                }
            }
        }

        //--------------------------------------------------------------
        // Decoder execution loop
        LDPC_decode_error_stats error_stats;

        // Generate test data
        for (auto& ctx : tb_contexts)
        {
            ctx.tv->generate();
        }

        auto launchLdpcBatches = [&]() -> cuphyStatus_t
        {
            for (size_t i = 0; i < kernelNodeParams.size(); ++i)
            {
                const CUDA_KERNEL_NODE_PARAMS& params = kernelNodeParams[i];
                CUresult cu_err = cuLaunchKernel(params.func,
                                                 params.gridDimX,
                                                 params.gridDimY,
                                                 params.gridDimZ,
                                                 params.blockDimX,
                                                 params.blockDimY,
                                                 params.blockDimZ,
                                                 params.sharedMemBytes,
                                                 reinterpret_cast<CUstream>(cuStrmMain.handle()),
                                                 params.kernelParams,
                                                 params.extra);
                if (cu_err != CUDA_SUCCESS)
                {
                    printf("ERROR: cuLaunchKernel failed for LDPC CB batch %zu (CUresult=%d)\n",
                           i, cu_err);
                    return CUPHY_STATUS_INTERNAL_ERROR;
                }
            }
            return CUPHY_STATUS_SUCCESS;
        };

        // Warmup run
        if (doWarmup)
        {
            if (useGraph && graphExec)
            {
                cudaError_t cuda_err = cudaGraphLaunch(graphExec, cuStrmMain.handle());
                status = (cuda_err == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
            }
            else
            {
                status = launchLdpcBatches();
            }
            cuStrmMain.synchronize();
            if (status != CUPHY_STATUS_SUCCESS)
            {
                printf("ERROR: Warmup decode failed (status=%d)\n", status);
                cleanupGraph();
                cleanupLaunchState();
                return 1;
            }
        }

        // Timed runs
        cuphy::event_timer tmr;
        tmr.record_begin(cuStrmMain.handle());

        for (unsigned int run = 0; run < numRuns; ++run)
        {
            if (useGraph && graphExec)
            {
                cudaError_t cuda_err = cudaGraphLaunch(graphExec, cuStrmMain.handle());
                status = (cuda_err == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
            }
            else
            {
                status = launchLdpcBatches();
            }
            if (status != CUPHY_STATUS_SUCCESS)
            {
                printf("ERROR: Decode failed (status=%d)\n", status);
                break;
            }
        }
        if (status != CUPHY_STATUS_SUCCESS)
        {
            cleanupGraph();
            cleanupLaunchState();
            return 1;
        }

        tmr.record_end(cuStrmMain.handle());
        tmr.synchronize();

        float elapsed_ms = tmr.elapsed_time_ms();
        float avg_time_us = (elapsed_ms * 1000.0f) / numRuns;
        uint64_t total_info_bits = 0;
        for (const auto& ctx : tb_contexts)
        {
            total_info_bits += static_cast<uint64_t>(ctx.cfg->B) *
                               static_cast<uint64_t>(ctx.num_cw);
        }
        float throughput_gbps = (static_cast<float>(total_info_bits) * numRuns) /
                                (elapsed_ms / 1000.0f) / 1.0e9f;

        printf("Average (%u runs) elapsed time = %.1f us, throughput = %.2f Gbps\n",
               numRuns, avg_time_us, throughput_gbps);

        // Compare decoder output
        if (compareDecodeOutput)
        {
            for (auto& ctx : tb_contexts)
            {
                auto srcBits = ctx.tv->src_bits_view();
                error_stats.update(srcBits, ctx.tDecode, cuStrmMain.handle());
            }
            printf("Bit error count = %lu, BER = %.5e, BLER = (%u / %u) = %.5e\n",
                   error_stats.bit_error_count(),
                   error_stats.BER(),
                   error_stats.block_error_count(),
                   error_stats.block_count(),
                   error_stats.BLER());
        }

        //--------------------------------------------------------------
        // Cleanup
        cleanupGraph();
        cleanupLaunchState();
        printf("\nLDPC CB launch resources released\n");
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "UNKNOWN EXCEPTION");
        returnValue = 2;
    }

    return returnValue;
}
