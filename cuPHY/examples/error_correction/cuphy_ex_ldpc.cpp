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

#include "cuphy.h"
#include <cstdio>
#include <limits>
#include <string>
#include <vector>
#include "CLI/CLI.hpp"  // CLI11 header
#include "cuphy.hpp"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "ldpc_clamp_validation.hpp"
#include "ldpc_decode_test_vec_file.hpp"
#include "ldpc_decode_test_vec_gen.hpp"
#include "ldpc_interm_check.hpp"
#include "ldpc/ldpc_api.hpp"

#include <chrono>
#include <cstdlib>
#include <sstream>

using namespace cuphy;

////////////////////////////////////////////////////////////////////////
// LDPC_decode_error_stats
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
    //------------------------------------------------------------------
    // update()
    template <class TSrc, class TDecoded>
    void update(TSrc& src, TDecoded& decoded)
    {
        typedef cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tensor_uint32_p_t;

        const int             B      = src.dimensions()[0];
        const int             NUM_CW = decoded.dimensions()[1];
        cuphy::tensor_device  xor_results(CUPHY_BIT, B, NUM_CW);
        tensor_uint32_p_t     err_count(1, NUM_CW);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Generate a reference to ONLY the information bits of the
        // decoder output, without any possible filler bits.
        // tDecodeB = tDecode(0:B-1, :)
        cuphy::tensor_ref tDecodeB = decoded.subset(cuphy::index_group(cuphy::index_range(0, B),
                                                                       cuphy::dim_all()));
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // XOR decoder output with source bits. Each set bit in the
        // xor_results output indicates a bit error.
        // xor_results = tDecodeB ^ src_bits
        cuphy::tensor_xor(xor_results,
                          tDecodeB,
                          src);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Count the number of set bits in each column (codeword)
        cuphy::tensor_reduction_sum(err_count, xor_results, 0);
        cudaStreamSynchronize(0);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Update error statistics
        update_statistics(err_count, B);
    }
    void update_statistics(err_count_tensor_t& tErrorCount,
                           int                 bitsPerCodeword)
    {
        // Input is uint32_t tensor with dimensions (1, NUM_CW)
        int NUM_CW = tErrorCount.dimensions()[1];

        for(int i = 0; i < NUM_CW; ++i)
        {
            uint32_t cwBitErrors = tErrorCount(0, i);
            //printf("%i: %u\n", i, err_count(0, i));
            bit_error_count_ += cwBitErrors;
            if(cwBitErrors > 0)
            {
                ++block_error_count_;
            }
        }
        bit_count_   += (bitsPerCodeword * NUM_CW);
        block_count_ += NUM_CW;
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
// LDPC_decode_timing_stats
class LDPC_decode_timing_stats
{
public:
    LDPC_decode_timing_stats() :
        run_count_(0),
        total_time_milliseconds_(0.0),
        total_bits_(0.0)
    {
    }
    //------------------------------------------------------------------
    // update()
    void update(float t_milliseconds, int nruns, int64_t bits_per_run)
    {
        run_count_               += nruns;
        total_time_milliseconds_ += t_milliseconds;
        total_bits_              += (bits_per_run * nruns);
        //printf("Average (%u runs) elapsed time in usec = %.1f, throughput = %.2f Gbps\n",
        //       nruns,
        //       t_milliseconds * 1000 / nruns,
        //       (bits_per_run * nruns) / (t_milliseconds / 1000.0f) / 1.0e9);
    }
    int64_t num_runs()          const { return run_count_; }
    float   average_time_usec() const { return (total_time_milliseconds_ * 1000) / run_count_; }
    float   throughput()        const { return (total_bits_ * 1.0e-9) / (total_time_milliseconds_ / 1000.0); }
private:
    int64_t run_count_;
    double  total_time_milliseconds_;
    double  total_bits_;
};

// Parse a "p spec" (list of parity-node counts) into ints:
//   ""             -> [fallback]        (empty spec falls back to this p)
//   "4-46"         -> 4,5,...,46         (inclusive range)
//   "4,8,12"       -> 4,8,12             (explicit list)
//   "4-9,22,30-34" -> ranges + singles mixed
// Entries are emitted in the order written (not re-sorted).
static std::vector<int> parse_p_spec(const std::string& s, int fallback)
{
    std::vector<int> out;
    if(s.empty()) { out.push_back(fallback); return out; }
    std::stringstream ss(s);
    std::string       tok;
    while(std::getline(ss, tok, ','))
    {
        if(tok.empty()) { continue; }
        const auto dash = tok.find('-');
        if(dash != std::string::npos)
        {
            const int lo = std::atoi(tok.substr(0, dash).c_str());
            const int hi = std::atoi(tok.substr(dash + 1).c_str());
            for(int p = lo; p <= hi; ++p) { out.push_back(p); }
        }
        else
        {
            out.push_back(std::atoi(tok.c_str()));
        }
    }
    return out;
}

// Compare ref vs each DUT at ONE (BG, Z, p). ref_algo + duts already parsed.
// Prints one 'LLR_DIFF algo<d> vs algo<ref> p=<p>: PASS|FAIL|N/A' line per DUT.
// Returns 1 if any DUT FAILed, else 0 (the reference-unsupported case emits a
// per-DUT N/A line and returns 0 -- it is not a DUT failure).
static int llr_diff_one_p(cuphy::context&         ctx,
                          cuphy::rng&             rng_gen,
                          int                     BG,
                          int                     Z,
                          int                     p,
                          int                     ref_algo,
                          const std::vector<int>& duts,
                          int                     numCB,
                          int                     numIterations,
                          float                   clampValue,
                          int                     log2QAM,
                          float                   SNR,
                          uint32_t                crc_type,
                          float                   p99errRmsThreshold,
                          float                   snrThreshold,
                          bool                    dumpH5)
{
    // fp16 for the reference and all DUTs: the APP comparison needs identical input.
    const cuphyDataType_t llr_type = CUPHY_R_16F;
    const int             num_cw   = (numCB > 0) ? numCB : 2;
    const int             maxItr   = (numIterations > 0) ? numIterations : 10;

    // Self-generated input: random codeword -> encode -> modulate -> AWGN -> demod.
    // Always punctured, PUSCH-style: the first 2Z systematic LLRs are zeroed.
    ldpc_decode_test_vec_gen tv(ctx, rng_gen,
                                test_vec_gen_params(llr_type, BG, Z, p,
                                                    num_cw, /*blockSize*/-1,
                                                    /*codeRate*/0.0f, /*modBits*/-1, log2QAM,
                                                    SNR, /*puncture*/true, crc_type));
    tv.print_config();
    tv.generate();

    const ldpc_decode_test_vec_config& c = tv.config();
    const int Kb        = c.Kb;
    const int Zc        = c.Z;
    const int nV_parity = c.mb;
    const int C         = c.num_cw;
    const int K         = c.K;
    const int NUM_VAR   = Zc * (Kb + nV_parity);
    const int N_core    = (Kb + 4) * Zc;   // systematic + 4 core parity cols only

    printf("\nllr_diff (TV-free): BG=%d Z=%d p=%d C=%d ref=algo%d, gates p99err<=%.3f snr>=%.1fdB\n",
           BG, Zc, nV_parity, C, ref_algo, p99errRmsThreshold, snrThreshold);

    // Reference APP capture. If it fails (e.g. small Z NOT_SUPPORTED), emit a
    // per-DUT N/A(reference) line so the sweep harness can mark REF-N/A, and
    // move on to the next p rather than aborting the whole range.
    std::vector<float> h_APP_ref;
    try
    {
        capture_app_device(ctx, tv.LLR_desc(), tv.LLR_addr(), BG, Zc, nV_parity, K, C,
                           maxItr, clampValue, llr_type, ref_algo, maxItr, h_APP_ref);
    }
    catch(const std::exception& e)
    {
        fprintf(stderr, "Reference algo%d p=%d failed to capture APP: %s\n", ref_algo, p, e.what());
        for(int dut : duts)
        {
            printf("LLR_DIFF algo%d vs algo%d p=%d: N/A (reference algo%d capture failed)\n",
                   dut, ref_algo, p, ref_algo);
        }
        return 0;
    }
    if(APP_history_is_empty(h_APP_ref))
    {
        fprintf(stderr,
                "Reference algo%d APP is all zeros -- the library was not built with the\n"
                "dump-twin path for this algo. Rebuild with -DCUPHY_LDPC_SPLIT_DUMP_KERNELS=ON.\n",
                ref_algo);
        for(int dut : duts)
        {
            printf("LLR_DIFF algo%d vs algo%d p=%d: N/A (reference algo%d APP all zeros)\n",
                   dut, ref_algo, p, ref_algo);
        }
        return 0;
    }

    // Optional: dump ref + per-DUT APP (+ inputLLR) to h5 for plot_ldpc_app.py.
    // cuphy's LLR is column-major [NUM_VAR, C]: each codeword's NUM_VAR values are
    // contiguous, codewords back-to-back. Those same bytes ARE a row-major
    // [C, NUM_VAR], where NUM_VAR are also contiguous.
    std::string        ref_h5;
    std::vector<float> h_LLR;
    if(dumpH5)
    {
        cuphy::tensor_device tLLR_f32(CUPHY_R_32F, NUM_VAR, C);
        cuphy::tensor_convert(tLLR_f32, tv.LLR_desc(), tv.LLR_addr(), nullptr);
        cudaStreamSynchronize(nullptr);
        h_LLR.resize((size_t)C * NUM_VAR);
        cudaMemcpy(h_LLR.data(), tLLR_f32.addr(), h_LLR.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
        char fn[256];
        snprintf(fn, sizeof(fn), "ldpc_ref_a%d_bg%d_z%d_p%d.h5", ref_algo, BG, Zc, nV_parity);
        write_dut_h5(fn, h_LLR.data(), C, NUM_VAR, h_APP_ref.data(), maxItr, llr_type,
                     BG, Zc, Kb, nV_parity, c.F);
        ref_h5 = fn;
    }

    int any_fail = 0;
    for(int dut : duts)
    {
        std::vector<float> h_APP_dut;
        try
        {
            capture_app_device(ctx, tv.LLR_desc(), tv.LLR_addr(), BG, Zc, nV_parity, K, C,
                               maxItr, clampValue, llr_type, dut, maxItr, h_APP_dut);
        }
        catch(const std::exception& e)
        {
            // Decoder gate (UNSUPPORTED_CONFIG) or other decode refusal. This occurs
            // when the (BG, Z, p) is out of the range supported by this decoder.
            // This is not a failure.
            printf("LLR_DIFF algo%d vs algo%d p=%d: N/A (%s)\n", dut, ref_algo, p, e.what());
            continue;
        }
        if(APP_history_is_empty(h_APP_dut))
        {
            printf("LLR_DIFF algo%d vs algo%d p=%d: N/A (no dump path compiled)\n", dut, ref_algo, p);
            continue;
        }
        const bool pass = check_APP(h_APP_ref.data(), h_APP_dut.data(),
                                    C, maxItr, NUM_VAR, N_core, p99errRmsThreshold, snrThreshold);
        printf("LLR_DIFF algo%d vs algo%d p=%d: %s\n", dut, ref_algo, p, pass ? "PASS" : "FAIL");
        if(dumpH5)
        {
            char fn[256];
            snprintf(fn, sizeof(fn), "ldpc_dut_a%d_bg%d_z%d_p%d.h5", dut, BG, Zc, nV_parity);
            write_dut_h5(fn, h_LLR.data(), C, NUM_VAR, h_APP_dut.data(), maxItr, llr_type,
                         BG, Zc, Kb, nV_parity, c.F);
            printf("  plot: python3 cuPHY/util/ldpc/plot_ldpc_app.py %s %s\n",
                   ref_h5.c_str(), fn);
        }
        if(!pass) { any_fail = 1; }
    }
    return any_fail;
}

////////////////////////////////////////////////////////////////////////
// LDPC_decode_graph
//
// Decodes a transport block the way production does, which is not the way
// the rest of this example does.
//
// cuphyPuschRx never calls cuphyErrorCorrectionLDPCTransportBlockDecode().
// It asks the decoder for a launch descriptor, parks that descriptor in a
// kernel node of a CUDA graph, and rewrites the node's parameters once per
// slot: PuschRx::setupCmnPhase2() asks for the descriptor, and
// PuschRx::updateFullSlotGraph() rewrites the node through
// PuschRx::updateCbLdpcNodes(). Nothing else in the tree drives
// that path, so a decoder can pass every gate this example applies and still
// have a launch descriptor nobody has ever launched. '--graph' closes that
// by issuing the same four calls in the same order.
//
// The placeholder the node is created with takes TWO pointer arguments,
// matching pusch_rx's m_emptyNode2paramsDriver. That detail is the point
// rather than an incidental copy: the p=35..46 band decoder is a THREE
// argument kernel, so the executable graph is updated across a change in
// argument count, and an easier placeholder would not test it.
class LDPC_decode_graph
{
public:
    //------------------------------------------------------------------
    // Build the graph once: a single kernel node holding an empty
    // placeholder. The real parameters only ever arrive through
    // cuGraphExecKernelNodeSetParams(), as they do in production.
    LDPC_decode_graph()
    {
        void*                   arg                   = nullptr;
        void*                   placeholderParams[2]  = {&arg, &arg};
        CUDA_KERNEL_NODE_PARAMS emptyParams{};
        if(CUPHY_STATUS_SUCCESS !=
           cuphySetGenericEmptyKernelNodeParams(&emptyParams, 2, &placeholderParams[0]))
        {
            throw std::runtime_error("cuphySetGenericEmptyKernelNodeParams()");
        }
        // Anything already created has to go back if a later call fails: the
        // destructor does not run for a constructor that threw.
        try
        {
            // Launch on an explicitly created stream, not the default one.
            // The default stream carries implicit-synchronization semantics
            // that production streams do not, so decoding on it would make
            // this path differ from pusch_rx.cpp in exactly the dimension
            // '--graph' exists to mirror. cudaStreamCreate() gives a blocking
            // stream, so the surrounding default-stream timing and
            // cudaDeviceSynchronize() still order correctly around it.
            if(cudaSuccess != cudaStreamCreate(&stream_))
            {
                throw std::runtime_error("cudaStreamCreate()");
            }
            check(cuGraphCreate(&graph_, 0), "cuGraphCreate");
            check(cuGraphAddKernelNode(&node_, graph_, nullptr, 0, &emptyParams),
                  "cuGraphAddKernelNode");
#if CUDA_VERSION >= 12000
            check(cuGraphInstantiate(&exec_, graph_, 0), "cuGraphInstantiate");
#else
            check(cuGraphInstantiate(&exec_, graph_, 0, 0, 0), "cuGraphInstantiate");
#endif
        }
        catch(...)
        {
            destroy();
            throw;
        }
    }
    //------------------------------------------------------------------
    // One "slot": refresh the descriptor, re-derive the launch config,
    // push it into the instantiated graph, launch.
    void decode(const cuphy::LDPC_decoder& dec,
                const cuphyLDPCDecodeDesc_t& desc)
    {
        launch_config_.decode_desc = desc;
        dec.get_launch_config(launch_config_);  // throws on refusal
        check(cuGraphExecKernelNodeSetParams(exec_,
                                             node_,
                                             &launch_config_.kernel_node_params_driver),
              "cuGraphExecKernelNodeSetParams");
        check(cuGraphLaunch(exec_, static_cast<CUstream>(stream_)), "cuGraphLaunch");
    }
    //------------------------------------------------------------------
    ~LDPC_decode_graph()
    {
        destroy();
    }
    LDPC_decode_graph(const LDPC_decode_graph&)            = delete;
    LDPC_decode_graph& operator=(const LDPC_decode_graph&) = delete;
private:
    void destroy() noexcept
    {
        if(exec_)   cuGraphExecDestroy(exec_);
        if(graph_)  cuGraphDestroy(graph_);
        if(stream_) cudaStreamDestroy(stream_);
        exec_   = nullptr;
        graph_  = nullptr;
        node_   = nullptr;
        stream_ = nullptr;
    }
    static void check(CUresult r, const char* what)
    {
        if(CUDA_SUCCESS != r)
        {
            const char* pErrStr = nullptr;
            cuGetErrorString(r, &pErrStr);
            throw std::runtime_error(std::string(what) + "(): " +
                                     (pErrStr ? pErrStr : "unknown CUDA driver error"));
        }
    }
    CUgraph                       graph_ = nullptr;
    CUgraphExec                   exec_  = nullptr;
    CUgraphNode                   node_  = nullptr;
    cudaStream_t                  stream_ = nullptr;
    cuphyLDPCDecodeLaunchConfig_t launch_config_{};
};

////////////////////////////////////////////////////////////////////////
// main()

// TV-free per-iteration APP (LLR) comparison  (--llr_diff)
// Self-generated input (random codeword + AWGN, punctured): capture the reference
// algo's APP each iteration and compare every DUT algo against it, printing one
// 'LLR_DIFF algo<d> vs algo<ref> p=<p>: PASS|FAIL|N/A' line per DUT.
////////////////////////////////////////////////////////////////////////
static int run_llr_diff_self_gen(cuphy::context&    ctx,
                                 cuphy::rng&        rng_gen,
                                 int                BG,
                                 int                Z,
                                 int                parityNodes,
                                 const std::string& pRangeArg,
                                 const std::string& zRangeArg,
                                 int                numCB,
                                 int                numIterations,
                                 float              clampValue,
                                 int                log2QAM,
                                 float              SNR,
                                 const std::string& snrListArg,
                                 uint32_t           crc_type,
                                 const std::string& algosArg,
                                 float              p99errRmsThreshold,
                                 float              snrThreshold,
                                 bool               dumpH5)
{
    // Parse "ref,dut1,dut2,..." (first entry = reference).
    std::vector<int> algos;
    {
        std::stringstream ss(algosArg);
        std::string       tok;
        while(std::getline(ss, tok, ','))
        {
            if(!tok.empty()) { algos.push_back(std::atoi(tok.c_str())); }
        }
    }
    if(algos.size() < 2)
    {
        fprintf(stderr, "Error: --llr_diff expects <ref,dut1,...> (>= 2 algos). Got: %s\n",
                algosArg.c_str());
        return 1;
    }
    const int              ref_algo = algos[0];
    const std::vector<int> duts(algos.begin() + 1, algos.end());

    // p-range: --llr_diff_prange overrides -p; empty -> single -p. Looping here
    // (rather than one process per p) amortizes CUDA context + module load,
    // which dominates the per-cell cost in a sweep.
    std::vector<int> plist = parse_p_spec(pRangeArg, parityNodes);

    // Optional per-p SNR list (matched 1:1 to the -prange as given); empty -> scalar SNR for all p.
    // Parsed before the p-filter so it can be pruned in lockstep with plist below.
    std::vector<float> snrList;
    if(!snrListArg.empty())
    {
        std::stringstream ss(snrListArg);
        std::string       tok;
        while(std::getline(ss, tok, ','))
        {
            if(!tok.empty()) { snrList.push_back(static_cast<float>(std::atof(tok.c_str()))); }
        }
        if(snrList.size() != plist.size())
        {
            fprintf(stderr, "Error: --llr_diff_snrs has %zu value(s) but --llr_diff_prange "
                    "expands to %zu p-value(s)\n", snrList.size(), plist.size());
            return 1;
        }
    }

    // Drop p values the base graph cannot represent (a convenience range like '4-46' then
    // degrades gracefully on BG2 (max 42) instead of aborting the grid on the first illegal
    // cell), pruning the per-p SNR list in lockstep so it stays 1:1 with the kept p. Matches
    // the clamp llr_diff_sweep.py already applies; the generator throws on these as a backstop.
    {
        const int max_p = ((1 == BG) ? CUPHY_LDPC_MAX_BG1_VAR_NODES
                                     : CUPHY_LDPC_MAX_BG2_VAR_NODES) - ((1 == BG) ? 22 : 10);
        std::vector<int>   kept;
        std::vector<float> keptSnr;
        std::vector<int>   dropped;
        for(size_t i = 0; i < plist.size(); ++i)
        {
            if(plist[i] <= max_p)
            {
                kept.push_back(plist[i]);
                if(!snrList.empty()) { keptSnr.push_back(snrList[i]); }
            }
            else
            {
                dropped.push_back(plist[i]);
            }
        }
        if(!dropped.empty())
        {
            printf("Note: BG%d supports p<=%d; skipping out-of-range p:", BG, max_p);
            for(int p : dropped) { printf(" %d", p); }
            printf("\n");
        }
        if(kept.empty())
        {
            fprintf(stderr, "Error: no p value in range for BG%d (max p=%d).\n", BG, max_p);
            return 1;
        }
        plist   = std::move(kept);
        snrList = std::move(keptSnr);
    }

    // z-range: --llr_diff_zrange overrides -Z; empty -> single -Z. Looping Z here
    // (like the p-range above) amortizes the CUDA + module load over the whole grid.
    const std::vector<int> zlist = parse_p_spec(zRangeArg, Z);

    const auto t_start = std::chrono::steady_clock::now();
    int any_fail = 0;
    for(const int z : zlist)
    {
        for(size_t i = 0; i < plist.size(); ++i)
        {
            const int   p     = plist[i];
            const float snr_p = snrList.empty() ? SNR : snrList[i];
            any_fail |= llr_diff_one_p(ctx, rng_gen, BG, z, p, ref_algo, duts, numCB,
                                       numIterations, clampValue, log2QAM, snr_p,
                                       crc_type, p99errRmsThreshold, snrThreshold, dumpH5);
        }
    }
    const double elapsed_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    printf("\nllr_diff elapsed: %.3f s  (%zu Z-value(s), %zu p-value(s), %zu DUT(s), ref=algo%d)\n",
           elapsed_s, zlist.size(), plist.size(), duts.size(), ref_algo);
    return any_fail;
}

int main(int argc, char* argv[])
{
    int returnValue = 0;

    cuphyNvlogFmtHelper nvlog_fmt("ldpc_decoder.log");

    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments using CLI11
        CLI::App app{
            "LDPC Decoder Example\n"
            "\n"
            "Examples:\n"
            "  # one (Z,p) cell: per-iteration APP diff table, algo55 vs the algo40 reference:\n"
            "  cuphy_ex_ldpc -g 1 -Z 384 -p 8 --llr_diff 40,55 -n 10 -S 12\n"
            "  # sweep a p-range in one process (amortizes CUDA init):\n"
            "  cuphy_ex_ldpc -g 1 -Z 384 --llr_diff 40,55 --llr_diff_prange 4,8 -n 10 -S 12\n"
            "  # decode + BLER, TB interface (QPSK):\n"
            "  cuphy_ex_ldpc -n 10 -Z 384 -w 2888 -p 4 -g 1 -r 1 -P -S 6.8615 -f fp16 -b -a 56\n"
            "  # decode + BLER, TB interface (QAM256):\n"
            "  cuphy_ex_ldpc -n 10 -Z 384 -w 2888 -p 4 -g 1 -r 1 -M QAM256 -S 24.745 -P -f fp16 -b -a 56"};

        // Command line options
        std::string     inputFilename;
        int             numIterations        = 1;
        float           clampValue           = 32.0f;
        cuphyDataType_t llrType              = CUPHY_R_16F;
        int             parityNodes          = 8;
        int             algoIndex            = 0;
        bool            skipCompareOutput    = false;
        unsigned int    numRuns              = 1;
        int             numCBLimit           = -1;
        bool            skipWarmup           = false;
        float           minSumNorm           = 0.0f;
        int             BG                   = 1;
        bool            puncture             = false;
        int             Zi                   = 384;
        float           SNR                  = 10.0f;
        bool            useTBInterface       = false;
        int             blockSize            = -1;
        float           codeRate             = 0.0f;
        int             modulatedBits        = -1;
        int             log2QAM              = CUPHY_QAM_4;  // Default to QPSK
        int             min_block_err_cnt    = 0;
        int             max_block_cnt        = 1000000;
        bool            chooseHighThroughput = false;
        bool            spreadTB             = false;
        bool            writeSoftOutputs     = false;
        uint32_t        crc_type             = CUPHY_LDPC_CRC_NONE;
        bool            enableEarlyTerm      = false;
        bool            writeIterCount       = false;
        bool            etLatencyDebug       = false;
        bool            forceEtKernel        = false;
        bool            useGraph             = false;
        std::string     outputFilename;
        std::string     llrDiffArg;
        std::string     llrDiffPrange;
        std::string     llrDiffZrange;
        std::string     llrDiffSnrs;
        bool            llrDiffDumpH5         = false;
        float           p99errRmsThreshold    = LLR_P99ERR_RMS_THRESHOLD_DEFAULT;
        float           snrThreshold          = LLR_SNR_THRESHOLD_DEFAULT;

        // Notes that apply to whole groups of options, rather than to any single one.
        // (These are a footer, not CLI11 option groups: an option group is a subcommand
        // that inherits its own '--help' flag, so empty ones only add duplicate help text.
        // The sections below come from the ->group(...) name given to each option.)
        // CLI11 re-wraps each footer paragraph, so leading/repeated spaces are not preserved:
        // keep every paragraph on one logical line rather than hand-aligning it.
        app.footer(
            "File Based Input:\n"
            "When using file based input, no additional puncturing or shortening is performed. "
            "BER/BLER will reflect the puncturing/shortening conditions used to generate the "
            "input data. The number of input information bits is determined from the "
            "'sourceData' data set in the input file, and the lifting size Z is derived "
            "appropriately.\n"
            "\n"
            "Generating Input Data:\n"
            "Ways to specify randomly generated input data: \n"
            "'-B -N' (input block size and number of modulated bits), \n" 
            "'-B -R' (input block size and code rate), or \n"
            "'-p -Z' (num parity nodes and lifting size).");

        // Execution (Common) Options
        app.add_option("-a", algoIndex,
            "Use specific implementation (default: 0 - let library decide). "
            "Availability of a positive index depends on GPU architecture and LDPC configuration.")
            ->check(CLI::NonNegativeNumber)
            ->group("Execution (Common) Options");

        app.add_flag("-b", useTBInterface,
            "Use the transport block LDPC interface (instead of the tensor interface)")
            ->group("Execution (Common) Options");

        app.add_flag("--graph", useGraph,
            "VALIDATION ONLY -- DO NOT USE FOR PERFORMANCE MEASUREMENT.\n"
            "Decode through the CUDA graph path that production uses: ask the decoder for a\n"
            "launch descriptor (cuphyErrorCorrectionLDPCDecodeGetLaunchDescriptor), hold it\n"
            "in a kernel node, and update that node per run, as cuphyPuschRx does -- instead\n"
            "of calling the transport block decode entry point directly. This exists to\n"
            "exercise that descriptor, which no other in-tree caller does; a CUDA graph\n"
            "wrapping a SINGLE kernel launch offers no throughput or latency benefit, and\n"
            "this path is measurably SLOWER because re-deriving and re-installing the node\n"
            "parameters every run is host work the direct call does not do. Any timing or\n"
            "throughput figure printed under '--graph' is meaningless as a decoder result.\n"
            "Requires '-b'.")
            ->group("Execution (Common) Options");

        app.add_option("-c", max_block_cnt,
            "Terminate the data loop when the given number of blocks has been decoded.\n"
            "When the '-e' error count option is provided, this option can be used to avoid\n"
            "an infinite loop, as the provided SNR may not generate any bit or block errors.")
            ->check(CLI::PositiveNumber)
            ->group("Execution (Common) Options");

        app.add_flag("-d", spreadTB,
            "When using the transport block interface, 'spread' the data over multiple\n"
            "transport blocks (" + std::to_string(CUPHY_LDPC_DECODE_DESC_MAX_TB) + "), instead of one large transport block")
            ->group("Execution (Common) Options");

        app.add_option("-e", min_block_err_cnt,
            "Generate data and accumulate error statistics until specified blocks containing\n"
            "an error have occurred. (Not used for file-based input.)")
            ->check(CLI::NonNegativeNumber)
            ->group("Execution (Common) Options");

        app.add_option("-f,--fptype", llrType,
            "Data type for input LLR and intermediate APP values (fp16, fp32, fp8e4m3, fp8e5m2)")
            ->transform(CLI::CheckedTransformer(CLI::TransformPairs<cuphyDataType_t>
                                                {
                                                    {"fp16",    CUPHY_R_16F},
                                                    {"fp32",    CUPHY_R_32F},
                                                    {"fp8e4m3", CUPHY_R_8F_E4M3},
                                                    {"fp8e5m2", CUPHY_R_8F_E5M2},
                                                }))
            ->group("Execution (Common) Options");

        app.add_flag("-k", skipWarmup,
            "Skip 'warmup' run before timing loop")
            ->default_val(false)
            ->group("Execution (Common) Options");

        app.add_option("-m", minSumNorm,
            "Normalization factor for min-sum. If no value is provided, the library\n"
            "will choose an appropriate value, based on the LDPC configuration.")
            ->group("Execution (Common) Options");

        app.add_option("-n", numIterations,
            "Maximum number of LDPC iterations (default: 1)")
            ->check(CLI::NonNegativeNumber)
            ->group("Execution (Common) Options");

        app.add_option("-C", clampValue, "Clamp value for floating point LLR values (default: 32.0)")
           ->check(CLI::PositiveNumber)
           ->group("Execution (Common) Options");

        app.add_option("-o", outputFilename,
            "Write output data to an HDF5 file with the given name.\n"
            "If an output file name is provided and the -u option is used,\n"
            "the soft output LLR values will be placed in the file.")
            ->group("Execution (Common) Options");

        app.add_option("-r", numRuns,
            "Number of times to perform batch decoding (default: 1)")
            ->check(CLI::PositiveNumber)
            ->group("Execution (Common) Options");

        app.add_flag("-s", skipCompareOutput,
            "Skip comparison of decoder output to input data")
            ->default_val(false)
            ->group("Execution (Common) Options");

        app.add_flag("-t", chooseHighThroughput,
            "Instruct the library algorithm chooser to choose a kernel optimized for\n"
            "throughput (instead of latency) when a high throughput kernel is available.\n"
            "(Only valid when algo_index is 0 or is not specified on the command line.)")
            ->group("Execution (Common) Options");

        app.add_flag("-u", writeSoftOutputs,
            "Write soft output data into a buffer. If an output file name is provided\n"
            "via the -o option, the soft output LLR values will be placed in the file.")
            ->group("Execution (Common) Options");

        app.add_option("--llr_diff", llrDiffArg,
            "TV-free per-iteration APP (LLR) comparison: --llr_diff <ref,dut1,...>\n"
            "(e.g. 40,35). Generates ONE input (random codeword + AWGN)\n"
            "for the -g/-Z/-p config, captures the reference algo's APP history, then\n"
            "compares each DUT algo's APP against it (gates: p99err_rms + SNR + last-3 sign-flip). Prints one\n"
            "'LLR_DIFF algo<d> vs algo<ref> p=<p>: PASS|FAIL|N/A' line per DUT/p. fp16\n"
            "only, and the input is always punctured (PUSCH-style).\n"
            "Validates per-iteration APP fidelity (core columns only), NOT product decode;\n"
            "store-elision decoders need a separate BLER/waterfall check.")
            ->group("Execution (Common) Options");

        app.add_option("--llr_diff_prange", llrDiffPrange,
            "Sweep a p-range in ONE process for --llr_diff (amortizes CUDA init):\n"
            "'4-46', list '4,8,12', or mixed '4-9,22,30-34'. Overrides -p. The\n"
            "reference APP is captured per p; one verdict line is printed per (dut,p).")
            ->group("Execution (Common) Options");

        app.add_option("--llr_diff_zrange", llrDiffZrange,
            "Sweep a Z-list in the SAME process as --llr_diff_prange (amortizes the CUDA\n"
            "module load across the whole Z x p grid). Comma-list '256,320,384' or range\n"
            "'32-384'; a range expands to every integer, so pass valid lifting sizes for\n"
            "clean output. Overrides -Z; omitted -> single -Z. Always the Z x p product.")
            ->group("Execution (Common) Options");

        app.add_option("--llr_diff_snrs", llrDiffSnrs,
            "Per-p AWGN SNR list (dB), comma-separated, matched 1:1 to the expanded\n"
            "--llr_diff_prange order. Overrides -S per p (e.g. Shannon+margin from the\n"
            "sweep script). If omitted, -S is used for every p.")
            ->group("Execution (Common) Options");

        app.add_flag("--dump-h5", llrDiffDumpH5,
            "For --llr_diff: dump the reference + each DUT APP history (and inputLLR)\n"
            "to ldpc_{ref,dut}_a<algo>_bg<N>_z<Z>_p<P>.h5 and print a ready-to-run\n"
            "plot_ldpc_app.py command. Intended for single-cell drill-down.")
            ->group("Execution (Common) Options");

        app.add_option("--p99err_rms_threshold", p99errRmsThreshold,
            "Per-iteration p99err_rms (= pctile99|ref-dut| / rms_ref) pass ceiling for --llr_diff.\n"
            "(default: 0.07)")
            ->group("Execution (Common) Options");

        app.add_option("--snr_threshold", snrThreshold,
            "Per-iteration SNR(dB) pass floor for --llr_diff. (default: 30)")
            ->group("Execution (Common) Options");

            app.add_flag("-x", enableEarlyTerm,
                "Enable CRC-based early termination. When enabled, the decoder may exit\n"
                "before reaching max_iterations if the CRC check passes. (TB interface only)")
                ->group("Execution (Common) Options");

            app.add_flag("-y", writeIterCount,
                "Write iteration count per codeblock to a buffer. Displays the number of\n"
                "iterations used for each codeblock after decoding. (TB interface only)")
                ->group("Execution (Common) Options");

            app.add_flag("--et_latency_debug", etLatencyDebug,
                "Exercise early-termination checks but force max-iteration latency. Implies -x.\n"
                "The final iteration still runs the real CRC check. (TB interface only)")
                ->group("Execution (Common) Options");

            app.add_flag("--force_et_kernel", forceEtKernel,
                "Launch the accessory-capable LDPC kernel without enabling ET or other\n"
                "accessories. Intended for compiled-in versus compiled-out timing.")
                ->group("Execution (Common) Options");

        app.add_option("-g", BG,
            "Base graph used to generate input data (default: 1)")
            ->check(CLI::Range(1, 2))
            ->group("Execution (Common) Options");

        // File Based Input options
        app.add_option("-i", inputFilename,
            "Input HDF5 file name, which must contain the following datasets:\n"
            "    sourceData:    uint8 data set with source information bits\n"
            "    inputLLR:      Log-likelihood ratios for coded, modulated symbols\n"
            "    inputCodeWord: uint8 data set with encoded bits (optional)\n"
            "                  (Initial bits are sourceData. No puncturing assumed.)")
            ->group("File Based Input");
        app.add_option("-p", parityNodes,
            "Number of parity nodes mb (must be between 4 and 46 for BG1, and between\n"
            "4 and 42 for BG2). This value is not used if the code rate 'R' is specified,\n"
            "or if the number of modulated bits 'N' is specified. (default: 8)")
            ->check(CLI::Range(4, 46))
            ->group("File Based Input");

        app.add_option("-w", numCBLimit,
            "For file input: Decode numCBLimit code blocks (instead of the total number contained\n"
            "in the input file). Must be less than or equal to the number of codewords in the file.\n"
            "For generated input: Number of codewords to generate (default: 80)")
            ->check(CLI::PositiveNumber)
            ->group("File Based Input");

        // Generating Input Data options
        app.add_option("-B", blockSize,
            "Input data block size (before LDPC encoding). Uses the base graph selection\n"
            "to determine the lifting size Z.")
            ->check(CLI::PositiveNumber)
            ->group("Generating Input Data");

        app.add_option("-M", "Modulation used before adding noise. Valid values are 'BPSK', 'QPSK',\n"
            "'QAM16', 'QAM64', or 'QAM256'. (default: 'QPSK')")
            ->transform([&log2QAM](const std::string& mod) {
                if(mod == "QAM256") { log2QAM = CUPHY_QAM_256; return std::string("QAM256"); }
                if(mod == "QAM64")  { log2QAM = CUPHY_QAM_64;  return std::string("QAM64"); }
                if(mod == "QAM16")  { log2QAM = CUPHY_QAM_16;  return std::string("QAM16"); }
                if(mod == "QPSK")   { log2QAM = CUPHY_QAM_4;   return std::string("QPSK"); }
                if(mod == "BPSK")   { log2QAM = CUPHY_QAM_2;   return std::string("BPSK"); }
                throw CLI::ValidationError("Invalid modulation");
            })
            ->group("Generating Input Data");

        app.add_option("-N", modulatedBits,
            "Number of modulated bits (info + parity) in each codeword.")
            ->check(CLI::PositiveNumber)
            ->group("Generating Input Data");

        app.add_option("-R", codeRate,
            "Code rate. Used with block size parameter 'B' to determine the number of parity\n"
            "nodes and punctured parity bits. Ignored if '-N' option is used, and instead\n"
            "derived from that value.")
            ->check(CLI::Range(0.0f, 1.0f))
            ->group("Generating Input Data");

        app.add_flag("-P", puncture,
            "Puncture the generated test vector data (default: false)")
            ->group("Generating Input Data");

        app.add_option("-S", SNR,
            "SNR (in dB) for generated noise. The (complex) noise variance is given by\n"
            "10^(-SNR_dB/10). The variance of the real and imaginary components are assumed\n"
            "to be equal, and in this case each is equal to half of the complex variance.\n"
            "(default SNR: 10)")
            ->group("Generating Input Data");

        app.add_option("-Z", Zi,
            "Lifting size for generated data. This option is only used if the data block\n"
            "size is NOT specified. If this option is specified, the number of filler bits\n"
            "is zero, and no parity bits are punctured.")
            ->check(CLI::PositiveNumber)
            ->group("Generating Input Data");

        app.add_option("--crc",
            "CRC type for generated data. Valid values are 'none', 'crc24a', 'crc24b', 'crc16'.\n"
            "(default: 'none')")
            ->transform([&crc_type](const std::string& crc) {
                if(crc == "crc24a") { crc_type = CUPHY_LDPC_CRC_24A; return std::string("crc24a"); }
                if(crc == "crc24b") { crc_type = CUPHY_LDPC_CRC_24B; return std::string("crc24b"); }
                if(crc == "crc16") { crc_type = CUPHY_LDPC_CRC_16; return std::string("crc16"); }
                if(crc == "none") { crc_type = CUPHY_LDPC_CRC_NONE; return std::string("none"); }
                throw CLI::ValidationError("Invalid CRC type");
            })
            ->group("Generating Input Data");

        // Parse command line arguments
        CLI11_PARSE(app, argc, argv);

        if(!ldpc_example::is_valid_clamp_value(llrType, clampValue))
        {
            NVLOGE_FMT(NVLOG_PUSCH,
                       AERIAL_CUPHY_EVENT,
                       "ERROR: Invalid clamp value {} for {}. It must be greater than 0 and less than {}.",
                       clampValue,
                       cuphyGetDataTypeString(llrType),
                       ldpc_example::max_finite_clamp_value(llrType));
            return 1;
        }

        //--------------------------------------------------------------
        // The graph path exists only for the transport block interface:
        // get_launch_config() builds a node for the decode_tb() kernel.
        if(useGraph && !useTBInterface)
        {
            NVLOGE_FMT(NVLOG_PUSCH,
                       AERIAL_CUPHY_EVENT,
                       "ERROR: '--graph' drives the transport block decode path. "
                       "Add the '-b' option.");
            return 1;
        }


        if(etLatencyDebug)
        {
            enableEarlyTerm = true;
        }

            if (enableEarlyTerm)
            {
                if (crc_type == CUPHY_LDPC_CRC_NONE)
                {
                    printf("WARNING: Early termination is enabled, but no CRC type is specified.  Defaulting to CRC-24B.\n");
                    crc_type = CUPHY_LDPC_CRC_24B;
                }
            }

        //--------------------------------------------------------------
        // Display device (GPU) info
        printf("*********************************************************************\n");
        cuphy::device gpuDevice;
        printf("%s\n", gpuDevice.desc().c_str());
        //--------------------------------------------------------------
        // Create a cuPHY context
        cuphy::context ctx;
        //--------------------------------------------------------------
        // Create a random number generator, in case we need it to
        // generate source input data
        unsigned long long rng_seed = 0;
        cuphy::rng rng_gen(rng_seed);
        //--------------------------------------------------------------
        // TV-free per-iteration APP comparison mode (--llr_diff). Handled
        // here, before the normal single-algo decode path.
        if(!llrDiffArg.empty())
        {
            if(llrType != CUPHY_R_16F)
            {
                fprintf(stderr,
                        "Error: --llr_diff runs fp16 only, but --fptype %s was given.\n",
                        cuphyGetDataTypeString(llrType));
                return 1;
            }
            return run_llr_diff_self_gen(ctx, rng_gen, BG, Zi, parityNodes, llrDiffPrange, llrDiffZrange, numCBLimit,
                                         numIterations, clampValue, log2QAM, SNR, llrDiffSnrs,
                                         crc_type, llrDiffArg,
                                         p99errRmsThreshold, snrThreshold, llrDiffDumpH5);
        }
        //--------------------------------------------------------------
        // Initialize test data and the LDPC configuration using command
        // line arguments.
        std::unique_ptr<ldpc_decode_test_vec> ptv;
        if(!inputFilename.empty())
        {
            // Load a test vector from an input file
            ptv.reset(new ldpc_decode_test_vec_file(test_vec_file_params(inputFilename.c_str(), // input file name
                                                                         llrType,               // LLR data type
                                                                         BG,                    // base graph
                                                                         parityNodes,           // num parity nodes
                                                                         numCBLimit)));         // limit num CWs
        }
        else
        {
            // Generate test vector data randomly
            ptv.reset(new ldpc_decode_test_vec_gen(ctx,                                  // cuPHY context
                                                   rng_gen,                              // random number generator
                                                   test_vec_gen_params(llrType,          // LLR data type
                                                                       BG,               // base graph
                                                                       Zi,               // lifting size
                                                                       parityNodes,      // num parity nodes
                                                                       numCBLimit,       // number of codewords
                                                                       blockSize,        // number of input bits
                                                                       codeRate,         // code rate
                                                                       modulatedBits,    // number of modulated bits
                                                                       log2QAM,          // modulation
                                                                       SNR,              // signal-to-noise ratio
                                                                       puncture,         // puncture first 2Z LLRs
                                                                       crc_type)));      // CRC type
        }
        //------------------------------------------------------------------
        // Display LDPC test vector configuration info
        ldpc_decode_test_vec&              tv  = *ptv;
        const ldpc_decode_test_vec_config& tv_cfg = tv.config();
        tv.print_config();

        //------------------------------------------------------------------
        // Allocate an output buffer for decoded bits
        tensor_device tDecode(CUPHY_BIT,
                              tv.config().K, //MAX_DECODED_CODE_BLOCK_BIT_SIZE,
                              tv.config().num_cw,
                              cuphy::tensor_flags::align_coalesce);
        //------------------------------------------------------------------
        // Create a tensor descriptor for soft outputs
        tensor_device tSoftOutputs;
        if(writeSoftOutputs)
        {
            tSoftOutputs = tensor_device(llrType,
                                         tv.config().K,
                                         tv.config().num_cw);
        }
        //------------------------------------------------------------------
        // Allocate buffers for early termination and iteration count
        // Use typed_tensor with pinned_alloc for CRC types (input to decoder)
        typedef cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tensor_uint32_p_t;
        typedef cuphy::typed_tensor<CUPHY_R_32I, cuphy::pinned_alloc> tensor_int32_p_t;

        tensor_uint32_p_t tCrcType(tv_cfg.num_cw);
        tensor_uint32_p_t tCrcResultHost(tv_cfg.num_cw);  // Host buffer for CRC results
        tensor_int32_p_t  tIter(tv_cfg.num_cw);           // Host buffer for iteration counts

        // Initialize CRC types for all codewords
        for(int i = 0; i < tv_cfg.num_cw; ++i)
        {
            tCrcType(i) = tv_cfg.crc_type;
        }

            // Device buffer for CRC output (will be copied back to host)
            unique_device_ptr<uint32_t> crcOutputDevice;
            if(enableEarlyTerm)
            {
                crcOutputDevice = make_unique_device<uint32_t>(tv.config().num_cw);
            }

        //printf("Decode: addr: %p, %s, size: %.1f kB\n\n",
        //       tDecode.addr(),
        //       tDecode.desc().get_info().to_string().c_str(),
        //       tDecode.desc().get_size_in_bytes() / 1024.0);
        //--------------------------------------------------------------
        // Create an LDPC decoder instance
        cuphy::LDPC_decoder dec(ctx);

        //--------------------------------------------------------------
        // Preconditions of the BG1 band decoders.
        //
        // These are checked here, in the example, because neither can be
        // checked by the library: can_decode_config() sees the configuration
        // descriptor, which says nothing about which decode interface the
        // caller will use and carries no indication of whether the input LLRs
        // are punctured. Only the harness knows both.
        {
            // The algorithms declare these themselves; ask rather than keep a
            // second copy here, which would go stale silently as algorithms are
            // added or change.
            uint32_t reqs = 0;
            if(CUPHY_STATUS_SUCCESS !=
               cuphyErrorCorrectionLDPCGetAlgoRequirements(dec.handle(), algoIndex, &reqs))
            {
                // Do not fail open on an explicitly named algorithm: skipping the
                // checks silently is exactly the trap they exist to prevent.
                if(0 != algoIndex)
                {
                    NVLOGE_FMT(NVLOG_PUSCH,
                               AERIAL_CUPHY_EVENT,
                               "ERROR: could not query the requirements of algorithm {}.",
                               algoIndex);
                    return 1;
                }
                reqs = 0;
            }
            const bool needs_tb_interface    = (0 != (reqs & CUPHY_LDPC_ALGO_REQUIRES_TB_INTERFACE));
            const bool needs_punctured_input = (0 != (reqs & CUPHY_LDPC_ALGO_REQUIRES_PUNCTURED_INPUT));
            {
                if(needs_tb_interface && !useTBInterface)
                {
                    NVLOGE_FMT(NVLOG_PUSCH,
                               AERIAL_CUPHY_EVENT,
                               "ERROR: algorithm {} is only implemented for the transport "
                               "block interface. Add the '-b' option.",
                               algoIndex);
                    return 1;
                }
                // With '-i' the LLRs come from the file and '-P' plays no part
                // in producing them, so the harness cannot know whether they
                // are punctured. It is still the same trap, so say so and let
                // the run continue -- refusing would block legitimately
                // punctured files, which is the normal case for '-i'.
                if(needs_punctured_input && !inputFilename.empty())
                {
                    NVLOGW_FMT(NVLOG_PUSCH,
                               "WARNING: algorithm {} assumes punctured input. The LLRs come "
                               "from '{}', so this cannot be checked here. If that file holds "
                               "UNPUNCTURED LLRs, the reported BER/BLER understates the decoder by "
                               "roughly the coding gain of columns V0/V1 (~1.24 dB at "
                               "BG1/p=4/Z=384), which the decoder never loads.",
                               algoIndex,
                               inputFilename);
                }
                // Only enforceable for generated input, where '-P' is what
                // produces the puncturing.
                if(needs_punctured_input && inputFilename.empty() && !puncture)
                {
                    NVLOGE_FMT(NVLOG_PUSCH,
                               AERIAL_CUPHY_EVENT,
                               "ERROR: algorithm {} assumes punctured input -- it never "
                               "loads the punctured columns V0/V1, so unpunctured LLRs would be "
                               "silently discarded (~1.24 dB of coding gain at BG1/p=4/Z=384). "
                               "Add the '-P' option to puncture the generated data.",
                               algoIndex);
                    return 1;
                }
            }
        }
        //--------------------------------------------------------------
        // Initialize an LDPC decode configuration. This is used for
        // both the tensor and transport block interfaces.
        uint32_t decode_flags = chooseHighThroughput ? CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT : 0;
        if(writeSoftOutputs)
        {
            decode_flags |= CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS;
        }
        if(enableEarlyTerm)
        {
            decode_flags |= CUPHY_LDPC_DECODE_EARLY_TERM;
        }
        if(writeIterCount)
        {
            decode_flags |= CUPHY_LDPC_DECODE_WRITE_ITER_COUNT;
        }
        if(etLatencyDebug)
        {
            decode_flags |= CUPHY_LDPC_DECODE_ET_LATENCY_DEBUG;
        }
        if(forceEtKernel)
        {
            decode_flags |= CUPHY_LDPC_DECODE_FORCE_ET_KERNEL;
        }
        cuphy::LDPC_decode_config dec_cfg(llrType,       // LLR type (fp16, fp32, ...)
                                          tv_cfg.mb,     // num parity nodes
                                          tv_cfg.Z,      // lifting size
                                          numIterations, // max num iterations
                                          clampValue,    // clamp value
                                          tv_cfg.Kb,     // info nodes
                                          minSumNorm,    // normalization value
                                          decode_flags,  // flags
                                          tv_cfg.BG,     // base graph
                                          algoIndex,     // algorithm index
                                          nullptr);      // workspace address
        //--------------------------------------------------------------
        // If no normalization value was provided, query the library for
        // an appropriate value.
        if(minSumNorm <= 0.0f)
        {
            dec.set_normalization(dec_cfg);
        }
        printf("Normalization                    = %f\n", dec_cfg.get_norm());
        printf("Number of iterations             = %i\n", numIterations);
        printf("LLR data type                    = %s\n",
               cuphyGetDataTypeString(llrType));
        // A '--graph' run prints elapsed time and throughput like any other,
        // and those numbers are not decoder performance. Say so in the run's
        // own output rather than only in --help: a misleading figure is quoted
        // from a captured log, which is where the warning has to be.
        if(useGraph)
        {
            printf("Decode path                      = CUDA graph via get_launch_config()\n");
            printf("*** VALIDATION MODE ('--graph'): timing and throughput below are NOT\n");
            printf("*** decoder performance. The node parameters are re-derived and\n");
            printf("*** re-installed every run, which the normal decode path does not do.\n");
        }
        printf("\n");
        //--------------------------------------------------------------
        // Initialize an LDPC decode descriptor structure. (This is only
        // used when the transport block interface is selected.)
        LDPC_decode_desc dec_desc(dec_cfg, CUPHY_LDPC_DECODE_DESC_MAX_TB);
        if(useTBInterface)
        {
                // Get pointers for early termination / iteration count (if enabled)
                uint32_t* pCrcType   = enableEarlyTerm ? &tCrcType(0) : nullptr;
                uint32_t* pCrcOutput = enableEarlyTerm ? crcOutputDevice.get() : nullptr;
                int32_t*  pIterOutput = writeIterCount ? &tIter(0) : nullptr;

            if(spreadTB)
            {
                // Spread the codewords out into multiple transport blocks,
                // with addresses that point back to the original input
                // tensor.
                const int         CW_PER_TB = (tv_cfg.num_cw + (CUPHY_LDPC_DECODE_DESC_MAX_TB - 1)) /
                                              CUPHY_LDPC_DECODE_DESC_MAX_TB;
                cuphy::tensor_ref tLLR(tv.LLR_desc(), tv.LLR_addr());
                int cwOffset = 0;
                for(int iCW = 0; iCW < tv_cfg.num_cw; iCW += CW_PER_TB)
                {
                    int cwEnd = std::min(iCW + CW_PER_TB, tv_cfg.num_cw);
                    int cwCount = cwEnd - iCW;
                    cuphy::index_group slice(cuphy::dim_all(),
                                             cuphy::index_range(iCW, cwEnd));
                    cuphy::tensor_ref  sLLR    = tLLR.subset(slice);
                    cuphy::tensor_ref  sDecode = tDecode.subset(slice);
                    //printf("start = %i, end = %i\n", slice.ranges()[1].start(), slice.ranges()[1].end());
                    if(writeSoftOutputs)
                    {
                        cuphy::tensor_ref  sSoftOutputs = tSoftOutputs.subset(slice);
                        if(enableEarlyTerm || writeIterCount)
                        {
                            dec_desc.add_tensor_as_tb(sLLR.desc(),         sLLR.addr(),
                                                      sDecode.desc(),      sDecode.addr(),
                                                      sSoftOutputs.desc(), sSoftOutputs.addr(),
                                                      pCrcType ? pCrcType + cwOffset : nullptr,
                                                      pCrcOutput ? pCrcOutput + cwOffset : nullptr,
                                                      pIterOutput ? pIterOutput + cwOffset : nullptr);
                        }
                        else
                        {
                            dec_desc.add_tensor_as_tb(sLLR.desc(),         sLLR.addr(),
                                                      sDecode.desc(),      sDecode.addr(),
                                                      sSoftOutputs.desc(), sSoftOutputs.addr());
                        }
                    }
                    else
                    {
                        if(enableEarlyTerm || writeIterCount)
                        {
                            dec_desc.add_tensor_as_tb(sLLR.desc(),    sLLR.addr(),
                                                      sDecode.desc(), sDecode.addr(),
                                                      pCrcType ? pCrcType + cwOffset : nullptr,
                                                      pCrcOutput ? pCrcOutput + cwOffset : nullptr,
                                                      pIterOutput ? pIterOutput + cwOffset : nullptr);
                        }
                        else
                        {
                            dec_desc.add_tensor_as_tb(sLLR.desc(),    sLLR.addr(),
                                                      sDecode.desc(), sDecode.addr());
                        }
                    }
                    cwOffset += cwCount;
                }
            }
            else
            {
                if(writeSoftOutputs)
                {
                    if(enableEarlyTerm || writeIterCount)
                    {
                        dec_desc.add_tensor_as_tb(tv.LLR_desc(),
                                                  tv.LLR_addr(),
                                                  tDecode.desc(),
                                                  tDecode.addr(),
                                                  tSoftOutputs.desc(),
                                                  tSoftOutputs.addr(),
                                                  pCrcType,
                                                  pCrcOutput,
                                                  pIterOutput);
                    }
                    else
                    {
                        dec_desc.add_tensor_as_tb(tv.LLR_desc(),
                                                  tv.LLR_addr(),
                                                  tDecode.desc(),
                                                  tDecode.addr(),
                                                  tSoftOutputs.desc(),
                                                  tSoftOutputs.addr());
                    }
                }
                else
                {
                    if(enableEarlyTerm || writeIterCount)
                    {
                        dec_desc.add_tensor_as_tb(tv.LLR_desc(),
                                                  tv.LLR_addr(),
                                                  tDecode.desc(),
                                                  tDecode.addr(),
                                                  pCrcType,
                                                  pCrcOutput,
                                                  pIterOutput);
                    }
                    else
                    {

                        dec_desc.add_tensor_as_tb(tv.LLR_desc(),
                                                  tv.LLR_addr(),
                                                  tDecode.desc(),
                                                  tDecode.addr());
                    }
                }
            }
        }
        //--------------------------------------------------------------
        // Initialize an LDPC decode tensor params structure. (This is
        // only used when the tensor-based decoder interface is selected.)
        LDPC_decode_tensor_params dec_tensor(dec_cfg,                 // LDPC configuration
                                             tDecode.desc().handle(), // output descriptor
                                             tDecode.addr(),          // output address
                                             tv.LLR_desc().handle(),  // LLR descriptor
                                             tv.LLR_addr(),           // LLR address
                                             writeSoftOutputs ? tSoftOutputs.desc().handle() : nullptr, // Soft output descriptor (optional)
                                             writeSoftOutputs ? tSoftOutputs.addr() : nullptr);         // Soft output address (optional)
        //--------------------------------------------------------------
        // The graph that '--graph' decodes through. Built once, outside the
        // loop, because production builds it once too; only the node's
        // parameters are refreshed per run.
        std::unique_ptr<LDPC_decode_graph> dec_graph;
        if(useGraph)
        {
            dec_graph.reset(new LDPC_decode_graph());
        }
        //--------------------------------------------------------------
        // Decoder execution loop
        LDPC_decode_error_stats  error_stats;
        LDPC_decode_timing_stats timing_stats;
        do
        {
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Generate test vector data
            tv.generate();
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Warmup run
            if(!skipWarmup)
            {
                if(useGraph)
                {
                    dec_graph->decode(dec, dec_desc);
                }
                else if(useTBInterface)
                {
                    dec.decode(dec_desc);
                }
                else
                {
                    dec.decode(dec_tensor);
                }
            }
            cudaDeviceSynchronize();
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Timed run
            cuphy::event_timer tmr;

            tmr.record_begin();
            for(unsigned int uRun = 0; uRun < numRuns; ++uRun)
            {
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Decode
                if(useGraph)
                {
                    dec_graph->decode(dec, dec_desc);
                }
                else if(useTBInterface)
                {
                    dec.decode(dec_desc);
                }
                else
                {
                    dec.decode(dec_tensor);
                }
            }
            tmr.record_end();
            tmr.synchronize();
            timing_stats.update(tmr.elapsed_time_ms(), numRuns, tv_cfg.B * tv_cfg.num_cw);
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Compare decoder output to source bits
            if(!skipCompareOutput)
            {
                error_stats.update(tv.src_bits(), tDecode);
            }
        } while((error_stats.block_count()       < max_block_cnt)     &&
                (error_stats.block_error_count() < min_block_err_cnt));
        //--------------------------------------------------------------
        // Optional: export to HDF5
        if(!outputFilename.empty())
        {
            if((outputFilename.length() < 3) ||
               (0 != strcmp(outputFilename.c_str() +  outputFilename.length() - 3, ".h5")))
            {
                outputFilename.append(".h5");
            }
            hdf5hpp::hdf5_file f = hdf5hpp::hdf5_file::create(outputFilename.c_str());
            tv.export_hdf5(f);
            // Also write out the soft output data. We will convert to FP32
            // for convenience in reading.
            if(writeSoftOutputs)
            {
                cuphy::tensor_device tSoftOutput_f32(CUPHY_R_32F, tSoftOutputs.layout());
                cuphy::tensor_convert(tSoftOutput_f32, tSoftOutputs);
                cuphy::write_HDF5_dataset(f, tSoftOutput_f32, "outputLLR");
            }
        }
        //--------------------------------------------------------------
        // Display aggregated timing and error statistics
        printf("Average (%li runs) elapsed time in usec = %.1f, throughput = %.2f Gbps\n",
               timing_stats.num_runs(),
               timing_stats.average_time_usec(),
               timing_stats.throughput());

            //--------------------------------------------------------------
            // Copy CRC results back to host and calculate secondary BLER
            if(enableEarlyTerm && useTBInterface && crcOutputDevice)
            {
                cudaMemcpy(&tCrcResultHost(0),
                           crcOutputDevice.get(),
                           tv_cfg.num_cw * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost);

                // Calculate secondary BLER from CRC results
                uint32_t crc_block_error_count = 0;
                for(int i = 0; i < tv_cfg.num_cw; ++i)
                {
                    if(tCrcResultHost(i) != 0)
                    {
                        ++crc_block_error_count;
                    }
                }
                float crc_bler = static_cast<float>(crc_block_error_count) / tv_cfg.num_cw;
                printf("CRC-based block error rate (BLER) = (%u / %u) = %.5e\n",
                       crc_block_error_count,
                       tv_cfg.num_cw,
                       crc_bler);
            }

            //--------------------------------------------------------------
            // Display iteration count histogram if requested (only with -y)
            if(writeIterCount && useTBInterface)
            {
                std::vector<int32_t> iter_histogram(numIterations + 1, 0);

                int32_t max_count = std::numeric_limits<int32_t>::min();
                // Iteration count output was set up to have all iteration counts in
                // a contiguous buffer.
                for(int i = 0; i < tv_cfg.num_cw; ++i)
                {
                    if ((tIter(i) < 0) ||(tIter(i) > numIterations))
                    {
                        fprintf(stderr,
                                "ERROR: Invalid number of iterations for codeword %d: %d\n",
                                i,
                                tIter(i));
                        continue;
                    }
                    ++iter_histogram[tIter(i)];
                    max_count = std::max(max_count, iter_histogram[tIter(i)]);
                }
                if(max_count > 0)
                {
                    printf("Iteration count histogram:\n");
                    const float MAX_PRINT_SYMBOLS = 80.0f;
                    // Make it so that the highest frequency iteration count prints 80 characters.
                    const float SYMBOLS_PER_ITERATION = MAX_PRINT_SYMBOLS / max_count;
                    for(size_t i = 0; i < iter_histogram.size(); ++i)
                    {
                        int32_t count = iter_histogram[i];
                        const int NUM_CHARS = static_cast<int>(std::round(count * SYMBOLS_PER_ITERATION));
                        printf("%3lu: [%5d] ", i, count);
                        for(int j = 0; j < NUM_CHARS; ++j)
                        {
                            printf("*");
                        }
                        printf("\n");
                    }
                }
            }

        if(!skipCompareOutput)
        {
            printf("bit error count = %lu, bit error rate (BER) = (%lu / %lu) = %.5e, block error rate (BLER) = (%u / %u) = %.5e\n",
                   error_stats.bit_error_count(),
                   error_stats.bit_error_count(),
                   error_stats.bit_count(),
                   error_stats.BER(),
                   error_stats.block_error_count(),
                   error_stats.block_count(),
                   error_stats.BLER());
        }
    }

    catch(const ldpc_invalid_crc_config& e)
    {
        printf("WARNING: %s\n", e.what());
        returnValue = 0;
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    return returnValue;
}
