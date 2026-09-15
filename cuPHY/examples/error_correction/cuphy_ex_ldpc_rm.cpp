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

#include "CLI/CLI.hpp"
#include "cuphy.h"
#include "cuphy.hpp"
#include "ldpc_interm_check.hpp"
#include "ldpc/ldpc_api.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr int DEFAULT_MAX_BATCH_TBS = 1000;

struct pusch_input
{
    int         num_prb{15};
    int         num_layers{1};
    int         mcs{27};
    int         mcs_table{2};
    int         rv{0};
    float       snr_db{25.0f};
    int         num_tbs{1000};
    int         max_iter{10};
    int         decode_runs{1};
    int         algo{40};
    float       norm{0.0f};
    float       clamp{32.0f};
    uint64_t    seed{0};
    int         max_batch_tbs{DEFAULT_MAX_BATCH_TBS};
    int         num_dmrs{2};
    int         cdm_no_data{2};
    bool        skip_warmup{false};
    bool        reuse_tb{false};
    std::string input_filename;
    std::string fptype{"auto"};
    bool        enable_ET{false};
    bool        write_iter_count{false};
};

struct iter_histogram_stats
{
    std::vector<int32_t> histogram;
    int32_t              max_count{0};
    int32_t              min_iter{std::numeric_limits<int32_t>::max()};
    int32_t              max_observed_iter{0};
    int64_t              sum_iter{0};
    int64_t              count{0};
};

const char* llr_type_name(cuphyDataType_t type)
{
    switch(type)
    {
        case CUPHY_R_16F: return "fp16";
        case CUPHY_R_32F: return "fp32";
        default: return "unknown";
    }
}

// The LLR type comes from --fptype alone; the decoder rejects a type it cannot handle.
cuphyDataType_t resolve_llr_type(const std::string& fptype)
{
    if(fptype.empty() || fptype == "auto")
    {
        return CUPHY_R_16F;
    }
    return cuphy_ex_ldpc_rm::llr_type_from_string(fptype);
}

void print_case(const pusch_input& input, const cuphy_ex_ldpc_rm::pusch_ldpc_case& c)
{
    const double effective_tb_code_rate = static_cast<double>(c.tb_size + c.tb_crc_len) / c.G;
    const double effective_ldpc_source_rate =
        static_cast<double>(c.tb_size + c.tb_crc_len + c.C * c.cb_crc_len) / c.G;

    std::printf("*********************************************************************\n");
    std::printf("PUSCH LDPC-RM Configuration\n");
    std::printf("*********************************************************************\n");
    std::printf("numPRB                           = %d\n", input.num_prb);
    std::printf("numLayer                         = %d\n", input.num_layers);
    std::printf("MCS                              = %d\n", input.mcs);
    std::printf("MCS table                        = %d\n", input.mcs_table);
    std::printf("RV                               = %d\n", input.rv);
    std::printf("SNR dB                           = %.2f\n", input.snr_db);
    std::printf("Number of TBs                    = %d\n", input.num_tbs);
    std::printf("Max batch TBs                    = %d\n", input.max_batch_tbs);
    std::printf("Reuse one TB                     = %s\n", input.reuse_tb ? "true" : "false");
    std::printf("LDPC algo                        = %d\n", input.algo);
    const cuphyDataType_t llr_type = resolve_llr_type(input.fptype);
    std::printf("LLR type                         = %s\n", llr_type_name(llr_type));
    std::printf("max iterations                   = %d\n", input.max_iter);
    std::printf("decode runs                      = %d\n", input.decode_runs);
    std::printf("TB size A                        = %d\n", c.tb_size);
    std::printf("TB CRC bits                      = %d\n", c.tb_crc_len);
    std::printf("CB CRC bits                      = %d\n", c.cb_crc_len);
    std::printf("BG                               = %d\n", c.bg);
    std::printf("Kb                               = %d\n", c.Kb);
    std::printf("Z                                = %d\n", c.Z);
    std::printf("p                                = %d\n", c.p);
    std::printf("F                                = %d\n", c.F);
    std::printf("K'                               = %d\n", c.K_prime);
    std::printf("K                                = %d\n", c.K);
    std::printf("Ncb                              = %d\n", c.Ncb);
    std::printf("k0                               = %d\n", c.k0);
    std::printf("G per TB                         = %d\n", c.G);
    std::printf("C                                = %d\n", c.C);
    std::printf("Modulation                       = %s\n", c.mod.c_str());
    std::printf("target code rate                 = %.4f\n", c.code_rate);
    std::printf("effective TB code rate           = %.4f  ((A + TB CRC bits) / G)\n", effective_tb_code_rate);
    std::printf("effective aggregate CB code rate = %.4f  ((A + TB CRC bits + C*CB CRC bits) / G)\n",
                effective_ldpc_source_rate);
    std::printf("E per CB                         =");
    for(int e : c.E)
    {
        std::printf(" %d", e);
    }
    std::printf("\n");
}

// Cheap, high-quality 64-bit mixer (deterministic, well-scrambled).
uint64_t splitmix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Distinct-but-reproducible seed per (batch, stream): mixes base_seed with the batch and
// stream ids so each batch's codewords/noise are independent yet re-runnable.
uint64_t batch_seed(uint64_t base_seed, int batch_idx, uint64_t stream_id)
{
    const uint64_t batch = static_cast<uint64_t>(batch_idx) + 1;
    return splitmix64(base_seed ^ (batch * 0x9e3779b97f4a7c15ULL) ^ ((stream_id + 1) * 0xbf58476d1ce4e5b9ULL));
}

void accumulate_stats(cuphy_ex_ldpc_rm::error_stats&       total,
                      const cuphy_ex_ldpc_rm::error_stats& batch)
{
    total.bit_errors += batch.bit_errors;
    total.bit_count += batch.bit_count;
    total.cb_errors += batch.cb_errors;
    total.cb_count += batch.cb_count;
    total.tb_errors += batch.tb_errors;
    total.tb_count += batch.tb_count;
}

void accumulate_iter_histogram(iter_histogram_stats&                                      stats,
                               const cuphy::typed_tensor<CUPHY_R_32I, cuphy::pinned_alloc>& tIter,
                               int                                                        total_cbs,
                               int                                                        max_iter)
{
    if(stats.histogram.empty())
    {
        stats.histogram.assign(max_iter + 1, 0);
    }
    const int32_t* iter_data = tIter.addr();
    for(int i = 0; i < total_cbs; ++i)
    {
        const int32_t iter = iter_data[i];
        if(iter >= 0 && iter <= max_iter)
        {
            ++stats.histogram[iter];
            stats.max_count = std::max(stats.max_count, stats.histogram[iter]);
            stats.min_iter = std::min(stats.min_iter, iter);
            stats.max_observed_iter = std::max(stats.max_observed_iter, iter);
            stats.sum_iter += iter;
            ++stats.count;
        }
    }
}

void print_iter_histogram(const iter_histogram_stats& stats)
{
    if(stats.max_count <= 0 || stats.count <= 0)
    {
        return;
    }
    std::printf("Iteration count min/mean/max      = %d / %.2f / %d\n",
                stats.min_iter,
                static_cast<double>(stats.sum_iter) / stats.count,
                stats.max_observed_iter);
    std::printf("Iteration count histogram:\n");
    const float max_print_symbols = 80.0f;
    const float symbols_per_iter = max_print_symbols / stats.max_count;
    for(size_t i = 0; i < stats.histogram.size(); ++i)
    {
        const int32_t count = stats.histogram[i];
        const int num_chars = static_cast<int>(std::round(count * symbols_per_iter));
        std::printf("%3lu: [%5d] ", i, count);
        for(int j = 0; j < num_chars; ++j)
        {
            std::printf("*");
        }
        std::printf("\n");
    }
}

} // namespace

int main(int argc, char* argv[])
{
    int return_value = 0;
    try
    {
        pusch_input input;
        bool llr_check = false;
        std::string llr_diff_arg;
        float llr_p99err_rms_threshold = LLR_P99ERR_RMS_THRESHOLD_DEFAULT;
        float llr_snr_threshold       = LLR_SNR_THRESHOLD_DEFAULT;

        CLI::App app{"Standalone no-UCI PUSCH LDPC RM AWGN test driver"};
        app.footer(
            "Examples:\n"
            "  # per-iteration APP check vs the TV's MATLAB golden:\n"
            "  cuphy_ex_ldpc_rm -i tv.h5 --llr_check -a 55 -n 20\n"
            "  # per-iteration APP DUT-vs-DUT on a TV (algoA vs algoB):\n"
            "  cuphy_ex_ldpc_rm -i tv.h5 --llr_diff 40,55 -n 20\n"
            "  # TB BLER / timing on self-generated PUSCH AWGN (500 decode runs):\n"
            "  cuphy_ex_ldpc_rm --n-prb 151 --nl 1 -m 27 --mcs-table 2 -S 24.6 -a 55 -n 10 -r 500");
        app.add_option("-i,--input", input.input_filename,
            "Input HDF5 TV for --llr_check / --llr_diff.")
            ->check(CLI::ExistingFile);
        app.add_option("--n-prb", input.num_prb, "Number of allocated PRBs")
            ->check(CLI::PositiveNumber);
        app.add_option("--nl", input.num_layers, "Number of PUSCH layers")
            ->check(CLI::PositiveNumber);
        app.add_option("-m,--mcs", input.mcs, "MCS index")
            ->check(CLI::Range(0, 31));
        app.add_option("--mcs-table", input.mcs_table, "MCS table index")
            ->check(CLI::Range(1, 3));
        app.add_option("--rv", input.rv, "Redundancy version")
            ->check(CLI::Range(0, 3));
        app.add_option("-S,--snr", input.snr_db, "AWGN SNR in dB");
        app.add_option("-w,--num-tbs", input.num_tbs, "Number of transport blocks")
            ->check(CLI::PositiveNumber);
        app.add_option("--dmrs-syms", input.num_dmrs, "Number of DMRS symbols (type-1 DMRS)")
            ->check(CLI::Range(1, 2))
            ->capture_default_str();
        app.add_option("--cdm-nodata", input.cdm_no_data, "Number of DMRS CDM groups without data")
            ->check(CLI::Range(1, 2))
            ->capture_default_str();
        app.add_option("--batch-tbs", input.max_batch_tbs, "Maximum number of resident transport blocks per batch")
            ->check(CLI::PositiveNumber);
        app.add_option("-n,--max-iter", input.max_iter, "Maximum LDPC iterations")
            ->check(CLI::PositiveNumber);
        app.add_option("-r,--decode-runs", input.decode_runs, "Number of times to decode the same generated batch")
            ->check(CLI::PositiveNumber);
        app.add_option("-a,--algo", input.algo, "LDPC decoder algorithm index");
        app.add_option("--fptype", input.fptype, "LLR data type override: auto, fp16, fp32");
        app.add_option("--norm", input.norm, "Normalization factor. If <= 0, use decoder default");
        app.add_option("-C,--clamp", input.clamp, "LLR clamp value")
            ->check(CLI::PositiveNumber);
        app.add_option("--seed", input.seed, "Random seed");
        app.add_flag("-k,--skip-warmup", input.skip_warmup, "Skip warmup decode before timing");
        app.add_flag("--reuse-tb", input.reuse_tb, "Reuse one generated TB/codeword for all trials; each trial uses independent AWGN");
        app.add_flag("--llr_check", llr_check,
            "Per-iteration APP check: requires -i <tv.h5>. Runs decoder with interm_results, "
            "writes ldpc_dut.h5, then compares DUT vs REF /APP_history.");
        app.add_option("--llr_diff", llr_diff_arg,
            "DUT-vs-DUT per-iteration APP check: --llr_diff <algoA>,<algoB>. "
            "Requires -i <tv.h5>. Decodes the same TV with both algos and compares APP_history dumps.");
        app.add_option("--p99err_rms_threshold", llr_p99err_rms_threshold,
            "Per-iteration pass ceiling on p99err_rms = pctile99|ref-dut| / rms_ref "
            "for --llr_check / --llr_diff (pass if p99err_rms <= threshold).")
            ->capture_default_str();
        app.add_option("--snr_threshold", llr_snr_threshold,
            "Per-iteration SNR(dB) pass threshold for --llr_check / --llr_diff "
            "(SNR = 10*log10(||ref||^2 / ||ref-dut||^2)).")
            ->capture_default_str();
        app.add_flag("-x,--et", input.enable_ET, "Enable CRC-based LDPC ET");
        app.add_flag("-y,--iter-count", input.write_iter_count, "Write/print per-CB LDPC iteration counts");

        CLI11_PARSE(app, argc, argv);

        int llr_diff_algo_a = -1;
        int llr_diff_algo_b = -1;
        if(!llr_diff_arg.empty())
        {
            if(std::sscanf(llr_diff_arg.c_str(), "%d,%d", &llr_diff_algo_a, &llr_diff_algo_b) != 2)
            {
                std::fprintf(stderr, "Error: --llr_diff expects <algoA>,<algoB> (e.g. 40,56). Got: %s\n",
                             llr_diff_arg.c_str());
                return 1;
            }
        }

        const cuphyDataType_t llr_type = resolve_llr_type(input.fptype);
        if(llr_type != CUPHY_R_16F && llr_type != CUPHY_R_32F)
        {
            throw std::runtime_error("unsupported LLR type");
        }

        if(llr_check)
        {
            if(input.input_filename.empty())
            {
                std::fprintf(stderr, "Error: --llr_check requires -i <tv.h5>\n");
                return 1;
            }
            try
            {
                const bool pass = run_llr_check(input.input_filename, llr_type, input.algo,
                                                input.max_iter, llr_p99err_rms_threshold, llr_snr_threshold);
                std::printf("LLR_CHECK %s\n", pass ? "PASS" : "FAIL");
                return pass ? 0 : 1;
            }
            catch(...)
            {
                std::printf("LLR_CHECK FAIL\n");
                throw;
            }
        }

        if(!llr_diff_arg.empty())
        {
            if(input.input_filename.empty())
            {
                std::fprintf(stderr, "Error: --llr_diff requires -i <tv.h5>\n");
                return 1;
            }
            try
            {
                const bool pass = run_llr_diff(input.input_filename, llr_type, llr_diff_algo_a, llr_diff_algo_b,
                                               input.max_iter, llr_p99err_rms_threshold, llr_snr_threshold);
                std::printf("LLR_DIFF %s\n", pass ? "PASS" : "FAIL");
                return pass ? 0 : 1;
            }
            catch(...)
            {
                std::printf("LLR_DIFF FAIL\n");
                throw;
            }
        }

        const cuphy_ex_ldpc_rm::pusch_ldpc_case c =
            cuphy_ex_ldpc_rm::derive_pusch_ldpc_case(input.num_prb,
                                                     input.num_layers,
                                                     input.mcs,
                                                     input.mcs_table,
                                                     input.rv,
                                                     input.num_dmrs,
                                                     input.cdm_no_data);
        print_case(input, c);

        const int encode_len = c.Z * ((c.bg == 1) ? CUPHY_LDPC_MAX_BG1_VAR_NODES : CUPHY_LDPC_MAX_BG2_VAR_NODES);
        const int batch_tbs_cap = std::min(input.max_batch_tbs, input.num_tbs);
        const int num_batches = (input.num_tbs + batch_tbs_cap - 1) / batch_tbs_cap;
        std::printf("Number of batches                = %d\n", num_batches);

        cuphy::context ctx;
        uint32_t decode_flags = 0;
        cuphy::LDPC_decode_config dec_cfg(llr_type,
                                          c.p,
                                          c.Z,
                                          input.max_iter,
                                          input.clamp,
                                          c.Kb,
                                          input.norm,
                                          decode_flags,
                                          c.bg,
                                          input.algo,
                                          nullptr);
        cuphy::LDPC_decoder dec(ctx);
        if(input.norm <= 0.0f)
        {
            dec.set_normalization(dec_cfg);
        }
        std::printf("Normalization                    = %f\n", dec_cfg.get_norm());

        const float noise_var = std::pow(10.0f, input.snr_db / -10.0f);
        const float noise_component_stddev = std::sqrt(noise_var / 2.0f);
        const cuComplex mean = make_cuFloatComplex(0.0f, 0.0f);
        const cuComplex stddev = make_cuFloatComplex(noise_component_stddev, noise_component_stddev);
        const int symbols_per_tb = c.G / c.Qm;
        const bool multi_batch = num_batches > 1;

        using tensor_uint32_p_t = cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc>;
        using tensor_int32_p_t = cuphy::typed_tensor<CUPHY_R_32I, cuphy::pinned_alloc>;
        uint32_t crc_cb_errors = 0;
        iter_histogram_stats total_iter_stats;
        std::vector<uint8_t> reuse_cb_bits;
        if(input.reuse_tb)
        {
            cuphy_ex_ldpc_rm::build_random_cb_inputs(c, 1, input.seed, reuse_cb_bits);
        }

        cuphy_ex_ldpc_rm::error_stats stats;
        double elapsed_ms = 0.0;
        for(int batch_idx = 0; batch_idx < num_batches; ++batch_idx)
        {
            const int batch_num_tbs = std::min(batch_tbs_cap, input.num_tbs - batch_idx * batch_tbs_cap);
            const int source_num_tbs = input.reuse_tb ? 1 : batch_num_tbs;
            const int source_total_cbs = source_num_tbs * c.C;
            const int batch_total_cbs = batch_num_tbs * c.C;
            const int64_t source_rm_bits_64 = static_cast<int64_t>(source_num_tbs) * c.G;
            const int64_t batch_rm_bits_64 = static_cast<int64_t>(batch_num_tbs) * c.G;
            if(source_rm_bits_64 > std::numeric_limits<int>::max() ||
               batch_rm_bits_64 > std::numeric_limits<int>::max())
            {
                throw std::runtime_error("--batch-tbs is too large for int-sized tensor dimensions");
            }
            const int source_rm_bits = static_cast<int>(source_rm_bits_64);
            const int batch_rm_bits = static_cast<int>(batch_rm_bits_64);
            if(batch_rm_bits % c.Qm != 0)
            {
                throw std::runtime_error("internal error: batch RM bits is not a Qm multiple");
            }

            std::vector<uint8_t> batch_cb_bits;
            const std::vector<uint8_t>* source_cb_bits = &reuse_cb_bits;
            if(!input.reuse_tb)
            {
                const uint64_t data_seed = multi_batch ? batch_seed(input.seed, batch_idx, 0) : input.seed;
                cuphy_ex_ldpc_rm::build_random_cb_inputs(c, batch_num_tbs, data_seed, batch_cb_bits);
                source_cb_bits = &batch_cb_bits;
            }

            cuphy::typed_tensor<CUPHY_R_8U, cuphy::pinned_alloc> hSrcU8(c.K, source_total_cbs);
            for(int cb = 0; cb < source_total_cbs; ++cb)
            {
                const uint8_t* src = source_cb_bits->data() + static_cast<size_t>(cb) * c.K;
                for(int k = 0; k < c.K; ++k)
                {
                    hSrcU8(k, cb) = src[k];
                }
            }

            cuphy::tensor_device tSrcU8(CUPHY_R_8U, c.K, source_total_cbs);
            tSrcU8.convert(hSrcU8);
            cuphy::tensor_device tSrcBits(CUPHY_BIT, c.K, source_total_cbs);
            cuphy::tensor_convert(tSrcBits, tSrcU8);

            cuphy::tensor_device tEncodeBits(CUPHY_BIT, encode_len, source_total_cbs);
            cuphy::ldpc_encode(tEncodeBits,
                               tSrcBits,
                               c.bg,
                               c.Z,
                               false,
                               0,
                               input.rv);
            cuphy::tensor_device tEncodeU8(CUPHY_R_8U, encode_len, source_total_cbs);
            cuphy::tensor_convert(tEncodeU8, tEncodeBits);

            cuphy::tensor_device tRmBitsU8(CUPHY_R_8U, source_rm_bits);
            cuphy_ex_ldpc_rm::launch_pusch_tx_rate_match(c, tEncodeU8, tRmBitsU8, source_num_tbs);
            cudaStreamSynchronize(0);
            if(cudaGetLastError() != cudaSuccess)
            {
                throw std::runtime_error("PUSCH TX RM kernel failed");
            }

            cuphy::tensor_device tRmBits(CUPHY_BIT, source_rm_bits);
            cuphy::tensor_convert(tRmBits, tRmBitsU8);

            const int source_symbols = source_rm_bits / c.Qm;
            const int batch_symbols = batch_rm_bits / c.Qm;
            cuphy::tensor_device tSourceSymbols(CUPHY_C_16F, source_symbols);
            cuphy::modulate_symbol(tSourceSymbols, tRmBits, c.Qm);
            cuphy::tensor_device* tSymbols = &tSourceSymbols;
            std::unique_ptr<cuphy::tensor_device> tRepeatedSymbols;
            if(input.reuse_tb && batch_num_tbs > source_num_tbs)
            {
                tRepeatedSymbols = std::make_unique<cuphy::tensor_device>(CUPHY_C_16F, batch_symbols);
                cuphy_ex_ldpc_rm::launch_repeat_modulated_symbols(tSourceSymbols,
                                                                  *tRepeatedSymbols,
                                                                  symbols_per_tb,
                                                                  source_num_tbs,
                                                                  batch_num_tbs);
                cudaStreamSynchronize(0);
                if(cudaGetLastError() != cudaSuccess)
                {
                    throw std::runtime_error("symbol repeat kernel failed");
                }
                tSymbols = tRepeatedSymbols.get();
            }

            const uint64_t noise_seed = multi_batch ? batch_seed(input.seed, batch_idx, 1) : input.seed + 1;
            cuphy::rng rng_gen(noise_seed);
            cuphy::tensor_device tNoise(CUPHY_C_16F, batch_symbols);
            rng_gen.normal(tNoise, mean, stddev);
            cuphy::tensor_device tSymbolsPlusNoise(CUPHY_C_16F, batch_symbols);
            cuphy::tensor_sum(tSymbolsPlusNoise, *tSymbols, tNoise);

            cuphy::tensor_device tRmLLR(llr_type, batch_rm_bits);
            ctx.demodulate_symbol(tRmLLR, tSymbolsPlusNoise, c.Qm, noise_var);

            cuphy::tensor_device tLLR(llr_type,
                                      c.llr_len,
                                      batch_total_cbs,
                                      cuphy::tensor_flags::align_coalesce);
            if(llr_type == CUPHY_R_32F)
            {
                cuphy_ex_ldpc_rm::launch_pusch_rx_derate_match_fp32(c, tRmLLR, tLLR, batch_num_tbs, input.clamp);
            }
            else
            {
                cuphy_ex_ldpc_rm::launch_pusch_rx_derate_match_fp16(c, tRmLLR, tLLR, batch_num_tbs, input.clamp);
            }
            cudaStreamSynchronize(0);
            if(cudaGetLastError() != cudaSuccess)
            {
                throw std::runtime_error("PUSCH RX deRM kernel failed");
            }

            cuphy::tensor_device tDecode(CUPHY_BIT,
                                         c.K,
                                         batch_total_cbs,
                                         cuphy::tensor_flags::align_coalesce);

            cuphy::LDPC_decode_desc dec_desc(dec_cfg, CUPHY_LDPC_DECODE_DESC_MAX_TB);

            tensor_uint32_p_t tCrcType(batch_total_cbs);
            tensor_int32_p_t tIter(batch_total_cbs);
            cuphy::unique_device_ptr<uint32_t> crcOutputDevice;
            for(int cb = 0; cb < batch_total_cbs; ++cb)
            {
                tCrcType(cb) = cuphy_ex_ldpc_rm::cb_crc_type(c);
                tIter(cb) = 0;
            }
            if(input.enable_ET)
            {
                crcOutputDevice = cuphy::make_unique_device<uint32_t>(batch_total_cbs);
            }
            if(input.enable_ET || input.write_iter_count)
            {
                dec_desc.add_tensor_as_tb(tLLR.desc(),
                                          tLLR.addr(),
                                          tDecode.desc(),
                                          tDecode.addr(),
                                          input.enable_ET ? &tCrcType(0) : nullptr,
                                          input.enable_ET ? crcOutputDevice.get() : nullptr,
                                          input.write_iter_count ? &tIter(0) : nullptr);
            }
            else
            {
                dec_desc.add_tensor_as_tb(tLLR.desc(),
                                          tLLR.addr(),
                                          tDecode.desc(),
                                          tDecode.addr());
            }

            if(!input.skip_warmup)
            {
                dec.decode(dec_desc);
                cudaDeviceSynchronize();
            }

            cuphy::event_timer tmr;
            tmr.record_begin();
            for(int run = 0; run < input.decode_runs; ++run)
            {
                dec.decode(dec_desc);
            }
            tmr.record_end();
            tmr.synchronize();
            elapsed_ms += tmr.elapsed_time_ms();

            const cuphy_ex_ldpc_rm::error_stats batch_stats =
                cuphy_ex_ldpc_rm::compare_decoded_bits(c, batch_num_tbs, tDecode, *source_cb_bits, input.reuse_tb);
            accumulate_stats(stats, batch_stats);

            if(input.enable_ET && crcOutputDevice)
            {
                tensor_uint32_p_t tCrcResult(batch_total_cbs);
                cudaMemcpy(&tCrcResult(0),
                           crcOutputDevice.get(),
                           batch_total_cbs * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost);
                for(int cb = 0; cb < batch_total_cbs; ++cb)
                {
                    if(tCrcResult(cb) != 0)
                    {
                        ++crc_cb_errors;
                    }
                }
            }
            if(input.write_iter_count)
            {
                accumulate_iter_histogram(total_iter_stats, tIter, batch_total_cbs, input.max_iter);
            }
        }

        const double avg_us = elapsed_ms * 1000.0 / input.decode_runs;
        const double throughput_gbps =
            (static_cast<double>(c.tb_size) * input.num_tbs * input.decode_runs) /
            (elapsed_ms / 1000.0) / 1.0e9;

        std::printf("Average (%d runs) elapsed time in usec = %.1f, throughput = %.2f Gbps\n",
                    input.decode_runs,
                    avg_us,
                    throughput_gbps);
        std::printf("info-bit BER                    = (%lu / %lu) = %.5e\n",
                    stats.bit_errors,
                    stats.bit_count,
                    static_cast<double>(stats.bit_errors) / stats.bit_count);
        std::printf("info-bit CB BLER                = (%u / %u) = %.5e\n",
                    stats.cb_errors,
                    stats.cb_count,
                    static_cast<double>(stats.cb_errors) / stats.cb_count);
        std::printf("info-bit TB BLER                = (%u / %u) = %.5e\n",
                    stats.tb_errors,
                    stats.tb_count,
                    static_cast<double>(stats.tb_errors) / stats.tb_count);

        if(input.enable_ET)
        {
            std::printf("CRC-based CB BLER               = (%u / %u) = %.5e\n",
                        crc_cb_errors,
                        stats.cb_count,
                        static_cast<double>(crc_cb_errors) / stats.cb_count);
        }
        if(input.write_iter_count)
        {
            print_iter_histogram(total_iter_stats);
        }
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return_value = 1;
    }
    catch(...)
    {
        std::fprintf(stderr, "UNKNOWN EXCEPTION\n");
        return_value = 2;
    }
    return return_value;
}
