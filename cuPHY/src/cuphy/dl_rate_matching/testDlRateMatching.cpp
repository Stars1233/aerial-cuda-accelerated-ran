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

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <sstream>
#include <vector>
#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_internal.h"
#include "dl_rate_matching.cuh"

using namespace std;
using namespace cuphy_i;

struct TestParams {
    int num_TBs;
    int num_layers;
    bool scrambling;
    bool layer_mapping;
    bool read_input_file;

    uint32_t rv; // redundancy version per TB; [0, 3]
    uint32_t Qm; // modulation order per TB: [2, 4, 6, 8]
    uint32_t bg; // base graph per TB; options are 1 or 2
    uint32_t Nl; // number of transmission layers per TB
    uint32_t num_CBs; // number of code blocks per TB
    uint32_t Zc; // lifting factor per TB

    uint32_t G; // number of rate matched bits available for TB transmission
    uint32_t F; // filler bits
    uint32_t cinit; // used to generate scrambling sequence; seed2 arg. of gold32

    uint32_t Nref; //used to determine Ncb if smaller than N
};

// Used to force Ncb be N instead of Nref in cuphySetTBParams. Reminder the min of the two is selected.
#define MAX_N_OVERRIDE_NREF (384* CUPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES)

/**
 * @brief Fill in layer map for all TBs. Assume all TBs map to the same number of layers.
 * param[in, out] kernel_params: config. parameters for all TBs to be processed in one kernel.
 */
void set_uniform_layer_mapping(std::vector<PdschPerTbParams> & kernel_params, std::vector<PdschDmrsParams> & h_dmrs_params) {

    for (int TB_id = 0; TB_id < kernel_params.size(); TB_id++) {
        h_dmrs_params[TB_id].n_scid = 0; // for convenience
        for (int layer_id = 0; layer_id < kernel_params[TB_id].Nl; layer_id++) {
            h_dmrs_params[TB_id].port_ids[layer_id] = TB_id * kernel_params[0].Nl + layer_id;
        }
    }
}


/**
 * @brief Generate scrambling_seq scrambling sequence. Generated per TB.
 * @param[in, out] scrambling_seq: generate entire scrambling seq
 * @param[in] c_init: initial value for scrambling sequence (based on RNTI: Radio Network Temporary Identifier,
 *                    cellId, and q.
 */
void generate_scrambling_sequence(vector<uint32_t> & scrambling_seq, int c_init) {
    int G = scrambling_seq.size(); // G: size, in bits, of a single Transport Block.
    const int Nc = 1600;
    std::vector<uint32_t> x1_seq(Nc + G);
    std::vector<uint32_t> x2_seq(Nc + G);

    // Compute c_init = RNTI * 2^15 + q * 2^14 + cellID
    // uint32_t c_init = (RNTI << 15)  + (q << 14) + 20;

    // Initialize x1[i] and x2[i] for i in [0, 30] (31-length gold sequence).
    // x1 and x2 are the 1st and 2nd polynomials for the 31-length gold sequence.
    for (int i = 0; i <= 30; i++) {
        x1_seq[i] = (i == 0) ? 1 : 0;
        x2_seq[i] = (c_init >> i) & 0x1ULL;
    }

    // Populate x1[i] and x2[i] for i in [31, NC + G)
    for (int j = 0; j < Nc + G - 31; j++) {
        x1_seq[j + 31] = (x1_seq[j + 3] + x1_seq[j]) & 0x1ULL;
        x2_seq[j + 31] = (x2_seq[j + 3] + x2_seq[j + 2] + x2_seq[j + 1] + x2_seq[j]) & 0x1ULL;
    }

    // Populate scrambling sequence
    for (int i = 0; i < G; i++) {
        scrambling_seq[i] = (x1_seq[i + Nc] + x2_seq[i + Nc]) & 0x1ULL;
    }
}

/**
 * @brief Generate output_sequence via bit selection on input_sequence (single CB).
 * @param[in] input_sequence: input sequence (single code block)
 * @param[in, out] output_sequence: bit selected output sequence; E bits wide.
 * @param[in] E: length of output bit sequence
 * @param[in] Ncb: circular buffer length
 * @paran[in] k0: starting position
 * @param[in] K: excludes 2*Zc punctured bits. For BG1 it's 20*Zc.
 * @param[in] F: number of filler bits
 */
void bit_selection(const unsigned char input_sequence[], unsigned char output_sequence[], int E, int Ncb, int k0, int K, int F) {

    int Kd = K - F;
    int output_bit_index = 0;
    int j = 0;
    while (output_bit_index < E) {
        int input_bit_index = (k0 + j) % Ncb;
        if ((input_bit_index < Kd) || (input_bit_index >= K)) { // Verify range is appropriate
            output_sequence[output_bit_index] = input_sequence[input_bit_index];
            output_bit_index += 1;
        }
        j += 1;
    }
}


/**
 * @brief Bit interleave input_sequence (single code block) based on square function.
 * @param[in] Er: number of bits of each of input_sequence and interleaved_sequence
 * @param[in] Qm: modulation order w/ Er % Qm == 0.
 * @param[in] input_sequence: input bit sequence w/ length Er bits.
 * @param[in, out] interleaved_sequence: output bit sequence, w/ length Er.
 */
void bit_interleaving(int Er, int Qm, unsigned char input_sequence[], unsigned char interleaved_sequence[]) {
    /*  Bit jth of ith E/Qm-side block of input maps to bit ith in jth Qm-wide block of output
        No interleaving if Qm is 1 or E. */
    int step = Er / Qm;
    for (int j = 0; j < step;  j++) {
        for (int i = 0; i < Qm; i++) {
            interleaved_sequence[i + j * Qm] = input_sequence[i* step + j];
        }
    }
}

/**
 * @brief Compute rate matching on CPU to use as reference across all TBs. If a reference output
 *        is provided in a file, this function compares the CPU computed data w/ the reference output
 *        and throws a runtime exception if false.
 * @param[in] num_TBs: number of transport blocks (TBs)
 * @param[in] ldpc_encoder_output: holds # code blocks (C) in transport block (TB). Each code block is N-bits wide.
 * @param[in] kernel_params: configuration parameters for all num_TBs TBs
 * @param[in] expected_output: to use as reference, if not empty.
 * @param[in, out] cpu_computed_output: CPU computed output when there's no layer_mapping
 * @param[in, out] layered_mapped_array: CPU computed output when layer mapping on;
 * @param[in] do_scrambling: scramble output sequence
 * @param[in] do_layer_mapping: do layer_mapping
 * @param[in,out] Er: num_TBs x num_CBs_per_TB array w/ rate matched length in bits
 * @param[in] num_layers: total number of layers for all num_TBs TBs
 */

void all_TBs_CPU_rate_matching(int num_TBs, const std::vector<unsigned char> & ldpc_encoder_output,
		               const std::vector<PdschPerTbParams> & kernel_params,
		               const std::vector<PdschDmrsParams> & h_dmrs_params,
                               const std::vector<unsigned char> & expected_output,
			       std::vector<unsigned char> & cpu_computed_output,
                               std::vector<unsigned char> layered_mapped_array[],
                               bool do_scrambling, bool do_layer_mapping,
                               std::vector<std::vector<uint32_t>> & Er, int num_layers) {

   int ldpc_output_TB_offset = 0;
   int expected_output_TB_offset = 0;
   uint32_t Emax = 0;
   bool updated_Emax;
   bool have_reference = !expected_output.empty(); // Have a reference output to compare CPU against.
   bool cpu_with_reference_match  = true;

   for (int TB_id = 0; TB_id < num_TBs; TB_id++) {

        // Compute rate matching length for all codeblocks in TB TB_id.
        compute_rate_matching_length(Er[TB_id].data(), kernel_params[TB_id].num_CBs, kernel_params[TB_id].Qm,
			             kernel_params[TB_id].Nl, kernel_params[TB_id].G, Emax, updated_Emax);

        // Generate scrambling sequence for entire TB.
        std::vector<uint32_t> scrambling_seq;
        if (do_scrambling) {
            scrambling_seq.resize(kernel_params[TB_id].G);
            generate_scrambling_sequence(scrambling_seq, kernel_params[TB_id].cinit);
        }

        // Do bit selection and interleaving for each code block.
        int k0 = compute_k0(kernel_params[TB_id].rv, kernel_params[TB_id].bg, kernel_params[TB_id].Ncb,
			    kernel_params[TB_id].Zc);

        int output_bit_offset = 0;
        int mismatch_cnt = 0;

        for (int code_block_id = 0; code_block_id < kernel_params[TB_id].num_CBs; code_block_id++) {
            uint32_t Er_CB = Er[TB_id][1] + ((code_block_id < Er[TB_id][0]) ? 0 : kernel_params[TB_id].Nl * kernel_params[TB_id].Qm);
            std::vector<unsigned char> bit_selected_seq(Er_CB);
            std::vector<unsigned char> rate_matched_output(Er_CB);

	    // Updated K arg as first 2*Zc bits are punctured and thus not included in LDPC encoder's output
            bit_selection(&ldpc_encoder_output[ldpc_output_TB_offset + code_block_id * kernel_params[TB_id].N],
			  bit_selected_seq.data(), Er_CB, kernel_params[TB_id].Ncb,
                          k0, kernel_params[TB_id].K - 2*kernel_params[TB_id].Zc, kernel_params[TB_id].F);
            bit_interleaving(Er_CB, kernel_params[TB_id].Qm, bit_selected_seq.data(), rate_matched_output.data());

            for (int i = 0; i < Er_CB; i++) {
                if (rate_matched_output[i] > 1) {
                    throw std::runtime_error("Invalid bit value for rate matched output sequence");
                }

                if (do_scrambling) {
                    rate_matched_output[i] = (rate_matched_output[i] + scrambling_seq[output_bit_offset + i]) & 1ULL;   // xor LS bit
                }

                if (have_reference && (!do_layer_mapping) && (expected_output[expected_output_TB_offset + output_bit_offset + i] != rate_matched_output[i])) {
                    // there is a reference output file to compare against; if layer mapping comparison is enabled, will compare later
                    mismatch_cnt += 1;
		    cpu_with_reference_match = false;
                }
                cpu_computed_output[expected_output_TB_offset + output_bit_offset + i] = rate_matched_output[i];
            }
            output_bit_offset += Er_CB;
        }
        if (mismatch_cnt != 0) { // mismatch check when there's no layer mapping
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "TB {}: compared {} bits w/ reference values", TB_id, output_bit_offset);
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "scrambling {} and mismatch cnt {}", do_scrambling, mismatch_cnt);
        }

        ldpc_output_TB_offset += (kernel_params[TB_id].num_CBs * kernel_params[TB_id].N);
        expected_output_TB_offset += kernel_params[TB_id].G;
    }

    if (do_layer_mapping) {

        uint32_t expected_rm_output_start = 0;
        for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
            for (int layer_id = 0; layer_id < kernel_params[TB_id].Nl; layer_id++) {
                int current_layer = h_dmrs_params[TB_id].port_ids[layer_id] + 8 * h_dmrs_params[TB_id].n_scid;
                int symbols_per_TB = kernel_params[TB_id].G / kernel_params[TB_id].Qm;
                int symbols_per_TB_per_layer = div_round_up<int>(symbols_per_TB, kernel_params[TB_id].Nl);

                for (int symbol_id = 0; symbol_id < symbols_per_TB_per_layer; symbol_id++) {
                    for (int bit_id = 0; bit_id < kernel_params[TB_id].Qm; bit_id++) {
                        uint32_t output_index = (kernel_params[TB_id].Nl * symbol_id * kernel_params[TB_id].Qm) + \
						layer_id * kernel_params[TB_id].Qm + bit_id;
                        if (output_index < kernel_params[TB_id].G) {
                            layered_mapped_array[current_layer].push_back(cpu_computed_output[expected_rm_output_start + output_index]);
                        }
                    }
                }
            }
            expected_rm_output_start += kernel_params[TB_id].G;
        }
	// Compare CPU computed layer-mapped output with reference from file (if available).
	if (have_reference) {
            int cnt = 0;
            for (int layer_id = 0; layer_id < num_layers; layer_id++) {
                int mismatch_cnt = 0;
                for (int i = 0; i < layered_mapped_array[layer_id].size(); i++) {
                    if (expected_output[cnt] != layered_mapped_array[layer_id][i]) {
                        mismatch_cnt += 1;
	                cpu_with_reference_match = false;
                    }
                    cnt += 1;
                }
                if (mismatch_cnt != 0) {
                    NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Layer {} mismatch_cnt {}", layer_id, mismatch_cnt);
                }
            }
	}
    }
    if (!cpu_with_reference_match) {
        throw std::runtime_error("CPU computed values do not match reference!");
    }
}

/**
 *  @brief Read elements from file and populate input to rate matching component.
 *        The file format uses an unsigned char per line to represent a single bit.
 *        The file includes -1 for filler bits; map that to a different unsigned
 *        char for now.
 *  @param[in, out] rm_input: input to rate matching unit.
 *  @param[in] input_filename: name of file with test vector
 */
void read_from_file(std::vector<unsigned char> & rm_input, string input_filename) {

    ifstream input_file;
    input_file.open(input_filename);
    if (!input_file) {
        string error_msg = "Error! Could not open file " + input_filename;
        throw std::runtime_error(error_msg);
    }
    string line;

    int cnts[3] = {0, 0, 0};
    int cnt = 0;
    while (getline(input_file, line) && (cnt < rm_input.size())) {
        stringstream str(line);
        float val;
        str >> val;
        if (val == 0.0f) {
            rm_input[cnt] = 0x0;
            cnts[0] += 1;
        } else if (val == 1.0f) {
            rm_input[cnt] = 0x1;
            cnts[1] += 1;
        } else {
            rm_input[cnt] = 0x2; // dummy; input file has -1 in place of filler bits
            cnts[2] += 1;
        }
        cnt++;
    }
    //std::cout << "Read " << cnt << " elements from file " << input_filename << "." << std::endl;
    //std::cout << "{0: " << cnts[0] << ", 1: " << cnts[1] << ", -1: " << cnts[2] << "}" << std::endl;
    input_file.close();
}

/**
 *  @brief Convert input test vector ldpc_encoder_output (unsigned char) to a packed h_rate_matching_input (uint32_t).
 *  @param[in, out] h_rate_matching_input: output array where all bits are packed in uint32_t elements.
 *                                         The first bit of encoder's output is mapped to the least significant bit
 *                                         of the first h_rate_matching_input element and so on.
 *  @params[in] ldpc_encoder_output:  test vector array w/ unsigned char elements; only least significant bit is used.
 */
void pre_processing(std::vector<uint32_t>& h_rate_matching_input, const std::vector<unsigned char>& ldpc_encoder_output) {

    for (int cnt = 0; cnt < h_rate_matching_input.size(); cnt += 1) {
        uint32_t tmp_val = 0;
        for (int bit_id = 0; bit_id < 32; bit_id++) {
            int index = cnt * 32 + bit_id;
            if (index < ldpc_encoder_output.size()) {
                tmp_val |= ((ldpc_encoder_output[index] & 0x1ULL) << bit_id);
            }
        }
        h_rate_matching_input[cnt] = tmp_val;
    }
}


/**
 *  @brief: Compare GPU output to expected CPU-generated output. Only applicable if layer mapping not enabled.
 *  @param[in] h_rate_matching_output: GPU-generated output; host buffer. Allocated as Cmax * Emax bits per TB.
 *  @param[in] expected_rm_output: CPU-generated output; used for comparison.
 *  @param[in] Emax: maximum number of Er bits across all TBs
 *  @param[in] Cmax: maximum number of code blocks across all TBs
 *  @param[in] expected_rm_output_elements: expected # rate matched bits
 *  @param[in] kernel_params: array of config. parameters for all TBs processed
 *  @param[in] Er: array of rate-matched bit length for all TBs processed
 *  @return number of mismatched bits between GPU and CPU
 */
int post_processing(const std::vector<uint32_t>& h_rate_matching_output, const std::vector<unsigned char> & expected_rm_output,
		    uint32_t Emax, uint32_t Cmax, uint32_t expected_rm_output_elements, const std::vector<PdschPerTbParams> & kernel_params,
                    const std::vector<std::vector<uint32_t>> & Er) {

    std::vector<unsigned char> gpu_computed_rm_output(expected_rm_output_elements);
    int gpu_mismatch = 0;
    int expected_index = 0;

    for (int cnt = 0; cnt < h_rate_matching_output.size(); cnt++) {
        uint32_t tmp_val = h_rate_matching_output[cnt];
        for (int bit_id = 0; bit_id < 32; bit_id++) {
            int index = cnt * 32 + bit_id;

            int TB_id = index / (Emax * Cmax);
            uint32_t intra_TB_index = index - TB_id * Emax * Cmax;

            int CB_id = intra_TB_index / Emax;
            int Er_bit = intra_TB_index % Emax;
            int Er_length = Er[TB_id][1] + ((CB_id < Er[TB_id][0]) ? 0 : kernel_params[TB_id].Nl * kernel_params[TB_id].Qm);
            if ((CB_id < kernel_params[TB_id].num_CBs) && (Er_bit < Er_length)) {

                gpu_computed_rm_output[expected_index] = (tmp_val >> bit_id) & 0x1ULL;
                if (gpu_computed_rm_output[expected_index] != expected_rm_output[expected_index]) {
                    if (gpu_mismatch == 0) {          
                        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "First mismatch @ {} TB_id {} CB_id {}, Er_bit {} : gpu {} vs. expected {}", expected_index, TB_id, CB_id, Er_bit, (uint32_t) gpu_computed_rm_output[expected_index], (uint32_t) expected_rm_output[expected_index]);
                    }
                    gpu_mismatch += 1;
                }
                expected_index += 1;
            }
        }
    }
    return gpu_mismatch;
}


/* @brief: Compare GPU output to expected CPU-generated output. Only applicable if layer mapping is enabled.
 * @param[in] h_output: GPU-generated output; host buffer. Allocated layer first.
 * @param[in] expected_output: CPU-generated output; used for comparison.
 * @param[in] Emax: maximum number of Er bits across all TBs
 * @param[in] Cmax: maximum number of code blocks across all TBs
 * @param[in] expected_rm_output_elements: expected # rate matched bits
 * @param[in] kernel_params: array of config. parameters for all TBs processed
 * @param[in] Er: array of rate-matched bit length for all TBs processed
 * @param[in] num_layers: total number of layers in h_output.
 * @return number of mismatched bits between GPU and CPU
 */
int layer_mapped_post_processing(const std::vector<uint32_t>& h_output, const std::vector<unsigned char> expected_output[],
		    uint32_t Emax, uint32_t Cmax, uint32_t expected_rm_output_elements, const std::vector<PdschPerTbParams> & kernel_params,
                    const std::vector<PdschDmrsParams> & h_dmrs_params,
                    const std::vector<std::vector<uint32_t>> & Er, int num_layers) {

    int num_TBs = kernel_params.size();

    std::vector<unsigned char> gpu_computed_rm_output(expected_rm_output_elements);
    int gpu_mismatch = 0;
    int expected_index = 0;
    int cpu_expected_index = 0;

    std::vector<int> layer_to_TBs(num_layers, -1);
    for (int i = 0; i < kernel_params.size(); i++) { // go over all TBs
        for (int layer_id = 0; layer_id < kernel_params[i].Nl; layer_id++) { // number of layers for TB
            int layer = h_dmrs_params[i].port_ids[layer_id] + 8 * h_dmrs_params[i].n_scid;
            if (layer_to_TBs[layer] != -1) {
                throw std::runtime_error("Currently we cannot have a layer have more than one TB.");
            }
            layer_to_TBs[layer] = i;
        }
    }
    uint32_t ref_check_cnt = 0;
    for (int cnt = 0; cnt < h_output.size(); cnt++) {
        uint32_t tmp_val = h_output[cnt];
        for (int bit_id = 0; bit_id < 32; bit_id++) {
            int index = cnt * 32 + bit_id;
            int layer_id = index / (Emax * Cmax * num_TBs);
            int intra_layer_index = index - layer_id * Emax * Cmax * num_TBs;

            int TB_id = intra_layer_index / (Emax * Cmax);
            int intra_TB_index = intra_layer_index - TB_id * Emax * Cmax;

            if (intra_layer_index == 0) {
                cpu_expected_index = 0;
            }
            int CB_id = intra_TB_index / Emax;
            int Er_layer_bit = intra_TB_index % Emax;
            if (layer_to_TBs[layer_id] != TB_id) { //TODO this does not support FDM vectors, i.e., it assumes only one TB mapped per layer
                TB_id = -1;
            }
            int Er_length = (TB_id == -1) ? 0 :
                            Er[TB_id][1] + ((CB_id < Er[TB_id][0]) ? 0 : kernel_params[TB_id].Nl * kernel_params[TB_id].Qm);

            if ((TB_id != -1) && (CB_id < kernel_params[TB_id].num_CBs) && (Er_layer_bit < (Er_length/kernel_params[TB_id].Nl))) {
                gpu_computed_rm_output[expected_index] = (tmp_val >> bit_id) & 0x1ULL;
                if (gpu_computed_rm_output[expected_index] != expected_output[layer_id][cpu_expected_index]) {
                    if (gpu_mismatch == 0) {
                        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "First mismatch @ {}, CPU expected index {} Layer id {} TB_id {} CB_id {}, Er_bit {}: gpu {} vs. expected {} intra layer index {}", expected_index, cpu_expected_index, layer_id, TB_id, CB_id, Er_layer_bit, (uint32_t) gpu_computed_rm_output[expected_index], (uint32_t) expected_output[layer_id][cpu_expected_index], intra_layer_index);
                    }
                    gpu_mismatch += 1;
                }
                expected_index += 1;
                cpu_expected_index += 1;
                ref_check_cnt += 1;
            }
        }
    }
    //NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, ""GPU- vs. CPU-computed mismatch (post layer-mapping) = {}", gpu_mismatch);
    //printf("ref check cnt %d\n", ref_check_cnt);
    return gpu_mismatch;
}

/**
 * @brief: Run testcase specified by test_params.
 * @param[in] test_params: run CPU impl., GPU impl. and compare results
 * @param[in, out] gpu_mismatch: number of mismatched bits between GPU and CPU impl.
 */
void test_rm_config(TestParams & test_params, int & gpu_mismatch) {

    bool GPU_run = true;
    int num_TBs = test_params.num_TBs;
    int num_layers = test_params.num_layers;
    bool scrambling = test_params.scrambling;
    bool layer_mapping = test_params.layer_mapping;

    string base_dir = "."; // update base dir as needed

    stringstream base_name;
    base_name << "tv_" << test_params.num_TBs << "_TBs_" << test_params.num_CBs << "_CBs_per_TB_" << test_params.Qm;
    base_name << "_Qm_" << test_params.Nl << "_Nl_" << test_params.rv << "_rv_" << test_params.bg;
    base_name << "_bg_" << test_params.Zc << "_Zc_" << test_params.G << "_G_" << test_params.F << "_F_";
    base_name << test_params.cinit << "_cinit.txt";

    string input_filename = base_dir + "/rm_in_" + base_name.str();
    //Output file for ref. comparison: {w/o scrambling, w/ scrambling}
    string expected_output_filename[2] = {base_dir + "/rm_out_" + base_name.str(),
                                 base_dir + "/rm_out_scram_" + base_name.str()};

    std::vector<PdschPerTbParams> kernel_params(num_TBs);
    std::vector<PdschDmrsParams>  h_dmrs_params(num_TBs);
    std::vector<std::vector<uint32_t>> Er_array(num_TBs);

    // Initialize config params for all transport blocks. Assumes identical hardcoded configs across  TBs for now.
    uint32_t ldpc_encoder_output_elements = 0;
    uint32_t expected_rm_output_elements = 0;
    for (int TB = 0; TB < num_TBs; TB++) {
        cuphyStatus_t status = cuphySetTBParams(&kernel_params[TB], test_params.rv, test_params.Qm, test_params.bg, test_params.Nl, test_params.num_CBs,
		      test_params.Zc, test_params.G, test_params.F, test_params.cinit, test_params.Nref); // different TBs have identical cinit for now
        if (status != CUPHY_STATUS_SUCCESS) {
            throw std::runtime_error("Invalid configuration parameter(s) in cuphySetTBParams function.");
        }

        Er_array[TB].resize(2);

	ldpc_encoder_output_elements += kernel_params[TB].num_CBs * kernel_params[TB].N;
	expected_rm_output_elements += kernel_params[TB].G;
    }
    set_uniform_layer_mapping(kernel_params, h_dmrs_params); // Update port_ids and scid in h_dmrs_params; used to compute layer_map
    srand(time(NULL));

    std::vector<unsigned char> ldpc_encoder_output(ldpc_encoder_output_elements); // input to rate matching unit; read from file or randomly generated
    std::vector<unsigned char> expected_rm_output; // expected output of rate matching unit; read from file or unavailable
    std::vector<unsigned char> cpu_computed_rm_output; // cpu computed rm_output when no layer_mapping
    std::vector<unsigned char> layered_mapped_array[num_layers]; // cpu computed output when there's layer mapping.

    if (test_params.read_input_file) { // read input from a file - multiple TBs
        read_from_file(ldpc_encoder_output, input_filename);
        expected_rm_output.resize(expected_rm_output_elements);
        read_from_file(expected_rm_output, expected_output_filename[scrambling]);
    } else { // randomly initialize.
        for (int j = 0; j < ldpc_encoder_output.size(); j++) { // TBs have randomly generated vals; each TB has different contents
            ldpc_encoder_output[j] = rand() % 2;
        }
    }
    cpu_computed_rm_output.resize(expected_rm_output_elements);

    // Do rate_matching on CPU.
    all_TBs_CPU_rate_matching(num_TBs, ldpc_encoder_output, kernel_params, h_dmrs_params,
                          expected_rm_output, cpu_computed_rm_output, layered_mapped_array, scrambling, layer_mapping,
                          Er_array, num_layers);

    if (GPU_run) {
        cudaStream_t strm = 0;

	uint32_t max_Emax = 22816; //TODO update as needed
	uint32_t Emax = 0;

        size_t allocated_workspace_size = cuphyDlRateMatchingWorkspaceSize(num_TBs);
        unique_device_ptr<uint32_t> config_workspace = make_unique_device<uint32_t>(div_round_up<uint32_t>(allocated_workspace_size, sizeof(uint32_t)));
        uint32_t* h_workspace = nullptr;
        uint8_t* h_rm_desc;
        CUDA_CHECK(cudaHostAlloc((void**)&h_workspace, (2 + 2) * num_TBs * sizeof(uint32_t), cudaHostAllocDefault));

        unique_device_ptr<PdschPerTbParams> d_tbPrmsArray = make_unique_device<PdschPerTbParams>(num_TBs);
        unique_device_ptr<PdschDmrsParams>  d_dmrs_params = make_unique_device<PdschDmrsParams>(num_TBs);
        CUDA_CHECK(cudaMemcpyAsync(d_tbPrmsArray.get(), kernel_params.data(), sizeof(PdschPerTbParams) * num_TBs, cudaMemcpyHostToDevice, strm));
        CUDA_CHECK(cudaMemcpyAsync(d_dmrs_params.get(), h_dmrs_params.data(), sizeof(PdschDmrsParams) * num_TBs, cudaMemcpyHostToDevice, strm));

        bool use_tensors = false;
	uint32_t N_max = 0, Cmax = 0;
	for (int i = 0; i < num_TBs; i++) {
	    N_max = std::max(N_max, kernel_params[i].N);
            Cmax = std::max(Cmax, kernel_params[i].num_CBs);
	}
        int rounded_Nmax = div_round_up<uint32_t>(N_max, 32);

	size_t input_elements = use_tensors ? rounded_Nmax * Cmax * num_TBs :  div_round_up<uint32_t>(ldpc_encoder_output.size(), 32);
	size_t max_output_elements = layer_mapping ? div_round_up<uint32_t>(num_TBs * max_Emax * Cmax * num_layers, 32) : div_round_up<uint32_t>(max_Emax * Cmax * num_TBs, 32);
        std::vector<uint32_t> h_rate_matching_input(input_elements);
        std::vector<uint32_t> h_rate_matching_output(max_output_elements);

        unique_device_ptr<uint32_t> d_rate_matching_output = make_unique_device<uint32_t>(max_output_elements);

	// Preprocessing: map ldpc_encoder_output (bit represented as unsigned char) to array of uint32_t (bit as bit)
        pre_processing(h_rate_matching_input, ldpc_encoder_output);

        unique_device_ptr<uint32_t> d_rate_matching_input = make_unique_device<uint32_t>(input_elements);
        CUDA_CHECK(cudaMemcpy(d_rate_matching_input.get(), h_rate_matching_input.data(), input_elements * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // Allocate launch config struct.
        std::unique_ptr<cuphyDlRateMatchingLaunchConfig> rm_hndl = make_unique<cuphyDlRateMatchingLaunchConfig>();

        // Allocate descriptors and setup rate matching component
        uint8_t desc_async_copy = 1; // Copy descriptor to the GPU during setup
        size_t desc_size=0, alloc_size=0;
        cuphyStatus_t status = cuphyDlRateMatchingGetDescrInfo(&desc_size, &alloc_size);
        if (status != CUPHY_STATUS_SUCCESS) {
            printf("cuphyModulationGetDescrInfo error %d\n", status);
        }
        unique_device_ptr<uint8_t> d_rm_desc = make_unique_device<uint8_t>(desc_size);
        CUDA_CHECK(cudaHostAlloc((void**)&h_rm_desc, desc_size, cudaHostAllocDefault));

        cuphyPdschStatusOut_t output_status; // populated but not used here

        status = cuphySetupDlRateMatching(rm_hndl.get(),
                                      &output_status,
                                      d_rate_matching_input.get(),
		                      d_rate_matching_output.get(),
                                      nullptr, /* no restructuring */
                                      nullptr, /* no modulation */
                                      nullptr, /* Xtf RE map */
                                      273,     /* max PRBs for Xtf RE maps; not used since no modulation */
                                      num_TBs,
                                      0,
                                      scrambling,
                                      layer_mapping,
                                      false, /* no modulation */
                                      0, /* no precoding for any of the TBs*/ 
                                      false,
                                      false,
                                      false,
                                      h_workspace,
                                      config_workspace.get(), // d_workspace - Explicit H2D copy as part of setup
                                      kernel_params.data(), // h_params
                                      d_tbPrmsArray.get(), // d_params - No H2D copy. FIXME make this configurable?
                                      d_dmrs_params.get(),
                                      nullptr,
                                      h_rm_desc,
                                      d_rm_desc.get(),
                                      desc_async_copy,
                                      strm);

        if (status != CUPHY_STATUS_SUCCESS) {
            throw std::runtime_error("Invalid argument(s) for cuphySetupDlRateMatching");
        }
        uint32_t* Er = h_workspace + 2 * num_TBs;
        for (int i = 0; i < num_TBs; i++) { //FIXME this was already computed in setup, could just copy back that value directly
            uint32_t Er_CB = Er[i*2 + 1] + (((kernel_params[i].num_CBs - 1) < Er[i*2 + 0]) ? 0 : kernel_params[i].Nl * kernel_params[i].Qm);
            Emax = std::max(Emax, Er_CB);
        }
        Emax = div_round_up<uint32_t>(Emax, 32) * 32; //Emax needs to be divisible by 32
        if (Emax > max_Emax) {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Emax {} but supported max Emax is {}", Emax, max_Emax);
            throw std::runtime_error("Emax exceeds max supported! Update max_Emax in this example.");
        }

        size_t output_elements = (layer_mapping) ? div_round_up<uint32_t>(num_TBs * num_layers * Cmax * Emax, 32) : div_round_up<uint32_t>(num_TBs * Cmax * Emax, 32);
        h_rate_matching_output.resize(output_elements);
        CUDA_CHECK(cudaDeviceSynchronize());

        const int num_iterations = 20;
        uint32_t time_kernel = 1;

        cuphy::event_timer cuphy_timer;

        cuphy_timer.record_begin();
        for (int iter = 0; iter < num_iterations; iter++) {

            // launch kernel on default stream; run for num_iterations and report the average time.
            CU_CHECK_EXCEPTION(launch_kernel_ex(rm_hndl.get()->m_kernelNodeParams[0], strm, true));
        }
        cuphy_timer.record_end();
        cuphy_timer.synchronize();

        float time1 = cuphy_timer.elapsed_time_ms() / num_iterations;

        if (time_kernel) {
            printf("DL Rate Matching Kernel: %.2f us (avg. over %d iterations)\n", time1 * 1000, num_iterations);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_rate_matching_output.data(), d_rate_matching_output.get(), output_elements * sizeof(uint32_t), cudaMemcpyDeviceToHost));

	// Expectation: GPU computed data will have been copied back to h_rate_matching_output
        if (!layer_mapping) { // Check non layer-mapped output
	    gpu_mismatch = post_processing(h_rate_matching_output, cpu_computed_rm_output, Emax, Cmax,
			                   expected_rm_output_elements, kernel_params, Er_array);
        } else { // Check layer-mapped output
           gpu_mismatch = layer_mapped_post_processing(h_rate_matching_output, layered_mapped_array, Emax, Cmax,
                                                       expected_rm_output_elements, kernel_params, h_dmrs_params, Er_array,
                                                       num_layers);
       }

       CUDA_CHECK(cudaFreeHost(h_workspace));
       CUDA_CHECK(cudaFreeHost(h_rm_desc));
    }
}

class RateMatchingTest: public ::testing::TestWithParam<TestParams> {
public:
  void basicTest() {
      params = ::testing::TestWithParam<TestParams>::GetParam();
      test_rm_config(params, gpu_mismatch);
  }

  void SetUp() override {basicTest(); }
  void TearDown() override {
      gpu_mismatch = -1;
  }

protected:
  TestParams params;
  int gpu_mismatch = -1;
};

TEST_P(RateMatchingTest, IDENTICAL_TB_CONFIGS) {
    //EXPECT_EQ(0, gpu_mismatch);
    ASSERT_TRUE(gpu_mismatch == 0);
}

const std::vector<TestParams> identical_TB_configs = {

#if 0 // FIXME Uncomment if input file available
   {16 /*num TBs */, 16 /*num layers (across all TBs)*/, true /*scrambling*/, true /*layer_mapping*/, true /*read_from_file*/,
    0 /*rv*/, 8 /*Qm*/, 1 /*bg*/, 1 /*Nl*/, 26 /*CBs*/, 384 /*Zc*/, 235872 /*G*/, 72 /*F*/, 9568296 /*cinit*/},
   {16, 16, false, false, true, 0, 8, 1, 1, 26, 384, 235872, 72, 9568296}, // No scrambling - No layer mapping
   {16, 16, true, false, true, 0, 8, 1, 1, 26, 384, 235872, 72, 9568296}, // only scrambling
   {16, 16, false, true, true, 0, 8, 1, 1, 26, 384, 235872, 72, 9568296},  // only layer mapping

   {7, 14, true, false, true, 0, 4, 1, 2, 18, 384, 235872, 0, 26705960}, // 7 TBs - 2 layers per TB - QAM-4; Layer-Mapping should be false for this config
   {8, 8, false, true, true, 0, 6, 1, 1, 13, 384, 117504, 72, 1507348}, // 8 TB - 1 layer per TB - QAM 6 - RNTI 46 & cellID 20
   {8, 8, true, true, true, 0, 6, 1, 1, 13, 384, 117504, 72, 1507348}, // same as above but w/ scrambling on
#endif

   /* Randomly generated vals, i.e., not reading inputs from a file. Identical config values across TBs. TB contents different.
      Cover all modulation orders, redundancy versions and base graph types. */
   {8, 8, true, true, false, 0, 6, 1, 1, 13, 384, 117504, 72, 1507348, MAX_N_OVERRIDE_NREF}, // rv = 0, BG=1
   {8, 8, true, true, false, 1, 6, 1, 1, 13, 384, 117504, 72, 1507348, MAX_N_OVERRIDE_NREF}, // rv = 1, BG=1
   {8, 8, true, true, false, 2, 6, 1, 1, 13, 384, 117504, 72, 1507348, MAX_N_OVERRIDE_NREF}, // rv = 2, BG=1
   {8, 8, true, true, false, 3, 6, 1, 1, 13, 384, 117504, 72, 1507348, MAX_N_OVERRIDE_NREF}, // rv = 3, BG=1, w/ wrap-around
   {8, 8, true, false, false, 0, 6, 1, 1, 13, 384, 117504, 72, 1507348, MAX_N_OVERRIDE_NREF}, // rv = 0, scrambling on, layer mapping off
   {8, 8, false, false, false, 0, 6, 1, 1, 13, 384, 117504, 72, 1507348, MAX_N_OVERRIDE_NREF}, // rv = 0, both scrambling and layer mapping are off
   {7, 14, true, false, false, 0, 4, 1, 2, 18, 384, 235872, 0, 26705960, MAX_N_OVERRIDE_NREF}, // 7 TBs - 2 layers per TB - QAM-4; Layer-Mapping should be false for this config

   /* Check all possible scenarios for rv > 0. Cover both base graph types. */
   {8, 8, true, true, false, 1 /* rv */, 2 /* Qm */, 2 /* BG */, 1, 4 /* CBs */, 352 /* Zc */, 72072 /* G */, 96 /* F */, 1507348, MAX_N_OVERRIDE_NREF}, // rv = 1, BG=2, wrap around
   {16, 16, true, true, false, 1 /* rv */, 8 /* Qm */, 1 /* BG */, 1, 26 /* CBs */, 384 /* Zc */, 235872 /* G */, 72 /* F */, 9568296, MAX_N_OVERRIDE_NREF}, // rv = 1, BG=1, F <= 3Zc so falls under case 0
   //TODO placeholder: is it possible to have rv=1, BG=1 and F > 3*Zc?
   {8, 8, true, true, false, 2 /* rv */, 2 /* Qm */, 2 /* BG */, 1, 4 /* CBs */, 352 /* Zc */, 72072 /* G */, 96 /* F */, 1507348, MAX_N_OVERRIDE_NREF}, // rv = 2, BG=2, wrap around
   {8, 8, true, true, false, 3 /* rv */, 2 /* Qm */, 2 /* BG */, 1, 4 /* CBs */, 352 /* Zc */, 72072 /* G */, 96 /* F */, 1507348, MAX_N_OVERRIDE_NREF}, // rv = 3, BG=2, wrap around

   /* Wrap-around for rv=0, i.e., low-code rate scenario */
   {1, 1, true, true, false, 0 /* rv */, 2 /* Qm */, 2 /* BG */, 1, 1 /* CBs */, 208 /* Zc */, 13104 /* G */, 40 /* F */, 0, MAX_N_OVERRIDE_NREF}, // rv = 0, BG=2, wrap around

   /* rv > 0 with k0 non divisible by 32 */
   {1, 1, true, true, false, 2 /* rv */, 2 /* Qm */, 2 /* BG */, 1, 1 /* CBs */, 11 /* Zc */, 360 /* G */, 46 /* F */, 0, MAX_N_OVERRIDE_NREF}, // rv =2, BG=2, (k0%32)!=0

   /* Ncb != N (it is Nref) to explore LBRM support. Nref specified as last arg. below.
    * This test-case also wraps around the Ncb buffer to ensure Ncb is considered in bit selection. */
   {1, 4, true, true, false, 3 /* rv */, 8 /* Qm */, 1 /* BG */, 4, 107 /* CBs */, 384 /* Zc */, 1362816 /* G */, 0 /* F */, 1146901, 17915 /* Nref */},
};

INSTANTIATE_TEST_CASE_P(RateMatchingTests, RateMatchingTest,
                        ::testing::ValuesIn(identical_TB_configs));


int main(int argc, char** argv) {

    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    return result;
}
