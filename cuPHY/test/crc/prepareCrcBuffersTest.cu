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

using namespace cuphy_i;

#define MAX_TBS 10

struct TestParams {
    int num_TBs;
    int TB_sizes[MAX_TBS]; // size in bytes; first num_TBs elements will be valid
};


uint32_t naive_brev(uint32_t value) {

   uint32_t ret_value = 0;
   for (int i = 0; i < 32; i++) {
       ret_value |= (((value >> i) & 0x1) << (31 - i));
   }
   return ret_value;
}


void init_h_input_on_CPU(std::vector<PdschPerTbParams>& h_pdsch_params, int num_TBs, std::vector<uint8_t>& h_input) {
   srand(time(NULL));

   for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
       uint32_t tb_size    = h_pdsch_params[TB_id].tbSize;
       uint32_t tb_offset  = h_pdsch_params[TB_id].tbStartOffset; //might not be divisible by sizeof(uint32_t)

       for (int byte = 0; byte < tb_size; byte++) {
           h_input[tb_offset + byte] = rand() % 256;
           //NVLOGC_FMT(NVLOG_PDSCH, "Initializing TB {} at offset {} byte {} with {:x}", TB_id, tb_offset, byte, h_input[tb_offset + byte]);
       }
   }
}

void prepare_crc_buffers_on_CPU(std::vector<PdschPerTbParams>& h_pdsch_params, int num_TBs, std::vector<uint8_t>& h_input, std::vector<uint32_t>& CPU_ref_output, int max_TB_size) {

   for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
       uint32_t tb_size    = h_pdsch_params[TB_id].tbSize;
       uint32_t tb_offset  = h_pdsch_params[TB_id].tbStartOffset; //might not be divisible by sizeof(uint32_t)

       int TB_start      = div_round_up<uint32_t>((TB_id == 0) ? 0 : h_pdsch_params[TB_id-1].cumulativeTbSizePadding, sizeof(uint32_t));
       int size_to_check = div_round_up<uint32_t>(h_pdsch_params[TB_id].cumulativeTbSizePadding - TB_start, sizeof(uint32_t)); // includes TB size and padding

       const uint8_t* TB_addr   = &h_input[tb_offset];//(d_inputOrigTBs == nullptr) ? TB_params->tbStartAddr : ((uint8_t*)d_inputOrigTBs + tb_offset);

       for (int i = 0; i < size_to_check; i++) {
           uint32_t value = 0;
           if(i < (tb_size >> 2))
           {
               const uint8_t* ptr = TB_addr + (i << 2);
               for(int byte_id = 0; byte_id < 4; byte_id++)
               {
                   value |= (ptr[byte_id] << (3 - byte_id) * 8);
               }
               CPU_ref_output[TB_start + i] = naive_brev(value);

           } else if(i < div_round_up<uint32_t>(tb_size, 4)) {
               const uint8_t* ptr = TB_addr + (i << 2);
               for(int byte_id = 0; byte_id < (tb_size & 0x3U); byte_id++)
               {
                   value |= (ptr[byte_id] << (3 - byte_id) * 8);
               }
               CPU_ref_output[TB_start + i] = naive_brev(value);
           }
           else if(i < size_to_check) {
               CPU_ref_output[TB_start + i] = 0;
           }
       }
   }
}


int ref_check(std::vector<PdschPerTbParams>& h_pdsch_params, int num_TBs, std::vector<uint32_t>& CPU_ref_output, std::vector<uint32_t>& GPU_output) {

#if 0
    // Definition of cumulativeTbSizePadding of TB i. It is the cumulativeTbSizePadding of TB (i-1), if it exists, and the TB size and padding bytes for TB i.
    // padding_bytes[i] is computed as the extra bytes needed to ensure that TB size and extra padding is evenly divisible by 4 (i.e., sizeof(uint32_t)) and
    // also includes extra space for at least 3 more bytes (CRC).
    //padding_bytes[i] = div_round_up<uint32_t>(tb_size[i], sizeof(uint32_t)) * sizeof(uint32_t) - tb_size[i];
    //padding_bytes[i] += (padding_bytes[i] <= 2) ? sizeof(uint32_t) : 0;

    // Reminder tbSize is always in B.
    // Modulo 4 will be 0, 1, 2, or 3
    // TB size % 4 = 0: padding bytes is 0 ; adding extra 4B
    // TB size % 4 = 1: padding bytes is 3 ; no extra
    // TB size % 4 = 2: padding bytes is 2 ; adding extra 4B, so total 6B
    // TB size % 4 = 3: padding bytes is 1 ; adding extra 4B, so total 5B

#endif

    int gpu_mismatched_bytes = 0;
    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
       int TB_start      = div_round_up<uint32_t>((TB_id == 0) ? 0 : h_pdsch_params[TB_id-1].cumulativeTbSizePadding, sizeof(uint32_t));
       int size_to_check = div_round_up<uint32_t>(h_pdsch_params[TB_id].cumulativeTbSizePadding - TB_start, sizeof(uint32_t)); // includes TB size and padding
       for (int i = 0; i < size_to_check; i++) {
           if (CPU_ref_output[TB_start + i] != GPU_output[TB_start + i]) {
               gpu_mismatched_bytes += 1;
               NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Mismatching TB {}, uint32_t element {} (buffer pos {}): GPU computed {:x} vs. ref {}",
                          TB_id, i, TB_start + i, GPU_output[TB_start + i], CPU_ref_output[TB_start + i]);
           }
       }
    }

    return gpu_mismatched_bytes;
}

/**
 * @brief: Run testcase specified by test_params.
 * @param[in] test_params: run CPU impl., GPU impl. and compare results
 * @param[in, out] gpu_mismatch: number of mismatched bits between GPU and CPU impl.
 */
void test_prepare_crc_buffers_config(TestParams & test_params, int & gpu_mismatch) {

    std::vector<PdschPerTbParams> h_pdsch_params;
    std::vector<int>  padding_bytes;

    cudaStream_t strm = 0;
    uint8_t* h_crc_prep_desc;

    // Populate the relevant fields of PdschPerTbParams
    int num_TBs = test_params.num_TBs;
    h_pdsch_params.resize(num_TBs);
    padding_bytes.resize(num_TBs);
    memset(h_pdsch_params.data(), 0, num_TBs * sizeof(PdschPerTbParams));
    //NVLOGC_FMT(NVLOG_PDSCH, "test_params.num_TBs {}", num_TBs);
    uint32_t running_TB_start_offset = 0;
    uint32_t max_tb_size_bytes = 0;
    for (int i = 0; i < num_TBs; i++) {
        //NVLOGC_FMT(NVLOG_PDSCH, "test_params.TB_sizes[{}] = {} B", i, test_params.TB_sizes[i]);
        // Only set fields used in prepare_crc_buffers kernel: tbSize, tbStartOffset, testModel, cumulativeTbSizePadding
        h_pdsch_params[i].tbSize = test_params.TB_sizes[i];
        h_pdsch_params[i].tbStartOffset = running_TB_start_offset;
        h_pdsch_params[i].testModel = 0; //TODO could extend to support testing mode too

        uint32_t tb_size =  h_pdsch_params[i].tbSize;
        padding_bytes[i] = div_round_up<uint32_t>(tb_size, sizeof(uint32_t)) * sizeof(uint32_t) - tb_size;
        padding_bytes[i] += (padding_bytes[i] <= 2) ? sizeof(uint32_t) : 0;

        h_pdsch_params[i].cumulativeTbSizePadding = ((i == 0) ? 0 : h_pdsch_params[i-1].cumulativeTbSizePadding) +  tb_size + padding_bytes[i];

        running_TB_start_offset += tb_size;
        max_tb_size_bytes = std::max(max_tb_size_bytes,  tb_size);
    }

    // Allocate launch config struct.
    std::unique_ptr<cuphyPrepareCrcEncodeLaunchConfig> prep_hndl = std::make_unique<cuphyPrepareCrcEncodeLaunchConfig>();

    // Allocate descriptor
    uint8_t desc_async_copy = 1; // Copy descriptor to the GPU during setup
    size_t desc_size=0, alloc_size=0;
    cuphyStatus_t status = cuphyPrepareCrcEncodeGetDescrInfo(&desc_size, &alloc_size);
    if (status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "cuphyPrepareCrcEncodeGetDescrInfo error {}", status);
    }

    unique_device_ptr<uint8_t> d_crc_prep_desc = make_unique_device<uint8_t>(desc_size);
    CUDA_CHECK(cudaHostAlloc((void**)&h_crc_prep_desc, desc_size, cudaHostAllocDefault));

    // Allocate on GPU
    // Perform H2D copies (if any)
    unique_device_ptr<PdschPerTbParams> d_tbPrmsArray = make_unique_device<PdschPerTbParams>(num_TBs);
    CUDA_CHECK(cudaMemcpyAsync(d_tbPrmsArray.get(), h_pdsch_params.data(), sizeof(PdschPerTbParams) * num_TBs, cudaMemcpyHostToDevice, strm));

    // Input and output
    cuphy::tensor_ref cell_group_crc_d_in_tensor_ref;

    int max_cells = 1;
    int max_CBs_per_TB =  MAX_N_CBS_PER_TB_SUPPORTED;
    int max_K_per_CB = CUPHY_LDPC_BG1_INFO_NODES * CUPHY_LDPC_MAX_LIFTING_SIZE;
    int per_cell_max_TBs = PDSCH_MAX_UES_PER_CELL;
    int max_per_cell_code_blocks_bytes = per_cell_max_TBs * max_CBs_per_TB * div_round_up<uint32_t>(max_K_per_CB, 8);
    unique_device_ptr<uint8_t> d_prepare_crc_input_buffer =  make_unique_device<uint8_t>(max_cells * max_per_cell_code_blocks_bytes);

    int  max_per_cell_crc_workspace_elements = 2 * div_round_up<uint32_t>(max_per_cell_code_blocks_bytes, sizeof(uint32_t)); //double as temp fix.

    unique_device_ptr<uint32_t> d_input_after_prep = make_unique_device<uint32_t>(max_cells * max_per_cell_crc_workspace_elements);
    std::vector<uint32_t> h_input_after_prep(max_cells * max_per_cell_crc_workspace_elements);

    // Allocate initial input buffer; memset it all to 0xff and then initialize TB contents with random data
    std::vector<uint8_t> h_input(max_cells * max_per_cell_crc_workspace_elements * sizeof(uint32_t));
    init_h_input_on_CPU(h_pdsch_params, num_TBs, h_input);

    // H2D copy
    //CUDA_CHECK(cudaMemcpyAsync(d_prepare_crc_input_buffer.get(), h_input.data(), max_cells * max_per_cell_code_blocks_bytes, cudaMemcpyHostToDevice, strm));
    int copy_min_bytes_needed = h_pdsch_params[num_TBs-1].tbStartOffset + h_pdsch_params[num_TBs-1].tbSize;
    CUDA_CHECK(cudaMemcpyAsync(d_prepare_crc_input_buffer.get(), h_input.data(), copy_min_bytes_needed, cudaMemcpyHostToDevice, strm));
    
    // Poison output buffer with non-zero pattern to detect uninitialized reads
    CU_CHECK(cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(d_input_after_prep.get()), 0xA5, max_cells * max_per_cell_crc_workspace_elements * sizeof(uint32_t), strm));

    status = cuphySetupPrepareCRCEncode(
            prep_hndl.get(),
            (uint32_t*) d_prepare_crc_input_buffer.get(),
            (uint32_t*) d_input_after_prep.get(), //input after prep.
            nullptr, // for TBs of cells in testing mode
            d_tbPrmsArray.get(),
            num_TBs, // total number of TBs across all cells
            0, // unused
            max_tb_size_bytes, //max TB size in bytes across all cells
            h_crc_prep_desc,
            d_crc_prep_desc.get(),
            desc_async_copy,
            strm);
    if (status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "cuphySetupPrepareCRCEncode returned {}", status);
        gpu_mismatch = 1; // forcing mismatch
        return;
    }

    // Launch prepare_crc_buffers kernel

    CUresult status_prepare = launch_kernel(prep_hndl.get()->m_kernelNodeParams, strm);
    if(status_prepare != CUDA_SUCCESS)
    {
       throw std::runtime_error("Prepare CRC Encode error(s)");
    }

    // D2H copies of output
    CUDA_CHECK(cudaMemcpyAsync(h_input_after_prep.data(), d_input_after_prep.get(), h_pdsch_params[num_TBs-1].cumulativeTbSizePadding, cudaMemcpyDeviceToHost, strm));
    CUDA_CHECK(cudaStreamSynchronize(strm));

    // CPU ref. work
    std::vector<uint32_t> CPU_ref_output;
    CPU_ref_output.resize(max_cells * max_per_cell_crc_workspace_elements);
    prepare_crc_buffers_on_CPU(h_pdsch_params, num_TBs, h_input, CPU_ref_output, max_tb_size_bytes);

    // Ref. comparison that sets gpu_mismatch accordingly
    gpu_mismatch = ref_check(h_pdsch_params, num_TBs, CPU_ref_output, h_input_after_prep);

    CUDA_CHECK(cudaFreeHost(h_crc_prep_desc));

}


class PrepareCrcBuffersTest: public ::testing::TestWithParam<TestParams> {
public:
  void basicTest() {
      params = ::testing::TestWithParam<TestParams>::GetParam();
      test_prepare_crc_buffers_config(params, gpu_mismatch);
  }

  void SetUp() override {basicTest(); }
  void TearDown() override {
      gpu_mismatch = -1;
  }

protected:
  TestParams params;
  int gpu_mismatch = -1;
};

TEST_P(PrepareCrcBuffersTest, TB_CONFIGS) {
    //EXPECT_EQ(0, gpu_mismatch); // continues after the failure
    ASSERT_TRUE(gpu_mismatch == 0); // doesn't continue
}

//Idea is I'll have a bunch of unique TB_configs and each test case will specify a number of them.
// TB size -> divisible by 4, %4 = {1, 2, or 3}, divisible by 32 too or not in each case

#define TB_SIZE_MOD_4_IS_0 2688 // TB size % 4 == 0 and also TB size % (4*32) == 0); 32 is warp size
#define TB_SIZE_MOD_4_IS_1 2689 // TB size % 4 == 1
#define TB_SIZE_MOD_4_IS_2 2690 // TB size % 4 == 2
#define TB_SIZE_MOD_4_IS_3 2691 // TB size % 4 == 3
#define TB_SIZE_MOD_4_IS_0_BUT_NOT_MOD_64 33962// TB size % 4 == 0 but TB size % (4 * 32) !+ 0; 32 is warp size
//FIXME does anything < 32 make sense?
#define TB_SIZE_32 32
#define TB_SIZE_8   8
#define TB_SIZE_31 31
#define TB_SIZE_30 30
#define TB_SIZE_29 29
#define TB_SIZE_1024 1024 // TB size % 4 == 0, multiple of 1024
#define TB_SIZE_2048 2048 // TB size % 4 == 0, multiple of 1024
#define TB_SIZE_3072 3072 // TB size % 4 == 0, multiple of 1024
#define TB_SIZE_3070 3070
#define TB_SIZE_3071 3071

const std::vector<TestParams> TB_configs = {

   // TB start addr and TB size (bytes), for all TBs, are both divisible by 4 (i.e., sizeof(uint32_t)
   {2, {TB_SIZE_MOD_4_IS_0, TB_SIZE_MOD_4_IS_0}}, //0

   // TB start addr is in both cases divible by 4, but TB size (bytes) isn't. Have special cases for every TB_size %4 != 0
   {2, {TB_SIZE_MOD_4_IS_0, TB_SIZE_MOD_4_IS_1}}, //1-3
   {2, {TB_SIZE_MOD_4_IS_0, TB_SIZE_MOD_4_IS_2}},
   {2, {TB_SIZE_MOD_4_IS_0, TB_SIZE_MOD_4_IS_3}},

   // TB start addr is not divisible by 4, but TB size is. Have special cases for every TB_addr %4 != 0
   {2, {TB_SIZE_MOD_4_IS_1, TB_SIZE_MOD_4_IS_0}}, //4-6
   {2, {TB_SIZE_MOD_4_IS_2, TB_SIZE_MOD_4_IS_0}},
   {2, {TB_SIZE_MOD_4_IS_3, TB_SIZE_MOD_4_IS_0}},

   // neither TB start addr. nor TB size divisible by 4. Have special cases for every TB_size %4 != 0 not divisible by 4 and for every TB_addr %4 != 0 combination
   {2, {TB_SIZE_MOD_4_IS_1, TB_SIZE_MOD_4_IS_1}}, //7-9
   {2, {TB_SIZE_MOD_4_IS_1, TB_SIZE_MOD_4_IS_2}},
   {2, {TB_SIZE_MOD_4_IS_1, TB_SIZE_MOD_4_IS_3}},

   {2, {TB_SIZE_MOD_4_IS_2, TB_SIZE_MOD_4_IS_1}}, //10-12
   {2, {TB_SIZE_MOD_4_IS_2, TB_SIZE_MOD_4_IS_2}},
   {2, {TB_SIZE_MOD_4_IS_2, TB_SIZE_MOD_4_IS_3}},

   {2, {TB_SIZE_MOD_4_IS_3, TB_SIZE_MOD_4_IS_1}}, //13-15
   {2, {TB_SIZE_MOD_4_IS_3, TB_SIZE_MOD_4_IS_2}},
   {2, {TB_SIZE_MOD_4_IS_3, TB_SIZE_MOD_4_IS_3}},
   //Config 4-8 and 10 had initcheck issues in orig config
#if 1
   {2, {TB_SIZE_MOD_4_IS_1, TB_SIZE_MOD_4_IS_2}}, //exposes initcheck issue without fix for 2nd TB
   {3, {TB_SIZE_MOD_4_IS_1, TB_SIZE_MOD_4_IS_1, TB_SIZE_MOD_4_IS_3/*, TB_SIZE_MOD_4_IS_2, TB_SIZE_MOD_4_IS_0_BUT_NOT_MOD_64, TB_SIZE_MOD_4_IS_3}*/}},
   {2, {TB_SIZE_MOD_4_IS_1, TB_SIZE_MOD_4_IS_1}},
   {2, {TB_SIZE_32, TB_SIZE_32}},

   {3, {TB_SIZE_MOD_4_IS_0_BUT_NOT_MOD_64, TB_SIZE_MOD_4_IS_2, TB_SIZE_MOD_4_IS_0}},
   {6, {TB_SIZE_MOD_4_IS_1, TB_SIZE_MOD_4_IS_1, TB_SIZE_MOD_4_IS_3, TB_SIZE_MOD_4_IS_2, TB_SIZE_MOD_4_IS_0_BUT_NOT_MOD_64, TB_SIZE_MOD_4_IS_3}},
   {6, {TB_SIZE_32, TB_SIZE_8, TB_SIZE_31, TB_SIZE_30, TB_SIZE_29, TB_SIZE_32}},

   // Mix large and small sizes
   {2, {TB_SIZE_3072, TB_SIZE_MOD_4_IS_1}},
   {2, {TB_SIZE_MOD_4_IS_3, TB_SIZE_2048}},
   {3, {TB_SIZE_1024, TB_SIZE_MOD_4_IS_2, TB_SIZE_3072}},
   
   // TB sizes at or near a 1024-byte boundary where padding bytes fall outside the grid.
   // The grid covers ceil(max_TB_size/4/256)*256 uint32_t elements = max_TB_size rounded
   // down to the nearest 1024B. Sizes 3072/3071/3070 all get tb_size_plus_padding=3076,
   // but the grid only covers up to 3072 bytes without the fix.
   {1, {TB_SIZE_3072}},
   {1, {TB_SIZE_3071}},
   {1, {TB_SIZE_3070}},
   {2, {TB_SIZE_3072, TB_SIZE_3071}}

#endif
};



INSTANTIATE_TEST_CASE_P(PrepareCrcBuffersTests, PrepareCrcBuffersTest,
                        ::testing::ValuesIn(TB_configs));

int main(int argc, char** argv) {

    cuphyNvlogFmtHelper nvlog_fmt("prepareCrcBuffersTest.log"); //needed if we want nvlog entries from tests
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    return result;
}
