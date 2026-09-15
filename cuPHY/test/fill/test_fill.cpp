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
#include "cuphy.hpp"

namespace
{

} // namespace


////////////////////////////////////////////////////////////////////////
// Fill.Basic
TEST(Fill, Basic)
{
    typedef cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc> tensor_32f;
    // Allocate a tensor
    const std::array<int, 4> DIMS = {{32, 64, 16, 8}};
    //const std::array<int, 4> DIMS = {{16, 16, 1, 1}};
    tensor_32f               t(cuphy::tensor_layout(DIMS.size(), DIMS.data(), nullptr));
    const float              FILL_VALUE = 1.0f;
    // Initialize to zero
    CUDA_CHECK_EXCEPTION(cudaMemset(t.addr(), 0, t.desc().get_size_in_bytes()));
    // Issue the fill operation
    t.fill(FILL_VALUE);
    // Wait for results to complete
    cudaStreamSynchronize(0);
    // Compare values to the expected.
    for(int i3 = 0; i3 < DIMS[3]; ++i3)
    {
        for(int i2 = 0; i2 < DIMS[2]; ++i2)
        {
            for(int i1 = 0; i1 < DIMS[1]; ++i1)
            {
                for(int i0 = 0; i0 < DIMS[0]; ++i0)
                {
                    EXPECT_EQ(t(i0, i1, i2, i3), FILL_VALUE);
                }
            }
        }
    }
}

template <cuphyDataType_t TType>
void do_fill_index_group_test(const std::array<int, 2>& dims,
                              const cuphy::index_group& grp)
{
    typedef cuphy::typed_tensor<TType, cuphy::pinned_alloc> tensor_p;
    typedef cuphy::typed_tensor_ref<TType>                  tensor_ref_p;
    typedef typename cuphy::type_traits<TType>::type        value_t;
    //------------------------------------------------------------------
    // Allocate a tensor
    tensor_p                 t(cuphy::tensor_layout(dims.size(), dims.data(), nullptr));
    //printf("t: %s\n", t.desc().get_info().to_string().c_str());
    //------------------------------------------------------------------
    // Initialize tensor memory to zero
    CUDA_CHECK_EXCEPTION(cudaMemset(t.addr(), 0, t.desc().get_size_in_bytes()));
    //------------------------------------------------------------------
    // Issue the fill operation
    const value_t      FILL_VALUE = static_cast<value_t>(1);

    //for(int i = 0; i < CUPHY_DIM_MAX; ++i)
    //{
    //  printf("[%i]: %i, %i\n", i, grp.ranges()[i].start(), grp.ranges()[i].end());
    //}
    tensor_ref_p s = t.subset(grp);
    //printf("subset: %s\n", s.desc().get_info().to_string().c_str());
    cuphy::tensor_fill(s, FILL_VALUE);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Compare values to expected.
    for(int i1 = 0; i1 < dims[1]; ++i1)
    {
        for(int i0 = 0; i0 < dims[0]; ++i0)
        {
            EXPECT_EQ(t(i0, i1),
                      grp.includes(i0, i1) ? FILL_VALUE : static_cast<value_t>(0))
                      << "DIMS = " << i0 << ", " << i1 << std::endl;
        }
    }
}

template <cuphyDataType_t TType>
void do_fill_index_group_test(const std::array<int, 4>& dims,
                              const cuphy::index_group& grp)
{
    typedef cuphy::typed_tensor<TType, cuphy::pinned_alloc> tensor_p;
    typedef cuphy::typed_tensor_ref<TType>                  tensor_ref_p;
    typedef typename cuphy::type_traits<TType>::type        value_t;
    //------------------------------------------------------------------
    // Allocate a tensor
    tensor_p                 t(cuphy::tensor_layout(dims.size(), dims.data(), nullptr));
    //printf("t: %s\n", t.desc().get_info().to_string().c_str());
    //------------------------------------------------------------------
    // Initialize tensor memory to zero
    CUDA_CHECK_EXCEPTION(cudaMemset(t.addr(), 0, t.desc().get_size_in_bytes()));
    //------------------------------------------------------------------
    // Issue the fill operation
    const value_t      FILL_VALUE = static_cast<value_t>(1);

    //for(int i = 0; i < CUPHY_DIM_MAX; ++i)
    //{
    //  printf("[%i]: %i, %i\n", i, grp.ranges()[i].start(), grp.ranges()[i].end());
    //}
    tensor_ref_p s = t.subset(grp);
    //printf("subset: %s\n", s.desc().get_info().to_string().c_str());
    cuphy::tensor_fill(s, FILL_VALUE);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Compare values to expected.
    for(int i3 = 0; i3 < dims[3]; ++i3)
    {
        for(int i2 = 0; i2 < dims[2]; ++i2)
        {
            for(int i1 = 0; i1 < dims[1]; ++i1)
            {
                for(int i0 = 0; i0 < dims[0]; ++i0)
                {
                  EXPECT_EQ(t(i0, i1, i2, i3),
                            grp.includes(i0, i1, i2, i3) ? FILL_VALUE : static_cast<value_t>(0))
                    << "DIMS = " << i0 << ", " << i1 << ", " << i2 << ", " << i3 << std::endl;
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////
// Fill.IndexGroup
TEST(Fill, IndexGroup)
{
    do_fill_index_group_test<CUPHY_R_32F>(std::array<int, 4>({32, 16, 1, 1}),
                                          cuphy::index_group(cuphy::index_range(16, 25),
                                                             cuphy::dim_all(),
                                                             cuphy::dim_all(),
                                                             cuphy::dim_all()));
    do_fill_index_group_test<CUPHY_R_32F>(std::array<int, 2>({32, 16}),
                                          cuphy::index_group(cuphy::dim_all(),
                                                             cuphy::index_range(4, 9)));
    do_fill_index_group_test<CUPHY_R_32I>(std::array<int, 4>({8, 16, 4, 32}),
                                          cuphy::index_group(cuphy::index_range(0, 8),
                                                             cuphy::index_range(8, 16),
                                                             cuphy::index_range(1, 3),
                                                             cuphy::dim_all()));
    do_fill_index_group_test<CUPHY_R_32U>(std::array<int, 4>({1, 1, 1, 1}),
                                          cuphy::index_group(cuphy::index_range(0, 1),
                                                             cuphy::index_range(0, 1),
                                                             cuphy::index_range(0, 1),
                                                             cuphy::dim_all()));
    do_fill_index_group_test<CUPHY_R_16U>(std::array<int, 4>({32, 16, 1, 1}),
                                          cuphy::index_group(cuphy::index_range(16, 25),
                                                             cuphy::dim_all(),
                                                             cuphy::dim_all(),
                                                             cuphy::dim_all()));
    do_fill_index_group_test<CUPHY_R_8U>(std::array<int, 4>({32, 16, 1, 1}),
                                          cuphy::index_group(cuphy::index_range(16, 25),
                                                             cuphy::dim_all(),
                                                             cuphy::dim_all(),
                                                             cuphy::dim_all()));
    do_fill_index_group_test<CUPHY_R_8I>(std::array<int, 4>({8, 16, 4, 32}),
                                          cuphy::index_group(cuphy::index_range(0, 8),
                                                             cuphy::index_range(8, 16),
                                                             cuphy::index_range(1, 3),
                                                             cuphy::dim_all()));
    do_fill_index_group_test<CUPHY_R_16I>(std::array<int, 4>({8, 16, 4, 32}),
                                          cuphy::index_group(cuphy::index_range(0, 8),
                                                             cuphy::index_range(8, 16),
                                                             cuphy::index_range(1, 3),
                                                             cuphy::dim_all()));
    do_fill_index_group_test<CUPHY_R_16F>(std::array<int, 4>({32, 16, 1, 1}),
                                          cuphy::index_group(cuphy::index_range(16, 25),
                                                             cuphy::dim_all(),
                                                             cuphy::dim_all(),
                                                             cuphy::dim_all()));
}


////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    return result;
}
