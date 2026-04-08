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

#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "../common/common.hpp"
#include "../reference/reference.hpp"

template<class BLAS, class ValueType = typename example::uniform_value_type_t<BLAS>>
__launch_bounds__(BLAS::max_threads_per_block) 
    __global__                                 
    void gemm_kernel(const ValueType* a,
                     const ValueType* b,
                     const ValueType* c,
                     const ValueType  alpha,
                     const ValueType  beta,
                     ValueType*       output,
                     const uint16_t   stride_abc,
                     ValueType*       inverse_norms
        )
{

    a += stride_abc*blockIdx.x;
    b += stride_abc*blockIdx.x;
    c += stride_abc*blockIdx.x;

    output += stride_abc*blockIdx.x;

    using value_type = ValueType;

    extern __shared__ __align__(16) char smem[];

    auto a_global_tensor = cublasdx::make_tensor(a, BLAS::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, BLAS::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, BLAS::get_layout_gmem_c());

    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<BLAS>(smem);

    auto a_shared_tensor = cublasdx::make_tensor(smem_a, BLAS::get_layout_smem_a());
    auto b_shared_tensor = cublasdx::make_tensor(smem_b, BLAS::get_layout_smem_b());
    auto c_shared_tensor = cublasdx::make_tensor(smem_c, BLAS::get_layout_smem_c());

    using alignment = cublasdx::alignment_of<BLAS>;

    cublasdx::copy<BLAS, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<BLAS, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy<BLAS, alignment::c>(c_global_tensor, c_shared_tensor);

    cublasdx::copy_wait();

    BLAS().execute(alpha, a_shared_tensor, b_shared_tensor, beta, c_shared_tensor);

    __syncthreads();

    auto out_global_tensor = cublasdx::make_tensor(output, BLAS::get_layout_gmem_c());

    cublasdx::copy<BLAS, alignment::c>(c_shared_tensor, out_global_tensor);

    // write out the first 16 inverse diagonal elements to the norms vector.
    // as yet, I don't know how c_shared_tensor is arranged - so just saving the first 16.
    // a different indexing wouldn't affect performance.
    // ... I'll get to the inverse square root later.
    if ( threadIdx.x < 16 ) {
        inverse_norms[threadIdx.x] = c_shared_tensor[blockIdx.x*256+threadIdx.x];
    }

}



// exactly the same interface as the previous kernel - just this time only scale - so I don't use all arguments
template<class BLAS, class ValueType = typename example::uniform_value_type_t<BLAS>>
//__launch_bounds__(BLAS::max_threads_per_block) 
    __global__                                 
    void gemm_kernel_scale(const ValueType* a,
                           const ValueType* b,
                           const ValueType* c,
                           const ValueType  alpha,
                           const ValueType  beta,
                           ValueType*       output,
                           const uint16_t   stride_abc,
                           ValueType*       inverse_norms
        )
{

    __shared__ ValueType s_norms[1536];

    output += stride_abc*blockIdx.x;

    // load norms
    for ( int i=threadIdx.x; i<1536; i+=blockDim.x ) {
        s_norms[i] = inverse_norms[ (blockIdx.x/6)*1536 + i ];
    }

    __syncthreads();

    // scale result - hard coded for now
    for ( int irow=0; irow<96; irow+=blockDim.x/256 ) {
        if ( irow+threadIdx.x/256 < 96 && irow*256+threadIdx.x < 96*256 ) {
            output[irow*256+threadIdx.x] *= s_norms[irow+threadIdx.x/256]*s_norms[threadIdx.x%256];
        }
    }

}

// This is an example of complex fp16 general matrix-matrix multiplication (GEMM) performed
// in a single CUDA block:
//
//              C = alpha * A * B + beta * C
//
// * A, B, and C are matrices containing complex half precision floating-point values.
// * alpha and beta are complex half precision floating-point values.
//
// Input data is generated on host using random number generators, and later copied to
// the global memory. Next, kernel with GEMM is executed, and then the matrix C (the result)
// is copied back to host memory. The results are verified against cuBLAS.
//
// In this example the number of threads participating in the GEMM operation is automatically
// selected by cuBLASDx. Setting operator BlockDim in the GEMM definition can be used to impose the
// number of threads that the GEMM will be performed with. Block dimensions are provided via
// BLAS::block_dim trait.
template<unsigned int Arch>
int simple_gemm() {
    // Parameters m, n, k define the dimensions of matrices A, B, and C

    // For cuMAC
    // 96 x 1536 x 64  doesn't fit
    // 96 x 768 x 64 doesn't fit
    // 96 x 384 x 64 doesn't fit
    // 96 x 128 - good
    // 96 x 256

    constexpr unsigned int nSrsPortsUpdatePerSLot = 96;
    constexpr unsigned int nSrsPortsEachCell      = 1536; // maximum number of SRS ports per cell. Assuming maximum 384 SRS enabled UEs per cell and 4 SRS ports per UE.
    constexpr unsigned int nBsAnt    = 64;
    constexpr unsigned int nPrgsPerSB = 2;
    constexpr unsigned int nSubBands = 4;
    constexpr unsigned int nTotPrgs = nPrgsPerSB*nSubBands;
    

    constexpr unsigned int m          = 96;
    constexpr unsigned int n          = 256;
    constexpr unsigned int k          = 64;
    constexpr unsigned int g          = nSrsPortsEachCell/n*nTotPrgs;

    printf ("\n  nSrsPortsUpdatePerSLot = %d \n",   nSrsPortsUpdatePerSLot);
    printf ("  nSrsPortsEachCell  = %d \n",   nSrsPortsEachCell);
    printf ("  nBsAnt    = %d \n",   nBsAnt);
    printf ("  nPrgsPerSB  = %d \n\n", nPrgsPerSB);
    printf ("  nSubBand  = %d \n\n", nSubBands);
    printf ("  nTotPrgs  = %d \n\n", nTotPrgs);
    printf ("  m, n, k   = %d, %d, %d \n", m, n, k);
    printf ("  grid      = (%d, 1, 1) \n\n", g);

    // GEMM definition using cuBLASDx operators:
    // 1. The size, the precision, and the type (real or complex) are set.
    // 2. The BLAS function is selected: MM (matrix multiplication).
    // 3. The data arrangements of A, B matrices are set (C is defaulted to column major).
    //    - Optional
    // 4. The data alignment of A, B and C matrices is set to the max accepted value using alias MaxAlignment.
    //    - Optional
    // 4. Block operator informs that GEMM should be performed on CUDA block level.
    // 5. Targeted CUDA compute capability is selected with SM operator.

    using BLAS = decltype(cublasdx::Size<m, n, k>() +
                          cublasdx::Precision<__half>() +
                          cublasdx::Type<cublasdx::type::complex>() +
                          cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>() +
                          cublasdx::MaxAlignment() +
                          cublasdx::Block() +
                          cublasdx::SM<Arch>());

    using value_type = typename example::uniform_value_type_t<BLAS>;

    // Allocate managed memory for a, b, c, and output
    value_type* inputs;
    value_type* output;
    value_type* norms;
    value_type* h_norms;

    constexpr auto global_a_size = example::global_memory_size_of<BLAS>::a_size;
    constexpr auto global_b_size = example::global_memory_size_of<BLAS>::b_size;
    constexpr auto global_c_size = example::global_memory_size_of<BLAS>::c_size;

    auto inputs_size       = (global_a_size + global_b_size + global_c_size)*g;
    auto inputs_size_bytes = inputs_size * sizeof(value_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&inputs, inputs_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&output, inputs_size * sizeof(value_type)));

    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&h_norms, nSrsPortsEachCell*nSubBands*sizeof(value_type)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&norms, nSrsPortsEachCell*nSubBands*sizeof(value_type)));
    for ( int i=0; i<nSrsPortsEachCell*nSubBands; ++i ) {
        h_norms[i] = 1.0;
    }
   
    value_type* a     = inputs;
    value_type* b     = a + (global_a_size);
    value_type* c     = b + (global_b_size);
    value_type  alpha = value_type(1.0, 0.0);
    value_type  beta  = value_type(0.0, 0.0);

    // Fill the A, B, C matrices with random values
    // I made a much larger, and jut filled it with random data and just copied it to the GPU.
    auto host_a = example::get_random_data<value_type>(0.1, 1.0, (global_a_size+global_b_size+global_c_size)*g);
    //auto host_b = example::get_random_data<value_type>(0.1, 1.0, global_b_size*g);
    //auto host_c = example::get_random_data<value_type>(0.1, 1.0, global_c_size*g);

    CUDA_CHECK_AND_EXIT(cudaMemcpy(a, host_a.data(), global_a_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(norms, h_norms, nSrsPortsEachCell*nSubBands * sizeof(value_type), cudaMemcpyHostToDevice));
    //CUDA_CHECK_AND_EXIT(cudaMemcpy(b, host_b.data(), global_b_size * sizeof(value_type), cudaMemcpyHostToDevice));
    //CUDA_CHECK_AND_EXIT(cudaMemcpy(c, host_c.data(), global_c_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Increase max dynamic shared memory for the kernel if needed
    CUDA_CHECK_AND_EXIT(
        cudaFuncSetAttribute(gemm_kernel<BLAS>, cudaFuncAttributeMaxDynamicSharedMemorySize, cublasdx::get_shared_storage_size<BLAS>()));

    
    // Execute kernel
    for ( int iter=0; iter<100; ++iter ) {

        // GEMM
        gemm_kernel<BLAS><<<g, BLAS::block_dim, cublasdx::get_shared_storage_size<BLAS>()>>>(a, b, c, alpha, beta, output,
                                                                                             global_a_size + global_b_size + global_c_size,
                                                                                             norms);
        // Scale (extra arguments ... not going to change performance)
        gemm_kernel_scale<BLAS><<<g, 1024>>>(a, b, c, alpha, beta, output,
                                            global_a_size + global_b_size + global_c_size,
                                            norms);
    } 
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    printf ("all 100 executed \n");

    // Copy results back to host
    std::vector<value_type> host_output(inputs_size);
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(host_output.data(), output, inputs_size * sizeof(value_type), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Free device memory
    CUDA_CHECK_AND_EXIT(cudaFree(inputs));
    CUDA_CHECK_AND_EXIT(cudaFree(output));


    // I havent had a chance to decode how cuBlasDx is doing this, or code something equivalent.

    // Calculate reference
    //auto reference_host_output = example::reference_gemm<BLAS>(alpha, host_a, host_a, beta, host_a);

    // Check against reference
    /*
    if (example::check_error<BLAS>(host_output, reference_host_output)) {
        std::cout << "Success" << std::endl;
        return 0;
    }
    */
    std::cout << "Failure" << std::endl;
    return 1;
}

struct simple_gemm_functor {
    template<int Arch>
    int operator()(std::integral_constant<int, Arch>) {
        return simple_gemm<Arch>();
    }
};

int main(int, char**) {
    return example::sm_runner(simple_gemm_functor{});
}
