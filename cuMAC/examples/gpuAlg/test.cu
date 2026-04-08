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

 #include "api.h"
 #include "cumac.h"
 
 using namespace std;
 using namespace cumac;
 
 #define numEntriesToSort 100
 
 static __global__ void testSort(float* pfMetricArr, uint16_t* idArr, uint16_t n);
 
 int main()
 {
     unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
     std::default_random_engine generator (seed);
     float stddev = 1.0;
     std::normal_distribution<double> distribution(0.0, stddev);
 
     // create CUDA stream
     cudaStream_t cuStrmMain;
     CUDA_CHECK_ERR(cudaStreamCreate(&cuStrmMain));
 
     float* pfMetricArr_d;
     float* pfMetricArr_h;
     float* pfMetricArr_hCpy;
     uint16_t* idArr_d;
     uint16_t* idArr_h;
     uint16_t* idArr_hCpy;
     uint16_t n = numEntriesToSort;
     uint16_t pow2N = 2;
 
     while(pow2N<n) {
         pow2N = pow2N << 1;
     }

     printf("Orignal array size: %d, rounded up to power of 2: %d\n", n, pow2N);
 
     pfMetricArr_h = new float[pow2N];
     pfMetricArr_hCpy = new float[n];
     idArr_h = new uint16_t[pow2N];
     idArr_hCpy = new uint16_t[n];
 
     uint32_t sizePfArr = sizeof(float)*pow2N;
     uint32_t sizeIdArr = sizeof(uint16_t)*pow2N;
 
     CUDA_CHECK_ERR(cudaMalloc((void **)&pfMetricArr_d, sizePfArr));
     CUDA_CHECK_ERR(cudaMalloc((void **)&idArr_d, sizeIdArr));
 
     printf("Before sorting:\n");
     for (int i = 0; i<n; i++) {
         pfMetricArr_h[i] = distribution(generator);
         pfMetricArr_hCpy[i] = pfMetricArr_h[i];
         idArr_h[i] = i;
         idArr_hCpy[i] = idArr_h[i];
         printf("(%f, %d)  ", pfMetricArr_h[i], idArr_h[i]);
     }
     printf("\n\n");
 
     for (int i = n; i<pow2N; i++) {
         pfMetricArr_h[i] = -std::numeric_limits<float>::max();
         idArr_h[i] = i;
     }
 
     CUDA_CHECK_ERR(cudaMemcpyAsync(pfMetricArr_d, pfMetricArr_h, sizePfArr, cudaMemcpyHostToDevice, cuStrmMain));
     CUDA_CHECK_ERR(cudaMemcpyAsync(idArr_d, idArr_h, sizeIdArr, cudaMemcpyHostToDevice, cuStrmMain));
     CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
 
     testSort<<<1, 1024, 0, cuStrmMain>>>(pfMetricArr_d, idArr_d, pow2N);
     CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
 
     CUDA_CHECK_ERR(cudaMemcpyAsync(pfMetricArr_h, pfMetricArr_d, sizePfArr, cudaMemcpyDeviceToHost, cuStrmMain));
     CUDA_CHECK_ERR(cudaMemcpyAsync(idArr_h, idArr_d, sizeIdArr, cudaMemcpyDeviceToHost, cuStrmMain));
     CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
 
     printf("After sorting:\n");
     for (int i = 0; i<n; i++) {
         printf("(%f, %d)  ", pfMetricArr_h[i], idArr_h[i]);
     }
     printf("\n");
 
     // verification
     bool success = true;
     for (int i = 0; i<n; i++) {
         bool found = false;
         for (int j = 0; j<n; j++) {
             if (pfMetricArr_hCpy[j] == pfMetricArr_h[i] && idArr_hCpy[j] == idArr_h[i]) {
                 found = true;
                 break;
             }
         }
 
         if (!found) {
             printf("Error: entry not found in original array\n");
             success = false;
             break;
         }
 
         if (i == n-1)
             break;
 
         if (pfMetricArr_h[i]<pfMetricArr_h[i+1]) {
             printf("Error: sorting result incorrect\n");
             success = false;
             break;
         }
     }

     if (success) 
        printf("Success: sorting result correct\n");
     
     CUDA_CHECK_ERR(cudaFree(pfMetricArr_d));
     CUDA_CHECK_ERR(cudaFree(idArr_d));
     delete pfMetricArr_h;
     delete pfMetricArr_hCpy;
     delete idArr_h;
     delete idArr_hCpy;
 
     return 0;
 }
 
 
 static __global__ void testSort(float* pfMetricArr, uint16_t* idArr, uint16_t n)
 {
     bitonicSort(pfMetricArr, idArr, n);
 }