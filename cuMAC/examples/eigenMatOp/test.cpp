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
#include <Eigen/Dense>

using namespace std;

// #define EIGEN_DEBUG_
#define EIGEN_TIME_MEASURE_
// #define EIGEN_OUTPUT_DEBUG_
#define nBsAntConst            4
#define nUeAntConst            4
#define nRoundConst            1
#define nRunTimesConst         1000

void matMultiplication(cuComplex* A, int A_R, int A_C, cuComplex* B, int B_C, cuComplex* Result);

int main()
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    float stddev = 0.5*sqrt(2);
    std::normal_distribution<double> distribution(0.0, stddev);

    const int M = nUeAntConst;
    const int N = nBsAntConst; 
    const int nRound = nRoundConst;

    Eigen::MatrixXcf h(M, N);
    Eigen::MatrixXcf g(N, N);
    Eigen::MatrixXcf result(M, N);

    cuComplex* hCpp = new cuComplex[M*N];
    cuComplex* gCpp = new cuComplex[N*N];
    cuComplex* rCpp = new cuComplex[M*N];

    for (int round = 0; round < nRound; round++) {
        for (int colIdx = 0; colIdx < N; colIdx++)
            for (int rowIdx = 0; rowIdx < M; rowIdx++) {
                hCpp[colIdx*M + rowIdx].x = distribution(generator);
                hCpp[colIdx*M + rowIdx].y = distribution(generator);
                gCpp[colIdx*M + rowIdx].x = distribution(generator);
                gCpp[colIdx*M + rowIdx].y = distribution(generator);
            }
        
        h = Eigen::Map<Eigen::MatrixXcf>(reinterpret_cast<complex<float>*>(hCpp), M, N);
        g = Eigen::Map<Eigen::MatrixXcf>(reinterpret_cast<complex<float>*>(gCpp), N, N);

 #ifdef EIGEN_DEBUG_
        printf("MatrixXcf: \n");
        std::cout<< h <<std::endl;
        printf("std: \n");
        for (int rowIdx = 0; rowIdx < M; rowIdx++) {
            for (int colIdx = 0; colIdx < N; colIdx++)
                std::cout<<"("<< hCpp[colIdx*M + rowIdx].x << "  "<<hCpp[colIdx*M + rowIdx].y<<")  ";
            std::cout<<std::endl;
        }
#endif

#ifdef EIGEN_TIME_MEASURE_ 
        std::clock_t c_start = std::clock();
        for (int j = 0; j<nRunTimesConst; j++) {
#endif 
        result = h*g;
#ifdef EIGEN_TIME_MEASURE_       
        }
        std::clock_t c_end = std::clock();
        long double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC / nRunTimesConst;
        std::cout << "Eigen CPU time used: " << time_elapsed_ms << " ms\n";
#endif
#ifdef EIGEN_OUTPUT_DEBUG_
        printf("Result - MatrixXcf: \n");
        std::cout<< result <<std::endl;
#endif

 #ifdef EIGEN_TIME_MEASURE_ 
        c_start = std::clock();
        for (int j = 0; j<nRunTimesConst; j++) {
 #endif 
        matMultiplication(hCpp, M, N, gCpp, N, rCpp);
 #ifdef EIGEN_TIME_MEASURE_       
        }
        c_end = std::clock();
        time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC / nRunTimesConst;
        std::cout << "STD CPU time used: " << time_elapsed_ms << " ms\n";
 #endif
 #ifdef EIGEN_OUTPUT_DEBUG_
        printf("Result - std: \n");
        for (int rowIdx = 0; rowIdx < M; rowIdx++) {
            for (int colIdx = 0; colIdx < N; colIdx++)
                std::cout<<"("<< rCpp[colIdx*M + rowIdx].x << "  "<<rCpp[colIdx*M + rowIdx].y<<")  ";
            std::cout<<std::endl;
        }
 #endif    
    }
    
    delete hCpp;
    delete gCpp;
    return 0;
}

void matMultiplication(cuComplex* A, int A_R, int A_C, cuComplex* B, int B_C, cuComplex* Result) 
{
    for (int rowIdx = 0; rowIdx < A_R; rowIdx++) {
        for (int colIdx = 0; colIdx < B_C; colIdx++) {
            Result[colIdx*A_R + rowIdx].x = 0;
            Result[colIdx*A_R + rowIdx].y = 0;
            for (int i = 0; i < A_C; i++) {
                cuComplex temp1 = A[i*A_R + rowIdx];
                cuComplex temp2 = B[colIdx*A_C + i];
                Result[colIdx*A_R + rowIdx].x += temp1.x*temp2.x - temp1.y*temp2.y;
                Result[colIdx*A_R + rowIdx].y += temp1.x*temp2.y + temp2.x*temp1.y;
            }
        }
    }
}