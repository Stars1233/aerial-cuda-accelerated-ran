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

#include <Eigen/Dense>
#include "api.h"
#include "cumac.h"

using namespace std;
using namespace cumac;

#define CPUMAT_TIME_MEASURE_
#define CPUMAT_VALIDATE_
#define algNumBsAntConst            256
#define algNumUeAntConst            4
#define algNumRoundConst            1
#define algNumRunTimesConst         10
#define algResultErrTolerance     2e-4

int main()
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    float stddev = 0.5*sqrt(2);
    std::normal_distribution<double> distribution(0.0, stddev);

    cpuMatAlg* cpuMat = new cpuMatAlg;

    const int M = algNumUeAntConst;
    const int N = algNumBsAntConst; 
    const int nRound = algNumRoundConst;

    Eigen::MatrixXcf h(M, N);
    Eigen::MatrixXcf g(N, N);
    Eigen::MatrixXcf resultAB(M, N);
    Eigen::MatrixXcf resultAHA(N, N);
    Eigen::MatrixXcf resultInvA(N, N);

    cuComplex* hCpp     = new cuComplex[M*N];
    cuComplex* gCpp     = new cuComplex[N*N];
    cuComplex* gCppCpy  = new cuComplex[N*N];
    cuComplex* rCppAB   = new cuComplex[M*N];
    cuComplex* rCppAHA  = new cuComplex[N*N];
    cuComplex* rCppInvA = new cuComplex[N*N];

    for (int round = 0; round < nRound; round++) {
        for (int colIdx = 0; colIdx < N; colIdx++)
            for (int rowIdx = 0; rowIdx < M; rowIdx++) {
                hCpp[colIdx*M + rowIdx].x = distribution(generator);
                hCpp[colIdx*M + rowIdx].y = distribution(generator);
            }

        for (int colIdx = 0; colIdx < N; colIdx++)
            for (int rowIdx = 0; rowIdx < N; rowIdx++) {
                gCpp[colIdx*N + rowIdx].x = distribution(generator);
                gCpp[colIdx*N + rowIdx].y = distribution(generator);
                gCppCpy[colIdx*N + rowIdx].x = gCpp[colIdx*N + rowIdx].x;
                gCppCpy[colIdx*N + rowIdx].y = gCpp[colIdx*N + rowIdx].y;
            }    

        h = Eigen::Map<Eigen::MatrixXcf>(reinterpret_cast<complex<float>*>(hCpp), M, N);
        g = Eigen::Map<Eigen::MatrixXcf>(reinterpret_cast<complex<float>*>(gCppCpy), N, N);

//////////////// A x B ////////////////
#ifdef CPUMAT_TIME_MEASURE_ 
        std::clock_t c_start;
        std::clock_t c_end;
        c_start = std::clock();
        for (int j = 0; j<algNumRunTimesConst; j++) {
#endif 
        resultAB = h * g;
#ifdef CPUMAT_TIME_MEASURE_       
        }
        c_end = std::clock();
        long double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC / algNumRunTimesConst;
        std::cout << "A x B Eigen CPU time used: " << time_elapsed_ms << " ms\n";
#endif

#ifdef CPUMAT_TIME_MEASURE_ 
        c_start = std::clock();
        for (int j = 0; j<algNumRunTimesConst; j++) {
#endif 
        cpuMat->matMultiplication_ab(hCpp, M, N, gCpp, N, rCppAB);
#ifdef CPUMAT_TIME_MEASURE_       
        }
        c_end = std::clock();
        time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC / algNumRunTimesConst;
        std::cout << "A x B STD CPU time used: " << time_elapsed_ms << " ms\n";
#endif
#ifdef CPUMAT_VALIDATE_
        bool same = true;
        for (int rIdx = 0; rIdx < M; rIdx++) {
            for (int cIdx = 0; cIdx < N; cIdx++) {
                if (abs(rCppAB[rIdx + cIdx*M].x-resultAB(rIdx, cIdx).real()) > algResultErrTolerance || abs(rCppAB[rIdx + cIdx*M].y-resultAB(rIdx, cIdx).imag()) > algResultErrTolerance) {
                    same = false;
                    printf("want: x = %f, y = %f\n", resultAB(rIdx, cIdx).real(), resultAB(rIdx, cIdx).imag());
                    printf("got: x = %f, y = %f\n", rCppAB[rIdx + cIdx*M].x, rCppAB[rIdx + cIdx*M].y);
                    printf("delta_x = %f, delta_y = %f\n", resultAB(rIdx, cIdx).real()-rCppAB[rIdx + cIdx*M].x, resultAB(rIdx, cIdx).imag()-rCppAB[rIdx + cIdx*M].y);
                    break;
                }
            }
            if (!same) {
                break;
            }
        }

        if (!same) {
            printf("Error: A x B results do not match\n");
            break;
        } else {
            printf("Success: A x B results match\n");
        }
#endif    
        printf("----------------------A x B test done----------------------\n");
//////////////// A^H x A ////////////////
#ifdef CPUMAT_TIME_MEASURE_ 
        c_start = std::clock();
        for (int j = 0; j<algNumRunTimesConst; j++) {
#endif    
        resultAHA = h.adjoint() * h;
#ifdef CPUMAT_TIME_MEASURE_       
        }
        c_end = std::clock();
        time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC / algNumRunTimesConst;
        std::cout << "A^H x A Eigen CPU time used: " << time_elapsed_ms << " ms\n";
#endif

#ifdef CPUMAT_TIME_MEASURE_ 
        c_start = std::clock();
        for (int j = 0; j<algNumRunTimesConst; j++) {
#endif 
        cpuMat->matMultiplication_aHa(hCpp, M, N, rCppAHA);
#ifdef CPUMAT_TIME_MEASURE_       
        }
        c_end = std::clock();
        time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC / algNumRunTimesConst;
        std::cout << "A^H x A STD CPU time used: " << time_elapsed_ms << " ms\n";
#endif       
#ifdef CPUMAT_VALIDATE_
        same = true;
        for (int rIdx = 0; rIdx < N; rIdx++) {
            for (int cIdx = 0; cIdx < N; cIdx++) {
                if (abs(rCppAHA[rIdx + cIdx*N].x-resultAHA(rIdx, cIdx).real()) > algResultErrTolerance || abs(rCppAHA[rIdx + cIdx*N].y-resultAHA(rIdx, cIdx).imag()) > algResultErrTolerance) {
                    same = false;
                    printf("want: x = %f, y = %f\n", resultAHA(rIdx, cIdx).real(), resultAHA(rIdx, cIdx).imag());
                    printf("got: x = %f, y = %f\n", rCppAHA[rIdx + cIdx*N].x, rCppAHA[rIdx + cIdx*N].y);
                    printf("delta_x = %f, delta_y = %f\n", resultAHA(rIdx, cIdx).real()-rCppAHA[rIdx + cIdx*N].x, resultAHA(rIdx, cIdx).imag()-rCppAHA[rIdx + cIdx*N].y);
                    break;
                }
            }
            if (!same) {
                break;
            }
        }

        if (!same) {
            printf("Error: A^H x A results do not match\n");
            break;
        } else {
            printf("Success: A^H x A results match\n");
        }
#endif
        printf("----------------------A^H x A test done----------------------\n");
    //////////////// A inverse ////////////////
#ifdef CPUMAT_TIME_MEASURE_ 
        c_start = std::clock();
        for (int j = 0; j<algNumRunTimesConst; j++) {
#endif      
        resultInvA = g.inverse();
#ifdef CPUMAT_TIME_MEASURE_       
        }
        c_end = std::clock();
        time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC / algNumRunTimesConst;
        std::cout << "A inverse Eigen CPU time used: " << time_elapsed_ms << " ms\n";
#endif

#ifdef CPUMAT_TIME_MEASURE_ 
        c_start = std::clock();
#endif 
        cpuMat->matInverse(gCpp, N, rCppInvA);
#ifdef CPUMAT_TIME_MEASURE_       
        c_end = std::clock();
        time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
        std::cout << "A inverse STD CPU time used: " << time_elapsed_ms << " ms\n";
#endif  
#ifdef CPUMAT_VALIDATE_
        // check if outputs of Eigen and cpuMatAlg functions match
        same = true;
        for (int rIdx = 0; rIdx < N; rIdx++) {
            for (int cIdx = 0; cIdx < N; cIdx++) {
                if (abs(rCppInvA[rIdx + cIdx*N].x-resultInvA(rIdx, cIdx).real()) > algResultErrTolerance || abs(rCppInvA[rIdx + cIdx*N].y-resultInvA(rIdx, cIdx).imag()) > algResultErrTolerance) {
                    same = false;
                    printf("want: x = %f, y = %f\n", resultInvA(rIdx, cIdx).real(), resultInvA(rIdx, cIdx).imag());
                    printf("got: x = %f, y = %f\n", rCppInvA[rIdx + cIdx*N].x, rCppInvA[rIdx + cIdx*N].y);
                    printf("delta_x = %f, delta_y = %f\n", resultInvA(rIdx, cIdx).real()-rCppInvA[rIdx + cIdx*N].x, resultInvA(rIdx, cIdx).imag()-rCppInvA[rIdx + cIdx*N].y);
                    break;
                }
            }
            if (!same) {
                break;
            }
        }

        if (!same) {
            printf("Error: A inverse results do not match\n");
            break;
        } else {
            printf("Success: A inverse results match\n");
        }
#endif
        printf("----------------------A inverse test done----------------------\n");

    }
    delete hCpp;
    delete gCpp;
    delete gCppCpy;
    delete rCppAB;
    delete rCppAHA;
    delete rCppInvA;
    return 0;
}