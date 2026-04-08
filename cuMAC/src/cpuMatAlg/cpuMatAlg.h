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

 #pragma once

 #include "api.h"
 #include "cumac.h"

 // cuMAC namespace
 namespace cumac {

 class cpuMatAlg {
 public:
    cpuMatAlg();
    ~cpuMatAlg();

    // column-major matrix access
    // multiply matrices A and B
    static void matMultiplication_ab(cuComplex* A, int A_R, int A_C, cuComplex* B, int B_C, cuComplex* Result);
    // multiply matrices A^H and B
    static void matMultiplication_aHb(cuComplex* A, int A_R, int A_C, cuComplex* B, int B_C, cuComplex* Result);
    // multiply matrices A^H and A
    static void matMultiplication_aHa(cuComplex* A, int A_R, int A_C, cuComplex* Result);
    // multiply matrices A and A^H
    static void matMultiplication_aaH(cuComplex* A, int A_R, int A_C, cuComplex* Result);
    // compute AxA^H + B, and store the result in B
    static void matMultiplication_aaHplusb(cuComplex* A, int A_R, int A_C, cuComplex* Result);
    // inverse matrix A 
    // ToFix: Current implemention has precision issue
    static void matInverse(cuComplex* A, int A_D, cuComplex* Result);

    // row-major matrix access
    // multiply matrices A and B
    static void matMultiplication_ab_rm(cuComplex* A, int A_R, int A_C, cuComplex* B, int B_C, cuComplex* Result);
    // multiply matrices A^H and B
    static void matMultiplication_aHb_rm(cuComplex* A, int A_R, int A_C, cuComplex* B, int B_C, cuComplex* Result);
    // multiply matrices A^H and A
    static void matMultiplication_aHa_rm(cuComplex* A, int A_R, int A_C, cuComplex* Result);
    // multiply matrices A and A^H
    static void matMultiplication_aaH_rm(cuComplex* A, int A_R, int A_C, cuComplex* Result);
    // compute AxA^H + B, and store the result in B
    static void matMultiplication_aaHplusb_rm(cuComplex* A, int A_R, int A_C, cuComplex* Result);
    // inverse matrix A 
    // ToFix: Current implemention has precision issue
    static void matInverse_rm(cuComplex* A, int A_D, cuComplex* Result);

    // inverse matrix A using Eigen
    static void matInverseEigen(cuComplex* A, int A_D, cuComplex* Result);
 private:

 };
 }