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
 #include "cumac.h"

 // cuMAC namespace
 namespace cumac {
 
 cpuMatAlg::cpuMatAlg()
 {
    
 }

 cpuMatAlg::~cpuMatAlg()
 {

 }

 // column-major matrix access
 void cpuMatAlg::matMultiplication_ab(cuComplex* A, int A_R, int A_C, cuComplex* B, int B_C, cuComplex* Result) 
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

 void cpuMatAlg::matMultiplication_aHb(cuComplex* A, int A_R, int A_C, cuComplex* B, int B_C, cuComplex* Result)
 {
    for (int rowIdx = 0; rowIdx < A_C; rowIdx++) {
        for (int colIdx = 0; colIdx < B_C; colIdx++) {
            Result[colIdx*A_C + rowIdx].x = 0;
            Result[colIdx*A_C + rowIdx].y = 0;
            for (int i = 0; i < A_R; i++) {
                cuComplex temp1 = A[rowIdx*A_R + i];
                cuComplex temp2 = B[colIdx*A_R + i];
                Result[colIdx*A_C + rowIdx].x += temp1.x*temp2.x + temp1.y*temp2.y;
                Result[colIdx*A_C + rowIdx].y += temp1.x*temp2.y - temp2.x*temp1.y;
            }
        }
    }
 }

 void cpuMatAlg::matMultiplication_aHa(cuComplex* A, int A_R, int A_C, cuComplex* Result)
 {
    for (int rowIdx = 0; rowIdx < A_C; rowIdx++) {
         for (int colIdx = 0; colIdx < A_C; colIdx++) {
             Result[colIdx*A_C + rowIdx].x = 0;
             Result[colIdx*A_C + rowIdx].y = 0;
             for (int i = 0; i < A_R; i++) {
                 cuComplex temp1 = A[rowIdx*A_R + i];
                 cuComplex temp2 = A[colIdx*A_R + i];
                 Result[colIdx*A_C + rowIdx].x += temp1.x*temp2.x + temp1.y*temp2.y;
                 Result[colIdx*A_C + rowIdx].y += temp1.x*temp2.y - temp2.x*temp1.y;
             }
         }
    }
 }

 void cpuMatAlg::matMultiplication_aaH(cuComplex* A, int A_R, int A_C, cuComplex* Result)
 {
    for (int rowIdx = 0; rowIdx < A_R; rowIdx++) {
         for (int colIdx = 0; colIdx < A_R; colIdx++) {
             Result[colIdx*A_R + rowIdx].x = 0;
             Result[colIdx*A_R + rowIdx].y = 0;

             for (int i = 0; i < A_C; i++) {
                 cuComplex temp1 = A[i*A_R + rowIdx];
                 cuComplex temp2 = A[i*A_R + colIdx];
                 Result[colIdx*A_R + rowIdx].x += temp1.x*temp2.x + temp1.y*temp2.y;
                 Result[colIdx*A_R + rowIdx].y += temp2.x*temp1.y - temp1.x*temp2.y;
             }
         }
    }
 }

 void cpuMatAlg::matMultiplication_aaHplusb(cuComplex* A, int A_R, int A_C, cuComplex* Result)
 {
    for (int rowIdx = 0; rowIdx < A_R; rowIdx++) {
         for (int colIdx = 0; colIdx < A_R; colIdx++) {
            for (int i = 0; i < A_C; i++) {
                 cuComplex temp1 = A[i*A_R + rowIdx];
                 cuComplex temp2 = A[i*A_R + colIdx];
                 Result[colIdx*A_R + rowIdx].x += temp1.x*temp2.x + temp1.y*temp2.y;
                 Result[colIdx*A_R + rowIdx].y += temp2.x*temp1.y - temp1.x*temp2.y;
             }
         }
    }
 }

 void cpuMatAlg::matInverse(cuComplex* A, int A_D, cuComplex* Result)
 {
    for (int col_i = 0; col_i < A_D; col_i++) {
        for (int row_i=0; row_i < A_D; row_i++) {
            if (row_i==col_i) {
                Result[col_i*A_D + row_i].x = 1.0;
                Result[col_i*A_D + row_i].y = 0;
            } else {
                Result[col_i*A_D + row_i].x = 0;
                Result[col_i*A_D + row_i].y = 0;
            }
        }
    }

    cuComplex c_coeff;
    cuComplex c_inv_coeff;
    cuComplex d_coeff;
    float     d_multp;
    cuComplex p_coeff;
    cuComplex p_inv_coeff;
    cuComplex l_coeff;

    for (int col_i = 0; col_i < A_D; col_i++) {
        d_coeff = A[col_i*A_D + col_i];
        d_multp = 1.0/(d_coeff.x*d_coeff.x + d_coeff.y*d_coeff.y);

        for (int col_j = 0; col_j < A_D; col_j++) {
            c_coeff = A[col_j*A_D + col_i];
            c_inv_coeff = Result[col_j*A_D + col_i];
            A[col_j*A_D + col_i].x = d_multp * (c_coeff.x*d_coeff.x + c_coeff.y*d_coeff.y);
            A[col_j*A_D + col_i].y = d_multp * (c_coeff.y*d_coeff.x - c_coeff.x*d_coeff.y);
            Result[col_j*A_D + col_i].x = d_multp * (c_inv_coeff.x*d_coeff.x + c_inv_coeff.y*d_coeff.y);
            Result[col_j*A_D + col_i].y = d_multp * (c_inv_coeff.y*d_coeff.x - c_inv_coeff.x*d_coeff.y);
        }

        for (int row_i = 0; row_i < A_D; row_i++) {
            if (row_i == col_i)
                continue;
            
            l_coeff = A[row_i+col_i*A_D];
            for (int col_j = 0; col_j < A_D; col_j++) {
                p_coeff = A[col_i+col_j*A_D];
                p_inv_coeff = Result[col_i+col_j*A_D];
                c_coeff = A[row_i+col_j*A_D];
                c_inv_coeff = Result[row_i+col_j*A_D];

                A[row_i+col_j*A_D].x = c_coeff.x - (l_coeff.x*p_coeff.x - l_coeff.y*p_coeff.y);
                A[row_i+col_j*A_D].y = c_coeff.y - (l_coeff.x*p_coeff.y + l_coeff.y*p_coeff.x);
                Result[row_i+col_j*A_D].x = c_inv_coeff.x - (l_coeff.x*p_inv_coeff.x - l_coeff.y*p_inv_coeff.y);
                Result[row_i+col_j*A_D].y = c_inv_coeff.y - (l_coeff.x*p_inv_coeff.y + l_coeff.y*p_inv_coeff.x);  
            }
        }
    }   
 }

 // row-major matrix access
 void cpuMatAlg::matMultiplication_ab_rm(cuComplex* A, int A_R, int A_C, cuComplex* B, int B_C, cuComplex* Result) 
 {
     for (int rowIdx = 0; rowIdx < A_R; rowIdx++) {
         for (int colIdx = 0; colIdx < B_C; colIdx++) {
             Result[colIdx + rowIdx*B_C].x = 0;
             Result[colIdx + rowIdx*B_C].y = 0;
             for (int i = 0; i < A_C; i++) {
                 cuComplex temp1 = A[i + rowIdx*A_C];
                 cuComplex temp2 = B[colIdx + i*B_C];
                 Result[colIdx + rowIdx*B_C].x += temp1.x*temp2.x - temp1.y*temp2.y;
                 Result[colIdx + rowIdx*B_C].y += temp1.x*temp2.y + temp2.x*temp1.y;
             }
         }
     }
 }

 void cpuMatAlg::matMultiplication_aHb_rm(cuComplex* A, int A_R, int A_C, cuComplex* B, int B_C, cuComplex* Result)
 {
    for (int rowIdx = 0; rowIdx < A_C; rowIdx++) {
        for (int colIdx = 0; colIdx < B_C; colIdx++) {
            Result[colIdx + rowIdx*B_C].x = 0;
            Result[colIdx + rowIdx*B_C].y = 0;
            for (int i = 0; i < A_R; i++) {
                cuComplex temp1 = A[rowIdx + i*A_C];
                cuComplex temp2 = B[colIdx + i*B_C];
                Result[colIdx + rowIdx*B_C].x += temp1.x*temp2.x + temp1.y*temp2.y;
                Result[colIdx + rowIdx*B_C].y += temp1.x*temp2.y - temp2.x*temp1.y;
            }
        }
    }
 }

 void cpuMatAlg::matMultiplication_aHa_rm(cuComplex* A, int A_R, int A_C, cuComplex* Result)
 {
    for (int rowIdx = 0; rowIdx < A_C; rowIdx++) {
         for (int colIdx = 0; colIdx < A_C; colIdx++) {
             Result[colIdx + rowIdx*A_C].x = 0;
             Result[colIdx + rowIdx*A_C].y = 0;
             for (int i = 0; i < A_R; i++) {
                 cuComplex temp1 = A[rowIdx + i*A_C];
                 cuComplex temp2 = A[colIdx + i*A_C];
                 Result[colIdx + rowIdx*A_C].x += temp1.x*temp2.x + temp1.y*temp2.y;
                 Result[colIdx + rowIdx*A_C].y += temp1.x*temp2.y - temp2.x*temp1.y;
             }
         }
    }
 }

 void cpuMatAlg::matMultiplication_aaH_rm(cuComplex* A, int A_R, int A_C, cuComplex* Result)
 {
    for (int rowIdx = 0; rowIdx < A_R; rowIdx++) {
         for (int colIdx = 0; colIdx < A_R; colIdx++) {
             Result[colIdx + rowIdx*A_R].x = 0;
             Result[colIdx + rowIdx*A_R].y = 0;

             for (int i = 0; i < A_C; i++) {
                 cuComplex temp1 = A[i + rowIdx*A_C];
                 cuComplex temp2 = A[i + colIdx*A_C];
                 Result[colIdx + rowIdx*A_R].x += temp1.x*temp2.x + temp1.y*temp2.y;
                 Result[colIdx + rowIdx*A_R].y += temp2.x*temp1.y - temp1.x*temp2.y;
             }
         }
    }
 }

 void cpuMatAlg::matMultiplication_aaHplusb_rm(cuComplex* A, int A_R, int A_C, cuComplex* Result)
 {
    for (int rowIdx = 0; rowIdx < A_R; rowIdx++) {
         for (int colIdx = 0; colIdx < A_R; colIdx++) {
            for (int i = 0; i < A_C; i++) {
                 cuComplex temp1 = A[i + rowIdx*A_C];
                 cuComplex temp2 = A[i + colIdx*A_C];
                 Result[colIdx + rowIdx*A_R].x += temp1.x*temp2.x + temp1.y*temp2.y;
                 Result[colIdx + rowIdx*A_R].y += temp2.x*temp1.y - temp1.x*temp2.y;
             }
         }
    }
 }

 void cpuMatAlg::matInverse_rm(cuComplex* A, int A_D, cuComplex* Result)
 {
    for (int col_i = 0; col_i < A_D; col_i++) {
        for (int row_i=0; row_i < A_D; row_i++) {
            if (row_i==col_i) {
                Result[col_i*A_D + row_i].x = 1.0;
                Result[col_i*A_D + row_i].y = 0;
            } else {
                Result[col_i + row_i*A_D].x = 0;
                Result[col_i + row_i*A_D].y = 0;
            }
        }
    }

    cuComplex c_coeff;
    cuComplex c_inv_coeff;
    cuComplex d_coeff;
    float     d_multp;
    cuComplex p_coeff;
    cuComplex p_inv_coeff;
    cuComplex l_coeff;

    for (int col_i = 0; col_i < A_D; col_i++) {
        d_coeff = A[col_i*A_D + col_i];
        d_multp = 1.0/(d_coeff.x*d_coeff.x + d_coeff.y*d_coeff.y);

        for (int col_j = 0; col_j < A_D; col_j++) {
            c_coeff = A[col_j + col_i*A_D];
            c_inv_coeff = Result[col_j + col_i*A_D];
            A[col_j + col_i*A_D].x = d_multp * (c_coeff.x*d_coeff.x + c_coeff.y*d_coeff.y);
            A[col_j + col_i*A_D].y = d_multp * (c_coeff.y*d_coeff.x - c_coeff.x*d_coeff.y);
            Result[col_j + col_i*A_D].x = d_multp * (c_inv_coeff.x*d_coeff.x + c_inv_coeff.y*d_coeff.y);
            Result[col_j + col_i*A_D].y = d_multp * (c_inv_coeff.y*d_coeff.x - c_inv_coeff.x*d_coeff.y);
        }

        for (int row_i = 0; row_i < A_D; row_i++) {
            if (row_i == col_i)
                continue;
            
            l_coeff = A[row_i*A_D+col_i];
            for (int col_j = 0; col_j < A_D; col_j++) {
                p_coeff = A[col_i*A_D+col_j];
                p_inv_coeff = Result[col_i*A_D+col_j];
                c_coeff = A[row_i*A_D+col_j];
                c_inv_coeff = Result[row_i*A_D+col_j];

                A[row_i*A_D+col_j].x = c_coeff.x - (l_coeff.x*p_coeff.x - l_coeff.y*p_coeff.y);
                A[row_i*A_D+col_j].y = c_coeff.y - (l_coeff.x*p_coeff.y + l_coeff.y*p_coeff.x);
                Result[row_i*A_D+col_j].x = c_inv_coeff.x - (l_coeff.x*p_inv_coeff.x - l_coeff.y*p_inv_coeff.y);
                Result[row_i*A_D+col_j].y = c_inv_coeff.y - (l_coeff.x*p_inv_coeff.y + l_coeff.y*p_inv_coeff.x);  
            }
        }
    }   
 }

 void cpuMatAlg::matInverseEigen(cuComplex* A, int A_D, cuComplex* Result)
 {
    Eigen::MatrixXcf g(A_D, A_D);
    Eigen::MatrixXcf resultInvA(A_D, A_D);
    g = Eigen::Map<Eigen::MatrixXcf>(reinterpret_cast<std::complex<float>*>(A), A_D, A_D);
    resultInvA = g.inverse();
    for (int col_j = 0; col_j < A_D; col_j++) {
        for (int row_i = 0; row_i < A_D; row_i++) {
            Result[row_i+col_j*A_D].x = resultInvA(row_i, col_j).real();
            Result[row_i+col_j*A_D].y = resultInvA(row_i, col_j).imag();
        }
    }
 }
}