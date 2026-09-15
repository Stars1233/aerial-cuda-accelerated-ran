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

#ifndef _TEST_CUDA_H_
#define _TEST_CUDA_H_

#include <stdio.h>

#if defined(__cplusplus)
extern "C" {
#endif

#define LOWER_CASE_SUB_VAL ((char) ('A' - 'a'))

static inline void cpu_to_lower_case(char* str, int length)
{
    for(int i = 0; i < length; i++)
    {
        if(str[i] >= 'A' && str[i] <= 'Z')
        {
            str[i] -= LOWER_CASE_SUB_VAL;
        }
    }
}

void cuda_to_lower_case(char* str, int length, int deviceId);

void test_cuda_to_lower_case(int deviceId, char* str, int length, int gpu);

// Must be called once from the main thread after CUDA context is active.
// Returns 0 on success, -1 on failure.
__attribute__((warn_unused_result)) int init_test_cuda_kernels_c(void);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#ifdef __cplusplus
[[nodiscard]] bool init_test_cuda_kernels();
#endif

#endif /* _TEST_CUDA_H_ */
