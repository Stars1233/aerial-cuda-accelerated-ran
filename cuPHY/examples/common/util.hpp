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

#if !defined(UTIL_HPP_INCLUDED_)
#define UTIL_HPP_INCLUDED_

#include "cuphy.h"

/**
 * Launch a GPU delay kernel for a specified number of milliseconds.
 * @param[in] delay_ms Delay duration in milliseconds.
 * @param[in] gpuId Target GPU index to run the delay on (default: 0).
 * @param[in] cuStrm CUDA stream used to launch the delay kernel (default: 0 / default stream).
 */
void gpu_ms_delay(uint32_t delay_ms, int gpuId = 0, cudaStream_t cuStrm = 0);
/**
 * Launch a GPU delay kernel for a specified number of microseconds.
 * @param[in] delay_us Delay duration in microseconds.
 * @param[in] gpuId Target GPU index to run the delay on (default: 0).
 * @param[in] cuStrm CUDA stream used to launch the delay kernel (default: 0 / default stream).
 * @param[in] singleThrdBlk If true, force a single-threaded block for serialized timing; if false, use the default kernel configuration (default: false).
 */
void gpu_us_delay(uint32_t delay_us, int gpuId = 0, cudaStream_t cuStrm = 0, bool singleThrdBlk = false);
/**
 * Sleep on GPU for a specified number of milliseconds.
 * @param[in] sleep_ms Sleep duration in milliseconds.
 * @param[in] gpuId Target GPU index to run the sleep on (default: 0).
 * @param[in] cuStrm CUDA stream used to launch the sleep kernel (default: 0 / default stream).
 */
void gpu_ms_sleep(uint32_t sleep_ms, int gpuId = 0, cudaStream_t cuStrm = 0);
/**
 * Busy-wait on GPU until start_time_d + time_offset_ns is reached.
 * @param[in] start_time_d Device pointer to the reference start timestamp.
 * @param[in] time_offset_ns Offset in nanoseconds from the reference timestamp.
 * @param[in] cuStrm CUDA stream used to launch the wait kernel.
 */
void gpu_ns_delay_until(uint64_t* start_time_d, uint64_t time_offset_ns, cudaStream_t cuStrm);
/**
 * Launch an empty GPU kernel on the specified stream.
 * @param[in] cuStrm CUDA stream used to launch the kernel (default: 0 / default stream).
 */
void gpu_empty_kernel(cudaStream_t cuStrm = 0);
/**
 * Collect SM IDs used by the delay kernel launch.
 * @param[in] gpuId Target GPU index to run the query on.
 * @param[out] pSmIds Device/output buffer to store collected SM IDs.
 * @param[in] smIdsCnt Number of SM IDs requested in pSmIds.
 * @param[in] cuStrm CUDA stream used to launch the query kernel (default: 0 / default stream).
 * @param[in] delay_us Delay duration in microseconds used before reading SM IDs (default: 1000).
 */
void get_sm_ids(int gpuId, uint32_t* pSmIds, uint32_t smIdsCnt, cudaStream_t cuStrm = 0, uint32_t delay_us = 1000);
/**
 * Read the current GPU global timer into the provided buffer.
 * @param[out] ptimer_d Device/output pointer that receives the GPU timestamp.
 * @param[in] cuStrm CUDA stream used to launch the timer read kernel.
 */
void get_gpu_time(uint64_t *ptimer_d, cudaStream_t cuStrm);

#endif // !defined(UTIL_HPP_INCLUDED_)
