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

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <thread>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>
#include <CLI/CLI.hpp>

#include "cuphy.hpp"
#include "util.hpp"
// Delay-kernel implementation reference:
// - gpu_us_delay(...) is defined in cuPHY/examples/common/util.cu
// - underlying CUDA kernel is delay_kernel_us in the same file
// Use project CUDA helpers/macros from cuPHY common utilities (e.g., CUDA_CHECK).
// CUDA_CHECK_PRINTF is defined in cuPHY/src/cuphy/common_utils.hpp (available via cuPHY headers).

constexpr int DEFAULT_GPU_ID = 0;
constexpr int DEFAULT_NUM_LAUNCHES = 100;
constexpr int DEFAULT_DELAY_US = 1000;
constexpr int DEFAULT_PERIOD_US = 5000;
constexpr int DEFAULT_REQUESTED_SM_COUNT = 0;
constexpr int LAUNCH_MODE_ALL_BLOCKS = 0;
constexpr int LAUNCH_MODE_SINGLE_BLOCK = 1;

// cuphy::cudaContext is defined in cuPHY/examples/common/cuphy.hpp.
/**
 * @brief Create and bind an optional CUDA subcontext for delay-kernel benchmarking.
 *
 * Creates an MPS programmatic subcontext when @p requested_sm_count is greater than 0,
 * and returns nullptr otherwise. On successful subcontext creation, the applied SM count
 * is written to @p applied_sm_count.
 *
 * @param gpu_id GPU ordinal used for context creation.
 * @param requested_sm_count Requested SM count for subcontext partitioning.
 * @param applied_sm_count Output reference set to the applied SM count by the driver.
 * @return std::unique_ptr<cuphy::cudaContext> Created and bound subcontext, or nullptr.
 */
static std::unique_ptr<cuphy::cudaContext> createDelaySubcontext(int gpu_id, int requested_sm_count, int& applied_sm_count)
{
    if(requested_sm_count <= DEFAULT_REQUESTED_SM_COUNT)
    {
        return nullptr;
    }

#if CUDART_VERSION < 11040
    NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT, "MPS programmatic API requires CUDA 11.4 or newer.");
    std::exit(EXIT_FAILURE);
#else
    auto cu_ctx = std::make_unique<cuphy::cudaContext>();
    try
    {
        cu_ctx->create(gpu_id, requested_sm_count, &applied_sm_count);
        cu_ctx->bind();
    }
    catch(const std::exception& e)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,
                   "Failed to create/bind CUDA subcontext (requested SMs: {}).", requested_sm_count);
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,
                   "This often indicates CUDA MPS is not running or is misconfigured.");
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT, "Manual MPS startup example:");
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT, "  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps");
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT, "  export CUDA_LOG_DIRECTORY=/tmp/nvidia-mps");
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT, "  mkdir -p ${{CUDA_MPS_PIPE_DIRECTORY}}");
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT, "  CUDA_VISIBLE_DEVICES=0 nvidia-cuda-mps-control -d");
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT, "Error detail: {}", e.what());
        throw;
    }
    return cu_ctx;
#endif
}

int main(int argc, char** argv)
{
    const int gpu_id = DEFAULT_GPU_ID;
    int num_launches = DEFAULT_NUM_LAUNCHES;
    int delay_us = DEFAULT_DELAY_US;
    int period_us = DEFAULT_PERIOD_US;
    int launch_mode = LAUNCH_MODE_SINGLE_BLOCK;
    int requested_sm_count = DEFAULT_REQUESTED_SM_COUNT;
    bool enable_event_timing = false;

    CLI::App app{"Delay kernel benchmark"};
    app.footer(
        "Notes:\n"
        "  - NVTX ranges are emitted per delay-kernel launch.\n"
        "  - If -s is provided with value > 0, start CUDA MPS daemon manually before running.\n"
        "    Example:\n"
        "      export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps\n"
        "      export CUDA_LOG_DIRECTORY=/tmp/nvidia-mps\n"
        "      mkdir -p ${CUDA_MPS_PIPE_DIRECTORY}\n"
        "      CUDA_VISIBLE_DEVICES=0 nvidia-cuda-mps-control -d");

    app.add_option("-r,--iterations", num_launches, "Number of kernel launches")
        ->default_val(DEFAULT_NUM_LAUNCHES)
        ->check(CLI::PositiveNumber);
    app.add_option("-d,--delay-us", delay_us, "Delay kernel runtime in microseconds")
        ->default_val(DEFAULT_DELAY_US)
        ->check(CLI::NonNegativeNumber);
    app.add_option("-t,--period-us", period_us, "Launch period from CPU in microseconds")
        ->default_val(DEFAULT_PERIOD_US)
        ->check(CLI::NonNegativeNumber);
    app.add_option("-p,--launch", launch_mode, "Launch mode: 0=multi-SM/all blocks, 1=single block (~1 SM)")
        ->default_val(LAUNCH_MODE_SINGLE_BLOCK)
        ->check(CLI::Range(LAUNCH_MODE_ALL_BLOCKS, LAUNCH_MODE_SINGLE_BLOCK));
    app.add_option("-s,--sm-count", requested_sm_count, "If provided and >0, request MPS subcontext with this SM count")
        ->default_val(DEFAULT_REQUESTED_SM_COUNT)
        ->check(CLI::NonNegativeNumber);
    app.add_flag("-e,--event-timing", enable_event_timing, "Enable CUDA event timing summary (min/mean/max kernel duration)");

    try
    {
        app.parse(argc, argv);
    }
    catch(const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    CUDA_CHECK(cudaSetDevice(gpu_id));

    int applied_sm_count = 0;
    std::unique_ptr<cuphy::cudaContext> subctx;
    try
    {
        subctx = createDelaySubcontext(gpu_id, requested_sm_count, applied_sm_count);
    }
    catch(const std::exception&)
    {
        return EXIT_FAILURE;
    }

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "Delay kernel benchmark configuration:");
    NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "  iterations: {}", num_launches);
    NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "  delay_us: {}", delay_us);
    NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "  period_us: {}", period_us);
    NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "  launch_mode: {}{}",
               launch_mode,
               (launch_mode == LAUNCH_MODE_ALL_BLOCKS ? " (multi-SM/all blocks)" : " (single block/~1 SM)"));
    NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "  mps_subctx_sm_count: {}", requested_sm_count);
    NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "  event_timing: {}", (enable_event_timing ? "enabled" : "disabled"));
    if(enable_event_timing)
    {
        NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY,
                   "  NOTE: --event-timing uses per-launch cudaEventSynchronize, which serializes each launch and can increase observed launch spacing.");
    }

    if(subctx)
    {
        NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "  applied_subctx_sm_count: {}", applied_sm_count);
    }
    else
    {
        NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "  applied_subctx_sm_count: none (primary context)");
    }

    const bool single_thrd_blk = (launch_mode == LAUNCH_MODE_SINGLE_BLOCK);
    cudaEvent_t evt_start = nullptr;
    cudaEvent_t evt_stop = nullptr;
    double kernel_ms_sum = 0.0;
    float kernel_ms_min = std::numeric_limits<float>::max();
    float kernel_ms_max = 0.0f;

    if(enable_event_timing)
    {
        CUDA_CHECK(cudaEventCreateWithFlags(&evt_start, cudaEventDefault));
        CUDA_CHECK(cudaEventCreateWithFlags(&evt_stop, cudaEventDefault));
    }

    using steady_clock = std::chrono::steady_clock;
    const auto start_time = steady_clock::now();
    steady_clock::time_point end_time;

    {
        nvtx3::scoped_range bench_range{"delay_kernel_benchmark"};
        for(int i = 0; i < num_launches; ++i)
        {
            nvtx3::scoped_range launch_range{"gpu_us_delay_launch"};
            if(enable_event_timing)
            {
                CUDA_CHECK(cudaEventRecord(evt_start, stream));
            }
            gpu_us_delay(static_cast<uint32_t>(delay_us), gpu_id, stream, single_thrd_blk);
            if(enable_event_timing)
            {
                CUDA_CHECK(cudaEventRecord(evt_stop, stream));
                CUDA_CHECK(cudaEventSynchronize(evt_stop));
                float kernel_ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, evt_start, evt_stop));
                kernel_ms_sum += static_cast<double>(kernel_ms);
                kernel_ms_min = std::min(kernel_ms_min, kernel_ms);
                kernel_ms_max = std::max(kernel_ms_max, kernel_ms);
            }

            if(period_us > 0)
            {
                const auto next_launch_time = start_time + std::chrono::microseconds(static_cast<int64_t>(i + 1) * period_us);
                std::this_thread::sleep_until(next_launch_time);
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        end_time = steady_clock::now();
    }

    const auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    CUDA_CHECK(cudaStreamDestroy(stream));
    if(enable_event_timing)
    {
        CUDA_CHECK(cudaEventDestroy(evt_start));
        CUDA_CHECK(cudaEventDestroy(evt_stop));
    }

    NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "Completed {} launches in {} us.", num_launches, total_us);
    NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "Average launch spacing observed: {} us.",
               (static_cast<double>(total_us) / num_launches));
    if(enable_event_timing)
    {
        const double kernel_ms_mean = kernel_ms_sum / static_cast<double>(num_launches);
        NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "Kernel duration (CUDA events): min={} ms, mean={} ms, max={} ms.",
                   kernel_ms_min, kernel_ms_mean, kernel_ms_max);
    }

    return EXIT_SUCCESS;
}
