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

#ifndef CUPHY_CP_TESTS_PARSER_EQUIVALENCE_HARNESS_HPP_
#define CUPHY_CP_TESTS_PARSER_EQUIVALENCE_HARNESS_HPP_

/**
 * @file parser_equivalence_harness.hpp
 * @brief Channel-agnostic scaffolding for parser equivalence tests and benchmarks.
 *
 * Provides two reusable pieces shared by every per-channel parser test
 * (PUSCH, PUCCH, PRACH, SRS, ...) and by the matching benchmarks:
 *
 *   - Data-asset resolution: locate the testMAC config YAML and a launch
 *     pattern relative to the cuBB root (env override first, then the shared
 *     get_full_path_file() helper).
 *   - @ref TestMacSlotSession: an RAII owner of a constructed @c test_mac that
 *     has already loaded a launch pattern and prebuilt its FAPI messages, and
 *     exposes bounds-safe iteration over the prebuilt per-slot messages.
 *
 * This header is intentionally GTest- and benchmark-free so it can back both
 * front-ends.
 */

#include <array>
#include <cstdint>
#include <cstdio>   // std::FILE / stderr for fmt::print diagnostics
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <system_error>
#include <utility>

#include <fmt/core.h>  // fmt::print — project formatting convention (libfmt, not std::print)

#include <cuda.h>  // CUDA driver API: cuInit / primary-context retain (prebuild needs a CUDA context)

#include "test_mac.hpp"
#include "common_utils.hpp"  // CONFIG_* path macros, MAX_PATH_LEN
#include "nvlog.hpp"         // get_full_path_file()

#include "scf_5g_fapi.h"
#include "nv_fapi_pdu_utils.hpp"  // nv::phy_mac_msg_desc

namespace cuphy_cp::tests
{

// ---------------------------------------------------------------------------
// Data-asset resolution
//
// Each asset resolves in two steps:
//   1. an explicit env-var override (custom CI vectors), else
//   2. the in-tree default, located relative to the cuBB root via the shared
//      get_full_path_file() helper -- identical to how the DLC test bench and
//      cuphycontroller locate config / launch-pattern files. The helper honours
//      the cuBB_SDK env var, then falls back to an exe-relative walk.
// A resolved path that does not exist on disk is the caller's cue to GTEST_SKIP
// (functional test) or skip-with-error (benchmark).
// ---------------------------------------------------------------------------

/// @brief Resolve @p file_name under @p relative_dir relative to the cuBB root
///        using the shared path helper.
/// @param[in] relative_dir Directory (relative to the cuBB root) to search.
/// @param[in] file_name    File name to resolve within @p relative_dir.
/// @return The resolved absolute path, or an empty path if resolution fails.
[[nodiscard]] inline std::filesystem::path resolve_under_cubb_root(const char* relative_dir,
                                                                   const char* file_name)
{
    std::array<char, MAX_PATH_LEN> buf{};
    const int length =
        get_full_path_file(buf.data(), relative_dir, file_name, CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
    if (length <= 0)
    {
        return {};
    }
    return std::filesystem::path{buf.data()};
}

/// @brief Resolve the testMAC application config YAML used to construct @c test_mac.
/// @param[in] env_override Name of an env var that, if set, overrides the path.
/// @return The resolved YAML path (env override or in-tree default).
[[nodiscard]] inline std::filesystem::path resolve_testmac_yaml(const char* env_override)
{
    if (const char* env = std::getenv(env_override); env != nullptr && env[0] != '\0')
    {
        return std::filesystem::path{env};
    }
    return resolve_under_cubb_root(CONFIG_TESTMAC_YAML_PATH, CONFIG_TESTMAC_YAML_NAME);
}

/// @brief Resolve the channel-bearing launch pattern YAML.
/// @param[in] env_override    Name of an env var that, if set, overrides the path.
/// @param[in] default_pattern In-tree default launch-pattern file name.
/// @return The resolved launch-pattern path (env override or in-tree default).
/// @note Only the file name is handed to test_mac::load_launch_pattern (it
///       re-resolves under CONFIG_LAUNCH_PATTERN_PATH), so an env override must
///       name a pattern that lives in testVectors/multi-cell/.
[[nodiscard]] inline std::filesystem::path resolve_launch_pattern(const char* env_override,
                                                                  const char* default_pattern)
{
    if (const char* env = std::getenv(env_override); env != nullptr && env[0] != '\0')
    {
        return std::filesystem::path{env};
    }
    return resolve_under_cubb_root(CONFIG_LAUNCH_PATTERN_PATH, default_pattern);
}

// ---------------------------------------------------------------------------
// CudaPrimaryContext
//
// testMAC's prebuild path stages FAPI data buffers through the CUDA *driver*
// API (cuMemAllocHost). The driver API -- unlike the runtime API, which inits
// lazily -- requires an explicit cuInit(0) plus a context current on the
// calling thread; without one the first cuMemAllocHost fails with
// CUDA_ERROR_NOT_INITIALIZED ("initialization error"). The full L1/testMAC
// application initialises CUDA during transport bring-up, but this standalone
// harness must do it itself -- exactly as the DLC test bench does
// (cuPHY-CP/tests/dlc_test_bench/sendCplaneUnitTest.cpp).
// ---------------------------------------------------------------------------

/**
 * @brief RAII owner of a GPU's CUDA driver primary context.
 *
 * Acquired via @ref try_acquire, which runs cuInit -> cuDeviceGet ->
 * cuDevicePrimaryCtxRetain -> cuCtxSetCurrent and returns @c std::nullopt
 * (rather than aborting) when no usable driver/GPU is present, so GPU-less CI
 * skips cleanly. The retained primary context stays current on the acquiring
 * thread for this object's lifetime and is released on destruction.
 *
 * @thread_safety Not thread-safe; acquire and use on a single thread.
 */
class CudaPrimaryContext final
{
public:
    CudaPrimaryContext(const CudaPrimaryContext&)            = delete;
    CudaPrimaryContext& operator=(const CudaPrimaryContext&) = delete;

    CudaPrimaryContext(CudaPrimaryContext&& other) noexcept
        : device_{other.device_}, active_{other.active_}
    {
        other.active_ = false;
    }

    CudaPrimaryContext& operator=(CudaPrimaryContext&& other) noexcept
    {
        if (this != &other)
        {
            release();
            device_       = other.device_;
            active_       = other.active_;
            other.active_ = false;
        }
        return *this;
    }

    ~CudaPrimaryContext() { release(); }

    /**
     * @brief Initialise the driver API and retain+bind @p gpu_id's primary context.
     * @param[in] gpu_id Device ordinal (within CUDA_VISIBLE_DEVICES) to bind.
     * @return A live context bound to the calling thread, or @c std::nullopt if
     *         the CUDA driver or the requested GPU is unavailable.
     */
    [[nodiscard]] static std::optional<CudaPrimaryContext> try_acquire(int gpu_id)
    {
        if (const CUresult rc = cuInit(0); rc != CUDA_SUCCESS)
        {
            log_failure("cuInit", rc);
            return std::nullopt;
        }
        CUdevice device{};
        if (const CUresult rc = cuDeviceGet(&device, gpu_id); rc != CUDA_SUCCESS)
        {
            log_failure("cuDeviceGet", rc);
            return std::nullopt;
        }
        CUcontext context{};
        if (const CUresult rc = cuDevicePrimaryCtxRetain(&context, device); rc != CUDA_SUCCESS)
        {
            log_failure("cuDevicePrimaryCtxRetain", rc);
            return std::nullopt;
        }
        if (const CUresult rc = cuCtxSetCurrent(context); rc != CUDA_SUCCESS)
        {
            log_failure("cuCtxSetCurrent", rc);
            cuDevicePrimaryCtxRelease(device);
            return std::nullopt;
        }
        return CudaPrimaryContext{device};
    }

private:
    explicit CudaPrimaryContext(CUdevice device) noexcept : device_{device}, active_{true} {}

    void release() noexcept
    {
        if (active_)
        {
            cuDevicePrimaryCtxRelease(device_);
            active_ = false;
        }
    }

    static void log_failure(const char* api, CUresult rc) noexcept
    {
        const char* err_str = nullptr;
        cuGetErrorString(rc, &err_str);
        fmt::print(stderr, "[parser_equivalence] {} failed: {}\n", api,
                   err_str != nullptr ? err_str : "unknown CUDA error");
    }

    CUdevice device_{};
    bool     active_{false};
};

// ---------------------------------------------------------------------------
// TestMacSlotSession
// ---------------------------------------------------------------------------

/**
 * @brief RAII owner of a @c test_mac that has loaded a launch pattern and
 *        prebuilt its FAPI messages, with iteration over the prebuilt slots.
 *
 * Construct via @ref try_open, which performs the construct -> load -> prebuild
 * sequence and returns @c std::nullopt (rather than throwing) when any step
 * fails or the data assets are unavailable. The owned @c test_mac -- and hence
 * every message buffer handed to @ref for_each_slot_message -- stays valid for
 * the lifetime of the session.
 *
 * @thread_safety Not thread-safe; intended for single-threaded test/benchmark
 *                setup and iteration.
 */
class TestMacSlotSession final
{
public:
    TestMacSlotSession(const TestMacSlotSession&)            = delete;
    TestMacSlotSession& operator=(const TestMacSlotSession&) = delete;
    TestMacSlotSession(TestMacSlotSession&&) noexcept        = default;
    TestMacSlotSession& operator=(TestMacSlotSession&&) noexcept = default;
    ~TestMacSlotSession()                                    = default;

    /**
     * @brief Construct a session: build @c test_mac, load the pattern, prebuild.
     * @param[in] testmac_yaml   Path to the testMAC config YAML.
     * @param[in] launch_pattern Path to the launch-pattern YAML.
     * @param[in] cell_mask      Bitmask of cells to activate.
     * @param[in] channel_mask   Bitmask of channels to schedule.
     * @return A ready session, or std::nullopt if assets are missing or any
     *         construction/load/prebuild step fails.
     */
    [[nodiscard]] static std::optional<TestMacSlotSession>
    try_open(const std::filesystem::path& testmac_yaml,
             const std::filesystem::path& launch_pattern,
             uint64_t                     cell_mask,
             uint32_t                     channel_mask)
    {
        // Diagnostics go to stderr (not GTEST_SKIP) so the exact failing step is
        // visible when the binary is run directly; ctest hides skipped output.
        if (testmac_yaml.empty() || !std::filesystem::exists(testmac_yaml))
        {
            fmt::print(stderr, "[parser_equivalence] testMAC config YAML not found: '{}'\n",
                       testmac_yaml.string());
            return std::nullopt;
        }
        if (launch_pattern.empty() || !std::filesystem::exists(launch_pattern))
        {
            fmt::print(stderr, "[parser_equivalence] launch pattern not found: '{}'\n",
                       launch_pattern.string());
            return std::nullopt;
        }

        fmt::print(stderr,
                   "[parser_equivalence] opening test_mac\n  config='{}'\n  pattern='{}'\n"
                   "  cell_mask=0x{:x} channel_mask=0x{:x}\n",
                   testmac_yaml.string(), launch_pattern.string(), cell_mask, channel_mask);

        // testMAC's prebuild stages buffers via the CUDA driver API
        // (cuMemAllocHost), which needs an initialised driver and a current
        // context on this thread. Acquire it up front so the failure surfaces
        // here (as a clean skip) rather than as a cryptic cuMemAllocHost error.
        std::optional<CudaPrimaryContext> cuda_ctx = CudaPrimaryContext::try_acquire(kGpuId);
        if (!cuda_ctx.has_value())
        {
            fmt::print(stderr,
                       "[parser_equivalence] CUDA initialisation failed; cannot prebuild FAPI "
                       "messages (testMAC stages buffers via cuMemAllocHost)\n");
            return std::nullopt;
        }

        try
        {
            auto mac = std::make_unique<test_mac>(testmac_yaml.string().c_str());
            // test_mac::load_launch_pattern() re-resolves the name under
            // CONFIG_LAUNCH_PATTERN_PATH via get_full_path_file() (see
            // launch_pattern::launch_pattern_parsing), so it must be given the
            // bare file name -- passing the already-absolute path doubles it.
            const std::string lp_name = launch_pattern.filename().string();

            // Because only the bare file name reaches load_launch_pattern, an env
            // override pointing outside CONFIG_LAUNCH_PATTERN_PATH would be
            // silently ignored (the in-tree file of the same name is loaded
            // instead). Verify the override IS that in-tree file and fail loudly
            // otherwise, rather than test a different vector than requested.
            const std::filesystem::path in_tree =
                resolve_under_cubb_root(CONFIG_LAUNCH_PATTERN_PATH, lp_name.c_str());
            std::error_code ec;
            if (in_tree.empty() || !std::filesystem::equivalent(in_tree, launch_pattern, ec) || ec)
            {
                fmt::print(stderr,
                           "[parser_equivalence] launch-pattern override '{}' is not the in-tree file "
                           "load_launch_pattern would resolve ('{}'); only patterns under "
                           "CONFIG_LAUNCH_PATTERN_PATH are supported\n",
                           launch_pattern.string(), in_tree.string());
                return std::nullopt;
            }

            if (const int rc = mac->load_launch_pattern(lp_name.c_str(), cell_mask, channel_mask); rc < 0)
            {
                fmt::print(stderr, "[parser_equivalence] load_launch_pattern('{}') failed rc={}\n",
                           lp_name, rc);
                return std::nullopt;
            }
            if (const int rc = mac->prebuild_fapi_messages(); rc < 0)
            {
                fmt::print(stderr, "[parser_equivalence] prebuild_fapi_messages() failed rc={}\n", rc);
                return std::nullopt;
            }
            return TestMacSlotSession{std::move(*cuda_ctx), std::move(mac)};
        }
        catch (const std::exception& e)
        {
            fmt::print(stderr, "[parser_equivalence] test_mac setup threw: {}\n", e.what());
            return std::nullopt;
        }
        catch (...)
        {
            // A non-std throw would otherwise unwind out of try_open and abort the
            // whole test/benchmark process; turn it into a clean skip instead.
            fmt::print(stderr, "[parser_equivalence] test_mac setup threw a non-std exception\n");
            return std::nullopt;
        }
    }

    /// @brief Number of scheduled slots in the loaded launch pattern.
    /// @return Scheduled slot count, or 0 if no launch pattern is loaded.
    [[nodiscard]] int scheduled_slot_count() const
    {
        launch_pattern* lp = mac_->get_launch_pattern();
        return (lp != nullptr) ? lp->get_sched_slot_num() : 0;
    }

    /**
     * @brief Invoke @p fn for every prebuilt message of @p cell_id across all
     *        scheduled slots.
     * @param[in] cell_id Cell whose prebuilt messages to walk.
     * @param[in] fn      Callable invoked as fn(sfn_slot_t, const nv::phy_mac_msg_desc&).
     *
     * Iterates slot indices [0, scheduled_slot_count()) and maps each to an
     * SFN/slot at numerology mu=1 (20 slots per frame). Callers filter by
     * message id and walk PDUs via the production bounds-safe helpers.
     */
    template <typename Fn>
    void for_each_slot_message(int cell_id, Fn&& fn) const
    {
        const int slots = scheduled_slot_count();
        for (const int slot_idx : std::views::iota(0, slots))
        {
            sfn_slot_t ss{};
            ss.u16.sfn  = static_cast<uint16_t>(slot_idx / kSlotsPerFrameMu1);
            ss.u16.slot = static_cast<uint16_t>(slot_idx % kSlotsPerFrameMu1);

            const std::span<const nv::phy_mac_msg_desc> msgs =
                mac_->get_prebuilt_slot_messages(cell_id, ss);
            for (const nv::phy_mac_msg_desc& msg : msgs)
            {
                fn(ss, msg);
            }
        }
    }

private:
    TestMacSlotSession(CudaPrimaryContext cuda_ctx, std::unique_ptr<test_mac> mac) noexcept
        : cuda_ctx_{std::move(cuda_ctx)}, mac_{std::move(mac)}
    {
    }

    /// Device ordinal (within CUDA_VISIBLE_DEVICES) the harness binds for prebuild.
    static constexpr int kGpuId = 0;
    /// Slots per radio frame at numerology mu=1 (30 kHz SCS).
    static constexpr int kSlotsPerFrameMu1 = 20;

    /// Declared before mac_ so the primary context outlives the test_mac (whose
    /// teardown frees CUDA host buffers): members destruct in reverse order.
    CudaPrimaryContext        cuda_ctx_;
    std::unique_ptr<test_mac> mac_{};
};

} // namespace cuphy_cp::tests

#endif // CUPHY_CP_TESTS_PARSER_EQUIVALENCE_HARNESS_HPP_
