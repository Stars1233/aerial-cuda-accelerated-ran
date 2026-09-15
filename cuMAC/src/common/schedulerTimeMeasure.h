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

// Centralized definition of cuMAC kernel/CPU time-measurement gating and the
// per-invocation averaging constant. Including this header replaces the
// per-file `constexpr uint16_t numRunSchKnlTimeMsr = 1000;` (or `#define`)
// duplicates that previously lived in every translation unit that wanted
// timing output, and exposes a single canonical build-time switch.
//
// How to enable timing measurement
// --------------------------------
//   - From CMake (preferred):
//         cmake -DCUMAC_ENABLE_KERNEL_TIME_MEASURE=ON ...
//     The cuMAC `src/CMakeLists.txt` translates that option into the
//     `CUMAC_KERNEL_TIME_MEASURE` compile definition.
//   - From the compile line directly:
//         -DCUMAC_KERNEL_TIME_MEASURE
//   - Per-source (legacy, still supported): uncomment the per-file
//     `#define SCHEDULER_KERNEL_TIME_MEASURE_` (or sibling) at the top of
//     the source file you want to instrument.
//
// When `CUMAC_KERNEL_TIME_MEASURE` is defined we additionally turn on every
// per-area legacy gate so the existing `#ifdef ..._TIME_MEASURE_` blocks
// across the codebase activate together; this avoids having to rewrite each
// instrumented site.

#pragma once

#include <cstdint>

#ifdef CUMAC_KERNEL_TIME_MEASURE
#  ifndef SCHEDULER_KERNEL_TIME_MEASURE_
#    define SCHEDULER_KERNEL_TIME_MEASURE_
#  endif
#  ifndef CELLASSOCIATION_KERNEL_TIME_MEASURE_
#    define CELLASSOCIATION_KERNEL_TIME_MEASURE_
#  endif
#  ifndef CPU_SCHEDULER_TIME_MEASURE_
#    define CPU_SCHEDULER_TIME_MEASURE_
#  endif
#endif

namespace cumac {

// Number of repetitions used when timing kernels and CPU paths; the elapsed
// time printed by an instrumented site is divided by this value to yield a
// per-invocation figure.
constexpr uint16_t numRunSchKnlTimeMsr = 1000;

}  // namespace cumac
