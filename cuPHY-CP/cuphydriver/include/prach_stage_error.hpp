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

#ifndef PRACH_STAGE_ERROR_HPP
#define PRACH_STAGE_ERROR_HPP

/**
 * Typed failure reason for the offload (`nonslot_lp`) PRACH reconfiguration staging
 * family (`PhyPrachAggr::stageConfig`/`createNewPhyObjFromStagedConfig` and
 * `PrachOffloadReconfig::stageCellConfig`/`createObjects`). Carried by the
 * `tl::expected<..., PrachStageError>` returns so a failure names its cause instead of a
 * bare `false`.
 *
 * Only the failure points the real bodies can actually hit are enumerated. The publish
 * step (`commitStagedConfig`/`commit`) is `void`/infallible (it swaps already-built,
 * already-validated buffers), so there is no commit-failed variant.
 *
 * The cause is consumed, not just recorded: `PrachOffloadReconfig::reconfigure()` maps it
 * to a `PrachReconfigOutcome` via `toReconfigOutcome()`, so the caller-facing outcome
 * follows the actual failure cause rather than which call happened to fail.
 */
enum class PrachStageError
{
    StageRejected,    //!< stageConfig rejected the update: invalid PRACH params or cell not present (applyPrachUpdate failed).
    TempCreateFailed, //!< createNewPhyObjFromStagedConfig failed: cuphyCreatePrachRx could not build the temp (PONG) handle.
};

#endif // PRACH_STAGE_ERROR_HPP
