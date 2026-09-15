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

#pragma once

////////////////////////////////////////////////////////////////////////
// ldpc_et_context
// Context for early-termination CRC computation.
template <typename PartialCrcT>
struct ldpc_et_context
{
    static constexpr int MAX_NUM_WORDS = 264; // 8448 / 32
    static constexpr int MAX_NUM_PARTIAL_CRCS = 12; // 384/32
    PartialCrcT partial_crcs[MAX_NUM_PARTIAL_CRCS];
    uint32_t two_k_mod_p[MAX_NUM_WORDS];
};

// One-CW kernels reduce one CRC per warp. Two-CW kernels reduce the packed
// codewords together and therefore need one pair of CRCs per warp.
using ldpc_et_context_t = ldpc_et_context<uint32_t>;
using ldpc_et_context_x2_t = ldpc_et_context<uint2>;
