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

#if !defined(LDPC_CLAMP_VALIDATION_HPP_INCLUDED_)
#define LDPC_CLAMP_VALIDATION_HPP_INCLUDED_

#include "cuphy.h"

namespace ldpc_example
{
/**
 * Returns the largest finite clamp value accepted for an LLR data type.
 *
 * @param[in] llr_type cuPHY data type used for LLR values.
 * @return Largest finite clamp value, or 0.0F for an unsupported type.
 */
[[nodiscard]] inline constexpr float max_finite_clamp_value(cuphyDataType_t llr_type)
{
    switch(llr_type)
    {
    case CUPHY_R_8F_E4M3: return CUPHY_FP8_E4M3_MAX_FINITE;
    case CUPHY_R_8F_E5M2: return CUPHY_FP8_E5M2_MAX_FINITE;
    case CUPHY_R_16F:
    case CUPHY_R_32F:     return 65504.0f;
    default:              return 0.0f;
    }
}

/**
 * Checks whether a clamp value is representable for an LLR data type.
 *
 * @param[in] llr_type    cuPHY data type used for LLR values.
 * @param[in] clamp_value Clamp magnitude to validate.
 * @return True when \p clamp_value is positive and below the type's finite limit; false for unsupported
 *         LLR types or values that are non-positive or at or above the finite limit.
 */
[[nodiscard]] inline constexpr bool is_valid_clamp_value(cuphyDataType_t llr_type, float clamp_value)
{
    return (clamp_value > 0.0f) && (clamp_value < max_finite_clamp_value(llr_type));
}
} // namespace ldpc_example

#endif // LDPC_CLAMP_VALIDATION_HPP_INCLUDED_
