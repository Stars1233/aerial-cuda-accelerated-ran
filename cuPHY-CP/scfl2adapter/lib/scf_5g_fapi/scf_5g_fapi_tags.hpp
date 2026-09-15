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

#if !defined(SCF_5G_FAPI_TAGS_HPP_INCLUDED_)
#define SCF_5G_FAPI_TAGS_HPP_INCLUDED_

#include "nvlog.h"
#include "nvlog_fmt.hpp"

/**
 * @file scf_5g_fapi_tags.hpp
 * @brief Shared nvlog tag constants for the scf_5g_fapi layer.
 *
 * All slot-processing, TTI-dispatch, and PDU-parser headers use
 * @c scf_5g_fapi::detail::k_tag (NVLOG_TAG_BASE_SCF_L2_ADAPTER + 5,
 * "SCF.L2ADAPTER").  Centralised here to avoid per-TU anonymous-namespace
 * copies and make tag value changes a single-line edit.
 */
namespace scf_5g_fapi::detail {

/** nvlog tag for the scf_5g_fapi slot-processing / PDU-parsing layer. */
inline constexpr uint16_t k_tag = NVLOG_TAG_BASE_SCF_L2_ADAPTER + 5; // "SCF.L2ADAPTER"

} // namespace scf_5g_fapi::detail

#endif // SCF_5G_FAPI_TAGS_HPP_INCLUDED_
