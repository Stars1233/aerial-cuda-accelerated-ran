/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

// Recommended defaults: 2GiB buffer size with 2 buffers. We buffer up as much data as possible and dump
// results to log.  If run is appropriate length the log dump will occur after run is complete, minimizing impact
// to the system.
// Note: Currently CUPTI limits to 2GiB per buffer.  This will likely be fixed in future CUPTI versions.
void cuphy_cupti_helper_init(uint64_t buffer_size = 2ULL * 1024 * 1024 * 1024, uint16_t num_buffers = 2);
void cuphy_cupti_helper_stop();
void cuphy_cupti_helper_flush();
void cuphy_cupti_helper_push_external_id(uint64_t id);
void cuphy_cupti_helper_pop_external_id();

class CuphyCuptiScopedExternalId
{
public:
    CuphyCuptiScopedExternalId(uint64_t id)
    {
        cuphy_cupti_helper_push_external_id(id);
    }

    ~CuphyCuptiScopedExternalId()
    {
        cuphy_cupti_helper_pop_external_id();
    }
};

#if defined(__cplusplus)
} /* extern "C" */
#endif /* defined(__cplusplus) */
