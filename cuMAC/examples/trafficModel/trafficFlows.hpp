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

class FlowData
{
public:
    int flow_id;
    int num_bytes;
    int last_arrival;
};

class FlowType
{
private:
    // Keep statistics about traffic (e.g. arrival times, buffer sizes)
    // Also store QoS config
    int num_bytes;
    int flow_id; // TODO this is never configured
public:
    constexpr static int MAX_BYTES = 1e9;
    void Enqueue(FlowData& flow_data){
        num_bytes += (num_bytes <= MAX_BYTES) * flow_data.num_bytes;
    }
    int MoveBytes(){
        auto tmp = num_bytes;
        num_bytes = 0;
        return tmp;
    }
};


