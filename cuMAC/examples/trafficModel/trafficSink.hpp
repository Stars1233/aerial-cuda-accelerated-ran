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

#pragma once

#include "../../src/api.h"

class TrafficSink
{
private:
    cumac::cumacCellGrpUeStatus* api_ue_status;
    int num_flows;
    int drain_rate;
    /* data */
public:
    TrafficSink(cumac::cumacCellGrpUeStatus* api, int num, int rate);
    void Update();
    ~TrafficSink();
};

TrafficSink::TrafficSink(cumac::cumacCellGrpUeStatus* api, int num, int rate) : api_ue_status(api), num_flows(num), drain_rate(rate) {}

void TrafficSink::Update()
{
    for(int i = 0; i<num_flows; i++)
    {
        auto& bufferSize = api_ue_status->bufferSize[i];
        if(bufferSize > drain_rate)
        {
            bufferSize -= drain_rate;
        } else
        {
            bufferSize = 0;
        }
    }
    
}

TrafficSink::~TrafficSink()
{
}
