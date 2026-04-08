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

#include <vector>
#include <memory>

#include "trafficGenerator.hpp"

class TrafficService
{
public:
    TrafficService(TrafficConfig& config, cumac::cumacCellGrpUeStatus* ue_status)
    {
        generator = std::make_unique<TrafficGenerator>(config);
        radio_rsrc = std::make_unique<RadioResource>(ue_status);
        generator->Attach(radio_rsrc.get());
    }
    TrafficService(TrafficConfig& config, cumac::cumacCellGrpUeStatus* ue_status,cumac::cumacCellGrpUeStatus* ue_status_gpu)
    {
        generator = std::make_unique<TrafficGenerator>(config);
        radio_rsrc = std::make_unique<RadioResourceGpu>(ue_status,ue_status_gpu);
        generator->Attach(radio_rsrc.get());
    }
    void Update(int step=1)
    {
        generator->Generate(step);
        generator->Send();
    }
    void Seed(int seed)
    {
        generator->Seed(seed);
    }
private:
    std::unique_ptr<TrafficGenerator> generator;
    std::unique_ptr<RadioResource> radio_rsrc;
};
