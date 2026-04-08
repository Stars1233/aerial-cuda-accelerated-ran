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

#include "trafficGenerator.hpp"

TrafficConfig::TrafficConfig(){}
TrafficConfig::TrafficConfig(int num_flows){
    AddFlows(TrafficType(),1);
}
TrafficConfig::TrafficConfig(TrafficType traffic,int num_flows){
    AddFlows(traffic,num_flows);
}
TrafficConfig::TrafficConfig(std::vector<TrafficType> traffic_types, std::vector<int> num_flows){
    int num_types = traffic_types.size();
    if(num_flows.empty())
    {
        num_flows = std::vector<int>(num_types,1);
    } else if(num_flows.size() > num_types)
    {
        throw std::invalid_argument(fmt::format("List of flow counts ({}) is longer than traffic types ({})",num_flows.size(),num_types));
    } else if (num_flows.size() < num_types)
    {
        throw std::invalid_argument(fmt::format("List of traffic types ({}) is longer than flow counts ({})",num_types,num_flows.size()));
    }
    for(int i = 0; i < num_types; i++)
    {
        AddFlows(traffic_types[i],num_flows[i]);
    }
    NVLOGD_FMT(NVLOG_TRAFFIC, "Size of num_flows: {}",num_flows.size());
}
TrafficConfig::TrafficConfig(int num_flows, std::vector<double> percents, std::vector<TrafficType> traffic_types){
    if((percents.size() != traffic_types.size()) || percents.empty()){
        throw std::invalid_argument(fmt::format("List of percentages ({}) and list of traffic types ({}) must be non-zero and match",percents.size(),traffic_types.size()));
    }
    std::vector<int> flow_counts{};
    int total = 0;
    for(int i=0;i<percents.size();i++){
        int count = round(percents[i]*num_flows);
        AddFlows(traffic_types[i],count);
        total+=count;
    }
    if(total != num_flows){
        throw std::invalid_argument(fmt::format("Sum of percentages * total flows ({}) does not match total flows ({})",total,num_flows));
    }
}
void TrafficConfig::AddFlows(TrafficType traffic, int num_flows){
    flow_configs.insert(flow_configs.end(),num_flows,traffic);
}
TrafficConfig::~TrafficConfig(){}

