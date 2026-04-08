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

#include <vector>
#include <utility>
#include <unordered_map>
#include <memory>
#include <list>
#include <random>
#include <fmt/core.h>
#include "nvlog.hpp"

#define NVLOG_TRAFFIC NVLOG_TAG_BASE_CUMAC + 10

#include "api.h"
#include "cumac.h"
#include "trafficFlows.hpp"

// Structure to hold parameter pairs (mean, stddev) to avoid std::pair ABI issues
struct TrafficParams
{
    float mean;
    float stddev;
};

enum class Arrival_t {uniform, poisson, full_buffer};

class TrafficType
{
private:
// Keep traffic parameters for generator
   float     arrival_rate; //!< Packet arrival rate in packets per TTI
   float     arrival_stddev;  //!< Packet arrival stddev
   Arrival_t arrival_type; //!< Specify arrival type (e.g. uniform, random, full buffer)
   float     packet_size;  //!< Mean packet size in bytes
   float     packet_stddev;   //!< Packet size stddev

public:
    TrafficParams GetPktSizeParams(){
        return {packet_size, packet_stddev};
    }
    TrafficParams GetArrivalParams(){
        return {arrival_rate, arrival_stddev};
    }
    Arrival_t GetArrivalType(){
        return arrival_type;
    }
    TrafficType(){
        arrival_type = Arrival_t::full_buffer;
    }
    TrafficType(float pkt_mean, float pkt_stddev, float rate){
        arrival_type = Arrival_t::poisson;
        packet_size = pkt_mean;
        packet_stddev = pkt_stddev;
        arrival_rate = rate;
    }
    TrafficType(Arrival_t type, float mean, float stddev, float rate)
    {
        arrival_type = type;
        packet_size = mean;
        packet_stddev = stddev;
        arrival_rate = rate;
    }
    ~TrafficType(){
        }
};



class TrafficConfig
{
private:
    std::vector<TrafficType> flow_configs;
    
public:

    /**
     * @brief Constructs an empty TrafficConfig.
     */
    TrafficConfig();

    /**
     * @brief Constructs a TrafficConfig with a specified number of UEs and default traffic type.
     * 
     * @param num_flows The number of UE flows.
     */
    TrafficConfig(int num_flows);

    /**
     * @brief Constructs a TrafficConfig with a specified traffic type and number of UEs.
     * 
     * @param traffic The traffic type.
     * @param num_flows The number of UEs.
     */
    TrafficConfig(TrafficType traffic, int num_flows);

    /**
     * @brief Constructs a TrafficConfig with specified traffic types and number of corresponding flows for each.
     * 
     * @param traffic_types A vector of traffic types.
     * @param num_flows A vector of the number of UEs for each traffic type. An empty vector for \p num_flows assumes 1 flow each
     */
    TrafficConfig(std::vector<TrafficType> traffic_types, std::vector<int> num_flows = {});

    /**
     * @brief Constructs a TrafficConfig with a specified number of flows and percentages of each flow type.
     * 
     * @param num_flows The number of flows.
     * @param percents A vector of percentages for each flow type.
     * @param traffic_types A vector of traffic types.
     */
    TrafficConfig(int num_flows, std::vector<double> percents, std::vector<TrafficType> traffic_types); // Specify number of flows and percentages of each flow type

    /**
     * @brief Adds \p num_flows of a specified traffic type to a TrafficConfig
     * 
     * @param traffic The traffic type to add to the traffic config
     * @param num_flows A number of UE flows to add of a specify traffic type. The default \p num_flows assumes 1 flow
     */
    void AddFlows(TrafficType traffic, int num_flows=1);
    const std::vector<TrafficType> GetFlowCfgs(){return flow_configs;}
    ~TrafficConfig();
};


class TrafficObserver
{
protected:
    std::vector<FlowType> traffic_flows;
public:
    virtual void Configure(std::vector<FlowData>& config){
        traffic_flows.resize(config.size());
        NVLOGI_FMT(NVLOG_TRAFFIC, "Configuring Observer for {} flows",config.size());
    }
    virtual void Enqueue(std::vector<FlowData>& data){
        // For now assume traffic_flows is indexed by flow_id
        for(auto& flow_data : data){
            traffic_flows[flow_data.flow_id].Enqueue(flow_data);
            NVLOGD_FMT(NVLOG_TRAFFIC, "Flow ID {} euqueued with {} bytes",flow_data.flow_id,flow_data.num_bytes);
        }
    }
};


class RadioResource : public TrafficObserver
{
protected:
    cumac::cumacCellGrpUeStatus* api_ue_status;
public:
    RadioResource(cumac::cumacCellGrpUeStatus* ue_status){
        api_ue_status = ue_status;
    }
    virtual void UpdateApi(){
        int max_idx = -1;
        uint32_t max_size = 0;
        for(int i = 0; i < traffic_flows.size(); i++)
        {
            auto& idx_buffer_size = api_ue_status->bufferSize[i];
            idx_buffer_size += traffic_flows[i].MoveBytes();
            NVLOGD_FMT(NVLOG_TRAFFIC, "Flow {} has {} bytes",i,idx_buffer_size);
  
            if(idx_buffer_size > max_size)
            {
                max_size = idx_buffer_size;
                max_idx = i;
            }
        }
        NVLOGD_FMT(NVLOG_TRAFFIC, "Largest Flow {} with {} bytes",max_idx,max_size);
    }
    void Enqueue(std::vector<FlowData>& data){
        TrafficObserver::Enqueue(data);
        UpdateApi();
    }
};

class RadioResourceGpu : public RadioResource{
private:
    cumac::cumacCellGrpUeStatus* api_ue_status_gpu;
public:
    RadioResourceGpu(cumac::cumacCellGrpUeStatus* ue_status_cpu, cumac::cumacCellGrpUeStatus* ue_status_gpu) : RadioResource(ue_status_cpu) {
    api_ue_status_gpu = ue_status_gpu;
}
void SetGpuApi(cumac::cumacCellGrpUeStatus* api){
    api_ue_status_gpu = api;
}
void UpdateApi(){
    RadioResource::UpdateApi();
    CUDA_CHECK_ERR(cudaMemcpy(api_ue_status_gpu->bufferSize, api_ue_status->bufferSize, traffic_flows.size()*sizeof(uint32_t), cudaMemcpyHostToDevice));
}

};


class TrafficGenerator
{
private:
    std::vector<TrafficObserver*> observer_list;
    std::vector<FlowData> pending_traffic;

    std::random_device rd;
    std::mt19937 gen;
    std::vector<std::normal_distribution<>> pkt_size_dist;
    std::vector<std::poisson_distribution<>> arrival_dist;
    std::vector<Arrival_t> arrival_type;
    int last_tti;
public:
    TrafficGenerator(TrafficConfig& config) : gen(rd()) {
        last_tti = 0;
        Configure(config);
    }
    void Attach(TrafficObserver* observer){
        observer->Configure(pending_traffic);
        observer_list.push_back(observer);
    }
    void Configure(TrafficConfig& config){
        // Set up sizes for pending traffic and store parameters for generating traffic
        auto flow_cfgs =  config.GetFlowCfgs();
        pending_traffic.resize(flow_cfgs.size());
        pkt_size_dist.resize(flow_cfgs.size());
        arrival_dist.resize(flow_cfgs.size());
        arrival_type.resize(flow_cfgs.size());
        // For now, flow IDs are just buffer indices
        for(int i = 0; i < pending_traffic.size(); i++)
        {
            pending_traffic[i].flow_id = i;
            auto pkt_size = flow_cfgs[i].GetPktSizeParams();
            pkt_size_dist[i] = std::normal_distribution<>(pkt_size.mean,pkt_size.stddev);
            auto arrival = flow_cfgs[i].GetArrivalParams();
            arrival_dist[i] = std::poisson_distribution<>(arrival.mean);
            arrival_type[i] = flow_cfgs[i].GetArrivalType();
        }
    }
    void Generate(int step = 1){
        last_tti += step;
        for(int i = 0; i < arrival_type.size(); i++)
        {
            int num_pkt = 0;
            int num_step_bytes = 0;
            auto& current_flow = pending_traffic[i];
            switch (arrival_type[i])
            {
            case Arrival_t::full_buffer:
                num_step_bytes = FlowType::MAX_BYTES; // Set to arbitrary large number, consider using special condition
                break;
            case Arrival_t::uniform:
                {
                    // Determine how many packets need to be enqueued
                    auto elapsed = last_tti - current_flow.last_arrival;
                    num_pkt = arrival_dist[i].mean()*elapsed;
                }
                break;
            case Arrival_t::poisson:
                num_pkt = arrival_dist[i](gen)*(last_tti - current_flow.last_arrival);
                break;
            default:
                break;
            }
            for(int k = 0; k < num_pkt; k++)
            {
                num_step_bytes += pkt_size_dist[i](gen);
            }
            current_flow.num_bytes = num_step_bytes;
        }
    }
    void Seed(int seed){
        gen.seed(seed);
    }
    void Send(){
        for(auto observer: observer_list)
        {
            observer->Enqueue(pending_traffic);
        }
        for(auto& flow : pending_traffic){
            if(flow.num_bytes > 0)
            {
                flow.last_arrival = last_tti;
                flow.num_bytes = 0;
            }
        }
    }
};
