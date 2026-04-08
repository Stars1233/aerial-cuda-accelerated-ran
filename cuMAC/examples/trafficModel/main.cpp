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

#include <iostream>
#include <unistd.h>
#include "trafficService.hpp"
#include "trafficSink.hpp"

void usage(char* arg)
{
    std::cout << "Traffic test" << std::endl
              << arg << ":" << std::endl
              << "\t -h                  Print help" << std::endl
              << "\t -t <number of TTI>  Specify number of TTI" << std::endl
              << "\t -n <number of UEs>  Specify number of UEs" << std::endl
              << "\t -r <average rate>   Specify traffic rate in bytes/TTI" << std::endl
              << "\t -d <rate>           Specify buffer drain rate/TTI" << std::endl
              << "\t -s <seed>           Specify random seed";
}


int main(int argc, char* argv[])
{
    int c;
    uint32_t num_ue = 5;
    int num_tti = 10;
    int data_rate = 1000;
    int drain_rate = 900;
    int data_var = 0;
    int arrival_rate = 1;
    int seed = -1;

    while ((c = getopt(argc, argv, "d:ht:n:r:s:")) != -1) 
    {
        switch(c) 
        {
            case 't':
                num_tti = atoi(optarg);
                break;
            case 'n':
                num_ue = atoi(optarg);
                break;
            case 'r':
                data_rate = atoi(optarg);
                break;
            case 'd':
                drain_rate = atoi(optarg);
                break;
            case 's':
                seed = atoi(optarg);
                break;
            case 'h':
            default:
                usage(argv[0]);
                return -1;
        }
    }


    std::vector<uint32_t> buffer_size(num_ue, 0);
    cumac::cumacCellGrpUeStatus ue_status;
    ue_status.bufferSize = buffer_size.data();
    TrafficType traffic_type(data_rate, data_var, arrival_rate);
    TrafficConfig tmp(traffic_type,num_ue);

    TrafficService traffic_svc(tmp,&ue_status);
    if(seed >= 0){
        traffic_svc.Seed(seed);
    }

    TrafficSink traffic_sink(&ue_status,num_ue,drain_rate);

    for(int i = 0; i < num_tti; i++){
        traffic_svc.Update();
        traffic_sink.Update();
    }
    return 0;
}