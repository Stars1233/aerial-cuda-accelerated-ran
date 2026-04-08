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

#include <unistd.h>
#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include <cuda_runtime.h>
#include <curand_kernel.h>
//#include "nvlog.hpp"
#pragma once

#define TTI_PER_SECOND 1000.0f
#define NVLOG_TRAFFIC NVLOG_TAG_BASE_CUMAC + 10

enum class Arrival_t {uniform, poisson, full_buffer};

/// @brief  Parameters describing a particular traffic class
struct TrafficDesc {
   float     arrivalRate; //!< Packet arrival rate in packets per second
   float     arrivalVar;  //!< Packet arrival variance
   Arrival_t arrivalType; //!< Specify arrival type (e.g. uniform, random, full buffer)
   float     packetSize;  //!< Mean packet size in bytes
   float     packetVar;   //!< Packet size variance
   std::vector<uint16_t> ueIds; //!< List of UE IDs that use this traffic type
};

/// @brief  Describes the parameters of the UE Traffic
/// Note:  This is a convenience structure for allocating and initializing data.  Contents should mirror that of \p ueContextPtrs
#pragma pack(push,4)
struct ueContext {
   uint16_t    ueId;
   uint8_t     traffType;               //!< Index of array of TrafficDesc describing traffic profile
   uint32_t    maxBufferSize;           //!< Maximum UE buffer in bytes, queued traffic above this will be dropped
   float       prioWeight;              //!< Scheduling priority

   uint32_t    bufferSize;              //!< Buffer size in bytes
   uint16_t    randSeed;                //!< Random seed for traffic generation
   uint16_t    lastScheduleSlot;        //!< last timeslot the UE had traffic scheduled
   int32_t     lastArrivalSlot;         //!< last timeslot the UE had data enqueued.  Negative numbers can be used to seed transmission starts
   uint16_t    numSchedule;             //!< Number of times UE has been scheduled
   uint32_t    totData;                 //!< Total data arrival at UE
   uint32_t    sumTimeBetweenSchedules; //!< sum of time gaps between schedules
   float       mtbs;                    //!< mean time gap between scheudles
   uint32_t    bufferDrop;              //!< Total data dropped because buffer size was exceeded
};
#pragma pack(pop)

/// @brief Structure of pointers to UE Traffic parameters
struct ueContextPtrs {
   uint16_t*   ueId;
   uint8_t*    traffType;               //!< Index of array of TrafficDesc describing traffic profile
   uint32_t*   maxBufferSize;           //!< Maximum UE buffer in bytes, queued traffic above this will be dropped
   float*      prioWeight;              //!< Scheduling priority

   uint32_t*   bufferSize;              //!< Current buffer size in bytes
   uint16_t*   randSeed;                //!< Random seed for traffic generation
   uint16_t*   lastScheduleSlot;        //!< last timeslot the UE had traffic scheduled
   int32_t*    lastArrivalSlot;         //!< last timeslot the UE had data enqueued.  Negative numbers can be used to seed transmission starts
   uint16_t*   numSchedule;             //!< Number of times UE has been scheduled
   uint32_t*   totData;                 //!< Total data arrival at UE
   uint32_t*   sumTimeBetweenSchedules; //!< sum of time gaps between schedules
   float*      mtbs;                    //!< mean time gap between scheudles
   uint32_t*   bufferDrop;              //!< Total data dropped because buffer size was exceeded
};

class trafficModel {
public:
   trafficModel(uint16_t numUe);
   trafficModel(std::string filename);
   ~trafficModel();
   void applyConfig(std::string filename, int startIdx=0);
   void setup(uint16_t slot, cudaStream_t strm);
   void generate();

   // TBD: Assume buffers are of length nUE
   uint32_t* getUeBuffers();
   uint32_t schedule(uint32_t ueBytes, uint16_t ueId); 

   void printTrafficTypes();
   void printUeParams();
   void printUeState();


 private:
   std::vector<ueContext> parseConfig(std::string configFile);
   cudaError_t allocTrafficData();
   cudaError_t setupTrafficData(std::vector<ueContext> ueParamList, int initialUEs = 0);
   std::vector<TrafficDesc>    trafficTypes;
   ueContextPtrs   h_ueContext;

   TrafficDesc*    d_trafficTypes;
   uint8_t*        d_ueContext;
   uint8_t*        h_ueData;

   uint16_t    nUe; // total number of UEs in the network
   uint16_t    numGbrUe; // total number of GBR UEs
   uint16_t    numNonGbrUe; // total number of non-GBR UEs
   uint16_t    m_curSlot;

   unsigned long long seed;

   dim3 gridDim, blockDim;
   cudaStream_t m_strm;

   std::vector<std::normal_distribution<float>> trafficRng;
   std::default_random_engine generator;
   curandState* d_randStates;

 };
