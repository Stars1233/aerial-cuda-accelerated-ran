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

//Utilities for retrieving performance monitoring unit (PMU) counters

#ifndef PMU_READER
#define PMU_READER

#include <cstdint>
#include <cerrno>
#include <unistd.h>
#include <asm/unistd.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <stdio.h>

#include "nvlog.hpp"
#define PMU_TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 42) // "DRV.PMU"
#define PMU_READER_INF_FUNC(...) NVLOGI(PMU_TAG,__VA_ARGS__)
#define PMU_READER_ERR_FUNC(...) NVLOGC(PMU_TAG,__VA_ARGS__)

#ifndef PMU_READER_INF_FUNC
#define PMU_READER_INF_FUNC(...) fprintf(stdout,__VA_ARGS__)
#endif

#ifndef PMU_READER_ERR_FUNC
#define PMU_READER_ERR_FUNC(...) fprintf(stderr,__VA_ARGS__)
#endif

#define MAX_PMU_COUNTERS 7

typedef enum PMU_TYPE {
    PMU_TYPE_DISABLED=0,
    PMU_TYPE_GENERAL=1,
    PMU_TYPE_TOPDOWN=2,
    PMU_TYPE_CACHE_METRICS=3,
} PMU_TYPE;

//Details on formats here: https://man7.org/linux/man-pages/man2/perf_event_open.2.html
struct read_format {
    uint64_t num_counters;
    uint64_t time_enabled;
    uint64_t time_running;

    struct {
        uint64_t value;
        uint64_t id;
    } data[MAX_PMU_COUNTERS];
};

class PMUReader {
public:
    PMUReader();
    ~PMUReader();

    int getNumCounters();
    bool addCounter(unsigned int type, unsigned long long config, bool disable);
    int readCounters(uint64_t* counters, int num_slots);


private:
    int num_counters_;
    int fids_[MAX_PMU_COUNTERS];
    int ids_[MAX_PMU_COUNTERS];

};

//General metrics - counts intended to be applicable across all platforms
struct GeneralMetrics {
    uint64_t cycle_count;          //Total number of cycles
    uint64_t instr_count;          //Total number of instructions retired
    uint64_t frontend_stall_count; //Total number of stalls due to frontend
    uint64_t backend_stall_count;  //Total number of stalls due to backend
    uint64_t page_faults;          //Total number of page faults
    uint64_t context_switches;     //Total number of context switches
};

class PMUReaderGeneral : public PMUReader {
    public:
    PMUReaderGeneral();

    void readMetrics(uint64_t* counts, GeneralMetrics* metrics);
};

//Topdown metrics (Grace specific for now) - percentages that indicate which section of CPU is limiting processing
//Please see: https://developer.arm.com/documentation/109528/0100/Metrics-by-metric-group-in-Neoverse-V2/Topdown-L1-metrics-for-Neoverse-V2?lang=en
struct TopdownMetrics {
    double retiring;        //Percentage of total time slots that retired operations (indicates cycles used efficiently)
    double bad_speculation; //Percentage of total time slots that executed but did not retire (indicates wasted effort due to bad speculation)
    double frontend_bound;  //Percentage of total time slots that stalled due to frontend
    double backend_bound;   //Percentage of total time slots that stalled due to backend
};

class PMUReaderTopdown : public PMUReader {
    public:
    PMUReaderTopdown();

    void readMetrics(uint64_t* counts, TopdownMetrics* metrics);
};

//Normalized (PKI = per kilo instruction) cache/memory metrics (Grace specific for now) - Shows numbers of misses/access pki
//Please see: https://developer.arm.com/documentation/109528/0100/Metrics-by-metric-group-in-Neoverse-V2/MPKI-metrics-for-Neoverse-V2?lang=en
struct CacheMetrics {
    double l1i_miss_pki;   //L1 instruction cache misses per kilo instruction
    double l1d_miss_pki;   //L1 data cache misses per kilo instruction
    double l2d_miss_pki;   //L2 data cache misses per kilo instruction
    double l3d_miss_pki;   //L3 data cache misses per kilo instruction
    double mem_access_pki; //Memory accesses per kilo instruction
    double ipc;            //Average instructions per clock
};

class PMUReaderCacheMetrics : public PMUReader {
    public:
    PMUReaderCacheMetrics();

    void readMetrics(uint64_t* counts, CacheMetrics* metrics);
};

//Miss ratios (misses / total accesses) - currently not exposed externally
//Overall misses pki and access pki are better metrics to directly rate values to performance impact, but these ratios
// can provide useful insight.
struct CacheMissRatioMetrics {
    double l1d_cache_miss_ratio; //L1 data cache misses / L1 data cache total access
    double l2d_cache_miss_ratio; //L2 data cache misses / L2 data cache total access 
    double l3d_cache_miss_ratio; //L3 data cache misses / L3 data cache total access
};

class PMUReaderCacheMissRatios : public PMUReader {
    public:
    PMUReaderCacheMissRatios();

    void readMetrics(uint64_t* counts, CacheMissRatioMetrics* metrics);
};

//Single interface that computes delta and format results in a string
class PMUDeltaSummarizer{
    public:
    /**
     * @brief Construct PMU delta summarizer and warm up counters.
     *
     * IMPORTANT: Must be constructed on the thread whose counters you want to
     * measure. The underlying perf_event_open syscall uses pid=0 (calling thread).
     * Counters attached here cannot be transferred to another thread.
     *
     * Performs one warmup recordStart()/recordStop() cycle to avoid cold-path
     * overhead on the first real measurement.
     *
     * @param pmu_type PMU counter configuration (DISABLED, GENERAL, TOPDOWN, CACHE_METRICS)
     */
    PMUDeltaSummarizer(PMU_TYPE pmu_type);
    ~PMUDeltaSummarizer();
    void recordStart();
    void recordStop();
    void formatCounterMetrics(char* out_str, int max_chars);
    private:
    PMU_TYPE pmu_type_;
    PMUReader* pmur_;
    uint64_t startCounters_[MAX_PMU_COUNTERS];
    uint64_t endCounters_[MAX_PMU_COUNTERS];
};

#endif