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

#include "pmu_reader.hpp"

PMUReader::PMUReader() {num_counters_=0;};

PMUReader::~PMUReader() {
    for (int ii = 0; ii < num_counters_; ii++) {
        ioctl(fids_[ii], PERF_EVENT_IOC_DISABLE, 0);
        close(fids_[ii]);
    }
};

int PMUReader::getNumCounters() {
    return num_counters_;
}

bool PMUReader::addCounter(unsigned int type, unsigned long long config, bool disable = false) {
    //Note: disable makes a particular counter return all 0s
    int current_fid = -1;
    if(!disable) {
        struct perf_event_attr pea = {0};

        pea.type = type;
        pea.size = sizeof(struct perf_event_attr);
        pea.config = config;
        pea.disabled = 1;
        pea.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        pea.exclude_kernel = 1;

        int trigger_fid = -1;
        if(num_counters_ > 0) {
            trigger_fid = fids_[0];
        }
        current_fid = syscall(__NR_perf_event_open, &pea, 0, -1, trigger_fid, 0);
        if (current_fid == -1) {
            //This counter may not be applicable for this hardware
            PMU_READER_ERR_FUNC("PMUReader :: Error opening counter %d\n", num_counters_);

        } else {
            //This counter has successfully been set up

            //Store counter id (not needed for current implementation, but this can be used if
            // counter type ever becomes dynamic)
            ioctl(current_fid, PERF_EVENT_IOC_ID, &ids_[num_counters_]);

            //Start the counter
            ioctl(current_fid, PERF_EVENT_IOC_RESET, 0);
            ioctl(current_fid, PERF_EVENT_IOC_ENABLE, 0);
        }
    }
    fids_[num_counters_] = current_fid;
    num_counters_ += 1;
    return true;
};

int PMUReader::readCounters(uint64_t* counters, int num_slots) {

    read_format data;
    data.num_counters = 0;

    if(num_counters_ > 0) {

        //Perform single read on group leader
        const ssize_t bytes_read = read(fids_[0], &data, sizeof(read_format));

        if(bytes_read < 0) {
            PMU_READER_ERR_FUNC("PMUReader :: Failed to read counters from file descriptor (errno=%d)\n", errno);
            return 0;
        }

        if(static_cast<size_t>(bytes_read) < sizeof(read_format)) {
            PMU_READER_ERR_FUNC("PMUReader :: Incomplete read of counter data (read %zd bytes, expected %zu bytes)\n",
                                bytes_read, sizeof(read_format));
            return 0;
        }

        //PMU_READER_INF_FUNC("Executed read.  time_enabled=%lli, time_running=%lli\n",data.time_enabled,data.time_running);

        //Save off all counters
        int read_index = 0;
        for(int ii=0; ii<data.num_counters && ii<num_slots; ii++) {
            if(fids_[ii] == -1) {
                //Counters that fail setup will populate all 0s
                counters[ii] = 0;
            } else {
                counters[ii] = data.data[read_index].value;
                read_index += 1;
            }
        }
    } else {
        PMU_READER_ERR_FUNC("PMUReader :: Attempting readCounters call when no counters initialized\n");
    }

    //Indicate the number of counters read
    return data.num_counters;

}

PMUReaderGeneral::PMUReaderGeneral() {
    addCounter(PERF_TYPE_HARDWARE,PERF_COUNT_HW_CPU_CYCLES);
    addCounter(PERF_TYPE_HARDWARE,PERF_COUNT_HW_INSTRUCTIONS);
    addCounter(PERF_TYPE_HARDWARE,PERF_COUNT_HW_STALLED_CYCLES_FRONTEND);
    addCounter(PERF_TYPE_HARDWARE,PERF_COUNT_HW_STALLED_CYCLES_BACKEND);
    addCounter(PERF_TYPE_SOFTWARE,PERF_COUNT_SW_PAGE_FAULTS);
    addCounter(PERF_TYPE_SOFTWARE,PERF_COUNT_SW_CONTEXT_SWITCHES);
}

void PMUReaderGeneral::readMetrics(uint64_t* counts, GeneralMetrics* metrics) {
    metrics->cycle_count = counts[0];
    metrics->instr_count = counts[1];
    metrics->frontend_stall_count = counts[2];
    metrics->backend_stall_count = counts[3];
    metrics->page_faults = counts[4];
    metrics->context_switches = counts[5];
}

// For documentation on Grace PMU raw event codes see:
// https://developer.arm.com/documentation/109528/0100/CPU-performance-analysis-methodology
//
//Note: Grace has access to 6 programmable PMU counters, plus a dedicated cycle counters
// If you include too many HW counters into the setup of the PMU - you will will get 0s instead of desired counts

PMUReaderTopdown::PMUReaderTopdown() {
    addCounter(PERF_TYPE_RAW, 0x0011);//CPU_CYCLES
    addCounter(PERF_TYPE_RAW, 0x003B);//OP_SPEC
    addCounter(PERF_TYPE_RAW, 0x003A);//OP_RETIRED
    addCounter(PERF_TYPE_RAW, 0x0010);//BR_MIS_PRED
    addCounter(PERF_TYPE_RAW, 0x003F);//STALL_SLOT
    addCounter(PERF_TYPE_RAW, 0x003E);//STALL_SLOT_FRONTEND
    addCounter(PERF_TYPE_RAW, 0x003D);//STALL_SLOT_BACKEND

}

void PMUReaderTopdown::readMetrics(uint64_t* counts, TopdownMetrics* metrics) {

    double CPU_CYCLES = counts[0];
    double OP_SPEC = counts[1];
    double OP_RETIRED = counts[2];
    double BR_MIS_PRED = counts[3];
    double STALL_SLOT = counts[4];
    double STALL_SLOT_FRONTEND = counts[5];
    double STALL_SLOT_BACKEND = counts[6];

    // PMU_READER_INF_FUNC("CPU_CYCLES/OP_SPEC/OP_RETIRED/BR_MIS_PRED/STALL_SLOT/STALL_SLOT_FRONTEND/STALL_SLOT_BACKEND = %f %f %f %f %f %f %f\n",
    // CPU_CYCLES,OP_SPEC,OP_RETIRED,BR_MIS_PRED,STALL_SLOT,STALL_SLOT_FRONTEND,STALL_SLOT_BACKEND);

    //Note - Definitions of these metrics located at:
    //https://developer.arm.com/documentation/109528/0100/Metrics-by-metric-group-in-Neoverse-V2/Topdown-L1-metrics-for-Neoverse-V2?lang=en
    metrics->retiring = 100 * ( OP_RETIRED / OP_SPEC * (1 - STALL_SLOT / ( CPU_CYCLES * 8)));
    metrics->bad_speculation = 100 * ((1 - OP_RETIRED / OP_SPEC) * (1 - STALL_SLOT / ( CPU_CYCLES * 8)) + BR_MIS_PRED * 4 / CPU_CYCLES);
    metrics->frontend_bound = 100 * ( STALL_SLOT_FRONTEND / ( CPU_CYCLES * 8) - BR_MIS_PRED / CPU_CYCLES);
    metrics->backend_bound = 100 * ( STALL_SLOT_BACKEND / ( CPU_CYCLES * 8) - BR_MIS_PRED * 3 / CPU_CYCLES);
}

PMUReaderCacheMetrics::PMUReaderCacheMetrics() {
    
    //Note - Currently (8/27/2024 kernel 6.5.0-1019-nvidia-64k) L3D_CACHE and LL_CACHE_RD return all 0s.  Until this is fixed
    // L3 result will not be properly populated.  Recommend to use MPKI metrics instead of miss ratios to stay consistent.

    //Counters for "Last Level" - should produce equivalent results to L3D counters
    // addCounter(PERF_TYPE_RAW, 0x0037);//LL_CACHE_MISS_RD
    // addCounter(PERF_TYPE_RAW, 0x0036);//LL_CACHE_RD

    addCounter(PERF_TYPE_RAW, 0x0011);//CPU_CYCLES
    addCounter(PERF_TYPE_RAW, 0x0008);//INST_RETIRED
    addCounter(PERF_TYPE_RAW, 0x0013);//MEM_ACCESS
    addCounter(PERF_TYPE_RAW, 0x002A);//L3D_CACHE_REFILL
    addCounter(PERF_TYPE_RAW, 0x0017);//L2D_CACHE_REFILL
    addCounter(PERF_TYPE_RAW, 0x0003);//L1D_CACHE_REFILL
    addCounter(PERF_TYPE_RAW, 0x0001);//L1I_CACHE_REFILL

}

void PMUReaderCacheMetrics::readMetrics(uint64_t* counts, CacheMetrics* metrics) {

    double CPU_CYCLES = counts[0];
    double INST_RETIRED = counts[1];
    double MEM_ACCESS = counts[2];
    double L3D_CACHE_REFILL = counts[3];
    double L2D_CACHE_REFILL = counts[4];
    double L1D_CACHE_REFILL = counts[5];
    double L1I_CACHE_REFILL = counts[6];

    // PMU_READER_INF_FUNC("CPU_CYCLES/INST_RETIRED/MEM_ACCESS/L3D_CACHE_REFILL/L2D_CACHE_REFILL/L1D_CACHE_REFILL/L1I_CACHE_REFILL = %f %f %f %f %f %f %f\n",
    // CPU_CYCLES,INST_RETIRED,MEM_ACCESS,L3D_CACHE_REFILL,L2D_CACHE_REFILL,L1D_CACHE_REFILL,L1I_CACHE_REFILL);


    //Note - Definitions of these metrics located at:
    //https://developer.arm.com/documentation/109528/0100/Metrics-by-metric-group-in-Neoverse-V2/MPKI-metrics-for-Neoverse-V2?lang=en
    metrics->l1i_miss_pki = 0;
    metrics->l1d_miss_pki = 0;
    metrics->l2d_miss_pki = 0;
    metrics->l3d_miss_pki = 0;
    metrics->mem_access_pki = 0;
    metrics->ipc = 0;
    if(INST_RETIRED!=0) {
        metrics->l1i_miss_pki = L1I_CACHE_REFILL / INST_RETIRED * 1000;
        metrics->l1d_miss_pki = L1D_CACHE_REFILL / INST_RETIRED * 1000;
        metrics->l2d_miss_pki = L2D_CACHE_REFILL / INST_RETIRED * 1000;
        metrics->l3d_miss_pki = L3D_CACHE_REFILL / INST_RETIRED * 1000;
        metrics->mem_access_pki = MEM_ACCESS / INST_RETIRED * 1000;
        metrics->ipc = INST_RETIRED / CPU_CYCLES;
    }
}

PMUReaderCacheMissRatios::PMUReaderCacheMissRatios() {
    
    //Note - Currently (8/27/2024 kernel 6.5.0-1019-nvidia-64k) L3D_CACHE and LL_CACHE_RD return all 0s.  Until this is fixed
    // L3 result will not be properly populated.  Recommend to use MPKI metrics instead of miss ratios to stay consistent.

    //Counters for "Last Level" - should produce equivalent results to L3D counters
    // addCounter(PERF_TYPE_RAW, 0x0037);//LL_CACHE_MISS_RD
    // addCounter(PERF_TYPE_RAW, 0x0036);//LL_CACHE_RD

    addCounter(PERF_TYPE_RAW, 0x0011);//CPU_CYCLES
    addCounter(PERF_TYPE_RAW, 0x002A);//L3D_CACHE_REFILL
    addCounter(PERF_TYPE_RAW, 0x002B);//L3D_CACHE
    addCounter(PERF_TYPE_RAW, 0x0017);//L2D_CACHE_REFILL
    addCounter(PERF_TYPE_RAW, 0x0016);//L2D_CACHE
    addCounter(PERF_TYPE_RAW, 0x0003);//L1D_CACHE_REFILL
    addCounter(PERF_TYPE_RAW, 0x0004);//L1D_CACHE
}

void PMUReaderCacheMissRatios::readMetrics(uint64_t* counts, CacheMissRatioMetrics* metrics) {

    double CPU_CYCLES = counts[0];
    double L3D_CACHE_REFILL = counts[1];
    double L3D_CACHE = counts[2];
    double L2D_CACHE_REFILL = counts[3];
    double L2D_CACHE = counts[4];
    double L1D_CACHE_REFILL = counts[5];
    double L1D_CACHE = counts[6];

    // PMU_READER_INF_FUNC("L3D_CACHE_REFILL/L3D_CACHE/L2D_CACHE_REFILL/L2D_CACHE/L1D_CACHE_REFILL/L1D_CACHE = %f %f %f %f %f %f\n",
    // L3D_CACHE_REFILL,L3D_CACHE,L2D_CACHE_REFILL,L2D_CACHE,L1D_CACHE_REFILL,L1D_CACHE);

    metrics->l1d_cache_miss_ratio = 0;
    metrics->l2d_cache_miss_ratio = 0;
    metrics->l3d_cache_miss_ratio = 0;
    if(L1D_CACHE!=0) {
        metrics->l1d_cache_miss_ratio = L1D_CACHE_REFILL / L1D_CACHE;
    }
    if(L2D_CACHE!=0) {
        metrics->l2d_cache_miss_ratio = L2D_CACHE_REFILL / L2D_CACHE;
    }
    if(L3D_CACHE!=0) {
        metrics->l3d_cache_miss_ratio = L3D_CACHE_REFILL / L3D_CACHE;
    }
}

static const char* pmu_type_to_string(PMU_TYPE pmu_type)
{
    switch (pmu_type)
    {
        case PMU_TYPE_DISABLED:      return "PMU_TYPE_DISABLED";
        case PMU_TYPE_GENERAL:       return "PMU_TYPE_GENERAL";
        case PMU_TYPE_TOPDOWN:       return "PMU_TYPE_TOPDOWN";
        case PMU_TYPE_CACHE_METRICS: return "PMU_TYPE_CACHE_METRICS";
        default:                     return "UNKNOWN";
    }
}

PMUDeltaSummarizer::PMUDeltaSummarizer(PMU_TYPE pmu_type) :
    pmu_type_(pmu_type),
    pmur_(nullptr)
{
#if !defined(__arm__) && !defined(__aarch64__)
    if(pmu_type_ != PMU_TYPE_DISABLED && pmu_type_ != PMU_TYPE_GENERAL) {
        PMU_READER_ERR_FUNC("PMUDeltaSummarizer :: Unable to set pmu_type=%s for non Grace system.  Disabling pmu_metrics.", pmu_type_to_string(pmu_type_));
        pmu_type_ = PMU_TYPE_DISABLED;
    }
#endif

    if(pmu_type_ == PMU_TYPE_GENERAL) {
        pmur_ = new PMUReaderGeneral();
    }

    //Note - these readers are Grace specific
#if defined(__arm__) || defined(__aarch64__)
    if(pmu_type_ == PMU_TYPE_TOPDOWN) {
        pmur_ = new PMUReaderTopdown();  
    } else if(pmu_type_ == PMU_TYPE_CACHE_METRICS) {
        pmur_ = new PMUReaderCacheMetrics();  
    }
#endif

    if(pmu_type_ == PMU_TYPE_DISABLED) {
        PMU_READER_INF_FUNC("PMUReader :: Format 0 :: \n");
    } else if (pmu_type_ == PMU_TYPE_GENERAL) {
        PMU_READER_INF_FUNC("PMUReader :: Format 1 :: cycle_count,instr_count,frontend_stall_count,backend_stall_count,page_faults,context_switches\n");
    } else if (pmu_type_ == PMU_TYPE_TOPDOWN) {
        PMU_READER_INF_FUNC("PMUReader :: Format 2 :: retiring,bad_speculation,frontend_bound,backend_bound\n");
    } else if (pmu_type_ == PMU_TYPE_CACHE_METRICS) {
        PMU_READER_INF_FUNC("PMUReader :: Format 3 :: l1i_miss_pki,l1d_miss_pki,l2d_miss_pki,l3d_miss_pki,mem_access_pki,ipc\n");
    }

    // Warmup: exercise the ioctl/read path once to avoid cold-path overhead
    if (pmu_type_ != PMU_TYPE_DISABLED) {
        recordStart();
        recordStop();
    }
}

PMUDeltaSummarizer::~PMUDeltaSummarizer()
{
    if(pmu_type_ != PMU_TYPE_DISABLED) {
        delete pmur_;
    }
}

void PMUDeltaSummarizer::recordStart()
{

    if(pmu_type_ != PMU_TYPE_DISABLED) {
        pmur_->readCounters(startCounters_,MAX_PMU_COUNTERS);
    }

}

void PMUDeltaSummarizer::recordStop()
{

    if(pmu_type_ != PMU_TYPE_DISABLED) {
        pmur_->readCounters(endCounters_,MAX_PMU_COUNTERS);
    }

}

void PMUDeltaSummarizer::formatCounterMetrics(char* out_str, int max_chars)
{

    //Write empty string (assuming pmu metrics are disabled)
    snprintf(out_str,max_chars,"%i",pmu_type_);

    uint64_t counters_diff[MAX_PMU_COUNTERS];
    if(pmu_type_ != PMU_TYPE_DISABLED) {
        for(int ii=0; ii<MAX_PMU_COUNTERS; ii++) {
            counters_diff[ii] = endCounters_[ii] - startCounters_[ii];
        }
    }

    if(pmu_type_ == PMU_TYPE_GENERAL) {
        GeneralMetrics metrics;
        static_cast<PMUReaderGeneral*>(pmur_)->readMetrics(counters_diff,&metrics);
        snprintf(out_str,max_chars,"%i,%llu,%llu,%llu,%llu,%llu,%llu",
        pmu_type_,
        metrics.cycle_count,
        metrics.instr_count, 
        metrics.frontend_stall_count,
        metrics.backend_stall_count,
        metrics.page_faults,
        metrics.context_switches);
    }
    
    //Note - these readers are Grace specific
#if defined(__arm__) || defined(__aarch64__)
    if(pmu_type_ == PMU_TYPE_TOPDOWN) {
        TopdownMetrics metrics;
        static_cast<PMUReaderTopdown*>(pmur_)->readMetrics(counters_diff,&metrics);
        snprintf(out_str,max_chars,"%i,%.2f,%.2f,%.2f,%.2f",
        pmu_type_,
        metrics.retiring,
        metrics.bad_speculation, 
        metrics.frontend_bound,
        metrics.backend_bound);
    } else if(pmu_type_ == PMU_TYPE_CACHE_METRICS) {
        CacheMetrics metrics;
        static_cast<PMUReaderCacheMetrics*>(pmur_)->readMetrics(counters_diff,&metrics);
        snprintf(out_str,max_chars,"%i,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",
        pmu_type_,
        metrics.l1i_miss_pki,
        metrics.l1d_miss_pki, 
        metrics.l2d_miss_pki,
        metrics.l3d_miss_pki,
        metrics.mem_access_pki,
        metrics.ipc
        );
    }
#endif

}
