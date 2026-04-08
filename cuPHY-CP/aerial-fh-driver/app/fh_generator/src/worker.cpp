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

#include "worker.hpp"

#include <algorithm>

#include "fh_generator.hpp"
#include "utils.hpp"
#include "gpudevice.hpp"
#include "yaml_parser.hpp"

#undef TAG
#define TAG "FHGEN.WORKER"

namespace fh_gen
{
void fronthaul_generator_dl_tx_worker(Worker* worker);
void fronthaul_generator_dl_tx_c_worker(Worker* worker);
void fronthaul_generator_dl_rx_worker(Worker* worker);
void fronthaul_generator_ul_rx_worker(Worker* worker);
void fronthaul_generator_ul_tx_worker(Worker* worker);

Worker::Worker(WorkerContext context, WorkerInfo info) :
    info_{info},
    context_{context}
{
    if(info.fh_gen_type_ == FhGenType::DU)
    {
        if(info.worker_type_ == WorkerType::DL_TX_U)
        {
            NVLOGI_FMT(TAG, "DU Launching DL TX U worker on core {}", info.cpu_core);
            thread_ = std::make_unique<std::thread>(fronthaul_generator_dl_tx_worker, this);
        }
        else if(info.worker_type_ == WorkerType::DL_TX_C)
        {
            NVLOGI_FMT(TAG, "DU Launching DL TX C worker on core {}", info.cpu_core);
            thread_ = std::make_unique<std::thread>(fronthaul_generator_dl_tx_c_worker, this);
        }
        else if(info.worker_type_ == WorkerType::UL_RX)
        {
            NVLOGI_FMT(TAG, "DU Launching UL RX worker on core {}", info.cpu_core);
            thread_ = std::make_unique<std::thread>(fronthaul_generator_ul_rx_worker, this);
        }
        else
        {
            NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Invalid worker type for FH Gen DU");
            THROW(StringBuilder() << "Invalid worker type for FH Gen DU");
        }
    }
    else
    {
        if(info.worker_type_ == WorkerType::DL_RX)
        {
            NVLOGI_FMT(TAG, "RU Launching DL RX worker on core {}", info.cpu_core);
            thread_ = std::make_unique<std::thread>(fronthaul_generator_dl_rx_worker, this);
        }
        else if(info.worker_type_ == WorkerType::UL_TX)
        {
            NVLOGI_FMT(TAG, "RU Launching UL TX worker on core {}", info.cpu_core);
            // sort_transmissions();

            thread_ = std::make_unique<std::thread>(fronthaul_generator_ul_tx_worker, this);
        }
        else
        {
            NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Invalid worker type for FH Gen RU");
            THROW(StringBuilder() << "Invalid worker type for FH Gen RU");
        }
    }
    set_affinity();
    set_priority();
    // print_context();
}

Worker::~Worker()
{
    exit_signal_ = true;
    thread_->join();
}

WorkerContext& Worker::get_context()
{
    return context_;
}

volatile bool Worker::exit_signal() const
{
    return exit_signal_.load();
}

void Worker::set_exit_signal()
{
    exit_signal_ = true;
}

void Worker::set_priority()
{
    sched_param schedprm{.__sched_priority = info_.priority};
    if(pthread_setschedparam(thread_->native_handle(), SCHED_FIFO, &schedprm))
    {
        THROW(StringBuilder() << "Failed to set priority SCHED_FIFO priority " << info_.priority << " for CPU " << info_.cpu_core << " worker");
    }
}

void Worker::set_affinity()
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(info_.cpu_core, &cpuset);

    if(pthread_setaffinity_np(thread_->native_handle(), sizeof(cpuset), &cpuset))
    {
        THROW(StringBuilder() << "Failed to set affinity for CPU " << info_.cpu_core << " worker");
    }
}

} // namespace fh_gen
