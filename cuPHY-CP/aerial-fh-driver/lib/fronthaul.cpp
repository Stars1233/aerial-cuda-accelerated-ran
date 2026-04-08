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

#include "fronthaul.hpp"

#include "nic.hpp"
#include "pdump_client.hpp"
#include "utils.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "rivermax.hpp"
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <doca_argp.h>
#pragma GCC diagnostic pop

#define TAG "FH.FH"

namespace aerial_fh
{
Fronthaul::Fronthaul(FronthaulInfo const* info) :
    info_{*info},
    docaParams_{}
{
    NVLOGI_FMT(TAG, "Opening Fronthaul interface");
    validate_input();
    tune_virtual_memory();

    if(!(info_.cuda_device_ids.empty()) || !(info_.cuda_device_ids_for_compute.empty())) {
        doca_gpu_setup();
        doca_init_logger();
    }
    else
        eal_init();

    setup_accurate_send_scheduling();
    setup_gpus();

    if (rmax_enabled())
        rmaxh_ = std::make_unique<RivermaxPrx>(this);

    if(this->pdump_enabled())
    {
        pdump_client_ = std::make_unique<PdumpClient>(this, info_.pdump_client_thread);
    }

    if(fh_stats_dump_enabled())
    {
        fh_stats_dump_ = std::make_unique<FHStatsDump>(this, info_.fh_stats_dump_cpu_core);
    }
}

Fronthaul::~Fronthaul()
{
    if(!(info_.cuda_device_ids.empty()))
    {
        doca_error_t ret;
        //FIXME the doca_gpu_destroy call results in a segfault. This prevents memory footprint from being printed properly, when the relevant TAG is enabled.
#if 0   //FIXME if 0 is temporary until the doca_gpu_destroy segfault is resolved
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        ret = doca_gpu_destroy(docaParams_.gpu);
#pragma GCC diagnostic pop
        if (ret != DOCA_SUCCESS)
            THROW_FH(ret, StringBuilder() << "doca_gpu_destroy returned error");
#endif
    }
    NVLOGI_FMT(TAG, "Closing Fronthaul interface");
}

void Fronthaul::validate_input()
{
    auto sched_res = info_.accu_tx_sched_res_ns;
    if((sched_res != 0) && ((sched_res < kMinAccuTxSchedResNs) || (sched_res > kMaxAccuTxSchedResNs)))
    {
        THROW_FH(EINVAL, StringBuilder() << "Invalid accurate send scheduling timer resolution: " << sched_res << " ns. Must be between " << kMinAccuTxSchedResNs << " and " << kMaxAccuTxSchedResNs);
    }
}

int Fronthaul::rte_eal_init_wrapper(int argc, char** argv)
{
    // Get the current thread ID
    pthread_t thread_id = pthread_self();

    // Get the current CPU affinity mask
    cpu_set_t cpu_affinity_mask;
    auto ret = sched_getaffinity(0, sizeof(cpu_affinity_mask), &cpu_affinity_mask);
    if(ret < 0)
    {
        int errno_copy = errno;
        THROW_FH(errno_copy, StringBuilder() << "Error getting calling thread's affinity: " << strerror(errno_copy));
    }

    // Get the current scheduling policy and parameters
    int scheduling_policy;
    struct sched_param scheduling_params;
    ret = pthread_getschedparam(thread_id, &scheduling_policy, &scheduling_params);
    if(ret < 0)
    {
        int errno_copy = errno;
        THROW_FH(errno_copy, StringBuilder() << "Error getting calling thread's scheduling parameters: " << strerror(errno_copy));
    }

    // Change the CPU affinity mask and scheduling policy
    cpu_set_t new_cpu_affinity_mask;
    CPU_ZERO(&new_cpu_affinity_mask);
    CPU_SET(info_.dpdk_thread, &new_cpu_affinity_mask);
    int new_scheduling_policy = SCHED_FIFO;
    struct sched_param new_scheduling_params;
    new_scheduling_params.sched_priority = 95;

    ret = sched_setaffinity(0, sizeof(new_cpu_affinity_mask), &new_cpu_affinity_mask);
    if(ret < 0)
    {
        int errno_copy = errno;
        THROW_FH(errno_copy, StringBuilder() << "Error setting CPU affinity mask: " << strerror(errno_copy));
    }

#ifdef ENABLE_SCHED_FIFO_ALL_RT
    ret = pthread_setschedparam(thread_id, new_scheduling_policy, &new_scheduling_params);
    if(ret < 0)
    {
        int errno_copy = errno;
        THROW_FH(errno_copy, StringBuilder() << "Error setting scheduling parameters: " << strerror(errno_copy));
    }
#endif

    ret = rte_eal_init(argc, argv);
    if(ret < 0)
    {
        int errno_copy     = errno;
        THROW_FH(EINVAL, StringBuilder() << "DPDK initialization failed: " << strerror(errno_copy));
    }

    // Restore the original CPU affinity mask and scheduling policy
    ret = sched_setaffinity(0, sizeof(cpu_affinity_mask), &cpu_affinity_mask);
    if(ret < 0)
    {
        int errno_copy = errno;
        THROW_FH(errno_copy, StringBuilder() << "Error restoring CPU affinity mask: " << strerror(errno_copy));
    }

    ret = pthread_setschedparam(thread_id, scheduling_policy, &scheduling_params);
    if(ret < 0)
    {
        int errno_copy = errno;
        THROW_FH(errno_copy, StringBuilder() << "Error restoring scheduling parameters: " << strerror(errno_copy));
    }
    return ret;
}

void Fronthaul::eal_init()
{
    std::vector<std::string> args;
    args.clear();
    args.emplace_back("./aerial-fh");
    args.push_back(std::string("--file-prefix=") + info_.dpdk_file_prefix);
    args.push_back(std::string("-l ") + std::to_string(info_.dpdk_thread)); //0,1-19
    args.push_back(std::string("--main-lcore=") + std::to_string(info_.dpdk_thread));

    if(info_.dpdk_verbose_logs)
    {
        args.push_back(std::string("--log-level=,8"));
    }

    for(auto cuda_device_id : info_.cuda_device_ids)
    {
        args.emplace_back("-a");
        args.push_back(Gpu::cuda_device_id_to_pci_bus_id(cuda_device_id));
    }

    args.emplace_back("-a");
    args.push_back("0000:00:0.0");

    args.emplace_back("--");

    const auto argc = args.size();
    std::vector<char*> argv(argc);

    for(size_t i = 0; i < argc; ++i)
    {
        argv[i] = (char*)args[i].c_str();
        NVLOGD_FMT(TAG, "DPDK EAL arg #{}: {}", i, argv[i]);
    }

    cpu_set_t mask;

    auto ret = sched_getaffinity(0, sizeof(cpu_set_t), &mask);
    if(ret < 0)
    {
        int errno_copy = errno;
        THROW_FH(errno_copy, StringBuilder() << "Failed to get calling thread's affinity: " << strerror(errno_copy));
    }

    ret = rte_eal_init_wrapper(argc, argv.data());
    if(ret < 0)
    {
        int rte_errno_copy = rte_errno;
        int ret_aff        = sched_setaffinity(0, sizeof(cpu_set_t), &mask);
        int errno_copy     = errno;
        THROW_FH(EINVAL, StringBuilder() << "DPDK initialization failed: " << rte_strerror(rte_errno_copy) << ((ret_aff == 0) ? std::string("") : (std::string("Additionally, resetting affinity failed: ") + strerror(errno_copy))));
    }

    ret = sched_setaffinity(0, sizeof(cpu_set_t), &mask);
    if(ret < 0)
    {
        int errno_copy = errno;
        THROW_FH(errno_copy, StringBuilder() << "Failed to restore calling thread's affinity: " << strerror(errno_copy));
    }
}

void Fronthaul::doca_gpu_setup()
{
	int ret;

	std::vector<std::string> args;
	args.clear();
	args.emplace_back("./aerial-fh");
	args.push_back(std::string("--file-prefix=") + info_.dpdk_file_prefix);
	args.push_back(std::string("-l ") + std::to_string(info_.dpdk_thread)); //0,1-19
	args.push_back(std::string("--main-lcore=") + std::to_string(info_.dpdk_thread));

	if(info_.dpdk_verbose_logs)
	{
		args.push_back(std::string("--log-level=,8"));
		// Add to enable mlx5 driver debug logs
		args.push_back("--log-level=pmd.net.mlx5:8");
	}

#ifdef ENABLE_DPDK_TX_PKT_TRACING
	args.push_back("--trace=pmd.net.mlx5.tx");
	args.push_back("--trace-dir=/tmp/dpdk_logs");
	args.push_back("--trace-mode=overwrite");
#endif

	for(auto cuda_device_id : info_.cuda_device_ids)
	{
		args.emplace_back("-a");
		args.push_back(Gpu::cuda_device_id_to_pci_bus_id(cuda_device_id));
	}

	args.emplace_back("-a");
	args.push_back("0000:00:0.0");

	const auto argc = args.size();
	std::vector<char*> argv(argc);

	for(size_t i = 0; i < argc; ++i)
	{
		argv[i] = (char*)args[i].c_str();
		NVLOGD_FMT(TAG, "DPDK EAL arg #{}: {}", i, argv[i]);
	}

	ret = rte_eal_init_wrapper(argc, argv.data());
	if (ret < 0)
		THROW_FH(ret, StringBuilder() << "DPDK init failed " << ret);

    printf("INFO: Fronthaul::doca_gpu_setup: cuda_device_ids.size() = %zu, cuda_device_ids_for_compute.size() = %zu\n", info_.cuda_device_ids.size(), info_.cuda_device_ids_for_compute.size());
	/* Initialize DOCA GPU instance. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    doca_error_t ret_doca = DOCA_SUCCESS;
	if(!info_.cuda_device_ids.empty())
	{
		ret_doca = doca_gpu_create(Gpu::cuda_device_id_to_pci_bus_id(info_.cuda_device_ids[0]).c_str(), &docaParams_.gpu);
	}
	else if(!info_.cuda_device_ids_for_compute.empty())
	{
		ret_doca = doca_gpu_create(Gpu::cuda_device_id_to_pci_bus_id(info_.cuda_device_ids_for_compute[0]).c_str(), &docaParams_.gpu);
	}
	if (ret_doca != DOCA_SUCCESS)
		THROW_FH(ret_doca, StringBuilder() << "doca_gpu_create returned error");
#pragma GCC diagnostic pop
}

void Fronthaul::doca_gpu_argp_start()
{
    std::vector<std::string> args;
    args.clear();
    args.emplace_back("./aerial-fh");
    args.push_back(std::string("--file-prefix=") + info_.dpdk_file_prefix);
    args.push_back(std::string("-l ") + std::to_string(info_.dpdk_thread)); //0,1-19
    args.push_back(std::string("--main-lcore=") + std::to_string(info_.dpdk_thread));

    if(info_.dpdk_verbose_logs)
    {
        args.push_back(std::string("--log-level=,8"));
    }

    for(auto cuda_device_id : info_.cuda_device_ids)
    {
        args.emplace_back("-a");
        args.push_back(Gpu::cuda_device_id_to_pci_bus_id(cuda_device_id));
    }

    args.emplace_back("-a");
    args.push_back("0000:00:0.0");

    args.emplace_back("--");

    const auto argc = args.size();
    std::vector<char*> argv(argc);

    for(size_t i = 0; i < argc; ++i)
    {
        argv[i] = (char*)args[i].c_str();
        NVLOGD_FMT(TAG, "DPDK EAL arg #{}: {}", i , argv[i]);
    }

    doca_error_t ret;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
	ret = doca_argp_start(argc, argv.data());
#pragma GCC diagnostic pop
	if (ret != DOCA_SUCCESS)
    {
		//APP_EXIT("Failed to parse application input: %s", doca_error_get_descr(ret));
		THROW_FH(ret, StringBuilder() << "Failed to parse application input in doca_argp_start");
    }
}

void Fronthaul::setup_accurate_send_scheduling()
{
    // if(info_.accu_tx_sched_disable)
    // {
    //     timestamp_mask_   = 0;
    //     timestamp_offset_ = offsetof(rte_mbuf, dynfield1[8]);
    //     return;
    // }

    static const rte_mbuf_dynfield dynfield_desc = {
        .name = RTE_MBUF_DYNFIELD_TIMESTAMP_NAME,
        .size = sizeof(uint64_t),
        .align = __alignof__(uint64_t)
    };

    static const rte_mbuf_dynflag dynflag_desc = {
        RTE_MBUF_DYNFLAG_TX_TIMESTAMP_NAME,
    };

    timestamp_offset_ = rte_mbuf_dynfield_register(&dynfield_desc);
    if(timestamp_offset_ < 0)
    {
        THROW_FH(rte_errno, StringBuilder() << RTE_MBUF_DYNFIELD_TIMESTAMP_NAME << " registration error: " << rte_strerror(rte_errno));
    }

    int32_t dynflag_bitnum = rte_mbuf_dynflag_register(&dynflag_desc);
    if(dynflag_bitnum == -1)
    {
        THROW_FH(rte_errno, StringBuilder() << RTE_MBUF_DYNFLAG_TX_TIMESTAMP_NAME << " registration error: " << rte_strerror(rte_errno));
    }

    auto dynflag_shift = static_cast<uint8_t>(dynflag_bitnum);
    timestamp_mask_    = 1ULL << dynflag_shift;
}

void Fronthaul::tune_virtual_memory()
{
    constexpr char kZoneReclaimCmd[] = "sysctl -w vm.zone_reclaim_mode=0";
    constexpr char kSwappinessCmd[]  = "sysctl -w vm.swappiness=0";

    NVLOGD_FMT(TAG, "Running {}", kZoneReclaimCmd);
    std::string zone_reclaim_cmd = StringBuilder() << kZoneReclaimCmd << " > /dev/null";
    if(std::system(zone_reclaim_cmd.c_str()))
    {
        NVLOGW_FMT(TAG, "Failed to {}", kZoneReclaimCmd);
    }

    NVLOGD_FMT(TAG, "Running {}",kSwappinessCmd);
    std::string swappiness_cmd = StringBuilder() << kSwappinessCmd << " > /dev/null";
    if(std::system(swappiness_cmd.c_str()))
    {
        NVLOGW_FMT(TAG, "Failed to {}", kSwappinessCmd);
    }
}

FronthaulInfo const& Fronthaul::get_info() const
{
    return info_;
}

bool Fronthaul::fh_stats_dump_enabled() const
{
    return info_.fh_stats_dump_cpu_core >= 0;
}


void Fronthaul::update_metrics() const
{
    for(auto& nic : nics_)
    {
        nic->update_metrics();
    }
}

bool Fronthaul::pdump_enabled() const
{
    return info_.pdump_client_thread >= 0;
}

std::vector<Nic*> const& Fronthaul::nics() const
{
    return nics_;
}

uint16_t Fronthaul::add_nic(Nic* nic)
{
    nics_.emplace_back(nic);
    return static_cast<uint16_t>(nics_.size() - 1);
}

void Fronthaul::remove_nic(Nic* nic)
{
    for(auto it = nics_.begin(); it != nics_.end(); it++)
    {
        if(*it == nic)
        {
            nics_.erase(it);
            return;
        }
    }
}

int32_t Fronthaul::get_timestamp_offset() const
{
    return timestamp_offset_;
}

uint64_t Fronthaul::get_timestamp_mask_() const
{
    return timestamp_mask_;
}

UniqueGpuMap& Fronthaul::gpus()
{
    return gpus_;
}

void Fronthaul::setup_gpus()
{
    for(auto cuda_device_id : info_.cuda_device_ids_for_compute)
    {
        gpus_[cuda_device_id] = std::make_unique<Gpu>(this, cuda_device_id);
    }
}

bool Fronthaul::rmax_enabled() const
{
    return info_.rivermax;
}


RivermaxPrx* Fronthaul::rmax_get() const
{
    if(rmax_enabled())
        return rmaxh_.get();
    return nullptr;
}

int Fronthaul::rmax_init_nic(Nic* nic)
{
    socket_handle sockh_;

    // Dry run if rmax is not enabled
    if(!rmax_enabled())
        return 0;

    if(rmaxh_->init_nic(nic->get_mac_address(), &sockh_))
        return -1;
    nic->set_socket(&sockh_);
    // sockh_ = nic->get_socket();
    // NVLOGC_FMT(TAG, "2 Fronthaul sockh_ {} &sockh_ {}", sockh_, &sockh_);

    return 0;
}

docaGpuParams_t* Fronthaul::get_docaGpuParams()
{
    return &docaParams_;
}

} // namespace aerial_fh
