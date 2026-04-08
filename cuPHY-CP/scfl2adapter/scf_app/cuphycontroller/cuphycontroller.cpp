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

#include "nv_phy_group.hpp"
#include "yaml.hpp"

// #include "l2_adapter_api.hpp"
#include "scf_5g_fapi.hpp"
#include "nv_tick_generator.hpp"
#include "nv_phy_driver_proxy.hpp"
#ifdef AERIAL_METRICS
#include "aerial_metrics.hpp"
#endif
#include "signal_handler.hpp"

#define TAG (NVLOG_TAG_BASE_SCF_L2_ADAPTER + 5) // "SCF.L2SA"

#define YAML_CONFIG_FILE "cuPHY-CP/scfl2adapter/scf_app/cuphycontroller/cuphycontroller.yaml"

using namespace std;
using namespace nv;

void usage()
{
    NVLOGI_FMT(TAG, "Usage: l2_adapter_cuphycontroller_scf <file>.yaml");
}

// std::unique_ptr<context> l2_api;

void l2sa_exit_handler()
{
    char ts[NVLOG_TIME_STRING_LEN] = {0};
    nvlog_gettimeofday_string(ts, NVLOG_TIME_STRING_LEN);

    NVLOGC_FMT(TAG, "{} {}: exit handler called", ts, __func__);
    printf("%s %s: exit handler called\n", ts, __func__);
    exit(EXIT_FAILURE);
}

/**
 * The signal handler to trigger L1 cleanup and exit
 * @param signum Signal number
 */
static void signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM) {
        // Change exit_handler_flag to L1_EXIT to let msg_processing thread stop sending SLOT.indication
        exit_handler::getInstance().set_exit_handler_flag(exit_handler::L1_EXIT);
    }

    // Note: It's not async-signal-safe to print log in signal handler, but it's really necessary to add log to avoid silent exiting.
    NVLOGC_FMT(TAG, "[signal_handler] received signal {} - {} - {}", signum, sigabbrev_np(signum), sigdescr_np(signum));
}

void oam_init()
{
    CuphyOAM* oam = CuphyOAM::getInstance();
    oam->init_everything();
}

void logcb(int64_t ns, fmtlog::LogLevel level, fmt::string_view location, size_t basePos, fmt::string_view threadName,
           fmt::string_view msg, size_t bodyPos, size_t logFilePos) {
  fmt::print("{}", msg);
}

int main(int argc, const char* argv[])
{
    // Debug starting CPU core issue
    printf("Started l2sa on CPU core %d\n", sched_getcpu());

    int return_value = 0;
    try
    {
        pthread_setname_np(pthread_self(), "phy_init");

        // Parse configuration yaml file path from command line arguments if exist
        char l2sa_cfg_file[1024];
        if(argc >= 2)
        {
            nvlog_safe_strncpy(l2sa_cfg_file, argv[1], 1024);
        }
        else
        {
            std::string relative_path = std::string("../../../../../").append(YAML_CONFIG_FILE);
            nv_get_absolute_path(l2sa_cfg_file, relative_path.c_str());
        }

        // Bind low-priority threads to configured core
        yaml::file_parser cfg_fp(l2sa_cfg_file);
        yaml::document cfg_doc = cfg_fp.next_document();
        yaml::node cfg_node = cfg_doc.root();
        int low_priority_core = -1;
        if (cfg_node.has_key("low_priority_core")) {
            low_priority_core = cfg_node["low_priority_core"].as<int>();
        }
        if (low_priority_core >= 0) {
            nv_assign_thread_cpu_core(low_priority_core);
        }
        NVLOGC_FMT(TAG, "low_priority_core={}", low_priority_core);

#ifdef AERIAL_METRICS
        std::string aerial_metrics_backend_addr = static_cast<std::string>(cfg_node["aerial_metrics_backend_address"]);
        auto& metrics_manager = aerial_metrics::AerialMetricsRegistrationManager::getInstance();
        metrics_manager.changeBackendAddress(aerial_metrics_backend_addr);
#endif

        // Relative path of this process is build/cuPHY-CP/scfl2adapter/scf_app/cuphycontroller/
        char        yaml_file[1024];
        std::string relative_path = std::string("../../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
        nv_get_absolute_path(yaml_file, relative_path.c_str());
        yaml::file_parser fp(yaml_file);
        yaml::document    doc       = fp.next_document();
        yaml::node        root_node = doc.root();

        int primary = 0;
        if(root_node.has_key("nvlog_observer"))
        {
            primary = root_node["nvlog_observer"].as<int>() != 0 ? 0 : 1;
        }

        pthread_t bg_thread_id = nvlog_fmtlog_init(yaml_file, "l2sa.log", l2sa_exit_handler);
        nvlog_fmtlog_thread_init();
        NVLOGC_FMT(TAG, "Thread {} initialized fmtlog", __FUNCTION__);

        // Register signal handlers after nvlog is initialized
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);

        // Initialize PHYDriverProxy
        thread_config thread_cfg;
        if(cfg_node.has_key("phydrv_thread_config"))
        {
            thread_cfg.name           = cfg_node["phydrv_thread_config"]["name"].as<std::string>();
            thread_cfg.cpu_affinity   = cfg_node["phydrv_thread_config"]["cpu_affinity"].as<int>();
            thread_cfg.sched_priority = cfg_node["phydrv_thread_config"]["sched_priority"].as<int>();
        }
        else
        {
            thread_cfg.name           = "phydrv_worker";
            thread_cfg.cpu_affinity   = 4;
            thread_cfg.sched_priority = 95;
        }
        nv::PHYDriverProxy::make(&thread_cfg, cfg_node["instances"].length());

        //--------------------------------------------------------------
        // Initialize an L1 group from the given configuration file
        scf_5g_fapi::init();

        nv::PHY_group grp(l2sa_cfg_file);

        oam_init();

        pthread_setname_np(pthread_self(), "phy_main");

        grp.start();

        grp.join();

        nvlog_fmtlog_close(bg_thread_id);
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "EXCEPTION: {}", e.what());
        return_value = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "UNKNOWN EXCEPTION");
        return_value = 2;
    }
    return return_value;
}
