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

#include "ru_emulator.hpp"

#define TAG (NVLOG_TAG_BASE_RU_EMULATOR + 0) // "RU"

pthread_t RU_Emulator::init(int argc, char ** argv)
{
    char        yaml_file[1024];
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    nv_get_absolute_path(yaml_file, relative_path.c_str());
    pthread_t thread_id = nvlog_fmtlog_init(yaml_file, "ru.log",NULL);
    if(thread_id == -1)
    {
        return -1;
    }
    signal_setup();
    channel_string_setup();
    initialize_arrays();
    set_default_configs();
    get_args(argc, argv);
    parse_yaml(opt_config_file);
    init_fh();
    return thread_id;
};

void RU_Emulator::init_minimal(int argc, char ** argv)
{
    // This function performs a minimal initialization of the RU Emulator for test bench purposes. 
    channel_string_setup();
    initialize_arrays();
    set_default_configs();
    get_args(argc, argv);
    parse_yaml(opt_config_file);
}

void *oam_thread_func_wrapper(void *arg)
{
    if (!arg) {
        do_throw(sb() << "Error: arg == nullptr with oam_thread_func_wrapper");
    }
    RU_Emulator *rue = reinterpret_cast<RU_Emulator *>(arg);
    return rue->oam_thread_func(arg);
}

void* RU_Emulator::oam_thread_func(void* arg)
{
    NVLOGI_FMT(TAG, "RU_Emulator::oam_thread_func()");
    nv_assign_thread_cpu_core(opt_low_priority_core);
    // Thread name lenth should be <= 16 including tail '\0'
    if(pthread_setname_np(pthread_self(), "ru_oam_func") != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "{}: set thread name failed", __func__);
    }

    //fapi_handler *_fapi_handler = reinterpret_cast<fapi_handler*>(arg);

    CuphyOAM* oam = CuphyOAM::getInstance();
    while(1)
    {
        CuphyOAMCellCtrlCmd* cmd;
        while((cmd = oam->get_cell_ctrl_cmd()) != nullptr)
        {
            NVLOGI_FMT(TAG,"get_cell_ctrl_cmd==> cell_ctrl_cmd: {}, cell_id: {} ", cmd->cell_ctrl_cmd, cmd->cell_id);

            switch(cmd->cell_ctrl_cmd)
            {
            case 0: //stop cell
                reset_cell_counters(cmd->cell_id);
                break;
            case 1: //start cell
                break;
            case 2: //config cell
                break;
            }
            oam->free_cell_ctrl_cmd(cmd);
        }

        CuphyOAMULUPlaneDropCmd* ul_u_plane_drop_cmd;
        while((ul_u_plane_drop_cmd = oam->get_ul_u_plane_drop_cmd()) != nullptr)
        {
            NVLOGC_FMT(TAG, "ul_u_plane_drop_cmd ==> cell_id: {}, channel_id: {}, drop_rate: {} %% single_drop: {}, drop_slot: {}, frame_id: {}, subframe_id: {}, slot_id: {}",
                ul_u_plane_drop_cmd->cell_id, ul_u_plane_drop_cmd->channel_id, ul_u_plane_drop_cmd->drop_rate,
                ul_u_plane_drop_cmd->single_drop, ul_u_plane_drop_cmd->drop_slot, ul_u_plane_drop_cmd->frame_id, ul_u_plane_drop_cmd->subframe_id, ul_u_plane_drop_cmd->slot_id);
            if(ul_u_plane_drop_cmd->cell_id >= MAX_CELLS_PER_SLOT)
            {
                NVLOGI_FMT(TAG, "Invalid cell id");
            }
            if(ul_u_plane_drop_cmd->drop_rate > 50 || ul_u_plane_drop_cmd->drop_rate < 0)
            {
                NVLOGI_FMT(TAG, "Invalid drop_rate. Valid range: [0~50]");
            }
            if(ul_u_plane_drop_cmd->channel_id > 4 || ul_u_plane_drop_cmd->channel_id < 1)
            {
                NVLOGI_FMT(TAG, "Invalid channel_id. Valid range: [1~4]");
            }

            if(ul_u_plane_drop_cmd->drop_rate == 0)
            {
                //printf("Stop Simulating UL U Plane pkts drop with cell_id: %d, channel_id: %d\n", ul_u_plane_drop_cmd->cell_id, ul_u_plane_drop_cmd->channel_id);
                ul_pkts_drop_test[ul_u_plane_drop_cmd->cell_id][ul_u_plane_drop_cmd->channel_id].enabled.store(false);
            }
            else
            {
                //printf("Simulating UL U Plane pkts drop with cell_id: %d, channel_id: %d, drop_rate: %d %%\n", ul_u_plane_drop_cmd->cell_id, ul_u_plane_drop_cmd->channel_id, ul_u_plane_drop_cmd->drop_rate);
                auto & drop_set = ul_pkts_drop_test[ul_u_plane_drop_cmd->cell_id][ul_u_plane_drop_cmd->channel_id].drop_set;
                drop_set.clear();
                while(drop_set.size() < ul_u_plane_drop_cmd->drop_rate)
                {
                    drop_set.insert(rand() % 100);
                }
                ul_pkts_drop_test[ul_u_plane_drop_cmd->cell_id][ul_u_plane_drop_cmd->channel_id].drop_rate = ul_u_plane_drop_cmd->drop_rate;
                ul_pkts_drop_test[ul_u_plane_drop_cmd->cell_id][ul_u_plane_drop_cmd->channel_id].cnt = 0;
                ul_pkts_drop_test[ul_u_plane_drop_cmd->cell_id][ul_u_plane_drop_cmd->channel_id].enabled.store(true);
            }

            if(ul_u_plane_drop_cmd->single_drop == 1)
            {
                ul_pkts_drop_test[ul_u_plane_drop_cmd->cell_id][ul_u_plane_drop_cmd->channel_id].single_drop.store(true);
            }

            if(ul_u_plane_drop_cmd->drop_slot == 1)
            {
                ul_pkts_drop_test[ul_u_plane_drop_cmd->cell_id][ul_u_plane_drop_cmd->channel_id].drop_slot.store(true);
                ul_pkts_drop_test[ul_u_plane_drop_cmd->cell_id][ul_u_plane_drop_cmd->channel_id].drop_slot_ts_set.store(false);
                ul_pkts_drop_test[ul_u_plane_drop_cmd->cell_id][ul_u_plane_drop_cmd->channel_id].drop_slot_start_ts.store(0);
                ul_pkts_drop_test[ul_u_plane_drop_cmd->cell_id][ul_u_plane_drop_cmd->channel_id].drop_slot_end_ts.store(0);
                ul_pkts_drop_test[ul_u_plane_drop_cmd->cell_id][ul_u_plane_drop_cmd->channel_id].drop_frame_id.store(ul_u_plane_drop_cmd->frame_id);
                ul_pkts_drop_test[ul_u_plane_drop_cmd->cell_id][ul_u_plane_drop_cmd->channel_id].drop_subframe_id.store(ul_u_plane_drop_cmd->subframe_id);
                ul_pkts_drop_test[ul_u_plane_drop_cmd->cell_id][ul_u_plane_drop_cmd->channel_id].drop_slot_id.store(ul_u_plane_drop_cmd->slot_id);
            }

            oam->free_ul_u_plane_drop_cmd(ul_u_plane_drop_cmd);
        }

        CuphyOAMSfnSlotSyncCmd* sfn_slot_sync_cmd;
        while((sfn_slot_sync_cmd = oam->get_sfn_slot_sync_cmd()) != nullptr)
        {
            NVLOGI_FMT(TAG, "sfn_slot_sync_cmd ==> sync_done from cmd: {}, curr_time: {}", sfn_slot_sync_cmd->sync_done,get_ns());
            oam->free_sfn_slot_sync_cmd(sfn_slot_sync_cmd);
        }

        CuphyOAMZeroUplaneRequest* zero_u_plane_request;
        while((zero_u_plane_request = oam->get_zero_u_plane_request()) != nullptr)
        {
            NVLOGC_FMT(TAG, "zero_u_plane_request ==> cell_id: {}, use_cell_mask: {}, cell_mask: {}, channel_id: {}", zero_u_plane_request->cell_id, zero_u_plane_request->use_cell_mask, zero_u_plane_request->cell_mask, zero_u_plane_request->channel_id);
            if(zero_u_plane_request->cell_id >= MAX_CELLS_PER_SLOT)
            {
                NVLOGC_FMT(TAG, "Invalid cell id received from OAM for zero uplane request, ignoring request, cell_id: {} range: [0-19]", zero_u_plane_request->cell_id);
                oam->free_zero_u_plane_request(zero_u_plane_request);
                break;
            }
            if(zero_u_plane_request->channel_id > 4 || zero_u_plane_request->channel_id < 1)
            {
                NVLOGC_FMT(TAG, "Invalid channel id received from OAM for zero uplane request, ignoring request, channel_id: {} range: [1-4]", zero_u_plane_request->channel_id);
                oam->free_zero_u_plane_request(zero_u_plane_request);
                break;
            }

            if(zero_u_plane_request->use_cell_mask == 1)
            {
                for(int i = 0; i < MAX_CELLS_PER_SLOT; i++)
                {
                    auto cell_mask = 1ULL << i;
                    if(zero_u_plane_request->cell_mask & cell_mask)
                    {
                        ul_pkts_zero_uplane_test[i][zero_u_plane_request->channel_id].enabled.store(true);
                    }
                }
            }
            else
            {
                ul_pkts_zero_uplane_test[zero_u_plane_request->cell_id][zero_u_plane_request->channel_id].enabled.store(true);
            }

            oam->free_zero_u_plane_request(zero_u_plane_request);
        }
        sleep(2);
    }

    NVLOGI_FMT(TAG, "RU_Emulator::oam_thread_func exit thread");
    return nullptr;
}

void RU_Emulator::oam_init()
{
    if (opt_oam_cell_ctrl_cmd == RE_ENABLED)
    {
        CuphyOAM *oam = CuphyOAM::getInstance();
        CuphyOAM::set_server_address("0.0.0.0:50052");

#if 0
    CuphyOAM::set_server_address("0.0.0.0:50051");
    const int   EAL_ARGS       = 3;
    const char* argv[EAL_ARGS] = {"./testmac_eal", "-c", "1"};
    int         ret            = rte_eal_init(EAL_ARGS, (char**)argv);
    if(ret < 0)
    {
        printf("rte_eal_init failed\n");
    }
#endif

        oam->init_everything();

        pthread_t thread_id;
        int ret = pthread_create(&thread_id, NULL, oam_thread_func_wrapper, this);
        if(ret != 0)
        {
            re_dbg("OAM pthread creation error");
        }
    }
}


int RU_Emulator::start()
{
    unsigned int lcore = 0;
    uint64_t time, curr_time, next_time;
    uint64_t dl_tput, ul_tput;
    uint16_t ul_slots, dl_slots, pbch_slots, pdcch_ul_slots, pdcch_dl_slots;
    uint64_t seconds_count = 0;
    uint8_t numFlows;
    char log_buffer[MAX_PRINT_LOG_LENGTH];
    int buffer_index;
    int next_core = 0;

    uint32_t ul_thread_count = 0;
    pthread_t tid_ul_core[MAX_RU_THREADS];

    uint32_t dl_thread_count = 0;
    pthread_t tid_dl_proc_core[MAX_RU_THREADS];
    pthread_t tid_dl_rx_core[MAX_RU_THREADS];
#ifdef STANDALONE
    pthread_t tid_standalone_core;
#endif

    int ul_core_index = 0;

    if(opt_ul_only != RE_ENABLED)
    {
    if(opt_ul_enabled == RE_ENABLED)
    {
        int num_core_ul = ul_core_list.size();
        int num_core_srs = 0;
        
        if(opt_enable_mmimo && enable_srs)
        {
            // SRS is enabled - require ul_srs_core_list to be configured
            if(ul_srs_core_list.empty())
            {
                do_throw(fmt::format("ERROR: SRS is enabled but ul_srs_core_list is empty. "
                                    "Please configure ul_srs_core_list in config.yaml. Aborting !!!"));
            }

            // Calculate required SRS cores based on cells and MAX_SRS_CELLS_PER_CORE
            if (opt_num_cells >= 6) {
                //For 6C and above assume we can hit enqueue in time for MAX_SRS_CELLS_PER_CORE per core
                num_core_srs = (opt_num_cells + MAX_SRS_CELLS_PER_CORE - 1) / MAX_SRS_CELLS_PER_CORE;
            } else {
                //Note: this allows one core per cell up to 5C.  This is a workaround for establishing scaling for 81a/81b/81c/81d
                num_core_srs = opt_num_cells;
            }
            
            // Validate we have enough SRS cores
            const int available_srs_cores = ul_srs_core_list.size();
            if(num_core_srs > available_srs_cores)
            {
                do_throw(fmt::format("ERROR: Requested {} SRS cores for {} cells, but only {} cores in ul_srs_core_list. "
                                    "Need at least {} cores to maintain MAX_SRS_CELLS_PER_CORE={} constraint. This will degrade SRS performance. Aborting !!!",
                                    num_core_srs, opt_num_cells, available_srs_cores, num_core_srs, MAX_SRS_CELLS_PER_CORE));
            }

            re_cons("Allocating {} non-SRS cores and {} SRS cores for {} cells", 
                    num_core_ul, num_core_srs, opt_num_cells);
        }

        if (num_core_ul == 0) {
            do_throw(fmt::format("No UL cores available – aborting start()."));
        }

        ul_cell_cpu_assignment_array ul_cell_cpu_assignment = get_cell_cpu_assignment(opt_num_cells, num_core_ul, opt_enable_mmimo, false, opt_min_ul_cores_per_cell_mmimo);

        for (int i = 0; i < num_core_ul; ++i)
        {
            cplane_core_params[i].rue = this;
            cplane_core_params[i].thread_id = ul_cell_cpu_assignment[i].thread_id;
            cplane_core_params[i].cpu_id = ul_core_list[i];
            cplane_core_params[i].start_cell_index = ul_cell_cpu_assignment[i].start_cell_index;
            cplane_core_params[i].num_cells_per_core = ul_cell_cpu_assignment[i].num_cells_per_core;
            cplane_core_params[i].is_srs = false;

            cpu_set_t tcpuset ;

            CPU_ZERO(&tcpuset);
            CPU_SET(ul_core_list[i],&tcpuset);

            int ret = pthread_create(&tid_ul_core[ul_thread_count], NULL /* attr */, cplane_core_wrapper, (void*)&(cplane_core_params[i]) );
            if(ret != 0)
            {
                re_dbg("pthread creation error");
            }

            ret = pthread_setaffinity_np(tid_ul_core[ul_thread_count], sizeof(tcpuset), &tcpuset);
            if(ret)
            {
                re_dbg("pthread set affinity error {}",ret);
                return -1;
            }
            pthread_detach(tid_ul_core[ul_thread_count]);
            ++ul_thread_count;
        }

        ul_cell_cpu_assignment_array srs_cell_cpu_assignment;
        if(num_core_srs > 0)
        {
            srs_cell_cpu_assignment = get_cell_cpu_assignment(opt_num_cells, num_core_srs, opt_enable_mmimo, true, opt_min_ul_cores_per_cell_mmimo);
        }

        for(int i = 0; i < num_core_srs; ++i)
        {
            const int thread_index = num_core_ul + i;
            
            // SRS cores come from dedicated ul_srs_core_list
            int srs_cpu_id = ul_srs_core_list[i];
            
            cplane_core_params[thread_index].rue = this;
            cplane_core_params[thread_index].thread_id = i;
            cplane_core_params[thread_index].cpu_id = srs_cpu_id;
            cplane_core_params[thread_index].num_cells_per_core = srs_cell_cpu_assignment[i].num_cells_per_core;
            cplane_core_params[thread_index].start_cell_index = srs_cell_cpu_assignment[i].start_cell_index;
            cplane_core_params[thread_index].is_srs = true;

            cpu_set_t tcpuset;
            CPU_ZERO(&tcpuset);
            CPU_SET(srs_cpu_id,&tcpuset);
            int ret = pthread_create(&tid_ul_core[ul_thread_count], NULL /* attr */, cplane_core_wrapper, (void*)&(cplane_core_params[thread_index]) );
            if(ret != 0)
            {
                re_dbg("pthread creation error");
            }
            ret = pthread_setaffinity_np(tid_ul_core[ul_thread_count], sizeof(tcpuset), &tcpuset);
            if(ret)
            {
                re_dbg("pthread set affinity error {}",ret);
                return -1;
            }
            pthread_detach(tid_ul_core[ul_thread_count]);
            ++ul_thread_count;
        }
    }

    if(opt_dl_enabled == RE_ENABLED)
    {
        int core_index = 0;
        int core_count = 0;
        int flow_count = 0;

        for(int cell_index = 0; cell_index < opt_num_cells; ++cell_index)
        {
            flow_count += cell_configs[cell_index].num_dl_flows;
        }

        int num_cores = dl_core_list.size();
        if (num_cores == 0)
        {
            do_throw("No DL cores available - aborting ...");
        }
        opt_num_flows_per_dl_thread = static_cast<float>(flow_count) / num_cores;

        if (flow_count == 0)
        {
            do_throw("No DL flows configured - aborting ...");
        }

        int num_dl_threads_per_flow = static_cast<int>(static_cast<float>(num_cores) / flow_count);
        if (num_dl_threads_per_flow > 1)
        {
            re_cons("Number of dl threads per flow: {}", num_dl_threads_per_flow);
        }

        flow_count = 0;
        for(int cell_index = 0; cell_index < opt_num_cells; ++cell_index)
        {
            numFlows = cell_configs[cell_index].num_dl_flows;
            int threads_per_flow = num_dl_threads_per_flow;
            for(int flow_index = 0; flow_index < numFlows; ++flow_index)
            {
                dl_core_info[core_index].rue = this;
                dl_core_info[core_index].core_index = core_index;
                dl_core_info[core_index].flow_infos[flow_count].flowId = flow_index;
                dl_core_info[core_index].flow_infos[flow_count].cell_index = cell_index;
                dl_core_info[core_index].flow_infos[flow_count].flowValue = cell_configs[cell_index].eAxC_DL[flow_index];
                ++flow_count;
                dl_core_info[core_index].flow_count = num_dl_threads_per_flow > 1 ? opt_num_flows_per_dl_thread : flow_count;
                core_count = core_index + 1;

                if(flow_count >= opt_num_flows_per_dl_thread)
                {
                    flow_count = 0;
                    ++core_index;
                    if (--threads_per_flow > 0)
                    {
                        flow_index--;
                    }
                    else
                    {
                        threads_per_flow = num_dl_threads_per_flow;
                    }

                    if(core_index > MAX_CELLS_PER_SLOT * MAX_FLOWS_PER_DL_CORE)
                    {
                        do_throw(sb() << "Unlikely: Cores needed for DL is greater than array size\n");
                    }
                    if(core_index > dl_core_list.size())
                    {
                        do_throw(sb() << "Cores needed for DL is greater than core size\n");
                    }
                }
            }
        }

        if(opt_enable_dl_proc_mt)
        {
            for(int cid = 0; cid < core_count; ++cid)
            {
                cpu_set_t tcpuset;
                CPU_ZERO(&tcpuset);

                CPU_SET(dl_core_list[cid],&tcpuset);

                int ret = pthread_create(&tid_dl_proc_core[dl_thread_count], NULL /* attr */, uplane_proc_validate_core_wrapper, (void*)&(dl_core_info[cid]) );
                if(ret != 0)
                {
                    re_dbg("pthread creation error");
                }

                ret = pthread_setaffinity_np(tid_dl_proc_core[dl_thread_count], sizeof(tcpuset), &tcpuset);
                if(ret)
                {
                    re_dbg("pthread set affinity error {}",ret);
                    return -1;
                }
                CPU_ZERO(&tcpuset);

                CPU_SET(dl_rx_core_list[cid],&tcpuset);

                ret = pthread_create(&tid_dl_rx_core[dl_thread_count], NULL /* attr */, uplane_proc_rx_core_wrapper, (void*)&(dl_core_info[cid]) );
                if(ret != 0)
                {
                    re_dbg("pthread creation error");
                }

                ret = pthread_setaffinity_np(tid_dl_rx_core[dl_thread_count], sizeof(tcpuset), &tcpuset);
                if(ret)
                {
                    re_dbg("pthread set affinity error {}",ret);
                    return -1;
                }
                pthread_detach(tid_dl_proc_core[dl_thread_count]);
                pthread_detach(tid_dl_rx_core[dl_thread_count]);
                ++dl_thread_count;
            }
        }
        else
        {
            for(int cid = 0; cid < core_count; ++cid)
            {
                cpu_set_t tcpuset;
                CPU_ZERO(&tcpuset);

                CPU_SET(dl_core_list[cid],&tcpuset);

                int ret = pthread_create(&tid_dl_proc_core[dl_thread_count], NULL /* attr */, uplane_proc_core_wrapper, (void*)&(dl_core_info[cid]) );
                if(ret != 0)
                {
                    re_dbg("pthread creation error");
                }

                ret = pthread_setaffinity_np(tid_dl_proc_core[dl_thread_count], sizeof(tcpuset), &tcpuset);
                if(ret)
                {
                    re_dbg("pthread set affinity error {}",ret);
                    return -1;
                }
                pthread_detach(tid_dl_proc_core[dl_thread_count]);
                ++dl_thread_count;
            }
        }
    }
    }
    else
    {
        for (int i = 0; i < opt_num_cells; ++i)
        {
            cplane_core_params[i].rue = this;
            cplane_core_params[i].thread_id = i;
            cplane_core_params[i].cpu_id = ul_core_list[i];
            cplane_core_params[i].start_cell_index = i;
            cplane_core_params[i].num_cells_per_core = 1;
            cplane_core_params[i].is_srs = false;

            cpu_set_t tcpuset ;

            CPU_ZERO(&tcpuset);
            CPU_SET(ul_core_list[ul_core_index],&tcpuset);
            ul_core_index++;

            int ret = pthread_create(&tid_ul_core[ul_thread_count], NULL /* attr */, uplane_tx_only_core_wrapper, (void*)&(cplane_core_params[i]) );
            if(ret != 0)
            {
                re_dbg("pthread creation error");
            }

            ret = pthread_setaffinity_np(tid_ul_core[ul_thread_count], sizeof(tcpuset), &tcpuset);
            if(ret)
            {
                re_dbg("pthread set affinity error {}",ret);
                return -1;
            }
            pthread_detach(tid_ul_core[ul_thread_count]);
            ++ul_thread_count;
        }
    }

#ifdef STANDALONE
    cpu_set_t tcpuset;

    CPU_ZERO(&tcpuset);

    CPU_SET(opt_standalone_core_id,&tcpuset);

    int ret = pthread_create(&tid_standalone_core, NULL /* attr */, standalone_core_wrapper, this );
    if(ret != 0)
    {
        re_dbg("pthread creation error");
    }

    ret = pthread_setaffinity_np(tid_standalone_core, sizeof(tcpuset), &tcpuset);
    if(ret)
    {
        re_dbg("pthread set affinity error {}",ret);
        return -1;
    }
    pthread_detach(tid_standalone_core);
#endif

    curr_time = get_ns();
    next_time = curr_time + 1 * NS_X_S;
    while(!check_force_quit())
    {
        time = get_ns();
        if(time < next_time)
        {
            if(next_time - time > 75 * NS_X_US)
            {
                usleep(50);
            }
            continue;
        }
        ++seconds_count;
        generate_throughput_log(log_buffer, seconds_count);
        curr_time = next_time;
        next_time = curr_time + 1 * NS_X_S;
    }
    for (auto nic : nic_list)
    {
        aerial_fh::print_stats(nic, true);
        aerial_fh::remove_nic(nic);
    }
    aerial_fh::close(fronthaul);
    re_cons("Closed Fronthaul...");

    lcore = 0;

    re_cons("Done, Bye!");
    return RE_OK;
}

void RU_Emulator::print_divider(int num_cells)
{
    int buffer_index;
    char divider[MAX_PRINT_LOG_LENGTH];
    buffer_index = snprintf(divider, MAX_PRINT_LOG_LENGTH, "|--------------------------------");
    for(int i = 0; i < num_cells; ++i)
    {
        buffer_index += snprintf(divider+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "--------------");
    }
    buffer_index += snprintf(divider+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "|");

    NVLOGC_FMT(TAG, "{}",divider);
}

int RU_Emulator::finalize_dlc_tb()
{
    re_cons("DLC testbench finalizing...");
    print_divider(opt_num_cells);
    char buffer[MAX_PRINT_LOG_LENGTH];
    int buffer_index;
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| Total Cells");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|      Cell%2d ", i);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    re_cons("{}",buffer);
    print_divider(opt_num_cells);

    if(opt_pbch_validation == RE_ENABLED)
    {
        generate_results_log(pbch_object);
        print_divider(opt_num_cells);
    }
    if(opt_pdsch_validation == RE_ENABLED)
    {
        normalize_counters(pdsch_object);
        generate_results_log(pdsch_object);
        print_divider(opt_num_cells);
    }
    if(opt_pdcch_ul_validation == RE_ENABLED)
    {
        generate_results_log(pdcch_ul_object);
        print_divider(opt_num_cells);
    }
    if(opt_pdcch_dl_validation == RE_ENABLED)
    {
        generate_results_log(pdcch_dl_object);
        print_divider(opt_num_cells);
    }

    if(opt_csirs_validation == RE_ENABLED)
    {
        generate_results_log(csirs_object);
        print_divider(opt_num_cells);
    }

    if(opt_bfw_dl_validation == RE_ENABLED)
    {
        generate_results_log(bfw_dl_object);
        print_divider(opt_num_cells);
    }

    if(opt_bfw_ul_validation == RE_ENABLED)
    {
        generate_results_log(bfw_ul_object);
        print_divider(opt_num_cells);
    }

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| TOT PUSCH slots");
    generate_results_string(buffer, pusch_object.total_slot_counters);
    print_divider(opt_num_cells);
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| TOT PRACH slots");
    generate_results_string(buffer, prach_object.total_slot_counters);

    print_divider(opt_num_cells);
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| TOT PUCCH slots ");
    generate_results_string(buffer, pucch_object.total_slot_counters);

    print_divider(opt_num_cells);
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| TOT SRS slots ");
    generate_results_string(buffer, srs_object.total_slot_counters);
    print_divider(opt_num_cells);

    // Print DL and UL C-plane section verification counters
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| DL C-Plane TOT Sections");
    generate_results_string(buffer, total_dl_section_counters);
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| DL C-Plane Err Sections");
    generate_results_string(buffer, error_dl_section_counters);
    if (opt_sectionid_validation == RE_ENABLED)
    {
        buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| DL Unannounced U-plane Sections");
        generate_results_string(buffer, section_id_trackers->error_dl_uplane);
    }
    print_divider(opt_num_cells);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| UL C-Plane TOT Sections");
    generate_results_string(buffer, total_ul_section_counters);
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| UL C-Plane Err Sections");
    generate_results_string(buffer, error_ul_section_counters);
    print_divider(opt_num_cells);

    if (opt_beamid_validation == RE_ENABLED)
    {
        buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| DL BeamID TOT Checked  ");
        generate_results_string(buffer, beamid_dl_total_counters);
        buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| DL BeamID Err Mismatch ");
        generate_results_string(buffer, beamid_dl_error_counters);
        buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| UL BeamID TOT Checked  ");
        generate_results_string(buffer, beamid_ul_total_counters);
        buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| UL BeamID Err Mismatch ");
        generate_results_string(buffer, beamid_ul_error_counters);
        print_divider(opt_num_cells);
    }

    return RE_OK;
}

int RU_Emulator::finalize()
{
    // printf("Finalizing...\n");
    re_cons("Finalizing...");
    flush_slot_timing_counters();
    //Assuming only 1 TX queue for the moment
    print_divider(opt_num_cells);
    ul_c_packet_stats.flush_counters(opt_num_cells, DLPacketCounterType::ULC);
    ul_c_packet_stats.flush_counters_file(opt_num_cells, "/tmp/ru_ulc.txt");
    print_divider(opt_num_cells);
    dl_c_packet_stats.flush_counters(opt_num_cells, DLPacketCounterType::DLC);
    dl_c_packet_stats.flush_counters_file(opt_num_cells, "/tmp/ru_dlc.txt");
    print_divider(opt_num_cells);
    dl_u_packet_stats.flush_counters(opt_num_cells, DLPacketCounterType::DLU);
    dl_u_packet_stats.flush_counters_file(opt_num_cells, "/tmp/ru_dlu.txt");

    print_divider(opt_num_cells);
    ul_u_prach_packet_stats.flush_counters(opt_num_cells, ULUPacketCounterType::ULU_PRACH);
    print_divider(opt_num_cells);
    ul_u_pucch_packet_stats.flush_counters(opt_num_cells, ULUPacketCounterType::ULU_PUCCH);
    print_divider(opt_num_cells);
    ul_u_pusch_packet_stats.flush_counters(opt_num_cells, ULUPacketCounterType::ULU_PUSCH);
    print_divider(opt_num_cells);
    ul_u_srs_packet_stats.flush_counters(opt_num_cells, ULUPacketCounterType::ULU_SRS);
    print_divider(opt_num_cells);

    char buffer[MAX_PRINT_LOG_LENGTH];
    int buffer_index;
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| Total Cells");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|      Cell%2d ", i);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    re_cons("{}",buffer);
    print_divider(opt_num_cells);

    generate_slot_count_log();
    print_divider(opt_num_cells);

    if(opt_pbch_validation == RE_ENABLED)
    {
        generate_results_log(pbch_object);
        print_divider(opt_num_cells);
    }
    if(opt_pdsch_validation == RE_ENABLED)
    {
        normalize_counters(pdsch_object);
        generate_results_log(pdsch_object);
        print_divider(opt_num_cells);
    }
    if(opt_pdcch_ul_validation == RE_ENABLED)
    {
        generate_results_log(pdcch_ul_object);
        print_divider(opt_num_cells);
    }
    if(opt_pdcch_dl_validation == RE_ENABLED)
    {
        generate_results_log(pdcch_dl_object);
        print_divider(opt_num_cells);
    }

    if(opt_csirs_validation == RE_ENABLED)
    {
        generate_results_log(csirs_object);
        print_divider(opt_num_cells);
    }

    if(opt_bfw_dl_validation == RE_ENABLED)
    {
        generate_results_log(bfw_dl_object);
        print_divider(opt_num_cells);
    }

    if(opt_bfw_ul_validation == RE_ENABLED)
    {
        generate_results_log(bfw_ul_object);
        print_divider(opt_num_cells);
    }

    if (opt_beamid_validation == RE_ENABLED)
    {
        buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| DL BeamID TOT Checked  ");
        generate_results_string(buffer, beamid_dl_total_counters);
        buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| DL BeamID Err Mismatch ");
        generate_results_string(buffer, beamid_dl_error_counters);
        buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| UL BeamID TOT Checked  ");
        generate_results_string(buffer, beamid_ul_total_counters);
        buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| UL BeamID Err Mismatch ");
        generate_results_string(buffer, beamid_ul_error_counters);
        print_divider(opt_num_cells);
    }

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| PUSCH slots sent ");
    generate_results_string(buffer, pusch_object.total_slot_counters);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| PUSCH C-Plane RX ");
    generate_results_string(buffer, pusch_object.c_plane_rx_tot);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| PUSCH U-Plane TX ");
    generate_results_string(buffer, pusch_object.u_plane_tx_tot);
    print_divider(opt_num_cells);
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| PRACH slots sent ");
    generate_results_string(buffer, prach_object.total_slot_counters);
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| PRACH C-Plane RX ");
    generate_results_string(buffer, prach_object.c_plane_rx_tot);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| PRACH U-Plane TX ");
    generate_results_string(buffer, prach_object.u_plane_tx_tot);
    print_divider(opt_num_cells);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| PUCCH slots sent ");
    generate_results_string(buffer, pucch_object.total_slot_counters);
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| PUCCH C-Plane RX ");
    generate_results_string(buffer, pucch_object.c_plane_rx_tot);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| PUCCH U-Plane TX ");
    generate_results_string(buffer, pucch_object.u_plane_tx_tot);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| SRS slots sent ");
    generate_results_string(buffer, srs_object.total_slot_counters);
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| SRS C-Plane RX ");
    generate_results_string(buffer, srs_object.c_plane_rx_tot);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| SRS U-Plane TX ");
    generate_results_string(buffer, srs_object.u_plane_tx_tot);
    print_divider(opt_num_cells);
    generate_max_sections_count_log();
    print_divider(opt_num_cells);
    generate_timing_results_table();
    print_divider(1);
    re_cons(" ORAN Timing:");
    re_cons("\tdl_c_plane_timing_delay: {}", oran_timing_info.dl_c_plane_timing_delay);
    re_cons("\tdl_c_plane_window_size: {}", oran_timing_info.dl_c_plane_window_size);
    re_cons("\tul_c_plane_timing_delay: {}", oran_timing_info.ul_c_plane_timing_delay);
    re_cons("\tul_c_plane_window_size: {}", oran_timing_info.ul_c_plane_window_size);
    re_cons("\tdl_u_plane_timing_delay: {}", oran_timing_info.dl_u_plane_timing_delay);
    re_cons("\tdl_u_plane_window_size: {}", oran_timing_info.dl_u_plane_window_size);
    re_cons("\tul_u_plane_tx_offset: {}", oran_timing_info.ul_u_plane_tx_offset);
    re_cons("\tul_u_plane_tx_offset_srs: {}", oran_timing_info.ul_u_plane_tx_offset_srs);


    re_cons("Finalizing...");
    if(opt_timing_histogram == RE_ENABLED)
    {
        re_info("Histogram of Timings");
        for(int i = 0; i < STATS_MAX_BINS; ++i)
        {
            re_info("[{:8d} ns from bound] {}", (i - STATS_MAX_BINS/2) * opt_timing_histogram_bin_size, timing_bins[i].load());
        }

        std::ofstream ofile;
        ofile.open ("timing_histogram.txt");
        for(int i = 0; i < STATS_MAX_BINS; ++i)
        {
            ofile << (i - STATS_MAX_BINS/2) * opt_timing_histogram_bin_size << "," <<  timing_bins[i].load() << "\n";
        }
        ofile.close();
    }
    re_cons("Done, Bye!");
    return RE_OK;
}

void RU_Emulator::generate_throughput_log(char* buffer, uint64_t seconds_count)
{
    int buffer_index = 0;
    uint64_t dl_tput, ul_tput;
    uint16_t ul_slots, dl_slots, pbch_slots, pdcch_ul_slots, pdcch_dl_slots, csirs_slots;
    uint16_t c_plane_rx, u_plane_tx;
    uint64_t ontime, early, late, tot;
    for(int cell_idx = 0; cell_idx < opt_num_cells; ++cell_idx)
    {
        buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "Cell %2d ", cell_idx);

        if(opt_pdsch_validation == RE_ENABLED)
        {
            dl_tput = pdsch_object.throughput_counters[cell_idx].load();
            pdsch_object.throughput_counters[cell_idx].store(0);

            dl_slots = pdsch_object.throughput_slot_counters[cell_idx].load();// dl_cores_per_cell;
            pdsch_object.throughput_slot_counters[cell_idx].store(0);

            buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "DL %7.2f Mbps %4d Slots | ", (double)dl_tput/B_X_MB * BIT_X_BYTE, dl_slots);
        }

        ul_tput = pusch_object.throughput_counters[cell_idx].load();
        pusch_object.throughput_counters[cell_idx].store(0);

        ul_slots = pusch_object.throughput_slot_counters[cell_idx].load();
        pusch_object.throughput_slot_counters[cell_idx].store(0);

        buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "UL %7.2f Mbps %4d Slots | ", (double)ul_tput/B_X_MB * BIT_X_BYTE, ul_slots);

        c_plane_rx = pusch_object.c_plane_rx[cell_idx].load();
        pusch_object.c_plane_rx[cell_idx].store(0);

        u_plane_tx = pusch_object.u_plane_tx[cell_idx].load();
        pusch_object.u_plane_tx[cell_idx].store(0);

        // buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "UL Sec1 C %5d pkts U %5d pkts | ", c_plane_rx, u_plane_tx);

        if(opt_pbch_validation == RE_ENABLED)
        {
            pbch_slots = pbch_object.throughput_slot_counters[cell_idx].load();
            pbch_object.throughput_slot_counters[cell_idx].store(0);
            buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "PBCH %5u | ", pbch_slots);
        }

        if(opt_pdcch_ul_validation == RE_ENABLED)
        {
            pdcch_ul_slots = pdcch_ul_object.throughput_slot_counters[cell_idx].load();
            pdcch_ul_object.throughput_slot_counters[cell_idx].store(0);
            buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "PDCCH_UL %5u | ", pdcch_ul_slots);
        }

        if(opt_pdcch_dl_validation == RE_ENABLED)
        {
            pdcch_dl_slots = pdcch_dl_object.throughput_slot_counters[cell_idx].load();
            pdcch_dl_object.throughput_slot_counters[cell_idx].store(0);
            buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "PDCCH_DL %5u | ", pdcch_dl_slots);
        }

        if(opt_csirs_validation == RE_ENABLED)
        {
            csirs_slots = csirs_object.throughput_slot_counters[cell_idx].load();
            csirs_object.throughput_slot_counters[cell_idx].store(0);
            buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "CSI_RS %5u | ", csirs_slots);
        }

        if(opt_bfw_dl_validation == RE_ENABLED)
        {
            auto slots = bfw_dl_object.throughput_slot_counters[cell_idx].load();
            bfw_dl_object.throughput_slot_counters[cell_idx].store(0);
            buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "BFW_DL %5u | ", slots);
        }

        if(opt_bfw_ul_validation == RE_ENABLED)
        {
            auto slots = bfw_ul_object.throughput_slot_counters[cell_idx].load();
            bfw_ul_object.throughput_slot_counters[cell_idx].store(0);
            buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "BFW_UL %5u | ", slots);
        }

        if(opt_prach_enabled)
        {
            c_plane_rx = prach_object.c_plane_rx[cell_idx].load();
            prach_object.c_plane_rx[cell_idx].store(0);
            u_plane_tx = prach_object.u_plane_tx[cell_idx].load();
            prach_object.u_plane_tx[cell_idx].store(0);
            ul_slots = prach_object.throughput_slot_counters[cell_idx].load();
            prach_object.throughput_slot_counters[cell_idx].store(0);
            buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "PRACH %4d Slots | ", ul_slots);
        }

        if(opt_pucch_enabled)
        {
            c_plane_rx = pucch_object.c_plane_rx[cell_idx].load();
            pucch_object.c_plane_rx[cell_idx].store(0);
            u_plane_tx = pucch_object.u_plane_tx[cell_idx].load();
            pucch_object.u_plane_tx[cell_idx].store(0);
            ul_slots = pucch_object.throughput_slot_counters[cell_idx].load();
            pucch_object.throughput_slot_counters[cell_idx].store(0);
            buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "PUCCH %4d Slots | ", ul_slots);
        }

        if(opt_srs_enabled)
        {
            c_plane_rx = srs_object.c_plane_rx[cell_idx].load();
            srs_object.c_plane_rx[cell_idx].store(0);
            u_plane_tx = srs_object.u_plane_tx[cell_idx].load();
            srs_object.u_plane_tx[cell_idx].store(0);
            ul_slots = srs_object.throughput_slot_counters[cell_idx].load();
            srs_object.throughput_slot_counters[cell_idx].store(0);
            buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "SRS %4d Slots | ", ul_slots);
        }

        if(opt_validate_dl_timing)
        {
            ontime = oran_packet_counters.dl_c_plane[cell_idx].ontime_slot.load();
            early = oran_packet_counters.dl_c_plane[cell_idx].early_slot.load();
            late = oran_packet_counters.dl_c_plane[cell_idx].late_slot.load();
            tot = ontime + early + late;
            tot = (tot == 0) ? 1 : tot;
            buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "DL_C_ON %6.2f%%",
                float(ontime) / tot * 100.0
            );
            ontime = oran_packet_counters.dl_u_plane[cell_idx].ontime_slot.load();
            early = oran_packet_counters.dl_u_plane[cell_idx].early_slot.load();
            late = oran_packet_counters.dl_u_plane[cell_idx].late_slot.load();
            tot = ontime + early + late;
            tot = (tot == 0) ? 1 : tot;
            buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, " DL_U_ON %6.2f%%",
                float(ontime) / tot * 100.0
            );
            ontime = oran_packet_counters.ul_c_plane[cell_idx].ontime_slot.load();
            early = oran_packet_counters.ul_c_plane[cell_idx].early_slot.load();
            late = oran_packet_counters.ul_c_plane[cell_idx].late_slot.load();
            tot = ontime + early + late;
            tot = (tot == 0) ? 1 : tot;
            buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, " UL_C_ON %6.2f%% |",
                float(ontime) / tot * 100.0
            );
        }
        buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "Seconds %6lu", seconds_count);
        NVLOGC_FMT(TAG, "{}", buffer);
    }
}

void RU_Emulator::generate_results_log(struct dl_tv_object& tv_object)
{
    char buffer[MAX_PRINT_LOG_LENGTH];
    int buffer_index;
    if (opt_dlc_tb == RE_DISABLED)
    {
        buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| GOOD %s slots processed ", dl_channel_string[tv_object.channel_type].c_str());
        generate_results_string(buffer, tv_object.good_slot_counters);
        buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| BAD %s slots processed ", dl_channel_string[tv_object.channel_type].c_str());
        generate_results_string(buffer, tv_object.error_slot_counters);
    }
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| TOT %s slots processed ", dl_channel_string[tv_object.channel_type].c_str());
    generate_results_string(buffer, tv_object.total_slot_counters);
}

void RU_Emulator::generate_results_string(char* base, std::array<std::atomic<uint32_t>, MAX_CELLS_PER_SLOT>& counter)
{
    char buffer[MAX_PRINT_LOG_LENGTH];
    int buffer_index;
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", base);
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11u ",  counter[i].load());
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);
}

void RU_Emulator::generate_results_string(char* base, std::array<std::atomic<uint64_t>, MAX_CELLS_PER_SLOT>& counter)
{
    char buffer[MAX_PRINT_LOG_LENGTH];
    int buffer_index;
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", base);
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11lu ",  counter[i].load());
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);
}

void RU_Emulator::reset_cell_counters(uint16_t cell_index)
{
    if(cell_index >= opt_num_cells) return;
    NVLOGC_FMT(TAG, "reset cell {} counters ", cell_index);
    for(int i = 0; i < launch_pattern_slot_size; ++i)
    {
        {
            auto& packet_timer = oran_packet_slot_timers.timers[DLPacketCounterType::DLC][cell_index][i];
            packet_timer.reset();
            packet_timer.first_packet = true;
        }
        {
            auto& packet_timer = oran_packet_slot_timers.timers[DLPacketCounterType::DLU][cell_index][i];
            packet_timer.reset();
            packet_timer.first_packet = true;
        }
        {
            auto& packet_timer = oran_packet_slot_timers.timers[DLPacketCounterType::ULC][cell_index][i];
            packet_timer.reset();
            packet_timer.first_packet = true;
        }
    }

    {
        oran_packet_counters.dl_c_plane[cell_index].ontime_slot.store(0);
        oran_packet_counters.dl_c_plane[cell_index].early_slot.store(0);
        oran_packet_counters.dl_c_plane[cell_index].late_slot.store(0);
        oran_packet_counters.dl_u_plane[cell_index].ontime_slot.store(0);
        oran_packet_counters.dl_u_plane[cell_index].early_slot.store(0);
        oran_packet_counters.dl_u_plane[cell_index].late_slot.store(0);
        oran_packet_counters.ul_c_plane[cell_index].ontime_slot.store(0);
        oran_packet_counters.ul_c_plane[cell_index].early_slot.store(0);
        oran_packet_counters.ul_c_plane[cell_index].late_slot.store(0);
    }
}

void RU_Emulator::flush_slot_timing_counters()
{
    // FLUSH remaining slots from counters
    for(int i = 0; i < launch_pattern_slot_size; ++i)
    {
        for(int cell_index = 0; cell_index < opt_num_cells; ++cell_index)
        {
            {
                auto& packet_timer = oran_packet_slot_timers.timers[DLPacketCounterType::DLC][cell_index][i];
                if(packet_timer.late != 0 || packet_timer.early != 0 || packet_timer.ontime != 0)
                {
                    flush_packet_timers(oran_pkt_dir::DIRECTION_DOWNLINK, ECPRI_MSG_TYPE_RTC, cell_index, packet_timer);
                    increment_oran_packet_counters(rx_packet_type::DL_C_PLANE, cell_index, packet_timer, fss_to_launch_pattern_slot(packet_timer.fss, launch_pattern_slot_size));
                }
            }
            {
                auto& packet_timer = oran_packet_slot_timers.timers[DLPacketCounterType::DLU][cell_index][i];
                if(packet_timer.late != 0 || packet_timer.early != 0 || packet_timer.ontime != 0)
                {
                    flush_packet_timers(oran_pkt_dir::DIRECTION_DOWNLINK, ECPRI_MSG_TYPE_IQ, cell_index, packet_timer);
                    increment_oran_packet_counters(rx_packet_type::DL_U_PLANE, cell_index, packet_timer, fss_to_launch_pattern_slot(packet_timer.fss, launch_pattern_slot_size));
                }
            }
            {
                auto& packet_timer = oran_packet_slot_timers.timers[DLPacketCounterType::ULC][cell_index][i];
                if(packet_timer.late != 0 || packet_timer.early != 0 || packet_timer.ontime != 0)
                {
                    flush_packet_timers(oran_pkt_dir::DIRECTION_UPLINK, ECPRI_MSG_TYPE_RTC, cell_index, packet_timer);
                    increment_oran_packet_counters(rx_packet_type::UL_C_PLANE, cell_index, packet_timer, fss_to_launch_pattern_slot(packet_timer.fss, launch_pattern_slot_size));
                }
            }
        }
    }
}

void RU_Emulator::print_aggr_times(std::string packet_type, std::string metric, float value)
{
    char buffer[MAX_PRINT_LOG_LENGTH];
    int buffer_index;
    // For some reason could not do this
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| %s %s ON TIME SLOTS ", packet_type.c_str(), metric.c_str());
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "%%");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %10.2f%% ",  value);
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);
}

void RU_Emulator::generate_slot_level_timing_results_table()
{
    char buffer[MAX_PRINT_LOG_LENGTH];
    int buffer_index;
    uint64_t ontime, early, late, tot;
    float avg;
    float min;
    float min_slot_percentage[MAX_CELLS_PER_SLOT];
    int min_slot[MAX_CELLS_PER_SLOT];

    for (size_t i = 0; i < MAX_CELLS_PER_SLOT; i++)
    {
        min_slot_percentage[i] = 100.0;
        min_slot[i] = -1;
    }

    for(int i = 0; i < launch_pattern_slot_size; ++i)
    {
        buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| ONTIME U-Plane Slot %d %% ", i);
        while(buffer_index < RESULTS_TABLE_START_INDEX)
        {
            buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
        }
        for(int cell = 0; cell < opt_num_cells; ++cell)
        {
            late = oran_packet_counters.dl_u_plane[cell].late_slots_for_slot_num[i].load();
            early = oran_packet_counters.dl_u_plane[cell].early_slots_for_slot_num[i].load();
            ontime = oran_packet_counters.dl_u_plane[cell].ontime_slots_for_slot_num[i].load();
            tot = oran_packet_counters.dl_u_plane[cell].total_slots_for_slot_num[i].load();
            tot = (tot == 0) ? 1 : tot;
            buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %10.2f%% ",  (float)(ontime)/tot * 100.0);

            if(oran_packet_counters.dl_u_plane[cell].total_slots_for_slot_num[i].load() > 0 && min_slot_percentage[cell] >= (float)(ontime)/tot * 100.0)
            {
                min_slot_percentage[cell] = (float)(ontime)/tot * 100.0;
                min_slot[cell] = i;
            }
        }
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
        NVLOGC_FMT(TAG, "{}", buffer);
    }

    print_divider(opt_num_cells);

    //DL U
    {
        float min_avg_slot_percentage = 100.0;
        int min_avg_slot = -1;
        for(int i = 0; i < launch_pattern_slot_size; ++i)
        {
            buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| AVG ONTIME U Slot %d %% ", i);
            while(buffer_index < RESULTS_TABLE_START_INDEX)
            {
                buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
            }

            tot = 0;
            ontime = 0;
            for(int cell = 0; cell < opt_num_cells; ++cell)
            {
                ontime += oran_packet_counters.dl_u_plane[cell].ontime_slots_for_slot_num[i].load();
                tot += oran_packet_counters.dl_u_plane[cell].total_slots_for_slot_num[i].load();
            }
            if(tot > 0 && min_avg_slot_percentage >= (float)(ontime)/tot * 100.0)
            {
                min_avg_slot_percentage = (float)(ontime)/tot * 100.0;
                min_avg_slot = i;
            }

            tot = (tot == 0) ? 1 : tot;
            buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %10.2f%% ",  (float)(ontime)/tot * 100.0);
            buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");

            NVLOGC_FMT(TAG, "{}", buffer);
        }
        print_divider(1);
        buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| WORST AVG ONTIME U Slot %%");
        while(buffer_index < RESULTS_TABLE_START_INDEX)
        {
            buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
        }
        if(min_avg_slot > -1)
        {
            buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| S%02d %6.2f%% ", min_avg_slot, min_avg_slot_percentage);
        }
        else
        {
            buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %10.2f%% ", 0.0);
        }
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
        NVLOGC_FMT(TAG, "{}", buffer);
        print_divider(opt_num_cells);
        buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| WORST ONTIME U-Plane Slot %%");
        while(buffer_index < RESULTS_TABLE_START_INDEX)
        {
            buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
        }
        for(int cell = 0; cell < opt_num_cells; ++cell)
        {
            if(min_slot[cell] > -1)
            {
                buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| S%02d %6.2f%% ", min_slot[cell], min_slot_percentage[cell]);
            }
            else
            {
                buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %10.2f%% ", 0.0);
            }
        }
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
        NVLOGC_FMT(TAG, "{}", buffer);
    }

    //DL C
    {
        float min_avg_slot_percentage = 100.0;
        int min_avg_slot = -1;
        for(int i = 0; i < launch_pattern_slot_size; ++i)
        {
            tot = 0;
            ontime = 0;
            for(int cell = 0; cell < opt_num_cells; ++cell)
            {
                ontime += oran_packet_counters.dl_c_plane[cell].ontime_slots_for_slot_num[i].load();
                tot += oran_packet_counters.dl_c_plane[cell].total_slots_for_slot_num[i].load();
            }
            if(tot > 0 && min_avg_slot_percentage >= (float)(ontime)/tot * 100.0)
            {
                min_avg_slot_percentage = (float)(ontime)/tot * 100.0;
                min_avg_slot = i;
            }

            tot = (tot == 0) ? 1 : tot;

        }
        buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| WORST ONTIME DL C-Plane Slot %%");
        while(buffer_index < RESULTS_TABLE_START_INDEX)
        {
            buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
        }
        for(int cell = 0; cell < opt_num_cells; ++cell)
        {
            if(min_slot[cell] > -1)
            {
                buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| S%02d %6.2f%% ", min_slot[cell], min_slot_percentage[cell]);
            }
            else
            {
                buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %10.2f%% ", 0.0);
            }
        }
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
        NVLOGC_FMT(TAG, "{}", buffer);
    }

    //UL C
    {
        float min_avg_slot_percentage = 100.0;
        int min_avg_slot = -1;
        for(int i = 0; i < launch_pattern_slot_size; ++i)
        {
            tot = 0;
            ontime = 0;
            for(int cell = 0; cell < opt_num_cells; ++cell)
            {
                ontime += oran_packet_counters.ul_c_plane[cell].ontime_slots_for_slot_num[i].load();
                tot += oran_packet_counters.ul_c_plane[cell].total_slots_for_slot_num[i].load();
            }
            if(tot > 0 && min_avg_slot_percentage >= (float)(ontime)/tot * 100.0)
            {
                min_avg_slot_percentage = (float)(ontime)/tot * 100.0;
                min_avg_slot = i;
            }

            tot = (tot == 0) ? 1 : tot;

        }
        buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| WORST ONTIME UL C-Plane Slot %%");
        while(buffer_index < RESULTS_TABLE_START_INDEX)
        {
            buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
        }
        for(int cell = 0; cell < opt_num_cells; ++cell)
        {
            if(min_slot[cell] > -1)
            {
                buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| S%02d %6.2f%% ", min_slot[cell], min_slot_percentage[cell]);
            }
            else
            {
                buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %10.2f%% ", 0.0);
            }
        }
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
        NVLOGC_FMT(TAG, "{}", buffer);
    }

    print_divider(opt_num_cells);

}

void RU_Emulator::generate_timing_results_table()
{
    char buffer[MAX_PRINT_LOG_LENGTH];
    int buffer_index;
    uint64_t ontime, early, late, tot;
    float avg;
    float min;
    generate_slot_level_timing_results_table();

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", "| DL C EARLY PACKETS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_c_plane[i].ontime_packet.load();
        early = oran_packet_counters.dl_c_plane[i].early_packet.load();
        late = oran_packet_counters.dl_c_plane[i].late_packet.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11lu ",  early);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", "| DL C ON TIME PACKETS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_c_plane[i].ontime_packet.load();
        early = oran_packet_counters.dl_c_plane[i].early_packet.load();
        late = oran_packet_counters.dl_c_plane[i].late_packet.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11lu ",  ontime);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", "| DL C LATE PACKETS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_c_plane[i].ontime_packet.load();
        early = oran_packet_counters.dl_c_plane[i].early_packet.load();
        late = oran_packet_counters.dl_c_plane[i].late_packet.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11lu ",  late);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);


    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", "| DL U EARLY PACKETS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_u_plane[i].ontime_packet.load();
        early = oran_packet_counters.dl_u_plane[i].early_packet.load();
        late = oran_packet_counters.dl_u_plane[i].late_packet.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11lu ",  early);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", "| DL U ON TIME PACKETS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_u_plane[i].ontime_packet.load();
        early = oran_packet_counters.dl_u_plane[i].early_packet.load();
        late = oran_packet_counters.dl_u_plane[i].late_packet.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11lu ",  ontime);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", "| DL U LATE PACKETS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_u_plane[i].ontime_packet.load();
        early = oran_packet_counters.dl_u_plane[i].early_packet.load();
        late = oran_packet_counters.dl_u_plane[i].late_packet.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11lu ",  late);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", "| UL C EARLY PACKETS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.ul_c_plane[i].ontime_packet.load();
        early = oran_packet_counters.ul_c_plane[i].early_packet.load();
        late = oran_packet_counters.ul_c_plane[i].late_packet.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11lu ",  early);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", "| UL C ON TIME PACKETS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.ul_c_plane[i].ontime_packet.load();
        early = oran_packet_counters.ul_c_plane[i].early_packet.load();
        late = oran_packet_counters.ul_c_plane[i].late_packet.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11lu ",  ontime);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", "| UL C LATE PACKETS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.ul_c_plane[i].ontime_packet.load();
        early = oran_packet_counters.ul_c_plane[i].early_packet.load();
        late = oran_packet_counters.ul_c_plane[i].late_packet.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11lu ",  late);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s %%", "| DL C ON TIME PACKET");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_c_plane[i].ontime_packet.load();
        early = oran_packet_counters.dl_c_plane[i].early_packet.load();
        late = oran_packet_counters.dl_c_plane[i].late_packet.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %10.2f%% ",  float(ontime) / tot * 100.0);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s %%", "| DL U ON TIME PACKET");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_u_plane[i].ontime_packet.load();
        early = oran_packet_counters.dl_u_plane[i].early_packet.load();
        late = oran_packet_counters.dl_u_plane[i].late_packet.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %10.2f%% ",  float(ontime) / tot * 100.0);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s %%", "| UL C ON TIME PACKET");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.ul_c_plane[i].ontime_packet.load();
        early = oran_packet_counters.ul_c_plane[i].early_packet.load();
        late = oran_packet_counters.ul_c_plane[i].late_packet.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %10.2f%% ",  float(ontime) / tot * 100.0);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", "| DL C EARLY SLOTS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_c_plane[i].ontime_slot.load();
        early = oran_packet_counters.dl_c_plane[i].early_slot.load();
        late = oran_packet_counters.dl_c_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11lu ",  early);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", "| DL C ON TIME SLOTS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_c_plane[i].ontime_slot.load();
        early = oran_packet_counters.dl_c_plane[i].early_slot.load();
        late = oran_packet_counters.dl_c_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11lu ",  ontime);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", "| DL C LATE SLOTS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_c_plane[i].ontime_slot.load();
        early = oran_packet_counters.dl_c_plane[i].early_slot.load();
        late = oran_packet_counters.dl_c_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11lu ",  late);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", "| DL U EARLY SLOTS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_u_plane[i].ontime_slot.load();
        early = oran_packet_counters.dl_u_plane[i].early_slot.load();
        late = oran_packet_counters.dl_u_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11lu ",  early);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", "| DL U ON TIME SLOTS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_u_plane[i].ontime_slot.load();
        early = oran_packet_counters.dl_u_plane[i].early_slot.load();
        late = oran_packet_counters.dl_u_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11lu ",  ontime);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", "| DL U LATE SLOTS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_u_plane[i].ontime_slot.load();
        early = oran_packet_counters.dl_u_plane[i].early_slot.load();
        late = oran_packet_counters.dl_u_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11lu ",  late);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", "| UL C EARLY SLOTS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.ul_c_plane[i].ontime_slot.load();
        early = oran_packet_counters.ul_c_plane[i].early_slot.load();
        late = oran_packet_counters.ul_c_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11lu ",  early);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", "| UL C ON TIME SLOTS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.ul_c_plane[i].ontime_slot.load();
        early = oran_packet_counters.ul_c_plane[i].early_slot.load();
        late = oran_packet_counters.ul_c_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11lu ",  ontime);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);
    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", "| UL C LATE SLOTS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.ul_c_plane[i].ontime_slot.load();
        early = oran_packet_counters.ul_c_plane[i].early_slot.load();
        late = oran_packet_counters.ul_c_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11lu ",  late);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s %%", "| DL C ON TIME SLOTS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_c_plane[i].ontime_slot.load();
        early = oran_packet_counters.dl_c_plane[i].early_slot.load();
        late = oran_packet_counters.dl_c_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %10.2f%% ",  float(ontime) / tot * 100.0);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s %%", "| DL U ON TIME SLOTS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_u_plane[i].ontime_slot.load();
        early = oran_packet_counters.dl_u_plane[i].early_slot.load();
        late = oran_packet_counters.dl_u_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %10.2f%% ",  float(ontime) / tot * 100.0);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s %%", "| UL C ON TIME SLOTS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.ul_c_plane[i].ontime_slot.load();
        early = oran_packet_counters.ul_c_plane[i].early_slot.load();
        late = oran_packet_counters.ul_c_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %10.2f%% ",  float(ontime) / tot * 100.0);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s %%", "| DL C NOT ON TIME SLOTS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_c_plane[i].ontime_slot.load();
        early = oran_packet_counters.dl_c_plane[i].early_slot.load();
        late = oran_packet_counters.dl_c_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %10.2f%% ",  float(early+late) / tot * 100.0);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s %%", "| DL U NOT ON TIME SLOTS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_u_plane[i].ontime_slot.load();
        early = oran_packet_counters.dl_u_plane[i].early_slot.load();
        late = oran_packet_counters.dl_u_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %10.2f%% ",  float(early+late) / tot * 100.0);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);

    buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s %%", "| UL C NOT ON TIME SLOTS");
    while(buffer_index < RESULTS_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.ul_c_plane[i].ontime_slot.load();
        early = oran_packet_counters.ul_c_plane[i].early_slot.load();
        late = oran_packet_counters.ul_c_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %10.2f%% ",  float(early+late) / tot * 100.0);
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
    NVLOGC_FMT(TAG, "{}", buffer);

    print_divider(opt_num_cells);

    avg = 0;
    min = 100;
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_c_plane[i].ontime_slot.load();
        early = oran_packet_counters.dl_c_plane[i].early_slot.load();
        late = oran_packet_counters.dl_c_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        min = (float(ontime) / tot * 100.0 < min) ? float(ontime) / tot * 100.0 : min;
        avg += float(ontime) / tot * 100.0;
    }

    print_aggr_times("DL C", "AVG", avg / opt_num_cells);
    print_aggr_times("DL C", "MIN", min);

    avg = 0;
    min = 100;
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_u_plane[i].ontime_slot.load();
        early = oran_packet_counters.dl_u_plane[i].early_slot.load();
        late = oran_packet_counters.dl_u_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        min = (float(ontime) / tot * 100.0 < min) ? float(ontime) / tot * 100.0 : min;
        avg += float(ontime) / tot * 100.0;
    }
    print_aggr_times("DL U", "AVG", avg / opt_num_cells);
    print_aggr_times("DL U", "MIN", min);

    avg = 0;
    min = 100;
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.ul_c_plane[i].ontime_slot.load();
        early = oran_packet_counters.ul_c_plane[i].early_slot.load();
        late = oran_packet_counters.ul_c_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        min = (float(ontime) / tot * 100.0 < min) ? float(ontime) / tot * 100.0 : min;
        avg += float(ontime) / tot * 100.0;
    }
    print_aggr_times("UL C", "AVG", avg / opt_num_cells);
    print_aggr_times("UL C", "MIN", min);

    avg = 0;
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_c_plane[i].ontime_slot.load();
        early = oran_packet_counters.dl_c_plane[i].early_slot.load();
        late = oran_packet_counters.dl_c_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        avg += float(early+late) / tot * 100.0;
    }
    print_aggr_times("DL C", "AVG NOT", avg / opt_num_cells);

    avg = 0;
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.dl_u_plane[i].ontime_slot.load();
        early = oran_packet_counters.dl_u_plane[i].early_slot.load();
        late = oran_packet_counters.dl_u_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        avg += float(early+late) / tot * 100.0;
    }
    print_aggr_times("DL U", "AVG NOT", avg / opt_num_cells);

    avg = 0;
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ontime = oran_packet_counters.ul_c_plane[i].ontime_slot.load();
        early = oran_packet_counters.ul_c_plane[i].early_slot.load();
        late = oran_packet_counters.ul_c_plane[i].late_slot.load();
        tot = ontime + early + late;
        tot = (tot == 0) ? 1 : tot;
        avg += float(early+late) / tot * 100.0;
    }
    print_aggr_times("UL C", "AVG NOT", avg / opt_num_cells);
}

void RU_Emulator::print_c_plane_beamid(aerial_fh::MsgReceiveInfo &info, uint8_t numberOfSections, uint8_t section_type)
{
    uint8_t* next = (uint8_t*) info.buffer;
    uint16_t frameId        = oran_cmsg_get_frame_id((uint8_t *)info.buffer);
    uint16_t subframeId     = oran_cmsg_get_subframe_id((uint8_t *)info.buffer);
    uint16_t slotId         = oran_cmsg_get_slot_id((uint8_t *)info.buffer);
    uint16_t eaxcId             = oran_msg_get_flowid((uint8_t *)info.buffer);
    uint16_t symId             = oran_cmsg_get_startsymbol_id((uint8_t *)info.buffer);
    next += (section_type == ORAN_CMSG_SECTION_TYPE_1) ? ORAN_CMSG_SECT1_FIELDS_OFFSET : ORAN_CMSG_SECT3_FIELDS_OFFSET;
    for(int i = 0; i < numberOfSections; ++i)
    {
        auto section = reinterpret_cast<oran_cmsg_sect1*>(next);
        uint8_t sectionId = section->sectionId;
        uint16_t beamId = section->beamId;
        if(section_type == ORAN_CMSG_SECTION_TYPE_1)
        {
            re_info("Section Type 1(Most DL/UL channels) F{}S{}S{} eAxC {} Sym {} section {} id {} beamId {}", frameId, subframeId, slotId, eaxcId, symId, i, sectionId, beamId);
        }
        else
        {
            re_info("Section Type 3(PRACH channel) F{}S{}S{} eAxC {} Sym {} section {} id {} beamId {}", frameId, subframeId, slotId, eaxcId, symId, i, sectionId, beamId);
        }
        next += (section_type == ORAN_CMSG_SECTION_TYPE_1) ? sizeof(oran_cmsg_sect1) : sizeof(oran_cmsg_sect3);
    }
}

void RU_Emulator::normalize_counters(struct dl_tv_object& tv_object)
{
#if 0
    uint64_t tmp;
    for(int i = 0; i < opt_num_cells; ++i)
    {
        tmp = tv_object.good_slot_counters[i].load();
        tv_object.good_slot_counters[i].store(tmp/dl_cores_per_cell);
        tmp = tv_object.error_slot_counters[i].load();
        tv_object.error_slot_counters[i].store(tmp/dl_cores_per_cell);
        tmp = tv_object.total_slot_counters[i].load();
        tv_object.total_slot_counters[i].store(tmp/dl_cores_per_cell);
    }
#endif
}

void RU_Emulator::generate_slot_count_log()
{
    char buffer[MAX_PRINT_LOG_LENGTH];
    int buffer_index = 0;

    for(int packet_type = 0; packet_type < ALL_PACKET_TYPES; ++packet_type)
    {
        buffer_index = 0;
        buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| %s slots received ", packet_type_array[packet_type]);
        while(buffer_index < RESULTS_TABLE_START_INDEX)
        {
            buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
        }
        for(int i = 0; i < opt_num_cells; ++i)
        {
            buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11lu ",  slot_count[packet_type][i].load());
        }
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
        NVLOGC_FMT(TAG, "{}", buffer);
    }
}


void RU_Emulator::generate_max_sections_count_log()
{
    char buffer[MAX_PRINT_LOG_LENGTH];
    int buffer_index = 0;

    for(int section_type = 0; section_type < ALL_SECTION_TYPES; ++section_type)
    {
        buffer_index = 0;
        buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "| Max %s sections per slot ", section_type_array[section_type]);
        while(buffer_index < RESULTS_TABLE_START_INDEX)
        {
            buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
        }
        for(int i = 0; i < opt_num_cells; ++i)
        {
            buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11" PRIu32 " ",  max_sections_per_slot[section_type][i]);
        }
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|");
        NVLOGC_FMT(TAG, "{}", buffer);
    }
}
