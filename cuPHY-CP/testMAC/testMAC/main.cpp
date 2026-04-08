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

#include <string.h>
#include <string_view>
#include <sys/time.h>
#include <semaphore.h>
#include <filesystem>

#include "nv_mac_factory.hpp"
#include "nvlog.hpp"

#include "test_mac_configs.hpp"
#include "launch_pattern.hpp"
#include "test_mac.hpp"

#include "cuphyoam.hpp"
#include "common_utils.hpp"
#include "signal_handler.hpp"

#include "nv_utils.h"
#include "yaml_sdk_version.hpp"

#ifdef AERIAL_CUMAC_ENABLE
#include "cumac_pattern.hpp"
#endif

using namespace std;
using namespace nv;

#define TAG (NVLOG_TAG_BASE_TEST_MAC + 0) // "MAC"

////////////////////////////////////////////////////////////////////////
// Function to create an instance of the local MAC class above
test_mac* create_test_mac(yaml::node node_config, uint32_t cell_num)
{
    struct timespec ts_start;
    nvlog_gettime_rt(&ts_start);

    NVLOGC_FMT(TAG, "create_test_mac start");
    test_mac* mac = new test_mac(node_config, cell_num);
    NVLOGC_FMT(TAG, "create_test_mac finished. time={}ms", nvlog_get_interval(&ts_start) / 1000 / 1000);
    return mac;
}

void oam_init(std::string& server_addr)
{
    NVLOGC_FMT(TAG, "{}: server_addr={}", __FUNCTION__, server_addr);
    CuphyOAM* oam = CuphyOAM::getInstance();
    CuphyOAM::set_server_address(server_addr);
    oam->init_everything();
}

////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    NVLOGC_FMT(TAG, "Usage: test_mac <Fxx> <xC> [options]");
    NVLOGC_FMT(TAG, "Options:");
    NVLOGC_FMT(TAG, "  --channels <channel_names>  Specify channels (e.g., PDSCH+PDCCH_DL+PDCCH_UL+PBCH)");
    NVLOGC_FMT(TAG, "  --cells <cell_mask>         Specify cell mask");
    NVLOGC_FMT(TAG, "  --mode <mode>               Specify mode (0=static, 1=dynamic)");
    NVLOGC_FMT(TAG, "  --config <yaml_file>        Specify config YAML file");
    NVLOGC_FMT(TAG, "  --thrput                    Show throughput");
    NVLOGC_FMT(TAG, "  --no-validation             Disable validation");
    NVLOGC_FMT(TAG, "  --help, -h                  Show this help message");
    NVLOGC_FMT(TAG, "");
    NVLOGC_FMT(TAG, "Example: test_mac F08 2C --channels PDSCH+PDCCH_DL+PDCCH_UL+PBCH");
}

int parse_integer_value(char* arg, uint64_t* mask)
{
    if(arg == NULL)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: null parameter", __FUNCTION__);
        return -1;
    }
    NVLOGI_FMT(TAG, "{}: argv={}", __FUNCTION__, arg);

    char* err_ptr = NULL;
    if(strncmp(arg, "0b", 2) == 0 || strncmp(arg, "0B", 2) == 0)
    {
        *mask = strtoull(arg + 2, &err_ptr, 2); // Binary
    }
    else
    {
        *mask = strtoull(arg, &err_ptr, 0); // Octal, Decimal, Hex
    }

    if(err_ptr == NULL || *err_ptr != '\0')
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: invalid integer parameter: {}", __FUNCTION__, arg);
        return -1;
    }

    return 0;
}

int parse_channel_mask(char* arg, uint32_t* mask)
{
    if(arg == NULL)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: null parameter", __FUNCTION__);
        return -1;
    }
    NVLOGI_FMT(TAG, "{}: argv={}", __FUNCTION__, arg);

    if(strncmp(arg, "P", 1) == 0 || strncmp(arg, "CSI_RS", 6) == 0 || strncmp(arg, "SRS", 3) == 0 || strncmp(arg, bfw_dl_str, 6) == 0 || (strncmp(arg, bfw_ul_str, 6)) == 0)
    {
        size_t total_len = strlen(arg);
        size_t offset    = 0;
        do
        {
            for(int ch = 0; ch < channel_type_t::CHANNEL_MAX; ch++)
            {
                if(strncmp(arg + offset, get_channel_name(ch), strlen(get_channel_name(ch))) == 0)
                {
                    *mask |= 1 << ch;
                    offset += strlen(get_channel_name(ch));
                    NVLOGD_FMT(TAG, "{}: channels add ch={}: [{}]", __FUNCTION__, ch, get_channel_name(ch));
                    break;
                }
                if(ch == channel_type_t::CHANNEL_MAX - 1)
                {
                    NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: invalid channel name: {}", __FUNCTION__, arg + offset);
                    return -1;
                }
            }
            offset++; // 1 character delimiter
        } while(offset < total_len);
    }
    else
    {
        char* err_ptr = NULL;
        if(strncmp(arg, "0b", 2) == 0 || strncmp(arg, "0B", 2) == 0)
        {
            *mask = strtol(arg + 2, &err_ptr, 2); // Binary
        }
        else
        {
            *mask = strtol(arg, &err_ptr, 0); // Octal, Decimal, Hex
        }

        if(err_ptr == NULL || *err_ptr != '\0')
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: invalid channel parameter: {}", __FUNCTION__, arg);
            return -1;
        }
    }

    if(*mask >= 1 << channel_type_t::CHANNEL_MAX)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: channel out of range: 0x{:02X} > 0x{:02X}", __FUNCTION__, *mask, (1 << channel_type_t::CHANNEL_MAX) - 1);
        return -1;
    }
    return 0;
}

/**
 * SIGNAL handler function
 *
 * @param[in] signum Signal number
 */
static void signal_handler(int signum)
{
    // Set the nvlog exit_handler atomic flag to trigger exiting the application
    exit_handler::getInstance().set_exit_handler_flag(exit_handler::l1_state::L1_EXIT);

    // Note: It's not async-signal-safe to print log in signal handler, but it's really necessary to add log to avoid silent exiting.
    NVLOGC_FMT(TAG, "[signal_handler] received signal {} - {} - {}", signum, sigabbrev_np(signum), sigdescr_np(signum));
}

////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Debug starting CPU core issue
    printf("Started test_mac on CPU core %d\n", sched_getcpu());

    // Print starting time to debug starting CPU core issue
    char ts_buf[32] = "";
    struct timeval tv;
    struct tm ptm;
    gettimeofday(&tv, NULL);
    if(localtime_r(&tv.tv_sec, &ptm) != NULL)
    {
        // size = 8 + 7 = 15
        size_t size = strftime(ts_buf, sizeof("00:00:00"), "%H:%M:%S", &ptm);
        size += snprintf(ts_buf + size, 8, ".%06ld", tv.tv_usec);
    }
    printf("Started test_mac on CPU core %d at: %s\n", sched_getcpu(), ts_buf);

    // Check for help flag first
    for(int i = 1; i < argc; i++)
    {
        const std::string_view arg(argv[i]);
        if(arg == "--help" || arg == "-h")
        {
            usage();
            return 0;
        }
    }

    // Process command line arguments
    if(argc < 3)
    {
        usage();
        return 1;
    }

    assign_thread_name("mac_init");

    int ret = EXIT_SUCCESS;
    test_mac* testmac = nullptr;
    try{
        // Parse input parameters
        int show_thrput = 0;
        int no_validation = 0;
        uint64_t cell_mask           = 0;
        uint32_t channel_mask        = 0;
        uint64_t mode = 0;
        string   pattern = "";
        string   config_yaml = CONFIG_TESTMAC_YAML_NAME;
        for(int i = 1; i < argc; i++)
        {
            if(strncmp(argv[i], "--channels", strlen("--channels")) == 0 && i < argc - 1)
            {
                // Parse channels
                i++;
                if(parse_channel_mask(argv[i], &channel_mask) < 0)
                {
                    return -1;
                }
            }
            else if(strncmp(argv[i], "--cells", strlen("--cells")) == 0)
            {
                // Parse cells
                i++;
                if(parse_integer_value(argv[i], &cell_mask) < 0)
                {
                    return -1;
                }
            }
            else if(strncmp(argv[i], "--mode", strlen("--mode")) == 0)
            {
                // Parse cells
                i++;
                if(parse_integer_value(argv[i], &mode) < 0)
                {
                    return -1;
                }
            }
            else if(strncmp(argv[i], "--config", strlen("--config")) == 0)
            {
                // Parse config yaml file name
                i++;
                config_yaml = std::string(argv[i]);
            }
            else if(strncmp(argv[i], "--thrput", strlen("--thrput")) == 0)
            {
                // Parse channels
                show_thrput = 1;
            }
            else if(strncmp(argv[i], "--no-validation", strlen("--no-validation")) == 0)
            {
                no_validation = 1;
            }
            else
            {
                // Parse pattern name
                pattern.append("_").append(argv[i]);
            }
        }
        // printf("Parsed args: cell_mask=0x%lX channel_mask=0x%X config=%s lp=%s.yaml\n", cell_mask, channel_mask, config_yaml.c_str(), pattern_file.c_str());

        // Open test_mac_config.yaml and create nvlog instance

        char test_mac_yaml_array[MAX_PATH_LEN];
        std::string temp_path = std::string(CONFIG_TESTMAC_YAML_PATH).append(config_yaml);
        get_full_path_file(test_mac_yaml_array, NULL, temp_path.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
        std::filesystem::path test_mac_yaml(test_mac_yaml_array);
        NVLOGC_FMT(TAG, "test_mac_yaml={}", test_mac_yaml.c_str());

        yaml::file_parser fp(test_mac_yaml.c_str());
        yaml::document    doc       = fp.next_document();
        yaml::node        yaml_root = doc.root();

        aerial::check_yaml_version(yaml_root, test_mac_yaml.c_str());

        // Bind low-priority threads to configured core
        int low_priority_core = -1;
        if (yaml_root.has_key("low_priority_core")) {
            low_priority_core = yaml_root["low_priority_core"].as<int>();
        }
        if (low_priority_core >= 0) {
            nv_assign_thread_cpu_core(low_priority_core);
        }
        NVLOGC_FMT(TAG, "low_priority_core={}", low_priority_core);

        // Load nvlog configuration and open the logger
        std::string log_name = yaml_root["log_name"].as<std::string>();
        std::string ipc_transport = yaml_root["transport"]["type"].as<std::string>();
        // Relative path of this process is $cuBB_SDK/build/cuPHY-CP/testMAC/testMAC/
        char        yaml_file_array[1024];
        //std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
        //nv_get_absolute_path(yaml_file_array, relative_path.c_str());
        std::string relative_path(NVLOG_DEFAULT_CONFIG_FILE);
        get_full_path_file(yaml_file_array, NULL, relative_path.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
        std::filesystem::path yaml_file(yaml_file_array); 
        NVLOGC_FMT(TAG, "nvlog config file={}", yaml_file.c_str());

        pthread_t bg_thread_id = nvlog_fmtlog_init(yaml_file.c_str(), log_name.c_str(), NULL);
        nvlog_fmtlog_thread_init();
        NVLOGC_FMT(TAG, "Thread {} initialized fmtlog", __FUNCTION__);

        // Register signal handlers after nvlog is initialized
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);

        string pattern_file;
        if (mode == 0) {
            pattern_file = "launch_pattern" + pattern + ".yaml";
        } else {
            pattern_file = "dynamic_pattern" + pattern + ".json";
        }
        NVLOGC_FMT(TAG, "Run {} {}", argv[0], pattern_file.c_str());

        // By default enable all channel
        if(channel_mask == 0)
        {
            channel_mask = (1 << channel_type_t::CHANNEL_MAX) - 1;
        }
        std::string enabled_channels;
        for(int ch = 0; ch < channel_type_t::CHANNEL_MAX; ch++)
        {
            if(channel_mask & (1 << ch))
            {
                enabled_channels.append(" ").append(get_channel_name(ch));
            }
        }
        NVLOGC_FMT(TAG, "Enabled channels:{} | channel_mask=0x{:02X}", enabled_channels.c_str(), channel_mask);

        pthread_setname_np(pthread_self(), "mac_main");

        test_mac_configs* configs = new test_mac_configs(yaml_root);
        if(no_validation)
        {
            configs->validate_enable = 0;
        }
        configs->app_mode = mode;

#ifdef AERIAL_CUMAC_ENABLE
        yaml::file_parser* p_cumac_yaml_file = nullptr;
        yaml::document cumac_yaml_doc;
        configs->cumac_configs = nullptr; // Explicitly initialize to nullptr to avoid coverity warning
        // Load TestCUMAC config if enabled
        if (yaml_root.has_key("test_cumac_config_file"))
        {
            std::string file_name = yaml_root["test_cumac_config_file"].as<std::string>();
            if (file_name.length() != 0 && file_name.compare("null") != 0) {            
                // test_mac app relative path: build/cuPHY-CP/testMAC/testMAC/test_mac
                std::string relative_path = std::string("../../../../cuPHY-CP/testMAC/testMAC/").append(file_name.c_str());
                if (yaml_root.has_key("test_cumac_config_path")) {
                    relative_path = yaml_root["test_cumac_config_path"].as<std::string>();
                    relative_path.append(file_name);
                }

                char        yaml_file[1024];
                nv_get_absolute_path(yaml_file, relative_path.c_str());
                NVLOGC_FMT(TAG, "Loaded test_cumac config from {}", yaml_file);
                p_cumac_yaml_file = new yaml::file_parser(yaml_file);
                cumac_yaml_doc = p_cumac_yaml_file->next_document();
                configs->cumac_configs = new test_cumac_configs(cumac_yaml_doc.root());
            }
        }

        cumac_handler* _cumac_handler = nullptr;
        if (configs->cumac_configs != nullptr)
        {
            // string cumac_pattern_file = "cumac_pattern" + pattern + ".yaml";
            string cumac_pattern_file = "cumac_pattern" + pattern + ".yaml";
            NVLOGC_FMT(TAG, "[CUMAC] Start parsing cuMAC pattern file: {}", cumac_pattern_file);
            _cumac_handler = new cumac_handler(configs, cumac_pattern_file.c_str(), channel_mask, cell_mask);
            _cumac_handler->start();
            NVLOGC_FMT(TAG, "[CUMAC] TestCUMAC started");

            if (_cumac_handler->get_cumac_configs()->cumac_cp_standalone) {
                _cumac_handler->join();
                goto quit;
            }
        }
#endif

        launch_pattern *lp = new launch_pattern(configs);
        if(lp->launch_pattern_parsing(pattern_file.c_str(), channel_mask, cell_mask) < 0)
        {
            goto quit;
        }

        int fapi_10_04 = 0;
#ifdef SCF_FAPI_10_04
        fapi_10_04 = 1;
#endif
        NVLOGC_FMT(TAG, "testmac_init: cell_num={} negative_test={} show_thrput={} fapi_10_04={}",
                lp->get_cell_num(), lp->get_negative_test(), show_thrput, fapi_10_04);
        if(show_thrput)
        {
            goto quit;
        }

        oam_init(configs->oam_server_addr);

        // Check app exiting before creating test_mac instance which may be blocked on nvIPC connection
        if (is_app_exiting())
        {
            goto quit;
        }

        testmac = create_test_mac(doc.root(), lp->get_cell_num());

        // Set the test_mac_config and launch_pattern pointer to test_mac
        testmac->set_launch_pattern_and_configs(configs, lp);

        // Start the MAC thread
        testmac->start();

        // Wait for the MAC thread to complete
        testmac->join();

        NVLOGC_FMT(TAG, "test_mac deconstructed");
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "test_mac exit with exception: {}", e.what());
        ret = EXIT_FAILURE;
    }

quit:
    NVLOGC_FMT(TAG, "main: start quitting ...");

    if (testmac != nullptr)
    {
        delete testmac;
    }

    nvlog_fmtlog_close();

    if (ret == EXIT_SUCCESS)
    {
        printf("EXIT successfully from main function\n");
    }
    else
    {
        printf("EXIT from main function with error_code=%d\n", ret);
    }

    return ret;
}
