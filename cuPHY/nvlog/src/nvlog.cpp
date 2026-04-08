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

#include <csignal>
#include <exception>
#include <libgen.h>
#include <atomic>
#include "nvlog.h"
#include "nvlog.hpp"
#include "memtrace.h"
#include "yaml_sdk_version.hpp"

#define TAG "NVLOG.CPP"

#define MAX_PATH_LEN 1024
#define CONFIG_CUBB_ROOT_ENV "CUBB_HOME"

static char logfile_base[MAX_PATH_LEN] = "/tmp";
static size_t max_log_file_size_bytes = 20000000000;
static constexpr int DEFAULT_MAX_ROTATING_FILE_NUM = 8;
static int32_t max_rotating_file_num = DEFAULT_MAX_ROTATING_FILE_NUM;
static std::atomic<int32_t> current_logfile_index{0};

// Avoid duplicate initiating
static std::atomic<int> fmt_log_initiated = 0;

static pthread_t g_fmtlog_thread_id = 0; //!< Background polling thread id.

exit_handler& pExitHandler=exit_handler::getInstance();

static inline fmtlog::LogLevel getfmtLogLevel(int level)
{
    switch (level)
    {
        case NVLOG_NONE:
            return fmtlog::OFF;
        case NVLOG_FATAL:
            return fmtlog::FAT;
        case NVLOG_ERROR:
            return fmtlog::ERR;
        case NVLOG_CONSOLE:
            return fmtlog::CON;
        case NVLOG_WARN:
            return fmtlog::WRN;
        case NVLOG_INFO:
            return fmtlog::INF;
        case NVLOG_DEBUG:
            return fmtlog::DBG;
        case NVLOG_VERBOSE:
            return fmtlog::VEB;
        default:
            printf("invalid log level %d, setting to WRN level\n", level);
            return fmtlog::WRN;
    }
    return fmtlog::OFF;
}


/**
 * Build the log file path for a given index.
 * @param buf Output buffer.
 * @param size Size of buf.
 * @param index Index 0 = logfile_base only; index > 0 = logfile_base + ".%d".
 */
static void build_logfile_path(char* buf, size_t size, int index)
{
    if (index == 0)
    {
        snprintf(buf, size, "%s", logfile_base);
    }
    else
    {
        snprintf(buf, size, "%s.%d", logfile_base, index);
    }
}

/**
 * Update the log filename.
 * @note Uses logfile_base and current_logfile_index. Index 0 = no suffix; index > 0 = .N. Wraps around.
 */
void update_log_filename()
{
    char new_path[MAX_PATH_LEN] = {0};
    // Get the next file index and wrap around.
    int new_index = current_logfile_index.fetch_add(1) + 1;
    if (new_index == max_rotating_file_num)
    {
        current_logfile_index.fetch_sub(max_rotating_file_num - 1);
    }

    // Wrap around file index to range [1, max_rotating_file_num]
    new_index = (new_index - 1) % (max_rotating_file_num - 1) + 1;
    build_logfile_path(new_path, sizeof(new_path), new_index);

    fmtlog::closeLogFile();
    fmtlog::setLogFile(new_path, true);
}

static std::atomic_bool anyLogWasFull{false};
void logfullcb_aerial(void* unused)
{
   anyLogWasFull.store(true);
}

void logcb_aerial(int64_t ns, fmtlog::LogLevel level, fmt::string_view location, size_t basePos, fmt::string_view threadName,
           fmt::string_view msg, size_t bodyPos, size_t logFilePos) {
    if (level >= fmtlog::WRN)
    {
        fmt::print("{}\n", msg);
        fflush(stdout);
    }

    if (logFilePos > max_log_file_size_bytes)
    {
        update_log_filename();
    }
}

static std::atomic_bool threadRunning{false};
static unsigned int usec_poll_period = 100000;

void *bg_fmtlog_collector(void *)
{
    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGTERM);
    sigaddset(&mask, SIGUSR1);
    pthread_sigmask(SIG_BLOCK, &mask, NULL);
    if(pthread_setname_np(pthread_self(), "bg_fmtlog") != 0)
    {
        NVLOGW_FMT(TAG, "{}: set thread name failed", __func__);
    }
    while (threadRunning.load() == true)
    {
        fmtlog::poll(false);
        usleep(usec_poll_period);
    }
    fmtlog::poll(true);
    printf("Exiting bg_fmtlog_collector - log queue ever was full: %d\n", anyLogWasFull.load());
    return NULL;
}

pthread_t startNVPollingThread()
{
    pthread_t thread_id;
    threadRunning.store(true);
    int ret = pthread_create(&thread_id, NULL, bg_fmtlog_collector, NULL);
    if(ret != 0)
    {
        threadRunning.store(false);
        printf("fmtlog background thread creation failed\n");
        return -1;
    }
    return thread_id;
}

/**
 * Close the fmtlog.
 * @note bg_thread_id is unused. Kept for API compatibility.
 */
void nvlog_fmtlog_close(pthread_t bg_thread_id)
{
    (void)bg_thread_id;  // Unused; uses g_fmtlog_thread_id. Kept for API compatibility.

    int last_index = current_logfile_index.load();
    if (last_index > 0) // If not the initial log file
    {
        last_index = (last_index -1) % (max_rotating_file_num - 1) + 1;
    }

    char thread_name[16];
    pthread_getname_np(pthread_self(), thread_name, 16);

    // Print the last log file path for debug
    char last_file_path[MAX_PATH_LEN] = {0};
    build_logfile_path(last_file_path, sizeof(last_file_path), last_index);
    NVLOGC_FMT(TAG, "{}: closing FMT log from [{}] thread on core {}, last file path is {}", __func__, thread_name, sched_getcpu(), last_file_path);

    if (threadRunning.exchange(false) == false)
    {
        printf("%s: fmtlog thread already stopped\n", __func__);
        return;
    }

    if (g_fmtlog_thread_id != 0)
    {
        pthread_join(g_fmtlog_thread_id, NULL);
        g_fmtlog_thread_id = 0;
    }
    fmtlog::closeLogFile();
    fmt_log_initiated.store(0);
    printf("%s: FMT log closed\n", __func__);
}

void nvlog_fmtlog_thread_init()
{
    fmtlog::preallocate();
}

void nvlog_fmtlog_thread_init(const char* name)
{
    nvlog_fmtlog_thread_init();
    fmtlog::setThreadName(name);
}

pthread_t nvlog_fmtlog_init(const char* yaml_file, const char* name,void (*exit_hdlr_cb)())
{
    if (fmt_log_initiated.fetch_add(1) != 0)
    {
        printf("FMT log already had been initiated");
        return -1;
    }

    // Start exit watchdog thread on the current CPU core
    int cpu_id = sched_getcpu();
    if (cpu_id < 0)
    {
        NVLOGW_FMT(TAG, "{}: failed to get current CPU core: cpu_id={}, using default core 0", __func__, cpu_id);
        cpu_id = 0;
    }

    if(exit_handler::getInstance().start_exit_watchdog_thread(cpu_id) != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "Failed to start exit watchdog thread on core {}: {}", cpu_id, std::strerror(errno));
    }

    if(exit_hdlr_cb)
    {
        pExitHandler.set_exit_handler_cb(exit_hdlr_cb);
    }

    for(size_t n = 0; n < NVLOG_FMTLOG_NUM_TAGS; n++)
    {
        g_nvlog_component_levels[n] = fmtlog::WRN;
    }

    if (yaml_file == NULL)
    {
        NVLOGC_FMT(TAG, "No nvlog config yaml found, using default nvlog configuration");
        // FIXME temporary set FHGEN to log info
        int i = 0;
        for(auto id : g_nvlog_component_ids)
        {
            if(strcmp (id.name, "FHGEN") == 0)
            {
                g_nvlog_component_levels[i] = fmtlog::INF;
            }
            ++i;
        }
    }
    else
    {
        NVLOGC_FMT(TAG, "Using {} for nvlog configuration", yaml_file);
        try
        {
            const YAML::Node root_node = YAML::LoadFile(yaml_file);
            aerial::check_yaml_version(root_node, yaml_file);
            // size_t num_tags = sizeof(g_nvlog_component_ids) / sizeof(nvlog_component_ids);

            const YAML::Node nvlog_node = root_node["nvlog"];

            int shm_log_level     = nvlog_node["shm_log_level"].as<int>();            // Log level of printing to SHM cache and disk file
            max_log_file_size_bytes = nvlog_node["max_file_size_bytes"].as<size_t>();      // maximum size of each file
            max_rotating_file_num   = nvlog_node["max_rotating_file_num"].as<int32_t>();    // Number of rotating log files
            if (max_rotating_file_num < 2) // At least 2 files are needed, one for reserving initial log, one for rotating tail logs
            {
                NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "{}: invalid yaml configuration: max_rotating_file_num={}, must >= 2, using default={}",
                    __func__, max_rotating_file_num, DEFAULT_MAX_ROTATING_FILE_NUM);
                max_rotating_file_num = DEFAULT_MAX_ROTATING_FILE_NUM;
            }

            std::string log_file_path = nvlog_node["log_file_path"].as<std::string>();
            nvlog_safe_strncpy(logfile_base, log_file_path.c_str(), MAX_PATH_LEN);

            fmtlog::LogLevel fmt_level = getfmtLogLevel(shm_log_level);

            for(size_t n = 0; n < NVLOG_FMTLOG_NUM_TAGS; n++)
            {
                g_nvlog_component_levels[n] = fmt_level;
            }

            if(YAML::Node all_tags = nvlog_node["nvlog_tags"]; all_tags.IsSequence())
            {
                for(YAML::const_iterator tag_node = all_tags.begin(); tag_node != all_tags.end(); ++tag_node)
                {
                    YAML::Node::const_iterator sub_it = tag_node->begin();
                    auto f = sub_it->first;
                    auto key = f.as<std::string>();
                    int itag      = f.as<int>();
                    if(itag >= 0 && itag < NVLOG_DEFAULT_TAG_NUM)
                    {
                        std::string tag_name = (*tag_node)[key.c_str()].as<std::string>();
                        // nvlog_safe_strncpy(tag.tag_name, tag_name.c_str(), cfg->max_tag_len);
                        bool found = false;
                        for(int i = 0; i < NVLOG_FMTLOG_NUM_TAGS; ++i)

                        // (auto &c : g_nvlog_component_ids)
                        {
                            auto &c = g_nvlog_component_ids[i];
                            if (itag == c.id)
                            {
                                found = true;
                                if((*tag_node)["shm_level"])
                                {
                                    int shm_level = (*tag_node)["shm_level"].as<int>();
                                    g_nvlog_component_levels[i] = getfmtLogLevel(shm_level);
                                }
                                // printf("NVLOG tag %s level set to %d\n", tag_name.c_str(), g_nvlog_component_levels[i]);
                            }
                        }

                        if(!found)
                        {
                            printf("NVLOG tag %s do not match the ones specified in nvlog_fmt.hpp, we currently do not support dynamic tag names, skipping\n", tag_name.c_str());
                            continue;
                        }

                    }
                }
            }
        }
        catch(const YAML::BadFile& badFile)
        {
            NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "Failed to load yaml file: {} ({})", yaml_file, badFile.what());
            return -1;
        }
        catch(const YAML::ParserException& parseError)
        {
            NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "{}: failed to parse yaml file: {} ({})", __func__, yaml_file, parseError.what());
            return -1;
        }
        catch(const YAML::Exception& yamlError)
        {
            NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "{}: yaml error while processing file: {} ({})", __func__, yaml_file, yamlError.what());
            return -1;
        }
        catch(const std::exception& ex)
        {
            NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "{}: exception while loading yaml file: {} ({})", __func__, yaml_file, ex.what());
            return -1;
        }
        catch(...)
        {
            NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "{}: unknown exception while loading yaml file: {}", __func__, yaml_file);
            return -1;
        }
    }

    // Overwrite log path if exported AERIAL_LOG_PATH in environment
    char* env = nullptr;
    if(env = std::getenv("AERIAL_LOG_PATH"); env != nullptr)
    {
        NVLOGC_FMT(TAG, "AERIAL_LOG_PATH set to {}\n", env);
        strcpy(logfile_base, env);
    }

    strcat(logfile_base, "/");
    strncat(logfile_base, name, 64);
    NVLOGC_FMT(TAG, "Output log file path {}", logfile_base);
    current_logfile_index.store(0);
    fmtlog::setLogFile(logfile_base, true);
    fmtlog::setHeaderPattern("{HMSf} {l} {t} {O} ");
    fmtlog::setLogCB(logcb_aerial, fmtlog::VEB);
    fmtlog::setLogLevel(fmtlog::VEB);
    fmtlog::setLogQFullCB(logfullcb_aerial,NULL);
    g_fmtlog_thread_id = startNVPollingThread();

    return g_fmtlog_thread_id;
}

extern "C" int is_fmt_log_initiated()
{
    return fmt_log_initiated;
}

extern "C" int fmt_log_level_validate(int level, int itag, const char** stag)
{
    int retVal = 0;
    fmtlog::LogLevel reqested_fmtLogLevel = getfmtLogLevel(level);
    for(int i = 0; i < NVLOG_FMTLOG_NUM_TAGS; ++i)
    {
        auto &c = g_nvlog_component_ids[i];
        if (itag == c.id)
        {
            *stag = c.name;
            if(reqested_fmtLogLevel >= g_nvlog_component_levels[i])
            {
                retVal = 1;
            }
            break;
        }
    }
    return retVal;
}

#define MAX_C_FORMATTED_STR_SIZE 1024

extern "C" void nvlog_vprint_fmt(int level, const char* stag, const char* format, va_list va)
{
    char buffer[MAX_C_FORMATTED_STR_SIZE];
    vsnprintf(buffer, MAX_C_FORMATTED_STR_SIZE, format, va);
    fmtlog::LogLevel fmt_log_level = getfmtLogLevel(level);
    MemtraceDisableScope md; // disable memtrace while this variable is in scope
    FMTLOG_ONCE(fmt_log_level, "[{}] {}", stag, buffer);
}

extern "C" void nvlog_e_vprint_fmt(int level, const char* event, const char* stag, const char* format, va_list va)
{
    char buffer[MAX_C_FORMATTED_STR_SIZE];
    vsnprintf(buffer, MAX_C_FORMATTED_STR_SIZE, format, va);
    fmtlog::LogLevel fmt_log_level = getfmtLogLevel(level);
    MemtraceDisableScope md; // disable memtrace while this variable is in scope
    FMTLOG_ONCE(fmt_log_level, "[{}] [{}] {}", stag, event, buffer);
}

void logcb_cunit(int64_t ns, fmtlog::LogLevel level, fmt::string_view location, size_t basePos, fmt::string_view threadName,
           fmt::string_view msg, size_t bodyPos, size_t logFilePos) {
    if (level >= fmtlog::WRN)
    {
        fmt::print("{}\n", msg);
        fflush(stdout);
    }

    if (logFilePos > max_log_file_size_bytes)
    {
        update_log_filename();
    }
}

extern "C" void nvlog_c_init_fmt(const char *file)
{
    if (fmt_log_initiated.fetch_add(1) != 0)
    {
        printf("FMT log already had been initiated");
        return;
    }

    current_logfile_index.store(0);
    nvlog_safe_strncpy(logfile_base, file, MAX_PATH_LEN);
    printf("FMT log initiated at %s\n", file);
    fmtlog::setLogFile(file, true);
    fmtlog::setHeaderPattern("{HMSf} {l} {t} {O} ");
    fmtlog::setLogCB(logcb_cunit, fmtlog::VEB);
    fmtlog::setLogLevel(fmtlog::INF);
    fmtlog::startPollingThread(1000L * 1000 * 100);
}

extern "C" void nvlog_c_close_fmt()
{
    fmtlog::stopPollingThread();
    fmtlog::closeLogFile();
    fmt_log_initiated.store(0);
}

extern "C" void nvlog_set_log_level_fmt(int log_level)
{
    fmtlog::setLogLevel(getfmtLogLevel(log_level));
}

extern "C" void nvlog_set_max_file_size_fmt(size_t size)
{
    printf("Set FMT log file max size to %lu MB\n", size / 1024 / 1024);
    max_log_file_size_bytes = size;
}

int get_root_path(char* path, int cubb_root_path_relative_num) {
    int length = -1;

    // If CUBB_HOME was set in system environment variables, return it
    char* env = getenv(CONFIG_CUBB_ROOT_ENV);
    if (env != NULL) {
        length = snprintf(path, MAX_PATH_LEN - 1, "%s", env);
        if (path[length - 1] != '/') {
            path[length] = '/';
            path[++length] = '\0';
        }
        return length;
    }

    // Get current process directory, and go up to
    char buf[MAX_PATH_LEN];
    size_t size = readlink("/proc/self/exe", buf, MAX_PATH_LEN - 1);
    if (size > 0 && size < MAX_PATH_LEN) {
        buf[size] = '\0';
        char* tmp = dirname(buf);
        for (int i = 0; i < cubb_root_path_relative_num; i++) {
            tmp = dirname(tmp);
        }
        length = snprintf(path, MAX_PATH_LEN - 1, "%s/", tmp);
    }
    return length;
}

int get_full_path_file(char* dest_buf, const char* relative_path, const char* file_name, int cubb_root_dir_relative_num)
{
    int length = get_root_path(dest_buf, cubb_root_dir_relative_num);

    if(relative_path != NULL)
    {
        length += snprintf(dest_buf + length, MAX_PATH_LEN - length, "%s", relative_path);
        if(dest_buf[length - 1] != '/')
        {
            dest_buf[length]   = '/';
            dest_buf[++length] = '\0';
        }
    }

    if(file_name != NULL)
    {
        length += snprintf(dest_buf + length, MAX_PATH_LEN - length, "%s", file_name);
    }
    NVLOGV_FMT(TAG, "{}: length={} full_path={}", __func__, length, dest_buf);
    return length;
}
