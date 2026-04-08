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

#include "aerial-fh-driver/pcap_logger.hpp"
#include <rte_mempool.h>
#include <chrono>
#include <cinttypes>
#include <ctime>
#include <sys/time.h>
namespace aerial_fh {

static constexpr size_t SHM_CACHE_SIZE_BYTES = (1UL << 27); // 128MB
static constexpr size_t MAX_FILE_SIZE_BYTES = (1UL << 28);  // 256MB
static constexpr size_t MAX_DATA_SIZE_BYTES = 8000;

static void configure_shm_logger(std::array<shmlogger_t*,kMaxPcapLoggerTypes> &logger_objs)
{
    shmlogger_config_t shm_cfg{};
    shm_cfg.save_to_file = 1;               // Start a background thread to save SHM cache to file before overflow
    shm_cfg.shm_cache_size = SHM_CACHE_SIZE_BYTES;
    shm_cfg.max_file_size = MAX_FILE_SIZE_BYTES;
    shm_cfg.file_saving_core = -1;          // CPU core ID for the background file saving if enabled
    shm_cfg.shm_caching_core = -1;          // CPU core ID for the background copying to shared memory if enabled
    shm_cfg.max_data_size = MAX_DATA_SIZE_BYTES;

    logger_objs.at(static_cast<int>(PcapLoggerType::DL_CPLANE)) = shmlogger_open(1, "dl_cplane_pcap", &shm_cfg);
    if (!logger_objs.at(static_cast<int>(PcapLoggerType::DL_CPLANE))) {
        throw std::runtime_error("Failed to create DL C-plane SHM logger");
    }

    logger_objs.at(static_cast<int>(PcapLoggerType::UL_CPLANE)) = shmlogger_open(1, "ul_cplane_pcap", &shm_cfg);
    if (!logger_objs.at(static_cast<int>(PcapLoggerType::UL_CPLANE))) {
        throw std::runtime_error("Failed to create UL C-plane SHM logger");
    }

    logger_objs.at(static_cast<int>(PcapLoggerType::DL_UPLANE)) = shmlogger_open(1, "dl_uplane_pcap", &shm_cfg);
    if (!logger_objs.at(static_cast<int>(PcapLoggerType::DL_UPLANE))) {
        throw std::runtime_error("Failed to create DL U-plane SHM logger");
    }
}

void PcapLogger::init(const struct PcapLoggerCfg &cfg)
{
    if (initialized_) {
        return;
    }

    logger_cfg_  = cfg;

    if (logger_cfg_.enableDlCplane || logger_cfg_.enableUlCplane || logger_cfg_.enableDlUplane) {
        configure_shm_logger(logger_objs_);

        // Create ring buffer for MP/SC operation
        ring_ = rte_ring_create(
            "pcap_logger_ring",
            RING_SIZE,
            SOCKET_ID_ANY,
            RING_F_SC_DEQ  // Multi-producer, single consumer. By default queues are MP/MC
        );

        if (!ring_) {
            throw std::runtime_error("Failed to create ring buffer for PcapLogger\n");
        }
    }

    initialized_ = true;

    printf ("PCAP_LOGGER Initialized\n");
}

PcapLogger::PcapLogger()
    : logger_cfg_{0}
    , ring_(nullptr)
    , logger_objs_{nullptr}
    , running_(false)
    , initialized_(false)
{}

PcapLogger::~PcapLogger()
{
    stop();

    // Kill the ring buffer
    if (ring_) {
        rte_ring_free(ring_);
    }

    for (int type = 0; type < kMaxPcapLoggerTypes; ++type) {
        if (logger_objs_.at(type)) {
            shmlogger_close (logger_objs_.at(type));
            logger_objs_.at(type) = nullptr;
        }
    }
}

void PcapLogger::start()
{
    if (running_ || !initialized_) {
        return;
    }

    if (!logger_cfg_.enableDlUplane && !logger_cfg_.enableDlCplane && !logger_cfg_.enableUlCplane) {
        return;
    }

    // logger is initialized, and thread is not yet running.
    running_ = true;
    consumer_thread_ = std::thread(&PcapLogger::run, this);
    int name_st = pthread_setname_np(consumer_thread_ .native_handle(), "pcap_log_thread");

    if (name_st != 0)
    {
        printf ("PCAP_LOGGER_THREAD Thread pthread_setname_np() failed with status: %s\n",std::strerror(name_st));
        return;
    }

    int         policy = 0;
    int         status = 0;

    cpu_set_t cpuset{};
    CPU_ZERO(&cpuset);
    CPU_SET(logger_cfg_.threadAffinity, &cpuset);
    status = pthread_setaffinity_np(consumer_thread_.native_handle(), sizeof(cpu_set_t), &cpuset);
    if(status)
    {
        printf("PCAP_LOGGER_THREAD setaffinity_np  failed with status : %s\n" , std::strerror(status));
        return;
    }

    int priority_val = logger_cfg_.threadPriority;
    if(priority_val > 0)
    {
        sched_param sch{};
        // Set thread priority
        status = pthread_getschedparam(consumer_thread_.native_handle(), &policy, &sch);
        if(status != 0)
        {
            printf ("PCAP_LOGGER_THREAD pthread_getschedparam failed with status : %s\n", std::strerror(status));
            return;
        }
        sch.sched_priority = priority_val;

#ifdef ENABLE_SCHED_FIFO_ALL_RT
        status = pthread_setschedparam(consumer_thread_.native_handle(), SCHED_FIFO, &sch);
        if(status != 0)
        {
            printf ("PCAP_LOGGER_THREAD setschedparam failed with status : %s" , std::strerror(status));
            return;
        }
#endif
    }
    printf ("PCAP_LOGGER_THREAD SETUP successfully");
}

void PcapLogger::stop()
{
    if (running_) {
        running_ = false;
        if (consumer_thread_.joinable()) {
            consumer_thread_.join();
        }
    }
}

bool PcapLogger::enqueue(rte_mbuf* mbuf, PcapLoggerType  pkt_type)
{
    if (!running_ || !initialized_) {
        return false;
    }

    // Create PacketLogInfo on the heap
    auto info = new PacketLogInfo(mbuf, pkt_type);

    // Multi-producer enqueue
    if (rte_ring_mp_enqueue(ring_, info) != 0) {
        delete info;
        return false;
    }

    return true;
}

static std::string generate_filename(PcapLoggerType type)
{
    struct timeval tv;
    struct tm* timeinfo;
    char timestamp[32];
    static std::array<uint64_t,kMaxPcapLoggerTypes> burst_counters_;

    gettimeofday(&tv, NULL);
    timeinfo = localtime(&tv.tv_sec);
    // Check for null pointer from localtime
    if(unlikely(timeinfo == nullptr))
    {
        // Fallback: use raw timestamp
        snprintf(timestamp, sizeof(timestamp), "%" PRId64, static_cast<int64_t>(tv.tv_sec));
    }
    else
    {
        strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", timeinfo);
    }

    uint64_t counter = 0;

    char filename[512];
    snprintf(filename, sizeof(filename),
             "%s_%s_C%" PRIu64,
             timestamp,
             logger_names.at(static_cast<int>(type)).data(),
             ++burst_counters_.at(static_cast<int>(type)));

    return std::string(filename);
}

int PcapLogger::process_queue(uint32_t numPkts [kMaxPcapLoggerTypes])
{
    void* dequeued = nullptr;
    int ret = rte_ring_sc_dequeue(ring_, &dequeued);

    if (ret != 0) {
        // Dequeue unsuccessful - bail early
        return ret;
    }

    // Successful dequeue - process the dequeued element.
    PacketLogInfo* info = static_cast<PacketLogInfo*>(dequeued);
    rte_mbuf* mbuf = info->mbuf;

    // Get packet metadata
    auto pkt = rte_pktmbuf_mtod(mbuf, uint8_t*);
    auto ecpri_len = oran_cmsg_get_ecpri_payload(pkt);

    // Write packet to appropriate SHM logger
    shmlogger_t* logger = logger_objs_.at(static_cast<int>(info->pktType));

    shmlogger_save_fh_buffer(logger,
                             reinterpret_cast<const char*>(pkt),
                             ecpri_len + ORAN_ETH_HDR_SIZE + 4, // +4 for padding
                             0,
                             0); // timestamp not used

    ++numPkts[static_cast<int>(info->pktType)];

    // Free the mbuf and info structure
    rte_pktmbuf_free(mbuf);
    delete info;

    return 0;
}

static void collectSHM(const PcapLoggerCfg &logger_cfg, std::array<shmlogger_t*,kMaxPcapLoggerTypes> &logger_objs,  uint32_t numPkts [kMaxPcapLoggerTypes])
{

    char output_filename [400];

    if (numPkts[static_cast<int>(PcapLoggerType::UL_CPLANE)]) {
        std::string fname = generate_filename(PcapLoggerType::UL_CPLANE);
        std::strcpy (output_filename, fname.c_str());
        // Collect logs periodically
        shmlog_collect_params_t params = {
            .prefix = "ul_cplane",
            .type = "pcap",
            .path = logger_cfg.output_path.c_str(),
            .fh_collect = 1,
            .output_filename = output_filename
        };

        shmlogger_collect_ex(&params);
        shmlogger_reset(logger_objs.at(static_cast<int>(PcapLoggerType::UL_CPLANE)));
    }

    if (numPkts[static_cast<int>(PcapLoggerType::DL_CPLANE)]) {
        std::string fname = generate_filename(PcapLoggerType::DL_CPLANE);
        std::strcpy (output_filename, fname.c_str());

        // Collect logs periodically
        shmlog_collect_params_t params = {
            .prefix = "dl_cplane",
            .type = "pcap",
            .path = logger_cfg.output_path.c_str(),
            .fh_collect = 1,
            .output_filename = output_filename
        };

        shmlogger_collect_ex(&params);
        shmlogger_reset(logger_objs.at(static_cast<int>(PcapLoggerType::DL_CPLANE)));
    }

    if (numPkts[static_cast<int>(PcapLoggerType::DL_UPLANE)]) {
        std::string fname = generate_filename(PcapLoggerType::DL_UPLANE);
        std::strcpy (output_filename, fname.c_str());

        // Collect logs periodically
        shmlog_collect_params_t params = {
            .prefix = "dl_uplane",
            .type = "pcap",
            .path = logger_cfg.output_path.c_str(),
            .fh_collect = 1,
            .output_filename = output_filename
        };

        shmlogger_collect_ex(&params);
        shmlogger_reset(logger_objs[static_cast<int>(PcapLoggerType::DL_UPLANE)]);
    }
}

void PcapLogger::run() {
    static unsigned long overall = 0;
    while (running_) {
        uint32_t numPkts [kMaxPcapLoggerTypes] = {0};
        // Dequeue all the queued elements
        while (process_queue(numPkts) == 0);
        collectSHM(logger_cfg_, logger_objs_, numPkts);
        if (numPkts[static_cast<int>(PcapLoggerType::UL_CPLANE)] ||
            numPkts[static_cast<int>(PcapLoggerType::DL_CPLANE)]) {
            overall +=  numPkts[static_cast<int>(PcapLoggerType::UL_CPLANE)];
            overall +=  numPkts[static_cast<int>(PcapLoggerType::DL_CPLANE)];
        }
        // Sleep to periodically poll
        std::this_thread::sleep_for(std::chrono::nanoseconds(SLEEP_NS));
    }

    if (running_ == false) {
        // Thread is instructed to stop, so process any remaining elements in
        // the queue before stopping
        uint32_t numPkts [kMaxPcapLoggerTypes] = {0};
        while (process_queue(numPkts) == 0);
        collectSHM(logger_cfg_, logger_objs_, numPkts);
    }
}

} // namespace aerial_fh
