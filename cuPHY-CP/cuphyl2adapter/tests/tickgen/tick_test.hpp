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

#include <stdint.h>

#include <chrono>
#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <chrono>
#include <queue>
#include <sys/epoll.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <atomic>
#include <sstream>
#include <memory>
#include <cstring>
#include <iostream>
#include <sys/timerfd.h>
#include <unistd.h>
#include <inttypes.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cstdio>
#include <fstream>
#include <functional>

using namespace std;
using namespace std::chrono;
// using namespace nv;

struct thread_config
{
    std::string name;
    size_t      cpu_affinity;
    int         sched_priority;
};

inline int assign_thread_name(const char* name)
{
    return pthread_setname_np(pthread_self(), name);
}

inline int assign_thread_cpu_core(int cpu_id)
{
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu_id, &mask);
    return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
}

inline int assign_thread_priority(int priority)
{
    struct sched_param param;
    param.__sched_priority = priority;
    pthread_t thread_me    = pthread_self();
    return pthread_setschedparam(thread_me, SCHED_FIFO, &param);
}

inline int config_thread_property(thread_config& config)
{
    int ret = 0;
    if(assign_thread_name(config.name.c_str()) != 0)
    {
        ret = -1;
    }
    if(assign_thread_cpu_core(config.cpu_affinity) != 0)
    {
        ret = -2;
    }
    if(assign_thread_priority(config.sched_priority) != 0)
    {
        ret = -3;
    }
    return ret;
}

inline std::size_t mu_to_ns(uint8_t mu)
{
    switch(mu)
    {
    case 0: return 1000 * 1000;
    case 1: return 500 * 1000;
    case 2: return 250 * 1000;
    case 3: return 125 * 1000;
    case 4: return 625 * 100;
    }
    return 0;
}
