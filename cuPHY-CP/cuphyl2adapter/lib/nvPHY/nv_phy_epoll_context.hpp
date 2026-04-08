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

#if !defined(NV_PHY_EPOLL_CONTEXT_HPP_INCLUDED_)
#define NV_PHY_EPOLL_CONTEXT_HPP_INCLUDED_

//#define dbg

#include "nv_phy_base_common.hpp"
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

namespace nv
{
class event_callback;

////////////////////////////////////////////////////////////////////////
// phy_epoll_context
// Wrapper for the epoll fd and the fds it tries to monitor
class phy_epoll_context {
public:
    //------------------------------------------------------------------
    // phy_epoll_context()
    // throw std::system_error(errno, std::generic_category()) on error
    phy_epoll_context();
    phy_epoll_context(phy_epoll_context&& t) :
        epoll_fd(std::move(t.epoll_fd)),
        fd_cache(std::move(t.fd_cache)),
        to_be_added_fd_list(std::move(t.to_be_added_fd_list)),
        to_be_removed_fd_list(std::move(t.to_be_removed_fd_list)),
        epoll_events(std::move(t.epoll_events)),
        active(std::move(t.active))
    {
    }
    ~phy_epoll_context();
    //------------------------------------------------------------------
    // add_fd()
    void add_fd(int fd, event_callback* cb, uint32_t events = EPOLLIN);
    //------------------------------------------------------------------
    // remove_fd()
    void remove_fd(int fd);
    //------------------------------------------------------------------
    // start_event_loop()
    void start_event_loop();
    //------------------------------------------------------------------
    // terminate()
    void terminate();

private:
    void fd_processing();
    void add_fd_internal(int fd, event_callback* cb, uint32_t events = EPOLLIN);
    void remove_fd_internal(int fd);

    phy_epoll_context& operator=(const phy_epoll_context&) = delete;
    phy_epoll_context(const phy_epoll_context&)            = delete;

    typedef std::vector<struct epoll_event> EventList;
    //------------------------------------------------------------------
    // Data
    int                                      epoll_fd;
    std::unordered_map<int, event_callback*> fd_cache;

    std::unordered_map<int, event_callback*> to_be_added_fd_list;
    std::unordered_set<int>                  to_be_removed_fd_list;

    EventList     epoll_events;
    volatile bool active;
};

class event_callback {
public:
    virtual ~event_callback() {}
    virtual void on_event() {}
};

template <typename T>
class member_event_callback : public event_callback {
public:
    typedef void (T::*event_handler)(void);
    member_event_callback(T* t, event_handler eh) :
        t(t),
        eh(eh) {}

    void on_event() override
    {
        (t->*eh)();
    }

private:
    T*            t;
    event_handler eh;
};

class timer_fd {
public:
    typedef void (*timer_handler)();
    timer_fd(std::size_t interval, bool repeat)
    {
        tfd = timerfd_create(CLOCK_MONOTONIC, 0);
        if(tfd == -1)
        {
            throw std::system_error(errno, std::generic_category());
        }

        time_spec.it_value.tv_sec  = 0;
        time_spec.it_value.tv_nsec = static_cast<long int>(interval);
        if(repeat)
        {
            time_spec.it_interval.tv_sec  = 0;
            time_spec.it_interval.tv_nsec = static_cast<long int>(interval);
        }
        else
        {
            time_spec.it_interval.tv_sec  = 0;
            time_spec.it_interval.tv_nsec = 0;
        }

        timerfd_settime(tfd, 0, &time_spec, nullptr);
    }

    ~timer_fd()
    {
        itimerspec timer = {.it_interval = {0, 0}, .it_value = {0, 0}};
        timerfd_settime(tfd, 0, &timer, nullptr);
        close(tfd);
    }

    int get_fd()
    {
        return tfd;
    }

    //This function must be called after an event is signaled
    void clear()
    {
        uint64_t res;
        //read the fd to clear the EPOLLIN event
        int ret = read(tfd, &res, sizeof(res));

#ifdef dbg
        NVLOGI_FMT(TAG, "read() returned {}, res=%" PRIu64 "\n", ret, res);
#endif
    }

private:
    int        tfd;
    itimerspec time_spec;
};

} // namespace nv

#endif // !defined(NV_PHY_EPOLL_CONTEXT_HPP_INCLUDED_)
