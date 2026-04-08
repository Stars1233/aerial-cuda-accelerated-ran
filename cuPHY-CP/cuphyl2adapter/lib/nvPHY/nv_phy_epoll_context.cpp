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

#include <errno.h>

#include "nv_phy_epoll_context.hpp"
#include "nvlog.hpp"
#include "memtrace.h"

#define TAG (NVLOG_TAG_BASE_L2_ADAPTER + 4) // "L2A.EPOLL"

namespace nv
{
////////////////////////////////////////////////////////////////////////
// phy_epoll_context::phy_epoll_context()
phy_epoll_context::phy_epoll_context() :
    active(false)
{
    epoll_fd = epoll_create1(EPOLL_CLOEXEC);
    if(epoll_fd == -1)
    {
        throw std::system_error(errno, std::generic_category());
    }
}

////////////////////////////////////////////////////////////////////////
// phy_epoll_context::~phy_epoll_context()
phy_epoll_context::~phy_epoll_context()
{
    if(close(epoll_fd) == -1)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "epoll fd close error: {}", strerror(errno));
    }
}

////////////////////////////////////////////////////////////////////////
// phy_epoll_context::add_fd()
void phy_epoll_context::add_fd(int fd, event_callback* cb, uint32_t events)
{
    if(epoll_fd == ipc_base::INVALID_FD)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "{}: Invalid fd: {}", __func__, fd);
        return;
    }

    if(cb == nullptr || fd == ipc_base::INVALID_FD)
    {
        std::string err_str = "";
        err_str += fd == ipc_base::INVALID_FD ? " Invalid fd " : "";
        err_str += cb == nullptr ? " NULL callback " : "";
        throw std::runtime_error(err_str + std::string(__PRETTY_FUNCTION__));
    }

    if(to_be_added_fd_list.find(fd) != to_be_added_fd_list.end())
    {
        to_be_added_fd_list[fd] = cb;
    }
    else
    {
        to_be_added_fd_list.insert(std::make_pair(fd, cb));
    }
}

////////////////////////////////////////////////////////////////////////
// phy_epoll_context::remove_fd()
void phy_epoll_context::remove_fd(int fd)
{
    if(epoll_fd == ipc_base::INVALID_FD)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "{}: Invalid epoll_fd: {}", __func__, epoll_fd);
        return;
    }

    if(fd == ipc_base::INVALID_FD)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "{}: Invalid fd: {}", __func__, fd);
        return;
    }
    to_be_removed_fd_list.insert(fd);
}

////////////////////////////////////////////////////////////////////////
// phy_epoll_context::add_fd_internal()
void phy_epoll_context::add_fd_internal(int fd, event_callback* cb, uint32_t events)
{
    if(epoll_fd == ipc_base::INVALID_FD)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "{}: Invalid fd: {}", __func__, fd);
        return;
    }

    if(cb == nullptr || fd == ipc_base::INVALID_FD)
    {
        std::string err_str = "";
        err_str += fd == ipc_base::INVALID_FD ? " Invalid fd " : "";
        err_str += cb == nullptr ? " NULL callback " : "";
        throw std::runtime_error(err_str + std::string(__PRETTY_FUNCTION__));
    }

    epoll_event event;
    std::memset(&event, 0, sizeof(event));
    event.data.fd = fd;
    event.events  = events;

    if(epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &event) == -1)
    {
        throw std::system_error(errno, std::generic_category());
    }

    //increase epoll_events size
    {
        MemtraceDisableScope mds;

        epoll_events.push_back(event);

        fd_cache.insert(std::make_pair(fd, cb));
    }
}

////////////////////////////////////////////////////////////////////////
// phy_epoll_context::remove_fd_internal()
void phy_epoll_context::remove_fd_internal(int fd)
{
    if(epoll_fd == ipc_base::INVALID_FD)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "{}: Invalid epoll_fd: {}", __func__, epoll_fd);
        return;
    }

    if(fd == ipc_base::INVALID_FD)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "{}: Invalid fd: {}", __func__, fd);
        return;
    }

    std::unordered_map<int, event_callback*>::iterator it = fd_cache.find(fd);
    if(it == fd_cache.end())
    {
        throw std::runtime_error("fd not found" + std::to_string(fd));
    }

    //reduce epoll_events size
    epoll_events.pop_back();

    fd_cache.erase(it);
    if(epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, NULL) == -1)
    {
        //The fd maybe closed before epoll delete, don't throw error here.
        //throw std::system_error(errno, std::generic_category());
    }
}

////////////////////////////////////////////////////////////////////////
// phy_epoll_context::fd_processing()
void phy_epoll_context::fd_processing()
{
    if(to_be_added_fd_list.empty() && to_be_removed_fd_list.empty())
    {
        return;
    }

    for(std::unordered_map<int, event_callback*>::iterator it = to_be_added_fd_list.begin(); it != to_be_added_fd_list.end(); it++)
    {
        add_fd_internal(it->first, it->second, EPOLLIN);
    }
    to_be_added_fd_list.clear();

    for(std::unordered_set<int>::iterator it = to_be_removed_fd_list.begin(); it != to_be_removed_fd_list.end(); it++)
    {
        remove_fd_internal(*it);
    }
    to_be_removed_fd_list.clear();
}

////////////////////////////////////////////////////////////////////////
// phy_epoll_context::start_event_loop()
void phy_epoll_context::start_event_loop()
{
    if(epoll_fd == ipc_base::INVALID_FD)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "{}: Invalid epoll_fd: {}", __func__, epoll_fd);
        return;
    }

    if(to_be_added_fd_list.empty())
    {
        return;
    }

    active = true;
    while(active)
    {
        //Add/remove fd on the fly
        fd_processing();
        int fd_events;
        do {
            // epoll_wait() may return EINTR when get unexpected signal SIGSTOP from system
            fd_events = epoll_wait(epoll_fd, epoll_events.data(), epoll_events.size(), -1);
        } while (fd_events == -1 && errno == EINTR);

        if(fd_events < 0)
        {
            NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "epoll_wait failed: epoll_fd={} fd_events={} err={} - {}", epoll_fd, fd_events, errno, strerror(errno));
            throw std::system_error(errno, std::generic_category());
        }

        for(int i = 0; i < fd_events; i++)
        {
            event_callback* cb       = fd_cache[epoll_events[i].data.fd];
            epoll_event     fd_event = epoll_events[i];
            if(fd_event.events & EPOLLIN)
            {
                cb->on_event();
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////
// phy_epoll_context::terminate()
void phy_epoll_context::terminate()
{
    active = false;
}

} // namespace nv
