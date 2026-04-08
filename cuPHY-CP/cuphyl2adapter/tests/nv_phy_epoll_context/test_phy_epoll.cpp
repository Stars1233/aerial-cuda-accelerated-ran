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

#include "nv_phy_epoll_context.hpp"
#include <pthread.h>
#include "nvlog.hpp"

#define TAG (NVLOG_TAG_BASE_L2_ADAPTER + 4) // "L2A.EPOLL"

using namespace std;
using namespace nv;

class oneshot_timer_test {
public:
    oneshot_timer_test() :
        timer_fd_p(new timer_fd(1000000, false))
    {
    }

    void oneshot_timer_handler()
    {
        NVLOGC_FMT(TAG, "oneshot_timer_handler\n");
        timer_fd_p->clear();
    }

    int get_fd()
    {
        return timer_fd_p->get_fd();
    }

private:
    unique_ptr<timer_fd> timer_fd_p;
};

class periodic_timer_test {
public:
    periodic_timer_test() :
        timer_fd_p(new timer_fd(1000000, true))
    {
    }

    void periodic_timer_handler()
    {
        NVLOGC_FMT(TAG, "periodic_timer_handler\n");
        timer_fd_p->clear();
    }

    int get_fd()
    {
        return timer_fd_p->get_fd();
    }

private:
    unique_ptr<timer_fd> timer_fd_p;
};

void* timer_fd_test(void* arg)
{
    phy_epoll_context* ep_ctx = (phy_epoll_context*)arg;
    ep_ctx->start_event_loop();
    return ((void*)0);
}

int main(int argc, const char* argv[])
{
    try {
        phy_epoll_context* ep_ctx_p = nullptr;

        try {
            ep_ctx_p = new phy_epoll_context();
        } catch(std::system_error& e) {
            NVLOGC_FMT(TAG, "Failed to create phy_epoll_context: {}", e.what());
            return -1;
        }

        unique_ptr<oneshot_timer_test>                        oneshot_timer_test_p(new oneshot_timer_test());
        unique_ptr<member_event_callback<oneshot_timer_test>> mcb_p(new member_event_callback<oneshot_timer_test>(oneshot_timer_test_p.get(), &oneshot_timer_test::oneshot_timer_handler));

        unique_ptr<periodic_timer_test>                        periodic_timer_test_p(new periodic_timer_test());
        unique_ptr<member_event_callback<periodic_timer_test>> mcb_p1(new member_event_callback<periodic_timer_test>(periodic_timer_test_p.get(), &periodic_timer_test::periodic_timer_handler));
        try {
            ep_ctx_p->add_fd(oneshot_timer_test_p->get_fd(), mcb_p.get());
            ep_ctx_p->add_fd(periodic_timer_test_p->get_fd(), mcb_p1.get());
        } catch(std::exception& e) {
            NVLOGC_FMT(TAG, "Failed to add fd to epoll context: {}", e.what());
            delete ep_ctx_p;
            return -1;
        }

        pthread_t thread_id;
        int       ret = pthread_create(&thread_id, NULL, timer_fd_test, ep_ctx_p);
        if(ret != 0)
        {
            NVLOGC_FMT(TAG, "pthread_create failed: {}", strerror(ret));
            delete ep_ctx_p;
            return -1;
        }
        else
        {
            sleep(1);
            ep_ctx_p->terminate();
            pthread_join(thread_id, NULL);
            delete ep_ctx_p;
        }
        return 0;
    } catch(std::exception& e) {
        NVLOGC_FMT(TAG, "Unhandled exception in main: {}", e.what());
        return -1;
    } catch(...) {
        NVLOGC_FMT(TAG, "Unknown exception in main");
        return -1;
    }
}
