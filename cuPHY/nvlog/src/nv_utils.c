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

#define _GNU_SOURCE

#include <string.h>
#include <libgen.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>
#include <pthread.h>
#include <stdlib.h>
#include <stddef.h>


#include "nvlog.h"
#include "nv_utils.h"

#define MAX_PATH_LEN 1024
#define NVLOG_CUBB_ROOT_ENV "CUBB_HOME"

#define TAG (NVLOG_TAG_BASE_NVLOG + 8) // "NVLOG.UTILS"

int nv_set_sched_fifo_priority(int priority)
{
    struct sched_param param;
    param.__sched_priority = priority;
    pthread_t thread_me    = pthread_self();
    if(pthread_setschedparam(thread_me, SCHED_FIFO, &param) != 0)
    {
        NVLOGE_NO(TAG, AERIAL_THREAD_API_EVENT, "%s: line %d errno=%d: %s", __func__, __LINE__, errno, strerror(errno));
        return -1;
    }
    else
    {
        NVLOGI(TAG, "%s: OK: thread=%ld priority=%d", __func__, thread_me, priority);
        return 0;
    }
}

int nv_assign_thread_cpu_core(int cpu_id)
{
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu_id, &mask);
    int ret;
    if ((ret = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask)) != 0) {
        NVLOGE_NO(TAG, AERIAL_THREAD_API_EVENT, "%s: cpu_core=%d ret=%d errno=%d: %s", __func__,
                cpu_id, ret, errno, strerror(errno));
        return -1;
    } else {
        NVLOGI(TAG, "%s: OK: thread=%ld cpu_id=%d", __func__, pthread_self(), cpu_id);
        return 0;
    }
}

static int get_process_parent_path(char* path, int step)
{
    int length = -1;

    // If CUBB_HOME was set in system environment variables, return it
    char* env = getenv(NVLOG_CUBB_ROOT_ENV);
    if (env != NULL) {
        length = snprintf(path, MAX_PATH_LEN - 1, "%s", env);
        if (path[length - 1] != '/') {
            path[length] = '/';
            path[++length] = '\0';
        }
        return length;
    }


    // Get current process directory, and go up to
    char   buf[MAX_PATH_LEN];
    size_t size = readlink("/proc/self/exe", buf, MAX_PATH_LEN - 1);
    if(size > 0 && size < MAX_PATH_LEN)
    {
        buf[size] = '\0';
        NVLOGV(TAG, "%s: readlink=%s size=%lu", __func__, buf, size);
        char* tmp = dirname(buf);
        for(int i = 0; i < step; i++)
        {
            tmp = dirname(tmp);
        }
        length = snprintf(path, MAX_PATH_LEN - 1, "%s/", tmp);
    }
    NVLOGD(TAG, "%s: path=%s length=%d", __func__, buf, length);
    return length;
}

// Convert process relative path to absolute path
int nv_get_absolute_path(char* absolute_path, const char* relative_path)
{
    if(absolute_path == NULL || relative_path == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_INVALID_PARAM_EVENT, "%s: invalid parameters", __func__);
        return -1;
    }

    int         step  = 0;
    const char* start = relative_path;

    while(*start == '.')
    {
        if(strncmp(start, "../", 3) == 0)
        {
            step++;
            start += 3;
        }
        else if(strncmp(start, "./", 2) == 0)
        {
            start += 2;
        }
        else
        {
            NVLOGE_NO(TAG, AERIAL_INVALID_PARAM_EVENT, "%s: invalid relative path: %s", __func__, relative_path);
            return -1;
        }
    }

    int length = get_process_parent_path(absolute_path, step);
    length += snprintf(absolute_path + length, MAX_PATH_LEN - length, "%s", start);
    NVLOGD(TAG, "%s: length=%d full_path=%s", __func__, length, absolute_path);

    return length;
}
