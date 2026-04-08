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

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>    /* For O_* constants */
#include <sys/stat.h> /* For mode constants */
#include <sys/types.h>
#include <errno.h>

#include <pthread.h>
#include <signal.h>

#include "stat_log.h"
#include "nvlog.h"

#define TAG (NVLOG_TAG_BASE_NVLOG + 4) // "NVLOG.STAT"

// #define DEFAULT_CLOCK_SOURCE CLOCK_MONOTONIC
#define DEFAULT_CLOCK_SOURCE CLOCK_REALTIME

#define STAT_LONG_MIN_VALUE (-(__LONG_MAX__ - 1))
#define STAT_LONG_MAX_VALUE (__LONG_MAX__)

#define OVERFLOW_VALUE (1L << (sizeof(int64_t) * 8 - 2))

typedef struct
{
    char name[STAT_NAME_MAX_LEN];

    clockid_t    clk_src;   // Clock source used for time stamp
    print_mode_t mode;      // Decide the statistic period by timer or counter
    int64_t      period;    // Statistic period
    int          log_level; // Statistic printing log level

    int32_t counter;
    int64_t max; // Max value
    int64_t min; // Min value
    int64_t sum;
    int64_t carry;

    int64_t limit_max; // Limit of maximal value
    int64_t limit_min; // Limit of minimal value

    // Print when timer timeout/interrupt
    timer_t           timer;
    struct sigevent   sigev;
    struct itimerspec its;

    // Last element time stamp
    struct timespec ts_last;
} priv_data_t;

int assert_time_order(struct timespec ts_old, struct timespec ts_new)
{
    if(ts_old.tv_sec < ts_new.tv_sec)
    {
        return 1;
    }
    else if(ts_old.tv_sec > ts_new.tv_sec)
    {
        return 0;
    }
    else
    {
        if(ts_old.tv_nsec <= ts_new.tv_nsec)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }
}

static inline priv_data_t* get_private_data(stat_log_t* stat)
{
    return (priv_data_t*)((int8_t*)stat + sizeof(stat_log_t));
}

static void timer_handler(union sigval sigv)
{
    if(sigv.sival_ptr == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_INVALID_PARAM_EVENT, "%s: sigv.sival_ptr==NULL", __func__);
        return;
    }

    stat_log_t*  stat      = (stat_log_t*)sigv.sival_ptr;
    priv_data_t* priv_data = get_private_data(stat);

    stat->print(stat);
    stat->init(stat);
}

static int stat_log_timer_open(stat_log_t* stat, int64_t ns)
{
    priv_data_t* priv_data = get_private_data(stat);

    long ts_sec  = ns / (1000L * 1000 * 1000);
    long ts_nsec = ns % (1000L * 1000 * 1000);

    memset(&priv_data->sigev, 0, sizeof(priv_data->sigev));
    priv_data->sigev.sigev_notify          = SIGEV_THREAD;
    priv_data->sigev.sigev_value.sival_ptr = stat;
    priv_data->sigev.sigev_notify_function = timer_handler;

    priv_data->its.it_interval.tv_sec  = ts_sec;
    priv_data->its.it_interval.tv_nsec = ts_nsec;
    priv_data->its.it_value.tv_sec     = ts_sec;
    priv_data->its.it_value.tv_nsec    = ts_nsec;

    if(timer_create(CLOCK_REALTIME, &priv_data->sigev, &priv_data->timer) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_CLOCK_API_EVENT, "%s: timer_create failed", priv_data->name);
        return -1;
    }

    if(timer_settime(priv_data->timer, 0, &priv_data->its, NULL) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_CLOCK_API_EVENT, "%s: timer_settime failed. Error: %s", priv_data->name, strerror(errno));
        return -1;
    }
    else
    {
        NVLOGI(TAG, "%s: period=%ld OK", priv_data->name, ns);
        return 0;
    }
}

static int stat_log_init(stat_log_t* stat)
{
    priv_data_t* priv_data = get_private_data(stat);

    priv_data->min     = STAT_LONG_MAX_VALUE; // Initiate to maximum value
    priv_data->max     = STAT_LONG_MIN_VALUE; // Initiate to minimum value
    priv_data->sum     = 0;
    priv_data->carry   = 0;
    priv_data->counter = 0;
    return 0;
}

static int stat_log_set_clock_source(stat_log_t* stat, clockid_t clk_src)
{
    priv_data_t* priv_data = get_private_data(stat);
    priv_data->clk_src     = clk_src;
    return 0;
}

static int stat_log_set_log_level(stat_log_t* stat, int log_level)
{
    priv_data_t* priv_data = get_private_data(stat);
    priv_data->log_level   = log_level;
    return 0;
}

static int stat_log_set_limit(stat_log_t* stat, int64_t min, int64_t max)
{
    priv_data_t* priv_data = get_private_data(stat);
    priv_data->limit_max   = max;
    priv_data->limit_min   = min;
    return 0;
}

static int stat_log_print(stat_log_t* stat)
{
    priv_data_t* priv_data = get_private_data(stat);

    int64_t avg = 0;
    if(priv_data->counter > 0)
    {
        if(priv_data->carry != 0)
        {
            NVLOG_C(priv_data->log_level, TAG, "[%s] overflow int64_t %ld calculate with __int128_t\n", priv_data->name, priv_data->carry);
            __int128_t total = (__int128_t)OVERFLOW_VALUE * (__int128_t)priv_data->carry + priv_data->sum;
            avg              = total / priv_data->counter;
        }
        else
        {
            avg = priv_data->sum / priv_data->counter;
        }

        if(priv_data->min < priv_data->limit_min || priv_data->max > priv_data->limit_max)
        {
            NVLOGC(TAG, "[%s][N=%d][MIN=%ld AVG=%ld MAX=%ld] exceed limit\n", priv_data->name, priv_data->counter, priv_data->min, avg, priv_data->max);
        }
        else
        {
            NVLOG_C(priv_data->log_level, TAG, "[%s][N=%d][MIN=%ld AVG=%ld MAX=%ld]\n", priv_data->name, priv_data->counter, priv_data->min, avg, priv_data->max);
        }
    }
    else
    {
        NVLOG_C(priv_data->log_level, TAG, "[%s][N=%d][MIN=NA AVG=NA MAX=NA]\n", priv_data->name, priv_data->counter);
    }

    return 0;
}

static int32_t stat_log_get_counter(stat_log_t* stat)
{
    priv_data_t* priv_data = get_private_data(stat);
    return priv_data->counter;
}

static int64_t stat_log_get_min_value(stat_log_t* stat)
{
    priv_data_t* priv_data = get_private_data(stat);
    return (priv_data->counter > 0) ? priv_data->min : 0;
}

static int64_t stat_log_get_max_value(stat_log_t* stat)
{
    priv_data_t* priv_data = get_private_data(stat);
    return (priv_data->counter > 0) ? priv_data->max : 0;
}

static int64_t stat_log_get_avg_value(stat_log_t* stat)
{
    priv_data_t* priv_data = get_private_data(stat);
    return (priv_data->counter > 0) ? priv_data->sum / priv_data->counter : 0;
}

static int stat_log_counting_print(stat_log_t* stat)
{
    priv_data_t* priv_data = get_private_data(stat);
    if(priv_data->counter >= priv_data->period)
    {
        stat_log_print(stat);
        stat_log_init(stat);
    }
    return 0;
}

static int stat_log_add(stat_log_t* stat, int64_t value)
{
    priv_data_t* priv_data = get_private_data(stat);

    if(priv_data->max < value)
    {
        priv_data->max = value;
    }
    if(priv_data->min > value)
    {
        priv_data->min = value;
    }

    priv_data->sum += value;
    if(priv_data->sum > OVERFLOW_VALUE)
    {
        priv_data->sum -= OVERFLOW_VALUE;
        priv_data->carry++;
    }
    else if(priv_data->sum < 0 - OVERFLOW_VALUE)
    {
        priv_data->sum += OVERFLOW_VALUE;
        priv_data->carry--;
    }
    priv_data->counter++;

    int ret = 0;
    if(priv_data->min < priv_data->limit_min || priv_data->max > priv_data->limit_max)
    {
        ret = -1;
    }

    if(priv_data->mode == STAT_MODE_COUNTER)
    {
        stat_log_counting_print(stat);
        ret = 1;
    }

    return ret;
}

static int stat_log_time_interval(stat_log_t* stat)
{
    priv_data_t* priv_data = get_private_data(stat);

    struct timespec ts_now;
    if(clock_gettime(priv_data->clk_src, &ts_now) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_CLOCK_API_EVENT, "%s clock_gettime %s", priv_data->name, strerror(errno));
        return -1;
    }

    if(priv_data->ts_last.tv_sec != 0)
    {
        stat_log_add(stat, nvlog_timespec_interval(&priv_data->ts_last, &ts_now));
    }
    priv_data->ts_last.tv_sec  = ts_now.tv_sec;
    priv_data->ts_last.tv_nsec = ts_now.tv_nsec;

    return 0;
}

static int stat_log_close(stat_log_t* stat)
{
    priv_data_t* priv_data = get_private_data(stat);

    int ret = 0;
    if(priv_data->mode == STAT_MODE_TIMER)
    {
        /* Note: Should not free(stat) because timer_handler() may be called after timer_delete().
         * See manual: https://linux.die.net/man/2/timer_delete
         * The treatment of any pending signal generated by the deleted timer is unspecified.
         * Will not fix it because it doesn't cause serious issue and only used in test code.
         */
        if(timer_delete(priv_data->timer) < 0)
        {
            NVLOGE_NO(TAG, AERIAL_CLOCK_API_EVENT, "%s: timer_delete %s failed", __func__, priv_data->name);
            return -1;
        }
        else
        {
            NVLOGI(TAG, "%s: name=%s OK", __func__, priv_data->name);
            return 0;
        }
    }
    else
    {
        NVLOGI(TAG, "%s: name=%s OK", __func__, priv_data->name);
        free(stat);
        return 0;
    }
}

stat_log_t* stat_log_open(const char* name, int mode, int64_t period)
{
    if(name == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_INVALID_PARAM_EVENT, "%s: invalid configuratio", __func__);
        return NULL;
    }

    int         size = sizeof(stat_log_t) + sizeof(priv_data_t);
    stat_log_t* stat = malloc(size);
    if(stat == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: memory malloc failed", __func__);
        return NULL;
    }

    memset(stat, 0, size);

    priv_data_t* priv_data = get_private_data(stat);
    nvlog_safe_strncpy(priv_data->name, name, STAT_NAME_MAX_LEN);
    priv_data->clk_src   = DEFAULT_CLOCK_SOURCE;
    priv_data->mode      = mode;
    priv_data->period    = period;
    priv_data->log_level = NVLOG_INFO;
    priv_data->limit_min = STAT_LONG_MIN_VALUE;
    priv_data->limit_max = STAT_LONG_MAX_VALUE;

    stat->init             = stat_log_init;
    stat->add              = stat_log_add;
    stat->time_interval    = stat_log_time_interval;
    stat->set_clock_source = stat_log_set_clock_source;
    stat->set_log_level    = stat_log_set_log_level;
    stat->set_limit        = stat_log_set_limit;

    stat->get_counter   = stat_log_get_counter;
    stat->get_min_value = stat_log_get_min_value;
    stat->get_max_value = stat_log_get_max_value;
    stat->get_avg_value = stat_log_get_avg_value;

    stat->print = stat_log_print;
    stat->close = stat_log_close;

    if(priv_data->mode == STAT_MODE_TIMER)
    {
        if(stat_log_timer_open(stat, priv_data->period) < 0)
        {
            NVLOGE_NO(TAG, AERIAL_NVLOG_EVENT, "%s: name=%s Failed", __func__, name);
            free(stat);
            return NULL;
        }
    }

    if(stat_log_init(stat) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVLOG_EVENT, "%s: name=%s Failed", __func__, name);
        stat_log_close(stat);
        return NULL;
    }
    else
    {
        NVLOGI(TAG, "%s: name=%s OK", __func__, name);
        return stat;
    }
}
