/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <errno.h>
#include <limits.h>

static long
perf_event_open(struct perf_event_attr *attr, pid_t pid, int cpu,
                int group_fd, unsigned long flags)
{
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

// Hard-code the tracepoint id for syscalls:sys_enter_futex
// You must read it from sysfs and pass it in, or look it up once:
//
//   cat /sys/kernel/debug/tracing/events/syscalls/sys_enter_futex/id
//
int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <tracepoint_id> <pid>\n", argv[0]);
        return 1;
    }

    char *end;
    long val;

    errno = 0;
    val = strtol(argv[1], &end, 10);
    if (end == argv[1] || *end != '\0' || errno == ERANGE || val < 1 || val > INT_MAX) {
        fprintf(stderr, "Invalid tracepoint_id: '%s' (must be a positive integer)\n", argv[1]);
        return 1;
    }
    int tracepoint_id = (int)val;

    errno = 0;
    val = strtol(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || errno == ERANGE || val < -1 || val > INT_MAX) {
        fprintf(stderr, "Invalid pid: '%s' (must be an integer, e.g. 0 for self or target PID)\n", argv[2]);
        return 1;
    }
    pid_t pid = (pid_t)val;

    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type = PERF_TYPE_TRACEPOINT;
    attr.size = sizeof(attr);
    attr.config = tracepoint_id;   // syscalls:sys_enter_futex id
    attr.sample_period = 1;
    attr.sample_type = PERF_SAMPLE_RAW;
    attr.wakeup_events = 1;

    int fd = perf_event_open(&attr, pid, -1, -1, 0);
    if (fd == -1) {
        perror("perf_event_open");
        return 1;
    }

    // Choose a mmap length like perf: 1 + 2^n pages.
    // Here n = 3 -> 1 + 8 pages.
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        fprintf(stderr, "sysconf(_SC_PAGESIZE) failed (returned %ld)\n", page_size);
        close(fd);
        return 1;
    }
    size_t pages = 1 + (1 << 3);  // 1 + 8 pages
    size_t mmap_len = (size_t)page_size * pages;

    struct perf_event_mmap_page *meta = mmap(NULL, mmap_len,
                                             PROT_READ | PROT_WRITE,
                                             MAP_SHARED, fd, 0);
    if (meta == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    printf("page_size   = %ld bytes\n", page_size);
    printf("mmap_len    = %zu bytes\n", mmap_len);
    printf("data_offset = %llu bytes\n",
           (unsigned long long)meta->data_offset);
    printf("data_size   = %llu bytes\n",
           (unsigned long long)meta->data_size);

    munmap(meta, mmap_len);
    close(fd);
    return 0;
}

