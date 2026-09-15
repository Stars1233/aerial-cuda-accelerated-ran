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

#pragma once

#include "nvlog_fmt.hpp"

//ti_generic - A generic set of task instrumentation macros
//
//Example usage:
//
//TI_GENERIC_INIT("instrumentation 1",10)
//TI_GENERIC_ADD("subtask 1")
//... code
//TI_GENERIC_ADD("subtask 2")
//... code
//TI_GENERIC_ADD("end task")
//TI_GENERIC_DURATION_NVLOGI(TAG)

#define MAX_SUBTASK_CHARS 32

#define TI_GENERIC_INIT(task_name,max_subtasks) \
std::stringstream ti_os; \
char ti_task_name[MAX_SUBTASK_CHARS]; \
int ti_max_subtasks = max_subtasks; \
strcpy(ti_task_name,task_name); \
std::chrono::nanoseconds ti_times[max_subtasks]; \
char ti_subtask_names[max_subtasks][MAX_SUBTASK_CHARS]; \
int ti_subtask_count = 0;

//Creates a subtask marker
#define TI_GENERIC_ADD(subtask_name) \
if(ti_subtask_count < ti_max_subtasks) { \
    strcpy(ti_subtask_names[ti_subtask_count],subtask_name); \
    ti_times[ti_subtask_count] = std::chrono::system_clock::now().time_since_epoch(); \
    ti_subtask_count += 1; \
}

//Prints percentage contribution between each subtask marker
#define TI_GENERIC_PERCENTAGE_NVLOG(LOG_LEVEL,TAG) \
char ti_subtask_results1[4096]; \
int ti_offset1=0; \
double ti_total1=static_cast<double>((ti_times[ti_subtask_count-1]-ti_times[0]).count())/1e3; \
for(int ii=0; ii<ti_subtask_count-1; ii++) { \
    double ti_percentage = 100.0*(static_cast<double>((ti_times[ii+1]-ti_times[ii]).count())/1e3)/ti_total1; \
    ti_offset1 += sprintf(&ti_subtask_results1[ti_offset1], "%s:%.1f,", ti_subtask_names[ii], ti_percentage); \
} \
ti_offset1 += sprintf(&ti_subtask_results1[ti_offset1], " (total: %.1fus),", ti_total1); \
NVLOG_FMT(LOG_LEVEL,TAG,"{{TI PERCENTAGE}} <{}> {}\n", ti_task_name, ti_subtask_results1);

#define TI_GENERIC_PERCENTAGE_NVLOGV(component_id) TI_GENERIC_PERCENTAGE_NVLOG(fmtlog::VEB,component_id)
#define TI_GENERIC_PERCENTAGE_NVLOGD(component_id) TI_GENERIC_PERCENTAGE_NVLOG(fmtlog::DBG,component_id)
#define TI_GENERIC_PERCENTAGE_NVLOGI(component_id) TI_GENERIC_PERCENTAGE_NVLOG(fmtlog::INF,component_id)
#define TI_GENERIC_PERCENTAGE_NVLOGW(component_id) TI_GENERIC_PERCENTAGE_NVLOG(fmtlog::WRN,component_id)
#define TI_GENERIC_PERCENTAGE_NVLOGC(component_id) TI_GENERIC_PERCENTAGE_NVLOG(fmtlog::CON,component_id)


//Prints duration between each subtask marker
#define TI_GENERIC_DURATION_NVLOG(LOG_LEVEL,TAG) \
char ti_subtask_results2[4096]; \
int ti_offset2=0; \
for(int ii=0; ii<ti_subtask_count-1; ii++) { \
    ti_offset2 += sprintf(&ti_subtask_results2[ti_offset2], "%s:%.3f,", ti_subtask_names[ii], static_cast<double>((ti_times[ii+1]-ti_times[ii]).count())/1e3); \
} \
NVLOG_FMT(LOG_LEVEL,TAG,"{{TI DURATION}} <{}> {}\n", ti_task_name, ti_subtask_results2);

#define TI_GENERIC_DURATION_NVLOGV(component_id) TI_GENERIC_DURATION_NVLOG(fmtlog::VEB,component_id)
#define TI_GENERIC_DURATION_NVLOGD(component_id) TI_GENERIC_DURATION_NVLOG(fmtlog::DBG,component_id)
#define TI_GENERIC_DURATION_NVLOGI(component_id) TI_GENERIC_DURATION_NVLOG(fmtlog::INF,component_id)
#define TI_GENERIC_DURATION_NVLOGW(component_id) TI_GENERIC_DURATION_NVLOG(fmtlog::WRN,component_id)
#define TI_GENERIC_DURATION_NVLOGC(component_id) TI_GENERIC_DURATION_NVLOG(fmtlog::CON,component_id)


//Prints timestamp of each subtask marker
#define TI_GENERIC_TIMESTAMP_NVLOG(LOG_LEVEL,TAG) \
char ti_subtask_results3[4096]; \
int ti_offset3=0; \
for(int ii=0; ii<ti_subtask_count; ii++) { \
    ti_offset3 += sprintf(&ti_subtask_results3[ti_offset3], "%s:%ld,", ti_subtask_names[ii], ti_times[ii].count()); \
} \
NVLOG_FMT(LOG_LEVEL,TAG,"{{TI TIMESTAMPS}} <{}> {}\n", ti_task_name, ti_subtask_results3);

#define TI_GENERIC_TIMESTAMP_NVLOGV(component_id) TI_GENERIC_TIMESTAMP_NVLOG(fmtlog::VEB,component_id)
#define TI_GENERIC_TIMESTAMP_NVLOGD(component_id) TI_GENERIC_TIMESTAMP_NVLOG(fmtlog::DBG,component_id)
#define TI_GENERIC_TIMESTAMP_NVLOGI(component_id) TI_GENERIC_TIMESTAMP_NVLOG(fmtlog::INF,component_id)
#define TI_GENERIC_TIMESTAMP_NVLOGW(component_id) TI_GENERIC_TIMESTAMP_NVLOG(fmtlog::WRN,component_id)
#define TI_GENERIC_TIMESTAMP_NVLOGC(component_id) TI_GENERIC_TIMESTAMP_NVLOG(fmtlog::CON,component_id)


//Prints all forms of TI_GENERIC log messages
#define TI_GENERIC_ALL_NVLOG(LOG_LEVEL,TAG) TI_GENERIC_PERCENTAGE_NVLOG(LOG_LEVEL,TAG); TI_GENERIC_DURATION_NVLOG(LOG_LEVEL,TAG); TI_GENERIC_TIMESTAMP_NVLOG(LOG_LEVEL,TAG);

#define TI_GENERIC_ALL_NVLOGV(component_id) TI_GENERIC_ALL_NVLOG(fmtlog::VEB,component_id)
#define TI_GENERIC_ALL_NVLOGD(component_id) TI_GENERIC_ALL_NVLOG(fmtlog::DBG,component_id)
#define TI_GENERIC_ALL_NVLOGI(component_id) TI_GENERIC_ALL_NVLOG(fmtlog::INF,component_id)
#define TI_GENERIC_ALL_NVLOGW(component_id) TI_GENERIC_ALL_NVLOG(fmtlog::WRN,component_id)
#define TI_GENERIC_ALL_NVLOGC(component_id) TI_GENERIC_ALL_NVLOG(fmtlog::CON,component_id)