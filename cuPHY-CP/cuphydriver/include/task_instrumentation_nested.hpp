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

#ifndef NESTED_TASK_INSTRUMENTATION_H
#define NESTED_TASK_INSTRUMENTATION_H

#include "time.hpp"

/**
 * @brief Global flag to enable/disable subtask instrumentation at compile time.
 */
static constexpr bool ENABLE_SUBTASK_INSTRUMENTATION = true; 

/**
 * @brief Maximum number of subtasks that can be tracked per task.
 */
static constexpr int MAX_NUM_SUBTASKS_NESTED = 32;

/**
 * @brief Maximum length of subtask name strings (including null terminator).
 */
static constexpr int MAX_SUBTASK_CHARS = 32;

/**
 * @brief Subtask instrumentation data structure.
 * 
 * Stores timing information for nested subtasks within a parent task.
 */
struct ti_subtask_info {
    char tname[MAX_NUM_SUBTASKS_NESTED][MAX_SUBTASK_CHARS]{}; ///< Array of subtask names
    int count{};                                         ///< Current number of recorded subtasks
    t_ns time[MAX_NUM_SUBTASKS_NESTED]{};                      ///< Array of subtask timestamps
}; 

/**
 * @brief Macro to add a subtask timing entry.
 * 
 * Records the current time and name for a subtask if instrumentation is enabled
 * and the maximum number of subtasks has not been reached.
 * 
 * @param task_info The ti_subtask_info structure to update
 * @param name The name of the subtask (string literal or char*)
 */
#define TI_SUBTASK_INFO_ADD(task_info,name) \
if ((task_info.count < MAX_NUM_SUBTASKS_NESTED) && (ENABLE_SUBTASK_INSTRUMENTATION == true))  { \
    strncpy(task_info.tname[task_info.count],name,MAX_SUBTASK_CHARS-1); \
    task_info.tname[task_info.count][MAX_SUBTASK_CHARS-1]='\0'; \
    task_info.time[task_info.count]=Time::nowNs(); \
    ++task_info.count; \
}

#endif // NESTED_TASK_INSTRUMENTATION_H
