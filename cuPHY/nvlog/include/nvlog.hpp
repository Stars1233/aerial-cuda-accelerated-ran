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

#ifndef _NVLOG_HPP_
#define _NVLOG_HPP_

#include <stdint.h>

#include <sstream>
#include <iostream>

#include "nvlog.h"
#include "nv_utils.h"

#include "yaml-cpp/yaml.h"

#include "nvlog_fmt.hpp"

/**
 * Initialize fmtlog from YAML config: log path, rotation, compression, and start polling thread.
 * @param yaml_file Path to nvlog config YAML; NULL for defaults.
 * @param name Log file base name (e.g. "phy.log").
 * @param exit_hdlr_cb Optional callback for fatal exit (e.g. L1 cleanup).
 * @return Polling thread ID on success, -1 if already initiated or on error.
 */
pthread_t nvlog_fmtlog_init(const char* yaml_file, const char* name, void (*exit_hdlr_cb)());

/** Initialize fmtlog for the current thread (preallocate). */
void nvlog_fmtlog_thread_init();

/**
 * Initialize fmtlog for the current thread and set its name for log output.
 * @param name Thread name.
 */
void nvlog_fmtlog_thread_init(const char* name);

/**
 * Close the fmtlog for nvlog: compress the final log file, stop the compression thread, and close fmtlog.
 * @param[in] bg_thread_id Background polling thread ID (unused; kept for API compatibility).
 */
void nvlog_fmtlog_close(pthread_t bg_thread_id = 0);

/**
 * Resolve CUBB root path (from CUBB_HOME or executable directory).
 * @param path Output buffer for the root path (with trailing slash).
 * @param cubb_root_path_relative_num Parent levels up from executable dir if CUBB_HOME not set.
 * @return Length of path string, or -1 on failure.
 */
int get_root_path(char* path, int cubb_root_path_relative_num);

/**
 * Build full path: root + relative_path + file_name.
 * @param dest_buf Output buffer.
 * @param relative_path Optional subdirectory; can be NULL.
 * @param file_name File name; can be NULL.
 * @param cubb_root_dir_relative_num Passed to get_root_path().
 * @return Total length of the constructed path.
 */
int get_full_path_file(char* dest_buf, const char* relative_path, const char* file_name, int cubb_root_dir_relative_num);

#endif /* _NVLOG_HPP_ */
