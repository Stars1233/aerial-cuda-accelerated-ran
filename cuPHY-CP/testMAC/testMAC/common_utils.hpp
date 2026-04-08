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

#ifndef _COMMON_UTILS_HPP_
#define _COMMON_UTILS_HPP_

#include <vector>
#include "nvlog_fmt.hpp"

#define MAX_PATH_LEN 1024 //!< Maximum path length for file operations

#ifndef CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM
  /**
   * Number of parent directories to traverse to reach cuBB_SDK root
   * For example: 4 means "../../../../" from current process directory
   */
  #define CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM 4
#endif

#define CONFIG_TESTMAC_YAML_PATH "cuPHY-CP/testMAC/testMAC/" //!< Relative path to test MAC YAML configs
#define CONFIG_TESTMAC_YAML_NAME "test_mac_config.yaml"      //!< Default test MAC configuration file name
#define CONFIG_LAUNCH_PATTERN_PATH "testVectors/multi-cell/" //!< Relative path to launch pattern files
#define CONFIG_TEST_VECTOR_PATH "testVectors/"               //!< Relative path to test vector files

/**
 * Bit manipulation macros for parsing FAPI message fields
 * 
 * Used to extract/set multi-bit fields from integer values (e.g., SFN, slot, carrier)
 */
#define INTEGER_GET_BITS(var, start, width) (((var) >> (start)) & ((1 << (width)) - 1)) //!< Extract bits from integer
#define INTEGER_SET_BITS(var, start, width, val) ((var & ~(((1 << width) - 1) << start)) | (val << start)) //!< Set bits in integer

/**
 * Get human-readable name for SCF FAPI message ID
 * 
 * @param[in] msg_id FAPI message ID
 * @return String name of the message type
 */
const char* get_scf_fapi_msg_name(uint8_t msg_id);

/**
 * Check if the application is exiting
 *
 * @return True if the application is exiting, false otherwise
 */
static inline bool is_app_exiting()
{
    return pExitHandler.test_exit_in_flight();
}

#endif /* _COMMON_UTILS_HPP_ */
