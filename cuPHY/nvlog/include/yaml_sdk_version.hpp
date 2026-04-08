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

#ifndef AERIAL_SDK_VERSION_HPP
#define AERIAL_SDK_VERSION_HPP

#include "yaml.hpp"
#include "yaml-cpp/yaml.h"
#include <type_traits>
#include <string_view>

// Check for concepts support
#ifndef AERIAL_HAS_CONCEPTS
#  if defined(__cpp_concepts) && __cpp_concepts >= 201907L
#    define AERIAL_HAS_CONCEPTS 1
#  else
#    define AERIAL_HAS_CONCEPTS 0
#  endif
#endif

namespace aerial {

/// @brief YAML version attribute aerial_sdk_version="... some ver..."
inline constexpr char YAML_PARAM_AERIAL_SDK_VERSION[] = "aerial_sdk_version";

// Define YamlNodeType constraint - use concept if available, otherwise use type trait
#if AERIAL_HAS_CONCEPTS
template<typename T>
concept YamlNodeType = 
    std::is_same_v<T, YAML::Node> || 
    std::is_same_v<T, yaml::node>;
#else
template<typename T>
struct is_yaml_node : std::false_type {};

template<>
struct is_yaml_node<YAML::Node> : std::true_type {};

template<>
struct is_yaml_node<yaml::node> : std::true_type {};
#endif

/**
 * Given the root YAML node, extract the sdk version and check validity of the version string
 * @param root node root to check the version value of the attribute
 * @param filename Filename of the file - for error messages purposes
 * @throw std::invalid_argument in case of value mismatch
 */
#if AERIAL_HAS_CONCEPTS
template<YamlNodeType NodeType>
#else
template<typename NodeType>
#endif
void check_yaml_version(const NodeType& root, std::string_view filename);

} // namespace aerial

// Include the template implementation
#include "yaml_sdk_version.tpp"

#endif //AERIAL_SDK_VERSION_HPP
