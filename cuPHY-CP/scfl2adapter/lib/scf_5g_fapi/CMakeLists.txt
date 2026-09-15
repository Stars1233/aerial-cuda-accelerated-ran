# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lib: scf_5g_fapi
set(SOURCES scf_5g_fapi.cpp
            scf_5g_fapi_mac.cpp
            scf_5g_fapi_phy.cpp
            scf_5g_fapi_config_decode.cpp
            scf_5g_slot_commands.cpp
            scf_5g_fapi_metrics.cpp
            scf_5g_fapi_ul_validate.cpp
            scf_5g_fapi_dl_validate.cpp
            scf_5g_slot_commands_common.cpp
            scf_5g_slot_commands_ssb_pbch.cpp
            scf_5g_slot_commands_pdcch.cpp
            scf_5g_slot_commands_pdsch_mod_comp.cpp
            scf_5g_slot_commands_pdsch_csirs.cpp
            scf_5g_fapi_slot_config_tlv_parser.cpp
            )

if(ENABLE_FAPI_STORE_REPLAY)
    list(APPEND SOURCES scf_5g_fapi_phy_non_slot_offload.cpp)
    list(REMOVE_ITEM SOURCES scf_5g_slot_commands_ssb_pbch.cpp)
endif()

add_library(scf_5g_fapi STATIC ${SOURCES})
set_property(TARGET scf_5g_fapi PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(scf_5g_fapi PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_compile_definitions(scf_5g_fapi PRIVATE
    $<$<BOOL:${ENABLE_FAPI_STORE_REPLAY}>:ENABLE_FAPI_STORE_REPLAY>)
target_compile_options(scf_5g_fapi PRIVATE ${AERIAL_ARCH_TUNE_FLAGS})
target_link_libraries(scf_5g_fapi PRIVATE cuphy_hdf5 cuphy_cpp_header_only nvphy nvipc slot_command aerial_common)

if(ENABLE_FAPI_STORE_REPLAY)
    # CSI-RS slot command helpers — standalone static library so nvphy can link it
    # without pulling in the full scf_5g_fapi target.
    add_library(scf_5g_csirs_helpers STATIC scf_5g_csirs_slot_command_helpers.cpp)
    set_property(TARGET scf_5g_csirs_helpers PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_include_directories(scf_5g_csirs_helpers PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
    # scf_5g_slot_commands_common.hpp (transitive include) pulls in nv_phy_utils.hpp,
    # nv_phy_config_option.hpp and nv_phy_driver_proxy.hpp from nvphy, plus cuphydriver
    # headers. Propagate those include dirs so downstream consumers compile cleanly.
    target_link_libraries(scf_5g_csirs_helpers PUBLIC
        cuphy_cpp_header_only slot_command aerial_common
        scf::scf_5g_fapi_h nvphy_generic cuphydriver)
    target_link_system_libraries(scf_5g_csirs_helpers PUBLIC nvlog_static)
    # sdk::warnings intentionally omitted: scf_5g_fapi.h (same directory) uses
    # GNU zero-length arrays that trigger -Werror=pedantic, and the non-SYSTEM
    # include path from this target overrides scf::scf_5g_fapi_h's SYSTEM property.
    target_link_libraries(scf_5g_csirs_helpers PRIVATE sdk::options)
    target_compile_options(scf_5g_csirs_helpers PRIVATE ${AERIAL_ARCH_TUNE_FLAGS} -Wno-unknown-pragmas)
    target_link_libraries(scf_5g_fapi PRIVATE scf_5g_csirs_helpers)

    # SSB/PBCH slot command helpers — used by the store-replay channel task path
    # after nvphy instantiates SsbPduParser<PhyModuleView>. Keep this as a
    # standalone archive so linkers can resolve the helper after libnvphy.a
    # introduces the symbol.
    add_library(scf_5g_ssb_helpers STATIC scf_5g_slot_commands_ssb_pbch.cpp)
    set_property(TARGET scf_5g_ssb_helpers PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_include_directories(scf_5g_ssb_helpers PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
    target_link_libraries(scf_5g_ssb_helpers PUBLIC
        cuphy_cpp_header_only slot_command aerial_common
        scf::scf_5g_fapi_h nvphy_generic cuphydriver)
    target_link_system_libraries(scf_5g_ssb_helpers PUBLIC nvlog_static)
    target_link_libraries(scf_5g_ssb_helpers PRIVATE sdk::options)
    target_compile_options(scf_5g_ssb_helpers PRIVATE ${AERIAL_ARCH_TUNE_FLAGS} -Wno-unknown-pragmas)
    target_link_libraries(scf_5g_fapi PRIVATE scf_5g_ssb_helpers)
endif()

# Header-only INTERFACE library for SCF 5G FAPI headers
add_library(scf_5g_fapi_h INTERFACE)
add_library(scf::scf_5g_fapi_h ALIAS scf_5g_fapi_h)

target_include_directories(scf_5g_fapi_h SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(scf_5g_fapi_h INTERFACE cxx_std_20)
target_link_libraries(scf_5g_fapi_h INTERFACE aerial_common)

if(SCF_FAPI_10_04)
    add_library(scf_5g_fapi_uci_config_parser STATIC scf_5g_fapi_uci_config_parser.cpp)
    set_property(TARGET scf_5g_fapi_uci_config_parser PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_include_directories(scf_5g_fapi_uci_config_parser PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
    target_compile_options(scf_5g_fapi_uci_config_parser PRIVATE ${AERIAL_ARCH_TUNE_FLAGS})
    target_link_libraries(scf_5g_fapi_uci_config_parser PUBLIC
        scf::scf_5g_fapi_h
        cuphy::cuphy_h)
    target_link_libraries(scf_5g_fapi PRIVATE scf_5g_fapi_uci_config_parser)
endif()

if(ENABLE_CONFORMANCE_TM_PDSCH_PDCCH)
    target_compile_definitions(scf_5g_fapi PUBLIC ENABLE_CONFORMANCE_TM_PDSCH_PDCCH)
    target_compile_definitions(scf_5g_fapi_h INTERFACE ENABLE_CONFORMANCE_TM_PDSCH_PDCCH)
endif()

#---- Aerial metrics
if (TARGET aerial_metrics)
    target_link_libraries(scf_5g_fapi PRIVATE aerial_metrics)
    target_compile_definitions(scf_5g_fapi PRIVATE AERIAL_METRICS)
endif()

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
