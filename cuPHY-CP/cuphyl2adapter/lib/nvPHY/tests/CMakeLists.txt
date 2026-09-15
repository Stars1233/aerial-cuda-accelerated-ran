# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# ---------------------------------------------------------------------------
# Pure-std, header-only unit test, kept ABOVE the ENABLE_FAPI_STORE_REPLAY
# guard so it builds and runs in every CI preset. prach_offload_handover.hpp has
# no cuphy/CUDA/FAPI-storage dependency, so it does not need the store-replay
# object libraries defined below the guard.
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)

add_executable(test_prach_offload_handover
    test_prach_offload_handover.cpp)

target_include_directories(test_prach_offload_handover PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../cuphydriver/include)

target_link_libraries(test_prach_offload_handover PRIVATE
    sdk::options
    sdk::warnings
    Threads::Threads
    GTest::gtest_main)

# prach_offload_reconfig.hpp (toReconfigOutcome / PrachReconfigOutcome) pulls in
# <tl/expected.hpp>, which the container provides as a system header. Link the tl::expected
# target for its include path only when it exists -- some presets (e.g. the cuphy-cp test
# lane) do not run the framework Dependencies.cmake that defines it, and cuphydriver itself
# builds prach_offload_reconfig.cpp against the system header without this target.
if(TARGET tl::expected)
    target_link_system_libraries(test_prach_offload_handover PRIVATE tl::expected)
endif()

add_test(
    NAME prach_offload_handover.unit_tests
    COMMAND test_prach_offload_handover --gtest_output=xml:test_prach_offload_handover.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(prach_offload_handover.unit_tests PROPERTIES LABELS "parallel;cuphy-cp")

# active_cell_mask.hpp is pure std (<atomic>/<bit>), no cuphy/CUDA/FAPI-storage
# dependency, so it builds and runs in every CI preset like the handover test.
add_executable(test_active_cell_mask
    test_active_cell_mask.cpp)

target_include_directories(test_active_cell_mask PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(test_active_cell_mask PRIVATE
    sdk::options
    sdk::warnings
    GTest::gtest_main)

add_test(
    NAME active_cell_mask.unit_tests
    COMMAND test_active_cell_mask --gtest_output=xml:test_active_cell_mask.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(active_cell_mask.unit_tests PROPERTIES LABELS "parallel;cuphy-cp")

# TX_DATA ownership static guard — builds in every preset (including no-replay).
add_executable(test_tx_data_ownership_noreplay
    test_tx_data_ownership_noreplay.cpp)
target_include_directories(test_tx_data_ownership_noreplay PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(test_tx_data_ownership_noreplay PRIVATE
    sdk::options
    sdk::warnings)
target_link_system_libraries(test_tx_data_ownership_noreplay PRIVATE GTest::gtest_main)
add_test(
    NAME tx_data.ownership_noreplay_guard
    COMMAND test_tx_data_ownership_noreplay
        --gtest_filter=TxDataOwnership.ArmSitesMatchBuildFlavor
        --gtest_output=xml:test_tx_data_ownership_noreplay.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(tx_data.ownership_noreplay_guard PROPERTIES LABELS "parallel;cuphy-cp;sanitizer")

if(NOT ENABLE_FAPI_STORE_REPLAY)
    return()
endif()

find_package(gsl-lite REQUIRED)

# Build storage sources once for both FAPI versions.
add_library(fapi_storage_objects OBJECT
    ../nv_fapi_message_storage.cpp
    ../nv_phy_mac_transport.cpp
    ../nv_slot_telemetry.cpp)

target_compile_definitions(fapi_storage_objects PUBLIC ENABLE_FAPI_STORE_REPLAY)

target_include_directories(fapi_storage_objects PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..)

# nvipc and nvlog_static are internal but not yet warning-clean under
# sdk::warnings. target_link_system_libraries propagates their include
# directories as SYSTEM (-isystem) to consumers, suppressing warnings from
# those headers while sdk::warnings still enforces strict checks on our code.
# scf::scf_5g_fapi_h is already declared SYSTEM in its own CMakeLists.
target_link_system_libraries(fapi_storage_objects PUBLIC nvipc nvlog_static gsl-lite::gsl-lite)
target_link_libraries(fapi_storage_objects PUBLIC scf::scf_5g_fapi_h)

# sdk::options: C++20 + sanitizer propagation to our implementation sources.
# sdk::warnings: full warning set (-Wall -Wextra -Wshadow ... -Werror).
# PRIVATE: neither propagates to test/bench consumers.
target_link_libraries(fapi_storage_objects PRIVATE sdk::options sdk::warnings)

# NVLOGE_FMT and similar macros emit vcast instrumentation pragmas that GCC
# does not recognise. Suppress until nvlog migrates away from VectorCAST.
target_compile_options(fapi_storage_objects PRIVATE -Wno-unknown-pragmas)

add_library(non_slot_dispatch_test_objects OBJECT
    ../nv_phy_non_slot_dispatch.cpp)

target_compile_definitions(non_slot_dispatch_test_objects PRIVATE
    NVPHY_NON_SLOT_DISPATCH_STANDALONE_TEST)

target_include_directories(non_slot_dispatch_test_objects PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_system_libraries(non_slot_dispatch_test_objects PUBLIC nvipc nvlog_static gsl-lite::gsl-lite)
target_link_libraries(non_slot_dispatch_test_objects PUBLIC scf::scf_5g_fapi_h)
target_link_libraries(non_slot_dispatch_test_objects PRIVATE sdk::options sdk::warnings)
target_compile_options(non_slot_dispatch_test_objects PRIVATE -Wno-unknown-pragmas)

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

add_executable(test_fapi_message_storage
    test_fapi_message_storage.cpp)

# GTest via target_link_system_libraries: SYSTEM includes (same as test_bfw_aggr).
target_link_libraries(test_fapi_message_storage PRIVATE
    fapi_storage_objects
    sdk::options
    sdk::warnings)
target_link_system_libraries(test_fapi_message_storage PRIVATE GTest::gtest_main)

# nv_fapi_pdu_utils.hpp expands NVLOGE_FMT which contains vcast_dont_instrument
# pragmas that GCC does not recognise. Suppress until nvlog migrates away from
# VectorCAST. Same pattern used on test_csirs_slot_helpers and on the
# fapi_storage_objects OBJECT library above.
target_compile_options(test_fapi_message_storage PRIVATE -Wno-unknown-pragmas)

add_test(
    NAME fapi_storage.unit_tests
    COMMAND test_fapi_message_storage --gtest_output=xml:test_fapi_message_storage.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(fapi_storage.unit_tests PROPERTIES LABELS "parallel;cuphy-cp")

# nv_tx_data_h2d_helpers.hpp needs nv_ipc_utils.h (sfn_slot_t) for
# endianness-agnostic slot extraction.  Pull the nvipc include directory
# as SYSTEM to keep the test free of nvphy's heavy transitive deps.
add_executable(test_tx_data_h2d_helpers test_tx_data_h2d_helpers.cpp)
target_include_directories(test_tx_data_h2d_helpers PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_system_libraries(test_tx_data_h2d_helpers PRIVATE nvipc)
target_link_libraries(test_tx_data_h2d_helpers PRIVATE GTest::gtest_main sdk::warnings)
add_test(
    NAME fapi_storage.tx_data_h2d_helpers
    COMMAND test_tx_data_h2d_helpers --gtest_output=xml:test_tx_data_h2d_helpers.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(fapi_storage.tx_data_h2d_helpers PROPERTIES LABELS "parallel;cuphy-cp")

# apply_tb_start_offsets_from_tx_data algorithm — tests the TLV-driven
# tbStartOffset override logic that mirrors legacy scf_5g_fapi_phy.cpp
# process_dl_tx_request().  Links fapi_storage_objects for real
# FapiSlotMessageStorage and FAPI type definitions.
add_executable(test_tb_start_offsets test_tb_start_offsets.cpp)

target_link_libraries(test_tb_start_offsets PRIVATE
    fapi_storage_objects
    GTest::gtest_main
    sdk::options
    sdk::warnings)

add_test(
    NAME fapi_storage.tb_start_offsets
    COMMAND test_tb_start_offsets --gtest_output=xml:test_tb_start_offsets.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(fapi_storage.tb_start_offsets PROPERTIES LABELS "parallel;cuphy-cp")

# ---------------------------------------------------------------------------
# nvphy_fapi_tasks — unit tests
#
# nvphy_fapi_tasks is defined in the parent CMakeLists.txt and compiled
# once there.  The test executable links to that shared target.
# ---------------------------------------------------------------------------

# MAX_CELLS_PER_SLOT fallback: cuphy.h is absent in test builds to keep the
# dependency chain thin.  Mirror the same 20-vs-40 cell mapping that cuphy.h
# derives from ENABLE_64C (20 cells when ENABLE_20C, 40 otherwise).
if(ENABLE_20C)
    set(_TEST_MAX_CELLS 20)
else()
    set(_TEST_MAX_CELLS 40)
endif()

# TX_DATA split-phase H2D unit tests (Phase 1 + Phase 2 + Step 6 Option A):
#   test_tx_data_ring_buffer.cpp — TxDataRingBuffer (stage/arm/reset/is_valid/deferred)
#   test_pdsch_tb_merge.cpp     — merge_and_patch<> / patch_tb_start_offsets<>
# Both TUs share identical deps; a single executable avoids compiling
# fapi_storage_objects twice and simplifies CTest naming.
add_executable(test_tx_data
    test_tx_data_ring_buffer.cpp
    test_pdsch_tb_merge.cpp)
target_include_directories(test_tx_data PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(test_tx_data PRIVATE
    fapi_storage_objects
    sdk::options
    sdk::warnings)
target_link_system_libraries(test_tx_data PRIVATE GTest::gtest_main)
target_compile_definitions(test_tx_data PRIVATE
    MAX_CELLS_PER_SLOT=${_TEST_MAX_CELLS}
    SLOT_STORAGE_DEPTH=3)
add_test(
    NAME tx_data.unit_tests
    COMMAND test_tx_data --gtest_output=xml:test_tx_data.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(tx_data.unit_tests PROPERTIES LABELS "parallel;cuphy-cp")

add_test(
    NAME tx_data.deferred_release_sanitizer
    COMMAND test_tx_data
        --gtest_filter=TxDataRingBuffer_DeferredRelease.*:TxDataRingBuffer_LeakRegression.*
        --gtest_output=xml:test_tx_data_deferred_release.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(tx_data.deferred_release_sanitizer PROPERTIES LABELS "parallel;cuphy-cp;sanitizer")

add_executable(test_tx_data_ownership
    test_tx_data_ownership.cpp)
target_include_directories(test_tx_data_ownership PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(test_tx_data_ownership PRIVATE
    fapi_storage_objects
    sdk::options
    sdk::warnings)
target_link_system_libraries(test_tx_data_ownership PRIVATE GTest::gtest_main Threads::Threads)
target_compile_definitions(test_tx_data_ownership PRIVATE
    MAX_CELLS_PER_SLOT=${_TEST_MAX_CELLS}
    SLOT_STORAGE_DEPTH=3)
add_test(
    NAME tx_data.ownership_sanitizer
    COMMAND test_tx_data_ownership
        --gtest_filter=TxDataOwnership.*:DeferredTxDataRelease.*
        --gtest_output=xml:test_tx_data_ownership.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(tx_data.ownership_sanitizer PROPERTIES LABELS "parallel;cuphy-cp;sanitizer")

add_executable(test_nvphy_fapi_tasks
    test_nvphy_fapi_tasks.cpp)

target_compile_definitions(test_nvphy_fapi_tasks PRIVATE MAX_CELLS_PER_SLOT=${_TEST_MAX_CELLS})

target_link_libraries(test_nvphy_fapi_tasks PRIVATE
    nvphy_fapi_tasks
    cuphydriver
    fapi_storage_objects
    GTest::gtest_main
    Threads::Threads
    sdk::options
    sdk::warnings)

add_test(
    NAME nvphy_fapi_tasks.unit_tests
    COMMAND test_nvphy_fapi_tasks --gtest_output=xml:test_nvphy_fapi_tasks.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(nvphy_fapi_tasks.unit_tests PROPERTIES LABELS "parallel;cuphy-cp")

add_test(
    NAME nvphy_fapi_tasks.deferred_release_sanitizer
    COMMAND test_nvphy_fapi_tasks
        --gtest_filter=DeferredTxDataRelease.*
        --gtest_output=xml:test_nvphy_fapi_tasks_deferred_release.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(nvphy_fapi_tasks.deferred_release_sanitizer PROPERTIES LABELS "parallel;cuphy-cp;sanitizer")

# ---------------------------------------------------------------------------
# BFW aggregation — unit tests (DL + UL, parameterized)
#
# Bridges storage and dispatch: verifies DL_BFW / UL_BFW messages are
# stored/retrieved correctly and that the task dispatch routes with the
# correct context. DL and UL paths are exercised by the same TEST_P bodies
# parameterized via DirectionTraits, so each test case runs once per
# direction (e.g. .DL and .UL suffixes in the gtest output).
# ---------------------------------------------------------------------------

# Links fapi_storage_objects (10.04-only); gate on SCF_FAPI_10_04.
if(SCF_FAPI_10_04)
add_executable(test_bfw_aggr
    test_bfw_aggr.cpp)

# sdk::warnings (-Werror) intentionally omitted: this TU transitively includes
# legacy headers (scf_5g_fapi.h, nvlog, nvIPC) that are not yet warning-clean.


# GTest is linked with target_link_system_libraries so its headers are treated
# as SYSTEM (-isystem), matching the pattern used for nvipc/nvlog_static on
# fapi_storage_objects (see comment above add_library(fapi_storage_objects)).

target_link_libraries(test_bfw_aggr PRIVATE
    fapi_storage_objects
    slot_command
    sdk::options)
target_link_system_libraries(test_bfw_aggr PRIVATE GTest::gtest_main)

add_test(
    NAME bfw_aggr.unit_tests
    COMMAND test_bfw_aggr --gtest_output=xml:test_bfw_aggr.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(bfw_aggr.unit_tests PROPERTIES LABELS "parallel;cuphy-cp")
endif()


add_executable(test_nvphy_non_slot_dispatch
    test_nvphy_non_slot_dispatch.cpp)

target_link_libraries(test_nvphy_non_slot_dispatch PRIVATE
    non_slot_dispatch_test_objects
    sdk::options
    sdk::warnings)
target_link_system_libraries(test_nvphy_non_slot_dispatch PRIVATE GTest::gtest_main)

add_test(
    NAME nvphy_non_slot_dispatch.unit_tests
    COMMAND test_nvphy_non_slot_dispatch --gtest_output=xml:test_nvphy_non_slot_dispatch.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(nvphy_non_slot_dispatch.unit_tests PROPERTIES LABELS "parallel;cuphy-cp")

# ---------------------------------------------------------------------------
# CSI-RS slot helpers — unit tests
# ---------------------------------------------------------------------------

# Depends on scf_5g_csirs_helpers, which is only built under ENABLE_FAPI_STORE_REPLAY.
if(ENABLE_FAPI_STORE_REPLAY)
    add_executable(test_csirs_slot_helpers
        test_csirs_slot_helpers.cpp)

    target_link_libraries(test_csirs_slot_helpers PRIVATE
        scf_5g_csirs_helpers
        fapi_storage_objects
        sdk::options
        sdk::warnings)
    target_link_system_libraries(test_csirs_slot_helpers PRIVATE GTest::gtest_main)

    target_compile_options(test_csirs_slot_helpers PRIVATE -Wno-unknown-pragmas)

    add_test(
        NAME csirs_slot_helpers.unit_tests
        COMMAND test_csirs_slot_helpers --gtest_output=xml:test_csirs_slot_helpers.xml
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(csirs_slot_helpers.unit_tests PROPERTIES LABELS "parallel;cuphy-cp")
endif()
