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

# ----------------------------------------------------------------------------
# ctest registration for cplane_test_bench.
#
# Included from cuPHY-CP/tests/cplane_test_bench/CMakeLists.txt via include().
# Delegates execution to the existing run_all_pattern_tests.sh, whose exit
# code is already an aggregate pass/fail signal across the patterns it ran.
#
# Three tests are registered, one per phase. Each carries exactly ONE label
# so that `ctest --print-labels` returns one label per test — preventing
# CI label-discovery loops from running the suite multiple times under
# overlapping labels.
#
#   ctest                              # no filter: runs all = full suite
#   ctest -L cplane_tb_4t4r    --output-on-failure   # 4T4R phase only
#   ctest -L cplane_tb_mmimo   --output-on-failure   # mMIMO phase only
#   ctest -L cplane_tb_nrsim   --output-on-failure   # nrSim phase only
#   ctest -R cplane_tb_       --output-on-failure   # by name regex (full suite)
#
# Both tests share RESOURCE_LOCK cplane_tb_global_config so that ctest -j N
# still serializes them — the per-pattern setup helpers mutate yaml files
# under cuPHY-CP/.../config/ and cannot run concurrently. ctest's default
# alphabetical ordering runs cplane_tb_4t4r before cplane_tb_nrsim, matching
# the phase order of run_all_pattern_tests.sh's own default flow.
# ----------------------------------------------------------------------------

enable_testing()

# Compute the build directory relative to cuBB_SDK so run_all_pattern_tests.sh
# locates the binary at $cuBB_SDK/$BUILD_DIR/cuPHY-CP/tests/cplane_test_bench/
# regardless of which preset/--build_dir the user actually built into.
file(RELATIVE_PATH _cptb_build_rel "${CUBB_HOME}" "${CMAKE_BINARY_DIR}")

set(_cptb_run_all "${CMAKE_CURRENT_SOURCE_DIR}/run_all_pattern_tests.sh")
set(_cptb_env     "cuBB_SDK=${CUBB_HOME};BUILD_DIR=${_cptb_build_rel}")

# 4T4R phase: patterns from test_patterns_4t4r.csv. `--cleanup` is required
# because run_all_pattern_tests.sh only auto-enables FORCE_CLEANUP when no
# phase flag is passed; a bare `--4t4r` would otherwise inherit yaml
# mutations from a previous run.
add_test(
    NAME    cplane_tb_4t4r
    COMMAND bash "${_cptb_run_all}" --4t4r --cleanup
    WORKING_DIRECTORY "${CUBB_HOME}"
)
set_tests_properties(cplane_tb_4t4r PROPERTIES
    ENVIRONMENT   "${_cptb_env}"
    LABELS        "requires-tvs;cuphy-cp;cplane_tb_4t4r"
    RESOURCE_LOCK "cplane_tb_global_config"
    TIMEOUT       7200
)

# mMIMO phase: patterns from test_patterns_mmimo.csv (BFW C-plane). Same
# --cleanup rationale as cplane_tb_4t4r above.
add_test(
    NAME    cplane_tb_mmimo
    COMMAND bash "${_cptb_run_all}" --mmimo --cleanup
    WORKING_DIRECTORY "${CUBB_HOME}"
)
set_tests_properties(cplane_tb_mmimo PROPERTIES
    ENVIRONMENT   "${_cptb_env}"
    LABELS        "requires-tvs;cuphy-cp;cplane_tb_mmimo"
    RESOURCE_LOCK "cplane_tb_global_config"
    TIMEOUT       7200
)

# nrSim phase: patterns from test_patterns_nrsim.csv. Same --cleanup
# rationale as cplane_tb_4t4r above.
add_test(
    NAME    cplane_tb_nrsim
    COMMAND bash "${_cptb_run_all}" --nrsim --cleanup
    WORKING_DIRECTORY "${CUBB_HOME}"
)
set_tests_properties(cplane_tb_nrsim PROPERTIES
    ENVIRONMENT   "${_cptb_env}"
    LABELS        "requires-tvs;cuphy-cp;cplane_tb_nrsim"
    RESOURCE_LOCK "cplane_tb_global_config"
    TIMEOUT       7200
)
