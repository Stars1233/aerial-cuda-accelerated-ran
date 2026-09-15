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

set (SOURCES test_ipc.c)
add_executable(test_ipc ${SOURCES})
# Force CXX host linker; CUDA device-link is not needed for this pure-C executable.
set_target_properties(test_ipc PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_ipc nvipc -pthread)
# sdk::options provides ASan/sanitizer flags when building within the cuBB tree.
# Guard with TARGET check so standalone nvipc builds (nvipc_src container) are unaffected.
if(TARGET sdk::options)
    target_link_libraries(test_ipc sdk::options)
endif()

if(NVIPC_CUDA_ENABLE)
    target_link_libraries(test_ipc cudaipctest)
endif()