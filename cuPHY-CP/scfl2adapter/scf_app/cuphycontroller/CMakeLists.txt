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

set (SOURCES cuphycontroller.cpp signal_handler.cpp
	${PROJECT_SOURCE_DIR}/dummy.cu)
add_executable(l2_adapter_cuphycontroller_scf ${SOURCES})

target_link_libraries(l2_adapter_cuphycontroller_scf PRIVATE
    pthread
    nvipc
    # scf_5g_fapi and nvphy have circular static-archive refs; rescan resolves both.
    "$<LINK_GROUP:RESCAN,scf_5g_fapi,nvphy>"
    yaml)

if (AERIAL_METRICS)
    target_link_libraries(l2_adapter_cuphycontroller_scf PRIVATE aerial_metrics)
endif()

set (SOURCES tick_unit_test.cpp ${PROJECT_SOURCE_DIR}/dummy.cu)
add_executable(tick_unit_test ${SOURCES})
target_link_libraries(tick_unit_test
    nvipc
    # scf_5g_fapi and nvphy have circular static-archive refs; rescan resolves both.
    "$<LINK_GROUP:RESCAN,scf_5g_fapi,nvphy>"
    yaml)
