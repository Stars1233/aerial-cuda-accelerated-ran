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

set(SOURCES ../common/util.cu cuphy_ex_pdsch_tx_multi_cell.cpp dummy.cu ../common/datasets.cpp)
add_executable(cuphy_ex_pdsch_tx_multi_cell ${SOURCES})
target_link_libraries(cuphy_ex_pdsch_tx_multi_cell cuphy_channels cuphy_hdf5 cuphy yaml)

#install(FILES CMakeLists.release.txt DESTINATION examples/pdsch_tx_multi_cell RENAME CMakeLists.txt)
