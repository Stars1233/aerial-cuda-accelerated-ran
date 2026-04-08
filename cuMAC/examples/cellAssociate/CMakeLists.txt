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

add_executable(cellAssociateTest)
target_sources(cellAssociateTest PRIVATE cellAssociateTest.cpp ../network.cu)
target_include_directories(cellAssociateTest PRIVATE ${HDF5_CXX_INCLUDE_DIRS} ${CHAN_MODELS_SRC_INCLUDE_DIR})
target_link_libraries(cellAssociateTest PRIVATE cumac ${HDF5_CXX_LIBRARIES} channInput chanModels)

enable_testing()
add_test(NAME cellAssociateTest COMMAND cellAssociateTest)
set_property(TEST cellAssociateTest PROPERTY -R "Success: CPU and GPU scheduling solutions match")