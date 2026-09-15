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

find_package(GTest REQUIRED)
find_package(CUDAToolkit REQUIRED COMPONENTS cudart)

add_subdirectory(cellAssociation)
add_subdirectory(multiCellUeSelection)
add_subdirectory(multiCellScheduler)
add_subdirectory(muMimoUserPairing)
add_subdirectory(roundRobinScheduler)
add_subdirectory(singleCellScheduler)
add_subdirectory(multiCellMuUeSort)
add_subdirectory(multiCellSrsScheduler)
