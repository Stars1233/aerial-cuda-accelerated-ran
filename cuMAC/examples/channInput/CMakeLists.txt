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

# set channInput as a library to be shared with other executables

set(CHANNINPUT_SOURCES channInput_src/channInput.cu)
add_library(channInput STATIC ${CHANNINPUT_SOURCES})
set_property(TARGET channInput PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET channInput PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# target_include_directories(channInput PRIVATE ${CMAKE_SOURCE_DIR}/asim/src/backend/bindings) # for asim.hpp
target_include_directories(channInput PUBLIC ${HDF5_CXX_INCLUDE_DIRS} ${CHAN_MODELS_SRC_INCLUDE_DIR} ${cumac_SOURCE_DIR}/src ${cumac_SOURCE_DIR}/examples/tools)
target_link_libraries(channInput PUBLIC ${HDF5_CXX_LIBRARIES} CUDA::cudart CUDA::cuda_driver chanModels nvipc)

add_executable(channInputTest)
target_sources(channInputTest PRIVATE channInput_example/channInputTest.cpp)
target_link_libraries(channInputTest PRIVATE channInput CUDA::cudart)
