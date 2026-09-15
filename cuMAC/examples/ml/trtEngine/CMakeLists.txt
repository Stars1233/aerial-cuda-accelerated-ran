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

add_executable(trtEngine)
target_sources(trtEngine PRIVATE main.cpp ../trtEngine.cpp)
target_link_libraries(trtEngine PRIVATE  CUDA::cudart CUDA::cuda_driver ${HDF5_CXX_LIBRARIES} ${NVINFER} ${NVONNXPARSER} nvlog nvipc)
target_include_directories(trtEngine PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${cumac_SOURCE_DIR}/src
    ${cumac_SOURCE_DIR}/examples/tools
    ${HDF5_CXX_INCLUDE_DIRS} 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Default location of infConfig.yaml / pt_to_onnx.py. With this fallback the
# benchmark runs with zero arguments from anywhere: when no infConfig.yaml is
# found in the current working directory or next to the binary, this source-tree
# config is used, and its relative model/script paths resolve against the source
# tree (where the trained model and exporter live).
target_compile_definitions(trtEngine PRIVATE CUMAC_ML_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")
