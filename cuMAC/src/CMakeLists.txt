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

set(CUMAC_SOURCES
    api.h
    ../lib/cumac_msg/cumac_pfm_sort.h
    cumac.h
    cpuMatAlg/cpuMatAlg.cpp
    cpuMatAlg/cpuMatAlg.h
    4T4R/cellAssociation.cu
    4T4R/cellAssociation.cuh
    4T4R/cellAssociationCpu.cpp
    4T4R/cellAssociationCpu.h
    4T4R/multiCellScheduler.cu
    4T4R/multiCellScheduler.cuh
    4T4R/roundRobinScheduler.cu
    4T4R/roundRobinScheduler.cuh
    4T4R/multiCellUeSelection.cu
    4T4R/multiCellUeSelection.cuh
    4T4R/roundRobinUeSel.cu
    4T4R/roundRobinUeSel.cuh
    4T4R/multiCellSchedulerCpu.cpp
    4T4R/multiCellSchedulerCpu.h
    4T4R/roundRobinSchedulerCpu.h
    4T4R/roundRobinSchedulerCpu.cpp
    4T4R/singleCellScheduler.cu
    4T4R/singleCellScheduler.cuh
    4T4R/singleCellSchedulerCpu.cpp
    4T4R/singleCellSchedulerCpu.h
    4T4R/svdPrecoding.cu
    4T4R/svdPrecoding.cuh
    4T4R/mcsSelectionLUT.cuh
    4T4R/mcsSelectionLUT.cu
    4T4R/mcsSelectionLUTCpu.h
    4T4R/mcsSelectionLUTCpu.cpp
    4T4R/multiCellLayerSel.cuh
    4T4R/multiCellLayerSel.cu
    4T4R/multiCellUeSelectionCpu.h
    4T4R/multiCellUeSelectionCpu.cpp
    4T4R/multiCellLayerSelCpu.h
    4T4R/multiCellLayerSelCpu.cpp
    4T4R/roundRobinUeSelCpu.h
    4T4R/roundRobinUeSelCpu.cpp
    64T64R/multiCellMuUeSort.cuh
    64T64R/multiCellMuUeSort.cu
    64T64R/multiCellMuUeGrp.cuh
    64T64R/multiCellMuUeGrp.cu
    64T64R/multiCellBeamform.cuh
    64T64R/multiCellBeamform.cu
    64T64R/muMimoSchedulerBaseCpu.h
    64T64R/muMimoSchedulerBaseCpu.cpp
    pfmSort/pfmSort.cuh
    pfmSort/pfmSort.cu
    multiCellSrsScheduler/multiCellSrsScheduler.cuh
    multiCellSrsScheduler/multiCellSrsScheduler.cu
    tools/multiCellSinrCal.cu
    tools/multiCellSinrCal.cuh
    muMimoUserPairing/muMimoUserPairing.cuh
    muMimoUserPairing/muMimoUserPairing.cu
    cumacSubcontext.h
    cumacSubcontext.cpp
    ../examples/tools/h5TvLoad.h
    ../examples/tools/h5TvLoad.cpp
    ../examples/tools/h5TvCreate.h
    ../examples/tools/h5TvCreate.cpp
)

if (NOT CMAKE_BUILD_TYPE MATCHES "Release" OR ENABLE_LINE_INFO)
    list(APPEND CUMAC_NVCC_FLAGS -lineinfo)
endif()
#message("lineinfo flag value: ${CUMAC_NVCC_FLAGS}")

find_package(Eigen3 REQUIRED)

add_library(cumac ${CUMAC_SOURCES})
set_property(TARGET cumac PROPERTY POSITION_INDEPENDENT_CODE ON)
if (BUILD_SHARED_LIBS)
    set_property(TARGET cumac PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET cumac PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

target_link_libraries(cumac PUBLIC ${HDF5_CXX_LIBRARIES} CUDA::cudart CUDA::cuda_driver CUDA::cusolver ${NVINFER} ${NVONNXPARSER} Eigen3::Eigen nvlog nvipc)
target_include_directories(cumac PRIVATE ${HDF5_CXX_INCLUDE_DIRS})
target_include_directories(cumac PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} multiCellSrsScheduler ${cumac_SOURCE_DIR}/examples/tools)
target_compile_options(cumac PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       ${CUMAC_PTXAS_OPTIONS}
                       --cudart static
                       ${CUMAC_NVCC_FLAGS}
                       --threads 0
                       --expt-relaxed-constexpr
                       >)

# install(TARGETS cumac DESTINATION lib)
