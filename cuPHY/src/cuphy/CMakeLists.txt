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

# CUDA::nvToolsExt is deprecated in cmake 3.25 and not present in newer CTKs.
# Define a fake target to avoid cmake 3.25 error message in find_package(CUDAToolkit) with newer CTKs.
if (NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt IMPORTED INTERFACE)
endif()
find_package(CUDAToolkit REQUIRED)
find_package(gsl-lite REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(WiseEnum REQUIRED)
find_package(mathdx REQUIRED COMPONENTS cufftdx CONFIG PATHS "/usr/local/nvidia/mathdx/25.06/")

# Generation of compile time LUTs for CRC

set (LUTS_DIR "${CMAKE_CURRENT_BINARY_DIR}/LUTS")

file(MAKE_DIRECTORY ${LUTS_DIR})

add_executable(genLUT crc/gen_crc_LUTs.cpp descrambling/descrambling.hpp)

target_include_directories(genLUT PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(genLUT PUBLIC aerial_utils)

add_custom_command( OUTPUT
            ${LUTS_DIR}/G_CRC_24_A_P_LUT.h
            ${LUTS_DIR}/G_CRC_24_B_P_LUT.h
            ${LUTS_DIR}/G_CRC_16_P_LUT.h
            ${LUTS_DIR}/GOLD_1_SEQ_LUT.h
            ${LUTS_DIR}/GOLD_2_32_P_LUT.h
            DEPENDS crc/crc.hpp crc/gen_crc_LUTs.cpp descrambling/descrambling.hpp cuphy.h
            COMMAND genLUT ${LUTS_DIR}
            COMMENT "Generating LUTs for CRC and Descrambling")

add_custom_target(CRCLUTGEN DEPENDS
            ${LUTS_DIR}/G_CRC_24_A_P_LUT.h
            ${LUTS_DIR}/G_CRC_24_B_P_LUT.h
            ${LUTS_DIR}/G_CRC_16_P_LUT.h
            ${LUTS_DIR}/GOLD_1_SEQ_LUT.h
            ${LUTS_DIR}/GOLD_2_32_P_LUT.h
            )

# Generation of compile time LUTs for PUCCH receiver Format 1

add_executable(genPucchLUT pucch_receiver/gen_pucch_receiver_LUTs.cpp)
target_include_directories(genPucchLUT PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(genPucchLUT PUBLIC aerial_utils)

add_custom_command( OUTPUT
            ${LUTS_DIR}/PUCCH_RECEIVER_F1_TOCC_VALUES_LUT.h
            ${LUTS_DIR}/PUCCH_RECEIVER_F1_TIME_SHIFT_SEQ_VALUES_LUT.h
            ${LUTS_DIR}/PUCCH_RECEIVER_F1_PAPR_SEQ_VALUES_LUT.h
            DEPENDS pucch_receiver/gen_pucch_receiver_LUTs.cpp
            COMMAND genPucchLUT ${LUTS_DIR}
            COMMENT "Generating LUTs for PUCCH receiver Format 1")

add_custom_target(PUCCHLUTGEN DEPENDS
            ${LUTS_DIR}/PUCCH_RECEIVER_F1_TOCC_VALUES_LUT.h
            ${LUTS_DIR}/PUCCH_RECEIVER_F1_TIME_SHIFT_SEQ_VALUES_LUT.h
            ${LUTS_DIR}/PUCCH_RECEIVER_F1_PAPR_SEQ_VALUES_LUT.h
            )

if (NOT CMAKE_BUILD_TYPE MATCHES "Release" OR ENABLE_LINE_INFO)
    list(APPEND CUPHY_NVCC_FLAGS -lineinfo)
endif()
#message("lineinfo flag value: ${CUPHY_NVCC_FLAGS}")

# LDPC library sources are a subset of the cuPHY library sources
set(CUPHY_LDPC_SOURCES
    cuphy.h
    cuphy_ldpc.cpp
    cuphy_common.cpp
    ldpc/ldpc_api.cpp
    ldpc/ldpc_api.hpp

    cuphy_context.hpp
    cuphy_context.cpp
    tensor_desc.cpp
    convert_tensor.cuh
    convert_tensor.cu

    soft_demapper/soft_demapper.hpp
    soft_demapper/soft_demapper.cu
    soft_demapper/soft_demapper.cuh
    soft_demapper/soft_demapper_tables.cuh
    soft_demapper/soft_demapper_tables.h

    error_correction/nrLDPC.cuh
    error_correction/nrLDPC_flat.cuh
    error_correction/ldpc.cuh
    error_correction/ldpc2_kernel.cuh
    error_correction/ldpc2_sign.cuh
    error_correction/ldpc2_sign_split.cuh
    error_correction/ldpc2_llr_loader.cuh
    error_correction/ldpc2_min_sum_update_half_0.cuh
    error_correction/ldpc2_min_sum_update_half_1.cuh
    error_correction/ldpc2_c2v.cuh
    error_correction/ldpc2_box_plus.cuh
    error_correction/ldpc.hpp
    error_correction/ldpc_load_store.cuh
    error_correction/ldpc2_dec_output.cuh
    error_correction/ldpc.cpp
    error_correction/ldpc2.hpp
    error_correction/ldpc2_bg_desc.cpp
    error_correction/ldpc2_bg_desc_half2.cpp
    error_correction/ldpc2_bg_desc.hpp
    error_correction/ldpc2_c2v_cache_register.cuh
    error_correction/ldpc2_c2v_cache_global.cuh
    error_correction/ldpc2_c2v_cache_shared.cuh
    error_correction/ldpc2_c2v_cache_split.cuh
    error_correction/ldpc2_schedule_fixed.cuh
    error_correction/ldpc2_schedule_dynamic.cuh
    error_correction/ldpc2_schedule_dynamic_desc.cuh
    error_correction/ldpc2_schedule_cluster.cuh
    error_correction/ldpc2_app_address.cuh
    error_correction/ldpc2_app_address_fp.cuh
    error_correction/ldpc2_app_address_fp_desc.cuh
    error_correction/ldpc2_app_address_fp_dp_desc.cuh
    error_correction/ldpc2_reg_index_fp_desc_dyn.hpp
    error_correction/ldpc2_reg_index_fp_desc_dyn.cu
    error_correction/ldpc2_reg_index_fp_desc_dyn_small.hpp
    error_correction/ldpc2_reg_index_fp_desc_dyn_small.cu
    error_correction/ldpc2_reg_index_fp_x2_desc_dyn.hpp
    error_correction/ldpc2_reg_index_fp_x2_desc_dyn.cu
    error_correction/ldpc2_reg_index_fp_desc_dyn_row_dep.hpp
    error_correction/ldpc2_reg_index_fp_desc_dyn_row_dep.cu
    error_correction/ldpc2_reg_index_fp_dp_desc_dyn_row_dep.hpp
    error_correction/ldpc2_reg_index_fp_dp_desc_dyn_row_dep.cu
    error_correction/ldpc2_shm_index_fp_desc_dyn.hpp
    error_correction/ldpc2_shm_index_fp_desc_dyn.cu
    error_correction/ldpc2_split_index_fp_x2_desc_dyn.hpp
    error_correction/ldpc2_split_index_fp_x2_desc_dyn.cu
    error_correction/ldpc2_reg_box_plus.hpp
    error_correction/ldpc2_reg_box_plus.cu
    error_correction/ldpc2_c2v_x2.cuh
    error_correction/ldpc2.cuh
    error_correction/ldpc2.cpp
    error_correction/ldpc_encode.cu
    error_correction/nrLDPC_templates.cuh

    rate_matching/rate_matching.cu

    crc/crc.cu
)

add_library(cuphy
    ${CUPHY_LDPC_SOURCES}
    cuphy_err.hpp
    cuphy_utils.hpp
    cuphy.cpp
    cuphy_internal.h
    common_utils.hpp
    device.hpp
    device.cpp
    type_convert.hpp
    empty_kernels.hpp
    empty_kernels.cu
    cuphy_kernel_util.cuh
    bfc/bfc.cu
    channel_est/channel_est.cu
    ch_est/IModule.hpp
    ch_est/IGraph_mgr.hpp
    ch_est/IStream.hpp
    ch_est/chest_factory.hpp
    ch_est/chest_factory.cpp
    ch_est/ch_est.cu
    ch_est/ch_est.hpp
    ch_est/ch_est_stream.hpp
    ch_est/ch_est_stream.cpp
    ch_est/ch_est_graph_mgr.hpp
    ch_est/ch_est_graph_mgr.cpp
    ch_est/ch_est_utils.hpp
    ch_est/ch_est_null_graphs.hpp
    ch_est/ch_est_types.hpp
    ch_est/trtengine_chest.cpp
    ch_est/trtengine_chest.hpp
    ch_est/ch_est_config_loader.hpp
    ch_est/ch_est_yaml_loader.cpp
    ch_est/ch_est_trtengine_pre_post_conversion.hpp
    ch_est/ch_est_trtengine_pre_post_converion.cu
    pusch_start_kernels_interface.hpp
    pusch_start_kernels.hpp
    pusch_start_kernels.cu
    pusch_noise_intf_est/pusch_noise_intf_est.cu
    cfo_ta_est/cfo_ta_est.hpp
    cfo_ta_est/cfo_ta_est.cu
    channel_eq/channel_eq.cu
    dl_rate_matching/dl_rate_matching.cuh
    dl_rate_matching/dl_rate_matching.cu
    descrambling/descrambling.cu
    pusch_rssi/pusch_rssi.cu
    pucch_receiver/pucch_receiver.cu
    pucch_receiver/rm_decoder.cu
    pucch_receiver/rm_decoder.hpp
    simplex_decoder/simplex_decoder.cu
    simplex_decoder/simplex_decoder.hpp
    prach_receiver/prach_receiver.cu
    modulation_mapper/modulation_mapper.cu
    pdcch/embed_pdcch_tf_signal.cu
    pdsch_dmrs/pdsch_dmrs.cu
    ss/ss.cu
    trt_engine/trt_engine.cpp
    trt_engine/trt_engine.hpp
    trt_engine/trt_engine_params.hpp
    trt_engine/trt_engine_interfaces.hpp
    polar_encoder/polar_encoder.cu
    comp_cwTreeTypes/comp_cwTreeTypes.cu
    comp_cwTreeTypes/comp_cwTreeTypes.hpp
    pucch_F0_receiver/pucch_F0_receiver.cu
    pucch_F0_receiver/pucch_F0_receiver.hpp
    pucch_F1_receiver/pucch_F1_receiver.cu
    pucch_F1_receiver/pucch_F1_receiver.hpp
    pucch_F2_front_end/pucch_F2_front_end.cu
    pucch_F2_front_end/pucch_F2_front_end.hpp
    pucch_F3_front_end/pucch_F3_front_end.cu
    pucch_F3_front_end/pucch_F3_front_end.hpp
    pucch_F3_front_end/pucch_F3_csi2Ctrl.cu
    pucch_F3_front_end/pucch_F3_csi2Ctrl.hpp
    pucch_F3_front_end/pucch_F3_segLLRs.cu
    pucch_F3_front_end/pucch_F3_segLLRs.hpp
    pucch_F234_uci_seg/pucch_F234_uci_seg.cu
    pucch_F234_uci_seg/pucch_F234_uci_seg.hpp
    csirs/csirs.cuh
    csirs/csirs_tf_signal.cu
    csirs_rx/csirs_rx.cuh
    csirs_rx/csirs_rx.cu
    srstx/srstx_tf_signal.cu
    rng/rng.hpp
    rng/rng.cu
    variant.hpp
    variant.cpp
    tensor_fill.hpp
    tensor_fill.cu
    tensor_tile.hpp
    tensor_tile.cu
    tensor_elementwise.hpp
    tensor_elementwise.cu
    tensor_reduction.hpp
    tensor_reduction.cu
    polar_seg_deRm_deItl/polar_seg_deRm_deItl.cu
    polar_seg_deRm_deItl/polar_seg_deRm_deItl.hpp
    uci_on_pusch/uciOnPusch_segLLRs1.cu
    uci_on_pusch/uciOnPusch_segLLRs1.hpp
    polar_decoder/polar_decoder.cu
    polar_decoder/polar_decoder.hpp
    uci_on_pusch/uciOnPusch_segLLRs2.cu
    uci_on_pusch/uciOnPusch_segLLRs2.hpp
    uci_on_pusch/uciOnPusch_segLLRs0.cu
    uci_on_pusch/uciOnPusch_segLLRs0.hpp
    uci_on_pusch/uciOnPusch_csi2Ctrl.cu
    uci_on_pusch/uciOnPusch_csi2Ctrl.hpp
    srs_chEst/srs_chEst.cu
    srs_chEst/srs_chEst.hpp
    cuphy_pti.cpp
    cuphy_pti.cu
    copy_kernel.cu
    cupti_helper.cpp
)


# ----------------------------------------------------------------------
# cuphy_pti library - Performance Timing Interface
# ----------------------------------------------------------------------
add_library(cuphy_pti STATIC cuphy_pti.cpp cuphy_pti.cu)
set_property(TARGET cuphy_pti PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(cuphy_pti PRIVATE ${AERIAL_ARCH_TUNE_FLAGS})

# Public headers for cuphy_pti
target_include_directories(cuphy_pti PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# cuphy_pti dependencies
set(NVLOG_LIB nvlog)
if (NOT BUILD_SHARED_LIBS)
    set(NVLOG_LIB nvlog_static)
endif()
target_link_libraries(cuphy_pti PUBLIC CUDA::cudart ${NVLOG_LIB})
target_link_libraries(cuphy_pti PUBLIC aerial_utils)

# Install cuphy_pti
install(TARGETS cuphy_pti DESTINATION lib)
install(FILES cuphy_pti.hpp DESTINATION include)

# Create an alias for namespaced usage
add_library(cuphy::cuphy_pti ALIAS cuphy_pti)

# ----------------------------------------------------------------------
# Main cuphy library
# ----------------------------------------------------------------------

set_property(TARGET cuphy PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_options(cuphy PRIVATE ${AERIAL_ARCH_TUNE_FLAGS})

target_link_options(cuphy PRIVATE $<$<CONFIG:Release>:-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cuphy.map -s>)

add_dependencies(cuphy CRCLUTGEN)
add_dependencies(cuphy PUCCHLUTGEN)
target_include_directories(cuphy PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        # If wise_enum is installed in the container, we could just have it in target_link_libraries WiseEnum::wise_enum
        # and the include would be wise_enum/wise_enum.
        ${wise_enum_SOURCE_DIR})
target_include_directories(cuphy PRIVATE crc
                                         ../../examples/common/
                                         ../compression_decompression
       	                                 bfc
                                         channel_est
                                         channel_eq
                                         ch_est
                                         csirs
                                         csirs_rx
                                         pusch_noise_intf_est
                                         Gold_sequence
                                         pucch_F0_receiver
                                         pucch_F1_receiver
                                         pucch_F2_front_end
                                         pucch_F3_front_end
                                         pucch_F234_uci_seg
                                         cfo_ta_est
                                         descrambling
                                         dl_rate_matching
                                         rate_matching
                                         modulation_mapper
                                         "${CMAKE_CURRENT_BINARY_DIR}/LUTS"
                                         error_correction
                                         pusch_rssi
                                         pucch_receiver
                                         simplex_decoder
                                         prach_receiver
                                         pdsch_dmrs
                                         polar_encoder
                                         pdcch
                                         ss
                                         trt_engine
                                         soft_demapper
                                         rng
                                         comp_cwTreeTypes
                                         polar_seg_deRm_deItl
                                         uci_on_pusch
                                         polar_decoder
                                         srs_chEst
                                         ../../nvlog
                                         )

set(CUPHY_OPTS ${CUPHY_PTXAS_OPTIONS} --cudart static ${CUPHY_NVCC_FLAGS} --threads 0 --expt-relaxed-constexpr)

target_compile_options(cuphy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUPHY_OPTS}>)

set_source_files_properties(error_correction/ldpc_ms_cta_shmem_layered.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(pucch_receiver/pucch_receiver.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(prach_receiver/prach_receiver.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(ss/ss.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(srs_chEst/srs_chEst.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(cfo_est/cfo_est.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(channel_eq/channel_eq.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(ch_est/ch_est.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(pusch_noise_intf_est/pusch_noise_intf_est.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(pusch_rssi/pusch_rssi.cu PROPERTIES COMPILE_FLAGS --use_fast_math)

install(TARGETS cuphy DESTINATION lib)
install(FILES cuphy.h utils.cuh DESTINATION include)

find_library(NVINFER libnvinfer.so)

set(NVLOG_LIB nvlog)
if (NOT BUILD_SHARED_LIBS)
    set(NVLOG_LIB nvlog_static)
endif()

# Link with cuda library for driver API
target_link_libraries(cuphy PUBLIC cuda CUDA::nvtx3 ${NVLOG_LIB} ${NVINFER} gsl-lite::gsl-lite yaml-cpp mathdx::cufftdx)
target_link_libraries(cuphy PUBLIC aerial_utils)
target_link_libraries(cuphy PUBLIC CUDA::cupti)

# Function to setup LDPC decoder cubins for a target
function(setup_ldpc_cubins TARGET_NAME)
    set(LDPC_WRAPPER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/error_correction/ldpc_cubin_wrappers")
    set(LDPC_CUBIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/error_correction/ldpc_decoder_cubin")

    if (ENABLE_CUPHY_INTERNAL_LDPC_DECODER)
        include(../../../../internal/cuPHY/src/cuphy/cuphy_lib_internal.cmake)
        target_include_directories(${TARGET_NAME} PRIVATE ${LDPC_WRAPPER_DIR})
        target_compile_definitions(${TARGET_NAME} PRIVATE USE_LDPC_CUBIN)
    else()
        if (EXISTS "${LDPC_WRAPPER_DIR}" AND EXISTS "${LDPC_CUBIN_DIR}/ldpc_decoder_cubin.s")
            target_sources(${TARGET_NAME} PRIVATE "${LDPC_CUBIN_DIR}/ldpc_decoder_cubin.s")
            target_sources(${TARGET_NAME} PRIVATE
                "${LDPC_WRAPPER_DIR}/ldpc2_reg_index_fp_desc_dyn_sm80.cu"
                "${LDPC_WRAPPER_DIR}/ldpc2_reg_index_fp_desc_dyn_row_dep_sm80.cu"
                "${LDPC_WRAPPER_DIR}/ldpc2_reg_index_fp_desc_dyn_row_dep_sm86.cu"
                "${LDPC_WRAPPER_DIR}/ldpc2_reg_index_fp_desc_dyn_row_dep_sm90.cu"
                "${LDPC_WRAPPER_DIR}/ldpc2_split_index_fp_x2_desc_dyn_sm86.cu"
                "${LDPC_WRAPPER_DIR}/ldpc2_split_index_fp_x2_desc_dyn_sm90.cu")

            target_include_directories(${TARGET_NAME} PRIVATE ${LDPC_WRAPPER_DIR} ${LDPC_CUBIN_DIR})
            target_compile_definitions(${TARGET_NAME} PRIVATE USE_LDPC_CUBIN)
        endif()
    endif()

endfunction()

# Setup LDPC decoder cubins for cuphy
setup_ldpc_cubins(cuphy)

# Header-only INTERFACE library for cuPHY
add_library(cuphy_h INTERFACE)
add_library(cuphy::cuphy_h ALIAS cuphy_h)

target_include_directories(cuphy_h INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(cuphy_h INTERFACE cxx_std_20)

# ----------------------------------------------------------------------
# Stripped down cuphy_ldpc library
# ----------------------------------------------------------------------

add_library(cuphy_ldpc ${CUPHY_LDPC_SOURCES} ${LDPC_CUDA_SOURCES})
add_library(cuphy::cuphy_ldpc ALIAS cuphy_ldpc)

add_dependencies(cuphy_ldpc CRCLUTGEN)

target_include_directories(cuphy_ldpc
    PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../examples/common>
    PRIVATE
        ../../nvlog
        error_correction
        error_correction/ldpc_decoder_cubin
        rate_matching
        descrambling
        crc
        "${CMAKE_CURRENT_BINARY_DIR}/LUTS"
)

target_link_libraries(cuphy_ldpc PUBLIC cuda CUDA::nvtx3 ${NVLOG_LIB} gsl-lite::gsl-lite)
target_link_libraries(cuphy_ldpc PUBLIC aerial_utils)

set_target_properties(cuphy_ldpc PROPERTIES
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:cuphy_ldpc,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_options(cuphy_ldpc PRIVATE
                       ${AERIAL_ARCH_TUNE_FLAGS}
                       $<$<COMPILE_LANGUAGE:CUDA>:${CUPHY_OPTS}>)

# Setup LDPC decoder cubins for cuphy_ldpc
setup_ldpc_cubins(cuphy_ldpc)
