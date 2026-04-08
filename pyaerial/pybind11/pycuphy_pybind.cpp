/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <vector>
#include <cstdint>
#include <complex>

#include "nvlog.h"
#include "cuda_array_interface.hpp"
#include "pycuphy_util.hpp"
#include "pycuphy_params.hpp"
#include "pycuphy_pusch.hpp"
#include "pycuphy_pdsch.hpp"
#include "pycuphy_dmrs.hpp"
#include "pycuphy_csirs_tx.hpp"
#include "pycuphy_csirs_rx.hpp"
#include "pycuphy_ldpc.hpp"
#include "pycuphy_crc_encode.hpp"
#include "pycuphy_crc_check.hpp"
#include "pycuphy_channel_est.hpp"
#include "pycuphy_noise_intf_est.hpp"
#include "pycuphy_cfo_ta_est.hpp"
#include "pycuphy_channel_eq.hpp"
#include "pycuphy_srs_chest.hpp"
#include "pycuphy_srs_tx.hpp"
#include "pycuphy_srs_rx.hpp"
#include "pycuphy_trt_engine.hpp"
#include "pycuphy_rsrp.hpp"
#include "pycuphy_chan_model.hpp"

// Add channel models includes
#include "chanModelsApi.hpp"
#include "chanModelsDataset.hpp"

namespace py = pybind11;

namespace pycuphy {

template <typename T>
void declare_cuda_array(py::module &m, const char *name) {
  py::class_<cuda_array_t<T>>(m, name)
      .def(py::init([](py::object obj) {
        if (!py::hasattr(obj, "__cuda_array_interface__")) {
          throw py::type_error(
              "Object must implement __cuda_array_interface__");
        }
        return std::make_unique<cuda_array_t<T>>(obj);
      }))
      .def(py::init([](intptr_t addr, const std::vector<size_t>& shape, const std::vector<size_t>& strides) {
        return std::make_unique<cuda_array_t<T>>(addr, shape, strides);
      }), py::arg("addr"), py::arg("shape"), py::arg("strides"))
      .def_property_readonly("shape", [](const cuda_array_t<T>& array) { return as_tuple(array.get_shape()); })
      .def_property_readonly("strides", [](const cuda_array_t<T>& array) { return as_tuple(array.get_strides()); })
      .def_property_readonly("size", &cuda_array_t<T>::get_size)
      .def_property_readonly("ndim", &cuda_array_t<T>::get_ndim)
      .def("is_readonly", &cuda_array_t<T>::is_readonly)
      .def("has_stride_info", &cuda_array_t<T>::has_stride_info)
      .def_property_readonly("__cuda_array_interface__", &cuda_array_t<T>::get_interface_dict);
}

// Helper function to convert Python list/vector to C-style array
template<typename T, size_t N>
void vector_to_carray(const std::vector<T>& vec, T (&dest)[N]) {
    if (vec.size() > N) {
        throw std::runtime_error("List size exceeds destination array capacity");
    }
    
    // Copy the source elements
    size_t copy_size = std::min(vec.size(), N);
    std::copy(vec.begin(), vec.begin() + copy_size, dest);
    
    // Zero-fill any remaining elements to prevent stale data
    if (copy_size < N) {
        std::fill(dest + copy_size, dest + N, T{});
    }
}

// Copy numpy/buffer array into fixed C array (at most N elements; remainder zero-filled).
template<typename T, size_t N>
void numpy_to_carray(const py::array_t<T>& arr, T (&dest)[N]) {
    py::buffer_info buf = arr.request();
    if (buf.size > N) {
        throw std::runtime_error("Array size exceeds destination array capacity");
    }
    
    // Copy the source elements
    T* ptr = static_cast<T*>(buf.ptr);
    size_t copy_size = std::min(static_cast<size_t>(buf.size), N);
    std::copy(ptr, ptr + copy_size, dest);
    
    // Zero-fill any remaining elements to prevent stale data
    if (copy_size < N) {
        std::fill(dest + copy_size, dest + N, T{});
    }
}

// C fixed-size array -> Python list (small channel-model vectors; CIR/CFR buffers stay ndarray).
template<typename T, size_t N>
py::list carray_to_pylist(const T (&src)[N]) {
    py::list out;
    for (size_t i = 0; i < N; ++i) {
        out.append(src[i]);
    }
    return out;
}

// Python list, tuple, or ndarray -> fixed C array (getters return list; setters accept array-likes).
template<typename T, size_t N>
void object_to_fixed_carray(const py::object& obj, T (&dest)[N]) {
    py::array_t<T> arr = py::array_t<T>::ensure(obj);
    if (arr) {
        numpy_to_carray(arr, dest);
        return;
    }
    py::sequence seq = obj.cast<py::sequence>();
    std::vector<T> vec;
    for (auto item : seq) {
        vec.push_back(py::cast<T>(item));
    }
    vector_to_carray(vec, dest);
}


}  // namespace pycuphy


PYBIND11_MODULE(_pycuphy, m) {
    m.doc() = "Python bindings for cuPHY"; // optional module docstring

    pycuphy::declare_cuda_array<int>(m, "CudaArrayInt");
    pycuphy::declare_cuda_array<uint8_t>(m, "CudaArrayUint8");
    pycuphy::declare_cuda_array<uint16_t>(m, "CudaArrayUint16");
    pycuphy::declare_cuda_array<uint32_t>(m, "CudaArrayUint32");
    pycuphy::declare_cuda_array<__half>(m, "CudaArrayHalf");
    pycuphy::declare_cuda_array<float>(m, "CudaArrayFloat");
    pycuphy::declare_cuda_array<std::complex<float>>(m, "CudaArrayComplexFloat");

    m.def("device_to_numpy", &pycuphy::deviceToNumpy<std::complex<float>>);
    m.def("device_to_numpy", &pycuphy::deviceToNumpy<float>);
    m.def("convert_to_complex64", &pycuphy::complexHalfToComplexFloat);
    m.def("get_tb_size", &pycuphy::get_tb_size);
    m.def("set_nvlog_level", &nvlog_set_log_level);

    // Enums here.
    py::enum_<pycuphy::EnableScrambling>(m, "EnableScrambling", py::arithmetic(), "Enable scrambling for RM")
        .value("ENABLED", pycuphy::EnableScrambling::ENABLED)
        .value("DISABLED", pycuphy::EnableScrambling::DISABLED)
        .export_values();

    py::enum_<cuphyPuschProcMode_t>(m, "PuschProcMode", py::arithmetic(), "PUSCH processing modes")
        .value("PUSCH_PROC_MODE_FULL_SLOT", cuphyPuschProcMode_t::PUSCH_PROC_MODE_FULL_SLOT)
        .value("PUSCH_PROC_MODE_FULL_SLOT_GRAPHS", cuphyPuschProcMode_t::PUSCH_PROC_MODE_FULL_SLOT_GRAPHS)
        .value("PUSCH_PROC_MODE_SUB_SLOT", cuphyPuschProcMode_t::PUSCH_PROC_MODE_SUB_SLOT)
        .value("PUSCH_MAX_PROC_MODES", cuphyPuschProcMode_t::PUSCH_MAX_PROC_MODES)
        .export_values();

    py::enum_<cuphyPuschLdpcKernelLaunch_t>(m, "PuschLdpcKernelLaunch", py::arithmetic(), "PUSCH kernel launch modes")
        .value("PUSCH_RX_ENABLE_DRIVER_LDPC_LAUNCH", cuphyPuschLdpcKernelLaunch_t::PUSCH_RX_ENABLE_DRIVER_LDPC_LAUNCH)
        .value("PUSCH_RX_LDPC_STREAM_POOL", cuphyPuschLdpcKernelLaunch_t::PUSCH_RX_LDPC_STREAM_POOL)
        .value("PUSCH_RX_LDPC_STREAM_SEQUENTIAL", cuphyPuschLdpcKernelLaunch_t::PUSCH_RX_LDPC_STREAM_SEQUENTIAL)
        .value("PUSCH_RX_ENABLE_LDPC_DEC_SINGLE_STREAM_OPT", cuphyPuschLdpcKernelLaunch_t::PUSCH_RX_ENABLE_LDPC_DEC_SINGLE_STREAM_OPT)
        .export_values();

    py::enum_<cuphyPuschWorkCancelMode_t>(m, "PuschWorkCancelMode", py::arithmetic(), "PUSCH work cancellation modes")
        .value("PUSCH_NO_WORK_CANCEL", cuphyPuschWorkCancelMode_t::PUSCH_NO_WORK_CANCEL)
        .value("PUSCH_COND_IF_NODES_W_KERNEL", cuphyPuschWorkCancelMode_t::PUSCH_COND_IF_NODES_W_KERNEL)
        .value("PUSCH_DEVICE_GRAPHS", cuphyPuschWorkCancelMode_t::PUSCH_DEVICE_GRAPHS)
        .value("PUSCH_MAX_WORK_CANCEL_MODES", cuphyPuschWorkCancelMode_t::PUSCH_MAX_WORK_CANCEL_MODES)
        .export_values();

    py::enum_<cuphyLdpcMaxItrAlgoType_t>(m, "LdpcMaxItrAlgoType", py::arithmetic(), "LDPC number of iterations algorithm types")
        .value("LDPC_MAX_NUM_ITR_ALGO_TYPE_FIXED", cuphyLdpcMaxItrAlgoType_t::LDPC_MAX_NUM_ITR_ALGO_TYPE_FIXED)
        .value("LDPC_MAX_NUM_ITR_ALGO_TYPE_LUT", cuphyLdpcMaxItrAlgoType_t::LDPC_MAX_NUM_ITR_ALGO_TYPE_LUT)
        .export_values();

    py::enum_<cuphyDataType_t>(m, "DataType", py::arithmetic(), "Data types")
        .value("CUPHY_VOID", cuphyDataType_t::CUPHY_VOID, "Uninitialized type")
        .value("CUPHY_BIT", cuphyDataType_t::CUPHY_BIT, "1-bit value")
        .value("CUPHY_R_8I", cuphyDataType_t::CUPHY_R_8I, "8-bit signed integer real values")
        .value("CUPHY_C_8I", cuphyDataType_t::CUPHY_C_8I, "8-bit signed integer complex values")
        .value("CUPHY_R_8U", cuphyDataType_t::CUPHY_R_8U, "8-bit unsigned integer real values")
        .value("CUPHY_C_8U", cuphyDataType_t::CUPHY_C_8U, "8-bit unsigned integer complex values")
        .value("CUPHY_R_16I", cuphyDataType_t::CUPHY_R_16I, "16-bit signed integer real values")
        .value("CUPHY_C_16I", cuphyDataType_t::CUPHY_C_16I, "16-bit signed integer complex values")
        .value("CUPHY_R_16U", cuphyDataType_t::CUPHY_R_16U, "16-bit unsigned integer real values")
        .value("CUPHY_C_16U", cuphyDataType_t::CUPHY_C_16U, "16-bit unsigned integer complex values")
        .value("CUPHY_R_32I", cuphyDataType_t::CUPHY_R_32I, "32-bit signed integer real values")
        .value("CUPHY_C_32I", cuphyDataType_t::CUPHY_C_32I, "32-bit signed integer complex values")
        .value("CUPHY_R_32U", cuphyDataType_t::CUPHY_R_32U, "32-bit unsigned integer real values")
        .value("CUPHY_C_32U", cuphyDataType_t::CUPHY_C_32U, "32-bit unsigned integer complex values")
        .value("CUPHY_R_16F", cuphyDataType_t::CUPHY_R_16F, "Half precision (16-bit) real values")
        .value("CUPHY_C_16F", cuphyDataType_t::CUPHY_C_16F, "Half precision (16-bit) complex values")
        .value("CUPHY_R_32F", cuphyDataType_t::CUPHY_R_32F, "Single precision (32-bit) real values")
        .value("CUPHY_C_32F", cuphyDataType_t::CUPHY_C_32F, "Single precision (32-bit) complex values")
        .value("CUPHY_R_64F", cuphyDataType_t::CUPHY_R_64F, "Double precision (64-bit) real values")
        .value("CUPHY_C_64F", cuphyDataType_t::CUPHY_C_64F, "Double precision (64-bit) complex values")
        .export_values();

    py::enum_<cuphyPuschSetupPhase_t>(m, "PuschSetupPhase", py::arithmetic(), "PUSCH setup phases")
        .value("PUSCH_SETUP_PHASE_INVALID", cuphyPuschSetupPhase_t::PUSCH_SETUP_PHASE_INVALID)
        .value("PUSCH_SETUP_PHASE_1", cuphyPuschSetupPhase_t::PUSCH_SETUP_PHASE_1)
        .value("PUSCH_SETUP_PHASE_2", cuphyPuschSetupPhase_t::PUSCH_SETUP_PHASE_2)
        .value("PUSCH_SETUP_MAX_PHASES", cuphyPuschSetupPhase_t::PUSCH_SETUP_MAX_PHASES)
        .value("PUSCH_SETUP_MAX_VALID_PHASES", cuphyPuschSetupPhase_t::PUSCH_SETUP_MAX_VALID_PHASES)
        .export_values();

    py::enum_<cuphyPuschRunPhase_t>(m, "PuschRunPhase", py::arithmetic(), "PUSCH run phases")
        .value("PUSCH_RUN_PHASE_INVALID", cuphyPuschRunPhase_t::PUSCH_RUN_PHASE_INVALID)
        .value("PUSCH_RUN_SUB_SLOT_PROC", cuphyPuschRunPhase_t::PUSCH_RUN_SUB_SLOT_PROC)
        .value("PUSCH_RUN_FULL_SLOT_PROC", cuphyPuschRunPhase_t::PUSCH_RUN_FULL_SLOT_PROC)
        .value("PUSCH_RUN_FULL_SLOT_COPY", cuphyPuschRunPhase_t::PUSCH_RUN_FULL_SLOT_COPY)
        .value("PUSCH_RUN_ALL_PHASES", cuphyPuschRunPhase_t::PUSCH_RUN_ALL_PHASES)
        .value("PUSCH_RUN_MAX_PHASES", cuphyPuschRunPhase_t::PUSCH_RUN_MAX_PHASES)
        .value("PUSCH_RUN_MAX_VALID_PHASES", cuphyPuschRunPhase_t::PUSCH_RUN_MAX_VALID_PHASES)
        .export_values();

    py::enum_<cuphyPuschEqCoefAlgoType_t>(m, "PuschEqCoefAlgoType", py::arithmetic(), "PUSCH equalizer algorithm types")
        .value("PUSCH_EQ_ALGO_TYPE_RZF", cuphyPuschEqCoefAlgoType_t::PUSCH_EQ_ALGO_TYPE_RZF)
        .value("PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE", cuphyPuschEqCoefAlgoType_t::PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE)
        .value("PUSCH_EQ_ALGO_TYPE_MMSE_IRC", cuphyPuschEqCoefAlgoType_t::PUSCH_EQ_ALGO_TYPE_MMSE_IRC)
        .value("PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_RBLW", cuphyPuschEqCoefAlgoType_t::PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_RBLW)
        .value("PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_OAS", cuphyPuschEqCoefAlgoType_t::PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_OAS)
        .value("PUSCH_EQ_ALGO_MAX_TYPES", cuphyPuschEqCoefAlgoType_t::PUSCH_EQ_ALGO_MAX_TYPES)
        .export_values();

    py::enum_<cuphyPuschStatusType_t>(m, "PuschStatusType", py::arithmetic(), "PUSCH status types")
        .value("CUPHY_PUSCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE", cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE)
        .value("CUPHY_PUSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB", cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB)
        .value("CUPHY_PUSCH_STATUS_TBSIZE_MISMATCH", cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_TBSIZE_MISMATCH)
        .value("CUPHY_MAX_PUSCH_STATUS_TYPES", cuphyPuschStatusType_t::CUPHY_MAX_PUSCH_STATUS_TYPES)
        .export_values();

    py::enum_<cuphyPuschChEstAlgoType_t>(m, "PuschChEstAlgoType", py::arithmetic(), "PUSCH channel estimation algorithm types")
        .value("PUSCH_CH_EST_ALGO_TYPE_LEGACY_MMSE", cuphyPuschChEstAlgoType_t::PUSCH_CH_EST_ALGO_TYPE_LEGACY_MMSE)
        .value("PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST", cuphyPuschChEstAlgoType_t::PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST)
        .value("PUSCH_CH_EST_ALGO_TYPE_RKHS", cuphyPuschChEstAlgoType_t::PUSCH_CH_EST_ALGO_TYPE_RKHS)
        .value("PUSCH_CH_EST_ALGO_TYPE_LS_ONLY", cuphyPuschChEstAlgoType_t::PUSCH_CH_EST_ALGO_TYPE_LS_ONLY)
        .export_values();

    // Full channel pipelines.
    py::class_<pycuphy::PdschPipeline>(m, "PdschPipeline")
        .def(py::init<const py::object&>())
        .def("setup_pdsch_tx", &pycuphy::PdschPipeline::setupPdschTx)
        .def("run_pdsch_tx", &pycuphy::PdschPipeline::runPdschTx)
        .def("get_ldpc_output", &pycuphy::PdschPipeline::getLdpcOutputPerTbPerCell);

    py::class_<pycuphy::PuschPipeline>(m, "PuschPipeline")
        .def(py::init<const py::object&, uint64_t>())
        .def("setup_pusch_rx", &pycuphy::PuschPipeline::setupPuschRx)
        .def("run_pusch_rx", &pycuphy::PuschPipeline::runPuschRx)
        .def("write_dbg_buf_synch", &pycuphy::PuschPipeline::writeDbgBufSynch);

    // Individual Tx/Rx components.
    py::class_<pycuphy::PyPdschDmrsTx>(m, "DmrsTx")
        .def(py::init<uint64_t, uint32_t, uint32_t>())
        .def("run", &pycuphy::PyPdschDmrsTx::run);

    py::class_<pycuphy::PyCsiRsTx>(m, "CsiRsTx")
        .def(py::init<const std::vector<uint16_t>&, const std::vector<uint16_t>&>())
        .def("run", &pycuphy::PyCsiRsTx::run);

    py::class_<pycuphy::PyCsiRsRx>(m, "CsiRsRx")
        .def(py::init<const std::vector<uint16_t>&>())
        .def("run", &pycuphy::PyCsiRsRx::run);

    py::class_<pycuphy::PyCrcEncoder>(m, "CrcEncoder")
        .def(py::init<uint64_t, uint32_t>())
        .def("encode", &pycuphy::PyCrcEncoder::encode)
        .def("get_num_info_bits", &pycuphy::PyCrcEncoder::getNumInfoBits);

    py::class_<pycuphy::PyLdpcEncoder>(m, "LdpcEncoder")
        .def(py::init<uint64_t, uint64_t>())
        .def("encode", &pycuphy::PyLdpcEncoder::encode)
        .def("set_puncturing", &pycuphy::PyLdpcEncoder::setPuncturing)
        .def("get_cb_size", &pycuphy::PyLdpcEncoder::getCbSize);

    py::class_<pycuphy::PyLdpcDecoder>(m, "LdpcDecoder")
        .def(py::init<const uint64_t>())
        .def("decode", &pycuphy::PyLdpcDecoder::decode)
        .def("set_num_iterations", &pycuphy::PyLdpcDecoder::setNumIterations)
        .def("get_soft_outputs", &pycuphy::PyLdpcDecoder::getSoftOutputs)
        .def("set_throughput_mode", &pycuphy::PyLdpcDecoder::setThroughputMode);

    py::class_<pycuphy::PyLdpcRateMatch>(m, "LdpcRateMatch")
        .def(py::init<pycuphy::EnableScrambling, uint16_t, uint32_t, uint32_t, uint64_t>())
        .def("rate_match", &pycuphy::PyLdpcRateMatch::rateMatch)
        .def("get_num_rm_bits", &pycuphy::PyLdpcRateMatch::getNumRmBitsPerCb)
        .def("rm_mod_layer_map", &pycuphy::PyLdpcRateMatch::rmModLayerMap);

    py::class_<pycuphy::PyLdpcDerateMatch>(m, "LdpcDerateMatch")
        .def(py::init<const bool, const uint64_t>())
        .def("derate_match", &pycuphy::PyLdpcDerateMatch::derateMatch);

    py::class_<pycuphy::PyCrcChecker>(m, "CrcChecker")
        .def(py::init<const uint64_t>())
        .def("check_crc", &pycuphy::PyCrcChecker::checkCrc)
        .def("get_tb_crcs", &pycuphy::PyCrcChecker::getTbCrcs)
        .def("get_cb_crcs", &pycuphy::PyCrcChecker::getCbCrcs);

    py::class_<pycuphy::PySrsChannelEstimator>(m, "SrsChannelEstimator")
        .def(py::init<uint8_t, uint8_t, float, uint8_t, const py::dict&, uint64_t>())
        .def("estimate", &pycuphy::PySrsChannelEstimator::estimate)
        .def("get_srs_report", &pycuphy::PySrsChannelEstimator::getSrsReport)
        .def("get_rb_snr_buffer", &pycuphy::PySrsChannelEstimator::getRbSnrBuffer)
        .def("get_rb_snr_buffer_offsets", &pycuphy::PySrsChannelEstimator::getRbSnrBufferOffsets);

    py::class_<pycuphy::PySrsTx>(m, "SrsTx")
        .def(py::init<uint16_t, uint16_t, uint16_t, uint64_t>())
        .def("run", &pycuphy::PySrsTx::run);

    py::class_<pycuphy::PySrsRx>(m, "SrsRx")
        .def(py::init<uint16_t, const std::vector<uint16_t>&, uint8_t, uint8_t, const pybind11::dict&, uint16_t, uint64_t>())
        .def("run", &pycuphy::PySrsRx::run)
        .def("get_ch_est_to_L2", &pycuphy::PySrsRx::getChEstToL2)
        .def("get_srs_report", &pycuphy::PySrsRx::getSrsReport)
        .def("get_rb_snr_buffer", &pycuphy::PySrsRx::getRbSnrBuffer)
        .def("get_rb_snr_buffer_offsets", &pycuphy::PySrsRx::getRbSnrBufferOffsets);

    py::class_<cuphySrsReport_t>(m, "SrsReport")  // A read-only struct for passing the SRS reports.
        .def(py::init<>())
        .def_property_readonly("to_est_ms", [](const cuphySrsReport_t& prm) { return prm.toEstMicroSec; })
        .def_property_readonly("wideband_snr", [](const cuphySrsReport_t& prm) { return prm.widebandSnr; })
        .def_property_readonly("wideband_noise_energy", [](const cuphySrsReport_t& prm) { return prm.widebandNoiseEnergy; })
        .def_property_readonly("wideband_signal_energy", [](const cuphySrsReport_t& prm) { return prm.widebandSignalEnergy; })
        .def_property_readonly("wideband_sc_corr", [](const cuphySrsReport_t& prm) { return std::complex<float>(__high2float(prm.widebandScCorr), __low2float(prm.widebandScCorr)); })
        .def_property_readonly("wideband_cs_corr_ratio_db", [](const cuphySrsReport_t& prm) { return prm.widebandCsCorrRatioDb; })
        .def_property_readonly("wideband_cs_corr_use", [](const cuphySrsReport_t& prm) { return prm.widebandCsCorrUse; })
        .def_property_readonly("wideband_cs_corr_not_use", [](const cuphySrsReport_t& prm) { return prm.widebandCsCorrNotUse; })
        .def_property_readonly("high_density_ant_port_flag", [](const cuphySrsReport_t& prm) { return prm.highDensityAntPortFlag; });

    py::class_<pycuphy::PyChannelEstimator>(m, "ChannelEstimator")
        .def(py::init<const pycuphy::PuschParams&, const uint64_t>())
        .def("estimate", &pycuphy::PyChannelEstimator::estimate);

    py::class_<pycuphy::PyNoiseIntfEstimator>(m, "NoiseIntfEstimator")
        .def(py::init<const uint64_t>())
        .def("estimate", &pycuphy::PyNoiseIntfEstimator::estimate)
        .def("get_info_noise_var_pre_eq", &pycuphy::PyNoiseIntfEstimator::getInfoNoiseVarPreEq);

    py::class_<pycuphy::PyChannelEqualizer>(m, "ChannelEqualizer")
        .def(py::init<const uint64_t>())
        .def("equalize", &pycuphy::PyChannelEqualizer::equalize)
        .def("get_data_eq", &pycuphy::PyChannelEqualizer::getDataEq)
        .def("get_eq_coef", &pycuphy::PyChannelEqualizer::getEqCoef)
        .def("get_ree_diag_inv", &pycuphy::PyChannelEqualizer::getReeDiagInv);

    py::class_<pycuphy::PyCfoTaEstimator>(m, "CfoTaEstimator")
        .def(py::init<const uint64_t>())
        .def("estimate", &pycuphy::PyCfoTaEstimator::estimate)
        .def("get_cfo_hz", &pycuphy::PyCfoTaEstimator::getCfoHz)
        .def("get_ta", &pycuphy::PyCfoTaEstimator::getTaEst)
        .def("get_cfo_phase_rot", &pycuphy::PyCfoTaEstimator::getCfoPhaseRot)
        .def("get_ta_phase_rot", &pycuphy::PyCfoTaEstimator::getTaPhaseRot);

    py::class_<pycuphy::PyRsrpEstimator>(m, "RsrpEstimator")
        .def(py::init<const uint64_t>())
        .def("estimate", &pycuphy::PyRsrpEstimator::estimate)
        .def("get_info_noise_var_post_eq", &pycuphy::PyRsrpEstimator::getInfoNoiseVarPostEq)
        .def("get_sinr_pre_eq", &pycuphy::PyRsrpEstimator::getSinrPreEq)
        .def("get_sinr_post_eq", &pycuphy::PyRsrpEstimator::getSinrPostEq);

    py::class_<pycuphy::PuschParams>(m, "PuschParams")
        .def(py::init<>())
        .def("set_filters", &pycuphy::PuschParams::setFilters)
        .def("print_stat_prms", &pycuphy::PuschParams::printStatPrms)
        .def("print_dyn_prms", &pycuphy::PuschParams::printDynPrms)
        .def("set_dyn_prms", py::overload_cast<const py::object&>(&pycuphy::PuschParams::setDynPrms))
        .def("set_stat_prms", py::overload_cast<const py::object&>(&pycuphy::PuschParams::setStatPrms))
        .def("set_chest_factory_settings_filename", &pycuphy::PuschParams::setChestFactorySettingsFilename);

    py::class_<pycuphy::PdschParams>(m, "PdschParams")
        .def(py::init<const py::object&>())
        .def("print_stat_prms", &pycuphy::PdschParams::printStatPrms)
        .def("set_dyn_prms", &pycuphy::PdschParams::setDynPrms);

    py::class_<pycuphy::PyTrtEngine>(m, "TrtEngine")
        .def(py::init<const std::string&,
                      const uint32_t,
                      const std::vector<std::string>&,
                      const std::vector<std::vector<int>>&,
                      const std::vector<cuphyDataType_t>&,
                      const std::vector<std::string>&,
                      const std::vector<std::vector<int>>&,
                      const std::vector<cuphyDataType_t>&,
                      uint64_t>())
        .def("run", &pycuphy::PyTrtEngine::run);

    // carrier configuration
    py::class_<cuphyCarrierPrms_t>(m, "CuphyCarrierPrms")
        .def(py::init<>())
        .def_readwrite("n_sc", &cuphyCarrierPrms_t::N_sc)
        .def_readwrite("n_fft", &cuphyCarrierPrms_t::N_FFT)
        .def_readwrite("n_bs_layer", &cuphyCarrierPrms_t::N_bsLayer)
        .def_readwrite("n_ue_layer", &cuphyCarrierPrms_t::N_ueLayer)
        .def_readwrite("id_slot", &cuphyCarrierPrms_t::id_slot)
        .def_readwrite("id_subframe", &cuphyCarrierPrms_t::id_subFrame)
        .def_readwrite("mu", &cuphyCarrierPrms_t::mu)
        .def_readwrite("cp_type", &cuphyCarrierPrms_t::cpType)
        .def_readwrite("f_c", &cuphyCarrierPrms_t::f_c)
        .def_readwrite("t_c", &cuphyCarrierPrms_t::T_c)
        .def_readwrite("f_samp", &cuphyCarrierPrms_t::f_samp)
        .def_readwrite("n_symbol_slot", &cuphyCarrierPrms_t::N_symbol_slot)
        .def_readwrite("k_const", &cuphyCarrierPrms_t::k_const)
        .def_readwrite("kappa_bits", &cuphyCarrierPrms_t::kappa_bits)
        .def_readwrite("ofdm_window_len", &cuphyCarrierPrms_t::ofdmWindowLen)
        .def_readwrite("rolloff_factor", &cuphyCarrierPrms_t::rolloffFactor)
        .def_readwrite("n_samp_slot", &cuphyCarrierPrms_t::N_samp_slot)

        // below are PRACH parameters
        .def_readwrite("n_u_mu", &cuphyCarrierPrms_t::N_u_mu)
        .def_readwrite("start_ra_sym", &cuphyCarrierPrms_t::startRaSym)
        .def_readwrite("delta_f_ra", &cuphyCarrierPrms_t::delta_f_RA)
        .def_readwrite("n_cp_ra", &cuphyCarrierPrms_t::N_CP_RA)
        .def_readwrite("k", &cuphyCarrierPrms_t::K)
        .def_readwrite("k1", &cuphyCarrierPrms_t::k1)
        .def_readwrite("k_bar", &cuphyCarrierPrms_t::kBar)
        .def_readwrite("n_u", &cuphyCarrierPrms_t::N_u)
        .def_readwrite("l_ra", &cuphyCarrierPrms_t::L_RA)
        .def_readwrite("n_slot_ra_sel", &cuphyCarrierPrms_t::n_slot_RA_sel)
        .def_readwrite("n_rep", &cuphyCarrierPrms_t::N_rep);

    // OFDM modulation
    py::class_<pycuphy::OfdmModulateWrapper<float, cuComplex>>(m, "OfdmModulate")
        .def(py::init<cuphyCarrierPrms_t*, uintptr_t, uintptr_t>(),
            py::arg("cuphy_carrier_prms"), py::arg("freq_data_in_gpu"), py::arg("stream_handle"))
        .def(py::init<cuphyCarrierPrms_t*, py::array_t<std::complex<float>>, uintptr_t>(),
            py::arg("cuphy_carrier_prms"), py::arg("freq_data_in_cpu"), py::arg("stream_handle"))
        .def("run", &pycuphy::OfdmModulateWrapper<float, cuComplex>::run,
            py::arg("freq_data_in_cpu") = py::array_t<std::complex<float>>(),
            py::arg("enable_swap_tx_rx") = 0)
        .def("print_time_sample", &pycuphy::OfdmModulateWrapper<float, cuComplex>::printTimeSample, py::arg("print_length") = 10)
        .def("get_time_data_out", &pycuphy::OfdmModulateWrapper<float, cuComplex>::getTimeDataOut, py::return_value_policy::reference)
        .def("get_time_data_length", &pycuphy::OfdmModulateWrapper<float, cuComplex>::getTimeDataLen)
        .def("get_each_symbol_len_with_cp", &pycuphy::OfdmModulateWrapper<float, cuComplex>::getEachSymbolLenWithCP);

    // OFDM demodulation
    py::class_<pycuphy::OfdmDeModulateWrapper<float, cuComplex>>(m, "OfdmDeModulate")
        .def(py::init<cuphyCarrierPrms_t*, uintptr_t, uintptr_t, bool, bool, uintptr_t>(),
            py::arg("cuphy_carrier_prms"), py::arg("time_data_in_gpu"), py::arg("freq_data_out_gpu"), py::arg("prach") = 0, py::arg("per_ant_samp") = false, py::arg("stream_handle"))
        .def(py::init<cuphyCarrierPrms_t*, uintptr_t, py::array_t<std::complex<float>>, bool, bool, uintptr_t>(),
            py::arg("cuphy_carrier_prms"), py::arg("time_data_in_gpu"), py::arg("freq_data_out_cpu"), py::arg("prach") = 0, py::arg("per_ant_samp") = false, py::arg("stream_handle"))
        .def("run", &pycuphy::OfdmDeModulateWrapper<float, cuComplex>::run,
            py::arg("freq_data_out_cpu") = py::array_t<std::complex<float>>(),
            py::arg("enable_swap_tx_rx") = 0)
        .def("print_freq_sample", &pycuphy::OfdmDeModulateWrapper<float, cuComplex>::printFreqSample, py::arg("print_length") = 10)
        .def("get_freq_data_out", &pycuphy::OfdmDeModulateWrapper<float, cuComplex>::getFreqDataOut, py::return_value_policy::reference);

    // TDL channel configuration
    py::class_<tdlConfig_t>(m, "TdlConfig")
        .def(py::init<>())
        .def_readwrite("use_simplified_pdp", &tdlConfig_t::useSimplifiedPdp)
        .def_readwrite("delay_profile", &tdlConfig_t::delayProfile)
        .def_readwrite("delay_spread", &tdlConfig_t::delaySpread)
        .def_readwrite("max_doppler_shift", &tdlConfig_t::maxDopplerShift)
        .def_readwrite("f_samp", &tdlConfig_t::f_samp)
        .def_readwrite("n_cell", &tdlConfig_t::nCell)
        .def_readwrite("n_ue", &tdlConfig_t::nUe)
        .def_readwrite("n_bs_ant", &tdlConfig_t::nBsAnt)
        .def_readwrite("n_ue_ant", &tdlConfig_t::nUeAnt)
        .def_readwrite("f_batch", &tdlConfig_t::fBatch)
        .def_readwrite("n_path", &tdlConfig_t::numPath)
        .def_readwrite("cfo_hz", &tdlConfig_t::cfoHz)
        .def_readwrite("delay", &tdlConfig_t::delay)
        .def_readwrite("signal_length_per_ant", &tdlConfig_t::sigLenPerAnt)
        .def_readwrite("n_sc", &tdlConfig_t::N_sc)
        .def_readwrite("n_sc_prbg", &tdlConfig_t::N_sc_Prbg)
        .def_readwrite("sc_spacing_hz", &tdlConfig_t::scSpacingHz)
        .def_readwrite("freq_convert_type", &tdlConfig_t::freqConvertType)
        .def_readwrite("sc_sampling", &tdlConfig_t::scSampling)
        .def_readwrite("run_mode", &tdlConfig_t::runMode)
        .def_readwrite("proc_sig_freq", &tdlConfig_t::procSigFreq)
        .def_readwrite("save_ant_pair_sample", &tdlConfig_t::saveAntPairSample)
        .def_readwrite("batch_len", &tdlConfig_t::batchLen)
        .def_readwrite("tx_signal_in", &tdlConfig_t::txSigIn);

    py::class_<pycuphy::TdlChanWrapper<float, cuComplex>>(m, "TdlChan")
        .def(py::init<tdlConfig_t*, uint16_t, uintptr_t>(),
            py::arg("tdl_cfg"), py::arg("rand_seed"), py::arg("stream_handle"))
        .def("run", &pycuphy::TdlChanWrapper<float, cuComplex>::run,
            py::arg("tx_signal_in"), py::arg("ref_time0") = 0.0f, py::arg("enable_swap_tx_rx") = 0, py::arg("tx_column_major_ind") = 0,
            "Run channel with CuPy/GPU array input")
        .def("get_rx_signal_out_array", &pycuphy::TdlChanWrapper<float, cuComplex>::getRxSignalOutArray,
            py::arg("enable_swap_tx_rx") = 0,
            "Get output signal as a CUDA array compatible with CuPy")
        .def("reset", &pycuphy::TdlChanWrapper<float, cuComplex>::reset)
        .def("get_time_chan", &pycuphy::TdlChanWrapper<float, cuComplex>::getTimeChan)
        .def("get_freq_chan_sc", &pycuphy::TdlChanWrapper<float, cuComplex>::getFreqChanSc)
        .def("get_freq_chan_prbg", &pycuphy::TdlChanWrapper<float, cuComplex>::getFreqChanPrbg)
        .def("get_rx_signal_out", &pycuphy::TdlChanWrapper<float, cuComplex>::getRxSigOut)
        .def("get_rx_time_ant_pair_signal_out", &pycuphy::TdlChanWrapper<float, cuComplex>::getRxTimeAntPairSigOut)
        .def("get_time_chan_size", &pycuphy::TdlChanWrapper<float, cuComplex>::getTimeChanSize)
        .def("get_freq_chan_sc_per_link_size", &pycuphy::TdlChanWrapper<float, cuComplex>::getFreqChanScPerLinkSize)
        .def("get_freq_chan_prbg_size", &pycuphy::TdlChanWrapper<float, cuComplex>::getFreqChanPrbgSize)
        .def("print_time_chan", &pycuphy::TdlChanWrapper<float, cuComplex>::printTimeChan,
            py::arg("cid") = 0, py::arg("uid") = 0, py::arg("print_length") = 10)
        .def("print_freq_sc_chan", &pycuphy::TdlChanWrapper<float, cuComplex>::printFreqScChan,
            py::arg("cid") = 0, py::arg("uid") = 0, py::arg("print_length") = 10)
        .def("print_freq_prbg_chan", &pycuphy::TdlChanWrapper<float, cuComplex>::printFreqPrbgChan,
            py::arg("cid") = 0, py::arg("uid") = 0, py::arg("print_length") = 10)
        .def("print_signal", &pycuphy::TdlChanWrapper<float, cuComplex>::printSig,
            py::arg("cid") = 0, py::arg("uid") = 0, py::arg("print_length") = 10)
        .def("print_gpu_memory_usage_mb", &pycuphy::TdlChanWrapper<float, cuComplex>::printGpuMemUseMB)
        .def("dump_cir", &pycuphy::TdlChanWrapper<float, cuComplex>::dumpCir)
        .def("dump_cfr_prbg", &pycuphy::TdlChanWrapper<float, cuComplex>::dumpCfrPrbg)
        .def("dump_cfr_sc", &pycuphy::TdlChanWrapper<float, cuComplex>::dumpCfrSc)
        .def("save_tdl_chan_to_h5_file", &pycuphy::TdlChanWrapper<float, cuComplex>::saveTdlChanToH5File,
            py::arg("pad_file_name_ending") = "");

    // CDL channel configuration
    py::class_<cdlConfig_t>(m, "CdlConfig")
        .def(py::init<>())
        .def_readwrite("delay_profile", &cdlConfig_t::delayProfile)
        .def_readwrite("delay_spread", &cdlConfig_t::delaySpread)
        .def_readwrite("max_doppler_shift", &cdlConfig_t::maxDopplerShift)
        .def_readwrite("f_samp", &cdlConfig_t::f_samp)
        .def_readwrite("n_cell", &cdlConfig_t::nCell)
        .def_readwrite("n_ue", &cdlConfig_t::nUe)
        .def_readwrite("bs_ant_size", &cdlConfig_t::bsAntSize)
        .def_readwrite("bs_ant_spacing", &cdlConfig_t::bsAntSpacing)
        .def_readwrite("bs_ant_polar_angles", &cdlConfig_t::bsAntPolarAngles)
        .def_readwrite("bs_ant_pattern", &cdlConfig_t::bsAntPattern)
        .def_readwrite("ue_ant_size", &cdlConfig_t::ueAntSize)
        .def_readwrite("ue_ant_spacing", &cdlConfig_t::ueAntSpacing)
        .def_readwrite("ue_ant_polar_angles", &cdlConfig_t::ueAntPolarAngles)
        .def_readwrite("ue_ant_pattern", &cdlConfig_t::ueAntPattern)
        .def_readwrite("v_direction", &cdlConfig_t::vDirection)
        .def_readwrite("f_batch", &cdlConfig_t::fBatch)
        .def_readwrite("n_ray", &cdlConfig_t::numRay)
        .def_readwrite("cfo_hz", &cdlConfig_t::cfoHz)
        .def_readwrite("delay", &cdlConfig_t::delay)
        .def_readwrite("signal_length_per_ant", &cdlConfig_t::sigLenPerAnt)
        .def_readwrite("n_sc", &cdlConfig_t::N_sc)
        .def_readwrite("n_sc_prbg", &cdlConfig_t::N_sc_Prbg)
        .def_readwrite("sc_spacing_hz", &cdlConfig_t::scSpacingHz)
        .def_readwrite("freq_convert_type", &cdlConfig_t::freqConvertType)
        .def_readwrite("sc_sampling", &cdlConfig_t::scSampling)
        .def_readwrite("run_mode", &cdlConfig_t::runMode)
        .def_readwrite("proc_sig_freq", &cdlConfig_t::procSigFreq)
        .def_readwrite("save_ant_pair_sample", &cdlConfig_t::saveAntPairSample)
        .def_readwrite("batch_len", &cdlConfig_t::batchLen)
        .def_readwrite("tx_signal_in", &cdlConfig_t::txSigIn);

    py::class_<pycuphy::CdlChanWrapper<float, cuComplex>>(m, "CdlChan")
        .def(py::init<cdlConfig_t*, uint16_t, uintptr_t>(),
            py::arg("cdl_cfg"), py::arg("rand_seed"), py::arg("stream_handle"))
        .def("run", &pycuphy::CdlChanWrapper<float, cuComplex>::run,
            py::arg("tx_signal_in"), py::arg("ref_time0") = 0.0f, py::arg("enable_swap_tx_rx") = 0, py::arg("tx_column_major_ind") = 0,
            "Run channel with CuPy/GPU array input")
        .def("get_rx_signal_out_array", &pycuphy::CdlChanWrapper<float, cuComplex>::getRxSignalOutArray,
            py::arg("enable_swap_tx_rx") = 0,
            "Get output signal as a CUDA array compatible with CuPy")
        .def("reset", &pycuphy::CdlChanWrapper<float, cuComplex>::reset)
        .def("get_time_chan", &pycuphy::CdlChanWrapper<float, cuComplex>::getTimeChan)
        .def("get_freq_chan_sc", &pycuphy::CdlChanWrapper<float, cuComplex>::getFreqChanSc)
        .def("get_freq_chan_prbg", &pycuphy::CdlChanWrapper<float, cuComplex>::getFreqChanPrbg)
        .def("get_rx_signal_out", &pycuphy::CdlChanWrapper<float, cuComplex>::getRxSigOut)
        .def("get_rx_time_ant_pair_signal_out", &pycuphy::CdlChanWrapper<float, cuComplex>::getRxTimeAntPairSigOut)
        .def("get_time_chan_size", &pycuphy::CdlChanWrapper<float, cuComplex>::getTimeChanSize)
        .def("get_freq_chan_sc_per_link_size", &pycuphy::CdlChanWrapper<float, cuComplex>::getFreqChanScPerLinkSize)
        .def("get_freq_chan_prbg_size", &pycuphy::CdlChanWrapper<float, cuComplex>::getFreqChanPrbgSize)
        .def("print_time_chan", &pycuphy::CdlChanWrapper<float, cuComplex>::printTimeChan,
            py::arg("cid") = 0, py::arg("uid") = 0, py::arg("print_length") = 10)
        .def("print_freq_sc_chan", &pycuphy::CdlChanWrapper<float, cuComplex>::printFreqScChan,
            py::arg("cid") = 0, py::arg("uid") = 0, py::arg("print_length") = 10)
        .def("print_freq_prbg_chan", &pycuphy::CdlChanWrapper<float, cuComplex>::printFreqPrbgChan,
            py::arg("cid") = 0, py::arg("uid") = 0, py::arg("print_length") = 10)
        .def("print_signal", &pycuphy::CdlChanWrapper<float, cuComplex>::printSig,
            py::arg("cid") = 0, py::arg("uid") = 0, py::arg("print_length") = 10)
        .def("print_gpu_memory_usage_mb", &pycuphy::CdlChanWrapper<float, cuComplex>::printGpuMemUseMB)
        .def("dump_cir", &pycuphy::CdlChanWrapper<float, cuComplex>::dumpCir)
        .def("dump_cfr_prbg", &pycuphy::CdlChanWrapper<float, cuComplex>::dumpCfrPrbg)
        .def("dump_cfr_sc", &pycuphy::CdlChanWrapper<float, cuComplex>::dumpCfrSc)
        .def("save_cdl_chan_to_h5_file", &pycuphy::CdlChanWrapper<float, cuComplex>::saveCdlChanToH5File,
            py::arg("pad_file_name_ending") = "");

    // Channel Models API bindings
    // Bind Scenario enum
    py::enum_<Scenario>(m, "Scenario", "Deployment scenario types for channel modeling")
        .value("UMa", Scenario::UMa, "Urban Macro scenario")
        .value("UMi", Scenario::UMi, "Urban Micro scenario") 
        .value("RMa", Scenario::RMa, "Rural Macro scenario")
        .value("Indoor", Scenario::Indoor, "Indoor scenario (TODO: Not supported yet)")
        .value("InF", Scenario::InF, "Indoor Factory scenario (TODO: Not supported yet)")
        .value("SMa", Scenario::SMa, "Suburban Macro scenario (TODO: Not supported yet)")
        .export_values();

    // Bind SensingTargetType enum (for ISAC)
    py::enum_<SensingTargetType>(m, "SensingTargetType", "Sensing target types for ISAC per 3GPP TR 38.901 Section 7.9")
        .value("UAV", SensingTargetType::UAV, "UAV sensing target (Table 7.9.1-1)")
        .value("AUTOMOTIVE", SensingTargetType::AUTOMOTIVE, "Automotive sensing target (Table 7.9.1-2)")
        .value("HUMAN", SensingTargetType::HUMAN, "Human sensing target (Table 7.9.1-3)")
        .value("AGV", SensingTargetType::AGV, "Automated Guided Vehicle sensing target (Table 7.9.1-4)")
        .value("HAZARD", SensingTargetType::HAZARD, "Hazards on roads/railways sensing target (Table 7.9.1-5)")
        .export_values();

    // Bind UeType enum
    py::enum_<UeType>(m, "UeType", "UE (User Equipment) device types per 3GPP categorization")
        .value("TERRESTRIAL", UeType::TERRESTRIAL, "Traditional handheld/fixed UE (smartphones, tablets, CPE)")
        .value("VEHICLE", UeType::VEHICLE, "Vehicular UE for V2X communication (cars, trucks, buses)")
        .value("AERIAL", UeType::AERIAL, "Aerial UE (drones, UAVs for communication)")
        .value("AGV", UeType::AGV, "Automated Guided Vehicle (industrial robots)")
        .value("RSU", UeType::RSU, "Road Side Unit (fixed V2X infrastructure)")
        .export_values();

    // Bind Coordinate struct
    py::class_<Coordinate>(m, "Coordinate", "3D coordinate structure for global coordinate system")
        .def(py::init<>(), "Default constructor")
        .def(py::init<float, float, float>(), 
             "Initialize with x, y, z coordinates",
             py::arg("x") = 0.0f, py::arg("y") = 0.0f, py::arg("z") = 0.0f)
        .def_readwrite("x", &Coordinate::x, "x-coordinate in global coordinate system")
        .def_readwrite("y", &Coordinate::y, "y-coordinate in global coordinate system")
        .def_readwrite("z", &Coordinate::z, "z-coordinate in global coordinate system")
        .def("__repr__", [](const Coordinate& c) {
            return "Coordinate(x=" + std::to_string(c.x) + 
                   ", y=" + std::to_string(c.y) + 
                   ", z=" + std::to_string(c.z) + ")";
        });

    // Bind SpstParam struct (SPST = Sub-Pixel Scattering Point for ISAC)
    py::class_<SpstParam>(m, "SpstParam", "SPST (Scattering Point) parameter configuration for ISAC sensing targets")
        .def(py::init<>(), "Default constructor")
        .def(py::init<uint32_t, const Coordinate&, float, float, float>(),
             "Initialize with SPST parameters",
             py::arg("spst_id"), py::arg("loc_in_st_lcs"), 
             py::arg("rcs_sigma_m_dbsm") = -12.81f, py::arg("rcs_sigma_d_dbsm") = 1.0f, 
             py::arg("rcs_sigma_s_db") = 3.74f)
        .def_readwrite("spst_id", &SpstParam::spst_id, "SPST ID within the ST (0-indexed)")
        .def_readwrite("loc_in_st_lcs", &SpstParam::loc_in_st_lcs, 
                      "Location of SPST in ST's local coordinate system")
        .def_readwrite("rcs_sigma_m_dbsm", &SpstParam::rcs_sigma_m_dbsm, 
                      "Mean monostatic RCS sigma_M in dBsm")
        .def_readwrite("rcs_sigma_d_dbsm", &SpstParam::rcs_sigma_d_dbsm, 
                      "Mean monostatic RCS sigma_D in dBsm")
        .def_readwrite("rcs_sigma_s_db", &SpstParam::rcs_sigma_s_db, 
                      "Standard deviation sigma_s_dB in dB")
        .def_readwrite("enable_forward_scattering", &SpstParam::enable_forward_scattering,
                      "Control forward scattering effect: 0=disable, 1=enable");

    // Bind StParam struct (Sensing Target for ISAC)
    py::class_<StParam>(m, "StParam", "Sensing Target (ST) parameter configuration for ISAC")
        .def(py::init<>(), "Default constructor")
        .def(py::init<uint32_t, uint8_t, const Coordinate&>(),
             "Initialize with basic ST parameters",
             py::arg("sid"), py::arg("outdoor_ind") = 1, py::arg("loc") = Coordinate())
        .def(py::init<uint32_t, SensingTargetType, uint8_t, const Coordinate&, uint8_t>(),
             "Initialize with target type and RCS model",
             py::arg("sid"), py::arg("target_type"), py::arg("outdoor_ind"), 
             py::arg("loc"), py::arg("rcs_model") = 1)
        .def_readwrite("sid", &StParam::sid, "Global ST ID (Sensing Target ID)")
        .def_readwrite("target_type", &StParam::target_type, "Type of sensing target")
        .def_readwrite("outdoor_ind", &StParam::outdoor_ind, "0: indoor, 1: outdoor")
        .def_readwrite("loc", &StParam::loc, "ST location in GCS")
        .def_readwrite("rcs_model", &StParam::rcs_model, 
                      "RCS model: 1=deterministic monostatic, 2=angular dependent")
        .def_property("n_spst",
            [](const StParam& self) { return self.n_spst; },
            [](StParam& self, uint32_t value) {
                self.n_spst = value;
                try {
                    self.validateSpstConsistency(true);
                } catch (const std::invalid_argument& e) {
                    throw py::value_error(e.what());
                }
            },
            "Number of scattering points (SPSTs)")
        .def_property("spst_configs",
            [](const StParam& self) { return self.spst_configs; },
            [](StParam& self, const std::vector<SpstParam>& value) {
                self.spst_configs = value;
                try {
                    self.validateSpstConsistency(false);
                } catch (const std::invalid_argument& e) {
                    throw py::value_error(e.what());
                }
            },
            "List of SPST parameter configurations")
        .def_property("velocity",
            [](const StParam& self) { return pycuphy::carray_to_pylist(self.velocity); },
            [](StParam& self, const py::object& obj) {
                pycuphy::object_to_fixed_carray(obj, self.velocity);
            },
            "Velocity vector [vx, vy, vz] in m/s")
        .def_property("target_orientation",
            [](const StParam& self) { return pycuphy::carray_to_pylist(self.orientation); },
            [](StParam& self, const py::object& obj) {
                pycuphy::object_to_fixed_carray(obj, self.orientation);
            },
            "Target orientation [azimuth, elevation] in degrees")
        .def_property("physical_size",
            [](const StParam& self) { return pycuphy::carray_to_pylist(self.physical_size); },
            [](StParam& self, const py::object& obj) {
                pycuphy::object_to_fixed_carray(obj, self.physical_size);
            },
            "Physical dimensions [length, width, height] in meters");

        // Bind AntPanelConfig struct with numpy array support
    py::class_<AntPanelConfig>(m, "AntPanelConfig", "Antenna panel configuration parameters")
        .def(py::init<>(), "Default constructor")
        .def(py::init<uint16_t, uint8_t>(), 
             "Initialize with number of antennas and antenna model",
             py::arg("n_ant"), py::arg("ant_model") = 1)
        .def(py::init([](uint16_t n_ant, const std::vector<uint16_t>& ant_size, const std::vector<float>& ant_spacing, const std::vector<float>& ant_polar_angles, uint8_t ant_model) {
                 auto result = std::make_unique<AntPanelConfig>();
                 result->nAnt = n_ant;
                 result->antModel = ant_model;
                 pycuphy::vector_to_carray(ant_size, result->antSize);
                 pycuphy::vector_to_carray(ant_spacing, result->antSpacing);
                 pycuphy::vector_to_carray(ant_polar_angles, result->antPolarAngles);
                 return result;
             }),
             "Initialize with antenna parameters for models 0 and 1",
             py::arg("n_ant"), py::arg("ant_size"), py::arg("ant_spacing"), py::arg("ant_polar_angles"), py::arg("ant_model") = 1)
        .def(py::init([](uint16_t n_ant, const std::vector<uint16_t>& ant_size, const std::vector<float>& ant_spacing, const std::vector<float>& ant_theta, const std::vector<float>& ant_phi, const std::vector<float>& ant_polar_angles, uint8_t ant_model) {
                 auto result = std::make_unique<AntPanelConfig>();
                 result->nAnt = n_ant;
                 result->antModel = ant_model;
                 pycuphy::vector_to_carray(ant_size, result->antSize);
                 pycuphy::vector_to_carray(ant_spacing, result->antSpacing);
                 pycuphy::vector_to_carray(ant_theta, result->antTheta);
                 pycuphy::vector_to_carray(ant_phi, result->antPhi);
                 pycuphy::vector_to_carray(ant_polar_angles, result->antPolarAngles);
                 return result;
             }),
             "Initialize with full antenna parameters including direct patterns",
             py::arg("n_ant"), py::arg("ant_size"), py::arg("ant_spacing"), py::arg("ant_theta"), py::arg("ant_phi"), py::arg("ant_polar_angles"), py::arg("ant_model") = 2)
        .def_readwrite("n_ant", &AntPanelConfig::nAnt, 
                      "Number of antennas in the array (nAnt = M_g * N_g * M * N * P)")
        .def_property("ant_size", 
            [](const AntPanelConfig& self) { 
                py::list result;
                for (size_t i = 0; i < 5; ++i) {
                    result.append(self.antSize[i]);
                }
                return result;
            },
            [](AntPanelConfig& self, const py::list& list) { 
                if (list.size() != 5) {
                    throw std::invalid_argument("ant_size must have exactly 5 elements");
                }
                for (size_t i = 0; i < 5; ++i) {
                    self.antSize[i] = py::cast<uint16_t>(list[i]);
                }
            },
            "Dimensions of the antenna array [M_g, N_g, M, N, P]")
        .def_property("ant_spacing",
            [](const AntPanelConfig& self) { 
                py::list result;
                for (size_t i = 0; i < 4; ++i) {
                    result.append(self.antSpacing[i]);
                }
                return result;
            },
            [](AntPanelConfig& self, const py::list& list) {
                if (list.size() != 4) {
                    throw std::invalid_argument("ant_spacing must have exactly 4 elements");
                }
                for (size_t i = 0; i < 4; ++i) {
                    self.antSpacing[i] = py::cast<float>(list[i]);
                }
            },
            "Spacing between antennas in wavelengths [d_g_h, d_g_v, d_h, d_v]")
        .def_property("ant_theta",
            [](const AntPanelConfig& self) { 
                py::list result;
                for (size_t i = 0; i < 181; ++i) {
                    result.append(self.antTheta[i]);
                }
                return result;
            },
            [](AntPanelConfig& self, const py::list& list) {
                if (list.size() != 181) {
                    throw std::invalid_argument("ant_theta must have exactly 181 elements (0-180 degrees)");
                }
                for (size_t i = 0; i < 181; ++i) {
                    self.antTheta[i] = py::cast<float>(list[i]);
                }
            },
            "Antenna pattern A(theta, phi=0) in dB, size 181 (0-180 degrees)")
        .def_property("ant_phi",
            [](const AntPanelConfig& self) { 
                py::list result;
                for (size_t i = 0; i < 360; ++i) {
                    result.append(self.antPhi[i]);
                }
                return result;
            },
            [](AntPanelConfig& self, const py::list& list) {
                if (list.size() != 360) {
                    throw std::invalid_argument("ant_phi must have exactly 360 elements (0-360 degrees)");
                }
                for (size_t i = 0; i < 360; ++i) {
                    self.antPhi[i] = py::cast<float>(list[i]);
                }
            },
            "Antenna pattern A(theta=90, phi) in dB, size 360 (0-360 degrees)")
        .def_property("ant_polar_angles",
            [](const AntPanelConfig& self) { 
                py::list result;
                for (size_t i = 0; i < 2; ++i) {
                    result.append(self.antPolarAngles[i]);
                }
                return result;
            },
            [](AntPanelConfig& self, const py::list& list) {
                if (list.size() != 2) {
                    throw std::invalid_argument("ant_polar_angles must have exactly 2 elements [roll_angle_first_polz, roll_angle_second_polz]");
                }
                for (size_t i = 0; i < 2; ++i) {
                    self.antPolarAngles[i] = py::cast<float>(list[i]);
                }
            },
            "Antenna polarization angles [roll_angle_first_polz, roll_angle_second_polz]")
        .def_readwrite("ant_model", &AntPanelConfig::antModel,
                      "Antenna model type: 0=isotropic, 1=directional, 2=direct pattern");

    // Bind UtParamCfg struct (public API) with numpy array support
    py::class_<UtParamCfg>(m, "UtParamCfg", "User Terminal parameter configuration")
        .def(py::init<>(), "Default constructor")
        .def(py::init<uint32_t, const Coordinate&, uint8_t, uint32_t, UeType>(),
             "Initialize with basic parameters",
             py::arg("uid"), py::arg("loc"), py::arg("outdoor_ind") = 0, py::arg("ant_panel_idx") = 0, py::arg("ue_type") = UeType::TERRESTRIAL)
        .def(py::init([](uint32_t uid, const Coordinate& loc, uint8_t outdoor_ind, uint32_t ant_panel_idx, const std::vector<float>& ant_panel_orientation, const std::vector<float>& velocity, UeType ue_type) {
                 auto result = std::make_unique<UtParamCfg>();
                 result->uid = uid;
                 result->loc = loc;
                 result->outdoor_ind = outdoor_ind;
                 result->ue_type = ue_type;
                 result->antPanelIdx = ant_panel_idx;
                 pycuphy::vector_to_carray(ant_panel_orientation, result->antPanelOrientation);
                 pycuphy::vector_to_carray(velocity, result->velocity);
                 return result;
             }),
             "Initialize with full parameters including orientation and velocity",
             py::arg("uid"), py::arg("loc"), py::arg("outdoor_ind"), py::arg("ant_panel_idx"), py::arg("ant_panel_orientation"), py::arg("velocity"), py::arg("ue_type") = UeType::TERRESTRIAL)
        .def_readwrite("uid", &UtParamCfg::uid, "Global UE ID")
        .def_readwrite("loc", &UtParamCfg::loc, "UE location")
        .def_readwrite("outdoor_ind", &UtParamCfg::outdoor_ind, "Outdoor indicator: 0=indoor, 1=outdoor")
        .def_readwrite("ue_type", &UtParamCfg::ue_type, "UE type: TERRESTRIAL, VEHICLE, AERIAL, AGV, RSU")
        .def_readwrite("ant_panel_idx", &UtParamCfg::antPanelIdx, "Antenna panel configuration index")
        .def_property("ant_panel_orientation",
            [](const UtParamCfg& self) { return pycuphy::carray_to_pylist(self.antPanelOrientation); },
            [](UtParamCfg& self, const py::object& obj) {
                pycuphy::object_to_fixed_carray(obj, self.antPanelOrientation);
            },
            "Antenna panel orientation in GCS [theta, phi, slant_offset]")
        .def_property("velocity",
            [](const UtParamCfg& self) { return pycuphy::carray_to_pylist(self.velocity); },
            [](UtParamCfg& self, const py::object& obj) {
                pycuphy::object_to_fixed_carray(obj, self.velocity);
            },
            "Velocity vector [vx, vy, vz] in m/s, vz=0 per 3GPP spec")
        .def_readwrite("monostatic_ind", &UtParamCfg::monostatic_ind, 
                      "0: not a monostatic sensing receiver, 1: monostatic sensing receiver")
        .def_readwrite("same_antenna_panel_ind", &UtParamCfg::same_antenna_panel_ind,
                      "0: use second antenna panel for sensing, 1: use same antenna panel")
        .def_readwrite("second_ant_panel_idx", &UtParamCfg::second_ant_panel_idx,
                      "Second antenna panel index for sensing RX (when monostatic_ind=1)")
        .def_property("second_ant_panel_orientation",
            [](const UtParamCfg& self) { return pycuphy::carray_to_pylist(self.second_ant_panel_orientation); },
            [](UtParamCfg& self, const py::object& obj) {
                pycuphy::object_to_fixed_carray(obj, self.second_ant_panel_orientation);
            },
            "Second antenna panel orientation for sensing RX [theta, phi, slant_offset]");

    // Bind CellParam struct with numpy array support
    py::class_<CellParam>(m, "CellParam", "Cell/Base Station parameters")
        .def(py::init<>(), "Default constructor")
        .def(py::init<uint32_t, uint32_t, Coordinate, uint32_t>(),
             "Initialize with basic parameters",
             py::arg("cid"), py::arg("site_id"), py::arg("loc"), py::arg("ant_panel_idx"))
        .def(py::init([](uint32_t cid, uint32_t site_id, Coordinate loc, uint32_t ant_panel_idx, const std::vector<float>& ant_panel_orientation) {
                 auto result = std::make_unique<CellParam>();
                 result->cid = cid;
                 result->siteId = site_id;
                 result->loc = loc;
                 result->antPanelIdx = ant_panel_idx;
                 pycuphy::vector_to_carray(ant_panel_orientation, result->antPanelOrientation);
                 return result;
             }),
             "Initialize with full parameters including antenna orientation",
             py::arg("cid"), py::arg("site_id"), py::arg("loc"), py::arg("ant_panel_idx"), py::arg("ant_panel_orientation"))
        .def_readwrite("cid", &CellParam::cid, "Global cell ID")
        .def_readwrite("site_id", &CellParam::siteId, "Site ID for LSP access")
        .def_readwrite("loc", &CellParam::loc, "Cell location")
        .def_readwrite("ant_panel_idx", &CellParam::antPanelIdx, "Antenna panel configuration index")
        .def_property("ant_panel_orientation",
            [](const CellParam& self) { return pycuphy::carray_to_pylist(self.antPanelOrientation); },
            [](CellParam& self, const py::object& obj) {
                pycuphy::object_to_fixed_carray(obj, self.antPanelOrientation);
            },
            "Antenna panel orientation in GCS [theta, phi, slant_offset]")
        .def_readwrite("monostatic_ind", &CellParam::monostatic_ind,
                      "0: not monostatic, 1: monostatic (BS acts as both TX and RX for sensing)")
        .def_readwrite("second_ant_panel_idx", &CellParam::second_ant_panel_idx,
                      "Second antenna panel index for sensing RX (when monostatic_ind=1)")
        .def_property("second_ant_panel_orientation",
            [](const CellParam& self) { return pycuphy::carray_to_pylist(self.second_ant_panel_orientation); },
            [](CellParam& self, const py::object& obj) {
                pycuphy::object_to_fixed_carray(obj, self.second_ant_panel_orientation);
            },
            "Second antenna panel orientation for sensing RX [theta, phi, slant_offset]");

    // Bind SystemLevelConfig struct
    py::class_<SystemLevelConfig>(m, "SystemLevelConfig", "System-level configuration parameters")
        .def(py::init<>(), "Default constructor")
        .def(py::init<Scenario, uint32_t, uint8_t, uint32_t, float>(),
             "Initialize with basic parameters",
             py::arg("scenario"), py::arg("n_site"), py::arg("n_sector_per_site"), py::arg("n_ut"), py::arg("isd") = 1732.0f)
        .def(py::init([](Scenario scenario, uint32_t n_site, uint32_t n_sector_per_site, uint32_t n_ut, float isd, const std::vector<uint32_t>& ut_drop_cells, uint8_t ut_drop_option, const std::vector<float>& ut_cell_2d_dist, uint8_t optional_pl_ind, uint8_t o2i_building_penetr_loss_ind, uint8_t o2i_car_penetr_loss_ind, uint8_t enable_near_field_effect, uint8_t enable_non_stationarity, const std::vector<float>& force_los_prob, const std::vector<float>& force_ut_speed, float force_indoor_ratio, uint8_t disable_pl_shadowing, uint8_t disable_small_scale_fading, uint8_t enable_per_tti_lsp, uint8_t enable_propagation_delay) {
                 auto result = std::make_unique<SystemLevelConfig>();
                 result->scenario = scenario;
                 result->n_site = n_site;
                 result->n_sector_per_site = n_sector_per_site;
                 result->n_ut = n_ut;
                 result->isd = isd;
                 result->optional_pl_ind = optional_pl_ind;
                 result->o2i_building_penetr_loss_ind = o2i_building_penetr_loss_ind;
                 result->o2i_car_penetr_loss_ind = o2i_car_penetr_loss_ind;
                 result->enable_near_field_effect = enable_near_field_effect;
                 result->enable_non_stationarity = enable_non_stationarity;
                 pycuphy::vector_to_carray(force_los_prob, result->force_los_prob);
                 pycuphy::vector_to_carray(force_ut_speed, result->force_ut_speed);
                 result->force_indoor_ratio = force_indoor_ratio;
                 result->disable_pl_shadowing = disable_pl_shadowing;
                 result->disable_small_scale_fading = disable_small_scale_fading;
                 result->enable_per_tti_lsp = enable_per_tti_lsp;
                 result->enable_propagation_delay = enable_propagation_delay;
                 result->ut_drop_option = ut_drop_option;
                 pycuphy::vector_to_carray(ut_cell_2d_dist, result->ut_cell_2d_dist);
                 pycuphy::vector_to_carray(ut_drop_cells, result->ut_drop_cells);
                 result->n_ut_drop_cells = static_cast<uint32_t>(ut_drop_cells.size());
                 return result;
             }),
             "Initialize with full system-level parameters",
             py::arg("scenario"), py::arg("n_site"), py::arg("n_sector_per_site"), py::arg("n_ut"), py::arg("isd"),
             py::arg("ut_drop_cells") = std::vector<uint32_t>{}, py::arg("ut_drop_option") = 0, py::arg("ut_cell_2d_dist") = std::vector<float>{-1.0f, -1.0f},
             py::arg("optional_pl_ind") = 0, py::arg("o2i_building_penetr_loss_ind") = 1, py::arg("o2i_car_penetr_loss_ind") = 0, py::arg("enable_near_field_effect") = 0,
             py::arg("enable_non_stationarity") = 0, py::arg("force_los_prob") = std::vector<float>{-1.0f, -1.0f}, py::arg("force_ut_speed") = std::vector<float>{-1.0f, -1.0f},
             py::arg("force_indoor_ratio") = -1.0f, py::arg("disable_pl_shadowing") = 0, py::arg("disable_small_scale_fading") = 0, py::arg("enable_per_tti_lsp") = 1, py::arg("enable_propagation_delay") = 1)
        .def_readwrite("scenario", &SystemLevelConfig::scenario, "Deployment scenario")
        .def_readwrite("isd", &SystemLevelConfig::isd, "Inter-site distance in meters")
        .def_readwrite("n_site", &SystemLevelConfig::n_site, "Number of sites")
        .def_readwrite("n_sector_per_site", &SystemLevelConfig::n_sector_per_site, "Sectors per site")
        .def_readwrite("n_ut", &SystemLevelConfig::n_ut, "Total number of UTs")
        .def_readwrite("isac_type", &SystemLevelConfig::isac_type,
                      "ISAC type: 0=communication only, 1=monostatic sensing, 2=bistatic sensing")
        .def_readwrite("n_st", &SystemLevelConfig::n_st, "Total number of sensing targets (STs)")
        .def_property("st_horizontal_speed",
            [](const SystemLevelConfig& self) { return pycuphy::carray_to_pylist(self.st_horizontal_speed); },
            [](SystemLevelConfig& self, const py::object& obj) {
                pycuphy::object_to_fixed_carray(obj, self.st_horizontal_speed);
            },
            "Horizontal speed range [min, max] in m/s for ISAC sensing targets "
            "(default: [8.33, 8.33] = fixed 30 km/h)")
        .def_readwrite("st_vertical_velocity", &SystemLevelConfig::st_vertical_velocity,
                      "Vertical velocity in m/s for ISAC sensing targets (vz component, default: 0.0)")
        .def_property("st_distribution_option",
            [](const SystemLevelConfig& self) { return pycuphy::carray_to_pylist(self.st_distribution_option); },
            [](SystemLevelConfig& self, const py::object& obj) {
                pycuphy::object_to_fixed_carray(obj, self.st_distribution_option);
            },
            "ST distribution option [horizontal, vertical]: 0=Option A, 1=Option B, 2=Option C")
        .def_property("st_height",
            [](const SystemLevelConfig& self) { return pycuphy::carray_to_pylist(self.st_height); },
            [](SystemLevelConfig& self, const py::object& obj) {
                pycuphy::object_to_fixed_carray(obj, self.st_height);
            },
            "ST height range [min, max] in meters for vertical Option B (default: [100, 100])")
        // Backward-compatible alias (deprecated): st_fixed_height
        .def_property("st_fixed_height",
            [](const SystemLevelConfig& self) { return pycuphy::carray_to_pylist(self.st_height); },
            [](SystemLevelConfig& self, const py::object& obj) {
                pycuphy::object_to_fixed_carray(obj, self.st_height);
            },
            "Deprecated alias of st_height")
        .def_readwrite("st_minimum_distance", &SystemLevelConfig::st_minimum_distance,
                      "Minimum distance between STs in meters (0=auto based on physical size)")
        .def_readwrite("st_size_ind", &SystemLevelConfig::st_size_ind,
                      "ST size index: 0=small, 1=medium, 2=large")
        .def_readwrite("st_min_dist_from_tx_rx", &SystemLevelConfig::st_min_dist_from_tx_rx,
                      "Minimum 3D distance from ST to any STX/SRX (BS/UE) in meters (default: 10m)")
        .def_readwrite("st_target_type", &SystemLevelConfig::st_target_type,
                      "Default target type for auto-generated STs (SensingTargetType enum)")
        .def_readwrite("st_rcs_model", &SystemLevelConfig::st_rcs_model,
                      "RCS model for STs: 1=deterministic monostatic, 2=angular dependent")
        .def_readwrite("path_drop_threshold_db", &SystemLevelConfig::path_drop_threshold_db,
                      "Path power drop threshold in dB for ISAC ray/path pruning (default: 40 dB)")
        .def_readwrite("isac_disable_background", &SystemLevelConfig::isac_disable_background,
                      "ISAC calibration mode: 0=combine target with background, 1=target CIR only")
        .def_readwrite("isac_disable_target", &SystemLevelConfig::isac_disable_target,
                      "ISAC calibration mode: 0=include target CIR, 1=background CIR only")
        .def_readwrite("optional_pl_ind", &SystemLevelConfig::optional_pl_ind, 
                      "Pathloss equation: 0=standard, 1=optional")
        .def_readwrite("o2i_building_penetr_loss_ind", &SystemLevelConfig::o2i_building_penetr_loss_ind,
                      "Building penetration loss: 0=none, 1=low-loss, 2=high-loss")
        .def_readwrite("o2i_car_penetr_loss_ind", &SystemLevelConfig::o2i_car_penetr_loss_ind,
                      "Car penetration loss: 0=none, 1=basic, 2=metallized")
        .def_readwrite("enable_near_field_effect", &SystemLevelConfig::enable_near_field_effect,
                      "Near field effect: 0=disable, 1=enable")
        .def_readwrite("enable_non_stationarity", &SystemLevelConfig::enable_non_stationarity,
                      "Non-stationarity: 0=disable, 1=enable")
        .def_property("force_los_prob",
            [](const SystemLevelConfig& self) { return pycuphy::carray_to_pylist(self.force_los_prob); },
            [](SystemLevelConfig& self, const py::object& obj) {
                pycuphy::object_to_fixed_carray(obj, self.force_los_prob);
            },
            "Force LOS probability [outdoor, indoor], -1 for auto calculation")
        .def_property("force_ut_speed",
            [](const SystemLevelConfig& self) { return pycuphy::carray_to_pylist(self.force_ut_speed); },
            [](SystemLevelConfig& self, const py::object& obj) {
                pycuphy::object_to_fixed_carray(obj, self.force_ut_speed);
            },
            "Force UT speed [outdoor, indoor] in m/s, -1 for auto calculation")
        .def_readwrite("force_indoor_ratio", &SystemLevelConfig::force_indoor_ratio,
                      "Force indoor ratio, -1 for auto calculation")
        .def_readwrite("disable_pl_shadowing", &SystemLevelConfig::disable_pl_shadowing,
                      "Disable pathloss/shadowing: 0=calculate, 1=disable")
        .def_readwrite("disable_small_scale_fading", &SystemLevelConfig::disable_small_scale_fading,
                      "Disable small scale fading: 0=calculate, 1=disable (fast fading = 1)")
        .def_readwrite("enable_per_tti_lsp", &SystemLevelConfig::enable_per_tti_lsp,
                      "LSP per TTI: 0=disable, 1=update PL/shadowing, 2=update all")
        .def_readwrite("enable_propagation_delay", &SystemLevelConfig::enable_propagation_delay,
                      "Propagation delay in CIR: 0=disable, 1=enable")
        .def_readwrite("ut_drop_option", &SystemLevelConfig::ut_drop_option,
                      "UT drop control: 0=random across region, 1=same UTs per site, 2=same UTs per sector");

    // Bind LinkLevelConfig struct
    py::class_<LinkLevelConfig>(m, "LinkLevelConfig", "Link-level configuration parameters")
        .def(py::init<>(), "Default constructor")
        .def(py::init<int, char, float>(),
             "Initialize with basic parameters",
             py::arg("fast_fading_type"), py::arg("delay_profile") = 'A', py::arg("delay_spread") = 30.0f)
        .def(py::init([](int fast_fading_type, char delay_profile, float delay_spread, const std::vector<float>& velocity, int num_ray, float cfo_hz, float delay) {
                 auto result = std::make_unique<LinkLevelConfig>(fast_fading_type, delay_profile, delay_spread);
                 pycuphy::vector_to_carray(velocity, result->velocity);
                 result->num_ray = num_ray;
                 result->cfo_hz = cfo_hz;
                 result->delay = delay;
                 return result;
             }),
             "Initialize with full parameters including velocity",
             py::arg("fast_fading_type"), py::arg("delay_profile"), py::arg("delay_spread"), 
             py::arg("velocity"), py::arg("num_ray"), py::arg("cfo_hz"), py::arg("delay"))
        .def_readwrite("fast_fading_type", &LinkLevelConfig::fast_fading_type,
                      "Fast fading type: 0=AWGN, 1=TDL, 2=CDL")
        .def_readwrite("delay_profile", &LinkLevelConfig::delay_profile,
                      "Delay profile: 'A' to 'C'")
        .def_readwrite("delay_spread", &LinkLevelConfig::delay_spread,
                      "Delay spread in nanoseconds")
        .def_property("velocity",
            [](const LinkLevelConfig& self) { return pycuphy::carray_to_pylist(self.velocity); },
            [](LinkLevelConfig& self, const py::object& obj) {
                pycuphy::object_to_fixed_carray(obj, self.velocity);
            },
            "Velocity vector [vx, vy, vz] in m/s, vz=0 per 3GPP spec")
        .def_readwrite("num_ray", &LinkLevelConfig::num_ray,
                      "Number of rays per path (default: 48 for TDL, 20 for CDL)")
        .def_readwrite("cfo_hz", &LinkLevelConfig::cfo_hz,
                      "Carrier frequency offset in Hz")
        .def_readwrite("delay", &LinkLevelConfig::delay,
                      "Delay in seconds");

    // Bind SimConfig struct
    py::class_<SimConfig>(m, "SimConfig", "Test configuration parameters")
        .def(py::init<>(), "Default constructor")
        .def(py::init<float, float, int>(),
             "Initialize with basic parameters",
             py::arg("center_freq_hz"), py::arg("bandwidth_hz"), py::arg("run_mode") = 0)
        .def(py::init<float, float, float, int, int>(),
             "Initialize with detailed parameters",
             py::arg("center_freq_hz"), py::arg("bandwidth_hz"), py::arg("sc_spacing_hz"), 
             py::arg("fft_size"), py::arg("run_mode") = 0)
        .def(py::init([](int link_sim_ind, float center_freq_hz, float bandwidth_hz, float sc_spacing_hz, int fft_size, int n_prb, int n_prbg, int n_snapshot_per_slot, int run_mode, int internal_memory_mode, int freq_convert_type, int sc_sampling, int proc_sig_freq, int optional_cfr_dim, int cpu_only_mode) {
                 return std::make_unique<SimConfig>(link_sim_ind, center_freq_hz, bandwidth_hz, sc_spacing_hz, fft_size, n_prb, n_prbg, n_snapshot_per_slot, run_mode, internal_memory_mode, freq_convert_type, sc_sampling, nullptr, proc_sig_freq, optional_cfr_dim, cpu_only_mode);
             }),
             "Initialize with full parameters",
             py::arg("link_sim_ind"), py::arg("center_freq_hz"), py::arg("bandwidth_hz"), 
             py::arg("sc_spacing_hz"), py::arg("fft_size"), py::arg("n_prb"), py::arg("n_prbg"), 
             py::arg("n_snapshot_per_slot"), py::arg("run_mode"), py::arg("internal_memory_mode"), 
             py::arg("freq_convert_type"), py::arg("sc_sampling"), py::arg("proc_sig_freq") = 0, py::arg("optional_cfr_dim") = 0, py::arg("cpu_only_mode") = 0)
        .def_readwrite("link_sim_ind", &SimConfig::link_sim_ind,
                      "Link simulation indicator")
        .def_readwrite("center_freq_hz", &SimConfig::center_freq_hz,
                      "Center frequency in Hz")
        .def_readwrite("bandwidth_hz", &SimConfig::bandwidth_hz,
                      "Bandwidth in Hz")
        .def_readwrite("sc_spacing_hz", &SimConfig::sc_spacing_hz,
                      "Subcarrier spacing in Hz")
        .def_readwrite("fft_size", &SimConfig::fft_size,
                      "FFT size")
        .def_readwrite("n_prb", &SimConfig::n_prb,
                      "Number of PRBs")
        .def_readwrite("n_prbg", &SimConfig::n_prbg,
                      "Number of PRB groups")
        .def_readwrite("n_snapshot_per_slot", &SimConfig::n_snapshot_per_slot,
                      "Channel realizations per slot (1 or 14)")
        .def_readwrite("run_mode", &SimConfig::run_mode,
                      "Run mode: 0=CIR only, 1=CIR+CFR on PRBG, 2=CIR+CFR on PRB/SC")
        .def_readwrite("internal_memory_mode", &SimConfig::internal_memory_mode,
                      "Memory mode: 0=external, 1=internal")
        .def_readwrite("freq_convert_type", &SimConfig::freq_convert_type,
                      "Frequency conversion type for CFR on SC to PRBG")
        .def_readwrite("sc_sampling", &SimConfig::sc_sampling,
                      "Subcarrier sampling within PRBG")
        .def_readwrite("proc_sig_freq", &SimConfig::proc_sig_freq,
                      "Signal processing frequency indicator")
        .def_readwrite("optional_cfr_dim", &SimConfig::optional_cfr_dim,
                      "Optional CFR dimension")
        .def_readwrite("cpu_only_mode", &SimConfig::cpu_only_mode,
                      "CPU only mode: 0=GPU mode, 1=CPU only mode")
        .def_readwrite("h5_dump_level", &SimConfig::h5_dump_level,
                      "H5 dump level: 0=minimal (topology+CIR/CFR+config), 1=full (default)");

    // Bind ExternalConfig struct
    py::class_<ExternalConfig>(m, "ExternalConfig", "External configuration parameters")
        .def(py::init<>(), "Default constructor")
        .def(py::init([](const std::vector<CellParam>& cell_config, const std::vector<UtParamCfg>& ut_config, const std::vector<AntPanelConfig>& ant_panel_config) {
                 auto result = std::make_unique<ExternalConfig>();
                 result->cell_config = cell_config;
                 result->ut_config = ut_config;
                 result->ant_panel_config = ant_panel_config;
                 return result;
             }),
             "Initialize with parameters (without ST config)",
             py::arg("cell_config"), py::arg("ut_config"), py::arg("ant_panel_config"))
        .def_readwrite("cell_config", &ExternalConfig::cell_config,
                      "Cell configuration list")
        .def_readwrite("ut_config", &ExternalConfig::ut_config,
                      "UT configuration list")
        .def_readwrite("ant_panel_config", &ExternalConfig::ant_panel_config,
                      "Antenna panel configuration list")
        .def_readwrite("st_config", &ExternalConfig::st_config,
                      "Sensing target (ST) configuration list for ISAC");

    // Bind StatisChanModelWrapper class
    py::class_<pycuphy::StatisChanModelWrapper<float, cuComplex>>(m, "StatisChanModel",
                                                                 "Stochastic channel model wrapper class")
        .def(py::init<const SimConfig&, const SystemLevelConfig&, 
                     const LinkLevelConfig&, const ExternalConfig&, uint32_t, uintptr_t>(),
             "Initialize channel model with all configuration parameters",
             py::arg("sim_config"), py::arg("system_level_config"), 
             py::arg("link_level_config"), py::arg("external_config"),
             py::arg("rand_seed") = 0, py::arg("stream_handle") = 0)
        .def(py::init<const SimConfig&, const SystemLevelConfig&, uint32_t, uintptr_t>(),
             "Initialize channel model with minimal configuration parameters",
             py::arg("sim_config"), py::arg("system_level_config"),
             py::arg("rand_seed") = 0, py::arg("stream_handle") = 0)
        .def("reset", &pycuphy::StatisChanModelWrapper<float, cuComplex>::reset,
             "Reset the channel model state")
        .def("run", &pycuphy::StatisChanModelWrapper<float, cuComplex>::run,
             "Run system-level channel model simulation",
             py::arg("ref_time") = 0.0f,
             py::arg("continuous_fading") = 1,
             py::arg("active_cell") = py::none(),
             py::arg("active_ut") = py::none(),
             py::arg("ut_new_loc") = py::none(),
             py::arg("ut_new_velocity") = py::none(),
             py::arg("cir_coe") = py::none(),
             py::arg("cir_norm_delay") = py::none(),
             py::arg("cir_n_taps") = py::none(),
             py::arg("cfr_sc") = py::none(),
             py::arg("cfr_prbg") = py::none())
        .def("run_link_level", &pycuphy::StatisChanModelWrapper<float, cuComplex>::run_link_level,
             "Run link-level channel model simulation",
             py::arg("ref_time0") = 0.0f,
             py::arg("continuous_fading") = 1,
             py::arg("enable_swap_tx_rx") = 0,
             py::arg("tx_column_major_ind") = 0)
        .def("dump_topology_to_yaml", &pycuphy::StatisChanModelWrapper<float, cuComplex>::dump_topology_to_yaml,
             "Dump topology to YAML file",
             py::arg("filename"))
        .def("save_sls_chan_to_h5_file", &pycuphy::StatisChanModelWrapper<float, cuComplex>::saveSlsChanToH5File,
             "Save SLS channel data to H5 file for debugging",
             py::arg("filename_ending") = "")
        .def("dump_los_nlos_stats", &pycuphy::StatisChanModelWrapper<float, cuComplex>::dump_los_nlos_stats,
             "Dump LOS/NLOS statistics for all links",
             py::arg("lost_nlos_stats") = py::array_t<float>())
        .def("dump_pl_sf_stats", &pycuphy::StatisChanModelWrapper<float, cuComplex>::dump_pl_sf_stats,
             "Dump pathloss and shadowing statistics for links",
             py::arg("pl_sf"),
             py::arg("active_cell") = py::array_t<int>(),
             py::arg("active_ut") = py::array_t<int>())
        .def("dump_pl_sf_ant_gain_stats", &pycuphy::StatisChanModelWrapper<float, cuComplex>::dump_pl_sf_ant_gain_stats,
             "Dump pathloss, shadowing and antenna gain statistics",
             py::arg("pl_sf_ant_gain"),
             py::arg("active_cell") = py::array_t<int>(),
             py::arg("active_ut") = py::array_t<int>());

    py::class_<pycuphy::GauNoiseAdderWrapper<float, cuComplex>>(m, "GauNoiseAdder")
        .def(py::init<uint32_t, int, uintptr_t>(),
            py::arg("num_threads"), py::arg("rand_seed"), py::arg("stream_handle"))
        .def("add_noise", &pycuphy::GauNoiseAdderWrapper<float, cuComplex>::addNoise,
            py::arg("d_signal"), py::arg("signal_size"), py::arg("snr_db"),
            "Add Gaussian noise in-place on GPU");
 }
