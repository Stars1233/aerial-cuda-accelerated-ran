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

}
