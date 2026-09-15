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

// Standalone nanobind module entry point and channel-model registrations.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <vector>
#include <cstdint>
#include <complex>

#include "cuda_array_interface.hpp"
#include "gpu3gppchan_bindings.hpp"
#include "gpu3gppchanApi.hpp"
#include "gpu3gppchanDataset.hpp"

namespace nb = nanobind;

namespace gpu3gppchan_bindings {

template <typename T>
void declare_cuda_array(nb::module_ &m, const char *name) {
  nb::class_<cuda_array_t<T>>(m, name)
      .def("__init__", [](cuda_array_t<T> *self, nb::object obj) {
        if (!nb::hasattr(obj, "__cuda_array_interface__")) {
          throw nb::type_error(
              "Object must implement __cuda_array_interface__");
        }
        new (self) cuda_array_t<T>(obj);
      })
      .def("__init__", [](cuda_array_t<T> *self, intptr_t addr, const std::vector<size_t>& shape, const std::vector<size_t>& strides) {
        new (self) cuda_array_t<T>(addr, shape, strides);
      }, nb::arg("addr"), nb::arg("shape"), nb::arg("strides"))
      .def_prop_ro("shape", [](const cuda_array_t<T>& array) { return as_tuple(array.get_shape()); })
      .def_prop_ro("strides", [](const cuda_array_t<T>& array) { return as_tuple(array.get_strides()); })
      .def_prop_ro("size", &cuda_array_t<T>::get_size)
      .def_prop_ro("ndim", &cuda_array_t<T>::get_ndim)
      .def("is_readonly", &cuda_array_t<T>::is_readonly)
      .def("has_stride_info", &cuda_array_t<T>::has_stride_info)
      .def("is_c_contiguous", &cuda_array_t<T>::is_c_contiguous)
      .def("is_f_contiguous", &cuda_array_t<T>::is_f_contiguous)
      .def_prop_ro("__cuda_array_interface__", &cuda_array_t<T>::get_interface_dict);
}

// Convert a Python sequence to a fixed-size C array.
template<typename T, size_t N>
void vector_to_fixed_carray(const std::vector<T>& vec, T (&dest)[N]) {
    if (vec.size() != N) {
        throw std::invalid_argument(
            "Sequence length must exactly match destination array capacity");
    }

    std::copy(vec.begin(), vec.end(), dest);
}

// Convert a bounded Python sequence to a capacity-sized C array.
template<typename T, size_t N>
void vector_to_capacity_carray(const std::vector<T>& vec, T (&dest)[N]) {
    if (vec.size() > N) {
        throw std::invalid_argument("Sequence length exceeds destination array capacity");
    }

    std::copy(vec.begin(), vec.end(), dest);
    std::fill(dest + vec.size(), dest + N, T{});
}

// C fixed-size array -> Python list (small channel-model vectors; CIR/CFR buffers stay ndarray).
template<typename T, size_t N>
nb::list carray_to_pylist(const T (&src)[N]) {
    nb::list out;
    for (size_t i = 0; i < N; ++i) {
        out.append(src[i]);
    }
    return out;
}

// Python list, tuple, or ndarray -> fixed C array (getters return list; setters accept array-likes).
template<typename T, size_t N>
void object_to_fixed_carray(const nb::object& obj, T (&dest)[N]) {
    std::vector<T> vec = nb::cast<std::vector<T>>(obj);
    if (vec.size() != N) {
        throw nb::value_error(
            "Sequence length must exactly match destination array capacity");
    }
    std::copy(vec.begin(), vec.end(), dest);
}

[[nodiscard]] uintptr_t stream_handle_from_owner(const nb::object& stream) {
    uintptr_t handle{};

    if (nb::hasattr(stream, "handle")) {
        handle = nb::cast<uintptr_t>(stream.attr("handle"));
    } else if (nb::hasattr(stream, "ptr")) {
        handle = nb::cast<uintptr_t>(stream.attr("ptr"));
    } else {
        throw nb::type_error(
            "stream_handle must be an owner object exposing 'handle' or 'ptr'");
    }

    if (handle == 0 ||
        handle == reinterpret_cast<uintptr_t>(cudaStreamLegacy) ||
        handle == reinterpret_cast<uintptr_t>(cudaStreamPerThread)) {
        throw nb::type_error(
            "stream_handle must reference a non-default CUDA stream");
    }

    return handle;
}


}  // namespace gpu3gppchan_bindings


NB_MODULE(_gpu3gppchan, m) {
    m.attr("__doc__") = "Python bindings for gpu3gppchan (3GPP channel models)";

    gpu3gppchan_bindings::declare_cuda_array<int>(m, "CudaArrayInt");
    gpu3gppchan_bindings::declare_cuda_array<uint8_t>(m, "CudaArrayUint8");
    gpu3gppchan_bindings::declare_cuda_array<uint16_t>(m, "CudaArrayUint16");
    gpu3gppchan_bindings::declare_cuda_array<uint32_t>(m, "CudaArrayUint32");
    gpu3gppchan_bindings::declare_cuda_array<__half>(m, "CudaArrayHalf");
    gpu3gppchan_bindings::declare_cuda_array<float>(m, "CudaArrayFloat");
    gpu3gppchan_bindings::declare_cuda_array<std::complex<float>>(m, "CudaArrayComplexFloat");


    // carrier configuration
    nb::class_<cuphyCarrierPrms_t>(m, "CarrierParams")
        .def(nb::init<>())
        .def_rw("n_sc", &cuphyCarrierPrms_t::N_sc)
        .def_rw("n_fft", &cuphyCarrierPrms_t::N_FFT)
        .def_rw("n_bs_layer", &cuphyCarrierPrms_t::N_bsLayer)
        .def_rw("n_ue_layer", &cuphyCarrierPrms_t::N_ueLayer)
        .def_rw("id_slot", &cuphyCarrierPrms_t::id_slot)
        .def_rw("id_subframe", &cuphyCarrierPrms_t::id_subFrame)
        .def_rw("mu", &cuphyCarrierPrms_t::mu)
        .def_rw("cp_type", &cuphyCarrierPrms_t::cpType)
        .def_rw("f_c", &cuphyCarrierPrms_t::f_c)
        .def_rw("t_c", &cuphyCarrierPrms_t::T_c)
        .def_rw("f_samp", &cuphyCarrierPrms_t::f_samp)
        .def_rw("n_symbol_slot", &cuphyCarrierPrms_t::N_symbol_slot)
        .def_rw("k_const", &cuphyCarrierPrms_t::k_const)
        .def_rw("kappa_bits", &cuphyCarrierPrms_t::kappa_bits)
        .def_rw("ofdm_window_len", &cuphyCarrierPrms_t::ofdmWindowLen)
        .def_rw("rolloff_factor", &cuphyCarrierPrms_t::rolloffFactor)
        .def_rw("n_samp_slot", &cuphyCarrierPrms_t::N_samp_slot)

        // below are PRACH parameters
        .def_rw("n_u_mu", &cuphyCarrierPrms_t::N_u_mu)
        .def_rw("start_ra_sym", &cuphyCarrierPrms_t::startRaSym)
        .def_rw("delta_f_ra", &cuphyCarrierPrms_t::delta_f_RA)
        .def_rw("n_cp_ra", &cuphyCarrierPrms_t::N_CP_RA)
        .def_rw("k", &cuphyCarrierPrms_t::K)
        .def_rw("k1", &cuphyCarrierPrms_t::k1)
        .def_rw("k_bar", &cuphyCarrierPrms_t::kBar)
        .def_rw("n_u", &cuphyCarrierPrms_t::N_u)
        .def_rw("l_ra", &cuphyCarrierPrms_t::L_RA)
        .def_rw("n_slot_ra_sel", &cuphyCarrierPrms_t::n_slot_RA_sel)
        .def_rw("n_rep", &cuphyCarrierPrms_t::N_rep);

    // OFDM modulation
    nb::class_<gpu3gppchan_bindings::OfdmModulateWrapper<float, cuComplex>>(m, "OfdmModulate")
        .def("__init__", [](gpu3gppchan_bindings::OfdmModulateWrapper<float, cuComplex>* self,
                            cuphyCarrierPrms_t* params,
                            const gpu3gppchan_bindings::cuda_array_complex_float& input,
                            const nb::object& stream) {
                new (self) gpu3gppchan_bindings::OfdmModulateWrapper<float, cuComplex>(
                    params, reinterpret_cast<uintptr_t>(input.get_device_ptr()),
                    gpu3gppchan_bindings::stream_handle_from_owner(stream));
            },
            nb::arg("carrier_params"), nb::arg("freq_data_in_gpu"), nb::arg("stream_handle"),
            nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>())
        .def("__init__", [](gpu3gppchan_bindings::OfdmModulateWrapper<float, cuComplex>* self,
                            cuphyCarrierPrms_t* params,
                            gpu3gppchan_bindings::HostComplexArray<float> input,
                            const nb::object& stream) {
                new (self) gpu3gppchan_bindings::OfdmModulateWrapper<float, cuComplex>(
                    params, input, gpu3gppchan_bindings::stream_handle_from_owner(stream));
            },
            nb::arg("carrier_params"), nb::arg("freq_data_in_cpu"), nb::arg("stream_handle"),
            nb::keep_alive<1, 4>())
        .def("run", &gpu3gppchan_bindings::OfdmModulateWrapper<float, cuComplex>::run,
            nb::lock_self(),
            nb::arg("freq_data_in_cpu") = nb::none(),
            nb::arg("enable_swap_tx_rx") = 0)
        .def("print_time_sample", &gpu3gppchan_bindings::OfdmModulateWrapper<float, cuComplex>::printTimeSample,
            nb::lock_self(), nb::arg("print_length") = 10)
        .def("get_time_data_out", &gpu3gppchan_bindings::OfdmModulateWrapper<float, cuComplex>::getTimeDataOutArray,
            nb::lock_self(), nb::keep_alive<0, 1>())
        .def("get_time_data_length", &gpu3gppchan_bindings::OfdmModulateWrapper<float, cuComplex>::getTimeDataLen,
            nb::lock_self())
        .def("get_each_symbol_len_with_cp", &gpu3gppchan_bindings::OfdmModulateWrapper<float, cuComplex>::getEachSymbolLenWithCP,
            nb::lock_self());

    // OFDM demodulation
    nb::class_<gpu3gppchan_bindings::OfdmDeModulateWrapper<float, cuComplex>>(m, "OfdmDeModulate")
        // prach/per_ant_samp are C++ bool; accept Python int (nanobind won't auto int->bool).
        .def("__init__", [](gpu3gppchan_bindings::OfdmDeModulateWrapper<float, cuComplex>* self, cuphyCarrierPrms_t* p, const gpu3gppchan_bindings::cuda_array_complex_float& time_data_in_gpu, const gpu3gppchan_bindings::cuda_array_complex_float& freq_data_out_gpu, int prach, int per_ant_samp, const nb::object& stream) {
                if (freq_data_out_gpu.is_readonly() || !freq_data_out_gpu.is_c_contiguous()) {
                    throw nb::value_error("freq_data_out_gpu must be writable and C-contiguous");
                }
                new (self) gpu3gppchan_bindings::OfdmDeModulateWrapper<float, cuComplex>(p, reinterpret_cast<uintptr_t>(time_data_in_gpu.get_device_ptr()), reinterpret_cast<uintptr_t>(freq_data_out_gpu.get_device_ptr()), prach != 0, per_ant_samp != 0, gpu3gppchan_bindings::stream_handle_from_owner(stream));
            },
            nb::arg("carrier_params"), nb::arg("time_data_in_gpu"), nb::arg("freq_data_out_gpu"), nb::arg("prach") = 0, nb::arg("per_ant_samp") = 0, nb::arg("stream_handle"),
            nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>(), nb::keep_alive<1, 7>())
        .def("__init__", [](gpu3gppchan_bindings::OfdmDeModulateWrapper<float, cuComplex>* self, cuphyCarrierPrms_t* p, const gpu3gppchan_bindings::cuda_array_complex_float& time_data_in_gpu, gpu3gppchan_bindings::HostComplexArray<float> freq_data_out_cpu, int prach, int per_ant_samp, const nb::object& stream) {
                new (self) gpu3gppchan_bindings::OfdmDeModulateWrapper<float, cuComplex>(p, reinterpret_cast<uintptr_t>(time_data_in_gpu.get_device_ptr()), freq_data_out_cpu, prach != 0, per_ant_samp != 0, gpu3gppchan_bindings::stream_handle_from_owner(stream));
            },
            nb::arg("carrier_params"), nb::arg("time_data_in_gpu"), nb::arg("freq_data_out_cpu"), nb::arg("prach") = 0, nb::arg("per_ant_samp") = 0, nb::arg("stream_handle"),
            nb::keep_alive<1, 3>(), nb::keep_alive<1, 7>())
        .def("run", &gpu3gppchan_bindings::OfdmDeModulateWrapper<float, cuComplex>::run,
            nb::lock_self(),
            nb::arg("freq_data_out_cpu") = nb::none(),
            nb::arg("enable_swap_tx_rx") = 0)
        .def("print_freq_sample", &gpu3gppchan_bindings::OfdmDeModulateWrapper<float, cuComplex>::printFreqSample,
            nb::lock_self(), nb::arg("print_length") = 10)
        .def("get_freq_data_out", &gpu3gppchan_bindings::OfdmDeModulateWrapper<float, cuComplex>::getFreqDataOut,
            nb::lock_self(), nb::rv_policy::reference);

    // TDL channel configuration
    nb::class_<tdlConfig_t>(m, "TdlConfig")
        .def(nb::init<>())
        // nanobind (unlike pybind11) won't implicitly convert int->bool, but the
        // Python wrapper assigns int(...). Accept int/bool here and store as bool.
        .def_prop_rw("use_simplified_pdp",
            [](const tdlConfig_t& c) -> bool { return c.useSimplifiedPdp; },
            [](tdlConfig_t& c, int v) { c.useSimplifiedPdp = (v != 0); })
        .def_rw("delay_profile", &tdlConfig_t::delayProfile)
        .def_rw("delay_spread", &tdlConfig_t::delaySpread)
        .def_rw("max_doppler_shift", &tdlConfig_t::maxDopplerShift)
        .def_rw("f_samp", &tdlConfig_t::f_samp)
        .def_rw("n_cell", &tdlConfig_t::nCell)
        .def_rw("n_ue", &tdlConfig_t::nUe)
        .def_rw("n_bs_ant", &tdlConfig_t::nBsAnt)
        .def_rw("n_ue_ant", &tdlConfig_t::nUeAnt)
        .def_rw("f_batch", &tdlConfig_t::fBatch)
        .def_rw("n_path", &tdlConfig_t::numPath)
        .def_rw("cfo_hz", &tdlConfig_t::cfoHz)
        .def_rw("delay", &tdlConfig_t::delay)
        .def_rw("signal_length_per_ant", &tdlConfig_t::sigLenPerAnt)
        .def_rw("n_sc", &tdlConfig_t::N_sc)
        .def_rw("n_sc_prbg", &tdlConfig_t::N_sc_Prbg)
        .def_rw("sc_spacing_hz", &tdlConfig_t::scSpacingHz)
        .def_rw("freq_convert_type", &tdlConfig_t::freqConvertType)
        .def_rw("sc_sampling", &tdlConfig_t::scSampling)
        .def_rw("run_mode", &tdlConfig_t::runMode)
        .def_rw("proc_sig_freq", &tdlConfig_t::procSigFreq)
        .def_rw("save_ant_pair_sample", &tdlConfig_t::saveAntPairSample)
        .def_rw("batch_len", &tdlConfig_t::batchLen)
        .def_rw("tx_signal_in", &tdlConfig_t::txSigIn);

    nb::class_<gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>>(m, "TdlChan")
        .def("__init__", [](gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>* self,
                            tdlConfig_t* config, uint16_t seed, const nb::object& stream) {
                new (self) gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>(
                    config, seed, gpu3gppchan_bindings::stream_handle_from_owner(stream));
            }, nb::arg("tdl_cfg"), nb::arg("rand_seed"), nb::arg("stream_handle"),
            nb::keep_alive<1, 4>())
        .def("run", &gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>::run,
            nb::lock_self(),
            nb::arg("tx_signal_in"), nb::arg("ref_time0") = 0.0f, nb::arg("enable_swap_tx_rx") = 0, nb::arg("tx_column_major_ind") = 0,
            "Run channel with CuPy/GPU array input")
        .def("get_rx_signal_out_array", &gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>::getRxSignalOutArray,
            nb::lock_self(),
            nb::arg("enable_swap_tx_rx") = 0,
            nb::keep_alive<0, 1>(),
            "Get output signal as a CUDA array compatible with CuPy")
        .def("reset", &gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>::reset, nb::lock_self())
        .def("get_time_chan", &gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>::getTimeChan, nb::lock_self())
        .def("get_freq_chan_sc", &gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>::getFreqChanSc, nb::lock_self())
        .def("get_freq_chan_prbg", &gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>::getFreqChanPrbg, nb::lock_self())
        .def("get_rx_signal_out", &gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>::getRxSigOut, nb::lock_self())
        .def("get_rx_time_ant_pair_signal_out", &gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>::getRxTimeAntPairSigOut, nb::lock_self())
        .def("get_time_chan_size", &gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>::getTimeChanSize, nb::lock_self())
        .def("get_freq_chan_sc_per_link_size", &gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>::getFreqChanScPerLinkSize, nb::lock_self())
        .def("get_freq_chan_prbg_size", &gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>::getFreqChanPrbgSize, nb::lock_self())
        .def("print_time_chan", &gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>::printTimeChan,
            nb::lock_self(),
            nb::arg("cid") = 0, nb::arg("uid") = 0, nb::arg("print_length") = 10)
        .def("print_freq_sc_chan", &gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>::printFreqScChan,
            nb::lock_self(),
            nb::arg("cid") = 0, nb::arg("uid") = 0, nb::arg("print_length") = 10)
        .def("print_freq_prbg_chan", &gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>::printFreqPrbgChan,
            nb::lock_self(),
            nb::arg("cid") = 0, nb::arg("uid") = 0, nb::arg("print_length") = 10)
        .def("print_signal", &gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>::printSig,
            nb::lock_self(),
            nb::arg("cid") = 0, nb::arg("uid") = 0, nb::arg("print_length") = 10)
        .def("print_gpu_memory_usage_mb", &gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>::printGpuMemUseMB, nb::lock_self())
        .def("dump_cir", &gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>::dumpCir, nb::lock_self())
        .def("dump_cfr_prbg", &gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>::dumpCfrPrbg, nb::lock_self())
        .def("dump_cfr_sc", &gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>::dumpCfrSc, nb::lock_self())
        .def("save_tdl_chan_to_h5_file", &gpu3gppchan_bindings::TdlChanWrapper<float, cuComplex>::saveTdlChanToH5File,
            nb::lock_self(),
            nb::arg("pad_file_name_ending") = "");

    // CDL channel configuration
    nb::class_<cdlConfig_t>(m, "CdlConfig")
        .def(nb::init<>())
        .def_rw("delay_profile", &cdlConfig_t::delayProfile)
        .def_rw("delay_spread", &cdlConfig_t::delaySpread)
        .def_rw("max_doppler_shift", &cdlConfig_t::maxDopplerShift)
        .def_rw("f_samp", &cdlConfig_t::f_samp)
        .def_rw("n_cell", &cdlConfig_t::nCell)
        .def_rw("n_ue", &cdlConfig_t::nUe)
        .def_rw("bs_ant_size", &cdlConfig_t::bsAntSize)
        .def_rw("bs_ant_spacing", &cdlConfig_t::bsAntSpacing)
        .def_rw("bs_ant_polar_angles", &cdlConfig_t::bsAntPolarAngles)
        .def_rw("bs_ant_pattern", &cdlConfig_t::bsAntPattern)
        .def_rw("ue_ant_size", &cdlConfig_t::ueAntSize)
        .def_rw("ue_ant_spacing", &cdlConfig_t::ueAntSpacing)
        .def_rw("ue_ant_polar_angles", &cdlConfig_t::ueAntPolarAngles)
        .def_rw("ue_ant_pattern", &cdlConfig_t::ueAntPattern)
        .def_rw("v_direction", &cdlConfig_t::vDirection)
        .def_rw("f_batch", &cdlConfig_t::fBatch)
        .def_rw("n_ray", &cdlConfig_t::numRay)
        .def_rw("cfo_hz", &cdlConfig_t::cfoHz)
        .def_rw("delay", &cdlConfig_t::delay)
        .def_rw("signal_length_per_ant", &cdlConfig_t::sigLenPerAnt)
        .def_rw("n_sc", &cdlConfig_t::N_sc)
        .def_rw("n_sc_prbg", &cdlConfig_t::N_sc_Prbg)
        .def_rw("sc_spacing_hz", &cdlConfig_t::scSpacingHz)
        .def_rw("freq_convert_type", &cdlConfig_t::freqConvertType)
        .def_rw("sc_sampling", &cdlConfig_t::scSampling)
        .def_rw("run_mode", &cdlConfig_t::runMode)
        .def_rw("proc_sig_freq", &cdlConfig_t::procSigFreq)
        .def_rw("save_ant_pair_sample", &cdlConfig_t::saveAntPairSample)
        .def_rw("batch_len", &cdlConfig_t::batchLen)
        .def_rw("tx_signal_in", &cdlConfig_t::txSigIn);

    nb::class_<gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>>(m, "CdlChan")
        .def("__init__", [](gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>* self,
                            cdlConfig_t* config, uint16_t seed, const nb::object& stream) {
                new (self) gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>(
                    config, seed, gpu3gppchan_bindings::stream_handle_from_owner(stream));
            }, nb::arg("cdl_cfg"), nb::arg("rand_seed"), nb::arg("stream_handle"),
            nb::keep_alive<1, 4>())
        .def("run", &gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>::run,
            nb::lock_self(),
            nb::arg("tx_signal_in"), nb::arg("ref_time0") = 0.0f, nb::arg("enable_swap_tx_rx") = 0, nb::arg("tx_column_major_ind") = 0,
            "Run channel with CuPy/GPU array input")
        .def("get_rx_signal_out_array", &gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>::getRxSignalOutArray,
            nb::lock_self(),
            nb::arg("enable_swap_tx_rx") = 0,
            nb::keep_alive<0, 1>(),
            "Get output signal as a CUDA array compatible with CuPy")
        .def("reset", &gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>::reset, nb::lock_self())
        .def("get_time_chan", &gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>::getTimeChan, nb::lock_self())
        .def("get_freq_chan_sc", &gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>::getFreqChanSc, nb::lock_self())
        .def("get_freq_chan_prbg", &gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>::getFreqChanPrbg, nb::lock_self())
        .def("get_rx_signal_out", &gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>::getRxSigOut, nb::lock_self())
        .def("get_rx_time_ant_pair_signal_out", &gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>::getRxTimeAntPairSigOut, nb::lock_self())
        .def("get_time_chan_size", &gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>::getTimeChanSize, nb::lock_self())
        .def("get_freq_chan_sc_per_link_size", &gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>::getFreqChanScPerLinkSize, nb::lock_self())
        .def("get_freq_chan_prbg_size", &gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>::getFreqChanPrbgSize, nb::lock_self())
        .def("print_time_chan", &gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>::printTimeChan,
            nb::lock_self(),
            nb::arg("cid") = 0, nb::arg("uid") = 0, nb::arg("print_length") = 10)
        .def("print_freq_sc_chan", &gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>::printFreqScChan,
            nb::lock_self(),
            nb::arg("cid") = 0, nb::arg("uid") = 0, nb::arg("print_length") = 10)
        .def("print_freq_prbg_chan", &gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>::printFreqPrbgChan,
            nb::lock_self(),
            nb::arg("cid") = 0, nb::arg("uid") = 0, nb::arg("print_length") = 10)
        .def("print_signal", &gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>::printSig,
            nb::lock_self(),
            nb::arg("cid") = 0, nb::arg("uid") = 0, nb::arg("print_length") = 10)
        .def("print_gpu_memory_usage_mb", &gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>::printGpuMemUseMB, nb::lock_self())
        .def("dump_cir", &gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>::dumpCir, nb::lock_self())
        .def("dump_cfr_prbg", &gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>::dumpCfrPrbg, nb::lock_self())
        .def("dump_cfr_sc", &gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>::dumpCfrSc, nb::lock_self())
        .def("save_cdl_chan_to_h5_file", &gpu3gppchan_bindings::CdlChanWrapper<float, cuComplex>::saveCdlChanToH5File,
            nb::lock_self(),
            nb::arg("pad_file_name_ending") = "");

    // Channel Models API bindings
    // Bind Scenario enum
    nb::enum_<Scenario>(m, "Scenario", "Deployment scenario types for channel modeling")
        .value("UMa", Scenario::UMa, "Urban Macro scenario")
        .value("UMi", Scenario::UMi, "Urban Micro scenario")
        .value("RMa", Scenario::RMa, "Rural Macro scenario")
        .value("Indoor", Scenario::Indoor, "Indoor scenario (TODO: Not supported yet)")
        .value("InF", Scenario::InF, "Indoor Factory scenario (TODO: Not supported yet)")
        .value("SMa", Scenario::SMa, "Suburban Macro scenario (TODO: Not supported yet)")
        .export_values();

    // Bind SensingTargetType enum (for ISAC)
    nb::enum_<SensingTargetType>(m, "SensingTargetType", "Sensing target types for ISAC per 3GPP TR 38.901 Section 7.9")
        .value("UAV", SensingTargetType::UAV, "UAV sensing target (Table 7.9.1-1)")
        .value("AUTOMOTIVE", SensingTargetType::AUTOMOTIVE, "Automotive sensing target (Table 7.9.1-2)")
        .value("HUMAN", SensingTargetType::HUMAN, "Human sensing target (Table 7.9.1-3)")
        .value("AGV", SensingTargetType::AGV, "Automated Guided Vehicle sensing target (Table 7.9.1-4)")
        .value("HAZARD", SensingTargetType::HAZARD, "Hazards on roads/railways sensing target (Table 7.9.1-5)")
        .export_values();

    // Bind UeType enum
    nb::enum_<UeType>(m, "UeType", "UE (User Equipment) device types per 3GPP categorization")
        .value("TERRESTRIAL", UeType::TERRESTRIAL, "Traditional handheld/fixed UE (smartphones, tablets, CPE)")
        .value("VEHICLE", UeType::VEHICLE, "Vehicular UE for V2X communication (cars, trucks, buses)")
        .value("AERIAL", UeType::AERIAL, "Aerial UE (drones, UAVs for communication)")
        .value("AGV", UeType::AGV, "Automated Guided Vehicle (industrial robots)")
        .value("RSU", UeType::RSU, "Road Side Unit (fixed V2X infrastructure)")
        .export_values();

    // Bind Coordinate struct
    nb::class_<Coordinate>(m, "Coordinate", "3D coordinate structure for global coordinate system")
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<float, float, float>(),
             "Initialize with x, y, z coordinates",
             nb::arg("x") = 0.0f, nb::arg("y") = 0.0f, nb::arg("z") = 0.0f)
        .def_rw("x", &Coordinate::x, "x-coordinate in global coordinate system")
        .def_rw("y", &Coordinate::y, "y-coordinate in global coordinate system")
        .def_rw("z", &Coordinate::z, "z-coordinate in global coordinate system")
        .def("__repr__", [](const Coordinate& c) {
            return "Coordinate(x=" + std::to_string(c.x) +
                   ", y=" + std::to_string(c.y) +
                   ", z=" + std::to_string(c.z) + ")";
        });

    // Bind SpstParam struct (SPST = Sub-Pixel Scattering Point for ISAC)
    nb::class_<SpstParam>(m, "SpstParam", "SPST (Scattering Point) parameter configuration for ISAC sensing targets")
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<uint32_t, const Coordinate&, float, float, float>(),
             "Initialize with SPST parameters",
             nb::arg("spst_id"), nb::arg("loc_in_st_lcs"),
             nb::arg("rcs_sigma_m_dbsm") = -12.81f, nb::arg("rcs_sigma_d_dbsm") = 1.0f,
             nb::arg("rcs_sigma_s_db") = 3.74f)
        .def_rw("spst_id", &SpstParam::spst_id, "SPST ID within the ST (0-indexed)")
        .def_rw("loc_in_st_lcs", &SpstParam::loc_in_st_lcs,
                      "Location of SPST in ST's local coordinate system")
        .def_rw("rcs_sigma_m_dbsm", &SpstParam::rcs_sigma_m_dbsm,
                      "Mean monostatic RCS sigma_M in dBsm")
        .def_rw("rcs_sigma_d_dbsm", &SpstParam::rcs_sigma_d_dbsm,
                      "Mean monostatic RCS sigma_D in dBsm")
        .def_rw("rcs_sigma_s_db", &SpstParam::rcs_sigma_s_db,
                      "Standard deviation sigma_s_dB in dB")
        .def_rw("enable_forward_scattering", &SpstParam::enable_forward_scattering,
                      "Control forward scattering effect: 0=disable, 1=enable");

    // Bind StParam struct (Sensing Target for ISAC)
    nb::class_<StParam>(m, "StParam", "Sensing Target (ST) parameter configuration for ISAC")
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<uint32_t, uint8_t, const Coordinate&>(),
             "Initialize with basic ST parameters",
             nb::arg("sid"), nb::arg("outdoor_ind") = 1, nb::arg("loc") = Coordinate())
        .def(nb::init<uint32_t, SensingTargetType, uint8_t, const Coordinate&, uint8_t>(),
             "Initialize with target type and RCS model",
             nb::arg("sid"), nb::arg("target_type"), nb::arg("outdoor_ind"),
             nb::arg("loc"), nb::arg("rcs_model") = 1)
        .def_rw("sid", &StParam::sid, "Global ST ID (Sensing Target ID)")
        .def_rw("target_type", &StParam::target_type, "Type of sensing target")
        .def_rw("outdoor_ind", &StParam::outdoor_ind, "0: indoor, 1: outdoor")
        .def_rw("loc", &StParam::loc, "ST location in GCS")
        .def_rw("rcs_model", &StParam::rcs_model,
                      "RCS model: 1=deterministic monostatic, 2=angular dependent")
        .def_prop_rw("n_spst",
            [](const StParam& self) { return self.n_spst; },
            [](StParam& self, uint32_t value) {
                StParam candidate = self;
                candidate.n_spst = value;
                try {
                    candidate.validateSpstConsistency(true);
                } catch (const std::invalid_argument& e) {
                    throw nb::value_error(e.what());
                }
                self.n_spst = value;
            },
            "Number of scattering points (SPSTs)")
        .def_prop_rw("spst_configs",
            [](const StParam& self) { return self.spst_configs; },
            [](StParam& self, const std::vector<SpstParam>& value) {
                StParam candidate = self;
                candidate.spst_configs = value;
                try {
                    candidate.validateSpstConsistency(false);
                } catch (const std::invalid_argument& e) {
                    throw nb::value_error(e.what());
                }
                self.spst_configs = value;
            },
            "List of SPST parameter configurations")
        .def_prop_rw("velocity",
            [](const StParam& self) { return gpu3gppchan_bindings::carray_to_pylist(self.velocity); },
            [](StParam& self, const nb::object& obj) {
                gpu3gppchan_bindings::object_to_fixed_carray(obj, self.velocity);
            },
            "Velocity vector [vx, vy, vz] in m/s")
        .def_prop_rw("target_orientation",
            [](const StParam& self) { return gpu3gppchan_bindings::carray_to_pylist(self.orientation); },
            [](StParam& self, const nb::object& obj) {
                gpu3gppchan_bindings::object_to_fixed_carray(obj, self.orientation);
            },
            "Target orientation [azimuth, elevation] in degrees")
        .def_prop_rw("physical_size",
            [](const StParam& self) { return gpu3gppchan_bindings::carray_to_pylist(self.physical_size); },
            [](StParam& self, const nb::object& obj) {
                gpu3gppchan_bindings::object_to_fixed_carray(obj, self.physical_size);
            },
            "Physical dimensions [length, width, height] in meters");

        // Bind AntPanelConfig struct with numpy array support
    nb::class_<AntPanelConfig>(m, "AntPanelConfig", "Antenna panel configuration parameters")
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<uint16_t, uint8_t>(),
             "Initialize with number of antennas and antenna model",
             nb::arg("n_ant"), nb::arg("ant_model") = 1)
        .def("__init__", [](AntPanelConfig* self, uint16_t n_ant, const std::vector<uint16_t>& ant_size, const std::vector<float>& ant_spacing, const std::vector<float>& ant_polar_angles, uint8_t ant_model) {
                 new (self) AntPanelConfig();
                 self->nAnt = n_ant;
                 self->antModel = ant_model;
                 gpu3gppchan_bindings::vector_to_fixed_carray(ant_size, self->antSize);
                 gpu3gppchan_bindings::vector_to_fixed_carray(ant_spacing, self->antSpacing);
                 gpu3gppchan_bindings::vector_to_fixed_carray(ant_polar_angles, self->antPolarAngles);
             },
             "Initialize with antenna parameters for models 0 and 1",
             nb::arg("n_ant"), nb::arg("ant_size"), nb::arg("ant_spacing"), nb::arg("ant_polar_angles"), nb::arg("ant_model") = 1)
        .def("__init__", [](AntPanelConfig* self, uint16_t n_ant, const std::vector<uint16_t>& ant_size, const std::vector<float>& ant_spacing, const std::vector<float>& ant_theta, const std::vector<float>& ant_phi, const std::vector<float>& ant_polar_angles, uint8_t ant_model) {
                 new (self) AntPanelConfig();
                 self->nAnt = n_ant;
                 self->antModel = ant_model;
                 gpu3gppchan_bindings::vector_to_fixed_carray(ant_size, self->antSize);
                 gpu3gppchan_bindings::vector_to_fixed_carray(ant_spacing, self->antSpacing);
                 gpu3gppchan_bindings::vector_to_fixed_carray(ant_theta, self->antTheta);
                 gpu3gppchan_bindings::vector_to_fixed_carray(ant_phi, self->antPhi);
                 gpu3gppchan_bindings::vector_to_fixed_carray(ant_polar_angles, self->antPolarAngles);
             },
             "Initialize with full antenna parameters including direct patterns",
             nb::arg("n_ant"), nb::arg("ant_size"), nb::arg("ant_spacing"), nb::arg("ant_theta"), nb::arg("ant_phi"), nb::arg("ant_polar_angles"), nb::arg("ant_model") = 2)
        .def_rw("n_ant", &AntPanelConfig::nAnt,
                      "Number of antennas in the array (nAnt = M_g * N_g * M * N * P)")
        .def_prop_rw("ant_size",
            [](const AntPanelConfig& self) {
                nb::list result;
                for (size_t i = 0; i < 5; ++i) {
                    result.append(self.antSize[i]);
                }
                return result;
            },
            [](AntPanelConfig& self, const nb::list& list) {
                if (list.size() != 5) {
                    throw std::invalid_argument("ant_size must have exactly 5 elements");
                }
                for (size_t i = 0; i < 5; ++i) {
                    self.antSize[i] = nb::cast<uint16_t>(list[i]);
                }
            },
            "Dimensions of the antenna array [M_g, N_g, M, N, P]")
        .def_prop_rw("ant_spacing",
            [](const AntPanelConfig& self) {
                nb::list result;
                for (size_t i = 0; i < 4; ++i) {
                    result.append(self.antSpacing[i]);
                }
                return result;
            },
            [](AntPanelConfig& self, const nb::list& list) {
                if (list.size() != 4) {
                    throw std::invalid_argument("ant_spacing must have exactly 4 elements");
                }
                for (size_t i = 0; i < 4; ++i) {
                    self.antSpacing[i] = nb::cast<float>(list[i]);
                }
            },
            "Spacing between antennas in wavelengths [d_g_h, d_g_v, d_h, d_v]")
        .def_prop_rw("ant_theta",
            [](const AntPanelConfig& self) {
                nb::list result;
                for (size_t i = 0; i < 181; ++i) {
                    result.append(self.antTheta[i]);
                }
                return result;
            },
            [](AntPanelConfig& self, const nb::list& list) {
                if (list.size() != 181) {
                    throw std::invalid_argument("ant_theta must have exactly 181 elements (0-180 degrees)");
                }
                for (size_t i = 0; i < 181; ++i) {
                    self.antTheta[i] = nb::cast<float>(list[i]);
                }
            },
            "Antenna pattern A(theta, phi=0) in dB, size 181 (0-180 degrees)")
        .def_prop_rw("ant_phi",
            [](const AntPanelConfig& self) {
                nb::list result;
                for (size_t i = 0; i < 360; ++i) {
                    result.append(self.antPhi[i]);
                }
                return result;
            },
            [](AntPanelConfig& self, const nb::list& list) {
                if (list.size() != 360) {
                    throw std::invalid_argument("ant_phi must have exactly 360 elements (0-360 degrees)");
                }
                for (size_t i = 0; i < 360; ++i) {
                    self.antPhi[i] = nb::cast<float>(list[i]);
                }
            },
            "Antenna pattern A(theta=90, phi) in dB, size 360 (0-360 degrees)")
        .def_prop_rw("ant_polar_angles",
            [](const AntPanelConfig& self) {
                nb::list result;
                for (size_t i = 0; i < 2; ++i) {
                    result.append(self.antPolarAngles[i]);
                }
                return result;
            },
            [](AntPanelConfig& self, const nb::list& list) {
                if (list.size() != 2) {
                    throw std::invalid_argument("ant_polar_angles must have exactly 2 elements [roll_angle_first_polz, roll_angle_second_polz]");
                }
                for (size_t i = 0; i < 2; ++i) {
                    self.antPolarAngles[i] = nb::cast<float>(list[i]);
                }
            },
            "Antenna polarization angles [roll_angle_first_polz, roll_angle_second_polz]")
        .def_rw("ant_model", &AntPanelConfig::antModel,
                      "Antenna model type: 0=isotropic, 1=directional, 2=direct pattern");

    // Bind UtParamCfg struct (public API) with numpy array support
    nb::class_<UtParamCfg>(m, "UtParamCfg", "User Terminal parameter configuration")
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<uint32_t, const Coordinate&, uint8_t, uint32_t, UeType>(),
             "Initialize with basic parameters",
             nb::arg("uid"), nb::arg("loc"), nb::arg("outdoor_ind") = 0, nb::arg("ant_panel_idx") = 0, nb::arg("ue_type") = UeType::TERRESTRIAL)
        .def("__init__", [](UtParamCfg* self, uint32_t uid, const Coordinate& loc, uint8_t outdoor_ind, uint32_t ant_panel_idx, const std::vector<float>& ant_panel_orientation, const std::vector<float>& velocity, UeType ue_type) {
                 new (self) UtParamCfg();
                 self->uid = uid;
                 self->loc = loc;
                 self->outdoor_ind = outdoor_ind;
                 self->ue_type = ue_type;
                 self->antPanelIdx = ant_panel_idx;
                 gpu3gppchan_bindings::vector_to_fixed_carray(ant_panel_orientation, self->antPanelOrientation);
                 gpu3gppchan_bindings::vector_to_fixed_carray(velocity, self->velocity);
             },
             "Initialize with full parameters including orientation and velocity",
             nb::arg("uid"), nb::arg("loc"), nb::arg("outdoor_ind"), nb::arg("ant_panel_idx"), nb::arg("ant_panel_orientation"), nb::arg("velocity"), nb::arg("ue_type") = UeType::TERRESTRIAL)
        .def_rw("uid", &UtParamCfg::uid, "Global UE ID")
        .def_rw("loc", &UtParamCfg::loc, "UE location")
        .def_rw("outdoor_ind", &UtParamCfg::outdoor_ind, "Outdoor indicator: 0=indoor, 1=outdoor")
        .def_rw("ue_type", &UtParamCfg::ue_type, "UE type: TERRESTRIAL, VEHICLE, AERIAL, AGV, RSU")
        .def_rw("ant_panel_idx", &UtParamCfg::antPanelIdx, "Antenna panel configuration index")
        .def_prop_rw("ant_panel_orientation",
            [](const UtParamCfg& self) { return gpu3gppchan_bindings::carray_to_pylist(self.antPanelOrientation); },
            [](UtParamCfg& self, const nb::object& obj) {
                gpu3gppchan_bindings::object_to_fixed_carray(obj, self.antPanelOrientation);
            },
            "Antenna panel orientation in GCS [theta, phi, slant_offset]")
        .def_prop_rw("velocity",
            [](const UtParamCfg& self) { return gpu3gppchan_bindings::carray_to_pylist(self.velocity); },
            [](UtParamCfg& self, const nb::object& obj) {
                gpu3gppchan_bindings::object_to_fixed_carray(obj, self.velocity);
            },
            "Velocity vector [vx, vy, vz] in m/s, vz=0 per 3GPP spec")
        .def_rw("monostatic_ind", &UtParamCfg::monostatic_ind,
                      "0: not a monostatic sensing receiver, 1: monostatic sensing receiver")
        .def_rw("same_antenna_panel_ind", &UtParamCfg::same_antenna_panel_ind,
                      "0: use second antenna panel for sensing, 1: use same antenna panel")
        .def_rw("second_ant_panel_idx", &UtParamCfg::second_ant_panel_idx,
                      "Second antenna panel index for sensing RX (when monostatic_ind=1)")
        .def_prop_rw("second_ant_panel_orientation",
            [](const UtParamCfg& self) { return gpu3gppchan_bindings::carray_to_pylist(self.second_ant_panel_orientation); },
            [](UtParamCfg& self, const nb::object& obj) {
                gpu3gppchan_bindings::object_to_fixed_carray(obj, self.second_ant_panel_orientation);
            },
            "Second antenna panel orientation for sensing RX [theta, phi, slant_offset]");

    // Bind CellParam struct with numpy array support
    nb::class_<CellParam>(m, "CellParam", "Cell/Base Station parameters")
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<uint32_t, uint32_t, Coordinate, uint32_t>(),
             "Initialize with basic parameters",
             nb::arg("cid"), nb::arg("site_id"), nb::arg("loc"), nb::arg("ant_panel_idx"))
        .def("__init__", [](CellParam* self, uint32_t cid, uint32_t site_id, Coordinate loc, uint32_t ant_panel_idx, const std::vector<float>& ant_panel_orientation) {
                 new (self) CellParam();
                 self->cid = cid;
                 self->siteId = site_id;
                 self->loc = loc;
                 self->antPanelIdx = ant_panel_idx;
                 gpu3gppchan_bindings::vector_to_fixed_carray(ant_panel_orientation, self->antPanelOrientation);
             },
             "Initialize with full parameters including antenna orientation",
             nb::arg("cid"), nb::arg("site_id"), nb::arg("loc"), nb::arg("ant_panel_idx"), nb::arg("ant_panel_orientation"))
        .def_rw("cid", &CellParam::cid, "Global cell ID")
        .def_rw("site_id", &CellParam::siteId, "Site ID for LSP access")
        .def_rw("loc", &CellParam::loc, "Cell location")
        .def_rw("ant_panel_idx", &CellParam::antPanelIdx, "Antenna panel configuration index")
        .def_prop_rw("ant_panel_orientation",
            [](const CellParam& self) { return gpu3gppchan_bindings::carray_to_pylist(self.antPanelOrientation); },
            [](CellParam& self, const nb::object& obj) {
                gpu3gppchan_bindings::object_to_fixed_carray(obj, self.antPanelOrientation);
            },
            "Antenna panel orientation in GCS [theta, phi, slant_offset]")
        .def_rw("monostatic_ind", &CellParam::monostatic_ind,
                      "0: not monostatic, 1: monostatic (BS acts as both TX and RX for sensing)")
        .def_rw("second_ant_panel_idx", &CellParam::second_ant_panel_idx,
                      "Second antenna panel index for sensing RX (when monostatic_ind=1)")
        .def_prop_rw("second_ant_panel_orientation",
            [](const CellParam& self) { return gpu3gppchan_bindings::carray_to_pylist(self.second_ant_panel_orientation); },
            [](CellParam& self, const nb::object& obj) {
                gpu3gppchan_bindings::object_to_fixed_carray(obj, self.second_ant_panel_orientation);
            },
            "Second antenna panel orientation for sensing RX [theta, phi, slant_offset]");

    // Bind SystemLevelConfig struct
    nb::class_<SystemLevelConfig>(m, "SystemLevelConfig", "System-level configuration parameters")
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<Scenario, uint32_t, uint8_t, uint32_t, float>(),
             "Initialize with basic parameters",
             nb::arg("scenario"), nb::arg("n_site"), nb::arg("n_sector_per_site"), nb::arg("n_ut"), nb::arg("isd") = 1732.0f)
        .def("__init__", [](SystemLevelConfig* self, Scenario scenario, uint32_t n_site, uint8_t n_sector_per_site, uint32_t n_ut, float isd, const std::vector<uint32_t>& ut_drop_cells, uint8_t ut_drop_option, const std::vector<float>& ut_cell_2d_dist, uint8_t optional_pl_ind, uint8_t o2i_building_penetr_loss_ind, uint8_t o2i_car_penetr_loss_ind, uint8_t enable_near_field_effect, uint8_t enable_non_stationarity, const std::vector<float>& force_los_prob, const std::vector<float>& force_ut_speed, float force_indoor_ratio, uint8_t disable_pl_shadowing, uint8_t disable_small_scale_fading, uint8_t enable_per_tti_lsp, uint8_t enable_propagation_delay) {
                 new (self) SystemLevelConfig();
                 self->scenario = scenario;
                 self->n_site = n_site;
                 self->n_sector_per_site = n_sector_per_site;
                 self->n_ut = n_ut;
                 self->isd = isd;
                 self->optional_pl_ind = optional_pl_ind;
                 self->o2i_building_penetr_loss_ind = o2i_building_penetr_loss_ind;
                 self->o2i_car_penetr_loss_ind = o2i_car_penetr_loss_ind;
                 self->enable_near_field_effect = enable_near_field_effect;
                 self->enable_non_stationarity = enable_non_stationarity;
                 gpu3gppchan_bindings::vector_to_fixed_carray(force_los_prob, self->force_los_prob);
                 gpu3gppchan_bindings::vector_to_fixed_carray(force_ut_speed, self->force_ut_speed);
                 self->force_indoor_ratio = force_indoor_ratio;
                 self->disable_pl_shadowing = disable_pl_shadowing;
                 self->disable_small_scale_fading = disable_small_scale_fading;
                 self->enable_per_tti_lsp = enable_per_tti_lsp;
                 self->enable_propagation_delay = enable_propagation_delay;
                 self->ut_drop_option = ut_drop_option;
                 gpu3gppchan_bindings::vector_to_fixed_carray(ut_cell_2d_dist, self->ut_cell_2d_dist);
                 gpu3gppchan_bindings::vector_to_capacity_carray(ut_drop_cells, self->ut_drop_cells);
                 self->n_ut_drop_cells = static_cast<uint32_t>(ut_drop_cells.size());
             },
             "Initialize with full system-level parameters",
             nb::arg("scenario"), nb::arg("n_site"), nb::arg("n_sector_per_site"), nb::arg("n_ut"), nb::arg("isd"),
             nb::arg("ut_drop_cells") = std::vector<uint32_t>{}, nb::arg("ut_drop_option") = 0, nb::arg("ut_cell_2d_dist") = std::vector<float>{-1.0f, -1.0f},
             nb::arg("optional_pl_ind") = 0, nb::arg("o2i_building_penetr_loss_ind") = 1, nb::arg("o2i_car_penetr_loss_ind") = 0, nb::arg("enable_near_field_effect") = 0,
             nb::arg("enable_non_stationarity") = 0, nb::arg("force_los_prob") = std::vector<float>{-1.0f, -1.0f}, nb::arg("force_ut_speed") = std::vector<float>{-1.0f, -1.0f},
             nb::arg("force_indoor_ratio") = -1.0f, nb::arg("disable_pl_shadowing") = 0, nb::arg("disable_small_scale_fading") = 0, nb::arg("enable_per_tti_lsp") = 1, nb::arg("enable_propagation_delay") = 1)
        .def_rw("scenario", &SystemLevelConfig::scenario, "Deployment scenario")
        .def_rw("isd", &SystemLevelConfig::isd, "Inter-site distance in meters")
        .def_rw("n_site", &SystemLevelConfig::n_site, "Number of sites")
        .def_rw("n_sector_per_site", &SystemLevelConfig::n_sector_per_site, "Sectors per site")
        .def_rw("n_ut", &SystemLevelConfig::n_ut, "Total number of UTs")
        .def_rw("isac_type", &SystemLevelConfig::isac_type,
                      "ISAC type: 0=communication only, 1=monostatic sensing, 2=bistatic sensing")
        .def_rw("n_st", &SystemLevelConfig::n_st, "Total number of sensing targets (STs)")
        .def_prop_rw("st_horizontal_speed",
            [](const SystemLevelConfig& self) { return gpu3gppchan_bindings::carray_to_pylist(self.st_horizontal_speed); },
            [](SystemLevelConfig& self, const nb::object& obj) {
                gpu3gppchan_bindings::object_to_fixed_carray(obj, self.st_horizontal_speed);
            },
            "Horizontal speed range [min, max] in m/s for ISAC sensing targets "
            "(default: [8.33, 8.33] = fixed 30 km/h)")
        .def_rw("st_vertical_velocity", &SystemLevelConfig::st_vertical_velocity,
                      "Vertical velocity in m/s for ISAC sensing targets (vz component, default: 0.0)")
        .def_prop_rw("st_distribution_option",
            [](const SystemLevelConfig& self) { return gpu3gppchan_bindings::carray_to_pylist(self.st_distribution_option); },
            [](SystemLevelConfig& self, const nb::object& obj) {
                gpu3gppchan_bindings::object_to_fixed_carray(obj, self.st_distribution_option);
            },
            "ST distribution option [horizontal, vertical]: 0=Option A, 1=Option B, 2=Option C")
        .def_prop_rw("st_height",
            [](const SystemLevelConfig& self) { return gpu3gppchan_bindings::carray_to_pylist(self.st_height); },
            [](SystemLevelConfig& self, const nb::object& obj) {
                gpu3gppchan_bindings::object_to_fixed_carray(obj, self.st_height);
            },
            "ST height range [min, max] in meters for vertical Option B (default: [100, 100])")
        // Backward-compatible alias (deprecated): st_fixed_height
        .def_prop_rw("st_fixed_height",
            [](const SystemLevelConfig& self) { return gpu3gppchan_bindings::carray_to_pylist(self.st_height); },
            [](SystemLevelConfig& self, const nb::object& obj) {
                gpu3gppchan_bindings::object_to_fixed_carray(obj, self.st_height);
            },
            "Deprecated alias of st_height")
        .def_rw("st_minimum_distance", &SystemLevelConfig::st_minimum_distance,
                      "Minimum distance between STs in meters (0=auto based on physical size)")
        .def_rw("st_size_ind", &SystemLevelConfig::st_size_ind,
                      "ST size index: 0=small, 1=medium, 2=large")
        .def_rw("st_min_dist_from_tx_rx", &SystemLevelConfig::st_min_dist_from_tx_rx,
                      "Minimum 3D distance from ST to any STX/SRX (BS/UE) in meters (default: 10m)")
        .def_rw("st_target_type", &SystemLevelConfig::st_target_type,
                      "Default target type for auto-generated STs (SensingTargetType enum)")
        .def_rw("st_rcs_model", &SystemLevelConfig::st_rcs_model,
                      "RCS model for STs: 1=deterministic monostatic, 2=angular dependent")
        .def_rw("path_drop_threshold_db", &SystemLevelConfig::path_drop_threshold_db,
                      "Path power drop threshold in dB for ISAC ray/path pruning (default: 40 dB)")
        .def_rw("isac_disable_background", &SystemLevelConfig::isac_disable_background,
                      "ISAC calibration mode: 0=combine target with background, 1=target CIR only")
        .def_rw("isac_disable_target", &SystemLevelConfig::isac_disable_target,
                      "ISAC calibration mode: 0=include target CIR, 1=background CIR only")
        .def_rw("optional_pl_ind", &SystemLevelConfig::optional_pl_ind,
                      "Pathloss equation: 0=standard, 1=optional")
        .def_rw("o2i_building_penetr_loss_ind", &SystemLevelConfig::o2i_building_penetr_loss_ind,
                      "Building penetration loss: 0=none, 1=low-loss, 2=high-loss")
        .def_rw("o2i_car_penetr_loss_ind", &SystemLevelConfig::o2i_car_penetr_loss_ind,
                      "Car penetration loss: 0=none, 1=basic, 2=metallized")
        .def_rw("enable_near_field_effect", &SystemLevelConfig::enable_near_field_effect,
                      "Near field effect: 0=disable, 1=enable")
        .def_rw("enable_non_stationarity", &SystemLevelConfig::enable_non_stationarity,
                      "Non-stationarity: 0=disable, 1=enable")
        .def_prop_rw("force_los_prob",
            [](const SystemLevelConfig& self) { return gpu3gppchan_bindings::carray_to_pylist(self.force_los_prob); },
            [](SystemLevelConfig& self, const nb::object& obj) {
                gpu3gppchan_bindings::object_to_fixed_carray(obj, self.force_los_prob);
            },
            "Force LOS probability [outdoor, indoor], -1 for auto calculation")
        .def_prop_rw("force_ut_speed",
            [](const SystemLevelConfig& self) { return gpu3gppchan_bindings::carray_to_pylist(self.force_ut_speed); },
            [](SystemLevelConfig& self, const nb::object& obj) {
                gpu3gppchan_bindings::object_to_fixed_carray(obj, self.force_ut_speed);
            },
            "Force UT speed [outdoor, indoor] in m/s, -1 for auto calculation")
        .def_rw("force_indoor_ratio", &SystemLevelConfig::force_indoor_ratio,
                      "Force indoor ratio, -1 for auto calculation")
        .def_rw("disable_pl_shadowing", &SystemLevelConfig::disable_pl_shadowing,
                      "Disable pathloss/shadowing: 0=calculate, 1=disable")
        .def_rw("disable_small_scale_fading", &SystemLevelConfig::disable_small_scale_fading,
                      "Disable small scale fading: 0=calculate, 1=disable (fast fading = 1)")
        .def_rw("enable_per_tti_lsp", &SystemLevelConfig::enable_per_tti_lsp,
                      "LSP per TTI: 0=disable, 1=update PL/shadowing, 2=update all")
        .def_rw("enable_propagation_delay", &SystemLevelConfig::enable_propagation_delay,
                      "Propagation delay in CIR: 0=disable, 1=enable")
        .def_rw("ut_drop_option", &SystemLevelConfig::ut_drop_option,
                      "UT drop control: 0=random across region, 1=same UTs per site, 2=same UTs per sector");

    // Bind LinkLevelConfig struct
    nb::class_<LinkLevelConfig>(m, "LinkLevelConfig", "Link-level configuration parameters")
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<int, char, float>(),
             "Initialize with basic parameters",
             nb::arg("fast_fading_type"), nb::arg("delay_profile") = 'A', nb::arg("delay_spread") = 30.0f)
        .def("__init__", [](LinkLevelConfig* self, int fast_fading_type, char delay_profile, float delay_spread, const std::vector<float>& velocity, int num_ray, float cfo_hz, float delay) {
                 new (self) LinkLevelConfig(fast_fading_type, delay_profile, delay_spread);
                 gpu3gppchan_bindings::vector_to_fixed_carray(velocity, self->velocity);
                 self->num_ray = num_ray;
                 self->cfo_hz = cfo_hz;
                 self->delay = delay;
             },
             "Initialize with full parameters including velocity",
             nb::arg("fast_fading_type"), nb::arg("delay_profile"), nb::arg("delay_spread"),
             nb::arg("velocity"), nb::arg("num_ray"), nb::arg("cfo_hz"), nb::arg("delay"))
        .def_rw("fast_fading_type", &LinkLevelConfig::fast_fading_type,
                      "Fast fading type: 0=AWGN, 1=TDL, 2=CDL")
        .def_rw("delay_profile", &LinkLevelConfig::delay_profile,
                      "Delay profile: 'A' to 'C'")
        .def_rw("delay_spread", &LinkLevelConfig::delay_spread,
                      "Delay spread in nanoseconds")
        .def_prop_rw("velocity",
            [](const LinkLevelConfig& self) { return gpu3gppchan_bindings::carray_to_pylist(self.velocity); },
            [](LinkLevelConfig& self, const nb::object& obj) {
                gpu3gppchan_bindings::object_to_fixed_carray(obj, self.velocity);
            },
            "Velocity vector [vx, vy, vz] in m/s, vz=0 per 3GPP spec")
        .def_rw("num_ray", &LinkLevelConfig::num_ray,
                      "Number of rays per path (default: 48 for TDL, 20 for CDL)")
        .def_rw("cfo_hz", &LinkLevelConfig::cfo_hz,
                      "Carrier frequency offset in Hz")
        .def_rw("delay", &LinkLevelConfig::delay,
                      "Delay in seconds");

    // Bind SimConfig struct
    nb::class_<SimConfig>(m, "SimConfig", "Test configuration parameters")
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<float, float, int>(),
             "Initialize with basic parameters",
             nb::arg("center_freq_hz"), nb::arg("bandwidth_hz"), nb::arg("run_mode") = 0)
        .def(nb::init<float, float, float, int, int>(),
             "Initialize with detailed parameters",
             nb::arg("center_freq_hz"), nb::arg("bandwidth_hz"), nb::arg("sc_spacing_hz"),
             nb::arg("fft_size"), nb::arg("run_mode") = 0)
        .def("__init__", [](SimConfig* self, int link_sim_ind, float center_freq_hz, float bandwidth_hz, float sc_spacing_hz, int fft_size, int n_prb, int n_prbg, int n_snapshot_per_slot, int run_mode, int internal_memory_mode, int freq_convert_type, int sc_sampling, int proc_sig_freq, int optional_cfr_dim, int cpu_only_mode) {
                 new (self) SimConfig(link_sim_ind, center_freq_hz, bandwidth_hz, sc_spacing_hz, fft_size, n_prb, n_prbg, n_snapshot_per_slot, run_mode, internal_memory_mode, freq_convert_type, sc_sampling, nullptr, proc_sig_freq, optional_cfr_dim, cpu_only_mode);
             },
             "Initialize with full parameters",
             nb::arg("link_sim_ind"), nb::arg("center_freq_hz"), nb::arg("bandwidth_hz"),
             nb::arg("sc_spacing_hz"), nb::arg("fft_size"), nb::arg("n_prb"), nb::arg("n_prbg"),
             nb::arg("n_snapshot_per_slot"), nb::arg("run_mode"), nb::arg("internal_memory_mode"),
             nb::arg("freq_convert_type"), nb::arg("sc_sampling"), nb::arg("proc_sig_freq") = 0, nb::arg("optional_cfr_dim") = 0, nb::arg("cpu_only_mode") = 0)
        .def_rw("link_sim_ind", &SimConfig::link_sim_ind,
                      "Link simulation indicator")
        .def_rw("center_freq_hz", &SimConfig::center_freq_hz,
                      "Center frequency in Hz")
        .def_rw("bandwidth_hz", &SimConfig::bandwidth_hz,
                      "Bandwidth in Hz")
        .def_rw("sc_spacing_hz", &SimConfig::sc_spacing_hz,
                      "Subcarrier spacing in Hz")
        .def_rw("fft_size", &SimConfig::fft_size,
                      "FFT size")
        .def_rw("n_prb", &SimConfig::n_prb,
                      "Number of PRBs")
        .def_rw("n_prbg", &SimConfig::n_prbg,
                      "Number of PRB groups")
        .def_rw("n_snapshot_per_slot", &SimConfig::n_snapshot_per_slot,
                      "Channel realizations per slot (1 or 14)")
        .def_rw("run_mode", &SimConfig::run_mode,
                      "Run mode: 0=CIR only, 1=CIR+CFR on PRBG, 2=CIR+CFR on PRB/SC")
        .def_rw("internal_memory_mode", &SimConfig::internal_memory_mode,
                      "Memory mode: 0=external, 1=internal")
        .def_rw("freq_convert_type", &SimConfig::freq_convert_type,
                      "Frequency conversion type for CFR on SC to PRBG")
        .def_rw("sc_sampling", &SimConfig::sc_sampling,
                      "Subcarrier sampling within PRBG")
        .def_rw("proc_sig_freq", &SimConfig::proc_sig_freq,
                      "Signal processing frequency indicator")
        .def_rw("optional_cfr_dim", &SimConfig::optional_cfr_dim,
                      "Optional CFR dimension")
        .def_rw("cpu_only_mode", &SimConfig::cpu_only_mode,
                      "CPU only mode: 0=GPU mode, 1=CPU only mode")
        .def_rw("h5_dump_level", &SimConfig::h5_dump_level,
                      "H5 dump level: 0=minimal (topology+CIR/CFR+config), 1=full (default)");

    // Bind ExternalConfig struct
    nb::class_<ExternalConfig>(m, "ExternalConfig", "External configuration parameters")
        .def(nb::init<>(), "Default constructor")
        .def("__init__", [](ExternalConfig* self, const std::vector<CellParam>& cell_config, const std::vector<UtParamCfg>& ut_config, const std::vector<AntPanelConfig>& ant_panel_config) {
                 new (self) ExternalConfig();
                 self->cell_config = cell_config;
                 self->ut_config = ut_config;
                 self->ant_panel_config = ant_panel_config;
             },
             "Initialize with parameters (without ST config)",
             nb::arg("cell_config"), nb::arg("ut_config"), nb::arg("ant_panel_config"))
        .def_rw("cell_config", &ExternalConfig::cell_config,
                      "Cell configuration list")
        .def_rw("ut_config", &ExternalConfig::ut_config,
                      "UT configuration list")
        .def_rw("ant_panel_config", &ExternalConfig::ant_panel_config,
                      "Antenna panel configuration list")
        .def_rw("st_config", &ExternalConfig::st_config,
                      "Sensing target (ST) configuration list for ISAC");

    // Per-link system-level snapshots. StatisChanModel returns copies, so these
    // instances remain valid after the model is run again or destroyed.
    nb::class_<LinkParams>(m, "LinkParams", "Per-link large-scale channel parameters")
        .def_ro("d2d", &LinkParams::d2d)
        .def_ro("d2d_in", &LinkParams::d2d_in)
        .def_ro("d2d_out", &LinkParams::d2d_out)
        .def_ro("d3d", &LinkParams::d3d)
        .def_ro("d3d_in", &LinkParams::d3d_in)
        .def_ro("d3d_out", &LinkParams::d3d_out)
        .def_ro("phi_LOS_AOD", &LinkParams::phi_LOS_AOD)
        .def_ro("theta_LOS_ZOD", &LinkParams::theta_LOS_ZOD)
        .def_ro("phi_LOS_AOA", &LinkParams::phi_LOS_AOA)
        .def_ro("theta_LOS_ZOA", &LinkParams::theta_LOS_ZOA)
        .def_ro("losInd", &LinkParams::losInd)
        .def_ro("pathloss", &LinkParams::pathloss)
        .def_ro("SF", &LinkParams::SF)
        .def_ro("K", &LinkParams::K, "Ricean K-factor in dB")
        .def_ro("DS", &LinkParams::DS)
        .def_ro("ASD", &LinkParams::ASD)
        .def_ro("ASA", &LinkParams::ASA)
        .def_ro("mu_lgZSD", &LinkParams::mu_lgZSD)
        .def_ro("sigma_lgZSD", &LinkParams::sigma_lgZSD)
        .def_ro("mu_offset_ZOD", &LinkParams::mu_offset_ZOD)
        .def_ro("ZSD", &LinkParams::ZSD)
        .def_ro("ZSA", &LinkParams::ZSA)
        .def_ro("delta_tau", &LinkParams::delta_tau)
        .def_ro("h_e", &LinkParams::h_e);

    nb::class_<ClusterParams>(m, "ClusterParams", "Per-link small-scale cluster parameters")
        .def_ro("nCluster", &ClusterParams::nCluster)
        .def_ro("nRayPerCluster", &ClusterParams::nRayPerCluster)
        .def_prop_ro("delays", [](const ClusterParams& params) {
            return std::vector<float>(params.delays, params.delays + params.nCluster);
        })
        .def_prop_ro("powers", [](const ClusterParams& params) {
            return std::vector<float>(params.powers, params.powers + params.nCluster);
        })
        .def_prop_ro("strongest2clustersIdx", [](const ClusterParams& params) {
            return std::vector<uint16_t>(params.strongest2clustersIdx,
                                         params.strongest2clustersIdx + 2);
        })
        .def_prop_ro("phi_n_AoA", [](const ClusterParams& params) {
            return std::vector<float>(params.phi_n_AoA, params.phi_n_AoA + params.nCluster);
        })
        .def_prop_ro("phi_n_AoD", [](const ClusterParams& params) {
            return std::vector<float>(params.phi_n_AoD, params.phi_n_AoD + params.nCluster);
        })
        .def_prop_ro("theta_n_ZOD", [](const ClusterParams& params) {
            return std::vector<float>(params.theta_n_ZOD, params.theta_n_ZOD + params.nCluster);
        })
        .def_prop_ro("theta_n_ZOA", [](const ClusterParams& params) {
            return std::vector<float>(params.theta_n_ZOA, params.theta_n_ZOA + params.nCluster);
        })
        .def_prop_ro("xpr", [](const ClusterParams& params) {
            const size_t n_rays = static_cast<size_t>(params.nCluster) * params.nRayPerCluster;
            return std::vector<float>(params.xpr, params.xpr + n_rays);
        })
        .def_prop_ro("randomPhases", [](const ClusterParams& params) {
            const size_t n_phases =
                static_cast<size_t>(params.nCluster) * params.nRayPerCluster * 4;
            nb::list sectors;
            for (size_t sector = 0; sector < ClusterParams::MAX_SECTORS; ++sector) {
                const float* start =
                    params.randomPhases + sector * ClusterParams::PHASE_SECTOR_STRIDE;
                sectors.append(std::vector<float>(start, start + n_phases));
            }
            return sectors;
        })
        .def_prop_ro("phi_n_m_AoA", [](const ClusterParams& params) {
            const size_t n_rays = static_cast<size_t>(params.nCluster) * params.nRayPerCluster;
            return std::vector<float>(params.phi_n_m_AoA, params.phi_n_m_AoA + n_rays);
        })
        .def_prop_ro("phi_n_m_AoD", [](const ClusterParams& params) {
            const size_t n_rays = static_cast<size_t>(params.nCluster) * params.nRayPerCluster;
            return std::vector<float>(params.phi_n_m_AoD, params.phi_n_m_AoD + n_rays);
        })
        .def_prop_ro("theta_n_m_ZOD", [](const ClusterParams& params) {
            const size_t n_rays = static_cast<size_t>(params.nCluster) * params.nRayPerCluster;
            return std::vector<float>(params.theta_n_m_ZOD, params.theta_n_m_ZOD + n_rays);
        })
        .def_prop_ro("theta_n_m_ZOA", [](const ClusterParams& params) {
            const size_t n_rays = static_cast<size_t>(params.nCluster) * params.nRayPerCluster;
            return std::vector<float>(params.theta_n_m_ZOA, params.theta_n_m_ZOA + n_rays);
        });

    // Bind StatisChanModelWrapper class
    nb::class_<gpu3gppchan_bindings::StatisChanModelWrapper<float, cuComplex>>(m, "StatisChanModel",
                                                                 "Stochastic channel model wrapper class")
        .def("__init__", [](gpu3gppchan_bindings::StatisChanModelWrapper<float, cuComplex>* self,
                            const SimConfig& simConfig,
                            const SystemLevelConfig& systemConfig,
                            const LinkLevelConfig& linkConfig,
                            const ExternalConfig& externalConfig,
                            uint32_t seed,
                            const nb::object& stream) {
                new (self) gpu3gppchan_bindings::StatisChanModelWrapper<float, cuComplex>(
                    simConfig, systemConfig, linkConfig, externalConfig, seed,
                    gpu3gppchan_bindings::stream_handle_from_owner(stream));
             },
             "Initialize channel model with all configuration parameters",
             nb::arg("sim_config"), nb::arg("system_level_config"),
             nb::arg("link_level_config"), nb::arg("external_config"),
             nb::arg("rand_seed"), nb::arg("stream_handle"), nb::keep_alive<1, 7>())
        .def("__init__", [](gpu3gppchan_bindings::StatisChanModelWrapper<float, cuComplex>* self,
                            const SimConfig& simConfig,
                            const SystemLevelConfig& systemConfig,
                            uint32_t seed,
                            const nb::object& stream) {
                new (self) gpu3gppchan_bindings::StatisChanModelWrapper<float, cuComplex>(
                    simConfig, systemConfig, seed,
                    gpu3gppchan_bindings::stream_handle_from_owner(stream));
             },
             "Initialize channel model with minimal configuration parameters",
             nb::arg("sim_config"), nb::arg("system_level_config"),
             nb::arg("rand_seed"), nb::arg("stream_handle"), nb::keep_alive<1, 5>())
        .def("reset", &gpu3gppchan_bindings::StatisChanModelWrapper<float, cuComplex>::reset,
             nb::lock_self(),
             "Reset the channel model state")
        .def("run", &gpu3gppchan_bindings::StatisChanModelWrapper<float, cuComplex>::run,
             nb::lock_self(),
             "Run system-level channel model simulation",
             nb::arg("ref_time") = 0.0f,
             nb::arg("continuous_fading") = 1,
             nb::arg("active_cell") = nb::none(),
             nb::arg("active_ut") = nb::none(),
             nb::arg("ut_new_loc") = nb::none(),
             nb::arg("ut_new_velocity") = nb::none(),
             nb::arg("cir_coe") = nb::none(),
             nb::arg("cir_norm_delay") = nb::none(),
             nb::arg("cir_n_taps") = nb::none(),
             nb::arg("cfr_sc") = nb::none(),
             nb::arg("cfr_prbg") = nb::none())
        .def("get_cir", &gpu3gppchan_bindings::StatisChanModelWrapper<float, cuComplex>::get_cir,
             nb::lock_self(),
             "Copy internally stored CIR data into caller-provided buffers.",
             nb::arg("cir_coe") = nb::none(),
             nb::arg("cir_norm_delay") = nb::none(),
             nb::arg("cir_n_taps") = nb::none())
        .def("get_cfr", &gpu3gppchan_bindings::StatisChanModelWrapper<float, cuComplex>::get_cfr,
             nb::lock_self(),
             "Copy internally stored CFR data into caller-provided buffers.",
             nb::arg("cfr_sc") = nb::none(),
             nb::arg("cfr_prbg") = nb::none())
        .def("run_link_level", &gpu3gppchan_bindings::StatisChanModelWrapper<float, cuComplex>::run_link_level,
             nb::lock_self(),
             "Run link-level channel model simulation",
             nb::arg("ref_time0") = 0.0f,
             nb::arg("continuous_fading") = 1,
             nb::arg("enable_swap_tx_rx") = 0,
             nb::arg("tx_column_major_ind") = 0)
        .def("dump_topology_to_yaml", &gpu3gppchan_bindings::StatisChanModelWrapper<float, cuComplex>::dump_topology_to_yaml,
             nb::lock_self(),
             "Dump topology to YAML file",
             nb::arg("filename"))
        .def("set_los_override", &gpu3gppchan_bindings::StatisChanModelWrapper<float, cuComplex>::set_los_override,
             nb::lock_self(),
             "Set one LOS override per site-UT link (1=LOS, 0=NLOS, 255=model draw); pass None to clear.",
             nb::arg("los_ind") = nb::none())
        .def("clear_los_override", &gpu3gppchan_bindings::StatisChanModelWrapper<float, cuComplex>::clear_los_override,
             nb::lock_self(),
             "Clear all LOS/NLOS overrides.")
        .def("get_num_site_ut_links", &gpu3gppchan_bindings::StatisChanModelWrapper<float, cuComplex>::get_num_site_ut_links,
             nb::lock_self(),
             "Return the number of site-UT links.")
        .def("get_link_params_host", &gpu3gppchan_bindings::StatisChanModelWrapper<float, cuComplex>::get_link_params_host,
             nb::lock_self(),
             "Return copied per-link LinkParams snapshots after run().")
        .def("get_cluster_params_host", &gpu3gppchan_bindings::StatisChanModelWrapper<float, cuComplex>::get_cluster_params_host,
             nb::lock_self(),
             "Return copied per-link ClusterParams snapshots after run().")
        .def("save_sls_chan_to_h5_file", &gpu3gppchan_bindings::StatisChanModelWrapper<float, cuComplex>::saveSlsChanToH5File,
             nb::lock_self(),
             "Save SLS channel data to H5 file for debugging",
             nb::arg("filename_ending") = "")
        .def("dump_los_nlos_stats", &gpu3gppchan_bindings::StatisChanModelWrapper<float, cuComplex>::dump_los_nlos_stats,
             nb::lock_self(),
             "Dump LOS/NLOS statistics for all links",
             nb::arg("los_nlos_stats") = nb::none())
        .def("dump_pl_sf_stats", &gpu3gppchan_bindings::StatisChanModelWrapper<float, cuComplex>::dump_pl_sf_stats,
             nb::lock_self(),
             "Dump pathloss and shadowing statistics for links",
             nb::arg("pl_sf"),
             nb::arg("active_cell") = nb::none(),
             nb::arg("active_ut") = nb::none())
        .def("dump_pl_sf_ant_gain_stats", &gpu3gppchan_bindings::StatisChanModelWrapper<float, cuComplex>::dump_pl_sf_ant_gain_stats,
             nb::lock_self(),
             "Dump pathloss, shadowing and antenna gain statistics",
             nb::arg("pl_sf_ant_gain"),
             nb::arg("active_cell") = nb::none(),
             nb::arg("active_ut") = nb::none());

    nb::class_<gpu3gppchan_bindings::GauNoiseAdderWrapper<float, cuComplex>>(m, "GauNoiseAdder")
        .def("__init__", [](gpu3gppchan_bindings::GauNoiseAdderWrapper<float, cuComplex>* self,
                            uint32_t threads, int seed, const nb::object& stream) {
                new (self) gpu3gppchan_bindings::GauNoiseAdderWrapper<float, cuComplex>(
                    threads, seed, gpu3gppchan_bindings::stream_handle_from_owner(stream));
            }, nb::arg("num_threads"), nb::arg("rand_seed"), nb::arg("stream_handle"),
            nb::keep_alive<1, 4>())
        .def("add_noise", &gpu3gppchan_bindings::GauNoiseAdderWrapper<float, cuComplex>::addNoise,
            nb::lock_self(),
            nb::arg("d_signal"), nb::arg("signal_size"), nb::arg("snr_db"),
            "Add Gaussian noise in-place on GPU");
}
