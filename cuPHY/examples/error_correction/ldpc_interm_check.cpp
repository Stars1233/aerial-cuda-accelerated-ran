/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Per-iter APP comparison for LDPC decoders.
//   run_llr_check : decode once, compare DUT against REF.
//   run_llr_diff  : decode twice (algoA, algoB), compare DUT-vs-DUT in memory.
// Shared helpers:
//   load_tv: TV -> host
//   capture_app_device: device decode -> host APP_history
//   decode_capture: TV wrapper around capture_app_device

#include "ldpc_interm_check.hpp"
#include "cuphy.hpp"
#include "ldpc/ldpc_api.hpp"
#include "cuphy_hdf5.hpp"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cuphy;

// APP(var, iter, cw) where var is 1st storage dim.
static int APP_offset(int cw, int iter, int NUM_ITER, int NUM_VAR)
{
    return cw * NUM_ITER * NUM_VAR + iter * NUM_VAR;
}

bool APP_history_is_empty(const std::vector<float>& h_APP)
{
    for(float v : h_APP)
    {
        if(v != 0.0f)
        {
            return false;
        }
    }
    return true;
}

static void require_APP_history_dumped(const std::vector<float>& h_APP, int algo)
{
    if(APP_history_is_empty(h_APP))
    {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "algo=%d captured APP_history is all zeros; kernel likely did not run a dumpApp path, "
                 "so --llr_check/--llr_diff would compare an empty dump.",
                 algo);
        throw std::runtime_error(msg);
    }
}

static int resolve_check_max_iter(int max_iter, int tv_max_iter)
{
    if(max_iter <= 0)
    {
        return tv_max_iter;
    }
    if(max_iter > tv_max_iter)
    {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "requested max_iter=%d exceeds TV APP_history maxItr=%d",
                 max_iter, tv_max_iter);
        throw std::runtime_error(msg);
    }
    return max_iter;
}

static std::vector<float> compact_APP_prefix(const std::vector<float>& h_APP,
                                             int C, int src_num_iter, int dst_num_iter, int NUM_VAR)
{
    std::vector<float> out((size_t)C * dst_num_iter * NUM_VAR);
    for(int cw = 0; cw < C; ++cw)
    {
        const float* src = h_APP.data() + APP_offset(cw, 0, src_num_iter, NUM_VAR);
        float*       dst = out.data()   + APP_offset(cw, 0, dst_num_iter, NUM_VAR);
        std::copy(src, src + (size_t)dst_num_iter * NUM_VAR, dst);
    }
    return out;
}

// Per-iteration APP comparison of DUT vs REF, with three pass gates:
//   p99err_rms = pctile99|ref-dut| / rms_ref   must be <= p99err_rms_threshold   (p99_fail)
//              (robust worst-case error; pkerr_rms = max is printed as a diagnostic only)
//   SNR(dB)    = 10*log10(||ref||^2 / ||ref-dut||^2)   must be >= snr_threshold  (snr_fail)
//   #sign_flip must be 0 in the LAST 3 iterations   (decision-stability gate; sf_fail)
// A row fails if ANY gate fails; overall pass requires every row to pass all three.
// max_APP/min_APP report the REFERENCE APP range to ground the error columns.
bool check_APP(
    const float* h_APP_ref,
    const float* h_APP_dut,
    int C, int NUM_ITER, int NUM_VAR, int N_var_checked,
    float p99err_rms_threshold, float snr_threshold)
{
    bool pass = true;
    printf("Per-iteration pass requires ALL gates (p99_fail / snr_fail / sf_fail mark which failed):\n");
    printf("  p99err_rms = pctile99|ref-dut| / rms_ref         must be <= p99err_rms_threshold = %g   (gate; pkerr_rms=max is diagnostic)\n",
           p99err_rms_threshold);
    printf("  SNR(dB)    = 10*log10(||ref||^2 / ||ref-dut||^2) must be >= snr_threshold       = %g dB\n",
           snr_threshold);
    printf("  #sign_flip must be 0 in the LAST 3 iterations    (decision-stability gate: sf_fail)\n");
    printf("Columns: max_abs_err = max|ref-dut|, rms_ref = RMS(ref), pkerr_rms = max_abs_err / rms_ref,\n");
    printf("         p99err_rms = pctile99|ref-dut| / rms_ref (robust to a few saturated columns)\n");
    std::vector<float> abs_errs(N_var_checked > 0 ? N_var_checked : 1);  // reused per row for p99
    for(int cw = 0; cw < C; ++cw)
    {
        printf("\n  CB%d:\n", cw);
        printf("  Iter | #sign_flip | max_abs_err |  rms_ref  | pkerr_rms | p99err_rms |   max_APP   |   min_APP   |  SNR(dB) | p99_fail | snr_fail | sf_fail\n");
        printf("  -----+------------+-------------+-----------+-----------+------------+-------------+-------------+----------+---------+---------+---------\n");
        for(int iter = 0; iter < NUM_ITER; ++iter)
        {
            int    num_sign_flip = 0;
            float  max_abs       = 0.0f;       // largest |ref-dut| over checked vars
            float  max_app       = -INFINITY;  // max reference APP over checked vars
            float  min_app       = +INFINITY;  // min reference APP over checked vars
            double sig_l2sq      = 0.0;        // ||ref||^2
            double err_l2sq      = 0.0;        // ||ref-dut||^2
            const float* ref = h_APP_ref + APP_offset(cw, iter, NUM_ITER, NUM_VAR);
            const float* dut = h_APP_dut + APP_offset(cw, iter, NUM_ITER, NUM_VAR);
            for(int n = 0; n < N_var_checked; ++n)
            {
                float err     = ref[n] - dut[n];
                float abs_err = std::fabs(err);
                abs_errs[n]   = abs_err;
                max_abs       = std::max(max_abs, abs_err);
                max_app       = std::max(max_app, ref[n]);
                min_app       = std::min(min_app, ref[n]);
                sig_l2sq     += (double)ref[n] * (double)ref[n];
                err_l2sq     += (double)err * (double)err;
                if(ref[n] * dut[n] < 0.0f) { ++num_sign_flip; }
            }
            double snr_db;
            if(err_l2sq == 0.0)      { snr_db = INFINITY; }   // bit-exact: no error energy
            else if(sig_l2sq == 0.0) { snr_db = -INFINITY; }  // no signal but nonzero error
            else                     { snr_db = 10.0 * std::log10(sig_l2sq / err_l2sq); }

            // rms_ref   = RMS of the reference APP (signal scale for this iter);
            // pkerr_rms = worst-case error as a scale-invariant fraction of that scale.
            double rms_ref = (N_var_checked > 0) ? std::sqrt(sig_l2sq / (double)N_var_checked) : 0.0;
            double pkerr_rms;
            if(rms_ref > 0.0)       { pkerr_rms = (double)max_abs / rms_ref; }
            else if(max_abs > 0.0f) { pkerr_rms = INFINITY; }  // no signal but nonzero error
            else                    { pkerr_rms = 0.0; }

            // p99err_rms: 99th-percentile |err| / rms_ref -- robust to a few saturated
            // outliers that spike max_abs at high p. nth_element is O(N) (no full sort).
            double p99err_rms;
            {
                float p99_abs = 0.0f;
                if(N_var_checked > 0)
                {
                    int k = (int)std::ceil(0.99 * N_var_checked) - 1;
                    if(k < 0)              { k = 0; }
                    if(k >= N_var_checked) { k = N_var_checked - 1; }
                    std::nth_element(abs_errs.begin(), abs_errs.begin() + k,
                                     abs_errs.begin() + N_var_checked);
                    p99_abs = abs_errs[k];
                }
                if(rms_ref > 0.0)       { p99err_rms = (double)p99_abs / rms_ref; }
                else if(p99_abs > 0.0f) { p99err_rms = INFINITY; }
                else                    { p99err_rms = 0.0; }
            }

            const bool in_last3 = (iter >= NUM_ITER - 3);                        // decision-stability window
            const bool p99_fail  = !(p99err_rms <= (double)p99err_rms_threshold);  // gate on p99, not the max
            const bool snr_fail = !(snr_db     >= (double)snr_threshold);
            const bool sf_fail  = in_last3 && (num_sign_flip > 0);               // strict: no flips in last 3 iters
            if(p99_fail || snr_fail || sf_fail) { pass = false; }

            char pkerr_str[24];
            if(std::isinf(pkerr_rms)) { snprintf(pkerr_str, sizeof(pkerr_str), "%9s", "inf"); }
            else                      { snprintf(pkerr_str, sizeof(pkerr_str), "%9.5f", pkerr_rms); }

            char p99_str[24];
            if(std::isinf(p99err_rms)) { snprintf(p99_str, sizeof(p99_str), "%10s", "inf"); }
            else                       { snprintf(p99_str, sizeof(p99_str), "%10.5f", p99err_rms); }

            char snr_str[24];
            if(std::isinf(snr_db)) { snprintf(snr_str, sizeof(snr_str), "%8s", snr_db > 0 ? "inf" : "-inf"); }
            else                   { snprintf(snr_str, sizeof(snr_str), "%8.2f", snr_db); }

            // sf_fail only applies to the last 3 iters; earlier iters print "-" (not gated).
            const char* sf_str = !in_last3 ? "-" : (num_sign_flip > 0 ? "FAIL" : "ok");

            printf("  %3d  |  %8d  | %11.6f | %9.4f | %s | %s | %+11.6f | %+11.6f | %s | %7s | %7s | %7s\n",
                   iter, num_sign_flip, max_abs, rms_ref, pkerr_str, p99_str, max_app, min_app, snr_str,
                   p99_fail ? "FAIL" : "ok", snr_fail ? "FAIL" : "ok", sf_str);
        }
    }
    printf("\nImportant: APP of extension clms (beyond core 4 parities) is NOT checked\n\n");
    return pass;
}

// ----------------------------------------------------------------------------
// Shared helpers
// ----------------------------------------------------------------------------

static inline bool is_fp8_llr(cuphyDataType_t llr_type)
{
    (void)llr_type;
    return false;
}

static inline void check_llr_type(cuphyDataType_t llr_type)
{
    if(llr_type != CUPHY_R_16F && llr_type != CUPHY_R_32F)
    {
        throw std::runtime_error(std::string("unsupported LLR type for the APP check: ") +
                                 cuphyGetDataTypeString(llr_type));
    }
}

// Load TV params + inputLLR + sourceData from the TV HDF5. Returns false (and
// prints err_bitmap) if sanity_check fails; both callers early-return on false.
// The RM utility in this branch supports fp16/fp32 LLR input, so sourceData is
// used as the reference decoded bits.
// Read a whole dataset into dst, rejecting a size that disagrees with the
// element count derived from /params rather than letting H5Dread overrun.
static void read_dataset_checked(hdf5hpp::hdf5_file& f,
                                 const char*         name,
                                 hid_t               h5_type,
                                 void*               dst,
                                 size_t              n_expected)
{
    hdf5hpp::hdf5_dataset ds       = f.open_dataset(name);
    const hsize_t         n_actual = ds.get_dataspace().get_num_elements();
    if(n_actual != n_expected)
    {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "dataset /%s has %llu elements, expected %zu from /params",
                      name, static_cast<unsigned long long>(n_actual), n_expected);
        throw std::runtime_error(msg);
    }
    if(H5Dread(ds.id(), h5_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, dst) < 0)
    {
        throw std::runtime_error(std::string("failed to read dataset /") + name);
    }
}

static bool load_tv(const std::string&    tv_path,
                    LdpcTvParams&         p,
                    std::vector<float>&   h_LLR_f32,
                    std::vector<uint8_t>& h_src,
                    cuphyDataType_t       llr_type)
{
    hdf5hpp::hdf5_file tv_file = hdf5hpp::hdf5_file::open(tv_path.c_str());
    p.read(tv_file);
    uint32_t err = p.sanity_check();
    if(err)
    {
        p.print();
        fprintf(stderr, "LdpcTvParams sanity check failed: err_bitmap=0x%x\n", err);
        return false;
    }
    p.print();

    const int Kb      = (p.BGN == 1) ? 22 : 10;
    const int NUM_VAR = p.Zc * (Kb + p.nV_parity);
    h_LLR_f32.resize((size_t)p.C * NUM_VAR);
    h_src.resize((size_t)p.C * p.K);

    check_llr_type(llr_type);
    const char* src_name = is_fp8_llr(llr_type) ? "sourceData_fp8" : "sourceData";
    printf("REF datasets: %s (sourceData)\n", src_name);
    read_dataset_checked(tv_file, "inputLLR", H5T_NATIVE_FLOAT, h_LLR_f32.data(), h_LLR_f32.size());
    read_dataset_checked(tv_file, src_name,   H5T_NATIVE_UINT8, h_src.data(),     h_src.size());
    return true;
}

void capture_app_device(cuphy::context&           ctx,
                        const cuphy::tensor_desc& llr_desc,
                        void*                     llr_addr,
                        int                       BGN,
                        int                       Zc,
                        int                       nV_parity,
                        int                       K,
                        int                       C,
                        int                       maxItr,
                        float                     clamp,
                        cuphyDataType_t           llr_type,
                        int                       algo,
                        int                       max_iter,
                        std::vector<float>&       h_APP_out,
                        std::vector<uint8_t>*     h_dec_out)
{
    const int Kb       = (BGN == 1) ? 22 : 10;
    const int NUM_VAR  = Zc * (Kb + nV_parity);
    const int NUM_ITER = resolve_check_max_iter(max_iter, maxItr);

    // to pass initcheck when padding exists
    cuphy::tensor_device tDecode(CUPHY_BIT, K, C,
                                 cuphy::tensor_flags::align_coalesce);
    CUDA_CHECK_EXCEPTION(cudaMemset(tDecode.addr(), 0, tDecode.desc().get_size_in_bytes()));

    LDPC_decode_config dec_cfg(llr_type,
                               nV_parity,
                               Zc,
                               NUM_ITER,
                               clamp,
                               Kb,
                               0.0f,  // norm: 0 -> set_normalization() picks it from BGN + nV_parity
                               CUPHY_LDPC_DECODE_DUMP_INTERM,
                               BGN,
                               algo,
                               nullptr);
    LDPC_decoder dec(ctx);
    dec.set_normalization(dec_cfg);
    printf("Normalization = %f\n", dec_cfg.get_norm());

    // APP-history device buffer, tight stride [C, maxItr, NUM_VAR].
    // These decoder dump paths write fp16 APP values.
    const int    stride_itr = NUM_VAR;
    const int    stride_cw  = NUM_ITER * stride_itr;
    const size_t elem_bytes = sizeof(__half);
    const size_t buf_bytes  = (size_t)C * stride_cw * elem_bytes;
    void* d_APP_history = nullptr;
    CUDA_CHECK_EXCEPTION(cudaMalloc(&d_APP_history, buf_bytes));
    // Resource Acquisition Is Initialization guard: frees d_APP_history on every exit path
    // (normal return or exception). Callers catch exceptions and continue the sweep, so a
    // leaked alloc per failed cell would eventually exhaust device memory.
    std::unique_ptr<void, decltype(&cudaFree)> d_APP_history_guard(d_APP_history, cudaFree);
    CUDA_CHECK_EXCEPTION(cudaMemset(d_APP_history, 0, buf_bytes));

    LDPC_decode_desc dec_desc(dec_cfg, CUPHY_LDPC_DECODE_DESC_MAX_TB);
    dec_desc.add_tensor_as_tb(llr_desc, llr_addr,
                              tDecode.desc(), tDecode.addr());
    cuphyTransportBlockIntermResults_t h_interm{};
    h_interm.app_addr                = d_APP_history;
    h_interm.app_stride_elements_cw  = stride_cw;
    h_interm.app_stride_elements_itr = stride_itr;
    h_interm.num_codewords           = C;

    cuphyTransportBlockIntermResults_t* d_interm = nullptr;
    CUDA_CHECK_EXCEPTION(cudaMalloc(&d_interm, sizeof(h_interm)));
    std::unique_ptr<void, decltype(&cudaFree)> d_interm_guard(d_interm, cudaFree);
    CUDA_CHECK_EXCEPTION(cudaMemcpy(d_interm, &h_interm, sizeof(h_interm), cudaMemcpyHostToDevice));
    dec_desc.interm_results = d_interm;

    // Pre-zero h_APP_out now (before decode) so any failure path -- kernel launch
    // rejection, exception, or a kernel that silently skips the dump -- leaves the
    // host buffer all-zeros.  APP_history_is_empty() in callers then catches it and
    // reports N/A rather than a spurious PASS from stale data in the same allocation.
    const size_t num_elems_pre = (size_t)C * stride_cw;
    h_APP_out.assign(num_elems_pre, 0.0f);

    printf("Running decoder: algo=%d  BG=%d  Zc=%d  mb=%d  C=%d  maxItr=%d  clamp=%.1f\n",
           algo, BGN, Zc, nV_parity, C, NUM_ITER, clamp);
    dec.decode(dec_desc, nullptr);
    CUDA_CHECK_EXCEPTION(cudaStreamSynchronize(nullptr));

    // Optional: hand back the packed decoded bits for a caller-side BER/BLER.
    if(h_dec_out)
    {
        const size_t total_bytes = tDecode.desc().get_size_in_bytes();
        h_dec_out->resize(total_bytes);
        CUDA_CHECK_EXCEPTION(cudaMemcpy(h_dec_out->data(), tDecode.addr(),
                                        total_bytes, cudaMemcpyDeviceToHost));
    }

    // D2H APP history. fp16 -> fp32.
    const size_t num_elems = (size_t)C * stride_cw;
    h_APP_out.resize(num_elems);
    std::vector<__half> h_APP_fp16(num_elems);
    CUDA_CHECK_EXCEPTION(cudaMemcpy(h_APP_fp16.data(), d_APP_history, buf_bytes,
                                    cudaMemcpyDeviceToHost));
    for(size_t i = 0; i < num_elems; ++i)
    {
        h_APP_out[i] = __half2float(h_APP_fp16[i]);
    }
}

static const char* cuphy_llr_type_name(cuphyDataType_t t)
{
    switch(t)
    {
    case CUPHY_R_16F:     return "fp16";
    case CUPHY_R_32F:     return "fp32";
    default:              return "unknown";
    }
}

void write_dut_h5(
    const char*     filename,
    const float*    llr_init,
    int             C,
    int             N,
    const float*    app_hist,
    int             maxItr,
    cuphyDataType_t llr_type,
    int             BGN,
    int             Zc,
    int             Kb,
    int             mb,
    int             F)
{
    hid_t fid = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if(fid < 0)
    {
        fprintf(stderr, "DUT HDF5 write FAILED: cannot create %s\n", filename);
        return;
    }
    bool ok = true;

    auto write_ds = [&](const char* name, const float* data,
                        int ndim, const hsize_t* dims)
    {
        hid_t sid = H5Screate_simple(ndim, dims, nullptr);
        hid_t did = H5Dcreate(fid, name, H5T_NATIVE_FLOAT, sid,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if(sid < 0 || did < 0 ||
           H5Dwrite(did, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0)
        {
            ok = false;
        }
        H5Dclose(did);
        H5Sclose(sid);
    };

    const hsize_t dims2[2] = {(hsize_t)C, (hsize_t)N};
    write_ds("inputLLR", llr_init, 2, dims2);

    const hsize_t dims3[3] = {(hsize_t)C, (hsize_t)maxItr, (hsize_t)N};
    write_ds("APP_history", app_hist, 3, dims3);

    // /llr_type : scalar variable-length string so plot tooling can auto-pick
    // the matching REF (FP8 vs FP16) from the TV.
    {
        const char* type_str = cuphy_llr_type_name(llr_type);
        hid_t       tid      = H5Tcopy(H5T_C_S1);
        H5Tset_size(tid, H5T_VARIABLE);
        hid_t sid = H5Screate(H5S_SCALAR);
        hid_t did = H5Dcreate(fid, "llr_type", tid, sid,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if(tid < 0 || sid < 0 || did < 0 ||
           H5Dwrite(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, &type_str) < 0)
        {
            ok = false;
        }
        H5Dclose(did);
        H5Sclose(sid);
        H5Tclose(tid);
    }

    // /params-lite: structural scalars so the plotter can mark the core-column
    // boundary. 0 = not provided -> plotter falls back to shape.
    if(BGN > 0)
    {
        auto write_i = [&](const char* name, int v)
        {
            hid_t sid = H5Screate(H5S_SCALAR);
            hid_t did = H5Dcreate(fid, name, H5T_NATIVE_INT, sid,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if(sid < 0 || did < 0 ||
               H5Dwrite(did, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v) < 0)
            {
                ok = false;
            }
            H5Dclose(did);
            H5Sclose(sid);
        };
        write_i("BGN", BGN);
        write_i("Zc",  Zc);
        write_i("Kb",  Kb);
        write_i("mb",  mb);
        write_i("F",   F);
    }

    if(H5Fclose(fid) < 0)
    {
        ok = false;
    }
    if(ok)
    {
        printf("DUT HDF5 written: %s (llr_type=%s)\n", filename, cuphy_llr_type_name(llr_type));
    }
    else
    {
        fprintf(stderr, "DUT HDF5 write FAILED (an HDF5 op failed): %s\n", filename);
    }
}

// Decode one (algo, llr_type) on the already-loaded TV data; capture APP_history
// to host (fp32), write DUT HDF5 (inputLLR + APP_history), and print BER/BLER vs
// sourceData. Thin TV wrapper around capture_app_device().
// inputLLR -> device tensor. H5 (from Matlab) and cuPHY differ in stride, so this
// goes through tensor_from_dataset rather than a memcpy of the host copy.
static cuphy::tensor_device load_llr_device(const std::string& tv_path,
                                           cuphyDataType_t    llr_type)
{
    hdf5hpp::hdf5_file tv_file = hdf5hpp::hdf5_file::open(tv_path.c_str());
    return cuphy::tensor_from_dataset(tv_file.open_dataset("inputLLR"),
                                      llr_type,
                                      cuphy::tensor_flags::align_coalesce);
}

static void decode_capture(cuphy::context&             ctx,
                           const cuphy::tensor_device& tLLR,
                           const LdpcTvParams&         p,
                           const std::vector<float>&   h_LLR_f32,
                           const std::vector<uint8_t>& h_src,
                           cuphyDataType_t             llr_type,
                           int                         algo,
                           int                         max_iter,
                           const char*                 dut_h5_filename,
                           std::vector<float>&         h_APP_f32_out)
{
    const int Kb       = (p.BGN == 1) ? 22 : 10;
    const int NUM_VAR  = p.Zc * (Kb + p.nV_parity);
    const int NUM_ITER = resolve_check_max_iter(max_iter, p.maxItr);
    const float clamp  = p.clamp_for(llr_type);

    // H2D inputLLR
    // H5 (from Matlab): inputLLR(N, C), where N is 1st storage dim, contiguously stored.
    // cuphy: inputLLR(N, C), where N also 1st storage dim, stride on "N" is bigger than N.
    // Warning: Due to storage diff, do not do one memcpy
    std::vector<uint8_t> h_dec;
    capture_app_device(ctx, tLLR.desc(), tLLR.addr(), p.BGN, p.Zc, p.nV_parity,
                       p.K, p.C, p.maxItr, clamp, llr_type, algo, max_iter,
                       h_APP_f32_out, &h_dec);

    // BER / BLER vs sourceData
    {
        const size_t total_bytes  = h_dec.size();
        const size_t stride_bytes = total_bytes / (size_t)p.C;
        int total_bit_err   = 0;
        int total_block_err = 0;
        for(int c = 0; c < p.C; ++c)
        {
            const uint8_t* packed = h_dec.data() + c * stride_bytes;
            const uint8_t* ref    = h_src.data() + (size_t)c * p.K;
            int cw_bit_err = 0;
            for(int k = 0; k < p.K; ++k)
            {
                int dec_bit = (packed[k / 8] >> (k % 8)) & 1;
                if(dec_bit != ref[k]) { ++cw_bit_err; }
            }
            total_bit_err   += cw_bit_err;
            total_block_err += (cw_bit_err > 0) ? 1 : 0;
        }
        printf("algo=%d  BER  = %d / %d = %.4e\n", algo, total_bit_err,   p.C * p.K, (float)total_bit_err   / (p.C * p.K));
        printf("algo=%d  BLER = %d / %d = %.4e\n", algo, total_block_err, p.C,       (float)total_block_err / p.C);
    }

    write_dut_h5(dut_h5_filename,
                 h_LLR_f32.data(), p.C, NUM_VAR,
                 h_APP_f32_out.data(), NUM_ITER,
                 llr_type, p.BGN, p.Zc, Kb, p.nV_parity, 0);
}

// ----------------------------------------------------------------------------
// Entry points
// ----------------------------------------------------------------------------

// Compare an already-captured DUT APP against the TV's golden APP_history.
static bool check_APP_against_tv(const std::string&        tv_path,
                                 const LdpcTvParams&       p,
                                 cuphyDataType_t           llr_type,
                                 int                       max_iter,
                                 const std::vector<float>& h_APP_dut,
                                 float                     p99err_rms_threshold,
                                 float                     snr_threshold)
{
    const int Kb          = (p.BGN == 1) ? 22 : 10;
    const int NUM_VAR     = p.Zc * (Kb + p.nV_parity);
    const int TV_NUM_ITER = p.maxItr;
    const int NUM_ITER    = resolve_check_max_iter(max_iter, TV_NUM_ITER);

    check_llr_type(llr_type);
    const char* app_ref_name = is_fp8_llr(llr_type) ? "APP_history_fp8" : "APP_history";
    printf("REF APP_history dataset: %s\n", app_ref_name);

    hdf5hpp::hdf5_file tv_file = hdf5hpp::hdf5_file::open(tv_path.c_str());
    std::vector<float> h_APP_ref_full((size_t)p.C * TV_NUM_ITER * NUM_VAR);
    read_dataset_checked(tv_file, app_ref_name, H5T_NATIVE_FLOAT,
                         h_APP_ref_full.data(), h_APP_ref_full.size());
    std::vector<float> h_APP_ref = compact_APP_prefix(h_APP_ref_full, p.C, TV_NUM_ITER, NUM_ITER, NUM_VAR);

    const int N_core_var = (Kb + 4) * p.Zc;  // skip ext parity clms; kernels do not update them during iters
    return check_APP(h_APP_ref.data(), h_APP_dut.data(), p.C, NUM_ITER, NUM_VAR, N_core_var,
                     p99err_rms_threshold, snr_threshold);
}

bool check_interm_results(const std::string& tv_path, const std::string& dut_path,
                          cuphyDataType_t llr_type,
                          int max_iter,
                          float p99err_rms_threshold, float snr_threshold)
{
    hdf5hpp::hdf5_file tv_file = hdf5hpp::hdf5_file::open(tv_path.c_str());
    LdpcTvParams p;
    p.read(tv_file);
    if(uint32_t err = p.sanity_check())
    {
        p.print();
        fprintf(stderr, "LdpcTvParams sanity check failed: err_bitmap=0x%x\n", err);
        return false;
    }
    p.print();

    const int Kb       = (p.BGN == 1) ? 22 : 10;
    const int NUM_VAR  = p.Zc * (Kb + p.nV_parity);
    const int TV_NUM_ITER = p.maxItr;
    const int NUM_ITER = resolve_check_max_iter(max_iter, TV_NUM_ITER);

    // FP16/FP32 algos compare against APP_history. (The APP_history_fp8 branch is a
    // stub for future fp8 support -- is_fp8_llr() returns false, so it is inactive.)
    std::vector<float> h_APP_dut((size_t)p.C * NUM_ITER * NUM_VAR);
    {
        hdf5hpp::hdf5_file dut_file = hdf5hpp::hdf5_file::open(dut_path.c_str());
        read_dataset_checked(dut_file, "APP_history", H5T_NATIVE_FLOAT,
                             h_APP_dut.data(), h_APP_dut.size());
    }
    return check_APP_against_tv(tv_path, p, llr_type, max_iter, h_APP_dut,
                                p99err_rms_threshold, snr_threshold);
}

bool run_llr_check(const std::string& tv_path, cuphyDataType_t llr_type, int algo,
                   int max_iter,
                   float p99err_rms_threshold, float snr_threshold)
{
    printf("TV: %s\n", tv_path.c_str());

    LdpcTvParams         p;
    std::vector<float>   h_LLR_f32;
    std::vector<uint8_t> h_src;
    if(!load_tv(tv_path, p, h_LLR_f32, h_src, llr_type)) return false;

    cuphy::context ctx;
    std::vector<float> h_APP_f32;
    cuphy::tensor_device tLLR = load_llr_device(tv_path, llr_type);
    decode_capture(ctx, tLLR, p, h_LLR_f32, h_src, llr_type, algo, max_iter,
                   "ldpc_dut.h5", h_APP_f32);
    require_APP_history_dumped(h_APP_f32, algo);

    const bool pass = check_APP_against_tv(tv_path, p, llr_type, max_iter, h_APP_f32,
                                          p99err_rms_threshold, snr_threshold);

    printf("\nPlot:\n python3 $cuBB_SDK/cuPHY/util/ldpc/plot_ldpc_app.py %s ldpc_dut.h5 \n\n",
           tv_path.c_str());
    return pass;
}

bool run_llr_diff(const std::string& tv_path, cuphyDataType_t llr_type,
                  int algoA, int algoB,
                  int max_iter,
                  float p99err_rms_threshold, float snr_threshold)
{
    printf("TV: %s   (algoA=%d vs algoB=%d, DUT-vs-DUT)\n", tv_path.c_str(), algoA, algoB);

    LdpcTvParams         p;
    std::vector<float>   h_LLR_f32;
    std::vector<uint8_t> h_src;
    if(!load_tv(tv_path, p, h_LLR_f32, h_src, llr_type)) return false;

    cuphy::context     ctx;
    std::vector<float> h_APP_a, h_APP_b;
    char nameA[64], nameB[64];
    snprintf(nameA, sizeof(nameA), "ldpc_dut_a%d.h5", algoA);
    snprintf(nameB, sizeof(nameB), "ldpc_dut_a%d.h5", algoB);

    cuphy::tensor_device tLLR = load_llr_device(tv_path, llr_type);
    decode_capture(ctx, tLLR, p, h_LLR_f32, h_src, llr_type, algoA, max_iter, nameA, h_APP_a);
    decode_capture(ctx, tLLR, p, h_LLR_f32, h_src, llr_type, algoB, max_iter, nameB, h_APP_b);
    require_APP_history_dumped(h_APP_a, algoA);
    require_APP_history_dumped(h_APP_b, algoB);

    const int Kb       = (p.BGN == 1) ? 22 : 10;
    const int NUM_VAR  = p.Zc * (Kb + p.nV_parity);
    const int NUM_ITER = resolve_check_max_iter(max_iter, p.maxItr);

    const int N_core_var = (Kb + 4) * p.Zc;  // skip ext parity clms; kernels do not update them during iters
    printf("\nComparing algo%d (REF column) vs algo%d (DUT column):\n", algoA, algoB);
    const bool pass = check_APP(h_APP_a.data(), h_APP_b.data(), p.C, NUM_ITER, NUM_VAR, N_core_var, p99err_rms_threshold, snr_threshold);

    printf("Plots:\n python3 $cuBB_SDK/cuPHY/util/ldpc/plot_ldpc_app.py %s %s\n\n",
           nameA, nameB);
    return pass;
}
