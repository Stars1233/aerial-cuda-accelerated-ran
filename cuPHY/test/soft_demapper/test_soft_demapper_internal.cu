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

//#define CUPHY_DEBUG 1
//#define GEN_MATLAB_CHECK 1
#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include <random>
#include "soft_demapper.cuh"
#include "soft_demapper.hpp"
#include "cuphy.hpp"

namespace
{

struct symbol_pos
{
    int          loc;
    unsigned int bits;
    bool bit_is_set(unsigned int idx) const
    {
        return 0 != ((1 << idx) & bits);
    }
};

const symbol_pos PAM2_pos[2] =
{
    { -1, 0x1}, // 1
    {  1, 0x0}, // 0
};
const symbol_pos PAM4_pos[4] =
{
    { -3, 0x3}, // 11
    { -1, 0x1}, // 01
    {  1, 0x0}, // 00
    {  3, 0x2}, // 10
};
const symbol_pos PAM8_pos[8] =
{
    { -7, 0x7}, // 111
    { -5, 0x3}, // 011
    { -3, 0x1}, // 001
    { -1, 0x5}, // 101
    {  1, 0x4}, // 100
    {  3, 0x0}, // 000
    {  5, 0x2}, // 010
    {  7, 0x6}, // 110
};
const symbol_pos PAM16_pos[16] =
{
    {-15, 0xF}, // 1111
    {-13, 0x7}, // 0111
    {-11, 0x3}, // 0011
    { -9, 0xB}, // 1011
    { -7, 0x9}, // 1001
    { -5, 0x1}, // 0001
    { -3, 0x5}, // 0101
    { -1, 0xD}, // 1101
    {  1, 0xC}, // 1100
    {  3, 0x4}, // 0100
    {  5, 0x0}, // 0000
    {  7, 0x8}, // 1000
    {  9, 0xA}, // 1010
    { 11, 0x2}, // 0010
    { 13, 0x6}, // 0110
    { 15, 0xE}, // 1110
};

////////////////////////////////////////////////////////////////////////
// QAM_demapper
// Log-sum approximation demapper
// See:
// "Simplified Soft-Output Demapper for Binary Interleaved COFDM with Application to
//  HIPERLAN/2," Tosato, F., and Bisaglia, P.
// and
// "A Low Complexity 256QAM Soft Demapper for 5G Mobile System," Mao, J.,
// Abdullah, M.A., Xiao, P., and Cao, A.
class QAM_demapper
{
public:
    QAM_demapper(unsigned int      QAM_bits,
                 const symbol_pos* pam_syms,
                 size_t            num_pam_symbols,
                 double            A) :
        A_(A),
        QAM_bits_(QAM_bits),
        PAM_bits_(QAM_bits / 2)
    {
        if(QAM_bits > 1)
        {
            std::copy(pam_syms, pam_syms + num_pam_symbols, std::back_inserter(PAM_symbols_));
        }
    }
    void append_LLRs(std::vector<float>&   LLRs,
                     const cuFloatComplex& sym,
                     float                 noiseVar)
    {
        // "real" noise variance, which is half of complex variance
        float PAMNoiseVar = noiseVar / 2;
        if(QAM_bits_ > 1)
        {

            std::vector<float> evenLLRs = get_PAM_LLRs(sym.x, PAMNoiseVar);
            std::vector<float> oddLLRs  = get_PAM_LLRs(sym.y, PAMNoiseVar);
            for(unsigned int i = 0; i < QAM_bits_; ++i)
            {
                LLRs.push_back((0 == (i % 2)) ? evenLLRs[i / 2] : oddLLRs[i / 2]);
            }
        }
        else
        {
            LLRs.push_back(2 * A_ * (sym.x + sym.y)  / PAMNoiseVar);
        }
    }
private:
    std::vector<float> get_PAM_LLRs(float sym, float noiseVar)
    {
        std::vector<float> LLRs;
        float              Zr_A = sym / A_;
        for(unsigned int i = 0; i < PAM_bits_; ++i)
        {
            int   x_0_A = std::numeric_limits<int>::max();
            int   x_1_A = std::numeric_limits<int>::max();
            for(auto& s : PAM_symbols_)
            {
                if(s.bit_is_set(i))
                {
                    if(std::abs(Zr_A - s.loc) < std::abs(Zr_A - x_1_A))
                    {
                        x_1_A = s.loc;
                        //printf("Setting x_1 to %i\n", x_1);
                    }
                }
                else
                {
                    if(std::abs(Zr_A - s.loc) < std::abs(Zr_A - x_0_A))
                    {
                        x_0_A = s.loc;
                        //printf("Setting x_0 to %i\n", x_1);
                    }
                }
            }
            float L = 0.5f * A_ * A_ * (std::pow(Zr_A - x_1_A, 2) - std::pow(Zr_A - x_0_A, 2)) / noiseVar;
            //printf("bit = %u, Zr = %f, Zr_A = %f, x_0_A = %i, x_1_A = %i\n", i, sym, Zr_A, x_0_A, x_1_A);
            LLRs.push_back(L);
            //printf("bit = %u, LLR = %f\n", i, L);
        }
        return LLRs;
    }
    //------------------------------------------------------------------
    // Data
    std::vector<symbol_pos> PAM_symbols_;
    double                  A_;
    unsigned int            QAM_bits_;
    unsigned int            PAM_bits_;
};

#if GEN_MATLAB_CHECK
////////////////////////////////////////////////////////////////////////
// gen_MATLAB_verification_file()
//
// Write a file that can be used to verify the host demapper
// implementation.
//
// MATLAB code:
//
// cmplx_noise_var = 2;
// A = 1 / sqrt(170);
// Zr = [-16*A:0.4*A:16*A];
// Zi = sin(2*pi*Zr / (16 * A));
// Z = (Zr + j * Zi * 16 * A).';
// out = nrSymbolDemodulate(Z, '256QAM', cmplx_noise_var);
// Z = load('matlab_LLR_check.txt');
// d = abs(out - Z);
// max(d)
//
// (output should be on the order of 5e-7)

void gen_MATLAB_verification_file()
{
    std::vector<float>          LLRs;
    std::vector<cuFloatComplex> symbols;
    const double                A = soft_demapper::QAM_traits<256>::A;
    float                       noiseVar = 2.0f;
    //------------------------------------------------------------------
    // Generate synthetic data
    for(int i = 0; i < 81; ++i)
    {
        // Synthetic "sine wave" of data to exercise demodulator
        float Zr = (-16 * A) + (0.4 * A * i);
        symbols.push_back(make_cuFloatComplex(Zr,
                                              16 * A * std::sin(2 * M_PI * Zr / (16 * A))));
    }
    //------------------------------------------------------------------
    // Execute the "reference" demapper
    QAM_demapper demapper(8, PAM16_pos, 16, A);
    for(auto& s : symbols)
    {
        demapper.append_LLRs(LLRs, s, noiseVar);
    }
    //------------------------------------------------------------------
    // Store data in a file
    FILE* fp = fopen("matlab_LLR_check.txt", "w");
    if(!fp)
    {
        fprintf(stderr, "Could not open file for writing\n");
        return;
    }
    for(size_t i = 0; i < LLRs.size(); ++i)
    {
        fprintf(fp, "%f\n", LLRs[i]);
    }
    fclose(fp);
}
#endif // #if GEN_MATLAB_CHECK

std::vector<cuFloatComplex> generate_random_symbols(size_t count,
                                                    float  maxVal)
{
    //------------------------------------------------------------------
    // Random number generation
    std::mt19937                     e2;
    std::uniform_real_distribution<> dist(-maxVal, maxVal);
    std::vector<cuFloatComplex>      symbols;
    symbols.reserve(count);
    for(size_t i = 0; i < count; ++i)
    {
        symbols.push_back(make_cuFloatComplex(dist(e2), dist(e2)));
    }
    return symbols;
}

} // namespace

////////////////////////////////////////////////////////////////////////
// test_soft_demapper_kernel()
template <int QAM>
__global__ void test_soft_demapper_kernel(cudaTextureObject_t   dmTexObj,
                                          const cuFloatComplex* symbols,
                                          size_t                symbolCount,
                                          __half2*              LLR,
                                          float                 noiseVarInv)
{
    typedef soft_demapper::QAM_traits<QAM>                       QAM_traits_t;
    typedef soft_demapper::soft_demapper<float, __half, QAM>     soft_demapper_t;
    typedef soft_demapper::LLR_group<__half, QAM_traits_t::bits> llr_group_t;

    llr_group_t llr_grp;

    int symbolIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(symbolIdx >= symbolCount) return;

    // PAM noise is 1/2 QAM noise. Since we are using the inverse, mul by 2.
    float PAMnoiseVarInv = noiseVarInv * 2.0f;

    soft_demapper_t::symbol_to_LLR_group(llr_grp,            // LLR output
                                         symbols[symbolIdx], // symbol input
                                         PAMnoiseVarInv,     // 1 / noise_var
                                         dmTexObj);
    if(soft_demapper::QAM_traits<QAM>::bits > 1)
    {
        llr_grp.write(LLR + (symbolIdx *(soft_demapper::QAM_traits<QAM>::bits / 2)));
    }
    else
    {
        // Write BPSK as __half instead of __half2
        __half* LLRh = reinterpret_cast<__half*>(LLR);
        llr_grp.write(LLRh + symbolIdx);
    }

    //printf("symbol = (%f, %f)\n", symbols[symbolIdx].x, symbols[symbolIdx].y);
    //printf("LLRs: (%f, %f, %f, %f, %f, %f, %f, %f)\n",
    //       __low2float(llr_grp.f16x2[0]),
    //       __high2float(llr_grp.f16x2[0]),
    //       __low2float(llr_grp.f16x2[1]),
    //       __high2float(llr_grp.f16x2[1]),
    //       __low2float(llr_grp.f16x2[2]),
    //       __high2float(llr_grp.f16x2[2]),
    //       __low2float(llr_grp.f16x2[3]),
    //       __high2float(llr_grp.f16x2[3]));
}

////////////////////////////////////////////////////////////////////////
// do_soft_demapper_test()
template <int QAM>
void do_soft_demapper_test(size_t            NUM_TEST_SYMBOLS,
                           double            symbolRange,
                           float             noiseVar,
                           const symbol_pos* pam_syms,
                           size_t            num_pam_symbols)
{
    std::vector<float>          LLRs;
    const size_t                BITS_PER_SYMBOL  = soft_demapper::QAM_traits<QAM>::bits;
    std::vector<cuFloatComplex> symbols          = generate_random_symbols(NUM_TEST_SYMBOLS,
                                                                           symbolRange);
    LLRs.reserve(NUM_TEST_SYMBOLS * BITS_PER_SYMBOL);
    //------------------------------------------------------------------
    // Create a host reference soft demapper and generate the LLRs
    QAM_demapper demapper(soft_demapper::QAM_traits<QAM>::bits,
                          pam_syms,
                          num_pam_symbols,
                          soft_demapper::QAM_traits<QAM>::A);
    for(auto& s : symbols)
    {
        demapper.append_LLRs(LLRs, s, noiseVar);
    }

    //------------------------------------------------------------------
    // Create a GPU soft demapper "context" to set up device textures
    cuphy_i::soft_demapper_context sd_context;
    //------------------------------------------------------------------
    // Allocate buffers for input and output, and copy input data
    typedef cuphy::buffer<cuFloatComplex, cuphy::device_alloc> complex_buffer_device_t;
    typedef cuphy::buffer<__half2, cuphy::pinned_alloc>        half2_buffer_pinned_t;

    half2_buffer_pinned_t LLR_output = half2_buffer_pinned_t((NUM_TEST_SYMBOLS * BITS_PER_SYMBOL) / 2);
    complex_buffer_device_t symbols_device(symbols); // Copy host symbol data to device
    //------------------------------------------------------------------
    // Execute the kernel on the device
    test_soft_demapper_kernel<QAM><<<1, NUM_TEST_SYMBOLS>>>(sd_context.QAM_tex().tex_obj().handle(),
                                                            symbols_device.addr(),
                                                            NUM_TEST_SYMBOLS,
                                                            LLR_output.addr(),
                                                            1.0f / noiseVar);
    cudaDeviceSynchronize();
    //------------------------------------------------------------------
    // Compare host reference and device output
    for(size_t i = 0; i < LLRs.size(); ++i)
    {
        cuFloatComplex sym  = symbols[i / BITS_PER_SYMBOL];
        float    LLR_device = (0 == (i % 2))                  ?
                              __low2float (LLR_output[i / 2]) :
                              __high2float(LLR_output[i / 2]);
        float    absError      = std::abs(LLR_device - LLRs[i]);
        float    relativeError = (LLRs[i] != 0.0f) ? absError / std::abs(LLRs[i]) : 0.0f;
        //printf("[%4lu]: %f   %f (e = %f, %f%%)\n", i, LLRs[i], LLR_device, absError, relativeError * 100);
        // Difficult to establish a relative error bound: for very small
        // LLRs, the relative error is oftern much greater than 10%.
        EXPECT_TRUE((relativeError < 0.10f) || (abs(LLRs[i]) < 0.02f))
            << "QAM = " << QAM << ", index = " << i << ", "
            << "symbol = (" << sym.x << ", " << sym.y  << "), "
            << "device = " << LLR_device << ", host = " << LLRs[i]
            << ", relative error = " << relativeError << std::endl;
        EXPECT_TRUE(absError < 0.005f)
            << "QAM = " << QAM  << ", index = " << i << ", "
            << "symbol = (" << sym.x << ", " << sym.y  << "), "
            << "device = " << LLR_device << ", host = " << LLRs[i]
            << ", abs error = " << absError << std::endl;

    }
}

////////////////////////////////////////////////////////////////////////
// SoftDemapper.BPSK
TEST(SoftDemapper, BPSK)
{
    // Note that texture fetches will clamp to the table border value,
    // and comparisons with the host reference that do not exhibit the
    // same behavior will fail for large symbol values. Also, for BPSK
    // the SUM of the real and imaginary components is used for the
    // texture fetch.
    do_soft_demapper_test<2>(256,                                  // number of test symbols
                             1 * soft_demapper::QAM_traits<2>::A,  // range for generated symbols
                             2.0f,                                 // noise variance
                             PAM2_pos,                             // PAM symbols for reference (same as QPSK)
                             2) ;                                  // number of PAM symbols
}

////////////////////////////////////////////////////////////////////////
// SoftDemapper.QPSK
TEST(SoftDemapper, QPSK)
{
    do_soft_demapper_test<4>(256,                                  // number of test symbols
                             2 * soft_demapper::QAM_traits<4>::A,  // range for generated symbols
                             2.0f,                                 // noise variance
                             PAM2_pos,                             // PAM symbols for reference
                             2) ;                                  // number of PAM symbols
}

////////////////////////////////////////////////////////////////////////
// SoftDemapper.QAM16
TEST(SoftDemapper, QAM16)
{
    do_soft_demapper_test<16>(256,                                  // number of test symbols
                              4 * soft_demapper::QAM_traits<16>::A, // range for generated symbols
                              2.0f,                                 // noise variance
                              PAM4_pos,                             // PAM symbols for reference
                              4) ;                                  // number of PAM symbols
}

////////////////////////////////////////////////////////////////////////
// SoftDemapper.QAM64
TEST(SoftDemapper, QAM64)
{
    do_soft_demapper_test<64>(256,                                  // number of test symbols
                              8 * soft_demapper::QAM_traits<64>::A, // range for generated symbols
                              2.0f,                                 // noise variance
                              PAM8_pos,                             // PAM symbols for reference
                              8) ;                                  // number of PAM symbols
}
////////////////////////////////////////////////////////////////////////
// SoftDemapper.QAM256
TEST(SoftDemapper, QAM256)
{
    do_soft_demapper_test<256>(256,                                    // number of test symbols
                               16 * soft_demapper::QAM_traits<256>::A, // range for generated symbols
                               2.0f,                                   // noise variance
                               PAM16_pos,                              // PAM symbols for reference
                               16);                                    // number of PAM symbols
}


////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
#if GEN_MATLAB_CHECK
    gen_MATLAB_verification_file();
#endif
    int result = RUN_ALL_TESTS();
    return result;
}
