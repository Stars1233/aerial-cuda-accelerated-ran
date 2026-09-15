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

#include "gauNoiseAdder.cuh"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <iostream>
#include <cmath>
#include <random>
#include <cstdlib> // For std::atoi


/**
 * @brief printout usage message
 * 
 */
void usage() {
    std::cout << "Gaussian noise adder test [options]" << std::endl;
    std::cout << "  Arguments:" << std::endl;
    printf("  -n  [signal size (default 1024)]\n");
    printf("  -s  [snr in dB (default 20)]\n");
    printf("  -h  display usage information \n");
    std::cout << "Example (default 1024 signal size, 20dB SNR): ./gau_noise_adder_ex " << std::endl;
    std::cout << "Example (61440 signal size, 10dB SNR): ./gau_noise_adder_ex -l 61440 -s 10 " << std::endl;
}

// Function to initialize the signal with random complex numbers of unit power
void initializeSignal(cuComplex* signal, uint32_t size) {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 1.0f);

    float scale = 1.0f / sqrtf(2.0f); // Scale to ensure unit power

    for (uint32_t i = 0; i < size; ++i) {
        float real = distribution(generator) * scale;
        float imag = distribution(generator) * scale;
        signal[i] = make_cuComplex(real, imag);
    }
}

// Function to calculate power of a complex signal
float calculatePower(const cuComplex* signal, uint32_t size) {
    float power = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        power += cuCrealf(signal[i]) * cuCrealf(signal[i]) + cuCimagf(signal[i]) * cuCimagf(signal[i]);
    }
    return power / size;
}

// Function to print the signal for debugging
void printSignal(const cuComplex* signal, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        std::cout << "Signal[" << i << "] = (" << cuCrealf(signal[i]) << ", " << cuCimagf(signal[i]) << ")\n";
    }
}

int main(int argc, char* argv[]) {
    // Default values
    float snr_db = 20.0f;
    int signalSize = 1024;
    uint32_t nThreads = 256;
    int seed = static_cast<int>(time(NULL));

    int iArg = 1;
    // read options from CLI
    while(iArg < argc) {
        if('-' == argv[iArg][0]) {
            switch(argv[iArg][1]) {
                case 's': // SNR in dB
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%f", &snr_db)))
                    {
                        fprintf(stderr, "ERROR: Invalid SNR value.\n");
                        exit(1);
                    }
                    ++iArg;
                    break;

                case 'l': // signal length
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%d", &signalSize)) || (signalSize <= 0))
                    {
                        fprintf(stderr, "ERROR: Invalid signal length.\n");
                        exit(1);
                    }
                    ++iArg;
                    break;

                case 'h': // help
                    usage();
                    exit(0);
                    break;

                default:
                    fprintf(stderr, "ERROR: Unknown option: %s\n", argv[iArg]);
                    usage();
                    exit(1);
                    break;
            }
        }
        else {
            fprintf(stderr, "ERROR: Invalid command line argument: %s\n", argv[iArg]);
            exit(1);
        }
    }

    printf("Gaussian noise adder test: signalSize: %d, snr_db: %f\n", signalSize, snr_db);
    // Allocate host memory for the original and noisy signals
    cuComplex* h_signal_original = new cuComplex[signalSize];
    cuComplex* h_signal_noisy = new cuComplex[signalSize];
    
    initializeSignal(h_signal_original, signalSize);

    // Calculate initial power of the original signal
    float initial_power = calculatePower(h_signal_original, signalSize);
    std::cout << "Initial Signal Power: " << initial_power << "\n";

    // Allocate device memory
    cuComplex* d_signal;
    cudaMalloc(&d_signal, signalSize * sizeof(cuComplex));
    
    // Copy original signal to device
    cudaMemcpy(d_signal, h_signal_original, signalSize * sizeof(cuComplex), cudaMemcpyHostToDevice);

    // Create a CUDA stream
    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_DEFAULT);

    // Create GauNoiseAdder object
    GauNoiseAdder<cuComplex> noise_adder(nThreads, seed, stream);

    // Add noise with specified SNR
    noise_adder.addNoise(d_signal, signalSize, snr_db);

    // Copy noisy results back to host
    cudaMemcpy(h_signal_noisy, d_signal, signalSize * sizeof(cuComplex), cudaMemcpyDeviceToHost);

    // Calculate noise by subtracting original from noisy signal
    float noise_power = 0.0f;
    
    for (uint32_t i = 0; i < signalSize; ++i) {
        float real_diff = cuCrealf(h_signal_noisy[i]) - cuCrealf(h_signal_original[i]);
        float imag_diff = cuCimagf(h_signal_noisy[i]) - cuCimagf(h_signal_original[i]);
        noise_power += real_diff * real_diff + imag_diff * imag_diff;
    }
    
    noise_power /= signalSize;

    // Calculate empirical SNR
    float empirical_snr = initial_power / noise_power;

    std::cout << "Empirical SNR: " << 10 * log10(empirical_snr) << " dB\n";

    // Print the modified signal
    printSignal(h_signal_noisy, 10); // Print first 10 elements for brevity

    // Clean up resources
    delete[] h_signal_original;
    delete[] h_signal_noisy;
    
    cudaFree(d_signal);
    
    cuStreamDestroy(stream);

    return 0;
}