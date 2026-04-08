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

#include "api.h"
#include "cumac.h"
#include "trtEngine.h"

void usage() {
    printf("TRT engine example [options]\n");
    printf("  Options:\n");
    printf("  -i  Input test vector file\n");
    printf("  -g  GPU device number\n");
    printf("  -h  Show this help\n");
    printf("Example: './trtEngine -i <tv_file> -g <GPU number>'\n");
}


int main(int argc, char* argv[]) {

    // Number of GPUs.
    int32_t nGPUs = 0;
    CUDA_CHECK_ERR(cudaGetDeviceCount(&nGPUs));

    // Parse arguments.
    int iArg = 1;
    int32_t gpuId = 0;
    std::string inputFileName;

    while(iArg < argc) {
        if('-' == argv[iArg][0]) {
            switch(argv[iArg][1])
            {
            case 'i':  // Input test vector file name
                if(++iArg >= argc)
                {
                    fprintf(stderr, "ERROR: No input test vector file name given.\n");
                    exit(1);
                }
                inputFileName.assign(argv[iArg++]);
                break;

            case 'g':  // Select GPU
                if((++iArg >= argc) ||
                    (1 != sscanf(argv[iArg], "%i", &gpuId)) ||
                    ((gpuId < 0) || (gpuId >= nGPUs)))
                {
                    fprintf(stderr, "ERROR: Invalid GPU Id (should be within [0, %d])\n", nGPUs - 1);
                    exit(1);
                }
                iArg++;
                break;

            case 'h':  // Print usage
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

    CUDA_CHECK_ERR(cudaSetDevice(gpuId));
    std::cout << "cuMAC TRT engine: Running on GPU device " << gpuId << std::endl;
    std::cout << "=========================================" << std::endl;

    // Create a CUDA stream
    cudaStream_t cudaStream;
    CUDA_CHECK_ERR(cudaStreamCreate(&cudaStream));

    // Read the test vector file.
    // Note this assumes that there is only one input and output.
    std::string modelFile;
    std::string inputName;
    std::string outputName;
    float* inputTensor;
    float* deviceInputTensor;
    float* outputTensor;
    float* deviceOutputTensor;
    float* refOutputTensor;
    std::vector<int> inputShape(5, 0);
    std::vector<int> outputShape(5, 0);

    H5::H5File file(inputFileName, H5F_ACC_RDONLY);

    // ONNX model file name. Replace the full path to correspond to where this example
    // was called from.
    H5::DataSet dataset = file.openDataSet("model_file");
    dataset.read(modelFile, dataset.getStrType(), dataset.getSpace());
    std::filesystem::path path(inputFileName);
    modelFile = path.replace_filename(modelFile).string();
    std::cout << "Model file: " << modelFile << std::endl;

    // Input tensor name (as in the onnx).
    dataset = file.openDataSet("input_name");
    dataset.read(inputName, dataset.getStrType(), dataset.getSpace());
    std::cout << "Input tensor name: " << inputName << std::endl;

    // Output tensor name (as in the onnx).
    dataset = file.openDataSet("output_name");
    dataset.read(outputName, dataset.getStrType(), dataset.getSpace());
    std::cout << "Output tensor name: " << outputName << std::endl;

    // Input tensor shape.
    dataset = file.openDataSet("input_shape");
    dataset.read(inputShape.data(), H5::PredType::NATIVE_UINT32);
    int numDims = std::find(inputShape.begin(), inputShape.end(), 0) - inputShape.begin();
    inputShape.resize(numDims);
    size_t nElems = 1;
    std::cout << "Input tensor shape: (";
    for(int i = 0; i < numDims; i++) {
        if(i > 0)
            std::cout << ", ";
        std::cout << inputShape[i];

        nElems *= inputShape[i];
    }
    std::cout << ")" << std::endl;

    // Input tensor.
    CUDA_CHECK_ERR(cudaMallocHost((void**)&inputTensor, nElems * sizeof(float)));
    dataset = file.openDataSet("input_tensor");
    dataset.read(inputTensor, H5::PredType::NATIVE_FLOAT);

    // Copy input tensor to device.
    CUDA_CHECK_ERR(cudaMalloc((void**)&deviceInputTensor, nElems * sizeof(float)));
    CUDA_CHECK_ERR(cudaMemcpyAsync((void*)deviceInputTensor, (void*)inputTensor, nElems * sizeof(float), cudaMemcpyHostToDevice, cudaStream));
    CUDA_CHECK_ERR(cudaStreamSynchronize(cudaStream));

    // Output tensor shape.
    dataset = file.openDataSet("output_shape");
    dataset.read(outputShape.data(), H5::PredType::NATIVE_UINT32);
    numDims = std::find(outputShape.begin(), outputShape.end(), 0) - outputShape.begin();
    outputShape.resize(numDims);
    nElems = 1;
    std::cout << "Output tensor shape: (";
    for(int i = 0; i < numDims; i++) {
        if(i > 0)
            std::cout << ", ";
        std::cout << outputShape[i];

        nElems *= outputShape[i];
    }
    std::cout << ")" << std::endl;

    // Output tensor.
    CUDA_CHECK_ERR(cudaMallocHost((void**)&refOutputTensor, nElems * sizeof(float)));
    dataset = file.openDataSet("output_tensor");
    dataset.read(refOutputTensor, H5::PredType::NATIVE_FLOAT);

    // These are for the model outputs (host/device).
    CUDA_CHECK_ERR(cudaMallocHost((void**)&outputTensor, nElems * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc((void**)&deviceOutputTensor, nElems * sizeof(float)));

    // Now create the TRT engine.
    const std::vector<cumac_ml::trtTensorPrms_t> inputTensorPrms = {{inputName, inputShape}};      // Input tensor parameters - name and shape
    const std::vector<cumac_ml::trtTensorPrms_t> outputTensorPrms = {{outputName, outputShape}};   // Output tensor parameters - name and shape
    const uint32_t maxBatchSize = inputShape[0];    // Maximum batch size. This is the maximum that can be envisioned,
                                                    // the actual current batch size is set for every call separately.
                                                    // For simplicity use the same size here.
    const bool parseFromOnnx = true;
    std::unique_ptr<cumac_ml::trtEngine> pTrtEngine = std::make_unique<cumac_ml::trtEngine>(modelFile.c_str(), parseFromOnnx, maxBatchSize, inputTensorPrms, outputTensorPrms);

    // Run setup - This merely sets the input addresses and the actual batch size.
    // One can omit the batch size to use the maximum batch size set upon init (omitted here since we used the same size).
    // Note: In case there are multiple inputs/outputs, the addresses need to be in the same order as in
    // inputTensorPrms/outputTensorPrms.
    std::vector<void*> inputBuffers = {(void*)deviceInputTensor};
    std::vector<void*> outputBuffers = {(void*)deviceOutputTensor};
    pTrtEngine->setup(inputBuffers, outputBuffers);

    // Run the engine.
    pTrtEngine->run(cudaStream);

    // Copy outputs to host.
    CUDA_CHECK_ERR(cudaMemcpyAsync((void*)outputTensor, (void*)deviceOutputTensor, nElems * sizeof(float), cudaMemcpyDeviceToHost, cudaStream));
    CUDA_CHECK_ERR(cudaStreamSynchronize(cudaStream));

    // Check against reference.
    bool success = true;
    for(int elem = 0; elem < nElems; elem++) {
        float absDiff = fabs(outputTensor[elem] - refOutputTensor[elem]);
        if(absDiff > 1e-5) {
            std::cout << "Mismatched output and reference value, " << outputTensor[elem] << " vs. " << refOutputTensor[elem] << std::endl;
            success = false;
        }
    }

    // Free everything.
    CUDA_CHECK_ERR(cudaFreeHost(inputTensor));
    CUDA_CHECK_ERR(cudaFree(deviceInputTensor));
    CUDA_CHECK_ERR(cudaFreeHost(refOutputTensor));
    CUDA_CHECK_ERR(cudaFreeHost(outputTensor));
    CUDA_CHECK_ERR(cudaFree(deviceOutputTensor));

    if(!success)
        exit(1);

    std::cout << "\033[1;32mPASSED!\033[0m" << std::endl;
    exit(0);
}