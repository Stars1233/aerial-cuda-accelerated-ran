#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Exit on first error
set -e

# Switch to PROJECT_ROOT directory
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)
echo $SCRIPT starting...
cd $PROJECT_ROOT

# Most of the unit tests might be skipped if the test vector location has not been given.
# Default location is "/mnt/cicd_tvs/develop/GPU_test_input/".
if [ -z "${TEST_VECTOR_DIR}" ]; then
    echo "Test vector directory is not set - defaulting to /mnt/cicd_tvs/develop/GPU_test_input/."
    echo "Unit tests will be skipped if test vectors are not found."
    echo "Set test vector directory as follows:"
    echo "export TEST_VECTOR_DIR=<test vector directory>"
    echo ""
fi

# Generate TRT models.
export CUDA_MODULE_LOADING=LAZY

mkdir $HOME/models

trtexec --onnx=$PROJECT_ROOT/models/llrnet.onnx \
        --saveEngine=$HOME/models/llrnet.trt \
        --skipInference \
        --minShapes=input:1x2 \
        --optShapes=input:12345x2 \
        --maxShapes=input:42588x2 \
        --inputIOFormats=fp32:chw \
        --outputIOFormats=fp32:chw

trtexec --onnx=$PROJECT_ROOT/models/neural_rx.onnx \
        --saveEngine=$HOME/models/neural_rx.trt \
        --skipInference \
        --shapes=rx_slot_real:1x3276x12x4,rx_slot_imag:1x3276x12x4,h_hat_real:1x4914x1x4,h_hat_imag:1x4914x1x4

echo pyAerial: Run Python unit tests...
pushd tests > /dev/null
python3 -m pytest -o cache_dir=$HOME/.pytest_cache -sv
popd > /dev/null

rm -rf $HOME/models

#echo pyAerial: Run C++ wrapper tests...
#CUPHY_ROOT=$(dirname $PROJECT_ROOT)
#PYCUPHY_TESTS=$CUPHY_ROOT/build/pyaerial/tests/cpp
#shopt -s extglob
#while read tv; do
#    file=/mnt/cicd_tvs/develop/GPU_test_input/TVnr_${tv}_PUSCH_gNB_CUPHY_s*.h5
#    $PYCUPHY_TESTS/test_pusch_rx_main -i $file
#done < $SCRIPT_DIR/pusch_tvs.txt

# Finished
echo $SCRIPT finished with success.
