#!/bin/bash

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

function gen_notebook() {
   echo Running notebook $1...
   jupyter nbconvert --execute $1 --output-dir notebooks/generated --to notebook
}

# Exit on first error
set -e

# Switch to PROJECT_ROOT directory
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)
echo $SCRIPT starting...
cd $PROJECT_ROOT

echo Run notebook examples...
rm -rf notebooks/generated
mkdir -p notebooks/generated

# pyAerial (cuPHY) examples.
gen_notebook notebooks/example_pusch_simulation.ipynb
gen_notebook notebooks/example_ldpc_coding.ipynb
gen_notebook notebooks/example_srs_tx_rx.ipynb
gen_notebook notebooks/example_csi_rs_tx_rx.ipynb

# ML and LLRNet examples.
gen_notebook notebooks/example_simulated_dataset.ipynb
gen_notebook notebooks/llrnet_dataset_generation.ipynb
gen_notebook notebooks/llrnet_model_training.ipynb
gen_notebook notebooks/example_neural_receiver.ipynb
gen_notebook notebooks/channel_estimation/channel_estimation.ipynb

# Data Lake examples.
gen_notebook notebooks/datalake_channel_estimation.ipynb
gen_notebook notebooks/datalake_pusch_decoding.ipynb
gen_notebook notebooks/datalake_pusch_multicell.ipynb

# Finished
echo $SCRIPT finished
