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

python3 auto_allocate_physical_cores.py l1_logical_cores_DU_CG1_RU_CG1.yaml  test_du1.yaml
python3 auto_allocate_physical_cores.py l1_logical_cores_DU_CG1_RU_R750.yaml  test_du2.yaml
python3 auto_allocate_physical_cores.py l1_logical_cores_DU_R750_RU_DEVKIT.yaml  test_du3.yaml
python3 auto_allocate_physical_cores.py l1_logical_cores_DU_R750_RU_DEVKIT_HT.yaml  test_du4.yaml

python3 auto_allocate_physical_cores.py l1_logical_cores_DU_CG1_RU_CG1.yaml  test_ru1.yaml -e
python3 auto_allocate_physical_cores.py l1_logical_cores_DU_CG1_RU_R750.yaml  test_ru2.yaml -e
python3 auto_allocate_physical_cores.py l1_logical_cores_DU_R750_RU_DEVKIT.yaml  test_ru3.yaml -e
python3 auto_allocate_physical_cores.py l1_logical_cores_DU_R750_RU_DEVKIT_HT.yaml  test_ru4.yaml -e

export BASE_PATH=../../../
python3 auto_override_yaml_cores.py  test_ru2.yaml -r $BASE_PATH/cuPHY-CP/ru-emulator/config/config.yaml
python3 auto_override_yaml_cores.py  test_du2.yaml -c $BASE_PATH/cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_CG1.yaml \
                                                   -l $BASE_PATH/cuPHY-CP/cuphycontroller/config/l2_adapter_config_F08_CG1.yaml \
                                                   -t $BASE_PATH/cuPHY-CP/testMAC/testMAC/test_mac_config.yaml