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

# Apply CONFIG_DIR logic after all parsing is complete
CONFIG_DIR=$cuBB_SDK
echo "CONFIG_DIR: $CONFIG_DIR"
#!/bin/bash
TEST_CONFIG_FILE=$CONFIG_DIR/testBenches/phase4_test_scripts/test_config_summary.sh
[ -f $TEST_CONFIG_FILE ] && source $TEST_CONFIG_FILE
while [[ ! -v DU_SETUP_COMPLETE ]] || [[ -v DU_SETUP_COMPLETE  && -v RU_SETUP_COMPLETE ]]; do
	echo "RU: $RU_SETUP_COMPLETE"
	echo "DU: $DU_SETUP_COMPLETE"
	if [[ -v RU_SETUP_COMPLETE ]]; then
		echo "$(date): Please rerun DU SETUP"
		rm $TEST_CONFIG_FILE
	else
		echo "$(date) DU Setup not complete. Waiting"
	fi
	sleep 5
	unset RU_SETUP_COMPLETE
	unset DU_SETUP_COMPLETE
	[ -f $TEST_CONFIG_FILE ] && source $TEST_CONFIG_FILE
done

./testBenches/phase4_test_scripts/setup2_RU.sh && ./testBenches/phase4_test_scripts/test_config_nrSim.sh && ./testBenches/phase4_test_scripts/run1_RU.sh

