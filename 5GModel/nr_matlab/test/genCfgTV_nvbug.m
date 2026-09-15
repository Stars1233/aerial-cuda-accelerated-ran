% SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
% SPDX-License-Identifier: Apache-2.0
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
% http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

function [errCnt, nTV] = genCfgTV_nvbug(caseSet)

errCnt = 0;
nTV = 0;

errFlag = runSim('cfg_bug3901585.yaml', 'bug3901585');
errCnt = errCnt + errFlag;
nTV = nTV + 1;

errFlag = runSim('cfg_bug3862516.yaml', 'bug3862516');
errCnt = errCnt + errFlag;
nTV = nTV + 1;

errFlag = runSim('cfg_bug4011756_cell0.yaml', 'bug4011756_cell0');
errCnt = errCnt + errFlag;
nTV = nTV + 1;

errFlag = runSim('cfg_bug4011756_cell2.yaml', 'bug4011756_cell2');
errCnt = errCnt + errFlag;
nTV = nTV + 1;

errFlag = runSim('cfg_bug4011756_cell3.yaml', 'bug4011756_cell3');
errCnt = errCnt + errFlag;
nTV = nTV + 1;

errFlag = runSim('cfg_bug3951344.yaml', 'bug3951344');
errCnt = errCnt + errFlag;
nTV = nTV + 1;

errFlag = runSim('cfg_bug4185251.yaml', 'bug4185251');
errCnt = errCnt + errFlag;
nTV = nTV + 1;

errFlag = runSim('cfg_bug5098017_PUSCH_sfn943_15.yaml', 'bug5098017_PUSCH_sfn943_15');
errCnt = errCnt + errFlag;
nTV = nTV + 1;

errFlag = runSim('cfg_bug5098017_PUSCH_sfn944_15.yaml', 'bug5098017_PUSCH_sfn944_15');
errCnt = errCnt + errFlag;
nTV = nTV + 1;

errFlag = runSim('cfg_bug5486238.yaml', 'bug5486238');
errCnt = errCnt + errFlag;
nTV = nTV + 1;

errFlag = runSim('cfg_sfn_555_4_pusch_neg.yaml', 'bug5714275_PUSCH');
errCnt = errCnt + errFlag;
nTV = nTV + 1;

testCompGenTV_dlmix('harq');
testCompGenTV_ulmix('harq');
errFlag = runSim_multiSlot('cfg_HARQ_MIX_list.yaml', 'HARQ_MIX');
delete cfg_HARQ*.yaml
errCnt = errCnt + errFlag;
nTV = nTV + 1;

return