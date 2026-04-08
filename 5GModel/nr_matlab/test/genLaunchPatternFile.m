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

function errFlag = genLaunchPatternFile(caseSet)

caseList = [1:2007];
if(nargin > 0)
    if(isnumeric(caseSet))
        caseList = caseSet;
    end
end

for caseNum = caseList
    gen_launch_pattern(caseNum);
end

errFlag = 0;

return

function gen_launch_pattern(caseNum)

% requested multi-cell TCs
if caseNum == 604 
    if exist('GPU_test_input/launch_pattern_F08_8C_57.yaml') 
        copyfile GPU_test_input/launch_pattern_F08_8C_57.yaml GPU_test_input/launch_pattern_nrSim_90604.yaml
    else
        warning('TC-90604: GPU_test_input/launch_pattern_F08_8C_57.yaml does not exist !!!')
    end
    return
end

nSlot = 20;

switch caseNum
    case 1 % SSB
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_1001_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_1001_gNB_FAPI_s0.h5'};
    case 2 % PDSCH
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_2001_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_2001_gNB_FAPI_s0.h5'};
    case 3 % PDCCH
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
    case 4 % CSIRS
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_4001_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_4001_gNB_FAPI_s0.h5'};
    case 5 % PRACH
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_5001_gNB_FAPI_s1.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_5001_gNB_FAPI_s1.h5'};
    case 6 % PUCCH
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_6301_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_6301_gNB_FAPI_s0.h5'};
    case 7 % PUSCH
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_7201_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_7201_gNB_FAPI_s0.h5'};
    case 11 % dlmix
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
    case 12 % ulmix
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_0603_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0603_gNB_FAPI_s0.h5'};
    case 13 % Special Slot (PDSCH [2:4], PUSCH [10:13])
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_3212_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
    case 14 % multi-slot single-cell w/o S-slot, POC2_1C_00
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0660_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_0603_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_0644_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_0652_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_0660_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};   
    case 15 % multi-slot single-cell w/ S-slot, POC2_1C_00
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0660_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_0603_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_0644_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_0652_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_0660_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
    case 16 % single-slot multi-cell DL, POC2_2C_00
        nCell = 2;
        LP = [];
        LP.Cell_Configs = {'TVnr_0204_gNB_FAPI_s0.h5','TVnr_0205_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};        
    case 17 % single-slot multi-cell UL, POC2_2C_00
        nCell = 2;
        LP = [];
        LP.Cell_Configs = {'TVnr_0644_gNB_FAPI_s0.h5','TVnr_0645_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0644_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{2}.channels = {'TVnr_0645_gNB_FAPI_s0.h5'};  
    case 18 % multi-slot multi-cell UL, POC2_2C_00
        nCell = 2;
        LP = [];
        LP.Cell_Configs = {'TVnr_0204_gNB_FAPI_s0.h5','TVnr_0205_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0660_gNB_FAPI_s0.h5','TVnr_0661_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_0636_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{2}.channels = {'TVnr_0637_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_0644_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{2}.channels = {'TVnr_0645_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};        
        LP.SCHED{8}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};        
        LP.SCHED{9}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};        
        LP.SCHED{10}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};        
        LP.SCHED{11}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};        
        LP.SCHED{12}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};        
        LP.SCHED{13}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};        
        LP.SCHED{14}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};        
        LP.SCHED{15}.config{1}.channels = {'TVnr_0652_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{2}.channels = {'TVnr_0653_gNB_FAPI_s0.h5'};        
        LP.SCHED{16}.config{1}.channels = {'TVnr_0660_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{2}.channels = {'TVnr_0661_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};        
        LP.SCHED{18}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};        
        LP.SCHED{19}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};        
        LP.SCHED{20}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};         
        LP.SCHED{20}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};
    case 19 % 2 cells:  PDSCH w/o precoding + PDSCH w/ precoding
        nCell = 2;
        LP = [];
        LP.Cell_Configs = {'TVnr_3201_gNB_FAPI_s0.h5','TVnr_3249_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{2}.channels = {'TVnr_3249_gNB_FAPI_s0.h5'};
    case 20 % 2 cells, BFP14 + BFP9
        nCell = 2;
        LP = [];
        LP.Cell_Configs = {'TVnr_7342_gNB_FAPI_s0.h5','TVnr_7343_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_7342_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{2}.channels = {'TVnr_7343_gNB_FAPI_s0.h5'};
    case 21 % multi-slot single-cell HARQ 4-rx TC-7324
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnrPUSCH_HARQ4_7324_gNB_FAPI_s0.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{4}.config{1}.channels = {'TVnrPUSCH_HARQ4_7324_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnrPUSCH_HARQ4_7324_gNB_FAPI_s1.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnrPUSCH_HARQ4_7324_gNB_FAPI_s2.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnrPUSCH_HARQ4_7324_gNB_FAPI_s3.h5'};
    case 22 % multi-slot single-cell HARQ 2-rx 2-UE TC-7326
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnrPUSCH_HARQ2_7326_gNB_FAPI_s0.h5'};
        LP.config_static_harq_proc_id = 1;  % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{4}.config{1}.channels = {'TVnrPUSCH_HARQ2_7326_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnrPUSCH_HARQ2_7326_gNB_FAPI_s1.h5'};
    case 23 % multi-slot multi-cell UL with empty slot, POC2_2C_00
        nCell = 2;
        LP = [];
        LP.Cell_Configs = {'TVnr_0204_gNB_FAPI_s0.h5','TVnr_0205_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0660_gNB_FAPI_s0.h5','TVnr_0661_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_0636_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{2}.channels = {'TVnr_0637_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_0644_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{2}.channels = {'TVnr_0645_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_0652_gNB_FAPI_s0.h5'};
%         LP.SCHED{15}.config{2}.channels = {'TVnr_0653_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_0660_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{2}.channels = {'TVnr_0661_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};
%         LP.SCHED{18}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};
    case 24 % DDDUUUUUUU, multi-slot single-cell, POC2_1C_00
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0612_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_0612_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_0613_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_0614_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_0615_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_0616_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_0617_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_0618_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_0652_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_0653_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_0654_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_0655_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_0656_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_0657_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_0658_gNB_FAPI_s0.h5'};
    case 25 % DDSUUUUUUU, multi-slot single-cell w/ S-slot, POC2_1C_00
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0612_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_0612_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_0613_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_0614_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_0615_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_0616_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_0617_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_0618_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_0652_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_0653_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_0654_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_0655_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_0656_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_0657_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_0658_gNB_FAPI_s0.h5'};
    case 26 % DDSUUUUUUU, multi-slot 2-cell w/ S-slot, POC2_2C_00
        nCell = 2;
        LP = [];
        LP.Cell_Configs = {'TVnr_0204_gNB_FAPI_s0.h5', 'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0612_gNB_FAPI_s0.h5','TVnr_0612_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{2}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_0612_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{2}.channels = {'TVnr_0612_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_0613_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{2}.channels = {'TVnr_0613_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_0614_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{2}.channels = {'TVnr_0614_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_0615_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{2}.channels = {'TVnr_0615_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_0616_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{2}.channels = {'TVnr_0616_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_0617_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{2}.channels = {'TVnr_0617_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_0618_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{2}.channels = {'TVnr_0618_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{2}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_0652_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{2}.channels = {'TVnr_0652_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_0653_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{2}.channels = {'TVnr_0653_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_0654_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{2}.channels = {'TVnr_0654_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_0655_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{2}.channels = {'TVnr_0655_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_0656_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{2}.channels = {'TVnr_0656_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_0657_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{2}.channels = {'TVnr_0657_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_0658_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{2}.channels = {'TVnr_0658_gNB_FAPI_s0.h5'};
    case 27 % DDUUU, multi-slot single-cell, POC2_1C_00
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0612_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_0612_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_0613_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_0614_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_0615_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_0616_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_0617_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_0652_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_0653_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_0654_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_0655_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_0656_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_0657_gNB_FAPI_s0.h5'};
    case 28 % DSUUU, multi-slot single-cell, POC2_1C_00
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0612_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_0612_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_0613_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_0614_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_0615_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_0616_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_0617_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_0652_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_0653_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_0654_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_0655_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_0656_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_0657_gNB_FAPI_s0.h5'};
    case 29 % DSUUU, multi-slot 2-cell, POC2_2C_00
        nCell = 2;
        LP = [];
        LP.Cell_Configs = {'TVnr_0204_gNB_FAPI_s0.h5', 'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0612_gNB_FAPI_s0.h5','TVnr_0612_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{2}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_0612_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{2}.channels = {'TVnr_0612_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_0613_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{2}.channels = {'TVnr_0613_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_0614_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{2}.channels = {'TVnr_0614_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{2}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_0615_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{2}.channels = {'TVnr_0615_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_0616_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{2}.channels = {'TVnr_0616_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_0617_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{2}.channels = {'TVnr_0617_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{2}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_0652_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{2}.channels = {'TVnr_0652_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_0653_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{2}.channels = {'TVnr_0653_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_0654_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{2}.channels = {'TVnr_0654_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{2}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_0655_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{2}.channels = {'TVnr_0655_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_0656_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{2}.channels = {'TVnr_0656_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_0657_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{2}.channels = {'TVnr_0657_gNB_FAPI_s0.h5'};
    case 30 % Mix of modcomp and BFP
        nCell = 2;
        LP = [];
        LP.Cell_Configs = {'TVnr_0190_gNB_FAPI_s0.h5', 'TVnr_0195_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0190_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{2}.channels = {'TVnr_0195_gNB_FAPI_s0.h5'};
    
    case 32 % combo PDCCH TC
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_2001_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_2001_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_2028_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_2033_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_2803_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_2025_gNB_FAPI_s0.h5'};
    case 33 % combo PDSCH TC
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_3230_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_3255_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_3321_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_3323_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_3004_gNB_FAPI_s0.h5'};
    case 34 % combo CSIRS TC
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_4008_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_4008_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_4052_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_4801_gNB_FAPI_s0.h5'};
    case 36 % combo PUCCH TC
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_6001_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_6001_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_6101_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_6201_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_6301_gNB_FAPI_s0.h5'};
    case 37 % combo PUSCH TC
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_7201_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{5}.config{1}.channels = {'TVnr_7201_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_7280_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_7321_gNB_FAPI_s0.h5'};
    case 41 % 4T4R SRS + UL (SU-MIMO)
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_0530_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{4}.config{1}.channels = {'TVnr_8301_gNB_FAPI_s0.h5'}; % SRS placeholder
        LP.SCHED{5}.config{1}.channels = {'TVnr_0531_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_0530_gNB_FAPI_s0.h5'}; % PRACH
        LP.SCHED{14}.config{1}.channels = {'TVnr_8302_gNB_FAPI_s0.h5'}; % SRS placeholder
        LP.SCHED{15}.config{1}.channels = {'TVnr_0531_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_0531_gNB_FAPI_s0.h5'};
    case 42 % 4T4R SRS + UL + DL (SU-MIMO)
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_0530_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0130_gNB_FAPI_s0.h5'}; % SSB
        LP.SCHED{2}.config{1}.channels = {'TVnr_0131_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_0131_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_8301_gNB_FAPI_s0.h5'}; % SRS placeholder
        LP.SCHED{5}.config{1}.channels = {'TVnr_0531_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_0530_gNB_FAPI_s0.h5'}; % PRACH
        LP.SCHED{7}.config{1}.channels = {'TVnr_0131_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_0131_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_0131_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_0131_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_0131_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_0131_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_0131_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_8302_gNB_FAPI_s0.h5'}; % SRS placeholder
        LP.SCHED{15}.config{1}.channels = {'TVnr_0531_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_0531_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_0131_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_0131_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_0131_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_0131_gNB_FAPI_s0.h5'};
    case 43 % 4T4R SRS +  UL(MU-MIMO 4L)
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_0532_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{4}.config{1}.channels = {'TVnr_8301_gNB_FAPI_s0.h5'}; % SRS placeholder
        LP.SCHED{5}.config{1}.channels = {'TVnr_0533_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_0532_gNB_FAPI_s0.h5'}; % PRACH
        LP.SCHED{14}.config{1}.channels = {'TVnr_8302_gNB_FAPI_s0.h5'}; % SRS placeholder
        LP.SCHED{15}.config{1}.channels = {'TVnr_0533_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_0533_gNB_FAPI_s0.h5'};
    case 44 % 4T4R SRS +  UL(MU-MIMO 4L) + DL(MU-MIMO 4L)
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_0532_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0132_gNB_FAPI_s0.h5'}; % SSB
        LP.SCHED{2}.config{1}.channels = {'TVnr_0133_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_0133_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_8301_gNB_FAPI_s0.h5'}; % SRS placeholder
        LP.SCHED{5}.config{1}.channels = {'TVnr_0533_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_0532_gNB_FAPI_s0.h5'}; % PRACH
        LP.SCHED{7}.config{1}.channels = {'TVnr_0133_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_0133_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_0133_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_0133_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_0133_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_0133_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_0133_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_8302_gNB_FAPI_s0.h5'}; % SRS placeholder
        LP.SCHED{15}.config{1}.channels = {'TVnr_0533_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_0533_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_0133_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_0133_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_0133_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_0133_gNB_FAPI_s0.h5'};
    case 45 % 32T32R SRS + UL(MU-MIMO 4L) + DL(MU-MIMO 8L)
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_0534_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0134_gNB_FAPI_s0.h5'}; % SSB
        LP.SCHED{2}.config{1}.channels = {'TVnr_0135_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_0135_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_8401_gNB_FAPI_s0.h5'}; % SRS placeholder
        LP.SCHED{5}.config{1}.channels = {'TVnr_0535_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_0534_gNB_FAPI_s0.h5'}; % PRACH
        LP.SCHED{7}.config{1}.channels = {'TVnr_0135_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_0135_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_0135_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_0135_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_0135_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_0135_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_0135_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_8402_gNB_FAPI_s0.h5'}; % SRS placeholder
        LP.SCHED{15}.config{1}.channels = {'TVnr_0535_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_0535_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_0135_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_0135_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_0135_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_0135_gNB_FAPI_s0.h5'};
    case 46 % 32T32R SRS + UL(MU-MIMO 4L)
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_0534_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{4}.config{1}.channels = {'TVnr_8401_gNB_FAPI_s0.h5'}; % SRS placeholder
        LP.SCHED{5}.config{1}.channels = {'TVnr_0535_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_0534_gNB_FAPI_s0.h5'}; % PRACH
        LP.SCHED{14}.config{1}.channels = {'TVnr_8402_gNB_FAPI_s0.h5'}; % SRS placeholder
        LP.SCHED{15}.config{1}.channels = {'TVnr_0535_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_0535_gNB_FAPI_s0.h5'};
    case 51 % 3 cells, mixed TM and non-TM
        nCell = 3;
        LP = [];
        LP.Cell_Configs = {'TVnr_0126_gNB_FAPI_s0.h5','TVnr_3201_gNB_FAPI_s0.h5', 'TVnr_3340_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0126_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{2}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{3}.channels = {'TVnr_3340_gNB_FAPI_s0.h5'};       
    case 52 % multi-cell pdsch + CSI-RS test (GT-6445)
        nCell = 4;
        LP = [];
        LP.Cell_Configs = {'TVnr_0144_gNB_FAPI_s0.h5', 'TVnr_0142_gNB_FAPI_s0.h5',...
            'TVnr_0143_gNB_FAPI_s0.h5', 'TVnr_0141_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0144_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{2}.channels = {'TVnr_0142_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{3}.channels = {'TVnr_0143_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{4}.channels = {'TVnr_0141_gNB_FAPI_s0.h5'};
    case 53 % 2-cell w/ S-slot (GT-6721)
        nCell = 2;
        LP = [];
        LP.Cell_Configs = {'TVnr_0204_gNB_FAPI_s0.h5', 'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0636_gNB_FAPI_s0.h5', 'TVnr_0637_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'}; % DL + UL
        LP.SCHED{4}.config{2}.channels = {'TVnr_3343_gNB_FAPI_s0.h5', 'TVnr_7299_gNB_FAPI_s0.h5'}; % DL + UL
        LP.SCHED{5}.config{1}.channels = {'TVnr_0636_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{2}.channels = {'TVnr_0637_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_0644_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{2}.channels = {'TVnr_0645_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_0204_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{2}.channels = {'TVnr_0205_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'}; % DL + UL
        LP.SCHED{14}.config{2}.channels = {'TVnr_7299_gNB_FAPI_s0.h5'}; % UL only
        LP.SCHED{15}.config{1}.channels = {'TVnr_0652_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{2}.channels = {'TVnr_0653_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_0660_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{2}.channels = {'TVnr_0661_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_0228_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{2}.channels = {'TVnr_0229_gNB_FAPI_s0.h5'};
    case 54 % OAM parameters: ul_gain_calibration
        nCell = 4;
        LP = [];
        LP.Cell_Configs = {'TVnr_7201_gNB_FAPI_s0.h5','TVnr_7350_gNB_FAPI_s0.h5','TVnr_7351_gNB_FAPI_s0.h5','TVnr_7352_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_7201_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{2}.channels = {'TVnr_7350_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{3}.channels = {'TVnr_7351_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{4}.channels = {'TVnr_7352_gNB_FAPI_s0.h5'};
    case 55 % OAM parameters: pusch_prb_stride, max_amp_ul
        nCell = 2;
        LP = [];
        LP.Cell_Configs = {'TVnr_7201_gNB_FAPI_s0.h5','TVnr_7353_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_7201_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{2}.channels = {'TVnr_7353_gNB_FAPI_s0.h5'};
    case 56 % heterogeneous PUCCH
        nCell = 2;
        LP = [];
        LP.Cell_Configs = {'TVnr_6375_gNB_FAPI_s0.h5','TVnr_6380_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_6375_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{2}.channels = {'TVnr_6380_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_6301_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{2}.channels = {'TVnr_6604_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_6598_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{2}.channels = {'TVnr_6101_gNB_FAPI_s0.h5'};
    case 57 % adaptive re-tx harq2
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnrPUSCH_HARQ2_7357_gNB_FAPI_s0.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, 20, nCell);
        LP.SCHED{5}.config{1}.channels = {'TVnrPUSCH_HARQ2_7357_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnrPUSCH_HARQ2_7357_gNB_FAPI_s1.h5'};
    case 58 % adaptive re-tx harq4
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnrPUSCH_HARQ4_7910_gNB_FAPI_s0.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, 40, nCell);
        LP.SCHED{5}.config{1}.channels = {'TVnrPUSCH_HARQ4_7910_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnrPUSCH_HARQ4_7910_gNB_FAPI_s1.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnrPUSCH_HARQ4_7910_gNB_FAPI_s2.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnrPUSCH_HARQ4_7910_gNB_FAPI_s3.h5'};
    case 59 % multi-slot single-cell HARQ 4-rx TC-7710
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnrPUSCH_HARQ4_7710_gNB_FAPI_s0.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{4}.config{1}.channels =  {'TVnrPUSCH_HARQ4_7710_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnrPUSCH_HARQ4_7710_gNB_FAPI_s1.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnrPUSCH_HARQ4_7710_gNB_FAPI_s2.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnrPUSCH_HARQ4_7710_gNB_FAPI_s3.h5'};
    case 60 % even and odd frames for SRS different startRB
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_8025_gNB_FAPI_s5.h5'}; % even frame, 0-based index
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{6}.config{1}.channels =  {'TVnr_8025_gNB_FAPI_s5.h5'}; % even frame, 0-based index
        LP.SCHED{26}.config{1}.channels = {'TVnr_8034_gNB_FAPI_s5.h5'}; % odd frame, 1-based index
    case 61 % DSUUU, multi-slot single-cell, 1 UE
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_7201_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_7201_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_7201_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_7201_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_7201_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_7201_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_7201_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_7201_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_7201_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_7201_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_7201_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_7201_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_7201_gNB_FAPI_s0.h5'};
    case 62 % DSUUU, multi-slot single-cell, 2 UEs
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_7235_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_7235_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_7235_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_7235_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_7235_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_7235_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_7235_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_7235_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_7235_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_7235_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_7235_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_7235_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_7235_gNB_FAPI_s0.h5'};   
    case 63 % DSUUU, multi-slot single-cell, 4 UEs
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_7250_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_7250_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_7250_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_7250_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_7250_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_7250_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_7250_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_7250_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_7250_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_7250_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_7250_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_7250_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_7250_gNB_FAPI_s0.h5'};  
    case 64 % DSUUU, multi-slot single-cell, 8 UEs ~ 16 UEs
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_7327_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_7327_gNB_FAPI_s0.h5'}; % 8 UEs
        LP.SCHED{4}.config{1}.channels = {'TVnr_7328_gNB_FAPI_s0.h5'}; % 8 UEs
        LP.SCHED{5}.config{1}.channels = {'TVnr_7329_gNB_FAPI_s0.h5'}; % 16 UEs
        LP.SCHED{6}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_7327_gNB_FAPI_s0.h5'}; % 8 UEs
        LP.SCHED{9}.config{1}.channels = {'TVnr_7328_gNB_FAPI_s0.h5'}; % 8 UEs
        LP.SCHED{10}.config{1}.channels = {'TVnr_7329_gNB_FAPI_s0.h5'}; % 16 UEs
        LP.SCHED{11}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_7327_gNB_FAPI_s0.h5'}; % 8 UEs
        LP.SCHED{14}.config{1}.channels = {'TVnr_7328_gNB_FAPI_s0.h5'}; % 8 UEs
        LP.SCHED{15}.config{1}.channels = {'TVnr_7329_gNB_FAPI_s0.h5'}; % 16 UEs
        LP.SCHED{16}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_7327_gNB_FAPI_s0.h5'}; % 8 UEs
        LP.SCHED{19}.config{1}.channels = {'TVnr_7328_gNB_FAPI_s0.h5'}; % 8 UEs
        LP.SCHED{20}.config{1}.channels = {'TVnr_7329_gNB_FAPI_s0.h5'}; % 16 UEs  
    case 65 % DSUUU, multi-slot single-cell, 1 UE, multi-UL-channel
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0508_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_0508_gNB_FAPI_s0.h5'}; % PRACH
        LP.SCHED{4}.config{1}.channels = {'TVnr_0509_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_0510_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_0511_gNB_FAPI_s0.h5'}; 
        LP.SCHED{9}.config{1}.channels = {'TVnr_0512_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_0513_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_0514_gNB_FAPI_s0.h5'}; 
        LP.SCHED{14}.config{1}.channels = {'TVnr_0515_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_0516_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_0517_gNB_FAPI_s0.h5'}; 
        LP.SCHED{19}.config{1}.channels = {'TVnr_0518_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_0519_gNB_FAPI_s0.h5'};
    case 66 % DSUUU, multi-slot single-cell, 2 UEs, multi-UL-channel
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0619_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_0619_gNB_FAPI_s0.h5'}; % PRACH
        LP.SCHED{4}.config{1}.channels = {'TVnr_0620_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_0621_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_0622_gNB_FAPI_s0.h5'}; 
        LP.SCHED{9}.config{1}.channels = {'TVnr_0623_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_0624_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_0625_gNB_FAPI_s0.h5'}; 
        LP.SCHED{14}.config{1}.channels = {'TVnr_0626_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_0627_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_0628_gNB_FAPI_s0.h5'}; 
        LP.SCHED{19}.config{1}.channels = {'TVnr_0629_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_0630_gNB_FAPI_s0.h5'};   
    case 67 % DSUUU, multi-slot single-cell, 4 UEs, multi-UL-channel
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0536_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_0536_gNB_FAPI_s0.h5'}; % PRACH
        LP.SCHED{4}.config{1}.channels = {'TVnr_0537_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_0538_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_0539_gNB_FAPI_s0.h5'}; 
        LP.SCHED{9}.config{1}.channels = {'TVnr_0540_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_0541_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_0542_gNB_FAPI_s0.h5'}; 
        LP.SCHED{14}.config{1}.channels = {'TVnr_0543_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_0544_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_3201_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_3212_gNB_FAPI_s0.h5', 'TVnr_7258_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_0545_gNB_FAPI_s0.h5'}; 
        LP.SCHED{19}.config{1}.channels = {'TVnr_0546_gNB_FAPI_s0.h5'}; 
        LP.SCHED{20}.config{1}.channels = {'TVnr_0547_gNB_FAPI_s0.h5'};  
    case 68 % multi-slot single-cell HARQ 4-rx TC-7711
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnrPUSCH_HARQ4_7711_gNB_FAPI_s0.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{4}.config{1}.channels =  {'TVnrPUSCH_HARQ4_7711_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnrPUSCH_HARQ4_7711_gNB_FAPI_s1.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnrPUSCH_HARQ4_7711_gNB_FAPI_s2.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnrPUSCH_HARQ4_7711_gNB_FAPI_s3.h5'};        
    case 69 % Simultaneous DL/UL in S-slot(PDCCH [0], PDSCH [1:4], PUSCH [10:13])
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_0191_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0191_gNB_FAPI_s0.h5', 'TVnr_7598_gNB_FAPI_s0.h5'};

% 32T32R launch patterns (70-89)
 
    case 70 %32T32R BFW + PDSCH slots 0-1
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_3850_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_3850_gNB_FAPI_s0.h5'};           
    case 73 %%32T32R BFW + PDSCH on all slots (DDDSUUDDDD)
        %Slot #     % 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
        %Slot type  % D   D   D   S   U   U   D   D   D   D   D   D   D   S   U   U   D   D   D   D
        %BFW type   % B   B   -   -   -   B   B   B   B   B   B   B   -   -   -   B   B   B   B   B
        %BFW map    %   \   \       \   \   \   \   \   \   \   \   \       \   \   \   \   \   \   \
        %PXSCH type % D   D   D   -   -   -   D   D   D   D   D   D   D   -   -   -   D   D   D   D
        %
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_3850_gNB_FAPI_s0.h5',};
        LP = init_launchPattern(LP, nSlot, nCell);
        % Subframe 0
        LP.SCHED{1}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5',};
        LP.SCHED{7}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        %Subframe 1
        LP.SCHED{11}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        
    case 74 %%32T32R BFW + PUSCH on all slots (DDDSUUDDDD)
        %Slot #     % 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
        %Slot type  % D   D   D   S   U   U   D   D   D   D   D   D   D   S   U   U   D   D   D   D
        %BFW type   % -   -   -   UB  UB  -   -   -   -   -   -   -   -   UB  UB  -   -   -   -   -
        %BFW map    %   \   \       \   \   \   \   \   \   \   \   \       \   \   \   \   \   \   \
        %PXSCH type % -   -   -   -   U   U   -   -   -   -   -   -   -   -   U   U   -   -   -   -
        %
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_7851_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        % Subframe 0        
        LP.SCHED{4}.config{1}.channels = {'TVnr_9001_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_9001_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_7851_gNB_FAPI_s0.h5'};        
        %Subframe 1
        LP.SCHED{14}.config{1}.channels = {'TVnr_9001_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_9001_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_7851_gNB_FAPI_s0.h5'};

    case 75 %30MHz  %%32T32R BFW + PDSCH on all slots (DDDSUUDDDD)
        %Slot #     % 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
        %Slot type  % D   D   D   S   U   U   D   D   D   D   D   D   D   S   U   U   D   D   D   D
        %BFW type   % B   B   -   -   -   B   B   B   B   B   B   B   -   -   -   B   B   B   B   B
        %BFW map    %   \   \       \   \   \   \   \   \   \   \   \       \   \   \   \   \   \   \
        %PXSCH type % D   D   D   -   -   -   D   D   D   D   D   D   D   -   -   -   D   D   D   D
        %
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_3851_gNB_FAPI_s0.h5',};
        LP = init_launchPattern(LP, nSlot, nCell);
        % Subframe 0
        LP.SCHED{1}.config{1}.channels = {'TVnr_9228_gNB_FAPI_s0.h5', 'TVnr_3851_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_9228_gNB_FAPI_s0.h5', 'TVnr_3851_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_3851_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_9228_gNB_FAPI_s0.h5',};
        LP.SCHED{7}.config{1}.channels = {'TVnr_9228_gNB_FAPI_s0.h5', 'TVnr_3851_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_9228_gNB_FAPI_s0.h5', 'TVnr_3851_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_9228_gNB_FAPI_s0.h5', 'TVnr_3851_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_9228_gNB_FAPI_s0.h5', 'TVnr_3851_gNB_FAPI_s0.h5'};
        %Subframe 1
        LP.SCHED{11}.config{1}.channels = {'TVnr_9228_gNB_FAPI_s0.h5', 'TVnr_3851_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_9228_gNB_FAPI_s0.h5', 'TVnr_3851_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_3851_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_9228_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_9228_gNB_FAPI_s0.h5', 'TVnr_3851_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_9228_gNB_FAPI_s0.h5', 'TVnr_3851_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_9228_gNB_FAPI_s0.h5', 'TVnr_3851_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9228_gNB_FAPI_s0.h5', 'TVnr_3851_gNB_FAPI_s0.h5'};

    case 76 % 30 MHz %%32T32R BFW + PUSCH on all slots (DDDSUUDDDD)
        %Slot #     % 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
        %Slot type  % D   D   D   S   U   U   D   D   D   D   D   D   D   S   U   U   D   D   D   D
        %BFW type   % -   -   -   UB  UB  -   -   -   -   -   -   -   -   UB  UB  -   -   -   -   -
        %BFW map    %   \   \       \   \   \   \   \   \   \   \   \       \   \   \   \   \   \   \
        %PXSCH type % -   -   -   -   U   U   -   -   -   -   -   -   -   -   U   U   -   -   -   -
        %
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_7852_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        % Subframe 0        
        LP.SCHED{4}.config{1}.channels = {'TVnr_9229_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_9229_gNB_FAPI_s0.h5', 'TVnr_7852_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_7852_gNB_FAPI_s0.h5'};        
        %Subframe 1
        LP.SCHED{14}.config{1}.channels = {'TVnr_9229_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_9229_gNB_FAPI_s0.h5', 'TVnr_7852_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_7852_gNB_FAPI_s0.h5'};
    case 77  %50 MHz %%32T32R BFW + PDSCH on all slots (DDDSUUDDDD)
        %Slot #     % 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
        %Slot type  % D   D   D   S   U   U   D   D   D   D   D   D   D   S   U   U   D   D   D   D
        %BFW type   % B   B   -   -   -   B   B   B   B   B   B   B   -   -   -   B   B   B   B   B
        %BFW map    %   \   \       \   \   \   \   \   \   \   \   \       \   \   \   \   \   \   \
        %PXSCH type % D   D   D   -   -   -   D   D   D   D   D   D   D   -   -   -   D   D   D   D
        %
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_3852_gNB_FAPI_s0.h5',};
        LP = init_launchPattern(LP, nSlot, nCell);
        % Subframe 0
        LP.SCHED{1}.config{1}.channels = {'TVnr_9230_gNB_FAPI_s0.h5', 'TVnr_3852_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_9230_gNB_FAPI_s0.h5', 'TVnr_3852_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_3852_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_9230_gNB_FAPI_s0.h5',};
        LP.SCHED{7}.config{1}.channels = {'TVnr_9230_gNB_FAPI_s0.h5', 'TVnr_3852_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_9230_gNB_FAPI_s0.h5', 'TVnr_3852_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_9230_gNB_FAPI_s0.h5', 'TVnr_3852_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_9230_gNB_FAPI_s0.h5', 'TVnr_3852_gNB_FAPI_s0.h5'};
        %Subframe 1
        LP.SCHED{11}.config{1}.channels = {'TVnr_9230_gNB_FAPI_s0.h5', 'TVnr_3852_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_9230_gNB_FAPI_s0.h5', 'TVnr_3852_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_3852_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_9230_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_9230_gNB_FAPI_s0.h5', 'TVnr_3852_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_9230_gNB_FAPI_s0.h5', 'TVnr_3852_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_9230_gNB_FAPI_s0.h5', 'TVnr_3852_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9230_gNB_FAPI_s0.h5', 'TVnr_3852_gNB_FAPI_s0.h5'};
    
    case 78 %50 MHz %%32T32R BFW + PUSCH on all slots (DDDSUUDDDD)
        %Slot #     % 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
        %Slot type  % D   D   D   S   U   U   D   D   D   D   D   D   D   S   U   U   D   D   D   D
        %BFW type   % -   -   -   UB  UB  -   -   -   -   -   -   -   -   UB  UB  -   -   -   -   -
        %BFW map    %   \   \       \   \   \   \   \   \   \   \   \       \   \   \   \   \   \   \
        %PXSCH type % -   -   -   -   U   U   -   -   -   -   -   -   -   -   U   U   -   -   -   -
        %
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_7853_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        % Subframe 0        
        LP.SCHED{4}.config{1}.channels = {'TVnr_9231_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_9231_gNB_FAPI_s0.h5', 'TVnr_7853_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_7853_gNB_FAPI_s0.h5'};        
        %Subframe 1
        LP.SCHED{14}.config{1}.channels = {'TVnr_9231_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_9231_gNB_FAPI_s0.h5', 'TVnr_7853_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_7853_gNB_FAPI_s0.h5'};
    case 79 % 39232 with CSI-RS
        %Slot #     % 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
        %Slot type  % D   D   D   S   U   U   D   D   D   D   D   D   D   S   U   U   D   D   D   D
        %BFW type   % B   B   -   -   -   B   B   B   B   B   B   B   -   -   -   B   B   B   B   B
        %BFW map    %   \   \       \   \   \   \   \   \   \   \   \       \   \   \   \   \   \   \
        %PXSCH type % D   D   D   -   -   -   D   D   D   D   D   D   D   -   -   -   D   D   D   D
        %
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_3853_gNB_FAPI_s0.h5',};
        LP = init_launchPattern(LP, nSlot, nCell);
        % Subframe 0
        LP.SCHED{1}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3853_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3853_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_3853_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5',};
        LP.SCHED{7}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3853_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3853_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3853_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3853_gNB_FAPI_s0.h5'};
        %Subframe 1
        LP.SCHED{11}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3853_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3853_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_3853_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3853_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3853_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3853_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3853_gNB_FAPI_s0.h5'};

    case 80 % Partial BW allocation PDSCH
        %Slot #     % 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
        %Slot type  % D   D   D   S   U   U   D   D   D   D   D   D   D   S   U   U   D   D   D   D
        %BFW type   % B   B   -   -   -   B   B   B   B   B   B   B   -   -   -   B   B   B   B   B
        %BFW map    %   \   \       \   \   \   \   \   \   \   \   \       \   \   \   \   \   \   \
        %PXSCH type % D   D   D   -   -   -   D   D   D   D   D   D   D   -   -   -   D   D   D   D
        %
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_3854_gNB_FAPI_s0.h5',};
        LP = init_launchPattern(LP, nSlot, nCell);
        % Subframe 0
        LP.SCHED{1}.config{1}.channels = {'TVnr_9234_gNB_FAPI_s0.h5', 'TVnr_3854_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_9234_gNB_FAPI_s0.h5', 'TVnr_3854_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_3854_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_9234_gNB_FAPI_s0.h5',};
        LP.SCHED{7}.config{1}.channels = {'TVnr_9234_gNB_FAPI_s0.h5', 'TVnr_3854_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_9234_gNB_FAPI_s0.h5', 'TVnr_3854_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_9234_gNB_FAPI_s0.h5', 'TVnr_3854_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_9234_gNB_FAPI_s0.h5', 'TVnr_3854_gNB_FAPI_s0.h5'};
        %Subframe 1
        LP.SCHED{11}.config{1}.channels = {'TVnr_9234_gNB_FAPI_s0.h5', 'TVnr_3854_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_9234_gNB_FAPI_s0.h5', 'TVnr_3854_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_3854_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_9234_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_9234_gNB_FAPI_s0.h5', 'TVnr_3854_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_9234_gNB_FAPI_s0.h5', 'TVnr_3854_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_9234_gNB_FAPI_s0.h5', 'TVnr_3854_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9234_gNB_FAPI_s0.h5', 'TVnr_3854_gNB_FAPI_s0.h5'};

    case 81 %Partial BW %%32T32R BFW + PUSCH on all slots (DDDSUUDDDD)
        %Slot #     % 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
        %Slot type  % D   D   D   S   U   U   D   D   D   D   D   D   D   S   U   U   D   D   D   D
        %BFW type   % -   -   -   UB  UB  -   -   -   -   -   -   -   -   UB  UB  -   -   -   -   -
        %BFW map    %   \   \       \   \   \   \   \   \   \   \   \       \   \   \   \   \   \   \
        %PXSCH type % -   -   -   -   U   U   -   -   -   -   -   -   -   -   U   U   -   -   -   -
        %
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_7854_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        % Subframe 0        
        LP.SCHED{4}.config{1}.channels = {'TVnr_9235_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_9235_gNB_FAPI_s0.h5', 'TVnr_7854_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_7854_gNB_FAPI_s0.h5'};        
        %Subframe 1
        LP.SCHED{14}.config{1}.channels = {'TVnr_9235_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_9235_gNB_FAPI_s0.h5', 'TVnr_7854_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_7854_gNB_FAPI_s0.h5'};

    
    

    case 82 %%32T32R BFW + PDSCH/PUSCH on all slots (DDDSUUDDDD)
        %Slot #     % 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
        %Slot type  % D   D   D   S   U   U   D   D   D   D   D   D   D   S   U   U   D   D   D   D
        %BFW type   % B   B   -   UB  UB  B   B   B   B   B   B   B   -   UB  UB  B   B   B   B   B
        %BFW map    %   \   \       \   \   \   \   \   \   \   \   \       \   \   \   \   \   \   \
        %PXSCH type % D   D   D   -   U   U   D   D   D   D   D   D   D   -   U   U   D   D   D   D
        %
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_3850_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        % Subframe 0
        LP.SCHED{1}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_9001_gNB_FAPI_s0.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_9001_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        %Subframe 1
        LP.SCHED{11}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_9001_gNB_FAPI_s0.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_9001_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};

    case 83 %%40 slot pattern: 32T32R SRS + BFW + PDSCH/PUSCH on all slots (DDDSUUDDDD)
        % SRS TVs 8412 & 8413 to match BFW 9000 & 9001
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_9226_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        %Subframe 0
        LP.SCHED{4}.config{1}.channels = {'TVnr_8412_gNB_FAPI_s0.h5'};
        %Subframe 1
        LP.SCHED{14}.config{1}.channels = {'TVnr_8413_gNB_FAPI_s0.h5'};
        %Subframe 2
        LP.SCHED{21}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9227_gNB_FAPI_s0.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9227_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        %Subframe 3
        LP.SCHED{31}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9227_gNB_FAPI_s0.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9227_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};

    case 84 %% 30 MHz 40 slot pattern: 32T32R SRS + BFW + PDSCH/PUSCH on all slots (DDDSUUDDDD)
        % SRS TVs 8412 & 8413 to match BFW 9000 & 9001
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_9226_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        %Subframe 0
        LP.SCHED{4}.config{1}.channels = {'TVnr_8412_gNB_FAPI_s0.h5'};
        %Subframe 1
        LP.SCHED{14}.config{1}.channels = {'TVnr_8413_gNB_FAPI_s0.h5'};
        %Subframe 2
        LP.SCHED{21}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9227_gNB_FAPI_s0.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9227_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        %Subframe 3
        LP.SCHED{31}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9227_gNB_FAPI_s0.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9227_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};

    case 85 %% 50 MHz 40 slot pattern: 32T32R SRS + BFW + PDSCH/PUSCH on all slots (DDDSUUDDDD)
        % SRS TVs 8412 & 8413 to match BFW 9000 & 9001
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_9226_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        %Subframe 0
        LP.SCHED{4}.config{1}.channels = {'TVnr_8412_gNB_FAPI_s0.h5'};
        %Subframe 1
        LP.SCHED{14}.config{1}.channels = {'TVnr_8413_gNB_FAPI_s0.h5'};
        %Subframe 2
        LP.SCHED{21}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9227_gNB_FAPI_s0.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9227_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        %Subframe 3
        LP.SCHED{31}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9227_gNB_FAPI_s0.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9227_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
    case 86 %%80 slot pattern: 32T32R SRS + BFW + PDSCH/PUSCH (DDDSUUDDDD)
        % SRS TVs 8412 & 8413 to match BFW 9000 & 9001
        nCell = 1;
        LP = [];
        nSlot = 80;
        LP.Cell_Configs = {'TVnr_9000_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        %Subframe 0
        LP.SCHED{4}.config{1}.channels = {'TVnr_8412_gNB_FAPI_s0.h5'};
        %Subframe 1
        %Subframe 2
        LP.SCHED{20}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_8413_gNB_FAPI_s0.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        %Subframe 3
        LP.SCHED{31}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        %Subframe 4
        LP.SCHED{41}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{42}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{43}.config{1}.channels = {'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{44}.config{1}.channels = {'TVnr_9001_gNB_FAPI_s0.h5'};
        LP.SCHED{45}.config{1}.channels = {'TVnr_9001_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{46}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{47}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{48}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{49}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{50}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        %Subframe 5
        LP.SCHED{51}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{52}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{53}.config{1}.channels = {'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{54}.config{1}.channels = {'TVnr_9001_gNB_FAPI_s0.h5'};
        LP.SCHED{55}.config{1}.channels = {'TVnr_9001_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{56}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{57}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{58}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{59}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{60}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        %Subframe 6
        LP.SCHED{61}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{62}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{63}.config{1}.channels = {'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{64}.config{1}.channels = {'TVnr_8412_gNB_FAPI_s0.h5', 'TVnr_9001_gNB_FAPI_s0.h5'};
        LP.SCHED{65}.config{1}.channels = {'TVnr_9001_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{66}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{67}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{68}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{69}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{70}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        %Subframe 7
        LP.SCHED{71}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{72}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{73}.config{1}.channels = {'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{74}.config{1}.channels = {'TVnr_9001_gNB_FAPI_s0.h5'};
        LP.SCHED{75}.config{1}.channels = {'TVnr_9001_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{76}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_7851_gNB_FAPI_s0.h5'};
        LP.SCHED{77}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{78}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{79}.config{1}.channels = {'TVnr_9000_gNB_FAPI_s0.h5', 'TVnr_3850_gNB_FAPI_s0.h5'};
        LP.SCHED{80}.config{1}.channels = {'TVnr_3850_gNB_FAPI_s0.h5'};

% 64T64R launch patterns (90-120)
    case 90 %64TR BFW + PDSCH slots 0-1
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_3870_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_9236_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_3870_gNB_FAPI_s0.h5'};

    case 91 %64TR BFW + PUSCH slots 4-5
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_7870_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{5}.config{1}.channels = {'TVnr_9237_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_7870_gNB_FAPI_s0.h5'};

    case 92 %64TR BFW + PDSCH slots 0-1
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_3871_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_9238_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_3871_gNB_FAPI_s0.h5'};

    case 93 %64TR BFW + PUSCH slots 4-5
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_7871_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{5}.config{1}.channels = {'TVnr_9239_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_7871_gNB_FAPI_s0.h5'};

    case 94 %64TR BFW + PDSCH slots 0-1
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_3872_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_9240_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_3872_gNB_FAPI_s0.h5'};

    case 95 %64TR BFW + PUSCH slots 4-5
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_7872_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{5}.config{1}.channels = {'TVnr_9241_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_7872_gNB_FAPI_s0.h5'};

    case 96 %64TR BFW + PDSCH slots 0-1
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_3873_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_9242_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_3873_gNB_FAPI_s0.h5'};

    case 97 %64TR BFW + PUSCH slots 4-5
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_7873_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{5}.config{1}.channels = {'TVnr_9243_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_7873_gNB_FAPI_s0.h5'};

    case 98 %64TR BFW + PDSCH slots 0-1
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_3874_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_9244_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_3874_gNB_FAPI_s0.h5'};

    case 99 %64TR BFW + PUSCH slots 4-5
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_7874_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{5}.config{1}.channels = {'TVnr_9245_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_7874_gNB_FAPI_s0.h5'};

    case 100 %64TR BFW + PDSCH slots 0-1
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_3875_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_9246_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_3875_gNB_FAPI_s0.h5'};

    case 101 %64TR BFW + PUSCH slots 4-5
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_7875_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{5}.config{1}.channels = {'TVnr_9247_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_7875_gNB_FAPI_s0.h5'};

    case 102 %64TR BFW + PDSCH slots 0-1
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_3876_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_9248_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_3876_gNB_FAPI_s0.h5'};


    case 103 %%40 slot pattern: 64TR SRS + BFW + PDSCH/PUSCH on all slots (DDDSUUDDDD)
        params.PDSCH_TV  = 'TVnr_3870_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8512_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9236_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7870_gNB_FAPI_s0.h5';
        params.BFW_UL_TV = 'TVnr_9237_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);

case 104 %%40 slot pattern: 64TR SRS + BFW + PDSCH/PUSCH on all slots (DDDSUUDDDD)
        params.PDSCH_TV  = 'TVnr_3871_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8514_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9238_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7871_gNB_FAPI_s0.h5';
        params.BFW_UL_TV = 'TVnr_9239_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);

case 105 %%40 slot pattern: 64TR SRS + BFW + PDSCH/PUSCH on all slots (DDDSUUDDDD)
        params.PDSCH_TV  = 'TVnr_3872_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8516_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9240_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7872_gNB_FAPI_s0.h5';
        params.BFW_UL_TV = 'TVnr_9241_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);

case 106 %%40 slot pattern: 64TR SRS + BFW + PDSCH/PUSCH on all slots (DDDSUUDDDD)
        params.PDSCH_TV  = 'TVnr_3873_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8518_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9242_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7873_gNB_FAPI_s0.h5';
        params.BFW_UL_TV = 'TVnr_9243_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);


case 107 %%40 slot pattern: 64TR SRS + BFW + PDSCH/PUSCH on all slots (DDDSUUDDDD)
        params.PDSCH_TV  = 'TVnr_3874_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8520_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9244_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7874_gNB_FAPI_s0.h5';
        params.BFW_UL_TV = 'TVnr_9245_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);

case 108 %%40 slot pattern: 64TR SRS + BFW + PDSCH/PUSCH on all slots (DDDSUUDDDD)
        params.PDSCH_TV  = 'TVnr_3875_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8522_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9246_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7875_gNB_FAPI_s0.h5';
        params.BFW_UL_TV = 'TVnr_9247_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);

case 109 %%40 slot pattern: 64TR SRS + BFW + PDSCH/PUSCH on all slots (DDDSUUDDDD)
        params.PDSCH_TV  = 'TVnr_3876_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8524_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9248_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7875_gNB_FAPI_s0.h5';
        params.BFW_UL_TV = 'TVnr_9249_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);

    case 110
        params.data_TV = 'TVnr_7879_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9259_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 111
        params.data_TV = 'TVnr_7880_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9261_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 112
        params.data_TV = 'TVnr_3879_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9260_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 113
        params.data_TV = 'TVnr_3880_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9262_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 114
        params.PDSCH_TV  = 'TVnr_3879_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8536_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9260_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7879_gNB_FAPI_s0.h5';
        params.BFW_UL_TV = 'TVnr_9259_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);
    case 115
        params.PDSCH_TV  = 'TVnr_3880_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8538_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9262_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7880_gNB_FAPI_s0.h5';
        params.BFW_UL_TV = 'TVnr_9261_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);


    case 116
        params.data_TV = 'TVnr_7881_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9265_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 117
        params.data_TV = 'TVnr_3882_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9266_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 118
        params.data_TV = 'TVnr_7882_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9267_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 119
        params.data_TV = 'TVnr_3883_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9268_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 120
        params.data_TV = 'TVnr_7883_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9269_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 121
        params.data_TV = 'TVnr_3884_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9270_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 122
        params.data_TV = 'TVnr_7884_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9271_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 123
        params.data_TV = 'TVnr_3885_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9272_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 124
        params.data_TV = 'TVnr_7885_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9273_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 125
        params.data_TV = 'TVnr_3886_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9274_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 126
        params.data_TV = 'TVnr_7886_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9275_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 127
        params.data_TV = 'TVnr_3887_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9276_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 128
        params.data_TV = 'TVnr_7887_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9277_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 129
        params.data_TV = 'TVnr_3888_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9278_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 130
        params.BFW_UL_TV = 'TVnr_9265_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7881_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8542_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9266_gNB_FAPI_s0.h5';
        params.PDSCH_TV  = 'TVnr_3882_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);
    case 131
        params.BFW_UL_TV = 'TVnr_9267_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7882_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8544_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9268_gNB_FAPI_s0.h5';
        params.PDSCH_TV  = 'TVnr_3883_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);
    case 132
        params.BFW_UL_TV = 'TVnr_9269_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7883_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8546_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9270_gNB_FAPI_s0.h5';
        params.PDSCH_TV  = 'TVnr_3884_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);
    case 133
        params.BFW_UL_TV = 'TVnr_9271_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7884_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8548_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9272_gNB_FAPI_s0.h5';
        params.PDSCH_TV  = 'TVnr_3885_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);
    case 134
        params.BFW_UL_TV = 'TVnr_9273_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7885_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8550_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9274_gNB_FAPI_s0.h5';
        params.PDSCH_TV  = 'TVnr_3886_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);
    case 135
        params.BFW_UL_TV = 'TVnr_9275_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7886_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8552_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9276_gNB_FAPI_s0.h5';
        params.PDSCH_TV  = 'TVnr_3887_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);
    case 136
        params.BFW_UL_TV = 'TVnr_9277_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7887_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8555_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9278_gNB_FAPI_s0.h5';
        params.PDSCH_TV  = 'TVnr_3888_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);

    case 137
        params.data_TV = 'TVnr_3881_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9264_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 138
        params.data_TV = 'TVnr_7878_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9263_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 139
        params.data_TV = 'TVnr_7876_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9251_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 140
        params.data_TV = 'TVnr_3877_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9250_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 141
        params.data_TV = 'TVnr_7877_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9257_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 142
        params.data_TV = 'TVnr_3878_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9256_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 143
        params.BFW_UL_TV = 'TVnr_9263_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7878_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8540_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9264_gNB_FAPI_s0.h5';
        params.PDSCH_TV  = 'TVnr_3881_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);
    case 144
        params.BFW_UL_TV = 'TVnr_9251_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7876_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8526_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9250_gNB_FAPI_s0.h5';
        params.PDSCH_TV  = 'TVnr_3877_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);
    case 145
        params.BFW_UL_TV = 'TVnr_9257_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7877_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8532_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9256_gNB_FAPI_s0.h5';
        params.PDSCH_TV  = 'TVnr_3878_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);
    case 146
        params.data_TV = 'TVnr_3889_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9306_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 147
        params.data_TV = 'TVnr_7888_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9307_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 148
        params.BFW_UL_TV = 'TVnr_9307_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7888_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8558_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9306_gNB_FAPI_s0.h5';
        params.PDSCH_TV  = 'TVnr_3889_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);
    case 149
        params.data_TV = 'TVnr_3890_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9294_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 150
        params.BFW_UL_TV = 'TVnr_9277_gNB_FAPI_s0.h5';
        params.PUSCH_TV  = 'TVnr_7887_gNB_FAPI_s0.h5';
        params.SRS_DL_TV = 'TVnr_8555_gNB_FAPI_s0.h5';
        params.BFW_DL_TV = 'TVnr_9294_gNB_FAPI_s0.h5';
        params.PDSCH_TV  = 'TVnr_3890_gNB_FAPI_s0.h5';
        LP = populateLPhelper40Slot(params);
    case 151 % nvbug 5368243 non-sequential order of DMRS ports PDSCH
        params.data_TV = 'TVnr_3897_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9812_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 152 % nvbug 5368243 non-sequential order of DMRS ports PDSCH
        params.data_TV = 'TVnr_3898_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9814_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 153 % nvbug 5368243 non-sequential order of DMRS ports PDSCH+CSI-RS
        params.data_TV = 'TVnr_DLMIX_20830_gNB_FAPI_s1.h5';
        params.bfw_TV  = 'TVnr_9812_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 154 % nvbug 5368243 non-sequential order of DMRS ports PUSCH
        params.data_TV = 'TVnr_7896_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9811_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 155 % multi-slot single-cell HARQ 4-rx TC-7895 (64TR)
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnrPUSCH_HARQ4_7895_gNB_FAPI_s0.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{4}.config{1}.channels = {'TVnrPUSCH_HARQ4_7895_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnrPUSCH_HARQ4_7895_gNB_FAPI_s1.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnrPUSCH_HARQ4_7895_gNB_FAPI_s2.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnrPUSCH_HARQ4_7895_gNB_FAPI_s3.h5'};
    case 156 % 24DL
        params.data_TV = 'TVnr_3899_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9816_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 157 % 32DL
        params.data_TV = 'TVnr_3900_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9818_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 158 % 32DL+CSI-RS
        params.data_TV = 'TVnr_DLMIX_20831_gNB_FAPI_s1.h5';
        params.bfw_TV  = 'TVnr_9818_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 159 % multi-slot single-cell UL TTI bundling 6-rx TC-7898 (64TR)
        nCell = 1;
        nSlot = 40;
        LP = [];
        LP.Cell_Configs = {'TVnrPUSCH_HARQ4_7898_gNB_FAPI_s0.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{5}.config{1}.channels  = {'TVnrPUSCH_HARQ4_7898_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels  = {'TVnrPUSCH_HARQ4_7898_gNB_FAPI_s1.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnrPUSCH_HARQ4_7898_gNB_FAPI_s2.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnrPUSCH_HARQ4_7898_gNB_FAPI_s3.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnrPUSCH_HARQ4_7898_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnrPUSCH_HARQ4_7898_gNB_FAPI_s5.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnrPUSCH_HARQ4_7898_gNB_FAPI_s6.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnrPUSCH_HARQ4_7898_gNB_FAPI_s3.h5'};
    case 160 % multi-slot single-cell one UL TTI bundling UE with 3-rx + three normal UEs with 0-rx TC-7899 (64TR)
        nCell = 1;
        nSlot = 20;
        LP = [];
        LP.Cell_Configs = {'TVnrPUSCH_HARQ4_7899_gNB_FAPI_s0.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{5}.config{1}.channels  = {'TVnrPUSCH_HARQ4_7899_gNB_FAPI_s0.h5'};
        LP.SCHED{6}.config{1}.channels  = {'TVnrPUSCH_HARQ4_7899_gNB_FAPI_s1.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnrPUSCH_HARQ4_7899_gNB_FAPI_s2.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnrPUSCH_HARQ4_7899_gNB_FAPI_s3.h5'};

    case 165 % multi-slot single-cell weighted average CFO estimation TC-7897 (64TR)
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnrPUSCH_HARQ4_7897_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{4}.config{1}.channels = {'TVnrPUSCH_HARQ4_7897_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnrPUSCH_HARQ4_7897_gNB_FAPI_s1.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnrPUSCH_HARQ4_7897_gNB_FAPI_s2.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnrPUSCH_HARQ4_7897_gNB_FAPI_s3.h5'}; 

    case 166 %%40 slot pattern: 64TR SRS (PRG size = 16) + BFW (PRG size = 16) + PDSCH
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_9308_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        %Subframe 0
        LP.SCHED{4}.config{1}.channels = {'TVnr_8046_gNB_FAPI_s0.h5'};
        %Subframe 2
        LP.SCHED{21}.config{1}.channels = {'TVnr_9308_gNB_FAPI_s0.h5', 'TVnr_3869_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9308_gNB_FAPI_s0.h5', 'TVnr_3869_gNB_FAPI_s0.h5'};
        LP.SCHED{23}.config{1}.channels = {                            'TVnr_3869_gNB_FAPI_s0.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9308_gNB_FAPI_s0.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9308_gNB_FAPI_s0.h5', 'TVnr_3869_gNB_FAPI_s0.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9308_gNB_FAPI_s0.h5', 'TVnr_3869_gNB_FAPI_s0.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9308_gNB_FAPI_s0.h5', 'TVnr_3869_gNB_FAPI_s0.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9308_gNB_FAPI_s0.h5', 'TVnr_3869_gNB_FAPI_s0.h5'};
        %Subframe 3
        LP.SCHED{31}.config{1}.channels = {'TVnr_9308_gNB_FAPI_s0.h5', 'TVnr_3869_gNB_FAPI_s0.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9308_gNB_FAPI_s0.h5', 'TVnr_3869_gNB_FAPI_s0.h5'};
        LP.SCHED{33}.config{1}.channels = {                            'TVnr_3869_gNB_FAPI_s0.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9308_gNB_FAPI_s0.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9308_gNB_FAPI_s0.h5', 'TVnr_3869_gNB_FAPI_s0.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9308_gNB_FAPI_s0.h5', 'TVnr_3869_gNB_FAPI_s0.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9308_gNB_FAPI_s0.h5', 'TVnr_3869_gNB_FAPI_s0.h5'};
        LP.SCHED{40}.config{1}.channels = {                            'TVnr_3869_gNB_FAPI_s0.h5'};

    case 167 % PUSCH-only; the mix of 64TR and 4TR with pusch_enable_perprgchest enabled
        nCell = 3;
        LP = [];
        LP.Cell_Configs = {'TVnr_7889_gNB_FAPI_s0.h5', 'TVnr_7889_gNB_FAPI_s0.h5', 'TVnr_7889_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_7869_gNB_FAPI_s0.h5'}; % 64TR   wideband BF 
        LP.SCHED{1}.config{2}.channels = {'TVnr_7204_gNB_FAPI_s0.h5'}; % 4TR          no BF
        LP.SCHED{1}.config{3}.channels = {'TVnr_7889_gNB_FAPI_s0.h5'}; % 64TR narrowband BF per-PRG CHEST

    case 190
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'TVnr_0190_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0190_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5'};




% 20 slot PDSCH template %%64TR BFW + PDSCH on all slots (DDDSUUDDDD)
%         %Slot #     % 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
%         %Slot type  % D   D   D   S   U   U   D   D   D   D   D   D   D   S   U   U   D   D   D   D
%         %BFW type   % B   B   -   -   -   B   B   B   B   B   B   B   -   -   -   B   B   B   B   B
%         %BFW map    %   \   \       \   \   \   \   \   \   \   \   \       \   \   \   \   \   \   \
%         %PXSCH type % D   D   D   -   -   -   D   D   D   D   D   D   D   -   -   -   D   D   D   D
%         %
%         nCell = 1;
%         LP = [];
%         nSlot = 20;
%         LP.Cell_Configs = {'TVnr_3870_gNB_FAPI_s0.h5',};
%         LP = init_launchPattern(LP, nSlot, nCell);
%         % Subframe 0
%         LP.SCHED{1}.config{1}.channels = {'TVnr_9236_gNB_FAPI_s0.h5', 'TVnr_3870_gNB_FAPI_s0.h5'};
%         LP.SCHED{2}.config{1}.channels = {'TVnr_9236_gNB_FAPI_s0.h5', 'TVnr_3870_gNB_FAPI_s0.h5'};
%         LP.SCHED{3}.config{1}.channels = {'TVnr_3870_gNB_FAPI_s0.h5'};
%         LP.SCHED{6}.config{1}.channels = {'TVnr_9236_gNB_FAPI_s0.h5',};
%         LP.SCHED{7}.config{1}.channels = {'TVnr_9236_gNB_FAPI_s0.h5', 'TVnr_3870_gNB_FAPI_s0.h5'};
%         LP.SCHED{8}.config{1}.channels = {'TVnr_9236_gNB_FAPI_s0.h5', 'TVnr_3870_gNB_FAPI_s0.h5'};
%         LP.SCHED{9}.config{1}.channels = {'TVnr_9236_gNB_FAPI_s0.h5', 'TVnr_3870_gNB_FAPI_s0.h5'};
%         LP.SCHED{10}.config{1}.channels = {'TVnr_9236_gNB_FAPI_s0.h5', 'TVnr_3870_gNB_FAPI_s0.h5'};
%         %Subframe 1
%         LP.SCHED{11}.config{1}.channels = {'TVnr_9236_gNB_FAPI_s0.h5', 'TVnr_3870_gNB_FAPI_s0.h5'};
%         LP.SCHED{12}.config{1}.channels = {'TVnr_9236_gNB_FAPI_s0.h5', 'TVnr_3870_gNB_FAPI_s0.h5'};
%         LP.SCHED{13}.config{1}.channels = {'TVnr_3870_gNB_FAPI_s0.h5'};
%         LP.SCHED{16}.config{1}.channels = {'TVnr_9236_gNB_FAPI_s0.h5'};
%         LP.SCHED{17}.config{1}.channels = {'TVnr_9236_gNB_FAPI_s0.h5', 'TVnr_3870_gNB_FAPI_s0.h5'};
%         LP.SCHED{18}.config{1}.channels = {'TVnr_9236_gNB_FAPI_s0.h5', 'TVnr_3870_gNB_FAPI_s0.h5'};
%         LP.SCHED{19}.config{1}.channels = {'TVnr_9236_gNB_FAPI_s0.h5', 'TVnr_3870_gNB_FAPI_s0.h5'};
%         LP.SCHED{20}.config{1}.channels = {'TVnr_9236_gNB_FAPI_s0.h5', 'TVnr_3870_gNB_FAPI_s0.h5'};
%         


        
%     20 slot PUSCH template %%64TR BFW + PUSCH on all slots (DDDSUUDDDD)
%         %Slot #     % 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
%         %Slot type  % D   D   D   S   U   U   D   D   D   D   D   D   D   S   U   U   D   D   D   D
%         %BFW type   % -   -   -   UB  UB  -   -   -   -   -   -   -   -   UB  UB  -   -   -   -   -
%         %BFW map    %   \   \       \   \   \   \   \   \   \   \   \       \   \   \   \   \   \   \
%         %PXSCH type % -   -   -   -   U   U   -   -   -   -   -   -   -   -   U   U   -   -   -   -
%         %
%         nCell = 1;
%         LP = [];
%         nSlot = 20;
%         LP.Cell_Configs = {'TVnr_79001_gNB_FAPI_s0.h5'};
%         LP = init_launchPattern(LP, nSlot, nCell);
%         % Subframe 0        
%         LP.SCHED{4}.config{1}.channels = {'TVnr_9001_gNB_FAPI_s0.h5'};
%         LP.SCHED{5}.config{1}.channels = {'TVnr_9001_gNB_FAPI_s0.h5', 'TVnr_79001_gNB_FAPI_s0.h5'};
%         LP.SCHED{6}.config{1}.channels = {'TVnr_79001_gNB_FAPI_s0.h5'};        
%         %Subframe 1
%         LP.SCHED{14}.config{1}.channels = {'TVnr_9001_gNB_FAPI_s0.h5'};
%         LP.SCHED{15}.config{1}.channels = {'TVnr_9001_gNB_FAPI_s0.h5', 'TVnr_79001_gNB_FAPI_s0.h5'};
%         LP.SCHED{16}.config{1}.channels = {'TVnr_79001_gNB_FAPI_s0.h5'};
% 
%     40 slot template  SRS + BFW + PDSCH/PUSCH on all slots (DDDSUUDDDD)
%         % SRS TVs 8412 & 8413 to match BFW 9000 & 9001
%         nCell = 1;
%         LP = [];
%         nSlot = 40;
%         LP.Cell_Configs = {'TVnr_9226_gNB_FAPI_s0.h5'};
%         LP = init_launchPattern(LP, nSlot, nCell);
%         %Subframe 0
%         LP.SCHED{4}.config{1}.channels = {'TVnr_8412_gNB_FAPI_s0.h5'};
%         %Subframe 1
%         LP.SCHED{14}.config{1}.channels = {'TVnr_8418_gNB_FAPI_s0.h5'};
%         %Subframe 2
%         LP.SCHED{21}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_39000_gNB_FAPI_s0.h5'};
%         LP.SCHED{22}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_39000_gNB_FAPI_s0.h5'};
%         LP.SCHED{23}.config{1}.channels = {'TVnr_39000_gNB_FAPI_s0.h5'};
%         LP.SCHED{24}.config{1}.channels = {'TVnr_9227_gNB_FAPI_s0.h5'};
%         LP.SCHED{25}.config{1}.channels = {'TVnr_9227_gNB_FAPI_s0.h5', 'TVnr_79001_gNB_FAPI_s0.h5'};
%         LP.SCHED{26}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_79001_gNB_FAPI_s0.h5'};
%         LP.SCHED{27}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_39000_gNB_FAPI_s0.h5'};
%         LP.SCHED{28}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_39000_gNB_FAPI_s0.h5'};
%         LP.SCHED{29}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_39000_gNB_FAPI_s0.h5'};
%         LP.SCHED{30}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_39000_gNB_FAPI_s0.h5'};
%         %Subframe 3
%         LP.SCHED{31}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_39000_gNB_FAPI_s0.h5'};
%         LP.SCHED{32}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_39000_gNB_FAPI_s0.h5'};
%         LP.SCHED{33}.config{1}.channels = {'TVnr_39000_gNB_FAPI_s0.h5'};
%         LP.SCHED{34}.config{1}.channels = {'TVnr_9227_gNB_FAPI_s0.h5'};
%         LP.SCHED{35}.config{1}.channels = {'TVnr_9227_gNB_FAPI_s0.h5', 'TVnr_79001_gNB_FAPI_s0.h5'};
%         LP.SCHED{36}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_79001_gNB_FAPI_s0.h5'};
%         LP.SCHED{37}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_39000_gNB_FAPI_s0.h5'};
%         LP.SCHED{38}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_39000_gNB_FAPI_s0.h5'};
%         LP.SCHED{39}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_39000_gNB_FAPI_s0.h5'};
%         LP.SCHED{40}.config{1}.channels = {'TVnr_9226_gNB_FAPI_s0.h5', 'TVnr_39000_gNB_FAPI_s0.h5'};

    case 200 % Mixed peak and average 2-cell pattern
        params.nCell = 2;
        params.neg_slot = [];
        params.neg_cell = [];
        LP = buildMixedCellPattern(params)
    case 201 % like 200, but create CRC failures on slot 14 of cell 1
        params.nCell = 2;
        params.neg_slot = [14];
        params.neg_cell = [1];
        LP = buildMixedCellPattern(params)

    % 903XX is for P5G patterns
    case 300 % P5G_PRACH
        nCell = 1;
        nSlot = 40;
        LP = [];
        LP.Cell_Configs = {'demo_msg3_gNB_FAPI_s5.h5'};
        LP.UL_Cell_Configs = {'demo_msg1_gNB_FAPI_s15.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'demo_ssb_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'demo_coreset0_gNB_FAPI_s2.h5'};
        LP.SCHED{16}.config{1}.channels = {'demo_msg1_gNB_FAPI_s15.h5'};
        LP.SCHED{22}.config{1}.channels = {'demo_msg2_gNB_FAPI_s1.h5'};
        LP.SCHED{26}.config{1}.channels = {'demo_msg3_gNB_FAPI_s5.h5'};
        LP.SCHED{32}.config{1}.channels = {'demo_msg4_gNB_FAPI_s11.h5'};

    case 301 % P5G_SIB1
        nCell = 1;
        nSlot = 40;
        LP = [];
        LP.Cell_Configs = {'demo_coreset0_gNB_FAPI_s2.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'demo_ssb_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'demo_coreset0_gNB_FAPI_s2.h5'};
        
    case 302 % P5G_SIB1_FXN
        nCell = 1;
        nSlot = 40;
        LP = [];
        LP.Cell_Configs = {'demo_msg3_fxn_gNB_FAPI_s15.h5'};
        LP.UL_Cell_Configs = {'demo_msg1_gNB_FAPI_s15.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'demo_ssb_fxn_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'demo_sib1_fxn_gNB_FAPI_s2.h5'};
        LP.SCHED{16}.config{1}.channels = {'demo_msg1_gNB_FAPI_s15.h5'};
        LP.SCHED{22}.config{1}.channels = {'demo_msg2_fxn_gNB_FAPI_s1.h5'};
        LP.SCHED{36}.config{1}.channels = {'demo_msg3_fxn_gNB_FAPI_s15.h5', 'demo_msg1_gNB_FAPI_s15.h5'};

    case 501 % bug 3901585 PDSCH + TRS
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'bug3901585_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'bug3901585_gNB_FAPI_s0.h5'};
    case 502 % bug 3862516 TestModel 2 PDSCH + 1 PDCCH
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'bug3862516_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'bug3862516_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'bug3862516_gNB_FAPI_s0.h5'};
        LP.SCHED{3}.config{1}.channels = {'bug3862516_gNB_FAPI_s0.h5'};
        LP.SCHED{4}.config{1}.channels = {'bug3862516_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'bug3862516_gNB_FAPI_s0.h5'};
        LP.SCHED{8}.config{1}.channels = {'bug3862516_gNB_FAPI_s0.h5'};
        LP.SCHED{9}.config{1}.channels = {'bug3862516_gNB_FAPI_s0.h5'};
        LP.SCHED{10}.config{1}.channels = {'bug3862516_gNB_FAPI_s0.h5'};
        LP.SCHED{11}.config{1}.channels = {'bug3862516_gNB_FAPI_s0.h5'};
        LP.SCHED{12}.config{1}.channels = {'bug3862516_gNB_FAPI_s0.h5'};
        LP.SCHED{13}.config{1}.channels = {'bug3862516_gNB_FAPI_s0.h5'};
        LP.SCHED{14}.config{1}.channels = {'bug3862516_gNB_FAPI_s0.h5'};
        LP.SCHED{17}.config{1}.channels = {'bug3862516_gNB_FAPI_s0.h5'};
        LP.SCHED{18}.config{1}.channels = {'bug3862516_gNB_FAPI_s0.h5'};
        LP.SCHED{19}.config{1}.channels = {'bug3862516_gNB_FAPI_s0.h5'};
        LP.SCHED{20}.config{1}.channels = {'bug3862516_gNB_FAPI_s0.h5'};
    case 503 % bug 4011756 PDSCH + ZP-CSIRS + NZP-CSIRS
        nCell = 3;
        LP = [];
        LP.Cell_Configs = {'bug4011756_cell0_gNB_FAPI_s0.h5', ...
            'bug4011756_cell2_gNB_FAPI_s0.h5', 'bug4011756_cell3_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'bug4011756_cell0_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{2}.channels = {'bug4011756_cell2_gNB_FAPI_s0.h5'};
        LP.SCHED{1}.config{3}.channels = {'bug4011756_cell3_gNB_FAPI_s0.h5'};
    case 504 % bug 3951344 PM = [1 0 1 0]
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'bug3951344_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'bug3951344_gNB_FAPI_s0.h5'};
    case 505 % bug 4185251 CSIRS + multiple PDSCH (PRB locations in descending order)
        nCell = 1;
        LP = [];
        LP.Cell_Configs = {'bug4185251_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'bug4185251_gNB_FAPI_s0.h5'};
    case 506 % negative test cases using negTV.enable = 1
        nCell = 2;
        nSlot = 40;
        LP = [];
        LP.Cell_Configs = {'TVnr_0609_gNB_FAPI_s24.h5','TVnr_0502_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{2}.channels = {'TVnr_0502_gNB_FAPI_s0.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_0500_gNB_FAPI_s6.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_0501_gNB_FAPI_s7.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_0609_gNB_FAPI_s24.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_0611_gNB_FAPI_s34.h5'};
    case 507 % bug 5098017
        nCell = 1;
        nSlot = 40;
        LP = [];
        LP.Cell_Configs = {'bug5098017_PUSCH_sfn944_15_gNB_FAPI_s15.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{16}.config{1}.channels = {'bug5098017_PUSCH_sfn943_15_gNB_FAPI_s15.h5'};
        LP.SCHED{36}.config{1}.channels = {'bug5098017_PUSCH_sfn944_15_gNB_FAPI_s15.h5'};
    case 601 % multi-channel DL (HARQ disabled)
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_0150_gNB_FAPI_s0.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'TVnr_0150_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'TVnr_0151_gNB_FAPI_s1.h5'};
        LP.SCHED{3}.config{1}.channels = {'TVnr_0152_gNB_FAPI_s2.h5'};
        LP.SCHED{4}.config{1}.channels = {'TVnr_0153_gNB_FAPI_s3.h5'};
%         LP.SCHED{5}.config{1}.channels = {'TVnr_0154_gNB_FAPI_s4.h5'};
%         LP.SCHED{6}.config{1}.channels = {'TVnr_0155_gNB_FAPI_s5.h5'};
        LP.SCHED{7}.config{1}.channels = {'TVnr_0156_gNB_FAPI_s6.h5'};
        LP.SCHED{8}.config{1}.channels = {'TVnr_0157_gNB_FAPI_s7.h5'};
        LP.SCHED{9}.config{1}.channels = {'TVnr_0158_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_0159_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_0160_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_0161_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_0162_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_0163_gNB_FAPI_s13.h5'};
%         LP.SCHED{15}.config{1}.channels = {'TVnr_0164_gNB_FAPI_s14.h5'};
%         LP.SCHED{16}.config{1}.channels = {'TVnr_0165_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_0166_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_0167_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_0168_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_0169_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_0170_gNB_FAPI_s20.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_0171_gNB_FAPI_s21.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_0172_gNB_FAPI_s22.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_0173_gNB_FAPI_s23.h5'};
%         LP.SCHED{25}.config{1}.channels = {'TVnr_0174_gNB_FAPI_s24.h5'};
%         LP.SCHED{26}.config{1}.channels = {'TVnr_0175_gNB_FAPI_s25.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_0176_gNB_FAPI_s26.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_0177_gNB_FAPI_s27.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_0178_gNB_FAPI_s28.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_0179_gNB_FAPI_s29.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_0180_gNB_FAPI_s30.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_0181_gNB_FAPI_s31.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_0182_gNB_FAPI_s32.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_0183_gNB_FAPI_s33.h5'};
%         LP.SCHED{35}.config{1}.channels = {'TVnr_0184_gNB_FAPI_s34.h5'};
%         LP.SCHED{36}.config{1}.channels = {'TVnr_0185_gNB_FAPI_s35.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_0186_gNB_FAPI_s36.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_0187_gNB_FAPI_s37.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_0188_gNB_FAPI_s38.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_0189_gNB_FAPI_s39.h5'};
    case 602 % multi-channel UL (HARQ disabled)
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_0150_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0555_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
%         LP.SCHED{1}.config{1}.channels = {'TVnr_0150_gNB_FAPI_s0.h5'};
%         LP.SCHED{2}.config{1}.channels = {'TVnr_0151_gNB_FAPI_s1.h5'};
%         LP.SCHED{3}.config{1}.channels = {'TVnr_0152_gNB_FAPI_s2.h5'};
%         LP.SCHED{4}.config{1}.channels = {'TVnr_0553_gNB_FAPI_s3.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_0554_gNB_FAPI_s4.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_0555_gNB_FAPI_s5.h5'};
%         LP.SCHED{7}.config{1}.channels = {'TVnr_0156_gNB_FAPI_s6.h5'};
%         LP.SCHED{8}.config{1}.channels = {'TVnr_0157_gNB_FAPI_s7.h5'};
%         LP.SCHED{9}.config{1}.channels = {'TVnr_0158_gNB_FAPI_s8.h5'};
%         LP.SCHED{10}.config{1}.channels = {'TVnr_0159_gNB_FAPI_s9.h5'};
%         LP.SCHED{11}.config{1}.channels = {'TVnr_0160_gNB_FAPI_s10.h5'};
%         LP.SCHED{12}.config{1}.channels = {'TVnr_0161_gNB_FAPI_s11.h5'};
%         LP.SCHED{13}.config{1}.channels = {'TVnr_0162_gNB_FAPI_s12.h5'};
%         LP.SCHED{14}.config{1}.channels = {'TVnr_0563_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_0564_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_0565_gNB_FAPI_s15.h5'};
%         LP.SCHED{17}.config{1}.channels = {'TVnr_0166_gNB_FAPI_s16.h5'};
%         LP.SCHED{18}.config{1}.channels = {'TVnr_0167_gNB_FAPI_s17.h5'};
%         LP.SCHED{19}.config{1}.channels = {'TVnr_0168_gNB_FAPI_s18.h5'};
%         LP.SCHED{20}.config{1}.channels = {'TVnr_0169_gNB_FAPI_s19.h5'};
%         LP.SCHED{21}.config{1}.channels = {'TVnr_0170_gNB_FAPI_s20.h5'};
%         LP.SCHED{22}.config{1}.channels = {'TVnr_0171_gNB_FAPI_s21.h5'};
%         LP.SCHED{23}.config{1}.channels = {'TVnr_0172_gNB_FAPI_s22.h5'};
%         LP.SCHED{24}.config{1}.channels = {'TVnr_0573_gNB_FAPI_s23.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_0574_gNB_FAPI_s24.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_0575_gNB_FAPI_s25.h5'};
%         LP.SCHED{27}.config{1}.channels = {'TVnr_0176_gNB_FAPI_s26.h5'};
%         LP.SCHED{28}.config{1}.channels = {'TVnr_0177_gNB_FAPI_s27.h5'};
%         LP.SCHED{29}.config{1}.channels = {'TVnr_0178_gNB_FAPI_s28.h5'};
%         LP.SCHED{30}.config{1}.channels = {'TVnr_0179_gNB_FAPI_s29.h5'};
%         LP.SCHED{31}.config{1}.channels = {'TVnr_0180_gNB_FAPI_s30.h5'};
%         LP.SCHED{32}.config{1}.channels = {'TVnr_0181_gNB_FAPI_s31.h5'};
%         LP.SCHED{33}.config{1}.channels = {'TVnr_0182_gNB_FAPI_s32.h5'};
%         LP.SCHED{34}.config{1}.channels = {'TVnr_0583_gNB_FAPI_s33.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_0584_gNB_FAPI_s34.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_0585_gNB_FAPI_s35.h5'};
%         LP.SCHED{37}.config{1}.channels = {'TVnr_0186_gNB_FAPI_s36.h5'};
%         LP.SCHED{38}.config{1}.channels = {'TVnr_0187_gNB_FAPI_s37.h5'};
%         LP.SCHED{39}.config{1}.channels = {'TVnr_0188_gNB_FAPI_s38.h5'};
%         LP.SCHED{40}.config{1}.channels = {'TVnr_0189_gNB_FAPI_s39.h5'};        
    case 603 % multi-channel DL + UL (HARQ enabled)
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'HARQ_MIX_DL_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'HARQ_MIX_UL_gNB_FAPI_s4.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s0.h5'};
        LP.SCHED{2}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s1.h5'};
        LP.SCHED{3}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s2.h5'};
        LP.SCHED{4}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s3.h5', 'HARQ_MIX_UL_gNB_FAPI_s3.h5'};
        LP.SCHED{5}.config{1}.channels = {'HARQ_MIX_UL_gNB_FAPI_s4.h5'};
        LP.SCHED{6}.config{1}.channels = {'HARQ_MIX_UL_gNB_FAPI_s5.h5'};
        LP.SCHED{7}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s6.h5'};
        LP.SCHED{8}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s7.h5'};
        LP.SCHED{9}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s13.h5', 'HARQ_MIX_UL_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'HARQ_MIX_UL_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{1}.channels = {'HARQ_MIX_UL_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s20.h5'};
        LP.SCHED{22}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s21.h5'};
        LP.SCHED{23}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s22.h5'};
        LP.SCHED{24}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s23.h5', 'HARQ_MIX_UL_gNB_FAPI_s23.h5'};
        LP.SCHED{25}.config{1}.channels = {'HARQ_MIX_UL_gNB_FAPI_s24.h5'};
        LP.SCHED{26}.config{1}.channels = {'HARQ_MIX_UL_gNB_FAPI_s25.h5'};
        LP.SCHED{27}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s26.h5'};
        LP.SCHED{28}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s27.h5'};
        LP.SCHED{29}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s28.h5'};
        LP.SCHED{30}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s29.h5'};
        LP.SCHED{31}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s30.h5'};
        LP.SCHED{32}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s31.h5'};
        LP.SCHED{33}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s32.h5'};
        LP.SCHED{34}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s33.h5', 'HARQ_MIX_UL_gNB_FAPI_s33.h5'};
        LP.SCHED{35}.config{1}.channels = {'HARQ_MIX_UL_gNB_FAPI_s34.h5'};
        LP.SCHED{36}.config{1}.channels = {'HARQ_MIX_UL_gNB_FAPI_s35.h5'};
        LP.SCHED{37}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s36.h5'};
        LP.SCHED{38}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s37.h5'};
        LP.SCHED{39}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s38.h5'};
        LP.SCHED{40}.config{1}.channels = {'HARQ_MIX_DL_gNB_FAPI_s39.h5'}; 

    % case 604: same with launch_pattern_F08_8C_57.yaml, skip
    
    % remove 90605 since we already have 90701
    % case 605 % multi-cell: PUCCH+SRS in one cell and only PUCCH for second cell (GT-8777), modified from 90056 heterogeneous PUCCH
    %         % PUCCH and SRS are on part of the following
    %         % S-slot (Slot 3/13) -> Symbols 10-13
    %         % UL slots (4, 5, 14, 15)
    %         nCell = 2;
    %         nSlot = 20;
    %         LP = [];
    %         LP.Cell_Configs = {'TVnr_ULMIX_3925_gNB_FAPI_s3.h5', 'TVnr_6001_gNB_FAPI_s0.h5'};
    %         LP = init_launchPattern(LP, nSlot, nCell);
    %         LP.SCHED{4}.config{1}.channels = {'TVnr_ULMIX_3925_gNB_FAPI_s3.h5'};
    %         LP.SCHED{4}.config{2}.channels = {'TVnr_6001_gNB_FAPI_s0.h5'};
    %         LP.SCHED{5}.config{1}.channels = {'TVnr_ULMIX_3925_gNB_FAPI_s3.h5'};
    %         LP.SCHED{6}.config{1}.channels = {'TVnr_ULMIX_3925_gNB_FAPI_s3.h5'};
    %         LP.SCHED{6}.config{2}.channels = {'TVnr_6002_gNB_FAPI_s0.h5'};
    %         LP.SCHED{14}.config{1}.channels = {'TVnr_ULMIX_3925_gNB_FAPI_s3.h5'};
    %         LP.SCHED{14}.config{2}.channels = {'TVnr_6204_gNB_FAPI_s0.h5'};
    %         LP.SCHED{15}.config{2}.channels = {'TVnr_6205_gNB_FAPI_s0.h5'};
    %         LP.SCHED{16}.config{1}.channels = {'TVnr_8301_gNB_FAPI_s0.h5'};
    %         LP.SCHED{16}.config{2}.channels = {'TVnr_ULMIX_3925_gNB_FAPI_s3.h5'};

    case 606 % % 64TR static + dynamic beamforming SU-MIMO 1-UEG
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_0300_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0705_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 1}.config{1}.channels = {'TVnr_0300_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{1}.channels = {'TVnr_0301_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{1}.channels = {'TVnr_0302_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_0703_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{1}.channels = {'TVnr_0704_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{1}.channels = {'TVnr_0705_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{1}.channels = {'TVnr_0306_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{1}.channels = {'TVnr_0307_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{1}.channels = {'TVnr_0308_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_0309_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_0310_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_0311_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_0312_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_0713_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_0714_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_0715_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_0316_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_0317_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_0318_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_0319_gNB_FAPI_s19.h5'};
        
    case 607 % % 64TR static + dynamic beamforming SU-MIMO 2-UEG
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_0320_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0725_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 1}.config{1}.channels = {'TVnr_0320_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{1}.channels = {'TVnr_0321_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{1}.channels = {'TVnr_0322_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_0723_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{1}.channels = {'TVnr_0724_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{1}.channels = {'TVnr_0725_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{1}.channels = {'TVnr_0326_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{1}.channels = {'TVnr_0327_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{1}.channels = {'TVnr_0328_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_0329_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_0330_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_0331_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_0332_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_0733_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_0734_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_0735_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_0336_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_0337_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_0338_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_0339_gNB_FAPI_s19.h5'};
    
    case 608 % % 64TR static + dynamic beamforming SU/MU-MIMO 3-UEG
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_0340_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0745_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_0743_gNB_FAPI_s3.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0340_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0341_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_0342_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9279_gNB_FAPI_s0.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9279_gNB_FAPI_s0.h5','TVnr_0744_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0745_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0346_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0347_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0348_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0349_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0350_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0351_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_0352_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9279_gNB_FAPI_s0.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9279_gNB_FAPI_s0.h5','TVnr_0754_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0755_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0356_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0357_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0358_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_0359_gNB_FAPI_s19.h5'};

    case 609 % % 64TR static + dynamic beamforming SU/MU-MIMO 1-UEG
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_0360_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0765_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_0763_gNB_FAPI_s3.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9284_gNB_FAPI_s0.h5','TVnr_0360_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9284_gNB_FAPI_s0.h5','TVnr_0361_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {                           'TVnr_0362_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9283_gNB_FAPI_s0.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9283_gNB_FAPI_s0.h5','TVnr_0764_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9284_gNB_FAPI_s0.h5','TVnr_0765_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9284_gNB_FAPI_s0.h5','TVnr_0366_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9284_gNB_FAPI_s0.h5','TVnr_0367_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9284_gNB_FAPI_s0.h5','TVnr_0368_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9284_gNB_FAPI_s0.h5','TVnr_0369_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9284_gNB_FAPI_s0.h5','TVnr_0370_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9284_gNB_FAPI_s0.h5','TVnr_0371_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {                           'TVnr_0372_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9283_gNB_FAPI_s0.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9283_gNB_FAPI_s0.h5','TVnr_0774_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9284_gNB_FAPI_s0.h5','TVnr_0775_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9284_gNB_FAPI_s0.h5','TVnr_0376_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9284_gNB_FAPI_s0.h5','TVnr_0377_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9284_gNB_FAPI_s0.h5','TVnr_0378_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_0379_gNB_FAPI_s19.h5'};

    case 610 % % 64TR static + dynamic beamforming SU/MU-MIMO 1-UEG
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_0380_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0785_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_0783_gNB_FAPI_s3.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9286_gNB_FAPI_s0.h5','TVnr_0380_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9286_gNB_FAPI_s0.h5','TVnr_0381_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {                           'TVnr_0382_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9285_gNB_FAPI_s0.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9285_gNB_FAPI_s0.h5','TVnr_0784_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9286_gNB_FAPI_s0.h5','TVnr_0785_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9286_gNB_FAPI_s0.h5','TVnr_0386_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9286_gNB_FAPI_s0.h5','TVnr_0387_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9286_gNB_FAPI_s0.h5','TVnr_0388_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9286_gNB_FAPI_s0.h5','TVnr_0389_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9286_gNB_FAPI_s0.h5','TVnr_0390_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9286_gNB_FAPI_s0.h5','TVnr_0391_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {                           'TVnr_0392_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9285_gNB_FAPI_s0.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9285_gNB_FAPI_s0.h5','TVnr_0794_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9286_gNB_FAPI_s0.h5','TVnr_0795_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9286_gNB_FAPI_s0.h5','TVnr_0396_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9286_gNB_FAPI_s0.h5','TVnr_0397_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9286_gNB_FAPI_s0.h5','TVnr_0398_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_0399_gNB_FAPI_s19.h5'};

    case 611 % % 64TR dynamic beamforming MU-MIMO 1-UEG
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_0400_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0805_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_0803_gNB_FAPI_s3.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9288_gNB_FAPI_s0.h5','TVnr_0400_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9288_gNB_FAPI_s0.h5','TVnr_0401_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {                           'TVnr_0402_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9287_gNB_FAPI_s0.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9287_gNB_FAPI_s0.h5','TVnr_0804_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9288_gNB_FAPI_s0.h5','TVnr_0805_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9288_gNB_FAPI_s0.h5','TVnr_0406_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9288_gNB_FAPI_s0.h5','TVnr_0407_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9288_gNB_FAPI_s0.h5','TVnr_0408_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9288_gNB_FAPI_s0.h5','TVnr_0409_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9288_gNB_FAPI_s0.h5','TVnr_0410_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9288_gNB_FAPI_s0.h5','TVnr_0411_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {                           'TVnr_0412_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9287_gNB_FAPI_s0.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9287_gNB_FAPI_s0.h5','TVnr_0814_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9288_gNB_FAPI_s0.h5','TVnr_0815_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9288_gNB_FAPI_s0.h5','TVnr_0416_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9288_gNB_FAPI_s0.h5','TVnr_0417_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9288_gNB_FAPI_s0.h5','TVnr_0418_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_0419_gNB_FAPI_s19.h5'};

    case 612 % % 64TR test case
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_0420_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0825_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_0823_gNB_FAPI_s3.h5'};
        LP.SCHED{21}.config{1}.channels = {                           'TVnr_0420_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0421_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_0422_gNB_FAPI_s2.h5'};
       %LP.SCHED{24}.config{1}.channels = {'TVnr_0423_gNB_FAPI_s3.h5'};
        LP.SCHED{24}.config{1}.channels = {};
        LP.SCHED{25}.config{1}.channels = {                           'TVnr_0824_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0825_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0426_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0427_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0428_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0429_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0430_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0431_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_0432_gNB_FAPI_s12.h5'};
       %LP.SCHED{34}.config{1}.channels = {'TVnr_0433_gNB_FAPI_s13.h5','TVnr_0833_gNB_FAPI_s13.h5'};
        LP.SCHED{34}.config{1}.channels = {};
        LP.SCHED{35}.config{1}.channels = {                           'TVnr_0834_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0835_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0436_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0437_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0438_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_0439_gNB_FAPI_s19.h5'};
    case 613
        params.data_TV = 'TVnr_0297_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9248_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 614
        params.data_TV = 'TVnr_0298_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9248_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 615
        params.data_TV = 'TVnr_0299_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9248_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 616
        params.data_TV = 'TVnr_0296_gNB_FAPI_s0.h5';
        params.bfw_TV  = 'TVnr_9296_gNB_FAPI_s0.h5';
        LP = populateLPhelper2Slot(params);
    case 617 % 64TR test case, BFPforCuphy = 9, modified from 612
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_0440_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0845_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_0843_gNB_FAPI_s3.h5'};
        LP.SCHED{21}.config{1}.channels = {                           'TVnr_0440_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9342_gNB_FAPI_s0.h5','TVnr_0441_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_0442_gNB_FAPI_s2.h5'};
       %LP.SCHED{24}.config{1}.channels = {'TVnr_0443_gNB_FAPI_s3.h5'};
        LP.SCHED{24}.config{1}.channels = {};
        LP.SCHED{25}.config{1}.channels = {                           'TVnr_0844_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9342_gNB_FAPI_s0.h5','TVnr_0845_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9342_gNB_FAPI_s0.h5','TVnr_0446_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9342_gNB_FAPI_s0.h5','TVnr_0447_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9342_gNB_FAPI_s0.h5','TVnr_0448_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9342_gNB_FAPI_s0.h5','TVnr_0449_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9342_gNB_FAPI_s0.h5','TVnr_0450_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9342_gNB_FAPI_s0.h5','TVnr_0451_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_0452_gNB_FAPI_s12.h5'};
       %LP.SCHED{34}.config{1}.channels = {'TVnr_0453_gNB_FAPI_s13.h5','TVnr_0853_gNB_FAPI_s13.h5'};
        LP.SCHED{34}.config{1}.channels = {};
        LP.SCHED{35}.config{1}.channels = {                           'TVnr_0854_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9342_gNB_FAPI_s0.h5','TVnr_0855_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9342_gNB_FAPI_s0.h5','TVnr_0456_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9342_gNB_FAPI_s0.h5','TVnr_0457_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9342_gNB_FAPI_s0.h5','TVnr_0458_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_0459_gNB_FAPI_s19.h5'};
    case 618 % 64TR test case, CSIRS 8 ports (row 6), modified from 612
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_0460_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0825_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_0823_gNB_FAPI_s3.h5'};
        LP.SCHED{21}.config{1}.channels = {                           'TVnr_0460_gNB_FAPI_s0.h5'};  % has CSI-RS, diff TV from 612
        LP.SCHED{22}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0461_gNB_FAPI_s1.h5'};  % has CSI-RS, diff TV from 612
        LP.SCHED{23}.config{1}.channels = {'TVnr_0422_gNB_FAPI_s2.h5'};
       %LP.SCHED{24}.config{1}.channels = {'TVnr_0423_gNB_FAPI_s3.h5'};
        LP.SCHED{24}.config{1}.channels = {};
        LP.SCHED{25}.config{1}.channels = {                           'TVnr_0824_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0825_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0426_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0427_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0428_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0429_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0430_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0431_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_0432_gNB_FAPI_s12.h5'};
       %LP.SCHED{34}.config{1}.channels = {'TVnr_0433_gNB_FAPI_s13.h5','TVnr_0833_gNB_FAPI_s13.h5'};
        LP.SCHED{34}.config{1}.channels = {};
        LP.SCHED{35}.config{1}.channels = {                           'TVnr_0834_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0835_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0436_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0437_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9290_gNB_FAPI_s0.h5','TVnr_0438_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_0439_gNB_FAPI_s19.h5'};
    case 619 % 64TR test case, 4 SRS symbols, modified from 612
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_0420_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0825_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_0863_gNB_FAPI_s3.h5'};
        LP.SCHED{21}.config{1}.channels = {                           'TVnr_0420_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9346_gNB_FAPI_s0.h5','TVnr_0421_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_0422_gNB_FAPI_s2.h5'};
       %LP.SCHED{24}.config{1}.channels = {'TVnr_0423_gNB_FAPI_s3.h5'};
        LP.SCHED{24}.config{1}.channels = {};
        LP.SCHED{25}.config{1}.channels = {                           'TVnr_0824_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9346_gNB_FAPI_s0.h5','TVnr_0825_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9346_gNB_FAPI_s0.h5','TVnr_0426_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9346_gNB_FAPI_s0.h5','TVnr_0427_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9346_gNB_FAPI_s0.h5','TVnr_0428_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9346_gNB_FAPI_s0.h5','TVnr_0429_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9346_gNB_FAPI_s0.h5','TVnr_0430_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9346_gNB_FAPI_s0.h5','TVnr_0431_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_0432_gNB_FAPI_s12.h5'};
       %LP.SCHED{34}.config{1}.channels = {'TVnr_0433_gNB_FAPI_s13.h5','TVnr_0833_gNB_FAPI_s13.h5'};
        LP.SCHED{34}.config{1}.channels = {};
        LP.SCHED{35}.config{1}.channels = {                           'TVnr_0834_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9346_gNB_FAPI_s0.h5','TVnr_0835_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9346_gNB_FAPI_s0.h5','TVnr_0436_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9346_gNB_FAPI_s0.h5','TVnr_0437_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9346_gNB_FAPI_s0.h5','TVnr_0438_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_0439_gNB_FAPI_s19.h5'};

    % 6** reserved for mMIMO test cases

    case 620  % 64TR static + dynamic beamforming SU/MU-MIMO 3-UEG, SRS on last two symbols of slot 4/14
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_DLMIX_20200_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21205_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_ULMIX_21203_gNB_FAPI_s3.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20200_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20201_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_DLMIX_20202_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9309_gNB_FAPI_s0.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9309_gNB_FAPI_s0.h5','TVnr_ULMIX_21404_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_ULMIX_21205_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20206_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20207_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20208_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20209_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20210_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20211_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_DLMIX_20212_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9309_gNB_FAPI_s0.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9309_gNB_FAPI_s0.h5','TVnr_ULMIX_21414_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_ULMIX_21215_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20216_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20217_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20218_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_DLMIX_20219_gNB_FAPI_s19.h5'};
    case 621  % 64TR static + dynamic beamforming SU/MU-MIMO 3-UEG, SRS on last two symbols of slot 5/15
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_DLMIX_20200_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21205_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_ULMIX_21203_gNB_FAPI_s3.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20200_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20201_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_DLMIX_20202_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9309_gNB_FAPI_s0.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9309_gNB_FAPI_s0.h5','TVnr_ULMIX_21204_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_ULMIX_21405_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20206_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20207_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20208_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20209_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20210_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20211_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_DLMIX_20212_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9309_gNB_FAPI_s0.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9309_gNB_FAPI_s0.h5','TVnr_ULMIX_21214_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_ULMIX_21415_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20216_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20217_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20218_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_DLMIX_20219_gNB_FAPI_s19.h5'};
    case 622  % 64TR static + dynamic beamforming SU/MU-MIMO 3-UEG, SRS on last two symbols of slot 4/5/14/15
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_DLMIX_20200_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21205_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_ULMIX_21203_gNB_FAPI_s3.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20200_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20201_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_DLMIX_20202_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9309_gNB_FAPI_s0.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9309_gNB_FAPI_s0.h5','TVnr_ULMIX_21404_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_ULMIX_21405_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20206_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20207_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20208_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20209_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20210_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20211_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_DLMIX_20212_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9309_gNB_FAPI_s0.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9309_gNB_FAPI_s0.h5','TVnr_ULMIX_21414_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_ULMIX_21415_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20216_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20217_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20218_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_DLMIX_20219_gNB_FAPI_s19.h5'};

    case 623 % 64TR test case, DL MU-MIMO in D/S slots, column B
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_DLMIX_20520_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21485_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 1}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20520_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20521_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20542_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_DLMIX_20543_gNB_FAPI_s3.h5','TVnr_0823_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{1}.channels = {'TVnr_ULMIX_21484_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_ULMIX_21485_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20546_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20547_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20548_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20549_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20550_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20551_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20552_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_DLMIX_20553_gNB_FAPI_s13.h5','TVnr_0833_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_ULMIX_21494_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_ULMIX_21495_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20556_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20557_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20558_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20559_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20540_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20541_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20542_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_DLMIX_20543_gNB_FAPI_s3.h5','TVnr_ULMIX_21503_gNB_FAPI_s3.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_ULMIX_21484_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_ULMIX_21485_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20546_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20547_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20548_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20549_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20550_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20551_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20552_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_DLMIX_20553_gNB_FAPI_s13.h5','TVnr_ULMIX_21513_gNB_FAPI_s13.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_ULMIX_21494_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_ULMIX_21495_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20556_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20557_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20558_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_9354_gNB_FAPI_s0.h5','TVnr_DLMIX_20559_gNB_FAPI_s19.h5'};

    case 624 % % 64TR test case 'column G'
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_DLMIX_20560_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21525_gNB_FAPI_s5.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 1}.config{1}.channels = {'TVnr_9364_gNB_FAPI_s0.h5','TVnr_DLMIX_20560_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{1}.channels = {'TVnr_9364_gNB_FAPI_s0.h5','TVnr_DLMIX_20561_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20562_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_9357_gNB_FAPI_s0.h5','TVnr_DLMIX_20563_gNB_FAPI_s3.h5','TVnr_ULMIX_21523_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{1}.channels = {'TVnr_9355_gNB_FAPI_s0.h5','TVnr_ULMIX_21524_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_ULMIX_21525_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20566_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20567_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20568_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20569_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20570_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20571_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20572_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_9357_gNB_FAPI_s0.h5','TVnr_DLMIX_20573_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_9355_gNB_FAPI_s0.h5','TVnr_ULMIX_21534_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_ULMIX_21535_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20576_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20577_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20578_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20579_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20580_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20581_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20582_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9357_gNB_FAPI_s0.h5','TVnr_DLMIX_20583_gNB_FAPI_s3.h5','TVnr_ULMIX_21533_gNB_FAPI_s3.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9355_gNB_FAPI_s0.h5','TVnr_ULMIX_21524_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_ULMIX_21525_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20586_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20587_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20588_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20589_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20590_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20591_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20592_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9357_gNB_FAPI_s0.h5','TVnr_DLMIX_20593_gNB_FAPI_s13.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9355_gNB_FAPI_s0.h5','TVnr_ULMIX_21534_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_ULMIX_21535_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20596_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20597_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20598_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_9364_gNB_FAPI_s0.h5','TVnr_DLMIX_20599_gNB_FAPI_s19.h5'};

    case 625 % 64TR test case, DL MU-MIMO in D/S slots, column B, w/ modcomp
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_DLMIX_20600_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21485_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 1}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20600_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20601_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20622_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_DLMIX_20623_gNB_FAPI_s3.h5','TVnr_0823_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{1}.channels = {'TVnr_ULMIX_21484_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_ULMIX_21485_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20626_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20627_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20628_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20629_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20630_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20631_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20632_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_DLMIX_20633_gNB_FAPI_s13.h5','TVnr_0833_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_ULMIX_21494_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_ULMIX_21495_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20636_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20637_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20638_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20639_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20620_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20621_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20622_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_DLMIX_20623_gNB_FAPI_s3.h5','TVnr_ULMIX_21503_gNB_FAPI_s3.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_ULMIX_21484_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_ULMIX_21485_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20626_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20627_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20628_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20629_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20630_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20631_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20632_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_DLMIX_20633_gNB_FAPI_s13.h5','TVnr_ULMIX_21513_gNB_FAPI_s13.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_ULMIX_21494_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_ULMIX_21495_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20636_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20637_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9350_gNB_FAPI_s0.h5','TVnr_DLMIX_20638_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_9354_gNB_FAPI_s0.h5','TVnr_DLMIX_20639_gNB_FAPI_s19.h5'};
    case 626 % % 64TR test case 'column H'
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_DLMIX_20560_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21565_gNB_FAPI_s5.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 1}.config{1}.channels = {'TVnr_9368_gNB_FAPI_s0.h5','TVnr_DLMIX_20560_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{1}.channels = {'TVnr_9368_gNB_FAPI_s0.h5','TVnr_DLMIX_20561_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20562_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_9365_gNB_FAPI_s0.h5','TVnr_DLMIX_20563_gNB_FAPI_s3.h5','TVnr_ULMIX_21563_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{1}.channels = {'TVnr_9367_gNB_FAPI_s0.h5','TVnr_ULMIX_21564_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_ULMIX_21565_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20566_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20567_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20568_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20569_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20570_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20571_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20572_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_9365_gNB_FAPI_s0.h5','TVnr_DLMIX_20573_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_9367_gNB_FAPI_s0.h5','TVnr_ULMIX_21574_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_ULMIX_21575_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20576_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20577_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20578_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20579_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20580_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20581_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20582_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9365_gNB_FAPI_s0.h5','TVnr_DLMIX_20583_gNB_FAPI_s3.h5','TVnr_ULMIX_21573_gNB_FAPI_s3.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9367_gNB_FAPI_s0.h5','TVnr_ULMIX_21564_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_ULMIX_21565_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20586_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20587_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20588_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20589_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20590_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20591_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20592_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9365_gNB_FAPI_s0.h5','TVnr_DLMIX_20593_gNB_FAPI_s13.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9367_gNB_FAPI_s0.h5','TVnr_ULMIX_21574_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_ULMIX_21575_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20596_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20597_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9366_gNB_FAPI_s0.h5','TVnr_DLMIX_20598_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_9368_gNB_FAPI_s0.h5','TVnr_DLMIX_20599_gNB_FAPI_s19.h5'};

    case 627 % % 64TR test case 'column G' 40 MHz
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_DLMIX_20640_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21585_gNB_FAPI_s5.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 1}.config{1}.channels = {'TVnr_9370_gNB_FAPI_s0.h5','TVnr_DLMIX_20640_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{1}.channels = {'TVnr_9370_gNB_FAPI_s0.h5','TVnr_DLMIX_20641_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20642_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_9369_gNB_FAPI_s0.h5','TVnr_DLMIX_20643_gNB_FAPI_s3.h5','TVnr_ULMIX_21583_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{1}.channels = {'TVnr_9371_gNB_FAPI_s0.h5','TVnr_ULMIX_21584_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_ULMIX_21585_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20646_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20647_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20648_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20649_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20650_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20651_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20652_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_9369_gNB_FAPI_s0.h5','TVnr_DLMIX_20653_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_9371_gNB_FAPI_s0.h5','TVnr_ULMIX_21594_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_ULMIX_21595_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20656_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20657_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20658_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20659_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20660_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20661_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20662_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9369_gNB_FAPI_s0.h5','TVnr_DLMIX_20663_gNB_FAPI_s3.h5','TVnr_ULMIX_21593_gNB_FAPI_s3.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9371_gNB_FAPI_s0.h5','TVnr_ULMIX_21584_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_ULMIX_21585_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20666_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20667_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20668_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20669_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20670_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20671_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20672_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9369_gNB_FAPI_s0.h5','TVnr_DLMIX_20673_gNB_FAPI_s13.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9371_gNB_FAPI_s0.h5','TVnr_ULMIX_21594_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_ULMIX_21595_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20676_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20677_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20678_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_9370_gNB_FAPI_s0.h5','TVnr_DLMIX_20679_gNB_FAPI_s19.h5'};
    
    case 628  % 3-cell 100 MHz / 40 MHz / 40 MHz
        % reusing 100 MHz TVs from 90624 for Cell 0; 40 MHz TVs from 90627 for Cell 1 and Cell 2. SSB on all three cells
        nCell = 3;
        LP = []; 
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_DLMIX_20560_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20740_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20743_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21525_gNB_FAPI_s5.h5', 'TVnr_ULMIX_21585_gNB_FAPI_s5.h5', 'TVnr_ULMIX_21585_gNB_FAPI_s5.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        % Cell 0 from 90624
        LP.SCHED{ 1}.config{1}.channels = {'TVnr_9364_gNB_FAPI_s0.h5','TVnr_DLMIX_20560_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{1}.channels = {'TVnr_9364_gNB_FAPI_s0.h5','TVnr_DLMIX_20561_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20562_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_9357_gNB_FAPI_s0.h5','TVnr_DLMIX_20563_gNB_FAPI_s3.h5','TVnr_ULMIX_21523_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{1}.channels = {'TVnr_9355_gNB_FAPI_s0.h5','TVnr_ULMIX_21524_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_ULMIX_21525_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20566_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20567_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20568_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20569_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20570_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20571_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20572_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_9357_gNB_FAPI_s0.h5','TVnr_DLMIX_20573_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_9355_gNB_FAPI_s0.h5','TVnr_ULMIX_21534_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_ULMIX_21535_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20576_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20577_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20578_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20579_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20580_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20581_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20582_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9357_gNB_FAPI_s0.h5','TVnr_DLMIX_20583_gNB_FAPI_s3.h5','TVnr_ULMIX_21533_gNB_FAPI_s3.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9355_gNB_FAPI_s0.h5','TVnr_ULMIX_21524_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_ULMIX_21525_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20586_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20587_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20588_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20589_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20590_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20591_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20592_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9357_gNB_FAPI_s0.h5','TVnr_DLMIX_20593_gNB_FAPI_s13.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9355_gNB_FAPI_s0.h5','TVnr_ULMIX_21534_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_ULMIX_21535_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20596_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20597_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20598_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_9364_gNB_FAPI_s0.h5','TVnr_DLMIX_20599_gNB_FAPI_s19.h5'};

        % Cell 1 from 90627
        LP.SCHED{ 1}.config{2}.channels = {'TVnr_9370_gNB_FAPI_s0.h5','TVnr_DLMIX_20740_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{2}.channels = {'TVnr_9370_gNB_FAPI_s0.h5','TVnr_DLMIX_20741_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20742_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{2}.channels = {'TVnr_9369_gNB_FAPI_s0.h5','TVnr_DLMIX_20643_gNB_FAPI_s3.h5','TVnr_ULMIX_21583_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{2}.channels = {'TVnr_9371_gNB_FAPI_s0.h5','TVnr_ULMIX_21584_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_ULMIX_21585_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20646_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20647_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20648_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20649_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20650_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20651_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20652_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{2}.channels = {'TVnr_9369_gNB_FAPI_s0.h5','TVnr_DLMIX_20653_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{2}.channels = {'TVnr_9371_gNB_FAPI_s0.h5','TVnr_ULMIX_21594_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_ULMIX_21595_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20656_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20657_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20658_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20659_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20660_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20661_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20662_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{2}.channels = {'TVnr_9369_gNB_FAPI_s0.h5','TVnr_DLMIX_20663_gNB_FAPI_s3.h5','TVnr_ULMIX_21593_gNB_FAPI_s3.h5'};
        LP.SCHED{25}.config{2}.channels = {'TVnr_9371_gNB_FAPI_s0.h5','TVnr_ULMIX_21584_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_ULMIX_21585_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20666_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20667_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20668_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20669_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20670_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20671_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20672_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{2}.channels = {'TVnr_9369_gNB_FAPI_s0.h5','TVnr_DLMIX_20673_gNB_FAPI_s13.h5'};
        LP.SCHED{35}.config{2}.channels = {'TVnr_9371_gNB_FAPI_s0.h5','TVnr_ULMIX_21594_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_ULMIX_21595_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20676_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20677_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20678_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{2}.channels = {'TVnr_9370_gNB_FAPI_s0.h5','TVnr_DLMIX_20679_gNB_FAPI_s19.h5'};

        % Cell 2 from 90627
        LP.SCHED{ 1}.config{3}.channels = {'TVnr_9370_gNB_FAPI_s0.h5','TVnr_DLMIX_20743_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{3}.channels = {'TVnr_9370_gNB_FAPI_s0.h5','TVnr_DLMIX_20744_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20745_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{3}.channels = {'TVnr_9369_gNB_FAPI_s0.h5','TVnr_DLMIX_20643_gNB_FAPI_s3.h5','TVnr_ULMIX_21583_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{3}.channels = {'TVnr_9371_gNB_FAPI_s0.h5','TVnr_ULMIX_21584_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_ULMIX_21585_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20646_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20647_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20648_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20649_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20650_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20651_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20652_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{3}.channels = {'TVnr_9369_gNB_FAPI_s0.h5','TVnr_DLMIX_20653_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{3}.channels = {'TVnr_9371_gNB_FAPI_s0.h5','TVnr_ULMIX_21594_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_ULMIX_21595_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20656_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20657_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20658_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20659_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20660_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20661_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20662_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{3}.channels = {'TVnr_9369_gNB_FAPI_s0.h5','TVnr_DLMIX_20663_gNB_FAPI_s3.h5','TVnr_ULMIX_21593_gNB_FAPI_s3.h5'};
        LP.SCHED{25}.config{3}.channels = {'TVnr_9371_gNB_FAPI_s0.h5','TVnr_ULMIX_21584_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_ULMIX_21585_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20666_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20667_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20668_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20669_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20670_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20671_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20672_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{3}.channels = {'TVnr_9369_gNB_FAPI_s0.h5','TVnr_DLMIX_20673_gNB_FAPI_s13.h5'};
        LP.SCHED{35}.config{3}.channels = {'TVnr_9371_gNB_FAPI_s0.h5','TVnr_ULMIX_21594_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_ULMIX_21595_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20676_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20677_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20678_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{3}.channels = {'TVnr_9370_gNB_FAPI_s0.h5','TVnr_DLMIX_20679_gNB_FAPI_s19.h5'};

    case 629 % % 64TR test case 'column G', 64 UEs per TTI
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_DLMIX_20690_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21605_gNB_FAPI_s5.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 1}.config{1}.channels = {'TVnr_9382_gNB_FAPI_s0.h5','TVnr_DLMIX_20690_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{1}.channels = {'TVnr_9382_gNB_FAPI_s0.h5','TVnr_DLMIX_20691_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20692_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_9381_gNB_FAPI_s0.h5','TVnr_DLMIX_20693_gNB_FAPI_s3.h5','TVnr_ULMIX_21603_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{1}.channels = {'TVnr_9383_gNB_FAPI_s0.h5','TVnr_ULMIX_21604_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_ULMIX_21605_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20696_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20697_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20698_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20699_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20700_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20701_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20702_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_9381_gNB_FAPI_s0.h5','TVnr_DLMIX_20703_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_9383_gNB_FAPI_s0.h5','TVnr_ULMIX_21614_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_ULMIX_21615_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20706_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20707_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20708_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20709_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20710_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20711_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20712_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9381_gNB_FAPI_s0.h5','TVnr_DLMIX_20713_gNB_FAPI_s3.h5','TVnr_ULMIX_21613_gNB_FAPI_s3.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9383_gNB_FAPI_s0.h5','TVnr_ULMIX_21604_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_ULMIX_21605_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20716_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20717_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20718_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20719_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20720_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20721_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20722_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9381_gNB_FAPI_s0.h5','TVnr_DLMIX_20723_gNB_FAPI_s13.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9383_gNB_FAPI_s0.h5','TVnr_ULMIX_21614_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_ULMIX_21615_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20726_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20727_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9384_gNB_FAPI_s0.h5','TVnr_DLMIX_20728_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_9382_gNB_FAPI_s0.h5','TVnr_DLMIX_20729_gNB_FAPI_s19.h5'};

    case 630  % 64TR static + dynamic beamforming SU/MU-MIMO 3-UEG -- 2 CELLS
        nCell = 2;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_DLMIX_20200_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20220_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21205_gNB_FAPI_s5.h5', 'TVnr_ULMIX_21225_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_ULMIX_21203_gNB_FAPI_s3.h5'};
        LP.SCHED{ 4}.config{2}.channels = {'TVnr_ULMIX_21223_gNB_FAPI_s3.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20200_gNB_FAPI_s0.h5'};
        LP.SCHED{21}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20220_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20201_gNB_FAPI_s1.h5'};
        LP.SCHED{22}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20221_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_DLMIX_20202_gNB_FAPI_s2.h5'};
        LP.SCHED{23}.config{2}.channels = {'TVnr_DLMIX_20222_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9309_gNB_FAPI_s0.h5'};
        LP.SCHED{24}.config{2}.channels = {'TVnr_9311_gNB_FAPI_s0.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9309_gNB_FAPI_s0.h5','TVnr_ULMIX_21204_gNB_FAPI_s4.h5'};
        LP.SCHED{25}.config{2}.channels = {'TVnr_9311_gNB_FAPI_s0.h5','TVnr_ULMIX_21224_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_ULMIX_21205_gNB_FAPI_s5.h5'};
        LP.SCHED{26}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_ULMIX_21225_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20206_gNB_FAPI_s6.h5'};
        LP.SCHED{27}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20226_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20207_gNB_FAPI_s7.h5'};
        LP.SCHED{28}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20227_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20208_gNB_FAPI_s8.h5'};
        LP.SCHED{29}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20228_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20209_gNB_FAPI_s9.h5'};
        LP.SCHED{30}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20229_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20210_gNB_FAPI_s10.h5'};
        LP.SCHED{31}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20230_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20211_gNB_FAPI_s11.h5'};
        LP.SCHED{32}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20231_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_DLMIX_20212_gNB_FAPI_s12.h5'};
        LP.SCHED{33}.config{2}.channels = {'TVnr_DLMIX_20232_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9309_gNB_FAPI_s0.h5'};
        LP.SCHED{34}.config{2}.channels = {'TVnr_9311_gNB_FAPI_s0.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9309_gNB_FAPI_s0.h5','TVnr_ULMIX_21214_gNB_FAPI_s14.h5'};
        LP.SCHED{35}.config{2}.channels = {'TVnr_9311_gNB_FAPI_s0.h5','TVnr_ULMIX_21234_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_ULMIX_21215_gNB_FAPI_s15.h5'};
        LP.SCHED{36}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_ULMIX_21235_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20216_gNB_FAPI_s16.h5'};
        LP.SCHED{37}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20236_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20217_gNB_FAPI_s17.h5'};
        LP.SCHED{38}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20237_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20218_gNB_FAPI_s18.h5'};
        LP.SCHED{39}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20238_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_DLMIX_20219_gNB_FAPI_s19.h5'};
        LP.SCHED{40}.config{2}.channels = {'TVnr_DLMIX_20239_gNB_FAPI_s19.h5'};
    case 631  % 64TR static + dynamic beamforming SU/MU-MIMO 3-UEG -- 3 CELLS
        nCell = 3;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_DLMIX_20200_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20220_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20240_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21205_gNB_FAPI_s5.h5', 'TVnr_ULMIX_21225_gNB_FAPI_s5.h5', 'TVnr_ULMIX_21245_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_ULMIX_21203_gNB_FAPI_s3.h5'};
        LP.SCHED{ 4}.config{2}.channels = {'TVnr_ULMIX_21223_gNB_FAPI_s3.h5'};
        LP.SCHED{ 4}.config{3}.channels = {'TVnr_ULMIX_21243_gNB_FAPI_s3.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20200_gNB_FAPI_s0.h5'};
        LP.SCHED{21}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20220_gNB_FAPI_s0.h5'};
        LP.SCHED{21}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20240_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20201_gNB_FAPI_s1.h5'};
        LP.SCHED{22}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20221_gNB_FAPI_s1.h5'};
        LP.SCHED{22}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20241_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_DLMIX_20202_gNB_FAPI_s2.h5'};
        LP.SCHED{23}.config{2}.channels = {'TVnr_DLMIX_20222_gNB_FAPI_s2.h5'};
        LP.SCHED{23}.config{3}.channels = {'TVnr_DLMIX_20242_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9309_gNB_FAPI_s0.h5'};
        LP.SCHED{24}.config{2}.channels = {'TVnr_9311_gNB_FAPI_s0.h5'};
        LP.SCHED{24}.config{3}.channels = {'TVnr_9313_gNB_FAPI_s0.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9309_gNB_FAPI_s0.h5','TVnr_ULMIX_21204_gNB_FAPI_s4.h5'};
        LP.SCHED{25}.config{2}.channels = {'TVnr_9311_gNB_FAPI_s0.h5','TVnr_ULMIX_21224_gNB_FAPI_s4.h5'};
        LP.SCHED{25}.config{3}.channels = {'TVnr_9313_gNB_FAPI_s0.h5','TVnr_ULMIX_21244_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_ULMIX_21205_gNB_FAPI_s5.h5'};
        LP.SCHED{26}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_ULMIX_21225_gNB_FAPI_s5.h5'};
        LP.SCHED{26}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_ULMIX_21245_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20206_gNB_FAPI_s6.h5'};
        LP.SCHED{27}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20226_gNB_FAPI_s6.h5'};
        LP.SCHED{27}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20246_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20207_gNB_FAPI_s7.h5'};
        LP.SCHED{28}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20227_gNB_FAPI_s7.h5'};
        LP.SCHED{28}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20247_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20208_gNB_FAPI_s8.h5'};
        LP.SCHED{29}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20228_gNB_FAPI_s8.h5'};
        LP.SCHED{29}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20248_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20209_gNB_FAPI_s9.h5'};
        LP.SCHED{30}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20229_gNB_FAPI_s9.h5'};
        LP.SCHED{30}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20249_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20210_gNB_FAPI_s10.h5'};
        LP.SCHED{31}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20230_gNB_FAPI_s10.h5'};
        LP.SCHED{31}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20250_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20211_gNB_FAPI_s11.h5'};
        LP.SCHED{32}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20231_gNB_FAPI_s11.h5'};
        LP.SCHED{32}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20251_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_DLMIX_20212_gNB_FAPI_s12.h5'};
        LP.SCHED{33}.config{2}.channels = {'TVnr_DLMIX_20232_gNB_FAPI_s12.h5'};
        LP.SCHED{33}.config{3}.channels = {'TVnr_DLMIX_20252_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9309_gNB_FAPI_s0.h5'};
        LP.SCHED{34}.config{2}.channels = {'TVnr_9311_gNB_FAPI_s0.h5'};
        LP.SCHED{34}.config{3}.channels = {'TVnr_9313_gNB_FAPI_s0.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9309_gNB_FAPI_s0.h5','TVnr_ULMIX_21214_gNB_FAPI_s14.h5'};
        LP.SCHED{35}.config{2}.channels = {'TVnr_9311_gNB_FAPI_s0.h5','TVnr_ULMIX_21234_gNB_FAPI_s14.h5'};
        LP.SCHED{35}.config{3}.channels = {'TVnr_9313_gNB_FAPI_s0.h5','TVnr_ULMIX_21254_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_ULMIX_21215_gNB_FAPI_s15.h5'};
        LP.SCHED{36}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_ULMIX_21235_gNB_FAPI_s15.h5'};
        LP.SCHED{36}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_ULMIX_21255_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20216_gNB_FAPI_s16.h5'};
        LP.SCHED{37}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20236_gNB_FAPI_s16.h5'};
        LP.SCHED{37}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20256_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20217_gNB_FAPI_s17.h5'};
        LP.SCHED{38}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20237_gNB_FAPI_s17.h5'};
        LP.SCHED{38}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20257_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9310_gNB_FAPI_s0.h5','TVnr_DLMIX_20218_gNB_FAPI_s18.h5'};
        LP.SCHED{39}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20238_gNB_FAPI_s18.h5'};
        LP.SCHED{39}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20258_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_DLMIX_20219_gNB_FAPI_s19.h5'};
        LP.SCHED{40}.config{2}.channels = {'TVnr_DLMIX_20239_gNB_FAPI_s19.h5'};
        LP.SCHED{40}.config{3}.channels = {'TVnr_DLMIX_20259_gNB_FAPI_s19.h5'};
    case 632 % % 64TR static + dynamic beamforming SU/MU-MIMO 1-UEG (Cell_0) + 3-UEG (Cell_1) -- 2 CELLS
        nCell = 2;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_DLMIX_20000_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20220_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21005_gNB_FAPI_s5.h5', 'TVnr_ULMIX_21225_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 1}.config{1}.channels = {'TVnr_DLMIX_20000_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{1}.channels = {'TVnr_DLMIX_20001_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{1}.channels = {'TVnr_DLMIX_20002_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_ULMIX_21003_gNB_FAPI_s3.h5'};
        LP.SCHED{ 4}.config{2}.channels = {'TVnr_ULMIX_21223_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{1}.channels = {'TVnr_ULMIX_21004_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{1}.channels = {'TVnr_ULMIX_21005_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{1}.channels = {'TVnr_DLMIX_20006_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{1}.channels = {'TVnr_DLMIX_20007_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{1}.channels = {'TVnr_DLMIX_20008_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_DLMIX_20009_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_DLMIX_20010_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_DLMIX_20011_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_DLMIX_20012_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_ULMIX_21013_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_ULMIX_21014_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_ULMIX_21015_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_DLMIX_20016_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_DLMIX_20017_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_DLMIX_20018_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_DLMIX_20019_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_DLMIX_20020_gNB_FAPI_s20.h5'};
        LP.SCHED{21}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20220_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_DLMIX_20021_gNB_FAPI_s21.h5'};  
        LP.SCHED{22}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20221_gNB_FAPI_s1.h5'};   
        LP.SCHED{23}.config{1}.channels = {'TVnr_DLMIX_20022_gNB_FAPI_s22.h5'};
        LP.SCHED{23}.config{2}.channels = {'TVnr_DLMIX_20222_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_ULMIX_21023_gNB_FAPI_s23.h5'};
        LP.SCHED{24}.config{2}.channels = {'TVnr_9311_gNB_FAPI_s0.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_ULMIX_21024_gNB_FAPI_s24.h5'};
        LP.SCHED{25}.config{2}.channels = {'TVnr_9311_gNB_FAPI_s0.h5','TVnr_ULMIX_21224_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_ULMIX_21025_gNB_FAPI_s25.h5'};
        LP.SCHED{26}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_ULMIX_21225_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_DLMIX_20026_gNB_FAPI_s26.h5'};
        LP.SCHED{27}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20226_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_DLMIX_20027_gNB_FAPI_s27.h5'};
        LP.SCHED{28}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20227_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_DLMIX_20028_gNB_FAPI_s28.h5'};
        LP.SCHED{29}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20228_gNB_FAPI_s8.h5'}; 
        LP.SCHED{30}.config{1}.channels = {'TVnr_DLMIX_20029_gNB_FAPI_s29.h5'};
        LP.SCHED{30}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20229_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_DLMIX_20030_gNB_FAPI_s30.h5'};
        LP.SCHED{31}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20230_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_DLMIX_20031_gNB_FAPI_s31.h5'};
        LP.SCHED{32}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20231_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_DLMIX_20032_gNB_FAPI_s32.h5'};
        LP.SCHED{33}.config{2}.channels = {'TVnr_DLMIX_20232_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_ULMIX_21033_gNB_FAPI_s33.h5'};
        LP.SCHED{34}.config{2}.channels = {'TVnr_9311_gNB_FAPI_s0.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_ULMIX_21034_gNB_FAPI_s34.h5'};
        LP.SCHED{35}.config{2}.channels = {'TVnr_9311_gNB_FAPI_s0.h5','TVnr_ULMIX_21234_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_ULMIX_21035_gNB_FAPI_s35.h5'};
        LP.SCHED{36}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_ULMIX_21235_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_DLMIX_20036_gNB_FAPI_s36.h5'};
        LP.SCHED{37}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20236_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_DLMIX_20037_gNB_FAPI_s37.h5'};
        LP.SCHED{38}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20237_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_DLMIX_20038_gNB_FAPI_s38.h5'};
        LP.SCHED{39}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20238_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_DLMIX_20039_gNB_FAPI_s39.h5'};
        LP.SCHED{40}.config{2}.channels = {'TVnr_DLMIX_20239_gNB_FAPI_s19.h5'};
    case 633 % % 64TR static + dynamic beamforming SU/MU-MIMO 1-UEG (Cell_0) + 3-UEG (Cell_1) -- 3 CELLS
        nCell = 3;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_DLMIX_20000_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20220_gNB_FAPI_s0.h5','TVnr_DLMIX_20240_gNB_FAPI_s0.h5' };
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21005_gNB_FAPI_s5.h5', 'TVnr_ULMIX_21225_gNB_FAPI_s5.h5', 'TVnr_ULMIX_21245_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 1}.config{1}.channels = {'TVnr_DLMIX_20000_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{1}.channels = {'TVnr_DLMIX_20001_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{1}.channels = {'TVnr_DLMIX_20002_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_ULMIX_21003_gNB_FAPI_s3.h5'};
        LP.SCHED{ 4}.config{2}.channels = {'TVnr_ULMIX_21223_gNB_FAPI_s3.h5'};
        LP.SCHED{ 4}.config{3}.channels = {'TVnr_ULMIX_21243_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{1}.channels = {'TVnr_ULMIX_21004_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{1}.channels = {'TVnr_ULMIX_21005_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{1}.channels = {'TVnr_DLMIX_20006_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{1}.channels = {'TVnr_DLMIX_20007_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{1}.channels = {'TVnr_DLMIX_20008_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_DLMIX_20009_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_DLMIX_20010_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_DLMIX_20011_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_DLMIX_20012_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_ULMIX_21013_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_ULMIX_21014_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_ULMIX_21015_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_DLMIX_20016_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_DLMIX_20017_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_DLMIX_20018_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_DLMIX_20019_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_DLMIX_20020_gNB_FAPI_s20.h5'};
        LP.SCHED{21}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20220_gNB_FAPI_s0.h5'};
        LP.SCHED{21}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20240_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_DLMIX_20021_gNB_FAPI_s21.h5'};  
        LP.SCHED{22}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20221_gNB_FAPI_s1.h5'};
        LP.SCHED{22}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20241_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_DLMIX_20022_gNB_FAPI_s22.h5'};
        LP.SCHED{23}.config{2}.channels = {'TVnr_DLMIX_20222_gNB_FAPI_s2.h5'};
        LP.SCHED{23}.config{3}.channels = {'TVnr_DLMIX_20242_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_ULMIX_21023_gNB_FAPI_s23.h5'};
        LP.SCHED{24}.config{2}.channels = {'TVnr_9311_gNB_FAPI_s0.h5'};
        LP.SCHED{24}.config{3}.channels = {'TVnr_9313_gNB_FAPI_s0.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_ULMIX_21024_gNB_FAPI_s24.h5'};
        LP.SCHED{25}.config{2}.channels = {'TVnr_9311_gNB_FAPI_s0.h5','TVnr_ULMIX_21224_gNB_FAPI_s4.h5'};
        LP.SCHED{25}.config{3}.channels = {'TVnr_9313_gNB_FAPI_s0.h5','TVnr_ULMIX_21244_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_ULMIX_21025_gNB_FAPI_s25.h5'};
        LP.SCHED{26}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_ULMIX_21225_gNB_FAPI_s5.h5'};
        LP.SCHED{26}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_ULMIX_21245_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_DLMIX_20026_gNB_FAPI_s26.h5'};
        LP.SCHED{27}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20226_gNB_FAPI_s6.h5'};
        LP.SCHED{27}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20246_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_DLMIX_20027_gNB_FAPI_s27.h5'};
        LP.SCHED{28}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20227_gNB_FAPI_s7.h5'};
        LP.SCHED{28}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20247_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_DLMIX_20028_gNB_FAPI_s28.h5'};
        LP.SCHED{29}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20228_gNB_FAPI_s8.h5'};
        LP.SCHED{29}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20248_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_DLMIX_20029_gNB_FAPI_s29.h5'};
        LP.SCHED{30}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20229_gNB_FAPI_s9.h5'};
        LP.SCHED{30}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20249_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_DLMIX_20030_gNB_FAPI_s30.h5'};
        LP.SCHED{31}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20230_gNB_FAPI_s10.h5'};
        LP.SCHED{31}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20250_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_DLMIX_20031_gNB_FAPI_s31.h5'};
        LP.SCHED{32}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20231_gNB_FAPI_s11.h5'};
        LP.SCHED{32}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20251_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_DLMIX_20032_gNB_FAPI_s32.h5'};
        LP.SCHED{33}.config{2}.channels = {'TVnr_DLMIX_20232_gNB_FAPI_s12.h5'};
        LP.SCHED{33}.config{3}.channels = {'TVnr_DLMIX_20252_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_ULMIX_21033_gNB_FAPI_s33.h5'};
        LP.SCHED{34}.config{2}.channels = {'TVnr_9311_gNB_FAPI_s0.h5'};
        LP.SCHED{34}.config{3}.channels = {'TVnr_9313_gNB_FAPI_s0.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_ULMIX_21034_gNB_FAPI_s34.h5'};
        LP.SCHED{35}.config{2}.channels = {'TVnr_9311_gNB_FAPI_s0.h5','TVnr_ULMIX_21234_gNB_FAPI_s14.h5'};
        LP.SCHED{35}.config{3}.channels = {'TVnr_9313_gNB_FAPI_s0.h5','TVnr_ULMIX_21254_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_ULMIX_21035_gNB_FAPI_s35.h5'};
        LP.SCHED{36}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_ULMIX_21235_gNB_FAPI_s15.h5'};
        LP.SCHED{36}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_ULMIX_21255_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_DLMIX_20036_gNB_FAPI_s36.h5'};
        LP.SCHED{37}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20236_gNB_FAPI_s16.h5'};
        LP.SCHED{37}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20256_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_DLMIX_20037_gNB_FAPI_s37.h5'};
        LP.SCHED{38}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20237_gNB_FAPI_s17.h5'};
        LP.SCHED{38}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20257_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_DLMIX_20038_gNB_FAPI_s38.h5'};
        LP.SCHED{39}.config{2}.channels = {'TVnr_9312_gNB_FAPI_s0.h5','TVnr_DLMIX_20238_gNB_FAPI_s18.h5'};
        LP.SCHED{39}.config{3}.channels = {'TVnr_9314_gNB_FAPI_s0.h5','TVnr_DLMIX_20258_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_DLMIX_20039_gNB_FAPI_s39.h5'};
        LP.SCHED{40}.config{2}.channels = {'TVnr_DLMIX_20239_gNB_FAPI_s19.h5'};  
        LP.SCHED{40}.config{3}.channels = {'TVnr_DLMIX_20259_gNB_FAPI_s19.h5'};

    case 634 %% 40 slot pattern: 64TR BFW + PDSCH/PUSCH on all slots (DDDSUUDDDD), 3 cells
        % SRS in slot 3 with 8 PDUs, PXSCH in slot 20~39
        nCell = 3;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_DLMIX_20460_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20480_gNB_FAPI_s0.h5','TVnr_DLMIX_20500_gNB_FAPI_s0.h5' };
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21425_gNB_FAPI_s25.h5', 'TVnr_ULMIX_21445_gNB_FAPI_s25.h5', 'TVnr_ULMIX_21465_gNB_FAPI_s25.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        % S
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_ULMIX_21433_gNB_FAPI_s3.h5'}; % SRS only, 8 UEs
        LP.SCHED{ 4}.config{2}.channels = {'TVnr_ULMIX_21453_gNB_FAPI_s3.h5'};
        LP.SCHED{ 4}.config{3}.channels = {'TVnr_ULMIX_21473_gNB_FAPI_s3.h5'};

        % DDD
        LP.SCHED{21}.config{1}.channels = {'TVnr_9336_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20460_gNB_FAPI_s0.h5'};
        LP.SCHED{21}.config{2}.channels = {'TVnr_9338_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20480_gNB_FAPI_s0.h5'};
        LP.SCHED{21}.config{3}.channels = {'TVnr_9340_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20500_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9336_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20461_gNB_FAPI_s1.h5'};
        LP.SCHED{22}.config{2}.channels = {'TVnr_9338_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20481_gNB_FAPI_s1.h5'};
        LP.SCHED{22}.config{3}.channels = {'TVnr_9340_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20501_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_DLMIX_20462_gNB_FAPI_s2.h5'};
        LP.SCHED{23}.config{2}.channels = {'TVnr_DLMIX_20482_gNB_FAPI_s2.h5'};
        LP.SCHED{23}.config{3}.channels = {'TVnr_DLMIX_20502_gNB_FAPI_s2.h5'};
        % SUU
        LP.SCHED{24}.config{1}.channels = {'TVnr_9335_gNB_FAPI_s0.h5'};
        LP.SCHED{24}.config{2}.channels = {'TVnr_9337_gNB_FAPI_s0.h5'};
        LP.SCHED{24}.config{3}.channels = {'TVnr_9339_gNB_FAPI_s0.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9335_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21424_gNB_FAPI_s24.h5'};
        LP.SCHED{25}.config{2}.channels = {'TVnr_9337_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21444_gNB_FAPI_s24.h5'};
        LP.SCHED{25}.config{3}.channels = {'TVnr_9339_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21464_gNB_FAPI_s24.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9336_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21425_gNB_FAPI_s25.h5'};
        LP.SCHED{26}.config{2}.channels = {'TVnr_9338_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21445_gNB_FAPI_s25.h5'};
        LP.SCHED{26}.config{3}.channels = {'TVnr_9340_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21465_gNB_FAPI_s25.h5'};
        % DDDD
        LP.SCHED{27}.config{1}.channels = {'TVnr_9336_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20466_gNB_FAPI_s6.h5'};
        LP.SCHED{27}.config{2}.channels = {'TVnr_9338_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20486_gNB_FAPI_s6.h5'};
        LP.SCHED{27}.config{3}.channels = {'TVnr_9340_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20506_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9336_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20467_gNB_FAPI_s7.h5'};
        LP.SCHED{28}.config{2}.channels = {'TVnr_9338_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20487_gNB_FAPI_s7.h5'};
        LP.SCHED{28}.config{3}.channels = {'TVnr_9340_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20507_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9336_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20468_gNB_FAPI_s8.h5'};
        LP.SCHED{29}.config{2}.channels = {'TVnr_9338_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20488_gNB_FAPI_s8.h5'};
        LP.SCHED{29}.config{3}.channels = {'TVnr_9340_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20508_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9336_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20469_gNB_FAPI_s9.h5'};
        LP.SCHED{30}.config{2}.channels = {'TVnr_9338_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20489_gNB_FAPI_s9.h5'};
        LP.SCHED{30}.config{3}.channels = {'TVnr_9340_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20509_gNB_FAPI_s9.h5'};

        % DDD
        LP.SCHED{31}.config{1}.channels = {'TVnr_9336_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20470_gNB_FAPI_s10.h5'};
        LP.SCHED{31}.config{2}.channels = {'TVnr_9338_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20490_gNB_FAPI_s10.h5'};
        LP.SCHED{31}.config{3}.channels = {'TVnr_9340_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20510_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9336_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20471_gNB_FAPI_s11.h5'};
        LP.SCHED{32}.config{2}.channels = {'TVnr_9338_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20491_gNB_FAPI_s11.h5'};
        LP.SCHED{32}.config{3}.channels = {'TVnr_9340_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20511_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_DLMIX_20472_gNB_FAPI_s12.h5'};
        LP.SCHED{33}.config{2}.channels = {'TVnr_DLMIX_20492_gNB_FAPI_s12.h5'};
        LP.SCHED{33}.config{3}.channels = {'TVnr_DLMIX_20512_gNB_FAPI_s12.h5'};
        % SUU
        LP.SCHED{34}.config{1}.channels = {'TVnr_9335_gNB_FAPI_s0.h5'};
        LP.SCHED{34}.config{2}.channels = {'TVnr_9337_gNB_FAPI_s0.h5'};
        LP.SCHED{34}.config{3}.channels = {'TVnr_9339_gNB_FAPI_s0.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9335_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21434_gNB_FAPI_s34.h5'};
        LP.SCHED{35}.config{2}.channels = {'TVnr_9337_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21454_gNB_FAPI_s34.h5'};
        LP.SCHED{35}.config{3}.channels = {'TVnr_9339_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21474_gNB_FAPI_s34.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9336_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21435_gNB_FAPI_s35.h5'};
        LP.SCHED{36}.config{2}.channels = {'TVnr_9338_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21455_gNB_FAPI_s35.h5'};
        LP.SCHED{36}.config{3}.channels = {'TVnr_9340_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21475_gNB_FAPI_s35.h5'};
        % DDDD
        LP.SCHED{37}.config{1}.channels = {'TVnr_9336_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20476_gNB_FAPI_s16.h5'};
        LP.SCHED{37}.config{2}.channels = {'TVnr_9338_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20496_gNB_FAPI_s16.h5'};
        LP.SCHED{37}.config{3}.channels = {'TVnr_9340_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20516_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9336_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20477_gNB_FAPI_s17.h5'};
        LP.SCHED{38}.config{2}.channels = {'TVnr_9338_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20497_gNB_FAPI_s17.h5'};
        LP.SCHED{38}.config{3}.channels = {'TVnr_9340_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20517_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9336_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20478_gNB_FAPI_s18.h5'};
        LP.SCHED{39}.config{2}.channels = {'TVnr_9338_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20498_gNB_FAPI_s18.h5'};
        LP.SCHED{39}.config{3}.channels = {'TVnr_9340_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20518_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_DLMIX_20479_gNB_FAPI_s19.h5'};
        LP.SCHED{40}.config{2}.channels = {'TVnr_DLMIX_20499_gNB_FAPI_s19.h5'};
        LP.SCHED{40}.config{3}.channels = {'TVnr_DLMIX_20519_gNB_FAPI_s19.h5'};

    case 635 %% 40 slot pattern: 64TR BFW + PDSCH/PUSCH on all slots (DDDSUUDDDD), 3 cells
        % SRS in slot 3 with 16 PDUs, PXSCH in slot 20~39
        nCell = 3;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_DLMIX_20400_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20420_gNB_FAPI_s0.h5','TVnr_DLMIX_20440_gNB_FAPI_s0.h5' };
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21425_gNB_FAPI_s25.h5', 'TVnr_ULMIX_21445_gNB_FAPI_s25.h5', 'TVnr_ULMIX_21465_gNB_FAPI_s25.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        % S
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_ULMIX_21423_gNB_FAPI_s3.h5'}; % SRS only, 16 UEs
        LP.SCHED{ 4}.config{2}.channels = {'TVnr_ULMIX_21443_gNB_FAPI_s3.h5'};
        LP.SCHED{ 4}.config{3}.channels = {'TVnr_ULMIX_21463_gNB_FAPI_s3.h5'};

        % DDD
        LP.SCHED{21}.config{1}.channels = {'TVnr_9330_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20400_gNB_FAPI_s0.h5'};
        LP.SCHED{21}.config{2}.channels = {'TVnr_9332_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20420_gNB_FAPI_s0.h5'};
        LP.SCHED{21}.config{3}.channels = {'TVnr_9334_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20440_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9330_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20401_gNB_FAPI_s1.h5'};
        LP.SCHED{22}.config{2}.channels = {'TVnr_9332_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20421_gNB_FAPI_s1.h5'};
        LP.SCHED{22}.config{3}.channels = {'TVnr_9334_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20441_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_DLMIX_20402_gNB_FAPI_s2.h5'};
        LP.SCHED{23}.config{2}.channels = {'TVnr_DLMIX_20422_gNB_FAPI_s2.h5'};
        LP.SCHED{23}.config{3}.channels = {'TVnr_DLMIX_20442_gNB_FAPI_s2.h5'};
        % SUU
        LP.SCHED{24}.config{1}.channels = {'TVnr_9329_gNB_FAPI_s0.h5'};
        LP.SCHED{24}.config{2}.channels = {'TVnr_9331_gNB_FAPI_s0.h5'};
        LP.SCHED{24}.config{3}.channels = {'TVnr_9333_gNB_FAPI_s0.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9329_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21424_gNB_FAPI_s24.h5'};
        LP.SCHED{25}.config{2}.channels = {'TVnr_9331_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21444_gNB_FAPI_s24.h5'};
        LP.SCHED{25}.config{3}.channels = {'TVnr_9333_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21464_gNB_FAPI_s24.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9330_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21425_gNB_FAPI_s25.h5'};
        LP.SCHED{26}.config{2}.channels = {'TVnr_9332_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21445_gNB_FAPI_s25.h5'};
        LP.SCHED{26}.config{3}.channels = {'TVnr_9334_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21465_gNB_FAPI_s25.h5'};
        % DDDD
        LP.SCHED{27}.config{1}.channels = {'TVnr_9330_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20406_gNB_FAPI_s6.h5'};
        LP.SCHED{27}.config{2}.channels = {'TVnr_9332_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20426_gNB_FAPI_s6.h5'};
        LP.SCHED{27}.config{3}.channels = {'TVnr_9334_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20446_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9330_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20407_gNB_FAPI_s7.h5'};
        LP.SCHED{28}.config{2}.channels = {'TVnr_9332_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20427_gNB_FAPI_s7.h5'};
        LP.SCHED{28}.config{3}.channels = {'TVnr_9334_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20447_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9330_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20408_gNB_FAPI_s8.h5'};
        LP.SCHED{29}.config{2}.channels = {'TVnr_9332_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20428_gNB_FAPI_s8.h5'};
        LP.SCHED{29}.config{3}.channels = {'TVnr_9334_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20448_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9330_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20409_gNB_FAPI_s9.h5'};
        LP.SCHED{30}.config{2}.channels = {'TVnr_9332_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20429_gNB_FAPI_s9.h5'};
        LP.SCHED{30}.config{3}.channels = {'TVnr_9334_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20449_gNB_FAPI_s9.h5'};

        % DDD
        LP.SCHED{31}.config{1}.channels = {'TVnr_9330_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20410_gNB_FAPI_s10.h5'};
        LP.SCHED{31}.config{2}.channels = {'TVnr_9332_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20430_gNB_FAPI_s10.h5'};
        LP.SCHED{31}.config{3}.channels = {'TVnr_9334_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20450_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9330_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20411_gNB_FAPI_s11.h5'};
        LP.SCHED{32}.config{2}.channels = {'TVnr_9332_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20431_gNB_FAPI_s11.h5'};
        LP.SCHED{32}.config{3}.channels = {'TVnr_9334_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20451_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_DLMIX_20412_gNB_FAPI_s12.h5'};
        LP.SCHED{33}.config{2}.channels = {'TVnr_DLMIX_20432_gNB_FAPI_s12.h5'};
        LP.SCHED{33}.config{3}.channels = {'TVnr_DLMIX_20452_gNB_FAPI_s12.h5'};
        % SUU
        LP.SCHED{34}.config{1}.channels = {'TVnr_9329_gNB_FAPI_s0.h5'};
        LP.SCHED{34}.config{2}.channels = {'TVnr_9331_gNB_FAPI_s0.h5'};
        LP.SCHED{34}.config{3}.channels = {'TVnr_9333_gNB_FAPI_s0.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9329_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21434_gNB_FAPI_s34.h5'};
        LP.SCHED{35}.config{2}.channels = {'TVnr_9331_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21454_gNB_FAPI_s34.h5'};
        LP.SCHED{35}.config{3}.channels = {'TVnr_9333_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21474_gNB_FAPI_s34.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9330_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21435_gNB_FAPI_s35.h5'};
        LP.SCHED{36}.config{2}.channels = {'TVnr_9332_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21455_gNB_FAPI_s35.h5'};
        LP.SCHED{36}.config{3}.channels = {'TVnr_9334_gNB_FAPI_s0.h5', 'TVnr_ULMIX_21475_gNB_FAPI_s35.h5'};
        % DDDD
        LP.SCHED{37}.config{1}.channels = {'TVnr_9330_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20416_gNB_FAPI_s16.h5'};
        LP.SCHED{37}.config{2}.channels = {'TVnr_9332_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20436_gNB_FAPI_s16.h5'};
        LP.SCHED{37}.config{3}.channels = {'TVnr_9334_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20456_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9330_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20417_gNB_FAPI_s17.h5'};
        LP.SCHED{38}.config{2}.channels = {'TVnr_9332_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20437_gNB_FAPI_s17.h5'};
        LP.SCHED{38}.config{3}.channels = {'TVnr_9334_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20457_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9330_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20418_gNB_FAPI_s18.h5'};
        LP.SCHED{39}.config{2}.channels = {'TVnr_9332_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20438_gNB_FAPI_s18.h5'};
        LP.SCHED{39}.config{3}.channels = {'TVnr_9334_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20458_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_DLMIX_20419_gNB_FAPI_s19.h5'};
        LP.SCHED{40}.config{2}.channels = {'TVnr_DLMIX_20439_gNB_FAPI_s19.h5'};
        LP.SCHED{40}.config{3}.channels = {'TVnr_DLMIX_20459_gNB_FAPI_s19.h5'};

    case 636  % carrier aggregation 3-cell 100 MHz / 40 MHz / 40 MHz
        % reusing 100 MHz TVs from 90624 for Cell 0; 40 MHz TVs from 90627 for Cell 1 and Cell 2, no SSB for CA, fill 20 PRBs with PDSCH in slot 0/1/2
        nCell = 3;
        LP = []; 
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_DLMIX_20560_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20644_gNB_FAPI_s0.h5', 'TVnr_DLMIX_20644_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21525_gNB_FAPI_s5.h5', 'TVnr_ULMIX_21585_gNB_FAPI_s5.h5', 'TVnr_ULMIX_21585_gNB_FAPI_s5.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        % Cell 0 from 90624
        LP.SCHED{ 1}.config{1}.channels = {'TVnr_9364_gNB_FAPI_s0.h5','TVnr_DLMIX_20560_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{1}.channels = {'TVnr_9364_gNB_FAPI_s0.h5','TVnr_DLMIX_20561_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20562_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_9357_gNB_FAPI_s0.h5','TVnr_DLMIX_20563_gNB_FAPI_s3.h5','TVnr_ULMIX_21523_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{1}.channels = {'TVnr_9355_gNB_FAPI_s0.h5','TVnr_ULMIX_21524_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_ULMIX_21525_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20566_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20567_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20568_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20569_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20570_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20571_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20572_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_9357_gNB_FAPI_s0.h5','TVnr_DLMIX_20573_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_9355_gNB_FAPI_s0.h5','TVnr_ULMIX_21534_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_ULMIX_21535_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20576_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20577_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20578_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20579_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20580_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20581_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20582_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9357_gNB_FAPI_s0.h5','TVnr_DLMIX_20583_gNB_FAPI_s3.h5','TVnr_ULMIX_21533_gNB_FAPI_s3.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9355_gNB_FAPI_s0.h5','TVnr_ULMIX_21524_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_ULMIX_21525_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20586_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20587_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20588_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20589_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20590_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20591_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20592_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9357_gNB_FAPI_s0.h5','TVnr_DLMIX_20593_gNB_FAPI_s13.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9355_gNB_FAPI_s0.h5','TVnr_ULMIX_21534_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_ULMIX_21535_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20596_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20597_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20598_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_9364_gNB_FAPI_s0.h5','TVnr_DLMIX_20599_gNB_FAPI_s19.h5'};
    
        % Cell 1 from 90627, no SSB for CA, fill 20 PRBs with PDSCH in slot 0/1/2
        LP.SCHED{ 1}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20644_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20654_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20655_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{2}.channels = {'TVnr_9369_gNB_FAPI_s0.h5','TVnr_DLMIX_20643_gNB_FAPI_s3.h5','TVnr_ULMIX_21583_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{2}.channels = {'TVnr_9371_gNB_FAPI_s0.h5','TVnr_ULMIX_21584_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_ULMIX_21585_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20646_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20647_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20648_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20649_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20650_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20651_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20652_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{2}.channels = {'TVnr_9369_gNB_FAPI_s0.h5','TVnr_DLMIX_20653_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{2}.channels = {'TVnr_9371_gNB_FAPI_s0.h5','TVnr_ULMIX_21594_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_ULMIX_21595_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20656_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20657_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20658_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20659_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20660_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20661_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20662_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{2}.channels = {'TVnr_9369_gNB_FAPI_s0.h5','TVnr_DLMIX_20663_gNB_FAPI_s3.h5','TVnr_ULMIX_21593_gNB_FAPI_s3.h5'};
        LP.SCHED{25}.config{2}.channels = {'TVnr_9371_gNB_FAPI_s0.h5','TVnr_ULMIX_21584_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_ULMIX_21585_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20666_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20667_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20668_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20669_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20670_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20671_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20672_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{2}.channels = {'TVnr_9369_gNB_FAPI_s0.h5','TVnr_DLMIX_20673_gNB_FAPI_s13.h5'};
        LP.SCHED{35}.config{2}.channels = {'TVnr_9371_gNB_FAPI_s0.h5','TVnr_ULMIX_21594_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_ULMIX_21595_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20676_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20677_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20678_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{2}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20679_gNB_FAPI_s19.h5'};
    
        % Cell 2 from 90627, no SSB for CA, fill 20 PRBs with PDSCH in slot 0/1/2
        LP.SCHED{ 1}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20644_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20654_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20655_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{3}.channels = {'TVnr_9369_gNB_FAPI_s0.h5','TVnr_DLMIX_20643_gNB_FAPI_s3.h5','TVnr_ULMIX_21583_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{3}.channels = {'TVnr_9371_gNB_FAPI_s0.h5','TVnr_ULMIX_21584_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_ULMIX_21585_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20646_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20647_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20648_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20649_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20650_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20651_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20652_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{3}.channels = {'TVnr_9369_gNB_FAPI_s0.h5','TVnr_DLMIX_20653_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{3}.channels = {'TVnr_9371_gNB_FAPI_s0.h5','TVnr_ULMIX_21594_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_ULMIX_21595_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20656_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20657_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20658_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20659_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20660_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20661_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20662_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{3}.channels = {'TVnr_9369_gNB_FAPI_s0.h5','TVnr_DLMIX_20663_gNB_FAPI_s3.h5','TVnr_ULMIX_21593_gNB_FAPI_s3.h5'};
        LP.SCHED{25}.config{3}.channels = {'TVnr_9371_gNB_FAPI_s0.h5','TVnr_ULMIX_21584_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_ULMIX_21585_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20666_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20667_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20668_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20669_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20670_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20671_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20672_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{3}.channels = {'TVnr_9369_gNB_FAPI_s0.h5','TVnr_DLMIX_20673_gNB_FAPI_s13.h5'};
        LP.SCHED{35}.config{3}.channels = {'TVnr_9371_gNB_FAPI_s0.h5','TVnr_ULMIX_21594_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_ULMIX_21595_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20676_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20677_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20678_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{3}.channels = {'TVnr_9372_gNB_FAPI_s0.h5','TVnr_DLMIX_20679_gNB_FAPI_s19.h5'};
    
    case 637 % TC 608 w/ ZP-CSI-RS, SU + MU-MIMO
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_0340_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_0745_gNB_FAPI_s5.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_0743_gNB_FAPI_s3.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0340_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0341_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_0342_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9279_gNB_FAPI_s0.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9279_gNB_FAPI_s0.h5','TVnr_0744_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0745_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_DLMIX_20736_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_DLMIX_20737_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0348_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0349_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0350_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0351_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_0352_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9279_gNB_FAPI_s0.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9279_gNB_FAPI_s0.h5','TVnr_0754_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0755_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0356_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0357_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9280_gNB_FAPI_s0.h5','TVnr_0358_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_0359_gNB_FAPI_s19.h5'};    
        
    case 638  % % 64TR test case 'column I', 80 slot pattern
        nCell = 1;
        LP = [];
        nSlot = 80;
        LP.Cell_Configs = {'TVnr_DLMIX_20750_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21625_gNB_FAPI_s5.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 1}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20750_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20751_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20752_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_9385_gNB_FAPI_s0.h5','TVnr_DLMIX_20753_gNB_FAPI_s3.h5','TVnr_ULMIX_21623_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{1}.channels = {'TVnr_9387_gNB_FAPI_s0.h5','TVnr_ULMIX_21624_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_ULMIX_21625_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20756_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20757_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20758_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_9390_gNB_FAPI_s0.h5','TVnr_DLMIX_20759_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20760_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20761_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20762_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_9385_gNB_FAPI_s0.h5','TVnr_DLMIX_20763_gNB_FAPI_s13.h5','TVnr_ULMIX_21633_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_9387_gNB_FAPI_s0.h5','TVnr_ULMIX_21634_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_ULMIX_21635_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20766_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20767_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20768_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20769_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20770_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20771_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20772_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9385_gNB_FAPI_s0.h5','TVnr_DLMIX_20773_gNB_FAPI_s3.h5','TVnr_ULMIX_21636_gNB_FAPI_s3.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9387_gNB_FAPI_s0.h5','TVnr_ULMIX_21624_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_ULMIX_21625_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20776_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20777_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20778_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20779_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20780_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20781_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20782_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9385_gNB_FAPI_s0.h5','TVnr_DLMIX_20783_gNB_FAPI_s13.h5','TVnr_ULMIX_21637_gNB_FAPI_s13.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9387_gNB_FAPI_s0.h5','TVnr_ULMIX_21634_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_ULMIX_21635_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20786_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20787_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20788_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_9386_gNB_FAPI_s0.h5','TVnr_DLMIX_20789_gNB_FAPI_s19.h5'};

        LP.SCHED{41}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20750_gNB_FAPI_s0.h5'};
        LP.SCHED{42}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20751_gNB_FAPI_s1.h5'};
        LP.SCHED{43}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20752_gNB_FAPI_s2.h5'};
        LP.SCHED{44}.config{1}.channels = {'TVnr_9385_gNB_FAPI_s0.h5','TVnr_DLMIX_20753_gNB_FAPI_s3.h5','TVnr_ULMIX_21623_gNB_FAPI_s3.h5'};
        LP.SCHED{45}.config{1}.channels = {'TVnr_9387_gNB_FAPI_s0.h5','TVnr_ULMIX_21624_gNB_FAPI_s4.h5'};
        LP.SCHED{46}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_ULMIX_21625_gNB_FAPI_s5.h5'};
        LP.SCHED{47}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20756_gNB_FAPI_s6.h5'};
        LP.SCHED{48}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20757_gNB_FAPI_s7.h5'};
        LP.SCHED{49}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20758_gNB_FAPI_s8.h5'};
        LP.SCHED{50}.config{1}.channels = {'TVnr_9390_gNB_FAPI_s0.h5','TVnr_DLMIX_20759_gNB_FAPI_s9.h5'};
        LP.SCHED{51}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20760_gNB_FAPI_s10.h5'};
        LP.SCHED{52}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20761_gNB_FAPI_s11.h5'};
        LP.SCHED{53}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20762_gNB_FAPI_s12.h5'};
        LP.SCHED{54}.config{1}.channels = {'TVnr_9385_gNB_FAPI_s0.h5','TVnr_DLMIX_20763_gNB_FAPI_s13.h5','TVnr_ULMIX_21633_gNB_FAPI_s13.h5'};
        LP.SCHED{55}.config{1}.channels = {'TVnr_9387_gNB_FAPI_s0.h5','TVnr_ULMIX_21634_gNB_FAPI_s14.h5'};
        LP.SCHED{56}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_ULMIX_21635_gNB_FAPI_s15.h5'};
        LP.SCHED{57}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20766_gNB_FAPI_s16.h5'};
        LP.SCHED{58}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20767_gNB_FAPI_s17.h5'};
        LP.SCHED{59}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20768_gNB_FAPI_s18.h5'};
        LP.SCHED{60}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20769_gNB_FAPI_s19.h5'};
        LP.SCHED{61}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20754_gNB_FAPI_s0.h5'};
        LP.SCHED{62}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20755_gNB_FAPI_s1.h5'};
        LP.SCHED{63}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20772_gNB_FAPI_s2.h5'};
        LP.SCHED{64}.config{1}.channels = {'TVnr_9385_gNB_FAPI_s0.h5','TVnr_DLMIX_20773_gNB_FAPI_s3.h5','TVnr_ULMIX_21636_gNB_FAPI_s3.h5'};
        LP.SCHED{65}.config{1}.channels = {'TVnr_9387_gNB_FAPI_s0.h5','TVnr_ULMIX_21624_gNB_FAPI_s4.h5'};
        LP.SCHED{66}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_ULMIX_21625_gNB_FAPI_s5.h5'};
        LP.SCHED{67}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20776_gNB_FAPI_s6.h5'};
        LP.SCHED{68}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20777_gNB_FAPI_s7.h5'};
        LP.SCHED{69}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20778_gNB_FAPI_s8.h5'};
        LP.SCHED{70}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20779_gNB_FAPI_s9.h5'};
        LP.SCHED{71}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20780_gNB_FAPI_s10.h5'};
        LP.SCHED{72}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20781_gNB_FAPI_s11.h5'};
        LP.SCHED{73}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20782_gNB_FAPI_s12.h5'};
        LP.SCHED{74}.config{1}.channels = {'TVnr_9385_gNB_FAPI_s0.h5','TVnr_DLMIX_20783_gNB_FAPI_s13.h5','TVnr_ULMIX_21637_gNB_FAPI_s13.h5'};
        LP.SCHED{75}.config{1}.channels = {'TVnr_9387_gNB_FAPI_s0.h5','TVnr_ULMIX_21634_gNB_FAPI_s14.h5'};
        LP.SCHED{76}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_ULMIX_21635_gNB_FAPI_s15.h5'};
        LP.SCHED{77}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20786_gNB_FAPI_s16.h5'};
        LP.SCHED{78}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20787_gNB_FAPI_s17.h5'};
        LP.SCHED{79}.config{1}.channels = {'TVnr_9388_gNB_FAPI_s0.h5','TVnr_DLMIX_20788_gNB_FAPI_s18.h5'};
        LP.SCHED{80}.config{1}.channels = {'TVnr_9386_gNB_FAPI_s0.h5','TVnr_DLMIX_20789_gNB_FAPI_s19.h5'}; 

    case 639 % % 64TR test case 'column G' with negative prgSize
        nCell = 1;
        LP = [];
        nSlot = 40;
        LP.Cell_Configs = {'TVnr_DLMIX_20560_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21525_gNB_FAPI_s5.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 1}.config{1}.channels = {'TVnr_9364_gNB_FAPI_s0.h5','TVnr_DLMIX_20560_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{1}.channels = {'TVnr_9364_gNB_FAPI_s0.h5','TVnr_DLMIX_20561_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20562_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_9357_gNB_FAPI_s0.h5','TVnr_DLMIX_20563_gNB_FAPI_s3.h5','TVnr_ULMIX_21523_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{1}.channels = {'TVnr_9355_gNB_FAPI_s0.h5','TVnr_ULMIX_21524_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_ULMIX_21525_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20584_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20585_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20594_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20595_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20570_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20571_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20572_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_9357_gNB_FAPI_s0.h5','TVnr_DLMIX_20573_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_9355_gNB_FAPI_s0.h5','TVnr_ULMIX_21534_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_ULMIX_21535_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20576_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20577_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20578_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20579_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20580_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20581_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20582_gNB_FAPI_s2.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_9357_gNB_FAPI_s0.h5','TVnr_DLMIX_20583_gNB_FAPI_s3.h5','TVnr_ULMIX_21533_gNB_FAPI_s3.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9355_gNB_FAPI_s0.h5','TVnr_ULMIX_21524_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_ULMIX_21525_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20586_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20587_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20588_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20589_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20590_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20591_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20592_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9357_gNB_FAPI_s0.h5','TVnr_DLMIX_20593_gNB_FAPI_s13.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9355_gNB_FAPI_s0.h5','TVnr_ULMIX_21534_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_ULMIX_21535_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20596_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20597_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9356_gNB_FAPI_s0.h5','TVnr_DLMIX_20598_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_9364_gNB_FAPI_s0.h5','TVnr_DLMIX_20599_gNB_FAPI_s19.h5'};
        
    case {640, 641}  % % 64TR test case Ph4 'column B', 80 slot pattern
        nCell = 1;
        LP = [];
        nSlot = 80;
        LP.Cell_Configs = {'TVnr_DLMIX_20790_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21645_gNB_FAPI_s5.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 1}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20790_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20791_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20792_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_9391_gNB_FAPI_s0.h5','TVnr_DLMIX_20793_gNB_FAPI_s3.h5','TVnr_ULMIX_21643_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{1}.channels = {'TVnr_9393_gNB_FAPI_s0.h5','TVnr_ULMIX_21644_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_ULMIX_21645_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20796_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20797_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20798_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_9396_gNB_FAPI_s0.h5','TVnr_DLMIX_20799_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20800_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20801_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20802_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_9391_gNB_FAPI_s0.h5','TVnr_DLMIX_20803_gNB_FAPI_s13.h5','TVnr_ULMIX_21653_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_9395_gNB_FAPI_s0.h5','TVnr_ULMIX_21654_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_ULMIX_21655_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20806_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20807_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20808_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20809_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20810_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20811_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_DLMIX_20812_gNB_FAPI_s2.h5'};  % only CSI-RS and SRS in slot 23 (next)
        if caseNum == 640
            LP.SCHED{24}.config{1}.channels = {'TVnr_9391_gNB_FAPI_s0.h5','TVnr_DLMIX_20813_gNB_FAPI_s3.h5','TVnr_ULMIX_21656_gNB_FAPI_s3.h5'};
        else
            LP.SCHED{24}.config{1}.channels = {'TVnr_9391_gNB_FAPI_s0.h5','TVnr_DLMIX_20805_gNB_FAPI_s3.h5','TVnr_ULMIX_21656_gNB_FAPI_s3.h5'};
        end
        LP.SCHED{25}.config{1}.channels = {'TVnr_9393_gNB_FAPI_s0.h5','TVnr_ULMIX_21644_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_ULMIX_21645_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20816_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20817_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20818_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20819_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20820_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20821_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20822_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9391_gNB_FAPI_s0.h5','TVnr_DLMIX_20823_gNB_FAPI_s13.h5','TVnr_ULMIX_21657_gNB_FAPI_s13.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9395_gNB_FAPI_s0.h5','TVnr_ULMIX_21654_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_ULMIX_21655_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20826_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20827_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20828_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_9392_gNB_FAPI_s0.h5','TVnr_DLMIX_20829_gNB_FAPI_s19.h5'};

        LP.SCHED{41}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20790_gNB_FAPI_s0.h5'};
        LP.SCHED{42}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20791_gNB_FAPI_s1.h5'};
        LP.SCHED{43}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20792_gNB_FAPI_s2.h5'};
        LP.SCHED{44}.config{1}.channels = {'TVnr_9391_gNB_FAPI_s0.h5','TVnr_DLMIX_20793_gNB_FAPI_s3.h5','TVnr_ULMIX_21643_gNB_FAPI_s3.h5'};
        LP.SCHED{45}.config{1}.channels = {'TVnr_9393_gNB_FAPI_s0.h5','TVnr_ULMIX_21644_gNB_FAPI_s4.h5'};
        LP.SCHED{46}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_ULMIX_21645_gNB_FAPI_s5.h5'};
        LP.SCHED{47}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20796_gNB_FAPI_s6.h5'};
        LP.SCHED{48}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20797_gNB_FAPI_s7.h5'};
        LP.SCHED{49}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20798_gNB_FAPI_s8.h5'};
        LP.SCHED{50}.config{1}.channels = {'TVnr_9396_gNB_FAPI_s0.h5','TVnr_DLMIX_20799_gNB_FAPI_s9.h5'};
        LP.SCHED{51}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20800_gNB_FAPI_s10.h5'};
        LP.SCHED{52}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20801_gNB_FAPI_s11.h5'};
        LP.SCHED{53}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20802_gNB_FAPI_s12.h5'};
        LP.SCHED{54}.config{1}.channels = {'TVnr_9391_gNB_FAPI_s0.h5','TVnr_DLMIX_20803_gNB_FAPI_s13.h5','TVnr_ULMIX_21653_gNB_FAPI_s13.h5'};
        LP.SCHED{55}.config{1}.channels = {'TVnr_9395_gNB_FAPI_s0.h5','TVnr_ULMIX_21654_gNB_FAPI_s14.h5'};
        LP.SCHED{56}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_ULMIX_21655_gNB_FAPI_s15.h5'};
        LP.SCHED{57}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20806_gNB_FAPI_s16.h5'};
        LP.SCHED{58}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20807_gNB_FAPI_s17.h5'};
        LP.SCHED{59}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20808_gNB_FAPI_s18.h5'};
        LP.SCHED{60}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20809_gNB_FAPI_s19.h5'};
        LP.SCHED{61}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20794_gNB_FAPI_s0.h5'};  % using 20794 instead of 20810
        LP.SCHED{62}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20795_gNB_FAPI_s1.h5'};  % using 20795 instead of 20811
        LP.SCHED{63}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20812_gNB_FAPI_s2.h5'};
        LP.SCHED{64}.config{1}.channels = {'TVnr_9391_gNB_FAPI_s0.h5','TVnr_DLMIX_20804_gNB_FAPI_s3.h5','TVnr_ULMIX_21656_gNB_FAPI_s3.h5'};  % using 20804 instead of 20813
        LP.SCHED{65}.config{1}.channels = {'TVnr_9393_gNB_FAPI_s0.h5','TVnr_ULMIX_21644_gNB_FAPI_s4.h5'};
        LP.SCHED{66}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_ULMIX_21645_gNB_FAPI_s5.h5'};
        LP.SCHED{67}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20816_gNB_FAPI_s6.h5'};
        LP.SCHED{68}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20817_gNB_FAPI_s7.h5'};
        LP.SCHED{69}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20818_gNB_FAPI_s8.h5'};
        LP.SCHED{70}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20819_gNB_FAPI_s9.h5'};
        LP.SCHED{71}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20820_gNB_FAPI_s10.h5'};
        LP.SCHED{72}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20821_gNB_FAPI_s11.h5'};
        LP.SCHED{73}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20822_gNB_FAPI_s12.h5'};
        LP.SCHED{74}.config{1}.channels = {'TVnr_9391_gNB_FAPI_s0.h5','TVnr_DLMIX_20823_gNB_FAPI_s13.h5','TVnr_ULMIX_21657_gNB_FAPI_s13.h5'};
        LP.SCHED{75}.config{1}.channels = {'TVnr_9395_gNB_FAPI_s0.h5','TVnr_ULMIX_21654_gNB_FAPI_s14.h5'};
        LP.SCHED{76}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_ULMIX_21655_gNB_FAPI_s15.h5'};
        LP.SCHED{77}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20826_gNB_FAPI_s16.h5'};
        LP.SCHED{78}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20827_gNB_FAPI_s17.h5'};
        LP.SCHED{79}.config{1}.channels = {'TVnr_9394_gNB_FAPI_s0.h5','TVnr_DLMIX_20828_gNB_FAPI_s18.h5'};
        LP.SCHED{80}.config{1}.channels = {'TVnr_9392_gNB_FAPI_s0.h5','TVnr_DLMIX_20829_gNB_FAPI_s19.h5'}; 

    case 642  % % 64TR test case 20 slot pattern
            nCell = 1;
            LP = [];
            nSlot = 20;
            LP.Cell_Configs = {'TVnr_DLMIX_20840_gNB_FAPI_s0.h5'};
            %LP.UL_Cell_Configs = {'TVnr_ULMIX_21645_gNB_FAPI_s5.h5'};
            LP.config_static_harq_proc_id = 1; % add for HARQ cases
            LP = init_launchPattern(LP, nSlot, nCell);
            LP.SCHED{ 1}.config{1}.channels = {'TVnr_DLMIX_20840_gNB_FAPI_s0.h5'};
            LP.SCHED{20}.config{1}.channels = {'TVnr_9400_gNB_FAPI_s0.h5'};

    case 643  % % 64TR test case 20 slot pattern
        nCell = 1;
        LP = [];
        nSlot = 20;
        LP.Cell_Configs = {'TVnr_DLMIX_20860_gNB_FAPI_s0.h5'};
        %LP.UL_Cell_Configs = {'TVnr_ULMIX_21645_gNB_FAPI_s5.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 1}.config{1}.channels = {'TVnr_DLMIX_20860_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{1}.channels = {'TVnr_DLMIX_20861_gNB_FAPI_s1.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9398_gNB_FAPI_s0.h5'};
    
    case {644,645,646,647}  % % 64TR test case 25-3 'column B', 80 slot pattern
    % 644: 4 SRS symbols in S slot, 8-port CSI_RS + SRS
    % 645: 2 SRS symbols in S slot, 1 SRS symbol in each U slot, 8-port CSI_RS + SRS
    % 646: 4 SRS symbols in S slot, 8-port + 32-port CSI_RS + SRS
    % 647: 2 SRS symbols in S slot, 1 SRS symbol in each U slot, 8-port + 32-port CSI_RS + SRS
        nCell = 1;
        LP = [];
        nSlot = 80;
        LP.Cell_Configs = {'TVnr_DLMIX_20870_gNB_FAPI_s0.h5'};
        if caseNum == 644 || caseNum == 646
            LP.UL_Cell_Configs = {'TVnr_ULMIX_21675_gNB_FAPI_s5.h5'};
            bfwTvNames = {
                'TVnr_9401_gNB_FAPI_s0.h5',
                'TVnr_9402_gNB_FAPI_s0.h5',
                'TVnr_9403_gNB_FAPI_s0.h5',
                'TVnr_9404_gNB_FAPI_s0.h5',
                'TVnr_9405_gNB_FAPI_s0.h5',
                'TVnr_9406_gNB_FAPI_s0.h5'
            };
        else
            LP.UL_Cell_Configs = {'TVnr_ULMIX_21695_gNB_FAPI_s5.h5'};
            bfwTvNames = {
                'TVnr_9407_gNB_FAPI_s0.h5',
                'TVnr_9408_gNB_FAPI_s0.h5',
                'TVnr_9409_gNB_FAPI_s0.h5',
                'TVnr_9410_gNB_FAPI_s0.h5',
                'TVnr_9411_gNB_FAPI_s0.h5',
                'TVnr_9412_gNB_FAPI_s0.h5'
            };
        end
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        for twoSubFrameIdx = 0:1  % per 40 slots
            LP.SCHED{twoSubFrameIdx * 40 +  1}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20870_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  2}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20871_gNB_FAPI_s1.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  3}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20872_gNB_FAPI_s2.h5'};
            % S/U/U configured below
            LP.SCHED{twoSubFrameIdx * 40 +  7}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20876_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  8}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20877_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  9}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20878_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 10}.config{1}.channels = {bfwTvNames{6},'TVnr_DLMIX_20879_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 11}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20880_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 12}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20881_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 13}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20882_gNB_FAPI_s12.h5'};
            % S/U/U configured below
            LP.SCHED{twoSubFrameIdx * 40 + 17}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20886_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 18}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20887_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 19}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20888_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 20}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20889_gNB_FAPI_s19.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 21}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20890_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 22}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20891_gNB_FAPI_s1.h5'};
            if twoSubFrameIdx == 0
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {'TVnr_DLMIX_20892_gNB_FAPI_s2.h5'};  % only CSI-RS and SRS in slot 23 (next)
            else
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20892_gNB_FAPI_s2.h5'};  % has PDSCH in slot 63 (next)
            end
            % S/U/U configured below
            LP.SCHED{twoSubFrameIdx * 40 + 27}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20896_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 28}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20897_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 29}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20898_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 30}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20899_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 31}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20900_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 32}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20901_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 33}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20902_gNB_FAPI_s12.h5'};
            % S/U/U configured below
            LP.SCHED{twoSubFrameIdx * 40 + 37}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20906_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 38}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20907_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 39}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20908_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 40}.config{1}.channels = {bfwTvNames{2},'TVnr_DLMIX_20909_gNB_FAPI_s19.h5'};   
        end

        % S/U/U config
        if caseNum == 644 || caseNum == 646
            for subframeIdx = 0:3
                dlmixTvSlot3 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s3.h5', 20873 + mod(subframeIdx, 2) * 20);
                if subframeIdx == 1  % slot 23 with CSI_RS + SRS
                    if caseNum == 644
                        dlmixTvSlot3 = 'TVnr_DLMIX_20893_gNB_FAPI_s3.h5';
                    else
                        dlmixTvSlot3 = 'TVnr_DLMIX_20894_gNB_FAPI_s3.h5';
                    end
                end
                LP.SCHED{subframeIdx * 20 + 4}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot3,'TVnr_ULMIX_21673_gNB_FAPI_s3.h5'};
                LP.SCHED{subframeIdx * 20 + 5}.config{1}.channels = {bfwTvNames{3},'TVnr_ULMIX_21674_gNB_FAPI_s4.h5'};
                LP.SCHED{subframeIdx * 20 + 6}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21675_gNB_FAPI_s5.h5'};
                dlmixTvSlot13 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s13.h5', 20883 + mod(subframeIdx, 2) * 20);
                LP.SCHED{subframeIdx * 20 + 14}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot13,'TVnr_ULMIX_21683_gNB_FAPI_s13.h5'};
                LP.SCHED{subframeIdx * 20 + 15}.config{1}.channels = {bfwTvNames{5},'TVnr_ULMIX_21684_gNB_FAPI_s14.h5'};
                LP.SCHED{subframeIdx * 20 + 16}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21685_gNB_FAPI_s15.h5'};
            end
        else
            for subframeIdx = 0:3
                dlmixTvSlot3 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s3.h5', 20873 + mod(subframeIdx, 2) * 20);
                if subframeIdx == 1  % slot 23 with CSI_RS + SRS
                    if caseNum == 645
                        dlmixTvSlot3 = 'TVnr_DLMIX_20893_gNB_FAPI_s3.h5';
                    else
                        dlmixTvSlot3 = 'TVnr_DLMIX_20894_gNB_FAPI_s3.h5';
                    end
                end
                LP.SCHED{subframeIdx * 20 + 4}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot3,'TVnr_ULMIX_21693_gNB_FAPI_s3.h5'};
                LP.SCHED{subframeIdx * 20 + 5}.config{1}.channels = {bfwTvNames{3},'TVnr_ULMIX_21694_gNB_FAPI_s4.h5'};
                LP.SCHED{subframeIdx * 20 + 6}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21695_gNB_FAPI_s5.h5'};
                dlmixTvSlot13 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s13.h5', 20883 + mod(subframeIdx, 2) * 20);
                LP.SCHED{subframeIdx * 20 + 14}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot13,'TVnr_ULMIX_21703_gNB_FAPI_s13.h5'};
                LP.SCHED{subframeIdx * 20 + 15}.config{1}.channels = {bfwTvNames{5},'TVnr_ULMIX_21704_gNB_FAPI_s14.h5'};
                LP.SCHED{subframeIdx * 20 + 16}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21705_gNB_FAPI_s15.h5'};
            end
        end

        % irregular slots 60,61,63
        LP.SCHED{61}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20874_gNB_FAPI_s0.h5'};  % using 20874 instead of 20890
        LP.SCHED{62}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20875_gNB_FAPI_s1.h5'};  % using 20875 instead of 20891
        if caseNum == 644 || caseNum == 646
            LP.SCHED{64}.config{1}.channels = {bfwTvNames{1},'TVnr_DLMIX_20884_gNB_FAPI_s3.h5','TVnr_ULMIX_21673_gNB_FAPI_s3.h5'};  % using 20884 instead of 20893
        else
            LP.SCHED{64}.config{1}.channels = {bfwTvNames{1},'TVnr_DLMIX_20884_gNB_FAPI_s3.h5','TVnr_ULMIX_21693_gNB_FAPI_s3.h5'};  % using 20884 instead of 20893
        end

    case 648  % % 64TR test case 25-3 'column E', 80 slot pattern, 32DL
    % 648: 4 SRS symbols in S slot
        nCell = 1;
        LP = [];
        nSlot = 80;
        LP.Cell_Configs = {'TVnr_DLMIX_20950_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21755_gNB_FAPI_s5.h5'};
        bfwTvNames = {
                'TVnr_9429_gNB_FAPI_s0.h5',
                'TVnr_9430_gNB_FAPI_s0.h5',
                'TVnr_9431_gNB_FAPI_s0.h5',
                'TVnr_9432_gNB_FAPI_s0.h5',
                'TVnr_9433_gNB_FAPI_s0.h5',
                'TVnr_9434_gNB_FAPI_s0.h5',
                'TVnr_9435_gNB_FAPI_s0.h5'
            };
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        for twoSubFrameIdx = 0:1  % per 40 slots
            LP.SCHED{twoSubFrameIdx * 40 +  1}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20950_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  2}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20951_gNB_FAPI_s1.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  3}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20952_gNB_FAPI_s2.h5'};
            % S/U/U configured below
            LP.SCHED{twoSubFrameIdx * 40 +  7}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20956_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  8}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20957_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  9}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20958_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 10}.config{1}.channels = {bfwTvNames{6},'TVnr_DLMIX_20959_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 11}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20960_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 12}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20961_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 13}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20962_gNB_FAPI_s12.h5'};
            % S/U/U configured below
            LP.SCHED{twoSubFrameIdx * 40 + 17}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20966_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 18}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20967_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 19}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20968_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 20}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20969_gNB_FAPI_s19.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 21}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20970_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 22}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20971_gNB_FAPI_s1.h5'};
            if twoSubFrameIdx == 0
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {'TVnr_DLMIX_20972_gNB_FAPI_s2.h5'};  % only CSI-RS and SRS in slot 23 (next)
            else
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20972_gNB_FAPI_s2.h5'};  % has PDSCH in slot 63 (next)
            end
            % S/U/U configured below
            LP.SCHED{twoSubFrameIdx * 40 + 27}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20976_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 28}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20977_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 29}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20978_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 30}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20979_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 31}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20980_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 32}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20981_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 33}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20982_gNB_FAPI_s12.h5'};
            % S/U/U configured below
            LP.SCHED{twoSubFrameIdx * 40 + 37}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20986_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 38}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20987_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 39}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20988_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 40}.config{1}.channels = {bfwTvNames{2},'TVnr_DLMIX_20989_gNB_FAPI_s19.h5'};   
        end

        % S/U/U config
        for subframeIdx = 0:3
            dlmixTvSlot3 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s3.h5', 20953 + mod(subframeIdx, 2) * 20);
            LP.SCHED{subframeIdx * 20 + 4}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot3,'TVnr_ULMIX_21673_gNB_FAPI_s3.h5'};
            LP.SCHED{subframeIdx * 20 + 5}.config{1}.channels = {bfwTvNames{3},'TVnr_ULMIX_21754_gNB_FAPI_s4.h5'};
            LP.SCHED{subframeIdx * 20 + 6}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21755_gNB_FAPI_s5.h5'};
            dlmixTvSlot13 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s13.h5', 20963 + mod(subframeIdx, 2) * 20);
            LP.SCHED{subframeIdx * 20 + 14}.config{1}.channels = {bfwTvNames{5},dlmixTvSlot13,'TVnr_ULMIX_21683_gNB_FAPI_s13.h5'};
            LP.SCHED{subframeIdx * 20 + 15}.config{1}.channels = {bfwTvNames{7},'TVnr_ULMIX_21764_gNB_FAPI_s14.h5'};
            LP.SCHED{subframeIdx * 20 + 16}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21765_gNB_FAPI_s15.h5'};
        end

        % irregular slots 60,61,63
        LP.SCHED{61}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20954_gNB_FAPI_s0.h5'};  % using 20954 instead of 20970
        LP.SCHED{62}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20955_gNB_FAPI_s1.h5'};  % using 20955 instead of 20971
        LP.SCHED{64}.config{1}.channels = {bfwTvNames{1},'TVnr_DLMIX_20964_gNB_FAPI_s3.h5','TVnr_ULMIX_21673_gNB_FAPI_s3.h5'};  % using 20964 instead of 20973
            
    case {649,650,651,652}  % % 64TR test case 25-3 'column G', 80 slot pattern, 64 UEs per TTI
    % 649: 4 SRS symbols in S slot
    % 650: 2 SRS symbols in S slot, 1 SRS symbol in each U slot
    % 651: 4 SRS symbols in S slot, 8-port + 32-port CSI_RS + SRS
    % 652: 2 SRS symbols in S slot, 1 SRS symbol in each U slot, 8-port + 32-port CSI_RS + SRS
        nCell = 1;
        LP = [];
        nSlot = 80;
        LP.Cell_Configs = {'TVnr_DLMIX_20910_gNB_FAPI_s0.h5'};
        if caseNum == 649 || caseNum == 651
            LP.UL_Cell_Configs = {'TVnr_ULMIX_21715_gNB_FAPI_s5.h5'};
            bfwTvNames = {
                'TVnr_9413_gNB_FAPI_s0.h5',
                'TVnr_9414_gNB_FAPI_s0.h5',
                'TVnr_9415_gNB_FAPI_s0.h5',
                'TVnr_9416_gNB_FAPI_s0.h5',
                'TVnr_9417_gNB_FAPI_s0.h5',
                'TVnr_9418_gNB_FAPI_s0.h5',
                'TVnr_9419_gNB_FAPI_s0.h5'
            };
        else
            LP.UL_Cell_Configs = {'TVnr_ULMIX_21735_gNB_FAPI_s5.h5'};
            bfwTvNames = {
                'TVnr_9421_gNB_FAPI_s0.h5',
                'TVnr_9422_gNB_FAPI_s0.h5',
                'TVnr_9423_gNB_FAPI_s0.h5',
                'TVnr_9424_gNB_FAPI_s0.h5',
                'TVnr_9425_gNB_FAPI_s0.h5',
                'TVnr_9426_gNB_FAPI_s0.h5',
                'TVnr_9427_gNB_FAPI_s0.h5'
            };
        end
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        for twoSubFrameIdx = 0:1  % per 40 slots
            LP.SCHED{twoSubFrameIdx * 40 +  1}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20910_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  2}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20911_gNB_FAPI_s1.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  3}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20912_gNB_FAPI_s2.h5'};
            % S/U/U configured below
            LP.SCHED{twoSubFrameIdx * 40 +  7}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20916_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  8}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20917_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  9}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20918_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 10}.config{1}.channels = {bfwTvNames{6},'TVnr_DLMIX_20919_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 11}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20920_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 12}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20921_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 13}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20922_gNB_FAPI_s12.h5'};
            % S/U/U configured below
            LP.SCHED{twoSubFrameIdx * 40 + 17}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20926_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 18}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20927_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 19}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20928_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 20}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20929_gNB_FAPI_s19.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 21}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20930_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 22}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20931_gNB_FAPI_s1.h5'};
            if twoSubFrameIdx == 0
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {'TVnr_DLMIX_20932_gNB_FAPI_s2.h5'};  % only CSI-RS and SRS in slot 23 (next)
            else
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20932_gNB_FAPI_s2.h5'};  % has PDSCH in slot 63 (next)
            end
            % S/U/U configured below
            LP.SCHED{twoSubFrameIdx * 40 + 27}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20936_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 28}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20937_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 29}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20938_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 30}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20939_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 31}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20940_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 32}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20941_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 33}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20942_gNB_FAPI_s12.h5'};
            % S/U/U configured below
            LP.SCHED{twoSubFrameIdx * 40 + 37}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20946_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 38}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20947_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 39}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20948_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 40}.config{1}.channels = {bfwTvNames{2},'TVnr_DLMIX_20949_gNB_FAPI_s19.h5'};   
        end

        % S/U/U config
        if caseNum == 649 || caseNum == 651
            for subframeIdx = 0:3
                dlmixTvSlot3 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s3.h5', 20913 + mod(subframeIdx, 2) * 20);
                if subframeIdx == 1  % slot 23 with CSI_RS + SRS
                    if caseNum == 649
                        dlmixTvSlot3 = 'TVnr_DLMIX_20933_gNB_FAPI_s3.h5';
                    else
                        dlmixTvSlot3 = 'TVnr_DLMIX_20934_gNB_FAPI_s3.h5';
                    end
                end
                LP.SCHED{subframeIdx * 20 + 4}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot3,'TVnr_ULMIX_21713_gNB_FAPI_s3.h5'};
                LP.SCHED{subframeIdx * 20 + 5}.config{1}.channels = {bfwTvNames{3},'TVnr_ULMIX_21714_gNB_FAPI_s4.h5'};
                LP.SCHED{subframeIdx * 20 + 6}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21715_gNB_FAPI_s5.h5'};
                dlmixTvSlot13 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s13.h5', 20923 + mod(subframeIdx, 2) * 20);
                LP.SCHED{subframeIdx * 20 + 14}.config{1}.channels = {bfwTvNames{5},dlmixTvSlot13,'TVnr_ULMIX_21723_gNB_FAPI_s13.h5'};
                LP.SCHED{subframeIdx * 20 + 15}.config{1}.channels = {bfwTvNames{7},'TVnr_ULMIX_21724_gNB_FAPI_s14.h5'};
                LP.SCHED{subframeIdx * 20 + 16}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21725_gNB_FAPI_s15.h5'};
            end
        else
            for subframeIdx = 0:3
                dlmixTvSlot3 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s3.h5', 20913 + mod(subframeIdx, 2) * 20);
                if subframeIdx == 1  % slot 23 with CSI_RS + SRS
                    if caseNum == 650
                        dlmixTvSlot3 = 'TVnr_DLMIX_20933_gNB_FAPI_s3.h5';
                    else
                        dlmixTvSlot3 = 'TVnr_DLMIX_20934_gNB_FAPI_s3.h5';
                    end
                end
                LP.SCHED{subframeIdx * 20 + 4}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot3,'TVnr_ULMIX_21733_gNB_FAPI_s3.h5'};
                LP.SCHED{subframeIdx * 20 + 5}.config{1}.channels = {bfwTvNames{3},'TVnr_ULMIX_21734_gNB_FAPI_s4.h5'};
                LP.SCHED{subframeIdx * 20 + 6}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21735_gNB_FAPI_s5.h5'};
                dlmixTvSlot13 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s13.h5', 20923 + mod(subframeIdx, 2) * 20);
                LP.SCHED{subframeIdx * 20 + 14}.config{1}.channels = {bfwTvNames{5},dlmixTvSlot13,'TVnr_ULMIX_21743_gNB_FAPI_s13.h5'};
                LP.SCHED{subframeIdx * 20 + 15}.config{1}.channels = {bfwTvNames{7},'TVnr_ULMIX_21744_gNB_FAPI_s14.h5'};
                LP.SCHED{subframeIdx * 20 + 16}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21745_gNB_FAPI_s15.h5'};
            end
        end

        % irregular slots 60,61,63
        LP.SCHED{61}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20914_gNB_FAPI_s0.h5'};  % using 20874 instead of 20930
        LP.SCHED{62}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20915_gNB_FAPI_s1.h5'};  % using 20875 instead of 20931
        if caseNum == 649 || caseNum == 651
            LP.SCHED{64}.config{1}.channels = {bfwTvNames{1},'TVnr_DLMIX_20924_gNB_FAPI_s3.h5','TVnr_ULMIX_21713_gNB_FAPI_s3.h5'};  % using 20924 instead of 20933 or 20934
        else
            LP.SCHED{64}.config{1}.channels = {bfwTvNames{1},'TVnr_DLMIX_20924_gNB_FAPI_s3.h5','TVnr_ULMIX_21733_gNB_FAPI_s3.h5'};  % using 20924 instead of 20933 or 20934
        end

    case 653  % % 64TR test case Ph4 'column B', srsPrgSize = 4 and bfwPrgSize = 16, 80 slot pattern
        nCell = 1;
        LP = [];
        nSlot = 80;
        LP.Cell_Configs = {'TVnr_DLMIX_20990_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21775_gNB_FAPI_s5.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 1}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_20990_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_20991_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_20992_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_9437_gNB_FAPI_s0.h5','TVnr_DLMIX_20993_gNB_FAPI_s3.h5','TVnr_ULMIX_21773_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{1}.channels = {'TVnr_9439_gNB_FAPI_s0.h5','TVnr_ULMIX_21774_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_ULMIX_21775_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_20996_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_20997_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_20998_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_9442_gNB_FAPI_s0.h5','TVnr_DLMIX_20999_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21000_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21001_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21002_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_9437_gNB_FAPI_s0.h5','TVnr_DLMIX_21003_gNB_FAPI_s13.h5','TVnr_ULMIX_21783_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_9441_gNB_FAPI_s0.h5','TVnr_ULMIX_21784_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_ULMIX_21785_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21006_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21007_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21008_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21009_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21010_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21011_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_DLMIX_21012_gNB_FAPI_s2.h5'};  % only CSI-RS and SRS in slot 23 (next)
        LP.SCHED{24}.config{1}.channels = {'TVnr_9437_gNB_FAPI_s0.h5','TVnr_DLMIX_21013_gNB_FAPI_s3.h5','TVnr_ULMIX_21786_gNB_FAPI_s3.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9439_gNB_FAPI_s0.h5','TVnr_ULMIX_21774_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_ULMIX_21775_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21016_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21017_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21018_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21019_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21020_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21021_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21022_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9437_gNB_FAPI_s0.h5','TVnr_DLMIX_21023_gNB_FAPI_s13.h5','TVnr_ULMIX_21787_gNB_FAPI_s13.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9441_gNB_FAPI_s0.h5','TVnr_ULMIX_21784_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_ULMIX_21785_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21026_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21027_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21028_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_9438_gNB_FAPI_s0.h5','TVnr_DLMIX_21029_gNB_FAPI_s19.h5'};

        LP.SCHED{41}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_20990_gNB_FAPI_s0.h5'};
        LP.SCHED{42}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_20991_gNB_FAPI_s1.h5'};
        LP.SCHED{43}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_20992_gNB_FAPI_s2.h5'};
        LP.SCHED{44}.config{1}.channels = {'TVnr_9437_gNB_FAPI_s0.h5','TVnr_DLMIX_20993_gNB_FAPI_s3.h5','TVnr_ULMIX_21773_gNB_FAPI_s3.h5'};
        LP.SCHED{45}.config{1}.channels = {'TVnr_9439_gNB_FAPI_s0.h5','TVnr_ULMIX_21774_gNB_FAPI_s4.h5'};
        LP.SCHED{46}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_ULMIX_21775_gNB_FAPI_s5.h5'};
        LP.SCHED{47}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_20996_gNB_FAPI_s6.h5'};
        LP.SCHED{48}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_20997_gNB_FAPI_s7.h5'};
        LP.SCHED{49}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_20998_gNB_FAPI_s8.h5'};
        LP.SCHED{50}.config{1}.channels = {'TVnr_9442_gNB_FAPI_s0.h5','TVnr_DLMIX_20999_gNB_FAPI_s9.h5'};
        LP.SCHED{51}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21000_gNB_FAPI_s10.h5'};
        LP.SCHED{52}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21001_gNB_FAPI_s11.h5'};
        LP.SCHED{53}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21002_gNB_FAPI_s12.h5'};
        LP.SCHED{54}.config{1}.channels = {'TVnr_9437_gNB_FAPI_s0.h5','TVnr_DLMIX_21003_gNB_FAPI_s13.h5','TVnr_ULMIX_21783_gNB_FAPI_s13.h5'};
        LP.SCHED{55}.config{1}.channels = {'TVnr_9441_gNB_FAPI_s0.h5','TVnr_ULMIX_21784_gNB_FAPI_s14.h5'};
        LP.SCHED{56}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_ULMIX_21785_gNB_FAPI_s15.h5'};
        LP.SCHED{57}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21006_gNB_FAPI_s16.h5'};
        LP.SCHED{58}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21007_gNB_FAPI_s17.h5'};
        LP.SCHED{59}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21008_gNB_FAPI_s18.h5'};
        LP.SCHED{60}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21009_gNB_FAPI_s19.h5'};
        LP.SCHED{61}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_20994_gNB_FAPI_s0.h5'};  % using 20994 instead of 21010
        LP.SCHED{62}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_20995_gNB_FAPI_s1.h5'};  % using 20995 instead of 21011
        LP.SCHED{63}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21012_gNB_FAPI_s2.h5'};
        LP.SCHED{64}.config{1}.channels = {'TVnr_9437_gNB_FAPI_s0.h5','TVnr_DLMIX_21004_gNB_FAPI_s3.h5','TVnr_ULMIX_21786_gNB_FAPI_s3.h5'};  % using 21004 instead of 21013
        LP.SCHED{65}.config{1}.channels = {'TVnr_9439_gNB_FAPI_s0.h5','TVnr_ULMIX_21774_gNB_FAPI_s4.h5'};
        LP.SCHED{66}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_ULMIX_21775_gNB_FAPI_s5.h5'};
        LP.SCHED{67}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21016_gNB_FAPI_s6.h5'};
        LP.SCHED{68}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21017_gNB_FAPI_s7.h5'};
        LP.SCHED{69}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21018_gNB_FAPI_s8.h5'};
        LP.SCHED{70}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21019_gNB_FAPI_s9.h5'};
        LP.SCHED{71}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21020_gNB_FAPI_s10.h5'};
        LP.SCHED{72}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21021_gNB_FAPI_s11.h5'};
        LP.SCHED{73}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21022_gNB_FAPI_s12.h5'};
        LP.SCHED{74}.config{1}.channels = {'TVnr_9437_gNB_FAPI_s0.h5','TVnr_DLMIX_21023_gNB_FAPI_s13.h5','TVnr_ULMIX_21787_gNB_FAPI_s13.h5'};
        LP.SCHED{75}.config{1}.channels = {'TVnr_9441_gNB_FAPI_s0.h5','TVnr_ULMIX_21784_gNB_FAPI_s14.h5'};
        LP.SCHED{76}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_ULMIX_21785_gNB_FAPI_s15.h5'};
        LP.SCHED{77}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21026_gNB_FAPI_s16.h5'};
        LP.SCHED{78}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21027_gNB_FAPI_s17.h5'};
        LP.SCHED{79}.config{1}.channels = {'TVnr_9440_gNB_FAPI_s0.h5','TVnr_DLMIX_21028_gNB_FAPI_s18.h5'};
        LP.SCHED{80}.config{1}.channels = {'TVnr_9438_gNB_FAPI_s0.h5','TVnr_DLMIX_21029_gNB_FAPI_s19.h5'};
    

    case 654  % % 64TR test case Ph4 'column B', srsPrgSize = 2 and bfwPrgSize = 16, 80 slot pattern
        nCell = 1;
        LP = [];
        nSlot = 80;
        LP.Cell_Configs = {'TVnr_DLMIX_20990_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21775_gNB_FAPI_s5.h5'};
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{ 1}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_20990_gNB_FAPI_s0.h5'};
        LP.SCHED{ 2}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_20991_gNB_FAPI_s1.h5'};
        LP.SCHED{ 3}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_20992_gNB_FAPI_s2.h5'};
        LP.SCHED{ 4}.config{1}.channels = {'TVnr_9443_gNB_FAPI_s0.h5','TVnr_DLMIX_20993_gNB_FAPI_s3.h5','TVnr_ULMIX_21772_gNB_FAPI_s3.h5'};
        LP.SCHED{ 5}.config{1}.channels = {'TVnr_9445_gNB_FAPI_s0.h5','TVnr_ULMIX_21774_gNB_FAPI_s4.h5'};
        LP.SCHED{ 6}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_ULMIX_21775_gNB_FAPI_s5.h5'};
        LP.SCHED{ 7}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_20996_gNB_FAPI_s6.h5'};
        LP.SCHED{ 8}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_20997_gNB_FAPI_s7.h5'};
        LP.SCHED{ 9}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_20998_gNB_FAPI_s8.h5'};
        LP.SCHED{10}.config{1}.channels = {'TVnr_9448_gNB_FAPI_s0.h5','TVnr_DLMIX_20999_gNB_FAPI_s9.h5'};
        LP.SCHED{11}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21000_gNB_FAPI_s10.h5'};
        LP.SCHED{12}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21001_gNB_FAPI_s11.h5'};
        LP.SCHED{13}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21002_gNB_FAPI_s12.h5'};
        LP.SCHED{14}.config{1}.channels = {'TVnr_9443_gNB_FAPI_s0.h5','TVnr_DLMIX_21003_gNB_FAPI_s13.h5','TVnr_ULMIX_21782_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_9447_gNB_FAPI_s0.h5','TVnr_ULMIX_21784_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_ULMIX_21785_gNB_FAPI_s15.h5'};
        LP.SCHED{17}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21006_gNB_FAPI_s16.h5'};
        LP.SCHED{18}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21007_gNB_FAPI_s17.h5'};
        LP.SCHED{19}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21008_gNB_FAPI_s18.h5'};
        LP.SCHED{20}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21009_gNB_FAPI_s19.h5'};
        LP.SCHED{21}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21010_gNB_FAPI_s0.h5'};
        LP.SCHED{22}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21011_gNB_FAPI_s1.h5'};
        LP.SCHED{23}.config{1}.channels = {'TVnr_DLMIX_21012_gNB_FAPI_s2.h5'};  % only CSI-RS and SRS in slot 23 (next)
        LP.SCHED{24}.config{1}.channels = {'TVnr_9443_gNB_FAPI_s0.h5','TVnr_DLMIX_21013_gNB_FAPI_s3.h5','TVnr_ULMIX_21788_gNB_FAPI_s3.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_9445_gNB_FAPI_s0.h5','TVnr_ULMIX_21774_gNB_FAPI_s4.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_ULMIX_21775_gNB_FAPI_s5.h5'};
        LP.SCHED{27}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21016_gNB_FAPI_s6.h5'};
        LP.SCHED{28}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21017_gNB_FAPI_s7.h5'};
        LP.SCHED{29}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21018_gNB_FAPI_s8.h5'};
        LP.SCHED{30}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21019_gNB_FAPI_s9.h5'};
        LP.SCHED{31}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21020_gNB_FAPI_s10.h5'};
        LP.SCHED{32}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21021_gNB_FAPI_s11.h5'};
        LP.SCHED{33}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21022_gNB_FAPI_s12.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_9443_gNB_FAPI_s0.h5','TVnr_DLMIX_21023_gNB_FAPI_s13.h5','TVnr_ULMIX_21789_gNB_FAPI_s13.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_9447_gNB_FAPI_s0.h5','TVnr_ULMIX_21784_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_ULMIX_21785_gNB_FAPI_s15.h5'};
        LP.SCHED{37}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21026_gNB_FAPI_s16.h5'};
        LP.SCHED{38}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21027_gNB_FAPI_s17.h5'};
        LP.SCHED{39}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21028_gNB_FAPI_s18.h5'};
        LP.SCHED{40}.config{1}.channels = {'TVnr_9444_gNB_FAPI_s0.h5','TVnr_DLMIX_21029_gNB_FAPI_s19.h5'};

        LP.SCHED{41}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_20990_gNB_FAPI_s0.h5'};
        LP.SCHED{42}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_20991_gNB_FAPI_s1.h5'};
        LP.SCHED{43}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_20992_gNB_FAPI_s2.h5'};
        LP.SCHED{44}.config{1}.channels = {'TVnr_9443_gNB_FAPI_s0.h5','TVnr_DLMIX_20993_gNB_FAPI_s3.h5','TVnr_ULMIX_21772_gNB_FAPI_s3.h5'};
        LP.SCHED{45}.config{1}.channels = {'TVnr_9445_gNB_FAPI_s0.h5','TVnr_ULMIX_21774_gNB_FAPI_s4.h5'};
        LP.SCHED{46}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_ULMIX_21775_gNB_FAPI_s5.h5'};
        LP.SCHED{47}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_20996_gNB_FAPI_s6.h5'};
        LP.SCHED{48}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_20997_gNB_FAPI_s7.h5'};
        LP.SCHED{49}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_20998_gNB_FAPI_s8.h5'};
        LP.SCHED{50}.config{1}.channels = {'TVnr_9448_gNB_FAPI_s0.h5','TVnr_DLMIX_20999_gNB_FAPI_s9.h5'};
        LP.SCHED{51}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21000_gNB_FAPI_s10.h5'};
        LP.SCHED{52}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21001_gNB_FAPI_s11.h5'};
        LP.SCHED{53}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21002_gNB_FAPI_s12.h5'};
        LP.SCHED{54}.config{1}.channels = {'TVnr_9443_gNB_FAPI_s0.h5','TVnr_DLMIX_21003_gNB_FAPI_s13.h5','TVnr_ULMIX_21782_gNB_FAPI_s13.h5'};
        LP.SCHED{55}.config{1}.channels = {'TVnr_9447_gNB_FAPI_s0.h5','TVnr_ULMIX_21784_gNB_FAPI_s14.h5'};
        LP.SCHED{56}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_ULMIX_21785_gNB_FAPI_s15.h5'};
        LP.SCHED{57}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21006_gNB_FAPI_s16.h5'};
        LP.SCHED{58}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21007_gNB_FAPI_s17.h5'};
        LP.SCHED{59}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21008_gNB_FAPI_s18.h5'};
        LP.SCHED{60}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21009_gNB_FAPI_s19.h5'};
        LP.SCHED{61}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_20994_gNB_FAPI_s0.h5'};  % using 20994 instead of 21010
        LP.SCHED{62}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_20995_gNB_FAPI_s1.h5'};  % using 20995 instead of 21011
        LP.SCHED{63}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21012_gNB_FAPI_s2.h5'};
        LP.SCHED{64}.config{1}.channels = {'TVnr_9443_gNB_FAPI_s0.h5','TVnr_DLMIX_21004_gNB_FAPI_s3.h5','TVnr_ULMIX_21788_gNB_FAPI_s3.h5'};  % using 21004 instead of 21013
        LP.SCHED{65}.config{1}.channels = {'TVnr_9445_gNB_FAPI_s0.h5','TVnr_ULMIX_21774_gNB_FAPI_s4.h5'};
        LP.SCHED{66}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_ULMIX_21775_gNB_FAPI_s5.h5'};
        LP.SCHED{67}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21016_gNB_FAPI_s6.h5'};
        LP.SCHED{68}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21017_gNB_FAPI_s7.h5'};
        LP.SCHED{69}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21018_gNB_FAPI_s8.h5'};
        LP.SCHED{70}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21019_gNB_FAPI_s9.h5'};
        LP.SCHED{71}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21020_gNB_FAPI_s10.h5'};
        LP.SCHED{72}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21021_gNB_FAPI_s11.h5'};
        LP.SCHED{73}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21022_gNB_FAPI_s12.h5'};
        LP.SCHED{74}.config{1}.channels = {'TVnr_9443_gNB_FAPI_s0.h5','TVnr_DLMIX_21023_gNB_FAPI_s13.h5','TVnr_ULMIX_21789_gNB_FAPI_s13.h5'};
        LP.SCHED{75}.config{1}.channels = {'TVnr_9447_gNB_FAPI_s0.h5','TVnr_ULMIX_21784_gNB_FAPI_s14.h5'};
        LP.SCHED{76}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_ULMIX_21785_gNB_FAPI_s15.h5'};
        LP.SCHED{77}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21026_gNB_FAPI_s16.h5'};
        LP.SCHED{78}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21027_gNB_FAPI_s17.h5'};
        LP.SCHED{79}.config{1}.channels = {'TVnr_9446_gNB_FAPI_s0.h5','TVnr_DLMIX_21028_gNB_FAPI_s18.h5'};
        LP.SCHED{80}.config{1}.channels = {'TVnr_9444_gNB_FAPI_s0.h5','TVnr_DLMIX_21029_gNB_FAPI_s19.h5'};

    case 655  % 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 100 MHz heavy
        nCell = 1;
        LP = [];
        nSlot = 80;
        LP.Cell_Configs = {'TVnr_DLMIX_21030_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21795_gNB_FAPI_s5.h5'};
        bfwTvNames = {
            'TVnr_9449_gNB_FAPI_s0.h5',
            'TVnr_9450_gNB_FAPI_s0.h5',
            'TVnr_9451_gNB_FAPI_s0.h5',
            'TVnr_9452_gNB_FAPI_s0.h5',
            'TVnr_9453_gNB_FAPI_s0.h5',
            'TVnr_9454_gNB_FAPI_s0.h5'
        };
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        for twoSubFrameIdx = 0:1  % per 40 slots
            % First 20 slots (frame 0)
            LP.SCHED{twoSubFrameIdx * 40 +  1}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21030_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  2}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21031_gNB_FAPI_s1.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  3}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21032_gNB_FAPI_s2.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  7}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21036_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  8}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21037_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  9}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21038_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 10}.config{1}.channels = {bfwTvNames{6},'TVnr_DLMIX_21039_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 11}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21040_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 12}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21041_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 13}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21042_gNB_FAPI_s12.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 17}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21046_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 18}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21047_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 19}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21048_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 20}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21049_gNB_FAPI_s19.h5'};
            % Second 20 slots (frame 1)
            LP.SCHED{twoSubFrameIdx * 40 + 21}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21050_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 22}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21051_gNB_FAPI_s1.h5'};
            if twoSubFrameIdx == 0
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {'TVnr_DLMIX_21052_gNB_FAPI_s2.h5'};  % only CSI-RS and SRS in slot 23 (next)
            else
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21052_gNB_FAPI_s2.h5'};  % has PDSCH in slot 63 (next)
            end
            LP.SCHED{twoSubFrameIdx * 40 + 27}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21056_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 28}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21057_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 29}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21058_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 30}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21059_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 31}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21060_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 32}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21061_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 33}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21062_gNB_FAPI_s12.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 37}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21066_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 38}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21067_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 39}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21068_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 40}.config{1}.channels = {bfwTvNames{2},'TVnr_DLMIX_21069_gNB_FAPI_s19.h5'};
        end

        % S/U/U config
        for subframeIdx = 0:3
            dlmixTvSlot3 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s3.h5', 21033 + mod(subframeIdx, 2) * 20);
            if subframeIdx == 0 || subframeIdx == 2
                LP.SCHED{subframeIdx * 20 + 4}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot3,'TVnr_ULMIX_21793_gNB_FAPI_s3.h5'};
            else
                LP.SCHED{subframeIdx * 20 + 4}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot3,'TVnr_ULMIX_21806_gNB_FAPI_s3.h5'};
            end
            LP.SCHED{subframeIdx * 20 + 5}.config{1}.channels = {bfwTvNames{3},'TVnr_ULMIX_21794_gNB_FAPI_s4.h5'};
            LP.SCHED{subframeIdx * 20 + 6}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21795_gNB_FAPI_s5.h5'};
            dlmixTvSlot13 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s13.h5', 21043 + mod(subframeIdx, 2) * 20);
            if subframeIdx == 0 || subframeIdx == 2
                LP.SCHED{subframeIdx * 20 + 14}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot13,'TVnr_ULMIX_21803_gNB_FAPI_s13.h5'};
            else
                LP.SCHED{subframeIdx * 20 + 14}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot13,'TVnr_ULMIX_21807_gNB_FAPI_s13.h5'};
            end
            LP.SCHED{subframeIdx * 20 + 15}.config{1}.channels = {bfwTvNames{5},'TVnr_ULMIX_21804_gNB_FAPI_s14.h5'};
            LP.SCHED{subframeIdx * 20 + 16}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21805_gNB_FAPI_s15.h5'};
        end

        % irregular slots 60,61,63
        LP.SCHED{61}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21034_gNB_FAPI_s0.h5'};  % using 21034 instead of 21050
        LP.SCHED{62}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21035_gNB_FAPI_s1.h5'};  % using 21035 instead of 21051
        LP.SCHED{64}.config{1}.channels = {bfwTvNames{1},'TVnr_DLMIX_21044_gNB_FAPI_s3.h5','TVnr_ULMIX_21793_gNB_FAPI_s3.h5'};  % using 21044 instead of 21053

    case 656  % 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 100 MHz light
        nCell = 1;
        LP = [];
        nSlot = 80;
        LP.Cell_Configs = {'TVnr_DLMIX_21070_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21815_gNB_FAPI_s5.h5'};
        bfwTvNames = {
            'TVnr_9455_gNB_FAPI_s0.h5',
            'TVnr_9456_gNB_FAPI_s0.h5',
            'TVnr_9457_gNB_FAPI_s0.h5',
            'TVnr_9458_gNB_FAPI_s0.h5',
            'TVnr_9459_gNB_FAPI_s0.h5',
            'TVnr_9460_gNB_FAPI_s0.h5'
        };
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        for twoSubFrameIdx = 0:1  % per 40 slots
            % First 20 slots (frame 0)
            LP.SCHED{twoSubFrameIdx * 40 +  1}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21070_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  2}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21071_gNB_FAPI_s1.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  3}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21072_gNB_FAPI_s2.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  7}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21076_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  8}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21077_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  9}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21078_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 10}.config{1}.channels = {bfwTvNames{6},'TVnr_DLMIX_21079_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 11}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21080_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 12}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21081_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 13}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21082_gNB_FAPI_s12.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 17}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21086_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 18}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21087_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 19}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21088_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 20}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21089_gNB_FAPI_s19.h5'};
            % Second 20 slots (frame 1)
            LP.SCHED{twoSubFrameIdx * 40 + 21}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21090_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 22}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21091_gNB_FAPI_s1.h5'};
            if twoSubFrameIdx == 0
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {'TVnr_DLMIX_21092_gNB_FAPI_s2.h5'};  % only CSI-RS and SRS in slot 23 (next)
            else
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21092_gNB_FAPI_s2.h5'};  % has PDSCH in slot 63 (next)
            end
            LP.SCHED{twoSubFrameIdx * 40 + 27}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21096_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 28}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21097_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 29}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21098_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 30}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21099_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 31}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21100_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 32}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21101_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 33}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21102_gNB_FAPI_s12.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 37}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21106_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 38}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21107_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 39}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21108_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 40}.config{1}.channels = {bfwTvNames{2},'TVnr_DLMIX_21109_gNB_FAPI_s19.h5'};
        end

        % S/U/U config
        for subframeIdx = 0:3
            dlmixTvSlot3 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s3.h5', 21073 + mod(subframeIdx, 2) * 20);
            if subframeIdx == 0 || subframeIdx == 2
                LP.SCHED{subframeIdx * 20 + 4}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot3,'TVnr_ULMIX_21813_gNB_FAPI_s3.h5'};
            else
                LP.SCHED{subframeIdx * 20 + 4}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot3,'TVnr_ULMIX_21826_gNB_FAPI_s3.h5'};
            end
            LP.SCHED{subframeIdx * 20 + 5}.config{1}.channels = {bfwTvNames{3},'TVnr_ULMIX_21814_gNB_FAPI_s4.h5'};
            LP.SCHED{subframeIdx * 20 + 6}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21815_gNB_FAPI_s5.h5'};
            dlmixTvSlot13 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s13.h5', 21083 + mod(subframeIdx, 2) * 20);
            if subframeIdx == 0 || subframeIdx == 2
                LP.SCHED{subframeIdx * 20 + 14}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot13,'TVnr_ULMIX_21823_gNB_FAPI_s13.h5'};
            else
                LP.SCHED{subframeIdx * 20 + 14}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot13,'TVnr_ULMIX_21827_gNB_FAPI_s13.h5'};
            end
            LP.SCHED{subframeIdx * 20 + 15}.config{1}.channels = {bfwTvNames{5},'TVnr_ULMIX_21824_gNB_FAPI_s14.h5'};
            LP.SCHED{subframeIdx * 20 + 16}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21825_gNB_FAPI_s15.h5'};
        end

        % irregular slots 60,61,63
        LP.SCHED{61}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21074_gNB_FAPI_s0.h5'};  % using 21074 instead of 21090
        LP.SCHED{62}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21075_gNB_FAPI_s1.h5'};  % using 21075 instead of 21091
        LP.SCHED{64}.config{1}.channels = {bfwTvNames{1},'TVnr_DLMIX_21084_gNB_FAPI_s3.h5','TVnr_ULMIX_21813_gNB_FAPI_s3.h5'};  % using 21084 instead of 21093

    case 657  % % 64TR test case 25-3 'column E', 80 slot pattern, 32DL, 192 SRS per 40ms
    % 657: 2 SRS symbols in S slot
        nCell = 1;
        LP = [];
        nSlot = 80;
        LP.Cell_Configs = {'TVnr_DLMIX_20950_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21755_gNB_FAPI_s5.h5'};
        bfwTvNames = {
                'TVnr_9461_gNB_FAPI_s0.h5',
                'TVnr_9462_gNB_FAPI_s0.h5',
                'TVnr_9463_gNB_FAPI_s0.h5',
                'TVnr_9464_gNB_FAPI_s0.h5',
                'TVnr_9465_gNB_FAPI_s0.h5',
                'TVnr_9466_gNB_FAPI_s0.h5',
                'TVnr_9467_gNB_FAPI_s0.h5'
            };
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        for twoSubFrameIdx = 0:1  % per 40 slots
            LP.SCHED{twoSubFrameIdx * 40 +  1}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20950_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  2}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20951_gNB_FAPI_s1.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  3}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20952_gNB_FAPI_s2.h5'};
            % S/U/U configured below
            LP.SCHED{twoSubFrameIdx * 40 +  7}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20956_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  8}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20957_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  9}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20958_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 10}.config{1}.channels = {bfwTvNames{6},'TVnr_DLMIX_20959_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 11}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20960_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 12}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20961_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 13}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20962_gNB_FAPI_s12.h5'};
            % S/U/U configured below
            LP.SCHED{twoSubFrameIdx * 40 + 17}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20966_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 18}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20967_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 19}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20968_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 20}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20969_gNB_FAPI_s19.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 21}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20970_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 22}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20971_gNB_FAPI_s1.h5'};
            if twoSubFrameIdx == 0
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {'TVnr_DLMIX_20972_gNB_FAPI_s2.h5'};  % only CSI-RS and SRS in slot 23 (next)
            else
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20972_gNB_FAPI_s2.h5'};  % has PDSCH in slot 63 (next)
            end
            % S/U/U configured below
            LP.SCHED{twoSubFrameIdx * 40 + 27}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20976_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 28}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20977_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 29}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20978_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 30}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20979_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 31}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20980_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 32}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20981_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 33}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20982_gNB_FAPI_s12.h5'};
            % S/U/U configured below
            LP.SCHED{twoSubFrameIdx * 40 + 37}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20986_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 38}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20987_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 39}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20988_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 40}.config{1}.channels = {bfwTvNames{2},'TVnr_DLMIX_20989_gNB_FAPI_s19.h5'};   
        end

        % S/U/U config
        for subframeIdx = 0:3
            dlmixTvSlot3 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s3.h5', 20953 + mod(subframeIdx, 2) * 20);
            LP.SCHED{subframeIdx * 20 + 4}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot3,'TVnr_ULMIX_21753_gNB_FAPI_s3.h5'};
            LP.SCHED{subframeIdx * 20 + 5}.config{1}.channels = {bfwTvNames{3},'TVnr_ULMIX_21754_gNB_FAPI_s4.h5'};
            LP.SCHED{subframeIdx * 20 + 6}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21755_gNB_FAPI_s5.h5'};
            dlmixTvSlot13 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s13.h5', 20963 + mod(subframeIdx, 2) * 20);
            LP.SCHED{subframeIdx * 20 + 14}.config{1}.channels = {bfwTvNames{5},dlmixTvSlot13,'TVnr_ULMIX_21763_gNB_FAPI_s13.h5'};
            LP.SCHED{subframeIdx * 20 + 15}.config{1}.channels = {bfwTvNames{7},'TVnr_ULMIX_21764_gNB_FAPI_s14.h5'};
            LP.SCHED{subframeIdx * 20 + 16}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21765_gNB_FAPI_s15.h5'};
        end

        % irregular slots 60,61,63
        LP.SCHED{61}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20954_gNB_FAPI_s0.h5'};  % using 20954 instead of 20970
        LP.SCHED{62}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_20955_gNB_FAPI_s1.h5'};  % using 20955 instead of 20971
        LP.SCHED{64}.config{1}.channels = {bfwTvNames{1},'TVnr_DLMIX_20964_gNB_FAPI_s3.h5','TVnr_ULMIX_21753_gNB_FAPI_s3.h5'};  % using 20964 instead of 20973

    case 658  % 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 90 MHz heavy
        nCell = 1;
        LP = [];
        nSlot = 80;
        LP.Cell_Configs = {'TVnr_DLMIX_21110_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21835_gNB_FAPI_s5.h5'};
        bfwTvNames = {
            'TVnr_9469_gNB_FAPI_s0.h5',
            'TVnr_9470_gNB_FAPI_s0.h5',
            'TVnr_9471_gNB_FAPI_s0.h5',
            'TVnr_9472_gNB_FAPI_s0.h5',
            'TVnr_9473_gNB_FAPI_s0.h5',
            'TVnr_9474_gNB_FAPI_s0.h5'
        };
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        for twoSubFrameIdx = 0:1  % per 40 slots
            % First 20 slots (frame 0)
            LP.SCHED{twoSubFrameIdx * 40 +  1}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21110_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  2}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21111_gNB_FAPI_s1.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  3}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21112_gNB_FAPI_s2.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  7}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21116_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  8}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21117_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  9}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21118_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 10}.config{1}.channels = {bfwTvNames{6},'TVnr_DLMIX_21119_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 11}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21120_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 12}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21121_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 13}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21122_gNB_FAPI_s12.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 17}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21126_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 18}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21127_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 19}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21128_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 20}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21129_gNB_FAPI_s19.h5'};
            % Second 20 slots (frame 1)
            LP.SCHED{twoSubFrameIdx * 40 + 21}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21130_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 22}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21131_gNB_FAPI_s1.h5'};
            if twoSubFrameIdx == 0
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {'TVnr_DLMIX_21132_gNB_FAPI_s2.h5'};  % only CSI-RS and SRS in slot 23 (next)
            else
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21132_gNB_FAPI_s2.h5'};  % has PDSCH in slot 63 (next)
            end
            LP.SCHED{twoSubFrameIdx * 40 + 27}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21136_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 28}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21137_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 29}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21138_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 30}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21139_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 31}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21140_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 32}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21141_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 33}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21142_gNB_FAPI_s12.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 37}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21146_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 38}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21147_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 39}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21148_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 40}.config{1}.channels = {bfwTvNames{2},'TVnr_DLMIX_21149_gNB_FAPI_s19.h5'};
        end

        % S/U/U config
        for subframeIdx = 0:3
            dlmixTvSlot3 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s3.h5', 21113 + mod(subframeIdx, 2) * 20);
            if subframeIdx == 0 || subframeIdx == 2
                LP.SCHED{subframeIdx * 20 + 4}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot3,'TVnr_ULMIX_21833_gNB_FAPI_s3.h5'};
            else
                LP.SCHED{subframeIdx * 20 + 4}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot3,'TVnr_ULMIX_21846_gNB_FAPI_s3.h5'};
            end
            LP.SCHED{subframeIdx * 20 + 5}.config{1}.channels = {bfwTvNames{3},'TVnr_ULMIX_21834_gNB_FAPI_s4.h5'};
            LP.SCHED{subframeIdx * 20 + 6}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21835_gNB_FAPI_s5.h5'};
            dlmixTvSlot13 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s13.h5', 21123 + mod(subframeIdx, 2) * 20);
            if subframeIdx == 0 || subframeIdx == 2
                LP.SCHED{subframeIdx * 20 + 14}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot13,'TVnr_ULMIX_21843_gNB_FAPI_s13.h5'};
            else
                LP.SCHED{subframeIdx * 20 + 14}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot13,'TVnr_ULMIX_21847_gNB_FAPI_s13.h5'};
            end
            LP.SCHED{subframeIdx * 20 + 15}.config{1}.channels = {bfwTvNames{5},'TVnr_ULMIX_21844_gNB_FAPI_s14.h5'};
            LP.SCHED{subframeIdx * 20 + 16}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21845_gNB_FAPI_s15.h5'};
        end

        % irregular slots 60,61,63
        LP.SCHED{61}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21114_gNB_FAPI_s0.h5'};  % using 21114 instead of 21130
        LP.SCHED{62}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21115_gNB_FAPI_s1.h5'};  % using 21115 instead of 21131
        LP.SCHED{64}.config{1}.channels = {bfwTvNames{1},'TVnr_DLMIX_21124_gNB_FAPI_s3.h5','TVnr_ULMIX_21833_gNB_FAPI_s3.h5'};  % using 21124 instead of 21133

    case 659  % 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 90 MHz light
        nCell = 1;
        LP = [];
        nSlot = 80;
        LP.Cell_Configs = {'TVnr_DLMIX_21150_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21855_gNB_FAPI_s5.h5'};
        bfwTvNames = {
            'TVnr_9475_gNB_FAPI_s0.h5',
            'TVnr_9476_gNB_FAPI_s0.h5',
            'TVnr_9477_gNB_FAPI_s0.h5',
            'TVnr_9478_gNB_FAPI_s0.h5',
            'TVnr_9479_gNB_FAPI_s0.h5',
            'TVnr_9480_gNB_FAPI_s0.h5'
        };
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        for twoSubFrameIdx = 0:1  % per 40 slots
            % First 20 slots (frame 0)
            LP.SCHED{twoSubFrameIdx * 40 +  1}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21150_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  2}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21151_gNB_FAPI_s1.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  3}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21152_gNB_FAPI_s2.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  7}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21156_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  8}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21157_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  9}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21158_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 10}.config{1}.channels = {bfwTvNames{6},'TVnr_DLMIX_21159_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 11}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21160_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 12}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21161_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 13}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21162_gNB_FAPI_s12.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 17}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21166_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 18}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21167_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 19}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21168_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 20}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21169_gNB_FAPI_s19.h5'};
            % Second 20 slots (frame 1)
            LP.SCHED{twoSubFrameIdx * 40 + 21}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21170_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 22}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21171_gNB_FAPI_s1.h5'};
            if twoSubFrameIdx == 0
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {'TVnr_DLMIX_21172_gNB_FAPI_s2.h5'};  % only CSI-RS and SRS in slot 23 (next)
            else
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21172_gNB_FAPI_s2.h5'};  % has PDSCH in slot 63 (next)
            end
            LP.SCHED{twoSubFrameIdx * 40 + 27}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21176_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 28}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21177_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 29}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21178_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 30}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21179_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 31}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21180_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 32}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21181_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 33}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21182_gNB_FAPI_s12.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 37}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21186_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 38}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21187_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 39}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21188_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 40}.config{1}.channels = {bfwTvNames{2},'TVnr_DLMIX_21189_gNB_FAPI_s19.h5'};
        end

        % S/U/U config
        for subframeIdx = 0:3
            dlmixTvSlot3 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s3.h5', 21153 + mod(subframeIdx, 2) * 20);
            if subframeIdx == 0 || subframeIdx == 2
                LP.SCHED{subframeIdx * 20 + 4}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot3,'TVnr_ULMIX_21853_gNB_FAPI_s3.h5'};
            else
                LP.SCHED{subframeIdx * 20 + 4}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot3,'TVnr_ULMIX_21866_gNB_FAPI_s3.h5'};
            end
            LP.SCHED{subframeIdx * 20 + 5}.config{1}.channels = {bfwTvNames{3},'TVnr_ULMIX_21854_gNB_FAPI_s4.h5'};
            LP.SCHED{subframeIdx * 20 + 6}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21855_gNB_FAPI_s5.h5'};
            dlmixTvSlot13 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s13.h5', 21163 + mod(subframeIdx, 2) * 20);
            if subframeIdx == 0 || subframeIdx == 2
                LP.SCHED{subframeIdx * 20 + 14}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot13,'TVnr_ULMIX_21863_gNB_FAPI_s13.h5'};
            else
                LP.SCHED{subframeIdx * 20 + 14}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot13,'TVnr_ULMIX_21867_gNB_FAPI_s13.h5'};
            end
            LP.SCHED{subframeIdx * 20 + 15}.config{1}.channels = {bfwTvNames{5},'TVnr_ULMIX_21864_gNB_FAPI_s14.h5'};
            LP.SCHED{subframeIdx * 20 + 16}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21865_gNB_FAPI_s15.h5'};
        end

        % irregular slots 60,61,63
        LP.SCHED{61}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21154_gNB_FAPI_s0.h5'};  % using 21154 instead of 21170
        LP.SCHED{62}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21155_gNB_FAPI_s1.h5'};  % using 21155 instead of 21171
        LP.SCHED{64}.config{1}.channels = {bfwTvNames{1},'TVnr_DLMIX_21164_gNB_FAPI_s3.h5','TVnr_ULMIX_21853_gNB_FAPI_s3.h5'};  % using 21164 instead of 21173

    case 660  % 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 60 MHz heavy
        nCell = 1;
        LP = [];
        nSlot = 80;
        LP.Cell_Configs = {'TVnr_DLMIX_21190_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21875_gNB_FAPI_s5.h5'};
        bfwTvNames = {
            'TVnr_9481_gNB_FAPI_s0.h5',
            'TVnr_9482_gNB_FAPI_s0.h5',
            'TVnr_9483_gNB_FAPI_s0.h5',
            'TVnr_9484_gNB_FAPI_s0.h5',
            'TVnr_9485_gNB_FAPI_s0.h5',
            'TVnr_9486_gNB_FAPI_s0.h5'
        };
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        for twoSubFrameIdx = 0:1  % per 40 slots
            % First 20 slots (frame 0)
            LP.SCHED{twoSubFrameIdx * 40 +  1}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21190_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  2}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21191_gNB_FAPI_s1.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  3}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21192_gNB_FAPI_s2.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  7}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21196_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  8}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21197_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  9}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21198_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 10}.config{1}.channels = {bfwTvNames{6},'TVnr_DLMIX_21199_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 11}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21200_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 12}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21201_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 13}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21202_gNB_FAPI_s12.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 17}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21206_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 18}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21207_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 19}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21208_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 20}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21209_gNB_FAPI_s19.h5'};
            % Second 20 slots (frame 1)
            LP.SCHED{twoSubFrameIdx * 40 + 21}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21210_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 22}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21211_gNB_FAPI_s1.h5'};
            if twoSubFrameIdx == 0
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {'TVnr_DLMIX_21212_gNB_FAPI_s2.h5'};  % only CSI-RS and SRS in slot 23 (next)
            else
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21212_gNB_FAPI_s2.h5'};  % has PDSCH in slot 63 (next)
            end
            LP.SCHED{twoSubFrameIdx * 40 + 27}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21216_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 28}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21217_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 29}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21218_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 30}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21219_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 31}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21220_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 32}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21221_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 33}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21222_gNB_FAPI_s12.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 37}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21226_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 38}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21227_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 39}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21228_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 40}.config{1}.channels = {bfwTvNames{2},'TVnr_DLMIX_21229_gNB_FAPI_s19.h5'};
        end

        % S/U/U config
        for subframeIdx = 0:3
            dlmixTvSlot3 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s3.h5', 21193 + mod(subframeIdx, 2) * 20);
            if subframeIdx == 0 || subframeIdx == 2
                LP.SCHED{subframeIdx * 20 + 4}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot3,'TVnr_ULMIX_21873_gNB_FAPI_s3.h5'};
            else
                LP.SCHED{subframeIdx * 20 + 4}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot3,'TVnr_ULMIX_21886_gNB_FAPI_s3.h5'};
            end
            LP.SCHED{subframeIdx * 20 + 5}.config{1}.channels = {bfwTvNames{3},'TVnr_ULMIX_21874_gNB_FAPI_s4.h5'};
            LP.SCHED{subframeIdx * 20 + 6}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21875_gNB_FAPI_s5.h5'};
            dlmixTvSlot13 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s13.h5', 21203 + mod(subframeIdx, 2) * 20);
            if subframeIdx == 0 || subframeIdx == 2
                LP.SCHED{subframeIdx * 20 + 14}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot13,'TVnr_ULMIX_21883_gNB_FAPI_s13.h5'};
            else
                LP.SCHED{subframeIdx * 20 + 14}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot13,'TVnr_ULMIX_21887_gNB_FAPI_s13.h5'};
            end
            LP.SCHED{subframeIdx * 20 + 15}.config{1}.channels = {bfwTvNames{5},'TVnr_ULMIX_21884_gNB_FAPI_s14.h5'};
            LP.SCHED{subframeIdx * 20 + 16}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21885_gNB_FAPI_s15.h5'};
        end

        % irregular slots 60,61,63
        LP.SCHED{61}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21194_gNB_FAPI_s0.h5'};  % using 21194 instead of 21210
        LP.SCHED{62}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21195_gNB_FAPI_s1.h5'};  % using 21195 instead of 21211
        LP.SCHED{64}.config{1}.channels = {bfwTvNames{1},'TVnr_DLMIX_21204_gNB_FAPI_s3.h5','TVnr_ULMIX_21873_gNB_FAPI_s3.h5'};  % using 21204 instead of 21213

    case 661  % 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 60 MHz light
        nCell = 1;
        LP = [];
        nSlot = 80;
        LP.Cell_Configs = {'TVnr_DLMIX_21230_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21895_gNB_FAPI_s5.h5'};
        bfwTvNames = {
            'TVnr_9487_gNB_FAPI_s0.h5',
            'TVnr_9488_gNB_FAPI_s0.h5',
            'TVnr_9489_gNB_FAPI_s0.h5',
            'TVnr_9490_gNB_FAPI_s0.h5',
            'TVnr_9491_gNB_FAPI_s0.h5',
            'TVnr_9492_gNB_FAPI_s0.h5'
        };
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        for twoSubFrameIdx = 0:1  % per 40 slots
            % First 20 slots (frame 0)
            LP.SCHED{twoSubFrameIdx * 40 +  1}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21230_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  2}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21231_gNB_FAPI_s1.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  3}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21232_gNB_FAPI_s2.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  7}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21236_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  8}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21237_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  9}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21238_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 10}.config{1}.channels = {bfwTvNames{6},'TVnr_DLMIX_21239_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 11}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21240_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 12}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21241_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 13}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21242_gNB_FAPI_s12.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 17}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21246_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 18}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21247_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 19}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21248_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 20}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21249_gNB_FAPI_s19.h5'};
            % Second 20 slots (frame 1)
            LP.SCHED{twoSubFrameIdx * 40 + 21}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21250_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 22}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21251_gNB_FAPI_s1.h5'};
            if twoSubFrameIdx == 0
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {'TVnr_DLMIX_21252_gNB_FAPI_s2.h5'};  % only CSI-RS and SRS in slot 23 (next)
            else
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21252_gNB_FAPI_s2.h5'};  % has PDSCH in slot 63 (next)
            end
            LP.SCHED{twoSubFrameIdx * 40 + 27}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21256_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 28}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21257_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 29}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21258_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 30}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21259_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 31}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21260_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 32}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21261_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 33}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21262_gNB_FAPI_s12.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 37}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21266_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 38}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21267_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 39}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21268_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 40}.config{1}.channels = {bfwTvNames{2},'TVnr_DLMIX_21269_gNB_FAPI_s19.h5'};
        end

        % S/U/U config
        for subframeIdx = 0:3
            dlmixTvSlot3 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s3.h5', 21233 + mod(subframeIdx, 2) * 20);
            if subframeIdx == 0 || subframeIdx == 2
                LP.SCHED{subframeIdx * 20 + 4}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot3,'TVnr_ULMIX_21893_gNB_FAPI_s3.h5'};
            else
                LP.SCHED{subframeIdx * 20 + 4}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot3,'TVnr_ULMIX_21906_gNB_FAPI_s3.h5'};
            end
            LP.SCHED{subframeIdx * 20 + 5}.config{1}.channels = {bfwTvNames{3},'TVnr_ULMIX_21894_gNB_FAPI_s4.h5'};
            LP.SCHED{subframeIdx * 20 + 6}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21895_gNB_FAPI_s5.h5'};
            dlmixTvSlot13 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s13.h5', 21243 + mod(subframeIdx, 2) * 20);
            if subframeIdx == 0 || subframeIdx == 2
                LP.SCHED{subframeIdx * 20 + 14}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot13,'TVnr_ULMIX_21903_gNB_FAPI_s13.h5'};
            else
                LP.SCHED{subframeIdx * 20 + 14}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot13,'TVnr_ULMIX_21907_gNB_FAPI_s13.h5'};
            end
            LP.SCHED{subframeIdx * 20 + 15}.config{1}.channels = {bfwTvNames{5},'TVnr_ULMIX_21904_gNB_FAPI_s14.h5'};
            LP.SCHED{subframeIdx * 20 + 16}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21905_gNB_FAPI_s15.h5'};
        end

        % irregular slots 60,61,63
        LP.SCHED{61}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21234_gNB_FAPI_s0.h5'};  % using 21234 instead of 21250
        LP.SCHED{62}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21235_gNB_FAPI_s1.h5'};  % using 21235 instead of 21251
        LP.SCHED{64}.config{1}.channels = {bfwTvNames{1},'TVnr_DLMIX_21244_gNB_FAPI_s3.h5','TVnr_ULMIX_21893_gNB_FAPI_s3.h5'};  % using 21244 instead of 21253

    case 662  % % 64TR test case 25-3 'column D', 80 slot pattern, 24DL, 192 SRS per 40ms
    % 662: 2 SRS symbols in S slot
        nCell = 1;
        LP = [];
        nSlot = 80;
        LP.Cell_Configs = {'TVnr_DLMIX_21270_gNB_FAPI_s0.h5'};
        LP.UL_Cell_Configs = {'TVnr_ULMIX_21915_gNB_FAPI_s5.h5'};
        bfwTvNames = {
                'TVnr_9493_gNB_FAPI_s0.h5',
                'TVnr_9494_gNB_FAPI_s0.h5',
                'TVnr_9495_gNB_FAPI_s0.h5',
                'TVnr_9496_gNB_FAPI_s0.h5',
                'TVnr_9497_gNB_FAPI_s0.h5',
                'TVnr_9498_gNB_FAPI_s0.h5',
                'TVnr_9499_gNB_FAPI_s0.h5'
            };
        LP.config_static_harq_proc_id = 1; % add for HARQ cases
        LP = init_launchPattern(LP, nSlot, nCell);
        for twoSubFrameIdx = 0:1  % per 40 slots
            LP.SCHED{twoSubFrameIdx * 40 +  1}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21270_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  2}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21271_gNB_FAPI_s1.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  3}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21272_gNB_FAPI_s2.h5'};
            % S/U/U configured below
            LP.SCHED{twoSubFrameIdx * 40 +  7}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21276_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  8}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21277_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 +  9}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21278_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 10}.config{1}.channels = {bfwTvNames{6},'TVnr_DLMIX_21279_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 11}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21280_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 12}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21281_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 13}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21282_gNB_FAPI_s12.h5'};
            % S/U/U configured below
            LP.SCHED{twoSubFrameIdx * 40 + 17}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21286_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 18}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21287_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 19}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21288_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 20}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21289_gNB_FAPI_s19.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 21}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21290_gNB_FAPI_s0.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 22}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21291_gNB_FAPI_s1.h5'};
            if twoSubFrameIdx == 0
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {'TVnr_DLMIX_21292_gNB_FAPI_s2.h5'};  % only CSI-RS and SRS in slot 23 (next)
            else
                LP.SCHED{twoSubFrameIdx * 40 + 23}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21292_gNB_FAPI_s2.h5'};  % has PDSCH in slot 63 (next)
            end
            % S/U/U configured below
            LP.SCHED{twoSubFrameIdx * 40 + 27}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21296_gNB_FAPI_s6.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 28}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21297_gNB_FAPI_s7.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 29}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21298_gNB_FAPI_s8.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 30}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21299_gNB_FAPI_s9.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 31}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21300_gNB_FAPI_s10.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 32}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21301_gNB_FAPI_s11.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 33}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21302_gNB_FAPI_s12.h5'};
            % S/U/U configured below
            LP.SCHED{twoSubFrameIdx * 40 + 37}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21306_gNB_FAPI_s16.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 38}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21307_gNB_FAPI_s17.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 39}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21308_gNB_FAPI_s18.h5'};
            LP.SCHED{twoSubFrameIdx * 40 + 40}.config{1}.channels = {bfwTvNames{2},'TVnr_DLMIX_21309_gNB_FAPI_s19.h5'};   
        end

        % S/U/U config
        for subframeIdx = 0:3
            dlmixTvSlot3 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s3.h5', 21273 + mod(subframeIdx, 2) * 20);
            LP.SCHED{subframeIdx * 20 + 4}.config{1}.channels = {bfwTvNames{1},dlmixTvSlot3,'TVnr_ULMIX_21753_gNB_FAPI_s3.h5'};
            LP.SCHED{subframeIdx * 20 + 5}.config{1}.channels = {bfwTvNames{3},'TVnr_ULMIX_21914_gNB_FAPI_s4.h5'};
            LP.SCHED{subframeIdx * 20 + 6}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21915_gNB_FAPI_s5.h5'};
            dlmixTvSlot13 = sprintf('TVnr_DLMIX_%d_gNB_FAPI_s13.h5', 21283 + mod(subframeIdx, 2) * 20);
            LP.SCHED{subframeIdx * 20 + 14}.config{1}.channels = {bfwTvNames{5},dlmixTvSlot13,'TVnr_ULMIX_21763_gNB_FAPI_s13.h5'};
            LP.SCHED{subframeIdx * 20 + 15}.config{1}.channels = {bfwTvNames{7},'TVnr_ULMIX_21924_gNB_FAPI_s14.h5'};
            LP.SCHED{subframeIdx * 20 + 16}.config{1}.channels = {bfwTvNames{4},'TVnr_ULMIX_21925_gNB_FAPI_s15.h5'};
        end

        % irregular slots 60,61,63
        LP.SCHED{61}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21274_gNB_FAPI_s0.h5'};  % using 21274 instead of 21290
        LP.SCHED{62}.config{1}.channels = {bfwTvNames{4},'TVnr_DLMIX_21275_gNB_FAPI_s1.h5'};  % using 21275 instead of 21291
        LP.SCHED{64}.config{1}.channels = {bfwTvNames{1},'TVnr_DLMIX_21284_gNB_FAPI_s3.h5','TVnr_ULMIX_21753_gNB_FAPI_s3.h5'};  % using 21284 instead of 21293

    case 700 % multi-cell: SRS only in consecutive slots
        % S slot: 3, 13, 23, 33
        % U slot: 4, 5, 14, 15, 24, 25, 34, 35
        nCell = 2;
        nSlot = 40;
        LP = [];
        LP.Cell_Configs = {'TVnr_ULMIX_4300_gNB_FAPI_s3.h5', 'TVnr_ULMIX_4300_gNB_FAPI_s3.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{4}.config{1}.channels = {'TVnr_ULMIX_4300_gNB_FAPI_s3.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_ULMIX_4301_gNB_FAPI_s4.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_ULMIX_4302_gNB_FAPI_s5.h5'};
        LP.SCHED{14}.config{2}.channels = {'TVnr_ULMIX_4303_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{2}.channels = {'TVnr_ULMIX_4304_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{2}.channels = {'TVnr_ULMIX_4305_gNB_FAPI_s15.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_ULMIX_4306_gNB_FAPI_s23.h5'};
        LP.SCHED{25}.config{2}.channels = {'TVnr_ULMIX_4307_gNB_FAPI_s24.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_ULMIX_4308_gNB_FAPI_s25.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_ULMIX_4309_gNB_FAPI_s33.h5'};
        LP.SCHED{34}.config{2}.channels = {'TVnr_ULMIX_4309_gNB_FAPI_s33.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_ULMIX_4310_gNB_FAPI_s34.h5'};
        LP.SCHED{35}.config{2}.channels = {'TVnr_ULMIX_4310_gNB_FAPI_s34.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_ULMIX_4311_gNB_FAPI_s35.h5'};
        LP.SCHED{36}.config{2}.channels = {'TVnr_ULMIX_4311_gNB_FAPI_s35.h5'};
    
      case 701 % multi-cell: SRS only or SRS + PUCCH in consecutive slots
        % S slot: 3, 13, 23, 33
        % U slot: 4, 5, 14, 15, 24, 25, 34, 35
        nCell = 2;
        nSlot = 40;
        LP = [];
        LP.Cell_Configs = {'TVnr_ULMIX_3928_gNB_FAPI_s3.h5', 'TVnr_ULMIX_3928_gNB_FAPI_s3.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{4}.config{1}.channels = {'TVnr_ULMIX_3928_gNB_FAPI_s3.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_ULMIX_3929_gNB_FAPI_s4.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_ULMIX_3930_gNB_FAPI_s5.h5'};
        LP.SCHED{14}.config{2}.channels = {'TVnr_ULMIX_3931_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{2}.channels = {'TVnr_ULMIX_3932_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{2}.channels = {'TVnr_ULMIX_3933_gNB_FAPI_s15.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_ULMIX_3934_gNB_FAPI_s23.h5'};
        LP.SCHED{25}.config{2}.channels = {'TVnr_ULMIX_3935_gNB_FAPI_s24.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_ULMIX_3936_gNB_FAPI_s25.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_ULMIX_3937_gNB_FAPI_s33.h5'};
        LP.SCHED{34}.config{2}.channels = {'TVnr_ULMIX_3937_gNB_FAPI_s33.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_ULMIX_3938_gNB_FAPI_s34.h5'};
        LP.SCHED{35}.config{2}.channels = {'TVnr_ULMIX_3938_gNB_FAPI_s34.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_ULMIX_3939_gNB_FAPI_s35.h5'};
        LP.SCHED{36}.config{2}.channels = {'TVnr_ULMIX_3939_gNB_FAPI_s35.h5'};  

    case 702 % multi-cell: SRS only or SRS + PUSCH in consecutive slots
        % S slot: 3, 13, 23, 33
        % U slot: 4, 5, 14, 15, 24, 25, 34, 35
        nCell = 2;
        nSlot = 40;
        LP = [];
        LP.Cell_Configs = {'TVnr_ULMIX_4312_gNB_FAPI_s3.h5', 'TVnr_ULMIX_4312_gNB_FAPI_s3.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{4}.config{1}.channels = {'TVnr_ULMIX_4312_gNB_FAPI_s3.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_ULMIX_4313_gNB_FAPI_s4.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_ULMIX_4314_gNB_FAPI_s5.h5'};
        LP.SCHED{14}.config{2}.channels = {'TVnr_ULMIX_4315_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{2}.channels = {'TVnr_ULMIX_4316_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{2}.channels = {'TVnr_ULMIX_4317_gNB_FAPI_s15.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_ULMIX_4318_gNB_FAPI_s23.h5'};
        LP.SCHED{25}.config{2}.channels = {'TVnr_ULMIX_4319_gNB_FAPI_s24.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_ULMIX_4320_gNB_FAPI_s25.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_ULMIX_4321_gNB_FAPI_s33.h5'};
        LP.SCHED{34}.config{2}.channels = {'TVnr_ULMIX_4321_gNB_FAPI_s33.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_ULMIX_4322_gNB_FAPI_s34.h5'};
        LP.SCHED{35}.config{2}.channels = {'TVnr_ULMIX_4322_gNB_FAPI_s34.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_ULMIX_4323_gNB_FAPI_s35.h5'};
        LP.SCHED{36}.config{2}.channels = {'TVnr_ULMIX_4323_gNB_FAPI_s35.h5'};  

    case 703 % multi-cell: SRS only or SRS + PRACH in consecutive slots
        % S slot: 3, 13, 23, 33
        % U slot: 4, 5, 14, 15, 24, 25, 34, 35
        nCell = 2;
        nSlot = 40;
        LP = [];
        LP.Cell_Configs = {'TVnr_ULMIX_4324_gNB_FAPI_s3.h5', 'TVnr_ULMIX_4327_gNB_FAPI_s13.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{4}.config{1}.channels = {'TVnr_ULMIX_4324_gNB_FAPI_s3.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_ULMIX_4325_gNB_FAPI_s4.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_ULMIX_4326_gNB_FAPI_s5.h5'};
        LP.SCHED{14}.config{2}.channels = {'TVnr_ULMIX_4327_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{2}.channels = {'TVnr_ULMIX_4328_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{2}.channels = {'TVnr_ULMIX_4329_gNB_FAPI_s15.h5'};
        LP.SCHED{24}.config{1}.channels = {'TVnr_ULMIX_4330_gNB_FAPI_s23.h5'};
        LP.SCHED{25}.config{2}.channels = {'TVnr_ULMIX_4331_gNB_FAPI_s24.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_ULMIX_4332_gNB_FAPI_s25.h5'};
        LP.SCHED{34}.config{1}.channels = {'TVnr_ULMIX_4333_gNB_FAPI_s33.h5'};
        LP.SCHED{34}.config{2}.channels = {'TVnr_ULMIX_4327_gNB_FAPI_s13.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_ULMIX_4334_gNB_FAPI_s34.h5'};
        LP.SCHED{35}.config{2}.channels = {'TVnr_ULMIX_4328_gNB_FAPI_s14.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_ULMIX_4335_gNB_FAPI_s35.h5'};
        LP.SCHED{36}.config{2}.channels = {'TVnr_ULMIX_4329_gNB_FAPI_s15.h5'};

    case 704 % multi-cell: SRS only, SRS + PRACH, SRS + PUCCH, SRS + PUSCH, SRS + PUSCH + PUCCH + PRACH in consecutive slots
        % S slot: 3, 13, 23, 33
        % U slot: 4, 5, 14, 15, 24, 25, 34, 35
        nCell = 2;
        nSlot = 40;
        LP = [];
        LP.Cell_Configs = {'TVnr_ULMIX_3920_gNB_FAPI_s3.h5', 'TVnr_ULMIX_3921_gNB_FAPI_s3.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        % only PUSCH data needs to use the exact slot index
        % below TVs are for slot {3,4,5} but can be used for slot 10*n+{3,4,5}
        % SRS only:    4300:4311
        % SRS + PRACH: 4324:4335
        % 3~5: mix
        LP.SCHED{4}.config{1}.channels = {'TVnr_ULMIX_3920_gNB_FAPI_s3.h5'};
        LP.SCHED{4}.config{2}.channels = {'TVnr_ULMIX_3921_gNB_FAPI_s3.h5'};
        LP.SCHED{5}.config{1}.channels = {'TVnr_ULMIX_3926_gNB_FAPI_s4.h5'};
        LP.SCHED{5}.config{2}.channels = {'TVnr_ULMIX_3940_gNB_FAPI_s4.h5'};
        LP.SCHED{6}.config{1}.channels = {'TVnr_ULMIX_3927_gNB_FAPI_s5.h5'};
        LP.SCHED{6}.config{2}.channels = {'TVnr_ULMIX_3941_gNB_FAPI_s5.h5'};
        % 13~15: mix
        LP.SCHED{14}.config{1}.channels = {'TVnr_ULMIX_4315_gNB_FAPI_s13.h5'};
        LP.SCHED{14}.config{2}.channels = {'TVnr_ULMIX_4303_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_ULMIX_4316_gNB_FAPI_s14.h5'};
        LP.SCHED{15}.config{2}.channels = {'TVnr_ULMIX_3932_gNB_FAPI_s14.h5'};
        LP.SCHED{16}.config{1}.channels = {'TVnr_ULMIX_4326_gNB_FAPI_s5.h5'};
        LP.SCHED{16}.config{2}.channels = {'TVnr_ULMIX_4329_gNB_FAPI_s15.h5'};
        % 23~25: mix
        LP.SCHED{24}.config{1}.channels = {'TVnr_ULMIX_3934_gNB_FAPI_s23.h5'};
        LP.SCHED{24}.config{2}.channels = {'TVnr_ULMIX_4318_gNB_FAPI_s23.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_ULMIX_4325_gNB_FAPI_s4.h5'};
        LP.SCHED{25}.config{2}.channels = {'TVnr_ULMIX_4328_gNB_FAPI_s14.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_ULMIX_4308_gNB_FAPI_s25.h5'};
        LP.SCHED{26}.config{2}.channels = {'TVnr_ULMIX_4320_gNB_FAPI_s25.h5'};
        % 33~35: mix
        LP.SCHED{34}.config{1}.channels = {'TVnr_ULMIX_4333_gNB_FAPI_s33.h5'};
        LP.SCHED{34}.config{2}.channels = {'TVnr_ULMIX_4336_gNB_FAPI_s33.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_ULMIX_4322_gNB_FAPI_s34.h5'};
        LP.SCHED{35}.config{2}.channels = {'TVnr_ULMIX_4310_gNB_FAPI_s34.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_ULMIX_4323_gNB_FAPI_s35.h5'};
        LP.SCHED{36}.config{2}.channels = {'TVnr_ULMIX_3939_gNB_FAPI_s35.h5'};

    case 705 % single-cell: SRS only in single/consecutive slots
        % S slot: 3, 13, 23, 33
        % U slot: 4, 5, 14, 15, 24, 25, 34, 35
        nCell = 1;
        nSlot = 40;
        LP = [];
        LP.Cell_Configs = {'TVnr_ULMIX_4300_gNB_FAPI_s3.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        % single slot
        LP.SCHED{4}.config{1}.channels = {'TVnr_ULMIX_4300_gNB_FAPI_s3.h5'};
        % two consecutive slots
        LP.SCHED{14}.config{1}.channels = {'TVnr_ULMIX_4303_gNB_FAPI_s13.h5'};
        LP.SCHED{15}.config{1}.channels = {'TVnr_ULMIX_4304_gNB_FAPI_s14.h5'};
        LP.SCHED{25}.config{1}.channels = {'TVnr_ULMIX_4307_gNB_FAPI_s24.h5'};
        LP.SCHED{26}.config{1}.channels = {'TVnr_ULMIX_4308_gNB_FAPI_s25.h5'};
        % three consecutive slots
        LP.SCHED{34}.config{1}.channels = {'TVnr_ULMIX_4309_gNB_FAPI_s33.h5'};
        LP.SCHED{35}.config{1}.channels = {'TVnr_ULMIX_4310_gNB_FAPI_s34.h5'};
        LP.SCHED{36}.config{1}.channels = {'TVnr_ULMIX_4311_gNB_FAPI_s35.h5'};


    case 2000 % Single cell launch pattern with specific slot configurations
        nCell = 1;  % Single cell configuration
        nSlot = 20; % 20 slots total
        LP = [];
        LP.Cell_Configs = {'TVnr_7816_gNB_FAPI_s4.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.config_static_harq_proc_id = 1; % add for HARQ cases

        % Configure specific slots with test vectors
        % Slot 4: First test vector
        LP.SCHED{5}.config{1}.channels = {'TVnr_7816_gNB_FAPI_s4.h5'};

        % Slot 9: Second test vector
        LP.SCHED{10}.config{1}.channels = {'TVnr_7817_gNB_FAPI_s9.h5'};

        % Slot 14: Third test vector
        LP.SCHED{15}.config{1}.channels = {'TVnr_7818_gNB_FAPI_s14.h5'};

        % Slot 19: Fourth test vector
        LP.SCHED{20}.config{1}.channels = {'TVnr_7819_gNB_FAPI_s19.h5'};

    case 2001 % Two-cell launch pattern with specific slot configurations
        nCell = 2;  % Two cell configuration
        nSlot = 20; % 20 slots total
        LP = [];
        LP.Cell_Configs = {'TVnr_7800_gNB_FAPI_s4.h5', 'TVnr_7804_gNB_FAPI_s4.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.config_static_harq_proc_id = 1; % add for HARQ cases

        % Configure specific slots with test vectors for both cells
        % Slot 4: First test vector pair
        LP.SCHED{5}.config{1}.channels = {'TVnr_7816_gNB_FAPI_s4.h5'};
        LP.SCHED{5}.config{2}.channels = {'TVnr_7804_gNB_FAPI_s4.h5'};

        % Slot 9: Second test vector pair
        LP.SCHED{10}.config{1}.channels = {'TVnr_7817_gNB_FAPI_s9.h5'};
        LP.SCHED{10}.config{2}.channels = {'TVnr_7805_gNB_FAPI_s9.h5'};

        % Slot 14: Third test vector pair
        LP.SCHED{15}.config{1}.channels = {'TVnr_7818_gNB_FAPI_s14.h5'};
        LP.SCHED{15}.config{2}.channels = {'TVnr_7806_gNB_FAPI_s14.h5'};

        % Slot 19: Fourth test vector pair
        LP.SCHED{20}.config{1}.channels = {'TVnr_7819_gNB_FAPI_s19.h5'};
        LP.SCHED{20}.config{2}.channels = {'TVnr_7807_gNB_FAPI_s19.h5'};

    case 2002 % Four-cell launch pattern with specific slot configurations
        nCell = 4;  % Four cell configuration
        nSlot = 20; % 20 slots total
        LP = [];
        LP.Cell_Configs = {'TVnr_7800_gNB_FAPI_s4.h5', 'TVnr_7804_gNB_FAPI_s4.h5', ...
                           'TVnr_7808_gNB_FAPI_s4.h5', 'TVnr_7812_gNB_FAPI_s4.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.config_static_harq_proc_id = 1; % add for HARQ cases

        % Configure specific slots with test vectors for all four cells
        % Slot 4: First test vector set
        LP.SCHED{5}.config{1}.channels = {'TVnr_7816_gNB_FAPI_s4.h5'};
        LP.SCHED{5}.config{2}.channels = {'TVnr_7804_gNB_FAPI_s4.h5'};
        LP.SCHED{5}.config{3}.channels = {'TVnr_7808_gNB_FAPI_s4.h5'};
        LP.SCHED{5}.config{4}.channels = {'TVnr_7812_gNB_FAPI_s4.h5'};

        % Slot 9: Second test vector set
        LP.SCHED{10}.config{1}.channels = {'TVnr_7817_gNB_FAPI_s9.h5'};
        LP.SCHED{10}.config{2}.channels = {'TVnr_7805_gNB_FAPI_s9.h5'};
        LP.SCHED{10}.config{3}.channels = {'TVnr_7809_gNB_FAPI_s9.h5'};
        LP.SCHED{10}.config{4}.channels = {'TVnr_7813_gNB_FAPI_s9.h5'};

        % Slot 14: Third test vector set
        LP.SCHED{15}.config{1}.channels = {'TVnr_7818_gNB_FAPI_s14.h5'};
        LP.SCHED{15}.config{2}.channels = {'TVnr_7806_gNB_FAPI_s14.h5'};
        LP.SCHED{15}.config{3}.channels = {'TVnr_7810_gNB_FAPI_s14.h5'};
        LP.SCHED{15}.config{4}.channels = {'TVnr_7814_gNB_FAPI_s14.h5'};

        % Slot 19: Fourth test vector set
        LP.SCHED{20}.config{1}.channels = {'TVnr_7819_gNB_FAPI_s19.h5'};
        LP.SCHED{20}.config{2}.channels = {'TVnr_7807_gNB_FAPI_s19.h5'};
        LP.SCHED{20}.config{3}.channels = {'TVnr_7811_gNB_FAPI_s19.h5'};
        LP.SCHED{20}.config{4}.channels = {'TVnr_7815_gNB_FAPI_s19.h5'};

    case 2003 % Two-cell launch pattern with TPT (Throughput) test vectors
        nCell = 2;  % Two cell configuration
        nSlot = 20; % 20 slots total
        LP = [];
        LP.Cell_Configs = {'TVnr_7800_gNB_FAPI_s4.h5', 'TVnr_7804_gNB_FAPI_s4.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.config_static_harq_proc_id = 1; % add for HARQ cases

        % Configure specific slots with test vectors for both cells
        % Slot 4: First test vector pair
        LP.SCHED{5}.config{1}.channels = {'TVnr_7824_gNB_FAPI_s4.h5'};
        LP.SCHED{5}.config{2}.channels = {'TVnr_7828_gNB_FAPI_s4.h5'};

        % Slot 9: Second test vector pair
        LP.SCHED{10}.config{1}.channels = {'TVnr_7825_gNB_FAPI_s9.h5'};
        LP.SCHED{10}.config{2}.channels = {'TVnr_7829_gNB_FAPI_s9.h5'};

        % Slot 14: Third test vector pair
        LP.SCHED{15}.config{1}.channels = {'TVnr_7826_gNB_FAPI_s14.h5'};
        LP.SCHED{15}.config{2}.channels = {'TVnr_7830_gNB_FAPI_s14.h5'};

        % Slot 19: Fourth test vector pair
        LP.SCHED{20}.config{1}.channels = {'TVnr_7827_gNB_FAPI_s19.h5'};
        LP.SCHED{20}.config{2}.channels = {'TVnr_7831_gNB_FAPI_s19.h5'};

    case 2004 % Two-cell launch pattern with TPT (Throughput) test vectors and 2 UL configurations
        nCell = 2;  % Two cells, but each cell has 2 UL configurations
        nSlot = 20; % 20 slots total
        LP = [];
        LP.Cell_Configs = {'TVnr_7800_gNB_FAPI_s4.h5', 'TVnr_7804_gNB_FAPI_s4.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.config_static_harq_proc_id = 1; % add for HARQ cases

        % Configure specific slots with test vectors for all four UL configurations
        % Slot 4: First test vector set
        LP.SCHED{5}.config{1}.channels = {'TVnr_7816_gNB_FAPI_s4.h5'};
        LP.SCHED{5}.config{2}.channels = {'TVnr_7820_gNB_FAPI_s4.h5'};

        % Slot 9: Second test vector set
        LP.SCHED{10}.config{1}.channels = {'TVnr_7817_gNB_FAPI_s9.h5'};
        LP.SCHED{10}.config{2}.channels = {'TVnr_7821_gNB_FAPI_s9.h5'};

        % Slot 14: Third test vector set
        LP.SCHED{15}.config{1}.channels = {'TVnr_7818_gNB_FAPI_s14.h5'};
        LP.SCHED{15}.config{2}.channels = {'TVnr_7822_gNB_FAPI_s14.h5'};

        % Slot 19: Fourth test vector set
        LP.SCHED{20}.config{1}.channels = {'TVnr_7819_gNB_FAPI_s19.h5'};
        LP.SCHED{20}.config{2}.channels = {'TVnr_7823_gNB_FAPI_s19.h5'};

    case 2005 % Four-cell launch pattern with TPT (Throughput) test vectors
        nCell = 4;  % Four cell configuration
        nSlot = 20; % 20 slots total
        LP = [];
        LP.Cell_Configs = {'TVnr_7800_gNB_FAPI_s4.h5', 'TVnr_7804_gNB_FAPI_s4.h5', ...
                           'TVnr_7808_gNB_FAPI_s4.h5', 'TVnr_7812_gNB_FAPI_s4.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.config_static_harq_proc_id = 1; % add for HARQ cases

        % Configure specific slots with test vectors for all four cells
        % Slot 4: First test vector set
        LP.SCHED{5}.config{1}.channels = {'TVnr_7824_gNB_FAPI_s4.h5'};
        LP.SCHED{5}.config{2}.channels = {'TVnr_7828_gNB_FAPI_s4.h5'};
        LP.SCHED{5}.config{3}.channels = {'TVnr_7824_gNB_FAPI_s4.h5'};
        LP.SCHED{5}.config{4}.channels = {'TVnr_7828_gNB_FAPI_s4.h5'};

        % Slot 9: Second test vector set
        LP.SCHED{10}.config{1}.channels = {'TVnr_7825_gNB_FAPI_s9.h5'};
        LP.SCHED{10}.config{2}.channels = {'TVnr_7829_gNB_FAPI_s9.h5'};
        LP.SCHED{10}.config{3}.channels = {'TVnr_7825_gNB_FAPI_s9.h5'};
        LP.SCHED{10}.config{4}.channels = {'TVnr_7829_gNB_FAPI_s9.h5'};

        % Slot 14: Third test vector set
        LP.SCHED{15}.config{1}.channels = {'TVnr_7826_gNB_FAPI_s14.h5'};
        LP.SCHED{15}.config{2}.channels = {'TVnr_7830_gNB_FAPI_s14.h5'};
        LP.SCHED{15}.config{3}.channels = {'TVnr_7826_gNB_FAPI_s14.h5'};
        LP.SCHED{15}.config{4}.channels = {'TVnr_7830_gNB_FAPI_s14.h5'};

        % Slot 19: Fourth test vector set
        LP.SCHED{20}.config{1}.channels = {'TVnr_7827_gNB_FAPI_s19.h5'};
        LP.SCHED{20}.config{2}.channels = {'TVnr_7831_gNB_FAPI_s19.h5'};
        LP.SCHED{20}.config{3}.channels = {'TVnr_7827_gNB_FAPI_s19.h5'};
        LP.SCHED{20}.config{4}.channels = {'TVnr_7831_gNB_FAPI_s19.h5'};

    case 2006 % Four-cell launch pattern with TPT (Throughput) test vectors and 2 UL configurations
        nCell = 4;  % Four cells, each with 2 UL configurations
        nSlot = 20; % 20 slots total
        LP = [];
        LP.Cell_Configs = {'TVnr_7800_gNB_FAPI_s4.h5', 'TVnr_7804_gNB_FAPI_s4.h5', ...
                           'TVnr_7808_gNB_FAPI_s4.h5', 'TVnr_7812_gNB_FAPI_s4.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.config_static_harq_proc_id = 1; % add for HARQ cases

        % Configure specific slots with test vectors for all four cells
        % Slot 4: First test vector set
        LP.SCHED{5}.config{1}.channels = {'TVnr_7816_gNB_FAPI_s4.h5'};
        LP.SCHED{5}.config{2}.channels = {'TVnr_7820_gNB_FAPI_s4.h5'};
        LP.SCHED{5}.config{3}.channels = {'TVnr_7816_gNB_FAPI_s4.h5'};
        LP.SCHED{5}.config{4}.channels = {'TVnr_7820_gNB_FAPI_s4.h5'};

        % Slot 9: Second test vector set
        LP.SCHED{10}.config{1}.channels = {'TVnr_7817_gNB_FAPI_s9.h5'};
        LP.SCHED{10}.config{2}.channels = {'TVnr_7821_gNB_FAPI_s9.h5'};
        LP.SCHED{10}.config{3}.channels = {'TVnr_7817_gNB_FAPI_s9.h5'};
        LP.SCHED{10}.config{4}.channels = {'TVnr_7821_gNB_FAPI_s9.h5'};

        % Slot 14: Third test vector set
        LP.SCHED{15}.config{1}.channels = {'TVnr_7818_gNB_FAPI_s14.h5'};
        LP.SCHED{15}.config{2}.channels = {'TVnr_7822_gNB_FAPI_s14.h5'};
        LP.SCHED{15}.config{3}.channels = {'TVnr_7818_gNB_FAPI_s14.h5'};
        LP.SCHED{15}.config{4}.channels = {'TVnr_7822_gNB_FAPI_s14.h5'};

        % Slot 19: Fourth test vector set
        LP.SCHED{20}.config{1}.channels = {'TVnr_7819_gNB_FAPI_s19.h5'};
        LP.SCHED{20}.config{2}.channels = {'TVnr_7823_gNB_FAPI_s19.h5'};
        LP.SCHED{20}.config{3}.channels = {'TVnr_7819_gNB_FAPI_s19.h5'};
        LP.SCHED{20}.config{4}.channels = {'TVnr_7823_gNB_FAPI_s19.h5'};

    case 2007 % Four-cell launch pattern with LSTN (Listen) test vectors
        nCell = 4;  % Four cells
        nSlot = 20; % 20 slots total
        LP = [];
        LP.Cell_Configs = {'TVnr_7800_gNB_FAPI_s4.h5', 'TVnr_7804_gNB_FAPI_s4.h5', ...
                           'TVnr_7808_gNB_FAPI_s4.h5', 'TVnr_7812_gNB_FAPI_s4.h5'};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.config_static_harq_proc_id = 1; % add for HARQ cases

        % Configure specific slots with test vectors for all four cells
        % Slot 4: First test vector set
        LP.SCHED{5}.config{1}.channels = {'TVnr_7816_gNB_FAPI_s4.h5'};
        LP.SCHED{5}.config{2}.channels = {'TVnr_7804_gNB_FAPI_s4.h5'};
        LP.SCHED{5}.config{3}.channels = {'TVnr_7808_gNB_FAPI_s4.h5'};
        LP.SCHED{5}.config{4}.channels = {'TVnr_7812_gNB_FAPI_s4.h5'};

        % Slot 9: Second test vector set
        LP.SCHED{10}.config{1}.channels = {'TVnr_7817_gNB_FAPI_s9.h5'};
        LP.SCHED{10}.config{2}.channels = {'TVnr_7805_gNB_FAPI_s9.h5'};
        LP.SCHED{10}.config{3}.channels = {'TVnr_7809_gNB_FAPI_s9.h5'};
        LP.SCHED{10}.config{4}.channels = {'TVnr_7813_gNB_FAPI_s9.h5'};

        % Slot 14: Third test vector set
        LP.SCHED{15}.config{1}.channels = {'TVnr_7818_gNB_FAPI_s14.h5'};
        LP.SCHED{15}.config{2}.channels = {'TVnr_7806_gNB_FAPI_s14.h5'};
        LP.SCHED{15}.config{3}.channels = {'TVnr_7810_gNB_FAPI_s14.h5'};
        LP.SCHED{15}.config{4}.channels = {'TVnr_7814_gNB_FAPI_s14.h5'};

        % Slot 19: Fourth test vector set
        LP.SCHED{20}.config{1}.channels = {'TVnr_7819_gNB_FAPI_s19.h5'};
        LP.SCHED{20}.config{2}.channels = {'TVnr_7807_gNB_FAPI_s19.h5'};
        LP.SCHED{20}.config{3}.channels = {'TVnr_7811_gNB_FAPI_s19.h5'};
        LP.SCHED{20}.config{4}.channels = {'TVnr_7815_gNB_FAPI_s19.h5'};
        
    otherwise
        return
end
tvDirName = 'GPU_test_input';
[status,msg] = mkdir(tvDirName);

TVname = sprintf('launch_pattern_nrSim_9%04d', caseNum);
yamlFileName = [tvDirName filesep TVname '.yaml'];

WriteYaml(yamlFileName, LP);

return

function LP = init_launchPattern(LP, nSlot, nCell)

for idxSlot = 1:nSlot
    for idxCell = 1:nCell
        LP.SCHED{idxSlot}.config{idxCell}.cell_index = idxCell-1;
        LP.SCHED{idxSlot}.config{idxCell}.channels = {};
        LP.SCHED{idxSlot}.slot = idxSlot-1;
    end
end

return

function LP = populateLPhelper2Slot(params)
    nCell = 1;
    nSlot = 20;
    LP = [];
    LP.Cell_Configs = {params.data_TV};
    LP = init_launchPattern(LP, nSlot, nCell);
    slot_adj = 0;
    if(params.data_TV(6)=='7')
        slot_adj = 4;
    end
    LP.SCHED{1+slot_adj}.config{1}.channels = {params.bfw_TV};
    LP.SCHED{2+slot_adj}.config{1}.channels = {params.data_TV};
return

function LP = populateLPhelper40Slot(params)
    %%40 slot pattern: SRS + BFW + PDSCH/PUSCH on all slots (DDDSUUDDDD)
    nCell = 1;
    LP = [];
    nSlot = 40;
    LP.Cell_Configs = {params.BFW_DL_TV};
    LP = init_launchPattern(LP, nSlot, nCell);
    %Subframe 0
    LP.SCHED{4}.config{1}.channels = {params.SRS_DL_TV};
    %Subframe 1
    if isfield(params,'SRS_UL_TV')
        LP.SCHED{14}.config{1}.channels = {params.SRS_UL_TV};
    end
    %Subframe 2
    LP.SCHED{21}.config{1}.channels = {params.BFW_DL_TV, params.PDSCH_TV};
    LP.SCHED{22}.config{1}.channels = {params.BFW_DL_TV, params.PDSCH_TV};
    LP.SCHED{23}.config{1}.channels = {params.PDSCH_TV};
    LP.SCHED{24}.config{1}.channels = {params.BFW_UL_TV};
    LP.SCHED{25}.config{1}.channels = {params.BFW_UL_TV, params.PUSCH_TV};
    LP.SCHED{26}.config{1}.channels = {params.BFW_DL_TV, params.PUSCH_TV};
    LP.SCHED{27}.config{1}.channels = {params.BFW_DL_TV, params.PDSCH_TV};
    LP.SCHED{28}.config{1}.channels = {params.BFW_DL_TV, params.PDSCH_TV};
    LP.SCHED{29}.config{1}.channels = {params.BFW_DL_TV, params.PDSCH_TV};
    LP.SCHED{30}.config{1}.channels = {params.BFW_DL_TV, params.PDSCH_TV};
    %Subframe 3
    LP.SCHED{31}.config{1}.channels = {params.BFW_DL_TV, params.PDSCH_TV};
    LP.SCHED{32}.config{1}.channels = {params.BFW_DL_TV, params.PDSCH_TV};
    LP.SCHED{33}.config{1}.channels = {params.PDSCH_TV};
    LP.SCHED{34}.config{1}.channels = {params.BFW_UL_TV};
    LP.SCHED{35}.config{1}.channels = {params.BFW_UL_TV, params.PUSCH_TV};
    LP.SCHED{36}.config{1}.channels = {params.BFW_DL_TV, params.PUSCH_TV};
    LP.SCHED{37}.config{1}.channels = {params.BFW_DL_TV, params.PDSCH_TV};
    LP.SCHED{38}.config{1}.channels = {params.BFW_DL_TV, params.PDSCH_TV};
    LP.SCHED{39}.config{1}.channels = {params.BFW_DL_TV, params.PDSCH_TV};
    LP.SCHED{40}.config{1}.channels = {params.PDSCH_TV};
return

function LP = buildMixedCellPattern(params)
    LP = [];
    DL_prefix = "TVnr_DLMIX_";
    UL_prefix = "TVnr_ULMIX_";
    postfix = "_gNB_FAPI_s0.h5";
    dl_tcs = [ 9472,  9512,  9552, 9592,   9632,  9632,  9792,  9672, ... %Pattern 59c
               9832,  9712,  9912, 10232,  9872,  9752,  9912,  9912, ...
               9912,  9912,  9912, 10232,  9952, 10072,  9992, 10112, ...
              10032, 10152,  9912, 10232,  9912,  9912,  9912,  9912, ...
               9472,  9512,  9552,  9592, 10192,  9632,  9672,  9672, ...
               9712,  9712,  9912, 10232,  9752,  9752,  9912,  9912, ...
               9912,  9912,  9912, 10232, 10072, 10072, 10112, 10112, ...
              10152, 10152,  9912, 10232,  9912,  9912,  9912,  9912;
               7872,  7892,  7912,  7932,  7952,  7952,  8032,  7972, ... % Pattern 61
               8052,  7992,  8092,  8252,  8072,  8012,  8092,  8092, ...
               8092,  8092,  8092,  8252,  8112,  8172,  8132,  8192, ...
               8152,  8212,  8092,  8252,  8092,  8092,  8092,  8092, ...
               7872,  7892,  7912,  7932,  8232,  7952,  7972,  7972, ...
               7992,  7992,  8092,  8252,  8012,  8012,  8092,  8092, ...
               8092,  8092,  8092,  8252,  8172,  8172,  8192,  8192, ...
               8212,  8212,  8092,  8252,  8092,  8092,  8092,  8092;
              10452, 10492, 10532, 10572, 10612, 10612, 10772, 10652, ... % Pattern 60
              10812, 10692, 10892, 11212, 10852, 10732, 10892, 10892, ...
              10892, 10892, 10892, 11212, 10932, 11052, 10972, 11092, ...
              11012, 11132, 10892, 11212, 10892, 10892, 10892, 10892, ...
              10452, 10492, 10532, 10572, 11172, 10612, 10652, 10652, ...
              10692, 10692, 10892, 11212, 10732, 10732, 10892, 10892, ...
              10892, 10892, 10892, 11212, 11052, 11052, 11092, 11092, ...
              11132, 11132, 10892, 11212, 10892, 10892, 10892, 10892];
    ul_slots = [4 5];
    spacing = [40 20 40];
    starting_ul = [4471 3360 3600];
    ul_config = [4511 3380 3640];
    negative_TV = "TVnr_7532_gNB_FAPI_s0.h5";
    neg_slot = [14];
    neg_cell = [1];
    nSlot = (length(dl_tcs)/(10-length(ul_slots)))*10; % 10 slots per frame
    nPatterns = size(dl_tcs,1);
    nCell = params.nCell;%nPatterns;
    LP.Cell_Configs = {};
    LP.UL_Cell_Configs = {};
    for cell = 1:nCell
        tc_num = dl_tcs(mod(cell-1,nPatterns)+1,1)+cell-1;
        LP.Cell_Configs{cell} =  char(DL_prefix + string(tc_num) + postfix);
        tc_num = ul_config(mod(cell-1,nPatterns)+1)+cell-1;
        LP.UL_Cell_Configs{cell} = char(UL_prefix + string(tc_num) + postfix);
    end
    LP = init_launchPattern(LP, nSlot, nCell);
    for cell = 1:nCell
        dl_idx = 1;
        ul_tc = starting_ul(mod(cell-1,nPatterns)+1)+cell-1;
        for slot = 1:nSlot
            if((sum(slot == params.neg_slot) > 0) && (sum(cell == params.neg_cell) > 0))
                file = negative_TV;
            elseif(sum(mod(slot-1,10) == ul_slots) > 0) % UL
                if(mod(slot-1,20) == ul_slots(1)) % Reset every 2 frames
                    ul_tc = starting_ul(mod(cell-1,nPatterns)+1)+cell-1;
                end
                file = UL_prefix + string(ul_tc) + postfix;
                ul_tc = ul_tc + spacing(mod(cell-1,length(starting_ul))+1);
            else
                file = DL_prefix + string(dl_tcs(mod(cell-1,nPatterns)+1,dl_idx)+cell-1) + postfix;
                dl_idx = dl_idx + 1;
            end
            LP.SCHED{slot}.config{cell}.channels = {char(file)};
        end
    end
return
