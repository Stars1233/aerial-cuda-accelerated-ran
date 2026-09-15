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

function errFlag = genLP_POC2(caseSet, cellCountToGenerate)

compact_TC = perfPatternTvYaml('compact_case_set');

if nargin < 2
    cellCountToGenerate = [];
elseif ~isempty(cellCountToGenerate)
    if ~isnumeric(cellCountToGenerate) || ~isscalar(cellCountToGenerate) || ...
       ~isfinite(cellCountToGenerate) || cellCountToGenerate < 0 || ...
       fix(cellCountToGenerate) ~= cellCountToGenerate
        error('cellCountToGenerate must be a non-negative integer scalar.');
    end
end

% Which per-pattern cell count bounds the launch pattern: the full set only
% generates full_cells TVs per config block, the compact set compact_cells.
% genLP_POC2 must not reference more cells than were generated.
useCompactCells = false;
if nargin == 0
    TcToTest = 'full';
elseif nargin >= 1
    if isnumeric(caseSet)
        TcToTest = caseSet;
    else
        switch caseSet
            case 'compact'
                TcToTest = compact_TC;
                useCompactCells = true;
            case 'full'
                TcToTest = "full";
            otherwise
                error('caseSet is not supported...\n');
        end
    end
end

    %% channels in a slot
    PDSCH = {'PDSCH'};
    PDSCH_PDCCH = {'PDSCH', 'PDCCH_DL'};
    PDSCH_PDCCHDLUL = {'PDSCH', 'PDCCH_DL', 'PDCCH_UL'};
    PDSCH_PDCCH_CSI = {'PDSCH', 'PDCCH_DL', 'CSI_RS'};
    PDSCH_PDCCHDLUL_CSI = {'PDSCH', 'CSI_RS', 'PDCCH_DL', 'PDCCH_UL'};
    PDSCH_SSB_PDCCH = {'PDSCH', 'PBCH', 'PDCCH_DL'};
    PDSCH_SSB_PDCCHDLUL = {'PDSCH', 'PBCH', 'PDCCH_DL', 'PDCCH_UL'};
    PDSCH_SSB_PDCCH_CSI = {'PBCH','PDSCH', 'PDCCH_DL', 'CSI_RS'};
    
    PUSCH = {'PUSCH'};
    PUSCH_PUCCH = {'PUSCH', 'PUCCH'};
    PUSCH_PUCCH_SRS = {'PUSCH', 'PUCCH', 'SRS'};
    PUSCH_PRACH = {'PUSCH', 'PRACH'};
    PUSCH_PUCCH_PRACH = {'PUSCH', 'PUCCH', 'PRACH'};
    PUSCH_PUCCH_PRACH_SRS = {'PUSCH', 'PUCCH', 'PRACH', 'SRS'};
    PUCCH_PRACH = {'PUCCH', 'PRACH'};
    Special_PUSCH_PUCCH_PRACH = {'Special!', 'PUSCH', 'PUCCH', 'PRACH'};
    
    SSB_PDCCHDLUL = {'PBCH', 'PDCCH_DL', 'PDCCH_UL'};
    PDCCHDLUL = {'PDCCH_DL', 'PDCCH_UL'};

    SSB_PDCCHDLUL_PUSCH = {'PBCH', 'PDCCH_DL', 'PDCCH_UL', 'PUSCH'};
    PDCCHDLUL_PUSCH = {'PDCCH_DL', 'PDCCH_UL', 'PUSCH'};

    SSB_PDCCHDLUL_SRS = {'PBCH', 'PDCCH_DL', 'PDCCH_UL', 'SRS'};
    PDCCHDLUL_SRS = {'PDCCH_DL', 'PDCCH_UL', 'SRS'};
    
    SSB_PDCCHDLUL_PDSCH = {'PBCH', 'PDCCH_DL', 'PDCCH_UL', 'PDSCH'};
    PDCCHDLUL_PDSCH = {'PDCCH_DL', 'PDCCH_UL', 'PDSCH'};

    PDSCH_PDCCHDLUL_SRS = {'PDSCH', 'PDCCH_DL', 'PDCCH_UL', 'SRS'};

    CSI_SRS = {'CSI_RS', 'SRS'};

    %% channel patterns in slots
    % legacy channels 
    channels_A = {1, 20};
                       % 0                          1                    2                    3                    4
    channels_A(1:5) =   {PDSCH_SSB_PDCCH_CSI,       PDSCH_PDCCH_CSI,     PDSCH_PDCCH_CSI,     PDSCH_PDCCH_CSI,     PUSCH_PUCCH};
                       % 5                          6                    7                    8                    9   
    channels_A(6:10) =  {Special_PUSCH_PUCCH_PRACH, PDSCH_PDCCH_CSI,     PDSCH_PDCCH_CSI,     PDSCH_PDCCH_CSI,     PDSCH_PDCCH_CSI};
                       % 10                         11                   12                   13                   14
    channels_A(11:15) = {PDSCH_PDCCH_CSI,           PDSCH_PDCCH_CSI,     PDSCH_PDCCH_CSI,     PDSCH_PDCCH_CSI,     PUSCH_PUCCH};
                       % 15                         16                   17                   18                   19
    channels_A(16:20) = {Special_PUSCH_PUCCH_PRACH, PDSCH_PDCCH_CSI,     PDSCH_PDCCH_CSI,     PDSCH_PDCCH_CSI,     PDSCH_PDCCH_CSI};
    
    %  SSB in first 4 slots
    channels_A2 = channels_A;
    for slot = 1:4
        channels_A2(slot) = {PDSCH_SSB_PDCCH_CSI};
    end

    % shared channel only
    channels_B = {1, 20}; 
    for slot = [1:4,7:14,17:20]
        channels_B(slot) = {PDSCH};
    end
    for slot = [5,6,15,16]
        channels_B(slot) = {PUSCH};
    end

    % 4 beams setting, SFN % 2 == 0
    channels_C = {1, 20};
                       % 0                          1                    2                    3                    4
    channels_C(1:5) =   {PDSCH_SSB_PDCCH,           PDSCH_SSB_PDCCH,     PDSCH_SSB_PDCCH,     PDSCH_PDCCH,     PUSCH_PUCCH};
                       % 5                          6                    7                    8                    9   
    channels_C(6:10) =  {PUSCH_PUCCH_PRACH,         PDSCH_PDCCH,         PDSCH_PDCCH,         PDSCH_PDCCH,     PDSCH_PDCCH};
                       % 10                         11                   12                   13                   14
    channels_C(11:15) = {PDSCH_PDCCH,               PDSCH_PDCCH,         PDSCH_PDCCH,         PDSCH_PDCCH,     PUSCH_PUCCH};
                       % 15                         16                   17                   18                   19
    channels_C(16:20) = {PUSCH_PUCCH,               PDSCH_PDCCH,         PDSCH_PDCCH,         PDSCH_PDCCH,     PDSCH_PDCCH};

   % 4 beams setting, SFN % 2 == 0, PDCCH UL + DL
    channels_D = {1, 20};
                       % 0                          1                    2                    3                    4
    channels_D(1:5) =   {PDSCH_SSB_PDCCHDLUL,       PDSCH_SSB_PDCCHDLUL, PDSCH_SSB_PDCCHDLUL, PDSCH_PDCCHDLUL,     PUSCH_PUCCH};
                       % 5                          6                    7                    8                    9   
    channels_D(6:10) =  {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
                       % 10                         11                   12                   13                   14
    channels_D(11:15) = {PDSCH_PDCCHDLUL,           PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PUSCH_PUCCH};
                       % 15                         16                   17                   18                   19
    channels_D(16:20) = {PUSCH_PUCCH,               PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};

   % 4 beams setting, SFN % 2 == 1, 
    channels_E = {1, 20};
    for slot = [1,2,11,12,17:20]
        channels_E(slot) = {PDSCH_PDCCH_CSI};
    end
    for slot = [3,4,7:10,13,14]
        channels_E(slot) = {PDSCH_PDCCH};
    end
    channels_E(5) = {PUSCH_PUCCH};
    channels_E(6) = {PUSCH_PUCCH_PRACH};
    channels_E(15) = {PUSCH_PUCCH};
    channels_E(16) = {PUSCH_PUCCH};

    % 4 beams setting, SFN % 2 == 1, PDCCH UL + DL
    channels_F = {1, 20};
    for slot = [1,2,11,12,17:20]
        channels_F(slot) = {PDSCH_PDCCHDLUL_CSI};
    end
    for slot = [3,4,7:10,13,14]
        channels_F(slot) = {PDSCH_PDCCHDLUL};
    end
    channels_F(5) = {PUSCH_PUCCH};
    channels_F(6) = {PUSCH_PUCCH_PRACH};
    channels_F(15) = {PUSCH_PUCCH};
    channels_F(16) = {PUSCH_PUCCH};

    % 4 beams setting, 80 slots
    channels_G = {1, 80};
    channels_G(1:20) = channels_D(1:20); % expect to change to channels_D
    channels_G(21:40) = channels_F(1:20); % expect to change to channels_F
    channels_G(41:60) = channels_D(1:20); % expect to change to channels_D
    channels_G(61:80) = channels_F(1:20); % expect to change to channels_F
    
    %% 7 beams setting, 80 slots
    channels_H = {1, 80};
    % frame 0:
                       % 0                          1                    2                    3                    4
    channels_H(1:5) =   {PDSCH_SSB_PDCCHDLUL,       PDSCH_SSB_PDCCHDLUL, PDSCH_SSB_PDCCHDLUL, SSB_PDCCHDLUL,       PUSCH_PUCCH};
                       % 5                          6                    7                    8                    9   
    channels_H(6:10) =  {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI};
                       % 10                         11                   12                   13                   14
    channels_H(11:15) = {PDSCH_PDCCHDLUL_CSI,       PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL,     PDCCHDLUL,           PUSCH_PUCCH};
                       % 15                         16                   17                   18                   19
    channels_H(16:20) = {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
    % frame 1:
                       % 0                          1                    2                    3                    4
    channels_H(21:25) = {PDSCH_PDCCHDLUL,           PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDCCHDLUL,           PUSCH_PUCCH};
                       % 5                          6                    7                    8                    9   
    channels_H(26:30) = {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI};
                       % 10                         11                   12                   13                   14
    channels_H(31:35) = {PDSCH_PDCCHDLUL_CSI,       PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL,     PDCCHDLUL,           PUSCH_PUCCH};
                       % 15                         16                   17                   18                   19
    channels_H(36:40) = {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
    % frame 2:
    channels_H(41:60) = channels_H(1:20); % channels are the same as frame 0 (content not the same, some with TRS, others with TRS+CSIRS) 
    % frame 3:
    channels_H(61:80) = channels_H(21:40); % channels are the same as frame 1 (content not the same, some with TRS, others with TRS+CSIRS) 

    % 40 MHz ave, slots 5 and 15 has no PUSCH
    channels_I = channels_H;
    channels_I{5+1} = PUCCH_PRACH;
    channels_I{15+1} = PUCCH_PRACH;
    channels_I{25+1} = PUCCH_PRACH;
    channels_I{35+1} = PUCCH_PRACH;
    channels_I{45+1} = PUCCH_PRACH;
    channels_I{65+1} = PUCCH_PRACH;
    channels_I{75+1} = PUCCH_PRACH;

    % 7 beams setting, 80 slots with special slot 
    channels_J = channels_H;
    channels_J{3+1} = SSB_PDCCHDLUL_PUSCH;
    channels_J{13+1} = PDCCHDLUL_PUSCH;
    channels_J{23+1} = PDCCHDLUL_PUSCH; 
    channels_J{33+1} = PDCCHDLUL_PUSCH;
    channels_J{43+1} = SSB_PDCCHDLUL_PUSCH;
    channels_J{53+1} = PDCCHDLUL_PUSCH;
    channels_J{63+1} = PDCCHDLUL_PUSCH;
    channels_J{73+1} = PDCCHDLUL_PUSCH;

    % 64TR, 80 slots with SRS and BFW 
    channels_K = channels_H;
    channels_K{3+1} = SSB_PDCCHDLUL_SRS;
    channels_K{13+1} = PDCCHDLUL_SRS;
    channels_K{23+1} = PDCCHDLUL_SRS; 
    channels_K{33+1} = PDCCHDLUL_SRS;
    channels_K{43+1} = SSB_PDCCHDLUL_SRS;
    channels_K{53+1} = PDCCHDLUL_SRS;
    channels_K{63+1} = PDCCHDLUL_SRS;
    channels_K{73+1} = PDCCHDLUL_SRS;
    
    % 7 beams setting, 80 slots, pdsch in special slot
    channels_L = channels_H;
    channels_L{3+1} = SSB_PDCCHDLUL_PDSCH;
    channels_L{13+1} = PDCCHDLUL_PDSCH;
    channels_L{23+1} = PDCCHDLUL_PDSCH; 
    channels_L{33+1} = PDCCHDLUL_PDSCH;
    channels_L{43+1} = SSB_PDCCHDLUL_PDSCH;
    channels_L{53+1} = PDCCHDLUL_PDSCH;
    channels_L{63+1} = PDCCHDLUL_PDSCH;
    channels_L{73+1} = PDCCHDLUL_PDSCH;

    % 40 slots, worst-case column B, based on 90623
    channels_M = {1, 40};
    % frame 0:
                       % 0                          1                    2                    3                    4
    channels_M(1:5) =   {PDSCH_SSB_PDCCHDLUL,       PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL_SRS, PUSCH_PUCCH};
                       % 5                          6                    7                    8                    9   
    channels_M(6:10) =  {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
                       % 10                         11                   12                   13                   14
    channels_M(11:15) = {PDSCH_PDCCHDLUL,           PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL_SRS, PUSCH_PUCCH};
                       % 15                         16                   17                   18                   19
    channels_M(16:20) = {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
    % frame 1:
                       % 0                          1                    2                    3                    4
    channels_M(21:25) = {PDSCH_PDCCHDLUL_CSI,       PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL_SRS, PUSCH_PUCCH};
                       % 5                          6                    7                    8                    9   
    channels_M(26:30) = {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
                       % 10                         11                   12                   13                   14
    channels_M(31:35) = {PDSCH_PDCCHDLUL,           PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL_SRS, PUSCH_PUCCH};
                       % 15                         16                   17                   18                   19
    channels_M(36:40) = {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};

    % 7 beams setting, 80 slots, pdsch in special slot, PUSCH only in U slot
    channels_N = channels_L;
    for i = 0:7
        channels_N{4 + 10*i + 1} = PUSCH;
        channels_N{5 + 10*i + 1} = PUSCH;
    end

    % 64TR setting, 40 slots, pdsch in special slot, PUSCH only in U slot
    channels_O = channels_M;
    for i = 0:3
        channels_O{4 + 10*i + 1} = PUSCH;
        channels_O{5 + 10*i + 1} = PUSCH;
    end
    channels_O{23 + 1} = PDSCH_PDCCHDLUL;
    channels_O{33 + 1} = PDSCH_PDCCHDLUL;

    % 40 slots, worst-case column G, based on 90624
    channels_P = {1, 40};
    % frame 0:
                        % 0                          1                    2                    3                    4
    channels_P(1:5) =   {PDSCH_SSB_PDCCHDLUL,       PDSCH_SSB_PDCCHDLUL, PDSCH_SSB_PDCCHDLUL, PDSCH_PDCCHDLUL_SRS, PUSCH_PUCCH};
                        % 5                          6                    7                    8                    9   
    channels_P(6:10) =  {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
                        % 10                         11                   12                   13                   14
    channels_P(11:15) = {PDSCH_PDCCHDLUL,           PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PUSCH_PUCCH};
                        % 15                         16                   17                   18                   19
    channels_P(16:20) = {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
    % frame 1:
                        % 0                          1                    2                    3                    4
    channels_P(21:25) = {PDSCH_PDCCHDLUL_CSI,       PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL_SRS, PUSCH_PUCCH};
                        % 5                          6                    7                    8                    9   
    channels_P(26:30) = {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
                        % 10                         11                   12                   13                   14
    channels_P(31:35) = {PDSCH_PDCCHDLUL_CSI,       PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PUSCH_PUCCH};
                        % 15                         16                   17                   18                   19
    channels_P(36:40) = {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI};

    % 40 slots, worst-case column H, based on 90626
    channels_Q = {1, 40};
    % frame 0:
                        % 0                          1                    2                    3                    4
    channels_Q(1:5) =   {PDSCH_SSB_PDCCHDLUL,       PDSCH_SSB_PDCCHDLUL, PDSCH_SSB_PDCCHDLUL, PDSCH_PDCCHDLUL_SRS, PUSCH_PUCCH_SRS};
                        % 5                          6                    7                    8                    9   
    channels_Q(6:10) =  {PUSCH_PUCCH_PRACH_SRS,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
                        % 10                         11                   12                   13                   14
    channels_Q(11:15) = {PDSCH_PDCCHDLUL,           PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PUSCH_PUCCH_SRS};
                        % 15                         16                   17                   18                   19
    channels_Q(16:20) = {PUSCH_PUCCH_PRACH_SRS,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
    % frame 1:
                        % 0                          1                    2                    3                    4
    channels_Q(21:25) = {PDSCH_PDCCHDLUL_CSI,       PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL_SRS, PUSCH_PUCCH_SRS};
                        % 5                          6                    7                    8                    9   
    channels_Q(26:30) = {PUSCH_PUCCH_PRACH_SRS,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
                        % 10                         11                   12                   13                   14
    channels_Q(31:35) = {PDSCH_PDCCHDLUL_CSI,       PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PUSCH_PUCCH_SRS};
                        % 15                         16                   17                   18                   19
    channels_Q(36:40) = {PUSCH_PUCCH_PRACH_SRS,     PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI};

    % 80 slots, worst-case Ph4 column B, PRACH on slot 5/15
    channels_R = {1, 80};
    % frame 0:
                        % 0                          1                    2                    3                    4
    channels_R(1:5) =   {PDSCH_SSB_PDCCHDLUL,       PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL_SRS, PUSCH_PUCCH};
                        % 5                          6                    7                    8                    9   
    channels_R(6:10) =  {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
                        % 10                         11                   12                   13                   14
    channels_R(11:15) = {PDSCH_PDCCHDLUL,           PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL_SRS, PUSCH_PUCCH};
                        % 15                         16                   17                   18                   19
    channels_R(16:20) = {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
    % frame 1:
                        % 0                          1                    2                    3                    4
    channels_R(21:25) = {PDSCH_PDCCHDLUL_CSI,       PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL_SRS, PUSCH_PUCCH};
                        % 5                          6                    7                    8                    9   
    channels_R(26:30) = {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
                        % 10                         11                   12                   13                   14
    channels_R(31:35) = {PDSCH_PDCCHDLUL,           PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL_SRS, PUSCH_PUCCH};
                        % 15                         16                   17                   18                   19
    channels_R(36:40) = {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
    % frame 3:
                        % 0                          1                    2                    3                    4
    channels_R(41:45) = {PDSCH_SSB_PDCCHDLUL,       PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL_SRS, PUSCH_PUCCH};
                        % 5                          6                    7                    8                    9   
    channels_R(46:50) = {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
                        % 10                         11                   12                   13                   14
    channels_R(51:55) = {PDSCH_PDCCHDLUL,           PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL_SRS, PUSCH_PUCCH};
                        % 15                         16                   17                   18                   19
    channels_R(56:60) = {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
    % frame 1:
                        % 0                          1                    2                    3                    4
    channels_R(61:65) = {PDSCH_PDCCHDLUL_CSI,       PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL_SRS, PUSCH_PUCCH};
                        % 5                          6                    7                    8                    9   
    channels_R(66:70) = {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
                        % 10                         11                   12                   13                   14
    channels_R(71:75) = {PDSCH_PDCCHDLUL,           PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL_SRS, PUSCH_PUCCH};
                        % 15                         16                   17                   18                   19
    channels_R(76:80) = {PUSCH_PUCCH_PRACH,         PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};

    % 80 slots, worst-case Ph4 column B, PRACH on slot 5, based on pattern 79 (64TR_nrSim_640)
    channels_S = channels_R;
    channels_S(16) = {PUSCH_PUCCH};  % frame 0, slot 15, only PUSCH+PUCCH (no PRACH)
    channels_S(36) = {PUSCH_PUCCH};  % frame 1, slot 15, only PUSCH+PUCCH (no PRACH)
    channels_S(56) = {PUSCH_PUCCH};  % frame 2, slot 15, only PUSCH+PUCCH (no PRACH)
    channels_S(76) = {PUSCH_PUCCH};  % frame 3, slot 15, only PUSCH+PUCCH (no PRACH)

    channels_S(24) = {CSI_SRS};  % frame 1, slot 3, only CSI-RS + SRS    
    % 40 slots, worst-case column G, based on 90624, SRS on all slots
    channels_T = {1, 40};
    % frame 0:
                        % 0                          1                    2                    3                    4
    channels_T(1:5) =   {PDSCH_SSB_PDCCHDLUL,       PDSCH_SSB_PDCCHDLUL, PDSCH_SSB_PDCCHDLUL, PDSCH_PDCCHDLUL_SRS, PUSCH_PUCCH_SRS};
                        % 5                          6                    7                    8                    9   
    channels_T(6:10) =  {PUSCH_PUCCH_PRACH_SRS,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
                        % 10                         11                   12                   13                   14
    channels_T(11:15) = {PDSCH_PDCCHDLUL,           PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL_SRS, PUSCH_PUCCH_SRS};
                        % 15                         16                   17                   18                   19
    channels_T(16:20) = {PUSCH_PUCCH_PRACH_SRS,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
    % frame 1:
                        % 0                          1                    2                    3                    4
    channels_T(21:25) = {PDSCH_PDCCHDLUL_CSI,       PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL_SRS, PUSCH_PUCCH_SRS};
                        % 5                          6                    7                    8                    9   
    channels_T(26:30) = {PUSCH_PUCCH_PRACH_SRS,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL};
                        % 10                         11                   12                   13                   14
    channels_T(31:35) = {PDSCH_PDCCHDLUL_CSI,       PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL,     PDSCH_PDCCHDLUL_SRS, PUSCH_PUCCH_SRS};
                        % 15                         16                   17                   18                   19
    channels_T(36:40) = {PUSCH_PUCCH_PRACH_SRS,     PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI};

    % 80 slots, worst-case Ph4 column B, PRACH on slot 5, SRS in S and U slots
    channels_U = channels_S;
    for subframeIdx = 0:3
        channels_U(subframeIdx*20 + 5)  = {PUSCH_PUCCH_SRS};
        channels_U(subframeIdx*20 + 6)  = {PUSCH_PUCCH_PRACH_SRS};
        channels_U(subframeIdx*20 + 15) = {PUSCH_PUCCH_SRS};
        channels_U(subframeIdx*20 + 16) = {PUSCH_PUCCH_SRS};
    end

    % Simple pattern: only PDSCH/PUSCH/PRACH (no PUCCH, no control channels)
    channels_V = {1, 40};
    % D slots: PDSCH only
    for slot = [1:3, 7:13, 17:20, 21:23, 27:33, 37:40]
        channels_V(slot) = {PDSCH};
    end
    % S slots: empty (no channels)
    channels_V(4) = {{}};   % slot 3
    channels_V(14) = {{}};  % slot 13
    channels_V(24) = {{}};  % slot 23
    channels_V(34) = {{}};  % slot 33
    % U slots: PUSCH only
    channels_V(5) = {PUSCH};   % slot 4
    channels_V(6) = {PUSCH_PRACH};  % slot 5 with PRACH
    channels_V(15) = {PUSCH};  % slot 14
    channels_V(16) = {PUSCH};  % slot 15
    channels_V(25) = {PUSCH};   % slot 24
    channels_V(26) = {PUSCH_PRACH};  % slot 25 with PRACH
    channels_V(35) = {PUSCH};  % slot 34
    channels_V(36) = {PUSCH};  % slot 35

    % Pattern 103: multichannel, 40-slot frame (SFN%2=0 + SFN%2=1)
    channels_103 = {1, 40};
    % Frame 0 (SFN%2=0): D slots have CSI-RS
    channels_103(1:3)   = {PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI};
    channels_103(4)     = {PDCCHDLUL_PDSCH};   % S slot 3
    channels_103(5)     = {PUSCH_PUCCH};        % slot 4
    channels_103(6)     = {PUSCH_PUCCH_PRACH};  % slot 5
    channels_103(7:13)  = {PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI};
    channels_103(14)    = {PDCCHDLUL_PDSCH};   % S slot 13
    channels_103(15)    = {PUSCH_PUCCH};        % slot 14
    channels_103(16)    = {PUSCH_PUCCH_PRACH};  % slot 15
    channels_103(17:20) = {PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI};
    % Frame 1 (SFN%2=1): D slots have TRS but no CQI CSI-RS
    channels_103(21:23) = {PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI};
    channels_103(24)    = {PDCCHDLUL_PDSCH};   % S slot 23
    channels_103(25)    = {PUSCH_PUCCH};        % slot 24
    channels_103(26)    = {PUSCH_PUCCH_PRACH};  % slot 25
    channels_103(27:33) = {PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI};
    channels_103(34)    = {PDCCHDLUL_PDSCH};   % S slot 33
    channels_103(35)    = {PUSCH_PUCCH};        % slot 34
    channels_103(36)    = {PUSCH_PUCCH_PRACH};  % slot 35
    channels_103(37:40) = {PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI, PDSCH_PDCCHDLUL_CSI};

    %% pattern config
    % Pattern rows live in perf_pattern/perf_pattern_helper.yaml and included pattern files. Channel layouts remain here
    % because they are MATLAB cell arrays of channel definitions.
    channelMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
    channelMap('channels_A') = channels_A;
    channelMap('channels_A2') = channels_A2;
    channelMap('channels_B') = channels_B;
    channelMap('channels_C') = channels_C;
    channelMap('channels_D') = channels_D;
    channelMap('channels_E') = channels_E;
    channelMap('channels_F') = channels_F;
    channelMap('channels_G') = channels_G;
    channelMap('channels_H') = channels_H;
    channelMap('channels_I') = channels_I;
    channelMap('channels_J') = channels_J;
    channelMap('channels_K') = channels_K;
    channelMap('channels_L') = channels_L;
    channelMap('channels_M') = channels_M;
    channelMap('channels_N') = channels_N;
    channelMap('channels_O') = channels_O;
    channelMap('channels_P') = channels_P;
    channelMap('channels_Q') = channels_Q;
    channelMap('channels_R') = channels_R;
    channelMap('channels_S') = channels_S;
    channelMap('channels_T') = channels_T;
    channelMap('channels_U') = channels_U;
    channelMap('channels_V') = channels_V;
    channelMap('channels_103') = channels_103;

    [CFG, fullCellCount, compactCellCount, S_SLOT_CFG, BFW_CFG, BFW_SRS_BIND_CFG, OVERRIDE_CFG] = perfPatternTvYaml('poc2_cfg', channelMap);

    %% run

[nCases, ~] = size(CFG);

if isnumeric(TcToTest)
    poc2Patterns = cell2mat(CFG(:, 1));
    requestedPatterns = reshape(TcToTest, 1, []);
    for idxPattern = 1:numel(requestedPatterns)
        requestedPattern = requestedPatterns(idxPattern);
        if ismember(requestedPattern, poc2Patterns)
            continue;
        end
        patternText = sprintf('%.10g', requestedPattern);
        try
            if ~perfPatternTvYaml('poc2_pattern', patternText)
                patternId = perfPatternTvYaml('pattern_id', patternText);
                warning('genLP_POC2:DeprecatedPattern', ...
                        'Pattern %.10g (%s) is deprecated and has no pattern_type; skipping LP generation.', ...
                        requestedPattern, patternId);
            end
        catch
            warning('genLP_POC2:UnknownPattern', ...
                    'Pattern %.10g is not defined in PERF pattern YAML; skipping LP generation.', ...
                    requestedPattern);
        end
    end
end

parfor n = 1:nCases
    patternNum = CFG{n, 1}; 
    if (isnumeric(TcToTest) && ismember(patternNum, TcToTest)) || ... 
       (~isnumeric(TcToTest) && TcToTest == "full")
        nTvCell = CFG{n, 2};
        patternType = CFG{n, 3};
        dlTvIdx1 = CFG{n, 4};
        dlTvIdx2Delta = CFG{n, 5};
        ulTvIdx = CFG{n, 6};
        channels =  CFG{n, 7};
        % Only full_cells (full set) / compact_cells (compact set) TVs per config
        % block are generated, so the launch pattern can reference at most that
        % many cells. config_cells (nTvCell) stays for the S-slot check below.
        if useCompactCells
            nGenCell = compactCellCount{n};
        else
            nGenCell = fullCellCount{n};
        end
        if isempty(cellCountToGenerate)
            nCellToGenerate = nGenCell;
        elseif cellCountToGenerate > nGenCell
            warning('genLP_POC2:CellCountClamped', ...
                    'Requested %d cells for pattern %.10g but only %d cells were generated; generating %d cells.', ...
                    cellCountToGenerate, patternNum, nGenCell, nGenCell);
            nCellToGenerate = nGenCell;
        else
            nCellToGenerate = cellCountToGenerate;
        end

        sSlotCfg = S_SLOT_CFG(cell2mat(S_SLOT_CFG(:,1)) == patternNum, :);

        if isempty(sSlotCfg) % this pattern has no special slot cfg
            sTvIdx = [0 0 0 0];
        else
            if (nTvCell > sSlotCfg{2})
                error("S slot cell count error!");
            end
            sTvIdx = sSlotCfg{3};
        end

        bfwCfg = cell2mat(BFW_CFG);
        if isempty(bfwCfg)
            bfw_index = [];
        else
            bfw_index = find(bfwCfg(:,1) == patternNum);
        end

        bfw_srs_bind_cfg = BFW_SRS_BIND_CFG(cell2mat(BFW_SRS_BIND_CFG(:,1)) == patternNum, :);

        if isempty(bfw_index) % this pattern has no special slot cfg
            bfwTvDlIdx = -1;
            bfwTvUlIdx = -1;
        else
            bfwTvDlIdx = bfwCfg(bfw_index, 2);
            bfwTvUlIdx = bfwCfg(bfw_index, 3);
        end
        
        override_cfg = OVERRIDE_CFG(cell2mat(OVERRIDE_CFG(:, 1)) == patternNum, :);

        if patternType == "CA"
            
            if length(nTvCell) == 2
                
                cell_bandwidth_combinations =   [1     1  % combvec([1:4],[1:4])', up to 4 cells per bandwidth
                                                 2     1
                                                 3     1
                                                 4     1
                                                 1     2
                                                 2     2
                                                 3     2
                                                 4     2
                                                 1     3
                                                 2     3
                                                 3     3
                                                 4     3
                                                 1     4
                                                 2     4
                                                 3     4
                                                 4     4];
     
            elseif length(nTvCell) == 3
                
                cell_bandwidth_combinations =   [1     1     1  % combvec([1:2],[1:2], [1:2])';  % up to 2 cells per bandwidth
                                                 2     1     1
                                                 1     2     1
                                                 2     2     1
                                                 1     1     2
                                                 2     1     2
                                                 1     2     2
                                                 2     2     2];
     
            else
                error("Not implemented! \n");
            end
            
            for i = 1 : length(cell_bandwidth_combinations)
            	gen_launch_pattern_POC2(patternNum, cell_bandwidth_combinations(i,:), nTvCell, patternType, dlTvIdx1, dlTvIdx2Delta, ulTvIdx, sTvIdx, bfwTvDlIdx, bfwTvUlIdx, bfw_srs_bind_cfg, channels, override_cfg);
            end
        
        else
            for nCell = 1:nCellToGenerate
                gen_launch_pattern_POC2(patternNum, nCell, nTvCell, patternType, dlTvIdx1, dlTvIdx2Delta, ulTvIdx, sTvIdx, bfwTvDlIdx, bfwTvUlIdx, bfw_srs_bind_cfg, channels, override_cfg);
            end
        end
    end
end

errFlag = 0;

return

function gen_launch_pattern_POC2(patternNum, nCell, nTvCell, patternType, dlTvIdx1, dlTvIdx2Delta, ulTvIdx, sTvIdx, bfwTvDlIdx, bfwTvUlIdx, bfw_srs_bind_cfg, channels, override_cfg)

nSlot = length(channels);

DL_TV = cell(nSlot, sum(nCell));
UL_TV = cell(nSlot, sum(nCell));


if patternType == "Legacy"  % old patterns 
    
    for slot = 1 : 10 
        for cell_idx = 1:nCell
            tmpTvIdx = cell_idx-1+dlTvIdx1;
            if tmpTvIdx < 1000
                DL_TV{slot,cell_idx} = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
            else
                DL_TV{slot,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
            end
        end
    end
    
    for slot = 11 : 20
        for cell_idx = 1:nCell
            tmpTvIdx = cell_idx-1 + dlTvIdx1 + dlTvIdx2Delta;
            if tmpTvIdx < 1000
                DL_TV{slot,cell_idx} = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
            else
                DL_TV{slot,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
            end
        end
    end
    
    UL_slots = [5,6,15,16];
    for slot_idx = 1 : length(UL_slots)
        slot = UL_slots(slot_idx);
        for cell_idx = 1 : nTvCell
            tmpTvIdx = cell_idx - 1 + ulTvIdx + (slot_idx-1)*nTvCell;
            if tmpTvIdx < 1000
                UL_TV{slot,cell_idx} = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
            else
                UL_TV{slot,cell_idx} = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
            end
        end
    end
elseif patternType == "SFN_0"  % SNF % 2 == 0
    DL_slots = [1:4, 7:14, 17:20];
    for i = 1 : length(DL_slots)
        slot = DL_slots(i);
        for cell_idx = 1 : nCell
            if ismember(slot, [1])
                tmpTvIdx = cell_idx+ dlTvIdx1 - 1;
            elseif ismember(slot, [2])
                tmpTvIdx = cell_idx + dlTvIdx1 - 1 + nTvCell;
            elseif ismember(slot, [3])
                tmpTvIdx = cell_idx + dlTvIdx1 - 1 + nTvCell*2;
            else
                tmpTvIdx = cell_idx + dlTvIdx1 - 1 + nTvCell*3;
            end
            if tmpTvIdx < 1000
                DL_TV{slot,cell_idx} = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
            else
                DL_TV{slot,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
            end
        end
    end
    UL_slots = [5,6,15,16];
    for i = 1 : length(UL_slots)
        slot = UL_slots(i);
        for cell_idx = 1 : nCell
            tmpTvIdx = cell_idx + ulTvIdx - 1 + (i-1) * nTvCell;
            if tmpTvIdx < 1000
                UL_TV{slot,cell_idx} = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
            else
                UL_TV{slot,cell_idx} = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
            end
        end
    end
elseif patternType == "SFN_1"  % SNF % 2 == 0
    DL_slots = [1:4, 7:14, 17:20];
    for i = 1 : length(DL_slots)
        slot = DL_slots(i);
        for cell_idx = 1 : nCell
            if ismember(slot, [1,11,17,19])
                tmpTvIdx = cell_idx+ dlTvIdx1 - 1;
            elseif ismember(slot, [2,12,18,20])
                tmpTvIdx = cell_idx + dlTvIdx1 - 1 + nTvCell;
            else 
                tmpTvIdx = cell_idx + dlTvIdx1 - 1 + nTvCell*2;
            end
            if tmpTvIdx < 1000
                DL_TV{slot,cell_idx} = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
            else
                DL_TV{slot,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
            end
        end
    end
    UL_slots = [5,6,15,16];
    for i = 1 : length(UL_slots)
        slot = UL_slots(i);
        for cell_idx = 1 : nCell
            tmpTvIdx = cell_idx + ulTvIdx - 1 + (i-1) * nTvCell;
            if tmpTvIdx < 1000
                UL_TV{slot,cell_idx} = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
            else
                UL_TV{slot,cell_idx} = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
            end
        end
    end
elseif (   patternType == "80slots" ...
       || patternType == "80slots_sf24" ...
       || patternType == "80slots_3676_1UEG" ...
       || patternType == "80slots_35367576_1UEG")
       
    DL_slots = [1:4, 7:14, 17:20];
    UL_slots = [5,6,15,16];
    for frame = 1:4
        if frame == 1 || frame == 3            
            for i = 1 : length(DL_slots)
                slot = DL_slots(i);
                for cell_idx = 1 : nCell
                    if ismember(slot, [1])
                        tmpTvIdx = cell_idx+ dlTvIdx1{1}(1) - 1;
                    elseif ismember(slot, [2])
                        tmpTvIdx = cell_idx + dlTvIdx1{1}(1) - 1 + nTvCell;
                    elseif ismember(slot, [3])
                        tmpTvIdx = cell_idx + dlTvIdx1{1}(1) - 1 + nTvCell*2;
                    else
                        tmpTvIdx = cell_idx + dlTvIdx1{1}(1) - 1 + nTvCell*3;
                    end

                    if tmpTvIdx < 1000
                        DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                    else
                        DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                    end
                end
            end
        elseif (frame == 2 && patternType ~= "80slots_sf24") || (frame == 4 && patternType == "80slots_sf24")
            for i = 1 : length(DL_slots)
                slot = DL_slots(i);
                for cell_idx = 1 : nCell
                    if ismember(slot, [1,11,17,19])
                        tmpTvIdx = cell_idx+ dlTvIdx1{1}(frame) - 1;
                    elseif ismember(slot, [2,12,18,20])
                        tmpTvIdx = cell_idx + dlTvIdx1{1}(frame) - 1 + nTvCell;
                    else 
                        tmpTvIdx = cell_idx + dlTvIdx1{1}(frame) - 1 + nTvCell*2;
                    end
                    if tmpTvIdx < 1000
                        DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                    else
                        DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                    end
                end
            end
        else % frame == 4
            for i = 1 : length(DL_slots)
                slot = DL_slots(i);
                for cell_idx = 1 : nCell
                    if ismember(slot, [1,2,11,12,17,18,19,20])
                        tmpTvIdx = cell_idx+ dlTvIdx1{1}(frame) - 1;
                    else 
                        tmpTvIdx = cell_idx + dlTvIdx1{1}(frame) - 1 + nTvCell;
                    end
                    if tmpTvIdx < 1000
                        DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                    else
                        DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                    end
                end
            end
        end
        
        % replace slot 36 (0-indexed) for "80slots_3676_1UEG"
        % also replace slot 76 (0-indexed) for "80slots_3676_1UEG"
        if (patternType == "80slots_3676_1UEG" || patternType == "80slots_35367576_1UEG") ...
            && (frame == 2 || frame == 4)
            slot = 17;
            for cell_idx = 1 : nCell
                if frame == 2 && slot == 17
                    tmpTvIdx = cell_idx + dlTvIdx1{1}(5) - 1;
                elseif frame == 4 && slot == 17
                    tmpTvIdx = cell_idx + dlTvIdx1{1}(5) - 1 + nTvCell;
                end
                if tmpTvIdx < 1000
                    DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                else
                    DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                end
            end            
        end
        
              
        for i = 1 : length(UL_slots)
            slot = UL_slots(i);
            for cell_idx = 1 : nCell
                if patternType ~= "80slots_35367576_1UEG"
                    tmpTvIdx = cell_idx + ulTvIdx - 1 + (i-1) * nTvCell;
                
                else % 1UEG for 35/75 in UL
                    if (frame == 2 || frame == 4) && slot == 16
                        tmpTvIdx = cell_idx + ulTvIdx{1}(2) - 1;
                    else
                        tmpTvIdx = cell_idx + ulTvIdx{1}(1) - 1 + (i-1) * nTvCell;                        
                    end
                end                    
                    
                if tmpTvIdx < 1000
                    UL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                else
                    UL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                end
            end
        end
    end
    [UL_TV_1{1:sum(nCell)}]=deal(UL_TV{6,1:sum(nCell)});
    
elseif patternType == "7beams" || patternType == "CA" || patternType == "7beams_sslot" || patternType == "64TR" || patternType == "7beams_puschOnlyU"
    DL_slots = [1:4, 7:14, 17:20];
    UL_slots = [5,6,15,16];
    for frame = 1:4
        switch frame 
            case 1  % frame 0
                for i = 1 : length(DL_slots)
                    slot = DL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)                            
                            switch (slot - 1)  % 0-indexed here
                                case 0
                                    tmpTvIdx = cell_idx_in_b+ dlTvIdx1{b}(1) - 1;
                                case 1
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;
                                case 2
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;
                                case 3
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;
                                case {6, 7}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*4;
                                case 9
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*5;
                                case 11
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*6;
                                case 17
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*7;
                                case 8
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*8;
                                case 10
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*9;
                                case 16
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*10;
                                case 13
                                            tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*19;
                                otherwise
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*11;
                                
                            end                            
                            
                            DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            % special slot add UL TV
                            if patternType ==  "7beams_sslot" || patternType == "64TR"
                                if ismember(slot-1, [3, 13])
                                    tmpSpeTvIdx = cell_idx_in_b + sTvIdx - 1;
                                    tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                    DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                                end
                            end
                            
                            cell_idx = cell_idx + 1;
                        end                 
                    end
                end
            case 2 % frame 1
                for i = 1 : length(DL_slots)
                    slot = DL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)
                            switch (slot - 1)  % 0-indexed here
                                case {3, 13}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*19;
                                case 6
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*12;
                                case 8
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*13;
                                case 10
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*14;
                                case 7
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*15;
                                case 9
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*16;
                                case 11
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*17;
                                otherwise
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*11;
                            end                 

                            DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            % special slot add UL TV
                            if patternType ==  "7beams_sslot" || patternType == "64TR"
                                if ismember(slot-1, [3, 13])
                                    tmpSpeTvIdx = cell_idx_in_b + sTvIdx - 1;
                                    tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                    DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                                end
                            end
                            
                            cell_idx = cell_idx + 1;
                        end
                    end
                end
                
            case 3 % frame 2      
                
                for i = 1 : length(DL_slots)
                    slot = DL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)
                            switch (slot - 1)  % 0-indexed here
                                case 0
                                    tmpTvIdx = cell_idx_in_b+ dlTvIdx1{b}(1) - 1;
                                case 1
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;
                                case 2
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;
                                case 3
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;
                                case 6
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*18;
                                case 7
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*4;
                                case {8, 9}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*5;
                                case {10, 11}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*6;
                                case 13
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*19;
                                case {16,17}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*7;
                                otherwise
                                    tmpTvIdx = cell_idx + dlTvIdx1{b}(1) - 1 + nTvCell(b)*11;
                            end                 

                            DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            % special slot add UL TV
                            if patternType ==  "7beams_sslot"
                                if ismember(slot-1, [3, 13])
                                    tmpSpeTvIdx = cell_idx_in_b + sTvIdx - 1;
                                    tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                    DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                                end
                            end
                            
                            cell_idx = cell_idx + 1;
                        end
                    end
                end
                
            case 4 % frame 3
                for i = 1 : length(DL_slots)
                    slot = DL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)
                            switch (slot - 1)  % 0-indexed here
                                case {3, 13}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*19;
                                case {6,7}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*15;
                                case {8,9}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*16;
                                case {10,11}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*17; 
                                otherwise
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*11;
                            end                 

                            DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            % special slot add UL TV
                            if patternType ==  "7beams_sslot"
                                if ismember(slot-1, [3, 13])
                                    tmpSpeTvIdx = cell_idx_in_b + sTvIdx - 1;
                                    tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                    DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                                end
                            end
                            
                            cell_idx = cell_idx + 1;
                        end
                    end
                end
            
        end
        
        for i = 1 : length(UL_slots)
            slot = UL_slots(i);
            cell_idx = 1;
                
            bandwidth_num = length(nCell);                
            for b = 1 : bandwidth_num
                for cell_idx_in_b = 1 : nCell(b)
                    if patternType == "7beams_puschOnlyU"
                        tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1;
                    else
                        tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + (i-1) * nTvCell(b);
                    end

                    if tmpTvIdx < 1000
                        UL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                    else
                        UL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                    end
                    cell_idx = cell_idx + 1;
                end
            end
        end       
    end
    [UL_TV_1{1:sum(nCell)}]=deal(UL_TV{6,1:sum(nCell)});
elseif patternType == "64TR_nrSim" || patternType == "64TR_nrSim_puschOnlyU"
    DL_slots = [1:4, 7:14, 17:20];
    UL_slots = [5,6,15,16];
    for frame = 1:2
        switch frame 
            case 1  % frame 0
                for i = 1 : length(DL_slots)
                    slot = DL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)                            
                            switch (slot - 1)  % 0-indexed here
                                case 0
                                    tmpTvIdx = cell_idx_in_b+ dlTvIdx1{b}(1) - 1;
                                case {1,2}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;
                                case 3
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;
                                case {6,7,8,9,10}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;
                                case {11,12}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;  % same with {1,2}
                                case 13
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;  % same with {3}
                                case {16,17,18,19}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;  % same with {6,7,8,9,10}
                                otherwise
                                    error('Invalid slot index.');
                            end                            
                            
                            DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            
                            % special slot add UL TV
                            if ismember(slot-1, 3) && sTvIdx(1) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(1) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            elseif ismember(slot-1, 13) && sTvIdx(2) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(2) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            end

                            cell_idx = cell_idx + 1;
                        end                 
                    end
                end
            case 2 % frame 1
                for i = 1 : length(DL_slots)
                    slot = DL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)
                            switch (slot - 1)  % 0-indexed here
                                case 0
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*4;
                                case 1
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*5;
                                case 2
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;  % same with {1,2}
                                case 3
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;  % same with {3}
                                case {6,7,8,9,10}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;  % same with {6,7,8,9,10}
                                case {11,12}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;  % same with {1,2}
                                case 13
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;  % same with {3}
                                case {16,17,18,19}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;  % same with {6,7,8,9,10}
                                otherwise
                                    error('Invalid slot index.');
                            end               

                            DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);

                            % special slot add UL TV
                            if ismember(slot-1, 3) && sTvIdx(3) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(3) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            elseif ismember(slot-1, 13) && sTvIdx(4) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(4) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            end

                            cell_idx = cell_idx + 1;
                        end
                    end
                end
    
        end
    end
        
    for frame = 1:2
        switch frame 
            case 1  % frame 0
                for i = 1 : length(UL_slots)
                    slot = UL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)                            
                            switch (slot - 1)  % 0-indexed here
                                case 4
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*1;
                                case 5
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*2;
                                case 14
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*4;
                                case 15
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*5;
                                otherwise
                                    error('Invalid slot index.');
                            end                            
                            % if PUSCH only in U slot, using the same TV
                            if patternType == "64TR_nrSim_puschOnlyU"
                                tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1;
                            end
                            
                            UL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            
                            cell_idx = cell_idx + 1;
                        end                 
                    end
                end
            case 2 % frame 1
                for i = 1 : length(UL_slots)
                    slot = UL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)
                            switch (slot - 1)  % 0-indexed here
                                case 4
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*1;
                                case 5
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*2;
                                case 14
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*4;
                                case 15
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*5;
                                otherwise
                                    error('Invalid slot index.');
                            end    
                            % if PUSCH only in U slot, using the same TV
                            if patternType == "64TR_nrSim_puschOnlyU"
                                tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1;
                            end 

                            UL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            
                            cell_idx = cell_idx + 1;
                        end
                    end
                end
    
        end  
    end
    [UL_TV_1{1:sum(nCell)}]=deal(UL_TV{6,1:sum(nCell)});
elseif patternType == "64TR_nrSim_624" || patternType == "64TR_nrSim_624_puschOnlyU"
    DL_slots = [1:4, 7:14, 17:20];
    UL_slots = [5,6,15,16];
    for frame = 1:2
        switch frame 
            case 1  % frame 0
                for i = 1 : length(DL_slots)
                    slot = DL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)                            
                            switch (slot - 1)  % 0-indexed here
                                case 0
                                    tmpTvIdx = cell_idx_in_b+ dlTvIdx1{b}(1) - 1;
                                case 1
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;
                                case 2
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;
                                case 3
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;
                                case {6,7,8,9,10}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*4;
                                case {11,12}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*5;
                                case 13
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;  % same with {3}
                                case {16,17,18,19}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*4;  % same with {6,7,8,9,10}
                                otherwise
                                    error('Invalid slot index.');
                            end                            
                            
                            DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            
                            % special slot add UL TV
                            if ismember(slot-1, 3) && sTvIdx(1) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(1) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            elseif ismember(slot-1, 13) && sTvIdx(2) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(2) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            end

                            cell_idx = cell_idx + 1;
                        end                 
                    end
                end
            case 2 % frame 1
                for i = 1 : length(DL_slots)
                    slot = DL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)
                            switch (slot - 1)  % 0-indexed here
                                case 0
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*6;
                                case 1
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*7;
                                case 2
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*5;  % same with {11,12}
                                case 3
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;  % same with {3}
                                case {6,7,8,9}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*4;  % same with {6,7,8,9,10}
                                case 10
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*8;
                                case 11
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*9;
                                case 12
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*5;
                                case 13
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;  % same with {3}
                                case 16
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*10;
                                case 17
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*11;
                                case 18
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*12;
                                case 19
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*13;
                                otherwise
                                    error('Invalid slot index.');
                            end               

                            DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);

                            % special slot add UL TV
                            if ismember(slot-1, 3) && sTvIdx(3) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(3) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            elseif ismember(slot-1, 13) && sTvIdx(4) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(4) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            end

                            cell_idx = cell_idx + 1;
                        end
                    end
                end
    
        end
    end
        
    for frame = 1:2
        switch frame 
            case 1  % frame 0
                for i = 1 : length(UL_slots)
                    slot = UL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)                            
                            switch (slot - 1)  % 0-indexed here
                                case 4
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1;
                                case 5
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*1;
                                case 14
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*2;
                                case 15
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*3;
                                otherwise
                                    error('Invalid slot index.');
                            end                            
                            % if PUSCH only in U slot, using the same TV
                            if patternType == "64TR_nrSim_624_puschOnlyU"
                                tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1;
                            end
                            
                            UL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            
                            cell_idx = cell_idx + 1;
                        end                 
                    end
                end
            case 2 % frame 1
                for i = 1 : length(UL_slots)
                    slot = UL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)
                            switch (slot - 1)  % 0-indexed here
                                case 4
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1;
                                case 5
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*1;
                                case 14
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*2;
                                case 15
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*3;
                                otherwise
                                    error('Invalid slot index.');
                            end    
                            % if PUSCH only in U slot, using the same TV
                            if patternType == "64TR_nrSim_624_puschOnlyU"
                                tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1;
                            end 

                            UL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            
                            cell_idx = cell_idx + 1;
                        end
                    end
                end
    
        end  
    end
    [UL_TV_1{1:sum(nCell)}]=deal(UL_TV{6,1:sum(nCell)});
elseif patternType == "64TR_nrSim_638"
    DL_slots = [1:4, 7:14, 17:20];
    UL_slots = [5,6,15,16];
    for frame = 1:4
        switch frame 
            case 1  % frame 0
                for i = 1 : length(DL_slots)
                    slot = DL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)                            
                            switch (slot - 1)  % 0-indexed here
                                case 0
                                    tmpTvIdx = cell_idx_in_b+ dlTvIdx1{b}(1) - 1;
                                case {1,2}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;
                                case 3
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;
                                case {6,7,8,9}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;
                                case 10
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*4;
                                case {11,12}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;  % same with {1,2}
                                case 13
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;  % same with {3}
                                case {16,17,18,19}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;  % same with {6,7,8,9}
                                otherwise
                                    error('Invalid slot index.');
                            end                            
                            
                            DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            
                            % special slot add UL TV
                            if ismember(slot-1, 3) && sTvIdx(1) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(1) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            elseif ismember(slot-1, 13) && sTvIdx(2) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(2) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            end

                            cell_idx = cell_idx + 1;
                        end                 
                    end
                end
            case 2 % frame 1
                for i = 1 : length(DL_slots)
                    slot = DL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)
                            switch (slot - 1)  % 0-indexed here
                                case 0
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*5;
                                case 1
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*6;
                                case 2
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;  % same with {1,2}
                                case 3
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;  % same with {3}
                                case {6,7,8,9,10}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;  % same with {6,7,8,9}
                                case {11,12}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;  % same with {1,2}
                                case 13
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;  % same with {3}
                                case {16,17,18,19}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;  % same with {6,7,8,9}
                                otherwise
                                    error('Invalid slot index.');
                            end               

                            DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);

                            % special slot add UL TV
                            if ismember(slot-1, 3) && sTvIdx(3) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(3) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            elseif ismember(slot-1, 13) && sTvIdx(4) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(4) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            end

                            cell_idx = cell_idx + 1;
                        end
                    end
                end
            case 3  % frame 2
                for i = 1 : length(DL_slots)
                    slot = DL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)                            
                            switch (slot - 1)  % 0-indexed here
                                case 0
                                    tmpTvIdx = cell_idx_in_b+ dlTvIdx1{b}(1) - 1;  % same with {0}
                                case {1,2}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;  % same with {1,2}
                                case 3
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;  % same with {3}
                                case {6,7,8,9}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;  % same with {6,7,8,9}
                                case 10
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*4;  % same with {10}
                                case {11,12}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;  % same with {1,2}
                                case 13
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;  % same with {3}
                                case {16,17,18,19}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;  % same with {6,7,8,9}
                                otherwise
                                    error('Invalid slot index.');
                            end                            
                            
                            DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            
                            % special slot add UL TV
                            if ismember(slot-1, 3) && sTvIdx(1) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(1) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            elseif ismember(slot-1, 13) && sTvIdx(2) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(2) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            end

                            cell_idx = cell_idx + 1;
                        end                 
                    end
                end
            case 4 % frame 3
                for i = 1 : length(DL_slots)
                    slot = DL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)
                            switch (slot - 1)  % 0-indexed here
                                case 0
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*7;
                                case 1
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*8;
                                case 2
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;  % same with {1,2}
                                case 3
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;  % same with {3}
                                case {6,7,8,9,10}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;  % same with {6,7,8,9}
                                case {11,12}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;  % same with {1,2}
                                case 13
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;  % same with {3}
                                case {16,17,18,19}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;  % same with {6,7,8,9}
                                otherwise
                                    error('Invalid slot index.');
                            end               

                            DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);

                            % special slot add UL TV
                            if ismember(slot-1, 3) && sTvIdx(3) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(3) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            elseif ismember(slot-1, 13) && sTvIdx(4) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(4) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            end

                            cell_idx = cell_idx + 1;
                        end
                    end
                end
    
        end
    end
        
    for frame = 1:4
        switch frame 
            case 1  % frame 0
                for i = 1 : length(UL_slots)
                    slot = UL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)                            
                            switch (slot - 1)  % 0-indexed here
                                case 4
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1;
                                case 5
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*1;
                                case 14
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*3;
                                case 15
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*4;
                                otherwise
                                    error('Invalid slot index.');
                            end
                            
                            UL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            
                            cell_idx = cell_idx + 1;
                        end                 
                    end
                end
            case 2 % frame 1
                for i = 1 : length(UL_slots)
                    slot = UL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)
                            switch (slot - 1)  % 0-indexed here
                                case 4
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1;
                                case 5
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*1;
                                case 14
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*3;
                                case 15
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*4;
                                otherwise
                                    error('Invalid slot index.');
                            end

                            UL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            
                            cell_idx = cell_idx + 1;
                        end
                    end
                end
            case 3  % frame 2
                for i = 1 : length(UL_slots)
                    slot = UL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)                            
                            switch (slot - 1)  % 0-indexed here
                                case 4
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1;
                                case 5
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*1;
                                case 14
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*3;
                                case 15
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*4;
                                otherwise
                                    error('Invalid slot index.');
                            end
                            
                            UL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            
                            cell_idx = cell_idx + 1;
                        end                 
                    end
                end
            case 4 % frame 3
                for i = 1 : length(UL_slots)
                    slot = UL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)
                            switch (slot - 1)  % 0-indexed here
                                case 4
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1;
                                case 5
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*1;
                                case 14
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*3;
                                case 15
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*4;
                                otherwise
                                    error('Invalid slot index.');
                            end

                            UL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            
                            cell_idx = cell_idx + 1;
                        end
                    end
                end
    
        end  
    end
    [UL_TV_1{1:sum(nCell)}]=deal(UL_TV{6,1:sum(nCell)});
elseif patternType == "64TR_nrSim_640"
    DL_slots = [1:4, 7:14, 17:20];
    UL_slots = [5,6,15,16];
    for frame = 1:4
        switch frame 
            case 1  % frame 0
                for i = 1 : length(DL_slots)
                    slot = DL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)                            
                            switch (slot - 1)  % 0-indexed here
                                case 0
                                    tmpTvIdx = cell_idx_in_b+ dlTvIdx1{b}(1) - 1;
                                case {1,2}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;
                                case 3
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;
                                case {6,7,8,9}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;
                                case 10
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*4;
                                case {11,12}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;  % same with {1,2}
                                case 13
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;  % same with {3}
                                case {16,17,18,19}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;  % same with {6,7,8,9}
                                otherwise
                                    error('Invalid slot index.');
                            end                            
                            
                            DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            
                            % special slot add UL TV
                            if ismember(slot-1, 3) && sTvIdx(1) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(1) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            elseif ismember(slot-1, 13) && sTvIdx(2) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(2) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            end

                            cell_idx = cell_idx + 1;
                        end                 
                    end
                end
            case 2 % frame 1
                for i = 1 : length(DL_slots)
                    slot = DL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)
                            switch (slot - 1)  % 0-indexed here
                                case 0
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*5;
                                case 1
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*6;
                                case 2
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;  % same with {1,2}
                                case 3
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*7;  % CSI_SRS in frame 1
                                case {6,7,8,9,10}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;  % same with {6,7,8,9}
                                case {11,12}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;  % same with {1,2}
                                case 13
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;  % same with {3} in frame 0
                                case {16,17,18,19}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;  % same with {6,7,8,9}
                                otherwise
                                    error('Invalid slot index.');
                            end               

                            DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);

                            % special slot add UL TV
                            if ismember(slot-1, 3) && sTvIdx(3) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(3) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            elseif ismember(slot-1, 13) && sTvIdx(4) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(4) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            end

                            cell_idx = cell_idx + 1;
                        end
                    end
                end
            case 3  % frame 2
                for i = 1 : length(DL_slots)
                    slot = DL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)                            
                            switch (slot - 1)  % 0-indexed here
                                case 0
                                    tmpTvIdx = cell_idx_in_b+ dlTvIdx1{b}(1) - 1;  % same with {0}
                                case {1,2}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;  % same with {1,2}
                                case 3
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;  % same with {3}
                                case {6,7,8,9}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;  % same with {6,7,8,9}
                                case 10
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*4;  % same with {10}
                                case {11,12}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;  % same with {1,2}
                                case 13
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;  % same with {3}
                                case {16,17,18,19}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;  % same with {6,7,8,9}
                                otherwise
                                    error('Invalid slot index.');
                            end                            
                            
                            DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            
                            % special slot add UL TV
                            if ismember(slot-1, 3) && sTvIdx(1) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(1) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            elseif ismember(slot-1, 13) && sTvIdx(2) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(2) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            end

                            cell_idx = cell_idx + 1;
                        end                 
                    end
                end
            case 4 % frame 3
                for i = 1 : length(DL_slots)
                    slot = DL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)
                            switch (slot - 1)  % 0-indexed here
                                case 0
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*8;
                                case 1
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*9;
                                case 2
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;  % same with {1,2}
                                case 3
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;  % same with {3}
                                case {6,7,8,9,10}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;  % same with {6,7,8,9}
                                case {11,12}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*1;  % same with {1,2}
                                case 13
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*2;  % same with {3}
                                case {16,17,18,19}
                                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b)*3;  % same with {6,7,8,9}
                                otherwise
                                    error('Invalid slot index.');
                            end               

                            DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);

                            % special slot add UL TV
                            if ismember(slot-1, 3) && sTvIdx(3) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(3) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            elseif ismember(slot-1, 13) && sTvIdx(4) > 0
                                tmpSpeTvIdx = cell_idx_in_b + sTvIdx(4) - 1;
                                tmp_ulTv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpSpeTvIdx);
                                DL_TV{slot+(frame-1)*20,cell_idx} = [DL_TV{slot+(frame-1)*20,cell_idx}, {tmp_ulTv_str}];
                            end

                            cell_idx = cell_idx + 1;
                        end
                    end
                end
    
        end
    end
        
    for frame = 1:4
        switch frame 
            case 1  % frame 0
                for i = 1 : length(UL_slots)
                    slot = UL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)                            
                            switch (slot - 1)  % 0-indexed here
                                case 4
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1;
                                case 5
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*1;
                                case 14
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*3;
                                case 15
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*4;
                                otherwise
                                    error('Invalid slot index.');
                            end
                            
                            UL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            
                            cell_idx = cell_idx + 1;
                        end                 
                    end
                end
            case 2 % frame 1
                for i = 1 : length(UL_slots)
                    slot = UL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)
                            switch (slot - 1)  % 0-indexed here
                                case 4
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1;
                                case 5
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*1;
                                case 14
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*3;
                                case 15
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*4;
                                otherwise
                                    error('Invalid slot index.');
                            end

                            UL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            
                            cell_idx = cell_idx + 1;
                        end
                    end
                end
            case 3  % frame 2
                for i = 1 : length(UL_slots)
                    slot = UL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)                            
                            switch (slot - 1)  % 0-indexed here
                                case 4
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1;
                                case 5
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*1;
                                case 14
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*3;
                                case 15
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*4;
                                otherwise
                                    error('Invalid slot index.');
                            end

                            UL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            
                            cell_idx = cell_idx + 1;
                        end
                    end
                end
            case 4 % frame 3
                for i = 1 : length(UL_slots)
                    slot = UL_slots(i);
                    cell_idx = 1;
                    bandwidth_num = length(nCell);                
                    for b = 1 : bandwidth_num
                        for cell_idx_in_b = 1 : nCell(b)
                            switch (slot - 1)  % 0-indexed here
                                case 4
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1;
                                case 5
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*1;
                                case 14
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*3;
                                case 15
                                    tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*4;
                                otherwise
                                    error('Invalid slot index.');
                            end

                            UL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                            
                            cell_idx = cell_idx + 1;
                        end
                    end
                end
    
        end  
    end
    [UL_TV_1{1:sum(nCell)}]=deal(UL_TV{6,1:sum(nCell)});
    
elseif patternType == "4TR_pusch_prach"
    % Simple pattern: only PDSCH/PUSCH/PRACH
    % DLMIX is 1 TV set for all D slots, ULMIX is 4 TV sets for 4/5/14/15 slots
    DL_slots = [1:3, 7:13, 17:20];
    UL_slots = [5,6,15,16];
    
    % DL TV assignment - single DL TV for all D slots across all frames
    for frame = 1:2
        for i = 1 : length(DL_slots)
            slot = DL_slots(i);
            cell_idx = 1;
            bandwidth_num = length(nCell);                
            for b = 1 : bandwidth_num
                for cell_idx_in_b = 1 : nCell(b)
                    % All D slots use the same TV base, just offset by cell index
                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1;
                    DL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                    cell_idx = cell_idx + 1;
                end                 
            end
        end
    end
    
    % UL TV assignment - 4/5/14/15 slots
    for frame = 1:2
        for i = 1 : length(UL_slots)
            slot = UL_slots(i);
            cell_idx = 1;
            bandwidth_num = length(nCell);                
            for b = 1 : bandwidth_num
                for cell_idx_in_b = 1 : nCell(b)                            
                    switch (slot - 1)  % 0-indexed here
                        case 4
                            tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1;
                        case 5
                            tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*1;
                        case 14
                            tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*2;
                        case 15
                            tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*3;
                        otherwise
                            error('Invalid slot index.');
                    end
                    
                    UL_TV{slot+(frame-1)*20,cell_idx} = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                    
                    cell_idx = cell_idx + 1;
                end                 
            end
        end
    end
    [UL_TV_1{1:sum(nCell)}]=deal(UL_TV{6,1:sum(nCell)});

elseif patternType == "4TR_allChan_103"
    % Pattern 103: 3 DL TV groups + 4 UL TV groups, 40-slot frame
    % DL Group A (offset 0):          SFN%2=0 D slots (frame 1) — PDCCH + PDSCH sym1-12 + TRS + CQI
    % DL Group B (offset nCell):      SFN%2=1 D slots (frame 2) — PDCCH + PDSCH sym1-13 + TRS
    % DL Group C (offset 2*nCell):    S slots (both frames) — PDCCH + PDSCH sym1-5
    % UL: 4 slot types x nCell, same as 4TR_pusch_prach
    D_slots = [1:3, 7:13, 17:20];
    S_slots = [4, 14];
    UL_slots = [5, 6, 15, 16];

    for frame = 1:2
        for i = 1:length(D_slots)
            slot = D_slots(i);
            cell_idx = 1;
            bandwidth_num = length(nCell);
            for b = 1:bandwidth_num
                for cell_idx_in_b = 1:nCell(b)
                    if frame == 1
                        tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1;              % Group A
                    else
                        tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + nTvCell(b); % Group B
                    end
                    DL_TV{slot+(frame-1)*20, cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                    cell_idx = cell_idx + 1;
                end
            end
        end
        for i = 1:length(S_slots)
            slot = S_slots(i);
            cell_idx = 1;
            bandwidth_num = length(nCell);
            for b = 1:bandwidth_num
                for cell_idx_in_b = 1:nCell(b)
                    tmpTvIdx = cell_idx_in_b + dlTvIdx1{b}(1) - 1 + 2*nTvCell(b);   % Group C
                    DL_TV{slot+(frame-1)*20, cell_idx} = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                    cell_idx = cell_idx + 1;
                end
            end
        end
    end

    for frame = 1:2
        for i = 1:length(UL_slots)
            slot = UL_slots(i);
            cell_idx = 1;
            bandwidth_num = length(nCell);
            for b = 1:bandwidth_num
                for cell_idx_in_b = 1:nCell(b)
                    switch (slot - 1)
                        case 4
                            tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1;
                        case 5
                            tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*1;
                        case 14
                            tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*2;
                        case 15
                            tmpTvIdx = cell_idx_in_b + ulTvIdx(b) - 1 + nTvCell(b)*3;
                        otherwise
                            error('Invalid slot index.');
                    end
                    UL_TV{slot+(frame-1)*20, cell_idx} = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s0.h5', tmpTvIdx);
                    cell_idx = cell_idx + 1;
                end
            end
        end
    end
    [UL_TV_1{1:sum(nCell)}]=deal(UL_TV{6,1:sum(nCell)});
end


% append BFW TV

if patternType == "64TR"

    for idxCell = 1:nCell
        for slot = 1:nSlot
            frame = floor((slot-1)/20);
            slotInFrame = mod((slot-1),20);
            switch slotInFrame % 0-indexed here
                case {0, 1}
                    if ismember(frame, [0, 2]) % DL slot with DL bfw (with ssb)           
                        tmp_tv_str = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', bfwTvDlIdx);
                        DL_TV{slot, idxCell} = [DL_TV{slot, idxCell}, {tmp_tv_str}];
                        % fprintf('frame %4d slot %4d DL with SSB: TV %s \n', frame, slotInFrame, tmp_tv_str);
                    else
                        tmp_tv_str = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', bfwTvDlIdx+1);
                        DL_TV{slot, idxCell} = [DL_TV{slot, idxCell}, {tmp_tv_str}];
                        % fprintf('frame %4d slot %4d DL with no SSB: TV %s \n', frame, slotInFrame, tmp_tv_str);
                    end
                case {19}
                    if ismember(frame, [1, 3]) % DL slot with DL bfw (with ssb)           
                        tmp_tv_str = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', bfwTvDlIdx);
                        DL_TV{slot, idxCell} = [DL_TV{slot, idxCell}, {tmp_tv_str}];
                        % fprintf('frame %4d slot %4d DL with SSB: TV %s \n', frame, slotInFrame, tmp_tv_str);
                    else
                        tmp_tv_str = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', bfwTvDlIdx+1);
                        DL_TV{slot, idxCell} = [DL_TV{slot, idxCell}, {tmp_tv_str}];
                        % fprintf('frame %4d slot %4d DL with no SSB: TV %s \n', frame, slotInFrame, tmp_tv_str);
                    end
                case {6, 7, 8, 9, 10, 11, 16, 17, 18} % DL slot with DL bfw (no ssb)
                    tmp_tv_str = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', bfwTvDlIdx+1);
                    DL_TV{slot, idxCell} = [DL_TV{slot, idxCell}, {tmp_tv_str}];
                    % fprintf('frame %4d slot %4d DL with no SSB: TV %s \n', frame, slotInFrame, tmp_tv_str);
                case {5, 15} % UL slot with DL bfw (no ssb)
                    tmp_tv_str = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', bfwTvDlIdx+1);
                    UL_TV{slot, idxCell} = [UL_TV{slot, idxCell}, {tmp_tv_str}];
                    % fprintf('frame %4d slot %4d UL with no SSB: TV %s \n', frame, slotInFrame, tmp_tv_str);
                case {3, 13} % special slot with UL bfw (no prach)
                    tmp_tv_str = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', bfwTvUlIdx);
                    DL_TV{slot, idxCell} = [DL_TV{slot, idxCell}, {tmp_tv_str}];
                    % fprintf('frame %4d slot %4d DL with no prach: TV %s \n', frame, slotInFrame, tmp_tv_str);
                case {4} % UL slot with UL bfw (4 prach)
                    tmp_tv_str = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', bfwTvUlIdx+1);
                    UL_TV{slot, idxCell} = [UL_TV{slot, idxCell}, {tmp_tv_str}];
                    % fprintf('frame %4d slot %4d UL with 4 prach: TV %s \n', frame, slotInFrame, tmp_tv_str);
                case {14} % UL slot with UL bfw (3 prach)
                    tmp_tv_str = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', bfwTvUlIdx+2);
                    UL_TV{slot, idxCell} = [UL_TV{slot, idxCell}, {tmp_tv_str}];
                    % fprintf('frame %4d slot %4d UL with 3 prach: TV %s \n', frame, slotInFrame, tmp_tv_str);
            end
        end
    end
end

% bind BFW and SRS based on bfw_srs_bind_cfg
if (patternType == "64TR_nrSim" || patternType == "64TR_nrSim_puschOnlyU" || patternType == "64TR_nrSim_624" || patternType == "64TR_nrSim_624_puschOnlyU" || patternType == "64TR_nrSim_638" || patternType == "64TR_nrSim_640") && ~isempty(bfw_srs_bind_cfg)
    for i = 1:size(bfw_srs_bind_cfg, 1)
        % extract the current row
        currentRow = bfw_srs_bind_cfg(i, :);
        
        % determine which cells to apply based on cellIdxOverride (9th column)
        cellIdxOverride = currentRow{9};
        if isempty(cellIdxOverride)
            % empty cellIdxOverride means apply to all cells
            cellsToApply = 1:nCell;  % 1-based indexing for MATLAB
        else
            % only apply to cells in cellIdxOverride that exist in current nCell config
            % cellIdxOverride is 0-based, convert to 1-based for MATLAB indexing
            availableCells = 0:(nCell-1);  % 0-based cell indices
            cellsToApply = intersect(cellIdxOverride, availableCells) + 1;  % convert to 1-based
        end
        
        for idxCell = cellsToApply
            % add ULMIX TV with SRS
            srsTvIdx = currentRow{2};
            srsSlotIdx = currentRow{3};
            srsTvNameSlotIdx = currentRow{4};
            if srsTvIdx < 20000
                ulmix_srs_tv_str = sprintf('TVnr_%04d_gNB_FAPI_s%d.h5', srsTvIdx, srsTvNameSlotIdx);
            else
                ulmix_srs_tv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s%d.h5', srsTvIdx, srsTvNameSlotIdx);
            end

            for eachSrsSlotIdx = srsSlotIdx
                if mod(eachSrsSlotIdx, 10) == 3 % S slot, append ULMIX TV with SRS
                    % S slot previous TV is controlled by S_SLOT_CFG
                    DL_TV{eachSrsSlotIdx+1, idxCell} = unique([DL_TV{eachSrsSlotIdx+1, idxCell}, {ulmix_srs_tv_str}], 'stable');
                else  % U slot, replace existing SRS TV
                    % Remove existing ULMIX TV files and add new one
                    existingTvs = UL_TV{eachSrsSlotIdx+1, idxCell};
                    if ~iscell(existingTvs)
                        existingTvs = {existingTvs};
                    end
                    % Filter out existing ULMIX TV files
                    nonUlmixTvs = {};
                    for j = 1:length(existingTvs)
                        if ~contains(existingTvs{j}, 'ULMIX')
                            nonUlmixTvs{end+1} = existingTvs{j};
                        end
                    end
                    % Add the new ULMIX TV
                    UL_TV{eachSrsSlotIdx+1, idxCell} = [nonUlmixTvs, {ulmix_srs_tv_str}];
                end
            end

            % add BFW TV
            bfwTvDlTvIdx = currentRow{5};
            if bfwTvDlTvIdx ~= 0  % has binded DL BFW TV
                dlbfw_tv_str = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', bfwTvDlTvIdx);
                for bfwDlSlotIdx = currentRow{6}
                    if mod(bfwDlSlotIdx, 10) == 5  % 2nd U slot
                        UL_TV{bfwDlSlotIdx+1, idxCell} = [UL_TV{bfwDlSlotIdx+1, idxCell}, {dlbfw_tv_str}];
                    else
                        DL_TV{bfwDlSlotIdx+1, idxCell} = [DL_TV{bfwDlSlotIdx+1, idxCell}, {dlbfw_tv_str}];
                    end
                    channels{bfwDlSlotIdx+1} = unique([channels{bfwDlSlotIdx+1}, {'BFW_DL'}], 'stable');
                end
            end
            bfwTvUlTvIdx = currentRow{7};
            if bfwTvUlTvIdx ~= 0  % has binded UL BFW TV
                ulbfw_tv_str = sprintf('TVnr_%04d_gNB_FAPI_s0.h5', bfwTvUlTvIdx);
                for bfwUlSlotIdx = currentRow{8}
                    if mod(bfwUlSlotIdx, 10) == 3  % S slot
                        DL_TV{bfwUlSlotIdx+1, idxCell} = [DL_TV{bfwUlSlotIdx+1, idxCell}, {ulbfw_tv_str}];
                    else  % U slot
                        UL_TV{bfwUlSlotIdx+1, idxCell} = [UL_TV{bfwUlSlotIdx+1, idxCell}, {ulbfw_tv_str}];
                    end
                    channels{bfwUlSlotIdx+1} = unique([channels{bfwUlSlotIdx+1}, {'BFW_UL'}], 'stable');
                end
            end
        end
    end
end

% Override TV config for negative test cases or reuse TVs
% This section replaces specific TVs in designated slots/cells with custom override TVs
if ~isempty(override_cfg)
    for i = 1:size(override_cfg, 1)
        override_cfg_row = override_cfg(i, :);
        % override_base_pattern_num = override_cfg_row{2};  % only for record keeping
        override_cell_idx_array = override_cfg_row{3};     % Cell index or array of indices (0-based)
        override_tv_idx_base = override_cfg_row{4};        % Base TV index to use for replacement
        override_tv_idx_incre_ind = override_cfg_row{5};   % 0=same TV for all cells, 1=increment TV index
        override_slot_idx = override_cfg_row{6};           % Slot index (0-based)
        override_tv_name_slot_idx = override_cfg_row{7};   % Slot index to use in TV filename
        override_mix_tv_ind = override_cfg_row{8};         % 0=regular, 1=ULMIX, 2=DLMIX in TV name
        override_target_tv_dl_ind = override_cfg_row{9};   % 0=search for ULMIX to replace, 1=search for DLMIX to replace
        override_verbose = override_cfg_row{10};          % 1=print out replacedment details

        % Convert cell index to array if it's a single value
        if ~isvector(override_cell_idx_array)
            override_cell_idx_array = [override_cell_idx_array];
        end

        % Filter override cells to only those that exist in the current pattern
        % Apply override to cells that exist, skip cells that don't (partial override)
        total_cells = sum(nCell);  % nCell can be a vector for some cases, so sum to get total

        % Loop through each cell in the array
        for cell_offset = 1:length(override_cell_idx_array)
            override_cell_idx_0based = override_cell_idx_array(cell_offset);  % Keep 0-based for comparison
            
            % Skip this cell if it doesn't exist in the current pattern
            if override_cell_idx_0based >= total_cells
                continue;  % Skip only this cell, not the entire override
            end
            
            override_cell_idx = override_cell_idx_0based + 1;  % Convert from 0-based to 1-based indexing for MATLAB
            
            % Calculate TV index based on increment indicator
            if override_tv_idx_incre_ind == 1
                override_tv_idx = override_tv_idx_base + (cell_offset - 1);
            else
                override_tv_idx = override_tv_idx_base;
            end

            % Generate TV filename based on type (ULMIX/DLMIX vs regular)
            if override_mix_tv_ind == 1
                tmp_tv_str = sprintf('TVnr_ULMIX_%04d_gNB_FAPI_s%d.h5', override_tv_idx, override_tv_name_slot_idx);
            elseif override_mix_tv_ind == 2
                tmp_tv_str = sprintf('TVnr_DLMIX_%04d_gNB_FAPI_s%d.h5', override_tv_idx, override_tv_name_slot_idx);
            else
                tmp_tv_str = sprintf('TVnr_%04d_gNB_FAPI_s%d.h5', override_tv_idx, override_tv_name_slot_idx);
            end

            % Determine which TV type to search for based on targetTvDlInd
            if override_target_tv_dl_ind == 0
                % Search for and replace ULMIX TV
                if ismember(override_slot_idx, [4, 5, 14, 15])
                    % UL slot: search in UL_TV
                    if override_cell_idx > size(UL_TV, 2)
                        continue;
                    end

                    ul_tv_cell = UL_TV{override_slot_idx+1, override_cell_idx};
                    if ~iscell(ul_tv_cell)
                        ul_tv_cell = {ul_tv_cell}; % Convert string to cell array
                    end
                    UL_TV_idx = find(cellfun(@(x) contains(x, 'ULMIX'), ul_tv_cell));
                    if ~isempty(UL_TV_idx)
                        original_tv = ul_tv_cell{UL_TV_idx(1)};
                        if iscell(UL_TV{override_slot_idx+1, override_cell_idx})
                            UL_TV{override_slot_idx+1, override_cell_idx}{UL_TV_idx(1)} = tmp_tv_str;
                        else
                            UL_TV{override_slot_idx+1, override_cell_idx} = tmp_tv_str;
                        end
                        if override_verbose == 1
                            fprintf('Pattern %.1f: Replaced UL TV %s with %s at slot %d, cell %d\n', patternNum, original_tv, tmp_tv_str, override_slot_idx, override_cell_idx-1);
                        end
                    else
                        UL_TV{override_slot_idx+1, override_cell_idx} = [UL_TV{override_slot_idx+1, override_cell_idx}, {tmp_tv_str}];
                        fprintf('Pattern %.1f: Added UL TV %s at slot %d, cell %d\n', patternNum, tmp_tv_str, override_slot_idx, override_cell_idx-1);
                    end
                else
                    % DL or S slot: search for ULMIX in DL_TV (for special slots with both DL and UL TVs)
                    if override_cell_idx > size(DL_TV, 2)
                        continue;
                    end

                    dl_tv_cell = DL_TV{override_slot_idx+1, override_cell_idx};
                    if ~iscell(dl_tv_cell)
                        dl_tv_cell = {dl_tv_cell}; % Convert string to cell array
                    end
                    ULMIX_TV_idx = find(cellfun(@(x) contains(x, 'ULMIX'), dl_tv_cell));
                    if ~isempty(ULMIX_TV_idx)
                        original_tv = dl_tv_cell{ULMIX_TV_idx(1)};
                        if iscell(DL_TV{override_slot_idx+1, override_cell_idx})
                            DL_TV{override_slot_idx+1, override_cell_idx}{ULMIX_TV_idx(1)} = tmp_tv_str;
                        else
                            DL_TV{override_slot_idx+1, override_cell_idx} = tmp_tv_str;
                        end
                        if override_verbose == 1
                            fprintf('Pattern %.1f: Replaced ULMIX TV %s with %s at slot %d, cell %d\n', patternNum, original_tv, tmp_tv_str, override_slot_idx, override_cell_idx-1);
                        end
                    else
                        DL_TV{override_slot_idx+1, override_cell_idx} = [DL_TV{override_slot_idx+1, override_cell_idx}, {tmp_tv_str}];
                        if override_verbose == 1
                            fprintf('Pattern %.1f: Added ULMIX TV %s at slot %d, cell %d\n', patternNum, tmp_tv_str, override_slot_idx, override_cell_idx-1);
                        end
                    end
                end
            else
                % Search for and replace DLMIX TV
                if override_cell_idx > size(DL_TV, 2)
                    continue;
                end

                dl_tv_cell = DL_TV{override_slot_idx+1, override_cell_idx};
                if ~iscell(dl_tv_cell)
                    dl_tv_cell = {dl_tv_cell}; % Convert string to cell array
                end
                DL_TV_idx = find(cellfun(@(x) contains(x, 'DLMIX'), dl_tv_cell));
                if ~isempty(DL_TV_idx)
                    original_tv = dl_tv_cell{DL_TV_idx(1)};
                    if iscell(DL_TV{override_slot_idx+1, override_cell_idx})
                        DL_TV{override_slot_idx+1, override_cell_idx}{DL_TV_idx(1)} = tmp_tv_str;
                    else
                        DL_TV{override_slot_idx+1, override_cell_idx} = tmp_tv_str;
                    end
                    if override_verbose == 1
                        fprintf('Pattern %.1f: Replaced DL TV %s with %s at slot %d, cell %d\n', patternNum, original_tv, tmp_tv_str, override_slot_idx, override_cell_idx-1);
                    end
                else
                    DL_TV{override_slot_idx+1, override_cell_idx} = [DL_TV{override_slot_idx+1, override_cell_idx}, {tmp_tv_str}];
                    if override_verbose == 1
                        fprintf('Pattern %.1f: Added DL TV %s at slot %d, cell %d\n', patternNum, tmp_tv_str, override_slot_idx, override_cell_idx-1);
                    end
                end
            end
        end
    end
end

LP = [];
bandwidth_num = length(nCell);  
cell_idx = 1;
for b = 1:bandwidth_num
    for idxCell = 1:nCell(b)
        DL_TV_1 = cellstr(DL_TV{1, cell_idx});
        LP.Cell_Configs{cell_idx} = DL_TV_1{1}; 
        cell_idx = cell_idx + 1;
    end
end
if(exist('UL_TV_1'))
    [LP.UL_Cell_Configs{1:sum(nCell)}] = deal(UL_TV_1{1:sum(nCell)});
end

% add static harq proc id config
if (ismember(patternNum, [69, 69.1, 69.2, 69.3, 69.4, 71, 73, 75, 77, 79, 79.1, 79.2, 81.1, 81.2, 81.3, 81.4, 83.1, 83.2, 83.3, 83.4, 85, 87, 89, 91, 101, 101.1, 102, 102.1, 201]))
    LP.config_static_harq_proc_id = 1.0;
end

LP = init_launchPattern(LP, nSlot, sum(nCell));

LP = gen_launchPattern(LP, sum(nCell), nSlot, DL_TV, UL_TV, channels, patternNum);


LPfilename = sprintf('F08');
for b = 1:length(nCell)
    LPfilename = sprintf(strcat(LPfilename, "_%dC"), nCell(b));
end
LPfilename = sprintf(strcat(LPfilename, "_%02d"), floor(patternNum));

if mod(patternNum*10,10) ~= 0
    sub_type = mod(patternNum*10,10);
    LPfilename = sprintf(strcat(LPfilename, "%s"), char(sub_type+'a'-1));
end

tvDirName = 'GPU_test_input';
[status,msg] = mkdir(tvDirName);

TVname = sprintf('launch_pattern_%s', LPfilename);
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

function LP = gen_launchPattern(LP, nCell, nSlot, DL_TV, UL_TV, channels, patternNum)

for idxCell = 1:nCell
    for slot = 1:nSlot
        slotInFrame = mod(slot,20);
        if ismember(slotInFrame, [5,6,15,16])
            % Handle empty or numeric UL_TV entries
            if isempty(UL_TV{slot, idxCell}) || isnumeric(UL_TV{slot, idxCell})
                LP.SCHED{slot}.config{idxCell}.channels = {};
            elseif iscell(UL_TV{slot, idxCell})
                LP.SCHED{slot}.config{idxCell}.channels = UL_TV{slot, idxCell};
            else
                LP.SCHED{slot}.config{idxCell}.channels = cellstr(UL_TV{slot, idxCell});
            end
            LP.SCHED{slot}.config{idxCell}.type = channels{slot};
        else
            % Handle empty or numeric DL_TV entries (e.g., special slots with no DL channels)
            if isempty(DL_TV{slot, idxCell}) || isnumeric(DL_TV{slot, idxCell})
                LP.SCHED{slot}.config{idxCell}.channels = {};
            elseif iscell(DL_TV{slot, idxCell})
                LP.SCHED{slot}.config{idxCell}.channels = DL_TV{slot, idxCell};
            else
                LP.SCHED{slot}.config{idxCell}.channels = cellstr(DL_TV{slot, idxCell});
            end
            LP.SCHED{slot}.config{idxCell}.type = channels{slot};
        end
    end
end

return
