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

function [nComp, errCnt, nTV, detErr] = testCompGenTV_pdsch(caseSet, compTvMode, subSetMod, relNum)

tic;
if nargin == 0
    caseSet = 'full';
    compTvMode = 'both';
    subSetMod = [0, 1];
    relNum = 10000;
elseif nargin == 1
    compTvMode = 'both';
    subSetMod = [0, 1];
    relNum = 10000;
elseif nargin == 2
    subSetMod = [0, 1];
    relNum = 10000;
elseif nargin == 3
    relNum = 10000;
end

switch compTvMode
    case 'both'
        genTV = 1;
        testCompliance = 1;
    case 'genTV'
        genTV = 1;
        testCompliance = 0;
    case 'testCompliance'
        genTV = 0;
        testCompliance = 1;
    otherwise
        error('compTvMode is not supported...\n');
end

selected_TC = [3201:3999];
if relNum == 2240
    disabled_TC = [3263, 3270, 3300, 3338, 3339, 3342, 3344, 3356, 3362];
else
    disabled_TC = [3270, 3300, 3342, 3362];
end
[~,TcIdx] = ismember(disabled_TC, selected_TC);
selected_TC(TcIdx) = [];

compact_TC = [3001:3999];
full_TC = [3001:3999];

if isnumeric(caseSet)
    TcToTest = caseSet;
else    
    switch caseSet
        case 'compact'
            TcToTest = compact_TC;
        case 'full'
            TcToTest = full_TC;
        case 'selected'
            TcToTest = selected_TC;
        otherwise
            error('caseSet is not supported...\n');
    end
end      
        
% The cases commented out with % % are supported by nrSim but not supported by cuPHY
CFG = {... 
% change only one parameter from the base case        
% TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 nAnt slotIdx
 3201,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % base case
 3202,  0,       28,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; %mcsTable = 0
 3203,  2,       28,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; %mcsTable = 2
 3204,  1,       27,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; %mcs = 27
 3205,  1,        1,   2,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; %nl = 2
 3206,  1,        1,   3,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; %nl = 3
 3207,  1,       27,   4,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; %nl = 4 
 3208,  1,        1,   1,  1,     1,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; %rb0, Nrb
 3209,  1,        1,   1,  0,   273,  0,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; %sym0 = 0
 3210,  1,        1,   1,  0,   273,  1,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; %sym0 = 1
 3211,  1,        1,   1,  0,   273,  3,     11,     0,    0,    273,    0,    0,      0,     3,      1,     0,      0,      2,   0,   4,     0; %sym0 = 3, dmrs0 = 3 
 3212,  1,        1,   1,  0,   273,  2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; %Nsym = 3
 3213,  1,        1,   1,  0,   273,  0,     14,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; %Nsym = 14
 3214,  1,        1,   1,  0,   273,  0,     14,     1,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,  16,     0; %SCID = 1
 3215,  1,        1,   1,  0,    73,  2,     12,     0,   50,    100,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; %BWP0, nBW
 3216,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,   41,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; %RNTI > 0
 3217,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; %addPos = 1
 3218,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; %addPos = 2
 3219,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     3,      0,      2,   0,   4,     0; %addPos = 3
 3220,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,    211,     2,      1,     0,      0,      2,   0,   4,     0; %datScId > 0
 3221,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     3,      2,     0,      0,      2,   0,   4,     0; %dmrs0 = 3, maxLen = 2
 3222,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      2,     0,      0,      2,   0,   4,     0; %dmrs0 = 2, maxLen = 2
 3223,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,     13,      2,   0,   4,     0; %dmrsScId > 0
 3224,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      1,   0,   4,     0; % nCdm = 1 
 3225,  1,        1,   4,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      2,     0,      0,      2,   2,   8,     0; %port0 = 2, nAnt = 8
 3226,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   1,     0; %nAnt = 1
 3227,  1,        1,   2,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   2,     0; %nAnt = 2
 3228,  1,        1,   4,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,  16,     0; %nAnt = 16
 3229,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     3; %slotIdx > 0
 3230,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    1,      0,     2,      1,     0,      0,      2,   0,   4,     0; % rvIdx = 1
 3231,  0,        1,   1,  0,     3,  2,      6,     0,    0,    273,    0,    2,      0,     2,      1,     0,      0,      2,   0,   4,     0; % rvIdx = 2, from 3014
 3232,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    3,      0,     2,      1,     0,      0,      2,   0,   4,     0; % rvIdx = 3
 3233,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % FDM
 3234,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % FDM
 3235,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % FDM
 3236,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % CDM
 3237,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % CDM
 3238,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % CDM
 3239,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % FDM+CDM
 3240,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % FDM+CDM
 3241,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      2,     0,      0,      2,   0,   4,     0; % FDM+CDM
 3242,  2,       15,   1,  0,   273,  2,      3,     0,    0,    273,    0,    1,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % rvIdx = 1, BGN = 1, from 3139
 3243,  1,        2,   1,  0,   273,  2,     12,     0,    0,    273,    0,    2,      0,     2,      1,     0,      0,      2,   0,   4,     0; % rvIdx = 2, BGN = 1
 3244,  1,        2,   1,  0,   273,  2,     12,     0,    0,    273,    0,    3,      0,     2,      1,     0,      0,      2,   0,   4,     0; % rvIdx = 3, BGN = 1
 3245,  1,        1,   1, 10,    54,  2,     12,     0,    0,    106,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mu = 0, dlGridSize = 106
 3246,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % data/dmrs for Ueg (FDM) 
 3247,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % data/dmrs for Ueg (FDM+CDM)
 3248,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % precoding (W14)
 3249,  1,        1,   2,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % precoding (W24)
 3250,  1,        1,   3,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % precoding (W34)
 3251,  1,        1,   4,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % precoding (W44)
 3252,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   2,     0; % precoding (W12)
 3253,  1,        1,   2,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   2,     0; % precoding (W22)
 3254,  1,        1,   2,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % precoding (W24 + W14)
 3255,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mapping type B
 3256,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mapping type B
 3257,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mapping type B
 3258,  1,        1,   2,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % precoding (W24 + noPrcd)
 3259,  1,        1,   1,  0,    20,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % precoding (mixed mod/nl)
 3260,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % precoding + csirs
 3261,  1,        1,   1, 10,    20,  2,     12,     0,  100,     50,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % refPoint = 1
 3262,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % TxPower
 3263,  1,       27,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % modComp
 3264,  1,        1,   1,  0,    20,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % precoding (mixed mod/nl/nPort)
 3265,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % TxPower with 2 UEs
 3266,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % diff rv with same tb config
 3267,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % diff rv with diff tb config
 3268,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      2,     0,      0,      2,   0,   4,     0; % 8+5 layers
 3269,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      2,     0,      0,      2,   0,   4,     0; % 16+10 layers
 % 3270,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      2,     0,      0,      2,   0,   4,     0; % BF with 32-ant 
 3271,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      1,   1,   2,     0; % nCdm = 1. nl = 1 
 3272,  1,        1,   2,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      1,   0,   2,     0; % nCdm = 1. nl = 2 
 3273,  1,       27,   2,  0,    73,  2,     12,     0,    0,    273,    0,    0,      0,     2,      2,     1,      0,      1,   0,   4,     0; % nCdm = 1. nl = 2 
 3274,  1,        1,   1,  0,   273,  0,      4,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      1,   0,   4,     0; % nCdm = 1, nSym = 4
 3275,  1,        1,   1,  0,   273,  0,     10,     0,    0,    273,    0,    0,      0,     2,      1,     1,      0,      1,   0,   4,     0; % nCdm = 1, nSym = 10
 3276,  1,        1,   1,  0,   273, 10,      4,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      1,   0,   4,     0; % nCdm = 1, mapping type B 
 3277,  0,        1,   1,  0,    3,   2,      6,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % BG=2, K not divisible by 8 or 32
 3278,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 4-BWP
 % resource allocation type 0
% TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 nAnt slotIdx
 3279,  1,        1,   1,  0,    73,  2,     12,     0,   50,    100,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; %BWP0, nBW, copied from 3215
 3280,  1,        1,   1, 10,    20,  2,     12,     0,  100,     50,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % refPoint = 1, copied from 3261
 3281,  1,       27,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % modComp, copied from 3263
 3282,  1,        1,   1,  0,   273,  0,      4,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      1,   0,   4,     0; % nCdm = 1, nSym = 4, copied from 3274
 3283,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % with two segments far away to each other
 3284,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % with max number of segments
 3285,  1,        1,   1,  0,    30,  2,     12,     0,   30,     30,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % <36 PRBs
 % additional TVs for nCdm = 1
 3286,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      1,   0,   4,     0; % FDM nCdm = 1 and 2
 3287,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      1,   0,   4,     0; % FDM nCdm = 1 and 2 and precoding
 3288,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      1,   0,   4,     0; % nCdm = 1 precoding (mixed mod/nl/nPort)
 % resource allocation type 0 with precoding
 3289,  1,        7,   4,  0,    73,  2,     12,     0,   50,    100,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; %BWP0, nBW, copied from 3215, change Qm=4, Nl=4
 3290,  1,        17,  1, 10,    20,  2,     12,     0,  100,     50,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % refPoint = 1, copied from 3261, changed to Qm=6
 3291,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % modComp, copied from 3263
 3292,  1,        1,   1,  0,   273,  0,      4,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      1,   0,   4,     0; % nCdm = 1, nSym = 4, copied from 3274
 3293,  1,        27,  1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % with two segments far away to each other
 3294,  1,        1,   2,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % with max number of segments, changed to Nl=2
 3295,  1,        1,   3,  0,    30,  2,     12,     0,   30,     30,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % <36 PRBs and 3 layers
 % testModel
 3296,   1,       1,   1,  3,   270,  0,     14,     0,    0,    273,    0,    0,      1,     2,      1,     1,      1,      1,   0,   4,     0; % testModel
 % MCS >= 28(29)
 3297,  1,       28,   1,  0,   270,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % base case
 % max Er
 3298,  2,        0,   2,  0,   247,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 1st-tx
 3299,  2,       29,   4,  0,   242,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % re-tx Er ~= 256k
 3300,  2,       29,   4,  0,   270,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % re-tx negtive TC with Er > 256k
% change multiple parameter from the base case        
% TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 nAnt slotIdx  nUE
% test MCS/CB size
 3301,  0,        0,   1,  0,     1,  2,      3,     0,    0,    273,   35,    0,     21,     2,      1,     0,     13,      2,   0,   1,     0; % table=0, min CB
 3302,  0,       28,   4,  0,   273,  0,     14,     0,    0,    273,   35,    0,     21,     2,      1,     0,     13,      2,   0,   4,     0; % table=0, max CB
 3303,  1,        0,   1,  0,     1,  2,      3,     0,    0,    273,   35,    0,     21,     2,      1,     0,     13,      2,   0,   1,     0; % table=1, min CB
 3304,  1,       27,   4,  0,   273,  0,     14,     0,    0,    273,   35,    0,     21,     2,      1,     0,     13,      2,   0,   4,     0; % table=1, max CB
 3305,  2,        0,   1,  0,     1,  2,      3,     0,    0,    273,   35,    0,     21,     2,      1,     0,     13,      2,   0,   1,     0; % table=2, min CB
 3306,  2,       28,   4,  0,   273,  0,     14,     0,    0,    273,   35,    0,     21,     2,      1,     0,     13,      2,   0,   4,     0; % table=2, max CB
% test pdsch/dmrs symbol location 
 3307,  1,        1,   1,  0,   273,  0,      3,     0,    0,    273,   35,    0,     21,     2,      1,     0,     13,      2,   0,   4,     0; % dmrs0=2, addPos, maxLen
 3308,  1,        1,   1,  0,   273,  1,      3,     0,    0,    273,   35,    0,     21,     2,      2,     0,     13,      2,   0,   4,     0; % dmrs0=2, addPos, maxLen
 3309,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,   35,    0,     21,     2,      1,     2,     13,      2,   0,   4,     0; % dmrs0=2, addPos, maxLen
 3310,  1,        1,   1,  0,   273,  2,     11,     0,    0,    273,   35,    0,     21,     2,      2,     1,     13,      2,   0,   4,     0; % dmrs0=2, addPos, maxLen
 3311,  1,        1,   1,  0,   273,  0,      5,     0,    0,    273,   35,    0,     21,     3,      1,     0,     13,      2,   0,   4,     0; % dmrs0=3, addPos, maxLen
 3312,  1,        1,   1,  0,   273,  1,      5,     0,    0,    273,   35,    0,     21,     3,      2,     0,     13,      2,   0,   4,     0; % dmrs0=3, addPos, maxLen
 3313,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,   35,    0,     21,     3,      1,     3,     13,      2,   0,   4,     0; % dmrs0=3, addPos, maxLen
 3314,  1,        1,   1,  0,   273,  3,     11,     0,    0,    273,   35,    0,     21,     3,      2,     1,     13,      2,   0,   4,     0; % dmrs0=3, addPos, maxLen
 % test nl/portIdx/nAnt
 3315,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,   35,    0,     21,     2,      1,     0,     13,      2,   0,   1,     0; %nl = 1, port0/nAnt
 3316,  1,        1,   2,  0,   273,  2,     12,     0,    0,    273,   35,    0,     21,     2,      1,     0,     13,      2,   0,   2,     0; %nl = 2, port0/nAnt
 3317,  1,        1,   3,  0,   273,  2,     12,     0,    0,    273,   35,    0,     21,     2,      1,     0,     13,      2,   0,   4,     0; %nl = 3, port0/nAnt
 3318,  1,        1,   3,  0,   273,  2,     12,     0,    0,    273,   35,    0,     21,     2,      1,     0,     13,      2,   1,   8,     0; %nl = 3, port0/nAnt
 3319,  1,        1,   4,  0,   273,  2,     12,     0,    0,    273,   35,    0,     21,     2,      1,     0,     13,      2,   0,  16,     0; %nl = 4, port0/nAnt   
 3320,  1,        1,   4,  0,   273,  2,     12,     0,    0,    273,   35,    0,     21,     2,      1,     0,     13,      2,   4,  16,     0; %nl = 4, port0/nAnt    
 % test LBRM
 3321,  1,       20,   4,  0,   273,  0,     14,     0,    0,    273,   35,    1,     21,     2,      1,     0,     13,      2,   0,   4,     0; % N > Nref, rvIdx = 2
 3322,  1,       20,   4,  0,   273,  0,     14,     0,    0,    273,   35,    3,     21,     2,      1,     0,     13,      2,   0,   4,     0; % N > Nref, rvIdx = 4
 % test Xtf_remap (overlap with CSI-RS) 
 3323,  1,        1,   1,  0,   273,  0,     14,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % full PRBs, full syms 
 3324,  1,        1,   1,  0,   273,  0,     10,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % full PRBs, non-full syms 
 3325,  1,        1,   2, 20,    80,  0,     14,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % non-full PRBs, full syms
 3326,  1,        1,   2, 20,    80,  0,     14,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % non-full PRBs, full syms
 3327,  1,        1,   4, 20,    80,  0,     14,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   8      0; % non-full PRBs, full syms
 3328,  1,        1,   4, 20,    80,  0,     14,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   8,     0; % non-full PRBs, full syms
 3329,  1,        1,   4, 20,    80,  0,     14,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   8,     0; % multiple CSI-RS
 3330,  1,        1,   4, 20,    80,  0,     14,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   8,     0; % multiple CSI-RS multiple PDSCH
 3331,  1,        1,   4, 20,    80,  0,     14,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   8,     0; % TRS 
 3332,  1,        1,   1,  0,   273,  0,     14,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % freqDensity = 0
 3333,  1,        1,   1,  0,   273,  0,     10,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % freqDensity = 1 
  % multiple UEs
 3334,  1,        1,   1,  0,   273,  0,     14,     0,    0,    273,    0,    0,      0,     2,      2,     0,      0,      2,   0,   8,     0; % 8 UEs FDM same nl
 3335,  1,        1,   1,  0,   273,  0,     14,     0,    0,    273,    0,    0,      0,     2,      2,     0,      0,      2,   0,   8,     0; % 8 UEs FDM mixed nl
 3336,  1,        1,   1,  0,   273,  0,     14,     0,    0,    273,    0,    0,      0,     2,      2,     0,      0,      2,   0,   8,     0; % 16 UEs FDM
 3337,  1,        1,   1,  0,   273,  0,     14,     0,    0,    273,    0,    0,      0,     2,      2,     0,      0,      2,   0,   8,     0; % 16 UEs FDM + CDM 
 % test Xtf_remap with resource allocation type 0
 %TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 nAnt slotIdx  nUE
 3338,  1,        1,   4, 20,    80,  0,     14,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   8,     0; % multiple CSI-RS multiple PDSCH, copied from 3330
 3339,  1,        1,   4, 20,    80,  0,     14,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % multiple CSI-RS multiple PDSCH with precoding, copied from 3330
 3340,  1,        1,   4, 20,    80,  0,     14,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % multiple UE groups w/out CSI-RS
 3341,  1,        1,   4, 20,    80,  0,     14,     0,    0,     80,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % multiple UE groups w/ mix of resource allocation type 0 & 1
 % max Er
 3342,  2,       29,   4,  0,   270,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % re-tx negtive TC with Er > 256k 2-UE
 % for S-slot multi-cell test
 3343,  1,        1,   1,  0,   273,  2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % copy from 3212
 % MU-MIMO
 %TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 nAnt slotIdx  nUE
 3344,  1,       27,   2,  20,  253,  2,     12,     0,    0,    273,   211,   0,     41,     2,      2,     1,     41,      2,   0,   8,     0; % nl = 2+2+2+2
 % nCdm=1 with different CSI-RS 
 3350,  1,        1,   1,  20,   80,  2,     12,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      1,   0,   4,     0; % CSI-RS between 2 DMRS symbols
 3351,  1,        1,   1,  20,   80,  2,      7,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      1,   0,   4,     0; % starting on symbol 0 w/ DMRS Type B
 3352,  1,       27,   1, 100,   80,  2,     12,     0,    0,    273,    1,    0,     41,     3,      1,     2,     41,      1,   0,   4,     0; % extra DMRS
 3353,  1,       17,   1,  50,  200,  0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     3,     41,      1,   0,   4,     0; % all 14 symbols and additional DMRS w/ CSI-RS at end
 
% testModel where TB size as computed by REs * Qm* Nl % 8 != 0
3354,   1,       1,   1,  0,   3,  0,     4,     0,    0,    273,    0,    0,      1,     2,      1,     1,      1,      1,   0,   4,     0; % testModel where TB size in bits %8 != 0
3355,   1,       1,   1,  0,   273,  0,     5,     0,    0,    273,    0,    0,      1,     2,      1,     1,      1,      1,   0,   4,     0; % testModel where TB size in bits %8 != 0 and TB size in bits > 25344

% nCdm=1 + modComp
 3356,  1,       27,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      1,   0,   4,     0; % copy of 3263 w/ nCdm=1
 3357,  1,       17,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % modComp Qm=6
 3358,  1,        7,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % modComp Qm=4
 3359,  1,       27,   1,  0,   136,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % modComp w/ partial BW
 3360,  1,       27,   2,  0,   253,  2,     12,     0,    0,    273,   211,   0,     41,     2,      2,     1,     41,      2,   0,  16,     0; % modComp w/ mixed modulation

 % MU-MIMO > 16 layers 
 3362,  1,       27,   2,  20,  253,  2,     12,     0,    0,    273,   211,   0,     41,     2,      2,     1,     41,      2,   0,   32,     0; % nl 32 = 4 x 8

 % different BW
 % mu = 1
% TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 nAnt slotIdx
 3401,  1,        1,   1,  0,    11,  2,     12,     0,    0,     11,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 5 MHz
 3402,  1,        1,   1,  0,    24,  2,     12,     0,    0,     24,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 10 MHz
 3403,  1,        1,   1,  0,    38,  2,     12,     0,    0,     38,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 15 MHz
 3404,  1,        1,   1,  0,    51,  2,     12,     0,    0,     51,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 20 MHz
 3405,  1,        1,   1,  0,    65,  2,     12,     0,    0,     65,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 25 MHz
 3406,  1,        1,   1,  0,    78,  2,     12,     0,    0,     78,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 30 MHz
 3407,  1,        1,   1,  0,   106,  2,     12,     0,    0,    106,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 40 MHz
 3408,  1,        1,   1,  0,   133,  2,     12,     0,    0,    133,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 50 MHz
 3409,  1,        1,   1,  0,   162,  2,     12,     0,    0,    162,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 60 MHz
 3410,  1,        1,   1,  0,   189,  2,     12,     0,    0,    189,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 70 MHz
 3411,  1,        1,   1,  0,   217,  2,     12,     0,    0,    217,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 80 MHz
 3412,  1,        1,   1,  0,   245,  2,     12,     0,    0,    245,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 90 MHz
 3413,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 100 MHz
 % mu = 0
 3414,  1,        1,   1,  0,    25,  2,     12,     0,    0,     25,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 5 MHz
 3415,  1,        1,   1,  0,    52,  2,     12,     0,    0,     52,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 10 MHz
 3416,  1,        1,   1,  0,    79,  2,     12,     0,    0,     79,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 15 MHz
 3417,  1,        1,   1,  0,   106,  2,     12,     0,    0,    106,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 20 MHz
 3418,  1,        1,   1,  0,   133,  2,     12,     0,    0,    133,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 25 MHz
 3419,  1,        1,   1,  0,   160,  2,     12,     0,    0,    160,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 30 MHz
 3420,  1,        1,   1,  0,   216,  2,     12,     0,    0,    216,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 40 MHz
 3421,  1,        1,   1,  0,   270,  2,     12,     0,    0,    270,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 50 MHz
 % additional BW test cases
 3422,  1,       27,   4,  0,   100,  2,     12,     0,    0,    100,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 40 MHz
 3423,  1,       27,   4,  0,    50,  2,     12,     0,    0,    100,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 40 MHz
 
 % Invalid PDSCH FAPI PDU
 % CFG is for generating I/Q samples, PDSCH params will be overwrittin using INVALID_CFG
 3501,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid pduBitmap = 1
 3502,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid BWPSize = 0
 3503,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid BWPStart = 275
 3504,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid SubcarrierSpacing = 5
 3505,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid CyclicPrefix = 2
 3506,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid NrOfCodewords = 2
 3507,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid targetCodeRate = 0
 3508,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid qamModOrder = 10
 3509,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid mcsIndex = 32
 3510,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid mcsTable = 5
 3511,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid rvIndex = 4
 3512,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid nrOfLayers = 5
 3513,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid refPoint = 2
 3514,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid DmrsSymbPos = 0
 3515,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid dmrsConfigType = 2
 3516,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid SCID = 2
 3517,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid numDmrsCdmGrpsNoData = 3
 3518,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid resourceAlloc = 2
 3519,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid rbStart = 275
 3520,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid rbSize = 276
 3521,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid VRBtoPRBMapping = 3
 3522,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid StartSymbolIndex = 14
 3523,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid NrOfSymbols = 15
 3524,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Cfg from base case 3201, overwrite PDU with invalid numPRGs/prgSize (numPRGs=17, prgSize=16)
    
 % requested TCs
 % numDmrsCdmGrpsNoData and dmrsPort
 3801,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      1,   0,   4,     0; % 3224, nCdm = 1
 3802,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 3224, nCdm = 2
 3803,  1,        1,   2,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      1,   0,   4,     0; % 3272, nCdm = 1. nl = 2, portIdx = [0 1]
 3804,  1,        1,   2,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 3272, nCdm = 2. nl = 2, portIdx = [0 1]
 3805,  1,        1,   2,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % 3272, nCdm = 2. nl = 2, portIdx = [0 2]

 % specific configuration for TV generation 
% TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 nAnt slotIdx
 3901,  0,        5,   1,  0,    31,  2,     5,     0,  108,     48, 65535,    0,     41,     2,      1,     0,     41,      2,   0,   2,     2; %demo_coreset0
 3902,  0,        5,   1,  0,    48,  2,     5,     0,  108,     48,   211,    0,     41,     2,      1,     0,     41,      2,   0,   2,     1; %demo_msg2
 3903,  0,        5,   1,  0,    48,  2,     5,     0,  108,     48, 20000,    0,     41,     2,      1,     0,     41,      2,   0,   2,    11; %demo_msg4
 3904,  1,       27,   1,148,   125,  2,    12,     0,    0,    273, 20000,    0,     41,     2,      1,     0,     41,      2,   0,   2,     1; %demo_traffic_dl
% 3905,  0,        6,   1,  0,     6,  2,     5,     0,    0,    273,     0,    0,      0,     2,      1,     0,      0,      2,   0,   1,     0; %bug3306948;
 3906,  0,       28,   2,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   2,     0; %2 layers MCS28
 
 
 % requested TCs 
 % PDSCH + CSIRS
 3907,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % 3260 w/o precoding
 % sym0 = 2, nSym = 12 
 % 1-layer 2-dmrs
 3908,  0,        1,   1,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=0, QPSK
 3909,  0,       16,   1,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=0, 16QAM 
 3910,  0,       28,   1,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=0, 64QAM
 3911,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=1, QPSK 
 3912,  1,       10,   1,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=1, 16QAM
 3913,  1,       19,   1,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=1, 64QAM
 3914,  1,       27,   1,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=1, 256QAM
 % 1-layer 3-dmrs
 3915,  0,        1,   1,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=0, QPSK
 3916,  0,       16,   1,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=0, 16QAM 
 3917,  0,       28,   1,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=0, 64QAM
 3918,  1,        1,   1,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=1, QPSK 
 3919,  1,       10,   1,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=1, 16QAM
 3920,  1,       19,   1,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=1, 64QAM
 3921,  1,       27,   1,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=1, 256QAM 
 % 2-layer 2-dmrs
 3922,  0,        1,   2,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=0, QPSK
 3923,  0,       16,   2,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=0, 16QAM 
 3924,  0,       28,   2,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=0, 64QAM
 3925,  1,        1,   2,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=1, QPSK 
 3926,  1,       10,   2,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=1, 16QAM
 3927,  1,       19,   2,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=1, 64QAM
 3928,  1,       27,   2,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=1, 256QAM
 % 2-layer 3-dmrs
 3929,  0,        1,   2,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=0, QPSK
 3930,  0,       16,   2,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=0, 16QAM 
 3931,  0,       28,   2,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=0, 64QAM
 3932,  1,        1,   2,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=1, QPSK 
 3933,  1,       10,   2,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=1, 16QAM
 3934,  1,       19,   2,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=1, 64QAM
 3935,  1,       27,   2,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=1, 256QAM  
% 4-layer 2-dmrs
 3936,  0,        1,   4,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=0, QPSK
 3937,  0,       16,   4,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=0, 16QAM 
 3938,  0,       28,   4,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=0, 64QAM
 3939,  1,        1,   4,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=1, QPSK 
 3940,  1,       10,   4,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=1, 16QAM
 3941,  1,       19,   4,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=1, 64QAM
 3942,  1,       27,   4,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=1, 256QAM
 % 4-layer 3-dmrs
 3943,  0,        1,   4,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=0, QPSK
 3944,  0,       16,   4,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=0, 16QAM 
 3945,  0,       28,   4,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=0, 64QAM
 3946,  1,        1,   4,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=1, QPSK 
 3947,  1,       10,   4,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=1, 16QAM
 3948,  1,       19,   4,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=1, 64QAM
 3949,  1,       27,   4,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     2,      0,      2,   0,   4,     0; % mcsTable=1, 256QAM   
 % sym0 = 2, nSym = 4
 % 1-layer 1-dmrs
 3950,  0,        1,   1,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=0, QPSK
 3951,  0,       16,   1,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=0, 16QAM 
 3952,  0,       28,   1,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=0, 64QAM
 3953,  1,        1,   1,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=1, QPSK 
 3954,  1,       10,   1,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=1, 16QAM
 3955,  1,       19,   1,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=1, 64QAM
 3956,  1,       27,   1,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=1, 256QAM   
 % 2-layer 1-dmrs
 3957,  0,        1,   2,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=0, QPSK
 3958,  0,       16,   2,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=0, 16QAM 
 3959,  0,       28,   2,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=0, 64QAM
 3960,  1,        1,   2,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=1, QPSK 
 3961,  1,       10,   2,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=1, 16QAM
 3962,  1,       19,   2,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=1, 64QAM
 3963,  1,       27,   2,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=1, 256QAM   
 % 4-layer 1-dmrs
 3964,  0,        1,   4,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=0, QPSK
 3965,  0,       16,   4,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=0, 16QAM 
 3966,  0,       28,   4,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=0, 64QAM
 3967,  1,        1,   4,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=1, QPSK 
 3968,  1,       10,   4,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=1, 16QAM
 3969,  1,       19,   4,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=1, 64QAM
 3970,  1,       27,   4,  0,   273,  2,      4,     0,    0,    273,    1,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % mcsTable=1, 256QAM  
 % 6 UEs, MCS27, 4layer
 3971,  1,       27,   4,  0,   273,  2,     12,     0,    0,    273,    1,    0,      0,     2,      1,     1,      0,      2,   0,   4,     0; % mcsTable=1, 256QAM
 % HARQ mcsTable = 0, McsIdx = 29
 3972,  0,        1,   4,  0,   273,  0,     14,     0,    0,    273,    1,    0,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-1
 3973,  0,       29,   4,  0,   273,  0,     14,     0,    0,    273,    1,    2,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-2
 3974,  0,       29,   4,  0,   273,  0,     14,     0,    0,    273,    1,    3,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-3
 3975,  0,       29,   4,  0,   273,  0,     14,     0,    0,    273,    1,    1,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-4
 % HARQ mcsTable = 0, McsIdx = 30
 3976,  0,       16,   4,  0,   273,  0,     14,     0,    0,    273,    1,    0,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-1
 3977,  0,       30,   4,  0,   273,  0,     14,     0,    0,    273,    1,    2,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-2
 3978,  0,       30,   4,  0,   273,  0,     14,     0,    0,    273,    1,    3,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-3
 3979,  0,       30,   4,  0,   273,  0,     14,     0,    0,    273,    1,    1,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-4
 % HARQ mcsTable = 0, McsIdx = 31
 3980,  0,       28,   4,  0,   273,  0,     14,     0,    0,    273,    1,    0,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-1
 3981,  0,       31,   4,  0,   273,  0,     14,     0,    0,    273,    1,    2,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-2
 3982,  0,       31,   4,  0,   273,  0,     14,     0,    0,    273,    1,    3,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-3
 3983,  0,       31,   4,  0,   273,  0,     14,     0,    0,    273,    1,    1,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-4
 % HARQ mcsTable = 1, McsIdx = 28
 3984,  1,        1,   4,  0,   273,  0,     14,     0,    0,    273,    1,    0,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-1
 3985,  1,       28,   4,  0,   273,  0,     14,     0,    0,    273,    1,    2,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-2
 3986,  1,       28,   4,  0,   273,  0,     14,     0,    0,    273,    1,    3,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-3
 3987,  1,       28,   4,  0,   273,  0,     14,     0,    0,    273,    1,    1,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-4
 % HARQ mcsTable = 1, McsIdx = 29
 3988,  1,       10,   4,  0,   273,  0,     14,     0,    0,    273,    1,    0,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-1
 3989,  1,       29,   4,  0,   273,  0,     14,     0,    0,    273,    1,    2,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-2
 3990,  1,       29,   4,  0,   273,  0,     14,     0,    0,    273,    1,    3,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-3
 3991,  1,       29,   4,  0,   273,  0,     14,     0,    0,    273,    1,    1,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-4
 % HARQ mcsTable = 1, McsIdx = 30
 3992,  1,       19,   4,  0,   273,  0,     14,     0,    0,    273,    1,    0,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-1
 3993,  1,       30,   4,  0,   273,  0,     14,     0,    0,    273,    1,    2,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-2
 3994,  1,       30,   4,  0,   273,  0,     14,     0,    0,    273,    1,    3,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-3
 3995,  1,       30,   4,  0,   273,  0,     14,     0,    0,    273,    1,    1,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-4
 % HARQ mcsTable = 1, McsIdx = 31
 3996,  1,       27,   4,  0,   273,  0,     14,     0,    0,    273,    1,    0,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-1
 3997,  1,       31,   4,  0,   273,  0,     14,     0,    0,    273,    1,    2,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-2
 3998,  1,       31,   4,  0,   273,  0,     14,     0,    0,    273,    1,    3,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-3
 3999,  1,       31,   4,  0,   273,  0,     14,     0,    0,    273,    1,    1,     21,     2,      1,     0,     13,      2,   0,   4,     0; % tx-4
 
% TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 nAnt slotIdx
 3001,  1,        1,   1,  0,   273,  0,     6,     0,     0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;
 3002,  1,        7,   2,  1,   73,   1,     10,    0,     0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;
 3003,  1,        17,  3,  5,   173,  0,     7,     0,     0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;
 3004,  1,        27,  4,  0,   273,  2,     11,    0,     0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;
 3005,  0,        5,   1,  0,   26,   2,     11,    0,     0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;
 3006,  0,        5,   1,  0,   43,   2,     11,    0,     0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;
 3007,  0,        5,   1,  0,   1,    2,     3,     0,     0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;
 3008,  0,        5,   1,  0,   73,   2,     11,    0,     0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;
 3009,  0,        7,   1,  0,   22,   2,     5,     0,     0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;
 3010,  1,        7,   1,  0,   22,   2,     5,     0,     0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;
 3011,  1,        1,   1,  0,   273,  2,     6,     0,     0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;
 3012,  1,        7,   2,  1,   73,   2,     10,    0,     0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;
 3013,  1,        17,  3,  5,   173,  2,     7,     0,     0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;
 3014,  0,        1,   1,  0,   3,    2,     6,     0,     0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % Catch Ninfo=ceil() bug
 3015,  0,        1,   1,  0,   43,   2,     10,    0,     0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0; % Catch Ninfo using wrong LUT of codeRate*qam bug

% TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 nAnt slotIdx 
% Lifting size sweep
% 3016,  0,        0,   1,  0,   0,     2,      0,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=2
% 3017,  0,        0,   1,  0,   0,     2,      0,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=4
 3018,  0,        1,   1,  0,   1,     2,     10,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=8
 3019,  0,        6,   1,  0,   1,     2,      9,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=16
 3020,  0,        9,   1,  0,   1,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=32
 3021,  0,       21,   1,  0,   1,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=64
 3022,  0,       19,   1,  0,   3,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=128
 3023,  1,       20,   1,  0,   4,     2,     11,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=256
% 3024,  0,        0,   1,  0,   0,     2,      0,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=3
% 3025,  0,        0,   1,  0,   0,     2,      0,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=6
 3026,  0,        3,   1,  0,   1,     2,     11,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=12
 3027,  0,        7,   1,  0,   1,     2,     11,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=24
 3028,  0,       18,   1,  0,   1,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=48
 3029,  0,       20,   1,  0,   2,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=96
 3030,  0,       20,   1,  0,   4,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=192
 3031,  0,       22,   1,  0,   7,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=384
% 3032,  0,        0,   1,  0,   0,     2,      0,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=5
 3033,  0,        1,   1,  0,   1,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=10
 3034,  0,        5,   1,  0,   1,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=20
 3035,  0,       14,   1,  0,   1,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=40
 3036,  0,       18,   1,  0,   2,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=80
 3037,  0,       21,   1,  0,   3,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=160
 3038,  0,       22,   1,  0,   6,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=320
%  3039,  0,        0,   1,  0,   1,     2,      2,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=7
 3040,  0,        3,   1,  0,   1,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=14
 3041,  0,        8,   1,  0,   1,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=28
 3042,  0,       19,   1,  0,   1,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=56
 3043,  0,       22,   1,  0,   2,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=112
 3044,  1,       20,   1,  0,   3,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=224
% 3045,  0,        0,   1,  0,   0,     2,      0,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=9
 3046,  0,        5,   1,  0,   1,     2,     11,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=18
 3047,  0,       13,   1,  0,   1,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=36
 3048,  0,       22,   1,  0,   1,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=72
 3049,  1,       20,   1,  0,   2,     2,     11,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=144
 3050,  1,       20,   1,  0,   4,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=288
 3051,  0,        2,   1,  0,   1,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=11
 3052,  0,        6,   1,  0,   1,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=22
 3053,  0,       15,   1,  0,   1,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=44
 3054,  0,       19,   1,  0,   2,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=88
 3055,  1,       20,   1,  0,   3,     2,     10,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=176
 3056,  1,       20,   1,  0,   5,     2,     11,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=352
% 3057,  0,        0,   1,  0,   0,     2,      0,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=13
 3058,  0,        8,   1,  0,   1,     2,     11,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=26
 3059,  0,       21,   1,  0,   1,     2,     10,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=52
 3060,  0,       21,   1,  0,   2,     2,     12,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=104
 3061,  1,       20,   1,  0,   3,     2,     11,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=208
 3062,  0,        4,   1,  0,   1,     2,     11,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=15
 3063,  0,       12,   1,  0,   1,     2,      9,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=30
 3064,  1,       20,   1,  0,   1,     2,      8,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=60
 3065,  1,       20,   1,  0,   2,     2,     10,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=120
 3066,  1,       20,   1,  0,   4,     2,     10,    0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % Zc=240

 3067,  0,        0,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3068,  0,        1,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3069,  0,        2,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3070,  0,        3,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3071,  0,        4,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3072,  0,        5,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3073,  0,        6,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3074,  0,        7,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3075,  0,        8,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3076,  0,        9,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3077,  0,       10,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3078,  0,       11,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3079,  0,       12,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3080,  0,       13,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3081,  0,       14,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3082,  0,       15,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3083,  0,       16,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3084,  0,       17,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3085,  0,       18,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3086,  0,       19,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3087,  0,       20,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3088,  0,       21,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3089,  0,       22,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3090,  0,       23,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3091,  0,       24,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3092,  0,       25,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3093,  0,       26,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3094,  0,       27,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3095,  0,       28,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3096,  1,        0,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3097,  1,        1,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3098,  1,        2,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3099,  1,        3,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3100,  1,        4,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3101,  1,        5,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3102,  1,        6,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3103,  1,        7,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3104,  1,        8,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3105,  1,        9,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3106,  1,       10,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3107,  1,       11,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3108,  1,       12,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3109,  1,       13,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3110,  1,       14,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3111,  1,       15,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3112,  1,       16,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3113,  1,       17,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3114,  1,       18,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3115,  1,       19,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3116,  1,       20,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3117,  1,       21,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3118,  1,       22,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3119,  1,       23,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3120,  1,       24,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3121,  1,       25,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3122,  1,       26,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3123,  1,       27,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3124,  2,        0,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3125,  2,        1,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3126,  2,        2,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3127,  2,        3,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3128,  2,        4,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3129,  2,        5,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3130,  2,        6,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3131,  2,        7,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3132,  2,        8,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3133,  2,        9,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3134,  2,       10,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3135,  2,       11,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3136,  2,       12,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3137,  2,       13,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3138,  2,       14,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3139,  2,       15,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3140,  2,       16,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3141,  2,       17,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3142,  2,       18,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3143,  2,       19,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3144,  2,       20,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3145,  2,       21,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3146,  2,       22,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3147,  2,       23,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3148,  2,       24,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3149,  2,       25,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3150,  2,       26,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3151,  2,       27,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
 3152,  2,       28,   1,  0, 273,     2,      3,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % mcs sweep
% manually constructed case 
% TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 nAnt slotIdx 
 3153,  0,        0,   1,  0,  24,     2,     10,    0,     0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % for K_b = 9 coverage
 3154,  0,        1,   1,  0, 100,     2,     12,     0,    0,    273,    0,    0,      0,     2,      1,     0,      0,      2,   0,   4,     0;  % C = 2, Zc = 208

% TVs matching with BFW
% 32T32R
 3850, 1,        27,   1,  0,  273,  2,     12,     0,    0,    273,   211,   0,     41,     2,      2,     1,     41,      2,   0,   8,     0; % nl = 1 x 8UE
 3851, 1,        27,   1,  0,   78,  2,     12,     0,    0,     78,   211,   0,     41,     2,      2,     1,     41,      2,   0,   8,     0;
 3852, 1,        27,   1,  0,  133,  2,     12,     0,    0,    133,   211,   0,     41,     2,      2,     1,     41,      2,   0,   8,     0;
 3853, 1,        27,   1,  0,  273,  2,     12,     0,    0,    273,   211,   0,     41,     2,      2,     1,     41,      2,   0,   8,     0;
 3854, 1,        27,   1,  0,  273,  2,     12,     0,    0,    273,   211,   0,     41,     2,      2,     1,     41,      2,   0,   8,     0;
 3855, 1,        27,   1,  0,  78,   2,     12,     0,    0,    78,   211,   0,     41,     2,      2,     1,     41,      2,   0,   8,     0; %30 8 layers/PDUs for BFW
 3856, 1,        27,   1,  0,  133,  2,     12,     0,    0,    133,   211,   0,     41,     2,      2,     1,     41,      2,   0,   8,     0; %50 8 layers/PDUs for BFW
 3857, 1,        27,   1,  0,  273,  0,     12,     0,    0,    273,   211,   0,     41,     2,      2,     1,     41,      2,   0,   8,     0; % 100 MHz with CSI-RS for BFW, adapted from TC3323 
 3858, 1,        27,   1,  10,  60,  2,     12,     0,    0,    273,   211,   0,     41,     2,      2,     1,     41,      2,   0,   8,     0; % nl = 1 x 8UE
 3859, 1,        27,   1,  0,  273,  2,     12,     0,    0,    273,   211,   0,     41,     2,      2,     1,     41,      2,   0,   8,     0; % nl = 1 x 8UE

% TVs matching with BFW
% reserved range 3869:3900
% 64T64R
% TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 nAnt slotIdx 
 3869,  1,        27,   1,  0,  273,  2,     12,     0,    0,    273,   1,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % nl = 1 x 16UE + PRG size = 16
 3870,  1,        27,   1,  0,  273,  2,     12,     0,    0,    273,   1,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % nl = 1 x 16UE
 3871   1,        27,   1,  0,  273,  2,     12,     0,    0,    273,   1,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0;  % 1 layer (1 UE)
 3872   1,        27,   2,  0,  273,  2,     12,     0,    0,    273,   1,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0;  % 2 layer (1 UE)
 3873   1,        27,   1,  0,  273,  2,     12,     0,    0,    273, 211,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % 1+1 layers (2 UE)
 3874   1,        27,   1,  0,  273,  2,     12,     0,    0,    273, 211,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0;  % 2+2 layers (2 UE)
 3875   1,        27,   1,  0,  273,  2,     12,     0,    0,    273, 211,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % 1+1+1+1 layers (4 UE)
 3876   1,        27,   1,  0,  273,  2,     12,     0,    0,    273, 211,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % 2+2+2+2 layers (4 UE)
 3877   1,        27,   1,100,   60,  2,     12,     0,    0,    273,   1,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0;  % 1 layer (1 UE)
 3878   1,        27,   1,160,   64,  2,     12,     0,    0,    273,   2,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0;  % 1 layer (1 UE)
 3879   1,        27,   1,  0,  273,  2,     12,     0,    0,    273, 211,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % 2+2+2+2+2+2+2+2 layers (8 UE)
 3880   1,        27,   1,  0,  273,  2,     12,     0,    0,    273, 211,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % 4+4+4+4 layers (4 UE)
 3881   1,        27,   1,  0,  273,  2,     12,     0,    0,    273, 211,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % 8 layers (8 UE)
 
 3882   1,        27,   1,  0,  273,  2,     12,     0,    0,    273, 211,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % TBD
 3883   1,        27,   1,  0,  273,  2,     12,     0,    0,    273, 211,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % TBD
 3884   1,        27,   1,  0,  273,  2,     12,     0,    0,    273, 211,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % TBD
 3885   1,        27,   1,  0,  273,  2,     12,     0,    0,    273, 211,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % TBD
 3886   1,        27,   1,  0,  273,  2,     12,     0,    0,    273, 211,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % TBD
 3887   1,        27,   1,  0,  273,  2,     12,     0,    0,    273, 211,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % TBD
 3888   1,        27,   1,  0,  273,  2,     12,     0,    0,    273, 211,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % TBD
 3889   1,        27,   2,160,   64,  2,     12,     0,    0,    273,   2,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % 2 layer (1 UE) prgSize=4
 3890   1,        27,   1,  0,  273,  2,     12,     0,    0,    273, 211,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % TBD
 3891   1,        27,   1,  0,  273,  2,     12,     0,    0,    273,   1,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % SU-MIMO (16 UE)
 3892   1,        27,   1,  0,  273,  2,     12,     0,    0,    273,   1,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % MU-MIMO (64 UE, 4-layer/UE)
 3893   1,        27,   1,  0,  273,  2,     12,     0,    0,    273,   1,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % SU/MU-MIMO (64 UE)
 3894   1,        27,   1,  0,  273,  2,     12,     0,    0,    273,   1,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % MU-MIMO (64 UE, 2-layer/UE)
 3895   1,        27,   1,  0,  106,  2,     12,     0,    0,    106,   1,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % 40 MHz
 3896   1,        27,   1,  0,   53,  2,     12,     0,    0,    106,   1,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % 40 MHz
 3897   1,        27,   1,  0,  273,  2,     12,     0,    0,    273,   1,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % non-sequential order of DMRS ports
 3898   1,        27,   1,  0,  273,  2,     12,     0,    0,    273,   1,   0,     41,     2,      2,     1,     41,      2,   0,   16,     0; % non-sequential order of DMRS ports
 3899,  1,        27,   2,  20, 253,  2,     12,     0,    0,    273,   1,   0,     41,     2,      2,     1,     41,      2,   0,   32,     0; % nl 24 = 2 x 12
 3900,  1,        27,   2,  0,  273,  2,     12,     0,    0,    273,   1,   0,     41,     2,      2,     1,     41,      2,   0,   32,     0; % nl 32 = 2 x 16
};

% Invalid PDSCH params
% PDSCH PDU fileds can be changed, total 23 params to overwrite
INVALID_CFG_FIELD_NAMES = {'pduBitmap', 'BWPSize', 'BWPStart', 'SubcarrierSpacing', 'CyclicPrefix', ...
    'NrOfCodewords', 'targetCodeRate', 'qamModOrder', 'mcsIndex', 'mcsTable', ... 
    'rvIndex', 'nrOfLayers', 'refPoint', 'DmrsSymbPos', 'dmrsConfigType', ...
    'SCID', 'numDmrsCdmGrpsNoData', 'resourceAlloc', 'rbStart','rbSize', ...
    'VRBtoPRBMapping', 'StartSymbolIndex', 'NrOfSymbols', 'numPRGs', 'prgSize'
    };
% Different names in spec -> pdu.fieldName: dlDmrsSymbPos -> DmrsSymbPos

% TC config for invalid params, -1 means no change
% INVALID_CFG has 25 params that can be overwritten
INVALID_CFG = { ...
    % TC#  pduBitmap  nBWP  BWP0  scSpace  cpType  nCodeW  codeR  qamM  mcsIdx  mcsTab  rv  nLayers  refP  dmrsPos  dmrsCfg  SCID  nDmrsGrpNoData  resAlloc  RB0  nRB  vPrbMap  sym0  nSym  numPRGs  prgSize
    3501,   1,        -1,    -1,  -1,      -1,     -1,     -1,    -1,   -1,     -1,     -1, -1,      -1,   -1,      -1,      -1,   -1,             -1,        -1,  -1, -1,      -1,   -1,  -1,      -1;  % base case 3201 with invalid pduBitmap = 1
    3502,  -1,         0,    -1,  -1,      -1,     -1,     -1,    -1,   -1,     -1,     -1, -1,      -1,   -1,      -1,      -1,   -1,             -1,        -1,  -1, -1,      -1,   -1,  -1,      -1;  % base case 3201 with invalid BWPSize = 0
    3503,  -1,        -1,   275,  -1,      -1,     -1,     -1,    -1,   -1,     -1,     -1, -1,      -1,   -1,      -1,      -1,   -1,             -1,        -1,  -1, -1,      -1,   -1,  -1,      -1;  % base case 3201 with invalid BWPStart = 275
    3504,  -1,        -1,    -1,   5,      -1,     -1,     -1,    -1,   -1,     -1,     -1, -1,      -1,   -1,      -1,      -1,   -1,             -1,        -1,  -1, -1,      -1,   -1,  -1,      -1;  % base case 3201 with invalid SubcarrierSpacing = 5
    3505,  -1,        -1,    -1,  -1,       2,     -1,     -1,    -1,   -1,     -1,     -1, -1,      -1,   -1,      -1,      -1,   -1,             -1,        -1,  -1, -1,      -1,   -1,  -1,      -1;  % base case 3201 with invalid CyclicPrefix = 2
    3506,  -1,        -1,    -1,  -1,      -1,      2,     -1,    -1,   -1,     -1,     -1, -1,      -1,   -1,      -1,      -1,   -1,             -1,        -1,  -1, -1,      -1,   -1,  -1,      -1;  % base case 3201 with invalid NrOfCodewords = 2
    3507,  -1,        -1,    -1,  -1,      -1,     -1,      0,    -1,   -1,     -1,     -1, -1,      -1,   -1,      -1,      -1,   -1,             -1,        -1,  -1, -1,      -1,   -1,  -1,      -1;  % base case 3201 with invalid targetCodeRate = 0
    3508,  -1,        -1,    -1,  -1,      -1,     -1,     -1,    10,   -1,     -1,     -1, -1,      -1,   -1,      -1,      -1,   -1,             -1,        -1,  -1, -1,      -1,   -1,  -1,      -1;  % base case 3201 with invalid qamModOrder = 10
    3509,  -1,        -1,    -1,  -1,      -1,     -1,     -1,    -1,   32,     -1,     -1, -1,      -1,   -1,      -1,      -1,   -1,             -1,        -1,  -1, -1,      -1,   -1,  -1,      -1;  % base case 3201 with invalid mcsIndex = 32
    3510,  -1,        -1,    -1,  -1,      -1,     -1,     -1,    -1,   -1,      5,     -1, -1,      -1,   -1,      -1,      -1,   -1,             -1,        -1,  -1, -1,      -1,   -1,  -1,      -1;  % base case 3201 with invalid mcsTable = 5
    3511,  -1,        -1,    -1,  -1,      -1,     -1,     -1,    -1,   -1,     -1,      4, -1,      -1,   -1,      -1,      -1,   -1,             -1,        -1,  -1, -1,      -1,   -1,  -1,      -1;  % base case 3201 with invalid rvIndex = 4
    3512,  -1,        -1,    -1,  -1,      -1,     -1,     -1,    -1,   -1,     -1,     -1,  5,      -1,   -1,      -1,      -1,   -1,             -1,        -1,  -1, -1,      -1,   -1,  -1,      -1;  % base case 3201 with invalid nrOfLayers = 5
    3513,  -1,        -1,    -1,  -1,      -1,     -1,     -1,    -1,   -1,     -1,     -1, -1,       2,   -1,      -1,      -1,   -1,             -1,        -1,  -1, -1,      -1,   -1,  -1,      -1;  % base case 3201 with invalid refPoint = 2
    3514,  -1,        -1,    -1,  -1,      -1,     -1,     -1,    -1,   -1,     -1,     -1, -1,      -1,    0,      -1,      -1,   -1,             -1,        -1,  -1, -1,      -1,   -1,  -1,      -1;  % base case 3201 with invalid DmrsSymbPos = 0
    3515,  -1,        -1,    -1,  -1,      -1,     -1,     -1,    -1,   -1,     -1,     -1, -1,      -1,   -1,       2,      -1,   -1,             -1,        -1,  -1, -1,      -1,   -1,  -1,      -1;  % base case 3201 with invalid dmrsConfigType = 2
    3516,  -1,        -1,    -1,  -1,      -1,     -1,     -1,    -1,   -1,     -1,     -1, -1,      -1,   -1,      -1,       2,   -1,             -1,        -1,  -1, -1,      -1,   -1,  -1,      -1;  % base case 3201 with invalid SCID = 2
    3517,  -1,        -1,    -1,  -1,      -1,     -1,     -1,    -1,   -1,     -1,     -1, -1,      -1,   -1,      -1,      -1,    3,             -1,        -1,  -1, -1,      -1,   -1,  -1,      -1;  % base case 3201 with invalid numDmrsCdmGrpsNoData = 3
    3518,  -1,        -1,    -1,  -1,      -1,     -1,     -1,    -1,   -1,     -1,     -1, -1,      -1,   -1,      -1,      -1,   -1,              2,        -1,  -1, -1,      -1,   -1,  -1,      -1;  % base case 3201 with invalid resourceAlloc = 2
    3519,  -1,        -1,    -1,  -1,      -1,     -1,     -1,    -1,   -1,     -1,     -1, -1,      -1,   -1,      -1,      -1,   -1,             -1,       275,  -1, -1,      -1,   -1,  -1,      -1;  % base case 3201 with invalid rbStart = 275
    3520,  -1,        -1,    -1,  -1,      -1,     -1,     -1,    -1,   -1,     -1,     -1, -1,      -1,   -1,      -1,      -1,   -1,             -1,        -1, 276, -1,      -1,   -1,  -1,      -1;  % base case 3201 with invalid rbSize = 276
    3521,  -1,        -1,    -1,  -1,      -1,     -1,     -1,    -1,   -1,     -1,     -1, -1,      -1,   -1,      -1,      -1,   -1,             -1,        -1,  -1,  3,      -1,   -1,  -1,      -1;  % base case 3201 with invalid VRBtoPRBMapping = 3
    3522,  -1,        -1,    -1,  -1,      -1,     -1,     -1,    -1,   -1,     -1,     -1, -1,      -1,   -1,      -1,      -1,   -1,             -1,        -1,  -1, -1,      14,   -1,  -1,      -1;  % base case 3201 with invalid StartSymbolIndex = 14
    3523,  -1,        -1,    -1,  -1,      -1,     -1,     -1,    -1,   -1,     -1,     -1, -1,      -1,   -1,      -1,      -1,   -1,             -1,        -1,  -1, -1,      -1,   15,  -1,      -1;  % base case 3201 with invalid NrOfSymbols = 15
    3524,  -1,        -1,    -1,  -1,      -1,     -1,     -1,    -1,   -1,     -1,     -1, -1,      -1,   -1,      -1,      -1,   -1,             -1,        -1,  -1, -1,      -1,   -1,  17,      16;  % base case 3201 with invalid numPRGs=17, prgSize=16 (more than prgSize of 10 not in allowed set for 4T4R)
    };

[NallTest, ~] = size(CFG);
errCnt = 0;
detErr = 0;
nTV = 0;
nComp = 0;

if (isnumeric(caseSet))
    caseSetStr = num2str(caseSet);
else
    caseSetStr = caseSet;
end
fprintf('PDSCH: genTV = %d, testCompliance = %d, caseSet = %s, relNum = %d', genTV, testCompliance, caseSetStr, relNum);

fprintf('\nTC#   mcsTable  mcs  nl  rb0 Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId dmrs0 maxLen addPos dmrsScId nCdm port0  nAnt slotIdx Pass Det\n');
fprintf('------------------------------------------------------------------------------------------------------------------------------------------------------\n');

parfor n = 1:NallTest
    rbBitmap1=[];
    rbBitmap2=[];
    rbBitmap3=[];
    testAlloc = [];
    testAlloc.dl = 1;
    testAlloc.ul = 0;
    testAlloc.pdsch = 1;
    idxSet = n;
    caseNum = CFG{idxSet, 1};
    if ismember(caseNum, TcToTest) && (mod(caseNum, subSetMod(2)) == subSetMod(1))
        rng(caseNum);
        SysPar = initSysPar(testAlloc);
        SysPar.SimCtrl.relNum = relNum;
        SysPar.SimCtrl.N_frame = 1;
        if genTV
            nTV = nTV + 1;
            SysPar.SimCtrl.genTV.enable = 1;
            SysPar.SimCtrl.genTV.TVname = sprintf('TVnr_%04d', caseNum);
            if ismember(caseNum, disabled_TC)
                SysPar.SimCtrl.genTV.TVname = sprintf('disabled_TVnr_%04d', caseNum);
            end
            SysPar.SimCtrl.N_slot_run = CFG{idxSet, 22} + 1;
%             SysPar.SimCtrl.genTV.forceSlotIdxFlag = 1;
            SysPar.SimCtrl.genTV.slotIdx = CFG{idxSet, 22};
        end
               
        SysPar.pdsch{1}.mcsTable = CFG{idxSet, 2};
        SysPar.pdsch{1}.mcsIndex =  CFG{idxSet, 3};
        SysPar.pdsch{1}.nrOfLayers = CFG{idxSet, 4};
        SysPar.pdsch{1}.rbStart = CFG{idxSet, 5};
        SysPar.pdsch{1}.rbSize = CFG{idxSet, 6};
        SysPar.pdsch{1}.StartSymbolIndex = CFG{idxSet, 7};                              
        SysPar.pdsch{1}.NrOfSymbols = CFG{idxSet, 8};
        SysPar.pdsch{1}.SCID =  CFG{idxSet, 9};
        SysPar.pdsch{1}.BWPStart =  CFG{idxSet, 10};
        SysPar.pdsch{1}.BWPSize =  CFG{idxSet, 11};
        SysPar.pdsch{1}.RNTI =  CFG{idxSet, 12};
        SysPar.pdsch{1}.rvIndex =  CFG{idxSet, 13};
        SysPar.pdsch{1}.dataScramblingId =  CFG{idxSet, 14};
        sym0 = SysPar.pdsch{1}.StartSymbolIndex;
        nSym = SysPar.pdsch{1}.NrOfSymbols;
        dmrs0 = CFG{idxSet, 15};
        SysPar.carrier.dmrsTypeAPos = dmrs0;
        maxLen = CFG{idxSet, 16};
        addPos = CFG{idxSet, 17};
        DmrsSymbPos = findDmrsSymbPos(sym0, nSym, dmrs0, maxLen, addPos, 'DL', 'typeA');
        SysPar.pdsch{1}.DmrsSymbPos = DmrsSymbPos;
        SysPar.pdsch{1}.DmrsScramblingId =  CFG{idxSet, 18};
        SysPar.pdsch{1}.numDmrsCdmGrpsNoData =  CFG{idxSet, 19};
        SysPar.pdsch{1}.portIdx = CFG{idxSet, 20} + [0:SysPar.pdsch{1}.nrOfLayers-1];
        SysPar.carrier.Nant_gNB =  CFG{idxSet, 21};
        SysPar.pdsch{1}.seed = caseNum;
        if ismember(caseNum, [3850, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859])
            SysPar.pdsch{1}.seed = 0;
            SysPar.SimCtrl.enable_dynamic_BF = 1;
            SysPar.carrier.N_FhPort_DL = SysPar.carrier.Nant_gNB;
            SysPar.carrier.Nant_gNB_srs = 32;
            SysPar.pdsch{1}.prgSize = 2;
            SysPar.pdsch{1}.numPRGs = ceil(SysPar.pdsch{1}.rbSize/SysPar.pdsch{1}.prgSize);
        end
        if caseNum == 3851
            SysPar.pdsch{1}.numPRGs = 39;
        elseif caseNum == 3852
            SysPar.pdsch{1}.numPRGs = 67;
        elseif caseNum == 3853 
            SysPar.pdsch{1}.numPRGs = 30;
        end
        
        if ismember(caseNum, [3869:3900])
            SysPar.pdsch{1}.seed = 0;
            SysPar.SimCtrl.enable_dynamic_BF = 1;
            SysPar.carrier.N_FhPort_DL = 16;
            SysPar.carrier.N_FhPort_UL = 16;
            if ismember(caseNum, [3899, 3900])
                SysPar.carrier.N_FhPort_DL = 32;
                SysPar.carrier.N_FhPort_UL = 16;
            end
            SysPar.carrier.Nant_gNB_srs = 64;
            SysPar.pdsch{1}.prgSize = 2;
            if ismember(caseNum, [3878 3889])
                SysPar.pdsch{1}.prgSize = 4;
            elseif ismember(caseNum, [3869])
                SysPar.pdsch{1}.prgSize = 16;
            end
            SysPar.pdsch{1}.numPRGs = ceil(SysPar.pdsch{1}.rbSize/SysPar.pdsch{1}.prgSize);
        end

        nUeCfg = [];
        csirs_cfg = [];
        % FDM
        if caseNum == 3233
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0  60   10  1  0   0    0;...
                      100  60   10  1  0   0    1];
        elseif caseNum == 3234
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0  60   10  1  0   0    0;...
                      100  80    5  1  0   0    1];             
        elseif caseNum == 3235
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0  60   10  1  0   0    0;...
                      100  80    5  2  0   0    1];                         
        % SDM
        elseif caseNum == 3236
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0   60  10  1  0   0    0;...
                        0   60  10  1  1   0    0];          
        elseif caseNum == 3237
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0   60  10  1  0   0    0;...
                        0   60   5  1  1   0    0];                     
        elseif caseNum == 3238
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0   60  10  1  0   0    0;...
                        0   60   5  2  2   0    0];                     
        % FDM + SDM
        elseif caseNum == 3239
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0   60   3  2  0   0    0;...
                        0   60   5  2  2   0    0;...
                      100   80   4  1  0   0    1;...
                      100   80   8  1  3   0    1];                                          
        elseif caseNum == 3240
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0   60   3  2  0   0    0;...
                        0   60   5  1  2   0    0;...
                        0   60   4  1  3   0    0;...
                      100   80   8  1  0   0    1];     
        elseif caseNum == 3241
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0   60   3  4  0   0    0;...
                        0   60   5  4  4   0    0;...
                        0   60   4  4  0   1    0;...
                        0   60   8  4  4   1    0]; 
            SysPar.carrier.Nant_gNB = 16;
        elseif caseNum == 3245
            SysPar.carrier.N_grid_size_mu = 106;
            SysPar.carrier.mu = 0;
        elseif caseNum == 3246
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg sym0 Nsym dmrs0 maxLen addPos 
                        0  60   10  1  0   0    0      2    4    2      1      0;...
                      100  80    5  1  0   0    1      0   14    2      2      1];     
        elseif caseNum == 3247
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg sym0 Nsym dmrs0 maxLen addPos
                        0   60   3  2  0   0    0      2    4    2      1      0;...
                        0   60   5  2  2   0    0      2    4    2      1      0;...
                      100   80   4  1  0   0    1      0   14    2      2      1;...     
                      100   80   8  1  3   0    1      0   14    2      2      1];   
       % Precoding           
        elseif caseNum == 3248 % PM_W14
            SysPar.pdsch{1}.prcdBf = 4; 
        elseif caseNum == 3249 % PM_W24
            SysPar.pdsch{1}.prcdBf = 8;
        elseif caseNum == 3250 % PM_W34
            SysPar.pdsch{1}.prcdBf = 10;
        elseif caseNum == 3251 % PM_W44
            SysPar.pdsch{1}.prcdBf = 12; 
        elseif caseNum == 3252 % PM_W12
            SysPar.pdsch{1}.prcdBf = 2; 
        elseif caseNum == 3253 % PM_W22
            SysPar.pdsch{1}.prcdBf = 6;            
        elseif caseNum == 3254 % PM_W24 + PM_W14
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg sym0 Nsym dmrs0 maxLen addPos
                        0 273    1  2  0   0    0      2    12    2      1      0;...
                        0 273    2  1  2   0    0      2    12    2      1      0];            
        elseif caseNum == 3255 || caseNum == 3276 % mapping type B
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg sym0 Nsym dmrs0 maxLen addPos
                        0  60   10  1  0   0    0     10     4    2      1      0];
        elseif caseNum == 3256 % mapping type B 2 UEs with same SLIV
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg sym0 Nsym dmrs0 maxLen addPos
                        0  60   10  1  0   0    0     10     4    2      1      0;...
                      100  80   10  1  0   0    1     10     4    2      1      0];         
        elseif caseNum == 3257 % mapping type B 2 UEs with different SLIV
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg sym0 Nsym dmrs0 maxLen addPos
                        0  60   10  1  0   0    0      0     7    2      1      1;...
                      100  80    5  1  0   0    1      4     4    2      1      0];
         elseif caseNum == 3258 % PM_W24 + noPrcd
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg sym0 Nsym dmrs0 maxLen addPos
                        0 273    1  2  0   0    0      2    12    2      1      0;...
                        0 273    2  1  2   0    0      2    12    2      1      0];
        elseif ismember(caseNum, [3259, 3264]) % precoding with mixed mod/nl/nPort
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0   10   4  1  0   0    0;...
                       10    5   4  2  0   0    1;...
                       20   10   4  3  0   0    2;...
                       30    5   4  4  0   0    3;... 
                       40   10   8  1  0   0    4;...
                       50    5   8  2  0   0    5;...
                       60   10   8  3  0   0    6;...
                       70    5   8  4  0   0    7;...
                       80   10  16  1  0   0    8;...
                       90    5  16  2  0   0    9;...
                      100   10  16  3  0   0   10;...
                      110    5  16  4  0   0   11;... 
                      120   10  24  1  0   0   12;...
                      130    5  24  2  0   0   13;...
                      140   10  24  3  0   0   14;...
                      150    5  24  4  0   0   15];                  
        elseif caseNum == 3260 % prcd + csirs
            SysPar.pdsch{1}.prcdBf = 4;
            csirs_cfg = {%Row CDM Density RB0  nRB  sym0  sym1  FreqDomain
                           1,  0,   3,     0,  273,   13,    8, {[1 1 1 0]}};
            [nCsirs, ~] = size(csirs_cfg);
            SysPar.csirs = gen_csirs_config(csirs_cfg);
            SysPar.testAlloc.csirs = nCsirs;
            for idxCsirs = 1:nCsirs
                SysPar.csirs{idxCsirs}.CSIType = 2; % force to ZP-CSI-RS to avoid impact to PDSCH compliance test
            end
        elseif caseNum == 3907 % prcd + csirs
            csirs_cfg = {%Row CDM Density RB0  nRB  sym0  sym1  FreqDomain
                           5,  1,   2,     0,  273,   10,    8, {[1 0 1 0 1 1]}};
            [nCsirs, ~] = size(csirs_cfg);
            SysPar.csirs = gen_csirs_config(csirs_cfg);
            SysPar.testAlloc.csirs = nCsirs;
            for idxCsirs = 1:nCsirs
                SysPar.csirs{idxCsirs}.CSIType = 2; % force to ZP-CSI-RS to avoid impact to PDSCH compliance test
            end
        elseif ismember(caseNum, [3261, 3901]) % refPoint = 1
            SysPar.pdsch{1}.refPoint = 1;
        elseif caseNum == 3262 % TxPower
            SysPar.pdsch{1}.powerControlOffset = 8 + 2;   % PDSCH/CSIRS = 2 dB         
            SysPar.pdsch{1}.powerControlOffsetSS = 0; % CSIRS/SSB = -3dB
        elseif ismember(caseNum, [3263, 3356:3359]) % modComp
            SysPar.pdsch{1}.powerControlOffset = 8 + 2;   % PDSCH/CSIRS = 2 dB         
            SysPar.pdsch{1}.powerControlOffsetSS = 0; % CSIRS/SSB = -3dB            
            SysPar.SimCtrl.genTV.fhMsgMode = 2;
        elseif caseNum == 3360
            SysPar.pdsch{1}.powerControlOffset = 8 + 2;   % PDSCH/CSIRS = 2 dB         
            SysPar.pdsch{1}.powerControlOffsetSS = 0; % CSIRS/SSB = -3dB
            SysPar.SimCtrl.genTV.fhMsgMode = 2;
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0  100  27  4  0   0    0;...
                        0  100  19  3  4   0    0;...
                        0  100  10  1  7   0    0;...
                        0  100   1  2  0   1    0;...
                      100  100  19  4  0   0    1;...
                      100  100  27  4  4   0    1];
        elseif caseNum == 3265
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0  60   10  1  0   0    0;...
                      100  60   10  1  0   0    1];
        elseif caseNum == 3266
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0  50   10  1  0   0    0;...
                       50  50   10  1  0   0    1;...
                      100  50   10  1  0   0    2;...
                      150  50   10  1  0   0    3];
        elseif caseNum == 3267
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0  10   10  1  0   0    0;...
                       50  20   15  1  0   0    1;...
                      100  30    5  1  0   0    2;...
                      150  40   20  1  0   0    3];
        elseif caseNum == 3268
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0   60   3  2  0   0    0;...
                        0   60   5  2  2   0    0;...
                        0   60   3  4  4   0    0;...
                       60   40   2  2  0   0    1;...
                       60   40   4  3  4   0    1]; 
            SysPar.carrier.Nant_gNB = 8; 
        elseif caseNum == 3269
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0   60   3  4  0   0    0;...
                        0   60   5  4  4   0    0;...
                        0   60   4  4  0   1    0;...
                        0   60   8  4  4   1    0;...
                       60   40   1  2  0   0    1;...
                       60   40   2  1  2   0    1;...
                       60   40   3  4  4   0    1;...
                       60   40   4  3  4   1    1]; 
            SysPar.carrier.Nant_gNB = 16; 
        elseif caseNum == 3270
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0   60   3  4  0   0    0;...
                        0   60   5  4  4   0    0;...
                        0   60   4  4  0   1    0;...
                        0   60   8  4  4   1    0;...
                       60   40   1  2  0   0    1;...
                       60   40   2  1  2   0    1;...
                       60   40   3  4  4   0    1;...
                       60   40   4  3  4   1    1]; 
            SysPar.carrier.Nant_gNB = 32;
            SysPar.SimCtrl.enableDlTxBf = 1;
        elseif caseNum == 3277
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg sym0 Nsym dmrs0 maxLen addPos
                        0   3    1  1  0   0    0      2     6    2      1      0;...
                        5   1    3  1  0   0    1      2    12    2      1      0];
        elseif caseNum == 3278 % 4-BWP
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg sym0 Nsym dmrs0 maxLen addPos
                        0   3    1  1  0   0    0      2     6    2      1      0;...
                        5   1    3  1  0   0    1      2    12    2      1      0;...
                        0   3    1  1  0   0    2      2     6    2      1      0;...
                        5   1    3  1  0   0    3      2    12    2      1      0];
        elseif ismember(caseNum, [3286, 3287]) % FDM nCDM = 1 and 2
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0   60   3  1  0   0    0;...
                      100   70   5  2  0   0    1];
        elseif ismember(caseNum, [3288]) % nCDM = 1 precoding with mixed mod/nl/nPort
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0   10   4  1  0   0    0;...
                       10    5   4  2  0   0    1;...
                       20   10   4  1  0   0    2;...
                       30    5   4  2  0   0    3;...
                       40   10   8  1  0   0    4;...
                       50    5   8  2  0   0    5;...
                       60   10   8  1  0   0    6;...
                       70    5   8  2  0   0    7;...
                       80   10  16  1  0   0    8;...
                       90    5  16  2  0   0    9;...
                      100   10  16  1  0   0   10;...
                      110    5  16  2  0   0   11;...
                      120   10  24  1  0   0   12;...
                      130    5  24  2  0   0   13;...
                      140   10  24  1  0   0   14;...
                      150    5  24  2  0   0   15];
        elseif ismember(caseNum, [3296, 3354, 3355]) % testModel
            SysPar.pdsch{1}.testModel = 1;
        elseif caseNum == 3297
            SysPar.pdsch{1}.targetCodeRate = 1930; % FAPI format: raw coderate x 1024 x 10
            SysPar.pdsch{1}.qamModOrder = 2;
            SysPar.pdsch{1}.TBSize = 1697;
        elseif caseNum == 3299
            SysPar.pdsch{1}.targetCodeRate = 30; % FAPI format: raw coderate x 1024 x 10
            SysPar.pdsch{1}.qamModOrder = 2;
            SysPar.pdsch{1}.TBSize = 478;
        elseif caseNum == 3300
            SysPar.pdsch{1}.targetCodeRate = 30; % FAPI format: raw coderate x 1024 x 10
            SysPar.pdsch{1}.qamModOrder = 2;
            SysPar.pdsch{1}.TBSize = 478;
        elseif caseNum == 3342
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0  270  29  4  0   0    0;...
                      270    3  10  1  0   0    1];
        % LBRM
        elseif ismember(caseNum, [3321:3322])
            SysPar.pdsch{1}.I_LBRM = 1;
        % CSI-RS    
        elseif ismember(caseNum, [3323:3333, 3338:3339, 3350:3353])
            if caseNum == 3323
                csirs_cfg = {%Row CDM Density RB0  nRB  sym0  sym1  FreqDomain
                              1,  0,   3,     0,  273,   13,    8, {[1 1 1 0]}};
            elseif caseNum == 3324
                csirs_cfg = {%Row CDM Density RB0  nRB  sym0  sym1  FreqDomain
                              2,  0,   3,     0,  273,   13,    8, {[ones(1,11), 0]}};
            elseif caseNum == 3325
                csirs_cfg = {%Row CDM Density RB0  nRB  sym0  sym1  FreqDomain
                              4,  1,   2,     0,  273,   13,    8, {[1 0 0]}};
            elseif caseNum == 3326
                csirs_cfg = {%Row CDM Density RB0  nRB  sym0  sym1  FreqDomain
                              5,  1,   2,    16,   72,   10,    8, {[1 0 1 0 1 1]}};
            elseif caseNum == 3327
                csirs_cfg = {%Row CDM Density RB0  nRB  sym0  sym1  FreqDomain
                              7,  1,   2,    16,   72,    8,    8, {[0 1 1 1 0 1]}};
            elseif caseNum == 3328
                csirs_cfg = {%Row CDM Density RB0  nRB  sym0  sym1  FreqDomain
                              8,  2,   2,    16,   72,    8,    8, {[1 1 1 1 1 1]}};
            elseif caseNum == 3329
                csirs_cfg = {%Row CDM Density RB0  nRB  sym0  sym1  FreqDomain
                              1,  0,   3,     0,  273,   13,    8, {[1 1 1 0]};...                    
                              8,  2,   2,    16,   72,    8,    8, {[1 1 1 1 1 1]}};          
            elseif caseNum == 3330
                csirs_cfg = {%Row CDM Density RB0  nRB  sym0  sym1  FreqDomain
                              1,  0,   3,     0,  273,   13,    8, {[1 1 1 0]};...                    
                              8,  2,   2,    16,   72,    8,    8, {[1 1 1 1 1 1]}}; 
                nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg 
                            0  10   10  1  0   0    0;...
                           20  40    5  1  0   0    1];           
            elseif caseNum == 3331
                csirs_cfg = {%Row CDM Density RB0  nRB  sym0  sym1  FreqDomain
                              1,  0,   3,     0,    52,   5,    8, {[0 1 0 0]};...                    
                              1,  0,   3,     0,    52,   9,    8, {[0 1 0 0]}};
            elseif caseNum == 3332
                csirs_cfg = {%Row CDM Density RB0  nRB  sym0  sym1  FreqDomain
                              2,  0,   0,     0,  273,   13,    8, {[ones(1,11), 0]}};
            elseif caseNum == 3333
                csirs_cfg = {%Row CDM Density RB0  nRB  sym0  sym1  FreqDomain
                              2,  0,   1,     0,  273,   13,    8, {[ones(1,11), 0]}};
            elseif ismember(caseNum, [3338:3339])
                csirs_cfg = {%Row CDM Density RB0  nRB  sym0  sym1  FreqDomain
                              1,  0,   3,     0,  273,   13,    8, {[1 1 1 0]};...
                              8,  2,   2,    16,   72,    8,    8, {[1 1 1 1 1 1]}};
                nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                            0  32   10  1  0   0    0;...
                           40  32    5  1  0   0    1];
            elseif caseNum == 3350
                csirs_cfg = {%Row CDM Density RB0  nRB  sym0  sym1  FreqDomain
                              1,  0,   3,     0,  273,    6,    0, {[0 1 0 0]};...
                              1,  0,   3,     0,  273,   10,    0, {[0 1 0 0]}};
            elseif caseNum == 3351
                csirs_cfg = {%Row CDM Density RB0  nRB  sym0  sym1  FreqDomain
                              1,  0,   3,     0,  273,    2,    0, {[0 1 0 0]};...
                              1,  0,   3,     0,  273,    3,    0, {[0 0 1 0]}};
                nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg sym0 Nsym dmrs0 maxLen addPos
                           20  80  10  1   0   0    0     0    7    0      1      1];
            elseif caseNum == 3352
                csirs_cfg = {%Row CDM Density RB0  nRB  sym0  sym1  FreqDomain
                              1,  0,   3,     0,  273,    4,    2, {[1 0 0 0]};...
                              1,  0,   3,     0,  273,   13,    0, {[0 1 0 0]}};
            elseif caseNum == 3353
                csirs_cfg = {%Row CDM Density RB0  nRB  sym0  sym1  FreqDomain
                              1,  0,   3,     0,  273,    1,    0, {[0 1 0 0]};...
                              1,  0,   3,     0,  273,   12,    0, {[0 0 0 1]};...
                             1,  0,   3,     0,  273,   13,    8, {[1 1 1 0]}};
            end
            [nCsirs, ~] = size(csirs_cfg);
            SysPar.csirs = gen_csirs_config(csirs_cfg);
            SysPar.testAlloc.csirs = nCsirs;
            for idxCsirs = 1:nCsirs
                SysPar.csirs{idxCsirs}.CSIType = 2; % force to ZP-CSI-RS to avoid impact to PDSCH compliance test
%                 SysPar.csirs{idxCsirs}.idxUE = idxCsirs-1;
                SysPar.csirs{idxCsirs}.idxUE = 0;
            end       

        elseif caseNum == 3334
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0   5    5  1  0   0    0;...
                       20  10   10  1  0   0    1;...
                       40   5   15  1  0   0    2;...
                       60  10   20  1  0   0    3;...
                       80   5   25  1  0   0    4;...
                      100  10   20  1  0   0    5;...
                      120   5   15  1  0   0    6;...
                      140  10   10  1  0   0    7];
        elseif caseNum == 3335
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0   5    5  1  0   0    0;...
                       20  10   10  2  0   0    1;...
                       40   5   15  1  0   0    2;...
                       60  10   20  2  0   0    3;...
                       80   5   25  1  0   0    4;...
                      100  10   20  2  0   0    5;...
                      120   5   15  1  0   0    6;...
                      140  10   10  4  0   0    7];
        elseif caseNum == 3336
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0   5    5  1  0   0    0;...
                       20  10   10  2  0   0    1;...
                       40   5   15  1  0   0    2;...
                       60  10   20  2  0   0    3;...
                       80   5   25  1  0   0    4;...
                      100  10   20  2  0   0    5;...
                      120   5   15  1  0   0    6;...
                      140  10   10  4  0   0    7;...
                      160   5    5  1  0   0    8;...
                      180  10   10  2  0   0    9;...
                      200   5   15  1  0   0   10;...
                      210  10   20  2  0   0   11;...
                      220   5   25  1  0   0   12;...
                      230  10   20  2  0   0   13;...
                      240   5   15  1  0   0   14;...
                      250  10   10  4  0   0   15];
        elseif caseNum == 3337
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0   5    5  1  0   0    0;...
                        0   5    7  4  1   0    0;...
                       20  10   10  2  0   0    1;...
                       20  10   12  1  2   0    1;...
                       40   5   13  1  0   0    2;...
                       40   5   15  1  1   0    2;...
                       60  10   22  1  0   0    3;...
                       60  10   20  2  2   0    3;...
                       80   5    5  2  0   0    4;...
                       80   5   25  2  2   0    4;...
                      100  10   20  1  0   0    5;...
                      100  10   10  1  1   0    5;...
                      120   5    5  2  0   0    6;...
                      120   5   15  2  2   0    6;...
                      140  10   20  2  0   0    7;...
                      140  10   10  4  2   0    7];
        elseif caseNum == 3344
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       20  252  27  2  0   0    0;...
                       20  252  27  2  2   0    0;...
                       20  252  27  2  4   0    0;...
                       20  252  27  2  6   0    0];
        elseif caseNum == 3362
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       20  252  27  4  0   0    0;...
                       20  252  27  4  4   0    0;...
                       20  252  27  4  0   1    0;...
                       20  252  27  4  4   1    0;...
                       20  252  27  4  0   0    0;...
                       20  252  27  4  4   0    0;...
                       20  252  27  4  0   1    0;...
                       20  252  27  4  4   1    0];
        elseif ismember(caseNum, [3850,3859]) 
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       0  273  27  1  0   0    0;...
                       0  273  27  1  1   0    0;...
                       0  273  27  1  2   0    0;...
                       0  273  27  1  3   0    0;...
                       0  273  27  1  4   0    0;...
                       0  273  27  1  5   0    0;...
                       0  273  27  1  6   0    0;...
                       0  273  27  1  7   0    0];
        
        elseif caseNum == 3851
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       0  78  27  1  0   0    0;...
                       0  78  27  1  1   0    0;...
                       0  78  27  1  2   0    0;...
                       0  78  27  1  3   0    0;...
                       0  78  27  1  4   0    0;...
                       0  78  27  1  5   0    0;...
                       0  78  27  1  6   0    0;...
                       0  78  27  1  7   0    0];

        elseif caseNum == 3852
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       0  133  27  1  0   0    0;...
                       0  133  27  1  1   0    0;...
                       0  133  27  1  2   0    0;...
                       0  133  27  1  3   0    0;...
                       0  133  27  1  4   0    0;...
                       0  133  27  1  5   0    0;...
                       0  133  27  1  6   0    0;...
                       0  133  27  1  7   0    0];

        elseif caseNum == 3853 
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       0  273  27  1  0   0    0;...
                       0  273  27  1  1   0    0;...
                       0  273  27  1  2   0    0;...
                       0  273  27  1  3   0    0;...
                       0  273  27  1  4   0    0;...
                       0  273  27  1  5   0    0;...
                       0  273  27  1  6   0    0;...
                       0  273  27  1  7   0    0];

            csirs_cfg = {%Row CDM Density RB0  nRB  sym0  sym1  FreqDomain
                           6,  1,   2,     0,  273,   0,    8, {ones(1,12)}};
            %1,  0,   3,     0,  273,   13,    8, {[1 1 1 0]}};
            [nCsirs, ~] = size(csirs_cfg);
            SysPar.csirs = gen_csirs_config(csirs_cfg);
            SysPar.testAlloc.csirs = nCsirs;
            for idxCsirs = 1:nCsirs
                SysPar.csirs{idxCsirs}.CSIType = 2; % force to ZP-CSI-RS to avoid impact to PDSCH compliance test
                %                 SysPar.csirs{idxCsirs}.idxUE = idxCsirs-1;
                SysPar.csirs{idxCsirs}.idxUE = 0;
                SysPar.csirs{idxCsirs}.numPRGs = 137;
                SysPar.csirs{idxCsirs}.prgSize = 2;
            end
            

        elseif caseNum == 3854
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       10  60  27  1  0   0    0;...
                       10  60  27  1  1   0    0;...
                       10  60  27  1  2   0    0;...
                       10  60  27  1  3   0    0;...
                       10  60  27  1  4   0    0;...
                       10  60  27  1  5   0    0;...
                       10  60  27  1  6   0    0;...
                       10  60  27  1  7   0    0];
             
        elseif caseNum == 3870 || caseNum == 3869
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       0  273  27  1  0   0    0;...
                       0  273  27  1  1   0    0;...
                       0  273  27  1  2   0    0;...
                       0  273  27  1  3   0    0;...
                       0  273  27  1  4   0    0;...
                       0  273  27  1  5   0    0;...
                       0  273  27  1  6   0    0;...
                       0  273  27  1  7   0    0;...
                       0  273  27  1  0   1    0;...
                       0  273  27  1  1   1    0;...
                       0  273  27  1  2   1    0;...
                       0  273  27  1  3   1    0;...
                       0  273  27  1  4   1    0;...
                       0  273  27  1  5   1    0;...
                       0  273  27  1  6   1    0;...
                       0  273  27  1  7   1    0];
        elseif caseNum == 3873
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       0  273  27  1  0   0    0;...
                       0  273  27  1  1   0    0];
        elseif caseNum == 3874
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       0  273  27  2  0   0    0;...
                       0  273  27  2  2   0    0];
        elseif caseNum == 3875
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       0  273  27  1  0   0    0;...
                       0  273  27  1  1   0    0;...
                       0  273  27  1  2   0    0;...
                       0  273  27  1  3   0    0];
        elseif caseNum == 3876
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       0  273  27  2  0   0    0;...
                       0  273  27  2  2   0    0;...
                       0  273  27  2  4   0    0;...
                       0  273  27  2  6   0    0];
        elseif caseNum == 3879
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       0  273  27  2  0   0    0;...
                       0  273  27  2  2   0    0;...
                       0  273  27  2  4   0    0;...
                       0  273  27  2  6   0    0;...
                       0  273  27  2  0   1    0;...
                       0  273  27  2  2   1    0;...
                       0  273  27  2  4   1    0;...
                       0  273  27  2  6   1    0];
        elseif caseNum == 3880
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       0  273  27  4  0   0    0;...
                       0  273  27  4  4   0    0;...
                       0  273  27  4  4   1    0;...
                       0  273  27  4  0   1    0];
        elseif caseNum == 3881
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                     160   64  27  1  0   0    0;...
                     160   64  27  1  2   0    0;...
                     160   64  27  1  4   0    0;...
                     160   64  27  1  6   0    0;...
                     160   64  27  1  1   0    0;...
                     160   64  27  1  3   0    0;...
                     160   64  27  1  5   0    0;...
                     160   64  27  1  7   0    0];
        elseif caseNum == 3882
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       0  100  27  2  0   0    0;...
                     100  100  27  2  2   0    1;...
                       0  100  27  2  4   0    0;...
                     100  100  27  2  6   0    1;...
                       0  100  27  2  0   1    0;...
                     100  100  27  2  2   1    1;...
                       0  100  27  2  4   1    0;...
                     100  100  27  2  6   1    1];
        elseif caseNum == 3883
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       0  100  27  2  0   0    0;...
                     100   50  27  2  2   0    1;...
                       0  100  27  2  4   0    0;...
                     100   50  27  2  6   0    1;...
                       0  100  27  2  0   1    0;...
                     100   50  27  2  2   1    1;...
                       0  100  27  2  4   1    0;...
                     100   50  27  2  6   1    1];
        elseif caseNum == 3884
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       0  100  27  2  0   0    0;...
                     110   80  27  2  2   0    1;...
                       0  100  27  2  4   0    0;...
                     110   80  27  2  6   0    1;...
                       0  100  27  2  0   1    0;...
                     110   80  27  2  2   1    1;...
                       0  100  27  2  4   1    0;...
                     110   80  27  2  6   1    1];
        elseif caseNum == 3885
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       0  100  27  1  0   0    0;...
                     100  100  27  2  1   0    1;...
                       0  100  27  4  3   0    0;...
                     100  100  27  1  7   0    1;...
                       0  100  27  2  0   1    0;...
                     100  100  27  2  2   1    1;...
                       0  100  27  2  4   1    0;...
                     100  100  27  2  6   1    1];
        elseif caseNum == 3886
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       0  100  27  1  0   0    0;...
                     120   80  27  2  1   0    1;...
                       0  100  27  4  3   0    0;...
                     120   80  27  1  7   0    1;...
                       0  100  27  2  0   1    0;...
                     120   80  27  2  2   1    1;...
                       0  100  27  2  4   1    0;...
                     120   80  27  2  6   1    1];
        elseif caseNum == 3887
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       0   10  27  1  0   0    0;...
                      10   60  27  2  1   0    1;...
                     100   50  27  4  3   0    2;...
                     150   50  27  1  7   0    3;...
                       0   10  27  4  0   1    0;...
                      10   60  27  1  4   1    1;...
                     100   50  27  2  5   1    2;...
                     150   50  27  1  7   1    3];
        elseif caseNum == 3888
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                      20   10  27  1  0   0    0;...
                      30   20  27  2  1   0    1;...
                      50   20  27  4  3   0    2;...
                       0   20  27  1  7   0    3;...
                      90   20  27  4  0   1    4;...
                     110   20  27  1  4   1    5;...
                     150   20  27  1  5   1    6;...
                     200   40  27  2  6   1    7];
        elseif caseNum == 3890
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                      20   20  27  3  0   0    0;...
                      20   20  27  4  3   0    0;...
                      40   20  27  4  0   0    1;...
                      40   20  27  1  4   0    1;...
                      40   20  27  1  5   0    1;...
                      40   20  27  2  6   0    1;...
                      40   20  27  1  0   1    1];
        elseif caseNum == 3891
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                      0    12   27  4  0   0    0;...
                      12   14   27  4  0   0    1;...
                      26   16   27  4  0   0    2;...
                      42   18   27  4  0   0    3;...
                      60   20   27  4  0   0    4;...
                      80   22   27  4  0   0    5;...
                      102  24   27  4  0   0    6;...
                      126  22   27  4  0   0    7;...
                      148  20   27  4  0   0    8;...
                      168  18   27  4  0   0    9;...
                      186  16   27  4  0   0   10;...
                      202  14   27  4  0   0   11;...
                      216  12   27  4  0   0   12;...
                      228  10   27  4  0   0   13;...
                      238   8   27  4  0   0   14;...
                      246  27   27  4  0   0   15];
        elseif caseNum == 3892
            T = readtable('pdsch_tc.xlsx','Sheet','tc3892')
            nUeCfg = T.Variables;
        elseif caseNum == 3893
            T = readtable('pdsch_tc.xlsx','Sheet','tc3893')
            nUeCfg = T.Variables;
        elseif caseNum == 3894
            T = readtable('pdsch_tc.xlsx','Sheet','tc3894')
            nUeCfg = T.Variables;
        elseif caseNum == 3895
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                      0    106  27  4  0   0    0;...
                      0    106  27  4  4   0    0;...
                      0    106  27  4  0   1    0;...
                      0    106  27  4  4   1    0];
        elseif caseNum == 3896
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                      0    53  27  4  0   0    0;...
                      0    53  27  4  4   0    0;...
                      0    53  27  4  0   1    0;...
                      0    53  27  4  4   1    0];
        elseif caseNum == 3897 % nvbug 5368243
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       0   273  27  2  0   0    0;...
                       0   273  27  1  4   1    0;...
                       0   273  27  1  5   1    0;...
                       0   273  27  2  2   0    0;...
                       0   273  27  2  4   0    0;...
                       0   273  27  2  6   0    0;...
                       0   273  27  2  0   1    0;...
                       0   273  27  2  2   1    0;...
                       0   273  27  1  6   1    0;...
                       0   273  27  1  7   1    0];
        elseif caseNum == 3898 % nvbug 5368243
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       0   273  27  2  0   0    0;...
                       0   273  27  1  4   1    0;...
                       0   273  27  1  5   1    0;...
                       0   273  27  2  2   0    0];
        elseif caseNum == 3899
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                       20  252  27  2  0   0    0;...
                       20  252  27  2  2   0    0;...
                       20  252  27  2  4   0    0;...
                       20  252  27  2  6   0    0;...
                       20  252  27  2  0   1    0;...
                       20  252  27  2  2   1    0;...
                       20  252  27  2  4   1    0;...
                       20  252  27  2  6   1    0;...
                       20  252  27  2  0   0    0;...
                       20  252  27  2  2   0    0;...
                       20  252  27  2  4   0    0;...
                       20  252  27  2  6   0    0];
        elseif caseNum == 3900
            nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                        0  273  27  2  0   0    0;...
                        0  273  27  2  2   0    0;...
                        0  273  27  2  4   0    0;...
                        0  273  27  2  6   0    0;...
                        0  273  27  2  0   1    0;...
                        0  273  27  2  2   1    0;...
                        0  273  27  2  4   1    0;...
                        0  273  27  2  6   1    0;...
                        0  273  27  2  0   0    0;...
                        0  273  27  2  2   0    0;...
                        0  273  27  2  4   0    0;...
                        0  273  27  2  6   0    0;...
                        0  273  27  2  0   1    0;...
                        0  273  27  2  2   1    0;...
                        0  273  27  2  4   1    0;...
                        0  273  27  2  6   1    0];
        end

        if ismember(caseNum, [3401:3413, 3422:3423, 3851, 3852, 3895:3896])
            SysPar.carrier.N_grid_size_mu = SysPar.pdsch{1}.BWPSize;
        elseif ismember(caseNum, [3414:3421])
            SysPar.carrier.N_grid_size_mu = SysPar.pdsch{1}.BWPSize;
            SysPar.carrier.mu = 0;
        end
        if ismember(caseNum, [3279:3285,3289:3295, 3338:3341, 3855]) % to test resource allocation type 0
            SysPar.pdsch{1}.resourceAlloc = 0;
            if ismember(caseNum,[3279,3289,3855]) % BWP start from RB50 to 149. Note that the first RBG size is 6, thus the first byte is 00111111 (63) if the sec RBG is empty
                SysPar.pdsch{1}.rbBitmap = [63.0, 192.0, 63.0, 0.0, 0.0, 0.0, 192.0, 63.0, 0.0, 0.0,...
                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...
                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0];
            elseif ismember(caseNum,[3280,3290]) % refPoint = 1
                SysPar.pdsch{1}.refPoint = 1;
                SysPar.pdsch{1}.rbBitmap = [0.0, 0.0, 240.0, 0.0, 240.0, 240.0, 0.0, 0.0, 0.0, 0.0,...
                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...
                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0];
            elseif ismember(caseNum,[3281]) % modComp
                SysPar.pdsch{1}.powerControlOffset = 8 + 2;   % PDSCH/CSIRS = 2 dB         
                SysPar.pdsch{1}.powerControlOffsetSS = 0; % CSIRS/SSB = -3dB            
                SysPar.SimCtrl.genTV.fhMsgMode = 2;
                SysPar.pdsch{1}.rbBitmap = [255.0, 255.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 255.0, 255.0, ...
                                            255.0, 255.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 255.0, 255.0, ...
                                            255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 1.0, 0.0];
            elseif ismember(caseNum,[3282,3292])  % nCdm = 1. nl = 1
                SysPar.pdsch{1}.rbBitmap = [255.0, 255.0, 255.0, 255.0, 0.0, 0.0, 0.0, 0.0, 255.0, 255.0, ...
                                            0.0, 0.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 0.0, 0.0, ...
                                            255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 1.0, 0.0];
            elseif ismember(caseNum,[3283,3293]) % two segments far away from each other
                SysPar.pdsch{1}.rbBitmap = [255.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0];
            elseif ismember(caseNum,[3284,3294])   % max number of segments
                SysPar.pdsch{1}.rbBitmap = [255.0, 255.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 255.0, 255.0, ...
                                            0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 255.0, 255.0, ...
                                            0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0];
            elseif ismember(caseNum,[3285,3295]) % BWP size=30<32 PRBs, RBG size is 2,
                SysPar.pdsch{1}.rbBitmap = [0.0, 195.0, 204.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            elseif ismember(caseNum,[3338:3339])
                % UE1: 11111111 11111111 00000000 00000000 11111111 11111111 00000000 00000000, 
                % UE2: 00000000 00000000 11111111 11111111 00000000 00000000 11111111 11111111
                rbBitmap1 = [255.0, 255.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 0.0, 0.0,...
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0];
                rbBitmap2 = [0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0,...
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0];
            elseif ismember(caseNum, [3340])
                rbBitmap1 = [255.0, 255.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 0.0, 0.0,...
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0];
                rbBitmap2 = [0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 255.0, 255.0, 0.0, 255.0,...
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0];
                nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                            0  32   10  1  0   0    0;...
                           40  32    5  1  0   0    1];
            elseif ismember(caseNum, [3341])
                rbBitmap1 = [15.0,0.0, 0.0, 0.0,   15.0, 15.0,240.0, 0.0,  0.0, 0.0,...
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0];
                rbBitmap2 = [0.0, 0.0, 0.0, 255.0,240.0,  0.0,  0.0,255.0, 0.0, 0.0,...
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0];
                rbBitmap3 = [255.0,255.0,0.0,255.0,240.0,   0.0,  0.0, 0.0,  0.0, 0.0,...
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0];
                nUeCfg = [%rb0 nRb mcs nl p0 SCID idxUeg
                            0  16   27  1  0   0    0;...
                            8  16   17  1  0   0    1;... % resourceAlloc 1
                           32  20    7  1  0   0    2;...
                          128  28    1  1  0   0    3];
            end

        end
        % Precoding matricies for Resource Allocation Type 0
        if caseNum == 3289
            SysPar.pdsch{1}.prcdBf = 12; % PM_W44
        elseif ismember(caseNum, [3290:3293])
            SysPar.pdsch{1}.prcdBf = 4;
        elseif ismember(caseNum, [3294])
            SysPar.pdsch{1}.prcdBf = 8;  % PM_W24
        elseif caseNum == 3295 
            SysPar.pdsch{1}.prcdBf = 10; % PM_W34
        end
        
        if caseNum == 3343
            SysPar.carrier.N_ID_CELL = 42;
        end

        nUe = 1;
        if ~isempty(nUeCfg)            
            [nUe, nPara] = size(nUeCfg);
            SysPar.testAlloc.pdsch = nUe;
            pdsch0 = SysPar.pdsch{1};
            for idxUe = 1:nUe
                SysPar.pdsch{idxUe} = pdsch0;
                SysPar.pdsch{idxUe}.rbStart = nUeCfg(idxUe, 1);
                SysPar.pdsch{idxUe}.rbSize = nUeCfg(idxUe, 2);
                SysPar.pdsch{idxUe}.mcsIndex = nUeCfg(idxUe, 3);
                SysPar.pdsch{idxUe}.nrOfLayers = nUeCfg(idxUe, 4);
                SysPar.pdsch{idxUe}.portIdx = nUeCfg(idxUe, 5);    
                SysPar.pdsch{idxUe}.SCID = nUeCfg(idxUe, 6);
                SysPar.pdsch{idxUe}.idxUeg = nUeCfg(idxUe, 7);
                SysPar.pdsch{idxUe}.seed = SysPar.pdsch{idxUe}.seed + idxUe -1;
                if SysPar.SimCtrl.enable_dynamic_BF
                    SysPar.pdsch{idxUe}.numPRGs = ceil(SysPar.pdsch{idxUe}.rbSize/SysPar.pdsch{idxUe}.prgSize);
                end
                if nPara > 7
                    SysPar.pdsch{idxUe}.StartSymbolIndex = nUeCfg(idxUe, 8);
                    SysPar.pdsch{idxUe}.NrOfSymbols = nUeCfg(idxUe, 9);
                    sym0 = SysPar.pdsch{idxUe}.StartSymbolIndex;
                    nSym = SysPar.pdsch{idxUe}.NrOfSymbols;
                    dmrs0 = nUeCfg(idxUe, 10);
                    maxLen = nUeCfg(idxUe, 11);
                    addPos = nUeCfg(idxUe, 12);
                    if ismember(caseNum, [3255:3257, 3276, 3351])
                        DmrsSymbPos = findDmrsSymbPos(sym0, nSym, dmrs0, maxLen, addPos, 'DL', 'typeB');
                        SysPar.pdsch{idxUe}.DmrsMappingType = 1;
                    else
                        DmrsSymbPos = findDmrsSymbPos(sym0, nSym, dmrs0, maxLen, addPos, 'DL', 'typeA');
                    end
                    SysPar.pdsch{idxUe}.DmrsSymbPos = DmrsSymbPos;
                end                    
                if caseNum == 3259 % precoding
                    prcdBf_vec = [4 8 10 12];                    
                    SysPar.pdsch{idxUe}.prcdBf = prcdBf_vec(mod(idxUe-1, 4)+1);    
                elseif caseNum == 3264 % precoding
                    prcdBf_vec = [2 6 10 12];                    
                    SysPar.pdsch{idxUe}.prcdBf = prcdBf_vec(mod(idxUe-1, 4)+1);  
                elseif caseNum == 3265
                    SysPar.pdsch{idxUe}.powerControlOffset = 8 + 2*idxUe;   % PDSCH/CSIRS = 2, 4 dB         
                    SysPar.pdsch{idxUe}.powerControlOffsetSS = 0; % CSIRS/SSB = -3dB    
                elseif ismember(caseNum, [3266:3267]) % set different rvIndex for different UE
                    SysPar.pdsch{idxUe}.rvIndex = mod(idxUe-1, 4);
                elseif caseNum == 3288 % nCdm = 1 precoding (mixed mod/nl/nPort)
                    prcdBf_vec = [4 8];
                    SysPar.pdsch{idxUe}.prcdBf = prcdBf_vec(mod(idxUe-1, 2)+1);
                end
                SysPar.pdsch{idxUe}.idxUE = idxUe-1;
                if ismember(caseNum, [3850, 3851, 3852, 3853, 3854, 3855, 3879, 3881, 3882:3888,3890]) % Combined BFW-PDSCH test vector
                    RNTI_ = [1 4 6 7 8 9 10 3];
                    SysPar.pdsch{idxUe}.RNTI = RNTI_(idxUe);
                end
                if ismember(caseNum, [3869, 3870]) % Combined BFW-PDSCH test vector
                    RNTI_ = [4 13 6 11 8 3 16 14 5 15 9 7 10 2 12 1];
                    SysPar.pdsch{idxUe}.RNTI = RNTI_(idxUe);
                end
                if ismember(caseNum, [3873,3874]) % Combined BFW-PDSCH test vector
                    RNTI_ = [1 4];
                    SysPar.pdsch{idxUe}.RNTI = RNTI_(idxUe);
                end
                if ismember(caseNum, [3875,3876,3880]) % Combined BFW-PDSCH test vector
                    RNTI_ = [1 4 6 7];
                    SysPar.pdsch{idxUe}.RNTI = RNTI_(idxUe);
                end
                if ismember(caseNum, [3891:3896])
                    SysPar.pdsch{idxUe}.RNTI = idxUe;
                end
                if ismember(caseNum, [3360,3897:3900])
                    SysPar.pdsch{idxUe}.RNTI = idxUe + 6;
                end
            end
            if caseNum == 3254 % PM_W24 + PM_W14
                SysPar.pdsch{1}.prcdBf = 8;
                SysPar.pdsch{2}.prcdBf = 4;
            elseif caseNum == 3258 % PM_W24 + noPrcd
                SysPar.pdsch{1}.prcdBf = 8;
                SysPar.pdsch{2}.prcdBf = 0;
            elseif caseNum == 3278
                SysPar.pdsch{1}.BWPSize = 72;
                SysPar.pdsch{1}.BWPStart = 0;
                SysPar.pdsch{2}.BWPSize = 60;
                SysPar.pdsch{2}.BWPStart = 72;
                SysPar.pdsch{3}.BWPSize = 66;
                SysPar.pdsch{3}.BWPStart = 132;
                SysPar.pdsch{4}.BWPSize = 75;
                SysPar.pdsch{4}.BWPStart = 198;
            elseif caseNum == 3286 % FDM nCDM = 1 and 2
                SysPar.pdsch{2}.numDmrsCdmGrpsNoData = 2;
            elseif caseNum == 3287 % FDM nCDM = 1 and 2 with precoding
                SysPar.pdsch{2}.numDmrsCdmGrpsNoData = 2;
                SysPar.pdsch{1}.prcdBf = 4; % PM_W14
                SysPar.pdsch{2}.prcdBf = 8; % PM_W24
            elseif ismember(caseNum, [3338,3340])
                SysPar.pdsch{1}.rbBitmap = rbBitmap1;
                SysPar.pdsch{2}.rbBitmap = rbBitmap2;
                SysPar.pdsch{2}.BWPStart =  CFG{idxSet, 10};
            elseif caseNum == 3339
                SysPar.pdsch{1}.rbBitmap = rbBitmap1;
                SysPar.pdsch{2}.rbBitmap = rbBitmap2;
                SysPar.pdsch{2}.BWPStart =  CFG{idxSet, 10};
                SysPar.pdsch{1}.prcdBf = 4; % PM_W14
                SysPar.pdsch{2}.prcdBf = 4; % PM_W14
            elseif caseNum == 3341
                SysPar.pdsch{1}.rbBitmap = rbBitmap1;
                SysPar.pdsch{3}.rbBitmap = rbBitmap2;
                SysPar.pdsch{4}.rbBitmap = rbBitmap3;
                SysPar.pdsch{2}.resourceAlloc = 1;
                SysPar.pdsch{3}.resourceAlloc = 0;
                SysPar.pdsch{4}.resourceAlloc = 0;
                SysPar.pdsch{2}.BWPStart =  CFG{idxSet, 10};
                SysPar.pdsch{3}.BWPStart =  CFG{idxSet, 10};
                SysPar.pdsch{4}.BWPStart =  128;
            elseif caseNum == 3342
                SysPar.pdsch{1}.targetCodeRate = 30; % FAPI format: raw coderate x 1024 x 10
                SysPar.pdsch{1}.qamModOrder = 2;
                SysPar.pdsch{1}.TBSize = 478;
            elseif caseNum == 3881 % per-UE DMRS Scrambling Id
                for idxUe = 1:8
                    SysPar.pdsch{idxUe}.DmrsScramblingId = idxUe*41;
                end
            elseif caseNum == 3362
                for idxUe = 5:8
                    SysPar.pdsch{idxUe}.nlAbove16 = 1;
                end
            elseif caseNum == 3899
                for idxUe = 9:12
                    SysPar.pdsch{idxUe}.nlAbove16 = 1;
                end
            elseif caseNum == 3900
                for idxUe = 9:16
                    SysPar.pdsch{idxUe}.nlAbove16 = 1;
                end
            end
            SysPar.SimCtrl.N_UE = nUe;
        end
        
        if caseNum == 3805
            SysPar.pdsch{1}.portIdx = [0 2];
        end
        
        if caseNum == 3971
            nUe = 6;
            for idxUE = 1:nUe
                SysPar.pdsch{idxUE} = SysPar.pdsch{1};
                SysPar.pdsch{idxUE}.RNTI = 1000 * idxUE;
                SysPar.pdsch{idxUE}.rbStart = (idxUE-1)*45;
                SysPar.pdsch{idxUE}.rbSize = 45;
                SysPar.pdsch{idxUE}.idxUeg = idxUE-1;
                SysPar.pdsch{idxUE}.idxUE = idxUE-1;
            end
            SysPar.SimCtrl.N_UE = nUe;
            SysPar.testAlloc.pdsch = nUe;
        end
        
               
            
        if ismember(caseNum, [3972:3999])
            SysPar.pdsch{1}.RNTI = 35;
            if ismember(caseNum, [3972:3975])
                SysPar.pdsch{1}.targetCodeRate = 1570;
                SysPar.pdsch{1}.qamModOrder = 2;
                SysPar.pdsch{1}.TBSize = 6535;
            elseif ismember(caseNum, [3976:3979])
                SysPar.pdsch{1}.targetCodeRate = 6580;
                SysPar.pdsch{1}.qamModOrder = 4;
                SysPar.pdsch{1}.TBSize = 54285;
            elseif ismember(caseNum, [3980:3983])
                SysPar.pdsch{1}.targetCodeRate = 9480;
                SysPar.pdsch{1}.qamModOrder = 6;
                SysPar.pdsch{1}.TBSize = 118873;
            elseif ismember(caseNum, [3984:3987])
                SysPar.pdsch{1}.targetCodeRate = 1930;
                SysPar.pdsch{1}.qamModOrder = 2;
                SysPar.pdsch{1}.TBSize = 8072;
            elseif ismember(caseNum, [3988:3991])
                SysPar.pdsch{1}.targetCodeRate = 6580;
                SysPar.pdsch{1}.qamModOrder = 4;
                SysPar.pdsch{1}.TBSize = 54285;
            elseif ismember(caseNum, [3992:3995])
                SysPar.pdsch{1}.targetCodeRate = 8730;
                SysPar.pdsch{1}.qamModOrder = 6;
                SysPar.pdsch{1}.TBSize = 108573;
            elseif ismember(caseNum, [3996:3999])
                SysPar.pdsch{1}.targetCodeRate = 9480;
                SysPar.pdsch{1}.qamModOrder = 8;
                SysPar.pdsch{1}.TBSize = 159749;
            end
        end
        
        for idxUe = 1:length(SysPar.pdsch)
            digBFInterfaces = 1;
            switch SysPar.pdsch{idxUe}.prcdBf
                case 0
                    digBFInterfaces = SysPar.pdsch{idxUe}.nrOfLayers;
                case {1, 2, 5, 6}
                    digBFInterfaces = 2;
                case {3, 4, 7, 8, 9, 10, 11, 12}
                    digBFInterfaces = 4;
                otherwise
                    error('prcdBf is not supported ... \n');
            end
            SysPar.pdsch{idxUe}.digBFInterfaces = digBFInterfaces;
            SysPar.pdsch{idxUe}.beamIdx = [1:digBFInterfaces];
        end
        
        if ismember(caseNum, [3850 3851 3852 3853 3854 3855])
            for idxUe = 1:length(SysPar.pdsch)
                SysPar.pdsch{idxUe}.digBFInterfaces = 8;
                SysPar.pdsch{idxUe}.PMidx = 0;
                SysPar.pdsch{idxUe}.beamIdx = [1:SysPar.pdsch{idxUe}.digBFInterfaces];
            end
        end
        if ismember(caseNum, [3869:3900])
            for idxUe = 1:length(SysPar.pdsch)
                SysPar.pdsch{idxUe}.digBFInterfaces = 0;
                SysPar.pdsch{idxUe}.PMidx = 0;
                SysPar.pdsch{idxUe}.beamIdx = [1:16];
                if ismember(caseNum, [3899, 3900])
                    SysPar.pdsch{idxUe}.beamIdx = [1:32];
                end
            end
            if ismember(caseNum, [3889, 3897:3900]) % modComp
                SysPar.SimCtrl.genTV.fhMsgMode = 2;
            end
        end

        bypassDet = 0;
        if strcmp(caseSet, 'full') || strcmp(caseSet, 'compact')
            SysPar.SimCtrl.enableUeRx = 1;
            % bypass detection for mu-MIMO case (CDM)
            if SysPar.pdsch{nUe}.idxUeg+1 < nUe
%                 SysPar.carrier.Nant_gNB = 32;
%                 SysPar.SimCtrl.enableDlTxBf = 1;
                bypassDet = 1;
            end
            if caseNum == 3270 % BF with 32 antennas 
                bypassDet = 0;
            end
        end
        
        % update Nre_max which is used to calculate beta value in BFP
        SysPar.SimCtrl.oranComp.Nre_max = SysPar.carrier.N_grid_size_mu*12;
        
        % save SysPar into Cfg_<TC#>.yaml config file
        if SysPar.SimCtrl.genTV.genYamlCfg
            fileName = sprintf('Cfg_%04d.yaml', caseNum);
            WriteYaml(fileName, SysPar);
        end
        
        if SysPar.SimCtrl.genTV.enable && SysPar.SimCtrl.genTV.launchPattern
            if ~ismember(caseNum, disabled_TC)
                LPFileName = 'launch_pattern_nrSim';
            else
                LPFileName = 'disabled_launch_pattern_nrSim';
            end
            slotIdx =  SysPar.SimCtrl.genTV.slotIdx;
            genSingleSlotLPFile(LPFileName, caseNum, slotIdx);
        end
        
        if ismember(caseNum, [3248:3254, 3258:3260, 3264, 3287:3295, 3339]) % precoding TCs
            SysPar.SimCtrl.alg.enablePrcdBf = 1;
        end
        
        % For invalid PDSCH test cases
        if ismember(caseNum, [3501:3524])  % Tvs with invalid PDSCH params
            SysPar.SimCtrl.negTV.enable = 1;  % enable negTV
            % SimCrtl.negTV.pdufieldName: List of fields to be changed
            % SimCrtl.negTV.pdufieldValue: List of values correspoinding to SimCrtl.negTV.pdufieldName
            invalidCfgList = cell2mat(INVALID_CFG(:,1));
            invalidIdx = (caseNum == invalidCfgList);
            temp_INVALID_CFG = INVALID_CFG(invalidIdx, :);
            % fileds to save negTV params
            SysPar.SimCtrl.negTV.pduFieldName = [];
            SysPar.SimCtrl.negTV.pduFieldValue = [];
            % overwrite single value param
            for cfgFieldIdx = 1:length(INVALID_CFG_FIELD_NAMES)
                if(temp_INVALID_CFG{cfgFieldIdx+1} ~= -1) % if not -1 need to overwrite
                    SysPar.SimCtrl.negTV.pduFieldName = [SysPar.SimCtrl.negTV.pduFieldName INVALID_CFG_FIELD_NAMES(cfgFieldIdx)];
                    SysPar.SimCtrl.negTV.pduFieldValue = [SysPar.SimCtrl.negTV.pduFieldValue temp_INVALID_CFG(cfgFieldIdx+1)];
                end
            end
        end

        [SysPar, UE, gNB] = nrSimulator(SysPar);        
                
        Detected = 1;
        if SysPar.SimCtrl.enableUeRx
            results = SysPar.SimCtrl.results.pdsch;
            nPdsch = length(results);
            for idxPdsch = 1:nPdsch
                if (results{idxPdsch}.tbErrorCnt > 0)
                    Detected = 0;
                end
            end
            
            % bypass detection check for rvIdx cases and mu-MIMO w/o BF
            if ismember(caseNum, [3266:3267, 3296, 3354, 3355, 3973:3975, 3977:3979,...
                    3981:3983, 3985:3987, 3989:3991, 3993:3995, 3997:3999]) || bypassDet
                Detected = 1;
            end
            
            if ~Detected
                detErr = detErr + 1;
            end
        end        
        
        testPass = 1;
        % bypass precoding TCs for DlTxBF enabled
        if ismember(caseNum, [3248:3254, 3258:3260, 3264, 3287:3295, 3296, 3339, 3354, 3355]) || SysPar.SimCtrl.enableDlTxBf
            bypassCompTest = 1;
        else
            bypassCompTest = 0;
        end        
        if testCompliance && ~bypassCompTest
            nComp = nComp + 1;
            pdsch = gNB.Phy.Config.pdsch;
            carrier = gNB.Phy.Config.carrier;
            table = gNB.Phy.Config.table;
            Xtf_5g = hPDSCHGen(pdsch, carrier, table);
            
            % compare the last slot
            Xtf_nr = gNB.Phy.tx.Xtf;
%             figure; plot(real(reshape(Xtf_nr(:,:,1),1,[]))); hold on;plot(real(reshape(Xtf_5g(:,:,1),1,[])));
            err_Xtf = sum(sum(sum(abs(Xtf_nr - Xtf_5g))));
            
            testPass = (err_Xtf < 1e-4);
            if ~testPass
                errCnt = errCnt + 1;
            end
        end
        fprintf('%4d    %4d   %3d  %2d  %3d  %3d   %3d    %2d   %4d  %4d   %4d  %5d %4d   %4d   %4d   %4d     %4d   %4d   %4d  %4d  %4d  %4d   %4d %3d',...
            CFG{idxSet, 1}, CFG{idxSet, 2}, CFG{idxSet, 3}, CFG{idxSet, 4}, ...
            CFG{idxSet, 5}, CFG{idxSet, 6}, CFG{idxSet, 7}, CFG{idxSet, 8}, ...
            CFG{idxSet, 9}, CFG{idxSet, 10}, CFG{idxSet, 11}, CFG{idxSet, 12},...
            CFG{idxSet, 13}, CFG{idxSet, 14}, CFG{idxSet, 15}, CFG{idxSet, 16}, ...
            CFG{idxSet, 17}, CFG{idxSet, 18}, CFG{idxSet, 19}, CFG{idxSet, 20}, ...
            CFG{idxSet, 21}, CFG{idxSet, 22}, testPass, Detected);

        % print invalid params in PDSCH TVs if exist
        if(SysPar.SimCtrl.negTV.enable) % negTV enabled, extract the invalid params from SimCtrl
            pduFieldName = SysPar.SimCtrl.negTV.pduFieldName;
            pduFieldValue = SysPar.SimCtrl.negTV.pduFieldValue;
            fprintf('  Invalid params: ');
            for pduFieldIdx = 1:length(pduFieldName)
                fprintf(' %-*s:%4d\t', 20, pduFieldName{pduFieldIdx}, pduFieldValue{pduFieldIdx});
            end
        end

        % print a line break
        fprintf('\n');
    end
end

fprintf('------------------------------------------------------------------------------------------------------------------------------------------------------\n');
fprintf('Total Compliance TC = %d, PASS = %d, FAIL = %d, Total TV generated = %d\n\n', nComp, nComp-errCnt, errCnt, nTV);
toc; 
fprintf('\n');
