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

function [nComp, errCnt, nCuphyTV, nFapiTV, detErr] = testCompGenTV_ulmix(caseSet, compTvMode, subSetMod, relNum, algSel, VARY_PRB_NUM)

tic;
if nargin == 0
    caseSet = 'full';
    compTvMode = 'both';
    subSetMod = [0, 1];
    relNum = 10000;
    algSel = struct();
    VARY_PRB_NUM = -1;
elseif nargin == 1
    compTvMode = 'both';
    subSetMod = [0, 1];
    relNum = 10000;
    algSel = struct();
    VARY_PRB_NUM = -1;
elseif nargin == 2
    subSetMod = [0, 1];
    relNum = 10000;
    algSel = struct();
    VARY_PRB_NUM = -1;
elseif nargin == 3
    relNum = 10000;
    algSel = struct();
    VARY_PRB_NUM = -1;
elseif nargin == 4
    algSel = struct();
    VARY_PRB_NUM = -1;
elseif nargin == 5  
    VARY_PRB_NUM = -1;
end

switch compTvMode
    case 'both'
        genTV = 1;
        testCompliance = 0;
        fprintf('only support ''genTV'' \n');
    case 'genTV'
        genTV = 1;
        testCompliance = 0;
    case 'testCompliance'
        genTV = 0;
        testCompliance = 1;
        error('only support ''genTV'' \n');
    otherwise
        error('compTvMode is not supported...\n');
end

selected_TC = [500:999, 1000:8555, 20000:29999];
disabled_TC = [];
% These are tests that are run from elsewhere.  Generating them explicity
% should work but we don't want them generated as part of selected TCs
% because it will create race conditions if they are simultaneously
% generated in 2 places
bypass_TC = [743,763,783,803,823,843,863 21203:20:21383, 21423:10:21493, 4791:4805, 5461:5475, 5581:5595, 5731:5745, 5851:5865, 4836:4850, 21523, 21563, 21583, 21603, 21623, 21640:21643, 21653, 21663, 21673, 21693, 21713, 21723, 21733:21735,21743, 21753, 21763, 21772, 21773, 21793:10:21903];
[~,TcIdx] = ismember([disabled_TC,bypass_TC], selected_TC);
TcIdx = TcIdx(TcIdx > 0);  % Filter out zero indices to prevent out-of-bounds error
selected_TC(TcIdx) = [];

HARQ_TC = [550:589];

% performance pattern table
% columns: pattern#, start_tv, end_tv, n_cell, n_cell_to_gen
% n_cell_to_gen: min(total_cells, 21) for 4TR (pattern#<=65), min(total_cells, 7) for 64TR (pattern#>65)
%                set to 1 if not main perf patterns
% TV configs are for n_cell but only generate n_cell_to_gen cells for compact set
perf_pattern_table = [
% pattern     start_tv  end_tv  n_cell, n_cell_to_gen
    39,         1632,    1695,    16,     1;  % 39
    40,         1696,    1743,    12,     1;  % 40
    41,         1744,    1791,    12,     1;  % 41
    41.1,       2304,    2351,    12,     1;  % 41a
    41.2,       2640,    2687,    12,     1;  % 41b
    41.3,       2688,    2735,    12,     1;  % 41c
    42,         1792,    1839,    12,     1;  % 42
    43,         1840,    1903,    16,     1;  % 43, 44
    44.1,       2352,    2415,    16,     1;  % 44a
    44.2,       2736,    2799,    16,     1;  % 44b
    44.3,       2800,    2863,    16,     1;  % 44c
    45,         1904,    1967,    16,     1;  % 45
    46,         1968,    2015,    12,     1;  % 46
    47,         2016,    2079,    16,     1;  % 47
    48,         2080,    2143,    16,    16;  % 48
    49,         2144,    2223,    20,    20;  % 49
    50,         2224,    2287,    16,    16;  % 50
    51,         2416,    2495,    20,    20;  % 51
    53,         2496,    2559,    16,     1;  % 53
    54,         2560,    2639,    20,     1;  % 54
    55,         2896,    2959,    16,     1;  % 55
    56,         2960,    3039,    20,     1;  % 56
    57,         2864,    2895,     8,     1;  % 57
    59,         3040,    3119,    20,    20;  % 59 (L2SA)
    59.2,       3760,    3839,    20,    20;  % 59b
    59.3,       4471,    4630,    40,    20;  % 59c
    59.4,       3520,    3599,    20,    20;  % 59d    
    59.5,       4631,    4790,    40,    20;  % 59e
    60,         3600,    3759,    40,    21;  % 60 (OAM)
    60.3,       5071,    5230,    40,    21;  % 60c
    60.4,       5231,    5390,    40,    21;  % 60d
    61,         3360,    3439,    20,     1;  % 61
    62.2,       4040,    4079,    20,    20;  % 62c (also needs 4471:4630 from 59c)
    63.2,       5391,    5430,    20,    20;  % 63c (also needs 5071:5230 from 60c)
    65.1,       5971,    6010,    40,    21;  % 65a
    65.2,       6011,    6050,    40,    21;  % 65b
    65.3,       6051,    6210,    40,    21;  % 65c
    65.4,       6211,    6370,    40,    21;  % 65d
    66.1,       5431,    5445,    15,     7;  % 66a
    66.2,       5446,    5460,    15,     7;  % 66b
    66.3,       5461,    5580,    15,     7;  % 66c
    66.4,       5581,    5700,    15,     7;  % 66d
    67,         4791,    4910,    15,     7;  % 67
    67.1,       5701,    5715,    15,     7;  % 67a
    67.2,       5716,    5730,    15,     7;  % 67b
    67.3,       5731,    5850,    15,     7;  % 67c
    67.4,       5851,    5970,    15,     7;  % 67d    
    69,         6621,    6695,    15,     7;  % 69
    69.1,       6696,    6770,    15,     7;  % 69a
    69.2,       6771,    6845,    15,     7;  % 69b
    69.3,       6371,    6445,    15,     7;  % 69c
    69.4,       6846,    6920,    15,     7;  % 69d
    69.5,       7260,    7364,    15,     7;  % 69e
    71,         6446,    6520,    15,     7;  % 71
    73,         6521,    6620,    20,    10;  % 73
    75,         6921,    6995,    15,     7;  % 75
    79,         7155,    7259,    15,     7;  % 79
    81.1,       7365,    7469,    15,     7;  % 81a, 81c
    81.2,       7470,    7574,    15,     7;  % 81b, 81d
    83.1,       7575,    7679,    15,     7;  % 83a, 83c
    83.2,       7680,    7784,    15,     7;  % 83b, 83d
    85,         7785,    7889,    15,     7;  % 85
    % 87 reuse the ULMIX TVs from pattern 85
    89,         7890,    7952,     9,     9;  % 89
    79.1,       7953,    8057,    15,     6;  % 79a
    79.2,       8058,    8162,    15,     6;  % 79b
    91,         8163,    8267,    15,     7;  % 91
    101,        4375,    4470,    24,    24;  % 101
    101.1,      8268,    8363,    24,    24;  % 101a
    102,        8364,    8459,    24,    24;  % 102
    102.1,      8460,    8555,    24,    24;  % 102a
];

% generate compact TV list from performance patterns
compact_TV_perf_pattern = [];
for i = 1:size(perf_pattern_table, 1)
    start_tv = perf_pattern_table(i, 2);
    end_tv = perf_pattern_table(i, 3);
    n_cell = perf_pattern_table(i, 4);
    n_cell_to_gen = perf_pattern_table(i, 5);
    
    n_tv_set = (end_tv - start_tv + 1) / n_cell;
    compact_TV_perf_pattern = [compact_TV_perf_pattern, repelem(start_tv:n_cell:end_tv, n_cell_to_gen) + repmat(0:(n_cell_to_gen - 1), 1, n_tv_set)];
end

% non-performance pattern compact TVs
compact_TV_non_perf_pattern = [508:519, 530:547, 550:589, 590, 603, 605, 606, 607, 608, 609, 610, 611, 612:618, 619:630, 636, 637, 644, 645, 652:658, 660, 661, 700:879, ...
                                3920:3927, ... % ULMIX TVs with PRACH + PUCCH + PUSCH + SRS
                                3928:3941, ... % SRS in consecutive slots with SRS + PUCCH
                                4300:4336, ... % SRS in consecutive slots with SRS + other channels
                                3840, 6996, ... % negative TC
                                20000:29999]; % mMIMO, multi-cell % negative TC

% combine both compact TV sets
compact_TC = [compact_TV_non_perf_pattern, compact_TV_perf_pattern];
compact_TC = unique(compact_TC);

% only generate FAPI TV in this set for per-MR cicd
% keep cuPHY TVs for one of each kind
% generate from perf_pattern_table using setdiff(start:end, start:n_cell:end)
compact_TV_FAPI_only = [];
for i = 1:size(perf_pattern_table, 1)
    start_tv = perf_pattern_table(i, 2);
    end_tv = perf_pattern_table(i, 3);
    n_cell = perf_pattern_table(i, 4);
    
    % keep all TCs except the first one of each cell (reserved for cuPHY)
    all_tvs = start_tv:end_tv;
    cuphy_tvs = start_tv:n_cell:end_tv;  % First TC of each cell
    fapi_only_tvs = setdiff(all_tvs, cuphy_tvs);
    compact_TV_FAPI_only = [compact_TV_FAPI_only, fapi_only_tvs];
end

% performance pattern compact TVs required for cuBB GPU test bench
compact_TV_perf_pattern_cuBB_gpu = [
    4544, 4548, 4566, ... % from 59c
    4382, 4417, ... % from 101
    8310, 8337, ... % from 101a
    8375, 8393, ... % from 102
];
compact_TV_FAPI_only = setdiff(compact_TV_FAPI_only, compact_TV_perf_pattern_cuBB_gpu);  % exclude cuBB GPU test bench TVs from FAPI-only set, will generate both cuPHY and FAPI TVs

full_TC = [500:999, 1000:8555, 20000:29999];

MIMO_64TR_TC = [590, 700:879, 3920:3941, 4300:4336, 4791:4910, 5431:5970, 6371:6445, 6446:6520, 6521:6620, 6621:6695, 6696:6770, 6771:6845, 6846:6920, 6921:6995, 6996, 7050:7952, 8163:8267, 21000:29999];

negative_TC = [605, 606, 607, 608, 609, 610, 611, 3840, 6996];

if isnumeric(caseSet)
    TcToTest = caseSet;
else
    switch caseSet
        case 'harq'
            TcToTest = HARQ_TC;
        case 'compact'
            TcToTest = compact_TC;
        case 'full'
            TcToTest = full_TC;
        case 'selected'
            TcToTest = selected_TC;
        otherwise
            error('caseSet is not supported...\n');
    end
    % Remove bypassed TCs - they are generated elsewhere
    TcToTest = setdiff(TcToTest,bypass_TC);
end

% TcFapiOnly is the set of TCs that we only generate FAPI TVs
if strcmp(caseSet, 'compact') || strcmp(caseSet, 'full')
    TcFapiOnly = compact_TV_FAPI_only;
else
    TcFapiOnly = [];
end
TcFapiOnly = [TcFapiOnly, negative_TC];

% 64TR beamIdx range 
DL_OFFSET = 0; % UL Beam IDs take lower numbers than DL so no offset needed
beamIdx_prach_static  = DL_OFFSET + [1024:1100];
beamIdx_pucch_static  = DL_OFFSET + [101:200];
beamIdx_srs_static    = DL_OFFSET + [201:250]; % not used
beamIdx_pusch_static  = DL_OFFSET + [251:370];
beamIdx_pusch_dynamic = DL_OFFSET + [371:500];

USAGE_BEAM_MGMT = 1;
USAGE_CODEBOOK = 2;
USAGE_NON_CODEBOOK = 4;

PrachBeamIdxMap = containers.Map('KeyType','int32', 'ValueType','any');
CellIdxInPatternMap = containers.Map('KeyType','int32', 'ValueType','any');
puschDynamicBfMap = containers.Map('KeyType','int32', 'ValueType','any');
UciOnPuschMap = containers.Map('KeyType','int32', 'ValueType','any');
enableOtaMap = containers.Map('KeyType','int32', 'ValueType','any');  % all TCs in MIMO_64TR_TC will enable OTA, in addition to those TCs in enableOtaMap
puschIdentityMap = containers.Map('KeyType','int32', 'ValueType','any');  % for generating scrambling sequence specifically for DFT-s-OFDM
enableUlRxBfMap = containers.Map('KeyType','int32', 'ValueType','any');  % for applying RX beamforming to PUSCH
earlyHarqDisableMap = containers.Map('KeyType','int32', 'ValueType','any');  % for disabling early HARQ feedback
harqProcessIdMap = containers.Map('KeyType','int32', 'ValueType','any');  % for setting HARQ process ID
srsRntiStartMap = containers.Map('KeyType','int32', 'ValueType','any');  % for setting SRS RNTI
ldpcFixMaxIterMap = containers.Map('KeyType','int32', 'ValueType','any');  % for setting fixed LDPC iteration number

CFG = {...
  % TC#   slotIdx   cell     prach     pucch      pusch      srs
%     500     0        1          {}        {}        {}        {};
%     501     1        1          {}        {}        {}        {};
%     502     2        1          {}        {}        {}        {};
%     503     3        1          {}    {1, 2}     {1, 2}       {};
    504     4        1          {}    {3, 4}     {3, 4}       {};
    505     5        1         {1}    {3, 4}     {5, 6}       {};
%     506     6        1          {}        {}        {}        {};
%     507     7        1          {}        {}        {}        {};
    % DSUUU 1UEs
      508      0        2         {4}     {7}        {550}     {3};
      509      0        2          {}     {7}        {550}     {3};
      510      0        2          {}     {7}        {550}     {3};
      511      0        2          {}     {7}        {550}     {3};
      512      0        2          {}     {7}        {550}     {3};    
      513      0        2          {}     {7}        {550}     {3};
      514      0        2          {}     {7}        {550}     {3};
      515      0        2          {}     {7}        {550}     {3};
      516      0        2          {}     {7}        {550}     {3};
      517      0        2          {}     {7}        {550}     {3};
      518      0        2          {}     {7}        {550}     {3};
      519      0        2          {}     {7}        {550}     {3};    
    % dynamic cell initialization
      520      0        2         {7}     {5}       {14}        {};
      521      0        2      {7, 8}     {5}       {14}        {};
      522      0        2        {11}     {5}       {14}        {};
    % 4T4R-SRS cases
      530      0        2        {12}      {}              {159}   {};
      531      0        2          {}      {}              {160}   {};
      532      0        2        {12}      {}         {303, 304}   {};
      533      0        2          {}      {}         {305, 306}   {};
    % 32T32R-SRS
      534      0        8        {12}      {}         {307, 308}   {};
      535      0        8          {}      {}         {309, 310}   {};
    % DSUUU 4UEs
      536      0        2         {4}     {7}   num2cell(550:553) {3}; 
      537      0        2          {}     {7}   num2cell(550:553) {3};
      538      0        2          {}     {7}   num2cell(550:553) {3};
      539      0        2          {}     {7}   num2cell(550:553) {3}; 
      540      0        2          {}     {7}   num2cell(550:553) {3};
      541      0        2          {}     {7}   num2cell(550:553) {3};
      542      0        2          {}     {7}   num2cell(550:553) {3}; 
      543      0        2          {}     {7}   num2cell(550:553) {3};
      544      0        2          {}     {7}   num2cell(550:553) {3};
      545      0        2          {}     {7}   num2cell(550:553) {3}; 
      546      0        2          {}     {7}   num2cell(550:553) {3};
      547      0        2          {}     {7}   num2cell(550:553) {3};
    % DSUUU 2UEs
      619      0        2         {4}     {7}   num2cell(550:551) {3};
      620      0        2          {}     {7}   num2cell(550:551) {3};
      621      0        2          {}     {7}   num2cell(550:551) {3};
      622      0        2          {}     {7}   num2cell(550:551) {3};
      623      0        2          {}     {7}   num2cell(550:551) {3};
      624      0        2          {}     {7}   num2cell(550:551) {3};
      625      0        2          {}     {7}   num2cell(550:551) {3};
      626      0        2          {}     {7}   num2cell(550:551) {3};
      627      0        2          {}     {7}   num2cell(550:551) {3};
      628      0        2          {}     {7}   num2cell(550:551) {3};
      629      0        2          {}     {7}   num2cell(550:551) {3};
      630      0        2          {}     {7}   num2cell(550:551) {3};

       
    % HARQ TC
  % TC#   slotIdx cell     prach               pucch             pusch          srs
      553      3   9               {}      num2cell(10:33)              {311}   {};
      554      4   9               {}                   {}  num2cell(312:317)   {};
      555      5   9  num2cell(41:44)      num2cell(34:41)                 {}   {};
      563     13   9               {}                   {}              {318}   {};
      564     14   9               {}                   {}              {319}   {};
      565     15   9               {}                   {}  num2cell(320:325)   {};
      573     23   9               {}      num2cell(10:33)              {326}   {};
      574     24   9               {}                   {}  num2cell(327:332)   {};
      575     25   9  num2cell(41:43)      num2cell(34:41)                 {}   {};
      583     33   9               {}                   {}              {333}   {};
      584     34   9               {}      num2cell(34:41)         {334, 335}   {};
      585     35   9               {}      num2cell(34:41)  num2cell(336:341)   {};
      
   % 64TR
      590      5  11  num2cell(61:64)      num2cell(60:67)  num2cell(610:615)  {9};

  % 64TR integration SU-MIMO 1-UEG
  % TC#   slotIdx cell     prach               pucch             pusch          srs
      703      3  11               {}                   {}                 {}  {7};
      704      4  11               {}      num2cell(46:53)              {610}   {};
      705      5  11  num2cell(61:64)      num2cell(46:53)              {610}   {};
      713     13  11               {}                   {}                 {}  {7};  % start RNTI=2 using srsRntiStartMap
      714     14  11               {}      num2cell(46:53)              {610}   {};
      715     15  11               {}      num2cell(46:53)              {610}   {};
  % 64TR integration SU-MIMO 2-UEG
      723      3  11               {}                   {}                 {}  {8};
      724      4  11               {}      num2cell(46:53)         {610, 614}   {};
      725      5  11  num2cell(61:64)      num2cell(46:53)         {610, 614}   {};
      733     13  11               {}                   {}                 {}  {8};  % start RNTI=3 using srsRntiStartMap
      734     14  11               {}      num2cell(46:53)         {610, 614}   {};
      735     15  11               {}      num2cell(46:53)         {610, 614}   {};
  % 64TR integration SU/MU-MIMO 3-UEG
      743      3  11               {}                   {}                 {}  {9};
      744      4  11               {}      num2cell(46:53)  num2cell(610:615)   {};
      745      5  11  num2cell(61:64)      num2cell(46:53)  num2cell(610:615)   {};
     %753     13  11               {}                   {}                 {}   {};
      754     14  11               {}      num2cell(46:53)  num2cell(610:615)   {};
      755     15  11               {}      num2cell(46:53)  num2cell(610:615)   {};
  % 64TR integration SU/MU-MIMO 1 dynamic + 1 static UEG
      763      3  11               {}                   {}                 {}  {9};
      764      4  11               {}      num2cell(46:53)          {610,614}   {};
      765      5  11  num2cell(61:64)      num2cell(46:53)          {610,614}   {};
     %773     13  11               {}                   {}                 {}   {};
      774     14  11               {}      num2cell(46:53)          {610,614}   {};
      775     15  11               {}      num2cell(46:53)          {610,614}   {};
  % 64TR integration SU/MU-MIMO 2 dynamic UE + 1 static UEG
      783      3  11               {}                   {}                 {}  {9};
      784      4  11               {}      num2cell(46:53)      {610,611,614}   {};
      785      5  11  num2cell(61:64)      num2cell(46:53)      {610,611,614}   {};
     %793     13  11               {}                   {}                 {}   {};
      794     14  11               {}      num2cell(46:53)      {610,611,614}   {};
      795     15  11               {}      num2cell(46:53)      {610,611,614}   {};
  % 64TR integration SU-MIMO 1-UEG (8 UE SRS)
      803      3  11               {}                   {}                 {}  {9};
      804      4  11               {}      num2cell(46:53)              {610}   {};
      805      5  11  num2cell(61:64)      num2cell(46:53)              {610}   {};
     %813     13  11               {}                   {}                 {}   {};
      814     14  11               {}      num2cell(46:53)              {610}   {};
      815     15  11               {}      num2cell(46:53)              {610}   {};
  % 64TR test case
      823      3  11               {}                   {}                 {} {12};
      824      4  11               {}                 {55}  num2cell(664:669)   {};
      825      5  11             {69}                 {56}  num2cell(664:669)   {};
      833     13  11               {}                   {}                 {} {13};
      834     14  11               {}                 {57}  num2cell(664:669)   {};
      835     15  11             {69}             {58, 59}  num2cell(664:669)   {};
  % 64TR test case, BFPforCuphy = 9
      843      3  11               {}                   {}                 {} {12};
      844      4  11               {}                 {55}  num2cell(664:669)   {};
      845      5  11             {69}                 {56}  num2cell(664:669)   {};
      853     13  11               {}                   {}                 {} {13};
      854     14  11               {}                 {57}  num2cell(664:669)   {};
      855     15  11             {69}             {58, 59}  num2cell(664:669)   {};
  % 64TR test case, 4 SRS symbols
      863      3  11               {}                   {}                 {} {20};
      873     13  11               {}                   {}                 {} {21};

  % 64TR integration SU-MIMO 1-UEG - 40 slots
  % TC#   slotIdx cell     prach               pucch             pusch          srs
    21003      3  11               {}                   {}                 {}  {7}; %cell-id 41
    21004      4  11               {}      num2cell(46:53)              {610}   {};
    21005      5  11  num2cell(61:64)      num2cell(46:53)              {610}   {};
    21013     13  11               {}                   {}                 {}  {25};
    21014     14  11               {}      num2cell(46:53)              {610}   {};
    21015     15  11               {}      num2cell(46:53)              {610}   {};
    21023     23  11               {}                   {}                 {}  {7};
    21024     24  11               {}      num2cell(46:53)              {610}   {};
    21025     25  11  num2cell(61:64)      num2cell(46:53)              {610}   {};
    21033     33  11               {}                   {}                 {}  {25};
    21034     34  11               {}      num2cell(46:53)              {610}   {};
    21035     35  11               {}      num2cell(46:53)              {610}   {};
    21043      3  11               {}                   {}                 {}  {7}; %cell-id 42
    21044      4  11               {}      num2cell(46:53)              {610}   {};
    21045      5  11  num2cell(61:64)      num2cell(46:53)              {610}   {};
    21053     13  11               {}                   {}                 {}  {7};
    21054     14  11               {}      num2cell(46:53)              {610}   {};
    21055     15  11               {}      num2cell(46:53)              {610}   {};
    21063     23  11               {}                   {}                 {}  {7};
    21064     24  11               {}      num2cell(46:53)              {610}   {};
    21065     25  11  num2cell(61:64)      num2cell(46:53)              {610}   {};
    21073     33  11               {}                   {}                 {}  {7};
    21074     34  11               {}      num2cell(46:53)              {610}   {};
    21075     35  11               {}      num2cell(46:53)              {610}   {};
    21083      3  11               {}                   {}                 {}  {7}; %cell-id 43
    21084      4  11               {}      num2cell(46:53)              {610}   {};
    21085      5  11  num2cell(61:64)      num2cell(46:53)              {610}   {};
    21093     13  11               {}                   {}                 {}  {7};
    21094     14  11               {}      num2cell(46:53)              {610}   {};
    21095     15  11               {}      num2cell(46:53)              {610}   {};
    21103     23  11               {}                   {}                 {}  {7};
    21104     24  11               {}      num2cell(46:53)              {610}   {};
    21105     25  11  num2cell(61:64)      num2cell(46:53)              {610}   {};
    21113     33  11               {}                   {}                 {}  {7};
    21114     34  11               {}      num2cell(46:53)              {610}   {};
    21115     35  11               {}      num2cell(46:53)              {610}   {};
    21123      3  11               {}                   {}                 {}  {7}; %cell-id 44
    21124      4  11               {}      num2cell(46:53)              {610}   {};
    21125      5  11  num2cell(61:64)      num2cell(46:53)              {610}   {};
    21133     13  11               {}                   {}                 {}  {7};
    21134     14  11               {}      num2cell(46:53)              {610}   {};
    21135     15  11               {}      num2cell(46:53)              {610}   {};
    21143     23  11               {}                   {}                 {}  {7};
    21144     24  11               {}      num2cell(46:53)              {610}   {};
    21145     25  11  num2cell(61:64)      num2cell(46:53)              {610}   {};
    21153     33  11               {}                   {}                 {}  {7};
    21154     34  11               {}      num2cell(46:53)              {610}   {};
    21155     35  11               {}      num2cell(46:53)              {610}   {};
    21163      3  11               {}                   {}                 {}  {7}; %cell-id 45
    21164      4  11               {}      num2cell(46:53)              {610}   {};
    21165      5  11  num2cell(61:64)      num2cell(46:53)              {610}   {};
    21173     13  11               {}                   {}                 {}  {7};
    21174     14  11               {}      num2cell(46:53)              {610}   {};
    21175     15  11               {}      num2cell(46:53)              {610}   {};
    21183     23  11               {}                   {}                 {}  {7};
    21184     24  11               {}      num2cell(46:53)              {610}   {};
    21185     25  11  num2cell(61:64)      num2cell(46:53)              {610}   {};
    21193     33  11               {}                   {}                 {}  {7};
    21194     34  11               {}      num2cell(46:53)              {610}   {};
    21195     35  11               {}      num2cell(46:53)              {610}   {};

   % 64TR integration SU/MU-MIMO 3-UEG
    21203      3  11               {}                   {}                 {}  {9}; %cell-41
    21204      4  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21205      5  11  num2cell(61:64)      num2cell(46:53)  num2cell(610:615)   {};
   %21213     13  11               {}                   {}                 {}   {};
    21214     14  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21215     15  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21223      3  11               {}                   {}                 {}  {9}; %cell-42
    21224      4  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21225      5  11  num2cell(61:64)      num2cell(46:53)  num2cell(610:615)   {};
   %21233     13  11               {}                   {}                 {}   {};
    21234     14  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21235     15  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21243      3  11               {}                   {}                 {}  {9}; %cell-43
    21244      4  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21245      5  11  num2cell(61:64)      num2cell(46:53)  num2cell(610:615)   {};
   %21253     13  11               {}                   {}                 {}   {};
    21254     14  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21255     15  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21263      3  11               {}                   {}                 {}  {9}; %cell-44
    21264      4  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21265      5  11  num2cell(61:64)      num2cell(46:53)  num2cell(610:615)   {};
   %21273     13  11               {}                   {}                 {}   {};
    21274     14  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21275     15  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21283      3  11               {}                   {}                 {}  {9}; %cell-45
    21284      4  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21285      5  11  num2cell(61:64)      num2cell(46:53)  num2cell(610:615)   {};
   %21293     13  11               {}                   {}                 {}   {};
    21294     14  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21295     15  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21303      3  11               {}                   {}                 {}  {9}; %cell-46
    21304      4  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21305      5  11  num2cell(61:64)      num2cell(46:53)  num2cell(610:615)   {};
   %21313     13  11               {}                   {}                 {}   {};
    21314     14  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21315     15  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21323      3  11               {}                   {}                 {}  {9}; %cell-47
    21324      4  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21325      5  11  num2cell(61:64)      num2cell(46:53)  num2cell(610:615)   {};
   %21333     13  11               {}                   {}                 {}   {};
    21334     14  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21335     15  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21343      3  11               {}                   {}                 {}  {9}; %cell-48
    21344      4  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21345      5  11  num2cell(61:64)      num2cell(46:53)  num2cell(610:615)   {};
   %21353     13  11               {}                   {}                 {}   {};
    21354     14  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21355     15  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21363      3  11               {}                   {}                 {}  {9}; %cell-49
    21364      4  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21365      5  11  num2cell(61:64)      num2cell(46:53)  num2cell(610:615)   {};
   %21373     13  11               {}                   {}                 {}   {};
    21374     14  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21375     15  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21383      3  11               {}                   {}                 {}  {9}; %cell-50
    21384      4  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21385      5  11  num2cell(61:64)      num2cell(46:53)  num2cell(610:615)   {};
   %21393     13  11               {}                   {}                 {}   {};
    21394     14  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21395     15  11               {}      num2cell(46:53)  num2cell(610:615)   {};
    21404      4  11               {}      num2cell(60:67)  num2cell(610:615) {16}; %744 + SRS on last 2 symbols
    21405      5  11  num2cell(61:64)      num2cell(60:67)  num2cell(610:615) {17}; %745 + SRS on last 2 symbols
    21414     14  11               {}      num2cell(60:67)  num2cell(610:615) {18}; %754 + SRS on last 2 symbols
    21415     15  11               {}      num2cell(60:67)  num2cell(610:615) {19}; %755 + SRS on last 2 symbols

% 64TR MU-MIMO, 3 peak cells
    % TC#   slotIdx cell     prach               pucch             pusch          srs
    21423      3  11               {}                   {}                 {}   {11}; %cell-id 41, 16 SRS UEs
    21424     24  11               {}                   {}  num2cell(682:689)   {};
    21425     25  11               {}                   {}  num2cell(682:689)   {};
    21433      3  11               {}                   {}                 {}   {9}; % 8 SRS UEs
    21434     34  11               {}                   {}  num2cell(682:689)   {};
    21435     35  11               {}                   {}  num2cell(682:689)   {};
    21443      3  11               {}                   {}                 {}   {11}; %cell-id 42, 16 SRS UEs
    21444     24  11               {}                   {}  num2cell(682:689)   {};
    21445     25  11               {}                   {}  num2cell(682:689)   {};
    21453      3  11               {}                   {}                 {}   {9}; % 8 SRS UEs
    21454     34  11               {}                   {}  num2cell(682:689)   {};
    21455     35  11               {}                   {}  num2cell(682:689)   {};
    21463      3  11               {}                   {}                 {}   {11}; %cell-id 43, 16 SRS UEs
    21464     24  11               {}                   {}  num2cell(682:689)   {};
    21465     25  11               {}                   {}  num2cell(682:689)   {};
    21473      3  11               {}                   {}                 {}   {9}; % 8 SRS UEs
    21474     34  11               {}                   {}  num2cell(682:689)   {};
    21475     35  11               {}                   {}  num2cell(682:689)   {};
  % 64TR 'worst case' column B
   %21483      3  11               {}                   {}                 {} {12}; % Reusing 823
    21484      4  11               {}                 {55}  num2cell(664:669)   {};
    21485      5  11             {69}                 {56}  num2cell(664:669)   {};
   %21493     13  11               {}                   {}                 {} {13}; % Reusing 833
    21494     14  11               {}                 {57}  num2cell(664:669)   {};
    21495     15  11             {69}             {58, 59}  num2cell(664:669)   {};
    21503      3  11               {}                   {}                 {} {22};
    21513     13  11               {}                   {}                 {} {23};
  % 64TR 'worst case' column G
    % MU-MIMO slots
    21523      3  11               {}                   {}                 {} {24};
    21524      4  11               {}                 {76}  num2cell(738:745)   {};
    21525      5  11          num2cell(70:73)         {73}  num2cell(746:753)   {};
    21533      3  11               {}                   {}                 {} {26};
    21534     14  11               {}              {75,74}  num2cell(738:745)   {};
    21535     15  11          num2cell(70:73)         {72}  num2cell(746:753)   {};
    % SU-MIMO slots
    % 21544      4  11               {}                 {76}  num2cell([754,665:669])   {};
    % 21545      5  11          num2cell(70:73)         {73}  num2cell(755:760)   {};
    % 21554     14  11               {}              {75,74}  num2cell([754,665:669])   {};
    % 21555     15  11          num2cell(70:73)         {72}  num2cell(755:760)   {};
  % 64TR 'worst case' column H
    % TC#   slotIdx cell     prach               pucch             pusch          srs
    % MU-MIMO slots
    21563      3  11               {}                   {}                 {}    {27};  % (SRS 80) = 80 PDUs
    21564      4  11               {}                 {81,82}  num2cell(767:774) {31};  % (PUCCH 16 + 72) + (PUSCH 8)  + (SRS 24) = 120 PDUs
    21565      5  11          num2cell(70:73)         {78,83}  num2cell(775:782) {32};  % (PRACH 4) + (PUCCH 32 + 72) + (PUSCH 8)  + (SRS 24) = 140 PDUs
    21573      3  11               {}                   {}                 {}    {28};  % (SRS 80) = 80 PDUs
    21574     14  11               {}              {80,79,82}  num2cell(767:774) {33};  % (PUCCH 1 + 21 + 72) + (PUSCH 8)  + (SRS 24) = 126 PDUs
    21575     15  11          num2cell(70:73)         {77,83}  num2cell(775:782) {34};  % (PRACH 4) + (PUCCH 64 + 72) + (PUSCH 8)  + (SRS 24) = 172 PDUs
  % 64TR 'worst case' column G, 40 MHz
    % MU-MIMO slots
    21583      3  13               {}                   {}                 {} {29};
    21584      4  13               {}                 {76}  num2cell(783:790)   {};
    21585      5  13          num2cell(76:79)         {73}  num2cell(791:798)   {};
    21593      3  13               {}                   {}                 {} {30};
    21594     14  13               {}              {75,74}  num2cell(783:790)   {};
    21595     15  13          num2cell(76:79)         {72}  num2cell(791:798)   {};
  % 64TR 'worst case' column G, 64 UEs per TTI
    % MU-MIMO slots
    21603      3  11               {}                   {}                 {}    {27};
    21604      4  11               {}                 {76}  num2cell(2001:2064)   {};
    21605      5  11          num2cell(70:73)         {73}  num2cell(2065:2128)   {};
    21613      3  11               {}                   {}                 {}    {28};
    21614     14  11               {}              {75,74}  num2cell(2001:2064)   {};
    21615     15  11          num2cell(70:73)         {72}  num2cell(2065:2128)   {};
  % 64TR 'worst case' column I
    21623      3  11               {}                   {}                 {}   {24};
    21624      4  11               {}              {81,82}  num2cell(2129:2136)   {};
    21625      5  11             {75}              {78,83}  num2cell(2137:2144)   {};
    21633     13  11               {}                   {}                 {}   {26};
    21634     14  11               {}           {80,79,82}  num2cell(2129:2136)   {};
    21635     15  11             {75}              {77,83}  num2cell(2137:2144)   {};
    % additional SRS for different RNTIs
    21636      3  11               {}                   {}                 {}   {31};  % slot 23
    21637     13  11               {}                   {}                 {}   {32};  % slot 33
  % 64TR 'worst case' Ph4 column B
    21640      3  11               {}                   {}                 {}   {36};  % for pattern 79a, 4 SRS symbols in S slot, 2 ports per UE
    21641      3  11               {}                   {}                 {}   {41};  % for pattern 79b, 4 SRS symbols in S slot, 4 ports per UE
    21642      3  11               {}                   {}                 {}   {24};  % for BFW with multiple SRS TVs
    21643      3  11               {}                   {}                 {}   {24};
    21644      4  11               {}              {81,82}  num2cell(2145:2152)   {};
    21645      5  11             {75}              {78,83}  num2cell(2153:2160)   {};
    21653     13  11               {}                   {}                 {}   {26};
    21654     14  11               {}           {80,79,82}  num2cell(2145:2152)   {};
    21655     15  11               {}              {77,83}  num2cell(2161:2168)   {};
    % additional SRS for different RNTIs
    21656      3  11               {}                   {}                 {}   {31};  % slot 23
    21657     13  11               {}                   {}                 {}   {32};  % slot 33
    % Heterogeneous pattern
    21663      3  11               {}                   {}                 {}   {35};
  % 64TR 25-3 column B, 4 SRS symbols in S slot
    21673      3  11               {}                   {}                 {}   {36};
    21674      4  11               {}              {81,82}  num2cell(2145:2152)   {};
    21675      5  11             {75}              {78,83}  num2cell(2153:2160)   {};
    21683     13  11               {}                   {}                 {}   {36};  % srs RNTI modified below in this TC
    21684     14  11               {}           {80,79,82}  num2cell(2145:2152)   {};
    21685     15  11               {}              {77,83}  num2cell(2161:2168)   {};
  % 64TR 25-3 column B, 2 SRS symbols in S slot, 1 SRS symbol in each U slot
    21693      3  11               {}                   {}                 {}   {24};
    21694      4  11               {}              {81,82}  num2cell(2169:2176) {37};
    21695      5  11             {75}              {78,83}  num2cell(2177:2184) {37};  % srs RNTI modified below in this TC
    21703     13  11               {}                   {}                 {}   {26};
    21704     14  11               {}           {80,79,82}  num2cell(2169:2176) {37};  % srs RNTI modified below in this TC
    21705     15  11               {}              {77,83}  num2cell(2185:2192) {37};  % srs RNTI modified below in this TC
  % 64TR 25-3 column G, 64 UEs per TTI, 4 SRS symbols in S slot
    21713      3  11               {}                   {}                 {}   {36};
    21714      4  11               {}              {88,89}  num2cell(2193:2200)   {};
    21715      5  11             {75}              {85,82}  num2cell(2201:2208)   {};
    21723     13  11               {}                   {}                 {}   {36};  % srs RNTI modified below in this TC
    21724     14  11               {}           {87,86,90}  num2cell(2209:2216)   {};
    21725     15  11               {}              {84,82}  num2cell(2217:2224)   {};
  % 64TR 25-3 column G, 64 UEs per TTI, 2 SRS symbols in S slot, 1 SRS symbol in each U slot
    21733      3  11               {}                   {}                 {}   {24};
    21734      4  11               {}              {88,89}  num2cell(2225:2232) {37};
    21735      5  11             {75}              {85,82}  num2cell(2233:2240) {37};  % srs RNTI modified below in this TC
    21743     13  11               {}                   {}                 {}   {26};
    21744     14  11               {}           {87,86,90}  num2cell(2241:2248) {37};  % srs RNTI modified below in this TC
    21745     15  11               {}              {84,82}  num2cell(2249:2256) {37};  % srs RNTI modified below in this TC
    % 64TR 25-3 column E, 32 DL layer with more PUCCH
    % 2 SRS symbols in S slot 21753 and 21763
    % 4 SRS symbols in S slot reuse 21673 and 21683
    21753      3  11               {}                   {}                 {}   {24};
    21754      4  11               {}              {91,92}  num2cell(2257:2264)   {};
    21755      5  11             {75}              {93,94}  num2cell(2265:2272)   {};
    21763     13  11               {}                   {}                 {}   {26};
    21764     14  11               {}           {95,96,97}  num2cell(2273:2280)   {};
    21765     15  11               {}              {98,94}  num2cell(2281:2288)   {};
  % 64TR 'worst case' Ph4 column B, srsPrgSize = 4 and bfwPrgSize = 16
    21772      3  11               {}                   {}                 {}   {24};  % for BFW with srsPrgSize = 2 and bfwPrgSize = 16
    21773      3  11               {}                   {}                 {}   {38};
    21774      4  11               {}              {81,82}  num2cell(2289:2296)   {};
    21775      5  11             {75}              {78,83}  num2cell(2297:2304)   {};
    21782     13  11               {}                   {}                 {}   {24};  % srs RNTI modified below in this TC, for BFW with srsPrgSize = 2 and bfwPrgSize = 16
    21783     13  11               {}                   {}                 {}   {38};  % srs RNTI modified below in this TC
    21784     14  11               {}           {80,79,82}  num2cell(2289:2296)   {};
    21785     15  11               {}              {77,83}  num2cell(2305:2312)   {};
  % additional SRS for different RNTIs
    21786      3  11               {}                   {}                 {}   {38};  % slot 23, srs RNTI modified below in this TC
    21787     13  11               {}                   {}                 {}   {38};  % slot 33, srs RNTI modified below in this TC
    21788      3  11               {}                   {}                 {}   {24};  % slot 23, srs RNTI modified below in this TC, for BFW with srsPrgSize = 2 and bfwPrgSize = 16
    21789     13  11               {}                   {}                 {}   {24};  % slot 33, srs RNTI modified below in this TC, for BFW with srsPrgSize = 2 and bfwPrgSize = 16
  % 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 100 MHz heavy
    21793      3  11               {}                   {}                       {}   {24};
    21794      4  11               {}              {81,82}        num2cell(847:866)     {};
    21795      5  11             {75}              {78,83}        num2cell(867:886)     {};
    21803     13  11               {}                   {}                       {}   {26};
    21804     14  11               {}           {80,79,82}        num2cell(847:866)     {};
    21805     15  11               {}              {77,83}  num2cell([867:885, 887])    {};
  % additional SRS for different RNTIs
    21806      3  11               {}                   {}                 {}   {31};  % slot 23
    21807     13  11               {}                   {}                 {}   {32};  % slot 33
% 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 100 MHz light
    21813      3  11               {}                   {}                       {}   {24};
    21814      4  11               {}              {81,82}        num2cell(888:899)     {};
    21815      5  11             {75}              {78,83}        num2cell(900:911)     {};
    21823     13  11               {}                   {}                       {}   {26};
    21824     14  11               {}           {80,79,82}        num2cell(888:899)     {};
    21825     15  11               {}              {77,83}  num2cell([900:910, 912])    {};
  % additional SRS for different RNTIs
    21826      3  11               {}                   {}                 {}   {31};  % slot 23
    21827     13  11               {}                   {}                 {}   {32};  % slot 33
% 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 90 MHz heavy
    21833      3  14               {}                   {}                       {}   {39};
    21834      4  14               {}              {81,82}        num2cell(913:932)     {};
    21835      5  14             {80}              {78,83}        num2cell(933:952)     {};
    21843     13  14               {}                   {}                       {}   {39};
    21844     14  14               {}           {80,79,82}        num2cell(913:932)     {};
    21845     15  14               {}              {77,83}  num2cell([933:951, 953])    {};
  % additional SRS for different RNTIs
    21846      3  14               {}                   {}                 {}   {39};  % slot 23
    21847     13  14               {}                   {}                 {}   {39};  % slot 33
% 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 90 MHz light
    21853      3  14               {}                   {}                       {}   {39};
    21854      4  14               {}              {81,82}        num2cell(954:965)     {};
    21855      5  14             {80}              {78,83}        num2cell(966:977)     {};
    21863     13  14               {}                   {}                       {}   {39};
    21864     14  14               {}           {80,79,82}        num2cell(954:965)     {};
    21865     15  14               {}              {77,83}  num2cell([966:976, 978])    {};
  % additional SRS for different RNTIs
    21866      3  14               {}                   {}                 {}   {39};  % slot 23
    21867     13  14               {}                   {}                 {}   {39};  % slot 33
% 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 60 MHz heavy
    21873      3  15               {}                   {}                       {}   {40};
    21874      4  15               {}              {81,82}        num2cell(979:998)     {};
    21875      5  15             {81}              {78,83}        num2cell(999:1018)    {};
    21883     13  15               {}                   {}                       {}   {40};
    21884     14  15               {}           {80,79,82}        num2cell(979:998)     {};
    21885     15  15               {}              {77,83}  num2cell([999:1017, 1019])   {};
  % additional SRS for different RNTIs
    21886      3  15               {}                   {}                 {}   {40};  % slot 23
    21887     13  15               {}                   {}                 {}   {40};  % slot 33
% 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 60 MHz light
    21893      3  15               {}                   {}                       {}   {40};
    21894      4  15               {}              {81,82}        num2cell(1020:1031)   {};
    21895      5  15             {81}              {78,83}        num2cell(1032:1043)   {};
    21903     13  15               {}                   {}                       {}   {40};
    21904     14  15               {}           {80,79,82}        num2cell(1020:1031)   {};
    21905     15  15               {}              {77,83}  num2cell([1032:1042, 1044])   {};
  % additional SRS for different RNTIs
    21906      3  15               {}                   {}                 {}   {40};  % slot 23
    21907     13  15               {}                   {}                 {}   {40};  % slot 33
  % 64TR 25-3 column D, 24 DL layer with more PUCCH
    % 2 SRS symbols in S slot 21753 and 21763
    21914      4  11               {}              {99, 100}  num2cell(2313:2320)   {};
    21915      5  11             {75}              {101,102}  num2cell(2321:2328)   {};
    21924     14  11               {}          {103,104,105}  num2cell(2329:2336)   {};
    21925     15  11               {}              {106,102}  num2cell(2337:2344)   {};

    %negative test cases
    605    4  2               {}                 {57}   num2cell(161:166)         {}; %overlapping PUSCH+PUCCH
    606    5  2               {}                 {}     num2cell([161:165,172])   {}; %overlapping resources for PUSCH between the UEs
    607    14  2              {}                 {}     num2cell(161:166)         {}; %same RNTI used for 3 UEs
    608    15  2              {}                 {7}    num2cell(161:166)        {}; %multiple PUCCH is allocated for the same UE
    609    24  2              {}                 {}     num2cell(161:166)        {}; %UCI PUSCH IQ data and FAPI PDU mismatch for CSI length (Num FAPI bits are greater)
    610    25  2              {}                 {}     num2cell(544:549)        {3}; %overlapping PUSCH on SRS symbol
    611    34  2              {}                 {}     num2cell(161:166)        {}; %UCI PUSCH IQ data and FAPI PDU mismatch for CSI length (Num FAPI bits are less)

    % F08 8 cell, No UCI, (prach + pucch + pusch, different RNTIs)
      603      0         2         {2}     {5}   num2cell(14:29)     {};
      612      0         2         {2}     {5}   num2cell(14:29)     {};
      613      0         2         {2}     {5}   num2cell(14:29)     {};
      614      0         2         {2}     {5}   num2cell(14:29)     {};
      615      0         2         {2}     {5}   num2cell(14:29)     {};
      616      0         2         {2}     {5}   num2cell(14:29)     {};
      617      0         2         {2}     {5}   num2cell(14:29)     {};
      618      0         2         {2}     {5}   num2cell(14:29)     {};
    % F08 8 cell UCI on pusch (prach + pucch + pusch) 50% traffic on PUSCH
      636      0         2         {2}     {5}   num2cell(62:77)     {};
      637      0         2         {2}     {5}   num2cell(62:77)     {};
%     638      0         2         {2}     {5}   num2cell(62:77)     {};
%     639      0         2         {2}     {5}   num2cell(62:77)     {};
%     640      0         2         {2}     {5}   num2cell(62:77)     {};
%     641      0         2         {2}     {5}   num2cell(62:77)     {};
%     642      0         2         {2}     {5}   num2cell(62:77)     {};
%     643      0         2         {2}     {5}   num2cell(62:77)     {};
      644      0         2         {2}     {5}   num2cell(62:77)     {};
      645      0         2         {2}     {5}   num2cell(62:77)     {};
%     646      0         2         {2}     {5}   num2cell(62:77)     {};
%     647      0         2         {2}     {5}   num2cell(62:77)     {};
%     648      0         2         {2}     {5}   num2cell(62:77)     {};
%     649      0         2         {2}     {5}   num2cell(62:77)     {};
%     650      0         2         {2}     {5}   num2cell(62:77)     {};
%     651      0         2         {2}     {5}   num2cell(62:77)     {};
      652      0         2         {2}     {5}   num2cell(62:77)     {};
      653      0         2         {2}     {5}   num2cell(62:77)     {};
      654      0         2         {2}     {5}   num2cell(62:77)     {};
      655      0         2         {2}     {5}   num2cell(62:77)     {};
      656      0         2         {2}     {5}   num2cell(62:77)     {};
      657      0         2         {2}     {5}   num2cell(62:77)     {};
      658      0         2         {2}     {5}   num2cell(62:77)     {};
%     659      0         2         {2}     {5}   num2cell(62:77)     {};
      660      0         2         {2}     {5}   num2cell(62:77)     {};
      661      0         2         {2}     {5}   num2cell(62:77)     {};
%     662      0         2         {2}     {5}   num2cell(62:77)     {};
%     663      0         2         {2}     {5}   num2cell(62:77)     {};
%     664      0         2         {2}     {5}   num2cell(62:77)     {};
%     665      0         2         {2}     {5}   num2cell(62:77)     {};
%     666      0         2         {2}     {5}   num2cell(62:77)     {};
%     667      0         2         {2}     {5}   num2cell(62:77)     {};
};

%% additional harq process id config
% column G 
harqProcessIdMap(21534) = 1;
harqProcessIdMap(21535) = 1;
% column H
harqProcessIdMap(21574) = 1;
harqProcessIdMap(21575) = 1;
% column G, 40 MHz
harqProcessIdMap(21594) = 1;
harqProcessIdMap(21595) = 1;
% column G, 64 UE per TTI
harqProcessIdMap(21605) = 1;
harqProcessIdMap(21614) = 2;
harqProcessIdMap(21615) = 3;
% column I
harqProcessIdMap(21634) = 1;
harqProcessIdMap(21645) = 1;
% Ph4 column B
harqProcessIdMap(21654) = 1;
harqProcessIdMap(21655) = 1;
% Rel-25-3 column B/C/G/E
harqProcessIdMap(21684) = 1;
harqProcessIdMap(21685) = 1;
harqProcessIdMap(21704) = 1;
harqProcessIdMap(21705) = 1;
harqProcessIdMap(21724) = 1;
harqProcessIdMap(21725) = 1;
harqProcessIdMap(21744) = 1;
harqProcessIdMap(21745) = 1;
harqProcessIdMap(21764) = 1;
harqProcessIdMap(21765) = 1;
harqProcessIdMap(21924) = 1;
harqProcessIdMap(21925) = 1;
% Ph4 column B, srsPrgSize = 4 and bfwPrgSize = 16
harqProcessIdMap(21784) = 1;
harqProcessIdMap(21785) = 1;
% 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 100 MHz heavy
harqProcessIdMap(21804) = 1;
harqProcessIdMap(21805) = 1;
% 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 100 MHz light
harqProcessIdMap(21824) = 1;
harqProcessIdMap(21825) = 1;
% 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 90 MHz heavy
harqProcessIdMap(21844) = 1;
harqProcessIdMap(21845) = 1;
% 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 90 MHz light
harqProcessIdMap(21864) = 1;
harqProcessIdMap(21865) = 1;
% 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 60 MHz heavy
harqProcessIdMap(21884) = 1;
harqProcessIdMap(21885) = 1;
% 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 60 MHz light
harqProcessIdMap(21904) = 1;
harqProcessIdMap(21905) = 1;


%% additional SRS start RNTI config
srsRntiStartMap(int32(508)) = 2;
srsRntiStartMap(int32(509)) = 6;
srsRntiStartMap(int32(510)) = 10;
srsRntiStartMap(int32(511)) = 14;
srsRntiStartMap(int32(512)) = 18;
srsRntiStartMap(int32(513)) = 22;
srsRntiStartMap(int32(514)) = 26;
srsRntiStartMap(int32(515)) = 30;
srsRntiStartMap(int32(516)) = 34;
srsRntiStartMap(int32(517)) = 38;
srsRntiStartMap(int32(518)) = 42;
srsRntiStartMap(int32(519)) = 46;
srsRntiStartMap(int32(536)) = 2;
srsRntiStartMap(int32(537)) = 6;
srsRntiStartMap(int32(538)) = 10;
srsRntiStartMap(int32(539)) = 14;
srsRntiStartMap(int32(540)) = 18;
srsRntiStartMap(int32(541)) = 22;
srsRntiStartMap(int32(542)) = 26;
srsRntiStartMap(int32(543)) = 30;
srsRntiStartMap(int32(544)) = 34;
srsRntiStartMap(int32(545)) = 38;
srsRntiStartMap(int32(546)) = 42;
srsRntiStartMap(int32(547)) = 46;
srsRntiStartMap(int32(619)) = 2;
srsRntiStartMap(int32(620)) = 6;
srsRntiStartMap(int32(621)) = 10;
srsRntiStartMap(int32(622)) = 14;
srsRntiStartMap(int32(623)) = 18;
srsRntiStartMap(int32(624)) = 22;
srsRntiStartMap(int32(625)) = 26;
srsRntiStartMap(int32(626)) = 30;
srsRntiStartMap(int32(627)) = 34;
srsRntiStartMap(int32(628)) = 38;
srsRntiStartMap(int32(629)) = 42;
srsRntiStartMap(int32(630)) = 46;
srsRntiStartMap(int32(713)) = 2;
srsRntiStartMap(int32(733)) = 3;

% for 25-3 patterns
srsRntiStartMap(int32(21683)) = 55;
srsRntiStartMap(int32(21723)) = 55;
srsRntiStartMap(int32(21695)) = 67;
srsRntiStartMap(int32(21704)) = 79;
srsRntiStartMap(int32(21705)) = 91;
srsRntiStartMap(int32(21735)) = 67;
srsRntiStartMap(int32(21744)) = 79;
srsRntiStartMap(int32(21745)) = 91;
% Ph4 column B, srsPrgSize = 4 and bfwPrgSize = 16
srsRntiStartMap(int32(21783)) = 31;
srsRntiStartMap(int32(21786)) = 55;
srsRntiStartMap(int32(21787)) = 79;
% Ph4 column B, srsPrgSize = 2 and bfwPrgSize = 16
srsRntiStartMap(int32(21782)) = 31;
srsRntiStartMap(int32(21788)) = 55;
srsRntiStartMap(int32(21789)) = 79;
% 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 90 MHz heavy
srsRntiStartMap(int32(21843)) = 31;
srsRntiStartMap(int32(21846)) = 55;
srsRntiStartMap(int32(21847)) = 79;
% 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 90 MHz light
srsRntiStartMap(int32(21863)) = 31;
srsRntiStartMap(int32(21866)) = 55;
srsRntiStartMap(int32(21867)) = 79;
% 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 60 MHz heavy
srsRntiStartMap(int32(21883)) = 31;
srsRntiStartMap(int32(21886)) = 55;
srsRntiStartMap(int32(21887)) = 79;
% 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 60 MHz light
srsRntiStartMap(int32(21903)) = 31;
srsRntiStartMap(int32(21906)) = 55;
srsRntiStartMap(int32(21907)) = 79;

    % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % F08 8 cell, relaxed complexity (prach + pucch + uci@pusch) 
    % 668      0         2         {2}     {7}   num2cell(78:83)     {};
    % 669      0         2         {2}     {7}   num2cell(78:83)     {};
    % 670      0         2         {2}     {7}   num2cell(78:83)     {};
    % 671      0         2         {2}     {7}   num2cell(78:83)     {};
    % 672      0         2         {2}     {7}   num2cell(78:83)     {};
    % 673      0         2         {2}     {7}   num2cell(78:83)     {};
    % 674      0         2         {2}     {7}   num2cell(78:83)     {};
    % 675      0         2         {2}     {7}   num2cell(78:83)     {};
    % 676      0         2         {2}     {7}   num2cell(78:83)     {};
    % 677      0         2         {2}     {7}   num2cell(78:83)     {};
    % 678      0         2         {2}     {7}   num2cell(78:83)     {};
    % 679      0         2         {2}     {7}   num2cell(78:83)     {};
    % 680      0         2         {2}     {7}   num2cell(78:83)     {};
    % 681      0         2         {2}     {7}   num2cell(78:83)     {};
    % 682      0         2         {2}     {7}   num2cell(78:83)     {};
    % 683      0         2         {2}     {7}   num2cell(78:83)     {};
    % 684      0         2         {2}     {7}   num2cell(78:83)     {};
    % 685      0         2         {2}     {7}   num2cell(78:83)     {};
    % 686      0         2         {2}     {7}   num2cell(78:83)     {};
    % 687      0         2         {2}     {7}   num2cell(78:83)     {};
    % 688      0         2         {2}     {7}   num2cell(78:83)     {};
    % 689      0         2         {2}     {7}   num2cell(78:83)     {};
    % 690      0         2         {2}     {7}   num2cell(78:83)     {};
    % 691      0         2         {2}     {7}   num2cell(78:83)     {};
    % 692      0         2         {2}     {7}   num2cell(78:83)     {};
    % 693      0         2         {2}     {7}   num2cell(78:83)     {};
    % 694      0         2         {2}     {7}   num2cell(78:83)     {};
    % 695      0         2         {2}     {7}   num2cell(78:83)     {};
    % 696      0         2         {2}     {7}   num2cell(78:83)     {};
    % 697      0         2         {2}     {7}   num2cell(78:83)     {};
    % 698      0         2         {2}     {7}   num2cell(78:83)     {};
    % 699      0         2         {2}     {7}   num2cell(78:83)     {};
    
    
%     % TC#   slotIdx   cell     prach     pucch      pusch      srs
%     % F08 16 cells, relaxed complexity, 50% load (prach + pucch + uci@pusch) 
%     700      0         2         {2}     {7}   num2cell(84:89)     {};
%     701      0         2         {2}     {7}   num2cell(84:89)     {};
%     702      0         2         {2}     {7}   num2cell(84:89)     {};
%     703      0         2         {2}     {7}   num2cell(84:89)     {};
%     704      0         2         {2}     {7}   num2cell(84:89)     {};
%     705      0         2         {2}     {7}   num2cell(84:89)     {};
%     706      0         2         {2}     {7}   num2cell(84:89)     {};
%     707      0         2         {2}     {7}   num2cell(84:89)     {};
%     708      0         2         {2}     {7}   num2cell(84:89)     {};
%     709      0         2         {2}     {7}   num2cell(84:89)     {};
%     710      0         2         {2}     {7}   num2cell(84:89)     {};
%     711      0         2         {2}     {7}   num2cell(84:89)     {};
%     712      0         2         {2}     {7}   num2cell(84:89)     {};
%     713      0         2         {2}     {7}   num2cell(84:89)     {};
%     714      0         2         {2}     {7}   num2cell(84:89)     {};
%     715      0         2         {2}     {7}   num2cell(84:89)     {};
%     716      0         2         {2}     {7}   num2cell(84:89)     {};
%     717      0         2         {2}     {7}   num2cell(84:89)     {};
%     718      0         2         {2}     {7}   num2cell(84:89)     {};
%     719      0         2         {2}     {7}   num2cell(84:89)     {};
%     720      0         2         {2}     {7}   num2cell(84:89)     {};
%     721      0         2         {2}     {7}   num2cell(84:89)     {};
%     722      0         2         {2}     {7}   num2cell(84:89)     {};
%     723      0         2         {2}     {7}   num2cell(84:89)     {};
%     724      0         2         {2}     {7}   num2cell(84:89)     {};
%     725      0         2         {2}     {7}   num2cell(84:89)     {};
%     726      0         2         {2}     {7}   num2cell(84:89)     {};
%     727      0         2         {2}     {7}   num2cell(84:89)     {};
%     728      0         2         {2}     {7}   num2cell(84:89)     {};
%     729      0         2         {2}     {7}   num2cell(84:89)     {};
%     730      0         2         {2}     {7}   num2cell(84:89)     {};
%     731      0         2         {2}     {7}   num2cell(84:89)     {};
%     732      0         2         {2}     {7}   num2cell(84:89)     {};
%     733      0         2         {2}     {7}   num2cell(84:89)     {};
%     734      0         2         {2}     {7}   num2cell(84:89)     {};
%     735      0         2         {2}     {7}   num2cell(84:89)     {};
%     736      0         2         {2}     {7}   num2cell(84:89)     {};
%     737      0         2         {2}     {7}   num2cell(84:89)     {};
%     738      0         2         {2}     {7}   num2cell(84:89)     {};
%     739      0         2         {2}     {7}   num2cell(84:89)     {};
%     740      0         2         {2}     {7}   num2cell(84:89)     {};
%     741      0         2         {2}     {7}   num2cell(84:89)     {};
%     742      0         2         {2}     {7}   num2cell(84:89)     {};
%     743      0         2         {2}     {7}   num2cell(84:89)     {};
%     744      0         2         {2}     {7}   num2cell(84:89)     {};
%     745      0         2         {2}     {7}   num2cell(84:89)     {};
%     746      0         2         {2}     {7}   num2cell(84:89)     {};
%     747      0         2         {2}     {7}   num2cell(84:89)     {};
%     748      0         2         {2}     {7}   num2cell(84:89)     {};
%     749      0         2         {2}     {7}   num2cell(84:89)     {};
%     750      0         2         {2}     {7}   num2cell(84:89)     {};
%     751      0         2         {2}     {7}   num2cell(84:89)     {};
%     752      0         2         {2}     {7}   num2cell(84:89)     {};
%     753      0         2         {2}     {7}   num2cell(84:89)     {};
%     754      0         2         {2}     {7}   num2cell(84:89)     {};
%     755      0         2         {2}     {7}   num2cell(84:89)     {};
%     756      0         2         {2}     {7}   num2cell(84:89)     {};
%     757      0         2         {2}     {7}   num2cell(84:89)     {};
%     758      0         2         {2}     {7}   num2cell(84:89)     {};
%     759      0         2         {2}     {7}   num2cell(84:89)     {};
%     760      0         2         {2}     {7}   num2cell(84:89)     {};
%     761      0         2         {2}     {7}   num2cell(84:89)     {};
%     762      0         2         {2}     {7}   num2cell(84:89)     {};
%     763      0         2         {2}     {7}   num2cell(84:89)     {};
%     };

%     % Reserve 764-827 for expanding 8C reduced complexity, half load

%     %%%%%   continue config CFG using "for" %%%%%%                 
    
%                              % TC#   slotIdx   cell     prach     pucch      pusch      srs
%     % % F08 16C, 2 PRBs / UEG, (prach + pucch + uci@pusch)
%     % row_count = length(CFG);
%     % for i = 828:891
%     %     row_count = row_count + 1;
%     %     CFG(row_count, 1:7) = { i    0         2         {2}      {7}   num2cell(90:95)   {}};
%     % end
%     % pattern 34: F08 16C, 2 PRBs / UEG, shared channel only
%     row_count = length(CFG);
%     for i = 892:955
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2          {}       {}    num2cell(90:95)   {}};
%     end

%     % 16C, 2 PRB / UEG, pusch(no uci) only
%     row_count = length(CFG);
%     for i = 1000:1063
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2          {}       {}        {96}          {}};
%     end
    
%                              % TC#   slotIdx   cell     prach     pucch      pusch      srs
%     % F08 16C, 4 prach occasions, relaxed complexity, 50% load, (prach + pucch + uci@pusch)
% %     row_count = length(CFG);
% %     for i = 1064:1127
% %         row_count = row_count + 1;
% %         CFG(row_count, 1:7) = { i    0         2       {3,4,5,6}   {7}   num2cell(84:89)   {}};
% %     end
    
%     % F08 8C, 4 prach occasions, peak cell, (prach + pucch + uci@pusch)
%     row_count = length(CFG);
%     for i = 1128:1159
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2       {3,4,5,6}   {5}   num2cell(97:112)  {}};
%     end
    
%     % pattern 22: F08 16C, 4 prach occasions, relaxed complexity, 50% load, 17 PRB/UEG (prach + pucch + uci@pusch)
% %     row_count = length(CFG);
% %     for i = 1160:1223
% %         row_count = row_count + 1;
% %         CFG(row_count, 1:7) = { i    0         2       {3,4,5,6}   {7}   num2cell(113:118)   {}};
% %     end
    
%     % pattern 23: F08 16C, 4 prach occasions, relaxed complexity, 50% load, 12 PRB/UEG (prach + pucch + uci@pusch)
% %     row_count = length(CFG);
% %     for i = 1224:1287
% %         row_count = row_count + 1;
% %         CFG(row_count, 1:7) = { i    0         2       {3,4,5,6}   {7}   num2cell(119:124)   {}};
% %     end
    
%     %% pattern 24,25: 16C, avg cell, 4 beams (4 prach occasions at slot 5, relaxed complexity, 50% load, (prach + pucch + uci@pusch))
%                             % TC#   slotIdx   cell     prach     pucch      pusch      srs
%     % slot 4, no prach
%     row_count = length(CFG);
%     for i = 1288:1303
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(84:89)   {}};
%     end
%     % slot 5, 4 prach
%     for i = 1304:1319
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2       {7,8,9,10}  {7}   num2cell(125:130)   {}};
%     end
%     % slot 14,15, no prach
%     for i = 1320:1351
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(84:89)   {}};
%     end
    
%     %% pattern 28: 8C, peak cell, 4 beams (4 prach occasions at slot 5, relaxed complexity, 100% load, (prach + pucch + uci@pusch))
%                             % TC#   slotIdx   cell     prach     pucch      pusch      srs
%     % slot 4, no prach
%     row_count = length(CFG);
%     for i = 1352:1359
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(131:136)   {}};
%     end
%     % slot 5, 4 prach
%     for i = 1360:1367
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2       {3,4,5,6}  {7}   num2cell(137:142)   {}};
%     end
%     % slot 14,15, no prach
%     for i = 1368:1383
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(131:136)   {}};
%     end
    
%     %% 8C, ~ peak cell (40 PRB/UEG), 4 beams (4 prach occasions at slot 5, relaxed complexity, (prach + pucch + uci@pusch))
%                             % TC#   slotIdx   cell     prach     pucch      pusch      srs
%     % slot 4, no prach
%     row_count = length(CFG);
%     for i = 1384:1391
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(143:148)   {}};
%     end
%     % slot 5, 4 prach
%     for i = 1392:1399
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2       {3,4,5,6}  {7}   num2cell(137:142)   {}};
%     end
%     % slot 14,15, no prach
%     for i = 1400:1415
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(143:148)   {}};
%     end
    
%     %% pattern 31: 16C, avg cell, 1UEG, 4 beams (4 prach occasions at slot 5, relaxed complexity, 50% load, (prach + pucch + uci@pusch))
%                             % TC#   slotIdx   cell     prach     pucch      pusch      srs
%     % slot 4, no prach
%     row_count = length(CFG);
%     for i = 1416:1431
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2        {}         {7}       {149}   {}};
%     end
%     % slot 5, 4 prach
%     for i = 1432:1447
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2       {7,8,9,10}  {7}       {150}   {}};
%     end
%     % slot 14,15, no prach
%     for i = 1448:1479
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2        {}         {7}       {149}   {}};
%     end
    
%         %% pattern 32: 8C, peak cell, 1UEG 4 beams (4 prach occasions at slot 5, relaxed complexity, 100% load, (prach + pucch + uci@pusch))
%                             % TC#   slotIdx   cell     prach     pucch      pusch      srs
%     % slot 4, no prach
%     row_count = length(CFG);
%     for i = 1480:1487
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2        {}         {7}       {151}   {}};
%     end
%     % slot 5, 4 prach
%     for i = 1488:1495
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2       {3,4,5,6}  {7}        {152}   {}};
%     end
%     % slot 14,15, no prach
%     for i = 1496:1511
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2        {}         {7}       {151}     {}};
%     end
    
%     %% pattern 24a,25a, 25b: 17 PRBs/UEG, 16C, avg cell, 4 beams (4 prach occasions at slot 5, relaxed complexity, 50% load, (prach + pucch + uci@pusch))
%                             % TC#   slotIdx   cell     prach     pucch      pusch      srs
%     % slot 4, no prach
%     row_count = length(CFG);
%     for i = 1512:1527
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(153:158)   {}};
%     end
%     % slot 5, 4 prach
%     for i = 1528:1543
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2       {7,8,9,10}  {7}   num2cell(125:130)   {}};
%     end
%     % slot 14,15, no prach
%     for i = 1544:1575
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(153:158)   {}};
%     end
    
    
    
    
%     %% pattern 35: 12C, peak cell, 4 beams (4 prach occasions at slot 5, relaxed complexity, 100% load, (prach + pucch + uci@pusch))
%                             % TC#   slotIdx   cell     prach     pucch      pusch      srs
%     % slot 4, no prach
%     row_count = length(CFG);
%     for i = 1584:1595
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(131:136)   {}};
%     end
%     % slot 5, 4 prach
%     for i = 1596:1607
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2       {3,4,5,6}  {7}   num2cell(137:142)   {}};
%     end
%     % slot 14,15, no prach
%     for i = 1608:1631
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(131:136)   {}};
%     end
    
    %% pattern 39: 16C, avg cell, 4 beams (same as pattern 33 but AWGN channel)
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 1632:1647
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(84:89)   {}};
    end
    % slot 5, 4 prach
    for i = 1648:1663
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {7,8,9,10}  {7}   num2cell(125:130)   {}};
    end
    % slot 14,15, no prach
    for i = 1664:1695
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(84:89)   {}};
    end
    
    %% pattern 40: 12C, peak cell, 4 beams (same as pattern 35 but AWGN channel)
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 1696:1707
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(131:136)   {}};
    end
    % slot 5, 4 prach
    for i = 1708:1719
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {3,4,5,6}  {7}   num2cell(137:142)   {}};
    end
    % slot 14,15, no prach
    for i = 1720:1743
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(131:136)   {}};
    end
    
    
    %% pattern 41: 12C, peak cell, 4 beams, OTA, P2P ch (4 prach occasions at slot 5 (prach + pucch + uci@pusch))
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 1744:1755
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(161:166)   {}};
    end
    % slot 5, 4 prach
    for i = 1756:1767
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {3,4,5,6}   {7}   num2cell(167:172)   {}};
    end
    % slot 14,15, no prach
    for i = 1768:1791
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(161:166)   {}};
    end
    
    %% pattern 42: 12C, peak cell, 4 beams, OTA, AWGN ch (4 prach occasions at slot 5 (prach + pucch + uci@pusch))
    row_count = length(CFG);
    for i = 1792:1803
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(161:166)   {}};
    end
    % slot 5, 4 prach
    for i = 1804:1815
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {3,4,5,6}   {7}   num2cell(167:172)   {}};
    end
    % slot 14,15, no prach
    for i = 1816:1839
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(161:166)   {}};
    end
    
    %% pattern 43, 44: 16C, avg cell, 4 beams, OTA, P2P (4 prach occasions at slot 5 (prach + pucch + uci@pusch))
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 1840:1855
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(173:178)   {}};
    end
    % slot 5, 4 prach
    for i = 1856:1871
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {7,8,9,10}  {7}   num2cell(179:184)   {}};
    end
    % slot 14,15, no prach
    for i = 1872:1903
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(173:178)   {}};
    end
    
    %% pattern 44, UL is the same as pattern 43
    
    %% pattern 45, 16C, avg cell, 4 beams, OTA, AWGN (4 prach occasions at slot 5 (prach + pucch + uci@pusch))
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 1904:1919
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(173:178)   {}};
    end
    % slot 5, 4 prach
    for i = 1920:1935
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {7,8,9,10}  {7}   num2cell(179:184)   {}};
    end
    % slot 14,15, no prach
    for i = 1936:1967
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(173:178)   {}};
    end
    
    
    
    %% pattern 46: 7 beams, 100 MHz (273 PRBs), 12C, peak cell, OTA, 4UL streams
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 1968:1979
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(161:166)   {}};
    end 
    % slot 5, 4 prach
    for i = 1980:1991
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {4,5,6}  {7}   num2cell(167:172)   {}}; % {3,4,5,6}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end
    % slot 14, no prach
    for i = 1992:2003
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(161:166)   {}};
    end
    % slot 15, 3 prach
    for i = 2004:2015
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {4,5,6}     {7}   num2cell(185:190)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
    end
    
    %% pattern 47: 7 beams, 100 MHz (273 PRBs), 16C, ave cell, OTA, 4UL streams
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 2016:2031
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(173:178)   {}};
    end
    % slot 5, 4 prach
    for i = 2032:2047
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {8,9,10}  {7}   num2cell(179:184)   {}}; % {3,4,5,6}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end
    % slot 14, no prach
    for i = 2048:2063
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(173:178)   {}};
    end
    % slot 15, 3 prach
    for i = 2064:2079
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {8,9,10}     {7}   num2cell(191:196)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
    end
    
    
    %% pattern 48: 7 beams, 40 MHz (106 PRBs), 16C, peak cell, OTA, 4UL streams
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 2080:2095
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         4        {}         {7}   num2cell(197:202)   {}};
    end
    % slot 5, 4 prach
    for i = 2096:2111
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         4       {14,15,16,13}  {7}   num2cell(203:208)   {}}; % {13,14,15,16}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end
    % slot 14, no prach
    for i = 2112:2127
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         4        {}         {7}   num2cell(197:202)   {}};
    end
    % slot 15, 3 prach
    for i = 2128:2143
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         4       {14,15,16}     {7}   num2cell(390:395)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
    end
    % end pattern 48
    
    %% pattern 49: 7 beams, 40 MHz (106 PRBs), 20C, ave cell, OTA, 4UL streams
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 2144:2163
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         4        {}         {7}   num2cell(209:214)   {}};
    end
    % slot 5, 4 prach
    for i = 2164:2183
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         4       {18,19,20,17}  {7}   {}   {}}; % {17,18,19,20}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end
    % slot 14, no prach
    for i = 2184:2203
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         4        {}         {7}   num2cell(209:214)   {}};
    end
    % slot 15, 3 prach
    for i = 2204:2223
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         4       {18,19,20}     {7}   {}   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
    end
    % end pattern 49
    
    %% pattern 50: 7 beams, 80 MHz (217 PRBs), 16C, peak cell, OTA, 4UL streams
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 2224:2239
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         5        {}         {7}   num2cell(215:220)   {}};
    end
    % slot 5, 4 prach
    for i = 2240:2255
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         5       {22,23,24,21}  {7}   num2cell(221:226)   {}}; % {21,22,23,24}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end
    % slot 14, no prach
    for i = 2256:2271
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         5        {}         {7}   num2cell(215:220)   {}};
    end
    % slot 15, 3 prach
    for i = 2272:2287
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         5       {22,23,24}     {7}   num2cell(396:401)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
    end
    % end pattern 50
    
    


    %% pattern 41a: 12C, peak cell, 4 beams, OTA, P2P ch, 24 PUCCH PRB (4 prach occasions at slot 5 (prach + pucch + uci@pusch))
                                % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 2304:2315
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {8}   num2cell(239:244)   {}};
    end
    % slot 5, 4 prach
    for i = 2316:2327
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {29,30,31,32}  {8}   num2cell(245:250)   {}};
    end
    % slot 14,15, no prach
    for i = 2328:2351
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {8}   num2cell(239:244)   {}};
    end

    
    %% pattern 44a: 16C, avg cell, 4 beams, OTA, P2P, 12 PUCCH PRB (4 prach occasions at slot 5 (prach + pucch + uci@pusch))
                                % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 2352:2367
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {9}   num2cell(251:256)   {}};
    end
    % slot 5, 4 prach
    for i = 2368:2383
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {33,34,35,36}  {9}   num2cell(257:262)   {}};
    end
    % slot 14,15, no prach
    for i = 2384:2415
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {9}   num2cell(251:256)   {}};
    end

        %% pattern 51: 7 beams, 80 MHz (217 PRBs), 20C, ave cell, OTA, 4UL streams
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 2416:2435
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         5        {}         {7}   num2cell(227:232)   {}};
    end
    % slot 5, 4 prach
    for i = 2436:2455
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         5       {26,27,28,25}  {7}   num2cell(233:238)   {}}; % {25,26,27,28}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end
    % slot 14, no prach
    for i = 2456:2475
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         5        {}         {7}   num2cell(227:232)   {}};
    end
    % slot 15, 3 prach
    for i = 2476:2495
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         5       {26,27,28}     {7}   num2cell(402:407)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
    end
    % end pattern 51

    %% pattern 53: 7 beams, 30 MHz (78 PRBs), 16C, peak cell, OTA, 4UL streams
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 2496:2511
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         6        {}         {7}   num2cell(267:272)   {}};
    end
    % slot 5, 4 prach
    for i = 2512:2527
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         6       {38,39,40,37}  {7}   num2cell(273:278)   {}}; % {37,38,39,40}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end
    % slot 14, no prach
    for i = 2528:2543
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         6        {}         {7}   num2cell(267:272)   {}};
    end
    % slot 15, 3 prach
    for i = 2544:2559
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         6       {38,39,40}     {7}   num2cell(408:413)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
    end
    % end pattern 53
    
    
    %% pattern 54: 7 beams, 30 MHz (78 PRBs), 20C, ave cell, OTA, 4UL streams
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 2560:2579
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         6        {}         {7}   num2cell(279:284)   {}};
    end
    % slot 5, 4 prach
    for i = 2580:2599
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         6       {17,18,19,16}  {7}   {}   {}}; % {16,17,18,19}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end
    % slot 14, no prach
    for i = 2600:2619
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         6        {}         {7}   num2cell(279:284)   {}};
    end
    % slot 15, 3 prach
    for i = 2620:2639
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         6       {17,18,19}     {7}   {}   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
    end
    % end pattern 54
    
    
    
    %% pattern 41b: 12C, peak cell, 4 beams, OTA, P2P ch, MCS0 (4 prach occasions at slot 5 (prach + pucch + uci@pusch))
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 2640:2651
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(342:347)   {}};
    end
    % slot 5, 4 prach
    for i = 2652:2663
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {3,4,5,6}   {7}   num2cell(348:353)   {}};
    end
    % slot 14,15, no prach
    for i = 2664:2687
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(342:347)   {}};
    end
    
    %% pattern 41c: 12C, peak cell, 4 beams, OTA, P2P ch, MCS1 (4 prach occasions at slot 5 (prach + pucch + uci@pusch))
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 2688:2699
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(354:359)   {}};
    end
    % slot 5, 4 prach
    for i = 2700:2711
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {3,4,5,6}   {7}   num2cell(360:365)   {}};
    end
    % slot 14,15, no prach
    for i = 2712:2735
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(354:359)   {}};
    end
    
    %% pattern 44b: 16C, avg cell, 4 beams, OTA, P2P, MCS 0 (4 prach occasions at slot 5 (prach + pucch + uci@pusch))
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 2736:2751
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(366:371)   {}};
    end
    % slot 5, 4 prach
    for i = 2752:2767
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {7,8,9,10}  {7}   num2cell(372:377)   {}};
    end
    % slot 14,15, no prach
    for i = 2768:2799
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(366:371)   {}};
    end
    
    
    %% pattern 44c: 16C, avg cell, 4 beams, OTA, P2P, MCS 1 (4 prach occasions at slot 5 (prach + pucch + uci@pusch))
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 2800:2815
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(378:383)   {}};
    end
    % slot 5, 4 prach
    for i = 2816:2831
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {7,8,9,10}  {7}   num2cell(384:389)   {}};
    end
    % slot 14,15, no prach
    for i = 2832:2863
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(378:383)   {}};
    end
    
    
                                % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % pattern 57: 4 frames, 8C, peak cell, different UEs at different cells, 4 beams OTA, 4UL steams 
    % slot 4, no prach
    row_count = length(CFG);
    for i = 2864:2871
        row_count = row_count + 1;
        cidx = mod(i - 2864, 8);
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(420:420+cidx)    {}};
    end
    % slot 5, 4 prach
    for i = 2872:2879
        row_count = row_count + 1;
        cidx = mod(i - 2864, 8);
        CFG(row_count, 1:7) = { i    0         2       {3,4,5,6}  {7}   num2cell(428:428+cidx)    {}};
    end
    % slot 14,15, no prach
    for i = 2880:2895
        row_count = row_count + 1;
        cidx = mod(i - 2864, 8);
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(420:420+cidx)    {}};
    end
    

    %% pattern 55: 7 beams, 50 MHz (133 PRBs), 16C, peak cell, OTA, 4UL streams
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 2896:2911
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         7        {}         {7}   num2cell(285:290)   {}};
    end
    % slot 5, 4 prach
    for i = 2912:2927
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         7       {8,9,10,7}  {7}   num2cell(291:296)   {}}; % {7,8,9,10}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end
    % slot 14, no prach
    for i = 2928:2943
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         7        {}         {7}   num2cell(285:290)   {}};
    end
    % slot 15, 3 prach
    for i = 2944:2959
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         7       {8,9,10}     {7}   num2cell(414:419)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
    end
    % end pattern 55
    
    %% pattern 56: 7 beams, 50 MHz (133 PRBs), 20C, ave cell, OTA, 4UL streams
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 2960:2979
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         7        {}         {7}   num2cell(297:302)   {}};
    end
    % slot 5, 4 prach
    for i = 2980:2999
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         7       {18,19,20,17}  {7}   {}   {}}; % {17,18,19,20}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end
    % slot 14, no prach
    for i = 3000:3019
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         7        {}         {7}   num2cell(297:302)   {}};
    end
    % slot 15, 3 prach
    for i = 3020:3039
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         7       {18,19,20}     {7}   {}   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
    end
    % end pattern 56

    % 4 PRACH versions of pattern 46 & 47
    %% pattern 59: 7 beams, 100 MHz (273 PRBs), 20C, peak cell, OTA, 4UL streams
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 3040:3059
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(161:166)   {}};
    end
    % slot 5, 4 prach
    for i = 3060:3079
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {4,5,6,3}  {7}   num2cell(167:172)   {}}; % {3,4,5,6}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end
    % slot 14, no prach
    for i = 3080:3099
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(161:166)   {}};
    end
    % slot 15, 3 prach
    for i = 3100:3119
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {4,5,6}     {7}   num2cell(185:190)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
    end
    % end pattern 59

    % [3120:3199] are available for future ULMIX TVs

    %% pattern 60: 7 beams, 100 MHz (273 PRBs), 40C, ave cell, OTA, 4UL streams
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot 4, no prach
    for i = 3600:3639
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(173:178)   {}};
        enableOtaMap(i) = 1;
        UciOnPuschMap(i) = "7beams_4_37_5";
        CellIdxInPatternMap(i) = mod(i-3600, 40);
    end
    % slot 5, 4 prach
    for i = 3640:3679
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {8,9,10,7}  {7}   num2cell(179:184)   {}}; % {7,8,9,10}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
        enableOtaMap(i) = 1;
        UciOnPuschMap(i) = "7beams_4_37_5";
        CellIdxInPatternMap(i) = mod(i-3600, 40);
    end
    % slot 14, no prach
    for i = 3680:3719
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(173:178)   {}};
        enableOtaMap(i) = 1;
        UciOnPuschMap(i) = "7beams_4_37_5";
        CellIdxInPatternMap(i) = mod(i-3600, 40);
    end
    % slot 15, 3 prach
    for i = 3720:3759
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {8,9,10}     {7}   num2cell(191:196)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
        enableOtaMap(i) = 1;
        UciOnPuschMap(i) = "7beams_4_37_5";
        CellIdxInPatternMap(i) = mod(i-3600, 40);
    end

    % Negative test vector for the first cell of F08 test cases (FAPI override with invalid params)
    i = 3840;
    row_count = row_count + 1;
    CFG(row_count, 1:7) = { i    0         2       {8,9,10,7}  {7}   num2cell(179:184)   {}}; % {7,8,9,10}
    PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    enableOtaMap(i) = 1;
    UciOnPuschMap(i) = "7beams_4_37_5";
    CellIdxInPatternMap(i) = 2;

    % end pattern 60: 7 beams, 100 MHz (273 PRBs), 40C, ave cell, OTA, 4UL streams

    %% pattern 59a: 7 beams, 100 MHz (273 PRBs), 20C, peak cell, OTA, 4UL streams
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 3200:3219
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(161:166)   {}};
    end
    % slot 5, 4 prach
    for i = 3220:3239
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {4,5,6,3}  {7}   num2cell(167:172)   {}}; % {3,4,5,6}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end
    % slot 14, no prach
    for i = 3240:3259
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(161:166)   {}};
    end
    % slot 15, 3 prach
    for i = 3260:3279
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {4,5,6}     {7}   num2cell(185:190)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
    end
    % end pattern 59a

    %% pattern 60a: 7 beams, 100 MHz (273 PRBs), 20C, ave cell, OTA, 4UL streams
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 3280:3299
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(173:178)   {}};
    end
    % slot 5, 4 prach
    for i = 3300:3319
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {8,9,10,7}  {7}   num2cell(179:184)   {}}; % {7,8,9,10}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end
    % slot 14, no prach
    for i = 3320:3339
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(173:178)   {}};
    end
    % slot 15, 3 prach
    for i = 3340:3359
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {8,9,10}     {7}   num2cell(191:196)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
    end
    % end pattern 60a

    %% pattern 61: 7 beams, 100 MHz (273 PRBs), 20C, light load, OTA, 4UL streams
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 3360:3379
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(436:441)   {}};
    end
    % slot 5, 4 prach
    for i = 3380:3399
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {17,18,19,16}  {7}   {}   {}}; % {16,17,18,19}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end
    % slot 14, no prach
    for i = 3400:3419
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {7}   num2cell(436:441)   {}};
    end
    % slot 15, 3 prach
    for i = 3420:3439
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {17,18,19}     {7}   {}   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
    end
    % end pattern 61
    

    %% pattern 59d: 7 beams, 100 MHz (273 PRBs), 20C, peak cell, OTA, 4UL streams, 24 PUCCH UCIs
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 3520:3539
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {43}   num2cell(460:465)   {}};
    end
    % slot 5, 4 prach
    for i = 3540:3559
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {46,47,48,45}  {43}   num2cell(466:471)   {}}; % {45,46,47,48}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end
    % slot 14, no prach
    for i = 3560:3579
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {43}   num2cell(460:465)   {}};
    end
    % slot 15, 3 prach
    for i = 3580:3599
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {46,47,48}     {43}   num2cell(472:477)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
    end
    % end pattern 59d

    
    %% pattern 60c: 7 beams, 100 MHz (273 PRBs), 40C, ave cell, OTA, 4UL streams, 18 PUCCH UCIs
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot 4, no prach
    for i = 5071:5110
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {42}   num2cell(478:483)   {}};
        enableOtaMap(i) = 1;
        UciOnPuschMap(i) = "7beams_4_37_5";
        CellIdxInPatternMap(i) = mod(i-5071, 40);
    end
    % slot 5, 4 prach
    for i = 5111:5150
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {50,51,52,49}  {42}   num2cell(484:489)   {}}; % {49,50,51,52}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
        enableOtaMap(i) = 1;
        UciOnPuschMap(i) = "7beams_4_37_5";
        CellIdxInPatternMap(i) = mod(i-5071, 40);
    end
    % slot 14, no prach
    for i = 5151:5190
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {42}   num2cell(478:483)   {}};
        enableOtaMap(i) = 1;
        UciOnPuschMap(i) = "7beams_4_37_5";
        CellIdxInPatternMap(i) = mod(i-5071, 40);
    end
    % slot 15, 3 prach
    for i = 5191:5230
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {50,51,52}     {42}   num2cell(490:495)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
        enableOtaMap(i) = 1;
        UciOnPuschMap(i) = "7beams_4_37_5";
        CellIdxInPatternMap(i) = mod(i-5071, 40);
    end
    % end pattern 60c: 7 beams, 100 MHz (273 PRBs), 40C, ave cell, OTA, 4UL streams, 18 PUCCH UCIs

    %% pattern 60d: 7 beams, 100 MHz (273 PRBs), 40C, ave cell, OTA, 4UL streams, 24 PUCCH UCIs
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot 4, no prach
    for i = 5231:5270
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {43}   num2cell(496:501)   {}};
        enableOtaMap(i) = 1;
        UciOnPuschMap(i) = "7beams_0_37_5";
        CellIdxInPatternMap(i) = mod(i-5231, 40);
    end
    % slot 5, 4 prach
    for i = 5271:5310
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {50,51,52,49}  {43}   num2cell(502:507)   {}}; % {49,50,51,52}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
        enableOtaMap(i) = 1;
        UciOnPuschMap(i) = "7beams_0_37_5";
        CellIdxInPatternMap(i) = mod(i-5231, 40);
    end
    % slot 14, no prach
    for i = 5311:5350
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {43}   num2cell(496:501)   {}};
        enableOtaMap(i) = 1;
        UciOnPuschMap(i) = "7beams_0_37_5";
        CellIdxInPatternMap(i) = mod(i-5231, 40);
    end
    % slot 15, 3 prach
    for i = 5351:5390
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {50,51,52}     {43}   num2cell(508:513)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
        enableOtaMap(i) = 1;
        UciOnPuschMap(i) = "7beams_0_37_5";
        CellIdxInPatternMap(i) = mod(i-5231, 40);
    end
    % end pattern 60d: 7 beams, 100 MHz (273 PRBs), 40C, ave cell, OTA, 4UL streams, 24 PUCCH UCIs

    %% pattern 59b: 7 beams, 100 MHz (273 PRBs), 20C, peak cell, OTA, 4UL streams, early-HARQ PUCCH + PUSCH
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 3760:3779
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {44}   num2cell(514:519)   {}};
    end
    % slot 5, 4 prach
    for i = 3780:3799
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {54,55,56,53}  {44}   num2cell(520:525)   {}}; % {53,54,55,56}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end
    % slot 14, no prach
    for i = 3800:3819
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {44}   num2cell(514:519)   {}};
    end
    % slot 15, 3 prach
    for i = 3820:3839
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {54,55,56}     {44}   num2cell(526:531)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
    end
    % end pattern 59b

    % [3841:3919] are available for future ULMIX TVs
    
    %% pattern 60b: 7 beams, 100 MHz (273 PRBs), 40C, ave cell, OTA, 4UL streams, early-HARQ PUCCH + PUSCH
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot 4, no prach
    for i = 4911:4950
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {44}   num2cell(532:537)   {}};
        enableOtaMap(i) = 1;
        UciOnPuschMap(i) = "7beams_4_37_5";
        CellIdxInPatternMap(i) = mod(i-4911, 40);
    end
    % slot 5, 4 prach
    for i = 4951:4990
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {58,59,60,57}  {44}   num2cell(538:543)   {}}; % {57,58,59,60}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
        enableOtaMap(i) = 1;
        UciOnPuschMap(i) = "7beams_4_37_5";
        CellIdxInPatternMap(i) = mod(i-4911, 40);
    end
    % slot 14, no prach
    for i = 4991:5030
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {44}   num2cell(532:537)   {}};
        enableOtaMap(i) = 1;
        UciOnPuschMap(i) = "7beams_4_37_5";
        CellIdxInPatternMap(i) = mod(i-4911, 40);
    end
    % slot 15, 3 prach
    for i = 5031:5070
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {58,59,60}     {44}   num2cell(544:549)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
        enableOtaMap(i) = 1;
        UciOnPuschMap(i) = "7beams_4_37_5";
        CellIdxInPatternMap(i) = mod(i-4911, 40);
    end
    % end pattern 60b: 7 beams, 100 MHz (273 PRBs), 40C, ave cell, OTA, 4UL streams, early-HARQ PUCCH + PUSCH

    %% UL MIX TVs for PRACH + PUCCH + PUSCH + SRS
    % modified from pattern 59 3060:3079, 6 UEG, 37 PRBs per UEG, OFDM Symbol 13: PUCCH 0, PUSCH 1~222, SRS 223~270
    for i = 3920
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    3         2       {4,5,6,3}  {7}   num2cell(167:172)   {1}};
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end

    % modified from pattern 59 3060:3079, 6 UEG, 37 PRBs per UEG; OFDM Symbol 13: PUCCH 0, PUSCH 1~222, SRS 223~270, Nap = 2
    for i = 3921
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    3         2       {4,5,6,3}  {7}   num2cell(167:172)   {2}};
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
        CellIdxInPatternMap(i) = 1;
    end

    % modified from pattern 59 3060:3079, 6 UEG, 37 PRBs per UEG; 4 SRS UEs, wideband on OFDM symbol 13
    for i = 3922
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    3         2       {4,5,6,3}  {7}   num2cell(550:555)   {3}};
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end

    % modified from pattern 59 3060:3079, 6 UEG, 37 PRBs per UEG; 4 SRS UEs, wideband on OFDM symbol 12 13, Nsym = 2
    for i = 3923
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    3         2       {4,5,6,3}  {7}   num2cell(556:561)   {4}};
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end

    % modified from pattern 59 3060:3079, 6 UEG, 37 PRBs per UEG; 8 SRS UEs, wideband, UE 1~4 on OFDM symbol 12; UE 5~8 on OFDM symbol 13
    for i = 3924 % S
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    3         2       {4,5,6,3}  {7}   num2cell(556:561)   {5}};
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end

    for i = 3925 % SRS + PUCCH
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    3         2       {}     {45}   {}   {6}};
    end

    % other two SRS + PUCCH + PUSCH + PRACH TVs with different RNTI
    for i = 3926 % U1
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    4         2       {4,5,6,3}  {7}   num2cell(556:561)   {14}};
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end

    for i = 3927 % U2
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    5         2       {4,5,6,3}  {7}   num2cell(556:561)   {15}};
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end

    % three SRS + PUCCH TVs with different RNTI
    for i = [3928 3931 3934 3937] % S
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    3 + 10*(i-3928)/3        2       {}     {7}   {}   {5}};
        srsRntiStartMap(int32(i)) = (i-3928) * 8 + 1;
    end

    for i = [3929 3932 3935 3938] % U1
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    4 + 10*(i-3929)/3        2       {}     {7}   {}   {14}};
        srsRntiStartMap(int32(i)) = (i-3928) * 8 + 1;
    end

    for i = [3930 3933 3936 3939] % U2
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    5 + 10*(i-3930)/3        2       {}     {7}   {}   {15}};
        srsRntiStartMap(int32(i)) = (i-3928) * 8 + 1;
    end

    % other two SRS + PUCCH + PUSCH + PRACH TVs with different RNTI, cellId = 42
    for i = 3940 % U1, same with 3926 except for cell id
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    4         2       {4,5,6,3}  {7}   num2cell(556:561)   {14}};
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
        CellIdxInPatternMap(i) = 1;
    end

    for i = 3941 % U2, same with 3927 except for cell id
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    5         2       {4,5,6,3}  {7}   num2cell(556:561)   {15}};
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
        CellIdxInPatternMap(i) = 1;
    end
    
    % [3942:4039] are available for future ULMIX TVs

    %% pattern 62c (S slot): 7 beams, 100 MHz (273 PRBs), 40C, peak cell
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 3/13, S slot
    row_count = length(CFG);
    for i = 4040:4079
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {}   num2cell(562:567)   {}};
        CellIdxInPatternMap(i) = mod(i-4040, 40);
    end
    % end pattern 62c (S slot): 7 beams, 100 MHz (273 PRBs), 40C, peak cell

    % [4080:4099] are available for future ULMIX TVs

    %% pattern 63c (S slot): 7 beams, 100 MHz (273 PRBs), 40C, ave cell
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot 3/13, S slot
    for i = 5391:5430
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {}   num2cell(568:573)   {}};
        CellIdxInPatternMap(i) = mod(i-5391, 40);    
        UciOnPuschMap(i) = "7beams_0_10_0";  
    end  
    % end pattern 63c (S slot): 7 beams, 100 MHz (273 PRBs), 40C, ave cell

    %% pattern 60e: 7 beams, 100 MHz (273 PRBs), 20C, ave cell, 1 dmrs, 4UL streams, 18 PUCCH UCIs
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 4220:4239
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {42}   num2cell(592:597)   {}};
    end
    % slot 5, 4 prach
    for i = 4240:4259
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {50,51,52,49}  {42}   num2cell(598:603)   {}}; % {49,50,51,52}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
    end
    % slot 14, no prach
    for i = 4260:4279
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {42}   num2cell(592:597)   {}};
    end
    % slot 15, 3 prach
    for i = 4280:4299
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {50,51,52}     {42}   num2cell(604:609)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
    end
    % end pattern 60e

    %% SRS in consecutive slots with SRS + other channels
    % modified based on 3924: { i    3         2       {4,5,6,3}  {7}   num2cell(556:561)   {5}}
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % three SRS only TVs with different RNTI
    for i = 4300:3:4309  % S
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    3 + 10*(i-4300)/3     2       {}     {}   {}   {5}};
        srsRntiStartMap(int32(i)) = (i-4300) * 8 + 1;
    end

    for i = 4301:3:4310  % U1
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    4 + 10*(i-4301)/3     2       {}     {}   {}   {14}};
        srsRntiStartMap(int32(i)) = (i-4300) * 8 + 1;
    end

    for i = 4302:3:4311  % U2
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    5 + 10*(i-4302)/3     2       {}     {}   {}   {15}};
        srsRntiStartMap(int32(i)) = (i-4300) * 8 + 1;
    end

    % three SRS + PUSCH TVs with different RNTI
    for i = 4312:3:4321  % S
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    3 + 10*(i-4312)/3     2      {}     {}   num2cell(556:561)   {5}};
        srsRntiStartMap(int32(i)) = (i-4312) * 8 + 1;
    end

    for i = 4313:3:4322  % U1
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    4 + 10*(i-4313)/3     2      {}     {}   num2cell(556:561)   {14}};
        srsRntiStartMap(int32(i)) = (i-4312) * 8 + 1;
    end

    for i = 4314:3:4323  % U2
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    5 + 10*(i-4314)/3     2      {}     {}   num2cell(556:561)   {15}};
        srsRntiStartMap(int32(i)) = (i-4312) * 8 + 1;
    end

    % three SRS + PRACH TVs with different RNTI
    for i = 4324:3:4333 % S
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    3 + 10*(i-4324)/3     2      {4,5,6,3}     {}   {}   {5}};
        srsRntiStartMap(int32(i)) = (i-4324) * 8 + 1;
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3
    end
    CellIdxInPatternMap(4327) = 1;

    for i = 4325:3:4334  % U1
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    4 + 10*(i-4325)/3     2      {4,5,6,3}     {}   {}   {14}};
        srsRntiStartMap(int32(i)) = (i-4324) * 8 + 1;
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3
    end
    CellIdxInPatternMap(4328) = 1;
    CellIdxInPatternMap(4331) = 1;
    srsRntiStartMap(4328) = 150;

    for i = 4326:3:4335 % U2
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    5 + 10*(i-4326)/3     2      {4,5,6,3}     {}   {}   {15}};
        srsRntiStartMap(int32(i)) = (i-4324) * 8 + 1;
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3
    end
    CellIdxInPatternMap(4329) = 1;
    srsRntiStartMap(4326) = 200;
    srsRntiStartMap(4329) = 250;
    srsRntiStartMap(4335) = 300;

    for i = 4336 % S, same with 4333 except for cell id
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i           33            2      {4,5,6,3}     {}   {}   {5}};
        srsRntiStartMap(int32(i)) = (i-4324) * 8 + 1;
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3
        CellIdxInPatternMap(i) = 1;
    end
    %%
    
    % [4337:4374] are available for future ULMIX TVs

    %% pattern 101: 1 UE SU-MIMO, 2 layers, MCS 27, 256QAM table, PUSCH + PRACH, 1 pilot @ 3rd symbol, type-A, cdm group 2, 13 data symbols, 10 fixed LDPC iterations, 24C
    % PUSCH: cfg 1045 (slot 4), cfg 1046 (slot 5 with PRACH), cfg 1047 (slot 14), cfg 1048 (slot 15)
    % PRACH: cfg 82
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}
    for i = 4375:4398
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0          2        {}         {}        {1045}     {}};
        CellIdxInPatternMap(i) = mod(i-4375, 24);
        enableOtaMap(i) = 1;
        ldpcFixMaxIterMap(int32(i)) = 10;
    end
    % slot {5}, prach
    for i = 4399:4422
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0          2        {82}       {}        {1046}     {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-4375, 24);
        enableOtaMap(i) = 1;
        ldpcFixMaxIterMap(int32(i)) = 10;
    end
    % slot {14}
    for i = 4423:4446
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0          2        {}         {}        {1047}     {}};
        CellIdxInPatternMap(i) = mod(i-4375, 24);
        enableOtaMap(i) = 1;
        ldpcFixMaxIterMap(int32(i)) = 10;
    end
    % slot {15}
    for i = 4447:4470
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0          2        {}         {}        {1048}     {}};
        CellIdxInPatternMap(i) = mod(i-4375, 24);
        enableOtaMap(i) = 1;
        ldpcFixMaxIterMap(int32(i)) = 10;
    end
    % end pattern 101: 1 UE SU-MIMO, 2 layers, MCS 27, 256QAM table, PUSCH + PRACH, 24C

    %% pattern 59c: 7 beams, 100 MHz (273 PRBs), 40C, peak cell, OTA, 4UL streams, 18 PUCCH UCIs
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 4471:4510
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {42}   num2cell(442:447)   {}};
        CellIdxInPatternMap(i) = mod(i-4471, 40);
        UciOnPuschMap(i) = "7beams_4_37_5";
        enableOtaMap(i) = 1;
    end
    % slot 5, 4 prach
    for i = 4511:4550
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {46,47,48,45}  {42}   num2cell(448:453)   {}}; % {45,46,47,48}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
        CellIdxInPatternMap(i) = mod(i-4471, 40);
        UciOnPuschMap(i) = "7beams_4_37_5";
        enableOtaMap(i) = 1;
    end
    % slot 14, no prach
    for i = 4551:4590
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {42}   num2cell(442:447)   {}};
        CellIdxInPatternMap(i) = mod(i-4471, 40);
        UciOnPuschMap(i) = "7beams_4_37_5";
        enableOtaMap(i) = 1;
    end
    % slot 15, 3 prach
    for i = 4591:4630
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {46,47,48}     {42}   num2cell(454:459)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
        CellIdxInPatternMap(i) = mod(i-4471, 40);
        UciOnPuschMap(i) = "7beams_4_37_5";
        enableOtaMap(i) = 1;
    end
    % end pattern 59c: 7 beams, 100 MHz (273 PRBs), 40C, peak cell, OTA, 4UL streams, 18 PUCCH UCIs

    %% pattern 59e: 7 beams, 100 MHz (273 PRBs), 40C, peak cell, 1 dmrs, 4UL streams, 18 PUCCH UCIs
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    row_count = length(CFG);
    for i = 4631:4670
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {42}   num2cell(574:579)   {}};
        CellIdxInPatternMap(i) = mod(i-4631, 40);
        UciOnPuschMap(i) = "7beams_4_37_5";
        enableOtaMap(i) = 1;
    end
    % slot 5, 4 prach
    for i = 4671:4710
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {46,47,48,45}  {42}   num2cell(580:585)   {}}; % {45,46,47,48}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
        CellIdxInPatternMap(i) = mod(i-4631, 40);
        UciOnPuschMap(i) = "7beams_4_37_5";
        enableOtaMap(i) = 1;
    end
    % slot 14, no prach
    for i = 4711:4750
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {42}   num2cell(574:579)   {}};
        CellIdxInPatternMap(i) = mod(i-4631, 40);
        UciOnPuschMap(i) = "7beams_4_37_5";
        enableOtaMap(i) = 1;
    end
    % slot 15, 3 prach
    for i = 4751:4790
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {46,47,48}     {42}   num2cell(586:591)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
        CellIdxInPatternMap(i) = mod(i-4631, 40);
        UciOnPuschMap(i) = "7beams_4_37_5";
        enableOtaMap(i) = 1;
    end
    % end pattern 59e: 7 beams, 100 MHz (273 PRBs), 40C, peak cell, 1 dmrs, 4UL streams, 18 PUCCH UCIs

    %% pattern 67: 64TR, 100 MHz (273 PRBs), 64TR worst case column B TC, 15C, based on 90623
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {3}
    % for i = 4791:4805
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {12}};
    %     CellIdxInPatternMap(i) = mod(i-4791, 15);
    % end
    % slot {4}
    for i = 4806:4820
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {55}   num2cell(664:669)   {}};
        CellIdxInPatternMap(i) = mod(i-4791, 15);
    end
    % slot {5}, prach
    for i = 4821:4835
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11      {69}         {56}   num2cell(664:669)   {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-4791, 15);
    end
    % slot {13}
    % for i = 4836:4850
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0         11        {}         {}     {}   {13}};
    %     CellIdxInPatternMap(i) = mod(i-4791, 15);
    % end
    % slot {14}
    for i = 4851:4865
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {57}   num2cell(664:669)   {}};
        CellIdxInPatternMap(i) = mod(i-4791, 15);
    end
    % slot {15}, prach
    for i = 4866:4880
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11      {69}         {58, 59}   num2cell(664:669)   {}};
        PrachBeamIdxMap(i) = [2 3];
        CellIdxInPatternMap(i) = mod(i-4791, 15);
    end
    % slot {23}
    for i = 4881:4895
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {22}};
        CellIdxInPatternMap(i) = mod(i-4791, 15);
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33}
    for i = 4896:4910
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {23}};
        CellIdxInPatternMap(i) = mod(i-4791, 15);
    end
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % end pattern 67: 64TR, 100 MHz (273 PRBs), 64TR worst case column B TC, 15C, based on 90623

    %% pattern 65a: 7 beams, 100 MHz (273 PRBs), 40C, peak cell, OTA, 4UL streams, PUSCH CP-OFDM, PUSCH only U slot
    row_count = length(CFG);
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4/5/14/15, PUSCH only U slot
    for i = 5971:6010
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {}   num2cell(690:695)   {}};
        CellIdxInPatternMap(i) = mod(i-5971, 40);
        UciOnPuschMap(i) = "7beams_4_37_5";
        enableOtaMap(i) = 1;
    end
    % end pattern 65a: 7 beams, 100 MHz (273 PRBs), 40C, peak cell, OTA, 4UL streams, PUSCH CP-OFDM, PUSCH only U slot

    %% pattern 65b: 7 beams, 100 MHz (273 PRBs), 40C, peak cell, OTA, 4UL streams, PUSCH DFT-s-OFDM, P6011USCH only U slot
    row_count = length(CFG);
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4/5/14/15, PUSCH only U slot
    for i = 6011:6050
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {}   num2cell(696:701)   {}};
        CellIdxInPatternMap(i) = mod(i-6011, 40);
        UciOnPuschMap(i) = "7beams_4_37_5";
        enableOtaMap(i) = 1;
        puschIdentityMap(i) = 128;
    end
    % end pattern 65b: 7 beams, 100 MHz (273 PRBs), 40C, peak cell, OTA, 4UL streams, PUSCH DFT-s-OFDM, P6011USCH only U slot

    %% pattern 65c: 7 beams, 100 MHz (273 PRBs), 40C, peak cell, OTA, 4UL streams, PUSCH CP-OFDM, Mixed PUCCH F1 HARQ and SR
    row_count = length(CFG);
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    for i = 6051:6090
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {68}   num2cell(702:707)   {}};
        CellIdxInPatternMap(i) = mod(i-6051, 40);
        UciOnPuschMap(i) = "7beams_4_37_5";
        enableOtaMap(i) = 1;
    end
    % slot 5, 4 prach
    for i = 6091:6130
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {4,5,6,3}  {69}   num2cell(708:713)   {}}; % {3,4,5,6}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
        CellIdxInPatternMap(i) = mod(i-6051, 40);
        UciOnPuschMap(i) = "7beams_4_37_5";
        enableOtaMap(i) = 1;
    end
    % slot 14, no prach
    for i = 6131:6170
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {70}   num2cell(702:707)   {}};
        CellIdxInPatternMap(i) = mod(i-6051, 40);
        UciOnPuschMap(i) = "7beams_4_37_5";
        enableOtaMap(i) = 1;
    end
    % slot 15, 3 prach
    for i = 6171:6210
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {4,5,6}     {71}   num2cell(714:719)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
        CellIdxInPatternMap(i) = mod(i-6051, 40);
        UciOnPuschMap(i) = "7beams_4_37_5";
        enableOtaMap(i) = 1;
    end
    % end pattern 65c: 7 beams, 100 MHz (273 PRBs), 40C, peak cell, OTA, 4UL streams, PUSCH CP-OFDM, Mixed PUCCH F1 HARQ and SR

    %% pattern 65d: 7 beams, 100 MHz (273 PRBs), 40C, peak cell, OTA, 4UL streams, PUSCH DFT-s-OFDM, Mixed PUCCH F1 HARQ and SR
    row_count = length(CFG);
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4, no prach
    for i = 6211:6250
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {68}   num2cell(702:707)   {}};
        CellIdxInPatternMap(i) = mod(i-6211, 40);
        UciOnPuschMap(i) = "7beams_4_37_5";
        enableOtaMap(i) = 1;
        puschIdentityMap(i) = 128;
    end
    % slot 5, 4 prach
    for i = 6251:6290
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {4,5,6,3}  {69}   num2cell(708:713)   {}}; % {3,4,5,6}
        PrachBeamIdxMap(i) = 0; % + prach{idx} 0,1,2,3 
        CellIdxInPatternMap(i) = mod(i-6211, 40);
        UciOnPuschMap(i) = "7beams_4_37_5";
        enableOtaMap(i) = 1;
        puschIdentityMap(i) = 128;
    end
    % slot 14, no prach
    for i = 6291:6330
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2        {}         {70}   num2cell(702:707)   {}};
        CellIdxInPatternMap(i) = mod(i-6211, 40);
        UciOnPuschMap(i) = "7beams_4_37_5";
        enableOtaMap(i) = 1;
        puschIdentityMap(i) = 128;
    end
    % slot 15, 3 prach
    for i = 6331:6370
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         2       {4,5,6}     {71}   num2cell(714:719)   {}};
        PrachBeamIdxMap(i) = 4; % + prach{idx} 4,5,6
        CellIdxInPatternMap(i) = mod(i-6211, 40);
        UciOnPuschMap(i) = "7beams_4_37_5";
        enableOtaMap(i) = 1;
        puschIdentityMap(i) = 128;
    end
    % end pattern 65d: 7 beams, 100 MHz (273 PRBs), 40C, peak cell, OTA, 4UL streams, PUSCH DFT-s-OFDM, Mixed PUCCH F1 HARQ and SR

    %% pattern 67a: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column B TC, 15C, based on 90623, PUSCH CP-OFDM, PUSCH only U slot
    row_count = length(CFG);
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4/5/14/15, PUSCH only U slot
    for i = 5701:5715
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   num2cell(690:695)   {}};
        CellIdxInPatternMap(i) = mod(i-5701, 15);
    end
    % end pattern 67a: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column B TC, 15C, based on 90623, PUSCH CP-OFDM, PUSCH only U slot

    %% pattern 67b: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column B TC, 15C, based on 90623, PUSCH DFT-s-OFDM, PUSCH only U slot
    row_count = length(CFG);
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4/5/14/15, PUSCH only U slot
    for i = 5716:5730
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   num2cell(696:701)   {}};
        CellIdxInPatternMap(i) = mod(i-5716, 15);
        puschIdentityMap(i) = 128;
    end
    % end pattern 67b: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column B TC, 15C, based on 90623, PUSCH DFT-s-OFDM, PUSCH only U slot
 
    %% pattern 67c: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column B TC, 15C, based on 90623, PUSCH CP-OFDM, Mixed PUCCH F1 HARQ and SR
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {3}
    % for i = 5731:5745
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {12}};
    %     CellIdxInPatternMap(i) = mod(i-5731, 15);
    % end
    % slot {4}
    for i = 5746:5760
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {68}   num2cell(702:707)   {}};
        CellIdxInPatternMap(i) = mod(i-5731, 15);
    end
    % slot {5}, prach
    for i = 5761:5775
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11      {75}         {69}   num2cell(761:766)   {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-5731, 15);
    end

   % Negative test vector for the first cell of F08 test cases (FAPI override with invalid params)
    i = 6996;
    row_count = row_count + 1;
    CFG(row_count, 1:7) = { i    0         11      {75}         {69}   num2cell(761:766)   {}};
    PrachBeamIdxMap(i) = [0 1];
    CellIdxInPatternMap(i) = 2;

    % slot {13}
    % for i = 5776:5790
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0         11        {}         {}     {}   {13}};
    %     CellIdxInPatternMap(i) = mod(i-5731, 15);
    % end
    % slot {14}
    for i = 5791:5805
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {70}   num2cell(702:707)   {}};
        CellIdxInPatternMap(i) = mod(i-5731, 15);
    end
    % slot {15}, prach
    for i = 5806:5820
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11      {75}         {71}   num2cell(761:766)   {}};
        PrachBeamIdxMap(i) = [2 3];
        CellIdxInPatternMap(i) = mod(i-5731, 15);
    end
    % slot {23}
    for i = 5821:5835
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {22}};
        CellIdxInPatternMap(i) = mod(i-5731, 15);
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33}
    for i = 5836:5850
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {23}};
        CellIdxInPatternMap(i) = mod(i-5731, 15);
    end
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % end pattern 67c: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column B TC, 15C, based on 90623, PUSCH CP-OFDM, Mixed PUCCH F1 HARQ and SR

    %% pattern 67d: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column B TC, 15C, based on 90623, PUSCH DFT-s-OFDM, Mixed PUCCH F1 HARQ and SR
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {3}
    % for i = 5851:5865
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {12}};
    %     CellIdxInPatternMap(i) = mod(i-5851, 15);
    % end
    % slot {4}
    for i = 5866:5880
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {68}   num2cell(702:707)   {}};
        CellIdxInPatternMap(i) = mod(i-5851, 15);
        puschIdentityMap(i) = 128;
    end
    % slot {5}, prach
    for i = 5881:5895
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11      {75}         {69}   num2cell(761:766)   {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-5851, 15);
        puschIdentityMap(i) = 128;
    end
    % slot {13}
    % for i = 5896:5910
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0         11        {}         {}     {}   {13}};
    %     CellIdxInPatternMap(i) = mod(i-5851, 15);
    % end
    % slot {14}
    for i = 5911:5925
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {70}   num2cell(702:707)   {}};
        CellIdxInPatternMap(i) = mod(i-5851, 15);
        puschIdentityMap(i) = 128;
    end
    % slot {15}, prach
    for i = 5926:5940
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11      {75}         {71}   num2cell(761:766)   {}};
        PrachBeamIdxMap(i) = [2 3];
        CellIdxInPatternMap(i) = mod(i-5851, 15);
        puschIdentityMap(i) = 128;
    end
    % slot {23}
    for i = 5941:5955
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {22}};
        CellIdxInPatternMap(i) = mod(i-5851, 15);
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33}
    for i = 5956:5970
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {23}};
        CellIdxInPatternMap(i) = mod(i-5851, 15);
    end
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % end pattern 67d: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column B TC, 15C, based on 90623, PUSCH DFT-s-OFDM, Mixed PUCCH F1 HARQ and SR
    
    %% pattern 66a: 64TR ave, 100 MHz (273 PRBs), OTA, 64TR worst case column B TC, 15C, based on 90623, PUSCH CP-OFDM, PUSCH only U slot
    row_count = length(CFG);
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4/5/14/15, PUSCH only U slot
    for i = 5431:5445
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   num2cell(720:725)   {}};
        CellIdxInPatternMap(i) = mod(i-5431, 15);
    end
    % end pattern 66a: 64TR ave, 100 MHz (273 PRBs), OTA, 64TR worst case column B TC, 15C, based on 90623, PUSCH CP-OFDM, PUSCH only U slot
 
    %% pattern 66b: 64TR ave, 100 MHz (273 PRBs), OTA, 64TR worst case column B TC, 15C, based on 90623, PUSCH DFT-s-OFDM, PUSCH only U slot
    row_count = length(CFG);
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    % slot 4/5/14/15, PUSCH only U slot
    for i = 5446:5460
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   num2cell(720:725)   {}};
        CellIdxInPatternMap(i) = mod(i-5446, 15);
        puschIdentityMap(i) = 128;
    end
    % end pattern 66b: 64TR ave, 100 MHz (273 PRBs), OTA, 64TR worst case column B TC, 15C, based on 90623, PUSCH DFT-s-OFDM, PUSCH only U slot

    %% pattern 66c: 64TR ave, 100 MHz (273 PRBs), OTA, 64TR worst case column B TC, 15C, based on 90623, PUSCH CP-OFDM, Mixed PUCCH F1 HARQ and SR
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {3}
    % for i = 5461:5475
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {12}};
    %     CellIdxInPatternMap(i) = mod(i-5461, 15);
    % end
    % slot {4}
    for i = 5476:5490
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {68}   num2cell(726:731)   {}};
        CellIdxInPatternMap(i) = mod(i-5461, 15);
    end
    % slot {5}, prach
    for i = 5491:5505
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11      {74}         {69}   num2cell(732:737)   {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-5461, 15);
    end
    % slot {13}
    % for i = 5506:5520
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0         11        {}         {}     {}   {13}};
    %     CellIdxInPatternMap(i) = mod(i-5461, 15);
    % end
    % slot {14}
    for i = 5521:5535
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {70}   num2cell(726:731)   {}};
        CellIdxInPatternMap(i) = mod(i-5461, 15);
    end
    % slot {15}, prach
    for i = 5536:5550
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11      {74}         {71}   num2cell(732:737)   {}};
        PrachBeamIdxMap(i) = [2 3];
        CellIdxInPatternMap(i) = mod(i-5461, 15);
    end
    % slot {23}
    for i = 5551:5565
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {22}};
        CellIdxInPatternMap(i) = mod(i-5461, 15);
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33}
    for i = 5566:5580
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {23}};
        CellIdxInPatternMap(i) = mod(i-5461, 15);
    end
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % end pattern 66c: 64TR ave, 100 MHz (273 PRBs), OTA, 64TR worst case column B TC, 15C, based on 90623, PUSCH CP-OFDM, Mixed PUCCH F1 HARQ and SR
    
    %% pattern 66d: 64TR ave, 100 MHz (273 PRBs), OTA, 64TR worst case column B TC, 15C, based on 90623, PUSCH DFT-s-OFDM, Mixed PUCCH F1 HARQ and SR
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {3}
    % for i = 5581:5595
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {12}};
    %     CellIdxInPatternMap(i) = mod(i-5581, 15);
    % end
    % slot {4}
    for i = 5596:5610
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {68}   num2cell(726:731)   {}};
        CellIdxInPatternMap(i) = mod(i-5581, 15);
        puschIdentityMap(i) = 128;
    end
    % slot {5}, prach
    for i = 5611:5625
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11      {74}         {69}   num2cell(732:737)   {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-5581, 15);
        puschIdentityMap(i) = 128;
    end
    % slot {13}
    % for i = 5626:5640
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0         11        {}         {}     {}   {13}};
    %     CellIdxInPatternMap(i) = mod(i-5581, 15);
    % end
    % slot {14}
    for i = 5641:5655
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {70}   num2cell(726:731)   {}};
        CellIdxInPatternMap(i) = mod(i-5581, 15);
        puschIdentityMap(i) = 128;
    end
    % slot {15}, prach
    for i = 5656:5670
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11      {74}         {71}   num2cell(732:737)   {}};
        PrachBeamIdxMap(i) = [2 3];
        CellIdxInPatternMap(i) = mod(i-5581, 15);
        puschIdentityMap(i) = 128;
    end
    % slot {23}
    for i = 5671:5685
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {22}};
        CellIdxInPatternMap(i) = mod(i-5581, 15);
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33}
    for i = 5686:5700
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {23}};
        CellIdxInPatternMap(i) = mod(i-5581, 15);
    end
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % end pattern 66d: 64TR ave, 100 MHz (273 PRBs), OTA, 64TR worst case column B TC, 15C, based on 90623, PUSCH DFT-s-OFDM, Mixed PUCCH F1 HARQ and SR

    %% pattern 69c: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column G TC, 15C, based on 90624, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEG, 8 UEs per UEG, 1 layer per UE
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}
    for i = 6371:6385
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {76}   num2cell(738:745)   {}};
        CellIdxInPatternMap(i) = mod(i-6371, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {5}, prach
    for i = 6386:6400
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11    num2cell(70:73)   {73}   num2cell(746:753)   {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-6371, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {13} is DL only
    % slot {14}
    for i = 6401:6415
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {75,74}   num2cell(738:745)   {}};
        CellIdxInPatternMap(i) = mod(i-6371, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {15}, prach
    for i = 6416:6430
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11    num2cell(70:73)   {72}   num2cell(746:753)   {}};
        PrachBeamIdxMap(i) = [2 3];
        CellIdxInPatternMap(i) = mod(i-6371, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {23}
    for i = 6431:6445
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {26}};
        CellIdxInPatternMap(i) = mod(i-6371, 15);
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33} is DL only
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % end pattern 69c: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column G TC, 15C, based on 90624, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEG, 8 UEs per UEG, 1 layer per UE

    %% pattern 69: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column G TC, 15C, based on 90624, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEG, 8 UEs per UEG, 1 layer per UE, Early HARQ disabled
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}
    for i = 6621:6635
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {76}   num2cell(738:745)   {}};
        CellIdxInPatternMap(i) = mod(i-6621, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
        earlyHarqDisableMap(i) = 1;
    end
    % slot {5}, prach
    for i = 6636:6650
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11    num2cell(70:73)   {73}   num2cell(746:753)   {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-6621, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
        earlyHarqDisableMap(i) = 1;
    end
    % slot {13} is DL only
    % slot {14}
    for i = 6651:6665
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {75,74}   num2cell(738:745)   {}};
        CellIdxInPatternMap(i) = mod(i-6621, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
        earlyHarqDisableMap(i) = 1;
    end
    % slot {15}, prach
    for i = 6666:6680
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11    num2cell(70:73)   {72}   num2cell(746:753)   {}};
        PrachBeamIdxMap(i) = [2 3];
        CellIdxInPatternMap(i) = mod(i-6621, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
        earlyHarqDisableMap(i) = 1;
    end
    % slot {23}
    for i = 6681:6695
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {26}};
        CellIdxInPatternMap(i) = mod(i-6621, 15);
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33} is DL only
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % end pattern 69: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column G TC, 15C, based on 90624, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEG, 8 UEs per UEG, 1 layer per UE, Early HARQ disabled
        
    %% pattern 69a: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column G TC, 15C, based on 90624, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 4 UEGs, 2 UEs per UEG, 1 layer per UE
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}
    for i = 6696:6710
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {76}   num2cell(799:806)   {}};
        CellIdxInPatternMap(i) = mod(i-6696, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {5}, prach
    for i = 6711:6725
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11    num2cell(70:73)   {73}   num2cell(807:814)   {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-6696, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {13} is DL only
    % slot {14}
    for i = 6726:6740
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {75,74}   num2cell(799:806)   {}};
        CellIdxInPatternMap(i) = mod(i-6696, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {15}, prach
    for i = 6741:6755
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11    num2cell(70:73)   {72}   num2cell(807:814)   {}};
        PrachBeamIdxMap(i) = [2 3];
        CellIdxInPatternMap(i) = mod(i-6696, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {23}
    for i = 6756:6770
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {26}};
        CellIdxInPatternMap(i) = mod(i-6696, 15);
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33} is DL only
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % end pattern 69a: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column G TC, 15C, based on 90624, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 4 UEGs, 2 UEs per UEG, 1 layer per UE

    %% pattern 69b: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column G TC, 15C, based on 90624, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 2 UEGs, 4 UEs per UEG, 1 layer per UE
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}
    for i = 6771:6785
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {76}   num2cell(815:822)   {}};
        CellIdxInPatternMap(i) = mod(i-6771, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {5}, prach
    for i = 6786:6800
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11    num2cell(70:73)   {73}   num2cell(823:830)   {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-6771, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {13} is DL only
    % slot {14}
    for i = 6801:6815
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {75,74}   num2cell(815:822)   {}};
        CellIdxInPatternMap(i) = mod(i-6771, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {15}, prach
    for i = 6816:6830
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11    num2cell(70:73)   {72}   num2cell(823:830)   {}};
        PrachBeamIdxMap(i) = [2 3];
        CellIdxInPatternMap(i) = mod(i-6771, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {23}
    for i = 6831:6845
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {26}};
        CellIdxInPatternMap(i) = mod(i-6771, 15);
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33} is DL only
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % end pattern 69b: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column G TC, 15C, based on 90624, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 2 UEGs, 4 UEs per UEG, 1 layer per UE

    %% pattern 69d: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column G TC, 15C, based on 90624, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEG, 8 UEs per UEG, 1 layer per UE, no HARQ bits for UCI-on-PUSCH in slot 5/15
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}
    for i = 6846:6860
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {76}   num2cell(738:745)   {}};
        CellIdxInPatternMap(i) = mod(i-6846, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {5}, prach
    for i = 6861:6875
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11    num2cell(70:73)   {73}   num2cell(746:753)   {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-6846, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_0_6_11";
    end
    % slot {13} is DL only
    % slot {14}
    for i = 6876:6890
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {75,74}   num2cell(738:745)   {}};
        CellIdxInPatternMap(i) = mod(i-6846, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {15}, prach
    for i = 6891:6905
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11    num2cell(70:73)   {72}   num2cell(746:753)   {}};
        PrachBeamIdxMap(i) = [2 3];
        CellIdxInPatternMap(i) = mod(i-6846, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_0_6_11";
    end
    % slot {23}
    for i = 6906:6920
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {26}};
        CellIdxInPatternMap(i) = mod(i-6846, 15);
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33} is DL only
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % end pattern 69d: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column G TC, 15C, based on 90624, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEG, 8 UEs per UEG, 1 layer per UE, no HARQ bits for UCI-on-PUSCH in slot 5/15
    
    %% pattern 71: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column H TC, 15C, based on 90626, PUSCH CP-OFDM, Mixed PUCCH F1/F3
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}
    for i = 6446:6460  % (PUCCH 16 + 72) + (PUSCH 8)  + (SRS 24) = 120 PDUs
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {81,82}   num2cell(767:774)   {31}};
        CellIdxInPatternMap(i) = mod(i-6446, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_31_11";
    end
    % slot {5}, prach
    for i = 6461:6475  % (PRACH 4) + (PUCCH 32 + 72) + (PUSCH 8)  + (SRS 24) = 140 PDUs
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11    num2cell(70:73)   {78,83}   num2cell(775:782)   {32}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-6446, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_31_11";
    end
    % slot {13} is DL only
    % slot {14}
    for i = 6476:6490  % (PUCCH 1 + 21 + 72) + (PUSCH 8)  + (SRS 24) = 126 PDUs
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {80,79,82}   num2cell(767:774)   {33}};
        CellIdxInPatternMap(i) = mod(i-6446, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_31_11";
    end
    % slot {15}, prach
    for i = 6491:6505   % (PRACH 4) + (PUCCH 64 + 72) + (PUSCH 8)  + (SRS 24) = 172 PDUs
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11    num2cell(70:73)   {77,83}   num2cell(775:782)   {34}};
        PrachBeamIdxMap(i) = [2 3];
        CellIdxInPatternMap(i) = mod(i-6446, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_31_11";
    end
    % slot {23}
    for i = 6506:6520  % (SRS 80) = 80 PDUs
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {28}};
        CellIdxInPatternMap(i) = mod(i-6446, 15);
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33} is DL only
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % end pattern 71: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column H TC, 15C, based on 90626, PUSCH CP-OFDM, Mixed PUCCH F1/F3
    
    %% pattern 73: 64TR peak, 40 MHz (106 PRBs), OTA, 64TR worst case column G TC, 20C, based on 90624, PUSCH CP-OFDM, Mixed PUCCH F1/F3
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}
    for i = 6521:6540
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         13        {}         {76}   num2cell(783:790)   {}};
        CellIdxInPatternMap(i) = mod(i-6521, 20);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {5}, prach
    for i = 6541:6560
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         13    num2cell(76:79)   {73}   num2cell(791:798)   {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-6521, 20);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {13} is DL only
    % slot {14}
    for i = 6561:6580
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         13        {}         {75,74}   num2cell(783:790)   {}};
        CellIdxInPatternMap(i) = mod(i-6521, 20);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {15}, prach
    for i = 6581:6600
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         13    num2cell(76:79)   {72}   num2cell(791:798)   {}};
        PrachBeamIdxMap(i) = [2 3];
        CellIdxInPatternMap(i) = mod(i-6521, 20);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {23}
    for i = 6601:6620
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         13        {}         {}   {}   {30}};
        CellIdxInPatternMap(i) = mod(i-6521, 20);
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33} is DL only
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % end pattern 73: 64TR peak, 40 MHz (106 PRBs), OTA, 64TR worst case column G TC, 20C, based on 90624, PUSCH CP-OFDM, Mixed PUCCH F1/F3

    %% pattern 75: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column G TC, 15C, based on 90624, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 16 UEG, 4 UEs per UEG, 2 layers per UE
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}
    for i = 6921:6935
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {76}   num2cell(2001:2064)   {}};
        CellIdxInPatternMap(i) = mod(i-6921, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {5}, prach
    for i = 6936:6950
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11    num2cell(70:73)   {73}   num2cell(2065:2128)   {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-6921, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {13} is DL only
    % slot {14}
    for i = 6951:6965
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {75,74}   num2cell(2001:2064)   {}};
        CellIdxInPatternMap(i) = mod(i-6921, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 2;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {15}, prach
    for i = 6966:6980
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11    num2cell(70:73)   {72}   num2cell(2065:2128)   {}};
        PrachBeamIdxMap(i) = [2 3];
        CellIdxInPatternMap(i) = mod(i-6921, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 3;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {23}
    for i = 6981:6995
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {28}};
        CellIdxInPatternMap(i) = mod(i-6921, 15);
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33} is DL only
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % end pattern 75: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column G TC, 15C, based on 90624, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 16 UEG, 4 UEs per UEG, 2 layers per UE

    % [6997:7049] are available for future ULMIX TVs, 6996 has been used for negative TC testing above
    
    %% pattern 77: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column I TC, 15C, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 2 UEG, 4 UEs per UEG, 1 layer per UE
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}
    for i = 7050:7064
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {81,82}   num2cell(2129:2136)   {}};
        CellIdxInPatternMap(i) = mod(i-7050, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_18_5";
    end
    % slot {5}, prach
    for i = 7065:7079
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {75}       {78,83}   num2cell(2137:2144)   {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-7050, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_18_5";
    end
    % slot {13}
    for i = 7080:7094
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {26}};
        CellIdxInPatternMap(i) = mod(i-7050, 15);
    end
    % slot {14}
    for i = 7095:7109
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {80,79,82}   num2cell(2129:2136)   {}};
        CellIdxInPatternMap(i) = mod(i-7050, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_18_5";
    end
    % slot {15}, prach
    for i = 7110:7124
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {75}       {77,83}   num2cell(2137:2144)   {}};
        PrachBeamIdxMap(i) = [2 3];
        CellIdxInPatternMap(i) = mod(i-7050, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_18_5";
    end
    % slot {23}
    for i = 7125:7139
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {31}};
        CellIdxInPatternMap(i) = mod(i-7050, 15);
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33}
    for i = 7140:7154
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {32}};
        CellIdxInPatternMap(i) = mod(i-7050, 15);
    end
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % slot {43,44,45,53,54,55,63,64,65,73,74,75} is the same with {* % 40}
    % end pattern 77: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column I TC, 15C, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 2 UEG, 4 UEs per UEG, 1 layer per UE
    
    %% pattern 79: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case Ph4 column B TC, 15C, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEG, 8 UEs per UEG, 1 layer per UE
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}
    for i = 7155:7169
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {81,82}   num2cell(2145:2152)   {}};
        CellIdxInPatternMap(i) = mod(i-7155, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_18_7";
    end
    % slot {5}, prach
    for i = 7170:7184
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {75}       {78,83}   num2cell(2153:2160)   {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-7155, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_18_7";
    end
    % slot {13}
    for i = 7185:7199
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {26}};
        CellIdxInPatternMap(i) = mod(i-7155, 15);
    end
    % slot {14}
    for i = 7200:7214
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {80,79,82}   num2cell(2145:2152)   {}};
        CellIdxInPatternMap(i) = mod(i-7155, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_18_7";
    end
    % slot {15}
    for i = 7215:7229
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11          {}       {77,83}      num2cell(2161:2168)   {}};
        CellIdxInPatternMap(i) = mod(i-7155, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_18_7";
    end
    % slot {23}
    for i = 7230:7244
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {31}};
        CellIdxInPatternMap(i) = mod(i-7155, 15);
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33}
    for i = 7245:7259
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {32}};
        CellIdxInPatternMap(i) = mod(i-7155, 15);
    end
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % slot {43,44,45,53,54,55,63,64,65,73,74,75} is the same with {* % 40}
    % end pattern 79: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case Ph4 column B TC, 15C, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEG, 8 UEs per UEG, 1 layer per UE
    
    %% pattern 69e: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column G TC, 15C, based on 90624, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 2 UEGs, 4 UEs per UEG, 1 layer per UE, Early HARQ enabled, SRS on all slots, ODC pattern
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}
    for i = 7260:7274
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {81}   num2cell(831:838)   {26}};
        CellIdxInPatternMap(i) = mod(i-7260, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        srsRntiStartMap(i) = 31;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {5}, prach
    for i = 7275:7289
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11    num2cell(70:73)   {78}   num2cell(839:846)   {26}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-7260, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        srsRntiStartMap(i) = 55;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {13} is S slot given below, SRS RNTI start at 79
    % slot {14}
    for i = 7290:7304
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {80,79}   num2cell(831:838)   {26}};
        CellIdxInPatternMap(i) = mod(i-7260, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        srsRntiStartMap(i) = 103;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {15}, prach
    for i = 7305:7319
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11    num2cell(70:73)   {77}   num2cell(839:846)   {26}};
        PrachBeamIdxMap(i) = [2 3];
        CellIdxInPatternMap(i) = mod(i-7260, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        srsRntiStartMap(i) = 127;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_6_11";
    end
    % slot {23} is S slot given below, SRS RNTI start at 151
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33} is S slot given below, SRS RNTI start at 175
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % slot {13}
    for i = 7320:7334
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {26}};
        CellIdxInPatternMap(i) = mod(i-7260, 15);
        srsRntiStartMap(i) = 79;
    end
    % slot {23}
    for i = 7335:7349
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {26}};
        CellIdxInPatternMap(i) = mod(i-7260, 15);
        srsRntiStartMap(i) = 151;
    end
    % slot {33}
    for i = 7350:7364
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {26}};
        CellIdxInPatternMap(i) = mod(i-7260, 15);
        srsRntiStartMap(i) = 175;
    end
    % end pattern 69e: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column G TC, 15C, based on 90624, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 2 UEGs, 4 UEs per UEG, 1 layer per UE, Early HARQ enabled, SRS on all slots, ODC pattern

    %% pattern 81a: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR 25-3 column B TC, 15C, SRS in S slot (4 symbols), PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEG, 8 UEs per UEG, 1 layer per UE
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}
    for i = 7365:7379
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {81,82}   num2cell(2145:2152)   {}};
        CellIdxInPatternMap(i) = mod(i-7365, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {5}, prach
    for i = 7380:7394
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {75}       {78,83}   num2cell(2153:2160)   {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-7365, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {13}, SRS in S slot (4 symbols)
    for i = 7395:7409
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {36}};
        CellIdxInPatternMap(i) = mod(i-7365, 15);
        srsRntiStartMap(i) = 55;
    end
    % slot {14}
    for i = 7410:7424
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {80,79,82}   num2cell(2145:2152)   {}};
        CellIdxInPatternMap(i) = mod(i-7365, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {15}
    for i = 7425:7439
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11          {}       {77,83}      num2cell(2161:2168)   {}};
        CellIdxInPatternMap(i) = mod(i-7365, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {23}, SRS in S slot (4 symbols)
    for i = 7440:7454
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {36}};
        CellIdxInPatternMap(i) = mod(i-7365, 15);
        srsRntiStartMap(i) = 103;
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33}, SRS in S slot (4 symbols)
    for i = 7455:7469
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {36}};
        CellIdxInPatternMap(i) = mod(i-7365, 15);
        srsRntiStartMap(i) = 151;
    end
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % slot {43,44,45,53,54,55,63,64,65,73,74,75} is the same with {* % 40}
    % end pattern 81a: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR 25-3 column B TC, 15C, SRS in S slot (4 symbols), PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEG, 8 UEs per UEG, 1 layer per UE

    %% pattern 81b: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR 25-3 column B TC, 15C, SRS in S slot (2 symbols), U slot, and U2 slot, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEG, 8 UEs per UEG, 1 layer per UE
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}, SRS in U slot
    for i = 7470:7484
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {81,82}   num2cell(2169:2176)   {37}};
        CellIdxInPatternMap(i) = mod(i-7470, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {5}, prach, SRS in U2 slot
    for i = 7485:7499
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {75}       {78,83}   num2cell(2177:2184)   {37}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-7470, 15);
        srsRntiStartMap(i) = 67;
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {13}, SRS in S slot (2 symbols)
    for i = 7500:7514
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {26}};
        CellIdxInPatternMap(i) = mod(i-7470, 15);
    end
    % slot {14}, SRS in U slot
    for i = 7515:7529
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {80,79,82}   num2cell(2169:2176)   {37}};
        CellIdxInPatternMap(i) = mod(i-7470, 15);
        srsRntiStartMap(i) = 79;
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {15}, SRS in U2 slot
    for i = 7530:7544
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11          {}       {77,83}      num2cell(2185:2192)   {37}};
        CellIdxInPatternMap(i) = mod(i-7470, 15);
        srsRntiStartMap(i) = 91;
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {23}, SRS in S slot (2 symbols)
    for i = 7545:7559
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {26}};
        srsRntiStartMap(i) = 103;
        CellIdxInPatternMap(i) = mod(i-7470, 15);
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33}, SRS in S slot (2 symbols)
    for i = 7560:7574
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {26}};
        CellIdxInPatternMap(i) = mod(i-7470, 15);
        srsRntiStartMap(i) = 127;
    end
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % slot {43,44,45,53,54,55,63,64,65,73,74,75} is the same with {* % 40}
    % end pattern 81b: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR 25-3 column B TC, 15C, SRS in S slot (2 symbols), U slot, and U2 slot, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEG, 8 UEs per UEG, 1 layer per U
    
    %% pattern 83a: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR 25-3 column G TC, 15C, 64 UEs per TTI, SRS in S slot (4 symbols), PUSCH CP-OFDM, Mixed PUCCH F1/F3, 8 UEG, 8 UEs per UEG, 1 layer per UE
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}
    for i = 7575:7589
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {88,89}   num2cell(2193:2200)   {}};
        CellIdxInPatternMap(i) = mod(i-7575, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {5}, prach
    for i = 7590:7604
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {75}       {85,82}   num2cell(2201:2208)   {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-7575, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {13}, SRS in S slot (4 symbols)
    for i = 7605:7619
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {36}};
        CellIdxInPatternMap(i) = mod(i-7575, 15);
    end
    % slot {14}
    for i = 7620:7634
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {87,86,90}   num2cell(2209:2216)   {}};
        CellIdxInPatternMap(i) = mod(i-7575, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {15}
    for i = 7635:7649
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11          {}       {84,82}      num2cell(2217:2224)   {}};
        CellIdxInPatternMap(i) = mod(i-7575, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {23}, SRS in S slot (4 symbols)
    for i = 7650:7664
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {36}};
        CellIdxInPatternMap(i) = mod(i-7575, 15);
        srsRntiStartMap(i) = 103;
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33}, SRS in S slot (4 symbols)
    for i = 7665:7679
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {36}};
        CellIdxInPatternMap(i) = mod(i-7575, 15);
        srsRntiStartMap(i) = 151;
    end
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % slot {43,44,45,53,54,55,63,64,65,73,74,75} is the same with {* % 40}
    % end pattern 83a: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR 25-3 column G TC, 15C, 64 UEs per TTI, SRS in S slot (4 symbols), PUSCH CP-OFDM, Mixed PUCCH F1/F3, 8 UEG, 8 UEs per UEG, 1 layer per UE
            
    %% pattern 83b: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR 25-3 column G TC, 15C, 64 UEs per TTI, SRS in S slot (2 symbols), U slot, and U2 slot, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 8 UEG, 8 UEs per UEG, 1 layer per UE
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}, SRS in U slot
    for i = 7680:7694
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {88,89}   num2cell(2225:2232)   {37}};
        CellIdxInPatternMap(i) = mod(i-7680, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {5}, prach, SRS in U2 slot
    for i = 7695:7709
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {75}       {85,82}   num2cell(2233:2240)   {37}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-7680, 15);
        srsRntiStartMap(i) = 67;
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {13}, SRS in S slot (2 symbols)
    for i = 7710:7724
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {26}};
        CellIdxInPatternMap(i) = mod(i-7680, 15);
    end
    % slot {14}, SRS in U slot
    for i = 7725:7739
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {87,86,90}   num2cell(2241:2248)   {37}};
        CellIdxInPatternMap(i) = mod(i-7680, 15);
        srsRntiStartMap(i) = 79;
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {15}, SRS in U2 slot
    for i = 7740:7754
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11          {}       {84,82}      num2cell(2249:2256)   {37}};
        CellIdxInPatternMap(i) = mod(i-7680, 15);
        srsRntiStartMap(i) = 91;
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {23}, SRS in S slot (2 symbols)
    for i = 7755:7769
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {26}};
        CellIdxInPatternMap(i) = mod(i-7680, 15);
        srsRntiStartMap(i) = 103;
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33}, SRS in S slot (2 symbols)
    for i = 7770:7784
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {26}};
        CellIdxInPatternMap(i) = mod(i-7680, 15);
        srsRntiStartMap(i) = 127;
    end
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % slot {43,44,45,53,54,55,63,64,65,73,74,75} is the same with {* % 40}
    % end pattern 83b: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR 25-3 column G TC, 15C, 64 UEs per TTI, SRS in S slot (2 symbols), U slot, and U2 slot, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 8 UEG, 8 UEs per UEG, 1 layer per UE

    %% pattern 85: 64TR peak, 100 MHz (273 PRBs), OTA, 25-3 column E TC, 15C, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEGs, 8 UEs per slot, 1 layer per UE, 32 DL layer with more PUCCH, SRS in S slot (2 symbols)
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}
    for i = 7785:7799
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {91,92}   num2cell(2257:2264)   {}};
        CellIdxInPatternMap(i) = mod(i-7785, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {5}, prach
    for i = 7800:7814
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {75}       {93,94}   num2cell(2265:2272)   {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-7785, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {13}
    % for i = 7815:7829
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {26}};
    %     CellIdxInPatternMap(i) = mod(i-7785, 15);
    % end
    % slot {14}
    for i = 7830:7844
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {95,96,97}   num2cell(2273:2280)   {}};
        CellIdxInPatternMap(i) = mod(i-7785, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {15}
    for i = 7845:7859
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11          {}       {98,94}      num2cell(2281:2288)   {}};
        CellIdxInPatternMap(i) = mod(i-7785, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {23}
    for i = 7860:7874
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {31}};
        CellIdxInPatternMap(i) = mod(i-7785, 15);
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33}
    for i = 7875:7889
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {32}};
        CellIdxInPatternMap(i) = mod(i-7785, 15);
    end
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % slot {43,44,45,53,54,55,63,64,65,73,74,75} is the same with {* % 40}
    % end pattern 85: 64TR peak, 100 MHz (273 PRBs), OTA, 25-3 column E TC, 15C, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEGs, 8 UEs per slot, 1 layer per UE, 32 DL layer with more PUCCH, SRS in S slot (2 symbols)
    
    % pattern 87 reuse the ULMIX TVs from pattern 85

    %% pattern 89: 64TR peak, Mixed 100 MHz (273 PRBs) + 90 MHz (244 PRBs) + 60 MHz (160 PRBs), OTA, realistic traffic, 9C, PUSCH CP-OFDM, Mixed PUCCH F1/F3. Based on 90655,90656,90658~90661
                                    % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    
    % slot {4}, PUCCH + PUSCH set 0
    for i = 7890:7898
        row_count = row_count + 1;
        cellIdx = mod(i-7890, 9);
        switch cellIdx
            case 0  % 100 MHz heavy
                CFG(row_count, 1:7) = { i    0         11        {}         {81,82}   num2cell(847:866)     {}};
            case 1  % 100 MHz light
                CFG(row_count, 1:7) = { i    0         11        {}         {81,82}   num2cell(888:899)     {}};
            case 2  % 100 MHz light
                CFG(row_count, 1:7) = { i    0         11        {}         {81,82}   num2cell(888:899)     {}};
            case 3  % 90 MHz heavy
                CFG(row_count, 1:7) = { i    0         14        {}         {81,82}   num2cell(913:932)     {}};
            case 4  % 90 MHz light
                CFG(row_count, 1:7) = { i    0         14        {}         {81,82}   num2cell(954:965)     {}};
            case 5  % 90 MHz light
                CFG(row_count, 1:7) = { i    0         14        {}         {81,82}   num2cell(954:965)     {}};
            case 6  % 60 MHz heavy
                CFG(row_count, 1:7) = { i    0         15        {}         {81,82}   num2cell(979:998)     {}};
            case 7  % 60 MHz light
                CFG(row_count, 1:7) = { i    0         15        {}         {81,82}   num2cell(1020:1031)   {}};
            case 8  % 60 MHz light
                CFG(row_count, 1:7) = { i    0         15        {}         {81,82}   num2cell(1020:1031)   {}};
        end
        CellIdxInPatternMap(i) = cellIdx;
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_18_7";
    end
    
    % slot {5}, PRACH + PUCCH + PUSCH set 1
    for i = 7899:7907
        row_count = row_count + 1;
        cellIdx = mod(i-7890, 9);
        switch cellIdx
            case 0  % 100 MHz heavy
                CFG(row_count, 1:7) = { i    0         11        {75}       {78,83}   num2cell(867:886)     {}};
            case 1  % 100 MHz light
                CFG(row_count, 1:7) = { i    0         11        {75}       {78,83}   num2cell(900:911)     {}};
            case 2  % 100 MHz light
                CFG(row_count, 1:7) = { i    0         11        {75}       {78,83}   num2cell(900:911)     {}};
            case 3  % 90 MHz heavy
                CFG(row_count, 1:7) = { i    0         14        {80}       {78,83}   num2cell(933:952)     {}};
            case 4  % 90 MHz light
                CFG(row_count, 1:7) = { i    0         14        {80}       {78,83}   num2cell(966:977)     {}};
            case 5  % 90 MHz light
                CFG(row_count, 1:7) = { i    0         14        {80}       {78,83}   num2cell(966:977)     {}};
            case 6  % 60 MHz heavy
                CFG(row_count, 1:7) = { i    0         15        {81}       {78,83}   num2cell(999:1018)    {}};
            case 7  % 60 MHz light
                CFG(row_count, 1:7) = { i    0         15        {81}       {78,83}   num2cell(1032:1043)   {}};
            case 8  % 60 MHz light
                CFG(row_count, 1:7) = { i    0         15        {81}       {78,83}   num2cell(1032:1043)   {}};
        end
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = cellIdx;
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_18_7";
    end

    % slot {13}, SRS only
    % for i = 7908:7916
    %     row_count = row_count + 1;
    %     cellIdx = mod(i-7890, 9);
    %     switch cellIdx
    %         case 0  % 100 MHz heavy
    %             CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {24}};
    %         case 1  % 100 MHz light
    %             CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {24}};
    %         case 2  % 100 MHz light
    %             CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {24}};
    %         case 3  % 90 MHz heavy
    %             CFG(row_count, 1:7) = { i    0         14        {}         {}   {}   {39}};
    %         case 4  % 90 MHz light
    %             CFG(row_count, 1:7) = { i    0         14        {}         {}   {}   {39}};
    %         case 5  % 90 MHz light
    %             CFG(row_count, 1:7) = { i    0         14        {}         {}   {}   {39}};
    %         case 6  % 60 MHz heavy
    %             CFG(row_count, 1:7) = { i    0         15        {}         {}   {}   {40}};
    %         case 7  % 60 MHz light
    %             CFG(row_count, 1:7) = { i    0         15        {}         {}   {}   {40}};
    %         case 8  % 60 MHz light
    %             CFG(row_count, 1:7) = { i    0         15        {}         {}   {}   {40}};
    %     end
    %     CellIdxInPatternMap(i) = cellIdx;
    %     srsRntiStartMap(int32(i)) = 31;  % slot 13 SRS RNTI start
    % end
    
    % slot {14}, PUCCH + PUSCH set 0
    for i = 7917:7925
        row_count = row_count + 1;
        cellIdx = mod(i-7890, 9);
        switch cellIdx
            case 0  % 100 MHz heavy
                CFG(row_count, 1:7) = { i    0         11        {}         {80,79,82}   num2cell(847:866)     {}};
            case 1  % 100 MHz light
                CFG(row_count, 1:7) = { i    0         11        {}         {80,79,82}   num2cell(888:899)     {}};
            case 2  % 100 MHz light
                CFG(row_count, 1:7) = { i    0         11        {}         {80,79,82}   num2cell(888:899)     {}};
            case 3  % 90 MHz heavy
                CFG(row_count, 1:7) = { i    0         14        {}         {80,79,82}   num2cell(913:932)     {}};
            case 4  % 90 MHz light
                CFG(row_count, 1:7) = { i    0         14        {}         {80,79,82}   num2cell(954:965)     {}};
            case 5  % 90 MHz light
                CFG(row_count, 1:7) = { i    0         14        {}         {80,79,82}   num2cell(954:965)     {}};
            case 6  % 60 MHz heavy
                CFG(row_count, 1:7) = { i    0         15        {}         {80,79,82}   num2cell(979:998)     {}};
            case 7  % 60 MHz light
                CFG(row_count, 1:7) = { i    0         15        {}         {80,79,82}   num2cell(1020:1031)   {}};
            case 8  % 60 MHz light
                CFG(row_count, 1:7) = { i    0         15        {}         {80,79,82}   num2cell(1020:1031)   {}};
        end
        CellIdxInPatternMap(i) = cellIdx;
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_18_7";
    end
    
    % slot {15}, PUCCH + PUSCH set 2 (full BW)
    for i = 7926:7934
        row_count = row_count + 1;
        cellIdx = mod(i-7890, 9);
        switch cellIdx
            case 0  % 100 MHz heavy
                CFG(row_count, 1:7) = { i    0         11          {}       {77,83}      num2cell([867:885, 887])   {}};
            case 1  % 100 MHz light
                CFG(row_count, 1:7) = { i    0         11          {}       {77,83}      num2cell([900:910, 912])   {}};
            case 2  % 100 MHz light
                CFG(row_count, 1:7) = { i    0         11          {}       {77,83}      num2cell([900:910, 912])   {}};
            case 3  % 90 MHz heavy
                CFG(row_count, 1:7) = { i    0         14          {}       {77,83}      num2cell([933:951, 953])   {}};
            case 4  % 90 MHz light
                CFG(row_count, 1:7) = { i    0         14          {}       {77,83}      num2cell([966:976, 978])   {}};
            case 5  % 90 MHz light
                CFG(row_count, 1:7) = { i    0         14          {}       {77,83}      num2cell([966:976, 978])   {}};
            case 6  % 60 MHz heavy
                CFG(row_count, 1:7) = { i    0         15          {}       {77,83}      num2cell([999:1017, 1019]) {}};
            case 7  % 60 MHz light
                CFG(row_count, 1:7) = { i    0         15          {}       {77,83}      num2cell([1032:1042, 1044]) {}};
            case 8  % 60 MHz light
                CFG(row_count, 1:7) = { i    0         15          {}       {77,83}      num2cell([1032:1042, 1044]) {}};
        end
        CellIdxInPatternMap(i) = cellIdx;
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_18_7";
    end
    
    % slot {23}, SRS only
    for i = 7935:7943
        row_count = row_count + 1;
        cellIdx = mod(i-7890, 9);
        switch cellIdx
            case 0  % 100 MHz heavy
                CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {24}};
            case 1  % 100 MHz light
                CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {24}};
            case 2  % 100 MHz light
                CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {24}};
            case 3  % 90 MHz heavy
                CFG(row_count, 1:7) = { i    0         14        {}         {}   {}   {39}};
            case 4  % 90 MHz light
                CFG(row_count, 1:7) = { i    0         14        {}         {}   {}   {39}};
            case 5  % 90 MHz light
                CFG(row_count, 1:7) = { i    0         14        {}         {}   {}   {39}};
            case 6  % 60 MHz heavy
                CFG(row_count, 1:7) = { i    0         15        {}         {}   {}   {40}};
            case 7  % 60 MHz light
                CFG(row_count, 1:7) = { i    0         15        {}         {}   {}   {40}};
            case 8  % 60 MHz light
                CFG(row_count, 1:7) = { i    0         15        {}         {}   {}   {40}};
        end
        CellIdxInPatternMap(i) = cellIdx;
        srsRntiStartMap(int32(i)) = 55;  % slot 23 SRS RNTI start
    end
    
    % slot {33}, SRS only
    for i = 7944:7952
        row_count = row_count + 1;
        cellIdx = mod(i-7890, 9);
        switch cellIdx
            case 0  % 100 MHz heavy
                CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {24}};
            case 1  % 100 MHz light
                CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {24}};
            case 2  % 100 MHz light
                CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {24}};
            case 3  % 90 MHz heavy
                CFG(row_count, 1:7) = { i    0         14        {}         {}   {}   {39}};
            case 4  % 90 MHz light
                CFG(row_count, 1:7) = { i    0         14        {}         {}   {}   {39}};
            case 5  % 90 MHz light
                CFG(row_count, 1:7) = { i    0         14        {}         {}   {}   {39}};
            case 6  % 60 MHz heavy
                CFG(row_count, 1:7) = { i    0         15        {}         {}   {}   {40}};
            case 7  % 60 MHz light
                CFG(row_count, 1:7) = { i    0         15        {}         {}   {}   {40}};
            case 8  % 60 MHz light
                CFG(row_count, 1:7) = { i    0         15        {}         {}   {}   {40}};
        end
        CellIdxInPatternMap(i) = cellIdx;
        srsRntiStartMap(int32(i)) = 79;  % slot 33 SRS RNTI start
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % slot {43,44,45,53,54,55,63,64,65,73,74,75} is the same with {* % 40}
    % end pattern 89: 64TR peak, Mixed 100 MHz (273 PRBs) + 90 MHz (244 PRBs) + 60 MHz (160 PRBs), OTA, realistic traffic, 9C, PUSCH CP-OFDM, Mixed PUCCH F1/F3. Based on 90655,90656,90658~90661
    
    %% pattern 79a: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case Ph4 column B TC, 15C, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEG, 8 UEs per UEG, 1 layer per UE, 4 SRS symbols in S slot, 2 SRS symbols in each U slot, 2 ports per UE
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {3} is S slot for BFW binding, already defined as TC 21640
    % slot {4}, U slot
    for i = 7953:7967
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {81,82}   num2cell(2345:2352)   {31}};
        CellIdxInPatternMap(i) = mod(i-7953, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_18_7";
        srsRntiStartMap(i) = 55;
    end
    % slot {5}, U slot, prach
    for i = 7968:7982
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {75}       {78,83}   num2cell(2353:2360)   {31}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-7953, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_18_7";
        srsRntiStartMap(i) = 79;
    end
    % slot {13}, S slot, SRS only
    for i = 7983:7997
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {36}};
        CellIdxInPatternMap(i) = mod(i-7953, 15);
        srsRntiStartMap(i) = 103;
    end
    % slot {14}, U slot
    for i = 7998:8012
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {80,79,82}   num2cell(2345:2352)   {31}};
        CellIdxInPatternMap(i) = mod(i-7953, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_18_7";
        srsRntiStartMap(i) = 151;
    end
    % slot {15}, U slot
    for i = 8013:8027
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11          {}       {77,83}      num2cell(2361:2368)   {31}};
        CellIdxInPatternMap(i) = mod(i-7953, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_18_7";
        srsRntiStartMap(i) = 175;
    end
    % slot {23}, S slot, SRS only
    for i = 8028:8042
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {36}};
        CellIdxInPatternMap(i) = mod(i-7953, 15);
        srsRntiStartMap(i) = 199;
    end
    % slot {33}, S slot, SRS only
    for i = 8043:8057
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {36}};
        CellIdxInPatternMap(i) = mod(i-7953, 15);
        srsRntiStartMap(i) = 247;
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % slot {43,44,45,53,54,55,63,64,65,73,74,75} is the same with {* % 40}
    % end pattern 79a: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case Ph4 column B TC, 15C, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEG, 8 UEs per UEG, 1 layer per UE, 4 SRS symbols in S slot, 2 SRS symbols in each U slot, 2 ports per UE

    %% pattern 79b: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case Ph4 column B TC, 15C, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEG, 8 UEs per UEG, 1 layer per UE, 4 SRS symbols in S slot, 2 SRS symbols in each U slot, 4 ports per UE
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {3} is S slot for BFW binding, already defined as TC 21641
    % slot {4}, U slot
    for i = 8058:8072
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {81,82}   num2cell(2345:2352)   {42}};
        CellIdxInPatternMap(i) = mod(i-8058, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_18_7";
        srsRntiStartMap(i) = 31;
    end
    % slot {5}, U slot, prach
    for i = 8073:8087
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {75}       {78,83}   num2cell(2353:2360)   {42}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-8058, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_18_7";
        srsRntiStartMap(i) = 43;
    end
    % slot {13}, S slot, SRS only
    for i = 8088:8102
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {41}};
        CellIdxInPatternMap(i) = mod(i-8058, 15);
        srsRntiStartMap(i) = 55;
    end
    % slot {14}, U slot
    for i = 8103:8117
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {80,79,82}   num2cell(2345:2352)   {42}};
        CellIdxInPatternMap(i) = mod(i-8058, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_18_7";
        srsRntiStartMap(i) = 79;
    end
    % slot {15}, U slot
    for i = 8118:8132
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11          {}       {77,83}      num2cell(2361:2368)   {42}};
        CellIdxInPatternMap(i) = mod(i-8058, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_18_7";
        srsRntiStartMap(i) = 91;
    end
    % slot {23}, S slot, SRS only
    for i = 8133:8147
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {41}};
        CellIdxInPatternMap(i) = mod(i-8058, 15);
        srsRntiStartMap(i) = 103;
    end
    % slot {33}, S slot, SRS only
    for i = 8148:8162
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {41}};
        CellIdxInPatternMap(i) = mod(i-8058, 15);
        srsRntiStartMap(i) = 127;
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % slot {43,44,45,53,54,55,63,64,65,73,74,75} is the same with {* % 40}
    % end pattern 79b: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case Ph4 column B TC, 15C, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEG, 8 UEs per UEG, 1 layer per UE, 4 SRS symbols in S slot, 2 SRS symbols in each U slot, 4 ports per UE

    %% pattern 91: 64TR peak, 100 MHz (273 PRBs), OTA, 25-3 column D TC, 15C, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEGs, 8 UEs per slot, 1 layer per UE, 24 DL layer with more PUCCH, SRS in S slot (2 symbols)
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}
    for i = 8163:8177
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {99,100}   num2cell(2313:2320)   {}};
        CellIdxInPatternMap(i) = mod(i-8163, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {5}, prach
    for i = 8178:8192
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {75}       {101,102}   num2cell(2321:2328)   {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-8163, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {13}
    % for i = 8193:8207
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {26}};
    %     CellIdxInPatternMap(i) = mod(i-8163, 15);
    % end
    % slot {14}
    for i = 8208:8222
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {103,104,105}   num2cell(2329:2336)   {}};
        CellIdxInPatternMap(i) = mod(i-8163, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {15}
    for i = 8223:8237
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11          {}       {106,102}      num2cell(2337:2344)   {}};
        CellIdxInPatternMap(i) = mod(i-8163, 15);
        puschDynamicBfMap(i) = 1;
        enableUlRxBfMap(i) = 1;
        harqProcessIdMap(i) = 1;
        UciOnPuschMap(i) = "64TR_4_13_10";
    end
    % slot {23}
    for i = 8238:8252
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {31}};
        CellIdxInPatternMap(i) = mod(i-8163, 15);
    end
    % slot {24} is the same with {4}
    % slot {25} is the same with {5}
    % slot {33}
    for i = 8253:8267
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         11        {}         {}   {}   {32}};
        CellIdxInPatternMap(i) = mod(i-8163, 15);
    end
    % slot {34} is the same with {14}
    % slot {35} is the same with {15}
    % slot {43,44,45,53,54,55,63,64,65,73,74,75} is the same with {* % 40}
    % end pattern 91: 64TR peak, 100 MHz (273 PRBs), OTA, 25-3 column D TC, 15C, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEGs, 8 UEs per slot, 1 layer per UE, 24 DL layer with more PUCCH, SRS in S slot (2 symbols)

    %% pattern 101a: 1 UE SU-MIMO, 2 layers, MCS 27, 256QAM table, PUSCH + PRACH + UCI-on-PUSCH, 1 pilot @ 3rd symbol, type-A, cdm group 2, 13 data symbols, 10 fixed LDPC iterations, 24C
    % PUSCH: cfg 1045 (slot 4), cfg 1046 (slot 5 with PRACH), cfg 1047 (slot 14), cfg 1048 (slot 15)
    % PRACH: cfg 82
    % UCI-on-PUSCH: "7beams_4_37_5"
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}
    for i = 8268:8291
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0          2        {}         {}        {1045}     {}};
        CellIdxInPatternMap(i) = mod(i-8268, 24);
        enableOtaMap(i) = 1;
        ldpcFixMaxIterMap(int32(i)) = 10;
        UciOnPuschMap(i) = "7beams_4_37_5";
    end
    % slot {5}, prach
    for i = 8292:8315
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0          2        {82}       {}        {1046}     {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-8268, 24);
        enableOtaMap(i) = 1;
        ldpcFixMaxIterMap(int32(i)) = 10;
        UciOnPuschMap(i) = "7beams_4_37_5";
    end
    % slot {14}
    for i = 8316:8339
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0          2        {}         {}        {1047}     {}};
        CellIdxInPatternMap(i) = mod(i-8268, 24);
        enableOtaMap(i) = 1;
        ldpcFixMaxIterMap(int32(i)) = 10;
        UciOnPuschMap(i) = "7beams_4_37_5";
    end
    % slot {15}
    for i = 8340:8363
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0          2        {}         {}        {1048}     {}};
        CellIdxInPatternMap(i) = mod(i-8268, 24);
        enableOtaMap(i) = 1;
        ldpcFixMaxIterMap(int32(i)) = 10;
        UciOnPuschMap(i) = "7beams_4_37_5";
    end
    % end pattern 101a: 1 UE SU-MIMO, 2 layers, MCS 27, 256QAM table, PUSCH + PRACH + UCI-on-PUSCH, 24C


    %% pattern 102: 1 UE SU-MIMO, 2 layers, MCS 27, 256QAM table, PUSCH (reduced PRB) + PRACH, 1 pilot @ 3rd symbol, type-A, cdm group 2, 13 data symbols, 10 fixed LDPC iterations, 24C
    % PUSCH: cfg 1049 (slot 4), cfg 1050 (slot 5 with PRACH), cfg 1051 (slot 14), cfg 1052 (slot 15)
    % PRACH: cfg 83
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}
    for i = 8364:8387
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0          2        {}         {}        {1049}     {}};
        CellIdxInPatternMap(i) = mod(i-8364, 24);
        enableOtaMap(i) = 1;
        ldpcFixMaxIterMap(int32(i)) = 10;
    end
    % slot {5}, prach
    for i = 8388:8411
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0          2        {83}       {}        {1050}     {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-8364, 24);
        enableOtaMap(i) = 1;
        ldpcFixMaxIterMap(int32(i)) = 10;
    end
    % slot {14}
    for i = 8412:8435
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0          2        {}         {}        {1051}     {}};
        CellIdxInPatternMap(i) = mod(i-8364, 24);
        enableOtaMap(i) = 1;
        ldpcFixMaxIterMap(int32(i)) = 10;
    end
    % slot {15}
    for i = 8436:8459
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0          2        {}         {}        {1052}     {}};
        CellIdxInPatternMap(i) = mod(i-8364, 24);
        enableOtaMap(i) = 1;
        ldpcFixMaxIterMap(int32(i)) = 10;
    end
    % end pattern 102: 1 UE SU-MIMO, 2 layers, MCS 27, 256QAM table, PUSCH (reduced PRB) + PRACH, 24C
    
    %% pattern 102a: 1 UE SU-MIMO, 2 layers, MCS 27, 256QAM table, PUSCH (reduced PRB) + PRACH + UCI-on-PUSCH, 1 pilot @ 3rd symbol, type-A, cdm group 2, 13 data symbols, 10 fixed LDPC iterations, 24C
    % PUSCH: cfg 1049 (slot 4), cfg 1050 (slot 5 with PRACH), cfg 1051 (slot 14), cfg 1052 (slot 15)
    % PRACH: cfg 83
    % UCI-on-PUSCH: "7beams_4_37_5"
                            % TC#   slotIdx   cell     prach     pucch      pusch      srs
    row_count = length(CFG);
    % slot {4}
    for i = 8460:8483
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0          2        {}         {}        {1049}     {}};
        CellIdxInPatternMap(i) = mod(i-8460, 24);
        enableOtaMap(i) = 1;
        ldpcFixMaxIterMap(int32(i)) = 10;
        UciOnPuschMap(i) = "7beams_4_37_5";
    end
    % slot {5}, prach
    for i = 8484:8507
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0          2        {83}       {}        {1050}     {}};
        PrachBeamIdxMap(i) = [0 1];
        CellIdxInPatternMap(i) = mod(i-8460, 24);
        enableOtaMap(i) = 1;
        ldpcFixMaxIterMap(int32(i)) = 10;
        UciOnPuschMap(i) = "7beams_4_37_5";
    end
    % slot {14}
    for i = 8508:8531
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0          2        {}         {}        {1051}     {}};
        CellIdxInPatternMap(i) = mod(i-8460, 24);
        enableOtaMap(i) = 1;
        ldpcFixMaxIterMap(int32(i)) = 10;
        UciOnPuschMap(i) = "7beams_4_37_5";
    end
    % slot {15}
    for i = 8532:8555
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0          2        {}         {}        {1052}     {}};
        CellIdxInPatternMap(i) = mod(i-8460, 24);
        enableOtaMap(i) = 1;
        ldpcFixMaxIterMap(int32(i)) = 10;
        UciOnPuschMap(i) = "7beams_4_37_5";
    end
    % end pattern 102a: 1 UE SU-MIMO, 2 layers, MCS 27, 256QAM table, PUSCH (reduced PRB) + PRACH + UCI-on-PUSCH, 24C
                            
    % NOTE: use getLargestTvNum(CFG, threshold) to see the current largest ULMIX TV numbers for <threshold and >= threshold
    %       Search for "available for future ULMIX TVs" to find the avaiable ULMIX TV numbers smaller than that
    largestTvNum = getLargestTvNum(CFG, 20000);

CFG_Cell = {...
 % mu N_grid_size_mu N_ID_CELL Nant_gNB Nant_UE   idx
    1      273            41        4       1;    % 1
    % perf test F08
    1      273            41        4       2;    % 2 
    % F09
    1      273            41        8       2;    % 3
    % 40 MHz
    1      106            61        4       2;    % 4
    % 80 MHz
    1      217            81        4       2;    % 5
    % 30 MHz
    1      78             41        4       2;    % 6
    % 50 MHz
    1      133            41        4       2;    % 7
    % 32T32R
    1      273            41       32       2;    % 8
    % HARQ TC
    1      273            41        4       2;    % 9
    % single UE 4T4R
    1      273            41        4       4;    % 10
    % 64TR
    1      273            41       16       4;    % 11    
    % 64TR perf % note: now it means 16 UL streams for PUSCH. To update after PUSCH/SRS Rx buffer separated
    1      273            41       16       4;    % 12
    % 64TR 40 MHz
    1      106            41       16       4;    % 13
    % 64TR 90 MHz
    1      244            41       16       4;    % 14
    % 64TR 60 MHz
    1      160            41       16       4;    % 15
    };

CFG_PRACH = {...
 % cfg#   duplex  mu  cfg restrictSet root  zone  prmbIdx  RA0
    1,        1,  1, 155,     0,        0,    5,     2,    132;  % format B4
    
    % perf test suppose F08 F09 both have one preamble
    2,        1,  1, 158,     0,        0,    5,     2,    261;
    % 4 prach occasions
    3,        1,  1, 158,     0,        0,    5,     0,    225;
    4,        1,  1, 158,     0,        0,    5,     1,    237;
    5,        1,  1, 158,     0,        0,    5,     2,    249;
    6,        1,  1, 158,     0,        0,    5,     3,    261;
    % 4 prach occasions, starting at 85
    7,        1,  1, 158,     0,        0,    5,     0,    85;
    8,        1,  1, 158,     0,        0,    5,     1,    97;
    9,        1,  1, 158,     0,        0,    5,     2,    109;
    10,       1,  1, 158,     0,        0,    5,     3,    121;
    11,       1,  1, 158,     0,        3,    7,     4,    97;
    % 32T32R cases
    12,       1,  1, 155,     0,        0,    5,     2,     0;
    % 4 prach occasions, starting at 55
    13,       1,  1, 158,     0,        0,    5,     0,    55;
    14,       1,  1, 158,     0,        0,    5,     1,    67;
    15,       1,  1, 158,     0,        0,    5,     2,    79;
    16,       1,  1, 158,     0,        0,    5,     3,    91;
    % 4 prach occasions, starting at 1
    17,        1,  1, 158,     0,        0,    5,     0,    1;
    18,        1,  1, 158,     0,        0,    5,     1,    13;
    19,        1,  1, 158,     0,        0,    5,     2,    25;
    20,        1,  1, 158,     0,        0,    5,     3,    37;
    % 4 prach occasions, starting at 169
    21,       1,  1, 158,     0,        0,    5,     0,    169;
    22,       1,  1, 158,     0,        0,    5,     1,    181;
    23,       1,  1, 158,     0,        0,    5,     2,    193;
    24,       1,  1, 158,     0,        0,    5,     3,    205;
    % 4 prach occasions, starting at 55
    25,       1,  1, 158,     0,        0,    5,     0,    55;
    26,       1,  1, 158,     0,        0,    5,     1,    67;
    27,       1,  1, 158,     0,        0,    5,     2,    79;
    28,       1,  1, 158,     0,        0,    5,     3,    91;
    % 4 prach occasions, starting at 210
    29,        1,  1, 158,     0,        0,    5,     0,    210;
    30,        1,  1, 158,     0,        0,    5,     1,    210+12;
    31,        1,  1, 158,     0,        0,    5,     2,    210+24;
    32,        1,  1, 158,     0,        0,    5,     3,    210+36;
    % 4 prach occasions, starting at 78
    33,        1,  1, 158,     0,        0,    5,     0,    78;
    34,        1,  1, 158,     0,        0,    5,     1,    78+12;
    35,        1,  1, 158,     0,        0,    5,     2,    78+24;
    36,        1,  1, 158,     0,        0,    5,     3,    78+36;
    %%%%%    
    % 4 prach occasions, starting at 25
    37,        1,  1, 158,     0,        0,    5,     0,    25;
    38,        1,  1, 158,     0,        0,    5,     1,    25+12;
    39,        1,  1, 158,     0,        0,    5,     2,    25+24;
    40,        1,  1, 158,     0,        0,    5,     3,    25+36;
    %%%%%    
    % HARQ TC
 % cfg#   duplex  mu  cfg restrictSet root  zone  prmbIdx  RA0
    41,        1,  1, 158,     0,        0,   14,     0,    60;
    42,        1,  1, 158,     0,        0,   14,     1,    60+12;
    43,        1,  1, 158,     0,        0,   14,     2,    60+24;
    44,        1,  1, 158,     0,        0,   14,     3,    60+36;
    % 4 prach occasions, starting at 222
    45,        1,  1, 158,     0,        0,    5,     0,    222;
    46,        1,  1, 158,     0,        0,    5,     1,    222+12;
    47,        1,  1, 158,     0,        0,    5,     2,    222+24;
    48,        1,  1, 158,     0,        0,    5,     3,    222+36;
    % 4 prach occasions, starting at 84
    49,        1,  1, 158,     0,        0,    5,     0,    84;
    50,        1,  1, 158,     0,        0,    5,     1,    84+12;
    51,        1,  1, 158,     0,        0,    5,     2,    84+24;
    52,        1,  1, 158,     0,        0,    5,     3,    84+36;
    %%%%%    
    % 4 prach occasions, starting at 224
    53,        1,  1, 158,     0,        0,    5,     0,    224;
    54,        1,  1, 158,     0,        0,    5,     1,    224+12;
    55,        1,  1, 158,     0,        0,    5,     2,    224+24;
    56,        1,  1, 158,     0,        0,    5,     3,    224+36;
    % 4 prach occasions, starting at 86
    57,        1,  1, 158,     0,        0,    5,     0,    86;
    58,        1,  1, 158,     0,        0,    5,     1,    86+12;
    59,        1,  1, 158,     0,        0,    5,     2,    86+24;
    60,        1,  1, 158,     0,        0,    5,     3,    86+36;
    % 64TR
    61,        1,  1, 158,     0,        0,   14,     0,    4;
    62,        1,  1, 158,     0,        0,   14,     1,    4+12;
    63,        1,  1, 158,     0,        0,   14,     2,    4+24;
    64,        1,  1, 158,     0,        0,   14,     3,    4+36;    
    %%%%%    
    % 4 prach occasions, starting at 88
    65,        1,  1, 158,     0,        0,    5,     0,    88;
    66,        1,  1, 158,     0,        0,    5,     1,    88+12;
    67,        1,  1, 158,     0,        0,    5,     2,    88+24;
    68,        1,  1, 158,     0,        0,    5,     3,    88+36;
    % 64TR test case
    69,        1,  1, 158,     0,        0,   14,     0,    260;
    % 64TR 'worst case' column G test case
    70,        1,  1, 158,     0,        0,   14,     0,    225;
    71,        1,  1, 158,     0,        0,   14,     1,    225+12;
    72,        1,  1, 158,     0,        0,   14,     2,    225+24;
    73,        1,  1, 158,     0,        0,   14,     3,    225+36;
    % 64TR test case
    74,        1,  1, 158,     0,        0,   14,     0,    124;
    % 64TR test case
    75,        1,  1, 158,     0,        0,   14,     0,    261;
    % 64TR 'worst case' column G test case, 40 MHz
    76,        1,  1, 158,     0,        0,   14,     0,    58;
    77,        1,  1, 158,     0,        0,   14,     1,    70;
    78,        1,  1, 158,     0,        0,   14,     2,    82;
    79,        1,  1, 158,     0,        0,   14,     3,    94;
    % 64TR realistic traffic with Center/Middle/Edge, 90 MHz heavy / light
    80,        1,  1, 158,     0,        0,   14,     0,   232;
    % 64TR realistic traffic with Center/Middle/Edge, 60 MHz heavy / light
    81,        1,  1, 158,     0,        0,   14,     0,   148;
    % 4TR one PRACH occasion at PRB 261
    82,        1,  1, 158,     0,        0,    5,     0,   261;
    % 4TR one PRACH occasion at PRB 42
    83,        1,  1, 158,     0,        0,    5,     0,   42;
    };

CFG_PUCCH = {...
% cfg# N_UE    Nb format   f0      f1  Nf t0  Nt cs0  tOCCidx freqH groupH SRFlag posSR DTX thr
    1    2     2    1      0        0   1 10   4   0   [0 1]     0      0      0     0    0   1;
    2    2     2    1     272     272   1 10   4   0   [0 1]     0      0      0     1    0   1;
    3    2     2    1      0        0   1  0  14   0   [0 1]     0      0      0     0    0   1;
    4    2     2    1     272     272   1  0  14   0   [0 1]     0      0      0     1    0   1;
    
    % perf test F08
    5    16    17   3    [0:15]     0   1  0   7   0     0       0      0      0     0    0   1;
    % F09
    6    8     17   3    [0:7]      0   1  0   7   0     0       0      0      0     0    0   1;      
    % F08 relaxed
    7   24     1    1     0         0   1  3   9 ...
        [0 0 0 0 2 2 2 2 4 4 4 4 6 6 6 6 8 8 8 8 10 10 10 10] ...%cs0
        [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0  1  2  3 ] ...%tOCCidx
         0      0      0     0    0   1;
    8  24    1   1    [0 1 2 3 4 5 6 7 8 9 10 11 261 262 263 264 265 266 267 268 269 270 271 272]    0  1 3  9 ...
        0 ...%cs0
        0 ...%tOCCidx
         0     0     0    0   0  1;
    9  12    1   1    [0 1 2 3 4 5 126 127 128 129 130 131]      0  1 3  9 ...
        0 ...%cs0
        0 ...%tOCCidx
         0     0     0    0   0  1;
    % HARQ TC
% cfg# N_UE    Nb format   f0      f1  Nf t0  Nt cs0  tOCCidx freqH groupH SRFlag posSR DTX thr
   10    1     2    1      0        0   1 10   4   0        1     0      0      0     0    0   1;
   11    1     2    1      1        0   1 10   4   0        1     0      0      0     0    0   1;
   12    1     2    1      2        0   1 10   4   0        1     0      0      0     0    0   1;
   13    1     2    1      3        0   1 10   4   0        1     0      0      0     0    0   1;
   14    1     2    1      4        0   1 10   4   0        1     0      0      0     0    0   1;
   15    1     2    1      5        0   1 10   4   0        1     0      0      0     0    0   1;
   16    1     2    1      6        0   1 10   4   0        1     0      0      0     0    0   1;
   17    1     2    1      7        0   1 10   4   0        1     0      0      0     0    0   1;
   18    1     2    1      8        0   1 10   4   0        1     0      0      0     0    0   1;
   19    1     2    1      9        0   1 10   4   0        1     0      0      0     0    0   1;
   20    1     2    1     10        0   1 10   4   0        1     0      0      0     0    0   1;
   21    1     2    1     11        0   1 10   4   0        1     0      0      0     0    0   1;
   22    1     2    1    261        0   1 10   4   0        1     0      0      0     0    0   1;
   23    1     2    1    262        0   1 10   4   0        1     0      0      0     0    0   1;
   24    1     2    1    263        0   1 10   4   0        1     0      0      0     0    0   1;
   25    1     2    1    264        0   1 10   4   0        1     0      0      0     0    0   1;
   26    1     2    1    265        0   1 10   4   0        1     0      0      0     0    0   1;
   27    1     2    1    266        0   1 10   4   0        1     0      0      0     0    0   1;
   28    1     2    1    267        0   1 10   4   0        1     0      0      0     0    0   1;
   29    1     2    1    268        0   1 10   4   0        1     0      0      0     0    0   1;
   30    1     2    1    269        0   1 10   4   0        1     0      0      0     0    0   1;
   31    1     2    1    270        0   1 10   4   0        1     0      0      0     0    0   1;
   32    1     2    1    271        0   1 10   4   0        1     0      0      0     0    0   1;
   33    1     2    1    272        0   1 10   4   0        1     0      0      0     0    0   1;
   % HARQ TC
   34    1     3    3      0        0   1  0  14   0        0     0      0      0     0    0   1;
   35    1     3    3      1        0   1  0  14   0        0     0      0      0     0    0   1;
   36    1     3    3      2        0   1  0  14   0        0     0      0      0     0    0   1;
   37    1     3    3      3        0   1  0  14   0        0     0      0      0     0    0   1;
   38    1     3    3    269        0   1  0  14   0        0     0      0      0     0    0   1;
   39    1     3    3    270        0   1  0  14   0        0     0      0      0     0    0   1;
   40    1     3    3    271        0   1  0  14   0        0     0      0      0     0    0   1;
   41    1     3    3    272        0   1  0  14   0        0     0      0      0     0    0   1;
% cfg# N_UE    Nb format   f0      f1  Nf t0  Nt cs0  tOCCidx freqH groupH SRFlag posSR DTX thr  
   42   18     1    1    [0:17]    0  1 0  4 ...
   0 ...%cs0
   0 ...%tOCCidx
   0     0     0    0    0  1;
   43   24     1    1    [0:23]    0  1 0  4 ...
   0 ...%cs0
   0 ...%tOCCidx
   0     0     0    0   0  1;
   44   24     1    1 ...
   [0 0 0 0 0 0 0 0 0 0 0  0  1 1 1 1 1 1 1 1 1 1 1 1] ... % f0
    0   1  0   4 ... % f1 Nf t0 Nt
   [0 0 2 2 4 4 6 6 8 8 10 10 0 0 2 2 4 4 6 6 8 8 10 10] ...%cs0
   [0 1 0 1 0 1 0 1 0 1 0  1  0 1 0 1 0 1 0 1 0 1 0  1 ] ...%tOCCidx
    0      0      0     0    0   1; 
   45    1     0    0     19    1   1 12   1   0        0     0      0      1     0    0   1;
   % 64TR
% 14 symbols
   46    1     3    3      0        0   1  0  14   0        0     0      0      0     0    0   1;
   47    1     3    3      1        0   1  0  14   0        0     0      0      0     0    0   1;
   48    1     3    3      2        0   1  0  14   0        0     0      0      0     0    0   1;
   49    1     3    3      3        0   1  0  14   0        0     0      0      0     0    0   1;
   50    1     3    3    269        0   1  0  14   0        0     0      0      0     0    0   1;
   51    1     3    3    270        0   1  0  14   0        0     0      0      0     0    0   1;
   52    1     3    3    271        0   1  0  14   0        0     0      0      0     0    0   1;
   53    1     3    3    272        0   1  0  14   0        0     0      0      0     0    0   1;   
   % cfg# N_UE    Nb format   f0                f1  Nf t0  Nt cs0  tOCCidx freqH groupH SRFlag posSR DTX thr  
   % 64TR perf
   54      48     1    1    repelem(0:3, 12)     0   1  0   14 ...
   repmat(repelem([0 3 6 9], 3), [1,4]) ...%cs0
   repmat([0 1 2], [1,16]) ...%tOCCidx
   0     0     0    0    0  1;
   % 64TR test case
% cfg# N_UE    Nb format   f0      f1  Nf t0  Nt cs0  tOCCidx freqH groupH SRFlag posSR DTX thr  
   55   32     1    1 ...
   [0 0 0 0 0 0 0 0 0 0 0  0  1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2] ... % f0
    0   1  0  14 ... % f1 Nf t0 Nt
   [0 0 2 2 4 4 6 6 8 8 10 10 0 0 2 2 4 4 6 6 8 8 10 10 0 0 2 2 4 4 6 6] ...%cs0
   [0 1 0 1 0 1 0 1 0 1 0  1  0 1 0 1 0 1 0 1 0 1 0  1 0 1 0 1 0 1 0 1] ...%tOCCidx
    0      0      0     0    0   1;    
   56   16     2    1 ...
   [0 0 0 0 0 0 1 1 1 1 1 1 2 2 2 2] ... % f0
    0   1  0  14 ... % f1 Nf t0 Nt
   [0 0 2 2 4 4 0 0 2 2 4 4 0 0 2 2] ...%cs0
   [0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1] ...%tOCCidx
    0      0      0     0    0   1;    
% cfg# N_UE    Nb format   f0      f1  Nf t0  Nt cs0  tOCCidx freqH groupH SRFlag posSR DTX thr  
   57    8      4   3    [0:7]     0   1  0   14   0     0       0      0      0     0    0   1;
   58   10      3   3    [0:9]     0   1  0   14   0     0       0      0      0     0    0   1;
   59    2      2   1 [10, 10]     0   1  0   14  [0 0] [0 1]    0      0      0     0    0   1;    

% 12 symbols
    60    1     3    3      0        0   1  0  12   0        0     0      0      0     0    0   1;
    61    1     3    3      1        0   1  0  12   0        0     0      0      0     0    0   1;
    62    1     3    3      2        0   1  0  12   0        0     0      0      0     0    0   1;
    63    1     3    3      3        0   1  0  12   0        0     0      0      0     0    0   1;
    64    1     3    3    269        0   1  0  12   0        0     0      0      0     0    0   1;
    65    1     3    3    270        0   1  0  12   0        0     0      0      0     0    0   1;
    66    1     3    3    271        0   1  0  12   0        0     0      0      0     0    0   1;
    67    1     3    3    272        0   1  0  12   0        0     0      0      0     0    0   1;   
    
% 64TR traffic model change, max number of multiplexible SRs
% cfg# N_UE    Nb format   f0      f1  Nf t0  Nt cs0  tOCCidx freqH groupH SRFlag posSR DTX thr  
    68   36    2   1    repelem(0:2, 12)     0   1  0   12   repmat(0:11,1,3)       0       0      0      0     0    0   1;  % HARQ, 12 UE multiplexing (by 12 cyclic shift) per 1RB, total 36 UEs on 3 RBs
    69   36    2   1    repelem(0:5, 6)      0   1  0   12   repmat(0:2:11,1,6)     0       0      0      0     0    0   1;  % HARQ, 6 UE multiplexing (by 6 cyclic shift) per 1RB, total 36 UEs on 6 RBs
    70   108   0   1    repelem(0:2, 36)     0   1  repmat(0:4:8,12,3)   4  repmat([0:2:10, 1:2:11],1,9)       repmat(0:1,6,9)       0      0      1     1    0   1;  % postive SR, 12 UE multiplexing (by 6 cyclic shift & 2 tOCC) per 4 OFDM symbols, 36 UEs per RB, total 108 UEs on 3 RBs
    71   108   0   1    repelem(0:5, 18)     0   1  repmat(0:4:8,6,6)    4  repmat([0:4:8, 1:4:9, 2:4:10, 3:4:11],1,9)     repmat(0:1,3,18)      0      0      1     1    0   1;  % postive SR, 6 UE multiplexing (by 3 cyclic shift & 2 tOCC) per 4 OFDM symbols, 18 UEs per RB, total 108 UEs on 6 RBs
% cfg# N_UE    Nb format   f0      f1  Nf t0  Nt cs0  tOCCidx freqH groupH SRFlag posSR DTX thr  
% 64TR TC PF1 1bitx64UEs (6 PRBs)
   72   64     1    1 ...
   floor([0:63]/12) ... % f0
    0   1  0  14 ... % f1 Nf t0 Nt
   floor(mod(0:63,12)/2)*2 ...%cs0
   mod(0:63,2) ...%tOCCidx
    0      0      0     0    0   1;    
% 64TR TC PF1 2bitx32UEs (6 PRBs)
   73   32     2    1 ...
   floor([0:31]/6) ... % f0
    0   1  0  14 ... % f1 Nf t0 Nt
   floor(mod(0:31,6)/2)*4 ...%cs0
   mod(0:31,2) ...%tOCCidx
    0      0      0     0    0   1;    
% cfg# N_UE    Nb format   f0      f1  Nf t0  Nt cs0  tOCCidx freqH groupH SRFlag posSR DTX thr  
% 64TR TC PF1 1bitx1UE
   74    1      1   1     21        0   1  0  14   0     0       0      0      0     0    0   1;
% 64TR TC PF3 3bitx21UEs
   75   21      3   3    [0:20]     0   1  0  14   0     0       0      0      0     0    0   1;
% 64TR TC PF3 4bitx16UEs (22 PRBs)
   76   16      4   3  [0:9,10:2:20] 0 [ones(1,10),2*ones(1,6)] 0  14   0     0       0      0      0     0    0   1;
% cfg# N_UE    Nb format   f0      f1  Nf t0  Nt cs0  tOCCidx freqH groupH SRFlag posSR DTX thr 
% 64TR TC PF1 1bitx64UEs (6 PRBs, 12 OFDM symbols)
   77   64     1    1 ...
   floor([0:63]/12) ... % f0
    0   1  0  12 ... % f1 Nf t0 Nt
   floor(mod(0:63,12)/2)*2 ...%cs0
   mod(0:63,2) ...%tOCCidx
    0      0      0     0    0   1;    
% 64TR TC PF1 2bitx32UEs (6 PRBs, 12 OFDM symbols)
   78   32     2    1 ...
   floor([0:31]/6) ... % f0
    0   1  0  12 ... % f1 Nf t0 Nt
   floor(mod(0:31,6)/2)*4 ...%cs0
   mod(0:31,2) ...%tOCCidx
    0      0      0     0    0   1;    
% cfg# N_UE    Nb format   f0      f1  Nf t0  Nt cs0  tOCCidx freqH groupH SRFlag posSR DTX thr  
% 64TR TC PF1 1bitx1UE (12 OFDM symbols)
   79    1      1   1     21        0   1  0  12   0     0       0      0      0     0    0   1;
% 64TR TC PF3 3bitx21UEs (12 OFDM symbols)
   80   21      3   3    [0:20]     0   1  0  12   0     0       0      0      0     0    0   1;
% 64TR TC PF3 4bitx16UEs (22 PRBs, 12 OFDM symbols)
   81   16      4   3  [0:9,10:2:20] 0 [ones(1,10),2*ones(1,6)] 0  12   0     0       0      0      0     0    0   1;
% 64TR TC PF1 1bitx72UEs (4 PRBs, PRB 22~25)
   82   72      0   1    repelem(22:25, 18)     0   1  repmat(0:4:8,6,4)    4  repmat([0:4:8, 1:4:9, 2:4:10, 3:4:11],1,6)     repmat(0:1,3,12)      0      0      1     1    0   1;  % postive SR, 6 UE multiplexing (by 3 cyclic shift & 2 tOCC) per 4 OFDM symbols, 18 UEs per RB, total 72 UEs on 4 RBs
% 64TR TC PF1 1bitx72UEs (4 PRBs, PRB 6~9)
   83   72      0   1    repelem(6:9, 18)     0   1  repmat(0:4:8,6,4)    4  repmat([0:4:8, 1:4:9, 2:4:10, 3:4:11],1,6)     repmat(0:1,3,12)      0      0      1     1    0   1;  % postive SR, 6 UE multiplexing (by 3 cyclic shift & 2 tOCC) per 4 OFDM symbols, 18 UEs per RB, total 72 UEs on 4 RBs
% 64TR TC PF1 1bitx256UEs (22 PRBs, 12 OFDM symbols)
   84   256     1    1 ...
   floor([0:255]/12) ... % f0
    0   1  0  12 ... % f1 Nf t0 Nt
   floor(mod(0:255,12)/2)*2 ...%cs0
   mod(0:255,2) ...%tOCCidx
    0      0      0     0    0   1;    
% 64TR TC PF1 2bitx128UEs (22 PRBs, 12 OFDM symbols)
   85   128     2    1 ...
   floor([0:127]/6) ... % f0
    0   1  0  12 ... % f1 Nf t0 Nt
   floor(mod(0:127,6)/2)*4 ...%cs0
   mod(0:127,2) ...%tOCCidx
    0      0      0     0    0   1;    
% cfg# N_UE    Nb format   f0      f1  Nf t0  Nt cs0  tOCCidx freqH groupH SRFlag posSR DTX thr  
% 64TR TC PF1 1bitx1UE (12 OFDM symbols)
   86    1      1   1     85        0   1  0  12   0     0       0      0      0     0    0   1;
% 64TR TC PF3 3bitx85UEs (12 OFDM symbols)
   87   85      3   3    [0:84]     0   1  0  12   0     0       0      0      0     0    0   1;
% 64TR TC PF3 4bitx64UEs (64 PRBs, 12 OFDM symbols)
   88   64      4   3  [0:63] 0 [ones(1,64)] 0  12   0     0       0      0      0     0    0   1;
% 64TR TC PF1 1bitx72UEs (4 PRBs, PRB 64~67)
   89   72      0   1    repelem(64:67, 18)     0   1  repmat(0:4:8,6,4)    4  repmat([0:4:8, 1:4:9, 2:4:10, 3:4:11],1,6)     repmat(0:1,3,12)      0      0      1     1    0   1;  % postive SR, 6 UE multiplexing (by 3 cyclic shift & 2 tOCC) per 4 OFDM symbols, 18 UEs per RB, total 72 UEs on 4 RBs
% 64TR TC PF1 1bitx72UEs (4 PRBs, PRB 86~89)
   90   72      0   1    repelem(86:89, 18)     0   1  repmat(0:4:8,6,4)    4  repmat([0:4:8, 1:4:9, 2:4:10, 3:4:11],1,6)     repmat(0:1,3,12)      0      0      1     1    0   1;  % postive SR, 6 UE multiplexing (by 3 cyclic shift & 2 tOCC) per 4 OFDM symbols, 18 UEs per RB, total 72 UEs on 4 RBs
% 64TR TC PF3 4bitx32UEs (32 PRBs, 12 OFDM symbols)
   91   32      4   3  [0:31] 0 [ones(1,32)] 0  12   0     0       0      0      0     0    0   1;
% 64TR TC PF1 1bitx72UEs (4 PRBs, PRB 32~35)
   92   72      0   1    repelem(32:35, 18)     0   1  repmat(0:4:8,6,4)    4  repmat([0:4:8, 1:4:9, 2:4:10, 3:4:11],1,6)     repmat(0:1,3,12)      0      0      1     1    0   1;  % postive SR, 6 UE multiplexing (by 3 cyclic shift & 2 tOCC) per 4 OFDM symbols, 18 UEs per RB, total 72 UEs on 4 RBs
% 64TR TC PF1 2bitx64UEs (11 PRBs, 12 OFDM symbols)
   93   64      2    1 ...
   floor([0:63]/6) ... % f0
    0   1  0  12 ... % f1 Nf t0 Nt
   floor(mod(0:63,6)/2)*4 ...%cs0
   mod(0:63,2) ...%tOCCidx
    0      0      0     0    0   1;   
% 64TR TC PF1 1bitx72UEs (4 PRBs, PRB 11~14)
   94   72      0   1    repelem(11:14, 18)     0   1  repmat(0:4:8,6,4)    4  repmat([0:4:8, 1:4:9, 2:4:10, 3:4:11],1,6)     repmat(0:1,3,12)      0      0      1     1    0   1;  % postive SR, 6 UE multiplexing (by 3 cyclic shift & 2 tOCC) per 4 OFDM symbols, 18 UEs per RB, total 72 UEs on 4 RBs 
% 64TR TC PF3 3bitx42UEs (12 OFDM symbols)
   95   42      3   3    [0:41]     0   1  0  12   0     0       0      0      0     0    0   1;
% 64TR TC PF1 1bitx1UE (12 OFDM symbols)
   96    1      1   1     42        0   1  0  12   0     0       0      0      0     0    0   1;
% 64TR TC PF1 1bitx72UEs (4 PRBs, PRB 43~46)
   97   72      0   1    repelem(43:46, 18)     0   1  repmat(0:4:8,6,4)    4  repmat([0:4:8, 1:4:9, 2:4:10, 3:4:11],1,6)     repmat(0:1,3,12)      0      0      1     1    0   1;  % postive SR, 6 UE multiplexing (by 3 cyclic shift & 2 tOCC) per 4 OFDM symbols, 18 UEs per RB, total 72 UEs on 4 RBs
   % 64TR TC PF1 1bitx128UEs (11 PRBs, 12 OFDM symbols)
   98   128     1    1 ...
   floor([0:127]/12) ... % f0
    0   1  0  12 ... % f1 Nf t0 Nt
   floor(mod(0:127,12)/2)*2 ...%cs0
   mod(0:127,2) ...%tOCCidx
    0      0      0     0    0   1;    
% 64TR TC PF3 4bitx24UEs (24 PRBs, 12 OFDM symbols)
   99   24      4   3  [0:23] 0 [ones(1,24)] 0  12   0     0       0      0      0     0    0   1;
% 64TR TC PF1 1bitx72UEs (4 PRBs, PRB 24~27)
  100   72      0   1    repelem(24:27, 18)     0   1  repmat(0:4:8,6,4)    4  repmat([0:4:8, 1:4:9, 2:4:10, 3:4:11],1,6)     repmat(0:1,3,12)      0      0      1     1    0   1;  % postive SR, 6 UE multiplexing (by 3 cyclic shift & 2 tOCC) per 4 OFDM symbols, 18 UEs per RB, total 72 UEs on 4 RBs
% 64TR TC PF1 2bitx48UEs (8 PRBs, 12 OFDM symbols)
  101   48      2    1 ...
   floor([0:47]/6) ... % f0
    0   1  0  12 ... % f1 Nf t0 Nt
   floor(mod(0:47,6)/2)*4 ...%cs0
   mod(0:47,2) ...%tOCCidx
    0      0      0     0    0   1;   
% 64TR TC PF1 1bitx72UEs (4 PRBs, PRB 8~11)
  102   72      0   1    repelem(8:11, 18)     0   1  repmat(0:4:8,6,4)    4  repmat([0:4:8, 1:4:9, 2:4:10, 3:4:11],1,6)     repmat(0:1,3,12)      0      0      1     1    0   1;  % postive SR, 6 UE multiplexing (by 3 cyclic shift & 2 tOCC) per 4 OFDM symbols, 18 UEs per RB, total 72 UEs on 4 RBs 
% 64TR TC PF3 3bitx32UEs (12 OFDM symbols)
  103   32      3   3    [0:31]     0   1  0  12   0     0       0      0      0     0    0   1;
% 64TR TC PF1 1bitx1UE (12 OFDM symbols)
  104    1      1   1     32        0   1  0  12   0     0       0      0      0     0    0   1;
% 64TR TC PF1 1bitx72UEs (4 PRBs, PRB 33~36)
  105   72      0   1    repelem(33:36, 18)     0   1  repmat(0:4:8,6,4)    4  repmat([0:4:8, 1:4:9, 2:4:10, 3:4:11],1,6)     repmat(0:1,3,12)      0      0      1     1    0   1;  % postive SR, 6 UE multiplexing (by 3 cyclic shift & 2 tOCC) per 4 OFDM symbols, 18 UEs per RB, total 72 UEs on 4 RBs
   % 64TR TC PF1 1bitx96UEs (8 PRBs, 12 OFDM symbols)
  106   96     1    1 ...
   floor([0:95]/12) ... % f0
    0   1  0  12 ... % f1 Nf t0 Nt
   floor(mod(0:95,12)/2)*2 ...%cs0
   mod(0:95,2) ...%tOCCidx
    0      0      0     0    0   1;    
   };

CFG_PUSCH = {...
% TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg 
    1,  1,        1,   1,  10,  150, 10,      4,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    0; % FDM
    2,  1,        4,   1, 160,  100, 10,      4,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    1; % FDM
    3,  1,        1,   1, 100,  150,  0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    0; % CDM
    4,  1,        4,   1, 100,  150,  0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   1,    0; % CDM
    5,  1,        1,   1,  10,  100,  0,     14,     0,    0,    273,    7,    0,     41,     2,      1,     0,     41,      2,   0,    0; % FDM
    6,  1,        4,   1, 200,   50,  0,     14,     0,    0,    273,    6,    0,     41,     2,      1,     0,     41,      2,   0,    1; % FDM
    
    % perf test F08 (with PRACH + PUCCH)  
    7,  1,        27,  2,  16,  240,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
    % F08 (with PRACH)
    8,  1,        27,  2,  0,   256,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
    % F09 (with PRACH + PUCCH)
    9,  1,        27,  2,  16,  240,  0,     14,     0,    0,    273,    0,    0,     41,     2,      2,     0,     41,      2,   0,    0; 
    10, 1,        27,  2,  16,  240,  0,     14,     0,    0,    273,    0,    0,     41,     2,      2,     0,     41,      2,   2,    0; 
    % F09 (with PRACH)
    11, 1,        27,  2,  0,   256,  0,     14,     0,    0,    273,    0,    0,     41,     2,      2,     0,     41,      2,   0,    0; 
    12, 1,        27,  2,  0,   256,  0,     14,     0,    0,    273,    0,    0,     41,     2,      2,     0,     41,      2,   2,    0; 
    % F08 (with PUCCH only, no PRACH)
    13, 1,        27,  2,  16,  256,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
 % TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg 
    % F08 240 PRBs, 16 UEGs (with PRACH + PUCCH)
    14,  1,        27,  2,  16,  15,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
    15,  1,        27,  2,  31,  15,  0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    1;
    16,  1,        27,  2,  46,  15,  0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    2;
    17,  1,        27,  2,  61,  15,  0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     0,     41,      2,   0,    3;
    18,  1,        27,  2,  76,  15,  0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   0,    4;
    19,  1,        27,  2,  91,  15,  0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    5;
    20,  1,        27,  2, 106,  15,  0,     14,     0,    0,    273,    6,    0,     41,     2,      1,     0,     41,      2,   0,    6;
    21,  1,        27,  2, 121,  15,  0,     14,     0,    0,    273,    7,    0,     41,     2,      1,     0,     41,      2,   0,    7;
    22,  1,        27,  2, 136,  15,  0,     14,     0,    0,    273,    8,    0,     41,     2,      1,     0,     41,      2,   0,    8;
    23,  1,        27,  2, 151,  15,  0,     14,     0,    0,    273,    9,    0,     41,     2,      1,     0,     41,      2,   0,    9;
    24,  1,        27,  2, 166,  15,  0,     14,     0,    0,    273,   10,    0,     41,     2,      1,     0,     41,      2,   0,    10;
    25,  1,        27,  2, 181,  15,  0,     14,     0,    0,    273,   11,    0,     41,     2,      1,     0,     41,      2,   0,    11;
    26,  1,        27,  2, 196,  15,  0,     14,     0,    0,    273,   12,    0,     41,     2,      1,     0,     41,      2,   0,    12;
    27,  1,        27,  2, 211,  15,  0,     14,     0,    0,    273,   13,    0,     41,     2,      1,     0,     41,      2,   0,    13;
    28,  1,        27,  2, 226,  15,  0,     14,     0,    0,    273,   14,    0,     41,     2,      1,     0,     41,      2,   0,    14;
    29,  1,        27,  2, 241,  15,  0,     14,     0,    0,    273,   15,    0,     41,     2,      1,     0,     41,      2,   0,    15;
    % F08 256 PRBs, 16 UEGs (with PRACH only)
    30,  1,        27,  2,   0,  16,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
    31,  1,        27,  2,  16,  16,  0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    1;
    32,  1,        27,  2,  32,  16,  0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    2;
    33,  1,        27,  2,  48,  16,  0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     0,     41,      2,   0,    3;
    34,  1,        27,  2,  64,  16,  0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   0,    4;
    35,  1,        27,  2,  80,  16,  0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    5;
    36,  1,        27,  2,  96,  16,  0,     14,     0,    0,    273,    6,    0,     41,     2,      1,     0,     41,      2,   0,    6;
    37,  1,        27,  2, 112,  16,  0,     14,     0,    0,    273,    7,    0,     41,     2,      1,     0,     41,      2,   0,    7;
    38,  1,        27,  2, 128,  16,  0,     14,     0,    0,    273,    8,    0,     41,     2,      1,     0,     41,      2,   0,    8;
    39,  1,        27,  2, 144,  16,  0,     14,     0,    0,    273,    9,    0,     41,     2,      1,     0,     41,      2,   0,    9;
    40,  1,        27,  2, 160,  16,  0,     14,     0,    0,    273,   10,    0,     41,     2,      1,     0,     41,      2,   0,    10;
    41,  1,        27,  2, 176,  16,  0,     14,     0,    0,    273,   11,    0,     41,     2,      1,     0,     41,      2,   0,    11;
    42,  1,        27,  2, 192,  16,  0,     14,     0,    0,    273,   12,    0,     41,     2,      1,     0,     41,      2,   0,    12;
    43,  1,        27,  2, 208,  16,  0,     14,     0,    0,    273,   13,    0,     41,     2,      1,     0,     41,      2,   0,    13;
    44,  1,        27,  2, 224,  16,  0,     14,     0,    0,    273,   14,    0,     41,     2,      1,     0,     41,      2,   0,    14;
    45,  1,        27,  2, 240,  16,  0,     14,     0,    0,    273,   15,    0,     41,     2,      1,     0,     41,      2,   0,    15;
    % F08 256 PRBs, 16 UEGs (with PUCCH only)
    46,  1,        27,  2,  16,  16,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
    47,  1,        27,  2,  32,  16,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    1;
    48,  1,        27,  2,  48,  16,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    2;
    49,  1,        27,  2,  64,  16,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    3;
    50,  1,        27,  2,  80,  16,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    4;
    51,  1,        27,  2,  96,  16,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    5;
    52,  1,        27,  2, 112,  16,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    6;
    53,  1,        27,  2, 128,  16,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    7;
    54,  1,        27,  2, 144,  16,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    8;
    55,  1,        27,  2, 160,  16,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    9;
    56,  1,        27,  2, 176,  16,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    10;
    57,  1,        27,  2, 192,  16,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    11;
    58,  1,        27,  2, 208,  16,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    12;
    59,  1,        27,  2, 224,  16,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    13;
    60,  1,        27,  2, 240,  16,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    14;
    61,  1,        27,  2, 256,  16,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    15;
% TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg 
    % F08 128 PRBs, 16 UEGs, 8 RB for each UE (with PRACH + PUCCH)
    62,  1,        27,  2,  16,   8,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
    63,  1,        27,  2,  31,   8,  0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    1;
    64,  1,        27,  2,  46,   8,  0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    2;
    65,  1,        27,  2,  61,   8,  0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     0,     41,      2,   0,    3;
    66,  1,        27,  2,  76,   8,  0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   0,    4;
    67,  1,        27,  2,  91,   8,  0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    5;
    68,  1,        27,  2, 106,   8,  0,     14,     0,    0,    273,    6,    0,     41,     2,      1,     0,     41,      2,   0,    6;
    69,  1,        27,  2, 121,   8,  0,     14,     0,    0,    273,    7,    0,     41,     2,      1,     0,     41,      2,   0,    7;
    70,  1,        27,  2, 136,   8,  0,     14,     0,    0,    273,    8,    0,     41,     2,      1,     0,     41,      2,   0,    8;
    71,  1,        27,  2, 151,   8,  0,     14,     0,    0,    273,    9,    0,     41,     2,      1,     0,     41,      2,   0,    9;
    72,  1,        27,  2, 166,   8,  0,     14,     0,    0,    273,   10,    0,     41,     2,      1,     0,     41,      2,   0,    10;
    73,  1,        27,  2, 181,   8,  0,     14,     0,    0,    273,   11,    0,     41,     2,      1,     0,     41,      2,   0,    11;
    74,  1,        27,  2, 196,   8,  0,     14,     0,    0,    273,   12,    0,     41,     2,      1,     0,     41,      2,   0,    12;
    75,  1,        27,  2, 211,   8,  0,     14,     0,    0,    273,   13,    0,     41,     2,      1,     0,     41,      2,   0,    13;
    76,  1,        27,  2, 226,   8,  0,     14,     0,    0,    273,   14,    0,     41,     2,      1,     0,     41,      2,   0,    14;
    77,  1,        27,  2, 241,   8,  0,     14,     0,    0,    273,   15,    0,     41,     2,      1,     0,     41,      2,   0,    15;
    
    % F08 258 PRBs, 6 UEGs, 43 PRB for each UE (with PRACH + PUCCH)
    78,  1,        27,  2,  2,    43, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
    79,  1,        27,  2,  45,   43, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    1;
    80,  1,        27,  2,  88,   43, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    2;
    81,  1,        27,  2,  131,  43, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     0,     41,      2,   0,    3;
    82,  1,        27,  2,  174,  43, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   0,    4;
    83,  1,        27,  2,  217,  43, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    5;
    
    % F08 132 PRBs, 6 UEGs, 22 PRB for each UE 
    84,  1,        27,  2,  1,    22, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
    85,  1,        27,  2,  23,   22, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    1;
    86,  1,        27,  2,  45,   22, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    2;
    87,  1,        27,  2,  67,   22, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     0,     41,      2,   0,    3;
    88,  1,        27,  2,  89,   22, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   0,    4;
    89,  1,        27,  2,  111,  22, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    5;
    
    % F08 6 UEGs, 2 PRB for each UE (with PRACH + PUCCH)
    90,  1,        27,  2,  2,     2, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
    91,  1,        27,  2,  4,     2, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    1;
    92,  1,        27,  2,  6,     2, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    2;
    93,  1,        27,  2,  8,     2, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     0,     41,      2,   0,    3;
    94,  1,        27,  2,  10,    2, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   0,    4;
    95,  1,        27,  2,  12,    2, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    5;
    % 1 UEG, 2 PRB for each UE 
    96,  1,        27,  2,  2,     2, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
 % TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg 
    % F08 208 PRBs, 16 UEGs (with 4 PRACH + PUCCH)
    97,  1,        27,  2,  16,  13,  0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
    98,  1,        27,  2,  29,  13,  0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    1;
    99,  1,        27,  2,  42,  13,  0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    2;
   100,  1,        27,  2,  55,  13,  0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     0,     41,      2,   0,    3;
   101,  1,        27,  2,  68,  13,  0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   0,    4;
   102,  1,        27,  2,  81,  13,  0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    5;
   103,  1,        27,  2,  94,  13,  0,     14,     0,    0,    273,    6,    0,     41,     2,      1,     0,     41,      2,   0,    6;
   104,  1,        27,  2, 107,  13,  0,     14,     0,    0,    273,    7,    0,     41,     2,      1,     0,     41,      2,   0,    7;
   105,  1,        27,  2, 120,  13,  0,     14,     0,    0,    273,    8,    0,     41,     2,      1,     0,     41,      2,   0,    8;
   106,  1,        27,  2, 133,  13,  0,     14,     0,    0,    273,    9,    0,     41,     2,      1,     0,     41,      2,   0,    9;
   107,  1,        27,  2, 146,  13,  0,     14,     0,    0,    273,   10,    0,     41,     2,      1,     0,     41,      2,   0,    10;
   108,  1,        27,  2, 159,  13,  0,     14,     0,    0,    273,   11,    0,     41,     2,      1,     0,     41,      2,   0,    11;
   109,  1,        27,  2, 172,  13,  0,     14,     0,    0,    273,   12,    0,     41,     2,      1,     0,     41,      2,   0,    12;
   110,  1,        27,  2, 185,  13,  0,     14,     0,    0,    273,   13,    0,     41,     2,      1,     0,     41,      2,   0,    13;
   111,  1,        27,  2, 198,  13,  0,     14,     0,    0,    273,   14,    0,     41,     2,      1,     0,     41,      2,   0,    14;
   112,  1,        27,  2, 211,  13,  0,     14,     0,    0,    273,   15,    0,     41,     2,      1,     0,     41,      2,   0,    15;
    % F08 102 PRBs, 6 UEGs, 17 PRB for each UE (with PRACH + PUCCH)
   113,  1,        27,  2,  2,    17, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
   114,  1,        27,  2,  19,   17, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    1;
   115,  1,        27,  2,  36,   17, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    2;
   116,  1,        27,  2,  53,   17, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     0,     41,      2,   0,    3;
   117,  1,        27,  2,  70,   17, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   0,    4;
   118,  1,        27,  2,  87,   17, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    5;
    % F08 72 PRBs, 6 UEGs, 12 PRB for each UE (with PRACH + PUCCH)
   119,  1,        27,  2,  2,    12, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
   120,  1,        27,  2,  14,   12, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    1;
   121,  1,        27,  2,  26,   12, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    2;
   122,  1,        27,  2,  38,   12, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     0,     41,      2,   0,    3;
   123,  1,        27,  2,  50,   12, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   0,    4;
   124,  1,        27,  2,  62,   12, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    5;
    % F08 84 PRBs, 6 UEGs, 14 PRB for each UE (with PRACH + PUCCH)
   125,  1,        27,  2,  1,    14, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
   126,  1,        27,  2,  15,   14, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    1;
   127,  1,        27,  2,  29,   14, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    2;
   128,  1,        27,  2,  43,   14, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     0,     41,      2,   0,    3;
   129,  1,        27,  2,  57,   14, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   0,    4;
   130,  1,        27,  2,  71,   14, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    5;
     % F08 270 PRBs, 6 UEGs, 45 PRB for each UE (with PUCCH)
   131,  1,        27,  2,  1,    45, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
   132,  1,        27,  2,  46,   45, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    1;
   133,  1,        27,  2,  91,   45, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    2;
   134,  1,        27,  2,  136,  45, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     0,     41,      2,   0,    3;
   135,  1,        27,  2,  181,  45, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   0,    4;
   136,  1,        27,  2,  226,  45, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    5;
     % F08 peak, 222 PRBs, 6 UEGs, 37 PRB for each UE (with PRACH + PUCCH)
   137,  1,        27,  2,  1,    37, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
   138,  1,        27,  2,  38,   37, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    1;
   139,  1,        27,  2,  75,   37, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    2;
   140,  1,        27,  2,  112,  37, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     0,     41,      2,   0,    3;
   141,  1,        27,  2,  149,  37, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   0,    4;
   142,  1,        27,  2,  186,  37, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    5;
     % F08 240 PRBs, 6 UEGs, 40 PRB for each UE (with PUCCH)
   143,  1,        27,  2,  1,    40, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
   144,  1,        27,  2,  41,   40, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    1;
   145,  1,        27,  2,  81,   40, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    2;
   146,  1,        27,  2,  121,  40, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     0,     41,      2,   0,    3;
   147,  1,        27,  2,  161,  40, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   0,    4;
   148,  1,        27,  2,  201,  40, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    5;
    % F08 132 PRBs, 1 UEG, 14 PRB for each UE (with PUCCH)
   149,  1,        27,  2,  1,    132, 0,    14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
    % F08 84 PRBs, 6 UEGs, 14 PRB for each UE (with PRACH + PUCCH)
   150,  1,        27,  2,  1,    84, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
    % F08 270 PRBs, 1 UEG, 270 PRB for each UE (with PUCCH)
   151,  1,        27,  2,  1,   270, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
    % F08 peak, 222 PRBs, 1 UEGs, 222 PRB for each UE (with PRACH + PUCCH)
   152,  1,        27,  2,  1,   222, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
    % F08 102 PRBs, 6 UEGs, 17 PRB for each UE (with PRACH + PUCCH), start at 1
   153,  1,        27,  2,  1,    17, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
   154,  1,        27,  2,  18,   17, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    1;
   155,  1,        27,  2,  35,   17, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    2;
   156,  1,        27,  2,  52,   17, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     0,     41,      2,   0,    3;
   157,  1,        27,  2,  69,   17, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   0,    4;
   158,  1,        27,  2,  86,   17, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    5;
   % 32T32R
% TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg 
   159,  1,        27,  2,  12, 261,  0,     14,     0,    0,    273,   201,    0,     41,     2,      1,    0,     41,      2,   0,    0; % FDM
   160,  1,        27,  2,   0, 273,  0,     14,     0,    0,    273,   201,    0,     41,     2,      1,    0,     41,      2,   0,    0; % FDM      
   % F08 270 PRBs, 6 UEGs, 45 PRB for each UE, OTA (addDmrs)
   161,  1,        27,  2,  1,    45, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   162,  1,        27,  2,  46,   45, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   163,  1,        27,  2,  91,   45, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   164,  1,        27,  2,  136,  45, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   165,  1,        27,  2,  181,  45, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   166,  1,        27,  2,  226,  45, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 peak, 222 PRBs, 6 UEGs, 37 PRB for each UE (with PRACH + PUCCH)
   167,  1,        27,  2,  1,    37, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   168,  1,        27,  2,  38,   37, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   169,  1,        27,  2,  75,   37, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   170,  1,        27,  2,  112,  37, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   171,  1,        27,  2,  149,  37, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   172,  1,        27,  2,  186,  37, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 132 PRBs, 6 UEGs, 22 PRB for each UE, OTA (addDmrs)
   173,  1,        27,  2,  1,    22, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   174,  1,        27,  2,  23,   22, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   175,  1,        27,  2,  45,   22, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   176,  1,        27,  2,  67,   22, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   177,  1,        27,  2,  89,   22, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   178,  1,        27,  2,  111,  22, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % F08 84 PRBs, 6 UEGs, 14 PRB for each UE (with PRACH + PUCCH)
   179,  1,        27,  2,  1,    14, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   180,  1,        27,  2,  15,   14, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   181,  1,        27,  2,  29,   14, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   182,  1,        27,  2,  43,   14, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   183,  1,        27,  2,  57,   14, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   184,  1,        27,  2,  71,   14, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 39 PRB for each UE,  (with 3 PRACH + PUCCH), OTA (addDmrs)
    185,  1,        27,  2,  1+0*39,   39, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
    186,  1,        27,  2,  1+1*39,   39, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
    187,  1,        27,  2,  1+2*39,   39, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
    188,  1,        27,  2,  1+3*39,   39, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
    189,  1,        27,  2,  1+4*39,   39, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
    190,  1,        27,  2,  1+5*39,   39, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 16 PRB for each UE, OTA (addDmrs)
   191,  1,        27,  2,  1+0*16,   16, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   192,  1,        27,  2,  1+1*16,   16, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   193,  1,        27,  2,  1+2*16,   16, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   194,  1,        27,  2,  1+3*16,   16, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   195,  1,        27,  2,  1+4*16,   16, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   196,  1,        27,  2,  1+5*16,   16, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
  % F08 peak, 40MHz, 102 PRBs, 6 UEGs, 17 PRB for each UE, OTA (addDmrs)
   197,  1,        27,  2,  1,    17, 0,     14,     0,    0,    106,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   198,  1,        27,  2,  18,   17, 0,     14,     0,    0,    106,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   199,  1,        27,  2,  35,   17, 0,     14,     0,    0,    106,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   200,  1,        27,  2,  52,   17, 0,     14,     0,    0,    106,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   201,  1,        27,  2,  69,   17, 0,     14,     0,    0,    106,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   202,  1,        27,  2,  86,   17, 0,     14,     0,    0,    106,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
% F08 peak, 40MHz, 54 PRBs, 6 UEGs, 9 PRB for each UE (with PRACH + PUCCH)
   203,  1,        27,  2,  1,    9,  0,     14,     0,    0,    106,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   204,  1,        27,  2,  10,   9,  0,     14,     0,    0,    106,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   205,  1,        27,  2,  19,   9,  0,     14,     0,    0,    106,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   206,  1,        27,  2,  28,   9,  0,     14,     0,    0,    106,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   207,  1,        27,  2,  37,   9,  0,     14,     0,    0,    106,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   208,  1,        27,  2,  46,   9,  0,     14,     0,    0,    106,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
% F08 ave, 40MHz, 48 PRBs, 6 UEGs, 8 PRB for each UE, OTA (addDmrs)
   209,  1,        27,  2,  1,    8,  0,     14,     0,    0,    106,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   210,  1,        27,  2,  9,    8,  0,     14,     0,    0,    106,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   211,  1,        27,  2,  17,   8,  0,     14,     0,    0,    106,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   212,  1,        27,  2,  25,   8,  0,     14,     0,    0,    106,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   213,  1,        27,  2,  33,   8,  0,     14,     0,    0,    106,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   214,  1,        27,  2,  41,   8,  0,     14,     0,    0,    106,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
  % F08 peak, 80MHz, 168 PRBs, 6 UEGs, 28 PRB for each UE, OTA (addDmrs)
   215,  1,        27,  2,  1,    36, 0,     14,     0,    0,    217,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   216,  1,        27,  2,  37,   36, 0,     14,     0,    0,    217,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   217,  1,        27,  2,  73,   36, 0,     14,     0,    0,    217,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   218,  1,        27,  2,  109,  36, 0,     14,     0,    0,    217,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   219,  1,        27,  2,  145,  36, 0,     14,     0,    0,    217,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   220,  1,        27,  2,  181,  36, 0,     14,     0,    0,    217,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
  % F08 peak, 80MHz, 168 PRBs, 6 UEGs, 28 PRB for each UE (with PRACH + PUCCH)
   221,  1,        27,  2,  1,    28,  0,     14,     0,    0,    217,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   222,  1,        27,  2,  29,   28,  0,     14,     0,    0,    217,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   223,  1,        27,  2,  57,   28,  0,     14,     0,    0,    217,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   224,  1,        27,  2,  85,   28,  0,     14,     0,    0,    217,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   225,  1,        27,  2,  113,  28,  0,     14,     0,    0,    217,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   226,  1,        27,  2,  141,  28,  0,     14,     0,    0,    217,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
 % F08 ave, 80MHz, 102 PRBs, 6 UEGs, 17 PRB for each UE, OTA (addDmrs)
   227,  1,        27,  2,  1,    17,  0,     14,     0,    0,    217,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   228,  1,        27,  2,  18,   17,  0,     14,     0,    0,    217,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   229,  1,        27,  2,  35,   17,  0,     14,     0,    0,    217,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   230,  1,        27,  2,  52,   17,  0,     14,     0,    0,    217,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   231,  1,        27,  2,  69,   17,  0,     14,     0,    0,    217,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   232,  1,        27,  2,  86,   17,  0,     14,     0,    0,    217,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
  % F08 ave, 80MHz, 54 PRBs, 6 UEGs, 9 PRB for each UE, OTA (addDmrs) (with PRACH + PUCCH)
   233,  1,        27,  2,  1,    9,   0,     14,     0,    0,    217,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   234,  1,        27,  2,  10,   9,   0,     14,     0,    0,    217,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   235,  1,        27,  2,  19,   9,   0,     14,     0,    0,    217,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   236,  1,        27,  2,  28,   9,   0,     14,     0,    0,    217,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   237,  1,        27,  2,  37,   9,   0,     14,     0,    0,    217,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   238,  1,        27,  2,  46,   9,   0,     14,     0,    0,    217,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
      % F08 6 UEGs, 41 PRB for each UE, OTA (addDmrs)
   239,  1,        27,  2,  12+0*41,   41, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   240,  1,        27,  2,  12+1*41,   41, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   241,  1,        27,  2,  12+2*41,   41, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   242,  1,        27,  2,  12+3*41,   41, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   243,  1,        27,  2,  12+4*41,   41, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   244,  1,        27,  2,  12+5*41,   41, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 6 UEGs, 41 PRB for each UE, OTA (addDmrs)
   245,  1,        27,  2,  12+0*33,   33, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   246,  1,        27,  2,  12+1*33,   33, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   247,  1,        27,  2,  12+2*33,   33, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   248,  1,        27,  2,  12+3*33,   33, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   249,  1,        27,  2,  12+4*33,   33, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   250,  1,        27,  2,  12+5*33,   33, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % F08 6 UEGs, 20 PRB for each UE, OTA (addDmrs)
   251,  1,        27,  2,  6+0*20,   20, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   252,  1,        27,  2,  6+1*20,   20, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   253,  1,        27,  2,  6+2*20,   20, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   254,  1,        27,  2,  6+3*20,   20, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   255,  1,        27,  2,  6+4*20,   20, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   256,  1,        27,  2,  6+5*20,   20, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 6 UEGs, 20 PRB for each UE, OTA (addDmrs)
   257,  1,        27,  2,  6+0*12,   12, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   258,  1,        27,  2,  6+1*12,   12, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   259,  1,        27,  2,  6+2*12,   12, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   260,  1,        27,  2,  6+3*12,   12, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   261,  1,        27,  2,  6+4*12,   12, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   262,  1,        27,  2,  6+5*12,   12, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % 32T32R - muMIMO
% TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg 
   263,  1,        27,  2,  12, 261,  0,     14,     0,    0,    273,   201,    0,     41,     2,      2,    1,     41,      2,   0,    0; % CDM
   264,  1,        27,  2,  12, 261,  0,     14,     0,    0,    273,   202,    0,     41,     2,      2,    1,     41,      2,   2,    0; % CDM
   265,  1,        27,  2,   0, 273,  0,     14,     0,    0,    273,   201,    0,     41,     2,      2,    1,     41,      2,   0,    0; % CDM
   266,  1,        27,  2,   0, 273,  0,     14,     0,    0,    273,   202,    0,     41,     2,      2,    1,     41,      2,   2,    0; % CDM
   % F08 30 MHz, 6 UEGs, 12 PRB for each UE, OTA (addDmrs)
   267,  1,        27,  2,  1+0*12,   12, 0,     14,     0,    0,    78,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   268,  1,        27,  2,  1+1*12,   12, 0,     14,     0,    0,    78,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   269,  1,        27,  2,  1+2*12,   12, 0,     14,     0,    0,    78,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   270,  1,        27,  2,  1+3*12,   12, 0,     14,     0,    0,    78,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   271,  1,        27,  2,  1+4*12,   12, 0,     14,     0,    0,    78,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   272,  1,        27,  2,  1+5*12,   12, 0,     14,     0,    0,    78,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 30 MHz, 6 UEGs, 4 PRB for each UE, OTA (addDmrs)
   273,  1,        27,  2,  1+0*4,   4, 0,     14,     0,    0,    78,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   274,  1,        27,  2,  1+1*4,   4, 0,     14,     0,    0,    78,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   275,  1,        27,  2,  1+2*4,   4, 0,     14,     0,    0,    78,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   276,  1,        27,  2,  1+3*4,   4, 0,     14,     0,    0,    78,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   277,  1,        27,  2,  1+4*4,   4, 0,     14,     0,    0,    78,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   278,  1,        27,  2,  1+5*4,   4, 0,     14,     0,    0,    78,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
      % F08 30 MHz, 6 UEGs, 6 PRB for each UE, OTA (addDmrs)
   279,  1,        27,  2,  1+0*6,   6, 0,     14,     0,    0,    78,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   280,  1,        27,  2,  1+1*6,   6, 0,     14,     0,    0,    78,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   281,  1,        27,  2,  1+2*6,   6, 0,     14,     0,    0,    78,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   282,  1,        27,  2,  1+3*6,   6, 0,     14,     0,    0,    78,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   283,  1,        27,  2,  1+4*6,   6, 0,     14,     0,    0,    78,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   284,  1,        27,  2,  1+5*6,   6, 0,     14,     0,    0,    78,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % F08 50 MHz, 6 UEGs, 22 PRB for each UE, OTA (addDmrs)
   285,  1,        27,  2,  1+0*22,   22, 0,     14,     0,    0,    133,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   286,  1,        27,  2,  1+1*22,   22, 0,     14,     0,    0,    133,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   287,  1,        27,  2,  1+2*22,   22, 0,     14,     0,    0,    133,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   288,  1,        27,  2,  1+3*22,   22, 0,     14,     0,    0,    133,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   289,  1,        27,  2,  1+4*22,   22, 0,     14,     0,    0,    133,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   290,  1,        27,  2,  1+5*22,   22, 0,     14,     0,    0,    133,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 50 MHz, 6 UEGs, 14 PRB for each UE, OTA (addDmrs)
   291,  1,        27,  2,  1+0*14,   14, 0,     14,     0,    0,    133,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   292,  1,        27,  2,  1+1*14,   14, 0,     14,     0,    0,    133,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   293,  1,        27,  2,  1+2*14,   14, 0,     14,     0,    0,    133,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   294,  1,        27,  2,  1+3*14,   14, 0,     14,     0,    0,    133,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   295,  1,        27,  2,  1+4*14,   14, 0,     14,     0,    0,    133,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   296,  1,        27,  2,  1+5*14,   14, 0,     14,     0,    0,    133,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 50 MHz, 6 UEGs, 11 PRB for each UE, OTA (addDmrs)
   297,  1,        27,  2,  1+0*11,   11, 0,     14,     0,    0,    133,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   298,  1,        27,  2,  1+1*11,   11, 0,     14,     0,    0,    133,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   299,  1,        27,  2,  1+2*11,   11, 0,     14,     0,    0,    133,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   300,  1,        27,  2,  1+3*11,   11, 0,     14,     0,    0,    133,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   301,  1,        27,  2,  1+4*11,   11, 0,     14,     0,    0,    133,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   302,  1,        27,  2,  1+5*11,   11, 0,     14,     0,    0,    133,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;  
   % SRS MU-MIMO
   303,  1,        27,  1,  12, 261,  0,     14,     0,    0,    273,   201,    0,     41,     2,      2,    1,     41,      2,   0,    0; % CDM
   304,  1,        27,  1,  12, 261,  0,     14,     0,    0,    273,   202,    0,     41,     2,      2,    1,     41,      2,   1,    0; % CDM
   305,  1,        27,  1,   0, 273,  0,     14,     0,    0,    273,   201,    0,     41,     2,      2,    1,     41,      2,   0,    0; % CDM
   306,  1,        27,  1,   0, 273,  0,     14,     0,    0,    273,   202,    0,     41,     2,      2,    1,     41,      2,   1,    0; % CDM
   307,  1,        27,  2,  12, 261,  0,     14,     0,    0,    273,   201,    0,     41,     2,      2,    1,     41,      2,   0,    0; % CDM
   308,  1,        27,  2,  12, 261,  0,     14,     0,    0,    273,   202,    0,     41,     2,      2,    1,     41,      2,   2,    0; % CDM
   309,  1,        27,  2,   0, 273,  0,     14,     0,    0,    273,   201,    0,     41,     2,      2,    1,     41,      2,   0,    0; % CDM
   310,  1,        27,  2,   0, 273,  0,     14,     0,    0,    273,   202,    0,     41,     2,      2,    1,     41,      2,   2,    0; % CDM    
   % 64TR TC
 % TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg    
   % slot 3
   311,  1,         0,  1,  12,  249,  10,    4,     0,    0,    273,     7,    0,     41,     0,      1,    0,     41,      1,   0,    0; % CDM
   % slot 4
   312,  1,        27,  2,   0,   50,   0,   14,     0,    0,    273,     1,    0,     41,     2,      1,    1,     41,      1,   0,    0; % CDM
   313,  1,        27,  2,  50,   50,   0,   14,     0,    0,    273,     2,    0,     41,     2,      1,    1,     41,      1,   0,    1; % CDM
   314,  1,        27,  2, 100,   50,   0,   14,     0,    0,    273,     3,    0,     41,     2,      1,    1,     41,      1,   0,    2; % CDM
   315,  1,        27,  2, 150,   50,   0,   14,     0,    0,    273,     4,    0,     41,     2,      1,    1,     41,      1,   0,    3; % CDM
   316,  1,        27,  1, 200,   50,   0,   14,     0,    0,    273,     5,    0,     41,     2,      1,    1,     41,      1,   0,    4; % CDM
   317,  1,        27,  1, 250,   23,   0,   14,     0,    0,    273,     6,    0,     41,     2,      1,    1,     41,      1,   0,    5; % CDM
   % slot 13
   318,  1,         5,  1,   0,  273,  10,    4,     0,    0,    273,     8,    0,     41,     0,      1,    0,     41,      1,   0,    0; % CDM
   % slot 14
   319,  1,        10,  1,   0,  273,   0,   14,     0,    0,    273,     9,    0,     41,     2,      1,    1,     41,      1,   0,    0; % CDM
   % slot 15
   320,  1,        27,  2,   0,   50,   0,   14,     0,    0,    273,     1,    0,     41,     2,      1,    1,     41,      1,   0,    0; % CDM
   321,  1,        31,  2,  50,   50,   0,   14,     0,    0,    273,     2,    3,     41,     2,      1,    1,     41,      1,   0,    1; % CDM
   322,  1,        31,  2, 100,   50,   0,   14,     0,    0,    273,     3,    3,     41,     2,      1,    1,     41,      1,   0,    2; % CDM
   323,  1,        31,  2, 150,   50,   0,   14,     0,    0,    273,     4,    3,     41,     2,      1,    1,     41,      1,   0,    3; % CDM
   324,  1,        31,  1, 200,   50,   0,   14,     0,    0,    273,     5,    3,     41,     2,      1,    1,     41,      1,   0,    4; % CDM
   325,  1,        31,  1, 250,   23,   0,   14,     0,    0,    273,     6,    3,     41,     2,      1,    1,     41,      1,   0,    5; % CDM
   % slot 23
   326,  1,        11,  1,  12,  249,  10,    4,     0,    0,    273,    10,    0,     41,     0,      1,    0,     41,      1,   0,    0; % CDM
   % slot 24
   327,  1,        31,  2,   0,   50,   0,   14,     0,    0,    273,     1,    3,     41,     2,      1,    1,     41,      1,   0,    0; % CDM
   328,  1,        30,  2,  50,   50,   0,   14,     0,    0,    273,     2,    2,     41,     2,      1,    1,     41,      1,   0,    1; % CDM
   329,  1,        30,  2, 100,   50,   0,   14,     0,    0,    273,     3,    2,     41,     2,      1,    1,     41,      1,   0,    2; % CDM
   330,  1,        30,  2, 150,   50,   0,   14,     0,    0,    273,     4,    2,     41,     2,      1,    1,     41,      1,   0,    3; % CDM
   331,  1,        30,  1, 200,   50,   0,   14,     0,    0,    273,     5,    2,     41,     2,      1,    1,     41,      1,   0,    4; % CDM
   332,  1,        30,  1, 250,   23,   0,   14,     0,    0,    273,     6,    2,     41,     2,      1,    1,     41,      1,   0,    5; % CDM
   % slot 33
   333,  1,        19,  1,   0,  273,  10,    4,     0,    0,    273,    11,    0,     41,     0,      1,    0,     41,      1,   0,    0; % CDM
   % slot 34
   334,  1,         4,  1,   4,    4,   0,   14,     0,    0,    273,    12,    0,     41,     2,      1,    2,     41,      1,   0,    0; % CDM
   335,  1,        20,  1,   8,  261,   0,   14,     0,    0,    273,    13,    0,     41,     2,      1,    1,     41,      1,   0,    1; % CDM
   % slot 35
   336,  1,        31,  2,   4,   46,   0,   14,     0,    0,    273,     1,    3,     41,     2,      1,    1,     41,      1,   0,    0; % CDM
   337,  1,        30,  2,  50,   50,   0,   14,     0,    0,    273,     2,    1,     41,     2,      1,    1,     41,      1,   0,    1; % CDM
   338,  1,        30,  2, 100,   50,   0,   14,     0,    0,    273,     3,    1,     41,     2,      1,    1,     41,      1,   0,    2; % CDM
   339,  1,        30,  2, 150,   50,   0,   14,     0,    0,    273,     4,    1,     41,     2,      1,    1,     41,      1,   0,    3; % CDM
   340,  1,        30,  1, 200,   50,   0,   14,     0,    0,    273,     5,    1,     41,     2,      1,    1,     41,      1,   0,    4; % CDM
   341,  1,        30,  1, 250,   19,   0,   14,     0,    0,    273,     6,    1,     41,     2,      1,    1,     41,      1,   0,    5; % CDM

   % F08 270 PRBs, 6 UEGs, 45 PRB for each UE, OTA (addDmrs), MCS 0
   342,  1,        0,  2,  1,    45, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   343,  1,        0,  2,  46,   45, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   344,  1,        0,  2,  91,   45, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   345,  1,        0,  2,  136,  45, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   346,  1,        0,  2,  181,  45, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   347,  1,        0,  2,  226,  45, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 peak, 222 PRBs, 6 UEGs, 37 PRB for each UE (with PRACH + PUCCH) , MCS 0
   348,  1,        0,  2,  1,    37, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   349,  1,        0,  2,  38,   37, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   350,  1,        0,  2,  75,   37, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   351,  1,        0,  2,  112,  37, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   352,  1,        0,  2,  149,  37, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   353,  1,        0,  2,  186,  37, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 270 PRBs, 6 UEGs, 45 PRB for each UE, OTA (addDmrs), MCS 1
   354,  1,        1,  2,  1,    45, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   355,  1,        1,  2,  46,   45, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   356,  1,        1,  2,  91,   45, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   357,  1,        1,  2,  136,  45, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   358,  1,        1,  2,  181,  45, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   359,  1,        1,  2,  226,  45, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 peak, 222 PRBs, 6 UEGs, 37 PRB for each UE (with PRACH + PUCCH) , MCS 1
   360,  1,        1,  2,  1,    37, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   361,  1,        1,  2,  38,   37, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   362,  1,        1,  2,  75,   37, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   363,  1,        1,  2,  112,  37, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   364,  1,        1,  2,  149,  37, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   365,  1,        1,  2,  186,  37, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 132 PRBs, 6 UEGs, 22 PRB for each UE, OTA (addDmrs), MCS 0
   366,  1,        0,  2,  1,    22, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   367,  1,        0,  2,  23,   22, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   368,  1,        0,  2,  45,   22, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   369,  1,        0,  2,  67,   22, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   370,  1,        0,  2,  89,   22, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   371,  1,        0,  2,  111,  22, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 84 PRBs, 6 UEGs, 14 PRB for each UE (with PRACH + PUCCH), MCS 0
   372,  1,        0,  2,  1,    14, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   373,  1,        0,  2,  15,   14, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   374,  1,        0,  2,  29,   14, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   375,  1,        0,  2,  43,   14, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   376,  1,        0,  2,  57,   14, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   377,  1,        0,  2,  71,   14, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 132 PRBs, 6 UEGs, 22 PRB for each UE, OTA (addDmrs), MCS 1
   378,  1,        1,  2,  1,    22, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   379,  1,        1,  2,  23,   22, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   380,  1,        1,  2,  45,   22, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   381,  1,        1,  2,  67,   22, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   382,  1,        1,  2,  89,   22, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   383,  1,        1,  2,  111,  22, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 84 PRBs, 6 UEGs, 14 PRB for each UE (with PRACH + PUCCH), MCS 1
   384,  1,        1,  2,  1,    14, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   385,  1,        1,  2,  15,   14, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   386,  1,        1,  2,  29,   14, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   387,  1,        1,  2,  43,   14, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   388,  1,        1,  2,  57,   14, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   389,  1,        1,  2,  71,   14, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 40 MHz, 6 UEGs, 11 PRB for each UE, OTA (addDmrs)
   390,  1,        27,  2,  1+0*11,   11, 0,     14,     0,    0,    106,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   391,  1,        27,  2,  1+1*11,   11, 0,     14,     0,    0,    106,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   392,  1,        27,  2,  1+2*11,   11, 0,     14,     0,    0,    106,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   393,  1,        27,  2,  1+3*11,   11, 0,     14,     0,    0,    106,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   394,  1,        27,  2,  1+4*11,   11, 0,     14,     0,    0,    106,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   395,  1,        27,  2,  1+5*11,   11, 0,     14,     0,    0,    106,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 80 MHz, 6 UEGs, 30 PRB for each UE, OTA (addDmrs)
   396,  1,        27,  2,  1+0*30,   30, 0,     14,     0,    0,    217,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   397,  1,        27,  2,  1+1*30,   30, 0,     14,     0,    0,    217,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   398,  1,        27,  2,  1+2*30,   30, 0,     14,     0,    0,    217,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   399,  1,        27,  2,  1+3*30,   30, 0,     14,     0,    0,    217,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   400,  1,        27,  2,  1+4*30,   30, 0,     14,     0,    0,    217,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   401,  1,        27,  2,  1+5*30,   30, 0,     14,     0,    0,    217,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 80 MHz, 6 UEGs, 11 PRB for each UE, OTA (addDmrs), with 3 PRACH
   402,  1,        27,  2,  1+0*11,   11, 0,     14,     0,    0,    217,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   403,  1,        27,  2,  1+1*11,   11, 0,     14,     0,    0,    217,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   404,  1,        27,  2,  1+2*11,   11, 0,     14,     0,    0,    217,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   405,  1,        27,  2,  1+3*11,   11, 0,     14,     0,    0,    217,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   406,  1,        27,  2,  1+4*11,   11, 0,     14,     0,    0,    217,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   407,  1,        27,  2,  1+5*11,   11, 0,     14,     0,    0,    217,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 30 MHz, 6 UEGs, 6 PRB for each UE, OTA (addDmrs), with 3 PRACH
   408,  1,        27,  2,  1+0*6,   6, 0,     14,     0,    0,    78,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   409,  1,        27,  2,  1+1*6,   6, 0,     14,     0,    0,    78,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   410,  1,        27,  2,  1+2*6,   6, 0,     14,     0,    0,    78,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   411,  1,        27,  2,  1+3*6,   6, 0,     14,     0,    0,    78,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   412,  1,        27,  2,  1+4*6,   6, 0,     14,     0,    0,    78,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   413,  1,        27,  2,  1+5*6,   6, 0,     14,     0,    0,    78,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 50 MHz, 6 UEGs, 16 PRB for each UE, OTA (addDmrs), with 3 PRACH
   414,  1,        27,  2,  1+0*16,   16, 0,     14,     0,    0,    133,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   415,  1,        27,  2,  1+1*16,   16, 0,     14,     0,    0,    133,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   416,  1,        27,  2,  1+2*16,   16, 0,     14,     0,    0,    133,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   417,  1,        27,  2,  1+3*16,   16, 0,     14,     0,    0,    133,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   418,  1,        27,  2,  1+4*16,   16, 0,     14,     0,    0,    133,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   419,  1,        27,  2,  1+5*16,   16, 0,     14,     0,    0,    133,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
     % F08 for different number of UEs at different cells, OTA (addPos) 
   420,  1,        2,   2,  1,    34, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   421,  1,        2,   2,  35,   34, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   422,  1,        4,   2,  69,   34, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   423,  1,        6,   2,  103,  34, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   424,  1,        6,   2,  137,  34, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   425,  1,        6,   2,  171,  34, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   426,  1,        8,   2,  205,  34, 0,     14,     0,    0,    273,    6,    0,     41,     2,      1,     1,     41,      2,   0,    6;
   427,  1,        8,   2,  239,  34, 0,     14,     0,    0,    273,    7,    0,     41,     2,      1,     1,     41,      2,   0,    7;
   % 
   428,  1,        2,   2,  1,    28, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   429,  1,        2,   2,  29,   28, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   430,  1,        4,   2,  57,   28, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   431,  1,        6,   2,  85,   28, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   432,  1,        6,   2,  113,  28, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   433,  1,        6,   2,  141,  28, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   434,  1,        8,   2,  169,  28, 0,     14,     0,    0,    273,    6,    0,     41,     2,      1,     1,     41,      2,   0,    6;
   435,  1,        8,   2,  197,  28, 0,     14,     0,    0,    273,    7,    0,     41,     2,      1,     1,     41,      2,   0,    7;
    % F08 light load, 6 UEGs, 6 PRB for each UE, OTA (addDmrs)
   436,  1,        27,  2,  1+0*6,   6, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   437,  1,        27,  2,  1+1*6,   6, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   438,  1,        27,  2,  1+2*6,   6, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   439,  1,        27,  2,  1+3*6,   6, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   440,  1,        27,  2,  1+4*6,   6, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   441,  1,        27,  2,  1+5*6,   6, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 42 PRB for each UE, OTA (addDmrs)
   442,  1,        27,  2,  18+0*42,   42, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   443,  1,        27,  2,  18+1*42,   42, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   444,  1,        27,  2,  18+2*42,   42, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   445,  1,        27,  2,  18+3*42,   42, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   446,  1,        27,  2,  18+4*42,   42, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   447,  1,        27,  2,  18+5*42,   42, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 34 PRB for each UE, OTA (addDmrs)
   448,  1,        27,  2,  18+0*34,   34, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   449,  1,        27,  2,  18+1*34,   34, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   450,  1,        27,  2,  18+2*34,   34, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   451,  1,        27,  2,  18+3*34,   34, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   452,  1,        27,  2,  18+4*34,   34, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   453,  1,        27,  2,  18+5*34,   34, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 36 PRB for each UE, OTA (addDmrs), with 3 PRACH
    454,  1,        27,  2,  18+0*36,   36, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
    455,  1,        27,  2,  18+1*36,   36, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
    456,  1,        27,  2,  18+2*36,   36, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
    457,  1,        27,  2,  18+3*36,   36, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
    458,  1,        27,  2,  18+4*36,   36, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
    459,  1,        27,  2,  18+5*36,   36, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 41 PRB for each UE, OTA (addDmrs)
    460,  1,        27,  2,  24+0*41,   41, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
    461,  1,        27,  2,  24+1*41,   41, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
    462,  1,        27,  2,  24+2*41,   41, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
    463,  1,        27,  2,  24+3*41,   41, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
    464,  1,        27,  2,  24+4*41,   41, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
    465,  1,        27,  2,  24+5*41,   41, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 100 MHz, 6 UEGs, 33 PRB for each UE, OTA (addDmrs)
   466,  1,        27,  2,  24+0*33,   33, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   467,  1,        27,  2,  24+1*33,   33, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   468,  1,        27,  2,  24+2*33,   33, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   469,  1,        27,  2,  24+3*33,   33, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   470,  1,        27,  2,  24+4*33,   33, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   471,  1,        27,  2,  24+5*33,   33, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 100 MHz, 6 UEGs, 35 PRB for each UE, OTA (addDmrs), with 3 PRACH
   472,  1,        27,  2,  24+0*35,   35, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   473,  1,        27,  2,  24+1*35,   35, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   474,  1,        27,  2,  24+2*35,   35, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   475,  1,        27,  2,  24+3*35,   35, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   476,  1,        27,  2,  24+4*35,   35, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   477,  1,        27,  2,  24+5*35,   35, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 19 PRB for each UE, OTA (addDmrs)
    478,  1,        27,  2,  18+0*19,   19, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
    479,  1,        27,  2,  18+1*19,   19, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
    480,  1,        27,  2,  18+2*19,   19, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
    481,  1,        27,  2,  18+3*19,   19, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
    482,  1,        27,  2,  18+4*19,   19, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
    483,  1,        27,  2,  18+5*19,   19, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 100 MHz, 6 UEGs, 11 PRB for each UE, OTA (addDmrs)
   484,  1,        27,  2,  18+0*11,   11, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   485,  1,        27,  2,  18+1*11,   11, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   486,  1,        27,  2,  18+2*11,   11, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   487,  1,        27,  2,  18+3*11,   11, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   488,  1,        27,  2,  18+4*11,   11, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   489,  1,        27,  2,  18+5*11,   11, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 100 MHz, 6 UEGs, 13 PRB for each UE, OTA (addDmrs), with 3 PRACH
   490,  1,        27,  2,  18+0*13,   13, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   491,  1,        27,  2,  18+1*13,   13, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   492,  1,        27,  2,  18+2*13,   13, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   493,  1,        27,  2,  18+3*13,   13, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   494,  1,        27,  2,  18+4*13,   13, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   495,  1,        27,  2,  18+5*13,   13, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 18 PRB for each UE, OTA (addDmrs)
    496,  1,        27,  2,  24+0*18,   18, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
    497,  1,        27,  2,  24+1*18,   18, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
    498,  1,        27,  2,  24+2*18,   18, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
    499,  1,        27,  2,  24+3*18,   18, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
    500,  1,        27,  2,  24+4*18,   18, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
    501,  1,        27,  2,  24+5*18,   18, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 100 MHz, 6 UEGs, 10 PRB for each UE, OTA (addDmrs)
   502,  1,        27,  2,  24+0*10,   10, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   503,  1,        27,  2,  24+1*10,   10, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   504,  1,        27,  2,  24+2*10,   10, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   505,  1,        27,  2,  24+3*10,   10, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   506,  1,        27,  2,  24+4*10,   10, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   507,  1,        27,  2,  24+5*10,   10, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 100 MHz, 6 UEGs, 12 PRB for each UE, OTA (addDmrs), with 3 PRACH
   508,  1,        27,  2,  24+0*12,   12, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   509,  1,        27,  2,  24+1*12,   12, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   510,  1,        27,  2,  24+2*12,   12, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   511,  1,        27,  2,  24+3*12,   12, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   512,  1,        27,  2,  24+4*12,   12, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   513,  1,        27,  2,  24+5*12,   12, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 45 PRB for each UE, OTA (addDmrs)
    514,  1,        27,  2,  2+0*45,   45, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
    515,  1,        27,  2,  2+1*45,   45, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
    516,  1,        27,  2,  2+2*45,   45, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
    517,  1,        27,  2,  2+3*45,   45, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
    518,  1,        27,  2,  2+4*45,   45, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
    519,  1,        27,  2,  2+5*45,   45, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 37 PRB for each UE, OTA (addDmrs)
    520,  1,        27,  2,  2+0*37,   37, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
    521,  1,        27,  2,  2+1*37,   37, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
    522,  1,        27,  2,  2+2*37,   37, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
    523,  1,        27,  2,  2+3*37,   37, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
    524,  1,        27,  2,  2+4*37,   37, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
    525,  1,        27,  2,  2+5*37,   37, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 39 PRB for each UE, OTA (addDmrs), with 3 PRACH
    526,  1,        27,  2,  2+0*39,   39, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
    527,  1,        27,  2,  2+1*39,   39, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
    528,  1,        27,  2,  2+2*39,   39, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
    529,  1,        27,  2,  2+3*39,   39, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
    530,  1,        27,  2,  2+4*39,   39, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
    531,  1,        27,  2,  2+5*39,   39, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 22 PRB for each UE, OTA (addDmrs)
    532,  1,        27,  2,  2+0*22,   22, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
    533,  1,        27,  2,  2+1*22,   22, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
    534,  1,        27,  2,  2+2*22,   22, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
    535,  1,        27,  2,  2+3*22,   22, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
    536,  1,        27,  2,  2+4*22,   22, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
    537,  1,        27,  2,  2+5*22,   22, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 100 MHz, 6 UEGs, 14 PRB for each UE, OTA (addDmrs)
   538,  1,        27,  2,  2+0*14,   14, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   539,  1,        27,  2,  2+1*14,   14, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   540,  1,        27,  2,  2+2*14,   14, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   541,  1,        27,  2,  2+3*14,   14, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   542,  1,        27,  2,  2+4*14,   14, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   543,  1,        27,  2,  2+5*14,   14, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
   % F08 100 MHz, 6 UEGs, 16 PRB for each UE, OTA (addDmrs), with 3 PRACH
   544,  1,        27,  2,  2+0*16,   16, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   545,  1,        27,  2,  2+1*16,   16, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   546,  1,        27,  2,  2+2*16,   16, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   547,  1,        27,  2,  2+3*16,   16, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   548,  1,        27,  2,  2+4*16,   16, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   549,  1,        27,  2,  2+5*16,   16, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
                                         
   % mix TV with PRACH + PUCCH + PUSCH + SRS
   % F08 peak, 222 PRBs, 6 UEGs, 37 PRB for each UE (with PRACH + PUCCH + SRS), modified from 167:172 with Nsym = 13
   550,  1,        27,  2,  1,    14, 0,     13,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   551,  1,        27,  2,  15,   14, 0,     13,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   552,  1,        27,  2,  29,   14, 0,     13,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   553,  1,        27,  2,  43,   14, 0,     13,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   554,  1,        27,  2,  57,   14, 0,     13,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   555,  1,        27,  2,  71,   14, 0,     13,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;

   % mix TV with PRACH + PUCCH + PUSCH + SRS
   % F08 peak, 222 PRBs, 6 UEGs, 37 PRB for each UE (with PRACH + PUCCH + SRS), modified from 167:172 with Nsym = 12
   556,  1,        27,  2,  1,    14, 0,     12,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
   557,  1,        27,  2,  15,   14, 0,     12,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
   558,  1,        27,  2,  29,   14, 0,     12,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
   559,  1,        27,  2,  43,   14, 0,     12,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
   560,  1,        27,  2,  57,   14, 0,     12,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
   561,  1,        27,  2,  71,   14, 0,     12,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
 
   % TC#   mcsTable  mcs  nl  rb0     Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg 
 
   % F08 100 MHz, 6 UEGs, 45 PRB for each UE, S slot
    562,  1,        27,  2,  0+0*45,   45, 10,     4,     0,    0,    273,    0,    0,     41,     0,      1,     0,     41,      2,   0,    0;
    563,  1,        27,  2,  0+1*45,   45, 10,     4,     0,    0,    273,    1,    0,     41,     0,      1,     0,     41,      2,   0,    1;
    564,  1,        27,  2,  0+2*45,   45, 10,     4,     0,    0,    273,    2,    0,     41,     0,      1,     0,     41,      2,   0,    2;
    565,  1,        27,  2,  0+3*45,   45, 10,     4,     0,    0,    273,    3,    0,     41,     0,      1,     0,     41,      2,   0,    3;
    566,  1,        27,  2,  0+4*45,   45, 10,     4,     0,    0,    273,    4,    0,     41,     0,      1,     0,     41,      2,   0,    4;
    567,  1,        27,  2,  0+5*45,   45, 10,     4,     0,    0,    273,    5,    0,     41,     0,      1,     0,     41,      2,   0,    5;

    % F08 100 MHz, 6 UEGs, 22 PRB for each UE, S slot
    568,  1,        27,  2,  0+0*22,   22, 10,     4,     0,    0,    273,    0,    0,     41,     0,      1,     0,     41,      2,   0,    0;
    569,  1,        27,  2,  0+1*22,   22, 10,     4,     0,    0,    273,    1,    0,     41,     0,      1,     0,     41,      2,   0,    1;
    570,  1,        27,  2,  0+2*22,   22, 10,     4,     0,    0,    273,    2,    0,     41,     0,      1,     0,     41,      2,   0,    2;
    571,  1,        27,  2,  0+3*22,   22, 10,     4,     0,    0,    273,    3,    0,     41,     0,      1,     0,     41,      2,   0,    3;
    572,  1,        27,  2,  0+4*22,   22, 10,     4,     0,    0,    273,    4,    0,     41,     0,      1,     0,     41,      2,   0,    4;
    573,  1,        27,  2,  0+5*22,   22, 10,     4,     0,    0,    273,    5,    0,     41,     0,      1,     0,     41,      2,   0,    5;
     
     % F08 100 MHz, 6 UEGs, 42 PRB for each UE, 1 dmrs
    574,  1,        27,  2,  18+0*42,   42, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
    575,  1,        27,  2,  18+1*42,   42, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    1;
    576,  1,        27,  2,  18+2*42,   42, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    2;
    577,  1,        27,  2,  18+3*42,   42, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     0,     41,      2,   0,    3;
    578,  1,        27,  2,  18+4*42,   42, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   0,    4;
    579,  1,        27,  2,  18+5*42,   42, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 34 PRB for each UE, 1 dmrs
    580,  1,        27,  2,  18+0*34,   34, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
    581,  1,        27,  2,  18+1*34,   34, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    1;
    582,  1,        27,  2,  18+2*34,   34, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    2;
    583,  1,        27,  2,  18+3*34,   34, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     0,     41,      2,   0,    3;
    584,  1,        27,  2,  18+4*34,   34, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   0,    4;
    585,  1,        27,  2,  18+5*34,   34, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 36 PRB for each UE, 1 dmrs, with 3 PRACH
    586,  1,        27,  2,  18+0*36,   36, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
    587,  1,        27,  2,  18+1*36,   36, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    1;
    588,  1,        27,  2,  18+2*36,   36, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    2;
    589,  1,        27,  2,  18+3*36,   36, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     0,     41,      2,   0,    3;
    590,  1,        27,  2,  18+4*36,   36, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   0,    4;
    591,  1,        27,  2,  18+5*36,   36, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 19 PRB for each UE, 1 dmrs
    592,  1,        27,  2,  18+0*19,   19, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
    593,  1,        27,  2,  18+1*19,   19, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    1;
    594,  1,        27,  2,  18+2*19,   19, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    2;
    595,  1,        27,  2,  18+3*19,   19, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     0,     41,      2,   0,    3;
    596,  1,        27,  2,  18+4*19,   19, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   0,    4;
    597,  1,        27,  2,  18+5*19,   19, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 11 PRB for each UE, 1 dmrs
    598,  1,        27,  2,  18+0*11,   11, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
    599,  1,        27,  2,  18+1*11,   11, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    1;
    600,  1,        27,  2,  18+2*11,   11, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    2;
    601,  1,        27,  2,  18+3*11,   11, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     0,     41,      2,   0,    3;
    602,  1,        27,  2,  18+4*11,   11, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   0,    4;
    603,  1,        27,  2,  18+5*11,   11, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 13 PRB for each UE, 1 dmrs, with 3 PRACH
    604,  1,        27,  2,  18+0*13,   13, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     0,     41,      2,   0,    0;
    605,  1,        27,  2,  18+1*13,   13, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     0,     41,      2,   0,    1;
    606,  1,        27,  2,  18+2*13,   13, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     0,     41,      2,   0,    2;
    607,  1,        27,  2,  18+3*13,   13, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     0,     41,      2,   0,    3;
    608,  1,        27,  2,  18+4*13,   13, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     0,     41,      2,   0,    4;
    609,  1,        27,  2,  18+5*13,   13, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     0,     41,      2,   0,    5;
    % 64TR
 % TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg        

   610,  1,         1,  2, 100,   40,   0,   12,     0,    0,    273,     1,    0,     41,     2,      2,    1,     41,      2,   0,    0; 
   611,  1,         2,  2, 100,   40,   0,   12,     0,    0,    273,     2,    0,     41,     2,      2,    1,     41,      2,   2,    0; 
   612,  1,         1,  2, 100,   40,   0,   12,     0,    0,    273,     3,    0,     41,     2,      2,    1,     41,      2,   4,    0; 
   613,  1,         2,  2, 100,   40,   0,   12,     0,    0,    273,     4,    0,     41,     2,      2,    1,     41,      2,   6,    0; 
   614,  1,        10,  1, 150,   40,   0,   12,     0,    0,    273,     5,    0,     41,     2,      2,    1,     41,      2,   0,    1; 
   615,  1,        19,  1, 200,   40,   0,   12,     0,    0,    273,     7,    0,     41,     2,      2,    1,     41,      2,   0,    2;    
    % 64TR 7, 6 UEGs, 16 UEs, OTA (addDmrs) 
    616,  1,        19,  2,  4,   23, 0,     14,     0,    0,    273,    1,    0,     41,     2,      2,     1,     41,      2,   0,    0;
    617,  1,        19,  2,  4,   23, 0,     14,     0,    0,    273,    2,    0,     41,     2,      2,     1,     41,      2,   2,    0;
    618,  1,        19,  2,  4,   23, 0,     14,     0,    0,    273,    3,    0,     41,     2,      2,     1,     41,      2,   4,    0;
    619,  1,        19,  2,  4,   23, 0,     14,     0,    0,    273,    4,    0,     41,     2,      2,     1,     41,      2,   6,    0;
    620,  1,        10,  2,  27,   21, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    1;
    621,  1,        10,  2,  27,   21, 0,     14,     0,    0,    273,    6,    0,     41,     2,      1,     1,     41,      2,   2,    1;
    622,  1,        10,  2,  48,   21, 0,     14,     0,    0,    273,    7,    0,     41,     2,      1,     1,     41,      2,   0,    2;
    623,  1,        10,  2,  48,   21, 0,     14,     0,    0,    273,    8,    0,     41,     2,      1,     1,     41,      2,   2,    2;
    624,  1,        10,  2,  69,   21, 0,     14,     0,    0,    273,    9,    0,     41,     2,      1,     1,     41,      2,   0,    3;
    625,  1,        10,  2,  69,   21, 0,     14,     0,    0,    273,    10,    0,     41,     2,      1,     1,     41,      2,   2,    3;
    626,  1,        10,  2,  90,   21, 0,     14,     0,    0,    273,    11,    0,     41,     2,      1,     1,     41,      2,   0,    4;
    627,  1,        10,  2,  90,   21, 0,     14,     0,    0,    273,    12,    0,     41,     2,      1,     1,     41,      2,   2,    4;
    628,  1,        4,  1,  111,   11, 0,     14,     0,    0,    273,    13,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    629,  1,        4,  1,  111,   11, 0,     14,     0,    0,    273,    14,    0,     41,     2,      1,     1,     41,      2,   1,    5;
    630,  1,        4,  1,  122,   11, 0,     14,     0,    0,    273,    15,    0,     41,     2,      1,     1,     41,      2,   0,    6;
    631,  1,        4,  1,  122,   11, 0,     14,     0,    0,    273,    16,    0,     41,     2,      1,     1,     41,      2,   1,    6;
    % 64TR 7, 6 UEGs, 16 UEs, OTA (addDmrs), with 4 PRACH 
    632,  1,        19,  2,  4,   16, 0,     14,     0,    0,    273,    1,    0,     41,     2,      2,     1,     41,      2,   0,    0;
    633,  1,        19,  2,  4,   16, 0,     14,     0,    0,    273,    2,    0,     41,     2,      2,     1,     41,      2,   2,    0;
    634,  1,        19,  2,  4,   16, 0,     14,     0,    0,    273,    3,    0,     41,     2,      2,     1,     41,      2,   4,    0;
    635,  1,        19,  2,  4,   16, 0,     14,     0,    0,    273,    4,    0,     41,     2,      2,     1,     41,      2,   6,    0;
    636,  1,        10,  2,  20,   14, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    1;
    637,  1,        10,  2,  20,   14, 0,     14,     0,    0,    273,    6,    0,     41,     2,      1,     1,     41,      2,   2,    1;
    638,  1,        10,  2,  34,   14, 0,     14,     0,    0,    273,    7,    0,     41,     2,      1,     1,     41,      2,   0,    2;
    639,  1,        10,  2,  34,   14, 0,     14,     0,    0,    273,    8,    0,     41,     2,      1,     1,     41,      2,   2,    2;
    640,  1,        10,  2,  48,   14, 0,     14,     0,    0,    273,    9,    0,     41,     2,      1,     1,     41,      2,   0,    3;
    641,  1,        10,  2,  48,   14, 0,     14,     0,    0,    273,    10,    0,     41,     2,      1,     1,     41,      2,   2,    3;
    642,  1,        10,  2,  62,   14, 0,     14,     0,    0,    273,    11,    0,     41,     2,      1,     1,     41,      2,   0,    4;
    643,  1,        10,  2,  62,   14, 0,     14,     0,    0,    273,    12,    0,     41,     2,      1,     1,     41,      2,   2,    4;
    644,  1,        4,  1,  76,   6, 0,     14,     0,    0,    273,    13,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    645,  1,        4,  1,  76,   6, 0,     14,     0,    0,    273,    14,    0,     41,     2,      1,     1,     41,      2,   1,    5;
    646,  1,        4,  1,  82,   6, 0,     14,     0,    0,    273,    15,    0,     41,     2,      1,     1,     41,      2,   0,    6;
    647,  1,        4,  1,  82,   6, 0,     14,     0,    0,    273,    16,    0,     41,     2,      1,     1,     41,      2,   1,    6;
    % 64TR 7, 6 UEGs, 16 UEs, OTA (addDmrs), with 3 PRACH 
    648,  1,        19,  2,  4,   18, 0,     14,     0,    0,    273,    1,    0,     41,     2,      2,     1,     41,      2,   0,    0;
    649,  1,        19,  2,  4,   18, 0,     14,     0,    0,    273,    2,    0,     41,     2,      2,     1,     41,      2,   2,    0;
    650,  1,        19,  2,  4,   18, 0,     14,     0,    0,    273,    3,    0,     41,     2,      2,     1,     41,      2,   4,    0;
    651,  1,        19,  2,  4,   18, 0,     14,     0,    0,    273,    4,    0,     41,     2,      2,     1,     41,      2,   6,    0;
    652,  1,        10,  2,  22,   16, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    1;
    653,  1,        10,  2,  22,   16, 0,     14,     0,    0,    273,    6,    0,     41,     2,      1,     1,     41,      2,   2,    1;
    654,  1,        10,  2,  38,   16, 0,     14,     0,    0,    273,    7,    0,     41,     2,      1,     1,     41,      2,   0,    2;
    655,  1,        10,  2,  38,   16, 0,     14,     0,    0,    273,    8,    0,     41,     2,      1,     1,     41,      2,   2,    2;
    656,  1,        10,  2,  54,   16, 0,     14,     0,    0,    273,    9,    0,     41,     2,      1,     1,     41,      2,   0,    3;
    657,  1,        10,  2,  54,   16, 0,     14,     0,    0,    273,    10,    0,     41,     2,      1,     1,     41,      2,   2,    3;
    658,  1,        10,  2,  70,   16, 0,     14,     0,    0,    273,    11,    0,     41,     2,      1,     1,     41,      2,   0,    4;
    659,  1,        10,  2,  70,   16, 0,     14,     0,    0,    273,    12,    0,     41,     2,      1,     1,     41,      2,   2,    4;
    660,  1,        4,  1,  86,   6, 0,     14,     0,    0,    273,    13,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    661,  1,        4,  1,  86,   6, 0,     14,     0,    0,    273,    14,    0,     41,     2,      1,     1,     41,      2,   1,    5;
    662,  1,        4,  1,  92,   6, 0,     14,     0,    0,    273,    15,    0,     41,     2,      1,     1,     41,      2,   0,    6;
    663,  1,        4,  1,  92,   6, 0,     14,     0,    0,    273,    16,    0,     41,     2,      1,     1,     41,      2,   1,    6;
       % 64TR test case
% TC#   mcsTable  mcs  nl  rb0     Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg 
    664,  1,        27,  1,  40,   40,   0,   14,     0,    0,    273,     1,    0,     41,     2,      1,    1,     41,      2,   0,    0; 
    665,  1,        26,  2,  80,   40,   0,   14,     0,    0,    273,     2,    0,     41,     2,      1,    1,     41,      2,   0,    1; 
    666,  1,        25,  1, 120,   40,   0,   14,     0,    0,    273,     3,    0,     41,     2,      1,    1,     41,      2,   0,    2; 
    667,  1,        24,  2, 160,   40,   0,   14,     0,    0,    273,     4,    0,     41,     2,      1,    1,     41,      2,   0,    3; 
    668,  1,        10,  2, 200,   40,   0,   14,     0,    0,    273,     5,    0,     41,     2,      1,    1,     41,      2,   0,    4; 
    669,  1,        19,  2, 240,   20,   0,   14,     0,    0,    273,     6,    0,     41,     2,      1,    1,     41,      2,   0,    5;    

    % 64TR 1 UEGs, 4 UEs, OTA (addDmrs) 
    670,  1,        19,  2,  4,   269, 0,     14,     0,    0,    273,    1,    0,     41,     2,      2,     1,     41,      2,   0,    0;
    671,  1,        19,  2,  4,   269, 0,     14,     0,    0,    273,    2,    0,     41,     2,      2,     1,     41,      2,   2,    0;
    672,  1,        19,  2,  4,   269, 0,     14,     0,    0,    273,    3,    0,     41,     2,      2,     1,     41,      2,   4,    0;
    673,  1,        19,  2,  4,   269, 0,     14,     0,    0,    273,    4,    0,     41,     2,      2,     1,     41,      2,   6,    0;
    % 64TR 1 UEGs, 4 UEs, OTA (addDmrs), with 4 PRACH 
    674,  1,        19,  2,  4,   221, 0,     14,     0,    0,    273,    1,    0,     41,     2,      2,     1,     41,      2,   0,    0;
    675,  1,        19,  2,  4,   221, 0,     14,     0,    0,    273,    2,    0,     41,     2,      2,     1,     41,      2,   2,    0;
    676,  1,        19,  2,  4,   221, 0,     14,     0,    0,    273,    3,    0,     41,     2,      2,     1,     41,      2,   4,    0;
    677,  1,        19,  2,  4,   221, 0,     14,     0,    0,    273,    4,    0,     41,     2,      2,     1,     41,      2,   6,    0;
    % 64TR 1 UEGs, 4 UEs, OTA (addDmrs), with 3 PRACH 
    678,  1,        19,  2,  4,   233, 0,     14,     0,    0,    273,    1,    0,     41,     2,      2,     1,     41,      2,   0,    0;
    679,  1,        19,  2,  4,   233, 0,     14,     0,    0,    273,    2,    0,     41,     2,      2,     1,     41,      2,   2,    0;
    680,  1,        19,  2,  4,   233, 0,     14,     0,    0,    273,    3,    0,     41,     2,      2,     1,     41,      2,   4,    0;
    681,  1,        19,  2,  4,   233, 0,     14,     0,    0,    273,    4,    0,     41,     2,      2,     1,     41,      2,   6,    0;

    % 64TR MU-MIMO, 3 peak cells
% TC#   mcsTable  mcs  nl  rb0     Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg 
    682,  0,        27,  1,  0,   273, 0,       14,     0,    0,    273,    1,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    683,  0,        27,  1,  0,   273, 0,       14,     0,    0,    273,    2,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    684,  0,        27,  1,  0,   273, 0,       14,     0,    0,    273,    3,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    685,  0,        27,  1,  0,   273, 0,       14,     0,    0,    273,    4,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    686,  0,        27,  1,  0,   273, 0,       14,     0,    0,    273,    5,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    687,  0,        27,  1,  0,   273, 0,       14,     0,    0,    273,    6,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    688,  0,        27,  1,  0,   273, 0,       14,     0,    0,    273,    7,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    689,  0,        27,  1,  0,   273, 0,       14,     0,    0,    273,    8,    0,     41,     2,      2,     1,     41,      2,   7,   0;

    % 4TR PUSCH only U slot, CP-OFDM, total 273 PUSCH PRBs
% TC#   mcsTable  mcs  nl  rb0     Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    690,  1,        27,  2,  0,    45, 0,       14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    691,  1,        27,  2,  45,   45, 0,       14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    692,  1,        27,  2,  90,   45, 0,       14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    693,  1,        27,  2,  135,  46, 0,       14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    694,  1,        27,  2,  181,  46, 0,       14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    695,  1,        27,  2,  227,  46, 0,       14,     0,    0,    273,    6,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % 4TR PUSCH only U slot, DFT-s-OFDM, total 273 PUSCH PRBs
    696,  1,        27,  1,  0,    45, 0,       14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    697,  1,        27,  1,  45,   45, 0,       14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    698,  1,        27,  1,  90,   45, 0,       14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    699,  1,        27,  1,  135,  45, 0,       14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    700,  1,        27,  1,  180,  45, 0,       14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    701,  1,        27,  1,  225,  48, 0,       14,     0,    0,    273,    6,    0,     41,     2,      1,     1,     41,      2,   0,   5;

% TC#   mcsTable  mcs  nl  rb0     Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg 
    % F08 100 MHz, 6 UEGs, 36/45/48 PRB for each UE, OTA (addDmrs), CP-OFDM/DFT-s-OFDM, total 270 PUSCH PRBs
    702,  1,        27,  2,    3,   36, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
    703,  1,        27,  2,   39,   45, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
    704,  1,        27,  2,   84,   45, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
    705,  1,        27,  2,  129,   48, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
    706,  1,        27,  2,  177,   48, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
    707,  1,        27,  2,  225,   48, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 30/36/45 PRB for each UE, OTA (addDmrs), CP-OFDM/DFT-s-OFDM, total 219 PUSCH PRBs
    708,  1,        27,  2,    6,   30, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
    709,  1,        27,  2,   36,   36, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
    710,  1,        27,  2,   72,   36, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
    711,  1,        27,  2,  108,   36, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
    712,  1,        27,  2,  144,   36, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
    713,  1,        27,  2,  180,   45, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 30/36/40/45 PRB for each UE, OTA (addDmrs), with 3 PRACH, CP-OFDM/DFT-s-OFDM, total 231 PUSCH PRBs
    714,  1,        27,  2,     6,     30, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
    715,  1,        27,  2,    36,     36, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
    716,  1,        27,  2,    72,     40, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
    717,  1,        27,  2,   112,     40, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
    718,  1,        27,  2,   152,     40, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
    719,  1,        27,  2,   192,     45, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
% TC#   mcsTable  mcs  nl  rb0     Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    % F08 100 MHz, 6 UEGs, 20/24 PRB for each UE, CP-OFDM/DFT-s-OFDM, total 136 PRBs
    720,  1,        27,  2,  0*20,   20, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
    721,  1,        27,  2,  1*20,   20, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
    722,  1,        27,  2,  2*20,   24, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
    723,  1,        27,  2,  3*20+4,   24, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
    724,  1,        27,  2,  4*20+8,   24, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
    725,  1,        27,  2,  5*20+12,   24, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 16/20/24/25 PRB for each UE, CP-OFDM/DFT-s-OFDM, total 133 PRBs PUSCH
    726,  1,        27,  2,    3,   16, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
    727,  1,        27,  2,   19,   20, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
    728,  1,        27,  2,   39,   24, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
    729,  1,        27,  2,   63,   24, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
    730,  1,        27,  2,   87,   24, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
    731,  1,        27,  2,  111,   25, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % F08 100 MHz, 6 UEGs, 18/20 PRB for each UE, CP-OFDM/DFT-s-OFDM, total 118 PRBs PUSCH
    732,  1,        27,  2,  6+0*20,     18, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
    733,  1,        27,  2,  6+1*20-2,   20, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
    734,  1,        27,  2,  6+2*20-2,   20, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
    735,  1,        27,  2,  6+3*20-2,   20, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
    736,  1,        27,  2,  6+4*20-2,   20, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
    737,  1,        27,  2,  6+5*20-2,   20, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % 64TR MU-MIMO column G
% TC#   mcsTable  mcs  nl  rb0     Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    738,  1,        27,  1, 22,   251, 0,       14,     0,    0,    273,    7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    739,  1,        27,  1, 22,   251, 0,       14,     0,    0,    273,    8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    740,  1,        27,  1, 22,   251, 0,       14,     0,    0,    273,    9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    741,  1,        27,  1, 22,   251, 0,       14,     0,    0,    273,   10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    742,  1,        27,  1, 22,   251, 0,       14,     0,    0,    273,   11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    743,  1,        27,  1, 22,   251, 0,       14,     0,    0,    273,   12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    744,  1,        27,  1, 22,   251, 0,       14,     0,    0,    273,   13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    745,  1,        27,  1, 22,   251, 0,       14,     0,    0,    273,   14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % different BWs...
    746,  1,        27,  1,  6,   219, 0,       14,     0,    0,    273,   15,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    747,  1,        27,  1,  6,   219, 0,       14,     0,    0,    273,   16,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    748,  1,        27,  1,  6,   219, 0,       14,     0,    0,    273,   17,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    749,  1,        27,  1,  6,   219, 0,       14,     0,    0,    273,   18,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    750,  1,        27,  1,  6,   219, 0,       14,     0,    0,    273,   19,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    751,  1,        27,  1,  6,   219, 0,       14,     0,    0,    273,   20,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    752,  1,        27,  1,  6,   219, 0,       14,     0,    0,    273,   21,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    753,  1,        27,  1,  6,   219, 0,       14,     0,    0,    273,   22,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % SU-MIMO
    754,  1,        27,  1,  22,   58,   0,   14,     0,    0,    273,     1,    0,     41,     2,      1,    1,     41,      2,   0,    0; 
    755,  1,        27,  1,   6,   74,   0,   14,     0,    0,    273,     1,    0,     41,     2,      1,    1,     41,      2,   0,    0; 
    756,  1,        26,  2,  80,   40,   0,   14,     0,    0,    273,     2,    0,     41,     2,      1,    1,     41,      2,   0,    1; 
    757,  1,        25,  1, 120,   40,   0,   14,     0,    0,    273,     3,    0,     41,     2,      1,    1,     41,      2,   0,    2; 
    758,  1,        24,  2, 160,   40,   0,   14,     0,    0,    273,     4,    0,     41,     2,      1,    1,     41,      2,   0,    3; 
    759,  1,        10,  2, 200,   10,   0,   14,     0,    0,    273,     5,    0,     41,     2,      1,    1,     41,      2,   0,    4; 
    760,  1,        19,  2, 210,   14,   0,   14,     0,    0,    273,     6,    0,     41,     2,      1,    1,     41,      2,   0,    5; 
    % F08 100 MHz, 6 UEGs, 40/45 PRB for each UE, with CP-OFDM/DFT-s-OFDM, total 255 PRBs PUSCH
    761,  1,        27,  2,    6,   40, 0,     14,     0,    0,    273,    0,    0,     41,     2,      1,     1,     41,      2,   0,    0;
    762,  1,        27,  2,   46,   40, 0,     14,     0,    0,    273,    1,    0,     41,     2,      1,     1,     41,      2,   0,    1;
    763,  1,        27,  2,   86,   40, 0,     14,     0,    0,    273,    2,    0,     41,     2,      1,     1,     41,      2,   0,    2;
    764,  1,        27,  2,  126,   45, 0,     14,     0,    0,    273,    3,    0,     41,     2,      1,     1,     41,      2,   0,    3;
    765,  1,        27,  2,  171,   45, 0,     14,     0,    0,    273,    4,    0,     41,     2,      1,     1,     41,      2,   0,    4;
    766,  1,        27,  2,  216,   45, 0,     14,     0,    0,    273,    5,    0,     41,     2,      1,     1,     41,      2,   0,    5;
    % 64TR MU-MIMO column G
% TC#   mcsTable  mcs  nl  rb0     Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    767,  1,        27,  1,  26,   247, 0,       12,     0,    0,    273,    7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    768,  1,        27,  1,  26,   247, 0,       12,     0,    0,    273,    8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    769,  1,        27,  1,  26,   247, 0,       12,     0,    0,    273,    9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    770,  1,        27,  1,  26,   247, 0,       12,     0,    0,    273,   10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    771,  1,        27,  1,  26,   247, 0,       12,     0,    0,    273,   11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    772,  1,        27,  1,  26,   247, 0,       12,     0,    0,    273,   12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    773,  1,        27,  1,  26,   247, 0,       12,     0,    0,    273,   13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    774,  1,        27,  1,  26,   247, 0,       12,     0,    0,    273,   14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % different BWs...
    775,  1,        27,  1,  10,   215, 0,       12,     0,    0,    273,   15,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    776,  1,        27,  1,  10,   215, 0,       12,     0,    0,    273,   16,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    777,  1,        27,  1,  10,   215, 0,       12,     0,    0,    273,   17,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    778,  1,        27,  1,  10,   215, 0,       12,     0,    0,    273,   18,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    779,  1,        27,  1,  10,   215, 0,       12,     0,    0,    273,   19,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    780,  1,        27,  1,  10,   215, 0,       12,     0,    0,    273,   20,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    781,  1,        27,  1,  10,   215, 0,       12,     0,    0,    273,   21,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    782,  1,        27,  1,  10,   215, 0,       12,     0,    0,    273,   22,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % 64TR MU-MIMO column G, 40 MHz
% TC#   mcsTable  mcs  nl  rb0     Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    783,  1,        27,  1, 22,   84, 0,       14,     0,    0,    106,    7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    784,  1,        27,  1, 22,   84, 0,       14,     0,    0,    106,    8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    785,  1,        27,  1, 22,   84, 0,       14,     0,    0,    106,    9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    786,  1,        27,  1, 22,   84, 0,       14,     0,    0,    106,   10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    787,  1,        27,  1, 22,   84, 0,       14,     0,    0,    106,   11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    788,  1,        27,  1, 22,   84, 0,       14,     0,    0,    106,   12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    789,  1,        27,  1, 22,   84, 0,       14,     0,    0,    106,   13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    790,  1,        27,  1, 22,   84, 0,       14,     0,    0,    106,   14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % different BWs...
    791,  1,        27,  1,  6,   52, 0,       14,     0,    0,    106,   15,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    792,  1,        27,  1,  6,   52, 0,       14,     0,    0,    106,   16,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    793,  1,        27,  1,  6,   52, 0,       14,     0,    0,    106,   17,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    794,  1,        27,  1,  6,   52, 0,       14,     0,    0,    106,   18,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    795,  1,        27,  1,  6,   52, 0,       14,     0,    0,    106,   19,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    796,  1,        27,  1,  6,   52, 0,       14,     0,    0,    106,   20,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    797,  1,        27,  1,  6,   52, 0,       14,     0,    0,    106,   21,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    798,  1,        27,  1,  6,   52, 0,       14,     0,    0,    106,   22,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % 64TR MU-MIMO column G w/ reduced layers, 4 UEGs, 2 UEs per UEG, 1 layer per UE
% TC#   mcsTable  mcs  nl  rb0     Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    799,  1,        27,  1,  22,    62, 0,       14,     0,    0,    273,    7,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    800,  1,        27,  1,  22,    62, 0,       14,     0,    0,    273,    8,    0,     41,     2,      1,     1,     41,      2,   1,   0;
    801,  1,        27,  1,  84,    62, 0,       14,     0,    0,    273,    9,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    802,  1,        27,  1,  84,    62, 0,       14,     0,    0,    273,   10,    0,     41,     2,      1,     1,     41,      2,   1,   1;
    803,  1,        27,  1, 146,    62, 0,       14,     0,    0,    273,   11,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    804,  1,        27,  1, 146,    62, 0,       14,     0,    0,    273,   12,    0,     41,     2,      1,     1,     41,      2,   1,   2;
    805,  1,        27,  1, 208,    65, 0,       14,     0,    0,    273,   13,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    806,  1,        27,  1, 208,    65, 0,       14,     0,    0,    273,   14,    0,     41,     2,      1,     1,     41,      2,   1,   3;
    % different BWs...
    807,  1,        27,  1,   6,    54, 0,       14,     0,    0,    273,   15,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    808,  1,        27,  1,   6,    54, 0,       14,     0,    0,    273,   16,    0,     41,     2,      1,     1,     41,      2,   1,   0;
    809,  1,        27,  1,  60,    54, 0,       14,     0,    0,    273,   17,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    810,  1,        27,  1,  60,    54, 0,       14,     0,    0,    273,   18,    0,     41,     2,      1,     1,     41,      2,   1,   1;
    811,  1,        27,  1, 114,    54, 0,       14,     0,    0,    273,   19,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    812,  1,        27,  1, 114,    54, 0,       14,     0,    0,    273,   20,    0,     41,     2,      1,     1,     41,      2,   1,   2;
    813,  1,        27,  1, 168,    57, 0,       14,     0,    0,    273,   21,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    814,  1,        27,  1, 168,    57, 0,       14,     0,    0,    273,   22,    0,     41,     2,      1,     1,     41,      2,   1,   3;
    % 64TR MU-MIMO column G w/ reduced layers, 2 UEGs, 4 UEs per UEG, 1 layer per UE
% TC#   mcsTable  mcs  nl  rb0     Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    815,  1,        27,  1,  22,   126, 0,       14,     0,    0,    273,    7,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    816,  1,        27,  1,  22,   126, 0,       14,     0,    0,    273,    8,    0,     41,     2,      1,     1,     41,      2,   1,   0;
    817,  1,        27,  1,  22,   126, 0,       14,     0,    0,    273,    9,    0,     41,     2,      1,     1,     41,      2,   2,   0;
    818,  1,        27,  1,  22,   126, 0,       14,     0,    0,    273,   10,    0,     41,     2,      1,     1,     41,      2,   3,   0;
    819,  1,        27,  1, 148,   125, 0,       14,     0,    0,    273,   11,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    820,  1,        27,  1, 148,   125, 0,       14,     0,    0,    273,   12,    0,     41,     2,      1,     1,     41,      2,   1,   1;
    821,  1,        27,  1, 148,   125, 0,       14,     0,    0,    273,   13,    0,     41,     2,      1,     1,     41,      2,   2,   1;
    822,  1,        27,  1, 148,   125, 0,       14,     0,    0,    273,   14,    0,     41,     2,      1,     1,     41,      2,   3,   1;
    % different BWs...
    823,  1,        27,  1,   6,   110, 0,       14,     0,    0,    273,   15,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    824,  1,        27,  1,   6,   110, 0,       14,     0,    0,    273,   16,    0,     41,     2,      1,     1,     41,      2,   1,   0;
    825,  1,        27,  1,   6,   110, 0,       14,     0,    0,    273,   17,    0,     41,     2,      1,     1,     41,      2,   2,   0;
    826,  1,        27,  1,   6,   110, 0,       14,     0,    0,    273,   18,    0,     41,     2,      1,     1,     41,      2,   3,   0;
    827,  1,        27,  1, 116,   109, 0,       14,     0,    0,    273,   19,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    828,  1,        27,  1, 116,   109, 0,       14,     0,    0,    273,   20,    0,     41,     2,      1,     1,     41,      2,   1,   1;
    829,  1,        27,  1, 116,   109, 0,       14,     0,    0,    273,   21,    0,     41,     2,      1,     1,     41,      2,   2,   1;
    830,  1,        27,  1, 116,   109, 0,       14,     0,    0,    273,   22,    0,     41,     2,      1,     1,     41,      2,   3,   1;
    % 64TR MU-MIMO column G w/ reduced layers, 2 UEGs, 4 UEs per UEG, 1 layer per UE
% TC#   mcsTable  mcs  nl  rb0     Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    831,  1,        27,  1,  22,   126, 0,       12,     0,    0,    273,    7,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    832,  1,        27,  1,  22,   126, 0,       12,     0,    0,    273,    8,    0,     41,     2,      1,     1,     41,      2,   1,   0;
    833,  1,        27,  1,  22,   126, 0,       12,     0,    0,    273,    9,    0,     41,     2,      1,     1,     41,      2,   2,   0;
    834,  1,        27,  1,  22,   126, 0,       12,     0,    0,    273,   10,    0,     41,     2,      1,     1,     41,      2,   3,   0;
    835,  1,        27,  1, 148,   125, 0,       12,     0,    0,    273,   11,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    836,  1,        27,  1, 148,   125, 0,       12,     0,    0,    273,   12,    0,     41,     2,      1,     1,     41,      2,   1,   1;
    837,  1,        27,  1, 148,   125, 0,       12,     0,    0,    273,   13,    0,     41,     2,      1,     1,     41,      2,   2,   1;
    838,  1,        27,  1, 148,   125, 0,       12,     0,    0,    273,   14,    0,     41,     2,      1,     1,     41,      2,   3,   1;
    % different BWs...
    839,  1,        27,  1,   6,   110, 0,       12,     0,    0,    273,   15,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    840,  1,        27,  1,   6,   110, 0,       12,     0,    0,    273,   16,    0,     41,     2,      1,     1,     41,      2,   1,   0;
    841,  1,        27,  1,   6,   110, 0,       12,     0,    0,    273,   17,    0,     41,     2,      1,     1,     41,      2,   2,   0;
    842,  1,        27,  1,   6,   110, 0,       12,     0,    0,    273,   18,    0,     41,     2,      1,     1,     41,      2,   3,   0;
    843,  1,        27,  1, 116,   109, 0,       12,     0,    0,    273,   19,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    844,  1,        27,  1, 116,   109, 0,       12,     0,    0,    273,   20,    0,     41,     2,      1,     1,     41,      2,   1,   1;
    845,  1,        27,  1, 116,   109, 0,       12,     0,    0,    273,   21,    0,     41,     2,      1,     1,     41,      2,   2,   1;
    846,  1,        27,  1, 116,   109, 0,       12,     0,    0,    273,   22,    0,     41,     2,      1,     1,     41,      2,   3,   1;
% TC#   mcsTable  mcs  nl  rb0     Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    % PUSCH set 0: PRB 26~272, OFDM symbol 0~13, 100 MHz heavy
    847,  1,        19,  1,  26,    70, 0,       14,     0,    0,    273,    7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
    848,  1,        19,  1,  26,    70, 0,       14,     0,    0,    273,    8,    0,     41,     2,      2,     1,     41,      2,   1,   0;  % Center UEG0 port1
    849,  1,        19,  1,  26,    70, 0,       14,     0,    0,    273,    9,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
    850,  1,        19,  1,  26,    70, 0,       14,     0,    0,    273,   10,    0,     41,     2,      2,     1,     41,      2,   3,   0;  % Center UEG0 port3
    851,  1,        19,  1,  26,    70, 0,       14,     0,    0,    273,   11,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
    852,  1,        19,  1,  26,    70, 0,       14,     0,    0,    273,   12,    0,     41,     2,      2,     1,     41,      2,   5,   0;  % Center UEG0 port5
    853,  1,        19,  1,  26,    70, 0,       14,     0,    0,    273,   13,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
    854,  1,        19,  1,  26,    70, 0,       14,     0,    0,    273,   14,    0,     41,     2,      2,     1,     41,      2,   7,   0;  % Center UEG0 port7
    855,  1,        10,  1,  96,    40, 0,       14,     0,    0,    273,   15,    0,     41,     2,      1,     1,     41,      2,   0,   1;  % Middle UEG1 port0
    856,  1,        10,  1,  96,    40, 0,       14,     0,    0,    273,   16,    0,     41,     2,      1,     1,     41,      2,   1,   1;  % Middle UEG1 port1
    857,  1,        10,  1,  96,    40, 0,       14,     0,    0,    273,   17,    0,     41,     2,      1,     1,     41,      2,   2,   1;  % Middle UEG1 port2
    858,  1,        10,  1,  96,    40, 0,       14,     0,    0,    273,   18,    0,     41,     2,      1,     1,     41,      2,   3,   1;  % Middle UEG1 port3
    859,  1,        10,  1, 136,    40, 0,       14,     0,    0,    273,   19,    0,     41,     2,      1,     1,     41,      2,   0,   2;  % Middle UEG2 port0
    860,  1,        10,  1, 136,    40, 0,       14,     0,    0,    273,   20,    0,     41,     2,      1,     1,     41,      2,   1,   2;  % Middle UEG2 port1
    861,  1,        10,  1, 136,    40, 0,       14,     0,    0,    273,   21,    0,     41,     2,      1,     1,     41,      2,   2,   2;  % Middle UEG2 port2
    862,  1,        10,  1, 136,    40, 0,       14,     0,    0,    273,   22,    0,     41,     2,      1,     1,     41,      2,   3,   2;  % Middle UEG2 port3
    863,  1,         4,  1, 176,    24, 0,       14,     0,    0,    273,   23,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
    864,  1,         4,  1, 200,    24, 0,       14,     0,    0,    273,   24,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
    865,  1,         4,  1, 224,    24, 0,       14,     0,    0,    273,   25,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
    866,  1,         4,  1, 248,    25, 0,       14,     0,    0,    273,   26,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PUSCH set 1: PRB 10~261, OFDM symbol 0~13, 100 MHz heavy
    867,  1,        19,  1,  10,    86, 0,       14,     0,    0,    273,   27,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
    868,  1,        19,  1,  10,    86, 0,       14,     0,    0,    273,   28,    0,     41,     2,      2,     1,     41,      2,   1,   0;  % Center UEG0 port1
    869,  1,        19,  1,  10,    86, 0,       14,     0,    0,    273,   29,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
    870,  1,        19,  1,  10,    86, 0,       14,     0,    0,    273,   30,    0,     41,     2,      2,     1,     41,      2,   3,   0;  % Center UEG0 port3
    871,  1,        19,  1,  10,    86, 0,       14,     0,    0,    273,   31,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
    872,  1,        19,  1,  10,    86, 0,       14,     0,    0,    273,   32,    0,     41,     2,      2,     1,     41,      2,   5,   0;  % Center UEG0 port5
    873,  1,        19,  1,  10,    86, 0,       14,     0,    0,    273,   33,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
    874,  1,        19,  1,  10,    86, 0,       14,     0,    0,    273,   34,    0,     41,     2,      2,     1,     41,      2,   7,   0;  % Center UEG0 port7
    875,  1,        10,  1,  96,    40, 0,       14,     0,    0,    273,   35,    0,     41,     2,      1,     1,     41,      2,   0,   1;  % Middle UEG1 port0
    876,  1,        10,  1,  96,    40, 0,       14,     0,    0,    273,   36,    0,     41,     2,      1,     1,     41,      2,   1,   1;  % Middle UEG1 port1
    877,  1,        10,  1,  96,    40, 0,       14,     0,    0,    273,   37,    0,     41,     2,      1,     1,     41,      2,   2,   1;  % Middle UEG1 port2
    878,  1,        10,  1,  96,    40, 0,       14,     0,    0,    273,   38,    0,     41,     2,      1,     1,     41,      2,   3,   1;  % Middle UEG1 port3
    879,  1,        10,  1, 136,    40, 0,       14,     0,    0,    273,   39,    0,     41,     2,      1,     1,     41,      2,   0,   2;  % Middle UEG2 port0
    880,  1,        10,  1, 136,    40, 0,       14,     0,    0,    273,   40,    0,     41,     2,      1,     1,     41,      2,   1,   2;  % Middle UEG2 port1
    881,  1,        10,  1, 136,    40, 0,       14,     0,    0,    273,   41,    0,     41,     2,      1,     1,     41,      2,   2,   2;  % Middle UEG2 port2
    882,  1,        10,  1, 136,    40, 0,       14,     0,    0,    273,   42,    0,     41,     2,      1,     1,     41,      2,   3,   2;  % Middle UEG2 port3
    883,  1,         4,  1, 176,    24, 0,       14,     0,    0,    273,   43,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
    884,  1,         4,  1, 200,    24, 0,       14,     0,    0,    273,   44,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
    885,  1,         4,  1, 224,    24, 0,       14,     0,    0,    273,   45,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
    886,  1,         4,  1, 248,    13, 0,       14,     0,    0,    273,   46,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PUSCH set 2: PRB 10~272, OFDM symbol 0~13, 100 MHz heavy
    % reuse cfg 867~885 from PUSCH set 1: PRB 10~261, OFDM symbol 0~13, 100 MHz heavy
    887,  1,         4,  1, 248,    25, 0,       14,     0,    0,    273,   46,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
% TC#   mcsTable  mcs  nl  rb0     Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    % PUSCH set 0: PRB 26~272, OFDM symbol 0~13, 100 MHz light
    888,  1,        19,  1,  26,    70, 0,       14,     0,    0,    273,    7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
    889,  1,        19,  1,  26,    70, 0,       14,     0,    0,    273,    8,    0,     41,     2,      2,     1,     41,      2,   1,   0;  % Center UEG0 port1
    890,  1,        19,  1,  26,    70, 0,       14,     0,    0,    273,    9,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
    891,  1,        19,  1,  26,    70, 0,       14,     0,    0,    273,   10,    0,     41,     2,      2,     1,     41,      2,   3,   0;  % Center UEG0 port3
    892,  1,        10,  1,  96,    40, 0,       14,     0,    0,    273,   11,    0,     41,     2,      1,     1,     41,      2,   0,   1;  % Middle UEG1 port0
    893,  1,        10,  1,  96,    40, 0,       14,     0,    0,    273,   12,    0,     41,     2,      1,     1,     41,      2,   1,   1;  % Middle UEG1 port1
    894,  1,        10,  1, 136,    40, 0,       14,     0,    0,    273,   13,    0,     41,     2,      1,     1,     41,      2,   0,   2;  % Middle UEG2 port0
    895,  1,        10,  1, 136,    40, 0,       14,     0,    0,    273,   14,    0,     41,     2,      1,     1,     41,      2,   1,   2;  % Middle UEG2 port1
    896,  1,         4,  1, 176,    24, 0,       14,     0,    0,    273,   15,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
    897,  1,         4,  1, 200,    24, 0,       14,     0,    0,    273,   16,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
    898,  1,         4,  1, 224,    24, 0,       14,     0,    0,    273,   17,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
    899,  1,         4,  1, 248,    25, 0,       14,     0,    0,    273,   18,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PUSCH set 1: PRB 10~261, OFDM symbol 0~13, 100 MHz light
    900,  1,        19,  1,  10,    86, 0,       14,     0,    0,    273,   19,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
    901,  1,        19,  1,  10,    86, 0,       14,     0,    0,    273,   20,    0,     41,     2,      2,     1,     41,      2,   1,   0;  % Center UEG0 port1
    902,  1,        19,  1,  10,    86, 0,       14,     0,    0,    273,   21,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
    903,  1,        19,  1,  10,    86, 0,       14,     0,    0,    273,   22,    0,     41,     2,      2,     1,     41,      2,   3,   0;  % Center UEG0 port3
    904,  1,        10,  1,  96,    40, 0,       14,     0,    0,    273,   23,    0,     41,     2,      1,     1,     41,      2,   0,   1;  % Middle UEG1 port0
    905,  1,        10,  1,  96,    40, 0,       14,     0,    0,    273,   24,    0,     41,     2,      1,     1,     41,      2,   1,   1;  % Middle UEG1 port1
    906,  1,        10,  1, 136,    40, 0,       14,     0,    0,    273,   25,    0,     41,     2,      1,     1,     41,      2,   0,   2;  % Middle UEG2 port0
    907,  1,        10,  1, 136,    40, 0,       14,     0,    0,    273,   26,    0,     41,     2,      1,     1,     41,      2,   1,   2;  % Middle UEG2 port1
    908,  1,         4,  1, 176,    24, 0,       14,     0,    0,    273,   27,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
    909,  1,         4,  1, 200,    24, 0,       14,     0,    0,    273,   28,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
    910,  1,         4,  1, 224,    24, 0,       14,     0,    0,    273,   29,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
    911,  1,         4,  1, 248,    13, 0,       14,     0,    0,    273,   30,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PUSCH set 2: PRB 10~272, OFDM symbol 0~13, 100 MHz light
    % reuse cfg 900~910 from PUSCH set 1: PRB 10~261, OFDM symbol 0~13, 100 MHz light
    912,  1,         4,  1, 248,    25, 0,       14,     0,    0,    273,   30,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PUSCH set 0: PRB 26~271, OFDM symbol 0~13, 90 MHz heavy
    913,  1,        19,  1,  26,    62, 0,       14,     0,    0,    244,    7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
    914,  1,        19,  1,  26,    62, 0,       14,     0,    0,    244,    8,    0,     41,     2,      2,     1,     41,      2,   1,   0;  % Center UEG0 port1
    915,  1,        19,  1,  26,    62, 0,       14,     0,    0,    244,    9,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
    916,  1,        19,  1,  26,    62, 0,       14,     0,    0,    244,   10,    0,     41,     2,      2,     1,     41,      2,   3,   0;  % Center UEG0 port3
    917,  1,        19,  1,  26,    62, 0,       14,     0,    0,    244,   11,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
    918,  1,        19,  1,  26,    62, 0,       14,     0,    0,    244,   12,    0,     41,     2,      2,     1,     41,      2,   5,   0;  % Center UEG0 port5
    919,  1,        19,  1,  26,    62, 0,       14,     0,    0,    244,   13,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
    920,  1,        19,  1,  26,    62, 0,       14,     0,    0,    244,   14,    0,     41,     2,      2,     1,     41,      2,   7,   0;  % Center UEG0 port7
    921,  1,        10,  1,  88,    34, 0,       14,     0,    0,    244,   15,    0,     41,     2,      1,     1,     41,      2,   0,   1;  % Middle UEG1 port0
    922,  1,        10,  1,  88,    34, 0,       14,     0,    0,    244,   16,    0,     41,     2,      1,     1,     41,      2,   1,   1;  % Middle UEG1 port1
    923,  1,        10,  1,  88,    34, 0,       14,     0,    0,    244,   17,    0,     41,     2,      1,     1,     41,      2,   2,   1;  % Middle UEG1 port2
    924,  1,        10,  1,  88,    34, 0,       14,     0,    0,    244,   18,    0,     41,     2,      1,     1,     41,      2,   3,   1;  % Middle UEG1 port3
    925,  1,        10,  1, 122,    34, 0,       14,     0,    0,    244,   19,    0,     41,     2,      1,     1,     41,      2,   0,   2;  % Middle UEG2 port0
    926,  1,        10,  1, 122,    34, 0,       14,     0,    0,    244,   20,    0,     41,     2,      1,     1,     41,      2,   1,   2;  % Middle UEG2 port1
    927,  1,        10,  1, 122,    34, 0,       14,     0,    0,    244,   21,    0,     41,     2,      1,     1,     41,      2,   2,   2;  % Middle UEG2 port2
    928,  1,        10,  1, 122,    34, 0,       14,     0,    0,    244,   22,    0,     41,     2,      1,     1,     41,      2,   3,   2;  % Middle UEG2 port3
    929,  1,         4,  1, 156,    22, 0,       14,     0,    0,    244,   23,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
    930,  1,         4,  1, 178,    22, 0,       14,     0,    0,    244,   24,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
    931,  1,         4,  1, 200,    22, 0,       14,     0,    0,    244,   25,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
    932,  1,         4,  1, 222,    22, 0,       14,     0,    0,    244,   26,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PUSCH set 1: PRB 10~231, OFDM symbol 0~13
    933,  1,        19,  1,  10,    78, 0,       14,     0,    0,    244,   27,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
    934,  1,        19,  1,  10,    78, 0,       14,     0,    0,    244,   28,    0,     41,     2,      2,     1,     41,      2,   1,   0;  % Center UEG0 port1
    935,  1,        19,  1,  10,    78, 0,       14,     0,    0,    244,   29,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
    936,  1,        19,  1,  10,    78, 0,       14,     0,    0,    244,   30,    0,     41,     2,      2,     1,     41,      2,   3,   0;  % Center UEG0 port3
    937,  1,        19,  1,  10,    78, 0,       14,     0,    0,    244,   31,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
    938,  1,        19,  1,  10,    78, 0,       14,     0,    0,    244,   32,    0,     41,     2,      2,     1,     41,      2,   5,   0;  % Center UEG0 port5
    939,  1,        19,  1,  10,    78, 0,       14,     0,    0,    244,   33,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
    940,  1,        19,  1,  10,    78, 0,       14,     0,    0,    244,   34,    0,     41,     2,      2,     1,     41,      2,   7,   0;  % Center UEG0 port7
    941,  1,        10,  1,  88,    34, 0,       14,     0,    0,    244,   35,    0,     41,     2,      1,     1,     41,      2,   0,   1;  % Middle UEG1 port0
    942,  1,        10,  1,  88,    34, 0,       14,     0,    0,    244,   36,    0,     41,     2,      1,     1,     41,      2,   1,   1;  % Middle UEG1 port1
    943,  1,        10,  1,  88,    34, 0,       14,     0,    0,    244,   37,    0,     41,     2,      1,     1,     41,      2,   2,   1;  % Middle UEG1 port2
    944,  1,        10,  1,  88,    34, 0,       14,     0,    0,    244,   38,    0,     41,     2,      1,     1,     41,      2,   3,   1;  % Middle UEG1 port3
    945,  1,        10,  1, 122,    34, 0,       14,     0,    0,    244,   39,    0,     41,     2,      1,     1,     41,      2,   0,   2;  % Middle UEG2 port0
    946,  1,        10,  1, 122,    34, 0,       14,     0,    0,    244,   40,    0,     41,     2,      1,     1,     41,      2,   1,   2;  % Middle UEG2 port1
    947,  1,        10,  1, 122,    34, 0,       14,     0,    0,    244,   41,    0,     41,     2,      1,     1,     41,      2,   2,   2;  % Middle UEG2 port2
    948,  1,        10,  1, 122,    34, 0,       14,     0,    0,    244,   42,    0,     41,     2,      1,     1,     41,      2,   3,   2;  % Middle UEG2 port3
    949,  1,         4,  1, 156,    22, 0,       14,     0,    0,    244,   43,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
    950,  1,         4,  1, 178,    22, 0,       14,     0,    0,    244,   44,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
    951,  1,         4,  1, 200,    22, 0,       14,     0,    0,    244,   45,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
    952,  1,         4,  1, 222,    10, 0,       14,     0,    0,    244,   46,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PUSCH set 2: PRB 10~243, OFDM symbol 0~13
    % reuse cfg 933~951 from PUSCH set 1: PRB 10~231, OFDM symbol 0~13
    953,  1,         4,  1, 222,    22, 0,       14,     0,    0,    244,   46,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PUSCH set 0: PRB 26~243, OFDM symbol 0~13
    954,  1,        19,  1,  26,    62, 0,       14,     0,    0,    244,    7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
    955,  1,        19,  1,  26,    62, 0,       14,     0,    0,    244,    8,    0,     41,     2,      2,     1,     41,      2,   1,   0;  % Center UEG0 port1
    956,  1,        19,  1,  26,    62, 0,       14,     0,    0,    244,    9,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
    957,  1,        19,  1,  26,    62, 0,       14,     0,    0,    244,   10,    0,     41,     2,      2,     1,     41,      2,   3,   0;  % Center UEG0 port3
    958,  1,        10,  1,  88,    34, 0,       14,     0,    0,    244,   11,    0,     41,     2,      1,     1,     41,      2,   0,   1;  % Middle UEG1 port0
    959,  1,        10,  1,  88,    34, 0,       14,     0,    0,    244,   12,    0,     41,     2,      1,     1,     41,      2,   1,   1;  % Middle UEG1 port1
    960,  1,        10,  1, 122,    34, 0,       14,     0,    0,    244,   13,    0,     41,     2,      1,     1,     41,      2,   0,   2;  % Middle UEG2 port0
    961,  1,        10,  1, 122,    34, 0,       14,     0,    0,    244,   14,    0,     41,     2,      1,     1,     41,      2,   1,   2;  % Middle UEG2 port1
    962,  1,         4,  1, 156,    22, 0,       14,     0,    0,    244,   15,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
    963,  1,         4,  1, 178,    22, 0,       14,     0,    0,    244,   16,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
    964,  1,         4,  1, 200,    22, 0,       14,     0,    0,    244,   17,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
    965,  1,         4,  1, 222,    22, 0,       14,     0,    0,    244,   18,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PUSCH set 1: PRB 10~231, OFDM symbol 0~13
    966,  1,        19,  1,  10,    78, 0,       14,     0,    0,    244,   19,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
    967,  1,        19,  1,  10,    78, 0,       14,     0,    0,    244,   20,    0,     41,     2,      2,     1,     41,      2,   1,   0;  % Center UEG0 port1
    968,  1,        19,  1,  10,    78, 0,       14,     0,    0,    244,   21,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
    969,  1,        19,  1,  10,    78, 0,       14,     0,    0,    244,   22,    0,     41,     2,      2,     1,     41,      2,   3,   0;  % Center UEG0 port3
    970,  1,        10,  1,  88,    34, 0,       14,     0,    0,    244,   23,    0,     41,     2,      1,     1,     41,      2,   0,   1;  % Middle UEG1 port0
    971,  1,        10,  1,  88,    34, 0,       14,     0,    0,    244,   24,    0,     41,     2,      1,     1,     41,      2,   1,   1;  % Middle UEG1 port1
    972,  1,        10,  1, 122,    34, 0,       14,     0,    0,    244,   25,    0,     41,     2,      1,     1,     41,      2,   0,   2;  % Middle UEG2 port0
    973,  1,        10,  1, 122,    34, 0,       14,     0,    0,    244,   26,    0,     41,     2,      1,     1,     41,      2,   1,   2;  % Middle UEG2 port1
    974,  1,         4,  1, 156,    22, 0,       14,     0,    0,    244,   27,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
    975,  1,         4,  1, 178,    22, 0,       14,     0,    0,    244,   28,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
    976,  1,         4,  1, 200,    22, 0,       14,     0,    0,    244,   29,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
    977,  1,         4,  1, 222,    10, 0,       14,     0,    0,    244,   30,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PUSCH set 2: PRB 10~243, OFDM symbol 0~13
    % reuse cfg 966~976 from PUSCH set 1: PRB 10~231, OFDM symbol 0~13
    978,  1,         4,  1, 222,    22, 0,       14,     0,    0,    244,   30,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PUSCH set 0: PRB 26~159, OFDM symbol 0~13
    979,  1,        19,  1,  26,    30, 0,       14,     0,    0,    160,    7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
    980,  1,        19,  1,  26,    30, 0,       14,     0,    0,    160,    8,    0,     41,     2,      2,     1,     41,      2,   1,   0;  % Center UEG0 port1
    981,  1,        19,  1,  26,    30, 0,       14,     0,    0,    160,    9,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
    982,  1,        19,  1,  26,    30, 0,       14,     0,    0,    160,   10,    0,     41,     2,      2,     1,     41,      2,   3,   0;  % Center UEG0 port3
    983,  1,        19,  1,  26,    30, 0,       14,     0,    0,    160,   11,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
    984,  1,        19,  1,  26,    30, 0,       14,     0,    0,    160,   12,    0,     41,     2,      2,     1,     41,      2,   5,   0;  % Center UEG0 port5
    985,  1,        19,  1,  26,    30, 0,       14,     0,    0,    160,   13,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
    986,  1,        19,  1,  26,    30, 0,       14,     0,    0,    160,   14,    0,     41,     2,      2,     1,     41,      2,   7,   0;  % Center UEG0 port7
    987,  1,        10,  1,  56,    24, 0,       14,     0,    0,    160,   15,    0,     41,     2,      1,     1,     41,      2,   0,   1;  % Middle UEG1 port0
    988,  1,        10,  1,  56,    24, 0,       14,     0,    0,    160,   16,    0,     41,     2,      1,     1,     41,      2,   1,   1;  % Middle UEG1 port1
    989,  1,        10,  1,  56,    24, 0,       14,     0,    0,    160,   17,    0,     41,     2,      1,     1,     41,      2,   2,   1;  % Middle UEG1 port2
    990,  1,        10,  1,  56,    24, 0,       14,     0,    0,    160,   18,    0,     41,     2,      1,     1,     41,      2,   3,   1;  % Middle UEG1 port3
    991,  1,        10,  1,  80,    24, 0,       14,     0,    0,    160,   19,    0,     41,     2,      1,     1,     41,      2,   0,   2;  % Middle UEG2 port0
    992,  1,        10,  1,  80,    24, 0,       14,     0,    0,    160,   20,    0,     41,     2,      1,     1,     41,      2,   1,   2;  % Middle UEG2 port1
    993,  1,        10,  1,  80,    24, 0,       14,     0,    0,    160,   21,    0,     41,     2,      1,     1,     41,      2,   2,   2;  % Middle UEG2 port2
    994,  1,        10,  1,  80,    24, 0,       14,     0,    0,    160,   22,    0,     41,     2,      1,     1,     41,      2,   3,   2;  % Middle UEG2 port3
    995,  1,         4,  1, 104,    14, 0,       14,     0,    0,    160,   23,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
    996,  1,         4,  1, 118,    14, 0,       14,     0,    0,    160,   24,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
    997,  1,         4,  1, 132,    14, 0,       14,     0,    0,    160,   25,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
    998,  1,         4,  1, 146,    14, 0,       14,     0,    0,    160,   26,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PUSCH set 1: PRB 10~147, OFDM symbol 0~13
    999,  1,        19,  1,  10,    46, 0,       14,     0,    0,    160,   27,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
   1000,  1,        19,  1,  10,    46, 0,       14,     0,    0,    160,   28,    0,     41,     2,      2,     1,     41,      2,   1,   0;  % Center UEG0 port1
   1001,  1,        19,  1,  10,    46, 0,       14,     0,    0,    160,   29,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
   1002,  1,        19,  1,  10,    46, 0,       14,     0,    0,    160,   30,    0,     41,     2,      2,     1,     41,      2,   3,   0;  % Center UEG0 port3
   1003,  1,        19,  1,  10,    46, 0,       14,     0,    0,    160,   31,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
   1004,  1,        19,  1,  10,    46, 0,       14,     0,    0,    160,   32,    0,     41,     2,      2,     1,     41,      2,   5,   0;  % Center UEG0 port5
   1005,  1,        19,  1,  10,    46, 0,       14,     0,    0,    160,   33,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
   1006,  1,        19,  1,  10,    46, 0,       14,     0,    0,    160,   34,    0,     41,     2,      2,     1,     41,      2,   7,   0;  % Center UEG0 port7
   1007,  1,        10,  1,  56,    24, 0,       14,     0,    0,    160,   35,    0,     41,     2,      1,     1,     41,      2,   0,   1;  % Middle UEG1 port0
   1008,  1,        10,  1,  56,    24, 0,       14,     0,    0,    160,   36,    0,     41,     2,      1,     1,     41,      2,   1,   1;  % Middle UEG1 port1
   1009,  1,        10,  1,  56,    24, 0,       14,     0,    0,    160,   37,    0,     41,     2,      1,     1,     41,      2,   2,   1;  % Middle UEG1 port2
   1010,  1,        10,  1,  56,    24, 0,       14,     0,    0,    160,   38,    0,     41,     2,      1,     1,     41,      2,   3,   1;  % Middle UEG1 port3
   1011,  1,        10,  1,  80,    24, 0,       14,     0,    0,    160,   39,    0,     41,     2,      1,     1,     41,      2,   0,   2;  % Middle UEG2 port0
   1012,  1,        10,  1,  80,    24, 0,       14,     0,    0,    160,   40,    0,     41,     2,      1,     1,     41,      2,   1,   2;  % Middle UEG2 port1
   1013,  1,        10,  1,  80,    24, 0,       14,     0,    0,    160,   41,    0,     41,     2,      1,     1,     41,      2,   2,   2;  % Middle UEG2 port2
   1014,  1,        10,  1,  80,    24, 0,       14,     0,    0,    160,   42,    0,     41,     2,      1,     1,     41,      2,   3,   2;  % Middle UEG2 port3
   1015,  1,         4,  1, 104,    14, 0,       14,     0,    0,    160,   43,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
   1016,  1,         4,  1, 118,    14, 0,       14,     0,    0,    160,   44,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
   1017,  1,         4,  1, 132,     8, 0,       14,     0,    0,    160,   45,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
   1018,  1,         4,  1, 140,     8, 0,       14,     0,    0,    160,   46,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PUSCH set 2: PRB 10~159, OFDM symbol 0~13
    % reuse cfg 999~1017 from PUSCH set 1: PRB 10~147, OFDM symbol 0~13
   1019,  1,         4,  1, 140,    20, 0,       14,     0,    0,    160,   46,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PUSCH set 0: PRB 26~159, OFDM symbol 0~13
   1020,  1,        19,  1,  26,    30, 0,       14,     0,    0,    160,    7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
   1021,  1,        19,  1,  26,    30, 0,       14,     0,    0,    160,    8,    0,     41,     2,      2,     1,     41,      2,   1,   0;  % Center UEG0 port1
   1022,  1,        19,  1,  26,    30, 0,       14,     0,    0,    160,    9,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
   1023,  1,        19,  1,  26,    30, 0,       14,     0,    0,    160,   10,    0,     41,     2,      2,     1,     41,      2,   3,   0;  % Center UEG0 port3
   1024,  1,        10,  1,  56,    24, 0,       14,     0,    0,    160,   11,    0,     41,     2,      1,     1,     41,      2,   0,   1;  % Middle UEG1 port0
   1025,  1,        10,  1,  56,    24, 0,       14,     0,    0,    160,   12,    0,     41,     2,      1,     1,     41,      2,   1,   1;  % Middle UEG1 port1
   1026,  1,        10,  1,  80,    24, 0,       14,     0,    0,    160,   13,    0,     41,     2,      1,     1,     41,      2,   0,   2;  % Middle UEG2 port0
   1027,  1,        10,  1,  80,    24, 0,       14,     0,    0,    160,   14,    0,     41,     2,      1,     1,     41,      2,   1,   2;  % Middle UEG2 port1
   1028,  1,         4,  1, 104,    14, 0,       14,     0,    0,    160,   15,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
   1029,  1,         4,  1, 118,    14, 0,       14,     0,    0,    160,   16,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
   1030,  1,         4,  1, 132,    14, 0,       14,     0,    0,    160,   17,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
   1031,  1,         4,  1, 146,    14, 0,       14,     0,    0,    160,   18,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PUSCH set 1: PRB 10~147, OFDM symbol 0~13
   1032,  1,        19,  1,  10,    46, 0,       14,     0,    0,    160,   19,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
   1033,  1,        19,  1,  10,    46, 0,       14,     0,    0,    160,   20,    0,     41,     2,      2,     1,     41,      2,   1,   0;  % Center UEG0 port1
   1034,  1,        19,  1,  10,    46, 0,       14,     0,    0,    160,   21,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
   1035,  1,        19,  1,  10,    46, 0,       14,     0,    0,    160,   22,    0,     41,     2,      2,     1,     41,      2,   3,   0;  % Center UEG0 port3
   1036,  1,        10,  1,  56,    24, 0,       14,     0,    0,    160,   23,    0,     41,     2,      1,     1,     41,      2,   0,   1;  % Middle UEG1 port0
   1037,  1,        10,  1,  56,    24, 0,       14,     0,    0,    160,   24,    0,     41,     2,      1,     1,     41,      2,   1,   1;  % Middle UEG1 port1
   1038,  1,        10,  1,  80,    24, 0,       14,     0,    0,    160,   25,    0,     41,     2,      1,     1,     41,      2,   0,   2;  % Middle UEG2 port0
   1039,  1,        10,  1,  80,    24, 0,       14,     0,    0,    160,   26,    0,     41,     2,      1,     1,     41,      2,   1,   2;  % Middle UEG2 port1
   1040,  1,         4,  1, 104,    14, 0,       14,     0,    0,    160,   27,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
   1041,  1,         4,  1, 118,    14, 0,       14,     0,    0,    160,   28,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
   1042,  1,         4,  1, 132,     8, 0,       14,     0,    0,    160,   29,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
   1043,  1,         4,  1, 140,     8, 0,       14,     0,    0,    160,   30,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PUSCH set 2: PRB 10~159, OFDM symbol 0~13
    % reuse cfg 1032~1042 from PUSCH set 1: PRB 10~147, OFDM symbol 0~13
   1044,  1,         4,  1, 140,    20, 0,       14,     0,    0,    160,   30,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
% TC#   mcsTable  mcs  nl  rb0     Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
   % 4TR PUSCH set 0: PRB 0~272 and 0~260, OFDM symbol 0~13
   1045,  1,        27,  2,   0,   273, 0,       14,     0,    0,    273,    7,    0,     41,     2,      1,     0,     41,      2,   0,   0;  % 1 UE SU-MIMO, slot 4
   1046,  1,        27,  2,   0,   261, 0,       14,     0,    0,    273,    8,    0,     41,     2,      1,     0,     41,      2,   0,   0;  % 1 UE SU-MIMO, slot 5
   1047,  1,        27,  2,   0,   273, 0,       14,     0,    0,    273,    9,    0,     41,     2,      1,     0,     41,      2,   0,   0;  % 1 UE SU-MIMO, slot 14
   1048,  1,        27,  2,   0,   273, 0,       14,     0,    0,    273,   10,    0,     41,     2,      1,     0,     41,      2,   0,   0;  % 1 UE SU-MIMO, slot 15
   % 4TR PUSCH set 0: PRB 0~53 and 0~41, OFDM symbol 0~13
   1049,  1,        27,  2,   0,    54, 0,       14,     0,    0,    273,    7,    0,     41,     2,      1,     0,     41,      2,   0,   0;  % 1 UE SU-MIMO, slot 4
   1050,  1,        27,  2,   0,    42, 0,       14,     0,    0,    273,    8,    0,     41,     2,      1,     0,     41,      2,   0,   0;  % 1 UE SU-MIMO, slot 5
   1051,  1,        27,  2,   0,    54, 0,       14,     0,    0,    273,    9,    0,     41,     2,      1,     0,     41,      2,   0,   0;  % 1 UE SU-MIMO, slot 14
   1052,  1,        27,  2,   0,    54, 0,       14,     0,    0,    273,   10,    0,     41,     2,      1,     0,     41,      2,   0,   0;  % 1 UE SU-MIMO, slot 15
   % 610,  1,         2,  2, 100,   40,   0,   14,     0,    0,    273,     1,    0,     41,     2,      2,    1,     41,      2,   0,    0; 
   % 611,  1,         2,  2, 100,   40,   0,   14,     0,    0,    273,     2,    0,     41,     2,      2,    1,     41,      2,   2,    0; 
   % 612,  1,         2,  2, 100,   40,   0,   14,     0,    0,    273,     3,    0,     41,     2,      2,    1,     41,      2,   4,    0; 
   % 613,  1,         2,  2, 100,   40,   0,   14,     0,    0,    273,     4,    0,     41,     2,      2,    1,     41,      2,   6,    0; 
   % 614,  1,        10,  1, 150,   40,   0,   14,     0,    0,    273,     5,    0,     41,     2,      2,    1,     41,      2,   0,    1; 
   % 615,  1,        10,  1, 150,   40,   0,   14,     0,    0,    273,     6,    0,     41,     2,      2,    1,     41,      2,   2,    1; 
   % 616,  1,        19,  1, 200,   40,   0,   14,     0,    0,    273,     7,    0,     41,     2,      2,    1,     41,      2,   0,    2;    
   % 
   % 617,  1,        17,  1,   4,  265,   0,   14,     0,    0,    273,     1,    0,     41,     2,      2,    1,     41,      2,   0,    0;    
   % 618,  1,        17,  2,  52,  217,   0,   14,     0,    0,    273,     2,    0,     41,     2,      2,    1,     41,      2,   0,    0;    
   % 
   % 619,  1,         7,  2,   4,  132,   0,   14,     0,    0,    273,     1,    0,     41,     2,      2,    1,     41,      2,   0,    0;    
   % 620,  1,        17,  1, 136,  133,   0,   14,     0,    0,    273,     2,    0,     41,     2,      2,    1,     41,      2,   0,    1;    
   % 621,  1,         7,  2,  52,  108,   0,   14,     0,    0,    273,     1,    0,     41,     2,      2,    1,     41,      2,   0,    0;    
   % 622,  1,        17,  1, 160,  109,   0,   14,     0,    0,    273,     2,    0,     41,     2,      2,    1,     41,      2,   0,    1;       
   % 
   % 623,  1,         7,  2,   4,  132,   0,   14,     0,    0,    273,     1,    0,     41,     2,      2,    1,     41,      2,   0,    0;    
   % 624,  1,         7,  2,   4,  132,   0,   14,     0,    0,    273,     2,    0,     41,     2,      2,    1,     41,      2,   2,    0;    
   % 625,  1,         7,  2,   4,  132,   0,   14,     0,    0,    273,     3,    0,     41,     2,      2,    1,     41,      2,   4,    0;    
   % 626,  1,         7,  2,   4,  132,   0,   14,     0,    0,    273,     4,    0,     41,     2,      2,    1,     41,      2,   6,    0;    
   % 627,  1,        17,  2, 136,  100,   0,   14,     0,    0,    273,     5,    0,     41,     2,      2,    1,     41,      2,   0,    1;    
   % 628,  1,        17,  1, 236,   33,   0,   14,     0,    0,    273,     6,    0,     41,     2,      2,    1,     41,      2,   0,    2;    
   % 
   % 629,  1,         7,  2,  52,   84,   0,   14,     0,    0,    273,     1,    0,     41,     2,      2,    1,     41,      2,   0,    0;    
   % 630,  1,         7,  2,  52,   84,   0,   14,     0,    0,    273,     2,    0,     41,     2,      2,    1,     41,      2,   2,    0;    
   % 631,  1,         7,  2,  52,   84,   0,   14,     0,    0,    273,     3,    0,     41,     2,      2,    1,     41,      2,   4,    0;    
   % 632,  1,         7,  2,  52,   84,   0,   14,     0,    0,    273,     4,    0,     41,     2,      2,    1,     41,      2,   6,    0;    
   % 633,  1,        17,  2, 136,  100,   0,   14,     0,    0,    273,     5,    0,     41,     2,      2,    1,     41,      2,   0,    1;    
   % 634,  1,        17,  1, 236,   33,   0,   14,     0,    0,    273,     6,    0,     41,     2,      2,    1,     41,      2,   0,    2;   
   % 
   % 635,  1,         7,  2,   4,  132,   0,   12,     0,    0,    273,     1,    0,     41,     2,      2,    1,     41,      2,   0,    0;    
   % 636,  1,         7,  2,   4,  132,   0,   12,     0,    0,    273,     2,    0,     41,     2,      2,    1,     41,      2,   2,    0;    
   % 637,  1,         7,  2,   4,  132,   0,   12,     0,    0,    273,     3,    0,     41,     2,      2,    1,     41,      2,   4,    0;    
   % 638,  1,         7,  2,   4,  132,   0,   12,     0,    0,    273,     4,    0,     41,     2,      2,    1,     41,      2,   6,    0;    
   % 639,  1,        17,  2, 136,  100,   0,   12,     0,    0,    273,     5,    0,     41,     2,      2,    1,     41,      2,   0,    1;    
   % 640,  1,        17,  1, 236,   33,   0,   12,     0,    0,    273,     6,    0,     41,     2,      2,    1,     41,      2,   0,    2;    

   };


% append generated PXSCH config
%                             pxsch_cfg_idx nUeg  nUePerUeg startRnti startPrb  endPrb  diffscId  nl  dmrsMaxLen  mcsIdx  sym0  Nsym  BWP0  nBWP  prgSize  prbAlloc
% 64TR MU-MIMO, 16 UEGs, 4 UEs per UEG, 2 layers per UE, PUSCH prb 22~272
% Base configuration struct
baseCfgUeg = struct();
baseCfgUeg.startRnti = 7;
baseCfgUeg.mcsIdx = 27;
baseCfgUeg.sym0 = 0;
baseCfgUeg.Nsym = 14;
baseCfgUeg.BWP0 = 0;
baseCfgUeg.nBWP = 273;
baseCfgUeg.prgSize = 2;
baseCfgUeg.diffscId = false;
baseCfgUeg.nl = 2;
baseCfgUeg.dmrsMaxLen = 2;

% 64TR MU-MIMO, 16 UEGs, 4 UEs per UEG, 2 layers per UE, PUSCH prb 22~272, CFG 2001~2064
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = 2001;  % first config index for generated PXSCH config
cfgUeg.nUeg = 16;
cfgUeg.nUePerUeg = 4;
cfgUeg.startPrb = 22;
cfgUeg.endPrb = 272;
cfgUeg.prbAlloc = [8,12,16,18,20,22,24,22,20,18,16,14,12,10,8,11];
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 16 UEGs, 4 UEs per UEG, 2 layers per UE, PUSCH prb 6~224, CFG 2065~2128
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.nUeg = 16;
cfgUeg.nUePerUeg = 4;
cfgUeg.startPrb = 6;
cfgUeg.endPrb = 224;
cfgUeg.prbAlloc = [6,10,14,16,18,20,22,20,18,16,14,12,10,8,6,9];
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 2 UEGs, 4 UEs per UEG, 1 layer per UE, PUSCH prb 26~272, CFG 2129~2136
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.nUeg = 2;
cfgUeg.nUePerUeg = 4;
cfgUeg.startPrb = 26;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
cfgUeg.dmrsMaxLen = 1;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 2 UEGs, 4 UEs per UEG, 1 layer per UE, PUSCH prb 10~260, CFG 2137~2144
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.startRnti = 15;
cfgUeg.nUeg = 2;
cfgUeg.nUePerUeg = 4;
cfgUeg.startPrb = 10;
cfgUeg.endPrb = 260;
cfgUeg.nl = 1;
cfgUeg.dmrsMaxLen = 1;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 26~272, CFG 2145~2152
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.startPrb = 26;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 10~260, CFG 2153~2160
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.startRnti = 15;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.startPrb = 10;
cfgUeg.endPrb = 260;
cfgUeg.nl = 1;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 10~272, CFG 2161~2168
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.startRnti = 15;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.startPrb = 10;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 26~272, 13 OFDM symbols, CFG 2169~2176
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.Nsym = 13;
cfgUeg.startPrb = 26;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 10~260, 13 OFDM symbols, CFG 2177~2184
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.startRnti = 15;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.Nsym = 13;
cfgUeg.startPrb = 10;
cfgUeg.endPrb = 260;
cfgUeg.nl = 1;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 10~272, 13 OFDM symbols, CFG 2185~2192
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.startRnti = 15;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.Nsym = 13;
cfgUeg.startPrb = 10;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 68~272, CFG 2193~2200
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.startPrb = 68;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 26~260, CFG 2201~2208
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.startRnti = 15;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.startPrb = 26;
cfgUeg.endPrb = 260;
cfgUeg.nl = 1;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 90~272, CFG 2209~2216
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.startPrb = 90;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 26~272, CFG 2217~2224
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.startRnti = 15;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.startPrb = 26;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 68~272, 13 OFDM symbols, CFG 2225~2232
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.Nsym = 13;
cfgUeg.startPrb = 68;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 26~260, 13 OFDM symbols, CFG 2233~2240
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.startRnti = 15;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.Nsym = 13;
cfgUeg.startPrb = 26;
cfgUeg.endPrb = 260;
cfgUeg.nl = 1;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 90~272, 13 OFDM symbols, CFG 2241~2248
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.Nsym = 13;
cfgUeg.startPrb = 90;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 26~272, 13 OFDM symbols, CFG 2249~2256
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.startRnti = 15;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.Nsym = 13;
cfgUeg.startPrb = 26;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 36~272, 13 OFDM symbols, CFG 2257~2264
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.startPrb = 36;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 15~260, 13 OFDM symbols, CFG 2265~2272
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.startRnti = 15;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.startPrb = 15;
cfgUeg.endPrb = 260;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 47~272, 13 OFDM symbols, CFG 2273~2280
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.startPrb = 47;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 15~272, 13 OFDM symbols, CFG 2281~2288
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.startRnti = 15;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.startPrb = 15;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 2~272, CFG 2289~2296
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.startPrb = 26;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 10~260, CFG 2297~2304
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.startRnti = 15;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.startPrb = 10;
cfgUeg.endPrb = 260;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 10~272, CFG 2305~2312
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.startRnti = 15;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.startPrb = 10;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 36~272, 13 OFDM symbols, CFG 2313~2320
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.startPrb = 28;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 15~260, 13 OFDM symbols, CFG 2321~2328
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.startRnti = 15;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.startPrb = 12;
cfgUeg.endPrb = 260;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 47~272, 13 OFDM symbols, CFG 2329~2336
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.startPrb = 37;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 15~272, 13 OFDM symbols, CFG 2337~2344
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.startRnti = 15;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.startPrb = 12;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 26~272, 12 OFDM symbols (for pattern 79a/79b slot 4/14), CFG 2345~2352
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.Nsym = 12;
cfgUeg.startPrb = 26;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 10~260, 12 OFDM symbols (for pattern 79a/79b slot 5), CFG 2353~2360
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.startRnti = 15;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.Nsym = 12;
cfgUeg.startPrb = 10;
cfgUeg.endPrb = 260;
cfgUeg.nl = 1;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 1 UEG, 8 UEs per UEG, 1 layer per UE, PUSCH prb 10~272, 12 OFDM symbols (for pattern 79a/79b slot 15), CFG 2361~2368
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PUSCH{end, 1} + 1;
cfgUeg.startRnti = 15;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 8;
cfgUeg.Nsym = 12;
cfgUeg.startPrb = 10;
cfgUeg.endPrb = 272;
cfgUeg.nl = 1;
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PUSCH = [CFG_PUSCH; CFG_PUSCH_temp];

% Check if first column (pxsch_cfg_idx) values are unique
first_col = cell2mat(CFG_PUSCH(:,1));
if length(unique(first_col)) ~= length(first_col)
    [unique_vals, ~, idx] = unique(first_col);
    counts = accumarray(idx, 1);
    duplicate_vals = unique_vals(counts > 1);
    error('genPxschUegCfg:DuplicateIdx', ...
          'First column (pxsch_cfg_idx) contains duplicate values: %s\n', mat2str(duplicate_vals));
end


CFG_SRS = {
  % TC#  rnti  Nap nSym  Nrep sym0 cfgIdx seqId bwIdx cmbSz cmbOffset cs  fPos fShift frqH grpH resType Tsrs  Toffset idxSlot N_UE 
    1    1     1    1     1    13    12     0    0     2        0      0    0    223    0    0     0      1      0     0       1; % 8001: base TC, SRS on PRB 223~270, OFDM symbol 13
    2    1     2    1     1    13    12     0    0     2        0      0    0    223    0    0     0      1      0     0       1; % 8003: SRS on PRB 223~270, OFDM symbol 13, Nap = 2
    3    1     1    1     1    13    63     0    0     4        0      0    0    0      0    0     0      1      0     0       4; % 8028: 4 combs 4 users, wideband on OFDM symbol 13
    4    1     1    2     2    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0       4; % 8028: 4 combs 4 users, wideband on OFDM symbol 12 13, Nsym = 2
    5    1     1    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0       8; % 8 users wideband. Time and comb multiplexed. UE 1~4 on OFDM symbol 12; UE 5~8 on OFDM symbol 13
    6 20001    1    1     1    13    10     0    0     2        0      0    0    12     0    0     2      40     3     3       1; % single UE     
    % 64TR
    7    1     4    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0       1; % 1 user wideband
    8    1     4    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0       2; % 2 users wideband
    9    1     4    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0       8; % 8 users wideband
   10    1     4    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0       8; % 8 users wideband - unused
    % 64TR perf
   11    1     4    1     1    10    63     0    0     4        0      0    0    0      0    0     0      1      0     0      16; % 16 users wideband
   % 64TR test case
   12    7     2    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0       8; % 8 users wideband. Time and comb multiplexed. UE 1~4 on OFDM symbol 12; UE 5~8 on OFDM symbol 13
   13   15     2    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0       8; % 8 users wideband. Time and comb multiplexed. UE 1~4 on OFDM symbol 12; UE 5~8 on OFDM symbol 13
   % consucutive SRS slots with different RNTI
   14    9     1    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0       8; % 8 users wideband. Time and comb multiplexed. UE 1~4 on OFDM symbol 12; UE 5~8 on OFDM symbol 13
   15   17     1    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0       8; % 8 users wideband. Time and comb multiplexed. UE 1~4 on OFDM symbol 12; UE 5~8 on OFDM symbol 13
   % 64TR additional SRS symbols
   16   15     4    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0       8; % 8 users wideband + OFDM symbols 12,13
   17   25     4    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0       8; % 8 users wideband + OFDM symbols 12,13 - rnti different
   18   35     4    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0       8; % 8 users wideband + OFDM symbols 12,13 - rnti different
   19   45     4    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0       8; % 8 users wideband + OFDM symbols 12,13 - rnti different
   % 64TR test case, 4 OFDM symbols
   20    1     2    1     1    10    63     0    0     4        0      0    0    0      0    0     0      1      0     0      16; % 16 users wideband. Time and comb multiplexed. UE 1~4 on OFDM symbol 10; UE 5~8 on OFDM symbol 11; UE 9~12 on OFDM symbol 12; UE 13~16 on OFDM symbol 13
   21   17     2    1     1    10    63     0    0     4        0      0    0    0      0    0     0      1      0     0      16; % 16 users wideband. Time and comb multiplexed. UE 1~4 on OFDM symbol 10; UE 5~8 on OFDM symbol 11; UE 9~12 on OFDM symbol 12; UE 13~16 on OFDM symbol 13
   % 64TR additional 2-port SRS w/ different RNTI
   22   25     2    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0       8; % 8 users wideband. Time and comb multiplexed. UE 1~4 on OFDM symbol 12; UE 5~8 on OFDM symbol 13
   23   35     2    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0       8; % 8 users wideband. Time and comb multiplexed. UE 1~4 on OFDM symbol 12; UE 5~8 on OFDM symbol 13
   24    7     2    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0      24; % 24 users wideband. Time and comb multiplexed. UE 1~12 on OFDM symbol 12; UE 13~24 on OFDM symbol 13
   25  101     4    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0       1; % 1 user wideband, high RNTI
   26   31     2    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0      24; % 24 users wideband. Time and comb multiplexed. UE 1~12 on OFDM symbol 12; UE 13~24 on OFDM symbol 13
   27    7     2    1     1    10    63     0    0     4        0      0    0    0      0    0     0      1      0     0      80; % 80 users wideband. Time and comb multiplexed. UE 1~20 on OFDM symbol 10; UE 21~40 on OFDM symbol 11; UE 41~60 on OFDM symbol 12; UE 61~80 on OFDM symbol 13
   28   87     2    1     1    10    63     0    0     4        0      0    0    0      0    0     0      1      0     0      80; % 80 users wideband. Time and comb multiplexed. UE 1~20 on OFDM symbol 10; UE 21~40 on OFDM symbol 11; UE 41~60 on OFDM symbol 12; UE 61~80 on OFDM symbol 13
   % 64TR 'worst case' column G test case, 40 MHz
   29    7     2    1     1    12    25     0    0     4        0      0    0    0      0    0     0      1      0     0      24; % 24 users wideband 104 PRBs. Time and comb multiplexed. UE 1~12 on OFDM symbol 12; UE 13~24 on OFDM symbol 13
   30   31     2    1     1    12    25     0    0     4        0      0    0    0      0    0     0      1      0     0      24; % 24 users wideband 104 PRBs. Time and comb multiplexed. UE 1~12 on OFDM symbol 12; UE 13~24 on OFDM symbol 13
   % 64TR SRS with different RNTI
   31  201     2    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0      24; % 24 users wideband. Time and comb multiplexed. UE 1~12 on OFDM symbol 12; UE 13~24 on OFDM symbol 13
   32  225     2    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0      24; % 24 users wideband. Time and comb multiplexed. UE 1~12 on OFDM symbol 12; UE 13~24 on OFDM symbol 13
   33  249     2    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0      24; % 24 users wideband. Time and comb multiplexed. UE 1~12 on OFDM symbol 12; UE 13~24 on OFDM symbol 13
   34  273     2    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0      24; % 24 users wideband. Time and comb multiplexed. UE 1~12 on OFDM symbol 12; UE 13~24 on OFDM symbol 13
   35    7     4    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0      16; % 16 users wideband. Time and comb multiplexed. UE 1~12 on OFDM symbol 12; UE 13~24 on OFDM symbol 13
   36    7     2    1     1    10    63     0    0     4        0      0    0    0      0    0     0      1      0     0      48; % 48 users wideband. Time and comb multiplexed. UE 1~12 on OFDM symbol 10; UE 13~24 on OFDM symbol 11; UE 25~36 on OFDM symbol 12; UE 37~48 on OFDM symbol 13
   37   55     2    1     1    13    63     0    0     4        0      0    0    0      0    0     0      1      0     0      12; % 12 users wideband. Time and comb multiplexed. UE 1~12 on OFDM symbol 13
   38    7     2    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0      24; % 24 users wideband. Time and comb multiplexed. UE 1~12 on OFDM symbol 12; UE 13~24 on OFDM symbol 13, prgSize = 4
   39    7     2    1     1    12    56     0    0     4        0      0    0    0      0    0     0      1      0     0      24; % 24 users wideband. Time and comb multiplexed. UE 1~12 on OFDM symbol 12; UE 13~24 on OFDM symbol 13, 240 PRBs, 90 MHz bandwidth  (244 PRBs)
   40    7     2    1     1    12    42     0    0     4        0      0    0    0      0    0     0      1      0     0      24; % 24 users wideband. Time and comb multiplexed. UE 1~12 on OFDM symbol 12; UE 13~24 on OFDM symbol 13, 160 PRBs, 60 MHz bandwidth (160 PRBs)
   41    7     4    1     1    10    63     0    0     4        0      0    0    0      0    0     0      1      0     0      24; % 24 users wideband. Time and comb multiplexed. UE 1~4 on OFDM symbol 10; UE 5~8 on OFDM symbol 11; UE 9~12 on OFDM symbol 12; UE 13~16 on OFDM symbol 13
   42   31     4    1     1    12    63     0    0     4        0      0    0    0      0    0     0      1      0     0      12; % 12 users wideband. Time and comb multiplexed. UE 1~6 on OFDM symbol 12; UE 7~12 on OFDM symbol 13
   };

[NallTest, ~] = size(CFG);
errCnt = 0;
detErr = 0;
nCuphyTV = 0;
nFapiTV = 0;
nComp = 0;

if (isnumeric(caseSet))
    caseSetStr = num2str(caseSet);
else
    caseSetStr = caseSet;
end
fprintf('ULMIX: genTV = %d, testCompliance = %d, caseSet = %s', genTV, testCompliance, caseSetStr);
fprintf('\nTC#  slotIdx cell  prach  pucch  pusch   srs\n');
fprintf('--------------------------------------------\n');

parfor n = 1:NallTest
    testAlloc = [];
    testAlloc.dl = 0;
    testAlloc.ul = 1;
    idxSet = n;
    caseNum = CFG{idxSet, 1};
    slotIdx = CFG{idxSet, 2};    
    if ismember(caseNum, TcToTest) && (mod(caseNum, subSetMod(2)) == subSetMod(1))
        if caseNum == 710
            rng(0);
        elseif caseNum == 602
            rng(1);
        elseif caseNum == 883
            rng(2);
        elseif ismember(caseNum, [834, 854])
            rng(100);
        elseif caseNum == 840
            rng(200);
        elseif caseNum == 859
            rng(300);
        elseif caseNum == 1064
            rng(400);
        elseif caseNum == 1100
            rng(500);
        elseif ismember(caseNum, [1128, 2107, 2451, 2453, 2487, 2519, 2524, 2525, 3589, 3630, 4254, 4358, 5397])
            rng(600);
        elseif ismember(caseNum, [1198, 1221, 1223, 1300, 1341, 1359, 1392, 1810, 2116, 2462, 2571, 2237, 2446, 2217, 2474, 2453, 2885, 3307, 3413, 4316, 4342, 4382, 4242, 4573, 5225, 5534, 5794, 5812, 6140, 21845])
            rng(caseNum + 100);
        else
            rng(caseNum);
        end
        
        SysPar = initSysPar(testAlloc);       
        SysPar.SimCtrl.relNum = relNum;        
        SysPar.SimCtrl.N_frame = 1;
        if genTV 
            if ismember(caseNum, TcFapiOnly) % disable cuPHY TV generation
                SysPar.SimCtrl.genTV.cuPHY = 0;
            else
                SysPar.SimCtrl.genTV.cuPHY = 1; % enable cuPHY TV generation
                nCuphyTV = nCuphyTV + 1;
            end
            SysPar.SimCtrl.genTV.FAPI = 1; % FAPI TV is always generated
            nFapiTV = nFapiTV + 1;

            SysPar.SimCtrl.genTV.enable = 1;
            SysPar.SimCtrl.genTV.slotIdx = slotIdx;
            SysPar.SimCtrl.genTV.forceSlotIdxFlag = 1;            
            if caseNum < 1000
                SysPar.SimCtrl.genTV.TVname = sprintf('TVnr_%04d', caseNum);
            else
                SysPar.SimCtrl.genTV.TVname = sprintf('TVnr_ULMIX_%04d', caseNum);
            end
            if ismember(caseNum, disabled_TC)
                if caseNum < 1000
                    SysPar.SimCtrl.genTV.TVname = sprintf('disabled_TVnr_%04d', caseNum);
                else
                    SysPar.SimCtrl.genTV.TVname = sprintf('disabled_TVnr_ULMIX_%04d', caseNum);
                end
            end
        end


        % FAPI override for negative test cases
        if (caseNum == 609)
            SysPar.SimCtrl.negTV.enable = 1;  % enable negTV
            SysPar.SimCtrl.negTV.channel = [SysPar.SimCtrl.negTV.channel, {'PUSCH'}];
            SysPar.SimCtrl.negTV.pduFieldName = {'csiPart1BitLength', 'rank'};
            SysPar.SimCtrl.negTV.pduFieldValue = {33, 1};
        end

        % FAPI override for negative test cases
        if (caseNum == 611)
            SysPar.SimCtrl.negTV.enable = 1;  % enable negTV
            SysPar.SimCtrl.negTV.channel = [SysPar.SimCtrl.negTV.channel, {'PUSCH'}];
            SysPar.SimCtrl.negTV.pduFieldName = {'csiPart1BitLength', 'rank'};
            SysPar.SimCtrl.negTV.pduFieldValue = {6, 1};
        end

        % FAPI override for negative test cases
        if (caseNum == 3840)||(caseNum == 6996)
            SysPar.SimCtrl.negTV.enable = 1;  % enable negTV
            SysPar.SimCtrl.negTV.channel = [SysPar.SimCtrl.negTV.channel, {'PRACH'}];
            SysPar.SimCtrl.negTV.roConfigFieldName = {'prachZeroCorrConf'};
            SysPar.SimCtrl.negTV.roConfigFieldValue = {16};
        end

        % config algSel
        if isfield(algSel, 'pusch')
            if isfield(algSel.pusch, 'enable_sub_slot_proc')
                if algSel.pusch.enable_sub_slot_proc == 1
                    SysPar.SimCtrl.subslot_proc_option = 2;
                    SysPar.SimCtrl.alg.enable_avg_nCov_prbs_fd = 1;
                    SysPar.SimCtrl.alg.win_size_avg_nCov_prbs_fd = 3;
                    SysPar.SimCtrl.alg.threshold_avg_nCov_prbs_fd_dB = 3;
                    SysPar.SimCtrl.alg.enable_instant_equ_coef_cfo_corr = 1;
                end
            end
        end

        % config Carrier
        % mu N_grid_size_mu N_ID_CELL Nant_gNB Nant_UE
        idxCfg = CFG{n,3};
        SysPar.carrier.mu = CFG_Cell{idxCfg, 1};
        SysPar.carrier.N_grid_size_mu = CFG_Cell{idxCfg, 2};
        SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3};
        if ismember(caseNum, [522 : 537]) % 4 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 522, 4);
        elseif ismember(caseNum, [540 : 555]) % 4 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 540, 4);
        elseif ismember(caseNum, [556 : 571]) % 4 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 556, 4);
        elseif ismember(caseNum, [572 : 603]) % 8 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 572, 8);
        elseif ismember(caseNum, [604 : 635]) % 8 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 604, 8);
        elseif ismember(caseNum, [636 : 667]) % 8 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 636, 8);
        elseif ismember(caseNum, [668 : 699]) % 8 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 668, 8);
        elseif ismember(caseNum, [700 : 763]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 700, 16);
        elseif ismember(caseNum, [828 : 891]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 828, 16);
        elseif ismember(caseNum, [892 : 955]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 892, 16);
        elseif ismember(caseNum, [1000 : 1063]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 1000, 16);
        elseif ismember(caseNum, [1064 : 1127]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 1064, 16);
        elseif ismember(caseNum, [1128 : 1159]) % 8 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 1128, 8);
        elseif ismember(caseNum, [1160 : 1223]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 1160, 16);
        elseif ismember(caseNum, [1224 : 1287]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 1224, 16);
        elseif ismember(caseNum, [1288 : 1351]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 1288, 16);
        elseif ismember(caseNum, [1352 : 1383]) % 8 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 1352, 8);
        elseif ismember(caseNum, [1384 : 1415]) % 8 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 1384, 8);
        elseif ismember(caseNum, [1416 : 1479]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 1416, 16);
        elseif ismember(caseNum, [1480 : 1511]) % 8 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 1480, 8);
        elseif ismember(caseNum, [1512 : 1575]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 1512, 16);
        elseif ismember(caseNum, [1584 : 1631]) % 12 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 1584, 12);
        elseif ismember(caseNum, [1632 : 1695]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 1632, 16);
        elseif ismember(caseNum, [1696 : 1743]) % 12 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 1696, 12);
        elseif ismember(caseNum, [1744 : 1791]) % 12 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 1744, 12);
        elseif ismember(caseNum, [1792 : 1839]) % 12 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 1792, 12);
        elseif ismember(caseNum, [1840 : 1903]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 1840, 16);
        elseif ismember(caseNum, [1904 : 1967]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 1904, 16);
        elseif ismember(caseNum, [1968 : 2015]) % 12 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 1968, 12);
        elseif ismember(caseNum, [2016 : 2079]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 2016, 16);
        elseif ismember(caseNum, [2080 : 2143]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 2080, 16);
        elseif ismember(caseNum, [2144 : 2223]) % 20 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 2144, 20);
        elseif ismember(caseNum, [2224 : 2287]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 2224, 16);
        elseif ismember(caseNum, [2416 : 2495]) % 20 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 2416, 20);
        elseif ismember(caseNum, [2304 : 2351]) % 12 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 2304, 12);
        elseif ismember(caseNum, [2352 : 2415]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 2352, 16);
        elseif ismember(caseNum, [2496 : 2559]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 2496, 16);
        elseif ismember(caseNum, [2560 : 2639]) % 20 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 2560, 20);
        elseif ismember(caseNum, [2896 : 2959]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 2896, 16);
        elseif ismember(caseNum, [2960 : 3039]) % 20 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 2960, 20);
        elseif ismember(caseNum, [2640 : 2687]) % 12 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 2640, 12);
        elseif ismember(caseNum, [2688 : 2735]) % 12 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 2688, 12);
        elseif ismember(caseNum, [2736 : 2799]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 2736, 16);
        elseif ismember(caseNum, [2800 : 2863]) % 16 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 2800, 16);
        elseif ismember(caseNum, [2864 : 2895]) % 8 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 2864, 8);
        elseif ismember(caseNum, [3040 : 3119]) % 20 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 3040, 20);
        elseif ismember(caseNum, [3200 : 3279]) % 20 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 3200, 20);
        elseif ismember(caseNum, [3280 : 3359]) % 20 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 3280, 20);
        elseif ismember(caseNum, [3360 : 3439]) % 20 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 3360, 20);
        elseif ismember(caseNum, [3520 : 3599]) % 20 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 3520, 20);
        elseif ismember(caseNum, [3760 : 3839]) % 20 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 3760, 20);
        elseif ismember(caseNum, [4220 : 4299]) % 20 cells
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + mod(caseNum - 4220, 20);
        end

        switch caseNum
            case 520
                SysPar.carrier.N_ID_CELL = 41;
            case 521
                SysPar.carrier.N_ID_CELL = 42;
            case 522
                SysPar.carrier.N_ID_CELL = 43;
            case {530, 531, 532, 533, 534, 535}
                SysPar.carrier.N_ID_CELL = 41;
            otherwise
                % TBD
        end   

        if ismember(caseNum, [550:589, MIMO_64TR_TC]) && ~CellIdxInPatternMap.isKey(caseNum)  % 64TR
            SysPar.carrier.N_ID_CELL = 41;
        end
        
        if ismember(caseNum, [21003:21195]) % 64TR
            SysPar.carrier.N_ID_CELL = 41 + floor((caseNum-21000)/40);
        end
        
        if ismember(caseNum, [21203:21415]) % 64TR
            SysPar.carrier.N_ID_CELL = 41 + floor((caseNum-21200)/20);
        end

        if ismember(caseNum, [21423:21475]) % 64TR
            SysPar.carrier.N_ID_CELL = 41 + floor((caseNum-21420)/20);
        end

        % negative TVs placed on the same cell
        if ismember(caseNum, negative_TC)
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3};   
        end

        % overwrite N_ID_CELL using CellIdxInPatternMap
        if CellIdxInPatternMap.isKey(caseNum)
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + CellIdxInPatternMap(caseNum);
        end   % end cell id config
        
        SysPar.carrier.Nant_gNB = CFG_Cell{idxCfg, 4};
        SysPar.carrier.Nant_UE = CFG_Cell{idxCfg, 5};
        if idxCfg == 8 % 32T32R
            SysPar.SimCtrl.enable_dynamic_BF = 1;
            SysPar.carrier.N_FhPort_DL = 8;
            SysPar.carrier.N_FhPort_UL = 4;
            SysPar.carrier.Nant_gNB_srs = 32;
        end
        
        if ismember(caseNum, 3925)
            SysPar.carrier.SFN_start = 964;
        end

        if ismember(idxCfg, [11, 12, 13, 14, 15]) % 64TR static and dynamic beamforming
            SysPar.SimCtrl.enable_static_dynamic_beamforming = 1;
            SysPar.carrier.N_FhPort_UL = 4; % default num of RX streams for static beamforming
            SysPar.carrier.Nant_gNB_srs = 64;            
        end

        % PRACH
        cfg = CFG{idxSet, 4};
        for idx = 1:length(cfg)
            idxCfg = find(cellfun(@(x) isequal(x,cfg{idx}), CFG_PRACH(:,1)));
            SysPar.prach{idx} = cfgPrach;            
            % cfg#   duplex  mu  cfg restrictSet root  zone  prmbIdx  RA0
            SysPar.carrier.duplex =  CFG_PRACH{idxCfg, 2};
            SysPar.carrier.mu =  CFG_PRACH{idxCfg, 3};
            SysPar.prach{idx}.configurationIndex = CFG_PRACH{idxCfg, 4};
            SysPar.prach{idx}.restrictedSet = CFG_PRACH{idxCfg, 5};
            SysPar.prach{idx}.rootSequenceIndex = CFG_PRACH{idxCfg, 6};
            SysPar.prach{idx}.zeroCorrelationZone = CFG_PRACH{idxCfg, 7};
            SysPar.prach{idx}.prmbIdx = CFG_PRACH{idxCfg, 8};
            SysPar.prach{idx}.n_RA_start = CFG_PRACH{idxCfg, 9};
            if  VARY_PRB_NUM > 0  % in study of sweeping PRB numbers, overwrite prach start
                if ismember(caseNum, [1856:1871, 2032:2047, 2256:2271, 3104:3119]) % pattern 44 with prach, pattern 47, 51, 60 with 4 prach
                    PRB_num_w_prach = min(VARY_PRB_NUM, floor((273-48-1)/6));
                    prach_start_PRB_idx = 1 + PRB_num_w_prach * 6;
                    SysPar.prach{idx}.n_RA_start = (idx - 1) * 12 + prach_start_PRB_idx;
                elseif ismember(caseNum, [2064:2079, 2288:2303, 3136:3151]) % pattern 47, 51, 60 with 3 prach
                    PRB_num_w_prach = min(VARY_PRB_NUM, floor((273-36-1)/6));
                    prach_start_PRB_idx = 1 + PRB_num_w_prach * 6;
                    SysPar.prach{idx}.n_RA_start = (idx - 1) * 12 + prach_start_PRB_idx;
                end
            end
            SysPar.prach{idx}.allSubframes = 1;
            
            SysPar.prach{idx}.idxUE = idx-1;  % use 0-indexing for idxUE
            
            SysPar.prach{1}.msg1_FDM = length(cfg); % only on {1} based on 0513
            SysPar.SimCtrl.timeDomainSim = 1;
        end
        
        % HARQ TC
        if caseNum == 555
            SysPar.prach{1}.idxUE = [0 1];
            SysPar.prach{2}.idxUE = [2 3];
            SysPar.prach{3}.idxUE = [4];
            SysPar.prach{4}.idxUE = [5];
        elseif caseNum == 575
            SysPar.prach{1}.idxUE = [0 1];
            SysPar.prach{2}.idxUE = [2 3];
            SysPar.prach{3}.idxUE = [4 5];
        elseif ismember(caseNum, [590]) % 64TR
            SysPar.prach{1}.idxUE = [0 1];
            SysPar.prach{2}.idxUE = [2 3];
            SysPar.prach{3}.idxUE = [4];
            SysPar.prach{4}.idxUE = [5];          
        elseif ismember(caseNum, [705, 725, 745, 785]) % 64TR
            SysPar.prach{1}.idxUE = [0];           
            SysPar.prach{2}.idxUE = [1];
            SysPar.prach{3}.idxUE = [2];
            SysPar.prach{4}.idxUE = [3];
        elseif ismember(caseNum, [825, 835, 845, 855]) % 64TR test case
            SysPar.prach{1}.idxUE = [0];            
        end
        
        testAlloc.prach = length(cfg);
        
        % default
        if ~PrachBeamIdxMap.isKey(caseNum)
            for idxUe = 1:length(SysPar.prach)
                digBFInterfaces = SysPar.carrier.N_FhPort_UL;
                SysPar.prach{idxUe}.digBFInterfaces = digBFInterfaces;
                SysPar.prach{idxUe}.beamIdx = [1:digBFInterfaces];
            end
        end

        if ismember(caseNum, [MIMO_64TR_TC]) || PrachBeamIdxMap.isKey(caseNum)  % 64TR or has mapping
            beamIdx_static_offset = beamIdx_prach_static(1);
            for idxUe = 1:length(SysPar.prach)
                if (PrachBeamIdxMap.isKey(caseNum))  % by mapping method
                    digBFInterfaces = length(PrachBeamIdxMap(caseNum));
                    SysPar.prach{idxUe}.digBFInterfaces = digBFInterfaces;
                    SysPar.prach{idxUe}.beamIdx = beamIdx_static_offset + PrachBeamIdxMap(caseNum) + idxUe * digBFInterfaces;
                else  
                    if ismember(caseNum, [820:879,21480:21925])
                        digBFInterfaces = 2;
                    else
                        digBFInterfaces = 4;
                    end
                    SysPar.prach{idxUe}.digBFInterfaces = digBFInterfaces;
                    SysPar.prach{idxUe}.beamIdx = beamIdx_static_offset + [1:digBFInterfaces];
                    beamIdx_static_offset = beamIdx_static_offset + digBFInterfaces;
                end
            end
        end     
        
        prach_cfg_idx = CFG{idxSet, 4};
        if ismember(cell2mat(prach_cfg_idx), [65, 66, 67, 68]) % 64TR
            for idxUe = 1:length(SysPar.prach)
                digBFInterfaces = 4;
                SysPar.prach{idxUe}.digBFInterfaces = digBFInterfaces;
                SysPar.prach{idxUe}.beamIdx = beamIdx_prach_static(1) + (PrachBeamIdxMap(caseNum) + idxUe - 1) * digBFInterfaces + [1:digBFInterfaces];
            end
        end  

        cfg = CFG{idxSet, 5};
        testAlloc.pucch = 0;
        totNumUe = 0;
        for idx = 1:length(cfg)
            idxCfg = find(cellfun(@(x) isequal(x,cfg{idx}), CFG_PUCCH(:,1)));
            % cfg# N_UE    Nb format    f0       f1    Nf t0  Nt cs0 tOCCidx freqH groupH SRFlag posSR DTX thr
            nUe = CFG_PUCCH{idxCfg, 2};
   
            for idxUe = 1:nUe
                pucchTemp = cfgPucch;

                if length(CFG_PUCCH{idxCfg, 3})==1 pucchTemp.BitLenHarq = CFG_PUCCH{idxCfg, 3}; ...
                else pucchTemp.BitLenHarq = CFG_PUCCH{idxCfg, 3}(idxUe); end
                
                if length(CFG_PUCCH{idxCfg, 4})==1 pucchTemp.FormatType = CFG_PUCCH{idxCfg, 4}; ...
                else pucchTemp.FormatType = CFG_PUCCH{idxCfg, 4}(idxUe); end
                
                if length(CFG_PUCCH{idxCfg, 5})==1 pucchTemp.prbStart = CFG_PUCCH{idxCfg, 5}; ...
                else pucchTemp.prbStart = CFG_PUCCH{idxCfg, 5}(idxUe); end
            
                if length(CFG_PUCCH{idxCfg, 6})==1 pucchTemp.secondHopPRB = CFG_PUCCH{idxCfg, 6}; ...
                else pucchTemp.secondHopPRB = CFG_PUCCH{idxCfg, 6}(idxUe); end
            
                if length(CFG_PUCCH{idxCfg, 7})==1 pucchTemp.prbSize = CFG_PUCCH{idxCfg, 7}; ...
                else pucchTemp.prbSize = CFG_PUCCH{idxCfg, 7}(idxUe); end
            
                if length(CFG_PUCCH{idxCfg, 8})==1 pucchTemp.startSym = CFG_PUCCH{idxCfg, 8}; ...
                else pucchTemp.startSym = CFG_PUCCH{idxCfg, 8}(idxUe); end
            
                if length(CFG_PUCCH{idxCfg, 9})==1 pucchTemp.nSym = CFG_PUCCH{idxCfg, 9}; ...
                else pucchTemp.nSym = CFG_PUCCH{idxCfg, 9}(idxUe); end
            
                if length(CFG_PUCCH{idxCfg, 10})==1 pucchTemp.cs0 = CFG_PUCCH{idxCfg, 10}; ...
                else pucchTemp.cs0 = CFG_PUCCH{idxCfg, 10}(idxUe); end
            
                if length(CFG_PUCCH{idxCfg, 11})==1 pucchTemp.tOCCidx = CFG_PUCCH{idxCfg, 11}; ...
                else pucchTemp.tOCCidx = CFG_PUCCH{idxCfg, 11}(idxUe); end
            
                if length(CFG_PUCCH{idxCfg, 12})==1 pucchTemp.freqHopFlag = CFG_PUCCH{idxCfg, 12}; ...
                else pucchTemp.freqHopFlag = CFG_PUCCH{idxCfg, 12}(idxUe); end
            
                if length(CFG_PUCCH{idxCfg, 13})==1 pucchTemp.groupHopFlag = CFG_PUCCH{idxCfg, 13}; ...
                else pucchTemp.groupHopFlag = CFG_PUCCH{idxCfg, 13}(idxUe); end
            
                if length(CFG_PUCCH{idxCfg, 14})==1 pucchTemp.SRFlag = CFG_PUCCH{idxCfg, 14}; ...
                else pucchTemp.SRFlag = CFG_PUCCH{idxCfg, 14}(idxUe); end
            
                if length(CFG_PUCCH{idxCfg, 15})==1 pucchTemp.positiveSR = CFG_PUCCH{idxCfg, 15}; ...
                else pucchTemp.positiveSR = CFG_PUCCH{idxCfg, 15}(idxUe); end
            
                if length(CFG_PUCCH{idxCfg, 16})==1 pucchTemp.DTX = CFG_PUCCH{idxCfg, 16}; ...
                else pucchTemp.DTX = CFG_PUCCH{idxCfg, 16}(idxUe); end
            
                if length(CFG_PUCCH{idxCfg, 17})==1 pucchTemp.DTXthreshold = CFG_PUCCH{idxCfg, 17}; ...
                else pucchTemp.DTXthreshold = CFG_PUCCH{idxCfg, 17}(idxUe); end
                
                pucchTemp.RNTI = idxUe - 1;
                if ismember(caseNum, [522 : 537]) 
                    pucchTemp.RNTI =  idxUe - 1 + 16 * (caseNum - 522);
                elseif ismember(caseNum, [540 : 555]) % RNTI 0 - 255 
                    pucchTemp.RNTI =  idxUe - 1 + 16 * (caseNum - 540);
                elseif ismember(caseNum, [556 : 571]) % RNTI 0 - 255 
                    pucchTemp.RNTI =  idxUe - 1 + 16 * (caseNum - 556);
                elseif ismember(caseNum, [572 : 603]) % RNTI 0 - 511 
                    pucchTemp.RNTI =  idxUe - 1 + 16 * (caseNum - 572);
                elseif ismember(caseNum, [604 : 635]) % RNTI 0 - 511 
                    pucchTemp.RNTI =  idxUe - 1 + 16 * (caseNum - 604);
                elseif ismember(caseNum, [636 : 667]) % RNTI 0 - 511 
                    pucchTemp.RNTI =  idxUe - 1 + 16 * (caseNum - 636);
                elseif ismember(caseNum, [668 : 699]) % (24 UEs) 
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 668);
                elseif ismember(caseNum, [700 : 763]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 700);
                elseif ismember(caseNum, [764 : 827]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 764);                     
                elseif ismember(caseNum, [828 : 891]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 828);
                elseif ismember(caseNum, [892 : 955]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 892);
                elseif ismember(caseNum, [1064 : 1127]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 1064);
                elseif ismember(caseNum, [1128 : 1159]) % (16 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 16 * (caseNum - 1128);
                elseif ismember(caseNum, [1160 : 1223]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 1160);
                elseif ismember(caseNum, [1224 : 1287]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 1224);
                elseif ismember(caseNum, [1288 : 1351]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 1288);
                elseif ismember(caseNum, [1352 : 1383]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 1288);
                elseif ismember(caseNum, [1384 : 1415]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 1384);
                elseif ismember(caseNum, [1416 : 1479]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 1416);
                elseif ismember(caseNum, [1480 : 1511]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 1480);
                elseif ismember(caseNum, [1512 : 1575]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 1512);
                elseif ismember(caseNum, [1584 : 1631]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 1584);
                elseif ismember(caseNum, [1632 : 1695]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 1632);
                elseif ismember(caseNum, [1696 : 1743]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 1696);
                elseif ismember(caseNum, [1744 : 1791]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 1744);
                elseif ismember(caseNum, [1792 : 1839]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 1792);
                elseif ismember(caseNum, [1840 : 1903]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 1840);
                elseif ismember(caseNum, [1904 : 1967]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 1904);
                elseif ismember(caseNum, [1968 : 2015]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 1968);
                elseif ismember(caseNum, [2016 : 2079]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 2016);
                elseif ismember(caseNum, [2080 : 2143]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 2080);
                elseif ismember(caseNum, [2144 : 2223]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 2144);
                elseif ismember(caseNum, [2224 : 2287]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 2224);
                elseif ismember(caseNum, [2416 : 2495]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 2416);
                elseif ismember(caseNum, [2304 : 2351]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 2304);
                elseif ismember(caseNum, [2352 : 2415]) % (12 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 12 * (caseNum - 2352);
                elseif ismember(caseNum, [2496 : 2559]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 2496);
                elseif ismember(caseNum, [2560 : 2639]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 2560);
                elseif ismember(caseNum, [2896 : 2959]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 2896);
                elseif ismember(caseNum, [2960 : 3039]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 2960);
                elseif ismember(caseNum, [2640 : 2687]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 2640);
                elseif ismember(caseNum, [2688 : 2735]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 2688);
                elseif ismember(caseNum, [2736 : 2799]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 2736);
                elseif ismember(caseNum, [2800 : 2863]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 2800);
                elseif ismember(caseNum, [2864 : 2895]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 2864);
                elseif ismember(caseNum, [3040 : 3119]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 3040);
                elseif ismember(caseNum, [3200 : 3279]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 3200);
                elseif ismember(caseNum, [3280 : 3359]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 3280);
                elseif ismember(caseNum, [3360 : 3439]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 3360);
                elseif ismember(caseNum, [3520 : 3599]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 3520);
                elseif ismember(caseNum, [3760 : 3839]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 3760);
                elseif ismember(caseNum, [3925])
                    pucchTemp.RNTI =  idxUe - 1 + 20001;
                elseif ismember(caseNum, [3926 : 3927]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 3926 + 1); % 0~23 used for TV 3924 
                elseif ismember(caseNum, [3928 : 3941]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * mod(caseNum - 3928, 3); % 12 TVs for PUCCH + SRS
                elseif ismember(caseNum, [4220 : 4299]) % (24 UEs)
                    pucchTemp.RNTI =  idxUe - 1 + 24 * (caseNum - 4220);
                end % end pucch rnti config

                if CellIdxInPatternMap.isKey(caseNum)
                    pucchTemp.RNTI = idxUe - 1 + nUe * CellIdxInPatternMap(caseNum);
                end
                
                % additional config
                if ismember(caseNum, [3925])
                    pucchTemp.hoppingId = 1;
                    pucchTemp.sequenceHopFlag = 1;
                    pucchTemp.secondHopPRB = 0;
                end
                            
                pucchTemp.idxUE = totNumUe + idxUe - 1;
                
                SysPar.pucch{totNumUe+idxUe} = pucchTemp;
                SysPar.Chan{totNumUe+idxUe} = cfgChan;              
                
                if ismember(idxCfg, [5])
                    SysPar.pucch{totNumUe+idxUe}.maxCodeRate = 2;
                    SysPar.pucch{totNumUe+idxUe}.pi2Bpsk = 0;
                    SysPar.pucch{totNumUe+idxUe}.AddDmrsFlag = 0;
                elseif ismember(idxCfg, [6])
                    SysPar.pucch{totNumUe+idxUe}.maxCodeRate = 2;
                    SysPar.pucch{totNumUe+idxUe}.pi2Bpsk = 0;
                    SysPar.pucch{totNumUe+idxUe}.AddDmrsFlag = 1;
                end
            end    
            totNumUe = totNumUe + nUe;
            
        end

        % negative test case - adding duplicate RNTI values
        if ismember(caseNum, 608)
            SysPar.pucch{3}.RNTI = SysPar.pucch{1}.RNTI;
            SysPar.pucch{8}.RNTI = SysPar.pucch{1}.RNTI;
            SysPar.pucch{10}.RNTI = SysPar.pucch{1}.RNTI;
            SysPar.pucch{20}.RNTI = SysPar.pucch{1}.RNTI;
        end
        
        testAlloc.pucch = totNumUe;
        
        % testAlloc.pucch = length(cfg);
        
         % read beam id from TV
        for idxUe = 1:length(SysPar.pucch)
            digBFInterfaces = SysPar.carrier.N_FhPort_UL;
            SysPar.pucch{idxUe}.digBFInterfaces = digBFInterfaces;
            SysPar.pucch{idxUe}.beamIdx = [1:digBFInterfaces];
        end
        
        pucch_cfg_idx = cell2mat(CFG{idxSet, 5});
        if ismember(caseNum, [MIMO_64TR_TC]) ... % 64TR
            || any(ismember(pucch_cfg_idx, [54]))  % 64TR PUCCH
            beamIdx_static_offset = beamIdx_pucch_static(1);
            for idxUe = 1:length(SysPar.pucch)
                digBFInterfaces = 4;
                SysPar.pucch{idxUe}.digBFInterfaces = digBFInterfaces;
                SysPar.pucch{idxUe}.beamIdx = beamIdx_static_offset + [1:digBFInterfaces];
                beamIdx_static_offset = beamIdx_static_offset + digBFInterfaces;
            end
        end  

        cfg = CFG{idxSet, 6};
        Chan0 = SysPar.Chan{1};
        if ismember(caseNum, [668 : 955, 1000:1743])
            SysPar.SimCtrl.alg.enableIrc = 1;
            SysPar.SimCtrl.alg.enableRssiMeas = 0;
        
        % 1 dmrs
        elseif ismember(caseNum, [4631:4790, 4220:4299])
            SysPar.SimCtrl.alg.enableCfoEstimation = 0;
            SysPar.SimCtrl.alg.enableCfoCorrection = 0;
            SysPar.SimCtrl.alg.enableToEstimation = 0;
            SysPar.SimCtrl.alg.enableToCorrection = 0;
            SysPar.SimCtrl.alg.TdiMode = 0;
            SysPar.SimCtrl.alg.listLength = 8;
            SysPar.SimCtrl.alg.enableIrc = 1;
            SysPar.SimCtrl.alg.enableRssiMeas = 1;

        % enable OTA features:
        elseif ismember(caseNum, [1744:1967, 1968:2079, 2080:2143, 2144:2223, 2224:2287, 2416:2495, 2304 : 2351,2352:2415, ...
                2496:2559, 2560:2639, 2896:2959, 2960:3039, 2640:2735, 2736:2863, 2864:2895, 3040:3119, 3200:3279, ...
                3280:3359, 3360:3439, 3520:3599, 3760:3839]) ...
                || (enableOtaMap.isKey(caseNum) && enableOtaMap(caseNum) == 1) || ismember(caseNum, MIMO_64TR_TC)  % also enable OTA for 64TR TCs
            SysPar.SimCtrl.alg.enableCfoEstimation = 1;
            SysPar.SimCtrl.alg.enableCfoCorrection = 1;
            SysPar.SimCtrl.alg.enableToEstimation = 1;
            SysPar.SimCtrl.alg.enableToCorrection = 0;
            SysPar.SimCtrl.alg.TdiMode = 1;
            SysPar.SimCtrl.alg.listLength = 8;
            SysPar.SimCtrl.alg.enableIrc = 1;  % end OTA
        end  
        % early harq feature
        %   from algorithm perspective, it is always enabled
        %   acutal early harq config depends on DMRS and alg.lastEarlyHarqSymZeroIndexed, see detPusch.m regarding numberOfEarlyHarqRmBits
        if (earlyHarqDisableMap.isKey(caseNum) && earlyHarqDisableMap(caseNum) == 1)
            SysPar.SimCtrl.alg.enableEarlyHarq = 0;
        end

        if SysPar.SimCtrl.enable_static_dynamic_beamforming == 1
            SysPar.SimCtrl.alg.enableRssiMeas = 1;
        end

        for idx = 1:length(cfg)
            idxCfg = find(cellfun(@(x) isequal(x,cfg{idx}), CFG_PUSCH(:,1)));
            SysPar.pusch{idx} = cfgPusch(SysPar.SimCtrl);
            % TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0
            SysPar.pusch{idx}.mcsTable = CFG_PUSCH{idxCfg, 2};
            SysPar.pusch{idx}.mcsIndex =  CFG_PUSCH{idxCfg, 3};
            SysPar.pusch{idx}.nrOfLayers = CFG_PUSCH{idxCfg, 4};
            SysPar.pusch{idx}.rbStart = CFG_PUSCH{idxCfg, 5};
            SysPar.pusch{idx}.rbSize = CFG_PUSCH{idxCfg, 6};
            if SysPar.SimCtrl.enable_static_dynamic_beamforming == 1
                SysPar.pusch{idx}.prgSize = 2;
                SysPar.pusch{idx}.numPRGs = ceil(SysPar.pusch{idx}.rbSize/SysPar.pusch{idx}.prgSize);
            end
            if ismember(CFG_PUSCH{idxCfg, 1}, [2257:2344])
                SysPar.pusch{idx}.prgSize = 16;
                SysPar.pusch{idx}.numPRGs = ceil(SysPar.pusch{idx}.rbSize/SysPar.pusch{idx}.prgSize);
            end
            if  VARY_PRB_NUM > 0  % in study of sweeping PRB numbers, overwrite pusch PRB numbers 
                if ismember(caseNum, [1840:1855, 1872:1903, 2016:2031, 2048:2063, 2240:2255, 2272:2287, 3088:3103, 3600:3639, 3680:3719]) % no prach pattern 44, 47, 51, 60
                    PRB_num_noPrach = min(VARY_PRB_NUM, floor((273-1)/6));
                    SysPar.pusch{idx}.rbStart = 1 + (idx - 1) * PRB_num_noPrach;
                    SysPar.pusch{idx}.rbSize = PRB_num_noPrach;
                elseif ismember(caseNum, [1856:1871, 2032:2047, 2256:2271, 3104:3119]) % with 4 prach pattern 44, 47, 51
                    PRB_num_w_prach = min(VARY_PRB_NUM, floor((273-48-1)/6));
                    SysPar.pusch{idx}.rbStart = 1 + (idx - 1) * PRB_num_w_prach;
                    SysPar.pusch{idx}.rbSize = PRB_num_w_prach;
                elseif ismember(caseNum, [2064:2079, 2288:2303, 3136:3151]) % with 3 prach pattern 47, 51, 60
                    PRB_num_w_prach = min(VARY_PRB_NUM, floor((273-36-1)/6));
                    SysPar.pusch{idx}.rbStart = 1 + (idx - 1) * PRB_num_w_prach;
                    SysPar.pusch{idx}.rbSize = PRB_num_w_prach;
                end
            end
            SysPar.pusch{idx}.StartSymbolIndex = CFG_PUSCH{idxCfg, 7};
            SysPar.pusch{idx}.NrOfSymbols = CFG_PUSCH{idxCfg, 8};
            SysPar.pusch{idx}.SCID =  CFG_PUSCH{idxCfg, 9};
            SysPar.pusch{idx}.BWPStart =  CFG_PUSCH{idxCfg, 10};
            SysPar.pusch{idx}.BWPSize =  CFG_PUSCH{idxCfg, 11};
            SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12};
            
            if ismember(caseNum, [508 : 519])
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 1 * (caseNum - 508);
            end

            if ismember(caseNum, [536 : 547])
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 4 * (caseNum - 536);
            end

            if ismember(caseNum, [604 : 635])
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 16 * (caseNum - 604);
            elseif ismember(caseNum, [636 : 667]) 
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 16 * (caseNum - 636);
            elseif ismember(caseNum, [668 : 699])
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 16 * (caseNum - 668);
            elseif ismember(caseNum, [700 : 763]) %(6 UEs) 
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 700);
            elseif ismember(caseNum, [764 : 827]) %(6 UEs) 
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 764);                
            elseif ismember(caseNum, [828 : 891]) %  (6 UEs) 
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 828);
            elseif ismember(caseNum, [892 : 955]) %  (6 UEs) 
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 892);
            elseif ismember(caseNum, [1000 : 1063]) %  (1 UEs) 
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 1 * (caseNum - 1000);
            elseif ismember(caseNum, [1064 : 1127]) %  (6 UEs) 
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 1064);
            elseif ismember(caseNum, [1128 : 1159]) %  (16 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 16 * (caseNum - 1128);
            elseif ismember(caseNum, [1160 : 1223]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 1160);
            elseif ismember(caseNum, [1224 : 1287]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 1224);
            elseif ismember(caseNum, [1288 : 1351]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 1288);
            elseif ismember(caseNum, [1352 : 1383]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 1352);
            elseif ismember(caseNum, [1384 : 1415]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 1384);
            elseif ismember(caseNum, [1416 : 1479]) %  (1 UE)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 1 * (caseNum - 1416);
            elseif ismember(caseNum, [1480 : 1511]) %  (1 UE)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 1 * (caseNum - 1480);
            elseif ismember(caseNum, [1512 : 1575]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 1512);
            elseif ismember(caseNum, [1584 : 1631]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 1584);
            elseif ismember(caseNum, [1632 : 1695]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 1632);
            elseif ismember(caseNum, [1696 : 1743]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 1696);
            elseif ismember(caseNum, [1744 : 1791]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 1744);
            elseif ismember(caseNum, [1792 : 1839]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 1792);
            elseif ismember(caseNum, [1840 : 1903]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 1840);
            elseif ismember(caseNum, [1904 : 1967]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 1904);
            elseif ismember(caseNum, [1968 : 2015]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 1968);
            elseif ismember(caseNum, [2016 : 2079]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 2016);
            elseif ismember(caseNum, [2080 : 2143]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 2080);
            elseif ismember(caseNum, [2144 : 2223]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 2144);
            elseif ismember(caseNum, [2224 : 2287]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 2224);
            elseif ismember(caseNum, [2416 : 2495]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 2416);
            elseif ismember(caseNum, [2304 : 2351]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 2304);
            elseif ismember(caseNum, [2352 : 2415]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 2352);
            elseif ismember(caseNum, [2496 : 2559]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 2496);
            elseif ismember(caseNum, [2560 : 2639]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 2560);
            elseif ismember(caseNum, [2896 : 2959]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 2896);
            elseif ismember(caseNum, [2960 : 3039]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 2960);
            elseif ismember(caseNum, [2640 : 2687]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 2640);
            elseif ismember(caseNum, [2688 : 2735]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 2688);
            elseif ismember(caseNum, [2736 : 2799]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 2736);
            elseif ismember(caseNum, [2800 : 2863]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 2800);
            elseif ismember(caseNum, [2864 : 2895]) %  (8 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 8 * (caseNum - 2864);
            elseif ismember(caseNum, [3040 : 3119]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 3040);
            elseif ismember(caseNum, [3200 : 3279]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 3200);
            elseif ismember(caseNum, [3280 : 3359]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 3280);
            elseif ismember(caseNum, [3360 : 3439]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 3360);
            elseif ismember(caseNum, [3520 : 3599]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 3520);
            elseif ismember(caseNum, [3760 : 3839]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 3760);
            elseif ismember(caseNum, [4220 : 4299]) %  (6 UEs)
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 6 * (caseNum - 4220);
            end % 

            if ismember(caseNum, [619 : 630])
                SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 2 * (caseNum - 619); % overwrite RNTI for 619~630
            end
    
            % PUSCH RNTI using mapping method
            if CellIdxInPatternMap.isKey(caseNum)
                if puschDynamicBfMap.isKey(caseNum) && puschDynamicBfMap(caseNum) == 1
                    % same RNTI range for all cells in dynamic beamforming cases since we resue BFW TC RNTI for all cells
                    SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12};
                else
                    % different RNTI range for each cell in static beamforming cases, assuming nUE <= 16
                    SysPar.pusch{idx}.RNTI =  CFG_PUSCH{idxCfg, 12} + 16 * CellIdxInPatternMap(caseNum);
                end
            end
            % enable DFT-s-OFDM and overwrite with 1 layer
            if puschIdentityMap.isKey(caseNum)
                SysPar.SimCtrl.alg.enableDftSOfdm = 1;
                SysPar.pusch{idx}.TransformPrecoding = 0; % enable DFT-s-OFDM
                SysPar.pusch{idx}.puschIdentity = puschIdentityMap(caseNum) + idx;
                SysPar.pusch{idx}.nrOfLayers = 1;  % force 1 layer for DFT-s-OFDM
            end
            
            SysPar.pusch{idx}.rvIndex =  CFG_PUSCH{idxCfg, 13};
            SysPar.pusch{idx}.dataScramblingId =  CFG_PUSCH{idxCfg, 14};
            sym0 = SysPar.pusch{idx}.StartSymbolIndex;
            nSym = SysPar.pusch{idx}.NrOfSymbols;
            dmrs0 = CFG_PUSCH{idxCfg, 15};
            SysPar.carrier.dmrsTypeAPos = dmrs0;
            maxLen = CFG_PUSCH{idxCfg, 16};
            addPos = CFG_PUSCH{idxCfg, 17};
            if addPos == 0 || SysPar.pusch{idx}.nrOfLayers > 4  % || SysPar.carrier.Nant_gNB > 8
                SysPar.SimCtrl.alg.TdiMode = 0;
                SysPar.SimCtrl.alg.enableCfoEstimation = 0;
                SysPar.SimCtrl.alg.enableCfoCorrection = 0;
                SysPar.SimCtrl.alg.enableToEstimation = 0;
            end
            
            if ismember(caseNum, [553, 563, 573, 583, 4040:4079, 5391:5430]) % HARQ TC S-slot
                DmrsSymbPos = findDmrsSymbPos(sym0, nSym, dmrs0, maxLen, addPos, 'UL', 'typeB');
                SysPar.pusch{idx}.DmrsMappingType = 1;
            else
                DmrsSymbPos = findDmrsSymbPos(sym0, nSym, dmrs0, maxLen, addPos, 'UL', 'typeA');
            end
            SysPar.pusch{idx}.DmrsSymbPos = DmrsSymbPos;
            SysPar.pusch{idx}.DmrsScramblingId =  CFG_PUSCH{idxCfg, 18};
            SysPar.pusch{idx}.numDmrsCdmGrpsNoData =  CFG_PUSCH{idxCfg, 19};
            SysPar.pusch{idx}.portIdx = CFG_PUSCH{idxCfg, 20} + [0:SysPar.pusch{idx}.nrOfLayers-1];
            if ismember(caseNum, [2429, 2431, 2432, 2433, 2439, 4098, 4133, 4254, 4242, 4053, 4058, 4073, 5417, 5530, 5534, 5756, 5756, 5760, 5997, 6213, 6247]) % need a different seed value to pass
                SysPar.pusch{idx}.seed = caseNum + 100;
            else
                SysPar.pusch{idx}.seed = caseNum;
            end
            SysPar.pusch{idx}.idxUE = idx-1;
            SysPar.pusch{idx}.idxUeg = CFG_PUSCH{idxCfg, 21};
            % use different harq process id for slot-based cases
            if (harqProcessIdMap.isKey(caseNum) && harqProcessIdMap(caseNum) ~= 0)
                SysPar.pusch{idx}.harqProcessID = harqProcessIdMap(caseNum);
            end

            if (ldpcFixMaxIterMap.isKey(caseNum) && ldpcFixMaxIterMap(caseNum) ~= 0)
                SysPar.SimCtrl.alg.LDPC_maxItr = ldpcFixMaxIterMap(caseNum); % max number of iterations of PUSCH LDPC decoder
                SysPar.SimCtrl.alg.LDPC_DMI_method = 'fixed'; % fixed: fix the num of LDPC iterations, ML: use machine learning model to predict, LUT_spef: use LUT for max num LDPC itrs, per_UE: use per-UE max num LDPC itrs provided by L2. 
            end
            
            % containing UCI
            MAX_NUM_CSI1_PRM     = SysPar.SimCtrl.MAX_NUM_CSI1_PRM;
            MAX_NUM_CSI2_REPORTS = SysPar.SimCtrl.MAX_NUM_CSI2_REPORTS;
            calcCsi2Size_csi2MapIdx = zeros(MAX_NUM_CSI2_REPORTS , 1);
            calcCsi2Size_nPart1Prms = zeros(MAX_NUM_CSI2_REPORTS , 1);
            calcCsi2Size_prmOffsets = zeros(MAX_NUM_CSI1_PRM     , MAX_NUM_CSI2_REPORTS);
            calcCsi2Size_prmSizes   = zeros(MAX_NUM_CSI1_PRM     , MAX_NUM_CSI2_REPORTS);
            calcCsi2Size_prmValues  = zeros(MAX_NUM_CSI1_PRM     , MAX_NUM_CSI2_REPORTS);
%             if ismember(caseNum, [556:603, 636:667])
            
            % UCI on PUSCH initialization
            if ismember(caseNum, [609, 611]) %[636:667])
                SysPar.pusch{idx}.pduBitmap = 2^0 + 2^1 + 2^5;
                SysPar.pusch{idx}.harqAckBitLength = 4;   % number of HARQ bits 
                SysPar.pusch{idx}.csiPart1BitLength = 13;
                SysPar.pusch{idx}.alphaScaling = 3;
                SysPar.pusch{idx}.betaOffsetHarqAck = 11; % Default value in Matlab 5G toolbox = 11
                SysPar.pusch{idx}.betaOffsetCsi1 = 13;    % Default value in Matlab 5G toolbox = 13
                SysPar.pusch{idx}.betaOffsetCsi2 = 13;
                SysPar.pusch{idx}.rank            = 3; % CSI-P2: 4 bits
                SysPar.pusch{idx}.rankBitOffset   = 0;
                SysPar.pusch{idx}.rankBitSize     = 2;
            end
            
            % F08 containing UCI (relaxed comlexity)
            if ismember(caseNum, [668:763, 828:891])
                SysPar.pusch{idx}.pduBitmap = 2^0 + 2^1 + 2^5;
                SysPar.pusch{idx}.harqAckBitLength = 4;   % number of HARQ bits 
                SysPar.pusch{idx}.csiPart1BitLength = 6;
                SysPar.pusch{idx}.alphaScaling = 3;
                SysPar.pusch{idx}.betaOffsetHarqAck = 11; % Default value in Matlab 5G toolbox = 11
                SysPar.pusch{idx}.betaOffsetCsi1 = 13;    % Default value in Matlab 5G toolbox = 13
                SysPar.pusch{idx}.betaOffsetCsi2 = 13;
                SysPar.pusch{idx}.rank            = 1; % CSI-P2: 5 bits
                SysPar.pusch{idx}.rankBitOffset   = 0;
                SysPar.pusch{idx}.rankBitSize     = 2;
            end
            
            % F08 4+33+5, (with 2 SSBs and 4 Prach) 
            if ismember(caseNum, [1064:1127, 1128:1159, 1160:1223, 1224:1287, 1288:1351, 1352:1383,1384:1415, ...
                    1416:1511,1512:1575,1584:1631,1632:1695,1696:1743, 1744:1791, 1792:1839, 1840:1903, 1904:1967, ...
                    2304:2351,2352:2415,  ...
                    2640:2735, 2736:2863, 2864 : 2895])
                
                SysPar.pusch{idx}.pduBitmap = 2^0 + 2^1 + 2^5;
                SysPar.pusch{idx}.harqAckBitLength = 4;   % number of HARQ bits 
                SysPar.pusch{idx}.csiPart1BitLength = 33;
                SysPar.pusch{idx}.alphaScaling = 3;
                SysPar.pusch{idx}.betaOffsetHarqAck = 11; % Default value in Matlab 5G toolbox = 11
                SysPar.pusch{idx}.betaOffsetCsi1 = 13;    % Default value in Matlab 5G toolbox = 13
                SysPar.pusch{idx}.betaOffsetCsi2 = 13;
                SysPar.pusch{idx}.rank            = 1; % CSI-P2: 5 bits
                SysPar.pusch{idx}.rankBitOffset   = 0;
                SysPar.pusch{idx}.rankBitSize     = 2;
            end
            
            % F08 7beams: 4+37+5 UCI
            if ismember(caseNum, [1968:2015, 2016:2079, 2080:2143, 2144:2223, 2224:2287, 2416:2495, 2496:2559, 2560:2639,...
                    2896:2959, 2960:3039, 3040:3119, 3200:3279, 3280:3359, 3360:3439, 3760:3839, 4220:4299]) ...
                || (UciOnPuschMap.isKey(caseNum) &&  UciOnPuschMap(caseNum) == "7beams_4_37_5")
                
                SysPar.pusch{idx}.pduBitmap = 2^0 + 2^1 + 2^5;
                SysPar.pusch{idx}.harqAckBitLength = 4;   % number of HARQ bits 
                SysPar.pusch{idx}.csiPart1BitLength = 37;
                SysPar.pusch{idx}.alphaScaling = 3;
                SysPar.pusch{idx}.betaOffsetHarqAck = 11; % Default value in Matlab 5G toolbox = 11
                SysPar.pusch{idx}.betaOffsetCsi1 = 13;    % Default value in Matlab 5G toolbox = 13
                SysPar.pusch{idx}.betaOffsetCsi2 = 13;
                SysPar.pusch{idx}.rank            = 1; % CSI-P2: 5 bits
                SysPar.pusch{idx}.rankBitOffset   = 0;
                SysPar.pusch{idx}.rankBitSize     = 2;
            end % end F08 7beams UCI

            % F08 7beams: 0+37+5 UCI (for 24 PUCCH UCIs)
            if ismember(caseNum, [3520:3599]) ...
                || (UciOnPuschMap.isKey(caseNum) &&  UciOnPuschMap(caseNum) == "7beams_0_37_5")           
                SysPar.pusch{idx}.pduBitmap = 2^0 + 2^1 + 2^5;
                SysPar.pusch{idx}.harqAckBitLength = 0;   % number of HARQ bits 
                SysPar.pusch{idx}.csiPart1BitLength = 37;
                SysPar.pusch{idx}.alphaScaling = 3;
                SysPar.pusch{idx}.betaOffsetHarqAck = 11; % Default value in Matlab 5G toolbox = 11
                SysPar.pusch{idx}.betaOffsetCsi1 = 13;    % Default value in Matlab 5G toolbox = 13
                SysPar.pusch{idx}.betaOffsetCsi2 = 13;
                SysPar.pusch{idx}.rank            = 1; % CSI-P2: 5 bits
                SysPar.pusch{idx}.rankBitOffset   = 0;
                SysPar.pusch{idx}.rankBitSize     = 2;
            end % end F08 7beams UCI

            % F08 7beams S slot UCI:  % TC-7596
            if ismember(caseNum, [4040:4079]) ...
                || (UciOnPuschMap.isKey(caseNum) &&  UciOnPuschMap(caseNum) == "7beams_0_10_0")
                SysPar.pusch{idx}.pduBitmap = 2^0 + 2^1 + 2^5;
                SysPar.pusch{idx}.harqAckBitLength = 0;
                SysPar.pusch{idx}.csiPart1BitLength = 10;
                SysPar.pusch{idx}.csiPart2BitLength = 0;
                SysPar.pusch{idx}.alphaScaling = 2;
                SysPar.pusch{idx}.betaOffsetHarqAck = 10;
                SysPar.pusch{idx}.betaOffsetCsi1 = 13;
                SysPar.pusch{idx}.betaOffsetCsi2 = 11;
                SysPar.pusch{idx}.rank           = 2;
                SysPar.pusch{idx}.rankBitOffset  = 0;
                SysPar.pusch{idx}.rankBitSize    = 1;
            end % end F08 7beams S slot UCI
            
            % HARQ TC
            if ismember(caseNum, [554, 564, 565, 574, 583, 584, 585])
                if caseNum == 583
                    SysPar.pusch{idx}.pduBitmap = 2^1 + 2^5;
                else
                    SysPar.pusch{idx}.pduBitmap = 2^0 + 2^1 + 2^5;
                end
                SysPar.pusch{idx}.harqAckBitLength = 4;
                SysPar.pusch{idx}.csiPart1BitLength = 37;
                SysPar.pusch{idx}.alphaScaling = 1;
                SysPar.pusch{idx}.betaOffsetHarqAck = 9;
                SysPar.pusch{idx}.betaOffsetCsi1 = 6;
                SysPar.pusch{idx}.betaOffsetCsi2 = 6;
                SysPar.pusch{idx}.rank            = 2;
                SysPar.pusch{idx}.rankBitOffset   = 5;
                SysPar.pusch{idx}.rankBitSize     = 2;
            end
            
            % 64TR test case
            if ismember(caseNum, [824, 825, 834, 835, 844, 845, 854, 855])
                SysPar.pusch{idx}.pduBitmap = 2^0 + 2^1 + 2^5;
                SysPar.pusch{idx}.harqAckBitLength = 4;
                SysPar.pusch{idx}.csiPart1BitLength = 6;
                SysPar.pusch{idx}.alphaScaling = 1;
                SysPar.pusch{idx}.betaOffsetHarqAck = 9;
                SysPar.pusch{idx}.betaOffsetCsi1 = 6;
                SysPar.pusch{idx}.betaOffsetCsi2 = 6;
                SysPar.pusch{idx}.rank            = 2;
                SysPar.pusch{idx}.rankBitOffset   = 1;
                SysPar.pusch{idx}.rankBitSize     = 2;
            end
            if ismember(caseNum, [4791:4910,5431:5970,6996,21480:21519])
                SysPar.pusch{idx}.pduBitmap = 2^0 + 2^1 + 2^5;
                SysPar.pusch{idx}.harqAckBitLength = 4;   % number of HARQ bits
                SysPar.pusch{idx}.csiPart1BitLength = 6;
                SysPar.pusch{idx}.alphaScaling = 3;
                SysPar.pusch{idx}.betaOffsetHarqAck = 11; % Default value in Matlab 5G toolbox = 11
                SysPar.pusch{idx}.betaOffsetCsi1 = 13;    % Default value in Matlab 5G toolbox = 13
                SysPar.pusch{idx}.betaOffsetCsi2 = 13;
                SysPar.pusch{idx}.rank            = 1; % CSI-P2: 5 bits
                SysPar.pusch{idx}.rankBitOffset   = 0;
                SysPar.pusch{idx}.rankBitSize     = 2;
            end
            if ismember(caseNum, [21520:21559, 21580:21619]) || (UciOnPuschMap.isKey(caseNum) && UciOnPuschMap(caseNum) == "64TR_4_6_11")
                SysPar.SimCtrl.enable_multi_csiP2_fapiv3 = 1;
                SysPar.pusch{idx}.pduBitmap = 2^0 + 2^1 + 2^5;
                SysPar.pusch{idx}.nCsi2Reports           = 1;
                SysPar.pusch{idx}.flagCsiPart2           = 65535;
                SysPar.pusch{idx}.harqAckBitLength = 4;   % number of HARQ bits
                SysPar.pusch{idx}.csiPart1BitLength = 6;
                calcCsi2Size_csi2MapIdx(1)   = 1;
                calcCsi2Size_nPart1Prms(1)   = 1;
                calcCsi2Size_prmOffsets(1,1) = 0;
                calcCsi2Size_prmSizes(1,1)   = 6;
                calcCsi2Size_prmValues(1,1)  = 5;
                % Legacy paramaters for computing CSI-P2 size:
                SysPar.pusch{idx}.alphaScaling      = 3;
                SysPar.pusch{idx}.betaOffsetHarqAck = 11; 
                SysPar.pusch{idx}.betaOffsetCsi1    = 13;   
                SysPar.pusch{idx}.betaOffsetCsi2    = 13;
            end
            if (UciOnPuschMap.isKey(caseNum) && UciOnPuschMap(caseNum) == "64TR_0_6_11")
                SysPar.SimCtrl.enable_multi_csiP2_fapiv3 = 1;
                SysPar.pusch{idx}.pduBitmap = 2^0 + 2^1 + 2^5;
                SysPar.pusch{idx}.nCsi2Reports           = 1;
                SysPar.pusch{idx}.flagCsiPart2           = 65535;
                SysPar.pusch{idx}.harqAckBitLength = 0;   % no HARQ bits for UCI-on-PUSCH
                SysPar.pusch{idx}.csiPart1BitLength = 6;
                calcCsi2Size_csi2MapIdx(1)   = 1;
                calcCsi2Size_nPart1Prms(1)   = 1;
                calcCsi2Size_prmOffsets(1,1) = 0;
                calcCsi2Size_prmSizes(1,1)   = 6;
                calcCsi2Size_prmValues(1,1)  = 5;
                % Legacy paramaters for computing CSI-P2 size:
                SysPar.pusch{idx}.alphaScaling      = 3;
                SysPar.pusch{idx}.betaOffsetHarqAck = 11; 
                SysPar.pusch{idx}.betaOffsetCsi1    = 13;   
                SysPar.pusch{idx}.betaOffsetCsi2    = 13;
            end
            if ismember(caseNum, [21560:21579]) || (UciOnPuschMap.isKey(caseNum) && UciOnPuschMap(caseNum) == "64TR_4_31_11")
                SysPar.SimCtrl.enable_multi_csiP2_fapiv3 = 1;
                SysPar.pusch{idx}.pduBitmap = 2^0 + 2^1 + 2^5;
                SysPar.pusch{idx}.harqAckBitLength = 4;   % number of HARQ bits
                SysPar.pusch{idx}.csiPart1BitLength = 31;
                SysPar.pusch{idx}.nCsi2Reports      = 1;
                SysPar.pusch{idx}.flagCsiPart2      = 65535;
                calcCsi2Size_csi2MapIdx(1)   = 1;
                calcCsi2Size_nPart1Prms(1)   = 1;
                calcCsi2Size_prmOffsets(1,1) = 0;
                calcCsi2Size_prmSizes(1,1)   = 6;
                calcCsi2Size_prmValues(1,1)  = 5;
                % Legacy paramaters for computing CSI-P2 size:
                SysPar.pusch{idx}.alphaScaling = 3;
                SysPar.pusch{idx}.betaOffsetHarqAck = 11; % Default value in Matlab 5G toolbox = 11
                SysPar.pusch{idx}.betaOffsetCsi1 = 13;    % Default value in Matlab 5G toolbox = 13
                SysPar.pusch{idx}.betaOffsetCsi2 = 13;
            end
            if ismember(caseNum, [21620:21639]) || (UciOnPuschMap.isKey(caseNum) && UciOnPuschMap(caseNum) == "64TR_4_18_5")
                SysPar.SimCtrl.enable_multi_csiP2_fapiv3 = 1;
                SysPar.pusch{idx}.pduBitmap = 2^0 + 2^1 + 2^5;
                SysPar.pusch{idx}.harqAckBitLength = 4;  % number of HARQ bits
                SysPar.pusch{idx}.csiPart1BitLength = 18;
                SysPar.pusch{idx}.alphaScaling = 3;
                SysPar.pusch{idx}.betaOffsetHarqAck = 11; % Default value in Matlab 5G toolbox = 11
                SysPar.pusch{idx}.betaOffsetCsi1 = 13;    % Default value in Matlab 5G toolbox = 13
                SysPar.pusch{idx}.betaOffsetCsi2 = 13;
                SysPar.pusch{idx}.rank            = 1; % CSI-P2: 5 bits
                SysPar.pusch{idx}.rankBitOffset   = 0;
                SysPar.pusch{idx}.rankBitSize     = 2;
            end
            if ismember(caseNum, [21640:21659, 21770:21909]) || (UciOnPuschMap.isKey(caseNum) && UciOnPuschMap(caseNum) == "64TR_4_18_7")
                SysPar.SimCtrl.enable_multi_csiP2_fapiv3 = 1;
                SysPar.pusch{idx}.pduBitmap = 2^0 + 2^1 + 2^5;
                SysPar.pusch{idx}.harqAckBitLength = 4;  % number of HARQ bits
                SysPar.pusch{idx}.csiPart1BitLength = 18;
                SysPar.pusch{idx}.nCsi2Reports      = 1;
                SysPar.pusch{idx}.flagCsiPart2      = 65535;
                calcCsi2Size_csi2MapIdx(1)   = 3;
                calcCsi2Size_nPart1Prms(1)   = 1;
                calcCsi2Size_prmOffsets(1,1) = 0;
                calcCsi2Size_prmSizes(1,1)   = 6;
                calcCsi2Size_prmValues(1,1)  = 1;
                % Legacy paramaters for computing CSI-P2 size:
                SysPar.pusch{idx}.alphaScaling = 3;
                SysPar.pusch{idx}.betaOffsetHarqAck = 11; % Default value in Matlab 5G toolbox = 11
                SysPar.pusch{idx}.betaOffsetCsi1 = 13;    % Default value in Matlab 5G toolbox = 13
                SysPar.pusch{idx}.betaOffsetCsi2 = 13;
            end

            if ismember(caseNum, [21670:21765,21914:21915,21924:21925]) || (UciOnPuschMap.isKey(caseNum) && UciOnPuschMap(caseNum) == "64TR_4_13_10")
                SysPar.SimCtrl.enable_multi_csiP2_fapiv3 = 1;
                SysPar.pusch{idx}.pduBitmap = 2^0 + 2^1 + 2^5;
                SysPar.pusch{idx}.harqAckBitLength = 4;  % number of HARQ bits
                SysPar.pusch{idx}.csiPart1BitLength = 13;
                SysPar.pusch{idx}.nCsi2Reports      = 1;
                SysPar.pusch{idx}.flagCsiPart2      = 65535;
                calcCsi2Size_csi2MapIdx(1)   = 1;
                calcCsi2Size_nPart1Prms(1)   = 1;
                calcCsi2Size_prmOffsets(1,1) = 0;
                calcCsi2Size_prmSizes(1,1)   = 6;
                calcCsi2Size_prmValues(1,1)  = 2;
                % Legacy paramaters for computing CSI-P2 size:
                SysPar.pusch{idx}.alphaScaling = 3;
                SysPar.pusch{idx}.betaOffsetHarqAck = 11; % Default value in Matlab 5G toolbox = 11
                SysPar.pusch{idx}.betaOffsetCsi1 = 13;    % Default value in Matlab 5G toolbox = 13
                SysPar.pusch{idx}.betaOffsetCsi2 = 13;
            end
            SysPar.pusch{idx}.calcCsi2Size_csi2MapIdx = calcCsi2Size_csi2MapIdx(:);
            SysPar.pusch{idx}.calcCsi2Size_nPart1Prms = calcCsi2Size_nPart1Prms(:);
            SysPar.pusch{idx}.calcCsi2Size_prmOffsets = calcCsi2Size_prmOffsets(:);
            SysPar.pusch{idx}.calcCsi2Size_prmSizes   = calcCsi2Size_prmSizes(:);
            SysPar.pusch{idx}.calcCsi2Size_prmValues  = calcCsi2Size_prmValues(:);


            % HARQ TC
            switch idxCfg
                case 320
                    SysPar.pusch{idx}.harqProcessID = 1;
%                     SysPar.pusch{idx}.targetCodeRate = 9480;
%                     SysPar.pusch{idx}.qamModOrder = 8;
%                     SysPar.pusch{idx}.TBSize = 14347;
%                     SysPar.pusch{idx}.newDataIndicator = 0;
                case 321
                    SysPar.pusch{idx}.targetCodeRate = 9480;
                    SysPar.pusch{idx}.qamModOrder = 8;
                    SysPar.pusch{idx}.TBSize = 14347;
                    SysPar.pusch{idx}.newDataIndicator = 0;
                case 322
                    SysPar.pusch{idx}.targetCodeRate = 9480;
                    SysPar.pusch{idx}.qamModOrder = 8;
                    SysPar.pusch{idx}.TBSize = 14347;
                    SysPar.pusch{idx}.newDataIndicator = 0;
                case 323
                    SysPar.pusch{idx}.targetCodeRate = 9480;
                    SysPar.pusch{idx}.qamModOrder = 8;
                    SysPar.pusch{idx}.TBSize = 14347;
                    SysPar.pusch{idx}.newDataIndicator = 0;
                case 324
                    SysPar.pusch{idx}.targetCodeRate = 9480;
                    SysPar.pusch{idx}.qamModOrder = 8;
                    SysPar.pusch{idx}.TBSize = 7172;
                    SysPar.pusch{idx}.newDataIndicator = 0;
                case 325
                    SysPar.pusch{idx}.targetCodeRate = 9480;
                    SysPar.pusch{idx}.qamModOrder = 8;
                    SysPar.pusch{idx}.TBSize = 3329;
                    SysPar.pusch{idx}.newDataIndicator = 0;
                case 327
                    SysPar.pusch{idx}.targetCodeRate = 9480;
                    SysPar.pusch{idx}.qamModOrder = 8;
                    SysPar.pusch{idx}.TBSize = 14347;
                    SysPar.pusch{idx}.newDataIndicator = 0;
                case 328
                    SysPar.pusch{idx}.targetCodeRate = 9480;
                    SysPar.pusch{idx}.qamModOrder = 6;
                    SysPar.pusch{idx}.TBSize = 14347;
                    SysPar.pusch{idx}.newDataIndicator = 0;
                case 329
                    SysPar.pusch{idx}.targetCodeRate = 9480;
                    SysPar.pusch{idx}.qamModOrder = 6;
                    SysPar.pusch{idx}.TBSize = 14347;
                    SysPar.pusch{idx}.newDataIndicator = 0;
                case 330
                    SysPar.pusch{idx}.targetCodeRate = 9480;
                    SysPar.pusch{idx}.qamModOrder = 6;
                    SysPar.pusch{idx}.TBSize = 14347;
                    SysPar.pusch{idx}.newDataIndicator = 0;
                case 331
                    SysPar.pusch{idx}.targetCodeRate = 9480;
                    SysPar.pusch{idx}.qamModOrder = 6;
                    SysPar.pusch{idx}.TBSize = 7172;
                    SysPar.pusch{idx}.newDataIndicator = 0;
                case 332
                    SysPar.pusch{idx}.targetCodeRate = 9480;
                    SysPar.pusch{idx}.qamModOrder = 6;
                    SysPar.pusch{idx}.TBSize = 3329;
                    SysPar.pusch{idx}.newDataIndicator = 0;
                case 336
                    SysPar.pusch{idx}.harqProcessID = 1;
                    SysPar.pusch{idx}.targetCodeRate = 9480;
                    SysPar.pusch{idx}.qamModOrder = 8;
                    SysPar.pusch{idx}.TBSize = 14347;
                    SysPar.pusch{idx}.newDataIndicator = 0;
                case 337
                    SysPar.pusch{idx}.targetCodeRate = 9480;
                    SysPar.pusch{idx}.qamModOrder = 6;
                    SysPar.pusch{idx}.TBSize = 14347;
                    SysPar.pusch{idx}.newDataIndicator = 0;
                case 338
                    SysPar.pusch{idx}.targetCodeRate = 9480;
                    SysPar.pusch{idx}.qamModOrder = 6;
                    SysPar.pusch{idx}.TBSize = 14347;
                    SysPar.pusch{idx}.newDataIndicator = 0;
                case 339
                    SysPar.pusch{idx}.targetCodeRate = 9480;
                    SysPar.pusch{idx}.qamModOrder = 6;
                    SysPar.pusch{idx}.TBSize = 14347;
                    SysPar.pusch{idx}.newDataIndicator = 0;
                case 340
                    SysPar.pusch{idx}.targetCodeRate = 9480;
                    SysPar.pusch{idx}.qamModOrder = 6;
                    SysPar.pusch{idx}.TBSize = 7172;
                    SysPar.pusch{idx}.newDataIndicator = 0;
                case 341
                    SysPar.pusch{idx}.targetCodeRate = 9480;
                    SysPar.pusch{idx}.qamModOrder = 6;
                    SysPar.pusch{idx}.TBSize = 3329;
                    SysPar.pusch{idx}.newDataIndicator = 0;
            end
            SysPar.Chan{idx} = Chan0;
            SysPar = convert_CSI2_prms_from_FAPI_02_to_04(SysPar);
        end

        testAlloc.pusch = length(cfg);

        % negative test case - duplicate RNTI numbers across UEs
        if ismember(caseNum, 607)
            SysPar.pusch{3}.RNTI = SysPar.pusch{1}.RNTI;
            SysPar.pusch{5}.RNTI = SysPar.pusch{1}.RNTI;
        end
   
        % read beam id from TV     
        for idxUe = 1:length(SysPar.pusch)
            digBFInterfaces = SysPar.carrier.N_FhPort_UL;
            SysPar.pusch{idxUe}.digBFInterfaces = digBFInterfaces;
            SysPar.pusch{idxUe}.beamIdx = [1:digBFInterfaces];
        end

        if ismember(caseNum, [534:535, MIMO_64TR_TC]) % 64TR
            beamIdx_dynamic_offset = beamIdx_pusch_dynamic(1);
            beamIdx_static_offset = beamIdx_pusch_static(1);
            for idxUe = 1:length(SysPar.pusch)
                % digBFInterfaces = SysPar.pusch{idxUe}.nrOfLayers;
                if (ismember(caseNum, [590, 740:799, 21203:21415]) && idxUe < length(SysPar.pusch)) || (ismember(caseNum, [534:535,800:819,21520:21925]))
                    SysPar.pusch{idxUe}.digBFInterfaces = 0; % RTW
                    SysPar.pusch{idxUe}.beamIdx = beamIdx_dynamic_offset + [1:digBFInterfaces];
                    beamIdx_dynamic_offset = beamIdx_dynamic_offset + digBFInterfaces;              
                elseif (ismember(caseNum, [21423:21475]))
                    SysPar.pusch{idxUe}.digBFInterfaces = 0;
                    SysPar.pusch{idxUe}.seed = 0;
                    SysPar.SimCtrl.enable_dynamic_BF = 1;
                    SysPar.SimCtrl.enable_static_dynamic_beamforming = 1; % 64TR
                    SysPar.carrier.Nant_gNB_srs = 64;
                    SysPar.carrier.N_FhPort_DL = CFG_PUSCH{idxCfg, 21};
                    SysPar.carrier.N_FhPort_UL = SysPar.carrier.Nant_gNB;
                    SysPar.pusch{idxUe}.prgSize = 2;
                else
                    SysPar.pusch{idxUe}.digBFInterfaces = digBFInterfaces; % non-RTW
                    SysPar.pusch{idxUe}.beamIdx = beamIdx_static_offset + [1:digBFInterfaces];
                    beamIdx_static_offset = beamIdx_static_offset + digBFInterfaces;                
                end
            end
        end 

        if puschDynamicBfMap.isKey(caseNum) && puschDynamicBfMap(caseNum) == 1
            beamIdx_dynamic_offset = beamIdx_pusch_dynamic(1);
            for idxUe = 1:length(SysPar.pusch)
                SysPar.pusch{idxUe}.digBFInterfaces = 0; % RTW
                SysPar.pusch{idxUe}.beamIdx = beamIdx_dynamic_offset + [1:digBFInterfaces];
                beamIdx_dynamic_offset = beamIdx_dynamic_offset + digBFInterfaces;               
            end
        end 

        % --------- START unpack SRS config
        totNumUe = 0; % using totNumUe in case multiple SRS config are used in one slot
        % currenly only one SRS config per TV
        cfg = CFG{idxSet, 7};
        for idx = 1:length(cfg)
            idxCfg = find(cellfun(@(x) isequal(x,cfg{idx}), CFG_SRS(:,1)));
            nUe = CFG_SRS{idxCfg, 21};
            for idxUe = (1:nUe) + totNumUe
                % unpack srs config from CFG_SRS
                % TC#  rnti  Nap nSym  Nrep sym0 cfgIdx seqId bwIdx cmbSz cmbOffset cs  fPos fShift frqH grpH resType Tsrs  Toffset idxSlot N_UE
                SysPar.srs{idxUe} = SysPar.srs{1};
                SysPar.srs{idxUe}.RNTI = CFG_SRS{idxCfg, 2};
                SysPar.srs{idxUe}.numAntPorts = CFG_SRS{idxCfg, 3};
                SysPar.srs{idxUe}.numSymbols = CFG_SRS{idxCfg, 4};
                SysPar.srs{idxUe}.numRepetitions = CFG_SRS{idxCfg, 5};
                SysPar.srs{idxUe}.timeStartPosition = CFG_SRS{idxCfg, 6};
                SysPar.srs{idxUe}.configIndex = CFG_SRS{idxCfg, 7};
                SysPar.srs{idxUe}.sequenceId = CFG_SRS{idxCfg, 8};
                SysPar.srs{idxUe}.bandwidthIndex = CFG_SRS{idxCfg, 9};
                SysPar.srs{idxUe}.combSize = CFG_SRS{idxCfg, 10};
                SysPar.srs{idxUe}.combOffset = CFG_SRS{idxCfg, 11};
                SysPar.srs{idxUe}.cyclicShift = CFG_SRS{idxCfg, 12};
                SysPar.srs{idxUe}.frequencyPosition = CFG_SRS{idxCfg, 13};
                SysPar.srs{idxUe}.frequencyShift = CFG_SRS{idxCfg, 14};
                SysPar.srs{idxUe}.frequencyHopping = CFG_SRS{idxCfg, 15};
                SysPar.srs{idxUe}.groupOrSequenceHopping = CFG_SRS{idxCfg, 16};
                SysPar.srs{idxUe}.resourceType = CFG_SRS{idxCfg, 17};
                SysPar.srs{idxUe}.Tsrs = CFG_SRS{idxCfg, 18};
                SysPar.srs{idxUe}.Toffset = CFG_SRS{idxCfg, 19};
    
                % unpack srs config from CFG_SRS
                SysPar.srs{idxUe}.usage = USAGE_CODEBOOK;
                digBFInterfaces                   = SysPar.carrier.Nant_gNB;
                SysPar.srs{idxUe}.idxUE           = idxUe-1;
                SysPar.srs{idxUe}.digBFInterfaces = digBFInterfaces;
                SysPar.srs{idxUe}.beamIdx         = [1:digBFInterfaces];
                if(SysPar.srs{idxUe}.RNTI == SysPar.srs{1}.RNTI)
                    SysPar.srs{idxUe}.RNTI = SysPar.srs{1}.RNTI + idxUe - 1;
                end
                if ismember(idxCfg,[3 4]) % 4 UEs 
                    % combSize = 4, combOffset = [0 1 2 3]
                    SysPar.srs{idxUe}.combOffset = idxUe - 1 - totNumUe;
                elseif(ismember(idxCfg,[5 14 15])) % 8 UEs              
                    % UE 1~4 on OFDM symbol 12, UE 5~8 on OFDM symbol 13;
                    % combSize = 4, combOffset = [0 1 2 3 0 1 2 3]
                    SysPar.srs{idxUe}.timeStartPosition = floor((idxUe - 1 - totNumUe)/4) + 12;
                    SysPar.srs{idxUe}.combOffset = mod(idxUe - 1 - totNumUe, SysPar.srs{idxUe}.combSize);
                end

                % diff prgSize for SRS
                if ismember(idxCfg, [38])
                    SysPar.srs{idxUe}.prgSize = 4;
                end

                % Assign SRS by multiplexing with combs, then codes, and
                % symbols, but only use codes if not enough symbols/combs
                if SysPar.SimCtrl.enable_static_dynamic_beamforming  % 64TR
                    start_sym = SysPar.srs{1}.timeStartPosition;
                    comb_size = SysPar.srs{1}.combSize;
                    n_srs_codeshift = ceil(nUe / ((14-start_sym) * comb_size));
                    srs_per_symbol = comb_size * n_srs_codeshift;
                    SysPar.srs{idxUe}.combOffset = mod(idxUe-1, comb_size);
                    % TODO: add support for heterogeneous antenna ports configuration
                    unique_codesets_per_comb = 12 / (n_srs_codeshift * SysPar.srs{idxUe}.numAntPorts);
                    if (unique_codesets_per_comb < 1)
                        error('SRS cyclic shift is not enough for the UE / ports allocations in the SRS config');
                    elseif (unique_codesets_per_comb == 1)
                        warning('all 12 SRS cyclic shifts will be used, noise/SINR estimation will be inaccurate');
                    end
                    SysPar.srs{idxUe}.cyclicShift = mod(floor((idxUe-1)/comb_size),n_srs_codeshift) * floor(unique_codesets_per_comb);
                    SysPar.srs{idxUe}.timeStartPosition = floor((idxUe - 1)/srs_per_symbol) + start_sym;
                end
                SysPar.srs{idxUe}.srsChestBufferIndex = SysPar.srs{idxUe}.RNTI - 1;

                if ismember(caseNum, MIMO_64TR_TC)
                    SysPar.srs{idxUe}.usage = USAGE_CODEBOOK + USAGE_BEAM_MGMT;
                end

            end
            % TODO This should be removed when SRS buffer is separated from Xtf
            if SysPar.SimCtrl.enable_static_dynamic_beamforming  % 64TR
                SysPar.SimCtrl.CellConfigPorts = SysPar.carrier.Nant_gNB;
                SysPar.carrier.Nant_gNB = SysPar.carrier.Nant_gNB_srs;
            end
            totNumUe = totNumUe + nUe; % update totNumUe
        end

        if srsRntiStartMap.isKey(int32(caseNum)) && srsRntiStartMap(int32(caseNum)) > 0
            for idxUe = 1:length(SysPar.srs)
                SysPar.srs{idxUe}.RNTI = srsRntiStartMap(int32(caseNum)) + idxUe - 1;
            end
        end

        testAlloc.srs = totNumUe; 
        % ---------  END unpack SRS config

        % update testAlloc from all channels
        SysPar.testAlloc = testAlloc;

        % update N_UE from all channels, using max
        SysPar.SimCtrl.N_UE = max([testAlloc.pusch, testAlloc.pucch, testAlloc.prach testAlloc.srs]);
        
        % using P2P channels, so that RU can use only 2 antennas while having good decoding performance
        if ismember(caseNum, [1288:1631, 1744:1791,1840:1903,2304:2351,2352:2415,...
                2640:2735, 2736:2863])
           for idx = 1 : SysPar.SimCtrl.N_UE
               SysPar.Chan{idx}.type = 'P2P';
           end
        end
        
        % set DTXThreshold for pucch
        for idxUe = 1:testAlloc.pucch
            if  strcmp(SysPar.Chan{idxUe}.type, 'P2P')
                SysPar.pucch{idxUe}.DTXthreshold = SysPar.carrier.Nant_UE/SysPar.carrier.Nant_gNB;
            else
                SysPar.pucch{idxUe}.DTXthreshold = 1;
            end
        end
        
        % overwrite default BFPforCuphy from the default value 14 to 9 for all MIMO 64TR TCs
        if (ismember(caseNum, MIMO_64TR_TC))
            SysPar.SimCtrl.BFPforCuphy = 9;
        end
        
        % update Nre_max which is used to calculate beta value in BFP
        SysPar.SimCtrl.oranComp.Nre_max = SysPar.carrier.N_grid_size_mu*12;

        % save SysPar into Cfg_<TC#>.yaml config file
        if SysPar.SimCtrl.genTV.genYamlCfg
            if caseNum < 1000
                fileName = sprintf('Cfg_%04d.yaml', caseNum);
            else
                fileName = sprintf('Cfg_ULMIX_%04d.yaml', caseNum);
            end
            WriteYaml(fileName, SysPar);
        end        
        
        SysPar = updateAlgFlag(SysPar);
        
        
        % HARQ TC
        if ismember(caseNum, [550:589])
            SysPar.SimCtrl.N_UE = 24;
            SysPar.SimCtrl.genTV.forceSlotIdxFlag = 1;
            SysPar.SimCtrl.timeDomainSim = 1;
            for idxUE = 1:24
                SysPar.Chan{idxUE} = cfgChan;
                SysPar.Chan{idxUE}.SNR = 40;
            end
            if ismember(caseNum, [555, 575])
                T_sample = 8.1380e-09;
                delay_vec = [0 0 1 1 3 3]*32*T_sample;
                for idxUE = 1:6
                    SysPar.Chan{idxUE}.delay = delay_vec(idxUE);
                end
            end
            SysPar.Chan{4}.CFO = 450;
            
            if strcmp(caseSet, 'harq') % enable HARQ
                if ismember(caseNum, [554, 565, 574, 585])
                    for idxUE = 1:6
                        SysPar.Chan{idxUE}.SNR = 20;
                    end
                end
                if caseNum ==  554
                    chan_dB = 15;
                    SysPar.Chan{1}.gain = 10^((chan_dB-20)/20);
                    for idxUE = 2:6
                        chan_dB = 10;
                        SysPar.Chan{idxUE}.gain = 10^((chan_dB-20)/20);
                    end
                end
                if caseNum ==  565
                    chan_dB = 15;
                    SysPar.Chan{1}.gain = 10^((chan_dB-20)/20);
                    for idxUE = 2:6
                        chan_dB = 5;
                        SysPar.Chan{idxUE}.gain = 10^((chan_dB-20)/20);
                    end
                end
                if caseNum ==  574
                    chan_dB = 25;
                    SysPar.Chan{1}.gain = 10^((chan_dB-20)/20);
                    for idxUE = 2:6
                        chan_dB = 5;
                        SysPar.Chan{idxUE}.gain = 10^((chan_dB-20)/20);
                    end
                end
                if caseNum ==  585
                    chan_dB = 25;
                    SysPar.Chan{1}.gain = 10^((chan_dB-20)/20);
                    for idxUE = 2:6
                        chan_dB = 20;
                        SysPar.Chan{idxUE}.gain = 10^((chan_dB-20)/20);
                    end
                end
                %             SysPar.SimCtrl.printCheckDet = 1;
                SysPar.SimCtrl.N_frame = 2;
                SysPar.SimCtrl.N_slot_run = 0;
                fileName = sprintf('cfg_HARQ_ULMIX_s%d.yaml', caseNum-550);
                WriteYaml(fileName, SysPar);
                continue;
                
            else % disable HARQ
                
                nPusch = SysPar.testAlloc.pusch;
                for idxPusch = 1:nPusch
                    mcsIndex = SysPar.pusch{idxPusch}.mcsIndex;
                    if ismember(mcsIndex, [31])
                        SysPar.pusch{idxPusch}.mcsIndex = 27;
                    elseif ismember(mcsIndex, [30])
                        SysPar.pusch{idxPusch}.mcsIndex = 19;
                    elseif mcsIndex == 29
                        SysPar.pusch{idxPusch}.mcsIndex = 10;
                    elseif mcsIndex == 28
                        SysPar.pusch{idxPusch}.mcsIndex = 1;
                    end
                    SysPar.pusch{idxPusch}.rvIndex = 0;
                    SysPar.pusch{idxPusch}.newDataIndicator = 1;
                    SysPar.pusch{idxPusch}.harqProcessID = 0;
                end
                SysPar.SimCtrl.N_frame = 1;
                SysPar.SimCtrl.N_slot_run = 1;
            end
        elseif ismember(caseNum, [2864 : 2895])
            for idx = 1 : SysPar.SimCtrl.N_UE
               SysPar.Chan{idx}.SNR = 6 + 3 * (idx - 1);
           end
        end
        
        % enable UL Rx BF for selected MIMO 64TR TCs
        % column G patterns: nrSim 90624 and perf pattern 69, 69a, 69b, 69c, 69d
        % column H patterns: nrSim 90626 and perf pattern 71
        % column G patterns: nrSim 90627 and perf pattern 73, 40 MHz
        % column G patterns: nrSim 90629 and perf pattern 75, 100 MHz, 64 UEs per TTI
        % column I patterns: nrSim 90638 and perf pattern 77, 100 MHz
        % Ph4 column B patterns: nrSim 90640 and perf pattern 79, 100 MHz
        % 25-3 column B patterns: nrSim 90644~90647 and perf pattern 81a, 81b, 81c, 81d, 100 MHz
        % 25-3 column D pattern: nrSim 90662, 100 MHz
        % 25-3 column E pattern: nrSim 90648/90657, 100 MHz
        % 25-3 column G patterns: nrSim 90649~90652 and perf pattern 83a, 83b, 83c, 83d, 100 MHz
        % Ph4 column B patterns nrSim 90653, srsPrgSize = 4 and bfwPrgSize = 16, 100 MHz
        % Ph4 column B patterns nrSim 90654, srsPrgSize = 2 and bfwPrgSize = 16, 100 MHz
        % 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 100 MHz heavy
        % 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 100 MHz light
        if (ismember(caseNum, [21523:29999]) || ...
            enableUlRxBfMap.isKey(caseNum) && enableUlRxBfMap(caseNum) == 1)
            SysPar.SimCtrl.enableUlRxBf = 1;
            SysPar.SimCtrl.CellConfigPorts = SysPar.carrier.Nant_gNB;
            SysPar.carrier.Nant_gNB = SysPar.carrier.Nant_gNB_srs;
        end

        if ismember(caseNum, [605, 606, 607, 608, 610])
            SysPar.SimCtrl.negTV.enable = 4;
            for idx = 1 : SysPar.SimCtrl.N_UE
                SysPar.Chan{idx}.SNR = 10;
            end
            
            if caseNum == 605
                SysPar.SimCtrl.negTV.enable = 6; % need to ignore ru-emulator result checking in cicd scripts
            end
        end

        [SysPar, UE, gNB] = nrSimulator(SysPar);        
        SimCtrl = SysPar.SimCtrl; 
        
        printFlag = 0;
        errFlag = checkDetError(testAlloc, SimCtrl, printFlag);
        detErr = detErr + errFlag;
        if errFlag
            fprintf('DetError caseNum: %2d \n', CFG{idxSet, 1});
        end
        
        fprintf('%2d    %2d     %2d    %2d     %2d     %2d     %2d',...
            CFG{idxSet, 1}, CFG{idxSet, 2}, CFG{idxSet, 3}, length(CFG{idxSet, 4}), ...
            length(CFG{idxSet, 5}), length(CFG{idxSet, 6}), length(CFG{idxSet, 7}));

        % print negative TV params
        if (SimCtrl.negTV.enable)
            fprintf('  Negative TV enabled');
            if (SimCtrl.negTV.enable == 1)&&(isfield(SimCtrl.negTV, 'pduFieldName')) % negTV enabled, extract the invalid params from SimCtrl
                pduFieldName = SimCtrl.negTV.pduFieldName;
                pduFieldValue = SimCtrl.negTV.pduFieldValue;
                fprintf('  Invalid params: ');
                for pduFieldIdx = 1:length(pduFieldName)
                    fprintf(' %s:%4d\t', pduFieldName{pduFieldIdx}, pduFieldValue{pduFieldIdx});
                end
            end
        end

        % print a line break
        fprintf('\n');

        if SysPar.SimCtrl.plotFigure.tfGrid
            figure; mesh(abs(gNB.Phy.rx.Xtf(:,:,1))); view(2); pause(1); 
        end
    end
    % explicitly clear SysPar, UE, gNB to release memory
    SysPar = []; UE = []; gNB = [];
end

fprintf('--------------------------------------------\n');
fprintf('Total cuPHY TV generated = %d, total FAPI TV generated = %d, det-FAIL = %d, \n\n', nCuphyTV, nFapiTV, detErr);
toc; 
fprintf('\n');

end

function largestTvNum = getLargestTvNum(CFG, threshold)
    allTvNums = cell2mat(CFG(:, 1));
    largestTvNum = [max(allTvNums(allTvNums < threshold)) max(allTvNums(allTvNums >= threshold))];
    fprintf('The largest TV number in the current ULMIX CFG < %d is: %d \n', threshold, largestTvNum(1));
    fprintf('The largest TV number in the current ULMIX CFG >= %d is: %d \n', threshold, largestTvNum(2));
end
