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

function [nComp, errCnt, nCuphyTV, nFapiTV, detErr] = testCompGenTV_dlmix(caseSet, compTvMode, subSetMod, relNum, VARY_PRB_NUM)

tic;
if nargin == 0
    caseSet = 'full';
    compTvMode = 'both';
    subSetMod = [0, 1];
    relNum = 10000;
    VARY_PRB_NUM = -1;
elseif nargin == 1
    compTvMode = 'both';
    subSetMod = [0, 1];
    relNum = 10000;
    VARY_PRB_NUM = -1;
elseif nargin == 2
    subSetMod = [0, 1];
    relNum = 10000;
    VARY_PRB_NUM = -1;
elseif nargin == 3
    relNum = 10000;
    VARY_PRB_NUM = -1;
elseif nargin == 4
    VARY_PRB_NUM = -1;  % for perf sweeping PRB_NUM profile
end

switch compTvMode
    case 'both'
        genTV = 1;
        testCompliance = 0;
        fprintf('only support ''genTV'' \n');
    case {'genTV', 'genCfg'}
        genTV = 1;
        testCompliance = 0;
    case 'testCompliance'
        genTV = 0;
        testCompliance = 1;
        error('only support ''genTV'' \n');
    otherwise
        error('compTvMode is not supported...\n');
end

selected_TC = [100:599, 1000:13900, 20000:29999];
disabled_TC = [];
[~,TcIdx] = ismember(disabled_TC, selected_TC);
selected_TC(TcIdx) = [];

HARQ_TC = [150:189];

% performance pattern table
% columns: pattern#, start_tv, end_tv, n_cell, n_cell_to_gen
% n_cell_to_gen: min(total_cells, 21) for 4TR (pattern#<=65), min(total_cells, 7) for 64TR (pattern#>65)
%                set to 1 if not main perf patterns
% TV configs are for n_cell but only generate n_cell_to_gen cells for compact set
perf_pattern_table = [
% pattern     start_tv  end_tv  n_cell, n_cell_to_gen
    41,         2828,    2935,    12,     1;  % 41, 42
    43,         2936,    3079,    16,     1;  % 43, 45
    44,         3080,    3159,    16,     1;  % 44
    46,         3160,    3399,    12,     1;  % 46
    47,         3400,    3719,    16,     1;  % 47
    48,         3720,    4039,    16,    16;  % 48
    49,         4040,    4439,    20,    20;  % 49
    50,         4440,    4759,    16,    16;  % 50
    51,         4760,    5159,    20,    20;  % 51
    53,         5160,    5479,    16,     1;  % 53
    54,         5480,    5879,    20,     1;  % 54
    55,         6032,    6351,    16,     1;  % 55
    56,         6352,    6751,    20,     1;  % 56
    57,         5960,    6031,     8,     1;  % 57
    58,         6752,    7071,    16,     1;  % 58
    59,         7072,    7471,    20,    20;  % 59, 59b, 59d
    59.3,       9472,    10271,   40,    21;  % 59c, 59e, 62c
    60,         10452,   11251,   40,    21;  % 60, 60b, 60c, 60d, 63c
    61,         7872,    8271,    20,     1;  % 61
    65,         11432,   12231,   40,    21;  % 65, 65a, 65b, 65c, 65d
    66,         11252,   11341,   15,     7;  % 66, 66a, 66b, 66c, 66d
    67,         11342,   11431,   15,     7;  % 67, 67a, 67b, 67c, 67d
    69,         12232,   12441,   15,     7;  % 69, 69a, 69b, 69c, 69d, 69e, 71
    73,         12442,   12721,   20,    10;  % 73
    75,         12722,   12931,   15,     7;  % 75
    79,         13067,   13216,   15,     7;  % 79, 79a, 79b
    81,         13217,   13381,   15,     7;  % 81, 81a, 81b
    83,         13382,   13546,   15,     7;  % 83, 83a, 83b
    85,         13547,   13696,   15,     7;  % 85
    87,         13697,   13711,   15,     7;  % 87, also need 13067~13216 from pattern 79
    89,         13712,   13801,    9,     9;  % 89
    91,         13802,   13951,   15,     7;  % 91
    101,        7472,    7495,    24,    24;  % 101, 101a
    102,        7496,    7519,    24,    24;  % 102, 102a
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
compact_TV_non_perf_pattern = [103, 120:128, 130:135, 141, 142, 143, 144, 150:189, 190, 192:195, 204, 205, 228, 229, 296:461, ...
                                500, 501, 502, ...
                                20000:29999]; % mMIMO, multi-cell

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
    9481, 9696, 9905, 9906, 10047, ... % from 59c
    7484, ... % from 101
    7481, ... % from 101a
    7516, ... % from 102
];
compact_TV_FAPI_only = setdiff(compact_TV_FAPI_only, compact_TV_perf_pattern_cuBB_gpu);  % exclude cuBB GPU test bench TVs from FAPI-only set, will generate both cuPHY and FAPI TVs

full_TC = [100:599, 1000:13951, 20000:29999];

MIMO_64TR_TC = [190, 192:195, 281, 296:461, 11252:11431, 12232:13951, 20000:21309];

negative_TC = [500, 501, 502, 20584, 20585, 20594, 20595, 20831, 20950:20989, 21270:21309, 13547:13696, 13802:13951];

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
end

% TcFapiOnly is the set of TCs that we only generate FAPI TVs
if strcmp(caseSet, 'compact') || strcmp(caseSet, 'full')
    TcFapiOnly = compact_TV_FAPI_only;
else
    TcFapiOnly = [];
end
TcFapiOnly = [TcFapiOnly, negative_TC];

% 64TR beamIdx range
UL_OFFSET = 500; % Add max UL beam ID to ensure uniqueness between them
beamIdx_ssb_static    = UL_OFFSET + [  1:  50];
beamIdx_pdcch_static  = UL_OFFSET + [ 51: 100];
beamIdx_csirs_static  = UL_OFFSET + [101: 150];
beamIdx_pdsch_static  = UL_OFFSET + [151: 300];
beamIdx_pdsch_dynamic = UL_OFFSET + [301: 500];

TrsBeamIdxMap = containers.Map('KeyType','int32', 'ValueType','any');
CsiBeamIdxMap = containers.Map('KeyType','int32', 'ValueType','any');
CsiZpBeamIdxMap = containers.Map('KeyType','int32', 'ValueType','any');
SsbBeamIdxMap = containers.Map('KeyType','int32', 'ValueType','any');
CellIdxInPatternMap = containers.Map('KeyType','int32', 'ValueType','any');
pdschDynamicBfMap = containers.Map('KeyType','int32', 'ValueType','any');
enableIdentityPrecoderMap = containers.Map('KeyType','int32', 'ValueType','any');
enableCsiRs32PortsMap = containers.Map('KeyType','int32', 'ValueType','any');

% Import CSI-RS type constants from centralized source
csirsType = cfgCsirsType();

CFG = {...
  % TC#  slotIdx   cell    ssb      pdcch      pdsch     csirs
    100      0       1     {1}     {1, 2}     {1, 2}        {};
    101      1       1     {2}     {1, 2}     {1, 2}        {};
    102      2       1     {3}     {1, 2}     {1, 2}        {};
    103      3       1      {}     {1, 2}        {3}        {};
%     104      4       1      {}         {}         {}        {};
%     105      5       1      {}         {}         {}        {};
    106      6       1      {}     {1, 2}     {1, 2}    {1, 2};
    107      7       1      {}     {1, 2}     {1, 2}        {};
    108      8       1      {}     {1, 2}     {4, 5}        {};
    109      9       1      {}     {1, 2}     {4, 5}        {};
    110     10       1      {}     {1, 2}     {1, 2}        {};
    111     11       1      {}     {1, 2}     {1, 2}        {};
    112     12       1      {}     {1, 2}     {1, 2}        {};
    113     13       1      {}     {1, 2}        {3}        {};
%     114     14       1      {}         {}         {}        {};
%     115     15       1      {}         {}         {}        {};
    116     16       1      {}     {1, 2}     {1, 2}    {1, 2};
    117     17       1      {}     {1, 2}     {1, 2}        {};
    118     18       1      {}     {1, 2}     {4, 5}        {};
    119     19       1      {}     {1, 2}     {4, 5}        {};

    % TC#  slotIdx   cell    ssb      pdcch      pdsch     csirs
    % 141-1 chapter-4 test model
    120      0       1      {}        {9}     {6, 7}        {}; % NR-FR1-TM1.1
    121      0       1      {}        {9} {14,15,16}        {}; % NR-FR1-TM1.2
    122      0       1      {}        {9}        {8}        {}; % NR-FR1-TM2
    123      0       1      {}        {9}        {9}        {}; % NR-FR1-TM2a
    124      0       1      {}        {9}   {10, 11}        {}; % NR-FR1-TM3.1
    125      0       1      {}        {9}   {12, 13}        {}; % NR-FR1-TM3.1a
    126      0       1      {}        {9} {14,15,16}        {}; % NR-FR1-TM3.2
    127      0       1      {}        {9} {14,15,16}        {}; % NR-FR1-TM3.3
    128      0       1      {}        {9}   {17, 18}        {}; % NR-FR1-TM1.1, 2 layers    

    % 4T4R-SRS test cases
    130      0       1     {9}       {10}               {156}        {}; % slot w/ SSB
    131      0       1      {}       {10}               {157}        {}; % slot w/o SSB
    132      0       1     {9}       {10}          {248, 249}        {}; % slot w/ SSB
    133      0       1      {}       {10}          {250, 251}        {}; % slot w/o SSB
    % 32T32R-SRS
    134      0       8     {9}       {10}   num2cell(312:313)        {}; % slot w/ SSB 
    135      0       8      {}       {10}   num2cell(316:317)        {}; % slot w/o SSB
%     134      0       8     {9}       {10}   num2cell(312:315)        {}; % slot w/ SSB 
%     135      0       8      {}       {10}   num2cell(316:319)        {}; % slot w/o SSB
    
    % multi-cell pdsch + CSI-RS test (GT-6445)
    140      0       1     {}         { }     {1}           {};  % pdsch
    141      0       1     {}         { }     {}            {1}; % NZP CSIRS
    142      0       1     {}         { }     {1}           {1}; % pdsch + NZP CSIRS
    143      0       1     {}         { }     {1}           {7}; % pdsch + ZP CSIRS
    144      0       1     {}         { }     {}            {7}; % ZP CSIRS
    
% HARQ TC    
  % TC#  slotIdx   cell    ssb      pdcch              pdsch            csirs
    150      0       9     {16}   {25 39}              {320}              {};
    151      1       9     {17}   {26 40}              {321}              {};
    152      2       9     {18}   {28 41}  num2cell(322:324)              {};    
    153      3       9     {19}   {27 42}              {325}              {};    
    156      6       9       {}   {29 43}  num2cell(326:328) num2cell(67:70);    
    157      7       9       {}   {33 44}  num2cell(329:334) num2cell(67:68);  
    158      8       9       {}   {33 45}  num2cell(335:340) num2cell(67:70);  
    159      9       9       {}   {34 46}  num2cell(335:340) num2cell(67:68);  
    160     10       9       {}   {35 47}  num2cell(341:346) num2cell(67:70);  
    161     11       9       {}   {36 48}  num2cell(341:346) num2cell(67:68);     
    162     12       9       {}   {37 49}  num2cell(347:352)              {};        
    163     13       9       {}   {25 50}              {353}              {};        
    166     16       9       {}   {27 51}              {354} num2cell(67:70);  
    167     17       9       {}   {27 52}              {355} num2cell(67:68);  
    168     18       9       {}   {30 39}  num2cell(356:358)              {};  
    169     19       9       {}   {31 40}  num2cell(359:361)              {};  
    170     20       9       {}   {38 41}  num2cell(329:334)              {};  
    171     21       9       {}   {33 42}  num2cell(335:340)              {};  
    172     22       9       {}   {34 43}  num2cell(341:346)              {};  
    173     23       9       {}   {35 44}  num2cell(362:367)              {};  
    176     26       9       {}   {36 45}  num2cell(347:352) num2cell(67:70);  
    177     27       9       {}   {37 46}  num2cell(347:352) num2cell(67:68);  
    178     28       9       {}   {25 47}              {354} num2cell(67:70);  
    179     29       9       {}   {26 47}              {355} num2cell(67:68);  
    180     30       9       {}   {32 48}  num2cell(356:358) num2cell(67:70);  
    181     31       9       {}   {28 49}  num2cell(359:361) num2cell(67:68);  
    182     32       9       {}   {38 50}  num2cell(329:334)             {};  
    183     33       9       {}   {33 51}  num2cell(368:373)             {};  
    186     36       9       {}   {34 52}  num2cell(335:340)             {};        
    187     37       9       {}   {34 39}  num2cell(341:346)             {};       
    188     38       9       {}   {35 40}  num2cell(341:346)             {};  
    189     39       9       {}   {36 41}  num2cell(347:352)             {};  

% 64TR 
  % TC#  slotIdx   cell    ssb      pdcch              pdsch            csirs
    190      0      10     {20}   {53 54}  num2cell(408:413) num2cell(71:72);
% Simultaneous DL/UL in S-slot
    191      0       9       {}      {55}              {415}             {};
% 64TR CSIRS 32 ports
    192      0      10       {}   {57 58}  num2cell(408:413)           {106}; % 16 CSI-RS antennas, 32 ports
    193      0      12       {}   {57 58}  num2cell([408,409,412])     {106}; % 8 CSI-RS antennas, 32 ports
    194      0      13       {}   {57 58}  num2cell([408,412])         {106}; % 4 CSI-RS antennas, 32 ports
% Based on 191 but no modcompression
    195      0      13       {}      {55}              {415}             {};

    280      0       9       {}        {}              {354}              {2};
    281      0      10       {}        {}              {416}              {2};
%TCs for FH fragmentation
    296      0      10       {}        {} num2cell(492:495)            {77}; % Full BW MIMO w/ Full BW CSIRS on symbol 0
    297      0      10       {}        {} num2cell(492:495)            {76}; % Full BW MIMO w/ Part BW CSIRS on symbol 0
    298      0      10       {}        {} num2cell(492:495)            {77}; % Full BW MIMO w/ Full BW CSIRS on symbol 0
    299      0      10       {}        {} num2cell(492:495)            {78}; % Full BW MIMO w/ Full BW CSIRS on symbol 6

% 64TR integration SU-MIMO 1-UEG
  % TC#  slotIdx   cell    ssb      pdcch              pdsch            csirs
    300      0      10     {20}   {53 54}              {408}              {};
    301      1      10     {21}   {53 54}              {408}              {};
    302      2      10       {}   {53 54}              {408}              {};    
  % 303      3      10       {}   {53 54}                 {}              {};    
    306      6      10       {}   {53 54}              {408} num2cell(71:72);    
    307      7      10       {}   {53 54}              {408} num2cell(71:72);  
    308      8      10       {}   {53 54}              {408} num2cell(71:72);  
    309      9      10       {}   {53 54}              {408} num2cell(71:72);  
    310     10      10       {}   {53 54}              {408}              {};  
    311     11      10       {}   {53 54}              {408}              {};     
    312     12      10       {}   {53 54}              {408}              {};        
  % 313     13      10       {}   {53 54}                 {}              {};        
    316     16      10       {}   {53 54}              {408}              {};  
    317     17      10       {}   {53 54}              {408}              {};  
    318     18      10       {}   {53 54}              {408}              {};  
    319     19      10       {}   {53 54}              {408}              {};  

% 64TR integration SU-MIMO 2-UEG
  % TC#  slotIdx   cell    ssb      pdcch              pdsch            csirs
    320      0      10     {20}   {53 54}         {408, 412}              {};
    321      1      10     {21}   {53 54}         {408, 412}              {};
    322      2      10       {}   {53 54}         {408, 412}              {};    
  % 323      3      10       {}   {53 54}                 {}              {};    
    326      6      10       {}   {53 54}         {408, 412} num2cell(71:72);    
    327      7      10       {}   {53 54}         {408, 412} num2cell(71:72);  
    328      8      10       {}   {53 54}         {408, 412} num2cell(71:72);  
    329      9      10       {}   {53 54}         {408, 412} num2cell(71:72);  
    330     10      10       {}   {53 54}         {408, 412}              {};  
    331     11      10       {}   {53 54}         {408, 412}              {};     
    332     12      10       {}   {53 54}         {408, 412}              {};        
  % 333     13      10       {}   {53 54}                 {}              {};        
    336     16      10       {}   {53 54}         {408, 412}              {};  
    337     17      10       {}   {53 54}         {408, 412}              {};  
    338     18      10       {}   {53 54}         {408, 412}              {};  
    339     19      10       {}   {53 54}         {408, 412}              {};  

% 64TR integration SU/MU-MIMO 3-UEG
  % TC#  slotIdx   cell    ssb      pdcch              pdsch            csirs
    340      0      10     {20}   {57 58}  num2cell(408:413)              {};
    341      1      10     {21}   {57 58}  num2cell(408:413)              {};
    342      2      10       {}   {57 58}  num2cell(408:413)              {}; 
  % 343      3      10       {}   {57 58}  num2cell(408:413)              {};
    346      6      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
    347      7      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
    348      8      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
    349      9      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
    350     10      10       {}   {57 58}  num2cell(408:413)              {}; 
    351     11      10       {}   {57 58}  num2cell(408:413)              {}; 
    352     12      10       {}   {57 58}  num2cell(408:413)              {}; 
  % 353     13      10       {}   {57 58}  num2cell(408:413)              {};
    356     16      10       {}   {57 58}  num2cell(408:413)              {};  
    357     17      10       {}   {57 58}  num2cell(408:413)              {};  
    358     18      10       {}   {57 58}  num2cell(408:413)              {};  
    359     19      10       {}   {57 58}  num2cell(408:413)              {};      

% 64TR integration SU/MU-MIMO 1 dynamic UE + 1 static UE
  % TC#  slotIdx   cell    ssb      pdcch              pdsch            csirs
    360      0      10     {20}   {57 58}         {408, 412}              {};
    361      1      10     {21}   {57 58}         {408, 412}              {};
    362      2      10       {}   {57 58}         {408, 412}              {};
  % 363      3      10       {}   {57 58}         {408, 412}              {};
    366      6      10       {}   {57 58}         {408, 412} num2cell(71:72);
    367      7      10       {}   {57 58}         {408, 412} num2cell(71:72);
    368      8      10       {}   {57 58}         {408, 412} num2cell(71:72);
    369      9      10       {}   {57 58}         {408, 412} num2cell(71:72);
    370     10      10       {}   {57 58}         {408, 412}              {};
    371     11      10       {}   {57 58}         {408, 412}              {};
    372     12      10       {}   {57 58}         {408, 412}              {};
  % 373     13      10       {}   {57 58}         {408, 412}              {};
    376     16      10       {}   {57 58}         {408, 412}              {};
    377     17      10       {}   {57 58}         {408, 412}              {};
    378     18      10       {}   {57 58}         {408, 412}              {};
    379     19      10       {}   {57 58}         {408, 412}              {};

% 64TR integration SU/MU-MIMO 2 dynamic UE + 1 static UE
  % TC#  slotIdx   cell    ssb      pdcch              pdsch            csirs
    380      0      10     {20}   {57 58}    {408, 409, 412}              {};
    381      1      10     {21}   {57 58}    {408, 409, 412}              {};
    382      2      10       {}   {57 58}    {408, 409, 412}              {};
  % 383      3      10       {}   {57 58}    {408, 409, 412}              {};
    386      6      10       {}   {57 58}    {408, 409, 412}              {};
    387      7      10       {}   {57 58}    {408, 409, 412}              {};
    388      8      10       {}   {57 58}    {408, 409, 412}              {};
    389      9      10       {}   {57 58}    {408, 409, 412}              {};
    390     10      10       {}   {57 58}    {408, 409, 412}              {};
    391     11      10       {}   {57 58}    {408, 409, 412}              {};
    392     12      10       {}   {57 58}    {408, 409, 412}              {};
  % 393     13      10       {}   {57 58}    {408, 409, 412}              {};
    396     16      10       {}   {57 58}    {408, 409, 412}              {};
    397     17      10       {}   {57 58}    {408, 409, 412}              {};
    398     18      10       {}   {57 58}    {408, 409, 412}              {};
    399     19      10       {}   {57 58}    {408, 409, 412}              {};

% 64TR integration SU/MU-MIMO 8 UE x 1 layer dynamic
  % TC#  slotIdx   cell    ssb      pdcch            pdsch            csirs
    400      0      10     {20}   {57 58} num2cell(456:463)             {};
    401      1      10     {21}   {57 58} num2cell(456:463)             {};
    402      2      10       {}   {57 58} num2cell(456:463)             {};
  % 403      3      10       {}   {57 58} num2cell(456:463)             {};
    406      6      10       {}   {57 58} num2cell(456:463)             {};
    407      7      10       {}   {57 58} num2cell(456:463)             {};
    408      8      10       {}   {57 58} num2cell(456:463)             {};
    409      9      10       {}   {57 58} num2cell(456:463)             {};
    410     10      10       {}   {57 58} num2cell(456:463)             {};
    411     11      10       {}   {57 58} num2cell(456:463)             {};
    412     12      10       {}   {57 58} num2cell(456:463)             {};
  % 413     13      10       {}   {57 58} num2cell(456:463)             {};
    416     16      10       {}   {57 58} num2cell(456:463)             {};
    417     17      10       {}   {57 58} num2cell(456:463)             {};
    418     18      10       {}   {57 58} num2cell(456:463)             {};
    419     19      10       {}   {57 58} num2cell(456:463)             {};
    
% 64TR test case
  % TC#  slotIdx   cell    ssb      pdcch              pdsch               csirs
    420      0      10     {22}     {61}     num2cell([465:469])         {73 74 75};
    421      1      10       {}     {61 62}  num2cell([478:483])         {73 74 75};
    422      2      10       {}     {61 62}  num2cell([484:485,478:483])    {};
  % 343      3      10       {}     {61}     num2cell(464:469)              {};
    426      6      10       {}     {61}     num2cell([470:477,464:469])    {};
    427      7      10       {}     {61}     num2cell([470:477,464:469])    {};
    428      8      10       {}     {61}     num2cell([470:477,464:469])    {};
    429      9      10       {}     {61}     num2cell([470:477,464:469])    {};
    430     10      10       {}     {61}     num2cell([470:477,464:469])    {};
    431     11      10       {}     {61 62}  num2cell([484:485,478:483])    {};
    432     12      10       {}     {61 62}  num2cell([484:485,478:483])    {};
  % 353     13      10       {}     {61}     num2cell(464:469)              {};
    436     16      10       {}     {61}     num2cell([470:477,464:469])    {};
    437     17      10       {}     {61}     num2cell([470:477,464:469])    {};
    438     18      10       {}     {61}     num2cell([470:477,464:469])    {};
    439     19      10       {}     {61}     num2cell([470:477,464:469])    {};

% 64TR test case, BFPforCuphy = 9
  % TC#  slotIdx   cell    ssb      pdcch              pdsch               csirs
    440      0      10     {22}     {61}     num2cell([465:469])         {73 74 75};
    441      1      10       {}     {61 62}  num2cell([478:483])         {73 74 75};
    442      2      10       {}     {61 62}  num2cell([484:485,478:483])    {};
  % 343      3      10       {}     {61}     num2cell(464:469)  
    446      6      10       {}     {61}     num2cell([470:477,464:469])    {};
    447      7      10       {}     {61}     num2cell([470:477,464:469])    {};
    448      8      10       {}     {61}     num2cell([470:477,464:469])    {};
    449      9      10       {}     {61}     num2cell([470:477,464:469])    {};
    450     10      10       {}     {61}     num2cell([470:477,464:469])    {};
    451     11      10       {}     {61 62}  num2cell([484:485,478:483])    {};
    452     12      10       {}     {61 62}  num2cell([484:485,478:483])    {};
  % 353     13      10       {}     {61}     num2cell(464:469)              {};
    456     16      10       {}     {61}     num2cell([470:477,464:469])    {};
    457     17      10       {}     {61}     num2cell([470:477,464:469])    {};
    458     18      10       {}     {61}     num2cell([470:477,464:469])    {};
    459     19      10       {}     {61}     num2cell([470:477,464:469])    {};

 % 64TR test case, CSIRS 8 ports (row 6)
  % TC#  slotIdx   cell    ssb      pdcch              pdsch               csirs
    460      0      10     {22}     {61}     num2cell([465:469])         {79 80 81};
    461      1      10       {}     {61 62}  num2cell([478:483])         {79 80 81};

  % negative TVs for robustness
    500     6       1       {}   {30 39}      num2cell([21:36])         {}; % PDSCH overlapping with PDCCH
    501     7       1       {}   {}       num2cell([21:28,37:44])       {}; % PDSCH allocations overlapping
    502     0       4      {10}  {}                        {}           {}; % SSB outside of BWP

    % 64TR integration SU-MIMO 1-UEG 40 slots 
  % TC#  slotIdx   cell    ssb      pdcch              pdsch            csirs
  20000      0      10     {20}   {53 54}              {408}              {}; %cell-41
  20001      1      10     {21}   {53 54}              {408}              {};
  20002      2      10       {}   {53 54}              {408}              {};    
 %20003      3      10       {}   {53 54}                 {}              {};    
  20006      6      10       {}   {53 54}              {408} num2cell(71:72);    
  20007      7      10       {}   {53 54}              {408} num2cell(71:72);  
  20008      8      10       {}   {53 54}              {408} num2cell(71:72);  
  20009      9      10       {}   {53 54}              {408} num2cell(71:72);  
  20010     10      10       {}   {53 54}              {408}              {};  
  20011     11      10       {}   {53 54}              {408}              {};     
  20012     12      10       {}   {53 54}              {408}              {};        
 %20013     13      10       {}   {53 54}                 {}              {};        
  20016     16      10       {}   {53 54}              {408}              {};  
  20017     17      10       {}   {53 54}              {408}              {};  
  20018     18      10       {}   {53 54}              {408}              {};  
  20019     19      10       {}   {53 54}              {408}              {};
  20020     20      10     {20}   {53 54}              {408}              {};
  20021     21      10     {21}   {53 54}              {408}              {};
  20022     22      10       {}   {53 54}              {408}              {};    
 %20023     23      10       {}   {53 54}                 {}              {};    
  20026     26      10       {}   {53 54}              {408} num2cell(71:72);    
  20027     27      10       {}   {53 54}              {408} num2cell(71:72);  
  20028     28      10       {}   {53 54}              {408} num2cell(71:72);  
  20029     29      10       {}   {53 54}              {408} num2cell(71:72);  
  20030     30      10       {}   {53 54}              {408}              {};  
  20031     31      10       {}   {53 54}              {408}              {};     
  20032     32      10       {}   {53 54}              {408}              {};        
 %20033     33      10       {}   {53 54}                 {}              {};        
  20036     36      10       {}   {53 54}              {408}              {};  
  20037     37      10       {}   {53 54}              {408}              {};  
  20038     38      10       {}   {53 54}              {408}              {};  
  20039     39      10       {}   {53 54}              {408}              {};
  20040      0      10     {20}   {53 54}              {408}              {}; %cell-42
  20041      1      10     {21}   {53 54}              {408}              {};
  20042      2      10       {}   {53 54}              {408}              {};    
 %20043      3      10       {}   {53 54}                 {}              {};    
  20046      6      10       {}   {53 54}              {408} num2cell(71:72);    
  20047      7      10       {}   {53 54}              {408} num2cell(71:72);  
  20048      8      10       {}   {53 54}              {408} num2cell(71:72);  
  20049      9      10       {}   {53 54}              {408} num2cell(71:72);  
  20050     10      10       {}   {53 54}              {408}              {};  
  20051     11      10       {}   {53 54}              {408}              {};     
  20052     12      10       {}   {53 54}              {408}              {};        
 %20053     13      10       {}   {53 54}                 {}              {};        
  20056     16      10       {}   {53 54}              {408}              {};  
  20057     17      10       {}   {53 54}              {408}              {};  
  20058     18      10       {}   {53 54}              {408}              {};  
  20059     19      10       {}   {53 54}              {408}              {};
  20060     20      10     {20}   {53 54}              {408}              {};
  20061     21      10     {21}   {53 54}              {408}              {};
  20062     22      10       {}   {53 54}              {408}              {};    
 %20063     23      10       {}   {53 54}                 {}              {};    
  20066     26      10       {}   {53 54}              {408} num2cell(71:72);    
  20067     27      10       {}   {53 54}              {408} num2cell(71:72);  
  20068     28      10       {}   {53 54}              {408} num2cell(71:72);  
  20069     29      10       {}   {53 54}              {408} num2cell(71:72);  
  20070     30      10       {}   {53 54}              {408}              {};  
  20071     31      10       {}   {53 54}              {408}              {};     
  20072     32      10       {}   {53 54}              {408}              {};        
 %20073     33      10       {}   {53 54}                 {}              {};        
  20076     36      10       {}   {53 54}              {408}              {};  
  20077     37      10       {}   {53 54}              {408}              {};  
  20078     38      10       {}   {53 54}              {408}              {};  
  20079     39      10       {}   {53 54}              {408}              {};
  20080      0      10     {20}   {53 54}              {408}              {}; %cell-43
  20081      1      10     {21}   {53 54}              {408}              {};
  20082      2      10       {}   {53 54}              {408}              {};    
 %20083      3      10       {}   {53 54}                 {}              {};    
  20086      6      10       {}   {53 54}              {408} num2cell(71:72);    
  20087      7      10       {}   {53 54}              {408} num2cell(71:72);  
  20088      8      10       {}   {53 54}              {408} num2cell(71:72);  
  20089      9      10       {}   {53 54}              {408} num2cell(71:72);  
  20090     10      10       {}   {53 54}              {408}              {};  
  20091     11      10       {}   {53 54}              {408}              {};     
  20092     12      10       {}   {53 54}              {408}              {};        
 %20093     13      10       {}   {53 54}                 {}              {};        
  20096     16      10       {}   {53 54}              {408}              {};  
  20097     17      10       {}   {53 54}              {408}              {};  
  20098     18      10       {}   {53 54}              {408}              {};  
  20099     19      10       {}   {53 54}              {408}              {};
  20100     20      10     {20}   {53 54}              {408}              {};
  20101     21      10     {21}   {53 54}              {408}              {};
  20102     22      10       {}   {53 54}              {408}              {};    
 %20103     23      10       {}   {53 54}                 {}              {};    
  20106     26      10       {}   {53 54}              {408} num2cell(71:72);    
  20107     27      10       {}   {53 54}              {408} num2cell(71:72);  
  20108     28      10       {}   {53 54}              {408} num2cell(71:72);  
  20109     29      10       {}   {53 54}              {408} num2cell(71:72);  
  20110     30      10       {}   {53 54}              {408}              {};  
  20111     31      10       {}   {53 54}              {408}              {};     
  20112     32      10       {}   {53 54}              {408}              {};        
 %20113     33      10       {}   {53 54}                 {}              {};        
  20116     36      10       {}   {53 54}              {408}              {};  
  20117     37      10       {}   {53 54}              {408}              {};  
  20118     38      10       {}   {53 54}              {408}              {};  
  20119     39      10       {}   {53 54}              {408}              {};
  20120      0      10     {20}   {53 54}              {408}              {}; %cell-44
  20121      1      10     {21}   {53 54}              {408}              {};
  20122      2      10       {}   {53 54}              {408}              {};    
 %20123      3      10       {}   {53 54}                 {}              {};    
  20126      6      10       {}   {53 54}              {408} num2cell(71:72);    
  20127      7      10       {}   {53 54}              {408} num2cell(71:72);  
  20128      8      10       {}   {53 54}              {408} num2cell(71:72);  
  20129      9      10       {}   {53 54}              {408} num2cell(71:72);  
  20130     10      10       {}   {53 54}              {408}              {};  
  20131     11      10       {}   {53 54}              {408}              {};     
  20132     12      10       {}   {53 54}              {408}              {};        
 %20133     13      10       {}   {53 54}                 {}              {};        
  20136     16      10       {}   {53 54}              {408}              {};  
  20137     17      10       {}   {53 54}              {408}              {};  
  20138     18      10       {}   {53 54}              {408}              {};  
  20139     19      10       {}   {53 54}              {408}              {};
  20140     20      10     {20}   {53 54}              {408}              {};
  20141     21      10     {21}   {53 54}              {408}              {};
  20142     22      10       {}   {53 54}              {408}              {};    
 %20143     23      10       {}   {53 54}                 {}              {};    
  20146     26      10       {}   {53 54}              {408} num2cell(71:72);    
  20147     27      10       {}   {53 54}              {408} num2cell(71:72);  
  20148     28      10       {}   {53 54}              {408} num2cell(71:72);  
  20149     29      10       {}   {53 54}              {408} num2cell(71:72);  
  20150     30      10       {}   {53 54}              {408}              {};  
  20151     31      10       {}   {53 54}              {408}              {};     
  20152     32      10       {}   {53 54}              {408}              {};        
 %20153     33      10       {}   {53 54}                 {}              {};        
  20156     36      10       {}   {53 54}              {408}              {};  
  20157     37      10       {}   {53 54}              {408}              {};  
  20158     38      10       {}   {53 54}              {408}              {};  
  20159     39      10       {}   {53 54}              {408}              {};
  20160      0      10     {20}   {53 54}              {408}              {}; %cell-45
  20161      1      10     {21}   {53 54}              {408}              {};
  20162      2      10       {}   {53 54}              {408}              {};    
 %20163      3      10       {}   {53 54}                 {}              {};    
  20166      6      10       {}   {53 54}              {408} num2cell(71:72);    
  20167      7      10       {}   {53 54}              {408} num2cell(71:72);  
  20168      8      10       {}   {53 54}              {408} num2cell(71:72);  
  20169      9      10       {}   {53 54}              {408} num2cell(71:72);  
  20170     10      10       {}   {53 54}              {408}              {};  
  20171     11      10       {}   {53 54}              {408}              {};     
  20172     12      10       {}   {53 54}              {408}              {};        
 %20173     13      10       {}   {53 54}                 {}              {};        
  20176     16      10       {}   {53 54}              {408}              {};  
  20177     17      10       {}   {53 54}              {408}              {};  
  20178     18      10       {}   {53 54}              {408}              {};  
  20179     19      10       {}   {53 54}              {408}              {};
  20180     20      10     {20}   {53 54}              {408}              {};
  20181     21      10     {21}   {53 54}              {408}              {};
  20182     22      10       {}   {53 54}              {408}              {};    
 %20183     23      10       {}   {53 54}                 {}              {};    
  20186     26      10       {}   {53 54}              {408} num2cell(71:72);    
  20187     27      10       {}   {53 54}              {408} num2cell(71:72);  
  20188     28      10       {}   {53 54}              {408} num2cell(71:72);  
  20189     29      10       {}   {53 54}              {408} num2cell(71:72);  
  20190     30      10       {}   {53 54}              {408}              {};  
  20191     31      10       {}   {53 54}              {408}              {};     
  20192     32      10       {}   {53 54}              {408}              {};        
 %20193     33      10       {}   {53 54}                 {}              {};        
  20196     36      10       {}   {53 54}              {408}              {};  
  20197     37      10       {}   {53 54}              {408}              {};  
  20198     38      10       {}   {53 54}              {408}              {};  
  20199     39      10       {}   {53 54}              {408}              {};

  % 64TR integration SU/MU-MIMO 3-UEG 
  % TC#  slotIdx   cell    ssb      pdcch              pdsch            csirs
  20200      0      10     {20}   {57 58}  num2cell(408:413)              {}; %cell-41
  20201      1      10     {21}   {57 58}  num2cell(408:413)              {};
  20202      2      10       {}   {57 58}  num2cell(408:413)              {}; 
 %20203      3      10       {}   {57 58}  num2cell(408:413)              {};
  20206      6      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20207      7      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20208      8      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20209      9      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20210     10      10       {}   {57 58}  num2cell(408:413)              {}; 
  20211     11      10       {}   {57 58}  num2cell(408:413)              {}; 
  20212     12      10       {}   {57 58}  num2cell(408:413)              {}; 
 %20213     13      10       {}   {57 58}  num2cell(408:413)              {};
  20216     16      10       {}   {57 58}  num2cell(408:413)              {};  
  20217     17      10       {}   {57 58}  num2cell(408:413)              {};  
  20218     18      10       {}   {57 58}  num2cell(408:413)              {};  
  20219     19      10       {}   {57 58}  num2cell(408:413)              {};
  20220      0      10     {20}   {57 58}  num2cell(408:413)              {}; %cell-42
  20221      1      10     {21}   {57 58}  num2cell(408:413)              {};
  20222      2      10       {}   {57 58}  num2cell(408:413)              {}; 
 %20223      3      10       {}   {57 58}  num2cell(408:413)              {};
  20226      6      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20227      7      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20228      8      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20229      9      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20230     10      10       {}   {57 58}  num2cell(408:413)              {}; 
  20231     11      10       {}   {57 58}  num2cell(408:413)              {}; 
  20232     12      10       {}   {57 58}  num2cell(408:413)              {}; 
 %20233     13      10       {}   {57 58}  num2cell(408:413)              {};
  20236     16      10       {}   {57 58}  num2cell(408:413)              {};  
  20237     17      10       {}   {57 58}  num2cell(408:413)              {};  
  20238     18      10       {}   {57 58}  num2cell(408:413)              {};  
  20239     19      10       {}   {57 58}  num2cell(408:413)              {};
  20240      0      10     {20}   {57 58}  num2cell(408:413)              {}; %cell-43
  20241      1      10     {21}   {57 58}  num2cell(408:413)              {};
  20242      2      10       {}   {57 58}  num2cell(408:413)              {}; 
 %20243      3      10       {}   {57 58}  num2cell(408:413)              {};
  20246      6      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20247      7      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20248      8      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20249      9      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20250     10      10       {}   {57 58}  num2cell(408:413)              {}; 
  20251     11      10       {}   {57 58}  num2cell(408:413)              {}; 
  20252     12      10       {}   {57 58}  num2cell(408:413)              {}; 
 %20253     13      10       {}   {57 58}  num2cell(408:413)              {};
  20256     16      10       {}   {57 58}  num2cell(408:413)              {};  
  20257     17      10       {}   {57 58}  num2cell(408:413)              {};  
  20258     18      10       {}   {57 58}  num2cell(408:413)              {};  
  20259     19      10       {}   {57 58}  num2cell(408:413)              {};
  20260      0      10     {20}   {57 58}  num2cell(408:413)              {}; %cell-44
  20261      1      10     {21}   {57 58}  num2cell(408:413)              {};
  20262      2      10       {}   {57 58}  num2cell(408:413)              {}; 
 %20263      3      10       {}   {57 58}  num2cell(408:413)              {};
  20266      6      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20267      7      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20268      8      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20269      9      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20270     10      10       {}   {57 58}  num2cell(408:413)              {}; 
  20271     11      10       {}   {57 58}  num2cell(408:413)              {}; 
  20272     12      10       {}   {57 58}  num2cell(408:413)              {}; 
 %20273     13      10       {}   {57 58}  num2cell(408:413)              {};
  20276     16      10       {}   {57 58}  num2cell(408:413)              {};  
  20277     17      10       {}   {57 58}  num2cell(408:413)              {};  
  20278     18      10       {}   {57 58}  num2cell(408:413)              {};  
  20279     19      10       {}   {57 58}  num2cell(408:413)              {};
  20280      0      10     {20}   {57 58}  num2cell(408:413)              {}; %cell-45
  20281      1      10     {21}   {57 58}  num2cell(408:413)              {};
  20282      2      10       {}   {57 58}  num2cell(408:413)              {}; 
 %20283      3      10       {}   {57 58}  num2cell(408:413)              {};
  20286      6      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20287      7      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20288      8      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20289      9      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20290     10      10       {}   {57 58}  num2cell(408:413)              {}; 
  20291     11      10       {}   {57 58}  num2cell(408:413)              {}; 
  20292     12      10       {}   {57 58}  num2cell(408:413)              {}; 
 %20293     13      10       {}   {57 58}  num2cell(408:413)              {};
  20296     16      10       {}   {57 58}  num2cell(408:413)              {};  
  20297     17      10       {}   {57 58}  num2cell(408:413)              {};  
  20298     18      10       {}   {57 58}  num2cell(408:413)              {};  
  20299     19      10       {}   {57 58}  num2cell(408:413)              {};
  20300      0      10     {20}   {57 58}  num2cell(408:413)              {}; %cell-46
  20301      1      10     {21}   {57 58}  num2cell(408:413)              {};
  20302      2      10       {}   {57 58}  num2cell(408:413)              {}; 
 %20303      3      10       {}   {57 58}  num2cell(408:413)              {};
  20306      6      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20307      7      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20308      8      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20309      9      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20310     10      10       {}   {57 58}  num2cell(408:413)              {}; 
  20311     11      10       {}   {57 58}  num2cell(408:413)              {}; 
  20312     12      10       {}   {57 58}  num2cell(408:413)              {}; 
 %20313     13      10       {}   {57 58}  num2cell(408:413)              {};
  20316     16      10       {}   {57 58}  num2cell(408:413)              {};  
  20317     17      10       {}   {57 58}  num2cell(408:413)              {};  
  20318     18      10       {}   {57 58}  num2cell(408:413)              {};  
  20319     19      10       {}   {57 58}  num2cell(408:413)              {};
  20320      0      10     {20}   {57 58}  num2cell(408:413)              {}; %cell-47
  20321      1      10     {21}   {57 58}  num2cell(408:413)              {};
  20322      2      10       {}   {57 58}  num2cell(408:413)              {}; 
 %20323      3      10       {}   {57 58}  num2cell(408:413)              {};
  20326      6      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20327      7      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20328      8      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20329      9      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20330     10      10       {}   {57 58}  num2cell(408:413)              {}; 
  20331     11      10       {}   {57 58}  num2cell(408:413)              {}; 
  20332     12      10       {}   {57 58}  num2cell(408:413)              {}; 
 %20333     13      10       {}   {57 58}  num2cell(408:413)              {};
  20336     16      10       {}   {57 58}  num2cell(408:413)              {};  
  20337     17      10       {}   {57 58}  num2cell(408:413)              {};  
  20338     18      10       {}   {57 58}  num2cell(408:413)              {};  
  20339     19      10       {}   {57 58}  num2cell(408:413)              {};
  20340      0      10     {20}   {57 58}  num2cell(408:413)              {}; %cell-48
  20341      1      10     {21}   {57 58}  num2cell(408:413)              {};
  20342      2      10       {}   {57 58}  num2cell(408:413)              {}; 
 %20343      3      10       {}   {57 58}  num2cell(408:413)              {};
  20346      6      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20347      7      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20348      8      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20349      9      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20350     10      10       {}   {57 58}  num2cell(408:413)              {}; 
  20351     11      10       {}   {57 58}  num2cell(408:413)              {}; 
  20352     12      10       {}   {57 58}  num2cell(408:413)              {}; 
 %20353     13      10       {}   {57 58}  num2cell(408:413)              {};
  20356     16      10       {}   {57 58}  num2cell(408:413)              {};  
  20357     17      10       {}   {57 58}  num2cell(408:413)              {};  
  20358     18      10       {}   {57 58}  num2cell(408:413)              {};  
  20359     19      10       {}   {57 58}  num2cell(408:413)              {};
  20360      0      10     {20}   {57 58}  num2cell(408:413)              {}; %cell-49
  20361      1      10     {21}   {57 58}  num2cell(408:413)              {};
  20362      2      10       {}   {57 58}  num2cell(408:413)              {}; 
 %20363      3      10       {}   {57 58}  num2cell(408:413)              {};
  20366      6      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20367      7      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20368      8      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20369      9      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20370     10      10       {}   {57 58}  num2cell(408:413)              {}; 
  20371     11      10       {}   {57 58}  num2cell(408:413)              {}; 
  20372     12      10       {}   {57 58}  num2cell(408:413)              {}; 
 %20373     13      10       {}   {57 58}  num2cell(408:413)              {};
  20376     16      10       {}   {57 58}  num2cell(408:413)              {};  
  20377     17      10       {}   {57 58}  num2cell(408:413)              {};  
  20378     18      10       {}   {57 58}  num2cell(408:413)              {};  
  20379     19      10       {}   {57 58}  num2cell(408:413)              {};
  20380      0      10     {20}   {57 58}  num2cell(408:413)              {}; %cell-50
  20381      1      10     {21}   {57 58}  num2cell(408:413)              {};
  20382      2      10       {}   {57 58}  num2cell(408:413)              {}; 
 %20383      3      10       {}   {57 58}  num2cell(408:413)              {};
  20386      6      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20387      7      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20388      8      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20389      9      10       {}   {57 58}  num2cell(408:413) num2cell(71:72); 
  20390     10      10       {}   {57 58}  num2cell(408:413)              {}; 
  20391     11      10       {}   {57 58}  num2cell(408:413)              {}; 
  20392     12      10       {}   {57 58}  num2cell(408:413)              {}; 
 %20393     13      10       {}   {57 58}  num2cell(408:413)              {};
  20396     16      10       {}   {57 58}  num2cell(408:413)              {};  
  20397     17      10       {}   {57 58}  num2cell(408:413)              {};  
  20398     18      10       {}   {57 58}  num2cell(408:413)              {};  
  20399     19      10       {}   {57 58}  num2cell(408:413)              {};

  % 64TR MU-MIMO, 3 peak cells, 16 UEs, one layer per UE
  % TC#  slotIdx   cell    ssb      pdcch              pdsch            csirs
  20400      0      10       {}      {}  num2cell(528:543)              {}; %cell-41
  20401      1      10       {}      {}  num2cell(528:543)              {};
  20402      2      10       {}      {}  num2cell(528:543)              {}; 
  %20403      3      10       {}      {}  num2cell(528:543)              {};
  20406      6      10       {}      {}  num2cell(528:543)              {}; 
  20407      7      10       {}      {}  num2cell(528:543)              {}; 
  20408      8      10       {}      {}  num2cell(528:543)              {}; 
  20409      9      10       {}      {}  num2cell(528:543)              {}; 
  20410     10      10       {}      {}  num2cell(528:543)              {}; 
  20411     11      10       {}      {}  num2cell(528:543)              {}; 
  20412     12      10       {}      {}  num2cell(528:543)              {}; 
  %20413     13      10       {}      {}  num2cell(528:543)              {};
  20416     16      10       {}      {}  num2cell(528:543)              {};  
  20417     17      10       {}      {}  num2cell(528:543)              {};  
  20418     18      10       {}      {}  num2cell(528:543)              {};  
  20419     19      10       {}      {}  num2cell(528:543)              {};
  20420      0      10       {}      {}  num2cell(528:543)              {}; %cell-42
  20421      1      10       {}      {}  num2cell(528:543)              {};
  20422      2      10       {}      {}  num2cell(528:543)              {}; 
  %20423      3      10       {}      {}  num2cell(528:543)              {};
  20426      6      10       {}      {}  num2cell(528:543)              {}; 
  20427      7      10       {}      {}  num2cell(528:543)              {}; 
  20428      8      10       {}      {}  num2cell(528:543)              {}; 
  20429      9      10       {}      {}  num2cell(528:543)              {}; 
  20430     10      10       {}      {}  num2cell(528:543)              {}; 
  20431     11      10       {}      {}  num2cell(528:543)              {}; 
  20432     12      10       {}      {}  num2cell(528:543)              {}; 
  %20433     13      10       {}      {}  num2cell(528:543)              {};
  20436     16      10       {}      {}  num2cell(528:543)              {};  
  20437     17      10       {}      {}  num2cell(528:543)              {};  
  20438     18      10       {}      {}  num2cell(528:543)              {};  
  20439     19      10       {}      {}  num2cell(528:543)              {};
  20440      0      10       {}      {}  num2cell(528:543)              {}; %cell-43
  20441      1      10       {}      {}  num2cell(528:543)              {};
  20442      2      10       {}      {}  num2cell(528:543)              {}; 
  %20443      3      10       {}      {}  num2cell(528:543)              {};
  20446      6      10       {}      {}  num2cell(528:543)              {}; 
  20447      7      10       {}      {}  num2cell(528:543)              {}; 
  20448      8      10       {}      {}  num2cell(528:543)              {}; 
  20449      9      10       {}      {}  num2cell(528:543)              {}; 
  20450     10      10       {}      {}  num2cell(528:543)              {}; 
  20451     11      10       {}      {}  num2cell(528:543)              {}; 
  20452     12      10       {}      {}  num2cell(528:543)              {}; 
  %20453     13      10       {}      {}  num2cell(528:543)              {};
  20456     16      10       {}      {}  num2cell(528:543)              {};  
  20457     17      10       {}      {}  num2cell(528:543)              {};  
  20458     18      10       {}      {}  num2cell(528:543)              {};  
  20459     19      10       {}      {}  num2cell(528:543)              {};
  % 64TR MU-MIMO, 3 peak cells, 8 UEs, 2 layers per UE
  % TC#  slotIdx   cell    ssb      pdcch              pdsch            csirs
  20460      0      10       {}      {}  num2cell(544:551)              {}; %cell-41
  20461      1      10       {}      {}  num2cell(544:551)              {};
  20462      2      10       {}      {}  num2cell(544:551)              {}; 
  %20463      3      10       {}      {}  num2cell(544:551)              {};
  20466      6      10       {}      {}  num2cell(544:551)              {}; 
  20467      7      10       {}      {}  num2cell(544:551)              {}; 
  20468      8      10       {}      {}  num2cell(544:551)              {}; 
  20469      9      10       {}      {}  num2cell(544:551)              {}; 
  20470     10      10       {}      {}  num2cell(544:551)              {}; 
  20471     11      10       {}      {}  num2cell(544:551)              {}; 
  20472     12      10       {}      {}  num2cell(544:551)              {}; 
  %20473     13      10       {}      {}  num2cell(544:551)              {};
  20476     16      10       {}      {}  num2cell(544:551)              {};  
  20477     17      10       {}      {}  num2cell(544:551)              {};  
  20478     18      10       {}      {}  num2cell(544:551)              {};  
  20479     19      10       {}      {}  num2cell(544:551)              {};
  20480      0      10       {}      {}  num2cell(544:551)              {}; %cell-42
  20481      1      10       {}      {}  num2cell(544:551)              {};
  20482      2      10       {}      {}  num2cell(544:551)              {}; 
  %20483      3      10       {}      {}  num2cell(544:551)              {};
  20486      6      10       {}      {}  num2cell(544:551)              {}; 
  20487      7      10       {}      {}  num2cell(544:551)              {}; 
  20488      8      10       {}      {}  num2cell(544:551)              {}; 
  20489      9      10       {}      {}  num2cell(544:551)              {}; 
  20490     10      10       {}      {}  num2cell(544:551)              {}; 
  20491     11      10       {}      {}  num2cell(544:551)              {}; 
  20492     12      10       {}      {}  num2cell(544:551)              {}; 
  %20493     13      10       {}      {}  num2cell(544:551)              {};
  20496     16      10       {}      {}  num2cell(544:551)              {};  
  20497     17      10       {}      {}  num2cell(544:551)              {};  
  20498     18      10       {}      {}  num2cell(544:551)              {};  
  20499     19      10       {}      {}  num2cell(544:551)              {};
  20500      0      10       {}      {}  num2cell(544:551)              {}; %cell-43
  20501      1      10       {}      {}  num2cell(544:551)              {};
  20502      2      10       {}      {}  num2cell(544:551)              {}; 
  %20503      3      10       {}      {}  num2cell(544:551)              {};
  20506      6      10       {}      {}  num2cell(544:551)              {}; 
  20507      7      10       {}      {}  num2cell(544:551)              {}; 
  20508      8      10       {}      {}  num2cell(544:551)              {}; 
  20509      9      10       {}      {}  num2cell(544:551)              {}; 
  20510     10      10       {}      {}  num2cell(544:551)              {}; 
  20511     11      10       {}      {}  num2cell(544:551)              {}; 
  20512     12      10       {}      {}  num2cell(544:551)              {}; 
  %20513     13      10       {}      {}  num2cell(544:551)              {};
  20516     16      10       {}      {}  num2cell(544:551)              {};  
  20517     17      10       {}      {}  num2cell(544:551)              {};  
  20518     18      10       {}      {}  num2cell(544:551)              {};  
  20519     19      10       {}      {}  num2cell(544:551)              {};
% 64TR "worst case" column B TC
  20520      0      10       {22}   {61}    num2cell(583:590)           {};
  20521      1      10       {}   {61 62}   num2cell(591:598)           {};
% commented out TCs are using SU-MIMO for previous mixed SU-MIMO + MU-MIMO patterns
% nrSim pattern 90623 is now fully MU-MIMO in D and S slots
%   20522      2      10       {}   {61 62}   num2cell(558:563)           {};
%   20523      3      10       {}     {61}    num2cell(564:569)           {}; % SU-MIMO for S slot
%   20526      6      10       {}     {61}    num2cell(552:557)           {};
%   20527      7      10       {}     {61}    num2cell(552:557)           {};
%   20528      8      10       {}     {61}    num2cell(552:557)           {};
%   20529      9      10       {}     {61}    num2cell(552:557)           {};
%   20530     10      10       {}     {61}    num2cell(552:557)           {82 83 84};
%   20531     11      10       {}   {61 62}   num2cell(558:563)           {82 83 84};
%   20532     12      10       {}   {61 62}   num2cell(558:563)           {};
%   20533     13      10       {}     {61}    num2cell(564:569)           {}; % SU-MIMO for S slot
%   20536     16      10       {}     {61}    num2cell(570:577)           {};
%   20537     17      10       {}     {61}    num2cell(570:577)           {};
%   20538     18      10       {}     {61}    num2cell(570:577)           {};
%   20539     19      10       {}     {61}    num2cell(570:577)           {};
  20540      0      10       {}     {61}    num2cell(570:577)           {82 83 84};
  20541      1      10       {}   {61 62}   num2cell(591:598)           {82 83 84};
  20542      2      10       {}   {61 62}   num2cell(591:598)           {};
  20543      3      10       {}     {61}    num2cell(703:710)           {}; % MU-MIMO for S slot
  20546      6      10       {}     {61}    num2cell(570:577)           {};
  20547      7      10       {}     {61}    num2cell(570:577)           {};
  20548      8      10       {}     {61}    num2cell(570:577)           {};
  20549      9      10       {}     {61}    num2cell(570:577)           {};
  20550     10      10       {}     {61}    num2cell(570:577)           {};
  20551     11      10       {}   {61 62}   num2cell(591:598)           {};
  20552     12      10       {}   {61 62}   num2cell(591:598)           {};
  20553     13      10       {}     {61}    num2cell(703:710)           {}; % MU-MIMO for S slot
  20556     16      10       {}     {61}    num2cell(570:577)           {};
  20557     17      10       {}     {61}    num2cell(570:577)           {};
  20558     18      10       {}     {61}    num2cell(570:577)           {};
  20559     19      10       {}     {61}    num2cell(570:577)           {};
  %TC#     slotIdx   cell     ssb   pdcch       pdsch                csirs
% 64TR "worst case" column G TC
  20560      0      10       {23}   {75}    num2cell(719:734)           {};
  20561      1      10       {24}   {75 74} num2cell(735:750)           {};
  20562      2      10       {25}   {75 74} num2cell(735:750)           {};
  20563      3      10       {}     {75}    num2cell(751:766)           {};
  20566      6      10       {}     {75}    num2cell(623:638)           {};
  20567      7      10       {}     {75}    num2cell(623:638)           {};
  20568      8      10       {}     {75}    num2cell(623:638)           {};
  20569      9      10       {}     {75}    num2cell(623:638)           {};
  20570     10      10       {}     {75}    num2cell(623:638)           {};
  20571     11      10       {}   {75 74}   num2cell(639:654)           {};
  20572     12      10       {}   {75 74}   num2cell(639:654)           {};
  20573     13      10       {}     {75}    num2cell(751:766)           {};
  20576     16      10       {}     {75}    num2cell(623:638)           {};
  20577     17      10       {}     {75}    num2cell(623:638)           {};
  20578     18      10       {}     {75}    num2cell(623:638)           {};
  20579     19      10       {}     {75}    num2cell(623:638)           {};
  20580      0      10       {}     {75}    num2cell(623:638)           {82 83 95};
  20581      1      10       {}   {75 74}   num2cell(639:654)           {82 83 95};
  20582      2      10       {}   {75 74}   num2cell(639:654)           {};
  20583      3      10       {}     {75}    num2cell(751:766)           {};
  20586      6      10       {}     {75}    num2cell(623:638)           {};
  20587      7      10       {}     {75}    num2cell(623:638)           {};
  20588      8      10       {}     {75}    num2cell(623:638)           {};
  20589      9      10       {}     {75}    num2cell(623:638)           {};
  20590     10      10       {}     {75}    num2cell(623:638)           {82 83 95};
  20591     11      10       {}   {75 74}   num2cell(639:654)           {82 83 95};
  20592     12      10       {}   {75 74}   num2cell(639:654)           {};
  20593     13      10       {}     {75}    num2cell(751:766)           {};
  20596     16      10       {}     {75}    num2cell(623:638)           {82 83 95};
  20597     17      10       {}     {75}    num2cell(623:638)           {82 83 95};
  20598     18      10       {}     {75}    num2cell(623:638)           {82 83 95};
  20599     19      10       {}     {75}    num2cell(623:638)           {82 83 95};
  % negative case, invalid prgSize
  20584      6      10       {}     {75}    num2cell(623:638)           {};  % invalid prgSize = 273, 20566
  20585      7      10       {}     {75}    num2cell(623:638)           {};  % invalid prgSize = 1, 20567
  20594      8      10       {}     {75}    num2cell(623:638)           {};  % invalid prgSize = 4, 20568
  20595      9      10       {}     {75}    num2cell(623:638)           {};  % invalid prgSize = 8, 20569
% 64TR "worst case" column B TC w/ modcomp
  20600      0      10       {22}   {61}    num2cell(583:590)           {};
  20601      1      10       {}   {61 62}   num2cell(591:598)           {};
  %
  20620      0      10       {}     {61}    num2cell(570:577)           {82 83 84};
  20621      1      10       {}   {61 62}   num2cell(591:598)           {82 83 84};
  20622      2      10       {}   {61 62}   num2cell(591:598)           {};
  20623      3      10       {}     {61}    num2cell(703:710)           {}; % MU-MIMO for S slot
  20626      6      10       {}     {61}    num2cell(570:577)           {};
  20627      7      10       {}     {61}    num2cell(570:577)           {};
  20628      8      10       {}     {61}    num2cell(570:577)           {};
  20629      9      10       {}     {61}    num2cell(570:577)           {};
  20630     10      10       {}     {61}    num2cell(570:577)           {};
  20631     11      10       {}   {61 62}   num2cell(591:598)           {};
  20632     12      10       {}   {61 62}   num2cell(591:598)           {};
  20633     13      10       {}     {61}    num2cell(703:710)           {}; % MU-MIMO for S slot
  20636     16      10       {}     {61}    num2cell(570:577)           {};
  20637     17      10       {}     {61}    num2cell(570:577)           {};
  20638     18      10       {}     {61}    num2cell(570:577)           {};
  20639     19      10       {}     {61}    num2cell(570:577)           {};
% 64TR "worst case" column G TC, 40 MHz
  20640      0      11       {23}   {69}    num2cell(767:782)           {};
  20641      1      11       {24}   {69 68} num2cell(783:798)           {};
  20642      2      11       {25}   {69 68} num2cell(783:798)           {};
  20643      3      11       {}     {69}    num2cell(799:814)           {};
  20646      6      11       {}     {69}    num2cell(815:830)           {};
  20647      7      11       {}     {69}    num2cell(815:830)           {};
  20648      8      11       {}     {69}    num2cell(815:830)           {};
  20649      9      11       {}     {69}    num2cell(815:830)           {};
  20650     10      11       {}     {69}    num2cell(815:830)           {};
  20651     11      11       {}   {69 68}   num2cell(831:846)           {};
  20652     12      11       {}   {69 68}   num2cell(831:846)           {};
  20653     13      11       {}     {69}    num2cell(799:814)           {};
  20656     16      11       {}     {69}    num2cell(815:830)           {};
  20657     17      11       {}     {69}    num2cell(815:830)           {};
  20658     18      11       {}     {69}    num2cell(815:830)           {};
  20659     19      11       {}     {69}    num2cell(815:830)           {};
  20660      0      11       {}     {69}    num2cell(815:830)           {100 101 102};
  20661      1      11       {}   {69 68}   num2cell(831:846)           {100 101 102};
  20662      2      11       {}   {69 68}   num2cell(831:846)           {};
  20663      3      11       {}     {69}    num2cell(799:814)           {};
  20666      6      11       {}     {69}    num2cell(815:830)           {};
  20667      7      11       {}     {69}    num2cell(815:830)           {};
  20668      8      11       {}     {69}    num2cell(815:830)           {};
  20669      9      11       {}     {69}    num2cell(815:830)           {};
  20670     10      11       {}     {69}    num2cell(815:830)           {100 101 102};
  20671     11      11       {}   {69 68}   num2cell(831:846)           {100 101 102};
  20672     12      11       {}   {69 68}   num2cell(831:846)           {};
  20673     13      11       {}     {69}    num2cell(799:814)           {};
  20676     16      11       {}     {69}    num2cell(815:830)           {100 101 102};
  20677     17      11       {}     {69}    num2cell(815:830)           {100 101 102};
  20678     18      11       {}     {69}    num2cell(815:830)           {100 101 102};
  20679     19      11       {}     {69}    num2cell(815:830)           {100 101 102};
  % no SSB for CA
  20644      0      11       {}     {69}    num2cell(815:830)           {};  % 20640 w/o SSB, fill 20 PRBs with PDSCH
  20654      1      11       {}     {69 68} num2cell(831:846)           {};  % 20641 w/o SSB, fill 20 PRBs with PDSCH
  20655      2      11       {}     {69 68} num2cell(831:846)           {};  % 20642 w/o SSB, fill 20 PRBs with PDSCH
% 64TR 'worst case' column G, 64 UEs per TTI
  20690      0      10       {23}   {75}    num2cell(2001:2064)           {};
  20691      1      10       {24}   {75 74} num2cell(2065:2128)           {};
  20692      2      10       {25}   {75 74} num2cell(2065:2128)           {};
  20693      3      10       {}     {75}    num2cell(2129:2192)           {};
  20696      6      10       {}     {75}    num2cell(2193:2256)           {};
  20697      7      10       {}     {75}    num2cell(2193:2256)           {};
  20698      8      10       {}     {75}    num2cell(2193:2256)           {};
  20699      9      10       {}     {75}    num2cell(2193:2256)           {};
  20700     10      10       {}     {75}    num2cell(2193:2256)           {};
  20701     11      10       {}   {75 74}   num2cell(2257:2320)           {};
  20702     12      10       {}   {75 74}   num2cell(2257:2320)           {};
  20703     13      10       {}     {75}    num2cell(2193:2256)           {};
  20706     16      10       {}     {75}    num2cell(2193:2256)           {};
  20707     17      10       {}     {75}    num2cell(2193:2256)           {};
  20708     18      10       {}     {75}    num2cell(2193:2256)           {};
  20709     19      10       {}     {75}    num2cell(2193:2256)           {};
  20710      0      10       {}     {75}    num2cell(2193:2256)           {82 83 95};
  20711      1      10       {}   {75 74}   num2cell(2257:2320)           {82 83 95};
  20712      2      10       {}   {75 74}   num2cell(2257:2320)           {};
  20713      3      10       {}     {75}    num2cell(2193:2256)           {};
  20716      6      10       {}     {75}    num2cell(2193:2256)           {};
  20717      7      10       {}     {75}    num2cell(2193:2256)           {};
  20718      8      10       {}     {75}    num2cell(2193:2256)           {};
  20719      9      10       {}     {75}    num2cell(2193:2256)           {};
  20720     10      10       {}     {75}    num2cell(2193:2256)           {82 83 95};
  20721     11      10       {}   {75 74}   num2cell(2257:2320)           {82 83 95};
  20722     12      10       {}   {75 74}   num2cell(2257:2320)           {};
  20723     13      10       {}     {75}    num2cell(2193:2256)           {};
  20726     16      10       {}     {75}    num2cell(2193:2256)           {82 83 95};
  20727     17      10       {}     {75}    num2cell(2193:2256)           {82 83 95};
  20728     18      10       {}     {75}    num2cell(2193:2256)           {82 83 95};
  20729     19      10       {}     {75}    num2cell(2193:2256)           {82 83 95};
  % TC 34x w/ ZP-CSI-RS
  20736      6      10       {}   {57 58}   num2cell(408:413) num2cell([71, 109]); 
  20737      7      10       {}   {57 58}   num2cell(408:413) num2cell([71, 109]); 
  % 64TR "worst case" column G TC, 40 MHz, different cell ID = 42
  20740      0      11       {23}   {69}    num2cell(767:782)           {};  % 20640 with cell ID = 42
  20741      1      11       {24}   {69 68} num2cell(783:798)           {};  % 20641 with cell ID = 42
  20742      2      11       {25}   {69 68} num2cell(783:798)           {};  % 20642 with cell ID = 42
  % 64TR "worst case" column G TC, 40 MHz, different cell ID = 43
  20743      0      11       {23}   {69}    num2cell(767:782)           {};  % 20640 with cell ID = 43
  20744      1      11       {24}   {69 68} num2cell(783:798)           {};  % 20641 with cell ID = 43
  20745      2      11       {25}   {69 68} num2cell(783:798)           {};  % 20642 with cell ID = 43
  % 64TR 'worst case' column I
  20750      0      10       {22}   {72}    num2cell(2321:2336)           {};
  20751      1      10       {}   {72 71}   num2cell(2337:2352)           {};
  20752      2      10       {}   {72 71}   num2cell(2337:2352)           {};
  20753      3      10       {}     {72}    num2cell(2353:2368)           {};
  20756      6      10       {}     {72}    num2cell(2369:2384)           {};
  20757      7      10       {}     {72}    num2cell(2369:2384)           {};
  20758      8      10       {}     {72}    num2cell(2369:2384)           {};
  20759      9      10       {}     {72}    num2cell(2369:2384)           {};
  20760     10      10       {}   {72 70}  num2cell([2385:2400,847])      {};  % both 72 and 70 PDDCH is on symbol 0, 847 is PDSCH for CCH
  20761     11      10       {}   {72 71}   num2cell(2337:2352)           {};
  20762     12      10       {}   {72 71}   num2cell(2337:2352)           {};
  20763     13      10       {}     {72}    num2cell(2353:2368)           {};
  20766     16      10       {}     {72}    num2cell(2369:2384)           {};
  20767     17      10       {}     {72}    num2cell(2369:2384)           {};
  20768     18      10       {}     {72}    num2cell(2369:2384)           {};
  20769     19      10       {}     {72}    num2cell(2369:2384)           {};
  20770      0      10       {}     {72}    num2cell(2369:2384)           {82 83 84 111};
  20771      1      10       {}   {72 71}   num2cell(2337:2352)           {82 83};
  20772      2      10       {}   {72 71}   num2cell(2337:2352)           {};
  20773      3      10       {}     {72}    num2cell(2353:2368)           {};
  20776      6      10       {}     {72}    num2cell(2369:2384)           {};
  20777      7      10       {}     {72}    num2cell(2369:2384)           {};
  20778      8      10       {}     {72}    num2cell(2369:2384)           {};
  20779      9      10       {}     {72}    num2cell(2369:2384)           {};
  20780     10      10       {}     {72}    num2cell(2369:2384)           {};
  20781     11      10       {}   {72 71}   num2cell(2337:2352)           {};
  20782     12      10       {}   {72 71}   num2cell(2337:2352)           {};
  20783     13      10       {}     {72}    num2cell(2353:2368)           {};
  20786     16      10       {}     {72}    num2cell(2369:2384)           {};
  20787     17      10       {}     {72}    num2cell(2369:2384)           {};
  20788     18      10       {}     {72}    num2cell(2369:2384)           {};
  20789     19      10       {}     {72}    num2cell(2369:2384)           {};
  % additional CSI-RS cases
  20754      0      10       {}     {72}    num2cell(2369:2384)           {82 83};  % slot 60
  20755      1      10       {}   {72 71}   num2cell(2337:2352)           {82 83};  % slot 61
  % 64TR 'worst case' Ph4 column B
  20790      0      10       {22}   {72}    num2cell(2321:2336)           {};
  20791      1      10       {}   {72 71}   num2cell(2337:2352)           {};
  20792      2      10       {}   {72 71}   num2cell(2337:2352)           {};
  20793      3      10       {}     {72}    num2cell(2353:2368)           {};
  20796      6      10       {}     {72}    num2cell(2369:2384)           {};
  20797      7      10       {}     {72}    num2cell(2369:2384)           {};
  20798      8      10       {}     {72}    num2cell(2369:2384)           {};
  20799      9      10       {}     {72}    num2cell(2369:2384)           {};
  20800     10      10       {}   {72 70}  num2cell([2385:2400,847])      {};  % both 72 and 70 PDDCH is on symbol 0, 847 is PDSCH for CCH
  20801     11      10       {}   {72 71}   num2cell(2337:2352)           {};
  20802     12      10       {}   {72 71}   num2cell(2337:2352)           {};
  20803     13      10       {}     {72}    num2cell(2353:2368)           {};
  20806     16      10       {}     {72}    num2cell(2369:2384)           {};
  20807     17      10       {}     {72}    num2cell(2369:2384)           {};
  20808     18      10       {}     {72}    num2cell(2369:2384)           {};
  20809     19      10       {}     {72}    num2cell(2369:2384)           {};
  20810      0      10       {}     {72}    num2cell(2369:2384)           {82 83};
  20811      1      10       {}   {72 71}   num2cell(2337:2352)           {82 83};
  20812      2      10       {}   {72 71}   num2cell(2337:2352)           {};
  20813      3      10       {}     {}                {}                  {113 114};  % slot 23: CSI_RS + SRS
  20816      6      10       {}     {72}    num2cell(2369:2384)           {};
  20817      7      10       {}     {72}    num2cell(2369:2384)           {};
  20818      8      10       {}     {72}    num2cell(2369:2384)           {};
  20819      9      10       {}     {72}    num2cell(2369:2384)           {};
  20820     10      10       {}     {72}    num2cell(2369:2384)           {};
  20821     11      10       {}   {72 71}   num2cell(2337:2352)           {};
  20822     12      10       {}   {72 71}   num2cell(2337:2352)           {};
  20823     13      10       {}     {72}    num2cell(2353:2368)           {};
  20826     16      10       {}     {72}    num2cell(2369:2384)           {};
  20827     17      10       {}     {72}    num2cell(2369:2384)           {};
  20828     18      10       {}     {72}    num2cell(2369:2384)           {};
  20829     19      10       {}     {72}    num2cell(2369:2384)           {};
  % additional CSI-RS cases
  20794      0      10       {}     {72}    num2cell(2369:2384)           {82 83};  % slot 60
  20795      1      10       {}   {72 71}   num2cell(2337:2352)           {82 83};  % slot 61
  20804      3      10       {}     {72}    num2cell(2353:2368)           {};       % slot 63
  20805      3      10       {}     {}                {}                  {115 116};  % slot 23: CSI_RS + SRS

  % nvbug 5368243
  20830      1      10       {}     {}      num2cell(848:857)             {95};     % slot 1
  
  % 32DL
  20831      1      10       {}     {}      num2cell(871:886)             {95};     % slot 1

  %TC#     slotIdx   cell     ssb   pdcch       pdsch                csirs
  % heterogeneous UE group pattern w/ modcomp TCs
  20840      0      10       {23}   {76 77}  num2cell(858:867)           {117 118};
  20841      1      10       {24}   {76 77}  num2cell(858:867)           {};
  20842      2      10       {25}   {76 77}  num2cell(858:867)           {};
  %20843      3      10       {}     {72}    num2cell(858:867)           {};
  20846      6      10       {}     {76 77}  num2cell(858:867)           {};
  20847      7      10       {}     {76 77}  num2cell(858:867)           {};
  20848      8      10       {}     {76 77}  num2cell(858:867)           {};
  20849      9      10       {}     {76 77}  num2cell(858:867)           {};
  20850     10      10       {}     {76 77}  num2cell(858:867)           {};
  20851     11      10       {}     {76 77}  num2cell(858:867)           {};
  20852     12      10       {}     {76 77}  num2cell(858:867)           {};
  %20853     13      10       {}     {72}    num2cell(858:867)           {};
  20856     16      10       {}     {76 77}  num2cell(858:867)           {};
  20857     17      10       {}     {76 77}  num2cell(858:867)           {};
  20858     18      10       {}     {76 77}  num2cell(858:867)           {};
  20859     19      10       {}     {76 77}  num2cell(858:867)           {};
  20860      0      10       {23}   {76 77}  num2cell([858,868:870])     {117 118};
  20861      1      10       {}     {79}     num2cell([858:861])         {};
  % 64TR 25-3 column B
  20870      0      10       {22}   {72}    num2cell(2321:2336)           {};
  20871      1      10       {}   {72 71}   num2cell(2337:2352)           {};
  20872      2      10       {}   {72 71}   num2cell(2337:2352)           {};
  20873      3      10       {}     {72}    num2cell(2353:2368)           {};
  20876      6      10       {}     {72}    num2cell(2369:2384)           {};
  20877      7      10       {}     {72}    num2cell(2369:2384)           {};
  20878      8      10       {}     {72}    num2cell(2369:2384)           {};
  20879      9      10       {}     {72}    num2cell(2369:2384)           {};
  20880     10      10       {}   {72 70}  num2cell([2385:2400,847])      {};  % both 72 and 70 PDDCH is on symbol 0, 847 is PDSCH for CCH
  20881     11      10       {}   {72 71}   num2cell(2337:2352)           {};
  20882     12      10       {}   {72 71}   num2cell(2337:2352)           {};
  20883     13      10       {}     {72}    num2cell(2353:2368)           {};
  20886     16      10       {}     {72}    num2cell(2369:2384)           {};
  20887     17      10       {}     {72}    num2cell(2369:2384)           {};
  20888     18      10       {}     {72}    num2cell(2369:2384)           {};
  20889     19      10       {}     {72}    num2cell(2369:2384)           {};
  20890      0      10       {}     {72}    num2cell(2369:2384)           {82 83};
  20891      1      10       {}   {72 71}   num2cell(2337:2352)           {82 83};
  20892      2      10       {}   {72 71}   num2cell(2337:2352)           {};
  20893      3      10       {}     {}                {}                  {113 114};  % slot 23: 8-port CSI_RS + SRS, for nrSim 90644, 90646
  20894      3      10       {}     {}                {}                  {119 120 114};  % slot 23: 8-port + 32-port CSI_RS + SRS, for nrSim 90645, 90647
  20896      6      10       {}     {72}    num2cell(2369:2384)           {};
  20897      7      10       {}     {72}    num2cell(2369:2384)           {};
  20898      8      10       {}     {72}    num2cell(2369:2384)           {};
  20899      9      10       {}     {72}    num2cell(2369:2384)           {};
  20900     10      10       {}     {72}    num2cell(2369:2384)           {};
  20901     11      10       {}   {72 71}   num2cell(2337:2352)           {};
  20902     12      10       {}   {72 71}   num2cell(2337:2352)           {};
  20903     13      10       {}     {72}    num2cell(2353:2368)           {};
  20906     16      10       {}     {72}    num2cell(2369:2384)           {};
  20907     17      10       {}     {72}    num2cell(2369:2384)           {};
  20908     18      10       {}     {72}    num2cell(2369:2384)           {};
  20909     19      10       {}     {72}    num2cell(2369:2384)           {};
  % additional CSI-RS cases
  20874      0      10       {}     {72}    num2cell(2369:2384)           {82 83};  % slot 60
  20875      1      10       {}   {72 71}   num2cell(2337:2352)           {82 83};  % slot 61
  20884      3      10       {}     {72}    num2cell(2353:2368)           {};       % slot 63
  % 64TR 25-3 column G, 64 UEs per TTI
  20910      0      10       {22}   {72}    num2cell(2401:2464)           {};
  20911      1      10       {}   {72 71}   num2cell(2465:2528)           {};
  20912      2      10       {}   {72 71}   num2cell(2465:2528)           {};
  20913      3      10       {}     {72}    num2cell(2529:2592)           {};
  20916      6      10       {}     {72}    num2cell(2593:2656)           {};
  20917      7      10       {}     {72}    num2cell(2593:2656)           {};
  20918      8      10       {}     {72}    num2cell(2593:2656)           {};
  20919      9      10       {}     {72}    num2cell(2593:2656)           {};
  20920     10      10       {}   {72 70}  num2cell([2657:2720,887])      {};  % both 72 and 70 PDDCH is on symbol 0, 887 is PDSCH for CCH
  20921     11      10       {}   {72 71}   num2cell(2465:2528)           {};
  20922     12      10       {}   {72 71}   num2cell(2465:2528)           {};
  20923     13      10       {}     {72}    num2cell(2529:2592)           {};
  20926     16      10       {}     {72}    num2cell(2593:2656)           {};
  20927     17      10       {}     {72}    num2cell(2593:2656)           {};
  20928     18      10       {}     {72}    num2cell(2593:2656)           {};
  20929     19      10       {}     {72}    num2cell(2593:2656)           {};
  20930      0      10       {}     {72}    num2cell(2593:2656)           {82 83};
  20931      1      10       {}   {72 71}   num2cell(2465:2528)           {82 83};
  20932      2      10       {}   {72 71}   num2cell(2465:2528)           {};
  20933      3      10       {}     {}                {}                  {113 114};  % slot 23: 8-port CSI_RS + SRS, for nrSim 90649, 90650
  20934      3      10       {}     {}                {}                  {119 120 114};  % slot 23: 8-port + 32-port CSI_RS + SRS, for nrSim 90651, 90652
  20936      6      10       {}     {72}    num2cell(2593:2656)           {};
  20937      7      10       {}     {72}    num2cell(2593:2656)           {};
  20938      8      10       {}     {72}    num2cell(2593:2656)           {};
  20939      9      10       {}     {72}    num2cell(2593:2656)           {};
  20940     10      10       {}     {72}    num2cell(2593:2656)           {};
  20941     11      10       {}   {72 71}   num2cell(2465:2528)           {};
  20942     12      10       {}   {72 71}   num2cell(2465:2528)           {};
  20943     13      10       {}     {72}    num2cell(2529:2592)           {};
  20946     16      10       {}     {72}    num2cell(2593:2656)           {};
  20947     17      10       {}     {72}    num2cell(2593:2656)           {};
  20948     18      10       {}     {72}    num2cell(2593:2656)           {};
  20949     19      10       {}     {72}    num2cell(2593:2656)           {};
  % additional CSI-RS cases
  20914      0      10       {}     {72}    num2cell(2593:2656)           {82 83};  % slot 60
  20915      1      10       {}   {72 71}   num2cell(2465:2528)           {82 83};  % slot 61
  20924      3      10       {}     {72}    num2cell(2529:2592)           {};       % slot 63
  % 64TR 25-3 column E 32DL
  20950      0      10       {22}   {80}    num2cell(2721:2752)           {};
  20951      1      10       {}   {80 71}   num2cell(2753:2784)           {};
  20952      2      10       {}   {80 71}   num2cell(2753:2784)           {};
  20953      3      10       {}     {80}    num2cell(2785:2816)           {};
  20956      6      10       {}     {80}    num2cell(2817:2848)           {};
  20957      7      10       {}     {80}    num2cell(2817:2848)           {};
  20958      8      10       {}     {80}    num2cell(2817:2848)           {};
  20959      9      10       {}     {80}    num2cell(2817:2848)           {};
  20960     10      10       {}   {80 70}  num2cell([2849:2880,847])      {};  % both 80 and 70 PDDCH is on symbol 0, 847 is PDSCH for CCH
  20961     11      10       {}   {80 71}   num2cell(2753:2784)           {};
  20962     12      10       {}   {80 71}   num2cell(2753:2784)           {};
  20963     13      10       {}     {80}    num2cell(2785:2816)           {};
  20966     16      10       {}     {80}    num2cell(2817:2848)           {};
  20967     17      10       {}     {80}    num2cell(2817:2848)           {};
  20968     18      10       {}     {80}    num2cell(2817:2848)           {};
  20969     19      10       {}     {80}    num2cell(2817:2848)           {};
  20970      0      10       {}     {80}    num2cell(2817:2848)           {82 83};
  20971      1      10       {}   {80 71}   num2cell(2753:2784)           {82 83};
  20972      2      10       {}   {80 71}   num2cell(2753:2784)           {};
  20973      3      10       {}     {}                {}                  {113 114};  % slot 23: CSI_RS + SRS
  20976      6      10       {}     {80}    num2cell(2817:2848)           {};
  20977      7      10       {}     {80}    num2cell(2817:2848)           {};
  20978      8      10       {}     {80}    num2cell(2817:2848)           {};
  20979      9      10       {}     {80}    num2cell(2817:2848)           {};
  20980     10      10       {}     {80}    num2cell(2817:2848)           {};
  20981     11      10       {}   {80 71}   num2cell(2753:2784)           {};
  20982     12      10       {}   {80 71}   num2cell(2753:2784)           {};
  20983     13      10       {}     {80}    num2cell(2785:2816)           {};
  20986     16      10       {}     {80}    num2cell(2817:2848)           {};
  20987     17      10       {}     {80}    num2cell(2817:2848)           {};
  20988     18      10       {}     {80}    num2cell(2817:2848)           {};
  20989     19      10       {}     {80}    num2cell(2817:2848)           {};
  % additional CSI-RS cases
  20954      0      10       {}     {80}    num2cell(2817:2848)           {82 83};  % slot 60
  20955      1      10       {}   {80 71}   num2cell(2753:2784)           {82 83};  % slot 61
  20964      3      10       {}     {80}    num2cell(2785:2816)           {};       % slot 63
  % 64TR 'worst case' Ph4 column B, srsPrbGrpSize 4 and bfwPrbGrpSize 16
  20990      0      10       {22}   {72}    num2cell(2881:2896)           {};
  20991      1      10       {}   {72 71}   num2cell(2897:2912)           {};
  20992      2      10       {}   {72 71}   num2cell(2897:2912)           {};
  20993      3      10       {}     {72}    num2cell(2913:2928)           {};
  20996      6      10       {}     {72}    num2cell(2929:2944)           {};
  20997      7      10       {}     {72}    num2cell(2929:2944)           {};
  20998      8      10       {}     {72}    num2cell(2929:2944)           {};
  20999      9      10       {}     {72}    num2cell(2929:2944)           {};
  21000     10      10       {}   {72 70}  num2cell([2945:2960,847])      {};  % both 72 and 70 PDDCH is on symbol 0, 847 is PDSCH for CCH
  21001     11      10       {}   {72 71}   num2cell(2897:2912)           {};
  21002     12      10       {}   {72 71}   num2cell(2897:2912)           {};
  21003     13      10       {}     {72}    num2cell(2913:2928)           {};
  21006     16      10       {}     {72}    num2cell(2929:2944)           {};
  21007     17      10       {}     {72}    num2cell(2929:2944)           {};
  21008     18      10       {}     {72}    num2cell(2929:2944)           {};
  21009     19      10       {}     {72}    num2cell(2929:2944)           {};
  21010      0      10       {}     {72}    num2cell(2929:2944)           {82 83};
  21011      1      10       {}   {72 71}   num2cell(2897:2912)           {82 83};
  21012      2      10       {}   {72 71}   num2cell(2897:2912)           {};
  21013      3      10       {}     {}                {}                  {113 114};  % slot 23: CSI_RS + SRS
  21016      6      10       {}     {72}    num2cell(2929:2944)           {};
  21017      7      10       {}     {72}    num2cell(2929:2944)           {};
  21018      8      10       {}     {72}    num2cell(2929:2944)           {};
  21019      9      10       {}     {72}    num2cell(2929:2944)           {};
  21020     10      10       {}     {72}    num2cell(2929:2944)           {};
  21021     11      10       {}   {72 71}   num2cell(2897:2912)           {};
  21022     12      10       {}   {72 71}   num2cell(2897:2912)           {};
  21023     13      10       {}     {72}    num2cell(2913:2928)           {};
  21026     16      10       {}     {72}    num2cell(2929:2944)           {};
  21027     17      10       {}     {72}    num2cell(2929:2944)           {};
  21028     18      10       {}     {72}    num2cell(2929:2944)           {};
  21029     19      10       {}     {72}    num2cell(2929:2944)           {};
  % additional CSI-RS cases
  20994      0      10       {}     {72}    num2cell(2929:2944)           {82 83};  % slot 60
  20995      1      10       {}   {72 71}   num2cell(2897:2912)           {82 83};  % slot 61
  21004      3      10       {}     {72}    num2cell(2913:2928)           {};       % slot 63
  21005      3      10       {}     {}                {}                  {115 116};  % slot 23: CSI_RS + SRS
  % 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 100 MHz heavy
  21030      0      10       {22}   {72}    num2cell(888:907)                    {};
  21031      1      10       {}   {72 71}   num2cell(908:927)                    {};
  21032      2      10       {}   {72 71}   num2cell(908:927)                    {};
  21033      3      10       {}     {72}    num2cell(928:947)                    {};
  21036      6      10       {}     {72}    num2cell([948:955, 896:907])         {};
  21037      7      10       {}     {72}    num2cell([948:955, 896:907])         {};
  21038      8      10       {}     {72}    num2cell([948:955, 896:907])         {};
  21039      9      10       {}     {72}    num2cell([948:955, 896:907])         {};
  21040     10      10       {}   {72 70}   num2cell([948:955, 896:906, 956, 957])   {};  % both 72 and 70 PDDCH is on symbol 0, 957 is PDSCH for CCH
  21041     11      10       {}   {72 71}   num2cell(908:927)                    {};
  21042     12      10       {}   {72 71}   num2cell(908:927)                    {};
  21043     13      10       {}     {72}    num2cell(928:947)                    {};
  21046     16      10       {}     {72}    num2cell([948:955, 896:907])         {};
  21047     17      10       {}     {72}    num2cell([948:955, 896:907])         {};
  21048     18      10       {}     {72}    num2cell([948:955, 896:907])         {};
  21049     19      10       {}     {72}    num2cell([948:955, 896:907])         {};
  21050      0      10       {}     {72}    num2cell([948:955, 896:907])         {82 83};
  21051      1      10       {}   {72 71}   num2cell(908:927)                    {82 83};
  21052      2      10       {}   {72 71}   num2cell(908:927)                    {};
  21053      3      10       {}      {}     {}                                   {113 114};  % slot 23: CSI_RS + SRS
  21056      6      10       {}     {72}    num2cell([948:955, 896:907])         {};
  21057      7      10       {}     {72}    num2cell([948:955, 896:907])         {};
  21058      8      10       {}     {72}    num2cell([948:955, 896:907])         {};
  21059      9      10       {}     {72}    num2cell([948:955, 896:907])         {};
  21060     10      10       {}     {72}    num2cell([948:955, 896:907])         {};
  21061     11      10       {}   {72 71}   num2cell(908:927)                    {};
  21062     12      10       {}   {72 71}   num2cell(908:927)                    {};
  21063     13      10       {}     {72}    num2cell(928:947)                    {};
  21066     16      10       {}     {72}    num2cell([948:955, 896:907])         {};
  21067     17      10       {}     {72}    num2cell([948:955, 896:907])         {};
  21068     18      10       {}     {72}    num2cell([948:955, 896:907])         {};
  21069     19      10       {}     {72}    num2cell([948:955, 896:907])         {};
  % additional CSI-RS cases
  21034      0      10       {}     {72}    num2cell([948:955, 896:907])         {82 83};  % slot 60
  21035      1      10       {}   {72 71}   num2cell(908:927)                    {82 83};  % slot 61
  21044      3      10       {}     {72}    num2cell(928:947)                    {};       % slot 63
  21045      3      10       {}     {}      {}                                   {115 116};  % slot 23: CSI_RS + SRS
  % 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 100 MHz light
  21070      0      10       {22}   {72}    num2cell(958:969)                    {};
  21071      1      10       {}   {72 71}   num2cell(970:981)                    {};
  21072      2      10       {}   {72 71}   num2cell(970:981)                    {};
  21073      3      10       {}     {72}    num2cell(982:993)                    {};
  21076      6      10       {}     {72}    num2cell([994:997, 962:969])         {};
  21077      7      10       {}     {72}    num2cell([994:997, 962:969])         {};
  21078      8      10       {}     {72}    num2cell([994:997, 962:969])         {};
  21079      9      10       {}     {72}    num2cell([994:997, 962:969])         {};
  21080     10      10       {}   {72 70}   num2cell([994:997, 962:968, 998, 999])   {};  % both 72 and 70 PDDCH is on symbol 0, 999 is PDSCH for CCH
  21081     11      10       {}   {72 71}   num2cell(970:981)                    {};
  21082     12      10       {}   {72 71}   num2cell(970:981)                    {};
  21083     13      10       {}     {72}    num2cell(982:993)                    {};
  21086     16      10       {}     {72}    num2cell([994:997, 962:969])         {};
  21087     17      10       {}     {72}    num2cell([994:997, 962:969])         {};
  21088     18      10       {}     {72}    num2cell([994:997, 962:969])         {};
  21089     19      10       {}     {72}    num2cell([994:997, 962:969])         {};
  21090      0      10       {}     {72}    num2cell([994:997, 962:969])         {82 83};
  21091      1      10       {}   {72 71}   num2cell(970:981)                    {82 83};
  21092      2      10       {}   {72 71}   num2cell(970:981)                    {};
  21093      3      10       {}      {}     {}                                   {113 114};  % slot 23: CSI_RS + SRS
  21096      6      10       {}     {72}    num2cell([994:997, 962:969])         {};
  21097      7      10       {}     {72}    num2cell([994:997, 962:969])         {};
  21098      8      10       {}     {72}    num2cell([994:997, 962:969])         {};
  21099      9      10       {}     {72}    num2cell([994:997, 962:969])         {};
  21100     10      10       {}     {72}    num2cell([994:997, 962:969])         {};
  21101     11      10       {}   {72 71}   num2cell(970:981)                    {};
  21102     12      10       {}   {72 71}   num2cell(970:981)                    {};
  21103     13      10       {}     {72}    num2cell(982:993)                    {};
  21106     16      10       {}     {72}    num2cell([994:997, 962:969])         {};
  21107     17      10       {}     {72}    num2cell([994:997, 962:969])         {};
  21108     18      10       {}     {72}    num2cell([994:997, 962:969])         {};
  21109     19      10       {}     {72}    num2cell([994:997, 962:969])         {};
  % additional CSI-RS cases
  21074      0      10       {}     {72}    num2cell([994:997, 962:969])         {82 83};  % slot 60
  21075      1      10       {}   {72 71}   num2cell(970:981)                    {82 83};  % slot 61
  21084      3      10       {}     {72}    num2cell(982:993)                    {};       % slot 63
  21085      3      10       {}     {}      {}                                   {115 116};  % slot 23: CSI_RS + SRS

  % 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 90 MHz heavy
  21110      0      14       {22}   {83}    num2cell(1000:1019)                  {};
  21111      1      14       {}   {83 82}   num2cell(1020:1039)                  {};
  21112      2      14       {}   {83 82}   num2cell(1020:1039)                  {};
  21113      3      14       {}     {83}    num2cell(1040:1059)                  {};
  21116      6      14       {}     {83}    num2cell([1060:1067, 1008:1019])     {};
  21117      7      14       {}     {83}    num2cell([1060:1067, 1008:1019])     {};
  21118      8      14       {}     {83}    num2cell([1060:1067, 1008:1019])     {};
  21119      9      14       {}     {83}    num2cell([1060:1067, 1008:1019])     {};
  21120     10      14       {}   {83 81}   num2cell([1060:1067, 1008:1018, 1068, 1069])   {};  % both 83 and 81 PDDCH is on symbol 0, 1069 is PDSCH for CCH
  21121     11      14       {}   {83 82}   num2cell(1020:1039)                  {};
  21122     12      14       {}   {83 82}   num2cell(1020:1039)                  {};
  21123     13      14       {}     {83}    num2cell(1040:1059)                  {};
  21126     16      14       {}     {83}    num2cell([1060:1067, 1008:1019])     {};
  21127     17      14       {}     {83}    num2cell([1060:1067, 1008:1019])     {};
  21128     18      14       {}     {83}    num2cell([1060:1067, 1008:1019])     {};
  21129     19      14       {}     {83}    num2cell([1060:1067, 1008:1019])     {};
  21130      0      14       {}     {83}    num2cell([1060:1067, 1008:1019])     {122 123};
  21131      1      14       {}   {83 82}   num2cell(1020:1039)                  {122 123};
  21132      2      14       {}   {83 82}   num2cell(1020:1039)                  {};
  21133      3      14       {}      {}     {}                                   {124 125};  % slot 23: CSI_RS + SRS
  21136      6      14       {}     {83}    num2cell([1060:1067, 1008:1019])     {};
  21137      7      14       {}     {83}    num2cell([1060:1067, 1008:1019])     {};
  21138      8      14       {}     {83}    num2cell([1060:1067, 1008:1019])     {};
  21139      9      14       {}     {83}    num2cell([1060:1067, 1008:1019])     {};
  21140     10      14       {}     {83}    num2cell([1060:1067, 1008:1019])     {};
  21141     11      14       {}   {83 82}   num2cell(1020:1039)                  {};
  21142     12      14       {}   {83 82}   num2cell(1020:1039)                  {};
  21143     13      14       {}     {83}    num2cell(1040:1059)                  {};
  21146     16      14       {}     {83}    num2cell([1060:1067, 1008:1019])     {};
  21147     17      14       {}     {83}    num2cell([1060:1067, 1008:1019])     {};
  21148     18      14       {}     {83}    num2cell([1060:1067, 1008:1019])     {};
  21149     19      14       {}     {83}    num2cell([1060:1067, 1008:1019])     {};
  % additional CSI-RS cases
  21114      0      14       {}     {83}    num2cell([1060:1067, 1008:1019])     {122 123};  % slot 60
  21115      1      14       {}   {83 82}   num2cell(1020:1039)                  {122 123};  % slot 61
  21124      3      14       {}     {83}    num2cell(1040:1059)                  {};       % slot 63
  21125      3      14       {}     {}      {}                                   {126 127};  % slot 23: CSI_RS + SRS

  % 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 90 MHz light
  21150      0      14       {22}   {83}    num2cell(1070:1081)                  {};
  21151      1      14       {}   {83 82}   num2cell(1082:1093)                  {};
  21152      2      14       {}   {83 82}   num2cell(1082:1093)                  {};
  21153      3      14       {}     {83}    num2cell(1094:1105)                  {};
  21156      6      14       {}     {83}    num2cell([1106:1109, 1074:1081])     {};
  21157      7      14       {}     {83}    num2cell([1106:1109, 1074:1081])     {};
  21158      8      14       {}     {83}    num2cell([1106:1109, 1074:1081])     {};
  21159      9      14       {}     {83}    num2cell([1106:1109, 1074:1081])     {};
  21160     10      14       {}   {83 81}   num2cell([1106:1109, 1074:1080, 1110, 1111])   {};  % both 83 and 81 PDDCH is on symbol 0, 1111 is PDSCH for CCH
  21161     11      14       {}   {83 82}   num2cell(1082:1093)                  {};
  21162     12      14       {}   {83 82}   num2cell(1082:1093)                  {};
  21163     13      14       {}     {83}    num2cell(1094:1105)                  {};
  21166     16      14       {}     {83}    num2cell([1106:1109, 1074:1081])     {};
  21167     17      14       {}     {83}    num2cell([1106:1109, 1074:1081])     {};
  21168     18      14       {}     {83}    num2cell([1106:1109, 1074:1081])     {};
  21169     19      14       {}     {83}    num2cell([1106:1109, 1074:1081])     {};
  21170      0      14       {}     {83}    num2cell([1106:1109, 1074:1081])     {122 123};
  21171      1      14       {}   {83 82}   num2cell(1082:1093)                  {122 123};
  21172      2      14       {}   {83 82}   num2cell(1082:1093)                  {};
  21173      3      14       {}      {}     {}                                   {124 125};  % slot 23: CSI_RS + SRS
  21176      6      14       {}     {83}    num2cell([1106:1109, 1074:1081])     {};
  21177      7      14       {}     {83}    num2cell([1106:1109, 1074:1081])     {};
  21178      8      14       {}     {83}    num2cell([1106:1109, 1074:1081])     {};
  21179      9      14       {}     {83}    num2cell([1106:1109, 1074:1081])     {};
  21180     10      14       {}     {83}    num2cell([1106:1109, 1074:1081])     {};
  21181     11      14       {}   {83 82}   num2cell(1082:1093)                  {};
  21182     12      14       {}   {83 82}   num2cell(1082:1093)                  {};
  21183     13      14       {}     {83}    num2cell(1094:1105)                  {};
  21186     16      14       {}     {83}    num2cell([1106:1109, 1074:1081])     {};
  21187     17      14       {}     {83}    num2cell([1106:1109, 1074:1081])     {};
  21188     18      14       {}     {83}    num2cell([1106:1109, 1074:1081])     {};
  21189     19      14       {}     {83}    num2cell([1106:1109, 1074:1081])     {};
  % additional CSI-RS cases
  21154      0      14       {}     {83}    num2cell([1106:1109, 1074:1081])     {122 123};  % slot 60
  21155      1      14       {}   {83 82}   num2cell(1082:1093)                  {122 123};  % slot 61
  21164      3      14       {}     {83}    num2cell(1094:1105)                  {};       % slot 63
  21165      3      14       {}     {}      {}                                   {126 127};  % slot 23: CSI_RS + SRS

  % 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 60 MHz heavy
  21190      0      15       {22}   {86}    num2cell(1112:1131)                  {};
  21191      1      15       {}   {86 85}   num2cell(1132:1151)                  {};
  21192      2      15       {}   {86 85}   num2cell(1132:1151)                  {};
  21193      3      15       {}     {86}    num2cell(1152:1171)                  {};
  21196      6      15       {}     {86}    num2cell([1172:1179, 1120:1131])     {};
  21197      7      15       {}     {86}    num2cell([1172:1179, 1120:1131])     {};
  21198      8      15       {}     {86}    num2cell([1172:1179, 1120:1131])     {};
  21199      9      15       {}     {86}    num2cell([1172:1179, 1120:1131])     {};
  21200     10      15       {}   {86 84}   num2cell([1172:1179, 1120:1130, 1180, 1181])   {};  % both 86 and 84 PDDCH is on symbol 0, 1181 is PDSCH for CCH
  21201     11      15       {}   {86 85}   num2cell(1132:1151)                  {};
  21202     12      15       {}   {86 85}   num2cell(1132:1151)                  {};
  21203     13      15       {}     {86}    num2cell(1152:1171)                  {};
  21206     16      15       {}     {86}    num2cell([1172:1179, 1120:1131])     {};
  21207     17      15       {}     {86}    num2cell([1172:1179, 1120:1131])     {};
  21208     18      15       {}     {86}    num2cell([1172:1179, 1120:1131])     {};
  21209     19      15       {}     {86}    num2cell([1172:1179, 1120:1131])     {};
  21210      0      15       {}     {86}    num2cell([1172:1179, 1120:1131])     {128 129};
  21211      1      15       {}   {86 85}   num2cell(1132:1151)                  {128 129};
  21212      2      15       {}   {86 85}   num2cell(1132:1151)                  {};
  21213      3      15       {}      {}     {}                                   {130 131};  % slot 23: CSI_RS + SRS
  21216      6      15       {}     {86}    num2cell([1172:1179, 1120:1131])     {};
  21217      7      15       {}     {86}    num2cell([1172:1179, 1120:1131])     {};
  21218      8      15       {}     {86}    num2cell([1172:1179, 1120:1131])     {};
  21219      9      15       {}     {86}    num2cell([1172:1179, 1120:1131])     {};
  21220     10      15       {}     {86}    num2cell([1172:1179, 1120:1131])     {};
  21221     11      15       {}   {86 85}   num2cell(1132:1151)                  {};
  21222     12      15       {}   {86 85}   num2cell(1132:1151)                  {};
  21223     13      15       {}     {86}    num2cell(1152:1171)                  {};
  21226     16      15       {}     {86}    num2cell([1172:1179, 1120:1131])     {};
  21227     17      15       {}     {86}    num2cell([1172:1179, 1120:1131])     {};
  21228     18      15       {}     {86}    num2cell([1172:1179, 1120:1131])     {};
  21229     19      15       {}     {86}    num2cell([1172:1179, 1120:1131])     {};
  % additional CSI-RS cases
  21194      0      15       {}     {86}    num2cell([1172:1179, 1120:1131])     {128 129};  % slot 60
  21195      1      15       {}   {86 85}   num2cell(1132:1151)                  {128 129};  % slot 61
  21204      3      15       {}     {86}    num2cell(1152:1171)                  {};       % slot 63
  21205      3      15       {}     {}      {}                                   {132 133};  % slot 23: CSI_RS + SRS

  % 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 60 MHz light
  21230      0      15       {22}   {86}    num2cell(1182:1193)                  {};
  21231      1      15       {}   {86 85}   num2cell(1194:1205)                  {};
  21232      2      15       {}   {86 85}   num2cell(1194:1205)                  {};
  21233      3      15       {}     {86}    num2cell(1206:1217)                  {};
  21236      6      15       {}     {86}    num2cell([1218:1221, 1186:1193])     {};
  21237      7      15       {}     {86}    num2cell([1218:1221, 1186:1193])     {};
  21238      8      15       {}     {86}    num2cell([1218:1221, 1186:1193])     {};
  21239      9      15       {}     {86}    num2cell([1218:1221, 1186:1193])     {};
  21240     10      15       {}   {86 84}   num2cell([1218:1221, 1186:1192, 1222, 1223])   {};  % both 86 and 84 PDDCH is on symbol 0, 1223 is PDSCH for CCH
  21241     11      15       {}   {86 85}   num2cell(1194:1205)                  {};
  21242     12      15       {}   {86 85}   num2cell(1194:1205)                  {};
  21243     13      15       {}     {86}    num2cell(1206:1217)                  {};
  21246     16      15       {}     {86}    num2cell([1218:1221, 1186:1193])     {};
  21247     17      15       {}     {86}    num2cell([1218:1221, 1186:1193])     {};
  21248     18      15       {}     {86}    num2cell([1218:1221, 1186:1193])     {};
  21249     19      15       {}     {86}    num2cell([1218:1221, 1186:1193])     {};
  21250      0      15       {}     {86}    num2cell([1218:1221, 1186:1193])     {128 129};
  21251      1      15       {}   {86 85}   num2cell(1194:1205)                  {128 129};
  21252      2      15       {}   {86 85}   num2cell(1194:1205)                  {};
  21253      3      15       {}      {}     {}                                   {130 131};  % slot 23: CSI_RS + SRS
  21256      6      15       {}     {86}    num2cell([1218:1221, 1186:1193])     {};
  21257      7      15       {}     {86}    num2cell([1218:1221, 1186:1193])     {};
  21258      8      15       {}     {86}    num2cell([1218:1221, 1186:1193])     {};
  21259      9      15       {}     {86}    num2cell([1218:1221, 1186:1193])     {};
  21260     10      15       {}     {86}    num2cell([1218:1221, 1186:1193])     {};
  21261     11      15       {}   {86 85}   num2cell(1194:1205)                  {};
  21262     12      15       {}   {86 85}   num2cell(1194:1205)                  {};
  21263     13      15       {}     {86}    num2cell(1206:1217)                  {};
  21266     16      15       {}     {86}    num2cell([1218:1221, 1186:1193])     {};
  21267     17      15       {}     {86}    num2cell([1218:1221, 1186:1193])     {};
  21268     18      15       {}     {86}    num2cell([1218:1221, 1186:1193])     {};
  21269     19      15       {}     {86}    num2cell([1218:1221, 1186:1193])     {};
  % additional CSI-RS cases
  21234      0      15       {}     {86}    num2cell([1218:1221, 1186:1193])     {128 129};  % slot 60
  21235      1      15       {}   {86 85}   num2cell(1194:1205)                  {128 129};  % slot 61
  21244      3      15       {}     {86}    num2cell(1206:1217)                  {};       % slot 63
  21245      3      15       {}     {}      {}                                   {132 133};  % slot 23: CSI_RS + SRS

  % 64TR 25-3 column D 24DL
  21270      0      10       {22}   {87}    num2cell(2961:2984)           {};
  21271      1      10       {}   {87 71}   num2cell(2985:3008)           {};
  21272      2      10       {}   {87 71}   num2cell(2985:3008)           {};
  21273      3      10       {}     {87}    num2cell(3009:3032)           {};
  21276      6      10       {}     {87}    num2cell(3033:3056)           {};
  21277      7      10       {}     {87}    num2cell(3033:3056)           {};
  21278      8      10       {}     {87}    num2cell(3033:3056)           {};
  21279      9      10       {}     {87}    num2cell(3033:3056)           {};
  21280     10      10       {}   {87 70}  num2cell([3057:3080,847])      {};  % both 80 and 70 PDDCH is on symbol 0, 847 is PDSCH for CCH
  21281     11      10       {}   {87 71}   num2cell(2985:3008)           {};
  21282     12      10       {}   {87 71}   num2cell(2985:3008)           {};
  21283     13      10       {}     {87}    num2cell(3009:3032)           {};
  21286     16      10       {}     {87}    num2cell(3033:3056)           {};
  21287     17      10       {}     {87}    num2cell(3033:3056)           {};
  21288     18      10       {}     {87}    num2cell(3033:3056)           {};
  21289     19      10       {}     {87}    num2cell(3033:3056)           {};
  21290      0      10       {}     {87}    num2cell(3033:3056)           {82 83};
  21291      1      10       {}   {87 71}   num2cell(2985:3008)           {82 83};
  21292      2      10       {}   {87 71}   num2cell(2985:3008)           {};
  21293      3      10       {}     {}                {}                  {113 114};  % slot 23: CSI_RS + SRS
  21296      6      10       {}     {87}    num2cell(3033:3056)           {};
  21297      7      10       {}     {87}    num2cell(3033:3056)           {};
  21298      8      10       {}     {87}    num2cell(3033:3056)           {};
  21299      9      10       {}     {87}    num2cell(3033:3056)           {};
  21300     10      10       {}     {87}    num2cell(3033:3056)           {};
  21301     11      10       {}   {87 71}   num2cell(2985:3008)           {};
  21302     12      10       {}   {87 71}   num2cell(2985:3008)           {};
  21303     13      10       {}     {87}    num2cell(3009:3032)           {};
  21306     16      10       {}     {87}    num2cell(3033:3056)           {};
  21307     17      10       {}     {87}    num2cell(3033:3056)           {};
  21308     18      10       {}     {87}    num2cell(3033:3056)           {};
  21309     19      10       {}     {87}    num2cell(3033:3056)           {};
  % additional CSI-RS cases
  21274      0      10       {}     {87}    num2cell(3033:3056)           {82 83};  % slot 60
  21275      1      10       {}   {87 71}   num2cell(2985:3008)           {82 83};  % slot 61
  21284      3      10       {}     {87}    num2cell(3009:3032)           {};       % slot 63

      % pattern 5: F08 8C (PDSCH + PDCCH + CSI-RS + PBCH , different RNTIs)
%       148     0        1      {4}        {3}    num2cell(21:36)  {3};
%       149     0        1      {4}        {3}    num2cell(21:36)  {3};
%       150     0        1      {4}        {3}    num2cell(21:36)  {3};
%       151     0        1      {4}        {3}    num2cell(21:36)  {3};
%       152     0        1      {4}        {3}    num2cell(21:36)  {3};
%       153     0        1      {4}        {3}    num2cell(21:36)  {3};
%       154     0        1      {4}        {3}    num2cell(21:36)  {3};
%       155     0        1      {4}        {3}    num2cell(21:36)  {3};
     % pattern 3: F08 8C (PDSCH + PDCCH + PBCH)
%       156     0        1      {4}        {3}    num2cell(21:36)  {};
%       157     0        1      {4}        {3}    num2cell(21:36)  {};
%       158     0        1      {4}        {3}    num2cell(21:36)  {};
%       159     0        1      {4}        {3}    num2cell(21:36)  {};
%       160     0        1      {4}        {3}    num2cell(21:36)  {};
%       161     0        1      {4}        {3}    num2cell(21:36)  {};
%       162     0        1      {4}        {3}    num2cell(21:36)  {};
%       163     0        1      {4}        {3}    num2cell(21:36)  {};
     % pattern 4: F08 8C ZP CSI-RS (PDSCH + PDCCH + PBCH + ZP CSI-RS)
%       164     0        1      {4}        {3}    num2cell(21:36)  {7};
%       165     0        1      {4}        {3}    num2cell(21:36)  {7};
%       166     0        1      {4}        {3}    num2cell(21:36)  {7};
%       167     0        1      {4}        {3}    num2cell(21:36)  {7};
%       168     0        1      {4}        {3}    num2cell(21:36)  {7};
%       169     0        1      {4}        {3}    num2cell(21:36)  {7};
%       170     0        1      {4}        {3}    num2cell(21:36)  {7};
%       171     0        1      {4}        {3}    num2cell(21:36)  {7};
      % pattern 5: F08 8C (PDSCH + PDCCH + CSI-RS + PBCH (slot 10-19) , different RNTIs)
%       172     0        1      {5}        {3}    num2cell(21:36)  {3};
%       173     0        1      {5}        {3}    num2cell(21:36)  {3};
%       174     0        1      {5}        {3}    num2cell(21:36)  {3};
%       175     0        1      {5}        {3}    num2cell(21:36)  {3};
%       176     0        1      {5}        {3}    num2cell(21:36)  {3};
%       177     0        1      {5}        {3}    num2cell(21:36)  {3};
%       178     0        1      {5}        {3}    num2cell(21:36)  {3};
%       179     0        1      {5}        {3}    num2cell(21:36)  {3};
%      % pattern 3: F08 8C (PDSCH + PDCCH + PBCH (slot 10-19))
%       180     0        1      {5}        {3}    num2cell(21:36)  {};
%       181     0        1      {5}        {3}    num2cell(21:36)  {};
%       182     0        1      {5}        {3}    num2cell(21:36)  {};
%       183     0        1      {5}        {3}    num2cell(21:36)  {};
%       184     0        1      {5}        {3}    num2cell(21:36)  {};
%       185     0        1      {5}        {3}    num2cell(21:36)  {};
%       186     0        1      {5}        {3}    num2cell(21:36)  {};
%       187     0        1      {5}        {3}    num2cell(21:36)  {};
%      % pattern 4: F08 8C ZP CSI-RS (PDSCH + PDCCH + PBCH (slot 10-19) + ZP CSI-RS)
%       188     0        1      {5}        {3}    num2cell(21:36)  {7};
%       189     0        1      {5}        {3}    num2cell(21:36)  {7};
%       190     0        1      {5}        {3}    num2cell(21:36)  {7};
%       191     0        1      {5}        {3}    num2cell(21:36)  {7};
%       192     0        1      {5}        {3}    num2cell(21:36)  {7};
%       193     0        1      {5}        {3}    num2cell(21:36)  {7};
%       194     0        1      {5}        {3}    num2cell(21:36)  {7};
%       195     0        1      {5}        {3}    num2cell(21:36)  {7};
      
      %% 8C 50% traffic
      % pattern 2: F08 8C 50% traffic (PDSCH + PDCCH + CSI-RS + PBCH , different RNTIs)
%       196     0        1      {4}        {3}    num2cell(53:68)  {3};
%       197     0        1      {4}        {3}    num2cell(53:68)  {3};
%       198     0        1      {4}        {3}    num2cell(53:68)  {3};
%       199     0        1      {4}        {3}    num2cell(53:68)  {3};
%       200     0        1      {4}        {3}    num2cell(53:68)  {3};
%       201     0        1      {4}        {3}    num2cell(53:68)  {3};
%       202     0        1      {4}        {3}    num2cell(53:68)  {3};
%       203     0        1      {4}        {3}    num2cell(53:68)  {3};
%      % pattern 0: F08 8C 50% traffic (PDSCH + PDCCH + PBCH)
        204     0        1      {4}        {3}    num2cell(53:68)  {};
        205     0        1      {4}        {3}    num2cell(53:68)  {};
%       206     0        1      {4}        {3}    num2cell(53:68)  {};
%       207     0        1      {4}        {3}    num2cell(53:68)  {};
%       208     0        1      {4}        {3}    num2cell(53:68)  {};
%       209     0        1      {4}        {3}    num2cell(53:68)  {};
%       210     0        1      {4}        {3}    num2cell(53:68)  {};
%       211     0        1      {4}        {3}    num2cell(53:68)  {};
%      % pattern 1: F08 8C ZP CSI-RS 50% traffic (PDSCH + PDCCH + PBCH + ZP CSI-RS)
%       212     0        1      {4}        {3}    num2cell(53:68)  {7};
%       213     0        1      {4}        {3}    num2cell(53:68)  {7};
%       214     0        1      {4}        {3}    num2cell(53:68)  {7};
%       215     0        1      {4}        {3}    num2cell(53:68)  {7};
%       216     0        1      {4}        {3}    num2cell(53:68)  {7};
%       217     0        1      {4}        {3}    num2cell(53:68)  {7};
%       218     0        1      {4}        {3}    num2cell(53:68)  {7};
%       219     0        1      {4}        {3}    num2cell(53:68)  {7};
      % pattern 2: F08 8C 50% traffic (PDSCH + PDCCH + CSI-RS + PBCH (slot 10-19) , different RNTIs)
%       220     0        1      {5}        {3}    num2cell(53:68)  {3};
%       221     0        1      {5}        {3}    num2cell(53:68)  {3};
%       222     0        1      {5}        {3}    num2cell(53:68)  {3};
%       223     0        1      {5}        {3}    num2cell(53:68)  {3};
%       224     0        1      {5}        {3}    num2cell(53:68)  {3};
%       225     0        1      {5}        {3}    num2cell(53:68)  {3};
%       226     0        1      {5}        {3}    num2cell(53:68)  {3};
%       227     0        1      {5}        {3}    num2cell(53:68)  {3};
%      % pattern 0: F08 8C 50% traffic (PDSCH + PDCCH + PBCH (slot 10-19))
        228     0        1      {5}        {3}    num2cell(53:68)  {};
        229     0        1      {5}        {3}    num2cell(53:68)  {};
%       230     0        1      {5}        {3}    num2cell(53:68)  {};
%       231     0        1      {5}        {3}    num2cell(53:68)  {};
%       232     0        1      {5}        {3}    num2cell(53:68)  {};
%       233     0        1      {5}        {3}    num2cell(53:68)  {};
%       234     0        1      {5}        {3}    num2cell(53:68)  {};
%       235     0        1      {5}        {3}    num2cell(53:68)  {};
%      % pattern 1: F08 8C ZP CSI-RS 50% traffic (PDSCH + PDCCH + PBCH (slot 10-19) + ZP CSI-RS)
%       236     0        1      {5}        {3}    num2cell(53:68)  {7};
%       237     0        1      {5}        {3}    num2cell(53:68)  {7};
%       238     0        1      {5}        {3}    num2cell(53:68)  {7};
%       239     0        1      {5}        {3}    num2cell(53:68)  {7};
%       240     0        1      {5}        {3}    num2cell(53:68)  {7};
%       241     0        1      {5}        {3}    num2cell(53:68)  {7};
%       242     0        1      {5}        {3}    num2cell(53:68)  {7};
%       243     0        1      {5}        {3}    num2cell(53:68)  {7};
      
    %   % TC#  slotIdx   cell    ssb      pdcch      pdsch     csirs
    %   %% F08 8C relaxed complexity (PDSCH + PDCCH + CSI-RS + PBCH)
    %   244     0        1      {4}        {5}    num2cell(69:74)  {3};
    %   245     0        1      {4}        {5}    num2cell(69:74)  {3};
    %   246     0        1      {4}        {5}    num2cell(69:74)  {3};
    %   247     0        1      {4}        {5}    num2cell(69:74)  {3};
    %   248     0        1      {4}        {5}    num2cell(69:74)  {3};
    %   249     0        1      {4}        {5}    num2cell(69:74)  {3};
    %   250     0        1      {4}        {5}    num2cell(69:74)  {3};
    %   251     0        1      {4}        {5}    num2cell(69:74)  {3};
    %  % F08 8C relaxed complexity (PDSCH + PDCCH + PBCH)
    %   252     0        1      {4}        {5}    num2cell(69:74)  {};
    %   253     0        1      {4}        {5}    num2cell(69:74)  {};
    %   254     0        1      {4}        {5}    num2cell(69:74)  {};
    %   255     0        1      {4}        {5}    num2cell(69:74)  {};
    %   256     0        1      {4}        {5}    num2cell(69:74)  {};
    %   257     0        1      {4}        {5}    num2cell(69:74)  {};
    %   258     0        1      {4}        {5}    num2cell(69:74)  {};
    %   259     0        1      {4}        {5}    num2cell(69:74)  {};
    %  % F08 8C relaxed complexity, ZP CSI-RS (PDSCH + PDCCH + PBCH + ZP CSI-RS)
    %   260     0        1      {4}        {5}    num2cell(69:74)  {7};
    %   261     0        1      {4}        {5}    num2cell(69:74)  {7};
    %   262     0        1      {4}        {5}    num2cell(69:74)  {7};
    %   263     0        1      {4}        {5}    num2cell(69:74)  {7};
    %   264     0        1      {4}        {5}    num2cell(69:74)  {7};
    %   265     0        1      {4}        {5}    num2cell(69:74)  {7};
    %   266     0        1      {4}        {5}    num2cell(69:74)  {7};
    %   267     0        1      {4}        {5}    num2cell(69:74)  {7};
    %   % F08 8C relaxed complexity (PDSCH + PDCCH + CSI-RS + PBCH (slot 10-19))
    %   268     0        1      {5}        {5}    num2cell(69:74)  {3};
    %   269     0        1      {5}        {5}    num2cell(69:74)  {3};
    %   270     0        1      {5}        {5}    num2cell(69:74)  {3};
    %   271     0        1      {5}        {5}    num2cell(69:74)  {3};
    %   272     0        1      {5}        {5}    num2cell(69:74)  {3};
    %   273     0        1      {5}        {5}    num2cell(69:74)  {3};
    %   274     0        1      {5}        {5}    num2cell(69:74)  {3};
    %   275     0        1      {5}        {5}    num2cell(69:74)  {3};
    %  % F08 8C relaxed complexity (PDSCH + PDCCH + PBCH (slot 10-19))
    %   276     0        1      {5}        {5}    num2cell(69:74)  {};
    %   277     0        1      {5}        {5}    num2cell(69:74)  {};
    %   278     0        1      {5}        {5}    num2cell(69:74)  {};
    %   279     0        1      {5}        {5}    num2cell(69:74)  {};
    %   280     0        1      {5}        {5}    num2cell(69:74)  {};
    %   281     0        1      {5}        {5}    num2cell(69:74)  {};
    %   282     0        1      {5}        {5}    num2cell(69:74)  {};
    %   283     0        1      {5}        {5}    num2cell(69:74)  {};
    %  % F08 8C ZP CSI-RS relaxed complexity (PDSCH + PDCCH + PBCH (slot 10-19) + ZP CSI-RS)
    %   284     0        1      {5}        {5}    num2cell(69:74)  {7};
    %   285     0        1      {5}        {5}    num2cell(69:74)  {7};
    %   286     0        1      {5}        {5}    num2cell(69:74)  {7};
    %   287     0        1      {5}        {5}    num2cell(69:74)  {7};
    %   288     0        1      {5}        {5}    num2cell(69:74)  {7};
    %   289     0        1      {5}        {5}    num2cell(69:74)  {7};
    %   290     0        1      {5}        {5}    num2cell(69:74)  {7};
    %   291     0        1      {5}        {5}    num2cell(69:74)  {7};
      
      
    %  % TC#  slotIdx   cell    ssb      pdcch      pdsch     csirs
    %   %% pattern 11: F08 16C relaxed complexity, 50% load, (PDSCH + PDCCH + CSI-RS + PBCH)
    %   292     0        1      {4}        {5}    num2cell(75:80)  {3};
    %   293     0        1      {4}        {5}    num2cell(75:80)  {3};
    %   294     0        1      {4}        {5}    num2cell(75:80)  {3};
    %   295     0        1      {4}        {5}    num2cell(75:80)  {3};
    %   296     0        1      {4}        {5}    num2cell(75:80)  {3};
    %   297     0        1      {4}        {5}    num2cell(75:80)  {3};
    %   298     0        1      {4}        {5}    num2cell(75:80)  {3};
    %   299     0        1      {4}        {5}    num2cell(75:80)  {3};
    %   300     0        1      {4}        {5}    num2cell(75:80)  {3};
    %   301     0        1      {4}        {5}    num2cell(75:80)  {3};
    %   302     0        1      {4}        {5}    num2cell(75:80)  {3};
    %   303     0        1      {4}        {5}    num2cell(75:80)  {3};
    %   304     0        1      {4}        {5}    num2cell(75:80)  {3};
    %   305     0        1      {4}        {5}    num2cell(75:80)  {3};
    %   306     0        1      {4}        {5}    num2cell(75:80)  {3};
    %   307     0        1      {4}        {5}    num2cell(75:80)  {3};
     % F08 16C relaxed complexity, 50% load, (PDSCH + PDCCH + PBCH)
%       308     0        1      {4}        {5}    num2cell(75:80)  {};
%       309     0        1      {4}        {5}    num2cell(75:80)  {};
%       310     0        1      {4}        {5}    num2cell(75:80)  {};
%       311     0        1      {4}        {5}    num2cell(75:80)  {};
%       312     0        1      {4}        {5}    num2cell(75:80)  {};
%       313     0        1      {4}        {5}    num2cell(75:80)  {};
%       314     0        1      {4}        {5}    num2cell(75:80)  {};
%       315     0        1      {4}        {5}    num2cell(75:80)  {};
%       316     0        1      {4}        {5}    num2cell(75:80)  {};
%       317     0        1      {4}        {5}    num2cell(75:80)  {};
%       318     0        1      {4}        {5}    num2cell(75:80)  {};
%       319     0        1      {4}        {5}    num2cell(75:80)  {};
%       320     0        1      {4}        {5}    num2cell(75:80)  {};
%       321     0        1      {4}        {5}    num2cell(75:80)  {};
%       322     0        1      {4}        {5}    num2cell(75:80)  {};
%       323     0        1      {4}        {5}    num2cell(75:80)  {};
     % F08 16C relaxed complexity, 50% load, ZP CSI-RS (PDSCH + PDCCH + PBCH + ZP CSI-RS)
%       324     0        1      {4}        {5}    num2cell(75:80)  {7};
%       325     0        1      {4}        {5}    num2cell(75:80)  {7};
%       326     0        1      {4}        {5}    num2cell(75:80)  {7};
%       327     0        1      {4}        {5}    num2cell(75:80)  {7};
%       328     0        1      {4}        {5}    num2cell(75:80)  {7};
%       329     0        1      {4}        {5}    num2cell(75:80)  {7};
%       330     0        1      {4}        {5}    num2cell(75:80)  {7};
%       331     0        1      {4}        {5}    num2cell(75:80)  {7};
%       332     0        1      {4}        {5}    num2cell(75:80)  {7};
%       333     0        1      {4}        {5}    num2cell(75:80)  {7};
%       334     0        1      {4}        {5}    num2cell(75:80)  {7};
%       335     0        1      {4}        {5}    num2cell(75:80)  {7};
%       336     0        1      {4}        {5}    num2cell(75:80)  {7};
%       337     0        1      {4}        {5}    num2cell(75:80)  {7};
%       338     0        1      {4}        {5}    num2cell(75:80)  {7};
%       339     0        1      {4}        {5}    num2cell(75:80)  {7};
      % pattern 11: F08 16C relaxed complexity, 50% load, (PDSCH + PDCCH + CSI-RS + PBCH (slot 10-19))
    %   340     0        1      {5}        {5}    num2cell(75:80)  {3};
    %   341     0        1      {5}        {5}    num2cell(75:80)  {3};
    %   342     0        1      {5}        {5}    num2cell(75:80)  {3};
    %   343     0        1      {5}        {5}    num2cell(75:80)  {3};
    %   344     0        1      {5}        {5}    num2cell(75:80)  {3};
    %   345     0        1      {5}        {5}    num2cell(75:80)  {3};
    %   346     0        1      {5}        {5}    num2cell(75:80)  {3};
    %   347     0        1      {5}        {5}    num2cell(75:80)  {3};
    %   348     0        1      {5}        {5}    num2cell(75:80)  {3};
    %   349     0        1      {5}        {5}    num2cell(75:80)  {3};
    %   350     0        1      {5}        {5}    num2cell(75:80)  {3};
    %   351     0        1      {5}        {5}    num2cell(75:80)  {3};
    %   352     0        1      {5}        {5}    num2cell(75:80)  {3};
    %   353     0        1      {5}        {5}    num2cell(75:80)  {3};
    %   354     0        1      {5}        {5}    num2cell(75:80)  {3};
    %   355     0        1      {5}        {5}    num2cell(75:80)  {3};
     % F08 16C relaxed complexity, 50% load, (PDSCH + PDCCH + PBCH (slot 10-19))
%       356     0        1      {5}        {5}    num2cell(75:80)  {};
%       357     0        1      {5}        {5}    num2cell(75:80)  {};
%       358     0        1      {5}        {5}    num2cell(75:80)  {};
%       359     0        1      {5}        {5}    num2cell(75:80)  {};
%       360     0        1      {5}        {5}    num2cell(75:80)  {};
%       361     0        1      {5}        {5}    num2cell(75:80)  {};
%       362     0        1      {5}        {5}    num2cell(75:80)  {};
%       363     0        1      {5}        {5}    num2cell(75:80)  {};
%       364     0        1      {5}        {5}    num2cell(75:80)  {};
%       365     0        1      {5}        {5}    num2cell(75:80)  {};
%       366     0        1      {5}        {5}    num2cell(75:80)  {};
%       367     0        1      {5}        {5}    num2cell(75:80)  {};
%       368     0        1      {5}        {5}    num2cell(75:80)  {};
%       369     0        1      {5}        {5}    num2cell(75:80)  {};
%       370     0        1      {5}        {5}    num2cell(75:80)  {};
%       371     0        1      {5}        {5}    num2cell(75:80)  {};
     % F08 16C ZP CSI-RS relaxed complexity, 50% load, (PDSCH + PDCCH + PBCH (slot 10-19) + ZP CSI-RS)
%       372     0        1      {5}        {5}    num2cell(75:80)  {7};
%       373     0        1      {5}        {5}    num2cell(75:80)  {7};
%       374     0        1      {5}        {5}    num2cell(75:80)  {7};
%       375     0        1      {5}        {5}    num2cell(75:80)  {7};
%       376     0        1      {5}        {5}    num2cell(75:80)  {7};
%       377     0        1      {5}        {5}    num2cell(75:80)  {7};
%       378     0        1      {5}        {5}    num2cell(75:80)  {7};
%       379     0        1      {5}        {5}    num2cell(75:80)  {7};
%       380     0        1      {5}        {5}    num2cell(75:80)  {7};
%       381     0        1      {5}        {5}    num2cell(75:80)  {7};
%       382     0        1      {5}        {5}    num2cell(75:80)  {7};
%       383     0        1      {5}        {5}    num2cell(75:80)  {7};
%       384     0        1      {5}        {5}    num2cell(75:80)  {7};
%       385     0        1      {5}        {5}    num2cell(75:80)  {7};
%       386     0        1      {5}        {5}    num2cell(75:80)  {7};
%       387     0        1      {5}        {5}    num2cell(75:80)  {7};
    };


    %%%%%   continue config CFG using "for" %%%%%%                 
    
                             % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    % % pattern 34: F08 16C, 1 PRB / UEG, shared channel only
    % row_count = length(CFG);
    % for i = 388:403
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}          {}    num2cell(81:86)   {}};
    % end

    % % 16C, 1 PRB / UEG, PDSCH only
    % row_count = length(CFG);
    % for i = 1000:1015
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}          {}       {87}           {}};
    % end
    
    %%% 16C, avg cell, compact PDCCH, different csi-rs config, 
    % (PDSCH + PDCCH + CSI-RS + PBCH)
%     for i = 1016:1031
%         row_count = row_count + 1;
%         if mod(i,2) == 0  % csi-rs on symbol 5
%             CFG(row_count, 1:7) = { i    0        1      {4}        {6}    num2cell(75:80)  {3}};
%         else  % csi-rs on symbol 6
%             CFG(row_count, 1:7) = { i    0        1      {4}        {6}    num2cell(75:80)  {8}};
%         end
%     end
%     % (PDSCH + PDCCH + PBCH)
%     for i = 1032:1047
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0        1      {4}        {6}    num2cell(75:80)  {}};
%     end    
    % pattern 14: (PDSCH + PDCCH + PBCH + ZP CSI-RS)
    % for i = 1048:1063
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0
    %         CFG(row_count, 1:7) = { i    0        1      {4}        {6}    num2cell(75:80)  {7}};
    %     else
    %         CFG(row_count, 1:7) = { i    0        1      {4}        {6}    num2cell(75:80)  {9}};
    %     end
    % end    
    % (PDSCH + PDCCH + CSI-RS + PBCH (slot 10-19))
%     for i = 1064:1079
%         row_count = row_count + 1;
%         if mod(i,2) == 0
%             CFG(row_count, 1:7) = { i    0        1      {5}        {6}    num2cell(75:80)  {3}};
%         else
%             CFG(row_count, 1:7) = { i    0        1      {5}        {6}    num2cell(75:80)  {8}};
%         end
%     end    
%     % (PDSCH + PDCCH + PBCH (slot 10-19))
%     for i = 1080:1095
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0        1      {5}        {6}    num2cell(75:80)  {}};
%     end    
    % pattern 14: (PDSCH + PDCCH + PBCH (slot 10-19) + ZP CSI-RS)
    % for i = 1096:1111
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0
    %         CFG(row_count, 1:7) = { i    0        1      {5}        {6}    num2cell(75:80)  {7}};
    %     else
    %         CFG(row_count, 1:7) = { i    0        1      {5}        {6}    num2cell(75:80)  {9}};
    %     end
    % end
    
    %%% 16C, fewer pdsch symbol for one user, avg cell, compact PDCCH, different csi-rs config,  
    % pattern 16: (PDSCH + PDCCH + CSI-RS + PBCH)
    % for i = 1112:1127
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0
    %         CFG(row_count, 1:7) = { i    0        1      {4}        {6}    num2cell(88:93)  {3}};
    %     else
    %         CFG(row_count, 1:7) = { i    0        1      {4}        {6}    num2cell(88:93)  {8}};
    %     end
    % end
%     % (PDSCH + PDCCH + PBCH)
%     for i = 1128:1143
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0        1      {4}        {6}    num2cell(88:93)  {}};
%     end    
%     % (PDSCH + PDCCH + PBCH + ZP CSI-RS)
%     for i = 1144:1159
%         row_count = row_count + 1;
%         if mod(i,2) == 0
%             CFG(row_count, 1:7) = { i    0        1      {4}        {6}    num2cell(88:93)  {7}};
%         else
%             CFG(row_count, 1:7) = { i    0        1      {4}        {6}    num2cell(88:93)  {9}};
%         end
%     end    
    % pattern 16: (PDSCH + PDCCH + CSI-RS + PBCH (slot 10-19))
    % for i = 1160:1175
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0
    %         CFG(row_count, 1:7) = { i    0        1      {5}        {6}    num2cell(88:93)  {3}};
    %     else
    %         CFG(row_count, 1:7) = { i    0        1      {5}        {6}    num2cell(88:93)  {8}};
    %     end
    % end    
    % (PDSCH + PDCCH + PBCH (slot 10-19))
%     for i = 1176:1191
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0        1      {5}        {6}    num2cell(88:93)  {}};
%     end    
%     % (PDSCH + PDCCH + PBCH (slot 10-19) + ZP CSI-RS)
%     for i = 1192:1207
%         row_count = row_count + 1;
%         if mod(i,2) == 0
%             CFG(row_count, 1:7) = { i    0        1      {5}        {6}    num2cell(88:93)  {7}};
%         else
%             CFG(row_count, 1:7) = { i    0        1      {5}        {6}    num2cell(88:93)  {9}};
%         end
%     end
    
                                 % TC#  slotIdx   cell    ssb      pdcch      pdsch     csirs
    %%% 16C, avg cell, 2 ssb, compact PDCCH, different csi-rs config, 
    % (PDSCH + PDCCH + CSI-RS + PBCH)
%     row_count = length(CFG);
%     for i = 1208:1223
%         row_count = row_count + 1;
%         if mod(i,2) == 0  % csi-rs on symbol 5
%             CFG(row_count, 1:7) = { i    0        1      {6}        {6}    num2cell(75:80)  {3}};
%         else  % csi-rs on symbol 6
%             CFG(row_count, 1:7) = { i    0        1      {6}        {6}    num2cell(75:80)  {8}};
%         end
%     end
%     % (PDSCH + PDCCH + PBCH)
%     for i = 1224:1239
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0        1      {6}        {6}    num2cell(75:80)  {}};
%     end    
%     % (PDSCH + PDCCH + PBCH + ZP CSI-RS)
%     for i = 1240:1255
%         row_count = row_count + 1;
%         if mod(i,2) == 0
%             CFG(row_count, 1:7) = { i    0        1      {6}        {6}    num2cell(75:80)  {7}};
%         else
%             CFG(row_count, 1:7) = { i    0        1      {6}        {6}    num2cell(75:80)  {9}};
%         end
%     end    
%     % (PDSCH + PDCCH + CSI-RS + PBCH (slot 10-19))
%     for i = 1256:1271
%         row_count = row_count + 1;
%         if mod(i,2) == 0
%             CFG(row_count, 1:7) = { i    0        1      {7}        {6}    num2cell(75:80)  {3}};
%         else
%             CFG(row_count, 1:7) = { i    0        1      {7}        {6}    num2cell(75:80)  {8}};
%         end
%     end    
%     % (PDSCH + PDCCH + PBCH (slot 10-19))
%     for i = 1272:1287
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0        1      {7}        {6}    num2cell(75:80)  {}};
%     end    
%     % (PDSCH + PDCCH + PBCH (slot 10-19) + ZP CSI-RS)
%     for i = 1288:1303
%         row_count = row_count + 1;
%         if mod(i,2) == 0
%             CFG(row_count, 1:7) = { i    0        1      {7}        {6}    num2cell(75:80)  {7}};
%         else
%             CFG(row_count, 1:7) = { i    0        1      {7}        {6}    num2cell(75:80)  {9}};
%         end
%     end
    
                                  % TC#  slotIdx   cell    ssb      pdcch      pdsch     csirs
    %%% 8C, peak cell, 2 ssb, compact PDCCH, different csi-rs config, 
    % (PDSCH + PDCCH + CSI-RS + PBCH)
    % row_count = length(CFG);
    % for i = 1304:1311
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % csi-rs on symbol 5
    %         CFG(row_count, 1:7) = { i    0        1      {6}        {3}    num2cell(21:36)  {3}};
    %     else  % csi-rs on symbol 6
    %         CFG(row_count, 1:7) = { i    0        1      {6}        {3}    num2cell(21:36)  {8}};
    %     end
    % end
    % % (PDSCH + PDCCH + PBCH)
    % for i = 1312:1319
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {6}        {3}    num2cell(21:36)  {}};
    % end    
    % % (PDSCH + PDCCH + PBCH + ZP CSI-RS)
    % for i = 1320:1327
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0
    %         CFG(row_count, 1:7) = { i    0        1      {6}        {3}    num2cell(21:36)  {7}};
    %     else
    %         CFG(row_count, 1:7) = { i    0        1      {6}        {3}    num2cell(21:36)  {9}};
    %     end
    % end    
    % (PDSCH + PDCCH + CSI-RS + PBCH (slot 10-19))
    % for i = 1328:1335
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0
    %         CFG(row_count, 1:7) = { i    0        1      {7}        {3}    num2cell(21:36)  {3}};
    %     else
    %         CFG(row_count, 1:7) = { i    0        1      {7}        {3}    num2cell(21:36)  {8}};
    %     end
    % end    
    % % (PDSCH + PDCCH + PBCH (slot 10-19))
    % for i = 1336:1343
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {7}        {3}    num2cell(21:36)  {}};
    % end    
    % % (PDSCH + PDCCH + PBCH (slot 10-19) + ZP CSI-RS)
    % for i = 1344:1351
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0
    %         CFG(row_count, 1:7) = { i    0        1      {7}        {3}    num2cell(21:36)  {7}};
    %     else
    %         CFG(row_count, 1:7) = { i    0        1      {7}        {3}    num2cell(21:36)  {9}};
    %     end
    % end
    
                                 % TC#  slotIdx   cell    ssb      pdcch      pdsch     csirs
    %%% pattern 22: 16C, avg cell, 2 ssb, compact PDCCH, different csi-rs config, 17 PRBs/ UEG
    % (PDSCH + PDCCH + CSI-RS + PBCH)
%     row_count = length(CFG);
%     for i = 1352:1367
%         row_count = row_count + 1;
%         if mod(i,2) == 0  % csi-rs on symbol 5
%             CFG(row_count, 1:7) = { i    0        1      {6}        {6}    num2cell(100:105)  {3}};
%         else  % csi-rs on symbol 6
%             CFG(row_count, 1:7) = { i    0        1      {6}        {6}    num2cell(100:105)  {8}};
%         end
%     end
%     % (PDSCH + PDCCH + CSI-RS + PBCH (slot 10-19))
%     for i = 1368:1383
%         row_count = row_count + 1;
%         if mod(i,2) == 0
%             CFG(row_count, 1:7) = { i    0        1      {7}        {6}    num2cell(100:105)  {3}};
%         else
%             CFG(row_count, 1:7) = { i    0        1      {7}        {6}    num2cell(100:105)  {8}};
%         end
%     end   
    %% pattern 23: 16C, avg cell, 2 ssb, compact PDCCH, different csi-rs config, 12 PRBs/ UEG
    % (PDSCH + PDCCH + CSI-RS + PBCH)
%     row_count = length(CFG);
%     for i = 1384:1399
%         row_count = row_count + 1;
%         if mod(i,2) == 0  % csi-rs on symbol 5
%             CFG(row_count, 1:7) = { i    0        1      {6}        {6}    num2cell(106:111)  {3}};
%         else  % csi-rs on symbol 6
%             CFG(row_count, 1:7) = { i    0        1      {6}        {6}    num2cell(106:111)  {8}};
%         end
%     end
%     % (PDSCH + PDCCH + CSI-RS + PBCH (slot 10-19))
%     for i = 1400:1415
%         row_count = row_count + 1;
%         if mod(i,2) == 0
%             CFG(row_count, 1:7) = { i    0        1      {7}        {6}    num2cell(106:111)  {3}};
%         else
%             CFG(row_count, 1:7) = { i    0        1      {7}        {6}    num2cell(106:111)  {8}};
%         end
%     end
    
                             % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% Pattern 24: 16C 4 beams, ave cell, SFN%2 == 0, (no  csirs)
    % slot 0, two ssb, 
    % row_count = length(CFG);
    % for i = 1416:1431
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {6}        {6}    num2cell(112:117)    {}};
    % end
    % % slot 1, ssb at 8-11, 
    % row_count = length(CFG);
    % for i = 1432:1447
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {8}        {6}    num2cell(112:117)    {}};
    % end
    % % slot 2, ssb at 2-5, 
    % row_count = length(CFG);
    % for i = 1448:1463
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {4}        {6}    num2cell(112:117)  {}};
    % end
    % % slot 3,6,7,8,9,10,11,12,13,16,17,18,19 no ssb or trs/csirs:
    % for i = 1464:1479
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {6}    num2cell(118:123)  {}};
    % end
    %                             % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    % %% Pattern 25: 16C 4 beams, ave cell, SFN%2 == 1, (w/o ssb, w csirs)
    % % slot 0, 10, 16, 18, trs symbol 6/10, csirs symbol 12/13
    % row_count = length(CFG);
    % for i = 1480:1495
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6, csi-rs on symbol 12
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(118:123)  {10,12}};
    %     else  % trs symbol 10, csi-rs on symbol 13
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(118:123)  {11,13}};
    %     end
    % end
    % % slot 1, 11, 17, 19
    % for i = 1496:1511
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(118:123)  {10}};
    %     else  % trs symbol 10
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(118:123)  {11}};
    %     end
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 1512:1527
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {6}    num2cell(118:123)  {}};
    % end
                             % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 27: 8C 4 beams, peak cell, SFN%2 == 0, (no  csirs)
    % slot 0, two ssb, 
    % row_count = length(CFG);
    % for i = 1528:1535
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {6}        {6}    num2cell(69:74)    {}};
    % end
    % % slot 1, ssb at 8-11, 
    % row_count = length(CFG);
    % for i = 1536:1543
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {8}        {6}    num2cell(69:74)    {}};
    % end
    % % slot 2, ssb at 2-5, 
    % row_count = length(CFG);
    % for i = 1544:1551
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {4}        {6}    num2cell(69:74)  {}};
    % end
    % % slot 3,6,7,8,9,10,11,12,13,16,17,18,19 no ssb or trs/csirs:
    % for i = 1552:1559
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {6}    num2cell(124:129)  {}};
    % end
    
    %                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    % %% pattern 28: 8C 4 beams, peak cell, SFN%2 == 1, (w/o ssb, w csirs)
    % % slot 0, 10, 16, 18, trs symbol 6/10, csirs symbol 12/13
    % row_count = length(CFG);
    % for i = 1560:1567
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6, csi-rs on symbol 12
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(124:129)  {10,12}};
    %     else  % trs symbol 10, csi-rs on symbol 13
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(124:129)  {11,13}};
    %     end
    % end
    % % slot 1, 11, 17, 19
    % for i = 1568:1575
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(124:129)  {10}};
    %     else  % trs symbol 10
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(124:129)  {11}};
    %     end
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 1576:1583
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {6}    num2cell(124:129)  {}};
    % end
    
                             % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    % %% 8C 4 beams, ~ peak cell (40 PRB/ UEG), SFN%2 == 0, (no  csirs)
    % % slot 0, two ssb, 
    % row_count = length(CFG);
    % for i = 1584:1591
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {6}        {6}    num2cell(136:141)    {}};
    % end
    % % slot 1, ssb at 8-11, 
    % row_count = length(CFG);
    % for i = 1592:1599
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {8}        {6}    num2cell(136:141)    {}};
    % end
    % % slot 2, ssb at 2-5, 
    % row_count = length(CFG);
    % for i = 1600:1607
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {4}        {6}    num2cell(136:141)  {}};
    % end
    % % slot 3,6,7,8,9,10,11,12,13,16,17,18,19 no ssb or trs/csirs:
    % for i = 1608:1615
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {6}    num2cell(130:135)  {}};
    % end
    
                               % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    % %% 8C 4 beams, ~ peak cell (40 PRB/ UEG), SFN%2 == 1, (w/o ssb, w csirs)
    % % slot 0, 10, 16, 18, trs symbol 6/10, csirs symbol 12/13
    % row_count = length(CFG);
    % for i = 1616:1623
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6, csi-rs on symbol 12
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(130:135)   {10,12}};
    %     else  % trs symbol 10, csi-rs on symbol 13
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(130:135)   {11,13}};
    %     end
    % end
    % % slot 1, 11, 17, 19
    % for i = 1624:1631
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(130:135)   {10}};
    %     else  % trs symbol 10
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(130:135)   {11}};
    %     end
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 1632:1639
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {6}    num2cell(130:135)   {}};
    % end
    
                                % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
%     %% pattern 31: 16C 4 beams, ave cell, 1UEG, SFN%2 == 1, (w/o ssb, w csirs), 
%     % slot 0, 10, 16, 18, trs symbol 6/10, csirs symbol 12/13
%     row_count = length(CFG);
%     for i = 1640:1655
%         row_count = row_count + 1;
%         if mod(i,2) == 0  % trs symbol 6, csi-rs on symbol 12
%             CFG(row_count, 1:7) = { i    0        1      {}        {6}       {142}      {10,12}};
%         else  % trs symbol 10, csi-rs on symbol 13
%             CFG(row_count, 1:7) = { i    0        1      {}        {6}       {142}      {11,13}};
%         end
%     end
%     % slot 1, 11, 17, 19
%     for i = 1656:1671
%         row_count = row_count + 1;
%         if mod(i,2) == 0  % trs symbol 6
%             CFG(row_count, 1:7) = { i    0        1      {}        {6}       {142}       {10}};
%         else  % trs symbol 10
%             CFG(row_count, 1:7) = { i    0        1      {}        {6}       {142}       {11}};
%         end
%     end
%     % other slots, no ssb, no csi-rs
%     for i = 1672:1687
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0        1      {}         {6}          {142}        {}};
%     end
%     
%                                % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
%     %% pattern 32: 8C 4 beams, peak cell 1 UEG, SFN%2 == 1, (w/o ssb, w csirs)
%     % slot 0, 10, 16, 18, trs symbol 6/10, csirs symbol 12/13
%     row_count = length(CFG);
%     for i = 1688:1695
%         row_count = row_count + 1;
%         if mod(i,2) == 0  % trs symbol 6, csi-rs on symbol 12
%             CFG(row_count, 1:7) = { i    0        1      {}        {6}        {143}       {10,12}};
%         else  % trs symbol 10, csi-rs on symbol 13
%             CFG(row_count, 1:7) = { i    0        1      {}        {6}        {143}       {11,13}};
%         end
%     end
%     % slot 1, 11, 17, 19
%     for i = 1696:1703
%         row_count = row_count + 1;
%         if mod(i,2) == 0  % trs symbol 6
%             CFG(row_count, 1:7) = { i    0        1      {}        {6}        {143}           {10}};
%         else  % trs symbol 10
%             CFG(row_count, 1:7) = { i    0        1      {}        {6}        {143}           {11}};
%         end
%     end
%     % other slots, no ssb, no csi-rs
%     for i = 1704:1711
%         row_count = row_count + 1;
%         CFG(row_count, 1:7) = { i    0        1      {}         {6}            {143}          {}};
%     end
    
    % %% pattern 33: 4 frames, 16C, ave cell
    % % frame 0: repeat pattern 24d
    % % frame 1: repeat pattern 25g
    % % frame 2: repeat pattern 24d
    % % frame 3: only have TRS:
    % % slot 0,1,10,11,16,17,18,19 trs symbol {6,10} / {5,9}, no csirs
    % row_count = length(CFG);
    % for i = 1712:1727
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol {6,10}, no csirs
    %         CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(118:123)  {10,11}};
    %     else  % trs symbol {5,9}, no csirs
    %         CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(118:123)  {14,15}};
    %     end
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 1728:1743
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {7,8}     num2cell(118:123)  {}};
    % end
    
    %                          % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    % %% Pattern 24a: 17PRBs/UEG, 16C 4 beams, ave cell, SFN%2 == 0, (no  csirs)
    % % slot 0, two ssb, 
    % row_count = length(CFG);
    % for i = 1744:1759
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {6}        {6}    num2cell(100:105)    {}};
    % end
    % % slot 1, ssb at 8-11, 
    % row_count = length(CFG);
    % for i = 1760:1775
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {8}        {6}    num2cell(100:105)    {}};
    % end
    % % slot 2, ssb at 2-5, 
    % row_count = length(CFG);
    % for i = 1776:1791
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {4}        {6}    num2cell(100:105)  {}};
    % end
    % % slot 3,6,7,8,9,10,11,12,13,16,17,18,19 no ssb or trs/csirs:
    % for i = 1792:1807
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {6}    num2cell(144:149)  {}};
    % end
    
    % %% Pattern 25a: 17PRBs/UEG, 16C 4 beams, ave cell, SFN%2 == 1, (w/o ssb, w csirs)
    % % slot 0, 10, 16, 18, trs symbol 6/10, csirs symbol 12/13
    % row_count = length(CFG);
    % for i = 1808:1823
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6, csi-rs on symbol 12
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(144:149)  {10,12}};
    %     else  % trs symbol 10, csi-rs on symbol 13
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(144:149)  {11,13}};
    %     end
    % end
    % % slot 1, 11, 17, 19
    % for i = 1824:1839
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(144:149)  {10}};
    %     else  % trs symbol 10
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(144:149)  {11}};
    %     end
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 1840:1855
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {6}    num2cell(144:149)  {}};
    % end
    
    % %% Pattern 25b: 17PRBs/UEG, 16C 4 beams, ave cell, SFN%2 == 1, (w/o ssb, w csirs), NZP csi-rs only on symbol 13
    % % slot 0, 10, 16, 18, trs symbol 6/10, csirs symbol 13
    % row_count = length(CFG);
    % for i = 1856:1871
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6, csi-rs on symbol 13
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(144:149)  {10,13}};
    %     else  % trs symbol 10, csi-rs on symbol 13
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(144:149)  {11,13}};
    %     end
    % end
    % % slot 1, 11, 17, 19
    % for i = 1872:1887
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(144:149)  {10}};
    %     else  % trs symbol 10
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(144:149)  {11}};
    %     end
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 1888:1903
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {6}    num2cell(144:149)  {}};
    % end
    
    %                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    % %% pattern 28b: 8C 4 beams, peak cell, SFN%2 == 1, (w/o ssb, w csirs) NZP csi-rs only on symbol 13
    % % slot 0, 10, 16, 18, trs symbol 6/10, NZP csi-rs only on symbol 13
    % row_count = length(CFG);
    % for i = 1904:1911
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6, csi-rs on symbol 13
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(124:129)  {10,13}};
    %     else  % trs symbol 10, csi-rs on symbol 13
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(124:129)  {11,13}};
    %     end
    % end
    % % slot 1, 11, 17, 19
    % for i = 1912:1919
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(124:129)  {10}};
    %     else  % trs symbol 10
    %         CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(124:129)  {11}};
    %     end
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 1920:1927
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {6}    num2cell(124:129)  {}};
    % end
    
    % %% Pattern 25c: all the cells use 2 TRS symbols (6 and 10), 17PRBs/UEG, 16C 4 beams, ave cell, SFN%2 == 1, (w/o ssb, w csirs), NZP csi-rs only on symbol 13
    % % slot 0, 10, 16, 18, trs symbol 6 and 10, csirs symbol 13
    % row_count = length(CFG);
    % for i = 1928:1943
    %     row_count = row_count + 1;   % trs symbol 6 and 10, csi-rs on symbol 13
    %     CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(144:149)  {10,11,13}};
    % end
    % % slot 1, 11, 17, 19
    % for i = 1944:1959
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(144:149)  {10,11}};
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 1960:1975
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {6}    num2cell(144:149)  {}};
    % end
    
    %                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    % %% pattern 28c: all the cells use 2 TRS symbols (6 and 10), 8C 4 beams, peak cell, SFN%2 == 1, (w/o ssb, w csirs) NZP csi-rs only on symbol 13
    % % slot 0, 10, 16, 18, trs symbol 6 and 10, NZP csi-rs only on symbol 13
    % row_count = length(CFG);
    % for i = 1976:1983
    %     row_count = row_count + 1;  % trs symbol 6 and 10, csi-rs on symbol 13
    %     CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(124:129)  {10,11,13}};
    % end
    % % slot 1, 11, 17, 19
    % for i = 1984:1991
    %     row_count = row_count + 1;
    %      % trs symbol 6 and 10
    %     CFG(row_count, 1:7) = { i    0        1      {}        {6}    num2cell(124:129)  {10,11}};
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 1992 :1999
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {6}    num2cell(124:129)  {}};
    % end
    
    %                          % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    % %% Pattern 24d: 16C 4 beams, ave cell, SFN%2 == 0, (no  csirs)
    % % slot 0, two ssb, 
    % row_count = length(CFG);
    % for i = 2000:2015
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {6}        {7,8}    num2cell(112:117)    {}};
    % end
    % % slot 1, ssb at 8-11, 
    % row_count = length(CFG);
    % for i = 2016:2031
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {8}        {7,8}    num2cell(112:117)    {}};
    % end
    % % slot 2, ssb at 2-5, 
    % row_count = length(CFG);
    % for i = 2032:2047
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {4}        {7,8}    num2cell(112:117)  {}};
    % end
    % % slot 3,6,7,8,9,10,11,12,13,16,17,18,19 no ssb or trs/csirs:
    % for i = 2048:2063
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(118:123)  {}};
    % end

    %                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    % %% Pattern 25d: 6+6 DCI, 16C 4 beams, ave cell, SFN%2 == 1, (w/o ssb, w csirs)
    % % slot 0, 10, 16, 18, trs symbol 6/10, csirs symbol 12/13
    % row_count = length(CFG);
    % for i = 2064:2079
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6, csi-rs on symbol 12
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(118:123)  {10,12}};
    %     else  % trs symbol 10, csi-rs on symbol 13
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(118:123)  {11,13}};
    %     end
    % end
    % % slot 1, 11, 17, 19
    % for i = 2080:2095
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(118:123)  {10}};
    %     else  % trs symbol 10
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(118:123)  {11}};
    %     end
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 2096:2111
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(118:123)  {}};
    % end
    

    
    %                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    % %% pattern 28d: 6+6 DCI, 8C 4 beams, peak cell, SFN%2 == 1, (w/o ssb, w csirs)
    % % slot 0, 10, 16, 18, trs symbol 6/10, csirs symbol 12/13
    % row_count = length(CFG);
    % for i = 2144:2151
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6, csi-rs on symbol 12
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(124:129)  {10,12}};
    %     else  % trs symbol 10, csi-rs on symbol 13
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(124:129)  {11,13}};
    %     end
    % end
    % % slot 1, 11, 17, 19
    % for i = 2152:2159
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(124:129)  {10}};
    %     else  % trs symbol 10
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(124:129)  {11}};
    %     end
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 2160:2167
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(124:129)  {}};
    % end
    
    % %% Pattern 25e: 16C, ave cell, 17 PRBs, PDCCH 6 DL DCI + 6 UL DCI, CSI-RS 6+10, 13
    % % slot 0, 10, 16, 18, trs symbol 6 and 10, csirs symbol 13
    % row_count = length(CFG);
    % for i = 2168:2183
    %     row_count = row_count + 1;   % trs symbol 6 and 10, csi-rs on symbol 13
    %     CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(144:149)  {10,11,13}};
    % end
    % % slot 1, 11, 17, 19
    % for i = 2184:2199
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(144:149)  {10,11}};
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 2200:2215
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(144:149)  {}};
    % end
    
    %                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    % %% Pattern 25f: 16C ave, 22 PRBs, PDCCH 6 DL DCI + 6 UL DCI, CSI-RS 6+10, 13
    % % slot 0, 10, 16, 18, trs symbol 6 and 10, csirs symbol 13
    % row_count = length(CFG);
    % for i = 2216:2231
    %     row_count = row_count + 1;
    %     % trs symbol 6 and 10, csi-rs on symbol 13
    %     CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(118:123)  {10,11,13}};
    % end
    % % slot 1, 11, 17, 19
    % for i = 2232:2247
    %     row_count = row_count + 1;  % trs symbol 6 and 10
    %     CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(118:123)  {10,11}};
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 2248:2263
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(118:123)  {}};
    % end
    
    %                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    % %% pattern 28f: 8C peak, peak cell, 45PRBS, PDCCH 6 DL DCI + 6 UL DCI, CSI-RS 6+10, 13
    % % slot 0, 10, 16, 18, trs symbol 6 and 10, csirs symbol 13
    % row_count = length(CFG);
    % for i = 2264:2271
    %     row_count = row_count + 1; % trs symbol 6 and 10, csi-rs on symbol 13
    %     CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(124:129)  {10,11,13}};
    % end
    % % slot 1, 11, 17, 19
    % for i = 2272:2279
    %     row_count = row_count + 1; % trs symbol 6 and 10
    %     CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(124:129)  {10,11}};
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 2280:2287
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(124:129)  {}};
    % end
    
    %                           % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    % %% Pattern 25g: TRS {6,10}/{5,9}, 6+6 DCI, 16C 4 beams, ave cell, SFN%2 == 1, (w/o ssb, w csirs)
    % % slot 0, 10, 16, 18, trs symbol {6,10}/{5,9}, csirs symbol 12/13
    % row_count = length(CFG);
    % for i = 2288:2303
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(118:123)  {10,11,12}};
    %     else  % trs symbol 5,9, csi-rs on symbol 13
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(118:123)  {14,15,13}};
    %     end
    % end
    % % slot 1, 11, 17, 19
    % for i = 2304:2319
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6,10
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(118:123)  {10,11}};
    %     else  % trs symbol 5,9
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(118:123)  {14,15}};
    %     end
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 2320:2335
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(118:123)  {}};
    % end
    
    
     
    
    % %% Pattern 25h: 16C, ave cell, 17 PRBs, PDCCH 6 DL DCI + 6 UL DCI, TRS {6,10}/{5,9},
    % % slot 0, 10, 16, 18, trs symbol 6 and 10, csirs symbol 13
    % row_count = length(CFG);
    % for i = 2360:2375
    %     row_count = row_count + 1;  
    %     if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(144:149)  {10,11,12}};
    %     else  % trs symbol 5,9, csi-rs on symbol 13
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(144:149)  {14,15,13}};
    %     end
    % end
    % % slot 1, 11, 17, 19
    % for i = 2376:2391
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6,10
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(144:149)  {10,11}};
    %     else  % trs symbol 5,9
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(144:149)  {14,15}};
    %     end
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 2392:2407
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(144:149)  {}};
    % end    
    
    
    % %% pattern 37: 16C ave cells, 4 frames, 52 TRS, 132 NZP CSIRS
    % % frame 0: pattern 24d
    % % frame 1: TRS {6,10}/{5,9}, 6+6 DCI, 16C 4 beams, ave cell, SFN%2 == 1, (w/o ssb, w csirs)
    % % slot 0, 10, 16, 18, trs symbol {6,10}/{5,9}, csirs symbol 12/13
    % row_count = length(CFG);
    % for i = 2512:2527
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(118:123)  {16,17,20}};
    %     else  % trs symbol 5,9, csi-rs on symbol 13
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(118:123)  {18,19,21}};
    %     end
    % end
    % % slot 1, 11, 17, 19
    % for i = 2528:2543
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6,10
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(118:123)  {16,17}};
    %     else  % trs symbol 5,9
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(118:123)  {18,19}};
    %     end
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 2544:2559
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(118:123)  {}};
    % end
    % % frame 2: pattern 24d
    % % frame 3: only have TRS:
    % % slot 0,1,10,11,16,17,18,19 trs symbol {6,10} / {5,9}, no csirs
    % row_count = length(CFG);
    % for i = 2560:2575
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol {6,10}, no csirs
    %         CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(118:123)  {16,17}};
    %     else  % trs symbol {5,9}, no csirs
    %         CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(118:123)  {18,19}};
    %     end
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 2576:2591
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {7,8}     num2cell(118:123)  {}};
    % end
    
    %                              % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    % %% pattern 38: 16C ave cells, disjoint PDSCH and CSI-RS 
    % % frame 0: pattern 24d
    % % frame 1: 
    % % slot 0, 10, 16, 18, PDSCH 0-77, symbols 1-11 TRS {6,10}/{5,9} on 78-129, CSIRS 12/13, 0-272
    % row_count = length(CFG);
    % for i = 2592:2607
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(150:155)  {22,23,12}};
    %     else  % trs symbol 5,9, csi-rs on symbol 13
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(150:155)  {24,25,13}};
    %     end
    % end
    % % slot 1, 11, 17, 19 : PDSCH 0-77, symbols 1-11 TRS {6,10}/{5,9} on 78-129
    % for i = 2608:2623
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6,10
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(150:155)  {22,23}};
    %     else  % trs symbol 5,9
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(150:155)  {24,25}};
    %     end
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 2624:2639
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(118:123)  {}};
    % end
    % % frame 2: pattern 24d
    % % frame 3: only have TRS:
    % % slot 0,1,10,11,16,17,18,19: PDSCH 0-77, symbols 1-11 TRS {6,10}/{5,9} on 78-129
    % row_count = length(CFG);
    % for i = 2640:2655
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol {6,10}, no csirs
    %         CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(150:155)  {22,23}};
    %     else  % trs symbol {5,9}, no csirs
    %         CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(150:155)  {24,25}};
    %     end
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 2656:2671
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {7,8}     num2cell(118:123)  {}};
    % end
    
    
    % %% pattern 38a: 16C ave cells, disjoint PDSCH and CSI-RS (fewer NZP CSI-RS)
    % % frame 0: pattern 24d
    % % frame 1: 
    % % slot 0, 10, 16, 18, PDSCH 0-77, symbols 1-11 TRS {6,10}/{5,9} on 78-129, CSIRS 12/13, 0-272
    % row_count = length(CFG);
    % for i = 2672:2687
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(150:155)  {22,23,26}};
    %     else  % trs symbol 5,9, csi-rs on symbol 13
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(150:155)  {24,25,26}};
    %     end
    % end
    % % slot 1, 11, 17, 19 : PDSCH 0-77, symbols 1-11 TRS {6,10}/{5,9} on 78-129
    % for i = 2688:2703
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6,10
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(150:155)  {22,23}};
    %     else  % trs symbol 5,9
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(150:155)  {24,25}};
    %     end
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 2704:2719
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(118:123)  {}};
    % end
    % % frame 2: pattern 24d
    % % frame 3: re-use frame 3 of pattern 38
    % % only have TRS:
    % % slot 0,1,10,11,16,17,18,19: PDSCH 0-77, symbols 1-11 TRS {6,10}/{5,9} on 78-129

    
    %                              % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    % %% pattern 27d: 6+6 DCI, 12C 4 beams, peak cell, SFN%2 == 0, (no  csirs)
    % % slot 0, two ssb, 
    % row_count = length(CFG);
    % for i = 2720:2731
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {6}        {7,8}    num2cell(69:74)    {}};
    % end
    % % slot 1, ssb at 8-11, 
    % row_count = length(CFG);
    % for i = 2732:2743
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {8}        {7,8}    num2cell(69:74)    {}};
    % end
    % % slot 2, ssb at 2-5, 
    % row_count = length(CFG);
    % for i = 2744:2755
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {4}        {7,8}    num2cell(69:74)  {}};
    % end
    % % slot 3,6,7,8,9,10,11,12,13,16,17,18,19 no ssb or trs/csirs:
    % for i = 2756:2767
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(124:129)  {}};
    % end
    
    
    %                           % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    % %% pattern 28g: TRS {6,10}/{5,9}, 6+6 DCI, 12C 4 beams, peak cell, SFN%2 == 1, (w/o ssb, w csirs)
    % % slot 0, 10, 16, 18, trs symbol {6,10}/{5,9}, csirs symbol 12/13
    % row_count = length(CFG);
    % for i = 2768:2779
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(124:129)  {10,11,12}};
    %     else  % trs symbol 5,9, csi-rs on symbol 13
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(124:129)  {14,15,13}};
    %     end
    % end
    % % slot 1, 11, 17, 19
    % for i = 2780:2791
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol 6,10
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(124:129)  {10,11}};
    %     else  % trs symbol 5,9
    %         CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(124:129)  {14,15}};
    %     end
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 2792:2803
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(124:129)  {}};
    % end
    
    % %% pattern 35: 4 frames, 12C, peak cell
    % % frame 0: repeat pattern 27d
    % % frame 1: repeat pattern 28g
    % % frame 2: repeat pattern 27d
    % % frame 3: only have TRS:
    % % slot 0,1,10,11,16,17,18,19 trs symbol {6,10} / {5,9}, no csirs
    % row_count = length(CFG);
    % for i = 2804:2815
    %     row_count = row_count + 1;
    %     if mod(i,2) == 0  % trs symbol {6,10}, no csirs
    %         CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(124:129)  {10,11}};
    %     else  % trs symbol {5,9}, no csirs
    %         CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(124:129)  {14,15}};
    %     end
    % end
    % % other slots, no ssb, no csi-rs
    % for i = 2816:2827
    %     row_count = row_count + 1;
    %     CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(124:129)  {}};
    % end
    
    
                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 41, 42: 4 frames, 12C, peak cell, OTA
    % frame 0: 6+6 DCI, 12C 4 beams, peak cell, SFN%2 == 0, (no  csirs)                     
    % slot 0, two ssb, 
    row_count = length(CFG);
    for i = 2828:2839
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {6}        {7,8}    num2cell(158:163)    {}};
    end
    % slot 1, ssb at 8-11, 
    row_count = length(CFG);
    for i = 2840:2851
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {8}        {7,8}    num2cell(158:163)    {}};
    end
    % slot 2, ssb at 2-5, 
    row_count = length(CFG);
    for i = 2852:2863
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {4}        {7,8}    num2cell(158:163)  {}};
    end
    % slot 3,6,7,8,9,10,11,12,13,16,17,18,19 no ssb or trs/csirs:
    for i = 2864:2875
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(164:169)  {}};
    end
    %%%%%%%%%%%%
    %%%% frame 1: TRS {6,10}/{5,9}, 6+6 DCI, 12C 4 beams, peak cell, SFN%2 == 1, (w/o ssb, w csirs)
    % slot 0, 10, 16, 18, trs symbol {6,10}/{5,9}, csirs symbol 12/13
    row_count = length(CFG);
    for i = 2876:2887
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {10,11,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {14,15,13}};
        end
    end
    % slot 1, 11, 17, 19
    for i = 2888:2899
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {10,11}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {14,15}};
        end
    end
    % other slots, no ssb, no csi-rs
    for i = 2900:2911
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(164:169)  {}};
    end
    %%%%%%%%%%%%
    %%%% frame 2: repeat frame 0
    %%%%%%%%%%%%
    %%%% frame 3: only have TRS:
    % slot 0,1,10,11,16,17,18,19 trs symbol {6,10} / {5,9}, no csirs
    row_count = length(CFG);
    for i = 2912:2923
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol {6,10}, no csirs
            CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(164:169)  {10,11}};
        else  % trs symbol {5,9}, no csirs
            CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(164:169)  {14,15}};
        end
    end
    % other slots, no ssb, no csi-rs
    for i = 2924:2935
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(164:169)  {}};
    end
    
    
    %% pattern 42: DL is the same as pattern 41
    
    %% pattern 43, 45: 4 frames, 16C, ave cell, OTA
    %%% frame 0: 16C 4 beams, ave cell, SFN%2 == 0, (no  csirs)
    % slot 0, two ssb, 
    row_count = length(CFG);
    for i = 2936:2951
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {6}        {7,8}    num2cell(170:175)    {}};
    end
    % slot 1, ssb at 8-11, 
    row_count = length(CFG);
    for i = 2952:2967
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {8}        {7,8}    num2cell(170:175)    {}};
    end
    % slot 2, ssb at 2-5, 
    row_count = length(CFG);
    for i = 2968:2983
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {4}        {7,8}    num2cell(170:175)  {}};
    end
    % slot 3,6,7,8,9,10,11,12,13,16,17,18,19 no ssb or trs/csirs:
    for i = 2984:2999
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(176:181)  {}};
    end
    %%%%%%
    %%%% frame 1: TRS {6,10}/{5,9}, 6+6 DCI, 16C 4 beams, ave cell, SFN%2 == 1, (w/o ssb, w csirs)
    % slot 0, 10, 16, 18, trs symbol {6,10}/{5,9}, csirs symbol 12/13
    row_count = length(CFG);
    for i = 3000:3015
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(176:181)  {10,11,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(176:181)  {14,15,13}};
        end
    end
    % slot 1, 11, 17, 19
    for i = 3016:3031
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(176:181)  {10,11}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(176:181)  {14,15}};
        end
    end
    % other slots, no ssb, no csi-rs
    for i = 3032:3047
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(176:181)  {}};
    end
    
    %%%%%%%%%%%%%%%%%%%  
    % frame 2: repeat frame 0
    %%%%%%%%%%%%%%%%%%%     
    % frame 3: only have TRS:
    % slot 0,1,10,11,16,17,18,19 trs symbol {6,10} / {5,9}, no csirs
    row_count = length(CFG);
    for i = 3048:3063
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol {6,10}, no csirs
            CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(176:181)  {10,11}};
        else  % trs symbol {5,9}, no csirs
            CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(176:181)  {14,15}};
        end
    end
    % other slots, no ssb, no csi-rs
    for i = 3064:3079
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}     num2cell(176:181)  {}};
    end
    
                                 % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 44: 16C ave cells, disjoint PDSCH and CSI-RS 
    % frame 0: repeat pattern 43 frame 0
    %%%%%
    %%%% frame 1: 
    % slot 0, 10, 16, 18, PDSCH 0-77, symbols 1-11 TRS {6,10}/{5,9} on 78-129, CSIRS 12/13, 0-272
    row_count = length(CFG);
    for i = 3080:3095
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {22,23,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {24,25,13}};
        end
    end
    % slot 1, 11, 17, 19 : PDSCH 0-77, symbols 1-11 TRS {6,10}/{5,9} on 78-129
    for i = 3096:3111
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {22,23}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {24,25}};
        end
    end
    % other slots, no ssb, no csi-rs
    for i = 3112:3127
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(176:181)  {}};
    end
    %%%%%%
    % frame 2: repeat pattern 43 frame 0
    %%%%%%
    % frame 3: only have TRS:
    % slot 0,1,10,11,16,17,18,19: PDSCH 0-77, symbols 1-11 TRS {6,10}/{5,9} on 78-129
    row_count = length(CFG);
    for i = 3128:3143
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol {6,10}, no csirs
            CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(182:187)  {22,23}};
        else  % trs symbol {5,9}, no csirs
            CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(182:187)  {24,25}};
        end
    end
    % other slots, no ssb, no csi-rs
    for i = 3144:3159
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}     num2cell(176:181)  {}};
    end
    %% end pattern 44
    
    %% pattern 45: DL is the same as 43
    
    
    
    
                                % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 46: 7 beams, 100 MHz (273 PRBs), 12C, peak cell, OTA
    %%%%% frame 0:         
    % slot 0,1,2 two ssb, idx {0, 1},{2,3},{4,5} 
    row_count = length(CFG);    
    for i = 3160:3195
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {6}        {7,8}    num2cell(158:163)    {}};
        SsbBeamIdxMap(i) = [0, 1] + 2 * floor((i-3160)/12);
    end
    % slot 3, ssb at 2-5, idx 6
    for i = 3196:3207
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {4}        {7,8}     {}     {}};
        SsbBeamIdxMap(i) = 6;
    end
    % slot {6, 7}, 9, 11, 17 TRS, idx 0, 1,2,3
    for i = 3208:3255
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {10,11}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {14,15}};
        end
        TrsBeamIdxMap(i) = floor((i-(3160+ 12*4))/12);
    end 
    % slot 8,10,16, TRS + CSIRS, idx 4-7, 8-11, 12-15
    for i = 3256:3291
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {10,11,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {14,15,13}};
        end
        TrsBeamIdxMap(i) = 1 + floor((i-(3160+ 12*8))/12);
        CsiBeamIdxMap(i) = [4, 5, 6, 7]  + 4 * floor((i-(3160+ 12*8))/12);
    end
    % 12, 18, 19, no ssb or trs/csirs:
    for i = 3292:3303
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(164:169)  {}};
    end
    % 13, s slot, pdcch only: see last set
    %%%%% frame 1: 
    % slot 6, 8, 10 TRS + CSIRS: idx 16-19, 20-23, 24-27
    for i = 3304:3339
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {10,11,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {14,15,13}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-(3160+ 12*12))/12);
        CsiBeamIdxMap(i) = [16, 17, 18, 19] + 4 * floor((i-(3160+ 12*12))/12);
    end 
    % slot 7, 9, 11 TRS: idx 4, 5, 6
    for i = 3340:3375
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {10,11}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {14,15}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-(3160+ 12*15))/12);
    end
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 2:
    % slot 0, 1, 2 SSB 2 blocks: same as frame 0 slots 0, 1, 2
    % slot 3 SSB 1 block: same as frame 0 slot 3
    % slot 6 TRS and CSIRS, idx 0-4
    for i = 3376:3387
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {10,11,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {14,15,13}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [0, 1, 2, 3];
    end 
    % slot 7, {8,9},{10,11},{16,17} TRS: same as frame 0 slots 7, 9, 11, 17 TRS idx 0, 1,2, 3
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 3:
    % slot {6,7},{8,9},{10,11} same as frame 1 slots 7, 9, 11 TRS: idx 4, 5, 6
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% slots 13, 23, 33, 53, 63, 73 pdcch only
    for i = 3388:3399
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    {}  {}};
    end
    %% end pattern 46
    
                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 47: 7 beams, 100 MHz (273 PRBs), 16C, ave cell, OTA, disjoint PDSCH and CSIRS
    %%%%% frame 0:         
    % slot 0,1,2 two ssb, idx {0, 1},{2,3},{4,5} 
    row_count = length(CFG);    
    for i = 3400:3447
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {6}        {7,8}    num2cell(170:175)    {}};
        SsbBeamIdxMap(i) = [0, 1] + 2 * floor((i-3400)/16);
    end
    % slot 3, ssb at 2-5, idx 6
    for i = 3448:3463
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {4}        {7,8}     {}     {}};
        SsbBeamIdxMap(i) = 6;
    end
    % slot {6, 7}, 9, 11, 17 TRS, idx 0, 1,2,3
    for i = 3464:3527
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {22,23}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {24,25}};
        end
        TrsBeamIdxMap(i) = floor((i-3464)/16);
    end 
    % slot 8,10,16, TRS + CSIRS, idx 4-7, 8-11, 12-15
    for i = 3528:3575
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {22,23,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {24,25,13}};
        end
        TrsBeamIdxMap(i) = 1 + floor((i-3528)/16);
        CsiBeamIdxMap(i) = [4, 5, 6, 7]  + 4 * floor((i-3528)/16);
    end
    % 12, 18, 19, no ssb or trs/csirs:
    for i = 3576:3591
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(176:181)  {}};
    end
    % 13, s slot, pdcch only: see last set
    %%%%% frame 1: 
    % slot 6, 8, 10 TRS + CSIRS: idx 16-19, 20-23, 24-27
    for i = 3592:3639
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {22,23,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {24,25,13}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-3592)/16);
        CsiBeamIdxMap(i) = [16, 17, 18, 19] + 4 * floor((i-3592)/16);
    end 
    % slot 7, 9, 11 TRS: idx 4, 5, 6
    for i = 3640:3687
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {22,23}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {24,25}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-3640)/16);
    end
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 2:
    % slot 0, 1, 2 SSB 2 blocks: same as frame 0 slots 0, 1, 2
    % slot 3 SSB 1 block: same as frame 0 slot 3
    % slot 6 TRS and CSIRS, idx 0-4
    for i = 3688:3703
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {22,23,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {24,25,13}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [0, 1, 2, 3];
    end 
    % slot 7, {8,9},{10,11},{16,17} TRS: same as frame 0 slots 7, 9, 11, 17 TRS idx 0, 1,2, 3
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 3:
    % slot {6,7},{8,9},{10,11} same as frame 1 slots 7, 9, 11 TRS: idx 4, 5, 6
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% slots 13, 23, 33, 53, 63, 73 pdcch only
    for i = 3704:3719
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    {}  {}};
    end
    %% end pattern 47
    
    
                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 48: 7 beams, 40 MHz (106 PRBs), 16C, peak cell, OTA
    %%%%% frame 0:         
    % slot 0,1,2 two ssb, idx {0, 1},{2,3},{4,5} 
    row_count = length(CFG);    
    for i = 3720:3767
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        4      {10}        {11,12}    num2cell(188:193)    {}};
        SsbBeamIdxMap(i) = [0, 1] + 2 * floor((i-3720)/16);
    end
    % slot 3, ssb at 2-5, idx 6
    for i = 3768:3783
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        4      {12}        {11,12}     {}     {}};
        SsbBeamIdxMap(i) = 6;
    end
    % slot {6, 7}, 9, 11, 17 TRS, idx 0, 1,2,3
    for i = 3784:3847
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        4      {}       {11,12}    num2cell(194:199)  {27,28}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        4      {}       {11,12}    num2cell(194:199)  {29,30}};
        end
        TrsBeamIdxMap(i) = floor((i-(3720+ 16*4))/16);
    end 
    % slot 8,10,16, TRS + CSIRS, idx 4-7, 8-11, 12-15
    for i = 3848:3895
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        4      {}       {11,12}    num2cell(194:199)  {27,28,31}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        4      {}       {11,12}    num2cell(194:199)  {29,30,32}};
        end
        TrsBeamIdxMap(i) = 1 + floor((i-3848)/16);
        CsiBeamIdxMap(i) = [4, 5, 6, 7]  + 4 * floor((i-3848)/16);
    end
    % 12, 18, 19, no ssb or trs/csirs:
    for i = 3896:3911
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        4      {}         {11,12}    num2cell(194:199)  {}};
    end
    % 13, s slot, pdcch only: see last set
    %%%%% frame 1: 
    % slot 6, 8, 10 TRS + CSIRS: idx 16-19, 20-23, 24-27
    for i = 3912:3959
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        4      {}       {11,12}    num2cell(194:199)  {27,28,31}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        4      {}       {11,12}    num2cell(194:199)  {29,30,32}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-3912)/16);
        CsiBeamIdxMap(i) = [16, 17, 18, 19] + 4 * floor((i-3912)/16);
    end 
    % slot 7, 9, 11 TRS: idx 4, 5, 6
    for i = 3960:4007
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        4      {}       {11,12}    num2cell(194:199)  {27,28}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        4      {}       {11,12}    num2cell(194:199)  {29,30}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-3960)/16);
    end
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 2:
    % slot 0, 1, 2 SSB 2 blocks: same as frame 0 slots 0, 1, 2
    % slot 3 SSB 1 block: same as frame 0 slot 3
    % slot 6 TRS and CSIRS, idx 0-4
    for i = 4008:4023
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        4      {}       {11,12}    num2cell(194:199)  {27,28,31}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        4      {}       {11,12}    num2cell(194:199)  {29,30,32}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [0, 1, 2, 3];
    end 
    % slot 7, {8,9},{10,11},{16,17} TRS: same as frame 0 slots 7, 9, 11, 17 TRS idx 0, 1,2, 3
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 3:
    % slot {6,7},{8,9},{10,11} same as frame 1 slots 7, 9, 11 TRS: idx 4, 5, 6
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% slots 13, 23, 33, 53, 63, 73 pdcch only
    for i = 4024:4039
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        4      {}         {11,12}    {}  {}};
    end
    % end pattern 48
    
                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 49: 7 beams, 40 MHz (106 PRBs), 20C, ave cell, OTA, disjoint PDSCH and CSIRS
    %%%%% frame 0:         
    % slot 0,1,2 two ssb, idx {0, 1},{2,3},{4,5} 
    row_count = length(CFG);    
    for i = 4040:4099
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        4      {10}        {13,14}    num2cell(206:211)    {}};
        SsbBeamIdxMap(i) = [0, 1] + 2 * floor((i-4040)/20);
    end
    % slot 3, ssb at 2-5, idx 6
    for i = 4100:4119
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        4      {12}        {13,14}     {}     {}};
        SsbBeamIdxMap(i) = 6;
    end
    % slot {6, 7}, 9, 11, 17 TRS, idx 0, 1,2,3
    for i = 4120:4199
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        4      {}       {13,14}    num2cell(200:205)  {33,34}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        4      {}       {13,14}    num2cell(200:205)  {35,36}};
        end
        TrsBeamIdxMap(i) = floor((i-4120)/20);
    end 
    % slot 8,10,16, TRS + CSIRS, idx 4-7, 8-11, 12-15
    for i = 4200:4259
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        4      {}       {13,14}    num2cell(200:205)  {33,34,31}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        4      {}       {13,14}    num2cell(200:205)  {35,36,32}};
        end
        TrsBeamIdxMap(i) = 1 + floor((i-4200)/20);
        CsiBeamIdxMap(i) = [4, 5, 6, 7]  + 4 * floor((i-4200)/20);
    end
    % 12, 18, 19, no ssb or trs/csirs:
    for i = 4260:4279
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        4      {}         {13,14}    num2cell(212:217)  {}};
    end
    % 13, s slot, pdcch only: see last set
    %%%%% frame 1: 
    % slot 6, 8, 10 TRS + CSIRS: idx 16-19, 20-23, 24-27
    for i = 4280:4339
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        4      {}       {13,14}    num2cell(200:205)  {33,34,31}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        4      {}       {13,14}    num2cell(200:205)  {35,36,32}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-4280)/20);
        CsiBeamIdxMap(i) = [16, 17, 18, 19] + 4 * floor((i-4280)/20);
    end 
    % slot 7, 9, 11 TRS: idx 4, 5, 6
    for i = 4340:4399
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        4      {}       {13,14}    num2cell(200:205)  {33,34}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        4      {}       {13,14}    num2cell(200:205)  {35,36}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-4340)/20);
    end
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 2:
    % slot 0, 1, 2 SSB 2 blocks: same as frame 0 slots 0, 1, 2
    % slot 3 SSB 1 block: same as frame 0 slot 3
    % slot 6 TRS and CSIRS, idx 0-4
    for i = 4400:4419
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        4      {}       {13,14}    num2cell(200:205)  {33,34,31}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        4      {}       {13,14}    num2cell(200:205)  {35,36,32}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [0, 1, 2, 3];
    end 
    % slot 7, {8,9},{10,11},{16,17} TRS: same as frame 0 slots 7, 9, 11, 17 TRS idx 0, 1,2, 3
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 3:
    % slot {6,7},{8,9},{10,11} same as frame 1 slots 7, 9, 11 TRS: idx 4, 5, 6
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% slots 13, 23, 33, 53, 63, 73 pdcch only
    for i = 4420:4439
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        4      {}         {13,14}    {}  {}};
    end
    % end pattern 49
    
     
    
    
                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 50: 7 beams, 80 MHz (217 PRBs), 16C, peak cell, OTA
    %%%%% frame 0:         
    % slot 0,1,2 two ssb, idx {0, 1},{2,3},{4,5} 
    row_count = length(CFG);    
    for i = 4440:4487
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        5      {13}        {15,16}    num2cell(218:223)    {}};
        SsbBeamIdxMap(i) = [0, 1] + 2 * floor((i-4440)/16);
    end
    % slot 3, ssb at 2-5, idx 6
    for i = 4488:4503
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        5      {15}        {15,16}     {}     {}};
        SsbBeamIdxMap(i) = 6;
    end
    % slot {6, 7}, 9, 11, 17 TRS, idx 0, 1,2,3
    for i = 4504:4567
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        5      {}       {15,16}    num2cell(224:229)  {37,38}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        5      {}       {15,16}    num2cell(224:229)  {39,40}};
        end
        TrsBeamIdxMap(i) = floor((i-(4440+ 16*4))/16);
    end 
    % slot 8,10,16, TRS + CSIRS, idx 4-7, 8-11, 12-15
    for i = 4568:4615
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        5      {}       {15,16}    num2cell(224:229)  {37,38,41}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        5      {}       {15,16}    num2cell(224:229)  {39,40,42}};
        end
        TrsBeamIdxMap(i) = 1 + floor((i-4568)/16);
        CsiBeamIdxMap(i) = [4, 5, 6, 7]  + 4 * floor((i-4568)/16);
    end
    % 12, 18, 19, no ssb or trs/csirs:
    for i = 4616:4631
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        5      {}         {15,16}    num2cell(224:229)  {}};
    end
    % 13, s slot, pdcch only: see last set
    %%%%% frame 1: 
    % slot 6, 8, 10 TRS + CSIRS: idx 16-19, 20-23, 24-27
    for i = 4632:4679
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        5      {}       {15,16}    num2cell(224:229)  {37,38,41}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        5      {}       {15,16}    num2cell(224:229)  {39,40,42}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-4632)/16);
        CsiBeamIdxMap(i) = [16, 17, 18, 19] + 4 * floor((i-4632)/16);
    end 
    % slot 7, 9, 11 TRS: idx 4, 5, 6
    for i = 4680:4727
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        5      {}       {15,16}    num2cell(224:229)  {37,38}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        5      {}       {15,16}    num2cell(224:229)  {39,40}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-4680)/16);
    end
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 2:
    % slot 0, 1, 2 SSB 2 blocks: same as frame 0 slots 0, 1, 2
    % slot 3 SSB 1 block: same as frame 0 slot 3
    % slot 6 TRS and CSIRS, idx 0-4
    for i = 4728:4743
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        5      {}       {15,16}    num2cell(224:229)  {37,38,41}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        5      {}       {15,16}    num2cell(224:229)  {39,40,42}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [0, 1, 2, 3];
    end 
    % slot 7, {8,9},{10,11},{16,17} TRS: same as frame 0 slots 7, 9, 11, 17 TRS idx 0, 1,2, 3
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 3:
    % slot {6,7},{8,9},{10,11} same as frame 1 slots 7, 9, 11 TRS: idx 4, 5, 6
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% slots 13, 23, 33, 53, 63, 73 pdcch only
    for i = 4744:4759
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        5      {}         {15,16}    {}  {}};
    end
    % end pattern 50
    
    
                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 51: 7 beams, 80 MHz (217 PRBs), 20C, ave cell, OTA, disjoint PDSCH and CSIRS
    %%%%% frame 0:         
    % slot 0,1,2 two ssb, idx {0, 1},{2,3},{4,5} 
    row_count = length(CFG);    
    for i = 4760:4819
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        5      {13}        {15,16}    num2cell(230:235)    {}};
        SsbBeamIdxMap(i) = [0, 1] + 2 * floor((i-4760)/20);
    end
    % slot 3, ssb at 2-5, idx 6
    for i = 4820:4839
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        5      {15}        {15,16}     {}     {}};
        SsbBeamIdxMap(i) = 6;
    end
    % slot {6, 7}, 9, 11, 17 TRS, idx 0, 1,2,3
    for i = 4840:4919
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        5      {}       {15,16}    num2cell(242:247)  {43,44}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        5      {}       {15,16}    num2cell(242:247)  {45,46}};
        end
        TrsBeamIdxMap(i) = floor((i-4840)/20);
    end 
    % slot 8,10,16, TRS + CSIRS, idx 4-7, 8-11, 12-15
    for i = 4920:4979
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        5      {}       {15,16}    num2cell(242:247)  {43,44,41}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        5      {}       {15,16}    num2cell(242:247)  {45,46,42}};
        end
        TrsBeamIdxMap(i) = 1 + floor((i-4920)/20);
        CsiBeamIdxMap(i) = [4, 5, 6, 7]  + 4 * floor((i-4920)/20);
    end
    % 12, 18, 19, no ssb or trs/csirs:
    for i = 4980:4999
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        5      {}         {15,16}    num2cell(236:241)  {}};
    end
    % 13, s slot, pdcch only: see last set
    %%%%% frame 1: 
    % slot 6, 8, 10 TRS + CSIRS: idx 16-19, 20-23, 24-27
    for i = 5000:5059
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        5      {}       {15,16}    num2cell(242:247)  {43,44,41}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        5      {}       {15,16}    num2cell(242:247)  {45,46,42}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-5000)/20);
        CsiBeamIdxMap(i) = [16, 17, 18, 19] + 4 * floor((i-5000)/20);
    end 
    % slot 7, 9, 11 TRS: idx 4, 5, 6
    for i = 5060:5119
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        5      {}       {15,16}    num2cell(242:247)  {43,44}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        5      {}       {15,16}    num2cell(242:247)  {45,46}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-5060)/20);
    end
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 2:
    % slot 0, 1, 2 SSB 2 blocks: same as frame 0 slots 0, 1, 2
    % slot 3 SSB 1 block: same as frame 0 slot 3
    % slot 6 TRS and CSIRS, idx 0-4
    for i = 5120:5139
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        5      {}       {15,16}    num2cell(242:247)  {43,44,41}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        5      {}       {15,16}    num2cell(242:247)  {45,46,42}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [0, 1, 2, 3];
    end 
    % slot 7, {8,9},{10,11},{16,17} TRS: same as frame 0 slots 7, 9, 11, 17 TRS idx 0, 1,2, 3
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 3:
    % slot {6,7},{8,9},{10,11} same as frame 1 slots 7, 9, 11 TRS: idx 4, 5, 6
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% slots 13, 23, 33, 53, 63, 73 pdcch only
    for i = 5140:5159
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        5      {}         {15,16}    {}  {}};
    end
    % end pattern 51
    
                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 53: 7 beams, 30 MHz (78 PRBs), 16C, peak cell, OTA
    %%%%% frame 0:         
    % slot 0,1,2 two ssb, idx {0, 1},{2,3},{4,5} 
    row_count = length(CFG);    
    for i = 5160:5207
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        6      {6}        {17,18}    num2cell(252:257)    {}};
        SsbBeamIdxMap(i) = [0, 1] + 2 * floor((i-5160)/16);
    end
    % slot 3, ssb at 2-5, idx 6
    for i = 5208:5223
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        6      {4}        {17,18}     {}     {}};
        SsbBeamIdxMap(i) = 6;
    end
    % slot {6, 7}, 9, 11, 17 TRS, idx 0, 1,2,3
    for i = 5224:5287
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        6      {}       {17,18}    num2cell(258:263)  {47,48}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        6      {}       {17,18}    num2cell(258:263)  {49,50}};
        end
        TrsBeamIdxMap(i) = floor((i-(5160+ 16*4))/16);
    end 
    % slot 8,10,16, TRS + CSIRS, idx 4-7, 8-11, 12-15
    for i = 5288:5335
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        6      {}       {17,18}    num2cell(258:263)  {47,48,51}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        6      {}       {17,18}    num2cell(258:263)  {49,50,52}};
        end
        TrsBeamIdxMap(i) = 1 + floor((i-5288)/16);
        CsiBeamIdxMap(i) = [4, 5, 6, 7]  + 4 * floor((i-5288)/16);
    end
    % 12, 18, 19, no ssb or trs/csirs:
    for i = 5336:5351
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        6      {}         {17,18}    num2cell(258:263)  {}};
    end
    % 13, s slot, pdcch only: see last set
    %%%%% frame 1: 
    % slot 6, 8, 10 TRS + CSIRS: idx 16-19, 20-23, 24-27
    for i = 5352:5399
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        6      {}       {17,18}    num2cell(258:263)  {47,48,51}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        6      {}       {17,18}    num2cell(258:263)  {49,50,52}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-5352)/16);
        CsiBeamIdxMap(i) = [16, 17, 18, 19] + 4 * floor((i-5352)/16);
    end 
    % slot 7, 9, 11 TRS: idx 4, 5, 6
    for i = 5400:5447
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        6      {}       {17,18}    num2cell(258:263)  {47,48}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        6      {}       {17,18}    num2cell(258:263)  {49,50}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-5400)/16);
    end
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 2:
    % slot 0, 1, 2 SSB 2 blocks: same as frame 0 slots 0, 1, 2
    % slot 3 SSB 1 block: same as frame 0 slot 3
    % slot 6 TRS and CSIRS, idx 0-4
    for i = 5448:5463
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        6      {}       {17,18}    num2cell(258:263)  {47,48,51}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        6      {}       {17,18}    num2cell(258:263)  {49,50,52}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [0, 1, 2, 3];
    end 
    % slot 7, {8,9},{10,11},{16,17} TRS: same as frame 0 slots 7, 9, 11, 17 TRS idx 0, 1,2, 3
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 3:
    % slot {6,7},{8,9},{10,11} same as frame 1 slots 7, 9, 11 TRS: idx 4, 5, 6
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% slots 13, 23, 33, 53, 63, 73 pdcch only
    for i = 5464:5479
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        6      {}         {17,18}    {}  {}};
    end
    % end pattern 53
    
                             % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 54: 7 beams, 30 MHz (78 PRBs), 20C, ave cell, OTA, disjoint PDSCH and CSIRS
    %%%%% frame 0:         
    % slot 0,1,2 two ssb, idx {0, 1},{2,3},{4,5} 
    row_count = length(CFG);    
    for i = 5480:5539
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        6      {6}        {19,20}    num2cell(264:269)    {}};
        SsbBeamIdxMap(i) = [0, 1] + 2 * floor((i-5480)/20);
    end
    % slot 3, ssb at 2-5, idx 6
    for i = 5540:5559
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        6      {4}        {19,20}     {}     {}};
        SsbBeamIdxMap(i) = 6;
    end
    % slot {6, 7}, 9, 11, 17 TRS, idx 0, 1,2,3
    for i = 5560:5639
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        6      {}       {19,20}    num2cell(276:281)  {53,54}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        6      {}       {19,20}    num2cell(276:281)  {55,56}};
        end
        TrsBeamIdxMap(i) = floor((i-5560)/20);
    end 
    % slot 8,10,16, TRS + CSIRS, idx 4-7, 8-11, 12-15
    for i = 5640:5699
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        6      {}       {19,20}    num2cell(276:281)  {53,54,51}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        6      {}       {19,20}    num2cell(276:281)  {55,56,52}};
        end
        TrsBeamIdxMap(i) = 1 + floor((i-5640)/20);
        CsiBeamIdxMap(i) = [4, 5, 6, 7]  + 4 * floor((i-5640)/20);
    end
    % 12, 18, 19, no ssb or trs/csirs:
    for i = 5700:5719
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        6      {}         {19,20}    num2cell(270:275)  {}};
    end
    % 13, s slot, pdcch only: see last set
    %%%%% frame 1: 
    % slot 6, 8, 10 TRS + CSIRS: idx 16-19, 20-23, 24-27
    for i = 5720:5779
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        6      {}       {19,20}    num2cell(276:281)  {53,54,51}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        6      {}       {19,20}    num2cell(276:281)  {55,56,52}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-5720)/20);
        CsiBeamIdxMap(i) = [16, 17, 18, 19] + 4 * floor((i-5720)/20);
    end 
    % slot 7, 9, 11 TRS: idx 4, 5, 6
    for i = 5780:5839
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        6      {}       {19,20}    num2cell(276:281)  {53,54}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        6      {}       {19,20}    num2cell(276:281)  {55,56}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-5780)/20);
    end
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 2:
    % slot 0, 1, 2 SSB 2 blocks: same as frame 0 slots 0, 1, 2
    % slot 3 SSB 1 block: same as frame 0 slot 3
    % slot 6 TRS and CSIRS, idx 0-4
    for i = 5840:5859
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        6      {}       {19,20}    num2cell(276:281)  {53,54,51}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        6      {}       {19,20}    num2cell(276:281)  {55,56,52}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [0, 1, 2, 3];
    end 
    % slot 7, {8,9},{10,11},{16,17} TRS: same as frame 0 slots 7, 9, 11, 17 TRS idx 0, 1,2, 3
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 3:
    % slot {6,7},{8,9},{10,11} same as frame 1 slots 7, 9, 11 TRS: idx 4, 5, 6
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% slots 13, 23, 33, 53, 63, 73 pdcch only
    for i = 5860:5879
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        6      {}         {19,20}    {}  {}};
    end
    % end pattern 54

                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 57: 4 frames, 8C, peak cell, OTA, different UEs at different cells
    % frame 0: 6+6 DCI, 12C 4 beams, peak cell, SFN%2 == 0, (no  csirs)                     
    % slot 0, two ssb, 
    row_count = length(CFG);
    for i = 5960:5967
        row_count = row_count + 1;
        cidx = mod(i - 5960, 8);
        CFG(row_count, 1:7) = { i    0        1      {6}        {7,8}    num2cell(374:374+cidx)    {}};
    end
    % slot 1, ssb at 8-11, 
    row_count = length(CFG);
    for i = 5968:5975
        row_count = row_count + 1;
        cidx = mod(i - 5960, 8);
        CFG(row_count, 1:7) = { i    0        1      {8}        {7,8}    num2cell(374:374+cidx)    {}};
    end
    % slot 2, ssb at 2-5, 
    row_count = length(CFG);
    for i = 5976:5983
        row_count = row_count + 1;
        cidx = mod(i - 5960, 8);
        CFG(row_count, 1:7) = { i    0        1      {4}        {7,8}    num2cell(374:374+cidx)  {}};
    end
    % slot 3,6,7,8,9,10,11,12,13,16,17,18,19 no ssb or trs/csirs:
    for i = 5984:5991
        row_count = row_count + 1;
        cidx = mod(i - 5960, 8);
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(382:382+cidx)  {}};
    end
    %%%%%%%%%%%%
    %%%% frame 1: TRS {6,10}/{5,9}, 6+6 DCI, 12C 4 beams, peak cell, SFN%2 == 1, (w/o ssb, w csirs)
    % slot 0, 10, 16, 18, trs symbol {6,10}/{5,9}, csirs symbol 12/13
    row_count = length(CFG);
    for i = 5992:5999
        row_count = row_count + 1;
        cidx = mod(i - 5960, 8);
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(382:382+cidx)  {10,11,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(382:382+cidx)  {14,15,13}};
        end
    end
    % slot 1, 11, 17, 19
    for i = 6000:6007
        row_count = row_count + 1;
        cidx = mod(i - 5960, 8);
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(382:382+cidx)  {10,11}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(382:382+cidx)  {14,15}};
        end
    end
    % other slots, no ssb, no csi-rs
    for i = 6008:6015
        row_count = row_count + 1;
        cidx = mod(i - 5960, 8);
        CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(382:382+cidx)  {}};
    end
    %%%%%%%%%%%%
    %%%% frame 2: repeat frame 0
    %%%%%%%%%%%%
    %%%% frame 3: only have TRS:
    % slot 0,1,10,11,16,17,18,19 trs symbol {6,10} / {5,9}, no csirs
    row_count = length(CFG);
    for i = 6016:6023
        row_count = row_count + 1;
        cidx = mod(i - 5960, 8);
        if mod(i,2) == 0  % trs symbol {6,10}, no csirs
            CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(382:382+cidx)  {10,11}};
        else  % trs symbol {5,9}, no csirs
            CFG(row_count, 1:7) = { i    0        1      {}        {7,8}    num2cell(382:382+cidx)  {14,15}};
        end
    end
    % other slots, no ssb, no csi-rs
    for i = 6024:6031
        row_count = row_count + 1;
        cidx = mod(i - 5960, 8);
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(382:382+cidx)  {}};
    end
    %% end pattern 57
    
    
                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 55: 7 beams, 50 MHz (133 PRBs), 16C, peak cell, OTA
    %%%%% frame 0:         
    % slot 0,1,2 two ssb, idx {0, 1},{2,3},{4,5} 
    row_count = length(CFG);    
    for i = 6032:6079
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        7      {6}        {21,22}    num2cell(282:287)    {}};
        SsbBeamIdxMap(i) = [0, 1] + 2 * floor((i-6032)/16);
    end
    % slot 3, ssb at 2-5, idx 6
    for i = 6080:6095
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        7      {4}        {21,22}     {}     {}};
        SsbBeamIdxMap(i) = 6;
    end
    % slot {6, 7}, 9, 11, 17 TRS, idx 0, 1,2,3
    for i = 6096:6159
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        7      {}       {21,22}    num2cell(288:293)  {57,58}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        7      {}       {21,22}    num2cell(288:293)  {59,60}};
        end
        TrsBeamIdxMap(i) = floor((i-(6032+ 16*4))/16);
    end 
    % slot 8,10,16, TRS + CSIRS, idx 4-7, 8-11, 12-15
    for i = 6160:6207
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        7      {}       {21,22}    num2cell(288:293)  {57,58,61}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        7      {}       {21,22}    num2cell(288:293)  {59,60,62}};
        end
        TrsBeamIdxMap(i) = 1 + floor((i-6160)/16);
        CsiBeamIdxMap(i) = [4, 5, 6, 7]  + 4 * floor((i-6160)/16);
    end
    % 12, 18, 19, no ssb or trs/csirs:
    for i = 6208:6223
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        7      {}         {21,22}    num2cell(288:293)  {}};
    end
    % 13, s slot, pdcch only: see last set
    %%%%% frame 1: 
    % slot 6, 8, 10 TRS + CSIRS: idx 16-19, 20-23, 24-27
    for i = 6224:6271
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        7      {}       {21,22}    num2cell(288:293)  {57,58,61}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        7      {}       {21,22}    num2cell(288:293)  {59,60,62}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-6224)/16);
        CsiBeamIdxMap(i) = [16, 17, 18, 19] + 4 * floor((i-6224)/16);
    end 
    % slot 7, 9, 11 TRS: idx 4, 5, 6
    for i = 6272:6319
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        7      {}       {21,22}    num2cell(288:293)  {57,58}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        7      {}       {21,22}    num2cell(288:293)  {59,60}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-6272)/16);
    end
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 2:
    % slot 0, 1, 2 SSB 2 blocks: same as frame 0 slots 0, 1, 2
    % slot 3 SSB 1 block: same as frame 0 slot 3
    % slot 6 TRS and CSIRS, idx 0-4
    for i = 6320:6335
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        7      {}       {21,22}    num2cell(288:293)  {57,58,61}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        7      {}       {21,22}    num2cell(288:293)  {59,60,62}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [0, 1, 2, 3];
    end 
    % slot 7, {8,9},{10,11},{16,17} TRS: same as frame 0 slots 7, 9, 11, 17 TRS idx 0, 1,2, 3
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 3:
    % slot {6,7},{8,9},{10,11} same as frame 1 slots 7, 9, 11 TRS: idx 4, 5, 6
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% slots 13, 23, 33, 53, 63, 73 pdcch only
    for i = 6336:6351
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        7      {}         {21,22}    {}  {}};
    end
    % end pattern 55
    
                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 56: 7 beams, 50 MHz (133 PRBs), 20C, ave cell, OTA, disjoint PDSCH and CSIRS
    %%%%% frame 0:         
    % slot 0,1,2 two ssb, idx {0, 1},{2,3},{4,5} 
    row_count = length(CFG);    
    for i = 6352:6411
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        7      {6}        {23,24}    num2cell(294:299)    {}};
        SsbBeamIdxMap(i) = [0, 1] + 2 * floor((i-6352)/20);
    end
    % slot 3, ssb at 2-5, idx 6
    for i = 6412:6431
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        7      {4}        {23,24}     {}     {}};
        SsbBeamIdxMap(i) = 6;
    end
    % slot {6, 7}, 9, 11, 17 TRS, idx 0, 1,2,3
    for i = 6432:6511
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        7      {}       {23,24}    num2cell(306:311)  {63,64}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        7      {}       {23,24}    num2cell(306:311)  {65,66}};
        end
        TrsBeamIdxMap(i) = floor((i-6432)/20);
    end 
    % slot 8,10,16, TRS + CSIRS, idx 4-7, 8-11, 12-15
    for i = 6512:6571
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        7      {}       {23,24}    num2cell(306:311)  {63,64,61}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        7      {}       {23,24}    num2cell(306:311)  {65,66,62}};
        end
        TrsBeamIdxMap(i) = 1 + floor((i-6512)/20);
        CsiBeamIdxMap(i) = [4, 5, 6, 7]  + 4 * floor((i-6512)/20);
    end
    % 12, 18, 19, no ssb or trs/csirs:
    for i = 6572:6591
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        7      {}         {23,24}    num2cell(300:305)  {}};
    end
    % 13, s slot, pdcch only: see last set
    %%%%% frame 1: 
    % slot 6, 8, 10 TRS + CSIRS: idx 16-19, 20-23, 24-27
    for i = 6592:6651
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        7      {}       {23,24}    num2cell(306:311)  {63,64,61}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        7      {}       {23,24}    num2cell(306:311)  {65,66,62}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-6592)/20);
        CsiBeamIdxMap(i) = [16, 17, 18, 19] + 4 * floor((i-6592)/20);
    end 
    % slot 7, 9, 11 TRS: idx 4, 5, 6
    for i = 6652:6711
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        7      {}       {23,24}    num2cell(306:311)  {63,64}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        7      {}       {23,24}    num2cell(306:311)  {65,66}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-6652)/20);
    end
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 2:
    % slot 0, 1, 2 SSB 2 blocks: same as frame 0 slots 0, 1, 2
    % slot 3 SSB 1 block: same as frame 0 slot 3
    % slot 6 TRS and CSIRS, idx 0-4
    for i = 6712:6731
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        7      {}       {23,24}    num2cell(306:311)  {63,64,61}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        7      {}       {23,24}    num2cell(306:311)  {65,66,62}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [0, 1, 2, 3];
    end 
    % slot 7, {8,9},{10,11},{16,17} TRS: same as frame 0 slots 7, 9, 11, 17 TRS idx 0, 1,2, 3
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 3:
    % slot {6,7},{8,9},{10,11} same as frame 1 slots 7, 9, 11 TRS: idx 4, 5, 6
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% slots 13, 23, 33, 53, 63, 73 pdcch only
    for i = 6732:6751
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        7      {}         {23,24}    {}  {}};
    end
    % end pattern 56
    
    
    %% pattern 58: 7 beams, 100 MHz (273 PRBs), 16C, ave cell, OTA, Full BW CSI-RS
    %%%%% frame 0:         
    % slot 0,1,2 two ssb, idx {0, 1},{2,3},{4,5} 
    row_count = length(CFG);    
    for i = 6752:6799
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {6}        {7,8}    num2cell(170:175)    {}};
        SsbBeamIdxMap(i) = [0, 1] + 2 * floor((i-6752)/16);
    end
    % slot 3, ssb at 2-5, idx 6
    for i = 6800:6815
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {4}        {7,8}     {}     {}};
        SsbBeamIdxMap(i) = 6;
    end
    % slot {6, 7}, 9, 11, 17 TRS, idx 0, 1,2,3
    for i = 6816:6879
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(176:181)  {10,11}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(176:181)  {14,15}};
        end
        TrsBeamIdxMap(i) = floor((i-6816)/16);
    end 
    % slot 8,10,16, TRS + CSIRS, idx 4-7, 8-11, 12-15
    for i = 6880:6927
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(176:181)  {10,11,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(176:181)  {14,15,13}};
        end
        TrsBeamIdxMap(i) = 1 + floor((i-6880)/16);
        CsiBeamIdxMap(i) = [4, 5, 6, 7]  + 4 * floor((i-6880)/16);
    end
    % 12, 18, 19, no ssb or trs/csirs:
    for i = 6928:6943
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(176:181)  {}};
    end
    % 13, s slot, pdcch only: see last set
    %%%%% frame 1: 
    % slot 6, 8, 10 TRS + CSIRS: idx 16-19, 20-23, 24-27
    for i = 6944:6991
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(176:181)  {10,11,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(176:181)  {14,15,13}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-6944)/16);
        CsiBeamIdxMap(i) = [16, 17, 18, 19] + 4 * floor((i-6944)/16);
    end 
    % slot 7, 9, 11 TRS: idx 4, 5, 6
    for i = 6992:7039
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(176:181)  {10,11}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(176:181)  {14,15}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-6992)/16);
    end
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 2:
    % slot 0, 1, 2 SSB 2 blocks: same as frame 0 slots 0, 1, 2
    % slot 3 SSB 1 block: same as frame 0 slot 3
    % slot 6 TRS and CSIRS, idx 0-4
    for i = 7040:7055
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(176:181)  {10,11,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(176:181)  {14,15,13}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [0, 1, 2, 3];
    end 
    % slot 7, {8,9},{10,11},{16,17} TRS: same as frame 0 slots 7, 9, 11, 17 TRS idx 0, 1,2, 3
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 3:
    % slot {6,7},{8,9},{10,11} same as frame 1 slots 7, 9, 11 TRS: idx 4, 5, 6
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% slots 13, 23, 33, 53, 63, 73 pdcch only
    for i = 7056:7071
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    {}  {}};
    end
    % end pattern 58

    %% pattern 59: 7 beams, 100 MHz (273 PRBs), 20C, peak cell, OTA
    %%%%% frame 0:         
    % slot 0,1,2 two ssb, idx {0, 1},{2,3},{4,5} 
    row_count = length(CFG);    
    for i = 7072:7131
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {6}        {7,8}    num2cell(158:163)    {}};
        SsbBeamIdxMap(i) = [0, 1] + 2 * floor((i-7072)/20);
    end
    % slot 3, ssb at 2-5, idx 6
    for i = 7132:7151
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {4}        {7,8}     num2cell(496:501)     {}};
        SsbBeamIdxMap(i) = 6;
    end
    % slot {6, 7}, 9, 11, 17 TRS, idx 0, 1,2,3
    for i = 7152:7231
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {10,11}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {14,15}};
        end
        TrsBeamIdxMap(i) = floor((i-(7072+ 20*4))/20);
    end 
    % slot 8,10,16, TRS + CSIRS, idx 4-7, 8-11, 12-15
    for i = 7232:7291
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {10,11,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {14,15,13}};
        end
        TrsBeamIdxMap(i) = 1 + floor((i-7232)/20);
        CsiBeamIdxMap(i) = [4, 5, 6, 7]  + 4 * floor((i-7232)/20);
    end
    % 12, 18, 19, no ssb or trs/csirs:
    for i = 7292:7311
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(164:169)  {}};
    end
    % 13, s slot, pdcch only: see last set
    %%%%% frame 1: 
    % slot 6, 8, 10 TRS + CSIRS: idx 16-19, 20-23, 24-27
    for i = 7312:7371
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {10,11,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {14,15,13}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-7312)/20);
        CsiBeamIdxMap(i) = [16, 17, 18, 19] + 4 * floor((i-7312)/20);
    end 
    % slot 7, 9, 11 TRS: idx 4, 5, 6
    for i = 7372:7431
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {10,11}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {14,15}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-7372)/20);
    end
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 2:
    % slot 0, 1, 2 SSB 2 blocks: same as frame 0 slots 0, 1, 2
    % slot 3 SSB 1 block: same as frame 0 slot 3
    % slot 6 TRS and CSIRS, idx 0-4
    for i = 7432:7451
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {10,11,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {14,15,13}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [0, 1, 2, 3];
    end 
    % slot 7, {8,9},{10,11},{16,17} TRS: same as frame 0 slots 7, 9, 11, 17 TRS idx 0, 1,2, 3
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 3:
    % slot {6,7},{8,9},{10,11} same as frame 1 slots 7, 9, 11 TRS: idx 4, 5, 6
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% slots 13, 23, 33, 53, 63, 73 pdcch + 5-symbol pdsch
    for i = 7452:7471
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(502:507)  {}};
    end
    % end pattern 59: 7 beams, 100 MHz (273 PRBs), 20C, peak cell, OTA

    %% pattern 101: 1 UE SU-MIMO, 4 layers, MCS 27, 256QAM, PDSCH only, all D slots, 24C
    % PDSCH: cfg 1224 (1 UE, all D slot)
                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    row_count = length(CFG);
    % all D slots, single UE wideband
    for i = 7472:7495
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         1         {}        {}        {1224}      {}};
        CellIdxInPatternMap(i) = mod(i-7472, 24);
    end
    % end pattern 101: 1 UE SU-MIMO, 4 layers, MCS 27, 256QAM, PDSCH only, 24C

    %% pattern 102: 1 UE SU-MIMO, 4 layers, MCS 27, 256QAM, PDSCH only (reduced PRB), all D slots, 24C
    % PDSCH: cfg 1225 (1 UE, all D slot, reduced PRB)
                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    row_count = length(CFG);
    % all D slots, single UE wideband
    for i = 7496:7519
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0         1         {}        {}        {1225}      {}};
        CellIdxInPatternMap(i) = mod(i-7496, 24);
    end
    % end pattern 102: 1 UE SU-MIMO, 4 layers, MCS 27, 256QAM, PDSCH only (reduced PRB), 24C
                  
    % [7520:7871] are available for future DLMIX TVs

                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 60: 7 beams, 100 MHz (273 PRBs), 40C, ave cell, OTA, disjoint PDSCH and CSIRS
    row_count = length(CFG);  
    %%%%% frame 0:         
    % slot 0,1,2 two ssb, idx {0, 1},{2,3},{4,5}   
    for i = 10452:10571
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {6}        {7,8}    num2cell(170:175)    {}};
        SsbBeamIdxMap(i) = [0, 1] + 2 * floor((i-10452)/40);
        CellIdxInPatternMap(i) = mod(i-10452, 40);
        enableIdentityPrecoderMap(i) = 1;
    end
    % slot 3, ssb at 2-5, idx 6
    for i = 10572:10611
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {4}        {7,8}     num2cell(508:513)     {}};
        SsbBeamIdxMap(i) = 6;
        CellIdxInPatternMap(i) = mod(i-10452, 40);
        enableIdentityPrecoderMap(i) = 1;
    end
    % slot {6, 7}, 9, 11, 17 TRS, idx 0, 1,2,3
    for i = 10612:10771
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {22,23}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {24,25}};
        end
        TrsBeamIdxMap(i) = floor((i-10612)/40);
        CellIdxInPatternMap(i) = mod(i-10452, 40);
        enableIdentityPrecoderMap(i) = 1;
    end 
    % slot 8,10,16, TRS + CSIRS, idx 4-7, 8-11, 12-15
    for i = 10772:10891
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {22,23,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {24,25,13}};
        end
        TrsBeamIdxMap(i) = 1 + floor((i-10772)/40);
        CsiBeamIdxMap(i) = [4, 5, 6, 7]  + 4 * floor((i-10772)/40);
        CellIdxInPatternMap(i) = mod(i-10452, 40);
        enableIdentityPrecoderMap(i) = 1;
    end
    % 12, 18, 19, no ssb or trs/csirs:
    for i = 10892:10931
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(176:181)  {}};
        CellIdxInPatternMap(i) = mod(i-10452, 40);
        enableIdentityPrecoderMap(i) = 1;
    end
    % 13, s slot, pdcch only: see last set
    %%%%% frame 1: 
    % slot 6, 8, 10 TRS + CSIRS: idx 16-19, 20-23, 24-27
    for i = 10932:11051
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {22,23,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {24,25,13}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-10932)/40);
        CsiBeamIdxMap(i) = [16, 17, 18, 19] + 4 * floor((i-10932)/40);
        CellIdxInPatternMap(i) = mod(i-10452, 40);
        enableIdentityPrecoderMap(i) = 1;
    end 
    % slot 7, 9, 11 TRS: idx 4, 5, 6
    for i = 11052:11171
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {22,23}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {24,25}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-11052)/40);
        CellIdxInPatternMap(i) = mod(i-10452, 40);
        enableIdentityPrecoderMap(i) = 1;
    end
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 2:
    % slot 0, 1, 2 SSB 2 blocks: same as frame 0 slots 0, 1, 2
    % slot 3 SSB 1 block: same as frame 0 slot 3
    % slot 6 TRS and CSIRS, idx 0-4
    for i = 11172:11211
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {22,23,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(182:187)  {24,25,13}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [0, 1, 2, 3];
        CellIdxInPatternMap(i) = mod(i-10452, 40);
        enableIdentityPrecoderMap(i) = 1;
    end 
    % slot 7, {8,9},{10,11},{16,17} TRS: same as frame 0 slots 7, 9, 11, 17 TRS idx 0, 1,2, 3
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 3:
    % slot {6,7},{8,9},{10,11} same as frame 1 slots 7, 9, 11 TRS: idx 4, 5, 6
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% slots 13, 23, 33, 53, 63, 73 pdcch + 5-symbol pdsch
    for i = 11212:11251
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(514:519)  {}};
        CellIdxInPatternMap(i) = mod(i-10452, 40);
        enableIdentityPrecoderMap(i) = 1;
    end
    % end pattern 60: 7 beams, 100 MHz (273 PRBs), 40C, ave cell, OTA, disjoint PDSCH and CSIRS
   

    %% pattern 61: 7 beams, 100 MHz (273 PRBs), 20C, light load, OTA, disjoint PDSCH and CSIRS
    %%%%% frame 0:         
    % slot 0,1,2 two ssb, idx {0, 1},{2,3},{4,5} 
    row_count = length(CFG);    
    for i = 7872:7931
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {6}        {7,8}    num2cell(390:395)    {}};
        SsbBeamIdxMap(i) = [0, 1] + 2 * floor((i-7872)/20);
    end
    % slot 3, ssb at 2-5, idx 6
    for i = 7932:7951
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {4}        {7,8}     {}     {}};
        SsbBeamIdxMap(i) = 6;
    end
    % slot {6, 7}, 9, 11, 17 TRS, idx 0, 1,2,3
    for i = 7952:8031
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(402:407)  {53,54}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(402:407)  {55,56}};
        end
        TrsBeamIdxMap(i) = floor((i-7952)/20);
    end 
    % slot 8,10,16, TRS + CSIRS, idx 4-7, 8-11, 12-15
    for i = 8032:8091
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(402:407)  {53,54,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(402:407)  {55,56,13}};
        end
        TrsBeamIdxMap(i) = 1 + floor((i-8032)/20);
        CsiBeamIdxMap(i) = [4, 5, 6, 7]  + 4 * floor((i-8032)/20);
    end
    % 12, 18, 19, no ssb or trs/csirs:
    for i = 8092:8111
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(396:401)  {}};
    end
    % 13, s slot, pdcch only: see last set
    %%%%% frame 1: 
    % slot 6, 8, 10 TRS + CSIRS: idx 16-19, 20-23, 24-27
    for i = 8112:8171
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(402:407)  {53,54,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(402:407)  {55,56,13}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-8112)/20);
        CsiBeamIdxMap(i) = [16, 17, 18, 19] + 4 * floor((i-8112)/20);
    end 
    % slot 7, 9, 11 TRS: idx 4, 5, 6
    for i = 8172:8231
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(402:407)  {53,54}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(402:407)  {55,56}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-8172)/20);
    end
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 2:
    % slot 0, 1, 2 SSB 2 blocks: same as frame 0 slots 0, 1, 2
    % slot 3 SSB 1 block: same as frame 0 slot 3
    % slot 6 TRS and CSIRS, idx 0-4
    for i = 8232:8251
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(402:407)  {53,54,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(402:407)  {55,56,13}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [0, 1, 2, 3];
    end 
    % slot 7, {8,9},{10,11},{16,17} TRS: same as frame 0 slots 7, 9, 11, 17 TRS idx 0, 1,2, 3
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 3:
    % slot {6,7},{8,9},{10,11} same as frame 1 slots 7, 9, 11 TRS: idx 4, 5, 6
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% slots 13, 23, 33, 53, 63, 73 pdcch only
    for i = 8252:8271
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    {}  {}};
    end
    % end pattern 61

    % [8872:9471] are available for future DLMIX TVs
    
    % 59 c/e can share the same DL TVs. Here create 59c 40C to work with 59 c/e 40C UL
                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 59c: 7 beams, 100 MHz (273 PRBs), 40C, peak cell, OTA
    %%%%% frame 0:         
    % slot 0,1,2 two ssb, idx {0, 1},{2,3},{4,5} 
    row_count = length(CFG);    
    for i = 9472:9591
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {6}        {7,8}    num2cell(158:163)    {}};
        SsbBeamIdxMap(i) = [0, 1] + 2 * floor((i-9472)/40);
        CellIdxInPatternMap(i) = mod(i-9472, 40);
        enableIdentityPrecoderMap(i) = 1;
    end
    % slot 3, ssb at 2-5, idx 6
    for i = 9592:9631
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {4}        {7,8}     num2cell(496:501)     {}};
        SsbBeamIdxMap(i) = 6;
        CellIdxInPatternMap(i) = mod(i-9472, 40);
        enableIdentityPrecoderMap(i) = 1;
    end
    % slot {6, 7}, 9, 11, 17 TRS, idx 0, 1,2,3
    for i = 9632:9791
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {10,11}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {14,15}};
        end
        TrsBeamIdxMap(i) = floor((i-(9472+ 40*4))/40);
        CellIdxInPatternMap(i) = mod(i-9472, 40);
        enableIdentityPrecoderMap(i) = 1;
    end 
    % slot 8,10,16, TRS + CSIRS, idx 4-7, 8-11, 12-15
    for i = 9792:9911
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {10,11,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {14,15,13}};
        end
        TrsBeamIdxMap(i) = 1 + floor((i-9792)/40);
        CsiBeamIdxMap(i) = [4, 5, 6, 7]  + 4 * floor((i-9792)/40);
        CellIdxInPatternMap(i) = mod(i-9472, 40);
        enableIdentityPrecoderMap(i) = 1;
    end
    % 12, 18, 19, no ssb or trs/csirs:
    for i = 9912:9951
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(164:169)  {}};
        CellIdxInPatternMap(i) = mod(i-9472, 40);
        enableIdentityPrecoderMap(i) = 1;
    end
    % 13, s slot, pdcch only: see last set
    %%%%% frame 1: 
    % slot 6, 8, 10 TRS + CSIRS: idx 16-19, 20-23, 24-27
    for i = 9952:10071
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {10,11,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {14,15,13}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-9952)/40);
        CsiBeamIdxMap(i) = [16, 17, 18, 19] + 4 * floor((i-9952)/40);
        CellIdxInPatternMap(i) = mod(i-9472, 40);
        enableIdentityPrecoderMap(i) = 1;
    end 
    % slot 7, 9, 11 TRS: idx 4, 5, 6
    for i = 10072:10191
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {10,11}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {14,15}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-10072)/40);
        CellIdxInPatternMap(i) = mod(i-9472, 40);
        enableIdentityPrecoderMap(i) = 1;
    end
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 2:
    % slot 0, 1, 2 SSB 2 blocks: same as frame 0 slots 0, 1, 2
    % slot 3 SSB 1 block: same as frame 0 slot 3
    % slot 6 TRS and CSIRS, idx 0-4
    for i = 10192:10231
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {10,11,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(164:169)  {14,15,13}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [0, 1, 2, 3];
        CellIdxInPatternMap(i) = mod(i-9472, 40);
        enableIdentityPrecoderMap(i) = 1;
    end 
    % slot 7, {8,9},{10,11},{16,17} TRS: same as frame 0 slots 7, 9, 11, 17 TRS idx 0, 1,2, 3
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 3:
    % slot {6,7},{8,9},{10,11} same as frame 1 slots 7, 9, 11 TRS: idx 4, 5, 6
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% slots 13, 23, 33, 53, 63, 73 pdcch + 5-symbol pdsch
    for i = 10232:10271
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(502:507)  {}};
        CellIdxInPatternMap(i) = mod(i-9472, 40);
        enableIdentityPrecoderMap(i) = 1;
    end
    % end pattern 59c: 7 beams, 100 MHz (273 PRBs), 40C, peak cell, OTA

    % [10272:10451] are available for future DLMIX TVs

                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 67: 64TR, 100 MHz (273 PRBs), 64TR worst case column B TC, 15C, based on 90623
    row_count = length(CFG); 
    %%%%% frame 0:         
    % slot {0}, 4 ssb, index 0~3
    for i = 11342:11356
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {22}        {61}    num2cell(583:590)    {}};
        SsbBeamIdxMap(i) = [0 1 2 3];
        CellIdxInPatternMap(i) = mod(i-11342, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {1,2}   
    for i = 11357:11371
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {61,62}    num2cell(591:598)    {}};
        CellIdxInPatternMap(i) = mod(i-11342, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {3}
    for i = 11372:11386
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {61}     num2cell(703:710)     {}};
        CellIdxInPatternMap(i) = mod(i-11342, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {6,7,8,9,10}
    for i = 11387:11401
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {61}     num2cell(570:577)     {}};
        CellIdxInPatternMap(i) = mod(i-11342, 15);
        pdschDynamicBfMap(i) = 1;
    end 
    % slot {11,12} is the same with {1,2}
    % slot {13} is the same with {3}
    % slot {16,17,18,19} is the same with {6,7,8,9,10}
    %%%%% frame 1:  
    % slot {20}, TRS index 0/0, CSIRS index 1~4
    for i = 11402:11416
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-11342, 15);        
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        10      {}        {61}     num2cell(570:577)     {82 83 84}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        10      {}        {61}     num2cell(570:577)     {91 92 93}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [1, 2, 3, 4];
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {21}, TRS + CSIRS, TRS index 5/5, CSIRS index 6~9
    for i = 11417:11431
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-11342, 15);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        10      {}        {61 62}     num2cell(591:598)     {82 83 84}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        10      {}        {61 62}     num2cell(591:598)     {91 92 93}};
        end
        TrsBeamIdxMap(i) = 5;
        CsiBeamIdxMap(i) = [6, 7, 8, 9];
        pdschDynamicBfMap(i) = 1;
    end
    % slot {22} is the same with {1,2}
    % slot {23} is the same with {3}
    % slot {26,27,28,29} is the same with {6,7,8,9,10}
    % slot {30} is the same with {6,7,8,9,10}
    % slot {31,32} is the same with {1,2}
    % slot {33} is the same with {13}
    % slot {36,37,38,39} is the same with {6,7,8,9,10}
    % end pattern 67: 64TR, 100 MHz (273 PRBs), 64TR worst case column B TC, 15C, based on 90623

                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 65: 7 beams, 100 MHz (273 PRBs), 40C, peak cell, OTA, PDSCH max 273 PRBs
    row_count = length(CFG);  
    %%%%% frame 0:         
    % slot 0,1,2 two ssb, idx {0, 1},{2,3},{4,5}   
    for i = 11432:11551
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {6}        {7,8}    num2cell(599:604)    {}};
        SsbBeamIdxMap(i) = [0, 1] + 2 * floor((i-11432)/40);
        CellIdxInPatternMap(i) = mod(i-11432, 40);
        enableIdentityPrecoderMap(i) = 1;
    end
    % slot 3, ssb at 2-5, idx 6
    for i = 11552:11591
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {4}        {7,8}     num2cell(611:616)     {}};
        SsbBeamIdxMap(i) = 6;
        CellIdxInPatternMap(i) = mod(i-11432, 40);
        enableIdentityPrecoderMap(i) = 1;
    end
    % slot {6, 7}, 9, 11, 17 TRS, idx 0, 1,2,3
    for i = 11592:11751
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(605:610)  {10,11}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(605:610)  {14,15}};
        end
        TrsBeamIdxMap(i) = floor((i-(11432+ 40*4))/40);
        CellIdxInPatternMap(i) = mod(i-11432, 40);
        enableIdentityPrecoderMap(i) = 1;
    end 
    % slot 8,10,16, TRS + CSIRS, idx 4-7, 8-11, 12-15
    for i = 11752:11871
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(605:610)  {10,11,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(605:610)  {14,15,13}};
        end
        TrsBeamIdxMap(i) = 1 + floor((i-11752)/40);
        CsiBeamIdxMap(i) = [4, 5, 6, 7]  + 4 * floor((i-11752)/40);
        CellIdxInPatternMap(i) = mod(i-11432, 40);
        enableIdentityPrecoderMap(i) = 1;
    end
    % 12, 18, 19, no ssb or trs/csirs:
    for i = 11872:11911
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(605:610)  {}};
        CellIdxInPatternMap(i) = mod(i-11432, 40);
        enableIdentityPrecoderMap(i) = 1;
    end
    % 13, s slot, pdcch only: see last set
    %%%%% frame 1: 
    % slot 6, 8, 10 TRS + CSIRS: idx 16-19, 20-23, 24-27
    for i = 11912:12031
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(605:610)  {10,11,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(605:610)  {14,15,13}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-11912)/40);
        CsiBeamIdxMap(i) = [16, 17, 18, 19] + 4 * floor((i-11912)/40);
        CellIdxInPatternMap(i) = mod(i-11432, 40);
        enableIdentityPrecoderMap(i) = 1;
    end 
    % slot 7, 9, 11 TRS: idx 4, 5, 6
    for i = 12032:12151
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(605:610)  {10,11}};
        else  % trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(605:610)  {14,15}};
        end
        TrsBeamIdxMap(i) = 4 + floor((i-12032)/40);
        CellIdxInPatternMap(i) = mod(i-11432, 40);
        enableIdentityPrecoderMap(i) = 1;
    end
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 2:
    % slot 0, 1, 2 SSB 2 blocks: same as frame 0 slots 0, 1, 2
    % slot 3 SSB 1 block: same as frame 0 slot 3
    % slot 6 TRS and CSIRS, idx 0-4
    for i = 12152:12191
        row_count = row_count + 1;
        if mod(i,2) == 0  % trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(605:610)  {10,11,12}};
        else  % trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        1      {}       {7,8}    num2cell(605:610)  {14,15,13}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [0, 1, 2, 3];
        CellIdxInPatternMap(i) = mod(i-11432, 40);
        enableIdentityPrecoderMap(i) = 1;
    end 
    % slot 7, {8,9},{10,11},{16,17} TRS: same as frame 0 slots 7, 9, 11, 17 TRS idx 0, 1,2, 3
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% frame 3:
    % slot {6,7},{8,9},{10,11} same as frame 1 slots 7, 9, 11 TRS: idx 4, 5, 6
    % other slots: same as frame 0 slots 12, 13, 18, 19
    %%%%% slots 13, 23, 33, 53, 63, 73 pdcch + 5-symbol pdsch
    for i = 12192:12231
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        1      {}         {7,8}    num2cell(617:622)  {}};
        CellIdxInPatternMap(i) = mod(i-11432, 40);
        enableIdentityPrecoderMap(i) = 1;
    end
    % end pattern 65: 7 beams, 100 MHz (273 PRBs), 40C, peak cell, OTA, PDSCH max 273 PRBs
 
                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 66: 64TR ave, 100 MHz (273 PRBs), 64TR worst case column B TC, PDSCH max 136 PRBs, 15C, based on 90623
    row_count = length(CFG); 
    %%%%% frame 0:         
    % slot {0}, 4 ssb, index 0~3
    for i = 11252:11266
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {22}        {64}    num2cell(687:694)    {}};
        SsbBeamIdxMap(i) = [0 1 2 3];
        CellIdxInPatternMap(i) = mod(i-11252, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {1,2}   
    for i = 11267:11281
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {64,65}    num2cell(695:702)    {}};
        CellIdxInPatternMap(i) = mod(i-11252, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {3}
    for i = 11282:11296
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {64}     num2cell(711:718)     {}};
        CellIdxInPatternMap(i) = mod(i-11252, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {6,7,8,9,10}
    for i = 11297:11311
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {64}     num2cell(674:681)     {}};
        CellIdxInPatternMap(i) = mod(i-11252, 15);
        pdschDynamicBfMap(i) = 1;
    end 
    % slot {11,12} is the same with {1,2}
    % slot {13} is the same with {3}
    % slot {16,17,18,19} is the same with {6,7,8,9,10}
    %%%%% frame 1:  
    % slot {20}, TRS index 0/0, CSIRS index 1~4
    for i = 11312:11326
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-11252, 15);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        10      {}        {64}     num2cell(674:681)     {82 83 84}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        10      {}        {64}     num2cell(674:681)     {91 92 93}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [1, 2, 3, 4];
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {21}, TRS + CSIRS, TRS index 5/5, CSIRS index 6~9
    for i = 11327:11341
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-11252, 15);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        10      {}        {64 65}     num2cell(695:702)     {82 83 84}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        10      {}        {64 65}     num2cell(695:702)     {91 92 93}};
        end
        TrsBeamIdxMap(i) = 5;
        CsiBeamIdxMap(i) = [6, 7, 8, 9];
        pdschDynamicBfMap(i) = 1;
    end
    % slot {22} is the same with {1,2}
    % slot {23} is the same with {3}
    % slot {26,27,28,29} is the same with {6,7,8,9,10}
    % slot {30} is the same with {6,7,8,9,10}
    % slot {31,32} is the same with {1,2}
    % slot {33} is the same with {13}
    % slot {36,37,38,39} is the same with {6,7,8,9,10}
    % end pattern 66: 64TR ave, 100 MHz (273 PRBs), 64TR worst case column B TC, PDSCH max 136 PRBs, 15C, based on 90623
 
                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 69: 64TR, 100 MHz (273 PRBs), 64TR worst case column G TC, 15C, based on 90624
    row_count = length(CFG); 
    %%%%% frame 0:         
    % slot {0}, 4 ssb, index 0~3   
    for i = 12232:12246
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {23}        {75}    num2cell(719:734)    {}};
        SsbBeamIdxMap(i) = [0 1 2 3];
        CellIdxInPatternMap(i) = mod(i-12232, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {1}, 4 ssb, index 4~7   
    for i = 12247:12261
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {24}        {75,74}    num2cell(735:750)    {}};
        SsbBeamIdxMap(i) = [4 5 6 7];
        CellIdxInPatternMap(i) = mod(i-12232, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {2}, 4 ssb, index 8~11   
    for i = 12262:12276
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {25}        {75,74}    num2cell(735:750)    {}};
        SsbBeamIdxMap(i) = [8 9 10 11];
        CellIdxInPatternMap(i) = mod(i-12232, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {3}, 4 ssb, index 12~15   
    for i = 12277:12291
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(751:766)     {}};
        CellIdxInPatternMap(i) = mod(i-12232, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {6,7,8,9,10}
    for i = 12292:12306
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(623:638)     {}};
        CellIdxInPatternMap(i) = mod(i-12232, 15);
        pdschDynamicBfMap(i) = 1;
    end 
    % slot {11,12}
    for i = 12307:12321
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {75,74}     num2cell(639:654)     {}};
        CellIdxInPatternMap(i) = mod(i-12232, 15);
        pdschDynamicBfMap(i) = 1;
    end 
    % slot {13} is the same with {3}
    % slot {16,17,18,19} is the same with {6,7,8,9,10}
    %%%%% frame 1:  
    % slot {20}, TRS index 0/0, CSIRS index 1~8
    for i = 12322:12336
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12232, 15);        
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(623:638)     {82 83 95}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(623:638)     {96 97 99}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [1, 2, 3, 4, 5, 6, 7, 8];
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {21}, TRS + CSIRS, TRS index 9/9, CSIRS index 10~17
    for i = 12337:12351
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12232, 15);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        10      {}        {75,74}     num2cell(639:654)     {82 83 95}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        10      {}        {75,74}     num2cell(639:654)     {96 97 99}};
        end
        TrsBeamIdxMap(i) = 9;
        CsiBeamIdxMap(i) = [10, 11, 12, 13, 14, 15, 16, 17];
        pdschDynamicBfMap(i) = 1;
    end
    % slot {22} is the same with {11,12}
    % slot {23} is the same with {3}
    % slot {26,27,28,29} is the same with {6,7,8,9,10}
    % slot {30}, TRS index 18/18, CSIRS index 19~26
    for i = 12352:12366
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12232, 15);        
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(623:638)     {82 83 95}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(623:638)     {96 97 99}};
        end
        TrsBeamIdxMap(i) = 18;
        CsiBeamIdxMap(i) = [19, 20, 21, 22, 23, 24, 25, 26];
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {31}, TRS + CSIRS, TRS index 27/27, CSIRS index 28~35
    for i = 12367:12381
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12232, 15);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        10      {}        {75,74}     num2cell(639:654)     {82 83 95}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        10      {}        {75,74}     num2cell(639:654)     {96 97 99}};
        end
        TrsBeamIdxMap(i) = 27;
        CsiBeamIdxMap(i) = [28, 29, 30, 31, 32, 33, 34, 35];
        pdschDynamicBfMap(i) = 1;
    end
    % slot {32} is the same with {11,12}
    % slot {33} is the same with {13}
    % slot {36}, TRS index 36/36, CSIRS index 37~44
    for i = 12382:12396
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12232, 15);        
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(623:638)     {82 83 95}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(623:638)     {96 97 99}};
        end
        TrsBeamIdxMap(i) = 36;
        CsiBeamIdxMap(i) = [37, 38, 39, 40, 41, 42, 43, 44];
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {37}, TRS + CSIRS, TRS index 45/45, CSIRS index 46~53
    for i = 12397:12411
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12232, 15);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(623:638)     {82 83 95}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(623:638)     {96 97 99}};
        end
        TrsBeamIdxMap(i) = 45;
        CsiBeamIdxMap(i) = [46, 47, 48, 49, 50, 51, 52, 53];
        pdschDynamicBfMap(i) = 1;
    end
    % slot {38}, TRS + CSIRS, TRS index 54/54, CSIRS index 55~62
    for i = 12412:12426
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12232, 15);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(623:638)     {82 83 95}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(623:638)     {96 97 99}};
        end
        TrsBeamIdxMap(i) = 54;
        CsiBeamIdxMap(i) = [55, 56, 57, 58, 59, 60, 61, 62];
        pdschDynamicBfMap(i) = 1;
    end
    % slot {39}, TRS index 63/63, CSIRS index 64~71
    for i = 12427:12441
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12232, 15);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(623:638)     {82 83 95}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(623:638)     {96 97 99}};
        end
        TrsBeamIdxMap(i) = 63;
        CsiBeamIdxMap(i) = [64, 65, 66, 67, 68, 69, 70, 71];
        pdschDynamicBfMap(i) = 1;
    end
    % end pattern 69: 64TR, 100 MHz (273 PRBs), 64TR worst case column G TC, 15C, based on 90624

                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 73: 64TR, 40 MHz (106 PRBs), 64TR worst case column G TC, 20C, based on 90624
    row_count = length(CFG); 
    %%%%% frame 0:         
    % slot {0}, 4 ssb, index 0~3   
    for i = 12442:12461
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        11      {23}        {69}    num2cell(767:782)    {}};
        SsbBeamIdxMap(i) = [0 1 2 3];
        CellIdxInPatternMap(i) = mod(i-12442, 20);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {1}, 4 ssb, index 4~7   
    for i = 12462:12481
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        11      {24}        {69,68}    num2cell(783:798)    {}};
        SsbBeamIdxMap(i) = [4 5 6 7];
        CellIdxInPatternMap(i) = mod(i-12442, 20);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {2}, 4 ssb, index 8~11   
    for i = 12482:12501
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        11      {25}        {69,68}    num2cell(783:798)    {}};
        SsbBeamIdxMap(i) = [8 9 10 11];
        CellIdxInPatternMap(i) = mod(i-12442, 20);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {3}, 4 ssb, index 12~15   
    for i = 12502:12521
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        11      {}        {69}     num2cell(799:814)     {}};
        CellIdxInPatternMap(i) = mod(i-12442, 20);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {6,7,8,9,10}
    for i = 12522:12541
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        11      {}        {69}     num2cell(815:830)     {}};
        CellIdxInPatternMap(i) = mod(i-12442, 20);
        pdschDynamicBfMap(i) = 1;
    end 
    % slot {11,12}
    for i = 12542:12561
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        11      {}        {69,68}     num2cell(831:846)     {}};
        CellIdxInPatternMap(i) = mod(i-12442, 20);
        pdschDynamicBfMap(i) = 1;
    end 
    % slot {13} is the same with {3}
    % slot {16,17,18,19} is the same with {6,7,8,9,10}
    %%%%% frame 1:  
    % slot {20}, TRS index 0/0, CSIRS index 1~8
    for i = 12562:12581
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12442, 20);        
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        11      {}        {69}     num2cell(815:830)     {100 101 102}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        11      {}        {69}     num2cell(815:830)     {103 104 105}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [1, 2, 3, 4, 5, 6, 7, 8];
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {21}, TRS + CSIRS, TRS index 9/9, CSIRS index 10~17
    for i = 12582:12601
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12442, 20);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        11      {}        {69,68}     num2cell(831:846)     {100 101 102}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        11      {}        {69,68}     num2cell(831:846)     {103 104 105}};
        end
        TrsBeamIdxMap(i) = 9;
        CsiBeamIdxMap(i) = [10, 11, 12, 13, 14, 15, 16, 17];
        pdschDynamicBfMap(i) = 1;
    end
    % slot {22} is the same with {11,12}
    % slot {23} is the same with {3}
    % slot {26,27,28,29} is the same with {6,7,8,9,10}
    % slot {30}, TRS index 18/18, CSIRS index 19~26
    for i = 12602:12621
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12442, 20);        
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        11      {}        {69}     num2cell(815:830)     {100 101 102}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        11      {}        {69}     num2cell(815:830)     {103 104 105}};
        end
        TrsBeamIdxMap(i) = 18;
        CsiBeamIdxMap(i) = [19, 20, 21, 22, 23, 24, 25, 26];
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {31}, TRS + CSIRS, TRS index 27/27, CSIRS index 28~35
    for i = 12622:12641
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12442, 20);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        11      {}        {69,68}     num2cell(831:846)     {100 101 102}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        11      {}        {69,68}     num2cell(831:846)     {103 104 105}};
        end
        TrsBeamIdxMap(i) = 27;
        CsiBeamIdxMap(i) = [28, 29, 30, 31, 32, 33, 34, 35];
        pdschDynamicBfMap(i) = 1;
    end
    % slot {32} is the same with {11,12}
    % slot {33} is the same with {13}
    % slot {36}, TRS index 36/36, CSIRS index 37~44
    for i = 12642:12661
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12442, 20);        
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        11      {}        {69}     num2cell(815:830)     {100 101 102}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        11      {}        {69}     num2cell(815:830)     {103 104 105}};
        end
        TrsBeamIdxMap(i) = 36;
        CsiBeamIdxMap(i) = [37, 38, 39, 40, 41, 42, 43, 44];
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {37}, TRS + CSIRS, TRS index 45/45, CSIRS index 46~53
    for i = 12662:12681
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12442, 20);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        11      {}        {69}     num2cell(815:830)     {100 101 102}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        11      {}        {69}     num2cell(815:830)     {103 104 105}};
        end
        TrsBeamIdxMap(i) = 45;
        CsiBeamIdxMap(i) = [46, 47, 48, 49, 50, 51, 52, 53];
        pdschDynamicBfMap(i) = 1;
    end
    % slot {38}, TRS + CSIRS, TRS index 54/54, CSIRS index 55~62
    for i = 12682:12701
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12442, 20);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        11      {}        {69}     num2cell(815:830)     {100 101 102}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        11      {}        {69}     num2cell(815:830)     {103 104 105}};
        end
        TrsBeamIdxMap(i) = 54;
        CsiBeamIdxMap(i) = [55, 56, 57, 58, 59, 60, 61, 62];
        pdschDynamicBfMap(i) = 1;
    end
    % slot {39}, TRS index 63/63, CSIRS index 64~71
    for i = 12702:12721
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12442, 20);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        11      {}        {69}     num2cell(815:830)     {100 101 102}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        11      {}        {69}     num2cell(815:830)     {103 104 105}};
        end
        TrsBeamIdxMap(i) = 63;
        CsiBeamIdxMap(i) = [64, 65, 66, 67, 68, 69, 70, 71];
        pdschDynamicBfMap(i) = 1;
    end
    % end pattern 73: 64TR, 40 MHz (106 PRBs), 64TR worst case column G TC, 20C, based on 90624

                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 75: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column G TC, 15C, based on 90624, 64 UEs per TTI
    row_count = length(CFG); 
    %%%%% frame 0:         
    % slot {0}, 4 ssb, index 0~3   
    for i = 12722:12736
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {23}        {75}    num2cell(2001:2064)    {}};
        SsbBeamIdxMap(i) = [0 1 2 3];
        CellIdxInPatternMap(i) = mod(i-12722, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {1}, 4 ssb, index 4~7   
    for i = 12737:12751
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {24}        {75,74}    num2cell(2065:2128)    {}};
        SsbBeamIdxMap(i) = [4 5 6 7];
        CellIdxInPatternMap(i) = mod(i-12722, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {2}, 4 ssb, index 8~11   
    for i = 12752:12766
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {25}        {75,74}    num2cell(2065:2128)    {}};
        SsbBeamIdxMap(i) = [8 9 10 11];
        CellIdxInPatternMap(i) = mod(i-12722, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {3}, 4 ssb, index 12~15   
    for i = 12767:12781
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(2129:2192)     {}};
        CellIdxInPatternMap(i) = mod(i-12722, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {6,7,8,9,10}
    for i = 12782:12796
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(2193:2256)     {}};
        CellIdxInPatternMap(i) = mod(i-12722, 15);
        pdschDynamicBfMap(i) = 1;
    end 
    % slot {11,12}
    for i = 12797:12811
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {75,74}     num2cell(2257:2320)     {}};
        CellIdxInPatternMap(i) = mod(i-12722, 15);
        pdschDynamicBfMap(i) = 1;
    end 
    % slot {13} is the same with {3}
    % slot {16,17,18,19} is the same with {6,7,8,9,10}
    %%%%% frame 1:  
    % slot {20}, TRS index 0/0, CSIRS index 1~8
    for i = 12812:12826
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12722, 15);        
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(2193:2256)     {82 83 95}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(2193:2256)     {96 97 99}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [1, 2, 3, 4, 5, 6, 7, 8];
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {21}, TRS + CSIRS, TRS index 9/9, CSIRS index 10~17
    for i = 12827:12841
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12722, 15);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        10      {}        {75,74}     num2cell(2257:2320)     {82 83 95}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        10      {}        {75,74}     num2cell(2257:2320)     {96 97 99}};
        end
        TrsBeamIdxMap(i) = 9;
        CsiBeamIdxMap(i) = [10, 11, 12, 13, 14, 15, 16, 17];
        pdschDynamicBfMap(i) = 1;
    end
    % slot {22} is the same with {11,12}
    % slot {23} is the same with {3}
    % slot {26,27,28,29} is the same with {6,7,8,9,10}
    % slot {30}, TRS index 18/18, CSIRS index 19~26
    for i = 12842:12856
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12722, 15);        
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(2193:2256)     {82 83 95}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(2193:2256)     {96 97 99}};
        end
        TrsBeamIdxMap(i) = 18;
        CsiBeamIdxMap(i) = [19, 20, 21, 22, 23, 24, 25, 26];
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {31}, TRS + CSIRS, TRS index 27/27, CSIRS index 28~35
    for i = 12857:12871
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12722, 15);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        10      {}        {75,74}     num2cell(2257:2320)     {82 83 95}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        10      {}        {75,74}     num2cell(2257:2320)     {96 97 99}};
        end
        TrsBeamIdxMap(i) = 27;
        CsiBeamIdxMap(i) = [28, 29, 30, 31, 32, 33, 34, 35];
        pdschDynamicBfMap(i) = 1;
    end
    % slot {32} is the same with {11,12}
    % slot {33} is the same with {13}
    % slot {36}, TRS index 36/36, CSIRS index 37~44
    for i = 12872:12886
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12722, 15);        
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(2193:2256)     {82 83 95}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(2193:2256)     {96 97 99}};
        end
        TrsBeamIdxMap(i) = 36;
        CsiBeamIdxMap(i) = [37, 38, 39, 40, 41, 42, 43, 44];
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {37}, TRS + CSIRS, TRS index 45/45, CSIRS index 46~53
    for i = 12887:12901
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12722, 15);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(2193:2256)     {82 83 95}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(2193:2256)     {96 97 99}};
        end
        TrsBeamIdxMap(i) = 45;
        CsiBeamIdxMap(i) = [46, 47, 48, 49, 50, 51, 52, 53];
        pdschDynamicBfMap(i) = 1;
    end
    % slot {38}, TRS + CSIRS, TRS index 54/54, CSIRS index 55~62
    for i = 12902:12916
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12722, 15);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(2193:2256)     {82 83 95}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(2193:2256)     {96 97 99}};
        end
        TrsBeamIdxMap(i) = 54;
        CsiBeamIdxMap(i) = [55, 56, 57, 58, 59, 60, 61, 62];
        pdschDynamicBfMap(i) = 1;
    end
    % slot {39}, TRS index 63/63, CSIRS index 64~71
    for i = 12917:12931
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12722, 15);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(2193:2256)     {82 83 95}};
        else  % odd cell: trs symbol 6,10, csi-rs on symbol 12
            CFG(row_count, 1:7) = { i    0        10      {}        {75}     num2cell(2193:2256)     {96 97 99}};
        end
        TrsBeamIdxMap(i) = 63;
        CsiBeamIdxMap(i) = [64, 65, 66, 67, 68, 69, 70, 71];
        pdschDynamicBfMap(i) = 1;
    end
    % end pattern 75: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column G TC, 15C, based on 90624, 64 UEs per TTI

                                % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 77: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column I TC, 15C, based on 90638
    row_count = length(CFG); 
    %%%%% frame 0:         
    % slot {0}, 4 ssb, index 0~3   
    for i = 12932:12946
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {22}        {72}    num2cell(2321:2336)    {}};
        SsbBeamIdxMap(i) = [0 1 2 3];
        CellIdxInPatternMap(i) = mod(i-12932, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {1,2} 
    for i = 12947:12961
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10       {}         {72,71}    num2cell(2337:2352)    {}};
        CellIdxInPatternMap(i) = mod(i-12932, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {3}
    for i = 12962:12976
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {72}     num2cell(2353:2368)     {}};
        CellIdxInPatternMap(i) = mod(i-12932, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {6,7,8,9}
    for i = 12977:12991
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {72}     num2cell(2369:2384)     {}};
        CellIdxInPatternMap(i) = mod(i-12932, 15);
        pdschDynamicBfMap(i) = 1;
    end 
    % slot {10}
    for i = 12992:13006
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}       {72,70}    num2cell([2385:2400,847])     {}};
        CellIdxInPatternMap(i) = mod(i-12932, 15);
        pdschDynamicBfMap(i) = 1;
    end 
    % slot {11,12} is the same with {1,2}
    % slot {13} is the same with {3}
    % slot {16,17,18,19} is the same with {6,7,8,9}
    %%%%% frame 1:  
    % slot {20}, TRS + CSIRS, TRS index 0/0, CSIRS index 1~4
    for i = 13007:13021
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12932, 15);        
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9, csi-rs on symbol 13, 8
            CFG(row_count, 1:7) = { i    0        10      {}        {72}     num2cell(2369:2384)     {82 83 84 111}};
        else  % odd cell: trs symbol 6,7, csi-rs on symbol 12,1
            CFG(row_count, 1:7) = { i    0        10      {}        {72}     num2cell(2369:2384)     {91 110 93 112}};
        end
        TrsBeamIdxMap(i) = 0;
        CsiBeamIdxMap(i) = [1, 2, 3, 4];
        CsiZpBeamIdxMap(i) = [5, 6, 7, 8];
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {21}, TRS, TRS index 9/9
    for i = 13022:13036
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12932, 15);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        10      {}        {72,71}     num2cell(2337:2352)     {82 83}};
        else  % odd cell: trs symbol 6,7
            CFG(row_count, 1:7) = { i    0        10      {}        {72,71}     num2cell(2337:2352)     {91 110}};
        end
        TrsBeamIdxMap(i) = 9;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {22} is the same with {1,2}
    % slot {23} is the same with {3}
    % slot {26,27,28,29,30} is the same with {6,7,8,9}
    % slot {31,32} is the same with {1,2}
    % slot {33} is the same with {3}
    % slot {36,37,38,39} is the same with {6,7,8,9}
    %%%%% frame 2: 
    % slot {40} is the same with {0}
    % slot {41,42} is the same with {1,2}
    % slot {43} is the same with {3}
    % slot {46,47,48,49} is the same with {6,7,8,9}
    % slot {50} is the same with {10}
    % slot {51,52} is the same with {1,2}
    % slot {53} is the same with {3}
    % slot {56,57,58,59} is the same with {6,7,8,9}
    %%%%% frame 3: 
    % slot {60}, TRS, TRS index 10/10
    for i = 13037:13051
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12932, 15);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        10      {}         {72}      num2cell(2369:2384)     {82 83}};
        else  % odd cell: trs symbol 6,7
            CFG(row_count, 1:7) = { i    0        10      {}         {72}      num2cell(2369:2384)     {91 110}};
        end
        TrsBeamIdxMap(i) = 10;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {61}, TRS, TRS index 11/11
    for i = 13052:13066
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-12932, 15);
        if mod(CellIdxInPatternMap(i),2) == 0  % even cell: trs symbol 5,9
            CFG(row_count, 1:7) = { i    0        10      {}        {72,71}     num2cell(2337:2352)     {82 83}};
        else  % odd cell: trs symbol 6,7
            CFG(row_count, 1:7) = { i    0        10      {}        {72,71}     num2cell(2337:2352)     {91 110}};
        end
        TrsBeamIdxMap(i) = 11;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {63} is the same with {3}
    % slot {66,67,68,69,70} is the same with {6,7,8,9}
    % slot {71,72} is the same with {1,2}
    % slot {73} is the same with {3}
    % slot {76,77,78,79} is the same with {6,7,8,9}
    % end pattern 77: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case column I TC, 15C, based on 90638

                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 79: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case Ph4 column B TC, 15C, based on 90640
    row_count = length(CFG); 
    %%%%% frame 0:         
    % slot {0}, 4 ssb, index 0~3   
    for i = 13067:13081
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {22}        {72}    num2cell(2321:2336)    {}};
        SsbBeamIdxMap(i) = [0 1 2 3];
        CellIdxInPatternMap(i) = mod(i-13067, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {1,2} 
    for i = 13082:13096
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10       {}         {72,71}    num2cell(2337:2352)    {}};
        CellIdxInPatternMap(i) = mod(i-13067, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {3}
    for i = 13097:13111
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {72}     num2cell(2353:2368)     {}};
        CellIdxInPatternMap(i) = mod(i-13067, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {6,7,8,9}
    for i = 13112:13126
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {72}     num2cell(2369:2384)     {}};
        CellIdxInPatternMap(i) = mod(i-13067, 15);
        pdschDynamicBfMap(i) = 1;
    end 
    % slot {10}
    for i = 13127:13141
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}       {72,70}    num2cell([2385:2400,847])     {}};
        CellIdxInPatternMap(i) = mod(i-13067, 15);
        pdschDynamicBfMap(i) = 1;
    end 
    % slot {11,12} is the same with {1,2}
    % slot {13} is the same with {3}
    % slot {16,17,18,19} is the same with {6,7,8,9}
    %%%%% frame 1:  
    % slot {20}, TRS, TRS index 0
    for i = 13142:13156
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13067, 15);        
        % all cells: trs symbol 5,9
        CFG(row_count, 1:7) = { i    0        10      {}        {72}     num2cell(2369:2384)     {82 83}};
        TrsBeamIdxMap(i) = 0;
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {21}, TRS, TRS index 1
    for i = 13157:13171
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13067, 15);
        % all cells: trs symbol 5,9
        CFG(row_count, 1:7) = { i    0        10      {}        {72,71}     num2cell(2337:2352)     {82 83}};
        TrsBeamIdxMap(i) = 1;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {22} is the same with {1,2}
    % slot {23}, NZP CSI-RS index 4~11, CSI_RS ZP index 12~15
    for i = 13172:13186
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13067, 15);        
        % all cells: NZP csi-rs on symbol 4,5, ZP csi-rs on symbol 2,3
        CFG(row_count, 1:7) = { i    0        10      {}          {}         {}     {113 114}};
        CsiBeamIdxMap(i) = [4, 5, 6, 7, 8, 9, 10, 11];
        CsiZpBeamIdxMap(i) = [12, 13, 14, 15];
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {26,27,28,29,30} is the same with {6,7,8,9}
    % slot {31,32} is the same with {1,2}
    % slot {33} is the same with {3}
    % slot {36,37,38,39} is the same with {6,7,8,9}
    %%%%% frame 2: 
    % slot {40} is the same with {0}
    % slot {41,42} is the same with {1,2}
    % slot {43} is the same with {3}
    % slot {46,47,48,49} is the same with {6,7,8,9}
    % slot {50} is the same with {10}
    % slot {51,52} is the same with {1,2}
    % slot {53} is the same with {3}
    % slot {56,57,58,59} is the same with {6,7,8,9}
    %%%%% frame 3: 
    % slot {60}, TRS, TRS index 2
    for i = 13187:13201
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13067, 15);
        % all cells: trs symbol 5,9
        CFG(row_count, 1:7) = { i    0        10      {}         {72}      num2cell(2369:2384)     {82 83}};
        TrsBeamIdxMap(i) = 2;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {61}, TRS, TRS index 3
    for i = 13202:13216
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13067, 15);
        % all cells: trs symbol 5,9
        CFG(row_count, 1:7) = { i    0        10      {}        {72,71}     num2cell(2337:2352)     {82 83}};
        TrsBeamIdxMap(i) = 3;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {63} is the same with {3}
    % slot {66,67,68,69,70} is the same with {6,7,8,9}
    % slot {71,72} is the same with {1,2}
    % slot {73} is the same with {3}
    % slot {76,77,78,79} is the same with {6,7,8,9}
    % end pattern 79: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case Ph4 column B TC, 15C, based on 90640

                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 81: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR 25-3 column B TC, 15C
    row_count = length(CFG); 
    %%%%% frame 0:         
    % slot {0}, 4 ssb, index 0~3   
    for i = 13217:13231
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {22}        {72}    num2cell(2321:2336)    {}};
        SsbBeamIdxMap(i) = [0 1 2 3];
        CellIdxInPatternMap(i) = mod(i-13217, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {1,2} 
    for i = 13232:13246
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10       {}         {72,71}    num2cell(2337:2352)    {}};
        CellIdxInPatternMap(i) = mod(i-13217, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {3}
    for i = 13247:13261
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {72}     num2cell(2353:2368)     {}};
        CellIdxInPatternMap(i) = mod(i-13217, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {6,7,8,9}
    for i = 13262:13276
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {72}     num2cell(2369:2384)     {}};
        CellIdxInPatternMap(i) = mod(i-13217, 15);
        pdschDynamicBfMap(i) = 1;
    end 
    % slot {10}
    for i = 13277:13291
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}       {72,70}    num2cell([2385:2400,847])     {}};
        CellIdxInPatternMap(i) = mod(i-13217, 15);
        pdschDynamicBfMap(i) = 1;
    end 
    % slot {11,12} is the same with {1,2}
    % slot {13} is the same with {3}
    % slot {16,17,18,19} is the same with {6,7,8,9}
    %%%%% frame 1:  
    % slot {20}, TRS, TRS index 0
    for i = 13292:13306
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13217, 15);        
        % all cells: trs symbol 5,9
        CFG(row_count, 1:7) = { i    0        10      {}        {72}     num2cell(2369:2384)     {82 83}};
        TrsBeamIdxMap(i) = 0;
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {21}, TRS, TRS index 1
    for i = 13307:13321
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13217, 15);
        % all cells: trs symbol 5,9
        CFG(row_count, 1:7) = { i    0        10      {}        {72,71}     num2cell(2337:2352)     {82 83}};
        TrsBeamIdxMap(i) = 1;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {22} is the same with {1,2}
    % slot {23}, NZP CSI-RS index 2~9, CSI_RS ZP index 10~13, for pattern 81 a/b
    for i = 13322:13336
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13217, 15);        
        % all cells: NZP csi-rs on symbol 4,5, ZP csi-rs on symbol 2,3
        CFG(row_count, 1:7) = { i    0        10      {}          {}         {}     {113 114}};
        CsiBeamIdxMap(i) = [2, 3, 4, 5, 6, 7, 8, 9];
        CsiZpBeamIdxMap(i) = [10, 11, 12, 13];
        pdschDynamicBfMap(i) = 1;       
    end
    % slot {26,27,28,29,30} is the same with {6,7,8,9}
    % slot {31,32} is the same with {1,2}
    % slot {33} is the same with {3}
    % slot {36,37,38,39} is the same with {6,7,8,9}
    %%%%% frame 2: 
    % slot {40} is the same with {0}
    % slot {41,42} is the same with {1,2}
    % slot {43} is the same with {3}
    % slot {46,47,48,49} is the same with {6,7,8,9}
    % slot {50} is the same with {10}
    % slot {51,52} is the same with {1,2}
    % slot {53} is the same with {3}
    % slot {56,57,58,59} is the same with {6,7,8,9}
    %%%%% frame 3: 
    % slot {60}, TRS, TRS index 38
    for i = 13337:13351
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13217, 15);
        % all cells: trs symbol 5,9
        CFG(row_count, 1:7) = { i    0        10      {}         {72}      num2cell(2369:2384)     {82 83}};
        TrsBeamIdxMap(i) = 38;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {61}, TRS, TRS index 39
    for i = 13352:13366
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13217, 15);
        % all cells: trs symbol 5,9
        CFG(row_count, 1:7) = { i    0        10      {}        {72,71}     num2cell(2337:2352)     {82 83}};
        TrsBeamIdxMap(i) = 39;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {63} is the same with {3}
    % slot {66,67,68,69,70} is the same with {6,7,8,9}
    % slot {71,72} is the same with {1,2}
    % slot {73} is the same with {3}
    % slot {76,77,78,79} is the same with {6,7,8,9}
    % slot {23}, NZP CSI-RS index 2~41, CSI_RS ZP index 42~45, for pattern 81 c/d
    for i = 13367:13381
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13217, 15);        
        % all cells: NZP csi-rs on symbol 2,3,4,5, ZP csi-rs on symbol 2,3
        CFG(row_count, 1:7) = { i    0        10      {}          {}         {}     {119 120 114}};
        CsiBeamIdxMap(i) = [2:41];  % 8 ports + 32 ports = 40 ports
        CsiZpBeamIdxMap(i) = [42:45];
        pdschDynamicBfMap(i) = 1;    
        enableCsiRs32PortsMap(i) = 1;
    end
    % end pattern 81: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR 25-3 column B TC, 15C

                            % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 83: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR 25-3 column G TC, 15C, 64 UEs per TTI
    row_count = length(CFG); 
    %%%%% frame 0:         
    % slot {0}, 4 ssb, index 0~3   
    for i = 13382:13396
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {22}        {72}    num2cell(2401:2464)    {}};
        SsbBeamIdxMap(i) = [0 1 2 3];
        CellIdxInPatternMap(i) = mod(i-13382, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {1,2} 
    for i = 13397:13411
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10       {}         {72,71}    num2cell(2465:2528)    {}};
        CellIdxInPatternMap(i) = mod(i-13382, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {3}
    for i = 13412:13426
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {72}     num2cell(2529:2592)     {}};
        CellIdxInPatternMap(i) = mod(i-13382, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {6,7,8,9}
    for i = 13427:13441
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {72}     num2cell(2593:2656)     {}};
        CellIdxInPatternMap(i) = mod(i-13382, 15);
        pdschDynamicBfMap(i) = 1;
    end 
    % slot {10}
    for i = 13442:13456
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}       {72,70}    num2cell([2657:2720,887])     {}};
        CellIdxInPatternMap(i) = mod(i-13382, 15);
        pdschDynamicBfMap(i) = 1;
    end 
    % slot {11,12} is the same with {1,2}
    % slot {13} is the same with {3}
    % slot {16,17,18,19} is the same with {6,7,8,9}
    %%%%% frame 1:  
    % slot {20}, TRS, TRS index 0
    for i = 13457:13471
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13382, 15);        
        % all cells: trs symbol 5,9
        CFG(row_count, 1:7) = { i    0        10      {}        {72}     num2cell(2593:2656)     {82 83}};
        TrsBeamIdxMap(i) = 0;
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {21}, TRS, TRS index 1
    for i = 13472:13486
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13382, 15);
        % all cells: trs symbol 5,9
        CFG(row_count, 1:7) = { i    0        10      {}        {72,71}     num2cell(2465:2528)     {82 83}};
        TrsBeamIdxMap(i) = 1;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {22} is the same with {1,2}
    % slot {23}, NZP CSI-RS index 2~9, CSI_RS ZP index 10~13, for pattern 83 a/b
    for i = 13487:13501
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13382, 15);        
        % all cells: NZP csi-rs on symbol 4,5, ZP csi-rs on symbol 2,3
        CFG(row_count, 1:7) = { i    0        10      {}          {}         {}     {113 114}};
        CsiBeamIdxMap(i) = [2, 3, 4, 5, 6, 7, 8, 9];
        CsiZpBeamIdxMap(i) = [10, 11, 12, 13];
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {26,27,28,29,30} is the same with {6,7,8,9}
    % slot {31,32} is the same with {1,2}
    % slot {33} is the same with {3}
    % slot {36,37,38,39} is the same with {6,7,8,9}
    %%%%% frame 2: 
    % slot {40} is the same with {0}
    % slot {41,42} is the same with {1,2}
    % slot {43} is the same with {3}
    % slot {46,47,48,49} is the same with {6,7,8,9}
    % slot {50} is the same with {10}
    % slot {51,52} is the same with {1,2}
    % slot {53} is the same with {3}
    % slot {56,57,58,59} is the same with {6,7,8,9}
    %%%%% frame 3: 
    % slot {60}, TRS, TRS index 38
    for i = 13502:13516
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13382, 15);
        % all cells: trs symbol 5,9
        CFG(row_count, 1:7) = { i    0        10      {}         {72}      num2cell(2593:2656)     {82 83}};
        TrsBeamIdxMap(i) = 38;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {61}, TRS, TRS index 39
    for i = 13517:13531
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13382, 15);
        % all cells: trs symbol 5,9
        CFG(row_count, 1:7) = { i    0        10      {}        {72,71}     num2cell(2465:2528)     {82 83}};
        TrsBeamIdxMap(i) = 39;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {63} is the same with {3}
    % slot {66,67,68,69,70} is the same with {6,7,8,9}
    % slot {71,72} is the same with {1,2}
    % slot {73} is the same with {3}
    % slot {76,77,78,79} is the same with {6,7,8,9}
    % slot {23}, NZP CSI-RS index 2~41, CSI_RS ZP index 42~45, for pattern 83 c/d
    for i = 13532:13546
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13382, 15);        
        % all cells: NZP csi-rs on symbol 2,3,4,5, ZP csi-rs on symbol 2,3
        CFG(row_count, 1:7) = { i    0        10      {}          {}         {}     {119 120 114}};
        CsiBeamIdxMap(i) = [2:41];  % 8 ports + 32 ports = 40 ports
        CsiZpBeamIdxMap(i) = [42:45];
        enableCsiRs32PortsMap(i) = 1;
    end
    % end pattern 83: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR 25-3 column G TC, 15C, 64 UEs per TTI
    
                                % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 85: 64TR peak, 100 MHz (273 PRBs), OTA, 25-3 column E TC, 15C, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEGs, 8 UEs per slot, 1 layer per UE, 32 DL layer with more PUCCH, SRS in S slot (2 symbols)
    row_count = length(CFG); 
    %%%%% frame 0:         
    % slot {0}, 4 ssb, index 0~3   
    for i = 13547:13561
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {22}        {80}    num2cell(2721:2752)    {}};
        SsbBeamIdxMap(i) = [0 1 2 3];
        CellIdxInPatternMap(i) = mod(i-13547, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {1,2} 
    for i = 13562:13576
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10       {}         {80,71}    num2cell(2753:2784)    {}};
        CellIdxInPatternMap(i) = mod(i-13547, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {3}
    for i = 13577:13591
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {80}     num2cell(2785:2816)     {}};
        CellIdxInPatternMap(i) = mod(i-13547, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {6,7,8,9}
    for i = 13592:13606
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {80}     num2cell(2817:2848)     {}};
        CellIdxInPatternMap(i) = mod(i-13547, 15);
        pdschDynamicBfMap(i) = 1;
    end 
    % slot {10}
    for i = 13607:13621
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}       {80,70}    num2cell([2849:2880,847])     {}};
        CellIdxInPatternMap(i) = mod(i-13547, 15);
        pdschDynamicBfMap(i) = 1;
    end 
    % slot {11,12} is the same with {1,2}
    % slot {13} is the same with {3}
    % slot {16,17,18,19} is the same with {6,7,8,9}
    %%%%% frame 1:  
    % slot {20}, TRS, TRS index 0
    for i = 13622:13636
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13547, 15);        
        % all cells: trs symbol 5,9
        CFG(row_count, 1:7) = { i    0        10      {}        {80}     num2cell(2817:2848)     {82 83}};
        TrsBeamIdxMap(i) = 0;
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {21}, TRS, TRS index 1
    for i = 13637:13651
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13547, 15);
        % all cells: trs symbol 5,9
        CFG(row_count, 1:7) = { i    0        10      {}        {80,71}     num2cell(2753:2784)     {82 83}};
        TrsBeamIdxMap(i) = 1;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {22} is the same with {1,2}
    % slot {23}, NZP CSI-RS index 2~9, CSI_RS ZP index 10~13
    for i = 13652:13666
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13547, 15);        
        % all cells: NZP csi-rs on symbol 4,5, ZP csi-rs on symbol 2,3
        CFG(row_count, 1:7) = { i    0        10      {}          {}         {}     {113 114}};
        CsiBeamIdxMap(i) = [2, 3, 4, 5, 6, 7, 8, 9];
        CsiZpBeamIdxMap(i) = [10, 11, 12, 13];
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {26,27,28,29,30} is the same with {6,7,8,9}
    % slot {31,32} is the same with {1,2}
    % slot {33} is the same with {3}
    % slot {36,37,38,39} is the same with {6,7,8,9}
    %%%%% frame 2: 
    % slot {40} is the same with {0}
    % slot {41,42} is the same with {1,2}
    % slot {43} is the same with {3}
    % slot {46,47,48,49} is the same with {6,7,8,9}
    % slot {50} is the same with {10}
    % slot {51,52} is the same with {1,2}
    % slot {53} is the same with {3}
    % slot {56,57,58,59} is the same with {6,7,8,9}
    %%%%% frame 3: 
    % slot {60}, TRS, TRS index 14
    for i = 13667:13681
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13547, 15);
        % all cells: trs symbol 5,9
        CFG(row_count, 1:7) = { i    0        10      {}         {80}      num2cell(2817:2848)     {82 83}};
        TrsBeamIdxMap(i) = 14;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {61}, TRS, TRS index 15
    for i = 13682:13696
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13547, 15);
        % all cells: trs symbol 5,9
        CFG(row_count, 1:7) = { i    0        10      {}        {80,71}     num2cell(2753:2784)     {82 83}};
        TrsBeamIdxMap(i) = 15;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {63} is the same with {3}
    % slot {66,67,68,69,70} is the same with {6,7,8,9}
    % slot {71,72} is the same with {1,2}
    % slot {73} is the same with {3}
    % slot {76,77,78,79} is the same with {6,7,8,9}
    % end pattern 85: 64TR peak, 100 MHz (273 PRBs), OTA, 25-3 column E TC, 15C, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEGs, 8 UEs per slot, 1 layer per UE, 32 DL layer with more PUCCH, SRS in S slot (2 symbols)

    % pattern 87: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case Ph4 column B TC, 32 port CSI-RS, 15C, based on 90641
    % reuse all TVs from pattern 79 except for slot {23} with NZP+ZP CSI-RS
    % pattern 79 is 8 port and pattern 87 is 32 port
    % slot {23}, NZP CSI-RS index 4~35, CSI_RS ZP index 36~67
    for i = 13697:13711
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13067, 15);        
        % all cells: NZP csi-rs on symbol 4,5, ZP csi-rs on symbol 2,3
        CFG(row_count, 1:7) = { i    0        10      {}          {}         {}     {115 116}};
        CsiBeamIdxMap(i) = [4:35];
        CsiZpBeamIdxMap(i) = [36:67];
        enableCsiRs32PortsMap(i) = 1;
    end
    % end pattern 87: 64TR peak, 100 MHz (273 PRBs), OTA, 64TR worst case Ph4 column B TC, 32 port CSI-RS, 15C, based on 90641

                                    % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 89: 64TR peak, Mixed 100 MHz (273 PRBs) + 90 MHz (244 PRBs) + 60 MHz (160 PRBs), OTA, realistic traffic, 9C, PUSCH CP-OFDM, Mixed PUCCH F1/F3. Based on 90655,90656,90658~90661
    row_count = length(CFG);
    
    %%%%% frame 0:
    % slot {0}, SSB + PDCCH + PDSCH set 0
    for i = 13712:13720
        row_count = row_count + 1;
        cellIdx = mod(i-13712, 9);
        switch cellIdx
            case 0  % 100 MHz heavy
                CFG(row_count, 1:7) = { i    0        10      {22}   {72}    num2cell(888:907)     {}};
            case 1  % 100 MHz light
                CFG(row_count, 1:7) = { i    0        10      {22}   {72}    num2cell(958:969)     {}};
            case 2  % 100 MHz light
                CFG(row_count, 1:7) = { i    0        10      {22}   {72}    num2cell(958:969)     {}};
            case 3  % 90 MHz heavy
                CFG(row_count, 1:7) = { i    0        14      {22}   {83}    num2cell(1000:1019)   {}};
            case 4  % 90 MHz light
                CFG(row_count, 1:7) = { i    0        14      {22}   {83}    num2cell(1070:1081)   {}};
            case 5  % 90 MHz light
                CFG(row_count, 1:7) = { i    0        14      {22}   {83}    num2cell(1070:1081)   {}};
            case 6  % 60 MHz heavy
                CFG(row_count, 1:7) = { i    0        15      {22}   {86}    num2cell(1112:1131)   {}};
            case 7  % 60 MHz light
                CFG(row_count, 1:7) = { i    0        15      {22}   {86}    num2cell(1182:1193)   {}};
            case 8  % 60 MHz light
                CFG(row_count, 1:7) = { i    0        15      {22}   {86}    num2cell(1182:1193)   {}};
        end
        SsbBeamIdxMap(i) = [0 1 2 3];
        CellIdxInPatternMap(i) = cellIdx;
        pdschDynamicBfMap(i) = 1;
    end
    
    % slot {1,2}, PDCCH + PDSCH set 1
    for i = 13721:13729
        row_count = row_count + 1;
        cellIdx = mod(i-13712, 9);
        switch cellIdx
            case 0  % 100 MHz heavy
                CFG(row_count, 1:7) = { i    0        10       {}   {72 71}    num2cell(908:927)     {}};
            case 1  % 100 MHz light
                CFG(row_count, 1:7) = { i    0        10       {}   {72 71}    num2cell(970:981)     {}};
            case 2  % 100 MHz light
                CFG(row_count, 1:7) = { i    0        10       {}   {72 71}    num2cell(970:981)     {}};
            case 3  % 90 MHz heavy
                CFG(row_count, 1:7) = { i    0        14       {}   {83 82}    num2cell(1020:1039)   {}};
            case 4  % 90 MHz light
                CFG(row_count, 1:7) = { i    0        14       {}   {83 82}    num2cell(1082:1093)   {}};
            case 5  % 90 MHz light
                CFG(row_count, 1:7) = { i    0        14       {}   {83 82}    num2cell(1082:1093)   {}};
            case 6  % 60 MHz heavy
                CFG(row_count, 1:7) = { i    0        15       {}   {86 85}    num2cell(1132:1151)   {}};
            case 7  % 60 MHz light
                CFG(row_count, 1:7) = { i    0        15       {}   {86 85}    num2cell(1194:1205)   {}};
            case 8  % 60 MHz light
                CFG(row_count, 1:7) = { i    0        15       {}   {86 85}    num2cell(1194:1205)   {}};
        end
        CellIdxInPatternMap(i) = cellIdx;
        pdschDynamicBfMap(i) = 1;
    end
    
    % slot {3}, PDCCH + PDSCH set 2
    for i = 13730:13738
        row_count = row_count + 1;
        cellIdx = mod(i-13712, 9);
        switch cellIdx
            case 0  % 100 MHz heavy
                CFG(row_count, 1:7) = { i    0        10      {}   {72}     num2cell(928:947)     {}};
            case 1  % 100 MHz light
                CFG(row_count, 1:7) = { i    0        10      {}   {72}     num2cell(982:993)     {}};
            case 2  % 100 MHz light
                CFG(row_count, 1:7) = { i    0        10      {}   {72}     num2cell(982:993)     {}};
            case 3  % 90 MHz heavy
                CFG(row_count, 1:7) = { i    0        14      {}   {83}     num2cell(1040:1059)   {}};
            case 4  % 90 MHz light
                CFG(row_count, 1:7) = { i    0        14      {}   {83}     num2cell(1094:1105)   {}};
            case 5  % 90 MHz light
                CFG(row_count, 1:7) = { i    0        14      {}   {83}     num2cell(1094:1105)   {}};
            case 6  % 60 MHz heavy
                CFG(row_count, 1:7) = { i    0        15      {}   {86}     num2cell(1152:1171)   {}};
            case 7  % 60 MHz light
                CFG(row_count, 1:7) = { i    0        15      {}   {86}     num2cell(1206:1217)   {}};
            case 8  % 60 MHz light
                CFG(row_count, 1:7) = { i    0        15      {}   {86}     num2cell(1206:1217)   {}};
        end
        CellIdxInPatternMap(i) = cellIdx;
        pdschDynamicBfMap(i) = 1;
    end
    
    % slot {6,7,8,9}, PDCCH + PDSCH set 3
    for i = 13739:13747
        row_count = row_count + 1;
        cellIdx = mod(i-13712, 9);
        switch cellIdx
            case 0  % 100 MHz heavy
                CFG(row_count, 1:7) = { i    0        10      {}   {72}     num2cell([948:955, 896:907])         {}};
            case 1  % 100 MHz light
                CFG(row_count, 1:7) = { i    0        10      {}   {72}     num2cell([994:997, 962:969])         {}};
            case 2  % 100 MHz light
                CFG(row_count, 1:7) = { i    0        10      {}   {72}     num2cell([994:997, 962:969])         {}};
            case 3  % 90 MHz heavy
                CFG(row_count, 1:7) = { i    0        14      {}   {83}     num2cell([1060:1067, 1008:1019])     {}};
            case 4  % 90 MHz light
                CFG(row_count, 1:7) = { i    0        14      {}   {83}     num2cell([1106:1109, 1074:1081])     {}};
            case 5  % 90 MHz light
                CFG(row_count, 1:7) = { i    0        14      {}   {83}     num2cell([1106:1109, 1074:1081])     {}};
            case 6  % 60 MHz heavy
                CFG(row_count, 1:7) = { i    0        15      {}   {86}     num2cell([1172:1179, 1120:1131])     {}};
            case 7  % 60 MHz light
                CFG(row_count, 1:7) = { i    0        15      {}   {86}     num2cell([1218:1221, 1186:1193])     {}};
            case 8  % 60 MHz light
                CFG(row_count, 1:7) = { i    0        15      {}   {86}     num2cell([1218:1221, 1186:1193])     {}};
        end
        CellIdxInPatternMap(i) = cellIdx;
        pdschDynamicBfMap(i) = 1;
    end
    
    % slot {10}, PDCCH + CCH PDSCH
    for i = 13748:13756
        row_count = row_count + 1;
        cellIdx = mod(i-13712, 9);
        switch cellIdx
            case 0  % 100 MHz heavy
                CFG(row_count, 1:7) = { i    0        10      {}   {72 70}   num2cell([948:955, 896:906, 956, 957])         {}};
            case 1  % 100 MHz light
                CFG(row_count, 1:7) = { i    0        10      {}   {72 70}   num2cell([994:997, 962:968, 998, 999])         {}};
            case 2  % 100 MHz light
                CFG(row_count, 1:7) = { i    0        10      {}   {72 70}   num2cell([994:997, 962:968, 998, 999])         {}};
            case 3  % 90 MHz heavy
                CFG(row_count, 1:7) = { i    0        14      {}   {83 81}   num2cell([1060:1067, 1008:1018, 1068, 1069])   {}};
            case 4  % 90 MHz light
                CFG(row_count, 1:7) = { i    0        14      {}   {83 81}   num2cell([1106:1109, 1074:1080, 1110, 1111])   {}};
            case 5  % 90 MHz light
                CFG(row_count, 1:7) = { i    0        14      {}   {83 81}   num2cell([1106:1109, 1074:1080, 1110, 1111])   {}};
            case 6  % 60 MHz heavy
                CFG(row_count, 1:7) = { i    0        15      {}   {86 84}   num2cell([1172:1179, 1120:1130, 1180, 1181])   {}};
            case 7  % 60 MHz light
                CFG(row_count, 1:7) = { i    0        15      {}   {86 84}   num2cell([1218:1221, 1186:1192, 1222, 1223])   {}};
            case 8  % 60 MHz light
                CFG(row_count, 1:7) = { i    0        15      {}   {86 84}   num2cell([1218:1221, 1186:1192, 1222, 1223])   {}};
        end
        CellIdxInPatternMap(i) = cellIdx;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {11,12} is the same with {1,2}
    % slot {13} is the same with {3}
    % slot {16,17,18,19} is the same with {6,7,8,9}
    
    %%%%% frame 1:
    % slot {20}, TRS, TRS index 0
    for i = 13757:13765
        row_count = row_count + 1;
        cellIdx = mod(i-13712, 9);
        switch cellIdx
            case 0  % 100 MHz heavy
                CFG(row_count, 1:7) = { i    0        10      {}   {72}     num2cell([948:955, 896:907])         {82 83}};
            case 1  % 100 MHz light
                CFG(row_count, 1:7) = { i    0        10      {}   {72}     num2cell([994:997, 962:969])         {82 83}};
            case 2  % 100 MHz light
                CFG(row_count, 1:7) = { i    0        10      {}   {72}     num2cell([994:997, 962:969])         {82 83}};
            case 3  % 90 MHz heavy
                CFG(row_count, 1:7) = { i    0        14      {}   {83}     num2cell([1060:1067, 1008:1019])     {122 123}};
            case 4  % 90 MHz light
                CFG(row_count, 1:7) = { i    0        14      {}   {83}     num2cell([1106:1109, 1074:1081])     {122 123}};
            case 5  % 90 MHz light
                CFG(row_count, 1:7) = { i    0        14      {}   {83}     num2cell([1106:1109, 1074:1081])     {122 123}};
            case 6  % 60 MHz heavy
                CFG(row_count, 1:7) = { i    0        15      {}   {86}     num2cell([1172:1179, 1120:1131])     {128 129}};
            case 7  % 60 MHz light
                CFG(row_count, 1:7) = { i    0        15      {}   {86}     num2cell([1218:1221, 1186:1193])     {128 129}};
            case 8  % 60 MHz light
                CFG(row_count, 1:7) = { i    0        15      {}   {86}     num2cell([1218:1221, 1186:1193])     {128 129}};
        end
        TrsBeamIdxMap(i) = 0;
        CellIdxInPatternMap(i) = cellIdx;
        pdschDynamicBfMap(i) = 1;
    end
    
    % slot {21}, TRS, TRS index 1
    for i = 13766:13774
        row_count = row_count + 1;
        cellIdx = mod(i-13712, 9);
        switch cellIdx
            case 0  % 100 MHz heavy
                CFG(row_count, 1:7) = { i    0        10      {}   {72 71}     num2cell(908:927)       {82 83}};
            case 1  % 100 MHz light
                CFG(row_count, 1:7) = { i    0        10      {}   {72 71}     num2cell(970:981)       {82 83}};
            case 2  % 100 MHz light
                CFG(row_count, 1:7) = { i    0        10      {}   {72 71}     num2cell(970:981)       {82 83}};
            case 3  % 90 MHz heavy
                CFG(row_count, 1:7) = { i    0        14      {}   {83 82}     num2cell(1020:1039)     {122 123}};
            case 4  % 90 MHz light
                CFG(row_count, 1:7) = { i    0        14      {}   {83 82}     num2cell(1082:1093)     {122 123}};
            case 5  % 90 MHz light
                CFG(row_count, 1:7) = { i    0        14      {}   {83 82}     num2cell(1082:1093)     {122 123}};
            case 6  % 60 MHz heavy
                CFG(row_count, 1:7) = { i    0        15      {}   {86 85}     num2cell(1132:1151)     {128 129}};
            case 7  % 60 MHz light
                CFG(row_count, 1:7) = { i    0        15      {}   {86 85}     num2cell(1194:1205)     {128 129}};
            case 8  % 60 MHz light
                CFG(row_count, 1:7) = { i    0        15      {}   {86 85}     num2cell(1194:1205)     {128 129}};
        end
        TrsBeamIdxMap(i) = 1;
        CellIdxInPatternMap(i) = cellIdx;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {22} is the same with {1,2}
    
    % slot {23}, NZP CSI-RS, CSI_RS ZP
    for i = 13775:13783
        row_count = row_count + 1;
        cellIdx = mod(i-13712, 9);
        switch cellIdx
            case 0  % 100 MHz heavy
                CFG(row_count, 1:7) = { i    0        10      {}          {}         {}     {113 114}};
            case 1  % 100 MHz light
                CFG(row_count, 1:7) = { i    0        10      {}          {}         {}     {113 114}};
            case 2  % 100 MHz light
                CFG(row_count, 1:7) = { i    0        10      {}          {}         {}     {113 114}};
            case 3  % 90 MHz heavy
                CFG(row_count, 1:7) = { i    0        14      {}          {}         {}     {124 125}};
            case 4  % 90 MHz light
                CFG(row_count, 1:7) = { i    0        14      {}          {}         {}     {124 125}};
            case 5  % 90 MHz light
                CFG(row_count, 1:7) = { i    0        14      {}          {}         {}     {124 125}};
            case 6  % 60 MHz heavy
                CFG(row_count, 1:7) = { i    0        15      {}          {}         {}     {130 131}};
            case 7  % 60 MHz light
                CFG(row_count, 1:7) = { i    0        15      {}          {}         {}     {130 131}};
            case 8  % 60 MHz light
                CFG(row_count, 1:7) = { i    0        15      {}          {}         {}     {130 131}};
        end
        CsiBeamIdxMap(i) = [4, 5, 6, 7, 8, 9, 10, 11];
        CsiZpBeamIdxMap(i) = [12, 13, 14, 15];
        CellIdxInPatternMap(i) = cellIdx;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {26,27,28,29,30} is the same with {6,7,8,9}
    % slot {31,32} is the same with {1,2}
    % slot {33} is the same with {3}
    % slot {36,37,38,39} is the same with {6,7,8,9}
    
    %%%%% frame 2:
    % slot {40} is the same with {0}
    % slot {41,42} is the same with {1,2}
    % slot {43} is the same with {3}
    % slot {46,47,48,49} is the same with {6,7,8,9}
    % slot {50} is the same with {10}
    % slot {51,52} is the same with {1,2}
    % slot {53} is the same with {3}
    % slot {56,57,58,59} is the same with {6,7,8,9}
    
    %%%%% frame 3:
    % slot {60}, TRS, TRS index 2
    for i = 13784:13792
        row_count = row_count + 1;
        cellIdx = mod(i-13712, 9);
        switch cellIdx
            case 0  % 100 MHz heavy
                CFG(row_count, 1:7) = { i    0        10      {}   {72}     num2cell([948:955, 896:907])         {82 83}};
            case 1  % 100 MHz light
                CFG(row_count, 1:7) = { i    0        10      {}   {72}     num2cell([994:997, 962:969])         {82 83}};
            case 2  % 100 MHz light
                CFG(row_count, 1:7) = { i    0        10      {}   {72}     num2cell([994:997, 962:969])         {82 83}};
            case 3  % 90 MHz heavy
                CFG(row_count, 1:7) = { i    0        14      {}   {83}     num2cell([1060:1067, 1008:1019])     {122 123}};
            case 4  % 90 MHz light
                CFG(row_count, 1:7) = { i    0        14      {}   {83}     num2cell([1106:1109, 1074:1081])     {122 123}};
            case 5  % 90 MHz light
                CFG(row_count, 1:7) = { i    0        14      {}   {83}     num2cell([1106:1109, 1074:1081])     {122 123}};
            case 6  % 60 MHz heavy
                CFG(row_count, 1:7) = { i    0        15      {}   {86}     num2cell([1172:1179, 1120:1131])     {128 129}};
            case 7  % 60 MHz light
                CFG(row_count, 1:7) = { i    0        15      {}   {86}     num2cell([1218:1221, 1186:1193])     {128 129}};
            case 8  % 60 MHz light
                CFG(row_count, 1:7) = { i    0        15      {}   {86}     num2cell([1218:1221, 1186:1193])     {128 129}};
        end
        TrsBeamIdxMap(i) = 2;
        CellIdxInPatternMap(i) = cellIdx;
        pdschDynamicBfMap(i) = 1;
    end
    
    % slot {61}, TRS, TRS index 3
    for i = 13793:13801
        row_count = row_count + 1;
        cellIdx = mod(i-13712, 9);
        switch cellIdx
            case 0  % 100 MHz heavy
                CFG(row_count, 1:7) = { i    0        10      {}   {72 71}     num2cell(908:927)       {82 83}};
            case 1  % 100 MHz light
                CFG(row_count, 1:7) = { i    0        10      {}   {72 71}     num2cell(970:981)       {82 83}};
            case 2  % 100 MHz light
                CFG(row_count, 1:7) = { i    0        10      {}   {72 71}     num2cell(970:981)       {82 83}};
            case 3  % 90 MHz heavy
                CFG(row_count, 1:7) = { i    0        14      {}   {83 82}     num2cell(1020:1039)     {122 123}};
            case 4  % 90 MHz light
                CFG(row_count, 1:7) = { i    0        14      {}   {83 82}     num2cell(1082:1093)     {122 123}};
            case 5  % 90 MHz light
                CFG(row_count, 1:7) = { i    0        14      {}   {83 82}     num2cell(1082:1093)     {122 123}};
            case 6  % 60 MHz heavy
                CFG(row_count, 1:7) = { i    0        15      {}   {86 85}     num2cell(1132:1151)     {128 129}};
            case 7  % 60 MHz light
                CFG(row_count, 1:7) = { i    0        15      {}   {86 85}     num2cell(1194:1205)     {128 129}};
            case 8  % 60 MHz light
                CFG(row_count, 1:7) = { i    0        15      {}   {86 85}     num2cell(1194:1205)     {128 129}};
        end
        TrsBeamIdxMap(i) = 3;
        CellIdxInPatternMap(i) = cellIdx;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {63} is the same with {3}
    % slot {66,67,68,69,70} is the same with {6,7,8,9}
    % slot {71,72} is the same with {1,2}
    % slot {73} is the same with {3}
    % slot {76,77,78,79} is the same with {6,7,8,9}
    % end pattern 89: 64TR peak, Mixed 100 MHz (273 PRBs) + 90 MHz (244 PRBs) + 60 MHz (160 PRBs), OTA, realistic traffic, 9C, PUSCH CP-OFDM, Mixed PUCCH F1/F3. Based on 90655,90656,90658~90661

                                % TC#  slotIdx   cell    ssb      pdcch      pdsch       csirs
    %% pattern 91: 64TR peak, 100 MHz (273 PRBs), OTA, 25-3 column D TC, 15C, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEGs, 8 UEs per slot, 1 layer per UE, 24 DL layer with more PUCCH, SRS in S slot (2 symbols)
    row_count = length(CFG); 
    %%%%% frame 0:         
    % slot {0}, 4 ssb, index 0~3   
    for i = 13802:13816
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {22}        {87}    num2cell(2961:2984)    {}};
        SsbBeamIdxMap(i) = [0 1 2 3];
        CellIdxInPatternMap(i) = mod(i-13802, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {1,2} 
    for i = 13817:13831
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10       {}         {87,71}    num2cell(2985:3008)    {}};
        CellIdxInPatternMap(i) = mod(i-13802, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {3}
    for i = 13832:13846
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {87}     num2cell(3009:3032)     {}};
        CellIdxInPatternMap(i) = mod(i-13802, 15);
        pdschDynamicBfMap(i) = 1;
    end
    % slot {6,7,8,9}
    for i = 13847:13861
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}        {87}     num2cell(3033:3056)     {}};
        CellIdxInPatternMap(i) = mod(i-13802, 15);
        pdschDynamicBfMap(i) = 1;
    end 
    % slot {10}
    for i = 13862:13876
        row_count = row_count + 1;
        CFG(row_count, 1:7) = { i    0        10      {}       {87,70}    num2cell([3057:3080,847])     {}};
        CellIdxInPatternMap(i) = mod(i-13802, 15);
        pdschDynamicBfMap(i) = 1;
    end 
    % slot {11,12} is the same with {1,2}
    % slot {13} is the same with {3}
    % slot {16,17,18,19} is the same with {6,7,8,9}
    %%%%% frame 1:  
    % slot {20}, TRS, TRS index 0
    for i = 13877:13891
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13802, 15);        
        % all cells: trs symbol 5,9
        CFG(row_count, 1:7) = { i    0        10      {}        {87}     num2cell(3033:3056)     {82 83}};
        TrsBeamIdxMap(i) = 0;
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {21}, TRS, TRS index 1
    for i = 13892:13906
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13802, 15);
        % all cells: trs symbol 5,9
        CFG(row_count, 1:7) = { i    0        10      {}        {87,71}     num2cell(2985:3008)     {82 83}};
        TrsBeamIdxMap(i) = 1;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {22} is the same with {1,2}
    % slot {23}, NZP CSI-RS index 2~9, CSI_RS ZP index 10~13
    for i = 13907:13921
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13802, 15);        
        % all cells: NZP csi-rs on symbol 4,5, ZP csi-rs on symbol 2,3
        CFG(row_count, 1:7) = { i    0        10      {}          {}         {}     {113 114}};
        CsiBeamIdxMap(i) = [2, 3, 4, 5, 6, 7, 8, 9];
        CsiZpBeamIdxMap(i) = [10, 11, 12, 13];
        pdschDynamicBfMap(i) = 1;        
    end
    % slot {26,27,28,29,30} is the same with {6,7,8,9}
    % slot {31,32} is the same with {1,2}
    % slot {33} is the same with {3}
    % slot {36,37,38,39} is the same with {6,7,8,9}
    %%%%% frame 2: 
    % slot {40} is the same with {0}
    % slot {41,42} is the same with {1,2}
    % slot {43} is the same with {3}
    % slot {46,47,48,49} is the same with {6,7,8,9}
    % slot {50} is the same with {10}
    % slot {51,52} is the same with {1,2}
    % slot {53} is the same with {3}
    % slot {56,57,58,59} is the same with {6,7,8,9}
    %%%%% frame 3: 
    % slot {60}, TRS, TRS index 14
    for i = 13922:13936
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13802, 15);
        % all cells: trs symbol 5,9
        CFG(row_count, 1:7) = { i    0        10      {}         {87}      num2cell(3033:3056)     {82 83}};
        TrsBeamIdxMap(i) = 14;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {61}, TRS, TRS index 15
    for i = 13937:13951
        row_count = row_count + 1;
        CellIdxInPatternMap(i) = mod(i-13802, 15);
        % all cells: trs symbol 5,9
        CFG(row_count, 1:7) = { i    0        10      {}        {87,71}     num2cell(2985:3008)     {82 83}};
        TrsBeamIdxMap(i) = 15;
        pdschDynamicBfMap(i) = 1;
    end
    % slot {63} is the same with {3}
    % slot {66,67,68,69,70} is the same with {6,7,8,9}
    % slot {71,72} is the same with {1,2}
    % slot {73} is the same with {3}
    % slot {76,77,78,79} is the same with {6,7,8,9}
    % end pattern 91: 64TR peak, 100 MHz (273 PRBs), OTA, 25-3 column D TC, 15C, PUSCH CP-OFDM, Mixed PUCCH F1/F3, 1 UEGs, 8 UEs per slot, 1 layer per UE, 24 DL layer with more PUCCH, SRS in S slot (2 symbols)

    % NOTE: use getLargestTvNum(CFG, threshold) to see the current largest DLMIX TV numbers for <threshold and >= threshold
    %       Search for "available for future DLMIX TVs" to find the avaiable DLMIX TV numbers smaller than that
    largestTvNum = getLargestTvNum(CFG, 20000);

    % additional cell ID map config
    CellIdxInPatternMap(20740) = 1;
    CellIdxInPatternMap(20741) = 1;
    CellIdxInPatternMap(20742) = 1;
    CellIdxInPatternMap(20743) = 2;
    CellIdxInPatternMap(20744) = 2;
    CellIdxInPatternMap(20745) = 2;

    % addtional CSI-RS 32 ports map
    enableCsiRs32PortsMap(192)   = 1;
    enableCsiRs32PortsMap(193)   = 1;
    enableCsiRs32PortsMap(194)   = 1;
    enableCsiRs32PortsMap(20805) = 1;
    enableCsiRs32PortsMap(20894) = 1;
    enableCsiRs32PortsMap(20934) = 1;

    %%
CFG_Cell = {...
 % mu N_grid_size_mu N_ID_CELL Nant_gNB Nant_UE   idx
    1      273            41        4       1;    % 1
    % perf test
    1      273            41        16      1;    % 2
    % FR1 test model
    1      273             1        4       1;    % 3
    % 40 MHz
    1      106            61        4       1;    % 4
    % 80 MHz
    1      217            81        4       1;    % 5
    % 30 MHz
    1      78             81        4       1;    % 6
    % 50 MHz
    1      133            81        4       1;    % 7
    % 32T32R
    1      273            41       32       2;    % 8
    % HARQ TC
    1      273            41        4       4;    % 9
    % 64TR
    1      273            41       16      64;    % 10
    % 64TR 40 MHz
    1      106            41       16      64;    % 11
    % 64TR
    1      273            41        8      64;    % 12
    1      273            41        4      64;    % 13
    % 64TR 90 MHz
    1      244            41       16      64;    % 14
    % 64TR 60 MHz
    1      160            41       16      64;    % 15
    };

CFG_SSB = {...
% cfg#  Nid  n_hf  L_max k_SSB offsetPA SFN  blockIdx
    1,   41,    0,    8,   22,   248,    0,    {0,1};
    2,   41,    0,    8,   22,   248,    0,    {2,3};
    3,   41,    0,    8,   22,   248,    0,    {4,5};

    % perf test
    4,   41,    0,    8,   0,      0,    0,    {0};
    5,   41,    1,    8,   0,      0,    0,    {0};
    % two ssb
    6,   41,    0,    8,   0,      0,    0,    {0,1};
    7,   41,    1,    8,   0,      0,    0,    {0,1};
    % ssb at symbol 8-11
    8,   41,    0,    8,   0,      0,    0,    {1};
    % ssb for 32T32R
    9,   41,    0,    8,   0,      0,    0,    {0};
    % 40 MHz    
    10,   61,    0,    8,   0,      0,    0,    {0,1};
    11,   61,    0,    8,   0,      0,    0,    {1};
    12,   61,    0,    8,   0,      0,    0,    {0};
    % 80 MHz
    13,   81,    0,    8,   0,      0,    0,    {0,1};
    14,   81,    0,    8,   0,      0,    0,    {1};
    15,   81,    0,    8,   0,      0,    0,    {0};
    % HARQ TC
    16,   41,    0,    8,   0,      0,    0,    {0,1};
    17,   41,    0,    8,   0,      0,    0,    {2,3};
    18,   41,    0,    8,   0,      0,    0,    {4,5};    
    19,   41,    0,    8,   0,      0,    0,    {6};    
    % 64TR
    20,   41,    0,    8,   0,      0,    0,    {0,1};
    21,   41,    0,    8,   0,      0,    0,    {2,3};
    22,   41,    0,    8,   0,      0,    0,    {0};
    23,   41,    0,    8,   0,      0,    0,    {0,1}; % SSB 1
    24,   41,    0,    8,   0,      0,    0,    {3}; % SSB 2
    25,   41,    0,    8,   0,      0,    0,    {4}; % SSB 3
    };

CFG_PDCCH = { ...
% cfg# nBWP BWP0 sym0  Nsym crstIdx intl nBndl nIntl nShift nDCI isCSS       rnti      scrbId   scrbRnti    aggrL    Npayload   coresetMap
    1,  273,  0,   0,    2,    1,    0,    6,    0,   0,     2,    0,  {211, 201},   {41, 41},    {0, 0},  {2, 2},  {41, 39},  {ones(1,20)} ;
    2,  273,  0,   0,    2,    2,    0,    6,    0,   0,     2,    0,  {221, 231},   {41, 41},    {0, 0},  {2, 2},  {45, 37},  {[zeros(1,20), ones(1,20)]} ;

    % perf test
    % F08
    3, 273,   0,   0,    1,    2,    0,    6,    0,   0,     12,    0,        {0},        {0},       {0},     {1},      {39},   {ones(1, 46)};
    % F09
    4, 273,   0,   0,    2,    2,    0,    6,    0,   0,     20,    0,        {0},        {0},       {0},     {1},      {39},   {ones(1, 46)};
    % F08 relaxed
    5, 273,   0,   0,    1,    2,    0,    6,    0,   0,     12,    0,        {0},        {0},       {0},     {1},      {39},   {ones(1, 46)};
    % F08 relaxed, compact PRBs
    6, 273,   0,   0,    1,    2,    0,    6,    0,   0,     12,    0,        {0},        {0},       {0},     {1},      {39},   {ones(1, 12)};
    % F08 relaxed, compact PRBs, 6 DL UCI + 6 UL UCI
    7, 273,   0,   0,    1,    2,    0,    6,    0,   0,      6,    0,        {0},        {0},       {0},     {1},      {39},   {ones(1, 6)};
    8, 273,   0,   0,    1,    2,    0,    6,    0,   0,      6,    0,        {0},        {0},       {0},     {1},      {39},   {[zeros(1,6) ones(1, 6)]};
    % FR1 test model
    9, 273,   0,   0,    2,    2,    0,    6,    0,   0,      1,    0,        {0},       {41},       {0},     {1},      {39},   {[1]};
    % 32T32R
   10, 273,   0,   0,    2,    1,    0,    6,    0,   0,      1,    0,      {201},       {41},       {0},     {2},      {39},  {ones(1,20)} ;    
    % F08 relaxed, 40 MHz, compact PRBs, 6 DL UCI + 6 UL UCI
   11, 106,   0,   0,    1,    2,    0,    6,    0,   0,      6,    0,        {0},        {0},       {0},     {1},      {39},   {ones(1, 6)};
   12, 106,   0,   0,    1,    2,    0,    6,    0,   0,      6,    0,        {0},        {0},       {0},     {1},      {39},   {[zeros(1,6) ones(1, 6)]};
     % F08 relaxed, 40 MHz, 2 symbols, compact PRBs, 6 DL UCI + 6 UL UCI
   13, 106,   0,   0,    2,    2,    0,    6,    0,   0,      6,    0,        {0},        {0},       {0},     {1},      {39},   {ones(1, 3)};
   14, 106,   0,   0,    2,    2,    0,    6,    0,   0,      6,    0,        {0},        {0},       {0},     {1},      {39},   {[zeros(1,3) ones(1, 3)]};
     % F08 relaxed, 40 MHz, compact PRBs, 6 DL UCI + 6 UL UCI
   15, 217,   0,   0,    1,    2,    0,    6,    0,   0,      6,    0,        {0},        {0},       {0},     {1},      {39},   {ones(1, 6)};
   16, 217,   0,   0,    1,    2,    0,    6,    0,   0,      6,    0,        {0},        {0},       {0},     {1},      {39},   {[zeros(1,6) ones(1, 6)]};
 % F08, 30 MHz, compact PRBs, 6 DL UCI + 6 UL UCI
   17, 78,   0,   0,    1,    2,    0,    6,    0,   0,      6,    0,        {0},        {0},       {0},     {1},      {39},   {ones(1, 6)};
   18, 78,   0,   0,    1,    2,    0,    6,    0,   0,      6,    0,        {0},        {0},       {0},     {1},      {39},   {[zeros(1,6) ones(1, 6)]};
 % F08, 30 MHz, 2 symbols, compact PRBs, 6 DL UCI + 6 UL UCI
   19, 78,   0,   0,    2,    2,    0,    6,    0,   0,      6,    0,        {0},        {0},       {0},     {1},      {39},   {ones(1, 3)};
   20, 78,   0,   0,    2,    2,    0,    6,    0,   0,      6,    0,        {0},        {0},       {0},     {1},      {39},   {[zeros(1,3) ones(1, 3)]};
   % F08, 50 MHz, compact PRBs, 6 DL UCI + 6 UL UCI
   21, 133,   0,   0,    1,    2,    0,    6,    0,   0,      6,    0,        {0},        {0},       {0},     {1},      {39},   {ones(1, 6)};
   22, 133,   0,   0,    1,    2,    0,    6,    0,   0,      6,    0,        {0},        {0},       {0},     {1},      {39},   {[zeros(1,6) ones(1, 6)]};
   % F08, 50 MHz, 2 symbols, compact PRBs, 6 DL UCI + 6 UL UCI
   23, 133,   0,   0,    2,    2,    0,    6,    0,   0,      6,    0,        {0},        {0},       {0},     {1},      {39},   {ones(1, 3)};
   24, 133,   0,   0,    2,    2,    0,    6,    0,   0,      6,    0,        {0},        {0},       {0},     {1},      {39},   {[zeros(1,3) ones(1, 3)]};
  
   % HARQ TC
% cfg# nBWP BWP0 sym0  Nsym crstIdx intl nBndl nIntl nShift nDCI isCSS       rnti      scrbId   scrbRnti    aggrL    Npayload   coresetMap
   25, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0},           {2},      {39},   {ones(1, 46)};
   26, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0},           {4},      {39},   {ones(1, 46)};
   27, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0},           {8},      {39},   {ones(1, 46)};
   28, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0},       {8 8 8},      {39},   {ones(1, 46)};
   29, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0},       {2 4 8},      {39},   {ones(1, 46)};
   30, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0},       {2 2 4},      {39},   {ones(1, 46)};
   31, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0},       {8 8 2},      {39},   {ones(1, 46)};
   32, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0},       {8 4 8},      {39},   {ones(1, 46)};
   33, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0}, {8 8 8 8 8 4},      {39},   {ones(1, 46)};
   34, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0}, {4 4 4 4 4 4},      {39},   {ones(1, 46)};
   35, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0}, {2 2 2 2 2 2},      {39},   {ones(1, 46)};
   36, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0}, {2 8 4 8 2 4},      {39},   {ones(1, 46)};
   37, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0}, {8 8 8 4 2 4},      {39},   {ones(1, 46)};
   38, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0}, {4 2 2 4 4 8},      {39},   {ones(1, 46)};   

   39, 273,   0,   1,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0},           {2},      {39},   {ones(1, 46)};
   40, 273,   0,   1,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0},           {4},      {39},   {ones(1, 46)};
   41, 273,   0,   1,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0},           {8},      {39},   {ones(1, 46)};
   42, 273,   0,   1,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0},       {8 8 8},      {39},   {ones(1, 46)};
   43, 273,   0,   1,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0},       {2 4 8},      {39},   {ones(1, 46)};
   44, 273,   0,   1,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0},       {2 2 4},      {39},   {ones(1, 46)};
   45, 273,   0,   1,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0},       {8 8 2},      {39},   {ones(1, 46)};
   46, 273,   0,   1,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0},       {8 4 8},      {39},   {ones(1, 46)};
   47, 273,   0,   1,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0}, {8 8 8 8 8 4},      {39},   {ones(1, 46)};
   48, 273,   0,   1,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0}, {4 4 4 4 4 4},      {39},   {ones(1, 46)};
   49, 273,   0,   1,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0}, {2 2 2 2 2 2},      {39},   {ones(1, 46)};
   50, 273,   0,   1,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0}, {2 8 4 8 2 4},      {39},   {ones(1, 46)};
   51, 273,   0,   1,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0}, {8 8 8 4 2 4},      {39},   {ones(1, 46)};
   52, 273,   0,   1,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0}, {4 2 2 4 4 8},      {39},   {ones(1, 46)};     
   
   % 64TR
   53, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0},           {2},      {39},   {ones(1, 46)};
   54, 273,   0,   1,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0},           {2},      {39},   {ones(1, 46)};
   % Simultaneous DL/UL in S-slot
   55, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0}, {4 2 2 4 4 8},      {39},   {ones(1, 46)};     
   %55, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0},       {8 8 8},      {39},   {ones(1, 46)};
   56, 273,   0,   1,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0},       {8 8 8},      {39},   {ones(1, 46)};   
   57, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0}, {4 2 2 4 4 8},      {39},   {ones(1, 46)};   
   58, 273,   0,   1,    1,    1,    0,    6,    0,   0,      1,    0,        {0},       {41},  {0}, {2 8 4 8 2 4},      {39},   {ones(1, 46)};
% cfg# nBWP BWP0 sym0  Nsym crstIdx intl nBndl nIntl nShift nDCI isCSS       rnti      scrbId   scrbRnti    aggrL    Npayload   coresetMap
    % 64TR, 16 DL UCI + 16 UL UCI
   59, 273,   0,   0,    2,    2,    0,    6,    0,   0,     16,    0,        {0},        {0},       {0},     {1},      {39},   {ones(1, 8)};
   60, 273,   0,   0,    2,    2,    0,    6,    0,   0,     16,    0,        {0},        {0},       {0},     {1},      {39},   {[zeros(1,8) ones(1, 8)]};
% 64TR test case
   61, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0}, {2 2 2 2 2 2 2 2}, {39},   {ones(1, 46)};
   62, 273,   0,   1,    1,    2,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0},     {2 2 2 2 2 2}, {39},   {ones(1, 46)};
   63, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0}, num2cell(2*ones(1,16)), {39},   {ones(1, 46)};
    % 64TR test case, half load
   64, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0}, {2 2 2 2 2 2 2 2}, {39},   {ones(1, 22)};
   65, 273,   0,   1,    1,    2,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0},     {2 2 2 2 2 2}, {39},   {ones(1, 22)};
   66, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0}, num2cell(2*ones(1,8)), {39},   {ones(1, 22)};
% cfg# nBWP BWP0 sym0  Nsym crstIdx intl nBndl nIntl nShift nDCI isCSS       rnti      scrbId   scrbRnti    aggrL    Npayload   coresetMap
   % 64TR 40 MHz test case
   67, 106,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0}, {2 2 2 2 2 2 2 2}, {39},   {ones(1, 17)};
   68, 106,   0,   1,    1,    2,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0}, {2 2 2 2 2 2 2 2}, {39},   {ones(1, 17)};
   69, 106,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0}, num2cell(2*ones(1,8)), {39},   {ones(1, 17)};
   % 64TR column I
   70, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    1,        {0},        {41},  {0}, {8},                  {39},   {[ones(1,8) zeros(1, 38)]};  % PDCCH for CCH
   71, 273,   0,   1,    1,    2,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0}, num2cell(ones(1,8)),  {39},   {ones(1, 46)};  % PDCCH for UL
   72, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0}, num2cell(ones(1,16)), {39},   {[zeros(1,8) ones(1, 38)]};  % PDCCH for DL
% 64TR test case, column G, 16 DL UCI + 8 UL UCI
   73, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0}, {2 2 2 2 2 2 2 2}, {39},   {ones(1, 46)};
   74, 273,   0,   1,    1,    2,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0}, {2 2 2 2 2 2 2 2}, {39},   {ones(1, 46)};
   75, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0}, num2cell(2*ones(1,16)), {39},   {ones(1, 46)};
% Heterogeneous DL UCIs
   76, 273,   0,   0,    1,    1,    1,    6,    2,   0,      1,    0,        {0},        {41},  {0}, {2 2 2 2 2 2 2},   {39},   {ones(1, 46)};
   77, 273,   0,   1,    1,    2,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0}, {2 2 2},           {39},   {ones(1, 46)};
   78, 273,   0,   0,    1,    1,    1,    6,    2,   0,      1,    0,        {0},        {41},  {0},       {2 2 2 2},   {39},   {ones(1, 46)};
   79, 273,   0,   0,    1,    2,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0},       {2 2 2 2},   {39},   {ones(1, 46)};
% PDCCH DCI1_1x32   
   80, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0}, num2cell(ones(1,32)), {39},   {[zeros(1,8) ones(1, 38)]};  % PDCCH for DL
% 64TR realistic traffic with Center/Middle/Edge, 90 MHz heavy / light
   81, 244,   0,   0,    1,    1,    0,    6,    0,   0,      1,    1,        {0},        {41},  {0}, {8},                  {39},   {[ones(1,8) zeros(1, 32)]};  % PDCCH for CCH
   82, 244,   0,   1,    1,    2,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0}, num2cell(ones(1,8)),  {39},   {ones(1, 40)};  % PDCCH for UL
   83, 244,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0}, num2cell(ones(1,16)), {39},   {[zeros(1,8) ones(1, 32)]};  % PDCCH for DL
% 64TR realistic traffic with Center/Middle/Edge, 60 MHz heavy / light
   84, 160,   0,   0,    1,    1,    0,    6,    0,   0,      1,    1,        {0},        {41},  {0}, {8},                  {39},   {[ones(1,8) zeros(1, 19)]};  % PDCCH for CCH
   85, 160,   0,   1,    1,    2,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0}, num2cell(ones(1,8)),  {39},   {ones(1, 27)};  % PDCCH for UL
   86, 160,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0}, num2cell(ones(1,16)), {39},   {[zeros(1,8) ones(1, 19)]};  % PDCCH for DL
   % PDCCH DCI1_1x24   
   87, 273,   0,   0,    1,    1,    0,    6,    0,   0,      1,    0,        {0},        {41},  {0}, num2cell(ones(1,24)), {39},   {[zeros(1,8) ones(1, 38)]};  % PDCCH for DL
   };
% NOTE: For HARQ test cases, coresetMap will be overwritting based on aggregation level (using mapOnes and mapZeros)


CFG_PDSCH = {...
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    1,   1,       1,   1,  0,   100,  2,     12,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    2,   1,       1,   1,200,    50,  2,     12,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    3,   1,       1,   1,  0,   273,  2,      4,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    4,   1,       1,   1,  0,   273,  2,     12,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    5,   1,       1,   1,  0,   273,  2,     12,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   1,   0;

    % 38.141-1 chapter-4 FR1 test model
    6,   1,       1,   1,  3,   270,  0,     14,     0,    0,    273,     0,    0,      1,     2,      1,     1,      1,      1,   0,   0;
    7,   1,       1,   1,  0,     3,  2,     12,     0,    0,    273,     2,    0,      1,     2,      1,     1,      1,      1,   0,   1;
    8,   1,      11,   1,  0,     1,  2,     12,     0,    0,    273,     2,    0,      1,     2,      1,     1,      1,      1,   0,   0;
    9,   1,      20,   1,  0,     1,  2,     12,     0,    0,    273,     2,    0,      1,     2,      1,     1,      1,      1,   0,   0;
   10,   1,      11,   1,  3,   270,  0,     14,     0,    0,    273,     0,    0,      1,     2,      1,     1,      1,      1,   0,   0;
   11,   1,      11,   1,  0,     3,  2,     12,     0,    0,    273,     2,    0,      1,     2,      1,     1,      1,      1,   0,   1;
   12,   1,      20,   1,  3,   270,  0,     14,     0,    0,    273,     0,    0,      1,     2,      1,     1,      1,      1,   0,   0;
   13,   1,      20,   1,  0,     3,  2,     12,     0,    0,    273,     2,    0,      1,     2,      1,     1,      1,      1,   0,   1;
   14,   1,       1,   1,  0,     3,  2,     12,     0,    0,    273,     2,    0,      1,     2,      1,     1,      1,      1,   0,   0;
   15,   1,       1,   1,  3,   270,  0,     14,     0,    3,    270,     0,    0,      1,     2,      1,     1,      1,      1,   0,   1;
   16,   1,       1,   1,  3,   270,  0,     14,     0,    3,    270,     1,    0,      1,     2,      1,     1,      1,      1,   0,   2;
   17,   1,       1,   2,  3,   270,  0,     14,     0,    0,    273,     0,    0,      1,     2,      1,     1,      1,      1,   0,   0;
   18,   1,       1,   2,  0,     3,  2,     12,     0,    0,    273,     2,    0,      1,     2,      1,     1,      1,      1,   0,   1;
   
    % -> 20 reserved     
    % for cuphy functional check when pdsch with csirs
    19,  1,       27,  4,  100, 100,  2,     12,     0,    0,    273,     0,    0,     41,     2,      2,     0,     41,      2,   4,   1;
    20,  1,       27,  4,  200,  73,  2,     12,     0,    0,    273,     0,    0,     41,     2,      2,     0,     41,      2,   0,   2;

% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    % for F08 240 PRBs for 16 UEG
    21,   1,       27,  4,  20,  15,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;
    22,   1,       27,  4,  35,  15,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    23,   1,       27,  4,  50,  15,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    24,   1,       27,  4,  65,  15,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    25,   1,       27,  4,  80,  15,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    26,   1,       27,  4,  95,  15,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    27,   1,       27,  4,  110, 15,  1,     13,     0,    0,    273,     6,    0,     41,     2,      1,     0,     41,      2,   0,   6;
    28,   1,       27,  4,  125, 15,  1,     13,     0,    0,    273,     7,    0,     41,     2,      1,     0,     41,      2,   0,   7;
    29,   1,       27,  4,  140, 15,  1,     13,     0,    0,    273,     8,    0,     41,     2,      1,     0,     41,      2,   0,   8;
    30,   1,       27,  4,  155, 15,  1,     13,     0,    0,    273,     9,    0,     41,     2,      1,     0,     41,      2,   0,   9;
    31,   1,       27,  4,  170, 15,  1,     13,     0,    0,    273,    10,    0,     41,     2,      1,     0,     41,      2,   0,   10;
    32,   1,       27,  4,  185, 15,  1,     13,     0,    0,    273,    11,    0,     41,     2,      1,     0,     41,      2,   0,   11;
    33,   1,       27,  4,  200, 15,  1,     13,     0,    0,    273,    12,    0,     41,     2,      1,     0,     41,      2,   0,   12;
    34,   1,       27,  4,  215, 15,  1,     13,     0,    0,    273,    13,    0,     41,     2,      1,     0,     41,      2,   0,   13;
    35,   1,       27,  4,  230, 15,  1,     13,     0,    0,    273,    14,    0,     41,     2,      1,     0,     41,      2,   0,   14;
    36,   1,       27,  4,  245, 15,  1,     13,     0,    0,    273,    15,    0,     41,     2,      1,     0,     41,      2,   0,   15;
    % for F08 272 PRBs for 16 UEG
    37,   1,       27,  4,  0,   17,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;
    38,   1,       27,  4,  17,  17,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    39,   1,       27,  4,  34,  17,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    40,   1,       27,  4,  51,  17,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    41,   1,       27,  4,  68,  17,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    42,   1,       27,  4,  85,  17,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    43,   1,       27,  4,  102, 17,  1,     13,     0,    0,    273,     6,    0,     41,     2,      1,     0,     41,      2,   0,   6;
    44,   1,       27,  4,  119, 17,  1,     13,     0,    0,    273,     7,    0,     41,     2,      1,     0,     41,      2,   0,   7;
    45,   1,       27,  4,  136, 17,  1,     13,     0,    0,    273,     8,    0,     41,     2,      1,     0,     41,      2,   0,   8;
    46,   1,       27,  4,  153, 17,  1,     13,     0,    0,    273,     9,    0,     41,     2,      1,     0,     41,      2,   0,   9;
    47,   1,       27,  4,  170, 17,  1,     13,     0,    0,    273,    10,    0,     41,     2,      1,     0,     41,      2,   0,   10;
    48,   1,       27,  4,  187, 17,  1,     13,     0,    0,    273,    11,    0,     41,     2,      1,     0,     41,      2,   0,   11;
    49,   1,       27,  4,  204, 17,  1,     13,     0,    0,    273,    12,    0,     41,     2,      1,     0,     41,      2,   0,   12;
    50,   1,       27,  4,  221, 17,  1,     13,     0,    0,    273,    13,    0,     41,     2,      1,     0,     41,      2,   0,   13;
    51,   1,       27,  4,  238, 17,  1,     13,     0,    0,    273,    14,    0,     41,     2,      1,     0,     41,      2,   0,   14;
    52,   1,       27,  4,  255, 17,  1,     13,     0,    0,    273,    15,    0,     41,     2,      1,     0,     41,      2,   0,   15;
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    % for F08 128 PRBs for 16 UEG, 8 RBs per UEG
    53,   1,       27,  4,  20,   8,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;
    54,   1,       27,  4,  28,   8,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    55,   1,       27,  4,  36,   8,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    56,   1,       27,  4,  44,   8,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    57,   1,       27,  4,  52,   8,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    58,   1,       27,  4,  60,   8,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    59,   1,       27,  4,  68,   8,  1,     13,     0,    0,    273,     6,    0,     41,     2,      1,     0,     41,      2,   0,   6;
    60,   1,       27,  4,  76,   8,  1,     13,     0,    0,    273,     7,    0,     41,     2,      1,     0,     41,      2,   0,   7;
    61,   1,       27,  4,  84,   8,  1,     13,     0,    0,    273,     8,    0,     41,     2,      1,     0,     41,      2,   0,   8;
    62,   1,       27,  4,  92,   8,  1,     13,     0,    0,    273,     9,    0,     41,     2,      1,     0,     41,      2,   0,   9;
    63,   1,       27,  4,  100,  8,  1,     13,     0,    0,    273,    10,    0,     41,     2,      1,     0,     41,      2,   0,   10;
    64,   1,       27,  4,  108,  8,  1,     13,     0,    0,    273,    11,    0,     41,     2,      1,     0,     41,      2,   0,   11;
    65,   1,       27,  4,  116,  8,  1,     13,     0,    0,    273,    12,    0,     41,     2,      1,     0,     41,      2,   0,   12;
    66,   1,       27,  4,  124,  8,  1,     13,     0,    0,    273,    13,    0,     41,     2,      1,     0,     41,      2,   0,   13;
    67,   1,       27,  4,  132,  8,  1,     13,     0,    0,    273,    14,    0,     41,     2,      1,     0,     41,      2,   0,   14;
    68,   1,       27,  4,  140,  8,  1,     13,     0,    0,    273,    15,    0,     41,     2,      1,     0,     41,      2,   0,   15;
    % for F08 relaxed, 6 UEGs, each with 42 PRBs 
    69,   1,       27,  4,  20,  42,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;
    70,   1,       27,  4,  62,  42,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    71,   1,       27,  4,  104, 42,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    72,   1,       27,  4,  146, 42,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    73,   1,       27,  4,  188, 42,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    74,   1,       27,  4,  230, 42,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    % for F08 relaxed, 6 UEGs, each with 22 PRBs 
    75,   1,       27,  4,  20,  22,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;
    76,   1,       27,  4,  42,  22,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    77,   1,       27,  4,  64,  22,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    78,   1,       27,  4,  86,  22,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    79,   1,       27,  4,  108, 22,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    80,   1,       27,  4,  130, 22,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    % for F08, 6 UEGs, 1 PRB / UE 
    81,   1,       27,  4,  20,   1,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;
    82,   1,       27,  4,  21,   1,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    83,   1,       27,  4,  22,   1,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    84,   1,       27,  4,  23,   1,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    85,   1,       27,  4,  24,   1,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    86,   1,       27,  4,  25,   1,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    % 16C, 1 PRB / UEG, PDSCH only
    87,   1,       27,  4,  20,   1,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;
    % for F08 relaxed, 1 UE with fewer symbols,6 UEGs, each with 22 PRBs 
    88,   1,       27,  4,   0,  22,  6,      7,     0,    0,    273,     0,    0,     41,     6,      1,     0,     41,      2,   0,   0;
    89,   1,       27,  4,  22,  22,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    90,   1,       27,  4,  44,  22,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    91,   1,       27,  4,  66,  22,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    92,   1,       27,  4,  88,  22,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    93,   1,       27,  4,  110, 22,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    % for F08 relaxed, 1 UE with fewer symbols (1-7),6 UEGs, each with 22 PRBs 
    94,   1,       27,  4,   0,  22,  1,      7,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;
    95,   1,       27,  4,  22,  22,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    96,   1,       27,  4,  44,  22,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    97,   1,       27,  4,  66,  22,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    98,   1,       27,  4,  88,  22,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    99,   1,       27,  4,  110, 22,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    % for F08 relaxed, 6 UEGs, each with 17 PRBs 
    100,  1,       27,  4,  20,  17,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;
    101,  1,       27,  4,  37,  17,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    102,  1,       27,  4,  54,  17,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    103,  1,       27,  4,  71,  17,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    104,  1,       27,  4,  88,  17,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    105,  1,       27,  4,  105, 17,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    % for F08 relaxed, 6 UEGs, each with 12 PRBs 
    106,  1,       27,  4,  20,  12,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;
    107,  1,       27,  4,  32,  12,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    108,  1,       27,  4,  44,  12,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    109,  1,       27,  4,  56,  12,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    110,  1,       27,  4,  68,  12,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    111,  1,       27,  4,  80,  12,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    % F08, 6 UEGs, each with 18 PRBs (with ssb)
    112,  1,       27,  4,  20,  18,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;
    113,  1,       27,  4,  38,  18,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    114,  1,       27,  4,  56,  18,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    115,  1,       27,  4,  74,  18,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    116,  1,       27,  4,  92,  18,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    117,  1,       27,  4,  110, 18,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    % for F08 relaxed, 6 UEGs, each with 22 PRBs, start with 0 
    118,  1,       27,  4,  0,   22,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;   
    119,  1,       27,  4,  22,  22,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    120,  1,       27,  4,  44,  22,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    121,  1,       27,  4,  66,  22,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    122,  1,       27,  4,  88,  22,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    123,  1,       27,  4,  110, 22,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    % for F08 peak, 6 UEGs, 270 PRBs, each with 45 PRBs, start with 0 
    124,  1,       27,  4,  0,   45,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;   
    125,  1,       27,  4,  45,  45,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    126,  1,       27,  4,  90,  45,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    127,  1,       27,  4,  135, 45,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    128,  1,       27,  4,  180, 45,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    129,  1,       27,  4,  225, 45,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    % for F08 ~peak, 6 UEGs, 240 PRBs, each with 40 PRBs, start with 0 
    130,  1,       27,  4,  0,   40,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;   
    131,  1,       27,  4,  40,  40,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    132,  1,       27,  4,  80,  40,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    133,  1,       27,  4,  120, 40,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    134,  1,       27,  4,  160, 40,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    135,  1,       27,  4,  200, 40,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    % for F08 ~peak, 6 UEGs, 240 PRBs, each with 40 PRBs, start with 20 
    136,  1,       27,  4,  20,  40,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;   
    137,  1,       27,  4,  60,  40,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    138,  1,       27,  4,  100, 40,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    139,  1,       27,  4,  140, 40,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    140,  1,       27,  4,  180, 40,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    141,  1,       27,  4,  220, 40,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    % for F08 ave, 1 UEG, 132 PRBs, each with 132 PRBs, start with 0 
    142,  1,       27,  4,  0,   132, 1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;   
    % for F08 peak, 1 UEG, 270 PRBs, each with 40 PRBs, start with 0 
    143,  1,       27,  4,  0,   270, 1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;   
    % for F08 relaxed, 6 UEGs, each with 17 PRBs, start with 0 
    144,  1,       27,  4,  0,   17,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;   
    145,  1,       27,  4,  17,  17,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    146,  1,       27,  4,  34,  17,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    147,  1,       27,  4,  51,  17,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    148,  1,       27,  4,  68,  17,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    149,  1,       27,  4,  85,  17,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    % for F08 relaxed, 6 UEGs, each with 13 PRBs, start with 0 (for disjoint PDSCH and CSI-RS)
    150,  1,       27,  4,  0,   13,  1,     11,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;   
    151,  1,       27,  4,  13,  13,  1,     11,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    152,  1,       27,  4,  26,  13,  1,     11,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    153,  1,       27,  4,  39,  13,  1,     11,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    154,  1,       27,  4,  52,  13,  1,     11,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    155,  1,       27,  4,  65,  13,  1,     11,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg    
    % for 32T32R
    156,  1,       27,  4,  20, 253,  2,     12,     0,    0,    273,   211,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    157,  1,       27,  4,   0, 273,  2,     12,     0,    0,    273,   211,    0,     41,     2,      1,     1,     41,      2,   0,   0;
   % for F08 6 UEGs, each with 42 PRBs, OTA (addPos) 
    158,  1,       27,  4,  20,  42,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    159,  1,       27,  4,  62,  42,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    160,  1,       27,  4,  104, 42,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    161,  1,       27,  4,  146, 42,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    162,  1,       27,  4,  188, 42,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    163,  1,       27,  4,  230, 42,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % for F08 peak, 6 UEGs, each with 45 PRBs, start with 0, OTA (addPos) 
    164,  1,       27,  4,  0,   45,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;   
    165,  1,       27,  4,  45,  45,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    166,  1,       27,  4,  90,  45,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    167,  1,       27,  4,  135, 45,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    168,  1,       27,  4,  180, 45,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    169,  1,       27,  4,  225, 45,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % F08, 6 UEGs, each with 18 PRBs (with ssb), OTA (addPos) 
    170,  1,       27,  4,  20,  18,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    171,  1,       27,  4,  38,  18,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    172,  1,       27,  4,  56,  18,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    173,  1,       27,  4,  74,  18,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    174,  1,       27,  4,  92,  18,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    175,  1,       27,  4,  110, 18,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % for F08, 6 UEGs, each with 22 PRBs, start with 0, OTA (addPos)
    176,  1,       27,  4,  0,   22,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;   
    177,  1,       27,  4,  22,  22,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    178,  1,       27,  4,  44,  22,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    179,  1,       27,  4,  66,  22,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    180,  1,       27,  4,  88,  22,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    181,  1,       27,  4,  110, 22,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % for F08 relaxed, 6 UEGs, each with 13 PRBs, start with 0 (for disjoint PDSCH and CSI-RS)
    182,  1,       27,  4,  0,   13,  1,     11,     0,    0,    273,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;   
    183,  1,       27,  4,  13,  13,  1,     11,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    184,  1,       27,  4,  26,  13,  1,     11,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    185,  1,       27,  4,  39,  13,  1,     11,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    186,  1,       27,  4,  52,  13,  1,     11,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    187,  1,       27,  4,  65,  13,  1,     11,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
   % for F08 40MHz, 6 UEGs, each with 14 PRBs, OTA (addPos) 
    188,  1,       27,  4,  20,  14,  1,     13,     0,    0,    106,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    189,  1,       27,  4,  34,  14,  1,     13,     0,    0,    106,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    190,  1,       27,  4,  48,  14,  1,     13,     0,    0,    106,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    191,  1,       27,  4,  62,  14,  1,     13,     0,    0,    106,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    192,  1,       27,  4,  76,  14,  1,     13,     0,    0,    106,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    193,  1,       27,  4,  90,  14,  1,     13,     0,    0,    106,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % for F08 40MHz peak, 6 UEGs, each with 17 PRBs, start with 0, OTA (addPos) 
    194,  1,       27,  4,  0,   17,  1,     13,     0,    0,    106,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;   
    195,  1,       27,  4,  17,  17,  1,     13,     0,    0,    106,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    196,  1,       27,  4,  34,  17,  1,     13,     0,    0,    106,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    197,  1,       27,  4,  51,  17,  1,     13,     0,    0,    106,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    198,  1,       27,  4,  68,  17,  1,     13,     0,    0,    106,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    199,  1,       27,  4,  85,  17,  1,     13,     0,    0,    106,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
     % for F08 relaxed, 6 UEGs, each with 8 PRBs, start with 0 (for disjoint PDSCH and CSI-RS)
    200,  1,       27,  4,  0,   8,   2,     10,     0,    0,    106,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;   
    201,  1,       27,  4,  8,   8,   2,     10,     0,    0,    106,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    202,  1,       27,  4,  16,  8,   2,     10,     0,    0,    106,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    203,  1,       27,  4,  24,  8,   2,     10,     0,    0,    106,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    204,  1,       27,  4,  32,  8,   2,     10,     0,    0,    106,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    205,  1,       27,  4,  40,  8,   2,     10,     0,    0,    106,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % F08, 6 UEGs, each with 5 PRBs (with ssb), OTA (addPos) 
    206,  1,       27,  4,  20,  5,   2,     12,     0,    0,    106,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    207,  1,       27,  4,  25,  5,   2,     12,     0,    0,    106,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    208,  1,       27,  4,  30,  5,   2,     12,     0,    0,    106,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    209,  1,       27,  4,  35,  5,   2,     12,     0,    0,    106,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    210,  1,       27,  4,  40,  5,   2,     12,     0,    0,    106,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    211,  1,       27,  4,  45,  5,   2,     12,     0,    0,    106,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % for F08, 6 UEGs, each with 8 PRBs, start with 0, OTA (addPos)
    212,  1,       27,  4,  0,   8,   2,     12,     0,    0,    106,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;   
    213,  1,       27,  4,  8,   8,   2,     12,     0,    0,    106,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    214,  1,       27,  4,  16,  8,   2,     12,     0,    0,    106,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    215,  1,       27,  4,  24,  8,   2,     12,     0,    0,    106,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    216,  1,       27,  4,  32,  8,   2,     12,     0,    0,    106,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    217,  1,       27,  4,  40,  8,   2,     12,     0,    0,    106,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % for F08 80MHz, 6 UEGs, each with 32 PRBs, OTA (addPos) 
    218,  1,       27,  4,  20,  32,  1,     13,     0,    0,    217,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    219,  1,       27,  4,  52,  32,  1,     13,     0,    0,    217,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    220,  1,       27,  4,  84,  32,  1,     13,     0,    0,    217,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    221,  1,       27,  4,  116, 32,  1,     13,     0,    0,    217,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    222,  1,       27,  4,  148, 32,  1,     13,     0,    0,    217,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    223,  1,       27,  4,  180, 32,  1,     13,     0,    0,    217,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % for F08 80MHz peak, 6 UEGs, each with 36 PRBs, start with 0, OTA (addPos) 
    224,  1,       27,  4,  0,   36,  1,     13,     0,    0,    217,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;   
    225,  1,       27,  4,  36,  36,  1,     13,     0,    0,    217,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    226,  1,       27,  4,  72,  36,  1,     13,     0,    0,    217,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    227,  1,       27,  4,  108, 36,  1,     13,     0,    0,    217,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    228,  1,       27,  4,  144, 36,  1,     13,     0,    0,    217,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    229,  1,       27,  4,  180, 36,  1,     13,     0,    0,    217,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
      % F08, 80MHz ave, 6 UEGs, each with 14 PRBs (with ssb), OTA (addPos) 
    230,  1,       27,  4,  20,  14,   1,     13,    0,    0,    217,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    231,  1,       27,  4,  34,  14,   1,     13,    0,    0,    217,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    232,  1,       27,  4,  48,  14,   1,     13,    0,    0,    217,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    233,  1,       27,  4,  62,  14,   1,     13,    0,    0,    217,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    234,  1,       27,  4,  76,  14,   1,     13,    0,    0,    217,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    235,  1,       27,  4,  90,  14,   1,     13,    0,    0,    217,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % for F08, 80MHz ave, 6 UEGs, each with 18 PRBs, start with 0, OTA (addPos)
    236,  1,       27,  4,  0,   18,   1,     13,     0,    0,    217,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;   
    237,  1,       27,  4,  18,  18,   1,     13,     0,    0,    217,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    238,  1,       27,  4,  36,  18,   1,     13,     0,    0,    217,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    239,  1,       27,  4,  54,  18,   1,     13,     0,    0,    217,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    240,  1,       27,  4,  72,  18,   1,     13,     0,    0,    217,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    241,  1,       27,  4,  90,  18,   1,     13,     0,    0,    217,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
 % for F08 relaxed, 80MHz ave, 6 UEGs, each with 9 PRBs, start with 0 (for disjoint PDSCH and CSI-RS)
    242,  1,       27,  4,  0,    9,   1,     11,     0,    0,    217,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;   
    243,  1,       27,  4,  9,    9,   1,     11,     0,    0,    217,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    244,  1,       27,  4,  18,   9,   1,     11,     0,    0,    217,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    245,  1,       27,  4,  27,   9,   1,     11,     0,    0,    217,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    246,  1,       27,  4,  36,   9,   1,     11,     0,    0,    217,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    247,  1,       27,  4,  45,   9,   1,     11,     0,    0,    217,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    % for 32T32R
    248,  1,       27,  2,  20, 253,  2,     12,     0,    0,    273,   211,    0,     41,     2,      2,     1,     41,      2,   0,   0; % 156
    249,  1,       27,  2,  20, 253,  2,     12,     0,    0,    273,   211,    0,     41,     2,      2,     1,     41,      2,   2,   0; % 156
    250,  1,       27,  2,   0, 273,  2,     12,     0,    0,    273,   211,    0,     41,     2,      2,     1,     41,      2,   0,   0; % 157
    251,  1,       27,  2,   0, 273,  2,     12,     0,    0,    273,   211,    0,     41,     2,      2,     1,     41,      2,   2,   0; % 157
        % F08, 6 UEGs, each with 9 PRBs (with ssb), OTA (addPos) 
    252,  1,       27,  4,  20+0*9,  9,  1,     13,     0,    0,    78,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    253,  1,       27,  4,  20+1*9,  9,  1,     13,     0,    0,    78,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    254,  1,       27,  4,  20+2*9,  9,  1,     13,     0,    0,    78,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    255,  1,       27,  4,  20+3*9,  9,  1,     13,     0,    0,    78,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    256,  1,       27,  4,  20+4*9,  9,  1,     13,     0,    0,    78,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    257,  1,       27,  4,  20+5*9,  9,  1,     13,     0,    0,    78,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
     % F08, 6 UEGs, each with 13 PRBs, OTA (addPos) 
    258,  1,       27,  4,  0*13,  13,  1,     13,     0,    0,    78,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    259,  1,       27,  4,  1*13,  13,  1,     13,     0,    0,    78,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    260,  1,       27,  4,  2*13,  13,  1,     13,     0,    0,    78,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    261,  1,       27,  4,  3*13,  13,  1,     13,     0,    0,    78,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    262,  1,       27,  4,  4*13,  13,  1,     13,     0,    0,    78,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    263,  1,       27,  4,  5*13,  13,  1,     13,     0,    0,    78,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
 % F08, 30 MHz 6 UEGs, each with 3 PRBs (with ssb), OTA (addPos) 
    264,  1,       27,  4,  20+0*3,  3,  2,     12,     0,    0,    78,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    265,  1,       27,  4,  20+1*3,  3,  2,     12,     0,    0,    78,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    266,  1,       27,  4,  20+2*3,  3,  2,     12,     0,    0,    78,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    267,  1,       27,  4,  20+3*3,  3,  2,     12,     0,    0,    78,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    268,  1,       27,  4,  20+4*3,  3,  2,     12,     0,    0,    78,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    269,  1,       27,  4,  20+5*3,  3,  2,     12,     0,    0,    78,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % F08, 30 MHz 6 UEGs, each with 6 PRBs, OTA (addPos) 
    270,  1,       27,  4,  0*6,  6,  2,     12,     0,    0,    78,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    271,  1,       27,  4,  1*6,  6,  2,     12,     0,    0,    78,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    272,  1,       27,  4,  2*6,  6,  2,     12,     0,    0,    78,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    273,  1,       27,  4,  3*6,  6,  2,     12,     0,    0,    78,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    274,  1,       27,  4,  4*6,  6,  2,     12,     0,    0,    78,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    275,  1,       27,  4,  5*6,  6,  2,     12,     0,    0,    78,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % F08, 30 MHz 6 UEGs, each with 4 PRBs (with disjoint csirs), OTA (addPos) 
    276,  1,       27,  4,  0*4,  4,  2,     10,     0,    0,    78,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    277,  1,       27,  4,  1*4,  4,  2,     10,     0,    0,    78,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    278,  1,       27,  4,  2*4,  4,  2,     10,     0,    0,    78,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    279,  1,       27,  4,  3*4,  4,  2,     10,     0,    0,    78,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    280,  1,       27,  4,  4*4,  4,  2,     10,     0,    0,    78,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    281,  1,       27,  4,  5*4,  4,  2,     10,     0,    0,    78,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
 % F08, 50 MHz 6 UEGs, each with 18 PRBs (with ssb), OTA (addPos) 
    282,  1,       27,  4,  20+0*18,  18,  1,     13,     0,    0,    133,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    283,  1,       27,  4,  20+1*18,  18,  1,     13,     0,    0,    133,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    284,  1,       27,  4,  20+2*18,  18,  1,     13,     0,    0,    133,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    285,  1,       27,  4,  20+3*18,  18,  1,     13,     0,    0,    133,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    286,  1,       27,  4,  20+4*18,  18,  1,     13,     0,    0,    133,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    287,  1,       27,  4,  20+5*18,  18,  1,     13,     0,    0,    133,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
   % F08, 50 MHz 6 UEGs, each with 22 PRBs, OTA (addPos) 
    288,  1,       27,  4,  0*22,  22,  1,     13,     0,    0,    133,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    289,  1,       27,  4,  1*22,  22,  1,     13,     0,    0,    133,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    290,  1,       27,  4,  2*22,  22,  1,     13,     0,    0,    133,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    291,  1,       27,  4,  3*22,  22,  1,     13,     0,    0,    133,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    292,  1,       27,  4,  4*22,  22,  1,     13,     0,    0,    133,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    293,  1,       27,  4,  5*22,  22,  1,     13,     0,    0,    133,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
 % F08, 50 MHz 6 UEGs, each with 7 PRBs (with ssb), OTA (addPos) 
    294,  1,       27,  4,  20+0*7,  7,  2,     12,     0,    0,    133,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    295,  1,       27,  4,  20+1*7,  7,  2,     12,     0,    0,    133,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    296,  1,       27,  4,  20+2*7,  7,  2,     12,     0,    0,    133,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    297,  1,       27,  4,  20+3*7,  7,  2,     12,     0,    0,    133,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    298,  1,       27,  4,  20+4*7,  7,  2,     12,     0,    0,    133,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    299,  1,       27,  4,  20+5*7,  7,  2,     12,     0,    0,    133,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % F08, 50 MHz 6 UEGs, each with 11 PRBs, OTA (addPos) 
    300,  1,       27,  4,  0*11,  11,  2,     12,     0,    0,    133,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    301,  1,       27,  4,  1*11,  11,  2,     12,     0,    0,    133,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    302,  1,       27,  4,  2*11,  11,  2,     12,     0,    0,    133,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    303,  1,       27,  4,  3*11,  11,  2,     12,     0,    0,    133,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    304,  1,       27,  4,  4*11,  11,  2,     12,     0,    0,    133,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    305,  1,       27,  4,  5*11,  11,  2,     12,     0,    0,    133,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % F08, 50 MHz 6 UEGs, each with 11 PRBs (with disjoint csirs), OTA (addPos) 
    306,  1,       27,  4,  0*11,  11,  2,     10,     0,    0,    133,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    307,  1,       27,  4,  1*11,  11,  2,     10,     0,    0,    133,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    308,  1,       27,  4,  2*11,  11,  2,     10,     0,    0,    133,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    309,  1,       27,  4,  3*11,  11,  2,     10,     0,    0,    133,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    310,  1,       27,  4,  4*11,  11,  2,     10,     0,    0,    133,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    311,  1,       27,  4,  5*11,  11,  2,     10,     0,    0,    133,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
% SRS MU-MIMO 
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    312,  1,       27,  2,  20, 253,  2,     12,     0,    0,    273,   211,    0,     41,     2,      2,     1,     41,      2,   0,   0; % 156
    313,  1,       27,  2,  20, 253,  2,     12,     0,    0,    273,   211,    0,     41,     2,      2,     1,     41,      2,   2,   0; % 156
    314,  1,       27,  2,  20, 253,  2,     12,     0,    0,    273,   211,    0,     41,     2,      2,     1,     41,      2,   4,   0; % 156
    315,  1,       27,  2,  20, 253,  2,     12,     0,    0,    273,   211,    0,     41,     2,      2,     1,     41,      2,   6,   0; % 156   
    316,  1,       27,  2,   0, 273,  2,     12,     0,    0,    273,   211,    0,     41,     2,      2,     1,     41,      2,   0,   0; % 157
    317,  1,       27,  2,   0, 273,  2,     12,     0,    0,    273,   211,    0,     41,     2,      2,     1,     41,      2,   2,   0; % 157
    318,  1,       27,  2,   0, 273,  2,     12,     0,    0,    273,   211,    0,     41,     2,      2,     1,     41,      2,   4,   0; % 157
    319,  1,       27,  2,   0, 273,  2,     12,     0,    0,    273,   211,    0,     41,     2,      2,     1,     41,      2,   6,   0; % 157
% HARQ TC    
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    320,  1,        1,  1,  20, 253,  2,     12,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      1,   0,   0; % pattern-1
    
    321,  1,       28,  1,  20, 253,  2,     12,     0,    0,    273,     1,    2,     41,     2,      1,     1,     41,      1,   0,   0; % pattern-2
    
    322,  1,        1,  4,  20,  80,  2,     12,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   0; % pattern-3
    323,  1,       10,  2, 100, 100,  2,     12,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      1,   0,   1; % pattern-3
    324,  1,       19,  4, 200,  73,  2,     12,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   2; % pattern-3
    
    325,  1,       27,  4,  20, 253,  2,      4,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   0; % pattern-9
    
    326,  1,       28,  4,   0, 100,  2,     12,     0,    0,    273,     1,    2,     41,     2,      1,     1,     41,      2,   0,   0; % pattern-4
    327,  1,       29,  2, 100, 100,  2,     12,     0,    0,    273,     2,    1,     41,     2,      1,     1,     41,      1,   0,   1; % pattern-4
    328,  1,       30,  4, 200,  73,  2,     12,     0,    0,    273,     3,    3,     41,     2,      1,     1,     41,      2,   0,   2; % pattern-4
    
    329,  1,        1,  4,   0,  50,  2,     12,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   0; % pattern-5
    330,  1,       10,  2,  50,  50,  2,     12,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      1,   0,   1; % pattern-5
    331,  1,       19,  4, 100,  50,  2,     12,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   2; % pattern-5
    332,  1,       27,  2, 150,  50,  2,     12,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      1,   0,   3; % pattern-5
    333,  1,       10,  4, 200,  50,  2,     12,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   4; % pattern-5
    334,  1,       19,  1, 250,  23,  2,     12,     0,    0,    273,     6,    0,     41,     2,      1,     1,     41,      1,   0,   5; % pattern-5
    
    335,  1,       28,  4,   0,  50,  2,     12,     0,    0,    273,     1,    1,     41,     2,      1,     1,     41,      2,   0,   0; % pattern-6
    336,  1,       29,  2,  50,  50,  2,     12,     0,    0,    273,     2,    3,     41,     2,      1,     1,     41,      1,   0,   1; % pattern-6
    337,  1,       30,  4, 100,  50,  2,     12,     0,    0,    273,     3,    1,     41,     2,      1,     1,     41,      2,   0,   2; % pattern-6
    338,  1,       31,  2, 150,  50,  2,     12,     0,    0,    273,     4,    2,     41,     2,      1,     1,     41,      1,   0,   3; % pattern-6
    339,  1,       29,  4, 200,  50,  2,     12,     0,    0,    273,     5,    1,     41,     2,      1,     1,     41,      2,   0,   4; % pattern-6
    340,  1,       30,  1, 250,  23,  2,     12,     0,    0,    273,     6,    2,     41,     2,      1,     1,     41,      1,   0,   5; % pattern-6    

    341,  1,       27,  4,   0,  50,  2,     12,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   0; % pattern-7
    342,  1,       27,  4,  50,  50,  2,     12,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   1; % pattern-7
    343,  1,       27,  4, 100,  50,  2,     12,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   2; % pattern-7
    344,  1,       27,  4, 150,  50,  2,     12,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   3; % pattern-7
    345,  1,       27,  4, 200,  50,  2,     12,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   4; % pattern-7
    346,  1,       27,  4, 250,  23,  2,     12,     0,    0,    273,     6,    0,     41,     2,      1,     1,     41,      2,   0,   5; % pattern-7    
    
    347,  1,       31,  4,   0,  50,  2,     12,     0,    0,    273,     1,    3,     41,     2,      1,     1,     41,      2,   0,   0; % pattern-8
    348,  1,       31,  4,  50,  50,  2,     12,     0,    0,    273,     2,    3,     41,     2,      1,     1,     41,      2,   0,   1; % pattern-8
    349,  1,       31,  4, 100,  50,  2,     12,     0,    0,    273,     3,    3,     41,     2,      1,     1,     41,      2,   0,   2; % pattern-8
    350,  1,       31,  4, 150,  50,  2,     12,     0,    0,    273,     4,    3,     41,     2,      1,     1,     41,      2,   0,   3; % pattern-8
    351,  1,       31,  4, 200,  50,  2,     12,     0,    0,    273,     5,    3,     41,     2,      1,     1,     41,      2,   0,   4; % pattern-8
    352,  1,       31,  4, 250,  23,  2,     12,     0,    0,    273,     6,    3,     41,     2,      1,     1,     41,      2,   0,   5; % pattern-8     

% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    353,  1,       31,  4,   0, 273,  2,      4,     0,    0,    273,     1,    3,     41,     2,      1,     0,     41,      2,   0,   0; % pattern-10
    
    354,  1,        1,  1,   0, 273,  2,     12,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      1,   0,   0; % pattern-1 w/o ssb
    
    355,  1,       28,  1,   0, 273,  2,     12,     0,    0,    273,     1,    2,     41,     2,      1,     1,     41,      1,   0,   0; % pattern-2 w/o ssb

    356,  1,        1,  4,   0, 100,  2,     12,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   0; % pattern-3 w/o ssb
    357,  1,       10,  2, 100, 100,  2,     12,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      1,   0,   1; % pattern-3 w/o ssb
    358,  1,       19,  4, 200,  73,  2,     12,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   2; % pattern-3 w/o ssb
    
    359,  1,       28,  4,   0, 100,  2,     12,     0,    0,    273,     1,    2,     41,     2,      1,     1,     41,      2,   0,   0; % pattern-4 w/o ssb
    360,  1,       29,  2, 100, 100,  2,     12,     0,    0,    273,     2,    1,     41,     2,      1,     1,     41,      1,   0,   1; % pattern-4 w/o ssb
    361,  1,       30,  4, 200,  73,  2,     12,     0,    0,    273,     3,    3,     41,     2,      1,     1,     41,      2,   0,   2; % pattern-4 w/o ssb
    
    362,  1,       27,  4,   0,  50,  2,      4,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   0; % pattern-11
    363,  1,       27,  4,  50,  50,  2,      4,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   1; % pattern-11
    364,  1,       27,  4, 100,  50,  2,      4,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   2; % pattern-11
    365,  1,       27,  4, 150,  50,  2,      4,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   3; % pattern-11
    366,  1,       27,  4, 200,  50,  2,      4,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   4; % pattern-11
    367,  1,       27,  4, 250,  23,  2,      4,     0,    0,    273,     6,    0,     41,     2,      1,     1,     41,      2,   0,   5; % pattern-11    
    
    368,  1,       31,  4,   0,  50,  2,      4,     0,    0,    273,     1,    3,     41,     2,      1,     1,     41,      2,   0,   0; % pattern-12
    369,  1,       31,  4,  50,  50,  2,      4,     0,    0,    273,     2,    3,     41,     2,      1,     1,     41,      2,   0,   1; % pattern-12
    370,  1,       31,  4, 100,  50,  2,      4,     0,    0,    273,     3,    3,     41,     2,      1,     1,     41,      2,   0,   2; % pattern-12
    371,  1,       31,  4, 150,  50,  2,      4,     0,    0,    273,     4,    3,     41,     2,      1,     1,     41,      2,   0,   3; % pattern-12
    372,  1,       31,  4, 200,  50,  2,      4,     0,    0,    273,     5,    3,     41,     2,      1,     1,     41,      2,   0,   4; % pattern-12
    373,  1,       31,  4, 250,  23,  2,      4,     0,    0,    273,     6,    3,     41,     2,      1,     1,     41,      2,   0,   5; % pattern-12        
    
    % for different number of UEs at different cells, OTA (addPos) 
    374,  1,       27,  4,  20,  31,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    375,  1,       27,  4,  51,  31,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    376,  1,       27,  4,  82,  31,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    377,  1,       27,  4,  113, 31,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    378,  1,       27,  4,  144, 31,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    379,  1,       27,  4,  175, 31,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    380,  1,       27,  4,  206, 31,  1,     13,     0,    0,    273,     6,    0,     41,     2,      1,     1,     41,      2,   0,   6;
    381,  1,       27,  4,  237, 31,  1,     13,     0,    0,    273,     7,    0,     41,     2,      1,     1,     41,      2,   0,   7;
    % 
    382,  1,       27,  4,  0,   34,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;   
    383,  1,       27,  4,  34,  34,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    384,  1,       27,  4,  68,  34,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    385,  1,       27,  4,  102, 34,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    386,  1,       27,  4,  136, 34,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    387,  1,       27,  4,  170, 34,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    388,  1,       27,  4,  204, 34,  1,     13,     0,    0,    273,     6,    0,     41,     2,      1,     1,     41,      2,   0,   6;
    389,  1,       27,  4,  238, 34,  1,     13,     0,    0,    273,     7,    0,     41,     2,      1,     1,     41,      2,   0,   7;
    % F08, light loaded, 6 UEGs, each with 3 PRBs (with ssb), OTA (addPos) 
    390,  1,       27,  4,  20+0*3,  3,  2,     12,     0,    0,    273,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    391,  1,       27,  4,  20+1*3,  3,  2,     12,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    392,  1,       27,  4,  20+2*3,  3,  2,     12,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    393,  1,       27,  4,  20+3*3,  3,  2,     12,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    394,  1,       27,  4,  20+4*3,  3,  2,     12,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    395,  1,       27,  4,  20+5*3,  3,  2,     12,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % F08, light loaded, 6 UEGs, each with 6 PRBs, OTA (addPos) 
    396,  1,       27,  4,  0*6,  6,  2,     12,     0,    0,    273,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    397,  1,       27,  4,  1*6,  6,  2,     12,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    398,  1,       27,  4,  2*6,  6,  2,     12,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    399,  1,       27,  4,  3*6,  6,  2,     12,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    400,  1,       27,  4,  4*6,  6,  2,     12,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    401,  1,       27,  4,  5*6,  6,  2,     12,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % F08, light loaded, 6 UEGs, each with 4 PRBs (with disjoint csirs), OTA (addPos) 
    402,  1,       27,  4,  0*4,  4,  2,     10,     0,    0,    273,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    403,  1,       27,  4,  1*4,  4,  2,     10,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    404,  1,       27,  4,  2*4,  4,  2,     10,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    405,  1,       27,  4,  3*4,  4,  2,     10,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    406,  1,       27,  4,  4*4,  4,  2,     10,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    407,  1,       27,  4,  5*4,  4,  2,     10,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
 % cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    % 64TR
    408,  1,        1,  4,  20,  80,  2,     12,     0,    0,    273,     1,    0,     41,     2,      2,     1,     41,      2,   0,   0; % 322
    409,  1,        1,  4,  20,  80,  2,     12,     0,    0,    273,     2,    0,     41,     2,      2,     1,     41,      2,   4,   0; % 322
    410,  1,        1,  4,  20,  80,  2,     12,     1,    0,    273,     3,    0,     41,     2,      2,     1,     41,      2,   0,   0; % 322
    411,  1,        1,  4,  20,  80,  2,     12,     1,    0,    273,     4,    0,     41,     2,      2,     1,     41,      2,   4,   0; % 322
    412,  1,       10,  2, 100, 100,  2,     12,     0,    0,    273,     5,    0,     41,     2,      2,     1,     41,      2,   0,   1; % 323
    413,  1,       19,  1, 200,  73,  2,     12,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   2; % 324

    % simultaneous DL/UL in S-slot
    414,  1,       19,  1, 200,  73,  2,     12,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   2; % 324
    415,  1,       19,  1,   0, 273,  1,      5,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   0; % 324    
    %408,  1,        1,  4,  20,  80,  2,     12,     0,    0,    273,     1,    0,     41,     2,      2,     1,     41,      2,   0,   0; 
    %409,  1,        1,  4,  20,  80,  2,     12,     0,    0,    273,     2,    0,     41,     2,      2,     1,     41,      2,   4,   0; 
    %410,  1,        1,  4,  20,  80,  2,     12,     1,    0,    273,     3,    0,     41,     2,      2,     1,     41,      2,   0,   0; 
    %411,  1,        1,  4,  20,  80,  2,     12,     1,    0,    273,     4,    0,     41,     2,      2,     1,     41,      2,   4,   0; 
    %412,  1,       10,  2, 100, 100,  2,     12,     0,    0,    273,     5,    0,     41,     2,      2,     1,     41,      2,   0,   1; 
    %413,  1,       10,  2, 100, 100,  2,     12,     0,    0,    273,     6,    0,     41,     2,      2,     1,     41,      2,   2,   1; 
    %414,  1,       19,  1, 200,  73,  2,     12,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   2; 
    
    %415,  1,       17,  4,  20, 253,  2,     12,     0,    0,    273,     1,    0,     41,     2,      2,     1,     41,      2,   0,   0; 
    416,  1,       17,  4,   0, 273,  2,     12,     0,    0,    273,     1,    0,     41,     2,      2,     1,     41,      2,   0,   0; 
    417,  1,       17,  4,   0, 273,  2,      4,     0,    0,    273,     1,    0,     41,     2,      2,     0,     41,      2,   0,   0; 
    
    418,  1,       17,  4,  20, 130,  2,     12,     0,    0,    273,     1,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    419,  1,       17,  4, 150, 123,  2,     12,     0,    0,    273,     2,    0,     41,     2,      2,     1,     41,      2,   0,   1; 
    420,  1,       17,  4,   0, 136,  2,     12,     0,    0,    273,     1,    0,     41,     2,      2,     1,     41,      2,   0,   0; 
    421,  1,       17,  4, 136, 137,  2,     12,     0,    0,    273,     2,    0,     41,     2,      2,     1,     41,      2,   0,   1;
    422,  1,       17,  4,   0, 136,  2,      4,     0,    0,    273,     1,    0,     41,     2,      2,     0,     41,      2,   0,   0;
    423,  1,       17,  4, 136, 137,  2,      4,     0,    0,    273,     2,    0,     41,     2,      2,     0,     41,      2,   0,   1;
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    % 64TR, 7 UEGs, 16 UEs (with ssb), OTA (addPos) 
    424,  1,       27,  4,  20,  21,  2,     12,     0,    0,    273,     1,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    425,  1,       27,  4,  20,  21,  2,     12,     0,    0,    273,     2,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    426,  1,       27,  4,  20,  21,  2,     12,     1,    0,    273,     3,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    427,  1,       27,  4,  20,  21,  2,     12,     1,    0,    273,     4,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    428,  1,       19,  4,  41,  19,  2,     12,     0,    0,    273,     5,    0,     41,     2,      2,     1,     41,      2,   0,   1;
    429,  1,       19,  4,  41,  19,  2,     12,     0,    0,    273,     6,    0,     41,     2,      2,     1,     41,      2,   4,   1;
    430,  1,       19,  4,  60,  19,  2,     12,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   2;
    431,  1,       19,  4,  60,  19,  2,     12,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   4,   2;
    432,  1,       19,  4,  79,  19,  2,     12,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   0,   3;
    433,  1,       19,  4,  79,  19,  2,     12,     0,    0,    273,     10,    0,     41,     2,      2,     1,     41,      2,   4,   3;
    434,  1,       19,  4,  98,  19,  2,     12,     0,    0,    273,     11,    0,     41,     2,      2,     1,     41,      2,   0,   4;
    435,  1,       19,  4,  98,  19,  2,     12,     0,    0,    273,     12,    0,     41,     2,      2,     1,     41,      2,   4,   4;
    436,  1,       10,  2,  117,  9,  2,     12,     0,    0,    273,     13,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    437,  1,       10,  2,  117,  9,  2,     12,     0,    0,    273,     14,    0,     41,     2,      1,     1,     41,      2,   2,   5;
    438,  1,       10,  2,  126,  9,  2,     12,     0,    0,    273,     15,    0,     41,     2,      1,     1,     41,      2,   0,   6;
    439,  1,       10,  2,  126,  9,  2,     12,     0,    0,    273,     16,    0,     41,     2,      1,     1,     41,      2,   2,   6;
 
    % 64TR, 7 UEGs, 16 UEs, OTA (addPos) 
    440,  1,       27,  4,  0,  24,  2,     12,     0,    0,    273,     1,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    441,  1,       27,  4,  0,  24,  2,     12,     0,    0,    273,     2,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    442,  1,       27,  4,  0,  24,  2,     12,     1,    0,    273,     3,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    443,  1,       27,  4,  0,  24,  2,     12,     1,    0,    273,     4,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    444,  1,       19,  4,  24,  22,  2,     12,     0,    0,    273,     5,    0,     41,     2,      2,     1,     41,      2,   0,   1;
    445,  1,       19,  4,  24,  22,  2,     12,     0,    0,    273,     6,    0,     41,     2,      2,     1,     41,      2,   4,   1;
    446,  1,       19,  4,  46,  22,  2,     12,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   2;
    447,  1,       19,  4,  46,  22,  2,     12,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   4,   2;
    448,  1,       19,  4,  68,  22,  2,     12,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   0,   3;
    449,  1,       19,  4,  68,  22,  2,     12,     0,    0,    273,     10,    0,     41,     2,      2,     1,     41,      2,   4,   3;
    450,  1,       19,  4,  90,  22,  2,     12,     0,    0,    273,     11,    0,     41,     2,      2,     1,     41,      2,   0,   4;
    451,  1,       19,  4,  90,  22,  2,     12,     0,    0,    273,     12,    0,     41,     2,      2,     1,     41,      2,   4,   4;
    452,  1,       10,  2,  112,  12,  2,     12,     0,    0,    273,     13,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    453,  1,       10,  2,  112,  12,  2,     12,     0,    0,    273,     14,    0,     41,     2,      1,     1,     41,      2,   2,   5;
    454,  1,       10,  2,  124,  12,  2,     12,     0,    0,    273,     15,    0,     41,     2,      1,     1,     41,      2,   0,   6;
    455,  1,       10,  2,  124,  12,  2,     12,     0,    0,    273,     16,    0,     41,     2,      1,     1,     41,      2,   2,   6;
    % 408,  1,        4,  2,  20,  80,  2,     12,     0,    0,    273,     1,    0,     41,     2,      2,     1,     41,      2,   0,   0; 
    % 409,  1,        1,  2,  20,  80,  2,     12,     0,    0,    273,     2,    0,     41,     2,      2,     1,     41,      2,   2,   0; 
    % 410,  1,        3,  2,  20,  80,  2,     12,     0,    0,    273,     3,    0,     41,     2,      2,     1,     41,      2,   4,   0; 
    % 411,  1,        2,  2,  20,  80,  2,     12,     0,    0,    273,     4,    0,     41,     2,      2,     1,     41,      2,   6,   0; 
    % 412,  1,       17,  2, 100, 100,  2,     12,     0,    0,    273,     5,    0,     41,     2,      2,     1,     41,      2,   0,   1; 
    % 413,  1,       10,  2, 100, 100,  2,     12,     0,    0,    273,     6,    0,     41,     2,      2,     1,     41,      2,   2,   1; 
    % 414,  1,       19,  1, 200,  73,  2,     12,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   2; 
    % 
    % 415,  1,       17,  2,  20, 253,  2,     12,     0,    0,    273,     1,    0,     41,     2,      2,     1,     41,      2,   0,   0; 
    % 416,  1,       17,  2,   0, 273,  2,     12,     0,    0,    273,     1,    0,     41,     2,      2,     1,     41,      2,   0,   0; 
    % 417,  1,       17,  2,   0, 273,  2,      4,     0,    0,    273,     1,    0,     41,     2,      2,     0,     41,      2,   0,   0; 
    % 
    % 418,  1,       17,  2,  20, 130,  2,     12,     0,    0,    273,     1,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    % 419,  1,       17,  2, 150, 123,  2,     12,     0,    0,    273,     2,    0,     41,     2,      2,     1,     41,      2,   0,   1; 
    % 420,  1,       17,  2,   0, 136,  2,     12,     0,    0,    273,     1,    0,     41,     2,      2,     1,     41,      2,   0,   0; 
    % 421,  1,       17,  2, 136, 137,  2,     12,     0,    0,    273,     2,    0,     41,     2,      2,     1,     41,      2,   0,   1;
    % 422,  1,       17,  2,   0, 136,  2,      4,     0,    0,    273,     1,    0,     41,     2,      2,     0,     41,      2,   0,   0;
    % 423,  1,       17,  2, 136, 137,  2,      4,     0,    0,    273,     2,    0,     41,     2,      2,     0,     41,      2,   0,   1;
    % 
    % 424,  1,       17,  2,  20, 130,  2,     12,     0,    0,    273,     1,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    % 425,  1,       17,  2,  20, 130,  2,     12,     0,    0,    273,     2,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    % 426,  1,       17,  2,  20, 130,  2,     12,     0,    0,    273,     3,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    % 427,  1,       17,  2,  20, 130,  2,     12,     0,    0,    273,     4,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    % 428,  1,       17,  2, 150,  50,  2,     12,     0,    0,    273,     5,    0,     41,     2,      2,     1,     41,      2,   0,   1;
    % 429,  1,       17,  1, 200,  73,  2,     12,     0,    0,    273,     6,    0,     41,     2,      2,     1,     41,      2,   0,   2;
    % 
    % 430,  1,       17,  2,   0, 150,  2,     12,     0,    0,    273,     1,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    % 431,  1,       17,  2,   0, 150,  2,     12,     0,    0,    273,     2,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    % 432,  1,       17,  2,   0, 150,  2,     12,     0,    0,    273,     3,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    % 433,  1,       17,  2,   0, 150,  2,     12,     0,    0,    273,     4,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    % 434,  1,       17,  2, 150,  50,  2,     12,     0,    0,    273,     5,    0,     41,     2,      2,     1,     41,      2,   0,   1;
    % 435,  1,       17,  1, 200,  73,  2,     12,     0,    0,    273,     6,    0,     41,     2,      2,     1,     41,      2,   0,   2;    
    % 
    % 436,  1,       17,  2,   0, 150,  2,      4,     0,    0,    273,     1,    0,     41,     2,      2,     0,     41,      2,   0,   0;
    % 437,  1,       17,  2,   0, 150,  2,      4,     0,    0,    273,     2,    0,     41,     2,      2,     0,     41,      2,   2,   0;
    % 438,  1,       17,  2,   0, 150,  2,      4,     0,    0,    273,     3,    0,     41,     2,      2,     0,     41,      2,   4,   0;
    % 439,  1,       17,  2,   0, 150,  2,      4,     0,    0,    273,     4,    0,     41,     2,      2,     0,     41,      2,   6,   0;
    % 440,  1,       17,  2, 150,  50,  2,      4,     0,    0,    273,     5,    0,     41,     2,      2,     0,     41,      2,   0,   1;
    % 441,  1,       17,  1, 200,  73,  2,      4,     0,    0,    273,     6,    0,     41,     2,      2,     0,     41,      2,   0,   2;
 % cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    % 64TR
      456,  1,        1,  1,  20,  80,  2,     12,     0,    0,    273,     1,    0,     41,     2,      2,     1,     41,      2,   0,   0;
      457,  1,        1,  1,  20,  80,  2,     12,     0,    0,    273,     2,    0,     41,     2,      2,     1,     41,      2,   1,   0;
      458,  1,        1,  1,  20,  80,  2,     12,     0,    0,    273,     3,    0,     41,     2,      2,     1,     41,      2,   2,   0;
      459,  1,        1,  1,  20,  80,  2,     12,     0,    0,    273,     4,    0,     41,     2,      2,     1,     41,      2,   3,   0;
      460,  1,        1,  1,  20,  80,  2,     12,     0,    0,    273,     5,    0,     41,     2,      2,     1,     41,      2,   4,   0;
      461,  1,        1,  1,  20,  80,  2,     12,     0,    0,    273,     6,    0,     41,     2,      2,     1,     41,      2,   5,   0;
      462,  1,        1,  1,  20,  80,  2,     12,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   6,   0;
      463,  1,        1,  1,  20,  80,  2,     12,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   7,   0;
% 64TR test case
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    464,  1,      27,   4,   0,  20,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   6;
    465,  1,      27,   4,  20,  20,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    466,  1,      27,   4,  40,  20,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    467,  1,      27,   4,  60,  20,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    468,  1,      27,   4,  80,  20,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    469,  1,      27,   4, 100,  20,  1,     13,     0,    0,    273,     6,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    470,  1,       2,   1, 120,  40,  1,     13,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   5;
    471,  1,       2,   1, 120,  40,  1,     13,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   1,   5;
    472,  1,       2,   1, 120,  40,  1,     13,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   2,   5;
    473,  1,       2,   1, 120,  40,  1,     13,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   3,   5;
    474,  1,       2,   1, 120,  40,  1,     13,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   4,   5;
    475,  1,       2,   1, 120,  40,  1,     13,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   5,   5;
    476,  1,       2,   1, 120,  40,  1,     13,     0,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   6,   5;
    477,  1,       2,   1, 120,  40,  1,     13,     0,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   7,   5;

    478,  1,       27,  1,   0,  20,  2,     12,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    479,  1,       27,  2,  20,  20,  2,     12,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    480,  1,       27,  3,  40,  20,  2,     12,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    481,  1,       27,  4,  60,  20,  2,     12,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    482,  1,       27,  4,  80,  20,  2,     12,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    483,  1,       27,  4, 100,  20,  2,     12,     0,    0,    273,     6,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    484,  1,       2,   1, 120,  40,  2,     12,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   6;
    485,  1,       2,   2, 120,  40,  2,     12,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   1,   6;
    
    486,  1,       27,  1,   0,  20,  1,      5,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    487,  1,       27,  2,  20,  20,  1,      5,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    488,  1,       27,  3,  40,  20,  1,      5,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    489,  1,       27,  4,  60,  20,  1,      5,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    490,  1,       27,  1,  80,  20,  1,      5,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    491,  1,       27,  2, 100, 173,  1,      5,     0,    0,    273,     6,    0,     41,     2,      1,     1,     41,      2,   0,   5;

% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg    
    492,  1,       27,  2,   0, 273,  0,     14,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    493,  1,       27,  2,   0, 273,  0,     14,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   2,   0;
    494,  1,       27,  2,   0, 273,  0,     14,     0,    0,    273,     6,    0,     41,     2,      1,     1,     41,      2,   4,   0;
    495,  1,       27,  2,   0, 273,  0,     14,     0,    0,    273,     7,    0,     41,     2,      1,     1,     41,      2,   6,   0;
    % for F08 6 UEGs, each with 42 PRBs, start with 20 5 symbols 
    496,  1,       27,  4,  20,  42,  1,      5,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;
    497,  1,       27,  4,  62,  42,  1,      5,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    498,  1,       27,  4,  104, 42,  1,      5,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    499,  1,       27,  4,  146, 42,  1,      5,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    500,  1,       27,  4,  188, 42,  1,      5,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    501,  1,       27,  4,  230, 42,  1,      5,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    % for F08 peak, 6 UEGs, each with 45 PRBs, start with 0, 5 symbols 
    502,  1,       27,  4,  0,   45,  1,      5,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;   
    503,  1,       27,  4,  45,  45,  1,      5,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    504,  1,       27,  4,  90,  45,  1,      5,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    505,  1,       27,  4,  135, 45,  1,      5,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    506,  1,       27,  4,  180, 45,  1,      5,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    507,  1,       27,  4,  225, 45,  1,      5,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    % F08, 6 UEGs, each with 18 PRBs (with ssb), 5 symbols 
    508,  1,       27,  4,  20,  18,  1,      5,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;
    509,  1,       27,  4,  38,  18,  1,      5,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    510,  1,       27,  4,  56,  18,  1,      5,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    511,  1,       27,  4,  74,  18,  1,      5,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    512,  1,       27,  4,  92,  18,  1,      5,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    513,  1,       27,  4,  110, 18,  1,      5,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    % for F08, 6 UEGs, each with 22 PRBs, start with 0, 5 symbols 
    514,  1,       27,  4,  0,   22,  1,      5,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;   
    515,  1,       27,  4,  22,  22,  1,      5,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    516,  1,       27,  4,  44,  22,  1,      5,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    517,  1,       27,  4,  66,  22,  1,      5,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    518,  1,       27,  4,  88,  22,  1,      5,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    519,  1,       27,  4,  110, 22,  1,      5,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    % 64TR, 1 UEGs, 4 UEs (with ssb), OTA (addPos) 
    520,  1,       27,  4,  20,  253,  2,     12,     0,    0,    273,     1,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    521,  1,       27,  4,  20,  253,  2,     12,     0,    0,    273,     2,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    522,  1,       27,  4,  20,  253,  2,     12,     1,    0,    273,     3,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    523,  1,       27,  4,  20,  253,  2,     12,     1,    0,    273,     4,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    % 64TR, 1 UEGs, 4 UEs, OTA (addPos) 
    524,  1,       27,  4,  0,  273,  2,     12,     0,    0,    273,     1,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    525,  1,       27,  4,  0,  273,  2,     12,     0,    0,    273,     2,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    526,  1,       27,  4,  0,  273,  2,     12,     1,    0,    273,     3,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    527,  1,       27,  4,  0,  273,  2,     12,     1,    0,    273,     4,    0,     41,     2,      2,     1,     41,      2,   4,   0;
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg 
    % 64TR, 1 UEGs, 16 UEs, 12 symbols
    528,  1,       27,   1,  0,  273,  2,   12,     0,    0,    273,     1,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    529,  1,       27,   1,  0,  273,  2,   12,     0,    0,    273,     2,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    530,  1,       27,   1,  0,  273,  2,   12,     0,    0,    273,     3,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    531,  1,       27,   1,  0,  273,  2,   12,     0,    0,    273,     4,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    532,  1,       27,   1,  0,  273,  2,   12,     0,    0,    273,     5,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    533,  1,       27,   1,  0,  273,  2,   12,     0,    0,    273,     6,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    534,  1,       27,   1,  0,  273,  2,   12,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    535,  1,       27,   1,  0,  273,  2,   12,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    536,  1,       27,   1,  0,  273,  2,   12,     1,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    537,  1,       27,   1,  0,  273,  2,   12,     1,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    538,  1,       27,   1,  0,  273,  2,   12,     1,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    539,  1,       27,   1,  0,  273,  2,   12,     1,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    540,  1,       27,   1,  0,  273,  2,   12,     1,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    541,  1,       27,   1,  0,  273,  2,   12,     1,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    542,  1,       27,   1,  0,  273,  2,   12,     1,    0,    273,    15,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    543,  1,       27,   1,  0,  273,  2,   12,     1,    0,    273,    16,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % 64TR, 1 UEGs, 8 UEs, 12 symbols
    544,  1,       27,   2,  0,  273,  2,   12,     0,    0,    273,     1,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    545,  1,       27,   2,  0,  273,  2,   12,     0,    0,    273,     2,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    546,  1,       27,   2,  0,  273,  2,   12,     0,    0,    273,     3,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    547,  1,       27,   2,  0,  273,  2,   12,     0,    0,    273,     4,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    548,  1,       27,   2,  0,  273,  2,   12,     1,    0,    273,     5,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    549,  1,       27,   2,  0,  273,  2,   12,     1,    0,    273,     6,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    550,  1,       27,   2,  0,  273,  2,   12,     1,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    551,  1,       27,   2,  0,  273,  2,   12,     1,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   6,   0;
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg 
    % 64TR test case 'worst case col B'
    % SU-MIMO, 4 layer per UE, FDM
    552,  1,      27,   4,   0,   50,  1,   13,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    553,  1,      27,   4,  50,   50,  1,   13,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    554,  1,      27,   4, 100,   50,  1,   13,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    555,  1,      27,   4, 150,   50,  1,   13,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    556,  1,      27,   4, 200,   50,  1,   13,     0,    0,    273,     6,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    557,  1,      27,   4, 250,   23,  1,   13,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    558,  1,      27,   4,   0,   50,  2,   12,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    559,  1,      27,   4,  50,   50,  2,   12,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    560,  1,      27,   4, 100,   50,  2,   12,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    561,  1,      27,   4, 150,   50,  2,   12,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    562,  1,      27,   4, 200,   50,  2,   12,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    563,  1,      27,   4, 250,   23,  2,   12,     0,    0,    273,     6,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    564,  1,      27,   4,   0,   50,  1,    5,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    565,  1,      27,   4,  50,   50,  1,    5,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    566,  1,      27,   4, 100,   50,  1,    5,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    567,  1,      27,   4, 150,   50,  1,    5,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    568,  1,      27,   4, 200,   50,  1,    5,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    569,  1,      27,   4, 250,   23,  1,    5,     0,    0,    273,     6,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % MU-MIMO, 8 UE, 1 layer per UE, PRB 0~272, OFDM symbol 1~13
    570,  1,      27,   1,   0,  273,  1,   13,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    571,  1,      27,   1,   0,  273,  1,   13,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    572,  1,      27,   1,   0,  273,  1,   13,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    573,  1,      27,   1,   0,  273,  1,   13,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    574,  1,      27,   1,   0,  273,  1,   13,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    575,  1,      27,   1,   0,  273,  1,   13,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    576,  1,      27,   1,   0,  273,  1,   13,     0,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    577,  1,      27,   1,   0,  273,  1,   13,     0,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % MU-MIMO, 4 UE, 2 layer per UE, PRB 0~272, OFDM symbol 1~13
    578,  1,      27,   2,   0,  273,  1,   13,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    579,  1,      27,   2,   0,  273,  1,   13,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    580,  1,      27,   2,   0,  273,  1,   13,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    581,  1,      27,   2,   0,  273,  1,   13,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    % SU-MIMO, 4 layer per UE, FDM
    582,  1,      27,   4,  20,   30,  1,   13,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    % MU-MIMO, 8 UE, 1 layer per UE, PRB 20~272, OFDM symbol 1~13
    583,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    584,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    585,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    586,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    587,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    588,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    589,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    590,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % MU-MIMO, 8 UE, 1 layer per UE, PRB 0~272, OFDM symbol 2~13
    591,  1,      27,   1,   0,  273,  2,   12,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    592,  1,      27,   1,   0,  273,  2,   12,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    593,  1,      27,   1,   0,  273,  2,   12,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    594,  1,      27,   1,   0,  273,  2,   12,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    595,  1,      27,   1,   0,  273,  2,   12,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    596,  1,      27,   1,   0,  273,  2,   12,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    597,  1,      27,   1,   0,  273,  2,   12,     0,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    598,  1,      27,   1,   0,  273,  2,   12,     0,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % for F08 6 UEGs, each with 42/43 PRBs, OTA (addPos), PDSCH mcsTable 2, modified from 158:163
    599,  1,       27,  4,  20,  42,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    600,  1,       27,  4,  62,  42,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    601,  1,       27,  4,  104, 42,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    602,  1,       27,  4,  146, 42,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    603,  1,       27,  4,  188, 42,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    604,  1,       27,  4,  230, 43,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % for F08 peak, 6 UEGs, each with 45/46 PRBs, start with 0, OTA (addPos), PDSCH mcsTable 2, modified from 164:169 
    605,  1,       27,  4,  0,   45,  1,     13,     0,    0,    273,     0,    0,     41,     2,      1,     1,     41,      2,   0,   0;   
    606,  1,       27,  4,  45,  45,  1,     13,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    607,  1,       27,  4,  90,  45,  1,     13,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    608,  1,       27,  4,  135, 46,  1,     13,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    609,  1,       27,  4,  181, 46,  1,     13,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    610,  1,       27,  4,  227, 46,  1,     13,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % for F08 6 UEGs, each with 42/43 PRBs, start with 20 5 symbols, PDSCH mcsTable 2, modified from 496:501
    611,  1,       27,  4,  20,  42,  1,      5,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;
    612,  1,       27,  4,  62,  42,  1,      5,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    613,  1,       27,  4,  104, 42,  1,      5,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    614,  1,       27,  4,  146, 42,  1,      5,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    615,  1,       27,  4,  188, 42,  1,      5,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    616,  1,       27,  4,  230, 43,  1,      5,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;
    % for F08 peak, 6 UEGs, each with 45/46 PRBs, start with 0, 5 symbols, PDSCH mcsTable 2, modified from 502:507
    617,  1,       27,  4,  0,   45,  1,      5,     0,    0,    273,     0,    0,     41,     2,      1,     0,     41,      2,   0,   0;   
    618,  1,       27,  4,  45,  45,  1,      5,     0,    0,    273,     1,    0,     41,     2,      1,     0,     41,      2,   0,   1;
    619,  1,       27,  4,  90,  45,  1,      5,     0,    0,    273,     2,    0,     41,     2,      1,     0,     41,      2,   0,   2;
    620,  1,       27,  4,  135, 46,  1,      5,     0,    0,    273,     3,    0,     41,     2,      1,     0,     41,      2,   0,   3;
    621,  1,       27,  4,  181, 46,  1,      5,     0,    0,    273,     4,    0,     41,     2,      1,     0,     41,      2,   0,   4;
    622,  1,       27,  4,  227, 46,  1,      5,     0,    0,    273,     5,    0,     41,     2,      1,     0,     41,      2,   0,   5;    
    % 64TR test case 'worst case col G'
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg 
    % MU-MIMO, 16 UE, 1 layer per UE, PRB 0~272, OFDM symbol 1~13 (for multiplexing with to 578:581)
    623,  1,      27,   1,   0,  273,  1,   13,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    624,  1,      27,   1,   0,  273,  1,   13,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    625,  1,      27,   1,   0,  273,  1,   13,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    626,  1,      27,   1,   0,  273,  1,   13,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    627,  1,      27,   1,   0,  273,  1,   13,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    628,  1,      27,   1,   0,  273,  1,   13,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    629,  1,      27,   1,   0,  273,  1,   13,     0,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    630,  1,      27,   1,   0,  273,  1,   13,     0,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    631,  1,      27,   1,   0,  273,  1,   13,     1,    0,    273,    15,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    632,  1,      27,   1,   0,  273,  1,   13,     1,    0,    273,    16,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    633,  1,      27,   1,   0,  273,  1,   13,     1,    0,    273,    17,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    634,  1,      27,   1,   0,  273,  1,   13,     1,    0,    273,    18,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    635,  1,      27,   1,   0,  273,  1,   13,     1,    0,    273,    19,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    636,  1,      27,   1,   0,  273,  1,   13,     1,    0,    273,    20,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    637,  1,      27,   1,   0,  273,  1,   13,     1,    0,    273,    21,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    638,  1,      27,   1,   0,  273,  1,   13,     1,    0,    273,    22,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % MU-MIMO, 16 UE, 1 layer per UE, PRB 0~272, OFDM symbol 2~13
    639,  1,      27,   1,   0,  273,  2,   12,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    640,  1,      27,   1,   0,  273,  2,   12,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    641,  1,      27,   1,   0,  273,  2,   12,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    642,  1,      27,   1,   0,  273,  2,   12,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    643,  1,      27,   1,   0,  273,  2,   12,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    644,  1,      27,   1,   0,  273,  2,   12,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    645,  1,      27,   1,   0,  273,  2,   12,     0,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    646,  1,      27,   1,   0,  273,  2,   12,     0,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    647,  1,      27,   1,   0,  273,  2,   12,     1,    0,    273,    15,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    648,  1,      27,   1,   0,  273,  2,   12,     1,    0,    273,    16,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    649,  1,      27,   1,   0,  273,  2,   12,     1,    0,    273,    17,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    650,  1,      27,   1,   0,  273,  2,   12,     1,    0,    273,    18,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    651,  1,      27,   1,   0,  273,  2,   12,     1,    0,    273,    19,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    652,  1,      27,   1,   0,  273,  2,   12,     1,    0,    273,    20,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    653,  1,      27,   1,   0,  273,  2,   12,     1,    0,    273,    21,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    654,  1,      27,   1,   0,  273,  2,   12,     1,    0,    273,    22,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % SU-MIMO, 4 layer per UE, FDM
    %643,  1,      27,   4,  20,   50,  2,   12,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    655,  1,      27,   4,  20,   30,  2,   12,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    % MU-MIMO, 16 UE, 1 layer per UE, PRB 20~272, OFDM symbol 1~13
  %  639,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
  %  640,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
  %  641,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
  %  642,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
  %  643,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
  %  644,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
  %  645,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
  %  646,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
  %  647,  1,      27,   1,  20,  253,  1,   13,     1,    0,    273,    15,    0,     41,     2,      2,     1,     41,      2,   0,   0;
  %  648,  1,      27,   1,  20,  253,  1,   13,     1,    0,    273,    16,    0,     41,     2,      2,     1,     41,      2,   1,   0;
  %  649,  1,      27,   1,  20,  253,  1,   13,     1,    0,    273,    17,    0,     41,     2,      2,     1,     41,      2,   2,   0;
  %  650,  1,      27,   1,  20,  253,  1,   13,     1,    0,    273,    18,    0,     41,     2,      2,     1,     41,      2,   3,   0;
  %  651,  1,      27,   1,  20,  253,  1,   13,     1,    0,    273,    19,    0,     41,     2,      2,     1,     41,      2,   4,   0;
  %  652,  1,      27,   1,  20,  253,  1,   13,     1,    0,    273,    20,    0,     41,     2,      2,     1,     41,      2,   5,   0;
  %  653,  1,      27,   1,  20,  253,  1,   13,     1,    0,    273,    21,    0,     41,     2,      2,     1,     41,      2,   6,   0;
  %  654,  1,      27,   1,  20,  253,  1,   13,     1,    0,    273,    22,    0,     41,     2,      2,     1,     41,      2,   7,   0;
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    % 64TR test case 'worst case col B', ave
    % SU-MIMO, 4 layer per UE, FDM
    656,  1,      27,   4,   0,   25,  1,   13,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    657,  1,      27,   4,  25,   25,  1,   13,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    658,  1,      27,   4,  50,   25,  1,   13,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    659,  1,      27,   4,  75,   25,  1,   13,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    660,  1,      27,   4, 100,   25,  1,   13,     0,    0,    273,     6,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    661,  1,      27,   4, 125,   11,  1,   13,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    662,  1,      27,   4,   0,   25,  2,   12,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    663,  1,      27,   4,  25,   25,  2,   12,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    664,  1,      27,   4,  50,   25,  2,   12,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    665,  1,      27,   4,  75,   25,  2,   12,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    666,  1,      27,   4, 100,   25,  2,   12,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    667,  1,      27,   4, 125,   11,  2,   12,     0,    0,    273,     6,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    668,  1,      27,   4,   0,   25,  1,    5,     0,    0,    273,     1,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    669,  1,      27,   4,  25,   25,  1,    5,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   1;
    670,  1,      27,   4,  50,   25,  1,    5,     0,    0,    273,     3,    0,     41,     2,      1,     1,     41,      2,   0,   2;
    671,  1,      27,   4,  75,   25,  1,    5,     0,    0,    273,     4,    0,     41,     2,      1,     1,     41,      2,   0,   3;
    672,  1,      27,   4, 100,   25,  1,    5,     0,    0,    273,     5,    0,     41,     2,      1,     1,     41,      2,   0,   4;
    673,  1,      27,   4, 125,   11,  1,    5,     0,    0,    273,     6,    0,     41,     2,      1,     1,     41,      2,   0,   5;
    % MU-MIMO, 8 UE, 1 layer per UE, PRB 0~135, OFDM symbol 1~13
    674,  1,      27,   1,   0,  136,  1,   13,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    675,  1,      27,   1,   0,  136,  1,   13,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    676,  1,      27,   1,   0,  136,  1,   13,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    677,  1,      27,   1,   0,  136,  1,   13,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    678,  1,      27,   1,   0,  136,  1,   13,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    679,  1,      27,   1,   0,  136,  1,   13,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    680,  1,      27,   1,   0,  136,  1,   13,     0,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    681,  1,      27,   1,   0,  136,  1,   13,     0,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % MU-MIMO, 4 UE, 2 layer per UE, PRB 0~135, OFDM symbol 1~13
    682,  1,      27,   2,   0,  136,  1,   13,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    683,  1,      27,   2,   0,  136,  1,   13,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    684,  1,      27,   2,   0,  136,  1,   13,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    685,  1,      27,   2,   0,  136,  1,   13,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    % SU-MIMO, 4 layer per UE, FDM
    686,  1,      27,   4,  20,   5,  1,   13,     0,    0,    273,     2,    0,     41,     2,      1,     1,     41,      2,   0,   0;
    % MU-MIMO, 8 UE, 1 layer per UE, PRB 20~135, OFDM symbol 1~13
    687,  1,      27,   1,  20,  116,  1,   13,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    688,  1,      27,   1,  20,  116,  1,   13,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    689,  1,      27,   1,  20,  116,  1,   13,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    690,  1,      27,   1,  20,  116,  1,   13,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    691,  1,      27,   1,  20,  116,  1,   13,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    692,  1,      27,   1,  20,  116,  1,   13,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    693,  1,      27,   1,  20,  116,  1,   13,     0,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    694,  1,      27,   1,  20,  116,  1,   13,     0,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % MU-MIMO, 8 UE, 1 layer per UE, PRB 0~135, OFDM symbol 2~13
    695,  1,      27,   1,   0,  136,  2,   12,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    696,  1,      27,   1,   0,  136,  2,   12,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    697,  1,      27,   1,   0,  136,  2,   12,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    698,  1,      27,   1,   0,  136,  2,   12,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    699,  1,      27,   1,   0,  136,  2,   12,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    700,  1,      27,   1,   0,  136,  2,   12,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    701,  1,      27,   1,   0,  136,  2,   12,     0,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    702,  1,      27,   1,   0,  136,  2,   12,     0,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % MU-MIMO, 8 UE, 1 layer per UE, PRB 0~272, OFDM symbol 1~5
    703,  1,      27,   1,   0,  273,  1,   5,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    704,  1,      27,   1,   0,  273,  1,   5,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    705,  1,      27,   1,   0,  273,  1,   5,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    706,  1,      27,   1,   0,  273,  1,   5,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    707,  1,      27,   1,   0,  273,  1,   5,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    708,  1,      27,   1,   0,  273,  1,   5,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    709,  1,      27,   1,   0,  273,  1,   5,     0,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    710,  1,      27,   1,   0,  273,  1,   5,     0,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % MU-MIMO, 8 UE, 1 layer per UE, PRB 0~135, OFDM symbol 1~5
    711,  1,      27,   1,   0,  136,  1,   5,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    712,  1,      27,   1,   0,  136,  1,   5,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    713,  1,      27,   1,   0,  136,  1,   5,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    714,  1,      27,   1,   0,  136,  1,   5,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    715,  1,      27,   1,   0,  136,  1,   5,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    716,  1,      27,   1,   0,  136,  1,   5,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    717,  1,      27,   1,   0,  136,  1,   5,     0,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    718,  1,      27,   1,   0,  136,  1,   5,     0,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg 
    % MU-MIMO, 16 UE, 1 layer per UE, PRB 20~272, OFDM symbol 1~13
    719,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    720,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    721,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    722,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    723,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    724,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    725,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    726,  1,      27,   1,  20,  253,  1,   13,     0,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    727,  1,      27,   1,  20,  253,  1,   13,     1,    0,    273,    15,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    728,  1,      27,   1,  20,  253,  1,   13,     1,    0,    273,    16,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    729,  1,      27,   1,  20,  253,  1,   13,     1,    0,    273,    17,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    730,  1,      27,   1,  20,  253,  1,   13,     1,    0,    273,    18,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    731,  1,      27,   1,  20,  253,  1,   13,     1,    0,    273,    19,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    732,  1,      27,   1,  20,  253,  1,   13,     1,    0,    273,    20,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    733,  1,      27,   1,  20,  253,  1,   13,     1,    0,    273,    21,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    734,  1,      27,   1,  20,  253,  1,   13,     1,    0,    273,    22,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % MU-MIMO, 16 UE, 1 layer per UE, PRB 20~272, OFDM symbol 2~13
    735,  1,      27,   1,  20,  253,  2,   12,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    736,  1,      27,   1,  20,  253,  2,   12,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    737,  1,      27,   1,  20,  253,  2,   12,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    738,  1,      27,   1,  20,  253,  2,   12,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    739,  1,      27,   1,  20,  253,  2,   12,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    740,  1,      27,   1,  20,  253,  2,   12,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    741,  1,      27,   1,  20,  253,  2,   12,     0,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    742,  1,      27,   1,  20,  253,  2,   12,     0,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    743,  1,      27,   1,  20,  253,  2,   12,     1,    0,    273,    15,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    744,  1,      27,   1,  20,  253,  2,   12,     1,    0,    273,    16,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    745,  1,      27,   1,  20,  253,  2,   12,     1,    0,    273,    17,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    746,  1,      27,   1,  20,  253,  2,   12,     1,    0,    273,    18,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    747,  1,      27,   1,  20,  253,  2,   12,     1,    0,    273,    19,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    748,  1,      27,   1,  20,  253,  2,   12,     1,    0,    273,    20,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    749,  1,      27,   1,  20,  253,  2,   12,     1,    0,    273,    21,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    750,  1,      27,   1,  20,  253,  2,   12,     1,    0,    273,    22,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % MU-MIMO, 16 UE, 1 layer per UE, PRB 0~272, OFDM symbol 1~5
    751,  1,      27,   1,   0,  273,  1,   5,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    752,  1,      27,   1,   0,  273,  1,   5,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    753,  1,      27,   1,   0,  273,  1,   5,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    754,  1,      27,   1,   0,  273,  1,   5,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    755,  1,      27,   1,   0,  273,  1,   5,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    756,  1,      27,   1,   0,  273,  1,   5,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    757,  1,      27,   1,   0,  273,  1,   5,     0,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    758,  1,      27,   1,   0,  273,  1,   5,     0,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    759,  1,      27,   1,   0,  273,  1,   5,     1,    0,    273,    15,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    760,  1,      27,   1,   0,  273,  1,   5,     1,    0,    273,    16,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    761,  1,      27,   1,   0,  273,  1,   5,     1,    0,    273,    17,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    762,  1,      27,   1,   0,  273,  1,   5,     1,    0,    273,    18,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    763,  1,      27,   1,   0,  273,  1,   5,     1,    0,    273,    19,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    764,  1,      27,   1,   0,  273,  1,   5,     1,    0,    273,    20,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    765,  1,      27,   1,   0,  273,  1,   5,     1,    0,    273,    21,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    766,  1,      27,   1,   0,  273,  1,   5,     1,    0,    273,    22,    0,     41,     2,      2,     1,     41,      2,   7,   0;
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg 
    % MU-MIMO, 16 UE, 1 layer per UE, PRB 20~105, OFDM symbol 1~13, 40 MHz
    767,  1,      27,   1,  20,  86,  1,   13,     0,    0,    106,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    768,  1,      27,   1,  20,  86,  1,   13,     0,    0,    106,     8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    769,  1,      27,   1,  20,  86,  1,   13,     0,    0,    106,     9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    770,  1,      27,   1,  20,  86,  1,   13,     0,    0,    106,    10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    771,  1,      27,   1,  20,  86,  1,   13,     0,    0,    106,    11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    772,  1,      27,   1,  20,  86,  1,   13,     0,    0,    106,    12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    773,  1,      27,   1,  20,  86,  1,   13,     0,    0,    106,    13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    774,  1,      27,   1,  20,  86,  1,   13,     0,    0,    106,    14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    775,  1,      27,   1,  20,  86,  1,   13,     1,    0,    106,    15,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    776,  1,      27,   1,  20,  86,  1,   13,     1,    0,    106,    16,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    777,  1,      27,   1,  20,  86,  1,   13,     1,    0,    106,    17,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    778,  1,      27,   1,  20,  86,  1,   13,     1,    0,    106,    18,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    779,  1,      27,   1,  20,  86,  1,   13,     1,    0,    106,    19,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    780,  1,      27,   1,  20,  86,  1,   13,     1,    0,    106,    20,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    781,  1,      27,   1,  20,  86,  1,   13,     1,    0,    106,    21,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    782,  1,      27,   1,  20,  86,  1,   13,     1,    0,    106,    22,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % MU-MIMO, 16 UE, 1 layer per UE, PRB 20~105, OFDM symbol 2~13, 40 MHz
    783,  1,      27,   1,  20,  86,  2,   12,     0,    0,    106,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    784,  1,      27,   1,  20,  86,  2,   12,     0,    0,    106,     8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    785,  1,      27,   1,  20,  86,  2,   12,     0,    0,    106,     9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    786,  1,      27,   1,  20,  86,  2,   12,     0,    0,    106,    10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    787,  1,      27,   1,  20,  86,  2,   12,     0,    0,    106,    11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    788,  1,      27,   1,  20,  86,  2,   12,     0,    0,    106,    12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    789,  1,      27,   1,  20,  86,  2,   12,     0,    0,    106,    13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    790,  1,      27,   1,  20,  86,  2,   12,     0,    0,    106,    14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    791,  1,      27,   1,  20,  86,  2,   12,     1,    0,    106,    15,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    792,  1,      27,   1,  20,  86,  2,   12,     1,    0,    106,    16,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    793,  1,      27,   1,  20,  86,  2,   12,     1,    0,    106,    17,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    794,  1,      27,   1,  20,  86,  2,   12,     1,    0,    106,    18,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    795,  1,      27,   1,  20,  86,  2,   12,     1,    0,    106,    19,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    796,  1,      27,   1,  20,  86,  2,   12,     1,    0,    106,    20,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    797,  1,      27,   1,  20,  86,  2,   12,     1,    0,    106,    21,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    798,  1,      27,   1,  20,  86,  2,   12,     1,    0,    106,    22,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % MU-MIMO, 16 UE, 1 layer per UE, PRB 0~105 OFDM symbol 1~5, 40 MHz
    799,  1,      27,   1,   0,  106,  1,   5,     0,    0,    106,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    800,  1,      27,   1,   0,  106,  1,   5,     0,    0,    106,     8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    801,  1,      27,   1,   0,  106,  1,   5,     0,    0,    106,     9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    802,  1,      27,   1,   0,  106,  1,   5,     0,    0,    106,    10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    803,  1,      27,   1,   0,  106,  1,   5,     0,    0,    106,    11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    804,  1,      27,   1,   0,  106,  1,   5,     0,    0,    106,    12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    805,  1,      27,   1,   0,  106,  1,   5,     0,    0,    106,    13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    806,  1,      27,   1,   0,  106,  1,   5,     0,    0,    106,    14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    807,  1,      27,   1,   0,  106,  1,   5,     1,    0,    106,    15,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    808,  1,      27,   1,   0,  106,  1,   5,     1,    0,    106,    16,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    809,  1,      27,   1,   0,  106,  1,   5,     1,    0,    106,    17,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    810,  1,      27,   1,   0,  106,  1,   5,     1,    0,    106,    18,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    811,  1,      27,   1,   0,  106,  1,   5,     1,    0,    106,    19,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    812,  1,      27,   1,   0,  106,  1,   5,     1,    0,    106,    20,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    813,  1,      27,   1,   0,  106,  1,   5,     1,    0,    106,    21,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    814,  1,      27,   1,   0,  106,  1,   5,     1,    0,    106,    22,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % MU-MIMO, 16 UE, 1 layer per UE, PRB 0~105, OFDM symbol 1~13, 40 MHz
    815,  1,      27,   1,   0,  106,  1,   13,     0,    0,    106,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    816,  1,      27,   1,   0,  106,  1,   13,     0,    0,    106,     8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    817,  1,      27,   1,   0,  106,  1,   13,     0,    0,    106,     9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    818,  1,      27,   1,   0,  106,  1,   13,     0,    0,    106,    10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    819,  1,      27,   1,   0,  106,  1,   13,     0,    0,    106,    11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    820,  1,      27,   1,   0,  106,  1,   13,     0,    0,    106,    12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    821,  1,      27,   1,   0,  106,  1,   13,     0,    0,    106,    13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    822,  1,      27,   1,   0,  106,  1,   13,     0,    0,    106,    14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    823,  1,      27,   1,   0,  106,  1,   13,     1,    0,    106,    15,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    824,  1,      27,   1,   0,  106,  1,   13,     1,    0,    106,    16,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    825,  1,      27,   1,   0,  106,  1,   13,     1,    0,    106,    17,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    826,  1,      27,   1,   0,  106,  1,   13,     1,    0,    106,    18,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    827,  1,      27,   1,   0,  106,  1,   13,     1,    0,    106,    19,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    828,  1,      27,   1,   0,  106,  1,   13,     1,    0,    106,    20,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    829,  1,      27,   1,   0,  106,  1,   13,     1,    0,    106,    21,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    830,  1,      27,   1,   0,  106,  1,   13,     1,    0,    106,    22,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    % MU-MIMO, 16 UE, 1 layer per UE, PRB 0~105, OFDM symbol 2~13, 40 MHz
    831,  1,      27,   1,   0,  106,  2,   12,     0,    0,    106,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    832,  1,      27,   1,   0,  106,  2,   12,     0,    0,    106,     8,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    833,  1,      27,   1,   0,  106,  2,   12,     0,    0,    106,     9,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    834,  1,      27,   1,   0,  106,  2,   12,     0,    0,    106,    10,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    835,  1,      27,   1,   0,  106,  2,   12,     0,    0,    106,    11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    836,  1,      27,   1,   0,  106,  2,   12,     0,    0,    106,    12,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    837,  1,      27,   1,   0,  106,  2,   12,     0,    0,    106,    13,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    838,  1,      27,   1,   0,  106,  2,   12,     0,    0,    106,    14,    0,     41,     2,      2,     1,     41,      2,   7,   0;
    839,  1,      27,   1,   0,  106,  2,   12,     1,    0,    106,    15,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    840,  1,      27,   1,   0,  106,  2,   12,     1,    0,    106,    16,    0,     41,     2,      2,     1,     41,      2,   1,   0;
    841,  1,      27,   1,   0,  106,  2,   12,     1,    0,    106,    17,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    842,  1,      27,   1,   0,  106,  2,   12,     1,    0,    106,    18,    0,     41,     2,      2,     1,     41,      2,   3,   0;
    843,  1,      27,   1,   0,  106,  2,   12,     1,    0,    106,    19,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    844,  1,      27,   1,   0,  106,  2,   12,     1,    0,    106,    20,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    845,  1,      27,   1,   0,  106,  2,   12,     1,    0,    106,    21,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    846,  1,      27,   1,   0,  106,  2,   12,     1,    0,    106,    22,    0,     41,     2,      2,     1,     41,      2,   7,   0;
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg 
    % 64TR PDSCH for CCH
    847,  1,      27,   1, 265,    8,  1,   13,     0,    0,    273, 65535,    0,     41,     2,      2,     1,     41,      2,   0,   1;
    % PDSCH TC3897/nvbug 5368243
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    848,  1,      27,   2,   0,  273,  1,   13,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    849,  1,      27,   1,   0,  273,  1,   13,     1,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    850,  1,      27,   1,   0,  273,  1,   13,     1,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   5,   0;
    851,  1,      27,   2,   0,  273,  1,   13,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    852,  1,      27,   2,   0,  273,  1,   13,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    853,  1,      27,   2,   0,  273,  1,   13,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    854,  1,      27,   2,   0,  273,  1,   13,     1,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    855,  1,      27,   2,   0,  273,  1,   13,     1,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    856,  1,      27,   1,   0,  273,  1,   13,     1,    0,    273,    15,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    857,  1,      27,   1,   0,  273,  1,   13,     1,    0,    273,    16,    0,     41,     2,      2,     1,     41,      2,   7,   0;

% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg 
    % additional heterogeneous UE groups, 4 UEGs with 4, 3, 2, 1 UEs per UEG respectively
    858,  1,      27,   4,  20,   80,  2,   12,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    859,  1,      27,   4,  20,   80,  2,   12,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    860,  1,      27,   4,  20,   80,  2,   12,     1,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    861,  1,      27,   4,  20,   80,  2,   12,     1,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    862,  1,      19,   2, 100,   50,  2,   12,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   0,   1;
    863,  1,      19,   2, 100,   50,  2,   12,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   2,   1;
    864,  1,      19,   2, 100,   50,  2,   12,     0,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   4,   1;
    865,  1,      10,   1, 150,   50,  2,   12,     0,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   0,   2;
    866,  1,      10,   1, 150,   50,  2,   12,     0,    0,    273,    15,    0,     41,     2,      2,     1,     41,      2,   1,   2;
    867,  1,       1,   1, 200,   73,  2,   12,     0,    0,    273,    16,    0,     41,     2,      2,     1,     41,      2,   0,   3;
    868,  1,      19,   4, 100,   50,  2,   12,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   0,   1;
    869,  1,      10,   4, 150,   50,  2,   12,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   0,   2;
    870,  1,       1,   4, 200,   73,  2,   12,     0,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   0,   3;

    % PDSCH TC3900/32DL
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    871,  1,      27,   2,   0,  273,  1,   13,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    872,  1,      27,   2,   0,  273,  1,   13,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    873,  1,      27,   2,   0,  273,  1,   13,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    874,  1,      27,   2,   0,  273,  1,   13,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    875,  1,      27,   2,   0,  273,  1,   13,     1,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    876,  1,      27,   2,   0,  273,  1,   13,     1,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    877,  1,      27,   2,   0,  273,  1,   13,     1,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    878,  1,      27,   2,   0,  273,  1,   13,     1,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    879,  1,      27,   2,   0,  273,  1,   13,     0,    0,    273,    15,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    880,  1,      27,   2,   0,  273,  1,   13,     0,    0,    273,    16,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    881,  1,      27,   2,   0,  273,  1,   13,     0,    0,    273,    17,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    882,  1,      27,   2,   0,  273,  1,   13,     0,    0,    273,    18,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    883,  1,      27,   2,   0,  273,  1,   13,     1,    0,    273,    19,    0,     41,     2,      2,     1,     41,      2,   0,   0;
    884,  1,      27,   2,   0,  273,  1,   13,     1,    0,    273,    20,    0,     41,     2,      2,     1,     41,      2,   2,   0;
    885,  1,      27,   2,   0,  273,  1,   13,     1,    0,    273,    21,    0,     41,     2,      2,     1,     41,      2,   4,   0;
    886,  1,      27,   2,   0,  273,  1,   13,     1,    0,    273,    22,    0,     41,     2,      2,     1,     41,      2,   6,   0;
    % 64TR PDSCH for CCH, UE group 4
    887,  1,      27,   1, 265,    8,  1,   13,     0,    0,    273, 65535,    0,     41,     2,      2,     1,     41,      2,   0,   4;
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    % PDSCH set 0: PRB 20~272, OFDM symbol 1~13, 100 MHz heavy
    888,  1,     19,   2,   20,   76,  1,   13,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
    889,  1,     19,   2,   20,   76,  1,   13,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
    890,  1,     19,   2,   20,   76,  1,   13,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
    891,  1,     19,   2,   20,   76,  1,   13,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
    892,  1,     19,   2,   20,   76,  1,   13,     1,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port8
    893,  1,     19,   2,   20,   76,  1,   13,     1,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port10
    894,  1,     19,   2,   20,   76,  1,   13,     1,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port12
    895,  1,     19,   2,   20,   76,  1,   13,     1,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port14
    896,  1,     10,   2,   96,   40,  1,   13,     0,    0,    273,    15,    0,     41,     2,      2,     1,     41,      2,   0,   1;  % Middle UEG1 port0
    897,  1,     10,   2,   96,   40,  1,   13,     0,    0,    273,    16,    0,     41,     2,      2,     1,     41,      2,   2,   1;  % Middle UEG1 port2
    898,  1,     10,   2,   96,   40,  1,   13,     0,    0,    273,    17,    0,     41,     2,      2,     1,     41,      2,   4,   1;  % Middle UEG1 port4
    899,  1,     10,   2,   96,   40,  1,   13,     0,    0,    273,    18,    0,     41,     2,      2,     1,     41,      2,   6,   1;  % Middle UEG1 port6
    900,  1,     10,   2,  136,   40,  1,   13,     0,    0,    273,    19,    0,     41,     2,      2,     1,     41,      2,   0,   2;  % Middle UEG2 port0
    901,  1,     10,   2,  136,   40,  1,   13,     0,    0,    273,    20,    0,     41,     2,      2,     1,     41,      2,   2,   2;  % Middle UEG2 port2
    902,  1,     10,   2,  136,   40,  1,   13,     0,    0,    273,    21,    0,     41,     2,      2,     1,     41,      2,   4,   2;  % Middle UEG2 port4
    903,  1,     10,   2,  136,   40,  1,   13,     0,    0,    273,    22,    0,     41,     2,      2,     1,     41,      2,   6,   2;  % Middle UEG2 port6
    904,  1,      4,   2,  176,   24,  1,   13,     0,    0,    273,    23,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
    905,  1,      4,   2,  200,   24,  1,   13,     0,    0,    273,    24,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
    906,  1,      4,   2,  224,   24,  1,   13,     0,    0,    273,    25,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
    907,  1,      4,   2,  248,   25,  1,   13,     0,    0,    273,    26,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH set 1: PRB 0~272, OFDM symbol 2~13, 100 MHz heavy
    908,  1,     19,   2,    0,   96,  2,   12,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
    909,  1,     19,   2,    0,   96,  2,   12,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
    910,  1,     19,   2,    0,   96,  2,   12,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
    911,  1,     19,   2,    0,   96,  2,   12,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
    912,  1,     19,   2,    0,   96,  2,   12,     1,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port8
    913,  1,     19,   2,    0,   96,  2,   12,     1,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port10
    914,  1,     19,   2,    0,   96,  2,   12,     1,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port12
    915,  1,     19,   2,    0,   96,  2,   12,     1,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port14
    916,  1,     10,   2,   96,   40,  2,   12,     0,    0,    273,    15,    0,     41,     2,      2,     1,     41,      2,   0,   1;  % Middle UEG1 port0
    917,  1,     10,   2,   96,   40,  2,   12,     0,    0,    273,    16,    0,     41,     2,      2,     1,     41,      2,   2,   1;  % Middle UEG1 port2
    918,  1,     10,   2,   96,   40,  2,   12,     0,    0,    273,    17,    0,     41,     2,      2,     1,     41,      2,   4,   1;  % Middle UEG1 port4
    919,  1,     10,   2,   96,   40,  2,   12,     0,    0,    273,    18,    0,     41,     2,      2,     1,     41,      2,   6,   1;  % Middle UEG1 port6
    920,  1,     10,   2,  136,   40,  2,   12,     0,    0,    273,    19,    0,     41,     2,      2,     1,     41,      2,   0,   2;  % Middle UEG2 port0
    921,  1,     10,   2,  136,   40,  2,   12,     0,    0,    273,    20,    0,     41,     2,      2,     1,     41,      2,   2,   2;  % Middle UEG2 port2
    922,  1,     10,   2,  136,   40,  2,   12,     0,    0,    273,    21,    0,     41,     2,      2,     1,     41,      2,   4,   2;  % Middle UEG2 port4
    923,  1,     10,   2,  136,   40,  2,   12,     0,    0,    273,    22,    0,     41,     2,      2,     1,     41,      2,   6,   2;  % Middle UEG2 port6
    924,  1,      4,   2,  176,   24,  2,   12,     0,    0,    273,    23,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
    925,  1,      4,   2,  200,   24,  2,   12,     0,    0,    273,    24,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
    926,  1,      4,   2,  224,   24,  2,   12,     0,    0,    273,    25,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
    927,  1,      4,   2,  248,   25,  2,   12,     0,    0,    273,    26,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH set 2: PRB 0~272, OFDM symbol 1~5, 100 MHz heavy
    928,  1,     19,   2,    0,   96,  1,    5,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
    929,  1,     19,   2,    0,   96,  1,    5,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
    930,  1,     19,   2,    0,   96,  1,    5,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
    931,  1,     19,   2,    0,   96,  1,    5,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
    932,  1,     19,   2,    0,   96,  1,    5,     1,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port8
    933,  1,     19,   2,    0,   96,  1,    5,     1,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port10
    934,  1,     19,   2,    0,   96,  1,    5,     1,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port12
    935,  1,     19,   2,    0,   96,  1,    5,     1,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port14
    936,  1,     10,   2,   96,   40,  1,    5,     0,    0,    273,    15,    0,     41,     2,      2,     1,     41,      2,   0,   1;  % Middle UEG1 port0
    937,  1,     10,   2,   96,   40,  1,    5,     0,    0,    273,    16,    0,     41,     2,      2,     1,     41,      2,   2,   1;  % Middle UEG1 port2
    938,  1,     10,   2,   96,   40,  1,    5,     0,    0,    273,    17,    0,     41,     2,      2,     1,     41,      2,   4,   1;  % Middle UEG1 port4
    939,  1,     10,   2,   96,   40,  1,    5,     0,    0,    273,    18,    0,     41,     2,      2,     1,     41,      2,   6,   1;  % Middle UEG1 port6
    940,  1,     10,   2,  136,   40,  1,    5,     0,    0,    273,    19,    0,     41,     2,      2,     1,     41,      2,   0,   2;  % Middle UEG2 port0
    941,  1,     10,   2,  136,   40,  1,    5,     0,    0,    273,    20,    0,     41,     2,      2,     1,     41,      2,   2,   2;  % Middle UEG2 port2
    942,  1,     10,   2,  136,   40,  1,    5,     0,    0,    273,    21,    0,     41,     2,      2,     1,     41,      2,   4,   2;  % Middle UEG2 port4
    943,  1,     10,   2,  136,   40,  1,    5,     0,    0,    273,    22,    0,     41,     2,      2,     1,     41,      2,   6,   2;  % Middle UEG2 port6
    944,  1,      4,   2,  176,   24,  1,    5,     0,    0,    273,    23,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
    945,  1,      4,   2,  200,   24,  1,    5,     0,    0,    273,    24,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
    946,  1,      4,   2,  224,   24,  1,    5,     0,    0,    273,    25,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
    947,  1,      4,   2,  248,   25,  1,    5,     0,    0,    273,    26,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH set 3: PRB 0~272, OFDM symbol 1~13, 100 MHz heavy
    % reuse cfg 896~907 from PDSCH set 0: PRB 20~272, OFDM symbol 1~13, 100 MHz heavy
    948,  1,     19,   2,    0,   96,  1,   13,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
    949,  1,     19,   2,    0,   96,  1,   13,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
    950,  1,     19,   2,    0,   96,  1,   13,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
    951,  1,     19,   2,    0,   96,  1,   13,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
    952,  1,     19,   2,    0,   96,  1,   13,     1,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port8
    953,  1,     19,   2,    0,   96,  1,   13,     1,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port10
    954,  1,     19,   2,    0,   96,  1,   13,     1,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port12
    955,  1,     19,   2,    0,   96,  1,   13,     1,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port14
    % PDSCH set 4: PRB 0~264, OFDM symbol 1~13, 100 MHz heavy
    % reuse cfg 896~906 from PDSCH set 0: PRB 20~272, OFDM symbol 1~13, 100 MHz heavy, cfg 948~955 from PDSCH set 3: PRB 0~272, OFDM symbol 1~13, 100 MHz heavy
    956,  1,      4,   2,  248,   17,  1,   13,     0,    0,    273,    26,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH for CCH
    957,  1,     27,   1,  265,    8,  1,   13,     0,    0,    273, 65535,    0,     41,     2,      2,     1,     41,      2,   0,   7;  % CCH config
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
    % PDSCH set 0: PRB 20~272, OFDM symbol 1~13, 100 MHz light
    958,  1,     19,   2,   20,   76,  1,   13,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
    959,  1,     19,   2,   20,   76,  1,   13,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
    960,  1,     19,   2,   20,   76,  1,   13,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
    961,  1,     19,   2,   20,   76,  1,   13,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
    962,  1,     10,   2,   96,   40,  1,   13,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   0,   1;  % Middle UEG1 port0
    963,  1,     10,   2,   96,   40,  1,   13,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   2,   1;  % Middle UEG1 port2
    964,  1,     10,   2,  136,   40,  1,   13,     0,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   0,   2;  % Middle UEG2 port0
    965,  1,     10,   2,  136,   40,  1,   13,     0,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   2,   2;  % Middle UEG2 port2
    966,  1,      4,   1,  176,   24,  1,   13,     0,    0,    273,    15,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
    967,  1,      4,   1,  200,   24,  1,   13,     0,    0,    273,    16,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
    968,  1,      4,   1,  224,   24,  1,   13,     0,    0,    273,    17,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
    969,  1,      4,   1,  248,   25,  1,   13,     0,    0,    273,    18,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH set 1: PRB 0~272, OFDM symbol 2~13, 100 MHz light
    970,  1,     19,   2,    0,   96,  2,   12,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
    971,  1,     19,   2,    0,   96,  2,   12,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
    972,  1,     19,   2,    0,   96,  2,   12,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
    973,  1,     19,   2,    0,   96,  2,   12,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
    974,  1,     10,   2,   96,   40,  2,   12,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   0,   1;  % Middle UEG1 port0
    975,  1,     10,   2,   96,   40,  2,   12,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   2,   1;  % Middle UEG1 port2
    976,  1,     10,   2,  136,   40,  2,   12,     0,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   0,   2;  % Middle UEG2 port0
    977,  1,     10,   2,  136,   40,  2,   12,     0,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   2,   2;  % Middle UEG2 port2
    978,  1,      4,   1,  176,   24,  2,   12,     0,    0,    273,    15,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
    979,  1,      4,   1,  200,   24,  2,   12,     0,    0,    273,    16,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
    980,  1,      4,   1,  224,   24,  2,   12,     0,    0,    273,    17,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
    981,  1,      4,   1,  248,   25,  2,   12,     0,    0,    273,    18,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH set 2: PRB 0~272, OFDM symbol 1~5, 100 MHz light
    982,  1,     19,   2,    0,   96,  1,    5,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
    983,  1,     19,   2,    0,   96,  1,    5,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
    984,  1,     19,   2,    0,   96,  1,    5,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
    985,  1,     19,   2,    0,   96,  1,    5,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
    986,  1,     10,   2,   96,   40,  1,    5,     0,    0,    273,    11,    0,     41,     2,      2,     1,     41,      2,   0,   1;  % Middle UEG1 port0
    987,  1,     10,   2,   96,   40,  1,    5,     0,    0,    273,    12,    0,     41,     2,      2,     1,     41,      2,   2,   1;  % Middle UEG1 port2
    988,  1,     10,   2,  136,   40,  1,    5,     0,    0,    273,    13,    0,     41,     2,      2,     1,     41,      2,   0,   2;  % Middle UEG2 port0
    989,  1,     10,   2,  136,   40,  1,    5,     0,    0,    273,    14,    0,     41,     2,      2,     1,     41,      2,   2,   2;  % Middle UEG2 port2
    990,  1,      4,   1,  176,   24,  1,    5,     0,    0,    273,    15,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
    991,  1,      4,   1,  200,   24,  1,    5,     0,    0,    273,    16,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
    992,  1,      4,   1,  224,   24,  1,    5,     0,    0,    273,    17,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
    993,  1,      4,   1,  248,   25,  1,    5,     0,    0,    273,    18,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH set 3: PRB 0~272, OFDM symbol 1~13, 100 MHz light
    % reuse cfg 962~969 from PDSCH set 0: PRB 20~272, OFDM symbol 1~13, 100 MHz light
    994,  1,     19,   2,    0,   96,  1,   13,     0,    0,    273,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
    995,  1,     19,   2,    0,   96,  1,   13,     0,    0,    273,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
    996,  1,     19,   2,    0,   96,  1,   13,     0,    0,    273,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
    997,  1,     19,   2,    0,   96,  1,   13,     0,    0,    273,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
    % PDSCH set 4: PRB 0~264, OFDM symbol 1~13, 100 MHz light
    % reuse cfg 962~968 from PDSCH set 0: PRB 20~272, OFDM symbol 1~13, 100 MHz light, cfg 994~997 from PDSCH set 3: PRB 0~272, OFDM symbol 1~13, 100 MHz light
    998,  1,      4,   1,  248,   17,  1,   13,     0,    0,    273,    18,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH for CCH
    999,  1,     27,   1,  265,    8,  1,   13,     0,    0,    273, 65535,    0,     41,     2,      2,     1,     41,      2,   0,   7;  % CCH config
   % cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
   % PDSCH set 0: PRB 20~243, OFDM symbol 1~13
   1000,  1,     19,   2,   20,   68,  1,   13,     0,    0,    244,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
   1001,  1,     19,   2,   20,   68,  1,   13,     0,    0,    244,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
   1002,  1,     19,   2,   20,   68,  1,   13,     0,    0,    244,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
   1003,  1,     19,   2,   20,   68,  1,   13,     0,    0,    244,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
   1004,  1,     19,   2,   20,   68,  1,   13,     1,    0,    244,    11,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port8
   1005,  1,     19,   2,   20,   68,  1,   13,     1,    0,    244,    12,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port10
   1006,  1,     19,   2,   20,   68,  1,   13,     1,    0,    244,    13,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port12
   1007,  1,     19,   2,   20,   68,  1,   13,     1,    0,    244,    14,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port14
   1008,  1,     10,   2,   88,   34,  1,   13,     0,    0,    244,    15,    0,     41,     2,      2,     1,     41,      2,   0,   1;  % Middle UEG1 port0
   1009,  1,     10,   2,   88,   34,  1,   13,     0,    0,    244,    16,    0,     41,     2,      2,     1,     41,      2,   2,   1;  % Middle UEG1 port2
   1010,  1,     10,   2,   88,   34,  1,   13,     0,    0,    244,    17,    0,     41,     2,      2,     1,     41,      2,   4,   1;  % Middle UEG1 port4
   1011,  1,     10,   2,   88,   34,  1,   13,     0,    0,    244,    18,    0,     41,     2,      2,     1,     41,      2,   6,   1;  % Middle UEG1 port6
   1012,  1,     10,   2,  122,   34,  1,   13,     0,    0,    244,    19,    0,     41,     2,      2,     1,     41,      2,   0,   2;  % Middle UEG2 port0
   1013,  1,     10,   2,  122,   34,  1,   13,     0,    0,    244,    20,    0,     41,     2,      2,     1,     41,      2,   2,   2;  % Middle UEG2 port2
   1014,  1,     10,   2,  122,   34,  1,   13,     0,    0,    244,    21,    0,     41,     2,      2,     1,     41,      2,   4,   2;  % Middle UEG2 port4
   1015,  1,     10,   2,  122,   34,  1,   13,     0,    0,    244,    22,    0,     41,     2,      2,     1,     41,      2,   6,   2;  % Middle UEG2 port6
   1016,  1,      4,   2,  156,   22,  1,   13,     0,    0,    244,    23,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
   1017,  1,      4,   2,  178,   22,  1,   13,     0,    0,    244,    24,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
   1018,  1,      4,   2,  200,   22,  1,   13,     0,    0,    244,    25,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
   1019,  1,      4,   2,  222,   22,  1,   13,     0,    0,    244,    26,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH set 1: PRB 0~243, OFDM symbol 2~13
   1020,  1,     19,   2,    0,   88,  2,   12,     0,    0,    244,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
   1021,  1,     19,   2,    0,   88,  2,   12,     0,    0,    244,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
   1022,  1,     19,   2,    0,   88,  2,   12,     0,    0,    244,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
   1023,  1,     19,   2,    0,   88,  2,   12,     0,    0,    244,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
   1024,  1,     19,   2,    0,   88,  2,   12,     1,    0,    244,    11,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port8
   1025,  1,     19,   2,    0,   88,  2,   12,     1,    0,    244,    12,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port10
   1026,  1,     19,   2,    0,   88,  2,   12,     1,    0,    244,    13,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port12
   1027,  1,     19,   2,    0,   88,  2,   12,     1,    0,    244,    14,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port14
   1028,  1,     10,   2,   88,   34,  2,   12,     0,    0,    244,    15,    0,     41,     2,      2,     1,     41,      2,   0,   1;  % Middle UEG1 port0
   1029,  1,     10,   2,   88,   34,  2,   12,     0,    0,    244,    16,    0,     41,     2,      2,     1,     41,      2,   2,   1;  % Middle UEG1 port2
   1030,  1,     10,   2,   88,   34,  2,   12,     0,    0,    244,    17,    0,     41,     2,      2,     1,     41,      2,   4,   1;  % Middle UEG1 port4
   1031,  1,     10,   2,   88,   34,  2,   12,     0,    0,    244,    18,    0,     41,     2,      2,     1,     41,      2,   6,   1;  % Middle UEG1 port6
   1032,  1,     10,   2,  122,   34,  2,   12,     0,    0,    244,    19,    0,     41,     2,      2,     1,     41,      2,   0,   2;  % Middle UEG2 port0
   1033,  1,     10,   2,  122,   34,  2,   12,     0,    0,    244,    20,    0,     41,     2,      2,     1,     41,      2,   2,   2;  % Middle UEG2 port2
   1034,  1,     10,   2,  122,   34,  2,   12,     0,    0,    244,    21,    0,     41,     2,      2,     1,     41,      2,   4,   2;  % Middle UEG2 port4
   1035,  1,     10,   2,  122,   34,  2,   12,     0,    0,    244,    22,    0,     41,     2,      2,     1,     41,      2,   6,   2;  % Middle UEG2 port6
   1036,  1,      4,   2,  156,   22,  2,   12,     0,    0,    244,    23,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
   1037,  1,      4,   2,  178,   22,  2,   12,     0,    0,    244,    24,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
   1038,  1,      4,   2,  200,   22,  2,   12,     0,    0,    244,    25,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
   1039,  1,      4,   2,  222,   22,  2,   12,     0,    0,    244,    26,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH set 2: PRB 0~243, OFDM symbol 1~5
   1040,  1,     19,   2,    0,   88,  1,    5,     0,    0,    244,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
   1041,  1,     19,   2,    0,   88,  1,    5,     0,    0,    244,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
   1042,  1,     19,   2,    0,   88,  1,    5,     0,    0,    244,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
   1043,  1,     19,   2,    0,   88,  1,    5,     0,    0,    244,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
   1044,  1,     19,   2,    0,   88,  1,    5,     1,    0,    244,    11,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port8
   1045,  1,     19,   2,    0,   88,  1,    5,     1,    0,    244,    12,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port10
   1046,  1,     19,   2,    0,   88,  1,    5,     1,    0,    244,    13,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port12
   1047,  1,     19,   2,    0,   88,  1,    5,     1,    0,    244,    14,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port14
   1048,  1,     10,   2,   88,   34,  1,    5,     0,    0,    244,    15,    0,     41,     2,      2,     1,     41,      2,   0,   1;  % Middle UEG1 port0
   1049,  1,     10,   2,   88,   34,  1,    5,     0,    0,    244,    16,    0,     41,     2,      2,     1,     41,      2,   2,   1;  % Middle UEG1 port2
   1050,  1,     10,   2,   88,   34,  1,    5,     0,    0,    244,    17,    0,     41,     2,      2,     1,     41,      2,   4,   1;  % Middle UEG1 port4
   1051,  1,     10,   2,   88,   34,  1,    5,     0,    0,    244,    18,    0,     41,     2,      2,     1,     41,      2,   6,   1;  % Middle UEG1 port6
   1052,  1,     10,   2,  122,   34,  1,    5,     0,    0,    244,    19,    0,     41,     2,      2,     1,     41,      2,   0,   2;  % Middle UEG2 port0
   1053,  1,     10,   2,  122,   34,  1,    5,     0,    0,    244,    20,    0,     41,     2,      2,     1,     41,      2,   2,   2;  % Middle UEG2 port2
   1054,  1,     10,   2,  122,   34,  1,    5,     0,    0,    244,    21,    0,     41,     2,      2,     1,     41,      2,   4,   2;  % Middle UEG2 port4
   1055,  1,     10,   2,  122,   34,  1,    5,     0,    0,    244,    22,    0,     41,     2,      2,     1,     41,      2,   6,   2;  % Middle UEG2 port6
   1056,  1,      4,   2,  156,   22,  1,    5,     0,    0,    244,    23,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
   1057,  1,      4,   2,  178,   22,  1,    5,     0,    0,    244,    24,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
   1058,  1,      4,   2,  200,   22,  1,    5,     0,    0,    244,    25,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
   1059,  1,      4,   2,  222,   22,  1,    5,     0,    0,    244,    26,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH set 3: PRB 0~243, OFDM symbol 1~13
    % reuse cfg 1008~1019 from PDSCH set 0: PRB 20~243, OFDM symbol 1~13
   1060,  1,     19,   2,    0,   88,  1,   13,     0,    0,    244,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
   1061,  1,     19,   2,    0,   88,  1,   13,     0,    0,    244,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
   1062,  1,     19,   2,    0,   88,  1,   13,     0,    0,    244,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
   1063,  1,     19,   2,    0,   88,  1,   13,     0,    0,    244,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
   1064,  1,     19,   2,    0,   88,  1,   13,     1,    0,    244,    11,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port8
   1065,  1,     19,   2,    0,   88,  1,   13,     1,    0,    244,    12,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port10
   1066,  1,     19,   2,    0,   88,  1,   13,     1,    0,    244,    13,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port12
   1067,  1,     19,   2,    0,   88,  1,   13,     1,    0,    244,    14,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port14
    % PDSCH set 4: PRB 0~235, OFDM symbol 1~13
    % reuse cfg 1008~1018 from PDSCH set 0: PRB 20~243, OFDM symbol 1~13, cfg 1060~1067 from PDSCH set 3: PRB 0~243, OFDM symbol 1~13
   1068,  1,      4,   2,  222,   14,  1,   13,     0,    0,    244,    26,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH for CCH
   1069,  1,     27,   1,  236,    8,  1,   13,     0,    0,    244, 65535,    0,     41,     2,      2,     1,     41,      2,   0,   7;  % CCH config
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
   % PDSCH set 0: PRB 20~243, OFDM symbol 1~13
   1070,  1,     19,   2,   20,   68,  1,   13,     0,    0,    244,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
   1071,  1,     19,   2,   20,   68,  1,   13,     0,    0,    244,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
   1072,  1,     19,   2,   20,   68,  1,   13,     0,    0,    244,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
   1073,  1,     19,   2,   20,   68,  1,   13,     0,    0,    244,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
   1074,  1,     10,   2,   88,   34,  1,   13,     0,    0,    244,    11,    0,     41,     2,      2,     1,     41,      2,   0,   1;  % Middle UEG1 port0
   1075,  1,     10,   2,   88,   34,  1,   13,     0,    0,    244,    12,    0,     41,     2,      2,     1,     41,      2,   2,   1;  % Middle UEG1 port2
   1076,  1,     10,   2,  122,   34,  1,   13,     0,    0,    244,    13,    0,     41,     2,      2,     1,     41,      2,   0,   2;  % Middle UEG2 port0
   1077,  1,     10,   2,  122,   34,  1,   13,     0,    0,    244,    14,    0,     41,     2,      2,     1,     41,      2,   2,   2;  % Middle UEG2 port2
   1078,  1,      4,   1,  156,   22,  1,   13,     0,    0,    244,    15,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
   1079,  1,      4,   1,  178,   22,  1,   13,     0,    0,    244,    16,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
   1080,  1,      4,   1,  200,   22,  1,   13,     0,    0,    244,    17,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
   1081,  1,      4,   1,  222,   22,  1,   13,     0,    0,    244,    18,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH set 1: PRB 0~243, OFDM symbol 2~13
   1082,  1,     19,   2,    0,   88,  2,   12,     0,    0,    244,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
   1083,  1,     19,   2,    0,   88,  2,   12,     0,    0,    244,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
   1084,  1,     19,   2,    0,   88,  2,   12,     0,    0,    244,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
   1085,  1,     19,   2,    0,   88,  2,   12,     0,    0,    244,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
   1086,  1,     10,   2,   88,   34,  2,   12,     0,    0,    244,    11,    0,     41,     2,      2,     1,     41,      2,   0,   1;  % Middle UEG1 port0
   1087,  1,     10,   2,   88,   34,  2,   12,     0,    0,    244,    12,    0,     41,     2,      2,     1,     41,      2,   2,   1;  % Middle UEG1 port2
   1088,  1,     10,   2,  122,   34,  2,   12,     0,    0,    244,    13,    0,     41,     2,      2,     1,     41,      2,   0,   2;  % Middle UEG2 port0
   1089,  1,     10,   2,  122,   34,  2,   12,     0,    0,    244,    14,    0,     41,     2,      2,     1,     41,      2,   2,   2;  % Middle UEG2 port2
   1090,  1,      4,   1,  156,   22,  2,   12,     0,    0,    244,    15,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
   1091,  1,      4,   1,  178,   22,  2,   12,     0,    0,    244,    16,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
   1092,  1,      4,   1,  200,   22,  2,   12,     0,    0,    244,    17,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
   1093,  1,      4,   1,  222,   22,  2,   12,     0,    0,    244,    18,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH set 2: PRB 0~243, OFDM symbol 1~5
   1094,  1,     19,   2,    0,   88,  1,    5,     0,    0,    244,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
   1095,  1,     19,   2,    0,   88,  1,    5,     0,    0,    244,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
   1096,  1,     19,   2,    0,   88,  1,    5,     0,    0,    244,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
   1097,  1,     19,   2,    0,   88,  1,    5,     0,    0,    244,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
   1098,  1,     10,   2,   88,   34,  1,    5,     0,    0,    244,    11,    0,     41,     2,      2,     1,     41,      2,   0,   1;  % Middle UEG1 port0
   1099,  1,     10,   2,   88,   34,  1,    5,     0,    0,    244,    12,    0,     41,     2,      2,     1,     41,      2,   2,   1;  % Middle UEG1 port2
   1100,  1,     10,   2,  122,   34,  1,    5,     0,    0,    244,    13,    0,     41,     2,      2,     1,     41,      2,   0,   2;  % Middle UEG2 port0
   1101,  1,     10,   2,  122,   34,  1,    5,     0,    0,    244,    14,    0,     41,     2,      2,     1,     41,      2,   2,   2;  % Middle UEG2 port2
   1102,  1,      4,   1,  156,   22,  1,    5,     0,    0,    244,    15,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
   1103,  1,      4,   1,  178,   22,  1,    5,     0,    0,    244,    16,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
   1104,  1,      4,   1,  200,   22,  1,    5,     0,    0,    244,    17,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
   1105,  1,      4,   1,  222,   22,  1,    5,     0,    0,    244,    18,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH set 3: PRB 0~243, OFDM symbol 1~13
    % reuse cfg 1074~1081 from PDSCH set 0: PRB 20~243, OFDM symbol 1~13
   1106,  1,     19,   2,    0,   88,  1,   13,     0,    0,    244,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
   1107,  1,     19,   2,    0,   88,  1,   13,     0,    0,    244,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
   1108,  1,     19,   2,    0,   88,  1,   13,     0,    0,    244,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
   1109,  1,     19,   2,    0,   88,  1,   13,     0,    0,    244,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
    % PDSCH set 4: PRB 0~235, OFDM symbol 1~13
    % reuse cfg 1074~1080 from PDSCH set 0: PRB 20~243, OFDM symbol 1~13, cfg 1106~1109 from PDSCH set 3: PRB 0~243, OFDM symbol 1~13
   1110,  1,      4,   1,  222,   14,  1,   13,     0,    0,    244,    18,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH for CCH
   1111,  1,     27,   1,  236,    8,  1,   13,     0,    0,    244, 65535,    0,     41,     2,      2,     1,     41,      2,   0,   7;  % CCH config
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
   % PDSCH set 0: PRB 20~159, OFDM symbol 1~13
   1112,  1,     19,   2,   20,   36,  1,   13,     0,    0,    160,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
   1113,  1,     19,   2,   20,   36,  1,   13,     0,    0,    160,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
   1114,  1,     19,   2,   20,   36,  1,   13,     0,    0,    160,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
   1115,  1,     19,   2,   20,   36,  1,   13,     0,    0,    160,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
   1116,  1,     19,   2,   20,   36,  1,   13,     1,    0,    160,    11,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port8
   1117,  1,     19,   2,   20,   36,  1,   13,     1,    0,    160,    12,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port10
   1118,  1,     19,   2,   20,   36,  1,   13,     1,    0,    160,    13,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port12
   1119,  1,     19,   2,   20,   36,  1,   13,     1,    0,    160,    14,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port14
   1120,  1,     10,   2,   56,   24,  1,   13,     0,    0,    160,    15,    0,     41,     2,      2,     1,     41,      2,   0,   1;  % Middle UEG1 port0
   1121,  1,     10,   2,   56,   24,  1,   13,     0,    0,    160,    16,    0,     41,     2,      2,     1,     41,      2,   2,   1;  % Middle UEG1 port2
   1122,  1,     10,   2,   56,   24,  1,   13,     0,    0,    160,    17,    0,     41,     2,      2,     1,     41,      2,   4,   1;  % Middle UEG1 port4
   1123,  1,     10,   2,   56,   24,  1,   13,     0,    0,    160,    18,    0,     41,     2,      2,     1,     41,      2,   6,   1;  % Middle UEG1 port6
   1124,  1,     10,   2,   80,   24,  1,   13,     0,    0,    160,    19,    0,     41,     2,      2,     1,     41,      2,   0,   2;  % Middle UEG2 port0
   1125,  1,     10,   2,   80,   24,  1,   13,     0,    0,    160,    20,    0,     41,     2,      2,     1,     41,      2,   2,   2;  % Middle UEG2 port2
   1126,  1,     10,   2,   80,   24,  1,   13,     0,    0,    160,    21,    0,     41,     2,      2,     1,     41,      2,   4,   2;  % Middle UEG2 port4
   1127,  1,     10,   2,   80,   24,  1,   13,     0,    0,    160,    22,    0,     41,     2,      2,     1,     41,      2,   6,   2;  % Middle UEG2 port6
   1128,  1,      4,   2,  104,   14,  1,   13,     0,    0,    160,    23,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
   1129,  1,      4,   2,  118,   14,  1,   13,     0,    0,    160,    24,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
   1130,  1,      4,   2,  132,   14,  1,   13,     0,    0,    160,    25,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
   1131,  1,      4,   2,  146,   14,  1,   13,     0,    0,    160,    26,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH set 1: PRB 0~159, OFDM symbol 2~13
   1132,  1,     19,   2,    0,   56,  2,   12,     0,    0,    160,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
   1133,  1,     19,   2,    0,   56,  2,   12,     0,    0,    160,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
   1134,  1,     19,   2,    0,   56,  2,   12,     0,    0,    160,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
   1135,  1,     19,   2,    0,   56,  2,   12,     0,    0,    160,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
   1136,  1,     19,   2,    0,   56,  2,   12,     1,    0,    160,    11,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port8
   1137,  1,     19,   2,    0,   56,  2,   12,     1,    0,    160,    12,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port10
   1138,  1,     19,   2,    0,   56,  2,   12,     1,    0,    160,    13,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port12
   1139,  1,     19,   2,    0,   56,  2,   12,     1,    0,    160,    14,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port14
   1140,  1,     10,   2,   56,   24,  2,   12,     0,    0,    160,    15,    0,     41,     2,      2,     1,     41,      2,   0,   1;  % Middle UEG1 port0
   1141,  1,     10,   2,   56,   24,  2,   12,     0,    0,    160,    16,    0,     41,     2,      2,     1,     41,      2,   2,   1;  % Middle UEG1 port2
   1142,  1,     10,   2,   56,   24,  2,   12,     0,    0,    160,    17,    0,     41,     2,      2,     1,     41,      2,   4,   1;  % Middle UEG1 port4
   1143,  1,     10,   2,   56,   24,  2,   12,     0,    0,    160,    18,    0,     41,     2,      2,     1,     41,      2,   6,   1;  % Middle UEG1 port6
   1144,  1,     10,   2,   80,   24,  2,   12,     0,    0,    160,    19,    0,     41,     2,      2,     1,     41,      2,   0,   2;  % Middle UEG2 port0
   1145,  1,     10,   2,   80,   24,  2,   12,     0,    0,    160,    20,    0,     41,     2,      2,     1,     41,      2,   2,   2;  % Middle UEG2 port2
   1146,  1,     10,   2,   80,   24,  2,   12,     0,    0,    160,    21,    0,     41,     2,      2,     1,     41,      2,   4,   2;  % Middle UEG2 port4
   1147,  1,     10,   2,   80,   24,  2,   12,     0,    0,    160,    22,    0,     41,     2,      2,     1,     41,      2,   6,   2;  % Middle UEG2 port6
   1148,  1,      4,   2,  104,   14,  2,   12,     0,    0,    160,    23,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
   1149,  1,      4,   2,  118,   14,  2,   12,     0,    0,    160,    24,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
   1150,  1,      4,   2,  132,   14,  2,   12,     0,    0,    160,    25,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
   1151,  1,      4,   2,  146,   14,  2,   12,     0,    0,    160,    26,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH set 2: PRB 0~159, OFDM symbol 1~5
   1152,  1,     19,   2,    0,   56,  1,    5,     0,    0,    160,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
   1153,  1,     19,   2,    0,   56,  1,    5,     0,    0,    160,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
   1154,  1,     19,   2,    0,   56,  1,    5,     0,    0,    160,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
   1155,  1,     19,   2,    0,   56,  1,    5,     0,    0,    160,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
   1156,  1,     19,   2,    0,   56,  1,    5,     1,    0,    160,    11,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port8
   1157,  1,     19,   2,    0,   56,  1,    5,     1,    0,    160,    12,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port10
   1158,  1,     19,   2,    0,   56,  1,    5,     1,    0,    160,    13,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port12
   1159,  1,     19,   2,    0,   56,  1,    5,     1,    0,    160,    14,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port14
   1160,  1,     10,   2,   56,   24,  1,    5,     0,    0,    160,    15,    0,     41,     2,      2,     1,     41,      2,   0,   1;  % Middle UEG1 port0
   1161,  1,     10,   2,   56,   24,  1,    5,     0,    0,    160,    16,    0,     41,     2,      2,     1,     41,      2,   2,   1;  % Middle UEG1 port2
   1162,  1,     10,   2,   56,   24,  1,    5,     0,    0,    160,    17,    0,     41,     2,      2,     1,     41,      2,   4,   1;  % Middle UEG1 port4
   1163,  1,     10,   2,   56,   24,  1,    5,     0,    0,    160,    18,    0,     41,     2,      2,     1,     41,      2,   6,   1;  % Middle UEG1 port6
   1164,  1,     10,   2,   80,   24,  1,    5,     0,    0,    160,    19,    0,     41,     2,      2,     1,     41,      2,   0,   2;  % Middle UEG2 port0
   1165,  1,     10,   2,   80,   24,  1,    5,     0,    0,    160,    20,    0,     41,     2,      2,     1,     41,      2,   2,   2;  % Middle UEG2 port2
   1166,  1,     10,   2,   80,   24,  1,    5,     0,    0,    160,    21,    0,     41,     2,      2,     1,     41,      2,   4,   2;  % Middle UEG2 port4
   1167,  1,     10,   2,   80,   24,  1,    5,     0,    0,    160,    22,    0,     41,     2,      2,     1,     41,      2,   6,   2;  % Middle UEG2 port6
   1168,  1,      4,   2,  104,   14,  1,    5,     0,    0,    160,    23,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
   1169,  1,      4,   2,  118,   14,  1,    5,     0,    0,    160,    24,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
   1170,  1,      4,   2,  132,   14,  1,    5,     0,    0,    160,    25,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
   1171,  1,      4,   2,  146,   14,  1,    5,     0,    0,    160,    26,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH set 3: PRB 0~159, OFDM symbol 1~13
    % reuse cfg 1120~1131 from PDSCH set 0: PRB 20~159, OFDM symbol 1~13
   1172,  1,     19,   2,    0,   56,  1,   13,     0,    0,    160,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
   1173,  1,     19,   2,    0,   56,  1,   13,     0,    0,    160,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
   1174,  1,     19,   2,    0,   56,  1,   13,     0,    0,    160,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
   1175,  1,     19,   2,    0,   56,  1,   13,     0,    0,    160,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
   1176,  1,     19,   2,    0,   56,  1,   13,     1,    0,    160,    11,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port8
   1177,  1,     19,   2,    0,   56,  1,   13,     1,    0,    160,    12,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port10
   1178,  1,     19,   2,    0,   56,  1,   13,     1,    0,    160,    13,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port12
   1179,  1,     19,   2,    0,   56,  1,   13,     1,    0,    160,    14,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port14
    % PDSCH set 4: PRB 0~151, OFDM symbol 1~13
    % reuse cfg 1120~1130 from PDSCH set 0: PRB 20~159, OFDM symbol 1~13, cfg 1172~1179 from PDSCH set 3: PRB 0~159, OFDM symbol 1~13
   1180,  1,      4,   2,  146,    6,  1,   13,     0,    0,    160,    26,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH for CCH
   1181,  1,     27,   1,  152,    8,  1,   13,     0,    0,    160, 65535,    0,     41,     2,      2,     1,     41,      2,   0,   7;  % CCH config
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
   % PDSCH set 0: PRB 20~159, OFDM symbol 1~13
   1182,  1,     19,   2,   20,   36,  1,   13,     0,    0,    160,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
   1183,  1,     19,   2,   20,   36,  1,   13,     0,    0,    160,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
   1184,  1,     19,   2,   20,   36,  1,   13,     0,    0,    160,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
   1185,  1,     19,   2,   20,   36,  1,   13,     0,    0,    160,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
   1186,  1,     10,   2,   56,   24,  1,   13,     0,    0,    160,    11,    0,     41,     2,      2,     1,     41,      2,   0,   1;  % Middle UEG1 port0
   1187,  1,     10,   2,   56,   24,  1,   13,     0,    0,    160,    12,    0,     41,     2,      2,     1,     41,      2,   2,   1;  % Middle UEG1 port2
   1188,  1,     10,   2,   80,   24,  1,   13,     0,    0,    160,    13,    0,     41,     2,      2,     1,     41,      2,   0,   2;  % Middle UEG2 port0
   1189,  1,     10,   2,   80,   24,  1,   13,     0,    0,    160,    14,    0,     41,     2,      2,     1,     41,      2,   2,   2;  % Middle UEG2 port2
   1190,  1,      4,   1,  104,   14,  1,   13,     0,    0,    160,    15,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
   1191,  1,      4,   1,  118,   14,  1,   13,     0,    0,    160,    16,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
   1192,  1,      4,   1,  132,   14,  1,   13,     0,    0,    160,    17,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
   1193,  1,      4,   1,  146,   14,  1,   13,     0,    0,    160,    18,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH set 1: PRB 0~159, OFDM symbol 2~13
   1194,  1,     19,   2,    0,   56,  2,   12,     0,    0,    160,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
   1195,  1,     19,   2,    0,   56,  2,   12,     0,    0,    160,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
   1196,  1,     19,   2,    0,   56,  2,   12,     0,    0,    160,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
   1197,  1,     19,   2,    0,   56,  2,   12,     0,    0,    160,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
   1198,  1,     10,   2,   56,   24,  2,   12,     0,    0,    160,    11,    0,     41,     2,      2,     1,     41,      2,   0,   1;  % Middle UEG1 port0
   1199,  1,     10,   2,   56,   24,  2,   12,     0,    0,    160,    12,    0,     41,     2,      2,     1,     41,      2,   2,   1;  % Middle UEG1 port2
   1200,  1,     10,   2,   80,   24,  2,   12,     0,    0,    160,    13,    0,     41,     2,      2,     1,     41,      2,   0,   2;  % Middle UEG2 port0
   1201,  1,     10,   2,   80,   24,  2,   12,     0,    0,    160,    14,    0,     41,     2,      2,     1,     41,      2,   2,   2;  % Middle UEG2 port2
   1202,  1,      4,   1,  104,   14,  2,   12,     0,    0,    160,    15,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
   1203,  1,      4,   1,  118,   14,  2,   12,     0,    0,    160,    16,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
   1204,  1,      4,   1,  132,   14,  2,   12,     0,    0,    160,    17,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
   1205,  1,      4,   1,  146,   14,  2,   12,     0,    0,    160,    18,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH set 2: PRB 0~159, OFDM symbol 1~5
   1206,  1,     19,   2,    0,   56,  1,    5,     0,    0,    160,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
   1207,  1,     19,   2,    0,   56,  1,    5,     0,    0,    160,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
   1208,  1,     19,   2,    0,   56,  1,    5,     0,    0,    160,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
   1209,  1,     19,   2,    0,   56,  1,    5,     0,    0,    160,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
   1210,  1,     10,   2,   56,   24,  1,    5,     0,    0,    160,    11,    0,     41,     2,      2,     1,     41,      2,   0,   1;  % Middle UEG1 port0
   1211,  1,     10,   2,   56,   24,  1,    5,     0,    0,    160,    12,    0,     41,     2,      2,     1,     41,      2,   2,   1;  % Middle UEG1 port2
   1212,  1,     10,   2,   80,   24,  1,    5,     0,    0,    160,    13,    0,     41,     2,      2,     1,     41,      2,   0,   2;  % Middle UEG2 port0
   1213,  1,     10,   2,   80,   24,  1,    5,     0,    0,    160,    14,    0,     41,     2,      2,     1,     41,      2,   2,   2;  % Middle UEG2 port2
   1214,  1,      4,   1,  104,   14,  1,    5,     0,    0,    160,    15,    0,     41,     2,      1,     1,     41,      2,   0,   3;  % Edge UEG3 port0
   1215,  1,      4,   1,  118,   14,  1,    5,     0,    0,    160,    16,    0,     41,     2,      1,     1,     41,      2,   0,   4;  % Edge UEG4 port0
   1216,  1,      4,   1,  132,   14,  1,    5,     0,    0,    160,    17,    0,     41,     2,      1,     1,     41,      2,   0,   5;  % Edge UEG5 port0
   1217,  1,      4,   1,  146,   14,  1,    5,     0,    0,    160,    18,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH set 3: PRB 0~159, OFDM symbol 1~13
    % reuse cfg 1186~1193 from PDSCH set 0: PRB 20~159, OFDM symbol 1~13
   1218,  1,     19,   2,    0,   56,  1,   13,     0,    0,    160,     7,    0,     41,     2,      2,     1,     41,      2,   0,   0;  % Center UEG0 port0
   1219,  1,     19,   2,    0,   56,  1,   13,     0,    0,    160,     8,    0,     41,     2,      2,     1,     41,      2,   2,   0;  % Center UEG0 port2
   1220,  1,     19,   2,    0,   56,  1,   13,     0,    0,    160,     9,    0,     41,     2,      2,     1,     41,      2,   4,   0;  % Center UEG0 port4
   1221,  1,     19,   2,    0,   56,  1,   13,     0,    0,    160,    10,    0,     41,     2,      2,     1,     41,      2,   6,   0;  % Center UEG0 port6
    % PDSCH set 4: PRB 0~151, OFDM symbol 1~13
    % reuse cfg 1186~1192 from PDSCH set 0: PRB 20~159, OFDM symbol 1~13, cfg 1218~1221 from PDSCH set 3: PRB 0~159, OFDM symbol 1~13
   1222,  1,      4,   1,  146,    6,  1,   13,     0,    0,    160,    18,    0,     41,     2,      1,     1,     41,      2,   0,   6;  % Edge UEG6 port0
    % PDSCH for CCH
   1223,  1,     27,   1,  152,    8,  1,   13,     0,    0,    160, 65535,    0,     41,     2,      2,     1,     41,      2,   0,   7;  % CCH config
% cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 idxUeg
   % 4TR PDSCH: PRB 0~272, OFDM symbol 0~13
   1224,  1,     27,   4,    0,  273,  0,   14,     0,    0,    273,     7,    0,     41,     2,      1,     0,     41,      2,   0,   0;  % 1 UE SU-MIMO, all D slot
   % 4TR PDSCH: PRB 0~53, OFDM symbol 0~13
   1225,  1,     27,   4,    0,   54,  0,   14,     0,    0,    273,     7,    0,     41,     2,      1,     0,     41,      2,   0,   0;  % 1 UE SU-MIMO, all D slot
};

% append generated PXSCH config
% Base configuration struct for PDSCH
baseCfgUeg = struct();
baseCfgUeg.startRnti = 7;
baseCfgUeg.mcsIdx = 27;
baseCfgUeg.BWP0 = 0;
baseCfgUeg.nBWP = 273;
baseCfgUeg.prgSize = 2;
baseCfgUeg.nUeg = 8;
baseCfgUeg.nUePerUeg = 8;
baseCfgUeg.diffscId = true;
baseCfgUeg.nl = 2;
baseCfgUeg.dmrsMaxLen = 2;

% 64TR MU-MIMO, 8 UEGs, 8 UEs per UEG, 2 layers per UE, PDSCH prb 20~272, OFDM symbol 1~13, 100 MHz, CFG 2001~2064
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = 2001;  % first config index for generated PXSCH config
cfgUeg.startPrb = 20;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 13;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 8 UEGs, 8 UEs per UEG, 2 layers per UE, PDSCH prb 20~272, OFDM symbol 2~13, 100 MHz, CFG 2065~2128
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.startPrb = 20;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 2;
cfgUeg.Nsym = 12;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 8 UEGs, 8 UEs per UEG, 2 layers per UE, PDSCH prb 0~272, OFDM symbol 1~5, 100 MHz, CFG 2129~2192
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 5;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 8 UEGs, 8 UEs per UEG, 2 layers per UE, PDSCH prb 0~272, OFDM symbol 1~13, 100 MHz, CFG 2193~2256
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 13;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 8 UEGs, 8 UEs per UEG, 2 layers per UE, PDSCH prb 0~272, OFDM symbol 2~13, 100 MHz, CFG 2257~2320
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 2;
cfgUeg.Nsym = 12;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 1 UEG, 16 UEs per UEG, 1 layer per UE, PDSCH prb 20~272, OFDM symbol 1~13, 100 MHz, CFG 2321~2336
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 16;
cfgUeg.startPrb = 20;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 13;
cfgUeg.nl = 1;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 1 UEG, 16 UEs per UEG, 1 layer per UE, PDSCH prb 0~272, OFDM symbol 2~13, 100 MHz, CFG 2337~2352
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 16;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 2;
cfgUeg.Nsym = 12;
cfgUeg.nl = 1;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 1 UEG, 16 UEs per UEG, 1 layer per UE, PDSCH prb 0~272, OFDM symbol 1~5, 100 MHz, CFG 2353~2368
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 16;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 5;
cfgUeg.nl = 1;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 1 UEG, 16 UEs per UEG, 1 layer per UE, PDSCH prb 0~272, OFDM symbol 1~13, 100 MHz, CFG 2369~2384
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 16;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 13;
cfgUeg.nl = 1;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 1 UEG, 16 UEs per UEG, 1 layer per UE, PDSCH prb 0~264, OFDM symbol 1~13, 100 MHz, CFG 2385~2400
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 16;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 264;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 13;
cfgUeg.nl = 1;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 4 UEG, 16 UEs per UEG, 1 layer per UE, PDSCH prb 20~272, OFDM symbol 1~13, 100 MHz, CFG 2401:2464
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 4;
cfgUeg.nUePerUeg = 16;
cfgUeg.startPrb = 20;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 13;
cfgUeg.nl = 1;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 4 UEG, 16 UEs per UEG, 1 layer per UE, PDSCH prb 0~272, OFDM symbol 2~13, 100 MHz, CFG 2465:2528
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 4;
cfgUeg.nUePerUeg = 16;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 2;
cfgUeg.Nsym = 12;
cfgUeg.nl = 1;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 4 UEG, 16 UEs per UEG, 1 layer per UE, PDSCH prb 0~272, OFDM symbol 1~5, 100 MHz, CFG 2529:2592
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 4;
cfgUeg.nUePerUeg = 16;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 5;
cfgUeg.nl = 1;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 4 UEG, 16 UEs per UEG, 1 layer per UE, PDSCH prb 0~272, OFDM symbol 1~13, 100 MHz, CFG 2593:2656
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 4;
cfgUeg.nUePerUeg = 16;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 13;
cfgUeg.nl = 1;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 4 UEG, 16 UEs per UEG, 1 layer per UE, PDSCH prb 0~264, OFDM symbol 1~13, 100 MHz, CFG 2657:2720
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 4;
cfgUeg.nUePerUeg = 16;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 264;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 13;
cfgUeg.nl = 1;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 1 UEG, 32 UEs per UEG, 1 layer per UE, PDSCH prb 20~272, OFDM symbol 1~13, 100 MHz, CFG 2721:2752
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 32;
cfgUeg.startPrb = 20;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 13;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 1 UEG, 32 UEs per UEG, 1 layer per UE, PDSCH prb 0~272, OFDM symbol 2~13, 100 MHz, CFG 2753:2784
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 32;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 2;
cfgUeg.Nsym = 12;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 1 UEG, 32 UEs per UEG, 1 layer per UE, PDSCH prb 0~272, OFDM symbol 1~5, 100 MHz, CFG 2785:2816
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 32;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 5;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 1 UEG, 32 UEs per UEG, 1 layer per UE, PDSCH prb 0~272, OFDM symbol 1~13, 100 MHz, CFG 2817:2848
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 32;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 13;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 1 UEG, 32 UEs per UEG, 1 layer per UE, PDSCH prb 0~264, OFDM symbol 1~13, 100 MHz, CFG 2849:2880
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 32;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 264;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 13;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 1 UEG, 16 UEs per UEG, 1 layer per UE, PDSCH prb 20~272, OFDM symbol 1~13, 100 MHz, CFG 2881~2896
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 16;
cfgUeg.startPrb = 20;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 13;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 1 UEG, 16 UEs per UEG, 1 layer per UE, PDSCH prb 0~272, OFDM symbol 2~13, 100 MHz, CFG 2897~2912
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 16;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 2;
cfgUeg.Nsym = 12;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 1 UEG, 16 UEs per UEG, 1 layer per UE, PDSCH prb 0~272, OFDM symbol 1~5, 100 MHz, CFG 2913~2928
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 16;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 5;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 1 UEG, 16 UEs per UEG, 1 layer per UE, PDSCH prb 0~272, OFDM symbol 1~13, 100 MHz, CFG 2929~2944
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 16;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 13;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 1 UEG, 16 UEs per UEG, 1 layer per UE, PDSCH prb 0~264, OFDM symbol 1~13, 100 MHz, CFG 2945~2960
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 16;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 264;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 13;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 1 UEG, 24 UEs per UEG, 1 layer per UE, PDSCH prb 20~272, OFDM symbol 1~13, 100 MHz, CFG 2961:2984
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 24;
cfgUeg.startPrb = 20;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 13;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 1 UEG, 24 UEs per UEG, 1 layer per UE, PDSCH prb 0~272, OFDM symbol 2~13, 100 MHz, CFG 2985:3008
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 24;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 2;
cfgUeg.Nsym = 12;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 1 UEG, 24 UEs per UEG, 1 layer per UE, PDSCH prb 0~272, OFDM symbol 1~5, 100 MHz, CFG 3009:3032
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 24;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 5;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 1 UEG, 24 UEs per UEG, 1 layer per UE, PDSCH prb 0~272, OFDM symbol 1~13, 100 MHz, CFG 3033:3056
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 24;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 272;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 13;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 1 UEG, 24 UEs per UEG, 1 layer per UE, PDSCH prb 0~264, OFDM symbol 1~13, 100 MHz, CFG 3057:3080
cfgUeg = baseCfgUeg;
cfgUeg.pxsch_cfg_idx = CFG_PDSCH{end, 1} + 1;
cfgUeg.nUeg = 1;
cfgUeg.nUePerUeg = 24;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 264;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 13;
cfgUeg.nl = 1;
cfgUeg.prgSize = 16;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PDSCH = [CFG_PDSCH; CFG_PDSCH_temp];

% Check if first column (pxsch_cfg_idx) values are unique
first_col = cell2mat(CFG_PDSCH(:,1));
if length(unique(first_col)) ~= length(first_col)
    [unique_vals, ~, idx] = unique(first_col);
    counts = accumarray(idx, 1);
    duplicate_vals = unique_vals(counts > 1);
    error('genPxschUegCfg:DuplicateIdx', ...
          'First column (pxsch_cfg_idx) contains duplicate values: %s\n', mat2str(duplicate_vals));
end

CFG_CSIRS = {...
 % cfg#  CSI Row CDM Density RB0  nRB  sym0  sym1   nID  FreqDomain
    1,   1,  4,  1,   2,     0,  273,  13,    8,   41,   {[0 1 0]};
    2,   1,  5,  1,   2,     0,  273,   4,    8,   41,   {[0 1 0]};

    % perf test
    3,   1,  4,  1,   2,     22,  250,   5,    8,   41,   {ones(1,12)}; % F08
    4,   1,  4,  1,   2,     0,   273,   5,    8,   41,   {ones(1,12)};
    5,   1,  11, 1,   2,     22,  250,   5,    8,   41,   {ones(1,12)}; % F09
    6,   1,  11, 1,   2,     0,   273,   5,    8,   41,   {ones(1,12)};
    % F08 ZP CSI-RS
    7,   2,  4,  1,   2,     22,  250,   5,    8,   41,   {ones(1,12)}; % F08
    % F08 at symbol 6 
    8,   1,  4,  1,   2,     22,  250,   6,    8,   41,   {ones(1,12)};
    % F08 ZP CSI-RS, at symbol 6 
    9,   2,  4,  1,   2,     22,  250,   6,    8,   41,   {ones(1,12)};
    % TRS on symbol 6
    10,  0,  1,  0,   3,     0,   273,   6,    8,   41,   {ones(1,12)};
    % TRS on symbol 10
    11,  0,  1,  0,   3,     0,   273,   10,   8,   41,   {ones(1,12)};
    % F08 at symbol 12
    12,  1,  4,  1,   2,     0,   273,   12,   8,   41,   {ones(1,12)};
    % F08 at symbol 13 
    13,  1,  4,  1,   2,     0,   273,   13,   8,   41,   {ones(1,12)};
    % TRS on symbol 5
    14,  0,  1,  0,   3,     0,   273,   5,    8,   41,   {ones(1,12)};
    % TRS on symbol 9
    15,  0,  1,  0,   3,     0,   273,   9,    8,   41,   {ones(1,12)};
    %%%%%    
    % TRS on symbol 6, 52 PRB
    16,  0,  1,  0,   3,     0,    52,   6,    8,   41,   {ones(1,12)};
    % TRS on symbol 10, 52 PRB
    17,  0,  1,  0,   3,     0,    52,   10,   8,   41,   {ones(1,12)};
    % TRS on symbol 5, 52 PRB
    18,  0,  1,  0,   3,     0,    52,   5,    8,   41,   {ones(1,12)};
    % TRS on symbol 9, 52 PRB
    19,  0,  1,  0,   3,     0,    52,   9,    8,   41,   {ones(1,12)};
    % F08 at symbol 12, 132 PRB
    20,  1,  4,  1,   2,     0,   132,   12,   8,   41,   {ones(1,12)};
    % F08 at symbol 13, 132 PRB 
    21,  1,  4,  1,   2,     0,   132,   13,   8,   41,   {ones(1,12)};
    %%%%%    
    % TRS on symbol 6, 52 PRB
    22,  0,  1,  0,   3,    78,    52,   6,    8,   41,   {ones(1,12)};
    % TRS on symbol 10, 52 PRB
    23,  0,  1,  0,   3,    78,    52,   10,   8,   41,   {ones(1,12)};
    % TRS on symbol 5, 52 PRB
    24,  0,  1,  0,   3,    78,    52,   5,    8,   41,   {ones(1,12)};
    % TRS on symbol 9, 52 PRB
    25,  0,  1,  0,   3,    78,    52,   9,    8,   41,   {ones(1,12)};
    %%%%%
    % NZP CSIRS at symbol 13, 78-129 PRB 
    26,  1,  4,  1,   2,    78,    52,   13,   8,   41,   {ones(1,12)};
    %%%%% 40MHz full 
    % TRS on symbol 6
    27,  0,  1,  0,   3,     0,   106,   6,    8,   41,   {ones(1,12)};
    % TRS on symbol 10
    28,  0,  1,  0,   3,     0,   106,   10,   8,   41,   {ones(1,12)};
    % TRS on symbol 5
    29,  0,  1,  0,   3,     0,   106,   5,    8,   41,   {ones(1,12)};
    % TRS on symbol 9
    30,  0,  1,  0,   3,     0,   106,   9,    8,   41,   {ones(1,12)};
    % F08 at symbol 12
    31,  1,  4,  1,   2,     0,   106,   12,   8,   41,   {ones(1,12)};
    % F08 at symbol 13 
    32,  1,  4,  1,   2,     0,   106,   13,   8,   41,   {ones(1,12)};
    %%%%%    
    % TRS on symbol 6, 52 PRB, start from 48
    33,  0,  1,  0,   3,    48,    52,   6,    8,   41,   {ones(1,12)};
    % TRS on symbol 10, 52 PRB, start from 48
    34,  0,  1,  0,   3,    48,    52,   10,   8,   41,   {ones(1,12)};
    % TRS on symbol 5, 52 PRB, start from 48
    35,  0,  1,  0,   3,    48,    52,   5,    8,   41,   {ones(1,12)};
    % TRS on symbol 9, 52 PRB, start from 48
    36,  0,  1,  0,   3,    48,    52,   9,    8,   41,   {ones(1,12)};
    %%%%%
  % cfg#  CSI Row CDM Density RB0  nRB  sym0  sym1   nID  FreqDomain 
  %%%%% 80MHz full 
    % TRS on symbol 6
    37,  0,  1,  0,   3,     0,   217,   6,    8,   41,   {ones(1,12)};
    % TRS on symbol 10
    38,  0,  1,  0,   3,     0,   217,   10,   8,   41,   {ones(1,12)};
    % TRS on symbol 5
    39,  0,  1,  0,   3,     0,   217,   5,    8,   41,   {ones(1,12)};
    % TRS on symbol 9
    40,  0,  1,  0,   3,     0,   217,   9,    8,   41,   {ones(1,12)};
    % F08 at symbol 12
    41,  1,  4,  1,   2,     0,   217,   12,   8,   41,   {ones(1,12)};
    % F08 at symbol 13 
    42,  1,  4,  1,   2,     0,   217,   13,   8,   41,   {ones(1,12)};
    %%%%%    
    % TRS on symbol 6, 52 PRB, start from 54
    43,  0,  1,  0,   3,    54,    52,   6,    8,   41,   {ones(1,12)};
    % TRS on symbol 10, 52 PRB, start from 54
    44,  0,  1,  0,   3,    54,    52,   10,   8,   41,   {ones(1,12)};
    % TRS on symbol 5, 52 PRB, start from 54
    45,  0,  1,  0,   3,    54,    52,   5,    8,   41,   {ones(1,12)};
    % TRS on symbol 9, 52 PRB, start from 54
    46,  0,  1,  0,   3,    54,    52,   9,    8,   41,   {ones(1,12)};
    %%%%% 30 MHz full 
    % TRS on symbol 6
    47,  0,  1,  0,   3,     0,   78,   6,    8,   41,   {ones(1,12)};
    % TRS on symbol 10
    48,  0,  1,  0,   3,     0,   78,   10,   8,   41,   {ones(1,12)};
    % TRS on symbol 5
    49,  0,  1,  0,   3,     0,   78,   5,    8,   41,   {ones(1,12)};
    % TRS on symbol 9
    50,  0,  1,  0,   3,     0,   78,   9,    8,   41,   {ones(1,12)};
    % F08 at symbol 12
    51,  1,  4,  1,   2,     0,   78,   12,   8,   41,   {ones(1,12)};
    % F08 at symbol 13 
    52,  1,  4,  1,   2,     0,   78,   13,   8,   41,   {ones(1,12)};
    %%%%%    
    % TRS on symbol 6, 52 PRB
    53,  0,  1,  0,   3,    24,    52,   6,    8,   41,   {ones(1,12)};
    % TRS on symbol 10, 52 PRB
    54,  0,  1,  0,   3,    24,    52,   10,   8,   41,   {ones(1,12)};
    % TRS on symbol 5, 52 PRB
    55,  0,  1,  0,   3,    24,    52,   5,    8,   41,   {ones(1,12)};
    % TRS on symbol 9, 52 PRB
    56,  0,  1,  0,   3,    24,    52,   9,    8,   41,   {ones(1,12)};
    %%%%% 50 MHz full 
    % TRS on symbol 6
    57,  0,  1,  0,   3,     0,   133,   6,    8,   41,   {ones(1,12)};
    % TRS on symbol 10
    58,  0,  1,  0,   3,     0,   133,   10,   8,   41,   {ones(1,12)};
    % TRS on symbol 5
    59,  0,  1,  0,   3,     0,   133,   5,    8,   41,   {ones(1,12)};
    % TRS on symbol 9
    60,  0,  1,  0,   3,     0,   133,   9,    8,   41,   {ones(1,12)};
    % F08 at symbol 12
    61,  1,  4,  1,   2,     0,   133,   12,   8,   41,   {ones(1,12)};
    % F08 at symbol 13 
    62,  1,  4,  1,   2,     0,   133,   13,   8,   41,   {ones(1,12)};
    % TRS on symbol 6, 52 PRB
    63,  0,  1,  0,   3,    66,    52,   6,    8,   41,   {ones(1,12)};
    % TRS on symbol 10, 52 PRB
    64,  0,  1,  0,   3,    66,    52,   10,   8,   41,   {ones(1,12)};
    % TRS on symbol 5, 52 PRB
    65,  0,  1,  0,   3,    66,    52,   5,    8,   41,   {ones(1,12)};
    % TRS on symbol 9, 52 PRB
    66,  0,  1,  0,   3,    66,    52,   9,    8,   41,   {ones(1,12)};
    % HARQ TC
 % cfg#  CSI Row CDM Density RB0  nRB  sym0  sym1   nID  FreqDomain
    67,  0,  1,  0,   3,    0,    273,   6,    0,   41,    {[0 1 0 0]}; % pattern-1
    68,  0,  1,  0,   3,    0,    273,  10,    0,   41,    {[0 1 0 0]}; % pattern-2
    69,  1,  4,  1,   2,    0,    273,  13,    0,   41,   {ones(1,12)}; % pattern-3
    70,  2,  5,  1,   2,    0,    273,   9,    0,   41,   {ones(1,12)}; % pattern-4
    % 64TR
    71,  0,  1,  0,   3,   20,    253,   6,    0,   41,    {[0 1 0 0]};
    72,  1,  4,  1,   2,   20,    253,   9,    0,   41,   {ones(1,12)};
    % 64TR test case
    73,  0,  1,  0,   3,    0,    273,   6,    0,   41,    {[0 1 0 0]};
    74,  0,  1,  0,   3,    0,    273,   9,    0,   41,    {[0 1 0 0]};
    75,  1,  4,  1,   2,    0,    273,  12,    0,   41,   {ones(1,12)};
    % Symbol 0 wideband CSI-RS for FH 
    76,  1,  4,  1,   2,    0,    256,   0,    0,   41,   {ones(1,12)};
    77,  1,  4,  1,   2,    0,    273,   0,    0,   41,   {ones(1,12)};
    78,  1,  4,  1,   2,    0,    273,   6,    0,   41,   {ones(1,12)};
    % 64TR test case, 8 ports (row 6)
    79,  0,  1,  0,   3,    0,    273,   6,    0,   41,    {[0 1 0 0]};
    80,  0,  1,  0,   3,    0,    273,   9,    0,   41,    {[0 1 0 0]};
    81,  1,  6,  1,   2,    0,    273,  12,    0,   41,   {ones(1,12)};
    % 64TR test 'worst case' col B, 1/4 ports
    % for odd cell 1,3,5 ..., even cell in 91~93
    82,  0,  1,  0,   3,    0,    273,   5,    0,   41,    {[0 1 0 0]};
    83,  0,  1,  0,   3,    0,    273,   9,    0,   41,    {[0 1 0 0]};
    84,  1,  4,  1,   2,    0,    273,  13,    0,   41,   {ones(1,12)};
    % 64TR pattern 64/65
 % cfg#  CSI Row CDM Density RB0  nRB  sym0  sym1   nID  FreqDomain
    % TRS on symbol 6
    85,  0,  1,  0,   3,     0,   273,   6,    8,   41,   {ones(1,12)};
    % TRS on symbol 10
    86,  0,  1,  0,   3,     0,   273,   10,   8,   41,   {ones(1,12)};
    % TRS on symbol 5
    87,  0,  1,  0,   3,     0,   273,   5,    8,   41,   {ones(1,12)};
    % TRS on symbol 9
    88,  0,  1,  0,   3,     0,   273,   9,    8,   41,   {ones(1,12)};
    % F08 at symbol 12
    89,  1,  4,  1,   2,     0,   273,   12,   8,   41,   {ones(1,12)};
    % F08 at symbol 13 
    90,  1,  4,  1,   2,     0,   273,   13,   8,   41,   {ones(1,12)};
    % 64TR test 'worst case' col B, 1/4 ports
    % for even cell 0,2,4 ..., odd cell in 82~84
    91,  0,  1,  0,   3,    0,    273,   6,    0,   41,    {[0 1 0 0]};
    92,  0,  1,  0,   3,    0,    273,   8,    0,   41,    {[0 1 0 0]};
    93,  1,  4,  1,   2,    0,    273,  12,    0,   41,   {ones(1,12)};
    % 64TR JFT test 'worst case' col G, 8/16 ports
    94,  1,  6,  1,   2,    0,    273,  13,    0,   41,   {ones(1,12)}; % 8 port, row 6 full BW, sym 13
    95,  1,  7,  1,   2,    0,    273,  12,   13,   41,   {ones(1,12)}; % 8 port, row 7 sym 12, 13
    96,  0,  1,  0,   3,    0,    273,   4,    0,   41,    {[0 1 0 0]};
    97,  0,  1,  0,   3,    0,    273,   6,    0,   41,    {[0 1 0 0]};
    98,  1,  6,  1,   2,    0,    273,   8,    0,   41,   {ones(1,12)}; % 8 port, row 6 full BW, sym 7
    99,  1,  7,  1,   2,    0,    273,   7,    8,   41,   {ones(1,12)}; % 8 port, row 7 sym 7, 8
    % column B patterns use {82 83 84} for even cell and {91 92 93} for odd cell
    % column G,H patterns use {82 83 95} for even cell and {96 97 99} for odd cell
 % cfg#  CSI Row CDM Density RB0  nRB  sym0  sym1   nID  FreqDomain
    % 64TR 40 MHz even cell
    100,  0,  1,  0,   3,    0,    106,   5,    0,   41,    {[0 1 0 0]};
    101,  0,  1,  0,   3,    0,    106,   9,    0,   41,    {[0 1 0 0]};
    102,  1,  7,  1,   2,    0,    106,  12,   13,   41,   {ones(1,12)}; % 8 port, row 7 sym 12, 13
    % 64TR 40 MHz odd cell
    103,  0,  1,  0,   3,    0,    106,   4,    0,   41,    {[0 1 0 0]};
    104,  0,  1,  0,   3,    0,    106,   6,    0,   41,    {[0 1 0 0]};
    105,  1,  7,  1,   2,    0,    106,   7,    8,   41,   {ones(1,12)}; % 8 port, row 7 sym 7, 8
    106,  1, 16,  1,   2,  150,     40,   8,   12,   41,   {ones(1,12)}; % 32 ports CSIRS 2 CDM
    107,  1, 18,  3,   2,  150,     40,   8,   12,   41,   {ones(1,12)}; % 32 ports CSIRS 8 CDM
    108,  1, 17,  2,   2,  150,     40,   8,   12,   41,   {ones(1,12)}; % 32 ports CSIRS 4 CDM
    109,  2,  4,  1,   2,   20,    253,   9,    0,   41,   {ones(1,12)}; % ZP-CSI-RS for use w/ modified SU-MIMO + MU-MIMO TC
 % cfg#  CSI Row CDM Density RB0  nRB  sym0  sym1   nID  FreqDomain
    % 64TR TRS    
    110,  0,  1,  0,   3,    0,    273,   7,    0,   41,    {[0 1 0 0]}; % 8 ports, row 1, sym 7, odd cell
    % 64TR ZP-CSI-RS (CSI-IM)
    111,  2,  5,  1,   2,    0,    273,   8,    0,   41,    {[0 0 0 1 0 0]}; % 8 ports, row 5, sym 8, even cell
    112,  2,  5,  1,   2,    0,    273,   1,    0,   41,    {[0 0 0 1 0 0]}; % 8 ports, row 5, sym 1, odd cell
    % column I patterns use {82 83 84 111} or {82 83} for even cell and {91 110 93 112} or {91 110} for odd cell
    113,  1,  7,  1,   2,    0,    273,   4,   5,   41,   {ones(1,12)}; % 8 port, row 7 sym 4, 5
    114,  2,  5,  1,   2,    0,    273,   2,   3,   41,   {[0 0 0 1 0 0]}; % 8 port, row 5 sym 2, 3, ZP CSI-RS
    % Ph4 column B patterns, both odd and even cells use {82 83} or {113 114}
    % 32-port CSI-RS variant of 8-port version above
    115,  1,  16, 1,   2,    0,    273,   4,   5,   41,   {ones(1,12)}; % 32-port, row 16 sym 4, 5
    116,  2,  16, 1,   2,    0,    273,   2,   3,   41,   {ones(1,12)}; % 32-port, row 16 sym 2, 3, ZP CSI-RS
    % Ph4 column B 32-port CSI-RS patterns, both odd and even cells use {82 83} or {115 116}
    % CSI-RS for heterogeneous UE group pattern
    117,  0,  1,  0,   3,   20,    253,   6,    0,   41,    {[0 1 0 0]};
    118,  1,  4,  1,   2,   20,    253,   9,    0,   41,    {ones(1,12)};
 % cfg#  CSI Row CDM Density RB0  nRB  sym0  sym1   nID  FreqDomain
    % mixed 8-port and 32-port CSI-RS pattern
    119,  1,   7, 1,   2,    0,    273,   4,   5,   41,   {[1 1 0 0 0 0]}; %  8-port, row 7 sym 4, 5
    120,  1,  16, 1,   2,    0,    273,   2,   4,   41,   {ones(1,12)};    % 32-port, row 16 sym 2, 3, 4, 5
    121,  1,  16, 1,   2,    0,    273,   4,   6,   41,   {[0 0 1 1 1 1]}; % 32-port, row 16 sym 4, 5, 6, 7 - unused
    % 25-3 column B patterns, both odd and even cells use {119 120 114}
    % 25-3 column G patterns, both odd and even cells use {119 120 114}
    % 64TR realistic traffic with Center/Middle/Edge, 90 MHz heavy / light
    122,  0,  1,  0,   3,    0,    244,   5,    0,   41,    {[0 1 0 0]};  % modified from 82
    123,  0,  1,  0,   3,    0,    244,   9,    0,   41,    {[0 1 0 0]};  % modified from 83
    124,  1,  7,  1,   2,    0,    244,   4,   5,   41,   {ones(1,12)}; % 8 port, row 7 sym 4, 5; modified from 113
    125,  2,  5,  1,   2,    0,    244,   2,   3,   41,   {[0 0 0 1 0 0]}; % 8 port, row 5 sym 2, 3, ZP CSI-RS; modified from 114
    126,  1,  16, 1,   2,    0,    244,   4,   5,   41,   {ones(1,12)}; % 32-port, row 16 sym 4, 5; modified from 115
    127,  2,  16, 1,   2,    0,    244,   2,   3,   41,   {ones(1,12)}; % 32-port, row 16 sym 2, 3, ZP CSI-RS; modified from 116
    % 64TR realistic traffic with Center/Middle/Edge, 60 MHz heavy / light
    128,  0,  1,  0,   3,    0,    160,   5,    0,   41,    {[0 1 0 0]};  % modified from 82
    129,  0,  1,  0,   3,    0,    160,   9,    0,   41,    {[0 1 0 0]};  % modified from 83
    130,  1,  7,  1,   2,    0,    160,   4,   5,   41,   {ones(1,12)}; % 8 port, row 7 sym 4, 5; modified from 113
    131,  2,  5,  1,   2,    0,    160,   2,   3,   41,   {[0 0 0 1 0 0]}; % 8 port, row 5 sym 2, 3, ZP CSI-RS; modified from 114
    132,  1,  16, 1,   2,    0,    160,   4,   5,   41,   {ones(1,12)}; % 32-port, row 16 sym 4, 5; modified from 115
    133,  2,  16, 1,   2,    0,    160,   2,   3,   41,   {ones(1,12)}; % 32-port, row 16 sym 2, 3, ZP CSI-RS; modified from 116
};

CFG_PRACH = {...
 % cfg#   duplex  mu  cfg restrictSet root  zone  prmbIdx  RA0   
    % HARQ TC
    41,        1,  1, 158,     0,        0,   14,     0,    60;
    42,        1,  1, 158,     0,        0,   14,     1,    60+12;
    43,        1,  1, 158,     0,        0,   14,     2,    60+24;
    44,        1,  1, 158,     0,        0,   14,     3,    60+36;
    45,        1,  1, 158,     0,        0,   14,     0,    260;    
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
fprintf('DLMIX: genTV = %d, testCompliance = %d, caseSet = %s\n', genTV, testCompliance, caseSetStr);
fprintf('\nTC#   slotIdx   cell    ssb  pdcch  pdsch  csirs\n');
fprintf('------------------------------------------------\n');

parfor n = 1:NallTest
    testAlloc = [];
    testAlloc.dl = 1;
    testAlloc.ul = 0;
    idxSet = n;
    caseNum =CFG{idxSet, 1};
    slotIdx = CFG{idxSet, 2};
    if ismember(caseNum, TcToTest) && (mod(caseNum, subSetMod(2)) == subSetMod(1))
        rng(caseNum);
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

            SysPar.SimCtrl.genTV.slotIdx = slotIdx;
            SysPar.SimCtrl.genTV.forceSlotIdxFlag = 1;
            SysPar.SimCtrl.genTV.enable = 1;
            if caseNum < 1000
                SysPar.SimCtrl.genTV.TVname = sprintf('TVnr_%04d', caseNum);
            else
                SysPar.SimCtrl.genTV.TVname = sprintf('TVnr_DLMIX_%04d', caseNum);
            end
            if ismember(caseNum, disabled_TC)
                if caseNum < 1000
                    SysPar.SimCtrl.genTV.TVname = sprintf('disabled_TVnr_%04d', caseNum);
                else
                    SysPar.SimCtrl.genTV.TVname = sprintf('disabled_TVnr_DLMIX_%04d', caseNum);
                end
            end
        end
        
        % config Carrier
        % mu N_grid_size_mu N_ID_CELL Nant_gNB Nant_UE
        idxCfg = CFG{idxSet,3};
        SysPar.carrier.mu = CFG_Cell{idxCfg, 1};
        SysPar.carrier.N_grid_size_mu = CFG_Cell{idxCfg, 2};
        if CellIdxInPatternMap.isKey(caseNum)
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3} + CellIdxInPatternMap(caseNum);
        else
            SysPar.carrier.N_ID_CELL = CFG_Cell{idxCfg, 3};
        end
        SysPar.carrier.Nant_gNB = CFG_Cell{idxCfg, 4};
        SysPar.carrier.Nant_UE = CFG_Cell{idxCfg, 5};

        if ismember(caseNum, MIMO_64TR_TC) && (SysPar.carrier.Nant_UE < 32)
            error('Case %d: Nant_UE must be greater than or equal to 32 for MIMO 64TR TCs.  Please review the cell configuration.', caseNum);
        end
        
         % FAPI override for negative test cases
        if (caseNum == 502)
            SysPar.SimCtrl.negTV.enable = 1;  % enable negTV 
            SysPar.SimCtrl.negTV.pduFieldName = {'SsbOffsetPointA'};
            SysPar.SimCtrl.negTV.pduFieldValue = {2*SysPar.carrier.N_grid_size_mu + 20};
        end

        if idxCfg == 8 % 32T32R
            SysPar.SimCtrl.enable_dynamic_BF = 1;
            SysPar.carrier.N_FhPort_DL = 8;
            SysPar.carrier.N_FhPort_UL = 4;
            SysPar.carrier.Nant_gNB_srs = 32;
        end

        if SysPar.carrier.Nant_UE == 64 % 64TR static and dynamic beamforming
            SysPar.SimCtrl.enable_static_dynamic_beamforming = 1;
            SysPar.carrier.Nant_gNB_srs = 64;
        end
        
        % config SSB
        cfg = CFG{idxSet, 4};
        for idx = 1:length(cfg)
            idxCfg = find(cellfun(@(x) isequal(x,cfg{idx}), CFG_SSB(:,1)));
            SysPar.ssb = cfgSsb;
            % cfg#  Nid  n_hf  L_max k_SSB offsetPA SFN  blockIdx
            SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2};
            if ismember(caseNum, [130 : 137])
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-130, 4);
            elseif ismember(caseNum, [140 : 147])
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-140, 4);
            elseif ismember(caseNum, [148 : 171])  % 8C (three sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-148, 8);
            elseif ismember(caseNum, [172 : 195])  % 8C (three sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-172, 8);
            elseif ismember(caseNum, [196 : 219])  % 8C (three sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-196, 8);
            elseif ismember(caseNum, [220 : 243])  % 8C (three sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-220, 8);
            elseif ismember(caseNum, [244 : 291])  % 8C (six sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-244, 8);
            elseif ismember(caseNum, [292 : 387])  % 16C (six sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-292, 16);
            elseif ismember(caseNum, [388 : 483])  % 16C (six sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-388, 16);
            elseif ismember(caseNum, [1000 : 1015])  % 16C (1 set)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1000, 16);
            elseif ismember(caseNum, [1016 : 1111])  % 16C (six sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1016, 16);
            elseif ismember(caseNum, [1112 : 1207])  % 16C (six sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1112, 16);
            elseif ismember(caseNum, [1208 : 1303])  % 16C (six sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1208, 16);
            elseif ismember(caseNum, [1304 : 1351])  % 8C (six sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1304, 8);
            elseif ismember(caseNum, [1352 : 1383])  % 16C (two sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1352, 16);
            elseif ismember(caseNum, [1384 : 1415])  % 16C (two sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1384, 16);
            elseif ismember(caseNum, [1416 : 1479])  % 16C (4 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1416, 16);
            elseif ismember(caseNum, [1480 : 1527])  % 16C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1480, 16);
            elseif ismember(caseNum, [1528 : 1559])  % 8C (4 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1528, 8);
            elseif ismember(caseNum, [1560 : 1583])  % 8C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1560, 8);
            elseif ismember(caseNum, [1584 : 1615])  % 8C (4 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1584, 8);
            elseif ismember(caseNum, [1616 : 1639])  % 8C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1616, 8);
            elseif ismember(caseNum, [1640 : 1687])  % 16C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1640, 16);
            elseif ismember(caseNum, [1688 : 1711])  % 8C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1688, 8);
            elseif ismember(caseNum, [1712 : 1743])  % 16C (2 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1712, 16);
            elseif ismember(caseNum, [1744 : 1807])  % 16C (4 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1744, 16);
            elseif ismember(caseNum, [1808 : 1855])  % 16C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1808, 16);
            elseif ismember(caseNum, [1856 : 1903])  % 16C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1856, 16);
            elseif ismember(caseNum, [1904 : 1927])  % 8C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1904, 8);
            elseif ismember(caseNum, [1928 : 1975])  % 16C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1928, 16);
            elseif ismember(caseNum, [1976 : 1999])  % 8C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-1976, 8);
            elseif ismember(caseNum, [2000 : 2063])  % 16C (4 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-2000, 16);
            elseif ismember(caseNum, [2064 : 2111])  % 16C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-2064, 16);
            elseif ismember(caseNum, [2144 : 2167])  % 8C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-2144, 8);
            elseif ismember(caseNum, [2168 : 2215])  % 16C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-2168, 16);
            elseif ismember(caseNum, [2216 : 2263])  % 16C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-2216, 16);
            elseif ismember(caseNum, [2264 : 2287])  % 8C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-2264, 8);
            elseif ismember(caseNum, [2288 : 2335])  % 16C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-2288, 16);
            elseif ismember(caseNum, [2336 : 2359])  % 8C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-2336, 8);
            elseif ismember(caseNum, [2360 : 2407])  % 16C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-2360, 16);
            elseif ismember(caseNum, [2512 : 2591])  % 16C (5 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-2512, 16);
            elseif ismember(caseNum, [2592 : 2671])  % 16C (5 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-2592, 16);
            elseif ismember(caseNum, [2672 : 2719])  % 16C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-2672, 16);
            elseif ismember(caseNum, [2720 : 2767])  % 12C (4 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-2720, 12);
            elseif ismember(caseNum, [2768 : 2803])  % 12C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-2768, 12);
            elseif ismember(caseNum, [2804 : 2827])  % 12C (2 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-2804, 12);
            elseif ismember(caseNum, [2828 : 2875])  % 12C (4 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-2828, 12);
            elseif ismember(caseNum, [2876 : 2911])  % 12C (3 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-2876, 12);
            elseif ismember(caseNum, [2912 : 2935])  % 12C (2 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-2912, 12);
            elseif ismember(caseNum, [2936 : 3079])  % 16C (9 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-2936, 16);
            elseif ismember(caseNum, [3080 : 3159])  % 16C (5 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-3080, 16);
            elseif ismember(caseNum, [3160 : 3399])  % 12C (20 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-3160, 12);
            elseif ismember(caseNum, [3400 : 3719])  % 16C (20 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-3400, 16);
            elseif ismember(caseNum, [3720 : 4039])  % 16C (20 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-3720, 16);
            elseif ismember(caseNum, [4040 : 4439])  % 20C (20 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-4040, 20);
            elseif ismember(caseNum, [4440 : 4759])  % 16C (20 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-4440, 16);
            elseif ismember(caseNum, [4760 : 5159])  % 20C (20 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-4760, 20);
            elseif ismember(caseNum, [5160 : 5479])  % 16C (20 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-5160, 16);
            elseif ismember(caseNum, [5480 : 5879])  % 20C (20 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-5480, 20);
            elseif ismember(caseNum, [6032 : 6351])  % 16C (20 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-6032, 16);
            elseif ismember(caseNum, [6352 : 6751])  % 20C (20 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-6352, 20);
            elseif ismember(caseNum, [5960 : 6031])  % 8C (9 sets)
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum-5960, 8);                
            elseif ismember(caseNum, [6752 : 7071]) % 16 cells
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum - 6752, 16);
            elseif ismember(caseNum, [7072 : 7471]) % 20 cells
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum - 7072, 20);
            elseif ismember(caseNum, [7872 : 8271]) % 20 cells
                SysPar.carrier.N_ID_CELL =  CFG_SSB{idxCfg, 2} + mod(caseNum - 7872, 20);
            end

            if ismember(caseNum, [130:135])
                SysPar.carrier.N_ID_CELL = 41;
            elseif caseNum == 140
                SysPar.carrier.N_ID_CELL = 40;
            elseif caseNum == 141
                SysPar.carrier.N_ID_CELL = 41;
            elseif caseNum == 142
                SysPar.carrier.N_ID_CELL = 42;
            elseif caseNum == 143
                SysPar.carrier.N_ID_CELL = 43;
            elseif caseNum == 144
                SysPar.carrier.N_ID_CELL = 44;
            end
    
            if ismember(caseNum, [150:189, MIMO_64TR_TC]) && ~CellIdxInPatternMap.isKey(caseNum) % HARQ TC, 64TR
                SysPar.carrier.N_ID_CELL = 41;
            end
    
            if ismember(caseNum, [20200:20399])
                SysPar.carrier.N_ID_CELL = 41 + floor((caseNum-20200)/20);
            end
            
            if ismember(caseNum, [20000:20199])
                SysPar.carrier.N_ID_CELL = 41 + floor((caseNum-20000)/40);
            end
    
            if ismember(caseNum, [20400:20459])
                SysPar.carrier.N_ID_CELL = 41 + floor((caseNum-20400)/20);
            end
    
            if ismember(caseNum, [20460:20519])
                SysPar.carrier.N_ID_CELL = 41 + floor((caseNum-20460)/20);
            end
    
            if CellIdxInPatternMap.isKey(caseNum)
                SysPar.carrier.N_ID_CELL = CFG_SSB{idxCfg, 2} + CellIdxInPatternMap(caseNum);
            end 
            % end cell id config
            
            SysPar.ssb.n_hf =  CFG_SSB{idxCfg, 3};
            SysPar.ssb.L_max = CFG_SSB{idxCfg, 4};        
            SysPar.ssb.ssbSubcarrierOffset = CFG_SSB{idxCfg, 5};
            SysPar.ssb.SsbOffsetPointA = CFG_SSB{idxCfg, 6};
            SysPar.carrier.SFN_start = CFG_SSB{idxCfg, 7};
            if SysPar.ssb.L_max == 8
                SysPar.ssb.symIdxInFrame = [2, 8, 16, 22, 30, 36, 44, 50];
            elseif SysPar.ssb.L_max == 4
                SysPar.ssb.symIdxInFrame = [2, 8, 16, 22];
            end
            if iscell(CFG_SSB{idxCfg, 8})
                blockIdx =  cell2mat(CFG_SSB{idxCfg, 8});
            else
                blockIdx =  CFG_SSB{idxCfg, 8};
            end
            SysPar.ssb.ssbBitMap = zeros(1, 8);
            SysPar.ssb.ssbBitMap(blockIdx+1) = 1;
            
            % read beam id from TV
            if SsbBeamIdxMap.isKey(caseNum)
                beam_idx = SsbBeamIdxMap(caseNum) + beamIdx_ssb_static(1);
                SysPar.ssb.prcdBf_vec = zeros(1,length(beam_idx));
                for i = 1:length(beam_idx)                    
                    SysPar.ssb = setfield(SysPar.ssb,'digBFInterfaces_' + string(i-1),1);
                    SysPar.ssb = setfield(SysPar.ssb,'beamIdx_' + string(i-1),beam_idx(i));
                end
            else  % default config
                start_beam_idx = beamIdx_ssb_static(1)-1;
                for idxSsb = 1:length(SysPar.ssb.prcdBf_vec)
                    digBFInterfaces = 1;
                    switch SysPar.ssb.prcdBf_vec(idxSsb)
                        case 0
                            digBFInterfaces = 1;
                        case {1, 2, 5, 6}
                            digBFInterfaces = 2;
                        case {3, 4, 7, 8, 9, 10, 11, 12}
                            digBFInterfaces = 4;
                        otherwise
                            error('prcdBf is not supported ... \n');
                    end
                    SysPar.ssb = setfield(SysPar.ssb,'digBFInterfaces_' + string(idxSsb-1),digBFInterfaces);
                    SysPar.ssb = setfield(SysPar.ssb,'beamIdx_' + string(idxSsb-1),start_beam_idx+[1:digBFInterfaces]);
                    start_beam_idx = start_beam_idx + digBFInterfaces;
                end
            end
        end

        if ismember(caseNum, 190)
            SysPar.ssb.beamIdx_0 = 1;
            SysPar.ssb.beamIdx_1 = 2;
            SysPar.ssb.beamIdx_2 = 3;
            SysPar.ssb.beamIdx_3 = 4;
        end

        testAlloc.ssb = length(cfg);

        % config PDCCH
        cfg = CFG{idxSet, 5};
        
        if ismember(caseNum, [150:189, MIMO_64TR_TC]) % HARQ TC, 64TR
            idx_dci = 0;
            mapZeros = zeros(1,14); % track for all OFDM symbols
            for idxDLUL = 1:length(cfg)
                idxCfg = find(cellfun(@(x) isequal(x,cfg{idxDLUL}), CFG_PDCCH(:,1)));
                nPdcch =  length(CFG_PDCCH{idxCfg, 16});
                for idxPdcch = 1:nPdcch
                    idx_dci = idx_dci + 1;
                    SysPar.pdcch{idx_dci} = cfgPdcch;
                    SysPar.pdcch{idx_dci}.BWPSize = CFG_PDCCH{idxCfg, 2};
                    SysPar.pdcch{idx_dci}.BWPStart = CFG_PDCCH{idxCfg, 3};
                    SysPar.pdcch{idx_dci}.StartSymbolIndex = CFG_PDCCH{idxCfg, 4};
                    SysPar.pdcch{idx_dci}.DurationSymbols = CFG_PDCCH{idxCfg, 5};
                    SysPar.pdcch{idx_dci}.coresetIdx = CFG_PDCCH{idxCfg, 6} + idx_dci - 1;
                    SysPar.pdcch{idx_dci}.CceRegMappingType = CFG_PDCCH{idxCfg, 7};
                    SysPar.pdcch{idx_dci}.RegBundleSize =  CFG_PDCCH{idxCfg, 8};
                    SysPar.pdcch{idx_dci}.InterleaverSize =  CFG_PDCCH{idxCfg, 9};
                    SysPar.pdcch{idx_dci}.ShiftIndex =  CFG_PDCCH{idxCfg, 10};
                    % SysPar.pdcch{idx_dci}.numDlDci =  CFG_PDCCH{idxCfg, 11};
                    SysPar.pdcch{idx_dci}.numDlDci =  1;
                    SysPar.pdcch{idx_dci}.isCSS =  CFG_PDCCH{idxCfg, 12};
                    SysPar.pdcch{idx_dci}.forceCceIndex = 1;
                    if ismember(idxCfg, [25:38, 53, 55, 57, 61, 63, 64, 66, 67, 69, 70, 72, 73, 75, 80, 81, 83, 84, 86]) % 64TR
                        SysPar.pdcch{idx_dci}.dciUL = 0;
                    elseif ismember(idxCfg, [39:52, 54, 56, 58, 62, 65, 68, 71, 74, 82, 85]) 
                        SysPar.pdcch{idx_dci}.dciUL = 1;
                    end
                    % SysPar.pdcch{idx_dci}.coresetMap = cell2mat(CFG_PDCCH{idxCfg, 18});
                    aggL = cell2mat(CFG_PDCCH{idxCfg, 16});
                    
                    ofdmSymbolIdx = SysPar.pdcch{idx_dci}.StartSymbolIndex + (0:SysPar.pdcch{idx_dci}.DurationSymbols-1);
                    mapOnes = aggL(idxPdcch);
                    SysPar.pdcch{idx_dci}.coresetMap = [zeros(1, max(mapZeros(ofdmSymbolIdx+1))), ones(1, mapOnes)];
                    mapZeros(ofdmSymbolIdx+1) = max(mapZeros(ofdmSymbolIdx+1)) + mapOnes;
                    
                    SysPar.pdcch{idx_dci}.idxUE = idx_dci-1;
                    
                    if length(CFG_PDCCH{idxCfg, 13})==1
                        SysPar.pdcch{idx_dci}.DCI{1}.RNTI = CFG_PDCCH{idxCfg, 13}{1} + idxPdcch;
                    else
                        SysPar.pdcch{idx_dci}.DCI{1}.RNTI = CFG_PDCCH{idxCfg, 13}{idxPdcch};
                    end
                    
                    if length(CFG_PDCCH{idxCfg, 14})==1
                        %                     SysPar.pdcch{idx_dci}.DCI{1}.ScramblingId = CFG_PDCCH{idxCfg, 14}{1} + idx;
                        SysPar.pdcch{idx_dci}.DCI{1}.ScramblingId = CFG_PDCCH{idxCfg, 14}{1};
                    else
                        SysPar.pdcch{idx_dci}.DCI{1}.ScramblingId = CFG_PDCCH{idxCfg, 14}{idxPdcch};
                    end
                    
                    if length(CFG_PDCCH{idxCfg, 15}) == 1
                        SysPar.pdcch{idx_dci}.DCI{1}.ScramblingRNTI = CFG_PDCCH{idxCfg, 15}{1} + idxPdcch;
                    else
                        SysPar.pdcch{idx_dci}.DCI{1}.ScramblingRNTI = CFG_PDCCH{idxCfg, 15}{idx_dci};
                    end
                    
                    if length(CFG_PDCCH{idxCfg, 16}) == 1
                        SysPar.pdcch{idx_dci}.DCI{1}.AggregationLevel = CFG_PDCCH{idxCfg, 16}{1};
                    else
                        SysPar.pdcch{idx_dci}.DCI{1}.AggregationLevel = CFG_PDCCH{idxCfg, 16}{idxPdcch};
                    end
                    
                    if length(CFG_PDCCH{idxCfg, 17}) == 1
                        SysPar.pdcch{idx_dci}.DCI{1}.PayloadSizeBits = CFG_PDCCH{idxCfg, 17}{1};
                    else
                        SysPar.pdcch{idx_dci}.DCI{1}.PayloadSizeBits = CFG_PDCCH{idxCfg, 17}{idxPdcch};
                    end
                    
                    if length(CFG_PDCCH{idxCfg, 17}) == 1
                        SysPar.pdcch{idx_dci}.DCI{1}.Payload = round(rand(1, CFG_PDCCH{idxCfg, 17}{1}));
                    else
                        SysPar.pdcch{idx_dci}.DCI{1}.Payload = round(rand(1, CFG_PDCCH{idxCfg, 17}{idxPdcch}));
                    end
                    SysPar.pdcch{idx_dci}.DCI{1}.cceIndex = 0;
                end
            end
            testAlloc.pdcch = idx_dci;
        else  % non HARQ TC      
            for idx = 1:length(cfg)
                idxCfg = find(cellfun(@(x) isequal(x,cfg{idx}), CFG_PDCCH(:,1)));
                SysPar.pdcch{idx} = cfgPdcch;
                % cfg# nBWP BWP0 sym0  Nsym crstIdx intl nBndl nIntl nShift nDCI isCSS       rnti      scrbId   scrbRnti    aggrL   Npaylod
                SysPar.pdcch{idx}.BWPSize = CFG_PDCCH{idxCfg, 2};
                SysPar.pdcch{idx}.BWPStart = CFG_PDCCH{idxCfg, 3};
                SysPar.pdcch{idx}.StartSymbolIndex = CFG_PDCCH{idxCfg, 4};
                SysPar.pdcch{idx}.DurationSymbols = CFG_PDCCH{idxCfg, 5};
                SysPar.pdcch{idx}.coresetIdx = CFG_PDCCH{idxCfg, 6};
                SysPar.pdcch{idx}.CceRegMappingType = CFG_PDCCH{idxCfg, 7};
                SysPar.pdcch{idx}.RegBundleSize =  CFG_PDCCH{idxCfg, 8};
                SysPar.pdcch{idx}.InterleaverSize =  CFG_PDCCH{idxCfg, 9};
                SysPar.pdcch{idx}.ShiftIndex =  CFG_PDCCH{idxCfg, 10};
                SysPar.pdcch{idx}.numDlDci =  CFG_PDCCH{idxCfg, 11};
                SysPar.pdcch{idx}.isCSS =  CFG_PDCCH{idxCfg, 12};
                SysPar.pdcch{idx}.coresetMap = cell2mat(CFG_PDCCH{idxCfg, 18});
                SysPar.pdcch{idx}.idxUE = idx-1;
                numDlDci = SysPar.pdcch{idx}.numDlDci;
                if ismember(caseNum, [120:128]) % DL test model
                    SysPar.pdcch{idx}.testModel = 1;
                end
                DCI = SysPar.pdcch{idx}.DCI{1};
                SysPar.pdcch{idx}.DCI = [];
                for idxDCI =  1:numDlDci
                    SysPar.pdcch{idx}.DCI{idxDCI} = DCI;
                    if length(CFG_PDCCH{idxCfg, 13})==1
                        SysPar.pdcch{idx}.DCI{idxDCI}.RNTI = CFG_PDCCH{idxCfg, 13}{1} + idxDCI - 1;
                    else
                        SysPar.pdcch{idx}.DCI{idxDCI}.RNTI = CFG_PDCCH{idxCfg, 13}{idxDCI};
                    end

                    if length(CFG_PDCCH{idxCfg, 14})==1
                        SysPar.pdcch{idx}.DCI{idxDCI}.ScramblingId = CFG_PDCCH{idxCfg, 14}{1} + idxDCI - 1;
                    else
                        SysPar.pdcch{idx}.DCI{idxDCI}.ScramblingId = CFG_PDCCH{idxCfg, 14}{idxDCI};
                    end

                    if length(CFG_PDCCH{idxCfg, 15}) == 1
                        SysPar.pdcch{idx}.DCI{idxDCI}.ScramblingRNTI = CFG_PDCCH{idxCfg, 15}{1} + idxDCI - 1;
                    else
                        SysPar.pdcch{idx}.DCI{idxDCI}.ScramblingRNTI = CFG_PDCCH{idxCfg, 15}{idxDCI};
                    end

                    if length(CFG_PDCCH{idxCfg, 16}) == 1
                        SysPar.pdcch{idx}.DCI{idxDCI}.AggregationLevel = CFG_PDCCH{idxCfg, 16}{1};
                    else
                        SysPar.pdcch{idx}.DCI{idxDCI}.AggregationLevel = CFG_PDCCH{idxCfg, 16}{idxDCI};
                    end

                    if length(CFG_PDCCH{idxCfg, 17}) == 1
                        SysPar.pdcch{idx}.DCI{idxDCI}.PayloadSizeBits = CFG_PDCCH{idxCfg, 17}{1};
                    else
                        SysPar.pdcch{idx}.DCI{idxDCI}.PayloadSizeBits = CFG_PDCCH{idxCfg, 17}{idxDCI};
                    end

                    if length(CFG_PDCCH{idxCfg, 17}) == 1
                        SysPar.pdcch{idx}.DCI{idxDCI}.Payload = round(rand(1, CFG_PDCCH{idxCfg, 17}{1}));
                    else
                        SysPar.pdcch{idx}.DCI{idxDCI}.Payload = round(rand(1, CFG_PDCCH{idxCfg, 17}{idxDCI}));
                    end

                    if ismember(idxCfg, [3,4,6,7,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24,59,60])
                        SysPar.pdcch{idx}.DCI{idxDCI}.cceIndex = idxDCI-1; % only valid for aggregation level == 1
                        SysPar.pdcch{idx}.forceCceIndex = 1;
                    end

                    % HARQ TC
                    if ismember(idxCfg, [25:52])
                        aggL = cell2mat(CFG_PDCCH{idxCfg, 16});
                        SysPar.pdcch{1}.DCI{idxDCI}.cceIndex = sum(aggL(1:idxDCI-1));
                        SysPar.pdcch{1}.forceCceIndex = 1;
                    end
                end

                SysPar.pdcch{idx}.dciUL = mod(idx-1, 2);
                if ismember(idxCfg, [7,11,13,15,17,19,21,23,59])
                    SysPar.pdcch{idx}.dciUL = 0;
                end
                if ismember(idxCfg, [8,12,14,16,18,20,22,24,60])
                     SysPar.pdcch{idx}.dciUL = 1;
                end
            end
            testAlloc.pdcch = length(cfg);
        end

        % read beam id from TV
        
        pdcch_beam_idx = 1;
        for idxUe = 1:length(SysPar.pdcch)
            for idxDCI = 1:SysPar.pdcch{idxUe}.numDlDci
                digBFInterfaces = 1;
                switch SysPar.pdcch{idxUe}.DCI{idxDCI}.prcdBf
                    case 0
                        digBFInterfaces = SysPar.carrier.N_FhPort_DL;
                    case {1, 2, 5, 6}
                        digBFInterfaces = 2;
                    case {3, 4, 7, 8, 9, 10, 11, 12}
                        digBFInterfaces = 4;
                    otherwise
                        error('prcdBf is not supported ... \n');
                end
                SysPar.pdcch{idxUe}.DCI{idxDCI}.digBFInterfaces = digBFInterfaces;
                SysPar.pdcch{idxUe}.DCI{idxDCI}.beamIdx = repelem(pdcch_beam_idx, digBFInterfaces); % different beamIdx for different DCIs
                pdcch_beam_idx = pdcch_beam_idx + 1;
            end
            
        end
        
        pdcch_cfg_idx = cell2mat(CFG{idxSet, 5});
        if ismember(caseNum,  [MIMO_64TR_TC]) ...  % 64TR
           || any(ismember(pdcch_cfg_idx, [59, 60]))  % 64TR PDCCH
            beamIdx_static_offset = beamIdx_pdcch_static(1);
            for idxUe = 1:length(SysPar.pdcch)
                for idxDCI = 1:SysPar.pdcch{idxUe}.numDlDci
                    digBFInterfaces = 1;
                    SysPar.pdcch{idxUe}.DCI{idxDCI}.digBFInterfaces = digBFInterfaces;
                    pdcch_beam_idx = beamIdx_static_offset + [1:digBFInterfaces];
                    SysPar.pdcch{idxUe}.DCI{idxDCI}.beamIdx = repelem(pdcch_beam_idx, digBFInterfaces); % different beamIdx for different DCIs
                    beamIdx_static_offset = beamIdx_static_offset + digBFInterfaces;
                end
            end
        end

        % config PDSCH
        cfg = CFG{idxSet, 6};
        for idx = 1:length(cfg)
            idxCfg = find(cellfun(@(x) isequal(x,cfg{idx}), CFG_PDSCH(:,1)));
            SysPar.pdsch{idx} = cfgPdsch;
            % cfg#  mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym   SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 nAnt
            SysPar.pdsch{idx}.mcsTable =  CFG_PDSCH{idxCfg, 2};
            SysPar.pdsch{idx}.mcsIndex =  CFG_PDSCH{idxCfg, 3};
            SysPar.pdsch{idx}.nrOfLayers = CFG_PDSCH{idxCfg, 4};
            SysPar.pdsch{idx}.rbStart = CFG_PDSCH{idxCfg, 5};
            SysPar.pdsch{idx}.rbSize  = CFG_PDSCH{idxCfg, 6};
            if SysPar.SimCtrl.enable_static_dynamic_beamforming == 1
                SysPar.pdsch{idx}.prgSize = 2;
                SysPar.pdsch{idx}.numPRGs = ceil(SysPar.pdsch{idx}.rbSize/SysPar.pdsch{idx}.prgSize);
            end
            if ismember(caseNum, 296)
                SysPar.pdsch{idx}.prgSize = SysPar.pdsch{idx}.rbSize;
                SysPar.pdsch{idx}.numPRGs = 1;
            end
            if ismember(CFG_PDSCH{idxCfg, 1}, [2721:3080])
                SysPar.pdsch{idx}.prgSize = 16;
                SysPar.pdsch{idx}.numPRGs = ceil(SysPar.pdsch{idx}.rbSize/SysPar.pdsch{idx}.prgSize);
            end
            if  VARY_PRB_NUM > 0  % in study of sweeping PRB numbers, overwrite pdsch PRB numbers 
                if ismember(caseNum, [2936:2983, 3400:3463, 4520:4583]) % having SSB in the slot
                    PRB_num_w_SSB = min(VARY_PRB_NUM, floor((273-20)/6));
                    SysPar.pdsch{idx}.rbStart = 20 + (idx - 1) * PRB_num_w_SSB;
                    SysPar.pdsch{idx}.rbSize = PRB_num_w_SSB;
                elseif ismember(caseNum, [2984:2999,3112:3127,3144:3159, 3576:3591, ... % pattern 44,47 no ssb, no csi
                        4696:4711]) % pattern 51
                    PRB_num_noSSB_noCSI = VARY_PRB_NUM;
                    SysPar.pdsch{idx}.rbStart = (idx - 1) * PRB_num_noSSB_noCSI;
                    SysPar.pdsch{idx}.rbSize = PRB_num_noSSB_noCSI;
                elseif ismember(caseNum, [3080:3111, 3128:3143, 3464:3575, 3592:3703, ... % pattern 44,47 with csi
                        4584:4695, 4712:4823]) % pattern 51
                    PRB_num_w_csi = min(VARY_PRB_NUM, floor((273-52)/6));
                    SysPar.pdsch{idx}.rbStart = (idx - 1) * PRB_num_w_csi;
                    SysPar.pdsch{idx}.rbSize = PRB_num_w_csi;
                end                
            end
            SysPar.pdsch{idx}.StartSymbolIndex = CFG_PDSCH{idxCfg, 7};
            SysPar.pdsch{idx}.NrOfSymbols = CFG_PDSCH{idxCfg, 8};
            SysPar.pdsch{idx}.SCID =  CFG_PDSCH{idxCfg, 9};
            SysPar.pdsch{idx}.BWPStart =  CFG_PDSCH{idxCfg, 10};
            SysPar.pdsch{idx}.BWPSize =  CFG_PDSCH{idxCfg, 11};
            SysPar.pdsch{idx}.RNTI =  CFG_PDSCH{idxCfg, 12};
            SysPar.pdsch{idx}.rvIndex =  CFG_PDSCH{idxCfg, 13};
            SysPar.pdsch{idx}.dataScramblingId =  CFG_PDSCH{idxCfg, 14};
            
            if ismember(caseNum, [120:128]) % DL test model
                SysPar.pdsch{idx}.testModel = 1;
            end
            
            % HARQ TC            
            if idxCfg == 321
                SysPar.pdsch{idx}.targetCodeRate = 1930;
                SysPar.pdsch{idx}.qamModOrder = 2;
                SysPar.pdsch{idx}.TBSize = 1569;
            elseif idxCfg == 326
                SysPar.pdsch{idx}.targetCodeRate = 1930;
                SysPar.pdsch{idx}.qamModOrder = 2;
                SysPar.pdsch{idx}.TBSize = 1793;
            elseif idxCfg == 327
                SysPar.pdsch{idx}.targetCodeRate = 6580;
                SysPar.pdsch{idx}.qamModOrder = 4;
                SysPar.pdsch{idx}.TBSize = 8448;
            elseif idxCfg == 328
                SysPar.pdsch{idx}.targetCodeRate = 8730;
                SysPar.pdsch{idx}.qamModOrder = 6;
                SysPar.pdsch{idx}.TBSize = 22547;
            elseif idxCfg == 335
                SysPar.pdsch{idx}.targetCodeRate = 1930;
                SysPar.pdsch{idx}.qamModOrder = 2;
                SysPar.pdsch{idx}.TBSize = 1122;
            elseif idxCfg == 336
                SysPar.pdsch{idx}.targetCodeRate = 6580;
                SysPar.pdsch{idx}.qamModOrder = 4;
                SysPar.pdsch{idx}.TBSize = 4227;                
            elseif idxCfg == 337
                SysPar.pdsch{idx}.targetCodeRate = 8730;
                SysPar.pdsch{idx}.qamModOrder = 6;
                SysPar.pdsch{idx}.TBSize = 15372;
            elseif idxCfg == 338
                SysPar.pdsch{idx}.targetCodeRate = 9480;
                SysPar.pdsch{idx}.qamModOrder = 8;
                SysPar.pdsch{idx}.TBSize = 12297;                
            elseif idxCfg == 339
                SysPar.pdsch{idx}.targetCodeRate = 6580;
                SysPar.pdsch{idx}.qamModOrder = 4;
                SysPar.pdsch{idx}.TBSize = 7685;
            elseif idxCfg == 340
                SysPar.pdsch{idx}.targetCodeRate = 8730;
                SysPar.pdsch{idx}.qamModOrder = 6;
                SysPar.pdsch{idx}.TBSize = 1953;                 
            elseif idxCfg == 347
                SysPar.pdsch{idx}.targetCodeRate = 9480;
                SysPar.pdsch{idx}.qamModOrder = 8;
                SysPar.pdsch{idx}.TBSize = 22026;   
            elseif idxCfg == 348
                SysPar.pdsch{idx}.targetCodeRate = 9480;
                SysPar.pdsch{idx}.qamModOrder = 8;
                SysPar.pdsch{idx}.TBSize = 22026;   
            elseif idxCfg == 349
                SysPar.pdsch{idx}.targetCodeRate = 9480;
                SysPar.pdsch{idx}.qamModOrder = 8;
                SysPar.pdsch{idx}.TBSize = 22026;   
            elseif idxCfg == 350
                SysPar.pdsch{idx}.targetCodeRate = 9480;
                SysPar.pdsch{idx}.qamModOrder = 8;
                SysPar.pdsch{idx}.TBSize = 22026;     
            elseif idxCfg == 351
                SysPar.pdsch{idx}.targetCodeRate = 9480;
                SysPar.pdsch{idx}.qamModOrder = 8;
                SysPar.pdsch{idx}.TBSize = 22026;   
            elseif idxCfg == 352
                SysPar.pdsch{idx}.targetCodeRate = 9480;
                SysPar.pdsch{idx}.qamModOrder = 8;
                SysPar.pdsch{idx}.TBSize = 10247;   
            elseif idxCfg == 353
                SysPar.pdsch{idx}.targetCodeRate = 9480;
                SysPar.pdsch{idx}.qamModOrder = 8;
                SysPar.pdsch{idx}.TBSize = 33822;          
            elseif idxCfg == 355
                SysPar.pdsch{idx}.targetCodeRate = 1930;
                SysPar.pdsch{idx}.qamModOrder = 2;
                SysPar.pdsch{idx}.TBSize = 1697;              
            elseif idxCfg == 359
                SysPar.pdsch{idx}.targetCodeRate = 1930;
                SysPar.pdsch{idx}.qamModOrder = 2;
                SysPar.pdsch{idx}.TBSize = 2242;
            elseif idxCfg == 360
                SysPar.pdsch{idx}.targetCodeRate = 6580;
                SysPar.pdsch{idx}.qamModOrder = 4;
                SysPar.pdsch{idx}.TBSize = 8448;
            elseif idxCfg == 361
                SysPar.pdsch{idx}.targetCodeRate = 8730;
                SysPar.pdsch{idx}.qamModOrder = 6;
                SysPar.pdsch{idx}.TBSize = 22547;    
           elseif idxCfg == 368
                SysPar.pdsch{idx}.targetCodeRate = 9480;
                SysPar.pdsch{idx}.qamModOrder = 8;
                SysPar.pdsch{idx}.TBSize = 6661;   
            elseif idxCfg == 369
                SysPar.pdsch{idx}.targetCodeRate = 9480;
                SysPar.pdsch{idx}.qamModOrder = 8;
                SysPar.pdsch{idx}.TBSize = 6661;   
            elseif idxCfg == 370
                SysPar.pdsch{idx}.targetCodeRate = 9480;
                SysPar.pdsch{idx}.qamModOrder = 8;
                SysPar.pdsch{idx}.TBSize = 6661;   
            elseif idxCfg == 371
                SysPar.pdsch{idx}.targetCodeRate = 9480;
                SysPar.pdsch{idx}.qamModOrder = 8;
                SysPar.pdsch{idx}.TBSize = 6661;     
            elseif idxCfg == 372
                SysPar.pdsch{idx}.targetCodeRate = 9480;
                SysPar.pdsch{idx}.qamModOrder = 8;
                SysPar.pdsch{idx}.TBSize = 6661;   
            elseif idxCfg == 373
                SysPar.pdsch{idx}.targetCodeRate = 9480;
                SysPar.pdsch{idx}.qamModOrder = 8;
                SysPar.pdsch{idx}.TBSize = 3072;                   
            end
            
            if idxCfg == 15
                SysPar.pdsch{idx}.resourceAlloc = 0;
                P = 16; % table 5.1.2.2.1-1 from TS 38.214 [18], configuration 1
                N_RB = 273;
                x = 0;
                switch caseNum
                    case 121
                        x = 0.4;
                        SysPar.pdsch{idx}.powerControlOffset = 8+3;
                    case 126
                        x = 0.6;
                        SysPar.pdsch{idx}.powerControlOffset = 8-3;
                        SysPar.pdsch{idx}.mcsIndex = 6;
                    case 127
                        x = 0.5;
                        SysPar.pdsch{idx}.powerControlOffset = 8-6;
                    otherwise
                        error('caseNum is not supported for idxCfg = 15...\n');
                end
                N_RBG = min(floor(x*(N_RB-3)/P), 0.5*(floor((N_RB-3+mod(3,P)-P)/P)-mod(floor((N_RB-3+mod(3,P)-P)/P),2))+1);
                idxRBG = [floor((N_RB-3+mod(3,P))/P)-1, 1:2:2*(N_RBG-2)+1];
                BWPSize = SysPar.pdsch{idx}.BWPSize;
                BWPStart = SysPar.pdsch{idx}.BWPStart;
                bitMap = rbgIdx2rbMap(idxRBG, P, BWPStart, BWPSize);
                rbBitMap = zeros(1, 36);
                for idxWord = 1:36
                    rbBitMap(idxWord) = bin2dec(num2str(bitMap(idxWord*8:-1:(idxWord-1)*8+1)));
                end
                SysPar.pdsch{idx}.rbBitmap = rbBitMap;
                
            elseif idxCfg == 16
                SysPar.pdsch{idx}.resourceAlloc = 0;
                P = 16; % table 5.1.2.2.1-1 from TS 38.214 [18], configuration 1
                N_RB = 273;
                x = 0;
                switch caseNum
                    case 121
                        x = 0.4;
                    case 126
                        x = 0.6;
                    case 127
                        x = 0.5;
                    otherwise
                        error('caseNum is not supported for PDSCH idxCfg = 16...\n');
                end
                N_RBG = min(floor(x*(N_RB-3)/P), 0.5*(floor((N_RB-3+mod(3,P)-P)/P)-mod(floor((N_RB-3+mod(3,P)-P)/P),2))+1);
                idxRBG = [floor((N_RB-3+mod(3,P))/P)-1, 1:2:2*(N_RBG-2)+1];
                BWPSize = SysPar.pdsch{idx}.BWPSize;
                BWPStart = SysPar.pdsch{idx}.BWPStart;
                bitMap = rbgIdx2rbMap(idxRBG, P, BWPStart, BWPSize);
                bitMap (1:BWPSize) = 1-bitMap (1:BWPSize);
                rbBitMap = zeros(1, 36);
                for idxWord = 1:36
                    rbBitMap(idxWord) = bin2dec(num2str(bitMap(idxWord*8:-1:(idxWord-1)*8+1)));
                end
                SysPar.pdsch{idx}.rbBitmap = rbBitMap;
                
                switch caseNum
                    case 121
                        deboosting = 10*log10((N_RB-3-10^(3/10)*P*N_RBG)/(N_RB-3-P*N_RBG));
                        SysPar.pdsch{idx}.powerControlOffset = 8+round(deboosting); % need FAPIv4 for sub-dB precision
                    case 126
                        boosting = 10*log10((N_RB-3-10^(-3/10)*P*N_RBG)/(N_RB-3-P*N_RBG));
                        SysPar.pdsch{idx}.powerControlOffset = 8+round(boosting); % need FAPIv4 for sub-dB precision
                    case 127
                        boosting = 10*log10((N_RB-3-10^(-6/10)*P*N_RBG)/(N_RB-3-P*N_RBG));
                        SysPar.pdsch{idx}.powerControlOffset = 8+round(boosting); % need FAPIv4 for sub-dB precision
                    otherwise
                        error('caseNum is not supported for PDSCH idxCfg = 16...\n');
                end  
            end

            if ismember(caseNum, [130 : 137])
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 130, 4);
            elseif ismember(caseNum, [140 : 147])
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 140, 4);
            elseif ismember(caseNum, [148 : 171])  % (8C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 148, 8);
            elseif ismember(caseNum, [172 : 195])  % (8C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 172, 8);
            elseif ismember(caseNum, [196 : 219])  % (8C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 196, 8);
            elseif ismember(caseNum, [220 : 243])  % (8C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 220, 8);
            elseif ismember(caseNum, [244 : 291])  % (8C six sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 244, 8);
            elseif ismember(caseNum, [292 : 387])  % (16C six sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 292, 16);
            elseif ismember(caseNum, [388 : 483])  % (16C six sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 388, 16);
            elseif ismember(caseNum, [1000 : 1015])  % (16C 1 set)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1000, 16);
            elseif ismember(caseNum, [1016 : 1111])  % (16C six sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1016, 16);
            elseif ismember(caseNum, [1112 : 1207])  % (16C six sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1112, 16);
            elseif ismember(caseNum, [1208 : 1303])  % (16C six sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1208, 16);
            elseif ismember(caseNum, [1304 : 1351])  % (8C six sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1304, 8);
            elseif ismember(caseNum, [1352 : 1383])  % (16C two sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1352, 16);
            elseif ismember(caseNum, [1384 : 1415])  % (16C two sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1384, 16);
            elseif ismember(caseNum, [1416 : 1479])  % (16C four sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1416, 16);
            elseif ismember(caseNum, [1480 : 1527])  % (16C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1480, 16);
            elseif ismember(caseNum, [1528 : 1559])  % (8C four sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1528, 8);
            elseif ismember(caseNum, [1560 : 1583])  % (8C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1560, 8);
            elseif ismember(caseNum, [1584 : 1615])  % (8C four sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1584, 8);
            elseif ismember(caseNum, [1616 : 1639])  % (8C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1616, 8);
            elseif ismember(caseNum, [1416 : 1479])  % (16C four sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1416, 16);
            elseif ismember(caseNum, [1688 : 1711])  % (8C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1688, 8);
            elseif ismember(caseNum, [1712 : 1743])  % (16C two sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1712, 16);
            elseif ismember(caseNum, [1744 : 1807])  % (16C four sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1744, 16);
            elseif ismember(caseNum, [1808 : 1855])  % (16C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1808, 16);
            elseif ismember(caseNum, [1856 : 1903])  % (16C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1856, 16);
            elseif ismember(caseNum, [1904 : 1927])  % (8C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1904, 8);
            elseif ismember(caseNum, [1928 : 1975])  % (16C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1928, 16);
            elseif ismember(caseNum, [1976 : 1999])  % (8C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 1976, 8);
            elseif ismember(caseNum, [2000 : 2063])  % (16C four sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 2000, 16);
            elseif ismember(caseNum, [2064 : 2111])  % (16C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 2064, 16);
            elseif ismember(caseNum, [2144 : 2167])  % (8C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 2144, 8);    
            elseif ismember(caseNum, [2168 : 2215])  % (16C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 2168, 16);
            elseif ismember(caseNum, [2216 : 2263])  % (16C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 2216, 16);
            elseif ismember(caseNum, [2264 : 2287])  % (8C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 2264, 8); 
            elseif ismember(caseNum, [2288 : 2335])  % (16C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 2288, 16);
            elseif ismember(caseNum, [2336 : 2359])  % (8C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 2336, 8); 
            elseif ismember(caseNum, [2360 : 2407])  % (16C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 2360, 16);
            elseif ismember(caseNum, [2512 : 2591])  % (16C five sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 2512, 16);
            elseif ismember(caseNum, [2592 : 2671])  % (16C five sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 2592, 16);
            elseif ismember(caseNum, [2672 : 2719])  % (16C three sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 2672, 16);
            elseif ismember(caseNum, [2720 : 2767])  % (12C 4 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 2720, 12);
            elseif ismember(caseNum, [2768 : 2803])  % (12C 3 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 2768, 12);
            elseif ismember(caseNum, [2804 : 2827])  % (12C 2 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 2804, 12);
            elseif ismember(caseNum, [2828 : 2875])  % (12C 4 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 2828, 12);
            elseif ismember(caseNum, [2876 : 2911])  % (12C 3 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 2876, 12);
            elseif ismember(caseNum, [2936 : 3079])  % (16C 9 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 2936, 16);
            elseif ismember(caseNum, [3080 : 3159])  % (16C 5 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 3080, 16);
            elseif ismember(caseNum, [3160 : 3399])  % 12C (20 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 3160, 12);
            elseif ismember(caseNum, [3400 : 3719])  % 16C (20 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 3400, 16);
            elseif ismember(caseNum, [3720 : 4039])  % 16C (20 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 3720, 16);
            elseif ismember(caseNum, [4040 : 4439])  % 20C (20 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 4040, 20);
            elseif ismember(caseNum, [4440 : 4759])  % 16C (20 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 4440, 16);
            elseif ismember(caseNum, [4760 : 5159])  % 20C (20 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 4760, 20);
            elseif ismember(caseNum, [5160 : 5479])  % 16C (20 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 5160, 16);
            elseif ismember(caseNum, [5480 : 5879])  % 20C (20 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 5480, 20);
            elseif ismember(caseNum, [6032 : 6351])  % 16C (20 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 6032, 16);
            elseif ismember(caseNum, [6352 : 6751])  % 20C (20 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 6352, 20);
            elseif ismember(caseNum, [5960 : 6031])  % 8C (9 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 5960, 8);
            elseif ismember(caseNum, [6752 : 7071])  % 16C (20 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 6752, 16);
            elseif ismember(caseNum, [7072 : 7471])  % 20C (20 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 7072, 20);
            elseif ismember(caseNum, [7872 : 8271])  % 20C (20 sets)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + mod(caseNum - 7872, 20);
            end % end pdsch dataScramblingId config

            if ismember(cfg{idx}, [879:886,2737:2752,2769:2784,2801:2816,2833:2848,2865:2880,2977:2984,3001:3008,3025:3032,3049:3056,3073:3080]) %32/24DL
                SysPar.pdsch{idx}.nlAbove16 = 1;
            end

            if CellIdxInPatternMap.isKey(caseNum)
                SysPar.pdsch{idx}.dataScramblingId = CFG_PDSCH{idxCfg, 14} + CellIdxInPatternMap(caseNum);
            end
    
            sym0 = SysPar.pdsch{idx}.StartSymbolIndex;
            nSym = SysPar.pdsch{idx}.NrOfSymbols;
            dmrs0 = CFG_PDSCH{idxCfg, 15};
            SysPar.carrier.dmrsTypeAPos = dmrs0;
            maxLen = CFG_PDSCH{idxCfg, 16};
            addPos = CFG_PDSCH{idxCfg, 17};
            DmrsSymbPos = findDmrsSymbPos(sym0, nSym, dmrs0, maxLen, addPos, 'DL', 'typeA');
            if ismember(idxCfg, [88])
                SysPar.pdsch{idx}.DmrsMappingType = 1;
                DmrsSymbPos = findDmrsSymbPos(sym0, nSym, dmrs0, maxLen, addPos, 'DL', 'typeB');
            end
            SysPar.pdsch{idx}.DmrsSymbPos = DmrsSymbPos;
            SysPar.pdsch{idx}.DmrsScramblingId =  CFG_PDSCH{idxCfg, 18};
            SysPar.pdsch{idx}.numDmrsCdmGrpsNoData =  CFG_PDSCH{idxCfg, 19};
            SysPar.pdsch{idx}.portIdx = CFG_PDSCH{idxCfg, 20} + [0:SysPar.pdsch{idx}.nrOfLayers-1];
            SysPar.pdsch{idx}.seed = caseNum;
            
            SysPar.pdsch{idx}.idxUE = idx-1;
            SysPar.pdsch{idx}.idxUeg = CFG_PDSCH{idxCfg, 21};

            % negative test case, overlapping PRBs
            if (caseNum == 501)
                SysPar.pdsch{idx}.idxUeg = idx-1;
            end

            % enable precoding 
            if ismember(caseNum, [1208:7519 7872:8271]) || (enableIdentityPrecoderMap.isKey(caseNum) && enableIdentityPrecoderMap(caseNum) == 1)
                switch SysPar.pdsch{idx}.nrOfLayers
                    case 4
                        SysPar.pdsch{idx}.prcdBf = 12;
                    case 2
                        SysPar.pdsch{idx}.prcdBf = 8;
                    case 1
                        SysPar.pdsch{idx}.prcdBf = 4;
                    otherwise
                        SysPar.pdsch{idx}.prcdBf = 0;
                end

                % precoding hack, using identity matrix
                if ismember(caseNum, [1208:7519 7872:8271]) || (enableIdentityPrecoderMap.isKey(caseNum) && enableIdentityPrecoderMap(caseNum) == 1)
                    SysPar.pdsch{idx}.prcdBf = 14;
                end
            end            
        end

        if ismember(caseNum, [126, 127]) % TM 3.2 and 3.3
            SysPar.pdsch{2}.RNTI = 1;
            SysPar.pdsch{3}.RNTI = 0;
        end

        % enable BetaForce for all perf TVs with/without pdsch (cuphydriver and RU apply it across all slots) 
        if ismember(caseNum, [1208:7519 7872:8271]) || (enableIdentityPrecoderMap.isKey(caseNum) && enableIdentityPrecoderMap(caseNum) == 1)
            SysPar.SimCtrl.oranCompressBetaForce = 1;
        end
        testAlloc.pdsch = length(cfg);
        
        
        % read beam id from TV
        
        for idxUe = 1:length(SysPar.pdsch)
            digBFInterfaces = 1;
            switch SysPar.pdsch{idxUe}.prcdBf
                case 0
                    digBFInterfaces = SysPar.pdsch{idxUe}.nrOfLayers;
                case {1, 2, 5, 6}
                    digBFInterfaces = 2;
                case {3, 4, 7, 8, 9, 10, 11, 12}
                    digBFInterfaces = 4;
                case 14  % precoding hack, 4x4 identity matrix
                    digBFInterfaces = 4;
                otherwise
                    error('prcdBf is not supported ... \n');
            end
            SysPar.pdsch{idxUe}.digBFInterfaces = digBFInterfaces;
            SysPar.pdsch{idxUe}.beamIdx = [1:digBFInterfaces];
        end

        beamIdx_dynamic_offset = beamIdx_pdsch_dynamic(1);
        beamIdx_static_offset = beamIdx_pdsch_static(1);
        if ismember(caseNum, [MIMO_64TR_TC]) % 64TR
            for idxUe = 1:length(SysPar.pdsch)
                digBFInterfaces = SysPar.pdsch{idxUe}.nrOfLayers;
                if (idxUe < length(SysPar.pdsch) && ismember(caseNum, [190, 340:399, 20200:20399, 20736,20737])) || (ismember(caseNum, [296:299,400:419])) || ...
                    (idxUe <= length(SysPar.pdsch)-6 && ismember(caseNum, [421:439 441:459 461])) || ...
                    (idxUe <= length(SysPar.pdsch)-5 && ismember(caseNum, [420 440 460])) || ...
                    (ismember(caseNum, 20400:20519)) || ...
                    (ismember(caseNum, 20520:20559))  || ... % column B, nrSim pattern 90623
                    (ismember(caseNum, 20560:20599))  || ... % column G, nrSim pattern 90624; also column H, nrSim pattern 90626
                    (ismember(caseNum, 20600:20639))  || ... % column B, nrSim pattern 90625, w/ modcomp
                    (ismember(caseNum, [20640:20679, 20740:20749]))  || ... % column G 40 MHz, nrSim pattern 90627, 20740-20749 with different cell ID
                    (ismember(caseNum, 20690:20729))  || ... % column G 100 MHz, nrSim pattern 90629, 64 UEs per TTI
                    (ismember(caseNum, 20750:20789))  || ... % column I, nrSim pattern 90638
                    (ismember(caseNum, [20790:20829, 20840:20860])) || ... % Ph4 column B, nrSim pattern 90640,  % Heterogeneous UE pattern
                    (ismember(caseNum, 20830))        || ... % nvbug 5368243
                    (ismember(caseNum, 20831))        || ... % 32DL
                    (ismember(caseNum, 20870:20909))  || ... % 25-3 column B, nrSim pattern 90644/90645/90646/90647
                    (ismember(caseNum, 20910:20949))  || ... % 25-3 column G, 64 UEs per TTI, nrSim pattern 90649/90650/90651/90652
                    (ismember(caseNum, 20950:20989))  || ... % 25-3 column E, nrSim pattern 90648/90657 32DL 
                    (ismember(caseNum, 20990:21029))  || ... % Ph4 column B, srsPrgSize = 2/4 and bfwPrgSize = 16, nrSim pattern 90653/90654, 100 MHz
                    (ismember(caseNum, 21030:21069))  || ... % 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 100 MHz heavy, nrSim pattern 90655
                    (ismember(caseNum, 21070:21109))  || ... % 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 100 MHz light, nrSim pattern 90656
                    (ismember(caseNum, 21110:21149))  || ... % 64TR MU-MIMO realistic traffic with Center/Middle/Edge,  90 MHz heavy, nrSim pattern 90658
                    (ismember(caseNum, 21150:21189))  || ... % 64TR MU-MIMO realistic traffic with Center/Middle/Edge,  90 MHz light, nrSim pattern 90659
                    (ismember(caseNum, 21190:21229))  || ... % 64TR MU-MIMO realistic traffic with Center/Middle/Edge,  60 MHz heavy, nrSim pattern 90660
                    (ismember(caseNum, 21230:21269))  || ... % 64TR MU-MIMO realistic traffic with Center/Middle/Edge, 60 MHz light, nrSim pattern 90661
                    (ismember(caseNum, 21270:21309))     ... % 25-3 column D, nrSim pattern 90662 24DL 
                    SysPar.pdsch{idxUe}.digBFInterfaces = 0; % RTW
                    SysPar.pdsch{idxUe}.beamIdx = beamIdx_dynamic_offset + [1:digBFInterfaces];
                    beamIdx_dynamic_offset = beamIdx_dynamic_offset + digBFInterfaces;               
                else
                    SysPar.pdsch{idxUe}.digBFInterfaces = digBFInterfaces; % non-RTW
                    SysPar.pdsch{idxUe}.beamIdx = beamIdx_static_offset + [1:digBFInterfaces];
                    beamIdx_static_offset = beamIdx_static_offset + digBFInterfaces;                
                end
            end
        end
        
        if pdschDynamicBfMap.isKey(caseNum) && pdschDynamicBfMap(caseNum) == 1 % 64TR PDSCH
            for idxUe = 1:length(SysPar.pdsch)
                digBFInterfaces = SysPar.pdsch{idxUe}.nrOfLayers;
                SysPar.pdsch{idxUe}.digBFInterfaces = 0; % RTW
                SysPar.pdsch{idxUe}.beamIdx = beamIdx_dynamic_offset + [1:digBFInterfaces];
                beamIdx_dynamic_offset = beamIdx_dynamic_offset + digBFInterfaces;               
            end
        end
        
        % disable precoding for PDSCH CFG for CCH
        for idxUe = 1:length(cfg)
            idxCfg = find(cellfun(@(x) isequal(x,cfg{idxUe}), CFG_PDSCH(:,1)));
            if ismember(CFG_PDSCH{idxCfg,1}, [847, 887, 957, 999, 1069, 1111, 1181, 1223])
                digBFInterfaces = SysPar.pdsch{idxUe}.nrOfLayers;
                SysPar.pdsch{idxUe}.digBFInterfaces = digBFInterfaces; % non-RTW
                SysPar.pdsch{idxUe}.beamIdx = beamIdx_static_offset + [1:digBFInterfaces];
                beamIdx_static_offset = beamIdx_static_offset + digBFInterfaces;
            end
        end
        
        % config CSI-RS
        cfg = CFG{idxSet, 7};
        for idx = 1:length(cfg)
            idxCfg = find(cellfun(@(x) isequal(x,cfg{idx}), CFG_CSIRS(:,1)));
            SysPar.csirs{idx} = cfgCsirs;
            % cfg#  CSI Row CDM Density RB0  nRB  sym0  sym1   nID  FreqDomain
            SysPar.csirs{idx}.CSIType = CFG_CSIRS{idxCfg, 2};
            SysPar.csirs{idx}.Row = CFG_CSIRS{idxCfg, 3};
            SysPar.csirs{idx}.CDMType = CFG_CSIRS{idxCfg, 4};
            SysPar.csirs{idx}.FreqDensity = CFG_CSIRS{idxCfg, 5};
            SysPar.csirs{idx}.StartRB = CFG_CSIRS{idxCfg, 6};
            SysPar.csirs{idx}.NrOfRBs = CFG_CSIRS{idxCfg, 7};
            if VARY_PRB_NUM > 0  % in study of sweeping PRB numbers, overwrite pdsch PRB numbers
                if ismember(caseNum, [3080:3111, 3128:3143, 3464:3575, 3592:3703, 4584:4695, 4712:4823])  % having csi in this slot
                    if SysPar.csirs{idx}.CSIType == csirsType.TRS
                        TRS_start_PRB_idx = PRB_num_w_csi * 6;
                        SysPar.csirs{idx}.StartRB = TRS_start_PRB_idx;
                    end
                end
            end
            SysPar.csirs{idx}.SymbL0 = CFG_CSIRS{idxCfg, 8};
            SysPar.csirs{idx}.SymbL1 = CFG_CSIRS{idxCfg, 9};
            SysPar.csirs{idx}.ScrambId = CFG_CSIRS{idxCfg, 10};
            SysPar.csirs{idx}.FreqDomain = cell2mat(CFG_CSIRS{idxCfg, 11});
            SysPar.csirs{idx}.BWPSize = SysPar.carrier.N_grid_size_mu; 
            SysPar.csirs{idx}.idxUE = idx-1;
        end
        testAlloc.csirs = length(cfg);
        
        % default method, may be overwritten in next if-else block
        for idxUe = 1:length(SysPar.csirs)
            digBFInterfaces = 1;
            switch SysPar.csirs{idxUe}.prcdBf
                case 0
                    digBFInterfaces = SysPar.carrier.N_FhPort_DL;
                case {1, 2, 5, 6}
                    digBFInterfaces = 2;
                case {3, 4, 7, 8, 9, 10, 11, 12}
                    digBFInterfaces = 4;
                otherwise
                    error('prcdBf is not supported ... \n');
            end
            SysPar.csirs{idxUe}.digBFInterfaces = digBFInterfaces;
            SysPar.csirs{idxUe}.beamIdx = [1:digBFInterfaces];
        end

        if ismember(caseNum, [MIMO_64TR_TC]) || CsiBeamIdxMap.isKey(caseNum) || TrsBeamIdxMap.isKey(caseNum)
            beamIdx_static_offset = beamIdx_csirs_static(1);
            [row2nPort, ~] = getCsirsConfig();
            % For NZP with CsiBeamIdxMap: one TV can have multiple NZP PDUs; map uses non-overlapping segments (e.g. PDU1 32 beams, PDU2 8 beams -> 40 entries)
            fullCsiBeamIdx = [];
            if CsiBeamIdxMap.isKey(caseNum)
                fullCsiBeamIdx = CsiBeamIdxMap(caseNum);
            end
            nzpReadOffset = 1;  % 1-based index into fullCsiBeamIdx for next NZP PDU
            for idxUe = 1:length(SysPar.csirs)
                digBFInterfaces = row2nPort(SysPar.csirs{idxUe}.Row);
                if (SysPar.csirs{idxUe}.CSIType == csirsType.TRS && TrsBeamIdxMap.isKey(caseNum))  % by mapping method
                    if digBFInterfaces ~= length(TrsBeamIdxMap(caseNum))
                        error('trs beams mapping error: digBFInterfaces = %d but TrsBeamIdxMap(%d) = %s \n', digBFInterfaces, caseNum, mat2str(TrsBeamIdxMap(caseNum)));
                    end
                    SysPar.csirs{idxUe}.beamIdx = beamIdx_static_offset + TrsBeamIdxMap(caseNum);    
                elseif (SysPar.csirs{idxUe}.CSIType == csirsType.NZP && CsiBeamIdxMap.isKey(caseNum))  % by mapping method
                    if nzpReadOffset + digBFInterfaces - 1 > length(fullCsiBeamIdx)
                        error('csi rs NZP beams mapping error: need %d beams for PDU but CsiBeamIdxMap(%d) has %d (used %d so far). \n', ...
                            digBFInterfaces, caseNum, length(fullCsiBeamIdx), nzpReadOffset - 1);
                    end
                    SysPar.csirs{idxUe}.beamIdx = beamIdx_static_offset + fullCsiBeamIdx(nzpReadOffset : nzpReadOffset + digBFInterfaces - 1);
                    nzpReadOffset = nzpReadOffset + digBFInterfaces;
                elseif (SysPar.csirs{idxUe}.CSIType == csirsType.ZP && CsiZpBeamIdxMap.isKey(caseNum))
                    if digBFInterfaces ~= length(CsiZpBeamIdxMap(caseNum))
                        error('csi rs ZP beams mapping error: digBFInterfaces = %d but CsiZpBeamIdxMap(%d) = %s \n', digBFInterfaces, caseNum, mat2str(CsiZpBeamIdxMap(caseNum)));
                    end
                    SysPar.csirs{idxUe}.beamIdx = beamIdx_static_offset + CsiZpBeamIdxMap(caseNum);  
                else % same across slots and cells
                    SysPar.csirs{idxUe}.beamIdx = beamIdx_static_offset + [1:digBFInterfaces];
                    beamIdx_static_offset = beamIdx_static_offset + digBFInterfaces;
                end
                if(SysPar.csirs{idxUe}.CSIType == csirsType.ZP)
                    SysPar.csirs{idxUe}.digBFInterfaces = 0;
                else
                    SysPar.csirs{idxUe}.digBFInterfaces = digBFInterfaces;
                end
            end
            % Sanity: total NZP beams assigned should match CsiBeamIdxMap length
            if CsiBeamIdxMap.isKey(caseNum) && (nzpReadOffset - 1) ~= length(fullCsiBeamIdx)
                error('csi rs NZP beams mapping error: CsiBeamIdxMap(%d) has %d entries but NZP PDUs use %d. \n', caseNum, length(fullCsiBeamIdx), nzpReadOffset - 1);
            end
        end
        
        % HARQ TC, 64TR
        if ismember(caseNum, [150, MIMO_64TR_TC])
            cfg = {41, 42, 43, 44};
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
                SysPar.prach{idx}.allSubframes = 1;
                
                SysPar.prach{idx}.idxUE = idx-1; % bug to be fixed
                SysPar.prach{1}.msg1_FDM = length(cfg); % only on {1} based on 0513
            end
            testAlloc.prach = length(cfg);
        
            % read beam id from TV
            if ismember(caseNum, [1980:1991, 2032:2047])
                for idxUe = 1:length(SysPar.prach)
                    SysPar.prach{idxUe}.digBFInterfaces = 1;
                    SysPar.prach{idxUe}.beamIdx = idxUe;
                end
            elseif ismember(caseNum, [2004:2015, 2064:2079])
                for idxUe = 1:length(SysPar.prach)
                    SysPar.prach{idxUe}.digBFInterfaces = 1;
                    SysPar.prach{idxUe}.beamIdx = 4 + idxUe;
                end
            else % default
                for idxUe = 1:length(SysPar.prach)
                    digBFInterfaces = SysPar.carrier.N_FhPort_DL;
                    SysPar.prach{idxUe}.digBFInterfaces = digBFInterfaces;
                    SysPar.prach{idxUe}.beamIdx = [1:digBFInterfaces];
                end
            end
                
        elseif caseNum >= 1304   % cell config to tell the max value of numPrachFdOccasion accross different slot. 
                                 % TVs larger than 1304 all have a max of 4 PRACH in the pattern
            SysPar.prach{1}.configurationIndex = 158;
            SysPar.prach{1}.restrictedSet = 0;
            SysPar.prach{1}.rootSequenceIndex = 0;
            SysPar.prach{1}.zeroCorrelationZone = 5;
            
            SysPar.prach{2} = SysPar.prach{1};
            SysPar.prach{3} = SysPar.prach{1};
            SysPar.prach{4} = SysPar.prach{1};
            
            SysPar.prach{1}.msg1_FDM = 4; 
        end        
        
        SysPar.SimCtrl.N_UE = max([2, testAlloc.pdsch, testAlloc.pdcch, testAlloc.csirs]);
        SysPar.testAlloc = testAlloc;
        
        if caseNum == 103
            SysPar.SimCtrl.enable_codebook_BF = 1;
        end

        % disable for saving running time
        if strcmp(caseSet, 'full') && ~ismember(caseNum, TcFapiOnly) % only enable UE Rx and print check det for full set with cuPHY TVs
            SysPar.SimCtrl.enableUeRx = 1;
            SysPar.SimCtrl.printCheckDet = 1;
        end

        % update Nre_max which is used to calculate beta value in BFP
        SysPar.SimCtrl.oranComp.Nre_max = SysPar.carrier.N_grid_size_mu*12;
        

        
        % save SysPar into Cfg_<TC#>.yaml config file
        if SysPar.SimCtrl.genTV.genYamlCfg
            if caseNum < 1000
                fileName = sprintf('Cfg_%04d.yaml', caseNum);
            else
                fileName = sprintf('Cfg_DLMIX_%04d.yaml', caseNum);
            end
            WriteYaml(fileName, SysPar);
        end
                
        if strcmp(compTvMode, 'genCfg')
            fileName = sprintf('cfg-%04d.yaml', caseNum);
            WriteYaml(fileName, SysPar);
            continue;
        end                
        
        % HARQ TC, 64TR XGT
        if ismember(caseNum, [150:189, 280, MIMO_64TR_TC])
            if ismember(caseNum, [150:189])
                SysPar.SimCtrl.N_UE = 24;
            end
            SysPar.SimCtrl.genTV.forceSlotIdxFlag = 1;
            SysPar.Chan{1}.SNR = 100;
            SysPar.SimCtrl.timeDomainSim = 1;
            
             if ismember(caseNum, [190,192:194,280,281,296,12232:13951,20560:20831,20840:20861,20870:21309])  % nrSim pattern 90624~90636,90638~90662 and perf pattern 69,69a,69b,69c,69d,69e,71,73,75,77,79,81,83,85,87,89,91
                SysPar.SimCtrl.genTV.fhMsgMode = 2; % enable modulation compression
                if enableCsiRs32PortsMap.isKey(caseNum) && enableCsiRs32PortsMap(caseNum) == 1
                    SysPar.SimCtrl.nPort_enable_csirs_compression = 32;
                end

                if ismember(caseNum, [20831, 20950:20989, 21270:21309, 13547:13696, 13802:13951])
                    SysPar.SimCtrl.enable_dynamic_BF = 1;
                    SysPar.carrier.Nant_gNB = 64;
                    SysPar.carrier.N_FhPort_DL = 32;
                    SysPar.carrier.N_FhPort_UL = 16;
                end
            end

            if strcmp(caseSet, 'harq') % enable HARQ
                
                SysPar.SimCtrl.puschHARQ.EnableAutoHARQ = 1;
                SysPar.SimCtrl.puschHARQ.MaxTransmissions = 4;
                SysPar.SimCtrl.puschHARQ.MinTransmissions = 1;

                SysPar.SimCtrl.N_frame = 2;
                SysPar.SimCtrl.N_slot_run = 0;
                %                 SysPar.SimCtrl.enableUeRx = 1;
                SysPar.SimCtrl.printCheckDet = 1;
                fileName = sprintf('cfg_HARQ_DLMIX_s%d.yaml', caseNum-150);
                WriteYaml(fileName, SysPar);
                continue;

            else % disable HARQ

                nPdsch = length(SysPar.pdsch);
                for idxPdsch = 1:nPdsch
                    if (ismember(caseNum, [150:189])) % only using a lower mcsIndex for 150:189
                        mcsIndex = SysPar.pdsch{idxPdsch}.mcsIndex;
                        % Need to reduce codeRate becasuse some REs are reserved for CSIRS
                        if ismember(mcsIndex, [27, 31])
                            SysPar.pdsch{idxPdsch}.mcsIndex = 23;
                        elseif ismember(mcsIndex, [19, 30])
                            SysPar.pdsch{idxPdsch}.mcsIndex = 17;
                        elseif mcsIndex == 29
                            SysPar.pdsch{idxPdsch}.mcsIndex = 10;
                        elseif mcsIndex == 28
                            SysPar.pdsch{idxPdsch}.mcsIndex = 1;
                        end
                    end
                    SysPar.pdsch{idxPdsch}.rvIndex = 0;
                end

                SysPar.SimCtrl.N_frame = 1;
                SysPar.SimCtrl.N_slot_run = 1;
            end
                

        end
        
        % overwrite default BFPforCuphy from the default value 14 to 9 for all MIMO 64TR TCs
        if (ismember(caseNum, MIMO_64TR_TC))
            SysPar.SimCtrl.BFPforCuphy = 9;
        end

        % negative case, invalid prgSize
        if ismember(caseNum, [20584])
            SysPar.SimCtrl.negTV.enable = 1;
            SysPar.SimCtrl.negTV.pduFieldName = {'numPRGs', 'prgSize'};
            SysPar.SimCtrl.negTV.pduFieldValue = {1, 273};
        elseif ismember(caseNum, [20585])
            SysPar.SimCtrl.negTV.enable = 1;
            SysPar.SimCtrl.negTV.pduFieldName = {'numPRGs', 'prgSize'};
            SysPar.SimCtrl.negTV.pduFieldValue = {273, 1};
        elseif ismember(caseNum, [20594])
            SysPar.SimCtrl.negTV.enable = 1;
            SysPar.SimCtrl.negTV.pduFieldName = {'numPRGs', 'prgSize'};
            SysPar.SimCtrl.negTV.pduFieldValue = {69, 4};
        elseif ismember(caseNum, [20595])
            SysPar.SimCtrl.negTV.enable = 1;
            SysPar.SimCtrl.negTV.pduFieldName = {'numPRGs', 'prgSize'};
            SysPar.SimCtrl.negTV.pduFieldValue = {35, 8};
        end
                
        [SysPar, UE, gNB] = nrSimulator(SysPar);


        fprintf('%3d     %2d      %2d      %2d     %2d     %2d    %2d \n',...
            CFG{idxSet, 1}, CFG{idxSet, 2}, CFG{idxSet, 3}, length(CFG{idxSet, 4}), length(CFG{idxSet, 5}),...
            length(CFG{idxSet, 6}), length(CFG{idxSet, 7}));

        if SysPar.SimCtrl.plotFigure.tfGrid 
            figure; mesh(abs(gNB.Phy.tx.Xtf(:,:,1)));  view(2); pause(1);
        end
    end
end
fprintf('------------------------------------------------\n');
fprintf('Total cuPHY TV generated = %d, total FAPI TV generated = %d, det-FAIL = %d, \n\n', nCuphyTV, nFapiTV, detErr);
toc; 
fprintf('\n');

end

function largestTvNum = getLargestTvNum(CFG, threshold)
    allTvNums = cell2mat(CFG(:, 1));
    largestTvNum = [max(allTvNums(allTvNums < threshold)) max(allTvNums(allTvNums >= threshold))];
    fprintf('The largest TV number in the current DLMIX CFG < %d is: %d \n', threshold, largestTvNum(1));
    fprintf('The largest TV number in the current DLMIX CFG >= %d is: %d \n', threshold, largestTvNum(2));
end
