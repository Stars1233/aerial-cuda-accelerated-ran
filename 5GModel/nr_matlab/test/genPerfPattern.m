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

function [perfResults, dlmix_cmds, ulmix_cmds, bfw_cmds] = genPerfPattern(caseSet, channelSet, exec_cmd)
% GENPERFPATTERN Generates TV commands for a given pattern set and channel set
%   caseSet: 'full', 'compact', 'selected', or numeric array (e.g., [59, 59.3, 60, ...])
%            'selected' is the same as 'compact'
%   channelSet: 'ulmix', 'allUL', 'dlmix', 'allDL', 'launchPatternFile', 'bfw', or 'allChannels'
%              'allUL'/ 'allDL' will generate BFW while 'ulmix'/ 'dlmix' will not
%   exec_cmd: Optional parameter to control command execution (default: 0)
%             If 0, only prints commands without execution
%             If non-zero, executes the commands
%   Returns: perfResults struct with fields for DLMIX, ULMIX, BFW results and totals
%            dlmix_cmds, ulmix_cmds, bfw_cmds as cell arrays of strings

tic;

if nargin < 1
    caseSet = 'compact';
    channelSet = 'allChannels';
    exec_cmd = 1;
elseif nargin < 2
    channelSet = 'allChannels';
    exec_cmd = 1;
elseif nargin < 3
    exec_cmd = 1;
end

validChannels = {'ulmix', 'allUL', 'dlmix', 'allDL', 'launchPatternFile', 'bfw', 'allChannels'};
if ischar(channelSet) || isstring(channelSet)
    channelSet = {char(channelSet)};
end
for i = 1:length(channelSet)
    if ~ismember(channelSet{i}, validChannels)
        error('Invalid channelSet: %s. Must be one of: %s', channelSet{i}, strjoin(validChannels, ', '));
    end
end
if ismember('allChannels', channelSet)
    channelSet = {'ulmix', 'allUL', 'dlmix', 'allDL', 'launchPatternFile', 'bfw'};
end

% translate channelSet to flag for each channel
gen_ulmix_flag = ismember('ulmix', channelSet) || ismember('allUL', channelSet);
gen_dlmix_flag = ismember('dlmix', channelSet) || ismember('allDL', channelSet);
gen_bfw_flag = ismember('bfw', channelSet) || ismember('allChannels', channelSet) || ismember('allUL', channelSet) || ismember('allDL', channelSet);  % ULBFW and DLBFW are generated in the same command
gen_lp_flag = ismember('launchPatternFile', channelSet) || ismember('allChannels', channelSet);

dlmix_cmds = {};
ulmix_cmds = {};
bfw_cmds = {};

% Initialize accumulators for totals
perfResults.dlmix.nTC = 0;
perfResults.dlmix.err = 0;
perfResults.dlmix.nCuphyTV = 0;
perfResults.dlmix.nFapiTV = 0;
perfResults.dlmix.detErr = 0;
perfResults.ulmix.nTC = 0;
perfResults.ulmix.err = 0;
perfResults.ulmix.nCuphyTV = 0;
perfResults.ulmix.nFapiTV = 0;
perfResults.ulmix.detErr = 0;
perfResults.bfw.nTC = 0;
perfResults.bfw.err = 0;
perfResults.bfw.nTV = 0;
perfResults.bfw.detErr = 0;

% Store per-pattern results if needed
perfResults.dlmix.pattern = {};
perfResults.ulmix.pattern = {};
perfResults.bfw.pattern = {};

% Define all available patterns
all_patterns = { ...
    '59', '59a', '59b', '59c', '59d', '59e', '59f', ...
    '60', '60a', '60b', '60c', '60d', '60e', ...
    '61', '62c', '63c', ...
    '65a', '65b', '65c', '65d', ...
    '66a', '66b', '66c', '66d', ...
    '67', '67a', '67b', '67c', '67d', '67e', ...
    '69', '69a', '69b', '69c', '69d', '69e', ...
    '71', ...
    '73', ...
    '75', ...
    '77', ...
    '79', '79a', '79b', ...
    '81a', '81b', '81c', '81d', ...
    '83a', '83b', '83c', '83d', ...
    '85', ...
    '87', ...
    '89', ...
    '91', ...
    '101', '101a', ...
    '102', '102a'
};

compact_patterns = {'59c', '60c', '69'};

% Determine which patterns to generate
if iscell(caseSet)
    caseSet = caseSet{1};
end

if ischar(caseSet) || isstring(caseSet)
    caseSet = char(caseSet);
    if strcmpi(caseSet, 'full')
        patterns = all_patterns;
    elseif strcmpi(caseSet, 'compact') || strcmpi(caseSet, 'selected')
        patterns = compact_patterns;
    else
        error('Unknown caseSet string: %s', caseSet);
    end
elseif isnumeric(caseSet)
    % Convert numeric patterns to string representation
    patterns = {};
    for k = 1:numel(caseSet)
        base_num = floor(caseSet(k));
        decimal_part = round((caseSet(k) - base_num) * 10);
        if decimal_part >= 1 && decimal_part <= 5
            patterns{end+1} = [num2str(base_num), char(96 + decimal_part)];
        else
            patterns{end+1} = num2str(caseSet(k));
        end
    end
else
    error('caseSet must be a string or numeric array');
end

for i = 1:length(patterns)
    pattern_str = patterns{i};
    % Generate the corresponding pattern number for LP command
    % (reverse the string->number logic)
    if length(pattern_str) > 2 && isstrprop(pattern_str(end), 'alpha')
        base_num = str2double(pattern_str(1:end-1));
        decimal_part = double(pattern_str(end)) - 96;
        pattern_num = base_num + decimal_part/10;
        lp_cmd = sprintf('genLP_POC2(%.1f)', pattern_num);
    else
        pattern_num = str2double(pattern_str);
        lp_cmd = sprintf('genLP_POC2(%d)', pattern_num);
    end
    dlmix_cmd = '';
    ulmix_cmd = '';
    bfw_cmd = '';
    switch pattern_str
        case '59'
            dlmix_cmd = 'testCompGenTV_dlmix([7072:7471])';
            ulmix_cmd = 'testCompGenTV_ulmix([3040:3119])';
        case '59a'
            dlmix_cmd = 'testCompGenTV_dlmix([7072:7471])';
            ulmix_cmd = 'testCompGenTV_ulmix([3200:3279])';
        case '59b'
            dlmix_cmd = 'testCompGenTV_dlmix([7072:7471])';
            ulmix_cmd = 'testCompGenTV_ulmix([3760:3839])';
        case '59c'
            dlmix_cmd = 'testCompGenTV_dlmix([9472:10271])';
            ulmix_cmd = 'testCompGenTV_ulmix([4471:4630])';
        case '59d'
            dlmix_cmd = 'testCompGenTV_dlmix([7072:7471])';
            ulmix_cmd = 'testCompGenTV_ulmix([3520:3599])';
        case '59e'
            dlmix_cmd = 'testCompGenTV_dlmix([9472:10271])';
            ulmix_cmd = 'testCompGenTV_ulmix([4631:4790])';
        case '59f'  % negative TC
            dlmix_cmd = 'testCompGenTV_dlmix([9472:10271])';
            ulmix_cmd = 'testCompGenTV_ulmix([3840 4471:4630])';
        case '60'
            dlmix_cmd = 'testCompGenTV_dlmix([10452:11251])';
            ulmix_cmd = 'testCompGenTV_ulmix([3600:3759])';
        case '60a'
            dlmix_cmd = 'testCompGenTV_dlmix([10452:11251])';
            ulmix_cmd = 'testCompGenTV_ulmix([3280:3359])';
        case '60b'
            dlmix_cmd = 'testCompGenTV_dlmix([10452:11251])';
            ulmix_cmd = 'testCompGenTV_ulmix([4911:5070])';
        case '60c'
            dlmix_cmd = 'testCompGenTV_dlmix([10452:11251])';
            ulmix_cmd = 'testCompGenTV_ulmix([5071:5230])';
        case '60d'
            dlmix_cmd = 'testCompGenTV_dlmix([10452:11251])';
            ulmix_cmd = 'testCompGenTV_ulmix([5231:5390])';
        case '60e'
            dlmix_cmd = 'testCompGenTV_dlmix([10452:11251])';
            ulmix_cmd = 'testCompGenTV_ulmix([4220:4299])';
        case '61'
            dlmix_cmd = 'testCompGenTV_dlmix([7872:8271])';
            ulmix_cmd = 'testCompGenTV_ulmix([3360:3439])';
        case '62c'
            dlmix_cmd = 'testCompGenTV_dlmix([9472:10271])';
            ulmix_cmd = 'testCompGenTV_ulmix([4471:4630, 4040:4079])';
        case '63c'
            dlmix_cmd = 'testCompGenTV_dlmix([10452:11251])';
            ulmix_cmd = 'testCompGenTV_ulmix([5071:5230, 5391:5430])';
        case '65a'
            dlmix_cmd = 'testCompGenTV_dlmix([11432:12231])';
            ulmix_cmd = 'testCompGenTV_ulmix([5971:6010])';
        case '65b'
            dlmix_cmd = 'testCompGenTV_dlmix([11432:12231])';
            ulmix_cmd = 'testCompGenTV_ulmix([6011:6050])';
        case '65c'
            dlmix_cmd = 'testCompGenTV_dlmix([11432:12231])';
            ulmix_cmd = 'testCompGenTV_ulmix([6051:6210])';
        case '65d'
            dlmix_cmd = 'testCompGenTV_dlmix([11432:12231])';
            ulmix_cmd = 'testCompGenTV_ulmix([6211:6370])';
        case '66a'
            dlmix_cmd = 'testCompGenTV_dlmix([11252:11341])';
            ulmix_cmd = 'testCompGenTV_ulmix([833 5431:5445])';
            bfw_cmd = 'testCompGenTV_bfw([9292])';
        case '66b'
            dlmix_cmd = 'testCompGenTV_dlmix([11252:11341])';
            ulmix_cmd = 'testCompGenTV_ulmix([833 5446:5460])';
            bfw_cmd = 'testCompGenTV_bfw([9292])';
        case '66c'
            dlmix_cmd = 'testCompGenTV_dlmix([11252:11341])';
            ulmix_cmd = 'testCompGenTV_ulmix([833 5461:5580])';
            bfw_cmd = 'testCompGenTV_bfw([9292])';
        case '66d'
            dlmix_cmd = 'testCompGenTV_dlmix([11252:11341])';
            ulmix_cmd = 'testCompGenTV_ulmix([833 5581:5700])';
            bfw_cmd = 'testCompGenTV_bfw([9292])';
        case '67'
            dlmix_cmd = 'testCompGenTV_dlmix([11342:11431])';
            ulmix_cmd = 'testCompGenTV_ulmix([833 4791:4910])';
            bfw_cmd = 'testCompGenTV_bfw([9292])';
        case '67a'
            dlmix_cmd = 'testCompGenTV_dlmix([11342:11431])';
            ulmix_cmd = 'testCompGenTV_ulmix([833 5701:5715])';
            bfw_cmd = 'testCompGenTV_bfw([9292])';
        case '67b'
            dlmix_cmd = 'testCompGenTV_dlmix([11342:11431])';
            ulmix_cmd = 'testCompGenTV_ulmix([833 5716:5730])';
            bfw_cmd = 'testCompGenTV_bfw([9292])';
        case '67c'
            dlmix_cmd = 'testCompGenTV_dlmix([11342:11431])';
            ulmix_cmd = 'testCompGenTV_ulmix([833 5731:5850])';
            bfw_cmd = 'testCompGenTV_bfw([9292])';
        case '67d'
            dlmix_cmd = 'testCompGenTV_dlmix([11342:11431])';
            ulmix_cmd = 'testCompGenTV_ulmix([833 5851:5970])';
            bfw_cmd = 'testCompGenTV_bfw([9292])';
        case '67e'  % negative TC
            dlmix_cmd = 'testCompGenTV_dlmix([11342:11431])';
            ulmix_cmd = 'testCompGenTV_ulmix([833 6996 5731:5850])';
            bfw_cmd = 'testCompGenTV_bfw([9292])';
        case '69'
            dlmix_cmd = 'testCompGenTV_dlmix([12232:12441])';
            ulmix_cmd = 'testCompGenTV_ulmix([6621:6695])';
            bfw_cmd = 'testCompGenTV_bfw([9355])';
        case '69a'
            dlmix_cmd = 'testCompGenTV_dlmix([12232:12441])';
            ulmix_cmd = 'testCompGenTV_ulmix([6696:6770])';
            bfw_cmd = 'testCompGenTV_bfw([9355])';
        case '69b'
            dlmix_cmd = 'testCompGenTV_dlmix([12232:12441])';
            ulmix_cmd = 'testCompGenTV_ulmix([6771:6845])';
            bfw_cmd = 'testCompGenTV_bfw([9355])';
        case '69c'
            dlmix_cmd = 'testCompGenTV_dlmix([12232:12441])';
            ulmix_cmd = 'testCompGenTV_ulmix([6371:6445])';
            bfw_cmd = 'testCompGenTV_bfw([9355])';
        case '69d'
            dlmix_cmd = 'testCompGenTV_dlmix([12232:12441])';
            ulmix_cmd = 'testCompGenTV_ulmix([6846:6920])';
            bfw_cmd = 'testCompGenTV_bfw([9355])';
        case '69e'
            dlmix_cmd = 'testCompGenTV_dlmix([12232:12441])';
            ulmix_cmd = 'testCompGenTV_ulmix([7260:7364])';
            bfw_cmd = 'testCompGenTV_bfw([9355])';
        case '71'
            dlmix_cmd = 'testCompGenTV_dlmix([12232:12441])';
            ulmix_cmd = 'testCompGenTV_ulmix([6446:6520])';
            bfw_cmd = 'testCompGenTV_bfw([9365])';
        case '73'
            dlmix_cmd = 'testCompGenTV_dlmix([12442:12721], ''genTV'')';
            ulmix_cmd = 'testCompGenTV_ulmix([6521:6620], ''genTV'')';
            bfw_cmd = 'testCompGenTV_bfw([9369])';
        case '75'
            dlmix_cmd = 'testCompGenTV_dlmix([12722:12931])';
            ulmix_cmd = 'testCompGenTV_ulmix([6921:6995])';
            bfw_cmd = 'testCompGenTV_bfw([9381])';
        case '77'
            dlmix_cmd = 'testCompGenTV_dlmix([12932:13066])';
            ulmix_cmd = 'testCompGenTV_ulmix([7050:7154])';
            bfw_cmd = 'testCompGenTV_bfw([9385])';
        case '79'
            dlmix_cmd = 'testCompGenTV_dlmix([13067:13216])';
            ulmix_cmd = 'testCompGenTV_ulmix([7155:7259])';
            bfw_cmd = 'testCompGenTV_bfw([9391])';
        case '79a'
            dlmix_cmd = 'testCompGenTV_dlmix([13067:13216])';
            ulmix_cmd = 'testCompGenTV_ulmix([7953:8057])';
            bfw_cmd = 'testCompGenTV_bfw([9501])';
        case '79b'
            dlmix_cmd = 'testCompGenTV_dlmix([13067:13216])';
            ulmix_cmd = 'testCompGenTV_ulmix([8058:8162])';
            bfw_cmd = 'testCompGenTV_bfw([9507])';
        case '81a'
            dlmix_cmd = 'testCompGenTV_dlmix([13217:13366])';
            ulmix_cmd = 'testCompGenTV_ulmix([7365:7469])';
            bfw_cmd = 'testCompGenTV_bfw([9401])';
        case '81b'
            dlmix_cmd = 'testCompGenTV_dlmix([13217:13366])';
            ulmix_cmd = 'testCompGenTV_ulmix([7470:7574])';
            bfw_cmd = 'testCompGenTV_bfw([9407])';
        case '81c'
            dlmix_cmd = 'testCompGenTV_dlmix([13217:13321 13337:13381])';
            ulmix_cmd = 'testCompGenTV_ulmix([7365:7469])';
            bfw_cmd = 'testCompGenTV_bfw([9401])';
        case '81d'
            dlmix_cmd = 'testCompGenTV_dlmix([13217:13321 13337:13381])';
            ulmix_cmd = 'testCompGenTV_ulmix([7470:7574])';
            bfw_cmd = 'testCompGenTV_bfw([9407])';
        case '83a'
            dlmix_cmd = 'testCompGenTV_dlmix([13382:13531])';
            ulmix_cmd = 'testCompGenTV_ulmix([7575:7679])';
            bfw_cmd = 'testCompGenTV_bfw([9413])';
        case '83b'
            dlmix_cmd = 'testCompGenTV_dlmix([13382:13531])';
            ulmix_cmd = 'testCompGenTV_ulmix([7680:7784])';
            bfw_cmd = 'testCompGenTV_bfw([9421])';
        case '83c'
            dlmix_cmd = 'testCompGenTV_dlmix([13382:13486 13502:13546])';
            ulmix_cmd = 'testCompGenTV_ulmix([7575:7679])';
            bfw_cmd = 'testCompGenTV_bfw([9413])';
        case '83d'
            dlmix_cmd = 'testCompGenTV_dlmix([13382:13486 13502:13546])';
            ulmix_cmd = 'testCompGenTV_ulmix([7680:7784])';
            bfw_cmd = 'testCompGenTV_bfw([9421])';
        case '85'
            dlmix_cmd = 'testCompGenTV_dlmix([13547:13696])';
            ulmix_cmd = 'testCompGenTV_ulmix([7785:7889])';
            bfw_cmd = 'testCompGenTV_bfw([9461])';
        case '87'
            dlmix_cmd = 'testCompGenTV_dlmix(setdiff([13067:13216 13697:13711], 13172:13186))';
            ulmix_cmd = 'testCompGenTV_ulmix([7155:7259])';
            bfw_cmd = 'testCompGenTV_bfw([9391])';
        case '89'
            dlmix_cmd = 'testCompGenTV_dlmix([13712:13801])';
            ulmix_cmd = 'testCompGenTV_ulmix([7890:7952])';
            bfw_cmd = 'testCompGenTV_bfw([9449 9455 9469 9475 9481 9487])';
        case '91'
            dlmix_cmd = 'testCompGenTV_dlmix([13802:13951])';
            ulmix_cmd = 'testCompGenTV_ulmix([8163:8267])';
            bfw_cmd = 'testCompGenTV_bfw([9461])';
        case '101'
            dlmix_cmd = 'testCompGenTV_dlmix([7472:7495])';
            ulmix_cmd = 'testCompGenTV_ulmix([4375:4470])';
        case '101a'
            dlmix_cmd = 'testCompGenTV_dlmix([7472:7495])';
            ulmix_cmd = 'testCompGenTV_ulmix([8268:8363])';
        case '102'
            dlmix_cmd = 'testCompGenTV_dlmix([7496:7519])';
            ulmix_cmd = 'testCompGenTV_ulmix([8364:8459])';
        case '102a'
            dlmix_cmd = 'testCompGenTV_dlmix([7496:7519])';
            ulmix_cmd = 'testCompGenTV_ulmix([8460:8555])';
        otherwise
            error('Pattern number not found: %s', pattern_str);
    end
    % Store commands
    dlmix_cmds{end+1} = dlmix_cmd;
    ulmix_cmds{end+1} = ulmix_cmd;
    bfw_cmds{end+1} = bfw_cmd;
    % Display commands
    fprintf('----------------------------------------\n');
    fprintf('Pattern %s:\n', pattern_str);
    if gen_lp_flag
        fprintf('LP Command: %s\n', lp_cmd);
    end
    if gen_dlmix_flag
        fprintf('DL MIX Command: %s\n', dlmix_cmd);
    end
    if gen_ulmix_flag
        fprintf('UL MIX Command: %s\n', ulmix_cmd);
    end
    if gen_bfw_flag && ~isempty(bfw_cmd)
        fprintf('BFW Command: %s\n', bfw_cmd);
    end
    % Execute commands if exec_cmd is non-zero, only for selected channels
    if exec_cmd
        eval(lp_cmd);
        if gen_dlmix_flag && ~isempty(dlmix_cmd)
            [nTC_dlmix, err_dlmix, nCuphyTV_dlmix, nFapiTV_dlmix, detErr_dlmix] = eval(dlmix_cmd);
            perfResults.dlmix.nTC = perfResults.dlmix.nTC + nTC_dlmix;
            perfResults.dlmix.err = perfResults.dlmix.err + err_dlmix;
            perfResults.dlmix.nCuphyTV = perfResults.dlmix.nCuphyTV + nCuphyTV_dlmix;
            perfResults.dlmix.nFapiTV = perfResults.dlmix.nFapiTV + nFapiTV_dlmix;
            perfResults.dlmix.detErr = perfResults.dlmix.detErr + detErr_dlmix;
            perfResults.dlmix.pattern{end+1} = struct('pattern', pattern_str, 'nTC', nTC_dlmix, 'err', err_dlmix, 'nCuphyTV', nCuphyTV_dlmix, 'nFapiTV', nFapiTV_dlmix, 'detErr', detErr_dlmix);
        end
        if gen_ulmix_flag && ~isempty(ulmix_cmd)
            [nTC_ulmix, err_ulmix, nCuphyTV_ulmix, nFapiTV_ulmix, detErr_ulmix] = eval(ulmix_cmd);
            perfResults.ulmix.nTC = perfResults.ulmix.nTC + nTC_ulmix;
            perfResults.ulmix.err = perfResults.ulmix.err + err_ulmix;
            perfResults.ulmix.nCuphyTV = perfResults.ulmix.nCuphyTV + nCuphyTV_ulmix;
            perfResults.ulmix.nFapiTV = perfResults.ulmix.nFapiTV + nFapiTV_ulmix;
            perfResults.ulmix.detErr = perfResults.ulmix.detErr + detErr_ulmix;
            perfResults.ulmix.pattern{end+1} = struct('pattern', pattern_str, 'nTC', nTC_ulmix, 'err', err_ulmix, 'nCuphyTV', nCuphyTV_ulmix, 'nFapiTV', nFapiTV_ulmix, 'detErr', detErr_ulmix);
        end
        if gen_bfw_flag && ~isempty(bfw_cmd)
            [nTC_bfw, err_bfw, nTV_bfw, detErr_bfw] = eval(bfw_cmd);
            perfResults.bfw.nTC = perfResults.bfw.nTC + nTC_bfw;
            perfResults.bfw.err = perfResults.bfw.err + err_bfw;
            perfResults.bfw.nTV = perfResults.bfw.nTV + nTV_bfw;
            perfResults.bfw.detErr = perfResults.bfw.detErr + detErr_bfw;
            perfResults.bfw.pattern{end+1} = struct('pattern', pattern_str, 'nTC', nTC_bfw, 'err', err_bfw, 'nTV', nTV_bfw, 'detErr', detErr_bfw);
        end
    end
end

% Print summary
fprintf('--------------------------------------------\n\n');
fprintf('Total patterns = %d\n', length(patterns));
if (exec_cmd)
    fprintf('Total DLMIX: nTC=%d, err=%d, nCuphyTV=%d, nFapiTV=%d, detErr=%d\n', perfResults.dlmix.nTC, perfResults.dlmix.err, perfResults.dlmix.nCuphyTV, perfResults.dlmix.nFapiTV, perfResults.dlmix.detErr);
    fprintf('Total ULMIX: nTC=%d, err=%d, nCuphyTV=%d, nFapiTV=%d, detErr=%d\n', perfResults.ulmix.nTC, perfResults.ulmix.err, perfResults.ulmix.nCuphyTV, perfResults.ulmix.nFapiTV, perfResults.ulmix.detErr);
    fprintf('Total BFW:   nTC=%d, err=%d, nTV=%d, detErr=%d\n', perfResults.bfw.nTC, perfResults.bfw.err, perfResults.bfw.nTV, perfResults.bfw.detErr);
end
toc;
fprintf('--------------------------------------------\n\n\n');
end 