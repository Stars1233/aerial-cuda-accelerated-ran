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

% Pattern names and TV ranges are stored in perf_pattern/perf_pattern_helper.yaml and included pattern files.
all_patterns = perfPatternTvYaml('genperf_pattern_names', 'full');
compact_patterns = perfPatternTvYaml('genperf_pattern_names', 'compact');

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
        % Support .1-.6 (letters a-f) per perf_pattern/perf_pattern_helper.yaml (e.g. 59.6 -> 59f).
        if decimal_part >= 1 && decimal_part <= 6
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
    pattern_num = perfPatternTvYaml('pattern_number', pattern_str);
    base_tv_only = perfPatternTvYaml('base_tv_only', pattern_str);
    if base_tv_only
        fprintf('----------------------------------------\n');
        fprintf('Pattern %.10g is a base TV only pattern; skipping direct generation.\n', pattern_num);
        continue;
    end

    if perfPatternTvYaml('poc2_pattern', pattern_str)
        lp_cmd = sprintf('genLP_POC2(%.10g)', pattern_num);
    else
        lp_cmd = '';
    end
    [dlmix_cmd, ulmix_cmd, bfw_cmd] = perfPatternTvYaml('genperf_commands', pattern_str);
    % Store commands
    dlmix_cmds{end+1} = dlmix_cmd;
    ulmix_cmds{end+1} = ulmix_cmd;
    bfw_cmds{end+1} = bfw_cmd;
    % Display commands
    fprintf('----------------------------------------\n');
    fprintf('Pattern %s:\n', pattern_str);
    if gen_lp_flag && ~isempty(lp_cmd)
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
        if gen_lp_flag && ~isempty(lp_cmd)
            eval(lp_cmd);
        end
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
