% SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

function varargout = perfPatternTvYaml(action, varargin)
%PERFPATTERNTVYAML Load shared PERF-pattern TV numbering metadata.

cfg = loadPerfPatternTvConfig();

switch lower(char(action))
    case 'tv_table'
        if numel(varargin) ~= 1
            error('perfPatternTvYaml(''tv_table'', channel) expects one channel argument.');
        end
        channel = lower(char(varargin{1}));
        if ~isfield(cfg, channel)
            error('Channel "%s" is not defined in perf_pattern/perf_pattern_helper.yaml.', channel);
        end
        channelCfg = cfg.(channel);
        perfPatternTable = expandTvTableFromPatterns(cfg.patterns, upper(channel));
        compactGpuTvs = reshape(yamlNumeric(channelCfg.compact_gpu_tvs), 1, []);
        if isfield(channelCfg, 'full_gpu_tvs')
            fullGpuTvs = reshape(yamlNumeric(channelCfg.full_gpu_tvs), 1, []);
        else
            fullGpuTvs = [];
        end
        tvRanges.selected_high = yamlScalar(channelCfg.tc_ranges.selected_high);
        tvRanges.full_high = yamlScalar(channelCfg.tc_ranges.full_high);
        varargout = {perfPatternTable, compactGpuTvs, tvRanges, fullGpuTvs};

    case 'compact_case_set'
        compactCaseSet = reshape(yamlNumeric(cfg.poc2.compact_patterns), 1, []);
        varargout = {compactCaseSet};

    case 'apply_poc2_cfg'
        if numel(varargin) ~= 1
            error('perfPatternTvYaml(''apply_poc2_cfg'', CFG) expects the POC2 CFG cell array.');
        end
        [poc2Cfg, fullCellCount, compactCellCount] = applyPoc2Config(varargin{1}, cfg.patterns);
        varargout = {poc2Cfg, fullCellCount, compactCellCount};

    case 'poc2_cfg'
        if numel(varargin) ~= 1
            error('perfPatternTvYaml(''poc2_cfg'', channelMap) expects one channel map argument.');
        end
        [poc2Cfg, fullCellCount, compactCellCount, sSlotCfg, bfwCfg, bfwSrsBindCfg, overrideCfg] = buildPoc2Config(cfg.patterns, varargin{1});
        varargout = {poc2Cfg, fullCellCount, compactCellCount, sSlotCfg, bfwCfg, bfwSrsBindCfg, overrideCfg};

    case 'genperf_pattern_names'
        if numel(varargin) > 1
            error('perfPatternTvYaml(''genperf_pattern_names'', caseSet) expects zero or one caseSet argument.');
        end
        if isempty(varargin)
            caseSet = 'full';
        else
            caseSet = lower(char(varargin{1}));
        end
        varargout = {genPerfPatternNames(cfg, caseSet)};

    case 'genperf_commands'
        if numel(varargin) ~= 1
            error('perfPatternTvYaml(''genperf_commands'', patternId) expects one pattern id argument.');
        end
        [dlmixCmd, ulmixCmd, bfwCmd] = genPerfCommands(cfg, char(varargin{1}));
        varargout = {dlmixCmd, ulmixCmd, bfwCmd};

    case 'base_tv_only'
        if numel(varargin) ~= 1
            error('perfPatternTvYaml(''base_tv_only'', patternId) expects one pattern id argument.');
        end
        patternEntry = genPerfPatternEntry(cfg.patterns, char(varargin{1}));
        varargout = {isBaseTvOnlyPattern(patternEntry)};

    case 'pattern_number'
        if numel(varargin) ~= 1
            error('perfPatternTvYaml(''pattern_number'', patternId) expects one pattern id argument.');
        end
        patternEntry = genPerfPatternEntry(cfg.patterns, char(varargin{1}));
        varargout = {yamlScalar(patternEntry.pattern)};

    case 'pattern_id'
        if numel(varargin) ~= 1
            error('perfPatternTvYaml(''pattern_id'', patternId) expects one pattern id argument.');
        end
        patternEntry = genPerfPatternEntry(cfg.patterns, char(varargin{1}));
        varargout = {yamlText(patternEntry.id)};

    case 'poc2_pattern'
        if numel(varargin) ~= 1
            error('perfPatternTvYaml(''poc2_pattern'', patternId) expects one pattern id argument.');
        end
        patternEntry = genPerfPatternEntry(cfg.patterns, char(varargin{1}));
        varargout = {isfield(patternEntry, 'pattern_type')};

    otherwise
        error('Unsupported perfPatternTvYaml action "%s".', char(action));
end

end

function cfg = loadPerfPatternTvConfig()

persistent cachedCfg
if ~isempty(cachedCfg)
    cfg = cachedCfg;
    return;
end

thisDir = fileparts(mfilename('fullpath'));
cfgFile = fullfile(thisDir, 'perf_pattern', 'perf_pattern_helper.yaml');
if exist(cfgFile, 'file') ~= 2
    error('Shared PERF pattern TV YAML not found: %s', cfgFile);
end

cachedCfg = ReadYaml(cfgFile, 1, 1);
cachedCfg = loadPatternFileIncludes(cachedCfg, fileparts(cfgFile));
if isfield(cachedCfg, 'patterns')
    validatePatternCellCounts(cachedCfg.patterns);
end
cfg = cachedCfg;

end

function cfg = loadPatternFileIncludes(cfg, cfgDir)

if ~isfield(cfg, 'pattern_files')
    return;
end

patternFiles = yamlTextList(cfg.pattern_files);
patterns = {};
if isfield(cfg, 'patterns')
    patterns = yamlEntries(cfg.patterns);
end

for idxFile = 1:numel(patternFiles)
    patternFile = fullfile(cfgDir, patternFiles{idxFile});
    if exist(patternFile, 'file') ~= 2
        error('PERF pattern definition YAML not found: %s', patternFile);
    end
    patternCfg = ReadYaml(patternFile, 1, 1);
    if ~isfield(patternCfg, 'patterns')
        error('PERF pattern definition YAML must define patterns: %s', patternFile);
    end
    patterns = [patterns, yamlEntries(patternCfg.patterns)];
end

cfg.patterns = sortPatternEntries(reshape(patterns, 1, []));

end

function patterns = sortPatternEntries(patterns)

if isempty(patterns)
    return;
end

patternNums = zeros(1, numel(patterns));
for idxPattern = 1:numel(patterns)
    if isfield(patterns{idxPattern}, 'pattern')
        patternNums(idxPattern) = yamlScalar(patterns{idxPattern}.pattern);
    else
        patternNums(idxPattern) = inf;
    end
end
[~, order] = sort(patternNums);
patterns = patterns(order);

end

function validatePatternCellCounts(patterns)

patternEntries = yamlEntries(patterns);
for idxEntry = 1:numel(patternEntries)
    patternEntry = patternEntries{idxEntry};
    pattern = yamlScalar(patternEntry.pattern);
    isBaseTvOnly = isBaseTvOnlyPattern(patternEntry);
    if isBaseTvOnly && isfield(patternEntry, 'pattern_type')
        error('Pattern %.10g cannot set base_tv_only with pattern_type.', pattern);
    end
    if isBaseTvOnly && isfield(patternEntry, 'genperf') && yamlBool(patternEntry.genperf)
        error('Pattern %.10g cannot set base_tv_only with genperf.', pattern);
    end
    if ~isBaseTvOnly && ~isfield(patternEntry, 'pattern_type') && ...
            (~isfield(patternEntry, 'genperf') || ~yamlBool(patternEntry.genperf))
        error('Pattern %.10g must define pattern_type, set genperf: true, or set base_tv_only: true.', pattern);
    end
    if isfield(patternEntry, 'config_cells')
        configCells = yamlRow(patternEntry.config_cells);
        validateConfigCells(pattern, 'pattern', configCells);
    else
        error('Pattern %.10g must define config_cells.', pattern);
    end
    if isfield(patternEntry, 'full_cells') && isfield(patternEntry, 'compact_cells')
        fullCells = yamlRow(patternEntry.full_cells);
        validateFullCompactCells(pattern, 'pattern', fullCells, yamlRow(patternEntry.compact_cells));
        validateFullConfigCells(pattern, 'pattern', fullCells, configCells);
    end
    validateRangeCellCounts(patternEntry, configCells);
end

end

function tf = isBaseTvOnlyPattern(patternEntry)

tf = isfield(patternEntry, 'base_tv_only') && yamlBool(patternEntry.base_tv_only);

end

function validateRangeCellCounts(patternEntry, configCells)

if ~isfield(patternEntry, 'ranges')
    return;
end

pattern = yamlScalar(patternEntry.pattern);
sides = fieldnames(patternEntry.ranges);
for idxSide = 1:numel(sides)
    side = sides{idxSide};
    sideCfg = patternEntry.ranges.(side);
    [rangeEntries, configCells, fullCells, compactCells] = sideRangeEntriesAndCells(sideCfg, patternEntry);
    validateConfigCells(pattern, sprintf('ranges.%s', side), configCells);
    if ~isempty(fullCells) && ~isempty(compactCells)
        validateFullCompactCells(pattern, sprintf('ranges.%s', side), fullCells, compactCells);
        validateFullConfigCells(pattern, sprintf('ranges.%s', side), fullCells, configCells);
    end

    for idxRange = 1:numel(rangeEntries)
        [startTv, endTv, ~, rangeConfigCells, rangeFullCells, rangeCompactCells] = ...
            tvTableRangeInfo(rangeEntries{idxRange}, configCells, fullCells, compactCells);
        if isempty(rangeConfigCells) || isempty(rangeFullCells) || isempty(rangeCompactCells)
            continue;
        end
        validateConfigCells(pattern, ...
            sprintf('ranges.%s [%d, %d]', side, startTv, endTv), ...
            rangeConfigCells);
        validateFullCompactCells(pattern, ...
            sprintf('ranges.%s [%d, %d]', side, startTv, endTv), ...
            rangeFullCells, rangeCompactCells);
        validateFullConfigCells(pattern, ...
            sprintf('ranges.%s [%d, %d]', side, startTv, endTv), ...
            rangeFullCells, rangeConfigCells);
    end
end

end

function validateConfigCells(pattern, context, configCells)

if isempty(configCells)
    return;
end
if any(configCells <= 0)
    error('Pattern %.10g %s config_cells must be positive.', pattern, context);
end

end

function validateFullCompactCells(pattern, context, fullCells, compactCells)

if isempty(fullCells) || isempty(compactCells)
    return;
end
if numel(compactCells) ~= numel(fullCells)
    error('Pattern %.10g %s compact_cells must have the same width as full_cells.', pattern, context);
end
if any(fullCells <= 0)
    error('Pattern %.10g %s full_cells must be positive.', pattern, context);
end
if any(compactCells < 0)
    error('Pattern %.10g %s compact_cells cannot be negative.', pattern, context);
end
if any(compactCells > fullCells)
    error('Pattern %.10g %s compact_cells cannot exceed full_cells.', pattern, context);
end

end

function validateFullConfigCells(pattern, context, fullCells, configCells)

if isempty(fullCells) || isempty(configCells)
    return;
end
if numel(configCells) == numel(fullCells)
    exceedsConfig = fullCells > configCells;
else
    exceedsConfig = fullCells > max(configCells);
end
if any(exceedsConfig)
    error('Pattern %.10g %s full_cells=%s cannot exceed config_cells=%s.', ...
        pattern, context, mat2str(fullCells), mat2str(configCells));
end

end


function rows = expandTvTableFromPatterns(patterns, side)

patternEntries = yamlEntries(patterns);
rows = [];
for idxEntry = 1:numel(patternEntries)
    rows = [rows; expandPatternSideTvRows(patternEntries{idxEntry}, side)];
end
rows = uniqueTvTableRows(rows);
if isempty(rows)
    error('No PERF TV table rows can be derived for side %s.', side);
end

end

function rows = expandPatternSideTvRows(patternEntry, side)

rows = [];
if ~isfield(patternEntry, 'ranges')
    return;
end

sideCfg = getStructField(patternEntry.ranges, side);
if isempty(sideCfg)
    return;
end

pattern = yamlScalar(patternEntry.pattern);
[rangeEntries, configCells, fullCells, compactCells] = sideRangeEntriesAndCells(sideCfg, patternEntry);
if isempty(configCells)
    return;
end

for idxRange = 1:numel(rangeEntries)
    rangeEntry = rangeEntries{idxRange};
    [startTv, endTv, includeInTable, rangeConfigCells, rangeFullCells, rangeCompactCells] = tvTableRangeInfo(rangeEntry, configCells, fullCells, compactCells);
    if ~includeInTable
        continue;
    end
    if isempty(rangeCompactCells)
        continue;
    end
    if isempty(rangeFullCells)
        rangeFullCells = rangeConfigCells;
    end
    % Keep rows with compact_cells == 0 in the table: a 0 compact count keeps
    % the pattern out of the compact set (repelem(..., 0) -> empty), but the
    % row is still needed so the full-set cuPHY/FAPI split uses the config_cells
    % stride and only the 1st cell of each config block gets a cuPHY TV.
    if endTv < startTv
        error('Pattern %.10g %s has an invalid TV range [%d, %d].', pattern, side, startTv, endTv);
    end
    if mod(endTv - startTv + 1, rangeConfigCells) ~= 0
        error('Pattern %.10g %s TV range [%d, %d] is not divisible by config_cells=%d.', pattern, side, startTv, endTv, rangeConfigCells);
    end
    if rangeFullCells > rangeConfigCells
        error('Pattern %.10g %s full_cells cannot exceed config_cells in range [%d, %d].', pattern, side, startTv, endTv);
    end
    if rangeCompactCells > rangeFullCells
        error('Pattern %.10g %s compact_cells cannot exceed full_cells in range [%d, %d].', pattern, side, startTv, endTv);
    end
    rows = [rows; pattern, startTv, endTv, rangeConfigCells, rangeCompactCells, rangeFullCells];
end

end

function rows = uniqueTvTableRows(rows)

if isempty(rows)
    return;
end

[~, uniqueIdx] = unique(rows(:, 2:6), 'rows', 'stable');
rows = rows(sort(uniqueIdx), :);

end

function [rangeEntries, configCells, fullCells, compactCells] = sideRangeEntriesAndCells(sideCfg, patternEntry)

if isstruct(sideCfg) && isfield(sideCfg, 'tvs')
    rangeEntries = yamlCommandRangeEntries(sideCfg.tvs);
    configCells = sideCellsOrDefault(sideCfg, patternEntry, 'config_cells');
    fullCells = sideCellsOrDefault(sideCfg, patternEntry, 'full_cells');
    compactCells = sideCellsOrDefault(sideCfg, patternEntry, 'compact_cells');
elseif isstruct(sideCfg)
    rangeEntries = yamlRangeEntries(sideCfg);
    configCells = sideCellsOrDefault(sideCfg, patternEntry, 'config_cells');
    fullCells = sideCellsOrDefault(sideCfg, patternEntry, 'full_cells');
    compactCells = sideCellsOrDefault(sideCfg, patternEntry, 'compact_cells');
else
    rangeEntries = yamlCommandRangeEntries(sideCfg);
    configCells = [];
    fullCells = [];
    compactCells = [];
end

if isempty(compactCells) && ~isempty(fullCells)
    compactCells = fullCells;
end

end

function cells = sideCellsOrDefault(sideCfg, patternEntry, fieldName)

cells = [];
if isfield(patternEntry, fieldName)
    cells = yamlScalar(patternEntry.(fieldName));
end

if isstruct(sideCfg) && isfield(sideCfg, fieldName)
    cells = yamlScalar(sideCfg.(fieldName));
end

end

function [startTv, endTv, includeInTable, configCells, fullCells, compactCells] = tvTableRangeInfo(rangeEntry, defaultConfigCells, defaultFullCells, defaultCompactCells)

includeInTable = true;
configCells = defaultConfigCells;
fullCells = defaultFullCells;
compactCells = defaultCompactCells;
hasConfigOverride = false;

if isstruct(rangeEntry)
    if isfield(rangeEntry, 'include_in_table')
        includeInTable = yamlBool(rangeEntry.include_in_table);
    end
    if isfield(rangeEntry, 'full_cells')
        fullCells = yamlScalar(rangeEntry.full_cells);
    end
    if isfield(rangeEntry, 'config_cells')
        configCells = yamlScalar(rangeEntry.config_cells);
        hasConfigOverride = true;
    end
    if isfield(rangeEntry, 'compact_cells')
        compactCells = yamlScalar(rangeEntry.compact_cells);
    end
    entryValue = tvRangeEntryValue(rangeEntry);
    if ~isempty(entryValue)
        rangeRow = yamlRow(entryValue);
        startTv = rangeRow(1);
        endTv = rangeRow(end);
    elseif isfield(rangeEntry, 'tv')
        startTv = yamlScalar(rangeEntry.tv);
        endTv = startTv;
    elseif isfield(rangeEntry, 'range')
        rangeRow = yamlRow(rangeEntry.range);
        startTv = rangeRow(1);
        endTv = rangeRow(end);
    elseif isfield(rangeEntry, 'start_tv') && isfield(rangeEntry, 'end_tv')
        startTv = yamlScalar(rangeEntry.start_tv);
        endTv = yamlScalar(rangeEntry.end_tv);
    else
        error('Structured TV range entries must define tv, range, or start_tv/end_tv.');
    end
else
    rangeRow = yamlRow(rangeEntry);
    if numel(rangeRow) == 1
        startTv = rangeRow(1);
        endTv = rangeRow(1);
    elseif numel(rangeRow) == 2
        startTv = rangeRow(1);
        endTv = rangeRow(2);
    else
        error('TV range entries must be a scalar TV or [start,end] pair.');
    end
end

if startTv == endTv && ~hasConfigOverride
    configCells = 1;
    if ~isempty(fullCells)
        fullCells = min(fullCells, 1);
    end
    if ~isempty(compactCells)
        compactCells = min(compactCells, 1);
    end
end

end

function names = genPerfPatternNames(cfg, caseSet)

switch caseSet
    case 'full'
        patternEntries = genPerfPatternEntries(cfg.patterns);
        names = cell(1, numel(patternEntries));
        for idxEntry = 1:numel(patternEntries)
            names{idxEntry} = yamlText(patternEntries{idxEntry}.id);
        end
    case {'compact', 'selected'}
        names = yamlTextList(cfg.genperf.compact_patterns);
    otherwise
        error('Unknown genPerfPattern caseSet string: %s', caseSet);
end

end

function [dlmixCmd, ulmixCmd, bfwCmd] = genPerfCommands(cfg, patternId)

patternEntry = genPerfPatternEntry(cfg.patterns, patternId);
dlmixCmd = genPerfSideCommand(patternEntry, 'DLMIX', 'testCompGenTV_dlmix');
ulmixCmd = genPerfSideCommand(patternEntry, 'ULMIX', 'testCompGenTV_ulmix');
bfwCmd = genPerfSideCommand(patternEntry, 'BFW', 'testCompGenTV_bfw');

end

function patternEntriesOut = genPerfPatternEntries(patterns)

patternEntries = yamlEntries(patterns);
patternEntriesOut = {};
for idxEntry = 1:numel(patternEntries)
    patternEntry = patternEntries{idxEntry};
    if isfield(patternEntry, 'genperf') && yamlBool(patternEntry.genperf)
        patternEntriesOut{end + 1} = patternEntry;
    end
end

end

function patternEntry = genPerfPatternEntry(patterns, patternId)

patternEntries = yamlEntries(patterns);
% Match on the exact id string first (covers letter ids like 59a and decimal
% ids like 52.1).
for idxEntry = 1:numel(patternEntries)
    patternEntry = patternEntries{idxEntry};
    if strcmpi(yamlText(patternEntry.id), patternId)
        return;
    end
end
% Fall back to matching the numeric pattern value. genPerfPattern maps a
% numeric input like 52.1 to the string "52a" via the .1-.6 -> a-f rule, but
% POC2-only patterns keep a decimal id ('52.1'). Resolving both forms here lets
% genPerfPattern still emit the launch-pattern command for such patterns.
patternNum = patternIdToNumber(patternId);
if ~isnan(patternNum)
    for idxEntry = 1:numel(patternEntries)
        patternEntry = patternEntries{idxEntry};
        if isfield(patternEntry, 'pattern') && ...
                abs(yamlScalar(patternEntry.pattern) - patternNum) < 1e-6
            return;
        end
    end
end
error('Pattern id "%s" is not defined in patterns loaded by perf_pattern/perf_pattern_helper.yaml.', patternId);

end

function patternNum = patternIdToNumber(patternId)
% Convert a genperf pattern id string to its numeric pattern value: letter
% suffixes ('59a' -> 59.1 .. '59f' -> 59.6), decimal ids ('52.1' -> 52.1), and
% plain integers ('60' -> 60). Returns NaN if it cannot be parsed.
patternId = char(patternId);
patternNum = NaN;
if isempty(patternId)
    return;
end
lastCh = lower(patternId(end));
if numel(patternId) > 1 && lastCh >= 'a' && lastCh <= 'f'
    baseNum = str2double(patternId(1:end - 1));
    decimalPart = double(lastCh) - double('a') + 1;
    if ~isnan(baseNum)
        patternNum = baseNum + decimalPart / 10;
    end
else
    patternNum = str2double(patternId);
end

end

function cmd = genPerfSideCommand(patternEntry, side, funcName)

% POC2-only patterns (e.g. CA pattern 52.1) carry no ranges block and generate
% no DLMIX/ULMIX/BFW TVs; return an empty command so the caller can still emit
% the launch-pattern command.
if ~isfield(patternEntry, 'ranges')
    cmd = '';
    return;
end
rangeValue = getStructField(patternEntry.ranges, side);
rangeExpr = genPerfCommandExpr(rangeValue);
if isempty(rangeExpr)
    cmd = '';
    return;
end

argText = '';
if isfield(patternEntry, 'command_args')
    argValue = getStructField(patternEntry.command_args, side);
    if ~isempty(argValue)
        argText = yamlText(argValue);
    end
end

if isempty(argText)
    cmd = sprintf('%s(%s)', funcName, rangeExpr);
else
    cmd = sprintf('%s(%s, ''%s'')', funcName, rangeExpr, argText);
end

end

function expr = genPerfCommandExpr(value)

if isstruct(value) && isfield(value, 'command_expr')
    expr = yamlText(value.command_expr);
else
    expr = genPerfRangeExpr(value);
end

end

function expr = genPerfRangeExpr(value)

rangeEntries = yamlCommandRangeEntries(sideRangeTvs(value));
if isempty(rangeEntries)
    expr = '';
    return;
end

parts = cell(1, numel(rangeEntries));
for idxEntry = 1:numel(rangeEntries)
    rangeRow = yamlRow(tvRangeEntryValue(rangeEntries{idxEntry}));
    if numel(rangeRow) == 1
        parts{idxEntry} = sprintf('%.10g', rangeRow(1));
    elseif numel(rangeRow) == 2
        if rangeRow(1) == rangeRow(2)
            parts{idxEntry} = sprintf('%.10g', rangeRow(1));
        else
            parts{idxEntry} = sprintf('%.10g:%.10g', rangeRow(1), rangeRow(2));
        end
    else
        error('genperf range entries must be scalar TVs or [start,end] pairs.');
    end
end
expr = sprintf('[%s]', strjoin(parts, ' '));

end

function value = tvRangeEntryValue(rangeEntry)

if isstruct(rangeEntry)
    value = getStructField(rangeEntry, 'main');
    if isempty(value)
        value = getStructField(rangeEntry, 'additional');
    end
    if isempty(value)
        value = getStructField(rangeEntry, 'tv');
    end
    if isempty(value)
        value = getStructField(rangeEntry, 'range');
    end
else
    value = rangeEntry;
end

end

function tv = poc2TvFromRange(patternEntry, side)

tv = [];
if ~isfield(patternEntry, 'ranges')
    return;
end

sideCfg = getStructField(patternEntry.ranges, side);
if isempty(sideCfg)
    return;
end

rangeEntries = yamlCommandRangeEntries(sideRangeTvs(sideCfg));
if isempty(rangeEntries)
    return;
end

mainEntry = [];
for idxEntry = 1:numel(rangeEntries)
    if isstruct(rangeEntries{idxEntry}) && ~isempty(getStructField(rangeEntries{idxEntry}, 'main'))
        mainEntry = rangeEntries{idxEntry};
        break;
    end
end
if isempty(mainEntry)
    mainEntry = rangeEntries{1};
end

rangeRow = yamlRow(tvRangeEntryValue(mainEntry));
if ~isempty(rangeRow)
    tv = rangeRow(1);
end

end

function value = sideRangeTvs(value)

if isstruct(value) && isfield(value, 'tvs')
    value = value.tvs;
end

end

function entries = yamlCommandRangeEntries(value)

if isempty(value)
    entries = {};
elseif iscell(value)
    entries = reshape(value, 1, []);
elseif isnumeric(value)
    if isscalar(value)
        entries = {value};
    elseif ~isvector(value) && size(value, 2) == 2
        entries = num2cell(value, 2);
    elseif isvector(value) && numel(value) == 2
        entries = {reshape(value, 1, [])};
    else
        entries = num2cell(reshape(value, 1, []));
    end
else
    error('Expected genperf ranges to be scalar TVs or [start,end] pairs.');
end

end

function [poc2Cfg, fullCellCount, compactCellCount] = applyPoc2Config(poc2Cfg, patterns)

fullCellCount = poc2Cfg(:, 2);
compactCellCount = poc2Cfg(:, 2);
patternEntries = poc2PatternEntries(patterns);

for idxEntry = 1:numel(patternEntries)
    patternEntry = patternEntries{idxEntry};
    pattern = yamlScalar(patternEntry.pattern);
    cfgIdx = find(cellfun(@(x) isnumeric(x) && isequal(x, pattern), poc2Cfg(:, 1)), 1);
    if isempty(cfgIdx)
        error('Pattern %.10g is present in perf_pattern/perf_pattern_helper.yaml but not in genLP_POC2 CFG.', pattern);
    end

    configCells = poc2ConfigCells(patternEntry);
    fullCells = poc2GeneratedCells(patternEntry, 'full_cells', configCells);
    compactCells = poc2GeneratedCells(patternEntry, 'compact_cells', fullCells);

    poc2Cfg{cfgIdx, 2} = configCells;
    dlTv = poc2TvValue(patternEntry, 'DLMIX', 'dl_tv');
    if isfield(patternEntry, 'pattern_type') && strcmp(string(yamlText(patternEntry.pattern_type)), "CA")
        poc2Cfg{cfgIdx, 4} = num2cell(dlTv);
    else
        poc2Cfg{cfgIdx, 4} = {dlTv};
    end
    if isfield(patternEntry, 'dl_tv_delta')
        poc2Cfg{cfgIdx, 5} = yamlRow(patternEntry.dl_tv_delta);
    end
    poc2Cfg{cfgIdx, 6} = poc2TvValue(patternEntry, 'ULMIX', 'ul_tv');
    fullCellCount{cfgIdx} = fullCells;
    compactCellCount{cfgIdx} = compactCells;
end

end

function [poc2Cfg, fullCellCount, compactCellCount, sSlotCfg, bfwCfg, bfwSrsBindCfg, overrideCfg] = buildPoc2Config(patterns, channelMap)

patternEntries = poc2PatternEntries(patterns);
poc2Cfg = cell(numel(patternEntries), 7);
fullCellCount = cell(numel(patternEntries), 1);
compactCellCount = cell(numel(patternEntries), 1);
sSlotCfg = cell(0, 3);
bfwCfg = cell(0, 3);
bfwSrsBindCfg = cell(0, 9);
overrideCfg = cell(0, 10);

for idxEntry = 1:numel(patternEntries)
    patternEntry = patternEntries{idxEntry};
    [cfgRow, fullCells, compactCells] = buildPoc2ConfigRow(patternEntry, channelMap);
    poc2Cfg(idxEntry, :) = cfgRow;
    fullCellCount{idxEntry} = fullCells;
    compactCellCount{idxEntry} = compactCells;
    sSlotCfg = [sSlotCfg; buildPoc2SSlotRows(patternEntry)];
    bfwCfg = [bfwCfg; buildPoc2BfwRows(patternEntry)];
    bfwSrsBindCfg = [bfwSrsBindCfg; buildPoc2BfwSrsBindRows(patternEntry)];
    overrideCfg = [overrideCfg; buildPoc2OverrideRows(patternEntry)];
end

end

function patternEntriesOut = poc2PatternEntries(patterns)

patternEntries = yamlEntries(patterns);
patternEntriesOut = {};
for idxEntry = 1:numel(patternEntries)
    patternEntry = patternEntries{idxEntry};
    if isfield(patternEntry, 'pattern_type')
        patternEntriesOut{end + 1} = patternEntry;
    end
end

end

function [cfgRow, fullCells, compactCells] = buildPoc2ConfigRow(patternEntry, channelMap)

pattern = yamlScalar(patternEntry.pattern);
configCells = poc2ConfigCells(patternEntry);
fullCells = poc2GeneratedCells(patternEntry, 'full_cells', configCells);
compactCells = poc2GeneratedCells(patternEntry, 'compact_cells', fullCells);

patternType = string(yamlText(patternEntry.pattern_type));
dlTv = poc2TvValue(patternEntry, 'DLMIX', 'dl_tv');
if strcmp(patternType, "CA")
    dlTvIdx1 = num2cell(dlTv);
else
    dlTvIdx1 = {dlTv};
end

if isfield(patternEntry, 'dl_tv_delta')
    dlTvIdx2Delta = yamlRow(patternEntry.dl_tv_delta);
else
    dlTvIdx2Delta = 0;
end
ulTvIdx = poc2TvValue(patternEntry, 'ULMIX', 'ul_tv');

channelName = yamlText(patternEntry.channel);
if ~isa(channelMap, 'containers.Map') || ~isKey(channelMap, channelName)
    error('Pattern %.10g references unknown POC2 channel "%s".', pattern, channelName);
end

channels = channelMap(channelName);

cfgRow = {pattern, configCells, patternType, dlTvIdx1, dlTvIdx2Delta, ulTvIdx, channels};

end

function cells = poc2ConfigCells(patternEntry)

cells = yamlRow(patternEntry.config_cells);

end

function cells = poc2GeneratedCells(patternEntry, fieldName, defaultCells)

if isfield(patternEntry, fieldName)
    cells = yamlRow(patternEntry.(fieldName));
else
    cells = defaultCells;
end

end

function tv = poc2TvValue(patternEntry, side, fieldName)

if isfield(patternEntry, fieldName)
    tv = yamlRow(patternEntry.(fieldName));
    return;
end

tv = poc2TvFromRange(patternEntry, side);
if isempty(tv)
    pattern = yamlScalar(patternEntry.pattern);
    error('Pattern %.10g must define %s or ranges.%s.tvs with a main range.', pattern, fieldName, side);
end

end

function rows = buildPoc2SSlotRows(patternEntry)

rows = cell(0, 3);
if ~isfield(patternEntry, 's_slot')
    return;
end

pattern = yamlScalar(patternEntry.pattern);
sSlotCfg = patternEntry.s_slot;
rows(1, :) = {pattern, poc2ConfigCells(patternEntry), yamlNumericSpec(sSlotCfg.tv)};

end

function rows = buildPoc2BfwRows(patternEntry)

rows = cell(0, 3);
if ~isfield(patternEntry, 'bfw')
    return;
end

pattern = yamlScalar(patternEntry.pattern);
bfwCfg = patternEntry.bfw;
rows(1, :) = {pattern, yamlNumericSpec(bfwCfg.dl_tv), yamlNumericSpec(bfwCfg.ul_tv)};

end

function rows = buildPoc2BfwSrsBindRows(patternEntry)

rows = cell(0, 9);
if ~isfield(patternEntry, 'bfw_srs_bind')
    return;
end

pattern = yamlScalar(patternEntry.pattern);
bindEntries = yamlEntries(patternEntry.bfw_srs_bind);
rows = cell(numel(bindEntries), 9);
for idxEntry = 1:numel(bindEntries)
    bindEntry = bindEntries{idxEntry};
    rows(idxEntry, :) = {pattern, ...
                         yamlNumericSpec(bindEntry.srs_tv), ...
                         yamlNumericSpec(bindEntry.srs_slot), ...
                         yamlNumericSpec(bindEntry.srs_name_slot), ...
                         yamlNumericSpec(bindEntry.bfw_dl_tv), ...
                         yamlNumericSpec(bindEntry.bfw_dl_slots), ...
                         yamlNumericSpec(bindEntry.bfw_ul_tv), ...
                         yamlNumericSpec(bindEntry.bfw_ul_slots), ...
                         yamlNumericSpec(bindEntry.cell_idx)};
end

end

function rows = buildPoc2OverrideRows(patternEntry)

rows = cell(0, 10);
if ~isfield(patternEntry, 'overrides')
    return;
end

pattern = yamlScalar(patternEntry.pattern);
overrideEntries = yamlEntries(patternEntry.overrides);
rows = cell(numel(overrideEntries), 10);
for idxEntry = 1:numel(overrideEntries)
    overrideEntry = overrideEntries{idxEntry};
    rows(idxEntry, :) = {pattern, ...
                         yamlNumericSpec(overrideEntry.base_pattern), ...
                         yamlNumericSpec(overrideEntry.cell_idx), ...
                         yamlNumericSpec(overrideEntry.tv), ...
                         yamlNumericSpec(overrideEntry.tv_increments), ...
                         yamlNumericSpec(overrideEntry.slot), ...
                         yamlNumericSpec(overrideEntry.tv_name_slot), ...
                         yamlNumericSpec(overrideEntry.mix_tv), ...
                         yamlNumericSpec(overrideEntry.target_dl), ...
                         yamlNumericSpec(overrideEntry.verbose)};
end

end

function entries = yamlRangeEntries(value)

if isempty(value)
    entries = {};
elseif isstruct(value) || (iscell(value) && all(cellfun(@isstruct, value)))
    entries = yamlEntries(value);
else
    rows = yamlNumeric(value);
    if isempty(rows)
        entries = {};
    elseif isvector(rows)
        entries = {reshape(rows, 1, [])};
    else
        entries = num2cell(rows, 2);
    end
end

end

function entries = yamlEntries(value)

if isempty(value)
    entries = {};
elseif iscell(value)
    entries = reshape(value, 1, []);
elseif isstruct(value)
    entries = num2cell(reshape(value, 1, []));
else
    error('Expected YAML sequence of mappings.');
end

end

function out = getStructField(value, fieldName)

out = [];
if isempty(value)
    return;
end
if ~isstruct(value)
    error('Expected a YAML mapping when looking for field "%s".', fieldName);
end

fields = fieldnames(value);
fieldIdx = find(strcmpi(fields, fieldName), 1);
if ~isempty(fieldIdx)
    out = value.(fields{fieldIdx});
end

end

function out = yamlTextList(value)

if isempty(value)
    out = {};
elseif iscell(value)
    out = cell(1, numel(value));
    for idx = 1:numel(value)
        out{idx} = yamlText(value{idx});
    end
elseif isnumeric(value)
    % Bare numeric sequence, e.g. compact_patterns: [59, 60, 69]. Each entry
    % is stringified so it matches the string pattern ids ('59', '69').
    out = cell(1, numel(value));
    for idx = 1:numel(value)
        out{idx} = yamlText(value(idx));
    end
elseif ischar(value) || isstring(value)
    out = {yamlText(value)};
else
    error('Expected a YAML string, numeric, or sequence of those.');
end

end

function out = yamlText(value)

if isstring(value)
    out = char(value);
elseif ischar(value)
    out = value;
elseif isnumeric(value) && isscalar(value)
    % Accept a bare YAML number (e.g. a compact_patterns entry written as 69
    % instead of '69') and use its value as the id string. Integers render
    % without a decimal point so they match string ids like '69'; fractional
    % ids (e.g. 59.3) keep their decimal form.
    if value == floor(value)
        out = num2str(value);
    else
        out = num2str(value, '%.10g');
    end
else
    error('Expected a YAML string or numeric scalar value.');
end

end

function out = yamlBool(value)

if islogical(value)
    out = value;
elseif isnumeric(value) && isscalar(value)
    out = value ~= 0;
elseif ischar(value) || isstring(value)
    text = lower(strtrim(char(value)));
    switch text
        case {'true', 'yes', '1'}
            out = true;
        case {'false', 'no', '0'}
            out = false;
        otherwise
            error('Expected a YAML boolean value.');
    end
else
    error('Expected a YAML boolean value.');
end

end

function out = yamlNumericSpec(value)

if isempty(value)
    out = [];
elseif ischar(value) || isstring(value)
    out = parseNumericSpec(char(value));
else
    out = yamlRow(value);
end

end

function out = parseNumericSpec(value)

value = strtrim(value);
if isempty(value) || strcmp(value, '[]')
    out = [];
    return;
end
if startsWith(value, '[') && endsWith(value, ']')
    value = strtrim(value(2:end-1));
end
if isempty(value)
    out = [];
    return;
end

tokens = strsplit(value, ',');
out = [];
for idxToken = 1:numel(tokens)
    token = strtrim(tokens{idxToken});
    if isempty(token)
        continue;
    end
    parts = strsplit(token, ':');
    nums = cellfun(@str2double, parts);
    if any(isnan(nums))
        error('Invalid numeric range expression "%s".', token);
    end
    switch numel(nums)
        case 1
            out = [out, nums(1)];
        case 2
            out = [out, nums(1):nums(2)];
        case 3
            out = [out, nums(1):nums(2):nums(3)];
        otherwise
            error('Invalid numeric range expression "%s".', token);
    end
end

end

function out = yamlNumeric(value)

if isempty(value)
    out = [];
elseif isnumeric(value)
    out = value;
elseif iscell(value)
    rows = cell(size(value));
    for idx = 1:numel(value)
        rows{idx} = yamlRow(value{idx});
    end
    if isempty(rows)
        out = [];
    elseif all(cellfun(@(x) isscalar(x), rows))
        out = cell2mat(rows);
    else
        rowWidths = cellfun(@numel, rows);
        if numel(unique(rowWidths)) ~= 1
            error('YAML numeric rows have inconsistent widths.');
        end
        out = vertcat(rows{:});
    end
else
    error('Expected a numeric YAML value.');
end

end

function out = yamlRow(value)

if isempty(value)
    out = [];
elseif isnumeric(value)
    out = reshape(value, 1, []);
elseif iscell(value)
    parts = cell(size(value));
    for idx = 1:numel(value)
        parts{idx} = yamlRow(value{idx});
    end
    out = [parts{:}];
else
    error('Expected a numeric YAML scalar or sequence.');
end

end

function out = yamlScalar(value)

out = yamlRow(value);
if numel(out) ~= 1
    error('Expected a scalar YAML value.');
end
out = out(1);

end
