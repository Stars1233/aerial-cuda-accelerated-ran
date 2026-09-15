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

function result = WriteYaml(filename, data, flowStyle)
%WRITEYAML Write the restricted YAML subset used by Aerial 5GModel.
%
% WRITEYAML(FILENAME, DATA) serializes DATA to FILENAME. Passing an
% empty filename returns the YAML text instead. FLOWSTYLE=1 serializes the
% root value using flow collections. The accepted MATLAB values are scalar
% structs, cells, numeric/logical arrays, and character vectors.

if nargin < 3 || isempty(flowStyle)
    flowStyle = 0;
end
if ~(isscalar(flowStyle) && (islogical(flowStyle) || isnumeric(flowStyle)) && ...
        any(flowStyle == [0, 1]))
    error('AERIAL:YAML:InvalidOption', 'flowStyle must be 0 or 1.');
end

validateValue(data, 'root');
if flowStyle
    yamlText = [encodeFlowValue(data, 0, 0, true), newline];
else
    outputLines = encodeBlockValue(data, 0);
    yamlText = [strjoin(outputLines, newline), newline];
end

if isempty(filename)
    result = yamlText;
    return;
end

fileName = char(filename);
[fileId, message] = fopen(fileName, 'w');
if fileId < 0
    error('AERIAL:YAML:FileOpenFailed', 'Cannot open %s for writing: %s', fileName, message);
end
cleanup = onCleanup(@() fclose(fileId));
count = fprintf(fileId, '%s', yamlText);
if count ~= length(yamlText)
    error('AERIAL:YAML:FileWriteFailed', 'Failed to write all YAML data to %s.', fileName);
end
clear cleanup;
result = [];
end

function validateValue(value, location)
if isstruct(value)
    if ~isscalar(value)
        error('AERIAL:YAML:UnsupportedType', ...
            'Struct arrays are not supported at %s; use a cell array of structs.', location);
    end
    names = fieldnames(value);
    for index = 1:numel(names)
        validateValue(value.(names{index}), [location, '.', names{index}]);
    end
elseif iscell(value)
    if ndims(value) > 2 %#ok<ISMAT>
        error('AERIAL:YAML:UnsupportedType', 'N-D cell arrays are not supported at %s.', location);
    end
    for index = 1:numel(value)
        validateValue(value{index}, sprintf('%s{%d}', location, index));
    end
elseif isnumeric(value) || islogical(value)
    if ndims(value) > 2 %#ok<ISMAT>
        error('AERIAL:YAML:UnsupportedType', 'N-D arrays are not supported at %s.', location);
    end
elseif ischar(value)
    if ~(isrow(value) || isempty(value))
        error('AERIAL:YAML:UnsupportedType', 'Character matrices are not supported at %s.', location);
    end
elseif isstring(value) && isscalar(value)
    % String scalars are accepted as a convenience; yamlmatlab used char.
else
    error('AERIAL:YAML:UnsupportedType', 'Cannot write MATLAB type %s at %s.', class(value), location);
end
end

function lines = encodeBlockValue(value, indent)
if isCompositeValue(value) && isAutoFlowCollection(value)
    lines = {[spaces(indent), encodeFlowValue(value, indent, indent + 2)]};
elseif isstruct(value)
    lines = encodeBlockMapping(value, indent);
elseif isSequenceValue(value)
    lines = encodeBlockSequence(sequenceRows(value), indent);
else
    lines = {[spaces(indent), encodeScalar(value, indent + 2)]};
end
end

function lines = encodeBlockMapping(value, indent)
names = fieldnames(value);
if isempty(names)
    lines = {[spaces(indent), '{}']};
    return;
end

lines = {};
for index = 1:numel(names)
    name = names{index};
    child = value.(name);
    key = encodeKey(name, false);
    if isBlockValue(child)
        lines{end + 1} = [spaces(indent), key, ':']; %#ok<AGROW>
        if isSequenceValue(child)
            childIndent = indent;
        else
            childIndent = indent + 2;
        end
        childLines = encodeBlockValue(child, childIndent);
        lines = [lines, childLines]; %#ok<AGROW>
    else
        prefix = [spaces(indent), key, ': '];
        childText = encodeFlowValue(child, length(prefix), indent + 2);
        childLines = splitEncodedText(childText);
        lines{end + 1} = [prefix, childLines{1}]; %#ok<AGROW>
        lines = [lines, childLines(2:end)]; %#ok<AGROW>
    end
end
end

function lines = encodeBlockSequence(rows, indent)
if isempty(rows)
    lines = {[spaces(indent), '[]']};
    return;
end

lines = {};
for rowIndex = 1:numel(rows)
    item = rows{rowIndex};
    if isBlockValue(item)
        childLines = encodeBlockValue(item, indent + 2);
        firstLine = childLines{1};
        firstLine = [spaces(indent), '- ', firstLine(indent + 3:end)];
        lines{end + 1} = firstLine; %#ok<AGROW>
        lines = [lines, childLines(2:end)]; %#ok<AGROW>
    else
        prefix = [spaces(indent), '- '];
        itemText = encodeFlowValue(item, length(prefix), indent + 2);
        itemLines = splitEncodedText(itemText);
        lines{end + 1} = [prefix, itemLines{1}]; %#ok<AGROW>
        lines = [lines, itemLines(2:end)]; %#ok<AGROW>
    end
end
end

function text = encodeFlowValue(value, startColumn, continuationIndent, forceFlow, inFlow)
if nargin < 2
    startColumn = 0;
end
if nargin < 3
    continuationIndent = 0;
end
if nargin < 4
    forceFlow = false;
end
if nargin < 5
    inFlow = false;
end
if isstruct(value)
    names = fieldnames(value);
    entries = cell(1, numel(names));
    for index = 1:numel(names)
        entries{index} = [encodeKey(names{index}, true), ': ', ...
            encodeFlowValue(value.(names{index}), 0, continuationIndent + 2, forceFlow, true)];
    end
    text = encodeFlowEntries(entries, '{', '}', startColumn, continuationIndent);
elseif isSequenceValue(value)
    rows = sequenceRows(value);
    entries = cell(1, numel(rows));
    for index = 1:numel(rows)
        entries{index} = encodeFlowValue(rows{index}, 0, continuationIndent + 2, forceFlow, true);
    end
    text = encodeFlowEntries(entries, '[', ']', startColumn, continuationIndent);
else
    text = encodeScalar(value, continuationIndent, forceFlow, inFlow);
end
end

function text = encodeFlowEntries(entries, opening, closing, startColumn, continuationIndent)
text = opening;
column = startColumn + 1;
for index = 1:numel(entries)
    if index > 1
        if column >= 80
            separator = [',', newline, spaces(continuationIndent)];
            column = continuationIndent;
        else
            separator = ', ';
            column = column + 2;
        end
        text = [text, separator]; %#ok<AGROW>
    end
    entry = entries{index};
    text = [text, entry]; %#ok<AGROW>
    lastNewline = find(entry == newline, 1, 'last');
    if isempty(lastNewline)
        column = column + length(entry);
    else
        column = length(entry) - lastNewline;
    end
end
text = [text, closing];
end

function lines = splitEncodedText(text)
lines = regexp(text, '\n', 'split');
end

function rows = sequenceRows(value)
if iscell(value)
    if isempty(value)
        rows = {};
    elseif isscalar(value)
        rows = value;
    elseif isrow(value)
        rows = reshape(value, 1, []);
    elseif iscolumn(value)
        rows = cell(size(value, 1), 1);
        for rowIndex = 1:size(value, 1)
            item = value{rowIndex};
            if iscell(item)
                rows{rowIndex} = item;
            else
                rows{rowIndex} = {item};
            end
        end
    else
        rows = cell(size(value, 1), 1);
        for rowIndex = 1:size(value, 1)
            rows{rowIndex} = value(rowIndex, :);
        end
    end
elseif isnumeric(value) || islogical(value)
    if isempty(value)
        rows = {};
    elseif isscalar(value)
        rows = {value};
    elseif isrow(value)
        rows = num2cell(value);
    elseif iscolumn(value)
        rows = cell(size(value, 1), 1);
        for rowIndex = 1:size(value, 1)
            rows{rowIndex} = {value(rowIndex, :)};
        end
    else
        rows = cell(size(value, 1), 1);
        for rowIndex = 1:size(value, 1)
            rows{rowIndex} = value(rowIndex, :);
        end
    end
else
    error('AERIAL:YAML:InternalError', 'sequenceRows received a scalar value.');
end
end

function result = isSequenceValue(value)
result = iscell(value) || ((isnumeric(value) || islogical(value)) && ~isscalar(value));
end

function result = isCompositeValue(value)
result = isstruct(value) || isSequenceValue(value);
end

function result = isBlockValue(value)
result = isCompositeValue(value) && ~isAutoFlowCollection(value);
end

function result = isAutoFlowCollection(value)
if isstruct(value)
    names = fieldnames(value);
    result = true;
    for index = 1:numel(names)
        if ~isPlainStyleScalar(value.(names{index}))
            result = false;
            return;
        end
    end
elseif isSequenceValue(value)
    rows = sequenceRows(value);
    result = true;
    for index = 1:numel(rows)
        if ~isPlainStyleScalar(rows{index})
            result = false;
            return;
        end
    end
else
    result = false;
end
end

function result = isPlainStyleScalar(value)
if ischar(value)
    result = ~any(value == newline) && ~any(value == sprintf('\r'));
elseif isstring(value) && isscalar(value)
    text = char(value);
    result = ~any(text == newline) && ~any(text == sprintf('\r'));
else
    result = (isnumeric(value) || islogical(value)) && isscalar(value);
end
end

function text = encodeScalar(value, continuationIndent, forceFlow, inFlow)
if nargin < 2
    continuationIndent = 0;
end
if nargin < 3
    forceFlow = false;
end
if nargin < 4
    inFlow = false;
end
if ischar(value)
    text = encodeString(value, continuationIndent, forceFlow, inFlow);
elseif isstring(value)
    text = encodeString(char(value), continuationIndent, forceFlow, inFlow);
elseif islogical(value)
    if value
        text = 'true';
    else
        text = 'false';
    end
elseif isnumeric(value)
    if ~isscalar(value)
        error('AERIAL:YAML:InternalError', 'encodeScalar received a numeric array.');
    end
    if isinteger(value)
        text = encodeInteger(value);
    else
        text = encodeNumber(double(value));
    end
else
    error('AERIAL:YAML:InternalError', 'encodeScalar received type %s.', class(value));
end
end

function text = encodeNumber(value)
if isnan(value)
    text = '.NaN';
elseif isinf(value)
    if value < 0
        text = '-.inf';
    else
        text = '.inf';
    end
elseif value == 0 && isinf(1 / value) && 1 / value < 0
    text = '-0.0';
else
    text = char(javaMethod('toString', 'java.lang.Double', value));
end
end

function text = encodeInteger(value)
if startsWith(class(value), 'uint')
    text = sprintf('%u', value);
else
    text = sprintf('%d', value);
end
end

function text = encodeString(value, continuationIndent, forceFlow, inFlow)
if nargin < 2
    continuationIndent = 0;
end
if nargin < 3
    forceFlow = false;
end
if nargin < 4
    inFlow = false;
end
if isempty(value)
    text = '''''';
elseif (any(value == newline) || any(value == sprintf('\r'))) && ~forceFlow
    text = encodeBlockString(value, continuationIndent);
elseif any(value == newline) || any(value == sprintf('\r')) || any(value == sprintf('\t'))
    text = jsonencode(value);
elseif canUsePlainString(value, inFlow)
    text = value;
else
    text = ['''', strrep(value, '''', ''''''), ''''];
end
end

function text = encodeBlockString(value, continuationIndent)
value = strrep(value, sprintf('\r\n'), newline);
value = strrep(value, sprintf('\r'), newline);
if endsWith(value, [newline, newline])
    header = '|+';
elseif endsWith(value, newline)
    header = '|';
else
    header = '|-';
end
if endsWith(value, newline) && ~strcmp(header, '|+')
    value(end) = [];
end
contentLines = regexp(value, '\n', 'split');
for index = 1:numel(contentLines)
    if ~isempty(contentLines{index})
        contentLines{index} = [spaces(continuationIndent), contentLines{index}];
    end
end
text = [header, newline, strjoin(contentLines, newline)];
end

function result = canUsePlainString(value, inFlow)
result = false;
if nargin < 2
    inFlow = false;
end
if isempty(value) || ~strcmp(value, strtrim(value)) || ...
        any(value == newline) || any(value == sprintf('\r')) || any(value == sprintf('\t'))
    return;
end
if any(value(1) == ['-', '?', ':', ',', '[', ']', '{', '}', '#', '&', '*', '!', ...
        '|', '>', '''', '"', '%', '@', '`']) || ...
        contains(value, ': ') || contains(value, ' #') || ...
        (inFlow && any(ismember(value, [',', '[', ']', '{', '}'])))
    return;
end
if any(strcmpi(value, {'~', 'null', 'true', 'false', 'yes', 'no', 'on', 'off', ...
        '.nan', '+.nan', '-.nan', '.inf', '+.inf', '-.inf'})) || ...
        ~isempty(regexp(value, '^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$', 'once')) || ...
        ~isempty(regexp(value, '^0[xX][0-9a-fA-F_]+$', 'once')) || ...
        ~isempty(regexp(value, '^[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}(?:[Tt ]|$)', 'once'))
    return;
end
result = true;
end

function text = encodeKey(value, inFlow)
if nargin < 2
    inFlow = false;
end
if ~isempty(regexp(value, '^[A-Za-z_][A-Za-z0-9_]*$', 'once'))
    text = value;
else
    text = encodeString(value, 0, false, inFlow);
end
end

function result = spaces(count)
result = repmat(' ', 1, count);
end
