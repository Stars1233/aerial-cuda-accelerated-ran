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

function result = ReadYaml(inputValue, noSuchFileAction, makeOrdinaryArrays, treatAsData, dictionary)
%READYAML Read the restricted YAML subset used by Aerial 5GModel.
%
% The supported subset includes block and flow mappings/sequences, quoted
% and plain strings, finite and non-finite numeric scalars, logical scalars,
% comments, and literal/folded block strings. YAML aliases, anchors, tags,
% merge keys, directives, and multiple documents are intentionally rejected.
%
% The optional arguments retain the legacy ReadYaml call contract used by
% 5GModel. When MAKEORDINARYARRAYS is true, aligned nested scalar sequences
% are converted to MATLAB arrays in the same way as yamlmatlab's
% makematrices helper. TREATASDATA parses INPUTVALUE directly instead of
% treating it as a file name. DICTIONARY substitutes matching string values.

if nargin < 2 || isempty(noSuchFileAction)
    noSuchFileAction = 0;
end
if nargin < 3 || isempty(makeOrdinaryArrays)
    makeOrdinaryArrays = 0;
end
if nargin < 4 || isempty(treatAsData)
    treatAsData = 0;
end

validateBinaryOption(noSuchFileAction, 'noSuchFileAction');
validateBinaryOption(makeOrdinaryArrays, 'makeOrdinaryArrays');
validateBinaryOption(treatAsData, 'treatAsData');

if treatAsData
    yamlText = char(inputValue);
    sourceName = '<data>';
else
    sourceName = char(inputValue);
    if exist(sourceName, 'file') ~= 2
        if noSuchFileAction
            error('MATLAB:MATYAML:FileNotFound', 'No such file to read: %s', sourceName);
        end
        warning('MATLAB:MATYAML:FileNotFound', 'No such file to read: %s', sourceName);
        result = struct();
        return;
    end
    yamlText = fileread(sourceName);
end

lines = preprocessYaml(yamlText, sourceName);
if isempty(lines)
    result = [];
    return;
end

if isSequenceLine(lines(1).text) || isMappingEntry(lines(1).text)
    [result, nextIndex] = parseBlock(lines, 1, lines(1).indent, sourceName);
else
    result = parseInlineValue(lines(1).text, sourceName, lines(1).number);
    nextIndex = 2;
end
if nextIndex <= numel(lines)
    parseError(sourceName, lines(nextIndex).number, ...
        'Unexpected content or a second YAML document.');
end

result = makeMatrices(result, logical(makeOrdinaryArrays));
if nargin >= 5
    if ~isstruct(dictionary)
        error('AERIAL:YAML:InvalidDictionary', 'dictionary must be a MATLAB struct.');
    end
    result = substituteDictionary(result, dictionary);
end
end

function validateBinaryOption(value, name)
if ~(isscalar(value) && (islogical(value) || isnumeric(value)) && any(value == [0, 1]))
    error('AERIAL:YAML:InvalidOption', '%s must be 0 or 1.', name);
end
end

function lines = preprocessYaml(yamlText, sourceName)
physicalLines = regexp(yamlText, '\r\n|\n|\r', 'split');
lines = struct('indent', {}, 'text', {}, 'number', {});
index = 1;
documentStartSeen = false;
documentHasContent = false;
documentEnded = false;

while index <= numel(physicalLines)
    sourceLine = index;
    rawLine = physicalLines{index};
    if hasLeadingTab(rawLine)
        parseError(sourceName, index, 'Tabs are not allowed for YAML indentation.');
    end

    indent = countIndent(rawLine);
    content = stripComment(rawLine(indent + 1:end));
    content = strtrim(content);
    if isempty(content)
        index = index + 1;
        continue;
    end
    if indent == 0 && strcmp(content, '---')
        if documentStartSeen || documentHasContent || documentEnded
            parseError(sourceName, index, 'Multiple YAML documents are not supported.');
        end
        documentStartSeen = true;
        index = index + 1;
        continue;
    end
    if indent == 0 && strcmp(content, '...')
        if documentEnded
            parseError(sourceName, index, 'Unexpected YAML document-end marker.');
        end
        documentEnded = true;
        index = index + 1;
        continue;
    end
    if documentEnded
        parseError(sourceName, index, 'Content after the YAML document-end marker is not supported.');
    end
    documentHasContent = true;
    if startsWith(content, '%')
        parseError(sourceName, index, 'YAML directives are not supported.');
    end

    [isBlockScalar, indicatorPosition, style, chomping, parentIndentOffset] = ...
        blockScalarIndicator(content);
    if isBlockScalar
        [blockValue, index] = collectBlockScalar(physicalLines, index, ...
            indent + parentIndentOffset, style, chomping);
        content = [content(1:indicatorPosition - 1), jsonencode(blockValue)];
    else
        index = index + 1;
    end

    startLine = sourceLine;
    balance = flowBalance(content);
    while balance > 0
        if index > numel(physicalLines)
            parseError(sourceName, startLine, 'Unterminated flow collection.');
        end
        continuation = physicalLines{index};
        if hasLeadingTab(continuation)
            parseError(sourceName, index, 'Tabs are not allowed for YAML indentation.');
        end
        continuation = strtrim(stripComment(continuation));
        index = index + 1;
        if isempty(continuation)
            continue;
        end
        content = [content, ' ', continuation]; %#ok<AGROW>
        balance = flowBalance(content);
    end

    if balance < 0
        parseError(sourceName, startLine, 'Unexpected flow collection terminator.');
    end
    lines(end + 1) = struct('indent', indent, 'text', content, ... %#ok<AGROW>
        'number', startLine);
end
end

function result = hasLeadingTab(line)
firstNonWhitespace = regexp(line, '\S', 'once');
if isempty(firstNonWhitespace)
    result = false;
else
    result = any(line(1:firstNonWhitespace - 1) == sprintf('\t'));
end
end

function indent = countIndent(line)
indent = 0;
while indent < length(line) && line(indent + 1) == ' '
    indent = indent + 1;
end
end

function output = stripComment(input)
singleQuoted = false;
doubleQuoted = false;
escaped = false;
index = 1;

while index <= length(input)
    character = input(index);
    if doubleQuoted
        if escaped
            escaped = false;
        elseif character == '\'
            escaped = true;
        elseif character == '"'
            doubleQuoted = false;
        end
    elseif singleQuoted
        if character == ''''
            if index < length(input) && input(index + 1) == ''''
                index = index + 1;
            else
                singleQuoted = false;
            end
        end
    elseif character == '''' && isQuoteOpening(input, index)
        singleQuoted = true;
    elseif character == '"' && isQuoteOpening(input, index)
        doubleQuoted = true;
    elseif character == '#' && (index == 1 || isspace(input(index - 1)))
        output = input(1:index - 1);
        return;
    end
    index = index + 1;
end
output = input;
end

function [found, position, style, chomping, parentIndentOffset] = blockScalarIndicator(content)
found = false;
position = 0;
style = '';
chomping = '';
parentIndentOffset = 0;

colonPosition = findTopLevelColon(content);
if colonPosition > 0
    valueText = strtrim(content(colonPosition + 1:end));
    if isBlockScalarHeader(valueText)
        found = true;
        position = colonPosition + find(content(colonPosition + 1:end) == valueText(1), 1);
        style = valueText(1);
        if length(valueText) == 2
            chomping = valueText(2);
        end
        parentIndentOffset = 2 * leadingSequencePrefixCount(content(1:colonPosition - 1));
        return;
    end
end

[sequencePrefixCount, valuePosition] = leadingSequencePrefixes(content);
valueText = strtrim(content(valuePosition:end));
if ~isBlockScalarHeader(valueText)
    return;
end
found = true;
position = valuePosition + find(content(valuePosition:end) == valueText(1), 1) - 1;
style = valueText(1);
if length(valueText) == 2
    chomping = valueText(2);
end
parentIndentOffset = 2 * max(sequencePrefixCount - 1, 0);
end

function result = isBlockScalarHeader(text)
result = ~isempty(text) && any(text(1) == ['|', '>']) && ...
    (isscalar(text) || (length(text) == 2 && any(text(2) == ['+', '-'])));
end

function count = leadingSequencePrefixCount(text)
[count, ~] = leadingSequencePrefixes(text);
end

function [count, position] = leadingSequencePrefixes(text)
count = 0;
position = 1;
while position <= length(text)
    while position <= length(text) && isspace(text(position))
        position = position + 1;
    end
    if position > length(text) || text(position) ~= '-' || ...
            (position < length(text) && ~isspace(text(position + 1)))
        return;
    end
    count = count + 1;
    position = position + 1;
end
end

function [value, nextIndex] = collectBlockScalar(lines, currentIndex, parentIndent, style, chomping)
nextIndex = currentIndex + 1;
collected = {};
indents = [];

while nextIndex <= numel(lines)
    candidate = lines{nextIndex};
    candidateIndent = countIndent(candidate);
    if ~isempty(strtrim(candidate)) && candidateIndent <= parentIndent
        break;
    end
    collected{end + 1} = candidate; %#ok<AGROW>
    if ~isempty(strtrim(candidate))
        indents(end + 1) = candidateIndent; %#ok<AGROW>
    end
    nextIndex = nextIndex + 1;
end

hasTrailingLineBreak = false;
if nextIndex > numel(lines)
    if ~isempty(collected) && isempty(lines{end})
        collected(end) = [];
        hasTrailingLineBreak = true;
    end
elseif ~isempty(collected)
    hasTrailingLineBreak = true;
end

if isempty(collected)
    value = '';
    return;
end
if isempty(indents)
    contentIndent = parentIndent + 1;
else
    contentIndent = min(indents);
end
for index = 1:numel(collected)
    line = collected{index};
    if length(line) >= contentIndent
        collected{index} = line(contentIndent + 1:end);
    else
        collected{index} = '';
    end
end

if ~strcmp(chomping, '+')
    while ~isempty(collected) && isempty(collected{end})
        collected(end) = [];
    end
end
if isempty(collected)
    value = '';
    return;
end

if style == '|'
    value = strjoin(collected, newline);
else
    value = foldBlockLines(collected);
end
if hasTrailingLineBreak && ~strcmp(chomping, '-')
    value = [value, newline];
end
end

function output = foldBlockLines(lines)
output = '';
for index = 1:numel(lines)
    if index > 1
        if isempty(lines{index - 1}) || isempty(lines{index})
            output = [output, newline]; %#ok<AGROW>
        else
            output = [output, ' ']; %#ok<AGROW>
        end
    end
    output = [output, lines{index}]; %#ok<AGROW>
end
end

function balance = flowBalance(text)
squareDepth = 0;
curlyDepth = 0;
singleQuoted = false;
doubleQuoted = false;
escaped = false;
index = 1;

while index <= length(text)
    character = text(index);
    if doubleQuoted
        if escaped
            escaped = false;
        elseif character == '\'
            escaped = true;
        elseif character == '"'
            doubleQuoted = false;
        end
    elseif singleQuoted
        if character == ''''
            if index < length(text) && text(index + 1) == ''''
                index = index + 1;
            else
                singleQuoted = false;
            end
        end
    else
        switch character
            case ''''
                singleQuoted = isQuoteOpening(text, index);
            case '"'
                doubleQuoted = isQuoteOpening(text, index);
            case '['
                squareDepth = squareDepth + 1;
            case ']'
                squareDepth = squareDepth - 1;
            case '{'
                curlyDepth = curlyDepth + 1;
            case '}'
                curlyDepth = curlyDepth - 1;
        end
    end
    if squareDepth < 0 || curlyDepth < 0
        balance = -1;
        return;
    end
    index = index + 1;
end
balance = squareDepth + curlyDepth + double(singleQuoted || doubleQuoted);
end

function result = isQuoteOpening(text, index)
previous = index - 1;
while previous >= 1 && isspace(text(previous))
    previous = previous - 1;
end
result = previous < 1 || any(text(previous) == [':', ',', '[', '{', '-']);
end

function [value, nextIndex] = parseBlock(lines, index, indent, sourceName)
if index > numel(lines)
    value = [];
    nextIndex = index;
    return;
end
if lines(index).indent ~= indent
    parseError(sourceName, lines(index).number, 'Unexpected indentation.');
end

if isSequenceLine(lines(index).text)
    [value, nextIndex] = parseSequence(lines, index, indent, sourceName);
else
    [value, nextIndex] = parseMapping(lines, index, indent, sourceName);
end
end

function [result, index] = parseMapping(lines, index, indent, sourceName)
result = struct();
while index <= numel(lines)
    line = lines(index);
    if line.indent ~= indent || isSequenceLine(line.text)
        break;
    end
    [key, valueText] = splitMappingEntry(line.text, sourceName, line.number);
    [value, index] = parseEntryValue(lines, index + 1, indent, valueText, sourceName);
    result = assignMappingField(result, key, value, sourceName, line.number);
end
end

function [result, index] = parseSequence(lines, index, indent, sourceName)
result = cell(0, 1);
while index <= numel(lines)
    line = lines(index);
    if line.indent ~= indent || ~isSequenceLine(line.text)
        break;
    end
    itemText = strtrim(line.text(2:end));
    if isempty(itemText)
        index = index + 1;
        if index > numel(lines) || lines(index).indent <= indent
            value = [];
        else
            [value, index] = parseBlock(lines, index, lines(index).indent, sourceName);
        end
    elseif isSequenceLine(itemText)
        compactLine = line;
        compactLine.indent = indent + 2;
        compactLine.text = itemText;
        nestedLines = [compactLine, lines(index + 1:end)];
        [value, nestedNextIndex] = parseSequence(nestedLines, 1, indent + 2, sourceName);
        index = index + nestedNextIndex - 1;
    elseif isMappingEntry(itemText)
        [value, index] = parseSequenceMapping(lines, index, indent, itemText, sourceName);
    else
        value = parseInlineValue(itemText, sourceName, line.number);
        index = index + 1;
        if index <= numel(lines) && lines(index).indent > indent
            parseError(sourceName, lines(index).number, ...
                'A scalar sequence item cannot have nested content.');
        end
    end
    result{end + 1, 1} = value; %#ok<AGROW>
end
end

function [result, index] = parseSequenceMapping(lines, index, sequenceIndent, firstEntry, sourceName)
result = struct();
entryIndent = sequenceIndent + 2;
lineNumber = lines(index).number;
[key, valueText] = splitMappingEntry(firstEntry, sourceName, lineNumber);
[value, index] = parseEntryValue(lines, index + 1, entryIndent, valueText, sourceName);
result = assignMappingField(result, key, value, sourceName, lineNumber);

while index <= numel(lines)
    line = lines(index);
    if line.indent ~= entryIndent || isSequenceLine(line.text)
        break;
    end
    [key, valueText] = splitMappingEntry(line.text, sourceName, line.number);
    [value, index] = parseEntryValue(lines, index + 1, entryIndent, valueText, sourceName);
    result = assignMappingField(result, key, value, sourceName, line.number);
end
end

function [value, nextIndex] = parseEntryValue(lines, nextIndex, parentIndent, valueText, sourceName)
if ~isempty(valueText)
    value = parseInlineValue(valueText, sourceName, ...
        previousLineNumber(lines, nextIndex));
    return;
end

if nextIndex > numel(lines)
    value = [];
    return;
end
nextLine = lines(nextIndex);
if nextLine.indent > parentIndent || ...
        (nextLine.indent == parentIndent && isSequenceLine(nextLine.text))
    [value, nextIndex] = parseBlock(lines, nextIndex, nextLine.indent, sourceName);
else
    value = [];
end
end

function lineNumber = previousLineNumber(lines, nextIndex)
if nextIndex <= 1
    lineNumber = lines(1).number;
else
    lineNumber = lines(min(nextIndex - 1, numel(lines))).number;
end
end

function result = isSequenceLine(text)
result = strcmp(text, '-') || (length(text) >= 2 && text(1) == '-' && isspace(text(2)));
end

function result = isMappingEntry(text)
result = findTopLevelColon(text) > 0;
end

function [key, valueText] = splitMappingEntry(text, sourceName, lineNumber)
colonPosition = findTopLevelColon(text);
if colonPosition == 0
    parseError(sourceName, lineNumber, 'Expected a mapping entry in the form key: value.');
end
keyText = strtrim(text(1:colonPosition - 1));
if isempty(keyText)
    parseError(sourceName, lineNumber, 'Mapping keys cannot be empty.');
end
if any(keyText(1) == ['''', '"'])
    key = parseInlineValue(keyText, sourceName, lineNumber);
    if ~ischar(key)
        parseError(sourceName, lineNumber, 'Mapping keys must be strings.');
    end
else
    key = keyText;
end
if strcmp(key, '<<')
    parseError(sourceName, lineNumber, 'YAML merge keys are not supported.');
end
valueText = strtrim(text(colonPosition + 1:end));
end

function position = findTopLevelColon(text)
position = 0;
squareDepth = 0;
curlyDepth = 0;
singleQuoted = false;
doubleQuoted = false;
escaped = false;
index = 1;

while index <= length(text)
    character = text(index);
    if doubleQuoted
        if escaped
            escaped = false;
        elseif character == '\'
            escaped = true;
        elseif character == '"'
            doubleQuoted = false;
        end
    elseif singleQuoted
        if character == ''''
            if index < length(text) && text(index + 1) == ''''
                index = index + 1;
            else
                singleQuoted = false;
            end
        end
    else
        switch character
            case ''''
                singleQuoted = isQuoteOpening(text, index);
            case '"'
                doubleQuoted = isQuoteOpening(text, index);
            case '['
                squareDepth = squareDepth + 1;
            case ']'
                squareDepth = squareDepth - 1;
            case '{'
                curlyDepth = curlyDepth + 1;
            case '}'
                curlyDepth = curlyDepth - 1;
            case ':'
                if squareDepth == 0 && curlyDepth == 0 && ...
                        (index == length(text) || isspace(text(index + 1)))
                    position = index;
                    return;
                end
        end
    end
    index = index + 1;
end
end

function result = assignMappingField(result, key, value, sourceName, lineNumber)
fieldName = matlab.lang.makeValidName(key, 'ReplacementStyle', 'underscore');
if isfield(result, fieldName)
    parseError(sourceName, lineNumber, 'Duplicate or colliding mapping key "%s".', key);
end
result.(fieldName) = value;
end

function value = parseInlineValue(text, sourceName, lineNumber)
text = strtrim(text);
if isempty(text)
    value = [];
    return;
end
if any(text(1) == ['&', '*', '!'])
    parseError(sourceName, lineNumber, 'YAML anchors, aliases, and tags are not supported.');
end

if text(1) == '[' || text(1) == '{'
    [value, position] = parseFlowValue(text, 1, sourceName, lineNumber);
    position = skipSpaces(text, position);
    if position <= length(text)
        parseError(sourceName, lineNumber, 'Unexpected content after flow value.');
    end
elseif text(1) == '''' || text(1) == '"'
    [value, position] = parseQuoted(text, 1, sourceName, lineNumber);
    position = skipSpaces(text, position);
    if position <= length(text)
        parseError(sourceName, lineNumber, 'Unexpected content after quoted scalar.');
    end
else
    value = parsePlainScalar(text, sourceName, lineNumber);
end
end

function [value, position] = parseFlowValue(text, position, sourceName, lineNumber)
position = skipSpaces(text, position);
if position > length(text)
    parseError(sourceName, lineNumber, 'Expected a value in flow collection.');
end
character = text(position);
switch character
    case '['
        [value, position] = parseFlowSequence(text, position, sourceName, lineNumber);
    case '{'
        [value, position] = parseFlowMapping(text, position, sourceName, lineNumber);
    case {'''', '"'}
        [value, position] = parseQuoted(text, position, sourceName, lineNumber);
    otherwise
        startPosition = position;
        while position <= length(text) && ~any(text(position) == [',', ']', '}'])
            position = position + 1;
        end
        token = strtrim(text(startPosition:position - 1));
        value = parsePlainScalar(token, sourceName, lineNumber);
end
end

function [result, position] = parseFlowSequence(text, position, sourceName, lineNumber)
position = skipSpaces(text, position + 1);
result = cell(0, 1);
if position <= length(text) && text(position) == ']'
    result = [];
    position = position + 1;
    return;
end
while true
    [value, position] = parseFlowValue(text, position, sourceName, lineNumber);
    result{end + 1, 1} = value; %#ok<AGROW>
    position = skipSpaces(text, position);
    if position > length(text)
        parseError(sourceName, lineNumber, 'Unterminated flow sequence.');
    elseif text(position) == ']'
        position = position + 1;
        return;
    elseif text(position) ~= ','
        parseError(sourceName, lineNumber, 'Expected a comma or ] in flow sequence.');
    end
    position = skipSpaces(text, position + 1);
    if position <= length(text) && text(position) == ']'
        position = position + 1;
        return;
    end
end
end

function [result, position] = parseFlowMapping(text, position, sourceName, lineNumber)
position = skipSpaces(text, position + 1);
result = struct();
if position <= length(text) && text(position) == '}'
    position = position + 1;
    return;
end
while true
    if position > length(text)
        parseError(sourceName, lineNumber, 'Unterminated flow mapping.');
    end
    if any(text(position) == ['''', '"'])
        [key, position] = parseQuoted(text, position, sourceName, lineNumber);
    else
        keyStart = position;
        while position <= length(text) && text(position) ~= ':'
            position = position + 1;
        end
        key = strtrim(text(keyStart:position - 1));
    end
    position = skipSpaces(text, position);
    if position > length(text) || text(position) ~= ':'
        parseError(sourceName, lineNumber, 'Expected a colon in flow mapping.');
    end
    [value, position] = parseFlowValue(text, position + 1, sourceName, lineNumber);
    result = assignMappingField(result, key, value, sourceName, lineNumber);
    position = skipSpaces(text, position);
    if position > length(text)
        parseError(sourceName, lineNumber, 'Unterminated flow mapping.');
    elseif text(position) == '}'
        position = position + 1;
        return;
    elseif text(position) ~= ','
        parseError(sourceName, lineNumber, 'Expected a comma or } in flow mapping.');
    end
    position = skipSpaces(text, position + 1);
    if position <= length(text) && text(position) == '}'
        position = position + 1;
        return;
    end
end
end

function [value, position] = parseQuoted(text, position, sourceName, lineNumber)
quote = text(position);
startPosition = position;
position = position + 1;
if quote == ''''
    output = '';
    while position <= length(text)
        if text(position) == ''''
            if position < length(text) && text(position + 1) == ''''
                output = [output, '''']; %#ok<AGROW>
                position = position + 2;
            else
                value = output;
                position = position + 1;
                return;
            end
        else
            output = [output, text(position)]; %#ok<AGROW>
            position = position + 1;
        end
    end
else
    escaped = false;
    while position <= length(text)
        if escaped
            escaped = false;
        elseif text(position) == '\'
            escaped = true;
        elseif text(position) == '"'
            token = text(startPosition:position);
            try
                value = jsondecode(token);
            catch exception
                parseError(sourceName, lineNumber, ...
                    'Invalid double-quoted scalar: %s', exception.message);
            end
            position = position + 1;
            return;
        end
        position = position + 1;
    end
end
parseError(sourceName, lineNumber, 'Unterminated quoted scalar.');
end

function position = skipSpaces(text, position)
while position <= length(text) && isspace(text(position))
    position = position + 1;
end
end

function value = parsePlainScalar(token, sourceName, lineNumber)
token = strtrim(token);
if isempty(token)
    parseError(sourceName, lineNumber, 'Empty flow scalar.');
end
lowerToken = lower(token);
if strcmp(lowerToken, 'true')
    value = true;
elseif strcmp(lowerToken, 'false')
    value = false;
elseif any(strcmp(lowerToken, {'null', '~'}))
    parseError(sourceName, lineNumber, 'YAML null values are not supported. Use [] instead.');
elseif any(strcmp(lowerToken, {'.nan', '+.nan', '-.nan'}))
    value = NaN;
elseif any(strcmp(lowerToken, {'.inf', '+.inf'}))
    value = Inf;
elseif strcmp(lowerToken, '-.inf')
    value = -Inf;
elseif ~isempty(regexp(token, ...
        '^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$', 'once'))
    value = str2double(token);
elseif startsWith(token, '0x') || startsWith(token, '0X')
    value = double(hex2dec(token(3:end)));
else
    value = token;
end
end

function result = makeMatrices(value, makeOrdinaryArrays)
if isstruct(value)
    result = struct();
    fieldNames = fieldnames(value);
    for index = 1:numel(fieldNames)
        fieldName = fieldNames{index};
        result.(fieldName) = makeMatrices(value.(fieldName), makeOrdinaryArrays);
    end
elseif iscell(value)
    if canFormMatrix(value)
        rows = cellfun(@cell2mat, value, 'UniformOutput', false);
        rows = cellfun(@(row) reshape(row, 1, []), rows, 'UniformOutput', false);
        matrix = vertcat(rows{:});
        if makeOrdinaryArrays
            result = matrix;
        else
            result = num2cell(matrix);
        end
    elseif isempty(value)
        result = [];
    else
        result = cell(1, numel(value));
        for index = 1:numel(value)
            result{index} = makeMatrices(value{index}, makeOrdinaryArrays);
        end
    end
else
    result = value;
end
end

function result = canFormMatrix(value)
result = isvector(value) && ~isempty(value) && all(cellfun(@iscell, value));
if ~result
    return;
end
result = all(cellfun(@isvector, value));
if ~result
    return;
end
rowLength = numel(value{1});
result = all(cellfun(@numel, value) == rowLength);
if ~result
    return;
end
for index = 1:numel(value)
    row = value{index};
    if isempty(row)
        result = false;
        return;
    end
    scalarItems = all(cellfun(@isscalar, row));
    homogeneous = all(cellfun(@isnumeric, row)) || ...
        all(cellfun(@islogical, row)) || all(cellfun(@isstruct, row));
    if ~(scalarItems && homogeneous)
        result = false;
        return;
    end
    firstClass = class(row{1});
    if ~all(cellfun(@(item) strcmp(class(item), firstClass), row))
        result = false;
        return;
    end
end
end

function result = substituteDictionary(value, dictionary)
if isstruct(value)
    result = value;
    fieldNames = fieldnames(value);
    for index = 1:numel(fieldNames)
        fieldName = fieldNames{index};
        result.(fieldName) = substituteDictionary(value.(fieldName), dictionary);
    end
elseif iscell(value)
    result = value;
    for index = 1:numel(value)
        result{index} = substituteDictionary(value{index}, dictionary);
    end
elseif ischar(value) && isfield(dictionary, value)
    result = dictionary.(value);
else
    result = value;
end
end

function parseError(sourceName, lineNumber, message, varargin)
detail = sprintf(message, varargin{:});
error('AERIAL:YAML:ParseError', '%s:%d: %s', sourceName, lineNumber, detail);
end
