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

%
if (~isdeployed)
    addpath(genpath('.'))
    if ~isfolder('yamlmatlab')
        if ismac
            [status, ~] = system('../aerial_mcore/scripts/download_yamlmatlab_mac.sh');
        elseif ispc
            % Bypass execution policy is used here to allow running unsigned scripts and not require user interaction.  
            % Script validates hash of downloaded code to prevent downloading unintended code
            [status, ~] = system('powershell -ExecutionPolicy Bypass -File "..\aerial_mcore\scripts\download_yamlmatlab_win.ps1"');
        else
            [status, ~] = system('../aerial_mcore/scripts/download_yamlmatlab.sh');
        end
        if status ~= 0
            error('Failed to download yamlmatlab package');
        else
            addpath('yamlmatlab');
        end
    end
end
