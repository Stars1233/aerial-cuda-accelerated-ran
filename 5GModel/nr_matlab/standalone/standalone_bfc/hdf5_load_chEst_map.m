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

function mapping = hdf5_load_chEst_map(filename)
    % HDF5_LOAD_CHEST_MAP Get channel estimate mapping from HDF5 file
    % 
    % Usage:
    %   mapping = hdf5_load_chEst_map('filename.h5')
    %
    % Inputs:
    %   filename  - Path to HDF5 file
    %
    % Returns:
    %   mapping   - Structure with fields:
    %               .pdu_indices   - Array of PDU indices that have channel estimates
    %               .rntis         - Array of corresponding RNTI values
    %               .hest_fields   - Cell array of IND*_Hest field names
    %               .startValidPrg - Array of starting valid PRG indices (for padding)
    %               .nValidPrg     - Array of number of valid PRGs (SRS estimated PRGs)
    %
    % This function:
    % 1. Lists all field names in the HDF5 file
    % 2. Filters to find IND*_Hest fields (exact match, not HestNorm or HestToL2)
    % 3. For each IND*_Hest field, loads the corresponding PDU*.RNTI
    % 4. Computes startValidPrg and nValidPrg from PDU parameters
    % 5. Returns mapping information for nrSimulator to use
    
    % Get file info to list all datasets without loading data
    file_info = h5info(filename);
    dataset_names = {file_info.Datasets.Name};
    
    % Filter to find IND*_Hest fields (exact match, exclude HestNorm, HestToL2, etc.)
    % Pattern: starts with 'IND', followed by digits, ends with '_Hest'
    hest_pattern = '^IND(\d+)_Hest$';
    hest_fields = {};
    hest_numbers = [];
    
    for i = 1:length(dataset_names)
        dataset_name = dataset_names{i};
        tokens = regexp(dataset_name, hest_pattern, 'tokens');
        if ~isempty(tokens)
            hest_fields{end+1} = dataset_name;
            hest_numbers(end+1) = str2double(tokens{1}{1});
        end
    end
    
    % Initialize output arrays
    pdu_indices = [];
    rntis = [];
    valid_hest_fields = {};
    startValidPrgs = [];
    nValidPrgs = [];
    
    % SRS bandwidth table from TS 38.211 Table 6.4.1.4.3-1
    % Format: [m_SRS_0, N_0, m_SRS_1, N_1, m_SRS_2, N_2, m_SRS_3, N_3] for each configIdx (0-63)
    SRS_BW_TABLE = getSrsBwTable();
    
    % Open HDF5 file once for all operations
    fileID = H5F.open(filename, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');
    
    try
        % Process each IND*_Hest field to get RNTI mapping
        for i = 1:length(hest_fields)
            hest_field = hest_fields{i};
            ind_number = hest_numbers(i);
            
            % Construct corresponding PDU and HestToL2 field names
            pdu_field = sprintf('PDU%d', ind_number);
            hestToL2_field = sprintf('IND%d_HestToL2', ind_number);
            
            % Check if corresponding PDU dataset exists
            if ismember(pdu_field, dataset_names)
                try
                    % Open PDU dataset and read
                    pduID = H5D.open(fileID, ['/' pdu_field]);
                    pdu_data = H5D.read(pduID);
                    H5D.close(pduID);
                    
                    % Extract RNTI if it exists
                    if isfield(pdu_data, 'RNTI')
                        rnti = pdu_data.RNTI;
                        
                        % Store mapping information
                        pdu_indices(end+1) = ind_number;
                        rntis(end+1) = rnti;
                        valid_hest_fields{end+1} = hest_field;
                        
                        % Compute startValidPrg and nValidPrg from PDU parameters
                        [startPrg, nPrg] = computeSrsValidRange(pdu_data, SRS_BW_TABLE, ...
                            dataset_names, fileID, hestToL2_field);
                        startValidPrgs(end+1) = startPrg;
                        nValidPrgs(end+1) = nPrg;
                    end
                catch ME
                    fprintf('Error processing %s: %s\n', hest_field, ME.message);
                end
            end
        end
    catch ME
        % Ensure file is closed even if there's an error
        H5F.close(fileID);
        rethrow(ME);
    end
    
    % Close the HDF5 file
    H5F.close(fileID);
    
    % Create output structure
    mapping = struct();
    mapping.pdu_indices = pdu_indices;
    mapping.rntis = rntis;
    mapping.hest_fields = valid_hest_fields;
    mapping.startValidPrg = startValidPrgs;
    mapping.nValidPrg = nValidPrgs;
end


function [startValidPrg, nValidPrg] = computeSrsValidRange(pdu_data, SRS_BW_TABLE, dataset_names, fileID, hestToL2_field)
    % Compute startValidPrg and nValidPrg from PDU parameters
    %
    % startValidPrg: floor(hopStartPrb / prgSize)
    % nValidPrg: from HestToL2 size or computed from nPrbsPerHop / prgSize
    
    % Extract PDU parameters
    configIdx = double(pdu_data.configIndex);
    bandwidthIdx = double(pdu_data.bandwidthIndex);
    frequencyShift = double(pdu_data.frequencyShift);
    frequencyPosition = double(pdu_data.frequencyPosition);
    frequencyHopping = double(pdu_data.frequencyHopping);
    prgSizeL2 = pdu_data.prgSize;
    prgSize = double(pdu_data.prgSize);
    if prgSizeL2 > 4 
        prgSize = double(2.0);
    end
    
    % Get nPrbsPerHop from SRS BW table
    % SRS_BW_TABLE format: column 2*b+1 is m_SRS_b, column 2*b+2 is N_b
    nPrbsPerHop = SRS_BW_TABLE(configIdx + 1, 2 * bandwidthIdx + 1);
    
    % Compute hopStartPrb (first hop starting PRB)
    % Based on TS 38.211 Section 6.4.1.4.3
    hopStartPrb = frequencyShift;
    for b = 0 : bandwidthIdx
        m_SRS_b = SRS_BW_TABLE(configIdx + 1, 2*b + 1);
        Nb = SRS_BW_TABLE(configIdx + 1, 2*b + 2);
        if frequencyHopping >= bandwidthIdx
            nb = mod(floor(4 * frequencyPosition / m_SRS_b), Nb);
        else
            if b <= frequencyHopping
                nb = mod(floor(4 * frequencyPosition / m_SRS_b), Nb);
            else
                % For hopping case, use slot 0 position (n_SRS = 0)
                nb = mod(floor(4 * frequencyPosition / m_SRS_b), Nb);
            end
        end
        hopStartPrb = hopStartPrb + m_SRS_b * nb;
    end
    
    % Compute startValidPrg
    startValidPrg = floor(hopStartPrb / prgSize);
    
    % Get nValidPrg from HestToL2 size if available, otherwise compute
    if ismember(hestToL2_field, dataset_names) && (prgSizeL2 <= 4)
        try
            % Get dataset info without loading data
            dsetID = H5D.open(fileID, ['/' hestToL2_field]);
            spaceID = H5D.get_space(dsetID);
            [~, dims, ~] = H5S.get_simple_extent_dims(spaceID);
            H5S.close(spaceID);
            H5D.close(dsetID);
            % HestToL2 dimensions are [nUeAnt, nRxAnt, nValidPrg]
            nValidPrg = dims(3);
        catch
            % Fallback to computed value
            nValidPrg = floor(nPrbsPerHop / prgSize);
        end
    else
        % Compute from nPrbsPerHop
        nValidPrg = floor(nPrbsPerHop / prgSize);
    end
end


function SRS_BW_TABLE = getSrsBwTable()
    % SRS bandwidth configuration table from TS 38.211 Table 6.4.1.4.3-1
    % Each row corresponds to configIdx 0-63
    % Columns: [m_SRS_0, N_0, m_SRS_1, N_1, m_SRS_2, N_2, m_SRS_3, N_3]
    SRS_BW_TABLE = [
        4,   1,   4,   1,   4,   1,   4,   1;   % 0
        8,   1,   4,   2,   4,   1,   4,   1;   % 1
        12,  1,   4,   3,   4,   1,   4,   1;   % 2
        16,  1,   4,   4,   4,   1,   4,   1;   % 3
        16,  1,   8,   2,   4,   2,   4,   1;   % 4
        20,  1,   4,   5,   4,   1,   4,   1;   % 5
        24,  1,   4,   6,   4,   1,   4,   1;   % 6
        24,  1,   12,  2,   4,   3,   4,   1;   % 7
        28,  1,   4,   7,   4,   1,   4,   1;   % 8
        32,  1,   16,  2,   8,   2,   4,   2;   % 9
        36,  1,   12,  3,   4,   3,   4,   1;   % 10
        40,  1,   20,  2,   4,   5,   4,   1;   % 11
        48,  1,   16,  3,   8,   2,   4,   2;   % 12
        48,  1,   24,  2,   12,  2,   4,   3;   % 13
        52,  1,   4,  13,   4,   1,   4,   1;   % 14
        56,  1,   28,  2,   4,   7,   4,   1;   % 15
        60,  1,   20,  3,   4,   5,   4,   1;   % 16
        64,  1,   32,  2,   16,  2,   4,   4;   % 17
        72,  1,   24,  3,   12,  2,   4,   3;   % 18
        72,  1,   36,  2,   12,  3,   4,   3;   % 19
        76,  1,   4,  19,   4,   1,   4,   1;   % 20
        80,  1,   40,  2,   20,  2,   4,   5;   % 21
        88,  1,   44,  2,   4,  11,   4,   1;   % 22
        96,  1,   32,  3,   16,  2,   4,   4;   % 23
        96,  1,   48,  2,   24,  2,   4,   6;   % 24
        104, 1,   52,  2,   4,  13,   4,   1;   % 25
        112, 1,   56,  2,   28,  2,   4,   7;   % 26
        120, 1,   60,  2,   20,  3,   4,   5;   % 27
        120, 1,   40,  3,   8,   5,   4,   2;   % 28
        120, 1,   24,  5,   12,  2,   4,   3;   % 29
        128, 1,   64,  2,   32,  2,   4,   8;   % 30
        128, 1,   64,  2,   16,  4,   4,   4;   % 31
        128, 1,   16,  8,   8,   2,   4,   2;   % 32
        132, 1,   44,  3,   4,  11,   4,   1;   % 33
        136, 1,   68,  2,   4,  17,   4,   1;   % 34
        144, 1,   72,  2,   36,  2,   4,   9;   % 35
        144, 1,   48,  3,   24,  2,   12,  2;   % 36
        144, 1,   48,  3,   16,  3,   4,   4;   % 37
        144, 1,   36,  4,   12,  3,   4,   3;   % 38
        152, 1,   76,  2,   4,  19,   4,   1;   % 39
        160, 1,   80,  2,   40,  2,   4,  10;   % 40
        160, 1,   80,  2,   20,  4,   4,   5;   % 41
        160, 1,   32,  5,   16,  2,   4,   4;   % 42
        168, 1,   84,  2,   28,  3,   4,   7;   % 43
        176, 1,   88,  2,   44,  2,   4,  11;   % 44
        184, 1,   92,  2,   4,  23,   4,   1;   % 45
        192, 1,   96,  2,   48,  2,   4,  12;   % 46
        192, 1,   96,  2,   24,  4,   4,   6;   % 47
        192, 1,   64,  3,   16,  4,   4,   4;   % 48
        192, 1,   24,  8,   8,   3,   4,   2;   % 49
        208, 1,  104,  2,   52,  2,   4,  13;   % 50
        216, 1,  108,  2,   36,  3,   4,   9;   % 51
        224, 1,  112,  2,   56,  2,   4,  14;   % 52
        240, 1,  120,  2,   60,  2,   4,  15;   % 53
        240, 1,  80,   3,   20,  4,   4,   5;   % 54
        240, 1,  48,   5,   16,  3,   8,   2;   % 55
        240, 1,  24,  10,   12,  2,   4,   3;   % 56
        256, 1,  128,  2,   64,  2,   4,  16;   % 57
        256, 1,  128,  2,   32,  4,   4,   8;   % 58
        256, 1,  16,  16,   8,   2,   4,   2;   % 59
        264, 1,  132,  2,   44,  3,   4,  11;   % 60
        272, 1,  136,  2,   68,  2,   4,  17;   % 61
        272, 1,  68,   4,   4,  17,   4,   1;   % 62
        272, 1,  16,  17,   8,   2,   4,   2;   % 63
    ];
end
