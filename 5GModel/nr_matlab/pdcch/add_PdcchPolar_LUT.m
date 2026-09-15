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
 
clear all; close all; fclose all;

%% Range of A and E in Polar of PDCCH
A_min = 12;     % 38.212 7.3.1 DCI payload bits (before CRC)
A_max = 140;    % 38.212 5.3.1.1 K_IL_MAX=164, L=24 => A_max=140
L = 24;
list_E = [108, 216, 432, 864, 1728]; % AL = [1,2,4,8,16]

%% Create cidx to udix table
K_min = A_min + L;
K_max = A_max + L;
num_AL = length(list_E);
num_tot_cfgs = (K_max - K_min + 1) * num_AL;
num_tot_elems = sum(K_min : K_max) * num_AL;
num_bytes_per_elem = 2; % use uint16 because N_MAX = 512
cidx2uidx = uint16(zeros(num_tot_elems, 1));
offset = 0;
num_invalid_cfg = 0;
num_valid_cfg = 0;
num_invalid_elems = 0;
for K = K_min : K_max
    for E = list_E
        c = zeros(K, 1);
        if K <= E
            [d, N, dbg] = polar_encode(c,K,E);
            valid_cfg = true;
            num_valid_cfg = num_valid_cfg + 1;
        else
            valid_cfg = false;
            num_invalid_cfg = num_invalid_cfg + 1;
            num_invalid_elems = num_invalid_elems + K;
        end
        if valid_cfg
            tmp = dbg.cIdx2uIdx; assert(length(tmp) == K);
            cidx2uidx(offset + (1:K)) = tmp;
        end
        offset = offset + K;
    end
end

%% print out memory consumption
num_bytes_in_table = num_tot_elems * num_bytes_per_elem;
disp(num2str([num_tot_elems K_min K_max num_bytes_in_table / 1024], 'PDCCH Polar LUT: cidx2uidx table generated: %d entries for K = %d..%d. Memory %.1f KiB.'));
disp(num2str([num_valid_cfg num_invalid_cfg], '(%d valid, %d invalid (K,E) configs.)'));
invalid_percent = 100 * num_invalid_elems / num_tot_elems;
disp(num2str([num_tot_elems num_invalid_elems invalid_percent], '(total %d elem (%d invalid, %.0f percent)'));
assert(offset == num_tot_elems);
assert(num_valid_cfg + num_invalid_cfg == num_tot_cfgs);
assert(whos('cidx2uidx').bytes == num_bytes_in_table);

%% write the header
fname = 'pdcch_polar_cidx2uidx_lut_cpu.h';
disp(['Writing ' fname ' at cuPHY/src/cuphy_channels']);
write_header(fname, cidx2uidx);

%% functions
function [] = write_header(fname, v)
fid = fopen(fname, 'w');
if fid < 0
    error('Could not open %s for writing.', fname);
end
N = 12;
fprintf(fid, 'static const uint16_t PDCCH_POLAR_CIDX2UIDX_LUT_CPU[PDCCH_POLAR_CIDX2UIDX_LUT_SIZE] = {\n');
fprintf(fid, '   ');
k = 0;
for i = 1:numel(v)
    fprintf(fid, '%3u', v(i));
    k = k + 1;
    if i < numel(v)
        fprintf(fid, ', ');
        if mod(k, N) == 0
            fprintf(fid, '\n   ');
        end
    end
end
fprintf(fid, '};\n');
fclose(fid);
fprintf('Wrote %s (%d values, %d per line).\n', fname, numel(v), N);
end