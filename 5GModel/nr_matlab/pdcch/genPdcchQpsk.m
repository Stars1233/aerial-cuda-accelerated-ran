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

function [x_qpsk,x_rm_bytes, c_scram_int] = genPdcchQpsk(payload, Npayload, rntiBits, rntiCrc, dmrsId, aggrL, testModel)

A = Npayload;                 % control channel payload size (bits)

%%
%SETUP
% from case DL_ctrl-TC2003
x = payload;
% x = [1 0 0 0 0 0 0 1 1 0 0 1 1 0 0 1 0 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0]';

if testModel % bypass CRC/polar/rateMatch
    x_rm = x;
    E = 2*9*6;
else
    % derive sizes:
    K = A + 24;        % number of pdcch payload + crc bits
    E = 2*9*6*aggrL;    % number of pdcch tx bits ( 2bits/QPSK * 9QPSK/REG * 6REG/CCE * nCCE)
    
    %%
    %START
    
    % step 1:
    x_crc = add_pdcch_crc(x,rntiCrc);
    
    % step 2:
    [x_encoded,N] = polar_encode(x_crc,K,E);
    
    % step 3:
    x_rm = polar_rate_match(x_encoded,N,K,E);
end
% Pad x_rm to a multiple of 8 bits so bit2int can pack full bytes
rem8 = mod(length(x_rm), 8);
x_rm_temp = x_rm;
if rem8 ~= 0
    x_rm_temp = [x_rm; zeros(8 - rem8, 1)];
end
x_rm_bytes = bit2int(x_rm_temp, 8, false);
% step 4:
[x_scram, c_scram] = pdcch_scrambling(x_rm,E,rntiBits,dmrsId);
rem32 = mod(length(c_scram), 32);
c_scram_temp = c_scram;
if rem32 ~= 0
    c_scram_temp = [c_scram; zeros(32 - rem32, 1)];
end
c_scram_int = bit2int(c_scram_temp, 32, true);

% step 5:
x_qpsk = qpsk_modulate(x_scram,E);

end

function c = add_pdcch_crc(x,rnti)

% function adds crc bits to pdcch payload

%inputs:
% x    --> pdcch payload. Dim: A x 1
% rnti --> user rnti number

%outputs:
% c --> pdcch payload w/h appended crc bits. Dim: K x 1

%%
%START

% append "1's" to pdcch payload:
x_app = [ones(24,1) ; x]; 

% compute crc bits:
[~,crc_bits] = add_CRC_LUT(x_app,'24C');

% scramble crc bits:
rnti_bits = flip(int2bin(rnti));
crc_bits(end - 16 + 1 : end) = xor(crc_bits(end - 16 + 1 : end), rnti_bits);

% append crc bits:
c = [x ; crc_bits];

end
