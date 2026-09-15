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

function [b,c] = pdcch_scrambling(e,E,rnti,N_id)

%function performs pdcch scrambling

%inputs:
% e    --> rate matched bits. Dim: E x 1
% E    --> number of transmit bits
% rnti --> users rnti number
% N_id --> physical cell id

%outputs:
% b    --> scrambled bits. Dim: E x 1

%%
%START

%Step 1: compute seed
c_init = mod(rnti*2^16 + N_id,2^31);

%Step 2: compute Gold sequence
c = build_Gold_sequence(c_init,E);

%Step 3: scramble
b = xor(c,e);

end







