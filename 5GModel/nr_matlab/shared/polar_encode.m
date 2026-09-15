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

function [d,N,dbg] = polar_encode(c,K,E)

%function performs polar encoding

%inputs:
% c  --> input bit sequence. Dim: K x 1
% K  --> number of information bits
% E  --> number of transmit bits

%outputs:
% d --> polar encoded bits.Dim: N x 1
% N --> codeblock length
% dbg (optional) --> struct of intermediate values

debug = (nargout >= 3);

%%
%CODEBLOCK SIZE

%here we determine N, the number of encoded bits (also equal to the number of encoded bits)

if E <= (9/8)*2^(ceil(log2(E)) - 1) && (K/E < 9/16)
    n1 = ceil(log2(E)) - 1;
else
    n1 = ceil(log2(E));
end

R_min = 1/8;
n_min = 5;
n_max = 9;

n2 = ceil(log2(K / R_min));

n = max(min([n1 n2 n_max]),n_min);

N = 2^n;

%%
%INTERLEAVING
%(note: matlab indexing!)


%compute interleaving indicies:
load('P_IL_max.mat'); %table: 5.3.1.1-1
K_max = 164;
Pi_IL = zeros(K,1);
k = 1;

for m = 1 : K_max
    if P_IL_max(m) >= (K_max - K)
        Pi_IL(k) = P_IL_max(m) - (K_max - K);
        k = k + 1;
    end
end

%perform interleaving:
if debug
    dbg.c = c;    
end

c = c(Pi_IL + 1); %note: +1 b/c of matlab indexing

if debug
    dbg.cpIdx2cIdx = Pi_IL;
    dbg.cIdx2cpIdx = zeros(size(dbg.cpIdx2cIdx));
    for cpIdx = 0 : length(c) - 1
        cIdx = dbg.cpIdx2cIdx(cpIdx + 1);
        dbg.cIdx2cpIdx(cIdx + 1) = cpIdx;
    end
    dbg.c_prime = c;
end

%%
%ZERO PADDING

% STEP1: load the polar sequence
load('Q_N_max.mat');


% STEP2: remove indicies >= N, while maintaining order
Q_0N = Q_N_max(Q_N_max < N);


% STEP3: compute forbidden indicies
Q_FN_temp = [];
if (E < N)

    load('P1.mat');
    J = zeros(N,1);
    for n = 0 : (N - 1)
        i = floor(32*n / N);
        J(n+1) = P(i+1)*(N / 32) + mod(n,N/32);
    end

    if (K/E <= 7/16)

        for n = 0 : (N - E - 1)
            Q_FN_temp = [Q_FN_temp J(n+1)];
        end

        if (E >= 3*N/4)
            Q_FN_temp = [Q_FN_temp (0 : (ceil(3*N/4 - E/2) - 1)) ];
        else
            Q_FN_temp = [Q_FN_temp (0 : (ceil(9*N/16 - E/4) - 1)) ];
        end

    else

        for n = E : (N-1)
            Q_FN_temp = [Q_FN_temp J(n+1)];
        end

    end

end

% STEP4: remove forbidden indicies, while maintaining order
Q_IN_tmp = setdiff(Q_0N,Q_FN_temp,'stable');

% STEP 5: extract K most reliable indicies
Q_IN = Q_IN_tmp(end - K + 1 : end);

% perform zero padding:
d = zeros(N,1);

idx_logical = zeros(N,1);
idx_logical(Q_IN + 1) = 1;
idx_logical = logical(idx_logical);

d(idx_logical) = c;

if debug
    dbg.Q_IN = Q_IN;
    dbg.cpIdx2uIdx = sort(Q_IN, 'ascend');
    dbg.u = d;
    dbg.cIdx2uIdx = zeros(size(dbg.cIdx2cpIdx));
    for cIdx = 0:length(dbg.c) - 1
        cpIdx = dbg.cIdx2cpIdx(cIdx + 1);
        uIdx = dbg.cpIdx2uIdx(cpIdx + 1);
        dbg.cIdx2uIdx(cIdx + 1) = uIdx;
    end
    % sanity check
    cIdx = 0 : length(dbg.c) - 1;
    cpIdx = dbg.cIdx2cpIdx(cIdx + 1);
    uIdx = dbg.cIdx2uIdx(cIdx + 1);
    assert(all(dbg.c_prime(cpIdx + 1) == dbg.c(cIdx + 1)));
    assert(all(dbg.u(uIdx + 1) == dbg.c_prime(cpIdx + 1)));
    assert(all(dbg.u(uIdx + 1) == dbg.c(cIdx + 1)));
end

%%
%BUTTFERFLY XOR

for i = 0 : (n - 1) %loop over log2(N) - 1 stages

    s = 2^i;
    m = N / (2*s);

    %parallel start (N/2 parallel XORS)
    for j = 1 : m
        start_idx = 2*s*(j-1);

        for k = 1 : s
            d(start_idx + k) = xor(d(start_idx + k), d(start_idx + k + s) );
        end
    end
    %parallel end

end
