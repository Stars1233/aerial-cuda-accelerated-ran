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

function saveCarrierChanPars(h5File, SimCtrl, carrier, prachInd)
    if nargin < 4 || isempty(prachInd)
        prachInd = 0;
    end

    % carrier params
    carrier_pars.N_sc = uint16(carrier.N_sc);
    carrier_pars.N_FFT = uint16(carrier.Nfft);
    carrier_pars.N_bsAnt = uint16(SimCtrl.gNB.Nant);
    carrier_pars.id_slot = uint16(carrier.idxSlot);
    carrier_pars.id_subFrame = uint16(carrier.idxSubframe);
    carrier_pars.mu = uint8(carrier.mu);
    carrier_pars.cpType = uint8(carrier.CpType);
    carrier_pars.f_c = single(1/carrier.T_c);
    carrier_pars.f_samp = single(carrier.f_samp);
    carrier_pars.N_symbol_slot = uint16(carrier.N_symb_slot);
    carrier_pars.kappa_bits = uint16(log2(carrier.k_const));
    if (prachInd)
        carrier_pars.N_ueAnt = uint16(SimCtrl.UE{1}.Nant);
        carrier_pars.N_u_mu = uint32(carrier.N_u_mu);
        carrier_pars.N_samp_slot = uint32(carrier.N_samp_slot);
        carrier_pars.k_const = uint16(carrier.k_const);
    else
        carrier_pars.N_ueAnt = uint16(SimCtrl.UE{1}.Nant);
    end
    % chan params
    chan_type = SimCtrl.channel_info.Chan_UL{1}.type;

    if strcmp(chan_type,'CDL')
        chan_type = [chan_type SimCtrl.channel_info.Chan_UL{1}.DelayProfile(end)];
    end

    %% chanType: 0 - AWGN, 1 - TDL, 2 - CDL, 3 - P2P 
    %% delayProfile: 0 - A, 1 - B, 2 - C, 3 - D, 4 - E; default is 0 for AWGN and P2P and not applicable
    if contains(chan_type,'TDLA')
        chan_pars.delayProfile = uint8(0);
        chan_pars.chanType = uint8(1);
    elseif contains(chan_type,'TDLB')
        chan_pars.delayProfile = uint8(1);
        chan_pars.chanType = uint8(1);
    elseif contains(chan_type,'TDLC')
        chan_pars.delayProfile = uint8(2);
        chan_pars.chanType = uint8(1);
    elseif contains(chan_type,'TDLD')
        chan_pars.delayProfile = uint8(3);
        chan_pars.chanType = uint8(1);
    elseif contains(chan_type,'TDLE')
        chan_pars.delayProfile = uint8(4);
        chan_pars.chanType = uint8(1);
    elseif contains(chan_type,'CDLA')
        chan_pars.delayProfile = uint8(0);
        chan_pars.chanType = uint8(2);
    elseif contains(chan_type,'CDLB')
        chan_pars.delayProfile = uint8(1);
        chan_pars.chanType = uint8(2);
    elseif contains(chan_type,'CDLC')
        chan_pars.delayProfile = uint8(2);
        chan_pars.chanType = uint8(2);
    elseif contains(chan_type,'CDLD')
        chan_pars.delayProfile = uint8(3);
        chan_pars.chanType = uint8(2);
    elseif contains(chan_type,'CDLE')
        chan_pars.delayProfile = uint8(4);
        chan_pars.chanType = uint8(2);
    elseif contains(chan_type,'AWGN')
        chan_pars.delayProfile = uint8(0);
        chan_pars.chanType = uint8(0);
    elseif contains(chan_type,'P2P')
        chan_pars.delayProfile = uint8(0);
        chan_pars.chanType = uint8(3);
    end
    if strcmp(chan_type, 'AWGN') || strcmp(chan_type, 'P2P')
        chan_pars.delaySpread = 0;
        chan_pars.maxDopplerShift = 0;
    elseif (contains(chan_type, 'TDLA') || contains(chan_type, 'TDLB') || contains(chan_type, 'TDLC') || contains(chan_type, 'TDLD') || contains(chan_type, 'TDLE'))
        [~, DelaySpread, MaxDopplerShift, ~] = deriveChanPar_TDL(chan_type);
        chan_pars.delaySpread = single(DelaySpread);
        chan_pars.maxDopplerShift = single(MaxDopplerShift);
    else
        chan_pars.delaySpread = single(SimCtrl.channel_info.Chan_UL{1}.DelaySpread * 1e9);  % saving in nanoseconds
        chan_pars.maxDopplerShift = single(SimCtrl.channel_info.Chan_UL{1}.MaximumDopplerShift);
    end
    chan_pars.delay = single(SimCtrl.channel_info.Chan_UL{1}.delay);%
    chan_pars.CFO = single(SimCtrl.channel_info.Chan_UL{1}.CFO);%
    chan_pars.f_samp = single(carrier.f_samp);
    chan_pars.numBsAnt = uint16(carrier_pars.N_bsAnt);
    chan_pars.numUeAnt = uint16(carrier_pars.N_ueAnt);
    chan_pars.fBatch = uint32(15e3);%
    
    if (chan_pars.chanType == 1)
        chan_pars.useSimplifiedPdp = uint8(SimCtrl.channel_info.Chan_UL{1}.simplifiedDelayProfile);% % 1 for simplified pdp in 38.141, 0 for 38.901
        chan_pars.numPath = uint32(48);%
        tmp2 = regexp(chan_type,'(\w+)(\W+)(\d+)(\W+)(\w+)','tokens','once');
        if ~isempty(tmp2)
            assert(strcmp(tmp2{5},'Low')); % the current GPU TDL channel model only support Low correlation
        end
    elseif (chan_pars.chanType == 2)
        chan_pars.numRay = 20;
        % Convert MATLAB 3-parameter antenna config to cuPHY CDL 3GPP format:
        % AntSize: [M_g, N_g, M, N, P], AntSpacing: [d_g_h, d_g_v, d_h, d_v]
        chan_pars.ueAntSize = uint16(toCdlAntSize5(SimCtrl.channel_info.Chan_UL{1}.UE_AntArraySize));
        chan_pars.ueAntSpacing = single(toCdlAntSpacing4(SimCtrl.channel_info.Chan_UL{1}.UE_AntSpacing)); 
        chan_pars.ueAntPolarAngles = single(SimCtrl.channel_info.Chan_UL{1}.UE_AntPolarizationAngles); 
        chan_pars.ueAntPattern = findAntPatternMap(SimCtrl.channel_info.Chan_UL{1}.UE_AntPattern); 

        chan_pars.bsAntSize = uint16(toCdlAntSize5(SimCtrl.channel_info.Chan_UL{1}.gNB_AntArraySize));
        chan_pars.bsAntSpacing = single(toCdlAntSpacing4(SimCtrl.channel_info.Chan_UL{1}.gNB_AntSpacing)); 
        chan_pars.bsAntPolarAngles = single(SimCtrl.channel_info.Chan_UL{1}.gNB_AntPolarizationAngles);
        chan_pars.bsAntPattern = findAntPatternMap(SimCtrl.channel_info.Chan_UL{1}.gNB_AntPattern);

        chan_pars.vDirection = single([90 0]);  % single(SimCtrl.channel_info.Chan_UL{1}.v_direction); fixed v_direction, see genCDL.m
    end
    
    hdf5_write_nv(h5File, 'carrier_pars', carrier_pars);
    hdf5_write_nv_exp(h5File, 'chan_pars', chan_pars);
end

function antPatternMap = findAntPatternMap(antPattern)
    if strcmp(antPattern,'isotropic')
        antPatternMap = uint16(0);
    elseif strcmp(antPattern,'38.901')
        antPatternMap = uint16(1);
    end
end


function [DelayProfile, DelaySpread, MaximumDopplerShift, MIMOCorrelation] = deriveChanPar_TDL(chanType)
    % Regular expression to parse the input string
    tokens = regexp(chanType, '^TDL([A-E])(\d+)-(\d+)-(\w+)$', 'tokens', 'once');
    
    if ~isempty(tokens)
        % Extract Delay Profile (e.g., 'TDL-A', 'TDL-B')
        DelayProfile = ['TDL-', tokens{1}];
        
        % Convert Delay Spread (e.g., 30 ns -> 30)
        DelaySpread = str2double(tokens{2});
        
        % Extract Maximum Doppler Shift
        MaximumDopplerShift = str2double(tokens{3}); % Convert Doppler shift to numeric
        
        % Extract MIMO Correlation
        MIMOCorrelation = tokens{4}; % Low, Medium, High, etc.
    else
        % Handle invalid input format
        error('Invalid chanType format. Expected format: TDL<Profile><DelaySpread>-<DopplerShift>-<MIMOCorrelation>');
    end
end

function antSize5 = toCdlAntSize5(antSizeIn)
    antSizeIn = double(antSizeIn(:).');
    if numel(antSizeIn) == 5
        antSize5 = antSizeIn;
    elseif numel(antSizeIn) == 3
        % Legacy MATLAB format: [M, N, P] -> [M_g, N_g, M, N, P], with single panel groups.
        antSize5 = [1, 1, antSizeIn(1), antSizeIn(2), antSizeIn(3)];
    else
        error('Invalid antenna size length %d. Expected 3 ([M,N,P]) or 5 ([M_g,N_g,M,N,P]).', numel(antSizeIn));
    end
end

function antSpacing4 = toCdlAntSpacing4(antSpacingIn)
    antSpacingIn = double(antSpacingIn(:).');
    if numel(antSpacingIn) == 4
        antSpacing4 = antSpacingIn;
    elseif numel(antSpacingIn) == 2
        % Legacy MATLAB format: [d_v, d_h] -> [d_g_h, d_g_v, d_h, d_v], with single panel groups.
        antSpacing4 = [1.0, 1.0, antSpacingIn(2), antSpacingIn(1)];
    else
        error('Invalid antenna spacing length %d. Expected 2 or 4.', numel(antSpacingIn));
    end
end