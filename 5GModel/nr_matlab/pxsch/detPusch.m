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

function [pusch_payload_list, tbErrList, cbErrList, harqState, rxDataList] = detPusch(pduList, table, carrier, Xtf, harqState, srsState, Chan_UL, interfChan_UL, Xtf_noNoise)

global SimCtrl;

puschTable = table;
nPdu = length(pduList);

PuschParamsList = [];
pusch_payload_list = [];
rxDataList = [];
tbErrList = [];
cbErrList = [];

lastPdu = pduList{nPdu};
nUeg = lastPdu.idxUeg + 1;
currentUeg = 0;
for idxUeg = 1:nUeg
    UegList{idxUeg}.idxPdu = [];
    UegList{idxUeg}.nlUeg = 0;
end
for idxPdu = 1:nPdu
    pdu = pduList{idxPdu};
    if pdu.idxUeg == currentUeg
        UegList{currentUeg+1}.idxPdu = [UegList{currentUeg+1}.idxPdu, idxPdu-1];
        UegList{currentUeg+1}.nlUeg = UegList{currentUeg+1}.nlUeg + pdu.nrOfLayers;
    else
        UegList{currentUeg+1+1}.idxPdu = [UegList{currentUeg+1+1}.idxPdu, idxPdu-1];
        UegList{currentUeg+1+1}.nlUeg = UegList{currentUeg+1+1}.nlUeg + pdu.nrOfLayers;
        currentUeg = currentUeg + 1;
    end
end

for idxUeg = 1:nUeg
    nUeInUeg = length(UegList{idxUeg}.idxPdu);
    PuschParamsUeg{idxUeg}.PuschParams = [];
    for idxUeInUeg = 1:nUeInUeg
        idxPdu = UegList{idxUeg}.idxPdu(idxUeInUeg);
        pdu = pduList{idxPdu+1};
        % derive configuration
        dmrs    = derive_dmrs_main(pdu, puschTable);
        alloc   = derive_alloc_main(pdu, dmrs);
        coding  = derive_coding_main(pdu, alloc, puschTable);

        % load parameters
        PuschParams.CRC = coding.CRC;
        PuschParams.C = coding.C;
        PuschParams.K = coding.K;
        PuschParams.F = coding.F;
        PuschParams.K_prime = coding.K_prime;
        PuschParams.BGN = coding.BGN;   %1 or 2. Indicates which base graph used
        PuschParams.i_LS = coding.i_LS; %lifting set index
        PuschParams.Zc = coding.Zc;     %lifting size
        PuschParams.qam = coding.qam;
        PuschParams.codeRate = coding.codeRate;        
        PuschParams.rvIdx = coding.rvIdx;
        PuschParams.nl = alloc.nl;
        PuschParams.N_data = alloc.N_data;
        PuschParams.N_id = alloc.dataScramblingId;
        PuschParams.n_rnti = alloc.RNTI;
        PuschParams.qamstr = coding.qamstr;
        PuschParams.portIdx = alloc.portIdx;         % user's antenna ports (matlab 1 indexing). Dim: nl x 1
        PuschParams.n_scid = alloc.SCID;             % 0 or 1. User's dmrs scrambling id
        PuschParams.nPrb = alloc.nPrb;               % number of prbs in allocation
        PuschParams.startPrb = alloc.startPrb;       % starting prb of allocation
        PuschParams.prgSize = pdu.prgSize;
        PuschParams.enable_prg_chest = pdu.enable_prg_chest;
        PuschParams.Nf_data = alloc.Nf_data;         % number of data subcarriers in allocation
        PuschParams.Nt_data = alloc.Nt_data;         % number of data symbols in allocation
        PuschParams.symIdx_data = alloc.symIdx_data; % indicies of data symbols. Dim: Nt_data x 1
        PuschParams.slotNumber = mod(carrier.idxSlotInFrame, carrier.N_slot_frame_mu);
        if isfield(SimCtrl.genTV,'fakeSlotNumber')
            PuschParams.slotNumber = SimCtrl.genTV.fakeSlotNumber;
            SimCtrl.alg.staticPuschSlotNum  = SimCtrl.genTV.fakeSlotNumber; % For FAPI TV
        end
        PuschParams.Nf = carrier.N_sc;%12*PuschParams.nPrb;%carrier.N_sc;
        PuschParams.Nt = carrier.N_symb_slot;
        PuschParams.N_dmrs_id = dmrs.DmrsScramblingId;
        PuschParams.symIdx_dmrs = dmrs.symIdx_dmrs;  % Indicies of dmrs symbols (matlab 1 indexing). Dim: Nt_dmrs x 1
        PuschParams.Nt_dmrs = dmrs.Nt_dmrs;          % number of dmrs symbols
        PuschParams.energy = dmrs.energy;            % dmrs energy
        PuschParams.Nref = coding.Nref;
        PuschParams.I_LBRM = coding.I_LBRM;
        PuschParams.maxLayers = coding.maxLayers;
        PuschParams.maxQm = coding.maxQm;
        PuschParams.n_PRB_LBRM = coding.n_PRB_LBRM;
        PuschParams.enableCfoEstimation = SimCtrl.alg.enableCfoEstimation;
        PuschParams.enableCfoCorrection = SimCtrl.alg.enableCfoCorrection;
        PuschParams.enableWeightedAverageCfo = SimCtrl.alg.enableWeightedAverageCfo;
        PuschParams.enableToEstimation = SimCtrl.alg.enableToEstimation;
        PuschParams.enableToCorrection = SimCtrl.alg.enableToCorrection;
        PuschParams.enableSinrMeas = SimCtrl.alg.enableSinrMeas;
        PuschParams.enableIrc = SimCtrl.alg.enableIrc;
        PuschParams.Rww_regularizer_val = SimCtrl.alg.Rww_regularizer_val; 
        PuschParams.enable_nCov_shrinkage = SimCtrl.alg.enable_nCov_shrinkage;
        PuschParams.nCov_shrinkage_method = SimCtrl.alg.nCov_shrinkage_method;
        PuschParams.enable_use_genie_nCov = SimCtrl.alg.enable_use_genie_nCov;
        PuschParams.genie_nCov_method = SimCtrl.alg.genie_nCov_method;
        PuschParams.enableNoiseEstForZf = SimCtrl.alg.enableNoiseEstForZf;
        PuschParams.enable_sphere_decoder = SimCtrl.alg.enable_sphere_decoder;
        PuschParams.sphere_decoder = init_sphere_decoder(SimCtrl,table,PuschParams.qamstr);        
        PuschParams.TdiMode = SimCtrl.alg.TdiMode;
        if SimCtrl.usePuschRxForPdsch
            PuschParams.DTXthreshold = 1;
        else
            PuschParams.DTXthreshold = pdu.DTXthreshold;
        end
        PuschParams.idxUeg = idxUeg-1;
        PuschParams.idxUE = pdu.idxUE;
        PuschParams.nlUeg = UegList{idxUeg}.nlUeg;
        PuschParams.numDmrsCdmGrpsNoData = pdu.numDmrsCdmGrpsNoData;
        if SimCtrl.usePuschRxForPdsch
            PuschParams.enableTfPrcd = 0;
            PuschParams.prcdBf = 0;
            PuschParams.digBFInterfaces = alloc.nl;
            PuschParams.puschIdentity = 0;
        else
            PuschParams.enableTfPrcd = (pdu.TransformPrecoding == 0);
            PuschParams.prcdBf = pdu.prcdBf;
            PuschParams.digBFInterfaces = pdu.digBFInterfaces;
            PuschParams.puschIdentity = pdu.puschIdentity;
        end
        PuschParams.N_slot_frame = carrier.N_slot_frame_mu;
        PuschParams.N_symb_slot = carrier.N_symb_slot;

        if SimCtrl.usePuschRxForPdsch
            PuschParams.groupOrSequenceHopping = 0;

            %Parameters for UCI on PUSCH
            PuschParams.pduBitmap = 1;
            PuschParams.alphaScaling = 3;
            PuschParams.betaOffsetHarqAck = 0;
            PuschParams.betaOffsetCsi1 = 0;
            PuschParams.betaOffsetCsi2 = 0;
            PuschParams.harqAckBitLength = 0;
            PuschParams.csiPart1BitLength = 0;
            PuschParams.flagCsiPart2            = 0;
            PuschParams.nCsi2Reports            = 0;
            PuschParams.calcCsi2Size_csi2MapIdx = 0;
            PuschParams.calcCsi2Size_nPart1Prms = 0;
            PuschParams.calcCsi2Size_prmOffsets = 0;
            PuschParams.calcCsi2Size_prmSizes   = 0;
            PuschParams.rankBitOffset           = 0;
            PuschParams.rankBitSize             = 2;
        else
            PuschParams.groupOrSequenceHopping = pdu.groupOrSequenceHopping;

            %Parameters for UCI on PUSCH
            PuschParams.pduBitmap = pdu.pduBitmap;
            PuschParams.alphaScaling = pdu.alphaScaling;
            PuschParams.betaOffsetHarqAck = pdu.betaOffsetHarqAck;
            PuschParams.betaOffsetCsi1 = pdu.betaOffsetCsi1;
            PuschParams.betaOffsetCsi2 = pdu.betaOffsetCsi2;
            PuschParams.harqAckBitLength = pdu.harqAckBitLength;
            PuschParams.csiPart1BitLength = pdu.csiPart1BitLength;
            PuschParams.flagCsiPart2            = pdu.flagCsiPart2;
            PuschParams.nCsi2Reports            = pdu.nCsi2Reports;
            PuschParams.calcCsi2Size_csi2MapIdx = pdu.calcCsi2Size_csi2MapIdx;
            PuschParams.calcCsi2Size_nPart1Prms = pdu.calcCsi2Size_nPart1Prms;
            PuschParams.calcCsi2Size_prmOffsets = pdu.calcCsi2Size_prmOffsets;
            PuschParams.calcCsi2Size_prmSizes   = pdu.calcCsi2Size_prmSizes;
            PuschParams.rankBitOffset           = pdu.rankBitOffset;
            PuschParams.rankBitSize             = pdu.rankBitSize;
        end

        % additional parameters for cuPHY API
        PuschParams.StartSymbolIndex = pdu.StartSymbolIndex;
        PuschParams.NrOfSymbols = pdu.NrOfSymbols;
        PuschParams.mcsTable = pdu.mcsTable;
        PuschParams.mcsIndex = pdu.mcsIndex;
        PuschParams.TBS = coding.TBS;
        PuschParams.dmrsConfigType = pdu.dmrsConfigType;
        PuschParams.AdditionalPosition = dmrs.AdditionalPosition;
        PuschParams.maxLength = dmrs.maxLength;
        PuschParams.n_scid = dmrs.n_scid;
        PuschParams.delta_f = carrier.delta_f;
        if SimCtrl.usePuschRxForPdsch
            PuschParams.newDataIndicator = 1;
            PuschParams.harqProcessID = 0;
        else
            PuschParams.newDataIndicator = pdu.newDataIndicator;
            PuschParams.harqProcessID = pdu.harqProcessID;
        end
        if PuschParams.enableWeightedAverageCfo == 1
            PuschParams.foForgetCoeff = pdu.foForgetCoeff;
            PuschParams.foCompensationBuffer = pdu.foCompensationBuffer;
        else
            PuschParams.foForgetCoeff = 0.0;
            PuschParams.foCompensationBuffer = 1.0;
        end

        if strcmp(SimCtrl.alg.LDPC_DMI_method,'per_UE') 
            PuschParams.ldpcEarlyTerminationPerUe = pdu.ldpcEarlyTerminationPerUe;
            PuschParams.ldpcMaxNumItrPerUe = pdu.ldpcMaxNumItrPerUe;
        else
            PuschParams.ldpcEarlyTerminationPerUe = 0;
            PuschParams.ldpcMaxNumItrPerUe = 10;
        end
        
        PuschParamsUeg{idxUeg}.PuschParams = [PuschParamsUeg{idxUeg}.PuschParams, PuschParams];
    end

    [pusch_payload_ueg, tbErrUeg, cbErrUeg, rxDataUeg, harqState, UciRmSeqLen, nPolUciSegs, polarBuffers] = detPusch_cuphy(Xtf, PuschParamsUeg{idxUeg}, puschTable, harqState, srsState, Chan_UL, interfChan_UL, Xtf_noNoise, carrier);
    
    if SimCtrl.alg.enableEarlyHarq
        [rxDataUeg] = detPusch_cuphy_early_harq_measurement(Xtf, PuschParamsUeg{idxUeg}, puschTable, harqState, srsState, Chan_UL, interfChan_UL, Xtf_noNoise, carrier, rxDataUeg);
    end

    for idxUeInUeg = 1:nUeInUeg
        idxPdu = UegList{idxUeg}.idxPdu(idxUeInUeg)+1;
        PuschParamsList{idxPdu} = PuschParamsUeg{idxUeg}.PuschParams(idxUeInUeg);
        pusch_payload_list{idxPdu} = pusch_payload_ueg{idxUeInUeg};
        rxDataList{idxPdu} = rxDataUeg{idxUeInUeg};
        cbErrList{idxPdu} = cbErrUeg{idxUeInUeg};
        tbErrList{idxPdu} = tbErrUeg{idxUeInUeg};
        UciRmSeqLenList{idxPdu} = UciRmSeqLen{idxUeInUeg};
    end
end

idxSlot = carrier.idxSlotInFrame;
if SimCtrl.usePuschRxForPdsch
    idxPdu = pdu.pdschPduIdx-1;
else
    idxPdu = pdu.puschPduIdx-1;
end

if SimCtrl.genTV.enable && SimCtrl.genTV.cuPHY && ...
        ismember(idxSlot, SimCtrl.genTV.slotIdx)

    if contains(SimCtrl.genTV.TVname, 'TV_cuphy_')
        TVname = SimCtrl.genTV.TVname;
    else
        TVname = [SimCtrl.genTV.TVname, '_PUSCH_gNB_CUPHY_s', num2str(idxSlot),...
        'p', num2str(idxPdu)];
    end
%     saveTV_pusch(SimCtrl.genTV.tvDirName, TVname, pusch_payload, PuschParams, Xtf);
    saveTV_pusch_cuphy(SimCtrl.genTV.tvDirName, TVname, UegList, pusch_payload_list, PuschParamsList, ...
        Xtf, carrier, rxDataList, cbErrList, tbErrList, puschTable, UciRmSeqLenList);

    if(SimCtrl.genTV.polDbg && (nPolUciSegs > 0))
        saveTV_polar_debug(nPolUciSegs, polarBuffers, TVname);
    end



    SimCtrl.genTV.idx = SimCtrl.genTV.idx + 1;
end

return


function [pusch_payload_ueg, tbErrUeg, cbErrUeg, rxDataUeg, harqState, UciRmSeqLen, nPolUciSegs, polarBuffers] = detPusch_cuphy(Xtf, PuschParamsUeg, puschTable, harqState, srsState, Chan_UL, interfChan_UL, Xtf_noNoise, carrier)

% frontEnd process
PuschParams = PuschParamsUeg.PuschParams(1);
nUe = length(PuschParamsUeg.PuschParams);


% polar buffers
nPolUciSegs  = 0;
polarBuffers = cell(1000,1);


% load parameters
n_scid = PuschParams.n_scid;            % 0 or 1. User's dmrs scrambling id
nPrb = PuschParams.nPrb;               % number of prbs in allocation
startPrb = PuschParams.startPrb;       % starting prb of allocation
prgSize = PuschParams.prgSize;  
symIdx_data = PuschParams.symIdx_data; % indicies of data symbols. Dim: Nt_data x 1
slotNumber = PuschParams.slotNumber;
Nf = PuschParams.Nf;
Nt = PuschParams.Nt;
N_dmrs_id = PuschParams.N_dmrs_id;
symIdx_dmrs = PuschParams.symIdx_dmrs;  % Indicies of dmrs symbols (matlab 1 indexing). Dim: Nt_dmrs x 1
Nt_dmrs = PuschParams.Nt_dmrs;          % number of dmrs symbols
delta_f = PuschParams.delta_f;
maxLength = PuschParams.maxLength;
AdditionalPosition = PuschParams.AdditionalPosition;
numDmrsCdmGrpsNoData = PuschParams.numDmrsCdmGrpsNoData;
enableTfPrcd = PuschParams.enableTfPrcd;
puschIdentity = PuschParams.puschIdentity;
N_slot_frame = PuschParams.N_slot_frame;
N_symb_slot = PuschParams.N_symb_slot;
groupOrSequenceHopping = PuschParams.groupOrSequenceHopping;

enableCfoEstimation = PuschParams.enableCfoEstimation;
enableCfoCorrection = PuschParams.enableCfoCorrection;
enableWeightedAverageCfo = PuschParams.enableWeightedAverageCfo;
enableToEstimation = PuschParams.enableToEstimation;
enableToCorrection = PuschParams.enableToCorrection;
enableIrc = PuschParams.enableIrc;
Rww_regularizer_val = PuschParams.Rww_regularizer_val;
enable_nCov_shrinkage = PuschParams.enable_nCov_shrinkage;
nCov_shrinkage_method = PuschParams.nCov_shrinkage_method;
enable_use_genie_nCov = PuschParams.enable_use_genie_nCov; 
genie_nCov_method = PuschParams.genie_nCov_method; % 'genie_interfNoise_based', 'genie_interfChanResponse_based'
enableNoiseEstForZf = PuschParams.enableNoiseEstForZf;
enable_sphere_decoder = PuschParams.enable_sphere_decoder;
sphere_decoder = PuschParams.sphere_decoder;
TdiMode = PuschParams.TdiMode;
DTXthreshold = PuschParams.DTXthreshold;

nl = 0;
portIdx = [];
vec_scid = [];
layer2Ue = [];

foForgetCoeff = zeros(1, nUe);
foCompensationBuffer = zeros(1, nUe);

for idxUe = 1:nUe
    PuschParams = PuschParamsUeg.PuschParams(idxUe);
    vec_scid = [vec_scid, PuschParams.n_scid*ones(1,PuschParams.nl)];
    portIdx = [portIdx, PuschParams.portIdx];
    layer2Ue(nl+1 : nl+PuschParams.nl) = idxUe-1;
    nl = nl + PuschParams.nl;
    foForgetCoeff(idxUe) = PuschParams.foForgetCoeff;
    foCompensationBuffer(idxUe) = PuschParams.foCompensationBuffer;
end

cfo_est = zeros(max(AdditionalPosition,1), nl);
to_est = zeros(1, nl);
cfo_est_Hz = zeros(max(AdditionalPosition,1), nl);
to_est_microsec = zeros(1, nl);

% 64TR
global SimCtrl;
if SimCtrl.enable_static_dynamic_beamforming
    if PuschParams.digBFInterfaces == 0
        Xtf = Xtf(:,:,1:nl);
    else
        Xtf = Xtf(:,:,1:PuschParams.digBFInterfaces);
    end
end

currentSnr = SimCtrl.alg.fakeSNRdBForZf;
N0_ref = 10^(-currentSnr/10); % current noise variance (linear)

% set Nt_dmrs and dmrsIdx
if maxLength == 1
    Nt_dmrs = 1;
    for posDmrs = 1:AdditionalPosition+1
        dmrsIdx{posDmrs} = symIdx_dmrs(posDmrs);
    end
else
    Nt_dmrs = 2;
    dmrsIdx{1} = symIdx_dmrs(1:2);
    if AdditionalPosition > 0
        dmrsIdx{2} = symIdx_dmrs(3:4);
    end
end

% normalize Xtf
if SimCtrl.enable_normalize_Xtf_perUeg   
    idx_RE_this_Ueg = (startPrb-1)*12+1:(startPrb+nPrb-1)*12;
    Xtf_this_Ueg = Xtf(idx_RE_this_Ueg,:,:);
    scale_val_this_Ueg = util_get_normalization_scale(Xtf_this_Ueg);
    Xtf(idx_RE_this_Ueg,:,:) = Xtf(idx_RE_this_Ueg,:,:)*scale_val_this_Ueg;
end

% step 1: generate DMRS signal
lowPaprGroupNumber = zeros(length(symIdx_dmrs), 1);
lowPaprSequenceNumber = zeros(length(symIdx_dmrs), 1);
if enableTfPrcd == 1
    [r_dmrs, lowPaprGroupNumber, lowPaprSequenceNumber] = gen_dmrs_dftsofdm(Nf, startPrb, nPrb, N_slot_frame, N_symb_slot, symIdx_dmrs, ...
    slotNumber, puschIdentity, groupOrSequenceHopping);
else
    r_dmrs = gen_dmrs_sym(slotNumber, Nf, Nt, N_dmrs_id);
end

if SimCtrl.genTV.add_LS_chEst
    H_LS_est = [];
    for posDmrs = 1:AdditionalPosition+1
        [H_LS_est{posDmrs}] = apply_ChEst_LS_main...
            (Xtf,puschTable,slotNumber,Nf,Nt,N_dmrs_id,...
            nl, portIdx, vec_scid, dmrsIdx{posDmrs}, Nt_dmrs, nPrb, startPrb, ...
            delta_f,maxLength, numDmrsCdmGrpsNoData, enableTfPrcd,...
            N_slot_frame, N_symb_slot, puschIdentity,...
            groupOrSequenceHopping, r_dmrs);
    end
    H_LS_est_save =  [];
    for posDmrs = 1:AdditionalPosition+1
        H_LS_est_save = cat(4, H_LS_est_save, H_LS_est{posDmrs}(:,:,:)); % dim: [nRxAnt, nTxLayers, Nsc, NdmrsSym]
    end
    H_LS_est_save = permute(H_LS_est_save, [3, 2, 1, 4])/sqrt(2);
end

% estimate channels on all DMRS symbols
if ((mod(nPrb,4) == 0) && (nPrb >= 8)) % use 8 PRB DMRS to estimate 4 PRBs 
    if SimCtrl.alg.ChEst_alg_selector==1 % 'MMSE_w_delay_and_spread_est'
        if SimCtrl.alg.ChEst_MMSE_w_delay_and_spread_est_opt == 1
            error('Undefined! Only defined for nPrbs%0~=0!')
        elseif SimCtrl.alg.ChEst_MMSE_w_delay_and_spread_est_opt == 2
            [H_est,delay_mean_microsec,delay_spread_microsec, dbg_chest] = pusch_ChEst_LS_delayEst_MMSE(Xtf,puschTable,slotNumber,Nf,Nt,N_dmrs_id,...
                        nl, portIdx, vec_scid, dmrsIdx, Nt_dmrs, nPrb, startPrb, ...
                        delta_f,maxLength, numDmrsCdmGrpsNoData, enableTfPrcd,...
                        N_slot_frame, N_symb_slot, puschIdentity,...
                        groupOrSequenceHopping, AdditionalPosition, carrier, prgSize, SimCtrl.bfw.enable_prg_chest, r_dmrs);
        else
            error('Unknown option!')
        end    
    elseif SimCtrl.alg.ChEst_alg_selector == 0   % 'MMSE'
        for posDmrs = 1:AdditionalPosition+1
            H_est{posDmrs} = chan_estimation(Xtf,puschTable,slotNumber,Nf,Nt,N_dmrs_id,...
                nl, portIdx, vec_scid, dmrsIdx{posDmrs}, Nt_dmrs, nPrb, startPrb,...
                delta_f, numDmrsCdmGrpsNoData, enableTfPrcd,...
                N_slot_frame, N_symb_slot, puschIdentity,...
                groupOrSequenceHopping);
        end
    elseif SimCtrl.alg.ChEst_alg_selector == 2   % 'RKHS'
        for posDmrs = 1:AdditionalPosition+1
            H_est{posDmrs} = pusch_rkhs_chEst(Xtf,puschTable,slotNumber,Nf,Nt,N_dmrs_id,...
                nl, portIdx, vec_scid, dmrsIdx{posDmrs}, Nt_dmrs, nPrb, startPrb, ...
                delta_f,maxLength, numDmrsCdmGrpsNoData, enableTfPrcd,...
                 N_slot_frame, N_symb_slot, puschIdentity,...
                 groupOrSequenceHopping,PuschParamsUeg,nUe);
        end

    end
else % use 4 PRB DMRS to estimate 2 PRBs
    if SimCtrl.alg.ChEst_alg_selector == 1 % 'MMSE_w_delay_and_spread_est'
        if SimCtrl.alg.ChEst_MMSE_w_delay_and_spread_est_opt == 1
            error('This option was removed!')
        elseif SimCtrl.alg.ChEst_MMSE_w_delay_and_spread_est_opt == 2
            [H_est,delay_mean_microsec,delay_spread_microsec, dbg_chest] = pusch_ChEst_LS_delayEst_MMSE(Xtf,puschTable,slotNumber,Nf,Nt,N_dmrs_id,...
                        nl, portIdx, vec_scid, dmrsIdx, Nt_dmrs, nPrb, startPrb, ...
                        delta_f,maxLength, numDmrsCdmGrpsNoData, enableTfPrcd,...
                        N_slot_frame, N_symb_slot, puschIdentity,...
                        groupOrSequenceHopping, AdditionalPosition, carrier, prgSize, SimCtrl.bfw.enable_prg_chest, r_dmrs);
        end
    elseif SimCtrl.alg.ChEst_alg_selector == 0 % 'MMSE'
        for posDmrs = 1:AdditionalPosition+1
            H_est{posDmrs} = apply_ChEst_main(Xtf,puschTable,slotNumber,Nf,Nt,N_dmrs_id,...
                nl, portIdx, vec_scid, dmrsIdx{posDmrs}, Nt_dmrs, nPrb, startPrb, ...
                delta_f,maxLength, numDmrsCdmGrpsNoData, enableTfPrcd,...
                 N_slot_frame, N_symb_slot, puschIdentity,...
                 groupOrSequenceHopping);
        end
    elseif SimCtrl.alg.ChEst_alg_selector == 2   % 'RKHS'
        for posDmrs = 1:AdditionalPosition+1
            H_est{posDmrs} = pusch_rkhs_chEst(Xtf,puschTable,slotNumber,Nf,Nt,N_dmrs_id,...
                nl, portIdx, vec_scid, dmrsIdx{posDmrs}, Nt_dmrs, nPrb, startPrb, ...
                delta_f,maxLength, numDmrsCdmGrpsNoData, enableTfPrcd,...
                 N_slot_frame, N_symb_slot, puschIdentity,...
                 groupOrSequenceHopping,PuschParamsUeg,nUe);
        end
    else
        error('Unknow ChEst alg!')
    end
end

scIdxs = [((startPrb-1)*12 : ((startPrb-1)+nPrb)*12 - 1) + 1];
% Save channel estimates before being corrected by CFO
H_est_save =  [];
for posDmrs = 1:AdditionalPosition+1
    H_est_save = cat(4, H_est_save, H_est{posDmrs}(:,:,scIdxs)); % dim: [nRxAnt, nTxLayers, Nsc, NdmrsSym]
end
if SimCtrl.genTV.enable_logging_dbg_pusch_chest
    H_LS_est_save =  [];
    for posDmrs = 1:AdditionalPosition+1
        H_LS_est_save = cat(4, H_LS_est_save, dbg_chest.H_LS_est{posDmrs}(scIdxs(1:2:end),:,:)); % dim: [nRxAnt, nTxLayers, Nsc, NdmrsSym]
    end
end

if SimCtrl.enable_get_genie_channel_matrix
    if isfield(Chan_UL{1}, 'chanMatrix_FD_oneSlot') && ~isempty(Chan_UL{1}.chanMatrix_FD_oneSlot) %contains(Chan_UL{1}.type,'TDL') || contains(Chan_UL{1}.type,'CDL') || contains(Chan_UL{1}.type,'AWGN') || contains(Chan_UL{1}.type,'P2P')
        H_genie_save = [];        
        for idxUe = 1:nUe
            numLayersThisUE = length(find(layer2Ue == (idxUe-1)));
            tmp_H_genie_save = Chan_UL{idxUe}.chanMatrix_FD_oneSlot(:,:,:,1:numLayersThisUE); % chanMatrix_FD_oneSlot dim: [Nsc, 14, Nrx, Ntx]
            if SimCtrl.normalize_pusch_tx_power_over_layers
                tmp_H_genie_save  = tmp_H_genie_save/sqrt(numLayersThisUE);
            end
            H_genie_save = cat(4, H_genie_save, tmp_H_genie_save);
        end 
        H_genie_save_all_sym = permute(H_genie_save, [3, 4, 1, 2]); % dim: [nRxAnt, nTxLayers, Nsc, 14]
        H_genie_save = H_genie_save_all_sym(:,:,scIdxs,cell2mat(dmrsIdx)); % dim: [nRxAnt, nTxLayers, Nsc, NdmrsSym]
%         figure; plot(real(squeeze(H_est_save(1,1,:,1)))); hold on; plot(real(squeeze(H_genie_save(1,1,:,1))));legend({'H_est','H_genie'})
        H_genie_all_sym = cell(14,1);
        for idxSym = 1:14            
            H_genie_all_sym(idxSym) = {H_genie_save_all_sym(:,:,:,idxSym)};%dim: [nRxAnt, nTxLayers, Nsc, 1]
        end
    else
        error('Genie channel matrix is not available!')
    end    
    chest_error = H_est_save - H_genie_save;
    NMSE_ChestError = sqrt(mean(abs(chest_error(:)).^2)) / sqrt(mean(abs(H_genie_save(:)).^2));
    NMSE_ChestError_dB = pow2db(NMSE_ChestError); % get the normalized MSEl

    % use genie channel for the following MIMO equalization
    if SimCtrl.alg.enable_use_genie_channel_for_equalizer
        H_est_save = H_genie_save;
        for posDmrs = 1:AdditionalPosition+1
            H_est{posDmrs}(:,:,scIdxs) = H_genie_save(:,:,:,posDmrs);
        end
    end
end

%%% Noise covariance estimation, RSSI estimation and SINR calculation

XtfDmrs    = embed_dmrs_UL(zeros(size(Xtf)), r_dmrs,  nl, portIdx, n_scid, symIdx_dmrs, ...
    PuschParams.Nt_dmrs, PuschParams.energy, nPrb, startPrb, puschTable, enableTfPrcd,...
    0, [], []);

%XtfDmrs    = embed_dmrs_UL(Xtf, r_dmrs,  nl, portIdx, n_scid, symIdx_dmrs, ...
%    Nt_dmrs, energy, nPrb, startPrb, puschTable, enablePrcdBf, PM_W);

%XtfDmrs    = embed_dmrs_UL(zeros(size(Xtf)), r_dmrs,  nl, portIdx, n_scid, symIdx_dmrs, ...
%    maxLength*Nt_dmrs, PuschParams.energy, nPrb, startPrb, puschTable, 0, []);

% step 2: calculate r_tilde for every DMRS RE and antenna by subtracting desired signal from received
% signal
nAnt = size(Xtf,3);
numDmrsSym = length(symIdx_dmrs);
r_tilde = zeros(nPrb*12,numDmrsSym,nAnt);
if SimCtrl.normalize_pusch_tx_power_over_layers % note that we may normalize DMRS REs w.r.t. nLayers in embed_dmrs_UL(), thus we need to denormalize it below.
    denormalization_factor = sqrt(nl);
else
    denormalization_factor = 1;
end
for iant = 1:nAnt
    r_tilde(:,:,iant) = Xtf((startPrb-1)*12+1:(startPrb+nPrb-1)*12,symIdx_dmrs,iant);
    for il=1:nl
        for idmrs=1:numDmrsSym
            r_tilde(:,idmrs,iant) = r_tilde(:,idmrs,iant) - denormalization_factor*squeeze(XtfDmrs((startPrb-1)*12+1:(startPrb+nPrb-1)*12,symIdx_dmrs(idmrs),il)) .* squeeze(H_est_save(iant,il,:,ceil(idmrs*(1+AdditionalPosition)/numDmrsSym)));
        end
    end
end

% step 3: calculate covariance matrix
nCov = zeros(nAnt,nAnt,nPrb,AdditionalPosition+1);
tmp_noiseVar = zeros(nPrb, AdditionalPosition+1);
shrinkage_params = struct();
num_samples_nCov = zeros(nPrb, AdditionalPosition+1);
for ii=1:nPrb
    tmp_nCov = zeros(nAnt,nAnt,AdditionalPosition+1) + 1e-10*repmat(eye(nAnt),[1,1,AdditionalPosition+1]);    
    for jj=1:1:12
        isDataRe = (numDmrsCdmGrpsNoData == 1 && mod(jj, 2) == 0);
        if ~isDataRe
            idx_dmrs_in_slot = 1;
            for posDmrs = 1:AdditionalPosition+1
                for kk=1:length(dmrsIdx{posDmrs})
                    tmp_nCov(:,:,posDmrs) = tmp_nCov(:,:,posDmrs) + squeeze(r_tilde((ii-1)*12+jj,idx_dmrs_in_slot,:)) * conj(transpose(squeeze(r_tilde((ii-1)*12+jj,idx_dmrs_in_slot,:))));
                    num_samples_nCov(ii,posDmrs) = num_samples_nCov(ii,posDmrs) + 1;
                    idx_dmrs_in_slot = idx_dmrs_in_slot + 1;
                end
            end
        end
    end
    for posDmrs = 1:AdditionalPosition+1
        if numDmrsCdmGrpsNoData == 1
            nCov(:,:,ii,posDmrs) = tmp_nCov(:,:,posDmrs)/(6*length(dmrsIdx{posDmrs})) + Rww_regularizer_val*eye(nAnt,nAnt);
        else
            nCov(:,:,ii,posDmrs) = tmp_nCov(:,:,posDmrs)/(12*length(dmrsIdx{posDmrs})) + Rww_regularizer_val*eye(nAnt,nAnt);
        end
    end
    for posDmrs = 1:AdditionalPosition+1
        tmp_noiseVar(ii,posDmrs) = tmp_noiseVar(ii,posDmrs) + sum(abs(diag(nCov(:,:,ii,posDmrs))))/nAnt;
    end 
end

%     % nCov shrinkage
%     if enable_nCov_shrinkage % refer to the paper "Shrinkage Algorithms for MMSE Covariance Estimation"
%         if SimCtrl.subslot_proc_option == 0 % full-slot processing
%             Rtmp = mean(nCov(:,:,ii,:), 4);
%             T = sum(num_samples_nCov);
%             Rtmp_shrinked = nCov_shrinkage(Rtmp, T, nCov_shrinkage_method);
%             for posDmrs = 1:AdditionalPosition+1
%                 nCov(:,:,ii,posDmrs) = Rtmp_shrinked;
%             end
%         elseif  ismember(SimCtrl.subslot_proc_option, [1,2,3]) % sub-slot processing. Need to shrink nCov for each DMRS position separately
%             for posDmrs = 1:AdditionalPosition+1
%                 Rtmp = nCov(:,:,ii,posDmrs);
%                 T = num_samples_nCov(posDmrs);
%                 nCov(:,:,ii,posDmrs) = nCov_shrinkage(Rtmp, T, nCov_shrinkage_method);
%             end
%         elseif SimCtrl.subslot_proc_option == 4
%             % do nothing here. The shrinkage will be done inside derive_equalizer_mmse_irc()
%             shrinkage_params.num_samples_nCov = num_samples_nCov;
%             shrinkage_params.nCov_shrinkage_method = nCov_shrinkage_method;
%         end
%     else
%        if SimCtrl.subslot_proc_option == 0 % full-slot processing
%            Rtmp = mean(nCov(:,:,ii,:), 4);
%            for posDmrs = 1:AdditionalPosition+1
%                nCov(:,:,ii,posDmrs) = Rtmp;
%            end
%        else
%            % do nothing           
%        end
%     end
%   end

%for ii = 1:273;a(ii)=mean(diag(real(nCov(:,:,ii,1))));end;figure;plot(diff(a));
% averaging nCov with neighboring PRBs to improve nCov est reliability. Useful when sub-slot proc enabled
if SimCtrl.alg.enable_avg_nCov_prbs_fd
    win_size_avg_nCov = SimCtrl.alg.win_size_avg_nCov_prbs_fd;
    one_side_win_size_avg_nCov = (win_size_avg_nCov-1)/2;   
    tmp_noiseVar_dB = pow2db(tmp_noiseVar);
    if SimCtrl.subslot_proc_option == 0
        max_posDmrs_allow_avg_nCov = AdditionalPosition+1;
    else
        max_posDmrs_allow_avg_nCov = AdditionalPosition; % for subslot processing, let's do freq nCov averaging on DMRS symbols except the last one
    end
    tmp2_nCov = nCov;
    for ii=1:nPrb
        for posDmrs = 1:max_posDmrs_allow_avg_nCov 
            tmp_ii_start = max(1, ii-one_side_win_size_avg_nCov);
            tmp_ii_end = min(nPrb, ii+one_side_win_size_avg_nCov);
            ii_start = min( max(tmp_ii_end-win_size_avg_nCov+1,1), tmp_ii_start);
            ii_end = max( min(tmp_ii_start+win_size_avg_nCov-1,nPrb), tmp_ii_end);
            noiseVar_dB_in_window = tmp_noiseVar_dB(ii_start:ii_end, posDmrs);
            noiseVar_dB_ii = tmp_noiseVar_dB(ii, posDmrs);
            idx_le_thres = (abs(noiseVar_dB_in_window-noiseVar_dB_ii)<=SimCtrl.alg.threshold_avg_nCov_prbs_fd_dB);
            nCov_in_window = tmp2_nCov(:,:,ii_start:ii_end,posDmrs);
            nCov(:,:,ii,posDmrs) = mean(nCov_in_window(:,:,idx_le_thres),3);
            num_samples_nCov(ii,posDmrs) = sum(idx_le_thres)*num_samples_nCov(ii,posDmrs);
%                 if ii<2
%                     nCov(:,:,1,posDmrs) = (tmp2_nCov(:,:,1,posDmrs)+tmp2_nCov(:,:,2,posDmrs))/2;
%                 elseif (ii>=2) && (ii<=nPrb-1)
%                     offset_idx_prb = [-1,1];
%                     noiseVar_offset_prb_dB = pow2db(noiseVar(ii+offset_idx_prb,posDmrs));
%                     [~,idx_min] = min(abs(noiseVar_offset_prb_dB-pow2db(noiseVar(ii,posDmrs))));
%                     offset_idx_min = offset_idx_prb(idx_min);
%                     nCov(:,:,ii,posDmrs) = (tmp2_nCov(:,:,ii,posDmrs)+tmp2_nCov(:,:,ii+offset_idx_min,posDmrs))/2;
%                 else
%                     nCov(:,:,nPrb,posDmrs) = (tmp2_nCov(:,:,nPrb-1,posDmrs)+tmp2_nCov(:,:,nPrb,posDmrs))/2;
%                 end
        end
    end
end

% nCov shrinkage
for ii=1:nPrb
    if enable_nCov_shrinkage % refer to the paper "Shrinkage Algorithms for MMSE Covariance Estimation"
        if SimCtrl.subslot_proc_option == 0 % full-slot processing
            Rtmp = mean(nCov(:,:,ii,:), 4);
            T = sum(num_samples_nCov(ii,:)); % for full-slot processing, nCov averaging happens over DMRS symbols as well
            Rtmp_shrinked = nCov_shrinkage(Rtmp, T, nCov_shrinkage_method);
            for posDmrs = 1:AdditionalPosition+1
                nCov(:,:,ii,posDmrs) = Rtmp_shrinked;
            end
        elseif  ismember(SimCtrl.subslot_proc_option, [1,2,3]) % sub-slot processing. Need to shrink nCov for each DMRS position separately
            for posDmrs = 1:AdditionalPosition+1
                Rtmp = nCov(:,:,ii,posDmrs);
                T = num_samples_nCov(ii,posDmrs);
                nCov(:,:,ii,posDmrs) = nCov_shrinkage(Rtmp, T, nCov_shrinkage_method);
            end
        end
    else
       if SimCtrl.subslot_proc_option == 0 % full-slot processing
           Rtmp = mean(nCov(:,:,ii,:), 4);
           for posDmrs = 1:AdditionalPosition+1
               nCov(:,:,ii,posDmrs) = Rtmp;
           end
       else
           % do nothing           
       end
    end
end

% get Measurement: noiseVar
noiseVar_fullSlotAvg = repmat(mean(tmp_noiseVar(:)),AdditionalPosition+1,1); % dim: (AdditionalPosition+1)x1. let's keep the same dim to the case in subslot proc

noiseVar_subSlot     = zeros(AdditionalPosition+1,1);
tmp2_noiseVar = mean(tmp_noiseVar,1); % avg across PRBs
for idx_dmrsAddPos = 1:AdditionalPosition+1
    noiseVar_subSlot(idx_dmrsAddPos,1) = mean(tmp2_noiseVar(1,1:idx_dmrsAddPos));
end

if SimCtrl.subslot_proc_option == 0 
    noiseVar = noiseVar_fullSlotAvg;
else 
    noiseVar = noiseVar_subSlot;
end

if enableNoiseEstForZf 
   N0_ref = noiseVar; % N0_ref dim: 1x(AdditionalPosition+1)
end
noiseVardB = 10*log10(mean(noiseVar,2)) + 0.5;                                          % remove 0.5 dB bias due to DMRS filtering

% get Measurement: RSRP
rsrp = zeros(AdditionalPosition+1, nUe);
for idxUe = 1:nUe
    layerIdx = find(layer2Ue == (idxUe-1));
    tmp_rsrp = zeros(AdditionalPosition+1,1);
    for ii=1:nPrb*12
        for jj=1:(AdditionalPosition+1)
            for kk=1:nAnt
                for ll=1:length(layerIdx)
                    tmp_rsrp(jj) = tmp_rsrp(jj) + abs(H_est_save(kk,layerIdx(ll),ii,jj)).^2;
                end
            end
        end
    end
    tmp_rsrp = tmp_rsrp / (nPrb*12*nAnt);
    rsrp(:, idxUe) = tmp_rsrp;    
end
if SimCtrl.subslot_proc_option == 0 % full-slot processing
    rsrp = repmat(mean(rsrp, 1), AdditionalPosition+1, 1);
else % sub-slot processing
    tmp_rsrp = rsrp;
    for idx_dmrsAddPos = 1:AdditionalPosition+1
        rsrp(idx_dmrsAddPos, :) = mean(tmp_rsrp(1:idx_dmrsAddPos, :), 1);
    end
end
rsrpdB = 10*log10(rsrp);

% get Measurement: SINR 
sinrdB = rsrpdB - repmat(noiseVardB, 1, nUe);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Get some Genie info %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% verify genie channel matrix
% Y_dmrs_genie = squeeze(XtfDmrs(:,3,1)).*squeeze(Chan_UL{idxUe}.chanMatrix_FD_oneSlot(:,3,1,1));
% Y_dmrs_genie = squeeze(XtfDmrs(:,3,1)).*squeeze(Chan_UL{idxUe}.chanMatrix_FD_oneSlot(:,3,1,1)) + squeeze(XtfDmrs(:,3,2)).*squeeze(Chan_UL{idxUe}.chanMatrix_FD_oneSlot(:,3,1,2));
% figure; plot(abs(Y_dmrs_genie).^2); hold on; plot(abs(squeeze(Xtf_noNoise(:,3,1)).').^2)
% figure; plot(real(Y_dmrs_genie)); hold on; plot(real(squeeze(Xtf_noNoise(:,3,1)).'))
% figure; plot(abs(squeeze(Chan_UL{idxUe}.chanMatrix_FD_oneSlot(:,3,1,1))).^2); hold on;plot(abs(squeeze(H_est_save(1,1,:,1))).^2); legend('Genie channel','Estimated channel')
% figure; plot(real(squeeze(Chan_UL{idxUe}.chanMatrix_FD_oneSlot(:,3,1,1)))); hold on;plot(real(squeeze(H_est_save(1,1,:,1)))); legend('Genie channel','Estimated channel')
% mse = mean(abs(squeeze(Chan_UL{idxUe}.chanMatrix_FD_oneSlot(:,3,1,1)) - squeeze(H_est_save(1,1,:,1))).^2)
% nmse = pow2db( mse / mean(abs(squeeze(Chan_UL{idxUe}.chanMatrix_FD_oneSlot(:,3,1,1)) )).^2 )
if SimCtrl.alg.enable_get_genie_meas
    % get genie RSRP
    idxPort = 0; 
    for idxUe = 1:nUe
        numLayersThisUE = length(find(layer2Ue == (idxUe-1)));
        switch Chan_UL{1}.type
            case {'AWGN', 'P2P'}
                if SimCtrl.enableUlRxBf
                    startAnt = sum(nAnt_ueg(1:idxUe-1));
                    W_ue{idxUe} = W_ueg(startAnt+1:startAnt+nPort_ueg(idxUe), :).';
                    tmp_genie_chanMatrix_FD = Chan_UL{idxUe}.chanMatrix * W_ue{idxUe}; % TBD: apply to RBs in Ueg only, not full slot
                    genie_chanMatrix_FD{idxUe} = repmat(tmp_genie_chanMatrix_FD, [1,1,PuschParams.nPrb*12,PuschParams.N_symb_slot]);
                    idxPort = idxPort + nPort_ueg(idxUe);
                else
                    genie_chanMatrix_FD{idxUe} = repmat(Chan_UL{idxUe}.chanMatrix, [1,1,PuschParams.nPrb*12,PuschParams.N_symb_slot]);
                end
                genie_chanMatrix_FD{idxUe} = permute(genie_chanMatrix_FD{idxUe}, [3,4,2,1]); % reshape channel matrix into dim [num allocated REs, num of sym, num Rx ant., num Tx ant]
                genie_chanMatrix_FD{idxUe} = genie_chanMatrix_FD{idxUe}(:,:,:,1:numLayersThisUE); % truncate channel matrix into dim [num allocated REs, num of sym, num Rx ant., num layers]
            case {'TDLA30-5-Low', 'TDLA30-10-Low', 'TDLB100-400-Low', 'TDLC300-100-Low','TDL', 'CDL', 'UMi', 'UMa', 'RMa'}
                if SimCtrl.enableUlRxBf
                    error('Undefined case!')
                end
                idx_RE_allocated = scIdxs;%(PuschParams.startPrb-1)*12+1:(PuschParams.startPrb-1)*12+PuschParams.nPrb*12;
                if strcmp(Chan_UL{1}.model_source, 'MATLAB5Gtoolbox') || strcmp(Chan_UL{1}.model_source, 'sionna')
                    if PuschParams.prcdBf
                        error('Undefined case: we need to get precoding matrix from PDU and then enable this case')
%                         genie_chanMatrix_FD{idxUe} = Chan_UL{idxUe}.chanMatrix_FD_oneSlot(idx_RE_allocated,:,:,:)*PuschParams.PM_W;
                    else                        
                        genie_chanMatrix_FD{idxUe} = Chan_UL{idxUe}.chanMatrix_FD_oneSlot(idx_RE_allocated,:,:,1:numLayersThisUE);
                    end
                else
                    error("Undefined FD channel matrix! Please set Chan.model_source to 'MATLAB5Gtoolbox' or 'sionna'!")
                    
                end
        end
    end
    for idxUe = 1:nUe
        tmp_rsrp_genie = mean(abs(genie_chanMatrix_FD{idxUe}(:,symIdx_dmrs,:,:)).^2,[1,2,3]); % average over REs, symbols and gNB Rx Ant
        genie_rsrp_dB(idxUe) = 10*log10(sum(tmp_rsrp_genie(:))); % sum over Tx Ant.
    end
    
    % get genie nCov and noise var
    if strcmp(genie_nCov_method, 'genie_interfNoise_based')
        tmp_genie_noise = Xtf - Xtf_noNoise;
%         tmp_genie_noise = Xtf;
%         for idxUe = 1:nUe
%             layer_indices_this_ue = find(layer2Ue == (idxUe-1));
%             num_layers_this_ue = length(layer_indices_this_ue);
%             for idxLayerThisUE = 1:num_layers_this_ue
%                 layer_idx = layer_indices_this_ue(idxLayerThisUE);
%                 tmp_genie_noise = tmp_genie_noise - repmat(XtfDmrs(:,:,layer_idx),[1,1,nAnt]).*genie_chanMatrix_FD{idxUe};
%             end
%         end
%         figure; plot(real(genie_chanMatrix_FD{1}(:,3,1)));hold on; plot(real(squeeze(H_est_save(1,1,:,1))))
%         figure; plot(abs(genie_chanMatrix_FD{1}(:,3,1)));hold on; plot(abs(squeeze(H_est_save(1,1,:,1))))
        genie_noise = tmp_genie_noise(:,symIdx_dmrs,:);
        for ii=1:nPrb
            tmp_nCov = zeros(nAnt,nAnt) + 1e-10*eye(nAnt);
            for jj=1:1:12
                isDataRe = (numDmrsCdmGrpsNoData == 1 && mod(jj, 2) == 0);
                if ~isDataRe
                    for kk=1:numDmrsSym
                        tmp_nCov = tmp_nCov + squeeze(genie_noise((ii-1)*12+jj,kk,:)) * conj(transpose(squeeze(genie_noise((ii-1)*12+jj,kk,:))));
                    end
                end
            end
            if numDmrsCdmGrpsNoData == 1
                genie_nCov(:,:,ii) = tmp_nCov/(6*numDmrsSym);
            else
                genie_nCov(:,:,ii) = tmp_nCov/(12*numDmrsSym); % Noise covariance matrix (per PRB), figure;plot(squeeze(abs(nCov(1,2,:))));hold on;plot(squeeze(abs(nCov_genie(1,2,:))))
            end
        end
    elseif strcmp(genie_nCov_method, 'genie_interfChanResponse_based')
        genie_nCov = zeros(nAnt,nAnt,nPrb);
        for idxInterfUe = 1:SimCtrl.N_Interfering_UE_UL
            genie_InterfChanMatrix_FD = 10^(-interfChan_UL{idxInterfUe}.SIR/20)*interfChan_UL{idxInterfUe}.interfChanMatrix_FD_oneSlot(idx_RE_allocated,:,:,:);  
            for ii = 1:nPrb
                for jj=1:1:12
                    for kk=1:14
                        tmp_h = squeeze(genie_InterfChanMatrix_FD((ii-1)*12+jj,kk,:,:));
                        genie_nCov(:,:,ii) = genie_nCov(:,:,ii) + tmp_h*tmp_h';
                    end
                end
            end
        end
        genie_nCov = genie_nCov/12/14 + 10^(-Chan_UL{1}.SNR/10)*eye(nAnt,nAnt);
    else
        error('Undefined nCov method!')
    end
    genie_noiseVar_per_Rx = real(diag(mean(genie_nCov,3))); 
    genie_noiseVar = mean(genie_noiseVar_per_Rx);
    genie_noiseVar_dB = 10*log10(genie_noiseVar);

    % get genie pre-SINR
    genie_pre_sinr_dB = genie_rsrp_dB - genie_noiseVar_dB;

    % get MMSE-IRC post-SINR
    genie_nCov_per_re1  = repmat(genie_nCov,[1,1,1,carrier.N_sc_RB]);
    genie_nCov_per_re2 = permute(genie_nCov_per_re1,[4,3,1,2]);
    genie_nCov_per_re3 = reshape(genie_nCov_per_re2,[nPrb*carrier.N_sc_RB,nAnt,nAnt]);
    genie_nCov_per_re  = permute(genie_nCov_per_re3,[2,3,1]); % dim: [nAnt, nAnt, N_sc]   
    genie_chanMatrix_concat = [];
    for idxUe = 1:nUe
        genie_chanMatrix_concat = cat(4, genie_chanMatrix_concat,genie_chanMatrix_FD{idxUe}(:,symIdx_dmrs,:,:));
    end
    nLayers_allUEs = length(layer2Ue);
    num_REs = size(genie_chanMatrix_concat,1);
    post_noiseVar_allLayers = zeros(nLayers_allUEs,1);
    post_noiseVar_allLayers_all_REs_allDMRS = zeros(nLayers_allUEs,num_REs,AdditionalPosition+1);
    for posDmrs = 1:AdditionalPosition+1       
        genie_chanMatrix_thisDMRS_allUEs = genie_chanMatrix_concat(:,posDmrs,:,:);        
        for idxRE = 1:num_REs
            genie_Hmat = reshape(genie_chanMatrix_concat(idxRE,posDmrs,:,:), [nAnt,nLayers_allUEs]);
            genie_nCov_this_RE = genie_nCov_per_re(:,:,idxRE);
            tmp_this_RE_post_noiseVar = real(diag(inv(genie_Hmat'*inv(genie_nCov_this_RE)*genie_Hmat + eye(nLayers_allUEs)))); 
            post_noiseVar_allLayers_all_REs_allDMRS(:,idxRE,posDmrs) = tmp_this_RE_post_noiseVar;
            post_noiseVar_allLayers = post_noiseVar_allLayers + tmp_this_RE_post_noiseVar;
        end
    end
    post_noiseVar_allLayers = post_noiseVar_allLayers/num_REs/(AdditionalPosition+1);
    genie_post_noiseVar_dB = zeros(nUe,1);
    genie_post_sinr_dB = zeros(nUe,1);
    genie_CQI(idxUe) = zeros(nUe,1);
    for idxUe = 1:nUe
        layerIdx = find(layer2Ue == (idxUe-1));
        genie_post_noiseVar_dB(idxUe) = 10*log10(mean(post_noiseVar_allLayers(layerIdx)));
        genie_post_sinr_dB(idxUe) = 10*log10(mean(1./post_noiseVar_allLayers_all_REs_allDMRS(layerIdx,:,:)-1,[1,2,3]));
        genie_CQI(idxUe) = sinr2cqi_mapping(puschTable.sinr2cqi, genie_post_sinr_dB(idxUe));
    end

end


%fprintf('***  Mean noiseVar (dB) %3.2f  ***\n', noiseVardB);   
%for idxUe = 1:nUe
%    fprintf('***  UE %d RSRP (dB) %2.2f ***  SINR (dB) %2.2f\n', idxUe, rsrpdB(idxUe), sinrdB(idxUe));
%end


%%% end noise covariance estimation, RSSI estimation and SINR calculation
%%% =====================================================================



[nAnt, nLayer, nSc] = size(H_est{1});

if (nAnt > 8) && enableIrc
    error('MMSE_IRC is not supported with nAnt > 8 ...');
end

% Early HARQ chEst does not have timing/frequency offset correction
if SimCtrl.alg.enableEarlyHarq
    H_est_earlyHarq = H_est{1};
end

% Estimate CFO
if enableCfoEstimation && (AdditionalPosition > 0)
    accum = zeros(AdditionalPosition, nLayer);
    for posDmrs = 1:AdditionalPosition
        for ii=1:nAnt
            for jj=scIdxs
                for ll =1:nLayer
                    accum(posDmrs, ll) = accum(posDmrs, ll) + H_est{posDmrs+1}(ii,ll,jj)*conj(H_est{posDmrs}(ii,ll,jj));
                end
            end
        end
    end

    for posDmrs = 1:AdditionalPosition
        dist_dmrs = dmrsIdx{posDmrs+1}(1)-dmrsIdx{posDmrs}(1);
        cfo_est(posDmrs,:) = 1/2/pi*angle(accum(posDmrs,:))/dist_dmrs;
    end

    if SimCtrl.subslot_proc_option == 0 % full-slot processing
        cfo_est = mean(cfo_est, 1);
        cfo_est = repmat(cfo_est, AdditionalPosition,1);
    else % sub-slot processing
        if SimCtrl.alg.enable_instant_equ_coef_cfo_corr
            tmp_cfo_est = cfo_est;
            for ii_dmrs = 1:AdditionalPosition
                cfo_est(ii_dmrs,:) = mean(tmp_cfo_est(1:ii_dmrs,:),1);
            end
        else
            cfo_est = mean(cfo_est, 1);
            cfo_est = repmat(cfo_est, AdditionalPosition,1);
        end
    end
    cfo_est_Hz = cfo_est * (delta_f/15e3)/(71.35e-6);

    % average CFO over layers for each UE
    for idxUe = 1:nUe
        layerIdx = find(layer2Ue == (idxUe-1));
        cfo_est(:,layerIdx) = repmat(mean(cfo_est(:,layerIdx),2),1,length(layerIdx));
        cfo_est_Hz(:,layerIdx) = repmat(mean(cfo_est_Hz(:,layerIdx),2),1,length(layerIdx));
    end

    if enableWeightedAverageCfo == 1 % only valid for SimCtrl.subslot_proc_option == 0
        for idxUe = 1:nUe
            layerIdx = find(layer2Ue == (idxUe-1));
            foCompensationBuffer(idxUe) = foForgetCoeff(idxUe)*foCompensationBuffer(idxUe) + (1 - foForgetCoeff(idxUe)) * nPrb * 10^(sinrdB(idxUe)/10.0) * exp(1j*2*pi*cfo_est(1, layerIdx(1)));
            cfo_est(1:AdditionalPosition, layerIdx) = 1/2/pi*angle(foCompensationBuffer(idxUe));
        end
        cfo_est_Hz = cfo_est * (delta_f/15e3)/(71.35e-6);
    end
end

% Estimate timing offset
if enableToEstimation
    accum = zeros(AdditionalPosition+1, nLayer);
    for ii=1:nAnt
        for idxPrb=0:(nPrb-1)
            scIdxs1 = [(startPrb+idxPrb-1)*12+1 : (startPrb+idxPrb)*12-1];
            for ll = 1:nLayer
                for posDmrs = 1:AdditionalPosition+1
                    for jj = 1:11
                        accum(posDmrs, ll) = accum(posDmrs, ll) + H_est{posDmrs}(ii,ll,scIdxs1(jj)+1)*conj(H_est{posDmrs}(ii,ll,scIdxs1(jj)));
                    end
                end
            end
        end
    end
    to_est = -1/2/pi*angle(accum);
    if SimCtrl.subslot_proc_option == 0 % full-slot processing
        to_est = repmat(mean(to_est,1), AdditionalPosition+1, 1);
    else % sub-slot processing
        tmp_to_est = to_est;
        for ii_dmrs = 1:AdditionalPosition+1
            to_est(ii_dmrs,:) = mean(tmp_to_est(1:ii_dmrs,:), 1);
        end
    end
    to_est_microsec = to_est/delta_f*1e6;

    % average TO over layers for each UE
    for idxUe = 1:nUe
        layerIdx = find(layer2Ue == (idxUe-1));
        to_est(:,layerIdx) = repmat(mean(to_est(:,layerIdx),2),1,length(layerIdx));
        to_est_microsec(:,layerIdx) = repmat(mean(to_est_microsec(:,layerIdx),2),1,length(layerIdx));
    end
end

% correct channel estimate by CFO
if enableCfoCorrection && (AdditionalPosition > 0) && (~SimCtrl.alg.enable_use_genie_channel_for_equalizer) % genie channel est has no CFO applied thus disabling CFO correction
    for posDmrs = 2:AdditionalPosition+1
        for idxUe = 1:nUe
            layerIdx = find(layer2Ue == (idxUe-1));
            if SimCtrl.alg.enable_instant_equ_coef_cfo_corr
                ue_cfo = cfo_est(posDmrs-1,layerIdx);
            else
                ue_cfo = cfo_est(end,layerIdx);
            end
            H_est{posDmrs}(:, layerIdx, :) = H_est{posDmrs}(:, layerIdx, :).*exp(-1j*2*pi*ue_cfo*(dmrsIdx{posDmrs}(1)-dmrsIdx{1}(1)));
        end
    end
end

% correct channel estimate by TO
if enableToCorrection % Not implemented by cuPHY
    for posDmrs = 1:AdditionalPosition+1
        for idxUe = 1:nUe
            layerIdx = find(layer2Ue == (idxUe-1));
            ue_to = to_est(layerIdx);
            for ll = 1:length(layerIdx)
                for idxAnt = 1:nAnt
                    H_est{posDmrs}(idxAnt, layerIdx(ll), :) = H_est{posDmrs}(idxAnt, layerIdx(ll), :).*reshape(exp(-1j*2*pi*ue_to(ll)*[-nSc/2:-1, 1:nSc/2]), 1, 1, nSc);
                end
            end
        end
    end
end

% get CFO for each symbol and layer
cfo_est_symbols_layers = repmat(cfo_est(end,:),14,1);
if SimCtrl.alg.enable_instant_equ_coef_cfo_corr
    start_sym_idx = 1;
    for posDmrs = 2:AdditionalPosition+1
        end_dmrs_sym_idx = dmrsIdx{posDmrs}(end);
        num_sym = end_dmrs_sym_idx - start_sym_idx + 1;
        cfo_est_symbols_layers(start_sym_idx:end_dmrs_sym_idx,:) = repmat(cfo_est(posDmrs-1,:),num_sym,1);
        start_sym_idx = end_dmrs_sym_idx + 1;
    end
end

if enableIrc
    % The equalizer works on the entire BWP regardless of allocated PRB. For
    % this reason extend the noise-interference covariance for the entire
    % bandwidth
%     noiseIntfCov = repmat(nCov(:,:,1), [1, 1, Nf/12]);
%     noiseIntfCov = repmat(mean(nCov,3), [1, 1, Nf/12]);
    %noiseIntfCov = repmat(N0_ref*eye(nAnt,nAnt), [1, 1, Nf/12]);

%    
    noiseIntfCov = repmat(nCov(:,:,1,:), [1, 1, Nf/12,1]);
    prbIdxs = startPrb : (startPrb + nPrb - 1); 
    if enable_use_genie_nCov
        noiseIntfCov(:,:,prbIdxs,:) = repmat(genie_nCov,[1,1,1,AdditionalPosition+1]);
    else
        noiseIntfCov(:,:,prbIdxs,:) =  nCov;
    end

end
% tic
if enable_sphere_decoder % Sphere Decoder
    if TdiMode == 2
        % Apply time domain interpolation to channel estimate
        H_est = apply_interpolation(H_est, symIdx_data, symIdx_dmrs, numDmrsCdmGrpsNoData);
        % Apply whitening to H_est and Xtf
        [Nrx,~,Nrb,~] = size(noiseIntfCov);
        whitening_matrix_per_RB = zeros(size(noiseIntfCov(:,:,:,end)));
        for idx_RB = 1:Nrb
            whitening_matrix_per_RB(:,:,idx_RB) = sqrtm(inv(noiseIntfCov(:,:,idx_RB,end)));
        end
        whitening_matrix_per_RE = permute( reshape( permute( repmat(whitening_matrix_per_RB, 1, 1, 1, 12), [4,3,1,2] ), 12*Nrb,Nrx,Nrx), [2,3,1] );
        for idx_dataSym = 1:length(symIdx_data)
            tmp_H_est = pagemtimes(whitening_matrix_per_RE,H_est{idx_dataSym});  
            H_est{idx_dataSym} = tmp_H_est(:,:,scIdxs);
        end
        Xtf = permute( pagemtimes(whitening_matrix_per_RE, permute(Xtf,[3,4,1,2])), [3,4,1,2] );
        if numDmrsCdmGrpsNoData == 2
            H_est_permuted        = cellfun(@(x)permute(x, [3, 2, 1]), H_est,'UniformOutput',false);
            H_est_mat             = cat(1, H_est_permuted{:});
            [NreNsym, Ntx, Nrx]   = size(H_est_mat);
%             NreNsym               = length(scIdxs)*length(symIdx_data);
            Xtf_mat               = reshape(Xtf(scIdxs,symIdx_data,:), NreNsym, Nrx);
            LLRseq_sphere_decoder = -sphere_decoder(Xtf_mat, H_est_mat);
            Ree                   = zeros(nl,Nf); % Note: Ree in sphere decoder is invalid for now
            W{1}                  = zeros(nl,Nrx,Nf); % just dummy variables
            X_est                 = zeros(Nf,14,nl); % just dummy variables
        else
            error('Sphere decoder just supports numDmrsCdmGrpsNoData=2 for now')
        end
    else
        error('Only TdiMode=2 works with sphere decoder!')
    end
else % ZF, MMSE, MMSE-IRC    
    if TdiMode == 0 || AdditionalPosition == 0 % only use the channel estimate on the first DMRS 
        posDmrs = 1;
        if enableIrc
            [W{1}, Ree] = derive_equalizer_mmse_irc(H_est{1}, noiseIntfCov, nl, Nf, posDmrs, startPrb, nPrb, SimCtrl.subslot_proc_option);
        else 
            [W{1}, Ree] = derive_equalizer(H_est{1}, N0_ref, nl, Nf, posDmrs, SimCtrl.subslot_proc_option); 
        end
        X_est = apply_equalizer_tdi(Xtf, W{1}, symIdx_data, symIdx_dmrs, nl, Nf, cfo_est_symbols_layers, to_est, enableCfoCorrection, enableToCorrection, 0, numDmrsCdmGrpsNoData);
    
    elseif TdiMode == 1 % apply time domain interpolation on equalizers
        Ree = zeros(nl, Nf, AdditionalPosition+1);
        % derive equalizer:
        for posDmrs = 1:AdditionalPosition+1
            if enableIrc
                [W{posDmrs}, Ree(:,:,posDmrs)] = derive_equalizer_mmse_irc(H_est{posDmrs}, noiseIntfCov, nl, Nf, posDmrs, startPrb, nPrb, SimCtrl.subslot_proc_option);
            else
                [W{posDmrs}, Ree(:,:,posDmrs)] = derive_equalizer(H_est{posDmrs}, N0_ref, nl, Nf, posDmrs, SimCtrl.subslot_proc_option);   
            end
        end
        % Apply time domain interpolation to equalizer
        W_int = apply_interpolation(W, symIdx_data, symIdx_dmrs, numDmrsCdmGrpsNoData);
        % Apply equalizer
        X_est = apply_equalizer_tdi(Xtf, W_int, symIdx_data, symIdx_dmrs, nl, Nf, cfo_est_symbols_layers, to_est, enableCfoCorrection, enableToCorrection, TdiMode, numDmrsCdmGrpsNoData);
    
    elseif TdiMode == 2 % apply time domain interpolation on channel estimate
        Ree = zeros(nl, Nf, length(symIdx_data));
        % Apply time domain interpolation to channel estimate
        if SimCtrl.alg.enable_use_genie_channel_for_equalizer
            H_est = H_genie_all_sym(symIdx_data);
        else
            H_est = apply_interpolation(H_est, symIdx_data, symIdx_dmrs, numDmrsCdmGrpsNoData);
        end
        % derive equalizer:
        for posSym = 1:length(symIdx_data)
            if enableIrc
                [W{posSym}, Ree(:,:,posSym)] = derive_equalizer_mmse_irc(H_est{posSym}, noiseIntfCov, nl, Nf, posDmrs, startPrb, nPrb, SimCtrl.subslot_proc_option);
            else
                [W{posSym}, Ree(:,:,posSym)] = derive_equalizer(H_est{posSym}, N0_ref, nl, Nf, posDmrs, SimCtrl.subslot_proc_option);
            end
        end

        % Apply equalizer
        X_est = apply_equalizer_tdi(Xtf, W, symIdx_data, symIdx_dmrs, nl, Nf, cfo_est_symbols_layers, to_est, enableCfoCorrection, enableToCorrection, TdiMode, numDmrsCdmGrpsNoData);
    
    end
end

% equalizer for early HARQ
if SimCtrl.alg.enableEarlyHarq
    [W_earlyHarq, Ree_earlyHarq] = derive_equalizer(H_est_earlyHarq, noiseVar_subSlot, nl, Nf, 1, 1);
    X_est_earlyHarq = apply_equalizer_tdi(Xtf, W_earlyHarq, symIdx_data, symIdx_dmrs, nl, Nf, cfo_est_symbols_layers, to_est, 0, 0, 0, numDmrsCdmGrpsNoData);
end

% save ChEst results and genie channel for ML dataset collection
if SimCtrl.ml.dataset.enable_save_dataset
    if TdiMode > 0  % apply time domain interpolation on channel estimate
        H_est_mat       = permute(cat(4, H_est{:}),[3,4,1,2]); % dim: [Nsc, Nsym_data, Nrx, Ntx_layers]                
        tmp_H_est_mat   = H_est_mat((startPrb-1)*12+1:(startPrb-1+nPrb)*12,:,:,:);
        for idxUe = 1:nUe
            layerIdx = find(layer2Ue == (idxUe-1));
            global_idxUE = PuschParamsUeg.PuschParams(idxUe).idxUE;
            local_Hest.Hest = tmp_H_est_mat(:,:,:,layerIdx);
            local_Hest.idxUE = global_idxUE;
            local_Hest.idxUeg = PuschParamsUeg.PuschParams(idxUe).idxUeg;
            idx_pdu_in_this_UE = length(SimCtrl.ml.dataset.ChEst_perUE{global_idxUE+1}.per_pdu) + 1;
            SimCtrl.ml.dataset.ChEst_perUE{global_idxUE+1}.per_pdu{idx_pdu_in_this_UE} = local_Hest;  
        end 
        if SimCtrl.enable_get_genie_channel_matrix
            if isfield(Chan_UL{1}, 'chanMatrix_FD_oneSlot') && ~isempty(Chan_UL{1}.chanMatrix_FD_oneSlot)
                for idxUe = 1:nUe
                    global_idxUE = PuschParamsUeg.PuschParams(idxUe).idxUE;
                    numLayersThisUE = length(find(layer2Ue == (idxUe-1)));
                    local_Hgenie.Hgenie = Chan_UL{idxUe}.chanMatrix_FD_oneSlot((startPrb-1)*12+1:(startPrb-1+nPrb)*12,:,:,1:numLayersThisUE); % dim: [Nsc, Nsym, Nrx, Nlayers]
                    local_Hgenie.idxUE = global_idxUE;
                    local_Hgenie.idxUeg = PuschParamsUeg.PuschParams(idxUe).idxUeg;
                    idx_pdu_in_this_UE = length(SimCtrl.ml.dataset.ChGenie_perUE{global_idxUE+1}.per_pdu) + 1;
                    SimCtrl.ml.dataset.ChGenie_perUE{global_idxUE+1}.per_pdu{idx_pdu_in_this_UE} = local_Hgenie;  
                end 
            else
                error('Genie channel acquisition is not available.')
            end
        else
            error('Please set SimCtrl.enable_get_genie_channel_matrix to 1 for genie channel collection.')
        end
    else
        error('For ML dataset collection, TidMode should be set to 1 or 2!')
    end
end

% Save post-eq SINR parameters
num_sym_Ree = size(Ree, 3);
postEqNoiseVardB = zeros(num_sym_Ree, nUe);
postEqSinrdB = zeros(num_sym_Ree, nUe);
if SimCtrl.subslot_proc_option == 0 % full-slot processing
%     Ree = repmat(mean(Ree,3), 1,1,num_sym_Ree);
%     Ree = repmat(Ree(:,:,end), 1,1,num_sym_Ree); % Note that cuPHY just uses the last DMRS symbol Ree for soft-demapper. In the future, may consider avg Ree.
% now cuPHY uses separate DMRS Ree in soft-demapper, thus no comment out the above line of code.
else % sub-slot processing
    tmp_Ree = Ree;
    for idx_sym = 1:num_sym_Ree
        Ree(:,:,idx_sym) = mean(tmp_Ree(:,:,1:idx_sym),3);
    end
end
for idxUe = 1:nUe
   layerIdx = find(layer2Ue == (idxUe-1));
   allocRee = Ree(layerIdx,(startPrb-1)*12+1:(startPrb-1)*12+nPrb*12,:);
   avg_linear_postEqSnr = real(squeeze(mean(allocRee.^(-1),[1,2])));
   postEqNoiseVardB(:,idxUe) = -10*log10(avg_linear_postEqSnr);   
end
postEqSinrdB = -postEqNoiseVardB;

enable_plot_drms_constellation = 0;
if enable_plot_drms_constellation
    figure;Xtf_rot = Xtf.*repmat(reshape(exp(1j*2*pi*to_est(1)*[-nSc/2+1:nSc/2]), nSc,1),[1,14,2]); dmrs_res = Xtf_rot(1:2:end,3,1);plot(dmrs_res(:), '*'); grid on; title('DMRS sym.')
    figure;plot(abs(squeeze(H_est{1}(1,1,:)).*reshape(exp(1j*2*pi*to_est(ll)*[-nSc/2+1:nSc/2]), nSc,1)),'r')
end

global SimCtrl
if SimCtrl.plotFigure.constellation
    figure;
    subplot(2,1,1); plot(Xtf(:), '*'); grid on; title('before equalizer');
    subplot(2,1,2); plot(X_est(:), '+'); grid on; title('after equalizer');
    pause(1);
end

nl_offset = 0;
% backend process
for idxUe = 1:nUe
    PuschParams = PuschParamsUeg.PuschParams(idxUe);
    % load parameters
    CRC = PuschParams.CRC;
    C = PuschParams.C;
    K = PuschParams.K;
    F = PuschParams.F;
    K_prime = PuschParams.K_prime;
    BGN = PuschParams.BGN;   %1 or 2. Indicates which base graph used
    i_LS = PuschParams.i_LS; %lifting set index
    Zc = PuschParams.Zc;     %lifting size
    qam = PuschParams.qam;
    rvIdx = PuschParams.rvIdx;
    newDataIndicator = PuschParams.newDataIndicator;
    harqProcessID = PuschParams.harqProcessID;
    nl = PuschParams.nl;
    N_data = PuschParams.N_data;
    N_id = PuschParams.N_id;
    n_rnti = PuschParams.n_rnti;
    n_scid = PuschParams.n_scid;            % 0 or 1. User's dmrs scrambling id
    nPrb = PuschParams.nPrb;               % number of prbs in allocation
    startPrb = PuschParams.startPrb;       % starting prb of allocation
    Nt_data = PuschParams.Nt_data;         % number of data symbols in allocation
    Nf = PuschParams.Nf;
    symIdx_dmrs = PuschParams.symIdx_dmrs;  % Indicies of dmrs symbols (matlab 1 indexing). Dim: Nt_dmrs x 1
    maxLength = PuschParams.maxLength;
    Nref = PuschParams.Nref;
    %Parameters for UCI on PUSCH
    pduBitmap = PuschParams.pduBitmap;              % SCF Table 3-46, supported values: 1,2,3-->UL data, UCI, UCI+ data
    alphaScaling = PuschParams.alphaScaling;
    betaOffsetHarqAck = PuschParams.betaOffsetHarqAck;
    betaOffsetCsi1 = PuschParams.betaOffsetCsi1;
    betaOffsetCsi2 = PuschParams.betaOffsetCsi2;
    nBitsHarq = PuschParams.harqAckBitLength;    % Number of HARQ-ACK payload bits (to be decoded)
    nBitsCsi1 = PuschParams.csiPart1BitLength;   % Number of CSI-1 payload bits (to be decoded)
    codeRate = PuschParams.codeRate / 1024;
    nPuschSym = PuschParams.NrOfSymbols;
    startSym = PuschParams.StartSymbolIndex + 1;  % Converted to MATLAB 1 indexing
    symIdx_data = PuschParams.symIdx_data;
    %Ref. Sec. 6.3.2.1.1 38.212
    isDataPresent = bitand(uint16(pduBitmap),uint16(2^0));                     % Bit0 = 1 in pduBitmap,if data is present
    enable_multi_csiP2_fapiv3 = SimCtrl.enable_multi_csiP2_fapiv3;
    if enable_multi_csiP2_fapiv3
        isCsi2Present = (PuschParams.flagCsiPart2 > 0);
    else
        isCsi2Present = bitand(uint16(pduBitmap),uint16(2^5));                     % Bit4 = 1 in pduBitmap, if CSI2 is present (NOT a FAPI defined field)
    end
    if (~isDataPresent) && (nBitsCsi1 && ~isCsi2Present)
        if(nBitsHarq <=1)
            nBitsHarq=2;
        end
    end

    if SimCtrl.usePuschRxForPdsch
        isCsi2Present = 0;
    end

    if isCsi2Present
        % flag indicating CSI2 present:
        flagCsiPart2 = PuschParams.flagCsiPart2;

        % number of csi2 reports:
        nCsi2Reports = PuschParams.nCsi2Reports;

        % constants:
        MAX_NUM_CSI1_PRM     = SimCtrl.MAX_NUM_CSI1_PRM;
        MAX_NUM_CSI2_REPORTS = SimCtrl.MAX_NUM_CSI2_REPORTS;

        % calc CSI2 size paramaters:
        calcCsi2Size_prmSizes   = reshape(PuschParams.calcCsi2Size_prmSizes    , MAX_NUM_CSI1_PRM , MAX_NUM_CSI2_REPORTS);
        calcCsi2Size_prmOffsets = reshape(PuschParams.calcCsi2Size_prmOffsets  , MAX_NUM_CSI1_PRM , MAX_NUM_CSI2_REPORTS);
        calcCsi2Size_csi2MapIdx = PuschParams.calcCsi2Size_csi2MapIdx;
        calcCsi2Size_nPart1Prms = PuschParams.calcCsi2Size_nPart1Prms;
        
        % 10.02 API paramaters:
        rankBitOffset = PuschParams.rankBitOffset;
        rankBitSize   = PuschParams.rankBitSize;
    end


     [G_harq, G_harq_rvd, G_csi1, G] = rateMatchSeqLen(nBitsHarq,nBitsCsi1,pduBitmap,alphaScaling,...
                                                       betaOffsetHarqAck,betaOffsetCsi1,...
                                                       nPuschSym,symIdx_data,symIdx_dmrs,nPrb,...
                                                       nl,qam,C,K,codeRate,startSym,numDmrsCdmGrpsNoData);


     % see if the user is eligable for early HARQ
     earlyHarqFlag = 0;
     if SimCtrl.alg.enableEarlyHarq
         firstDmrsSymZeroIndexed  = symIdx_dmrs(1) - 1;
         firstHarqSymZeroIndexed  = firstDmrsSymZeroIndexed + maxLength;
         symOffset = 0;
         if (SimCtrl.enable_static_dynamic_beamforming && (maxLength == 2))
           symOffset = 1;
         end
         numberOfEarlyHarqSymbols = SimCtrl.alg.lastEarlyHarqSymZeroIndexed + symOffset - firstHarqSymZeroIndexed + 1;
         numberOfEarlyHarqRmBits  = numberOfEarlyHarqSymbols * nPrb * 12 * qam * nl;

        if(G_harq <= numberOfEarlyHarqRmBits)
            earlyHarqFlag = 1;
        end
     end

     harqRvdFlag = 1;
     UciRmSeqLen{idxUe}.G_harq      = G_harq ;
     UciRmSeqLen{idxUe}.G_harq_rvd  = G_harq_rvd ;
     UciRmSeqLen{idxUe}.G_csi1      = G_csi1;
     UciRmSeqLen{idxUe}.G_csi2      = 0;
     UciRmSeqLen{idxUe}.nBitsCsi2   = 0;
     UciRmSeqLen{idxUe}.G           = G;
     UciRmSeqLen{idxUe}.harqRvdFlag = harqRvdFlag;

     % DFT spread OFDM
     if enableTfPrcd == 1
         if SimCtrl.alg.dft_s_ofdm_enable_bluestein_fft
             bluestein_fft_size_lut_row1 = SimCtrl.alg.dft_s_ofdm_bluestein_fft_size_lut_row1;
             bluestein_fft_size_lut_row2 = SimCtrl.alg.dft_s_ofdm_bluestein_fft_size_lut_row2;
             idx_fft_size = find((bluestein_fft_size_lut_row1-nPrb)>=0,1,'first');
             bluestein_fft_size = bluestein_fft_size_lut_row2(idx_fft_size);
         end
         for idxSym = 1:length(symIdx_data)
             idxFreq = [(startPrb-1)*12 + 1:(startPrb-1+nPrb)*12];
             if SimCtrl.alg.dft_s_ofdm_enable_bluestein_fft
                X_est(idxFreq, idxSym) = sqrt(nPrb*12)*bluestein_ifft(X_est(idxFreq, idxSym),bluestein_fft_size);
             else
                X_est(idxFreq, idxSym) = sqrt(nPrb*12)*ifft(X_est(idxFreq, idxSym));
             end
         end

         if SimCtrl.alg.enableEarlyHarq
              for idxSym = 1:length(symIdx_data)
                 idxFreq = [(startPrb-1)*12 + 1:(startPrb-1+nPrb)*12];
                 if SimCtrl.alg.dft_s_ofdm_enable_bluestein_fft
                    X_est_earlyHarq(idxFreq, idxSym) = sqrt(nPrb*12)*bluestein_ifft(X_est_earlyHarq(idxFreq, idxSym),bluestein_fft_size);
                 else
                    X_est_earlyHarq(idxFreq, idxSym) = sqrt(nPrb*12)*ifft(X_est_earlyHarq(idxFreq, idxSym));
                 end
              end
         end

     end

    if enable_sphere_decoder
        tmp1_LLRseq = LLRseq_sphere_decoder(:,(nl_offset+1):(nl_offset+nl)); %reshape(LLRseq_sphere_decoder(:,(nl_offset+1):(nl_offset+nl)).',[],1);
        tmp2_LLRseq = reshape(tmp1_LLRseq.', nl, qam, Nt_data*length(scIdxs));
        tmp3_LLRseq = permute(tmp2_LLRseq, [2,1,3]);
        LLRseq      = reshape(tmp3_LLRseq, [], 1);
        LLR_demap = [];
    else
        [LLRseq, LLR_demap] = softDemapper(X_est, Ree, puschTable, nl, Nt_data, Nf, nPrb, startPrb, ...
            nl_offset, qam, n_scid, TdiMode, symIdx_data, symIdx_dmrs, dmrsIdx, numDmrsCdmGrpsNoData, isDataPresent);    
    end

    % early HARQ LLRs:
    if earlyHarqFlag
        dmrsIdx_earlyHarq     = cell(1);
        dmrsIdx_earlyHarq{1}  = dmrsIdx{1};
        symIdx_dmrs_earlyHarq = symIdx_dmrs(1);

        [LLRseq_earlyHarq, LLR_demap_earlyHarq] = softDemapper(X_est_earlyHarq, Ree_earlyHarq, puschTable, nl, Nt_data, Nf, nPrb, startPrb, ...
        nl_offset, qam, n_scid, 0, symIdx_data, symIdx_dmrs_earlyHarq, dmrsIdx_earlyHarq, numDmrsCdmGrpsNoData, isDataPresent);
    end

    if 0
        gpu_res = hdf5_load_nv('/home/Aerial/TVs_for_tim/TV_1_output.h5'); % load soft demapper out from cuPHY test
        gpuLlrs = eval(['gpu_res.LLR',num2str(0)]);  % Output layout is [nLLRs nLayers nTotalDataPRB*12 Nh]
        gpuLlrs = gpuLlrs.*1;
        tmp_gpuLlrs = gpuLlrs(1:4,:,:,:);
        tmp2_gpuLlrs = tmp_gpuLlrs(:);
        LLRseq = tmp2_gpuLlrs;
    end
    nl_offset = nl_offset + nl;
%     toc
    isUciPresent = bitand(uint16(pduBitmap),uint16(2^1));
    if isUciPresent

        % descriptor for deMux0
        uciOnPuschDeMuxDescr = compute_uciOnPuschMux_descriptor(nPrb, length(symIdx_data), symIdx_data, symIdx_dmrs,...
                                                                    nl, qam, G_harq, G_harq_rvd, G_csi1, 0, G,...
                                                                    nBitsHarq, N_id, n_rnti,numDmrsCdmGrpsNoData,...
                                                                    isDataPresent);
        harq_LLR_descr_alt = zeros(G_harq,1);
        csi1_LLR_descr_alt = zeros(G_csi1,1);
        LLR_descr_alt      = zeros(G,1);

        [harq_LLR_descr, csi1_LLR_descr, LLR_descr] = uciUlschDemuxDescram_part0(harq_LLR_descr_alt, csi1_LLR_descr_alt, LLR_descr_alt,...
                                                                                      LLRseq, nBitsHarq, nBitsCsi1, uciOnPuschDeMuxDescr,...
                                                                                      numDmrsCdmGrpsNoData, isDataPresent);

        if earlyHarqFlag
            [harq_LLR_descr_earlyHarq, ~, ~] = uciUlschDemuxDescram_part0(harq_LLR_descr_alt, csi1_LLR_descr_alt, LLR_descr_alt,...
                                                                                          LLRseq_earlyHarq, nBitsHarq, nBitsCsi1, uciOnPuschDeMuxDescr,...
                                                                                          numDmrsCdmGrpsNoData, isDataPresent);
        end


%         [harq_LLR_descr, csi1_LLR_descr, LLR_descr, ~]= uciUlschDemuxDescram(G_harq,G_harq_rvd,G_csi1,G,...
%                                                                              LLRseq,nPuschSym,startSym,symIdx_data,symIdx_dmrs,...
%                                                                              nPrb,nl,qam,nBitsHarq,N_id,n_rnti);

        global SimCtrl
        dtxModeUciOnPusch = SimCtrl.alg.dtxModeUciOnPusch;
        harqDTX           = 0;
        harqDTX_earlyHarq = 0;
        csi1DTX = 0;
        csi2DTX = 0;
        harqCrcFlag = 0;
        harqCrcFlag_earlyHarq = 0;
        csi1CrcFlag = 0;
        csi2CrcFlag = 0;
        harqDetStatus = 0;
        csi1DetStatus = 0;
        csi2DetStatus = 0;
        switch qam
            case {1, 2}
                confLevelFactor = 1;
            case 4
                confLevelFactor = 1.5;
            case 6
                confLevelFactor = 2;
            case 8
                confLevelFactor = 3;
            otherwise
                error('qam is not supported ...\n');
        end

        % Decode early HARQ:
        if (G_harq && earlyHarqFlag) 
            if nBitsHarq <=2
                [harqBits_est_earlyHarq, confLevel] = simplexDecode(harq_LLR_descr_earlyHarq, nBitsHarq, G_harq, qam);
                if dtxModeUciOnPusch
                    thr = 0.2;
                    confLevelThr = thr * DTXthreshold;
                    if mean(noiseVar(:)) < SimCtrl.alg.Rww_regularizer_val + 1e-10 % to handle forceRxZero case
                        harqDTX_earlyHarq = 1;
                    else
                        harqDTX_earlyHarq = (confLevel < confLevelThr);
                    end
                    if harqDTX_earlyHarq == 1
                        harqDetStatus_earlyHarq = 3;
                    else
                        harqDetStatus_earlyHarq = 4;
                    end
                end
            elseif nBitsHarq <=11
                fecDec = FecRmObj(0,G_harq,nBitsHarq);
                [strm, confLevel] = fecDec(harq_LLR_descr);
                confLevel = confLevelFactor * confLevel;
                harqBits_est_earlyHarq = str2num(strm);
                if dtxModeUciOnPusch
                    thr = 0.2; % optimized with E = 64 and A = 4
                    thr = thr * sqrt(64/G_harq)*sqrt(nBitsHarq/4);
                    thr = max(min(0.8, thr), 0.1);
                    confLevelThr = thr * DTXthreshold;
                    if mean(noiseVar(:)) < SimCtrl.alg.Rww_regularizer_val + 1e-10 % to handle forceRxZero case
                        harqDTX_earlyHarq = 1;
                    else
                        harqDTX_earlyHarq  = (confLevel < confLevelThr);
                    end
                    if harqDTX_earlyHarq  == 1
                        harqDetStatus_earlyHarq = 3;
                    else
                        harqDetStatus_earlyHarq = 4;
                    end
                end
            else
                nPolUciSegs  = nPolUciSegs + 1;
                listLength   = SimCtrl.alg.listLength;
                [harqBits_est_earlyHarq, harqCrcFlag_earlyHarq, polarBuffers{nPolUciSegs}] = uciSegPolarDecode(nBitsHarq, G_harq, listLength, harq_LLR_descr);
                if harqCrcFlag_earlyHarq == 1
                    harqDetStatus_earlyHarq = 2;
                else
                    harqDetStatus_earlyHarq = 1;
                end
            end
        end
        if G_harq
            if nBitsHarq <=2
                [harqBits_est, confLevel] = simplexDecode(harq_LLR_descr, nBitsHarq, G_harq, qam);
                if dtxModeUciOnPusch
                    thr = 0.2;
                    confLevelThr = thr * DTXthreshold;
                    if mean(noiseVar(:)) < SimCtrl.alg.Rww_regularizer_val + 1e-10 % to handle forceRxZero case
                        harqDTX = 1;
                    else
                        harqDTX = (confLevel < confLevelThr);
                    end
                    if harqDTX == 1
                        harqDetStatus = 3;
                    else
                        harqDetStatus = 4;
                    end
                end
            elseif nBitsHarq <=11
                fecDec = FecRmObj(0,G_harq,nBitsHarq);
                [strm, confLevel] = fecDec(harq_LLR_descr);
                confLevel = confLevelFactor * confLevel;
                harqBits_est = str2num(strm);
                if dtxModeUciOnPusch
                    thr = 0.2; % optimized with E = 64 and A = 4
                    thr = thr * sqrt(64/G_harq)*sqrt(nBitsHarq/4);
                    thr = max(min(0.8, thr), 0.1);
                    confLevelThr = thr * DTXthreshold;
                    if mean(noiseVar(:)) < SimCtrl.alg.Rww_regularizer_val + 1e-10 % to handle forceRxZero case
                        harqDTX = 1;
                    else
                        harqDTX = (confLevel < confLevelThr);
                    end
                    if harqDTX == 1
                        harqDetStatus = 3;
                    else
                        harqDetStatus = 4;
                    end
                end
            else
                nPolUciSegs  = nPolUciSegs + 1;
                listLength   = SimCtrl.alg.listLength;
                [harqBits_est, harqCrcFlag, polarBuffers{nPolUciSegs}] = uciSegPolarDecode(nBitsHarq, G_harq, listLength, harq_LLR_descr);
                if harqCrcFlag == 1
                    harqDetStatus = 2;
                else
                    harqDetStatus = 1;
                end
            end
        end
        if G_csi1
            if nBitsCsi1 <=2
                [csi1Bits_est, confLevel] = simplexDecode(csi1_LLR_descr, nBitsCsi1, G_csi1, qam);
                if dtxModeUciOnPusch
                    thr = 0.2;
                    confLevelThr = thr * DTXthreshold;
                    if mean(noiseVar(:)) < SimCtrl.alg.Rww_regularizer_val + 1e-10 % to handle forceRxZero case
                        csi1DTX = 1;
                    else
                        csi1DTX = (confLevel < confLevelThr);
                    end
                    if csi1DTX == 1
                        csi1DetStatus = 3;
                    else
                        csi1DetStatus = 4;
                    end
                end
            elseif nBitsCsi1 <=11
                fecDec = FecRmObj(0,G_csi1,nBitsCsi1);
                [strm, confLevel] = fecDec(csi1_LLR_descr);
                confLevel = confLevelFactor * confLevel;
                csi1Bits_est = str2num(strm);
                if dtxModeUciOnPusch
                    thr = 0.2; % optimized with E = 64 and A = 4
                    thr = thr * sqrt(64/G_csi1)*sqrt(nBitsCsi1/4);
                    thr = max(min(0.8, thr), 0.1);
                    confLevelThr = thr * DTXthreshold;
                    if mean(noiseVar(:)) < SimCtrl.alg.Rww_regularizer_val + 1e-10 % to handle forceRxZero case
                        csi1DTX = 1;
                    else
                        csi1DTX = (confLevel < confLevelThr);
                    end
                    if csi1DTX == 1
                        csi1DetStatus = 3;
                    else
                        csi1DetStatus = 4;
                    end
                end
            else
                nPolUciSegs  = nPolUciSegs + 1;
                listLength   = SimCtrl.alg.listLength;
                [csi1Bits_est, csi1CrcFlag, polarBuffers{nPolUciSegs}] = uciSegPolarDecode(nBitsCsi1, G_csi1, listLength, csi1_LLR_descr);
                if csi1CrcFlag == 1
                    csi1DetStatus = 2;
                else
                    csi1DetStatus = 1;
                end
            end
        end


        if(isCsi2Present)
            % Compute number of CSI-P2 bits:
            if enable_multi_csiP2_fapiv3
                if SimCtrl.forceCsiPart2Length
                    nBitsCsi2 = SimCtrl.forceCsiPart2Length;
                else
                    % CSI2 Map paramaters:
                    csi2Maps_bufferStartIdxs  = SimCtrl.csi2Maps_bufferStartIdxs;
                    csi2Maps_buffer           = SimCtrl.csi2Maps_buffer;

                    for csi2ReportIdx = 0 : (nCsi2Reports - 1)

                        % compute aggregated CSI1 paramaters from CSI1
                        % payload:
                        nCsi1Prms         = calcCsi2Size_nPart1Prms(csi2ReportIdx + 1);
                        aggregatedCsi1Prm = [];

                        for csi1PrmIdx = 0 : (nCsi1Prms - 1)
                            % csi1 paramater size and offset:
                            csi1PrmSize   = calcCsi2Size_prmSizes(csi1PrmIdx + 1, csi2ReportIdx + 1);
                            csi1PrmOffset = calcCsi2Size_prmOffsets(csi1PrmIdx + 1, csi2ReportIdx + 1);

                            % extract CSI1 paramater:
                            bufferIdxs = csi1PrmOffset : (csi1PrmOffset + csi1PrmSize - 1);
                            csi1Prm    = csi1Bits_est(bufferIdxs + 1).';

                            % aggreate:
                            aggregatedCsi1Prm = [aggregatedCsi1Prm csi1Prm];
                        end
                        aggregatedCsi1Prm = bin2dec(num2str(aggregatedCsi1Prm));

                        % CSI2 map:
                        csi2MapIdx         = calcCsi2Size_csi2MapIdx(csi2ReportIdx + 1);
                        csi2BufferStartIdx = csi2Maps_bufferStartIdxs(csi2MapIdx + 1);

                        % use CSI2 map and aggregated CSI1 paramater to compute
                        % size of CSI2 payload:
                        nBitsCsi2(csi2ReportIdx + 1) = csi2Maps_buffer(csi2BufferStartIdx + aggregatedCsi1Prm + 1);
                    end

                    %total number of CSI2 bits:
                    nBitsCsi2 = sum(nBitsCsi2);
                end
            else
                rankBits    = csi1Bits_est(rankBitOffset + (1 : rankBitSize));
                rank        = bin2dec(num2str(rankBits.')) + 1;
                if SimCtrl.forceCsiPart2Length
                    nBitsCsi2 = SimCtrl.forceCsiPart2Length;
                else
                    [nBitsCsi2] = csiP2PayloadSizeCalc(rank);
                end
            end

                
            % Compute number of CSI-P2 and SCH rateMatched bits:
            [~, ~, ~, G_csi2, G]= rateMatchSeqLenTx(nBitsHarq,nBitsCsi1,nBitsCsi2,pduBitmap,alphaScaling,...
                                                    betaOffsetHarqAck,betaOffsetCsi1,betaOffsetCsi2,...
                                                    nPuschSym,symIdx_data,symIdx_dmrs,nPrb,...
                                                    nl,qam,C,K,codeRate,startSym,numDmrsCdmGrpsNoData);


            UciRmSeqLen{idxUe}.G_csi2     =  G_csi2;
            UciRmSeqLen{idxUe}.nBitsCsi2  =  nBitsCsi2;
            UciRmSeqLen{idxUe}.G          =  G;


            % Segment + deScramble CSI-P2 and SCH LLRs:
            uciOnPuschDeMuxDescr = compute_uciOnPuschMux_descriptor(nPrb, length(symIdx_data), symIdx_data, symIdx_dmrs,...
                                                                    nl, qam, G_harq, G_harq_rvd, G_csi1, G_csi2, G,...
                                                                    nBitsHarq, N_id, n_rnti, numDmrsCdmGrpsNoData,...
                                                                    isDataPresent);

            csi2_LLR_descr              = zeros(G_csi2,1);
            LLR_descr                   = zeros(G,1);
            [csi2_LLR_descr, LLR_descr] = uciUlschDemuxDescram_part2(...
                csi2_LLR_descr, LLR_descr, LLRseq, uciOnPuschDeMuxDescr,...
                nBitsCsi2, numDmrsCdmGrpsNoData, isDataPresent);

            % Decode CSI-P2 payload:
            if nBitsCsi2 <=2
                [csi2Bits_est, confLevel] = simplexDecode(csi2_LLR_descr, nBitsCsi2, G_csi2, qam);
                if dtxModeUciOnPusch
                    thr = 0.2;
                    confLevelThr = thr * DTXthreshold;
                    if mean(noiseVar(:)) < SimCtrl.alg.Rww_regularizer_val + 1e-10 % to handle forceRxZero case
                        csi2DTX = 1;
                    else
                        csi2DTX = (confLevel < confLevelThr);
                    end
                    if csi2DTX == 1
                        csi2DetStatus = 3;
                    else
                        csi2DetStatus = 4;
                    end
                end
            elseif nBitsCsi2 <=11
                fecDec = FecRmObj(0,G_csi2,nBitsCsi2);
                [strm, confLevel] = fecDec(csi2_LLR_descr);
                confLevel = confLevelFactor * confLevel;
                csi2Bits_est = str2num(strm);
                if dtxModeUciOnPusch
                    thr = 0.2; % optimized with E = 64 and A = 4
                    thr = thr * sqrt(64/G_csi2)*sqrt(nBitsCsi2/4);
                    thr = max(min(0.8, thr), 0.1);
                    confLevelThr = thr * DTXthreshold;
                    if mean(noiseVar(:)) < SimCtrl.alg.Rww_regularizer_val + 1e-10 % to handle forceRxZero case
                        csi2DTX = 1;
                    else
                        csi2DTX = (confLevel < confLevelThr);
                    end
                    if csi2DTX == 1
                        csi2DetStatus = 3;
                    else
                        csi2DetStatus = 4;
                    end
                end
            else
                nPolUciSegs = nPolUciSegs + 1;
                listLength  = SimCtrl.alg.listLength;
                [csi2Bits_est, csi2CrcFlag, polarBuffers{nPolUciSegs}] = uciSegPolarDecode(nBitsCsi2, G_csi2, listLength, csi2_LLR_descr);
                if csi2CrcFlag == 1
                    csi2DetStatus = 2;
                else
                    csi2DetStatus = 1;
                end
            end
        end
    else
        % Descramble SCH:
        if numDmrsCdmGrpsNoData == 1
            LLRseq_new = [];
            symAll = [min(symIdx_data(1), symIdx_dmrs(1)) : max(symIdx_data(end), symIdx_dmrs(end))];
            symIdx_first = symAll(1);
            for symIdx = symAll % symIdx_data(1):symIdx_data(end)
                if ismember(symIdx, symIdx_dmrs)
                    LLR_demap_sym = LLR_demap(:, :, 2:2:end, symIdx - symIdx_first + 1);
                else
                    LLR_demap_sym = LLR_demap(:, :, :, symIdx - symIdx_first + 1);
                end
                LLRseq_new = [LLRseq_new; LLR_demap_sym(:)];
            end
            LLRseq = LLRseq_new;
        end
        [~,LLR_descr] = descramble_bits(LLRseq, N_id, n_rnti);
    end

    if isDataPresent

        [derateCbs, nV_parity, derateCbsIndices, derateCbsIndicesSizes] = derate_match(LLR_descr, BGN, C, qam, K, F, K_prime, Zc, nl, N_data, rvIdx, Nref,G);

        harqStatePresent = isfield(harqState,['hpid_',num2str(harqProcessID),'_rnti_',num2str(n_rnti)]);
        if (~newDataIndicator && ~harqStatePresent)
            warning(['newDataIndicator=0 and HARQ buffer not found for RNTI ',num2str(n_rnti),' and harqProcessID ',num2str(harqProcessID)]);
        end
        if (~harqStatePresent)
            temp = struct;
            temp.rxAttemptCount = 1;
            temp.llrBuffer = derateCbs;
            harqState.(['hpid_',num2str(harqProcessID),'_rnti_',num2str(n_rnti)]) = temp;
        else
            temp = harqState.(['hpid_',num2str(harqProcessID),'_rnti_',num2str(n_rnti)]);
            temp.rxAttemptCount = temp.rxAttemptCount + 1;
            fp_flag_out_llr = SimCtrl.fp_flag_pusch_demapper_out_llr;
            tmp_buffer = Varray(temp.llrBuffer, fp_flag_out_llr) + Varray(derateCbs, fp_flag_out_llr);
            temp.llrBuffer = tmp_buffer.value;
            harqState.(['hpid_',num2str(harqProcessID),'_rnti_',num2str(n_rnti)]) = temp;
            derateCbs = temp.llrBuffer;
            % Clamp LLRs
            derateCbs(derateCbs > 10000) = 10000;
            derateCbs(derateCbs < -10000) = -10000;
        end
        if 0
            gpu_res = hdf5_load_nv('/home/Aerial/TVs_for_tim/TV_1_output.h5'); % load intermediate results from cuPHY test
            gpuRateDematchCbs = eval(['gpu_res.deRmLLR',num2str(0)]); 
            derateCbs = gpuRateDematchCbs(2*Zc+1:end,:);
        end
        if strcmp(SimCtrl.alg.LDPC_DMI_method, 'fixed')
            maxNumItr_CBs = SimCtrl.alg.LDPC_maxItr*ones(1,C);
        elseif strcmp(SimCtrl.alg.LDPC_DMI_method,'LUT_spef') % LUT based on QAM and codeing rate. Note that MCS was depricated in cuPHY, thus we use QAM and coding rate here instead
            spef = qam*codeRate;
            if spef > 7.2 % hardcoding the LUT for now
                maxNumItr_CBs = 7*ones(1,C);
            elseif spef < 0.4
                 maxNumItr_CBs = 20*ones(1,C); % use more LDPC itr for MCS 0 and 1 in the MCS table with 256QAM
            else
                maxNumItr_CBs = 10*ones(1,C);
            end
        elseif strcmp(SimCtrl.alg.LDPC_DMI_method,'per_UE') 
            maxNumItr_CBs = PuschParams.ldpcMaxNumItrPerUe*ones(1,C);
        elseif strcmp(SimCtrl.alg.LDPC_DMI_method, 'LUT')  % LUT based on SINR
            effective_SNR = pow2db(2^(qam*codeRate)-1);
            SINR_thres = effective_SNR + SimCtrl.alg.LDPC_DMI_LUT_SNRmargin_dB;
            if sinrdB(end,idxUe) >= SINR_thres
                maxNumItr_CBs = SimCtrl.alg.LDPC_DMI_LUT_numItr_vec(2)*ones(1,C);
            else
                maxNumItr_CBs = SimCtrl.alg.LDPC_DMI_LUT_numItr_vec(1)*ones(1,C);
            end
        elseif strcmp(SimCtrl.alg.LDPC_DMI_method, 'ML')
            tmp_derateCCbs_percentiles = zeros(9,C);
            for idxCW = 1:C
                tmp_derateCbs = derateCbs(:,idxCW);
                tmp_derateCbs = tmp_derateCbs(tmp_derateCbs~=10000);
                tmp_derateCbs = tmp_derateCbs/db2pow(postEqSinrdB(end,idxUe));
                tmp_derateCCbs_percentiles(:,idxCW) = eval('prctile(tmp_derateCbs,[10:10:90],1)'); % figure;hold on; for ii=1:numCWs;plot(tmp_derateCCbs_percentiles(:,ii),[0.1:0.1:0.9]);end
            end
            model_input = zeros(12, C);
            model_input(1:9,:) = tmp_derateCCbs_percentiles;
            model_input(10,:) = postEqSinrdB(end,idxUe);
            model_input(11,:) = sinrdB(end,idxUe);
            model_input(12,:) = pow2db(2^(qam*codeRate)-1); % effevtive SNR
            model_input = model_input.';
            [num_samples, num_features] = size(model_input);
            model_output_numpy = eval('SimCtrl.ML.models.model_LDPC_DMI.predict(py.numpy.array(model_input).reshape(py.int(num_samples),py.int(num_features)))');
            model_output = double(model_output_numpy);
            model_output_softmax = eval("softmax(model_output.').'");
            num_model_out = size(model_output_softmax,2);
            if num_model_out == 2
                list_maxNumItr = [5, 10]; % this is the model out categories
            elseif num_model_out == 3
                list_maxNumItr = [0, 5, 10];
            else
                error('Undefined num of model out!')
            end
            [val_max, idx_max] = max(model_output_softmax,[],2); 
            maxNumItr_CBs = list_maxNumItr(idx_max);
            maxNumItr_CBs(val_max<SimCtrl.alg.LDPC_DMI_ML_confidence_thres) = SimCtrl.alg.LDPC_maxItr;
        end

        if SimCtrl.alg.LDPC_use_5Gtoolbox_flag
            TbCbs_est = nrLDPCDecode(derateCbs, BGN, maxNumItr_CBs(1), 'OutputFormat' ,'info', 'DecisionType', 'hard', ...
                                     'Algorithm', 'Normalized min-sum', 'ScalingFactor', 0.75, 'Termination', 'max');
            TbCbs_est = double(TbCbs_est);
            numItr = [];
            badItrCnt = [];
        else
            [TbCbs_est, numItr, badItrCnt] = LDPC_decode(derateCbs,nV_parity,puschTable, Zc, C, BGN, i_LS, K_prime, CRC, maxNumItr_CBs);
        end
        [TbCrc_est, cbErr] = block_desegment(TbCbs_est, C, K_prime, puschTable);
        [Tb_est, tbErr]    = CRC_decode(TbCrc_est, CRC, puschTable);

        % return payload
        pusch_payload_ueg{idxUe} = Tb_est';

        % for TB with only 1 CB, set cbErr = tbErr
        if length(cbErr) == 1
            cbErr = tbErr;
        end

        cbErrUeg{idxUe}  = cbErr;
        tbErrUeg{idxUe}  = tbErr;

        % HARQ buffer cleanup
        if (((tbErr == 0) && (harqState.(['hpid_',num2str(harqProcessID),'_rnti_',num2str(n_rnti)]).rxAttemptCount >= SimCtrl.puschHARQ.MinTransmissions)) || ...
            (harqState.(['hpid_',num2str(harqProcessID),'_rnti_',num2str(n_rnti)]).rxAttemptCount >= SimCtrl.puschHARQ.MaxTransmissions)) || ...
            (SimCtrl.puschHARQ.EnableAutoHARQ == 0) % when HARQ was disabled, we should also clear up HARQ buffer
            %display(['Clearing harqState {',num2str(harqProcessID),',',num2str(n_rnti),'} after ',num2str(harqState.(['hpid_',num2str(harqProcessID),'_rnti_',num2str(n_rnti)]).rxAttemptCount),' transmissions']);
            harqState = rmfield(harqState,['hpid_',num2str(harqProcessID),'_rnti_',num2str(n_rnti)]);
        end

        % Check number of HARQ buffers used in gNB - generate warning only
        gNB_harq_buffers_used = length(fieldnames(harqState));
        if (gNB_harq_buffers_used > SimCtrl.puschHARQ.MAX_gNB_HARQ_BUFFERS)
            warning(['PUSCH HARQ: Using ',num2str(gNB_harq_buffers_used),' > ',num2str(SimCtrl.puschHARQ.MAX_gNB_HARQ_BUFFERS),' supported.']);
        end
    else
        TbCbs_est = [];
        TbCrc_est = [];
        Tb_est = [];
        derateCbs = [];
        derateCbsIndices = [];
        derateCbsIndicesSizes = [];
        pusch_payload_ueg{idxUe} = [];
        cbErrUeg{idxUe} = [];
        tbErrUeg{idxUe} = 0;
        numItr = [];
        badItrCnt = [];
        cbErr = [];
    end


    % additional data for TV generation
    rxDataUeg{idxUe}.N0_ref = N0_ref;
    W_perm = permute(cat(4,W{:}), [4,1,2,3]);
    rxDataUeg{idxUe}.eqCoef = W_perm(:,:,:,scIdxs);
    rxDataUeg{idxUe}.X_est = X_est(scIdxs,:,:);
    if TdiMode == 2
        rxDataUeg{idxUe}.Ree = Ree(:,scIdxs,:);
    else
        rxDataUeg{idxUe}.Ree = Ree(:,scIdxs,:);
    end
    rxDataUeg{idxUe}.H_est = H_est_save;
    if SimCtrl.genTV.add_LS_chEst
        rxDataUeg{idxUe}.H_LS_est = H_LS_est_save;
    end
    rxDataUeg{idxUe}.LLR_demap = LLR_demap;
    rxDataUeg{idxUe}.TbCbs_est = TbCbs_est;
    rxDataUeg{idxUe}.numItr = numItr;
    rxDataUeg{idxUe}.badItrCnt = badItrCnt;
    rxDataUeg{idxUe}.cbErr = cbErr;
    rxDataUeg{idxUe}.derateCbs = derateCbs;
    rxDataUeg{idxUe}.derateCbsIndices = derateCbsIndices;
    rxDataUeg{idxUe}.derateCbsIndicesSizes = derateCbsIndicesSizes;
    rxDataUeg{idxUe}.LLR_descr = LLR_descr;
    rxDataUeg{idxUe}.lowPaprGroupNumber = lowPaprGroupNumber(1);
    rxDataUeg{idxUe}.lowPaprSequenceNumber = lowPaprSequenceNumber(1);
    rxDataUeg{idxUe}.cfoEstHz = cfo_est_Hz(:,find(layer2Ue == idxUe-1, 1));
    rxDataUeg{idxUe}.cfoRot = exp(-1i*2*pi*cfo_est(:,find(layer2Ue == idxUe-1, 1))*((1:14)-symIdx_dmrs(1))).';
    rxDataUeg{idxUe}.cfoAngle = cfo_est(:,find(layer2Ue == idxUe-1, 1));
    rxDataUeg{idxUe}.toEst = to_est(:,find(layer2Ue == idxUe-1, 1));
    rxDataUeg{idxUe}.toEstMicroSec = to_est_microsec(:,find(layer2Ue == idxUe-1, 1));
    rxDataUeg{idxUe}.TbCrc_est = TbCrc_est;
    rxDataUeg{idxUe}.TbPayload = Tb_est;
    rxDataUeg{idxUe}.LLR_seq = LLRseq;
    rxDataUeg{idxUe}.harq_LLR_descr           = [];
    rxDataUeg{idxUe}.harq_LLR_descr_earlyHarq = [];
    rxDataUeg{idxUe}.csi1_LLR_descr = [];
    rxDataUeg{idxUe}.csi2_LLR_descr = [];
    rxDataUeg{idxUe}.sch_LLR_descr  = [];
    rxDataUeg{idxUe}.isEarlyHarq    = 0;
    rxDataUeg{idxUe}.harq_uci_est           = [];
    rxDataUeg{idxUe}.harq_uci_est_earlyHarq = [];
    rxDataUeg{idxUe}.csi1_uci_est   = [];
    rxDataUeg{idxUe}.csi2_uci_est   = [];
    rxDataUeg{idxUe}.harqDTX   = 0;
    rxDataUeg{idxUe}.harqDTX_earlyHarq = 0;
    rxDataUeg{idxUe}.csi1DTX   = 0;
    rxDataUeg{idxUe}.csi2DTX   = 0;
    rxDataUeg{idxUe}.harqCrcFlag           = 0;
    rxDataUeg{idxUe}.harqCrcFlag_earlyHarq = 0;
    rxDataUeg{idxUe}.csi1CrcFlag   = 0;
    rxDataUeg{idxUe}.csi2CrcFlag   = 0;
    rxDataUeg{idxUe}.harqDetStatus           = 0;
    rxDataUeg{idxUe}.harqDetStatus_earlyHarq = 0;
    rxDataUeg{idxUe}.csi1DetStatus           = 0;
    rxDataUeg{idxUe}.csi2DetStatus           = 0;
    if SimCtrl.genTV.enable_logging_dbg_pusch_chest == 1
        rxDataUeg{idxUe}.ChEst_w_delay_est_H_LS_est_save = H_LS_est_save;
        rxDataUeg{idxUe}.ChEst_w_delay_est_R1 = dbg_chest.R1;
        rxDataUeg{idxUe}.ChEst_w_delay_est_delay_mean = dbg_chest.delay_mean;
        rxDataUeg{idxUe}.ChEst_w_delay_est_shiftSeq = dbg_chest.shiftSeq;
        rxDataUeg{idxUe}.ChEst_w_delay_est_shiftSeq4 = dbg_chest.shiftSeq4;
        rxDataUeg{idxUe}.ChEst_w_delay_est_unShiftSeq = dbg_chest.unShiftSeq;
        rxDataUeg{idxUe}.ChEst_w_delay_est_unShiftSeq4 = dbg_chest.unShiftSeq4;
    end
    if SimCtrl.enable_get_genie_channel_matrix && SimCtrl.batchsim.save_results_PUSCH_ChEst 
        rxDataUeg{idxUe}.H_genie = Chan_UL{idxUe}.chanMatrix_FD_oneSlot;
        if SimCtrl.N_Interfering_UE_UL>0
            rxDataUeg{idxUe}.interf_Hgenie = 10^(-interfChan_UL{1}.SIR/20)*interfChan_UL{1}.interfChanMatrix_FD_oneSlot; % just log the first interf UE genie CFR
        end
    end

    if isUciPresent
        rxDataUeg{idxUe}.sch_LLR_descr  = LLR_descr;
        if(G_harq > 0)
            rxDataUeg{idxUe}.harq_LLR_descr = harq_LLR_descr;
            rxDataUeg{idxUe}.harq_uci_est   = harqBits_est;
            rxDataUeg{idxUe}.harqDTX = harqDTX;
            rxDataUeg{idxUe}.harqCrcFlag = harqCrcFlag;
            rxDataUeg{idxUe}.harqDetStatus = harqDetStatus;

            if(earlyHarqFlag == 1)
                rxDataUeg{idxUe}.harq_LLR_descr_earlyHarq = harq_LLR_descr_earlyHarq;
                rxDataUeg{idxUe}.harq_uci_est_earlyHarq   = harqBits_est_earlyHarq;
                rxDataUeg{idxUe}.harqDTX_earlyHarq        = harqDTX_earlyHarq;
                rxDataUeg{idxUe}.harqCrcFlag_earlyHarq    = harqCrcFlag_earlyHarq;
                rxDataUeg{idxUe}.harqDetStatus_earlyHarq  = harqDetStatus_earlyHarq;
                rxDataUeg{idxUe}.isEarlyHarq              = 1;
            end

        end
        if(G_csi1 > 0)
            rxDataUeg{idxUe}.csi1_LLR_descr = csi1_LLR_descr;
            rxDataUeg{idxUe}.csi1_uci_est   = csi1Bits_est;
            rxDataUeg{idxUe}.csi1DTX = csi1DTX;
            rxDataUeg{idxUe}.csi1CrcFlag = csi1CrcFlag;
            rxDataUeg{idxUe}.csi1DetStatus = csi1DetStatus;
        end
        if(isCsi2Present)
            rxDataUeg{idxUe}.csi2_LLR_descr = csi2_LLR_descr;
            rxDataUeg{idxUe}.csi2_uci_est   = csi2Bits_est;
            rxDataUeg{idxUe}.csi2DTX = csi2DTX;
            rxDataUeg{idxUe}.csi2CrcFlag = csi2CrcFlag;
            rxDataUeg{idxUe}.csi2DetStatus = csi2DetStatus;
        end
    end

    % Metrics
    [dmrsRssiReportedDb, dmrsRssiReportedDb_ehq, dmrsRssiDb] = measureRssi(Xtf,startPrb,nPrb,symIdx_dmrs, maxLength);
    rxDataUeg{idxUe}.dmrsRssiDb = dmrsRssiDb;
    rxDataUeg{idxUe}.dmrsRssiReportedDb = dmrsRssiReportedDb;
    rxDataUeg{idxUe}.dmrsRssiReportedDb_ehq = dmrsRssiReportedDb_ehq;

    rxDataUeg{idxUe}.nCov = nCov;
    rxDataUeg{idxUe}.postEqNoiseVardB = postEqNoiseVardB(:,idxUe);
    rxDataUeg{idxUe}.noiseVar = noiseVar;
    rxDataUeg{idxUe}.noiseVardB = noiseVardB;
    rxDataUeg{idxUe}.rsrpdB = rsrpdB(:,idxUe);
    rxDataUeg{idxUe}.sinrdB = sinrdB(:,idxUe);
    rxDataUeg{idxUe}.sinrdB_ehq = 0;
    rxDataUeg{idxUe}.rsrpdB_ehq = 0;
    rxDataUeg{idxUe}.postEqSinrdB = postEqSinrdB(:,idxUe);
    % Genie metrics
    if SimCtrl.alg.enable_get_genie_meas
        rxDataUeg{idxUe}.genie_rsrp_dB = genie_rsrp_dB(idxUe);
        rxDataUeg{idxUe}.genie_noiseVar_dB = genie_noiseVar_dB;
        rxDataUeg{idxUe}.genie_pre_sinr_dB = genie_pre_sinr_dB(idxUe);
        rxDataUeg{idxUe}.genie_post_noiseVar_dB = genie_post_noiseVar_dB(idxUe);
        rxDataUeg{idxUe}.genie_post_sinr_dB = genie_post_sinr_dB(idxUe);
        rxDataUeg{idxUe}.genie_nCov = genie_nCov;
        rxDataUeg{idxUe}.genie_CQI = genie_CQI;
    end
    % ChEst error
    if  SimCtrl.enable_get_genie_channel_matrix
        rxDataUeg{idxUe}.NMSE_ChestError_dB = NMSE_ChestError_dB;
    end
    if SimCtrl.alg.ChEst_alg_selector == 1 %'MMSE_w_delay_and_spread_est'
        rxDataUeg{idxUe}.ChEst_delay_mean_microsec = delay_mean_microsec;
        rxDataUeg{idxUe}.ChEst_delay_spread_microsec = delay_spread_microsec;
    end

    % Weighted average CFO estimation
    rxDataUeg{idxUe}.foCompensationBuffer = PuschParams.foCompensationBuffer;
    rxDataUeg{idxUe}.foCompensationBufferOutput = foCompensationBuffer(idxUe);

    %fprintf('***  UE %d RSRP (dB) %2.2f *** Pre-Eq-SINR (dB) %2.2f Post-Eq-SINR (dB) %2.2f Post - Pre Eq-SINR (dB) %2.2f\n', idxUe, rxDataUeg{idxUe}.rsrpdB, rxDataUeg{idxUe}.sinrdB, rxDataUeg{idxUe}.postEqSinrdB, rxDataUeg{idxUe}.postEqSinrdB - rxDataUeg{idxUe}.sinrdB);

    if SimCtrl.usePuschRxForPdsch
        pusch_payload_ueg{idxUe} = [];
        pusch_payload_ueg{idxUe}.Tb = Tb_est;
        pusch_payload_ueg{idxUe}.tbErr = tbErr;
        pusch_payload_ueg{idxUe}.cbErr = cbErr;
    end

end

return

function [rxDataUeg] = detPusch_cuphy_early_harq_measurement(Xtf, PuschParamsUeg, puschTable, harqState, srsState, Chan_UL, interfChan_UL, Xtf_noNoise, carrier, rxDataUeg)

% frontEnd process
PuschParams = PuschParamsUeg.PuschParams(1);
nUe = length(PuschParamsUeg.PuschParams);

% polar buffers
nPolUciSegs  = 0;
polarBuffers = cell(1000,1);


% load parameters
n_scid = PuschParams.n_scid;            % 0 or 1. User's dmrs scrambling id
nPrb = PuschParams.nPrb;               % number of prbs in allocation
startPrb = PuschParams.startPrb;       % starting prb of allocation
prgSize = PuschParams.prgSize;
symIdx_data = PuschParams.symIdx_data; % indicies of data symbols. Dim: Nt_data x 1
slotNumber = PuschParams.slotNumber;
Nf = PuschParams.Nf;
Nt = PuschParams.Nt;
N_dmrs_id = PuschParams.N_dmrs_id;
symIdx_dmrs = PuschParams.symIdx_dmrs;  % Indicies of dmrs symbols (matlab 1 indexing). Dim: Nt_dmrs x 1
Nt_dmrs = PuschParams.Nt_dmrs;          % number of dmrs symbols
delta_f = PuschParams.delta_f;
maxLength = PuschParams.maxLength;
AdditionalPosition = 0;
numDmrsCdmGrpsNoData = PuschParams.numDmrsCdmGrpsNoData;
enableTfPrcd = PuschParams.enableTfPrcd;
puschIdentity = PuschParams.puschIdentity;
N_slot_frame = PuschParams.N_slot_frame;
N_symb_slot = PuschParams.N_symb_slot;
groupOrSequenceHopping = PuschParams.groupOrSequenceHopping;

enableCfoEstimation = PuschParams.enableCfoEstimation;
enableCfoCorrection = PuschParams.enableCfoCorrection;
enableWeightedAverageCfo = PuschParams.enableWeightedAverageCfo;
enableToEstimation = PuschParams.enableToEstimation;
enableToCorrection = PuschParams.enableToCorrection;
enableIrc = PuschParams.enableIrc;
Rww_regularizer_val = PuschParams.Rww_regularizer_val;
enable_nCov_shrinkage = PuschParams.enable_nCov_shrinkage;
nCov_shrinkage_method = PuschParams.nCov_shrinkage_method;
enable_use_genie_nCov = PuschParams.enable_use_genie_nCov; 
genie_nCov_method = PuschParams.genie_nCov_method; % 'genie_interfNoise_based', 'genie_interfChanResponse_based'
enableNoiseEstForZf = PuschParams.enableNoiseEstForZf;
enable_sphere_decoder = PuschParams.enable_sphere_decoder;
sphere_decoder = PuschParams.sphere_decoder;
TdiMode = PuschParams.TdiMode;
DTXthreshold = PuschParams.DTXthreshold;

nl = 0;
portIdx = [];
vec_scid = [];
layer2Ue = [];
for idxUe = 1:nUe
    PuschParams = PuschParamsUeg.PuschParams(idxUe);
    vec_scid = [vec_scid, PuschParams.n_scid*ones(1,PuschParams.nl)];
    portIdx = [portIdx, PuschParams.portIdx];
    layer2Ue(nl+1 : nl+PuschParams.nl) = idxUe-1;
    nl = nl + PuschParams.nl;
end

% 64TR
global SimCtrl;
if SimCtrl.enable_static_dynamic_beamforming
    if PuschParams.digBFInterfaces == 0
        Xtf = Xtf(:,:,1:nl);
    else
        Xtf = Xtf(:,:,1:PuschParams.digBFInterfaces);
    end
end

currentSnr = SimCtrl.alg.fakeSNRdBForZf;
N0_ref = 10^(-currentSnr/10); % current noise variance (linear)

cfo_est = zeros(max(AdditionalPosition,1), nl);
to_est = zeros(1, nl);
cfo_est_Hz = zeros(max(AdditionalPosition,1), nl);
to_est_microsec = zeros(1, nl);

% set Nt_dmrs and dmrsIdx
if maxLength == 1
    Nt_dmrs = 1;
    for posDmrs = 1:AdditionalPosition+1
        dmrsIdx{posDmrs} = symIdx_dmrs(posDmrs);
    end
else
    Nt_dmrs = 2;
    dmrsIdx{1} = symIdx_dmrs(1:2);
    if AdditionalPosition > 0
        dmrsIdx{2} = symIdx_dmrs(3:4);
    end
end

% normalize Xtf
if SimCtrl.enable_normalize_Xtf_perUeg   
    idx_RE_this_Ueg = (startPrb-1)*12+1:(startPrb+nPrb-1)*12;
    Xtf_this_Ueg = Xtf(idx_RE_this_Ueg,:,:);
    scale_val_this_Ueg = util_get_normalization_scale(Xtf_this_Ueg);
    Xtf(idx_RE_this_Ueg,:,:) = Xtf(idx_RE_this_Ueg,:,:)*scale_val_this_Ueg;
end

% step 1: generate DMRS signal
lowPaprGroupNumber = zeros(length(symIdx_dmrs), 1);
lowPaprSequenceNumber = zeros(length(symIdx_dmrs), 1);
if enableTfPrcd == 1
    [r_dmrs, lowPaprGroupNumber, lowPaprSequenceNumber] = gen_dmrs_dftsofdm(Nf, startPrb, nPrb, N_slot_frame, N_symb_slot, symIdx_dmrs, ...
    slotNumber, puschIdentity, groupOrSequenceHopping);
else
    r_dmrs = gen_dmrs_sym(slotNumber, Nf, Nt, N_dmrs_id);
end

% estimate channels on all DMRS symbols
if ((mod(nPrb,4) == 0) && (nPrb >= 8)) % use 8 PRB DMRS to estimate 4 PRBs 
    if SimCtrl.alg.ChEst_alg_selector==1 % 'MMSE_w_delay_and_spread_est'
        if SimCtrl.alg.ChEst_MMSE_w_delay_and_spread_est_opt == 1
            error('Undefined! Only defined for nPrbs%0~=0!')
        elseif SimCtrl.alg.ChEst_MMSE_w_delay_and_spread_est_opt == 2
            [H_est,delay_mean_microsec,delay_spread_microsec, dbg_chest] = pusch_ChEst_LS_delayEst_MMSE(Xtf,puschTable,slotNumber,Nf,Nt,N_dmrs_id,...
                        nl, portIdx, vec_scid, dmrsIdx, Nt_dmrs, nPrb, startPrb, ...
                        delta_f,maxLength, numDmrsCdmGrpsNoData, enableTfPrcd,...
                        N_slot_frame, N_symb_slot, puschIdentity,...
                        groupOrSequenceHopping, AdditionalPosition, carrier, prgSize, SimCtrl.bfw.enable_prg_chest, r_dmrs);
        else
            error('Unknown option!')
        end    
    elseif SimCtrl.alg.ChEst_alg_selector == 0   % 'MMSE'
        for posDmrs = 1:AdditionalPosition+1
            H_est{posDmrs} = chan_estimation(Xtf,puschTable,slotNumber,Nf,Nt,N_dmrs_id,...
                nl, portIdx, vec_scid, dmrsIdx{posDmrs}, Nt_dmrs, nPrb, startPrb,...
                delta_f, numDmrsCdmGrpsNoData, enableTfPrcd,...
                N_slot_frame, N_symb_slot, puschIdentity,...
                groupOrSequenceHopping);
        end
    elseif SimCtrl.alg.ChEst_alg_selector == 2   % 'RKHS'
        for posDmrs = 1:AdditionalPosition+1
            H_est{posDmrs} = pusch_rkhs_chEst(Xtf,puschTable,slotNumber,Nf,Nt,N_dmrs_id,...
                nl, portIdx, vec_scid, dmrsIdx{posDmrs}, Nt_dmrs, nPrb, startPrb, ...
                delta_f,maxLength, numDmrsCdmGrpsNoData, enableTfPrcd,...
                 N_slot_frame, N_symb_slot, puschIdentity,...
                 groupOrSequenceHopping,PuschParamsUeg,nUe);
        end

    end
else % use 4 PRB DMRS to estimate 2 PRBs
    if SimCtrl.alg.ChEst_alg_selector == 1 % 'MMSE_w_delay_and_spread_est'
        if SimCtrl.alg.ChEst_MMSE_w_delay_and_spread_est_opt == 1
            error('This option was removed!')
        elseif SimCtrl.alg.ChEst_MMSE_w_delay_and_spread_est_opt == 2
            [H_est,delay_mean_microsec,delay_spread_microsec, dbg_chest] = pusch_ChEst_LS_delayEst_MMSE(Xtf,puschTable,slotNumber,Nf,Nt,N_dmrs_id,...
                        nl, portIdx, vec_scid, dmrsIdx, Nt_dmrs, nPrb, startPrb, ...
                        delta_f,maxLength, numDmrsCdmGrpsNoData, enableTfPrcd,...
                        N_slot_frame, N_symb_slot, puschIdentity,...
                        groupOrSequenceHopping, AdditionalPosition, carrier, prgSize, SimCtrl.bfw.enable_prg_chest, r_dmrs);
        end
    elseif SimCtrl.alg.ChEst_alg_selector == 0 % 'MMSE'
        for posDmrs = 1:AdditionalPosition+1
            H_est{posDmrs} = apply_ChEst_main(Xtf,puschTable,slotNumber,Nf,Nt,N_dmrs_id,...
                nl, portIdx, vec_scid, dmrsIdx{posDmrs}, Nt_dmrs, nPrb, startPrb, ...
                delta_f,maxLength, numDmrsCdmGrpsNoData, enableTfPrcd,...
                 N_slot_frame, N_symb_slot, puschIdentity,...
                 groupOrSequenceHopping);
        end
    elseif SimCtrl.alg.ChEst_alg_selector == 2   % 'RKHS'
        for posDmrs = 1:AdditionalPosition+1
            H_est{posDmrs} = pusch_rkhs_chEst(Xtf,puschTable,slotNumber,Nf,Nt,N_dmrs_id,...
                nl, portIdx, vec_scid, dmrsIdx{posDmrs}, Nt_dmrs, nPrb, startPrb, ...
                delta_f,maxLength, numDmrsCdmGrpsNoData, enableTfPrcd,...
                 N_slot_frame, N_symb_slot, puschIdentity,...
                 groupOrSequenceHopping,PuschParamsUeg,nUe);
        end
    else
        error('Unknow ChEst alg!')
    end
end

scIdxs = [((startPrb-1)*12 : ((startPrb-1)+nPrb)*12 - 1) + 1];
% Save channel estimates before being corrected by CFO
H_est_save =  [];
for posDmrs = 1:AdditionalPosition+1
    H_est_save = cat(4, H_est_save, H_est{posDmrs}(:,:,scIdxs)); % dim: [nRxAnt, nTxLayers, Nsc, NdmrsSym]
end
if SimCtrl.genTV.enable_logging_dbg_pusch_chest
    H_LS_est_save =  [];
    for posDmrs = 1:AdditionalPosition+1
        H_LS_est_save = cat(4, H_LS_est_save, dbg_chest.H_LS_est{posDmrs}(scIdxs(1:2:end),:,:)); % dim: [nRxAnt, nTxLayers, Nsc, NdmrsSym]
    end
end

if SimCtrl.enable_get_genie_channel_matrix
    if isfield(Chan_UL{1}, 'chanMatrix_FD_oneSlot') && ~isempty(Chan_UL{1}.chanMatrix_FD_oneSlot) %contains(Chan_UL{1}.type,'TDL') || contains(Chan_UL{1}.type,'CDL') || contains(Chan_UL{1}.type,'AWGN') || contains(Chan_UL{1}.type,'P2P')
        H_genie_save = [];        
        for idxUe = 1:nUe
            numLayersThisUE = length(find(layer2Ue == (idxUe-1)));
            tmp_H_genie_save = Chan_UL{idxUe}.chanMatrix_FD_oneSlot(:,:,:,1:numLayersThisUE); % chanMatrix_FD_oneSlot dim: [Nsc, 14, Nrx, Ntx]
            if SimCtrl.normalize_pusch_tx_power_over_layers
                tmp_H_genie_save  = tmp_H_genie_save/sqrt(numLayersThisUE);
            end
            H_genie_save = cat(4, H_genie_save, tmp_H_genie_save);
        end 
        H_genie_save_all_sym = permute(H_genie_save, [3, 4, 1, 2]); % dim: [nRxAnt, nTxLayers, Nsc, 14]
        H_genie_save = H_genie_save_all_sym(:,:,scIdxs,cell2mat(dmrsIdx)); % dim: [nRxAnt, nTxLayers, Nsc, NdmrsSym]
%         figure; plot(real(squeeze(H_est_save(1,1,:,1)))); hold on; plot(real(squeeze(H_genie_save(1,1,:,1))));legend({'H_est','H_genie'})
        H_genie_all_sym = cell(14,1);
        for idxSym = 1:14            
            H_genie_all_sym(idxSym) = {H_genie_save_all_sym(:,:,:,idxSym)};%dim: [nRxAnt, nTxLayers, Nsc, 1]
        end
    else
        error('Genie channel matrix is not available!')
    end    
    chest_error = H_est_save - H_genie_save;
    NMSE_ChestError = sqrt(mean(abs(chest_error(:)).^2)) / sqrt(mean(abs(H_genie_save(:)).^2));
    NMSE_ChestError_dB = pow2db(NMSE_ChestError); % get the normalized MSEl

    % use genie channel for the following MIMO equalization
    if SimCtrl.alg.enable_use_genie_channel_for_equalizer
        H_est_save = H_genie_save;
        for posDmrs = 1:AdditionalPosition+1
            H_est{posDmrs}(:,:,scIdxs) = H_genie_save(:,:,:,posDmrs);
        end
    end
end

%%% Noise covariance estimation, RSSI estimation and SINR calculation

% step 1: generate DMRS signal
lowPaprGroupNumber = zeros(length(symIdx_dmrs), 1);
lowPaprSequenceNumber = zeros(length(symIdx_dmrs), 1);
if enableTfPrcd == 1
    [r_dmrs, lowPaprGroupNumber, lowPaprSequenceNumber] = gen_dmrs_dftsofdm(Nf, startPrb, nPrb, N_slot_frame, N_symb_slot, symIdx_dmrs, ...
    slotNumber, puschIdentity, groupOrSequenceHopping);
else
    r_dmrs = gen_dmrs_sym(slotNumber, Nf, Nt, N_dmrs_id);
end

XtfDmrs    = embed_dmrs_UL(zeros(size(Xtf)), r_dmrs,  nl, portIdx, n_scid, symIdx_dmrs, ...
    PuschParams.Nt_dmrs, PuschParams.energy, nPrb, startPrb, puschTable, enableTfPrcd,...
    0, []);

%XtfDmrs    = embed_dmrs_UL(Xtf, r_dmrs,  nl, portIdx, n_scid, symIdx_dmrs, ...
%    Nt_dmrs, energy, nPrb, startPrb, puschTable, enablePrcdBf, PM_W);

%XtfDmrs    = embed_dmrs_UL(zeros(size(Xtf)), r_dmrs,  nl, portIdx, n_scid, symIdx_dmrs, ...
%    maxLength*Nt_dmrs, PuschParams.energy, nPrb, startPrb, puschTable, 0, []);



% step 2: calculate r_tilde for every DMRS RE and antenna by subtracting desired signal from received
% signal
nAnt = size(Xtf,3);
numDmrsSym = length(symIdx_dmrs);
r_tilde = zeros(nPrb*12,numDmrsSym,nAnt);
if SimCtrl.normalize_pusch_tx_power_over_layers % note that we may normalize DMRS REs w.r.t. nLayers in embed_dmrs_UL(), thus we need to denormalize it below.
    denormalization_factor = sqrt(nl);
else
    denormalization_factor = 1;
end
for iant = 1:nAnt
    r_tilde(:,:,iant) = Xtf((startPrb-1)*12+1:(startPrb+nPrb-1)*12,symIdx_dmrs,iant);
    for il=1:nl
        for idmrs=1:numDmrsSym
            r_tilde(:,idmrs,iant) = r_tilde(:,idmrs,iant) - denormalization_factor*squeeze(XtfDmrs((startPrb-1)*12+1:(startPrb+nPrb-1)*12,symIdx_dmrs(idmrs),il)) .* squeeze(H_est_save(iant,il,:,ceil(idmrs*(1+AdditionalPosition)/numDmrsSym)));
        end
    end
end


% step 3: calculate covariance matrix
nCov = zeros(nAnt,nAnt,nPrb,AdditionalPosition+1);
tmp_noiseVar = zeros(nPrb, AdditionalPosition+1);
shrinkage_params = struct();
num_samples_nCov = zeros(nPrb, AdditionalPosition+1);
for ii=1:nPrb
    tmp_nCov = zeros(nAnt,nAnt,AdditionalPosition+1) + 1e-10*repmat(eye(nAnt),[1,1,AdditionalPosition+1]);    
    for jj=1:1:12
        isDataRe = (numDmrsCdmGrpsNoData == 1 && mod(jj, 2) == 0);
        if ~isDataRe
            idx_dmrs_in_slot = 1;
            for posDmrs = 1:AdditionalPosition+1
                for kk=1:length(dmrsIdx{posDmrs})
                    tmp_nCov(:,:,posDmrs) = tmp_nCov(:,:,posDmrs) + squeeze(r_tilde((ii-1)*12+jj,idx_dmrs_in_slot,:)) * conj(transpose(squeeze(r_tilde((ii-1)*12+jj,idx_dmrs_in_slot,:))));
                    num_samples_nCov(ii,posDmrs) = num_samples_nCov(ii,posDmrs) + 1;
                    idx_dmrs_in_slot = idx_dmrs_in_slot + 1;
                end
            end
        end
    end
    for posDmrs = 1:AdditionalPosition+1
        if numDmrsCdmGrpsNoData == 1
            nCov(:,:,ii,posDmrs) = tmp_nCov(:,:,posDmrs)/(6*length(dmrsIdx{posDmrs})) + Rww_regularizer_val*eye(nAnt,nAnt);
        else
            nCov(:,:,ii,posDmrs) = tmp_nCov(:,:,posDmrs)/(12*length(dmrsIdx{posDmrs})) + Rww_regularizer_val*eye(nAnt,nAnt);
        end
    end
    for posDmrs = 1:AdditionalPosition+1
        tmp_noiseVar(ii,posDmrs) = tmp_noiseVar(ii,posDmrs) + sum(abs(diag(nCov(:,:,ii,posDmrs))))/nAnt;
    end 
%     % nCov shrinkage
%     if enable_nCov_shrinkage % refer to the paper "Shrinkage Algorithms for MMSE Covariance Estimation"
%         if SimCtrl.subslot_proc_option == 0 % full-slot processing
%             Rtmp = mean(nCov(:,:,ii,:), 4);
%             T = sum(num_samples_nCov);
%             Rtmp_shrinked = nCov_shrinkage(Rtmp, T, nCov_shrinkage_method);
%             for posDmrs = 1:AdditionalPosition+1
%                 nCov(:,:,ii,posDmrs) = Rtmp_shrinked;
%             end
%         elseif  ismember(SimCtrl.subslot_proc_option, [1,2,3]) % sub-slot processing. Need to shrink nCov for each DMRS position separately
%             for posDmrs = 1:AdditionalPosition+1
%                 Rtmp = nCov(:,:,ii,posDmrs);
%                 T = num_samples_nCov(posDmrs);
%                 nCov(:,:,ii,posDmrs) = nCov_shrinkage(Rtmp, T, nCov_shrinkage_method);
%             end
%         elseif SimCtrl.subslot_proc_option == 4
%             % do nothing here. The shrinkage will be done inside derive_equalizer_mmse_irc()
%             shrinkage_params.num_samples_nCov = num_samples_nCov;
%             shrinkage_params.nCov_shrinkage_method = nCov_shrinkage_method;
%         end
%     else
%        if SimCtrl.subslot_proc_option == 0 % full-slot processing
%            Rtmp = mean(nCov(:,:,ii,:), 4);
%            for posDmrs = 1:AdditionalPosition+1
%                nCov(:,:,ii,posDmrs) = Rtmp;
%            end
%        else
%            % do nothing           
%        end
%     end

end %for ii = 1:273;a(ii)=mean(diag(real(nCov(:,:,ii,1))));end;figure;plot(diff(a));
% averaging nCov with neighboring PRBs to improve nCov est reliability. Useful when sub-slot proc enabled
if SimCtrl.alg.enable_avg_nCov_prbs_fd
    win_size_avg_nCov = SimCtrl.alg.win_size_avg_nCov_prbs_fd;
    one_side_win_size_avg_nCov = (win_size_avg_nCov-1)/2;   
    tmp_noiseVar_dB = pow2db(tmp_noiseVar);
    if SimCtrl.subslot_proc_option == 0
        max_posDmrs_allow_avg_nCov = AdditionalPosition+1;
    else
        max_posDmrs_allow_avg_nCov = AdditionalPosition; % for subslot processing, let's do freq nCov averaging on DMRS symbols except the last one
    end
    tmp2_nCov = nCov;
    for ii=1:nPrb
        for posDmrs = 1:max_posDmrs_allow_avg_nCov 
            tmp_ii_start = max(1, ii-one_side_win_size_avg_nCov);
            tmp_ii_end = min(nPrb, ii+one_side_win_size_avg_nCov);
            ii_start = min( max(tmp_ii_end-win_size_avg_nCov+1,1), tmp_ii_start);
            ii_end = max( min(tmp_ii_start+win_size_avg_nCov-1,nPrb), tmp_ii_end);
            noiseVar_dB_in_window = tmp_noiseVar_dB(ii_start:ii_end, posDmrs);
            noiseVar_dB_ii = tmp_noiseVar_dB(ii, posDmrs);
            idx_le_thres = (abs(noiseVar_dB_in_window-noiseVar_dB_ii)<=SimCtrl.alg.threshold_avg_nCov_prbs_fd_dB);
            nCov_in_window = tmp2_nCov(:,:,ii_start:ii_end,posDmrs);
            nCov(:,:,ii,posDmrs) = mean(nCov_in_window(:,:,idx_le_thres),3);
            num_samples_nCov(ii,posDmrs) = sum(idx_le_thres)*num_samples_nCov(ii,posDmrs);
%                 if ii<2
%                     nCov(:,:,1,posDmrs) = (tmp2_nCov(:,:,1,posDmrs)+tmp2_nCov(:,:,2,posDmrs))/2;
%                 elseif (ii>=2) && (ii<=nPrb-1)
%                     offset_idx_prb = [-1,1];
%                     noiseVar_offset_prb_dB = pow2db(noiseVar(ii+offset_idx_prb,posDmrs));
%                     [~,idx_min] = min(abs(noiseVar_offset_prb_dB-pow2db(noiseVar(ii,posDmrs))));
%                     offset_idx_min = offset_idx_prb(idx_min);
%                     nCov(:,:,ii,posDmrs) = (tmp2_nCov(:,:,ii,posDmrs)+tmp2_nCov(:,:,ii+offset_idx_min,posDmrs))/2;
%                 else
%                     nCov(:,:,nPrb,posDmrs) = (tmp2_nCov(:,:,nPrb-1,posDmrs)+tmp2_nCov(:,:,nPrb,posDmrs))/2;
%                 end
        end
    end
end
% nCov shrinkage
for ii=1:nPrb
    if enable_nCov_shrinkage % refer to the paper "Shrinkage Algorithms for MMSE Covariance Estimation"
        if SimCtrl.subslot_proc_option == 0 % full-slot processing
            Rtmp = mean(nCov(:,:,ii,:), 4);
            T = sum(num_samples_nCov(ii,:)); % for full-slot processing, nCov averaging happens over DMRS symbols as well
            Rtmp_shrinked = nCov_shrinkage(Rtmp, T, nCov_shrinkage_method);
            for posDmrs = 1:AdditionalPosition+1
                nCov(:,:,ii,posDmrs) = Rtmp_shrinked;
            end
        elseif  ismember(SimCtrl.subslot_proc_option, [1,2,3]) % sub-slot processing. Need to shrink nCov for each DMRS position separately
            for posDmrs = 1:AdditionalPosition+1
                Rtmp = nCov(:,:,ii,posDmrs);
                T = num_samples_nCov(ii,posDmrs);
                nCov(:,:,ii,posDmrs) = nCov_shrinkage(Rtmp, T, nCov_shrinkage_method);
            end
        end
    else
       if SimCtrl.subslot_proc_option == 0 % full-slot processing
           Rtmp = mean(nCov(:,:,ii,:), 4);
           for posDmrs = 1:AdditionalPosition+1
               nCov(:,:,ii,posDmrs) = Rtmp;
           end
       else
           % do nothing           
       end
    end
end

% get Measurement: noiseVar
noiseVar_fullSlotAvg = repmat(mean(tmp_noiseVar(:)),AdditionalPosition+1,1); % dim: (AdditionalPosition+1)x1. let's keep the same dim to the case in subslot proc

noiseVar_subSlot     = zeros(AdditionalPosition+1,1);
tmp2_noiseVar = mean(tmp_noiseVar,1); % avg across PRBs
for idx_dmrsAddPos = 1:AdditionalPosition+1
    noiseVar_subSlot(idx_dmrsAddPos,1) = mean(tmp2_noiseVar(1,1:idx_dmrsAddPos));
end

if SimCtrl.subslot_proc_option == 0 
    noiseVar = noiseVar_fullSlotAvg;
else 
    noiseVar = noiseVar_subSlot;
end

if enableNoiseEstForZf 
   N0_ref = noiseVar; % N0_ref dim: 1x(AdditionalPosition+1)
end
noiseVardB = 10*log10(mean(noiseVar,2)) + 0.5;                                          % remove 0.5 dB bias due to DMRS filtering

% get Measurement: RSRP
rsrp = zeros(AdditionalPosition+1, nUe);
for idxUe = 1:nUe
    layerIdx = find(layer2Ue == (idxUe-1));
    tmp_rsrp = zeros(AdditionalPosition+1,1);
    for ii=1:nPrb*12
        for jj=1:(AdditionalPosition+1)
            for kk=1:nAnt
                for ll=1:length(layerIdx)
                    tmp_rsrp(jj) = tmp_rsrp(jj) + abs(H_est_save(kk,layerIdx(ll),ii,jj)).^2;
                end
            end
        end
    end
    tmp_rsrp = tmp_rsrp / (nPrb*12*nAnt);
    rsrp(:, idxUe) = tmp_rsrp;    
end
if SimCtrl.subslot_proc_option == 0 % full-slot processing
    rsrp = repmat(mean(rsrp, 1), AdditionalPosition+1, 1);
else % sub-slot processing
    tmp_rsrp = rsrp;
    for idx_dmrsAddPos = 1:AdditionalPosition+1
        rsrp(idx_dmrsAddPos, :) = mean(tmp_rsrp(1:idx_dmrsAddPos, :), 1);
    end
end
rsrpdB = 10*log10(rsrp);

% get Measurement: SINR 
sinrdB = rsrpdB - repmat(noiseVardB, 1, nUe);

for idxUe = 1:nUe
    rxDataUeg{idxUe}.sinrdB_ehq = sinrdB(idxUe);
    rxDataUeg{idxUe}.rsrpdB_ehq = rsrpdB(idxUe);
end

return

function saveTV_pusch_cuphy(tvDirName, TVname, UegList, pusch_payload_list, PuschParamsList, Xtf, carrier, rxDataList, cbErrList, tbErrList, puschTable, UciRmSeqLenList)

global SimCtrl;

[status,msg] = mkdir(tvDirName);
h5File  = H5F.create([tvDirName filesep TVname '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

nPdu = length(PuschParamsList);
nUeg = length(UegList);
ueGrp_pars = [];
tb_pars = [];
reference_uci_pars =[];
tb_data = [];
accumTbPayload = [];
[~,~,nAnt] = size(Xtf);
nUcisHarq   = 0;
nUcisCsiOne = 0;
nUcisCsiTwo = 0;
nUciUes     = 0;

% ue buffer offsets
ue_refBufferOffsets = [];

% uci buffers
reference_uciPayloads  = [];
reference_uciCrcFlags  = [];
reference_uciDTXs      = [];
reference_uciHarqDetStatus = [];
reference_uciCsi1DetStatus = [];
reference_uciCsi2DetStatus = [];
nUciPayloadBytes = 0;
nUciSegs         = 0;
nPolUciSegs      = 0;

% Intermediate variables
reference_LLR_demap = [];
reference_LLR_descr = [];
reference_derateCbs = [];
reference_derateCbsIndices = [];
reference_derateCbsIndicesSizes = [];
reference_TbCbs_est = [];
expDerateCbs = [];
StartPrb = zeros(nPdu,1);

% construct ueGrp_pars
for idxUeg = 1:nUeg
    ueGrp_pars(idxUeg).nUes      = uint16(length(UegList{idxUeg}.idxPdu));
    ueGrp_pars(idxUeg).UePrmIdxs = uint16(UegList{idxUeg}.idxPdu);
    idxPdu = UegList{idxUeg}.idxPdu(1)+1;
    PuschParams = PuschParamsList{idxPdu};
    startPrb = PuschParams.startPrb;
    ueGrp_pars(idxUeg).startPrb  = uint16(startPrb-1);
    nPrb = PuschParams.nPrb;
    ueGrp_pars(idxUeg).nPrb = uint16(nPrb);
    ueGrp_pars(idxUeg).StartSymbolIndex = uint8(PuschParams.StartSymbolIndex);
    ueGrp_pars(idxUeg).NrOfSymbols = uint8(PuschParams.NrOfSymbols);
    ueGrp_pars(idxUeg).prgSize = uint16(PuschParams.prgSize);
    ueGrp_pars(idxUeg).enablePerPrgChEst = uint16(PuschParams.enable_prg_chest);

    dmrsSymLocBmsk = 0;
    for jj = 1 : PuschParams.Nt_dmrs
        dmrsIdx = PuschParams.symIdx_dmrs(jj) - 1;
        dmrsSymLocBmsk = dmrsSymLocBmsk + 2^dmrsIdx;
    end
    ueGrp_pars(idxUeg).dmrsSymLocBmsk = uint16(dmrsSymLocBmsk);
    ueGrp_pars(idxUeg).rssiSymLocBmsk = uint16(dmrsSymLocBmsk);

    if SimCtrl.enable_static_dynamic_beamforming % 64TR
        if PuschParams.digBFInterfaces == 0
            ueGrp_pars(idxUeg).nUplinkStreams = uint16(UegList{idxUeg}.nlUeg);
        else
            ueGrp_pars(idxUeg).nUplinkStreams = uint16(PuschParams.digBFInterfaces);
        end
    else
        [~, ~, ueGrp_pars(idxUeg).nUplinkStreams] = size(Xtf);
    end    
end

% find max number of CBs for all UEs
maxNcb = 0;
for idxPdu = 1:nPdu
    maxNcb = max(maxNcb, PuschParamsList{idxPdu}.C);
end
reference_ldpcNumItr = zeros(nPdu, maxNcb); % >0 means the actual number of iterations in LDPC decoding for current CB; 0 means not a CB
for idxPdu = 1:nPdu
    % initialize buffer offsets
    ue_refBufferOffsets(idxPdu).harqPayloadByteOffset = 0;
    ue_refBufferOffsets(idxPdu).nHarqBytes            = 0;
    ue_refBufferOffsets(idxPdu).harqCrcFlagOffset     = 0;
    ue_refBufferOffsets(idxPdu).csi1PayloadByteOffset = 0;
    ue_refBufferOffsets(idxPdu).nCsi1Bytes            = 0;
    ue_refBufferOffsets(idxPdu).csi1CrcFlagOffset     = 0;
    ue_refBufferOffsets(idxPdu).csi2PayloadByteOffset = 0;
    ue_refBufferOffsets(idxPdu).nCsi2Bytes            = 0;
    ue_refBufferOffsets(idxPdu).csi2CrcFlagOffset     = 0;

    PuschParams = PuschParamsList{idxPdu};
    rxData = rxDataList{idxPdu};
    pusch_payload = pusch_payload_list{idxPdu};
    UciRmSeqLen = UciRmSeqLenList{idxPdu};

    % construct tb_pars
    tb_pars(idxPdu).nRnti            = uint32(PuschParams.n_rnti);
    tb_pars(idxPdu).numLayers        = uint32(PuschParams.nl);

    % Resource allocation
    tb_pars(idxPdu).startSym = uint32(PuschParams.StartSymbolIndex);
    tb_pars(idxPdu).numSym = uint32(PuschParams.NrOfSymbols);

    % User group parameters
    tb_pars(idxPdu).userGroupIndex = uint32(PuschParams.idxUeg);

    % Back-end parameters
    tb_pars(idxPdu).dataScramId = uint32(PuschParams.N_id);
    tb_pars(idxPdu).mcsTableIndex = uint32(PuschParams.mcsTable+1);
    tb_pars(idxPdu).mcsIndex = uint32(PuschParams.mcsIndex);
    tb_pars(idxPdu).rv = uint32(PuschParams.rvIdx);
    tb_pars(idxPdu).ndi = uint32(PuschParams.newDataIndicator);
    tb_pars(idxPdu).nTbByte = uint32(PuschParams.TBS/8);
    tb_pars(idxPdu).nCb = uint32(PuschParams.C);
    tb_pars(idxPdu).I_LBRM = uint8(PuschParams.I_LBRM);
    tb_pars(idxPdu).maxLayers = uint8(PuschParams.maxLayers);
    tb_pars(idxPdu).maxQm = uint8(PuschParams.maxQm);
    tb_pars(idxPdu).n_PRB_LBRM = uint16(PuschParams.n_PRB_LBRM);
    tb_pars(idxPdu).qamModOrder = uint8(PuschParams.qam);
    tb_pars(idxPdu).targetCodeRate = uint16(PuschParams.codeRate * 10);

    % UCI on PUSCH
    tb_pars(idxPdu).nBitsHarq         = uint16(PuschParams.harqAckBitLength);
    tb_pars(idxPdu).nBitsCsi1         = uint16(PuschParams.csiPart1BitLength);
    tb_pars(idxPdu).pduBitmap         = uint16(PuschParams.pduBitmap);
    tb_pars(idxPdu).alphaScaling      = uint8(PuschParams.alphaScaling);
    tb_pars(idxPdu).betaOffsetHarqAck = uint8(PuschParams.betaOffsetHarqAck);
    tb_pars(idxPdu).betaOffsetCsi1    = uint8(PuschParams.betaOffsetCsi1);
    tb_pars(idxPdu).betaOffsetCsi2    = uint8(PuschParams.betaOffsetCsi2);

    % if(SimCtrl.enable_multi_csiP2_fapiv3 == 1)
    %     tb_pars(idxPdu).rankBitOffset = uint8(0);
    %     tb_pars(idxPdu).rankBitSize   = uint8(0);
    % else
    %     tb_pars(idxPdu).rankBitOffset = uint8(PuschParams.rankBitOffset(1));
    %     tb_pars(idxPdu).rankBitSize   = uint8(PuschParams.rankBitSize(1));
    % end

    % check if csi2 present:
    tb_pars(idxPdu).nPart1Prms     = uint8(PuschParams.calcCsi2Size_nPart1Prms);
    tb_pars(idxPdu).prmSizes       = uint8(PuschParams.calcCsi2Size_prmSizes);        
    tb_pars(idxPdu).prmOffsets     = uint16(PuschParams.calcCsi2Size_prmOffsets);
    tb_pars(idxPdu).csi2sizeMapIdx = uint16(PuschParams.calcCsi2Size_csi2MapIdx);
    tb_pars(idxPdu).nCsi2Reports   = uint16(PuschParams.nCsi2Reports);
    tb_pars(idxPdu).flagCsiPart2   = uint16(PuschParams.flagCsiPart2);
    tb_pars(idxPdu).rankBitOffset  = uint8(PuschParams.rankBitOffset(1));
    tb_pars(idxPdu).rankBitSize    = uint8(PuschParams.rankBitSize(1));


    % DMRS parameters
    tb_pars(idxPdu).dmrsAddlPosition = uint32(PuschParams.AdditionalPosition);
    tb_pars(idxPdu).dmrsMaxLength = uint32(PuschParams.maxLength);
    tb_pars(idxPdu).dmrsScramId = uint32(PuschParams.N_dmrs_id);
    tb_pars(idxPdu).nSCID = uint32(PuschParams.n_scid);
    dmrsPortBmsk = 0;
    for jj = 1 : PuschParams.nl
        portIdx = PuschParams.portIdx(jj) - 1;
        dmrsPortBmsk = dmrsPortBmsk + 2^portIdx;
    end
    tb_pars(idxPdu).dmrsPortBmsk = uint32(dmrsPortBmsk);

    dmrsSymLocBmsk = 0;
    for jj = 1 : PuschParams.Nt_dmrs
        dmrsIdx = PuschParams.symIdx_dmrs(jj) - 1;
        dmrsSymLocBmsk = dmrsSymLocBmsk + 2^dmrsIdx;
    end
    tb_pars(idxPdu).dmrsSymLocBmsk = uint32(dmrsSymLocBmsk);
    tb_pars(idxPdu).rssiSymLocBmsk = uint32(dmrsSymLocBmsk);
    tb_pars(idxPdu).numDmrsCdmGrpsNoData = uint8(PuschParams.numDmrsCdmGrpsNoData);
    
    % DTX 
    tb_pars(idxPdu).DTXthreshold= single(PuschParams.DTXthreshold);

    % DFT-s-OFDM
    tb_pars(idxPdu).enableTfPrcd = uint8(PuschParams.enableTfPrcd);
    tb_pars(idxPdu).puschIdentity = uint8(PuschParams.puschIdentity);
    tb_pars(idxPdu).N_slot_frame = uint8(PuschParams.N_slot_frame);
    tb_pars(idxPdu).N_symb_slot = uint8(PuschParams.N_symb_slot);
    tb_pars(idxPdu).groupOrSequenceHopping = uint8(PuschParams.groupOrSequenceHopping);
    tb_pars(idxPdu).lowPaprGroupNumber = uint8(rxData.lowPaprGroupNumber);
    tb_pars(idxPdu).lowPaprSequenceNumber = uint16(rxData.lowPaprSequenceNumber);
    
    % Weighted average CFO estimation
    tb_pars(idxPdu).foForgetCoeff             = single(PuschParams.foForgetCoeff);
    tb_pars(idxPdu).ldpcEarlyTerminationPerUe = single(PuschParams.ldpcEarlyTerminationPerUe);
    tb_pars(idxPdu).ldpcMaxNumItrPerUe        = single(PuschParams.ldpcMaxNumItrPerUe);

    % Output parameters for UCI on PUSCH RM function
    reference_uci_pars(idxPdu).G_harq = uint32(UciRmSeqLen.G_harq);
    reference_uci_pars(idxPdu).G_harq_rvd = uint32(UciRmSeqLen.G_harq_rvd);
    reference_uci_pars(idxPdu).G_csi1 = uint32(UciRmSeqLen.G_csi1);
    reference_uci_pars(idxPdu).G_csi2 = uint32(UciRmSeqLen.G_csi2);
    reference_uci_pars(idxPdu).nBitsCsi2 = uint16(UciRmSeqLen.nBitsCsi2);
    reference_uci_pars(idxPdu).G = uint32(UciRmSeqLen.G);
    reference_uci_pars(idxPdu).harqRvdFlag = uint8(UciRmSeqLen.harqRvdFlag); %To be removed later

    % TB data
    if UciRmSeqLen.G > 0
        TbCrc = rxData.TbCrc_est;
        if mod(length(TbCrc), 32) ~= 0
            paddedTbCrc = [TbCrc; zeros(32 - mod(length(TbCrc), 32), 1)];
        else
            paddedTbCrc = TbCrc;
        end
        tb_data = [tb_data; paddedTbCrc];
        % TB payload only
        tbPayload = rxData.TbPayload;
        if mod(length(tbPayload), 32) ~= 0
            paddedTbPayload = [tbPayload; zeros(32 - mod(length(tbPayload), 32), 1)];
        else
            paddedTbPayload = tbPayload;
        end
        accumTbPayload = [accumTbPayload; paddedTbPayload];
    end

    %zero padding up to 8 LLRs per RE
    qam = PuschParams.qam;
    LLR_demap = rxData.LLR_seq;
    numZeros = 8 - qam;
    LLR_demap_tmp = reshape(LLR_demap, qam, length(LLR_demap)/qam);
    LLR_demap_tmp = [LLR_demap_tmp ; zeros(numZeros, size(LLR_demap_tmp,2))];
    LLR_demap_tmp = LLR_demap_tmp(:);
    reference_LLR_demap = [reference_LLR_demap; LLR_demap_tmp];
    LLR_descr = rxData.LLR_descr;
    reference_LLR_descr = [reference_LLR_descr; LLR_descr];
    if UciRmSeqLen.G > 0
        K = PuschParams.K;
        Zc = PuschParams.Zc;
        derateCbs = rxData.derateCbs;
        C = size(derateCbs, 2);
        expDerateCbs = [zeros(2*Zc, C); derateCbs; zeros(3*K -2*Zc -size(derateCbs, 1), C) ];
        reference_derateCbs = [reference_derateCbs; reshape(expDerateCbs, size(expDerateCbs,1)*size(expDerateCbs,2), 1)];

        for c = 1:C
            L = rxData.derateCbsIndicesSizes(c);
            reference_derateCbsIndicesSizes = [reference_derateCbsIndicesSizes; L];
            reference_derateCbsIndices = [reference_derateCbsIndices; rxData.derateCbsIndices(1:L,c)];
        end
    end

    % write rm output LLRs:
    expDerateCbs_fp16 = expDerateCbs(:);
    expDerateCbs_fp16 = single(fp16nv(expDerateCbs_fp16, SimCtrl.fp16AlgoSel));
    saveStr = strcat('reference_rmOutLLRs',num2str(idxPdu - 1));
    hdf5_write_nv(h5File, saveStr, single(expDerateCbs_fp16));

    % Weighted average CFO estimation
    saveStr = strcat('reference_foCompensationBuffer',num2str(idxPdu - 1));
    hdf5_write_nv(h5File, saveStr, rxData.foCompensationBuffer);
    saveStr = strcat('reference_foCompensationBufferOutput',num2str(idxPdu - 1));
    hdf5_write_nv(h5File, saveStr, rxData.foCompensationBufferOutput);

    % save uci on pusch payload
    isUciPresent = bitand(uint16(PuschParamsList{idxPdu}.pduBitmap),uint16(2^1));
    if(isUciPresent)

        nBitsHarq = PuschParamsList{idxPdu}.harqAckBitLength;
        if(nBitsHarq > 0)
            nBytesHarq = ceil(nBitsHarq / 8);
            harqValues = rxData.harq_uci_est(:);
            harqValues = [harqValues; zeros(nBytesHarq*8 - nBitsHarq,1)];

            ue_refBufferOffsets(idxPdu).harqPayloadByteOffset = nUciPayloadBytes;
            ue_refBufferOffsets(idxPdu).nHarqBytes            = nBytesHarq;
            if (nBitsHarq > 11)
                ue_refBufferOffsets(idxPdu).harqCrcFlagOffset     = nPolUciSegs;
                nPolUciSegs = nPolUciSegs + 1;
            else
                ue_refBufferOffsets(idxPdu).harqCrcFlagOffset     = 0;
            end

            for byteIdx = 0 : (nBytesHarq - 1)
                byte = 0;
                for i = 0 : 7
                    byte = byte + harqValues(byteIdx*8 + i + 1) * 2^i;
                end
                reference_uciPayloads(byteIdx + nUciPayloadBytes + 1) = uint8(byte);
            end

            nUciPayloadBytes = nUciPayloadBytes + nBytesHarq;
            nUciSegs         = nUciSegs + 1;
        end


        nBitsCsi1 = PuschParamsList{idxPdu}.csiPart1BitLength;
        if(nBitsCsi1 > 0)
            nBytesCsi1 = ceil(nBitsCsi1 / 8);
            csi1Values = rxData.csi1_uci_est(:);
            csi1Values = [csi1Values; zeros(nBytesCsi1*8 - nBitsCsi1,1)];

            ue_refBufferOffsets(idxPdu).csi1PayloadByteOffset = nUciPayloadBytes;
            ue_refBufferOffsets(idxPdu).nCsi1Bytes            = nBytesCsi1;
            if(nBitsCsi1 > 11)
                ue_refBufferOffsets(idxPdu).csi1CrcFlagOffset     = nPolUciSegs;
                nPolUciSegs = nPolUciSegs + 1;
            else
                ue_refBufferOffsets(idxPdu).csi1CrcFlagOffset     = 0;
            end

            for byteIdx = 0 : (nBytesCsi1 - 1)
                byte = 0;
                for i = 0 : 7
                    byte = byte + csi1Values(byteIdx*8 + i + 1) * 2^i;
                end
                reference_uciPayloads(byteIdx + nUciPayloadBytes + 1) = uint8(byte);
            end

            nUciPayloadBytes = nUciPayloadBytes + nBytesCsi1;
            nUciSegs         = nUciSegs + 1;
        end

        nBitsCsi2 = UciRmSeqLen.nBitsCsi2;
        if(nBitsCsi2 > 0)
            nBytesCsi2 = ceil(nBitsCsi2 / 8);
            csi2Values = rxData.csi2_uci_est(:);
            csi2Values = [csi2Values; zeros(nBytesCsi2*8 - nBitsCsi2,1)];

            ue_refBufferOffsets(idxPdu).csi2PayloadByteOffset = nUciPayloadBytes;
            ue_refBufferOffsets(idxPdu).nCsi2Bytes            = nBytesCsi2;
            if(nBitsCsi2 > 11)
                ue_refBufferOffsets(idxPdu).csi2CrcFlagOffset     = nPolUciSegs;
                nPolUciSegs = nPolUciSegs + 1;
            else
                ue_refBufferOffsets(idxPdu).csi2CrcFlagOffset     = 0;
            end

            for byteIdx = 0 : (nBytesCsi2 - 1)
                byte = 0;
                for i = 0 : 7
                    byte = byte + csi2Values(byteIdx*8 + i + 1) * 2^i;
                end
                reference_uciPayloads(byteIdx + nUciPayloadBytes + 1) = uint8(byte);
            end

            nUciPayloadBytes = nUciPayloadBytes + nBytesCsi2;
            nUciSegs         = nUciSegs + 1;
        end
    end

    % if uci on pusch, write deSeg LLRs
    if(UciRmSeqLenList{idxPdu}.G_harq > 0)
        harqLLRs = single(fp16nv(rxData.harq_LLR_descr, SimCtrl.fp16AlgoSel));
        saveStr = strcat('reference_harqLLRs',num2str(nUcisHarq));
        hdf5_write_nv(h5File, saveStr, single(harqLLRs));
        if PuschParamsList{idxPdu}.harqAckBitLength > 11
            reference_uciCrcFlags = [reference_uciCrcFlags rxData.harqCrcFlag];
        end
        nUcisHarq = nUcisHarq + 1;
    end
    reference_uciDTXs  = [reference_uciDTXs, rxData.harqDTX];

    if(UciRmSeqLenList{idxPdu}.G_csi1 > 0)
        csiOneLLRs = single(fp16nv(rxData.csi1_LLR_descr, SimCtrl.fp16AlgoSel));
        saveStr = strcat('reference_csi1LLRs',num2str(nUcisCsiOne));
        hdf5_write_nv(h5File, saveStr, single(csiOneLLRs));
        if PuschParamsList{idxPdu}.csiPart1BitLength > 11
            reference_uciCrcFlags = [reference_uciCrcFlags rxData.csi1CrcFlag];
        end
        nUcisCsiOne = nUcisCsiOne + 1;
    end
    reference_uciDTXs  = [reference_uciDTXs, rxData.csi1DTX];

    if(UciRmSeqLenList{idxPdu}.G_csi2 > 0)
        csiTwoLLRs = single(fp16nv(rxData.csi2_LLR_descr, SimCtrl.fp16AlgoSel));
        saveStr = strcat('reference_csi2LLRs',num2str(nUcisCsiTwo));
        hdf5_write_nv(h5File, saveStr, single(csiTwoLLRs));
        if UciRmSeqLenList{idxPdu}.nBitsCsi2 > 11
            reference_uciCrcFlags = [reference_uciCrcFlags rxData.csi2CrcFlag];
        end
        nUcisCsiTwo = nUcisCsiTwo + 1;
    end
    reference_uciDTXs  = [reference_uciDTXs, rxData.csi2DTX];

    if((UciRmSeqLenList{idxPdu}.G_csi1 > 0) || (UciRmSeqLenList{idxPdu}.G_harq > 0))
        schLLRs = single(fp16nv(rxData.sch_LLR_descr, SimCtrl.fp16AlgoSel));
        saveStr = strcat('reference_schLLRs',num2str(nUciUes));
        hdf5_write_nv(h5File, saveStr, single(schLLRs));
        nUciUes = nUciUes + 1;
    end

    reference_uciHarqDetStatus = [reference_uciHarqDetStatus, rxData.harqDetStatus];
    reference_uciCsi1DetStatus = [reference_uciCsi1DetStatus, rxData.csi1DetStatus];
    reference_uciCsi2DetStatus = [reference_uciCsi2DetStatus, rxData.csi2DetStatus];

    saveStr = strcat('reference_TbCbs_est',num2str(idxPdu - 1));
    hdf5_write_nv(h5File, saveStr, uint8(rxData.TbCbs_est));
    TbCbs_est = rxData.TbCbs_est;
    reference_TbCbs_est = [reference_TbCbs_est; reshape(TbCbs_est, size(TbCbs_est,1)*size(TbCbs_est,2), 1)];

    StartPrb(idxPdu) = PuschParams.startPrb - 1;

    % cast buffer offsets
    ue_refBufferOffsets(idxPdu).harqPayloadByteOffset = uint32(ue_refBufferOffsets(idxPdu).harqPayloadByteOffset);
    ue_refBufferOffsets(idxPdu).nHarqBytes            = uint32(ue_refBufferOffsets(idxPdu).nHarqBytes);
    ue_refBufferOffsets(idxPdu).harqCrcFlagOffset     = uint32(ue_refBufferOffsets(idxPdu).harqCrcFlagOffset);
    ue_refBufferOffsets(idxPdu).csi1PayloadByteOffset = uint32(ue_refBufferOffsets(idxPdu).csi1PayloadByteOffset);
    ue_refBufferOffsets(idxPdu).nCsi1Bytes            = uint32(ue_refBufferOffsets(idxPdu).nCsi1Bytes);
    ue_refBufferOffsets(idxPdu).csi1CrcFlagOffset     = uint32(ue_refBufferOffsets(idxPdu).csi1CrcFlagOffset);
    ue_refBufferOffsets(idxPdu).csi2PayloadByteOffset = uint32(ue_refBufferOffsets(idxPdu).csi2PayloadByteOffset);
    ue_refBufferOffsets(idxPdu).nCsi2Bytes            = uint32(ue_refBufferOffsets(idxPdu).nCsi2Bytes);
    ue_refBufferOffsets(idxPdu).csi2CrcFlagOffset     = uint32(ue_refBufferOffsets(idxPdu).csi2CrcFlagOffset);

    % actual number of iterations in LDPC decoding for current UE's CBs
    if UciRmSeqLen.G > 0 % if there are decoded TB data
        reference_ldpcNumItr(idxPdu, 1:PuschParams.C) = uint8(rxData.numItr);
    end
end

% tb_data_uint8 = zeros(1, length(tb_data)/8);
% for ii=1:length(tb_data)/8
%     tmp = num2str(tb_data((ii-1)*8+1:ii*8)');
%     tb_data_uint8(ii) = bin2dec(tmp);
% end
tb_data_uint8 = uint8_convert(tb_data, 0)';

% contruct debug paramaters
debug_pars = [];
if SimCtrl.forceCsiPart2Length
    debug_pars.forceNumCsi2BitsFlag = uint8(1);
    debug_pars.forcedNumCsi2Bits    = uint16(SimCtrl.forceCsiPart2Length);
else
    debug_pars.forceNumCsi2BitsFlag = uint8(0);
    debug_pars.forcedNumCsi2Bits    = uint16(0);
end
    


% construct gnb_pars
gnb_pars.nUserGroups = nUeg;
gnb_pars.mu = uint32(carrier.mu);
gnb_pars.nRx = uint32(nAnt);  % CHECK
gnb_pars.nPrb = uint32(carrier.N_grid_size_mu);
gnb_pars.cellId = uint32(carrier.N_ID_CELL);
gnb_pars.slotNumber = uint32(PuschParams.slotNumber);
gnb_pars.numTb = uint32(nPdu);    % CHECK
gnb_pars.enableEarlyHarq = uint8(SimCtrl.alg.enableEarlyHarq);
gnb_pars.enableCfoCorrection = uint8(SimCtrl.alg.enableCfoCorrection); %uint32(0);
gnb_pars.enableCfoEstimation = uint8(SimCtrl.alg.enableCfoEstimation); %uint32(0);
gnb_pars.enableWeightedAverageCfo = uint8(SimCtrl.alg.enableWeightedAverageCfo); %uint32(0);
gnb_pars.enableToEstimation = uint8(SimCtrl.alg.enableToEstimation); %uint32(0);
gnb_pars.enableToCorrection = uint8(SimCtrl.alg.enableToCorrection); %uint32(0);
gnb_pars.TdiMode = uint8(SimCtrl.alg.TdiMode); %uint32(0);
gnb_pars.enableDftSOfdm = uint8(SimCtrl.alg.enableDftSOfdm); 
gnb_pars.enableRssiMeasurement = uint8(SimCtrl.alg.enableRssiMeas);
gnb_pars.enableSinrMeasurement = uint8(SimCtrl.alg.enableSinrMeas);
gnb_pars.enable_static_dynamic_beamforming = uint8(SimCtrl.enable_static_dynamic_beamforming);
% gnb_pars.enableIrc = uint8(SimCtrl.alg.enableIrc); % deprecated flag in cuPHY
gnb_pars.ldpcEarlyTermination = uint32(SimCtrl.alg.LDPC_enableEarlyTerm);
gnb_pars.ldpcAlgoIndex = uint32(0);
gnb_pars.ldpcFlags = uint32(SimCtrl.alg.LDPC_flags);
gnb_pars.ldpcUseHalf = uint32(1);
gnb_pars.numBbuLayers = uint32(PuschParams.nlUeg);
gnb_pars.ldpcMaxNumItr = uint8(SimCtrl.alg.LDPC_maxItr);
if strcmp(SimCtrl.alg.LDPC_DMI_method,'fixed')
    gnb_pars.ldpcMaxNumItrAlgIdx = uint8(0);
elseif strcmp(SimCtrl.alg.LDPC_DMI_method,'LUT_spef')
    gnb_pars.ldpcMaxNumItrAlgIdx = uint8(1); % for simplicity, the LUT will be hardcoded instead of API configurable for now.
elseif strcmp(SimCtrl.alg.LDPC_DMI_method,'per_UE')
    gnb_pars.ldpcMaxNumItrAlgIdx = uint8(2);
elseif strcmp(SimCtrl.alg.LDPC_DMI_method,'ML')  
    gnb_pars.ldpcMaxNumItrAlgIdx = uint8(3); % placeholder for machine learning method
end
if SimCtrl.alg.ChEst_alg_selector == 0 % 'MMSE'
    gnb_pars.dmrsChEstAlgIdx = uint8(0);
elseif SimCtrl.alg.ChEst_alg_selector == 1 %'MMSE_w_delay_and_spread_est'
    gnb_pars.dmrsChEstAlgIdx = uint8(1);
elseif SimCtrl.alg.ChEst_alg_selector == 2 %'RKHS ChEst'
    gnb_pars.dmrsChEstAlgIdx = uint8(2);
end

gnb_pars.enablePerPrgChEst = uint8(SimCtrl.bfw.enable_prg_chest);

eqCoeffAlgoIdx = 0; % rZF
if(SimCtrl.alg.enableNoiseEstForZf)
    eqCoeffAlgoIdx = 1; % noise diagonal MMSE
end
if(SimCtrl.alg.enableIrc)
    eqCoeffAlgoIdx = 2; % MMSE-IRC without shrinkage
    if SimCtrl.alg.enable_nCov_shrinkage
        if SimCtrl.alg.nCov_shrinkage_method==0
            eqCoeffAlgoIdx = 3; % MMSE-IRC with shrinkage of method RBLW
        elseif SimCtrl.alg.nCov_shrinkage_method==1
            eqCoeffAlgoIdx = 4; % MMSE-IRC with shrinkage of method OAS
        else
        end
    end
end
gnb_pars.eqCoeffAlgoIdx    = uint8(eqCoeffAlgoIdx);
gnb_pars.listLength        = uint8(SimCtrl.alg.listLength);
gnb_pars.enableCsiP2Fapiv3 = uint8(SimCtrl.enable_multi_csiP2_fapiv3);

% CSI-P2 size mapping paramaters:
gnb_pars(1).nCsi2Maps                = uint16(SimCtrl.nCsi2Maps);
gnb_pars(1).csi2Maps_sumOfPrmSizes   = uint8(SimCtrl.csi2Maps_sumOfPrmSizes);
gnb_pars(1).csi2Maps_bufferStartIdxs = uint32(SimCtrl.csi2Maps_bufferStartIdxs);
gnb_pars(1).csi2Maps_buffer          = uint16(SimCtrl.csi2Maps_buffer);


hdf5_write_nv(h5File, 'tb_data', uint8(tb_data_uint8));
hdf5_write_nv(h5File, 'tb_payload', uint8(uint8_convert(accumTbPayload, 0)'));
hdf5_write_nv_exp(h5File, 'gnb_pars', gnb_pars);
hdf5_write_nv_exp(h5File, 'ueGrp_pars', ueGrp_pars);
hdf5_write_nv_exp(h5File, 'tb_pars', tb_pars);
hdf5_write_nv(h5File, 'debug_pars', debug_pars);

hdf5_write_nv(h5File, 'reference_LLR_descr'            , single(reference_LLR_descr));
hdf5_write_nv(h5File, 'reference_derateCbs'            , single(reference_derateCbs));
hdf5_write_nv(h5File, 'reference_derateCbsIndices'     , uint32(reference_derateCbsIndices));
hdf5_write_nv(h5File, 'reference_derateCbsIndicesSizes', uint32(reference_derateCbsIndicesSizes));
hdf5_write_nv(h5File, 'reference_TbCbs_est'            , single(reference_TbCbs_est));
hdf5_write_nv(h5File, 'reference_uci_pars'             , reference_uci_pars);
hdf5_write_nv(h5File, 'reference_ldpcNumItr'           , reference_ldpcNumItr);

% write harq uci
firstHarqUciCb = -1;
for idxUe = 1:nPdu
    harq_uci_est = rxDataList{idxUe}.harq_uci_est;
    nHarqBits    = length(harq_uci_est);

    if(nHarqBits > 0)
        firstHarqUciCb = 0;
        for bitIdx = 0 : (nHarqBits - 1)
            if(harq_uci_est(bitIdx + 1) == '1')
                firstHarqUciCb = firstHarqUciCb + 2^bitIdx;
            end
        end
        break;
    end
end

if(firstHarqUciCb >= 0)
    hdf5_write_nv(h5File, 'reference_firstHarqUciCb', uint32(firstHarqUciCb));
end


% Write equalizer output LLRs
max_num_dmrsPos = 0;
for ueGrpIdx = 0 : (nUeg - 1)
    if size(rxDataList{ueGrpIdx+1}.noiseVardB,1)>=max_num_dmrsPos
        max_num_dmrsPos = size(rxDataList{ueGrpIdx+1}.noiseVardB,1);
    end
end
noiseVardB_vec = zeros(nUeg,max_num_dmrsPos);
ueGrpIdxOffset = 0;
for ueGrpIdx = 0 : (nUeg - 1)
    ueGrpRxData = rxDataList{ueGrpIdxOffset + 1};
    eqOutLLRs = ueGrpRxData.LLR_demap;
    % zero pad LLRs to 8 bits:
    [nBits,nLayers,nSubcarriers,nSymbols] = size(eqOutLLRs);
    eqOutputLLRs_zp = zeros(8,0,nSubcarriers,nSymbols);
    for idxPdu = 1:nPdu
        if tb_pars(idxPdu).userGroupIndex == ueGrpIdx 
            eqOutLLRs_tmp = rxDataList{idxPdu}.LLR_demap;
            [nBits,nLayers,nSubcarriers,nSymbols] = size(eqOutLLRs_tmp);
            eqOutputLLRs_zp_tmp = zeros(8,nLayers,nSubcarriers,nSymbols);
            eqOutputLLRs_zp_tmp(1 : nBits,:,:,:) = eqOutLLRs_tmp;
            eqOutputLLRs_zp = cat(2, eqOutputLLRs_zp, eqOutputLLRs_zp_tmp);
        end
    end
    eqOutputLLRs_zp = fp16nv(eqOutputLLRs_zp, SimCtrl.fp16AlgoSel);
    saveStr = strcat('reference_eqOutLLRs',num2str(ueGrpIdx));
    hdf5_write_nv(h5File, saveStr, single(eqOutputLLRs_zp));
    saveStr = strcat('reference_H_est',num2str(ueGrpIdx));
    hdf5_write_nv(h5File, saveStr, complex(single(ueGrpRxData.H_est)));
    if SimCtrl.genTV.add_LS_chEst
        saveStr = strcat('reference_H_LS_est',num2str(ueGrpIdx));
        hdf5_write_nv(h5File, saveStr, complex(single(ueGrpRxData.H_LS_est)));
    end
    saveStr = strcat('reference_Eq_coef',num2str(ueGrpIdx));
    hdf5_write_nv(h5File, saveStr, complex(single(ueGrpRxData.eqCoef)));
    saveStr = strcat('reference_X_est',num2str(ueGrpIdx));
    hdf5_write_nv(h5File, saveStr, complex(single(ueGrpRxData.X_est)));
    saveStr = strcat('reference_Ree',num2str(ueGrpIdx));
    hdf5_write_nv(h5File, saveStr, single(ueGrpRxData.Ree));
    saveStr = strcat('reference_cfoEst',num2str(ueGrpIdx));
    hdf5_write_nv(h5File, saveStr, complex(single(ueGrpRxData.cfoRot)));
    saveStr = strcat('reference_cfoAngle',num2str(ueGrpIdx));
    hdf5_write_nv(h5File, saveStr, single(ueGrpRxData.cfoAngle));
    saveStr = strcat('reference_cfoEstHz',num2str(ueGrpIdx));
    hdf5_write_nv(h5File, saveStr, single(ueGrpRxData.cfoEstHz));
    saveStr = strcat('reference_taEst',num2str(ueGrpIdx));
    hdf5_write_nv(h5File, saveStr, single(ueGrpRxData.toEst));
    saveStr = strcat('reference_taEstSec',num2str(ueGrpIdx)); % for cuPHY unit test TV
    hdf5_write_nv(h5File, saveStr, single(ueGrpRxData.toEstMicroSec*1e-6));
    saveStr = strcat('reference_taEstMicroSec',num2str(ueGrpIdx));
    hdf5_write_nv(h5File, saveStr, single(ueGrpRxData.toEstMicroSec));
    saveStr = strcat('reference_rssiFull',num2str(ueGrpIdx));
    hdf5_write_nv(h5File, saveStr, single(ueGrpRxData.dmrsRssiDb));
    saveStr = strcat('reference_rssi',num2str(ueGrpIdx));
    hdf5_write_nv(h5File, saveStr, single(ueGrpRxData.dmrsRssiReportedDb));
    saveStr = strcat('reference_rssi_ehq',num2str(ueGrpIdx));
    hdf5_write_nv(h5File, saveStr, single(ueGrpRxData.dmrsRssiReportedDb_ehq));
    saveStr = strcat('reference_nCov',num2str(ueGrpIdx));
    hdf5_write_nv(h5File, saveStr, single(ueGrpRxData.nCov),'fp32');
    noiseVardB_vec(ueGrpIdx+1,1:length(ueGrpRxData.noiseVardB)) = ueGrpRxData.noiseVardB;
    %postEqNoiseVardB_vec(ueGrpIdx+1) = ueGrpRxData.postEqNoiseVardB;
    if SimCtrl.genTV.enable_logging_dbg_pusch_chest == 1
        saveStr = strcat('reference_ChEst_w_delay_est_H_LS_est_save',num2str(ueGrpIdx));
        hdf5_write_nv(h5File, saveStr, complex(single(ueGrpRxData.ChEst_w_delay_est_H_LS_est_save)));
        saveStr = strcat('reference_ChEst_w_delay_est_R1',num2str(ueGrpIdx));
        hdf5_write_nv(h5File, saveStr, complex(single(ueGrpRxData.ChEst_w_delay_est_R1)));
        saveStr = strcat('reference_ChEst_w_delay_est_delay_mean',num2str(ueGrpIdx));
        hdf5_write_nv(h5File, saveStr, (single(ueGrpRxData.ChEst_w_delay_est_delay_mean)));
        saveStr = strcat('reference_ChEst_w_delay_est_shiftSeq',num2str(ueGrpIdx));
        hdf5_write_nv(h5File, saveStr, complex(single(ueGrpRxData.ChEst_w_delay_est_shiftSeq)));
        saveStr = strcat('reference_ChEst_w_delay_est_shiftSeq4',num2str(ueGrpIdx));
        hdf5_write_nv(h5File, saveStr, complex(single(ueGrpRxData.ChEst_w_delay_est_shiftSeq4)));
        saveStr = strcat('reference_ChEst_w_delay_est_unShiftSeq',num2str(ueGrpIdx));
        hdf5_write_nv(h5File, saveStr, complex(single(ueGrpRxData.ChEst_w_delay_est_unShiftSeq)));
        saveStr = strcat('reference_ChEst_w_delay_est_unShiftSeq4',num2str(ueGrpIdx));
        hdf5_write_nv(h5File, saveStr, complex(single(ueGrpRxData.ChEst_w_delay_est_unShiftSeq4)));
    end
    ueGrpIdxOffset = ueGrpIdxOffset + ueGrp_pars(ueGrpIdx + 1).nUes; % Note: indexing assumes UEs follow a linear sequential access order across UE groups
end
%hdf5_write_nv(h5File, 'reference_noiseVardB', single(noiseVardB_vec),'fp32');
%hdf5_write_nv(h5File, 'reference_postEqNoiseVardB', single(postEqNoiseVardB_vec),'fp32');

% write buffer offsets
hdf5_write_nv_exp(h5File, 'ue_refBufferOffsets', ue_refBufferOffsets);

% write uci buffers
if(nUciUes > 0)
    hdf5_write_nv_exp(h5File, 'reference_uciPayloads', uint8(reference_uciPayloads(:)));
    hdf5_write_nv_exp(h5File, 'reference_uciDTXs', uint8(reference_uciDTXs'));
    if(nPolUciSegs > 0)
        hdf5_write_nv_exp(h5File, 'reference_uciCrcFlags', uint8(reference_uciCrcFlags'));
    else
        hdf5_write_nv_exp(h5File, 'reference_uciCrcFlags', uint8([0]));
    end
end

hdf5_write_nv_exp(h5File, 'reference_uciHarqDetStatus', uint8(reference_uciHarqDetStatus'));
hdf5_write_nv_exp(h5File, 'reference_uciCsi1DetStatus', uint8(reference_uciCsi1DetStatus'));
hdf5_write_nv_exp(h5File, 'reference_uciCsi2DetStatus', uint8(reference_uciCsi2DetStatus'));

% save tbErr
hdf5_write_nv(h5File, 'tbErr', uint32(cell2mat(tbErrList)));

% save cbErr
for idxPdu = 1:nPdu
    hdf5_write_nv(h5File, ['cbErr', num2str(idxPdu-1)], uint32(cbErrList{idxPdu}));
end

% For CFO estimation
hdf5_write_nv(h5File, 'dmrsSymbIdxs'      , PuschParams.symIdx_dmrs);
% To be removed
hdf5_write_nv(h5File, 'reference_H_est'   , complex(single(rxData.H_est)));
if SimCtrl.genTV.add_LS_chEst
    hdf5_write_nv(h5File, 'reference_H_LS_est'   , complex(single(rxData.H_LS_est)));
end
hdf5_write_nv(h5File, 'reference_cfoEst'  , single(rxData.cfoRot));
hdf5_write_nv(h5File, 'reference_cfoAngle', single(rxData.cfoAngle));
hdf5_write_nv(h5File, 'reference_rssiFull', single(rxData.dmrsRssiDb));
hdf5_write_nv(h5File, 'reference_rssi'    , single(rxData.dmrsRssiReportedDb));
hdf5_write_nv(h5File, 'reference_rssi_ehq'    , single(rxData.dmrsRssiReportedDb_ehq));

Xtf0 = Xtf;
Xtf = reshape(fp16nv(double(real(Xtf)), SimCtrl.fp16AlgoSel) + 1i*fp16nv(double(imag(Xtf)), SimCtrl.fp16AlgoSel), [size(Xtf)]);
% [rssiReportedDb, rssiDb] = measureRssi(Xtf,tb_pars.startPrb+1,tb_pars.numPRb,PuschParams.symIdx_dmrs);

hdf5_write_nv(h5File, 'DataRx', complex(single(Xtf)),'fp16');
hdf5_write_nv(h5File, 'X_tf', complex(single(Xtf)));

% dump Tx X_tf
global SimCtrl
if SimCtrl.enable_snapshot_gNB_UE_into_SimCtrl && SimCtrl.genTV.enable_logging_tx_Xtf
    num_UEs = length(SimCtrl.gNBUE_snapshot.UE);
    for idx_UE = 1:num_UEs
        idxStr = ['_', num2str(idx_UE-1)];
        X_tf_transmitted_from_UE = SimCtrl.gNBUE_snapshot.UE{idx_UE}.Phy.tx.Xtf;
        hdf5_write_nv(h5File, ['X_tf_transmitted_from_UE', idxStr], X_tf_transmitted_from_UE);
    end
end
if SimCtrl.genTV.enable_logging_carrier_and_channel_info
    saveCarrierChanPars(h5File, SimCtrl, carrier);
end

% ChEst filters
W_middle   = puschTable.W_middle;
W_upper    = puschTable.W_upper;
W_lower    = puschTable.W_lower;

WFreq      = reshape([W_middle W_lower W_upper], [size(W_middle,1) size(W_middle, 2) 3]);
hdf5_write_nv_exp(h5File, 'WFreq', single(WFreq));

W4_middle   = puschTable.W4_middle;
W4_upper    = puschTable.W4_upper;
W4_lower    = puschTable.W4_lower;

W4Freq      = reshape([W4_middle W4_lower W4_upper], [size(W4_middle,1) size(W4_middle, 2) 3]);
hdf5_write_nv_exp(h5File, 'WFreq4', single(W4Freq));

W3_padded = puschTable.W3;

W2_padded = zeros(37,18);
W2_padded(1 : 25, 1:12) = puschTable.W2;

W1_padded = zeros(37,18);
W1_padded(1 : 13, 1 : 6) = puschTable.W1;

WSmallFreq = reshape([W1_padded W2_padded W3_padded], [size(W1_padded,1) size(W1_padded, 2) 3]);
hdf5_write_nv_exp(h5File, 'WFreqSmall', single(WSmallFreq));

%ChEst sequences:
s_grid   = puschTable.shiftSeq;
s = puschTable.unShiftSeq;
hdf5_write_nv(h5File, 'ShiftSeq', single(s_grid(1:8*6)),'fp16');
hdf5_write_nv(h5File, 'UnShiftSeq', single(s(1:97)),'fp16');

shiftSeq4   = puschTable.shiftSeq4;
unShiftSeq4 = puschTable.unShiftSeq4;
hdf5_write_nv(h5File, 'ShiftSeq4', single(shiftSeq4),'fp16');
hdf5_write_nv(h5File, 'UnShiftSeq4', single(unShiftSeq4),'fp16');

hdf5_write_nv(h5File, 'StartPrb', uint16(StartPrb));
% channel equalization inputs
hdf5_write_nv(h5File, 'Data_sym_loc', uint8(PuschParams.symIdx_data));
hdf5_write_nv(h5File, 'RxxInv', single(1));
% N0_est:
nBSAnts = nAnt;
RwwInv  = (1/rxData.N0_ref(end))*eye(nBSAnts);
nMaxNoiseEstPrb = 0;
for i = 1:length(PuschParamsList)
    nMaxNoiseEstPrb = max(PuschParamsList{i}.nPrb, nMaxNoiseEstPrb);
end

N0_est  = reshape(repmat(RwwInv(:), [nMaxNoiseEstPrb 1]), [nBSAnts, nBSAnts, nMaxNoiseEstPrb]);
N0_est  = complex(single(N0_est), single(zeros(size(N0_est))));
hdf5_write_nv_exp(h5File, 'Noise_pwr', N0_est);

% hdf5_write_nv(h5File, 'PuschParams', PuschParams);
% hdf5_write_nv(h5File, 'x_payload', uint32(pusch_payload));

% SINR metrics
num_dmrsPos = max_num_dmrsPos;
num_rxData = length(rxDataList);
rsrpdB_vec = zeros(num_dmrsPos, num_rxData);
sinrdB_vec = zeros(num_dmrsPos, num_rxData);
sinrdB_ehq_vec = zeros(1, num_rxData);
rsrpdB_ehq_vec = zeros(1, num_rxData);
postEqSinrdB_vec = zeros(num_dmrsPos, num_rxData);
noiseVardBPerUe_vec = zeros(num_dmrsPos, num_rxData);
postEqNoiseVardB_vec = zeros(num_dmrsPos, num_rxData);
cfoEstHzPerUe_vec = zeros(max(1,num_dmrsPos-1), num_rxData);
taEstMicroSecPerUe_vec = zeros(num_dmrsPos, num_rxData);
postEqSinrdB = zeros(1, num_rxData);
postEqNoiseVardB = zeros(1, num_rxData);

for ii=1:length(rxDataList)
    %if ii==1
        %hdf5_write_nv(h5File, 'nCov', single(rxDataList{1}.nCov),'fp32');
        %hdf5_write_nv(h5File, 'postEqNoiseVardB', single(rxDataList{1}.postEqNoiseVardB),'fp32');
    %end
    num_valid_dmrsPos = length(rxDataList{ii}.rsrpdB);
    rsrpdB_vec(1:num_valid_dmrsPos,ii) = rxDataList{ii}.rsrpdB;
    sinrdB_vec(1:num_valid_dmrsPos,ii) = rxDataList{ii}.sinrdB;
    sinrdB_ehq_vec(ii) = rxDataList{ii}.sinrdB_ehq;
    rsrpdB_ehq_vec(ii) = rxDataList{ii}.rsrpdB_ehq;
    postEqSinrdB_vec(1:num_valid_dmrsPos,ii) = rxDataList{ii}.postEqSinrdB;
    noiseVardBPerUe_vec(1:num_valid_dmrsPos,ii) = rxDataList{ii}.noiseVardB;
    postEqNoiseVardB_vec(1:num_valid_dmrsPos,ii) = rxDataList{ii}.postEqNoiseVardB;
    cfoEstHzPerUe_vec(1:length(rxDataList{ii}.cfoEstHz),ii) = rxDataList{ii}.cfoEstHz;
    taEstMicroSecPerUe_vec(1:num_valid_dmrsPos,ii) = rxDataList{ii}.toEstMicroSec;
    postEqSinrdB(ii) = rxDataList{ii}.postEqSinrdB(end);
    postEqNoiseVardB(ii) = rxDataList{ii}.postEqNoiseVardB(end);
end
hdf5_write_nv(h5File, 'reference_rsrpdB', single(transpose(rsrpdB_vec)),'fp32');
hdf5_write_nv(h5File, 'reference_rsrpdB_ehq', single(transpose(rsrpdB_ehq_vec)),'fp32');
hdf5_write_nv(h5File, 'reference_sinrdB', single(transpose(sinrdB_vec)),'fp32');
hdf5_write_nv(h5File, 'reference_sinrdB_ehq', single(transpose(sinrdB_ehq_vec)),'fp32');

%hdf5_write_nv(h5File, 'rsrpdB', single(transpose(rsrpdB_vec)),'fp16');
hdf5_write_nv(h5File, 'reference_postEqSinrdBVec', single(transpose(postEqSinrdB_vec)),'fp32');
hdf5_write_nv(h5File, 'reference_noiseVardB', single(transpose(noiseVardB_vec)),'fp32');
hdf5_write_nv(h5File, 'reference_noiseVardBPerUe', single(transpose(noiseVardBPerUe_vec)),'fp32');
hdf5_write_nv(h5File, 'reference_postEqNoiseVardBvec', single(transpose(postEqNoiseVardB_vec)),'fp32');
hdf5_write_nv(h5File, 'reference_cfoEstHzPerUe', single(transpose(cfoEstHzPerUe_vec)),'fp32');
hdf5_write_nv(h5File, 'reference_taEstMicroSecPerUe', single(transpose(taEstMicroSecPerUe_vec)),'fp32');

hdf5_write_nv(h5File, 'reference_postEqSinrdB', single(transpose(postEqSinrdB)),'fp32');
hdf5_write_nv(h5File, 'reference_postEqNoiseVardB', single(transpose(postEqNoiseVardB)),'fp32');

global SimCtrl
bypassComp = SimCtrl.genTV.bypassComp;
if ~bypassComp
    Xtf = Xtf0;
    [cSamples_uint8, X_tf_fp16] = oranCompress(Xtf, 1); % generate ORAN compressed samples
    hdf5_write_nv(h5File, 'X_tf_fp16', X_tf_fp16, 'fp16');
    for k=1:length(SimCtrl.oranComp.iqWidth)
        hdf5_write_nv(h5File, ['X_tf_cSamples_bfp',num2str(SimCtrl.oranComp.iqWidth(k))], uint8(cSamples_uint8{k}));
    end
end




% RKHS PUSCH objects
if SimCtrl.alg.ChEst_alg_selector == 2   % 'RKHS'
    push_rkhs_tables = puschTable.push_rkhs_tables;
    prbRksPrms = [];
    
    % per Prb RKHS prms:
    for i = 1 : push_rkhs_tables.rkhsPrms.nPrbSizes
        prbRksPrms(i).nZpDmrsSc           = uint16(push_rkhs_tables.prbPrms{i}.nZpDmrsSc);
        prbRksPrms(i).zpIdx               = uint8(push_rkhs_tables.prbPrms{i}.zpIdx);
        prbRksPrms(i).nCpInt              = uint8(push_rkhs_tables.prbPrms{i}.nCpInt);
        prbRksPrms(i).sumEigValues        = single(push_rkhs_tables.prbPrms{i}.sumEigValues);
    
        str1 = 'corr_half_nZpDmrsSc';
        str2 = num2str(i);
        str  = append(str1, str2);
        hdf5_write_nv(h5File, str, complex(single(push_rkhs_tables.prbPrms{i}.corr_half_nZpDmrsSc)),'fp16');
    
        single(fp16nv(real(eqOutputLLRs_zp), SimCtrl.fp16AlgoSel));
    
        str1 = 'eigVecCob';
        str2   = num2str(i);
        str    = append(str1, str2);
        buffer = push_rkhs_tables.eigVecCobTable{i};
        buffer = single(fp16nv(real(buffer), SimCtrl.fp16AlgoSel));
        hdf5_write_nv(h5File, str, buffer,'fp16');
    
        str1 = 'corr';
        str2 = num2str(i);
        str  = append(str1, str2);
        hdf5_write_nv(h5File, str, complex(single(push_rkhs_tables.corrTable{i})),'fp16');
    
        str1 = 'eigVal';
        str2 = num2str(i);
        str  = append(str1, str2);
        hdf5_write_nv(h5File, str, real(single(push_rkhs_tables.eigValTable{i})),'fp16');
    
        str1 = 'interpCob';
        str2 = num2str(i);
        str  = append(str1, str2);
        hdf5_write_nv(h5File, str, real(single(push_rkhs_tables.interpCobTable{i}.')),'fp16');
    end
    
    % per zp prms:
    zpRksPrms = [];
    
    for zpIdx = 0 : (push_rkhs_tables.rkhsPrms.nZpSizes - 1)
        zpRksPrms(zpIdx + 1).secondStageFourierSize = uint8(push_rkhs_tables.zpFftPrms{zpIdx + 1}.secondStageFourierSize);
        zpRksPrms(zpIdx + 1).nZpDmrsSc              = uint16(push_rkhs_tables.zpFftPrms{zpIdx + 1}.nZpDmrsSc);
    
        str1 = 'zpInterpVec';
        str2 = num2str(zpIdx);
        str  = append(str1, str2);
        buffer = push_rkhs_tables.zpInterpVecTable{zpIdx + 1}.';
        buffer = single(fp16nv(real(buffer), SimCtrl.fp16AlgoSel));
        hdf5_write_nv(h5File, str, buffer,'fp16');
    
        str1 = 'zpDmrsScEigenVec';
        str2 = num2str(zpIdx);
        str  = append(str1, str2);
        buffer = push_rkhs_tables.zpDmrsScEigenVecTable{zpIdx + 1};
        buffer = single(fp16nv(real(buffer), SimCtrl.fp16AlgoSel));
        hdf5_write_nv(h5File, str, buffer,'fp16');
    
        str1 = 'zpSecondStageTwiddleFactors';
        str2 = num2str(zpIdx);
        str  = append(str1, str2);
        buffer = conj(push_rkhs_tables.zpFftPrms{zpIdx + 1}.secondStageTwiddleFactors);
        % buffer = single(fp16nv(buffer, SimCtrl.fp16AlgoSel));
        hdf5_write_nv(h5File, str, complex(single(buffer)),'fp16');
    
        str1 = 'zpSecondStageFourierPerm';
        str2 = num2str(zpIdx);
        str  = append(str1, str2);
        buffer = push_rkhs_tables.zpFftPrms{zpIdx + 1}.secondStageFourierPerm;
        buffer = uint8(buffer);
        hdf5_write_nv(h5File, str, buffer);
    end
    
    rkhsPrms               = [];
    rkhsPrms.nEigs         = uint8(push_rkhs_tables.rkhsPrms.nEigs);
    rkhsPrms.nZpDmrsScEigs = uint8(push_rkhs_tables.rkhsPrms.nZpDmrsScEigs);
    rkhsPrms.nIterpEigs    = uint8(push_rkhs_tables.rkhsPrms.nIterpEigs);
    rkhsPrms.nPrbSizes     = uint16(push_rkhs_tables.rkhsPrms.nPrbSizes);
    rkhsPrms.nZpSizes      = uint8(push_rkhs_tables.rkhsPrms.nZpSizes);
    
    
    hdf5_write_nv(h5File, 'zpRksPrms', zpRksPrms);
    hdf5_write_nv(h5File, 'rkhsPrms', rkhsPrms);
    hdf5_write_nv(h5File, 'prbRksPrms', prbRksPrms);
end

% logging DCI_0_0
% for idxPdu=1:nPdu
%     if (PuschParamsList{idxPdu}.nPrb-1) < floor(carrier.N_BWP_size/2)
%         RIV = carrier.N_BWP_size*(PuschParamsList{idxPdu}.nPrb-1) + PuschParamsList{idxPdu}.startPrb;
%     else
%         RIV = carrier.N_BWP_size*(carrier.N_BWP_size - PuschParamsList{idxPdu}.nPrb + 1) + (carrier.N_BWP_size - 1 - (PuschParamsList{idxPdu}.startPrb-1));
%     end
%     RIV_length = ceil(log2(carrier.N_BWP_size*(carrier.N_BWP_size+1)/2));
%     DCI_bitFields = gen_DCIFormat0_0_payload(RIV_length);
%     DCI_bitFields.Identifier = 0; % always set to 0, indicating an UL DCI format, refer to Sec. 7.3.1.1.1 in TS 38.212
%     DCI_bitFields.FrequencyDomainResources = RIV;
%     DCI_bitFields.TimeDomainResources = 0; % hard coding for now
%     DCI_bitFields.FrequencyHoppingFlag = PuschParamsList{idxPdu}.groupOrSequenceHopping;
%     DCI_bitFields.ModulationCoding = PuschParamsList{idxPdu}.mcsIndex;
%     DCI_bitFields.NewDataIndicator = PuschParamsList{idxPdu}.newDataIndicator;
%     DCI_bitFields.RedundancyVersion = PuschParamsList{idxPdu}.rvIdx;
%     DCI_bitFields.HARQprocessNumber = PuschParamsList{idxPdu}.harqProcessID;
%     DCI_bitFields.TPCcommand = 0; % hard coding for now
%     DCI_bitFields.AlignedWidth = 40;
% %     DCI_bitFields.UL_SUL_indicator = 0;
%     DCI0_0(idxPdu).payload_bits = toBits(DCI_bitFields);
%     DCI0_0(idxPdu).K2 = 0; % hard coding for now
% 
%     hdf5_write_nv(h5File, sprintf('DCI0_0_pdu_%d',idxPdu), DCI0_0(idxPdu));
% end


H5F.close(h5File);

return


function saveTV_polar_debug(nPolUciSegs, polarBuffers, TVname)
    % create h5 file
    tvDirName    = 'GPU_test_input';
    tvName       = strcat(TVname,'_uciPolarDebug');
    [status,msg] = mkdir(tvDirName);
    h5File       = H5F.create([tvDirName filesep tvName '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

    % save polarUciSegPrms
    polarUciSegPrms = [];
    for segIdx = 0 : (nPolUciSegs - 1)
        polarUciSegPrms(segIdx + 1).nCbs            = uint8(polarBuffers{segIdx + 1}.polarUciSegPrms.nCbs);
        polarUciSegPrms(segIdx + 1).K_cw            = uint16(polarBuffers{segIdx + 1}.polarUciSegPrms.K_cw);
        polarUciSegPrms(segIdx + 1).E_cw            = uint32(polarBuffers{segIdx + 1}.polarUciSegPrms.E_cw);
        polarUciSegPrms(segIdx + 1).zeroInsertFlag  = uint8(polarBuffers{segIdx + 1}.polarUciSegPrms.zeroInsertFlag);
        polarUciSegPrms(segIdx + 1).n_cw            = uint8(polarBuffers{segIdx + 1}.polarUciSegPrms.n_cw);
        polarUciSegPrms(segIdx + 1).N_cw            = uint16(polarBuffers{segIdx + 1}.polarUciSegPrms.N_cw);
        polarUciSegPrms(segIdx + 1).E_seg           = uint32(polarBuffers{segIdx + 1}.polarUciSegPrms.E_seg);
        polarUciSegPrms(segIdx + 1).nCrcBits        = uint8(polarBuffers{segIdx + 1}.polarUciSegPrms.nCrcBits);
    end
    hdf5_write_nv_exp(h5File, 'polarUciSegPrms', polarUciSegPrms);

    % save cwTreeTypes
    for segIdx = 0 : (nPolUciSegs - 1)
        nameStr = strcat('cwTreeTypes',num2str(segIdx));
        hdf5_write_nv(h5File, nameStr, uint8(polarBuffers{segIdx + 1}.cwTreeTypes));
    end

    global SimCtrl
    % save uci segment LLRs
    for segIdx = 0 : (nPolUciSegs - 1)
        uciSegLLRs = single(fp16nv(polarBuffers{segIdx + 1}.uciSegLLRs, SimCtrl.fp16AlgoSel));
        nameStr    = strcat('uciSegLLRs',num2str(segIdx));
        hdf5_write_nv(h5File, nameStr, uciSegLLRs);
    end

    nCbs          = 0;
    crcErrorFlags = [];
    for segIdx = 0 : (nPolUciSegs - 1)
        for i = 1 : polarUciSegPrms(segIdx + 1).nCbs
            % save codeword LLRs
            cwLLRs  = single(fp16nv(polarBuffers{segIdx + 1}.cwLLRs(:,i), SimCtrl.fp16AlgoSel));
            nameStr = strcat('cwLLRs',num2str(nCbs));
            hdf5_write_nv(h5File, nameStr, cwLLRs);

             % save codeblock estimates
             cbEst_bits = polarBuffers{segIdx + 1}.cbEsts(:,i);
             nBits      = length(cbEst_bits);
             nWords     = ceil(nBits / 32);

             cbEst_words = zeros(nWords,1);
            for wordIdx = 0 : (nWords - 1)
                for j = 0 : 31
                    bitIdx = wordIdx*32 + j;

                    if(bitIdx >= nBits)
                        break;
                    else
                        cbEst_words(wordIdx+1) = cbEst_words(wordIdx+1) + cbEst_bits(bitIdx + 1)*2^j;
                    end
                end
            end
            cbEst_words = uint32(cbEst_words);
            nameStr     = strcat('cbEst',num2str(nCbs));
            hdf5_write_nv(h5File, nameStr, cbEst_words);

            % save crc flag
            crcErrorFlags = [crcErrorFlags; polarBuffers{segIdx + 1}.cbCrcErrorFlags(i)];

            % update number of cbs
            nCbs = nCbs + 1;
        end
    end
    hdf5_write_nv(h5File, 'crcErrorFlags', uint8(crcErrorFlags));


    % sizes
    sizes = [];
    sizes.nPolCws     = uint16(nCbs);
    sizes.nPolUciSegs = uint16(nPolUciSegs);
    hdf5_write_nv_exp(h5File, 'sizes', sizes);

return;

