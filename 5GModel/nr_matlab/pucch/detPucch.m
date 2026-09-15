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

function PucchDataOut = detPucch(pucchPduList, table, carrier, Xtf)
global SimCtrl;

%function applies full pucch receiver pipeline for all users.
% NOTE: currently fully supports pucch format 0 and 1

pucchTable = table;

%% derive API parameters
[cuphyCellStatPrm, cuphyPucchDynPrms] = deriveApiParam(pucchPduList, carrier, Xtf, pucchTable);

%% PUCCH detection main function
[cuphyPucchDynPrms] = detPucch_cuphy(cuphyCellStatPrm, cuphyPucchDynPrms);

%% obtain UCIs
PucchDataOut = cuphyPucchDynPrms.cuphyPucchDataOut;


%% save TV
idxSlot = carrier.idxSlotInFrame;
if SimCtrl.genTV.enable && SimCtrl.genTV.cuPHY && ismember(idxSlot, SimCtrl.genTV.slotIdx)
    TVname = [SimCtrl.genTV.TVname, '_PUCCH_F', num2str(pucchPduList{1}.FormatType), '_gNB_CUPHY_s', num2str(carrier.idxSlotInFrame), 'p', num2str(pucchPduList{end}.pucchPduIdx)];

    % Collect polar intermediate buffers in cuPHY's polar-seg order: F2 UCIs
    % first, then F3 UCIs; within each UCI, seg1 then (when present) seg2
    % (CSI2). Written inline into the main PUCCH TV by saveTV_pucch_cuphy
    % and consumed by PucchRx::loadPolarRefForSkip on --skip-polar runs.
    polarBuffers = {};
    if isfield(PucchDataOut, 'pUciF2')
        for fIdx = 1:length(PucchDataOut.pUciF2)
            uci = PucchDataOut.pUciF2{fIdx};
            if isstruct(uci) && isfield(uci, 'polarInterBuffers')
                for s = 1:length(uci.polarInterBuffers)
                    polarBuffers{end+1} = uci.polarInterBuffers{s};
                end
            end
        end
    end
    if isfield(PucchDataOut, 'pUciF3')
        for fIdx = 1:length(PucchDataOut.pUciF3)
            uci = PucchDataOut.pUciF3{fIdx};
            if isstruct(uci) && isfield(uci, 'polarInterBuffers')
                for s = 1:length(uci.polarInterBuffers)
                    polarBuffers{end+1} = uci.polarInterBuffers{s};
                end
            end
        end
    end

    % Collect raw per-UCI front-end LLR refs in cuPHY allocation order:
    % all F2 (0..nF2-1), then all F3 (0..nF3-1). For F2 this is descrmLLR
    % (length E_seg1, no seg2). For F3 this is post-demux descrmLLRSeq1
    % (length E_seg1). PucchRx::compareFrontEndRefForBackendSkip applies
    % the cuPHY-only front-end noise-bias scale when comparing against GPU
    % m_F{2,3}seg1LLRaddrsVec.
    frontEndBuffers = struct('f2', {{}}, 'f3', {{}});
    if isfield(PucchDataOut, 'pUciF2')
        for fIdx = 1:length(PucchDataOut.pUciF2)
            uci = PucchDataOut.pUciF2{fIdx};
            if isstruct(uci) && isfield(uci, 'descrmLLR')
                frontEndBuffers.f2{end+1} = reshape(uci.descrmLLR, 1, []);
            else
                frontEndBuffers.f2{end+1} = [];
            end
        end
    end
    if isfield(PucchDataOut, 'pUciF3')
        for fIdx = 1:length(PucchDataOut.pUciF3)
            uci = PucchDataOut.pUciF3{fIdx};
            if isstruct(uci) && isfield(uci, 'descrmLLRSeq1')
                frontEndBuffers.f3{end+1} = reshape(uci.descrmLLRSeq1, 1, []);
            else
                frontEndBuffers.f3{end+1} = [];
            end
        end
    end

    saveTV_pucch_cuphy(SimCtrl.genTV.tvDirName, TVname, cuphyCellStatPrm, cuphyPucchDynPrms, Xtf, PucchDataOut, carrier, polarBuffers, frontEndBuffers);
%     saveTV_pusch_cuphy(SimCtrl.genTV.tvDirName, TVname, UegList, pusch_payload_list, PuschParamsList, ...
%         Xtf, carrier, rxDataList, puschTable);

    SimCtrl.genTV.idx = SimCtrl.genTV.idx + 1;
end

end

%% PUCCH detection main function
function [cuphyPucchDynPrms] = detPucch_cuphy(cuphyCellStatPrm, cuphyPucchDynPrms)   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CPU code
Xtf = cuphyPucchDynPrms.cuphyPucchDataIn;

%% PUCCH grouping for format 0
[PucchF0ParamGrps, nF0Grps] = PucchGrouping(cuphyPucchDynPrms.cuphyPucchUciPrmF0);
    
%% PUCCH grouping for format 1
[PucchF1ParamGrps, nF1Grps] = PucchGrouping(cuphyPucchDynPrms.cuphyPucchUciPrmF1);

%% PUCCH grouping for format 2
[PucchF2ParamGrps, nF2Grps] = PucchGrouping(cuphyPucchDynPrms.cuphyPucchUciPrmF2);

%% PUCCH grouping for format 3
[PucchF3ParamGrps, nF3Grps] = PucchGrouping(cuphyPucchDynPrms.cuphyPucchUciPrmF3);

%% construct static and dynamic descriptors for format 0
[pucchF0tables, pucchF0dynamicDesc] = constructDescriptors(0, cuphyCellStatPrm, cuphyPucchDynPrms.cuphyPucchCellDynPrm, PucchF0ParamGrps, nF0Grps, Xtf);    

%% construct static and dynamic descriptors for format 1
[pucchF1tables, pucchF1dynamicDesc] = constructDescriptors(1, cuphyCellStatPrm, cuphyPucchDynPrms.cuphyPucchCellDynPrm, PucchF1ParamGrps, nF1Grps, Xtf);    

%% construct static and dynamic descriptors for format 2 
[pucchF2tables, pucchF2dynamicDesc] = constructDescriptors(2, cuphyCellStatPrm, cuphyPucchDynPrms.cuphyPucchCellDynPrm, PucchF2ParamGrps, nF2Grps, Xtf);    

%% construct static and dynamic descriptors for format 3 
[pucchF3tables, pucchF3dynamicDesc] = constructDescriptors(3, cuphyCellStatPrm, cuphyPucchDynPrms.cuphyPucchCellDynPrm, PucchF3ParamGrps, nF3Grps, Xtf);    


%% allocate memory on GPU
%% format 0
pF0UcisOut = cell(1, length(cuphyPucchDynPrms.cuphyPucchDataOut.pUciF0));

%% format 1
pF1UcisOut = cell(1, length(cuphyPucchDynPrms.cuphyPucchDataOut.pUciF1));

%% format 2
pF2UcisOut = cell(1, length(cuphyPucchDynPrms.cuphyPucchDataOut.pUciF2));

%% format 3
pF3UcisOut = cell(1, length(cuphyPucchDynPrms.cuphyPucchDataOut.pUciF3));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GPU CUDA kernel
%% format 0
for grpIdx = 1:nF0Grps
    [pF0UcisOut] = pucchF0_rx_kernel(cuphyCellStatPrm.nRxAnt, pucchF0tables, pucchF0dynamicDesc, pF0UcisOut, grpIdx);
end

%% format 1
for grpIdx = 1:nF1Grps
    [pF1UcisOut] = pucchF1_rx_kernel(cuphyCellStatPrm.nRxAnt, pucchF1tables, pucchF1dynamicDesc, pF1UcisOut, grpIdx);
end

%% format 2
%% PF2 receiver kernel processes all UCIs by a single function call
[pF2UcisOut] = pucchF2_rx_kernel(cuphyCellStatPrm.nRxAnt, pucchF2tables, pucchF2dynamicDesc, pF2UcisOut);


%%% format 3
%% PF3 receiver kernel processes all UCIs by a single function call
[pF3UcisOut] = pucchF3_rx_kernel(cuphyCellStatPrm.nRxAnt, pucchF3tables, pucchF3dynamicDesc, pF3UcisOut);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CPU code 
%% transfer UCIs from GPU to CPU
%% format 0
cuphyPucchDynPrms.cuphyPucchDataOut.pUciF0 = pF0UcisOut;

%% format 1
cuphyPucchDynPrms.cuphyPucchDataOut.pUciF1 = pF1UcisOut;

%% format 2
cuphyPucchDynPrms.cuphyPucchDataOut.pUciF2 = pF2UcisOut;

%% format 3
cuphyPucchDynPrms.cuphyPucchDataOut.pUciF3 = pF3UcisOut;

end


%% helper functions
function [cuphyCellStatPrm, cuphyPucchDynPrms] = deriveApiParam(pucchPduList, carrier, Xtf, pucchTable)

global SimCtrl

%% cell static paramaters

cuphyCellStatPrm.phyCellId   = carrier.N_ID_CELL;   
if SimCtrl.enable_static_dynamic_beamforming % 64TR
    cuphyCellStatPrm.nRxAnt = pucchPduList{1}.digBFInterfaces;
else
    cuphyCellStatPrm.nRxAnt = carrier.numRxPort;
end
cuphyCellStatPrm.nTxAnt      = carrier.numTxPort;
cuphyCellStatPrm.nPrbUlBwp   = carrier.N_grid_size_mu;      
cuphyCellStatPrm.nPrbDlBwp   = carrier.N_grid_size_mu;      
cuphyCellStatPrm.mu          = carrier.mu;             
cuphyCellStatPrm.beta_dmrs   = 1;     
cuphyCellStatPrm.beta_qam    = 1;


%% cell dynamic parameter
% Note: currently only supports single cell processing

cuphyPucchDynPrms.cuphyPucchDataIn = Xtf;
cuphyPucchDynPrms.cuphyPucchCellDynPrm.pucchHoppingId = pucchPduList{1}.hoppingId;
cuphyPucchDynPrms.cuphyPucchCellDynPrm.slotNum = mod(carrier.idxSlotInFrame, carrier.N_slot_frame_mu);
cuphyPucchDynPrms.cuphyPucchCellDynPrm.pucchTable = pucchTable;

%% per-UCI parameters

nF0pduIdx = [];
nF1pduIdx = [];
nF2pduIdx = [];
nF3pduIdx = [];

nPDU = length(pucchPduList);

for PduIdx = 1:nPDU
    FormatType = pucchPduList{PduIdx}.FormatType;
    
    if FormatType == 0
        nF0pduIdx = [nF0pduIdx PduIdx];
    elseif FormatType == 1
        nF1pduIdx = [nF1pduIdx PduIdx];
    elseif FormatType == 2
        nF2pduIdx = [nF2pduIdx PduIdx];
    elseif FormatType == 3
        nF3pduIdx = [nF3pduIdx PduIdx]; 
    else
        error('Error. \nFormat is not supported.')
    end  
end

%% PUCCH format 0 per-UCI parameters

cuphyPucchDynPrms.cuphyPucchUciPrmF0 = {};

nF0uci = length(nF0pduIdx);

for j = 1:nF0uci
    PduIdx = nF0pduIdx(j);
    
    PDU = pucchPduList{PduIdx};
    
    cuphyPucchUciPrmF0.BWPStart             = PDU.BWPStart;
    cuphyPucchUciPrmF0.uciOutputIdx         = j;  
    cuphyPucchUciPrmF0.formatType           = PDU.FormatType; 
    cuphyPucchUciPrmF0.rnti                 = PDU.RNTI; 
    cuphyPucchUciPrmF0.multiSlotTxIndicator = PDU.multiSlotTxIndicator; 
    cuphyPucchUciPrmF0.pi2Bpsk              = PDU.pi2Bpsk; 
    cuphyPucchUciPrmF0.startPrb             = PDU.prbStart;
    cuphyPucchUciPrmF0.prbSize              = PDU.prbSize;
    cuphyPucchUciPrmF0.startSym             = PDU.StartSymbolIndex;
    cuphyPucchUciPrmF0.nSym                 = PDU.NrOfSymbols;  
    cuphyPucchUciPrmF0.freqHopFlag          = PDU.freqHopFlag;    
    cuphyPucchUciPrmF0.secondHopPrb         = PDU.secondHopPRB;   
    cuphyPucchUciPrmF0.groupHopFlag         = PDU.groupHopFlag;    
    cuphyPucchUciPrmF0.sequenceHopFlag      = PDU.sequenceHopFlag;    
    cuphyPucchUciPrmF0.initialCyclicShift   = PDU.InitialCyclicShift;    
    cuphyPucchUciPrmF0.timeDomainOccIdx     = PDU.TimeDomainOccIdx;    
    cuphyPucchUciPrmF0.srFlag               = PDU.SRFlag;    
    cuphyPucchUciPrmF0.bitLenHarq           = PDU.BitLenHarq;   
    cuphyPucchUciPrmF0.DTXthreshold         = PDU.DTXthreshold;
    
    cuphyPucchDynPrms.cuphyPucchUciPrmF0{j} = cuphyPucchUciPrmF0;
end

cuphyPucchDynPrms.cuphyPucchDataOut.pUciF0 = cell(1,nF0uci);

%% PUCCH format 1 per-UCI parameters

cuphyPucchDynPrms.cuphyPucchUciPrmF1 = {};

nF1uci = length(nF1pduIdx);

for j = 1:nF1uci
    PduIdx = nF1pduIdx(j);
    
    PDU = pucchPduList{PduIdx};
    
    cuphyPucchUciPrmF1.BWPStart             = PDU.BWPStart;
    cuphyPucchUciPrmF1.uciOutputIdx         = j;  
    cuphyPucchUciPrmF1.formatType           = PDU.FormatType; 
    cuphyPucchUciPrmF1.rnti                 = PDU.RNTI; 
    cuphyPucchUciPrmF1.multiSlotTxIndicator = PDU.multiSlotTxIndicator; 
    cuphyPucchUciPrmF1.pi2Bpsk              = PDU.pi2Bpsk; 
    cuphyPucchUciPrmF1.startPrb             = PDU.prbStart;
    cuphyPucchUciPrmF1.prbSize              = PDU.prbSize;
    cuphyPucchUciPrmF1.startSym             = PDU.StartSymbolIndex;
    cuphyPucchUciPrmF1.nSym                 = PDU.NrOfSymbols;  
    cuphyPucchUciPrmF1.freqHopFlag          = PDU.freqHopFlag;    
    cuphyPucchUciPrmF1.secondHopPrb         = PDU.secondHopPRB;   
    cuphyPucchUciPrmF1.groupHopFlag         = PDU.groupHopFlag;    
    cuphyPucchUciPrmF1.sequenceHopFlag      = PDU.sequenceHopFlag;    
    cuphyPucchUciPrmF1.initialCyclicShift   = PDU.InitialCyclicShift;    
    cuphyPucchUciPrmF1.timeDomainOccIdx     = PDU.TimeDomainOccIdx;    
    cuphyPucchUciPrmF1.srFlag               = PDU.SRFlag;    
    cuphyPucchUciPrmF1.bitLenHarq           = PDU.BitLenHarq;   
    cuphyPucchUciPrmF1.DTXthreshold         = PDU.DTXthreshold;   
    
    cuphyPucchDynPrms.cuphyPucchUciPrmF1{j} = cuphyPucchUciPrmF1;
end

cuphyPucchDynPrms.cuphyPucchDataOut.pUciF1 = cell(1,nF1uci);

%% PUCCH format 2 per-UCI parameters
cuphyPucchDynPrms.cuphyPucchUciPrmF2 = {};

nF2uci = length(nF2pduIdx);

for j = 1:nF2uci
    PduIdx = nF2pduIdx(j);
    PDU = pucchPduList{PduIdx};
    
    cuphyPucchUciPrmF2.BWPStart             = PDU.BWPStart;
    cuphyPucchUciPrmF2.uciOutputIdx         = j;  
    cuphyPucchUciPrmF2.formatType           = PDU.FormatType; 
    cuphyPucchUciPrmF2.rnti                 = PDU.RNTI; 
    cuphyPucchUciPrmF2.multiSlotTxIndicator = PDU.multiSlotTxIndicator; 
    cuphyPucchUciPrmF2.startPrb             = PDU.prbStart;
    cuphyPucchUciPrmF2.prbSize              = PDU.prbSize;
    cuphyPucchUciPrmF2.startSym             = PDU.StartSymbolIndex;
    cuphyPucchUciPrmF2.nSym                 = PDU.NrOfSymbols;  
    cuphyPucchUciPrmF2.freqHopFlag          = PDU.freqHopFlag;    
    cuphyPucchUciPrmF2.secondHopPrb         = PDU.secondHopPRB;   
    cuphyPucchUciPrmF2.bitLenHarq           = PDU.BitLenHarq;   
    cuphyPucchUciPrmF2.DTXthreshold         = PDU.DTXthreshold; 
    cuphyPucchUciPrmF2.dataScramblingId     = PDU.dataScramblingId;
    cuphyPucchUciPrmF2.DmrsScramblingId     = PDU.DmrsScramblingId;
    cuphyPucchUciPrmF2.maxCodeRate          = PDU.maxCodeRate;
    cuphyPucchUciPrmF2.BitLenSr             = PDU.BitLenSr;
    cuphyPucchUciPrmF2.BitLenCsiPart1       = PDU.BitLenCsiPart1;
    cuphyPucchDynPrms.cuphyPucchUciPrmF2{j} = cuphyPucchUciPrmF2;
end

cuphyPucchDynPrms.cuphyPucchDataOut.pUciF2 = cell(1,nF2uci);

%% PUCCH format 3 per-UCI parameters

cuphyPucchDynPrms.cuphyPucchUciPrmF3 = {};

nF3uci = length(nF3pduIdx);

for j = 1:nF3uci
    PduIdx = nF3pduIdx(j);
    
    PDU = pucchPduList{PduIdx};
    
    cuphyPucchUciPrmF3.BWPStart             = PDU.BWPStart;
    cuphyPucchUciPrmF3.uciOutputIdx         = j;  
    cuphyPucchUciPrmF3.formatType           = PDU.FormatType; 
    cuphyPucchUciPrmF3.rnti                 = PDU.RNTI; 
    cuphyPucchUciPrmF3.multiSlotTxIndicator = PDU.multiSlotTxIndicator; 
    cuphyPucchUciPrmF3.pi2Bpsk              = PDU.pi2Bpsk; 
    cuphyPucchUciPrmF3.startPrb             = PDU.prbStart;
    cuphyPucchUciPrmF3.prbSize              = PDU.prbSize;
    cuphyPucchUciPrmF3.startSym             = PDU.StartSymbolIndex;
    cuphyPucchUciPrmF3.nSym                 = PDU.NrOfSymbols;  
    cuphyPucchUciPrmF3.freqHopFlag          = PDU.freqHopFlag;    
    cuphyPucchUciPrmF3.secondHopPrb         = PDU.secondHopPRB;   
    cuphyPucchUciPrmF3.groupHopFlag         = PDU.groupHopFlag;    
    cuphyPucchUciPrmF3.sequenceHopFlag      = PDU.sequenceHopFlag;    
    cuphyPucchUciPrmF3.initialCyclicShift   = PDU.InitialCyclicShift;    
    cuphyPucchUciPrmF3.timeDomainOccIdx     = PDU.TimeDomainOccIdx;    
    cuphyPucchUciPrmF3.bitLenHarq           = PDU.BitLenHarq;   
    cuphyPucchUciPrmF3.DTXthreshold         = PDU.DTXthreshold; 
    cuphyPucchUciPrmF3.dataScramblingId     = PDU.dataScramblingId;
    cuphyPucchUciPrmF3.AddDmrsFlag          = PDU.AddDmrsFlag; 
    cuphyPucchUciPrmF3.maxCodeRate          = PDU.maxCodeRate;
    cuphyPucchUciPrmF3.BitLenSr             = PDU.BitLenSr;
    cuphyPucchUciPrmF3.BitLenCsiPart1       = PDU.BitLenCsiPart1;
    cuphyPucchUciPrmF3.rankBitOffset        = PDU.rankBitOffset;
    cuphyPucchUciPrmF3.rankBitSize          = PDU.rankBitSize;
    cuphyPucchUciPrmF3.numPart2s            = PDU.UciP1ToP2Crpd.numPart2s;
    cuphyPucchDynPrms.cuphyPucchUciPrmF3{j} = cuphyPucchUciPrmF3;
end

cuphyPucchDynPrms.cuphyPucchDataOut.pUciF3 = cell(1,nF3uci);

end


function [PucchParamGrps, nGrps] = PucchGrouping(cuphyPucchUciPrm)
    
PucchParamGrps = {};
ParamGrpIdx = 1;

nUciPrm = length(cuphyPucchUciPrm);

if nUciPrm>0
    
    PucchParamGrps{1} = cuphyPucchUciPrm{1};
    PucchParamGrps{1}.nUciInGrp = 1;
    ParamGrpIdx=2;
    
    for j = 2:nUciPrm
        UciPrm = cuphyPucchUciPrm{j};
        startPrb = UciPrm.startPrb;
        startSym = UciPrm.startSym;
        BWPStart = UciPrm.BWPStart;
        
        newGrp = true;
        
        for k = 1:(ParamGrpIdx-1)
            
            if ((startPrb + BWPStart) == (PucchParamGrps{k}.startPrb + PucchParamGrps{k}.BWPStart)) && ... 
                    (startSym == PucchParamGrps{k}.startSym)
                
                PucchParamGrps{k}.nUciInGrp = PucchParamGrps{k}.nUciInGrp + 1;
                
                initialCyclicShift = UciPrm.initialCyclicShift;
                timeDomainOccIdx = UciPrm.timeDomainOccIdx;
                bitLenHarq = UciPrm.bitLenHarq;
                uciOutputIdx = UciPrm.uciOutputIdx;
                srFlag = UciPrm.srFlag;
                DTXthreshold = UciPrm.DTXthreshold;
                
                PucchParamGrps{k}.initialCyclicShift = ...
                    [PucchParamGrps{k}.initialCyclicShift initialCyclicShift];
                
                PucchParamGrps{k}.timeDomainOccIdx = ...
                    [PucchParamGrps{k}.timeDomainOccIdx timeDomainOccIdx];
                
                PucchParamGrps{k}.bitLenHarq = ...
                    [PucchParamGrps{k}.bitLenHarq bitLenHarq];
                
                PucchParamGrps{k}.uciOutputIdx = ...
                    [PucchParamGrps{k}.uciOutputIdx uciOutputIdx];
                
                PucchParamGrps{k}.srFlag = ...
                    [PucchParamGrps{k}.srFlag srFlag];
                
                PucchParamGrps{k}.DTXthreshold = ...
                    [PucchParamGrps{k}.DTXthreshold DTXthreshold];
                
                newGrp = false;
                break;
            end
        end
        
        if newGrp
            PucchParamGrps{ParamGrpIdx} = UciPrm;
            PucchParamGrps{ParamGrpIdx}.nUciInGrp = 1;
            ParamGrpIdx = ParamGrpIdx + 1;
        end
    end
end

nGrps = ParamGrpIdx - 1;
end

function [pucchtables, dynamicDesc] = constructDescriptors(formatType, cuphyCellStatPrm, cuphyPucchCellDynPrm, PucchParamGrps, nGrps, Xtf)
%% cell common parameters
hop_id          = cuphyPucchCellDynPrm.pucchHoppingId; % hopping id
n_ID            = hop_id;
c_init          = n_ID;
slotNum         = cuphyPucchCellDynPrm.slotNum;
pucchTable      = cuphyPucchCellDynPrm.pucchTable;
scs             = 2^cuphyCellStatPrm.mu * 15*10^3;

dynamicDesc.scs = scs;

%% construct lookup table

if formatType == 0
    pucchtables.rBase        = pucchTable.r;
    pucchtables.csPhaseRamp  = derive_cs_freq;
elseif formatType == 1
    pucchtables.rBase        = pucchTable.r;
    pucchtables.csPhaseRamp  = derive_cs_freq;
    mu                       = cuphyCellStatPrm.mu;
    pucchtables.Wf           = pucch_freq_filter(mu);
    pucchtables.s            = derive_pucch_shift(mu);
    pucchtables.tOCC_cell    = pucchTable.tOCC_cell;
   [pucchtables.W_noiseIso, pucchtables.noiseDim] = pucch_noiseIso_filter(mu);

    
    %% time domain filter matrices
    %% refer to cuPHY/src/cuphy/pucch_F1_receiver/pucch_F1_receiver.cu
    Wt{1} = 1/2*ones(2,2); % d_Wt1_2_2
    Wt{2} = 1/2*ones(2,2); % d_Wt2_2_2
    Wt{3} = 1/2*ones(2,3); % d_Wt_2_3
    Wt{4} = 1/3*ones(3,2); % d_Wt_3_2
    Wt{5} = 1/3*ones(3,3); % d_Wt1_3_3
    Wt{6} = 1/3*ones(3,3); % d_Wt2_3_3
    Wt{7} = 1/3*ones(3,4); % d_Wt_3_4
    Wt{8} = 1/4*ones(4,3); % d_Wt_4_3
    Wt{9} = 1/4*ones(4,4); % d_Wt_4_4
    Wt{10} = 1/5*ones(5,4); % d_Wt_5_4
    Wt{11} = 1/5*ones(5,5); % d_Wt_5_5
    Wt{12} = 1/6*ones(6,5); % d_Wt_6_5
    Wt{13} = 1/6*ones(6,6); % d_Wt_6_6
    Wt{14} = 1/7*ones(7,6); % d_Wt_7_6
    Wt{15} = 1/7*ones(7,7); % d_Wt_7_7
    
    pucchtables.Wt = Wt;
    
elseif formatType == 2
%     tmp = [1 1 1 0 0 0 0 0 0 0 0 0 ]';
%     tmpW1 = [];
%     for ii=0:3
% %        tmpW1 = [tmpW1 circshift(tmp,3*ii)];
%     end
%     tmpW2 = blkdiag(tmpW1,tmpW1);
%     tmpW3 = blkdiag(tmpW1,tmpW1,tmpW1);
%     tmpW4 = blkdiag(tmpW1,tmpW1,tmpW1,tmpW1);
    
%     pucchtables.W1 = tmpW1;%eye(12);
%     pucchtables.W2 = tmpW2;%eye(24);
%     pucchtables.W3 = tmpW3;%eye(36);
%     pucchtables.W4 = tmpW4;%eye(48);
    mu = cuphyCellStatPrm.mu;
    pucchtables.W1 = PF2_FreqChEst_filter(1,mu);
    pucchtables.W2 = PF2_FreqChEst_filter(2,mu);
    pucchtables.W3 = PF2_FreqChEst_filter(3,mu);
    pucchtables.W4 = PF2_FreqChEst_filter(4,mu);
       
elseif formatType == 3
    mu = cuphyCellStatPrm.mu;
    pucchtables.W1 = PF3_FreqChEst_filter(1,mu);
    pucchtables.W2 = PF3_FreqChEst_filter(2,mu);
    pucchtables.W3 = PF3_FreqChEst_filter(3,mu);
    pucchtables.W4 = PF3_FreqChEst_filter(4,mu);
end


%% construct dynamic descriptor
dynamicDesc.Xtf         = Xtf;
dynamicDesc.slotNumber  = slotNum;

for grpIdx = 1:nGrps % iterate over PUCCH UCI groups
    freqHopFlag    = PucchParamGrps{grpIdx}.freqHopFlag;
    startSym       = PucchParamGrps{grpIdx}.startSym;
    nSym           = PucchParamGrps{grpIdx}.nSym;
    prbSize        = PucchParamGrps{grpIdx}.prbSize;
    
    if formatType ~= 2
        pi2Bpsk        = PucchParamGrps{grpIdx}.pi2Bpsk;
        groupHopFlag   = PucchParamGrps{grpIdx}.groupHopFlag;
        sequenceHopFlag = PucchParamGrps{grpIdx}.sequenceHopFlag;
        %% pre-compute for csArray
        c= build_Gold_sequence(c_init,14*8*slotNum + 8*(startSym + nSym - 1) + 8);
        csCommon = zeros(nSym,1);
        for i = 1 : nSym
            for m = 0 : 7
                csCommon(i) = csCommon(i) + 2^m * c(14*8*slotNum + 8*(startSym + i  - 1) + m + 1);
            end
        end
        
        dynamicDesc.csCommon{grpIdx} = csCommon;
        
        %% compute group and sequence indecies
        cSequenceEnable = build_Gold_sequence(floor(n_ID / 30),16*slotNum + 16);
        seqLen = 2*slotNum+2;
        if seqLen<4
            seqLen = 4;
        end
        cSequenceDisable = build_Gold_sequence(32*floor(n_ID / 30)+mod(n_ID, 30), seqLen);
        v=0;
        u=0;
        if ~groupHopFlag && ~sequenceHopFlag
            f_gh = 0;
            f_ss = mod(n_ID, 30);
            u = mod(f_gh+f_ss, 30);
            v = 0;
        elseif groupHopFlag && ~sequenceHopFlag
            f_ss = mod(n_ID, 30);
            temp = 0;
            for m=0:7
                temp = temp + 2^m*cSequenceEnable(16*slotNum + m + 1);
            end
            f_gh = mod(temp, 30);
            u = mod(f_gh+f_ss, 30);
            v = 0;
            if freqHopFlag
                temp1 = 0;
                for m=0:7
                    temp1 = temp1 + 2^m*cSequenceEnable(16*slotNum + 8 + m + 1);
                end
                
                f_gh = mod(temp1, 30);
                u_temp = mod(f_gh+f_ss, 30);
                u = [u u_temp];
            end
        elseif ~groupHopFlag && sequenceHopFlag
            f_gh = 0;
            f_ss = mod(n_ID, 30);
            u = mod(f_gh+f_ss, 30);
            v = cSequenceDisable(2*slotNum+1);
            
            v = [v cSequenceDisable(2*slotNum+2)];
        else
            error('Error. \nGroup hopping and sequence hopping cannot be both enabled.')
        end
        
        dynamicDesc.groupHopFlag{grpIdx}        = groupHopFlag;
        dynamicDesc.sequenceHopFlag{grpIdx}     = sequenceHopFlag;
        dynamicDesc.u{grpIdx}                   = u;
        dynamicDesc.cs0{grpIdx}                 = PucchParamGrps{grpIdx}.initialCyclicShift;
        dynamicDesc.pi2Bpsk{grpIdx}             = pi2Bpsk;
    end
   
    dynamicDesc.BWPStart{grpIdx}            = PucchParamGrps{grpIdx}.BWPStart;
    dynamicDesc.nUciInGrp{grpIdx}           = PucchParamGrps{grpIdx}.nUciInGrp;
    dynamicDesc.startSym{grpIdx}            = startSym;
    dynamicDesc.startPrb{grpIdx}            = PucchParamGrps{grpIdx}.startPrb;
    dynamicDesc.nSym{grpIdx}                = nSym;
    dynamicDesc.secondHopPrb{grpIdx}        = PucchParamGrps{grpIdx}.secondHopPrb;
    dynamicDesc.freqHopFlag{grpIdx}         = freqHopFlag;
    dynamicDesc.bitLenHarq{grpIdx}          = PucchParamGrps{grpIdx}.bitLenHarq;
    dynamicDesc.uciOutputIdx{grpIdx}        = PucchParamGrps{grpIdx}.uciOutputIdx;
    dynamicDesc.DTXthreshold{grpIdx}        = PucchParamGrps{grpIdx}.DTXthreshold;
    dynamicDesc.prbSize{grpIdx}             = prbSize;
    
    if formatType == 1
        nSym_data = floor(nSym/2); % number of data symbols
        nSym_dmrs = nSym - nSym_data; % number of dmrs symbols
        
        dynamicDesc.nSym_data{grpIdx} = nSym_data;
        dynamicDesc.nSym_dmrs{grpIdx} = nSym_dmrs;
        if freqHopFlag
            nSymDataFirstHop = floor(nSym_data/2);
            nSymFirstHop = floor(nSym/2);
            nSymDMRSFirstHop = nSymFirstHop - nSymDataFirstHop;
            nSymDataSecondHop = nSym_data - nSymDataFirstHop;
            nSymDMRSSecondHop = nSym_dmrs - nSymDMRSFirstHop;
            
            dynamicDesc.nSymDataFirstHop{grpIdx}    = nSymDataFirstHop;
            dynamicDesc.nSymFirstHop{grpIdx}        = nSymFirstHop;
            dynamicDesc.nSymDMRSFirstHop{grpIdx}    = nSymDMRSFirstHop;
            dynamicDesc.nSymDataSecondHop{grpIdx}   = nSymDataSecondHop;
            dynamicDesc.nSymDMRSSecondHop{grpIdx}   = nSymDMRSSecondHop;
        end
        dynamicDesc.srFlag{grpIdx}              = PucchParamGrps{grpIdx}.srFlag;
        dynamicDesc.timeDomainOccIdx{grpIdx}  = PucchParamGrps{grpIdx}.timeDomainOccIdx;
    elseif formatType == 2
        BitLenCsiPart1 = PucchParamGrps{grpIdx}.BitLenCsiPart1;
        BitLenHarq = PucchParamGrps{grpIdx}.bitLenHarq;
        BitLenSr   = PucchParamGrps{grpIdx}.BitLenSr;
        
        dynamicDesc.BitLenCsiPart1{grpIdx}  = BitLenCsiPart1;
        dynamicDesc.BitLenHarq{grpIdx}      = BitLenHarq;

        E_seg1 = nSym*prbSize*16;
        dynamicDesc.A_seg1{grpIdx}          = BitLenSr + BitLenHarq + BitLenCsiPart1;
        dynamicDesc.E_seg1{grpIdx}          = E_seg1;
        dynamicDesc.noiseVar{grpIdx}        = 10^(-20/10);
     
        DmrsScramblingId = PucchParamGrps{grpIdx}.DmrsScramblingId;
        cInitScrm = PucchParamGrps{grpIdx}.rnti*2^15 + PucchParamGrps{grpIdx}.dataScramblingId;
        dynamicDesc.randomSeqScrm{grpIdx}   = build_Gold_sequence(cInitScrm, E_seg1);
        
        mStartFirstHop = 4*(PucchParamGrps{grpIdx}.startPrb + PucchParamGrps{grpIdx}.BWPStart);
        if freqHopFlag
            mStartSecondHop = 4*(PucchParamGrps{grpIdx}.secondHopPrb + PucchParamGrps{grpIdx}.BWPStart);
        else
            mStartSecondHop = mStartFirstHop;
        end
       
        dmrsRandomSeqLen = prbSize*8 + 2*mStartFirstHop;
        
        c_init = mod(2^17 * (14*slotNum + startSym + 1) * (2*DmrsScramblingId + 1) + 2*DmrsScramblingId, 2^31);
        randomSeqDmrs1 = build_Gold_sequence(c_init, dmrsRandomSeqLen);
        
        dmrsRandomSeqLen = prbSize*8 + 2*mStartSecondHop;
        randomSeqDmrs2 = zeros(dmrsRandomSeqLen,1);
        if nSym > 1
            c_init = mod(2^17 * (14*slotNum + startSym + 2) * (2*DmrsScramblingId + 1) + 2*DmrsScramblingId, 2^31);
            randomSeqDmrs2 = build_Gold_sequence(c_init, dmrsRandomSeqLen);
        end
        
        dynamicDesc.mStartFirstHop{grpIdx}   = mStartFirstHop;
        dynamicDesc.mStartSecondHop{grpIdx}  = mStartSecondHop;
        dynamicDesc.randomSeqDmrs1{grpIdx}   = randomSeqDmrs1;
        dynamicDesc.randomSeqDmrs2{grpIdx}   = randomSeqDmrs2;
        dynamicDesc.BitLenSr{grpIdx}         = BitLenSr;
    elseif formatType == 3
        BitLenCsiPart1 = PucchParamGrps{grpIdx}.BitLenCsiPart1;
        BitLenHarq = PucchParamGrps{grpIdx}.bitLenHarq;
        BitLenSr = PucchParamGrps{grpIdx}.BitLenSr;
        numPart2s = PucchParamGrps{grpIdx}.numPart2s;
        if numPart2s > 0
            BitLenCsiPart2 = 1; %% can be any value > 0
        else
            BitLenCsiPart2 = 0;
        end
        
        %% derive PF3 parameters
        [F3Para] = deriveF3UciSeqTxSize(BitLenSr, BitLenHarq, BitLenCsiPart1, ...
        BitLenCsiPart2, freqHopFlag, PucchParamGrps{grpIdx}.AddDmrsFlag, ...
        PucchParamGrps{grpIdx}.maxCodeRate, pi2Bpsk, nSym, prbSize);
        dynamicDesc.nSymData{grpIdx}        = F3Para.nSymData;
        dynamicDesc.nSymDmrs{grpIdx}        = nSym - F3Para.nSymData;
        dynamicDesc.SetSymData{grpIdx}      = F3Para.SetSymData;
        dynamicDesc.SetSymDmrs{grpIdx}      = F3Para.SetSymDmrs;
        dynamicDesc.A_seg1{grpIdx}          = F3Para.A_seg(1);
        dynamicDesc.E_seg1{grpIdx}          = F3Para.E_seq(1);
        dynamicDesc.E_seg2{grpIdx}          = F3Para.E_seq(2);
        dynamicDesc.BitLenHarq{grpIdx}      = BitLenHarq;
        dynamicDesc.BitLenCsiPart1{grpIdx}  = BitLenCsiPart1;
        dynamicDesc.noiseVar{grpIdx}        = 10^(-40/10);
        dynamicDesc.addDmrsFlag{grpIdx}     = PucchParamGrps{grpIdx}.AddDmrsFlag;
        dynamicDesc.v{grpIdx}               = v;
        dynamicDesc.rankBitOffset{grpIdx}   = PucchParamGrps{grpIdx}.rankBitOffset;
        dynamicDesc.rankBitSize{grpIdx}     = PucchParamGrps{grpIdx}.rankBitSize;
        dynamicDesc.BitLenSr{grpIdx}        = PucchParamGrps{grpIdx}.BitLenSr;
        
        %% generate scrambling sequence
        cInitScrm = PucchParamGrps{grpIdx}.rnti*2^15 + PucchParamGrps{grpIdx}.dataScramblingId;
        dynamicDesc.randomSeqScrm{grpIdx}   = build_Gold_sequence(cInitScrm, F3Para.E_seq(1) + F3Para.E_seq(2));
    else % formatType == 0
        dynamicDesc.srFlag{grpIdx}              = PucchParamGrps{grpIdx}.srFlag;
    end
end
if formatType == 2 || formatType == 3
    dynamicDesc.nGrps = nGrps; % for PF3, the number of groups is equal to the number of UCIs since there is no UCI multiplexing on the same PRB
end

end

function saveTV_pucch_cuphy(tvDirName, TVname, cellStatPrm, cuphyPucchDynPrms, Xtf, PucchDataOut, carrier, polarBuffers, frontEndBuffers)
    if nargin < 8
        polarBuffers = {};
    end
    if nargin < 9
        frontEndBuffers = struct('f2', {{}}, 'f3', {{}});
    end

    global SimCtrl;

    %%create h5 file
    [status,msg] = mkdir(tvDirName);
    h5File       = H5F.create([tvDirName filesep TVname '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
    
    global SimCtrl
    %% static cell paramaters
    
    cellStatPrm.phyCellId  = uint16(cellStatPrm.phyCellId);
    cellStatPrm.nRxAnt     = uint16(cellStatPrm.nRxAnt);
    cellStatPrm.nTxAnt     = uint16(SimCtrl.UE{1}.Nant);
    cellStatPrm.nPrbUlBwp  = uint16(cellStatPrm.nPrbUlBwp);
    cellStatPrm.nPrbDlBwp  = uint16(cellStatPrm.nPrbDlBwp);
    cellStatPrm.mu         = uint8(cellStatPrm.mu);
    cellStatPrm.beta_dmrs  = single(cellStatPrm.beta_dmrs);
    cellStatPrm.beta_qam   = single(cellStatPrm.beta_qam);
    cellStatPrm.listLength = uint8(SimCtrl.alg.listLength);
    
    %% dynamic pucch cell paramaters
    
    cellDynPrm                = [];
    cellDynPrm.pucchHoppingId = uint16(cuphyPucchDynPrms.cuphyPucchCellDynPrm.pucchHoppingId);
    cellDynPrm.slotNum        = uint16(cuphyPucchDynPrms.cuphyPucchCellDynPrm.slotNum);
    
    %% F0 Uci Prms
    
    cuphyPucchUciPrmF0 = cuphyPucchDynPrms.cuphyPucchUciPrmF0;
    nF0Ucis = length(cuphyPucchUciPrmF0);
    F0UciPrms = [];
    
    for uciIdx = 1 : nF0Ucis
        F0UciPrms(uciIdx).BWPStart               = uint16(cuphyPucchUciPrmF0{uciIdx}.BWPStart);
        F0UciPrms(uciIdx).uciOutputIdx           = uint16(cuphyPucchUciPrmF0{uciIdx}.uciOutputIdx);   
        F0UciPrms(uciIdx).formatType             = uint8(cuphyPucchUciPrmF0{uciIdx}.formatType);   
        F0UciPrms(uciIdx).rnti                   = uint16(cuphyPucchUciPrmF0{uciIdx}.rnti);   
        F0UciPrms(uciIdx).multiSlotTxIndicator   = uint8(cuphyPucchUciPrmF0{uciIdx}.multiSlotTxIndicator);   
        F0UciPrms(uciIdx).pi2Bpsk                = uint8(cuphyPucchUciPrmF0{uciIdx}.pi2Bpsk);  
        F0UciPrms(uciIdx).startPrb               = uint16(cuphyPucchUciPrmF0{uciIdx}.startPrb);  
        F0UciPrms(uciIdx).startSym               = uint8(cuphyPucchUciPrmF0{uciIdx}.startSym);  
        F0UciPrms(uciIdx).nSym                   = uint8(cuphyPucchUciPrmF0{uciIdx}.nSym);  
        F0UciPrms(uciIdx).freqHopFlag            = uint8(cuphyPucchUciPrmF0{uciIdx}.freqHopFlag);  
        F0UciPrms(uciIdx).secondHopPrb           = uint16(cuphyPucchUciPrmF0{uciIdx}.secondHopPrb);  
        F0UciPrms(uciIdx).groupHopFlag           = uint8(cuphyPucchUciPrmF0{uciIdx}.groupHopFlag);  
        F0UciPrms(uciIdx).sequenceHopFlag        = uint8(cuphyPucchUciPrmF0{uciIdx}.sequenceHopFlag);  
        F0UciPrms(uciIdx).initialCyclicShift     = uint16(cuphyPucchUciPrmF0{uciIdx}.initialCyclicShift);  
        F0UciPrms(uciIdx).timeDomainOccIdx       = uint8(cuphyPucchUciPrmF0{uciIdx}.timeDomainOccIdx);  
        F0UciPrms(uciIdx).srFlag                 = uint8(cuphyPucchUciPrmF0{uciIdx}.srFlag);  
        F0UciPrms(uciIdx).bitLenHarq             = uint16(cuphyPucchUciPrmF0{uciIdx}.bitLenHarq); 
        if cuphyPucchUciPrmF0{uciIdx}.DTXthreshold ~= 1.0
            F0UciPrms(uciIdx).DTXthreshold       = single(cuphyPucchUciPrmF0{uciIdx}.DTXthreshold);
        else
            F0UciPrms(uciIdx).DTXthreshold       = single(-100.0);
        end
    end
    
    %% F1 Uci Prms
    cuphyPucchUciPrmF1 = cuphyPucchDynPrms.cuphyPucchUciPrmF1;
    nF1Ucis = length(cuphyPucchUciPrmF1);
    F1UciPrms = [];
    
    for uciIdx = 1 : nF1Ucis
        F1UciPrms(uciIdx).BWPStart               = uint16(cuphyPucchUciPrmF1{uciIdx}.BWPStart);
        F1UciPrms(uciIdx).uciOutputIdx           = uint16(cuphyPucchUciPrmF1{uciIdx}.uciOutputIdx);   
        F1UciPrms(uciIdx).formatType             = uint8(cuphyPucchUciPrmF1{uciIdx}.formatType);   
        F1UciPrms(uciIdx).rnti                   = uint16(cuphyPucchUciPrmF1{uciIdx}.rnti);   
        F1UciPrms(uciIdx).multiSlotTxIndicator   = uint8(cuphyPucchUciPrmF1{uciIdx}.multiSlotTxIndicator);   
        F1UciPrms(uciIdx).pi2Bpsk                = uint8(cuphyPucchUciPrmF1{uciIdx}.pi2Bpsk);  
        F1UciPrms(uciIdx).startPrb               = uint16(cuphyPucchUciPrmF1{uciIdx}.startPrb);  
        F1UciPrms(uciIdx).startSym               = uint8(cuphyPucchUciPrmF1{uciIdx}.startSym);  
        F1UciPrms(uciIdx).nSym                   = uint8(cuphyPucchUciPrmF1{uciIdx}.nSym);  
        F1UciPrms(uciIdx).freqHopFlag            = uint8(cuphyPucchUciPrmF1{uciIdx}.freqHopFlag);  
        F1UciPrms(uciIdx).secondHopPrb           = uint16(cuphyPucchUciPrmF1{uciIdx}.secondHopPrb);  
        F1UciPrms(uciIdx).groupHopFlag           = uint8(cuphyPucchUciPrmF1{uciIdx}.groupHopFlag);  
        F1UciPrms(uciIdx).sequenceHopFlag        = uint8(cuphyPucchUciPrmF1{uciIdx}.sequenceHopFlag);  
        F1UciPrms(uciIdx).initialCyclicShift     = uint16(cuphyPucchUciPrmF1{uciIdx}.initialCyclicShift);  
        F1UciPrms(uciIdx).timeDomainOccIdx       = uint8(cuphyPucchUciPrmF1{uciIdx}.timeDomainOccIdx);  
        F1UciPrms(uciIdx).srFlag                 = uint8(cuphyPucchUciPrmF1{uciIdx}.srFlag);  
        F1UciPrms(uciIdx).bitLenHarq             = uint16(cuphyPucchUciPrmF1{uciIdx}.bitLenHarq); 
        if cuphyPucchUciPrmF1{uciIdx}.DTXthreshold ~= 1.0
            F1UciPrms(uciIdx).DTXthreshold       = single(cuphyPucchUciPrmF1{uciIdx}.DTXthreshold);
        else
            F1UciPrms(uciIdx).DTXthreshold       = single(-100.0);
        end
    end
    
    
    %% F2 Uci Prms
    cuphyPucchUciPrmF2 = cuphyPucchDynPrms.cuphyPucchUciPrmF2;
    nF2Ucis = length(cuphyPucchUciPrmF2);
    F2UciPrms = [];
    
    for uciIdx = 1:nF2Ucis
        F2UciPrms(uciIdx).BWPStart               = uint16(cuphyPucchUciPrmF2{uciIdx}.BWPStart);
        F2UciPrms(uciIdx).uciOutputIdx           = uint16(cuphyPucchUciPrmF2{uciIdx}.uciOutputIdx);
        F2UciPrms(uciIdx).formatType             = uint8(cuphyPucchUciPrmF2{uciIdx}.formatType);
        F2UciPrms(uciIdx).rnti                   = uint16(cuphyPucchUciPrmF2{uciIdx}.rnti);
        F2UciPrms(uciIdx).startPrb               = uint16(cuphyPucchUciPrmF2{uciIdx}.startPrb);
        F2UciPrms(uciIdx).startSym               = uint8(cuphyPucchUciPrmF2{uciIdx}.startSym);
        F2UciPrms(uciIdx).prbSize                = uint8(cuphyPucchUciPrmF2{uciIdx}.prbSize);
        F2UciPrms(uciIdx).nSym                   = uint8(cuphyPucchUciPrmF2{uciIdx}.nSym);
        F2UciPrms(uciIdx).secondHopPrb           = uint16(cuphyPucchUciPrmF2{uciIdx}.secondHopPrb);
        F2UciPrms(uciIdx).freqHopFlag            = uint8(cuphyPucchUciPrmF2{uciIdx}.freqHopFlag);
        F2UciPrms(uciIdx).bitLenHarq             = uint16(cuphyPucchUciPrmF2{uciIdx}.bitLenHarq);
        F2UciPrms(uciIdx).bitLenSr               = uint16(cuphyPucchUciPrmF2{uciIdx}.BitLenSr);
        F2UciPrms(uciIdx).bitLenCsiPart1         = uint16(cuphyPucchUciPrmF2{uciIdx}.BitLenCsiPart1);
        F2UciPrms(uciIdx).dataScramblingId       = uint16(cuphyPucchUciPrmF2{uciIdx}.dataScramblingId);
        F2UciPrms(uciIdx).DmrsScramblingId       = uint16(cuphyPucchUciPrmF2{uciIdx}.DmrsScramblingId);     
        F2UciPrms(uciIdx).maxCodeRate            = uint8(cuphyPucchUciPrmF2{uciIdx}.maxCodeRate);
        F2UciPrms(uciIdx).DTXthreshold           = single(cuphyPucchUciPrmF2{uciIdx}.DTXthreshold); 
    end
    
    
    %% F3 Uci Prms
    cuphyPucchUciPrmF3 = cuphyPucchDynPrms.cuphyPucchUciPrmF3;
    nF3Ucis = length(cuphyPucchUciPrmF3);
    F3UciPrms = [];
    
    for uciIdx = 1:nF3Ucis
        F3UciPrms(uciIdx).BWPStart               = uint16(cuphyPucchUciPrmF3{uciIdx}.BWPStart);
        F3UciPrms(uciIdx).uciOutputIdx           = uint16(cuphyPucchUciPrmF3{uciIdx}.uciOutputIdx);
        F3UciPrms(uciIdx).formatType             = uint8(cuphyPucchUciPrmF3{uciIdx}.formatType);
        F3UciPrms(uciIdx).rnti                   = uint16(cuphyPucchUciPrmF3{uciIdx}.rnti);
        F3UciPrms(uciIdx).pi2Bpsk                = uint8(cuphyPucchUciPrmF3{uciIdx}.pi2Bpsk);
        F3UciPrms(uciIdx).startPrb               = uint16(cuphyPucchUciPrmF3{uciIdx}.startPrb);
        F3UciPrms(uciIdx).startSym               = uint8(cuphyPucchUciPrmF3{uciIdx}.startSym);
        F3UciPrms(uciIdx).prbSize                = uint8(cuphyPucchUciPrmF3{uciIdx}.prbSize);
        F3UciPrms(uciIdx).nSym                   = uint8(cuphyPucchUciPrmF3{uciIdx}.nSym);
        F3UciPrms(uciIdx).secondHopPrb           = uint16(cuphyPucchUciPrmF3{uciIdx}.secondHopPrb);
        F3UciPrms(uciIdx).freqHopFlag            = uint8(cuphyPucchUciPrmF3{uciIdx}.freqHopFlag);
        F3UciPrms(uciIdx).groupHopFlag           = uint8(cuphyPucchUciPrmF3{uciIdx}.groupHopFlag);
        F3UciPrms(uciIdx).sequenceHopFlag        = uint8(cuphyPucchUciPrmF3{uciIdx}.sequenceHopFlag);
        F3UciPrms(uciIdx).bitLenHarq             = uint16(cuphyPucchUciPrmF3{uciIdx}.bitLenHarq);
        F3UciPrms(uciIdx).bitLenSr               = uint16(cuphyPucchUciPrmF3{uciIdx}.BitLenSr);
        F3UciPrms(uciIdx).bitLenCsiPart1         = uint16(cuphyPucchUciPrmF3{uciIdx}.BitLenCsiPart1);
        F3UciPrms(uciIdx).AddDmrsFlag            = uint8(cuphyPucchUciPrmF3{uciIdx}.AddDmrsFlag);
        F3UciPrms(uciIdx).dataScramblingId       = uint16(cuphyPucchUciPrmF3{uciIdx}.dataScramblingId);
        F3UciPrms(uciIdx).numPart2s              = uint16(cuphyPucchUciPrmF3{uciIdx}.numPart2s);
        F3UciPrms(uciIdx).maxCodeRate            = uint8(cuphyPucchUciPrmF3{uciIdx}.maxCodeRate);
        F3UciPrms(uciIdx).DTXthreshold           = single(cuphyPucchUciPrmF3{uciIdx}.DTXthreshold); 
    end
    
    %% F0 Uci output
    
    F0UciOutRef = [];
    pF0UciOut = PucchDataOut.pUciF0;

    for uciIdx = 1 : nF0Ucis
        F0UciOutRef(uciIdx).taEstMicroSec       = single(pF0UciOut{uciIdx}.taEstMicroSec);
        F0UciOutRef(uciIdx).SinrDB              = single(pF0UciOut{uciIdx}.SinrDB);
        F0UciOutRef(uciIdx).InterfDB            = single(pF0UciOut{uciIdx}.InterfDB);
        F0UciOutRef(uciIdx).RSSI                = single(pF0UciOut{uciIdx}.RSSI);
        F0UciOutRef(uciIdx).RSRP                = single(pF0UciOut{uciIdx}.RSRP);
        F0UciOutRef(uciIdx).SRindication        = uint8(pF0UciOut{uciIdx}.SRindication);
        F0UciOutRef(uciIdx).NumHarq             = uint8(pF0UciOut{uciIdx}.NumHarq);
        F0UciOutRef(uciIdx).HarqValue0          = uint8(pF0UciOut{uciIdx}.HarqValues(1));   
        F0UciOutRef(uciIdx).HarqValue1          = uint8(pF0UciOut{uciIdx}.HarqValues(2)); 
        F0UciOutRef(uciIdx).SRconfidenceLevel   = uint8(pF0UciOut{uciIdx}.SRconfidenceLevel);
        F0UciOutRef(uciIdx).HarqconfidenceLevel = uint8(pF0UciOut{uciIdx}.HarqconfidenceLevel);
    end
    
    %% F1 Uci output
    
    F1UciOutRef = [];
    pF1UciOut = PucchDataOut.pUciF1;
    
    for uciIdx = 1 : nF1Ucis
        F1UciOutRef(uciIdx).taEstMicroSec       = single(pF1UciOut{uciIdx}.taEstMicroSec);
        F1UciOutRef(uciIdx).SinrDB              = single(pF1UciOut{uciIdx}.SinrDB);
        F1UciOutRef(uciIdx).InterfDB            = single(pF1UciOut{uciIdx}.InterfDB);
        F1UciOutRef(uciIdx).RSSI                = single(pF1UciOut{uciIdx}.RSSI);
        F1UciOutRef(uciIdx).RSRP                = single(pF1UciOut{uciIdx}.RSRP);
        F1UciOutRef(uciIdx).SRindication        = uint8(pF1UciOut{uciIdx}.SRindication);
        F1UciOutRef(uciIdx).NumHarq             = uint8(pF1UciOut{uciIdx}.NumHarq);
        F1UciOutRef(uciIdx).HarqValue0          = uint8(pF1UciOut{uciIdx}.HarqValues(1));   
        F1UciOutRef(uciIdx).HarqValue1          = uint8(pF1UciOut{uciIdx}.HarqValues(2)); 
        F1UciOutRef(uciIdx).SRconfidenceLevel   = uint8(pF1UciOut{uciIdx}.SRconfidenceLevel);
        F1UciOutRef(uciIdx).HarqconfidenceLevel = uint8(pF1UciOut{uciIdx}.HarqconfidenceLevel);
    end
    
    %% F2 and F3 Uci output
    pucchF234_refLLRbuffer     = [];
    pucchF234_refSeg1LLRbuffer = [];
    pucchF234_refSeg2LLRbuffer = [];
    LLRsOffset                 = 0;
    Seg1LLRsOffset             = 0;
    Seg2LLRsOffset             = 0;
    
    pucchF234_refPayloadBuffer = [];
    nPayloadBytes              = 0;
    
    pucchF234_refDTXbuffer  = [];
    nDTXflags               = 0;
    nSnrs                   = 0;
    nTas                    = 0;
    
    pucchF234_refSnrBuffer    = [];
    pucchF234_refRsrpBuffer   = [];
    pucchF234_refRssiBuffer   = [];
    pucchF234_refInterfBuffer = [];
    pucchF234_refTaBuffer     = [];
    
    % detection status
    pucchF234_refHarqDetStatBuffer     = [];
    pucchF234_refCsiPart1DetStatBuffer = [];
    pucchF234_refCsiPart2DetStatBuffer = [];
    nDetStat                           = 0;
    
    %% F2
    pF2UciOut = PucchDataOut.pUciF2;
    
    pucchF2_refBufferOffsets = [];
    
    for uciIdx = 0 : (nF2Ucis - 1)
        
        pucchF234_refDTXbuffer    = [pucchF234_refDTXbuffer pF2UciOut{uciIdx + 1}.DTX];
        pucchF234_refSnrBuffer    = [pucchF234_refSnrBuffer    pF2UciOut{uciIdx + 1}.SinrDB];
        pucchF234_refRsrpBuffer   = [pucchF234_refRsrpBuffer   pF2UciOut{uciIdx + 1}.RSRP];
        pucchF234_refRssiBuffer   = [pucchF234_refRssiBuffer   pF2UciOut{uciIdx + 1}.RSSI];
        pucchF234_refInterfBuffer = [pucchF234_refInterfBuffer pF2UciOut{uciIdx + 1}.InterfDB];
        pucchF234_refTaBuffer     = [pucchF234_refTaBuffer     pF2UciOut{uciIdx + 1}.taEstMicroSec];
        pucchF234_refHarqDetStatBuffer     = [pucchF234_refHarqDetStatBuffer     pF2UciOut{uciIdx + 1}.HarqDetectionStatus];
        pucchF234_refCsiPart1DetStatBuffer = [pucchF234_refCsiPart1DetStatBuffer pF2UciOut{uciIdx + 1}.CsiPart1DetectionStatus];
        pucchF234_refCsiPart2DetStatBuffer = [pucchF234_refCsiPart2DetStatBuffer pF2UciOut{uciIdx + 1}.CsiPart2DetectionStatus];
         % initialize offset paramaters:
        pucchF2_refBufferOffsets(uciIdx + 1).dtxFlagOffset         = nDTXflags;
        nDTXflags = nDTXflags + 1;
        
        pucchF2_refBufferOffsets(uciIdx + 1).snrOffset             = nSnrs;
        pucchF2_refBufferOffsets(uciIdx + 1).RSRPoffset            = nSnrs;
        pucchF2_refBufferOffsets(uciIdx + 1).RSSIoffset            = nSnrs;
        pucchF2_refBufferOffsets(uciIdx + 1).InterfOffset          = nSnrs;
        pucchF2_refBufferOffsets(uciIdx + 1).taOffset              = nTas;
        
        nTas = nTas + 1;
        nSnrs = nSnrs + 1;
        
        pucchF2_refBufferOffsets(uciIdx + 1).harqDetStatOffset     = nDetStat;
        pucchF2_refBufferOffsets(uciIdx + 1).csiPart1DetStatOffset = nDetStat;
        pucchF2_refBufferOffsets(uciIdx + 1).csiPart2DetStatOffset = nDetStat;
        nDetStat = nDetStat + 1;
        
        pucchF2_refBufferOffsets(uciIdx + 1).uciSeg1PayloadByteOffset = 0;
        pucchF2_refBufferOffsets(uciIdx + 1).nUciSeg1Bytes            = 0;
        pucchF2_refBufferOffsets(uciIdx + 1).harqPayloadByteOffset    = 0;
        pucchF2_refBufferOffsets(uciIdx + 1).nHarqBytes               = 0;
        pucchF2_refBufferOffsets(uciIdx + 1).srPayloadByteOffset      = 0;
        pucchF2_refBufferOffsets(uciIdx + 1).nSrBytes                 = 0;
        pucchF2_refBufferOffsets(uciIdx + 1).csiP1PayloadByteOffset   = 0;
        pucchF2_refBufferOffsets(uciIdx + 1).nCsiP1Bytes              = 0;
        pucchF2_refBufferOffsets(uciIdx + 1).LLRsoffset               = 0;
        pucchF2_refBufferOffsets(uciIdx + 1).nSegLLRs                 = 0;
        
        % uci segment-1 LLRs
        segLLRs  = reshape(pF2UciOut{uciIdx + 1}.descrmLLR, 1, []);
        nSegLLRs = length(segLLRs);
        
        pucchF2_refBufferOffsets(uciIdx + 1).LLRsoffset = LLRsOffset;
        pucchF2_refBufferOffsets(uciIdx + 1).nSegLLRs   = nSegLLRs;
        
        pucchF234_refLLRbuffer = [pucchF234_refLLRbuffer segLLRs]; 
        LLRsOffset             = LLRsOffset + nSegLLRs;
        
        % UciSeg1 payload
        bitLenUciSeg1 = double(F2UciPrms(uciIdx + 1).bitLenHarq) + double(F2UciPrms(uciIdx + 1).bitLenSr) + double(F2UciPrms(uciIdx + 1).bitLenCsiPart1);
        if(bitLenUciSeg1 > 0)
            nUciSeg1Bytes = 4*ceil(bitLenUciSeg1 / 32);
            uciSeg1Values = pF2UciOut{uciIdx + 1}.uciSeg1;
            uciSeg1Values = [uciSeg1Values; zeros(nUciSeg1Bytes*8-bitLenUciSeg1,1)];
            
            pucchF2_refBufferOffsets(uciIdx + 1).uciSeg1PayloadByteOffset = nPayloadBytes;
            pucchF2_refBufferOffsets(uciIdx + 1).nUciSeg1Bytes            = nUciSeg1Bytes;
            
            for byteIdx = 0 : (nUciSeg1Bytes - 1)    
                byte = 0;
                for i = 0 : 7
                    if(uciSeg1Values(byteIdx*8 + i + 1))
                        byte = byte + 2^i;
                    end
                end
                pucchF234_refPayloadBuffer(byteIdx + nPayloadBytes + 1) = uint8(byte);
            end
            nPayloadBytes = nPayloadBytes + nUciSeg1Bytes;
        end
        
        % harq payload
        bitLenHarq = double(F2UciPrms(uciIdx + 1).bitLenHarq);
        if(bitLenHarq > 0)
            nHarqBytes = 4*ceil(bitLenHarq / 32);
            HarqValues = pF2UciOut{uciIdx + 1}.HarqValues;
            HarqValues = [HarqValues; zeros(nHarqBytes*8-bitLenHarq, 1)];
            
            pucchF2_refBufferOffsets(uciIdx + 1).harqPayloadByteOffset = nPayloadBytes;
            pucchF2_refBufferOffsets(uciIdx + 1).nHarqBytes            = nHarqBytes;
            
            for byteIdx = 0 : (nHarqBytes - 1)    
                byte = 0;
                for i = 0 : 7
                    if (HarqValues(byteIdx*8 + i + 1))
                        byte = byte + 2^i;
                    end
                end
                pucchF234_refPayloadBuffer(byteIdx + nPayloadBytes + 1) = uint8(byte);
            end
            nPayloadBytes = nPayloadBytes + nHarqBytes;
        end
        
        % Sr payload
        BitLenSr = double(F2UciPrms(uciIdx + 1).bitLenSr);
        if (BitLenSr > 0)
           nSrBytes = 4*ceil(BitLenSr / 32);
           SrValues = pF2UciOut{uciIdx + 1}.SrValues;
           SrValues = [SrValues; zeros(nSrBytes*8-BitLenSr, 1)];
           
           pucchF2_refBufferOffsets(uciIdx + 1).srPayloadByteOffset = nPayloadBytes;
           pucchF2_refBufferOffsets(uciIdx + 1).nSrBytes            = nSrBytes;
           
           for byteIdx = 0 : (nSrBytes - 1)
               byte = 0;
               for i = 0 : 7
                   if (SrValues(byteIdx*8 + i + 1))
                       byte = byte + 2^i;
                   end
               end
               pucchF234_refPayloadBuffer(byteIdx + nPayloadBytes + 1) = uint8(byte);
           end
           nPayloadBytes = nPayloadBytes + nSrBytes;
        end
        
        % CsiP1 payload
        BitLenCsiPart1 = double(F2UciPrms(uciIdx + 1).bitLenCsiPart1);
        if (BitLenCsiPart1 > 0)
           nCsiP1Bytes = 4*ceil(BitLenCsiPart1 / 32); 
           CsiP1Values = pF2UciOut{uciIdx + 1}.CsiP1Values;
           CsiP1Values = [CsiP1Values; zeros(nCsiP1Bytes*8-BitLenCsiPart1, 1)];
           
           pucchF2_refBufferOffsets(uciIdx + 1).csiP1PayloadByteOffset = nPayloadBytes;
           pucchF2_refBufferOffsets(uciIdx + 1).nCsiP1Bytes            = nCsiP1Bytes;
           
           for byteIdx = 0 : (nCsiP1Bytes - 1)
               byte = 0;
               for i = 0 : 7
                   if (CsiP1Values(byteIdx*8 + i + 1))
                       byte = byte + 2^i;
                   end
               end
               pucchF234_refPayloadBuffer(byteIdx + nPayloadBytes + 1) = uint8(byte);
           end
           nPayloadBytes = nPayloadBytes + nCsiP1Bytes;
        end
        
        % cast output offsets
        pucchF2_refBufferOffsets(uciIdx + 1).dtxFlagOffset            = uint32(pucchF2_refBufferOffsets(uciIdx + 1).dtxFlagOffset);
        pucchF2_refBufferOffsets(uciIdx + 1).snrOffset                = uint32(pucchF2_refBufferOffsets(uciIdx + 1).snrOffset);
        pucchF2_refBufferOffsets(uciIdx + 1).RSRPoffset               = uint32(pucchF2_refBufferOffsets(uciIdx + 1).RSRPoffset);
        pucchF2_refBufferOffsets(uciIdx + 1).RSSIoffset               = uint32(pucchF2_refBufferOffsets(uciIdx + 1).RSSIoffset);
        pucchF2_refBufferOffsets(uciIdx + 1).InterfOffset             = uint32(pucchF2_refBufferOffsets(uciIdx + 1).InterfOffset);
        pucchF2_refBufferOffsets(uciIdx + 1).taOffset                 = uint32(pucchF2_refBufferOffsets(uciIdx + 1).taOffset);
        pucchF2_refBufferOffsets(uciIdx + 1).uciSeg1PayloadByteOffset = uint32(pucchF2_refBufferOffsets(uciIdx + 1).uciSeg1PayloadByteOffset);
        pucchF2_refBufferOffsets(uciIdx + 1).nUciSeg1Bytes            = uint32(pucchF2_refBufferOffsets(uciIdx + 1).nUciSeg1Bytes);
        pucchF2_refBufferOffsets(uciIdx + 1).harqPayloadByteOffset    = uint32(pucchF2_refBufferOffsets(uciIdx + 1).harqPayloadByteOffset);
        pucchF2_refBufferOffsets(uciIdx + 1).nHarqBytes               = uint32(pucchF2_refBufferOffsets(uciIdx + 1).nHarqBytes);
        pucchF2_refBufferOffsets(uciIdx + 1).srPayloadByteOffset      = uint32(pucchF2_refBufferOffsets(uciIdx + 1).srPayloadByteOffset);
        pucchF2_refBufferOffsets(uciIdx + 1).nSrBytes                 = uint32(pucchF2_refBufferOffsets(uciIdx + 1).nSrBytes);
        pucchF2_refBufferOffsets(uciIdx + 1).csiP1PayloadByteOffset   = uint32(pucchF2_refBufferOffsets(uciIdx + 1).csiP1PayloadByteOffset);
        pucchF2_refBufferOffsets(uciIdx + 1).nCsiP1Bytes              = uint32(pucchF2_refBufferOffsets(uciIdx + 1).nCsiP1Bytes);
        pucchF2_refBufferOffsets(uciIdx + 1).LLRsoffset               = uint32(pucchF2_refBufferOffsets(uciIdx + 1).LLRsoffset);
        pucchF2_refBufferOffsets(uciIdx + 1).nSegLLRs                 = uint32(pucchF2_refBufferOffsets(uciIdx + 1).nSegLLRs);
        pucchF2_refBufferOffsets(uciIdx + 1).harqDetStatOffset        = uint32(pucchF2_refBufferOffsets(uciIdx + 1).harqDetStatOffset);
        pucchF2_refBufferOffsets(uciIdx + 1).csiPart1DetStatOffset    = uint32(pucchF2_refBufferOffsets(uciIdx + 1).csiPart1DetStatOffset);
        pucchF2_refBufferOffsets(uciIdx + 1).csiPart2DetStatOffset    = uint32(pucchF2_refBufferOffsets(uciIdx + 1).csiPart2DetStatOffset);
    end
    
    
    %% F3 
    
    pF3UciOut = PucchDataOut.pUciF3;
    
    pucchF3_refBufferOffsets = [];
    
    for uciIdx = 0 : (nF3Ucis - 1)
        
        % DTX flag (for now always zero)
        pucchF234_refDTXbuffer = [pucchF234_refDTXbuffer pF3UciOut{uciIdx + 1}.DTX];
        pucchF234_refSnrBuffer    = [pucchF234_refSnrBuffer    pF3UciOut{uciIdx + 1}.SinrDB];
        pucchF234_refRsrpBuffer   = [pucchF234_refRsrpBuffer   pF3UciOut{uciIdx + 1}.RSRP];
        pucchF234_refRssiBuffer   = [pucchF234_refRssiBuffer   pF3UciOut{uciIdx + 1}.RSSI];
        pucchF234_refInterfBuffer = [pucchF234_refInterfBuffer pF3UciOut{uciIdx + 1}.InterfDB];
        pucchF234_refTaBuffer     = [pucchF234_refTaBuffer     pF3UciOut{uciIdx + 1}.taEstMicroSec];
        pucchF234_refHarqDetStatBuffer     = [pucchF234_refHarqDetStatBuffer     pF3UciOut{uciIdx + 1}.HarqDetectionStatus];
        pucchF234_refCsiPart1DetStatBuffer = [pucchF234_refCsiPart1DetStatBuffer pF3UciOut{uciIdx + 1}.CsiPart1DetectionStatus];
        pucchF234_refCsiPart2DetStatBuffer = [pucchF234_refCsiPart2DetStatBuffer pF3UciOut{uciIdx + 1}.CsiPart2DetectionStatus];
        
        % initialize offset paramaters:
        pucchF3_refBufferOffsets(uciIdx + 1).dtxFlagOffset         = nDTXflags;
        nDTXflags = nDTXflags + 1;
        
        pucchF3_refBufferOffsets(uciIdx + 1).snrOffset             = nSnrs;
        pucchF3_refBufferOffsets(uciIdx + 1).RSRPoffset            = nSnrs;
        pucchF3_refBufferOffsets(uciIdx + 1).RSSIoffset            = nSnrs;
        pucchF3_refBufferOffsets(uciIdx + 1).InterfOffset          = nSnrs;
        pucchF3_refBufferOffsets(uciIdx + 1).taOffset              = nTas;
        
        nTas = nTas + 1;
        nSnrs = nSnrs + 1;
        
        pucchF3_refBufferOffsets(uciIdx + 1).harqDetStatOffset     = nDetStat;
        pucchF3_refBufferOffsets(uciIdx + 1).csiPart1DetStatOffset = nDetStat;
        pucchF3_refBufferOffsets(uciIdx + 1).csiPart2DetStatOffset = nDetStat;
        nDetStat = nDetStat + 1;
        
        pucchF3_refBufferOffsets(uciIdx + 1).uciSeg1PayloadByteOffset = 0;
        pucchF3_refBufferOffsets(uciIdx + 1).nUciSeg1Bytes            = 0;
        pucchF3_refBufferOffsets(uciIdx + 1).harqPayloadByteOffset    = 0;
        pucchF3_refBufferOffsets(uciIdx + 1).nHarqBytes               = 0;
        pucchF3_refBufferOffsets(uciIdx + 1).srPayloadByteOffset      = 0;
        pucchF3_refBufferOffsets(uciIdx + 1).nSrBytes                 = 0;
        pucchF3_refBufferOffsets(uciIdx + 1).csiP1PayloadByteOffset   = 0;
        pucchF3_refBufferOffsets(uciIdx + 1).nCsiP1Bytes              = 0;
        pucchF3_refBufferOffsets(uciIdx + 1).csiP2PayloadByteOffset   = 0;
        pucchF3_refBufferOffsets(uciIdx + 1).nCsiP2Bytes              = 0;
        pucchF3_refBufferOffsets(uciIdx + 1).BitLenCsiPart2           = 0;
        pucchF3_refBufferOffsets(uciIdx + 1).LLRsoffset               = 0;
        pucchF3_refBufferOffsets(uciIdx + 1).Seg1LLRsoffset           = 0;
        pucchF3_refBufferOffsets(uciIdx + 1).Seg2LLRsoffset           = 0;
        pucchF3_refBufferOffsets(uciIdx + 1).nSegLLRs                 = 0;
        pucchF3_refBufferOffsets(uciIdx + 1).nSeg1LLRs                = 0;
        pucchF3_refBufferOffsets(uciIdx + 1).nSeg2LLRs                = 0;
        
        % uci segment-1 LLRs
        segLLRs   = reshape(pF3UciOut{uciIdx + 1}.descrmLLR, 1, []);
        seg1LLRs  = reshape(pF3UciOut{uciIdx + 1}.descrmLLRSeq1, 1, []);
        seg2LLRs  = reshape(pF3UciOut{uciIdx + 1}.descrmLLRSeq2, 1, []);
        nSegLLRs  = length(segLLRs);
        nSeg1LLRs = length(seg1LLRs);
        nSeg2LLRs = length(seg2LLRs);
        
        pucchF3_refBufferOffsets(uciIdx + 1).LLRsoffset     = LLRsOffset;
        pucchF3_refBufferOffsets(uciIdx + 1).Seg1LLRsoffset = Seg1LLRsOffset;
        pucchF3_refBufferOffsets(uciIdx + 1).Seg2LLRsoffset = Seg2LLRsOffset;
        pucchF3_refBufferOffsets(uciIdx + 1).nSegLLRs       = nSegLLRs;
        pucchF3_refBufferOffsets(uciIdx + 1).nSeg1LLRs      = nSeg1LLRs;
        pucchF3_refBufferOffsets(uciIdx + 1).nSeg2LLRs      = nSeg2LLRs;
        
        pucchF234_refLLRbuffer     = [pucchF234_refLLRbuffer segLLRs];   
        pucchF234_refSeg1LLRbuffer = [pucchF234_refSeg1LLRbuffer seg1LLRs];
        pucchF234_refSeg2LLRbuffer = [pucchF234_refSeg2LLRbuffer seg2LLRs];
        
        LLRsOffset                 = LLRsOffset + nSegLLRs;
        Seg1LLRsOffset             = Seg1LLRsOffset + nSeg1LLRs;
        Seg2LLRsOffset             = Seg2LLRsOffset + nSeg2LLRs;
                
        % UciSeg1 payload
        bitLenUciSeg1 = double(F3UciPrms(uciIdx + 1).bitLenHarq) + double(F3UciPrms(uciIdx + 1).bitLenSr) + double(F3UciPrms(uciIdx + 1).bitLenCsiPart1);
        if(bitLenUciSeg1 > 0)
            nUciSeg1Bytes = 4*ceil(bitLenUciSeg1 / 32);
            uciSeg1Values = pF3UciOut{uciIdx + 1}.uciSeg1;
            uciSeg1Values = [uciSeg1Values; zeros(nUciSeg1Bytes*8-bitLenUciSeg1,1)];
            
            pucchF3_refBufferOffsets(uciIdx + 1).uciSeg1PayloadByteOffset = nPayloadBytes;
            pucchF3_refBufferOffsets(uciIdx + 1).nUciSeg1Bytes            = nUciSeg1Bytes;
            
            for byteIdx = 0 : (nUciSeg1Bytes - 1)    
                byte = 0;
                for i = 0 : 7
                    if(uciSeg1Values(byteIdx*8 + i + 1))
                        byte = byte + 2^i;
                    end
                end
                pucchF234_refPayloadBuffer(byteIdx + nPayloadBytes + 1) = uint8(byte);
            end
            nPayloadBytes = nPayloadBytes + nUciSeg1Bytes;
        end
 
        % harq payload
        bitLenHarq = double(F3UciPrms(uciIdx + 1).bitLenHarq);
        if(bitLenHarq > 0)
            nHarqBytes = 4*ceil(bitLenHarq / 32);
            HarqValues = pF3UciOut{uciIdx + 1}.HarqValues;
            HarqValues = [HarqValues; zeros(nHarqBytes*8-bitLenHarq,1)];
            
            pucchF3_refBufferOffsets(uciIdx + 1).harqPayloadByteOffset = nPayloadBytes;
            pucchF3_refBufferOffsets(uciIdx + 1).nHarqBytes            = nHarqBytes;

            for byteIdx = 0 : (nHarqBytes - 1)    
                byte = 0;
                for i = 0 : 7
                    if(HarqValues(byteIdx*8 + i + 1))
                        byte = byte + 2^i;
                    end
                end
                pucchF234_refPayloadBuffer(byteIdx + nPayloadBytes + 1) = uint8(byte);
            end
            nPayloadBytes = nPayloadBytes + nHarqBytes;
        end
        
        % Sr payload
        BitLenSr = double(F3UciPrms(uciIdx + 1).bitLenSr);
        if (BitLenSr > 0)
           nSrBytes = 4*ceil(BitLenSr / 32);
           SrValues = pF3UciOut{uciIdx + 1}.SrValues;
           SrValues = [SrValues; zeros(nSrBytes*8-BitLenSr, 1)];
           
           pucchF3_refBufferOffsets(uciIdx + 1).srPayloadByteOffset = nPayloadBytes;
           pucchF3_refBufferOffsets(uciIdx + 1).nSrBytes            = nSrBytes;
           
           for byteIdx = 0 : (nSrBytes - 1)
               byte = 0;
               for i = 0 : 7
                   if (SrValues(byteIdx*8 + i + 1))
                       byte = byte + 2^i;
                   end
               end
               pucchF234_refPayloadBuffer(byteIdx + nPayloadBytes + 1) = uint8(byte);
           end
           nPayloadBytes = nPayloadBytes + nSrBytes;
        end
        
        % CsiP1 payload
        BitLenCsiPart1 = double(F3UciPrms(uciIdx + 1).bitLenCsiPart1);
        if (BitLenCsiPart1 > 0)
           nCsiP1Bytes = 4*ceil(BitLenCsiPart1 / 32); 
           CsiP1Values = pF3UciOut{uciIdx + 1}.CsiP1Values;
           CsiP1Values = [CsiP1Values; zeros(nCsiP1Bytes*8-BitLenCsiPart1, 1)];
           
           pucchF3_refBufferOffsets(uciIdx + 1).csiP1PayloadByteOffset = nPayloadBytes;
           pucchF3_refBufferOffsets(uciIdx + 1).nCsiP1Bytes            = nCsiP1Bytes;
           
           for byteIdx = 0 : (nCsiP1Bytes - 1)
               byte = 0;
               for i = 0 : 7
                   if (CsiP1Values(byteIdx*8 + i + 1))
                       byte = byte + 2^i;
                   end
               end
               pucchF234_refPayloadBuffer(byteIdx + nPayloadBytes + 1) = uint8(byte);
           end
           nPayloadBytes = nPayloadBytes + nCsiP1Bytes;
        end
        
        % CsiP2 payload
        BitLenCsiPart2 = length(pF3UciOut{uciIdx + 1}.CsiP2Values);
        if (BitLenCsiPart2 > 0)
            nCsiP2Bytes = 4*ceil(BitLenCsiPart2 / 32); 
            CsiP2Values = pF3UciOut{uciIdx + 1}.CsiP2Values;
            CsiP2Values = [CsiP2Values; zeros(nCsiP2Bytes*8-BitLenCsiPart2, 1)];
            
            pucchF3_refBufferOffsets(uciIdx + 1).csiP2PayloadByteOffset = nPayloadBytes;
            pucchF3_refBufferOffsets(uciIdx + 1).nCsiP2Bytes            = nCsiP2Bytes;
            pucchF3_refBufferOffsets(uciIdx + 1).BitLenCsiPart2         = BitLenCsiPart2;
            
            for byteIdx = 0 : (nCsiP2Bytes - 1)
                byte = 0;
                for i = 0 : 7
                   if (CsiP2Values(byteIdx*8 + i + 1))
                       byte = byte + 2^i;
                   end
                end
                pucchF234_refPayloadBuffer(byteIdx + nPayloadBytes + 1) = uint8(byte);
            end
            nPayloadBytes = nPayloadBytes + nCsiP2Bytes;
        end
        
        % cast output offsets
        pucchF3_refBufferOffsets(uciIdx + 1).dtxFlagOffset            = uint32(pucchF3_refBufferOffsets(uciIdx + 1).dtxFlagOffset);
        pucchF3_refBufferOffsets(uciIdx + 1).snrOffset                = uint32(pucchF3_refBufferOffsets(uciIdx + 1).snrOffset);
        pucchF3_refBufferOffsets(uciIdx + 1).RSRPoffset               = uint32(pucchF3_refBufferOffsets(uciIdx + 1).RSRPoffset);
        pucchF3_refBufferOffsets(uciIdx + 1).RSSIoffset               = uint32(pucchF3_refBufferOffsets(uciIdx + 1).RSSIoffset);
        pucchF3_refBufferOffsets(uciIdx + 1).InterfOffset             = uint32(pucchF3_refBufferOffsets(uciIdx + 1).InterfOffset);
        pucchF3_refBufferOffsets(uciIdx + 1).taOffset                 = uint32(pucchF3_refBufferOffsets(uciIdx + 1).taOffset);
        pucchF3_refBufferOffsets(uciIdx + 1).uciSeg1PayloadByteOffset = uint32(pucchF3_refBufferOffsets(uciIdx + 1).uciSeg1PayloadByteOffset);
        pucchF3_refBufferOffsets(uciIdx + 1).nUciSeg1Bytes            = uint32(pucchF3_refBufferOffsets(uciIdx + 1).nUciSeg1Bytes);
        pucchF3_refBufferOffsets(uciIdx + 1).harqPayloadByteOffset    = uint32(pucchF3_refBufferOffsets(uciIdx + 1).harqPayloadByteOffset);
        pucchF3_refBufferOffsets(uciIdx + 1).nHarqBytes               = uint32(pucchF3_refBufferOffsets(uciIdx + 1).nHarqBytes);
        pucchF3_refBufferOffsets(uciIdx + 1).srPayloadByteOffset      = uint32(pucchF3_refBufferOffsets(uciIdx + 1).srPayloadByteOffset);
        pucchF3_refBufferOffsets(uciIdx + 1).nSrBytes                 = uint32(pucchF3_refBufferOffsets(uciIdx + 1).nSrBytes);
        pucchF3_refBufferOffsets(uciIdx + 1).csiP1PayloadByteOffset   = uint32(pucchF3_refBufferOffsets(uciIdx + 1).csiP1PayloadByteOffset);
        pucchF3_refBufferOffsets(uciIdx + 1).nCsiP1Bytes              = uint32(pucchF3_refBufferOffsets(uciIdx + 1).nCsiP1Bytes);
        pucchF3_refBufferOffsets(uciIdx + 1).csiP2PayloadByteOffset   = uint32(pucchF3_refBufferOffsets(uciIdx + 1).csiP2PayloadByteOffset);
        pucchF3_refBufferOffsets(uciIdx + 1).nCsiP2Bytes              = uint32(pucchF3_refBufferOffsets(uciIdx + 1).nCsiP2Bytes);
        pucchF3_refBufferOffsets(uciIdx + 1).BitLenCsiPart2           = uint32(pucchF3_refBufferOffsets(uciIdx + 1).BitLenCsiPart2);
        pucchF3_refBufferOffsets(uciIdx + 1).LLRsoffset               = uint32(pucchF3_refBufferOffsets(uciIdx + 1).LLRsoffset);
        pucchF3_refBufferOffsets(uciIdx + 1).Seg1LLRsoffset           = uint32(pucchF3_refBufferOffsets(uciIdx + 1).Seg1LLRsoffset);
        pucchF3_refBufferOffsets(uciIdx + 1).Seg2LLRsoffset           = uint32(pucchF3_refBufferOffsets(uciIdx + 1).Seg2LLRsoffset);
        pucchF3_refBufferOffsets(uciIdx + 1).nSegLLRs                 = uint32(pucchF3_refBufferOffsets(uciIdx + 1).nSegLLRs);
        pucchF3_refBufferOffsets(uciIdx + 1).nSeg1LLRs                = uint32(pucchF3_refBufferOffsets(uciIdx + 1).nSeg1LLRs);
        pucchF3_refBufferOffsets(uciIdx + 1).nSeg2LLRs                = uint32(pucchF3_refBufferOffsets(uciIdx + 1).nSeg2LLRs);
        pucchF3_refBufferOffsets(uciIdx + 1).harqDetStatOffset        = uint32(pucchF3_refBufferOffsets(uciIdx + 1).harqDetStatOffset);
        pucchF3_refBufferOffsets(uciIdx + 1).csiPart1DetStatOffset    = uint32(pucchF3_refBufferOffsets(uciIdx + 1).csiPart1DetStatOffset);
        pucchF3_refBufferOffsets(uciIdx + 1).csiPart2DetStatOffset    = uint32(pucchF3_refBufferOffsets(uciIdx + 1).csiPart2DetStatOffset);
    end

    %% Cell grp paramaters
    
    cellGrpDynPrm = [];
    cellGrpDynPrm.nF0Ucis = uint16(nF0Ucis);
    cellGrpDynPrm.nF1Ucis = uint16(nF1Ucis);
    cellGrpDynPrm.nF2Ucis = uint16(nF2Ucis);
    cellGrpDynPrm.nF3Ucis = uint16(nF3Ucis);
    
    %% write to H5    
    
    hdf5_write_nv_exp(h5File, 'cellStatPrm'  , cellStatPrm);
    hdf5_write_nv_exp(h5File, 'cellDynPrm'   , cellDynPrm);
    hdf5_write_nv_exp(h5File, 'F0UciPrms'    , F0UciPrms);
    hdf5_write_nv_exp(h5File, 'F0UcisOutRef' , F0UciOutRef);
    hdf5_write_nv_exp(h5File, 'F1UciPrms'    , F1UciPrms);
    hdf5_write_nv_exp(h5File, 'F1UcisOutRef' , F1UciOutRef);
    hdf5_write_nv_exp(h5File, 'F2UciPrms'    , F2UciPrms);
    hdf5_write_nv_exp(h5File, 'F3UciPrms'    , F3UciPrms);
    hdf5_write_nv_exp(h5File, 'cellGrpDynPrm', cellGrpDynPrm);
    hdf5_write_nv_exp(h5File, 'pucchF2_refBufferOffsets'   , pucchF2_refBufferOffsets);
    hdf5_write_nv_exp(h5File, 'pucchF3_refBufferOffsets'   , pucchF3_refBufferOffsets);
    hdf5_write_nv_exp(h5File, 'pucchF234_refPayloadBuffer' , uint8(pucchF234_refPayloadBuffer(:)));
    hdf5_write_nv_exp(h5File, 'pucchF234_refLLRbuffer'     , fp16nv(pucchF234_refLLRbuffer(:), SimCtrl.fp16AlgoSel));
    hdf5_write_nv_exp(h5File, 'pucchF234_refSeg1LLRbuffer' , fp16nv(pucchF234_refSeg1LLRbuffer(:), SimCtrl.fp16AlgoSel));
    hdf5_write_nv_exp(h5File, 'pucchF234_refSeg2LLRbuffer' , fp16nv(pucchF234_refSeg2LLRbuffer(:), SimCtrl.fp16AlgoSel));
    
    hdf5_write_nv_exp(h5File, 'pucchF234_refDTXbuffer'            , uint8(pucchF234_refDTXbuffer(:)));
    hdf5_write_nv_exp(h5File, 'pucchF234_refSnrBuffer'            , fp16nv(pucchF234_refSnrBuffer(:),    SimCtrl.fp16AlgoSel));
    hdf5_write_nv_exp(h5File, 'pucchF234_refRsrpBuffer'           , fp16nv(pucchF234_refRsrpBuffer(:),   SimCtrl.fp16AlgoSel));
    hdf5_write_nv_exp(h5File, 'pucchF234_refRssiBuffer'           , fp16nv(pucchF234_refRssiBuffer(:),   SimCtrl.fp16AlgoSel));
    hdf5_write_nv_exp(h5File, 'pucchF234_refInterfBuffer'         , fp16nv(pucchF234_refInterfBuffer(:), SimCtrl.fp16AlgoSel));
    hdf5_write_nv_exp(h5File, 'pucchF234_refTaBuffer'             , fp16nv(pucchF234_refTaBuffer(:), SimCtrl.fp16AlgoSel));
    hdf5_write_nv_exp(h5File, 'pucchF234_refHarqDetStatBuffer'    , uint8(pucchF234_refHarqDetStatBuffer(:)));
    hdf5_write_nv_exp(h5File, 'pucchF234_refCsiPart1DetStatBuffer', uint8(pucchF234_refCsiPart1DetStatBuffer(:)));
    hdf5_write_nv_exp(h5File, 'pucchF234_refCsiPart2DetStatBuffer', uint8(pucchF234_refCsiPart2DetStatBuffer(:)));
    
    Xtf = reshape(fp16nv(real(Xtf), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(Xtf), SimCtrl.fp16AlgoSel), [size(Xtf)]);
    hdf5_write_nv(h5File, 'DataRx', complex(single(Xtf)));
    % dump Tx X_tf
    
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

    % --- Inline polar-decoder reference outputs (consumed by
    % PucchRx::loadPolarRefForSkip when running with --skip-polar).
    % Schema matches the standalone polar TV (sizes / polarUciSegPrms /
    % cbEst{N} / crcErrorFlags) so existing UciPolarDataset code reads
    % these unchanged. Dataset names don't collide with any other field
    % currently written to the main PUCCH TV.
    nPolUciSegs = length(polarBuffers);
    if nPolUciSegs > 0
        polarUciSegPrms = [];
        nCbs            = 0;
        crcErrorFlags   = [];
        for segIdx = 0 : (nPolUciSegs - 1)
            prms = polarBuffers{segIdx + 1}.polarUciSegPrms;
            polarUciSegPrms(segIdx + 1).nCbs           = uint8(prms.nCbs);
            polarUciSegPrms(segIdx + 1).K_cw           = uint16(prms.K_cw);
            polarUciSegPrms(segIdx + 1).E_cw           = uint32(prms.E_cw);
            polarUciSegPrms(segIdx + 1).zeroInsertFlag = uint8(prms.zeroInsertFlag);
            polarUciSegPrms(segIdx + 1).n_cw           = uint8(prms.n_cw);
            polarUciSegPrms(segIdx + 1).N_cw           = uint16(prms.N_cw);
            polarUciSegPrms(segIdx + 1).E_seg          = uint32(prms.E_seg);
            polarUciSegPrms(segIdx + 1).nCrcBits       = uint8(prms.nCrcBits);
        end
        hdf5_write_nv_exp(h5File, 'polarUciSegPrms', polarUciSegPrms);

        for segIdx = 0 : (nPolUciSegs - 1)
            prms = polarUciSegPrms(segIdx + 1);
            for i = 1 : prms.nCbs
                cbEst_bits  = polarBuffers{segIdx + 1}.cbEsts(:, i);
                nBitsCb     = length(cbEst_bits);
                nWordsCb    = ceil(nBitsCb / 32);
                cbEst_words = zeros(nWordsCb, 1);
                for w = 0 : (nWordsCb - 1)
                    for j = 0 : 31
                        b = w*32 + j;
                        if b >= nBitsCb
                            break;
                        end
                        cbEst_words(w + 1) = cbEst_words(w + 1) + cbEst_bits(b + 1) * 2^j;
                    end
                end
                hdf5_write_nv(h5File, ['cbEst', num2str(nCbs)], uint32(cbEst_words));
                crcErrorFlags = [crcErrorFlags; polarBuffers{segIdx + 1}.cbCrcErrorFlags(i)];
                nCbs          = nCbs + 1;
            end
        end
        hdf5_write_nv(h5File, 'crcErrorFlags', uint8(crcErrorFlags));

        sizes = [];
        sizes.nPolCws     = uint16(nCbs);
        sizes.nPolUciSegs = uint16(nPolUciSegs);
        hdf5_write_nv_exp(h5File, 'sizes', sizes);
    end

    % --- Per-UCI front-end LLR references (consumed by
    % PucchRx::compareFrontEndRefForBackendSkip in PUCCH_PIPELINE_SKIP_BACKEND runs).
    % Schema: frontEndSizes{nF2Ucis, nF3Ucis} + f{2,3}FrontEndLLRs{N}
    % (CUPHY_R_16F, length E_seg1, raw MATLAB LLRs written via fp16nv).
    nF2FE = 0;
    nF3FE = 0;
    if isstruct(frontEndBuffers)
        if isfield(frontEndBuffers, 'f2')
            nF2FE = length(frontEndBuffers.f2);
        end
        if isfield(frontEndBuffers, 'f3')
            nF3FE = length(frontEndBuffers.f3);
        end
    end
    % Sanity: the aggregator above appends one cell per F2/F3 UCI in
    % PucchDataOut. If a UCI is missing descrmLLR / descrmLLRSeq1 it
    % appends []. Reject those cases now rather than silently writing
    % malformed refs that fail later in compareFrontEndRefForBackendSkip.
    nF2Ucis = 0;
    if isfield(PucchDataOut, 'pUciF2')
        nF2Ucis = length(PucchDataOut.pUciF2);
    end
    nF3Ucis = 0;
    if isfield(PucchDataOut, 'pUciF3')
        nF3Ucis = length(PucchDataOut.pUciF3);
    end
    if (nF2FE ~= nF2Ucis) || (nF3FE ~= nF3Ucis)
        error(['saveTV_pucch_cuphy: front-end ref count mismatch: expected ', ...
               num2str(nF2Ucis), ' F2 + ', num2str(nF3Ucis), ' F3, got ', ...
               num2str(nF2FE), ' + ', num2str(nF3FE), '.']);
    end

    frontEndSizes = [];
    frontEndSizes.nF2Ucis = uint16(nF2FE);
    frontEndSizes.nF3Ucis = uint16(nF3FE);
    hdf5_write_nv_exp(h5File, 'frontEndSizes', frontEndSizes);

    for uciIdx = 0 : (nF2FE - 1)
        llrs = frontEndBuffers.f2{uciIdx + 1};
        llrs = llrs(:);
        if isempty(llrs)
            error(['saveTV_pucch_cuphy: missing F2 front-end LLR ref for UCI ', num2str(uciIdx), '.']);
        end
        hdf5_write_nv_exp(h5File, ['f2FrontEndLLRs', num2str(uciIdx)], fp16nv(llrs, SimCtrl.fp16AlgoSel));
    end
    for uciIdx = 0 : (nF3FE - 1)
        llrs = frontEndBuffers.f3{uciIdx + 1};
        llrs = llrs(:);
        if isempty(llrs)
            error(['saveTV_pucch_cuphy: missing F3 front-end LLR ref for UCI ', num2str(uciIdx), '.']);
        end
        hdf5_write_nv_exp(h5File, ['f3FrontEndLLRs', num2str(uciIdx)], fp16nv(llrs, SimCtrl.fp16AlgoSel));
    end
end
