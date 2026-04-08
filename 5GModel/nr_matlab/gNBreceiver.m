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

function gNB = gNBreceiver(gNB, rxSamp, rxSamp_prach, rxSamp_noNoise, rxSamp_prach_noNoise)
% function gNB = gNBreceiver(gNB, rxSamp, rxSamp_prach, rxSamp_noNoise, rxSamp_prach_noNoise)
%
% This function simulates gNB functionality
%
% Input:    gNB: structure for a single UE
%           rxSamp: received samples for non-PRACH channels
%           rxSamp_prach: received samples for PRACH channel
%           rxSamp_noNoise: received samples for non-PRACH channels without noise
%           rxSamp_prach_noNoise: received samples for PRACH channel without noise
%
% Output:   gNB: structure for a single gNB
%

% Now we generate time domain samples slot by slot, which significantly
% increases the process latency and memeoy size.  We will need to generate
% time domain samples symbol by symbol in later version.
%

Mac = gNB.Mac;
Phy = gNB.Phy;

[FAPIpdu, Mac] = gNbUlMacSendPduToPhy(Mac);

global SimCtrl
if SimCtrl.timeDomainSim
    prachSlotInfo = Mac.Config.prachSlotInfo;
    [rxSamp_noNoise, rxSamp_prach_noNoise] = gNbUlGenFreqDomainSig(Phy, rxSamp_noNoise, prachSlotInfo);
    [rxSamp, rxSamp_prach, Phy] = gNbUlGenFreqDomainSig(Phy, rxSamp, prachSlotInfo);
end

[Mac, Phy, FAPIpdu] = gNbUlPhyDetSig(Mac, Phy, FAPIpdu, rxSamp, rxSamp_prach, rxSamp_noNoise, rxSamp_prach_noNoise);

gNB.Mac = Mac;
gNB.FAPIpdu = FAPIpdu;
gNB.Phy = Phy;

return;


function [FAPIpdu, Mac] = gNbUlMacSendPduToPhy(Mac)

carrier = Mac.Config.carrier;
global SimCtrl
if SimCtrl.genTV.forceSlotIdxFlag
    carrier.idxSlotInFrame = SimCtrl.genTV.slotIdx(1);
end
NsymPerSlot = carrier.N_symb_slot;
N_slot_frame = carrier.N_slot_frame_mu;
N_slot_subframe = carrier.N_slot_subframe_mu;
idxSlot = carrier.idxSlot;
idxSubframe = carrier.idxSubframe;
idxFrame = carrier.idxFrame;
idxSlotInFrame = mod(carrier.idxSlotInFrame, carrier.N_slot_frame_mu);
idxPdu = 1;
prachIdx = 1;
pucchIdx = 1;
puschIdx = 1;
srsIdx = 1;

FAPIpdu = [];
Mac.Config.prachSlotInfo.isPrachSlotInFrame = 0;
Mac.Config.prachSlotInfo.isLastPrachSlot = 0;
alloc = Mac.rx.alloc;
N_alloc = length(alloc);

lastPuschIdx = 0;
lastPucchIdx = 0;
lastSrsIdx = 0;
lastPrachIdx = 0;
for idxAlloc = 1:N_alloc
    thisAlloc = alloc{idxAlloc};
    allocType = thisAlloc.type;
    if strcmp(allocType, 'pusch')
        lastPuschIdx = idxAlloc;
    elseif strcmp(allocType, 'pucch')
        lastPucchIdx = idxAlloc;
    elseif strcmp(allocType, 'srs')
        lastSrsIdx = idxAlloc;
    elseif strcmp(allocType, 'prach')
        lastPrachIdx = idxAlloc;
    end
end

for idxAlloc = 1:N_alloc
    thisAlloc = alloc{idxAlloc};
    allocType = thisAlloc.type;
    switch allocType
        case 'prach'
            prach = Mac.Config.prach;
            carrier = Mac.Config.carrier;
            
            raSubframeNum = prach{prachIdx}.subframeNum;
            raSlotNum = prach{prachIdx}.n_slot_RA_sel;
            isRaSlot = ismember(idxSlot, raSlotNum);
            slotSpan = findPrachSlotSpan(prach{prachIdx}.preambleFormat, carrier.mu);
            isLastRaSlot = ismember(idxSlotInFrame, raSlotNum + raSubframeNum * N_slot_subframe + [slotSpan-1]);
            isRaSlotInFrame = ismember(idxSlotInFrame, raSlotNum + (raSubframeNum' * N_slot_subframe + [0:slotSpan-1]));

            Mac.Config.prachSlotInfo.isPrachSlotInFrame = isRaSlotInFrame;
            Mac.Config.prachSlotInfo.isLastPrachSlot = isLastRaSlot;
            
            if prach{prachIdx}.allSubframes
                if (slotSpan == 1)
                    if isRaSlot
                        Mac.Config.prachSlotInfo.isPrachSlotInFrame = 1;
                        Mac.Config.prachSlotInfo.isLastPrachSlot = 1;
                    else
                        Mac.Config.prachSlotInfo.isPrachSlotInFrame = 0;
                        Mac.Config.prachSlotInfo.isLastPrachSlot = 0;
                    end
                elseif (slotSpan == 2)
                    if isRaSlot
                        Mac.Config.prachSlotInfo.isPrachSlotInFrame = 1;
                        Mac.Config.prachSlotInfo.isLastPrachSlot = 0;
                    else
                        Mac.Config.prachSlotInfo.isPrachSlotInFrame = 1;
                        Mac.Config.prachSlotInfo.isLastPrachSlot = 1;
                    end
                elseif (slotSpan > 2)
                    error('prach.allSubframes = 1 is not supported for this config ... \n');
                end
            end
            if SimCtrl.genTV.forceSlotIdxFlag
                Mac.Config.prachSlotInfo.isPrachSlotInFrame = 1;
                Mac.Config.prachSlotInfo.isLastPrachSlot = 1;
            end

            if (Mac.Config.prachSlotInfo.isLastPrachSlot)
                [pdu, Mac] = gNbUlGenMac2PhyPdu(Mac, 'prach', prachIdx);
                pdu.prachPduIdx = prachIdx;
                prachIdx = prachIdx + 1;
                if idxAlloc == lastPrachIdx
                    pdu.lastPrachPdu = 1;
                else
                    pdu.lastPrachPdu = 0;
                end
                FAPIpdu{idxPdu} = pdu;
                idxPdu = idxPdu + 1;
            end            
        case 'pucch'
            [pdu, Mac] = gNbUlGenMac2PhyPdu(Mac, 'pucch', pucchIdx);
            pdu.pucchPduIdx = pucchIdx;
            pucchIdx = pucchIdx + 1;
            if idxAlloc == lastPucchIdx
                pdu.lastPucchPdu = 1;
            else
                pdu.lastPucchPdu = 0;
            end
            FAPIpdu{idxPdu} = pdu;
            idxPdu = idxPdu + 1;
        case 'pusch'
            [pdu, Mac] = gNbUlGenMac2PhyPdu(Mac, 'pusch', puschIdx);
            pdu.puschPduIdx = puschIdx;
            puschIdx = puschIdx + 1;
            if idxAlloc == lastPuschIdx
                pdu.lastPuschPdu = 1;
            else
                pdu.lastPuschPdu = 0;
            end
            FAPIpdu{idxPdu} = pdu;
            idxPdu = idxPdu + 1;
        case 'srs'
            srs = Mac.Config.srs;
            if (mod(N_slot_frame * idxFrame + idxSlotInFrame - srs{1}.Toffset, srs{1}.Tsrs) == 0)
                is_SRS_slot  = 1;
            else
                is_SRS_slot = 0;
            end
            if (srs{1}.resourceType > 0 && is_SRS_slot) || (srs{1}.resourceType == 0)
                [pdu, Mac] = gNbUlGenMac2PhyPdu(Mac, 'srs', srsIdx);
                pdu.srsPduIdx = srsIdx;
                srsIdx = srsIdx + 1;
                if idxAlloc == lastSrsIdx
                    pdu.lastSrsPdu = 1;
                else
                    pdu.lastSrsPdu = 0;
                end
                FAPIpdu{idxPdu} = pdu;
                idxPdu = idxPdu + 1;
            end
        otherwise

    end
end

if SimCtrl.enableUlDlCoSim == 0
    idxSlot = idxSlot + 1;
end

carrier = updateTimeIndex(carrier, idxSlot, SimCtrl.genTV.forceSlotIdxFlag);

Mac.Config.carrier = carrier;

return


function [pdu, Mac] = gNbUlGenMac2PhyPdu(Mac, pduType, allocIdx)

carrier = Mac.Config.carrier;
global SimCtrl
if SimCtrl.genTV.forceSlotIdxFlag
    carrier.idxSlotInFrame = SimCtrl.genTV.slotIdx(1);
end
table = Mac.Config.table;

idxSlotInFrame = carrier.idxSlotInFrame;

switch pduType
    case 'prach'
        prach = Mac.Config.prach{allocIdx};
        pdu.type = 'prach';
        pdu.physCellID = carrier.N_ID_CELL;
        pdu.NumPrachOcas = prach.N_t_RA_slot;
        pdu.prachFormat = convertPrachFormatToSCF(prach.preambleFormat);
        % force NumPrachOcas to 1 for long preambles to meet the range 
        % requirement (1->7) in FAPI PRACH PDU
        if pdu.prachFormat <= 3 
            pdu.NumPrachOcas = 1; 
        end
        pdu.numRa = allocIdx - 1;
        pdu.prachStartSymbol = prach.startingSym;
        pdu.numCs = prach.N_CS;
        % RX precoding and beaforming PDU SCF-FAPI table 3-53
        pdu.numPRGs = 1; % only support 1
        if pdu.prachFormat <= 3 % long preamble
            pdu.prgSize = 70; % 839/12
        else % short preamble
            pdu.prgSize = 12; % 139/12
        end
        pdu.digBFInterfaces = prach.digBFInterfaces;
        pdu.beamIdx = prach.beamIdx;

%         pdu.prmbIdx = []; % prach.prmbIdx;
    case 'pucch'
        pucch = Mac.Config.pucch{allocIdx};
        pdu.type = 'pucch';
        pdu.RNTI = pucch.RNTI;
        pdu.BWPSize = pucch.BWPSize;
        pdu.BWPStart = pucch.BWPStart;
        pdu.SubcarrierSpacing = carrier.mu;
        pdu.CyclicPrefix = carrier.CpType;
        pdu.FormatType = pucch.FormatType;
        pdu.multiSlotTxIndicator = pucch.multiSlotTxIndicator;
        pdu.pi2Bpsk = pucch.pi2Bpsk;
        pdu.prbStart = pucch.prbStart;
        pdu.prbSize = pucch.prbSize;
        pdu.StartSymbolIndex = pucch.startSym;
        pdu.NrOfSymbols = pucch.nSym;
        pdu.freqHopFlag = pucch.freqHopFlag;
        pdu.secondHopPRB = pucch.secondHopPRB;
        pdu.groupHopFlag = pucch.groupHopFlag;
        pdu.sequenceHopFlag = pucch.sequenceHopFlag;
        pdu.hoppingId = pucch.hoppingId;
        pdu.InitialCyclicShift = pucch.cs0;
        pdu.dataScramblingId = pucch.dataScramblingId;
        pdu.TimeDomainOccIdx = pucch.tOCCidx;
        pdu.PreDftOccIdx = pucch.PreDftOccIdx;
        pdu.PreDftOccLen = pucch.PreDftOccLen;
        pdu.AddDmrsFlag = pucch.AddDmrsFlag;
        pdu.DmrsScramblingId = pucch.DmrsScramblingId;
        pdu.DMRScyclicshift = pucch.DMRScyclicshift;
        pdu.maxCodeRate = pucch.maxCodeRate;
        pdu.BitLenSr = pucch.BitLenSr;
        pdu.BitLenHarq = pucch.BitLenHarq;

        switch pdu.FormatType
            case {0, 1}
                pdu.SRFlag = pucch.SRFlag;
            case {2, 3}
                pdu.maxCodeRate    = pucch.maxCodeRate;
                pdu.BitLenCsiPart1 = pucch.BitLenCsiPart1;
                pdu.rankBitOffset  = pucch.rankBitOffset;
                pdu.rankBitSize    = pucch.rankBitSize;
                pdu.UciP1ToP2Crpd.numPart2s = pucch.UciP1ToP2Crpd.numPart2s;
            otherwise
                error('PUCCH format is not supported ...\n');
        end

        % RX precoding and beaforming PDU SCF-FAPI table 3-53
        pdu.numPRGs = 1; % only support 1
        pdu.prgSize = pdu.prbSize;
        pdu.digBFInterfaces = pucch.digBFInterfaces;
        pdu.beamIdx = pucch.digBFInterfaces;
        if(isfield(pucch,'beamIdx'))
            pdu.beamIdx = pucch.beamIdx;
        end
        pdu.idxUE = pucch.idxUE;
        pdu.DTXthreshold = pucch.DTXthreshold;
    case 'pusch'
        pusch = Mac.Config.pusch{allocIdx};
        pdu.type = 'pusch';
        pdu.pduBitmap = pusch.pduBitmap;
        pdu.RNTI = pusch.RNTI;
        pdu.pduIndex = 0;
        pdu.BWPSize = pusch.BWPSize;
        pdu.BWPStart = pusch.BWPStart;
        pdu.SubcarrierSpacing = carrier.mu;
        pdu.CyclicPrefix = carrier.CpType;
        pdu.NrOfCodewords = pusch.NrOfCodewords;
        puschTable = table;
        mcs = pusch.mcsIndex;
        mcsTable = pusch.mcsTable;
        switch(mcsTable)
            case 0
                mcs_table = puschTable.McsTable1;
            case 1
                mcs_table = puschTable.McsTable2;
            case 2
                mcs_table = puschTable.McsTable3;
            case 3
                mcs_table = puschTable.McsTable4;
            case 4
                mcs_table = puschTable.McsTable5;
        end
        if mcs == 100
            qam = 2;
            codeRate = 1024;
        elseif (ismember(mcsTable, [1, 3, 4]) && mcs > 27) || ...
                (ismember(mcsTable, [0, 2]) && mcs > 28)
            qam = pusch.qamModOrder;
            codeRate = pusch.targetCodeRate/10;
        else
            if pusch.pi2BPSK && ((mcsTable == 3 && mcs <= 1) || ...
                    (mcsTable == 4 && mcs <= 5))
                factor = 2;
            else
                factor = 1;
            end
            qam = mcs_table(mcs+1, 2)/factor;
            codeRate = mcs_table(mcs+1, 3)*factor;
        end
        pdu.targetCodeRate = codeRate*10; % per SCF-FAPI spec
        codeRate = codeRate/1024;
        pdu.qamModOrder = qam;
        pdu.mcsIndex = mcs;
        pdu.mcsTable = mcsTable;
        pdu.TransformPrecoding = pusch.TransformPrecoding;
        DmrsSymbPos = pusch.DmrsSymbPos;
        nDmrsSymb = sum(DmrsSymbPos(pusch.StartSymbolIndex + 1:...
            pusch.StartSymbolIndex + pusch.NrOfSymbols));
        % half of REs on DMRS symbols can be used for data
        isDataPresent = bitand(uint16(pdu.pduBitmap),uint16(2^0));
        if pusch.numDmrsCdmGrpsNoData == 1 && isDataPresent
            nDmrsSymb = nDmrsSymb/2;
        end
        nDataSymb = pusch.NrOfSymbols - nDmrsSymb;
        Ninfo = pusch.rbSize*min(156,12*nDataSymb)*pusch.nrOfLayers*codeRate*qam;
        TBS_table = puschTable.TBS_table;
        if Ninfo <= 3824
            %for "small" sizes, look up TBS in a table. First round the
            %number of information bits.
            n = max(3,(floor(log2(Ninfo)) - 6));
            Ninfo_prime = max(24, 2^n*floor(Ninfo / 2^n));

            %next lookup in table closest TBS (without going over).
            compare = Ninfo_prime - TBS_table;
            compare(compare > 0) = -100000;
            [~,max_index] = max(compare);
            TBS = TBS_table(max_index);
            C = 1;
        else
            %for "large" sizes, compute TBS. First round the number of
            %information bits to a power of two.
             n = floor(log2(Ninfo-24)) - 5;
             Ninfo_prime = max(3840, 2^n*round((Ninfo-24)/2^n));

            %Next, compute the number of code words. For large code rates,
            %use base-graph 1. For small code rate use base-graph 2.
            if codeRate < 1/4
                C = ceil( (Ninfo + 24) / 3816);
                TBS = 8*C*ceil( (Ninfo_prime + 24) / (8*C) ) - 24;
            else
                if Ninfo_prime > 8424
                    C = ceil( (Ninfo_prime + 24) / 8424);
                    TBS = 8*C*ceil( (Ninfo_prime + 24) / (8*C) ) - 24;
                else
                    C = 1;
                    TBS = 8*C*ceil( (Ninfo_prime + 24) / (8*C) ) - 24;
                end
            end
        end
        if isDataPresent
            pdu.TBSize = ceil(TBS/8);
        else
            pdu.TBSize = 0;
        end
        if (ismember(mcsTable, [1, 3, 4]) && mcs > 27) || ...
                (ismember(mcsTable, [0, 2]) && mcs > 28)
            pdu.TBSize = pusch.TBSize;
        end
        if 0, % HARQ TC
            retx.targetCodeRate = pdu.targetCodeRate;
            retx.qamModOrder = qam;
            retx.TBSize = pdu.TBSize
        end
        pdu.dataScramblingId = pusch.dataScramblingId;
        pdu.nrOfLayers = pusch.nrOfLayers;
        pdu.DmrsSymbPos = pusch.DmrsSymbPos;
        pdu.DmrsMappingType = pusch.DmrsMappingType; % not defined in FAPI PDU, for compliance test only
        pdu.dmrsConfigType = pusch.dmrsConfigType;
        pdu.DmrsScramblingId = pusch.DmrsScramblingId;
        pdu.puschIdentity = pusch.puschIdentity;
        pdu.groupOrSequenceHopping = pusch.groupOrSequenceHopping;
        pdu.SCID = pusch.SCID;
        pdu.numDmrsCdmGrpsNoData = pusch.numDmrsCdmGrpsNoData;
        dmrsPorts = zeros(1, 16);
        portIdx = pusch.portIdx;
        nL = pdu.nrOfLayers;
        if length(portIdx) > nL
            portIdx = portIdx(1:nL);
        elseif length(portIdx) < nL
            portIdx = [portIdx, portIdx(end)+[1:(nL-length(portIdx))]];
        end
        dmrsPorts(portIdx+1) = 1;
        % flip based on SCF FAPI spec. (first port -> LSB)
        pdu.dmrsPorts = flip(dmrsPorts);
        pdu.resourceAlloc = pusch.resourceAlloc;
%         pdu.rbBitmap = pusch.rbBitmap;
        pdu.rbStart = pusch.rbStart;
        pdu.rbSize = pusch.rbSize;
        pdu.VRBtoPRBMapping = pusch.VRBtoPRBMapping;
        pdu.FrequencyHopping = pusch.FrequencyHopping;
        pdu.txDirectCurrentLocation = pusch.txDirectCurrentLocation;
        pdu.uplinkFrequencyShift7p5khz = pusch.uplinkFrequencyShift7p5khz;
        pdu.StartSymbolIndex = pusch.StartSymbolIndex;
        pdu.NrOfSymbols = pusch.NrOfSymbols;
        pdu.prcdBf = 0; % not used
        pdu.rvIndex = pusch.rvIndex;
        pdu.harqProcessID = pusch.harqProcessID;
        pdu.newDataIndicator = pusch.newDataIndicator;
        pdu.numCb = 0; % not used
        pdu.cbPresentAndPosition = 0; % not used
        % for LBRM
        pdu.I_LBRM = pusch.I_LBRM;
        pdu.maxLayers = pusch.maxLayers;
        if mcsTable == 1
            pdu.maxQm = 8;
        else
            pdu.maxQm = 6;
        end
        pdu.n_PRB_LBRM = pusch.n_PRB_LBRM;
        pdu.idxUeg = pusch.idxUeg;
        pdu.idxUE = pusch.idxUE;

        %for UCI on PUSCH
        pdu.alphaScaling = pusch.alphaScaling;
        pdu.betaOffsetHarqAck = pusch.betaOffsetHarqAck;
        pdu.betaOffsetCsi1 = pusch.betaOffsetCsi1;
        pdu.betaOffsetCsi2 = pusch.betaOffsetCsi2;
        pdu.harqAckBitLength = pusch.harqAckBitLength;
        pdu.csiPart1BitLength = pusch.csiPart1BitLength;

        pdu.nCsi2Reports            = pusch.nCsi2Reports;
        pdu.calcCsi2Size_csi2MapIdx = pusch.calcCsi2Size_csi2MapIdx;
        pdu.calcCsi2Size_nPart1Prms = pusch.calcCsi2Size_nPart1Prms;
        pdu.calcCsi2Size_prmOffsets = pusch.calcCsi2Size_prmOffsets;
        pdu.calcCsi2Size_prmSizes   = pusch.calcCsi2Size_prmSizes;
        pdu.flagCsiPart2            = pusch.flagCsiPart2;
        pdu.rankBitOffset           = pusch.rankBitOffset;
        pdu.rankBitSize             = pusch.rankBitSize;
        pdu.flagCsiPart2            = pusch.flagCsiPart2;

        % check if csi2 present:
        enable_multi_csiP2_fapiv3 = SimCtrl.enable_multi_csiP2_fapiv3;
        if enable_multi_csiP2_fapiv3
            csiP2exist = (pusch.flagCsiPart2 > 0);
        else
            csiP2exist = bitand(uint16(pusch.pduBitmap),uint16(2^5));
        end

        if (csiP2exist > 0)
            pdu.csiPart2BitLength = 2^8*pusch.rankBitSize(1) + pusch.rankBitOffset(1);
        end
    
        if SimCtrl.forceCsiPart2Length
            pdu.csiPart2BitLength = SimCtrl.forceCsiPart2Length;
        end
        pdu.DTXthreshold = pusch.DTXthreshold;

        % RX precoding and beamforming PDU SCF-FAPI v10.02 table 3-53
        pdu.numPRGs         = 1;
        pdu.prgSize         = pusch.rbSize;
        if(isfield(pusch,'numPRGs'))
            pdu.numPRGs         = pusch.numPRGs;
        end
        if(isfield(pusch,'prgSize'))
            pdu.prgSize         = pusch.prgSize;
        end
        if(isfield(pusch,'enable_prg_chest'))
            pdu.enable_prg_chest = pusch.enable_prg_chest;
        else
            pdu.enable_prg_chest = 0;
        end

        % static beamforming
        if SimCtrl.enable_static_dynamic_beamforming && pusch.digBFInterfaces
            pdu.numPRGs         = 1;
            pdu.prgSize         = pusch.rbSize;
        end

        pdu.digBFInterfaces = pusch.digBFInterfaces;
        pdu.beamIdx = pusch.beamIdx;
        
        if(isfield(pusch,'foForgetCoeff'))
            pdu.foForgetCoeff = pusch.foForgetCoeff;
        else
            pdu.foForgetCoeff = 0.0;
        end
        if(isfield(pusch,'foCompensationBuffer'))
            pdu.foCompensationBuffer = pusch.foCompensationBuffer;
        else
            pdu.foCompensationBuffer = 1.0;
        end

        if(isfield(pusch,'ldpcEarlyTerminationPerUe'))
            pdu.ldpcEarlyTerminationPerUe = pusch.ldpcEarlyTerminationPerUe;
        else
            pdu.ldpcEarlyTerminationPerUe = 0;
        end

        if(isfield(pusch,'ldpcMaxNumItrPerUe'))
            pdu.ldpcMaxNumItrPerUe = pusch.ldpcMaxNumItrPerUe;
        else
            pdu.ldpcMaxNumItrPerUe = 10;
        end

    case 'srs'
        srs = Mac.Config.srs{allocIdx};
        pdu.type = 'srs';
        pdu.RNTI = srs.RNTI;
        if isfield(srs,'srsChestBufferIndex')
            pdu.srsChestBufferIndex = srs.srsChestBufferIndex;
        else
            pdu.srsChestBufferIndex = srs.RNTI-1;
        end
        pdu.BWPSize = srs.BWPSize;
        pdu.BWPStart = srs.BWPStart;
        switch srs.numAntPorts
            case 1
                pdu.numAntPorts = 0;
            case 2
                pdu.numAntPorts = 1;
            case 4
                pdu.numAntPorts = 2;
            otherwise
                error('numAntPorts is not supported ...\n');
        end
        switch srs.numSymbols
            case 1
                pdu.numSymbols = 0;
            case 2
                pdu.numSymbols = 1;
            case 4
                pdu.numSymbols = 2;
            otherwise
                error('numSymbols is not supported ...\n');
        end

        switch srs.numRepetitions
            case 1
                pdu.numRepetitions = 0;
            case 2
                pdu.numRepetitions = 1;
            case 4
                pdu.numRepetitions = 2;
            otherwise
                error('numRepetitions is not supported ...\n');
        end
        switch srs.combSize
            case 2
                pdu.combSize = 0;
            case 4
                pdu.combSize = 1;
            otherwise
                error('combSize is not supported ...\n');
        end
        pdu.timeStartPosition = srs.timeStartPosition;
        pdu.configIndex = srs.configIndex;
        pdu.sequenceId = srs.sequenceId;
        pdu.bandwidthIndex = srs.bandwidthIndex;
        pdu.combOffset = srs.combOffset;
        pdu.cyclicShift = srs.cyclicShift;
        pdu.frequencyPosition = srs.frequencyPosition;
        pdu.frequencyShift = srs.frequencyShift;
        pdu.frequencyHopping = srs.frequencyHopping;
        pdu.groupOrSequenceHopping = srs.groupOrSequenceHopping;
        pdu.resourceType = srs.resourceType;
        pdu.Tsrs = srs.Tsrs;
        pdu.Toffset = srs.Toffset;
        pdu.Beamforming = srs.Beamforming;

        % RX precoding and beaforming PDU SCF-FAPI table 3-53
        C_SRS = pdu.configIndex;
        B_SRS = pdu.bandwidthIndex;
        srs_BW_table = table.srs_BW_table;
        m_SRS_b = srs_BW_table(C_SRS+1,2*B_SRS+1);
        nPRB = m_SRS_b/srs.combSize;
        pdu.prgSize = srs.prgSize;
        pdu.numPRGs = m_SRS_b/srs.prgSize;
        pdu.digBFInterfaces = srs.digBFInterfaces;
        pdu.beamIdx = srs.beamIdx;
        
        % FAPIv4:
        pdu.numTotalUeAntennas = srs.numAntPorts;
        pdu.ueAntennasInThisSrsResourceSet = 0;
        for antIdx = 0 : (srs.numAntPorts - 1)
            pdu.ueAntennasInThisSrsResourceSet = pdu.ueAntennasInThisSrsResourceSet + 2^antIdx;
        end
        pdu.sampledUeAntennas = pdu.ueAntennasInThisSrsResourceSet;
        pdu.usage = srs.usage;
       

    otherwise
        pdu = [];
end

return


function [Mac, Phy, FAPIpdu] = gNbUlPhyDetSig(Mac, Phy, FAPIpdu, rxSamp, rxSamp_prach, rxSamp_noNoise, rxSamp_prach_noNoise)

global SimCtrl
carrier = Phy.Config.carrier;
Chan_UL = Phy.Chan_UL;
if SimCtrl.N_Interfering_UE_UL>0
    interfChan_UL = Phy.interfChan_UL;
else
    interfChan_UL = [];
end

if SimCtrl.genTV.forceSlotIdxFlag
    carrier.idxSlotInFrame = SimCtrl.genTV.slotIdx(1);
end
NsymPerSlot = carrier.N_symb_slot;
idxSlot = carrier.idxSlot;
idxSubframe = carrier.idxSubframe;
idxFrame = carrier.idxFrame;
idxSlotInFrame = carrier.idxSlotInFrame;
Nsubcarrier = carrier.N_grid_size_mu*carrier.N_sc_RB;
Nant = carrier.numRxPort;
table = Phy.Config.table;

% Note on rxSamp, Xtf* and Xtf*_uncomp:
% rxSamp: original received samples in double precision, will be modified by FP16 quantization and assigned to Xtf* after certain processing
% Xtf*: will be modified by fpCompDecomp and FP16 quantization again before baseband processing. Should be the same data for baseband processing in 5GModel and cuPHY
% Xtf*_uncomp: Xtf before bfpCompDecomp, will be used when saving FAPI TV

% Xtf is used for PUSCH/PUCCH detection
% Xtf_srs is used for SRS detection
% Xtf_prach is used for PRACH detection


rxSamp = reshape(fp16nv(real(rxSamp), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(rxSamp), SimCtrl.fp16AlgoSel), [size(rxSamp)]);
rxSamp_noNoise = reshape(fp16nv(real(rxSamp_noNoise), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(rxSamp_noNoise), SimCtrl.fp16AlgoSel), [size(rxSamp_noNoise)]);

global  fp_flag; 
switch SimCtrl.fp_flag_global 
    case 0
        fp_flag = 'fp64ieee754';
    case 1
        fp_flag = 'fp32ieee754';    
    case 2
        fp_flag = 'fp16ieee754';
    case 3
        fp_flag = 'fp16cleve';
    case 4
        fp_flag = 'fp16nv';
end

if SimCtrl.forceRxZero
    rxSamp = rxSamp * 0;
    rxSamp_noNoise = rxSamp_noNoise * 0;
end

Xtf = double(rxSamp);
Xtf_noNoise = double(rxSamp_noNoise);

if SimCtrl.capSamp.enable && SimCtrl.genTV.slotIdx == idxSlotInFrame
    [Nf,Nt,Na] = size(Xtf);
    iqWidth = 16;
    if (SimCtrl.capSamp.isComp == 1)
        iqWidth = SimCtrl.oranComp.iqWidth(1);
        FSOffset = SimCtrl.oranComp.FSOffset(1);
        Ref_c = SimCtrl.oranComp.Ref_c(1);
        Nre_max = SimCtrl.oranComp.Nre_max;
        max_amp_ul = SimCtrl.oranComp.max_amp_ul;
    end
    Xtf = loadCapSamp(SimCtrl.capSamp.fileName, 1, iqWidth, FSOffset, Ref_c, Nre_max, max_amp_ul, Nf/12, Nt, Na);
    Xtf_srs = Xtf;
end

% process SRS first in case needed for rx beamforming
Xtf_srs = Xtf;
Xtf_srs_uncomp = Xtf_srs;
Xtf_srs_noNoise = Xtf_noNoise;
srsPduList = [];
idxSrs = 1;
srsIdxVec = [];
nPdu = length(FAPIpdu);
for idxPdu = 1:nPdu
    pdu = FAPIpdu{idxPdu};
    if strcmp(pdu.type, 'srs')
        srsPduList{idxSrs} = pdu;
        srs = Phy.Config.srs;
        srsIdxVec = [srsIdxVec, idxPdu];
        Phy.Config.srs.pdu = pdu;
        if pdu.lastSrsPdu
            if (SimCtrl.BFPforCuphy ~= 16) && (SimCtrl.capSamp.enable == 0)
            [validBFP, idxBFP] = ismember(SimCtrl.BFPforCuphy, SimCtrl.oranComp.iqWidth);
                if validBFP
                    iqWidth = SimCtrl.oranComp.iqWidth(idxBFP);
                    Ref_c = SimCtrl.oranComp.Ref_c(idxBFP);
                    FSOffset = SimCtrl.oranComp.FSOffset(idxBFP);
                    Nre_max = SimCtrl.oranComp.Nre_max;
                    max_amp_ul = SimCtrl.oranComp.max_amp_ul;
                    Xtf_srs = bfpCompDecomp(Xtf_srs, iqWidth, Ref_c, FSOffset, Nre_max, max_amp_ul, 1);
                end
            end
            Xtf_srs = reshape(fp16nv(real(Xtf_srs), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(Xtf_srs), SimCtrl.fp16AlgoSel), [size(Xtf_srs)]);
            SrsOutputList = detSrs_alt(srsPduList, table, carrier, Xtf_srs);
            if SimCtrl.checkSrsHestErr
                SrsOutputList_noNoise = detSrs_alt(srsPduList, table, carrier, Xtf_srs_noNoise);
            end
            for idx = 1:length(srsIdxVec)
                srsIdx = srsIdxVec(idx);
                if SimCtrl.checkSrsHestErr
                    srsOutput_noNoise = SrsOutputList_noNoise{idx};
                    srsOutput = SrsOutputList{idx};
                    FAPIpdu{srsIdx} = srsPduList{idx};
                    estErr = abs(srsOutput.Hest - srsOutput_noNoise.Hest).^2;
                    srsOutput.hestErr = mean(estErr(:))/mean(abs(srsOutput_noNoise.Hest(:)));
                else
                    srsOutput = SrsOutputList{idx};
                    FAPIpdu{srsIdx} = srsPduList{idx};
                    srsOutput.hestErr = -100;
                end
                FAPIpdu{srsIdx}.srsOutput = srsOutput;
                if SimCtrl.enableSrsState
                    srsState = Phy.srsState;
                    RNTI = FAPIpdu{srsIdx}.RNTI;
                    srsStatePresent = isfield(srsState,['rnti_',num2str(RNTI)]);
                    if (~srsStatePresent) || (srsStatePresent && srsState.(['rnti_',num2str(RNTI)]).slotCnt == SimCtrl.nSlotSrsUpdate)
                        temp = struct;
                        temp.slotCnt = 0;
                        if SimCtrl.checkSrsHestErr
                            temp.srsOutput_noNoise = srsOutput_noNoise;
                        end
                        temp.srsOutput = srsOutput;
                        srsState.(['rnti_',num2str(RNTI)]) = temp;
                    end
                    srsState.(['rnti_',num2str(RNTI)]).slotCnt = srsState.(['rnti_',num2str(RNTI)]).slotCnt + 1;
                    Phy.srsState = srsState;
                end
            end
        end
        idxSrs = idxSrs + 1;
    end
end

if SimCtrl.enable_dynamic_BF
    % simple BFW: take the signal from the first numRxPort of RX antennas
    BFW = [diag(ones(1, carrier.numRxPort)); zeros(carrier.numRxAnt - ...
        carrier.numRxPort, carrier.numRxPort)];
    
    [d1, d2, d3] = size(Xtf);
    Xtf = reshape(reshape(Xtf, d1*d2, d3) * BFW, d1, d2, carrier.numRxPort);
    Xtf_noNoise = reshape(reshape(Xtf_noNoise, d1*d2, d3) * BFW, d1, d2, carrier.numRxPort);
    
    % if SimCtrl.timeDomainSim
    %     rxSamp_prach = cell2mat(rxSamp_prach);
    % end
    % [d1, d2] = size(rxSamp_prach);
    % temp = reshape(reshape(rxSamp_prach, d1, d2) * BFW, d1, carrier.numRxPort);
    % rxSamp_prach = [];
    % if SimCtrl.timeDomainSim
    %     rxSamp_prach{1} = temp;
    % else
    %     rxSamp_prach = temp;
    % end
    % if SimCtrl.timeDomainSim
    %     rxSamp_prach_noNoise = cell2mat(rxSamp_prach_noNoise);
    % end
    % [d1, d2] = size(rxSamp_prach_noNoise);
    % temp = reshape(reshape(rxSamp_prach_noNoise, d1, d2) * BFW, d1, carrier.numRxPort);
    % rxSamp_prach_noNoise = [];
    % if SimCtrl.timeDomainSim
    %     rxSamp_prach_noNoise{1} = temp;
    % else
    %     rxSamp_prach_noNoise = temp;
    % end
end
    
if SimCtrl.timeDomainSim
    Xtf_prach = rxSamp_prach;
    Xtf_prach_noNoise = rxSamp_prach_noNoise;
else
    Xtf_prach{1} = rxSamp_prach;
    Xtf_prach_noNoise{1} = rxSamp_prach_noNoise;
end
Xtf_prach_uncomp = Xtf_prach;  % will be modified later before bfpCompDecomp for Xtf_prach 

if idxSubframe == 0 && idxSlot == 0
    Phy.tx.Xtf_frame = [];
end

if SimCtrl.enableUlRxBf  % apply rx beamforming before compression
    [Xtf, Xtf_noNoise] = ulRxBf(Xtf, Xtf_noNoise, FAPIpdu, Phy.srsState, table, Chan_UL, SimCtrl);
end

if SimCtrl.forceRxVal==1
    Xtf = (65504.0 + 65504.0*1j)*ones(size(Xtf)); % largest_normal = (2-2^(-10))*2^15 for SimCtrl.fp16AlgoSel = 2
elseif SimCtrl.forceRxVal==2
    Xtf = (-65504.0 - 65504.0*1j)*ones(size(Xtf));
elseif SimCtrl.forceRxVal==3
    Xtf = 65504.0*(randi([0, 1],size(Xtf))*2-1) + 65504.0*1j*(randi([0, 1],size(Xtf))*2-1);
elseif SimCtrl.forceRxVal==4
    Xtf = 65504.0*(rand(size(Xtf))*2-1) + 65504.0*1j*(rand(size(Xtf))*2-1);
end

Xtf_uncomp = Xtf;
if  (SimCtrl.capSamp.enable == 0)
    if (SimCtrl.FixedPointforCuphy > 0)
        iqWidth = SimCtrl.oranFixedPoint.iqWidth_UL;
        beta = SimCtrl.oranFixedPoint.beta_UL;
        Xtf = oranFL2FX2FL(Xtf, iqWidth, beta);
    elseif (SimCtrl.BFPforCuphy ~= 16)
        [validBFP, idxBFP] = ismember(SimCtrl.BFPforCuphy, SimCtrl.oranComp.iqWidth);
        if validBFP
            iqWidth = SimCtrl.oranComp.iqWidth(idxBFP);
            Ref_c = SimCtrl.oranComp.Ref_c(idxBFP);
            FSOffset = SimCtrl.oranComp.FSOffset(idxBFP);
            Nre_max = SimCtrl.oranComp.Nre_max;
            max_amp_ul = SimCtrl.oranComp.max_amp_ul;
            Xtf = bfpCompDecomp(Xtf, iqWidth, Ref_c, FSOffset, Nre_max, max_amp_ul, 1);
        end
    end
end

Xtf  = fp16nv(real(Xtf), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(Xtf), SimCtrl.fp16AlgoSel);
Xtf_noNoise = fp16nv(real(Xtf_noNoise), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(Xtf_noNoise), SimCtrl.fp16AlgoSel);

puschPduList = [];
idxPusch = 1;
puschIdxVec = [];
pucchIdxVec = [];
pucchPduList = {};
prachPduList = [];
idxPrach = 1;
prachIdxVec = [];

nPdu = length(FAPIpdu);
for idxPdu = 1:nPdu
    pdu = FAPIpdu{idxPdu};
    switch pdu.type
        case 'prach'
            prachPduList{idxPrach} = pdu;
            prachIdxVec = [prachIdxVec, idxPdu];
            prachConfigList{idxPrach} = Phy.Config.prach(idxPrach);
            Xtf_prach{idxPrach} = getPrachSamplesOran(prachConfigList{idxPrach}, Xtf_prach{idxPrach}.').';
            if SimCtrl.capSamp.enable && SimCtrl.genTV.slotIdx == idxSlotInFrame
                if strcmp(prachConfigList{idxPrach}.preambleFormat, 'B4')
                    offsetSc = (idxPrach-1)*144;
                    Xtf_prach{idxPrach} = reshape(Xtf(([1:144]+offsetSc), 1:12, :), 144*12, Nant);
                else
                    error('preambleFormat is not supported ...\n');
                end
            end
            Xtf_prach{idxPrach} = reshape(fp16nv(real(Xtf_prach{idxPrach}),SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(Xtf_prach{idxPrach}), SimCtrl.fp16AlgoSel), [size(Xtf_prach{idxPrach})]);
            Xtf_prach_uncomp{idxPrach} = Xtf_prach{idxPrach};

            if (SimCtrl.BFPforCuphy ~= 16) && (SimCtrl.capSamp.enable == 0)
                [validBFP, idxBFP] = ismember(SimCtrl.BFPforCuphy, SimCtrl.oranComp.iqWidth);
                if validBFP
                    iqWidth = SimCtrl.oranComp.iqWidth(idxBFP);
                    Ref_c = SimCtrl.oranComp.Ref_c(idxBFP);
                    FSOffset = SimCtrl.oranComp.FSOffset(idxBFP);
                    Nre_max = SimCtrl.oranComp.Nre_max;
                    max_amp_ul = SimCtrl.oranComp.max_amp_ul;
                    Xtf_prach{idxPrach} = bfpCompDecomp(Xtf_prach{idxPrach}, iqWidth, Ref_c, FSOffset, Nre_max, max_amp_ul, 1);
                end
            end
            
            if SimCtrl.forceRxZero
                Xtf_prach{idxPrach} = Xtf_prach{idxPrach} * 0;
                Xtf_prach_uncomp{idxPrach}  = Xtf_prach{idxPrach};
            end
            if pdu.lastPrachPdu
                payloadList = detPrach(prachPduList, prachConfigList, table, carrier, Xtf_prach);
                for idx = 1:length(prachIdxVec)
                    prachIdx = prachIdxVec(idx);
                    payload = payloadList(idx);
                    payload = payload{1};
                    payload.SymbolIndex = pdu.prachStartSymbol;
                    payload.SlotIndex = mod(idxSlotInFrame, carrier.N_slot_frame_mu);
                    payload.FreqIndex = idx-1;
                    if idx == length(prachIdxVec)
                        pdu.payload = payload;
                        Phy.Config.prach(idx).pdu = pdu;
                    else
                        FAPIpdu{prachIdx}.payload = payload;
                        Phy.Config.prach(idx).pdu = FAPIpdu{prachIdx};
                    end
                end
            end
            FAPIpdu{idxPdu} = pdu;
            idxPrach = idxPrach + 1;
        case 'pucch'
            pucchPduList = [pucchPduList pdu];
            pucchIdxVec = [pucchIdxVec idxPdu];
            if pdu.lastPucchPdu
                PucchDataOut = detPucch(pucchPduList, table, carrier, Xtf);

                pUciF0 = PucchDataOut.pUciF0;
                pUciF1 = PucchDataOut.pUciF1;
                pUciF2 = PucchDataOut.pUciF2;
                pUciF3 = PucchDataOut.pUciF3;

                uciOutputIdxF0 = 1;
                uciOutputIdxF1 = 1;
                uciOutputIdxF2 = 1;
                uciOutputIdxF3 = 1;

                for idx = 1:length(pucchIdxVec)
                    pucchIdx = pucchIdxVec(idx);
                    tempPdu = pucchPduList{idx};

                    if tempPdu.FormatType == 0
                        pucchPduList{idx}.taEstMicroSec = pUciF0{uciOutputIdxF0}.taEstMicroSec;
                        pucchPduList{idx}.payload = pUciF0{uciOutputIdxF0}.HarqValues;
                        pucchPduList{idx}.nBits = pUciF0{uciOutputIdxF0}.NumHarq;
                        pucchPduList{idx}.SRindication = pUciF0{uciOutputIdxF0}.SRindication;
                        pucchPduList{idx}.SRconfidenceLevel = pUciF0{uciOutputIdxF0}.SRconfidenceLevel;
                        pucchPduList{idx}.HarqconfidenceLevel = pUciF0{uciOutputIdxF0}.HarqconfidenceLevel;
                        pucchPduList{idx}.snrdB = pUciF0{uciOutputIdxF0}.SinrDB;
                        pucchPduList{idx}.RSSI = pUciF0{uciOutputIdxF0}.RSSI;
                        pucchPduList{idx}.RSRP = pUciF0{uciOutputIdxF0}.RSRP;
                        uciOutputIdxF0 = uciOutputIdxF0 + 1;
                    elseif tempPdu.FormatType == 1
                        pucchPduList{idx}.taEstMicroSec = pUciF1{uciOutputIdxF1}.taEstMicroSec;
                        pucchPduList{idx}.payload = pUciF1{uciOutputIdxF1}.HarqValues;
                        pucchPduList{idx}.nBits = pUciF1{uciOutputIdxF1}.NumHarq;
                        pucchPduList{idx}.SRindication = pUciF1{uciOutputIdxF1}.SRindication;
                        pucchPduList{idx}.SRconfidenceLevel = pUciF1{uciOutputIdxF1}.SRconfidenceLevel;
                        pucchPduList{idx}.HarqconfidenceLevel = pUciF1{uciOutputIdxF1}.HarqconfidenceLevel;
                        pucchPduList{idx}.snrdB = pUciF1{uciOutputIdxF1}.SinrDB;
                        pucchPduList{idx}.RSSI = pUciF1{uciOutputIdxF1}.RSSI;
                        pucchPduList{idx}.RSRP = pUciF1{uciOutputIdxF1}.RSRP;
                        uciOutputIdxF1 = uciOutputIdxF1 + 1;
                    elseif tempPdu.FormatType == 2
                        pucchPduList{idx}.taEstMicroSec = pUciF2{uciOutputIdxF2}.taEstMicroSec;
                        pucchPduList{idx}.payload = pUciF2{uciOutputIdxF2}.uciSeg1;
                        pucchPduList{idx}.SrValues = pUciF2{uciOutputIdxF2}.SrValues;
                        pucchPduList{idx}.NumHarq = pUciF2{uciOutputIdxF2}.NumHarq;
                        pucchPduList{idx}.HarqValues = pUciF2{uciOutputIdxF2}.HarqValues;
                        pucchPduList{idx}.CsiP1Values = pUciF2{uciOutputIdxF2}.CsiP1Values;
%                         pucchPduList{idx}.CsiP2Values = pUciF2{uciOutputIdxF2}.CsiP2Values; % missing?
                        pucchPduList{idx}.snrdB = pUciF2{uciOutputIdxF2}.SinrDB;
                        pucchPduList{idx}.RSSI = pUciF2{uciOutputIdxF2}.RSSI;
                        pucchPduList{idx}.RSRP = pUciF2{uciOutputIdxF2}.RSRP;
                        pucchPduList{idx}.InterfDB = pUciF2{uciOutputIdxF2}.InterfDB;
                        pucchPduList{idx}.DTX = pUciF2{uciOutputIdxF2}.DTX;
                        pucchPduList{idx}.HarqDetectionStatus = pUciF2{uciOutputIdxF2}.HarqDetectionStatus;
                        pucchPduList{idx}.CsiPart1DetectionStatus = pUciF2{uciOutputIdxF2}.CsiPart1DetectionStatus;
                        pucchPduList{idx}.CsiPart2DetectionStatus = pUciF2{uciOutputIdxF2}.CsiPart2DetectionStatus;
                        uciOutputIdxF2 = uciOutputIdxF2 + 1;
                    elseif tempPdu.FormatType == 3
                        pucchPduList{idx}.taEstMicroSec = pUciF3{uciOutputIdxF3}.taEstMicroSec;
                        pucchPduList{idx}.payload = pUciF3{uciOutputIdxF3}.uciSeg1;
                        pucchPduList{idx}.SrValues = pUciF3{uciOutputIdxF3}.SrValues;
                        pucchPduList{idx}.NumHarq = pUciF3{uciOutputIdxF3}.NumHarq;
                        pucchPduList{idx}.HarqValues = pUciF3{uciOutputIdxF3}.HarqValues;
                        pucchPduList{idx}.CsiP1Values = pUciF3{uciOutputIdxF3}.CsiP1Values;
                        pucchPduList{idx}.CsiP2Values = pUciF3{uciOutputIdxF3}.CsiP2Values;
                        pucchPduList{idx}.snrdB = pUciF3{uciOutputIdxF3}.SinrDB;
                        pucchPduList{idx}.RSSI = pUciF3{uciOutputIdxF3}.RSSI;
                        pucchPduList{idx}.RSRP = pUciF3{uciOutputIdxF3}.RSRP;
                        pucchPduList{idx}.InterfDB = pUciF3{uciOutputIdxF3}.InterfDB;
                        pucchPduList{idx}.DTX = pUciF3{uciOutputIdxF3}.DTX;
                        pucchPduList{idx}.HarqDetectionStatus = pUciF3{uciOutputIdxF3}.HarqDetectionStatus;
                        pucchPduList{idx}.CsiPart1DetectionStatus = pUciF3{uciOutputIdxF3}.CsiPart1DetectionStatus;
                        pucchPduList{idx}.CsiPart2DetectionStatus = pUciF3{uciOutputIdxF3}.CsiPart2DetectionStatus;
                        uciOutputIdxF3 = uciOutputIdxF3 + 1;
                    end
                    FAPIpdu{pucchIdx} = pucchPduList{idx};
                end
                Phy.Config.pucch.pdu = pucchPduList;
            end

        case 'pusch'
            puschPduList{idxPusch} = pdu;
            pusch = Phy.Config.pusch;
            harqState = pusch.harqState;
            srsState = Phy.srsState;
            puschIdxVec = [puschIdxVec, idxPdu];
            Phy.Config.pusch.pdu = pdu;
            if pdu.lastPuschPdu
                [payloadList, tbErrList, cbErrList, harqState, rxDataList] = detPusch(puschPduList, table, carrier, Xtf, harqState, srsState, Chan_UL, interfChan_UL, Xtf_noNoise);
                Phy.Config.pusch.harqState = harqState;
                for idx = 1:length(puschIdxVec)
                    puschIdx = puschIdxVec(idx);
                    payload = payloadList(idx);
                    payload = payload{1};
                    tbErr = tbErrList(idx);
                    tbErr = tbErr{1};
                    cbErr = cbErrList(idx);
                    cbErr = cbErr{1};
                    Mac.Config.pusch{idx}.foCompensationBuffer = rxDataList{idx}.foCompensationBufferOutput;
                    if (SimCtrl.puschHARQ.EnableAutoHARQ == 2) && (tbErr == 0) % PUSCH TC7899
                        Mac.Config.pusch{idx}.harqProcessID = Mac.Config.pusch{idx}.harqProcessID + 1;
                    end
                    toEstMicroSec = rxDataList{idx}.toEstMicroSec(end,1); % just report the meas result at the last DMRS Pos
                    cfoEstHz = rxDataList{idx}.cfoEstHz(end,1); % just report the meas result at the last DMRS Pos
                    dmrsRssiReportedDb = rxDataList{idx}.dmrsRssiReportedDb(end,1); % just report the meas result at the last DMRS Pos
                    dmrsRssiReportedDb_ehq = rxDataList{idx}.dmrsRssiReportedDb_ehq(end,1); % just report the meas result at the last DMRS Pos
                    postEqNoiseVardB = rxDataList{idx}.postEqNoiseVardB(end,1); % just report the meas result at the last DMRS Pos
                    noiseVardB = rxDataList{idx}.noiseVardB(end,1); % just report the meas result at the last DMRS Pos
                    rsrpdB = rxDataList{idx}.rsrpdB(end,1); % just report the meas result at the last DMRS Pos
                    rsrpdB_ehq = rxDataList{idx}.rsrpdB_ehq(end,1); % just report the meas result at the last DMRS Pos
                    sinrdB = rxDataList{idx}.sinrdB(end,1); % just report the meas result at the last DMRS Pos
                    sinrdB_ehq = rxDataList{idx}.sinrdB_ehq(end,1);
                    postEqSinrdB = rxDataList{idx}.postEqSinrdB(end,1); % just report the meas result at the last DMRS Pos                 
                    harqUci           = rxDataList{idx}.harq_uci_est;
                    harqUci_earlyHarq = rxDataList{idx}.harq_uci_est_earlyHarq;
                    csi1Uci = rxDataList{idx}.csi1_uci_est;
                    csi2Uci = rxDataList{idx}.csi2_uci_est;
                    % harqDTX = rxDataList{idx}.harqDTX;
                    % csi1DTX = rxDataList{idx}.csi1DTX;
                    % csi2DTX = rxDataList{idx}.csi2DTX;
                    isEarlyHarq             = rxDataList{idx}.isEarlyHarq;
                    harqDetStatus           = rxDataList{idx}.harqDetStatus;
                    harqDetStatus_earlyHarq = rxDataList{idx}.harqDetStatus_earlyHarq;
                    csi1DetStatus = rxDataList{idx}.csi1DetStatus;
                    csi2DetStatus = rxDataList{idx}.csi2DetStatus;
                    % harqCrcFlag = rxDataList{idx}.harqCrcFlag;
                    % csi1CrcFlag = rxDataList{idx}.csi1CrcFlag;
                    % csi2CrcFlag = rxDataList{idx}.csi2CrcFlag;
                    lowPaprGroupNumber = rxDataList{idx}.lowPaprGroupNumber;
                    lowPaprSequenceNumber = rxDataList{idx}.lowPaprSequenceNumber;
                    if SimCtrl.alg.enable_get_genie_meas
                        genie_sinrdB = rxDataList{idx}.genie_pre_sinr_dB;
                        genie_postEqSinrdB = rxDataList{idx}.genie_post_sinr_dB;
                        genie_nCov = rxDataList{idx}.genie_nCov;
                        genie_CQI = rxDataList{idx}.genie_CQI;
                        if idx == length(puschIdxVec)
                            global_slot_idx = mod(carrier.idxSlotInFrame, carrier.N_slot_frame_mu) + carrier.idxFrame*carrier.N_slot_frame_mu;
                            SimCtrl.results.pusch{idx}.genie_sinrdB(global_slot_idx+1) = genie_sinrdB;
                            SimCtrl.results.pusch{idx}.genie_postEqSinrdB(global_slot_idx+1) = genie_postEqSinrdB;
                            SimCtrl.results.pusch{idx}.genie_CQI(global_slot_idx+1) = genie_CQI;
                        end
                    end
                    if SimCtrl.enable_get_genie_channel_matrix
                        global_slot_idx = mod(carrier.idxSlotInFrame, carrier.N_slot_frame_mu) + carrier.idxFrame*carrier.N_slot_frame_mu;
                        SimCtrl.results.pusch{idx}.NMSE_ChestError_dB(global_slot_idx+1) = rxDataList{idx}.NMSE_ChestError_dB;
                    end
                    if SimCtrl.alg.ChEst_alg_selector==1 % enable logging when using 'MMSE_w_delay_and_spread_est'
                        global_slot_idx = mod(carrier.idxSlotInFrame, carrier.N_slot_frame_mu) + carrier.idxFrame*carrier.N_slot_frame_mu;
                        SimCtrl.results.pusch{idx}.ChEst.ChEst_delay_mean_microsec(global_slot_idx+1) = rxDataList{idx}.ChEst_delay_mean_microsec(end);
                        SimCtrl.results.pusch{idx}.ChEst.ChEst_delay_spread_microsec(global_slot_idx+1) = rxDataList{idx}.ChEst_delay_spread_microsec(end);
                    end
                    if SimCtrl.batchsim.save_results_LDPC
                        SimCtrl.results.pusch{idx}.LDPC_numItr = [SimCtrl.results.pusch{idx}.LDPC_numItr, rxDataList{idx}.numItr];
                        SimCtrl.results.pusch{idx}.LDPC_badItrCnt = [SimCtrl.results.pusch{idx}.LDPC_badItrCnt, rxDataList{idx}.badItrCnt];
                        SimCtrl.results.pusch{idx}.cbErr = [SimCtrl.results.pusch{idx}.cbErr, rxDataList{idx}.cbErr];
                        [cwLength,numCWs] = size(rxDataList{idx}.derateCbs);
%                         SimCtrl.results.pusch{idx}.derateCbs = cat(2,SimCtrl.results.pusch{idx}.derateCbs, reshape(rxDataList{idx}.derateCbs.',numCWs,1,cwLength)); % shape: [numCWs, numTBs, CWlength]
                        tmp_derateCCbs_percentiles = zeros(9,numCWs);
                        num_centralMoments = 4;
                        tmp_centralMoments = zeros(num_centralMoments,numCWs);
                        for idxCW = 1:numCWs
                            tmp_derateCbs = rxDataList{idx}.derateCbs(:,idxCW);
                            tmp_derateCbs = tmp_derateCbs(tmp_derateCbs~=10000);
%                             tmp_derateCbs = tmp_derateCbs/max(abs(tmp_derateCbs));
%                           Note: remove eval() for LDPC simulation, but prctile() and moment() require Statistics_Toolbox
                            tmp_derateCCbs_percentiles(:,idxCW) = eval('prctile(tmp_derateCbs,[10:10:90],1)'); % figure;hold on; for ii=1:numCWs;plot(tmp_derateCCbs_percentiles(:,ii),[0.1:0.1:0.9]);end
                            for idx_moment = 1:num_centralMoments
                                tmp_centralMoments(idx_moment, idxCW) = eval('moment(tmp_derateCbs,idx_moment)');
                            end
                        end
                        SimCtrl.results.pusch{idx}.derateCbs_percentiles = cat(2,SimCtrl.results.pusch{idx}.derateCbs_percentiles,reshape( tmp_derateCCbs_percentiles.',numCWs,1,9) ); %shape:[numCWs, numTBs, 9] 
                        SimCtrl.results.pusch{idx}.derateCbs_centralMoments = cat(2,SimCtrl.results.pusch{idx}.derateCbs_centralMoments,reshape( tmp_centralMoments.',numCWs,1,num_centralMoments) ); %shape:[numCWs, numTBs, num_centralMoments]
                    end
                    if SimCtrl.batchsim.save_results_PUSCH_ChEst
                        SimCtrl.results.pusch{idx}.Hest = cat(5,SimCtrl.results.pusch{idx}.Hest, rxDataList{idx}.H_est); % dim of SimCtrl.results.pusch{idx}.Hest: [nRxAnt, nTxLayers, Nsc, NdmrsSym, num_slots]
                        if SimCtrl.enable_get_genie_channel_matrix
                            SimCtrl.results.pusch{idx}.Hgenie = cat(5,SimCtrl.results.pusch{idx}.Hgenie, permute(rxDataList{idx}.H_genie,[3,4,1,2])); % dim of SimCtrl.results.pusch{idx}.Hgenie: [nRxAnt, nTxLayers, Nsc, Nsym, num_slots]
                            if SimCtrl.N_Interfering_UE_UL>0
                                SimCtrl.results.pusch{idx}.interf_Hgenie = cat(5,SimCtrl.results.pusch{idx}.interf_Hgenie, permute(rxDataList{idx}.interf_Hgenie,[3,4,1,2]));
                                SimCtrl.results.pusch{idx}.genie_nCov = cat(4,SimCtrl.results.pusch{idx}.genie_nCov, permute(rxDataList{idx}.genie_nCov,[3,1,2]));
                            end
                        end

                    end
                    if idx == length(puschIdxVec)
                        pdu.payload = payload;
                        pdu.tbErr = tbErr;
                        pdu.cbErr = cbErr;
                        pdu.toEstMicroSec = toEstMicroSec;
                        pdu.cfoEstHz = cfoEstHz;
                        pdu.dmrsRssiReportedDb = dmrsRssiReportedDb;
                        pdu.dmrsRssiReportedDb_ehq = dmrsRssiReportedDb_ehq;
                        pdu.postEqNoiseVardB = postEqNoiseVardB;
                        pdu.noiseVardB = noiseVardB;
                        pdu.rsrpdB = rsrpdB;
                        pdu.rsrpdB_ehq = rsrpdB_ehq;
                        pdu.sinrdB = sinrdB;
                        pdu.sinrdB_ehq= sinrdB_ehq;
                        pdu.postEqSinrdB = postEqSinrdB;
                        pdu.harqUci           = harqUci;
                        pdu.harqUci_earlyHarq = harqUci_earlyHarq;
                        pdu.isEarlyHarq       = isEarlyHarq;
                        pdu.csi1Uci = csi1Uci;
                        pdu.csi2Uci = csi2Uci;
                        % pdu.harqDTX = harqDTX;
                        % pdu.csi1DTX = csi1DTX;
                        % pdu.csi2DTX = csi2DTX;
                        % pdu.harqCrcFlag = harqCrcFlag;
                        % pdu.csi1CrcFlag = csi1CrcFlag;
                        % pdu.csi2CrcFlag = csi2CrcFlag;
                        pdu.harqDetStatus           = harqDetStatus;
                        pdu.harqDetStatus_earlyHarq = harqDetStatus_earlyHarq;
                        pdu.csi1DetStatus = csi1DetStatus;
                        pdu.csi2DetStatus = csi2DetStatus;
                        pdu.lowPaprGroupNumber = lowPaprGroupNumber;
                        pdu.lowPaprSequenceNumber = lowPaprSequenceNumber;
                    else
                        FAPIpdu{puschIdx}.payload = payload;
                        FAPIpdu{puschIdx}.tbErr = tbErr;
                        FAPIpdu{puschIdx}.cbErr = cbErr;
                        FAPIpdu{puschIdx}.toEstMicroSec = toEstMicroSec;
                        FAPIpdu{puschIdx}.cfoEstHz = cfoEstHz;
                        FAPIpdu{puschIdx}.dmrsRssiReportedDb = dmrsRssiReportedDb;
                        FAPIpdu{puschIdx}.dmrsRssiReportedDb_ehq = dmrsRssiReportedDb_ehq;
                        FAPIpdu{puschIdx}.postEqNoiseVardB = postEqNoiseVardB;
                        FAPIpdu{puschIdx}.noiseVardB = noiseVardB;
                        FAPIpdu{puschIdx}.rsrpdB = rsrpdB;
                        FAPIpdu{puschIdx}.rsrpdB_ehq = rsrpdB_ehq;
                        FAPIpdu{puschIdx}.sinrdB = sinrdB;
                        FAPIpdu{puschIdx}.sinrdB_ehq = sinrdB_ehq;
                        FAPIpdu{puschIdx}.postEqSinrdB = postEqSinrdB;
                        FAPIpdu{puschIdx}.harqUci           = harqUci;
                        FAPIpdu{puschIdx}.harqUci_earlyHarq = harqUci_earlyHarq;
                        FAPIpdu{puschIdx}.isEarlyHarq       = isEarlyHarq;
                        FAPIpdu{puschIdx}.csi1Uci = csi1Uci;
                        FAPIpdu{puschIdx}.csi2Uci = csi2Uci;
                        % FAPIpdu{puschIdx}.harqDTX = harqDTX;
                        % FAPIpdu{puschIdx}.csi1DTX = csi1DTX;
                        % FAPIpdu{puschIdx}.csi2DTX = csi2DTX;
                        % FAPIpdu{puschIdx}.harqCrcFlag = harqCrcFlag;
                        % FAPIpdu{puschIdx}.csi1CrcFlag = csi1CrcFlag;
                        % FAPIpdu{puschIdx}.csi2CrcFlag = csi2CrcFlag;
                        FAPIpdu{puschIdx}.harqDetStatus           = harqDetStatus;
                        FAPIpdu{puschIdx}.harqDetStatus_earlyHarq = harqDetStatus_earlyHarq;
                        FAPIpdu{puschIdx}.csi1DetStatus = csi1DetStatus;
                        FAPIpdu{puschIdx}.csi2DetStatus = csi2DetStatus;
                    end
                end
            end
            idxPusch = idxPusch + 1;
            FAPIpdu{idxPdu} = pdu;
    end
end

if SimCtrl.genTV.enable && SimCtrl.genTV.FAPI ...
        && ismember(idxSlotInFrame, SimCtrl.genTV.slotIdx)
    node = 'gNB';
    Xtf_remap = [];
    Xtf_remap_trsnzp = [];
    fhMsg = [];
    modCompMsg = [];
    if idxSrs == 1 % no SRS PDU allocated
        Xtf_srs = [];
    end
    saveTV_FAPI(SimCtrl.genTV, FAPIpdu, Phy.Config, Xtf_uncomp, Xtf, Xtf_prach_uncomp, Xtf_prach, Xtf_srs_uncomp, Xtf_srs, Xtf_remap, Xtf_remap_trsnzp, Phy.srsChEstDatabase, fhMsg, modCompMsg, idxSlotInFrame, node, table, 1);
end

% ML dataset collection
if SimCtrl.ml.dataset.enable_save_dataset
    new_ML_dataset_content.FAPIpdu = FAPIpdu;
    new_ML_dataset_content.Ytf = Xtf; % Rx data at gNB receiver
    new_ML_dataset_content.Gtruth = SimCtrl.ml.dataset.Gtruth;
    new_ML_dataset_content.Xtf = SimCtrl.ml.dataset.Xtf;
    SimCtrl.ml.dataset.table_ML_datasets = proc_ML_dataset(SimCtrl, carrier, new_ML_dataset_content);
    SimCtrl.ml.dataset.current_time_sec = SimCtrl.ml.dataset.current_time_sec + carrier.T_subframe/2^carrier.mu;  % increment one slot
    % clear ChEst
    for idx = 1:SimCtrl.N_UE
        SimCtrl.ml.dataset.ChEst_perUE{idx}.per_pdu = {};
        SimCtrl.ml.dataset.ChGenie_perUE{idx}.per_pdu = {};
    end
end

if SimCtrl.enableUlDlCoSim == 0
    idxSlot = idxSlot + 1;
end

carrier = updateTimeIndex(carrier, idxSlot, SimCtrl.genTV.forceSlotIdxFlag);

Phy.Config.carrier = carrier;
Phy.rx.Xtf = Xtf;

return

function [rxSamp, rxSamp_prach, Phy] = gNbUlGenFreqDomainSig(Phy, rxSamp, prachSlotInfo)

carrier = Phy.Config.carrier;
NsymPerSlot = carrier.N_symb_slot;
Nant = carrier.numRxAnt;
Nfft = carrier. Nfft;
mu = carrier.mu;
N_sc = carrier.N_sc;

Xt = rxSamp;
kappa = 64; % constants defined in 38.211
lenCP0 = (144*2^(-carrier.mu)+16)*kappa*carrier.T_c/carrier.T_samp;%(144+16)/2048*Nfft;
lenCP1 = (144*2^(-carrier.mu))*kappa*carrier.T_c/carrier.T_samp;%144/2048*Nfft;

Xtf = zeros(N_sc, NsymPerSlot, Nant);

prachArray = Phy.Config.prach;
nPrach = length(prachArray);

Xtf_prach = [];

% for non-PRACH
for idxAnt = 1:Nant
    slotTD = Xt(:, idxAnt);
    symStart = 0;
    for idxSym = 1:NsymPerSlot
        idxSymInOneSubframe = carrier.idxSlot*carrier.N_symb_slot + idxSym;
        % Remove CP
        if idxSymInOneSubframe == 1 || idxSymInOneSubframe == 1+7*2^carrier.mu
            lenCP = lenCP0;
        else
            lenCP = lenCP1;
        end
        fft_in = slotTD(symStart+lenCP+1:symStart+lenCP+Nfft);
        % FFT
        fft_out = fft(fft_in, Nfft)/sqrt(Nfft);
        symFD = zeros(1, N_sc);
        symFD(N_sc/2+1:end) = fft_out(1:N_sc/2);
        symFD(1:N_sc/2) = fft_out(end-N_sc/2+1:end);
        Xtf(:,idxSym, idxAnt) = symFD;
        symStart = symStart + lenCP + Nfft;
    end
end
rxSamp = Xtf;

% for PRACH
if prachSlotInfo.isPrachSlotInFrame
    for idxPrach = 1:nPrach
        prach = prachArray(idxPrach);
        if Phy.Config.preambleBufferValid(idxPrach) == 0
            prach.rxSamp = Xt.';
            if prachSlotInfo.isLastPrachSlot
                prach = convertPreamble(prach, carrier, 'gNB');
                y_uv_rx = prach.y_uv_rx.';
                Xtf_prach{idxPrach} = y_uv_rx;
            else
                Phy.rx.preambleBuffer{idxPrach} = Xt.';
                Phy.Config.preambleBufferValid(idxPrach) = 1;
            end
        else
            prach.rxSamp = [Phy.rx.preambleBuffer{idxPrach}, Xt.'];
            if prachSlotInfo.isLastPrachSlot
                prach = convertPreamble(prach, carrier, 'gNB');
                y_uv_rx = prach.y_uv_rx.';
                Xtf_prach{idxPrach} = y_uv_rx;
                Phy.rx.preambleBuffer{idxPrach} = [];
                Phy.Config.preambleBufferValid(idxPrach) = 0;
            else
                Phy.rx.preambleBuffer{idxPrach} = prach.rxSamp;
            end
        end
    end
end
rxSamp_prach = Xtf_prach;

return

