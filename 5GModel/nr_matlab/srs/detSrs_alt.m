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

function SrsOutputList = detSrs_alt(pduList, table, carrier, Xtf)

srsTable = table;

nPdu = length(pduList);


SrsParamsList = [];

% cell paramaters:
cellParams = [];
cellParams.slotNum     = mod(carrier.idxSlotInFrame, carrier.N_slot_frame_mu);
cellParams.frameNum    = carrier.idxFrame;
cellParams.mu          = carrier.mu;
cellParams.nRxAntSrs   = size(Xtf,3);
cellParams.nSymPerSlot = carrier.N_symb_slot;
cellParams.srsStartSym = 100;
cellParams.srsEndSym   = 0;
cellParams.nSrsSym     = 0;
cellParams.nPrbs       = size(Xtf,1) / 12;

% convert FAPI SRS user paramaters to cuPHY:
for idxPdu = 1:nPdu
    pdu = pduList{idxPdu};    
    numAntPorts_mapping = [1 2 4];
    SrsParamsList(idxPdu).nAntPorts = numAntPorts_mapping(pdu.numAntPorts+1);
    numSymbols_mapping = [1 2 4];
    SrsParamsList(idxPdu).nSyms = numSymbols_mapping(pdu.numSymbols+1);
    numRepetitions_mapping = [1 2 4];
    SrsParamsList(idxPdu).nRepetitions = numRepetitions_mapping(pdu.numRepetitions+1);
    combSize_mapping = [2 4];
    SrsParamsList(idxPdu).combSize = combSize_mapping(pdu.combSize+1);
    
    SrsParamsList(idxPdu).startSym = pdu.timeStartPosition;
    SrsParamsList(idxPdu).sequenceId = pdu.sequenceId;
    SrsParamsList(idxPdu).configIdx = pdu.configIndex;
    SrsParamsList(idxPdu).bandwidthIdx = pdu.bandwidthIndex;
    SrsParamsList(idxPdu).combOffset = pdu.combOffset;
    SrsParamsList(idxPdu).cyclicShift = pdu.cyclicShift;
    SrsParamsList(idxPdu).frequencyPosition = pdu.frequencyPosition;
    SrsParamsList(idxPdu).frequencyShift = pdu.frequencyShift;
    SrsParamsList(idxPdu).frequencyHopping = pdu.frequencyHopping;
    SrsParamsList(idxPdu).resourceType = pdu.resourceType;
    SrsParamsList(idxPdu).Tsrs = pdu.Tsrs;
    SrsParamsList(idxPdu).Toffset = pdu.Toffset;
    SrsParamsList(idxPdu).groupOrSequenceHopping = pdu.groupOrSequenceHopping;
    SrsParamsList(idxPdu).prgSizeL2 = pdu.prgSize;
    if pdu.prgSize > 4
        SrsParamsList(idxPdu).prgSize = 2;
    else
        SrsParamsList(idxPdu).prgSize = pdu.prgSize;
    end
    SrsParamsList(idxPdu).RNTI = pdu.RNTI;

    nAntPorts                      = SrsParamsList(idxPdu).nAntPorts;
    srsAntPortToUeAntMap           = zeros(nAntPorts, 1);
    ueAntennasInThisSrsResourceSet = uint32(pdu.ueAntennasInThisSrsResourceSet);
    srsAntPortIdx = 0;
    ueAntIdx      = 0;
    while(srsAntPortIdx < nAntPorts)
        if(bitand(bitshift(ueAntennasInThisSrsResourceSet,-ueAntIdx), uint32(1)))
            srsAntPortToUeAntMap(srsAntPortIdx + 1) = ueAntIdx;
            srsAntPortIdx = srsAntPortIdx + 1;
        end
        ueAntIdx = ueAntIdx + 1;
    end
    SrsParamsList(idxPdu).srsAntPortToUeAntMap = srsAntPortToUeAntMap;    
    
    
    cellParams.srsStartSym = min(cellParams.srsStartSym, pdu.timeStartPosition);
    cellParams.srsEndSym   = max(cellParams.srsEndSym, SrsParamsList(idxPdu).startSym + SrsParamsList(idxPdu).nSyms - 1);
end
cellParams.nSrsSym = cellParams.srsEndSym - cellParams.srsStartSym + 1;
srsSymIdx          = cellParams.srsStartSym : cellParams.srsEndSym;
Xtf_srs            = Xtf(:, srsSymIdx + 1, :);



SrsOutputList = detSrs_cuphy(Xtf_srs, nPdu, SrsParamsList, cellParams, srsTable);

global SimCtrl;
idxSlot = carrier.idxSlotInFrame;
idxPdu = pdu.srsPduIdx-1;

if SimCtrl.genTV.enable && SimCtrl.genTV.cuPHY && ...
        ismember(idxSlot, SimCtrl.genTV.slotIdx)
    TVname = [SimCtrl.genTV.TVname, '_SRS_gNB_CUPHY_s', num2str(idxSlot),...
        'p', num2str(idxPdu)];
    saveTV_srs(SimCtrl.genTV.tvDirName, TVname, SrsParamsList, Xtf_srs, SrsOutputList, cellParams, srsTable);
    SimCtrl.genTV.idx = SimCtrl.genTV.idx + 1;
end

end

%%
function srsRxDatabase = detSrs_cuphy(Xtf_srs, nSrsUes, SrsParamsList, cellParams, srsTable)

srsRxDatabase = init_srs_rx_dataBase(nSrsUes, SrsParamsList, cellParams);

% compute descriptors:
[srsUeDescriptors, srsUeGroupDescriptors, nComputeBlocks, computeBlockDescriptors, nRkhsCompBlocks, rkhsCompBlockDescriptors, srsRxDatabase] = computeSrsDescriptors(nSrsUes, SrsParamsList, cellParams, srsTable, srsRxDatabase);


% launch mmse compute blocks:
global SimCtrl;
if(SimCtrl.alg.srs_chEst_alg_selector == 0)
    for compBlockIdx = 0 : (nComputeBlocks - 1)
        srsRxDatabase = srsComputeBlock(computeBlockDescriptors(compBlockIdx + 1), srsUeDescriptors, srsUeGroupDescriptors, srsRxDatabase, Xtf_srs, srsTable); 
    end
end

% launch rkhs compute blocks:
if(SimCtrl.alg.srs_chEst_alg_selector == 1)
    for rkhsCompBlockIdx = 0 : (nRkhsCompBlocks - 1)
         srsRxDatabase = srsRkhsKernel(rkhsCompBlockDescriptors(rkhsCompBlockIdx + 1), srsUeDescriptors, srsRxDatabase, Xtf_srs, srsTable); 
    end
end

% finalize srs output:
srsRxDatabase = finalizeSrsOutput(nSrsUes, srsRxDatabase, srsUeDescriptors);

% TODO: enable TO for RKHS ChEst
if(SimCtrl.alg.srs_chEst_alg_selector == 1)
    for ueIdx = 0 : (nSrsUes - 1)
        srsRxDatabase{ueIdx + 1}.toEstMicroSec = 0;
    end
end

end

%%
function [srsUeDescriptors, srsUeGroupDescriptors, nComputeBlocks, computeBlockDescriptors, nRkhsCompBlocks, rkhsCompBlockDescriptors, srsRxDatabase] = computeSrsDescriptors(nSrsUes, SrsParamsList, cellParams, srsTable, srsRxDatabase)
    
    % constants:
    MAX_N_ANT_PORTS   = 4;
    MAX_N_SYM         = 4;
    MAX_N_COMB_PER_UE = 2;
    N_SC_PER_PRB      = 12;
    SRS_BW_TABLE      = srsTable.srs_BW_table;
    PRIMES_TABLE      = srsTable.srsPrimes;
    N_PRIMES          = length(PRIMES_TABLE);
    N_SYM_PER_SLOT    = 14;
    
    nComputeBlocks          = 0;
    srsUeDescriptors        = [];

    nRkhsCompBlocks          = 0;
    rkhsCompBlockDescriptors = [];
    
    for ueIdx = 0 : (nSrsUes - 1)
        nAntPorts              = SrsParamsList(ueIdx + 1).nAntPorts;
        nSyms                  = SrsParamsList(ueIdx + 1).nSyms;
        nRepetitions           = SrsParamsList(ueIdx + 1).nRepetitions;
        combSize               = SrsParamsList(ueIdx + 1).combSize;
        startSym               = SrsParamsList(ueIdx + 1).startSym;
        sequenceId             = SrsParamsList(ueIdx + 1).sequenceId;
        configIdx              = SrsParamsList(ueIdx + 1).configIdx;
        bandwidthIdx           = SrsParamsList(ueIdx + 1).bandwidthIdx;
        combOffset             = SrsParamsList(ueIdx + 1).combOffset;
        cyclicShift            = SrsParamsList(ueIdx + 1).cyclicShift;
        frequencyPosition      = SrsParamsList(ueIdx + 1).frequencyPosition;
        frequencyShift         = SrsParamsList(ueIdx + 1).frequencyShift;
        frequencyHopping       = SrsParamsList(ueIdx + 1).frequencyHopping;
        resourceType           = SrsParamsList(ueIdx + 1).resourceType;
        Tsrs                   = SrsParamsList(ueIdx + 1).Tsrs;
        Toffset                = SrsParamsList(ueIdx + 1).Toffset;
        groupOrSequenceHopping = SrsParamsList(ueIdx + 1).groupOrSequenceHopping;
        srsAntPortToUeAntMap   = SrsParamsList(ueIdx + 1).srsAntPortToUeAntMap;
        prgSize                = SrsParamsList(ueIdx + 1).prgSize;
        prgSizeL2              = SrsParamsList(ueIdx + 1).prgSizeL2;
        chEstBuffStartPrbGrp   = srsRxDatabase{ueIdx + 1}.chEstBuffer_startPrbGrp;
        
        slotNum     = cellParams.slotNum;
        frameNum    = cellParams.frameNum;
        mu          = cellParams.mu;
        
        % numerology:
        nSlotsPerFrame = 10 * 2^mu;

        % compute frequency allocation size:
        nPrbsPerHop = SRS_BW_TABLE(configIdx + 1, 2 * bandwidthIdx + 1);
        M_sc_b_SRS  = nPrbsPerHop * N_SC_PER_PRB / combSize;

        % compute phase ramp alpha:
        if combSize == 4
            n_SRS_cs_max = 12;
        elseif combSize == 2
            n_SRS_cs_max = 8;
        else
            error('combSize is not supported ...\n');
        end

        % compute SRS sequence group u and sequence number v
        c = build_Gold_sequence(sequenceId, 10 * nSlotsPerFrame * N_SYM_PER_SLOT);
        u = zeros(MAX_N_SYM, 1);
        v = zeros(MAX_N_SYM, 1);
        for l_prime = 0:(nSyms - 1)
            if groupOrSequenceHopping == 0
                f_gh           = 0;
                v(l_prime + 1) = 0;
            elseif groupOrSequenceHopping == 1
                f_gh = 0;
                for m = 0:7
                    idxSeq = 8 * (slotNum * N_SYM_PER_SLOT + startSym + l_prime) + m;
                    f_gh   = f_gh + c(idxSeq + 1) * 2^m;
                end
                f_gh = mod(f_gh, 30);
                v(l_prime + 1) = 0;
            elseif groupOrSequenceHopping == 2
                f_gh = 0;
                if M_sc_b_SRS >= 6 * N_SC_PER_PRB
                    idxSeq = slotNum * N_SYM_PER_SLOT + startSym + l_prime;
                    v(l_prime + 1) = c(idxSeq + 1);
                else
                    v(l_prime + 1) = 0;
                end
            else
                error('groupOrSequenceHopping is not supported ...\n');
            end
            u(l_prime + 1) = mod(f_gh + sequenceId, 30);
        end
        
        % determine if user gets multiple combs
        nCombs        = 1;
        nPortsPerComb = nAntPorts;
        if (cyclicShift >= n_SRS_cs_max / 2) && (nAntPorts == 4)
            nCombs        = 2;
            nPortsPerComb = 2;
        end
        
        
        % cyclic shifts:
        alphaCommon = 0;
        % cyclicShift = mod(cyclicShift, n_SRS_cs_max);
        % alphaCommon = 2 * pi * mod(cyclicShift, n_SRS_cs_max) / n_SRS_cs_max;

        portToFoccMap = zeros(MAX_N_ANT_PORTS, MAX_N_COMB_PER_UE);
        for combIdx = 0 : (nCombs - 1)
            for portIdx = 0 : (nPortsPerComb - 1)
                portToFoccMap(portIdx + 1, combIdx + 1) = mod(cyclicShift + portIdx * n_SRS_cs_max / nPortsPerComb + combIdx * n_SRS_cs_max / 4, n_SRS_cs_max);
            end
        end
        
        % mapping from ant ports to ue ant buffer:
        portToUeAntMap      = zeros(MAX_N_ANT_PORTS, MAX_N_COMB_PER_UE);
        portToL2OutUeAntMap = zeros(MAX_N_ANT_PORTS, MAX_N_COMB_PER_UE);
        for combIdx = 0 : (nCombs - 1)
            for portIdx = 0 : (nPortsPerComb - 1)
                portToUeAntMap(portIdx + 1, combIdx + 1)      = srsAntPortToUeAntMap(portIdx * nCombs + combIdx + 1);
                portToL2OutUeAntMap(portIdx + 1, combIdx + 1) = portIdx * nCombs + combIdx;
            end
        end
        
        % comb offsets:
        combOffsets = zeros(MAX_N_COMB_PER_UE, 1);
        for combIdx = 0 : (nCombs - 1)
            combOffsets(combIdx + 1) = mod(combOffset + combIdx * combSize / 2, combSize);
        end
          
        % compute starting Prbs
        hopStartPrbs = frequencyShift * ones(MAX_N_SYM, 1);
        nHopsInSlot  = nSyms / nRepetitions;
        
        for hopIdx = 0 : (nHopsInSlot - 1)
            for b = 0 : bandwidthIdx
                if frequencyHopping >= bandwidthIdx
                    Nb      = SRS_BW_TABLE(configIdx + 1, 2*b + 2);
                    m_SRS_b = SRS_BW_TABLE(configIdx + 1, 2*b + 1);
                    nb      = mod(floor(4 * frequencyPosition / m_SRS_b), Nb);
                else
                    Nb      = SRS_BW_TABLE(configIdx + 1, 2*b + 2);
                    m_SRS_b = SRS_BW_TABLE(configIdx + 1, 2*b + 1);
                    if b <= frequencyHopping
                        nb = mod(floor(4 * frequencyPosition / m_SRS_b), Nb);
                    else
                        if resourceType == 0
                            n_SRS = hopIdx;
                        else
                            slotIdx = nSlotsPerFrame * frameNum + slotNum - Toffset;
                            if mod(slotIdx, Tsrs) == 0
                                n_SRS = (slotIdx / Tsrs) * (nSyms / nRepetitions) + hopIdx;
                            else
                                warning('Not an SRS slot ...\n');
                                n_SRS = 0;
                            end
                        end
                        PI_bm1 = 1;
                        for b_prime = frequencyHopping + 1 : (b - 1)
                            PI_bm1 = PI_bm1 * SRS_BW_TABLE(configIdx + 1, 2*b_prime + 2);
                        end
                        PI_b = PI_bm1 * Nb;
                        if mod(Nb, 2) == 0
                            Fb = (Nb / 2) * floor(mod(n_SRS, PI_b) / PI_bm1) + floor(mod(n_SRS, PI_b) / (2 * PI_bm1));
                        else
                            Fb = floor(Nb / 2) * floor(n_SRS / PI_bm1);
                        end
                        nb = mod(Fb + floor(4 * frequencyPosition / m_SRS_b), Nb);
                    end
                end
                
                hopStartPrbs(hopIdx + 1) = hopStartPrbs(hopIdx + 1) +  m_SRS_b * nb;
            end
        end
        
        % low PAPR sequence paramaters:
        nCombScPerPrb   = 12 / combSize;
        nSubcarriers    = m_SRS_b * nCombScPerPrb;
        lowPaprTableIdx = 255;
        lowPaprPrime    = 0;
        q               = zeros(MAX_N_SYM, 1);
        
        if(nSubcarriers == 12)
            lowPaprTableIdx = 0;
        elseif(nSubcarriers == 24)
            lowPaprTableIdx = 1;
        else
            for i = 2 : N_PRIMES
                if(PRIMES_TABLE(i) > nSubcarriers)
                    lowPaprPrime = PRIMES_TABLE(i-1);
                    break;
                end
            end
            
            for symIdx = 0 : (nSyms - 1)
                qBar          = lowPaprPrime * (u(symIdx + 1) + 1) / 31;
                q(symIdx + 1) = floor(qBar + 0.5) + v(symIdx + 1)*(-1)^(floor(2*qBar));
            end     
        end
        
        % combine hops with the same start PRB
        nHops_prime       = 1;
        nRepPerHop_prime  = zeros(MAX_N_SYM, 1);
        repSymIdxs_prime  = zeros(MAX_N_SYM, MAX_N_SYM);
        hopStartPrbs_prime = zeros(MAX_N_SYM, 1);
        
        nRepPerHop_prime(1) = nRepetitions;
        for repIdx = 0 : (nRepetitions - 1)
            repSymIdxs_prime(repIdx + 1, 1) = startSym - cellParams.srsStartSym + repIdx;
        end
        nRepPerHop_prime(1)  = nRepetitions;
        hopStartPrbs_prime(1) = hopStartPrbs(1);
        
        for hopIdx = 1 : (nHopsInSlot - 1)
            newHopPrime_flag = 1;
            for hopIdx_prime = 0 : (nHops_prime - 1)
                if(hopStartPrbs_prime(hopIdx_prime + 1) == hopStartPrbs(hopIdx + 1))
                    for repIdx = 0 : (nRepetitions - 1)
                        repSymIdxs_prime(nRepPerHop_prime(hopIdx_prime + 1) + repIdx + 1, hopIdx_prime + 1) = startSym - cellParams.srsStartSym + hopIdx * nRepetitions + repIdx;
                    end
                    nRepPerHop_prime(hopIdx_prime + 1) = nRepPerHop_prime(hopIdx_prime + 1) + nRepetitions;
                    newHopPrime_flag                   = 0;
                    break;
                end
            end
            
            if(newHopPrime_flag == 1)
                nRepPerHop_prime(nHops_prime + 1) = nRepetitions;
                for repIdx = 0 : (nRepetitions - 1)
                    repSymIdxs_prime(repIdx + 1, nHops_prime + 1) = startSym - cellParams.srsStartSym + hopIdx * nRepetitions + repIdx;
                end
                nRepPerHop_prime(nHops_prime + 1)   = nRepetitions;
                hopStartPrbs_prime(nHops_prime + 1) = hopStartPrbs(hopIdx + 1);
                nHops_prime                         = nHops_prime + 1;
            end
        end
        
        % set tensor dimension for ChEst report to L2:
        srsRxDatabase{ueIdx + 1}.prgSize       = prgSize;  
        srsRxDatabase{ueIdx + 1}.prgSizeL2     = prgSizeL2;        
        srsRxDatabase{ueIdx + 1}.HestToL2Inner = zeros(nPrbsPerHop/prgSize *nHops_prime, srsRxDatabase{ueIdx + 1}.chEstBuffer_nRxAntSrs, nAntPorts);
        srsRxDatabase{ueIdx + 1}.HestToL2      = zeros(nPrbsPerHop/prgSizeL2 *nHops_prime, srsRxDatabase{ueIdx + 1}.chEstBuffer_nRxAntSrs, nAntPorts);
        srsRxDatabase{ueIdx + 1}.nPrbGrps      = nPrbsPerHop / prgSize * nHops_prime;
        srsRxDatabase{ueIdx + 1}.nPrbGrpsL2    = nPrbsPerHop / prgSizeL2 * nHops_prime;
        srsRxDatabase{ueIdx + 1}.startValidPrg = floor(hopStartPrbs(1) / prgSize);
        
        % save user descriptor:
        srsUeDescriptors(ueIdx + 1).repSymIdxs              = repSymIdxs_prime;
        srsUeDescriptors(ueIdx + 1).hopStartPrbs            = hopStartPrbs_prime;
        srsUeDescriptors(ueIdx + 1).nRepPerHop              = nRepPerHop_prime; 
        srsUeDescriptors(ueIdx + 1).nPrbsPerHop             = nPrbsPerHop;
        srsUeDescriptors(ueIdx + 1).u                       = u;
        srsUeDescriptors(ueIdx + 1).q                       = q;
        srsUeDescriptors(ueIdx + 1).alphaCommon             = alphaCommon;
        srsUeDescriptors(ueIdx + 1).n_SRS_cs_max            = n_SRS_cs_max;
        srsUeDescriptors(ueIdx + 1).lowPaprTableIdx         = lowPaprTableIdx;
        srsUeDescriptors(ueIdx + 1).lowPaprPrime            = lowPaprPrime;
        srsUeDescriptors(ueIdx + 1).nPorts                  = nCombs * nPortsPerComb;
        srsUeDescriptors(ueIdx + 1).nPortsPerComb           = nPortsPerComb;
        srsUeDescriptors(ueIdx + 1).portToFoccMap           = portToFoccMap;
        srsUeDescriptors(ueIdx + 1).combSize                = combSize;
        srsUeDescriptors(ueIdx + 1).combOffsets             = combOffsets;
        srsUeDescriptors(ueIdx + 1).nCombScPerPrb           = nCombScPerPrb;
        srsUeDescriptors(ueIdx + 1).chEstBuffStartPrbGrp    = chEstBuffStartPrbGrp;
        srsUeDescriptors(ueIdx + 1).portToUeAntMap          = portToUeAntMap;
        srsUeDescriptors(ueIdx + 1).portToL2OutUeAntMap     = portToL2OutUeAntMap;
        srsUeDescriptors(ueIdx + 1).mu                      = mu;
        srsUeDescriptors(ueIdx + 1).nUniqueHops             = nHops_prime;
        srsUeDescriptors(ueIdx + 1).prgSize                 = prgSize;
        srsUeDescriptors(ueIdx + 1).nHops_prime             = nHops_prime;
        srsUeDescriptors(ueIdx + 1).nCombs                  = nCombs;

        % % save compute block descriptors:
        % nPrbsPerComputeBlock = 4; % fixed 4 PRBs per block
        % nBlocksFreq          = m_SRS_b / nPrbsPerComputeBlock;
        % 
        % for freqBlockIdx = 0 : (nBlocksFreq - 1)
        %     for hopIdx = 0 : (nHops_prime - 1)
        %         for combIdx = 0 : (nCombs - 1)
        %             computeBlockDescriptors(nComputeBlocks + 1).ueIdx         = ueIdx;
        %             computeBlockDescriptors(nComputeBlocks + 1).hopIdx        = hopIdx;
        %             computeBlockDescriptors(nComputeBlocks + 1).combIdx       = combIdx;
        %             computeBlockDescriptors(nComputeBlocks + 1).blockStartPrb = hopStartPrbs(hopIdx + 1) + freqBlockIdx * nPrbsPerComputeBlock;
        % 
        %             nComputeBlocks = nComputeBlocks + 1;
        %         end
        %     end
        % end

        % save RKHS compute block descriptors:
        nPol = 2;
        for combIdx = 0 : (nCombs - 1)
            for portIdx = 0 : (srsUeDescriptors(ueIdx + 1).nPortsPerComb - 1)
                for polIdx = 0 : (nPol - 1)
                    rkhsCompBlockDescriptors(nRkhsCompBlocks + 1).ueIdx   = ueIdx;
                    rkhsCompBlockDescriptors(nRkhsCompBlocks + 1).combIdx = combIdx;
                    rkhsCompBlockDescriptors(nRkhsCompBlocks + 1).portIdx = portIdx;
                    rkhsCompBlockDescriptors(nRkhsCompBlocks + 1).polIdx  = polIdx;
            
                    nRkhsCompBlocks = nRkhsCompBlocks + 1;
                end
            end
        end
    end



    % Batch users which lie on the same SRS comb and symbol:
    srsUeGroupDescriptors = [];
    nUeGroups             = 0;
    
    for ueIdx = 0 : (nSrsUes - 1)
        % number of compute blocks the user is located in:
        nHops_prime      = srsUeDescriptors(ueIdx + 1).nHops_prime;
        nCombs           = srsUeDescriptors(ueIdx + 1).nCombs;
        nPortsPerComb    = srsUeDescriptors(ueIdx + 1).nPortsPerComb;

        % loop over the users time/frequency blocks:
        for hopIdx = 0 : (nHops_prime - 1)
            for combIdx = 0 : (nCombs - 1)

                newUeGroupFlag = 1;
                for ueGroupIdx = 0 : (nUeGroups - 1)
                    firstUeInBlockIdx     = srsUeGroupDescriptors(ueGroupIdx + 1).ueIdxs(1);
                    firstUeInBlockHopIdx  = srsUeGroupDescriptors(ueGroupIdx + 1).ueHopIdxs(1);
                    firstUeInBlockCombIdx = srsUeGroupDescriptors(ueGroupIdx + 1).ueCombIdxs(1);

                    nPrbsPerHop_check          = (srsUeDescriptors(ueIdx + 1).nPrbsPerHop              == srsUeDescriptors(firstUeInBlockIdx + 1).nPrbsPerHop);
                    ueStartSrsSym_check        = (srsUeDescriptors(ueIdx + 1).repSymIdxs(1, 1)         == srsUeDescriptors(firstUeInBlockIdx + 1).repSymIdxs(1, 1));
                    hopStartPrb_check          = (srsUeDescriptors(ueIdx + 1).hopStartPrbs(hopIdx + 1) == srsUeDescriptors(firstUeInBlockIdx + 1).hopStartPrbs(firstUeInBlockHopIdx + 1));
                    nRepPerHop_check           = (srsUeDescriptors(ueIdx + 1).nRepPerHop(hopIdx + 1)   == srsUeDescriptors(firstUeInBlockIdx + 1).nRepPerHop(firstUeInBlockHopIdx + 1));
                    alphaCommon_check          = (srsUeDescriptors(ueIdx + 1).alphaCommon              == srsUeDescriptors(firstUeInBlockIdx + 1).alphaCommon);
                    n_SRS_cs_max_check         = (srsUeDescriptors(ueIdx + 1).n_SRS_cs_max             == srsUeDescriptors(firstUeInBlockIdx + 1).n_SRS_cs_max);
                    lowPaprTableIdx_check      = (srsUeDescriptors(ueIdx + 1).lowPaprTableIdx          == srsUeDescriptors(firstUeInBlockIdx + 1).lowPaprTableIdx);
                    lowPaprPrime_check         = (srsUeDescriptors(ueIdx + 1).lowPaprPrime             == srsUeDescriptors(firstUeInBlockIdx + 1).lowPaprPrime);
                    combSize_check             = (srsUeDescriptors(ueIdx + 1).combSize                 == srsUeDescriptors(firstUeInBlockIdx + 1).combSize);
                    combOffset_check           = (srsUeDescriptors(ueIdx + 1).combOffsets(combIdx + 1) == srsUeDescriptors(firstUeInBlockIdx + 1).combOffsets(firstUeInBlockCombIdx + 1));
                    prgSize_check              = (srsUeDescriptors(ueIdx + 1).prgSize                  == srsUeDescriptors(firstUeInBlockIdx + 1).prgSize);
                    u_check                    = not(logical(sum(srsUeDescriptors(ueIdx + 1).u - srsUeDescriptors(firstUeInBlockIdx + 1).u)));
                    q_check                    = not(logical(sum(srsUeDescriptors(ueIdx + 1).q - srsUeDescriptors(firstUeInBlockIdx + 1).q)));
                    if(nPrbsPerHop_check && ueStartSrsSym_check && hopStartPrb_check && nRepPerHop_check && u_check && q_check && alphaCommon_check && n_SRS_cs_max_check && lowPaprTableIdx_check && lowPaprPrime_check && combSize_check && combOffset_check && prgSize_check)
                        srsUeGroupDescriptors(ueGroupIdx + 1).ueIdxs      = [srsUeGroupDescriptors(ueGroupIdx + 1).ueIdxs ueIdx];
                        srsUeGroupDescriptors(ueGroupIdx + 1).ueCombIdxs  = [srsUeGroupDescriptors(ueGroupIdx + 1).ueCombIdxs combIdx];
                        srsUeGroupDescriptors(ueGroupIdx + 1).nUes        = srsUeGroupDescriptors(ueGroupIdx + 1).nUes + 1;
                        srsUeGroupDescriptors(ueGroupIdx + 1).ueHopIdxs   = [srsUeGroupDescriptors(ueGroupIdx + 1).ueHopIdxs hopIdx];
                        srsUeGroupDescriptors(ueGroupIdx + 1).nAntPorts   = srsUeGroupDescriptors(ueGroupIdx + 1).nAntPorts + nPortsPerComb;
                        newUeGroupFlag = 0;
                        break;
                    end
                end

                % create new compute block if needed:
                if(newUeGroupFlag == 1)
                    srsUeGroupDescriptors(nUeGroups + 1).ueIdxs        = ueIdx;
                    srsUeGroupDescriptors(nUeGroups + 1).ueCombIdxs    = combIdx;
                    srsUeGroupDescriptors(nUeGroups + 1).ueHopIdxs     = hopIdx;
                    srsUeGroupDescriptors(nUeGroups + 1).nUes          = 1;
                    srsUeGroupDescriptors(nUeGroups + 1).nAntPorts     = nPortsPerComb;
                    nUeGroups = nUeGroups + 1;
                end
            end
        end
    end


    % Break SRS UE groups into compute blocks:
    nPrbsPerComputeBlock    = 4; % fixed 4 PRBs per block
    nCompBlocks             = 0;
    computeBlockDescriptors = [];

    for ueGroupIdx = 0 : (nUeGroups - 1)
        % extract paramaters common to users in group:
        firstUeInBlockIdx    = srsUeGroupDescriptors(ueGroupIdx + 1).ueIdxs(1);
        firstUeInBlockHopIdx = srsUeGroupDescriptors(ueGroupIdx + 1).ueHopIdxs(1);
        startPrb             = srsUeDescriptors(firstUeInBlockIdx + 1).hopStartPrbs(firstUeInBlockHopIdx + 1);
        nPrbs                = srsUeDescriptors(firstUeInBlockIdx + 1).nPrbsPerHop;
        nRxAnts              = cellParams.nRxAntSrs;

        % break RX antennas into different compute blocks:
        if((nRxAnts > 16) && (srsUeGroupDescriptors(ueGroupIdx + 1).nAntPorts > 4))
            maxNumAntsPerBlock = 16;
            nRxAntBlocks       = ceil(nRxAnts / maxNumAntsPerBlock);
        else
            maxNumAntsPerBlock = nRxAnts;
            nRxAntBlocks       = 1;
        end
        nRxAntsInLastBlock = nRxAnts - maxNumAntsPerBlock * (nRxAntBlocks - 1);

        % break frequency allocation into different blocks:
        nPrbsBlocks = nPrbs / nPrbsPerComputeBlock;

        % Compute blokcs:
        for freqBlockIdx = 0 : (nPrbsBlocks - 1)
            blockStartPrb = startPrb + freqBlockIdx * nPrbsPerComputeBlock;

            for antBlockIdx = 0 : (nRxAntBlocks - 2)
                computeBlockDescriptors(nComputeBlocks + 1).ueGroupIdx    = ueGroupIdx;
                computeBlockDescriptors(nComputeBlocks + 1).blockStartPrb = blockStartPrb;
                computeBlockDescriptors(nComputeBlocks + 1).nRxAnts       = maxNumAntsPerBlock;
                computeBlockDescriptors(nComputeBlocks + 1).blockStartAnt = antBlockIdx * maxNumAntsPerBlock;
                nComputeBlocks = nComputeBlocks + 1;
            end

            computeBlockDescriptors(nComputeBlocks + 1).ueGroupIdx    = ueGroupIdx;
            computeBlockDescriptors(nComputeBlocks + 1).blockStartPrb = blockStartPrb;
            computeBlockDescriptors(nComputeBlocks + 1).nRxAnts       = nRxAntsInLastBlock;
            computeBlockDescriptors(nComputeBlocks + 1).blockStartAnt = nRxAnts - nRxAntsInLastBlock;
            nComputeBlocks = nComputeBlocks + 1;
        end
    end
end




%%
% SRS RKHS KERNEL

function srsRxDatabase = srsRkhsKernel(rkhsCompBlockDescriptor, srsUeDescriptors, srsRxDatabase, Xtf_srs, srsTable)

    % compute block descriptor:
    ueIdx   = rkhsCompBlockDescriptor.ueIdx;
    hopIdx  = 0;
    combIdx = rkhsCompBlockDescriptor.combIdx;
    polIdx  = rkhsCompBlockDescriptor.polIdx;
    portIdx = rkhsCompBlockDescriptor.portIdx;

    % User descriptor:
    repSymIdxs           = srsUeDescriptors(ueIdx + 1).repSymIdxs(:, hopIdx + 1);
    ueStartSrsSym        = srsUeDescriptors(ueIdx + 1).repSymIdxs(1, 1);
    hopStartPrb          = srsUeDescriptors(ueIdx + 1).hopStartPrbs(hopIdx + 1);
    nRepPerHop           = srsUeDescriptors(ueIdx + 1).nRepPerHop(hopIdx + 1);
    nPrbsPerHop          = srsUeDescriptors(ueIdx + 1).nPrbsPerHop;
    u                    = srsUeDescriptors(ueIdx + 1).u;
    q                    = srsUeDescriptors(ueIdx + 1).q;
    alphaCommon          = srsUeDescriptors(ueIdx + 1).alphaCommon;
    n_SRS_cs_max         = srsUeDescriptors(ueIdx + 1).n_SRS_cs_max;
    lowPaprTableIdx      = srsUeDescriptors(ueIdx + 1).lowPaprTableIdx;
    lowPaprPrime         = srsUeDescriptors(ueIdx + 1).lowPaprPrime;
    nPorts               = srsUeDescriptors(ueIdx + 1).nPorts;
    nPortsPerComb        = srsUeDescriptors(ueIdx + 1).nPortsPerComb;
    portToFoccMap        = srsUeDescriptors(ueIdx + 1).portToFoccMap(:, combIdx + 1);
    combSize             = srsUeDescriptors(ueIdx + 1).combSize;
    combOffset           = srsUeDescriptors(ueIdx + 1).combOffsets(combIdx + 1);
    nCombScPerPrb        = srsUeDescriptors(ueIdx + 1).nCombScPerPrb;
    portToUeAntMap       = srsUeDescriptors(ueIdx + 1).portToUeAntMap(:, combIdx + 1);
    portToL2OutUeAntMap  = srsUeDescriptors(ueIdx + 1).portToL2OutUeAntMap(:, combIdx + 1);
    chEstBuffStartPrbGrp = srsUeDescriptors(ueIdx + 1).chEstBuffStartPrbGrp;
    prgSize              = srsUeDescriptors(ueIdx + 1).prgSize;

    % noise measurments paramaters:
    tau               = 1 / (combSize * 30 * 10^3);
    baseNoiseOffset   = 200;
    noiseRegionLength = 40;

    % Frequency size:
    nSrsSc   = 816;
    nZpSrsSc = 1024;
    lambda   = 2.31*10^(-6) * 120 * 10^3;
    % nCpInts  = ceil(lambda *  nZpSrsSc);
    nCpInts = 230;

    % Cell paramaters:
    nVerAnt  = 4;
    nHorAnt  = 8;
    nAnt     = nVerAnt * nHorAnt;

    % Total dimension size:
    totalDim = nVerAnt * nHorAnt * nSrsSc;

    % Cover codes:
    LOW_PAPR_TABLE_0  = srsTable.srsLowPaprTable0;
    LOW_PAPR_TABLE_1  = srsTable.srsLowPaprTable1;
    FOOC_TABLE_COMB_2 = srsTable.srsFocc_comb2;
    FOOC_TABLE_COMB_4 = srsTable.srsFocc_comb4;

   switch combSize
       case 2
            FOCC_TABLE  = FOOC_TABLE_COMB_2;
            FOCC_LENGTH = 8;
       case 4
            FOCC_TABLE  = FOOC_TABLE_COMB_4;
            FOCC_LENGTH = 12;
   end

    % Eigenvectors:
    verAntEigVec = srsTable.srs_rkhs_tables.eigenVecTable{1};
    verAntEigVal = srsTable.srs_rkhs_tables.eigValTable{1};
    verAntCorr   = srsTable.srs_rkhs_tables.corrTable{1};
    verNumEigs   = srsTable.srs_rkhs_tables.gridPrms{1}.nEig;

    horAntEigVec = srsTable.srs_rkhs_tables.eigenVecTable{2};
    horAntEigVal = srsTable.srs_rkhs_tables.eigValTable{2};
    horAntCorr   = srsTable.srs_rkhs_tables.corrTable{2};
    horNumEigs   = srsTable.srs_rkhs_tables.gridPrms{2}.nEig;

    scEigVec  = srsTable.srs_rkhs_tables.eigenVecTable{3};
    scEigVal  = srsTable.srs_rkhs_tables.eigValTable{3};
    scCorr    = srsTable.srs_rkhs_tables.corrTable{3};
    scNumEigs = srsTable.srs_rkhs_tables.gridPrms{3}.nEig;

    eigVal = zeros(scNumEigs, verNumEigs, horNumEigs);
    for freqEigIdx = 0 : (scNumEigs - 1)
        for verAntEigIdx = 0 : (verNumEigs - 1)
            for horAntEigIdx = 0 : (horNumEigs - 1)
                eigVal(freqEigIdx + 1, verAntEigIdx + 1, horAntEigIdx + 1) = scEigVal(freqEigIdx + 1) * verAntEigVal(verAntEigIdx + 1) * horAntEigVal(horAntEigIdx + 1);
            end
        end
    end

    eigValAnts = zeros(verNumEigs, horNumEigs);
    for verAntEigIdx = 0 : (verNumEigs - 1)
        for horAntEigIdx = 0 : (horNumEigs - 1)
            eigValAnts(verAntEigIdx + 1, horAntEigIdx + 1) = verAntEigVal(verAntEigIdx + 1) * horAntEigVal(horAntEigIdx + 1);
        end
    end

    %% STEP 1: Load Rx SRS subcarriers, remove ZC cover-code, average repititions.
    rxSrs = zeros(nSrsSc, nAnt);

    totalNumSc  = size(Xtf_srs, 1);
    totalNumSym = size(Xtf_srs, 2);
    Xtf_srs     = reshape(Xtf_srs, totalNumSc, totalNumSym, 2, nAnt);
    
    for antIdx = 0 : (nAnt - 1)
        for scIdx = 0 : (nSrsSc - 1)
           ZcScIdx   = scIdx;
           loadScIdx = scIdx * combSize + combOffset;

           for repIdx = 0 : (nRepPerHop - 1)
               symIdx = repSymIdxs(repIdx + 1);
               
               % extract subcarrier for this repetition:
               y = Xtf_srs(loadScIdx + 1, symIdx + 1, polIdx + 1, antIdx + 1);
               
               % compute subcarrier ZC coverCode for this repetition:
               conj_r = 0;
               if(lowPaprTableIdx == 0)
                   u_repIdx = u(symIdx - ueStartSrsSym + 1);
                   conj_r   = exp(-1i*(pi*LOW_PAPR_TABLE_0(u_repIdx + 1 , ZcScIdx + 1) / 4 + alphaCommon*scIdx));
               elseif(lowPaprTableIdx == 1)
                   u_repIdx = u(symIdx - ueStartSrsSym + 1);
                   conj_r   = exp(-1i*(pi*LOW_PAPR_TABLE_1(u_repIdx + 1, ZcScIdx + 1) / 4 + alphaCommon*scIdx));
               else
                   q_repIdx = q(symIdx - ueStartSrsSym + 1);
                   m        = mod(ZcScIdx, lowPaprPrime);
                   conj_r   = exp(-1i*(- pi*q_repIdx*m*(m+1)/lowPaprPrime + alphaCommon*scIdx));
               end

               % cyclic shift:
               foocIdx = portToFoccMap(portIdx + 1);
               conj_cs = conj(FOCC_TABLE(mod(ZcScIdx, FOCC_LENGTH) + 1, foocIdx + 1));
               
               % remove ZC coverCode and add:
               rxSrs(scIdx + 1, antIdx + 1) = rxSrs(scIdx + 1, antIdx + 1) + conj_r * conj_cs * y;
           end
           
           % normalize:
           rxSrs(scIdx + 1, antIdx + 1) = rxSrs(scIdx + 1, antIdx + 1) / nRepPerHop;
        end
    end
    rxSrs = reshape(rxSrs, nSrsSc, nVerAnt, nHorAnt);

    %% STEP 2: frequency projection / noise estimation

    W = hamming(nSrsSc);
    W = W / sqrt(sum(abs(W).^2));
    
    freqProj    = zeros(nCpInts, nVerAnt, nHorAnt, scNumEigs);
    hammingProj = zeros(2*nCpInts, nVerAnt, nHorAnt);

    for verAntIdx = 0 : (nVerAnt - 1)
        for horAntIdx = 0 : (nHorAnt - 1)
            freqCorr                = zeros(nZpSrsSc, scNumEigs);
            freqCorr(1 : nSrsSc, :) = rxSrs(:, verAntIdx + 1, horAntIdx + 1) .* conj(scEigVec);
            freqCorr                = ifft(freqCorr, nZpSrsSc, 1) * nZpSrsSc;

            freqProj(:, verAntIdx + 1, horAntIdx + 1, :) = freqCorr(1 : nCpInts, :);

            hammingCorr             = zeros(nZpSrsSc, 1);
            hammingCorr(1 : nSrsSc) = rxSrs(:, verAntIdx + 1, horAntIdx + 1) .* W;
            hammingCorr             = ifft(hammingCorr) * nZpSrsSc;

            hammingProj(:, verAntIdx + 1, horAntIdx + 1) = hammingCorr(1 : 2*nCpInts);
        end
    end

    %% STEP 3: measure interference + noise.

    noiseAndIntEnergy = zeros(nVerAnt, nHorAnt);

    for cpIntIdx = baseNoiseOffset : (baseNoiseOffset + noiseRegionLength - 1)
        intSlice = squeeze(hammingProj(cpIntIdx + 1, :, :)); % nVerAnt x nHorAnt

        % project onto vertical antenna eigenvectors:
        verProj = zeros(nVerAnt, nHorAnt, verNumEigs);
        for horAntIdx = 0 : (nHorAnt - 1)
            verProj(:, horAntIdx + 1, :) = squeeze(intSlice(:, horAntIdx + 1)) .* conj(verAntEigVec);
        end
        verProj = ifft(verProj, nVerAnt, 1) * nVerAnt;

        % project onto horizantal antenna eigenvectors:
        verHorProj = zeros(nVerAnt, nHorAnt, verNumEigs, horNumEigs);
        for verAntIdx = 0 : (nVerAnt - 1)
            for verEigIdx = 0 : (verNumEigs - 1)
                verHorProj(verAntIdx + 1, :, verEigIdx + 1, :) = squeeze(verProj(verAntIdx + 1, :, verEigIdx + 1)).' .* horAntEigVec;
            end
        end
        verHorProj = ifft(verHorProj, nHorAnt, 2) * nHorAnt;
        verHorProj = permute(verHorProj, [3 4 1 2]); % verEig x horEig x verAntIntIdx x horAntIntIdx

        E = abs(verHorProj).^2 .* repmat(eigValAnts, 1, 1, nVerAnt, nHorAnt);
        E = sum(E, 1);
        E = sum(E, 2);
        E = squeeze(E) / sum(eigValAnts(:));

        noiseAndIntEnergy = noiseAndIntEnergy + E;
    end

    noiseAndIntEnergy = noiseAndIntEnergy / noiseRegionLength;
    N0                = mean(noiseAndIntEnergy(:));

    %% STEP 3: antenna projection. Identify boxes above the noise floor. Compute signal enerngy.

    nOccupiedBoxes = 0;
    maxNumBoxes    = 620;

    projCoeffs            = zeros(scNumEigs, verNumEigs, horNumEigs, maxNumBoxes);
    projCoeffFreqIntIdx   = zeros(maxNumBoxes, 1);
    projCoeffVerAntIntIdx = zeros(maxNumBoxes, 1);
    projCoeffHorAntIntIdx = zeros(maxNumBoxes, 1);
    nTries      = 0;
    maxNumTries = 20;
    boxThreshold = 6*N0;

    while (nTries < maxNumTries) 

        for cpIntIdx = 0 : (nCpInts - 1)
            intSlice = freqProj(cpIntIdx + 1, :, : , :, :);
            intSlice = reshape(intSlice, nVerAnt, nHorAnt, scNumEigs);
    
            % project onto vertical antenna eigenvectors:
            verProj = zeros(nVerAnt, nHorAnt, scNumEigs, verNumEigs);
            for horAntIdx = 0 : (nHorAnt - 1)
                for freqEigIdx = 0 : (scNumEigs - 1)
                    verProj(:, horAntIdx + 1, freqEigIdx + 1, :) = squeeze(intSlice(:, horAntIdx + 1, freqEigIdx + 1)) .* conj(verAntEigVec);
                end
            end
            verProj = ifft(verProj, nVerAnt, 1) * nVerAnt;
    
            % project onto horizantal antenna eigenvectors:
            verHorProj = zeros(nVerAnt, nHorAnt, scNumEigs, verNumEigs, horNumEigs);
            for verAntIdx = 0 : (nVerAnt - 1)
                for freqEigIdx = 0 : (scNumEigs - 1)
                    for verEigIdx = 0 : (verNumEigs - 1)
                        verHorProj(verAntIdx + 1, :, freqEigIdx + 1, verEigIdx + 1, :) = squeeze(verProj(verAntIdx + 1, :, freqEigIdx + 1, verEigIdx + 1)).' .* horAntEigVec;
                    end
                end
            end
            verHorProj = ifft(verHorProj, nHorAnt, 2) * nHorAnt;
            verHorProj = permute(verHorProj, [3 4 5 1 2]); % freqEig x verEig x horEig x verAntIntIdx x horAntIntIdx
    
            E = abs(verHorProj).^2 .* repmat(eigVal, 1, 1, 1, nVerAnt, nHorAnt);
            E = sum(E, 1);
            E = sum(E, 2);
            E = sum(E, 3);
            E = squeeze(E) / sum(eigVal(:));
    
            for horAntIdx = 0 : (nHorAnt - 1)
                for verIntIdx = 0 : (nVerAnt - 1)
             
                    if((E(verIntIdx + 1, horAntIdx + 1) - noiseAndIntEnergy(verIntIdx + 1, horAntIdx + 1)) > boxThreshold)
                        projCoeffFreqIntIdx(nOccupiedBoxes + 1)   = cpIntIdx;
                        projCoeffVerAntIntIdx(nOccupiedBoxes + 1) = verIntIdx;
                        projCoeffHorAntIntIdx(nOccupiedBoxes + 1) = horAntIdx;
                        projCoeffs(:, :, :, nOccupiedBoxes + 1)   = verHorProj(:, :, :, verIntIdx + 1, horAntIdx + 1);
                        nOccupiedBoxes = nOccupiedBoxes + 1;
                    end
                end
            end
        end

        nTries = nTries + 1;

        if(nTries < maxNumTries)
            if nOccupiedBoxes > (maxNumBoxes - 32)
                boxThreshold = 2 * boxThreshold; % too many boxes selected, raise noise floor
                nOccupiedBoxes = 0;
            elseif (nOccupiedBoxes == 0)
                boxThreshold = boxThreshold / 2; % no boxes selected, lower noise floor
            else
                break;
            end
        end
    end

    projCoeffFreqIntIdx   = projCoeffFreqIntIdx(1 : nOccupiedBoxes);
    projCoeffVerAntIntIdx = projCoeffVerAntIntIdx(1 : nOccupiedBoxes);
    projCoeffHorAntIntIdx = projCoeffHorAntIntIdx(1 : nOccupiedBoxes);
    projCoeffs            = projCoeffs(:, :, :,  1 : nOccupiedBoxes);

    N0_occupiedBoxes = zeros(nOccupiedBoxes, 1);
    for boxIdx = 0 : (nOccupiedBoxes - 1)
        N0_occupiedBoxes(boxIdx + 1) = noiseAndIntEnergy(projCoeffVerAntIntIdx(boxIdx + 1) + 1, projCoeffHorAntIntIdx(boxIdx + 1) + 1);
    end

    %% STEP 4: matching pursuit

    nEqBoxes            = 0;
    eqCoeffBoxIdx       = [];
    eqCoeffFreqIntIdx   = zeros(nOccupiedBoxes, 1);
    eqCoeffVerAntIntIdx = zeros(nOccupiedBoxes, 1);
    eqCoeffHorAntIntIdx = zeros(nOccupiedBoxes, 1);
    eqCoeffs            = zeros(scNumEigs, verNumEigs, horNumEigs, nOccupiedBoxes);
  
    for mpIdx = 0 : (nOccupiedBoxes - 1)

            % find max projection coefficent:
            PSD = abs(projCoeffs).^2 .* repmat(eigVal, 1, 1, 1, nOccupiedBoxes);
            PSD = sum(PSD, 1);
            PSD = sum(PSD, 2);
            PSD = sum(PSD, 3);
            PSD = PSD / sum(eigVal(:));
            PSD = reshape(PSD, nOccupiedBoxes, 1);
            PSD = PSD - N0_occupiedBoxes;
            PSD(eqCoeffBoxIdx + 1) = 0;

            [maxP, maxP_idx] = max(PSD);
            maxP_idx = maxP_idx - 1;

            % check for early exit:
            upperBound = 3*N0;
            if((maxP < upperBound) && (mpIdx > 0))
                break;
            end

            % save interval indicies:
            eqCoeffBoxIdx(nEqBoxes + 1)       = maxP_idx;
            eqCoeffFreqIntIdx(nEqBoxes + 1)   = projCoeffFreqIntIdx(maxP_idx + 1);
            eqCoeffVerAntIntIdx(nEqBoxes + 1) = projCoeffVerAntIntIdx(maxP_idx + 1);
            eqCoeffHorAntIntIdx(nEqBoxes + 1) = projCoeffHorAntIntIdx(maxP_idx + 1);

            % compute new equalization coefficents:
            N0_ant          = noiseAndIntEnergy(eqCoeffVerAntIntIdx(nEqBoxes + 1) + 1, eqCoeffHorAntIntIdx(nEqBoxes + 1) + 1);
            updatePsd       = PSD(maxP_idx + 1);
            updatePsd       = (updatePsd - N0_ant) / totalDim;
            updateProjCoeff = projCoeffs(:,:,:,maxP_idx + 1);
            updateEqCoeff   = ((updatePsd * eigVal) ./ (updatePsd * eigVal + N0_ant)) .* updateProjCoeff;

            % update project coefficents:
            projCoeffs = update_projCoeffs(projCoeffs, updateEqCoeff, zeros(scNumEigs, verNumEigs, horNumEigs), ...
                                            projCoeffFreqIntIdx, projCoeffVerAntIntIdx, projCoeffHorAntIntIdx, eqCoeffFreqIntIdx(nEqBoxes + 1), eqCoeffVerAntIntIdx(nEqBoxes + 1), ...
                                            eqCoeffHorAntIntIdx(nEqBoxes + 1), scCorr, verAntCorr, horAntCorr, nOccupiedBoxes, scNumEigs, verNumEigs, horNumEigs);

            % save updated box:
            eqCoeffFreqIntIdx(nEqBoxes + 1)   = projCoeffFreqIntIdx(maxP_idx + 1);
            eqCoeffVerAntIntIdx(nEqBoxes + 1) = projCoeffVerAntIntIdx(maxP_idx + 1);
            eqCoeffHorAntIntIdx(nEqBoxes + 1) = projCoeffHorAntIntIdx(maxP_idx + 1);
            eqCoeffs(:,:,:,nEqBoxes + 1)      = updateEqCoeff;
            nEqBoxes = nEqBoxes + 1;
    end

    eqCoeffFreqIntIdx   = eqCoeffFreqIntIdx(1 : nEqBoxes);
    eqCoeffVerAntIntIdx = eqCoeffVerAntIntIdx(1 : nEqBoxes);
    eqCoeffHorAntIntIdx = eqCoeffHorAntIntIdx(1 : nEqBoxes);
    eqCoeffs            = eqCoeffs(:, :, :, 1 : nEqBoxes);

    %% STEP 5: channel reconstruction

    H_est = zeros(nVerAnt, nHorAnt, nSrsSc);

    for freqEigIdx = 0 : (scNumEigs - 1)
        for verAntEigIdx = 0 : (verNumEigs - 1)
            for horAntEigIdx = 0 : (horNumEigs - 1)

                eigVec = kron(scEigVec(:,freqEigIdx + 1), horAntEigVec(:, horAntEigIdx + 1));
                eigVec = kron(eigVec, verAntEigVec(:, verAntEigIdx + 1));
                eigVec = reshape(eigVec, nVerAnt, nHorAnt, nSrsSc);

                h_eigVec = zeros(nVerAnt, nHorAnt, nZpSrsSc);
                for boxIdx = 0 : (nEqBoxes - 1)
                    h_eigVec(eqCoeffVerAntIntIdx(boxIdx + 1) + 1, eqCoeffHorAntIntIdx(boxIdx + 1) + 1, eqCoeffFreqIntIdx(boxIdx + 1) + 1) = ...
                        squeeze(eqCoeffs(freqEigIdx + 1, verAntEigIdx + 1, horAntEigIdx + 1, boxIdx + 1));
                end
                H_eigVec = fftn(h_eigVec, [nVerAnt nHorAnt nZpSrsSc]);

                H_est = H_est + eigVec .* H_eigVec(:,:,1:nSrsSc);
            end
        end
    end

    %% STEP 6: sub-sample for output
    
    nPrbGroups      = 272 / prgSize;
    nSrsScPerPrbGrp = prgSize * 3;
    srsScOffset     = round((prgSize * 6 - combOffset) / 4);

    outputIdxs = 0 : (nPrbGroups - 1);
    outputIdxs = outputIdxs * nSrsScPerPrbGrp + srsScOffset;

    % load('H_B.mat');
    % E = abs(H - H_est).^2;
    % S = abs(H).^2;
    % snr = 10*log10(mean(S(:)) / mean(E(:)));

    H_est = H_est(:, :, outputIdxs + 1);
    H_est = reshape(H_est, nAnt, nPrbGroups);

    %% STEP 7: estimate energy
    
    E         = abs(H_est).^2;
    avgEnergy = mean(E(:));

    %% STEP 8: save to buffer

    H_est = permute(H_est, [2 1]);

    antIdx = 0 : (nAnt - 1);
    antIdx = 2 * antIdx;
    srsRxDatabase{ueIdx + 1}.Hest(1 : nPrbGroups, antIdx + polIdx + 1, portToUeAntMap(portIdx + 1) + 1) = H_est;
    
    srsRxDatabase{ueIdx + 1}.widebandNoiseEnergy  = srsRxDatabase{ueIdx + 1}.widebandNoiseEnergy + N0;
    srsRxDatabase{ueIdx + 1}.widebandSignalEnergy = srsRxDatabase{ueIdx + 1}.widebandSignalEnergy + avgEnergy;

end

%%
function srsRxDatabase = srsComputeBlock(computeBlockDescriptor, srsUeDescriptors, srsUeGroupDescriptors, srsRxDatabase, Xtf_srs, srsTable)

    % Function performs SRS ChEst for a single hop, over small block of frequency (4 prbs)
    global SimCtrl;
    % constants:
    MAX_N_SC            = 24;
    N_SC_PER_PRB        = 12;
    MAX_N_ANT_PORT      = 12;
    N_PRB_PER_COMP_BLK  = 4;
    
    LOW_PAPR_TABLE_0  = srsTable.srsLowPaprTable0;
    LOW_PAPR_TABLE_1  = srsTable.srsLowPaprTable1;
    FOOC_TABLE_COMB_2 = srsTable.srsFocc_comb2;
    FOOC_TABLE_COMB_4 = srsTable.srsFocc_comb4;

    W_COMB2_N_PORTS_1_WIDE = srsTable.W_comb2_nPorts1_wide;
    W_COMB2_N_PORTS_2_WIDE = srsTable.W_comb2_nPorts2_wide;
    W_COMB2_N_PORTS_4_WIDE = srsTable.W_comb2_nPorts4_wide;
    W_COMB2_N_PORTS_8_WIDE = srsTable.W_comb2_nPorts8_wide;

    W_COMB4_N_PORTS_1_WIDE  = srsTable.W_comb4_nPorts1_wide;
    W_COMB4_N_PORTS_2_WIDE  = srsTable.W_comb4_nPorts2_wide;
    W_COMB4_N_PORTS_4_WIDE  = srsTable.W_comb4_nPorts4_wide;
    W_COMB4_N_PORTS_6_WIDE  = srsTable.W_comb4_nPorts6_wide;
    W_COMB4_N_PORTS_12_WIDE = srsTable.W_comb4_nPorts12_wide;
    
    W_COMB2_N_PORTS_1_NARROW = srsTable.W_comb2_nPorts1_narrow;
    W_COMB2_N_PORTS_2_NARROW = srsTable.W_comb2_nPorts2_narrow;
    W_COMB2_N_PORTS_4_NARROW = srsTable.W_comb2_nPorts4_narrow;
    W_COMB2_N_PORTS_8_NARROW = srsTable.W_comb2_nPorts8_narrow;

    W_COMB4_N_PORTS_1_NARROW  = srsTable.W_comb4_nPorts1_narrow;
    W_COMB4_N_PORTS_2_NARROW  = srsTable.W_comb4_nPorts2_narrow;
    W_COMB4_N_PORTS_4_NARROW  = srsTable.W_comb4_nPorts4_narrow;
    W_COMB4_N_PORTS_6_NARROW  = srsTable.W_comb4_nPorts6_narrow;
    W_COMB4_N_PORTS_12_NARROW = srsTable.W_comb4_nPorts12_narrow;

    NOISE_EST_DEBIAS_COMB2_PORTS1 = srsTable.noisEstDebias_comb2_nPorts1;
    NOISE_EST_DEBIAS_COMB2_PORTS2 = srsTable.noisEstDebias_comb2_nPorts2;
    NOISE_EST_DEBIAS_COMB2_PORTS4 = srsTable.noisEstDebias_comb2_nPorts4;
    NOISE_EST_DEBIAS_COMB2_PORTS8 = srsTable.noisEstDebias_comb2_nPorts8;

    NOISE_EST_DEBIAS_COMB4_PORTS1  = srsTable.noisEstDebias_comb4_nPorts1;
    NOISE_EST_DEBIAS_COMB4_PORTS2  = srsTable.noisEstDebias_comb4_nPorts2;
    NOISE_EST_DEBIAS_COMB4_PORTS4  = srsTable.noisEstDebias_comb4_nPorts4;
    NOISE_EST_DEBIAS_COMB4_PORTS6  = srsTable.noisEstDebias_comb4_nPorts6;
    NOISE_EST_DEBIAS_COMB4_PORTS12 = srsTable.noisEstDebias_comb4_nPorts12;



    % compute block paramaters:
    ueGroupIdx    = computeBlockDescriptor.ueGroupIdx;
    blockStartPrb = computeBlockDescriptor.blockStartPrb;
    nRxAntSrs     = computeBlockDescriptor.nRxAnts;
    blockStartAnt = computeBlockDescriptor.blockStartAnt;

    % paramaters common to users in compute block:
    ueGroupDescriptor     = srsUeGroupDescriptors(ueGroupIdx + 1);
    nUes                  = ueGroupDescriptor.nUes;
    firstUeInBlockIdx     = ueGroupDescriptor.ueIdxs(1);
    firstUeInBlockCombIdx = ueGroupDescriptor.ueCombIdxs(1);
    firstUeInBlockHopIdx  = ueGroupDescriptor.ueHopIdxs(1);
    nAntPorts             = ueGroupDescriptor.nAntPorts;

    repSymIdxs       = srsUeDescriptors(firstUeInBlockIdx + 1).repSymIdxs(:, firstUeInBlockHopIdx + 1);
    ueStartSrsSym    = srsUeDescriptors(firstUeInBlockIdx + 1).repSymIdxs(1, 1);
    hopStartPrb      = srsUeDescriptors(firstUeInBlockIdx + 1).hopStartPrbs(firstUeInBlockHopIdx + 1);
    nRepPerHop       = srsUeDescriptors(firstUeInBlockIdx + 1).nRepPerHop(firstUeInBlockHopIdx + 1);
    nPrbsPerHop      = srsUeDescriptors(firstUeInBlockIdx + 1).nPrbsPerHop;
    u                = srsUeDescriptors(firstUeInBlockIdx + 1).u;
    q                = srsUeDescriptors(firstUeInBlockIdx + 1).q;
    alphaCommon      = srsUeDescriptors(firstUeInBlockIdx + 1).alphaCommon;
    lowPaprTableIdx  = srsUeDescriptors(firstUeInBlockIdx + 1).lowPaprTableIdx;
    lowPaprPrime     = srsUeDescriptors(firstUeInBlockIdx + 1).lowPaprPrime;
    combSize         = srsUeDescriptors(firstUeInBlockIdx + 1).combSize;
    combOffset       = srsUeDescriptors(firstUeInBlockIdx + 1).combOffsets(firstUeInBlockCombIdx + 1);
    nCombScPerPrb    = srsUeDescriptors(firstUeInBlockIdx + 1).nCombScPerPrb;
    nPortsPerComb    = srsUeDescriptors(firstUeInBlockIdx + 1).nPortsPerComb;
    prgSize          = srsUeDescriptors(firstUeInBlockIdx + 1).prgSize;
    n_SRS_cs_max     = srsUeDescriptors(firstUeInBlockIdx + 1).n_SRS_cs_max;

    % update nPrbGrps per compute block
    nPrgPerCompBlk  = N_PRB_PER_COMP_BLK / prgSize;

   %% SETUP
   
    % pick ChEst filter
    switch combSize
        case 2
            if(nAntPorts == 1)
                W_wide         = W_COMB2_N_PORTS_1_WIDE;
                W_narrow       = W_COMB2_N_PORTS_1_NARROW;
                noiseEstDebias = NOISE_EST_DEBIAS_COMB2_PORTS1;
            elseif(nAntPorts == 2)
                W_wide         = W_COMB2_N_PORTS_2_WIDE;
                W_narrow       = W_COMB2_N_PORTS_2_NARROW;
                noiseEstDebias = NOISE_EST_DEBIAS_COMB2_PORTS2;
            elseif(nAntPorts <= 4)
                W_wide         = W_COMB2_N_PORTS_4_WIDE;
                W_narrow       = W_COMB2_N_PORTS_4_NARROW;
                noiseEstDebias = NOISE_EST_DEBIAS_COMB2_PORTS4;
            else
                W_wide         = W_COMB2_N_PORTS_8_WIDE;
                W_narrow       = W_COMB2_N_PORTS_8_NARROW;
                noiseEstDebias = NOISE_EST_DEBIAS_COMB2_PORTS8;
            end
        case 4
            if(nAntPorts == 1)
                W_wide         = W_COMB4_N_PORTS_1_WIDE;
                W_narrow       = W_COMB4_N_PORTS_1_NARROW;
                noiseEstDebias = NOISE_EST_DEBIAS_COMB4_PORTS1;
            elseif(nAntPorts == 2)
                W_wide         = W_COMB4_N_PORTS_2_WIDE;
                W_narrow       = W_COMB4_N_PORTS_2_NARROW;
                noiseEstDebias = NOISE_EST_DEBIAS_COMB4_PORTS2;
            elseif(nAntPorts <= 4)
                W_wide         = W_COMB4_N_PORTS_4_WIDE;
                W_narrow       = W_COMB4_N_PORTS_4_NARROW;
                noiseEstDebias = NOISE_EST_DEBIAS_COMB4_PORTS4;
            elseif(nAntPorts <= 6)
                W_wide         = W_COMB4_N_PORTS_6_WIDE;
                W_narrow       = W_COMB4_N_PORTS_6_NARROW;
                noiseEstDebias = NOISE_EST_DEBIAS_COMB4_PORTS6;
            else
                W_wide         = W_COMB4_N_PORTS_12_WIDE;
                W_narrow       = W_COMB4_N_PORTS_12_NARROW;
                noiseEstDebias = NOISE_EST_DEBIAS_COMB4_PORTS12;
            end
    end

   switch combSize
       case 2
            FOCC_TABLE  = FOOC_TABLE_COMB_2;
            FOCC_LENGTH = 8;
       case 4
            FOCC_TABLE  = FOOC_TABLE_COMB_4;
            FOCC_LENGTH = 12;
   end
             
    % number of srs subcarrier in block:
    nSrsScBlock = 24;
    if(combSize == 4)
        nSrsScBlock = 12;
    end
    
    %% STEP 1: Load Rx SRS subcarriers, remove ZC cover-code, average repititions.
    
    rxSrs = zeros(MAX_N_SC, nRxAntSrs);
    
    for antIdx = 0 : (nRxAntSrs - 1)
        for scIdx = 0 : (nSrsScBlock - 1)
           ZcScIdx   = scIdx + nCombScPerPrb * (blockStartPrb - hopStartPrb);
           loadScIdx = N_SC_PER_PRB * blockStartPrb + scIdx * combSize + combOffset;

           for repIdx = 0 : (nRepPerHop - 1)
               symIdx = repSymIdxs(repIdx + 1);
               
               % extract subcarrier for this repetition:
               y = Xtf_srs(loadScIdx + 1, symIdx + 1, blockStartAnt + antIdx + 1);
               
               % compute subcarrier ZC coverCode for this repetition:
               conj_r = 0;
               if(lowPaprTableIdx == 0)
                   u_repIdx = u(symIdx - ueStartSrsSym + 1);
                   conj_r   = exp(-1i*(pi*LOW_PAPR_TABLE_0(u_repIdx + 1 , ZcScIdx + 1) / 4 + alphaCommon*scIdx));
               elseif(lowPaprTableIdx == 1)
                   u_repIdx = u(symIdx - ueStartSrsSym + 1);
                   conj_r   = exp(-1i*(pi*LOW_PAPR_TABLE_1(u_repIdx + 1, ZcScIdx + 1) / 4 + alphaCommon*scIdx));
               else
                   q_repIdx = q(symIdx - ueStartSrsSym + 1);
                   m        = mod(ZcScIdx, lowPaprPrime);
                   conj_r   = exp(-1i*(- pi*q_repIdx*m*(m+1)/lowPaprPrime + alphaCommon*scIdx));
               end
               
               % remove ZC coverCode and add:
               rxSrs(scIdx + 1, antIdx + 1) = rxSrs(scIdx + 1, antIdx + 1) + conj_r * y;
           end
           
           % normalize:
           rxSrs(scIdx + 1, antIdx + 1) = rxSrs(scIdx + 1, antIdx + 1) / nRepPerHop;
        end
    end
    
    %% STEP 2: remove cyclic shifts and apply wide filter to estimate channel
    
    Hest          = zeros(MAX_N_ANT_PORT, MAX_N_SC, nRxAntSrs);
    antPortOffset = 0;

    for ueInCompBlockIdx = 0 : (nUes - 1)
        ueIdx          = ueGroupDescriptor.ueIdxs(ueInCompBlockIdx + 1);
        combIdx        = ueGroupDescriptor.ueCombIdxs(ueInCompBlockIdx + 1);
        nPortsPerComb  = srsUeDescriptors(ueIdx + 1).nPortsPerComb;
        portToFoccMap  = srsUeDescriptors(ueIdx + 1).portToFoccMap(:, combIdx + 1);

        for antIdx = 0 : (nRxAntSrs - 1)
            useSpeedupCode = 1;
            if useSpeedupCode == 0 % original code
                for scIdx = 0 : (nSrsScBlock - 1)
                    for portIdx = 0 : (nPortsPerComb - 1)
                        foocIdx = portToFoccMap(portIdx + 1);

                        for inputScIdx = 0 : (nSrsScBlock - 1)
                            Hest(portIdx + antPortOffset + 1, scIdx + 1, antIdx + 1) = Hest(portIdx + antPortOffset + 1, scIdx + 1, antIdx + 1) + ...
                                W_wide(scIdx + 1, inputScIdx + 1) * conj(FOCC_TABLE(mod(inputScIdx,FOCC_LENGTH) + 1, foocIdx + 1)) * rxSrs(inputScIdx + 1, antIdx + 1);
                        end
                    end
                end
            else % speedup code equivalent to original code
                for portIdx = 0 : (nPortsPerComb - 1)
                    foocIdx = portToFoccMap(portIdx + 1);
                    scIdx = 0 : (nSrsScBlock - 1);
                    inputScIdx = 0 : (nSrsScBlock - 1);
                    term1 = W_wide(scIdx + 1, inputScIdx + 1);
                    term2 = conj(FOCC_TABLE(mod(inputScIdx,FOCC_LENGTH) + 1, foocIdx + 1));
                    term3 = rxSrs(inputScIdx + 1, antIdx + 1);
                    term4 = term2.*term3;
                    Hest(portIdx + antPortOffset + 1, scIdx + 1, antIdx + 1) = (term1*term4).';
                end
            end
        end
        antPortOffset = antPortOffset + nPortsPerComb;
    end
    
    %% STEP 3: estimate delay phase ramp
    
    phaseRamp     = zeros(MAX_N_ANT_PORT, 1);
    avgScCorr     = zeros(MAX_N_ANT_PORT, 1);
    antPortOffset = 0;

    for ueInCompBlockIdx = 0 : (nUes - 1)
        ueIdx          = ueGroupDescriptor.ueIdxs(ueInCompBlockIdx + 1);
        nPortsPerComb  = srsUeDescriptors(ueIdx + 1).nPortsPerComb;

        for antIdx = 0 : (nRxAntSrs - 1)
            for scIdx = 0 : (nSrsScBlock - 2)
                for portIdx = 0 : (nPortsPerComb - 1)
                    avgScCorr(ueInCompBlockIdx + 1) = avgScCorr(ueInCompBlockIdx + 1) + conj(Hest(antPortOffset + portIdx + 1, scIdx + 1, antIdx + 1)) * Hest(antPortOffset + portIdx + 1, scIdx + 2, antIdx + 1);
                end
            end
        end

        antPortOffset                   = antPortOffset + nPortsPerComb;
        avgScCorr(ueInCompBlockIdx + 1) = avgScCorr(ueInCompBlockIdx + 1) / (nRxAntSrs * (nSrsScBlock - 1) * nPortsPerComb);
        phaseRamp(ueInCompBlockIdx + 1) = atan(imag(avgScCorr(ueInCompBlockIdx + 1)) / real(avgScCorr(ueInCompBlockIdx + 1))) / combSize;
    end

        
%     %% STEP 4: remove delay phase ramp from rx signal
% 
%     for antIdx = 0 : (nRxAntSrs - 1)
%         for scIdx = 0 : (nSrsScBlock - 1)
% %             scIdx_global = scIdx * combSize + N_SC_PER_PRB * blockStartPrb + combOffset;
%             scIdx_global = scIdx * combSize;
%             phase_conj   = exp(-1i*phaseRamp*scIdx_global);            
%             rxSrs(scIdx + 1, antIdx + 1) = phase_conj * rxSrs(scIdx + 1, antIdx + 1);
%         end
%     end
    
    %% STEP 5: remove cyclic shifts and apply narrow filter to estimate channel

    antPortOffset = 0;

    for ueInCompBlockIdx = 0 : (nUes - 1)
        ueIdx          = ueGroupDescriptor.ueIdxs(ueInCompBlockIdx + 1);
        combIdx        = ueGroupDescriptor.ueCombIdxs(ueInCompBlockIdx + 1);
        nPortsPerComb  = srsUeDescriptors(ueIdx + 1).nPortsPerComb;
        portToFoccMap  = srsUeDescriptors(ueIdx + 1).portToFoccMap(:, combIdx + 1);

        for antIdx = 0 : (nRxAntSrs - 1)
            useSpeedupCode = 1;
            if useSpeedupCode == 0 % original code
                for scIdx = 0 : (nSrsScBlock - 1)
                    for portIdx = 0 : (nPortsPerComb - 1)
                        foocIdx = portToFoccMap(portIdx + 1);

                        Hest(antPortOffset + portIdx + 1, scIdx + 1, antIdx + 1) = 0;
                        for inputScIdx = 0 : (nSrsScBlock - 1)
                            scIdx_global = inputScIdx * combSize;
                            phase_conj   = exp(-1i * phaseRamp(ueInCompBlockIdx + 1) * scIdx_global);

                            Hest(antPortOffset + portIdx + 1, scIdx + 1, antIdx + 1) = Hest(antPortOffset + portIdx + 1, scIdx + 1, antIdx + 1) + ...
                                W_narrow(scIdx + 1, inputScIdx + 1) * conj(FOCC_TABLE(mod(inputScIdx,FOCC_LENGTH) + 1, foocIdx + 1)) * phase_conj * rxSrs(inputScIdx + 1, antIdx + 1);
                        end
                    end
                end
            else % speedup code equivalent to original code
                for portIdx = 0 : (nPortsPerComb - 1)
                    foocIdx = portToFoccMap(portIdx + 1);
                    scIdx = 0 : (nSrsScBlock - 1);
                    inputScIdx = 0 : (nSrsScBlock - 1);
                    scIdx_global = inputScIdx * combSize;
                    phase_conj   = exp(-1i * phaseRamp(ueInCompBlockIdx + 1) .* scIdx_global);
                    term1 = W_narrow(scIdx + 1, inputScIdx + 1);
                    term2 = conj(FOCC_TABLE(mod(inputScIdx,FOCC_LENGTH) + 1, foocIdx + 1));
                    term3 = phase_conj.';
                    term4 = rxSrs(inputScIdx + 1, antIdx + 1);
                    term5 = term2.*term3.*term4;
                    Hest(antPortOffset + portIdx + 1, scIdx + 1, antIdx + 1) = (term1*term5).';
                end
            end
        end
        antPortOffset = antPortOffset + nPortsPerComb;
    end
     
    %% STEP 6: average estimates. Estimate energy and noise
    
    avgHest            = zeros(MAX_N_ANT_PORT, nPrgPerCompBlk, nRxAntSrs);
    avgSignalEnergyPrb = zeros(N_PRB_PER_COMP_BLK, MAX_N_ANT_PORT);
    avgNoiseEnergy     = 0;
    avgSignalEnergy    = zeros(MAX_N_ANT_PORT, 1);
    
    for antIdx = 0 : (nRxAntSrs - 1)
        for scIdx = 0 : (nSrsScBlock - 1)
            
            scRxEst     = 0;
            avgHestIdx  = floor(scIdx / (prgSize * nCombScPerPrb));
            prbIdx      = floor(scIdx /  nCombScPerPrb);
            
            antPortOffset = 0;
            for ueInCompBlockIdx = 0 : (nUes - 1)
                ueIdx          = ueGroupDescriptor.ueIdxs(ueInCompBlockIdx + 1);
                combIdx        = ueGroupDescriptor.ueCombIdxs(ueInCompBlockIdx + 1);
                nPortsPerComb  = srsUeDescriptors(ueIdx + 1).nPortsPerComb;
                portToFoccMap  = srsUeDescriptors(ueIdx + 1).portToFoccMap(:, combIdx + 1);

                scIdx_global = scIdx * combSize;
                phase        = exp(1i * phaseRamp(ueInCompBlockIdx + 1) * scIdx_global); 

                for portIdx = 0 : (nPortsPerComb - 1)
                    % apply fOCC to Hest and update scRxEst:
                    foocIdx = portToFoccMap(portIdx + 1);
                    scRxEst = scRxEst + FOCC_TABLE(mod(scIdx,FOCC_LENGTH) + 1, foocIdx + 1) * phase * Hest(antPortOffset + portIdx + 1, scIdx + 1, antIdx + 1);
                    
                    % update average Ests:
                    avgHest(antPortOffset + portIdx + 1, avgHestIdx + 1, antIdx + 1) = avgHest(antPortOffset + portIdx + 1, avgHestIdx + 1, antIdx + 1) + Hest(antPortOffset + portIdx + 1, scIdx + 1, antIdx + 1);
                    
                    % subcarrier energy:
                    scEnergy = abs(Hest(antPortOffset + portIdx + 1, scIdx + 1, antIdx + 1))^2;

                    % update average signal energy:
                    avgSignalEnergyPrb(prbIdx + 1, ueInCompBlockIdx + 1) = avgSignalEnergyPrb(prbIdx + 1, ueInCompBlockIdx + 1) + scEnergy;
                    avgSignalEnergy(ueInCompBlockIdx + 1)                = avgSignalEnergy(ueInCompBlockIdx + 1) + scEnergy; 
                end
                antPortOffset = antPortOffset + nPortsPerComb;
            end
                
            % update block averaged signal/noise energy:
            avgNoiseEnergy  = avgNoiseEnergy + abs(scRxEst - rxSrs(scIdx + 1, antIdx + 1))^2;
        end
    end
    
    % normalize user signal energy:
    for ueInCompBlockIdx = 0 : (nUes - 1)
        ueIdx          = ueGroupDescriptor.ueIdxs(ueInCompBlockIdx + 1);
        nPortsPerComb  = srsUeDescriptors(ueIdx + 1).nPortsPerComb;

        avgSignalEnergy(ueInCompBlockIdx + 1) = avgSignalEnergy(ueInCompBlockIdx + 1) / (nRxAntSrs * nPortsPerComb * nSrsScBlock);
    end
    
    % debias noise energy:
    avgNoiseEnergy = noiseEstDebias * nRepPerHop * avgNoiseEnergy / (nRxAntSrs * nSrsScBlock);
    if((nPortsPerComb ~= 1) && (combSize == 4)) || ((nPortsPerComb == 4) && (combSize == 2))
        avgNoiseEnergy = max(1e-6, avgNoiseEnergy - mean(avgSignalEnergy(1:nUes))*10^(-3)); % use lower bound 1e-6 to avoid negative avgNoiseEnergy
    end

    
    %% Step 7: calculate correlation w.r.t. cyclic shift in use and not use: sum over, PRB, antenna, cyclic shift
    % Use L2 norm across antennas and compute blocks to reduce dependences

    
    % calculate correlation over cyclic shifts, sum over PRBs
    corrRxSrsAllCs  = zeros(nRxAntSrs, n_SRS_cs_max);
    normFactor = 1.0 / sqrt(nSrsScBlock * nRxAntSrs);
    for antIdx = 0 : (nRxAntSrs - 1)
        for csIdx = 0 : (n_SRS_cs_max - 1)
            for scIdx = 0 : (nSrsScBlock - 1)
                corrRxSrsAllCs(antIdx + 1, csIdx + 1) = corrRxSrsAllCs(antIdx + 1, csIdx + 1) + normFactor * rxSrs(scIdx + 1, antIdx + 1) * exp(-1i*((2*pi*csIdx/n_SRS_cs_max)*scIdx));
            end
        end
    end

    % computeBlockDescriptor.csToUeIdxWithBlockMap;

    avgCsCorrNotUse = 0;
    avgCsCorrInUse  = zeros(nAntPorts, 1);

    for antIdx = 0 : nRxAntSrs - 1
        for csIdx = 0 : n_SRS_cs_max - 1 
            ueIdxWithinBlock = 65535;
            normalizer       = 1;

            for i = 0 : (nUes - 1)
                ueIdx   = ueGroupDescriptor.ueIdxs(i + 1);
                combIdx = ueGroupDescriptor.ueCombIdxs(i + 1);

                nUePorts      = srsUeDescriptors(ueIdx + 1).nPortsPerComb;
                portToFoccMap = srsUeDescriptors(ueIdx + 1).portToFoccMap(:, combIdx + 1);

                for uePortIdx = 0 : (nUePorts - 1)
                    if(portToFoccMap(uePortIdx + 1) == csIdx)
                        ueIdxWithinBlock = i;
                        normalizer       = 1 / nUePorts;
                        break;
                    end
                end
            end

            corrEnergy = abs(corrRxSrsAllCs(antIdx + 1, csIdx+1))^2;
            if(ueIdxWithinBlock == 65535)
                avgCsCorrNotUse = avgCsCorrNotUse + corrEnergy / (n_SRS_cs_max - nAntPorts);
            else
                avgCsCorrInUse(ueIdxWithinBlock + 1) = avgCsCorrInUse(ueIdxWithinBlock + 1) + normalizer * corrEnergy;
            end
        end
    end


    %% STEP 8: save output to buffers
        
    estNormalizer   = 1 / (nCombScPerPrb * prgSize);
    avgHest = reshape(fp16nv(real(avgHest * estNormalizer), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(avgHest * estNormalizer), SimCtrl.fp16AlgoSel), [size(avgHest)]);
    for antIdx = 0 : (nRxAntSrs - 1)
        antPortOffset = 0;
        for ueInCompBlockIdx = 0 : (nUes - 1)
            ueIdx   = ueGroupDescriptor.ueIdxs(ueInCompBlockIdx + 1);
            combIdx = ueGroupDescriptor.ueCombIdxs(ueInCompBlockIdx + 1);
            hopIdx  = ueGroupDescriptor.ueHopIdxs(ueInCompBlockIdx + 1);

            nPortsPerComb       = srsUeDescriptors(ueIdx + 1).nPortsPerComb;
            portToUeAntMap      = srsUeDescriptors(ueIdx + 1).portToUeAntMap(:, combIdx + 1);
            portToL2OutUeAntMap = srsUeDescriptors(ueIdx + 1).portToL2OutUeAntMap(:, combIdx + 1);

            chEstBuffStartPrbGrp = srsUeDescriptors(ueIdx + 1).chEstBuffStartPrbGrp;
            chEstBuffOffset      = floor(blockStartPrb / prgSize) - chEstBuffStartPrbGrp;
            chEstToL2Offset      = floor((blockStartPrb - hopStartPrb) / prgSize) + floor(nPrbsPerHop / prgSize) * hopIdx;

            for portIdx = 0 : (nPortsPerComb - 1)
                for avgHestIdx = 0 : (nPrgPerCompBlk - 1)
                    srsRxDatabase{ueIdx + 1}.Hest(chEstBuffOffset + avgHestIdx + 1, blockStartAnt + antIdx + 1, portToUeAntMap(portIdx + 1) + 1)          = avgHest(antPortOffset + portIdx + 1, avgHestIdx + 1, antIdx + 1);
                    srsRxDatabase{ueIdx + 1}.HestToL2Inner(chEstToL2Offset + avgHestIdx + 1, blockStartAnt + antIdx + 1, portToL2OutUeAntMap(portIdx + 1) + 1) = avgHest(antPortOffset + portIdx + 1, avgHestIdx + 1, antIdx + 1);
                end
            end
            antPortOffset = antPortOffset + nPortsPerComb;
        end
    end
    
    % Compute and save per-prb SNR:
    for ueInCompBlockIdx = 0 : (nUes - 1) 
        ueIdx                  = ueGroupDescriptor.ueIdxs(ueInCompBlockIdx + 1);
        nPortsPerComb          = srsUeDescriptors(ueIdx + 1).nPortsPerComb;
        signalEnergyNormalizer = 1 / (nRxAntSrs * nCombScPerPrb * nPortsPerComb);

        for prbIdx = 0 : (N_PRB_PER_COMP_BLK - 1)
            avgSignalEnergyPrb(prbIdx + 1, ueInCompBlockIdx + 1) = avgSignalEnergyPrb(prbIdx + 1, ueInCompBlockIdx + 1) * signalEnergyNormalizer;
            
            if((combIdx == 0) && (blockStartAnt == 0))
                rbSnr = 10*log10(avgSignalEnergyPrb(prbIdx + 1, ueInCompBlockIdx + 1)) - 10*log10(avgNoiseEnergy);
                srsRxDatabase{ueIdx + 1}.rbSnrs(nPrbsPerHop * hopIdx + blockStartPrb - hopStartPrb + prbIdx + 1) = rbSnr;
            end        
        end

        % Increment wideband noise/signal/SC-correlation:
        srsRxDatabase{ueIdx + 1}.widebandNoiseEnergy  = srsRxDatabase{ueIdx + 1}.widebandNoiseEnergy + avgNoiseEnergy;
        srsRxDatabase{ueIdx + 1}.widebandSignalEnergy = srsRxDatabase{ueIdx + 1}.widebandSignalEnergy + avgSignalEnergy(ueInCompBlockIdx + 1);
        srsRxDatabase{ueIdx + 1}.widebandScCorr       = srsRxDatabase{ueIdx + 1}.widebandScCorr + avgScCorr(ueInCompBlockIdx + 1);
    
        % Increment correlation of CS in use and CS not use
        srsRxDatabase{ueIdx + 1}.widebandCsCorrUse    = srsRxDatabase{ueIdx + 1}.widebandCsCorrUse + avgCsCorrInUse(ueInCompBlockIdx + 1);
        srsRxDatabase{ueIdx + 1}.widebandCsCorrNotUse = srsRxDatabase{ueIdx + 1}.widebandCsCorrNotUse + avgCsCorrNotUse;
    end
end

%% Finalize SRS output

function srsRxDatabase = finalizeSrsOutput(nSrsUes, srsRxDatabase, srsUeDescriptors)
    for ueIdx = 0 : (nSrsUes - 1)
        % wideband snr:
        srsRxDatabase{ueIdx + 1}.widebandSnr = 10*log10(srsRxDatabase{ueIdx + 1}.widebandSignalEnergy) - 10*log10(srsRxDatabase{ueIdx + 1}.widebandNoiseEnergy);
        
        % timing advance:
        mu             = srsUeDescriptors(ueIdx + 1).mu;
        scs            = 2^mu * 15*10^3;
        combSize       = srsUeDescriptors(ueIdx + 1).combSize;
        widebandScCorr = srsRxDatabase{ueIdx + 1}.widebandScCorr;
        
        srsRxDatabase{ueIdx + 1}.toEstMicroSec = -10^6 * atan(imag(widebandScCorr) / real(widebandScCorr)) / (2*pi*scs*combSize);  
        srsRxDatabase{ueIdx + 1}.nUniqueHops   = srsUeDescriptors(ueIdx + 1).nUniqueHops;
        srsRxDatabase{ueIdx + 1}.nPrbsPerHop   = srsUeDescriptors(ueIdx + 1).nPrbsPerHop;
        srsRxDatabase{ueIdx + 1}.hopStartPrbs  = srsUeDescriptors(ueIdx + 1).hopStartPrbs;

        % check correlation over other CS based on nPortsPerComb
        srsRxDatabase{ueIdx + 1}.widebandCsCorrRatioDb = 10*log10(srsRxDatabase{ueIdx + 1}.widebandCsCorrUse / srsRxDatabase{ueIdx + 1}.widebandCsCorrNotUse);
        
        
        prgSize   = srsRxDatabase{ueIdx + 1}.prgSize;
        prgSizeL2 = srsRxDatabase{ueIdx + 1}.prgSizeL2;

        if prgSizeL2 > 4  
            % down-selection
            sz = size(srsRxDatabase{ueIdx + 1}.HestToL2);
            prgSizeRatio = prgSizeL2 / prgSize;
            prgSizeOffset = floor(prgSizeRatio / 2);
            for prgIdx = 0 : sz(1)-1
                srsRxDatabase{ueIdx + 1}.HestToL2(prgIdx+1,:,:) = srsRxDatabase{ueIdx + 1}.HestToL2Inner(prgSizeOffset+prgIdx*prgSizeRatio,:,:);
            end

        else
            srsRxDatabase{ueIdx + 1}.HestToL2 = srsRxDatabase{ueIdx + 1}.HestToL2Inner;
        end

        global SimCtrl;
        if(SimCtrl.alg.srs_chEst_toL2_normalization_algo_selector == 0)
            srsRxDatabase{ueIdx + 1}.HestNormToL2 = srsRxDatabase{ueIdx + 1}.HestToL2*SimCtrl.alg.srs_chEst_toL2_constant_scaler;
        elseif(SimCtrl.alg.srs_chEst_toL2_normalization_algo_selector == 1)
            srsRxDatabase{ueIdx + 1}.HestNormToL2 = srsRxDatabase{ueIdx + 1}.HestToL2 / max(abs(srsRxDatabase{ueIdx + 1}.HestToL2(:))) * 32768;
        end
    end
end


%% Initialize srs output database

function [srsRxDatabase, SrsParamsList] = init_srs_rx_dataBase(nSrsUes, SrsParamsList, cellParams)
    nRxAntSrs = cellParams.nRxAntSrs;
    % TODO: using max nPrbs for buffer allocation to avoid dimension mismatch in cuPHY ref check
    nPrbs     = 273; %cellParams.nPrbs
    
    srsRxDatabase = cell(nSrsUes, 1);
    for srsUeIdx = 0 : (nSrsUes - 1)
        nUeAnt = SrsParamsList(srsUeIdx + 1).nAntPorts;
        % TODO: ignoring the last "parital" PRBG; i.e., 273 PRBs, prgSize = 2 -> 136 PRBGs
        nPrbGrps  = floor(nPrbs / SrsParamsList(srsUeIdx + 1).prgSize) ; % maximum nPrbGrps

        ueSrsRx                      = [];
        ueSrsRx.rbSnrs               = zeros(nPrbs, 1);
        ueSrsRx.Hest                 = zeros(nPrbGrps, nRxAntSrs, nUeAnt);
        ueSrsRx.HestToL2Inner        = [];
        ueSrsRx.HestToL2             = [];
        ueSrsRx.HestNormToL2         = [];
        ueSrsRx.toEstMicroSec        = 0;
        ueSrsRx.widebandSignalEnergy = 0;
        ueSrsRx.widebandNoiseEnergy  = 0;
        ueSrsRx.widebandSnr          = 0;
        ueSrsRx.widebandScCorr       = 0;
        ueSrsRx.widebandCsCorrUse    = 0;
        ueSrsRx.widebandCsCorrNotUse = 0;
        ueSrsRx.widebandCsCorrRatioDb= 0;
        ueSrsRx.prgSizeL2            = 0;
        ueSrsRx.prgSize              = 0;
        ueSrsRx.nPrbGrpsL2           = 0;
        ueSrsRx.nPrbGrps             = 0;
        ueSrsRx.startValidPrg        = 0;
        
        ueSrsRx.chEstBuffer_nRxAntSrs   = nRxAntSrs;
        ueSrsRx.chEstBuffer_nPrbGrps    = nPrbGrps;
        ueSrsRx.chEstBuffer_nUeAnt      = nUeAnt;
        ueSrsRx.chEstBuffer_startPrbGrp = 0;
        
        srsRxDatabase{srsUeIdx + 1} = ueSrsRx;
    end
end



%%
function saveTV_srs(tvDirName, TVname,  SrsParamsList, Xtf_srs, SrsOutputList, cellParams, srsTable)

    global SimCtrl
    [status,msg] = mkdir(tvDirName);
    h5File       = H5F.create([tvDirName filesep TVname '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
    nUes         = length(SrsParamsList);

    % save input buffer:
    % Xtf_srs = reshape(fp16nv(real(Xtf_srs), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(Xtf_srs), SimCtrl.fp16AlgoSel), [size(Xtf_srs)]);  % alreayd done in gNBreceiver.m
    
    hdf5_write_nv(h5File, 'DataRx', complex(single(Xtf_srs)));
    
    % Save cuPHY SRS UE paramaters:
    srsUePrms = [];
    for ueIdx = 0 : (nUes - 1)
        srsUePrms(ueIdx + 1).nAntPorts                = uint8(SrsParamsList(ueIdx + 1).nAntPorts);
        srsUePrms(ueIdx + 1).nSyms                    = uint8(SrsParamsList(ueIdx + 1).nSyms);
        srsUePrms(ueIdx + 1).nRepetitions             = uint8(SrsParamsList(ueIdx + 1).nRepetitions);
        srsUePrms(ueIdx + 1).combSize                 = uint8(SrsParamsList(ueIdx + 1).combSize);
        srsUePrms(ueIdx + 1).startSym                 = uint8(SrsParamsList(ueIdx + 1).startSym);
        srsUePrms(ueIdx + 1).sequenceId               = uint16(SrsParamsList(ueIdx + 1).sequenceId);
        srsUePrms(ueIdx + 1).configIdx                = uint8(SrsParamsList(ueIdx + 1).configIdx);
        srsUePrms(ueIdx + 1).bandwidthIdx             = uint8(SrsParamsList(ueIdx + 1).bandwidthIdx);
        srsUePrms(ueIdx + 1).combOffset               = uint8(SrsParamsList(ueIdx + 1).combOffset);
        srsUePrms(ueIdx + 1).cyclicShift              = uint8(SrsParamsList(ueIdx + 1).cyclicShift);
        srsUePrms(ueIdx + 1).frequencyPosition        = uint8(SrsParamsList(ueIdx + 1).frequencyPosition);
        srsUePrms(ueIdx + 1).frequencyShift           = uint16(SrsParamsList(ueIdx + 1).frequencyShift);
        srsUePrms(ueIdx + 1).frequencyHopping         = uint8(SrsParamsList(ueIdx + 1).frequencyHopping);
        srsUePrms(ueIdx + 1).resourceType             = uint8(SrsParamsList(ueIdx + 1).resourceType);
        srsUePrms(ueIdx + 1).Tsrs                     = uint16(SrsParamsList(ueIdx + 1).Tsrs);
        srsUePrms(ueIdx + 1).Toffset                  = uint16(SrsParamsList(ueIdx + 1).Toffset);
        srsUePrms(ueIdx + 1).groupOrSequenceHopping   = uint8(SrsParamsList(ueIdx + 1).groupOrSequenceHopping);
        srsUePrms(ueIdx + 1).prgSize                  = uint16(SrsParamsList(ueIdx + 1).prgSizeL2);
        srsUePrms(ueIdx + 1).RNTI                     = uint16(SrsParamsList(ueIdx + 1).RNTI);

        srsAntPortToUeAntMap = 0;
        for antPortIdx = 0 : (SrsParamsList(ueIdx + 1).nAntPorts - 1)
            bitShift = 8*antPortIdx;
            srsAntPortToUeAntMap = srsAntPortToUeAntMap + 2^bitShift * SrsParamsList(ueIdx + 1).srsAntPortToUeAntMap(antPortIdx + 1);
        end   
        srsUePrms(ueIdx + 1).srsAntPortToUeAntMap = uint32(srsAntPortToUeAntMap);
    end
    hdf5_write_nv_exp(h5File, 'srsUePrms', srsUePrms);
    
    % save cUPHY SRS cell paramaters:
    srsCellParams              = [];
    srsCellParams.slotNum      = uint16(cellParams.slotNum);
    srsCellParams.frameNum     = uint16(cellParams.frameNum);
    srsCellParams.mu           = uint8(cellParams.mu);
    srsCellParams.nRxAntSrs    = uint16(cellParams.nRxAntSrs);
    srsCellParams.nRxAnt       = uint16(cellParams.nRxAntSrs);
    srsCellParams.srsStartSym  = uint8(cellParams.srsStartSym);
    srsCellParams.nPrbs        = uint16(cellParams.nPrbs);
    srsCellParams.nSrsSym      = uint8(cellParams.nSrsSym);
    srsCellParams.nSrsUes      = uint16(nUes);
    srsCellParams.chEstAlgoIdx = uint8(SimCtrl.alg.srs_chEst_alg_selector);
    srsCellParams.chEstToL2NormalizationAlgo = uint8(SimCtrl.alg.srs_chEst_toL2_normalization_algo_selector);
    srsCellParams.chEstToL2ConstantScaler = single(SimCtrl.alg.srs_chEst_toL2_constant_scaler);

    hdf5_write_nv_exp(h5File, 'srsCellParams', srsCellParams);
        
    % save filters:
    save_C_FP16_table_to_H5(srsTable.srsFocc      , 'focc_table'      , h5File, SimCtrl.fp16AlgoSel);
    save_C_FP16_table_to_H5(srsTable.srsFocc_comb2, 'focc_table_comb2', h5File, SimCtrl.fp16AlgoSel);
    save_C_FP16_table_to_H5(srsTable.srsFocc_comb4, 'focc_table_comb4', h5File, SimCtrl.fp16AlgoSel);

     
    save_C_FP16_table_to_H5(srsTable.W_comb2_nPorts1_wide, 'W_comb2_nPorts1_wide', h5File, SimCtrl.fp16AlgoSel);
    save_C_FP16_table_to_H5(srsTable.W_comb2_nPorts2_wide, 'W_comb2_nPorts2_wide', h5File, SimCtrl.fp16AlgoSel);
    save_C_FP16_table_to_H5(srsTable.W_comb2_nPorts4_wide, 'W_comb2_nPorts4_wide', h5File, SimCtrl.fp16AlgoSel);
    save_C_FP16_table_to_H5(srsTable.W_comb2_nPorts8_wide, 'W_comb2_nPorts8_wide', h5File, SimCtrl.fp16AlgoSel);

    save_C_FP16_table_to_H5(srsTable.W_comb4_nPorts1_wide,  'W_comb4_nPorts1_wide' , h5File, SimCtrl.fp16AlgoSel);
    save_C_FP16_table_to_H5(srsTable.W_comb4_nPorts2_wide,  'W_comb4_nPorts2_wide' , h5File, SimCtrl.fp16AlgoSel);
    save_C_FP16_table_to_H5(srsTable.W_comb4_nPorts4_wide,  'W_comb4_nPorts4_wide' , h5File, SimCtrl.fp16AlgoSel);
    save_C_FP16_table_to_H5(srsTable.W_comb4_nPorts6_wide,  'W_comb4_nPorts6_wide' , h5File, SimCtrl.fp16AlgoSel);
    save_C_FP16_table_to_H5(srsTable.W_comb4_nPorts12_wide, 'W_comb4_nPorts12_wide', h5File, SimCtrl.fp16AlgoSel);

    save_C_FP16_table_to_H5(srsTable.W_comb2_nPorts1_narrow, 'W_comb2_nPorts1_narrow', h5File, SimCtrl.fp16AlgoSel);
    save_C_FP16_table_to_H5(srsTable.W_comb2_nPorts2_narrow, 'W_comb2_nPorts2_narrow', h5File, SimCtrl.fp16AlgoSel);
    save_C_FP16_table_to_H5(srsTable.W_comb2_nPorts4_narrow, 'W_comb2_nPorts4_narrow', h5File, SimCtrl.fp16AlgoSel);
    save_C_FP16_table_to_H5(srsTable.W_comb2_nPorts8_narrow, 'W_comb2_nPorts8_narrow', h5File, SimCtrl.fp16AlgoSel);

    save_C_FP16_table_to_H5(srsTable.W_comb4_nPorts1_narrow,  'W_comb4_nPorts1_narrow' , h5File, SimCtrl.fp16AlgoSel);
    save_C_FP16_table_to_H5(srsTable.W_comb4_nPorts2_narrow,  'W_comb4_nPorts2_narrow' , h5File, SimCtrl.fp16AlgoSel);
    save_C_FP16_table_to_H5(srsTable.W_comb4_nPorts4_narrow,  'W_comb4_nPorts4_narrow' , h5File, SimCtrl.fp16AlgoSel);
    save_C_FP16_table_to_H5(srsTable.W_comb4_nPorts6_narrow,  'W_comb4_nPorts6_narrow' , h5File, SimCtrl.fp16AlgoSel);
    save_C_FP16_table_to_H5(srsTable.W_comb4_nPorts12_narrow, 'W_comb4_nPorts12_narrow', h5File, SimCtrl.fp16AlgoSel);

    debiasPrms = [];
    debiasPrms.noisEstDebias_comb2_nPorts1 = single(srsTable.noisEstDebias_comb2_nPorts1);
    debiasPrms.noisEstDebias_comb2_nPorts2 = single(srsTable.noisEstDebias_comb2_nPorts2);
    debiasPrms.noisEstDebias_comb2_nPorts4 = single(srsTable.noisEstDebias_comb2_nPorts4);
    debiasPrms.noisEstDebias_comb2_nPorts8 = single(srsTable.noisEstDebias_comb2_nPorts8);

    debiasPrms.noisEstDebias_comb4_nPorts1  = single(srsTable.noisEstDebias_comb4_nPorts1);
    debiasPrms.noisEstDebias_comb4_nPorts2  = single(srsTable.noisEstDebias_comb4_nPorts2);
    debiasPrms.noisEstDebias_comb4_nPorts4  = single(srsTable.noisEstDebias_comb4_nPorts4);
    debiasPrms.noisEstDebias_comb4_nPorts6  = single(srsTable.noisEstDebias_comb4_nPorts6);
    debiasPrms.noisEstDebias_comb4_nPorts12 = single(srsTable.noisEstDebias_comb4_nPorts12);

    hdf5_write_nv_exp(h5File, 'debiasPrms', debiasPrms);

      
%     if SimCtrl.fp16AlgoSel == 0
%         FOCC_TABLE = reshape(single(half(real(FOCC_TABLE))) + 1i*single(half(imag(FOCC_TABLE))), [size(FOCC_TABLE)]);
%     else
%         FOCC_TABLE = reshape(single(vfp16(real(FOCC_TABLE))) + 1i*single(vfp16(imag(FOCC_TABLE))), [size(FOCC_TABLE)]);
%     end
%     hdf5_write_nv(h5File, 'FOCC_TABLE', complex(single(FOCC_TABLE)));
%     
%     W_COMB2_N_PORTS_1_WIDE = srsTable.W_comb2_nPorts1_wide;
%     if SimCtrl.fp16AlgoSel == 0
%         W_COMB2_N_PORTS_1_WIDE = reshape(single(half(real(W_COMB2_N_PORTS_1_WIDE))) + 1i*single(half(imag(W_COMB2_N_PORTS_1_WIDE))), [size(W_COMB2_N_PORTS_1_WIDE)]);
%     else
%         W_COMB2_N_PORTS_1_WIDE = reshape(single(vfp16(real(W_COMB2_N_PORTS_1_WIDE))) + 1i*single(vfp16(imag(W_COMB2_N_PORTS_1_WIDE))), [size(W_COMB2_N_PORTS_1_WIDE)]);
%     end
%     hdf5_write_nv(h5File, 'W_COMB2_N_PORTS_1_WIDE', complex(single(W_COMB2_N_PORTS_1_WIDE)));
   
    % save SRS Rx output:
    widebandSrsStats   = [];
    srsChEstBufferInfo = [];
    srsChEstToL2Prms   = [];
    for ueIdx = 0 : (nUes - 1)
        ueSrsRx = SrsOutputList{ueIdx + 1};
        
        % wideband stats:
        widebandSrsStats(ueIdx + 1).widebandNoiseEnergy   = single(ueSrsRx.widebandNoiseEnergy);
        widebandSrsStats(ueIdx + 1).widebandSignalEnergy  = single(ueSrsRx.widebandSignalEnergy);
        widebandSrsStats(ueIdx + 1).widebandSnr           = single(ueSrsRx.widebandSnr);
        widebandSrsStats(ueIdx + 1).toEstMicroSec         = single(ueSrsRx.toEstMicroSec);
        widebandSrsStats(ueIdx + 1).widebandCsCorrUse     = single(ueSrsRx.widebandCsCorrUse);
        widebandSrsStats(ueIdx + 1).widebandCsCorrNotUse  = single(ueSrsRx.widebandCsCorrNotUse);
        widebandSrsStats(ueIdx + 1).widebandCsCorrRatioDb = single(ueSrsRx.widebandCsCorrRatioDb);
        widebandSrsStats(ueIdx + 1).startValidPrg         = uint16(ueSrsRx.startValidPrg);
        widebandSrsStats(ueIdx + 1).nValidPrg             = uint16(ueSrsRx.nPrbGrps);

        % prbSnrs:
        nameStr = strcat('rbSnrsUe',num2str(ueIdx));
        hdf5_write_nv(h5File, nameStr, single(ueSrsRx.rbSnrs));
        
        % SRS ChEst buffer:
        Hest    = ueSrsRx.Hest;
        nameStr = strcat('HestUe',num2str(ueIdx));
        hdf5_write_nv(h5File, nameStr, complex(single(Hest)));
        % Write to buffer using int8+int8 format
        % Normalization
        HestNorm = Hest / max(abs(Hest(:)));
        % Convert to int8
        Hest16bit_R = real(HestNorm) * 128;
        Hest16bit_I = imag(HestNorm) * 128;
        nameStr = strcat('Hest_16bitR_Ue',num2str(ueIdx));
        hdf5_write_nv(h5File, nameStr, int8(Hest16bit_R));
        nameStr = strcat('Hest_16bitI_Ue',num2str(ueIdx));
        hdf5_write_nv(h5File, nameStr, int8(Hest16bit_I));
        % Normalization - same as int8
        % Convert to int32
        Hest32bit_R = real(HestNorm) * 32768;
        Hest32bit_I = imag(HestNorm) * 32768;
        nameStr = strcat('Hest_32bitR_Ue',num2str(ueIdx));
        hdf5_write_nv(h5File, nameStr, int16(Hest32bit_R));
        nameStr = strcat('Hest_32bitI_Ue',num2str(ueIdx));
        hdf5_write_nv(h5File, nameStr, int16(Hest32bit_I));        

        % SRS ChEst report to L2:
        HestToL2 = ueSrsRx.HestToL2;
        nameStr  = strcat('HestToL2Ue',num2str(ueIdx));
        hdf5_write_nv(h5File, nameStr, complex(single(HestToL2)));
        
        HestNormToL2 = ueSrsRx.HestNormToL2;
        nameStr  = strcat('HestNormToL2Ue',num2str(ueIdx));
        hdf5_write_nv(h5File, nameStr, complex(int16(HestNormToL2)));

        srsChEstToL2Prms(ueIdx + 1).nPrbGrps   = ueSrsRx.nPrbGrpsL2;
        srsChEstToL2Prms(ueIdx + 1).prgSize    = ueSrsRx.prgSizeL2;
        
        % SRS ChEst buffer paramaters:
        srsChEstBufferInfo(ueIdx + 1).nRxAnt      = uint16(ueSrsRx.chEstBuffer_nRxAntSrs);
        srsChEstBufferInfo(ueIdx + 1).nPrbGrps    = uint16(ueSrsRx.chEstBuffer_nPrbGrps);
        srsChEstBufferInfo(ueIdx + 1).nUeAnt      = uint8(ueSrsRx.chEstBuffer_nUeAnt);
        srsChEstBufferInfo(ueIdx + 1).startPrbGrp = uint16(ueSrsRx.chEstBuffer_startPrbGrp);   
        
    end
    hdf5_write_nv_exp(h5File, 'srsChEstToL2Prms'  , srsChEstToL2Prms);
    hdf5_write_nv_exp(h5File, 'widebandSrsStats'  , widebandSrsStats);
    hdf5_write_nv_exp(h5File, 'srsChEstBufferInfo', srsChEstBufferInfo);


    % save SRS RKHS paramaters:
    srs_rkhs_tables = srsTable.srs_rkhs_tables;

    srsRkhsPrms        = [];
    rkhsGridPrms       = [];
    srsRkhsPrms.nGrids = uint8(srs_rkhs_tables.rkhsPrms.nGridSizes);

    for gridIdx = 0 : (srsRkhsPrms.nGrids - 1)
        rkhsGridPrms(gridIdx + 1).nEigs      = uint8(srs_rkhs_tables.gridPrms{gridIdx + 1}.nEig);
        rkhsGridPrms(gridIdx + 1).gridSize   = uint16(srs_rkhs_tables.gridPrms{gridIdx + 1}.gridSize);
        rkhsGridPrms(gridIdx + 1).zpGridSize = uint16(srs_rkhs_tables.gridPrms{gridIdx + 1}.zpGridSize);

        str1 = 'srsRkhs_eigenVecs_grid';
        str2 = num2str(gridIdx);
        str  = append(str1, str2);
        buffer = srs_rkhs_tables.eigenVecTable{gridIdx + 1};
        buffer = fp16nv(real(buffer), SimCtrl.fp16AlgoSel);
        hdf5_write_nv(h5File, str, buffer,'fp16');

        str1 = 'srsRkhs_eigValues_grid';
        str2 = num2str(gridIdx);
        str  = append(str1, str2);
        buffer = srs_rkhs_tables.eigValTable{gridIdx + 1};
        buffer = fp16nv(real(buffer), SimCtrl.fp16AlgoSel);
        hdf5_write_nv(h5File, str, buffer,'fp16');

        str1 = 'srsRkhs_eigenCorr_grid';
        str2 = num2str(gridIdx);
        str  = append(str1, str2);
        buffer = srs_rkhs_tables.corrTable{gridIdx + 1};
        buffer = fp16nv(complex(buffer), SimCtrl.fp16AlgoSel);
        hdf5_write_nv(h5File, str, complex(single(buffer)),'fp16');

        str1 = 'srsRkhs_secondStageTwiddleFactors_grid';
        str2 = num2str(gridIdx);
        str  = append(str1, str2);
        buffer = srs_rkhs_tables.secondStageTwiddleFactorsTable{gridIdx + 1};
        buffer = fp16nv(complex(buffer), SimCtrl.fp16AlgoSel);
        hdf5_write_nv(h5File, str, complex(single(buffer)),'fp16');

        str1 = 'srsRkhs_secondStageFourierPerm_grid';
        str2 = num2str(gridIdx);
        str  = append(str1, str2);
        buffer = srs_rkhs_tables.secondStageFourierPermTable{gridIdx + 1};
        buffer = uint8(buffer);
        hdf5_write_nv(h5File, str, buffer);
    end

    hdf5_write_nv(h5File, 'srsRkhsPrms', srsRkhsPrms);
    hdf5_write_nv(h5File, 'rkhsGridPrms', rkhsGridPrms);
    
end

function save_C_FP16_table_to_H5(table, table_name, h5File, fp16AlgoSel)
%     if fp16AlgoSel == 0
%         table = reshape(single(half(real(table))) + 1i*single(half(imag(table))), [size(table)]);
%     else
%         table = reshape(single(vfp16(real(table))) + 1i*single(vfp16(imag(table))), [size(table)]);
%     end
    table = reshape(fp16nv(real(table), fp16AlgoSel) + 1i*fp16nv(imag(table), fp16AlgoSel), [size(table)]);
    hdf5_write_nv(h5File, table_name, complex(single(table)));
end
