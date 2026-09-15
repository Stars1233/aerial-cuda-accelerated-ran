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

function [pUciF3] = pucchF3_rx_kernel(nRxAnt, pucchF3tables, pucchF3dynamicDesc, pUciF3)
%%Currently only support SR, HARQ and CSI part 1

prbConst = 12;

% Load parameters from pucchF3tables
W1                  = pucchF3tables.W1;
W2                  = pucchF3tables.W2;
W3                  = pucchF3tables.W3;
W4                  = pucchF3tables.W4;

% Load number of PF3 UCIs
numUcis = pucchF3dynamicDesc.nGrps;
 
for uciIdx = 1:numUcis
    % Load parameters from pucchF3dynamicDesc
    Xtf                 = pucchF3dynamicDesc.Xtf;
    BWPStart            = pucchF3dynamicDesc.BWPStart{uciIdx};
    freqHopFlag         = pucchF3dynamicDesc.freqHopFlag{uciIdx};
    startSym            = pucchF3dynamicDesc.startSym{uciIdx} + 1;  % Matlab 1-based indexing
    startPrb            = pucchF3dynamicDesc.startPrb{uciIdx};
    nSym                = pucchF3dynamicDesc.nSym{uciIdx};
    prbSize             = pucchF3dynamicDesc.prbSize{uciIdx};    
    secondHopPrb        = pucchF3dynamicDesc.secondHopPrb{uciIdx};
    u                   = pucchF3dynamicDesc.u{uciIdx};
    v                   = pucchF3dynamicDesc.v{uciIdx};
    csCommon            = pucchF3dynamicDesc.csCommon{uciIdx};
    AddDmrsFlag         = pucchF3dynamicDesc.addDmrsFlag{uciIdx};
    cs0                 = 0;
    noiseVar            = pucchF3dynamicDesc.noiseVar{uciIdx};
    GroupHopFlag        = pucchF3dynamicDesc.groupHopFlag{uciIdx};
    sequenceHopFlag     = pucchF3dynamicDesc.sequenceHopFlag{uciIdx};
    pi2Bpsk             = pucchF3dynamicDesc.pi2Bpsk{uciIdx};
    nSymData            = pucchF3dynamicDesc.nSymData{uciIdx};
    nSymDmrs            = pucchF3dynamicDesc.nSymDmrs{uciIdx};
    SetSymData          = pucchF3dynamicDesc.SetSymData{uciIdx};
    SetSymDmrs          = pucchF3dynamicDesc.SetSymDmrs{uciIdx}; 
    randomSeqScrm       = pucchF3dynamicDesc.randomSeqScrm{uciIdx};
    uciOutputIdx        = pucchF3dynamicDesc.uciOutputIdx{uciIdx};
    A_seg1              = pucchF3dynamicDesc.A_seg1{uciIdx};
    E_seg1              = pucchF3dynamicDesc.E_seg1{uciIdx};
    E_seg2              = pucchF3dynamicDesc.E_seg2{uciIdx};
    rankBitOffset       = pucchF3dynamicDesc.rankBitOffset{uciIdx};
    rankBitSize         = pucchF3dynamicDesc.rankBitSize{uciIdx};
    BitLenHarq          = pucchF3dynamicDesc.BitLenHarq{uciIdx};
    BitLenCsiPart1      = pucchF3dynamicDesc.BitLenCsiPart1{uciIdx};
    BitLenSr            = pucchF3dynamicDesc.BitLenSr{uciIdx};
    DTXthreshold        = pucchF3dynamicDesc.DTXthreshold{uciIdx};
    scs                 = pucchF3dynamicDesc.scs;

    startCrb            = startPrb + BWPStart;
    secondHopCrb        = secondHopPrb + BWPStart;
    
    totNumSC = 12*prbSize;
    totNumSym = totNumSC*nSymData;
    
    nSymDataFirstHop = floor(nSymData/2);
    nSymFirstHop = floor(nSym/2);
    nSymDmrsFirstHop = nSymDmrs/2;
    nSymDataSecondHop = nSymData - nSymDataFirstHop;
    nSymDmrsSecondHop = nSymDmrs - nSymDmrsFirstHop;
    
    %%Cyclic shift hopping    
    cs = zeros(nSym,1);
    for i = 1 : nSym
        cs(i) = mod(csCommon(i) + cs0, 12);
    end
    
    % STEP 1 load data from Xtf
    Y_pucch = [];
    x_dmrs = [];
    freq_idx = (startCrb*12+1) : startCrb*12+totNumSC;
    
    if freqHopFlag
        %%First hop
        time_idx = (startSym) : (startSym + nSymFirstHop -1);
        
        Y_pucch = Xtf(freq_idx,time_idx,1:nRxAnt);
        
        %%Second hop
        time_idx = (startSym + nSymFirstHop):(startSym + nSym -1);
        freq_idx = (secondHopCrb*12+1) : secondHopCrb*12+totNumSC;
        
        Y_pucch = [Y_pucch Xtf(freq_idx,time_idx,1:nRxAnt)];
    else
        time_idx = (startSym) : (startSym + nSym -1);
        Y_pucch = Xtf(freq_idx,time_idx,1:nRxAnt);
    end
    
    % STEP 2 seperate pucch signal into dmrs and data signals
    Y_dmrs = Y_pucch(:, 1+SetSymDmrs,:);
    Y_data = Y_pucch(:, 1+SetSymData,:);

    % calculate RSSI
    rssiTemp = sum(abs(Y_dmrs).^2, 1);
    rssiTemp = reshape(rssiTemp, [size(rssiTemp, 2), size(rssiTemp, 3)]); % remove the first dimension which is a frequency dimension
    rssi = sum(mean(rssiTemp,1),2);% average across all symbols and sum across all Rx antenna
    rssi_dB = 10*log10(rssi);
    
    % STEP 3 Generate DMRS
    % Only one DMRS per hop. If multiple DMRS per hop, no error is produced but they are ignored
  
    chEst = [];
    for k = 1:nSymDmrs
        % Sequence generation
        if (k==1 && ~AddDmrsFlag) || (k<=2 && AddDmrsFlag)
            firstHop = true;
        else
            firstHop = false;
        end
        
        if firstHop
        r = LowPaprSeqGen(totNumSC, u(1), v(1));
        elseif ~firstHop && ~GroupHopFlag && ~sequenceHopFlag
            r = LowPaprSeqGen(totNumSC, u(1), v(1));
        elseif ~firstHop && GroupHopFlag && ~sequenceHopFlag
            r = LowPaprSeqGen(totNumSC, u(2), v(1));
        elseif ~firstHop && ~GroupHopFlag && sequenceHopFlag
            r = LowPaprSeqGen(totNumSC, u(1), v(2));
        end
        
        CsIdx = (0:(totNumSC-1))*2*pi/12;
        CsIdx = transpose(CsIdx);
        x_dmrs(:,k) = r.*exp(1i* cs(SetSymDmrs(k)+1) .* CsIdx); 

        % Remove code (multiply)
        for ii=1:size(Y_dmrs,3)
            chEst(:,k,ii) = Y_dmrs(:,k,ii).*conj(x_dmrs(:,k));
        end
    end
    
    % STEP 4 Filter channel estimation coefficients in freq domain
    % Uses filters for 1,2,3,4 PRBs. If more than 4, filter in chunks of 4
    
    if prbSize == 1
        for hh=1:nSymDmrs
            for ii=1:nRxAnt
                chEst(:,hh,ii) = W1 * chEst(:,hh,ii);
            end
        end
    elseif prbSize == 2
        for hh=1:nSymDmrs
            for ii=1:nRxAnt
                chEst(:,hh,ii) = W2 * chEst(:,hh,ii);
            end
        end
    elseif prbSize == 3
        for hh=1:nSymDmrs
            for ii=1:nRxAnt
                chEst(:,hh,ii) = W3 * chEst(:,hh,ii);
            end
        end
    elseif prbSize >=4
        prbBlockSize = 4;
        nPrbBlocks = ceil(prbSize/prbBlockSize);
        
        % loop over antennas
        for hh=1:nSymDmrs
            for ii=1:size(Y_dmrs,3)
                for jj=1:nPrbBlocks
                    if jj*prbBlockSize < prbSize
                        tmp = W4*chEst((jj-1)*prbBlockSize*prbConst+1:jj*prbBlockSize*prbConst,hh,ii);
                        chEst((jj-1)*prbBlockSize*prbConst+1:jj*prbBlockSize*prbConst,hh,ii) = tmp;
                    else
                        % last block overlaps with previous one when prbSize/prbBlockSize
                        % not integer
                        numRowsSkip = (nPrbBlocks*prbBlockSize - prbSize) * prbConst;
                        tmp = W4*chEst((jj-1)*prbBlockSize*prbConst+1-numRowsSkip:jj*prbBlockSize*prbConst-numRowsSkip,hh,ii);
                        chEst((jj-1)*prbBlockSize*prbConst+1:prbSize*prbConst,hh,ii) = tmp(numRowsSkip+1:4*prbConst);
                    end
                end
            end
            % time interpolation: time interpolation is not supported at this time
        end
    end
   
    % STEP 4.5 check SINR to detect DTX. It uses interpolated channel
    % and noise estimation
    % RSRP 
    rsrp = 0;
    for ii=1:prbSize*prbConst
        for jj=1:nSymDmrs
            for kk=1:nRxAnt
                rsrp = rsrp + abs(chEst(ii,jj,kk)).^2;
            end
        end
    end
    rsrp = rsrp / (nSymDmrs*prbSize*prbConst*nRxAnt);
    rsrp_dB = 10*log10(rsrp);
    
    % Noise power
    % Generate DMRS
    % -> it is already available from channel estimation algorithm
    % Generate r_tilde (RX signal without DMRS)
    
    r_tilde = zeros(prbSize*12, nSymDmrs, nRxAnt);
    for ii=1:nRxAnt
        for jj=1:nSymDmrs
            r_tilde(:,jj,ii) = Y_dmrs(:,jj,ii) - x_dmrs(:,jj).*chEst(1:1:prbSize*12,jj,ii);
        end
    end
       
    % Calculate r_tilde power
    % a bias correction factor is added to compensate for noise
    % filtered out in the channel estimation process 
    tmp = abs(r_tilde).^2;
    noiseVar = mean(tmp(:));
    noiseVardB = 10*log10(noiseVar) +0.5;
    
    % Calculate SINR
    snr_dB = rsrp_dB - noiseVardB;
    
    % TA estimate
    avgScCorr = 0;
    for scIdx = 0:(totNumSC-2)
        for symIdx = 0:(nSymDmrs-1)
            for antIdx = 0:(nRxAnt-1)
                avgScCorr = avgScCorr + conj(chEst(scIdx+1, symIdx+1, antIdx+1)) * chEst(scIdx+2, symIdx+1, antIdx+1);
            end
        end
    end
       
    avgScCorr = avgScCorr / (nRxAnt * (totNumSC-1) * nSymDmrs);
    taEstMicroSec = -10^6 * atan2(imag(avgScCorr), real(avgScCorr)) / (2*pi*scs);
        
    % STEP 5 derive equalizer coefficients
    % Note: USES ONLY FIRST DMRS PER HOP. NO TIME INTERPOLATION SUPPORTED
    % Note: Even when frequency hopping is disabled the concept of first
    % and second hop is retained and indicates which DMRS is used to derive
    % the equalizer
    
    global SimCtrl
    TdiModePf3 = SimCtrl.alg.TdiModePf3;
    if TdiModePf3 == 2 && nSymDmrs == 4 % TdiModePf3 = 2 is for testing only. Do not implement.
        
        for idxSymDmrs = 1:nSymDmrsFirstHop
            chEst1FirstHop{idxSymDmrs} = squeeze(chEst(:, idxSymDmrs, :));
        end
        
        for idxSymDmrs = 1:nSymDmrsSecondHop
            chEst1SecondHop{idxSymDmrs} = squeeze(chEst(:, idxSymDmrs + nSymDmrsFirstHop, :));
        end
        
        SetSymDataFirstHop = SetSymData(1:nSymDataFirstHop);
        SetSymDataSecondHop = SetSymData(nSymDataFirstHop+1:end);
        
        SetSymDmrsFirstHop = SetSymDmrs(1:nSymDmrsSecondHop);
        SetSymDmrsSecondHop = SetSymDmrs(nSymDmrsSecondHop+1:end);
        
        nSymFirstHop = nSymDataFirstHop + nSymDmrsFirstHop;
        
        chEst2FirstHop = apply_interpolation_pf3(chEst1FirstHop, SetSymDataFirstHop + 1, SetSymDmrsFirstHop + 1);
        chEst2SecondHop = apply_interpolation_pf3(chEst1SecondHop, SetSymDataSecondHop + 1 - nSymFirstHop, SetSymDmrsSecondHop + 1 - nSymFirstHop);
        
        for idxSym = 1:nSymDataFirstHop
            for idxSc=1:totNumSC
                chFirstHop(idxSc, idxSym) = sum(abs(squeeze(chEst2FirstHop{idxSym}(idxSc,:))).^2);
                eqFirstHop(idxSc, idxSym) = (1+noiseVar)/1 * 1/(chFirstHop(idxSc, idxSym)+noiseVar);
            end
        end
        
        for idxSym = 1:nSymDataSecondHop
            for idxSc=1:totNumSC
                chSecondHop(idxSc, idxSym) = sum(abs(squeeze(chEst2SecondHop{idxSym}(idxSc,:))).^2);
                eqSecondHop(idxSc, idxSym) = (1+noiseVar)/1 * 1/(chSecondHop(idxSc, idxSym)+noiseVar);
            end
        end
        
        for idxSym=1:nSymDataFirstHop
            for idxSc=1:totNumSC
                Z_data(idxSc,idxSym) = eqFirstHop(idxSc, idxSym) * conj(squeeze(chEst2FirstHop{idxSym}(idxSc,:))) * squeeze(Y_data(idxSc,idxSym,:));
            end
        end
        
        for idxSym=nSymDataFirstHop+1:nSym - nSymDmrs
            for idxSc=1:totNumSC
                Z_data(idxSc,idxSym) = eqSecondHop(idxSc, idxSym - nSymDataFirstHop) * conj(squeeze(chEst2SecondHop{idxSym - nSymDataFirstHop}(idxSc,:))) * squeeze(Y_data(idxSc,idxSym,:));
            end
        end
        
    else
        
        % equalizer first hop
        for ii=1:totNumSC
            if TdiModePf3 == 1 && nSymDmrs == 4
                chFirstHop(ii) = sum(abs(squeeze((chEst(ii,1,:) + chEst(ii,2,:))/2)).^2);
            else
                chFirstHop(ii) = sum(abs(squeeze(chEst(ii,1,:))).^2);
            end
            eqFirstHop(ii) = (1+noiseVar)/1 * 1/(chFirstHop(ii)+noiseVar);
        end
        
        if nSymDmrs == 1
            % do nothing. will use eqFirstHop only
        elseif nSymDmrs == 2
            % equalizer second hop
            for ii=1:totNumSC
                chSecondHop(ii) = sum(abs(squeeze(chEst(ii,2,:))).^2);
                eqSecondHop(ii) = (1+noiseVar)/1 * 1/(chSecondHop(ii)+noiseVar);
            end
        elseif nSymDmrs == 4
            % equalizer second hop when additional DMRS is present 3GPP 38.211
            % Table 6.4.1.3.3.2-1
            for ii=1:totNumSC
                if TdiModePf3 == 1
                    chSecondHop(ii) = sum(abs(squeeze((chEst(ii,3,:) + chEst(ii,4,:))/2)).^2);
                else
                    chSecondHop(ii) = sum(abs(squeeze(chEst(ii,3,:))).^2);
                end
                eqSecondHop(ii) = (1+noiseVar)/1 * 1/(chSecondHop(ii)+noiseVar);
            end
        end
        
        % STEP 6 equalize signal in frequency domain
        %multiply by channel conjugate and equalize
        for ii=1:nSymDataFirstHop
            for jj=1:totNumSC
                if TdiModePf3 == 1 && nSymDmrs == 4
                    Z_data(jj,ii) = eqFirstHop(jj) * conj(squeeze((chEst(jj,1,:)+chEst(jj,2,:))/2)).' * squeeze(Y_data(jj,ii,:));
                else
                    Z_data(jj,ii) = eqFirstHop(jj) * conj(squeeze(chEst(jj,1,:))).' * squeeze(Y_data(jj,ii,:));
                end
            end
        end
        for ii=nSymDataFirstHop+1:nSym - nSymDmrs
            if nSymDmrs == 1
                for jj=1:totNumSC
                    Z_data(jj,ii) = eqFirstHop(jj) * conj(squeeze(chEst(jj,1,:))).' * squeeze(Y_data(jj,ii,:));
                end
            elseif nSymDmrs == 2
                for jj=1:totNumSC
                    Z_data(jj,ii) = eqSecondHop(jj) * conj(squeeze(chEst(jj,2,:))).' * squeeze(Y_data(jj,ii,:));
                end
            elseif nSymDmrs == 4
                for jj=1:totNumSC
                    if TdiModePf3 == 1
                        Z_data(jj,ii) = eqSecondHop(jj) * conj(squeeze((chEst(jj,3,:)+chEst(jj,4,:))/2)).' * squeeze(Y_data(jj,ii,:));
                    else
                        Z_data(jj,ii) = eqSecondHop(jj) * conj(squeeze(chEst(jj,3,:))).' * squeeze(Y_data(jj,ii,:));
                    end
                end
            end
        end
    end
    
    % STEP 7 remove DFT spreading
    fftSize = totNumSC;
    z = zeros(fftSize*(nSym-nSymDmrs),1);
    for ii=1:nSym-nSymDmrs
        z(fftSize*(ii-1)+1:fftSize*ii) = sqrt(fftSize)*ifft(Z_data(:,ii));
    end
    
    %% start back-end processing
    % STEP 8 demodulate
    scrmLLR = zeros(E_seg1 + E_seg2, 1);
    
    if pi2Bpsk
        for ii=1:totNumSym
            if mod(ii,2)
                scrmLLR(ii) = real(2/sqrt(2) * (1-1i) * z(ii) / noiseVar);
            else
                scrmLLR(ii) = imag(2/sqrt(2) * (1-1i) * z(ii) / noiseVar);
            end
        end
    else
        for ii=1:totNumSym
            scrmLLR(2*ii-1) = real(2 * z(ii) / noiseVar);
            scrmLLR(2*ii) = imag(2 * z(ii) / noiseVar);
        end
    end
    
    % STEP 9 descramble
    descrmLLR = zeros(E_seg1 + E_seg2, 1);
    for bitIdx = 1:(E_seg1 + E_seg2)
       descrmLLR(bitIdx) = (1 - 2*randomSeqScrm(bitIdx)) * scrmLLR(bitIdx);
    end
    
    % demultiplexing LLR array for bit sequence 1 and 2
    [descrmLLRSeq1, descrmLLRSeq2] = PucchF34UciDeMultiplexing(nSym, nSymDmrs, pi2Bpsk, totNumSC, E_seg1, E_seg2, descrmLLR);
    
    %% decode
    DTX = 0;
    
    if A_seg1<12
        fecEnc = FecRmObj(0, E_seg1, A_seg1);
        [UCIstrm, confLevel] = fecEnc(descrmLLRSeq1); 
        decodedUciSeg1 = str2num(UCIstrm);
        
        global SimCtrl
        dtxModePf3 = SimCtrl.alg.dtxModePf3;
        
%         global confLevel_vec
%         confLevel_vec = [confLevel_vec, confLevel];
        
        if dtxModePf3
            thr = 0.2; % optimized with E = 64 and A = 4
            
            % Adjust thr with different E and A values based on 
            % limited experiments. May need further optimization.
            thr = thr * sqrt(64/E_seg1)*sqrt(A_seg1/4);
            thr = max(min(0.8, thr), 0.1); 
            
            confLevelThr = thr * DTXthreshold;
            if noiseVar == 0 % to handle forceRxZero case
                DTX = 1;
            else
                DTX = (confLevel < confLevelThr);
            end            
        else
            DTX = 0;
        end
        
        % determine detection status
        pUciF3{uciOutputIdx}.HarqDetectionStatus     = 4; % default
        pUciF3{uciOutputIdx}.CsiPart1DetectionStatus = 4; % default
        pUciF3{uciOutputIdx}.CsiPart2DetectionStatus = 4; % default
        if DTX
            pUciF3{uciOutputIdx}.HarqDetectionStatus     = 3;
            pUciF3{uciOutputIdx}.CsiPart1DetectionStatus = 3;
            pUciF3{uciOutputIdx}.CsiPart2DetectionStatus = 3;
        end
    else
        crcErrorFlag = 0;
        global SimCtrl
        % Collect polar intermediate buffers (seg1 first, then seg2 below when present)
        % so the TV generator can emit an `_uciPolarDebug.h5` sidecar.
        pUciF3{uciOutputIdx}.polarInterBuffers = {};
        if SimCtrl.alg.useNrUCIDecode
            decodedUciSeg1 = nrUCIDecode(descrmLLRSeq1, A_seg1);
        else
            listLength = SimCtrl.alg.listLength;
            [decodedUciSeg1, crcErrorFlag, interBuffers] = uciSegPolarDecode(A_seg1, E_seg1, listLength, descrmLLRSeq1);
            pUciF3{uciOutputIdx}.polarInterBuffers{end+1} = interBuffers;
        end

        % determine detection status
        pUciF3{uciOutputIdx}.HarqDetectionStatus     = 2; % default
        pUciF3{uciOutputIdx}.CsiPart1DetectionStatus = 2; % default
        pUciF3{uciOutputIdx}.CsiPart2DetectionStatus = 5; % default
        if BitLenHarq > 0
            pUciF3{uciOutputIdx}.HarqDetectionStatus = 1 + crcErrorFlag;
        end
        if BitLenCsiPart1 > 0
            pUciF3{uciOutputIdx}.CsiPart1DetectionStatus = 1 + crcErrorFlag;
        end
    end
    
    %% CSI part 2 decode
    decodedUciSeg2 = [];
    A_seg2 = 0;

    if E_seg2 > 0
        rank = 0;
        for bitInd = 1:rankBitSize
            rank = rank + decodedUciSeg1(rankBitOffset + bitInd) * 2^(bitInd-1);
        end
        
        A_seg2 = csiP2PayloadSizeCalc(rank);
        
        if A_seg2<12
            fecEnc = FecRmObj(0, E_seg2, A_seg2);
            decodedUciSeg2 = str2num(fecEnc(descrmLLRSeq2));
        else
            crcErrorFlag = 0;
            global SimCtrl
            if SimCtrl.alg.useNrUCIDecode
                decodedUciSeg2 = nrUCIDecode(descrmLLRSeq2, A_seg2);
            else
                listLength = SimCtrl.alg.listLength;
                [decodedUciSeg2, crcErrorFlag, interBuffers] = uciSegPolarDecode(A_seg2, E_seg2, listLength, descrmLLRSeq2);
                if ~isfield(pUciF3{uciOutputIdx}, 'polarInterBuffers')
                    pUciF3{uciOutputIdx}.polarInterBuffers = {};
                end
                pUciF3{uciOutputIdx}.polarInterBuffers{end+1} = interBuffers;
            end

            % determine detection status
            pUciF3{uciOutputIdx}.CsiPart2DetectionStatus = 1 + crcErrorFlag;
        end
    end
    
    %% currently do not support the division of decoded bits into SR, HARQ and CSI part 1
    
    pUciF3{uciOutputIdx}.uciSeg1 = decodedUciSeg1;
    pUciF3{uciOutputIdx}.uciSeg2 = decodedUciSeg2;
    if BitLenHarq > 0
        pUciF3{uciOutputIdx}.HarqValues = decodedUciSeg1(1:BitLenHarq);
    else
        pUciF3{uciOutputIdx}.HarqValues = [];
    end
    if BitLenSr > 0 
        pUciF3{uciOutputIdx}.SrValues = decodedUciSeg1(BitLenHarq + 1:BitLenHarq + BitLenSr);
    else
        pUciF3{uciOutputIdx}.SrValues = [];
    end
    if BitLenCsiPart1 > 0 
        pUciF3{uciOutputIdx}.CsiP1Values = decodedUciSeg1(BitLenHarq + BitLenSr + 1:BitLenHarq + BitLenSr + BitLenCsiPart1);
    else
        pUciF3{uciOutputIdx}.CsiP1Values = [];
    end
    pUciF3{uciOutputIdx}.CsiP2Values = decodedUciSeg2;
    pUciF3{uciOutputIdx}.NumHarq = A_seg1;
    pUciF3{uciOutputIdx}.E_seg1 = E_seg1;
    pUciF3{uciOutputIdx}.descrmLLR = descrmLLR;
    pUciF3{uciOutputIdx}.descrmLLRSeq1 = descrmLLRSeq1;
    pUciF3{uciOutputIdx}.descrmLLRSeq2 = descrmLLRSeq2;
    pUciF3{uciOutputIdx}.SinrDB     = min(max(snr_dB,-99),99);
    pUciF3{uciOutputIdx}.RSRP       = min(max(rsrp_dB,-99),99);
    pUciF3{uciOutputIdx}.RSSI       = min(max(rssi_dB,-99),99);
    pUciF3{uciOutputIdx}.InterfDB   = min(max(noiseVardB,-99),99);
    pUciF3{uciOutputIdx}.DTX = DTX;
    if DTX
        taEstMicroSec = 0;
        pUciF3{uciOutputIdx}.uciSeg1 = 2*ones(size(pUciF3{uciOutputIdx}.uciSeg1));
        pUciF3{uciOutputIdx}.uciSeg2 = 2*ones(size(pUciF3{uciOutputIdx}.uciSeg2));
        pUciF3{uciOutputIdx}.HarqValues = 2*ones(size(pUciF3{uciOutputIdx}.HarqValues));
        pUciF3{uciOutputIdx}.SrValues = 2*ones(size(pUciF3{uciOutputIdx}.SrValues));
        pUciF3{uciOutputIdx}.CsiP1Values = 2*ones(size(pUciF3{uciOutputIdx}.CsiP1Values));
        pUciF3{uciOutputIdx}.CsiP2Values = 2*ones(size(pUciF3{uciOutputIdx}.CsiP2Values));
    end
    pUciF3{uciOutputIdx}.taEstMicroSec = taEstMicroSec;
end

return
