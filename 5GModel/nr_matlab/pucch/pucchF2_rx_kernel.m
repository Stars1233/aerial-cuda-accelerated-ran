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

function [pUciF2] = pucchF2_rx_kernel(numRF, pucchF2tables, pucchF2dynamicDesc, pUciF2)

% Load parameters from pucchF2tables
W1                  = pucchF2tables.W1;
W2                  = pucchF2tables.W2;
W3                  = pucchF2tables.W3;
W4                  = pucchF2tables.W4;

% Load number of PF2 UCIs
numUcis = pucchF2dynamicDesc.nGrps;

for uciIdx = 1:numUcis
    % Load parameters from pucchF2dynamicDesc
    Xtf                 = pucchF2dynamicDesc.Xtf;
    BWPStart            = pucchF2dynamicDesc.BWPStart{uciIdx};
    randomSeqScrm       = pucchF2dynamicDesc.randomSeqScrm{uciIdx};
    randomSeqDmrs1      = pucchF2dynamicDesc.randomSeqDmrs1{uciIdx};
    randomSeqDmrs2      = pucchF2dynamicDesc.randomSeqDmrs2{uciIdx};
    startPrb            = pucchF2dynamicDesc.startPrb{uciIdx};
    prbSize             = pucchF2dynamicDesc.prbSize{uciIdx};
    startSym            = pucchF2dynamicDesc.startSym{uciIdx} + 1;  % Matlab 1-based indexing
    nSym                = pucchF2dynamicDesc.nSym{uciIdx};
    E_seg1              = pucchF2dynamicDesc.E_seg1{uciIdx};
    A_seg1              = pucchF2dynamicDesc.A_seg1{uciIdx};
    noiseVar            = pucchF2dynamicDesc.noiseVar{uciIdx};
    uciOutputIdx        = pucchF2dynamicDesc.uciOutputIdx{uciIdx};
    freqHopFlag         = pucchF2dynamicDesc.freqHopFlag{uciIdx};
    secondHopPrb        = pucchF2dynamicDesc.secondHopPrb{uciIdx};
    DTXthreshold        = pucchF2dynamicDesc.DTXthreshold{uciIdx};
    BitLenHarq          = pucchF2dynamicDesc.BitLenHarq{uciIdx};
    BitLenSr            = pucchF2dynamicDesc.BitLenSr{uciIdx};
    BitLenCsiPart1      = pucchF2dynamicDesc.BitLenCsiPart1{uciIdx};
    scs                 = pucchF2dynamicDesc.scs;
    mStartFirstHop      = pucchF2dynamicDesc.mStartFirstHop{uciIdx};
    mStartSecondHop     = pucchF2dynamicDesc.mStartSecondHop{uciIdx};

    startCrb            = startPrb + BWPStart;
    secondHopCrb        = secondHopPrb + BWPStart;
    
    % constants and derived params
    prbConst = 12;
    totNumSC = prbConst*prbSize;
    dataRePrb = 8;
    dmrsRePrb = prbConst-dataRePrb;
    % STEP 1 load data from Xtf
    Y_pucch = [];
    
    if freqHopFlag && nSym>1
        freq_idx_firstHop = (startCrb*12+1) : startCrb*12+totNumSC;
        time_idx_firstHop = startSym;
        Y_pucch = Xtf(freq_idx_firstHop,time_idx_firstHop,1:numRF);
        
        freq_idx_secHop = (secondHopCrb*12+1) : secondHopCrb*12+totNumSC;
        time_idx_secHop = startSym + nSym - 1;
        Y_pucch(:,2,:) = Xtf(freq_idx_secHop,time_idx_secHop,1:numRF);
    else
        freq_idx = (startCrb*12+1) : startCrb*12+totNumSC;
        time_idx = (startSym) : (startSym + nSym - 1);
        Y_pucch = Xtf(freq_idx,time_idx,1:numRF);
    end

    % calculate RSSI
    rssiTemp = sum(abs(Y_pucch).^2, 1);
    rssiTemp = reshape(rssiTemp, [size(rssiTemp, 2), size(rssiTemp, 3)]); % remove the first dimension which is a frequency dimension
    rssi = sum(mean(rssiTemp,1),2);% average across all symbols and sum across all Rx antenna
    rssi_dB = 10*log10(rssi);
    
    % STEP 2 extract DMRS REs
    Y_dmrs = Y_pucch(2:3:end,:,:);
    
    % STEP 3 Generate DMRS    
    x_dmrs = [];
    
    c = randomSeqDmrs1;
    x_dmrs = [1/sqrt(2) * (1-2*c((2*mStartFirstHop+1):2:(prbSize*8 + 2*mStartFirstHop)) + 1i*(1-2*c((2*mStartFirstHop+2):2:(prbSize*8 + 2*mStartFirstHop))))];
    
    if nSym > 1
        c = randomSeqDmrs2;
        x_dmrs = [x_dmrs; 1/sqrt(2) * (1-2*c((2*mStartSecondHop+1):2:(prbSize*8 + 2*mStartSecondHop)) + 1i*(1-2*c((2*mStartSecondHop+2):2:(prbSize*8 + 2*mStartSecondHop))))];
    end
     
    x_dmrs = reshape(x_dmrs, prbSize*4, nSym);
    
    % STEP 4 Remove code (multiply)
    for k = 1:nSym
        for ii=1:numRF
            chEstDmrs(:,k,ii) = Y_dmrs(:,k,ii).*conj(x_dmrs(:, k));
        end
    end
    
    %% TA estimation
    avgScCorr = 0;
    for scIdx = 0:(dmrsRePrb-2)
        for symIdx = 0:(nSym-1)
            for antIdx = 0:(numRF-1)
                avgScCorr = avgScCorr + conj(chEstDmrs(scIdx+1, symIdx+1, antIdx+1)) * chEstDmrs(scIdx+2, symIdx+1, antIdx+1);
            end
        end
    end
   
   avgScCorr = avgScCorr / (numRF * (dmrsRePrb-1) * nSym);
   taEstMicroSec = -10^6 * atan2(imag(avgScCorr), real(avgScCorr)) / (2*pi*scs*3);
           
    % STEP 5 Filter channel estimation coefficients in freq domain
    % filter and interpolate channel estimation coefficients
    % No filter in time domain for now. It would only be useful when no
    % frequency hopping is enabled
    
    if prbSize == 1
        for ii=1:numRF
            for kk=1:nSym
                chEstFull(:,kk,ii) = W1 * chEstDmrs(:,kk,ii);
            end
        end
    elseif prbSize == 2
        for ii=1:numRF
            for kk=1:nSym
                chEstFull(:,kk,ii) = W2 * chEstDmrs(:,kk,ii);
            end
        end
    elseif prbSize == 3
        for ii=1:numRF
            for kk=1:nSym
                chEstFull(:,kk,ii) = W3 * chEstDmrs(:,kk,ii);
            end
        end
    elseif prbSize >= 4
        prbBlockSize = 4;
        nPrbBlocks = ceil(prbSize/prbBlockSize);
        
        % loop over antennas
        for hh=1:nSym
            for ii=1:numRF
                for jj=1:nPrbBlocks
                    if jj*prbBlockSize < prbSize
                        tmp = W4*chEstDmrs((jj-1)*prbBlockSize*4+1:jj*prbBlockSize*4,hh,ii);
                        chEstFull((jj-1)*prbBlockSize*prbConst+1:jj*prbBlockSize*prbConst,hh,ii) = tmp;
                    else
                        % last block overlaps with previous one when prbSize/prbBlockSize
                        % not integer
                        numRowsSkip = (nPrbBlocks*prbBlockSize - prbSize) * 4;
                        tmp = W4*chEstDmrs((jj-1)*prbBlockSize*4+1-numRowsSkip:jj*prbBlockSize*4-numRowsSkip,hh,ii);
                        chEstFull((jj-1)*prbBlockSize*prbConst+1:prbSize*prbConst,hh,ii) = tmp(numRowsSkip*3+1:4*prbConst);
                    end
                end
            end
        end
    end
     
    % STEP 6 check SINR to detect DTX. It uses interpolated channel
    % and noise estimation
    % RSRP 
    rsrp = 0;
    for ii=1:prbSize*prbConst
        for jj=1:nSym
            for kk=1:numRF
                rsrp = rsrp + abs(chEstFull(ii,jj,kk)).^2;
            end
        end
    end
    rsrp = rsrp / (nSym*prbSize*prbConst*numRF);
    rsrp_dB = 10*log10(rsrp);

    % Noise power
    % Generate DMRS
    % -> it is already available from channel estimation algorithm
    % Generate r_tilde (RX signal without DMRS)
    
    r_tilde = zeros(prbSize*4, nSym, numRF);
    for ii=1:numRF
        for jj=1:nSym
            r_tilde(:,jj,ii) = Y_dmrs(:,jj,ii) - x_dmrs(:,jj).*chEstFull(2:3:prbSize*12,jj,ii);
        end
    end
       
    % Calculate r_tilde power
    % a bias correction factor is added to compensate for noise
    % filtered out in the channel estimation process (low-pass filtering of
    % noise with equivalent BW=1.25/3
    tmp = abs(r_tilde).^2;
    noiseVar = mean(tmp(:));
    noiseVardB = 10*log10(noiseVar) - 10*log10(1.75/3);
    
    % Calculate SINR
    snr_dB = rsrp_dB - noiseVardB ;
             
    % STEP 7 multiply by H^*
    kk = 1;
    for ii=1:nSym
        for jj=1:totNumSC
            if (mod(jj,3) ~= 2)
                z(kk) = conj(squeeze(chEstFull(jj,ii,:))).' * squeeze(Y_pucch(jj,ii,:));
                kk = kk + 1;
            end
        end
    end
    
    % STEP 8 QPSK demodulate
    scrmLLR = zeros(E_seg1, 1);

    for ii=1:prbSize*dataRePrb*nSym
        scrmLLR(2*ii-1) = real(2 * z(ii) / noiseVar);
        scrmLLR(2*ii) = imag(2 * z(ii) / noiseVar);
    end
    
    % STEP 9 descramble
    descrmLLR = zeros(E_seg1, 1);
    for bitIdx = 1:E_seg1
       descrmLLR(bitIdx) = (1 - 2*randomSeqScrm(bitIdx)) * scrmLLR(bitIdx);
    end
    
    %% STEP 10 decode
    DTX = 0;
            
    if A_seg1<12
        fecEnc = FecRmObj(0, E_seg1, A_seg1);
        [UCIstrm, confLevel] = fecEnc(descrmLLR);
        decodedUCI = str2num(UCIstrm);

        global SimCtrl
        dtxModePf2 = SimCtrl.alg.dtxModePf2;
        
%         global confLevel_vec
%         confLevel_vec = [confLevel_vec, confLevel];
        
        if dtxModePf2
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
                        
        pUciF2{uciOutputIdx}.HarqDetectionStatus     = 4; % default
        pUciF2{uciOutputIdx}.CsiPart1DetectionStatus = 4; % default
        pUciF2{uciOutputIdx}.CsiPart2DetectionStatus = 4; % default
        if DTX
            % determine detection status
            pUciF2{uciOutputIdx}.HarqDetectionStatus     = 3;
            pUciF2{uciOutputIdx}.CsiPart1DetectionStatus = 3;
            pUciF2{uciOutputIdx}.CsiPart2DetectionStatus = 3;
            
            if BitLenHarq > 0
                pUciF2{uciOutputIdx}.HarqValues = 2*ones(BitLenHarq, 1);
            else
                pUciF2{uciOutputIdx}.HarqValues = [];
            end
            if BitLenSr > 0
                pUciF2{uciOutputIdx}.SrValues = 2*ones(BitLenSr, 1);
            else
                pUciF2{uciOutputIdx}.SrValues = [];
            end
            if BitLenCsiPart1 > 0
                pUciF2{uciOutputIdx}.CsiP1Values = 2*ones(BitLenCsiPart1, 1);
            else
                pUciF2{uciOutputIdx}.CsiP1Values = [];
            end
            pUciF2{uciOutputIdx}.taEstMicroSec  = 0;
            pUciF2{uciOutputIdx}.uciSeg1        = 2*ones(A_seg1, 1);
            pUciF2{uciOutputIdx}.NumHarq        = BitLenHarq;
            pUciF2{uciOutputIdx}.descrmLLR      = descrmLLR;
            pUciF2{uciOutputIdx}.E_seg1         = E_seg1;
            pUciF2{uciOutputIdx}.SinrDB         = min(max(snr_dB,-99),99);
            pUciF2{uciOutputIdx}.RSRP           = min(max(rsrp_dB,-99),99);
            pUciF2{uciOutputIdx}.RSSI           = min(max(rssi_dB,-99),99);
            pUciF2{uciOutputIdx}.InterfDB       = min(max(noiseVardB,-99),99);
            pUciF2{uciOutputIdx}.DTX            = DTX;
            continue;
        end
    else
        crcErrorFlag = 0;
        global SimCtrl
        % Initialize polar intermediate buffer collector; populated only on the
        % polar (vs Reed-Muller / nrUCIDecode) branch so the TV generator can emit
        % an `_uciPolarDebug.h5` sidecar that mirrors PUSCH's saveTV_polar_debug.
        pUciF2{uciOutputIdx}.polarInterBuffers = {};
        if SimCtrl.alg.useNrUCIDecode
            decodedUCI = nrUCIDecode(descrmLLR, A_seg1);
        else
            listLength = SimCtrl.alg.listLength;
            [decodedUCI, crcErrorFlag, interBuffers] = uciSegPolarDecode(A_seg1, E_seg1, listLength, descrmLLR);
            pUciF2{uciOutputIdx}.polarInterBuffers{end+1} = interBuffers;
        end

        % determine detection status
        pUciF2{uciOutputIdx}.HarqDetectionStatus     = 2; % default
        pUciF2{uciOutputIdx}.CsiPart1DetectionStatus = 2; % default
        pUciF2{uciOutputIdx}.CsiPart2DetectionStatus = 5; % default
        if BitLenHarq > 0
            pUciF2{uciOutputIdx}.HarqDetectionStatus = 1 + crcErrorFlag;
        end
        if BitLenCsiPart1 > 0
            pUciF2{uciOutputIdx}.CsiPart1DetectionStatus = 1 + crcErrorFlag;
        end
    end
    
    pUciF2{uciOutputIdx}.uciSeg1    = decodedUCI;
    if BitLenHarq > 0
        pUciF2{uciOutputIdx}.HarqValues = decodedUCI(1:BitLenHarq);
    else
        pUciF2{uciOutputIdx}.HarqValues = [];
    end
    if BitLenSr > 0 
        pUciF2{uciOutputIdx}.SrValues = decodedUCI(BitLenHarq + 1:BitLenHarq + BitLenSr);
    else
        pUciF2{uciOutputIdx}.SrValues = [];
    end
    if BitLenCsiPart1 > 0 
        pUciF2{uciOutputIdx}.CsiP1Values = decodedUCI(BitLenHarq + BitLenSr + 1:BitLenHarq + BitLenSr + BitLenCsiPart1);
    else
        pUciF2{uciOutputIdx}.CsiP1Values = [];
    end
    
    pUciF2{uciOutputIdx}.taEstMicroSec = taEstMicroSec;
    pUciF2{uciOutputIdx}.NumHarq    = BitLenHarq;
    pUciF2{uciOutputIdx}.E_seg1     = E_seg1;
    pUciF2{uciOutputIdx}.descrmLLR  = descrmLLR;
    pUciF2{uciOutputIdx}.SinrDB     = min(max(snr_dB,-99),99);
    pUciF2{uciOutputIdx}.RSRP       = min(max(rsrp_dB,-99),99);
    pUciF2{uciOutputIdx}.RSSI       = min(max(rssi_dB,-99),99);
    pUciF2{uciOutputIdx}.InterfDB   = min(max(noiseVardB,-99),99);
    pUciF2{uciOutputIdx}.DTX        = DTX;
end

return


