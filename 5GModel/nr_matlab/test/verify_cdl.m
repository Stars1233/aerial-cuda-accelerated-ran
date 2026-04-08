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

%% Verify GPU CDL vs our own MATLAB CDL implementation
% using the h5 file generated from cdl_chan_ex, which include the random numbers needed to generated CDL channel mode

% Usage:
% 1. Run cdl_chan_ex with '-d', GPU CDL will generate cdlChan*.h5 files
% 2. Run this file with the above h5 files

% reason to use matlab verification: need to use the random initial phase generated on C platform, see phiRand, which is saved per link

% tvFileName: h5 file name
% verbose: 0: only print final results; 1: print the max abs error per link
% ttiIdx: TTI index, use to calculate the current time stamp. Can be auto detected from TV file name or manual specify (not recommend)

% example: verify_cdl('cdlChan_1cell1Ue_4x4_A30_dopp5_cfo200_runMode0_freqConvert1_scSampling1_FP32_swap0_TTI0.h5')
% example: verify_cdl('cdlChan_1cell1Ue_4x4_A30_dopp5_cfo200_runMode0_freqConvert1_scSampling1_FP32_swap0_TTI0.h5', 1)

% for testing a batch of h5 flies, use below
% h5_dir = '.';
% h5_files = dir(fullfile(h5_dir, 'cdlChan*.h5'));
% for i = 1:numel(h5_files)
%     h5_file_name = fullfile(h5_dir, h5_files(i).name);
%     h5_info = h5info(h5_file_name);  % Query HDF5 file information
%     verify_cdl(h5_file_name);  % Call the verify function with the file name
% end

function checkRes = verify_cdl(tvFileName, verbose, ttiIdx)
fprintf("Checking CDL channel using file %s \n", tvFileName);

if nargin < 2
    verbose = 0;
end

if nargin < 3
    % use regular expression to extract the number between 'TTI' and '.h5'
    expression = 'TTI(\d+)';
    tokens = regexp(tvFileName, expression, 'tokens');

    if ~isempty(tokens)
        ttiIdx = str2double(tokens{1}{1});
        fprintf('Auto detected TTI index: %d \n', ttiIdx);
    else
        ttiIdx = 0;
        fprintf('Warning: TTI index not found, using default 0 \n');
    end
end

% detect precision from TV name: FP16 or FP32
expression = '\_FP(\d+)\_';
tokens = regexp(tvFileName, expression, 'tokens');

if ~isempty(tokens)
    precision = str2double(tokens{1}{1});
    fprintf('Auto detected data precision: FP%d \n', precision);
else
    precision = 32;
    fprintf('Warning: data precision not found, using default FP32 \n');
end

% detect precision from swap tx/rx
expression = '\_swap(\d+)\_';
tokens = regexp(tvFileName, expression, 'tokens');

if ~isempty(tokens)
    enableSwapTxRx = str2double(tokens{1}{1});
    if enableSwapTxRx
        fprintf('Auto detected swap tx/rx: %d (enabled) \n', enableSwapTxRx);
    else
        fprintf('Auto detected swap tx/rx: %d (disabled) \n', enableSwapTxRx);
    end
else
    enableSwapTxRx = 0;
    fprintf('Warning: swap tx/rx not found, using 0 (disabled) \n');
end

% read CDL config parameters
cdlCfg = h5read(tvFileName,'/cdlCfg');
delayProfile = char(cdlCfg.delayProfile);
DelaySpread = double(cdlCfg.delaySpread);
nCell = double(cdlCfg.nCell);
nUe = double(cdlCfg.nUe);
UeAntSize = double(h5read(tvFileName,'/ueAntSize'));
UeAntSpacing = double(h5read(tvFileName,'/ueAntSpacing'));
UeAntPolarAngles = double(h5read(tvFileName,'/ueAntPolarAngles'));
UeAntPattern = double(cdlCfg.ueAntPattern);
BsAntSize = double(h5read(tvFileName,'/bsAntSize'));
BsAntSpacing = double(h5read(tvFileName,'/bsAntSpacing'));
BsAntPolarAngles = double(h5read(tvFileName,'/bsAntPolarAngles'));
BsAntPattern = double(cdlCfg.bsAntPattern);
nBsAnt = prod(BsAntSize);
nUeAnt = prod(UeAntSize);
f_doppler = double(cdlCfg.maxDopplerShift);
f_samp = double(cdlCfg.f_samp);
f_batch = double(cdlCfg.fBatch); % update rate of quasi-static channel
% defualt f_batch is 15e3, which is 50x larger than max doppler freq (300Hz). It should be fast enough to capture the channel variation over time
nRay = double(cdlCfg.numRay);
cfoHz = double(cdlCfg.cfoHz);
sigLenPerAnt = double(cdlCfg.sigLenPerAnt);
timeDelay = double(cdlCfg.delay);
N_sc = double(cdlCfg.N_sc);
N_sc_Prbg = double(cdlCfg.N_sc_Prbg);
scSpacingHz = double(cdlCfg.scSpacingHz);
runMode = double(cdlCfg.runMode);
procSigFreq = double(cdlCfg.procSigFreq);
freqConvertType = double(cdlCfg.freqConvertType);
% freqConvertType 0: use first SC for CFR on the Prbg
% freqConvertType 1: use center SC for CFR on the Prbg, e.g., sc 6 for sc 0,1,2,...,11
% freqConvertType 2: use last SC for CFR on the Prbg
% freqConvertType 3: use average SC for CFR on the Prbg
% freqConvertType 4: use average SC for CFR on the Prbg with removing frequency ramping
scSampling = double(cdlCfg.scSampling);
useFFT = 0;

% Read CDL non-zero fir index, non-zero fir power, time channel
firNzIdxGpu = h5read(tvFileName,'/firNzIdx');
firNzPwGpu = h5read(tvFileName,'/firNzPw');
batchCumLen = double(h5read(tvFileName,'/batchCumLen')); % start tx sample per batch
tBatch = double(h5read(tvFileName,'/tBatch'))'; % start time per batch
nBatch = length(tBatch);
timeChan = h5read(tvFileName,'/timeChan');
fprintf("CDL channel verification: %d cells, %d UEs, %d nBsAnt, %d nUeAnt, runMode %d, freqConvertType %d. \n", nCell, nUe, nBsAnt, nUeAnt, runMode, freqConvertType);

ttiLen = 0.001 / (scSpacingHz / 15e3); % 500 us
N_fft = 4096;
N_Prbg = ceil(N_sc/N_sc_Prbg);
corrMatrix = diag(ones(1, nBsAnt*nUeAnt)); % load MIMO correlation matrix; TOOD: only support low MIMO correlation, so corrMatrix is an identity matrix
[DPA, PCP] = loadCdlParam(['CDL-' delayProfile]);
T_samp = 1/f_samp;
delay_samp = round(timeDelay/T_samp);

d2pi = pi/180;
% BsAntOrientation = [0 0 0];
% UeAntOrientation = [0 0 0];
v_direction = [90, 0];

PathDelays = DPA(:,1)';
AveragePathGains = DPA(:,2)';
AnglesAoD = DPA(:,4)';
AnglesAoA = DPA(:,3)';
AnglesZoD = DPA(:,6)';
AnglesZoA = DPA(:,5)';
% set per-cluster parameters
c_ASD = PCP(2);
c_ASA = PCP(1);
c_ZSD = PCP(4);
c_ZSA = PCP(3);
XPR = PCP(5);

% 38.901 table 7.5-3
RayOffsetAngles = [0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, ...
    0.3715, -0.3715, 0.5129, -0.5129, 0.6797, -0.6797, 0.8844, -0.8844, ...
    1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551];

PathDelays = PathDelays*DelaySpread;
numClusters = length(PathDelays);
numRay = 20;
% Quantize to FIR filter with T_samp interval
firTapMap = round(PathDelays * 1e-9 / T_samp) + 1;  % map to the same tap dure to sampling
firPw = 10.^(AveragePathGains/10);
lenFir = round(PathDelays(end) * 1e-9 / T_samp)+1;
firPw = firPw/sum(firPw);  % normalize PDP
Ntap = length(unique(firTapMap));  % number of taps in final CIR

timeChanSizePerLink = nBatch * Ntap * nBsAnt * nUeAnt;
switch(runMode)
    case 0
        freqChanPrbgSizePerLink = 0;
    case {1, 2}
        freqChanPrbg = h5read(tvFileName,'/freqChanPrbg'); % freq chan on PRBG
        freqChanPrbgSizePerLink = nBsAnt * nUeAnt * N_Prbg;
end

if(sigLenPerAnt > 0)
    txSigIn = h5read(tvFileName,'/txSigIn'); % tx time signal
    rxSigOut = h5read(tvFileName,'/rxSigOut'); % rx time signal
    if enableSwapTxRx
        sigOutLenPerLink = sigLenPerAnt * nBsAnt;
        assert(length(rxSigOut.re) == sigOutLenPerLink * nCell * nUe, "Error: length of input signal does not match sigLenPerAnt in CDL config!")
    else
        sigOutLenPerLink = sigLenPerAnt * nUeAnt;
        assert(length(rxSigOut.re) == sigOutLenPerLink * nCell * nUe, "Error: length of input signal does not match sigLenPerAnt in CDL config!")
    end
end

checkRes = zeros(nCell*nUe, 8); % saving CDL verification results, max dimension 8, format: [cid, uid, firNzIdx firNzPw, timeChan, sigOut, freqChanSc, freqChanPrg]

% check non-zero fir index and power, which are stored as sparse matrix on GPU
% results of non-zero fir index, same for all links
firNzIdx = sort(unique(round(PathDelays * 1e-9 * f_samp)));
checkRes(:, 3) = max(firNzIdx' - double(firNzIdxGpu));
% results of non-zero fir power, same for all links
checkRes(:, 4) = max(firPw' - firNzPwGpu.^2 * numRay);

for cellIdx = 1 : nCell
    for ueIdx = 1 : nUe
        cidUidIdx = (cellIdx - 1) * nUe + ueIdx;
        % save cid
        checkRes(cidUidIdx, 1) = cellIdx - 1;
        % save uid
        checkRes(cidUidIdx, 2) = ueIdx - 1;

        % check time channel
        phiRand_file = h5read(tvFileName,['/phiRandLink' num2str(cidUidIdx - 1)]);
        rayCouplingRand_file = h5read(tvFileName,['/rayCouplingRandLink' num2str(cidUidIdx - 1)]);
        % read random phase used on GPU CDL
        phiRand = ones(numClusters, nRay, 4); % Initialize the 3D array
        for iTap = 1:numClusters
            for iPath = 1:nRay
                % Calculate the position indices for the 1D array
                pos = ((iTap - 1) * nRay + (iPath - 1)) * 4 + (1:4);
                % Assign values from phiRand_file to the appropriate slice
                phiRand(iTap, iPath, :) = phiRand_file(pos);
            end
        end


        rayCouplingRand = ones(numClusters, nRay, 4); % Initialize the 3D array
        for iTap = 1:numClusters
            for i = 1:4
                % Calculate the position indices for the 1D array
                pos = ((iTap - 1) * 4 + (i - 1)) * nRay + (1:nRay);
                % Assign values from rayCouplingRand_file to the appropriate slice
                rayCouplingRand(iTap, :, i) = rayCouplingRand_file(pos);
            end
        end

        % Preallocate channel matrix
        chanMatrix = zeros(nBsAnt, nUeAnt, lenFir, nBatch);

        % Precompute time sequence
        timeSeq = tBatch + ttiIdx * ttiLen;

        % Precompute antenna locations and polar angles
        ueAntLocs = zeros(3, nUeAnt); % [x; y; z] for each Rx antenna
        bsAntLocs = zeros(3, nBsAnt); % [x; y; z] for each Tx antenna
        ueAntPolarAngles = zeros(1, nUeAnt); % Polar angles for Rx antennas
        bsAntPolarAngles = zeros(1, nBsAnt); % Polar angles for Tx antennas
        [ueDh, ueDv] = getElemSpacing(UeAntSpacing);
        [bsDh, bsDv] = getElemSpacing(BsAntSpacing);

        for u = 1:nUeAnt
            [mAnt, uAnt, pAnt] = findAntLoc(u, UeAntSize);
            ueAntLocs(:, u) = [0; (uAnt-1) * ueDh; (mAnt-1) * ueDv];
            ueAntPolarAngles(u) = UeAntPolarAngles(pAnt);
        end

        for s = 1:nBsAnt
            [mAnt, uAnt, pAnt]  = findAntLoc(s, BsAntSize);
            bsAntLocs(:, s) = [0; (uAnt-1) * bsDh; (mAnt-1) * bsDv];
            bsAntPolarAngles(s) = BsAntPolarAngles(pAnt);
        end

        % Precompute inverse square root of kappa for XPR
        kappa_inv_sqrt = sqrt(1 / (10^(XPR / 10)));

        % Main loop
        for u = 1:nUeAnt
            d_bar_rx_u = ueAntLocs(:, u);
            zetaUeAnt = ueAntPolarAngles(u);

            for s = 1:nBsAnt
                d_bar_tx_s = bsAntLocs(:, s);
                zetaBsAnt = bsAntPolarAngles(s);

                for n = 1:numClusters
                    H_u_s_n = zeros(1, nBatch);

                    % Precompute angles for the cluster
                    phi_n_AOD = AnglesAoD(n);
                    phi_n_AOA = AnglesAoA(n);
                    theta_n_ZOD = AnglesZoD(n);
                    theta_n_ZOA = AnglesZoA(n);

                    % Extract ray coupling indices for the cluster
                    idxASD = squeeze(rayCouplingRand(n, :, 1)) + 1;
                    idxASA = squeeze(rayCouplingRand(n, :, 2)) + 1;
                    idxZSD = squeeze(rayCouplingRand(n, :, 3)) + 1;
                    idxZSA = squeeze(rayCouplingRand(n, :, 4)) + 1;

                    % Loop over Ray within the cluster
                    for m = 1:numRay           
                        % Compute angles for the current ray
                        phi_n_m_AOD = phi_n_AOD + c_ASD * RayOffsetAngles(idxASD(m));
                        phi_n_m_AOA = phi_n_AOA + c_ASA * RayOffsetAngles(idxASA(m));
                        theta_n_m_ZOD = theta_n_ZOD + c_ZSD * RayOffsetAngles(idxZSD(m));
                        theta_n_m_ZOA = theta_n_ZOA + c_ZSA * RayOffsetAngles(idxZSA(m));

                        % Compute field components and terms
                        [F_rx_u_theta, F_rx_u_phi] = calc_Field(UeAntPattern, theta_n_m_ZOA, phi_n_m_AOA, zetaUeAnt);
                        term1 = [F_rx_u_theta; F_rx_u_phi]';

                        PHI_4 = squeeze(phiRand(n, m, :));
                        term2 = [exp(1j * PHI_4(1)), kappa_inv_sqrt * exp(1j * PHI_4(2)); ...
                                kappa_inv_sqrt * exp(1j * PHI_4(3)), exp(1j * PHI_4(4))];

                        [F_tx_s_theta, F_tx_s_phi] = calc_Field(BsAntPattern, theta_n_m_ZOD, phi_n_m_AOD, zetaBsAnt);
                        term3 = [F_tx_s_theta; F_tx_s_phi];

                        r_head_rx_n_m = [sin(theta_n_m_ZOA*d2pi)*cos(phi_n_m_AOA*d2pi); ...
                                        sin(theta_n_m_ZOA*d2pi)*sin(phi_n_m_AOA*d2pi); ...
                                        cos(theta_n_m_ZOA*d2pi)];
                        term4 = exp(1j * 2 * pi * r_head_rx_n_m' * d_bar_rx_u);

                        r_head_tx_n_m = [sin(theta_n_m_ZOD*d2pi)*cos(phi_n_m_AOD*d2pi); ...
                                        sin(theta_n_m_ZOD*d2pi)*sin(phi_n_m_AOD*d2pi); ...
                                        cos(theta_n_m_ZOD*d2pi)];
                        term5 = exp(1j * 2 * pi * r_head_tx_n_m' * d_bar_tx_s);

                        v_head_rx = [sin(v_direction(1)*d2pi)*cos(v_direction(2)*d2pi), ...
                                    sin(v_direction(1)*d2pi)*sin(v_direction(2)*d2pi), ...
                                    cos(v_direction(1)*d2pi)]';
                        term6 = exp(1j * 2*pi*(r_head_rx_n_m' * v_head_rx) * f_doppler .* timeSeq);

                        % Accumulate channel coefficients
                        H_u_s_n = H_u_s_n + term1 * term2 * term3 * term4 * term5 * term6;
                    end

                    % Update channel matrix
                    chanMatrix(s, u, firTapMap(n), :) = chanMatrix(s, u, firTapMap(n), :) + ...
                                                    reshape(sqrt(firPw(n)) / sqrt(numRay) * H_u_s_n, [1, 1, 1, nBatch]);
                end
            end
        end

        % Check normalize chanMatrix power, long term average should be 1
        % sqrt(sum(sum(sum(sum(abs(chanMatrix).^2))))/(nUeAnt*nBatch));

        % compare cdl time chan
        chan_diff = zeros(nBatch * nBsAnt * nUeAnt, 1);
        for batchIdx = 1:nBatch
            for txIdx = 1:nBsAnt
                for rxIdx = 1:nUeAnt
                    batchtxRxIdx = ((batchIdx-1)*nUeAnt + rxIdx-1)*nBsAnt + txIdx-1;
                    timeChan_ref = squeeze(chanMatrix(txIdx, rxIdx, :, batchIdx));
                    chan_gpu_pos = (cidUidIdx - 1) * timeChanSizePerLink + batchtxRxIdx*Ntap+ (1:Ntap);
                    tempChan_gpu = timeChan.re(chan_gpu_pos) + 1i*timeChan.im(chan_gpu_pos);
                    timeChan_gpu = zeros(size(timeChan_ref));
                    timeChan_gpu(firNzIdxGpu+1) = tempChan_gpu;
                    chan_diff(batchtxRxIdx + 1) = max(abs(timeChan_ref - timeChan_gpu));
                end
            end
        end
        checkRes(cidUidIdx, 5) = max(abs(chan_diff(:)));

        % compare freq chan on sc and/or prbg, only generate in the first batch
        % freqChanSc is saved and read per link
        freqChanSC_ref    = zeros(nBatch, nUeAnt, nBsAnt, N_sc);
        freqChanSc_diff   = zeros(nBatch*nBsAnt*nUeAnt, 1);
        freqChanPrbg_diff = zeros(nBatch*nBsAnt*nUeAnt, 1);
        if(runMode > 0)
            freqChanNormalizeCoe = 1; % sqrt(nBsAnt);
            if (runMode == 2)
                freqChanSc = h5read(tvFileName,['/freqChanScLink' num2str(cidUidIdx - 1)]);
            end
            for batchIdx = 1:nBatch
                if (runMode == 2) % CFR on SC calculated by GPU
                    freqChanPerBatchLen = nUeAnt * nBsAnt * N_sc;
                    freqChanPerBatchIdx = (batchIdx-1)*freqChanPerBatchLen+1:batchIdx*freqChanPerBatchLen;
                    freqChanScBatch = freqChanSc.re(freqChanPerBatchIdx) + 1i * freqChanSc.im(freqChanPerBatchIdx);
                end
                for txIdx = 1:nBsAnt
                    for rxIdx = 1:nUeAnt
                        txRxIdx = (rxIdx-1)*nBsAnt + txIdx-1;
                        cfr = calCfrFromCir(N_sc, N_sc_Prbg, scSpacingHz, f_samp, timeDelay, cfoHz, timeSeq(batchIdx), squeeze(chanMatrix(txIdx, rxIdx, :, batchIdx)), useFFT, N_fft, freqConvertType);
                        freqChanAntSC_ref = freqChanNormalizeCoe * cfr;
                        freqChanSC_ref(batchIdx, rxIdx, txIdx, :) = freqChanAntSC_ref;
                        if (scSampling > 1)
                            scIdx = 0:N_sc-1;
                            notCalSc = scIdx(mod(0:N_sc-1, scSampling) ~= 0);
                            freqChanAntSC_ref(notCalSc + 1) = 0;
                        end

                        if (runMode == 2)
                            scIdx = 0:N_sc-1;
                            notCalSc = scIdx(mod(0:N_sc-1, scSampling) ~= 0);
                            freqChanScBatch(notCalSc + 1) = 0;
                            freqChanSCGpu = freqChanScBatch(N_sc*txRxIdx + (1:N_sc));
                            freqChanSc_diff((batchIdx-1)*nUeAnt*nBsAnt+txRxIdx+1) = max(abs(freqChanAntSC_ref-freqChanSCGpu));
                        end

                        freqChanPrbg_ref = zeros(N_Prbg,1);
                        switch freqConvertType
                            case 0 % use first SC for CFR on the Prbg
                                freqChanPrbg_ref = freqChanAntSC_ref(1:N_sc_Prbg:end);
                            
                            case 1 % use center SC for CFR on the Prbg, e.g., sc 6 for sc 0,1,2,...,11
                                if(N_Prbg * N_sc_Prbg == N_sc)    
                                    freqChanPrbg_ref = freqChanAntSC_ref(N_sc_Prbg/2+1 : N_sc_Prbg : end);
                                else
                                    freqChanPrbg_ref(1:N_Prbg-1) = freqChanAntSC_ref(N_sc_Prbg/2+1 : N_sc_Prbg : end);
                                    freqChanPrbg_ref(N_Prbg)     = freqChanAntSC_ref((N_Prbg-1)*N_sc_Prbg + floor((N_sc - (N_Prbg-1)*N_sc_Prbg)/2) + 1);
                                end
                            
                            case 2 % use last SC for CFR on the Prbg
                                if(N_Prbg * N_sc_Prbg == N_sc)    
                                    freqChanPrbg_ref = freqChanAntSC_ref(N_sc_Prb : N_sc_Prbg : end);
                                else
                                    freqChanPrbg_ref(1:N_Prbg-1) = freqChanAntSC_ref(N_sc_Prbg : N_sc_Prbg : end);
                                    freqChanPrbg_ref(N_Prbg)     = freqChanAntSC_ref(N_sc);
                                end

                            case {3, 4} % use average SC for CFR on the Prbg
                                if(N_Prbg * N_sc_Prbg == N_sc)    
                                    freqChanPrbg_ref = transpose(mean(reshape(freqChanAntSC_ref, N_sc_Prbg, N_Prbg))) * N_sc_Prbg / length(0:scSampling:N_sc_Prbg-1);
                                else
                                    N_sc_last_Prbg = N_sc - (ceil(N_sc / N_sc_Prbg) - 1) * N_sc_Prbg;
                                    freqChanPrbg_ref(1:N_Prbg-1) = transpose(mean(reshape(freqChanAntSC_ref(1:(N_Prbg-1)*N_sc_Prbg), N_sc_Prbg, N_Prbg-1))) * N_sc_Prbg / length(0:scSampling:N_sc_Prbg-1);
                                    freqChanPrbg_ref(N_Prbg)     = mean(freqChanAntSC_ref((N_Prbg-1)*N_sc_Prbg+1:end)) * N_sc_last_Prbg / length(0:scSampling:N_sc_last_Prbg-1);
                                end
                            
                            otherwise
                                fprintf('Invalid freqConvertType %d\n', freqConvertType);
                        end
                                        
                        freqChanPrbgOffset = ((cidUidIdx - 1) * nBatch + batchIdx - 1) * freqChanPrbgSizePerLink;
                        freqChanPrbgGpu = freqChanPrbg.re(freqChanPrbgOffset + N_Prbg*txRxIdx + (1:N_Prbg)) + 1i * freqChanPrbg.im(freqChanPrbgOffset + N_Prbg*txRxIdx + (1:N_Prbg));
                        freqChanPrbg_diff((batchIdx-1)*nUeAnt*nBsAnt+txRxIdx+1) = max(abs(freqChanPrbg_ref-freqChanPrbgGpu));
                    end
                end
            end
            % results of freq chan on sc 
            checkRes(cidUidIdx, 7) = max(abs(freqChanSc_diff));
            % results of freq chan on prbg 
            checkRes(cidUidIdx, 8) = max(abs(freqChanPrbg_diff));
        end

        % generate and compare rx singal
        if (sigLenPerAnt > 0)
            if enableSwapTxRx
                nTxAnt = nUeAnt;
                nRxAnt = nBsAnt;
            else
                nTxAnt = nBsAnt;
                nRxAnt = nUeAnt;
            end
            if (procSigFreq == 0)  % processing signal in time domain
                rxSigOut_ref  = zeros(sigOutLenPerLink, 1);
                for batchIdx = 1:nBatch
                    if (batchCumLen(batchIdx) + 1 > sigLenPerAnt) % end of tx data samples
                        break;
                    end
                    batchSampleIdx = batchCumLen(batchIdx)+1 : min(sigLenPerAnt, batchCumLen(batchIdx + 1));
                    for txIdx = 1: nTxAnt % UL: nUeAnt, DL: nBsAnt
                        % prepare tx samples to be filtered by CDL time chan
                        sigInIdx = sigLenPerAnt*((cidUidIdx - 1) * nTxAnt + txIdx - 1) + batchSampleIdx;
                        temp_sigIn = txSigIn.re(sigInIdx) + 1i * txSigIn.im(sigInIdx);
                        N_padding = lenFir;
                        if(batchIdx == 1)
                            pad_sigIn = zeros(N_padding, 1);
                        else
                            pad_sigInIdx = sigLenPerAnt*((cidUidIdx - 1) * nTxAnt + txIdx - 1) + batchSampleIdx(1)-(N_padding:-1:1);
                            pad_sigIn = txSigIn.re(pad_sigInIdx) + 1i * txSigIn.im(pad_sigInIdx);
                        end
                        temp_sigIn = [pad_sigIn ;temp_sigIn];
                        for rxIdx = 1:nRxAnt
                            if (enableSwapTxRx)
                                timeChan_ref = squeeze(chanMatrix(rxIdx, txIdx, :, batchIdx));
                            else
                                timeChan_ref = squeeze(chanMatrix(txIdx, rxIdx, :, batchIdx));
                            end
                            % compare rx singal
                            if(sigLenPerAnt > 0)
                                temp_rxSigOut = filter(timeChan_ref, 1, temp_sigIn);
                                temp_rxSigOut = temp_rxSigOut(N_padding+1:end);
                                temp_rxSigOut = circshift(temp_rxSigOut, delay_samp); % add delay
                                rxTimeIdx = sigLenPerAnt*(rxIdx-1)+batchSampleIdx;
                                rxSigOut_ref(rxTimeIdx) = rxSigOut_ref(rxTimeIdx) + temp_rxSigOut;
                            end
                        end
                    end
                end

                % add CFO
                CFOseq = exp(1j*2*pi*([0:sigLenPerAnt-1]'*T_samp + ttiIdx * ttiLen)*cfoHz);
                rxSigOut_ref = rxSigOut_ref.*repmat(CFOseq(:), [nRxAnt, 1]);
                
                % read GPU data out
                rxSigOutOffset = (cidUidIdx - 1) * sigOutLenPerLink;
                rxSigOutGpu = rxSigOut.re(rxSigOutOffset + (1 : sigOutLenPerLink)) + 1i*rxSigOut.im(rxSigOutOffset + (1 : sigOutLenPerLink));
            else
                % freqChanAntSC_ref is the CFR on all N_sc
                % txSigIn is the input signal
                rxSigOut_ref  = zeros(nUeAnt * N_sc * nBatch, 1);
                for batchIdx = 1:nBatch
                    batchSampleIdx = (batchIdx-1)*N_sc+1 : batchIdx*N_sc;
                    for rxIdx = 1:nRxAnt
                        rxTimeIdx = (rxIdx-1) * nBatch * N_sc + batchSampleIdx;
                        for txIdx = 1:nBsAnt
                            sigInIdx = ((cidUidIdx - 1) * nTxAnt + txIdx - 1) * nBatch * N_sc + batchSampleIdx;
                            temp_sigIn = txSigIn.re(sigInIdx) + 1i * txSigIn.im(sigInIdx);
                            freqChanAntSC_ref = squeeze(freqChanSC_ref(batchIdx, rxIdx, txIdx, :));
                            rxSigOut_ref(rxTimeIdx) = rxSigOut_ref(rxTimeIdx) + temp_sigIn .* freqChanAntSC_ref;
                        end
                    end
                end
                % read GPU data out
                rxSigOutOffset = (cidUidIdx - 1) * nRxAnt * N_sc * nBatch;
                rxSigOutGpu = rxSigOut.re(rxSigOutOffset + (1 : nRxAnt * N_sc * nBatch)) + 1i*rxSigOut.im(rxSigOutOffset + (1 : nRxAnt * N_sc * nBatch));
            end
            % results of rx signal comparison
            checkRes(cidUidIdx, 6) = max(abs(rxSigOut_ref - rxSigOutGpu));
        end
    end
end

% print verification results
printTestRes(checkRes, runMode, sigLenPerAnt, precision, verbose);
end

% load power delay profile (PDP) in 38.141
function [DPA, PCP] = loadCdlParam(DelayProfile)
    % load CDL parameters
    switch DelayProfile
        case 'CDL-A' 
            load CDLparam;
            DPA = CDLparam.DPA_A;
            PCP = CDLparam.PCP_A;
        case 'CDL-B'
            load CDLparam;
            DPA = CDLparam.DPA_B;
            PCP = CDLparam.PCP_B;
        case 'CDL-C'
            load CDLparam;
            DPA = CDLparam.DPA_C;
            PCP = CDLparam.PCP_C;
        case 'CDL-D'
            load CDLparam;
            DPA = CDLparam.DPA_D;
            PCP = CDLparam.PCP_D;
        case 'CDL-E'
            load CDLparam;
            DPA = CDLparam.DPA_E;
            PCP = CDLparam.PCP_E;
        case 'CDL_customized'
            DPA = Chan.CDL_DPA;
            PCP = Chan.CDL_PCP;
        otherwise 
            error('chanType is not supported ... \n');
    end

end

function printTestRes(checkRes, runMode, sigLenPerAnt, precision, verbose)
maxAbsError = max(max(abs(checkRes(:,3:end))));
% FP16 processing input signal or freq chan typically has a higher error compared to FP32
% these threshold are emperical, can be adjusted
if( (maxAbsError < 0.015 && precision == 16) || (maxAbsError < 0.002 && precision == 32))
    fprintf("CDL verification PASS! maxAbsError is %e\n", maxAbsError);
else
    fprintf("CDL verification FAIL! maxAbsError is %e\n", maxAbsError);
end

if(verbose)
    fprintf("CDL verification results (max absolute error per link)\n")
    if(sigLenPerAnt <= 0) % no input tx signals
        switch(runMode)
            case 0
                fprintf("\t cid \t uid \t firNzIdx \t firNzPw \t timeChan\n");
                display(checkRes(:,1:5));
            case 1
                fprintf("\t cid \t uid \t firNzIdx \t firNzPw \t timeChan \t freqChanPrg\n");
                display(checkRes(:,[1:5 8:end]));
            case 2
                fprintf("\t cid \t uid \t firNzIdx \t firNzPw \t timeChan \t freqChanSc \t freqChanPrg\n");
                display(checkRes(:,[1:5 7:end]));
        end
    else
        switch(runMode)
            case 0
                fprintf("\t cid \t uid \t firNzIdx \t firNzPw \t timeChan \t sigOut\n");
                display(checkRes(:,1:6));
            case 1
                fprintf("\t cid \t uid \t firNzIdx \t firNzPw \t timeChan \t sigOut \t freqChanPrg\n");
                display(checkRes(:,[1:6 8:end]));
            case 2
                fprintf("\t cid \t uid \t firNzIdx \t firNzPw \t timeChan \t sigOut \tfreqChanSc \t freqChanPrg\n");
                display(checkRes); % all results printed
        end
    end
end
end

function cfr = calCfrFromCir(N_sc, N_sc_Prbg, scSpacingHz, f_samp, timeDelay, cfoHz, tBatch, cir, useFFT, N_fft, freqConvertType)
    % Calculate the CFR from the complex interference response (CIR)
    %
    % Inputs:
    % N_sc: number of subcarriers
    % scSpacingHz: subcarrier spacing in Hz
    % f_samp: sampling frequency in Hz
    % timeDelay: time domain delay in second
    % cfoHz: carrier offset frequency in Hz
    % cfrBatchRotationCfo: rotation of CFR per batch by CFO
    % cir: time chan (firNzLen x 1)
    % useFFT: use FFT for CFR calculation
    % N_fft: number of FFT points
    % freqConvertType: conversion of CFR on Sc to Prbg, removing freq ramping if freqConvertType == 4

    % Outputs:
    % cfr: freq chan

    N_Prbg = ceil(N_sc / N_sc_Prbg);
    N_sc_last_Prbg = N_sc - (N_Prbg - 1) * N_sc_Prbg;
    
    if(useFFT)
        tmp = fftshift(fft(cir, N_fft)); % only generate in the first batch
        cfr = tmp((N_fft - N_sc)/2 + (1:N_sc));
    else
        delays = (0:length(cir)-1) / f_samp;
        if (freqConvertType == 4)
            tmpScIdx = repmat([0:N_sc_Prbg:N_sc-N_sc_last_Prbg-1] + N_sc_Prbg/2, N_sc_Prbg, 1);
            tmpScIdx = tmpScIdx(:);
            scIdx = [tmpScIdx; ((N_Prbg-1) * N_sc_Prbg + N_sc_last_Prbg/2) * ones(N_sc_last_Prbg, 1)] - N_sc/2;
        else
            scIdx = (-N_sc/2 : N_sc/2-1)';
        end
        % add impact of CFO per SC: shift of sc frequency
        frequencies = scSpacingHz * scIdx - cfoHz;
        firNzDelayScFreq2Pi = exp(-1j * 2 * pi * frequencies * delays);
        cfr = firNzDelayScFreq2Pi * cir;
        % add impact of time delay per SC: phase ratation of cfr per Sc
        cfrPhaseShift = exp(-1j * 2 * pi * scIdx * round(timeDelay * f_samp) / N_fft);
        cfr = cfr .* cfrPhaseShift;
        % add impact of CFO per batch (whole SCs)
        cfrBatchRotationCfo = exp(1j * 2 * pi * tBatch * cfoHz);
        cfr = cfr * cfrBatchRotationCfo;
    end
end

function [mAnt, nAnt, pAnt] = findAntLoc(u, AntSize)
    % AntSize convention: [M_g, N_g, M, N, P]
    assert(numel(AntSize) == 5, ...
        'AntSize must be [M_g, N_g, M, N, P] (5 elements).');
    M = AntSize(3);
    N = AntSize(4);
    P = AntSize(5);

    u = u-1;
    pAnt = mod(u, P)+1;
    nAnt = mod(floor(u/P), N)+1;
    mAnt = mod(floor(u/(P*N)), M)+1;
    
end

function [d_h, d_v] = getElemSpacing(AntSpacing)
    % AntSpacing convention: [d_g_h, d_g_v, d_h, d_v]
    assert(numel(AntSpacing) == 4, ...
        'AntSpacing must be [d_g_h, d_g_v, d_h, d_v] (4 elements).');
    d_h = AntSpacing(3);
    d_v = AntSpacing(4);
end

function A_dB_3D = calc_A_dB_3D(theta, phi)

theta_3dB = 65;
SLA_v = 30;
A_dB_theta = -min(12*((theta-90)/theta_3dB)^2, SLA_v);

phi_3dB = 65;
A_max = 30;
A_dB_phi = -min(12*(phi/phi_3dB)^2, A_max);

A_dB_3D = -min(-(A_dB_theta + A_dB_phi), A_max);

end

function [F_theta, F_phi] = calc_Field(antPattern, theta, phi, zeta)
antPatternMap = {'isotropic', '38.901'};
switch(antPatternMap{antPattern + 1})
    case '38.901'
        G_E_max = 8; 
        A_dB_3D = G_E_max + calc_A_dB_3D(theta, phi);
        A = 10^(A_dB_3D/10);
    case 'isotropic'
        A = 1;
    otherwise
        error('antPettern is not supported ...\n')
end

F_theta = sqrt(A)*cos(zeta*pi/180);
F_phi = sqrt(A)*sin(zeta*pi/180);

end

function theta_out = normalize_theta(theta_in)  % normalize to [0, 180], Eq. 7.5-18

    theta_out = wrapTo360(theta_in);

    if theta_out > 180
        theta_out = 360 - theta_out;

    end
end
    