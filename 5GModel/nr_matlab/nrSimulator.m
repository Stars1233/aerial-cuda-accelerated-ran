% SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

function [SysPar, UE, gNB] = nrSimulator(SysPar)
%
% This is the main function to catch and print the error message from the core function.
%
% Input:    SysPar: structure with all configurations
%
% Output:   SysPar: structure with all configurations
%           UE: structure for UE
%           gNB: structure for gNB
%

try
    [SysPar, UE, gNB] = nrSimulatorCoreFunction(SysPar);
catch ME
    % Print a clean, readable error message
    fprintf(2, '\n========================================\n');
    fprintf(2, 'ERROR processing %s\n', SysPar.SimCtrl.genTV.TVname);
    fprintf(2, '========================================\n');
    fprintf(2, 'Message: %s\n', ME.message);
    
    if ~isempty(ME.identifier)
        fprintf(2, 'ID: %s\n', ME.identifier);
    end
    
    fprintf(2, '\nCall Stack:\n');
    for k = 1:numel(ME.stack)
        fprintf(2, '  [%d] %s (line %d)\n', k, ME.stack(k).name, ME.stack(k).line);
        if ~isempty(ME.stack(k).file)
            fprintf(2, '      %s\n', ME.stack(k).file);
        end
    end
    fprintf(2, '========================================\n\n');
    
    % Rethrow the caught exception so that the program will terminate
    rethrow(ME);
end
    
function [SysPar, UE, gNB] = nrSimulatorCoreFunction(SysPar)
%
% This is the core function to perform end to end UL simulation for NR
%
% Input:    SysPar: structure with all configurations
%
% Output:   SysPar: structure with all configurations
%
%           UE: structure for UE
%           gNB: structure for gNB

if nargin == 0
    SysPar = initSysPar;
end

nSysPar = length(SysPar);
if nSysPar > 1
    SysParList = SysPar;
    SysPar = SysParList{1}.SysParDl;
end

global SimCtrl;
SimCtrl = SysPar.SimCtrl;

SysPar.carrier = updateCarrier(SysPar.carrier);
SysPar         = updateAlloc(SysPar);
SysPar         = convert_CSI2_prms_from_FAPI_02_to_04(SysPar);

SimCtrl = SysPar.SimCtrl;

% % config the floating-point format used in signal processing that are implemented with Varry
% global  fp_flag; % fp_flag is the global variable that is used inside Varry
% switch SimCtrl.fp_flag_global 
%     case 0
%         fp_flag = 'fp64ieee754';
%     case 1
%         fp_flag = 'fp32ieee754';    
%     case 2
%         fp_flag = 'fp16ieee754';
%     case 3
%         fp_flag = 'fp16cleve';
%     case 4
%         fp_flag = 'fp16nv';
% end

% setup Python
if SimCtrl.ml.enable || strcmp(SysPar.Chan{1}.model_source, 'sionna')
    % set Python version
    try 
        eval('pyenv(Version=SimCtrl.ml.python_executable, ExecutionMode=SimCtrl.ml.python_ExecutionMode);');
    catch
        eval('terminate(pyenv)')
        eval('pyenv(Version=SimCtrl.ml.python_executable, ExecutionMode=SimCtrl.ml.python_ExecutionMode);');
    end
    eval("py.importlib.import_module('numpy');"); fprintf('Numpy was loaded into pyenv!\n')
    RTLD_NOW=2; % refer to https://www.mathworks.com/matlabcentral/answers/709228-issue-with-python-interface-to-tensorflow-and-other-third-party-libraries-in-mac-and-linux
    RTLD_DEEPBIND=8;
    flag=bitor(RTLD_NOW, RTLD_DEEPBIND);
    eval('py.sys.setdlopenflags(int32(flag));');
    % NOTE: don't change the ordering of the following import!!
    eval("py.importlib.import_module('matplotlib');"); % 'matplotlib' and 'mitsuba' are required by 'sionna'.
    eval("py.importlib.import_module('mitsuba');");
    eval("py.importlib.import_module('tensorflow');"); fprintf('TensorFlow was loaded into pyenv!\n')
    eval("py.tensorflow.random.set_seed(rng().Seed);"); fprintf('TensorFlow random seed was set!\n')
    if strcmp(SysPar.Chan{1}.model_source, 'sionna')
        eval("py.importlib.import_module('sionna');"); fprintf('Sionna was loaded into pyenv!\n')
    end
    if strcmp(SimCtrl.alg.LDPC_DMI_method,'ML')  
        SimCtrl.ML.models.model_LDPC_DMI = eval('py.tensorflow.keras.models.load_model(SimCtrl.alg.LDPC_DMI_ML_model_path)');
    end
end

N_UE = SysPar.SimCtrl.N_UE;
for idxUE = 1:N_UE
   UE{idxUE}      = initUE(SysPar, idxUE);
   Chan_UL{idxUE} = initChan(SysPar, idxUE, 'UL');
   Chan_DL{idxUE} = initChan(SysPar, idxUE, 'DL');
   if SimCtrl.enableUlDlCoSim
       Chan_DL{idxUE}.chanMatrix = permute(Chan_UL{idxUE}.chanMatrix, [2, 1, 3, 4]);
   end
end
SysPar.Chan_DL = Chan_DL;
SysPar.Chan_UL = Chan_UL;

% initialize RF impairments
if SimCtrl.enable_UL_Rx_RF_impairments
    SysPar.RF = initRF_UL_Rx(SysPar);
end
if SimCtrl.enable_DL_Tx_RF_impairments
    SysPar.RF = initRF_DL_Tx(SysPar);
end

% set interference for UL
if SimCtrl.N_Interfering_UE_UL>0
    for idxUE_inteference = 1:SimCtrl.N_Interfering_UE_UL
        Chan_UL_interference{idxUE_inteference} = initChan(SysPar, idxUE_inteference, 'UL', 1);
    end
end

N_SRS_CHEST_BUFF = SysPar.SimCtrl.bf.N_SRS_CHEST_BUFF;
chan_BF = [];
for idxSrsBuff = 0 : (N_SRS_CHEST_BUFF - 1)
    srsChEstBuffPrms              = SysPar.srsChEstBuff{idxSrsBuff + 1};
    chan_BF{idxSrsBuff + 1}       = initBfChan(srsChEstBuffPrms);
end

% combined SRS-BFW test vector
if isfield(SysPar.bfw{1},'srsTv')
     % Only run SRS TVs for non-loadOnly case
    if(~isfield(SysPar.bfw{1},'loadOnly'))
        % Run SRS TVs
        for srsTvNum = SysPar.bfw{1}.srsTv;
            % Save workspace
            filename = strcat('GPU_test_input/tempMatlabWorkspace', num2str(SysPar.bfw{1}.bfwTv(:,1)), '.mat');
            save(filename);
            if(ismember(srsTvNum, [8000:8999]) )
                testCompGenTV_srs(srsTvNum,'genTV');
            else
                testCompGenTV_ulmix(srsTvNum,'genTV');
            end
        end
        if size(SysPar.bfw{1}.bfwTv,2) > 1
            parfor i = 2:size(SysPar.bfw{1}.bfwTv,2)
                testCompGenTV_bfw(SysPar.bfw{1}.bfwTv(:,i),'genTV');
            end
        end

        % Load workspace
        clearvars -except filename
        load(filename);
        % recover SimCtrl from the workspace
        global SimCtrl;
        SimCtrl = SysPar.SimCtrl;
        delete(filename);
    end

    % Read SRS TV - support multiple SRS TVs per BFW
    % Collect all unique SRS TVs from all BFW groups
    allSrsTvs = [];
    for bfwIdx = 1:length(SysPar.bfw)
        if isfield(SysPar.bfw{bfwIdx}, 'srsTv')
            srsTvs = SysPar.bfw{bfwIdx}.srsTv;
            allSrsTvs = [allSrsTvs, srsTvs(:)']; % convert to row vector
        end
    end
    % Remove duplicates and sort
    allSrsTvs = unique(allSrsTvs);
    
    for srsTvNum = allSrsTvs
        if(srsTvNum > 21000)
            tvName = strcat(SimCtrl.genTV.tvDirName,'/TVnr_ULMIX_', num2str(srsTvNum,'%04d'), '_gNB_FAPI_s*.h5');
        else
            tvName = strcat(SimCtrl.genTV.tvDirName,'/TVnr_', num2str(srsTvNum,'%04d'), '_gNB_FAPI_s*.h5');
        end
        tvName = ls(tvName);
        tvName = split(tvName(1,:),'/');
        tvName = strcat(SimCtrl.genTV.tvDirName, '/', tvName{end});  % TV path with correct slot number
        srsChestMap = hdf5_load_chEst_map(tvName);
        % copy SRS to buffers
        % Open HDF5 file once for all channel estimate reads
        fileID = H5F.open(tvName, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');
        for i = 1:length(srsChestMap.rntis)
            srsIdx = srsChestMap.rntis(i);
            
            % Read channel estimate using low-level HDF5 API
            dsetID = H5D.open(fileID, ['/' srsChestMap.hest_fields{i}]);
            SRS_IND_Hest = H5D.read(dsetID);
            if isstruct(SRS_IND_Hest) && isfield(SRS_IND_Hest, 're') && isfield(SRS_IND_Hest, 'im')   % currently all SRS chEsts are complex
                SRS_IND_Hest = SRS_IND_Hest.re + 1i * SRS_IND_Hest.im;
            end
            H5D.close(dsetID);
            
            nPrg = size(SRS_IND_Hest,1);
            % Populate Channel Estimate buffers based on SRS RNTI
            chan_BF{srsIdx}(1:nPrg,:,:) = SRS_IND_Hest;
            
            % Get SRS valid PRG range for padding (SRS may not cover full BW)
            startValidPrg = srsChestMap.startValidPrg(i);  % 0-based
            nValidPrg = srsChestMap.nValidPrg(i);
            
            % Only perform padding if there are valid PRGs
            if nValidPrg > 0
                endValidPrg = startValidPrg + nValidPrg - 1;  % 0-based
                
                % Pad PRGs before startValidPrg by copying first valid PRG
                if startValidPrg > 0
                    for padPrgIdx = 1 : startValidPrg  % 1-based index
                        chan_BF{srsIdx}(padPrgIdx,:,:) = chan_BF{srsIdx}(startValidPrg + 1,:,:);
                    end
                end
                
                % Pad PRGs after endValidPrg by copying last valid PRG
                % First pad within nPrg (SRS buffer size)
                if endValidPrg < (nPrg - 1)
                    for padPrgIdx = (endValidPrg + 2) : nPrg  % 1-based index
                        chan_BF{srsIdx}(padPrgIdx,:,:) = chan_BF{srsIdx}(endValidPrg + 1,:,:);
                    end
                end
            end
            
            % Then pad if BFW buffer is larger than SRS buffer
            if(size(chan_BF{srsIdx},1) > nPrg)
                for padPrgIdx = (nPrg+1) : size(chan_BF{srsIdx},1)
                    chan_BF{srsIdx}(padPrgIdx,:,:) = chan_BF{srsIdx}(nPrg,:,:);
                end
            end
        end
        H5F.close(fileID);
    end    
    SimCtrl.genTV.srsTv = allSrsTvs; % Store all SRS TV numbers
end

SysPar.chan_BF = chan_BF;

gNB = initgNB(SysPar);



N_UE = SimCtrl.N_UE;
N_frame = SimCtrl.N_frame;
N_subframe = SysPar.carrier.N_subframe;
N_slot = SysPar.carrier.N_slot_subframe_mu;

if nSysPar > 1
    idxSysParList = 0;
end
testAlloc = SysPar.testAlloc;
TVname = SysPar.SimCtrl.genTV.TVname;

for idxFrame = 1:N_frame
    if SimCtrl.enable_print_idxFrame
        fprintf('Frame: %d\n',idxFrame)
    end
    SimCtrl.idxFrame = idxFrame;
    for idxSubframe = 1:N_subframe
        SimCtrl.idxSubframe = idxSubframe;
        for idxSlot = 1:N_slot
            SimCtrl.idxSlot = idxSlot;
            SimCtrl.global_idxSlot = 2^(SysPar.carrier.mu) * ((idxFrame-1)*10 + idxSubframe-1) + idxSlot;

            if SimCtrl.enable_freeze_tx_and_channel
                if (idxFrame==1) && (idxSubframe==1) && (idxSlot==1)
                    rng_state0 = rng; % store the rng state so that we can always reset rng to rng_state0, which actually freeze Tx and channel fading
                    if contains(Chan_UL{1}.type, 'TDL') && SimCtrl.timeDomainSim && strcmp(Chan_UL{1}.model_source, 'MATLAB5Gtoolbox')
                        for idxUE = 1:N_UE
                           Chan_UL{idxUE}.tdl.RandomStream = 'mt19937ar with seed';
                           Chan_UL{idxUE}.tdl.Seed = SimCtrl.seed;
                           reset(Chan_UL{idxUE}.tdl)
                           Chan_DL{idxUE}.tdl.RandomStream = 'mt19937ar with seed';
                           Chan_DL{idxUE}.tdl.Seed = SimCtrl.seed;
                           reset(Chan_DL{idxUE}.tdl)
                        end
                    elseif contains(Chan_UL{1}.type, 'CDL') && SimCtrl.timeDomainSim && strcmp(Chan_UL{1}.model_source, 'MATLAB5Gtoolbox')
                        for idxUE = 1:N_UE
                           Chan_UL{idxUE}.cdl.RandomStream = 'mt19937ar with seed';
                           Chan_UL{idxUE}.cdl.Seed = SimCtrl.seed;
                           reset(Chan_UL{idxUE}.cdl)
                           Chan_DL{idxUE}.cdl.RandomStream = 'mt19937ar with seed';
                           Chan_DL{idxUE}.cdl.Seed = SimCtrl.seed;
                           reset(Chan_DL{idxUE}.cdl)
                        end
                    end
                else
                    rng_state1 = rng; % store the rng state so that we can restore it before AWGN   
                    rng(rng_state0);
                    if contains(Chan_UL{1}.type, 'TDL') && SimCtrl.timeDomainSim && strcmp(Chan_UL{1}.model_source, 'MATLAB5Gtoolbox')
                        for idxUE = 1:N_UE
                           reset(Chan_UL{idxUE}.tdl)
                           reset(Chan_DL{idxUE}.tdl)
                        end
                    elseif contains(Chan_UL{1}.type, 'CDL') && SimCtrl.timeDomainSim && strcmp(Chan_UL{1}.model_source, 'MATLAB5Gtoolbox')
                        for idxUE = 1:N_UE
                           reset(Chan_UL{idxUE}.cdl)
                           reset(Chan_DL{idxUE}.cdl)
                        end 
                    end
                end
            elseif SimCtrl.enable_freeze_tx
                if (idxFrame==1) && (idxSubframe==1) && (idxSlot==1)
                    rng_state0 = rng; % store the rng state so that we can always reset rng to rng_state0, which actually freeze Tx
                else
                    rng_state1 = rng; % store the rng state so that we can restore it before AWGN   
                    rng(rng_state0);
                end

            end
                        
            if nSysPar > 1
                idxSysParList = idxSysParList + 1;
                SysParUl = [];
                if isfield(SysParList{idxSysParList}, 'SysParUl')
                    SysParUl = SysParList{idxSysParList}.SysParUl;
                    slotIdxLocal = SysParUl.SimCtrl.genTV.slotIdx;
                    SysParUl.SimCtrl = SysPar.SimCtrl;
                    SysParUl         = updateAlloc(SysParUl);
                    SimCtrl.results  = SysParUl.SimCtrl.results;
                    SimCtrl.genTV.slotIdx = slotIdxLocal;
                    SimCtrl.genTV.TVname = [TVname, '_UL'];
                else
                    SysParUl = resetAlloc(SysParUl, N_UE);
                end
                testAlloc = SysParUl.testAlloc;
            end
            
            % Run UL Simulation
            if testAlloc.ul
                
                % L2 Scheduler before Tx
                [gNB, UE] = L2Scheduler_UpdateSlotConfigBeforeTx(SimCtrl, gNB, UE);
                
               for idxUE = 1:N_UE
                   if nSysPar > 1
                       UE{idxUE} = updateUE(SysParUl, UE{idxUE}, idxUE);
                       if length(SysParUl.Chan) < idxUE
                           SysParUl.Chan{idxUE} = SysParUl.Chan{1};
                       end
                       Chan_UL{idxUE}.SNR = SysParUl.Chan{idxUE}.SNR;
                       Chan_UL{idxUE}.CFO = SysParUl.Chan{idxUE}.CFO;
                       Chan_UL{idxUE}.delay = SysParUl.Chan{idxUE}.delay;
                       Chan_UL{idxUE}.gain = SysParUl.Chan{idxUE}.gain;
                   end
                   UE{idxUE} = UEtransmitter(UE{idxUE}, idxUE);
                   if SimCtrl.timeDomainSim
                       txSamp = UE{idxUE}.Phy.tx.Xt;
                       if SimCtrl.alg.enableWeightedAverageCfo
                            Chan_UL{idxUE}.SNR = Chan_UL{idxUE}.SNR-(idxSlot-1)*15.0;
                       end
                       [chSamp, Chan_UL{idxUE}] = Channel(txSamp, Chan_UL{idxUE}, SimCtrl, SysPar.carrier);
                   else
                       % for non-PRACH
                       txSamp = UE{idxUE}.Phy.tx.Xtf;
                       [chSamp, Chan_UL{idxUE}] = Channel(txSamp, Chan_UL{idxUE}, SimCtrl, SysPar.carrier);
                       % for PRACH
                       if strcmp(Chan_UL{idxUE}.model_source,'sionna') 
                           chSamp_prach = zeros(size(txSamp,1),SysPar.carrier.Nant_gNB);
                       else
                           txSamp_prach = UE{idxUE}.Phy.tx.Xtf_prach;
                           [chSamp_prach, Chan_UL{idxUE}] = Channel(txSamp_prach, Chan_UL{idxUE}, SimCtrl, SysPar.carrier);
                       end
                   end
                   if idxUE == 1
                       rxSamp = chSamp;
                       if SimCtrl.timeDomainSim == 0
                           rxSamp_prach = chSamp_prach;
                       end
                   else
                       rxSamp = rxSamp + chSamp;
                       if SimCtrl.timeDomainSim == 0
                           rxSamp_prach = rxSamp_prach + chSamp_prach;
                       end
                   end
               end
               SNR = Chan_UL{1}.SNR;
               gNB.Phy.Chan_DL = Chan_DL; % update channel every slot, which is useful for genie channel extraction
               gNB.Phy.Chan_UL = Chan_UL;
               gNB.UE = UE;
               if SimCtrl.enable_snapshot_gNB_UE_into_SimCtrl
                   SimCtrl.gNBUE_snapshot = gNB;
               end
               if SimCtrl.genTV.enable_logging_carrier_and_channel_info
                   SimCtrl.channel_info.Chan_UL = Chan_UL;
                   SimCtrl.channel_info.Chan_DL = Chan_DL;
               end
               if SimCtrl.ml.dataset.enable_save_dataset
                    SimCtrl.ml.dataset.Gtruth = cellfun(@(x) {x.Phy.Gtruth}, UE);
                    SimCtrl.ml.dataset.Xtf = cellfun(@(x) {x.Phy.tx.Xtf}, UE);
                    SimCtrl.ml.dataset.channel_model = Chan_UL{1};
                    if SimCtrl.N_Interfering_UE_UL>0
                        SimCtrl.ml.dataset.interference_channel_model = Chan_UL_interference{1};
                    end
               end
               if SimCtrl.enable_freeze_tx_and_channel || SimCtrl.enable_freeze_tx
                   if ~((idxFrame==1) && (idxSubframe==1) && (idxSlot==1))
                       rng(rng_state1);
                   end
               end
               
                if nSysPar > 1
                    gNB = updategNB(SysParUl, gNB);
                end 
               rxSamp_scale_factor = sqrt(db2pow(SimCtrl.UL_Rx_power_scaling_dB));
               if SimCtrl.timeDomainSim
                   rxSamp_noNoise = rxSamp;
                   if SimCtrl.enable_UL_Rx_RF_impairments
                       rxSamp = addRfImpairments_UL_Rx(SysPar, Chan_UL{1}, rxSamp);
                   else
                       rxSamp = addNoise(rxSamp, SNR);
                       if SimCtrl.N_Interfering_UE_UL>0
                           SIR = Chan_UL_interference{1}.SIR;
                           [rxSamp, Chan_UL_interference] = addInterference(rxSamp, SIR,Chan_UL_interference, SysPar.carrier, SimCtrl);
                           gNB.Phy.interfChan_UL = Chan_UL_interference; % update storing interference channels in gNB.Phy per slot so that we can extract genie interference channel in detPusch.m
                       end
                   end
                   rxSamp = rxSamp*rxSamp_scale_factor;
                   rxSamp_prach_noNoise = [];
                   rxSamp_prach = [];  
                   gNB = gNBreceiver(gNB, rxSamp, rxSamp_prach, rxSamp_noNoise, rxSamp_prach_noNoise);
               else
                   if SimCtrl.enable_UL_Rx_RF_impairments
                       error('timeDomainSim is needed for RF impairments simulations!')
                   end
                   rxSamp_noNoise = rxSamp*rxSamp_scale_factor;
                   if SimCtrl.N_Interfering_UE_UL>0
                       SIR = Chan_UL_interference{1}.SIR;
                       [rxSamp, Chan_UL_interference] = addInterference(rxSamp, SIR,Chan_UL_interference, SysPar.carrier, SimCtrl);
                       gNB.Phy.interfChan_UL = Chan_UL_interference; % update storing interference channels in gNB.Phy per slot so that we can extract genie interference channel in detPusch.m
                   end
                   rxSamp = addNoise(rxSamp, SNR)*rxSamp_scale_factor;
                   rxSamp_prach_noNoise = rxSamp_prach*rxSamp_scale_factor;
                   rxSamp_prach = addNoise(rxSamp_prach, SNR)*rxSamp_scale_factor;
                   gNB = gNBreceiver(gNB, rxSamp, rxSamp_prach, rxSamp_noNoise, rxSamp_prach_noNoise);
               end
               
                SimCtrl = collectData(SimCtrl, UE, gNB, SysPar, Chan_DL, Chan_UL);
                
                if nSysPar > 1
                    fprintf('\nslot:   ssb pdcch pdsch csirs prach pucch pusch srs\n');
                    fprintf('----------------------------------------------------\n');
                    fprintf('%4d: %4d %4d  %4d  %4d  %4d  %4d  %4d %4d\n\n', slotIdxLocal, testAlloc.ssb, testAlloc.pdcch,...
                        testAlloc.pdsch, testAlloc.csirs, testAlloc.prach, testAlloc.pucch,...
                        testAlloc.pusch, testAlloc.srs);                    
                end
                errFlag = 0;
                printFlag = SimCtrl.printCheckDet;
                if printFlag
                    if (checkDetError(testAlloc, SimCtrl, printFlag))
                        errFlag = 1;
                    end
                end
            end


            if nSysPar > 1
                SysParDl = [];
                if isfield(SysParList{idxSysParList}, 'SysParDl')
                    SysParDl = SysParList{idxSysParList}.SysParDl;
                    slotIdxLocal = SysParDl.SimCtrl.genTV.slotIdx;
                    SysParDl.SimCtrl = SysPar.SimCtrl;
                    SysParDl         = updateAlloc(SysParDl);
                    SimCtrl.results  = SysParDl.SimCtrl.results;
                    SimCtrl.genTV.slotIdx = slotIdxLocal;
                    SimCtrl.genTV.TVname = [TVname, '_DL'];
                else
                    SysParDl = resetAlloc(SysParDl, N_UE);
                end
                testAlloc = SysParDl.testAlloc;
            end

            % Run DL Simulation
            if testAlloc.dl
                if nSysPar > 1
                    gNB = updategNB(SysParDl, gNB);
                end
                gNB = gNBtransmitter(gNB);
                if SimCtrl.enable_snapshot_gNB_UE_into_SimCtrl
                   SimCtrl.gNBUE_snapshot = gNB;
                end
                if SimCtrl.enableUeRx
                    if SimCtrl.timeDomainSim
                        txSamp = gNB.Phy.tx.Xt;
                        if SimCtrl.enable_DL_Tx_RF_impairments
                            txSamp = addRfImpairments_DL_Tx(SysPar, txSamp);
                        end
                    else
                        txSamp = gNB.Phy.tx.Xtf;
                        if SimCtrl.enable_DL_Tx_RF_impairments
                            error('DL Tx RF impairments can only be added in time domain sim!')
                        end
                    end
                    for idxUE = 1:N_UE
                        if nSysPar > 1
                            UE{idxUE} = updateUE(SysParDl, UE{idxUE}, idxUE);
                        end
                        [chSamp, Chan_DL{idxUE}] = Channel(txSamp, Chan_DL{idxUE}, SimCtrl, SysPar.carrier);
                        SNR = Chan_DL{idxUE}.SNR;
                        rxSamp_noNoise = chSamp;
                        rxSamp = addNoise(chSamp, SNR);
                        UE{idxUE} = UEreceiver(UE{idxUE}, rxSamp, rxSamp_noNoise);
                    end
                end

                SimCtrl = collectData(SimCtrl, UE, gNB, SysPar, Chan_DL, Chan_UL);

                if nSysPar > 1
                    fprintf('\nslot:   ssb pdcch pdsch csirs prach pucch pusch srs\n');
                    fprintf('----------------------------------------------------\n');
                    fprintf('%4d: %4d %4d  %4d  %4d  %4d  %4d  %4d %4d\n\n', slotIdxLocal, testAlloc.ssb, testAlloc.pdcch,...
                        testAlloc.pdsch, testAlloc.csirs, testAlloc.prach, testAlloc.pucch,...
                        testAlloc.pusch, testAlloc.srs);
                end
                errFlag = 0;
                printFlag = SimCtrl.printCheckDet;
                if printFlag
                    if (checkDetError(testAlloc, SimCtrl, printFlag))
                        errFlag = 1;
                    end
                end
            end

%             SimCtrl = collectData(SimCtrl, UE, gNB, SysPar);                  

            slotIdxFrame = (idxSubframe-1)*N_slot + idxSlot-1;
            if slotIdxFrame == (SimCtrl.N_slot_run-1)
                SysPar.SimCtrl = SimCtrl;
                SysPar.Chan_UL = Chan_UL;
                SysPar.Chan_DL = Chan_DL;
                return;
            end

        end % idxSlot = 1:N_slot
    end  % idxSubframe = 1:N_subframe
end % idxFrame = 1:N_frame

%if SimCtrl.runUL
%    outputResult(SimCtrl);
%end

SysPar.SimCtrl = SimCtrl;
SysPar.Chan_DL = Chan_DL;
SysPar.Chan_UL = Chan_UL;

% clear global SimCtrl to release memory
clear global SimCtrl

return

function [gNB, UE] = L2Scheduler_UpdateSlotConfigBeforeTx(SimCtrl, gNB, UE)
    N_UE = SimCtrl.N_UE;

    if (SimCtrl.puschHARQ.EnableAutoHARQ)
        for idxUE = 1:N_UE
            for k = 1:length(UE{idxUE}.Mac.Config.pusch)
                % Get HARQ feedback from gNB
                harqProcessID = UE{idxUE}.Mac.Config.pusch{k}.harqProcessID;
                n_rnti = UE{idxUE}.Mac.Config.pusch{k}.RNTI;
                rxAttemptCount = 0;
                if (isfield(gNB.Phy.Config.pusch.harqState,['hpid_',num2str(harqProcessID),'_rnti_',num2str(n_rnti)]))
                    rxAttemptCount = gNB.Phy.Config.pusch.harqState.(['hpid_',num2str(harqProcessID),'_rnti_',num2str(n_rnti)]).rxAttemptCount;
                end

                % Configure ndi and rv based on rxAttemptCount
                ndi = rxAttemptCount == 0;
                rvIndexIdx = mod(rxAttemptCount,length(SimCtrl.puschHARQ.rvTable))+1;

                % Automatically set ndi and rv for UE for this PUSCH
                UE{idxUE}.Mac.Config.pusch{k}.newDataIndicator = ndi;
                UE{idxUE}.Mac.Config.pusch{k}.rvIndex = SimCtrl.puschHARQ.rvTable(rvIndexIdx);

                % Automatically set ndi and rv for gNB for this PUSCH
                gNB_idxPusch = UE{idxUE}.Mac.Config.pusch{k}.gNB_idxPusch;
                gNB.Mac.Config.pusch{gNB_idxPusch}.newDataIndicator = ndi;
                gNB.Mac.Config.pusch{gNB_idxPusch}.rvIndex = SimCtrl.puschHARQ.rvTable(rvIndexIdx);
                
                if (rxAttemptCount > 0) && isfield(UE{idxUE}.Mac.Config.pusch{k}, 'mcsIndex_retx')
                    UE{idxUE}.Mac.Config.pusch{k}.mcsIndex = UE{idxUE}.Mac.Config.pusch{k}.mcsIndex_retx(rxAttemptCount);
                    UE{idxUE}.Mac.Config.pusch{k}.qamModOrder = UE{idxUE}.Mac.Config.pusch{k}.qamModOrder_retx(rxAttemptCount);
                    UE{idxUE}.Mac.Config.pusch{k}.rbSize = UE{idxUE}.Mac.Config.pusch{k}.rbSize_retx(rxAttemptCount);
                    UE{idxUE}.Mac.Config.pusch{k}.targetCodeRate = UE{idxUE}.Mac.Config.pusch{k}.targetCodeRate_retx(rxAttemptCount);
                    UE{idxUE}.Mac.Config.pusch{k}.TBSize = UE{idxUE}.Mac.Config.pusch{k}.TBSize_retx(rxAttemptCount);
                    
                    gNB.Mac.Config.pusch{gNB_idxPusch}.mcsIndex = UE{idxUE}.Mac.Config.pusch{k}.mcsIndex_retx(rxAttemptCount);
                    gNB.Mac.Config.pusch{gNB_idxPusch}.qamModOrder = UE{idxUE}.Mac.Config.pusch{k}.qamModOrder_retx(rxAttemptCount);
                    gNB.Mac.Config.pusch{gNB_idxPusch}.rbSize = UE{idxUE}.Mac.Config.pusch{k}.rbSize_retx(rxAttemptCount);
                    gNB.Mac.Config.pusch{gNB_idxPusch}.targetCodeRate = UE{idxUE}.Mac.Config.pusch{k}.targetCodeRate_retx(rxAttemptCount);
                    gNB.Mac.Config.pusch{gNB_idxPusch}.TBSize = UE{idxUE}.Mac.Config.pusch{k}.TBSize_retx(rxAttemptCount);
                end
            end
        end
    end
return




