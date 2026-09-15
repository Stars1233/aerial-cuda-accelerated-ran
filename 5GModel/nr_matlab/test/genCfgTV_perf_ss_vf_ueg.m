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

function [errCnt, nTV] = genCfgTV_perf_ss_vf_ueg(spreadsheet, tc_list, is_fapi)

if nargin == 0
    error('Please specify the spreadsheet to use for the generation');
elseif nargin == 1
    tc_list = [];
    is_fapi = 0;
elseif nargin == 2
    is_fapi = 0;
elseif nargin > 3
    error('Only 3 arguments can be accepted');
end

% Check for the spreadsheet in some specific paths
    if (exist(spreadsheet,'file'))
        % Use the filename exactly as specified
    else
        fprintf(['Unable to find spreadsheet ',spreadsheet,'\n']);
        error('Exiting');
    end

    [status,msg] = mkdir('GPU_test_input');
    
    % Hashmap for modulation order given MCS table and index
    
    k_set_mcs_1 = 0:28;
    k_set_mcs_2 = 0:27;
    k_set_mcs_3 = 0:28;
    
    v_set_mcs_1 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6];
    v_set_mcs_2 = [2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8];
    v_set_mcs_3 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6];
    
    mod_ord_1 = containers.Map(k_set_mcs_1, v_set_mcs_1);
    mod_ord_2 = containers.Map(k_set_mcs_2, v_set_mcs_2);
    mod_ord_3 = containers.Map(k_set_mcs_3, v_set_mcs_3);
    
    % Hashmap for max number of PRB in FT grid for given numerology index

    k_set_mu = [0, 1];
    v_set_mu = [106, 273];
    grid_mu_table = containers.Map(k_set_mu, v_set_mu);
    
    sheets = sheetnames(spreadsheet);
    
    if isempty(tc_list)
        tc_list = sheets;
    end

    errCnt = 0;
    nTV = 0;
    
    for i= 1:length(sheets)
    % check last 5 characters match PUSCH or PDSCH

        if sum(contains(tc_list, sheets{i})) == 0
            continue
        end
        
        if (strlength(sheets{i}) < 5)
            continue
        end
        
        usecase_name = sheets{i}(1:3);
        pipeline_name = sheets{i}(end-4:end);
%         display(pipeline_name)

        
        if (pipeline_name == "PUSCH" || pipeline_name == "PDSCH")
            fprintf("\n**** Start Processing Sheet: %s ****\n", sheets{i});
            data = readtable(spreadsheet,'Sheet',sheets{i}, 'PreserveVariableNames', true);
            data = data(1:17,3:end);

            [~,nCases] = size(data);
            headers = readcell(spreadsheet,'Sheet',sheets{i}, 'Range', [2 2 2 nCases + 3]);
            headers = headers(1,3:end);
            
            duplex = readcell(spreadsheet,'Sheet',sheets{i}, 'Range', '20:20');
            duplex = duplex(1,3:end);
            
            if duplex{1} == "FDD"
                SysPar.carrier.duplex = 0;
            end

            for tc = 1 : nCases
                
                tc_str = headers{tc};
                
                if ismissing(headers{tc})
                    continue;
                end
                
                UE_group_num = 1;
                for ue_grp_case = tc + 1 : nCases
                    if ismissing(headers{ue_grp_case})
                        UE_group_num = UE_group_num + 1;
                    else
                        break;
                    end
                end
                    

                nTV = nTV + 1;                
                SysPar = initSysPar;

                if (pipeline_name == "PDSCH")
                    buffer = SysPar.pdsch{1};
                else
                    buffer = SysPar.pusch{1};
                end

                SysPar = [];

                tc_par_mu = 0;
                tc_par_gNB_rf_chains = 0;
                idxUE = 1;

                tc_par_PRB_pre = 0;
                                
                for ue_group = tc : tc + UE_group_num - 1    
                    % extracting parameters from the spreadsheet
                    tc_par_grid = data{2, ue_group};
                    tc_par_ueg = data{4, ue_group}-1;
                    tc_par_mu = data{6, ue_group};
                    tc_par_gNB_rf_chains = data{7, ue_group};
                    tc_par_PRB = data{9, ue_group};
                    tc_par_symbols = data{10, ue_group};
                    tc_par_dmrs_symbols = data{11, ue_group};
                    tc_par_dmrs_max_len = data{12, ue_group};
                    tc_par_mcs_table = data{14, ue_group};
                    tc_par_mcs_index = data{15, ue_group};
                    tc_par_users = data{16, ue_group};
                    tc_par_layers_per_user = data{17, ue_group};
                    
                    
                    
                    tc_par_mod_order = nan;
                    
                    if tc_par_mcs_table == 1
                        tc_par_mod_order = mod_ord_1(tc_par_mcs_index);
                    else
                        if tc_par_mcs_table == 2
                            tc_par_mod_order = mod_ord_2(tc_par_mcs_index);
                        else
                            if tc_par_mcs_table == 3
                                tc_par_mod_order = mod_ord_3(tc_par_mcs_index);
                            end
                        end
                    end         
                    
                    if (pipeline_name == "PDSCH")
                        for nUE = 1:tc_par_users
                            SysPar.pdsch{idxUE} = buffer;
                            
                            % enable beamforming for V08
                            if ismember(usecase_name(1:3),["V08", "V15", "V16"])
                                switch tc_par_layers_per_user
                                    case 4
                                        SysPar.pdsch{idxUE}.prcdBf = 12;
                                    case 2
                                        SysPar.pdsch{idxUE}.prcdBf = 8;
                                    case 1
                                        SysPar.pdsch{idxUE}.prcdBf = 4;
                                    otherwise
                                        SysPar.pdsch{idxUE}.prcdBf = 0;
                                end
                            end
                            
                            SysPar.pdsch{idxUE}.rbStart = tc_par_PRB_pre;
                            SysPar.pdsch{idxUE}.rbSize = tc_par_PRB;
                            SysPar.pdsch{idxUE}.StartSymbolIndex = 14 - tc_par_symbols;
                            SysPar.pdsch{idxUE}.NrOfSymbols = tc_par_symbols; %  + tc_par_dmrs_symbols;
                            second_dmrs_symbol = tc_par_dmrs_max_len - 1;
                            SysPar.pdsch{idxUE}.DmrsSymbPos = [0 0 1 second_dmrs_symbol 0 0 0 0 0 0 0 0 0 0];
                            SysPar.pdsch{idxUE}.nrOfLayers = tc_par_layers_per_user;
                            SysPar.pdsch{idxUE}.mcsTable = tc_par_mcs_table - 1;
                            SysPar.pdsch{idxUE}.mcsIndex = tc_par_mcs_index;
                            SysPar.pdsch{idxUE}.portIdx = (nUE-1)* tc_par_layers_per_user + [0:tc_par_layers_per_user-1];
                            SysPar.pdsch{idxUE}.portIdx = mod((nUE-1)* tc_par_layers_per_user + [0:tc_par_layers_per_user-1], 8);    
                            SysPar.pdsch{idxUE}.SCID = floor((nUE-1)*tc_par_layers_per_user/8);
                            SysPar.pdsch{idxUE}.idxUeg = tc_par_ueg;
                            SysPar.pdsch{idxUE}.idxUE = idxUE-1;
                            SysPar.pdsch{idxUE}.seed = sum(int32(tc_str)) + idxUE-1;
                            idxUE = idxUE + 1;
                        end
                        tc_par_PRB_pre = tc_par_PRB_pre + tc_par_PRB;
                        
                        % with 3 CSI-RS parameters
                        if ismember(usecase_name(1:3), ["V08", "F08"])
                            csirs_cfg = {...
                                         % cfg#  CSI Row CDM Density RB0  nRB  sym0  sym1   nID  FreqDomain
                                            % TRS on symbol 6
                                            10,  0,  1,  0,   3,     0,   273,   6,    8,   41,   {ones(1,12)};
                                            % TRS on symbol 10
                                            11,  0,  1,  0,   3,     0,   273,   10,   8,   41,   {ones(1,12)};
                                            % F08 at symbol 12
                                            12,  1,  4,  1,   2,     0,   273,   12,   8,   41,   {ones(1,12)}
                                        };
                            if ismember(tc_str, ["V08-DS-42", "V08-DS-43"]) % 20 MHz
                                csirs_cfg = {...
                                         % cfg#  CSI Row CDM Density RB0  nRB  sym0  sym1   nID  FreqDomain
                                            % F08 at symbol 12
                                            12,  1,  4,  1,   2,     0,   51,   12,   8,   41,   {ones(1,12)};
                                        };
                            end
                        
                            % config CSI-RS
                            [nCsirs, ~] = size(csirs_cfg);
                            for idx = 1:nCsirs
                                idxCfg = idx;
                                SysPar.csirs{idx} = cfgCsirs;
                                % cfg#  CSI Row CDM Density RB0  nRB  sym0  sym1   nID  FreqDomain
                                SysPar.csirs{idx}.CSIType = csirs_cfg{idxCfg, 2};
                                SysPar.csirs{idx}.Row = csirs_cfg{idxCfg, 3};
                                SysPar.csirs{idx}.CDMType = csirs_cfg{idxCfg, 4};
                                SysPar.csirs{idx}.FreqDensity = csirs_cfg{idxCfg, 5};
                                SysPar.csirs{idx}.StartRB = csirs_cfg{idxCfg, 6};
                                SysPar.csirs{idx}.NrOfRBs = csirs_cfg{idxCfg, 7};
                                SysPar.csirs{idx}.SymbL0 = csirs_cfg{idxCfg, 8};
                                SysPar.csirs{idx}.SymbL1 = csirs_cfg{idxCfg, 9};
                                SysPar.csirs{idx}.ScrambId = csirs_cfg{idxCfg, 10};
                                SysPar.csirs{idx}.FreqDomain = cell2mat(csirs_cfg{idxCfg, 11});
                                SysPar.csirs{idx}.idxUE = idx-1;
                            end
                            SysPar.testAlloc.csirs = nCsirs;
                        else
                            SysPar.testAlloc.csirs = 0;
                        end
                  
                        cfgFile = ['GPU_test_input/TV_cuphy_', tc_str, '_slot', num2str(0), '_MIMO', num2str(tc_par_users * tc_par_layers_per_user), 'x',num2str(tc_par_gNB_rf_chains), '_PRB', num2str(tc_par_PRB), '_DataSyms', num2str(tc_par_symbols - tc_par_dmrs_symbols),  '_qam' , num2str(2^tc_par_mod_order), '.yaml'];
                        tvFile =  ['TV_cuphy_', tc_str, '_slot', num2str(0), '_MIMO', num2str(tc_par_users * tc_par_layers_per_user), 'x',num2str(tc_par_gNB_rf_chains), '_PRB', num2str(tc_par_PRB), '_DataSyms', num2str(tc_par_symbols - tc_par_dmrs_symbols),  '_qam' , num2str(2^tc_par_mod_order)];
                    else
                                                
                        for nUE = 1:tc_par_users
                            
                            SysPar.pusch{idxUE} = buffer;
                            
                            if ismember(tc_str, ...
                                   ["V15-US-03", "V15-US-04", "V15-US-13", "V15-US-14", "V15-US-23", "V15-US-24" ...
                                    "V16-US-03", "V16-US-04", "V16-US-13", "V16-US-14", "V16-US-23", "V16-US-24"])
                                SysPar.SimCtrl.alg.enableDftSOfdm = 1;
                                if tc_par_ueg < 2 % first 2 UEGs DFT-s-OFDM
                                    SysPar.pusch{idxUE}.TransformPrecoding = 0;
                                else  % next 3 UEGs CP-OFDM
                                    SysPar.pusch{idxUE}.TransformPrecoding = 1;
                                end                                 
                            end
                            
                            SysPar.pusch{idxUE}.rbStart = tc_par_PRB_pre;
                            SysPar.pusch{idxUE}.rbSize = tc_par_PRB;
                            SysPar.pusch{idxUE}.StartSymbolIndex = 14 - tc_par_symbols;
                            SysPar.pusch{idxUE}.NrOfSymbols = tc_par_symbols; %  + tc_par_dmrs_symbols;
                            second_dmrs_symbol = tc_par_dmrs_max_len - 1;
                            SysPar.pusch{idxUE}.DmrsSymbPos = [0 0 1 second_dmrs_symbol 0 0 0 0 0 0 0 0 0 0];
                            SysPar.pusch{idxUE}.nrOfLayers = tc_par_layers_per_user;
                            SysPar.pusch{idxUE}.mcsTable = tc_par_mcs_table - 1;
                            SysPar.pusch{idxUE}.mcsIndex = tc_par_mcs_index;
                            SysPar.pusch{idxUE}.portIdx = (nUE-1)* tc_par_layers_per_user + [0:tc_par_layers_per_user-1];
                            SysPar.pusch{idxUE}.idxUeg = tc_par_ueg;
                            SysPar.pusch{idxUE}.idxUE = idxUE-1;
                            SysPar.pusch{idxUE}.seed = sum(int32(tc_str)) + idxUE-1;
                            
                            
                            % containing UCI  4 + 33 + 5 
                            if ismember(sheets{i}(1:3), ["U08", "V16"])
                                SysPar.pusch{idxUE}.pduBitmap = 2^0 + 2^1 + 2^5;
                                SysPar.pusch{idxUE}.harqAckBitLength = 4;   % number of HARQ bits 
                                SysPar.pusch{idxUE}.csiPart1BitLength = 33;
                                SysPar.pusch{idxUE}.alphaScaling = 3;
                                SysPar.pusch{idxUE}.betaOffsetHarqAck = 11; % Default value in Matlab 5G toolbox = 11
                                SysPar.pusch{idxUE}.betaOffsetCsi1 = 13;    % Default value in Matlab 5G toolbox = 13
                                SysPar.pusch{idxUE}.betaOffsetCsi2 = 13;
                                SysPar.pusch{idxUE}.rank            = 1; % CSI-P2: 5 bits
                                SysPar.pusch{idxUE}.rankBitOffset   = 0;
                                SysPar.pusch{idxUE}.rankBitSize     = 2;
                            end
                            
                            % F09 containing UCI  4 + 6 + 5 
                            if ismember(sheets{i}(1:3), ["U09", "V15"])
                                SysPar.pusch{idxUE}.pduBitmap = 2^0 + 2^1 + 2^5;
                                SysPar.pusch{idxUE}.harqAckBitLength = 4;   % number of HARQ bits 
                                SysPar.pusch{idxUE}.csiPart1BitLength = 6;
                                SysPar.pusch{idxUE}.alphaScaling = 3;
                                SysPar.pusch{idxUE}.betaOffsetHarqAck = 11; % Default value in Matlab 5G toolbox = 11
                                SysPar.pusch{idxUE}.betaOffsetCsi1 = 13;    % Default value in Matlab 5G toolbox = 13
                                SysPar.pusch{idxUE}.betaOffsetCsi2 = 13;
                                SysPar.pusch{idxUE}.rank            = 1; % CSI-P2: 5 bits
                                SysPar.pusch{idxUE}.rankBitOffset   = 0;
                                SysPar.pusch{idxUE}.rankBitSize     = 2;
                            end
                            
                            
                            % F14 
                            if sheets{i}(1:3) == "U14"
                                SysPar.pusch{idxUE}.pduBitmap = 2^0 + 2^1 + 2^5;
                                SysPar.pusch{idxUE}.harqAckBitLength = 4;   % number of HARQ bits 
                                SysPar.pusch{idxUE}.csiPart1BitLength = 37;
                                SysPar.pusch{idxUE}.alphaScaling = 3;
                                SysPar.pusch{idxUE}.betaOffsetHarqAck = 11; % Default value in Matlab 5G toolbox = 11
                                SysPar.pusch{idxUE}.betaOffsetCsi1 = 13;    % Default value in Matlab 5G toolbox = 13
                                SysPar.pusch{idxUE}.betaOffsetCsi2 = 13;
                                SysPar.pusch{idxUE}.rank            = 1;  % CSI-P2: 5 bits
                                SysPar.pusch{idxUE}.rankBitOffset   = 0;
                                SysPar.pusch{idxUE}.rankBitSize     = 2;
                            end
                            
                            idxUE = idxUE + 1;
                        end
                        tc_par_PRB_pre = tc_par_PRB_pre + tc_par_PRB;
                        cfgFile = ['GPU_test_input/TV_cuphy_', tc_str, '_snrdb40.00_MIMO', num2str(tc_par_users * tc_par_layers_per_user), 'x',num2str(tc_par_gNB_rf_chains), '_PRB', num2str(tc_par_PRB), '_DataSyms', num2str(tc_par_symbols - tc_par_dmrs_symbols),  '_qam' , num2str(2^tc_par_mod_order), '.yaml'];
                        tvFile =  ['TV_cuphy_', tc_str, '_snrdb40.00_MIMO', num2str(tc_par_users * tc_par_layers_per_user), 'x',num2str(tc_par_gNB_rf_chains), '_PRB', num2str(tc_par_PRB), '_DataSyms', num2str(tc_par_symbols - tc_par_dmrs_symbols),  '_qam' , num2str(2^tc_par_mod_order)];
                    end
                end

                SysPar.testAlloc.ssb = 0;
                SysPar.testAlloc.pdcch = 0;                
                SysPar.testAlloc.prach = 0;
                SysPar.testAlloc.pucch = 0;
                SysPar.testAlloc.srs = 0;

                SysPar.SimCtrl.N_slot_run = 1;
                SysPar.SimCtrl.genTV.slotIdx = 0;
                SysPar.SimCtrl.genTV.FAPI = is_fapi;
                SysPar.SimCtrl.N_UE = idxUE-1;
                
                SysPar.carrier.N_grid_size_mu = tc_par_grid; % grid_mu_table(tc_par_mu);
                SysPar.carrier.mu = tc_par_mu;
                SysPar.carrier.Nant_gNB = tc_par_gNB_rf_chains;
                SysPar.carrier.Nant_UE = 4;

                if (pipeline_name == "PDSCH")
                    SysPar.testAlloc.dl = 1;
                    SysPar.testAlloc.ul = 0;
                    SysPar.testAlloc.pdsch = idxUE-1;
                    SysPar.testAlloc.pusch = 0;
                else
                    SysPar.testAlloc.dl = 0;
                    SysPar.testAlloc.ul = 1;
                    SysPar.testAlloc.pdsch = 0;
                    SysPar.testAlloc.pusch = idxUE-1;
                end                
                
                WriteYaml(cfgFile, SysPar);
                errFlag = runSim(cfgFile, tvFile);
                errCnt = errCnt + errFlag;
                
                % delete the seperate CSI-RS TV, if any. (perf pipeline used it as PDSCH TV)
                csirs_filename = strcat('./GPU_test_input/*', tvFile, '*CSIRS_gNB_CUPHY*');
                delete(csirs_filename)
            end
        end
    end
return
end
