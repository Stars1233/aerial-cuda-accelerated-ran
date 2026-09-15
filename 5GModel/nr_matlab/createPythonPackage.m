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

clear all;
close all;
startup();
generatedDirs = {'bin','CompilerSDKOutput'};
for k = 1:length(generatedDirs)
    if isfolder(generatedDirs{k})
        rmpath(genpath(generatedDirs{k}));
    end
end

functionPattern = {
    dir('genCfgTemplate.m'),
    dir('gNBreceiver.m'),
    dir('gNBtransmitter.m'),
    dir('nrSimulator.m'),
    dir('runRegression.m'),
    dir('runSim.m'),
    dir('UEreceiver.m'),
    dir('UEtransmitter.m'),

    dir('pxsch/gen_WFreq_h5.m'),
    dir('pxsch/apply_ChEst_main.m'),
    dir('pxsch/derive_upper_filter.m'),
    dir('pxsch/derive_middle_filter.m'),
    dir('pxsch/derive_lower_filter.m'),
    dir('pxsch/derive_small_filter.m'),
    dir('pxsch/gen_dmrs_sym.m'),

    dir('channel/addNoise.m'),
    dir('channel/Channel.m'),

    dir('config/cfg*.m'),
    dir('config/initChan.m'),
    dir('config/initgNB.m'),
    dir('config/initSysPar.m'),
    dir('config/initUE.m'),
    dir('config/setSimCtrl.m'),
    dir('config/updateAlloc.m'),
    dir('config/updateCarrier.m'),

    dir('shared/genCuPhyChEstCoeffs.m'),

    dir('test/testCompGenTV_*.m'),
    dir('test/genLP_POC2.m'),
    dir('test/genPerfPattern.m'),
    dir('test/testPerformance*.m'),
    dir('test/genCfgTV_perf_ss.m'),
    dir('test/genCfgTV_perf_ss_vf.m'),
    dir('test/genCfgTV_cuphycp.m')
};

display('Creating NRSimulator package with the following 5GModel interfaces:');
m = 0;
for k = 1:length(functionPattern)
    for l = 1:length(functionPattern{k})
        m = m + 1;
        functionFiles{m} = [functionPattern{k}(l).folder,'/',functionPattern{k}(l).name];
        display(['   ',functionFiles{m}]);
    end
end
display(' ');

additionalPattern = {
    dir('test/*.txt'),
    dir('test/*.xlsm'),
    dir('test/*.xlsx'),
    dir('test/*.yaml'),
    dir('test/*.json'),
    dir('test/perf_pattern/*.yaml')
};

display('Including additional data files in package:');
m = 0;
for k = 1:length(additionalPattern)
    for l = 1:length(additionalPattern{k})
        m = m + 1;
        additionalFiles{m} = [additionalPattern{k}(l).folder,'/',additionalPattern{k}(l).name];
        display(['   ',additionalFiles{m}]);
    end
end

buildResults = compiler.build.pythonPackage(functionFiles,...
                   'AdditionalFiles',additionalFiles,...
                   'PackageName','aerial_mcore',...
                   'OutputDir','CompilerSDKOutput/')
