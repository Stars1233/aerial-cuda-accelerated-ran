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

function SimCtrl = cfgSimCtrl

% release number
SimCtrl.relNum = 10000; % 10000: latest, 2240: Rel-22-4

% controller for simulation setup
SimCtrl.N_UE = 1;
SimCtrl.N_frame = 1;
SimCtrl.N_slot_run = 1;  % number of slots to run. 0: run all slots

SimCtrl.timeDomainSim = 0; % run time domain simulation
SimCtrl.enableCS = 0; % run cell search
SimCtrl.prachFalseAlarmTest = 0; % PRACH false alarm test (mute prach signal)
SimCtrl.puschHARQ = struct;
SimCtrl.puschHARQ.EnableAutoHARQ = 0; % If not 0, automatically set newDataIndication and rvIndex
SimCtrl.puschHARQ.MaxTransmissions = 1; % Max PUSCH number of transmissions per TB for HARQ performance testing
SimCtrl.puschHARQ.MinTransmissions = 1; % Min transmissions - retransmission will occur even on successful decode if > 1
SimCtrl.puschHARQ.rvTable = [0, 2, 3, 1];
SimCtrl.puschHARQ.MAX_gNB_HARQ_BUFFERS = 32;

% control PUCCH payload
SimCtrl.force_pucch_payload_as = 'random'; % 'random': PUCCH payload will be randomly generated. 'zeros': all zeros. 'ones': all ones

% controller for display
SimCtrl.plotFigure.tfGrid = 0;  % plot time/freq domain signal
SimCtrl.plotFigure.constellation = 0; % plot constellation before/after equalizer

% print checkDetError
SimCtrl.printCheckDet = 0;

% srs chEst bank size:
SimCtrl.bf.N_SRS_CHEST_BUFF = 0;

% Flag to enable codebook beamforming for cuBB test
SimCtrl.enable_codebook_BF = 1;

% enable dynamic beamforming (e.g. 32T32R)
SimCtrl.enable_dynamic_BF = 0;

% controller for test vector generation
SimCtrl.genTV.enable = 0;
SimCtrl.genTV.enableUE = 0;
SimCtrl.genTV.tvDirName = 'GPU_test_input';
SimCtrl.genTV.genYamlCfg = 0;    % generate yaml config file for test cases
SimCtrl.genTV.cuPHY = 1; % generate cuPHY TV
SimCtrl.genTV.FAPI = 1; % generate FAPI TV
SimCtrl.genTV.launchPattern = 0; % generate launch pattern file
SimCtrl.genTV.fhMsgMode = 0; % 1: modComp SE4 only, 2: modComp SE4/5
SimCtrl.genTV.FAPIyaml = 0;
SimCtrl.genTV.slotIdx = [0]; % slot index to generate TV
SimCtrl.genTV.forceSlotIdxFlag = 0; % force slotNum = genTV.slotIdx(1) for every slot
SimCtrl.genTV.bypassComp = 0;%0;
SimCtrl.genTV.idx = 0;
SimCtrl.genTV.TVname = [];
SimCtrl.genTV.polDbg = 0;
SimCtrl.genTV.enable_logging_tx_Xtf = 0;
SimCtrl.genTV.enable_logging_carrier_and_channel_info = 0;
SimCtrl.genTV.enable_logging_dbg_pusch_chest = 1;
SimCtrl.genTV.add_LS_chEst = 0;

% for captured sample analysis and additional BFP samples in FAPI TV
SimCtrl.oranComp.iqWidth = [14 9];
SimCtrl.oranComp.Ref_c = [0 0];
SimCtrl.oranComp.FSOffset = [0 0];
SimCtrl.oranComp.Nre_max = 273*12;
SimCtrl.oranComp.max_amp_ul = 65504; % maximum representable value for FP16

% for ORAN fixed point
SimCtrl.oranFixedPoint.iqWidth_UL = 16; % bitwidth for oran fixed point, 0: disable fixed point and use BFP
SimCtrl.oranFixedPoint.beta_UL = 2^11; % scaling factor from floating point to fixed point
SimCtrl.oranFixedPoint.iqWidth_DL = 16; % bitwidth for oran fixed point, 0: disable fixed point and use BFP
SimCtrl.oranFixedPoint.beta_DL = 2^11; % scaling factor from floating point to fixed point

% BFP/FixedPoint setting for cuPHY UL TV generation and UL performance simulation
SimCtrl.BFPforCuphy = 14; % 16, 14 or 9 for FP16, BFP14 or BFP9
SimCtrl.FixedPointforCuphy = 0; % 0: disable; 16: fixed point 16, 

% algorithm related flags
SimCtrl.alg.enableRssiMeas = 1;% flag indicating if RSSI needs to be measured
SimCtrl.alg.enableSinrMeas = 1; % flag indicating if SINR need to be measured
SimCtrl.alg.enableIrc = 1; % flag indicating if MMSE-IRC is used instead of MMSE
SimCtrl.alg.Rww_regularizer_val = 1e-8; % diagonal loading factor to the noise covariance matrix
SimCtrl.alg.enable_nCov_shrinkage = 1; % 
SimCtrl.alg.nCov_shrinkage_method = 0; % 'RBLW', 'OAS'
SimCtrl.alg.pusch_sinr_selector = 1; % flag indicating SINR to be reported in cuPHY-CP. 0: no report; 1: postEqSinr; 2: preEqSinr
SimCtrl.alg.LDPC_maxItr = 10; % max number of iterations of PUSCH LDPC decoder
SimCtrl.alg.LDPC_enableEarlyTerm = 0; % 1: enable early termination in LDPC
SimCtrl.alg.LDPC_earlyTermAlg = 'SAFEET'; % early termination alg, PCE: parity check equation (i.e, syndrome check), CRC: CRC check of info bits, NBF: no bit flipping, CPCE: only check min(4,nZ) * Zc check nodes, SAFEET: CPCE + tolerate at most badItrThres badItrs
SimCtrl.alg.LDPC_earlyTerm_NBF_num_consecutive_itr = 2; % num of consecutive iterations that have no sign flipping
SimCtrl.alg.LDPC_earlyTerm_SAFEET_badItrThres = 2; % max num of bad iters
SimCtrl.alg.LDPC_flags = 2; % CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT
SimCtrl.alg.LDPC_DMI_method = 'LUT_spef'; % fixed: fix the num of LDPC iterations, ML: use machine learning model to predict, LUT_spef: use LUT for max num LDPC itrs, per_UE: use per-UE max num LDPC itrs provided by L2. 
SimCtrl.alg.LDPC_DMI_ML_model_path = ''; % path of the saved ML model file trained from AMLAsim
SimCtrl.alg.LDPC_DMI_ML_confidence_thres = 0; % if model softmax out >= this threshold, apply the prediction, otherwise still use the default maxNumItr
SimCtrl.alg.LDPC_DMI_LUT_SNRmargin_dB = 0; % the SNRmargin to be added to the effective SNR for LUT base LDPC maxItr selection
SimCtrl.alg.LDPC_DMI_LUT_numItr_vec = [10, 7]; % the lookup value. when postEqSINR < threshold, use the first value for LDPC maxItr; otherwise use the second one
SimCtrl.alg.LDPC_use_5Gtoolbox_flag = 0; % option to use MATHWORKS 5G toolbox LDPC decoder
SimCtrl.alg.tdi1_alg = 'linInterp';
SimCtrl.alg.ChEst_use_vPRBs = 0;
SimCtrl.alg.enableEarlyHarq = 1;
SimCtrl.alg.lastEarlyHarqSymZeroIndexed = 3;
SimCtrl.alg.dft_s_ofdm_enable_bluestein_fft = 1;
SimCtrl.alg.dft_s_ofdm_bluestein_fft_size_lut_row1 = [5, 10, 21, 42, 85, 170, 270];
SimCtrl.alg.dft_s_ofdm_bluestein_fft_size_lut_row2 = [128, 256, 512, 1024, 2048, 4096, 8192];
SimCtrl.alg.ChEst_alg_selector = 1; % flag indicating PUSCH channel estimation algorithm to be used in 5GModel and cuPHY-CP. 0: MMSE; 1: MMSE_w_delay_and_spread_est . 2: RKHS ChEst
SimCtrl.alg.ChEst_MMSE_w_delay_and_spread_est_opt = 2; 
SimCtrl.alg.ChEst_enable_update_W = 0;
SimCtrl.alg.ChEst_enable_quantize_delay_spread_est = 1;
SimCtrl.alg.ChEst_quantize_levels_delay_spread_est = [0.0722    0.1443    0.2887    0.5774]*1e-6;
SimCtrl.alg.pusch_equalizer_bias_corr_limit = 10; % limit the bias correction factor in PUSCH equalization
SimCtrl.alg.srs_chEst_alg_selector = 0; % flag indicating SRS channel estimation algorithm to be used in 5GModel and cuPHY-CP. 0: MMSE; 1: RKHS
SimCtrl.alg.srs_chEst_toL2_normalization_algo_selector = 1; % Normalization/scaling algorithm selection: 0: constant scaling; 1: peak value normalization
SimCtrl.alg.srs_chEst_toL2_constant_scaler = 32768; % Constant scaling factor which is valid when srs_chEst_toL2_normalization_algo_selector = 0
% flag indicating if estimated noise is to be used for ZF equalizer
% not supported by cuPHY
SimCtrl.alg.enableNoiseEstForZf = 1; 

SimCtrl.alg.fakeSNRdBForZf = 36;

SimCtrl.delaySpread = 1e-6;

% findLic = license('checkout', 'fixed_point_toolbox');
% if(findLic)
%     SimCtrl.fp16AlgoSel = 0; % use fixed point toolbox
% else
%     SimCtrl.fp16AlgoSel = 1; % use Cleve toolbox
% end
% if (~isdeployed)
%     SimCtrl.fp16AlgoSel = 0; % 0: use fixed point toolbox; 1: use Cleve toolbox; 2: use Nvidia fp16
% else
%     SimCtrl.fp16AlgoSel = 1; % Matlab Compiler bug workaround
% end
SimCtrl.fp16AlgoSel = 2; % 0: use fixed point toolbox; 1: use Cleve toolbox; 2: use Nvidia fp16

SimCtrl.fp_flag_global = 0; % set the default FP format used in signal processing. 0: FP64, 1: FP32, 2: TF32, 3: FP16, 4: BF16, 5: FP8 E4M3, 6: FP8 E5M2
% Note that the FP flag set in each module below will force the corresponding module ignore the above global flag
SimCtrl.fp_flag_pusch_chest = 0; % set the FP format used in PUSCH ChEst.
SimCtrl.fp_flag_pusch_equalizer = 0; % set the FP format used in PUSCH equalizer.
SimCtrl.fp_flag_pusch_demapper = 0; % set the FP format used in PUSCH soft demapper.
SimCtrl.fp_flag_pusch_demapper_out_llr = 0; % set the FP format used in PUSCH soft demapper.
SimCtrl.fp_flag_pusch_ldpc_decoder = 0; % set the FP format used in PUSCH LDPC decoder.


SimCtrl.forceRxZero = 0; % force RX samples to 0
SimCtrl.forceRxVal = 0; % 0: disable 1: maximum value 2: mininum value 3: a combination of maximum value and minimum value 4: large fluctuation
SimCtrl.useCuphySoftDemapper = 2; % 1: use cuPHY soft demapper model based on LUT 2: use new cuPHY soft demapper

SimCtrl.alg.enableCfoEstimation = 1; % enable CFO estimation
SimCtrl.alg.enableCfoCorrection = 1; % enable CFO correction
SimCtrl.alg.enableWeightedAverageCfo = 0; % disable weight average CFO
SimCtrl.alg.enableToEstimation = 1; % enable Timing Offset estimation
SimCtrl.alg.enableToCorrection = 0; % enable Timing Offset correction 
SimCtrl.alg.enableDftSOfdm = 0;    % disable DFT-s-OFDM for PUSCH
SimCtrl.alg.TdiMode = 1;    % enable CEE time domain interpolation for PUSCH
SimCtrl.alg.TdiModePf3 = 1;  % enable CEE time domain interpolation for PUCCH-3
SimCtrl.alg.dtxModePf2 = 1;    % enable PUCCH-2 DTX detection
SimCtrl.alg.dtxModePf3 = 1;    % enable PUCCH-3 DTX detection
SimCtrl.alg.dtxModeUciOnPusch = 1;    % enable UCI on PUSCH DTX detection
SimCtrl.alg.useNrUCIDecode = 0; % use Matlab 5G Toolbox nrUCIDecode function
SimCtrl.alg.listLength = 8; % list length for Polar decoder
SimCtrl.alg.enablePrcdBf = 0; % flag to enable Precoding
SimCtrl.alg.enable_get_genie_meas = 0; % flag to enable getting Genie channel matrix, Genie RSRP, Genie preSINR and postSINR
SimCtrl.alg.enable_use_genie_nCov = 0;
SimCtrl.alg.genie_nCov_method = 'genie_interfNoise_based'; % 'genie_interfNoise_based', 'genie_interfChanResponse_based'
SimCtrl.alg.enable_sphere_decoder = 0; % enable sphere decoder for PUSCH MIMO equalizer
SimCtrl.alg.enable_use_genie_channel_for_equalizer = 0; % 1: use genie channel for MIMO equalization
SimCtrl.alg.enable_avg_nCov_prbs_fd = 0; % 1: enable averaging nCov of PRBs in freq domain to improve nCov est reliability. This is useful especially in sub-slot proc
SimCtrl.alg.win_size_avg_nCov_prbs_fd = 3; % the num. of PRBs used to average nCov. For example, 7 means using 6 neighbors (3 at each side). Should be an odd number.
SimCtrl.alg.threshold_avg_nCov_prbs_fd_dB = 3; % only PRBs that have less or equal to this threshold can be used for averaging nCov
SimCtrl.alg.enable_instant_equ_coef_cfo_corr = 1;  % enable CFO correction on equalization coef immediately after each DMRS position
SimCtrl.checkSrsHestErr = 0; % enable check SRS channel estimation error
SimCtrl.forceCsiPart2Length = 0; % 0: not force; > 0, force to the value of forceCsiPart2Length

% control of capture samples analysis
SimCtrl.capSamp.enable = 0;
SimCtrl.capSamp.fileName = [];
SimCtrl.capSamp.isComp = 0;

% O-RU dBm value correspoding to 0 dB power at baseband
% Need to get this value through O-RU spec/calibration
SimCtrl.ul_gain_calibration = 48.68;

% enable UE receiver
SimCtrl.enableUeRx = 0;

% enable beamforming
SimCtrl.enableDlTxBf = 0;
SimCtrl.enableUlRxBf = 0;
SimCtrl.enableSrsState = 0;
SimCtrl.BfKnownChannel = 0;
SimCtrl.nSlotSrsUpdate = 1;

% record test results
SimCtrl.results = [];

% set interfering UE in UL
SimCtrl.N_Interfering_UE_UL = 0;

% Batchsim
SimCtrl.batchsim.save_results = 0;
SimCtrl.batchsim.save_results_short = 0;  % only save SysPar.Chan and SysPar.SimCtrl.results
SimCtrl.batchsim.save_results_LDPC = 0;
SimCtrl.batchsim.save_results_PUSCH_ChEst = 0;

% set seed used in runSim.m
SimCtrl.seed = 0;

% set to get genie channel matrix
SimCtrl.enable_get_genie_channel_matrix = 0; % 1: enable getting genie channel matrix when using 5G toolbox for channel generation, which can be used to get genie meas

% set freezing Tx and channel
SimCtrl.enable_freeze_tx_and_channel = 0; % 1: freeze Tx signal and channel fading to the first slot. This mode is used for PerformanceMatchTest_5GModel_cuPHY
SimCtrl.enable_freeze_tx = 0; % 1: freeze Tx signal

% machine learning related config
SimCtrl.ml.enable = 0;
SimCtrl.ml.python_executable = '/home/Aerial/python_env/anaconda3/envs/ENV5G_v3/bin/python';
SimCtrl.ml.python_ExecutionMode = 'OutOfProcess';
SimCtrl.ml.dataset.enable_save_dataset = 0;
SimCtrl.ml.dataset.channel = 'pusch';

SimCtrl.oranCompressBetaForce = 0; % 1: if enabled, force BFP9: beta = 65536 ( = 2^16)
                                                        % BFP14: beta = 2097152 ( = 2^21)
SimCtrl.enable_print_idxFrame = 0; % 1: print out idxFrame

SimCtrl.enable_snapshot_gNB_UE_into_SimCtrl = 0;

% BFP for BFW coefficient compression
SimCtrl.bfw.compressBitWidth = 9; % 0: no compression, 9: BFP-9
SimCtrl.bfw.beta = 2^14; % scaling factor to convert BFW coef from FP16 to integer

% BFW settings
SimCtrl.bfw.enable_prg_chest = 0; % 1: enable PUSCH per-PRG channel estimation, 0: disable

% sub-slot processing
SimCtrl.subslot_proc_option = 0; 
% 0: disable sub-slot processing and use the legacy full-slot processing; 
% 1: calc. the first and second DMRS sym equ. coef separately
% 2: calc. the first DMRS sym. whiting matrix and the equ. coef, then calc. the the second DMRS sym. whiting matrix, avg. w/ the first one, and calc. the second DMRS sym. equ. coef
% 3: calc. the first DMRS sym. shrinked nCov, whiting matrix and the equ. coef, then calc. the second DMRS sym. shrinked nCov, avg. w/ the first one, and calc. the whitening matrix and equ. coef 

% normalize Tx power of PUSCH layers
SimCtrl.normalize_pusch_tx_power_over_layers = 0;

% invalid params in TV generation
SimCtrl.negTV.enable = 0; % by default negTV is disabled
SimCtrl.negTV.channel = {}; % channel types for negative TV (e.g., {'PUSCH'}, {'PRACH'})
% 0: no invalid params in TV
% 1: has invalid params in TV, as defined in SimCrtl.negTV.pdufieldName and SimCrtl.negTV.pdufieldValue

SimCtrl.enable_UL_Rx_RF_impairments = 0; % 1: enable RF impairments Rx
SimCtrl.enable_DL_Tx_RF_impairments = 0;

SimCtrl.enable_EVM_calculation = 0; % 1: enable collect DL Tx EVM stats

SimCtrl.UL_Rx_power_scaling_dB = 0; % 

SimCtrl.enable_normalize_Xtf_perUeg = 0; % 1: enable normalize the rx signal per UE group

% FAPIv3 support for multiple csiP2
SimCtrl.enable_multi_csiP2_fapiv3 = 0;
SimCtrl.nCsi2Maps                 = 4;
SimCtrl.MAX_NUM_CSI1_PRM          = 4;
SimCtrl.MAX_NUM_CSI2_MAPS         = 4;
SimCtrl.MAX_NUM_CSI2_REPORTS      = 16;

% initalize map paramaters:
SimCtrl.csi2Maps_nPart1Prms       = zeros(SimCtrl.MAX_NUM_CSI2_MAPS, 1);
SimCtrl.csi2Maps_part1PrmSizes    = zeros(SimCtrl.MAX_NUM_CSI1_PRM * SimCtrl.MAX_NUM_CSI2_MAPS, 1);
SimCtrl.csi2Maps_sumOfPrmSizes    = zeros(SimCtrl.MAX_NUM_CSI2_MAPS, 1);
SimCtrl.csi2Maps_bufferStartIdxs  = zeros(SimCtrl.MAX_NUM_CSI2_MAPS, 1);
SimCtrl.csi2Maps_buffer           = [];

% first csi2 map paramaters:
SimCtrl.csi2Maps_nPart1Prms(1)      = 1;
SimCtrl.csi2Maps_part1PrmSizes(1)   = 2;
SimCtrl.csi2Maps_sumOfPrmSizes(1)   = 2;
SimCtrl.csi2Maps_bufferStartIdxs(1) = 0;
SimCtrl.csi2Maps_buffer             = [SimCtrl.csi2Maps_buffer [5 5 4 4]];

% second csi2 map paramaters:
SimCtrl.csi2Maps_nPart1Prms(2)      = 2;
SimCtrl.csi2Maps_part1PrmSizes(1 + SimCtrl.MAX_NUM_CSI1_PRM) = 1;
SimCtrl.csi2Maps_part1PrmSizes(2 + SimCtrl.MAX_NUM_CSI1_PRM) = 2;
SimCtrl.csi2Maps_sumOfPrmSizes(2)   = 3;
SimCtrl.csi2Maps_bufferStartIdxs(2) = 4;
SimCtrl.csi2Maps_buffer             = [SimCtrl.csi2Maps_buffer [4 8 10 5 2 11 20 8]];

% third csi2 map paramaters:
SimCtrl.csi2Maps_nPart1Prms(3)      = 1;
SimCtrl.csi2Maps_part1PrmSizes(1 + 2*SimCtrl.MAX_NUM_CSI1_PRM) = 2;
SimCtrl.csi2Maps_sumOfPrmSizes(3)   = 2;
SimCtrl.csi2Maps_bufferStartIdxs(3) = 12;
SimCtrl.csi2Maps_buffer             = [SimCtrl.csi2Maps_buffer [8 8 3 3]];

% fourth csi2 map paramaters:
SimCtrl.csi2Maps_nPart1Prms(4)      = 1;
SimCtrl.csi2Maps_part1PrmSizes(1 + 3*SimCtrl.MAX_NUM_CSI1_PRM) = 2;
SimCtrl.csi2Maps_sumOfPrmSizes(4)   = 2;
SimCtrl.csi2Maps_bufferStartIdxs(4) = 16;
SimCtrl.csi2Maps_buffer             = [SimCtrl.csi2Maps_buffer [5 7 8 11]];

SimCtrl.csi2Maps_part1PrmSizes = SimCtrl.csi2Maps_part1PrmSizes(:);

% Enable static and dynamic beamforming
SimCtrl.enable_static_dynamic_beamforming = 0;
SimCtrl.num_static_beamIdx = 1024;
SimCtrl.num_TRX_beamforming = 64;

SimCtrl.alg.bfwPowerNormAlg_selector = 1; % 0 : Frobenius normalizer. 1 : Per-UE two-stage normalizer.

% when number of CSIRS ports (nPort) is greater than nPort_enable_csirs_compression, 
% enable CSIRS compression by storing CSIRS IQ samples to CSIRS nCDM (1/2/4/8)
% layers instead of nPort (1/2/4/8/12/16/24/32) layers on Xtf Tensor. 
SimCtrl.nPort_enable_csirs_compression = 33; % default 33 to disable CSIRS compression

SimCtrl.UL_Rx_power_scaling_dB = 0; 

SimCtrl.enable_normalize_Xtf_perUeg = 0; % 1: enable normalize the rx signal per UE group
% Channel processing timeline info
SimCtrl.timeline.pusch_seg0_Tchan_start_offset = 1000;
SimCtrl.timeline.pusch_seg0_Tchan_duration     = 4500;
SimCtrl.timeline.pusch_seg1_Tchan_start_offset = 1000;
SimCtrl.timeline.pusch_seg1_Tchan_duration     = 2000;

SimCtrl.usePuschRxForPdsch = 0;
SimCtrl.enableUlDlCoSim = 0;

return
