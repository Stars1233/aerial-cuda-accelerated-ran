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

function pusch = cfgPusch(SimCtrl)

% pusch related config
pusch.pduBitmap = 1; % bit0: data, bit1: UCI, bit5: CSI-P2
pusch.BWPSize = 273;
pusch.BWPStart = 0;
pusch.RNTI = 46;
pusch.NrOfCodewords = 1; % not used
pusch.pi2BPSK = 0;
pusch.targetCodeRate = 0; % valid only when mcsIndex >= 28(29), FAPI format: raw coderate x 1024 x 10
pusch.qamModOrder = 0; % valid only when mcsIndex >= 28(29)
pusch.TBSize = 0;  % valid only when mcsIndex >= 28(29)
pusch.mcsIndex = 27;
pusch.mcsTable = 1;
pusch.rvIndex = 0;
pusch.newDataIndicator = 1;
pusch.harqProcessID = 0;
pusch.TransformPrecoding = 1; % 0: enabled, 1: disabled
pusch.dataScramblingId = 41;
pusch.nrOfLayers = 1;
pusch.portIdx = [0:pusch.nrOfLayers-1];
% pusch.DmrsSymbPos = [0 0 1 1 0 0 0 0 0 0 0 0 0 0];
pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 0 0 0];
pusch.DmrsMappingType = 0; % 0:typeA, 1:typeB (not defined in FAPI PDU, for compliance test only)
pusch.dmrsConfigType = 0; % 1: not supported
pusch.DmrsScramblingId = 41;
pusch.puschIdentity = 0;
pusch.SCID = 0;
pusch.numDmrsCdmGrpsNoData = 2; % not used
pusch.resourceAlloc = 1; % 0: not supported
% pusch.rbBitmap = ones(1, 273); % not used
pusch.rbStart = 120; %
pusch.rbSize = 100; %
pusch.VRBtoPRBMapping = 0;  % 1: not supported
pusch.FrequencyHopping = 0;
pusch.txDirectCurrentLocation = 0;
pusch.uplinkFrequencyShift7p5khz = 0;
pusch.StartSymbolIndex = 0;
pusch.NrOfSymbols = 8;
pusch.prcdBf = 0;
% for LBRM
pusch.I_LBRM = 0;
pusch.maxLayers = 4;
pusch.maxQm = 8;
pusch.n_PRB_LBRM = 273;
pusch.digBFInterfaces = 4;
pusch.beamIdx = [1 2 3 4];
pusch.seed = 0;
pusch.payload = [];
pusch.idxUE = 0;
pusch.idxUeg = 0;
%Values only used when UCIonPUSCH is enabled (w/wo UL-SCH)
pusch.harqAckBitLength = 0;     % 0:11 -> small block length, 12:1706: polar, does not included CRC
pusch.harqPayload = [];
pusch.csiPart1BitLength = 0;    % 0:11 -> small block length, 12:1706: polar, does not included CRC
pusch.csiPart1Payload = [];
pusch.csiPart2Payload = [];
pusch.alphaScaling = 3;         % values: 0,1,2,3 --> maps to 0.5, 0.65, 0.8 and 1
pusch.betaOffsetHarqAck = 0;    % Range 0:15
pusch.betaOffsetCsi1 = 0;       % Range 0:18
pusch.betaOffsetCsi2 = 0;       % Range 0:18
pusch.rank           = 1;
pusch.rankBitOffset  = 0;
pusch.rankBitSize    = 2;
pusch.DTXthreshold = 1;
pusch.groupOrSequenceHopping = 0;
% FAPIv3 support for multiple csiP2
pusch.flagCsiPart2 = 0;
pusch.part2SizeMapping = [5, 5, 4, 4];
% pusch.lowPaprGroupNumber = 0;
% pusch.lowPaprSequenceNumber = 0;
pusch.prgSize = 273;
pusch.enable_prg_chest = 0;
pusch.csiPart2BitLength = 255;

% 10.04 FAPI CSI2 constants:
MAX_NUM_CSI1_PRM     = SimCtrl.MAX_NUM_CSI1_PRM;
MAX_NUM_CSI2_REPORTS = SimCtrl.MAX_NUM_CSI2_REPORTS;

% 10.04 FAPI CSI2 default paramaters:
pusch.calcCsi2Size_csi2MapIdx = zeros(MAX_NUM_CSI2_REPORTS , 1);
pusch.calcCsi2Size_nPart1Prms = zeros(MAX_NUM_CSI2_REPORTS , 1);
pusch.calcCsi2Size_prmOffsets = zeros(MAX_NUM_CSI1_PRM * MAX_NUM_CSI2_REPORTS, 1);
pusch.calcCsi2Size_prmSizes   = zeros(MAX_NUM_CSI1_PRM * MAX_NUM_CSI2_REPORTS, 1);
pusch.calcCsi2Size_prmValues  = zeros(MAX_NUM_CSI1_PRM * MAX_NUM_CSI2_REPORTS, 1);
pusch.nCsi2Reports            = 0;
pusch.flagCsiPart2            = 0;
pusch.foForgetCoeff           = 0.0;
pusch.foCompensationBuffer    = 1.0;
pusch.ldpcEarlyTerminationPerUe = 0;
pusch.ldpcMaxNumItrPerUe      = 10;


return
