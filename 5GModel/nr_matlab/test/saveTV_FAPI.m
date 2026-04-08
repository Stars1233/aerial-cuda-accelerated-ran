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

function saveTV_FAPI(genTV, FAPIpdu, Config, X_tf_uncomp, X_tf, X_tf_prach_uncomp, X_tf_prach, X_tf_srs_uncomp, X_tf_srs, Xtf_remap, ...
    Xtf_remap_trsnzp, srsChEstDatabase, fhMsg, fhMsg_new, idxSlotInFrame, node, table, sim_is_uplink)

global SimCtrl

carrier = Config.carrier;

tvDirName = genTV.tvDirName;
[status,msg] = mkdir(tvDirName);

if ~isempty(FAPIpdu)
    if contains(genTV.TVname, 'TV_cuphy_')
        TVname = strrep(genTV.TVname,'_cuphy_', '_fapi_');
    else
        TVname = [genTV.TVname, '_', node, '_FAPI_s', num2str(idxSlotInFrame)];
    end
    h5File  = H5F.create([tvDirName filesep TVname '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

    if genTV.FAPIyaml
        yamlFileName = [tvDirName filesep TVname '.yaml'];
        WriteYaml(yamlFileName, FAPIpdu);
    end

    hdf5_write_nv_exp(h5File, 'SFN', uint32(carrier.idxFrame));
    hdf5_write_nv_exp(h5File, 'BFPforCuphy', uint32(SimCtrl.BFPforCuphy));
    hdf5_write_nv_exp(h5File, 'FixedPointforCuphy', uint32(SimCtrl.FixedPointforCuphy));
    hdf5_write_nv_exp(h5File, 'fhMsgMode', uint32(SimCtrl.genTV.fhMsgMode));

    Cell_Config = setCellConfig(carrier, SimCtrl);
    hdf5_write_nv_exp(h5File, 'Cell_Config', Cell_Config);

    savePrachConfig(h5File, Config.prach, carrier, SimCtrl.negTV);

    savePrecodingMatrix(h5File, table);

    saveDigitalBeamformingTable(h5File, SimCtrl);

    saveAlgConfig(h5File, SimCtrl);
    saveCsi2MapPrms(h5File, SimCtrl);

    nPdu = length(FAPIpdu);
    hdf5_write_nv_exp(h5File, 'nPdu', uint32(nPdu));
    idxIndication = 1;
    timeline_pusch_seg0_save_once = true;
    timeline_pusch_seg1_save_once = true;
    nTimelines = 0;
    for idxPdu = 1:nPdu
        pdu = FAPIpdu{idxPdu};
        idxIndication = savePdu(h5File, pdu, idxPdu, idxIndication, carrier);
        [timeline_pusch_seg0_save_once, timeline_pusch_seg1_save_once, nTimelines] = saveTimeline(h5File, pdu, SimCtrl, timeline_pusch_seg0_save_once, timeline_pusch_seg1_save_once, nTimelines);
    end
    if nTimelines ~= 0
        hdf5_write_nv_exp(h5File, 'nTimelines', uint32(nTimelines));
    end
    hdf5_write_nv_exp(h5File, 'X_tf', complex(single(X_tf)));
    if ~isempty(X_tf_prach)
        nPrach = length(X_tf_prach);
        hdf5_write_nv_exp(h5File, 'nPrach', uint32(nPrach));
        for idxPrach = 1:nPrach
            prachName = ['X_tf_prach_', num2str(idxPrach)];
            hdf5_write_nv_exp(h5File, prachName, complex(single(X_tf_prach{idxPrach})));
        end
    end
    if ~isempty(X_tf_srs)
        hdf5_write_nv_exp(h5File, 'X_tf_srs', complex(single(X_tf_srs)));
        [X_tf_srs_cSamples_uint8] = oranCompress(X_tf_srs_uncomp, sim_is_uplink); % generate ORAN compressed samples
        X_tf_srs_fp16 = reshape(fp16nv(real(X_tf_srs), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(X_tf_srs), SimCtrl.fp16AlgoSel), [size(X_tf_srs)]);
        hdf5_write_nv_exp(h5File, 'X_tf_srs_fp16', complex(X_tf_srs_fp16), 'fp16');
        for k=1:length(SimCtrl.oranComp.iqWidth)
            hdf5_write_nv_exp(h5File, ['X_tf_srs_cSamples_bfp',num2str(SimCtrl.oranComp.iqWidth(k))], uint8(X_tf_srs_cSamples_uint8{k}));
        end
        % Fixed Point
        if sim_is_uplink
            iqWidth = SimCtrl.oranFixedPoint.iqWidth_UL;
            beta = SimCtrl.oranFixedPoint.beta_UL;
        else
            iqWidth = SimCtrl.oranFixedPoint.iqWidth_DL;
            beta = SimCtrl.oranFixedPoint.beta_DL;
        end
        X_tf_srs_fx = oranFL2FX(X_tf_srs, iqWidth, beta); % generate ORAN fixed point samples
        hdf5_write_nv_exp(h5File, 'X_tf_srs_fx', uint8(X_tf_srs_fx));
    end    
    if ~isempty(Xtf_remap)
        hdf5_write_nv_exp(h5File, 'Xtf_remap', uint32(Xtf_remap));
    end
    if ~isempty(Xtf_remap_trsnzp)
        hdf5_write_nv_exp(h5File, 'Xtf_remap_trsnzp', uint32(Xtf_remap_trsnzp));
    end
    
    if ~isempty(srsChEstDatabase)
        srsChEstBuff = srsChEstDatabase.srsChEstBuff;
        startPrbGrps = srsChEstDatabase.startPrbGrps;
        nUesInBuff   = length(srsChEstBuff);
        for ueIdx = 0 : (nUesInBuff - 1)
            temp         = srsChEstBuff{ueIdx + 1};
            Hest         = reshape(fp16nv(real(temp), 2) + 1i*fp16nv(imag(temp), 2), [size(temp)]);
            srsChEstName = ['srsChEst', num2str(ueIdx)];
            hdf5_write_nv_exp(h5File, srsChEstName, complex(single(Hest)));
            srsChEstName = ['srsChEstHalf', num2str(ueIdx)];
            hdf5_write_nv_exp(h5File, srsChEstName, complex(single(Hest)),'fp16');
            
            srsChEstInfo               = [];
            srsChEstInfo.nGnbAnt       = size(Hest,2);
            srsChEstInfo.nPrbGrps      = size(Hest,1);
            srsChEstInfo.nUeAnt        = size(Hest,3);
            srsChEstInfo.startPrbGrp   = startPrbGrps(ueIdx + 1);
            srsChEstInfo.srsPrbGrpSize = srsChEstDatabase.prbGrpSize;
            srsChEstInfo               = formatU32Struct(srsChEstInfo);
            
            srsChEstInfoName = ['srsChEstInfo', num2str(ueIdx)];
            hdf5_write_nv_exp(h5File, srsChEstInfoName, srsChEstInfo);
        end  
    end
    if ~isempty(Config.bfw)
        if ~isempty(Config.bfw.bfwBuf)
            nUeGrp = length(Config.bfw.bfwBuf);
            for ueGrpIdx = 0 : (nUeGrp - 1)
                bfwName = ['bfwUeGrp', num2str(ueGrpIdx)];
                hdf5_write_nv_exp(h5File, bfwName, complex(single(Config.bfw.bfwBuf{ueGrpIdx + 1})));
                bfwCompName = ['bfwCompUeGrp', num2str(ueGrpIdx)];
                hdf5_write_nv_exp(h5File, bfwCompName, uint8(Config.bfw.bfwCompBuf{ueGrpIdx + 1}));
            end
        end
    end

    isNew = 0;
    nMsg = length(fhMsg);
    hdf5_write_nv_exp(h5File, 'nMsg', uint32(nMsg));
    for idxMsg = 1:nMsg
        msg = fhMsg{idxMsg};
        saveMsg(h5File, msg, idxMsg, isNew);
    end

    isNew = 1; % New FH messages supporting both SE 4 and 5
    nMsg_new = length(fhMsg_new);
    hdf5_write_nv_exp(h5File, 'nMsg_new', uint32(nMsg_new));
    for idxMsg = 1:nMsg_new
        msg_new = fhMsg_new{idxMsg};
        saveMsg(h5File, msg_new, idxMsg, isNew);
    end
    
    bypassComp = genTV.bypassComp;
    if ~bypassComp
        % BFP compression
        [X_tf_cSamples_uint8] = oranCompress(X_tf_uncomp, sim_is_uplink); % generate ORAN compressed samples
        X_tf_fp16 = reshape(fp16nv(real(X_tf), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(X_tf), SimCtrl.fp16AlgoSel), [size(X_tf)]);
        hdf5_write_nv_exp(h5File, 'X_tf_fp16', complex(X_tf_fp16), 'fp16');
        for k=1:length(SimCtrl.oranComp.iqWidth)
            hdf5_write_nv_exp(h5File, ['X_tf_cSamples_bfp',num2str(SimCtrl.oranComp.iqWidth(k))], uint8(X_tf_cSamples_uint8{k}));
        end
        nPrach = length(X_tf_prach);
        for idxPrach = 1:nPrach
            [X_tf_prach_cSamples_uint8] = oranCompress(X_tf_prach_uncomp{idxPrach}, 1); % generate ORAN compressed samples
            X_tf_prach_fp16 = reshape(fp16nv(real(X_tf_prach{idxPrach}), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(X_tf_prach{idxPrach}), SimCtrl.fp16AlgoSel), [size(X_tf_prach{idxPrach})]);
            prachName = ['X_tf_prach_', num2str(idxPrach), '_fp16'];
            hdf5_write_nv_exp(h5File, prachName, complex(X_tf_prach_fp16), 'fp16');
            for k=1:length(SimCtrl.oranComp.iqWidth)
                prachName = ['X_tf_prach_', num2str(idxPrach), '_cSamples_bfp'];
                hdf5_write_nv_exp(h5File, [prachName,num2str(SimCtrl.oranComp.iqWidth(k))], uint8(X_tf_prach_cSamples_uint8{k}));
            end
        end
        % Fixed Point
        if sim_is_uplink
            iqWidth = SimCtrl.oranFixedPoint.iqWidth_UL;
            beta = SimCtrl.oranFixedPoint.beta_UL;
        else
            iqWidth = SimCtrl.oranFixedPoint.iqWidth_DL;
            beta = SimCtrl.oranFixedPoint.beta_DL;
        end
        X_tf_fx = oranFL2FX(X_tf, iqWidth, beta); % generate ORAN fixed point samples
        hdf5_write_nv_exp(h5File, 'X_tf_fx', uint8(X_tf_fx));
        nPrach = length(X_tf_prach);
        for idxPrach = 1:nPrach
            X_tf_prach_fx = oranFL2FX(X_tf_prach{idxPrach}, iqWidth, beta); % generate ORAN fixed point samples
            prachName = ['X_tf_prach_', num2str(idxPrach), '_fx'];
            hdf5_write_nv_exp(h5File, prachName, uint8(X_tf_prach_fx));
        end        
    end

    H5F.close(h5File);
end

return

function idxIndication = savePdu(h5File, pdu, idxPdu, idxIndication, carrier)

global SimCtrl
ul_gain_calibration = SimCtrl.ul_gain_calibration;

pduName = ['PDU', num2str(idxPdu)];
switch pdu.type
    case 'ssb'
        pdu.type = 1;
        PM_W = pdu.PM_W;
        pdu = rmfield(pdu, 'PM_W');
        hdf5_write_nv_exp(h5File, [pduName, '_PM_W'], complex(single(PM_W)));
        
        % Overwrite the invalid params in SimCtrl.negTV SSB TV
        if(SimCtrl.negTV.enable == 1)
            pduFieldName = SimCtrl.negTV.pduFieldName;
            pduFieldValue = SimCtrl.negTV.pduFieldValue;
            
            if length(pduFieldName) ~= length(pduFieldValue)
                error('negTV configuration error: pduFieldName and pduFieldValue must have the same length');
            end

            for pduFieldIdx = 1:length(pduFieldName) % extact the params and overwrite pdu
                pdu.(pduFieldName{pduFieldIdx}) = pduFieldValue{pduFieldIdx};
            end
        end
        pdu = formatU32Struct(pdu);
        hdf5_write_nv_exp(h5File, pduName, pdu);
    case 'pdcch'
        pdu.type = 2;
        DCI = pdu.DCI;
        pdu = rmfield(pdu, 'DCI');
        pdu = formatU32Struct(pdu);
        hdf5_write_nv_exp(h5File, pduName, pdu);
        nDCI = length(DCI);
        for n = 1:nDCI
            thisDCI = DCI{n};
            PM_W = thisDCI.PM_W;
            thisDCI = rmfield(thisDCI, 'PM_W');
            Payload = thisDCI.Payload;
            thisDCI = rmfield(thisDCI, 'Payload');
            powerControlOffsetSSProfileNR = thisDCI.powerControlOffsetSSProfileNR;
            thisDCI = rmfield(thisDCI, 'powerControlOffsetSSProfileNR');
            thisDCI = formatU32Struct(thisDCI);
            thisDCI.powerControlOffsetSSProfileNR = int32(powerControlOffsetSSProfileNR);
            DCIname = [pduName,'_DCI', num2str(n)];
            hdf5_write_nv_exp(h5File, DCIname, thisDCI);
            Payloadname = [pduName,'_DCI', num2str(n), '_Payload'];
            hdf5_write_nv_exp(h5File, Payloadname, uint8_convert(Payload, 0));
            Payloadname = [pduName,'_DCI', num2str(n), '_PM_W'];
            hdf5_write_nv_exp(h5File, Payloadname, complex(single(PM_W)));
        end
    case 'pdsch'
        pdu.type = 3;
        pdu.targetCodeRate = pdu.targetCodeRate;
        pdu.DmrsSymbPos = bin2dec(num2str(fliplr(pdu.DmrsSymbPos)));
        pdu.dmrsPorts = bin2dec(num2str(pdu.dmrsPorts));
        nlAbove16BitLoc = 15; % use the same value defined as PDSCH_ABOVE_16_LAYERS_DMRSPORTS_BIT_LOC in cuphy.h
        pdu.dmrsPorts = pdu.dmrsPorts + bitsll(pdu.nlAbove16, nlAbove16BitLoc); 
        payload = pdu.payload;
        pdu = rmfield(pdu, 'payload');
        PM_W = pdu.PM_W;
        pdu = rmfield(pdu, 'PM_W');
        hdf5_write_nv_exp(h5File, [pduName, '_PM_W'], complex(single(PM_W)));

        % Overwrite the invalid params in SimCtrl.negTV PDSCH TV
        if(SimCtrl.negTV.enable == 1) % enable negTV by overwriting the invalid PDSCH params
            pduFieldName = SimCtrl.negTV.pduFieldName;
            pduFieldValue = SimCtrl.negTV.pduFieldValue;
            
            if length(pduFieldName) ~= length(pduFieldValue)
                error('negTV configuration error: pduFieldName and pduFieldValue must have the same length');
            end

            for pduFieldIdx = 1:length(pduFieldName) % extact the params and overwrite pdu
                pdu.(pduFieldName{pduFieldIdx}) = pduFieldValue{pduFieldIdx};
            end
        end
        pdu = formatU32Struct(pdu);
        hdf5_write_nv_exp(h5File, pduName, pdu);
        hdf5_write_nv_exp(h5File, [pduName, '_payload'], uint8_convert(payload, 0));
    case 'csirs'
        pdu.type = 4;
        PM_W = pdu.PM_W;
        pdu = rmfield(pdu, 'PM_W');
        hdf5_write_nv_exp(h5File, [pduName, '_PM_W'], complex(single(PM_W)));
        pdu = formatU32Struct(pdu);
        hdf5_write_nv_exp(h5File, pduName, pdu);
    case 'prach'
        pdu.type = 6;
%         formatSet = {'0', '1', '2', '3', 'A1', 'A2', 'A3', 'B1', ...
%             'B4', 'C0', 'C2', 'A1/B1', 'A2/B2', 'A3/B3'};
%         [ismem, idx] = ismember(pdu.prachFormat, formatSet);
%         pdu.prachFormat = idx - 1;
        if isfield(pdu, 'payload')
            payload = pdu.payload;
            pdu = rmfield(pdu, 'payload');
        end
        pdu.idxInd = idxIndication;
        pdu.nInd = 1;
        
        % Overwrite the invalid params in SimCtrl.negTV PRACH TV
        if(SimCtrl.negTV.enable == 1) && any(strcmp(SimCtrl.negTV.channel, {'PRACH'})) && (isfield(SimCtrl.negTV, 'pduFieldName')) % enable negTV by overwriting the invalid PRACH params
            pduFieldName = SimCtrl.negTV.pduFieldName;
            pduFieldValue = SimCtrl.negTV.pduFieldValue;
            
            if length(pduFieldName) ~= length(pduFieldValue)
                error('negTV configuration error: pduFieldName and pduFieldValue must have the same length');
            end

            for pduFieldIdx = 1:length(pduFieldName) % extact the params and overwrite pdu
                pdu.(pduFieldName{pduFieldIdx}) = pduFieldValue{pduFieldIdx};
            end
        end

        pdu = formatU32Struct(pdu);
        hdf5_write_nv_exp(h5File, pduName, pdu);
        detection = [];
        numPrmb = uint32(payload.detIdx);
        for idxPrmb = 1:numPrmb
            prmbIdx_det(idxPrmb) = payload.prmbIdx_det(idxPrmb);
            peak_det(idxPrmb) = payload.peak_det(idxPrmb);
            delay_time_det(idxPrmb) = payload.delay_time_det(idxPrmb);
        end
        hdf5_write_nv_exp(h5File, [pduName, '_numPrmb'], uint32(numPrmb));
        if numPrmb > 0
            hdf5_write_nv_exp(h5File, [pduName, '_prmbIdx'], uint32(prmbIdx_det));
            hdf5_write_nv_exp(h5File, [pduName, '_peak'], single(peak_det));
            hdf5_write_nv_exp(h5File, [pduName, '_delay'], single(delay_time_det));
        end
        hdf5_write_nv_exp(h5File, [pduName, '_rssi'], single(payload.rssi_det));
        hdf5_write_nv_exp(h5File, [pduName, '_numAnt'], uint32(length(payload.antRssi_det)));
        hdf5_write_nv_exp(h5File, [pduName, '_antRssi'], single(payload.antRssi_det));
        hdf5_write_nv_exp(h5File, [pduName, '_noise'], single(payload.noise_det));

        % RACH.indication
        indicationName = ['IND', num2str(idxIndication)];
        indication = [];
        indication.idxPdu = uint32(idxPdu);
        indication.type = uint32(16);
        indication.SymbolIndex = uint32(payload.SymbolIndex);
        indication.SlotIndex = uint32(payload.SlotIndex);
        indication.FreqIndex = uint32(payload.FreqIndex);
        indication.avgRssi = uint32(round((payload.rssi_det + 64)/0.5)); % [0:254] => [-64:0.5:63] dB
        indication.avgSnr = uint32(255); % invalid  [0:254] => [-64:0.5:63] dB
        indication.avgNoise = uint32(round(10*(payload.noise_det - ul_gain_calibration + 152))); % [0:1520] => [-152:0.1:0] dBm
        indication.numPreamble =  uint32(numPrmb);
        hdf5_write_nv_exp(h5File, indicationName, indication);
        if numPrmb > 0
            preambleIndex = prmbIdx_det;
            hdf5_write_nv_exp(h5File, [indicationName,'_preambleIndex'], uint32(preambleIndex));
            Tc = 1/(480e3*4096);
            mu = carrier.mu;
            TimingAdvance = round(delay_time_det/(16*64*Tc/2^mu)); % [0:3846]
            hdf5_write_nv_exp(h5File, [indicationName,'_TimingAdvance'], uint32(TimingAdvance));
            TimingAdvanceNano = min(2005000, round(delay_time_det*1e9)); % [0, 2005000]
            hdf5_write_nv_exp(h5File, [indicationName,'_TimingAdvanceNano'], uint32(TimingAdvanceNano));
            PreamblePwr = (10*log10(peak_det) + 140 - ul_gain_calibration)*1000; % [0:170000] => [-140:0.001:30] dBm
            hdf5_write_nv_exp(h5File, [indicationName,'_PreamblePwr'], uint32(PreamblePwr));
        end
        idxIndication = idxIndication + 1;
    case 'pucch'
        pdu.type = 7;
        if isfield(pdu, 'payload')
            payload = pdu.payload;
            pdu = rmfield(pdu, 'payload');
        end
        if isfield(pdu, 'SrValues')
            SrValues = pdu.SrValues;
            pdu = rmfield(pdu, 'SrValues');
        end
        if isfield(pdu, 'HarqValues')
            HarqValues = pdu.HarqValues;
            pdu = rmfield(pdu, 'HarqValues');
        end
        if isfield(pdu, 'UciP1ToP2Crpd')
            UciP1ToP2Crpd_numPart2s = pdu.UciP1ToP2Crpd.numPart2s;
            pdu = rmfield(pdu, 'UciP1ToP2Crpd');
            pdu.UciP1ToP2Crpd_numPart2s = UciP1ToP2Crpd_numPart2s;
        end
        CsiP1Exist = 0;
        if isfield(pdu, 'CsiP1Values')
            CsiP1Values = pdu.CsiP1Values;
            pdu = rmfield(pdu, 'CsiP1Values');
            CsiP1Exist = 1;
        end
        CsiP2Exist = 0;
        if isfield(pdu, 'CsiP2Values')
            CsiP2Values = pdu.CsiP2Values;
            pdu = rmfield(pdu, 'CsiP2Values');
            CsiP2Exist = 1;
        end
        pdu.idxInd = idxIndication;
        pdu.nInd = 1;

        %% UCI inditation
        indicationName = ['IND', num2str(idxIndication)];
        indication = [];
        
        %% TA measurement
        % TA report not supported for PF0 yet
        indication.TimingAdvance     = 0xFFFF;
        indication.TimingAdvanceNano = int16(-1); % 0xFFFF = -1 in int16


        %% load SNR, RSSI, RSRP values
        RSSI = pdu.RSSI - ul_gain_calibration; % value in dBm
        indication.RSSI = uint16(min(1280, max(0, round((RSSI+128)*10))));
        
        RSRP = pdu.RSRP - ul_gain_calibration; % value in dBm
        indication.RSRP = uint16(min(1280, max(0, round((RSRP+140)*10))));

        snrdB = pdu.snrdB;
        if pdu.FormatType~=0 % SNR report not supported for PF0 yet
            indication.SNR = int16(max(-32767,round(snrdB/0.002)));
        else
            indication.SNR = int16(-1); % 0xFFFF = -1 in int16
        end
        
        if pdu.DTXthreshold == 1.0
            pdu.DTXthreshold = -100.0;
        end

        %% load interference/noise variance values
        if pdu.FormatType == 2 || pdu.FormatType == 3
            indication.noiseVardB = int16(min(1520 ,max(0, round(10*(pdu.InterfDB - ul_gain_calibration + 152))))); % [0:1520] => [-152:0.1:0] dBm
        end
        
        pdu = formatU32Struct(pdu, {'RSSI','RSRP','InterfDB','snrdB','taEstMicroSec', 'DTXthreshold'});
        
        hdf5_write_nv_exp(h5File, pduName, pdu);
        hdf5_write_nv_exp(h5File, [pduName, '_payload'], uint32(payload));

        %% UCI.indication for PUCCH
        switch pdu.FormatType
            case {0, 1}
                indication.idxPdu = uint32(idxPdu);
                indication.type = uint32(17);
                indication.PucchFormat = uint32(pdu.FormatType);
                if pdu.FormatType == 0
                    indication.UL_CQI = uint32(255); % invalid for PF0
                else
                    indication.UL_CQI = uint32(min(255 ,max(0, round((snrdB+64)*2)))); % [0:255] => [-64:0.5:63.5] dB
                end
                
                % TA measurement
                if pdu.FormatType == 1 && payload(1) ~= 2
                    Tc = 1/(480e3*4096);
                    mu = carrier.mu;
                    indication.TimingAdvance     = uint16(min(63, max(0, round(pdu.taEstMicroSec*1e-6/(16*64*Tc/2^mu)) + 31)));
                    indication.TimingAdvanceNano = int16(min(16800, max(-16800,round(1000*pdu.taEstMicroSec))));
                end
                
                indication.SRindication = uint32(pdu.SRindication);
                indication.SRconfidenceLevel = uint32(pdu.SRconfidenceLevel);
                indication.NumHarq = uint32(pdu.BitLenHarq);
                indication.HarqconfidenceLevel = uint32(pdu.HarqconfidenceLevel);
                hdf5_write_nv_exp(h5File, indicationName, indication);
                hdf5_write_nv_exp(h5File, [indicationName, '_HarqValue'], uint32(payload(1:pdu.BitLenHarq))); % for FAPI 10.04

                if payload(1) == 2
                    hdf5_write_nv_exp(h5File, [indicationName, '_HarqValueFapi1002'], uint32(payload(1:pdu.BitLenHarq))); % for FAPI 10.02
                else
                    hdf5_write_nv_exp(h5File, [indicationName, '_HarqValueFapi1002'], uint32(ones(1, pdu.BitLenHarq)-payload(1:pdu.BitLenHarq))); % for FAPI 10.02
                end
                idxIndication = idxIndication + 1;
            case {2, 3}
                indication.idxPdu = uint32(idxPdu);
                indication.type = uint32(17);
                indication.PucchFormat = uint32(pdu.FormatType);
                indication.UL_CQI = uint32(round(snrdB+64)*2); % [0:255] => [-64:0.5:63] dB
                indication.SrBitLen = uint32(length(SrValues));
                indication.HarqDetectionStatus = uint8(pdu.HarqDetectionStatus);
                indication.CsiPart1DetectionStatus = uint8(pdu.CsiPart1DetectionStatus);
                indication.CsiPart2DetectionStatus = uint8(pdu.CsiPart2DetectionStatus);
                
                if pdu.DTX
                    indication.HarqCrc = uint32(2);
                else
                    if length(payload) < 12
                        indication.HarqCrc = uint32(2);
                    else
                        indication.HarqCrc = uint32(0); % always set to PASS
                    end
                    
                    % TA measurement
                    Tc = 1/(480e3*4096);
                    mu = carrier.mu;
                    indication.TimingAdvance     = uint16(min(63, max(0, round(pdu.taEstMicroSec*1e-6/(16*64*Tc/2^mu)) + 31)));
                    indication.TimingAdvanceNano = int16(min(16800, max(-16800,round(1000*pdu.taEstMicroSec))));
                end
                indication.BitLenHarq = uint32(pdu.BitLenHarq);
                if CsiP1Exist
                    if pdu.DTX
                        indication.CsiPart1Crc = uint32(2);
                    else
                        indication.CsiPart1Crc = uint32(0); % always set to PASS
                    end
                    indication.CsiPart1BitLen = uint32(length(CsiP1Values));
                end
                if CsiP2Exist
                    if pdu.DTX
                        indication.CsiPart2Crc = uint32(2);
                    else
                        indication.CsiPart2Crc = uint32(0); % always set to PASS
                    end
                    indication.CsiPart2BitLen = uint32(length(CsiP2Values));
                end
                hdf5_write_nv_exp(h5File, indicationName, indication);
                hdf5_write_nv_exp(h5File, [indicationName, '_SrPayload'], uint32(SrValues));
                hdf5_write_nv_exp(h5File, [indicationName, '_HarqPayload'], uint32(HarqValues));
                if CsiP1Exist
                    hdf5_write_nv_exp(h5File, [indicationName, '_CsiPart1Payload'], uint32(CsiP1Values));
                end
                if CsiP2Exist
                    hdf5_write_nv_exp(h5File, [indicationName, '_CsiPart2Payload'], uint32(CsiP2Values));
                end
                idxIndication = idxIndication + 1;
            otherwise
                error('PUCCH format is not supported ...');
        end
    case 'pusch'
        pdu.type = 8;
        pdu.targetCodeRate = pdu.targetCodeRate;
        pdu.DmrsSymbPos = bin2dec(num2str(fliplr(pdu.DmrsSymbPos)));
        pdu.dmrsPorts = bin2dec(num2str(pdu.dmrsPorts));
        payload = pdu.payload;
        harqUci = pdu.harqUci;
        harqUci_earlyHarq = pdu.harqUci_earlyHarq;
        csi1Uci = pdu.csi1Uci;
        csi2Uci = pdu.csi2Uci;
        cbErr_isfield = 0;
        if isfield(pdu, 'cbErr')
            cbErr_isfield = 1;
            cbErr = pdu.cbErr;
            pdu = rmfield(pdu, 'cbErr');
        end

        if isfield(pdu, 'foCompensationBuffer')
            pdu = rmfield(pdu, 'foCompensationBuffer');
        end
%         if (pdu.newDataIndicator == 0)
%             % L2 will signal a bogus mcsIndex and L1 needs to recall the correct mcsIndex from the HARQ buffer
%             % We don't simulate this in cuPHY, since cuPHY-CP (cuphydriver) handles HARQ buffering.  So we only
%             % output the bogus mcsIndex to the FAPI TV.h5.
%             pdu.mcsIndex = 29;
%         end
        pdu = rmfield(pdu, 'payload');
        pdu = rmfield(pdu, 'harqUci');
        pdu = rmfield(pdu, 'harqUci_earlyHarq');
        pdu = rmfield(pdu, 'csi1Uci');
        pdu = rmfield(pdu, 'csi2Uci');

        pdu.idxInd = idxIndication;
        isDataPresent = bitand(uint16(pdu.pduBitmap),uint16(2^0));
        isUciPresent = bitand(uint16(pdu.pduBitmap),uint16(2^1));
        pdu.nInd = 0;
        if isDataPresent
            pdu.nInd = pdu.nInd + 1;
        end
        if isUciPresent
            pdu.nInd = pdu.nInd + 1;
        end
        
        % Add measurements to indication field
        measurements.rsrpdB = pdu.rsrpdB;
        measurements.rsrpdB_ehq = pdu.rsrpdB_ehq;
        measurements.dmrsRssiReportedDb = pdu.dmrsRssiReportedDb;
        measurements.dmrsRssiReportedDb_ehq = pdu.dmrsRssiReportedDb_ehq;

        measurements.sinrdB = pdu.sinrdB;
        measurements.sinrdB_ehq = pdu.sinrdB_ehq;
        measurements.postEqSinrdB = pdu.postEqSinrdB;
        
        measurements.noiseVardB = pdu.noiseVardB;
        measurements.postEqNoiseVardB = pdu.postEqNoiseVardB;
        
        % Remove measurements fields from 'pdu' as they are invalued values
        % after conversion to uint32
        %pdu = rmfield(pdu,'postEqNoiseVardB');
        %pdu = rmfield(pdu,'noiseVardB');
        %pdu = rmfield(pdu,'rsrpdB');
        %pdu = rmfield(pdu,'sinrdB');
        %pdu = rmfield(pdu,'postEqSinrdB');
        %pdu = rmfield(pdu,'dmrsRssiReportedDb');
        
        %Convert measurements to FAPI format
        %measurements.rsrpdB = uint16(max(0,min(1280,measurements.rsrpdB/0.1)));
        %measurements.dmrsRssiReportedDb = uint16(max(-120,min(1520,measurements.dmrsRssiReportedDb/0.1+120)));
        measurements.sinrdB = int16(max(-32767,round(measurements.sinrdB/0.002)));
        measurements.sinrdB_ehq = int16(max(-32767,round(measurements.sinrdB_ehq/0.002)));
        measurements.postEqSinrdB = int16(max(-32767,round(measurements.postEqSinrdB/0.002)));
%         measurements.noiseVardB = int16(max(-32767,round(measurements.noiseVardB/0.002)));
%         measurements.postEqNoiseVardB = int16(max(-32767,round(measurements.postEqNoiseVardB/0.002)));
        measurements.noiseVardB = int16(min(1520, max(0,round((measurements.noiseVardB+152-ul_gain_calibration)*10))));
        measurements.postEqNoiseVardB = int16(min(1520, max(0,round((measurements.postEqNoiseVardB+152-ul_gain_calibration)*10))));
        
        % convert pdu format and write to TV
%         pdu = formatU32Struct(pdu);
        pdu = formatU32Struct(pdu, {'foForgetCoeff','ldpcEarlyTerminationPerUe','ldpcMaxNumItrPerUe','postEqNoiseVardB','noiseVardB','rsrpdB', 'rsrpdB_ehq', ...
            'sinrdB', 'sinrdB_ehq', 'postEqSinrdB','dmrsRssiReportedDb', 'dmrsRssiReportedDb_ehq', 'cfoEstHz', 'toEstMicroSec'});

        % Overwrite the invalid params in SimCtrl.negTV PUSCH TV
        if((SimCtrl.negTV.enable == 1) || (SimCtrl.negTV.enable == 5)) && any(strcmp(SimCtrl.negTV.channel, {'PUSCH'})) && (isfield(SimCtrl.negTV, 'pduFieldName')) % enable negTV by overwriting the invalid PUSCH params
            pduFieldName = SimCtrl.negTV.pduFieldName;
            pduFieldValue = SimCtrl.negTV.pduFieldValue;
            
            if length(pduFieldName) ~= length(pduFieldValue)
                error('negTV configuration error: pduFieldName and pduFieldValue must have the same length');
            end

            for pduFieldIdx = 1:length(pduFieldName) % extact the params and overwrite pdu
                pdu.(pduFieldName{pduFieldIdx}) = pduFieldValue{pduFieldIdx};
            end

            if(SimCtrl.negTV.enable == 5)
                pdu.tbErr = uint32(1); % FAPI overwrite for TC7712
            end
        end

        hdf5_write_nv_exp(h5File, pduName, pdu);
        hdf5_write_nv_exp(h5File, [pduName, '_payload']          , uint8_convert(payload, 0));
        hdf5_write_nv_exp(h5File, [pduName, '_harqUci']          , uint8_convert(harqUci));
        hdf5_write_nv_exp(h5File, [pduName, '_harqUci_earlyHarq'], uint8_convert(harqUci_earlyHarq));
        hdf5_write_nv_exp(h5File, [pduName, '_csi1Uci']          , uint8_convert(csi1Uci));
        hdf5_write_nv_exp(h5File, [pduName, '_csi2Uci']          , uint8_convert(csi2Uci));
        if cbErr_isfield
            if(SimCtrl.negTV.enable == 5)
                hdf5_write_nv_exp(h5File, [pduName, '_cbErr'], uint8_convert(ones(length(cbErr), 1), 0));
            else
                hdf5_write_nv_exp(h5File, [pduName, '_cbErr'], uint8_convert(cbErr, 0));
            end
        end

        % CRC.indication with PUSCH data
        if isDataPresent
            indicationName = ['IND', num2str(idxIndication)];
            indication = [];
            indication.idxPdu = uint32(idxPdu);
            indication.type = uint32(18);
            indication.TbCrcStatus = uint32(pdu.tbErr);
            indication.NumCb = uint32(length(cbErr));
            %         indication.CbCrcStatus = uint32(bin2dec(num2str(fliplr(cbErr(:)')))); % LSB is the first cbErr
            indication.UL_CQI = uint32(round(pdu.sinrdB+64)*2); % [0:255] => [-64:0.5:63] dB
            indication.UL_CQI_POSTEQ = uint32(round(pdu.postEqSinrdB+64)*2); % [0:255] => [-64:0.5:63] dB
            Tc = 1/(480e3*4096);
            mu = carrier.mu;
            TimingAdvance = round(pdu.toEstMicroSec*1e-6/(16*64*Tc/2^mu));
            indication.TimingAdvance = uint32(TimingAdvance + 31); % [0:63] => [-31:32]
            indication.TimingAdvanceNano = int16(min(16800, max(-16800,round(1000*pdu.toEstMicroSec)))); % [-16800, 16800]
            RSSI = measurements.dmrsRssiReportedDb - ul_gain_calibration;
            if(RSSI < -128.0) RSSI=-128.0; elseif(RSSI > 0) RSSI=0; end
            indication.RSSI = uint32(round(RSSI+128)*10); % [0:1280] => [-128:0.1:0] dBm
            RSSI_ehq = measurements.dmrsRssiReportedDb_ehq - ul_gain_calibration;
            if(RSSI_ehq < -128.0) RSSI_ehq=-128.0; elseif(RSSI_ehq > 0) RSSI_ehq=0; end
            indication.RSSI_ehq = uint32(round(RSSI_ehq+128)*10); % [0:1280] => [-128:0.1:0] dBm
            % RSRP for SCF-FAPIv4
            RSRP = measurements.rsrpdB - ul_gain_calibration;
            indication.RSRP = uint32(round(RSRP+140)*10); % [0:1280] => [-140:0.1:-12] dBm
            RSRP_ehq = measurements.rsrpdB_ehq - ul_gain_calibration;
            indication.RSRP_ehq = uint32(round(RSRP_ehq+140)*10); % [0:1280] => [-140:0.1:-12] dBm
            % SINR and Pn 
            indication.sinrdB = measurements.sinrdB;
            indication.sinrdB_ehq = measurements.sinrdB_ehq;
            indication.postEqSinrdB = measurements.postEqSinrdB;
            indication.noiseVardB = measurements.noiseVardB ;
            indication.postEqNoiseVardB = measurements.postEqNoiseVardB;
            hdf5_write_nv_exp(h5File, indicationName, indication);
            if(SimCtrl.negTV.enable == 5)
                hdf5_write_nv_exp(h5File, [indicationName, '_CbCrcStatus'], uint8_convert(ones(length(cbErr), 1), 0));
            else
                hdf5_write_nv_exp(h5File, [indicationName, '_CbCrcStatus'], uint8_convert(cbErr, 0));
            end
%             hdf5_write_nv_exp(h5File, [indicationName, '_data'], uint8_convert(payload, 0));
            idxIndication = idxIndication + 1;
        end

        % UCI.indication for UciOnPUSCH
        if isUciPresent
            indicationName = ['IND', num2str(idxIndication)];
            indication = [];
            indication.idxPdu = uint32(idxPdu);
            indication.type = uint32(15);
            indication.UL_CQI = uint32(round(pdu.sinrdB+64)*2); % [0:255] => [-64:0.5:63] dB
            indication.UL_CQI_POSTEQ = uint32(round(pdu.postEqSinrdB+64)*2); % [0:255] => [-64:0.5:63] dB
            Tc = 1/(480e3*4096);
            mu = carrier.mu;
            TimingAdvance = round(pdu.toEstMicroSec*1e-6/(16*64*Tc/2^mu));
            indication.TimingAdvance = uint32(TimingAdvance + 31); % [0:63] => [-31:32]
            indication.TimingAdvanceNano = int16(min(16800, max(-16800,round(1000*pdu.toEstMicroSec)))); % [-16800, 16800]
            RSSI = measurements.dmrsRssiReportedDb - ul_gain_calibration;
            if(RSSI < -128.0) RSSI=-128.0; elseif(RSSI > 0) RSSI=0; end
            indication.RSSI = uint32(round(RSSI+128)*10); % [0:1280] => [-128:0.1:0] dBm
            RSSI_ehq = measurements.dmrsRssiReportedDb_ehq - ul_gain_calibration;
            if(RSSI_ehq < -128.0) RSSI_ehq=-128.0; elseif(RSSI_ehq > 0) RSSI_ehq=0; end
            indication.RSSI_ehq = uint32(round(RSSI_ehq+128)*10); % [0:1280] => [-128:0.1:0] dBm
            % RSRP for SCF-FAPIv4
            RSRP = measurements.rsrpdB - ul_gain_calibration;
            indication.RSRP = uint32(round(RSRP+140)*10); % [0:1280] => [-140:0.1:-12] dBm
            RSRP_ehq = measurements.rsrpdB_ehq - ul_gain_calibration;
            indication.RSRP_ehq = uint32(round(RSRP_ehq+140)*10); % [0:1280] => [-140:0.1:-12] dBm
            
            % SINR and Pn
            indication.sinrdB = measurements.sinrdB;
            indication.sinrdB_ehq = measurements.sinrdB_ehq;
            indication.postEqSinrdB = measurements.postEqSinrdB;
            indication.noiseVardB = measurements.noiseVardB ;
            indication.postEqNoiseVardB = measurements.postEqNoiseVardB;
            
            % if pdu.harqDTX
            %     indication.HarqCrc = uint32(2);
            % else
            %     indication.HarqCrc = uint32(pdu.harqCrcFlag);
            % end
            indication.HarqBitLen = uint32(length(harqUci));
            % if pdu.csi1DTX
            %     indication.CsiPart1Crc = uint32(2);
            % else
            %     indication.CsiPart1Crc = uint32(pdu.csi1CrcFlag);
            % end
            indication.CsiPart1BitLen = uint32(length(csi1Uci));
            % if pdu.csi2DTX
            %     indication.CsiPart2Crc = uint32(2);
            % else
            %     indication.CsiPart2Crc = uint32(pdu.csi2CrcFlag);
            % end
            indication.CsiPart2BitLen = uint32(length(csi2Uci));

            if pdu.harqDetStatus == 3
                indication.HarqCrc = uint32(2);
            elseif pdu.harqDetStatus == 2
                indication.HarqCrc = uint32(1);
            else
                indication.HarqCrc = uint32(0);
            end

            if pdu.harqDetStatus_earlyHarq == 3
                indication.HarqCrc_earlyHarq = uint32(2);
            elseif pdu.harqDetStatus_earlyHarq == 2
                indication.HarqCrc_earlyHarq = uint32(1);
            else
                indication.HarqCrc_earlyHarq = uint32(0);
            end

            if pdu.csi1DetStatus == 3
                indication.CsiPart1Crc = uint32(2);
            elseif pdu.csi1DetStatus == 2
                indication.CsiPart1Crc = uint32(1);
            else
                indication.CsiPart1Crc = uint32(0);
            end

            if pdu.csi2DetStatus == 3
                indication.CsiPart2Crc = uint32(2);
            elseif pdu.csi2DetStatus == 2
                indication.CsiPart2Crc = uint32(1);
            else
                indication.CsiPart2Crc = uint32(0);
            end

            indication.harqDetStatus            = pdu.harqDetStatus;
            indication.harqDetStatus_earlyHarq  = pdu.harqDetStatus_earlyHarq;
            indication.csi1DetStatus            = pdu.csi1DetStatus;
            indication.csi2DetStatus            = pdu.csi2DetStatus;
            indication.isEarlyHarq              = pdu.isEarlyHarq;

            hdf5_write_nv_exp(h5File, indicationName, indication);
            hdf5_write_nv_exp(h5File, [indicationName, '_HarqPayload']          , uint8_convert(harqUci));
            if (~isempty(harqUci_earlyHarq))
                hdf5_write_nv_exp(h5File, [indicationName, '_HarqPayload_earlyHarq'], uint8_convert(harqUci_earlyHarq));
            end
            hdf5_write_nv_exp(h5File, [indicationName, '_CsiPart1Payload']      , uint8_convert(csi1Uci));
            hdf5_write_nv_exp(h5File, [indicationName, '_CsiPart2Payload']      , uint8_convert(csi2Uci));
            idxIndication = idxIndication + 1;
        end
    case 'srs'
        indicationName    = ['IND', num2str(idxIndication)];
        indication        = [];
        indication.idxPdu = uint32(idxPdu);
        indication.type   = 9;

        % SRS report0: PRG SNR. See table 3-141 of FAPIv3
        Tc                                    = 1/(480e3*4096);
        mu                                    = carrier.mu;
        if isnan(pdu.srsOutput.toEstMicroSec)
            pdu.srsOutput.toEstMicroSec = 0.0;
        end
        TimingAdvance                         = round(pdu.srsOutput.toEstMicroSec*1e-6/(16*64*Tc/2^mu));
        report0 = [];
        report0.TimingAdvance      = uint32(TimingAdvance + 31); % [0:63] => [-31:32]
        report0.TimingAdvanceNano  = int16(min(16800, max(-16800,round(1000*pdu.srsOutput.toEstMicroSec)))); % [-16800, 16800]
        report0.wideBandSNR        = uint8(2*pdu.srsOutput.widebandSnr + 128); %[-64,63] dB step size of 0.5 dB
        report0.prgSize            = uint16(1); % cuPHY currently reports SNR for PRG size of 1
        report0.numSymbols         = uint8(2^pdu.numSymbols); % cuPHY aggregates SRS sym for SNR reports
        report0.numReportedSymbols = uint8(1); % cuPHY aggregate SRS sym for SNR reports
        report0.numPRGs            = uint16(pdu.srsOutput.nUniqueHops * pdu.srsOutput.nPrbsPerHop / report0.prgSize);
        rbSNR                      = uint8(2 * pdu.srsOutput.rbSnrs(1:report0.numPRGs)  + 128); %[-64,63] dB step size of 0.5 dB

        % SRS report1: channel I/Q matrix. See table 3-142 of FAPIv3.
        % Note: I/Q samples store in FP32 rather then FAPI format.
        report1 = [];
        report1.numUeSrsAntPorts      = uint16(2^pdu.numAntPorts);
        report1.numGnbAntennaElements = uint16(pdu.srsOutput.chEstBuffer_nRxAntSrs);
        report1.prgSize               = uint16(pdu.srsOutput.prgSizeL2); % cuPHY currently reports channel I/Q matrix per PRBG with variable prgSize
        report1.numPRGs               = uint16(pdu.srsOutput.nUniqueHops * pdu.srsOutput.nPrbsPerHop / pdu.srsOutput.prgSizeL2);

        HestBuff = complex(single(pdu.srsOutput.Hest));     % Hest buffer for the user. May contain PRBs / ueAnts not sounded by this SRS PDU
        HestToL2 = complex(single(pdu.srsOutput.HestToL2)); % Hest to be reported to L2. Only contains PRBs / ueAnts sounded by this SRS PDU
        HestNormToL2 = complex(single(int16(pdu.srsOutput.HestNormToL2))); % show error (Complex integer arithmetic is not supported) in s.(dset.Name) = data.re + (i * data.im); without single() 

        pdu.type = 9;
        pdu = formatU32Struct(pdu);
        pdu.idxInd = idxIndication;
        pdu.nInd   = 1;
        hdf5_write_nv_exp(h5File, pduName, pdu);
        hdf5_write_nv_exp(h5File, [indicationName 'report0']   , report0);
        hdf5_write_nv_exp(h5File, [indicationName 'report1']   , report1);
        hdf5_write_nv_exp(h5File, [indicationName, '_rbSNR']   , rbSNR);
        hdf5_write_nv_exp(h5File, [indicationName, '_Hest']    , HestBuff);
        hdf5_write_nv_exp(h5File, [indicationName, '_HestToL2'], HestToL2);
        hdf5_write_nv_exp(h5File, [indicationName, '_HestNormToL2'], HestNormToL2);

        idxIndication = idxIndication + 1;
    case 'bfw'
        indicationName = ['IND', num2str(idxIndication)];
        indication        = [];
        indication.idxPdu = uint32(idxPdu);
        indication.type   = 10;
        
        pdu.type = 10;
        pdu = force1dArray(pdu);
        pdu = formatU32Struct(pdu);
        hdf5_write_nv_exp(h5File, pduName, pdu);
        idxIndication = idxIndication + 1;
end

return


function saveMsg(h5File, Msg, idxMsg, isNew)

if isNew
    msgName = ['MSG', num2str(idxMsg), '_header_new'];
else
    msgName = ['MSG', num2str(idxMsg), '_header'];
end
if Msg.header.extType == 5
    Msg.header.reMask = bin2dec(num2str(Msg.header.reMask));
    Msg.header.mcScaleReMask = bin2dec(num2str(Msg.header.mcScaleReMask));
    msg_header = formatU32Struct(Msg.header);
    msg_header.mcScaleOffset = Msg.header.mcScaleOffset;
else
    Msg.header.reMask = bin2dec(num2str(Msg.header.reMask));
    msg_header = formatU32Struct(Msg.header);
    msg_header.modCompScaler = Msg.header.modCompScaler;
end
hdf5_write_nv_exp(h5File, msgName, msg_header);

if isNew
    msgName = ['MSG', num2str(idxMsg), '_payload_new'];
else
    msgName = ['MSG', num2str(idxMsg), '_payload'];
end
msg_payload = uint8(Msg.payload);
hdf5_write_nv_exp(h5File, msgName, msg_payload);

return


function Cell_Config = setCellConfig(carrier, SimCtrl)

Cell_Config.dlGridSize = uint32(carrier.N_grid_size_mu);
Cell_Config.ulGridSize = uint32(carrier.N_grid_size_mu);
Cell_Config.dlBandwidth = uint32(grid2bw(carrier.N_grid_size_mu,carrier.mu));
Cell_Config.ulBandwidth = uint32(grid2bw(carrier.N_grid_size_mu,carrier.mu));
Cell_Config.numTxAnt = uint32(carrier.numTxAnt);
Cell_Config.numRxAnt = uint32(carrier.numRxAnt);
Cell_Config.numRxAntSrs = uint32(carrier.Nant_gNB_srs);
% TODO HACK not needed once SRS buffer is seperated from Xtf
if isfield(SimCtrl, 'CellConfigPorts')
    carrier.numTxPort = SimCtrl.CellConfigPorts;
    carrier.numRxPort = SimCtrl.CellConfigPorts;
end
Cell_Config.numTxPort = uint32(carrier.numTxPort);
Cell_Config.numRxPort = uint32(carrier.numRxPort);
Cell_Config.mu = uint32(carrier.mu);
Cell_Config.phyCellId = uint32(carrier.N_ID_CELL);
Cell_Config.dmrsTypeAPos = uint32(carrier.dmrsTypeAPos);
Cell_Config.FrameDuplexType = uint32(carrier.duplex);
Cell_Config.enable_fapiv3_csi2_api = uint32(SimCtrl.enable_multi_csiP2_fapiv3);

Cell_Config.ul_gain_calibration = SimCtrl.ul_gain_calibration;
Cell_Config.enable_codebook_BF = uint32(SimCtrl.enable_codebook_BF);
Cell_Config.max_amp_ul = SimCtrl.oranComp.max_amp_ul;
Cell_Config.enable_dynamic_BF = uint32(SimCtrl.enable_dynamic_BF);
Cell_Config.enable_static_dynamic_beamforming = uint32(SimCtrl.enable_static_dynamic_beamforming);
Cell_Config.negTV_enable = uint32(SimCtrl.negTV.enable);
% 0: normal;
% 1: TV has invalid params;
% 2: Tvs with valid HARQ re-tx test cases;
% 3: Tvs with valid forceRxZero/low SNR/NDI=0 test cases;
% 4: Tvs with overlapped PRB allocation
if(SimCtrl.enable_dynamic_BF == 1 || SimCtrl.enable_static_dynamic_beamforming == 1)
    Cell_Config.numTxAnt =uint32(carrier.Nant_gNB_srs);
    Cell_Config.numRxAnt =uint32(carrier.Nant_gNB_srs);
end
Cell_Config.pusch_seg0_Tchan_start_offset = SimCtrl.timeline.pusch_seg0_Tchan_start_offset;
Cell_Config.pusch_seg0_Tchan_duration     = SimCtrl.timeline.pusch_seg0_Tchan_duration;
Cell_Config.pusch_seg1_Tchan_start_offset = SimCtrl.timeline.pusch_seg1_Tchan_start_offset;
Cell_Config.pusch_seg1_Tchan_duration     = SimCtrl.timeline.pusch_seg1_Tchan_duration;
return

function [timeline_pusch_seg0_save_once, timeline_pusch_seg1_save_once, nTimelines] = saveTimeline(h5File, pdu, SimCtrl, timeline_pusch_seg0_save_once, timeline_pusch_seg1_save_once, nTimelines)

switch (pdu.type)
    case 'pusch'
        if(timeline_pusch_seg0_save_once)
            Timeline_pusch_0.pduType = 8;
            Timeline_pusch_0.Tchan_start_offset = SimCtrl.timeline.pusch_seg0_Tchan_start_offset;
            Timeline_pusch_0.Tchan_duration     = SimCtrl.timeline.pusch_seg0_Tchan_duration;
            hdf5_write_nv_exp(h5File, 'Timeline_0', Timeline_pusch_0);
            nTimelines = nTimelines + 1;
            timeline_pusch_seg0_save_once = false;
        end

        if ((pdu.isEarlyHarq == 1) && (timeline_pusch_seg1_save_once))
            Timeline_pusch_1.pduType = 8;
            Timeline_pusch_1.Tchan_start_offset = SimCtrl.timeline.pusch_seg1_Tchan_start_offset;
            Timeline_pusch_1.Tchan_duration     = SimCtrl.timeline.pusch_seg1_Tchan_duration;
            hdf5_write_nv_exp(h5File, 'Timeline_1', Timeline_pusch_1);
            nTimelines = nTimelines + 1;
            timeline_pusch_seg1_save_once = false;
        end
end

return


function savePrecodingMatrix(h5File, table)

global SimCtrl;

PMI = 0;

PM = table.PM_W;
nPM = length(PM);
for idxPM = 1:nPM
    PMI = PMI + 1;
    PMW = PM{idxPM};
    [nPorts, nLayers] = size(PMW);
    PMW_dim.nLayers = uint32(nLayers);
    PMW_dim.nPorts = uint32(nPorts);
    PMW_coef = fp16nv(real(PMW(:)), SimCtrl.fp16AlgoSel) + 1j*fp16nv(imag(PMW(:)), SimCtrl.fp16AlgoSel);

    dataSetName = ['PM', num2str(PMI), '_dim'];
    hdf5_write_nv_exp(h5File, dataSetName, PMW_dim);
    dataSetName = ['PM', num2str(PMI), '_coef_real'];
    hdf5_write_nv_exp(h5File, dataSetName, real(PMW_coef), 'fp16');
    dataSetName = ['PM', num2str(PMI), '_coef_imag'];
    hdf5_write_nv_exp(h5File, dataSetName, imag(PMW_coef), 'fp16');
end

hdf5_write_nv_exp(h5File, 'nPM', nPM);

return

function savePrachConfig(h5File, prach, carrier, negTV)

if prach(1).L_RA == 139
    Prach_Config.prachSequenceLength = uint32(1);
else
    Prach_Config.prachSequenceLength = uint32(0);
end
Prach_Config.prachSubCSpacing = uint32(carrier.mu);
Prach_Config.restrictedSetConfig = uint32(prach(1).restrictedSet);
Prach_Config.numPrachFdOccasions = uint32(prach(1).msg1_FDM);
Prach_Config.prachConfigIndex = uint32(prach(1).configurationIndex);
Prach_Config.SsbPerRach = uint32(0); % not used
Prach_Config.prachMultipleCarriersInABand = uint32(0); % not used

hdf5_write_nv_exp(h5File, 'Prach_Config', Prach_Config);

hdf5_write_nv_exp(h5File, 'n_RO_Config', uint32(Prach_Config.numPrachFdOccasions));

for idxRO = 1:Prach_Config.numPrachFdOccasions
    RO_Config.prachRootSequenceIndex = uint32(prach(idxRO).prachRootSequenceIndex);
    RO_Config.numRootSequences = uint32(64); % not used
    RO_Config.k1  = uint32((prach(idxRO).k1 + carrier.N_grid_size_mu*12/2)/12);
    RO_Config.prachZeroCorrConf  = uint32(prach(idxRO).zeroCorrelationZone);
    RO_Config.numUnusedRootSequences = uint32(0); % not used

    % Overwrite the invalid params in negTV PRACH TV
    if(negTV.enable == 1) && any(strcmp(negTV.channel, {'PRACH'}))
        pduFieldName = negTV.roConfigFieldName;
        pduFieldValue = negTV.roConfigFieldValue;
        
        if length(pduFieldName) ~= length(pduFieldValue)
            error('negTV configuration error: roConfigFieldName and roConfigFieldValue must have the same length');
        end

        for pduFieldIdx = 1:length(pduFieldName) % extact the params and overwrite pdu
            RO_Config.(pduFieldName{pduFieldIdx}) = pduFieldValue{pduFieldIdx};
        end
    end


    hdf5_write_nv_exp(h5File, ['RO_Config_', num2str(idxRO)], RO_Config);
end

return

function saveAlgConfig(h5File, SimCtrl)

Alg_Config = SimCtrl.alg;
Alg_Config = rmfield(Alg_Config, 'enableNoiseEstForZf');
Alg_Config = rmfield(Alg_Config, 'useNrUCIDecode');
Alg_Config = rmfield(Alg_Config, 'LDPC_DMI_ML_model_path');
Alg_Config.enablePerPrgChEst = uint8(SimCtrl.bfw.enable_prg_chest);

if strcmp(SimCtrl.alg.LDPC_DMI_method,'fixed')
    Alg_Config.LDPC_DMI_method = uint8(0);
elseif strcmp(SimCtrl.alg.LDPC_DMI_method,'LUT_spef')
    Alg_Config.LDPC_DMI_method = uint8(1); 
elseif strcmp(SimCtrl.alg.LDPC_DMI_method,'per_UE')  
    Alg_Config.LDPC_DMI_method = uint8(2);
elseif strcmp(SimCtrl.alg.LDPC_DMI_method,'ML')  
    Alg_Config.LDPC_DMI_method = uint8(3); 
end

Alg_Config = formatU32Struct(Alg_Config);
hdf5_write_nv_exp(h5File, 'Alg_Config', Alg_Config);

return

function saveCsi2MapPrms(h5File, SimCtrl)
    csi2MapPrms = [];
    nCsi2Maps   = SimCtrl.nCsi2Maps;
    MAX_NUM_CSI1_PRM = SimCtrl.MAX_NUM_CSI1_PRM;
    MAX_NUM_CSI2_MAPS = SimCtrl.MAX_NUM_CSI2_MAPS;
    SimCtrl.csi2Maps_part1PrmSizes = reshape(SimCtrl.csi2Maps_part1PrmSizes, MAX_NUM_CSI1_PRM, MAX_NUM_CSI2_MAPS);

    currentBufferIdx = 0;
    for csi2MapIdx = 0 : (nCsi2Maps - 1)
        numPart1Params   = SimCtrl.csi2Maps_nPart1Prms(csi2MapIdx + 1);
        sizesPart1Params = SimCtrl.csi2Maps_part1PrmSizes(1 : numPart1Params, csi2MapIdx + 1);
        mapBitWidth      = SimCtrl.csi2Maps_sumOfPrmSizes(csi2MapIdx + 1);

        mapBufferIdx     = currentBufferIdx : (currentBufferIdx + 2^mapBitWidth - 1);
        map              = SimCtrl.csi2Maps_buffer(mapBufferIdx + 1);
        currentBufferIdx = currentBufferIdx + 2^mapBitWidth;

        csi2MapPrms(csi2MapIdx + 1).numPart1Params   = uint8(numPart1Params);
        csi2MapPrms(csi2MapIdx + 1).sizesPart1Params = uint8(sizesPart1Params);
        csi2MapPrms(csi2MapIdx + 1).mapBitWidth      = uint8(mapBitWidth);
        csi2MapPrms(csi2MapIdx + 1).map              = uint16(map);
    end

    hdf5_write_nv_exp(h5File, 'csi2MapPrms', csi2MapPrms);
    hdf5_write_nv_exp(h5File, 'nCsi2Maps', nCsi2Maps);
return



function saveDigitalBeamformingTable(h5File, SimCtrl)

enable_static_dynamic_beamforming = SimCtrl.enable_static_dynamic_beamforming;
hdf5_write_nv_exp(h5File, 'enable_static_dynamic_beamforming', ...
    enable_static_dynamic_beamforming, 'int16');
if enable_static_dynamic_beamforming
    num_static_beamIdx = SimCtrl.num_static_beamIdx;
    num_TRX_beamforming = SimCtrl.num_TRX_beamforming;
    DBT_coef = sqrt(0.5) * (randn(num_static_beamIdx, num_TRX_beamforming) +...
        j*randn(num_static_beamIdx, num_TRX_beamforming));
    hdf5_write_nv_exp(h5File, 'num_static_beamIdx', num_static_beamIdx, 'int16');
    hdf5_write_nv_exp(h5File, 'num_TRX_beamforming', num_TRX_beamforming, 'int16');
    hdf5_write_nv_exp(h5File, 'DBT_real', real(DBT_coef), 'fp16');
    hdf5_write_nv_exp(h5File, 'DBT_imag', imag(DBT_coef), 'fp16');
end

return


