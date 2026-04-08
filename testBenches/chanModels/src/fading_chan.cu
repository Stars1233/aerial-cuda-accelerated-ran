/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <inttypes.h>

#include "fading_chan.cuh"

template <typename Tcomplex>
fadingChan<Tcomplex>::fadingChan(Tcomplex* Tx, Tcomplex* freqRx, cudaStream_t strm, uint8_t fadingMode, uint16_t randSeed, uint8_t phyChannType) :
m_enableSwapTxRx(0)
{
    m_strm = strm;
    m_Tx = Tx; // frequency-domain TX signal for non-PRACH; time-domain TX signal for PRACH
    m_freqRxNoisy = freqRx;
    m_fadingMode = fadingMode;
    m_randSeed = randSeed;
    m_phyChannType = phyChannType;

    m_prach = m_phyChannType == 2? true : false;

    // get carrier and channel params
    m_carrierPrms = new cuphyCarrierPrms_t;
    m_tdlCfg = fadingMode == 1 ? new tdlConfig_t : nullptr;
    m_cdlCfg = fadingMode == 2 ? new cdlConfig_t : nullptr;

    cudaMallocHost((void**)&m_fadingChanDynDescrCpu, sizeof(fadingChanDynDescr_t<Tcomplex>));
    cudaMalloc((void**)&m_fadingChanDynDescrGpu, sizeof(fadingChanDynDescr_t<Tcomplex>));
}

template <typename Tcomplex>
fadingChan<Tcomplex>::~fadingChan()
{
    // free up configuration
    if (m_carrierPrms) delete m_carrierPrms;
    if (m_tdlCfg) delete m_tdlCfg;
    if (m_cdlCfg) delete m_cdlCfg;

    // free up tdl and ofdm classes
    if (m_tdl_chan) delete m_tdl_chan;
    if (m_cdl_chan) delete m_cdl_chan;
    if (m_ofdmMod) delete m_ofdmMod;
    if (m_ofdmDeMod) delete m_ofdmDeMod;

    // free up noise and freq rx noise free buffer
    delete m_gauNoiseAdder;
    cudaFree(m_freqRxNoiseFree);

    cudaFreeHost(m_fadingChanDynDescrCpu);
    cudaFree(m_fadingChanDynDescrGpu);
}

inline void carrier_pars_from_dataset_elem(cuphyCarrierPrms * carrierPrms, const hdf5hpp::hdf5_dataset_elem& dset_carrier);
inline void prach_carrier_pars_from_dataset_elem(cuphyCarrierPrms * carrierPrms, const hdf5hpp::hdf5_dataset_elem& dset_carrier, const hdf5hpp::hdf5_dataset_elem& prachParams);
inline void tdl_pars_from_dataset_elem(tdlConfig_t * tdlCfg, const hdf5hpp::hdf5_dataset_elem& dset_elem);
inline void cdl_pars_from_dataset_elem(cdlConfig_t * cdlCfg, const hdf5hpp::hdf5_dataset_elem& dset_elem);

template <typename Tcomplex>
void fadingChan<Tcomplex>::setup(hdf5hpp::hdf5_file& inputFile, uint8_t enableSwapTxRx)
{
    using myTscalar = decltype(type_convert(getScalarType<Tcomplex>{}));
    m_enableSwapTxRx = enableSwapTxRx || m_prach;  // prach means swap tx and rx
    // get configurations from TV file
    readCarrierChanPar(inputFile); 
    
    if (m_prach) { // PRACH
        uint32_t Nsamp_oran = m_carrierPrms->L_RA == 139? 144 : 864;
        m_freqRxDataSizeUl = Nsamp_oran * m_carrierPrms->N_rep * m_carrierPrms->N_bsLayer;
        cudaMalloc((void**)&m_freqRxNoiseFree, sizeof(Tcomplex)*m_freqRxDataSizeUl);

        // read data from TV
        read_Xtf_prach(inputFile);

        if (m_fadingMode == 1) { // TDL
            m_tdlCfg -> sigLenPerAnt = m_carrierPrms -> N_samp_slot;
            m_tdlCfg -> txSigIn = m_Tx;
            m_tdl_chan = new tdlChan<myTscalar, Tcomplex>(m_tdlCfg, m_randSeed, m_strm);

            /*-----------------------  OFDM demodulation     ---------------------*/
            // get time domain output address
            m_timeRx = m_tdl_chan -> getRxSigOut();
            m_ofdmDeMod = new ofdm_demodulate::ofdmDeModulate<myTscalar, Tcomplex>(m_carrierPrms, m_timeRx, m_freqRxNoiseFree, m_prach, 0 /*perAntSamp*/, m_strm);
        } else if (m_fadingMode == 2) {
            m_cdlCfg -> sigLenPerAnt = m_carrierPrms -> N_samp_slot;
            m_cdlCfg -> txSigIn = m_Tx;
            m_cdl_chan = new cdlChan<myTscalar, Tcomplex>(m_cdlCfg, m_randSeed, m_strm);

            /*-----------------------  OFDM demodulation     ---------------------*/
            // get time domain output address
            m_timeRx = m_cdl_chan -> getRxSigOut();
            m_ofdmDeMod = new ofdm_demodulate::ofdmDeModulate<myTscalar, Tcomplex>(m_carrierPrms, m_timeRx, m_freqRxNoiseFree, m_prach, 0 /*perAntSamp*/, m_strm);
        }
        m_ofdmMod = nullptr;
    } else {
        // allocate buffer for noise free freq rx and noise
        m_freqRxDataSizeDl = (m_carrierPrms -> N_sc) * (m_carrierPrms -> N_ueLayer) * (m_carrierPrms -> N_symbol_slot);
        m_freqRxDataSizeUl = (m_carrierPrms -> N_sc) * (m_carrierPrms -> N_bsLayer) * (m_carrierPrms -> N_symbol_slot);
        cudaMalloc((void**)&m_freqRxNoiseFree, sizeof(Tcomplex)*std::max(m_freqRxDataSizeDl, m_freqRxDataSizeUl));

        // read data from TV
        read_Xtf(inputFile);

        if(m_fadingMode == 1)
        {
            /*-----------------------  OFDM modulation     ---------------------*/
            m_ofdmMod = new ofdm_modulate::ofdmModulate<myTscalar, Tcomplex>(m_carrierPrms, m_Tx, m_strm);

            /*-----------------------  TDL channel modulation  ---------------------*/
            // get total time sample length
            uint timeTxLen = m_ofdmMod -> getTimeDataLen();
            m_tdlCfg -> sigLenPerAnt = timeTxLen / (m_carrierPrms -> N_bsLayer);  // default is downlink
            // get time GPU address
            m_timeTx = m_ofdmMod -> getTimeDataOut(); // tx time data after ofdm modulation
            m_tdlCfg -> txSigIn = m_timeTx;
            m_tdl_chan = new tdlChan<myTscalar, Tcomplex>(m_tdlCfg, m_randSeed, m_strm);

            /*-----------------------  OFDM demodulation     ---------------------*/
            // get time domain output address
            m_timeRx = m_tdl_chan -> getRxSigOut();
            m_ofdmDeMod = new ofdm_demodulate::ofdmDeModulate<myTscalar, Tcomplex>(m_carrierPrms, m_timeRx, m_freqRxNoiseFree, m_prach, 0 /*perAntSamp*/, m_strm);
        }
        else if (m_fadingMode == 2)
        {
            /*-----------------------  OFDM modulation     ---------------------*/
            m_ofdmMod = new ofdm_modulate::ofdmModulate<myTscalar, Tcomplex>(m_carrierPrms, m_Tx, m_strm);

            /*-----------------------  CDL channel modulation  ---------------------*/
            // get total time sample length
            uint timeTxLen = m_ofdmMod -> getTimeDataLen();    // default is downlink
            m_cdlCfg -> sigLenPerAnt = timeTxLen / (m_carrierPrms -> N_bsLayer);
            // get time GPU address
            m_timeTx = m_ofdmMod -> getTimeDataOut(); // tx time data after ofdm modulation
            m_cdlCfg -> txSigIn = m_timeTx;
            m_cdl_chan = new cdlChan<myTscalar, Tcomplex>(m_cdlCfg, m_randSeed, m_strm);

            /*-----------------------  OFDM demodulation     ---------------------*/
            // get time domain output address
            m_timeRx = m_cdl_chan -> getRxSigOut();
            m_ofdmDeMod = new ofdm_demodulate::ofdmDeModulate<myTscalar, Tcomplex>(m_carrierPrms, m_timeRx, m_freqRxNoiseFree, m_prach, 0 /*perAntSamp*/, m_strm);
        }
    }
    m_gauNoiseAdder = new GauNoiseAdder<Tcomplex>(1024 /*nThreads*/, m_randSeed, m_strm);

    // set dynamic descriptors
    m_fadingChanDynDescrCpu -> sigNoiseFree = m_freqRxNoiseFree;
    m_fadingChanDynDescrCpu -> sigNoisy     = m_freqRxNoisy;
    m_fadingChanDynDescrCpu -> sigLenDl     = m_freqRxDataSizeDl;
    m_fadingChanDynDescrCpu -> sigLenUl     = m_freqRxDataSizeUl;
    m_fadingChanDynDescrCpu -> seed         = m_randSeed;
    cudaMemcpy(m_fadingChanDynDescrGpu, m_fadingChanDynDescrCpu, sizeof(fadingChanDynDescr_t<Tcomplex>), cudaMemcpyHostToDevice);
}

template <typename Tcomplex>
void fadingChan<Tcomplex>::readCarrierChanPar(hdf5hpp::hdf5_file& inputFile)
{
    hdf5hpp::hdf5_dataset dset_carrier  = inputFile.open_dataset("carrier_pars");
    if (m_prach) {
        hdf5hpp::hdf5_dataset prachParams   = inputFile.open_dataset("prachParams_0");
        prach_carrier_pars_from_dataset_elem(m_carrierPrms, dset_carrier[0], prachParams[0]);
    } else {
        carrier_pars_from_dataset_elem(m_carrierPrms, dset_carrier[0]);
    }
    
    hdf5hpp::hdf5_dataset dset_chan  = inputFile.open_dataset("chan_pars");
    if (m_fadingMode == 1)
        tdl_pars_from_dataset_elem(m_tdlCfg, dset_chan[0]);
    else if (m_fadingMode == 2)
        cdl_pars_from_dataset_elem(m_cdlCfg, dset_chan[0]);

    // obtain the tx layers, check the parameters setting
    if (!m_prach) {
        hdf5hpp::hdf5_dataset Xtf_dataset = inputFile.open_dataset("X_tf_transmitted_from_UE_0");
        hdf5hpp::hdf5_dataspace XtfDataSpace = Xtf_dataset.get_dataspace();
        int ndims = XtfDataSpace.get_rank();
        std::vector<hsize_t> dims = XtfDataSpace.get_dimensions();

        // check input layers
        uint16_t ntxLayer = m_enableSwapTxRx ? m_carrierPrms -> N_ueLayer : m_carrierPrms -> N_bsLayer;
        if (!(((ndims == 2 && ntxLayer == 1) || (ndims == 3 && ntxLayer == dims[0]))))
        {
            printf("Input Xtf format error with rank %d, dims: [ ", ndims);
            for(auto i : dims)
            {
                printf("%llu ", (unsigned long long)i);
            }
            printf("] but ntxLayer = %d\n", ntxLayer);
            exit(1);
        }
    }
    /*--------------------------Below is for overwriting parameters, use caution---------------------------------*/
    // m_carrierPrms -> N_sc = 3276; // 12 * num of RBs
    // m_carrierPrms -> N_FFT = pow(2, ceilf(log2(N_sc)));  // also N_IFFT
    // m_carrierPrms -> N_bsLayer = 1;
    // m_carrierPrms -> N_ueLayer = 1;
    // m_carrierPrms -> id_slot = 0;  // per sub frame
    // m_carrierPrms -> id_subFrame = 0; // per frame
    // m_carrierPrms -> mu = 1; // numerology
    // m_carrierPrms -> cpType = 0;
    // m_carrierPrms -> f_c = 480e3 * 4096; // delta_f_max * N_f based on 38.211
    // m_carrierPrms -> f_samp = 15e3 * 8192; // 1ee3 * 2^mu * Nfft
    // m_carrierPrms -> N_symbol_slot = OFDM_SYMBOLS_PER_SLOT; // 14 OFDMs per slot
    // m_carrierPrms -> kappa_bits = 6; // kappa = 64 (2^6); constants defined in 38.211
    // m_carrierPrms -> ofdmWindowLen = 0; // ofdm windowing, not used
    // m_carrierPrms -> rolloffFactor = 0.5; // ofdm windowing, not used

    if (m_fadingMode == 1)
    {
        ASSERT(m_carrierPrms -> N_bsLayer <= m_tdlCfg -> nBsAnt, "number of bs layers must be no more than number of bs antennas");
        ASSERT(m_carrierPrms -> N_ueLayer <= m_tdlCfg -> nUeAnt, "number of ue layers must be no more than number of ue antennas");
        m_tdlCfg -> nBsAnt = m_carrierPrms -> N_bsLayer;
        m_tdlCfg -> nUeAnt = m_carrierPrms -> N_ueLayer;

        /*--------------------------Below is for overwriting parameters, use caution---------------------------------*/
        // change defualt paramters
        // m_tdlCfg -> useSimplifiedPpd = false;
        // m_tdlCfg -> delayProfile = 'A';
        // m_tdlCfg -> delaySpread = 30;
        // m_tdlCfg -> maxDopplerShift = 5;
        // m_tdlCfg -> f_samp = m_carrierPrms -> f_samp; //8192 * 15e3;
        // m_tdlCfg -> mimoCorrMat = nullptr;
        // m_tdlCfg -> nBsAnt = m_carrierPrms -> N_bsLayer;
        // m_tdlCfg -> nUeAnt = m_carrierPrms -> N_ueLayer;
        m_tdlCfg    -> nCell  = 1; // hard code for link-level
        m_tdlCfg    -> nUe    = 1; // hard code for link-level
        // m_tdlCfg -> normChannOutput = true;
        // m_tdlCfg -> fBatch = 15e3;
        // m_tdlCfg -> numPath = 48;
    }
    else if (m_fadingMode == 2)
    {
        uint16_t m_nBsAnt = std::accumulate(m_cdlCfg -> bsAntSize.begin(), m_cdlCfg -> bsAntSize.end(), 1U, std::multiplies<uint32_t>());
        uint16_t m_nUeAnt = std::accumulate(m_cdlCfg -> ueAntSize.begin(), m_cdlCfg -> ueAntSize.end(), 1U, std::multiplies<uint32_t>());
        ASSERT(m_carrierPrms -> N_bsLayer == m_nBsAnt, "number of bs layers must be equal to number of bs antennas");  // TODO: change == to <= when OFDM can handle layer mapping to ant
        ASSERT(m_carrierPrms -> N_ueLayer == m_nUeAnt, "number of ue layers must be equal to number of bs antennas");  // TODO: change == to <= when OFDM can handle layer mapping to ant
        /*--------------------------Below is for overwriting parameters, use caution---------------------------------*/
        // change defualt paramters
        // m_cdlCfg -> delayProfile = 'A';
        // m_cdlCfg -> delaySpread = 30;
        // m_cdlCfg -> maxDopplerShift = 5;
        // m_cdlCfg -> f_samp = m_carrierPrms -> f_samp; //8192 * 15e3;
        // m_cdlCfg -> mimoCorrMat = nullptr;
        // m_cdlCfg -> bsAntSize = {};
        // m_cdlCfg -> bsAntSpacing = {};
        // m_cdlCfg -> bsAntPolarAngles = {};
        // m_cdlCfg -> bsAntPattern = 1;  // 0: isotropic; 1: 38.901
        // m_cdlCfg -> ueAntSize = {};
        // m_cdlCfg -> ueAntSpacing = {};
        // m_cdlCfg -> ueAntPolarAngles = {};
        // m_cdlCfg -> ueAntPattern = 0;  // 0: isotropic; 1: 38.901
        // m_cdlCfg -> vDirection = {};
        m_cdlCfg    -> nCell  = 1; // hard code for link-level
        m_cdlCfg    -> nUe    = 1; // hard code for link-level
        // m_cdlCfg -> normChannOutput = true;
        // m_cdlCfg -> fBatch = 15e3;
        // m_cdlCfg -> numRay = 20;
    }
}

template <typename Tcomplex>
void fadingChan<Tcomplex>::read_Xtf(hdf5hpp::hdf5_file& inputFile)
{    
    m_freqTxDataSizeUl = (m_carrierPrms -> N_sc) * (m_carrierPrms -> N_ueLayer) * (m_carrierPrms -> N_symbol_slot);
    m_freqTxDataSizeDl = (m_carrierPrms -> N_sc) * (m_carrierPrms -> N_bsLayer) * (m_carrierPrms -> N_symbol_slot);
    m_freqRxDataSizeUl = (m_carrierPrms -> N_sc) * (m_carrierPrms -> N_bsLayer) * (m_carrierPrms -> N_symbol_slot);
    m_freqRxDataSizeDl = (m_carrierPrms -> N_sc) * (m_carrierPrms -> N_ueLayer) * (m_carrierPrms -> N_symbol_slot);

    uint32_t freqRxDataSize = m_enableSwapTxRx ? m_freqRxDataSizeUl : m_freqRxDataSizeDl;
    uint32_t freqTxDataSize = m_enableSwapTxRx ? m_freqTxDataSizeUl : m_freqTxDataSizeDl;
    switch(m_fadingMode)
    {
        case 0: // AWGN, read freq rx from TV, float32 in TV
        {
            float * readOutBuffer = new float[freqRxDataSize * 2];
            Tcomplex * freqRxCpu = new Tcomplex[freqRxDataSize];
            // Read input HDF5 file to read rate-matching output.
            printf("Reading %d freq rx symbols from TV \n", freqRxDataSize);

            if (m_phyChannType == 0) { // PUSCH
                hdf5hpp::hdf5_dataset Xtf_dataset = inputFile.open_dataset("X_tf");
                Xtf_dataset.read(readOutBuffer);
            } else if (m_phyChannType == 1) { // PUCCH
                hdf5hpp::hdf5_dataset Xtf_dataset = inputFile.open_dataset("DataRx");
                Xtf_dataset.read(readOutBuffer);
            }
            
            for(int i=0; i<freqRxDataSize; i++)
            {
                if(typeid(Tcomplex) == typeid(__half2)) // float16 is used, type conversion needed
                {   
                    freqRxCpu[i].x = __float2half(readOutBuffer[i*2]);
                    freqRxCpu[i].y = __float2half(readOutBuffer[i*2 + 1]);
                }
                else
                {
                    freqRxCpu[i].x = readOutBuffer[i*2];
                    freqRxCpu[i].y = readOutBuffer[i*2 + 1];  
                }
            }
            cudaMemcpyAsync(m_freqRxNoiseFree, freqRxCpu, sizeof(Tcomplex)*freqRxDataSize, cudaMemcpyHostToDevice, m_strm);
            delete[] readOutBuffer; 
            delete[] freqRxCpu;
            break;
        }

        case 1:  // TDL, read freq tx from TV, double in TV
        case 2:  // CDL, read freq tx from TV, double in TV
        {
            double * readOutBuffer = new double[freqTxDataSize * 2];
            Tcomplex * freqTxCpu = new Tcomplex[freqTxDataSize];
            // Read input HDF5 file to read rate-matching output.
            printf("Reading %d freq tx symbols from TV \n", freqTxDataSize);
            hdf5hpp::hdf5_dataset Xtf_dataset = inputFile.open_dataset("X_tf_transmitted_from_UE_0");
            Xtf_dataset.read(readOutBuffer);
            for(int i=0; i<freqTxDataSize; i++)
            {
                if(typeid(Tcomplex) == typeid(__half2)) // float16 is used, type conversion needed
                {   
                    freqTxCpu[i].x = __double2half(readOutBuffer[i*2]);
                    freqTxCpu[i].y = __double2half(readOutBuffer[i*2 + 1]);
                }
                else
                {
                    freqTxCpu[i].x = float(readOutBuffer[i*2]);
                    freqTxCpu[i].y = float(readOutBuffer[i*2 + 1]);
                }
            }
            cudaMemcpyAsync(m_Tx, freqTxCpu, sizeof(Tcomplex)*freqTxDataSize, cudaMemcpyHostToDevice, m_strm);  
            delete[] readOutBuffer; 
            delete[] freqTxCpu;
            break;
        }
        default: // report error
        {
            fprintf(stderr, "Error! unsupported fading mode \n");
            exit(1);
        }
    }
}

template <typename Tcomplex>
void fadingChan<Tcomplex>::read_Xtf_prach(hdf5hpp::hdf5_file& inputFile)
{
    if (m_fadingMode == 1 || m_fadingMode == 2) { // TDL or CDL
        m_timeTxDataSizeUl = m_carrierPrms -> N_samp_slot * m_carrierPrms -> N_ueLayer;

        double * readOutBuffer = new double[m_timeTxDataSizeUl * 2];
        Tcomplex * timeTxCpu = new Tcomplex[m_timeTxDataSizeUl];

        printf("Reading %d time-domain tx samples from TV \n", m_timeTxDataSizeUl);
        hdf5hpp::hdf5_dataset Xt_dataset = inputFile.open_dataset("X_t_transmitted_from_UE_0");
        Xt_dataset.read(readOutBuffer);
        for(int i=0; i<m_timeTxDataSizeUl; i++) {
            if(typeid(Tcomplex) == typeid(__half2)) { // float16 is used, type conversion needed
                timeTxCpu[i].x = __double2half(readOutBuffer[i*2]);
                timeTxCpu[i].y = __double2half(readOutBuffer[i*2 + 1]);
            } else {
                timeTxCpu[i].x = float(readOutBuffer[i*2]);
                timeTxCpu[i].y = float(readOutBuffer[i*2 + 1]);
            }
        }
        cudaMemcpyAsync(m_Tx, timeTxCpu, sizeof(Tcomplex)*m_timeTxDataSizeUl, cudaMemcpyHostToDevice, m_strm);  
        delete[] readOutBuffer; 
        delete[] timeTxCpu;
    } else { // AWGN
        float * readOutBuffer = new float[m_freqRxDataSizeUl * 2];
        Tcomplex * freqRxCpu = new Tcomplex[m_freqRxDataSizeUl];
        // Read input HDF5 file to read rate-matching output.
        printf("Reading %d frequency-domain rx symbols from TV \n", m_freqRxDataSizeUl);

        hdf5hpp::hdf5_dataset Xtf_dataset = inputFile.open_dataset("y_uv_rx_0");
        Xtf_dataset.read(readOutBuffer);
            
        for(int i=0; i<m_freqRxDataSizeUl; i++)
            {
                if(typeid(Tcomplex) == typeid(__half2)) // float16 is used, type conversion needed
                {   
                    freqRxCpu[i].x = __float2half(readOutBuffer[i*2]);
                    freqRxCpu[i].y = __float2half(readOutBuffer[i*2 + 1]);
                }
                else
                {
                    freqRxCpu[i].x = readOutBuffer[i*2];
                    freqRxCpu[i].y = readOutBuffer[i*2 + 1];  
                }
            }
            cudaMemcpyAsync(m_freqRxNoiseFree, freqRxCpu, sizeof(Tcomplex)*m_freqRxDataSizeUl, cudaMemcpyHostToDevice, m_strm);
            delete[] readOutBuffer; 
            delete[] freqRxCpu;
    }
}

template <typename Tcomplex>
void fadingChan<Tcomplex>::run(float refTime0, float targetSNR, uint8_t enableSwapTxRx)
{
    m_enableSwapTxRx = enableSwapTxRx || m_prach;  // prach means swap tx and rx
    if(m_fadingMode == 1) // only run in TDL mode
    {
        if (!m_prach) {
            // OFDM modulation
            m_ofdmMod -> run(m_enableSwapTxRx, m_strm);
        }
           
        // apply TDL
        m_tdl_chan -> run(refTime0, m_enableSwapTxRx);
        // OFDM demodulation
        m_ofdmDeMod -> run(m_enableSwapTxRx, m_strm);
    }
    else if (m_fadingMode == 2) // only run in CDL mode
    {
        if (!m_prach) {
            // OFDM modulation
            m_ofdmMod -> run(m_enableSwapTxRx, m_strm);
        }
           
        // apply CDL
        m_cdl_chan -> run(refTime0, m_enableSwapTxRx);
        // OFDM demodulation
        m_ofdmDeMod -> run(m_enableSwapTxRx, m_strm);
    }

    // add noise in freq domain
    addNoiseFreq(targetSNR);
}

template <typename Tcomplex>
void fadingChan<Tcomplex>::addNoiseFreq(float targetSNR)
{
    uint32_t freqDataSize = m_enableSwapTxRx ? m_freqRxDataSizeUl : m_freqRxDataSizeDl;
    cudaMemcpyAsync(m_freqRxNoisy, m_freqRxNoiseFree, freqDataSize * sizeof(Tcomplex), cudaMemcpyDeviceToDevice, m_strm);
    m_gauNoiseAdder -> addNoise(m_freqRxNoisy, freqDataSize, targetSNR);
}

template <typename Tcomplex>
void fadingChan<Tcomplex>::savefadingChanToH5File()
{
    // Initialize HDF5
    hid_t fadingChanHdf5File = H5Fcreate("fadingChanOuputData.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); // non empty existing file will be overwritten
    if (fadingChanHdf5File < 0)
    {
        fprintf(stderr, "Failed to create HDF5 file!\n");
        exit(EXIT_FAILURE);
    } 

    // m_freqRxDataSize = (m_carrierPrms -> N_sc) * (m_carrierPrms -> N_ueLayer) * (m_carrierPrms -> N_symbol_slot);
    uint16_t N_sc           = m_carrierPrms -> N_sc;
    uint16_t N_symbol_slot  = m_carrierPrms -> N_symbol_slot;
    uint16_t N_bsLayer      = m_carrierPrms -> N_bsLayer;
    uint16_t N_ueLayer      = m_carrierPrms -> N_ueLayer;
    
    // Create a compound datatype based on Tcomplex datatype
    hid_t complexDataType = H5Tcreate(H5T_COMPOUND, sizeof(Tcomplex));
    if(typeid(Tcomplex) == typeid(__half2))
    {
        hid_t fp16Type        = generate_native_HDF5_fp16_type();
        H5Tinsert(complexDataType, "re", HOFFSET(__half2, x), fp16Type);
        H5Tinsert(complexDataType, "im", HOFFSET(__half2, y), fp16Type);
    }
    else if(typeid(Tcomplex) == typeid(cuComplex))
    {
        H5Tinsert(complexDataType, "re", HOFFSET(cuComplex, x), H5T_NATIVE_FLOAT);
        H5Tinsert(complexDataType, "im", HOFFSET(cuComplex, y), H5T_NATIVE_FLOAT);
    }

    // Create a dataset in the HDF5 file to store the 3 dimension array
    uint8_t rank = 3;
    hsize_t dims[rank] = {N_sc, N_symbol_slot, 1};
    
    // write tx signals
    dims[2] = N_bsLayer;
    writeHdf5DatasetFromGpu<Tcomplex>(fadingChanHdf5File, "/Tx", complexDataType, m_Tx, dims, rank);

    // write freq rx noisy samples, freq rx nosie free samples, noise
    dims[2] = N_ueLayer;
    writeHdf5DatasetFromGpu<Tcomplex>(fadingChanHdf5File, "/freqRxNoisy", complexDataType, m_freqRxNoisy, dims, rank);
    writeHdf5DatasetFromGpu<Tcomplex>(fadingChanHdf5File, "/freqRxNoiseFree", complexDataType, m_freqRxNoiseFree, dims, rank);

    // Close HDF5 objects and free GPU memory
    H5Fclose(fadingChanHdf5File);

    if(!m_SNR.empty())
    {
        std::ofstream outputFile("SNR.txt");
    
        if (outputFile.is_open())
        {
            for (int i = 0; i < m_SNR.size(); ++i)
            {
                outputFile << m_SNR[i] << '\t';
            }
            
            outputFile.close();
        }
        else
        {
            std::cout << "Unable to open file." << std::endl;
        }
        printf("Average SNR: %f, (avg over %d iterations)\n", std::reduce(m_SNR.begin(), m_SNR.end())/float(m_SNR.size()), int(m_SNR.size()/(m_carrierPrms -> N_ueLayer)));

        outputFile.close();
    }
}

template <typename Tcomplex>
void fadingChan<Tcomplex>::calSnr(uint16_t ofdmSymIdx, uint16_t startSC, uint16_t endSC)
{
    uint16_t nUeAnt = m_carrierPrms -> N_ueLayer;
    uint16_t N_sc = m_carrierPrms -> N_sc;
    uint16_t N_symbol_slot = m_carrierPrms -> N_symbol_slot;
    uint32_t freqRxDataSize = m_enableSwapTxRx ? m_freqRxDataSizeUl : m_freqRxDataSizeDl;

    Tcomplex * tempFreqNoiseFreeBuffer = new Tcomplex[freqRxDataSize];
    Tcomplex * tempFreqNoisyBuffer = new Tcomplex[freqRxDataSize];
    cudaMemcpy(tempFreqNoiseFreeBuffer, m_freqRxNoiseFree, sizeof(Tcomplex) * freqRxDataSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(tempFreqNoisyBuffer, m_freqRxNoisy, sizeof(Tcomplex) * freqRxDataSize, cudaMemcpyDeviceToHost);

    for(uint8_t ueAntIdx = 0; ueAntIdx < nUeAnt; ueAntIdx++)
    {
        uint scOffset = (ueAntIdx * N_symbol_slot + ofdmSymIdx) * N_sc;
        float sigalSum = 0.0f;
        float noiseSum = 0.0f;
        Tcomplex tempSamp, tempNoise;
        for(uint scIdx = startSC; scIdx < endSC; scIdx++)
        {
            tempSamp = tempFreqNoiseFreeBuffer[scOffset + scIdx];
            sigalSum += float(tempSamp.x) * float(tempSamp.x) + float(tempSamp.y) * float(tempSamp.y);
        }
        sigalSum /= (endSC - startSC);
        for(uint scIdx = 0; scIdx < N_sc; scIdx++)
        {
            tempNoise.x = tempFreqNoisyBuffer[scOffset + scIdx].x - tempFreqNoiseFreeBuffer[scOffset + scIdx].x;
            tempNoise.y = tempFreqNoisyBuffer[scOffset + scIdx].y - tempFreqNoiseFreeBuffer[scOffset + scIdx].y;
            noiseSum += float(tempNoise.x) * float(tempNoise.x) + float(tempNoise.y) * float(tempNoise.y);
        }
        noiseSum /= N_sc;
        m_SNR.push_back(sigalSum / noiseSum);
    }

    delete[] tempFreqNoiseFreeBuffer;
    delete[] tempFreqNoisyBuffer;
}

/*-----------------------Below are configurations for carrier and chan_pars---------------------------------------*/
// use caution when overwriting the params read from TV
inline void carrier_pars_from_dataset_elem(cuphyCarrierPrms * carrierPrms, const hdf5hpp::hdf5_dataset_elem& dset_carrier)
{
    carrierPrms -> N_sc                   = dset_carrier["N_sc"].as<uint16_t>();
    carrierPrms -> N_FFT                  = dset_carrier["N_FFT"].as<uint16_t>();
    carrierPrms -> N_bsLayer              = dset_carrier["N_bsAnt"].as<uint16_t>();
    carrierPrms -> N_ueLayer              = dset_carrier["N_ueAnt"].as<uint16_t>();
    carrierPrms -> id_slot                = dset_carrier["id_slot"].as<uint16_t>();
    carrierPrms -> id_subFrame            = dset_carrier["id_subFrame"].as<uint16_t>();
    carrierPrms -> mu                     = dset_carrier["mu"].as<uint16_t>();
    carrierPrms -> cpType                 = dset_carrier["cpType"].as<uint16_t>();
    carrierPrms -> f_c                    = dset_carrier["f_c"].as<uint32_t>();
    carrierPrms -> f_samp                 = dset_carrier["f_samp"].as<uint32_t>();
    carrierPrms -> N_symbol_slot          = dset_carrier["N_symbol_slot"].as<uint16_t>();
    carrierPrms -> kappa_bits             = dset_carrier["kappa_bits"].as<uint16_t>();
    carrierPrms -> ofdmWindowLen          = 0;//dsedset_carriert_elem["ofdmWindowLen"].as<uint16_t>();
    carrierPrms -> rolloffFactor          = 0.5;//dset_carrier["rolloffFactor"].as<float>();
}

inline void prach_carrier_pars_from_dataset_elem(cuphyCarrierPrms * carrierPrms, const hdf5hpp::hdf5_dataset_elem& dset_carrier, const hdf5hpp::hdf5_dataset_elem& prachParams)
{
    carrierPrms -> N_sc                   = dset_carrier["N_sc"].as<uint16_t>();
    carrierPrms -> N_FFT                  = dset_carrier["N_FFT"].as<uint16_t>();
    carrierPrms -> N_bsLayer              = dset_carrier["N_bsAnt"].as<uint16_t>();
    carrierPrms -> N_ueLayer              = dset_carrier["N_ueAnt"].as<uint16_t>();
    carrierPrms -> id_slot                = dset_carrier["id_slot"].as<uint16_t>();
    carrierPrms -> id_subFrame            = dset_carrier["id_subFrame"].as<uint16_t>();
    carrierPrms -> mu                     = dset_carrier["mu"].as<uint16_t>();
    carrierPrms -> cpType                 = dset_carrier["cpType"].as<uint16_t>();
    carrierPrms -> f_c                    = dset_carrier["f_c"].as<uint32_t>();
    carrierPrms -> f_samp                 = dset_carrier["f_samp"].as<uint32_t>();
    carrierPrms -> N_symbol_slot          = dset_carrier["N_symbol_slot"].as<uint16_t>();
    carrierPrms -> kappa_bits             = dset_carrier["kappa_bits"].as<uint16_t>();
    carrierPrms -> ofdmWindowLen          = 0;//dsedset_carriert_elem["ofdmWindowLen"].as<uint16_t>();
    carrierPrms -> rolloffFactor          = 0.5;//dset_carrier["rolloffFactor"].as<float>();
    carrierPrms -> T_c                    = 1.0/float(carrierPrms -> f_c);
    carrierPrms -> N_samp_slot            = dset_carrier["N_samp_slot"].as<uint32_t>();
    carrierPrms -> k_const                = dset_carrier["k_const"].as<uint16_t>();
    carrierPrms -> N_u_mu                 = dset_carrier["N_u_mu"].as<uint32_t>();
    carrierPrms -> startRaSym             = prachParams["startRaSym"].as<uint32_t>();
    carrierPrms -> delta_f_RA             = prachParams["delta_f_RA"].as<uint32_t>();
    carrierPrms -> N_CP_RA                = prachParams["N_CP_RA"].as<uint32_t>();
    carrierPrms -> K                      = prachParams["K"].as<uint32_t>();
    carrierPrms -> k1                     = prachParams["k1"].as<int32_t>();
    carrierPrms -> kBar                   = prachParams["kBar"].as<uint32_t>();
    carrierPrms -> N_u                    = prachParams["N_u"].as<uint32_t>();
    carrierPrms -> L_RA                   = prachParams["L_RA"].as<uint32_t>();
    carrierPrms -> n_slot_RA_sel          = prachParams["n_slot_RA_sel"].as<uint32_t>();
    carrierPrms -> N_rep                  = prachParams["N_rep"].as<uint32_t>();
}


inline void tdl_pars_from_dataset_elem(tdlConfig_t * tdlCfg, const hdf5hpp::hdf5_dataset_elem& dset_elem)
{
    tdlCfg -> useSimplifiedPdp = dset_elem["useSimplifiedPdp"].as<uint8_t>(); // true for simplified pdp in 38.141, false for 38.901
    tdlCfg -> delayProfile = dset_elem["delayProfile"].as<uint8_t>() + 'A';
    tdlCfg -> delaySpread = dset_elem["delaySpread"].as<float>();
    tdlCfg -> maxDopplerShift = dset_elem["maxDopplerShift"].as<float>();
    tdlCfg -> f_samp = dset_elem["f_samp"].as<uint32_t>();
    tdlCfg -> nBsAnt = dset_elem["numBsAnt"].as<uint16_t>();
    tdlCfg -> nUeAnt = dset_elem["numUeAnt"].as<uint16_t>();
    tdlCfg -> fBatch = dset_elem["fBatch"].as<uint32_t>(); // update rate of quasi-static channel
    tdlCfg -> numPath = dset_elem["numPath"].as<uint16_t>();
    tdlCfg -> cfoHz = dset_elem["CFO"].as<float>();
    tdlCfg -> delay = dset_elem["delay"].as<float>(); 
    /*  Below are are frequency domain chan generation, not used for cuPHY testing*/
    tdlCfg -> N_sc = 0; // max 273 PRBs, setting this to 0 will not provide freq channel
    tdlCfg -> runMode = 0; // set to 1 or 2 will also generate freq channel
};

inline void cdl_pars_from_dataset_elem(cdlConfig_t * cdlCfg, const hdf5hpp::hdf5_dataset_elem& dset_elem)
{
    cdlCfg -> delayProfile = dset_elem["delayProfile"].as<uint8_t>() + 'A';
    cdlCfg -> delaySpread = dset_elem["delaySpread"].as<float>();
    cdlCfg -> maxDopplerShift = dset_elem["maxDopplerShift"].as<float>();
    cdlCfg -> f_samp = dset_elem["f_samp"].as<uint32_t>();
    cdlCfg -> fBatch = dset_elem["fBatch"].as<uint32_t>(); // update rate of quasi-static channel
    cdlCfg -> numRay = dset_elem["numRay"].as<uint16_t>();
    cdlCfg -> cfoHz = dset_elem["CFO"].as<float>();
    cdlCfg -> delay = dset_elem["delay"].as<float>();
    // antenna parameters
    cdlCfg -> bsAntSize = dset_elem["bsAntSize"].as<std::vector<uint16_t>>();
    cdlCfg -> bsAntSpacing = dset_elem["bsAntSpacing"].as<std::vector<float>>();
    cdlCfg -> bsAntPolarAngles = dset_elem["bsAntPolarAngles"].as<std::vector<float>>();
    cdlCfg -> bsAntPattern = dset_elem["bsAntPattern"].as<uint8_t>();
    cdlCfg -> ueAntSize = dset_elem["ueAntSize"].as<std::vector<uint16_t>>();
    cdlCfg -> ueAntSpacing = dset_elem["ueAntSpacing"].as<std::vector<float>>();
    cdlCfg -> ueAntPolarAngles = dset_elem["ueAntPolarAngles"].as<std::vector<float>>();
    cdlCfg -> ueAntPattern = dset_elem["ueAntPattern"].as<uint8_t>();
    cdlCfg -> vDirection = dset_elem["vDirection"].as<std::vector<float>>();
    /*  Below are are frequency domain chan generation, not used for cuPHY testing*/
    cdlCfg -> N_sc = 0; // max 273 PRBs, setting this to 0 will not provide freq channel
    cdlCfg -> runMode = 0; // set to 1 or 2 will also generate freq channel
};
