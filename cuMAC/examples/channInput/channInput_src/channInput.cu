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

#include "channInput.h"

template <typename inChan_T, typename outChan_T>
channInput<inChan_T, outChan_T>::channInput(outChan_T* cpuOutChannData, const cumacCellGrpPrms* cellGrpPrms, const cumacSimParam* simParam, std::string iFileName, cudaStream_t strm, float amplifyCoe, uint8_t asimChanFlag)
{
    m_iFileName = iFileName;

    if(m_iFileName.empty())
    {
        m_chanMode = 0;
    }
    else
    {
        switch (m_iFileName.back() )
        {
            case 't':
                if(m_iFileName.at(m_iFileName.length()-3) == 'm') // .mat
                {
                    m_chanMode = 1;
                }
                else if(m_iFileName.at(m_iFileName.length()-3) == 'd') // .dat
                {
                    m_chanMode = 2;
                }
                break;
            case 'v': //.csv
                m_chanMode = 3;
                break;
            default: 
                m_chanMode = 0;
                break;
        }
    }
    if(asimChanFlag) // only for ASIM channel, its MAT file is different with QuaDriGa
    {
        m_chanMode = 4; 
    }
/* -----------------------     Ouput buffer based on max network size --------------------------*/
    // setup outChannDim: [nPrbGrp, nUe, totNumCell, nBsAnt, nUeAnt]
        // outChan Descr
    m_chanDescrCpu = new outChanDescr_t<outChan_T>;
    CUDA_CHECK_ERR(cudaMalloc((void**)&m_chanDescrGpu, sizeof(outChanDescr_t<outChan_T>)));
    m_chanDescrCpu -> nPrbGrp   = cellGrpPrms -> nPrbGrp;
    m_chanDescrCpu -> nUe       = cellGrpPrms -> nUe;
    m_chanDescrCpu -> nCell     = simParam -> totNumCell;
    m_chanDescrCpu -> nBsAnt    = cellGrpPrms -> nBsAnt;
    m_chanDescrCpu -> nUeAnt    = cellGrpPrms -> nUeAnt;

    m_outChannDim = new uint16_t[m_outChannMaxDim];
    m_outChannDim[0] = m_chanDescrCpu -> nPrbGrp;
    m_outChannDim[1] = m_chanDescrCpu -> nUe;
    m_outChannDim[2] = m_chanDescrCpu -> nCell;
    m_outChannDim[3] = m_chanDescrCpu -> nBsAnt;
    m_outChannDim[4] = m_chanDescrCpu -> nUeAnt;

    // setup output channel memory
#pragma unroll
    m_outChannSize = 1;
    for(short index=0; index < m_outChannMaxDim; index ++) 
    {
        m_outChannSize *= m_outChannDim[index];
    }
    m_CpuOutChannData = cpuOutChannData;

    // set up cell assocation buffer
    cellAssoc = new uint8_t[(m_chanDescrCpu -> nCell) * (m_chanDescrCpu -> nUe)];
#ifdef CHANN_INPUT_DEBUG_
    printf("Output channel buffer allocated with m_outChannSize = %d \n", m_outChannSize);
#endif

/* -----------------------     Input buffer --------------------------*/
    m_rawChannDim = new uint16_t[m_outChannMaxDim];
    for(int rawChannDimIdx = 0; rawChannDimIdx < m_outChannMaxDim; rawChannDimIdx++)
    {
        m_rawChannDim[rawChannDimIdx] = m_outChannDim[rawChannDimIdx];
    }
    m_rawChannSize = m_outChannSize;
    m_rawChannData = new inChan_T[m_outChannSize];

    // set buffer location, check template
    m_GPUestH_fr = reinterpret_cast<outChan_T*>(cellGrpPrms -> estH_fr);

    //m_amplifyCoe = amplifyCoe;
    m_amplifyCoe = 1.0;
#ifdef CHANN_INPUT_DEBUG_
    printf("Input channel buffer allocated with m_rawChannSize = %d \n", m_rawChannSize);
    printf("channInput created\n");
#endif

    // set cudda stream
    m_cuStream = strm;
    // set coopcells, cellIds and ueIds 
    m_numCells = simParam -> totNumCell;
    m_cellsIDs = new uint16_t[m_numCells];
    for (int cIdx = 0; cIdx < m_numCells; cIdx++) {
        m_cellsIDs[cIdx] = cIdx;
    }
    
    m_numUes = cellGrpPrms -> nUe;
    m_ueIDs = new uint16_t[m_numUes];
    m_longTermDataRate = new float[m_numUes];
    for(int ueIdx = 0; ueIdx < m_chanDescrCpu -> nUe; ueIdx++)
    {
        m_ueIDs[ueIdx]            = ueIdx;
        m_longTermDataRate[ueIdx] = 1.0f;
    }
    // default method to set cell bias
    m_biasGain = new float[(m_chanDescrCpu -> nUe) * (m_chanDescrCpu -> nCell)];
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0,1.0);
    for(int ueIdx = 0; ueIdx < m_chanDescrCpu -> nUe; ueIdx++)
    {
        int bestCell = std::rand() % (m_chanDescrCpu -> nCell);
        for(int cellIdx = 0;  cellIdx < m_chanDescrCpu -> nCell; cellIdx++)
        {
            m_biasGain[cellIdx * m_chanDescrCpu -> nUe + ueIdx] = 1e-4 *((cellIdx == bestCell) ? 2 : distribution(generator));
            // m_biasGain[cellIdx * m_chanDescrCpu -> nUe + ueIdx] = 1.0;
        }
    }

    // set nosie variance
    m_sigmaSqrd = sigmaSqrdConst; // 4.4668e-13;

    // launch config, arguments will always be channel descriptor
    m_outChannSize = (m_chanDescrCpu -> nPrbGrp) * (m_chanDescrCpu -> nUe) * (m_chanDescrCpu -> nCell) * (m_chanDescrCpu -> nBsAnt) * (m_chanDescrCpu -> nUeAnt);    
    m_chanDescrCpu -> outChanSize = m_outChannSize;
    m_chanDescrCpu -> chanPtr = m_GPUestH_fr;
    m_chanDescrCpu -> sigmaSqrd = m_sigmaSqrd;
    
    // setup done
    #ifdef CHANN_INPUT_DEBUG_
        printf("channInput created\n");
    #endif

}

template <typename inChan_T, typename outChan_T>
channInput<inChan_T, outChan_T>::~channInput()
{
    delete[] m_rawChannData;
    delete[] m_rawChannDim;
    delete[] m_cellsIDs;
    delete[] m_outChannDim;
    delete[] m_biasGain;
    delete[] m_ueIDs;
    delete[] m_longTermDataRate;
    delete[] cellAssoc;

    // free channel descriptors
    CUDA_CHECK_ERR(cudaFree(m_chanDescrGpu));
    delete m_chanDescrCpu;
}

template <typename inChan_T, typename outChan_T>
uint16_t* channInput<inChan_T, outChan_T>::getRawChannDim()
{
    return m_rawChannDim;
}

template <typename inChan_T, typename outChan_T>
uint16_t* channInput<inChan_T, outChan_T>::getOutChannDim()
{
    return m_outChannDim;
}

template <typename inChan_T, typename outChan_T>
float* channInput<inChan_T, outChan_T>::getBiasGain()
{
    return m_biasGain;
}

template <typename inChan_T, typename outChan_T>
float * channInput<inChan_T, outChan_T>::getLongTermDataRate()
{
    return m_longTermDataRate;
};

template <typename inChan_T, typename outChan_T>
void channInput<inChan_T, outChan_T>::setCellsIndex(uint16_t *cellsID)
{
    for(int i=0; i < m_numCells; i++)
    {
        m_cellsIDs[i] = cellsID[i];
    }

    // cudaMemcpyAsync(m_cellsIDs, cellGrpPrms -> cellID, m_numCells * sizeof(uint16_t), cudaMemcpyDeviceToHost, m_cuStream);
    // cudaStreamSynchronize(m_cuStream);
}

template <typename inChan_T, typename outChan_T>
void channInput<inChan_T, outChan_T>::setCellBias(float * biasGain)
{
    for(int idx = 0; idx < (m_chanDescrCpu -> nUe) * (m_chanDescrCpu -> nCell); idx++)
    {
        m_biasGain[idx] = biasGain[idx];
    }
}

template <typename inChan_T, typename outChan_T>
void channInput<inChan_T, outChan_T>::run(int TTI_index, bool channRenew)
{
    if(channRenew)
    {
        extractRawChannelData(TTI_index);
        if(m_chanMode != 0 && m_chanMode != 4) // no need to precess random channel or Asim channel
        {
            preProcessing(); // To do transfer 
        }
        transferToGPU();
        // scaling channel and noise for precision
        #ifdef USE_SNR_BASED_SCALING
        snrBasedScaling();
        #else
        autoScaling();
        #endif
        transferToCPU();
        // get latest out channel dimentions
        m_outChannDim[0] = m_chanDescrCpu -> nPrbGrp;
        m_outChannDim[1] = m_chanDescrCpu -> nUe;
        m_outChannDim[2] = m_chanDescrCpu -> nCell;
        m_outChannDim[3] = m_chanDescrCpu -> nBsAnt;
        m_outChannDim[4] = m_chanDescrCpu -> nUeAnt;
    }
}

template <typename inChan_T, typename outChan_T>
void channInput<inChan_T, outChan_T>::printRawChann(short printRawSize)
{
    for (int index=0; index < printRawSize; index++)
    {
        printf("index: %d: %1.4e + %1.4e i\n", index, float(m_rawChannData[index].x), float(m_rawChannData[index].y));
    }
    printf("Done printing raw channel data \n");
}

template <typename inChan_T, typename outChan_T>
void channInput<inChan_T, outChan_T>::printCellAssoc()
{
    for (uint cellIdx = 0; cellIdx < m_chanDescrCpu -> nCell; cellIdx ++)
    {
        printf("cellId %d: ", m_cellsIDs[cellIdx]);
        for (uint ueIdx = 0; ueIdx < m_chanDescrCpu -> nUe; ueIdx ++)
        {
            if(cellAssoc[cellIdx * (m_chanDescrCpu -> nUe) + ueIdx] == 1)
            {
                printf("%d ", m_ueIDs[ueIdx]);
            }
        }
        printf("\n");
    }
    printf("Done printing cell associations \n");
}

template <typename inChan_T, typename outChan_T>
outChan_T *channInput<inChan_T, outChan_T>::getCpuOutChann()
{
    return m_CpuOutChannData;
}

template <typename inChan_T, typename outChan_T>
outChan_T * channInput<inChan_T, outChan_T>::getGpuOutChann() 
{
    return m_GPUestH_fr;
}

template <typename inChan_T, typename outChan_T>
void channInput<inChan_T, outChan_T>::printCpuOutChann(short printOutSize)
{
    for (int index=0; index < printOutSize; index++)
    {
        printf("index: %d: %1.4e + %1.4e  i\n", index, float(m_CpuOutChannData[index].x), float(m_CpuOutChannData[index].y));
    }
    printf("Done printing out channel data from CPU \n");
}

template <typename inChan_T, typename outChan_T>
void channInput<inChan_T, outChan_T>::printGpuOutChann(short printOutSize)
{
    outChan_T * temp_CPU_buffer = new outChan_T[printOutSize];

    CUDA_CHECK_ERR(cudaMemcpyAsync(temp_CPU_buffer, m_GPUestH_fr, printOutSize * sizeof(outChan_T), cudaMemcpyDeviceToHost, m_cuStream));
    CUDA_CHECK_ERR(cudaStreamSynchronize(m_cuStream));

    for (int index=0; index< printOutSize; index++)
    {
        printf("index: %d: %1.4e + %1.4e  i\n", index, float(temp_CPU_buffer[index].x), float(temp_CPU_buffer[index].y));
    }
    printf("Done printing output channel data from GPU \n");

    delete[] temp_CPU_buffer;
}

template <typename inChan_T, typename outChan_T>
void channInput<inChan_T, outChan_T>::extractRawChannelData(int TTI_index)
{    
    int TTI_offset = TTI_index;

    switch (m_chanMode)
    {
        case 0:
            genRandomChann();
            break;

        case 1:
            readMatFile(TTI_offset);
            break;
        
        case 2: 
            readDatFile(TTI_offset);
            break;

        case 3: readCsvFile(TTI_offset);
            break;
        
        //case 4: readAsimFile(TTI_offset);
        //    break;

        default: genRandomChann();
            break;
    }
}

template <typename inChan_T, typename outChan_T>
void channInput<inChan_T, outChan_T>::genRandomChann()
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    float stddev = 0.5*sqrt(2);
    std::normal_distribution<double> distribution(0.0, stddev);
    
    #pragma unroll
    for (int prbIdx = 0; prbIdx < m_chanDescrCpu -> nPrbGrp; prbIdx++) {
        for (int ueIdx = 0; ueIdx < m_chanDescrCpu -> nUe; ueIdx++) {
            for (int cellIdx = 0; cellIdx < m_chanDescrCpu -> nCell; cellIdx++) {
                // generate channel coefficients per antenna pair
                for (int txAntIdx = 0; txAntIdx < m_chanDescrCpu -> nBsAnt; txAntIdx++) {
                    for (int rxAntIdx = 0; rxAntIdx < m_chanDescrCpu -> nUeAnt; rxAntIdx++) {
                        int index = prbIdx*(m_chanDescrCpu -> nUe)*(m_chanDescrCpu -> nCell)*(m_chanDescrCpu -> nBsAnt)*(m_chanDescrCpu -> nUeAnt);
                        index += ueIdx*(m_chanDescrCpu -> nCell)*(m_chanDescrCpu -> nBsAnt)*(m_chanDescrCpu -> nUeAnt);
                        index += cellIdx*(m_chanDescrCpu -> nBsAnt)*(m_chanDescrCpu -> nUeAnt);
                        index += txAntIdx*(m_chanDescrCpu -> nUeAnt);
                        index += rxAntIdx;
                        
                        // bias included
                        m_CpuOutChannData[index].x = distribution(generator) * m_biasGain[cellIdx * (m_chanDescrCpu -> nUe) + ueIdx];
                        m_CpuOutChannData[index].y = distribution(generator) * m_biasGain[cellIdx * (m_chanDescrCpu -> nUe) + ueIdx];
                        // no bias included
                        // m_CpuOutChannData[index].x = distribution(generator);
                        // m_CpuOutChannData[index].y = distribution(generator);
                    }
                }
            }
        }
    }
}

template <typename inChan_T, typename outChan_T>
void channInput<inChan_T, outChan_T>::readMatFile(int TTI_offset)
{
    H5::CompType m_H5CompType( sizeof(inChan_T) );
    m_H5CompType.insertMember( "real", HOFFSET(inChan_T, x), H5::PredType::NATIVE_FLOAT);
    m_H5CompType.insertMember( "imag", HOFFSET(inChan_T, y), H5::PredType::NATIVE_FLOAT);
    
    H5::H5File * file = new H5::H5File( m_iFileName.c_str() , H5F_ACC_RDONLY );
    if(file == NULL)
    {
        printf("Error opening HDF5 file\n");
    }
#ifdef CHANN_INPUT_DEBUG_  
    printf("Succesfully opened HDF5 file\n");
#endif

    // read dataset
    H5::DataSet dataset = file -> openDataSet( "H_fr_randChnRlz" );

    H5::DataSpace fspace = dataset.getSpace(); // six dimentions, [nTTI, nPrbGrp, nBsAnt, nUeAnt, nUe, totNumCell]
    hsize_t rawDims[m_outChannMaxDim+1];
    fspace.getSimpleExtentDims(rawDims, NULL);

    hsize_t start[m_outChannMaxDim+1] = {0};
    start[0] = TTI_offset; //select the current TTI of interest
    for(int outChannDimIdx = 0; outChannDimIdx < m_outChannMaxDim; outChannDimIdx++)
    {
        start[outChannDimIdx] = 0;
    }
    start[m_outChannMaxDim] = m_cellsIDs[0];
        // outChannDim: [nPrbGrp, nUe, totNumCell, nBsAnt, nUeAnt]
    hsize_t count[m_outChannMaxDim+1]  = {1,m_chanDescrCpu -> nPrbGrp,m_chanDescrCpu -> nBsAnt, m_chanDescrCpu -> nUeAnt, m_chanDescrCpu -> nUe, 1};
    fspace.selectHyperslab(H5S_SELECT_SET, count, start); //select the current BS ID of interest

    for(int cellIdx = 1; cellIdx < m_numCells; cellIdx++)
    {
        start[m_outChannMaxDim] = m_cellsIDs[cellIdx];
        // outChannDim: [nPrbGrp, nUe, totNumCell, nBsAnt, nUeAnt]
        // hsize_t count[m_outChannMaxDim+1]  = {1,m_chanDescrCpu -> nPrbGrp,m_chanDescrCpu -> nBsAnt, m_chanDescrCpu -> nUeAnt, m_chanDescrCpu -> nUe, 1};
        fspace.selectHyperslab(H5S_SELECT_OR, count, start); //select the current BS ID of interest
    }
    hsize_t fspaceSelNum = fspace.getSelectNpoints();
    hsize_t m_dim[1] = {static_cast<hsize_t>(m_rawChannSize)}; 
    H5::DataSpace mspace(1, m_dim); 
    // mspace.selectHyperslab(H5S_SELECT_SET, m_count, m_start ); 
    dataset.read(m_rawChannData, m_H5CompType, mspace, fspace);
    std::cout << "Read channel from mat file \n"; 
}

// temporarily removing reading ASIM file, keep for reference
/*
template <typename inChan_T, typename outChan_T>
void channInput<inChan_T, outChan_T>::readAsimFile(int TTI_offset)
{
    using ChannelsValueType = aerialsim::ChannelsValueType;
    using cidType = aerialsim::cidType;
    using uidType = aerialsim::uidType;

    // Step 1: create classes related to ASIM, reset CPU channel buffer
    auto channels = aerialsim::Channels<ChannelsValueType, cidType , uidType>(m_iFileName);
    auto associations = aerialsim::Associations<cidType, uidType>("associations.dat");
    auto l2_buffer = aerialsim::l2_buffer<uidType>(associations.uids);

    // Step 2: copy CID and UIDs from ASIM based on class associations
    // Note this mapping will be used globally
    auto & cidVec = associations.cids;
    auto & uidVec = associations.uids;

    uint nCellsFile = cidVec.size();
    uint nUesFile = uidVec.size();
    // check no buffer overflow of CID and UID; 
    // m_cellsIDs was allocated with size m_numCells, unit16_t
    // m_ueIDs was allocated with size m_numUes, unit16_t
    assert( (nCellsFile <= m_numCells) && (nUesFile <= m_numUes) );

    // copy UID and CIDs in case needed from outside of channInput class
    // aerialsim::Associations uses int for IDs, need type cast before copying
    for (uint cellIdx = 0; cellIdx < nCellsFile; cellIdx ++)
    {
        m_cellsIDs[cellIdx] = static_cast<uint16_t>(cidVec[cellIdx]);
    }
    for (int ueIdx = 0; ueIdx < nUesFile; ueIdx++) 
    {
        m_ueIDs[ueIdx] = static_cast<uint16_t>(uidVec[ueIdx]);
    }

    // legacy way to obtain cid and uid based on ASIM file associations.dat, no need to add asim.hpp
    // replaced with using aerialsim::Associations
    // uint nCellsFile = 0;
    // uint nUesFile = 0;
    // auto assocDict = pybind11::cast<pybind11::dict>(pickle::load("associations.dat"));
    // for (auto &cidKey : assocDict){
    //     m_cellsIDs[nCellsFile++] = cidKey.first.cast<uint16_t>();
    //     // printf("cidKey = %d, cidKey.first = %d \n", cidKey, cidKey.first);
    //     for (auto &uidKey : assocDict[cidKey.first])
    //     {
    //         m_ueIDs[nUesFile++] = uidKey.cast<uint16_t>();
    //     }
    // }   

    // Step 3: Read channel per CID and UID iteratively, avg OFDM symbols and SCs -> channel per PRBG, set channel dimensions
    //--------------------  per-slot params: nCells, nUE  -------// 
    bool updatedChanDimFlag = false;
    m_chanDescrCpu -> nUe = nUesFile; // cellGrpPrms -> nUe;
    m_chanDescrCpu -> nCell = nCellsFile; // simParam -> totNumCell;
        // save data to cuMAC buffer:
        // asim dimentions [nCell, nUe, nOfdmSymbol, nSCs, nTx, nRx]; E.g., [4, 11, 14, 3276, 4, 4]
        // cuMAC dimension [nPRB, nUE, nCell, nTx, nRx]; E.g., [273, 4, 11, 4, 4]
    for (uint cellIdx = 0; cellIdx < m_chanDescrCpu -> nCell; cellIdx ++)
    {
        uint cid = m_cellsIDs[cellIdx];
        for (int ueIdx = 0; ueIdx < m_chanDescrCpu -> nUe; ueIdx++) 
        {
            uint uid = m_ueIDs[ueIdx];
            // read the channels per cell into 4D matX tensor and combine them into a big array on CPU as output channel data
            // NOTE: Assuming full cell <-> pair, each cell sees all UEs
                      
            // read valid cid <-> uid pair before read channel
            if(channels.has_channel(cid, uid))
            {
                auto chanCidUid = channels.get(cid, uid);

                // --------------------  per-slot params: nPRBG, nBsAnt, nUeAnt   -------// 
                if( !updatedChanDimFlag ) // get from first valid cid <-> uid pair
                {
                    m_chanDescrCpu -> nPrbGrp = std::min<int>(m_chanDescrCpu -> nPrbGrp, chanCidUid.Size(1) / nPrbsPerGrpConst); // cellGrpPrms -> nPrbGrp; // TODO: need nPRBs per PRBG (nPrbsPerGrpConst) from ASIM assuming nPRB < 273
                    m_chanDescrCpu -> nBsAnt = chanCidUid.Size(2); // cellGrpPrms -> nBsAnt;
                    m_chanDescrCpu -> nUeAnt = chanCidUid.Size(3); // cellGrpPrms -> nUeAnt;

                    updatedChanDimFlag = true;
                }
    
                // read and permutation channel, take avg for channel on PRBG
                #pragma unroll
                for (int prbIdx = 0; prbIdx < m_chanDescrCpu -> nPrbGrp; prbIdx++) 
                {
                    for (int txAntIdx = 0; txAntIdx < m_chanDescrCpu -> nBsAnt; txAntIdx++) 
                    {
                        for (int rxAntIdx = 0; rxAntIdx < m_chanDescrCpu -> nUeAnt; rxAntIdx++) 
                        {
                            // calcualte cuMAC channel index
                            uint cumacChanIdx = prbIdx*(m_chanDescrCpu -> nUe)*(m_chanDescrCpu -> nCell)*(m_chanDescrCpu -> nBsAnt)*(m_chanDescrCpu -> nUeAnt);
                            cumacChanIdx += ueIdx*(m_chanDescrCpu -> nCell)*(m_chanDescrCpu -> nBsAnt)*(m_chanDescrCpu -> nUeAnt);
                            cumacChanIdx += cellIdx*(m_chanDescrCpu -> nBsAnt)*(m_chanDescrCpu -> nUeAnt);
                            cumacChanIdx += txAntIdx*(m_chanDescrCpu -> nUeAnt);
                            cumacChanIdx += rxAntIdx;
    
                            cuComplex tempChan = {0.0f, 0.0f};
                            for(uint ofdmSymIdx = 0; ofdmSymIdx < 14; ofdmSymIdx ++)
                            {
                                for(uint scIdx = 0; scIdx < nPrbsPerGrpConst * 12; scIdx ++)
                                {
                                    // using average to calculate channel on PRBG
                                    tempChan.x += chanCidUid(ofdmSymIdx, prbIdx * nPrbsPerGrpConst * 12 + scIdx, txAntIdx, rxAntIdx).real();
                                    tempChan.y += chanCidUid(ofdmSymIdx, prbIdx * nPrbsPerGrpConst * 12 + scIdx, txAntIdx, rxAntIdx).imag();
                                }
                            }
                            m_CpuOutChannData[cumacChanIdx].x = tempChan.x / 12.0f / nPrbsPerGrpConst / 14.0f;
                            m_CpuOutChannData[cumacChanIdx].y = tempChan.y / 12.0f / nPrbsPerGrpConst / 14.0f;
                        }
                    }
                }
                #ifdef CHANN_INPUT_DEBUG_
                if( (cellIdx == 0) && (ueIdx == 0) ) // read a specific channel for cid and uic
                {
                    printf("Channel under cellID %d, ueID: %d \n", cid, uid);
                    printf("chanCidUid(0,0,0,0) = %f + %f i \n", chanCidUid(0,0,0,0).real(), chanCidUid(0,0,0,0).imag());
                }
                #endif
            }
            else // invalid cid <-> uid, set 0 to channel coe
            {
                for (int prbIdx = 0; prbIdx < m_chanDescrCpu -> nPrbGrp; prbIdx++) 
                {
                    for (int txAntIdx = 0; txAntIdx < m_chanDescrCpu -> nBsAnt; txAntIdx++) 
                    {
                        for (int rxAntIdx = 0; rxAntIdx < m_chanDescrCpu -> nUeAnt; rxAntIdx++) 
                        {
                            // calcualte cuMAC channel index
                            uint cumacChanIdx = prbIdx*(m_chanDescrCpu -> nUe)*(m_chanDescrCpu -> nCell)*(m_chanDescrCpu -> nBsAnt)*(m_chanDescrCpu -> nUeAnt);
                            cumacChanIdx += ueIdx*(m_chanDescrCpu -> nCell)*(m_chanDescrCpu -> nBsAnt)*(m_chanDescrCpu -> nUeAnt);
                            cumacChanIdx += cellIdx*(m_chanDescrCpu -> nBsAnt)*(m_chanDescrCpu -> nUeAnt);
                            cumacChanIdx += txAntIdx*(m_chanDescrCpu -> nUeAnt);
                            cumacChanIdx += rxAntIdx;
                            
                            m_CpuOutChannData[cumacChanIdx].x = 0.0f;
                            m_CpuOutChannData[cumacChanIdx].y = 0.0f;
                        }
                    }
                }
            }
        }
    }

    // Step 4: read long-term data rate
    // try read long-term data rates, if not found, put as 0
    try
    {
        for (int ueIdx = 0; ueIdx < m_chanDescrCpu -> nUe; ueIdx++) 
        {
            // read long term data rate from ASIM
            // TODO: temporarily add lower bound 0.01 to avoid initial 0 data rate
            m_longTermDataRate[ueIdx] = std::min(0.01f, l2_buffer.get(m_ueIDs[ueIdx]).longTermDataRate); 
        }
    }
    catch(const std::exception& e)
    {
        printf("Warning: longTermDataRate read error, use 1.0 as longTermDataRate !\n");
        for (int ueIdx = 0; ueIdx < m_chanDescrCpu -> nUe; ueIdx++) 
        {
            m_longTermDataRate[ueIdx] = 1.0f;
        }
        // std::cerr << e.what() << '\n';
    }

    // step 5: read cell assocation convert to cuMAC data
    // in cuMAC we save cellAssoc[assocCellIdx * nUe + ueIdx] = 1; 0 otherwise
    for (uint cellIdx = 0; cellIdx < m_chanDescrCpu -> nCell; cellIdx ++)
    {
        // clear cell association
        for (uint ueIdx = 0; ueIdx < m_chanDescrCpu -> nUe; ueIdx ++)
        {
            cellAssoc[cellIdx * (m_chanDescrCpu -> nUe) + ueIdx] = 0;
        }

        auto uidxInCidxMap = associations.uidx_in_cidx[cellIdx]; // an unordered map correspoding to a cid
        for(const auto& pair : uidxInCidxMap)
        {
            cellAssoc[cellIdx * (m_chanDescrCpu -> nUe) + pair.second] = 1;
        }
    }

    // step 6: update channel dynamic descriptor
    m_outChannSize = (m_chanDescrCpu -> nPrbGrp) * (m_chanDescrCpu -> nUe) * (m_chanDescrCpu -> nCell) * (m_chanDescrCpu -> nBsAnt) * (m_chanDescrCpu -> nUeAnt);    
    m_chanDescrCpu -> outChanSize = m_outChannSize;
    m_chanDescrCpu -> chanPtr = m_GPUestH_fr;
    m_chanDescrCpu -> sigmaSqrd = 4.4668e-13;    // set nosie variance
}
*/

template <typename inChan_T, typename outChan_T>
void channInput<inChan_T, outChan_T>::readDatFile(int TTI_offset)
{
    // place holder for read .dat file

    return;
}


template <typename inChan_T, typename outChan_T>
void channInput<inChan_T, outChan_T>::readCsvFile(int TTI_offset)
{
    std::ifstream chl_in;
    chl_in.open (m_iFileName, std::ifstream::in);

    chl_in.ignore(m_rawChannSize * TTI_offset * sizeof(float) * 2); // ignore the used input channel
    float temp_read;

    for(int index=0; index<m_rawChannSize; index++)
    {
        chl_in >> temp_read;
        m_rawChannData[index].x = temp_read;
    }
    for(int index=0; index<m_rawChannSize; index++)
    {
        chl_in >> temp_read;
        m_rawChannData[index].y = temp_read;
    }  
    chl_in.close(); 
}

template <typename inChan_T, typename outChan_T>
void channInput<inChan_T, outChan_T>::preProcessing()
{
    // Permute chanenl dimensions for cuMAC
    permuteChannDim(); 
}

template <typename inChan_T, typename outChan_T>
void channInput<inChan_T, outChan_T>::permuteChannDim()
// // converted from channel dim [nBS, nUE, nPRB, nTx, nRX] to [nPRB, nUE, nBS, nTx, nRx]
{
    int cumacChanIdx = 0;
    int index_old = 0;

    switch (m_chanMode)
    {
    case 1: // From Mat file
    {    /*------------- Change precision and Permute  -----------*/
        // converted from channel dim [nPRB, nRx, nTx, nUE, nBS] column-major  to [nPRB, nUE, nBS, nTx, nRx] row-major
        // MATLAB use row major
        // HDF5 and C use column major
        #ifdef CHANN_INPUT_DEBUG_
        FILE * dbg_file = fopen("debug.txt", "w");
        #endif

        #pragma unroll
        for (int prbIdx = 0; prbIdx < m_chanDescrCpu -> nPrbGrp; prbIdx++) {
            for (int ueIdx = 0; ueIdx < m_chanDescrCpu -> nUe; ueIdx++) {
                for (int cellIdx = 0; cellIdx < m_chanDescrCpu -> nCell; cellIdx++) {
                    // generate channel coefficients per antenna pair
                    for (int txAntIdx = 0; txAntIdx < m_chanDescrCpu -> nBsAnt; txAntIdx++) {
                        for (int rxAntIdx = 0; rxAntIdx < m_chanDescrCpu -> nUeAnt; rxAntIdx++) {
                            
                            cumacChanIdx = prbIdx*(m_chanDescrCpu -> nUe)*(m_chanDescrCpu -> nCell)*(m_chanDescrCpu -> nBsAnt)*(m_chanDescrCpu -> nUeAnt);
                            cumacChanIdx += ueIdx*(m_chanDescrCpu -> nCell)*(m_chanDescrCpu -> nBsAnt)*(m_chanDescrCpu -> nUeAnt);
                            cumacChanIdx += cellIdx*(m_chanDescrCpu -> nBsAnt)*(m_chanDescrCpu -> nUeAnt);
                            cumacChanIdx += txAntIdx*(m_chanDescrCpu -> nUeAnt);
                            cumacChanIdx += rxAntIdx;

                            // change column major to row major, permuate cellIdx * nUE * nRx * nTx * nPRB
                            index_old = prbIdx * (m_chanDescrCpu -> nBsAnt)*(m_chanDescrCpu -> nUeAnt)*(m_chanDescrCpu -> nUe)*(m_chanDescrCpu -> nCell);
                            index_old += rxAntIdx * (m_chanDescrCpu -> nUeAnt)*(m_chanDescrCpu -> nUe)*(m_chanDescrCpu -> nCell);
                            index_old += txAntIdx * (m_chanDescrCpu -> nUe)*(m_chanDescrCpu -> nCell);
                            index_old += ueIdx * (m_chanDescrCpu -> nCell);
                            index_old += cellIdx;

                            m_CpuOutChannData[cumacChanIdx].x = m_rawChannData[index_old].x * m_amplifyCoe;
                            m_CpuOutChannData[cumacChanIdx].y = m_rawChannData[index_old].y * m_amplifyCoe;

                            #ifdef CHANN_INPUT_DEBUG_
                            fprintf(dbg_file,"cumac index: %d ,-> old index: %d: %f + %f  i\n", cumacChanIdx, index_old, float(m_rawChannData[index_old].x), float(m_rawChannData[index_old].y));
                            #endif

                        }
                    }
                }
            }
        }

        #ifdef CHANN_INPUT_DEBUG_
        fclose(dbg_file);
        #endif
        
        return;
    }
    
    case 2: // From dat file
    {

        return;
    }

    case 3: // From csv file
    {
        #pragma unroll
        for (int prbIdx = 0; prbIdx < m_chanDescrCpu -> nPrbGrp; prbIdx++) {
            for (int ueIdx = 0; ueIdx < m_chanDescrCpu -> nUe; ueIdx++) {
                for (int cellIdx = 0; cellIdx < m_chanDescrCpu -> nCell; cellIdx++) {
                    // generate channel coefficients per antenna pair
                    for (int txAntIdx = 0; txAntIdx < m_chanDescrCpu -> nBsAnt; txAntIdx++) {
                        for (int rxAntIdx = 0; rxAntIdx < m_chanDescrCpu -> nUeAnt; rxAntIdx++) {
                            cumacChanIdx = prbIdx*(m_chanDescrCpu -> nUe)*(m_chanDescrCpu -> nCell)*(m_chanDescrCpu -> nBsAnt)*(m_chanDescrCpu -> nUeAnt);
                            cumacChanIdx += ueIdx*(m_chanDescrCpu -> nCell)*(m_chanDescrCpu -> nBsAnt)*(m_chanDescrCpu -> nUeAnt);
                            cumacChanIdx += cellIdx*(m_chanDescrCpu -> nBsAnt)*(m_chanDescrCpu -> nUeAnt);
                            cumacChanIdx += txAntIdx*(m_chanDescrCpu -> nUeAnt);
                            cumacChanIdx += rxAntIdx;

                            index_old = cellIdx * (m_chanDescrCpu -> nUe)*(m_chanDescrCpu -> nPrbGrp)*(m_chanDescrCpu -> nBsAnt)*(m_chanDescrCpu -> nUeAnt);
                            index_old += ueIdx * (m_chanDescrCpu -> nPrbGrp)*(m_chanDescrCpu -> nBsAnt)*(m_chanDescrCpu -> nUeAnt);
                            index_old += prbIdx * (m_chanDescrCpu -> nBsAnt)*(m_chanDescrCpu -> nUeAnt);
                            index_old += txAntIdx * m_chanDescrCpu -> nUeAnt;
                            index_old += rxAntIdx;

                            m_CpuOutChannData[cumacChanIdx].x = m_rawChannData[index_old].x * m_amplifyCoe;
                            m_CpuOutChannData[cumacChanIdx].y = m_rawChannData[index_old].y * m_amplifyCoe;
                        }
                    }
                }
            }
        }
        return;
    }
    default: 
        return;
    }
}

template <typename inChan_T, typename outChan_T>
void channInput<inChan_T, outChan_T>::autoScaling()
{
    m_gridDim = {1,1,1};
    m_blockDim = {m_chanDescrCpu -> nUe, 1, 1};
    CUDA_CHECK_ERR(cudaGetFuncBySymbol(&m_functionPtr, reinterpret_cast<void*>(autoScalingKernel<outChan_T>)));
    CUDA_CHECK_ERR(cudaMemcpyAsync(m_chanDescrGpu, m_chanDescrCpu, sizeof(outChanDescr_t<outChan_T>), cudaMemcpyHostToDevice, m_cuStream));
    m_args[0] = &m_chanDescrGpu;

    // launch auto scaling kernel
    CUresult status = cuLaunchKernel(m_functionPtr, m_gridDim.x, m_gridDim.y, m_gridDim.z, m_blockDim.x, m_blockDim.y, m_blockDim.z, sizeof(float) * (m_chanDescrCpu -> nUe), m_cuStream, m_args, NULL);

    // update noise variance
    CUDA_CHECK_ERR(cudaMemcpy(m_chanDescrCpu, m_chanDescrGpu, sizeof(outChanDescr_t<outChan_T>), cudaMemcpyDeviceToHost)); // m_cuStream);
    m_sigmaSqrd = m_chanDescrCpu -> sigmaSqrd;
    // autoScalingKernel<outChan_T> <<< 1, m_chanDescrCpu -> nUe, sizeof(float) * (m_chanDescrCpu -> nUe), m_cuStream >>>(m_chanDescrGpu);
}

template <typename inChan_T, typename outChan_T>
void channInput<inChan_T, outChan_T>::snrBasedScaling()
{
    // hardcoded scaling factor
    m_chanDescrCpu -> scalingFactor = (1.0f * AFTER_SCALING_SIGMA_CONST) / sqrt(m_sigmaSqrd);
    m_sigmaSqrd = pow(AFTER_SCALING_SIGMA_CONST, 2);
    // update noise variance
    m_chanDescrCpu -> sigmaSqrd = m_sigmaSqrd;
    CUDA_CHECK_ERR(cudaMemcpyAsync(m_chanDescrGpu, m_chanDescrCpu, sizeof(outChanDescr_t<outChan_T>), cudaMemcpyHostToDevice, m_cuStream));
    m_args[0] = &m_chanDescrGpu;

    // kernel launch paramters
    m_gridDim = {1,1,1};
    m_blockDim = {1024, 1, 1};
    CUDA_CHECK_ERR(cudaGetFuncBySymbol(&m_functionPtr, reinterpret_cast<void*>(snrBasedScalingKernel<outChan_T>)));
    // launch auto scaling kernel
    CUresult status = cuLaunchKernel(m_functionPtr, m_gridDim.x, m_gridDim.y, m_gridDim.z, m_blockDim.x, m_blockDim.y, m_blockDim.z, 0, m_cuStream, m_args, NULL);
}

template <typename inChan_T, typename outChan_T>
void channInput<inChan_T, outChan_T>::transferToGPU()
{
    CUDA_CHECK_ERR(cudaMemcpyAsync(m_GPUestH_fr, m_CpuOutChannData, m_outChannSize * sizeof(outChan_T), cudaMemcpyHostToDevice, m_cuStream));
    CUDA_CHECK_ERR(cudaStreamSynchronize(m_cuStream));

    //cudaMemcpy((void *)m_GPUestH_fr, m_CpuOutChannData, m_outChannSize * sizeof(outChan_T), cudaMemcpyHostToDevice, m_cuStream);
}

template <typename inChan_T, typename outChan_T>
void channInput<inChan_T, outChan_T>::transferToCPU()
{
    CUDA_CHECK_ERR(cudaMemcpyAsync(m_CpuOutChannData, m_GPUestH_fr, m_outChannSize * sizeof(outChan_T), cudaMemcpyDeviceToHost, m_cuStream));
}

template <typename outChan_T> 
__global__ void snrBasedScalingKernel(outChanDescr_t<outChan_T> * chanDescr)
{
    float scalingFactor = chanDescr -> scalingFactor;
    uint outChanSize    = chanDescr -> outChanSize;
    outChan_T * chanPtr = chanDescr -> chanPtr;
    
    uint globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process elements in chunks of 'total threads'
    for (int chanIdx = globalThreadIdx; chanIdx < outChanSize; chanIdx += blockDim.x * gridDim.x)
    {
        chanPtr[chanIdx].x = float(chanPtr[chanIdx].x) * scalingFactor;
        chanPtr[chanIdx].y = float(chanPtr[chanIdx].y) * scalingFactor;

        // optional: check whether raw SNR within [-60 dB, 60 dB] range
        // float tempPerPrbAntSnr = float(chanPtr[chanIdx].x * chanPtr[chanIdx].x + chanPtr[chanIdx].y * chanPtr[chanIdx].y) * 0.2 / (chanDescr -> nBsAnt)  / SCALING_NOISE_SIGMA_CONST / SCALING_NOISE_SIGMA_CONST;
        // if( tempPerPrbAntSnr > 1e6 || tempPerPrbAntSnr < 1e-6)
        // {
        //     printf("Warning: out-of-range channel coe detected at chanIdx = %d: %f + %f i, tempPerPrbAntSnr = %e \n", chanIdx, float(chanPtr[chanIdx].x), float(chanPtr[chanIdx].y), tempPerPrbAntSnr);
        // }
    }
}

template <typename outChan_T> 
__global__ void autoScalingKernel(outChanDescr_t<outChan_T> * chanDescr)
{
    //      gridDim = {1,1,1};
    //      blockDim = {nUe, 1, 1};
    uint ueIdx     = threadIdx.x;
    uint nPrbGrp   = chanDescr -> nPrbGrp;
    uint nUe       = chanDescr -> nUe;
    uint nCell     = chanDescr -> nCell;
    uint nBsAnt    = chanDescr -> nBsAnt;
    uint nUeAnt    = chanDescr -> nUeAnt;
    outChan_T * chanPtr = chanDescr -> chanPtr;

    extern __shared__ float allUeSum[];

    float bestUeSum = 0.0f;
    float sigInterfSum = 0.0f;
    uint index = 0;
    
    #pragma unroll
    for (int cellIdx = 0; cellIdx < nCell; cellIdx++)
    {
        float cellUeSum = 0.0f;
        for (int prbIdx = 0; prbIdx < nPrbGrp; prbIdx++) 
        {
            index = ((prbIdx*nUe + ueIdx) * nCell + cellIdx)*nBsAnt*nUeAnt;
            for (int antIdx = 0; antIdx < nBsAnt * nUeAnt; antIdx++) 
            {
                cellUeSum +=  float(chanPtr[index].x) * float(chanPtr[index].x) + float(chanPtr[index].y) * float(chanPtr[index].y);
                index ++;
            }
        }
        sigInterfSum += cellUeSum; 
        if(bestUeSum < cellUeSum)
        {
            bestUeSum = cellUeSum; // find the best cell
        }
    }
    allUeSum[ueIdx] = bestUeSum;
    __syncthreads();

    // use paralel reudction to calculate scaling factor
    uint h = nUe;
    uint s = ceilf(h*0.5f);
    #pragma unroll
    while(s > 1)
    {
        if(ueIdx < h-s)
        {
            allUeSum[ueIdx] += allUeSum[ueIdx + s];
        }
        h = s; s = ceilf(h*0.5f);
        __syncthreads();
    }
    if(ueIdx == 0)
    {
        allUeSum[0] += allUeSum[1];
        allUeSum[0] = targetChanCoeRangeConst / sqrt(allUeSum[0]);

        float min_noise_scaling = sqrt(MinNoiseRangeConst / (chanDescr -> sigmaSqrd));
        if(allUeSum[0] < min_noise_scaling)
        {
            allUeSum[0] = min_noise_scaling;
        }
        chanDescr -> sigmaSqrd *= (allUeSum[0] * allUeSum[0]); // keep ratio between channel gain & noise = practical setting + m_desiredSNRoffSetdB; decrease noise by m_desiredSNRoffSetdB
    }
    __syncthreads();


    // apply scaling, use bestUeSum to store scaling factor
    bestUeSum = allUeSum[0]; // normlized to (nPRB * nUE)

    // apply scaling factor to all channel coe
    #pragma unroll
    for (int cellIdx = 0; cellIdx < nCell; cellIdx++)
    {
        for (int prbIdx = 0; prbIdx < nPrbGrp; prbIdx++) 
        {
            index = ((prbIdx*nUe + ueIdx) * nCell + cellIdx)*nBsAnt*nUeAnt;
            for (int antIdx = 0; antIdx < nBsAnt * nUeAnt; antIdx++) 
            {
                chanPtr[index].x = float(chanPtr[index].x) * bestUeSum;
                chanPtr[index].y = float(chanPtr[index].y) * bestUeSum;
                index ++;
            }
        }
    }
}
