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

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_api.h"
#include "pycuphy_util.hpp"
#include "pycuphy_params.hpp"
#include "pycuphy_csirs_util.hpp"

namespace py = pybind11;

namespace pycuphy {


void printPdschStatPrms(const cuphyPdschStatPrms_t& statPrms) {

    int nCells = statPrms.nCells;
    printf("======================================\n");
    printf("PDSCH static parameters for %d cells\n", nCells);
    printf("======================================\n");

    // Parameters common across all cells
    printf("read_TB_CRC:           %4d\n", statPrms.read_TB_CRC);
    printf("full_slot_processing:  %4d\n", statPrms.full_slot_processing);
    printf("stream_priority:       %4d\n", statPrms.stream_priority);
    printf("nMaxCellsPerSlot:      %4d\n", statPrms.nMaxCellsPerSlot);
    printf("nMaxUesPerCellGroup:   %4d\n", statPrms.nMaxUesPerCellGroup);
    printf("nMaxCBsPerTB:          %4d\n", statPrms.nMaxCBsPerTB);
    printf("nMaxPrb:               %4u\n", statPrms.nMaxPrb);

    // Cell specific parameters
    printf("PDSCH static parameters for %d cells\n", nCells);
    cuphyCellStatPrm_t* cellStatPrms = statPrms.pCellStatPrms;
    cuphyPdschDbgPrms_t* cellDbgPrms = statPrms.pDbg;
    for (int cellIdx = 0; cellIdx < nCells; cellIdx++) {
        printf("  ------------------------------------\n");
        printf("  Cell %d\n", cellIdx);
        printf("  ------------------------------------\n");
        printf("  phyCellId:      %4d\n", cellStatPrms[cellIdx].phyCellId);
        printf("  nRxAnt:         %4d\n", cellStatPrms[cellIdx].nRxAnt);
        printf("  nTxAnt:         %4d\n", cellStatPrms[cellIdx].nTxAnt);
        printf("  nPrbUlBwp:      %4d\n", cellStatPrms[cellIdx].nPrbUlBwp);
        printf("  nPrbDlBwp:      %4d\n", cellStatPrms[cellIdx].nPrbDlBwp);
        printf("  mu:             %4d\n", cellStatPrms[cellIdx].mu);
        // Debug fields
        if(cellDbgPrms) {
            printf("\n  DBG:\n");
            printf("  pCfgFileName:             %s\n", cellDbgPrms[cellIdx].pCfgFileName);
            printf("  checkTbSize:              %d\n", cellDbgPrms[cellIdx].checkTbSize);
            printf("  refCheck:                 %d\n", cellDbgPrms[cellIdx].refCheck);
            printf("  cfgIdenticalLdpcEncCfgs:  %d\n", cellDbgPrms[cellIdx].cfgIdenticalLdpcEncCfgs);
        }
        printf("\n");
    }
    printf("======================================\n\n");
}


void printPdschDynPrms(const cuphyPdschDynPrms_t& dynPrms) {

    printf("======================================\n");
    printf("\nPDSCH dynamic parameters\n\n");
    printf("======================================\n");

    const cuphyPdschCellGrpDynPrm_t* cellGrpDynPrm = dynPrms.pCellGrpDynPrm;

    printf("procModeBmsk:       %4lu\n", dynPrms.procModeBmsk);

    // PDSCH data input buffers
    if (dynPrms.pDataIn == nullptr)
        printf("pDataIn:      %p\n", dynPrms.pDataIn);
    else {
        printf("pDataIn.pBufferType:   %s\n", (dynPrms.pDataIn->pBufferType == cuphyPdschDataIn_t::CPU_BUFFER) ? "CPU_BUFFER" : "GPU_BUFFER");
        printf("pDataIn.pTbInput:      {");
        for (int i = 0; i <  cellGrpDynPrm->nCells; i++) {
            if (i != 0) printf(", ");
            printf("%p", dynPrms.pDataIn->pTbInput[i]);
        }
        printf("}\n");
    }

    // PDSCH TB-CRC
    if (dynPrms.pTbCRCDataIn == nullptr)
        printf("pTbCRCDataIn:      %p\n", dynPrms.pTbCRCDataIn);
    else {
        printf("pTbCRCDataIn.pBufferType:   %s\n", (dynPrms.pTbCRCDataIn->pBufferType == cuphyPdschDataIn_t::CPU_BUFFER) ? "CPU_BUFFER" : "GPU_BUFFER");
        if (dynPrms.pTbCRCDataIn->pTbInput == nullptr) {
            printf("pTbCRCDataIn.pTbInput:      %p\n", dynPrms.pTbCRCDataIn->pTbInput);
        }
        else {
            printf("pTbCRCDataIn.pTbInput:     {");
            for (int i = 0; i <  cellGrpDynPrm->nCells; i++) {
                if (i != 0) printf(", ");
                printf("%p", dynPrms.pTbCRCDataIn->pTbInput[i]);
            }
            printf("}\n");
        }
    }

    // PDSCH data output buffers
    if (dynPrms.pDataOut == nullptr)
        printf("pDataOut:      %p\n", dynPrms.pDataOut);
    else {
        printf("pDataOut.pTDataTx[i].pAddr:    {"); //TODO could add descriptor info too
        for (int i = 0; i < cellGrpDynPrm->nCells; i++) {
            if (i != 0) printf(", ");
            printf("%p", dynPrms.pDataOut->pTDataTx[i].pAddr);
        }
        printf("}\n");
        printf("\n");
    }

    printPdschCellGrpDynPrms(*cellGrpDynPrm);
    printf("======================================\n\n");
}


void printPdschCellGrpDynPrms(const cuphyPdschCellGrpDynPrm_t& cellGrpDynPrms) {

    // Print information for all cells
    int nCells = cellGrpDynPrms.nCells;
    printf("PDSCH cell group dynamic parameters for %d cells\n", nCells);

    cuphyPdschCellDynPrm_t* pdschCellDynPrms = cellGrpDynPrms.pCellPrms;

    for (int cellId = 0; cellId < nCells; cellId++) {
        printf("  ------------------------------------\n");
        printf("  Cell %d\n", cellId);
        printf("  ------------------------------------\n");

        printf("  cellPrmStatIdx:  %4d\n", pdschCellDynPrms[cellId].cellPrmStatIdx);
        printf("  cellPrmDynIdx:   %4d\n", pdschCellDynPrms[cellId].cellPrmDynIdx);

        printf("  slotNum:         %4d\n", pdschCellDynPrms[cellId].slotNum);
        printf("  nCsiRsPrms:      %4d\n", pdschCellDynPrms[cellId].nCsiRsPrms);
        printf("  csiRsPrmsOffset: %4d\n", pdschCellDynPrms[cellId].csiRsPrmsOffset);
        printf("  testModel:       %4d\n", pdschCellDynPrms[cellId].testModel);

        // NB: pdschStartSym, nPdschSym and dmrsSymLocBmsk may be provided in UE group instead.
        // If so, the values here will be 0.
        printf("  pdschStartSym:   %4d\n", pdschCellDynPrms[cellId].pdschStartSym);
        printf("  nPdschSym:       %4d\n", pdschCellDynPrms[cellId].nPdschSym);
        printf("  dmrsSymLocBmsk:  %4d\n", pdschCellDynPrms[cellId].dmrsSymLocBmsk);
        printf("\n");
    }

    // Print information for all UE groups
    int numUeGrps = cellGrpDynPrms.nUeGrps;
    printf("PDSCH cell group dynamic parameters for %d UE groups\n", numUeGrps);
    cuphyPdschUeGrpPrm_t* ueGrpPrms = cellGrpDynPrms.pUeGrpPrms;

    for (int ueGroupId = 0; ueGroupId < numUeGrps; ueGroupId++) {
        printf("  ------------------------------------\n");
        printf("  UE group %d\n", ueGroupId);
        printf("  ------------------------------------\n");

        printf("  resourceAlloc:   %4d\n", ueGrpPrms[ueGroupId].resourceAlloc);
        uint32_t* rbBitmap = reinterpret_cast<uint32_t*>(ueGrpPrms[ueGroupId].rbBitmap);
        for (int i = 0; i < MAX_RBMASK_UINT32_ELEMENTS && (ueGrpPrms[ueGroupId].resourceAlloc==0); i++) {
            printf(" 0x%X\n",rbBitmap[i]);
        }
        printf("  startPrb:        %4d\n", ueGrpPrms[ueGroupId].startPrb);
        printf("  nPrb:            %4d\n", ueGrpPrms[ueGroupId].nPrb);

        // NB: pdschStartSym, nPdschSym and dmrsSymLocBmsk may be provided in the cell instead, and thus be common across all UE groups.
        // If so, the values here will be 0.
        printf("  pdschStartSym:   %4d\n", ueGrpPrms[ueGroupId].pdschStartSym);
        printf("  nPdschSym:       %4d\n", ueGrpPrms[ueGroupId].nPdschSym);
        printf("  dmrsSymLocBmsk:  %4d\n", ueGrpPrms[ueGroupId].dmrsSymLocBmsk);

        printf("  nUes:            %4d\n", ueGrpPrms[ueGroupId].nUes);

        // Indices to pUePrms array
        printf("  pUePrmIdxs:      {");
        for (int i = 0; i <  ueGrpPrms[ueGroupId].nUes; i++) {
            if (i != 0) printf(", ");
            printf("%d", ueGrpPrms[ueGroupId].pUePrmIdxs[i]);
        }
        printf("}\n");

        // pDmrsDynPrm -> pointer to DMRS info
        printf("  nDmrsCdmGrpsNoData:  %4d\n", ueGrpPrms[ueGroupId].pDmrsDynPrm->nDmrsCdmGrpsNoData);

        // pCellPrm -> pointer to parent group's dynamic params.
        printf("  UE group's parent static cell Idx:    %4d\n", ueGrpPrms[ueGroupId].pCellPrm->cellPrmStatIdx);
        printf("  UE group's parent dynamic cell Idx:   %4d\n", ueGrpPrms[ueGroupId].pCellPrm->cellPrmDynIdx);

        printf("\n");
    }
    printf("\n");

    // Print information for all UEs
    int numUes = cellGrpDynPrms.nUes;
    printf("PDSCH cell group dynamic parameters for %d UEs\n", numUes);

    cuphyPdschUePrm_t* uePrms = cellGrpDynPrms.pUePrms;

    for (int ueId = 0; ueId <  numUes; ueId++) {
        printf("  ------------------------------------\n");
        printf("  UE %d\n", ueId);
        printf("  ------------------------------------\n");

        printf("  scid:         %4d\n", uePrms[ueId].scid);
        printf("  dmrsScrmId:   %4d\n", uePrms[ueId].dmrsScrmId);
        printf("  nUeLayers:    %4d\n", uePrms[ueId].nUeLayers);
        printf("  dmrsPortBmsk: %4d\n", uePrms[ueId].dmrsPortBmsk);
        printf("  refPoint:     %4d\n", uePrms[ueId].refPoint);
        printf("  BWPStart:     %4d\n", uePrms[ueId].BWPStart);
        printf("  beta_drms:     %f\n", uePrms[ueId].beta_dmrs);
        printf("  beta_qam:      %f\n", uePrms[ueId].beta_qam);
        printf("  rnti:         %4d\n", uePrms[ueId].rnti);
        printf("  dataScramId:  %4d\n", uePrms[ueId].dataScramId);
        printf("  nCw:          %4d\n", uePrms[ueId].nCw);

        // Indices to pCwPrms array
        printf("  pCwIdxs: {");
        for (int i = 0; i <  uePrms[ueId].nCw; i++) {
            if (i != 0) printf(", ");
            printf("%d", uePrms[ueId].pCwIdxs[i]);
        }
        printf("}\n");

        // Precoding parameters
        printf("  enablePrcdBf: %4d\n", uePrms[ueId].enablePrcdBf);
        printf("  pmwPrmIdx:    %4d\n", uePrms[ueId].pmwPrmIdx);
        printf("  nlAbove16:    %4d\n", uePrms[ueId].nlAbove16);

        // Add pointer to parent UE group
        bool found = false;
        int ueGroupId = 0;
        while ((!found) && (ueGroupId < numUeGrps)) {
            if (uePrms[ueId].pUeGrpPrm == &ueGrpPrms[ueGroupId]) {
              found = true;
          }
          ueGroupId += 1;
        }
        printf("  Parent UE group: %4d\n", found ? (ueGroupId - 1) : -1);
    }
    printf("\n");

    // Print information for all CWs
    int numCws = cellGrpDynPrms.nCws;
    printf("PDSCH cell group dynamic parameters for %d CWs\n", numCws);

    cuphyPdschCwPrm_t* cwPrms = cellGrpDynPrms.pCwPrms;

    for (int cwId = 0; cwId < numCws; cwId++) {
        printf("  ------------------------------------\n");
        printf("  CW %d\n", cwId);
        printf("  ------------------------------------\n");

        printf("  mcsTableIndex:         %4d\n", cwPrms[cwId].mcsTableIndex);
        printf("  mcsIndex:              %4d\n", cwPrms[cwId].mcsIndex);
        printf("  targetCodeRate:        %4d\n", cwPrms[cwId].targetCodeRate);
        printf("  qamModOrder:           %4d\n", cwPrms[cwId].qamModOrder);
        printf("  rv:                    %4d\n", cwPrms[cwId].rv);
        printf("  tbStartOffset:         %4d\n", cwPrms[cwId].tbStartOffset);
        printf("  tbSize:                %4d\n", cwPrms[cwId].tbSize);
        printf("  n_PRB_LBRM:            %4d\n", cwPrms[cwId].n_PRB_LBRM);
        printf("  maxLayers:             %4d\n", cwPrms[cwId].maxLayers);
        printf("  maxQm:                 %4d\n", cwPrms[cwId].maxQm);

        // FIXME pointer to parent UE
        bool found = false;
        int ueId = 0;
        while ((!found) && (ueId < numUes)) {
            if (cwPrms[cwId].pUePrm == &uePrms[ueId]) {
              found = true;
          }
          ueId += 1;
        }
        printf("  Parent UE:             %4d\n", found ? (ueId - 1): -1);
    }
    printf("\n");

    // Print information for all CSI-RS
    int numCsiRs = cellGrpDynPrms.nCsiRsPrms;
    printf("PDSCH cell group dynamic parameters for %d CSI-RS params\n", numCsiRs);

    _cuphyCsirsRrcDynPrm* csiRsPrms = cellGrpDynPrms.pCsiRsPrms;

    for (int csiRsIdx = 0; csiRsIdx <  numCsiRs; csiRsIdx++) {
        printf("  ------------------------------------\n");
        printf("  CSI RS %d\n", csiRsIdx);
        printf("  ------------------------------------\n");

        printf("  startRb:               %4d\n", csiRsPrms[csiRsIdx].startRb);
        printf("  nRb:                   %4d\n", csiRsPrms[csiRsIdx].nRb);
        printf("  freqDomain:            %4d\n", csiRsPrms[csiRsIdx].freqDomain);
        printf("  row:                   %4d\n", csiRsPrms[csiRsIdx].row);
        printf("  symbL0:                %4d\n", csiRsPrms[csiRsIdx].symbL0);
        printf("  symbL1:                %4d\n", csiRsPrms[csiRsIdx].symbL1);
        printf("  freqDensity:           %4d\n", csiRsPrms[csiRsIdx].freqDensity);
    }
    printf("\n");

    // Print precoding information
    int numPmw = cellGrpDynPrms.nPrecodingMatrices;
    printf("PDSCH cell group dynamic parameters for %d precoding params\n", numPmw);

    cuphyPmW_t* pmwPrms = cellGrpDynPrms.pPmwPrms;
    for (int precodingIdx = 0; precodingIdx < numPmw; precodingIdx++) {
        printf("  ------------------------------------\n");
        printf("  Precoding matrix %d\n", precodingIdx);
        printf("  ------------------------------------\n");

        uint8_t numPorts = pmwPrms[precodingIdx].nPorts;
        printf("  nPorts:                 %4d\n", numPorts);
        printf("  W:\n");
        for (int layerIdx = 0; layerIdx < MAX_DL_LAYERS_PER_TB; layerIdx++) { // Not all layers used
            for (int portIdx = 0; portIdx < numPorts; portIdx++) {
                __half2 val = pmwPrms[precodingIdx].matrix[layerIdx * numPorts + portIdx];
                printf("{%5.5f, %5.5f} ", float(val.x), float(val.y));
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");


}

PdschParams::PdschParams(const py::object& statPrms) {

    const py::list cellStatPrmList = statPrms.attr("cellStatPrms");
    uint16_t nCells = cellStatPrmList.size();
    m_pdschStatPrms.nCells = nCells;

    m_pdschStatPrms.read_TB_CRC          = statPrms.attr("read_TB_CRC").cast<bool>();
    m_pdschStatPrms.full_slot_processing = statPrms.attr("full_slot_processing").cast<bool>();
    m_pdschStatPrms.stream_priority      = statPrms.attr("stream_priority").cast<int>();
    m_pdschStatPrms.nMaxCellsPerSlot     = statPrms.attr("nMaxCellsPerSlot").cast<uint16_t>();
    m_pdschStatPrms.nMaxUesPerCellGroup  = statPrms.attr("nMaxUesPerCellGroup").cast<uint16_t>();
    m_pdschStatPrms.nMaxCBsPerTB         = statPrms.attr("nMaxCBsPerTB").cast<uint16_t>();
    m_pdschStatPrms.nMaxPrb              = statPrms.attr("nMaxPrb").cast<uint16_t>();

    m_dbgPrm.resize(nCells);
    m_dbgFilenames.resize(nCells);
    m_cellStatPrms.resize(nCells);

    for (int cellIdx = 0; cellIdx < nCells; cellIdx++) {
        const py::object cellStatPrm = cellStatPrmList[cellIdx];

        m_cellStatPrms[cellIdx].phyCellId    = cellStatPrm.attr("phyCellId").cast<uint16_t>();
        m_cellStatPrms[cellIdx].nRxAnt       = cellStatPrm.attr("nRxAnt").cast<uint16_t>();
        m_cellStatPrms[cellIdx].nTxAnt       = cellStatPrm.attr("nTxAnt").cast<uint16_t>();
        m_cellStatPrms[cellIdx].nPrbUlBwp    = cellStatPrm.attr("nPrbUlBwp").cast<uint16_t>();
        m_cellStatPrms[cellIdx].nPrbDlBwp    = cellStatPrm.attr("nPrbDlBwp").cast<uint16_t>();
        m_cellStatPrms[cellIdx].mu           = cellStatPrm.attr("mu").cast<uint8_t>();
        if (m_cellStatPrms[cellIdx].mu > 1) {
            throw std::runtime_error("Unsupported numerology value!");
        }
    }

    if (!(std::string(py::str(statPrms.attr("dbg"))) == "None")) {
        const py::list dbg = statPrms.attr("dbg");

        for (int cellIdx = 0; cellIdx < nCells; cellIdx++) {
            const py::object dbgPrms = dbg[cellIdx];
            m_dbgFilenames[cellIdx] = std::string(py::str(dbgPrms.attr("cfgFilename")));
            if ((m_dbgFilenames[cellIdx] == "None") || (m_dbgFilenames[cellIdx] == ""))
                m_dbgPrm[cellIdx].pCfgFileName = "";
            else
                m_dbgPrm[cellIdx].pCfgFileName = m_dbgFilenames[cellIdx].c_str();
            m_dbgPrm[cellIdx].refCheck = dbgPrms.attr("refCheck").cast<bool>();
            m_dbgPrm[cellIdx].checkTbSize = dbgPrms.attr("checkTbSize").cast<uint8_t>();
            m_dbgPrm[cellIdx].cfgIdenticalLdpcEncCfgs = dbgPrms.attr("cfgIdenticalLdpcEncCfgs").cast<bool>();
        }

    }
    else {
        for (int cellIdx = 0; cellIdx < nCells; cellIdx++) {
             // If nullptr, it will cause pipeline broken by
             // hdf5_filename = static_params->pDbg->pCfgFileName in pdsch_tx.cpp.
            m_dbgPrm[cellIdx].pCfgFileName = "";
            m_dbgPrm[cellIdx].checkTbSize = 0;
            m_dbgPrm[cellIdx].refCheck = false;
            m_dbgPrm[cellIdx].cfgIdenticalLdpcEncCfgs = 1;
        }
    }

    m_tracker.pMemoryFootprint = nullptr;

    m_pdschStatPrms.pCellStatPrms = m_cellStatPrms.data();
    m_pdschStatPrms.pDbg = m_dbgPrm.data();
    m_pdschStatPrms.pOutInfo = &m_tracker;

    // Allocate memory for dynamic parameters. Note: Only one cell group here.
    m_cellDynPrms.resize(nCells);
    m_cellMetrics.resize(nCells);
    m_ueGrpPrms.resize(PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP);
    m_uePrms.resize(PDSCH_MAX_UES_PER_CELL_GROUP);
    m_cwPrms.resize(PDSCH_MAX_CWS_PER_CELL_GROUP);
    m_pmwPrms.resize(PDSCH_MAX_UES_PER_CELL_GROUP);
    m_pdschDmrsPrms.resize(PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP);

    m_pdschDynPrms.pCellGrpDynPrm = &m_cellGrpDynPrms;
    m_cellGrpDynPrms.pCellPrms  = m_cellDynPrms.data();
    m_cellGrpDynPrms.pCellMetrics  = m_cellMetrics.data();
    m_cellGrpDynPrms.pUeGrpPrms = m_ueGrpPrms.data();
    m_cellGrpDynPrms.pUePrms    = m_uePrms.data();
    m_cellGrpDynPrms.pCwPrms    = m_cwPrms.data();
    m_cellGrpDynPrms.pPmwPrms   = m_pmwPrms.data();

    for (int i = 0; i < PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP; i++) {
        m_cellGrpDynPrms.pUeGrpPrms[i].rbBitmap = new uint8_t[MAX_RBMASK_BYTE_SIZE];
        m_cellGrpDynPrms.pUeGrpPrms[i].pUePrmIdxs = new uint16_t[PDSCH_MAX_UES_PER_CELL_GROUP];
        m_cellGrpDynPrms.pUeGrpPrms[i].pDmrsDynPrm = &m_pdschDmrsPrms[i];
    }

    for (int i = 0; i < PDSCH_MAX_UES_PER_CELL_GROUP; i++) {
        m_cellGrpDynPrms.pUePrms[i].pCwIdxs = new uint16_t[2];
    }

    m_dataTxTensor.resize(nCells);
    m_tbInputPtr.resize(nCells);
    m_outputStatus.resize(1);
    m_outputTensorPrm.resize(nCells);

    m_pdschDynPrms.pDataOut           = &m_outputData;
    m_pdschDynPrms.pDataOut->pTDataTx = m_outputTensorPrm.data();
    m_pdschDynPrms.pStatusInfo        = m_outputStatus.data();

}


PdschParams::~PdschParams() {
    for (int i = 0; i < PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP; i++){
        delete[] m_cellGrpDynPrms.pUeGrpPrms[i].rbBitmap;
        delete[] m_cellGrpDynPrms.pUeGrpPrms[i].pUePrmIdxs;
    }

    for (int i = 0; i < PDSCH_MAX_UES_PER_CELL_GROUP; i++){
        delete[] m_cellGrpDynPrms.pUePrms[i].pCwIdxs;
    }
}


void PdschParams::setDynPrms(const py::object& dynPrms) {

    cudaStream_t cuStream               = reinterpret_cast<cudaStream_t>(dynPrms.attr("cuStream").cast<uint64_t>());
    m_pdschDynPrms.cuStream             = cuStream;
    m_pdschDynPrms.procModeBmsk         = dynPrms.attr("procModeBmsk").cast<uint64_t>();

    const py::object cellGrpDynPrm      = dynPrms.attr("cellGrpDynPrm");
    const py::object dataIn             = dynPrms.attr("dataIn");
    const py::object dataOut            = dynPrms.attr("dataOut");
    const py::list cellPrmsList         = cellGrpDynPrm.attr("cellPrms");
    const py::list ueGrpPrmsList        = cellGrpDynPrm.attr("ueGrpPrms");
    const py::list uePrmsList           = cellGrpDynPrm.attr("uePrms");
    const py::list cwPrmsList           = cellGrpDynPrm.attr("cwPrms");

    py::list csiRsPrmsList;
    if (!(std::string(py::str(cellGrpDynPrm.attr("csiRsPrms"))) == "None")) {
        csiRsPrmsList = cellGrpDynPrm.attr("csiRsPrms");
    }

    py::list pmwPrmsList;
    if (!(std::string(py::str(cellGrpDynPrm.attr("pmwPrms"))) == "None")) {
        pmwPrmsList = cellGrpDynPrm.attr("pmwPrms");
    }

    // Cell dynamic parameters.
    uint16_t nCells = cellPrmsList.size();
    m_cellGrpDynPrms.nCells = nCells;
    for (int cellIdx = 0; cellIdx < nCells; cellIdx++) {
        const py::object cellDynPrms = cellPrmsList[cellIdx];

        m_cellGrpDynPrms.pCellPrms[cellIdx].nCsiRsPrms      = cellDynPrms.attr("nCsiRsPrms").cast<uint16_t>();
        m_cellGrpDynPrms.pCellPrms[cellIdx].csiRsPrmsOffset = cellDynPrms.attr("csiRsPrmsOffset").cast<uint16_t>();
        m_cellGrpDynPrms.pCellPrms[cellIdx].cellPrmStatIdx  = cellDynPrms.attr("cellPrmStatIdx").cast<uint16_t>();
        m_cellGrpDynPrms.pCellPrms[cellIdx].cellPrmDynIdx   = cellDynPrms.attr("cellPrmDynIdx").cast<uint16_t>();
        m_cellGrpDynPrms.pCellPrms[cellIdx].slotNum         = cellDynPrms.attr("slotNum").cast<uint16_t>();
        m_cellGrpDynPrms.pCellPrms[cellIdx].pdschStartSym   = cellDynPrms.attr("pdschStartSym").cast<uint8_t>();
        m_cellGrpDynPrms.pCellPrms[cellIdx].nPdschSym       = cellDynPrms.attr("nPdschSym").cast<uint8_t>();
        m_cellGrpDynPrms.pCellPrms[cellIdx].dmrsSymLocBmsk  = cellDynPrms.attr("dmrsSymLocBmsk").cast<uint16_t>();
        m_cellGrpDynPrms.pCellPrms[cellIdx].testModel       = cellDynPrms.attr("testModel").cast<uint8_t>();
    }

    // UE group parameters.
    uint16_t nUeGrps = ueGrpPrmsList.size();
    m_cellGrpDynPrms.nUeGrps = nUeGrps;
    for (int ueGroupIdx = 0; ueGroupIdx < nUeGrps; ueGroupIdx++) {
        const py::object ueGrpPrms = ueGrpPrmsList[ueGroupIdx];

        int cellPrmIdx = ueGrpPrms.attr("cellPrmIdx").cast<int>();
        m_cellGrpDynPrms.pUeGrpPrms[ueGroupIdx].pCellPrm = &m_cellGrpDynPrms.pCellPrms[cellPrmIdx];
        m_cellGrpDynPrms.pUeGrpPrms[ueGroupIdx].resourceAlloc = ueGrpPrms.attr("resourceAlloc").cast<uint8_t>();
        const py::list rbBitmap = ueGrpPrms.attr("rbBitmap");
        for (int idx = 0; idx < rbBitmap.size(); idx++) {
            m_cellGrpDynPrms.pUeGrpPrms[ueGroupIdx].rbBitmap[idx] = rbBitmap[idx].cast<uint8_t>();
        }
        m_cellGrpDynPrms.pUeGrpPrms[ueGroupIdx].startPrb = ueGrpPrms.attr("startPrb").cast<uint16_t>();
        m_cellGrpDynPrms.pUeGrpPrms[ueGroupIdx].nPrb = ueGrpPrms.attr("nPrb").cast<uint16_t>();
        m_cellGrpDynPrms.pUeGrpPrms[ueGroupIdx].dmrsSymLocBmsk = ueGrpPrms.attr("dmrsSymLocBmsk").cast<uint16_t>();
        m_cellGrpDynPrms.pUeGrpPrms[ueGroupIdx].pdschStartSym = ueGrpPrms.attr("pdschStartSym").cast<uint8_t>();
        m_cellGrpDynPrms.pUeGrpPrms[ueGroupIdx].nPdschSym = ueGrpPrms.attr("nPdschSym").cast<uint8_t>();

        py::list uePrmIdxs = ueGrpPrms.attr("uePrmIdxs");
        uint16_t nUes = uePrmIdxs.size();
        m_cellGrpDynPrms.pUeGrpPrms[ueGroupIdx].nUes = nUes;
        for (int idx = 0; idx < nUes; idx++) {
            m_cellGrpDynPrms.pUeGrpPrms[ueGroupIdx].pUePrmIdxs[idx] = uePrmIdxs[idx].cast<uint16_t>();
        }

        cuphyPdschDmrsPrm_t* pDmrsDynPrm = m_cellGrpDynPrms.pUeGrpPrms[ueGroupIdx].pDmrsDynPrm;
        pDmrsDynPrm->nDmrsCdmGrpsNoData = ueGrpPrms.attr("nDmrsCdmGrpsNoData").cast<uint8_t>();
    }

    // UE parameters.
    uint16_t nUes = uePrmsList.size();
    m_cellGrpDynPrms.nUes = nUes;
    for (int ueIdx = 0; ueIdx < nUes; ueIdx++) {
        const py::object uePrms = uePrmsList[ueIdx];

        int ueGrpPrmIdx                                    = uePrms.attr("ueGrpPrmIdx").cast<int>();
        m_cellGrpDynPrms.pUePrms[ueIdx].pUeGrpPrm          = &m_cellGrpDynPrms.pUeGrpPrms[ueGrpPrmIdx];
        m_cellGrpDynPrms.pUePrms[ueIdx].scid               = uePrms.attr("scid").cast<uint8_t>();
        m_cellGrpDynPrms.pUePrms[ueIdx].dmrsScrmId         = uePrms.attr("dmrsScrmId").cast<uint16_t>();
        m_cellGrpDynPrms.pUePrms[ueIdx].nUeLayers          = uePrms.attr("nUeLayers").cast<uint8_t>();
        m_cellGrpDynPrms.pUePrms[ueIdx].dmrsPortBmsk       = uePrms.attr("dmrsPortBmsk").cast<uint16_t>();
        m_cellGrpDynPrms.pUePrms[ueIdx].BWPStart           = uePrms.attr("BWPStart").cast<uint16_t>();
        m_cellGrpDynPrms.pUePrms[ueIdx].refPoint           = uePrms.attr("refPoint").cast<uint8_t>();
        m_cellGrpDynPrms.pUePrms[ueIdx].beta_dmrs          = uePrms.attr("beta_dmrs").cast<float>();
        m_cellGrpDynPrms.pUePrms[ueIdx].beta_qam           = uePrms.attr("beta_qam").cast<float>();
        m_cellGrpDynPrms.pUePrms[ueIdx].rnti               = uePrms.attr("rnti").cast<uint16_t>();
        m_cellGrpDynPrms.pUePrms[ueIdx].dataScramId        = uePrms.attr("dataScramId").cast<uint16_t>();
        m_cellGrpDynPrms.pUePrms[ueIdx].enablePrcdBf       = uePrms.attr("enablePrcdBf").cast<uint8_t>();

        if (m_cellGrpDynPrms.pUePrms[ueIdx].enablePrcdBf)
            m_cellGrpDynPrms.pUePrms[ueIdx].pmwPrmIdx      = uePrms.attr("pmwPrmIdx").cast<uint16_t>();
        else
            m_cellGrpDynPrms.pUePrms[ueIdx].pmwPrmIdx      = 0;

        m_cellGrpDynPrms.pUePrms[ueIdx].nlAbove16          = 0; // Not supported 32 layer PDSCH yet

        py::list cwIdxs                                    = uePrms.attr("cwIdxs");
        uint8_t nCw                                        = cwIdxs.size();
        m_cellGrpDynPrms.pUePrms[ueIdx].nCw                = nCw;
        for (int cwIdx = 0; cwIdx < nCw; cwIdx++ ){
            m_cellGrpDynPrms.pUePrms[ueIdx].pCwIdxs[cwIdx] = cwIdxs[cwIdx].cast<uint16_t>();
        }
    }


    // Codeword parameters.
    uint16_t nCws = cwPrmsList.size();
    m_cellGrpDynPrms.nCws = nCws;
    for (int cwIdx = 0; cwIdx < nCws; cwIdx++) {
        const py::object cwPrms = cwPrmsList[cwIdx];
        int uePrmIdx                                      = cwPrms.attr("uePrmIdx").cast<int>();
        m_cellGrpDynPrms.pCwPrms[cwIdx].pUePrm            = &m_cellGrpDynPrms.pUePrms[uePrmIdx];
        m_cellGrpDynPrms.pCwPrms[cwIdx].mcsTableIndex     = cwPrms.attr("mcsTableIndex").cast<uint8_t>();
        m_cellGrpDynPrms.pCwPrms[cwIdx].mcsIndex          = cwPrms.attr("mcsIndex").cast<uint8_t>();
        m_cellGrpDynPrms.pCwPrms[cwIdx].targetCodeRate    = cwPrms.attr("targetCodeRate").cast<uint16_t>();
        m_cellGrpDynPrms.pCwPrms[cwIdx].qamModOrder       = cwPrms.attr("qamModOrder").cast<uint8_t>();
        m_cellGrpDynPrms.pCwPrms[cwIdx].rv                = cwPrms.attr("rv").cast<uint8_t>();
        m_cellGrpDynPrms.pCwPrms[cwIdx].tbStartOffset     = cwPrms.attr("tbStartOffset").cast<uint32_t>();
        m_cellGrpDynPrms.pCwPrms[cwIdx].tbSize            = cwPrms.attr("tbSize").cast<uint32_t>();
        m_cellGrpDynPrms.pCwPrms[cwIdx].n_PRB_LBRM        = cwPrms.attr("n_PRB_LBRM").cast<uint16_t>();
        m_cellGrpDynPrms.pCwPrms[cwIdx].maxLayers         = cwPrms.attr("maxLayers").cast<uint8_t>();
        m_cellGrpDynPrms.pCwPrms[cwIdx].maxQm             = cwPrms.attr("maxQm").cast<uint8_t>();
    }

    // CSI-RS parameters.
    uint16_t nCsiRsPrms = csiRsPrmsList.size();
    m_csirsPrms.resize(nCsiRsPrms);
    for(int csiRsPrmIdx = 0; csiRsPrmIdx < nCsiRsPrms; csiRsPrmIdx++) {
        const py::object& pyCsiRsRrcDynPrms = csiRsPrmsList[csiRsPrmIdx];
        readCsiRsRrcDynPrms(pyCsiRsRrcDynPrms, m_csirsPrms[csiRsPrmIdx]);
    }
    m_cellGrpDynPrms.nCsiRsPrms = nCsiRsPrms;
    m_cellGrpDynPrms.pCsiRsPrms = m_csirsPrms.data();

    // Precoding parameters.
    uint16_t nPrecodingMatrices = pmwPrmsList.size();
    m_cellGrpDynPrms.nPrecodingMatrices = nPrecodingMatrices;
    if (nPrecodingMatrices > 0) {
        for (int pmwIdx = 0; pmwIdx < nPrecodingMatrices; pmwIdx++) {
            py::object pmwPrms = pmwPrmsList[pmwIdx];
            py::array temp = pmwPrms.attr("w");
            py::array_t<std::complex<float>> pmwArray = temp;
            readPrecodingMatrix(pmwArray, m_cellGrpDynPrms.pPmwPrms[pmwIdx].matrix, m_cellGrpDynPrms.pPmwPrms[pmwIdx].nPorts);
        }
    }

    // Data input.
    py::list tbInputList = dataIn.attr("tbInput");
    if( (uint16_t)tbInputList.size() != nCells) {
        throw std::runtime_error("Cell number for tbInput does not match cellPrms!");
    }

    for (int cellIdx = 0; cellIdx < nCells; cellIdx++) {
        const cuda_array_t<uint8_t>& tbInput = tbInputList[cellIdx].cast<cuda_array_t<uint8_t>>();
        m_tbInputPtr[cellIdx] = static_cast<uint8_t*>(tbInput.get_device_ptr());
    }
    m_dataIn = {m_tbInputPtr.data(), cuphyPdschDataIn_t::GPU_BUFFER};
    m_pdschDynPrms.pDataIn = &m_dataIn;

    // Optional TB CRC data input.
    py::object pyTbCrcDataIn;
    if (!(std::string(py::str(dynPrms.attr("tbCRCDataIn"))) == "None")) {
        pyTbCrcDataIn = dynPrms.attr("tbCRCDataIn");

        py::list tbCrcInput = pyTbCrcDataIn.attr("tbInput");
        if( (uint16_t)tbCrcInput.size() != nCells) {
            throw std::runtime_error("Cell number for tbCRCInput does not match cellPrms!");
        }
        std::vector<uint8_t*> tbCrcInputPtr(nCells);

        for (int cellIdx = 0; cellIdx < nCells; cellIdx ++) {
            const cuda_array_t<uint8_t>& tbCrcInputBuff = tbCrcInput[cellIdx].cast<cuda_array_t<uint8_t>>();
            tbCrcInputPtr[cellIdx] = static_cast<uint8_t*>(tbCrcInputBuff.get_device_ptr());
        }

        m_tbCrcDataIn               = {tbCrcInputPtr.data(), cuphyPdschDataIn_t::GPU_BUFFER};
        m_pdschDynPrms.pTbCRCDataIn = &m_tbCrcDataIn;
    }
    else {
        m_tbCrcDataIn               = {nullptr, cuphyPdschDataIn_t::GPU_BUFFER};
        m_pdschDynPrms.pTbCRCDataIn = &m_tbCrcDataIn;
    }


    // Data output.
    py::list dataTx = dataOut.attr("dataTx");
    if( (uint16_t)dataTx.size() != nCells) {
        throw std::runtime_error("Cell number for dataTx does not match cellPrms!");
    }

    for (int cellIdx = 0; cellIdx < nCells; cellIdx++) {
        py::object tensorDesc = dataTx[cellIdx];
        py::list dimensions = tensorDesc.attr("dimensions");
        if (dimensions.size() != 3) {
            throw std::runtime_error("DataOut tensor is not 3 dimensions!");
        }

        int dim0 = dimensions[0].cast<int>();
        int dim1 = dimensions[1].cast<int>();
        int dim2 = dimensions[2].cast<int>();
        uint64_t pAddr = tensorDesc.attr("pAddr").cast<uint64_t>();
        cuphyDataType_t type = tensorDesc.attr("dataType").cast<cuphyDataType_t>();
        if (type != CUPHY_C_32F) {  // CUPHY_C_16F not supported with the Python interface.
            throw std::runtime_error("DataOut type should be CUPHY_C_32F!");
        }
        m_dataTxTensor[cellIdx] = cuphy::tensor_device((void*)pAddr, type, dim0, dim1, dim2, cuphy::tensor_flags::align_tight);
        m_pdschDynPrms.pDataOut->pTDataTx[cellIdx].desc = m_dataTxTensor[cellIdx].desc().handle();
        m_pdschDynPrms.pDataOut->pTDataTx[cellIdx].pAddr = m_dataTxTensor[cellIdx].addr();
    }
}


void PdschParams::printStatPrms() const {
    printPdschStatPrms(m_pdschStatPrms);
}


void PdschParams::printDynPrms() const {
    printPdschDynPrms(m_pdschDynPrms);
}

}