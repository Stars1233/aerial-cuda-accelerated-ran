/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nv_simulate_phy_driver.hpp"
#include "nvlog.hpp"

#define TAG_PARAM (NVLOG_TAG_BASE_L2_ADAPTER + 9) // "L2A.PARAM"
#define TAG (NVLOG_TAG_BASE_L2_ADAPTER + 10)      // "L2A.SIM"

#define getName(var) #var

#define PUSCH_PRINT_PARAMS

using namespace std;

uint8_t  SimulatePhyDriver::zero_u8[1000L * 1000];
uint16_t SimulatePhyDriver::zero_u16[1000L * 1000];
uint32_t SimulatePhyDriver::zero_u32[1000L * 1000];
float    SimulatePhyDriver::zero_float[1000L * 1000];

void printPuschCellGroupParams(slot_command_api::pusch_params* pusch)
{
#ifdef PUSCH_PRINT_PARAMS
    if(pusch == nullptr)
    {
        NVLOGI_FMT(TAG_PARAM, "pusch params is null");
        return;
    }

    NVLOGI_FMT(TAG_PARAM, "PUSCH param nCells: {}", pusch->cell_index_list.size());
    for(int i = 0; i < pusch->cell_index_list.size(); ++i)
    {
        NVLOGI_FMT(TAG_PARAM, "{} Cell Index: {}, PhyCellId: {}", i, pusch->cell_index_list[i], pusch->phy_cell_index_list[i]);
    }
#endif
}

void printParameters(cuphyPuschCellGrpDynPrm_t* l2)
{
#ifdef PUSCH_PRINT_PARAMS
    NVLOGD_FMT(TAG_PARAM, "\n===============================================");
    if(l2 == nullptr)
    {
        NVLOGI_FMT(TAG_PARAM, "L2 params is null");
        return;
    }

    NVLOGI_FMT(TAG_PARAM, "{} L2: nUeGrps: {} nCells: {} nUes: {}", __FUNCTION__, l2->nUeGrps, l2->nCells, l2->nUes);
    for(uint16_t i = 0; i < l2->nCells; i++)
    {
        cuphyPuschCellDynPrm_t* dyn = &l2->pCellPrms[i];
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: pCellPrms:{}", __FUNCTION__, reinterpret_cast<void*>(dyn));
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: cellPrmStatIdx:{}", __FUNCTION__, dyn->cellPrmStatIdx);
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: cellPrmDynIdx:{}", __FUNCTION__, dyn->cellPrmDynIdx);
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: slotNum:{}", __FUNCTION__, dyn->slotNum);
    }

    for(uint16_t i = 0; i < l2->nUeGrps; i++)
    {
        cuphyPuschUeGrpPrm_t* dyn = &l2->pUeGrpPrms[i];
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: pUeGrpPrms:{}", __FUNCTION__, reinterpret_cast<void*>(dyn));
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: cuphyPuschUeGrpPrm->pCellPrms:{}", __FUNCTION__, reinterpret_cast<void*>(dyn->pCellPrm));
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: cuphyPuschUeGrpPrm->pDmrsDynPrm:{}", __FUNCTION__, reinterpret_cast<void*>(dyn->pDmrsDynPrm));
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: startPrb:{}", __FUNCTION__, dyn->startPrb);
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: nPrb:{}", __FUNCTION__, dyn->nPrb);
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: nUes:{}", __FUNCTION__, dyn->nUes);

        for(uint16_t j = 0; j < dyn->nUes; j++)
        {
            NVLOGD_FMT(TAG_PARAM, "{} \t\tL2: pUePrmIdxs:{}", __FUNCTION__, dyn->pUePrmIdxs[j]);
        }
        cuphyPuschDmrsPrm_t* dmrs = dyn->pDmrsDynPrm;
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: dmrsAddlnPos:{}", __FUNCTION__, dmrs->dmrsAddlnPos);
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: dmrsMaxLen:{}", __FUNCTION__, dmrs->dmrsMaxLen);
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: nDmrsCdmGrpsNoData:{}", __FUNCTION__, dmrs->nDmrsCdmGrpsNoData);
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: dmrsScrmId:{}", __FUNCTION__, dmrs->dmrsScrmId);
    }

    for(uint16_t i = 0; i < l2->nUes; i++)
    {
        cuphyPuschUePrm_t* dyn = &l2->pUePrms[i];
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: pUePrms:{}", __FUNCTION__, reinterpret_cast<void*>(dyn));
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: cuphyPuschUePrm->pUeGrpPrm:{}", __FUNCTION__, reinterpret_cast<void*>(dyn->pUeGrpPrm));
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: ueGrpIdx:{}", __FUNCTION__, dyn->ueGrpIdx);
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: scid:{}", __FUNCTION__, dyn->scid);
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: dmrsPortBmsk:{}", __FUNCTION__, dyn->dmrsPortBmsk);
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: mcsTableIndex:{}", __FUNCTION__, dyn->mcsTableIndex);
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: rv:{}", __FUNCTION__, dyn->rv);
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: rnti:{}", __FUNCTION__, dyn->rnti);
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: dataScramId:{}", __FUNCTION__, dyn->dataScramId);
        NVLOGD_FMT(TAG_PARAM, "{} \tL2: nUeLayers:{}", __FUNCTION__, dyn->nUeLayers);

        if(dyn->pUciPrms != nullptr)
        {
            cuphyUciOnPuschPrm_t* uci = dyn->pUciPrms;
            NVLOGD_FMT(TAG_PARAM, "{} \tL2: pUciPrms:{}", __FUNCTION__, reinterpret_cast<void*>(uci));
            NVLOGD_FMT(TAG_PARAM, "{} \tL2: nBitsHarq:{}", __FUNCTION__, uci->nBitsHarq);
            NVLOGD_FMT(TAG_PARAM, "{} \tL2: nBitsCsi1:{}", __FUNCTION__, uci->nBitsCsi1);
            NVLOGD_FMT(TAG_PARAM, "{} \tL2: alphaScaling:{}", __FUNCTION__, uci->alphaScaling);
            NVLOGD_FMT(TAG_PARAM, "{} \tL2: betaOffsetHarqAck:{}", __FUNCTION__, uci->betaOffsetHarqAck);
            NVLOGD_FMT(TAG_PARAM, "{} \tL2: betaOffsetCsi1:{}", __FUNCTION__, uci->betaOffsetCsi1);
        }
    }
    NVLOGD_FMT(TAG_PARAM, "===============================================");
#endif
}

void printParameters(const cuphyPdschCellGrpDynPrm_t* l2)
{
#if 1
    if(l2 == nullptr)
    {
        NVLOGD_FMT(TAG, "L2 params is null");
        return;
    }

    NVLOGD_FMT(TAG, "\n===============================================");

    NVLOGD_FMT(TAG, "{}  L2: nCells: {}", __FUNCTION__, l2->nCells);
    for(uint16_t i = 0; i < l2->nCells; i++)
    {
        cuphyPdschCellDynPrm_t* dyn = &l2->pCellPrms[i];
        NVLOGD_FMT(TAG, "{} L2: cellPrmStatIdx:{}", __FUNCTION__, dyn->cellPrmStatIdx);
        NVLOGD_FMT(TAG, "{} L2: cellPrmDynIdx:{}", __FUNCTION__, dyn->cellPrmDynIdx);
        NVLOGD_FMT(TAG, "{} L2: slotNum:{}", __FUNCTION__, dyn->slotNum);
        NVLOGD_FMT(TAG, "{} L2: pdschStartSym:{}", __FUNCTION__, dyn->pdschStartSym);
        NVLOGD_FMT(TAG, "{} L2: nPdschSym:{}", __FUNCTION__, dyn->nPdschSym);
        NVLOGD_FMT(TAG, "{} L2: dmrsSymLocBmsk:{}", __FUNCTION__, dyn->dmrsSymLocBmsk);
    }

    NVLOGD_FMT(TAG, "{} L2: nUeGrps: {}", __FUNCTION__, l2->nUeGrps);

    for(uint16_t i = 0; i < l2->nUeGrps; i++)
    {
        cuphyPdschUeGrpPrm_t* dyn = &l2->pUeGrpPrms[i];
        NVLOGD_FMT(TAG, "{} L2: startPrb:{}", __FUNCTION__, dyn->startPrb);
        NVLOGD_FMT(TAG, "{} L2: nPrb:{}", __FUNCTION__, dyn->nPrb);
        NVLOGD_FMT(TAG, "{} L2: nUes:{}", __FUNCTION__, dyn->nUes);

        for(uint16_t j = 0; j < dyn->nUes; j++)
        {
            NVLOGD_FMT(TAG, "{} L2: pUePrmIdxs:{}", __FUNCTION__, dyn->pUePrmIdxs[j]);
        }
        cuphyPdschDmrsPrm_t* dmrs = dyn->pDmrsDynPrm;
        NVLOGD_FMT(TAG, "{} L2: nDmrsCdmGrpsNoData:{}", __FUNCTION__, dmrs->nDmrsCdmGrpsNoData);
    }

    NVLOGD_FMT(TAG, "{} L2: nUes: {}", __FUNCTION__, l2->nUes);
    for(uint16_t i = 0; i < l2->nUes; i++)
    {
        cuphyPdschUePrm_t* dyn = &l2->pUePrms[i];
        NVLOGD_FMT(TAG, "{} L2: scid:{}", __FUNCTION__, dyn->scid);
        NVLOGD_FMT(TAG, "{} L2: dmrsScrmId:{}", __FUNCTION__, dyn->dmrsScrmId);
        NVLOGD_FMT(TAG, "{} L2: nUeLayers:{}", __FUNCTION__, dyn->nUeLayers);
        NVLOGD_FMT(TAG, "{} L2: dmrsPortBmsk:{}", __FUNCTION__, dyn->dmrsPortBmsk);
        NVLOGD_FMT(TAG, "{} L2: rnti:{}", __FUNCTION__, dyn->rnti);
        NVLOGD_FMT(TAG, "{} L2: dataScramId:{}", __FUNCTION__, dyn->dataScramId);
        NVLOGD_FMT(TAG, "{} L2: nCw:{}", __FUNCTION__, dyn->nCw);
    }

    for(uint16_t i = 0; i < l2->nCws; i++)
    {
        cuphyPdschCwPrm_t* dyn = &l2->pCwPrms[i];
        NVLOGD_FMT(TAG, "{} L2: mcsTableIndex:{}", __FUNCTION__, dyn->mcsTableIndex);
        NVLOGD_FMT(TAG, "{} L2: mcsIndex:{}", __FUNCTION__, dyn->mcsIndex);
        NVLOGD_FMT(TAG, "{} L2: targetCodeRate:{}", __FUNCTION__, dyn->targetCodeRate);
        NVLOGD_FMT(TAG, "{} L2: qamModOrder:{}", __FUNCTION__, dyn->qamModOrder);
        NVLOGD_FMT(TAG, "{} L2: rv:{}", __FUNCTION__, dyn->rv);
        NVLOGD_FMT(TAG, "{} L2: tbStartOffset:{}", __FUNCTION__, dyn->tbStartOffset);
        NVLOGD_FMT(TAG, "{} L2: tbSize:{}", __FUNCTION__, dyn->tbSize);
    }
    NVLOGD_FMT(TAG, "===============================================");
#endif
}

void printParameters(const cuphyCsirsRrcDynPrm_t* l2)
{
    NVLOGD_FMT(TAG_PARAM, "{} L2: {:>25s}:{:10d}", __FUNCTION__, getName(l2->startRb), l2->startRb);
    NVLOGD_FMT(TAG_PARAM, "{} L2: {:>25s}:{:10d}", __FUNCTION__, getName(l2->nRb), l2->nRb);
    NVLOGD_FMT(TAG_PARAM, "{} L2: {:>25s}:{:10d}", __FUNCTION__, getName(l2->freqDomain), l2->freqDomain);
    NVLOGD_FMT(TAG_PARAM, "{} L2: {:>25s}:{:10d}", __FUNCTION__, getName(l2->row), l2->row);
    NVLOGD_FMT(TAG_PARAM, "{} L2: {:>25s}:{:10d}", __FUNCTION__, getName(l2->symbL0), l2->symbL0);
    NVLOGD_FMT(TAG_PARAM, "{} L2: {:>25s}:{:10d}", __FUNCTION__, getName(l2->symbL1), l2->symbL1);
    NVLOGD_FMT(TAG_PARAM, "{} L2: {:>25s}:{:10d}", __FUNCTION__, getName(l2->freqDensity), l2->freqDensity);
    NVLOGD_FMT(TAG_PARAM, "{} L2: {:>25s}:{:10d}", __FUNCTION__, getName(l2->scrambId), l2->scrambId);
    NVLOGD_FMT(TAG_PARAM, "{} L2: {:>25s}:{:10d}", __FUNCTION__, getName(l2->idxSlotInFrame), l2->idxSlotInFrame);
    NVLOGD_FMT(TAG_PARAM, "{} L2: {:>25s}:{:10d}", __FUNCTION__, getName(l2->csiType), +l2->csiType);
    NVLOGD_FMT(TAG_PARAM, "{} L2: {:>25s}:{:10d}", __FUNCTION__, getName(l2->cdmType), +l2->cdmType);
    NVLOGD_FMT(TAG_PARAM, "{} L2: {:>25s}:{:10f}", __FUNCTION__, getName(l2->beta), l2->beta);
}

void printParameters(slot_command_api::pucch_params* pucch_params, cuphyPucchDataOut_t* PucchDataOut)
{
    NVLOGI_FMT(TAG_PARAM, "{} \tL2:PUCCH params no of cells :{}", __FUNCTION__, pucch_params->grp_dyn_pars.nCells);
    uint32_t uci_offset = 0;
    bool     first_uci  = true;
    for(int i = 0; i < pucch_params->grp_dyn_pars.nCells; i++)
    {
        NVLOGI_FMT(TAG_PARAM, "{} \tL2:PUCCH params cellPrmDynIdx: {} cell_id :{} PHY_cell_id {}", __FUNCTION__, i, pucch_params->cell_index_list[i], pucch_params->phy_cell_index_list[i]);
        for(uint16_t f = 0; f < 5; f++)
        {
            switch(f)
            {
            case 0: {
                for(uint16_t n = 0; n < pucch_params->grp_dyn_pars.nF0Ucis; n++)
                {
                    cuphyPucchUciPrm_t& uciParam = pucch_params->grp_dyn_pars.pF0UciPrms[n];
                    if(i != uciParam.cellPrmDynIdx)
                    {
                        continue;
                    }
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params cellPrmDynIdx: {} ", __FUNCTION__, uciParam.cellPrmDynIdx);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params uciOutputIdx: {} ", __FUNCTION__, uciParam.uciOutputIdx);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params formatType: {} ", __FUNCTION__, uciParam.formatType);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params rnti: {} ", __FUNCTION__, uciParam.rnti);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params multiSlotTxIndicator: {} ", __FUNCTION__, uciParam.multiSlotTxIndicator);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params pi2Bpsk: {} ", __FUNCTION__, uciParam.pi2Bpsk);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params startPrb: {} ", __FUNCTION__, uciParam.startPrb);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params prbSize: {} ", __FUNCTION__, uciParam.prbSize);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params startSym: {} ", __FUNCTION__, uciParam.startSym);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params nSym: {} ", __FUNCTION__, uciParam.nSym);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params freqHopFlag: {} ", __FUNCTION__, uciParam.freqHopFlag);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params secondHopPrb: {} ", __FUNCTION__, uciParam.secondHopPrb);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params groupHopFlag: {} ", __FUNCTION__, uciParam.groupHopFlag);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params sequenceHopFlag: {} ", __FUNCTION__, uciParam.sequenceHopFlag);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params initialCyclicShift: {} ", __FUNCTION__, uciParam.initialCyclicShift);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params timeDomainOccIdx: {} ", __FUNCTION__, uciParam.timeDomainOccIdx);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params srFlag: {} ", __FUNCTION__, uciParam.srFlag);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params bitLenSr: {} ", __FUNCTION__, uciParam.bitLenSr);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params bitLenHarq: {} ", __FUNCTION__, uciParam.bitLenHarq);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params bitLenCsiPart1: {} ", __FUNCTION__, uciParam.bitLenCsiPart1);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params AddDmrsFlag: {} ", __FUNCTION__, uciParam.AddDmrsFlag);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params dataScramblingId: {} ", __FUNCTION__, uciParam.dataScramblingId);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params DmrsScramblingId: {} ", __FUNCTION__, uciParam.DmrsScramblingId);
                }
            }
            break;
            case 1: {
                for(uint16_t n = 0; n < pucch_params->grp_dyn_pars.nF1Ucis; n++)
                {
                    cuphyPucchUciPrm_t& uciParam = pucch_params->grp_dyn_pars.pF1UciPrms[n];
                    if(i != uciParam.cellPrmDynIdx)
                    {
                        continue;
                    }
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params cellPrmDynIdx: {} ", __FUNCTION__, uciParam.cellPrmDynIdx);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params uciOutputIdx: {} ", __FUNCTION__, uciParam.uciOutputIdx);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params formatType: {} ", __FUNCTION__, uciParam.formatType);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params rnti: {} ", __FUNCTION__, uciParam.rnti);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params multiSlotTxIndicator: {} ", __FUNCTION__, uciParam.multiSlotTxIndicator);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params pi2Bpsk: {} ", __FUNCTION__, uciParam.pi2Bpsk);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params startPrb: {} ", __FUNCTION__, uciParam.startPrb);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params prbSize: {} ", __FUNCTION__, uciParam.prbSize);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params startSym: {} ", __FUNCTION__, uciParam.startSym);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params nSym: {} ", __FUNCTION__, uciParam.nSym);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params freqHopFlag: {} ", __FUNCTION__, uciParam.freqHopFlag);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params secondHopPrb: {} ", __FUNCTION__, uciParam.secondHopPrb);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params groupHopFlag: {} ", __FUNCTION__, uciParam.groupHopFlag);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params sequenceHopFlag: {} ", __FUNCTION__, uciParam.sequenceHopFlag);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params initialCyclicShift: {} ", __FUNCTION__, uciParam.initialCyclicShift);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params timeDomainOccIdx: {} ", __FUNCTION__, uciParam.timeDomainOccIdx);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params srFlag: {} ", __FUNCTION__, uciParam.srFlag);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params bitLenSr: {} ", __FUNCTION__, uciParam.bitLenSr);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params bitLenHarq: {} ", __FUNCTION__, uciParam.bitLenHarq);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params bitLenCsiPart1: {} ", __FUNCTION__, uciParam.bitLenCsiPart1);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params AddDmrsFlag: {} ", __FUNCTION__, uciParam.AddDmrsFlag);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params dataScramblingId: {} ", __FUNCTION__, uciParam.dataScramblingId);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params DmrsScramblingId: {} ", __FUNCTION__, uciParam.DmrsScramblingId);
                }
            }
            break;
            case 2: {
                for(uint16_t n = 0; n < pucch_params->grp_dyn_pars.nF2Ucis; n++)
                {
                    cuphyPucchUciPrm_t& uciParam = pucch_params->grp_dyn_pars.pF2UciPrms[n];
                    if(i != uciParam.cellPrmDynIdx)
                    {
                        continue;
                    }
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params cellPrmDynIdx: {} ", __FUNCTION__, uciParam.cellPrmDynIdx);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params uciOutputIdx: {} ", __FUNCTION__, uciParam.uciOutputIdx);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params formatType: {} ", __FUNCTION__, uciParam.formatType);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params rnti: {} ", __FUNCTION__, uciParam.rnti);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params multiSlotTxIndicator: {} ", __FUNCTION__, uciParam.multiSlotTxIndicator);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params pi2Bpsk: {} ", __FUNCTION__, uciParam.pi2Bpsk);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params startPrb: {} ", __FUNCTION__, uciParam.startPrb);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params prbSize: {} ", __FUNCTION__, uciParam.prbSize);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params startSym: {} ", __FUNCTION__, uciParam.startSym);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params nSym: {} ", __FUNCTION__, uciParam.nSym);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params freqHopFlag: {} ", __FUNCTION__, uciParam.freqHopFlag);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params secondHopPrb: {} ", __FUNCTION__, uciParam.secondHopPrb);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params groupHopFlag: {} ", __FUNCTION__, uciParam.groupHopFlag);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params sequenceHopFlag: {} ", __FUNCTION__, uciParam.sequenceHopFlag);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params initialCyclicShift: {} ", __FUNCTION__, uciParam.initialCyclicShift);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params timeDomainOccIdx: {} ", __FUNCTION__, uciParam.timeDomainOccIdx);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params srFlag: {} ", __FUNCTION__, uciParam.srFlag);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params bitLenSr: {} ", __FUNCTION__, uciParam.bitLenSr);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params bitLenHarq: {} ", __FUNCTION__, uciParam.bitLenHarq);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params bitLenCsiPart1: {} ", __FUNCTION__, uciParam.bitLenCsiPart1);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params AddDmrsFlag: {} ", __FUNCTION__, uciParam.AddDmrsFlag);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params dataScramblingId: {} ", __FUNCTION__, uciParam.dataScramblingId);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params DmrsScramblingId: {} ", __FUNCTION__, uciParam.DmrsScramblingId);

                    if(first_uci)
                    {
                        PucchDataOut->pPucchF2OutOffsets[n].uciSeg1PayloadByteOffset = 0;
                        first_uci                                                    = false;
                        uci_offset += (uciParam.bitLenSr + uciParam.bitLenHarq + uciParam.bitLenCsiPart1 + 7) >> 3;
                    }
                    else
                    {
                        PucchDataOut->pPucchF2OutOffsets[n].uciSeg1PayloadByteOffset = uci_offset;
                        uci_offset += (uciParam.bitLenSr + uciParam.bitLenHarq + uciParam.bitLenCsiPart1 + 7) >> 3;
                    }
                }
            }
            break;
            case 3: {
                for(uint16_t n = 0; n < pucch_params->grp_dyn_pars.nF3Ucis; n++)
                {
                    cuphyPucchUciPrm_t& uciParam = pucch_params->grp_dyn_pars.pF3UciPrms[n];
                    if(i != uciParam.cellPrmDynIdx)
                    {
                        continue;
                    }
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params cellPrmDynIdx: {} ", __FUNCTION__, uciParam.cellPrmDynIdx);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params uciOutputIdx: {} ", __FUNCTION__, uciParam.uciOutputIdx);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params formatType: {} ", __FUNCTION__, uciParam.formatType);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params rnti: {} ", __FUNCTION__, uciParam.rnti);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params multiSlotTxIndicator: {} ", __FUNCTION__, uciParam.multiSlotTxIndicator);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params pi2Bpsk: {} ", __FUNCTION__, uciParam.pi2Bpsk);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params startPrb: {} ", __FUNCTION__, uciParam.startPrb);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params prbSize: {} ", __FUNCTION__, uciParam.prbSize);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params startSym: {} ", __FUNCTION__, uciParam.startSym);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params nSym: {} ", __FUNCTION__, uciParam.nSym);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params freqHopFlag: {} ", __FUNCTION__, uciParam.freqHopFlag);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params secondHopPrb: {} ", __FUNCTION__, uciParam.secondHopPrb);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params groupHopFlag: {} ", __FUNCTION__, uciParam.groupHopFlag);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params sequenceHopFlag: {} ", __FUNCTION__, uciParam.sequenceHopFlag);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params initialCyclicShift: {} ", __FUNCTION__, uciParam.initialCyclicShift);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params timeDomainOccIdx: {} ", __FUNCTION__, uciParam.timeDomainOccIdx);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params srFlag: {} ", __FUNCTION__, uciParam.srFlag);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params bitLenSr: {} ", __FUNCTION__, uciParam.bitLenSr);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params bitLenHarq: {} ", __FUNCTION__, uciParam.bitLenHarq);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params bitLenCsiPart1: {} ", __FUNCTION__, uciParam.bitLenCsiPart1);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params AddDmrsFlag: {} ", __FUNCTION__, uciParam.AddDmrsFlag);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params dataScramblingId: {} ", __FUNCTION__, uciParam.dataScramblingId);
                    NVLOGD_FMT(TAG_PARAM, "{} \tL2:PUCCH params DmrsScramblingId: {} ", __FUNCTION__, uciParam.DmrsScramblingId);

                    if(first_uci)
                    {
                        PucchDataOut->pPucchF3OutOffsets[n].uciSeg1PayloadByteOffset = 0;
                        first_uci                                                    = false;
                        uci_offset += (uciParam.bitLenSr + uciParam.bitLenHarq + uciParam.bitLenCsiPart1 + 7) >> 3;
                    }
                    else
                    {
                        PucchDataOut->pPucchF3OutOffsets[n].uciSeg1PayloadByteOffset = uci_offset;
                        uci_offset += (uciParam.bitLenSr + uciParam.bitLenHarq + uciParam.bitLenCsiPart1 + 7) >> 3;
                    }
                }
            }
            break;
            case 4: {
            }
            break;
            }
        }
    }
}

void printSlotInfo(slot_info_t* l2)
{
}

//////////////////////////////

void SimulatePhyDriver::phy_driver_thread_func()
{
    if(worker_cfg != nullptr)
    {
        nv::config_thread_property(*worker_cfg);
    }

    SlotTask* task;
    while(sem_wait(&sem) == 0)
    {
        while((task = dequeue_task()) != nullptr)
        {
            onSlotTask(task);
            delete task;
        }
    }
}

SimulatePhyDriver::SimulatePhyDriver(nv::thread_config* cfg)
{
    memset(zero_u8, 0, sizeof(zero_u8));
    memset(zero_u32, 0, sizeof(zero_u32));
    memset(zero_float, 0, sizeof(zero_float));

    memset(pStartOffsetsCbCrc, 0, sizeof(pStartOffsetsCbCrc));
    memset(pStartOffsetsTbCrc, 0, sizeof(pStartOffsetsTbCrc));
    memset(pStartOffsetsTbPayload, 0, sizeof(pStartOffsetsTbPayload));

    // PUSCH output buffers
    puschDataOut.pUciOnPuschOutOffsets = new cuphyUciOnPuschOutOffsets_t[slot_command_api::MAX_PUSCH_UE_PER_TTI];

    puschDataOut.h_harqBufferSizeInBytes = zero_u32;
    puschDataOut.pCfoHz                  = zero_float;
    puschDataOut.pUciCrcFlags            = zero_u8;
    puschDataOut.pNumCsi2Bits            = zero_u16;
    puschDataOut.pTaEsts                 = zero_float;
    puschDataOut.pRssi                   = zero_float;
    puschDataOut.pRsrp                   = zero_float;
    puschDataOut.pNoiseVarPreEq          = zero_float;
    puschDataOut.pNoiseVarPostEq         = zero_float;
    puschDataOut.pSinrPreEq              = zero_float;
    puschDataOut.pSinrPostEq             = zero_float;
    puschDataOut.HarqDetectionStatus     = zero_u8;
    puschDataOut.CsiP1DetectionStatus    = zero_u8;
    puschDataOut.CsiP2DetectionStatus    = zero_u8;

    puschDataOut.pTbCrcs      = zero_u32;
    puschDataOut.pCbCrcs      = zero_u32;
    puschDataOut.pTbPayloads  = zero_u8;
    puschDataOut.pUciPayloads = zero_u8;

    puschDataOut.pStartOffsetsCbCrc     = pStartOffsetsCbCrc;
    puschDataOut.pStartOffsetsTbCrc     = pStartOffsetsTbCrc;
    puschDataOut.pStartOffsetsTbPayload = pStartOffsetsTbPayload;

    puschDataOut.totNumTbs = 0;

    // PUCCH output buffers
    pucchDataOut.pF0UcisOut         = new cuphyPucchF0F1UciOut_t[slot_command_api::MAX_PUSCH_UE_PER_TTI];
    pucchDataOut.pF1UcisOut         = new cuphyPucchF0F1UciOut_t[slot_command_api::MAX_PUSCH_UE_PER_TTI];
    pucchDataOut.pPucchF2OutOffsets = new cuphyPucchF234OutOffsets_t[slot_command_api::MAX_PUSCH_UE_PER_TTI];
    pucchDataOut.pPucchF3OutOffsets = new cuphyPucchF234OutOffsets_t[slot_command_api::MAX_PUSCH_UE_PER_TTI];
    pucchDataOut.pPucchF4OutOffsets = new cuphyPucchF234OutOffsets_t[slot_command_api::MAX_PUSCH_UE_PER_TTI];
    pucchDataOut.pUciPayloads       = new uint8_t[20 * slot_command_api::MAX_PUSCH_UE_PER_TTI];

    srsDataOut.pSrsReports = new cuphySrsReport_t[slot_command_api::MAX_SRS_UE_PER_TTI];
    srsDataOut.pRbSnrBuffer = zero_float; //new float[slot_command_api::MAX_SRS_UE_PER_TTI * slot_command_api::MAX_SRS_PRG_SIZE];
    srsDataOut.pRbSnrBuffOffsets = zero_u32; //new uint32_t[slot_command_api::MAX_SRS_UE_PER_TTI * slot_command_api::MAX_SRS_PRG_SIZE];

    worker_cfg = cfg;
    sem_init(&sem, 0, 0);
    thread = std::thread(&SimulatePhyDriver::phy_driver_thread_func, this);
}

int SimulatePhyDriver::set_output_callback(callbacks& cb)
{
    ul_cb = cb.ul_cb;
    dl_cb = cb.dl_cb;
    return 0;
}

void SimulatePhyDriver::send_uci_indications(slot_indication& si, cell_sub_command& csc)
{
    pucch_params* params = csc.params.pucch.get();
    if(params == nullptr)
    {
        return;
    }

    if(ul_cb.uci_cb_fn2)
    {
        cuphyPucchF0F1UciOut_t pUciOut0[params->params[0].size()];
        cuphyPucchF0F1UciOut_t pUciOut1[params->params[1].size()];

        cuphyPucchDataOut_t pucchOut = {nullptr, nullptr};
        if(params->params[0].size() > 0)
        {
            pucchOut.pF0UcisOut = pUciOut0;
        }
        if(params->params[1].size() > 0)
        {
            pucchOut.pF1UcisOut = pUciOut1;
        }

        for(int fmt = 0; fmt < 2; fmt++)
        {
            cuphyPucchF0F1UciOut_t*  pUciOut;
            cuphyPucchF0F1UciOut_t** ppUciOut;

            if(fmt == 0)
            {
                pUciOut  = pUciOut0;
                ppUciOut = &pucchOut.pF0UcisOut;
            }
            else if(fmt == 1)
            {
                pUciOut  = pUciOut1;
                ppUciOut = &pucchOut.pF1UcisOut;
            }

            uci_params_t& uci_v = params->params[fmt];
            if(uci_v.size() > 0)
            {
                *ppUciOut = pUciOut;
            }
            else
            {
                *ppUciOut = nullptr;
            }

            for(int nUci = 0; nUci < uci_v.size(); nUci++)
            {
                cuphyPucchUciPrm_t&     uciPrm = uci_v[nUci];
                cuphyPucchF0F1UciOut_t& uci    = *(pUciOut + nUci);
                uci.SRindication               = uciPrm.srFlag;
                uci.SRconfidenceLevel          = 1;
                uci.NumHarq                    = uciPrm.bitLenHarq;
                uci.HarqconfidenceLevel        = 1;
                uci.HarqValues[0]              = 0x01;
                uci.HarqValues[1]              = 0x00;
                NVLOGD_FMT(TAG, "{}: callback PHY_UL_HARQ_INDICATION cell={} nUci={} SRindication={} NumHarq={}", __FUNCTION__, params->cell_index_list[0], nUci, uci.SRindication, uci.NumHarq);
            }
        }

        NVLOGI_FMT(TAG, "{}: callback PHY_UL_HARQ_INDICATION cell={} nF0Ucis={} nF1Ucis={}", __FUNCTION__, params->cell_index_list[0], params->grp_dyn_pars.nF0Ucis, params->grp_dyn_pars.nF1Ucis);

        ul_cb.uci_cb_fn2(ul_cb.uci_cb_fn2_context, si, *params, pucchOut);
    }
    else if(ul_cb.uci_cb_fn)
    {
        // PucchParams& pucch = params->pucch[0];
        // NVLOGI_FMT(TAG, "{}: callback PHY_UL_HARQ_INDICATION cell={} numHarq={} format={} num_ue={}", __FUNCTION__,
        //     params->cell_index_list[0], outParams.numHarq, pucch.format, pucch.num_pucch_ue);

        // ul_cb.uci_cb_fn(si, *params, outParams);
    }
}

int SimulatePhyDriver::enqueue_task(SlotTask* task)
{
    NVLOGV_FMT(TAG, "{}: task={}", __func__, reinterpret_cast<void*>(task));
    queue_mutex.lock();
    task_queue.push(task);
    queue_mutex.unlock();
    return 0;
}

SlotTask* SimulatePhyDriver::dequeue_task()
{
    SlotTask* task = nullptr;
    queue_mutex.lock();
    if(task_queue.size() > 0)
    {
        task = task_queue.front();
        task_queue.pop();
    }
    queue_mutex.unlock();
    NVLOGV_FMT(TAG, "{}: task={}", __func__, reinterpret_cast<void*>(task));
    return task;
}

int SimulatePhyDriver::enqueue_phy_work(slot_command& command)
{
    SlotTask* task = new SlotTask(command);
    enqueue_task(task);
    sem_post(&sem);
    return 0;
}

class vec_to_str {
public:
    template <typename T, std::size_t N>
    vec_to_str(std::array<T, N> vec, char separator = ' ')
    {
        ss << "[" << vec.size() << ": ";
        bool first = true;
        for(auto val : vec)
        {
            if(first)
            {
                first = false;
            }
            else
            {
                ss << " ";
            }

            ss << val;
        }
        ss << "]";
    }

    std::string str()
    {
        return ss.str();
    }

private:
    std::stringstream ss;
};

int SimulatePhyDriver::onSlotTask(SlotTask* task)
{
    slot_command*                      p_command = task->get_slot_cmd();
    slot_command&                      command   = *p_command;
    cell_group_command&                cgc       = command.cell_groups;
    slot_command_api::slot_indication& si        = cgc.slot.slot_3gpp;

    vec_to_str chs = vec_to_str(cgc.channels);
    NVLOGI_FMT(TAG, "SFN {}.{} SlotTask: cgc.slot.type={} group.channels={} sub_cmds={}", si.sfn_, si.slot_, static_cast<int>(cgc.slot.type), chs.str().c_str(), command.cells.size());

    for (int i =0; i < cgc.channel_array_size; i++)
    //  /for(channel_type channel : cgc.channels)
    {
        auto channel = cgc.channels[i];
        if(channel == channel_type::PUSCH)
        {
            slot_command_api::ul_output_msg_buffer buf;
            slot_command_api::pusch_params*        pusch = cgc.pusch.get();
            if(pusch == nullptr)
            {
                break;
            }
            for(uint32_t i = 1; i < slot_command_api::MAX_PUSCH_UE_PER_TTI; i++)
            {
                puschDataOut.pStartOffsetsTbPayload[i] = pusch->ue_tb_size[i - 1];
            }
            for(int i = 0; i < 64; ++i)
            {
                puschDataOut.pStartOffsetsTbCrc[i] = i;
            }
            puschDataOut.totNumTbs = 1;

            printPuschCellGroupParams(pusch);
            printParameters(&pusch->cell_grp_info);

            // TODO: UCI on PUSCH
            uint16_t              nUe     = pusch->cell_grp_info.nUes;
            uint16_t              nUeGrps = pusch->cell_grp_info.nUeGrps;
            uint16_t              nCells  = pusch->cell_grp_info.nCells;
            cuphyPuschUeGrpPrm_t& ue_grp  = pusch->ue_grp_info[0]; //nUeGrps - 1
            cuphyPuschUePrm_t&    ue      = pusch->ue_info[0];     // nUe -1
            if(ue.pduBitmap & 0x2)
            {
                // PUSCH UCI Data
                cuphyUciOnPuschPrm_t* uci_prms = ue.pUciPrms;
                // uci_prms->nBitsHarq;
                puschDataOut.totNumUciSegs = 0;
            }

            ul_cb.callback_fn(ul_cb.callback_fn_context, 0, buf, si, *pusch, &puschDataOut, &puschStatPrms);
        }
        if(channel == channel_type::PUCCH)
        {
            slot_command_api::pucch_params* pucch = cgc.pucch.get();
            if(pucch == nullptr)
            {
                break;
            }
            printParameters(pucch, &pucchDataOut);
            ul_cb.uci_cb_fn2(ul_cb.uci_cb_fn2_context, si, *pucch, pucchDataOut);
        }
        if(channel == channel_type::SRS)
        {
            slot_command_api::ul_output_msg_buffer buf;
            slot_command_api::srs_params* srs = cgc.srs.get();
            if(srs == nullptr)
            {
                break;
            }
            //printParameters(srs, &srsDataOut);
            std::array<bool,UL_MAX_CELLS_PER_SLOT> srs_order_cell_timeout_list = {false};
            ul_cb.srs_cb_fn(ul_cb.srs_cb_context, buf, si, *srs, &srsDataOut, &srsStatPrms, srs_order_cell_timeout_list);
        }

        if(channel == channel_type::PDSCH)
        {
            slot_command_api::pdsch_params* dlParam = command.cell_groups.get_pdsch_params();
            if(dlParam == nullptr)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: slot={} channel={}-{} pdsch_params={}", __FUNCTION__, si.slot_, cgc.channel_array_size, +channel, reinterpret_cast<void*>(cgc.pdsch.get()));
                break;
            }
            NVLOGD_FMT(TAG, "{}: slot={} channel={}-{} pdsch_params={}", __FUNCTION__, si.slot_, cgc.channel_array_size, +channel, reinterpret_cast<void*>(cgc.pdsch.get()));
            //dl_cb.callback_fn(si);
            dl_cb.callback_fn(dl_cb.callback_fn_context, dlParam);
            printParameters(&(dlParam->cell_grp_info));
        }
    }
    NVLOGI_FMT(TAG, "SFN {}.{} SlotTask: channels={} - finished sub_cmds={}", si.sfn_, si.slot_, cgc.channel_array_size, command.cells.size());

    if(cgc.channel_array_size > 0)
    {
        return 0;
    }

#if 1
    for(cell_sub_command& csc : command.cells)
    {
        slot_command_api::slot_indication& si = csc.slot.slot_3gpp;

        for (int i =0; i < csc.channel_array_size; i++)
        // for(channel_type channel : csc.channels)
        {
            auto channel = csc.channels[i];
            NVLOGI_FMT(TAG, "{}: cell={} slot={} type={} channel={}-{}", __FUNCTION__, csc.cell, si.slot_, static_cast<int>(csc.slot.type), csc.channel_array_size, static_cast<int>(channel));

            switch(csc.slot.type)
            {
            case slot_command_api::SLOT_UPLINK: {
                if(channel == channel_type::PUSCH)
                {
                    slot_command_api::ul_output_msg_buffer buf;
                    slot_command_api::pusch_params*        pusch = csc.get_pusch_params();
                    for(uint32_t i = 1; i < slot_command_api::MAX_PUSCH_UE_PER_TTI; i++)
                    {
                        puschDataOut.pStartOffsetsTbPayload[i] = pusch->ue_tb_size[i - 1];
                    }
                    for(int i = 0; i < 64; ++i)
                    {
                        puschDataOut.pStartOffsetsTbCrc[i] = i;
                    }
                    ul_cb.callback_fn(ul_cb.callback_fn_context, 0, buf, si, *pusch, &puschDataOut, &puschStatPrms);
                    printParameters(&pusch->cell_grp_info);
                }
                if(channel == channel_type::PRACH)
                {
                    int      num_detectedPrmb[24]    = {0};
                    uint32_t prmbIndex_estimates[24] = {0};
                    float    prmbDelay_estimates[24] = {0};
                    float    prmbPower_estimates[24] = {0};
                    float    rssi                    = 0;
                    float    ant_rssi                = 0;
                    float    interference            = 0;

                    slot_command_api::prach_params* params = csc.get_prach_params();
                    ul_cb.prach_cb_fn(ul_cb.prach_cb_context, si, *params, (const unsigned int*)&num_detectedPrmb, (const void*)&prmbIndex_estimates, (const void*)&prmbDelay_estimates, (const void*)&prmbPower_estimates, (const void*)&rssi, (const void*)&ant_rssi, (const void*)&interference);
                }

                if(channel == channel_type::PUCCH)
                {
                    send_uci_indications(si, csc);
                }
            }
            break;

            case slot_command_api::SLOT_DOWNLINK: {
                if(channel == channel_type::PDSCH)
                {
                    slot_command_api::pdsch_params* dlParam = command.cell_groups.get_pdsch_params();
                    if(dlParam == nullptr)
                    {
                        break;
                    }
                    //TODO: Fix this
                    //dl_cb.callback_fn(si);
                    //dl_cb.callback_fn(dlParam);
                    printParameters(&(dlParam->cell_grp_info));
                }
                if(channel == channel_type::CSI_RS)
                {
                    slot_command_api::csirs_params* dlParam = csc.params.csi_rs.get();
                    if(dlParam == nullptr)
                    {
                        break;
                    }
                    for(auto& csirs_pdu : dlParam->csirsList)
                    {
                        printParameters(&csirs_pdu);
                    }
                    //TODO: Fix this
                    //dl_cb.callback_fn(si);
                    //dl_cb.callback_fn(dlParam);
                }
            }
            break;

            case slot_command_api::SLOT_SPECIAL:
            default: {
                if(channel == channel_type::PUCCH)
                {
                    // slot_command_api::pucch_params *params = csc.get_pucch_params();
                    // slot_command_api::pucch_params *params = csc.params.pucch.get();
                    // PucchParams& pucch = params->pucch[0];

                    // slot_command_api::uci_output_params outParams;
                    // outParams.numHarq = 1;
                    // outParams.harqConfidenceLevel = 0;
                    // outParams.harq_pdu[0] = 0x80;
                    // outParams.srIndication = 1;
                    // outParams.srConfidenceLevel = 0;
                    // cuphyPucchDataOut_t outParams2;

                    // NVLOGI_FMT(TAG, "{}: callback PHY_UL_HARQ_INDICATION cell={} numHarq={} format={} num_ue={}", __FUNCTION__,
                    //         params->cell_index, outParams.numHarq, pucch.format, pucch.num_pucch_ue);
                    // // pucch.format = 1;
                    // // pucch.num_pucch_ue = 1;
                    // // params->cell_index = 0;

                    // ul_cb.uci_cb_fn(si, *params, outParams);
                    send_uci_indications(si, csc);
                }
            }
            break;
            }
        }
    }
#endif
    return 0;
}
