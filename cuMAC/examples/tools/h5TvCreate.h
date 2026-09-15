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

#pragma once

#include "api.h"
#include <H5Cpp.h>

void saveToH5_Asim(const std::string&                 filename,
                   cumac::cumacCellGrpUeStatus*       cellGrpUeStatus,
                   cumac::cumacCellGrpPrms*           cellGrpPrms,
                   cumac::cumacSchdSol*               schdSol,
                   uint8_t                            saveBfWeights = 0);

void saveToH5(const std::string&                 filename,
              cumac::cumacCellGrpUeStatus*       cellGrpUeStatus,
              cumac::cumacCellGrpPrms*           cellGrpPrms,
              cumac::cumacSchdSol*               schdSol);

void saveToH5_CPU(const std::string&             filename,
                  cumac::cumacCellGrpUeStatus*   cellGrpUeStatus,
                  cumac::cumacCellGrpPrms*       cellGrpPrms,
                  cumac::cumacSchdSol*           schdSol);

void saveToH5_testMAC_perCell(const std::string&                 filename,
                              uint16_t                           cellId,
                              cumac::cumacCellGrpUeStatus*       cellGrpUeStatus,
                              cumac::cumacCellGrpPrms*           cellGrpPrms,
                              cumac::cumacSchdSol*               schdSol);

/**
 * @brief Write cuMAC multi-cell MU-MIMO TV to HDF5 including per-slot PHY and scheduler logs.
 *
 * Extends saveToH5() by also serializing per-UE and per-cell time series (MCS, SINR/SNR/SIR,
 * coupling metrics, rates, etc.) into additional datasets. Empty outer vectors skip writing the
 * corresponding dataset.
 *
 * @param filename Path to the output HDF5 file (truncated if it already exists).
 * @param cellGrpUeStatus Device pointer: UE status per cell group (host-staged copy for write).
 * @param cellGrpPrms Device pointer: cell-group parameters (dimensions drive tensor shapes).
 * @param schdSol Device pointer: schedule solution (precoders, UE maps, layer selections, etc.).
 * @param perUEperSlotMcs Selected MCS index per active UE per simulation slot; 2D [ue][slot] (unitless).
 * @param perUEperSlotLayerSel Selected layer / rank count per active UE per slot; 2D [ue][slot] (unitless).
 * @param perUEperSlotAvgSinr Post-equalization average SINR per UE per slot; 2D [ue][slot], linear power ratio (not dB).
 * @param perUEperRbgperSlotGeometrySinr Geometry (coupling-loss) SINR per UE per PRBG per slot; 3D [ue][prbGroup][slot], linear.
 * @param perUEperSlotServingCellChannelGain Serving-cell channel gain per UE per slot; 2D [ue][slot], dB.
 * @param perUEperCellperSlotAllCellsChannelGain Channel gain to each cell per UE per slot; 3D [ue][cell][slot], dB.
 * @param perUEperSlotServingCellPathLossAndSF Combined path loss and shadow fading (serving cell) per UE per slot; 2D [ue][slot], dB.
 * @param perUEperCellperSlotAllCellsPathLossAndSF Path loss + shadow fading per neighbor cell per UE per slot; 3D [ue][cell][slot], dB.
 * @param perUEperRbgperSlotGeometrySir Geometry SIR per UE per PRBG per slot; 3D [ue][prbGroup][slot], linear.
 * @param perUEperRbgperSlotGeometrySnr Geometry SNR per UE per PRBG per slot; 3D [ue][prbGroup][slot], linear.
 * @param perUEperRbgperSlotRawPreEqSinr Raw pre-equalization SINR per UE per PRBG per slot; 3D [ue][prbGroup][slot], linear.
 * @param perUEperRbgperSlotRawPreEqSir Raw pre-equalization SIR per UE per PRBG per slot; 3D [ue][prbGroup][slot], linear.
 * @param perUEperRbgperSlotRawPreEqSnr Raw pre-equalization SNR per UE per PRBG per slot; 3D [ue][prbGroup][slot], linear.
 * @param perUEperRbgperLayerperSlotRawSinr Raw SINR per stream layer per UE per PRBG per slot; 4D [ue][prbGroup][layer][slot], linear.
 * @param perUEperSlotTbErr Transport-block error indicator per UE per slot; 2D [ue][slot] (simulator convention, typically 0/1).
 * @param perUEperSlotBler Block error rate per UE per slot; 2D [ue][slot], fraction in [0,1].
 * @param perUEperSlotInsRate Instantaneous throughput per UE per slot; 2D [ue][slot], simulator rate units.
 * @param perUEperSlotAvgRate Filtered / average throughput per UE per slot; 2D [ue][slot], same units as instantaneous rate.
 * @param perCellperSlotNumScheUEs Number of scheduled UEs per cell per slot; 2D [cell][slot] (count).
 * @param perCellperSlotTbErr Cell-aggregate TB error metric per slot; 2D [cell][slot] (simulator-defined float).
 * @param perCellperSlotInsRate Cell aggregate instantaneous rate per slot; 2D [cell][slot], simulator rate units.
 * @param perCellperGrpperSlotNumScheLayers Scheduled layer count per cell per PRG group per slot; 3D [cell][group][slot] (count).
 */
void saveToH5_perSlotLog(const std::string&                 filename,
                         cumac::cumacCellGrpUeStatus*       cellGrpUeStatus,
                         cumac::cumacCellGrpPrms*           cellGrpPrms,
                         cumac::cumacSchdSol*               schdSol,
                         const std::vector<std::vector<int>>& perUEperSlotMcs,
                         const std::vector<std::vector<int>>& perUEperSlotLayerSel,
                         const std::vector<std::vector<float>>& perUEperSlotAvgSinr,
                         const std::vector<std::vector<std::vector<float>>>& perUEperRbgperSlotGeometrySinr,
                         const std::vector<std::vector<float>>& perUEperSlotServingCellChannelGain,
                         const std::vector<std::vector<std::vector<float>>>& perUEperCellperSlotAllCellsChannelGain,
                         const std::vector<std::vector<float>>& perUEperSlotServingCellPathLossAndSF,
                         const std::vector<std::vector<std::vector<float>>>& perUEperCellperSlotAllCellsPathLossAndSF,
                         const std::vector<std::vector<std::vector<float>>>& perUEperRbgperSlotGeometrySir,
                         const std::vector<std::vector<std::vector<float>>>& perUEperRbgperSlotGeometrySnr,
                         const std::vector<std::vector<std::vector<float>>>& perUEperRbgperSlotRawPreEqSinr,
                         const std::vector<std::vector<std::vector<float>>>& perUEperRbgperSlotRawPreEqSir,
                         const std::vector<std::vector<std::vector<float>>>& perUEperRbgperSlotRawPreEqSnr,
                         const std::vector<std::vector<std::vector<std::vector<float>>>>& perUEperRbgperLayerperSlotRawSinr,
                         const std::vector<std::vector<int>>& perUEperSlotTbErr,
                         const std::vector<std::vector<float>>& perUEperSlotBler,
                         const std::vector<std::vector<float>>& perUEperSlotInsRate,
                         const std::vector<std::vector<float>>& perUEperSlotAvgRate,
                         const std::vector<std::vector<int>>& perCellperSlotNumScheUEs,
                         const std::vector<std::vector<float>>& perCellperSlotTbErr,
                         const std::vector<std::vector<float>>& perCellperSlotInsRate,
                         const std::vector<std::vector<std::vector<int>>>& perCellperGrpperSlotNumScheLayers);
