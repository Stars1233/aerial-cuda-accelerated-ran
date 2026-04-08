# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""pyAerial - Utilities for handling cuPHY parameters."""
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from aerial.pycuphy.chest_filters import CUPHY_CHEST_COEFF_FILE
from aerial.pycuphy.chest_filters import pusch_chest_params_from_hdf5
from aerial.pycuphy.types import (
    PdschStatPrms, CellStatPrm,
    CuPHYTracker, PdschDbgPrms, PuschStatPrms, PuschStatDbgPrms,
    PuschDataInOut, PuschDynPrms, PuschSetupPhase,
    PuschEqCoefAlgoType, PuschLdpcKernelLaunch, LdpcMaxItrAlgoType,
    PuschChEstAlgoType, PuschWorkCancelMode
)

__all__ = [
    "get_pdsch_stat_prms",
    "get_pusch_stat_prms",
    "get_pusch_dyn_prms_phase_2"
]

# Constant definitions.
NUM_RE_PER_PRB = 12
NUM_PRB_MAX = 273
NUM_SYMBOLS = 14


def get_pdsch_stat_prms(
        *,
        cell_id: int,
        num_rx_ant: int,
        num_tx_ant: int,
        num_rx_ant_srs: Optional[int] = None,
        num_prb_ul_bwp: int = NUM_PRB_MAX,
        num_prb_dl_bwp: int = NUM_PRB_MAX,
        mu: int = 1) -> PdschStatPrms:
    """Get a simple PdschStatPrms object based on given parameters.

    Args:
        cell_id (int): Physical cell ID.
        num_rx_ant (int): Number of receive antennas.
        num_tx_ant (int): Number of transmit antennas.
        num_rx_ant_srs (int): Number of receive antennas for SRS.
        num_prb_ul_bwp (int): Number of PRBs in a uplink bandwidth part.
            Default: 273.
        num_prb_dl_bwp (int): Number of PRBs in a downlink bandwidth part.
            Default: 273.
        mu (int): Numerology. Values in [0, 3]. Default: 1.

    Returns:
        PdschStatPrms: The PdschStatPrms object.
    """
    num_rx_ant_srs = num_rx_ant_srs or num_rx_ant
    cell_stat_prm = CellStatPrm(
        phyCellId=np.uint16(cell_id),
        nRxAnt=np.uint16(num_rx_ant),
        nRxAntSrs=np.uint16(num_rx_ant_srs),
        nTxAnt=np.uint16(num_tx_ant),
        nPrbUlBwp=np.uint16(num_prb_ul_bwp),
        nPrbDlBwp=np.uint16(num_prb_dl_bwp),
        mu=np.uint8(mu)
    )

    cuphy_tracker = CuPHYTracker(
        memoryFootprint=[]  # Not used by pycuphy code.
    )

    dbg_params = PdschDbgPrms(
        cfgFilename=None,
        checkTbSize=np.uint8(1),
        refCheck=False,
        cfgIdenticalLdpcEncCfgs=False
    )

    pdsch_tx_stat_prms = PdschStatPrms(
        outInfo=[cuphy_tracker],
        cellStatPrms=[cell_stat_prm],
        dbg=[dbg_params],
        read_TB_CRC=False,
        full_slot_processing=True,
        stream_priority=0,
        nMaxCellsPerSlot=np.uint16(1),
        nMaxUesPerCellGroup=np.uint16(0),
        nMaxCBsPerTB=np.uint16(0),
        nMaxPrb=np.uint16(0)
    )

    return pdsch_tx_stat_prms


def get_pusch_stat_prms(  # pylint: disable=too-many-arguments
        *,
        cell_id: int = 41,
        num_rx_ant: int = 4,
        num_tx_ant: int = 1,
        num_rx_ant_srs: Optional[int] = None,
        num_prb_ul_bwp: int = 273,
        num_prb_dl_bwp: int = 273,
        mu: int = 1,
        enable_cfo_correction: int = 0,
        enable_weighted_ave_cfo_est: int = 0,
        enable_to_estimation: int = 0,
        enable_pusch_tdi: int = 0,
        ch_est_algo: PuschChEstAlgoType = PuschChEstAlgoType.PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST,  # noqa: E501 # pylint: disable=line-too-long
        enable_per_prg_chest: int = 0,
        enable_ul_rx_bf: int = 0,
        eq_coeff_algo: PuschEqCoefAlgoType = PuschEqCoefAlgoType.PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
        ldpc_kernel_launch: PuschLdpcKernelLaunch = PuschLdpcKernelLaunch.PUSCH_RX_ENABLE_DRIVER_LDPC_LAUNCH,  # noqa: E501 # pylint: disable=line-too-long
        chest_factory_settings_filename: Optional[str] = None,
        debug_file_name: Optional[str] = None) -> PuschStatPrms:
    """Get a PuschStatPrms object based on given parameters.

    Args:
        cell_id (int): Physical cell ID.
        num_rx_ant (int): Number of receive antennas.
        num_tx_ant (int): Number of transmit antennas.
        num_rx_ant_srs (int): Number of receive antennas for SRS.
        num_prb_ul_bwp (int): Number of PRBs in a uplink bandwidth part.
            Default: 273.
        num_prb_dl_bwp (int): Number of PRBs in a downlink bandwidth part.
            Default: 273.
        mu (int): Numerology. Values in [0, 3]. Default: 1.
        enable_cfo_correction (int): Enable/disable CFO correction:

            - 0: Disable (default).
            - 1: Enable.

        enable_weighted_ave_cfo_est (int): Enable/disable CFO weighted average estimation:

            - 0: Disable (default).
            - 1: Enable.

        enable_to_estimation (int): Enable/disable time offset estimation:

            - 0: Disable (default).
            - 1: Enable.

        enable_pusch_tdi (int): Enable/disable time domain interpolation on PUSCH:

            - 0: Disable (default).
            - 1: Enable.

        ch_est_algo (PuschChEstAlgoType): Channel estimation algorithm.

            - 0: MMSE
            - 1: MMSE with delay estimation/compensation
            - 2: RKHS not supported by pyAerial yet
            - 3: LS channel estimation only, no interpolation, LS channel estimates get returned.

        enable_per_prg_chest (int): Enable/disable PUSCH per-PRG channel estimation.

            - 0: Disable (default).
            - 1: Enable.

        enable_ul_rx_bf (int): Enable/disable beamforming for PUSCH.

            - 0: Disable (default).
            - 1: Enable.

        eq_coeff_algo (PuschEqCoefAlgoType): Algorithm for equalizer coefficient computation.

            - 0 - ZF.
            - 1 - MMSE (default).
            - 2 - MMSE-IRC.

        chest_factory_settings_filename (str): Chest factory settings filename.

        ldpc_kernel_launch (PuschLdpcKernelLaunch): LDPC kernel launch method.
        debug_file_name (str): Debug dump filename. Default: None (no debugging).

    Returns:
        PuschStatPrms: The PuschStatPrms object.
    """
    num_rx_ant_srs = num_rx_ant_srs or num_rx_ant
    cell_stat_prm = CellStatPrm(
        phyCellId=np.uint16(cell_id),
        nRxAnt=np.uint16(num_rx_ant),
        nTxAnt=np.uint16(num_tx_ant),
        nRxAntSrs=np.uint16(num_rx_ant_srs),
        nPrbUlBwp=np.uint16(num_prb_ul_bwp),
        nPrbDlBwp=np.uint16(num_prb_dl_bwp),
        mu=np.uint8(mu)
    )

    cuphy_tracker = CuPHYTracker(
        memoryFootprint=[]
    )

    # Load channel estimation filters.
    filters = pusch_chest_params_from_hdf5(CUPHY_CHEST_COEFF_FILE)

    pusch_stat_prms = PuschStatPrms(
        outInfo=[cuphy_tracker],
        WFreq=filters["WFreq"],
        WFreq4=filters["WFreq4"],
        WFreqSmall=filters["WFreqSmall"],
        ShiftSeq=filters["ShiftSeq"],
        UnShiftSeq=filters["UnShiftSeq"],
        ShiftSeq4=filters["ShiftSeq4"],
        UnShiftSeq4=filters["UnShiftSeq4"],
        enableCfoCorrection=np.uint8(enable_cfo_correction),
        enableWeightedAverageCfo=np.uint8(enable_weighted_ave_cfo_est),
        enableToEstimation=np.uint8(enable_to_estimation),
        enablePuschTdi=np.uint8(enable_pusch_tdi),
        enableDftSOfdm=np.uint8(0),  # Disable this feature now.
        enableTbSizeCheck=np.uint8(1),  # Always enabled.
        enableUlRxBf=np.uint8(enable_ul_rx_bf),
        enableDebugEqOutput=np.uint8(0),
        ldpcnIterations=np.uint8(10),  # To be deprecated.
        ldpcEarlyTermination=np.uint8(0),
        ldpcUseHalf=np.uint8(1),
        ldpcAlgoIndex=np.uint8(0),
        ldpcFlags=np.uint8(2),
        ldpcKernelLaunch=ldpc_kernel_launch,
        ldpcMaxNumItrAlgo=LdpcMaxItrAlgoType.LDPC_MAX_NUM_ITR_ALGO_TYPE_LUT,
        fixedMaxNumLdpcItrs=np.uint8(10),
        ldpcClampValue=np.float32(128.0),
        polarDcdrListSz=np.uint8(8),
        nMaxTbPerNode=np.uint8(32),
        chEstAlgo=PuschChEstAlgoType(ch_est_algo),
        enablePerPrgChEst=np.uint8(enable_per_prg_chest),
        eqCoeffAlgo=eq_coeff_algo,
        enableRssiMeasurement=np.uint8(0),
        enableSinrMeasurement=np.uint8(0),
        enableCsiP2Fapiv3=np.uint8(0),
        stream_priority=0,
        nMaxCells=np.uint16(1),
        nMaxCellsPerSlot=np.uint16(1),
        cellStatPrms=[cell_stat_prm],
        nMaxTbs=np.uint32(0),
        nMaxCbsPerTb=np.uint32(0),
        nMaxTotCbs=np.uint32(0),
        nMaxRx=np.uint32(0),
        nMaxPrb=np.uint32(273),
        nMaxLdpcHetConfigs=np.uint32(32),
        dbg=PuschStatDbgPrms(
            outFileName=debug_file_name,
            descrmOn=np.uint8(1),
            enableApiLogging=np.uint8(0),
            forcedNumCsi2Bits=np.uint16(0)
        ),
        enableDeviceGraphLaunch=np.uint8(0),
        enableEarlyHarq=np.uint8(0),
        earlyHarqProcNodePriority=np.int32(0),
        workCancelMode=PuschWorkCancelMode.PUSCH_NO_WORK_CANCEL,  # update as needed
        enableBatchedMemcpy=np.uint8(0),
        chestFactorySettingsFilename=chest_factory_settings_filename,
    )

    return pusch_stat_prms


def get_pusch_dyn_prms_phase_2(
        pusch_dyn_prms_phase1: PuschDynPrms,
        harq_buffer: Union[int, List[int]]) -> PuschDynPrms:
    """Get dynamic PUSCH phase 2 setup parameters."""
    if isinstance(harq_buffer, int):
        harq_buffer = [harq_buffer]

    pusch_dyn_prms_phase2 = pusch_dyn_prms_phase1._replace(
        setupPhase=PuschSetupPhase.PUSCH_SETUP_PHASE_2,
        dataInOut=PuschDataInOut(harqBuffersInOut=harq_buffer)
    )
    return pusch_dyn_prms_phase2
