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

"""pyAerial - Pipeline configuration classes."""
from dataclasses import dataclass
from dataclasses import field
from typing import List, Optional

import numpy as np
import cupy as cp  # type: ignore

from aerial.util.cuda import CudaStream
from aerial.phy5g.csirs.csirs_api import CsiRsConfig
from aerial.phy5g.api import SlotConfig
from aerial.phy5g.api import PipelineConfig
from aerial.util.fapi import dmrs_bit_array_to_fapi
from aerial import pycuphy
from aerial.pycuphy.types import CsiRsRrcDynPrms
from aerial.pycuphy.types import DataType
from aerial.pycuphy.types import PuschLdpcKernelLaunch
from aerial.pycuphy.types import PuschCellDynPrm
from aerial.pycuphy.types import PuschUeGrpPrm
from aerial.pycuphy.types import PuschDmrsPrm
from aerial.pycuphy.types import PuschCellGrpDynPrm
from aerial.pycuphy.types import PuschUePrm
from aerial.pycuphy.types import PuschDataIn
from aerial.pycuphy.types import PuschDataOut
from aerial.pycuphy.types import PuschDataInOut
from aerial.pycuphy.types import PuschDynPrms
from aerial.pycuphy.types import PuschSetupPhase
from aerial.pycuphy.types import PuschStatusOut
from aerial.pycuphy.types import PuschStatusType
from aerial.pycuphy.types import PuschDynDbgPrms
from aerial.pycuphy.types import PdschUeGrpPrm
from aerial.pycuphy.types import PdschUePrm
from aerial.pycuphy.types import PdschCellDynPrm
from aerial.pycuphy.types import PdschCwPrm
from aerial.pycuphy.types import PdschCellGrpDynPrm
from aerial.pycuphy.types import PdschDataIn
from aerial.pycuphy.types import CuPHYTensor
from aerial.pycuphy.types import PdschDataOut
from aerial.pycuphy.types import PdschDynPrms
from aerial.pycuphy.types import PmW


@dataclass
class PuschUeConfig:
    """A class holding all dynamic PUSCH parameters for a single slot, single UE.

    Args:
        scid (int): DMRS sequence initialization [TS38.211, sec 7.4.1.1.2].
        layers (int): Number of layers.
        dmrs_ports (int): Allocated DMRS ports.
        rnti (int): The 16-bit RNTI value of the UE.
        data_scid (List[int]): Data scrambling ID, more precisely `dataScramblingIdentityPdsch`
            [TS38.211, sec 7.3.1.1].
        mcs_table (int): MCS table to use (see TS 38.214).
        mcs_index (int): MCS index to use.
        code_rate (int): Code rate, expressed as the number of information
            bits per 1024 coded bits expressed in 0.1 bit units.
        mod_order (int): Modulation order.
        tb_size (int): TB size in bytes.
        rv (List[int]): Redundancy version.
        ndi (List[int]): New data indicator.
    """
    scid: int = 0
    layers: int = 1
    dmrs_ports: int = 1
    rnti: int = 1
    data_scid: int = 41
    mcs_table: int = 0
    mcs_index: int = 0
    code_rate: int = 1930
    mod_order: int = 2
    tb_size: int = 96321
    rv: int = 0
    ndi: int = 1
    harq_process_id: int = 0


@dataclass
class PuschConfig(SlotConfig):
    """A class holding all dynamic PUSCH parameters for a single slot, single UE group.

    Args:
        num_dmrs_cdm_grps_no_data (int): Number of DMRS CDM groups without data
            [3GPP TS 38.212, sec 7.3.1.1]. Value: 1->3.
        dmrs_scrm_id (int): DMRS scrambling ID.
        start_prb (int): Start PRB index of the UE group allocation.
        num_prbs (int): Number of allocated PRBs for the UE group.
        prg_size (int): The Size of PRG in PRB for PUSCH per-PRG channel estimation.
        num_ul_streams (int): The number of active streams for this PUSCH.
        dmrs_syms (List[int]): For the UE group, a list of binary numbers each indicating whether
            the corresponding symbol is a DMRS symbol.
        dmrs_max_len (int): The `maxLength` parameter, value 1 or 2, meaning that DMRS are
            single-symbol DMRS or single- or double-symbol DMRS. Note that this needs to be
            consistent with `dmrs_syms`.
        dmrs_add_ln_pos (int): Number of additional DMRS positions.  Note that this needs to be
            consistent with `dmrs_syms`.
        start_sym (int): Start OFDM symbol index for the UE group allocation.
        num_symbols (int): Number of symbols in the UE group allocation.
    """
    # UE parameters.
    ue_configs: List[PuschUeConfig]

    # UE group parameters.
    num_dmrs_cdm_grps_no_data: int = 2
    dmrs_scrm_id: int = 41
    start_prb: int = 0
    num_prbs: int = 273
    prg_size: int = 1
    num_ul_streams: int = 1
    dmrs_syms: List[int] = \
        field(default_factory=lambda: [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
    dmrs_max_len: int = 2
    dmrs_add_ln_pos: int = 1
    start_sym: int = 2
    num_symbols: int = 12


@dataclass
class AerialPuschRxConfig(PipelineConfig):
    """Aerial PUSCH receiver pipeline configuration.

    Args:
        cell_id (int): Physical cell ID.
        num_rx_ant (int): Number of receive antennas.
        num_ul_bwp (int): Number of PRBs in a uplink bandwidth part.
            Default: 273.
        num_dl_bwp (int): Number of PRBs in a downlink bandwidth part.
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

        enable_pusch_tdi (int): Time domain interpolation on PUSCH.

            - 0: Disable (default).
            - 1: Enable.

        eq_coeff_algo (int): Algorithm for equalizer coefficient computation.

            - 0 - ZF.
            - 1 - MMSE (default).
            - 2 - MMSE-IRC.

        enable_per_prg_chest (int): Enable/disable PUSCH per-PRG channel estimation.

            - 0: Disable (default).
            - 1: Enable.

        enable_ul_rx_bf (int): Enable/disable beamforming for PUSCH.

            - 0: Disable (default).
            - 1: Enable.

        chest_factory_settings_filename (Optional[str]): Filename for chestFactorySettings.

            - None: (default)

        ldpc_kernel_launch (PuschLdpcKernelLaunch): LDPC kernel launch method.
    """
    cell_id: int
    num_rx_ant: int
    num_ul_bwp: int = 273
    num_dl_bwp: int = 273
    mu: int = 1
    enable_cfo_correction: int = 0
    enable_weighted_ave_cfo_est: int = 0
    enable_to_estimation: int = 0
    enable_pusch_tdi: int = 0
    eq_coeff_algo: int = 1
    enable_per_prg_chest: int = 0
    enable_ul_rx_bf: int = 0
    chest_factory_settings_filename: Optional[str] = None
    ldpc_kernel_launch: PuschLdpcKernelLaunch = PuschLdpcKernelLaunch.PUSCH_RX_ENABLE_DRIVER_LDPC_LAUNCH  # noqa: E501 # pylint: disable=line-too-long


def _pusch_config_to_cuphy(
        cuda_stream: CudaStream,
        rx_data: List[cp.ndarray],
        slot: int,
        pusch_configs: List[PuschConfig]) -> PuschDynPrms:
    """Convert pyAerial PuschConfig to format that the cuPHY PUSCH components want."""
    cell_dyn_prm = PuschCellDynPrm(
        cellPrmStatIdx=np.uint16(0),
        cellPrmDynIdx=np.uint16(0),
        slotNum=np.uint16(slot)
    )

    ue_grp_prms = []
    ue_prms = []
    num_ues = 0

    for ue_grp_idx, pusch_config in enumerate(pusch_configs):
        num_ues_in_grp = len(pusch_config.ue_configs)
        ue_grp_prms.append(PuschUeGrpPrm(
            cellPrmIdx=0,
            dmrsDynPrm=PuschDmrsPrm(
                dmrsAddlnPos=np.uint8(pusch_config.dmrs_add_ln_pos),
                dmrsMaxLen=np.uint8(pusch_config.dmrs_max_len),
                numDmrsCdmGrpsNoData=np.uint8(pusch_config.num_dmrs_cdm_grps_no_data),
                dmrsScrmId=np.uint16(pusch_config.dmrs_scrm_id)
            ),
            startPrb=np.uint16(pusch_config.start_prb),
            nPrb=np.uint16(pusch_config.num_prbs),
            prgSize=np.uint16(pusch_config.prg_size),
            nUplinkStreams=np.uint16(pusch_config.num_ul_streams),
            puschStartSym=np.uint8(pusch_config.start_sym),
            nPuschSym=np.uint8(pusch_config.num_symbols),
            dmrsSymLocBmsk=np.uint16(dmrs_bit_array_to_fapi(pusch_config.dmrs_syms)),
            rssiSymLocBmsk=np.uint16(dmrs_bit_array_to_fapi(pusch_config.dmrs_syms)),
            uePrmIdxs=list(np.array(range(num_ues, num_ues + num_ues_in_grp), dtype=np.uint16))
        ))

        for ue_params in pusch_config.ue_configs:
            ue_prms.append(PuschUePrm(
                pduBitmap=np.uint16(1),
                ueGrpIdx=np.uint16(ue_grp_idx),
                enableTfPrcd=np.uint8(0),  # Disabled, not supported by pyAerial.
                scid=np.uint8(ue_params.scid),
                dmrsPortBmsk=np.uint16(ue_params.dmrs_ports),
                nlAbove16=np.uint8(0),
                mcsTable=np.uint8(ue_params.mcs_table),
                mcsIndex=np.uint8(ue_params.mcs_index),
                TBSize=np.uint32(ue_params.tb_size),
                targetCodeRate=np.uint16(ue_params.code_rate),
                qamModOrder=np.uint8(ue_params.mod_order),
                rv=np.uint8(ue_params.rv),
                rnti=np.uint16(ue_params.rnti),
                dataScramId=np.uint16(ue_params.data_scid),
                nUeLayers=np.uint8(ue_params.layers),
                ndi=np.uint8(ue_params.ndi),
                harqProcessId=np.uint8(ue_params.harq_process_id),
                # The following hard-coded.
                i_lbrm=np.uint8(0),
                maxLayers=np.uint8(4),
                maxQm=np.uint8(8),
                n_PRB_LBRM=np.uint16(273)
            ))

        num_ues += num_ues_in_grp

    cell_grp_dyn_prm = PuschCellGrpDynPrm(
        cellPrms=[cell_dyn_prm],
        ueGrpPrms=ue_grp_prms,
        uePrms=ue_prms,
    )

    # Wrap CuPy arrays into pycuphy types.
    rx_data = [pycuphy.CudaArrayComplexFloat(elem) for elem in rx_data]
    data_in = PuschDataIn(tDataRx=rx_data)

    data_out = PuschDataOut(
        harqBufferSizeInBytes=np.zeros([num_ues], dtype=np.uint32),
        totNumTbs=np.zeros([1], dtype=np.uint32),
        totNumCbs=np.zeros([1], dtype=np.uint32),
        totNumPayloadBytes=np.zeros([1], dtype=np.uint32),
        totNumUciSegs=np.zeros([1], dtype=np.uint16),
        cbCrcs=np.ones([1000], dtype=np.uint32),
        tbCrcs=np.ones([num_ues], dtype=np.uint32),
        tbPayloads=np.zeros([200000], dtype=np.uint8),
        uciPayloads=None,
        uciCrcFlags=None,
        numCsi2Bits=None,
        startOffsetsCbCrc=np.zeros([num_ues], dtype=np.uint32),
        startOffsetsTbCrc=np.zeros([num_ues], dtype=np.uint32),
        startOffsetsTbPayload=np.zeros([num_ues], dtype=np.uint32),
        taEsts=np.zeros([num_ues], dtype=float),
        rssi=np.zeros([1], dtype=float),
        rsrp=np.zeros([num_ues], dtype=float),
        noiseVarPreEq=np.zeros([num_ues], dtype=float),
        noiseVarPostEq=np.zeros([num_ues], dtype=float),
        sinrPreEq=np.zeros([num_ues], dtype=float),
        sinrPostEq=np.zeros([num_ues], dtype=float),
        cfoHz=np.zeros([num_ues], dtype=float),
        HarqDetectionStatus=np.zeros([num_ues], dtype=np.uint8),
        CsiP1DetectionStatus=np.zeros([num_ues], dtype=np.uint8),
        CsiP2DetectionStatus=np.zeros([num_ues], dtype=np.uint8),
        preEarlyHarqWaitStatus=np.zeros([1], dtype=np.uint8),
        postEarlyHarqWaitStatus=np.zeros([1], dtype=np.uint8),
    )

    data_in_out = PuschDataInOut(
        harqBuffersInOut=[]
    )
    pusch_dyn_prms = PuschDynPrms(
        phase1Stream=cuda_stream.handle,
        phase2Stream=cuda_stream.handle,
        setupPhase=PuschSetupPhase.PUSCH_SETUP_PHASE_1,
        procModeBmsk=np.uint64(0),  # Controls PUSCH mode (e.g., will use CUDA graphs
                                    # if least significant bit is 1; streams if 0)
        waitTimeOutPreEarlyHarqUs=np.uint16(1000),
        waitTimeOutPostEarlyHarqUs=np.uint16(1500),
        cellGrpDynPrm=cell_grp_dyn_prm,
        dataIn=data_in,
        dataOut=data_out,
        dataInOut=data_in_out,
        cpuCopyOn=np.uint8(1),
        statusOut=PuschStatusOut(
            status=PuschStatusType.CUPHY_PUSCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE,
            cellPrmStatIdx=np.uint16(0),
            ueIdx=np.uint16(0)
        ),
        dbg=PuschDynDbgPrms(
            enableApiLogging=np.uint8(0)
        ))

    return pusch_dyn_prms


def _csi_rs_config_to_cuphy(csi_rs_config: CsiRsConfig) -> CsiRsRrcDynPrms:
    """Convert pyAerial CsiRsConfig to format that the cuPHY CSI-RS components want."""
    enable_precoding = False

    # TODO: Add precoding

    csi_rs_rrc_dyn_prms = CsiRsRrcDynPrms(
        start_prb=np.uint16(csi_rs_config.start_prb),
        num_prb=np.uint16(csi_rs_config.num_prb),
        freq_alloc=csi_rs_config.freq_alloc,

        row=np.uint8(csi_rs_config.row),
        symb_L0=np.uint8(csi_rs_config.symb_L0),
        symb_L1=np.uint8(csi_rs_config.symb_L1),
        freq_density=np.uint8(csi_rs_config.freq_density),
        scramb_id=np.uint16(csi_rs_config.scramb_id),
        idx_slot_in_frame=np.uint8(csi_rs_config.idx_slot_in_frame),
        csi_type=np.uint8(1),  # Only NZP supported.
        cdm_type=np.uint8(csi_rs_config.cdm_type),
        beta=csi_rs_config.beta,
        enable_precoding=np.uint8(enable_precoding),
        pmw_prm_idx=None
    )

    return csi_rs_rrc_dyn_prms


@dataclass
class AerialPdschTxConfig(PipelineConfig):
    """Aerial PDSCH transmitter pipeline configuration.

    Args:
        cell_id (int): Physical cell ID.
        num_tx_ant (int): Number of transmit antennas.
        num_dl_bwp (int): Number of PRBs in a downlink bandwidth part.
            Default: 273.
        mu (int): Numerology. Values in [0, 3]. Default: 1.
    """
    cell_id: int
    num_tx_ant: int
    num_dl_bwp: int = 273
    mu: int = 1


@dataclass
class PdschCwConfig:
    """A class holding all dynamic PDSCH parameters for a single slot, single codeword.

    Args:
        mcs_table (int): MCS table index.
        mcs_index (int): MCS index.
        code_rate (int): Code rate, expressed as the number of information
            bits per 1024 coded bits expressed in 0.1 bit units.
        mod_order (int): Modulation order.
        rvs (int): Redundancy version (default: 0).
        num_prb_lbrm (int): Number of PRBs used for LBRM TB size computation.
            Possible values: {32, 66, 107, 135, 162, 217, 273}.
        max_layers (int): Number of layers used for LBRM TB size computation (at most 4).
        max_qm (int): Modulation order used for LBRM TB size computation. Value: 6 or 8.
    """
    mcs_table: int = 0
    mcs_index: int = 0
    code_rate: int = 1930
    mod_order: int = 2
    rv: int = 0
    num_prb_lbrm: int = 273
    max_layers: int = 4
    max_qm: int = 8


@dataclass
class PdschUeConfig:
    """A class holding all dynamic PDSCH parameters for a single slot, single UE.

    Args:
        scid (int): DMRS sequence initialization [TS38.211, sec 7.4.1.1.2].
        dmrs_scrm_id (int): Downlink DMRS scrambling ID.
        layers (int): Number of layers.
        dmrs_ports (int): Allocated DMRS ports. The format of the entry is in the
            SCF FAPI format as follows: A bitmap (mask) starting from the LSB where each bit
            indicates whether the corresponding DMRS port index is used.
        bwp_start (int): Bandwidth part start (PRB number starting from 0).
            Used only if reference point is 1.
        ref_point (int): DMRS reference point. Value 0 or 1.
        beta_qam (float): Amplitude factor of QAM signal.
        beta_dmrs (float): Amplitude factor of DMRS signal.
        rnti (int) RNTI for the UE.
        data_scid (List[int]): Data scrambling ID for the UE, more precisely
            `dataScramblingIdentityPdsch` [TS38.211, sec 7.3.1.1].
        precoding_matrix (np.ndarray): Precoding matrix. The shape of the matrix is
            number of layers x number of Tx antennas. If set to None, precoding is disabled.
    """
    # Codeword configurations.
    cw_configs: List[PdschCwConfig]

    scid: int = 0
    dmrs_scrm_id: int = 41
    layers: int = 1
    dmrs_ports: int = 1
    bwp_start: int = 0
    ref_point: int = 0
    beta_qam: float = 1.0
    beta_dmrs: float = 1.0
    rnti: int = 1
    data_scid: int = 1
    precoding_matrix: np.ndarray = None


@dataclass
class PdschConfig(SlotConfig):
    """A class holding all dynamic PDSCH parameters for a single slot, single UE group.

    Args:
        num_dmrs_cdm_grps_no_data (int): Number of DMRS CDM groups without data
            [3GPP TS 38.212, sec 7.3.1.1]. Value: 1->3.
        resource_alloc (int): Resource allocation type.
        prb_bitmap (List[int]): Array of bits indicating bitmask for allocated RBs.
        start_prb (int): Start PRB index of the UE group allocation.
        num_prbs (int): Number of allocated PRBs for the UE group.
        dmrs_syms (List[int]): For the UE group, a list of binary numbers each indicating whether
            the corresponding symbol is a DMRS symbol.
        start_sym (int): Start OFDM symbol index for the UE group allocation.
        num_symbols (int): Number of symbols in the UE group allocation.
    """
    # UE parameters.
    ue_configs: List[PdschUeConfig]

    # UE group parameters.
    num_dmrs_cdm_grps_no_data: int = 2
    resource_alloc: int = 1
    prb_bitmap: List[int] = \
        field(default_factory=lambda: [0, ] * 36)
    start_prb: int = 0
    num_prbs: int = 273
    dmrs_syms: List[int] = \
        field(default_factory=lambda: [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
    start_sym: int = 2
    num_symbols: int = 12


@dataclass
class PdschDmrsConfig:
    """A class holding all dynamic PDSCH DMRS parameters for a single slot, single UE group.

    Args:
        cell_index_in_cell_group (int): Cell index in cell group. This is used to index the
            Tx buffers, for example.
        num_bwp_prbs (int): Number of PRBs in the BWP.
        num_dmrs_cdm_grps_no_data (int): Number of DMRS CDM groups without data.
        dmrs_scrm_id (int): Downlink DMRS scrambling ID.
        resource_alloc (int): Resource allocation type. 0 or 1.
        prb_bitmap (List[int]): A bitmap indicating allocated RBs per FAPI Table 3-70.
        start_prb (int): Start PRB index of the UE group allocation. Not valid for RA type 0.
            Value: 0-274.
        num_prbs (int): Number of allocated PRBs for the UE group. Must be populated correctly
            regardless of RA Type. Value: 1-275.
        dmrs_syms (List[int]): For the UE group, a list of binary numbers each indicating whether
            the corresponding symbol is a DMRS symbol.
        start_sym (int): Start OFDM symbol index for the UE group allocation.
        num_pdsch_syms (int): Number of PDSCH symbols in the UE group allocation.
        scid (int): DMRS sequence initialization [TS38.211, sec 7.4.1.1.2].
        layers (int): Number of layers.
        dmrs_ports (int): Allocated DMRS ports. The actual port (as per TS 38.211) is value of
            DMRS port + 1000. Only the first `layers` values are valid.
        bwp_start (int): Bandwidth part start (PRB number starting from 0).
        ref_point (int): DMRS reference point. Value 0 or 1.
        beta_qam (float): Amplitude factor of QAM signal.
        beta_dmrs (float): Amplitude factor of DMRS signal.
        precoding_matrix (np.ndarray): Precoding matrix. The shape of the matrix is
            number of layers x number of antenna ports. If set to None, precoding is disabled.
    """
    cell_index_in_cell_group: int = 0
    num_bwp_prbs: int = 273
    num_dmrs_cdm_grps_no_data: int = 2
    dmrs_scrm_id: int = 41
    resource_alloc: int = 1
    prb_bitmap: List[int] = \
        field(default_factory=lambda: [0, ] * 36)
    start_prb: int = 0
    num_prbs: int = 273
    dmrs_syms: List[int] = \
        field(default_factory=lambda: [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
    start_sym: int = 2
    num_pdsch_syms: int = 12
    scid: int = 0
    layers: int = 1
    dmrs_ports: int = 1
    bwp_start: int = 0
    ref_point: int = 0
    beta_qam: float = 1.0
    beta_dmrs: float = 1.0
    precoding_matrix: np.ndarray = None

    @classmethod
    def from_pdsch_config(
            cls,
            pdsch_config: PdschConfig,
            num_bwp_prbs: int) -> List['PdschDmrsConfig']:
        """Convert pyAerial PdschConfig to a list of PdschDmrsConfig objects.

        Args:
            pdsch_config (PdschConfig): The PDSCH configuration to convert.

        Returns:
            List[PdschDmrsConfig]: A list of PdschDmrsConfig objects.
        """
        dmrs_configs = []
        for ue_config in pdsch_config.ue_configs:
            dmrs_config = PdschDmrsConfig(
                cell_index_in_cell_group=0,
                num_bwp_prbs=num_bwp_prbs,
                num_dmrs_cdm_grps_no_data=pdsch_config.num_dmrs_cdm_grps_no_data,
                resource_alloc=pdsch_config.resource_alloc,
                prb_bitmap=pdsch_config.prb_bitmap,
                start_prb=pdsch_config.start_prb,
                num_prbs=pdsch_config.num_prbs,
                dmrs_syms=pdsch_config.dmrs_syms,
                start_sym=pdsch_config.start_sym,
                num_pdsch_syms=pdsch_config.num_symbols,
                scid=ue_config.scid,
                dmrs_scrm_id=ue_config.dmrs_scrm_id,
                layers=ue_config.layers,
                dmrs_ports=ue_config.dmrs_ports,
                bwp_start=ue_config.bwp_start,
                ref_point=ue_config.ref_point,
                beta_qam=ue_config.beta_qam,
                beta_dmrs=ue_config.beta_dmrs,
                precoding_matrix=ue_config.precoding_matrix
            )
            dmrs_configs.append(dmrs_config)
        return dmrs_configs


def _pdsch_config_to_cuphy(  # pylint: disable=too-many-locals
        *,
        cuda_stream: CudaStream,
        tb_inputs: List[cp.ndarray],
        tx_output_mem: np.uint64,
        slot: int,
        pdsch_configs: List[PdschConfig],
        csi_rs_configs: List[CsiRsConfig] = None) -> PdschDynPrms:
    """Convert pyAerial PdschConfig to format that the cuPHY PDSCH components want."""
    num_csi_prms = 0
    csi_rs_prms = None
    if csi_rs_configs is not None:
        num_csi_prms = len(csi_rs_configs)
        csi_rs_prms = [_csi_rs_config_to_cuphy(csi_rs_config)
                       for csi_rs_config in csi_rs_configs]

    cell_dyn_prms = PdschCellDynPrm(
        nCsiRsPrms=np.uint16(num_csi_prms),
        csiRsPrmsOffset=np.uint16(0),
        cellPrmStatIdx=np.uint16(0),
        cellPrmDynIdx=np.uint16(0),
        slotNum=np.uint16(slot),
        pdschStartSym=np.uint8(0),
        nPdschSym=np.uint8(0),
        dmrsSymLocBmsk=np.uint16(0),
        testModel=np.uint8(0)
    )

    ue_grp_prms = []
    ue_prms = []
    cw_prms = []
    num_ues = 0
    ue_idx = 0
    num_cws = 0
    tb_offset = 0
    pmw_prms = None
    pmw_prm_idx = 0
    for ue_grp_idx, pdsch_config in enumerate(pdsch_configs):
        num_ues_in_grp = len(pdsch_config.ue_configs)

        ue_grp_prms.append(PdschUeGrpPrm(
            cellPrmIdx=0,
            nDmrsCdmGrpsNoData=np.uint8(pdsch_config.num_dmrs_cdm_grps_no_data),
            resourceAlloc=np.uint8(pdsch_config.resource_alloc),
            rbBitmap=list(np.array(pdsch_config.prb_bitmap).astype(np.uint8)),
            startPrb=np.uint16(pdsch_config.start_prb),
            nPrb=np.uint16(pdsch_config.num_prbs),
            dmrsSymLocBmsk=np.uint16(dmrs_bit_array_to_fapi(pdsch_config.dmrs_syms)),
            pdschStartSym=np.uint8(pdsch_config.start_sym),
            nPdschSym=np.uint8(pdsch_config.num_symbols),
            uePrmIdxs=list(np.array(range(num_ues, num_ues + num_ues_in_grp), dtype=np.uint16))
        ))

        for ue_params in pdsch_config.ue_configs:

            for cw_params in ue_params.cw_configs:
                cw_prms.append(PdschCwPrm(
                    uePrmIdx=ue_idx,
                    mcsTableIndex=np.uint8(cw_params.mcs_table),
                    mcsIndex=np.uint8(cw_params.mcs_index),
                    targetCodeRate=np.uint16(cw_params.code_rate),
                    qamModOrder=np.uint8(cw_params.mod_order),
                    rv=np.uint8(cw_params.rv),
                    tbStartOffset=np.uint32(tb_offset),
                    tbSize=np.uint32(tb_inputs[ue_idx].size),
                    n_PRB_LBRM=np.uint16(cw_params.num_prb_lbrm),
                    maxLayers=np.uint8(cw_params.max_layers),
                    maxQm=np.uint8(cw_params.max_qm)
                ))

            # Precoding enabled if a precoding matrix is given for the UE.
            enable_prcd_bf = False
            ue_pmw_prm_idx = None
            if ue_params.precoding_matrix is not None and ue_params.precoding_matrix.size > 0:
                enable_prcd_bf = True
                ue_pmw_prm_idx = pmw_prm_idx
                pmw = PmW(
                    w=ue_params.precoding_matrix,
                    nPorts=np.uint8(ue_params.precoding_matrix.shape[1])
                )
                if pmw_prm_idx == 0:
                    pmw_prms = [pmw]
                else:
                    pmw_prms += [pmw]  # type: ignore
                pmw_prm_idx += 1

            num_cws_per_ue = len(ue_params.cw_configs)
            ue_prms.append(PdschUePrm(
                ueGrpPrmIdx=ue_grp_idx,
                scid=np.uint8(ue_params.scid),
                dmrsScrmId=np.uint16(ue_params.dmrs_scrm_id),
                nUeLayers=np.uint8(ue_params.layers),
                dmrsPortBmsk=np.uint16(ue_params.dmrs_ports),
                BWPStart=np.uint16(ue_params.bwp_start),
                refPoint=np.uint8(ue_params.ref_point),
                beta_dmrs=np.float32(ue_params.beta_dmrs),
                beta_qam=np.float32(ue_params.beta_qam),
                rnti=np.uint16(ue_params.rnti),
                dataScramId=np.uint16(ue_params.data_scid),
                cwIdxs=list(np.array(range(num_cws, num_cws + num_cws_per_ue), dtype=np.uint16)),
                enablePrcdBf=enable_prcd_bf,
                pmwPrmIdx=ue_pmw_prm_idx,
                nlAbove16=np.uint8(0)  # Not supporting 32 layer PDSCH
            ))

            tb_offset += tb_inputs[ue_idx].size
            ue_idx += 1

            # TODO: Support multiple.
            num_cws += num_cws_per_ue

        num_ues += num_ues_in_grp

    cell_grp_dyn_prm = PdschCellGrpDynPrm(
        cellPrms=[cell_dyn_prms],
        ueGrpPrms=ue_grp_prms,
        uePrms=ue_prms,
        cwPrms=cw_prms,
        csiRsPrms=csi_rs_prms,
        pmwPrms=pmw_prms
    )

    with cuda_stream:
        tb_inputs = cp.concatenate(tb_inputs)

    # Wrap CuPy arrays into pycuphy types.
    tb_inputs = pycuphy.CudaArrayUint8(tb_inputs)

    data_in = PdschDataIn(
        tbInput=[tb_inputs]
    )

    cuphy_tensor = CuPHYTensor(
        dimensions=[273 * 12, 14, 16],
        strides=[1, 3744, 45864],
        dataType=DataType.CUPHY_C_32F,
        pAddr=tx_output_mem
    )

    data_out = PdschDataOut(
        dataTx=[cuphy_tensor]
    )

    pdsch_dyn_prms = PdschDynPrms(
        cuStream=cuda_stream.handle,
        procModeBmsk=np.uint64(4),  # Enable inter-cell batching.
        cellGrpDynPrm=cell_grp_dyn_prm,
        dataIn=data_in,
        tbCRCDataIn=None,
        dataOut=data_out
    )

    return pdsch_dyn_prms
