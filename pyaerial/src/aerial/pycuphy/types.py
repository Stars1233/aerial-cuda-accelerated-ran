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

# pylint: disable=too-many-lines
"""pyAerial - cuPHY API types definition."""
from typing import List, Optional
from typing import NamedTuple

import numpy

# Import directly pybinded enums.
from aerial import pycuphy  # type: ignore
from aerial.pycuphy import PuschProcMode  # type: ignore # pylint: disable=E0611
from aerial.pycuphy import PuschLdpcKernelLaunch  # type: ignore # pylint: disable=E0611
from aerial.pycuphy import LdpcMaxItrAlgoType  # type: ignore # pylint: disable=E0611
from aerial.pycuphy import DataType  # type: ignore # pylint: disable=E0611
from aerial.pycuphy import PuschSetupPhase  # type: ignore # pylint: disable=E0611
from aerial.pycuphy import PuschEqCoefAlgoType  # type: ignore # pylint: disable=E0611
from aerial.pycuphy import PuschStatusType  # type: ignore # pylint: disable=E0611
from aerial.pycuphy import PuschChEstAlgoType  # type: ignore # pylint: disable=E0611
from aerial.pycuphy import PuschWorkCancelMode  # type: ignore # pylint: disable=E0611


__all__ = [
    "PuschProcMode",
    "PuschLdpcKernelLaunch",
    "LdpcMaxItrAlgoType",
    "DataType",
    "PuschSetupPhase",
    "PuschEqCoefAlgoType",
    "PuschStatusType",
    "PuschChEstAlgoType",
    "CuPHYTensor",
    "CuPHYTracker",
    "CellStatPrm",
    "PdschDbgPrms",
    "PdschStatPrms",
    "PdschCellDynPrm",
    "PdschUeGrpPrm",
    "PdschUePrm",
    "PdschCwPrm",
    "CsiRsRrcDynPrms",
    "PmW",
    "PdschCellGrpDynPrm",
    "PdschDataIn",
    "PdschDataOut",
    "PdschDynPrms",
    "PuschStatDbgPrms",
    "PuschStatPrms",
    "PuschCellDynPrm",
    "PuschDmrsPrm",
    "PuschUeGrpPrm",
    "PuschUePrm",
    "PuschCellGrpDynPrm",
    "PuschDataIn",
    "PuschDataOut",
    "PuschDataInOut",
    "PuschDynDbgPrms",
    "PuschStatusOut",
    "PuschDynPrms"
]


class CuPHYTensor(NamedTuple):
    """Implement a cuPHY Tensor.

    Args:
        dimensions (List[int]): Logical dimensions of the tensor.
        strides (List[int]): Physical stride dimensions of the tensor.
        dataType (DataType): Defines the type of each element.
        pAddr (numpy.uint64): Raw pointer to the tensor buffer.
    """
    dimensions : List[int]
    strides : List[int]
    dataType : DataType
    pAddr : numpy.uint64


class CuPHYTracker(NamedTuple):
    """Implement cuPHY tracker type.

    Args:
        memoryFootprint List[numpy_uint64]: A single element list with a pointer
            to a cuphyMemoryFootprint object.
    """
    memoryFootprint : List[numpy.uint64]


class CellStatPrm(NamedTuple):
    """Implement cuPHY cell static parameters common to all channels.

    This corresponds to the cuPHY `cuphyCellStatPrm_t` struct.

    Args:
        phyCellId (numpy.uint16): Physical cell ID.
        nRxAnt (numpy.uint16): Number of receiving antennas.
        nTxAnt (numpy.uint16): Number of transmitting antennas.
        nRxAntSrs (numpy.uint16): Number of receiving antennas for SRS.
        nPrbUlBwp (numpy.uint16): Number of PRBs (Physical Resource Blocks) allocated in
            UL BWP (bandwidth part).
        nPrbDlBwp (numpy.uint16): Number of PRBs allocated in DL BWP.
        mu (numpy.uint8): Numerology [0, 3].
    """
    phyCellId : numpy.uint16
    nRxAnt : numpy.uint16
    nTxAnt : numpy.uint16
    nRxAntSrs : numpy.uint16
    nPrbUlBwp : numpy.uint16
    nPrbDlBwp : numpy.uint16
    mu : numpy.uint8


class PdschDbgPrms(NamedTuple):
    """Implement PDSCH channel debug parameters.

    This corresponds to the cuPHY `cuphyPdschDbgPrms_t` struct.

    Args:
        cfgFilename (str): Name of HDF5 file that drives the DL pipeline. No file, if empty.
        checkTbSize (numpy.uint8): If 1, cuPHY PDSCH will recompute TB size for initial
            transmission. Value: 0 or 1.
        refCheck (bool): If True, compare the output of each pipeline component with the reference
            output from the cfgFileName file that drives the pipeline.
        cfgIdenticalLdpcEncCfgs (bool): Enable single cuPHY LDPC call for all TBs, if True.
            Will be reset at runtime if LDPC config. params are different across TBs.
    """
    cfgFilename : str
    checkTbSize: numpy.uint8
    refCheck : bool
    cfgIdenticalLdpcEncCfgs : bool


class PdschStatPrms(NamedTuple):
    """Implement PDSCH static parameters.

    This corresponds to the cuPHY `cuphyPdschStatPrms_t` struct. The field
    `nCells` is missing, but equals to the length of the `cellStatPrms` array.

    Args:
        cellStatPrms (List[CellStatPrm]): List of cell-specific static parameters with
            `nCells` elements.
        dbg (List[PdschDbgPrms]): List of cell-specific debug parameters with `nCells`
            elements.
        read_TB_CRC (bool): If True, TB CRCs are read from input buffers and not computed.
        full_slot_processing (bool): If false, all cells ran on this PdschTx will undergo: TB-CRC +
            CB-CRC/segmentation + LDPC encoding + rate-matching/scrambling. If true, all cells ran
            on this PdschTx will undergo full slot processing:  TB-CRC + CB-CRC/segmentation +
            LDPC encoding + rate-matching/scrambling/layer-mapping + modulation + DMRS.
            NB: This mode is an a priori known characteristic of the cell; a cell will never switch
            between modes.
        stream_priority (int): CUDA stream priority for all internal to PDSCH streams. Should match
            the priority of CUDA stream passed in PdschDynPrms during setup.
        nMaxCellsPerSlot (numpy.uint16): Maximum number of cells supported.
            nCells <= nMaxCellsPerSlot and nMaxCellsPerSlot <= `PDSCH_MAX_CELLS_PER_CELL_GROUP`.
            If 0, cuPHY compile-time constant `PDSCH_MAX_CELLS_PER_CELL_GROUP` is used.
        nMaxUesPerCellGroup (numpy.uint16): Maximum number of UEs supported in a cell group, i.e.,
            across all the cells. nMaxUesPerCellGroup <= `PDSCH_MAX_UES_PER_CELL_GROUP`.
            If 0, the compile-time constant `PDSCH_MAX_UES_PER_CELL_GROUP` is used.
        nMaxCBsPerTB (numpy.uint16): Maximum number of CBs supported per TB; limit valid for any UE
            in that cell. nMaxCBsPerTb <= `MAX_N_CBS_PER_TB_SUPPORTED`.
            If 0, the compile-time constant `MAX_N_CBS_PER_TB_SUPPORTED` is used.
        nMaxPrb (numpy.uint16): Maximum value of CellStatPrm.nPrbDlBwp supported by PdschTx
            object. nMaxPrb <= 273. If 0, 273 is used.
        outInfo (List[T_cuPHYTracker]): pointer to cuPHY tracker
    """
    cellStatPrms : List[CellStatPrm]
    dbg : List[PdschDbgPrms]
    read_TB_CRC : bool
    full_slot_processing : bool
    stream_priority : int
    nMaxCellsPerSlot : numpy.uint16
    nMaxUesPerCellGroup : numpy.uint16
    nMaxCBsPerTB : numpy.uint16
    nMaxPrb : numpy.uint16
    outInfo : List[CuPHYTracker]


class PdschCellDynPrm(NamedTuple):
    """Implement PDSCH dynamic cell parameters.

    This corresponds to the cuPHY `cuphyPdschCellDynPrm_t` struct.

    Note about PDSCH time domain resource allocation:
    The pdschStartSym, nPdschSym and dmrsSymLocBmsk fields are also added at the user group level.
    The current expectation is that the caller uses the UE-group fields only if nPdschSym (cell
    level) and dmrsSymLocBmsk (cell level) are zero. If these fields are not zero, then the
    cell-level fields are used, and the implementation assumes these values are identical
    across all UEs and all UE groups belonging to this cell.

    Args:
        nCsiRsPrms (numpy.uint16): Number of CSI-RS params co-scheduled for this cell.
        csiRsPrmsOffset (numpy.uint16): Start index for this cell's nCsiRsPrms elements in the
            csiRsPrms array of `PdschCellGrpDynPrm`. All elements are allocated continuously.
        cellPrmStatIdx (numpy.uint16): Index to cell-static parameter information, i.e., to the
            `cellStatPrms` array of the `PdschStatPrms` struct.
        cellPrmDynIdx (numpy.uint16): Index to cell-dynamic parameter information, i.e., to the
            `cellPrms` array of the `PdschCellGrpDynPrm` struct.
        slotNum (numpy.uint16): Slot number. Value: 0 -> 319.
        pdschStartSym (numpy.uint8): PDSCH start symbol location (0-indexing). Value: 0->13.
        nPdschSym (numpy.uint8): PDSCH DMRS + data symbol count. Value: 1->14.
        dmrsSymLocBmsk (numpy.uint16): DMRS symbol location bitmask (least significant 14 bits
            are valid). A set bit i, specifies symbol i is DMRS.
            For example if symbols 2 and 3 are DMRS, then: dmrsSymLocBmsk = 0000 0000 0000 1100.
        testModel (numpy.uint8): Specifies the cell is in testing mode if set to 1. Value: 0-1.
            For cells in testing mode, the TB payload buffers hold bits from the PN23
            (pseudorandom) sequence instead of a TB payload.
    """
    nCsiRsPrms : numpy.uint16
    csiRsPrmsOffset : numpy.uint16
    cellPrmStatIdx : numpy.uint16
    cellPrmDynIdx : numpy.uint16
    slotNum : numpy.uint16
    pdschStartSym: numpy.uint8
    nPdschSym: numpy.uint8
    dmrsSymLocBmsk: numpy.uint16
    testModel : numpy.uint8


class PdschUeGrpPrm(NamedTuple):
    """Implement PDSCH UE group parameters.

    This corresponds to the cuPHY `cuphyPdschUeGrpPrm_t` struct, except the `cuphyPdschDmrsPrm_t`
    (which contains some obsolete parameters) is flattened into this structure too.

    Args:
        cellPrmIdx (int): Index of UE group's parent cell dynamic parameters (`PdschCellDynPrm`).
        nDmrsCdmGrpsNoData (numpy.uint8): Number of DM-RS CDM groups without data [TS38.212
            sec 7.3.1.2, TS38.214 Table 4.1-1]. It determines the ratio of PDSCH EPRE to DM-RS EPRE
            Value: 1 -> 3.
        resourceAlloc (numpy.uint8): For specifying resource allocation
            type [TS38.214, sec 5.1.2.2].
        rbBitmap List[numpy.uint8]: For resource alloc type 0. [TS38.212, sec 7.3.1.2.2].
            Bitmap of RBs, rounded up to multiple of 32. LSB of byte 0 of the bitmap
            represents the RB 0.
        startPrb (numpy.uint16): For resource allocation type 1. [TS38.214, sec 5.1.2.2.2].
            The starting resource block within the BWP for this PDSCH. Value: 0 -> 274.
        nPrb (numpy.uint16): Number of allocated PRBs. Must be populated correctly regardless of
            RA Type. Value: 1-275.
        dmrsSymLocBmsk (numpy.uint16): DMRS symbol location bitmask (least significant 14 bits are
            valid). A set bit i, specifies symbol i is DMRS. For example if symbols 2 and 3 are
            DMRS, then: dmrsSymLocBmsk = 0000 0000 0000 1100. This field will only have a valid
            value if the corresponding cell level field is zero.
        pdschStartSym (numpy.uint8): Start symbol index of PDSCH mapping from the start of the
            slot, S. [TS38.214, Table 5.1.2.1-1]. Value: 0 -> 13.
        nPdschSym (numpy.uint8): PDSCH duration in symbols, L. [TS38.214, Table 5.1.2.1-1].
            Value: 1 -> 14.
        uePrmIdxs List[numpy.uint16]: List, of length `nUes`, of indices into the uePrms list of
            `PdschCellGrpDynPrm`.
    """
    cellPrmIdx : int
    nDmrsCdmGrpsNoData : numpy.uint8
    resourceAlloc : numpy.uint8
    rbBitmap : List[numpy.uint8]
    startPrb : numpy.uint16
    nPrb : numpy.uint16
    dmrsSymLocBmsk : numpy.uint16
    pdschStartSym : numpy.uint8
    nPdschSym : numpy.uint8
    uePrmIdxs : List[numpy.uint16]


class PdschUePrm(NamedTuple):
    """Implement PDSCH UE parameters.

    This corresponds to the cuPHY `cuphyPdschUePrm_t` struct.

    Args:
        ueGrpPrmIdx (int): Index to parent UE group.
        scid (numpy.uint8): DMRS sequence initialization [TS38.211, sec 7.4.1.1.2].
            Should match what is sent in DCI 1_1, otherwise set to 0.
            Value : 0 -> 1.
        dmrsScrmId (numpy.uint16): UL-DMRS-Scrambling-ID [TS38.211, sec 7.4.1.1.2].
            Value: 0 -> 65535.
        nUeLayers (numpy.uint8): Number of layers [TS38.211, sec 7.3.1.3].
            Value: 1 -> 8.
        dmrsPortBmsk (numpy.uint16): Set bits in bitmask specify DMRS ports used. Port 0
            corresponds to least significant bit. Used to compute layers.
        BWPStart (numpy.uint16): Bandwidth part start RB index from
            reference CRB [TS38.213 sec 12]. Used only if ref. point is 1.
        refPoint (numpy.uint8): DMRS reference point. Value 0 -> 1.
        beta_dmrs (numpy.float32): Fronthaul DMRS amplitude scaling.
        beta_qam (numpy.float32): Fronthaul QAM amplitude scaling.
        rnti (numpy.uint16): The RNTI used for identifying the UE when receiving the PDU.
            RNTI == Radio Network Temporary Identifier. Value: 1 -> 65535.
        dataScramId (numpy.uint16): dataScramblingIdentityPdsch [TS38.211, sec 7.3.1.1].
            Value: 0 -> 65535.
        cwIdxs (List[int]): List of `nCw` elements containing indices into the cwPrms list of
            `PdschCellGrpDynPrm`.
        enablePrcdBf (bool): Enable pre-coding for this UE.
        pmwPrmIdx (int): Index to pre-coding matrix array, i.e., to the pmwPrms list of the
            `PdschCellGrpDynPrm`.
        nlAbove16 (numpy.uint8): Number of layers above 16. Value: 0 -> 1.
    """
    ueGrpPrmIdx : int
    scid : numpy.uint8
    dmrsScrmId : numpy.uint16
    nUeLayers : numpy.uint8
    dmrsPortBmsk : numpy.uint16
    BWPStart : numpy.uint16
    refPoint : numpy.uint8
    beta_dmrs : numpy.float32
    beta_qam : numpy.float32
    rnti : numpy.uint16
    dataScramId : numpy.uint16
    cwIdxs : List[int]
    enablePrcdBf : bool
    pmwPrmIdx : int
    nlAbove16 : numpy.uint8 = numpy.uint8(0)


class PdschCwPrm(NamedTuple):
    """Implement PDSCH codeword parameters.

    This corresponds to the cuPHY `cuphyPdschCwPrm_t` struct.

    Args:
        uePrmIdx (int): Index to parent UE.
        mcsTableIndex (numpy.uint8): Solely used in optional TB size checking. Use `targetCodeRate`
            and `qamModOrder` for everything else. MCS (Modulation and Coding Scheme) Table Id.
            Value: 0 - 2.

            - 0: Table 5.1.3.1-1
            - 1: Table 5.1.3.1-2
            - 2: Table 5.1.3.1-3

        mcsIndex (numpy.uint8): Solely used in optional TB size checking. Use `targetCodeRate`
            and `qamModOrder` for everything else. MCS index within the `mcsTableIndex` table.
            Value: 0 - 31.
        targetCodeRate (numpy.uint16): Target code rate. Assuming code rate is x/1024.0,
           where x contains a single digit after decimal point,
           `targetCodeRate` = x * 10 = code rate * 1024 * 10.
        qamModOrder (numpy.uint8): Modulation order. Value: 2, 4, 6 or 8.
        rv (numpy.uint8): Redundancy version index [TS38.212, Table 5.4.2.1- 2 and 38.214,
            Table 5.1.2.1-2], should match value sent in DCI. Value: 0 -> 3.
        tbStartOffset (numpy.uint32): Starting index (in bytes) of transport block within
            `tbInput` array in `PdschDataIn`.
        tbSize (numpy.uint32): Transmit block size (in bytes) [TS38.214 sec 5.1.3.2].
        n_PRB_LBRM (numpy.uint16): Number of PRBs used for LBRM TB size computation.
            Possible values: {32, 66, 107, 135, 162, 217, 273}.
        maxLayers (numpy.uint8): Number of layers used for LBRM TB size computation (at most 4).
        maxQm (numpy.uint8): Modulation order used for LBRM TB size computation. Value: 6 or 8.
    """
    uePrmIdx : int
    mcsTableIndex: numpy.uint8
    mcsIndex: numpy.uint8
    targetCodeRate : numpy.uint16
    qamModOrder : numpy.uint8
    rv : numpy.uint8
    tbStartOffset : numpy.uint32
    tbSize : numpy.uint32
    n_PRB_LBRM : numpy.uint16
    maxLayers : numpy.uint8
    maxQm : numpy.uint8


class CsiRsRrcDynPrms(NamedTuple):
    """CSI-RS RRC parameters.

    The RRC parameters for CSI-RS. Used together with PDSCH Tx and
    CSI-RS Tx.

    Args:
        start_prb (numpy.uint16): PRB where this CSI resource starts. Expected value < 273.
        num_prb (numpy.uint16): Number of PRBs across which this CSI resource spans.
            Expected value <= 273 - start_prb.
        freq_alloc (List[int]): Bitmap defining frequency domain allocation. Counting is started
            from least significant bit (first element of the list).
        row (numpy.uint8): Row entry into the CSI resource location table. Valid values 1-18.
        symb_L0 (numpy.uint8): Time domain location L0.
        symb_L1 (numpy.uint8): Time domain location L1.
        freq_density (numpy.uint8): The density field, p and comb offset (for dot5),
            0: dot5 (even RB), 1: dot5 (odd RB), 2: one, 3: three.
        scramb_id (numpy.uint16): Scrambling ID of CSI-RS.
        idx_slot_in_frame (numpy.uint8): Slot index in frame.
        csi_type (numpy.uint8): CSI Type. Only CSI-RS NZP supported currently.

            - 0: TRS
            - 1: CSI-RS NZP
            - 2: CSI-RS ZP

        cdm_type (numpy.uint8): CDM Type.

            - 0: noCDM
            - 1: fd-CDM2
            - 2: cdm4-FD2-TD2
            - 3: cdm8-FD2-TD4

        beta (float): Power scaling factor
        enable_precoding (numpy.uint8): Enable pre-coding for this CSI-RS.
        pwm_prm_idx (numpy.uint16): Index to pre-coding matrix array, i.e., to the
            `precoding_matrices` array given to `CsiRsTx.run`.
    """
    start_prb: numpy.uint16
    num_prb: numpy.uint16
    freq_alloc: List[int]
    row: numpy.uint8
    symb_L0: numpy.uint8
    symb_L1: numpy.uint8
    freq_density: numpy.uint8
    scramb_id: numpy.uint16
    idx_slot_in_frame: numpy.uint8
    csi_type: numpy.uint8
    cdm_type: numpy.uint8
    beta: float
    enable_precoding: numpy.uint8
    pmw_prm_idx: numpy.uint16


class CsiRsCellDynPrms(NamedTuple):
    """CSI-RS cell dynamic parameters.

    Args:
        rrc_dyn_prms (List[CsiRsRrcDynPrms]): List of CSI-RS RRC dynamic parameters, see above.
    """
    rrc_dyn_prms: List[CsiRsRrcDynPrms]


class CsiRsPmwOneLayer(NamedTuple):
    """Precoding matrix for CSI-RS.

    Args:
        precoding_matrix (numpy.ndarray): The actual precoding matrix.
        num_ports (int): Number of ports for this UE.
    """
    precoding_matrix: numpy.ndarray
    num_ports: int


class PmW(NamedTuple):
    """Implement PDSCH precoding matrix.

    This corresponds to the cuPHY `cuphyPmW_t` struct.

    Pre-coding matrix used only if `PdschUePrm.enablePrcdBf` is True.
    Layout of the data is such that `PdschUePrm.nUeLayers` is lower dimension.
    The `nPorts` is the number of columns.
    Memory layout in expected to be in following manner with row-major layout.
    If a transport block has 2 layers and output tensor has 4 ports, the following
    layout should be used:
    --------------------------------------------
    |       | Port 0 | Port 1 | Port 3 | Port 4|
    --------------------------------------------
    | TB L0 |        |        |        |       |
    |-------|-----------------------------------
    | TB L1 |        |        |        |       |
    --------------------------------------------

    Args:
        w (numpy.ndarray): Pre-coding matrix.
        nPorts (numpy.uint8): Number of ports for this UE.
    """
    w : numpy.ndarray  # pylint: disable=invalid-name
    nPorts : numpy.uint8


class PdschCellGrpDynPrm(NamedTuple):
    """Implement PDSCH dynamic cell group parameters.

    This corresponds to the cuPHY `cuphyPdschCellGrpDynPrm_t` struct.

    Args:
        cellPrms (List[PdschCellDynPrm]): List of per-cell dynamic parameters with `nCells`
            elements.
        ueGrpPrms (List[PdschUeGrpPrm]): List of per-UE-group parameters with `nUeGrps` elements.
        uePrms (List[PdschUePrm]): List of per-UE parameters with `nUes` elements.
        cwPrms (List[PdschCwPrm]): List of per-CW parameters with `nCws` elements.
        csiRsPrms (List[CsiRsRrcDynPrms]): List of per-cell CSI-RS parameters with `nCsiRsPrms`
            elements.
        pmwPrms (List[PmW]): List of pre-coding matrices with `nPrecodingMatrices` elements.
    """
    cellPrms : List[PdschCellDynPrm]
    ueGrpPrms : List[PdschUeGrpPrm]
    uePrms : List[PdschUePrm]
    cwPrms : List[PdschCwPrm]
    csiRsPrms : List[CsiRsRrcDynPrms]
    pmwPrms : List[PmW]


class PdschDataIn(NamedTuple):
    """Implement PDSCH data input.

    This corresponds to the cuPHY `cuphyPdschDataIn_t` struct.

    Args:
        tbInput (List[pycuphy.CudaArrayUint8]): A list of transport block input buffers, one
            buffer per cell, indexed by `cellPrmDynIdx`. Each `tbInput` element points to a
            flat array with all TBs for that cell. Currently per-cell TB allocations are
            contiguous, zero-padded to byte boundary. Each element of the flat TB arrays is
            a numpy.uint8 byte. When a cell is in test mode, then the buffer contains the PN23
            (pseudorandom sequence) rather than a payload.
    """
    tbInput : List[pycuphy.CudaArrayUint8]  # type: ignore


class PdschDataOut(NamedTuple):
    """Implement PDSCH data output.

    This corresponds to the cuPHY `cuphyPdschDataOut_t` struct.

    Args:
        dataTx (List[CuPHYTensor]): Array of tensors with each tensor
            (indexed by `cellPrmDynIdx`) representing the transmit slot buffer of a cell
            in the cell group. Each cell's tensor may have a different geometry.
    """
    dataTx : List[CuPHYTensor]


class PdschDynPrms(NamedTuple):
    """Implement PDSCH dynamic parameters.

    This corresponds to the cuPHY `cuphyPdschDynPrms_t` struct.

    Args:
        cuStream (int): CUDA stream on which pipeline is launched.
        procModeBmsk (numpy.uint64): Processing modes (e.g., full-slot processing w/ profile 0
            PDSCH_PROC_MODE_FULL_SLOT|PDSCH_PROC_MODE_PROFILE0).
        cellGrpDynPrm (PdschCellGrpDynPrm): Cell group configuration parameters. Each pipeline
            will process a single cell-group.
        dataIn (PdschDataIn): PDSCH data input.
        tbCRCDataIn (PdschDataIn): Optional TB CRCs.
        dataOut (PdschDataOut): PDSCH data output that will contain `cellGrpDynPrm.nCells`
            tensors.
    """
    cuStream : int
    procModeBmsk : numpy.uint64
    cellGrpDynPrm : PdschCellGrpDynPrm
    dataIn : PdschDataIn
    tbCRCDataIn : PdschDataIn
    dataOut : PdschDataOut


class PuschStatDbgPrms(NamedTuple):
    """Implement cuPHY PUSCH Rx static debug parameters.

    This corresponds to the cuPHY `cuphyPuschStatDbgPrms_t` struct.

    Args:
        outFileName (str): Output file capturing pipeline intermediate states. No capture if None.
        descrmOn (numpy.uint8) : Descrambling enable/disable.
        enableApiLogging (numpy.uint8): Control the API logging of PUSCH static parameters.
        forcedNumCsi2Bits (nunmpy.uint16): If > 0 cuPHY assumes all csi2 UCIs have
            forcedNumCsi2Bits bits.
    """
    outFileName : str
    descrmOn : numpy.uint8
    enableApiLogging : numpy.uint8
    forcedNumCsi2Bits: numpy.uint16


class PuschStatPrms(NamedTuple):
    """Implement cuPHY PUSCH cell static parameters.

    This corresponds to the cuPHY `cuphyPuschStatPrms_t` struct.

    Args:
        outInfo (List[cuPHYTracker]): Pointer to cuPHY tracker.
        WFreq (numpy.ndarray): Channel estimation filter for wide bandwidth.
        WFreq4 (numpy.ndarray): Channel estimation filter for medium bandwidth.
        WFreqSmall (numpy.ndarray): Channel estimation filter for small bandwidth.
        ShiftSeq (numpy.ndarray): Channel estimation shift sequence for nominal bandwidth.
        UnShiftSeq (numpy.ndarray): Channel estimation unshift sequence for nominal bandwidth.
        ShiftSeq4 (numpy.ndarray): Channel estimation shift sequence for medium bandwidth.
        UnShiftSeq4 (numpy.ndarray): Channel estimation unshift sequence for medium bandwidth.
        enableCfoCorrection (numpy.uint8): Carrier frequency offset estimation/correction flag.

            - 0: Disable.
            - 1: Enable.

        enableWeightedAverageCfo (numpy.uint8): CFO weighted average estimation flag.

            - 0: Disable.
            - 1: Enable.

        enableToEstimation (numpy.uint8): Time offset estimation flag.

            - 0: Disable.
            - 1: Enable.

        enablePuschTdi (numpy.uint8): Time domain interpolation flag.

            - 0: Disable.
            - 1: Enable.

        enableDftSOfdm (numpy.uint8): Global DFT-S-OFDM enabling flag.

            - 0: Disable.
            - 1: Enable.

        enableTbSizeCheck (numpy.uint8): Global PUSCH tbSizeCheck enabling flag.

            - 0: Disable.
            - 1: Enable.

        enableUlRxBf (numpy.uint8): Flag for UL Rx beamforming.

            - 0 - Disable.
            - 1 - Enable.

        enableDebugEqOutput (numpy.uint8): Debugging output of post-equalized receive data samples
            flag.

            - 0 - Disable.
            - 1 - Enable.

        ldpcnIterations (numpy.uint8): Number of LDPC decoder iterations.
        ldpcEarlyTermination (numpy.uint8): LDPC decoder early termination flag.

            - 0: Run `ldpcnInterations` always.
            - 1: Terminate early on passing CRC.

        ldpcUseHalf (numpy.uint8): LDPC use FP16 flag.

            - 0: Use FP32 LLRs.
            - 1: Use FP16 LLRs.

        ldpcAlgoIndex (numpy.uint8): LDPC Decoder algorithm index. See cuPHY documentation for
            a list of algorithms.
        ldpcFlags (numpy.uint8): LDPC decoder configuration flags. See cuPHY documentation for
            flags.
        ldpcKernelLaunch (PuschLdpcKernelLaunch): LDPC launch configuration flag. See cuPHY
            documentation for kernel launch.
        ldpcMaxNumItrAlgo (LdpcMaxItrAlgoType): Algorithm to use for determining the number of
            LDPC iterations.
        fixedMaxNumLdpcItrs (numpy.uint8): Used when ldpcMaxNumItrAlgo = FIXED.
        ldpcClampValue (numpy.float32): 128.0.
        polarDcdrListSz (numpy.uint8): Polar decoder list size. Default size for PUSCH
            to be set to 1.
        nMaxTbPerNode (numpy.uint8): Maximum number of transport blocks per node.
        chEstAlgo (PuschChEstAlgoType): Channel estimation algorithm.

            - 0 - Legacy MMSE
            - 1 - Multi-stage MMSE with delay estimation
            - 2 - RKHS not supported by pyAerial yet.
            - 3 - LS channel estimation only

        enablePerPrgChEst (numpy.uint8): Per-PRG CHEST enable/disable flag

            - 0 - Disable.
            - 1 - Enable.

        eqCoeffAlgo (PuschEqCoefAlgoType): PUSCH equalizer algorithm.

            - 0 - ZF.
            - 1 - MMSE (default).
            - 2 - MMSE-IRC.

        enableRssiMeasurement (numpy.uint8): Flag for RSSI measurement.

            - 0 - Disabled.
            - 1 - Enabled.

        enableSinrMeasurement (numpy.uint8): Flag for SINR measurement.

            - 0 - Disabled.
            - 1 - Enabled.

        enableCsiP2Fapiv3 (numpy.uint8): FAPIv3 method for computing CSI-P2 size flag.

            - 0 - Disabled.
            - 1 - Enabled.

        stream_priority (int): CUDA stream priority for internal PUSCH streams-pool.
            Should match the priority of CUDA stream passed in `cuphyCreatePuschRx()`.
        nMaxCells (numpy.uint16): Total # of cell configurations supported by the pipeline during
            its lifetime. Maximum # of cells scheduled in a slot. Out of `nMaxCells`, the
            `nMaxCellsPerSlot` most resource hungry cells are used for resource provisioning
            purposes.
        nMaxCellsPerSlot (numpy.uint16): Must be <= `nMaxCells`.
        cellStatPrms (List[CellStatPrm]): Static cell parameters common to all channels.
        nMaxTbs (numpy.uint32): Maximum number of transport blocks that will be supported by
            PuschRx object.
        nMaxCbsPerTb (numpy.uint32) : Maximum number of code blocks per transport block that
            will be supported by PuschRx object.
        nMaxTotCbs (numpy.uint32): Total number of code blocks (sum of # code blocks across all
            transport blocks) that will be supported by PuschRx object.
        nMaxRx (numpy.uint32): Maximum number of Rx antennas that will be supported by PuschRx
            object.
        nMaxPrb (numpy.uint32): Maximum number of PRBs that will be supported by PuschRx object.
        nMaxLdpcHetConfigs (numpy.uint32): The maximum # of LDPC heterogeneous configs.
        dbg (PuschStatDbgPrms): Debug parameters.
        enableDeviceGraphLaunch (numpy.uint8): Static flag to control device graph launch in PUSCH
        enableEarlyHarq (numpy.uint8): Static flag to control construction of early-HARQ related
            members in PUSCH.
        earlyHarqProcNodePriority (numpy.int32): Elevated priority used for early-HARQ processing
            nodes in graphs mode. The priority values are same as CUDA stream priorities with lower
            numbers imply greater priorities.
        workCancelMode (PuschWorkCancelMode): Flag to control work cancellation mode in PUSCH
        enableBatchedMemcpy (numpy.uint8): Flag to control batched memory copy in PUSCH.
        chestFactorySettingsFilename (Optional[str]): Path to the chest factory settings file.
    """
    outInfo : List[CuPHYTracker]
    WFreq : numpy.ndarray
    WFreq4 : numpy.ndarray
    WFreqSmall : numpy.ndarray
    ShiftSeq : numpy.ndarray
    UnShiftSeq : numpy.ndarray
    ShiftSeq4 : numpy.ndarray
    UnShiftSeq4 : numpy.ndarray
    enableCfoCorrection : numpy.uint8
    enableWeightedAverageCfo: numpy.uint8
    enableToEstimation : numpy.uint8
    enablePuschTdi : numpy.uint8
    enableDftSOfdm : numpy.uint8
    enableTbSizeCheck: numpy.uint8
    enableUlRxBf : numpy.uint8
    enableDebugEqOutput : numpy.uint8
    ldpcnIterations : numpy.uint8
    ldpcEarlyTermination : numpy.uint8
    ldpcUseHalf : numpy.uint8
    ldpcAlgoIndex : numpy.uint8
    ldpcFlags : numpy.uint8
    ldpcKernelLaunch : PuschLdpcKernelLaunch
    ldpcMaxNumItrAlgo: LdpcMaxItrAlgoType
    fixedMaxNumLdpcItrs: numpy.uint8
    ldpcClampValue: numpy.float32
    polarDcdrListSz : numpy.uint8
    nMaxTbPerNode : numpy.uint8
    chEstAlgo: PuschChEstAlgoType
    enablePerPrgChEst: numpy.uint8
    eqCoeffAlgo : PuschEqCoefAlgoType
    enableRssiMeasurement : numpy.uint8
    enableSinrMeasurement : numpy.uint8
    enableCsiP2Fapiv3 : numpy.uint8
    stream_priority : int
    nMaxCells : numpy.uint16
    nMaxCellsPerSlot : numpy.uint16
    cellStatPrms : List[CellStatPrm]
    nMaxTbs : numpy.uint32
    nMaxCbsPerTb : numpy.uint32
    nMaxTotCbs : numpy.uint32
    nMaxRx : numpy.uint32
    nMaxPrb : numpy.uint32
    nMaxLdpcHetConfigs: numpy.uint32
    dbg : PuschStatDbgPrms
    enableDeviceGraphLaunch : numpy.uint8
    enableEarlyHarq : numpy.uint8
    earlyHarqProcNodePriority : numpy.int32
    workCancelMode : PuschWorkCancelMode
    enableBatchedMemcpy : numpy.uint8
    chestFactorySettingsFilename : Optional[str]


class PuschCellDynPrm(NamedTuple):
    """Implement cuPHY PUSCH cell dynamic parameters.

    This corresponds to cuPHY `cuphyPuschCellDynPrm_t` struct.

    Args:
        cellPrmStatIdx: Index to cell-static parameter information.
        cellPrmDynIdx: Index to cell-dynamic parameter information.
        slotNum: Slot number.
    """
    cellPrmStatIdx : numpy.uint16
    cellPrmDynIdx : numpy.uint16
    slotNum : numpy.uint16


class PuschDmrsPrm(NamedTuple):
    """Implement cuPHY PUSCH DMRS parameters.

    This corresponds to cuPHY `cuphyPuschDmrsPrm_t` struct.

    Args:
        dmrsAddlnPos (numpy.uint8): Number of additional DMRS positions.
        dmrsMaxLen (numpy.uint8): DMRS max length.
        numDmrsCdmGrpsNoData (numpy.uint8): Number of DM-RS CDM groups without data
            [TS38.212 sec 7.3.1.1, TS38.214 Table 4.1-1]. Value: 1 -> 3.
        dmrsScrmId (numpy.uint16): UL-DMRS-Scrambling-ID [TS38.211, sec 6.4.1.1.1].
            Value: 0 -> 65535.
    """
    dmrsAddlnPos : numpy.uint8
    dmrsMaxLen : numpy.uint8
    numDmrsCdmGrpsNoData : numpy.uint8
    dmrsScrmId : numpy.uint16


class PuschUeGrpPrm(NamedTuple):
    """Implement cuPHY PUSCH co-scheduled UE group dynamic parameters.

    This corresponds to cuPHY `cuphyPuschUeGrpPrm_t` struct.

    Args:
        cellPrmIdx (int): UE group's parent cell dynamic parameters index.
        dmrsDynPrm (PuschDmrsPrm): DMRS information.
        startPrb (numpy.uint16): The starting resource block within the BWP for this PUSCH.
            Value: 0 -> 274.
        nPrb (numpy.uint16): The number of resource block within for this PUSCH. Value: 1 -> 275.
        prgSize (numpy.uint16): PRG size for channel estimation.
        nUplinkStreams (numpy.uint16): The number of active streams for this PUSCH. Value: 1 -> 8.
        puschStartSym (numpy.uint8): Start symbol index of PUSCH mapping from the start of the
            slot, S. [TS38.214, Table 6.1.2.1-1]. Value: 0 -> 13.
        nPuschSym (numpy.uint8): PUSCH duration in symbols, L. [TS38.214, Table 6.1.2.1-1].
            Value: 1 -> 14.
        dmrsSymLocBmsk (numpy.uint16): DMRS location bitmask (LSB 14 bits).
            PUSCH symbol locations derived from dmrsSymLocBmsk.
            Bit i is "1" if symbol i is DMRS.
            For example if there are DMRS are symbols 2 and 3, then:
            dmrsSymLocBmsk = 0000 0000 0000 1100.
        rssiSymLocBmsk (numpy.uint16): Symbol location bitmask for RSSI measurement (LSB 14 bits).
            Bit i is "1" if symbol i needs be to measured, 0 disables RSSI calculation.
            For example to measure RSSI on DMRS symbols 2, 6 and 9, use:
            rssiSymLocBmsk = 0000 0010 0100 0100.
        uePrmIdxs List[numpy.uint16]: List of UE indices.
    """
    cellPrmIdx : int
    dmrsDynPrm : PuschDmrsPrm
    startPrb : numpy.uint16
    nPrb : numpy.uint16
    prgSize : numpy.uint16
    nUplinkStreams : numpy.uint16
    puschStartSym : numpy.uint8
    nPuschSym : numpy.uint8
    dmrsSymLocBmsk : numpy.uint16
    rssiSymLocBmsk : numpy.uint16
    uePrmIdxs : List[numpy.uint16]


class PuschUePrm(NamedTuple):
    """Implement cuPHY PUSCH UE dynamic parameters.

    This corresponds to cuPHY `cuphyPuschUePrm_t` struct.

    Note: DFT-S-OFDM parameters are missing as DFT-S-OFDM is not yet
    supported by pycuphy. Also UCI parameters are missing.

    Args:
        pduBitmap (numpy.uint16):

            - Bit 0 indicates if data present.
            - Bit 1 indicates if UCI present.
            - Bit 2 indicates if PTRS present.
            - Bit 3 indicates DFT-S transmission.
            - Bit 4 indicates if SCH data present.
            - Bit 5 indicates if CSI-P2 present.

        ueGrpIdx (numpy.uint16): Index to parent UE Group.
        enableTfPrcd (numpy.uint8): DFT-S-OFDM enabling per UE. For now always disabled.
        scid (numpy.uint8): DMRS sequence initialization [TS38.211, sec 7.4.1.1.2].
            Should match what is sent in DCI 1_1, otherwise set to 0. Value : 0 -> 1.
        dmrsPortBmsk (numpy.uint16): Use to map DMRS port to fOCC/DMRS-grid/tOCC.
        nlAbove16 (numpy.uint8): Number of layers above 16. Value: 0 -> 1.
            Not supported for PUSCH. Keeping it in parity with pdsch.
            Will always be set to 0.
        mcsTable (numpy.uint8): MCS (Modulation and Coding Scheme) Table Id.
            [TS38.214, sec 6.1.4.1].

            - 0: notqam256. Table 5.1.3.1-1.
            - 1: qam256. Table 5.1.3.1-2.
            - 2: qam64LowSE. Table 5.1.3.1-3.
            - 3: notqam256-withTransformPrecoding [TS38.214, table 6.1.4.1-1].
            - 4: qam64LowSE-withTransformPrecoding [TS38.214, table 6.1.4.1-2].

        mcsIndex (numpy.uint8): MCS index within the mcsTableIndex table. [TS38.214, sec 6.1.4.1].
            Value: 0 -> 31.
        targetCodeRate (numpy.uint16): Target coding rate [TS38.214 sec 6.1.4.1].
            This is the number of information bits per 1024 coded bits expressed in 0.1 bit units.
        qamModOrder (numpy.uint8): QAM modulation [TS38.214 sec 6.1.4.1].
        TBSize (numpy.uint32): TBSize in bytes provided by L2 based on FAPI 10.04.
        rv (numpy.uint8): Redundancy version index [TS38.214, sec 6.1.4],
            should match value sent in DCI. Value: 0 -> 3.
        rnti (numpy.uint16): The RNTI used for identifying the UE when receiving the PDU.
            RNTI == Radio Network Temporary Identifier. Value: 1 -> 65535.
        dataScramId (numpy.uint16): dataScramblingIdentityPusch [TS38.211, sec 6.3.1.1].
            Value: 0 -> 65535.
        nUeLayers (numpy.uint8): Number of layers [TS38.211, sec 6.3.1.3].
            Value: 1 -> 4.
        ndi (numpy.uint8): Indicates if this new data or a retransmission
            [TS38.212, sec 7.3.1.1]. Value:

            - 0: Retransmission.
            - 1: New data.

        harqProcessId (numpy.uint8): HARQ process number [TS38.212, sec 7.3.1.1].
        i_lbrm (numpy.uint8) : 0 = Do not use LBRM. 1 = Use LBRM per 38.212 5.4.2.1 and 6.2.5.
        maxLayers (numpy.uint8): Number of layers used for LBRM TB size computation (at most 4).
        maxQm (numpy.uint8): Modulation order used for LBRM TB size computation.
            Value: 6 or 8.
        n_PRB_LBRM (numpy.uint16): number of PRBs used for LBRM TB size computation.
            Possible values: {32, 66, 107, 135, 162, 217, 273}.
    """
    pduBitmap : numpy.uint16
    ueGrpIdx : numpy.uint16
    enableTfPrcd: numpy.uint8
    scid : numpy.uint8
    dmrsPortBmsk : numpy.uint16
    nlAbove16 : numpy.uint8
    mcsTable : numpy.uint8
    mcsIndex : numpy.uint8
    TBSize: numpy.uint32
    targetCodeRate : numpy.uint16
    qamModOrder : numpy.uint8
    rv : numpy.uint8
    rnti : numpy.uint16
    dataScramId : numpy.uint16
    nUeLayers : numpy.uint8
    ndi : numpy.uint8
    harqProcessId : numpy.uint8
    i_lbrm : numpy.uint8
    maxLayers : numpy.uint8
    maxQm : numpy.uint8
    n_PRB_LBRM : numpy.uint16


class PuschCellGrpDynPrm(NamedTuple):
    """Implement cuPHY PUSCH cell group dynamic parameters.

    This corresponds to cuPHY `cuphyPuschCellGrpDynPrm_t` struct.

    Args:
        cellPrms (List[PuschCellDynPrm]): List of cell dynamic parameters, one entry per cell.
        ueGrpPrms (List[PuschUeGrpPrm]): List of UE group dynamic parameters, one entry per UE
            group.
        uePrms (List[PuschUePrm]): List of UE dynamic parameters, one entry per UE.
    """
    cellPrms : List[PuschCellDynPrm]
    ueGrpPrms : List[PuschUeGrpPrm]
    uePrms : List[PuschUePrm]


class PuschDataIn(NamedTuple):
    """Implement PUSCH data input.

    This corresponds to cuPHY `cuphyPuschDataIn_t` struct.

    Args:
        tDataRx (List[numpy.ndarray]): List of tensors with each tensor (indexed by
            `cellPrmDynIdx`) representing the receive slot buffer of a cell in the cell group.
            Each cell's tensor may have a different geometry.
    """
    tDataRx : List[numpy.ndarray]


class PuschDataOut(NamedTuple):
    """Implement PUSCH data output.

    This corresponds to cuPHY `cuphyPuschDataOut_t` struct.

    Args:
        harqBufferSizeInBytes (numpy.ndarray): HARQ buffer sizes, returned during setup
            phase 1.
        totNumTbs (numpy.ndarray):
        totNumCbs (numpy.ndarray):
        totNumPayloadBytes (numpy.ndarray):
        totNumUciSegs (numpy.ndarray):

        cbCrcs (numpy.ndarray): Array of CB CRCs.
        tbCrcs (numpy.ndarray): Array of TB CRCs.
        tbPayloads (numpy.ndarray): Array of TB payloads.
        uciPayloads (numpy.ndarray): TODO
        uciCrcFlags (numpy.ndarray): TODO
        numCsi2Bits (numpy.ndarray): TODO
        startOffsetsCbCrc (numpy.ndarray): nUes offsets providing start offset of UE CB-CRCs within
            `cbCrcs`. The UE ordering is identical to input UE ordering.
        startOffsetsTbCrc (numpy.ndarray): nUes offsets providing start offset of UE TB-CRCs within
            `tbCrcs`. The UE ordering is identical to input UE ordering.
        startOffsetsTbPayload (numpy.ndarray): nUes offsets providing start offset of UE TB-payload
            within `tbPayloads`. The UE ordering is identical to input UE ordering.
        taEsts (numpy.ndarray): Array of nUes estimates in microseconds. UE ordering identical
            to input UE ordering.
        rssi (numpy.ndarray): Array of nUeGrps estimates in dB. Per UE group total power
            (signal + noise + interference) averaged over allocated PRBs, DMRS additional positions
            and summed over Rx antenna.
        rsrp (numpy.ndarray): Array of nUes RSRP estimates in dB. Per UE signal power averaged over
            allocated PRBs, DMRS additional positions, Rx antenna and summed over layers.
        noiseVarPreEq (numpy.ndarray): Array of nUeGrps pre-equalizer noise variance estimates
            in dB.
        noiseVarPostEq (numpy.ndarray): Array of nUes post equalizer noise variance estimates
            in dB.
        sinrPreEq (numpy.ndarray): Array of nUes pre-equalizer SINR estimates in dB.
        sinrPostEq (numpy.ndarray): Array of nUes post-equalizer estimates SINR in dB.
        cfoHz (numpy.ndarray): Array of nUEs carrier frequency offsets in Hz.
        HarqDetectionStatus (numpy.ndarray): Value:

            - 1 = CRC Pass
            - 2 = CRC Failure
            - 3 = DTX
            - 4 = No DTX (indicates UCI detection).

            Note that FAPI also defined value 5 to be "DTX not checked", which is not considered in
            cuPHY since DTX detection is present.
        CsiP1DetectionStatus (numpy.ndarray): Value:

            - 1 = CRC Pass
            - 2 = CRC Failure
            - 3 = DTX
            - 4 = No DTX (indicates UCI detection).

            Note that FAPI also defined value 5 to be "DTX not checked", which is not considered in
            cuPHY since DTX detection is present.
        CsiP2DetectionStatus (numpy.ndarray): Value:

            - 1 = CRC Pass
            - 2 = CRC Failure
            - 3 = DTX
            - 4 = No DTX (indicates UCI detection).

            Note that FAPI also defined value 5 to be "DTX not checked", which is not considered in
            cuPHY since DTX detection is present.
    """
    harqBufferSizeInBytes : numpy.ndarray
    totNumTbs : numpy.ndarray  # length-1 numpy.uint32
    totNumCbs : numpy.ndarray  # length-1 numpy.uint32
    totNumPayloadBytes : numpy.ndarray  # length-1 numpy.uint32
    totNumUciSegs : numpy.ndarray  # length-1 numpy.uint16
    cbCrcs : numpy.ndarray
    tbCrcs : numpy.ndarray
    tbPayloads : numpy.ndarray
    uciPayloads : numpy.ndarray
    uciCrcFlags : numpy.ndarray
    numCsi2Bits : numpy.ndarray
    startOffsetsCbCrc : numpy.ndarray
    startOffsetsTbCrc : numpy.ndarray
    startOffsetsTbPayload : numpy.ndarray
    taEsts : numpy.ndarray
    rssi : numpy.ndarray
    rsrp : numpy.ndarray
    noiseVarPreEq : numpy.ndarray
    noiseVarPostEq : numpy.ndarray
    sinrPreEq : numpy.ndarray
    sinrPostEq : numpy.ndarray
    cfoHz : numpy.ndarray
    HarqDetectionStatus : numpy.ndarray
    CsiP1DetectionStatus : numpy.ndarray
    CsiP2DetectionStatus : numpy.ndarray
    preEarlyHarqWaitStatus : numpy.ndarray
    postEarlyHarqWaitStatus : numpy.ndarray


class PuschDataInOut(NamedTuple):
    """Implement cuPHY PUSCH data input/output.

    This corresponds to cuPHY `cuphyPuschDataInOut_t` struct.

    Args:
        harqBuffersInOut (List[int]): Array of In/Out HARQ buffers (pointers to GPU memory).
            The In/Out HARQ buffers will be read or written depending on NDI and TB CRC pass result.
            The In/Out HARQ buffers themselves are located in GPU memory.
            The “array of pointers” must be read-able from a GPU kernel (handled at the C++ binding
            side).
    """
    harqBuffersInOut : List[int]


class PuschDynDbgPrms(NamedTuple):
    """Implement PUSCH channel dynamic debug parameters.

    This corresponds to cuPHY `cuphyPuschDynDbgPrms_t` struct.

    Args:
        enableApiLogging (numpy.uint8): control the API logging of PUSCH dynamic parameters
    """
    enableApiLogging : numpy.uint8


class PuschStatusOut(NamedTuple):
    """Implement PUSCH output status.

    This corresponds to cuPHY `cuphyPuschStatusOut_t` struct.

    Args:
        status (PuschStatusType): cuPHY PUSCH status after setup call. Currently used to
            highlight if CUPHY_PUSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB occurred.
        cellPrmStatIdx (numpy.uint16):
        ueIdx (numpy.uint16):
    """
    status : PuschStatusType
    cellPrmStatIdx : numpy.uint16
    ueIdx : numpy.uint16


class PuschDynPrms(NamedTuple):
    """Implement cuPHY PUSCH pipeline dynamic parameters.

    This corresponds to cuPHY `cuphyPuschDynPrms_t` struct.

    Args:
        phase1Stream (int): CUDA stream on which pipeline is launched (phase 1).
        phase2Stream (int): CUDA stream on which pipeline is launched (phase 2).
        setupPhase (PuschSetupPhase): Setup phase.
        procModeBmsk (numpy.uint64): Processing modes bitmask.
        waitTimeOutPreEarlyHarqUs (numpy.uint16): Time-out threshold for wait kernel
            prior to starting early HARQ processing.
        waitTimeOutPostEarlyHarqUs (numpy.uint16): Time-out threshold for wait kernel
            after finishing early HARQ processing.
        cellGrpDynPrm (PuschCellGrpDynPrm): Cell group dynamic parameters.
        dataIn (PuschDataIn): Input data parameters.
        dataOut (PuschDataOut): Output data parameters.
        dataInOut (PuschDataInOut): Input/output data parameters.
        cpuCopyOn (numpy.uint8): Flag. Indicates if reciever output copied to CPU.
        statusOut (PuschStatusOut): PUSCH status.
        dbg (PuschDynDbgPrms): PUSCH debug parameters.
    """
    phase1Stream : int
    phase2Stream : int
    setupPhase : PuschSetupPhase
    procModeBmsk : numpy.uint64
    waitTimeOutPreEarlyHarqUs : numpy.uint16
    waitTimeOutPostEarlyHarqUs : numpy.uint16
    cellGrpDynPrm : PuschCellGrpDynPrm
    dataIn : PuschDataIn
    dataOut : PuschDataOut
    dataInOut : PuschDataInOut
    cpuCopyOn : numpy.uint8
    statusOut : PuschStatusOut
    dbg : PuschDynDbgPrms
