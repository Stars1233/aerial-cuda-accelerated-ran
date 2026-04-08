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

#if !defined(CUPHY_HDF5_HPP_INCLUDED_)
#define CUPHY_HDF5_HPP_INCLUDED_

#include "cuphy_hdf5.h"
#include "hdf5hpp.hpp"
#include "cuphy.hpp"
#include <array>
#include <exception>
#include <cstring>

namespace cuphy
{

#define PDSCH_PRINT_CONFIG 0

////////////////////////////////////////////////////////////////////////
// cuphy::cuphyHDF5_exception
// Exception class for errors from the cuphy_hdf5 library
class cuphyHDF5_exception : public std::exception //
{
public:
    cuphyHDF5_exception(cuphyHDF5Status_t s) :
        status_(s) {}
    virtual ~cuphyHDF5_exception() = default;
    virtual const char* what() const noexcept { return cuphyHDF5GetErrorString(status_); }
    cuphyHDF5Status_t status() const { return status_; }
private:
    cuphyHDF5Status_t status_;
};

////////////////////////////////////////////////////////////////////////
// cuphy::cuphyHDF5_struct
class cuphyHDF5_struct
{
public:
    cuphyHDF5_struct(cuphyHDF5Struct_t s = nullptr) : s_(s) {}
    cuphyHDF5_struct(cuphyHDF5_struct&& hdf5Struct) :
        s_(hdf5Struct.s_)
    {
        hdf5Struct.s_ = nullptr;
    }
    ~cuphyHDF5_struct() { if(s_) cuphyHDF5ReleaseStruct(s_); }
    cuphyHDF5_struct(const cuphyHDF5_struct&)             = delete;
    cuphyHDF5_struct&  operator=(const cuphyHDF5_struct&) = delete;
    cuphyHDF5_struct&  operator=(cuphyHDF5_struct&& hdf5Struct)
    {
        if(s_) cuphyHDF5ReleaseStruct(s_);
        s_ = hdf5Struct.s_;
        hdf5Struct.s_ = nullptr;
        return *this;
    }
    cuphyVariant_t get_value(const char* name) const
    {
        cuphyVariant_t v;
        cuphyHDF5Status_t status = cuphyHDF5GetStructScalar(&v,
                                                            s_,
                                                            name,
                                                            CUPHY_VOID);
        if(CUPHYHDF5_STATUS_SUCCESS != status)
        {
            throw cuphyHDF5_exception(status);
        }
        return v;
    }
    template <typename T>
    T get_value_as(const char* name) const
    {
        variant v;
        cuphyHDF5Status_t status = cuphyHDF5GetStructScalar(&v,
                                                            s_,
                                                            name,
                                                            type_to_cuphy_type<T>::value);
        if(CUPHYHDF5_STATUS_SUCCESS != status)
        {
            throw cuphyHDF5_exception(status);
        }
        return v.as<T>();
    }
private:
    cuphyHDF5Struct_t s_;
};

////////////////////////////////////////////////////////////////////////
// cuphy::get_HDF5_struct()
inline
cuphyHDF5_struct get_HDF5_struct(hdf5hpp::hdf5_dataset& dset,
                                 size_t                 numDim = 0,
                                 const hsize_t*         coords = nullptr)
{
    cuphyHDF5Struct_t s      = nullptr;
    cuphyHDF5Status_t status = cuphyHDF5GetStruct(dset.id(), numDim, coords, &s);
    if(CUPHYHDF5_STATUS_SUCCESS != status)
    {
        throw cuphyHDF5_exception(status);
    }
    return cuphyHDF5_struct(s);
}

////////////////////////////////////////////////////////////////////////
// cuphy::get_HDF5_struct()
inline
cuphyHDF5_struct get_HDF5_struct(hdf5hpp::hdf5_file& f,
                                 const char*         name,
                                 size_t              numDim = 0,
                                 const hsize_t*      coords = nullptr)
{
    hdf5hpp::hdf5_dataset dset = f.open_dataset(name);
    return get_HDF5_struct(dset, numDim, coords);
}

////////////////////////////////////////////////////////////////////////
// cuphy::get_HDF5_struct_index()
inline
cuphyHDF5_struct get_HDF5_struct_index(hdf5hpp::hdf5_dataset& dset,
                                       hsize_t                idx)
{
    const size_t numDim = 1;
    return get_HDF5_struct(dset, numDim, &idx);
}

////////////////////////////////////////////////////////////////////////
// cuphy::get_HDF5_struct_index()
inline
cuphyHDF5_struct get_HDF5_struct_index(hdf5hpp::hdf5_dataset& dset,
                                       hsize_t                idx0,
                                       hsize_t                idx1)
{
    const size_t numDim = 2;
    const hsize_t coords[2] = {idx0, idx1};
    return get_HDF5_struct(dset, numDim, coords);
}

// clang-format off
inline cuphy::tensor_info get_HDF5_dataset_info(const hdf5hpp::hdf5_dataset& dset)
{
    cuphyDataType_t                dtype;
    cuphy::vec<int, CUPHY_DIM_MAX> dim;
    int                            rank;
    cuphyHDF5Status_t              s = cuphyHDF5GetDatasetInfo(dset.id(),
                                                               CUPHY_DIM_MAX,
                                                               &dtype,
                                                               &rank,
                                                               dim.begin());
    if(CUPHYHDF5_STATUS_SUCCESS != s)
    {
        throw cuphyHDF5_exception(s);
    }
    return cuphy::tensor_info(dtype, cuphy::tensor_layout(rank, dim.begin(), nullptr));
}
// clang-format on


template <class TTensor>
inline void read_HDF5_dataset(TTensor&                     t,
                              const hdf5hpp::hdf5_dataset& dset,
                              cudaStream_t                 strm = 0)
{
    cuphyHDF5Status_t s = cuphyHDF5ReadDataset(t.desc().handle(),
                                               t.addr(),
                                               dset.id(),
                                               strm);
    if(CUPHYHDF5_STATUS_SUCCESS != s)
    {
        throw cuphyHDF5_exception(s);
    }
}


template <class TTensor>
inline void write_HDF5_dataset(hdf5hpp::hdf5_file& f,
                               const TTensor&      t,
                               const cuphy::tensor_desc& desc, 
                               const char*         name,
                               cudaStream_t        strm = 0)
{
    cuphyHDF5Status_t s = cuphyHDF5WriteDataset(f.id(),
                                                name,
                                                desc.handle(),
                                                t.addr(),
                                                strm);
    if(CUPHYHDF5_STATUS_SUCCESS != s)
    {
        throw cuphyHDF5_exception(s);
    }
}

template <class TTensor>
inline void write_HDF5_dataset(hdf5hpp::hdf5_file& f,
                               const TTensor&      t,
                               const char*         name,
                               cudaStream_t        strm = 0)
{
    cuphyHDF5Status_t s = cuphyHDF5WriteDataset(f.id(),
                                                name,
                                                t.desc().handle(),
                                                t.addr(),
                                                strm);
    if(CUPHYHDF5_STATUS_SUCCESS != s)
    {
        throw cuphyHDF5_exception(s);
    }
}

inline void write_HDF5_dataset(hdf5hpp::hdf5_file& f, const char* name, const cuphyDataType_t type, const int32_t size, void* pData)      
{
    cuphyHDF5Status_t s = cuphyHDF5WriteDatasetFromCPU(f.id(), name, type, size, pData);
    if(CUPHYHDF5_STATUS_SUCCESS != s)
    {
        throw cuphyHDF5_exception(s);
    }
}


// Write tensor defined by cuphyTensorPrm_t to H5 file
inline void write_HDF5_dataset(hdf5hpp::hdf5_file&     f,
                               const cuphyTensorPrm_t& prm,                               
                               const char*             name,
                               cudaStream_t            strm = 0)
{
    cuphyHDF5Status_t s = cuphyHDF5WriteDataset(f.id(),
                                                name,
                                                prm.desc,
                                                prm.pAddr,
                                                strm);
    if(CUPHYHDF5_STATUS_SUCCESS != s)
    {
        throw cuphyHDF5_exception(s);
    }
} 

// Allocate and return a tensor, initializing data from the given HDF5 dataset
template <class TAlloc = device_alloc>
inline tensor<TAlloc> tensor_from_dataset(const hdf5hpp::hdf5_dataset& dset,
                                          cuphy::tensor_flags          tensorDescFlags = tensor_flags::align_default,
                                          cudaStream_t                 strm            = 0)
{
    tensor<TAlloc> t(get_HDF5_dataset_info(dset), tensorDescFlags); 
    read_HDF5_dataset(t, dset, strm);
    return t;
}

// Allocate and return a tensor, initializing data from the given HDF5 dataset
// (with conversion)
template <class TAlloc = device_alloc>
inline tensor<TAlloc> tensor_from_dataset(const hdf5hpp::hdf5_dataset& dset,
                                          cuphyDataType_t              convertToType,
                                          cuphy::tensor_flags          tensorDescFlags = tensor_flags::align_default,
                                          cudaStream_t                 strm            = 0)
{
    // Create a tensor info with the requested conversion type, but the original layout
    tensor<TAlloc> t(tensor_info(convertToType, get_HDF5_dataset_info(dset).layout()),
                     tensorDescFlags); 
    read_HDF5_dataset(t, dset, strm);
    return t;
}

// Allocate and return a tensor, initializing data from the given HDF5 dataset
template <cuphyDataType_t TType, class TAlloc = device_alloc>
inline typed_tensor<TType, TAlloc> typed_tensor_from_dataset(const hdf5hpp::hdf5_dataset& dset,
                                                             cuphy::tensor_flags          tensorDescFlags = tensor_flags::align_default,
                                                             cudaStream_t                 strm            = 0)
{
    typed_tensor<TType, TAlloc> t(get_HDF5_dataset_info(dset).layout(), tensorDescFlags); 
    read_HDF5_dataset(t, dset, strm);
    return t;
}

inline void disable_hdf5_error_print()
{
    H5Eset_auto(H5E_DEFAULT, (H5E_auto2_t)nullptr, nullptr);
}

inline void enable_hdf5_error_print()
{
    H5Eset_auto(H5E_DEFAULT, (H5E_auto2_t)H5Eprint, stderr);
}

inline gnb_pars gnb_pars_from_dataset_elem(const hdf5hpp::hdf5_dataset_elem& dset_elem)
{
    gnb_pars pars;
    std::memset(&pars, 0, sizeof(pars));
    pars.fc                   = dset_elem["fc"].as<uint32_t>();
    pars.mu                   = dset_elem["mu"].as<uint32_t>();
    pars.nRx                  = dset_elem["nRx"].as<uint32_t>();
    pars.nPrb                 = dset_elem["nPrb"].as<uint32_t>();
    pars.cellId               = dset_elem["cellId"].as<uint32_t>();
    pars.slotNumber           = dset_elem["slotNumber"].as<uint32_t>();
    pars.Nf                   = dset_elem["Nf"].as<uint32_t>();
    pars.Nt                   = dset_elem["Nt"].as<uint32_t>();
    pars.df                   = dset_elem["df"].as<uint32_t>();
    pars.dt                   = dset_elem["dt"].as<uint32_t>();
    pars.numBsAnt             = dset_elem["numBsAnt"].as<uint32_t>();
    pars.numBbuLayers         = dset_elem["numBbuLayers"].as<uint32_t>();
    pars.numTb                = dset_elem["numTb"].as<uint32_t>();
    pars.ldpcEarlyTermination = dset_elem["ldpcEarlyTermination"].as<uint32_t>();
    pars.ldpcAlgoIndex        = dset_elem["ldpcAlgoIndex"].as<uint32_t>();
    pars.ldpcFlags            = dset_elem["ldpcFlags"].as<uint32_t>();
    pars.ldpcKernelLaunch     = dset_elem["ldpcKernelLaunch"].as<uint32_t>();
    pars.ldpcUseHalf          = dset_elem["ldpcUseHalf"].as<uint32_t>();
    // Not currently stored in HDF5 files
    //pars.slotType             = dset_elem["slotType"].as<uint32_t>();
    return pars;
}

inline tb_pars tb_pars_from_dataset_elem(const hdf5hpp::hdf5_dataset_elem& dset_elem)
{
    tb_pars pars;
    std::memset(&pars, 0, sizeof(pars));
    pars.nRnti            = dset_elem["nRnti"].as<uint32_t>();
    pars.numLayers        = dset_elem["numLayers"].as<uint32_t>();
    pars.layerMap         = dset_elem["layerMap"].as<uint64_t>();
    pars.startPrb         = dset_elem["startPrb"].as<uint32_t>();
    pars.numPrb           = dset_elem["numPRb"].as<uint32_t>();
    pars.startSym         = dset_elem["startSym"].as<uint32_t>();
    pars.numSym           = dset_elem["numSym"].as<uint32_t>();
    pars.dataScramId      = dset_elem["dataScramId"].as<uint32_t>();
    pars.rv               = dset_elem["rv"].as<uint32_t>();
    pars.dmrsType         = dset_elem["dmrsType"].as<uint32_t>();
    pars.dmrsAddlPosition = dset_elem["dmrsAddlPosition"].as<uint32_t>();
    pars.dmrsMaxLength    = dset_elem["dmrsMaxLength"].as<uint32_t>();
    pars.dmrsScramId      = dset_elem["dmrsScramId"].as<uint32_t>();
    pars.dmrsEnergy       = dset_elem["dmrsEnergy"].as<uint32_t>();
    pars.dmrsCfg          = dset_elem["dmrsCfg"].as<uint32_t>();
    pars.nSCID            = dset_elem["nSCID"].as<uint32_t>();
    pars.nPortIndex       = dset_elem["nPortIndex"].as<uint32_t>();
    return pars;
}

inline void read_cell_static_pars_from_file(cuphyCellStatPrm_t* cell_static_params,
                                            hdf5hpp::hdf5_dataset& cell_static_dataset,
                                            int num_cells,
                                            int cells_base,
                                            uint16_t& max_BWP /* Up to the caller to reset to 0 before calling */ ) {

    for (int cell_id = 0; cell_id < num_cells; cell_id++) {
        hdf5hpp::hdf5_dataset_elem static_cell_config = cell_static_dataset[cell_id];
        cell_static_params[cell_id].phyCellId         = static_cell_config["phyCellId"].as<uint16_t>();
        cell_static_params[cell_id].nRxAnt    = static_cell_config["nRxAnt"].as<uint16_t>();
        cell_static_params[cell_id].nTxAnt    = static_cell_config["nTxAnt"].as<uint16_t>();
        cell_static_params[cell_id].nPrbUlBwp = static_cell_config["nPrbUlBwp"].as<uint16_t>();
        cell_static_params[cell_id].nPrbDlBwp = static_cell_config["nPrbDlBwp"].as<uint16_t>();
        cell_static_params[cell_id].mu        = static_cell_config["mu"].as<uint8_t>();
        //TODO placeholder for cpType and duplexingMode?

        max_BWP = std::max(max_BWP, cell_static_params[cell_id].nPrbDlBwp);

        // FIXME Eventually add a consistent way for error checking for all params.
        if (cell_static_params[cell_id].mu > 1) {
            throw std::runtime_error("Unsupported numerology value!");
        }

        if(PDSCH_PRINT_CONFIG)
        {
            NVLOGC_FMT(NVLOG_PDSCH, "read_cell_static_pars_from_file: cell_id = {}", cell_id + cells_base);
            NVLOGC_FMT(NVLOG_PDSCH, "---------------------");
            NVLOGC_FMT(NVLOG_PDSCH, "phyCellId:      {:4d}", cell_static_params[cell_id].phyCellId);
            NVLOGC_FMT(NVLOG_PDSCH, "nRxAnt:         {:4d}", cell_static_params[cell_id].nRxAnt);
            NVLOGC_FMT(NVLOG_PDSCH, "nTxAnt:         {:4d}", cell_static_params[cell_id].nTxAnt);
            NVLOGC_FMT(NVLOG_PDSCH, "nPrbUlBwp:      {:4d}", cell_static_params[cell_id].nPrbUlBwp);
            NVLOGC_FMT(NVLOG_PDSCH, "nPrbDlBwp:      {:4d}", cell_static_params[cell_id].nPrbDlBwp);
            NVLOGC_FMT(NVLOG_PDSCH, "mu:             {:4d}", cell_static_params[cell_id].mu);
            NVLOGC_FMT(NVLOG_PDSCH, "---------------------");
        }
    }
}

inline void pdsch_params_cleanup(cuphyPdschStatPrms_t& pdsch_static_params,
                                std::vector<cuphyPdschCellGrpDynPrm_t>& cell_grp_dyn_params) {

    delete pdsch_static_params.pDbg;
    delete[] pdsch_static_params.pCellStatPrms;

    for (int cell_group_id = 0; cell_group_id < cell_grp_dyn_params.size(); cell_group_id++) {
        cuphyPdschCellGrpDynPrm_t* cell_group = &cell_grp_dyn_params[cell_group_id];

        for (int ue_group_id = 0; ue_group_id < cell_group->nUeGrps; ue_group_id++) {
            delete[] cell_group->pUeGrpPrms[ue_group_id].rbBitmap;
            delete[] cell_group->pUeGrpPrms[ue_group_id].pUePrmIdxs;
            delete[] cell_group->pUeGrpPrms[ue_group_id].pDmrsDynPrm;
        }

        for (int ue_id = 0; ue_id < cell_group->nUes; ue_id++) {
            delete[] cell_group->pUePrms[ue_id].pCwIdxs;
        }

        delete[] cell_group->pCellPrms;
        delete[] cell_group->pCellMetrics;
        delete[] cell_group->pUeGrpPrms;
        delete[] cell_group->pUePrms;
        delete[] cell_group->pCwPrms;
        delete[] cell_group->pCsiRsPrms;
    }
}

inline void read_pdsch_static_pars_from_file(cuphyPdschStatPrms_t& pdsch_static_params,
                                             hdf5hpp::hdf5_file& input_file,
                                             const char* filename,
                                             bool ref_check,
                                             bool identical_ldpc_configs)  {

    hdf5hpp::hdf5_dataset cell_static_dataset = input_file.open_dataset("cellStat_pars");
    int num_cells = cell_static_dataset.get_dataspace().get_dimensions()[0];
    pdsch_static_params.nCells = num_cells;
    pdsch_static_params.stream_priority = PDSCH_STREAM_PRIORITY;
    pdsch_static_params.pCellStatPrms = new cuphyCellStatPrm_t[num_cells];
    uint16_t max_BWP = 0;
    read_cell_static_pars_from_file(pdsch_static_params.pCellStatPrms, cell_static_dataset, num_cells, 0 /* cells base */, max_BWP);

    //Note: There are runtime checks in PdschTx to ensure the current numbers of cells and UEs (TBs) do not exceed the specified max.
#if 0  // Enable if clause if you want to use compilte time max. values
    pdsch_static_params.nMaxCellsPerSlot    = 0;
    pdsch_static_params.nMaxUesPerCellGroup = 0;
    pdsch_static_params.nMaxCBsPerTB        = 0;
    pdsch_static_params.nMaxPrb             = 0;
#else  // Compute maximum values for this specific TV
    pdsch_static_params.nMaxCellsPerSlot    = num_cells;

    hdf5hpp::hdf5_dataset_elem cell_grp_dyn_config = input_file.open_dataset("cellGrpDyn_pars")[0];
    int16_t new_UEs                                = cell_grp_dyn_config["nUes"].as<uint16_t>();
    pdsch_static_params.nMaxUesPerCellGroup        = new_UEs;

    pdsch_static_params.nMaxCBsPerTB      = 0;
    for (int UE_idx = 0; UE_idx < new_UEs; UE_idx++) {
        std::string CBs_dataset_name      = "tb" + std::to_string(UE_idx) + "_cbs";
        hdf5hpp::hdf5_dataset CBs_dataset = input_file.open_dataset(CBs_dataset_name.c_str());
        uint16_t num_CBs_for_TB           = CBs_dataset.get_dataspace().get_dimensions()[0];
        pdsch_static_params.nMaxCBsPerTB  = std::max(pdsch_static_params.nMaxCBsPerTB, num_CBs_for_TB);
    }

    pdsch_static_params.nMaxPrb           = max_BWP;
#endif
    pdsch_static_params.enableBatchedMemcpy = 0; //update as needed

    pdsch_static_params.pDbg = new cuphyPdschDbgPrms_t({filename, 1 /* check TB size*/, ref_check}); //identical_ldpc_configs not set; is deprecated
}

inline void read_pdsch_static_pars_from_file_v2(cuphyPdschStatPrms_t& pdsch_static_params,
                                                hdf5hpp::hdf5_file& input_file,
                                                const char* filename,
                                                bool ref_check,
                                                bool identical_ldpc_configs)  {

    read_pdsch_static_pars_from_file(pdsch_static_params, input_file, filename, ref_check, identical_ldpc_configs);
}


inline void read_ue_groups_pars_from_file(cuphyPdschUeGrpPrm_t* pdsch_ue_group_params,
                                          hdf5hpp::hdf5_file & input_file,
                                          cuphyPdschCellDynPrm_t* cell,
                                          std::vector<cuphyPdschDmrsPrm_t>& pdsch_dmrs_pars,
                                          int ue_groups_base,
                                          int ues_base) {

    hdf5hpp::hdf5_dataset pdsch_ue_groups_dataset = input_file.open_dataset("ueGrp_pars");
    int num_ue_groups = pdsch_ue_groups_dataset.get_dataspace().get_dimensions()[0];

    for (int ue_group_id = 0; ue_group_id < num_ue_groups; ue_group_id++) {

        hdf5hpp::hdf5_dataset_elem ue_group_config = pdsch_ue_groups_dataset[ue_group_id];
        pdsch_ue_group_params[ue_group_id].startPrb = ue_group_config["startPrb"].as<uint16_t>();
        pdsch_ue_group_params[ue_group_id].nPrb = ue_group_config["nPrb"].as<uint16_t>();
        pdsch_ue_group_params[ue_group_id].nUes = ue_group_config["nUes"].as<uint16_t>();

        try {
            pdsch_ue_group_params[ue_group_id].resourceAlloc  = ue_group_config["resourceAlloc"].as<uint8_t>();
            pdsch_ue_group_params[ue_group_id].rbBitmap = new uint8_t[MAX_RBMASK_BYTE_SIZE];
            auto rbBitmap = ue_group_config["rbBitmap"].as<std::vector<uint8_t>>();
            uint8_t idx = 0;
            for(auto const& rbByte : rbBitmap)
            {
                pdsch_ue_group_params[ue_group_id].rbBitmap[idx++] = rbByte;
            }
            pdsch_ue_group_params[ue_group_id].pdschStartSym  = ue_group_config["pdschStartSym"].as<uint8_t>();
            pdsch_ue_group_params[ue_group_id].nPdschSym      = ue_group_config["nPdschSym"].as<uint8_t>();
            pdsch_ue_group_params[ue_group_id].dmrsSymLocBmsk = ue_group_config["dmrsSymLocBmsk"].as<uint16_t>();
        }
        catch(...)
        {
           // It's OK for now as these fields may not exist in older TVs; there they only exist at the cell level.
           pdsch_ue_group_params[ue_group_id].resourceAlloc  = 1;
           pdsch_ue_group_params[ue_group_id].pdschStartSym  = 0;
           pdsch_ue_group_params[ue_group_id].nPdschSym      = 0;
           pdsch_ue_group_params[ue_group_id].dmrsSymLocBmsk = 0;
        }

        pdsch_ue_group_params[ue_group_id].pUePrmIdxs = new uint16_t[pdsch_ue_group_params[ue_group_id].nUes];

        // Add pointer to parent cell.
        pdsch_ue_group_params[ue_group_id].pCellPrm = &cell[ue_group_config["cellIdx"].as<uint16_t>()];

        typedef std::vector<uint16_t> veci;
        veci ue_prm_idxs =  pdsch_ue_groups_dataset[ue_group_id]["UePrmIdxs"].as<veci>();
        std::transform(ue_prm_idxs.begin(), ue_prm_idxs.end(), pdsch_ue_group_params[ue_group_id].pUePrmIdxs, [&](uint16_t x){return(x + ues_base);});

        pdsch_ue_group_params[ue_group_id].pDmrsDynPrm = new cuphyPdschDmrsPrm_t[1];
        pdsch_ue_group_params[ue_group_id].pDmrsDynPrm[0] = pdsch_dmrs_pars[ue_group_config["dmrsIdx"].as<uint16_t>()];

        if(PDSCH_PRINT_CONFIG)
        {
            NVLOGC_FMT(NVLOG_PDSCH, "read_ue_groups_pars_from_file: ue_group_id = {}", ue_group_id + ue_groups_base);
            NVLOGC_FMT(NVLOG_PDSCH, "---------------------");
            NVLOGC_FMT(NVLOG_PDSCH, "startPrb:       {:4d}", pdsch_ue_group_params[ue_group_id].startPrb);
            NVLOGC_FMT(NVLOG_PDSCH, "nPrb:           {:4d}", pdsch_ue_group_params[ue_group_id].nPrb);
            NVLOGC_FMT(NVLOG_PDSCH, "nUes:           {:4d}", pdsch_ue_group_params[ue_group_id].nUes);
            // Either the UE-group or the cell-level config. parameters will be used. Printing values from both locations.
            NVLOGC_FMT(NVLOG_PDSCH, "pdschStartSym:  {:4d}", pdsch_ue_group_params[ue_group_id].pdschStartSym);
            NVLOGC_FMT(NVLOG_PDSCH, "nPdschSym:      {:4d}", pdsch_ue_group_params[ue_group_id].nPdschSym);
            NVLOGC_FMT(NVLOG_PDSCH, "dmrsSymLocBmsk: {:4d}", pdsch_ue_group_params[ue_group_id].dmrsSymLocBmsk);
            NVLOGC_FMT(NVLOG_PDSCH, "---------------------");
        }
    }
}


inline void read_ue_pars_from_file(cuphyPdschUePrm_t* pdsch_ue_params,
                                   hdf5hpp::hdf5_file & input_file,
                                   cuphyPdschUeGrpPrm_t* ue_group,
                                   cuphyPdschCellGrpDynPrm_t& cell_group,
                                   uint32_t ues_base) {

    hdf5hpp::hdf5_dataset pdsch_ues_dataset = input_file.open_dataset("ue_pars");
    int num_ues = pdsch_ues_dataset.get_dataspace().get_dimensions()[0];

    //uint16_t pmwIndex = 0;
    uint16_t pmwIndex = cell_group.nPrecodingMatrices;
    std::vector<cuphyPmW_t> pmwArray;

    for (int ue_id = 0; ue_id < num_ues; ue_id++) {
        hdf5hpp::hdf5_dataset_elem ue_config = pdsch_ues_dataset[ue_id];
        pdsch_ue_params[ue_id].scid = ue_config["scid"].as<uint8_t>();
        pdsch_ue_params[ue_id].dmrsScrmId = ue_config["dmrsScramId"].as<uint16_t>();
        pdsch_ue_params[ue_id].nlAbove16 = ue_config["nlAbove16"].as<uint8_t>();
        pdsch_ue_params[ue_id].nUeLayers = ue_config["nUeLayers"].as<uint8_t>();
        try
        {
            pdsch_ue_params[ue_id].dmrsPortBmsk = ue_config["dmrsPortBmsk"].as<uint16_t>();
        }
        catch(...)
        {
            // compute dmrsPortBmsk from nPortIndex
            uint32_t n_port_index = ue_config["nPortIndex"].as<uint32_t>();
            pdsch_ue_params[ue_id].dmrsPortBmsk = 0;
            for (int i = 0; i <  pdsch_ue_params[ue_id].nUeLayers; i++) {
                pdsch_ue_params[ue_id].dmrsPortBmsk |= (1 << ((n_port_index >> (28 - 4 *i)) & 0x0FU));
            }
        }
        pdsch_ue_params[ue_id].refPoint   = ue_config["refPoint"].as<uint8_t>();
        pdsch_ue_params[ue_id].BWPStart   = ue_config["BWPStart"].as<uint16_t>();
        pdsch_ue_params[ue_id].beta_dmrs  = ue_config["beta_dmrs"].as<float>();
        pdsch_ue_params[ue_id].beta_qam   = ue_config["beta_qam"].as<float>();
        pdsch_ue_params[ue_id].rnti = ue_config["rnti"].as<uint16_t>();
        pdsch_ue_params[ue_id].dataScramId = ue_config["dataScramId"].as<uint16_t>();

        pdsch_ue_params[ue_id].nCw = ue_config["nCw"].as<uint8_t>();
        pdsch_ue_params[ue_id].pCwIdxs = new uint16_t[pdsch_ue_params[ue_id].nCw];

        try
        {
            pdsch_ue_params[ue_id].enablePrcdBf = ue_config["enablePrcdBf"].as<uint8_t>();
        }
        catch(...)
        {
            pdsch_ue_params[ue_id].enablePrcdBf = 0;
        }

        if(pdsch_ue_params[ue_id].enablePrcdBf)
        {
            // read pre-coding matrix and push in vector
            pdsch_ue_params[ue_id].pmwPrmIdx = pmwIndex;
            std::string pm_dataset_name = "tb" + std::to_string(ue_id) + "_PM_W";

            hdf5hpp::hdf5_dataset pm_dataset = input_file.open_dataset(pm_dataset_name.c_str());
            uint16_t ue_antenna_ports = pm_dataset.get_dataspace().get_dimensions()[1];
            cuphyPmW_t pmw;
            memset(&pmw.matrix, 0, sizeof(pmw.matrix)); // Reminder pmw.matrix can be overprovisioned compared to the dataset
            pm_dataset.read(&pmw.matrix);
            pmw.nPorts = ue_antenna_ports;
            pmwArray.push_back(pmw);

            ++pmwIndex;
        }

        // Add pointer to parent UE group
        pdsch_ue_params[ue_id].pUeGrpPrm = &ue_group[ue_config["ueGrpIdx"].as<uint16_t>()];

        // Specify indices to codewords belonging to this UE.
        typedef std::vector<uint16_t> veci;
        veci cw_idxs =  pdsch_ues_dataset[ue_id]["CwIdxs"].as<veci>();
        //std::copy(cw_idxs.begin(), cw_idxs.end(), pdsch_ue_params[ue_id].pCwIdxs);
        std::transform(cw_idxs.begin(), cw_idxs.end(), pdsch_ue_params[ue_id].pCwIdxs, [&](uint16_t x){return(x + ues_base);});
        if(PDSCH_PRINT_CONFIG)
        {
            NVLOGC_FMT(NVLOG_PDSCH, "read_ue_pars_from_file: ue_id = {}", ue_id + ues_base);
            NVLOGC_FMT(NVLOG_PDSCH, "---------------------");
            NVLOGC_FMT(NVLOG_PDSCH, "scid:         {:4d}", pdsch_ue_params[ue_id].scid);
            NVLOGC_FMT(NVLOG_PDSCH, "dmrsScrmId:   {:4d}", pdsch_ue_params[ue_id].dmrsScrmId);
            NVLOGC_FMT(NVLOG_PDSCH, "nlAbove16:    {:4d}", pdsch_ue_params[ue_id].nlAbove16);
            NVLOGC_FMT(NVLOG_PDSCH, "nUeLayers:    {:4d}", pdsch_ue_params[ue_id].nUeLayers);
            NVLOGC_FMT(NVLOG_PDSCH, "dmrsPortBmsk  {:#x}", pdsch_ue_params[ue_id].dmrsPortBmsk);
            NVLOGC_FMT(NVLOG_PDSCH, "refPoint:     {:4d}", pdsch_ue_params[ue_id].refPoint);
            NVLOGC_FMT(NVLOG_PDSCH, "BWPStart:     {:4d}", pdsch_ue_params[ue_id].BWPStart);
            NVLOGC_FMT(NVLOG_PDSCH, "beta_drms:    {:f}",  pdsch_ue_params[ue_id].beta_dmrs);
            NVLOGC_FMT(NVLOG_PDSCH, "beta_qam:     {:f}",  pdsch_ue_params[ue_id].beta_qam);
            NVLOGC_FMT(NVLOG_PDSCH, "rnti:         {:4d}", pdsch_ue_params[ue_id].rnti);
            NVLOGC_FMT(NVLOG_PDSCH, "dataScramId:  {:4d}", pdsch_ue_params[ue_id].dataScramId);
            NVLOGC_FMT(NVLOG_PDSCH, "nCw:          {:4d}", pdsch_ue_params[ue_id].nCw);
            NVLOGC_FMT(NVLOG_PDSCH, "---------------------");
        }
    }

    int old_nPrecodingMatrices =  cell_group.nPrecodingMatrices;
    cell_group.nPrecodingMatrices += pmwArray.size();
    if(cell_group.nPrecodingMatrices > 0)
    {
        std::copy(pmwArray.begin(), pmwArray.end(), cell_group.pPmwPrms + old_nPrecodingMatrices);
    }
#if 0
    else
    {
        cell_group.pPmwPrms = nullptr; // This is problematic in phase-2 w/ -g option if the first cell does not have precoding enabled. It's OK for phase-3 because the pPwmPrms is reset.
    }
#endif
}


/**
 * @brief Compute target code rate for downlink based on the appropriate
 *        Modulation and Coding Scheme (MCS) table Id and index.
 * @param[in] mcs_table_index: MCS table Id. Values  [0, 3); tb_pars.mcsTableIndex starts from 1.
 * @param[in] mcs_index: MCS index within the mcs_table_index table.
 * @return code rate
 */
inline uint16_t derive_pdsch_target_code_rate(uint32_t mcs_table_index, uint32_t mcs_index) {


    float pdsch_code_rate_mcs_table[3][29] = {
        /* MCS index Table 1 for PDSCH */
        {120, 157, 193, 251, 308, 379, 449, 526, 602, 679, /* end of Qm=2 */
         340, 378, 434, 490, 553, 616, 658, /* end of Qm=4 */
         438, 466, 517, 567, 616, 666, 719, 772, 822, 873, 910, 948}, /* end of Qm=6 */

        /* MCS index Table 2 for PDSCH */
        {120, 193, 308, 449, 602, /* end of Qm=2 */
         378, 434, 490, 553, 616, 658, /* end of Qm=4 */
         466, 517, 567, 616, 666, 719, 772, 822, 873, /* end of Qm=6 */
         682.5, 711, 754, 797, 841, 885, 916.5, 948}, /* end of Qm=8 */

        /* MCS index Table 3 for PDSCH */
        {30, 40, 50, 64, 78, 99, 120, 157, 193, 251, 308, 379, 449, 526, 602, /* end of Qm=2 */
         340, 378, 434, 490, 553, 616, /* end of Qm=4 */
         438, 466, 517, 567, 616, 666, 719, 772} /* end of Qm=6 */
        };

    if ((mcs_table_index >= 3) || (mcs_index >= 29)) {
        throw std::runtime_error("Invalid MCS index.");
    }
    return  (pdsch_code_rate_mcs_table[mcs_table_index][mcs_index] * 10);
}

/**
 * @brief Compute modulation order Qm for downlink based on the appropriate
 *        Modulation and Coding Scheme (MCS) table Id and index.
 * @param[in] mcs_table_index: MCS table Id. Values  [0, 3); tb_pars.mcsTableIndex starts from 1.
 * @param[in] mcs_index: MCS index within the mcs_table_index table.
 * @return modulation order
 */
inline int derive_pdsch_modulation_order(uint32_t mcs_table_index, uint32_t mcs_index) {

    /* MCS index Tables 1 to 3 for PDSCH */
    uint32_t pdsch_qm_mcs_table[3][29] = {
        {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6},
        {2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 0},
        {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6}
    };
    if ((mcs_table_index >= 3) || (mcs_index >= 29)) {
        throw std::runtime_error("Invalid MCS index.");
    }
    return  pdsch_qm_mcs_table[mcs_table_index][mcs_index];
}


inline void read_cw_pars_from_file(cuphyPdschCwPrm_t* pdsch_cw_params,
                                   hdf5hpp::hdf5_file & input_file,
                                   cuphyPdschUePrm_t* ue,
                                   uint32_t cws_base) {

    hdf5hpp::hdf5_dataset pdsch_cws_dataset = input_file.open_dataset("cw_pars");
    int num_cws = pdsch_cws_dataset.get_dataspace().get_dimensions()[0];

    for (int cw_id = 0; cw_id < num_cws; cw_id++) {
        hdf5hpp::hdf5_dataset_elem cw_config = pdsch_cws_dataset[cw_id];
        // For new TVs, MCS table and index fields are only used in the optional TB size check.
        // That check is configurable via checkTbSize static parameter.
        uint8_t mcs_table_index = cw_config["mcsTableIndex"].as<uint8_t>();
        uint8_t mcs_index       = cw_config["mcsIndex"].as<uint8_t>();
        try {
            pdsch_cw_params[cw_id].targetCodeRate = cw_config["targetCodeRate"].as<uint16_t>();
            pdsch_cw_params[cw_id].qamModOrder    = cw_config["qamModOrder"].as<uint8_t>();
        } catch(...)
        {
            // For old TVs, use the MCS index and table to compute targetCodeRate and qamModOrder
            pdsch_cw_params[cw_id].targetCodeRate = derive_pdsch_target_code_rate(mcs_table_index, mcs_index);
            pdsch_cw_params[cw_id].qamModOrder    = derive_pdsch_modulation_order(mcs_table_index, mcs_index);
        }
        pdsch_cw_params[cw_id].mcsTableIndex  = mcs_table_index;
        pdsch_cw_params[cw_id].mcsIndex       = mcs_index;

        pdsch_cw_params[cw_id].rv = cw_config["rv"].as<uint8_t>();

        pdsch_cw_params[cw_id].tbStartOffset = cw_config["tbStartOffset"].as<uint32_t>();
        pdsch_cw_params[cw_id].tbSize = cw_config["tbSize"].as<uint32_t>();

        pdsch_cw_params[cw_id].n_PRB_LBRM    = cw_config["n_PRB_LBRM"].as<uint16_t>();
        pdsch_cw_params[cw_id].maxLayers     = cw_config["maxLayers"].as<uint8_t>();
        pdsch_cw_params[cw_id].maxQm         = cw_config["maxQm"].as<uint8_t>();

        // Add pointer to parent UE
        pdsch_cw_params[cw_id].pUePrm = &ue[cw_config["ueIdx"].as<uint16_t>()];

        if(PDSCH_PRINT_CONFIG)
        {
            NVLOGC_FMT(NVLOG_PDSCH, "read_cw_pars_from_file: cw_id = {}", cw_id + cws_base);
            NVLOGC_FMT(NVLOG_PDSCH, "---------------------");
            NVLOGC_FMT(NVLOG_PDSCH, "mcsTableIndex:  {:4d}", pdsch_cw_params[cw_id].mcsTableIndex);
            NVLOGC_FMT(NVLOG_PDSCH, "mcsIndex:       {:4d}", pdsch_cw_params[cw_id].mcsIndex);
            NVLOGC_FMT(NVLOG_PDSCH, "targetCodeRate: {:4d}", pdsch_cw_params[cw_id].targetCodeRate);
            NVLOGC_FMT(NVLOG_PDSCH, "qamModOrder:    {:4d}", pdsch_cw_params[cw_id].qamModOrder);
            NVLOGC_FMT(NVLOG_PDSCH, "rv:             {:4d}", pdsch_cw_params[cw_id].rv);
            NVLOGC_FMT(NVLOG_PDSCH, "tbSize:         {:4d}", pdsch_cw_params[cw_id].tbSize);

            NVLOGC_FMT(NVLOG_PDSCH, "maxLayers:      {:4d}", pdsch_cw_params[cw_id].maxLayers);
            NVLOGC_FMT(NVLOG_PDSCH, "maxQm:          {:4d}", pdsch_cw_params[cw_id].maxQm);
            NVLOGC_FMT(NVLOG_PDSCH, "n_PRB_LBRM:     {:4d}", pdsch_cw_params[cw_id].n_PRB_LBRM);
            NVLOGC_FMT(NVLOG_PDSCH, "---------------------");
        }
    }
}

inline void read_dmrs_pars_from_file(std::vector<cuphyPdschDmrsPrm_t> & pdsch_dmrs_params,
                                     hdf5hpp::hdf5_file & input_file,
                                     int ue_groups_base) {

    hdf5hpp::hdf5_dataset pdsch_dmrs_dataset = input_file.open_dataset("dmrs_pars");
    int num_ue_groups = pdsch_dmrs_dataset.get_dataspace().get_dimensions()[0];
    pdsch_dmrs_params.resize(num_ue_groups);

    for (int ue_group_id = 0; ue_group_id < num_ue_groups; ue_group_id++) {
        hdf5hpp::hdf5_dataset_elem dmrs_config = pdsch_dmrs_dataset[ue_group_id];

        pdsch_dmrs_params[ue_group_id].nDmrsCdmGrpsNoData = dmrs_config["nDmrsCdmGrpsNoData"].as<uint8_t>();

        if(PDSCH_PRINT_CONFIG)
        {
            NVLOGC_FMT(NVLOG_PDSCH, "read_dmrs_pars_from_file: ue_group_id = {}", ue_group_id + ue_groups_base);
            NVLOGC_FMT(NVLOG_PDSCH, "---------------------");
            NVLOGC_FMT(NVLOG_PDSCH, "nDmrsCdmGrpsNoData: {:4d}", pdsch_dmrs_params[ue_group_id].nDmrsCdmGrpsNoData);
            NVLOGC_FMT(NVLOG_PDSCH, "---------------------");
        }
    }
}

// Assumption: if cells_base != 0, then pdsch_cell_dynamic_params points to the start of the new allocation
inline void read_cell_dynamic_pars_from_file(cuphyPdschCellDynPrm_t* pdsch_cell_dynamic_params,
                                             hdf5hpp::hdf5_file & input_file,
                                             uint32_t cells_base,
                                             uint32_t csirs_base) {

    hdf5hpp::hdf5_dataset pdsch_cell_dynamic_pars_dataset = input_file.open_dataset("cellDyn_pars");
    int num_cells = pdsch_cell_dynamic_pars_dataset.get_dataspace().get_dimensions()[0];

    for (int cell_id = 0; cell_id < num_cells; cell_id++) {
        hdf5hpp::hdf5_dataset_elem cell_dynamic_config = pdsch_cell_dynamic_pars_dataset[cell_id];
        pdsch_cell_dynamic_params[cell_id].slotNum = cell_dynamic_config["slotNum"].as<uint16_t>();
        pdsch_cell_dynamic_params[cell_id].pdschStartSym = cell_dynamic_config["pdschStartSym"].as<uint8_t>();
        pdsch_cell_dynamic_params[cell_id].nPdschSym = cell_dynamic_config["nPdschSym"].as<uint8_t>();
        pdsch_cell_dynamic_params[cell_id].dmrsSymLocBmsk = cell_dynamic_config["dmrsSymLocBmsk"].as<uint16_t>();

        //Read test model information; set to 0 if not present in TV
        try
        {
            pdsch_cell_dynamic_params[cell_id].testModel = cell_dynamic_config["testModel"].as<uint8_t>();
        }
        catch(...)
        {
            pdsch_cell_dynamic_params[cell_id].testModel = 0; // not in testing mode
        }

        //Add indices to static and dynamic cells.
        pdsch_cell_dynamic_params[cell_id].cellPrmStatIdx = cell_dynamic_config["cellStatIdx"].as<uint16_t>() + cells_base;
        pdsch_cell_dynamic_params[cell_id].cellPrmDynIdx = cell_dynamic_config["cellDynIdx"].as<int16_t>() + cells_base;

        // Read cell-specific CSI-RS information
        uint16_t num_csirs_prms  = cell_dynamic_config["nCsirsPrms"].as<uint16_t>();
        pdsch_cell_dynamic_params[cell_id].nCsiRsPrms = num_csirs_prms;
        pdsch_cell_dynamic_params[cell_id].csiRsPrmsOffset = cell_dynamic_config["csirsPrmsOffset"].as<uint16_t>() + csirs_base;
        if(PDSCH_PRINT_CONFIG)
        {
            NVLOGC_FMT(NVLOG_PDSCH, "read_cell_dynamic_pars_from_file: cell_id = {}", cell_id + cells_base);
            NVLOGC_FMT(NVLOG_PDSCH, "---------------------");
            NVLOGC_FMT(NVLOG_PDSCH, "slotNum:         {:4d}", pdsch_cell_dynamic_params[cell_id].slotNum);
            NVLOGC_FMT(NVLOG_PDSCH, "pdschStartSym:   {:4d}", pdsch_cell_dynamic_params[cell_id].pdschStartSym);
            NVLOGC_FMT(NVLOG_PDSCH, "nPdschSym:       {:4d}", pdsch_cell_dynamic_params[cell_id].nPdschSym);
            NVLOGC_FMT(NVLOG_PDSCH, "dmrsSymLocBmsk:  {:4d}", pdsch_cell_dynamic_params[cell_id].dmrsSymLocBmsk);
            NVLOGC_FMT(NVLOG_PDSCH, "cellPrmStatIdx:  {:4d}", pdsch_cell_dynamic_params[cell_id].cellPrmStatIdx);
            NVLOGC_FMT(NVLOG_PDSCH, "cellPrmDynIdx:   {:4d}", pdsch_cell_dynamic_params[cell_id].cellPrmDynIdx);
            NVLOGC_FMT(NVLOG_PDSCH, "nCsirsPrms:      {:4d}", pdsch_cell_dynamic_params[cell_id].nCsiRsPrms);
            NVLOGC_FMT(NVLOG_PDSCH, "csirsPrmsOffset: {:4d}", pdsch_cell_dynamic_params[cell_id].csiRsPrmsOffset);
            NVLOGC_FMT(NVLOG_PDSCH, "testModel:       {:4d}", pdsch_cell_dynamic_params[cell_id].testModel);
            NVLOGC_FMT(NVLOG_PDSCH, "---------------------");
        }
     }
}

inline void read_pdsch_csirs_pars_from_file(_cuphyCsirsRrcDynPrm* csirs_params, int num_csirs, hdf5hpp::hdf5_dataset & csirs_dyn_pars_dataset)
{
    // Only update CSI-RS fields needed to find CSI-RS RE location. NB: Other fields set to 0.
    std::memset(csirs_params, 0, num_csirs * sizeof(_cuphyCsirsRrcDynPrm));

    for (int csirs_idx  = 0; csirs_idx < num_csirs; csirs_idx++) {
        hdf5hpp::hdf5_dataset_elem csirs_dyn_config = csirs_dyn_pars_dataset[csirs_idx];
        csirs_params[csirs_idx].startRb     = csirs_dyn_config["StartRB"].as<uint16_t>();
        csirs_params[csirs_idx].nRb         = csirs_dyn_config["NrOfRBs"].as<uint16_t>();
        csirs_params[csirs_idx].freqDomain  = csirs_dyn_config["FreqDomain"].as<uint16_t>();
        csirs_params[csirs_idx].row         = csirs_dyn_config["Row"].as<uint8_t>();
        csirs_params[csirs_idx].symbL0      = csirs_dyn_config["SymbL0"].as<uint8_t>();
        csirs_params[csirs_idx].symbL1      = csirs_dyn_config["SymbL1"].as<uint8_t>();
        csirs_params[csirs_idx].freqDensity = csirs_dyn_config["FreqDensity"].as<uint8_t>();

        if(PDSCH_PRINT_CONFIG)
        {
            NVLOGC_FMT(NVLOG_PDSCH, "CSIRS params: CSIRS_idx= {}", csirs_idx);
            NVLOGC_FMT(NVLOG_PDSCH, "---------------------");
            NVLOGC_FMT(NVLOG_PDSCH, "startRb:         {:4d}", csirs_params[csirs_idx].startRb);
            NVLOGC_FMT(NVLOG_PDSCH, "nRb:             {:4d}", csirs_params[csirs_idx].nRb);
            NVLOGC_FMT(NVLOG_PDSCH, "freqDomain:      {:4d}", csirs_params[csirs_idx].freqDomain);
            NVLOGC_FMT(NVLOG_PDSCH, "row:             {:4d}", csirs_params[csirs_idx].row);
            NVLOGC_FMT(NVLOG_PDSCH, "symbL0:          {:4d}", csirs_params[csirs_idx].symbL0);
            NVLOGC_FMT(NVLOG_PDSCH, "symbL1:          {:4d}", csirs_params[csirs_idx].symbL1);
            NVLOGC_FMT(NVLOG_PDSCH, "freqDensity:     {:4d}", csirs_params[csirs_idx].freqDensity);
            NVLOGC_FMT(NVLOG_PDSCH, "---------------------");
        }
    }
}


inline void read_cell_group_dynamic_pars_from_file(std::vector<cuphyPdschCellGrpDynPrm_t> & cell_grp_dyn_params,
                                                   hdf5hpp::hdf5_file & input_file) {

    hdf5hpp::hdf5_dataset cell_grp_dyn_pars_dataset = input_file.open_dataset("cellGrpDyn_pars");
    int num_cell_groups = cell_grp_dyn_pars_dataset.get_dataspace().get_dimensions()[0];

    if (num_cell_groups != 1) {
        throw std::runtime_error("Only a single cell group is supported per pipeline!");
    }

    hdf5hpp::hdf5_dataset csirs_dyn_pars_dataset    = input_file.open_dataset("csirs_pars");
    int                   num_csirs                 = csirs_dyn_pars_dataset.get_dataspace().get_dimensions()[0];

    for (int cell_group_id = 0; cell_group_id < num_cell_groups; cell_group_id++) {
        hdf5hpp::hdf5_dataset_elem cell_grp_dyn_config = cell_grp_dyn_pars_dataset[cell_group_id];

        //Read # cells, UE groups, UEs, and Cws and update the relevant fields in this cell group.
        uint16_t num_cells = cell_grp_dyn_config["nCells"].as<uint16_t>();
        cell_grp_dyn_params[cell_group_id].nCells = num_cells;

        uint16_t num_ue_groups = cell_grp_dyn_config["nUeGrps"].as<uint16_t>();
        cell_grp_dyn_params[cell_group_id].nUeGrps = num_ue_groups;

        uint16_t num_ues = cell_grp_dyn_config["nUes"].as<uint16_t>();
        cell_grp_dyn_params[cell_group_id].nUes = num_ues;

        uint16_t num_cws = cell_grp_dyn_config["nCws"].as<uint16_t>();
        cell_grp_dyn_params[cell_group_id].nCws = num_cws;

        std::vector<cuphyPdschDmrsPrm_t> pdsch_dmrs_pars;
        read_dmrs_pars_from_file(pdsch_dmrs_pars, input_file, 0 /* ue_groups_base*/);

        cell_grp_dyn_params[cell_group_id].nCsiRsPrms  = num_csirs;

        // Allocate arrays of cells, UE groups, UEs, CWs and CSI-RS params
        cell_grp_dyn_params[cell_group_id].pCellPrms = new cuphyPdschCellDynPrm_t[num_cells];
        cell_grp_dyn_params[cell_group_id].pCellMetrics = new cuphyPdschCellAerialMetrics_t[num_cells]; // can also set to nullptr to check behavior; FYI there's a delete[] in pdsch_params_cleanup
        cell_grp_dyn_params[cell_group_id].pUeGrpPrms = new cuphyPdschUeGrpPrm_t[num_ue_groups];
        cell_grp_dyn_params[cell_group_id].pUePrms    = new cuphyPdschUePrm_t[num_ues];
        cell_grp_dyn_params[cell_group_id].pCwPrms    = new cuphyPdschCwPrm_t[num_cws];
        cell_grp_dyn_params[cell_group_id].pCsiRsPrms = new _cuphyCsirsRrcDynPrm[num_csirs];
        cell_grp_dyn_params[cell_group_id].pPmwPrms   = new cuphyPmW_t[num_ues];// FIXME

        // Populate arrays. Includes setting pointer to parent/children structs etc., so it should
        // happen *after* all previous mem. allocations.
        read_cell_dynamic_pars_from_file(cell_grp_dyn_params[cell_group_id].pCellPrms, input_file, 0 /* cells base */, 0 /* CSI-RS base */);

        read_ue_groups_pars_from_file(cell_grp_dyn_params[cell_group_id].pUeGrpPrms, input_file, cell_grp_dyn_params[cell_group_id].pCellPrms, pdsch_dmrs_pars, 0 /* UE groups base */, 0/* UEs base */);

        read_ue_pars_from_file(cell_grp_dyn_params[cell_group_id].pUePrms, input_file, cell_grp_dyn_params[cell_group_id].pUeGrpPrms, cell_grp_dyn_params[cell_group_id], 0 /* UEs base */);
        read_cw_pars_from_file(cell_grp_dyn_params[cell_group_id].pCwPrms, input_file, cell_grp_dyn_params[cell_group_id].pUePrms, 0 /* CWs base */);

        // Only update CSI-RS fields needed to find CSI-RS RE location. NB: Other fields set to 0.
        _cuphyCsirsRrcDynPrm* csirs_params = cell_grp_dyn_params[cell_group_id].pCsiRsPrms;
        read_pdsch_csirs_pars_from_file(csirs_params, num_csirs, csirs_dyn_pars_dataset);
     }
}


inline void read_cell_group_dynamic_pars_from_file_v2(std::vector<cuphyPdschCellGrpDynPrm_t> & cell_grp_dyn_params,
                                                      hdf5hpp::hdf5_file & input_file) {
    read_cell_group_dynamic_pars_from_file(cell_grp_dyn_params, input_file);
}


inline void read_dci_pdcch_params(cuphyPdcchDciPrm_t* dci_params,  hdf5hpp::hdf5_file& input_file, int num_DCIs, std::vector<cuphyPmWOneLayer_t>& pdcch_precoding_matrix, uint16_t& PDCCHPrecoded_base, int coreset=0)
{
    bool use_new_dset = input_file.is_valid_dataset("DciParams_coreset_0_dci_0");
    // Read num_DCIs DCIs belonging to coreset coreset.
    for (int i = 0; i < num_DCIs; i++) {
        std::string dci_dataset_name = (use_new_dset) ? ("DciParams_coreset_" + std::to_string(coreset) + "_dci_" + std::to_string(i)) : ("DciParams" + std::to_string(i + 1));
        hdf5hpp::hdf5_dataset dci_params_dataset = input_file.open_dataset(dci_dataset_name.c_str());
        hdf5hpp::hdf5_dataset_elem h5_dci_params = dci_params_dataset[0];

        // All are uint32_t except for beta_qam and beta_dmrs which are floats
        dci_params[i].Npayload = h5_dci_params["Npayload"].as<uint32_t>();
        dci_params[i].rntiCrc =  h5_dci_params["rntiCrc"].as<uint32_t>();
        dci_params[i].rntiBits =  h5_dci_params["rntiBits"].as<uint32_t>();
        dci_params[i].dmrs_id =  h5_dci_params["dmrsId"].as<uint32_t>();
        dci_params[i].aggr_level =  h5_dci_params["aggrL"].as<uint32_t>();
        dci_params[i].cce_index =  h5_dci_params["cceIdx"].as<uint32_t>();
        dci_params[i].beta_qam =  h5_dci_params["beta_qam"].as<float>();
        dci_params[i].beta_dmrs =  h5_dci_params["beta_dmrs"].as<float>();
        dci_params[i].enablePrcdBf =  h5_dci_params["enablePrcdBf"].as<uint8_t>();
        dci_params[i].pmwPrmIdx = 0xFFFF;
        if( dci_params[i].enablePrcdBf)
        {
            std::string pm_dataset_name = "DciPmW_coreset_" + std::to_string(coreset) + "_dci_" + std::to_string(i);
            typed_tensor<CUPHY_C_16F, pinned_alloc> pdcch_pm_w = typed_tensor_from_dataset<CUPHY_C_16F, pinned_alloc>(input_file.open_dataset(pm_dataset_name.c_str()));
            uint8_t nPorts = cuphy::get_HDF5_dataset_info(input_file.open_dataset(pm_dataset_name.c_str())).layout().dimensions()[0];
            pdcch_precoding_matrix[PDCCHPrecoded_base].nPorts = nPorts; 
            for(uint8_t idx=0; idx<nPorts; idx++)
            {
                pdcch_precoding_matrix[PDCCHPrecoded_base].matrix[idx] = pdcch_pm_w(idx);
            }
            dci_params[i].pmwPrmIdx = PDCCHPrecoded_base;
            PDCCHPrecoded_base++;
        }
   }
}

inline void read_pdcch_coreset_dyn_params_from_file_v2(std::vector<cuphyPdcchCoresetDynPrm_t>& coreset_dyn_params, hdf5hpp::hdf5_file& input_file, int coreset_base=0, int dci_base=0, int slot_buffer_idx=0) {

    cuphyPdcchCoresetDynPrm_t& params = coreset_dyn_params[coreset_base];

    hdf5hpp::hdf5_dataset dset = input_file.open_dataset("PdcchParams");
    int num_coresets = dset.get_dataspace().get_dimensions()[0];
    int cumulative_dci_base = dci_base;
    for (int i = 0; i < num_coresets; i++) {
        cuphyPdcchCoresetDynPrm_t& params = coreset_dyn_params[coreset_base + i];
        hdf5hpp::hdf5_dataset_elem h5_params = dset[i];

        params.slot_number = h5_params["slotNumber"].as<uint32_t>();
        params.start_rb    = h5_params["start_rb"].as<uint32_t>();
        params.start_sym   = h5_params["start_sym"].as<uint32_t>();
        params.n_sym   = h5_params["n_sym"].as<uint32_t>();
        params.bundle_size  = h5_params["bundleSize"].as<uint32_t>();
        params.interleaver_size = h5_params["interleaveSize"].as<uint32_t>();
        params.shift_index = h5_params["shiftIdx"].as<uint32_t>();
        params.interleaved = h5_params["interleaved"].as<uint32_t>();
        params.n_f         = h5_params["n_f"].as<uint32_t>();

        params.freq_domain_resource  = (h5_params["FreqDomainResource0"].as<uint32_t>() | 0ULL) << 32;
        params.freq_domain_resource  |= h5_params["FreqDomainResource1"].as<uint32_t>();

        params.nDci = h5_params["numDlDci"].as<uint32_t>();
        params.coreset_type = h5_params["CoreSetType"].as<uint32_t>();

        //Read test model information; set to 0 if not present in TV
        try
        {
            params.testModel = h5_params["testModel"].as<uint8_t>();
        }
        catch(...)
        {
            params.testModel = 0; // not in testing mode
        }

        //Added values for single CORESET first. TODO update
        params.dciStartIdx = cumulative_dci_base;
        params.slotBufferIdx = slot_buffer_idx;

        //Update cumulative_dci_base if there are more than one coresets here:
        cumulative_dci_base += params.nDci;
    }
}

inline void read_pdcch_coreset_dyn_params_from_file(std::vector<cuphyPdcchCoresetDynPrm_t>& coreset_dyn_params, hdf5hpp::hdf5_file& input_file, int coreset_base=0) {
    //FIXME some work here assumes single coreset. Will update
    cuphyPdcchCoresetDynPrm_t& params = coreset_dyn_params[coreset_base];

    hdf5hpp::hdf5_dataset dset = input_file.open_dataset("PdcchParams");
    int num_coresets = dset.get_dataspace().get_dimensions()[0];
    if (num_coresets > 1) {
        NVLOGC_FMT(NVLOG_PDCCH, "TV contains {} coresets but only the first one will be used for now.", num_coresets);
    }
    hdf5hpp::hdf5_dataset_elem h5_params = dset[0]; // hard-coded for single coreset

    params.slot_number = h5_params["slotNumber"].as<uint32_t>();
    params.start_rb    = h5_params["start_rb"].as<uint32_t>();
    params.start_sym   = h5_params["start_sym"].as<uint32_t>();
    params.n_sym   = h5_params["n_sym"].as<uint32_t>();
    params.bundle_size  = h5_params["bundleSize"].as<uint32_t>();
    params.interleaver_size = h5_params["interleaveSize"].as<uint32_t>();
    params.shift_index = h5_params["shiftIdx"].as<uint32_t>();
    params.interleaved = h5_params["interleaved"].as<uint32_t>();
    params.n_f         = h5_params["n_f"].as<uint32_t>();

    params.freq_domain_resource  = (h5_params["FreqDomainResource0"].as<uint32_t>() | 0ULL) << 32;
    params.freq_domain_resource  |= h5_params["FreqDomainResource1"].as<uint32_t>();

    //params.nDCIs = h5_params["numDlDci"].as<uint32_t>();
    params.nDci = h5_params["numDlDci"].as<uint32_t>();
    params.coreset_type = h5_params["CoreSetType"].as<uint32_t>();

    //Read test model information; set to 0 if not present in TV
    try
    {
        params.testModel = h5_params["testModel"].as<uint8_t>();
    }
    catch(...)
    {
        params.testModel = 0; // not in testing mode
    }


    //Added values for single CORESET first. TODO update
    params.dciStartIdx = 0;
    params.slotBufferIdx = 0;

    NVLOGC_FMT(NVLOG_PDCCH, "params.nDci {}", params.nDci);
}

inline void read_common_pdcch_params(cuphyPdcchCoresetDynPrm_t& params, hdf5hpp::hdf5_file& input_file) {

    hdf5hpp::hdf5_dataset dset = input_file.open_dataset("PdcchParams");
    int num_coresets = dset.get_dataspace().get_dimensions()[0];
    if (num_coresets > 1) {
        NVLOGC_FMT(NVLOG_PDCCH, "TV contains {} coresets but only the first one will be used for now.", num_coresets);
    }
    hdf5hpp::hdf5_dataset_elem h5_params = dset[0]; //FIXME hard-coded for single coreset

    params.slot_number = h5_params["slotNumber"].as<uint32_t>();
    params.start_rb    = h5_params["start_rb"].as<uint32_t>();
    params.start_sym   = h5_params["start_sym"].as<uint32_t>();
    params.n_sym   = h5_params["n_sym"].as<uint32_t>();
    params.bundle_size  = h5_params["bundleSize"].as<uint32_t>();
    params.interleaver_size = h5_params["interleaveSize"].as<uint32_t>();
    params.shift_index = h5_params["shiftIdx"].as<uint32_t>();
    params.interleaved = h5_params["interleaved"].as<uint32_t>();
    params.n_f         = h5_params["n_f"].as<uint32_t>();

    params.freq_domain_resource  = (h5_params["FreqDomainResource0"].as<uint32_t>() | 0ULL) << 32;
    params.freq_domain_resource  |= h5_params["FreqDomainResource1"].as<uint32_t>();

    //params.nDCIs = h5_params["numDlDci"].as<uint32_t>();
    params.nDci = h5_params["numDlDci"].as<uint32_t>();
    params.coreset_type = h5_params["CoreSetType"].as<uint32_t>();

}

inline void read_pdcch_coreset_dynamic_params_from_file(std::vector<cuphyPdcchCoresetDynPrm_t> & coreset_dyn_params,
                                                      hdf5hpp::hdf5_file & input_file) {

    NVLOGC_FMT(NVLOG_PDCCH, "FIXME. Function body commented out. Did you forget to fix this?");
#if 0
    // Read Coreset parameters. Common across all DCIs
    read_common_pdcch_params(coreset_dyn_params[0], input_file);
    //int num_DCIs =  coreset_dyn_params[0].nDCIs;
    int num_DCIs =  coreset_dyn_params[0].nDci;

    // Read DCI specific parameters.
    coreset_dyn_params[0].pDciPrms = new cuphyPdcchDciPrm_t[num_DCIs];
    read_dci_pdcch_params(coreset_dyn_params[0].pDciPrms, input_file, num_DCIs);
#endif
}

inline void read_csirs_dynamic_params_from_file(cuphyCsirsRrcDynPrm_t* params,
                                                int numParamList,
                                                hdf5hpp::hdf5_file & input_file,
                                                std::vector<cuphyPmWOneLayer_t>& csirs_precoding_matrix, 
                                                uint16_t& CSIRSPrecoded_base) 
{

    // Read CSI-RS RRC parameters
    hdf5hpp::hdf5_dataset dset = input_file.open_dataset("CsirsParamsList");
    for(int i = 0; i < numParamList; ++i)
    {
        const hdf5hpp::hdf5_dataset_elem& h5_params = dset[i];

        params[i].startRb = h5_params["StartRB"].as<uint16_t>();
        params[i].nRb     = h5_params["NrOfRBs"].as<uint16_t>();
        params[i].csiType = static_cast<cuphyCsiType_t>(h5_params["CSIType"].as<uint8_t>());
        params[i].row   = h5_params["Row"].as<uint8_t>();
        params[i].freqDomain  = h5_params["FreqDomain"].as<uint16_t>();
        params[i].symbL0 = h5_params["SymbL0"].as<uint8_t>();
        params[i].symbL1 = h5_params["SymbL1"].as<uint8_t>();
        params[i].cdmType = static_cast<cuphyCdmType_t>(h5_params["CDMType"].as<uint8_t>());
        params[i].freqDensity = h5_params["FreqDensity"].as<uint8_t>();
        params[i].scrambId    = h5_params["ScrambId"].as<uint16_t>();
        params[i].beta    = h5_params["beta"].as<float>();
        params[i].idxSlotInFrame = h5_params["idxSlotInFrame"].as<uint8_t>();
        params[i].enablePrcdBf = h5_params["enablePrcdBf"].as<uint8_t>();
        params[i].pmwPrmIdx = 0xFFFF;
        if(params[i].enablePrcdBf)
        {
            std::string pm_dataset_name = "Csirs_PM_W" + std::to_string(i);
            typed_tensor<CUPHY_C_16F, pinned_alloc> csirs_pm_w = typed_tensor_from_dataset<CUPHY_C_16F, pinned_alloc>(input_file.open_dataset(pm_dataset_name.c_str()));
            uint8_t nPorts = cuphy::get_HDF5_dataset_info(input_file.open_dataset(pm_dataset_name.c_str())).layout().dimensions()[0];
            csirs_precoding_matrix[CSIRSPrecoded_base].nPorts = nPorts; 
            for(uint8_t idx=0; idx<nPorts; idx++)
            {
                csirs_precoding_matrix[CSIRSPrecoded_base].matrix[idx] = csirs_pm_w(idx);
            }
            params[i].pmwPrmIdx = CSIRSPrecoded_base;
            CSIRSPrecoded_base++;
        }
    }
}

inline void read_ue_srstx_dynamic_params_from_file(cuphyUeSrsTxPrm_t* params, hdf5hpp::hdf5_file & input_file) 
{

    // Read CSI-RS RRC parameters
    hdf5hpp::hdf5_dataset dset = input_file.open_dataset("SrsParams");
    const hdf5hpp::hdf5_dataset_elem& h5_params = dset[0];
    
    params->nAntPorts               = h5_params["N_ap_SRS"].as<uint8_t>();
    params->nSyms                   = h5_params["N_symb_SRS"].as<uint8_t>();
    params->nRepetitions            = h5_params["R"].as<uint8_t>();
    params->combSize                = h5_params["K_TC"].as<uint8_t>();
    params->startSym                = h5_params["l0"].as<uint8_t>();
    params->sequenceId              = h5_params["n_ID_SRS"].as<uint16_t>();
    params->configIdx               = h5_params["C_SRS"].as<uint8_t>();
    params->bandwidthIdx            = h5_params["B_SRS"].as<uint8_t>();
    params->combOffset              = h5_params["k_TC_bar"].as<uint8_t>();
    params->cyclicShift             = h5_params["n_SRS_cs"].as<uint8_t>();
    params->frequencyPosition       = h5_params["n_RRC"].as<uint8_t>();
    params->frequencyShift          = h5_params["n_shift"].as<uint16_t>();
    params->frequencyHopping        = h5_params["b_hop"].as<uint8_t>();
    params->resourceType            = h5_params["resourceType"].as<uint8_t>();
    params->Tsrs                    = h5_params["Tsrs"].as<uint16_t>();
    params->Toffset                 = h5_params["Toffset"].as<uint16_t>();
    params->groupOrSequenceHopping  = h5_params["groupOrSequenceHopping"].as<uint8_t>();
    params->idxSlotInFrame          = h5_params["idxSlotInFrame"].as<uint16_t>();
    params->idxFrame                = h5_params["idxFrame"].as<uint16_t>();
}

inline void print_pdsch_static(cuphyPdschStatPrms_t* static_params)
{
    if (static_params == nullptr) return;
    int n_cells = static_params->nCells;
    NVLOGC_FMT(NVLOG_PDSCH, "PDSCH Static parameters for {} cells", n_cells);

    // Parameters common across all cells
    NVLOGC_FMT(NVLOG_PDSCH, "read_TB_CRC:           {:4d}", static_params->read_TB_CRC);
    NVLOGC_FMT(NVLOG_PDSCH, "full_slot_processing:  {:4d}", static_params->full_slot_processing);
    NVLOGC_FMT(NVLOG_PDSCH, "stream_priority:       {:4d}", static_params->stream_priority);
    NVLOGC_FMT(NVLOG_PDSCH, "nMaxCellsPerSlot:      {:4d}", static_params->nMaxCellsPerSlot);
    NVLOGC_FMT(NVLOG_PDSCH, "nMaxUesPerCellGroup:   {:4d}", static_params->nMaxUesPerCellGroup);
    NVLOGC_FMT(NVLOG_PDSCH, "nMaxCBsPerTB:          {:4d}", static_params->nMaxCBsPerTB);
    NVLOGC_FMT(NVLOG_PDSCH, "nMaxPrb:               {:4d}", static_params->nMaxPrb);
    NVLOGC_FMT(NVLOG_PDSCH, "enableBatchedMemcpy    {:4d}", static_params->enableBatchedMemcpy);

    // Cell specific parameters
    cuphyCellStatPrm_t* cell_static_params = static_params->pCellStatPrms;
    cuphyPdschDbgPrms_t* cell_dbg_params   = static_params->pDbg;
    for (int cell_id = 0; cell_id < n_cells; cell_id++) {
        NVLOGC_FMT(NVLOG_PDSCH, "------------------------------------");
        NVLOGC_FMT(NVLOG_PDSCH, "Cell {}", cell_id);
        NVLOGC_FMT(NVLOG_PDSCH, "------------------------------------");
        NVLOGC_FMT(NVLOG_PDSCH, "phyCellId:      {:4d}", cell_static_params[cell_id].phyCellId);
        NVLOGC_FMT(NVLOG_PDSCH, "nRxAnt:         {:4d}", cell_static_params[cell_id].nRxAnt);
        NVLOGC_FMT(NVLOG_PDSCH, "nTxAnt:         {:4d}", cell_static_params[cell_id].nTxAnt);
        NVLOGC_FMT(NVLOG_PDSCH, "nPrbUlBwp:      {:4d}", cell_static_params[cell_id].nPrbUlBwp);
        NVLOGC_FMT(NVLOG_PDSCH, "nPrbDlBwp:      {:4d}", cell_static_params[cell_id].nPrbDlBwp);
        NVLOGC_FMT(NVLOG_PDSCH, "mu:             {:4d}", cell_static_params[cell_id].mu);
        // Debug fields
        NVLOGC_FMT(NVLOG_PDSCH, "DBG:");
        NVLOGC_FMT(NVLOG_PDSCH, "pCfgFileName:             {}", cell_dbg_params[cell_id].pCfgFileName);
        NVLOGC_FMT(NVLOG_PDSCH, "checkTbSize:              {}", cell_dbg_params[cell_id].checkTbSize);
        NVLOGC_FMT(NVLOG_PDSCH, "refCheck:                 {}", cell_dbg_params[cell_id].refCheck);
        NVLOGC_FMT(NVLOG_PDSCH, "");
    }
}

inline void print_pdsch_cell_aerial_metrics(const cuphyPdschCellGrpDynPrm_t* cell_group_params)
{
    // The per cell metrics are updated in PDSCH setup, and so this function should be called after setup
    // has been called.
    if (cell_group_params == nullptr) return;

    // Print information for all cells
    const auto n_cells = cell_group_params->nCells;

    const cuphyPdschCellAerialMetrics_t* pdsch_cell_aerial_metrics = cell_group_params->pCellMetrics;
    if(pdsch_cell_aerial_metrics == nullptr)
    {
        NVLOGC_FMT(NVLOG_PDSCH, "No PDSCH cell metrics will be printed. pCellMetrics is nullptr.");
        return;
    }
    NVLOGC_FMT(NVLOG_PDSCH, "PDSCH cell metrics for {} cells", n_cells);

    for (int cell_id = 0; cell_id < n_cells; cell_id++) {
        NVLOGC_FMT(NVLOG_PDSCH, "------------------------------------");
        NVLOGC_FMT(NVLOG_PDSCH, "Cell {}", cell_id);
        NVLOGC_FMT(NVLOG_PDSCH, "------------------------------------");

        NVLOGC_FMT(NVLOG_PDSCH, "nTBs:            {:4d}", pdsch_cell_aerial_metrics[cell_id].nTBs);
        NVLOGC_FMT(NVLOG_PDSCH, "tbSize (bytes):  {:4d}", pdsch_cell_aerial_metrics[cell_id].tbSize);
        NVLOGC_FMT(NVLOG_PDSCH, "");
    }
}


inline void print_pdsch_dynamic_cell_group(const cuphyPdschCellGrpDynPrm_t* cell_group_params)
{
    if (cell_group_params == nullptr) return;

    // Print information for all cells
    int n_cells = cell_group_params->nCells;
    NVLOGC_FMT(NVLOG_PDSCH, "PDSCH cell group dynamic parameters for {} cells", n_cells);

    cuphyPdschCellDynPrm_t* pdsch_cell_dynamic_params = cell_group_params->pCellPrms;

    for (int cell_id = 0; cell_id < n_cells; cell_id++) {
        NVLOGC_FMT(NVLOG_PDSCH, "------------------------------------");
        NVLOGC_FMT(NVLOG_PDSCH, "Cell {}", cell_id);
        NVLOGC_FMT(NVLOG_PDSCH, "------------------------------------");

        NVLOGC_FMT(NVLOG_PDSCH, "cellPrmStatIdx:  {:4d}", pdsch_cell_dynamic_params[cell_id].cellPrmStatIdx);
        NVLOGC_FMT(NVLOG_PDSCH, "cellPrmDynIdx:   {:4d}", pdsch_cell_dynamic_params[cell_id].cellPrmDynIdx);

        NVLOGC_FMT(NVLOG_PDSCH, "slotNum:         {:4d}", pdsch_cell_dynamic_params[cell_id].slotNum);
        NVLOGC_FMT(NVLOG_PDSCH, "nCsiRsPrms:      {:4d}", pdsch_cell_dynamic_params[cell_id].nCsiRsPrms);
        NVLOGC_FMT(NVLOG_PDSCH, "csiRsPrmsOffset: {:4d}", pdsch_cell_dynamic_params[cell_id].csiRsPrmsOffset);
        NVLOGC_FMT(NVLOG_PDSCH, "testModel:       {:4d}", pdsch_cell_dynamic_params[cell_id].testModel);

        //NB: pdschStartSym, nPdschSym and dmrsSymLocBmsk may be provided in UE group instead. If so, the values here will be 0.
        NVLOGC_FMT(NVLOG_PDSCH, "pdschStartSym:   {:4d}", pdsch_cell_dynamic_params[cell_id].pdschStartSym);
        NVLOGC_FMT(NVLOG_PDSCH, "nPdschSym:       {:4d}", pdsch_cell_dynamic_params[cell_id].nPdschSym);
        NVLOGC_FMT(NVLOG_PDSCH, "dmrsSymLocBmsk:  {:4d}", pdsch_cell_dynamic_params[cell_id].dmrsSymLocBmsk);
        NVLOGC_FMT(NVLOG_PDSCH, "");
    }

    // Print information for all UE groups
    int n_ue_groups = cell_group_params->nUeGrps;
    NVLOGC_FMT(NVLOG_PDSCH, "PDSCH cell group dynamic parameters for {} UE groups", n_ue_groups);
    cuphyPdschUeGrpPrm_t* pdsch_ue_group_dynamic_params = cell_group_params->pUeGrpPrms;

    for (int ue_group_id = 0; ue_group_id <  n_ue_groups; ue_group_id++) {
        NVLOGC_FMT(NVLOG_PDSCH, "------------------------------------");
        NVLOGC_FMT(NVLOG_PDSCH, "UE group {}", ue_group_id);
        NVLOGC_FMT(NVLOG_PDSCH, "------------------------------------");

        NVLOGC_FMT(NVLOG_PDSCH, "resourceAlloc:   {:4d}", pdsch_ue_group_dynamic_params[ue_group_id].resourceAlloc);
        uint32_t* rbBitmap = reinterpret_cast<uint32_t*>(pdsch_ue_group_dynamic_params[ue_group_id].rbBitmap);
        for (int i = 0; i < MAX_RBMASK_UINT32_ELEMENTS && (pdsch_ue_group_dynamic_params[ue_group_id].resourceAlloc==0); i++){
            NVLOGC_FMT(NVLOG_PDSCH, " {:#x}",rbBitmap[i]);
        }
        NVLOGC_FMT(NVLOG_PDSCH, "startPrb:        {:4d}", pdsch_ue_group_dynamic_params[ue_group_id].startPrb);
        NVLOGC_FMT(NVLOG_PDSCH, "nPrb:            {:4d}", pdsch_ue_group_dynamic_params[ue_group_id].nPrb);

        //NB: pdschStartSym, nPdschSym and dmrsSymLocBmsk may be provided in the cell instead, and thus be common across all UE groups. If so, the values here will be 0.
        NVLOGC_FMT(NVLOG_PDSCH, "pdschStartSym:   {:4d}", pdsch_ue_group_dynamic_params[ue_group_id].pdschStartSym);
        NVLOGC_FMT(NVLOG_PDSCH, "nPdschSym:       {:4d}", pdsch_ue_group_dynamic_params[ue_group_id].nPdschSym);
        NVLOGC_FMT(NVLOG_PDSCH, "dmrsSymLocBmsk:  {:4d}", pdsch_ue_group_dynamic_params[ue_group_id].dmrsSymLocBmsk);

        NVLOGC_FMT(NVLOG_PDSCH, "nUes:            {:4d}", pdsch_ue_group_dynamic_params[ue_group_id].nUes);

        // Indices to pUePrms array
        std::stringstream ue_prm_str;
        for (int i = 0; i <  pdsch_ue_group_dynamic_params[ue_group_id].nUes; i++) {
            if (i != 0) ue_prm_str << ", ";
            ue_prm_str << std::to_string(pdsch_ue_group_dynamic_params[ue_group_id].pUePrmIdxs[i]);
        }
        NVLOGC_FMT(NVLOG_PDSCH, "pUePrmIdxs:      {{{}}}", ue_prm_str.str());

        //pDmrsDynPrm -> pointer to DMRS info
        NVLOGC_FMT(NVLOG_PDSCH, "nDmrsCdmGrpsNoData:  {:4d}", pdsch_ue_group_dynamic_params[ue_group_id].pDmrsDynPrm->nDmrsCdmGrpsNoData);

        //pCellPrm -> pointer to parent group's dynamic params.
        NVLOGC_FMT(NVLOG_PDSCH, "UE group's parent static cell Idx:    {:4d}", pdsch_ue_group_dynamic_params[ue_group_id].pCellPrm->cellPrmStatIdx);
        NVLOGC_FMT(NVLOG_PDSCH, "UE group's parent dynamic cell Idx:   {:4d}", pdsch_ue_group_dynamic_params[ue_group_id].pCellPrm->cellPrmDynIdx);

        NVLOGC_FMT(NVLOG_PDSCH, "");
    }
    NVLOGC_FMT(NVLOG_PDSCH, "");

    // Print information for all UEs
    int n_ues = cell_group_params->nUes;
    NVLOGC_FMT(NVLOG_PDSCH, "PDSCH cell group dynamic parameters for {} UEs", n_ues);

    cuphyPdschUePrm_t* pdsch_ue_dynamic_params = cell_group_params->pUePrms;

    for (int ue_id = 0; ue_id <  n_ues; ue_id++) {
        NVLOGC_FMT(NVLOG_PDSCH, "------------------------------------");
        NVLOGC_FMT(NVLOG_PDSCH, "UE {}", ue_id);
        NVLOGC_FMT(NVLOG_PDSCH, "------------------------------------");

        NVLOGC_FMT(NVLOG_PDSCH, "scid:         {:4d}", pdsch_ue_dynamic_params[ue_id].scid);
        NVLOGC_FMT(NVLOG_PDSCH, "dmrsScrmId:   {:4d}", pdsch_ue_dynamic_params[ue_id].dmrsScrmId);
        NVLOGC_FMT(NVLOG_PDSCH, "nUeLayers:    {:4d}", pdsch_ue_dynamic_params[ue_id].nUeLayers);
        NVLOGC_FMT(NVLOG_PDSCH, "dmrsPortBmsk: {:#x}", pdsch_ue_dynamic_params[ue_id].dmrsPortBmsk);
        NVLOGC_FMT(NVLOG_PDSCH, "refPoint:     {:4d}", pdsch_ue_dynamic_params[ue_id].refPoint);
        NVLOGC_FMT(NVLOG_PDSCH, "BWPStart:     {:4d}", pdsch_ue_dynamic_params[ue_id].BWPStart);
        NVLOGC_FMT(NVLOG_PDSCH, "beta_drms:     {:f}", pdsch_ue_dynamic_params[ue_id].beta_dmrs);
        NVLOGC_FMT(NVLOG_PDSCH, "beta_qam:      {:f}", pdsch_ue_dynamic_params[ue_id].beta_qam);
        NVLOGC_FMT(NVLOG_PDSCH, "rnti:         {:4d}", pdsch_ue_dynamic_params[ue_id].rnti);
        NVLOGC_FMT(NVLOG_PDSCH, "dataScramId:  {:4d}", pdsch_ue_dynamic_params[ue_id].dataScramId);

        NVLOGC_FMT(NVLOG_PDSCH, "nCw:          {:4d}", pdsch_ue_dynamic_params[ue_id].nCw);

        // Indices to pCwPrms array
        std::stringstream cw_prm_str;
        for (int i = 0; i <  pdsch_ue_dynamic_params[ue_id].nCw; i++) {
            if (i != 0) cw_prm_str << ", ";
            cw_prm_str << std::to_string(pdsch_ue_dynamic_params[ue_id].pCwIdxs[i]);
        }
        NVLOGC_FMT(NVLOG_PDSCH, "pCwIdxs: {{{}}}", cw_prm_str.str());

        // Precoding parameters
        NVLOGC_FMT(NVLOG_PDSCH, "enablePrcdBf: {:4d}", pdsch_ue_dynamic_params[ue_id].enablePrcdBf);
        NVLOGC_FMT(NVLOG_PDSCH, "pmwPrmIdx:    {:4d}", pdsch_ue_dynamic_params[ue_id].pmwPrmIdx);

        // Add pointer to parent UE group
        bool found = false;
        int ue_group_id = 0;
        while ((!found) && (ue_group_id < n_ue_groups)) {
            if (pdsch_ue_dynamic_params[ue_id].pUeGrpPrm == &pdsch_ue_group_dynamic_params[ue_group_id]) {
              found = true;
          }
          ue_group_id += 1;
        }
        NVLOGC_FMT(NVLOG_PDSCH, "parent UE group: {:4d}", found ? (ue_group_id - 1): -1);
    }
    NVLOGC_FMT(NVLOG_PDSCH, "");

    // Print information for all CWs
    int n_cws = cell_group_params->nCws;
    NVLOGC_FMT(NVLOG_PDSCH, "PDSCH cell group dynamic parameters for {} CWs", n_cws);

    cuphyPdschCwPrm_t* pdsch_cw_dynamic_params = cell_group_params->pCwPrms;

    for (int cw_id = 0; cw_id <  n_cws; cw_id++) {
        NVLOGC_FMT(NVLOG_PDSCH, "------------------------------------");
        NVLOGC_FMT(NVLOG_PDSCH, "CW {}", cw_id);
        NVLOGC_FMT(NVLOG_PDSCH, "------------------------------------");

        NVLOGC_FMT(NVLOG_PDSCH, "mcsTableIndex:         {:4d}", pdsch_cw_dynamic_params[cw_id].mcsTableIndex);
        NVLOGC_FMT(NVLOG_PDSCH, "mcsIndex:              {:4d}", pdsch_cw_dynamic_params[cw_id].mcsIndex);
        NVLOGC_FMT(NVLOG_PDSCH, "targetCodeRate:        {:4d}", pdsch_cw_dynamic_params[cw_id].targetCodeRate);
        NVLOGC_FMT(NVLOG_PDSCH, "qamModOrder:           {:4d}", pdsch_cw_dynamic_params[cw_id].qamModOrder);
        NVLOGC_FMT(NVLOG_PDSCH, "rv:                    {:4d}", pdsch_cw_dynamic_params[cw_id].rv);
        NVLOGC_FMT(NVLOG_PDSCH, "tbStartOffset:         {:4d}", pdsch_cw_dynamic_params[cw_id].tbStartOffset);
        NVLOGC_FMT(NVLOG_PDSCH, "tbSize:                {:4d}", pdsch_cw_dynamic_params[cw_id].tbSize);

        NVLOGC_FMT(NVLOG_PDSCH, "n_PRB_LBRM:            {:4d}", pdsch_cw_dynamic_params[cw_id].n_PRB_LBRM);
        NVLOGC_FMT(NVLOG_PDSCH, "maxLayers:             {:4d}", pdsch_cw_dynamic_params[cw_id].maxLayers);
        NVLOGC_FMT(NVLOG_PDSCH, "maxQm:                 {:4d}", pdsch_cw_dynamic_params[cw_id].maxQm);

        // FIXME pointer to parent UE
        bool found = false;
        int ue_id = 0;
        while ((!found) && (ue_id < n_ues)) {
            if (pdsch_cw_dynamic_params[cw_id].pUePrm == &pdsch_ue_dynamic_params[ue_id]) {
              found = true;
          }
          ue_id += 1;
        }
        NVLOGC_FMT(NVLOG_PDSCH, "parent UE:             {:4d}", found ? (ue_id - 1): -1);
    }
    NVLOGC_FMT(NVLOG_PDSCH, "");

    // Print information for all CSI-RS
    int n_csi_rs = cell_group_params->nCsiRsPrms;
    NVLOGC_FMT(NVLOG_PDSCH, "PDSCH cell group dynamic parameters for {} CSI-RS params", n_csi_rs);

    _cuphyCsirsRrcDynPrm* pdsch_csi_rs_dynamic_params = cell_group_params->pCsiRsPrms;

    for (int csirs_idx = 0; csirs_idx <  n_csi_rs; csirs_idx++) {
        NVLOGC_FMT(NVLOG_PDSCH, "------------------------------------");
        NVLOGC_FMT(NVLOG_PDSCH, "CSI RS {}", csirs_idx);
        NVLOGC_FMT(NVLOG_PDSCH, "------------------------------------");

        NVLOGC_FMT(NVLOG_PDSCH, "startRb:               {:4d}", pdsch_csi_rs_dynamic_params[csirs_idx].startRb);
        NVLOGC_FMT(NVLOG_PDSCH, "nRb:                   {:4d}", pdsch_csi_rs_dynamic_params[csirs_idx].nRb);
        NVLOGC_FMT(NVLOG_PDSCH, "freqDomain:            {:4d}", pdsch_csi_rs_dynamic_params[csirs_idx].freqDomain);
        NVLOGC_FMT(NVLOG_PDSCH, "row:                   {:4d}", pdsch_csi_rs_dynamic_params[csirs_idx].row);
        NVLOGC_FMT(NVLOG_PDSCH, "symbL0:                {:4d}", pdsch_csi_rs_dynamic_params[csirs_idx].symbL0);
        NVLOGC_FMT(NVLOG_PDSCH, "symbL1:                {:4d}", pdsch_csi_rs_dynamic_params[csirs_idx].symbL1);
        NVLOGC_FMT(NVLOG_PDSCH, "freqDensity:           {:4d}", pdsch_csi_rs_dynamic_params[csirs_idx].freqDensity);
    }
    NVLOGC_FMT(NVLOG_PDSCH, "");

    // Print precoding information
    int n_precoding_matrices = cell_group_params->nPrecodingMatrices;
    NVLOGC_FMT(NVLOG_PDSCH, "PDSCH cell group dynamic parameters for {} precoding params", n_precoding_matrices);

    cuphyPmW_t* pdsch_pmw_dynamic_params = cell_group_params->pPmwPrms;
    for (int precoding_idx = 0; precoding_idx < n_precoding_matrices; precoding_idx++) {
        NVLOGC_FMT(NVLOG_PDSCH, "------------------------------------");
        NVLOGC_FMT(NVLOG_PDSCH, "Precoding matrix {}", precoding_idx);
        NVLOGC_FMT(NVLOG_PDSCH, "------------------------------------");

        uint8_t n_ports = pdsch_pmw_dynamic_params[precoding_idx].nPorts;
        NVLOGC_FMT(NVLOG_PDSCH, "nPorts:                 {:4d}", n_ports);
        NVLOGC_FMT(NVLOG_PDSCH, "matrix:");
        //max. rows printed even if not relevant; these extra rows should contain {0, 0}
        for (int layer_idx = 0; layer_idx < MAX_DL_LAYERS_PER_TB; layer_idx++) { // Not all layers used
            std::stringstream matrix_row;
            matrix_row.precision(5);
            for (int port_idx = 0; port_idx < n_ports; port_idx++) {
                if (port_idx != 0) matrix_row << ", ";
                __half2 val = pdsch_pmw_dynamic_params[precoding_idx].matrix[layer_idx * n_ports + port_idx];
                matrix_row << std::fixed << "{" << (float)val.x << ", " <<  (float)val.y << "}";
            }
            NVLOGC_FMT(NVLOG_PDSCH, "{}", matrix_row.str());
        }
        NVLOGC_FMT(NVLOG_PDSCH, "");
    }
    NVLOGC_FMT(NVLOG_PDSCH, "");

}


inline void print_pdsch_dynamic(cuphyPdschDynPrms_t* dynamic_params)
{
    if (dynamic_params == nullptr) return;
    NVLOGC_FMT(NVLOG_PDSCH, "PDSCH dynamic parameters");

    const cuphyPdschCellGrpDynPrm_t* cell_group_params = dynamic_params->pCellGrpDynPrm;

    NVLOGC_FMT(NVLOG_PDSCH, "procModeBmsk:       {}", dynamic_params->procModeBmsk);

    //PDSCH data input buffers
    if (dynamic_params->pDataIn == nullptr) return;
    NVLOGC_FMT(NVLOG_PDSCH, "pDataIn.pBufferType:   {}", (dynamic_params->pDataIn->pBufferType == cuphyPdschDataIn_t::CPU_BUFFER) ? "CPU_BUFFER" : "GPU_BUFFER");
    std::stringstream ptbinput_strm;
    for (int i = 0; i <  cell_group_params->nCells; i++) {
        if (i != 0) ptbinput_strm << ", ";
        ptbinput_strm << static_cast<void*>(dynamic_params->pDataIn->pTbInput[i]);
    }
    NVLOGC_FMT(NVLOG_PDSCH, "pDataIn.pTbInput:      {{{}}}", ptbinput_strm.str());

    //PDSCH TB-CRC
    if (dynamic_params->pTbCRCDataIn == nullptr) return;
    NVLOGC_FMT(NVLOG_PDSCH, "pTbCRCDataIn.pBufferType:   {}", (dynamic_params->pTbCRCDataIn->pBufferType == cuphyPdschDataIn_t::CPU_BUFFER) ? "CPU_BUFFER" : "GPU_BUFFER");
    if (dynamic_params->pTbCRCDataIn->pTbInput == nullptr)
    {
        NVLOGC_FMT(NVLOG_PDSCH, "pTbCRCDataIn.pTbInput:      {:p}", (void*)dynamic_params->pTbCRCDataIn->pTbInput);
    } else {
        std::stringstream ptbcrcinput_strm;
        for (int i = 0; i <  cell_group_params->nCells; i++) {
            if (i != 0) ptbcrcinput_strm << ", ";
            ptbcrcinput_strm << static_cast<void*>(dynamic_params->pTbCRCDataIn->pTbInput[i]);
        }
        NVLOGC_FMT(NVLOG_PDSCH, "pTbCRCDataIn.pTbInput:     {{{}}}", ptbcrcinput_strm.str());
    }

    // PDSCH data output buffers
    if (dynamic_params->pDataOut == nullptr) return;
    std::stringstream pdataout_strm;
    for (int i = 0; i <  cell_group_params->nCells; i++) {
        if (i != 0) pdataout_strm << ", ";
        pdataout_strm << static_cast<void*>(dynamic_params->pDataOut->pTDataTx[i].pAddr);
    }
    NVLOGC_FMT(NVLOG_PDSCH, "pDataOut.pTDataTx addr.:    {{{}}}", pdataout_strm.str()); //TODO could add descriptor info too
    NVLOGC_FMT(NVLOG_PDSCH, "");

    print_pdsch_dynamic_cell_group(const_cast<cuphyPdschCellGrpDynPrm_t*>(cell_group_params));
}

inline void read_SSB_dyn_params_from_file(std::vector<cuphyPerSsBlockDynPrms_t>& per_SS_block_dyn_params,
    std::vector<cuphyPerCellSsbDynPrms_t>& per_cell_SSB_dyn_params, std::vector<cuphyPmWOneLayer_t>& ssb_precoding_matrix, hdf5hpp::hdf5_file& input_file, int num_REs, uint16_t& SSBPrecoded_base, int SSB_base=0, int cell_base=0) {

    hdf5hpp::hdf5_dataset dset = input_file.open_dataset("SSTxParams");
    int num_SSBs = dset.get_dataspace().get_dimensions()[0];
    for (int i = 0; i < num_SSBs; i++) {
        cuphyPerSsBlockDynPrms_t& params = per_SS_block_dyn_params[SSB_base + i];
        hdf5hpp::hdf5_dataset_elem h5_params = dset[i];

        params.f0 = h5_params["f0"].as<uint16_t>();
        params.t0 = h5_params["t0"].as<uint8_t>();
        params.blockIndex = h5_params["blockIndex"].as<uint8_t>();
        params.beta_pss = h5_params["beta_pss"].as<float>();
        params.beta_sss = h5_params["beta_sss"].as<float>();
        params.enablePrcdBf = h5_params["enablePrcdBf"].as<uint8_t>();
        params.pmwPrmIdx = 0xFFFF;
        params.cell_index = cell_base;
        if(params.enablePrcdBf)
        {
            std::string pm_dataset_name = "Ssb_PM_W" + std::to_string(i);
            typed_tensor<CUPHY_C_16F, pinned_alloc> ssb_pm_w = typed_tensor_from_dataset<CUPHY_C_16F, pinned_alloc>(input_file.open_dataset(pm_dataset_name.c_str()));
            uint8_t nPorts = cuphy::get_HDF5_dataset_info(input_file.open_dataset(pm_dataset_name.c_str())).layout().dimensions()[0];
            ssb_precoding_matrix[SSBPrecoded_base].nPorts = nPorts; 
            for(uint8_t idx=0; idx<nPorts; idx++)
            {
                ssb_precoding_matrix[SSBPrecoded_base].matrix[idx] = ssb_pm_w(idx);
            }
            params.pmwPrmIdx = SSBPrecoded_base;
            SSBPrecoded_base++;
        }
    }

    cuphyPerCellSsbDynPrms_t& cell_params = per_cell_SSB_dyn_params[cell_base];
    hdf5hpp::hdf5_dataset_elem h5_params = dset[0];
    cell_params.NID = h5_params["NID"].as<uint16_t>();
    cell_params.nHF  = h5_params["nHF"].as<uint16_t>();
    cell_params.Lmax = h5_params["Lmax"].as<uint16_t>();
    cell_params.SFN = h5_params["SFN"].as<uint16_t>();
    cell_params.k_SSB = h5_params["k_SSB"].as<uint16_t>();
    cell_params.nF = num_REs;
    cell_params.slotBufferIdx = cell_base; //FIXME
}



} // namespace cuphy

#endif // !defined(CUPHY_HDF5_HPP_INCLUDED_)
