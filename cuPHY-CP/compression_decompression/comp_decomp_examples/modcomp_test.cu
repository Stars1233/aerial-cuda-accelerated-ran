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

#include <unistd.h>
#include <iostream>
#include <cuda_fp16.h>

#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "QAM_param.cuh"
#include "QAM_comp.cuh"
#include "QAM_decomp.cuh"

// TODO  - include from slot_command.hpp
#include "oran.hpp"
typedef struct {
    uint16_t   modCompScaler; 
    uint8_t    ef;
    uint8_t    extType;
    uint8_t    udIqWidth;
    uint8_t     csf;
} mod_comp_info_t;

class ModCompHdr{
public:
ModCompHdr(hdf5hpp::hdf5_dataset dset)
{
    startSymbolId = dset[0]["startSymbolid"].as<uint8_t>();
    sectionType   = dset[0]["sectionType"  ].as<uint8_t>();
    rb            = dset[0]["rb"           ].as<uint16_t>();
    symInc        = dset[0]["symInc"       ].as<uint8_t>();
    startPrbc     = dset[0]["startPrbc"    ].as<uint16_t>();
    numPrbc       = dset[0]["numPrbc"      ].as<uint16_t>();
    reMask        = dset[0]["reMask"       ].as<uint16_t>();
    udIqWidth     = dset[0]["udIqWidth"    ].as<uint8_t>();
    udCompMeth    = dset[0]["udCompMeth"   ].as<uint8_t>();
    extType       = dset[0]["extType"      ].as<uint8_t>();
    csf           = dset[0]["csf"          ].as<uint8_t>();
    modCompScaler = dset[0]["modCompScaler"].as<float>();
    portIdx       = dset[0]["portIdx"      ].as<uint8_t>();
}
std::string string()
{
    std::string str;
    str += "{startSymbolid=" + std::to_string(startSymbolId);
    str += ", sectionType="  + std::to_string(sectionType  );
    str += ", rb="           + std::to_string(rb           );
    str += ", symInc="       + std::to_string(symInc       );
    str += ", startPrbc="    + std::to_string(startPrbc    );
    str += ", numPrbc="      + std::to_string(numPrbc      );
    str += ", reMask="       + std::to_string(reMask       );
    str += ", udIqWidth="    + std::to_string(udIqWidth    );
    str += ", udCompMeth="   + std::to_string(udCompMeth   );
    str += ", extType="      + std::to_string(extType      );
    str += ", csf="          + std::to_string(csf          );
    str += ", modCompScaler="+ std::to_string(modCompScaler);
    str += ", portIdx="      + std::to_string(portIdx      );
    str +=  "}";
    return str;
}

//    private:
uint8_t  startSymbolId;
uint8_t  sectionType;
uint16_t rb;
uint8_t  symInc;
uint16_t startPrbc;
uint16_t numPrbc;
uint16_t reMask;
uint8_t  udIqWidth;
uint8_t  udCompMeth;
uint8_t  extType;
uint8_t  csf;
float    modCompScaler;
uint8_t  portIdx;

};

class ModCompPayload{
    public:
    ModCompPayload(hdf5hpp::hdf5_dataset dset){
        elements.resize(dset.get_num_elements());
        dset.read(elements.data());
    }
    std::string string()
    {
        std::string str;
        for(auto& elem : elements)
        {
            str += std::to_string(elem) + " ";
        }
        return str;
    }
    size_t getSize(){return elements.size();}
    std::vector<uint8_t> elements;
};

void usage(char* arg)
{
    std::cout << "ModComp Test" << std::endl
        << "\t -h               Print help" << std::endl
        << "\t -f <filename>    Load test vector" << std::endl;

}



int main(int argc, char** argv)
{
    int res = -1;
    int c;
    bool verbose = false;
    std::string filename;
    char optstr[] = "f:hv";
    hdf5hpp::hdf5_file h5file;
    while((c = getopt(argc, argv, optstr)) != -1)
    {
        switch(c)
        {
            case 'f':
                filename.assign(optarg);
                break;
            case 'v':
                verbose = true;
                break;
            case 'h':
            default:
                usage(argv[0]);
                return -1;
        }
    }

    try
    {
        h5file = hdf5hpp::hdf5_file::open(filename.c_str());
    }
    catch(std::exception& e)
    {
        std::cerr << "Unable to open " << filename << std::endl;
        return res;
    }
    std::string dset_name = "nMsg";
    if(!h5file.is_valid_dataset(dset_name.c_str()))
    {
        std::cout << "No modcomp data" << std::endl;
        return res;
    }
    hdf5hpp::hdf5_dataset dset = h5file.open_dataset(dset_name.c_str());
    uint32_t nMsg;
    dset.read(&nMsg);

    std::cout << "Num modcomp elements: " << nMsg << std::endl;
    std::vector<ModCompHdr> comp_hdrs;
    std::vector<ModCompPayload> comp_payld;

    cuphy::typed_tensor<CUPHY_C_16F, cuphy::pinned_alloc> tx_ref_output = cuphy::typed_tensor_from_dataset<CUPHY_C_16F, cuphy::pinned_alloc>(h5file.open_dataset("X_tf_fp16"));

    std::vector<int>           num_prb_vec   (nMsg);
    std::vector<__half2*>      input_vec     (nMsg);
    std::vector<__half2*>      decomp_vec    (nMsg);
    std::vector<QamListParam>  list_param_vec(nMsg);
    std::vector<QamPrbParam*>  prb_param_vec (nMsg);
    std::vector<float2>        scaler_vec    (nMsg);
    std::vector<uint8_t*>      output_vec    (nMsg);

    constexpr unsigned int align = 16; // 16-byte alignment
    int total_payload = 0;
    int total_prbs = 0;
    for(int i=0;i<nMsg;i++)
    {
        std::string hdr_name = "MSG" + std::to_string(i+1) + "_header";
        std::string pyld_name = "MSG" + std::to_string(i+1) + "_payload";
        comp_hdrs.push_back(h5file.open_dataset(hdr_name.c_str()));
        if(verbose)
        {
        std::cout << "dset_name: " << comp_hdrs.back().string() << std::endl;
        }
        num_prb_vec[i] = comp_hdrs.back().numPrbc;
        input_vec[i] = &tx_ref_output(comp_hdrs.back().startPrbc*PRB_NUM_RE,comp_hdrs.back().startSymbolId,0);
        list_param_vec[i].set(static_cast<QamListParam::qamwidth>(comp_hdrs.back().udIqWidth),comp_hdrs.back().csf,0);
        scaler_vec[i] = float2{comp_hdrs.back().modCompScaler,0};
        comp_payld.push_back(h5file.open_dataset(pyld_name.c_str()));
        if(verbose)
        {
        std::cout << "Payload: " << comp_payld.back().string() << std::endl;
        }
        total_payload += comp_payld.back().getSize();
        total_prbs += num_prb_vec[i];
    }
    std::cout << "Total Payload bytes: " << total_payload << std::endl;
    std::cout << "Total prbs: " << total_prbs << std::endl;
    uint8_t* output_buf;

    int *nprbs;
    half **inputs;
    half **decomp;
    __half2 *decomp_buf;
    uint8_t **outputs;
    QamListParam *list_params;
    QamPrbParam **prb_params;
    QamPrbParam *prb_param_buf;
    float2 *scalers;

    CUDA_CHECK(cudaMallocManaged((void **)&output_buf,    total_payload + nMsg*align                ));
    CUDA_CHECK(cudaMallocManaged((void **)&prb_param_buf, total_prbs*sizeof(QamPrbParam)            ));
    CUDA_CHECK(cudaMallocManaged((void **)&nprbs,            num_prb_vec.size()*sizeof(int)         ));
    CUDA_CHECK(cudaMallocManaged((void **)&inputs,             input_vec.size()*sizeof(__half2*)    ));
    CUDA_CHECK(cudaMallocManaged((void **)&decomp,           num_prb_vec.size()*sizeof(__half2*)    ));
    CUDA_CHECK(cudaMallocManaged((void **)&decomp_buf,  (PRB_NUM_RE*total_prbs)*sizeof(__half2)     ));
    CUDA_CHECK(cudaMallocManaged((void **)&outputs,           output_vec.size()*sizeof(uint8_t*)    ));
    CUDA_CHECK(cudaMallocManaged((void **)&list_params,   list_param_vec.size()*sizeof(QamListParam)));
    CUDA_CHECK(cudaMallocManaged((void **)&prb_params,     prb_param_vec.size()*sizeof(QamPrbParam*)));
    CUDA_CHECK(cudaMallocManaged((void **)&scalers,           scaler_vec.size()*sizeof(float2)      ));

    int prb_offset = 0;
    int output_offset = 0;
    int decomp_offset = 0;
    for(int i=0;i<nMsg;i++)
    {
        prb_param_vec[i] = prb_param_buf + prb_offset;
        output_vec[i]    = output_buf    + output_offset;
        decomp_vec[i]    = decomp_buf    + decomp_offset;
        prb_offset       += num_prb_vec[i];
        output_offset    += ((comp_payld[i].getSize() + align-1) & ~(align-1));
        decomp_offset    += (num_prb_vec[i]*PRB_NUM_RE);
        // set QamPrbParam values
        QamPrbParam prb_param;
        prb_param.set(comp_hdrs[i].reMask,0);
        std::vector<QamPrbParam> tmp_vec(num_prb_vec[i],prb_param);
        cudaMemcpy(prb_param_vec[i],tmp_vec.data(),tmp_vec.size()*sizeof(QamPrbParam), cudaMemcpyHostToDevice);
    }

    // TODO copy vector data to GPU mem
    CUDA_CHECK(cudaMemcpy(nprbs,            num_prb_vec.data(),    num_prb_vec.size()*sizeof(int)         , cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(inputs,             input_vec.data(),      input_vec.size()*sizeof(__half2*)    , cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(outputs,           output_vec.data(),     output_vec.size()*sizeof(uint8_t*)    , cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(list_params,   list_param_vec.data(), list_param_vec.size()*sizeof(QamListParam), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(prb_params,     prb_param_vec.data(),  prb_param_vec.size()*sizeof(QamPrbParam*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scalers,           scaler_vec.data(),     scaler_vec.size()*sizeof(float2)      , cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(decomp,            decomp_vec.data(),     decomp_vec.size()*sizeof(__half2*)    , cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());

    // Compress
    QAM_Comp::gpu_compress_QAM_lists(inputs, list_params, prb_params, scalers, outputs, nprbs, nMsg);
    //QAM_Comp::cpu_compress_QAM_lists(inputs, list_params, prb_params, scalers, cpu_outputs, nprbs, nlists);

    CUDA_CHECK(cudaDeviceSynchronize());
    res = 0;

    uint8_t tmp[273*PRB_NUM_RE]={0};
    for(int i = 0;i<comp_payld.size();i++)
    {
        int payloadSize = comp_payld[i].getSize();
        CUDA_CHECK(cudaMemcpy(tmp,output_vec[i],payloadSize,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        if(memcmp(tmp,comp_payld[i].elements.data(),comp_payld[i].elements.size()))
        {
            std::cerr << "Payload "<< i <<" does not match" << std::endl;
            res++;

            std::cout << "Res Payload[" << i << "]: ";
            for(int j=0;j<payloadSize;j++)
            {
                std::cout << std::to_string(tmp[j]) <<" ";
            }
            std::cout << std::endl;
        }
    }

    QAM_Decomp::gpu_decompress_QAM_lists(outputs, list_params, prb_params, scalers, decomp, nprbs, nMsg);
    CUDA_CHECK(cudaDeviceSynchronize());
    if(verbose)
    {
    for(int i = 11; i<12;i++)
    {
        int nPrb = num_prb_vec[i];
        std::cout << "Compressed " << i << ":\n\t";
        __half2 hlf_tmp = *decomp_vec[i];
        for(int j = 0; j<nPrb*PRB_NUM_RE; j++)
        {
            std::cout << "(" << __half2float(decomp_vec[i][j].x) << ", " << __half2float(decomp_vec[i][j].y) <<") ";
        }
        std::cout << std::endl << "Ref: \n\t";
        for(int j = 0; j<nPrb*PRB_NUM_RE; j++)
        {
            std::cout << "(" << __half2float(input_vec[i][j].x) << ", " << __half2float(input_vec[i][j].y) <<") ";
        }
        std::cout << std::endl;
    }
    }
std::cout << res << " mismatches" << std::endl;
if(res == 0)
{
    std::cout << "PASS" << std::endl;
} else
{
    std::cout << "FAIL" << std::endl;
}
    return res;

}
