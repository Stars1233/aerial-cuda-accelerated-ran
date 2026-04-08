#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pandas as pd
import glob
import os
from abc import ABC, abstractmethod
from enum import Enum
import json

class OutputFormat(Enum):
    FEATHER = 0
    PARQUET = 1
    HDF5 = 2
    JSON = 3
    CSV = 4

#Parsing function used by top level python scripts
#
# Takes an input that is either a list of folders or a list containing alternating phy/testmac/ru logs
#
# Transforms data to list of lists to match IO formats
def input_data_parser(input_data_list):
    #Prepare input_data for PerfMetricsIO
    if(os.path.isdir(input_data_list[0])):
        #Assume we are processing binary files
        num_inputs = len(input_data_list)
        input_data_list = [[aa] for aa in input_data_list]
    elif(os.path.isfile(input_data_list[0])):
        #Assume we are processing log files directly
        num_inputs = len(input_data_list)//3
        input_data_list = [[input_data_list[3*ii],input_data_list[3*ii+1],input_data_list[3*ii+2]] for ii in range(num_inputs)]
    else:
        #We have failed to find the specified file
        print("ERROR :: unable to find %s in input_data_parser"%input_data_list[0])
        return None

    return input_data_list

#Function useful for comparing two dataframes have identical data
def hash_dataframe(df):
    import hashlib
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def str2format(str_format):
    if(str_format.lower() == 'feather'):
        return OutputFormat.FEATHER
    elif(str_format.lower() == 'parquet'):
        return OutputFormat.PARQUET
    elif(str_format.lower() == 'hdf5'):
        return OutputFormat.HDF5
    elif(str_format.lower() == 'json'):
        return OutputFormat.JSON
    elif(str_format.lower() == 'csv'):
        return OutputFormat.CSV
    else:
        return None
    
def format2ext(output_format):
    if(output_format==OutputFormat.FEATHER):
        return ".feather"
    elif(output_format==OutputFormat.PARQUET):
        return ".parquet"
    elif(output_format==OutputFormat.HDF5):
        return ".hdf5"
    elif(output_format==OutputFormat.JSON):
        return ".json"
    elif(output_format==OutputFormat.CSV):
        return ".csv"
    else:
        return ""
    
def ext2readfunc(ext):
    if(ext == ".feather"):
        return pd.read_feather
    elif(ext == ".parquet"):
        return pd.read_parquet
    elif(ext == ".hdf5"):
        return pd.read_hdf
    elif(ext == ".json"):
        return pd.read_json
    elif(ext == ".csv"):
        return pd.read_csv
    else:
        print("Error :: Unknown ext in ext2loadfunc")
        return None
    
def ext2writefuncname(ext):
    if(ext == ".feather"):
        return "to_feather"
    elif(ext == ".parquet"):
        return "to_parquet"
    elif(ext == ".hdf5"):
        return "to_hdf"
    elif(ext == ".json"):
        return "to_json"
    elif(ext == ".csv"):
        return "to_csv"
    else:
        print("Error :: Unknown ext in ext2writefuncname")
        return None
    

#Abstract Class that provides generic parsing, saving, and loading of Parsenator results
class ParsenatorIO(ABC):
    def __init__(self):
        self.data = None
        self.metadata_file = "metadata.json"

        #Deriving classes must specify
        #self.type_name (str)
        #self.df_files (list of str)

    def save(self,output_folder,str_output_format='feather'):

        #Make sure output folder exists
        if(not os.path.isdir(output_folder)):
            os.makedirs(output_folder, exist_ok=True)
        
        #Write metadata to file
        mfile = output_folder + "/" + self.metadata_file
        metadata = {'type_name': self.type_name}
        with open(mfile, 'w') as fp:
            json.dump(metadata, fp)

        #Determine proper format
        output_format = str2format(str_output_format)

        if self.data:
            for key in self.data.keys():

                if(len(self.data[key])==0):
                    #Special case - it appears these binary writers have trouble with empty dataframes
                    #Write empty file with .empty extension
                    ext = ".empty"
                    output_file = output_folder + "/%s%s"%(key,ext)
                    open(output_file, 'a').close()
                    
                else:
                    #Determine proper extension and output filename
                    ext = format2ext(output_format)
                    output_file = output_folder + "/%s%s"%(key,ext)

                    #Retrieve write function and call that function to write file
                    wfn = ext2writefuncname(ext)
                    wf = getattr(self.data[key], wfn)
                    if(wfn == 'to_hdf'):
                        #Odd, but hdf writing requires a label with no defaults...
                        wf(output_file,key='data')
                    else:
                        wf(output_file)

            return True
        else:
            print("ParsenatorIO Error :: Attempted save when no data initialized")
            return False

    #Loads data into self.data
    def load(self,input_folder,start_tir,end_tir):
        self.data = {}

        #Attempt to read metadata file to verify this data is the correct type
        mfile = input_folder + "/" + self.metadata_file
        metadata_found = os.path.exists(mfile)
        if(metadata_found):
            with open(mfile, 'r') as fp:
                metadata = json.load(fp)

            if(not (metadata['type_name'] == self.type_name)):
                print("ERROR :: Folder %s is type %s, not desired type of %s"%(input_folder,metadata['type_name'],self.type_name))
                return False
        else:
            print("WARNING :: Ignoring metadata checking (this may be old data).  Assuming folder %s is of type %s"%(input_folder,self.type_name))

        #Loop over expected files and load based on extension
        for df_file in self.df_files:
            current_files = glob.glob(input_folder+"/%s.*"%df_file)
            if(len(current_files) > 0):
                if(len(current_files) > 1):
                    #Warn if multiple files are found
                    print("WARNING :: multiple files found in folder %s labeled %s:"%(input_folder,df_file))
                    for ff in current_files:
                        print(ff)
                    
                    #Remove empty file as option
                    current_files = [aa for aa in current_files if aa.find(".empty") < 0]

                #Use first file in the list
                first_file = current_files[0]
                aa,ext = os.path.splitext(first_file)

                if(ext == ".empty"):
                    #Special case used for empty dataframes
                    self.data[df_file] = pd.DataFrame()

                else:
                    #Call read function to read in data
                    readfunc = ext2readfunc(ext)
                    if readfunc:
                        print("Loading data from file %s..."%first_file)
                        self.data[df_file] = readfunc(first_file)

                        if('tir' in self.data[df_file].columns):
                            print("Loaded data from %f to %f sec in run"%(self.data[df_file].tir.min(),self.data[df_file].tir.max()))
                            self.data[df_file] = self.data[df_file][(self.data[df_file].tir >= start_tir) & (self.data[df_file].tir <= end_tir)]
                            print("Trimmed data from %f to %f sec in run (trim: %f to %f)"%(self.data[df_file].tir.min(),self.data[df_file].tir.max(),start_tir,end_tir))
            else:
                print("ERROR :: Unable to load df_file %s for input_folder %s"%(df_file,input_folder))
                return False
            
        #Make sure that if we have a tir-filtered dataset, at least one tir df results in data
        #Note - I can see edge cases where tir data may not be populated in the log, but I think this check is worth it
        contains_tir_filtered_data = any(['tir' in self.data[df_file].columns for df_file in self.df_files])
        if(contains_tir_filtered_data):
            all_empty = all([(len(self.data[df_file]) == 0) for df_file in self.df_files if 'tir' in self.data[df_file].columns])
            if(all_empty):
                print("ERROR :: All dataframes are empty for input_folder=%s.  Maybe check the trimming?"%input_folder)
                return False
            
        #We have successfully loaded everything
        return True

    @abstractmethod
    #Returns success/failure
    def parse(self,input_data,start_tir,end_tir,slot_list=None,num_proc=8):
        #Deriving class must parse data based on either folder input (call load) or
        # phy/testmac/log filename input
        pass

class PerfMetricsIO(ParsenatorIO):
    def __init__(self):
        super().__init__()
        self.type_name = "PerfMetrics"
        self.df_files = ["df_ti","df_l2","df_gpu","df_compression","df_testmac","df_du_ontime","df_ru_ontime","df_tick"]

    def parse(self,input_data,start_tir,end_tir,slot_list=None,num_proc=8):
        if(type(input_data) != list):
            print("ERROR :: input_data must be list in parsing function")
            return False

        if(len(input_data)==1 and os.path.isdir(input_data[0])):
            #TODO - right now we are not performing any metadata checks
            # Need to make sure loaded data is precisely what we wanted to load

            #Load from file
            load_success = self.load(input_data[0],start_tir,end_tir)

            if(load_success):

                #Apply slot filter if applicable
                if(slot_list is not None):
                    for df_file in self.df_files:
                        if('slot' in self.data[df_file].columns):
                            self.data[df_file] = self.data[df_file][self.data[df_file].slot.isin(slot_list)]

                return True
            else:
                print("ERROR :: Problem loading %s"%input_data[0])
                return False

        elif(len(input_data) == 3 and os.path.isfile(input_data[0]) and os.path.isfile(input_data[1]) and os.path.isfile(input_data[2])):
            #Run parsing
            from aerial_postproc.parsenator_logparse import perf_metrics_parse
            df_ti_list,df_l2_list,df_gpu_list,df_compression_list,df_testmac_list,df_du_ontime_list,df_ru_ontime_list,df_tick_list = perf_metrics_parse([input_data[0]],
                                                                                                                                                        [input_data[1]],
                                                                                                                                                        [input_data[2]],
                                                                                                                                                        start_tir,
                                                                                                                                                        end_tir,
                                                                                                                                                        slot_list=slot_list,
                                                                                                                                                        num_proc=num_proc)
            self.data = {}
            self.data["df_ti"] = df_ti_list[0]
            self.data["df_l2"] = df_l2_list[0]
            self.data["df_gpu"] = df_gpu_list[0]
            self.data["df_compression"] = df_compression_list[0]
            self.data["df_testmac"] = df_testmac_list[0]
            self.data["df_du_ontime"] = df_du_ontime_list[0]
            self.data["df_ru_ontime"] = df_ru_ontime_list[0]
            self.data["df_tick"] = df_tick_list[0]
            return True

        else:
            print("Error:: Invalid input_data %s"%input_data)
            return False
            

class LatencySummaryIO(ParsenatorIO):
    def __init__(self,mmimo_enable=False):
        super().__init__()

        self.mmimo_enable = mmimo_enable

        self.type_name = "LatencySummary"
        self.df_files = ["df_du_rx_timing","df_du_rx_srs_timing","df_ti","df_tx_timing","df_tx_timing_sum","df_ru_rx_timing"]

    def parse(self,input_data,start_tir,end_tir,slot_list=None,num_proc=8):
        if(type(input_data) != list):
            print("ERROR :: input_data must be list in parsing function")
            return False

        if(len(input_data)==1 and os.path.isdir(input_data[0])):
            #TODO - right now we are not performing any metadata checks
            # Need to make sure loaded data is precisely what we wanted to load

            #Load from file
            load_success = self.load(input_data[0],start_tir,end_tir)

            if(load_success):

                #Apply slot filter if applicable
                if(slot_list is not None):
                    for df_file in self.df_files:
                        if('slot' in self.data[df_file].columns):
                            self.data[df_file] = self.data[df_file][self.data[df_file].slot.isin(slot_list)]

                return True
            else:
                print("ERROR :: Problem loading %s"%input_data[0])
                return False

        elif(len(input_data) == 3 and os.path.isfile(input_data[0]) and os.path.isfile(input_data[1]) and os.path.isfile(input_data[2])):
            #Run parsing
            from aerial_postproc.parsenator_logparse import latency_parsing
            df_du_rx_timing_list,df_du_rx_srs_timing_list,df_ti_list,df_tx_timing_list,df_tx_timing_sum_list,df_ru_rx_timing_list = latency_parsing([input_data[0]],
                                                                                                                                                    [input_data[2]],
                                                                                                                                                    start_tir,
                                                                                                                                                    end_tir,
                                                                                                                                                    slot_list=slot_list,
                                                                                                                                                    num_proc=num_proc,
                                                                                                                                                    mmimo_enable=self.mmimo_enable)

            self.data = {}
            self.data["df_du_rx_timing"] = df_du_rx_timing_list[0]
            self.data["df_du_rx_srs_timing"] = df_du_rx_srs_timing_list[0]
            self.data["df_ti"] = df_ti_list[0]
            self.data["df_tx_timing"] = df_tx_timing_list[0]
            self.data["df_tx_timing_sum"] = df_tx_timing_sum_list[0]
            self.data["df_ru_rx_timing"] = df_ru_rx_timing_list[0]
            return True

        else:
            print("Error:: Invalid input_data %s"%input_data)
            return False
            
#Parse a string list of integer values (all ranges inclusive)
#Example "1-4,7,9-11" --> [1,2,3,4,7,9,10,11]
def parse_range_list(range_list):
    result = []
    for part in range_list.split(","):
        if '-' in part:
            start, end = part.split('-')
            result.extend(range(int(start),int(end)+1))
        else:
            result.append(int(part))

    return result

def parse_labels(labels,num_files):
    #Initialize to generic file index if not specified
    if not labels:
        labels = ["Unlabeled File %02i"%ii for ii in range(num_files)]
    while(len(labels) < num_files):
        labels.append("Unlabeled File %02i"%(len(labels)-1))

    return labels

def parse_all_PerfMetricsIO(input_data_list,ignore_duration,max_duration,num_proc,slot_selection=None):
    #Get slot list if enabled
    if(slot_selection):
        #Parse the slot selection values
        slot_list = parse_range_list(slot_selection)
    else:
        slot_list = None

    #Prepare input_data for PerfMetricsIO
    from aerial_postproc.parsenator_formats import input_data_parser
    input_data_list = input_data_parser(input_data_list)
    if(len(input_data_list) == 0):
        print("ERROR :: Unable to parse input data %s.  Must be preparsed folders or phy/testmac/ru alternating logs"%input_data_list)
        return

    #Parse or load all data
    df_ti_list = []
    df_l2_list = []
    df_gpu_list = []
    df_compression_list = []
    df_tick_list = []
    df_testmac_list = []
    for input_data in input_data_list:
        pmio = PerfMetricsIO()
        parse_success = pmio.parse(input_data,ignore_duration,max_duration,slot_list=slot_list,num_proc=num_proc)
        if(parse_success):
            df_ti_list.append(pmio.data['df_ti'])
            df_l2_list.append(pmio.data['df_l2'])
            df_gpu_list.append(pmio.data['df_gpu'])
            df_compression_list.append(pmio.data['df_compression'])
            df_tick_list.append(pmio.data['df_tick'])
            df_testmac_list.append(pmio.data['df_testmac'])
        else:
            print("ERROR :: Invalid input_data %s"%input_data)
            return
        
    return df_ti_list,df_l2_list,df_gpu_list,df_compression_list,df_tick_list,df_testmac_list,slot_list
