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
import time
from multiprocessing import Pool

#Revamped message processing class that includes multi-threaded capability
class Parsenator:

    #Takes in the log name and list of parsers.  Parsers process a single line as input
    # and return a list of dictionaries representing successfully parsed data (empty list
    # if nothing successfully parsed)
    #
    #Also takes in a list of column filters (on entry in list for each parser).  Columns filters
    # are lists of column names.  If a column filter is None, all columns are returned.  If a list is specified
    # then only the columns specified will be returned.
    #
    #Note: While 't0_timestamp' is not a required field, the start/end tir can only be applied
    # to messages containing this field.  This means that when applying a time filter messages that
    # do not contain 't0_timestamp' will not properly be filtered.
    def __init__(self,log,parser_list,column_filter_list,start_tir=None,end_tir=None,num_processes=8,ref_t0=None):
        self.log = log
        self.parser_list = parser_list
        self.column_filter_list = column_filter_list
        self.start_tir = start_tir
        self.end_tir = end_tir
        self.num_processes = num_processes
        self.ref_t0 = ref_t0

        #Must have a definition for column filters for every parser
        assert(len(self.parser_list)==len(self.column_filter_list))

    @staticmethod
    def apply_time_filter(data,start_tir,end_tir,ref_t0):
        good_indices = []
        finished = False
        max_tir = None

        #Determine if this data has t0_timestamp key
        can_apply_filter = (len(data) > 0 and 't0_timestamp' in data[0].keys() and start_tir is not None and end_tir is not None and ref_t0 is not None)

        if(can_apply_filter):
            #Only apply time filter to data that contains t0 timestamp key
            for ii in range(len(data)):
                current_tir = (data[ii]['t0_timestamp'] - ref_t0) / 1e9

                #Apply filter and determine if we are finished
                if(start_tir <= current_tir and current_tir <= end_tir):
                    good_indices.append(ii)
                elif(current_tir > end_tir):
                    finished = True

                #Keep track of max tir
                if(max_tir is None):
                    max_tir = current_tir
                else:
                    max_tir = max(max_tir,current_tir)
                    
        else:
            #We cannot apply a filter, let all data pass
            good_indices = [ii for ii in range(len(data))]
            finished = False
            max_tir = None

        #Apply filter and return resulting data
        return [data[jj] for jj in range(len(data)) if jj in good_indices], finished, max_tir

    @staticmethod
    def parsing_func(line_list, parser_list, column_filter_list, start_tir, end_tir, ref_t0):
        output_results = [[] for ii in range(len(parser_list))]

        #Process all lines
        for line in line_list:
            for ii,parser in enumerate(parser_list):
                current_column_filter = column_filter_list[ii]
                results = parser(line)
                for res in results:
                    if(current_column_filter is not None):
                        #Down-select only keys that we need
                        temp_dict = {key: val for key, val in res.items() if key in column_filter_list[ii]}
                        output_results[ii].append(temp_dict)
                    else:
                        output_results[ii].append(res)

        #Apply time filter to all results
        finished = False
        max_tir = None
        for ii in range(len(output_results)):
            #Filter results based on time
            output_results[ii],current_finished,current_max_tir = Parsenator.apply_time_filter(output_results[ii],start_tir,end_tir,ref_t0)
            finished = finished or current_finished #We are done if any of the time filters have gone beyond end_tir
            if(current_max_tir is not None):
                if(max_tir is None):
                    max_tir = current_max_tir
                else:
                    max_tir = max(max_tir,current_max_tir)

        return [output_results,finished,max_tir]

    #For each parser in the list, returns a list of dataframes
    # progress_queue - A queue that this thread can put float values between 0 and 100 to indicate progress
    # stop_queue - A queue that this thread checks.  If anything is placed in it, the processing will stop
    #   and return what it has processed thus far
    def parse(self,progress_queue=None,stop_queue=None,max_lines=None):
        time1 = time.time()
        print(f"Parsing log %s, num_processes=%i"%(self.log,self.num_processes))

        NUM_PROCESSES = self.num_processes
        WORKLOADS_PER_RUN = 100
        TOTAL_LINES_PER_WORKLOAD = 100

        with Pool(NUM_PROCESSES) as pp:
            output_results = [[] for ii in range(len(self.parser_list))]

            fid = open(self.log)
            finished = False
            line_idx = 0
            max_tir = None

            current_line_list = []
            current_work_list = []
            while(not finished):
                line = fid.readline()
                line_idx += 1
                line_is_empty = (line == "")

                current_line_list.append(line)

                if(len(current_line_list) >= TOTAL_LINES_PER_WORKLOAD or line_is_empty and len(current_line_list) > 0):
                    #Add this workload to the list
                    current_work_list.append((current_line_list,self.parser_list,self.column_filter_list,self.start_tir,self.end_tir,self.ref_t0))
                    current_line_list = []

                if(len(current_work_list) >= WORKLOADS_PER_RUN or line_is_empty and len(current_work_list) > 0):
                    ##RUN PARSING - We now have enough workloads
                    if(self.ref_t0 is None):
                        #Process sequentially in main thread until we set a ref t0
                        for aa,bb,cc,dd,ee,ff in current_work_list:
                            current_results,current_finished,current_max_tir = Parsenator.parsing_func(aa,bb,cc,dd,ee,ff)

                            #Add these results, and set ref_t0 in the process
                            for ii,result in enumerate(current_results):
                                output_results[ii].extend(result)
                                if(len(output_results[ii]) > 0 and 't0_timestamp' in output_results[ii][0].keys()):
                                    if(self.ref_t0 is None):
                                        self.ref_t0 = output_results[ii][0]['t0_timestamp']
                                    else:
                                        self.ref_t0 = min(self.ref_t0,output_results[ii][0]['t0_timestamp'])

                        if(self.ref_t0 is not None):
                            #Now that we have established a ref_t0, we must apply the time filter on all of the current data
                            # (during the first pass the parsing could not apply the time filter)
                            for ii in range(len(output_results)):
                                output_results[ii],current_finished,current_max_tir = Parsenator.apply_time_filter(output_results[ii],self.start_tir,self.end_tir,self.ref_t0)
                                finished = finished or current_finished
                                if(current_max_tir is not None):
                                    if(max_tir is None):
                                        max_tir = current_max_tir
                                    else:
                                        max_tir = max(max_tir,current_max_tir)

                    else:
                        #Now that ref_t0 is established, we can run multi-threaded processing
                        for result_set in pp.starmap(Parsenator.parsing_func, current_work_list):
                            current_results = result_set[0]
                            current_finished = result_set[1]
                            current_max_tir = result_set[2]
                            for ii,result in enumerate(current_results):
                                output_results[ii].extend(result)

                            finished = finished or current_finished

                            if(current_max_tir is not None):
                                if(max_tir is None):
                                    max_tir = current_max_tir
                                else:
                                    max_tir = max(max_tir,current_max_tir)

                    #Reset the work list
                    current_work_list = []

                #Check our max_lines limit if applicable
                max_lines_processed = (max_lines is not None and line_idx>=max_lines)

                finished = finished or line_is_empty or max_lines_processed

                # Print progress periodically
                if (line_idx % 100000 == 0):
                    time2 = time.time()
                    if(self.end_tir is not None):
                        if(max_tir is None):
                            print("[%9.3f] Processed line %i (??? of %.1fsec)..."%(time2 - time1,line_idx,self.end_tir))
                        else:
                            print("[%9.3f] Processed line %i (%.1f of %.1fsec)..."%(time2 - time1,line_idx,max_tir,self.end_tir))

                            #Note: we can only provide progress if max_tir is specified
                            if(progress_queue is not None):
                                progress_queue.put(100*(max_tir/self.end_tir))
                            if(stop_queue is not None and not stop_queue.empty()):
                                _ = stop_queue.get()
                                return None
                            
                    else:
                        print("[%9.3f] Processed line %i (no end_tir specified)..."%(time2 - time1,line_idx))


            fid.close()

            df_list = [pd.DataFrame(aa) for aa in output_results]

            #Add TIR field if t0_timestamp is in data
            for ii in range(len(df_list)):
                has_t0 = 't0_timestamp' in df_list[ii].columns
                has_start = 'start_timestamp' in df_list[ii].columns
                has_end = 'end_timestamp' in df_list[ii].columns
                if has_t0:
                    #Add Time In Run (TIR) in seconds
                    df_list[ii]['tir'] = (df_list[ii]['t0_timestamp'] - self.ref_t0) / 1e9
                if has_t0 and has_start and has_end:
                    #Add deadlines times and duration in usec
                    df_list[ii]['start_deadline'] = (df_list[ii]['start_timestamp'] - df_list[ii]['t0_timestamp']) / 1000.
                    df_list[ii]['end_deadline'] = (df_list[ii]['end_timestamp'] - df_list[ii]['t0_timestamp']) / 1000.
                    df_list[ii]['duration'] = df_list[ii]['end_deadline'] - df_list[ii]['start_deadline']

            if(progress_queue is not None):
                progress_queue.put(100.0)

        return df_list

    
    #Useful for retrieving the calculated ref_t0 after processing is complete
    def get_ref_t0(self):
        return self.ref_t0


if __name__ == '__main__':
    #UNIT TEST - example with testmac log
    import argparse
    import time
    parser = argparse.ArgumentParser(
        description="Unit test for parsenator"
    )
    parser.add_argument(
        "testmac_log", help="Input testmac log containing MAC.PROCESSING_TIMES message"
    )
    args = parser.parse_args()

    from aerial_postproc.logparse import parse_testmac_times
    start_tir = 0.0
    end_tir = 30.0
    #Example using multiple parsers (each parser running testmac parsing)
    p1 = Parsenator(args.testmac_log,
                    [parse_testmac_times,parse_testmac_times,parse_testmac_times,parse_testmac_times],
                    [None,None,None,None],
                    start_tir,end_tir,num_processes=1)
    p2 = Parsenator(args.testmac_log,
                    [parse_testmac_times,parse_testmac_times,parse_testmac_times,parse_testmac_times],
                    [None,None,None,None],
                    start_tir,end_tir,num_processes=8)

    time1 = time.time()
    data1 = p1.parse()
    time2 = time.time()
    print("Single-threaded processed in %.6f seconds, ref_t0=%i"%((time2 - time1), p1.get_ref_t0()))

    time1 = time.time()
    data2 = p2.parse()
    time2 = time.time()
    print("Multi-threaded processed in %.6f seconds, ref_t0=%s"%((time2 - time1), p2.get_ref_t0()))

    print("Completed parsenator unit test.")
