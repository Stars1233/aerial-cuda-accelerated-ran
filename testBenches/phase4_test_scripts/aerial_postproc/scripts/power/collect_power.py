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

import argparse
import os
import time
import glob
import subprocess
import csv
import ntplib
import traceback

# Import Raritan PDU utilities
from pdu_utils import RaritanPDU

VERBOSE = False

#Retrieves current ntp offset
#Note: Aerial servers do not lock to ntp, as they use PTP to ensure relative time between DU and RU
#      The goal here is to collect the data using an absolute time base.
def get_ntp_offset(num_tries):
    for ii in range(num_tries):
        try:
            client = ntplib.NTPClient()
            response = client.request('pool.ntp.org')
            return response.offset
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print('Could not sync with time server.')
            time.sleep(0.2)
    return 0

def print_wrapper(data):
    global VERBOSE
    if(VERBOSE):
        print(data)

def ls(match_string):
    return glob.glob(match_string)

def read_file(file_name):
    with open(file_name,'r') as fid:
        data = fid.read()

    return data

def get_gpu_data():
    print_wrapper("Retrieving GPU data...")
    #Note: use "nvidia-smi --help-query-gpu" to see descriptions of all fields
    fields = ["clocks.current.graphics","clocks.current.sm","clocks.current.memory","clocks.current.video",
              "power.limit","power.draw.average","power.draw.instant",
              "temperature.gpu.tlimit","temperature.gpu","temperature.memory",
              "utilization.gpu","utilization.memory",
              "clocks_event_reasons.active",
              "clocks_event_reasons_counters.sw_power_cap",
              "clocks_event_reasons_counters.hw_thermal_slowdown",
              "clocks_event_reasons_counters.hw_power_brake_slowdown",
              "clocks_event_reasons_counters.sw_thermal_slowdown",
              "clocks_event_reasons_counters.sync_boost"]
    field_types = [float,float,float,float,
                   float,float,float,
                   float,float,float,
                   float,float,
                   str,
                   int,int,int,int,int]

    cmd = "nvidia-smi --id=0 --query-gpu=%s --format=csv,noheader,nounits"%(",".join(fields))
    print_wrapper(cmd)
    values1 = subprocess.check_output(cmd, shell=True, text=True)
    cmd = "nvidia-smi -q -d Power"
    print_wrapper(cmd)
    values2 = subprocess.check_output(cmd, shell=True, text=True)

    #Post process the first command
    print_wrapper("values1:\n%s"%values1)
    values1 = values1.split(",")
    values1 = [aa.strip() for aa in values1]
    print_wrapper(values1)

    out_dict = {}
    for ii in range(len(fields)):
        value = values1[ii].strip()
        field_name = fields[ii]
        field_type = field_types[ii]
        
        # Handle 'N/A' values and type conversion
        if value == 'N/A' or value == '[N/A]':
            print(f"WARNING: GPU field '{field_name}' returned 'N/A'")
            if field_type == float:
                print_wrapper(f"WARNING: GPU field '{field_name}' returned 'N/A', using 0.0")
                out_dict[field_name] = 0.0
            elif field_type == int:
                print_wrapper(f"WARNING: GPU field '{field_name}' returned 'N/A', using 0")
                out_dict[field_name] = 0
            elif field_type == str:
                print_wrapper(f"INFO: GPU field '{field_name}' returned 'N/A', keeping as string")
                out_dict[field_name] = 'N/A'
            else:
                print_wrapper(f"WARNING: GPU field '{field_name}' returned 'N/A', using None")
                out_dict[field_name] = None
        else:
            try:
                out_dict[field_name] = field_type(value)
            except (ValueError, TypeError) as e:
                print(f"WARNING: GPU field '{field_name}' conversion failed for value '{value}': {e}")
                if field_type == float:
                    out_dict[field_name] = 0.0
                elif field_type == int:
                    out_dict[field_name] = 0
                elif field_type == str:
                    out_dict[field_name] = str(value)
                else:
                    out_dict[field_name] = None

    #Post process the second command
    print_wrapper("values2:\n%s"%values2)

    return out_dict

def init_module_data():
    hwmon_name_files = ls("/sys/class/hwmon/hwmon*/device/power1_oem_info")

    print("Setting module averaging interval to 1000ms.")

    for file in hwmon_name_files:
        cmd1 = "echo 1000 | sudo tee %s"%file.replace("power1_oem_info","power1_average_interval")
        print_wrapper(cmd1)
        subprocess.check_output(cmd1, shell=True, text=True)

def get_module_data():
    print_wrapper("Retrieving module data...")

    #Grab power information for GH module, CPU+SysIO, CPU, and SysIO
    hwmon_name_files = ls("/sys/class/hwmon/hwmon*/device/power1_oem_info")
    out_dict = {}
    for file in hwmon_name_files:
        name = read_file(file).strip()
        power = int(read_file(file.replace("power1_oem_info","power1_average"))) / 1e6
        interval = int(read_file(file.replace("power1_oem_info","power1_average_interval"))) / 1000.0
        print_wrapper("%s: %f watts over %f interval"%(name,power,interval))
        out_dict[name+" Power"] = power
        out_dict[name+" Interval"] = interval

    #Grab thermal information
    thermal_types = ls("/sys/class/thermal/thermal_zone*/type")
    for file in thermal_types:
        #Value is read in millidegrees Celcius
        name = read_file(file.replace("type","device/description")).lstrip("Thermal Zone ").strip()
        degrees = int(read_file(file.replace("type","temp"))) / 1e3
        # current_type = read_file(file).strip()

        out_dict["%s Temp"%name] = degrees

        #Check for trip points
        for tp_file in ls(file.replace("type","trip_point_*_type")):
            tp_type = read_file(tp_file).strip()
            tp_temp = int(read_file(tp_file.replace("_type","_temp"))) / 1e3
            out_dict["%s %s Temp"%(name,tp_type)] = tp_temp


    return out_dict

def get_pdu_data_cyberpower(ip, outlet_list):
    print_wrapper("Retrieving PDU data...")

    #Check if MIB files exists, if not then pull it and place in tmp
    mib_install_location = "/tmp/CyberPower_MIB_v2.11.MIB"
    if not os.path.exists(mib_install_location):
        print("WARNING :: MIB file not found.  Pulling to %s"%mib_install_location)
        cmd1 = "wget https://dl4jz3rbrsfum.cloudfront.net/software/CyberPower_MIB_v2.11.MIB.zip -O %s.zip"%mib_install_location
        cmd2 = "unzip %s.zip -d /tmp"%mib_install_location
        print(cmd1)
        subprocess.check_output(cmd1,shell=True, text=True)
        print(cmd2)
        subprocess.check_output(cmd2,shell=True, text=True)

    cmd3 = "snmpget -m %s -v1 -c public %s"%(mib_install_location,ip)
    cmd3 += " ePDULoadStatusVoltage.1"
    for outlet in outlet_list:
        cmd3 += " ePDUOutletStatusLoad.%i ePDUOutletStatusActivePower.%i"%(outlet,outlet)

    print_wrapper(cmd3)
    values = subprocess.check_output(cmd3, shell=True, text=True)
    #Example result:
    #CPS-MIB::ePDULoadStatusVoltage.1 = INTEGER: 1144
    #CPS-MIB::ePDUOutletStatusLoad.1 = Gauge32: 32
    #CPS-MIB::ePDUOutletStatusActivePower.1 = Gauge32: 370
    #CPS-MIB::ePDUOutletStatusLoad.2 = Gauge32: 0
    #CPS-MIB::ePDUOutletStatusActivePower.2 = Gauge32: 0
    split_list = values.split("\n")
    print_wrapper(split_list)
    current_idx = 0
    total_current = 0
    total_power = 0

    #Not sure why, but PDU appears to have two difference voltage values
    # Is it possible the 8 outlets are split into banks?
    #Using the first value for now
    bank1_vv = float(split_list[current_idx].split(":")[3])/10.0
    current_idx += 1

    out_dict = {}

    for outlet in outlet_list:
        ii = float(split_list[current_idx].split(":")[3])/10.0
        pp = float(split_list[current_idx+1].split(":")[3])
        current_idx += 2

        power = ii*bank1_vv

        out_dict["outlet_%i_current"%(outlet)] = ii
        out_dict["outlet_%i_power"%(outlet)] = pp
        out_dict["outlet_%i_voltage"%(outlet)] = bank1_vv  # Add per-outlet voltage

        total_current += ii
        total_power += pp #Use power measurement directly rather than multiplying current and voltage readings
        print_wrapper("get_pdu_data_cyberpower :: outlet%i: %f %f %f %f power_est_diff=%f"%(outlet,bank1_vv,ii,power,pp,pp-power))

    out_dict.update({'total_current':total_current,
                     'total_power':total_power,
                    })
    return out_dict

def get_pdu_data_tripplite(ip, outlet_list):
    print_wrapper("Retrieving PDU data...")

    # Assume we have all MIBs installed for now

    #Check if MIB files exists, if not then pull it and place in tmp
    mib_install_base_folder = "/tmp/mibs" #Note - mibs must be located in folder on their own (this is why we create special folder for tripplite mibs)
    mib_install_locations = [mib_install_base_folder + "/RFC-1628-UPS.MIB",
                             mib_install_base_folder + "/TRIPPLITE.MIB",
                             mib_install_base_folder + "/TRIPPLITE-PRODUCTS.MIB"]
    
    if not all([os.path.exists(aa) for aa in mib_install_locations]):
        print("WARNING :: MIB files not found.  Pulling to %s"%mib_install_base_folder)
        cmd1 = "mkdir -p %s"%mib_install_base_folder
        cmd2 = "wget https://assets.tripplite.com/firmware/tripplite-mib.zip -O %s/tripplite-mib.zip"%mib_install_base_folder
        cmd3 = "unzip %s/tripplite-mib.zip -d %s"%(mib_install_base_folder,mib_install_base_folder)
        print(cmd1)
        subprocess.check_output(cmd1,shell=True, text=True)
        print(cmd2)
        subprocess.check_output(cmd2,shell=True, text=True)
        print(cmd3)
        subprocess.check_output(cmd3,shell=True, text=True)

    cmd3 = "snmpget -M %s:/usr/share/snmp/mibs:/usr/share/snmp/mibs/iana:/usr/share/snmp/mibs/ietf -mALL -v2c -c Aerial %s"%(mib_install_base_folder,ip)
    for outlet in outlet_list:
        cmd3 += " tlpPduOutletCurrent.1.%i tlpPduOutletActivePower.1.%i"%(outlet,outlet)

    print_wrapper(cmd3)
    values = subprocess.check_output(cmd3, shell=True, text=True)
    #Example result:
    # TRIPPLITE-PRODUCTS::tlpPduOutletCurrent.1.1 = Gauge32: 179 0.01 Amps
    # TRIPPLITE-PRODUCTS::tlpPduOutletActivePower.1.1 = Gauge32: 211 Watts
    # TRIPPLITE-PRODUCTS::tlpPduOutletCurrent.1.2 = Gauge32: 143 0.01 Amps
    # TRIPPLITE-PRODUCTS::tlpPduOutletActivePower.1.2 = Gauge32: 157 Watts
    split_list = values.split("\n")
    print_wrapper(split_list)
    current_idx = 0
    total_current = 0
    total_power = 0

    out_dict = {}

    for outlet in outlet_list:
        ii = float(split_list[current_idx].split(":")[3].strip().split()[0])/100.0
        pp = float(split_list[current_idx+1].split(":")[3].strip().split()[0])
        current_idx += 2

        vv = pp/ii
        power = ii*vv

        out_dict["outlet_%i_current"%(outlet)] = ii
        out_dict["outlet_%i_power"%(outlet)] = pp
        out_dict["outlet_%i_voltage"%(outlet)] = vv  # Add per-outlet voltage

        total_current += ii
        total_power += pp #Use power measurement directly rather than multiplying current and voltage readings
        print_wrapper("get_pdu_data_cyberpower :: outlet%i: %f %f %f %f power_est_diff=%f"%(outlet,vv,ii,power,pp,pp-power))

    out_dict.update({'total_current':total_current,
                     'total_power':total_power,
                    })
    return out_dict

def get_cpu_data():

    #Determine what cpus are available in sysfs
    sysfs_cpus = [int(k.path.split('/sys/devices/system/cpu/cpu')[1]) for k in os.scandir('/sys/devices/system/cpu/')
                if k.is_dir() and
                   len(k.path.split('/sys/devices/system/cpu/cpu')) == 2 and
                   k.path.split('/sys/devices/system/cpu/cpu')[1].isnumeric()]
    sysfs_cpus.sort()

    out_dict = {}
    for cpu in sysfs_cpus:
        file1 = "/sys/devices/system/cpu/cpu%i/cpufreq/scaling_governor"%cpu
        file2 = "/sys/devices/system/cpu/cpu%i/cpufreq/scaling_min_freq"%cpu
        file3 = "/sys/devices/system/cpu/cpu%i/cpufreq/scaling_max_freq"%cpu
        file4 = "/sys/devices/system/cpu/cpu%i/cpufreq/scaling_cur_freq"%cpu
        file5 = "/sys/devices/system/cpu/cpu%i/cpufreq/cpuinfo_cur_freq"%cpu

        out_dict['sclgov%02i'%cpu] = read_file(file1).strip()
        out_dict['sclmin%02i'%cpu] = int(read_file(file2).strip())
        out_dict['sclmax%02i'%cpu] = int(read_file(file3).strip())
        out_dict['sclcur%02i'%cpu] = int(read_file(file4).strip())
        out_dict['cpucur%02i'%cpu] = int(read_file(file5).strip())

    return out_dict

def add_time_fields(data,ref_time,start_time,end_time,ntp_offset):
    data['collection_start_time'] = end_time + ntp_offset
    data['collection_end_time'] = end_time + ntp_offset
    data['collection_start_tir'] = start_time - ref_time
    data['collection_end_tir'] = end_time - ref_time
    data['collection_duration'] = data['collection_end_tir'] - data['collection_start_tir']  

def main(args):
    global VERBOSE

    if(args.verbose):
        VERBOSE = True

    # Validate credentials upfront for Raritan PDU type
    if not args.no_pdu and args.pdu_type.lower() == 'raritan':
        if not args.pdu_username or not args.pdu_password:
            print("ERROR: Raritan PDU type requires --pdu-username and --pdu-password arguments")
            return

    # Initialize PDU connection for Raritan type
    raritan_pdu = None
    if not args.no_pdu and args.pdu_type.lower() == 'raritan':
        pdu_outlet_list = [int(aa) for aa in args.pdu_outlets.split(",")]
        raritan_pdu = RaritanPDU(args.pdu_ip, pdu_outlet_list, args.pdu_username, args.pdu_password)

    #Determine system offset from ntp server
    ntp_offset = get_ntp_offset(100)

    #Set module averaging intevalS
    init_module_data()

    print("Writing to %s"%args.output_csv)
    fid = open(args.output_csv,'w')

    CPU_CLOCK_CSV_ENABLED = args.cpu_clock_csv is not None
    if CPU_CLOCK_CSV_ENABLED:
        cpu_fid = open(args.cpu_clock_csv,'w')
        cpu_first_write = True

    first_write = True
    still_running = True
    start_time = time.time()
    next_measurement_time = start_time + args.period  # Schedule first measurement
    print("Starting collection of power measurements (period=%f, duration=%f).  Use Ctrl-C at any time to stop collection."%(args.period,args.duration))
    count = 0
    try:
        while(still_running):

            #Collect all power measurements
            time1 = time.time()
            try:
                gpu_data = get_gpu_data()
                gpu_data = {"[GPU]"+key: value for key,value in gpu_data.items()}
                module_data = get_module_data()
                module_data = {"[MOD]"+key: value for key,value in module_data.items()}

                if not args.no_pdu:
                    pdu_outlet_list = [int(aa) for aa in args.pdu_outlets.split(",")]
                    if(args.pdu_type.lower() == 'cyberpower'):
                        pdu_data = get_pdu_data_cyberpower(args.pdu_ip,pdu_outlet_list)
                    elif(args.pdu_type.lower() == 'tripplite'):
                        pdu_data = get_pdu_data_tripplite(args.pdu_ip,pdu_outlet_list)
                    elif(args.pdu_type.lower() == 'raritan'):
                        pdu_data = raritan_pdu.get_pdu_data()
                    else:
                        print("Unknown PDU type %s, options: cyberpower, tripplite, or raritan"%args.pdu_type)
                        return
                    pdu_data = {"[PDU]"+key: value for key,value in pdu_data.items()}
                else:
                    pdu_data = {}

                collection_success = True
            except Exception as ee:
                print(f"ERROR: {ee}")
                print(f"ERROR TYPE: {type(ee)}")
                import traceback
                print("FULL TRACEBACK:")
                print(traceback.format_exc())
                print("WARNING :: Failed to retrieve all powers.  Skipping this data point.")
                collection_success = False

            time2 = time.time()

            if collection_success:
                #Collect all cpu clock frequencies
                if CPU_CLOCK_CSV_ENABLED:
                    cpu_data = get_cpu_data()
                    time3 = time.time()

                    #Add time fields
                    add_time_fields(cpu_data,start_time,time2,time3,ntp_offset)

                    #Write CPU data
                    if(cpu_first_write):
                        cpu_first_write = False
                        cpu_ww = csv.DictWriter(cpu_fid,fieldnames=cpu_data.keys())
                        cpu_ww.writeheader()
                    print_wrapper("Writing data to file %s."%args.cpu_clock_csv)
                    cpu_ww.writerows([cpu_data])

                #Combine data and add time fields
                final_data = {}
                final_data.update(gpu_data)
                final_data.update(module_data)
                final_data.update(pdu_data)
                add_time_fields(final_data,start_time,time1,time2,ntp_offset)

                # Note: for reading these times: datetime.datetime.strptime("2007-03-04T21:08:12Z", "%Y-%m-%dT%H:%M:%SZ")
                print_wrapper(final_data)

                #Write data
                if(first_write):
                    first_write = False
                    ww = csv.DictWriter(fid,fieldnames=final_data.keys())
                    ww.writeheader()

                print_wrapper("Writing data to file %s."%args.output_csv)
                ww.writerows([final_data])

            #Calculate dynamic sleep time to maintain accurate period
            current_time = time.time()
            time_until_next = next_measurement_time - current_time

            if time_until_next > 0:
                # Sleep only the remaining time needed to hit the target period
                time.sleep(time_until_next)
                print_wrapper(f"Slept for {time_until_next:.3f}s to maintain {args.period}s period")
            else:
                # Collection took longer than the period - warn but continue
                print_wrapper(f"WARNING: Data collection took {-time_until_next:.3f}s longer than period ({args.period}s)")

            # Schedule the next measurement
            next_measurement_time += args.period

            #Determine if we should stop collection
            current_time = time.time()
            still_running = (current_time - start_time) < args.duration

            count += 1

            if(count %10 == 0):
                actual_interval = (current_time - start_time) / count if count > 0 else 0
                print("[%.1f of %.1f] Collected %i data points. Avg interval: %.3fs (target: %.3fs)"%(
                    (current_time - start_time), args.duration, count, actual_interval, args.period))

    except KeyboardInterrupt:
        print("\nCtrl-C received. Stopping collection.")
    finally:
        fid.flush()
        fid.close()
        if CPU_CLOCK_CSV_ENABLED:
            cpu_fid.flush()
            cpu_fid.close()
        print("Finished collecting %d measurements."%count)

if __name__ == "__main__":
    default_duration = 350
    default_period = 1
    default_pdu_ip = "10.112.210.187"
    default_pdu_outlets = "1,2"
    default_pdu_type = "raritan"

    parser = argparse.ArgumentParser(
        description="Script able to collect both power and clock frequency information"
    )
    parser.add_argument(
        "output_csv", type=str, help="Output csv containing raw power measurements"
    )
    parser.add_argument(
        "-c", "--cpu_clock_csv", help="If specified enables CPU clock collection and write to this csv file"
    )
    parser.add_argument(
        "-d", "--duration", default=default_duration, type=float, help="Duration in seconds to collect power data"
    )
    parser.add_argument(
        "-p", "--period", default=default_period, type=float, help="Approximate period between measurements"
    )
    parser.add_argument(
        "-i", "--pdu_ip", default=default_pdu_ip, help="PDU IP or hostname (default=%s)"%default_pdu_ip
    )
    parser.add_argument(
        "-o", "--pdu_outlets", default=default_pdu_outlets, help="Comma-separated list of PDU Outlet (default=%s)"%default_pdu_outlets
    )
    parser.add_argument(
        "-t", "--pdu_type", default=default_pdu_type, help="PDU type.  Options: cyberpower, tripplite, or raritan (default=%s)"%default_pdu_type
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enables debug on retrieval of each data point"
    )
    parser.add_argument(
        "--no-pdu", action="store_true", help="If specified, do not collect PDU data"
    )
    parser.add_argument(
        "--pdu-username", help="Username for Raritan PDU authentication (required for raritan type)"
    )
    parser.add_argument(
        "--pdu-password", help="Password for Raritan PDU authentication (required for raritan type)"
    )
    args = parser.parse_args()

    main(args)
