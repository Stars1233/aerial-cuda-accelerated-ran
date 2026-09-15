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

#--------------------------------------------------------------------
#This script is to be run on DU side for cell throughput post check
#--------------------------------------------------------------------


import argparse
import re
import os
from datetime import datetime, timedelta


class cell_expect_perf:
    # get the expected tput and slot from mac log
    def __init__(self, log_file_mac, cell_id):
        self._log_file_mac = log_file_mac
        self._cell_id = cell_id

    def __del__(self):
        self._log_file_mac = None
        self._cell_id = None

    def get_cell_expected_tput(self):
        ul_expected_tput = 0
        dl_expected_tput = 0
        log_file_mac = self._log_file_mac
        cell_id = self._cell_id
        cell_info = 'Cell=' + cell_id

        with open(log_file_mac, 'r') as f:
            for line in f:
                if 'ExpectedData' in line and cell_info in line:
                    ul_expected_tput, dl_expected_tput = self.extract_expected_tput(line)

        return ul_expected_tput, dl_expected_tput

    def extract_expected_tput(self, src_str):
        tput_ul = 0
        tput_dl = 0

        match_tput_ul = re.findall(r'UL=\d+\.\d+', src_str)
        match_tput_dl = re.findall(r'DL=\d+\.\d+', src_str)

        if match_tput_ul:
            tput_ul = match_tput_ul[0].split('=')[-1]
        if match_tput_dl:
            tput_dl = match_tput_dl[0].split('=')[-1]


        return tput_ul, tput_dl

    def get_cell_expected_slot(self):
        ul_expected_slot = 0
        dl_expected_slot = 0
        log_file_mac = self._log_file_mac
        cell_id = self._cell_id
        cell_info = 'Cell=' + cell_id

        with open(log_file_mac, 'r') as f:
            for line in f:
                if 'ExpectedSlots' in line and cell_info in line:
                    ul_expected_slot, dl_expected_slot = self.extract_expected_slot(line)

        return ul_expected_slot, dl_expected_slot

    def extract_expected_slot(self, src_str):
        slot_ul = 0
        slot_dl = 0

        match_slot_ul = re.findall(r'PUSCH=\d+', src_str)
        match_slot_dl = re.findall(r'PDSCH=\d+', src_str)

        if match_slot_ul:
            slot_ul = match_slot_ul[0].split('=')[-1]
        if match_slot_dl:
            slot_dl = match_slot_dl[0].split('=')[-1]

        return slot_ul, slot_dl


def convert_datetime_format(currtime):
    # currtime： Sat Jan 18 03:28:54 AM UTC 2025
    # return： 2025-01-18 03:28:54
    # Strip timezone abbreviation before parsing — %Z is unreliable for non-UTC zones
    parts = currtime.split()
    currtime_no_tz = ' '.join(parts[:5] + parts[6:])
    dt_object = datetime.strptime(currtime_no_tz, '%a %b %d %I:%M:%S %p %Y')
    formatted_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')
    # print("convert time from: {} to {}".format(currtime, formatted_time))
    return formatted_time


def convert_datetime_to_timestamp(datetime_string):
    # input time: 2025-01-18 03:28:54
    datetime_format = '%Y-%m-%d %H:%M:%S'
    datetime_object = datetime.strptime(datetime_string, datetime_format)
    timestamp = float(datetime_object.timestamp())
    return timestamp


def convert_timestamp_to_datetime(timestamp):
    # return: 2025-01-18 03:28:54
    dt = datetime.fromtimestamp(timestamp)
    # print("time stamp: {}, date_time: {}".format(timestamp, dt))
    return dt


def get_previous_day(date_str):
    # date_str: 2025-01-18
    # return: 2025-01-17
    date_format = "%Y-%m-%d"
    date_new = datetime.strptime(date_str, date_format).date()
    previous_date = date_new - timedelta(days=1)
    previous_date_str = previous_date.strftime(date_format)
    return previous_date_str


def get_next_day(date_str):
    # date_str: 2025-01-18
    # return: 2025-01-19
    date_format = "%Y-%m-%d"
    date_new = datetime.strptime(date_str, date_format).date()
    date_next = date_new + timedelta(days=1)
    date_next_str = date_next.strftime(date_format)
    return date_next_str


def is_timeslot_match(log_datetime, check_timestamp_start, check_timestamp_end):
    # log_datetime 2025-01-18 22:49:41
    # print("log datetime: {}".format(log_datetime))
    log_timestamp = convert_datetime_to_timestamp(log_datetime)

    if log_timestamp >= int(check_timestamp_start) and log_timestamp <= int(check_timestamp_end):
        return True
    else:
        return False


def get_log_type(log_file):
    if 'phy' in os.path.basename(log_file):
        return 'phy'
    elif 'ru' in os.path.basename(log_file):
        return 'ru'
    elif '_cumac_cp' in os.path.basename(log_file):
        return 'cumac_cp'
    elif '_mac' in os.path.basename(log_file):
        return 'mac'

    else:
        return ''


class cell_real_perf:
    def __init__(self, log_file, cell_id, date_time, before_sec, after_sec):
        self._log_file = log_file
        self._cell_id = cell_id
        # self._date_time = date_time
        # self._before_sec = before_sec
        # self._after_sec = after_sec

        self._curr_datetime = convert_datetime_format(date_time)
        time_curr_stamp = convert_datetime_to_timestamp(self._curr_datetime)
        self._check_timestamp_start = time_curr_stamp - int(before_sec) - 1
        self._check_timestamp_end = time_curr_stamp + int(after_sec) + 1

    def __del__(self):
        self._log_file = None
        self._cell_id = None
        self._datetime = None
        self._before_sec = None
        self._after_sec = None

    def fetch_logs_by_cell_from_log_file(self, cell_id):
        log_file = self._log_file
        curr_datetime = self._curr_datetime
        check_timestamp_start = self._check_timestamp_start
        check_timestamp_end = self._check_timestamp_end
        log_type = get_log_type(log_file)
        log_datetime_list = []
        cell_tput_log = []
        print("log_type: {}".format(log_type))
        with open(log_file, 'r') as f:
            for line in f:
                # 00:00:01 Cell 0: MAC:[22:49:40.640492 DL=558.90/1600 UL=79.91/400 Prmb=700 HARQ=9600 CSI1=2400 CSI2=2400]
                if self.filter_valid_logs(log_type, cell_id, line) is not True:
                    continue

                log_time = self.extracted_log_time(line)
                # print("mac log time: {}".format(mac_log_time))
                log_time_hh = log_time.split(':')[0]
                if len(log_datetime_list) == 0:
                    curr_time_hh = curr_datetime.split()[1].split(':')[0]
                    if int(log_time_hh) > int(curr_time_hh):
                        log_date = get_previous_day(curr_datetime.split()[0])
                    else:
                        log_date = curr_datetime.split()[0]
                else:
                    previous_log_date = log_datetime_list[-1].split()[0]
                    previous_log_time = log_datetime_list[-1].split()[-1]
                    if int(log_time_hh) < int(previous_log_time.split(':')[0]):
                        log_date = get_next_day(previous_log_date)
                    else:
                        log_date = previous_log_date

                log_datetime = log_date + " " + log_time
                log_datetime_list.append(log_datetime)
                if is_timeslot_match(log_datetime, check_timestamp_start, check_timestamp_end) is True:
                    cell_tput_log.append(line)

        return cell_tput_log

    def filter_valid_logs(self, log_type, cell_id, line):
        # RU ==>
        # 07:39:39.488811 CON 73397 0 [RU] Cell  0 DL    0.00 Mbps    0 Slots | UL
        # PHY ==>
        # 07:41:35.600005 CON timer_thread 0 [SCF.PHY] Cell  2 | DL  558.90 Mbps 1600 Slots | UL   94.65 Mbps  400 Slots CRC   0 (     0) | Tick 6000
        # MAC =>
        # 07:41:45.600485 CON 87361 0 [MAC.FAPI] Cell  3 | DL  558.90 Mbps 1600 Slots | UL   94.65 Mbps  400 Slots | Prmb  700 | HARQ 12000 | SR    0 | CSI1 2400 | CSI2 2400 | SRS    0 | ERR    0 | INV    0 | Slots 26000
        # cuMAC =>
        # 01:24:12.960032 CON 179936 0 [CUMCP.HANDLER] Cell  0 | CUMAC 1999 | ERR    0 | Slots 2000
        cell_info = "Cell {:>2}".format(cell_id)

        if log_type == 'ru':
            if cell_info in line and '[RU]' in line and 'DL' in line and 'UL' in line and 'Mbps' in line:
                return True
        elif log_type == 'phy':
            if cell_info in line and '[SCF.PHY]' in line and 'DL' in line and 'UL' in line and 'Mbps' in line:
                return True
        elif log_type == 'mac':
            if cell_info in line and '[MAC.FAPI]' in line and 'DL' in line and 'UL' in line and 'Mbps' in line:
                return True
        elif log_type == 'cumac_cp':
            if cell_info in line and '[CUMCP.HANDLER]' in line and 'CUMAC' in line and 'ERR' in line and 'Slots' in line:
                return True
        else:
            return False

    def extracted_log_time(self, line):
        # print(line.split()[0])
        return line.split()[0].split('.')[0]

    def get_cell_real_tput(self):
        cell_id = self._cell_id
        cell_tput_logs = self.fetch_logs_by_cell_from_log_file(cell_id)
        cell_info = "Cell {:>2}".format(cell_id)
        ul_real_tput_list = []
        dl_real_tput_list = []
        for cell_tput in cell_tput_logs:
            if cell_info not in cell_tput:
                continue
            tput_ul, tput_dl = (self.extract_real_tput(cell_tput))
            if tput_ul is not None:
                ul_real_tput_list.append(tput_ul)
            if tput_dl is not None:
                dl_real_tput_list.append(tput_dl)
        return ul_real_tput_list, dl_real_tput_list

    def get_cell_real_tput_cumac(self):
        cell_id = self._cell_id
        cell_tput_logs = self.fetch_logs_by_cell_from_log_file(cell_id)
        cell_info = "Cell {:>2}".format(cell_id)
        tput_cumac_list = []
        err_cumac_list = []
        for cell_tput in cell_tput_logs:
            if cell_info not in cell_tput:
                continue
            tput_cumac = self.extract_real_tput_cumac(cell_tput)
            err_cumac = self.extract_real_err_cumac(cell_tput)
            if tput_cumac is not None:
                tput_cumac_list.append(tput_cumac)
            if err_cumac is not None:
                err_cumac_list.append(err_cumac)
        return tput_cumac_list, err_cumac_list

    def extract_real_tput(self, line):
        tput_dl = self.extract_real_tput_dl(line)
        tput_ul = self.extract_real_tput_ul(line)
        return tput_ul, tput_dl

    def extract_real_tput_dl(self, line):
        tput_dl = None
        match_dl = re.findall(r'DL.*\d+\.\d+.* \| UL', line)
        # 'DL  558.90 Mbps 1600 Slots | UL'
        if match_dl:
            tput = match_dl[0].split()[1]
            tput_dl = tput  # Always extract, including 0.0

        return tput_dl

    def extract_real_tput_ul(self, line):
        # 'UL    0.00 Mbps'
        tput_ul = None
        match_ul = re.findall(r'UL.*\d+\.\d+.* Mbps', line)
        if match_ul:
            tput = match_ul[0].split()[1]
            tput_ul = tput  # Always extract, including 0.0

        return tput_ul

    def extract_real_tput_cumac(self, line):
        # 'CUMAC 1999'
        tput_cumac = None
        match_cumac = re.findall(r'CUMAC \d+', line)
        if match_cumac:
            tput_cumac = int(match_cumac[0].split()[1])
        return tput_cumac

    def extract_real_err_cumac(self, line):
        # 'ERR    0'
        err_cumac = None
        match_err = re.findall(r'ERR.*\d+', line)
        if match_err:
            err_cumac = int(match_err[0].split()[1])
        return err_cumac

def parse_args():
    parser = argparse.ArgumentParser(description="description")
    parser.add_argument('--cells', '-c', help='cell id list, e.g 1,2,2,3,4,6,7,8', required=True, type=str)
    parser.add_argument('--log', '-g', help='log file, e.g screenlog_mac.log, screenlog_phy.log, screenlog_ru.log, screenlog_cumac_cp.log', required=True, type=str)
    # parser.add_argument('--mac', '-m', help='mac log, e.g screenlog_mac', required=True, type=str)
    parser.add_argument('--datetime', '-t', help="check point time, used command date , e.g Sun Jan 19 12:44:33 PM UTC 2025", required=True, type=str)
    parser.add_argument('--beforesec', '-b', help='The time before check point time, in second', required=True, type=str)
    parser.add_argument('--aftersec', '-a', help='The time after check point time, in second', required=True, type=str)
    parser.add_argument('--tolerance', '-tol', help='Optional: Tolerance value for throughput comparison (in percentage, e.g. 5 means ±5%)', type=float, default=None)
    parser.add_argument('--tput_cumac', '-tc', help='cumac expected value, e.g 1999', required=False, type=int)
    parser.add_argument('--err_cumac', '-ec', help='cumac error value, e.g 0', required=False, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    print("Cell ids: {}".format(args.cells))
    print("Log file: {}".format(args.log))
    print("Date time: {}".format(args.datetime))
    if args.tolerance is not None:
        print("Tolerance: ±{}%".format(args.tolerance))
    mac_log_file = os.path.join(os.path.dirname(args.log), 'screenlog_mac.log')
    if os.path.exists(args.log) is False or os.path.exists(mac_log_file) is False:
        print("Error, log: {} {} not exits".format(args.log, mac_log_file))

    curr_datetime = convert_datetime_format(args.datetime)
    time_curr_stamp = convert_datetime_to_timestamp(curr_datetime)
    check_timestamp_start = time_curr_stamp - int(args.beforesec) - 1
    check_timestamp_end = time_curr_stamp + int(args.aftersec) + 1
    check_datetime_start = convert_timestamp_to_datetime(check_timestamp_start)
    check_datetime_end = convert_timestamp_to_datetime(check_timestamp_end)

    for cell in args.cells.split(','):
        cell_real_perf_obj = cell_real_perf(args.log, cell, args.datetime, args.beforesec, args.aftersec)
        if args.tput_cumac is not None:
            tput_real_cumac_list, err_real_cumac_list = cell_real_perf_obj.get_cell_real_tput_cumac()
            print("cell: {}, cuMAC tput: {}".format(cell, tput_real_cumac_list))
            print("cell: {}, cuMAC error: {}".format(cell, err_real_cumac_list))

            # Calculate tolerance ranges if tolerance is specified
            if args.tolerance is not None:
                if args.tput_cumac == 0:
                    tput_cumac_min = 0
                    tput_cumac_max = args.tolerance
                else:
                    tput_cumac_min = args.tput_cumac * (1 - args.tolerance / 100)
                    tput_cumac_max = args.tput_cumac * (1 + args.tolerance / 100)
                if args.err_cumac is not None:
                    if args.err_cumac == 0:
                        err_cumac_min = 0
                        err_cumac_max = args.tolerance
                    else:
                        err_cumac_min = args.err_cumac * (1 - args.tolerance / 100)
                        err_cumac_max = args.err_cumac * (1 + args.tolerance / 100)
            else:
                tput_cumac_min = args.tput_cumac
                tput_cumac_max = args.tput_cumac
                if args.err_cumac is not None:
                    err_cumac_min = args.err_cumac
                    err_cumac_max = args.err_cumac

            tput_cumac_check_required = args.tput_cumac is not None
            err_cumac_check_required = args.err_cumac is not None

            if tput_cumac_check_required and not tput_real_cumac_list:
                print("Cell: {} TEST FAILED, no valid cuMAC tput data found in the time window from {} to {} (expected: {})".format(cell, check_datetime_start, check_datetime_end, args.tput_cumac))
                continue
            if err_cumac_check_required and not err_real_cumac_list:
                print("Cell: {} TEST FAILED, no valid cuMAC error data found in the time window from {} to {} (expected: {})".format(cell, check_datetime_start, check_datetime_end, args.err_cumac))
                continue

            # Calculate average values
            tput_cumac_avg = sum(tput_real_cumac_list) / len(tput_real_cumac_list) if tput_real_cumac_list else 0
            err_cumac_avg = sum(err_real_cumac_list) / len(err_real_cumac_list) if err_real_cumac_list else 0

            # Check tput_cumac
            tput_cumac_pass = True
            if tput_cumac_check_required:
                if args.tolerance is not None:
                    tput_cumac_pass = tput_cumac_min <= tput_cumac_avg <= tput_cumac_max
                else:
                    tput_cumac_pass = abs(tput_cumac_avg - args.tput_cumac) < 1e-2

                if not tput_cumac_pass:
                    if args.tolerance is not None:
                        print("Cell: {} TEST FAILED, cuMAC tput {:.2f} not within {}% of expected {} from {} to {}".format(cell, tput_cumac_avg, args.tolerance, args.tput_cumac, check_datetime_start, check_datetime_end))
                    else:
                        print("Cell: {} TEST FAILED, cuMAC tput {:.2f} not equal to expected {} from {} to {}".format(cell, tput_cumac_avg, args.tput_cumac, check_datetime_start, check_datetime_end))
                    print("cell: {}, cuMAC tput values: {}".format(cell, tput_real_cumac_list))
                    continue

            # Check err_cumac
            err_cumac_pass = True
            if err_cumac_check_required:
                if args.tolerance is not None:
                    err_cumac_pass = err_cumac_min <= err_cumac_avg <= err_cumac_max
                else:
                    err_cumac_pass = abs(err_cumac_avg - args.err_cumac) < 1e-2

                if not err_cumac_pass:
                    if args.tolerance is not None:
                        print("Cell: {} TEST FAILED, cuMAC error {:.2f} not within {}% of expected {} from {} to {}".format(cell, err_cumac_avg, args.tolerance, args.err_cumac, check_datetime_start, check_datetime_end))
                    else:
                        print("Cell: {} TEST FAILED, cuMAC error {:.2f} not equal to expected {} from {} to {}".format(cell, err_cumac_avg, args.err_cumac, check_datetime_start, check_datetime_end))
                    print("cell: {}, cuMAC error values: {}".format(cell, err_real_cumac_list))
                    continue

            if tput_cumac_pass and err_cumac_pass:
                print("Cell: {} TEST PASS, cuMAC tput: {:.2f} (samples: {}), cuMAC error: {:.2f} (samples: {})".format(cell, tput_cumac_avg, len(tput_real_cumac_list), err_cumac_avg, len(err_real_cumac_list)))
            print("")
        else:
            cell_expect_perf_obj = cell_expect_perf(mac_log_file, cell)
            ul_expected_tput, dl_expected_tput = cell_expect_perf_obj.get_cell_expected_tput()
            print("cell: {}, Expected tput ul: {} Mbps, dl: {} Mbps".format(cell, ul_expected_tput, dl_expected_tput))
            # ul_expected_slot, dl_expected_slot = cell_expect_perf_obj.get_cell_expected_slot()

            ul_expected_tput = float(ul_expected_tput)
            dl_expected_tput = float(dl_expected_tput)

            # Round expected values to 2 decimal places to match log precision
            ul_expected_tput = round(ul_expected_tput, 2)
            dl_expected_tput = round(dl_expected_tput, 2)
            ul_real_tput_list, dl_real_tput_list = cell_real_perf_obj.get_cell_real_tput()
            # Check if we have valid data: if expected is 0, we can skip that direction
            ul_check_required = ul_expected_tput != 0.0
            dl_check_required = dl_expected_tput != 0.0

            if ul_check_required and not ul_real_tput_list:
                print("Cell: {} TEST FAILED, no valid UL throughput data found in the time window from {} to {} (expected: {:.2f} Mbps)".format(cell, check_datetime_start, check_datetime_end, ul_expected_tput))
                continue

            if dl_check_required and not dl_real_tput_list:
                print("Cell: {} TEST FAILED, no valid DL throughput data found in the time window from {} to {} (expected: {:.2f} Mbps)".format(cell, check_datetime_start, check_datetime_end, dl_expected_tput))
                continue

            # Calculate tolerance ranges if tolerance is specified
            if args.tolerance is not None:
                ul_min = ul_expected_tput * (1 - args.tolerance / 100)
                ul_max = ul_expected_tput * (1 + args.tolerance / 100)
                dl_min = dl_expected_tput * (1 - args.tolerance / 100)
                dl_max = dl_expected_tput * (1 + args.tolerance / 100)

            # Check uplink throughput (skip if expected is 0)
            if ul_check_required:
                if ul_real_tput_list:
                    ul_real = sum([float(x) for x in ul_real_tput_list]) / len(ul_real_tput_list)
                else:
                    ul_real = 0.0

                if args.tolerance is not None:
                    ul_pass = ul_min <= ul_real <= ul_max
                else:
                    ul_pass = float_equal(ul_real, ul_expected_tput)

                if ul_pass:
                    print("Cell: {} TEST PASS, Uplink   tput: {:.2f} Mbps (actual: {:.2f} Mbps, samples: {})".format(
                        cell, ul_expected_tput, ul_real, len(ul_real_tput_list)))
                else:
                    if args.tolerance is not None:
                        print("Cell: {} TEST FAILED, real uplink throughput {:.2f} Mbps not within {}% of expected {:.2f} Mbps from {} to {}".format(
                            cell, ul_real, args.tolerance, ul_expected_tput, check_datetime_start, check_datetime_end))
                    else:
                        print("Cell: {} TEST FAILED, real uplink throughput {:.2f} Mbps not equal to expected {:.2f} Mbps from {} to {}".format(
                            cell, ul_real, ul_expected_tput, check_datetime_start, check_datetime_end))
                    print("cell: {}, uplink real tput values: {}".format(cell, ul_real_tput_list))
            else:
                print("Cell: {} SKIP, Uplink throughput check skipped (expected: 0.00 Mbps)".format(cell))

            # Check downlink throughput (skip if expected is 0)
            if dl_check_required:
                if dl_real_tput_list:
                    dl_real = sum([float(x) for x in dl_real_tput_list]) / len(dl_real_tput_list)
                else:
                    dl_real = 0.0

                if args.tolerance is not None:
                    dl_pass = dl_min <= dl_real <= dl_max
                else:
                    dl_pass = float_equal(dl_real, dl_expected_tput)

                if dl_pass:
                    print("Cell: {} TEST PASS, Downlink tput: {:.2f} Mbps (actual: {:.2f} Mbps, samples: {})".format(
                        cell, dl_expected_tput, dl_real, len(dl_real_tput_list)))
                else:
                    if args.tolerance is not None:
                        print("Cell: {} TEST FAILED, real downlink throughput {:.2f} Mbps not within {}% of expected {:.2f} Mbps from {} to {}".format(
                            cell, dl_real, args.tolerance, dl_expected_tput, check_datetime_start, check_datetime_end))
                    else:
                        print("Cell: {} TEST FAILED, real downlink throughput {:.2f} Mbps not equal to expected {:.2f} Mbps from {} to {}".format(
                            cell, dl_real, dl_expected_tput, check_datetime_start, check_datetime_end))
                    print("cell: {}, downlink real tput values: {}".format(cell, dl_real_tput_list))
            else:
                print("Cell: {} SKIP, Downlink throughput check skipped (expected: 0.00 Mbps)".format(cell))

            print("")


def float_equal(a, b, eps=1e-2):
    return abs(a - b) < eps


if __name__ == '__main__':
    main()
