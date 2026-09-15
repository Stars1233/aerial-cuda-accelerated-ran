#!/usr/bin/env python3

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

# Examples of running:
# python ../cuPHY/util/ldpc/ldpc_perf_collect.py --use_fp16
# python ../cuPHY/util/ldpc/ldpc_perf_collect.py -m latency -f
# python ../cuPHY/util/ldpc/ldpc_perf_collect.py --mode ber -i ldpc_BG1_K8448_SNR%g_800_p_m.h5 --num_parity 5 -n 32 --use_fp16
# python ../cuPHY/util/ldpc/ldpc_perf_collect.py --mode ber -i ldpc_BG2_K3840_SNR%g_800_p_m.h5 -g 2 --num_parity 7 -n 32 --use_fp16 --min_snr 1 --max_snr 3.25 --normalization 0.8125 -o ldpc_ber_7.txt
# python ../cuPHY/util/ldpc/ldpc_perf_collect.py --mode ber -i ldpc_BG1_K8448_SNR%g_800_p_m.h5 --num_parity 46 -n 32 --use_fp16 --min_snr -2.5 --max_snr -1 --normalization 0.6875 -o ldpc_ber_46.txt

# Comparison with Xilinx:
# https://www.xilinx.com/support/documentation/ip_documentation/pl/sd-fec-ber-plots.html
# ../cuPHY/util/ldpc/ldpc_perf_collect.py --mode ber -Z 384 --min_snr -2.5 --max_snr -0.9 --snr_step 0.1 --num_parity 46 -n 32 --use_fp16 -w 800 -P
# ../cuPHY/util/ldpc/ldpc_perf_collect.py --mode ber -Z 384 --min_snr 4    --max_snr 6.3  --snr_step 0.1 --num_parity 5  -n 32 --use_fp16 -w 800 -P

# Normalization mode
# Calculates a normalization value for a given combination of lifting
# size and number of parity nodes, and adds it to a database JSON file.
# ../cuPHY/util/ldpc/ldpc_perf_collect.py -m norm -f -P -g 1 -n 32 -r 1 -w 400 -Z 384 --num_parity 20

# Range mode
# Calculates a nominal SNR range for a given combination of lifting
# size and number of parity nodes, and adds it to a database JSON file.
# ../cuPHY/util/ldpc/ldpc_perf_collect.py -m range -f -P -g 1 -n 32 -r 1 -w 400 -Z 384 --num_parity 20

# Test mode
# ../cuPHY/util/ldpc/ldpc_perf_collect.py --mode test -i ../cuPHY/util/ldpc/test/ldpc_decode_BG1_Z384_BLER0.1.txt -n 10 -f -w 800 -P
# ../cuPHY/util/ldpc/ldpc_perf_collect.py --mode test -i ../cuPHY/util/ldpc/test/ldpc_decode_BG2_Z384_BLER0.1.txt -n 10 -f -w 800 -P

# SNR mode (find SNR values that are "close" to a give BER or BLER value)
# ../cuPHY/util/ldpc/ldpc_perf_collect.py --mode SNR -g 2 -Z 384 -f -w 800 -P -n 10 --min_num_parity 4 --max_num_parity 42 --BLER 0.09 -o temp.txt
import os
import subprocess
import re
import argparse
import math
import sys
import json

#***********************************************************************
# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--use_fp16",        action="store_true",                               help="Use fp16 input LLR values")
parser.add_argument("-P", "--puncture",        action="store_true",                               help="Puncture first 2*Z generated values")
parser.add_argument("-B", "--block_size",      type=int,                                          help="Input data block size")
parser.add_argument("-N", "--mod_bits",        type=int,                                          help="Total number of modulated bits")
parser.add_argument("-R", "--code_rate",       type=float,                                        help="Code rate (input block size / modulated bits)")
parser.add_argument("-M", "--modulation",                  choices=["BPSK", "QPSK", "QAM16", "QAM64", "QAM256"], default=None, help="Modulation for generated input data")
parser.add_argument("-g", "--bg",              type=int,   default=1,                             help="Base graph (1 or 2)", choices=[1, 2])
parser.add_argument("-i", "--input_file",                  default="",                            help="Optional input file (for latency mode) or input file SNR format string (for ber mode)")
parser.add_argument("-m", "--mode",                        default="latency",                     help="perf collection mode ('latency', 'ber', or 'norm')")
parser.add_argument("-n", "--num_iter",        type=int,   default=10,                            help="Number of LDPC iterations")
parser.add_argument("-o", "--output_file",                                                        help="Output file name")
parser.add_argument("-r", "--num_runs",        type=int,   default=100,                           help="Number of runs to time (latency mode only)")
parser.add_argument("-w", "--num_words",       type=int,   default=1,                             help="Number of input codewords")
parser.add_argument("-Z", "--lifting_size",    type=int,   default=384,                           help="Base graph lifting size")
parser.add_argument("-a", "--algo",            type=int,   default=0,                             help="Algorithm to use")
parser.add_argument("-b", "--transport_block", action="store_true",                               help="Use the transport block decoder interface")
parser.add_argument("-t", "--throughput",      action="store_true",                               help="Choose an implementation that favors throughput over latency when available")
parser.add_argument("-x", "--early_termination", action="store_true",                            help="Enable LDPC CRC early termination")
parser.add_argument("-y", "--write_iter_count", action="store_true",                            help="Write and report LDPC iteration counts when supported")
parser.add_argument("--et_latency_debug", action="store_true",                                   help="Exercise LDPC ET checks while forcing max-iteration latency")
parser.add_argument("--force_et_kernel", action="store_true",                                    help="Launch the accessory-capable LDPC kernel without enabling accessories")
parser.add_argument("--crc", choices=["crc16", "crc24a", "crc24b", "none"], default=None,        help="CRC type to pass to cuphy_ex_ldpc")
parser.add_argument("-d", "--tb_spread",       action="store_true",                               help="Spread codewords over multiple transport blocks when using the transport block interface")
parser.add_argument("-c", "--compare_file",                default="",                            help="Comparison file (from previous run) ('latency' mode only)")
parser.add_argument("--exe_dir",                           default="./cuPHY/examples/error_correction", help="Path to cuphy_ex_ldpc executable")
parser.add_argument("--exe",                               default="cuphy_ex_ldpc",               help="Name of executable")
parser.add_argument("--data_dir",                          default="../../cuPHY_data",            help="Path to input HDF5 files")
parser.add_argument("--num_parity",            type=int,   default=4,                             help="Number of parity check nodes (ber mode only)")
parser.add_argument("--min_num_parity",        type=int,   default=4,                             help="Minimum number of parity check nodes (latency mode only)")
parser.add_argument("--max_num_parity",        type=int,   default=-1,                            help="Maximum number of parity check nodes (latency mode only, default uses max for BG)")
parser.add_argument("--min_snr",               type=float,                                        help="Minimum SNR (ber mode only)")
parser.add_argument("--max_snr",               type=float,                                        help="Maximum SNR (ber mode only)")
parser.add_argument("--snr_step",              type=float, default=0.1,                           help="SNR step size (ber mode only)")
parser.add_argument("-S", "--snr",             type=float, default=10.0,                          help="SNR (latency mode only)")
parser.add_argument("--normalization",         type=float,                                        help="Min-sum normalization factor")
parser.add_argument("--min_block_errors",      type=int,                                          help="Loop until the at least the given number of block errors occurs")
parser.add_argument("--BER",                   type=float,                                        help="Target BER in SNR search mode. (BER or BLER must be provided.")
parser.add_argument("--BLER",                  type=float,                                        help="Target BLER in SNR search mode. (BER or BLER must be provided.")
args = parser.parse_args()

db_name = 'ldpc_perf_database.json'

########################################################################
# get_comparison_results()
# Read table results from a file containing results from previous run,
# as stored with the output_file option.
def get_comparison_results(cfile):
    res = {}
    f = open(cfile, 'r')
    while True:
        line = f.readline().strip()
        if not line:
            break
        if line.startswith('#'):
            continue
        fields = line.split()
        # File format assumed:
        # num_parity  latency BER BLER throughput
        mb = int(fields[0])
        res[mb] = [float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4])]
    return res

########################################################################
# code_block_params()
# Determine Kb, K, and Z for a given base graph (1 or 2) and payload size
# Inputs:
#   BG:       Base graph (1 or 2)
#   block_sz: Number of info bits in code block
# Returns (Kb, K, Z, F):
#   Kb: Number of information nodes
#   K:  Number of info bits (before puncturing)
#   Z:  Lifting size
#   F:  Number of filler bits
def code_block_params(BG, block_sz):
    # ------------------------------------------------------------------
    # 38212, Sec. 5.2.2
    B       = block_sz
    Kcb_max = [0, 8448, 3840]
    if B > Kcb_max[BG]:
        raise RuntimeError('Invalid block size: "%d"' % block_sz)
    # ------------------------------------------------------------------
    # Determine Kb
    if 1 == BG:
        Kb = 22
    else:
        if B > 640:
            Kb = 10
        elif B > 560:
            Kb = 9
        elif B > 192:
            Kb = 8
        else:
            Kb = 6
    # % Assuming a single code block (no segmentation) so Bprime = B
    C  = 1;
    L  = 0;
    Bp = B;
    # ------------------------------------------------------------------
    # Determine Zc
    Zc_all = [   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,
                 12,  13,  14,  15,  16,  18,  20,  22,  24,  26,
                 28,  30,  32,  36,  40,  44,  48,  52,  56,  60,
                 64,  72,  80,  88,  96, 104, 112, 120, 128, 144,
                 160, 176, 192, 208, 224, 240, 256, 288, 320, 352,
                 384]
    Kp = Bp / C;
    # % Find the first Zc for which Kb * Zc >= Kprime
    Zcheck = [Zc for Zc in Zc_all if (Kb * Zc) >= Kp]
    Z = Zcheck[0]
    # ------------------------------------------------------------------
    # Determine K
    Kscale = [0, 22, 10]
    K = Kscale[BG] * Z
    # ------------------------------------------------------------------
    # Determine the number of filler bits
    F = K - Kp
    return (Kb, K, Z, F)

#***********************************************************************
# get_num_parity()
def get_num_parity(args):
    B = args.block_size
    BG = args.bg
    Kb, K, Z, F = code_block_params(BG, B)
    if args.mod_bits:
        N = args.mod_bits
    else:
        N = int(round(B) / args.code_rate)
    num_parity = int(math.ceil((N - B + (2 * Z)) / float(Z)))
    return num_parity


#***********************************************************************
# gen_cmd()
# Returns a string representation of the command to execute, given a
# dictionary of parameters
def gen_cmd(param_dict):
    cmd = [os.path.join(param_dict['exe_dir'], param_dict['exe'])]
    if param_dict.get('input_file'):
        cmd += ['-i', param_dict['input_file']]
    if param_dict.get('Z'):
        cmd += ['-Z', str(param_dict['Z'])]
    if param_dict.get('num_parity'):
        cmd += ['-p', str(param_dict['num_parity'])]
    if param_dict.get('block_size'):
        cmd += ['-B', str(param_dict['block_size'])]
    if param_dict.get('mod_bits'):
        cmd += ['-N', str(param_dict['mod_bits'])]
    if param_dict.get('code_rate'):
        cmd += ['-R', '%f' % param_dict['code_rate']]

    cmd += ['-n', str(param_dict['num_iter']), '-r', str(param_dict['num_runs'])]

    if param_dict.get('normalization'):
        cmd += ['-m', '%f' % param_dict['normalization']]
    cmd += ['-g', str(param_dict['BG'])]
    if param_dict.get('use_fp16'):
        cmd += ['-f', 'fp16']
    if 'num_words' in param_dict and param_dict.get('num_words') is not None:
        cmd += ['-w', str(param_dict['num_words'])]
    if not param_dict.get('input_file'):
        modulation = param_dict.get('modulation', args.modulation)
        if modulation:
            cmd += ['-M', modulation]
        cmd += ['-S', '%g' % param_dict['snr']]
    cmd += ['-a', str(param_dict['algo'])]
    if (not param_dict.get('input_file')) and param_dict.get('puncture'):
        cmd.append('-P')
    if param_dict.get('use_tb') or param_dict.get('early_termination', args.early_termination):
        cmd.append('-b')
    if param_dict.get('min_block_errors'):
        cmd += ['-e', str(param_dict['min_block_errors'])]
    if param_dict.get('choose_throughput'):
        cmd.append('-t')
    if param_dict.get('tb_spread', False):
        cmd.append('-d')
    if param_dict.get('early_termination', args.early_termination):
        cmd.append('-x')
    if param_dict.get('write_iter_count', args.write_iter_count):
        cmd.append('-y')
    if param_dict.get('et_latency_debug', args.et_latency_debug):
        cmd.append('--et_latency_debug')
    if param_dict.get('force_et_kernel', args.force_et_kernel):
        cmd.append('--force_et_kernel')
    crc = param_dict.get('crc', args.crc)
    if crc:
        cmd += ['--crc', crc]
    return ' '.join(cmd)

max_num_parity = [-1, 46, 42]

# re_str = r'Average \(([-+]?\d*\.\d+|\d+) runs\) elapsed time in usec = , throughput =  Gbps'
re_latency_str = r'Average \((\d+) runs\) elapsed time in usec = (.+), throughput = (.+) Gbps'

# bit error count = 0, bit error rate (BER) = (0 / 8448) = 0.00000e+00, block error rate (BLER) = (0 / 1) = 0.00000e+00
# re_ber_str = r'bit error count = (\d+), bit error rate (BER) = \(\d+ / \d+\) = .+, block error rate (BLER) = \(\d+ / \d+\) = .+'
re_ber_str = r'bit error count = \d+, bit error rate \(BER\) = \((\d+) / (\d+)\) = (.+), block error rate \(BLER\) = \((\d+) / (\d+)\) = (.+)'
re_crc_bler_str = r'CRC-based block error rate \(BLER\) = \((\d+) / (\d+)\) = (.+)'

#***********************************************************************
# run_config()
def run_config(params, verbose):
    #-------------------------------------------------------------------
    # Generate the command line string
    cmd = gen_cmd(params)
    if verbose:
        print(cmd)
    #-------------------------------------------------------------------
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    res = {'cmd': cmd, 'iteration_lines': []}
    output_lines = []
    while True:
        raw = proc.stdout.readline()
        if not raw:
            break
        line = raw.decode('ascii', errors='replace')
        output_lines.append(line.rstrip())
        m = re.search(re_latency_str, line)
        if m:
            if params['num_parity']:
                res['num_parity'] = params['num_parity']
            res['num_runs']   = int(m.group(1))
            res['latency']    = float(m.group(2))
            res['throughput'] = float(m.group(3))
            continue

        m = re.search(re_ber_str, line)
        if m:
            res['bit_error_count']   = int(m.group(1))
            res['total_bits']        = int(m.group(2))
            res['bit_error_rate']    = float(m.group(3))
            res['block_error_count'] = int(m.group(4))
            res['total_blocks']      = int(m.group(5))
            res['block_error_rate']  = float(m.group(6))
            continue

        m = re.search(re_crc_bler_str, line)
        if m:
            res['crc_block_error_count'] = int(m.group(1))
            res['crc_total_blocks']      = int(m.group(2))
            res['crc_block_error_rate']  = float(m.group(3))
            continue

        # The histogram header says "Iteration count histogram:", but its bins are
        # "  <n>: [ <count> ] ****" and carry no 'iter' substring; match both.
        if re.search(r'iter', line, re.IGNORECASE) or re.match(r'\s*\d+:\s*\[', line):
            res['iteration_lines'].append(line.strip())

    res['return_code'] = proc.wait()
    if res['return_code'] != 0:
        tail = '\n'.join(output_lines[-20:])
        raise RuntimeError('Command failed with return code %d: "%s"\n%s' % (res['return_code'], cmd, tail))
    if not any(k in res for k in ('latency', 'bit_error_rate', 'block_error_rate')):
        warning_lines = [line for line in output_lines if line.startswith('WARNING:')]
        if warning_lines:
            res.update({'skipped': True,
                        'skip_reason': '; '.join(warning_lines),
                        'num_runs': 0,
                        'latency': float('nan'),
                        'throughput': float('nan'),
                        'bit_error_count': 0,
                        'total_bits': 0,
                        'bit_error_rate': float('nan'),
                        'block_error_count': 0,
                        'total_blocks': 0,
                        'block_error_rate': float('nan')})
            if not res['iteration_lines']:
                del res['iteration_lines']
            if verbose:
                print(res['skip_reason'])
            return res
        raise RuntimeError('No output results string found for command "%s"' % cmd)
    if not res['iteration_lines']:
        del res['iteration_lines']
    if verbose:
        crc_str = ''
        if 'crc_block_error_rate' in res:
            crc_str = ', crc_bler=%e' % res['crc_block_error_rate']
        if 'num_parity' in res:
            print('num_parity=%i, latency=%.1f, throughput=%.2f, ber=%e, bler=%e%s' %
                  (res['num_parity'], res['latency'], res['throughput'], res['bit_error_rate'], res['block_error_rate'], crc_str))
        else:
            print('latency=%.1f, throughput=%.2f, ber=%e, bler=%e%s' %
                  (res['latency'], res['throughput'], res['bit_error_rate'], res['block_error_rate'], crc_str))
        if 'iteration_lines' in res:
            for line in res['iteration_lines']:
                print(line)
    return res

#***********************************************************************
# do_latency_mode()
def do_latency_mode():
    results = []

    params = {'exe'               : args.exe,
              'exe_dir'           : args.exe_dir,
              'BG'                : args.bg,
              'input_file'        : os.path.join(args.data_dir, args.input_file) if args.input_file else '',
              'num_parity'        : 0,
              'num_iter'          : args.num_iter,
              'num_runs'          : args.num_runs,
              'use_fp16'          : args.use_fp16,
              'num_words'         : args.num_words,
              'Z'                 : args.lifting_size if not args.input_file else 0,
              'snr'               : args.snr if not args.input_file else '',
              'algo'              : args.algo,
              'puncture'          : args.puncture,
              'use_tb'            : args.transport_block,
              'block_size'        : args.block_size,
              'mod_bits'          : args.mod_bits,
              'code_rate'         : args.code_rate,
              'normalization'     : args.normalization,
              'min_block_errors'  : args.min_block_errors,
              'choose_throughput' : args.throughput,
              'tb_spread'         : args.tb_spread}

    if args.normalization:
        params['normalization'] = args.normalization
    max_num_parity_value = max_num_parity[params['BG']] if args.max_num_parity < 0 else args.max_num_parity
    # Iterate over code rates (i.e. number of parity nodes)
    for mb in range(args.min_num_parity, max_num_parity_value + 1):
        params['num_parity'] = mb
        results.append(run_config(params, True))

    for r in results:
        print('%2d %7.1f %e %e %.2f' % (r['num_parity'], r['latency'], r['bit_error_rate'], r['block_error_rate'], r['throughput']))
    if args.output_file:
        with open(args.output_file, 'w') as f:
            for r in results:
                f.write('%2d %7.1f %e %e %.2f\n' % (r['num_parity'], r['latency'], r['bit_error_rate'], r['block_error_rate'], r['throughput']))
    if args.compare_file:
        max_pct_change = 0
        prev_res = get_comparison_results(args.compare_file)
        print('# NUM_PARITY LATENCY PREV_LATENCY LATENCY_CHANGE PCT_CHANGE')
        for r in results:
            r['prev_latency'] = prev_res[r['num_parity']][0]
            r['latency_change'] = r['latency'] - r['prev_latency']
            r['latency_change_pct'] = 100.0 * r['latency_change'] / r['prev_latency']
            max_pct_change = max(max_pct_change, abs(r['latency_change_pct']))
        for r in results:
            print('%12d %7.1f %12.1f %14.1f %+10.1f' % (r['num_parity'], r['latency'], r['prev_latency'], r['latency_change'], r['latency_change_pct']))
        print('Maximum percent latency change: %.1f %%' % max_pct_change)

#***********************************************************************
# do_ber_mode()
def do_ber_mode():
    results = []
    # Don't specify the number of words - use the number in the input file.
    # Do only 1 run - no need to average over multiple iterations for timing.
    params = {'exe'               : args.exe,
              'exe_dir'           : args.exe_dir,
              'BG'                : args.bg,
              'input_file'        : '',
              'num_iter'          : args.num_iter,
              'num_runs'          : 1,
              'use_fp16'          : args.use_fp16,
              'num_words'         : args.num_words,
              'algo'              : args.algo,
              'puncture'          : args.puncture,
              'use_tb'            : args.transport_block,
              'block_size'        : args.block_size,
              'mod_bits'          : args.mod_bits,
              'code_rate'         : args.code_rate,
              'Z'                 : args.lifting_size if not args.block_size else None,
              'num_parity'        : args.num_parity if not args.block_size else None,
              'normalization'     : args.normalization,
              'min_block_errors'  : args.min_block_errors,
              'choose_throughput' : args.throughput,
              'tb_spread'         : args.tb_spread}

    # Iterate over an SNR range that is appropriate for the given number
    # of parity check nodes. If values were provided on the command line
    # use those, otherwise we can a.) read them from the database, or
    # b.) determine them
    if args.min_snr and args.max_snr:
        snr_min = args.min_snr
        snr_max = args.max_snr
    else:
        # Load any previously stored and cached performance data
        db = ldpc_perf_database(db_name)
        # Determine parameters necessary to look up an SNR range
        args_query = args
        # print(args_query)
        if args_query.block_size:
            # This BER run may not use an integral number of parity nodes,
            # but the SNR query is currently set up for that situation.
            # (It updates an internal database.)
            Kb, K, Z, F = code_block_params(args_query.bg, args_query.block_size)
            args_query.lifting_size = Z
            args_query.num_parity = get_num_parity(args_query)
            # print(Kb, K, Z, F, args_query.lifting_size, args_query.block_size, args_query.num_parity)
        # else:
        #    num_parity = args.num_parity
        #    Z          = args.lifting_size
        # key_list = ['%d' % args.bg,
        #            '%d' % args_query.lifting_size,
        #            'snr_range',
        #            '%d' % args_num_parity]
        # snr_range = db.get_value(key_list)
        # Note: may have to search for a range
        snr_range = db.get_SNR_range(args_query);
        print('SNR range: [%f, %f]' % (snr_range[0], snr_range[1]))
        db.store()
        if not snr_range:
            raise RuntimeError('SNR range not found in database')
        snr_min = snr_range[0]
        snr_max = snr_range[1]
    num_SNR = int(round((snr_max - snr_min) / args.snr_step)) + 1
    for I_SNR in range(num_SNR):
        snr = snr_min + (I_SNR * args.snr_step)
        params['snr'] = snr
        res = run_config(params, True)
        res['snr'] = snr
        results.append(res)

    for r in results:
        print('%.2f %e %e' % (r['snr'], r['bit_error_rate'], r['block_error_rate']))
    if args.output_file:
        with open(args.output_file, 'w') as f:
            for r in results:
                f.write('%.2f %e %e\n' % (r['snr'], r['bit_error_rate'], r['block_error_rate']))


# (Debugging function for binary search implementation)
class fn:
    def __call__(self, v):
        return (4 * v) - 4

#***********************************************************************
# decode_find_log10_error_rate
# Callable object to run the LDPC decoder example program for a
# specified SNR and extract the error rates from the output.
class decode_find_log10_error_rate:
    def __init__(self, params, value_string, log10_BLER, verbose):
        self.params                  = params
        self.error_rate_target_log10 = log10_BLER
        self.value_string            = value_string
        self.verbose                 = verbose
    def __call__(self, SNR):
        self.params['snr'] = SNR
        res = run_config(self.params, self.verbose)
        # print('%.2f %e %e' % (SNR, res['bit_error_rate'], res['block_error_rate']))
        # math domain error when taking log10(0)
        error_rate = max(res[self.value_string], sys.float_info.epsilon)
        return (math.log10(error_rate) - self.error_rate_target_log10)

#***********************************************************************
# binary_search()
# Perform a binary search for the root of a callable object given by
# f. The root is assumed to lie between a and b. The value tol
# dictates how close the function evaluation must be to 0 before
# success is assumed.
def binary_search(f, a, b, tol, max_count, verbose):
    m   = (float(a) + float(b)) / 2
    f_a = f(a)
    f_b = f(b)
    f_m = f(m)
    if math.copysign(1, f_a) == math.copysign(1, f_b):
        raise RuntimeError('Monotonic binary search bounds do not contain a solution')
    if verbose:
        print('[%10.6f, %10.6f, %10.6f] --> [%10.6f, %10.6f, %10.6f]' % (a, m, b, f_a, f_m, f_b))
    count = 0
    while (abs(f_m) > tol) and (count <= max_count):
        if math.copysign(1, f_m) == math.copysign(1, f_a):
            a = m
            m = (m + b) / 2
            f_a = f_m
        else:
            b = m
            m = (m + a) / 2
            f_b = f_m
        f_m = f(m)
        if verbose:
            print('[%10.6f, %10.6f, %10.6f] --> [%10.6f, %10.6f, %10.6f]' % (a, m, b, f_a, f_m, f_b))
        count = count + 1
    # if count > max_count:
    #    raise RuntimeError('Binary search did not reach a solution in the given number of iterations')
    return m

#***********************************************************************
# get_initial_norm()
# Return an approximate normalization for the given number of parity
# nodes. Used to determine the SNR range, before optimizing for the
# "best" normalization value. (A poor initial value can result in a
# lack of convergence for the binary search.)
def get_initial_norm(bg, num_parity):
    if 1 == bg:
        m = (0.6875 - 0.8125) / (46.0 - 5.0)
    else:
        m = (0.5625 - 0.8125) / (42.0 - 5.0)
    return (m * (num_parity - 5)) + 0.8125

#***********************************************************************
# find_item_recursive()
def find_item_recursive(d, key_list):
    k = key_list[0]
    if k in d:
        if len(key_list) > 1:    # More keys?
            if isinstance(d[k], dict):
                return find_item_recursive(d[k], key_list[1:])
            else:
                return None
        else:
            return d[k] # Last key in list, return dict item
    else:
        return None

#***********************************************************************
# set_item_recursive()
def set_item_recursive(d, key_list, value):
    k = key_list[0]
    if len(key_list) > 1:    # More keys?
        if not k in d:
            d[k] = {}
        d = d[k]
        set_item_recursive(d, key_list[1:], value)
    else:
        d[k] = value # Set value for last key in list

#***********************************************************************
# find_SNR_value()
# f = fn()
# x = binary_search(f, -10, 10, 0.0001, 100, True)
# print('x = %f, f(x) = %f' % (x, f(x)))
def find_SNR_value(params, type_str, value, value_tol, min_block_errors):
    params['min_block_errors'] = min_block_errors
    res_field = 'block_error_rate' if 'BLER' == type_str else 'bit_error_rate'
    f         = decode_find_log10_error_rate(params,
                                             res_field,
                                             math.log10(value),
                                             False)
    SNR       = binary_search(f,
                              -10,
                              10,
                              math.log10(value_tol),
                              25,
                             True)
    print('Target %s: %g, SNR: %f' % (type_str, value, SNR))
    return SNR

#***********************************************************************
# find_SNR_range()
def find_SNR_range(args,
                   low_type,
                   low,
                   low_tol,
                   hi_type,
                   hi,
                   hi_tol):
    # Do only 1 run - no need to average over multiple iterations for timing.
    # Specify a nominal normalization value to detect the SNR range for
    # further analysis
    params = {'exe'               : args.exe,
              'exe_dir'           : args.exe_dir,
              'BG'                : args.bg,
              'input_file'        : '',
              'num_parity'        : args.num_parity,
              'num_iter'          : args.num_iter,
              'num_runs'          : 1,
              'use_fp16'          : args.use_fp16,
              'num_words'         : args.num_words,
              'Z'                 : args.lifting_size,
              'algo'              : args.algo,
              'puncture'          : args.puncture,
              'use_tb'            : args.transport_block,
              'normalization'     : None,
              'min_block_errors'  : 0,
              'block_size'        : None,
              'code_rate'         : None,
              'mod_bits'          : None,
              'choose_throughput' : args.throughput}
    SNR_a = find_SNR_value(params, low_type, low, low_tol, 0)
    SNR_b = find_SNR_value(params, hi_type,  hi,  hi_tol, 32)
    # Round up to the nearest 0.1 dB
    SNR_b = math.ceil(SNR_b * 10) / 10.0
    # Round down to the nearest 0.1 dB, and clamp the range
    SNR_a = math.floor(SNR_a * 10) / 10.0
    SNR_a = max(SNR_a, SNR_b - 4.0)
    return [SNR_a, SNR_b]

#***********************************************************************
# find_best_norm()
def find_best_norm(args, SNR_range):
    print('Determining best normalization for BG = %d, Z = %d, num_parity = %d, SNR = [%.1f, %.1f]' %
          (args.bg, args.lifting_size, args.num_parity, SNR_range[0], SNR_range[1]))
    # Do only 1 run - no need to average over multiple iterations for timing.
    params = {'exe'               : args.exe,
              'exe_dir'           : args.exe_dir,
              'BG'                : args.bg,
              'input_file'        : '',
              'num_parity'        : args.num_parity,
              'num_iter'          : args.num_iter,
              'num_runs'          : 1,
              'use_fp16'          : args.use_fp16,
              'num_words'         : args.num_words,
              'Z'                 : args.lifting_size,
              'algo'              : args.algo,
              'puncture'          : args.puncture,
              'use_tb'            : args.transport_block,
              'normalization'     : 0.0,
              'min_block_errors'  : 128,
              'choose_throughput' : args.throughput}
    def find_best_norm_inner(params, SNR_range, norm_min, norm_delta, norm_count):
        delta_SNR  = 0.1
        num_SNR    = int(round((SNR_range[1] - SNR_range[0]) / delta_SNR)) + 1
        BER_sum    = [0.0 for N in range(norm_count)]
        BLER_sum   = [0.0 for N in range(norm_count)]
        for I_SNR in range(num_SNR):
            SNR = SNR_range[0] + (I_SNR * delta_SNR)
            params['snr'] = SNR
            BER           = [0.0 for N in range(norm_count)]
            BLER          = [0.0 for N in range(norm_count)]
            print('# SNR: %f' % SNR)
            for I_NORM in range(norm_count):
                NORM = norm_min + (I_NORM * norm_delta)
                params['normalization'] = NORM
                res = run_config(params, False)
                BER[I_NORM]  = res['bit_error_rate']
                BLER[I_NORM] = res['block_error_rate']
                # BER_sum[I_NORM]  = BER_sum[I_NORM]  + res['bit_error_rate']
                # BLER_sum[I_NORM] = BLER_sum[I_NORM] + res['block_error_rate']
                print('%.2f %e %e' % (NORM, res['bit_error_rate'], res['block_error_rate']))
            print('# SNR_normalized: %f' % SNR)
            min_BER  = min(BER)
            min_BLER = min(BLER)
            # If the BER is zero, use a 1 bit error out of 1e6 codewords
            # in its place, to avoid dividing by zero.
            if 0.0 == min_BER:
                Kb = 22 if args.bg == 1 else 10
                num_bits = 1.0e6 * Kb * args.lifting_size
                min_BER = 1.0 / num_bits
            if 0.0 == min_BLER:
                min_BLER = 1.0e-6
            for I_NORM in range(norm_count):
                BER_norm         = BER[I_NORM] / min_BER if BER[I_NORM] > 0.0 else 1.0
                BER_sum[I_NORM]  = BER_sum[I_NORM]  + BER_norm
                BLER_norm        = BLER[I_NORM] / min_BLER if BLER[I_NORM] > 0.0 else 1.0
                BLER_sum[I_NORM] = BLER_sum[I_NORM] + BLER_norm
                NORM = norm_min + (I_NORM * norm_delta)
                print('%.2f %e %e' % (NORM, BER_norm, BLER_norm))
        print('# AVG_normalized:')
        for I_NORM in range(norm_count):
            print('%.2f %e %e' % (norm_min + (I_NORM * norm_delta), BER_sum[I_NORM] / num_SNR, BLER_sum[I_NORM] / num_SNR))
        # t = [(r['bit_error_rate'], r['normalization']) for r in results]
        # tmin = min(t, key=lambda x : x[0])
        min_BER_avg = min(BER_sum)
        min_BER_idx = BER_sum.index(min_BER_avg)
        min_BLER_avg = min(BLER_sum)
        min_BLER_idx = BLER_sum.index(min_BLER_avg)
        # min_BER_avg = min_BER_avg / num_SNR
        best_norm_BLER = norm_min + (min_BLER_idx * norm_delta)
        best_norm_BER  = norm_min + (min_BER_idx  * norm_delta)
        print('norm(BLER) = %f, norm(BER) = %f' % (best_norm_BLER, best_norm_BER))
        best_norm      = best_norm_BLER
        return best_norm
    # print(tmin)
    # Look at norm intervals of 0.1, ranging from 0.3 to 0.9
    print('Coarse search, [0.3, 0.9]')
    best_norm_coarse = find_best_norm_inner(params,
                                            SNR_range,
                                            0.3,
                                            0.1,
                                            7)
    # Look at norm intervals of 0.01, adjacent to the coarse norm found
    # above
    print('Fine search around %.1f' % best_norm_coarse)
    best_norm_fine   = find_best_norm_inner(params,
                                            SNR_range,
                                            best_norm_coarse - 0.09,
                                            0.01,
                                            19)
    print('Mininum average BER at norm = %.2f' % best_norm_fine)
    return best_norm_fine

#***********************************************************************
# ldpc_perf_database
class ldpc_perf_database:
    def __init__(self, fname):
        self.updated     = False
        self.db_filename = fname
        if os.path.exists(fname):
            print("Loading perf database from '%s'" % fname)
            with open(fname, 'r') as f:
                self.db_dict = json.load(f)
        else:
            print("Initializing empty perf database")
            self.db_dict = {}
    def store(self):
        if self.updated:
            print("Writing updated database to '%s'" % self.db_filename)
            # print(self.db_dict)
            with open(self.db_filename, 'w') as f:
                json.dump(self.db_dict, f, indent=4, sort_keys=True)
    #-------------------------------------------------------------------
    # get_SNR_range()
    # Returns a nominal SNR range useful for plotting results
    def get_SNR_range(self, args):
        key_list = ['%d' % args.bg,
                    '%d' % args.lifting_size,
                    'snr_range',
                    '%d' % args.num_parity]
        r = find_item_recursive(self.db_dict, key_list)
        if r:
            return r
        else:
            print('Measuring nominal SNR range for BG = %d, Z = %d, num_parity = %d' %
                  (args.bg, args.lifting_size, args.num_parity))
            r = find_SNR_range(args,
                               'BER',
                               0.25,
                               1.05,
                               'BER',
                               1.0e-7,
                               1.1)
            set_item_recursive(self.db_dict, key_list, r)
            self.updated = True
            return r
    #-------------------------------------------------------------------
    # get_value()
    def get_value(self, key_list):
        r = find_item_recursive(self.db_dict, key_list)
        return r
    #-------------------------------------------------------------------
    # get_SNR_opt_range()
    # Returns an SNR range useful for optimizing normalization values.
    # This can be different than the nominal SNR range returned by
    # get_SNR_range() to avoid averaging over SNRs that are not in the
    # intended use case domain.
    def get_SNR_opt_range(self, args):
        key_list = ['%d' % args.bg,
                    '%d' % args.lifting_size,
                    'snr_opt_range',
                    '%d' % args.num_parity]
        r = find_item_recursive(self.db_dict, key_list)
        if not r:
            print('Measuring optimization SNR range for BG = %d, Z = %d, num_parity = %d' %
                  (args.bg, args.lifting_size, args.num_parity))
            r = find_SNR_range(args,
                               'BLER',
                               0.2,
                               1.05,
                               'BER',
                               1.0e-6,
                               1.1)
            set_item_recursive(self.db_dict, key_list, r)
            self.updated = True
        return r
    #-------------------------------------------------------------------
    # get_norm()
    # Returns a normalization value
    def get_norm(self, args):
        key_list = ['%d' % args.bg,
                    '%d' % args.lifting_size,
                    'norm',
                    '%d' % args.num_parity]
        r = find_item_recursive(self.db_dict, key_list)
        if not r:
            SNR_range = self.get_SNR_opt_range(args)
            r         = find_best_norm(args, SNR_range)
            set_item_recursive(self.db_dict, key_list, r)
            self.updated = True
        return r

#***********************************************************************
# do_norm_mode()
def do_norm_mode():
    # Load any previously stored and cached performance data
    db      = ldpc_perf_database(db_name)
    try:
        norm = db.get_norm(args)
        print('best norm: %f\n' % norm)
    except:
        print('Exception:', sys.exc_info()[0])
    finally:
        # Store the database back to a file if any data has changed
        db.store()

#***********************************************************************
# do_range_mode()
def do_range_mode():
    # Load any previously stored and cached performance data
    db = ldpc_perf_database(db_name)
    try:
        SNR_range = db.get_SNR_range(args)
        print('SNR range: [%f, %f]\n' % (SNR_range[0], SNR_range[1]))
    except:
        print('Exception:', sys.exc_info()[0])
    finally:
        # Store the database back to a file if any data has changed
        db.store()

#***********************************************************************
# load_test_config()
def load_test_config():
    configs = []
    with open(args.input_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break # EOF
            if line.startswith('#'):
                continue
            line = line.strip()
            if not line:
                continue # Empty line
            fields = line.split()
            c = {}
            # File format assumed:
            # BG Z num_parity num_iter SNR max_BER max_BLER
            c['BG']         = int(fields[0])
            c['Z']          = int(fields[1])
            c['num_parity'] = int(fields[2])
            c['num_iter']   = int(fields[3])
            c['SNR']        = float(fields[4])
            c['max_BER']    = float(fields[5])
            c['max_BLER']   = float(fields[6])
            configs.append(c)
    return configs

#***********************************************************************
# do_test_mode()
def do_test_mode():
    # Don't specify the number of words - use the number in the input file.
    # Do only 1 run - no need to average over multiple iterations for timing.
    params = {'exe'               : args.exe,
              'exe_dir'           : args.exe_dir,
              'BG'                : 0,     # will populate for each config
              'input_file'        : None,
              'num_iter'          : 0,     # will populate for each config
              'num_runs'          : 1,
              'use_fp16'          : args.use_fp16,
              'num_words'         : args.num_words,
              'algo'              : args.algo,
              'puncture'          : args.puncture,
              'use_tb'            : args.transport_block,
              'block_size'        : None,
              'mod_bits'          : None,
              'code_rate'         : None,
              'Z'                 : 0,     # will populate for each config
              'num_parity'        : 0,     # will populate for each config
              'snr'               : 0,     # will populate for each config
              'normalization'     : None,
              'min_block_errors'  : args.min_block_errors,
              'choose_throughput' : args.throughput,
              'tb_spread'         : args.tb_spread}
    results = []
    configs = load_test_config()
    for c in configs:
        res_dict = c
        params['BG']         = c['BG']
        params['num_iter']   = c['num_iter']
        params['Z']          = c['Z']
        params['num_parity'] = c['num_parity']
        params['snr']        = c['SNR']
        res = run_config(params, True)
        res_dict['BER'] = res['bit_error_rate']
        res_dict['BLER'] = res['block_error_rate']
        if res.get('skipped', False):
            res_dict['SKIP_REASON'] = res['skip_reason']
        if 'crc_block_error_rate' in res:
            res_dict['CRC_BLER'] = res['crc_block_error_rate']
        if 'iteration_lines' in res:
            res_dict['ITERATION_INFO'] = '; '.join(res['iteration_lines'])
        results.append(res_dict)
    # print(results)
    report_crc = any('CRC_BLER' in r for r in results)
    report_iter = any('ITERATION_INFO' in r for r in results)
    report_skip = any('SKIP_REASON' in r for r in results)
    extra_header = ('     CRC_BLER' if report_crc else '') + ('  ITERATION_INFO' if report_iter else '') + ('  SKIP_REASON' if report_skip else '')
    print('# BG     Z num_parity num_iter      SNR      max_BER          BER     max_BLER         BLER  STATUS%s' % extra_header)
    pass_count = 0
    fail_count = 0
    skip_count = 0
    for r in results:
        if 'SKIP_REASON' in r:
            s = 'SKIP'
            skip_count = skip_count + 1
        elif (r['BER'] > r['max_BER']) or (r['BLER'] > r['max_BLER']):
            s = 'FAIL'
            fail_count = fail_count + 1
        else:
            s = 'PASS'
            pass_count = pass_count + 1
        r['STATUS'] = s
        extra = (' %12e' % r['CRC_BLER']) if report_crc and 'CRC_BLER' in r else (' %12s' % '') if report_crc else ''
        extra += ('  %s' % r['ITERATION_INFO']) if report_iter and 'ITERATION_INFO' in r else ''
        extra += ('  %s' % r['SKIP_REASON']) if report_skip and 'SKIP_REASON' in r else ''
        base = '%4i %5i %10i %8i %8.3f %9e %9e %11e %11e    %s' % \
               (r['BG'], r['Z'], r['num_parity'], r['num_iter'], r['SNR'], r['max_BER'], r['BER'], r['max_BLER'], r['BLER'], s)
        print(base + extra)
    print('%i TESTS PASSED, %i TESTS FAILED, %i TESTS SKIPPED' % (pass_count, fail_count, skip_count))
    if args.output_file:
        with open(args.output_file, 'w') as f:
            for r in results:
                extra = (' %12e' % r['CRC_BLER']) if report_crc and 'CRC_BLER' in r else (' %12s' % '') if report_crc else ''
                extra += ('  %s' % r['ITERATION_INFO']) if report_iter and 'ITERATION_INFO' in r else ''
                extra += ('  %s' % r['SKIP_REASON']) if report_skip and 'SKIP_REASON' in r else ''
                base = '%4i %5i %10i %8i %8.3f %9e %9e %11e %11e    %s' % \
                       (r['BG'], r['Z'], r['num_parity'], r['num_iter'], r['SNR'], r['max_BER'], r['BER'], r['max_BLER'], r['BLER'], r['STATUS'])
                f.write(base + extra + '\n')

    return 1 if (fail_count > 0) else 0

#***********************************************************************
# do_find_SNR_mode()
def do_find_SNR_mode():
    # Don't specify the number of words - use the number in the input file.
    # Do only 1 run - no need to average over multiple iterations for timing.
    params = {'exe'               : args.exe,
              'exe_dir'           : args.exe_dir,
              'BG'                : args.bg,
              'input_file'        : None,
              'num_iter'          : args.num_iter,
              'num_runs'          : 1,
              'use_fp16'          : args.use_fp16,
              'num_words'         : args.num_words,
              'algo'              : args.algo,
              'puncture'          : args.puncture,
              'use_tb'            : args.transport_block,
              'block_size'        : None,
              'mod_bits'          : None,
              'code_rate'         : None,
              'Z'                 : args.lifting_size,
              'num_parity'        : 0,  # Will be populated in loop
              'snr'               : 0,
              'normalization'     : None,
              'min_block_errors'  : args.min_block_errors,
              'choose_throughput' : args.throughput,
              'tb_spread'         : args.tb_spread}
    snr_results = []
    # Tolerances may need to be arguments...
    if args.BER:
        if args.BER <= 0:
            raise RuntimeError('BER must be greater than zero')
        value_type = 'BER'
        value = args.BER
        value_tol = 1.05
    else:
        if args.BLER <= 0:
            raise RuntimeError('BLER must be greater than zero')
        value_type = 'BLER'
        value = args.BLER
        # value_tol = 1.1
        value_tol = 1.05
    print('Performing SNR search for %s = %f, log10(value) = %f, tolerance = %f, log10(tolerance) = %f' %
          (value_type, value, math.log10(value), value_tol, math.log10(value_tol)))
    max_num_parity_value = max_num_parity[params['BG']] if args.max_num_parity < 0 else args.max_num_parity
    # Iterate over code rates (i.e. number of parity nodes)
    for mb in range(args.min_num_parity, max_num_parity_value + 1):
        params['num_parity'] = mb
        print('BG = %i, Z = %i, num_parity = %i' % (args.bg, args.lifting_size, mb))
        SNR = find_SNR_value(params, value_type, value, value_tol, args.min_block_errors)
        snr_results.append((mb, SNR))
    # Run again at the calculated SNR, to observe how "close" we are to the target value
    test_results = []
    for r in snr_results:
        params['num_parity'] = r[0]
        params['snr'] = r[1]
        res = run_config(params, True)
        test_results.append((params['BG'], params['Z'], params['num_parity'], params['snr'], res['bit_error_rate'], res['block_error_rate']))
    print('#BG    Z   p        SNR          BER        BLER')
    for r in test_results:
        print('%2i  %3i  %2i  %10.6f  %12e %12e' % (r[0], r[1], r[2], r[3], r[4], r[5]))
    if args.output_file:
        with open(args.output_file, 'w') as f:
            for r in test_results:
                f.write('%2i  %3i  %2i  %10.6f  %12e %12e\n' % (r[0], r[1], r[2], r[3], r[4], r[5]))
########################################################################
# main()
return_code = 0

if args.mode == 'latency':
    do_latency_mode()
elif args.mode == 'ber':
    do_ber_mode()
elif args.mode == 'norm':
    do_norm_mode()
elif args.mode == 'range':
    do_range_mode()
elif args.mode == 'test':
    return_code = do_test_mode()
elif args.mode == 'SNR':
    do_find_SNR_mode()
else:
    raise RuntimeError('Invalid mode: "%s"' % args.mode)

sys.exit(return_code)
